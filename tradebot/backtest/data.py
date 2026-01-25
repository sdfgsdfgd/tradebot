"""Historical data access with IBKR + on-disk cache."""
from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date, timezone
from pathlib import Path
from typing import Iterable

from ib_insync import IB, ContFuture, Index, Stock, util
from zoneinfo import ZoneInfo

from .models import Bar
from ..config import load_config
from ..signals import parse_bar_size

_FUTURE_EXCHANGES = {
    "MNQ": "CME",
    "MBT": "CME",
}
_INDEX_EXCHANGES = {
    "TICK-NYSE": "NYSE",
    "TICK-AMEX": "AMEX",
}

_ET_ZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ContractMeta:
    symbol: str
    exchange: str
    multiplier: float
    min_tick: float


class IBKRHistoricalData:
    def __init__(self) -> None:
        self._config = load_config()
        self._ib = IB()

    def connect(self) -> None:
        if self._ib.isConnected():
            return
        self._ib.connect(
            self._config.host,
            self._config.port,
            clientId=self._config.client_id + 50,
            timeout=10,
        )

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def resolve_contract(self, symbol: str, exchange: str | None) -> tuple[object, ContractMeta]:
        if exchange is None:
            exchange = _FUTURE_EXCHANGES.get(symbol) or _INDEX_EXCHANGES.get(symbol) or "SMART"
        if symbol in _INDEX_EXCHANGES:
            contract = Index(symbol=symbol, exchange=exchange, currency="USD")
        elif exchange != "SMART" and symbol in _FUTURE_EXCHANGES:
            contract = ContFuture(symbol=symbol, exchange=exchange, currency="USD")
        else:
            contract = Stock(symbol=symbol, exchange="SMART", currency="USD")
        self.connect()
        qualified = self._ib.qualifyContracts(contract)
        resolved = qualified[0] if qualified else contract
        multiplier = _parse_float(getattr(resolved, "multiplier", None)) or 1.0
        min_tick = _parse_float(getattr(resolved, "minTick", None)) or 0.01
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=min_tick)
        return resolved, meta

    def load_or_fetch_bars(
        self,
        symbol: str,
        exchange: str | None,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir: Path,
    ) -> list[Bar]:
        cache_path = _cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = _read_cache(cache_path)
            if cached:
                return _normalize_bars(cached, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
        contract, _ = self.resolve_contract(symbol, exchange)
        bars = self._fetch_bars(contract, start, end, bar_size, use_rth)
        normalized = _normalize_bars(bars, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
        # Cache raw IBKR timestamps (bar-start for intraday, midnight for daily). Normalization is
        # applied on read so cached files remain stable across timestamp policy changes.
        _write_cache(cache_path, bars)
        return normalized

    def load_cached_bars(
        self,
        symbol: str,
        exchange: str | None,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir: Path,
    ) -> list[Bar]:
        cache_path = _cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        if cache_path.exists():
            return _normalize_bars(_read_cache(cache_path), symbol=symbol, bar_size=bar_size, use_rth=use_rth)

        covering = _find_covering_cache_path(
            cache_dir=cache_dir,
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if covering is None:
            raise FileNotFoundError(f"No cached bars found at {cache_path}")
        sliced = [bar for bar in _read_cache(covering) if start <= bar.ts <= end]
        return _normalize_bars(sliced, symbol=symbol, bar_size=bar_size, use_rth=use_rth)

    def _fetch_bars(
        self,
        contract: object,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> list[Bar]:
        self.connect()
        duration = _duration_for_bar_size(bar_size)
        span_days = max(int((end - start).total_seconds() // 86400), 0)
        wants_progress = span_days >= 14 or "min" in str(bar_size).lower()
        contract_sym = str(getattr(contract, "symbol", "") or getattr(contract, "localSymbol", "") or "").strip() or "?"
        if wants_progress:
            tag = "RTH" if use_rth else "FULL"
            print(
                f"[IBKR] fetching {contract_sym} {bar_size} {tag} "
                f"{start.date().isoformat()}→{end.date().isoformat()} (chunk={duration})",
                flush=True,
            )
        timeout_sec = 60.0
        raw_timeout = os.environ.get("TRADEBOT_IBKR_HIST_TIMEOUT_SEC")
        if raw_timeout:
            try:
                timeout_sec = float(raw_timeout)
            except (TypeError, ValueError):
                timeout_sec = 60.0
        timeout_sec = max(1.0, timeout_sec)

        # IBKR does not allow setting endDateTime for continuous futures ("CONTFUT").
        # For those, request a single window and slice locally.
        if getattr(contract, "secType", None) == "CONTFUT" or isinstance(contract, ContFuture):
            # `endDateTime=""` anchors the request to "now", not to our requested `end`.
            # So we must request enough history to cover `start`, then slice to `[start, end]`.
            now_utc = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            days = max(int((now_utc - start).total_seconds() // 86400), 0)
            if days <= 7:
                duration = "1 W"
            elif days <= 14:
                duration = "2 W"
            elif days <= 31:
                duration = "1 M"
            elif days <= 93:
                duration = "3 M"
            elif days <= 186:
                duration = "6 M"
            elif days <= 366:
                duration = "1 Y"
            else:
                duration = "2 Y"
            chunk = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
                timeout=timeout_sec,
            )
            if not chunk:
                return []
            bars = [_convert_bar(bar) for bar in chunk]
            return [bar for bar in bars if start <= bar.ts <= end]
        cursor = end
        collected: list[Bar] = []
        req_idx = 0
        while cursor >= start:
            req_idx += 1
            if wants_progress:
                print(
                    f"[IBKR] reqHistoricalData #{req_idx} end={cursor.date().isoformat()} dur={duration} bar={bar_size}",
                    flush=True,
                )
            chunk = self._ib.reqHistoricalData(
                contract,
                endDateTime=cursor,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
                timeout=timeout_sec,
            )
            if not chunk:
                break
            bars = [_convert_bar(bar) for bar in chunk]
            collected = bars + collected
            earliest = bars[0].ts
            cursor = earliest - timedelta(seconds=1)
            if wants_progress:
                print(
                    f"[IBKR]  ↳ got {len(bars)} bars; earliest={earliest.date().isoformat()}",
                    flush=True,
                )
            if earliest <= start:
                break
        return [bar for bar in collected if start <= bar.ts <= end]


def _duration_for_bar_size(bar_size: str) -> str:
    if bar_size.lower().startswith("1 hour"):
        return "1 M"
    if bar_size.lower().startswith("4 hour"):
        return "1 M"
    if bar_size.lower().startswith("30 mins"):
        return "1 M"
    if bar_size.lower().startswith("15 mins"):
        return "1 M"
    if bar_size.lower().startswith("1 day"):
        return "1 Y"
    return "1 M"


def _cache_path(cache_dir: Path, symbol: str, start: datetime, end: datetime, bar: str, use_rth: bool) -> Path:
    tag = "rth" if use_rth else "full"
    safe_bar = bar.replace(" ", "")
    return cache_dir / symbol / f"{symbol}_{start.date()}_{end.date()}_{safe_bar}_{tag}.csv"


def _read_cache(path: Path) -> list[Bar]:
    return list(_read_cache_cached(str(path)))


@lru_cache(maxsize=32)
def _read_cache_cached(path: str) -> tuple[Bar, ...]:
    rows: list[Bar] = []
    with Path(path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                Bar(
                    ts=datetime.fromisoformat(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return tuple(rows)


def _write_cache(path: Path, bars: Iterable[Bar]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "ts": bar.ts.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )


def _convert_bar(bar) -> Bar:
    dt = bar.date
    if isinstance(dt, str):
        dt = util.parseIBDatetime(dt)
    if isinstance(dt, date) and not isinstance(dt, datetime):
        # IBKR returns daily bar timestamps as a date only (no time). Keep them at midnight here;
        # `_normalize_bars` will align them to a safe session-close timestamp (UTC-naive).
        dt = datetime.combine(dt, time(0, 0))
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.replace(tzinfo=None)
    return Bar(
        ts=dt,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume or 0),
    )


def _parse_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _daily_close_time_et(*, symbol: str, use_rth: bool) -> time:
    sym = str(symbol or "").strip().upper()
    # Our entire codebase treats tz-naive timestamps as UTC (see `_ts_to_et` in decision_core).
    # So for daily bars, we:
    #   1) interpret the bar's date as an ET trade date,
    #   2) pick a session-close time in ET,
    #   3) convert that close timestamp to UTC (still stored tz-naive).
    #
    # This prevents multi-timeframe gates from "seeing" the day's OHLC before the session is complete.
    if sym in _FUTURE_EXCHANGES:
        # CME index futures: Globex daily break is 17:00 ET. For RTH-only daily bars, align to 16:00 ET.
        return time(16, 0) if use_rth else time(17, 0)
    # Equities / indices: RTH close is 16:00 ET; extended session ends 20:00 ET.
    return time(16, 0) if use_rth else time(20, 0)


def _normalize_bars(bars: list[Bar], *, symbol: str, bar_size: str, use_rth: bool) -> list[Bar]:
    """Normalize bar timestamps so MTF alignment doesn't leak future information."""
    if not bars:
        return bars
    label = str(bar_size or "").strip().lower()
    if label.startswith("1 day"):
        close_et = _daily_close_time_et(symbol=symbol, use_rth=use_rth)
        out: list[Bar] = []
        for bar in bars:
            ts_et = datetime.combine(bar.ts.date(), close_et, tzinfo=_ET_ZONE)
            ts_utc = ts_et.astimezone(timezone.utc).replace(tzinfo=None)
            out.append(
                Bar(
                    ts=ts_utc,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            )
        return out

    bar_def = parse_bar_size(str(bar_size))
    if bar_def is None:
        return bars
    dur = bar_def.duration
    if dur <= timedelta(0):
        return bars
    # IBKR intraday bars are timestamped at the bar *start*. Shift them forward so ts represents
    # the bar *close*, preventing within-bar and multi-timeframe lookahead leakage.
    return [
        Bar(
            ts=bar.ts + dur,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        for bar in bars
    ]


def _find_covering_cache_path(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> Path | None:
    folder = cache_dir / symbol
    if not folder.exists():
        return None

    tag = "rth" if use_rth else "full"
    safe_bar = str(bar_size).replace(" ", "")
    prefix = f"{symbol}_"
    suffix = f"_{safe_bar}_{tag}.csv"
    # Example: MNQ_2025-01-08_2026-01-08_1hour_full.csv
    pattern = re.compile(
        rf"^{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})_{re.escape(safe_bar)}_{tag}\.csv$"
    )

    start_d = start.date()
    end_d = end.date()
    candidates: list[tuple[int, Path]] = []
    for path in folder.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        m = pattern.match(name)
        if not m:
            continue
        try:
            file_start = date.fromisoformat(m.group(1))
            file_end = date.fromisoformat(m.group(2))
        except ValueError:
            continue
        if file_start <= start_d and file_end >= end_d:
            span_days = (file_end - file_start).days
            candidates.append((span_days, path))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]
