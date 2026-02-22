"""Historical data access with IBKR + on-disk cache."""
from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
from pathlib import Path
from typing import Iterable

from ib_insync import IB, ContFuture, Index, Stock, util
from zoneinfo import ZoneInfo

from .models import Bar
from ..series import BarSeries, BarSeriesMeta
from ..config import load_config
from ..signals import parse_bar_size
from ..time_utils import (
    ET_ZONE as _ET_ZONE,
    NaiveTsMode,
    UTC as _UTC,
    to_et as _to_et_shared,
    to_utc_naive as _to_utc_naive_shared,
)

_FUTURE_EXCHANGES = {
    "MNQ": "CME",
    "MBT": "CME",
}
_INDEX_EXCHANGES = {
    "TICK-NYSE": "NYSE",
    "TICK-AMEX": "AMEX",
}

_OVERNIGHT_START_ET = time(20, 0)
_PREMARKET_START_ET = time(4, 0)

_APPROVED_CACHE_BAR_TOKENS = frozenset(
    {
        "1min",
        "2min",
        "2mins",
        "5min",
        "5mins",
        "10min",
        "10mins",
        "15min",
        "15mins",
        "30min",
        "30mins",
        "1hour",
        "4hour",
        "4hours",
        "1day",
    }
)
_CACHE_NAME_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9.\-]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?P<bar>[a-z0-9]+)_(?P<tag>rth|full24|full)\.csv$"
)
_IBKR_DURATION_BY_BAR_TOKEN = {
    "1min": "1 W",
    "15mins": "1 M",
    "30mins": "1 M",
    "1hour": "1 M",
    "4hours": "1 M",
    "1day": "1 Y",
}


@dataclass(frozen=True)
class CacheFileMeta:
    symbol: str
    start_date: date
    end_date: date
    bar_token: str
    tag: str


def _canonical_bar_token(bar_size: str) -> str:
    parsed = parse_bar_size(str(bar_size))
    if parsed is not None:
        return parsed.label.replace(" ", "")
    return str(bar_size or "").strip().replace(" ", "").lower()


def _bar_token_variants(bar_size: str) -> tuple[str, ...]:
    canonical = _canonical_bar_token(bar_size)
    raw = str(bar_size or "").strip().replace(" ", "").lower()
    out: list[str] = []
    for token in (canonical, raw):
        if token and token not in out:
            out.append(token)
    return tuple(out)


def parse_cache_filename(path: Path) -> CacheFileMeta | None:
    m = _CACHE_NAME_RE.match(path.name)
    if not m:
        return None
    bar_token = str(m.group("bar")).lower()
    if bar_token not in _APPROVED_CACHE_BAR_TOKENS:
        return None
    try:
        start_date = date.fromisoformat(str(m.group("start")))
        end_date = date.fromisoformat(str(m.group("end")))
    except ValueError:
        return None
    if end_date < start_date:
        return None
    return CacheFileMeta(
        symbol=str(m.group("symbol")),
        start_date=start_date,
        end_date=end_date,
        bar_token=bar_token,
        tag=str(m.group("tag")).lower(),
    )


def _ibkr_bar_zone() -> ZoneInfo:
    raw = str(os.environ.get("TRADEBOT_IBKR_BAR_TZ", "America/New_York") or "").strip()
    if not raw:
        return _ET_ZONE
    try:
        return ZoneInfo(raw)
    except Exception:
        return _ET_ZONE


_IBKR_BAR_ZONE = _ibkr_bar_zone()


def _as_utc_aware(ts: datetime) -> datetime:
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=_UTC)
    return ts.astimezone(_UTC)


def _series_meta(
    *,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    source: str,
    source_path: Path | None,
    start: datetime,
    end: datetime,
) -> BarSeriesMeta:
    return BarSeriesMeta(
        symbol=str(symbol),
        bar_size=str(bar_size),
        tz_mode="utc_naive",
        session_mode="rth" if bool(use_rth) else "full24",
        source=str(source),
        source_path=(str(source_path) if source_path is not None else None),
        requested_start=start,
        requested_end=end,
    )


@dataclass(frozen=True)
class ContractMeta:
    symbol: str
    exchange: str
    multiplier: float
    min_tick: float


class IBKRHistoricalData:
    def __init__(self, *, client_id_offset: int = 50) -> None:
        self._config = load_config()
        self._ib = IB()
        self._client_id_offset = int(client_id_offset)

    def connect(self) -> None:
        if self._ib.isConnected():
            return
        self._ib.connect(
            self._config.host,
            self._config.port,
            clientId=self._config.client_id + self._client_id_offset,
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
            # For stocks we usually use SMART, but explicit overrides (e.g. OVERNIGHT) must be honored.
            stock_exchange = str(exchange or "SMART").strip() or "SMART"
            contract = Stock(symbol=symbol, exchange=stock_exchange, currency="USD")
        self.connect()
        qualified = self._ib.qualifyContracts(contract)
        resolved = qualified[0] if qualified else contract
        multiplier = _parse_float(getattr(resolved, "multiplier", None)) or 1.0
        min_tick = _parse_float(getattr(resolved, "minTick", None)) or 0.01
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=min_tick)
        return resolved, meta

    def load_or_fetch_bar_series(
        self,
        symbol: str,
        exchange: str | None,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir: Path,
    ) -> BarSeries[Bar]:
        cache_file = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            cached = read_cache(cache_file)
            if cached:
                normalized = _normalize_bars(cached, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
                return BarSeries(
                    bars=tuple(normalized),
                    meta=_series_meta(
                        symbol=symbol,
                        bar_size=bar_size,
                        use_rth=use_rth,
                        source="cache",
                        source_path=cache_file,
                        start=start,
                        end=end,
                    ),
                )

        debug_stitch = str(os.environ.get("TRADEBOT_CACHE_STITCH_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}

        def _fmt_ranges(ranges: list[tuple[date, date]]) -> list[str]:
            out: list[str] = []
            for s, e in ranges:
                if s == e:
                    out.append(s.isoformat())
                else:
                    out.append(f"{s.isoformat()}..{e.isoformat()}")
            return out

        def _is_intraday() -> bool:
            bar_def = parse_bar_size(str(bar_size))
            if bar_def is None:
                return "day" not in str(bar_size).lower()
            return bar_def.duration < timedelta(days=1)

        def _is_overnight_bar(ts: datetime) -> bool:
            t = _to_et_shared(ts, naive_ts_mode=NaiveTsMode.UTC).timetz().replace(tzinfo=None)
            return (t >= _OVERNIGHT_START_ET) or (t < _PREMARKET_START_ET)

        def _merge_full24(
            *,
            smart: list[Bar],
            overnight: list[Bar],
            fetch_start: datetime,
            fetch_end: datetime,
        ) -> list[Bar]:
            by_ts: dict[datetime, Bar] = {}
            for bar in smart:
                by_ts[bar.ts] = bar
            for bar in overnight:
                # Prefer OVERNIGHT for bars that are in the overnight band; otherwise keep SMART.
                if _is_overnight_bar(bar.ts) or bar.ts not in by_ts:
                    by_ts[bar.ts] = bar
            merged = [by_ts[k] for k in sorted(by_ts.keys())]
            return [b for b in merged if fetch_start <= b.ts <= fetch_end]

        contract: object | None = None
        overnight_contract: object | None = None

        def _resolve_primary_contract() -> object:
            nonlocal contract
            if contract is None:
                contract, _ = self.resolve_contract(symbol, exchange)
            return contract

        def _resolve_overnight_contract() -> object:
            nonlocal overnight_contract
            if overnight_contract is None:
                overnight_contract, _ = self.resolve_contract(symbol, exchange="OVERNIGHT")
            return overnight_contract

        def _fetch_window(fetch_start: datetime, fetch_end: datetime) -> list[Bar]:
            primary = _resolve_primary_contract()
            if (
                not bool(use_rth)
                and str(getattr(primary, "secType", "") or "") == "STK"
                and _is_intraday()
            ):
                bars_smart = self._fetch_bars(primary, fetch_start, fetch_end, bar_size, use_rth=False)
                bars_overnight = self._fetch_bars(
                    _resolve_overnight_contract(),
                    fetch_start,
                    fetch_end,
                    bar_size,
                    use_rth=False,
                )
                return _merge_full24(
                    smart=bars_smart,
                    overnight=bars_overnight,
                    fetch_start=fetch_start,
                    fetch_end=fetch_end,
                )
            return self._fetch_bars(primary, fetch_start, fetch_end, bar_size, use_rth)

        overlap_paths = find_overlapping_cache_paths(
            cache_dir=cache_dir,
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if overlap_paths:
            stitched_by_ts: dict[datetime, Bar] = {}
            covered_ranges: list[tuple[date, date]] = []
            for overlap in overlap_paths:
                meta = parse_cache_filename(overlap)
                if meta is not None:
                    covered_ranges.append((meta.start_date, meta.end_date))
                for bar in read_cache(overlap):
                    if start <= bar.ts <= end:
                        stitched_by_ts[bar.ts] = bar

            uncovered_ranges = _uncovered_date_ranges(
                request_start=start.date(),
                request_end=end.date(),
                covered_ranges=covered_ranges,
            )
            reused_ranges: list[tuple[date, date]] = []
            cursor = start.date()
            for gap_start, gap_end in uncovered_ranges:
                if gap_start > cursor:
                    reused_ranges.append((cursor, gap_start - timedelta(days=1)))
                cursor = max(cursor, gap_end + timedelta(days=1))
                if cursor > end.date():
                    break
            if cursor <= end.date():
                reused_ranges.append((cursor, end.date()))
            if debug_stitch:
                session_tag = "RTH" if bool(use_rth) else "FULL"
                print(
                    f"[CACHE_STITCH] {symbol} {bar_size} {session_tag} "
                    f"{start.date().isoformat()}→{end.date().isoformat()} "
                    f"cache_reused_ranges={_fmt_ranges(reused_ranges)} "
                    f"fetched_gap_ranges={_fmt_ranges(uncovered_ranges)}",
                    flush=True,
                )
            for gap_start, gap_end in uncovered_ranges:
                fetch_start = datetime.combine(gap_start, time(0, 0))
                fetch_end = datetime.combine(gap_end, time(23, 59))
                for bar in _fetch_window(fetch_start, fetch_end):
                    if start <= bar.ts <= end:
                        stitched_by_ts[bar.ts] = bar

            if stitched_by_ts:
                stitched = [stitched_by_ts[k] for k in sorted(stitched_by_ts.keys())]
                normalized = _normalize_bars(stitched, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
                write_cache(cache_file, stitched)
                source = "cache-stitched" if not uncovered_ranges else "cache+ibkr"
                return BarSeries(
                    bars=tuple(normalized),
                    meta=_series_meta(
                        symbol=symbol,
                        bar_size=bar_size,
                        use_rth=use_rth,
                        source=source,
                        source_path=cache_file,
                        start=start,
                        end=end,
                    ),
                )

        if debug_stitch:
            session_tag = "RTH" if bool(use_rth) else "FULL"
            print(
                f"[CACHE_STITCH] {symbol} {bar_size} {session_tag} "
                f"{start.date().isoformat()}→{end.date().isoformat()} "
                f"cache_reused_ranges=[] "
                f"fetched_gap_ranges={_fmt_ranges([(start.date(), end.date())])}",
                flush=True,
            )
        bars = _fetch_window(start, end)
        normalized = _normalize_bars(bars, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
        # Cache canonical UTC-naive intraday timestamps (bar-start), and date-marked daily bars
        # (midnight, normalized on read). This keeps naive-ts semantics consistent across modules.
        write_cache(cache_file, bars)
        return BarSeries(
            bars=tuple(normalized),
            meta=_series_meta(
                symbol=symbol,
                bar_size=bar_size,
                use_rth=use_rth,
                source="ibkr",
                source_path=cache_file,
                start=start,
                end=end,
            ),
        )

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
        return self.load_or_fetch_bar_series(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        ).as_list()

    def load_cached_bar_series(
        self,
        symbol: str,
        exchange: str | None,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir: Path,
    ) -> BarSeries[Bar]:
        cache_file = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        covering = find_covering_cache_path(
            cache_dir=cache_dir,
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if covering is not None:
            if covering == cache_file:
                sliced = read_cache(cache_file)
                source = "cache"
            else:
                sliced = [bar for bar in read_cache(covering) if start <= bar.ts <= end]
                source = "cache-covering"
            normalized = _normalize_bars(sliced, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
            return BarSeries(
                bars=tuple(normalized),
                meta=_series_meta(
                    symbol=symbol,
                    bar_size=bar_size,
                    use_rth=use_rth,
                    source=source,
                    source_path=covering,
                    start=start,
                    end=end,
                ),
            )

        overlap_paths = find_overlapping_cache_paths(
            cache_dir=cache_dir,
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if not overlap_paths:
            raise FileNotFoundError(f"No cached bars found at {cache_file}")

        stitched_by_ts: dict[datetime, Bar] = {}
        covered_ranges: list[tuple[date, date]] = []
        for overlap in overlap_paths:
            meta = parse_cache_filename(overlap)
            if meta is not None:
                covered_ranges.append((meta.start_date, meta.end_date))
            for bar in read_cache(overlap):
                if start <= bar.ts <= end:
                    stitched_by_ts[bar.ts] = bar

        uncovered_ranges = _uncovered_date_ranges(
            request_start=start.date(),
            request_end=end.date(),
            covered_ranges=covered_ranges,
        )
        if uncovered_ranges:
            missing = []
            for s, e in uncovered_ranges:
                if s == e:
                    missing.append(s.isoformat())
                else:
                    missing.append(f"{s.isoformat()}..{e.isoformat()}")
            raise FileNotFoundError(
                f"No cached bars found at {cache_file}; missing date ranges: {', '.join(missing)}"
            )

        if not stitched_by_ts:
            raise FileNotFoundError(f"No cached bars found at {cache_file}")

        stitched = [stitched_by_ts[k] for k in sorted(stitched_by_ts.keys())]
        write_cache(cache_file, stitched)
        normalized = _normalize_bars(stitched, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
        return BarSeries(
            bars=tuple(normalized),
            meta=_series_meta(
                symbol=symbol,
                bar_size=bar_size,
                use_rth=use_rth,
                source="cache-stitched",
                source_path=cache_file,
                start=start,
                end=end,
            ),
        )

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
        return self.load_cached_bar_series(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        ).as_list()

    def _fetch_bars(
        self,
        contract: object,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> list[Bar]:
        self.connect()
        start_utc_aware = _as_utc_aware(start)
        end_utc_aware = _as_utc_aware(end)
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
            now_utc = datetime.now(tz=_UTC).replace(tzinfo=None)
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
        cursor = end_utc_aware
        collected: list[Bar] = []
        req_idx = 0
        empty_streak = 0
        is_overnight_contract = str(getattr(contract, "exchange", "") or "").strip().upper() == "OVERNIGHT"
        while cursor >= start_utc_aware:
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
                empty_streak += 1
                if is_overnight_contract and empty_streak >= 2:
                    if wants_progress:
                        print(
                            "[IBKR]  ↳ overnight returned empty repeatedly; short-circuiting this leg",
                            flush=True,
                        )
                    break
                step = _duration_to_timedelta(duration)
                if step <= timedelta(0):
                    break
                cursor = cursor - step
                if wants_progress:
                    print(
                        f"[IBKR]  ↳ empty chunk; stepping cursor back by {step}",
                        flush=True,
                    )
                continue
            empty_streak = 0
            bars = [_convert_bar(bar) for bar in chunk]
            collected = bars + collected
            earliest = bars[0].ts
            earliest_utc_aware = _as_utc_aware(earliest)
            cursor = earliest_utc_aware - timedelta(seconds=1)
            if wants_progress:
                print(
                    f"[IBKR]  ↳ got {len(bars)} bars; earliest={earliest.date().isoformat()}",
                    flush=True,
                )
            if earliest_utc_aware <= start_utc_aware:
                break
        return [bar for bar in collected if start <= bar.ts <= end]


def _duration_for_bar_size(bar_size: str) -> str:
    # 1-month pulls for 1m bars are prone to IBKR timeouts; use smaller chunks.
    token = _canonical_bar_token(bar_size)
    mapped = _IBKR_DURATION_BY_BAR_TOKEN.get(token)
    if mapped is not None:
        return mapped
    return "1 M"


def _duration_to_timedelta(duration: str) -> timedelta:
    cleaned = str(duration or "").strip().upper()
    if not cleaned:
        return timedelta(0)
    parts = cleaned.split()
    if len(parts) != 2:
        return timedelta(0)
    try:
        qty = int(parts[0])
    except (TypeError, ValueError):
        return timedelta(0)
    unit = parts[1]
    if qty <= 0:
        return timedelta(0)
    if unit == "D":
        return timedelta(days=qty)
    if unit == "W":
        return timedelta(days=7 * qty)
    if unit == "M":
        return timedelta(days=31 * qty)
    if unit == "Y":
        return timedelta(days=366 * qty)
    return timedelta(0)


def cache_path(cache_dir: Path, symbol: str, start: datetime, end: datetime, bar: str, use_rth: bool) -> Path:
    # Naming: `rth` for RTH-only. For non-RTH we prefer `full24` (explicitly meaning 24/5 for STK,
    # and "extended/full session" for everything else).
    tag = "rth" if use_rth else "full24"
    safe_bar = _canonical_bar_token(bar)
    return cache_dir / symbol / f"{symbol}_{start.date()}_{end.date()}_{safe_bar}_{tag}.csv"


def read_cache(path: Path) -> list[Bar]:
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
    needs_fix = False
    prev_ts = None
    for bar in rows:
        if prev_ts is not None and bar.ts <= prev_ts:
            needs_fix = True
            break
        prev_ts = bar.ts
    if needs_fix:
        rows.sort(key=lambda b: b.ts)
        deduped: list[Bar] = []
        last_ts = None
        for bar in rows:
            if last_ts is not None and bar.ts == last_ts:
                deduped[-1] = bar
            else:
                deduped.append(bar)
                last_ts = bar.ts
        rows = deduped
    return tuple(rows)


def write_cache(path: Path, bars: Iterable[Bar]) -> None:
    rows = list(bars)
    if not rows:
        return
    rows.sort(key=lambda b: b.ts)
    deduped: list[Bar] = []
    last_ts = None
    for bar in rows:
        if last_ts is not None and bar.ts == last_ts:
            deduped[-1] = bar
        else:
            deduped.append(bar)
            last_ts = bar.ts

    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in deduped:
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
    os.replace(tmp, path)


def _convert_bar(bar) -> Bar:
    dt = bar.date
    if isinstance(dt, str):
        dt = util.parseIBDatetime(dt)
    if isinstance(dt, date) and not isinstance(dt, datetime):
        # IBKR daily bars are date-only. Keep date markers at midnight; `_normalize_bars` maps
        # them to a safe session-close timestamp.
        dt = datetime.combine(dt, time(0, 0))
    elif isinstance(dt, datetime):
        # Canonicalize intraday bars to UTC-naive. If IBKR returns a naive datetime, treat it as
        # exchange/session-local (default ET, override via TRADEBOT_IBKR_BAR_TZ).
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=_IBKR_BAR_ZONE)
        dt = _to_utc_naive_shared(dt)
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
    # Our entire codebase treats tz-naive timestamps as UTC (see `_ts_to_et` in engine).
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
    ordered = sorted(bars, key=lambda b: b.ts)
    label = str(bar_size or "").strip().lower()
    if label.startswith("1 day"):
        close_et = _daily_close_time_et(symbol=symbol, use_rth=use_rth)
        out: list[Bar] = []
        for bar in ordered:
            ts_et = datetime.combine(bar.ts.date(), close_et, tzinfo=_ET_ZONE)
            ts_utc = _to_utc_naive_shared(ts_et)
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
        return ordered
    dur = bar_def.duration
    if dur <= timedelta(0):
        return ordered

    close_et = _daily_close_time_et(symbol=symbol, use_rth=use_rth) if use_rth else None
    out: list[Bar] = []
    for i, bar in enumerate(ordered):
        # Baseline: convert bar-start timestamps to bar-close timestamps.
        close_ts = bar.ts + dur

        # Some IBKR RTH higher-timeframe bars are session-fragmented (short first/last bars).
        # When the next bar starts earlier than `bar.ts + duration`, use that as the close.
        if i + 1 < len(ordered):
            next_ts = ordered[i + 1].ts
            if next_ts > bar.ts and next_ts < close_ts:
                close_ts = next_ts

        # Clamp RTH closes at session close to avoid pushing tail bars past the market close.
        if close_et is not None:
            start_et = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsMode.UTC)
            session_close_et = datetime.combine(start_et.date(), close_et, tzinfo=_ET_ZONE)
            session_close_utc = _to_utc_naive_shared(session_close_et)
            if close_ts > session_close_utc:
                close_ts = session_close_utc

        out.append(
            Bar(
                ts=close_ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
        )
    return out


def find_covering_cache_path(
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

    bar_tokens = set(_bar_token_variants(bar_size))
    tags = ("rth",) if use_rth else ("full24", "full")
    tag_rank = {tag: rank for rank, tag in enumerate(tags)}
    symbol_upper = str(symbol).upper()
    start_d = start.date()
    end_d = end.date()
    candidates: list[tuple[int, int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        meta = parse_cache_filename(path)
        if meta is None:
            continue
        if str(meta.symbol).upper() != symbol_upper:
            continue
        if meta.tag not in tag_rank:
            continue
        if meta.bar_token not in bar_tokens:
            continue
        if meta.start_date <= start_d and meta.end_date >= end_d:
            span_days = (meta.end_date - meta.start_date).days
            candidates.append((tag_rank[meta.tag], span_days, path))
    if candidates:
        # Prefer the broadest covering cache. Narrower window-specific caches can
        # diverge from the canonical wide-range cache (IBKR replay drift), which
        # breaks deterministic backtest parity.
        candidates.sort(key=lambda t: (t[0], -t[1]))
        return candidates[0][2]
    return None


def find_overlapping_cache_paths(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> list[Path]:
    folder = cache_dir / symbol
    if not folder.exists():
        return []

    bar_tokens = set(_bar_token_variants(bar_size))
    tags = ("rth",) if use_rth else ("full24", "full")
    tag_rank = {tag: rank for rank, tag in enumerate(tags)}
    symbol_upper = str(symbol).upper()
    start_d = start.date()
    end_d = end.date()
    candidates: list[tuple[int, date, date, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        meta = parse_cache_filename(path)
        if meta is None:
            continue
        if str(meta.symbol).upper() != symbol_upper:
            continue
        if meta.tag not in tag_rank:
            continue
        if meta.bar_token not in bar_tokens:
            continue
        if meta.start_date > end_d or meta.end_date < start_d:
            continue
        candidates.append((tag_rank[meta.tag], meta.start_date, meta.end_date, path))
    if not candidates:
        return []
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return [c[3] for c in candidates]


def cache_covers_window(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> tuple[bool, Path | None, list[tuple[date, date]]]:
    """Return whether cache files fully cover a requested window.

    Coverage is true if either:
    - one covering cache file exists, or
    - overlapping cache files can fully span the requested date range.

    Returns:
      (covers, covering_path, missing_ranges)
    where `covering_path` is set only when a single covering file exists, and
    `missing_ranges` contains uncovered date spans when coverage is incomplete.
    """
    covering = find_covering_cache_path(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if covering is not None:
        return True, covering, []

    overlap_paths = find_overlapping_cache_paths(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if not overlap_paths:
        return False, None, [(start.date(), end.date())]

    covered_ranges: list[tuple[date, date]] = []
    for overlap in overlap_paths:
        meta = parse_cache_filename(overlap)
        if meta is not None:
            covered_ranges.append((meta.start_date, meta.end_date))

    missing_ranges = _uncovered_date_ranges(
        request_start=start.date(),
        request_end=end.date(),
        covered_ranges=covered_ranges,
    )
    if missing_ranges:
        return False, None, missing_ranges
    return True, None, []


def ensure_offline_cached_window(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
    hydrate_overlap: bool = True,
) -> tuple[bool, Path, Path | None, list[tuple[date, date]], str | None]:
    """Validate offline cache availability using runtime-equivalent loader logic.

    Returns:
      (ok, expected_path, resolved_path, missing_ranges, error_text)

    Behavior:
    - Fast path checks filename coverage metadata first.
    - If coverage is overlap-only and no exact file exists, optionally hydrate by
      invoking `load_cached_bar_series`, which performs deterministic stitch and
      persists the exact cache window.
    """
    expected = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
    cache_ok, covering, missing_ranges = cache_covers_window(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    if not cache_ok:
        return False, expected, covering, missing_ranges, None

    resolved = covering if covering is not None else (expected if expected.exists() else None)
    if not bool(hydrate_overlap):
        return True, expected, resolved, [], None
    if resolved is not None:
        return True, expected, resolved, [], None

    # Coverage is metadata-complete but exact file is missing. Reuse the same
    # loader path as runtime so stitched windows are materialized consistently.
    try:
        series = data.load_cached_bar_series(
            symbol=str(symbol),
            exchange=exchange,
            start=start,
            end=end,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            cache_dir=cache_dir,
        )
    except FileNotFoundError as exc:
        cache_ok2, covering2, missing_ranges2 = cache_covers_window(
            cache_dir=cache_dir,
            symbol=str(symbol),
            start=start,
            end=end,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
        )
        missing_eff = missing_ranges2 if missing_ranges2 else missing_ranges
        resolved_eff = covering2 if cache_ok2 else None
        return False, expected, resolved_eff, missing_eff, str(exc)

    source_path = str(getattr(series.meta, "source_path", "") or "").strip()
    if source_path:
        try:
            resolved = Path(source_path)
        except Exception:
            resolved = expected if expected.exists() else None
    elif expected.exists():
        resolved = expected
    return True, expected, resolved, [], None


def _uncovered_date_ranges(
    *,
    request_start: date,
    request_end: date,
    covered_ranges: Iterable[tuple[date, date]],
) -> list[tuple[date, date]]:
    merged: list[tuple[date, date]] = []
    for raw_start, raw_end in sorted(covered_ranges):
        cur_start = max(raw_start, request_start)
        cur_end = min(raw_end, request_end)
        if cur_end < cur_start:
            continue
        if not merged:
            merged.append((cur_start, cur_end))
            continue
        prev_start, prev_end = merged[-1]
        if cur_start <= (prev_end + timedelta(days=1)):
            merged[-1] = (prev_start, max(prev_end, cur_end))
            continue
        merged.append((cur_start, cur_end))

    if not merged:
        return [(request_start, request_end)]

    out: list[tuple[date, date]] = []
    cursor = request_start
    for cov_start, cov_end in merged:
        if cov_start > cursor:
            out.append((cursor, cov_start - timedelta(days=1)))
        cursor = max(cursor, cov_end + timedelta(days=1))
        if cursor > request_end:
            break
    if cursor <= request_end:
        out.append((cursor, request_end))
    return out
