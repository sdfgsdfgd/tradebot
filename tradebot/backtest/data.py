"""IBKR historical-data acquisition and canonical bar normalization."""
from __future__ import annotations

import asyncio
import os
from threading import Lock
from time import monotonic
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
from pathlib import Path

from ib_insync import IB, ContFuture, Index, Stock, util
from zoneinfo import ZoneInfo

from ..contract_identity import (
    future_exchange_for_symbol,
    index_exchange_for_symbol,
    is_future_symbol,
)
from ..engines.market import is_early_close_day
from .config import ConfigBundle
from .models import Bar
from ..chart_data.history import (
    _canonical_bar_token,
    _intraday_resample_preferred,
    cache_path,
    load_history_window,
    normalize_bars_to_close as _normalize_bars,
    read_cache,
    write_cache,
    write_history_chunk,
)
from ..chart_data.series import BarSeries, BarSeriesMeta, bars_list
from ..config import load_config
from ..signals import parse_bar_size
from ..time_utils import (
    ET_ZONE as _ET_ZONE,
    NaiveTsMode,
    UTC as _UTC,
    to_et as _to_et_shared,
    to_utc_naive as _to_utc_naive_shared,
)

_OVERNIGHT_START_ET = time(20, 0)
_PREMARKET_START_ET = time(4, 0)

_IBKR_DURATION_BY_BAR_TOKEN = {
    "1min": "1 W",
    "15mins": "1 M",
    "30mins": "1 M",
    "1hour": "1 M",
    "4hours": "1 M",
    "1day": "1 Y",
}

_HISTORICAL_LOCK_GUARD = Lock()
_HISTORICAL_CONTRACT_LOCKS: dict[tuple[object, ...], object] = {}


def _historical_contract_lock(contract: object) -> object:
    """Serialize one IBKR contract while allowing independent tapes in parallel."""
    key = (
        getattr(contract, "conId", None),
        getattr(contract, "secType", None),
        getattr(contract, "symbol", None),
        getattr(contract, "exchange", None),
    )
    with _HISTORICAL_LOCK_GUARD:
        return _HISTORICAL_CONTRACT_LOCKS.setdefault(key, Lock())


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


def load_backtest_series(
    *,
    data: "IBKRHistoricalData",
    cfg: ConfigBundle,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> BarSeries[Bar]:
    """Load the canonical cached or live series selected by backtest policy."""
    loader = data.load_cached_bar_series if bool(cfg.backtest.offline) else data.load_or_fetch_bar_series
    return loader(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        cache_dir=cfg.backtest.cache_dir,
    )


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
        symbol = str(symbol).strip().upper()
        future_exchange = future_exchange_for_symbol(symbol)
        index_exchange = index_exchange_for_symbol(symbol)
        if exchange is None:
            exchange = future_exchange or index_exchange or "SMART"
        if index_exchange is not None:
            contract = Index(symbol=symbol, exchange=exchange, currency="USD")
        elif exchange != "SMART" and is_future_symbol(symbol):
            contract = ContFuture(symbol=symbol, exchange=exchange, currency="USD")
        else:
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

        history = load_history_window(
            cache_dir=cache_dir,
            symbol=symbol,
            start_et=start,
            end_et=end,
            bar_size=bar_size,
            use_rth=use_rth,
            naive_ts_mode=NaiveTsMode.UTC,
        )
        if history.bars and not history.missing_ranges:
            source_path = history.source_paths[0] if len(history.source_paths) == 1 else None
            source = (
                "cache"
                if source_path == cache_file
                else "cache-covering"
                if source_path is not None
                else "cache-stitched"
            )
            normalized = _normalize_bars(history.bars, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
            return BarSeries(
                bars=tuple(normalized),
                meta=_series_meta(
                    symbol=symbol,
                    bar_size=bar_size,
                    use_rth=use_rth,
                    source=source,
                    source_path=source_path,
                    start=start,
                    end=end,
                ),
            )

        if history.bars:
            missing_ranges = list(history.missing_ranges)
            if debug_stitch:
                session_tag = "RTH" if bool(use_rth) else "FULL"
                print(
                    f"[CACHE_STITCH] {symbol} {bar_size} {session_tag} "
                    f"{start.date().isoformat()}→{end.date().isoformat()} "
                    f"cache_rows={len(history.bars)} "
                    f"fetched_gap_ranges={_fmt_ranges(missing_ranges)}",
                    flush=True,
                )
            for gap_start, gap_end in missing_ranges:
                fetch_start = max(start, datetime.combine(gap_start, time(0, 0)))
                fetch_end = min(end, datetime.combine(gap_end, time(23, 59, 59)))
                fetched = _fetch_window(fetch_start, fetch_end)
                write_history_chunk(
                    cache_dir=cache_dir,
                    symbol=symbol,
                    start_date=gap_start,
                    end_date=gap_end,
                    bar_size=bar_size,
                    use_rth=use_rth,
                    bars=fetched,
                )

            healed = load_history_window(
                cache_dir=cache_dir,
                symbol=symbol,
                start_et=start,
                end_et=end,
                bar_size=bar_size,
                use_rth=use_rth,
                naive_ts_mode=NaiveTsMode.UTC,
            )
            if healed.bars and not healed.missing_ranges:
                normalized = _normalize_bars(healed.bars, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
                return BarSeries(
                    bars=tuple(normalized),
                    meta=_series_meta(
                        symbol=symbol,
                        bar_size=bar_size,
                        use_rth=use_rth,
                        source="cache+ibkr",
                        source_path=None,
                        start=start,
                        end=end,
                    ),
                )
            missing = ", ".join(_fmt_ranges(list(healed.missing_ranges)))
            raise RuntimeError(f"Incomplete history after sparse fetch: {missing or 'no bars'}")

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

        def _try_resampled() -> BarSeries[Bar] | None:
            if not _intraday_resample_preferred(bar_size):
                return None
            try:
                from .cache_ops import resample_cached_window

                rs_out = resample_cached_window(
                    data=self,
                    cache_dir=cache_dir,
                    symbol=symbol,
                    exchange=exchange,
                    start=start,
                    end=end,
                    dst_bar_size=bar_size,
                    use_rth=use_rth,
                    src_bar_size=None,
                )
                if not (rs_out.ok and rs_out.dst_path.exists()):
                    return None
                normalized = _normalize_bars(read_cache(cache_file), symbol=symbol, bar_size=bar_size, use_rth=use_rth)
                return BarSeries(
                    bars=tuple(normalized),
                    meta=_series_meta(
                        symbol=symbol,
                        bar_size=bar_size,
                        use_rth=use_rth,
                        source="cache-resampled",
                        source_path=cache_file,
                        start=start,
                        end=end,
                    ),
                )
            except Exception:
                return None

        resampled = _try_resampled()
        if resampled is not None:
            return resampled

        history = load_history_window(
            cache_dir=cache_dir,
            symbol=symbol,
            start_et=start,
            end_et=end,
            bar_size=bar_size,
            use_rth=use_rth,
            naive_ts_mode=NaiveTsMode.UTC,
        )
        if not history.bars:
            raise FileNotFoundError(f"No cached bars found at {cache_file}")
        if history.missing_ranges:
            resampled = _try_resampled()
            if resampled is not None:
                return resampled
            missing = []
            for s, e in history.missing_ranges:
                if s == e:
                    missing.append(s.isoformat())
                else:
                    missing.append(f"{s.isoformat()}..{e.isoformat()}")
            raise FileNotFoundError(
                f"No cached bars found at {cache_file}; missing date ranges: {', '.join(missing)}"
            )

        source_path = history.source_paths[0] if len(history.source_paths) == 1 else None
        source = (
            "cache"
            if source_path == cache_file
            else "cache-covering"
            if source_path is not None
            else "cache-stitched"
        )
        normalized = _normalize_bars(history.bars, symbol=symbol, bar_size=bar_size, use_rth=use_rth)
        return BarSeries(
            bars=tuple(normalized),
            meta=_series_meta(
                symbol=symbol,
                bar_size=bar_size,
                use_rth=use_rth,
                source=source,
                source_path=source_path,
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

    def _historical_head_timestamp(
        self,
        contract: object,
        *,
        use_rth: bool,
        timeout_sec: float,
    ) -> datetime | None:
        """Return broker-declared availability without leaving a live request."""
        client = getattr(self._ib, "client", None)
        wrapper = getattr(self._ib, "wrapper", None)
        runner = getattr(self._ib, "_run", None)
        if client is None or wrapper is None or runner is None:
            return None
        req_id = client.getReqId()
        future = wrapper.startReq(req_id, contract)
        client.reqHeadTimeStamp(req_id, contract, "TRADES", bool(use_rth), 2)
        try:
            value = runner(
                asyncio.wait_for(
                    future,
                    timeout=max(1.0, min(float(timeout_sec), 15.0)),
                )
            )
            return _as_utc_aware(value) if isinstance(value, datetime) else None
        except Exception:
            return None
        finally:
            try:
                client.cancelHeadTimeStamp(req_id)
            except Exception:
                pass

    def _request_historical_chunk(
        self,
        contract: object,
        *,
        end: datetime | str,
        duration: str,
        bar_size: str,
        use_rth: bool,
        timeout_sec: float,
    ) -> tuple[list[object], str]:
        """Fetch one chunk without ever treating transport ambiguity as absent data."""
        ladder = ("2 Y", "1 Y", "6 M", "3 M", "1 M", "2 W", "1 W", "3 D", "1 D")
        if duration in ladder:
            start = ladder.index(duration)
            attempts = list(ladder[start : start + 3])
            attempts.extend([attempts[-1]] * (3 - len(attempts)))
        else:
            attempts = [duration] * 3
        errors: list[tuple[int, str]] = []

        def _capture_error(
            _req_id: int,
            code: int,
            message: str,
            _contract: object,
        ) -> None:
            if int(code) not in {2104, 2106, 2107, 2108, 2158}:
                errors.append((int(code), str(message or "").strip()))

        error_event = getattr(self._ib, "errorEvent", None)
        lock = _historical_contract_lock(contract)
        with lock:
            if error_event is not None:
                error_event += _capture_error
            try:
                for attempt, request_duration in enumerate(attempts, start=1):
                    before = len(errors)
                    started = monotonic()
                    try:
                        chunk = self._ib.reqHistoricalData(
                            contract,
                            endDateTime=end,
                            durationStr=request_duration,
                            barSizeSetting=bar_size,
                            whatToShow="TRADES",
                            useRTH=1 if use_rth else 0,
                            formatDate=1,
                            keepUpToDate=False,
                            timeout=(
                                float(timeout_sec)
                                if attempt == 1
                                else max(15.0, float(timeout_sec) * (1.0 - 0.25 * (attempt - 1)))
                            ),
                        )
                    except Exception as exc:
                        chunk = []
                        errors.append((-1, f"{type(exc).__name__}: {exc}"))
                    elapsed = monotonic() - started
                    attempt_errors = errors[before:]
                    error_text = " | ".join(text for _, text in attempt_errors).lower()
                    error_codes = {code for code, _ in attempt_errors}

                    if chunk:
                        return list(chunk), request_duration
                    if error_codes & {200, 321, 354, 10089, 10167} or any(
                        token in error_text
                        for token in ("permission", "subscription", "security definition")
                    ):
                        raise RuntimeError(
                            "historical_request_rejected: "
                            + (" | ".join(text for _, text in attempt_errors) or "broker rejection")
                        )
                    if 166 in error_codes:
                        raise RuntimeError(
                            "historical_unavailable: "
                            + (" | ".join(text for _, text in attempt_errors) or "expired contract")
                        )
                    if attempt < len(attempts):
                        delay = (1.0, 3.0)[attempt - 1]
                        con_id = int(getattr(contract, "conId", 0) or 0)
                        delay += ((con_id + attempt) % 4) * 0.1
                        print(
                            f"[IBKR]  ↳ empty/failed {request_duration} response "
                            f"after {elapsed:.1f}s; retry {attempt + 1}/{len(attempts)} "
                            f"in {delay:.1f}s",
                            flush=True,
                        )
                        self._ib.sleep(delay)

                head = self._historical_head_timestamp(
                    contract,
                    use_rth=use_rth,
                    timeout_sec=timeout_sec,
                )
                end_utc = _as_utc_aware(end) if isinstance(end, datetime) else None
                evidence = " | ".join(
                    f"{code}:{message}" for code, message in errors[-6:]
                )
                if end_utc is not None and head is not None and end_utc < head:
                    raise RuntimeError(
                        "historical_unavailable_before_head: "
                        f"requested_end={end_utc.isoformat()} earliest={head.isoformat()}"
                    )
                if any(
                    code == 165 or "no data" in message.lower()
                    for code, message in errors
                ):
                    raise RuntimeError(
                        "historical_unavailable_reported_by_ibkr: "
                        f"head={head.isoformat() if head else 'unknown'}"
                        + (f" errors={evidence}" if evidence else "")
                    )
                raise RuntimeError(
                    "historical_fetch_exhausted: "
                    f"head={head.isoformat() if head else 'unknown'}"
                    + (f" errors={evidence}" if evidence else " no broker error evidence")
                )
            finally:
                if error_event is not None:
                    error_event -= _capture_error

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
        span_seconds = max(0.0, (end - start).total_seconds())
        span_days_ceil = max(1, int((span_seconds + 86_399) // 86_400))
        range_duration = (
            "1 D"
            if span_days_ceil <= 1
            else "1 W"
            if span_days_ceil <= 7
            else "2 W"
            if span_days_ceil <= 14
            else "1 M"
        )
        if _duration_to_timedelta(range_duration) < _duration_to_timedelta(duration):
            duration = range_duration
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
            chunk, _ = self._request_historical_chunk(
                contract,
                end="",
                duration=duration,
                bar_size=bar_size,
                use_rth=use_rth,
                timeout_sec=timeout_sec,
            )
            bars = [_convert_bar(bar) for bar in chunk]
            return _filter_requested_session(
                [bar for bar in bars if start <= bar.ts <= end],
                contract=contract,
                bar_size=bar_size,
                use_rth=use_rth,
            )
        cursor = end_utc_aware
        collected: list[Bar] = []
        req_idx = 0
        while cursor >= start_utc_aware:
            req_idx += 1
            request_end = cursor
            if wants_progress:
                print(
                    f"[IBKR] reqHistoricalData #{req_idx} end={cursor.date().isoformat()} dur={duration} bar={bar_size}",
                    flush=True,
                )
            chunk, duration = self._request_historical_chunk(
                contract,
                end=cursor,
                duration=duration,
                bar_size=bar_size,
                use_rth=use_rth,
                timeout_sec=timeout_sec,
            )
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
            if (
                earliest_utc_aware <= start_utc_aware
                or request_end - _duration_to_timedelta(duration) <= start_utc_aware
            ):
                break
        return _filter_requested_session(
            [bar for bar in collected if start <= bar.ts <= end],
            contract=contract,
            bar_size=bar_size,
            use_rth=use_rth,
        )


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


def _filter_requested_session(
    bars: list[Bar],
    *,
    contract: object,
    bar_size: str,
    use_rth: bool,
) -> list[Bar]:
    """Normalize IBKR's extended index stream to the requested US RTH window."""
    parsed = parse_bar_size(str(bar_size))
    if (
        not use_rth
        or str(getattr(contract, "secType", "") or "").upper() != "IND"
        or parsed is None
        or parsed.duration >= timedelta(days=1)
    ):
        return bars
    out: list[Bar] = []
    for bar in bars:
        stamp_et = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsMode.UTC)
        session_end = time(13, 0) if is_early_close_day(stamp_et.date()) else time(16, 0)
        if time(9, 30) <= stamp_et.time() < session_end:
            out.append(bar)
    return out




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


def load_bars(
    data: IBKRHistoricalData,
    *,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
) -> list[Bar]:
    """Load one canonical historical tape through the selected data policy."""
    loader = data.load_cached_bar_series if offline else data.load_or_fetch_bar_series
    return bars_list(
        loader(
            symbol=symbol,
            exchange=exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
    )
