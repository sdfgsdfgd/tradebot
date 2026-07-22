"""Deterministic cached-bar resampling and fallback policy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from ...signals import parse_bar_size
from ...time_utils import NaiveTsMode, to_et as _to_et_shared
from ..cache import cache_path, ensure_offline_cached_window, read_cache, write_cache
from ..data import IBKRHistoricalData
from ..models import Bar


_EPOCH = datetime(1970, 1, 1)

@dataclass(frozen=True)
class _ResampleStats:
    kept: int
    dropped_incomplete: int


@dataclass(frozen=True)
class CacheResampleOutcome:
    ok: bool
    dst_path: Path
    src_bar_size: str | None = None
    src_path: Path | None = None
    src_rows: int = 0
    dst_rows: int = 0
    dropped_incomplete: int = 0
    error: str | None = None


def _floor_bucket_start(ts: datetime, *, bucket: timedelta) -> datetime:
    if bucket <= timedelta(0):
        return ts
    delta = ts - _EPOCH
    micros = (delta.days * 86400 * 1_000_000) + (delta.seconds * 1_000_000) + delta.microseconds
    bucket_micros = int(bucket.total_seconds() * 1_000_000)
    bucket_idx = micros // bucket_micros
    start_micros = bucket_idx * bucket_micros
    return _EPOCH + timedelta(microseconds=int(start_micros))


def _resample_intraday_ohlcv(
    bars: list[Bar],
    *,
    src_bar_size: str,
    dst_bar_size: str,
    use_rth: bool = False,
    allow_day_from_partial: bool = False,
) -> tuple[list[Bar], _ResampleStats]:
    src = parse_bar_size(src_bar_size)
    dst = parse_bar_size(dst_bar_size)
    if src is None:
        raise SystemExit(f"Invalid src bar size: {src_bar_size!r}")
    if dst is None:
        raise SystemExit(f"Invalid dst bar size: {dst_bar_size!r}")
    if src.duration <= timedelta(0) or dst.duration <= timedelta(0):
        raise SystemExit(f"Invalid bar durations: src={src.duration} dst={dst.duration}")
    if dst.duration < src.duration:
        raise SystemExit("dst_bar_size must be >= src_bar_size")
    if (dst.duration.total_seconds() % src.duration.total_seconds()) != 0:
        raise SystemExit(f"Non-integer resample ratio: {src_bar_size!r} -> {dst_bar_size!r}")

    if not bars:
        return [], _ResampleStats(kept=0, dropped_incomplete=0)

    bars = sorted(bars, key=lambda b: b.ts)

    # RTH streams do not contain 24h coverage, so 1-day bars cannot satisfy strict
    # ratio-based chunk completeness (390 != 1440). Allow deterministic day grouping
    # when explicitly requested by caller.
    if allow_day_from_partial and src.duration < timedelta(days=1) and dst.duration >= timedelta(days=1):
        by_day: dict[date, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_day[bar.ts.date()].append(bar)
        out: list[Bar] = []
        for day in sorted(by_day.keys()):
            chunk = by_day[day]
            if not chunk:
                continue
            first = chunk[0]
            last = chunk[-1]
            out.append(
                Bar(
                    ts=datetime.combine(day, time(0, 0)),
                    open=first.open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=last.close,
                    volume=sum(float(b.volume or 0.0) for b in chunk),
                )
            )
        return out, _ResampleStats(kept=len(out), dropped_incomplete=0)

    factor = max(1, int(dst.duration.total_seconds() // src.duration.total_seconds()))
    out: list[Bar] = []
    cur_bucket = None
    cur: list[Bar] = []
    dropped = 0

    def _flush(bucket_start: datetime | None, chunk: list[Bar]) -> None:
        nonlocal dropped
        if bucket_start is None or not chunk:
            return
        if len(chunk) != factor:
            dropped += 1
            return
        first = chunk[0]
        last = chunk[-1]
        out.append(
            Bar(
                ts=bucket_start,
                open=first.open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=last.close,
                volume=sum(float(b.volume or 0.0) for b in chunk),
            )
        )

    if bool(use_rth) and src.duration < timedelta(days=1) and dst.duration < timedelta(days=1):
        by_day: dict[date, list[Bar]] = defaultdict(list)
        for bar in bars:
            day = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsMode.UTC).date()
            by_day[day].append(bar)
        for day in sorted(by_day.keys()):
            session_bars = sorted(by_day[day], key=lambda b: b.ts)
            if not session_bars:
                continue
            session_start = session_bars[0].ts
            cur_bucket = None
            cur = []
            for bar in session_bars:
                delta = bar.ts - session_start
                if delta < timedelta(0):
                    continue
                offset = int(delta.total_seconds() // src.duration.total_seconds())
                bucket_idx = int(offset // factor)
                bucket_start = session_start + (dst.duration * int(bucket_idx))
                if cur_bucket is None:
                    cur_bucket = bucket_start
                if bucket_start != cur_bucket:
                    _flush(cur_bucket, cur)
                    cur_bucket = bucket_start
                    cur = []
                cur.append(bar)
            _flush(cur_bucket, cur)
    else:
        for bar in bars:
            bucket_start = _floor_bucket_start(bar.ts, bucket=dst.duration)
            if cur_bucket is None:
                cur_bucket = bucket_start
            if bucket_start != cur_bucket:
                _flush(cur_bucket, cur)
                cur_bucket = bucket_start
                cur = []
            cur.append(bar)
    _flush(cur_bucket, cur)

    return out, _ResampleStats(kept=len(out), dropped_incomplete=int(dropped))


def _cache_resample_source_candidates(dst_bar_size: str) -> tuple[str, ...]:
    dst = parse_bar_size(str(dst_bar_size))
    if dst is None:
        return tuple()
    base = ("1 min", "2 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour", "4 hours", "1 day")
    out: list[tuple[float, str]] = []
    for src_label in base:
        src = parse_bar_size(src_label)
        if src is None:
            continue
        if src.duration >= dst.duration:
            continue
        if (dst.duration.total_seconds() % src.duration.total_seconds()) != 0:
            continue
        out.append((float(src.duration.total_seconds()), src.label))
    out.sort(key=lambda row: row[0])
    return tuple(label for _seconds, label in out)


def _native_rth_epoch_resample(dst_bar_size: str, *, use_rth: bool) -> bool:
    if not bool(use_rth):
        return False
    dst = parse_bar_size(str(dst_bar_size))
    if dst is None:
        return False
    return dst.duration == timedelta(hours=4)


def resample_cached_window(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    dst_bar_size: str,
    use_rth: bool,
    src_bar_size: str | None = None,
) -> CacheResampleOutcome:
    dst_path = cache_path(cache_dir, symbol, start, end, dst_bar_size, use_rth)
    src_candidates = (
        (str(src_bar_size).strip(),)
        if str(src_bar_size or "").strip()
        else _cache_resample_source_candidates(dst_bar_size)
    )
    if not src_candidates:
        return CacheResampleOutcome(
            ok=False,
            dst_path=dst_path,
            error=f"no_resample_source_candidates for {dst_bar_size!r}",
        )

    last_err: str | None = None
    resample_use_rth = bool(use_rth) and not _native_rth_epoch_resample(dst_bar_size, use_rth=use_rth)
    for src_label in src_candidates:
        try:
            src_series = data.load_cached_bar_series(
                symbol=symbol,
                exchange=exchange,
                start=start,
                end=end,
                bar_size=str(src_label),
                use_rth=use_rth,
                cache_dir=cache_dir,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            last_err = f"source_load_error({src_label}): {exc}"
            continue

        source_path_raw = str(getattr(src_series.meta, "source_path", "") or "").strip()
        source_path = Path(source_path_raw) if source_path_raw else None
        if source_path is not None and source_path.exists():
            src_bars = [bar for bar in read_cache(source_path) if start <= bar.ts <= end]
        else:
            src_bars = [bar for bar in src_series.as_list() if start <= bar.ts <= end]
        if not src_bars:
            last_err = f"source_empty_after_slice({src_label})"
            continue

        try:
            dst_bars, stats = _resample_intraday_ohlcv(
                src_bars,
                src_bar_size=str(src_label),
                dst_bar_size=str(dst_bar_size),
                use_rth=bool(resample_use_rth),
                allow_day_from_partial=bool(use_rth),
            )
        except SystemExit as exc:
            last_err = f"resample_error({src_label}): {exc}"
            continue

        if not dst_bars:
            last_err = f"resample_empty({src_label})"
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        write_cache(dst_path, dst_bars)
        return CacheResampleOutcome(
            ok=True,
            dst_path=dst_path,
            src_bar_size=str(src_label),
            src_path=source_path,
            src_rows=len(src_bars),
            dst_rows=len(dst_bars),
            dropped_incomplete=int(stats.dropped_incomplete),
            error=None,
        )

    return CacheResampleOutcome(ok=False, dst_path=dst_path, error=last_err)


def ensure_cached_window_with_policy(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
    cache_policy: str = "strict",
) -> tuple[bool, Path, Path | None, list[tuple[date, date]], str | None]:
    policy = str(cache_policy or "strict").strip().lower() or "strict"
    if policy not in {"strict", "auto"}:
        raise ValueError(f"Unsupported cache_policy: {cache_policy!r}")

    ok, expected, resolved, missing_ranges, err = ensure_offline_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if ok or policy == "strict":
        return ok, expected, resolved, missing_ranges, err

    attempt_notes: list[str] = []
    rs_out = resample_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        dst_bar_size=bar_size,
        use_rth=use_rth,
        src_bar_size=None,
    )
    if rs_out.ok:
        src_note = (
            f"{rs_out.src_bar_size} ({rs_out.src_path})"
            if rs_out.src_path is not None
            else str(rs_out.src_bar_size)
        )
        attempt_notes.append(f"auto_resample_ok:{src_note}")
        ok2, expected2, resolved2, missing2, err2 = ensure_offline_cached_window(
            data=data,
            cache_dir=cache_dir,
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if ok2:
            return ok2, expected2, resolved2, missing2, err2
        err = err2
        expected = expected2
        resolved = resolved2
        missing_ranges = missing2
    elif rs_out.error:
        attempt_notes.append(str(rs_out.error))

    try:
        data.load_or_fetch_bar_series(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
        attempt_notes.append("auto_fetch_attempted")
    except Exception as exc:
        attempt_notes.append(f"auto_fetch_error:{exc}")

    ok3, expected3, resolved3, missing3, err3 = ensure_offline_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if ok3:
        return ok3, expected3, resolved3, missing3, err3

    details = [str(err3 or err or "").strip(), *[str(x).strip() for x in attempt_notes if str(x).strip()]]
    err_out = "; ".join([x for x in details if x])
    return ok3, expected3, resolved3, missing3, (err_out or None)
