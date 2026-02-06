"""Derive a larger-bar cache from an existing smaller-bar cache.

Goal: avoid IBKR refetch drift by deterministically resampling from an on-disk cache.

Example (SLV, FULL24):
  python -m tradebot.backtest.tools.resample_cache \\
    --symbol SLV --start 2016-01-08 --end 2026-01-08 \\
    --src-bar-size "5 mins" --dst-bar-size "10 mins" \\
    --cache-dir db
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

from ..data import _cache_path, _find_covering_cache_path, _read_cache, _write_cache
from ..models import Bar
from ...signals import parse_bar_size


_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class _ResampleStats:
    kept: int
    dropped_incomplete: int


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
        raise SystemExit(f"Non-integer resample ratio: {src_bar_size!r} → {dst_bar_size!r}")

    factor = int(dst.duration.total_seconds() // src.duration.total_seconds())
    factor = max(1, factor)

    if not bars:
        return [], _ResampleStats(kept=0, dropped_incomplete=0)

    bars = sorted(bars, key=lambda b: b.ts)

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


def _parse_date(value: str) -> datetime.date:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise SystemExit("Missing date")
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError as e:
        raise SystemExit(f"Invalid date: {value!r}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Resample cached OHLCV bars (no IBKR refetch).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--src-bar-size", required=True, help='e.g. "5 mins"')
    ap.add_argument("--dst-bar-size", required=True, help='e.g. "10 mins"')
    ap.add_argument("--cache-dir", default="db")
    ap.add_argument("--use-rth", action="store_true", help="Use RTH-only cache tag (default: FULL24).")
    args = ap.parse_args()

    symbol = str(args.symbol).strip().upper()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    start_dt = datetime.combine(start, time(0, 0))
    end_dt = datetime.combine(end, time(23, 59))
    cache_dir = Path(args.cache_dir)
    use_rth = bool(args.use_rth)
    src_bar_size = str(args.src_bar_size).strip()
    dst_bar_size = str(args.dst_bar_size).strip()

    src_path = _cache_path(cache_dir, symbol, start_dt, end_dt, src_bar_size, use_rth)
    if not src_path.exists():
        covering = _find_covering_cache_path(
            cache_dir=cache_dir,
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            bar_size=src_bar_size,
            use_rth=use_rth,
        )
        if covering is None:
            raise SystemExit(f"Source cache not found: {src_path}")
        src_path = covering

    src_bars = [b for b in _read_cache(src_path) if start_dt <= b.ts <= end_dt]
    if not src_bars:
        raise SystemExit(f"Source cache is empty after slicing: {src_path}")

    dst_bars, stats = _resample_intraday_ohlcv(
        src_bars,
        src_bar_size=src_bar_size,
        dst_bar_size=dst_bar_size,
    )
    if not dst_bars:
        raise SystemExit("Resample produced 0 bars (likely misaligned source cache).")

    dst_path = _cache_path(cache_dir, symbol, start_dt, end_dt, dst_bar_size, use_rth)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(dst_path, dst_bars)

    print("")
    print("=== cache resample ===")
    print(f"- symbol={symbol} use_rth={use_rth}")
    print(f"- src={src_bar_size} rows={len(src_bars)} path={src_path}")
    print(f"- dst={dst_bar_size} rows={len(dst_bars)} path={dst_path}")
    print(f"- dropped_incomplete={stats.dropped_incomplete}")
    print(f"- first={dst_bars[0].ts} last={dst_bars[-1].ts}")
    print("")


if __name__ == "__main__":
    main()

