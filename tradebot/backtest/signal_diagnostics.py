"""Diagnostics for EMA reversal signals on spot/futures bars.

This is analysis-only tooling to answer:
- Are EMA crosses actually near local tops/bottoms?
- What does MAE/MFE look like after a signal?
- How do simple gates (regime filter, debounce/confirm bars) change signal quality?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .data import IBKRHistoricalData
from .models import Bar
from ..signals import ema_cross, ema_next, ema_periods, ema_slope_pct, ema_spread_pct, parse_bar_size


@dataclass(frozen=True)
class _Event:
    ts: datetime
    direction: str  # "up" | "down"
    close: float
    ema_fast: float
    ema_slow: float
    ema_spread_pct: float
    ema_slope_pct: float | None
    rv: float | None
    bars_in_day: int


@dataclass(frozen=True)
class _EventMetrics:
    mfe: float
    mae: float
    mfe_pct: float
    mae_pct: float
    rank_in_window: float | None
    dist_to_local_extreme_pct: float | None


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA signal diagnostics (MAE/MFE + turning-point proximity)")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: 14 days ago)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--bar-size", default="15 mins")
    parser.add_argument("--use-rth", action="store_true", help="Use regular trading hours only (default: false)")
    parser.add_argument("--offline", action="store_true", help="Use cache only (do not fetch)")
    parser.add_argument("--cache-dir", default="db")

    parser.add_argument("--entry-ema", default="3/7", help="EMA preset like 3/7")
    parser.add_argument("--regime-ema", default="", help="Optional EMA preset like 20/50 (blank disables)")
    parser.add_argument("--confirm-bars", type=int, default=0, help="Bars to confirm after a cross (0 = immediate)")

    parser.add_argument("--lookahead-bars", type=int, default=16, help="Bars ahead for MAE/MFE")
    parser.add_argument(
        "--extrema-window-bars",
        type=int,
        default=16,
        help="Bars on each side to compute local top/bottom proximity",
    )

    parser.add_argument("--out-dir", default="backtests/diagnostics")
    args = parser.parse_args()

    today = datetime.now(tz=timezone.utc).date()
    end = _parse_date(args.end) if args.end else today
    start = _parse_date(args.start) if args.start else (end - timedelta(days=14))

    if start >= end:
        raise SystemExit("start must be before end")

    bar_size = str(args.bar_size)
    use_rth = bool(args.use_rth)

    entry_periods = ema_periods(str(args.entry_ema))
    if entry_periods is None:
        raise SystemExit(f"Invalid --entry-ema: {args.entry_ema!r}")
    entry_fast_p, entry_slow_p = entry_periods

    regime_raw = str(args.regime_ema or "").strip()
    regime_periods = ema_periods(regime_raw) if regime_raw else None
    confirm_bars = max(0, int(args.confirm_bars))
    lookahead = max(1, int(args.lookahead_bars))
    window = max(1, int(args.extrema_window_bars))

    data = IBKRHistoricalData()
    start_dt = datetime.combine(start, time(0, 0))
    end_dt = datetime.combine(end, time(23, 59))
    cache_dir = Path(args.cache_dir)
    if bool(args.offline):
        bars = data.load_cached_bars(
            symbol=str(args.symbol).strip().upper(),
            exchange=args.exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
    else:
        bars = data.load_or_fetch_bars(
            symbol=str(args.symbol).strip().upper(),
            exchange=args.exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
    data.disconnect()
    if not bars:
        raise SystemExit("No bars loaded")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "rth" if use_rth else "full"
    safe_bar = bar_size.replace(" ", "")
    sym = str(args.symbol).strip().upper()
    ex_tag = str(args.exchange).strip().upper() if args.exchange else ""
    entry_tag = str(args.entry_ema).strip().replace("/", "x").replace("-", "x").replace("_", "x")
    regime_tag = (
        str(regime_raw).strip().replace("/", "x").replace("-", "x").replace("_", "x")
        if regime_raw
        else "none"
    )
    base_parts = [sym]
    if ex_tag:
        base_parts.append(ex_tag)
    base_parts.extend(
        [
            str(start),
            str(end),
            safe_bar,
            tag,
            f"e{entry_tag}",
            f"r{regime_tag}",
            f"c{confirm_bars}",
            stamp,
        ]
    )
    base = "_".join(base_parts)
    events_csv = out_dir / f"{base}_events.csv"
    summary_json = out_dir / f"{base}_summary.json"

    # Precompute bars-in-day.
    bars_in_day: list[int] = []
    cur_day = None
    n = 0
    for bar in bars:
        day = bar.ts.date()
        if day != cur_day:
            cur_day = day
            n = 0
        n += 1
        bars_in_day.append(n)

    closes = [bar.close for bar in bars]
    rvs = _rv_series(closes, bar_size=bar_size, use_rth=use_rth, lookback=60, lam=0.94)

    entry_ema_fast = None
    entry_ema_slow = None
    prev_entry_fast = None
    prev_entry_slow = None
    prev_entry_fast_for_slope = None

    regime_ema_fast = None
    regime_ema_slow = None

    entry_state = None
    entry_streak = 0
    entry_streak_started_by_change = False

    events: list[_Event] = []
    gated_events: list[_Event] = []

    for idx, bar in enumerate(bars):
        close = float(bar.close)
        if close <= 0:
            continue

        prev_entry_fast = entry_ema_fast
        prev_entry_slow = entry_ema_slow
        prev_entry_fast_for_slope = entry_ema_fast
        entry_ema_fast = ema_next(entry_ema_fast, close, entry_fast_p)
        entry_ema_slow = ema_next(entry_ema_slow, close, entry_slow_p)

        if regime_periods is not None:
            reg_fast_p, reg_slow_p = regime_periods
            regime_ema_fast = ema_next(regime_ema_fast, close, reg_fast_p)
            regime_ema_slow = ema_next(regime_ema_slow, close, reg_slow_p)

        if entry_ema_fast is None or entry_ema_slow is None:
            continue
        if idx < entry_slow_p:
            continue

        # Entry signal (raw cross-up/down).
        cross_up = False
        cross_down = False
        if prev_entry_fast is not None and prev_entry_slow is not None:
            cross_up, cross_down = ema_cross(prev_entry_fast, prev_entry_slow, entry_ema_fast, entry_ema_slow)

        state = "up" if entry_ema_fast > entry_ema_slow else "down" if entry_ema_fast < entry_ema_slow else None
        if state is None:
            entry_state = None
            entry_streak = 0
            entry_streak_started_by_change = False
        else:
            if state == entry_state:
                entry_streak += 1
            else:
                entry_state = state
                entry_streak = 1
                entry_streak_started_by_change = True

        slope = None
        if prev_entry_fast_for_slope is not None:
            slope = ema_slope_pct(entry_ema_fast, prev_entry_fast_for_slope, close)

        e = None
        if cross_up:
            e = _Event(
                ts=bar.ts,
                direction="up",
                close=close,
                ema_fast=float(entry_ema_fast),
                ema_slow=float(entry_ema_slow),
                ema_spread_pct=ema_spread_pct(entry_ema_fast, entry_ema_slow, close),
                ema_slope_pct=float(slope) if slope is not None else None,
                rv=rvs[idx],
                bars_in_day=bars_in_day[idx],
            )
        elif cross_down:
            e = _Event(
                ts=bar.ts,
                direction="down",
                close=close,
                ema_fast=float(entry_ema_fast),
                ema_slow=float(entry_ema_slow),
                ema_spread_pct=ema_spread_pct(entry_ema_fast, entry_ema_slow, close),
                ema_slope_pct=float(slope) if slope is not None else None,
                rv=rvs[idx],
                bars_in_day=bars_in_day[idx],
            )
        if e is not None:
            events.append(e)

        # Regime gating + confirm-bars view of the same stream.
        gated_dir = _gated_entry_direction(
            cross_up=cross_up,
            cross_down=cross_down,
            entry_state=entry_state,
            entry_streak=entry_streak,
            entry_streak_started_by_change=entry_streak_started_by_change,
            confirm_bars=confirm_bars,
            regime_ema_fast=regime_ema_fast,
            regime_ema_slow=regime_ema_slow,
        )
        if gated_dir in ("up", "down"):
            gated_events.append(
                _Event(
                    ts=bar.ts,
                    direction=str(gated_dir),
                    close=close,
                    ema_fast=float(entry_ema_fast),
                    ema_slow=float(entry_ema_slow),
                    ema_spread_pct=ema_spread_pct(entry_ema_fast, entry_ema_slow, close),
                    ema_slope_pct=float(slope) if slope is not None else None,
                    rv=rvs[idx],
                    bars_in_day=bars_in_day[idx],
                )
            )
            entry_streak_started_by_change = False

    # Metrics computation.
    event_rows = []
    for e in events:
        idx = _index_for_ts(bars, e.ts)
        m = _event_metrics(bars, idx, e.direction, lookahead_bars=lookahead, extrema_window=window)
        event_rows.append({**asdict(e), **asdict(m), "gated": False})
    gated_rows = []
    for e in gated_events:
        idx = _index_for_ts(bars, e.ts)
        m = _event_metrics(bars, idx, e.direction, lookahead_bars=lookahead, extrema_window=window)
        gated_rows.append({**asdict(e), **asdict(m), "gated": True})

    all_rows = event_rows + gated_rows
    _write_csv(events_csv, all_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(args.symbol).strip().upper(),
        "exchange": args.exchange,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bar_size": bar_size,
        "use_rth": use_rth,
        "entry_ema": str(args.entry_ema),
        "regime_ema": regime_raw if regime_raw else None,
        "confirm_bars": confirm_bars,
        "lookahead_bars": lookahead,
        "extrema_window_bars": window,
        "counts": {
            "raw_events": len(events),
            "gated_events": len(gated_events),
        },
        "raw": _aggregate(event_rows),
        "gated": _aggregate(gated_rows),
        "paths": {
            "events_csv": str(events_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote: {events_csv}")
    print(f"Wrote: {summary_json}")


def _parse_date(value: str) -> date:
    year, month, day = str(value).split("-")
    return date(int(year), int(month), int(day))


def _index_for_ts(bars: list[Bar], ts: datetime) -> int:
    # Bars are in-order; use linear scan to keep this dependency-free.
    for i, bar in enumerate(bars):
        if bar.ts == ts:
            return i
    raise ValueError(f"Event timestamp not found in bars: {ts!r}")


def _annualization_factor(bar_size: str, use_rth: bool) -> float:
    label = str(bar_size or "").strip().lower()
    if "day" in label:
        return 252.0
    bar_def = parse_bar_size(bar_size)
    if bar_def is None:
        return 252.0
    hours = bar_def.duration.total_seconds() / 3600.0
    if hours <= 0:
        return 252.0
    session_hours = 6.5 if use_rth else 24.0
    return 252.0 * (session_hours / hours)


def _rv_series(
    closes: list[float],
    *,
    bar_size: str,
    use_rth: bool,
    lookback: int,
    lam: float,
) -> list[float | None]:
    if len(closes) < 2:
        return [None for _ in closes]
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = float(closes[i - 1])
        cur = float(closes[i])
        if prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
        else:
            returns.append(0.0)

    ann = math.sqrt(_annualization_factor(bar_size, use_rth))
    out: list[float | None] = [None]
    for i in range(1, len(closes)):
        start = max(0, i - lookback)
        window = returns[start:i]
        if not window:
            out.append(None)
            continue
        variance = 0.0
        for r in window:
            variance = lam * variance + (1.0 - lam) * (r * r)
        out.append(math.sqrt(max(0.0, variance)) * ann)
    return out


def _gated_entry_direction(
    *,
    cross_up: bool,
    cross_down: bool,
    entry_state: str | None,
    entry_streak: int,
    entry_streak_started_by_change: bool,
    confirm_bars: int,
    regime_ema_fast: float | None,
    regime_ema_slow: float | None,
) -> str | None:
    # Cross-based entry, optionally delayed by confirm_bars.
    if entry_state not in ("up", "down"):
        return None

    if confirm_bars <= 0:
        if cross_up:
            cand = "up"
        elif cross_down:
            cand = "down"
        else:
            return None
    else:
        # After a state change, wait N bars and only then trigger.
        if not entry_streak_started_by_change:
            return None
        if entry_streak != (confirm_bars + 1):
            return None
        cand = entry_state

    # Optional regime gate.
    if regime_ema_fast is not None and regime_ema_slow is not None:
        regime = "up" if regime_ema_fast > regime_ema_slow else "down" if regime_ema_fast < regime_ema_slow else None
        if regime != cand:
            return None
    return cand


def _event_metrics(bars: list[Bar], idx: int, direction: str, *, lookahead_bars: int, extrema_window: int) -> _EventMetrics:
    entry = float(bars[idx].close)
    end = min(len(bars), idx + 1 + lookahead_bars)
    future = bars[idx + 1 : end]
    if not future:
        return _EventMetrics(
            mfe=0.0,
            mae=0.0,
            mfe_pct=0.0,
            mae_pct=0.0,
            rank_in_window=None,
            dist_to_local_extreme_pct=None,
        )

    highs = [float(b.high) for b in future]
    lows = [float(b.low) for b in future]
    max_high = max(highs)
    min_low = min(lows)
    if direction == "down":
        mfe = entry - min_low
        mae = max_high - entry
    else:
        mfe = max_high - entry
        mae = entry - min_low

    mfe_pct = (mfe / entry) if entry > 0 else 0.0
    mae_pct = (mae / entry) if entry > 0 else 0.0

    # Turning-point proximity in a local window around the signal.
    left = max(0, idx - extrema_window)
    right = min(len(bars), idx + extrema_window + 1)
    window_closes = [float(b.close) for b in bars[left:right]]
    if len(window_closes) < 3:
        rank = None
        dist = None
    else:
        lo = min(window_closes)
        hi = max(window_closes)
        if hi <= lo or entry <= 0:
            rank = None
            dist = None
        else:
            rank = (entry - lo) / (hi - lo)
            if direction == "up":
                dist = (entry - lo) / entry
            else:
                dist = (hi - entry) / entry

    return _EventMetrics(
        mfe=float(mfe),
        mae=float(mae),
        mfe_pct=float(mfe_pct),
        mae_pct=float(mae_pct),
        rank_in_window=float(rank) if rank is not None else None,
        dist_to_local_extreme_pct=float(dist) if dist is not None else None,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"by_dir": {}, "overall": {}}

    def _num(key: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out

    def _pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        vals = sorted(vals)
        k = int(round((len(vals) - 1) * p))
        k = max(0, min(len(vals) - 1, k))
        return float(vals[k])

    def _stats(sub: list[dict]) -> dict:
        mfe = _num_from(sub, "mfe_pct")
        mae = _num_from(sub, "mae_pct")
        rank = _num_from(sub, "rank_in_window")
        dist = _num_from(sub, "dist_to_local_extreme_pct")
        return {
            "count": len(sub),
            "mfe_pct_p50": _pct(mfe, 0.50),
            "mfe_pct_p90": _pct(mfe, 0.90),
            "mae_pct_p50": _pct(mae, 0.50),
            "mae_pct_p90": _pct(mae, 0.90),
            "rank_p50": _pct(rank, 0.50),
            "dist_to_extreme_pct_p50": _pct(dist, 0.50),
        }

    def _num_from(sub: list[dict], key: str) -> list[float]:
        vals = []
        for r in sub:
            v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return vals

    by_dir = {}
    for d in ("up", "down"):
        sub = [r for r in rows if r.get("direction") == d]
        if sub:
            by_dir[d] = _stats(sub)

    return {"by_dir": by_dir, "overall": _stats(rows)}


if __name__ == "__main__":
    main()
