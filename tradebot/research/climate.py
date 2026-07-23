"""Research adapters for archived daily climate experiments."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from ..backtest.engine import run_backtest
from ..backtest.spot_codec import (
    filters_from_payload,
    make_bundle,
    metrics_from_summary,
    strategy_from_payload,
)
from ..climate_router import DailyBar, host_policy, year_slice
from ..engines.signals import SupertrendEngine
from ..spot.codec import effective_filters_payload


def load_daily_bars_from_intraday_csv(path: Path) -> list[DailyBar]:
    days: list[DailyBar] = []
    current_day: str | None = None
    day_open = 0.0
    day_high = 0.0
    day_low = 0.0
    day_close = 0.0

    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            day = str(row["ts"])[:10]
            o = float(row["open"])
            h = float(row["high"])
            low = float(row["low"])
            c = float(row["close"])
            if day != current_day:
                if current_day is not None:
                    days.append(DailyBar(ts=current_day, open=day_open, high=day_high, low=day_low, close=day_close))
                current_day = day
                day_open = o
                day_high = h
                day_low = low
                day_close = c
            else:
                if h > day_high:
                    day_high = h
                if low < day_low:
                    day_low = low
                day_close = c

    if current_day is not None:
        days.append(DailyBar(ts=current_day, open=day_open, high=day_high, low=day_low, close=day_close))
    return days


def _pdd_from_equity_curve(curve: list[float]) -> tuple[float, float, float]:
    equity = curve[-1] if curve else 100_000.0
    peak = 100_000.0
    maxdd = 0.0
    for value in curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > maxdd:
            maxdd = dd
    pnl = equity - 100_000.0
    pdd = (pnl / 100_000.0) / maxdd if maxdd > 0 else (float("inf") if pnl > 0 else 0.0)
    return float(pnl), float(maxdd), float(pdd)


def buyhold_year_pdd(days: list[DailyBar], year: int) -> tuple[float, float, float]:
    seg = year_slice(days, year)
    if len(seg) < 2:
        raise SystemExit(f"Not enough daily bars for year {year}")
    curve: list[float] = []
    equity = 100_000.0
    prev_close = seg[0].close
    for bar in seg:
        ret = (bar.close / prev_close) - 1.0
        equity *= 1.0 + ret
        curve.append(equity)
        prev_close = bar.close
    return _pdd_from_equity_curve(curve)


def moving_average_year_pdd(
    days: list[DailyBar],
    year: int,
    *,
    window: int = 200,
    entry_buffer: float = 0.0,
    exit_buffer: float = 0.0,
) -> tuple[float, float, float]:
    closes = [bar.close for bar in days]
    ma: list[float | None] = [None] * len(days)
    for i in range(int(window) - 1, len(days)):
        ma[i] = sum(closes[i - int(window) + 1 : i + 1]) / float(window)

    idxs = [i for i, bar in enumerate(days) if f"{year:04d}-" <= bar.ts < f"{year + 1:04d}-"]
    if len(idxs) < 2:
        raise SystemExit(f"Not enough daily bars for year {year}")

    curve: list[float] = []
    equity = 100_000.0
    prev_close = days[idxs[0] - 1].close if idxs[0] > 0 else days[idxs[0]].close
    pos = 0.0
    for i in idxs:
        prev_idx = i - 1
        prev_ma = ma[prev_idx] if prev_idx >= 0 else None
        prev_price = days[prev_idx].close if prev_idx >= 0 else days[i].close
        if prev_ma is not None:
            if pos <= 0.0 and prev_price >= float(prev_ma) * (1.0 + float(entry_buffer)):
                pos = 1.0
            elif pos > 0.0 and prev_price <= float(prev_ma) * (1.0 - float(exit_buffer)):
                pos = 0.0
        ret = (days[i].close / prev_close) - 1.0
        equity *= 1.0 + (pos * ret)
        curve.append(equity)
        prev_close = days[i].close
    return _pdd_from_equity_curve(curve)


def drawdown_kill_year_pdd(
    days: list[DailyBar],
    year: int,
    *,
    on_dd: float,
    off_dd: float,
    ma_window: int = 0,
    reentry_buffer: float = 0.0,
) -> tuple[float, float, float]:
    idxs = [i for i, bar in enumerate(days) if f"{year:04d}-" <= bar.ts < f"{year + 1:04d}-"]
    if len(idxs) < 2:
        raise SystemExit(f"Not enough daily bars for year {year}")

    closes = [bar.close for bar in days]
    sma: list[float | None] = [None] * len(days)
    if int(ma_window) > 0:
        for i in range(int(ma_window) - 1, len(days)):
            sma[i] = sum(closes[i - int(ma_window) + 1 : i + 1]) / float(ma_window)

    equity = 100_000.0
    curve: list[float] = []
    prev_close = days[idxs[0] - 1].close if idxs[0] > 0 else days[idxs[0]].close
    price_peak = days[idxs[0]].close
    pos = 1.0
    for i in idxs:
        close = days[i].close
        ret = (close / prev_close) - 1.0
        equity *= 1.0 + (pos * ret)
        curve.append(equity)
        prev_close = close
        if close > price_peak:
            price_peak = close
        price_dd = (price_peak - close) / price_peak if price_peak > 0 else 0.0
        if pos > 0.0 and price_dd >= float(on_dd):
            pos = 0.0
        elif pos <= 0.0:
            ma_ok = True
            if int(ma_window) > 0:
                prev_ma = sma[i]
                ma_ok = prev_ma is not None and close >= float(prev_ma) * (1.0 + float(reentry_buffer))
            if price_dd <= float(off_dd) and bool(ma_ok):
                pos = 1.0
                price_peak = close
    return _pdd_from_equity_curve(curve)


def named_host_year_pdd(days: list[DailyBar], year: int, host_name: str) -> tuple[float, float, float]:
    host = host_policy(host_name).name
    if host == "buyhold":
        return buyhold_year_pdd(days, year)
    if host == "bull_ma200_v1":
        return moving_average_year_pdd(days, year, window=200)
    if host == "sma200":
        return moving_average_year_pdd(days, year, window=200)
    if host == "lf_defensive_long_v1":
        return moving_average_year_pdd(days, year, window=50, entry_buffer=0.02, exit_buffer=0.0)
    if host == "lf_defensive_long_v2":
        return drawdown_kill_year_pdd(days, year, on_dd=0.15, off_dd=0.08, ma_window=0, reentry_buffer=0.0)
    if host == "lf_defensive_long_st_v1":
        return supertrend_long_year_pdd(days, year, atr_period=21, multiplier=1.5, reentry_confirm_days=0)
    raise SystemExit(f"Unknown host: {host_name!r}")


def supertrend_long_year_pdd(
    days: list[DailyBar],
    year: int,
    *,
    atr_period: int,
    multiplier: float,
    reentry_confirm_days: int = 0,
) -> tuple[float, float, float]:
    seg = year_slice(days, year)
    if len(seg) < 2:
        raise SystemExit(f"Not enough daily bars for year {year}")

    engine = SupertrendEngine(atr_period=int(atr_period), multiplier=float(multiplier), source="hl2")
    equity = 100_000.0
    curve: list[float] = []
    prev_close = seg[0].close
    pos = 0.0
    pending_up = 0
    pending_down = 0
    for bar in seg:
        ret = (bar.close / prev_close) - 1.0
        equity *= 1.0 + (pos * ret)
        curve.append(equity)
        prev_close = bar.close
        snap = engine.update(high=float(bar.high), low=float(bar.low), close=float(bar.close))
        if snap.ready and snap.direction in ("up", "down"):
            if str(snap.direction) == "up":
                pending_up += 1
                pending_down = 0
            else:
                pending_down += 1
                pending_up = 0
            if str(snap.direction) == "up" and pending_up >= max(1, int(reentry_confirm_days) + 1):
                pos = 1.0
            elif str(snap.direction) == "down" and pending_down >= 1:
                pos = 0.0
    return _pdd_from_equity_curve(curve)


def load_hf_host_strategy(milestones_path: Path):
    payload = json.loads(milestones_path.read_text())
    groups = payload.get("groups") or []
    if not groups or not isinstance(groups[0], dict):
        raise SystemExit(f"No groups in milestones payload: {milestones_path}")
    group = groups[0]
    entries = group.get("entries") or []
    if not entries or not isinstance(entries[0], dict):
        raise SystemExit(f"No entries in milestones payload: {milestones_path}")
    entry = entries[0]
    strategy_payload = entry.get("strategy") or {}
    if not isinstance(strategy_payload, dict):
        raise SystemExit(f"Invalid strategy payload in {milestones_path}")
    filters_payload = effective_filters_payload(group_filters=group.get("filters"), strategy=strategy_payload)
    filters = filters_from_payload(filters_payload)
    strategy = strategy_from_payload(strategy_payload, filters=filters)
    bar_size = str(strategy_payload.get("signal_bar_size") or "5 mins")
    use_rth = bool(strategy_payload.get("signal_use_rth", True))
    return strategy, bar_size, use_rth


def hf_host_year_stats(*, strategy, year: int, bar_size: str, use_rth: bool) -> tuple[float, float, float, int, float]:
    cfg = make_bundle(
        strategy=strategy,
        start=date(year, 1, 1),
        end=date(year + 1, 1, 1),
        bar_size=bar_size,
        use_rth=use_rth,
        cache_dir="db",
        offline=True,
    )
    res = run_backtest(cfg)
    summary = metrics_from_summary(res.summary)
    return (
        float(summary["pnl"]),
        float(summary["dd"]),
        float(summary["pnl_over_dd"]),
        int(summary["trades"]),
        float(summary["win_rate"]),
    )
