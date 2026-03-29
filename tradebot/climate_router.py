"""Central daily climate router primitives.

This module is intentionally host-agnostic at the contract level:
- extract daily bars from a tradable time series
- compute yearly climate features
- classify climate
- evaluate simple daily hosts
- evaluate an HF host through the real backtest engine

Backtest tools can wrap this.
Live routing can also import the same primitives later.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, median, pstdev

from .backtest.engine import run_backtest
from .backtest.spot_codec import effective_filters_payload, filters_from_payload, make_bundle, metrics_from_summary, strategy_from_payload


@dataclass(frozen=True)
class DailyBar:
    ts: str
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class YearFeatures:
    year: int
    ret: float
    maxdd: float
    rv: float
    atr_med: float
    atr_mean: float
    up_frac: float
    efficiency: float
    dd_frac_ge_10pct: float


@dataclass(frozen=True)
class ClimateDecision:
    climate: str
    chosen_host: str


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
            l = float(row["low"])
            c = float(row["close"])
            if day != current_day:
                if current_day is not None:
                    days.append(DailyBar(ts=current_day, open=day_open, high=day_high, low=day_low, close=day_close))
                current_day = day
                day_open = o
                day_high = h
                day_low = l
                day_close = c
            else:
                if h > day_high:
                    day_high = h
                if l < day_low:
                    day_low = l
                day_close = c

    if current_day is not None:
        days.append(DailyBar(ts=current_day, open=day_open, high=day_high, low=day_low, close=day_close))
    return days


def year_slice(days: list[DailyBar], year: int) -> list[DailyBar]:
    lo = f"{year:04d}-"
    hi = f"{year + 1:04d}-"
    return [bar for bar in days if lo <= bar.ts < hi]


def compute_year_features(days: list[DailyBar], year: int) -> YearFeatures:
    seg = year_slice(days, year)
    if len(seg) < 2:
        raise SystemExit(f"Not enough daily bars for year {year}")

    closes = [bar.close for bar in seg]
    rets = [(seg[i].close / seg[i - 1].close) - 1.0 for i in range(1, len(seg))]
    tr_pcts: list[float] = []
    for i in range(1, len(seg)):
        prev = seg[i - 1].close
        tr = max(
            seg[i].high - seg[i].low,
            abs(seg[i].high - prev),
            abs(seg[i].low - prev),
        ) / prev
        tr_pcts.append(tr)

    peak = closes[0]
    maxdd = 0.0
    dd_days = 0
    for close in closes:
        if close > peak:
            peak = close
        dd = (peak - close) / peak
        if dd > maxdd:
            maxdd = dd
        if dd >= 0.10:
            dd_days += 1

    total_ret = (closes[-1] / closes[0]) - 1.0
    rv = (pstdev(rets) * math.sqrt(252)) if len(rets) > 1 else 0.0
    path = sum(abs(ret) for ret in rets)
    efficiency = (abs(total_ret) / path) if path > 0 else 0.0
    up_frac = (sum(1 for ret in rets if ret > 0) / len(rets)) if rets else 0.0
    dd_frac = dd_days / len(closes)

    return YearFeatures(
        year=int(year),
        ret=float(total_ret),
        maxdd=float(maxdd),
        rv=float(rv),
        atr_med=float(median(tr_pcts)) if tr_pcts else 0.0,
        atr_mean=float(mean(tr_pcts)) if tr_pcts else 0.0,
        up_frac=float(up_frac),
        efficiency=float(efficiency),
        dd_frac_ge_10pct=float(dd_frac),
    )


def classify_climate_v2(features: YearFeatures) -> ClimateDecision:
    if (
        features.ret > 0.0
        and features.maxdd <= 0.40
        and features.rv <= 0.55
        and features.dd_frac_ge_10pct < 0.45
    ):
        return ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")
    if features.ret > 0.0:
        return ClimateDecision(climate="positive_high_stress_transition", chosen_host="hf_host")
    if (
        features.maxdd >= 0.70
        or features.rv >= 0.85
        or features.dd_frac_ge_10pct >= 0.80
    ):
        return ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")
    return ClimateDecision(climate="negative_transition_bear", chosen_host="sma200")


def classify_climate_v3(features: YearFeatures) -> ClimateDecision:
    if (
        features.ret > 0.0
        and features.maxdd <= 0.40
        and features.rv <= 0.55
        and features.dd_frac_ge_10pct < 0.45
    ):
        return ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")
    if features.ret > 0.0:
        return ClimateDecision(climate="positive_high_stress_transition", chosen_host="hf_host")
    if (
        features.maxdd >= 0.70
        or features.rv >= 0.85
        or features.dd_frac_ge_10pct >= 0.80
    ):
        return ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")
    return ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v1")


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


def named_host_year_pdd(days: list[DailyBar], year: int, host_name: str) -> tuple[float, float, float]:
    host = str(host_name).strip().lower()
    if host == "buyhold":
        return buyhold_year_pdd(days, year)
    if host == "sma200":
        return moving_average_year_pdd(days, year, window=200)
    if host == "lf_defensive_long_v1":
        return moving_average_year_pdd(days, year, window=50, entry_buffer=0.02, exit_buffer=0.0)
    raise SystemExit(f"Unknown host: {host_name!r}")


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
