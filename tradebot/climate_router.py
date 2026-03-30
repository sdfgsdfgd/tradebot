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
from collections.abc import Mapping

from .engine import SupertrendEngine
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


@dataclass(frozen=True)
class HostPolicy:
    name: str
    host_managed: bool


@dataclass(frozen=True)
class RollingClimateState:
    ts: str
    fast_features: YearFeatures
    slow_features: YearFeatures
    proposed: ClimateDecision
    active: ClimateDecision
    dwell_days: int


@dataclass(frozen=True)
class RegimeRouterConfig:
    enabled: bool = False
    fast_window_days: int = 63
    slow_window_days: int = 126
    min_dwell_days: int = 10


@dataclass(frozen=True)
class RegimeRouterSnapshot:
    ready: bool
    climate: str | None
    chosen_host: str | None
    effective_entry_dir: str | None
    host_managed: bool = False
    bull_sovereign_ok: bool = False
    dwell_days: int = 0


def _get(source: Mapping[str, object] | object | None, key: str, default: object = None):
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def regime_router_config(strategy: Mapping[str, object] | object | None) -> RegimeRouterConfig:
    enabled = bool(_get(strategy, "regime_router", False))
    fast = int(_get(strategy, "regime_router_fast_window_days", 63) or 63)
    slow = int(_get(strategy, "regime_router_slow_window_days", 126) or 126)
    dwell = int(_get(strategy, "regime_router_min_dwell_days", 10) or 10)
    return RegimeRouterConfig(
        enabled=bool(enabled),
        fast_window_days=max(2, int(fast)),
        slow_window_days=max(max(2, int(fast)), int(slow)),
        min_dwell_days=max(1, int(dwell)),
    )


_HOST_POLICIES: dict[str, HostPolicy] = {
    "hf_host": HostPolicy(name="hf_host", host_managed=False),
    "buyhold": HostPolicy(name="buyhold", host_managed=True),
    "bull_ma200_v1": HostPolicy(name="bull_ma200_v1", host_managed=True),
    "sma200": HostPolicy(name="sma200", host_managed=True),
    "lf_defensive_long_v1": HostPolicy(name="lf_defensive_long_v1", host_managed=True),
    "lf_defensive_long_v2": HostPolicy(name="lf_defensive_long_v2", host_managed=True),
    "lf_defensive_long_st_v1": HostPolicy(name="lf_defensive_long_st_v1", host_managed=True),
}


def host_policy(host_name: str) -> HostPolicy:
    key = str(host_name).strip().lower()
    policy = _HOST_POLICIES.get(key)
    if policy is None:
        raise SystemExit(f"Unknown host: {host_name!r}")
    return policy


def regime_router_dwell_days(
    *,
    active: ClimateDecision | None,
    proposed: ClimateDecision,
    base_dwell_days: int,
) -> int:
    dwell = max(1, int(base_dwell_days))
    if active is None:
        return 1
    if proposed == active:
        return dwell
    if proposed.climate == "negative_extreme_bear" and active.chosen_host != "hf_host":
        return 1
    if active.chosen_host == "hf_host" and proposed.chosen_host != "hf_host":
        return max(1, min(dwell, 5))
    return dwell


def bull_sovereign_entry_ok(
    *,
    climate: str | None,
    chosen_host: str | None,
    fast_features: YearFeatures | None,
    slow_features: YearFeatures | None,
) -> bool:
    if str(climate or "") != "bull_grind_low_vol" or str(chosen_host or "") != "buyhold":
        return False
    if fast_features is None or slow_features is None:
        return False
    return bool(
        (
            fast_features.efficiency >= 0.10
            and slow_features.efficiency >= 0.09
            and fast_features.maxdd <= 0.24
            and slow_features.maxdd <= 0.38
            and slow_features.rv <= 0.75
        )
        or (
            fast_features.efficiency >= 0.15
            and slow_features.efficiency >= 0.06
            and fast_features.maxdd <= 0.20
            and slow_features.maxdd <= 0.30
        )
    )


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


def compute_window_features(days: list[DailyBar], *, label: int, start_idx: int, end_idx: int) -> YearFeatures:
    seg = days[int(start_idx) : int(end_idx)]
    if len(seg) < 2:
        raise SystemExit(f"Not enough daily bars for label {label}")

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
        year=int(label),
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


def classify_climate_v4(features: YearFeatures) -> ClimateDecision:
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
    return ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v2")


def classify_rolling_climate_v5(
    *,
    crash_features: YearFeatures,
    fast_features: YearFeatures,
    slow_features: YearFeatures,
    active: ClimateDecision | None = None,
) -> ClimateDecision:
    slow = classify_climate_v4(slow_features)
    fast = classify_climate_v4(fast_features)

    crash_now = (
        crash_features.ret <= -0.18
        and crash_features.maxdd >= 0.30
        and crash_features.rv >= 0.80
    )
    if crash_now:
        return ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")

    if slow_features.ret > 0.0:
        fast_bull_recovery = (
            fast_features.ret > 0.0
            and fast_features.maxdd <= 0.22
            and fast_features.rv <= 0.55
            and fast_features.efficiency >= 0.15
            and slow_features.maxdd <= 0.38
            and slow_features.dd_frac_ge_10pct <= 0.50
            and slow_features.rv <= 0.75
        )
        stressed_positive = (
            slow_features.maxdd >= 0.40
            or slow_features.rv >= 0.75
            or slow_features.dd_frac_ge_10pct >= 0.55
            or (fast_features.ret < 0.0 and fast_features.maxdd >= 0.18 and fast_features.rv >= 0.65)
            or (
                slow_features.efficiency < 0.10
                and fast_features.rv >= 0.55
                and fast_features.efficiency < 0.10
            )
        )
        if active is not None and active.chosen_host == "hf_host" and fast_bull_recovery:
            return ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")
        if slow.climate == "bull_grind_low_vol":
            return slow
        if stressed_positive:
            return ClimateDecision(climate="positive_high_stress_transition", chosen_host="hf_host")
        return ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")

    if fast.climate == "negative_extreme_bear" or slow.climate == "negative_extreme_bear":
        return ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")
    return ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v2")


def classify_rolling_climate_v4(
    *,
    fast_features: YearFeatures,
    slow_features: YearFeatures,
    active: ClimateDecision | None = None,
) -> ClimateDecision:
    fast = classify_climate_v4(fast_features)
    slow = classify_climate_v4(slow_features)
    if fast.climate == "negative_extreme_bear":
        return fast
    if active is not None and active.chosen_host == "hf_host" and fast.climate == "bull_grind_low_vol":
        return fast
    return slow


def rolling_climate_states(
    days: list[DailyBar],
    *,
    fast_window_days: int = 63,
    slow_window_days: int = 126,
    min_dwell_days: int = 10,
) -> list[RollingClimateState]:
    fast_n = max(2, int(fast_window_days))
    slow_n = max(int(fast_n), int(slow_window_days))
    crash_n = max(2, min(21, fast_n))
    dwell_min = max(1, int(min_dwell_days))
    if len(days) < slow_n:
        return []

    out: list[RollingClimateState] = []
    active: ClimateDecision | None = None
    pending: ClimateDecision | None = None
    pending_days = 0

    for end in range(slow_n, len(days) + 1):
        crash = compute_window_features(days, label=end, start_idx=end - crash_n, end_idx=end)
        fast = compute_window_features(days, label=end, start_idx=end - fast_n, end_idx=end)
        slow = compute_window_features(days, label=end, start_idx=end - slow_n, end_idx=end)
        proposed = classify_rolling_climate_v5(
            crash_features=crash,
            fast_features=fast,
            slow_features=slow,
            active=active,
        )

        if active is None:
            active = proposed
            pending = None
            pending_days = 0
        elif proposed == active:
            pending = None
            pending_days = 0
        else:
            if pending is None or pending != proposed:
                pending = proposed
                pending_days = 1
            else:
                pending_days += 1
            dwell_req = regime_router_dwell_days(active=active, proposed=proposed, base_dwell_days=dwell_min)
            if pending_days >= dwell_req:
                active = proposed
                pending = None
                pending_days = 0

        out.append(
            RollingClimateState(
                ts=str(days[end - 1].ts),
                fast_features=fast,
                slow_features=slow,
                proposed=proposed,
                active=active,
                dwell_days=int(pending_days),
            )
        )
    return out


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


def moving_average_target_dir(
    days: list[DailyBar],
    *,
    window: int,
    entry_buffer: float = 0.0,
    exit_buffer: float = 0.0,
) -> str | None:
    if len(days) < max(2, int(window)):
        return None
    closes = [bar.close for bar in days]
    ma: list[float | None] = [None] * len(days)
    for i in range(int(window) - 1, len(days)):
        ma[i] = sum(closes[i - int(window) + 1 : i + 1]) / float(window)
    pos = 0.0
    for i in range(1, len(days)):
        prev_ma = ma[i - 1]
        prev_price = days[i - 1].close
        if prev_ma is not None:
            if pos <= 0.0 and prev_price >= float(prev_ma) * (1.0 + float(entry_buffer)):
                pos = 1.0
            elif pos > 0.0 and prev_price <= float(prev_ma) * (1.0 - float(exit_buffer)):
                pos = 0.0
    return "up" if pos > 0.0 else None


def drawdown_kill_target_dir(
    days: list[DailyBar],
    *,
    on_dd: float,
    off_dd: float,
    ma_window: int = 0,
    reentry_buffer: float = 0.0,
) -> str | None:
    if len(days) < 2:
        return None
    closes = [bar.close for bar in days]
    sma: list[float | None] = [None] * len(days)
    if int(ma_window) > 0:
        for i in range(int(ma_window) - 1, len(days)):
            sma[i] = sum(closes[i - int(ma_window) + 1 : i + 1]) / float(ma_window)

    price_peak = days[0].close
    pos = 1.0
    for i, bar in enumerate(days):
        close = float(bar.close)
        if close > price_peak:
            price_peak = close
        price_dd = (price_peak - close) / price_peak if price_peak > 0 else 0.0
        if pos > 0.0 and price_dd >= float(on_dd):
            pos = 0.0
        elif pos <= 0.0:
            ma_ok = True
            if int(ma_window) > 0:
                ma_now = sma[i]
                ma_ok = ma_now is not None and close >= float(ma_now) * (1.0 + float(reentry_buffer))
            if price_dd <= float(off_dd) and bool(ma_ok):
                pos = 1.0
                price_peak = close
    return "up" if pos > 0.0 else None


def supertrend_long_target_dir(
    days: list[DailyBar],
    *,
    atr_period: int,
    multiplier: float,
    reentry_confirm_days: int = 0,
) -> str | None:
    if len(days) < max(2, int(atr_period)):
        return None
    engine = SupertrendEngine(atr_period=int(atr_period), multiplier=float(multiplier), source="hl2")
    pos = 0.0
    pending_up = 0
    pending_down = 0
    for bar in days:
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
    return "up" if pos > 0.0 else None


def named_host_target_dir(days: list[DailyBar], host_name: str) -> str | None:
    host = host_policy(host_name).name
    if host == "buyhold":
        return "up" if days else None
    if host == "bull_ma200_v1":
        return moving_average_target_dir(days, window=200)
    if host == "sma200":
        return moving_average_target_dir(days, window=200)
    if host == "lf_defensive_long_v1":
        return moving_average_target_dir(days, window=50, entry_buffer=0.02, exit_buffer=0.0)
    if host == "lf_defensive_long_v2":
        return drawdown_kill_target_dir(days, on_dd=0.15, off_dd=0.08, ma_window=0, reentry_buffer=0.0)
    if host == "lf_defensive_long_st_v1":
        return supertrend_long_target_dir(days, atr_period=21, multiplier=1.5, reentry_confirm_days=0)
    if host == "hf_host":
        return None
    raise SystemExit(f"Unknown host: {host_name!r}")


class DailyRegimeRouterEngine:
    def __init__(self, *, config: RegimeRouterConfig) -> None:
        self._cfg = config
        self._completed_days: list[DailyBar] = []
        self._current_day: str | None = None
        self._day_open = 0.0
        self._day_high = 0.0
        self._day_low = 0.0
        self._day_close = 0.0
        self._active: ClimateDecision | None = None
        self._pending: ClimateDecision | None = None
        self._pending_days = 0
        self._last_fast_features: YearFeatures | None = None
        self._last_slow_features: YearFeatures | None = None

    def _finalize_day(self) -> None:
        if self._current_day is None:
            return
        self._completed_days.append(
            DailyBar(
                ts=str(self._current_day),
                open=float(self._day_open),
                high=float(self._day_high),
                low=float(self._day_low),
                close=float(self._day_close),
            )
        )
        self._recompute_state()

    def _recompute_state(self) -> None:
        if len(self._completed_days) < int(self._cfg.slow_window_days):
            self._active = None
            self._pending = None
            self._pending_days = 0
            return
        crash_n = max(2, min(21, int(self._cfg.fast_window_days)))
        crash = compute_window_features(
            self._completed_days,
            label=len(self._completed_days),
            start_idx=len(self._completed_days) - int(crash_n),
            end_idx=len(self._completed_days),
        )
        fast = compute_window_features(
            self._completed_days,
            label=len(self._completed_days),
            start_idx=len(self._completed_days) - int(self._cfg.fast_window_days),
            end_idx=len(self._completed_days),
        )
        slow = compute_window_features(
            self._completed_days,
            label=len(self._completed_days),
            start_idx=len(self._completed_days) - int(self._cfg.slow_window_days),
            end_idx=len(self._completed_days),
        )
        self._last_fast_features = fast
        self._last_slow_features = slow
        proposed = classify_rolling_climate_v5(
            crash_features=crash,
            fast_features=fast,
            slow_features=slow,
            active=self._active,
        )
        if self._active is None:
            self._active = proposed
            self._pending = None
            self._pending_days = 0
            return
        if proposed == self._active:
            self._pending = None
            self._pending_days = 0
            return
        if self._pending is None or self._pending != proposed:
            self._pending = proposed
            self._pending_days = 1
        else:
            self._pending_days += 1
        dwell_req = regime_router_dwell_days(
            active=self._active,
            proposed=proposed,
            base_dwell_days=int(self._cfg.min_dwell_days),
        )
        if self._pending_days >= int(dwell_req):
            self._active = proposed
            self._pending = None
            self._pending_days = 0

    def update_bar(
        self,
        *,
        ts: str,
        open: float,
        high: float,
        low: float,
        close: float,
        hf_entry_dir: str | None,
    ) -> RegimeRouterSnapshot:
        day = str(ts)[:10]
        if self._current_day is None:
            self._current_day = day
            self._day_open = float(open)
            self._day_high = float(high)
            self._day_low = float(low)
            self._day_close = float(close)
        elif day != self._current_day:
            self._finalize_day()
            self._current_day = day
            self._day_open = float(open)
            self._day_high = float(high)
            self._day_low = float(low)
            self._day_close = float(close)
        else:
            self._day_high = max(float(self._day_high), float(high))
            self._day_low = min(float(self._day_low), float(low))
            self._day_close = float(close)

        if not bool(self._cfg.enabled):
            return RegimeRouterSnapshot(
                ready=False,
                climate=None,
                chosen_host=None,
                effective_entry_dir=str(hf_entry_dir) if hf_entry_dir in ("up", "down") else None,
                host_managed=False,
                bull_sovereign_ok=False,
                dwell_days=0,
            )
        if self._active is None:
            return RegimeRouterSnapshot(
                ready=False,
                climate=None,
                chosen_host="hf_host",
                effective_entry_dir=str(hf_entry_dir) if hf_entry_dir in ("up", "down") else None,
                host_managed=False,
                bull_sovereign_ok=False,
                dwell_days=int(self._pending_days),
            )
        chosen_host = str(self._active.chosen_host)
        bull_sovereign_ok = bull_sovereign_entry_ok(
            climate=str(self._active.climate),
            chosen_host=chosen_host,
            fast_features=self._last_fast_features,
            slow_features=self._last_slow_features,
        )
        effective_host = "bull_ma200_v1" if bool(bull_sovereign_ok) and chosen_host == "buyhold" else chosen_host
        policy = host_policy(effective_host)
        if policy.name == "hf_host":
            effective_dir = str(hf_entry_dir) if hf_entry_dir in ("up", "down") else None
        else:
            effective_dir = named_host_target_dir(self._completed_days, policy.name)
        return RegimeRouterSnapshot(
            ready=True,
            climate=str(self._active.climate),
            chosen_host=policy.name,
            effective_entry_dir=str(effective_dir) if effective_dir in ("up", "down") else None,
            host_managed=bool(policy.host_managed),
            bull_sovereign_ok=bool(bull_sovereign_ok),
            dwell_days=int(self._pending_days),
        )


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
