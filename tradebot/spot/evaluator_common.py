"""Shared types and accessors for spot-signal evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..chart_data.series import BarSeries, bars_list
from ..engines.risk import RiskOverlaySnapshot
from ..engines.signals import EmaDecisionSnapshot

# region Protocols / Helpers
class BarLike(Protocol):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _bars_input_list(values: list[BarLike] | BarSeries[BarLike] | None) -> list[BarLike]:
    if values is None:
        return []
    return bars_list(values)

def _get(obj: Mapping[str, object] | object | None, key: str, default: object = None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
# endregion


# region Models
@dataclass(frozen=True)
class SpotSignalSnapshot:
    bar_ts: datetime
    close: float
    signal: EmaDecisionSnapshot
    bars_in_day: int
    rv: float | None
    volume: float | None
    volume_ema: float | None
    volume_ema_ready: bool
    shock: bool | None
    shock_dir: str | None
    shock_detector: str | None
    shock_direction_source_effective: str | None
    shock_scale_detector: str | None
    shock_dir_ret_sum_pct: float | None
    shock_atr_pct: float | None
    shock_drawdown_pct: float | None
    shock_drawdown_on_pct: float | None
    shock_drawdown_off_pct: float | None
    shock_drawdown_dist_on_pct: float | None
    shock_drawdown_dist_on_vel_pp: float | None
    shock_drawdown_dist_on_accel_pp: float | None
    shock_prearm_down_streak_bars: int | None
    shock_drawdown_dist_off_pct: float | None
    shock_scale_drawdown_pct: float | None
    shock_peak_close: float | None
    shock_dir_down_streak_bars: int | None
    shock_dir_up_streak_bars: int | None
    risk: RiskOverlaySnapshot | None
    atr: float | None
    regime2_dir: str | None
    regime2_ready: bool
    regime2_bear_hard_dir: str | None
    regime2_bear_hard_ready: bool
    regime2_bear_hard_release_age_bars: int | None
    regime4_state: str | None
    regime4_transition_hot: bool
    regime4_owner: str | None
    or_high: float | None
    or_low: float | None
    or_ready: bool
    entry_dir: str | None = None
    entry_branch: str | None = None
    ratsv_side_rank: float | None = None
    ratsv_tr_ratio: float | None = None
    ratsv_fast_slope_pct: float | None = None
    ratsv_fast_slope_med_pct: float | None = None
    ratsv_fast_slope_vel_pct: float | None = None
    ratsv_slow_slope_med_pct: float | None = None
    ratsv_slow_slope_vel_pct: float | None = None
    ratsv_slope_vel_consistency: float | None = None
    ratsv_cross_age_bars: int | None = None
    shock_atr_vel_pct: float | None = None
    shock_atr_accel_pct: float | None = None
    shock_ramp: dict[str, object] | None = None
    regime_router_ready: bool = False
    regime_router_climate: str | None = None
    regime_router_host: str | None = None
    regime_router_entry_dir: str | None = None
    regime_router_host_managed: bool = False
    regime_router_bull_sovereign_ok: bool = False
    regime_router_dwell_days: int = 0
    regime_router_crash_ret: float | None = None
    regime_router_crash_maxdd: float | None = None
    regime_router_crash_rv: float | None = None
    regime_router_fast_ret: float | None = None
    regime_router_slow_ret: float | None = None
# endregion
