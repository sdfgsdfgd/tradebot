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


@dataclass(frozen=True)
class SpotRegimeState:
    """Canonical fast/hard regime evidence consumed by entry gates."""

    label: str | None = None
    owner: str = "primary"
    transition_hot: bool = False
    fast_dir: str | None = None
    fast_ready: bool = False
    hard_dir: str | None = None
    hard_ready: bool = False
    hard_release_age_bars: int | None = None

    def entry_context(self) -> dict[str, object]:
        return {
            "regime4_state": self.label,
            "hard_dir": self.hard_dir if self.hard_dir in ("up", "down") else None,
            "release_age_bars": self.hard_release_age_bars,
        }


@dataclass(frozen=True)
class SpotEntryCandidate:
    direction: str | None
    branch: str | None = None
    blocked_by: str | None = None

    @property
    def active(self) -> bool:
        return self.direction in ("up", "down")

    def block(self, gate: str) -> "SpotEntryCandidate":
        return SpotEntryCandidate(direction=None, branch=None, blocked_by=str(gate))


@dataclass(frozen=True)
class SpotSignalSelection:
    """One normalized result from every supported entry-signal family."""

    signal: EmaDecisionSnapshot | None
    candidate: SpotEntryCandidate
    branch_key: str | None = None


@dataclass(frozen=True)
class SpotEntryGateContext:
    bar_ts: datetime
    regime: SpotRegimeState
    shock_dir: str | None = None
    shock_atr_pct: float | None = None
    shock_dir_ret_sum_pct: float | None = None
    shock_drawdown_dist_on_vel_pp: float | None = None
    router: object | None = None


@dataclass(frozen=True)
class SpotGateBand:
    minimum: float | int | None = None
    maximum: float | int | None = None

    def contains(self, value: float | int | None) -> bool:
        return bool(
            value is not None
            and self.minimum is not None
            and self.maximum is not None
            and self.minimum <= value < self.maximum
        )


@dataclass(frozen=True)
class SpotRegimeGatePolicy:
    """Semantic gate configuration decoded from legacy `regime2_*` keys."""

    crash_atr_min: float | None
    crash_block_longs: bool
    transition_hot_atr_min: float | None
    transition_hot_release_age_max: int | None
    crash_prearm_scope: str
    crash_prearm_atr_min: float | None
    crash_prearm_ret_max: float | None
    crash_prearm_branch_a_atr_min: float | None
    crash_prearm_branch_a_ret_max: float | None
    repair_branch_b_block: bool
    repair_branch_b_atr_max: float | None
    repair_branch_b_after_hour: int | None
    upcorridor_branch_a_mid_atr: SpotGateBand
    upcorridor_branch_a_extreme_atr_min: float | None
    upcorridor_branch_a_fresh_age_max: int | None
    upcorridor_branch_a_stale_age_min: int | None
    upcorridor_branch_b_stale_age_min: int | None
    upcorridor_branch_b_flat_low_atr_max: float | None
    upcorridor_branch_b_flat_low_stale_age_min: int | None
    upcorridor_branch_b_flat_atr_max: float | None
    upcorridor_branch_b_flat_ddv_abs_max: float | None
    trenddown_branch_b_release_age: SpotGateBand
    trenddown_branch_b_atr: SpotGateBand
    trenddown_branch_b_ddv: SpotGateBand
    trenddown_branch_b_recovery_atr: SpotGateBand
    trenddown_branch_b_recovery_ddv: SpotGateBand
    continuation_branch_b_release_age: SpotGateBand
    continuation_branch_a_release_age_max: int | None
    continuation_branch_a_atr: SpotGateBand
    continuation_branch_a_ddv_max: float | None


def _bars_input_list(
    values: list[BarLike] | BarSeries[BarLike] | None,
) -> list[BarLike]:
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
    regime: SpotRegimeState | None = None

    def regime_state(self) -> SpotRegimeState:
        if self.regime is not None:
            return self.regime
        return SpotRegimeState(
            label=self.regime4_state,
            owner=str(self.regime4_owner or "primary"),
            transition_hot=bool(self.regime4_transition_hot),
            fast_dir=self.regime2_dir,
            fast_ready=bool(self.regime2_ready),
            hard_dir=self.regime2_bear_hard_dir,
            hard_ready=bool(self.regime2_bear_hard_ready),
            hard_release_age_bars=self.regime2_bear_hard_release_age_bars,
        )

    def entry_context(self) -> dict[str, object]:
        return {
            "branch": self.entry_branch if self.entry_branch in ("a", "b") else None,
            "shock_dir": self.shock_dir if self.shock_dir in ("up", "down") else None,
            **self.regime_state().entry_context(),
        }

    def lifecycle_inputs(self) -> dict[str, object]:
        risk = self.risk
        return {
            "signal_entry_dir": self.entry_dir
            if self.entry_dir in ("up", "down")
            else None,
            "shock_atr_pct": float(self.shock_atr_pct)
            if self.shock_atr_pct is not None
            else None,
            "shock_atr_vel_pct": float(self.shock_atr_vel_pct)
            if self.shock_atr_vel_pct is not None
            else None,
            "shock_atr_accel_pct": float(self.shock_atr_accel_pct)
            if self.shock_atr_accel_pct is not None
            else None,
            "tr_ratio": float(self.ratsv_tr_ratio)
            if self.ratsv_tr_ratio is not None
            else None,
            "tr_median_pct": (
                float(risk.tr_median_pct)
                if risk is not None and risk.tr_median_pct is not None
                else None
            ),
            "slope_med_pct": (
                float(self.ratsv_fast_slope_med_pct)
                if self.ratsv_fast_slope_med_pct is not None
                else None
            ),
            "slope_vel_pct": (
                float(self.ratsv_fast_slope_vel_pct)
                if self.ratsv_fast_slope_vel_pct is not None
                else None
            ),
            "slope_med_slow_pct": (
                float(self.ratsv_slow_slope_med_pct)
                if self.ratsv_slow_slope_med_pct is not None
                else None
            ),
            "slope_vel_slow_pct": (
                float(self.ratsv_slow_slope_vel_pct)
                if self.ratsv_slow_slope_vel_pct is not None
                else None
            ),
        }

    def lifecycle_trace(self) -> dict[str, object]:
        values = self.lifecycle_inputs()
        return {
            "shock_atr_pct": values["shock_atr_pct"],
            "shock_atr_vel_pct": values["shock_atr_vel_pct"],
            "shock_atr_accel_pct": values["shock_atr_accel_pct"],
            "ratsv_tr_ratio": values["tr_ratio"],
            "risk_tr_median_pct": values["tr_median_pct"],
            "ratsv_fast_slope_med_pct": values["slope_med_pct"],
            "ratsv_fast_slope_vel_pct": values["slope_vel_pct"],
            "ratsv_slow_slope_med_pct": values["slope_med_slow_pct"],
            "ratsv_slow_slope_vel_pct": values["slope_vel_slow_pct"],
            "ratsv_slope_vel_consistency": (
                float(self.ratsv_slope_vel_consistency)
                if self.ratsv_slope_vel_consistency is not None
                else None
            ),
        }


# endregion
