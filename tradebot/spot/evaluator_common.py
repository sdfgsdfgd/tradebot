"""Shared types and accessors for spot-signal evaluation."""

from __future__ import annotations

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

    @classmethod
    def from_snapshot(cls, snapshot: object | None) -> "SpotRegimeState":
        canonical = getattr(snapshot, "regime", None)
        if isinstance(canonical, cls):
            return canonical
        return cls(
            label=str(getattr(snapshot, "regime4_state", "") or "") or None,
            owner=str(getattr(snapshot, "regime4_owner", "") or "primary"),
            transition_hot=bool(
                getattr(snapshot, "regime4_transition_hot", False)
            ),
            fast_dir=getattr(snapshot, "regime2_dir", None),
            fast_ready=bool(getattr(snapshot, "regime2_ready", False)),
            hard_dir=getattr(snapshot, "regime2_bear_hard_dir", None),
            hard_ready=bool(
                getattr(snapshot, "regime2_bear_hard_ready", False)
            ),
            hard_release_age_bars=getattr(
                snapshot, "regime2_bear_hard_release_age_bars", None
            ),
        )

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
    regime: SpotRegimeState = SpotRegimeState()

    def regime_state(self) -> SpotRegimeState:
        return self.regime

    @property
    def regime2_dir(self) -> str | None:
        return self.regime.fast_dir

    @property
    def regime2_ready(self) -> bool:
        return self.regime.fast_ready

    @property
    def regime2_bear_hard_dir(self) -> str | None:
        return self.regime.hard_dir

    @property
    def regime2_bear_hard_ready(self) -> bool:
        return self.regime.hard_ready

    @property
    def regime2_bear_hard_release_age_bars(self) -> int | None:
        return self.regime.hard_release_age_bars

    @property
    def regime4_state(self) -> str | None:
        return self.regime.label

    @property
    def regime4_transition_hot(self) -> bool:
        return self.regime.transition_hot

    @property
    def regime4_owner(self) -> str:
        return self.regime.owner

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
