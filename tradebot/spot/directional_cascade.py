"""Versioned state ownership for directional-impulse admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..engines.directional_impulse import (
    DirectionalImpulseHorizon,
    DirectionalImpulseSnapshot,
)
from ..engines.market import xsp_bar_session_label_et, xsp_bar_trading_date
from ..time_utils import NaiveTsModeInput


@dataclass(frozen=True)
class DirectionalCascadePolicy:
    """Frozen GTH state owner used by Opening Edge v2."""

    mode: str = "opening_edge_v2_gth"
    fast_velocity_min: int = 2
    slow_velocity_exact: int = 1
    atr_ratio_min: float = 1.0
    up_opposed_horizon_bars: int = 6
    initial_down_coherence_max: float = 0.75
    down_reaffirm_fast_slope_max: int = 2
    down_reaffirm_slow_slope_max: int = 1
    initial_down_maturation_bars: int = 1
    initial_down_maturation_fast_velocity_min: int = 2

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None,
    ) -> "DirectionalCascadePolicy | None":
        if not isinstance(raw, Mapping):
            return None
        mode = str(raw.get("state_mode", "off") or "off").strip().lower()
        if mode in ("", "off", "none", "disabled"):
            return None
        if mode in ("xsp_gth_down_cascade", "opening_edge_v2_gth"):
            return cls()
        raise ValueError(f"unsupported directional impulse state mode: {mode}")

    def as_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "session": "GTH",
            "fast_velocity_min": self.fast_velocity_min,
            "slow_velocity_exact": self.slow_velocity_exact,
            "atr_ratio_min": self.atr_ratio_min,
            "up_opposed_horizon_bars": self.up_opposed_horizon_bars,
            "initial_down_coherence_max": self.initial_down_coherence_max,
            "down_reaffirm_fast_slope_max": self.down_reaffirm_fast_slope_max,
            "down_reaffirm_slow_slope_max": self.down_reaffirm_slow_slope_max,
            "initial_down_maturation_bars": self.initial_down_maturation_bars,
            "initial_down_maturation_fast_velocity_min": (
                self.initial_down_maturation_fast_velocity_min
            ),
        }


@dataclass(frozen=True)
class DirectionalCascadeDecision:
    direction: str | None
    reason: str
    controls: tuple[str, ...]


@dataclass(frozen=True)
class _PendingDown:
    bar_index: int
    session: str
    close: float


def _rows(
    impulse: DirectionalImpulseSnapshot | None,
) -> dict[int, DirectionalImpulseHorizon]:
    return (
        {int(row.bars): row for row in impulse.horizons}
        if impulse is not None
        else {}
    )


def _aligned_votes(
    rows: Mapping[int, DirectionalImpulseHorizon],
    *,
    direction: str,
    horizons: tuple[int, ...],
    field: str,
) -> int:
    sign = 1.0 if direction == "up" else -1.0
    return sum(
        row is not None
        and (value := getattr(row, field, None)) is not None
        and sign * float(value) > 0.0
        for horizon in horizons
        if (row := rows.get(horizon)) is not None
    )


class DirectionalCascadeEngine:
    """Retain causal direction state while requiring versioned reversal proof."""

    def __init__(self, policy: DirectionalCascadePolicy) -> None:
        self.policy = policy
        self._day: date | None = None
        self._bar_index = -1
        self._incumbent: str | None = None
        self._incumbent_session: str | None = None
        self._pending_down: _PendingDown | None = None

    def _reset(self, day: date | None) -> None:
        self._day = day
        self._bar_index = -1
        self._incumbent = None
        self._incumbent_session = None
        self._pending_down = None

    def _transition_passes(
        self,
        impulse: DirectionalImpulseSnapshot | None,
        direction: str,
    ) -> bool:
        rows = _rows(impulse)
        return bool(
            _aligned_votes(
                rows,
                direction=direction,
                horizons=(1, 3, 6),
                field="slope_velocity_pct_per_bar",
            )
            >= self.policy.fast_velocity_min
            and _aligned_votes(
                rows,
                direction=direction,
                horizons=(12, 24),
                field="slope_velocity_pct_per_bar",
            )
            == self.policy.slow_velocity_exact
            and float(getattr(impulse, "atr_ratio", 0.0) or 0.0)
            >= self.policy.atr_ratio_min
        )

    def _up_is_opposed(
        self,
        impulse: DirectionalImpulseSnapshot | None,
    ) -> bool:
        row = _rows(impulse).get(self.policy.up_opposed_horizon_bars)
        return bool(row is not None and float(row.slope_pct_per_bar) <= 0.0)

    def _down_reaffirm_passes(
        self,
        impulse: DirectionalImpulseSnapshot | None,
    ) -> bool:
        rows = _rows(impulse)
        return bool(
            _aligned_votes(
                rows,
                direction="down",
                horizons=(1, 3, 6),
                field="slope_pct_per_bar",
            )
            <= self.policy.down_reaffirm_fast_slope_max
            and _aligned_votes(
                rows,
                direction="down",
                horizons=(12, 24),
                field="slope_pct_per_bar",
            )
            <= self.policy.down_reaffirm_slow_slope_max
        )

    def _mature_pending_down(
        self,
        *,
        impulse: DirectionalImpulseSnapshot | None,
        close: float,
        session: str,
    ) -> bool:
        pending = self._pending_down
        if pending is None:
            return False
        self._pending_down = None
        rows = _rows(impulse)
        required = (1, 3, 6, 12, 24)
        return bool(
            self._bar_index - pending.bar_index
            == self.policy.initial_down_maturation_bars
            and session == pending.session
            and float(close) < pending.close
            and all(horizon in rows for horizon in required)
            and _aligned_votes(
                rows,
                direction="down",
                horizons=(1, 3, 6),
                field="slope_velocity_pct_per_bar",
            )
            >= self.policy.initial_down_maturation_fast_velocity_min
        )

    def update(
        self,
        *,
        proposed_direction: str | None,
        impulse: DirectionalImpulseSnapshot | None,
        close: float,
        ts: datetime,
        bar_duration: timedelta,
        naive_ts_mode: NaiveTsModeInput,
    ) -> DirectionalCascadeDecision:
        day = xsp_bar_trading_date(
            ts,
            bar_duration=bar_duration,
            naive_ts_mode=naive_ts_mode,
        )
        session = xsp_bar_session_label_et(
            ts,
            bar_duration=bar_duration,
            naive_ts_mode=naive_ts_mode,
        )
        if day != self._day:
            self._reset(day)
        if session != "GTH":
            self._pending_down = None
            return DirectionalCascadeDecision(
                None,
                "outside_gth",
                ("directional_impulse_cascade:idle:outside_gth",),
            )

        self._bar_index += 1
        if self._mature_pending_down(
            impulse=impulse,
            close=close,
            session=session,
        ):
            self._incumbent = "down"
            self._incumbent_session = session
            return DirectionalCascadeDecision(
                "down",
                "initial_down_matured",
                ("directional_impulse_cascade:pass:initial_down_matured",),
            )

        direction = (
            proposed_direction
            if proposed_direction in ("up", "down")
            and self._transition_passes(impulse, proposed_direction)
            else None
        )
        if direction is None:
            reason = (
                "transition"
                if proposed_direction in ("up", "down")
                else "no_proposal"
            )
            return DirectionalCascadeDecision(
                None,
                reason,
                (f"directional_impulse_cascade:block:{reason}",),
            )

        initial = self._incumbent is None
        reversal = self._incumbent is not None and direction != self._incumbent
        cross_session = reversal and session != self._incumbent_session
        authority = (
            "initial"
            if initial
            else "reaffirm"
            if not reversal
            else "cross_session_reversal"
            if cross_session
            else "same_session_reversal"
        )
        allowed = True
        if direction == "up":
            allowed = self._up_is_opposed(impulse)
        elif authority == "initial":
            allowed = (
                float(getattr(impulse, "coherence", 0.0) or 0.0)
                <= self.policy.initial_down_coherence_max
            )
        elif authority == "reaffirm":
            allowed = self._down_reaffirm_passes(impulse)

        if not allowed and direction == "down" and authority == "initial":
            self._pending_down = _PendingDown(
                bar_index=self._bar_index,
                session=session,
                close=float(close),
            )
            return DirectionalCascadeDecision(
                None,
                "initial_down_armed",
                ("directional_impulse_cascade:block:initial_down_armed",),
            )
        if not allowed:
            return DirectionalCascadeDecision(
                None,
                authority,
                (
                    f"directional_impulse_cascade:block:"
                    f"{authority}_{direction}",
                ),
            )

        self._incumbent = direction
        self._incumbent_session = session
        self._pending_down = None
        return DirectionalCascadeDecision(
            direction,
            authority,
            (
                f"directional_impulse_cascade:pass:"
                f"{authority}_{direction}",
            ),
        )
