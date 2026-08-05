"""Permission-only freshness and pressure evidence for the crowned XSP target."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


XSP_PRESSURE_ONRAMP_VERSION = (
    "xsp.opening-edge-v3-freshness-pressure-onramp.v2"
)
XSP_PRESSURE_ONRAMP_ACTIONS = ("ADMIT_NOW", "WAIT", "REVOKE")
XSP_PRESSURE_ONRAMP_REFERENCE_SESSIONS = 63
XSP_PRESSURE_ONRAMP_MINIMUM_REFERENCE_SESSIONS = 21


@dataclass(frozen=True)
class XspPressureProfile:
    path_state: str
    volatility_state: str
    observed_horizons: int
    velocity_horizons: int
    aligned_returns: int
    opposed_returns: int
    aligned_slopes: int
    opposed_slopes: int
    aligned_velocities: int
    opposed_velocities: int
    atr_ratio: float | None
    atr_velocity_pct: float | None
    atr_acceleration_pct: float | None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class XspPressureOnrampDecision:
    action: str
    reason: str
    target_direction: str
    wait_invocations: int
    xsp: XspPressureProfile
    spy: XspPressureProfile
    spy_pressure_level: str
    spy_volume_rank: float | None
    spy_true_range_rank: float | None

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": XSP_PRESSURE_ONRAMP_VERSION,
            "action": self.action,
            "reason": self.reason,
            "target_direction": self.target_direction,
            "wait_invocations": self.wait_invocations,
            "xsp": self.xsp.as_payload(),
            "spy": self.spy.as_payload(),
            "spy_pressure_level": self.spy_pressure_level,
            "spy_volume_rank": self.spy_volume_rank,
            "spy_true_range_rank": self.spy_true_range_rank,
            "direction_authority": "opening_edge_v3_crown_only",
            "order_authority": "none",
        }


def causal_rank(value: float, prior: Sequence[float]) -> float | None:
    """Return an expanding, prior-only percentile with finite-value discipline."""

    clean = tuple(float(row) for row in prior if math.isfinite(float(row)))
    if (
        not math.isfinite(float(value))
        or len(clean) < XSP_PRESSURE_ONRAMP_MINIMUM_REFERENCE_SESSIONS
    ):
        return None
    reference = clean[-XSP_PRESSURE_ONRAMP_REFERENCE_SESSIONS:]
    return (1.0 + sum(row <= float(value) for row in reference)) / (
        len(reference) + 1.0
    )


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _profile(
    impulse: Mapping[str, object] | None,
    *,
    target_direction: str,
) -> XspPressureProfile:
    if target_direction not in {"up", "down"}:
        raise ValueError("XSP pressure on-ramp target direction is invalid")
    sign = 1.0 if target_direction == "up" else -1.0
    raw_horizons = impulse.get("horizons") if isinstance(impulse, Mapping) else ()
    horizons = tuple(
        row for row in raw_horizons if isinstance(row, Mapping)
    ) if isinstance(raw_horizons, Sequence) else ()
    returns = tuple(
        sign * value
        for row in horizons
        if (value := _number(row.get("return_pct"))) is not None
    )
    slopes = tuple(
        sign * value
        for row in horizons
        if (value := _number(row.get("slope_pct_per_bar"))) is not None
    )
    velocities = tuple(
        sign * value
        for row in horizons
        if (
            value := _number(row.get("slope_velocity_pct_per_bar"))
        ) is not None
    )
    observed = min(len(returns), len(slopes))
    aligned_returns = sum(value > 0.0 for value in returns)
    opposed_returns = sum(value < 0.0 for value in returns)
    aligned_slopes = sum(value > 0.0 for value in slopes)
    opposed_slopes = sum(value < 0.0 for value in slopes)
    aligned_velocities = sum(value > 0.0 for value in velocities)
    opposed_velocities = sum(value < 0.0 for value in velocities)
    velocity_majority = math.ceil((2.0 * len(velocities)) / 3.0)
    aligned_path = bool(
        observed >= 3
        and aligned_returns == len(returns)
        and aligned_slopes == len(slopes)
    )
    opposed_path = bool(
        observed >= 3
        and opposed_returns == len(returns)
        and opposed_slopes == len(slopes)
    )
    if observed < 3 or len(velocities) < 2:
        path_state = "UNDERWARMED"
    elif aligned_path and aligned_velocities >= velocity_majority:
        path_state = "ALIGNED_ACCELERATING"
    elif aligned_path and opposed_velocities >= velocity_majority:
        path_state = "ALIGNED_DECELERATING"
    elif aligned_path:
        path_state = "ALIGNED_PERSISTENT"
    elif opposed_path and opposed_velocities >= velocity_majority:
        path_state = "OPPOSED_ACCELERATING"
    elif opposed_path and aligned_velocities >= velocity_majority:
        path_state = "OPPOSED_DECELERATING"
    elif opposed_path:
        path_state = "OPPOSED_PERSISTENT"
    else:
        path_state = "MIXED"
    atr_velocity = (
        _number(impulse.get("atr_velocity_pct"))
        if isinstance(impulse, Mapping)
        else None
    )
    atr_acceleration = (
        _number(impulse.get("atr_acceleration_pct"))
        if isinstance(impulse, Mapping)
        else None
    )
    if (
        atr_velocity is not None
        and atr_acceleration is not None
        and atr_velocity > 0.0
        and atr_acceleration > 0.0
    ):
        volatility_state = "EXPANDING"
    elif (
        atr_velocity is not None
        and atr_acceleration is not None
        and atr_velocity < 0.0
        and atr_acceleration < 0.0
    ):
        volatility_state = "CONTRACTING"
    else:
        volatility_state = "TRANSITIONAL"
    return XspPressureProfile(
        path_state=path_state,
        volatility_state=volatility_state,
        observed_horizons=observed,
        velocity_horizons=len(velocities),
        aligned_returns=aligned_returns,
        opposed_returns=opposed_returns,
        aligned_slopes=aligned_slopes,
        opposed_slopes=opposed_slopes,
        aligned_velocities=aligned_velocities,
        opposed_velocities=opposed_velocities,
        atr_ratio=(
            _number(impulse.get("atr_ratio"))
            if isinstance(impulse, Mapping)
            else None
        ),
        atr_velocity_pct=atr_velocity,
        atr_acceleration_pct=atr_acceleration,
    )


def _pressure_level(
    volume_rank: float | None,
    true_range_rank: float | None,
) -> str:
    if volume_rank is None or true_range_rank is None:
        return "UNDERWARMED"
    if min(volume_rank, true_range_rank) >= 0.95:
        return "SHOCK"
    if min(volume_rank, true_range_rank) >= 0.80:
        return "ELEVATED"
    if max(volume_rank, true_range_rank) >= 0.95:
        return "ONE_SIDED"
    return "ORDINARY"


def xsp_pressure_onramp_decision(
    *,
    target_direction: str,
    xsp_impulse: Mapping[str, object] | None,
    spy_impulse: Mapping[str, object] | None,
    spy_volume_rank: float | None,
    spy_true_range_rank: float | None,
    wait_invocations: int = 0,
) -> XspPressureOnrampDecision:
    """Permit timing only; never invent or reverse the crowned direction."""

    if wait_invocations not in {0, 1}:
        raise ValueError("XSP pressure on-ramp permits at most one wait")
    xsp = _profile(xsp_impulse, target_direction=target_direction)
    spy = _profile(spy_impulse, target_direction=target_direction)
    pressure = _pressure_level(spy_volume_rank, spy_true_range_rank)
    opposed = {
        "OPPOSED_ACCELERATING",
        "OPPOSED_PERSISTENT",
        "OPPOSED_DECELERATING",
    }
    xsp_authoritative = bool(
        xsp.path_state == "OPPOSED_ACCELERATING"
        and xsp.volatility_state == "EXPANDING"
    )
    spy_authoritative = bool(
        spy.path_state == "OPPOSED_ACCELERATING"
        and pressure in {"ELEVATED", "SHOCK"}
    )
    if xsp_authoritative and spy.path_state in opposed:
        action, reason = "REVOKE", "dual_authoritative_opposition"
    elif spy_authoritative and xsp.path_state in opposed:
        action, reason = "REVOKE", "pressure_confirmed_opposition"
    elif wait_invocations == 1:
        if xsp.path_state in opposed or spy_authoritative:
            action, reason = "REVOKE", "one_wait_opposition_persisted"
        else:
            action, reason = "ADMIT_NOW", "one_wait_opposition_cleared"
    elif xsp.path_state in opposed or spy_authoritative:
        action, reason = "WAIT", "single_source_or_cresting_opposition"
    else:
        action, reason = "ADMIT_NOW", "no_authoritative_fresh_opposition"
    return XspPressureOnrampDecision(
        action=action,
        reason=reason,
        target_direction=target_direction,
        wait_invocations=wait_invocations,
        xsp=xsp,
        spy=spy,
        spy_pressure_level=pressure,
        spy_volume_rank=spy_volume_rank,
        spy_true_range_rank=spy_true_range_rank,
    )
