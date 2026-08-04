"""Pure levelled MCL shock, velocity-crest, and continuation state owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal


MCL_SHOCK_CREST_VERSION = "mcl.shock-crest-continuation-relay.v2"
MCL_SHOCK_CREST_AUTHORITY = "signal_state_only_no_orders_no_capital"
MclShockLevel = Literal[
    "NORMAL_UNDER_5X",
    "ELEVATED_5_TO_10X",
    "MAJOR_PROTECT_10_TO_12X",
    "TRADEABLE_SHOCK_12_TO_20X",
    "REGIME_20X_PLUS",
]
MclShockPhase = Literal[
    "STATE",
    "SHOCK_LATCHED",
    "CREST_CONFIRMED",
    "CONTINUATION",
    "ROTATION_ARMED",
    "ROTATION_EXIT",
    "REVERSAL_ELIGIBLE",
    "NORMALIZED",
]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("MCL shock timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"MCL shock {name} must be finite")
    return result


def _sign(value: float) -> int | None:
    return 1 if value > 0.0 else -1 if value < 0.0 else None


@dataclass(frozen=True, slots=True)
class MclShockCrestPolicy:
    """One preregistered categorical shock and execution-urgency law."""

    elevated_multiple: float = 5.0
    major_multiple: float = 10.0
    tradeable_multiple: float = 12.0
    regime_multiple: float = 20.0
    crest_lower_observations: int = 3
    crest_min_seconds: float = 3.0
    rotation_seconds: float = 5.0
    reversal_seconds: float = 15.0
    normalization_seconds: float = 60.0
    tradeable_phase_speed_multiplier: float = 1.5
    regime_phase_speed_multiplier: float = 2.0
    major_exit_patience_multiplier: float = 2.0
    tradeable_exit_patience_multiplier: float = 3.0
    regime_exit_patience_multiplier: float = 4.0

    def __post_init__(self) -> None:
        levels = (
            self.elevated_multiple,
            self.major_multiple,
            self.tradeable_multiple,
            self.regime_multiple,
        )
        if any(not math.isfinite(value) for value in levels) or not (
            0.0 < levels[0] < levels[1] <= levels[2] < levels[3]
        ):
            raise ValueError("MCL shock volume levels must increase")
        if self.crest_lower_observations < 2:
            raise ValueError("MCL shock crest requires repeated observations")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.crest_min_seconds,
                self.rotation_seconds,
                self.reversal_seconds,
                self.normalization_seconds,
            )
        ):
            raise ValueError("MCL shock persistence clocks must be positive")
        if not 1.0 <= self.tradeable_phase_speed_multiplier <= 4.0 or not (
            self.tradeable_phase_speed_multiplier
            <= self.regime_phase_speed_multiplier
            <= 4.0
        ):
            raise ValueError("MCL shock execution multipliers are invalid")
        if not (
            1.0 <= self.major_exit_patience_multiplier
            <= self.tradeable_exit_patience_multiplier
            <= self.regime_exit_patience_multiplier
            <= 8.0
        ):
            raise ValueError("MCL shock exit-patience multipliers are invalid")

    def level(self, multiple: float) -> MclShockLevel:
        value = _finite(multiple, name="volume multiple")
        if value < 0.0:
            raise ValueError("MCL shock volume multiple cannot be negative")
        return (
            "REGIME_20X_PLUS"
            if value >= self.regime_multiple
            else "TRADEABLE_SHOCK_12_TO_20X"
            if value >= self.tradeable_multiple
            else "MAJOR_PROTECT_10_TO_12X"
            if value >= self.major_multiple
            else "ELEVATED_5_TO_10X"
            if value >= self.elevated_multiple
            else "NORMAL_UNDER_5X"
        )


@dataclass(frozen=True, slots=True)
class MclShockBookEvidence:
    """One book's causal seconds and completed-fifteen-minute evidence."""

    velocity_5s: float
    velocity_15s: float
    velocity_60s: float
    slope_15m: float
    velocity_15m: float
    acceleration_15m: float
    signed_flow_15s: float
    volume_velocity_5s: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _finite(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class MclShockObservation:
    """One valid canonical second joined to causal slow context."""

    observed_at_utc: datetime
    contract_key: str
    mcl_microprice: float
    volume_multiple: float
    cl: MclShockBookEvidence
    mcl: MclShockBookEvidence
    spread_eligible: bool = True
    fresh_top: bool = True

    def __post_init__(self) -> None:
        _utc(self.observed_at_utc)
        if not self.contract_key.strip():
            raise ValueError("MCL shock observation requires a contract key")
        if _finite(self.mcl_microprice, name="microprice") <= 0.0:
            raise ValueError("MCL shock microprice must be positive")
        if _finite(self.volume_multiple, name="volume multiple") < 0.0:
            raise ValueError("MCL shock volume multiple cannot be negative")


@dataclass(frozen=True, slots=True)
class MclShockDecision:
    """One pure shock transition; execution remains a downstream owner."""

    observed_at_utc: datetime
    phase: MclShockPhase
    current_level: MclShockLevel
    latched_level: MclShockLevel
    shock_direction: int | None
    peak_speed_ticks_per_second: float | None
    crest_at_utc: datetime | None
    continuation_direction: int | None
    countertrend_inversion_eligible: bool
    opposing_position_must_flatten: bool
    phase_speed_multiplier: float
    exit_patience_multiplier: float
    fast_rotation_flatten_authority: bool
    reason: str

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": MCL_SHOCK_CREST_VERSION,
            "authority": MCL_SHOCK_CREST_AUTHORITY,
            "observed_at_utc": _utc(self.observed_at_utc).isoformat(),
            "phase": self.phase,
            "current_level": self.current_level,
            "latched_level": self.latched_level,
            "shock_direction": self.shock_direction,
            "peak_speed_ticks_per_second": self.peak_speed_ticks_per_second,
            "crest_at_utc": (
                _utc(self.crest_at_utc).isoformat()
                if self.crest_at_utc is not None
                else None
            ),
            "continuation_direction": self.continuation_direction,
            "countertrend_inversion_eligible": bool(
                self.countertrend_inversion_eligible
            ),
            "opposing_position_must_flatten": bool(
                self.opposing_position_must_flatten
            ),
            "phase_speed_multiplier": float(self.phase_speed_multiplier),
            "exit_patience_multiplier": float(self.exit_patience_multiplier),
            "fast_rotation_flatten_authority": bool(
                self.fast_rotation_flatten_authority
            ),
            "reason": self.reason,
            "submitted_orders": 0,
        }


_LEVEL_ORDER: dict[MclShockLevel, int] = {
    "NORMAL_UNDER_5X": 0,
    "ELEVATED_5_TO_10X": 1,
    "MAJOR_PROTECT_10_TO_12X": 2,
    "TRADEABLE_SHOCK_12_TO_20X": 3,
    "REGIME_20X_PLUS": 4,
}


class MclShockCrestEngine:
    """Own level hysteresis, a causal velocity crest, and post-crest actions."""

    def __init__(self, policy: MclShockCrestPolicy | None = None) -> None:
        self.policy = policy or MclShockCrestPolicy()
        self._contract_key: str | None = None
        self._last_at: datetime | None = None
        self._latched_level: MclShockLevel = "NORMAL_UNDER_5X"
        self._direction: int | None = None
        self._peak_speed: float | None = None
        self._peak_at: datetime | None = None
        self._below_peak_started: datetime | None = None
        self._below_peak_observations = 0
        self._crest_at: datetime | None = None
        self._previous_speed: float | None = None
        self._continuation = False
        self._shock_extreme: float | None = None
        self._rotation_started: datetime | None = None
        self._rotation_observations = 0
        self._rotation_emitted = False
        self._reversal_started: datetime | None = None
        self._reversal_observations = 0
        self._reversal_emitted = False
        self._normal_started: datetime | None = None

    def reset(self, *, contract_key: str | None = None) -> None:
        policy = self.policy
        self.__init__(policy)
        self._contract_key = contract_key

    @staticmethod
    def _joint_direction(
        observation: MclShockObservation,
        *,
        include_slow_acceleration: bool,
        include_flow: bool,
    ) -> int | None:
        values = [
            observation.cl.velocity_5s,
            observation.mcl.velocity_5s,
            observation.cl.velocity_15s,
            observation.mcl.velocity_15s,
            observation.cl.velocity_60s,
            observation.mcl.velocity_60s,
            observation.cl.velocity_15m,
            observation.mcl.velocity_15m,
        ]
        if include_slow_acceleration:
            values.extend(
                (
                    observation.cl.acceleration_15m,
                    observation.mcl.acceleration_15m,
                )
            )
        if include_flow:
            values.extend(
                (
                    observation.cl.signed_flow_15s,
                    observation.mcl.signed_flow_15s,
                )
            )
        signs = {_sign(value) for value in values}
        return signs.pop() if len(signs) == 1 and None not in signs else None

    @staticmethod
    def _speed(observation: MclShockObservation, direction: int) -> float | None:
        values = (
            direction * observation.cl.velocity_5s,
            direction * observation.mcl.velocity_5s,
        )
        return min(values) if min(values) > 0.0 else None

    def _phase_speed(self) -> float:
        order = _LEVEL_ORDER[self._latched_level]
        return (
            self.policy.regime_phase_speed_multiplier
            if order >= _LEVEL_ORDER["REGIME_20X_PLUS"]
            else self.policy.tradeable_phase_speed_multiplier
            if order >= _LEVEL_ORDER["TRADEABLE_SHOCK_12_TO_20X"]
            else 1.0
        )

    def _exit_patience(self) -> float:
        order = _LEVEL_ORDER[self._latched_level]
        return (
            self.policy.regime_exit_patience_multiplier
            if order >= _LEVEL_ORDER["REGIME_20X_PLUS"]
            else self.policy.tradeable_exit_patience_multiplier
            if order >= _LEVEL_ORDER["TRADEABLE_SHOCK_12_TO_20X"]
            else self.policy.major_exit_patience_multiplier
            if order >= _LEVEL_ORDER["MAJOR_PROTECT_10_TO_12X"]
            else 1.0
        )

    def update(self, observation: MclShockObservation) -> MclShockDecision:
        at = _utc(observation.observed_at_utc)
        if self._last_at is not None and at <= self._last_at:
            raise ValueError("MCL shock observations must increase")
        if self._contract_key not in (None, observation.contract_key):
            self.reset(contract_key=observation.contract_key)
        self._contract_key = observation.contract_key
        self._last_at = at
        current_level = self.policy.level(observation.volume_multiple)
        previous_level = self._latched_level
        if _LEVEL_ORDER[current_level] > _LEVEL_ORDER[self._latched_level]:
            self._latched_level = current_level

        aligned = self._joint_direction(
            observation,
            include_slow_acceleration=False,
            include_flow=False,
        )
        if (
            self._direction is None
            and aligned in (-1, 1)
            and _LEVEL_ORDER[self._latched_level]
            >= _LEVEL_ORDER["ELEVATED_5_TO_10X"]
        ):
            self._direction = aligned
            self._shock_extreme = observation.mcl_microprice

        phase: MclShockPhase = "STATE"
        reason = "observed_without_transition"
        continuation_direction: int | None = None
        countertrend = False
        direction = self._direction
        speed = self._speed(observation, direction) if direction is not None else None

        if _LEVEL_ORDER[self._latched_level] > _LEVEL_ORDER[previous_level]:
            phase = "SHOCK_LATCHED"
            reason = f"latched_{self._latched_level.lower()}"

        if direction is not None:
            prior_extreme = self._shock_extreme
            new_extreme = (
                prior_extreme is None
                or direction * observation.mcl_microprice
                > direction * prior_extreme
            )
            if new_extreme:
                self._shock_extreme = observation.mcl_microprice

            if (
                _LEVEL_ORDER[self._latched_level]
                >= _LEVEL_ORDER["MAJOR_PROTECT_10_TO_12X"]
                and speed is not None
            ):
                if self._peak_speed is None or speed > self._peak_speed:
                    self._peak_speed = speed
                    self._peak_at = at
                    self._below_peak_started = None
                    self._below_peak_observations = 0
                elif self._crest_at is None and speed < self._peak_speed:
                    if self._below_peak_started is None:
                        self._below_peak_started = at
                    self._below_peak_observations += 1
                    elapsed = (at - self._peak_at).total_seconds() if self._peak_at else 0.0
                    if (
                        self._below_peak_observations
                        >= self.policy.crest_lower_observations
                        and elapsed >= self.policy.crest_min_seconds
                    ):
                        self._crest_at = at
                        phase = "CREST_CONFIRMED"
                        reason = "running_fast_velocity_peak_causally_confirmed"

            fully_aligned = self._joint_direction(
                observation,
                include_slow_acceleration=True,
                include_flow=True,
            )
            if (
                not self._continuation
                and self._crest_at is not None
                and _LEVEL_ORDER[self._latched_level]
                >= _LEVEL_ORDER["TRADEABLE_SHOCK_12_TO_20X"]
                and fully_aligned == direction
                and speed is not None
                and self._previous_speed is not None
                and speed > self._previous_speed
                and observation.spread_eligible
                and observation.fresh_top
            ):
                self._continuation = True
                continuation_direction = direction
                phase = "CONTINUATION"
                reason = "post_crest_multiscale_reacceleration"

            opposite = -direction
            opposite_fast = self._joint_direction(
                observation,
                include_slow_acceleration=False,
                include_flow=False,
            ) == opposite
            if self._continuation and opposite_fast and not new_extreme:
                if self._rotation_started is None:
                    self._rotation_started = at
                    self._rotation_observations = 0
                self._rotation_observations += 1
                if (
                    not self._rotation_emitted
                    and self._rotation_observations >= 4
                    and (at - self._rotation_started).total_seconds()
                    >= self.policy.rotation_seconds
                    * (
                        1.0
                        if self._latched_level == "REGIME_20X_PLUS"
                        else self._exit_patience()
                    )
                ):
                    self._rotation_emitted = True
                    if self._latched_level == "REGIME_20X_PLUS":
                        phase = "ROTATION_ARMED"
                        reason = "regime_shock_fast_rotation_arms_protection_only"
                    else:
                        phase = "ROTATION_EXIT"
                        reason = "level_adjusted_persistent_opposite_velocity"
            else:
                self._rotation_started = None
                self._rotation_observations = 0

            reversal = (
                opposite_fast
                and self._joint_direction(
                    observation,
                    include_slow_acceleration=True,
                    include_flow=True,
                )
                == opposite
                and observation.cl.volume_velocity_5s <= 0.0
                and observation.mcl.volume_velocity_5s <= 0.0
                and not new_extreme
            )
            if reversal:
                if self._reversal_started is None:
                    self._reversal_started = at
                    self._reversal_observations = 0
                self._reversal_observations += 1
                if (
                    not self._reversal_emitted
                    and self._reversal_observations >= 4
                    and (at - self._reversal_started).total_seconds()
                    >= self.policy.reversal_seconds
                ):
                    self._reversal_emitted = True
                    countertrend = True
                    phase = "REVERSAL_ELIGIBLE"
                    reason = "persistent_multiscale_opposite_rotation"
            else:
                self._reversal_started = None
                self._reversal_observations = 0

            normalized = (
                self._latched_level != "REGIME_20X_PLUS"
                and current_level == "NORMAL_UNDER_5X"
                and aligned != direction
                and _sign(observation.cl.velocity_15m) != direction
                and _sign(observation.mcl.velocity_15m) != direction
            )
            if normalized:
                self._normal_started = self._normal_started or at
                if (
                    at - self._normal_started
                ).total_seconds() >= (
                    self.policy.normalization_seconds * self._exit_patience()
                ):
                    phase = "NORMALIZED"
                    reason = "persistent_volume_and_velocity_normalization"
            else:
                self._normal_started = None

        self._previous_speed = speed
        decision = MclShockDecision(
            observed_at_utc=at,
            phase=phase,
            current_level=current_level,
            latched_level=self._latched_level,
            shock_direction=direction,
            peak_speed_ticks_per_second=self._peak_speed,
            crest_at_utc=self._crest_at,
            continuation_direction=continuation_direction,
            countertrend_inversion_eligible=countertrend,
            opposing_position_must_flatten=(
                direction is not None
                and _LEVEL_ORDER[self._latched_level]
                >= _LEVEL_ORDER["MAJOR_PROTECT_10_TO_12X"]
            ),
            phase_speed_multiplier=self._phase_speed(),
            exit_patience_multiplier=self._exit_patience(),
            fast_rotation_flatten_authority=(
                self._latched_level != "REGIME_20X_PLUS"
            ),
            reason=reason,
        )
        if phase == "NORMALIZED":
            self.reset(contract_key=observation.contract_key)
        return decision
