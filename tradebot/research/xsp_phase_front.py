"""Strictly causal multiscale XSP phase-front morphology."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math


XSP_PHASE_FRONT_VERSION = "xsp.multiscale-phase-front.v1"
XSP_PHASE_FRONT_AUTHORITY = (
    "causal_morphology_only_no_classifier_no_permission_no_orders_no_capital"
)
XSP_PHASE_FRONT_TIERS = ("micro", "fast", "structural", "slow")
XSP_PHASE_FRONT_TRANSPORT = (
    "UNAVAILABLE",
    "PARTIAL",
    "FULL_ACCEPTANCE",
    "REFUSAL",
)


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


@dataclass(frozen=True)
class XspPhaseFrontClock:
    name: str
    horizon_seconds: int
    tier: str
    slope: float
    velocity: float
    acceleration: float | None = None
    jerk: float | None = None

    def __post_init__(self) -> None:
        values = (self.slope, self.velocity)
        if (
            not self.name
            or self.horizon_seconds <= 0
            or self.tier not in XSP_PHASE_FRONT_TIERS
            or not all(math.isfinite(value) for value in values)
            or (
                self.acceleration is not None
                and not math.isfinite(self.acceleration)
            )
            or (self.jerk is not None and not math.isfinite(self.jerk))
        ):
            raise ValueError("XSP phase-front clock is malformed")

    def as_payload(self, direction: int | None = None) -> dict[str, object]:
        eta = None
        if (
            direction is not None
            and direction * self.slope < 0.0
            and direction * self.velocity > 0.0
        ):
            eta = abs(float(self.slope) / float(self.velocity))
        return {
            "name": self.name,
            "horizon_seconds": int(self.horizon_seconds),
            "tier": self.tier,
            "slope": float(self.slope),
            "velocity": float(self.velocity),
            "acceleration": self.acceleration,
            "jerk": self.jerk,
            "slope_sign": _sign(self.slope),
            "velocity_sign": _sign(self.velocity),
            "eta": eta,
        }


@dataclass(frozen=True)
class XspPhaseFrontObservation:
    observed_at_utc: datetime
    session_key: str
    clocks: tuple[XspPhaseFrontClock, ...]
    transport: str = "UNAVAILABLE"
    shock_direction: int | None = None
    contiguous: bool = True

    def __post_init__(self) -> None:
        if (
            self.observed_at_utc.tzinfo is None
            or not self.session_key
            or not self.clocks
            or self.transport not in XSP_PHASE_FRONT_TRANSPORT
            or self.shock_direction not in {None, -1, 1}
            or len({clock.name for clock in self.clocks}) != len(self.clocks)
            or tuple(clock.horizon_seconds for clock in self.clocks)
            != tuple(sorted(clock.horizon_seconds for clock in self.clocks))
        ):
            raise ValueError("XSP phase-front observation is malformed")


@dataclass(frozen=True)
class XspPhaseFrontSnapshot:
    observed_at_utc: datetime
    session_key: str
    phase: str
    incumbent_direction: int | None
    candidate_direction: int | None
    slope_breadth: int
    velocity_breadth: int
    fast_slope_breadth: int
    fast_velocity_breadth: int
    structural_slope_breadth: int
    structural_velocity_breadth: int
    eta_contracting: tuple[str, ...]
    velocity_arrivals: tuple[tuple[str, datetime], ...]
    slope_arrivals: tuple[tuple[str, datetime], ...]
    velocity_ordered: bool
    slope_ordered: bool
    recoil_seen: bool
    transport: str
    clocks: tuple[XspPhaseFrontClock, ...]
    reason: str

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": XSP_PHASE_FRONT_VERSION,
            "authority": XSP_PHASE_FRONT_AUTHORITY,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "session_key": self.session_key,
            "phase": self.phase,
            "incumbent_direction": (
                "up"
                if self.incumbent_direction == 1
                else "down"
                if self.incumbent_direction == -1
                else None
            ),
            "candidate_direction": (
                "up"
                if self.candidate_direction == 1
                else "down"
                if self.candidate_direction == -1
                else None
            ),
            "slope_breadth": int(self.slope_breadth),
            "velocity_breadth": int(self.velocity_breadth),
            "fast_slope_breadth": int(self.fast_slope_breadth),
            "fast_velocity_breadth": int(self.fast_velocity_breadth),
            "structural_slope_breadth": int(self.structural_slope_breadth),
            "structural_velocity_breadth": int(self.structural_velocity_breadth),
            "eta_contracting": list(self.eta_contracting),
            "velocity_arrivals": {
                name: observed.isoformat()
                for name, observed in self.velocity_arrivals
            },
            "slope_arrivals": {
                name: observed.isoformat() for name, observed in self.slope_arrivals
            },
            "velocity_ordered": bool(self.velocity_ordered),
            "slope_ordered": bool(self.slope_ordered),
            "recoil_seen": bool(self.recoil_seen),
            "transport": self.transport,
            "clocks": [
                clock.as_payload(self.candidate_direction) for clock in self.clocks
            ],
            "reason": self.reason,
            "classifier": "none",
            "permission": "none",
            "outcomes": None,
            "submitted_orders": 0,
        }


class XspPhaseFrontEngine:
    """Track sign, order, contraction, recoil, and transport without outcomes."""

    def __init__(self, *, persistence_observations: int = 2) -> None:
        if persistence_observations < 1:
            raise ValueError("XSP phase-front persistence must be positive")
        self._persistence = int(persistence_observations)
        self._session_key: str | None = None
        self._last_at: datetime | None = None
        self._incumbent: int | None = None
        self._candidate: int | None = None
        self._phase = "UNDERWARMED"
        self._velocity_streaks: dict[str, int] = {}
        self._velocity_arrivals: dict[str, datetime] = {}
        self._slope_arrivals: dict[str, datetime] = {}
        self._prior_eta: dict[str, float] = {}
        self._recoil_seen = False
        self._reacceleration_seen = False

    def _reset(self, session_key: str) -> None:
        self._session_key = session_key
        self._last_at = None
        self._incumbent = None
        self._clear_candidate()
        self._phase = "UNDERWARMED"

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._velocity_streaks.clear()
        self._velocity_arrivals.clear()
        self._slope_arrivals.clear()
        self._prior_eta.clear()
        self._recoil_seen = False
        self._reacceleration_seen = False

    @staticmethod
    def _majority(values: tuple[int, ...]) -> int | None:
        up = sum(value > 0 for value in values)
        down = sum(value < 0 for value in values)
        return 1 if up > down else -1 if down > up else None

    @staticmethod
    def _ordered(
        clocks: tuple[XspPhaseFrontClock, ...],
        arrivals: dict[str, datetime],
    ) -> bool:
        observed = [
            arrivals[clock.name] for clock in clocks if clock.name in arrivals
        ]
        return all(right >= left for left, right in zip(observed, observed[1:]))

    def _breadth(
        self,
        clocks: tuple[XspPhaseFrontClock, ...],
        direction: int,
    ) -> tuple[int, int, int, int, int, int]:
        fast = tuple(clock for clock in clocks if clock.tier in {"micro", "fast"})
        structural = tuple(
            clock for clock in clocks if clock.tier in {"structural", "slow"}
        )
        return (
            sum(direction * clock.slope > 0.0 for clock in clocks),
            sum(direction * clock.velocity > 0.0 for clock in clocks),
            sum(direction * clock.slope > 0.0 for clock in fast),
            sum(direction * clock.velocity > 0.0 for clock in fast),
            sum(direction * clock.slope > 0.0 for clock in structural),
            sum(direction * clock.velocity > 0.0 for clock in structural),
        )

    def _snapshot(
        self,
        observation: XspPhaseFrontObservation,
        *,
        phase: str,
        direction: int | None,
        eta_contracting: tuple[str, ...] = (),
        reason: str,
    ) -> XspPhaseFrontSnapshot:
        breadth = (
            self._breadth(observation.clocks, direction)
            if direction is not None
            else (0, 0, 0, 0, 0, 0)
        )
        return XspPhaseFrontSnapshot(
            observed_at_utc=observation.observed_at_utc,
            session_key=observation.session_key,
            phase=phase,
            incumbent_direction=self._incumbent,
            candidate_direction=direction,
            slope_breadth=breadth[0],
            velocity_breadth=breadth[1],
            fast_slope_breadth=breadth[2],
            fast_velocity_breadth=breadth[3],
            structural_slope_breadth=breadth[4],
            structural_velocity_breadth=breadth[5],
            eta_contracting=eta_contracting,
            velocity_arrivals=tuple(
                (clock.name, self._velocity_arrivals[clock.name])
                for clock in observation.clocks
                if clock.name in self._velocity_arrivals
            ),
            slope_arrivals=tuple(
                (clock.name, self._slope_arrivals[clock.name])
                for clock in observation.clocks
                if clock.name in self._slope_arrivals
            ),
            velocity_ordered=self._ordered(
                observation.clocks, self._velocity_arrivals
            ),
            slope_ordered=self._ordered(observation.clocks, self._slope_arrivals),
            recoil_seen=self._recoil_seen,
            transport=observation.transport,
            clocks=observation.clocks,
            reason=reason,
        )

    def update(self, observation: XspPhaseFrontObservation) -> XspPhaseFrontSnapshot:
        if observation.session_key != self._session_key:
            self._reset(observation.session_key)
        if self._last_at is not None and observation.observed_at_utc <= self._last_at:
            raise ValueError("XSP phase-front timestamps must increase")
        self._last_at = observation.observed_at_utc
        if not observation.contiguous:
            self._reset(observation.session_key)
            self._last_at = observation.observed_at_utc
            return self._snapshot(
                observation,
                phase="UNDERWARM_OR_GAP",
                direction=None,
                reason="required causal clock is discontinuous",
            )

        slow = tuple(
            _sign(clock.slope)
            for clock in observation.clocks
            if clock.tier in {"structural", "slow"}
        )
        if self._incumbent is None:
            self._incumbent = self._majority(slow)
            self._phase = "INCUMBENT" if self._incumbent is not None else "UNDERWARMED"
            return self._snapshot(
                observation,
                phase=self._phase,
                direction=None,
                reason=(
                    "structural slope majority owns the initial direction"
                    if self._incumbent is not None
                    else "structural slope direction is unresolved"
                ),
            )

        if (
            observation.shock_direction in {-1, 1}
            and observation.shock_direction != self._incumbent
            and observation.transport == "FULL_ACCEPTANCE"
        ):
            self._candidate = int(observation.shock_direction)
            self._phase = "HANDOFF"
            snapshot = self._snapshot(
                observation,
                phase="HANDOFF",
                direction=self._candidate,
                reason="independent opposite shock has full transport authority",
            )
            self._incumbent = self._candidate
            self._clear_candidate()
            return snapshot

        candidate = self._candidate or -self._incumbent
        fast = tuple(
            clock for clock in observation.clocks if clock.tier in {"micro", "fast"}
        )
        fast_velocity = sum(candidate * clock.velocity > 0.0 for clock in fast)
        if self._candidate is None:
            if len(fast) < 2 or fast_velocity < 2:
                return self._snapshot(
                    observation,
                    phase="INCUMBENT",
                    direction=None,
                    reason="new-direction fast velocity has not formed a repair spark",
                )
            self._candidate = candidate
            self._phase = "REPAIR_SPARK"

        direction = int(self._candidate)
        for clock in observation.clocks:
            aligned = direction * clock.velocity > 0.0
            self._velocity_streaks[clock.name] = (
                self._velocity_streaks.get(clock.name, 0) + 1 if aligned else 0
            )
            if (
                self._velocity_streaks[clock.name] >= self._persistence
                and clock.name not in self._velocity_arrivals
            ):
                self._velocity_arrivals[clock.name] = observation.observed_at_utc
            if direction * clock.slope > 0.0 and clock.name not in self._slope_arrivals:
                self._slope_arrivals[clock.name] = observation.observed_at_utc

        breadth = self._breadth(observation.clocks, direction)
        eta = {
            clock.name: abs(clock.slope / clock.velocity)
            for clock in observation.clocks
            if direction * clock.slope < 0.0 and direction * clock.velocity > 0.0
        }
        contracting = tuple(
            clock.name
            for clock in observation.clocks
            if clock.name in eta
            and clock.name in self._prior_eta
            and eta[clock.name] < self._prior_eta[clock.name]
        )
        self._prior_eta = eta
        velocity_ordered = self._ordered(
            observation.clocks, self._velocity_arrivals
        )
        slope_ordered = self._ordered(observation.clocks, self._slope_arrivals)
        fast_count = len(fast)

        if observation.transport == "REFUSAL" and breadth[2] < 2:
            phase, reason = "RELAPSED", "cross-instrument transport refused before fast slope ownership"
        elif breadth[3] < 2 and breadth[2] < 2:
            phase, reason = "RELAPSED", "fast velocity relapsed before fast slope propagation"
        elif breadth[2] >= 2 and breadth[3] < 2:
            self._recoil_seen = True
            phase, reason = "RECOIL_TEST", "fast velocity recoiled while fast slope ownership survived"
        elif self._recoil_seen and breadth[2] >= 2 and breadth[3] >= 2:
            if not self._reacceleration_seen:
                self._reacceleration_seen = True
                phase, reason = "REACCELERATION", "fast velocity reacquired after the first recoil"
            elif breadth[4] >= 2 and slope_ordered:
                if observation.transport == "FULL_ACCEPTANCE":
                    phase, reason = (
                        "PIVOT_ACCEPTED",
                        "reacceleration, ordered cascade, and transport agree",
                    )
                elif observation.transport == "UNAVAILABLE":
                    phase, reason = (
                        "SLOW_SLOPE_CASCADE",
                        "reacceleration and ordered structural cascade are morphology-only without transport",
                    )
                else:
                    phase, reason = (
                        "REACCELERATION",
                        "reacceleration awaits cross-instrument acceptance",
                    )
            else:
                phase, reason = "REACCELERATION", "reacceleration awaits ordered structural acceptance"
        elif breadth[2] >= min(2, fast_count) and breadth[4] >= 2 and slope_ordered:
            phase = "SLOW_SLOPE_CASCADE"
            reason = (
                "ordered structural slopes crossed but recoil survival is unproved"
                if observation.transport != "UNAVAILABLE"
                else "ordered structural cascade advances morphology only because transport is unavailable"
            )
        elif breadth[2] >= min(2, fast_count):
            phase, reason = "FAST_SLOPE_CROSS", "fast slopes crossed while slower propagation remains incomplete"
        elif len(contracting) >= 2:
            phase, reason = "CROSS_CONVERGING", "multiple causal zero-cross estimates contracted"
        elif (
            velocity_ordered
            and sum(name in self._velocity_arrivals for name in (clock.name for clock in fast)) >= 2
            and any(
                clock.name in self._velocity_arrivals
                for clock in observation.clocks
                if clock.tier in {"structural", "slow"}
            )
        ):
            phase, reason = "FRONT_PROPAGATING", "persistent velocity reached a slower clock in order"
        else:
            phase, reason = "REPAIR_SPARK", "new-direction fast velocity is attention only"

        self._phase = phase
        snapshot = self._snapshot(
            observation,
            phase=phase,
            direction=direction,
            eta_contracting=contracting,
            reason=reason,
        )
        if phase == "PIVOT_ACCEPTED":
            self._incumbent = direction
            self._clear_candidate()
        elif phase == "SLOW_SLOPE_CASCADE" and observation.transport == "UNAVAILABLE":
            self._incumbent = direction
            self._clear_candidate()
        elif phase == "RELAPSED":
            self._clear_candidate()
        return snapshot
