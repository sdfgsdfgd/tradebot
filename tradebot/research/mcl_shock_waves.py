"""Authority-bound waves over the immutable MCL shock-crest owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from .mcl_shock_crest import (
    MclShockCrestEngine,
    MclShockCrestPolicy,
    MclShockDecision,
    MclShockLevel,
    MclShockObservation,
)


MCL_SHOCK_WAVE_VERSION = "mcl.authority-bound-shock-waves.v1"
MCL_SHOCK_WAVE_AUTHORITY = "signal_state_only_no_orders_no_capital"
MclShockWaveEvent = Literal[
    "STATE",
    "ATTENTION_OPENED",
    "LEVEL_ESCALATED",
    "AUTHORITY_PENDING",
    "AUTHORITY_BOUND",
    "AUTHORITY_ESCALATED",
    "AUTHORITY_HANDOFF",
    "CREST_CONFIRMED",
    "CONTINUATION",
    "ROTATION_ARMED",
    "ROTATION_EXIT",
    "REVERSAL_ELIGIBLE",
    "NORMALIZED",
]


_LEVELS: tuple[MclShockLevel, ...] = (
    "NORMAL_UNDER_5X",
    "ELEVATED_5_TO_10X",
    "MAJOR_PROTECT_10_TO_12X",
    "TRADEABLE_SHOCK_12_TO_20X",
    "REGIME_20X_PLUS",
)
_LEVEL_RANK = {level: rank for rank, level in enumerate(_LEVELS)}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("MCL shock-wave timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sign(value: float) -> int | None:
    return 1 if value > 0.0 else -1 if value < 0.0 else None


def mcl_shock_full_alignment(observation: MclShockObservation) -> int | None:
    """Return one direction only when every frozen fast/slow/flow clock agrees."""

    if not observation.spread_eligible or not observation.fresh_top:
        return None
    values = tuple(
        value
        for book in (observation.cl, observation.mcl)
        for value in (
            book.velocity_5s,
            book.velocity_15s,
            book.velocity_60s,
            book.velocity_15m,
            book.acceleration_15m,
            book.signed_flow_15s,
        )
    )
    signs = {_sign(value) for value in values}
    return signs.pop() if len(signs) == 1 and None not in signs else None


@dataclass(frozen=True, slots=True)
class MclShockWaveDecision:
    """One authority transition plus the unchanged inner crest decision."""

    observed_at_utc: datetime
    event: MclShockWaveEvent
    current_level: MclShockLevel
    maximum_level: MclShockLevel
    authority_direction: int | None
    authority_level: MclShockLevel | None
    wave_sequence: int
    handoff_from_direction: int | None
    crest: MclShockDecision | None
    episode_active: bool
    episode_terminal: bool
    reason: str

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": MCL_SHOCK_WAVE_VERSION,
            "authority": MCL_SHOCK_WAVE_AUTHORITY,
            "observed_at_utc": _utc(self.observed_at_utc).isoformat(),
            "event": self.event,
            "current_level": self.current_level,
            "maximum_level": self.maximum_level,
            "authority_direction": self.authority_direction,
            "authority_level": self.authority_level,
            "wave_sequence": int(self.wave_sequence),
            "handoff_from_direction": self.handoff_from_direction,
            "crest": self.crest.as_payload() if self.crest is not None else None,
            "episode_active": bool(self.episode_active),
            "episode_terminal": bool(self.episode_terminal),
            "reason": self.reason,
            "submitted_orders": 0,
        }


class MclAuthorityBoundShockWaveEngine:
    """Bind at 10x and reset crest memory only on a higher-level handoff."""

    def __init__(self, policy: MclShockCrestPolicy | None = None) -> None:
        self.policy = policy or MclShockCrestPolicy()
        self._contract_key: str | None = None
        self._last_at: datetime | None = None
        self._episode = False
        self._maximum_rank = 0
        self._authority_rank: int | None = None
        self._direction: int | None = None
        self._wave_sequence = 0
        self._inner: MclShockCrestEngine | None = None

    def reset(self, *, contract_key: str | None = None) -> None:
        policy = self.policy
        self.__init__(policy)
        self._contract_key = contract_key

    def _new_inner(
        self, observation: MclShockObservation
    ) -> MclShockDecision:
        self._inner = MclShockCrestEngine(self.policy)
        return self._inner.update(observation)

    def update(self, observation: MclShockObservation) -> MclShockWaveDecision:
        at = _utc(observation.observed_at_utc)
        if self._last_at is not None and at <= self._last_at:
            raise ValueError("MCL shock-wave observations must increase")
        if self._contract_key not in (None, observation.contract_key):
            self.reset(contract_key=observation.contract_key)
        self._contract_key = observation.contract_key
        self._last_at = at
        current_level = self.policy.level(observation.volume_multiple)
        current_rank = _LEVEL_RANK[current_level]
        prior_maximum = self._maximum_rank
        self._maximum_rank = max(self._maximum_rank, current_rank)
        full_direction = mcl_shock_full_alignment(observation)
        handoff_from: int | None = None
        crest: MclShockDecision | None = None
        event: MclShockWaveEvent = "STATE"
        reason = "observed_without_transition"

        if not self._episode and current_rank >= 1:
            self._episode = True
            if current_rank >= 2 and full_direction in (-1, 1):
                self._direction = full_direction
                self._authority_rank = current_rank
                self._wave_sequence = 1
                crest = self._new_inner(observation)
                event = "AUTHORITY_BOUND"
                reason = "fresh_full_alignment_bound_at_major_or_higher_level"
            else:
                crest = self._new_inner(observation)
                event = (
                    "AUTHORITY_PENDING" if current_rank >= 2 else "ATTENTION_OPENED"
                )
                reason = (
                    "major_level_requires_fresh_full_alignment"
                    if current_rank >= 2
                    else "five_x_attention_is_directionless"
                )
        elif not self._episode:
            return MclShockWaveDecision(
                observed_at_utc=at,
                event=event,
                current_level=current_level,
                maximum_level=_LEVELS[self._maximum_rank],
                authority_direction=None,
                authority_level=None,
                wave_sequence=0,
                handoff_from_direction=None,
                crest=None,
                episode_active=False,
                episode_terminal=False,
                reason=reason,
            )
        elif self._direction is None:
            if current_rank >= 2 and full_direction in (-1, 1):
                self._direction = full_direction
                self._authority_rank = current_rank
                self._wave_sequence = 1
                crest = self._new_inner(observation)
                event = "AUTHORITY_BOUND"
                reason = "fresh_full_alignment_bound_at_major_or_higher_level"
            else:
                assert self._inner is not None
                crest = self._inner.update(observation)
                if current_rank >= 2 and current_rank > prior_maximum:
                    event = "AUTHORITY_PENDING"
                    reason = "higher_level_has_no_fresh_full_alignment"
                elif current_rank > prior_maximum:
                    event = "LEVEL_ESCALATED"
                    reason = "directionless_attention_level_escalated"
        else:
            assert self._inner is not None and self._authority_rank is not None
            if current_rank > self._authority_rank and full_direction == -self._direction:
                handoff_from = self._direction
                self._direction = full_direction
                self._authority_rank = current_rank
                self._wave_sequence += 1
                crest = self._new_inner(observation)
                event = "AUTHORITY_HANDOFF"
                reason = "higher_level_opposite_full_alignment_reset_crest_state"
            else:
                crest = self._inner.update(observation)
                if current_rank > self._authority_rank and full_direction == self._direction:
                    self._authority_rank = current_rank
                    event = "AUTHORITY_ESCALATED"
                    reason = "higher_level_confirmed_incumbent_authority"
                elif current_rank > self._authority_rank and current_rank > prior_maximum:
                    event = "AUTHORITY_PENDING"
                    reason = "higher_level_has_no_fresh_full_alignment"

        if crest is not None and crest.phase != "STATE" and event == "STATE":
            event = crest.phase
            reason = crest.reason
        terminal = crest is not None and crest.phase == "NORMALIZED"
        decision = MclShockWaveDecision(
            observed_at_utc=at,
            event=event,
            current_level=current_level,
            maximum_level=_LEVELS[self._maximum_rank],
            authority_direction=self._direction,
            authority_level=(
                _LEVELS[self._authority_rank]
                if self._authority_rank is not None
                else None
            ),
            wave_sequence=self._wave_sequence,
            handoff_from_direction=handoff_from,
            crest=crest,
            episode_active=True,
            episode_terminal=terminal,
            reason=reason,
        )
        if terminal:
            self.reset(contract_key=observation.contract_key)
        return decision
