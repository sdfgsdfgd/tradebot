"""Frozen completed-minute MCL shock-onset and release owner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median

from ..chart_data.series import OhlcvBar


MCL_MINUTE_SHOCK_VERSION = "mcl.shock-minute-plateau.v106"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("MCL shock timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sign(value: float) -> int | None:
    return 1 if value > 0.0 else -1 if value < 0.0 else None

@dataclass(frozen=True, slots=True)
class MclShockMinute:
    """One completed matched-contract minute for the frozen Stage-106 lane."""

    contract_key: str
    cl: OhlcvBar
    mcl: OhlcvBar

    def __post_init__(self) -> None:
        if not self.contract_key.strip():
            raise ValueError("MCL shock minute requires a contract key")
        if self.cl.ts != self.mcl.ts:
            raise ValueError("MCL shock minute timestamps do not match")
        _utc(self.cl.ts)

    @property
    def ts(self) -> datetime:
        return _utc(self.cl.ts)


@dataclass(frozen=True, slots=True)
class MclMinuteShockContext:
    """Causal dual-book volume, range, and multiscale direction anatomy."""

    volume_multiple: float
    strict_direction: int | None
    one_direction: int | None
    spine_direction: int | None
    velocity_direction: int | None
    acceleration_direction: int | None
    reversal_direction: int | None
    range_multiple: float


@dataclass(frozen=True, slots=True)
class MclMinuteShockTransition:
    """One Stage-106 transition; entries and exits are due at this minute open."""

    observed_at_utc: datetime
    entry_direction: int | None
    entry_signal_at_utc: datetime | None
    exit_reason: str | None
    scheduled_entry_direction: int | None
    scheduled_entry_signal_at_utc: datetime | None
    scheduled_exit_reason: str | None
    active_direction_at_open: int | None
    active_direction_after_close: int | None
    context: MclMinuteShockContext | None
    contract_reset: bool
    gap_reset: bool


def _shared_direction(values: list[float]) -> int | None:
    signs = {_sign(value) for value in values}
    return signs.pop() if len(signs) == 1 and None not in signs else None


def _minute_slope(values: list[float], bars: int, end: int = 0) -> float:
    stop = len(values) + end
    return 100.0 * ((values[stop - 1] / values[stop - bars - 1]) - 1.0) / bars


def _true_range(bar: OhlcvBar, previous_close: float) -> float:
    return max(
        float(bar.high) - float(bar.low),
        abs(float(bar.high) - previous_close),
        abs(float(bar.low) - previous_close),
    )


def mcl_minute_shock_context(
    history: deque[MclShockMinute],
) -> MclMinuteShockContext | None:
    """Project the exact frozen Stage-93 causal context used by Stage 106."""

    if len(history) < 19:
        return None
    rows = list(history)
    volumes = [float(row.mcl.volume) for row in rows]
    baseline = median(volumes[-11:-1])
    if baseline <= 0.0:
        return None
    closes = {
        side: [float(getattr(row, side).close) for row in rows]
        for side in ("cl", "mcl")
    }
    full: list[float] = []
    spine: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []
    reversal: list[float] = []
    one: list[float] = []
    for values in closes.values():
        move1 = values[-1] - values[-2]
        move5 = values[-1] - values[-6]
        move15 = values[-1] - values[-16]
        current_slope = _minute_slope(values, 15)
        previous_slope = _minute_slope(values, 15, -1)
        velocity = current_slope - previous_slope
        acceleration = velocity - (
            previous_slope - _minute_slope(values, 15, -2)
        )
        one.append(move1)
        spine.extend((move15, current_slope))
        velocities.append(velocity)
        accelerations.append(acceleration)
        reversal.extend((move1, move5, move15, velocity, acceleration))
        full.extend((move1, move5, move15, current_slope, velocity, acceleration))
    range_multiples = []
    for side in ("cl", "mcl"):
        current = getattr(rows[-1], side)
        prior_ranges = [
            _true_range(
                getattr(rows[index], side),
                float(getattr(rows[index - 1], side).close),
            )
            for index in range(len(rows) - 11, len(rows) - 1)
        ]
        baseline_range = median(prior_ranges)
        current_range = _true_range(
            current, float(getattr(rows[-2], side).close)
        )
        range_multiples.append(
            current_range / baseline_range if baseline_range > 0.0 else 0.0
        )
    return MclMinuteShockContext(
        volume_multiple=volumes[-1] / baseline,
        strict_direction=_shared_direction(full),
        one_direction=_shared_direction(one),
        spine_direction=_shared_direction(spine),
        velocity_direction=_shared_direction(velocities),
        acceleration_direction=_shared_direction(accelerations),
        reversal_direction=_shared_direction(reversal),
        range_multiple=min(range_multiples),
    )


class MclMinuteShockEngine:
    """Own the selected Stage-106 onset, maturation, and release law exactly."""

    onset_volume_multiple = 15.0
    onset_range_multiple = 1.5
    confirmation_minutes = 3
    confirmation_volume_floor = 3.0
    admission_range_floor = 2.875
    admission_volume_floor = 4.0
    normalization_volume_floor = 3.0
    soft_release_minutes = 3
    hard_reversal_minutes = 2
    maturation_minutes = 1

    def __init__(self) -> None:
        self._history: deque[MclShockMinute] = deque(maxlen=64)
        self._previous: MclShockMinute | None = None
        self._provisional_direction: int | None = None
        self._provisional_baseline: float | None = None
        self._provisional_count = 0
        self._provisional_peak = 0.0
        self._shock_direction: int | None = None
        self._shock_baseline: float | None = None
        self._shock_peak = 0.0
        self._release_count = 0
        self._hard_reversal_count = 0
        self._maturation_direction: int | None = None
        self._maturation_signal: datetime | None = None
        self._maturation_deadline: datetime | None = None
        self._pending_entry: tuple[int, datetime] | None = None
        self._pending_exit: str | None = None
        self._lane_position: int | None = None

    @property
    def active_direction(self) -> int | None:
        return self._shock_direction

    def _clear_provisional(self) -> None:
        self._provisional_direction = None
        self._provisional_baseline = None
        self._provisional_count = 0
        self._provisional_peak = 0.0

    def _clear_maturation(self) -> None:
        self._maturation_direction = None
        self._maturation_signal = None
        self._maturation_deadline = None

    def _clear_shock(self) -> None:
        self._shock_direction = None
        self._shock_baseline = None
        self._shock_peak = 0.0
        self._release_count = 0
        self._hard_reversal_count = 0
        self._clear_maturation()

    def _reset_path(self) -> None:
        self._history.clear()
        self._clear_provisional()
        self._clear_shock()
        self._pending_entry = None
        self._pending_exit = None
        self._lane_position = None

    def _starts(self, context: MclMinuteShockContext) -> bool:
        return bool(
            context.volume_multiple >= self.onset_volume_multiple
            and context.range_multiple >= self.onset_range_multiple
            and context.strict_direction in (-1, 1)
        )

    def _start(
        self, context: MclMinuteShockContext, *, volume: float
    ) -> None:
        self._provisional_direction = int(context.strict_direction)
        self._provisional_baseline = volume / context.volume_multiple
        self._provisional_count = 1
        self._provisional_peak = context.volume_multiple

    def _confirm(
        self, context: MclMinuteShockContext, minute: MclShockMinute
    ) -> None:
        direction = int(self._provisional_direction)
        baseline = float(self._provisional_baseline or 0.0)
        frozen_multiple = float(minute.mcl.volume) / baseline
        self._shock_direction = direction
        self._shock_baseline = baseline
        self._shock_peak = self._provisional_peak
        self._release_count = 0
        self._hard_reversal_count = 0
        if (
            context.range_multiple >= self.admission_range_floor
            and frozen_multiple >= self.admission_volume_floor
        ):
            if context.velocity_direction == direction:
                self._pending_entry = (direction, minute.ts)
            else:
                self._maturation_direction = direction
                self._maturation_signal = minute.ts
                self._maturation_deadline = minute.ts + timedelta(
                    minutes=self.maturation_minutes
                )
        self._clear_provisional()

    def _mature(
        self, context: MclMinuteShockContext, minute: MclShockMinute
    ) -> None:
        direction = self._maturation_direction
        deadline = self._maturation_deadline
        if direction is None or deadline is None:
            return
        if (
            context.spine_direction != direction
            or context.reversal_direction == -direction
        ):
            self._clear_maturation()
        elif context.velocity_direction == direction:
            self._pending_entry = (direction, minute.ts)
            self._clear_maturation()
        elif minute.ts >= deadline:
            self._clear_maturation()

    def _release(self, reason: str) -> None:
        self._pending_entry = None
        self._clear_maturation()
        if self._lane_position is not None:
            self._pending_exit = reason
        self._clear_shock()

    def _advance_state(
        self, context: MclMinuteShockContext | None, minute: MclShockMinute
    ) -> None:
        if context is None:
            self._clear_provisional()
            return
        volume = float(minute.mcl.volume)
        if self._shock_direction is None:
            if self._provisional_direction is None:
                if self._starts(context):
                    self._start(context, volume=volume)
                return
            direction = int(self._provisional_direction)
            baseline = float(self._provisional_baseline or 0.0)
            multiple = volume / baseline if baseline > 0.0 else 0.0
            if (
                multiple >= self.confirmation_volume_floor
                and context.one_direction == direction
                and context.spine_direction == direction
            ):
                self._provisional_count += 1
                self._provisional_peak = max(self._provisional_peak, multiple)
                if self._provisional_count >= self.confirmation_minutes:
                    self._confirm(context, minute)
                return
            self._clear_provisional()
            if self._starts(context):
                self._start(context, volume=volume)
            return

        self._mature(context, minute)
        direction = int(self._shock_direction)
        baseline = float(self._shock_baseline or 0.0)
        multiple = volume / baseline if baseline > 0.0 else 0.0
        self._shock_peak = max(self._shock_peak, multiple)
        full_reversal = context.reversal_direction == -direction
        self._hard_reversal_count = (
            self._hard_reversal_count + 1 if full_reversal else 0
        )
        if self._hard_reversal_count >= self.hard_reversal_minutes:
            self._release("joint_reversal_two_minute_hard")
            return
        soft = (
            multiple < self.normalization_volume_floor
            and context.spine_direction != direction
        )
        self._release_count = self._release_count + 1 if soft else 0
        if self._release_count >= self.soft_release_minutes:
            self._release("quiet_spine_loss")

    def update(self, minute: MclShockMinute) -> MclMinuteShockTransition:
        previous = self._previous
        if previous is not None and minute.ts <= previous.ts:
            raise ValueError("MCL shock minute timestamps must increase")
        contract_reset = previous is not None and (
            minute.contract_key != previous.contract_key
        )
        gap_reset = previous is not None and (
            minute.ts - previous.ts != timedelta(minutes=1)
        )
        if previous is None or contract_reset or gap_reset:
            self._reset_path()

        entry = self._pending_entry
        exit_reason = self._pending_exit
        self._pending_entry = None
        self._pending_exit = None
        if exit_reason is not None:
            self._lane_position = None
        if entry is not None and self._lane_position is None:
            self._lane_position = entry[0]
        active_at_open = self._shock_direction

        self._history.append(minute)
        context = mcl_minute_shock_context(self._history)
        self._advance_state(context, minute)
        scheduled_entry = self._pending_entry
        self._previous = minute
        return MclMinuteShockTransition(
            observed_at_utc=minute.ts,
            entry_direction=entry[0] if entry is not None else None,
            entry_signal_at_utc=entry[1] if entry is not None else None,
            exit_reason=exit_reason,
            scheduled_entry_direction=(
                scheduled_entry[0] if scheduled_entry is not None else None
            ),
            scheduled_entry_signal_at_utc=(
                scheduled_entry[1] if scheduled_entry is not None else None
            ),
            scheduled_exit_reason=self._pending_exit,
            active_direction_at_open=active_at_open,
            active_direction_after_close=self._shock_direction,
            context=context,
            contract_reset=contract_reset,
            gap_reset=gap_reset,
        )
