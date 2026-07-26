"""Causal multihorizon directional and turning evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from ..time_utils import to_et as _to_et


@dataclass(frozen=True)
class DirectionalImpulseHorizon:
    """Causal price-path evidence for one lookback."""

    bars: int
    elapsed_minutes: float | None
    observations: int
    anchor_lag_minutes: float | None
    return_pct: float
    slope_pct_per_bar: float
    slope_velocity_pct_per_bar: float | None
    slope_angle_deg: float
    efficiency: float
    tr_mean_pct: float
    turn: str | None
    turn_age_bars: int | None

    def as_payload(self) -> dict[str, object]:
        return {
            "bars": int(self.bars),
            "elapsed_minutes": self.elapsed_minutes,
            "observations": int(self.observations),
            "anchor_lag_minutes": self.anchor_lag_minutes,
            "return_pct": float(self.return_pct),
            "slope_pct_per_bar": float(self.slope_pct_per_bar),
            "slope_velocity_pct_per_bar": (
                float(self.slope_velocity_pct_per_bar)
                if self.slope_velocity_pct_per_bar is not None
                else None
            ),
            "slope_angle_deg": float(self.slope_angle_deg),
            "efficiency": float(self.efficiency),
            "tr_mean_pct": float(self.tr_mean_pct),
            "turn": self.turn,
            "turn_age_bars": self.turn_age_bars,
        }


DIRECTIONAL_IMPULSE_HORIZONS = (1, 3, 6, 12, 24)
DIRECTIONAL_IMPULSE_WARMUP_BARS = max(DIRECTIONAL_IMPULSE_HORIZONS) + 1


@dataclass(frozen=True)
class DirectionalTurnPolicy:
    """Frozen causal interpretation of XSP's early-session impulse evidence."""

    smooth_alpha: float = 0.90
    initial_score: float = 0.075
    turn_score: float = 0.02
    retrace_atr: float = 0.75
    min_state_bars: int = 3
    cooldown_bars: int = 3
    min_observed_horizons: int = 3
    bar_duration: timedelta = timedelta(minutes=5)
    start_et: time = time(9, 30)
    end_et: time = time(11, 45)

    def as_payload(self) -> dict[str, object]:
        return {
            "smooth_alpha": float(self.smooth_alpha),
            "initial_score": float(self.initial_score),
            "turn_score": float(self.turn_score),
            "retrace_atr": float(self.retrace_atr),
            "min_state_bars": int(self.min_state_bars),
            "cooldown_bars": int(self.cooldown_bars),
            "min_observed_horizons": int(self.min_observed_horizons),
            "bar_duration_seconds": float(self.bar_duration.total_seconds()),
            "start_et": self.start_et.isoformat(timespec="minutes"),
            "end_et": self.end_et.isoformat(timespec="minutes"),
        }


@dataclass(frozen=True)
class DirectionalImpulseAdmissionPolicy:
    """Optional causal admission layer; the raw turn still owns source state."""

    start_minute_et: int = 570
    core_end_minute_et: int = 675
    late_up_end_minute_et: int = 685
    atr_velocity_min: float = 0.0
    atr_velocity_max: float = 0.055
    down_retrace_min: float = 1.25
    late_up_retrace_min: float = 1.25
    late_up_retrace_max: float = 1.70
    late_up_coherence_min: float = 0.75

    def __post_init__(self) -> None:
        if not (
            0 <= self.start_minute_et
            <= self.core_end_minute_et
            <= self.late_up_end_minute_et
            < 1440
        ):
            raise ValueError("invalid directional impulse admission time window")
        if not 0.0 <= self.atr_velocity_min < self.atr_velocity_max:
            raise ValueError("invalid directional impulse ATR velocity band")
        if not 0.0 <= self.late_up_retrace_min < self.late_up_retrace_max:
            raise ValueError("invalid directional impulse late-up retrace band")
        if not 0.0 <= self.late_up_coherence_min <= 1.0:
            raise ValueError("invalid directional impulse coherence floor")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None,
    ) -> "DirectionalImpulseAdmissionPolicy | None":
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("directional_impulse_admission must be an object")
        mode = str(raw.get("mode", "off") or "off").strip().lower()
        if mode in ("off", "none", "disabled"):
            return None
        if mode != "opening_edge":
            raise ValueError(f"unsupported directional impulse admission mode: {mode}")

        def value(name: str, default: int | float, cast):
            try:
                return cast(raw.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid directional impulse admission value: {name}"
                ) from exc

        return cls(
            start_minute_et=value("start_minute_et", 570, int),
            core_end_minute_et=value("core_end_minute_et", 675, int),
            late_up_end_minute_et=value("late_up_end_minute_et", 685, int),
            atr_velocity_min=value("atr_velocity_min", 0.0, float),
            atr_velocity_max=value("atr_velocity_max", 0.055, float),
            down_retrace_min=value("down_retrace_min", 1.25, float),
            late_up_retrace_min=value("late_up_retrace_min", 1.25, float),
            late_up_retrace_max=value("late_up_retrace_max", 1.70, float),
            late_up_coherence_min=value("late_up_coherence_min", 0.75, float),
        )

    def allows(
        self,
        *,
        direction: str | None,
        minute_et: int,
        atr_velocity: float | None,
        retrace_atr: float | None,
        coherence: float | None,
    ) -> tuple[bool, str]:
        if direction not in ("up", "down"):
            return False, "no_turn"
        velocity = float(atr_velocity or 0.0)
        if not self.atr_velocity_min < velocity < self.atr_velocity_max:
            return False, "atr_velocity"
        retrace = float(retrace_atr or 0.0)
        if self.start_minute_et <= int(minute_et) <= self.core_end_minute_et:
            return (
                (True, "core")
                if direction == "up" or retrace >= self.down_retrace_min
                else (False, "down_retrace")
            )
        if direction == "up" and int(minute_et) <= self.late_up_end_minute_et:
            allowed = bool(
                self.late_up_retrace_min <= retrace < self.late_up_retrace_max
                and float(coherence or 0.0) >= self.late_up_coherence_min
            )
            return allowed, "late_up" if allowed else "late_up_quality"
        return False, "time"

    def as_payload(self) -> dict[str, object]:
        return {
            "mode": "opening_edge",
            "start_minute_et": int(self.start_minute_et),
            "core_end_minute_et": int(self.core_end_minute_et),
            "late_up_end_minute_et": int(self.late_up_end_minute_et),
            "atr_velocity_min": float(self.atr_velocity_min),
            "atr_velocity_max": float(self.atr_velocity_max),
            "down_retrace_min": float(self.down_retrace_min),
            "late_up_retrace_min": float(self.late_up_retrace_min),
            "late_up_retrace_max": float(self.late_up_retrace_max),
            "late_up_coherence_min": float(self.late_up_coherence_min),
        }


@dataclass(frozen=True)
class DirectionalImpulseSnapshot:
    """Scale-free multihorizon direction plus non-directional volatility support."""

    ready: bool
    direction: str | None
    abstain_reason: str | None
    direction_score: float | None
    coherence: float | None
    conviction: float | None
    atr_fast_pct: float | None
    atr_slow_pct: float | None
    atr_ratio: float | None
    atr_velocity_pct: float | None
    atr_acceleration_pct: float | None
    turn_sequence_direction: str | None
    turn_sequence_order: str | None
    turn_sequence_span_bars: int | None
    observed_horizons: int
    required_turn_horizons: int | None
    turn_ready: bool
    turn_abstain_reason: str | None
    smoothed_direction_score: float | None
    trend_state: str | None
    state_age_bars: int | None
    retrace_atr: float | None
    turn_event: str | None
    horizons: tuple[DirectionalImpulseHorizon, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "ready": bool(self.ready),
            "direction": self.direction,
            "abstain_reason": self.abstain_reason,
            "direction_score": self.direction_score,
            "coherence": self.coherence,
            "conviction": self.conviction,
            "atr_fast_pct": self.atr_fast_pct,
            "atr_slow_pct": self.atr_slow_pct,
            "atr_ratio": self.atr_ratio,
            "atr_velocity_pct": self.atr_velocity_pct,
            "atr_acceleration_pct": self.atr_acceleration_pct,
            "turn_sequence_direction": self.turn_sequence_direction,
            "turn_sequence_order": self.turn_sequence_order,
            "turn_sequence_span_bars": self.turn_sequence_span_bars,
            "observed_horizons": int(self.observed_horizons),
            "required_turn_horizons": self.required_turn_horizons,
            "turn_ready": bool(self.turn_ready),
            "turn_abstain_reason": self.turn_abstain_reason,
            "smoothed_direction_score": self.smoothed_direction_score,
            "trend_state": self.trend_state,
            "state_age_bars": self.state_age_bars,
            "retrace_atr": self.retrace_atr,
            "turn_event": self.turn_event,
            "horizons": [row.as_payload() for row in self.horizons],
        }


class DirectionalImpulseEngine:
    """Fuse signed path velocity across horizons without giving volatility direction."""

    def __init__(
        self,
        *,
        horizons: tuple[int, ...] = DIRECTIONAL_IMPULSE_HORIZONS,
        min_direction_score: float = 0.20,
        min_coherence: float = 0.60,
        bar_duration: timedelta | None = None,
        max_anchor_lag: timedelta = timedelta(0),
        turn_policy: DirectionalTurnPolicy | None = None,
    ) -> None:
        cleaned = tuple(sorted({max(1, int(value)) for value in horizons}))
        if not cleaned:
            raise ValueError("directional impulse requires at least one horizon")
        self._horizons = cleaned
        self._min_direction_score = max(0.0, min(1.0, float(min_direction_score)))
        self._min_coherence = max(0.0, min(1.0, float(min_coherence)))
        if bar_duration is not None and bar_duration.total_seconds() <= 0.0:
            raise ValueError("bar_duration must be positive")
        if max_anchor_lag.total_seconds() < 0.0:
            raise ValueError("max_anchor_lag cannot be negative")
        self._bar_duration = bar_duration
        self._max_anchor_lag = max_anchor_lag
        self._turn_policy = turn_policy
        self._session_key: object | None = None
        self._closes: deque[float] = deque(maxlen=max(cleaned) + 1)
        self._tr_pct: deque[float] = deque(maxlen=max(cleaned) + 1)
        self._timestamps: deque[datetime] = deque(maxlen=max(cleaned) + 1)
        self._previous_slopes: dict[int, float] = {}
        self._turn_ages: dict[int, int | None] = {}
        self._previous_atr_fast: float | None = None
        self._previous_atr_velocity: float | None = None
        self._turn_index = -1
        self._turn_smooth: float | None = None
        self._trend_state: int | None = None
        self._trend_state_index: int | None = None
        self._trend_peak: float | None = None
        self._trend_trough: float | None = None
        self._last_turn_index: int | None = None

    @property
    def warmup_bars(self) -> int:
        """Price samples required to measure the longest return horizon."""
        return max(self._horizons) + 1

    def _reset(self) -> None:
        self._closes.clear()
        self._tr_pct.clear()
        self._timestamps.clear()
        self._previous_slopes.clear()
        self._turn_ages.clear()
        self._previous_atr_fast = None
        self._previous_atr_velocity = None
        self._turn_index = -1
        self._turn_smooth = None
        self._trend_state = None
        self._trend_state_index = None
        self._trend_peak = None
        self._trend_trough = None
        self._last_turn_index = None

    @staticmethod
    def _mean_tail(values: deque[float], bars: int) -> float | None:
        if len(values) < int(bars):
            return None
        tail = tuple(values)[-int(bars) :]
        return float(sum(tail) / len(tail))

    def update(
        self,
        *,
        high: float,
        low: float,
        close: float,
        session_key: object | None = None,
        ts: datetime | None = None,
    ) -> DirectionalImpulseSnapshot:
        if session_key is not None and session_key != self._session_key:
            self._reset()
            self._session_key = session_key

        previous_close = self._closes[-1] if self._closes else None
        if self._bar_duration is not None:
            if ts is None:
                raise ValueError("timestamp required for elapsed-time horizons")
            if self._timestamps and ts <= self._timestamps[-1]:
                raise ValueError("directional impulse timestamps must increase")
        if close <= 0.0 or high <= 0.0 or low <= 0.0:
            return DirectionalImpulseSnapshot(
                ready=False,
                direction=None,
                abstain_reason="invalid_bar",
                direction_score=None,
                coherence=None,
                conviction=None,
                atr_fast_pct=None,
                atr_slow_pct=None,
                atr_ratio=None,
                atr_velocity_pct=None,
                atr_acceleration_pct=None,
                turn_sequence_direction=None,
                turn_sequence_order=None,
                turn_sequence_span_bars=None,
                observed_horizons=0,
                required_turn_horizons=(
                    self._turn_policy.min_observed_horizons
                    if self._turn_policy is not None
                    else None
                ),
                turn_ready=False,
                turn_abstain_reason=(
                    "invalid_bar" if self._turn_policy is not None else None
                ),
                smoothed_direction_score=self._turn_smooth,
                trend_state=(
                    "up"
                    if self._trend_state == 1
                    else "down"
                    if self._trend_state == -1
                    else None
                ),
                state_age_bars=None,
                retrace_atr=None,
                turn_event=None,
                horizons=(),
            )

        true_range = max(0.0, float(high) - float(low))
        if previous_close is not None:
            true_range = max(
                true_range,
                abs(float(high) - float(previous_close)),
                abs(float(low) - float(previous_close)),
            )
        self._closes.append(float(close))
        self._tr_pct.append((float(true_range) / float(close)) * 100.0)
        if ts is not None:
            self._timestamps.append(ts)

        rows: list[DirectionalImpulseHorizon] = []
        close_values = tuple(self._closes)
        tr_values = tuple(self._tr_pct)
        timestamp_values = tuple(self._timestamps)
        for bars in self._horizons:
            elapsed_minutes = None
            anchor_lag_minutes = None
            if self._bar_duration is None:
                if len(close_values) <= int(bars) or len(tr_values) < int(bars):
                    continue
                start_idx = len(close_values) - int(bars) - 1
                elapsed_bars = float(bars)
                tr_window = tr_values[-int(bars) :]
            else:
                target = timestamp_values[-1] - (self._bar_duration * int(bars))
                start_idx = next(
                    (
                        idx
                        for idx in range(len(timestamp_values) - 1, -1, -1)
                        if timestamp_values[idx] <= target
                    ),
                    -1,
                )
                if start_idx < 0:
                    continue
                anchor_lag = target - timestamp_values[start_idx]
                if anchor_lag > self._max_anchor_lag:
                    continue
                elapsed = timestamp_values[-1] - timestamp_values[start_idx]
                elapsed_bars = elapsed / self._bar_duration
                if elapsed_bars <= 0.0:
                    continue
                elapsed_minutes = elapsed.total_seconds() / 60.0
                anchor_lag_minutes = anchor_lag.total_seconds() / 60.0
                tr_window = tr_values[start_idx + 1 :]
                if not tr_window:
                    continue

            start = float(close_values[start_idx])
            if start <= 0.0:
                continue
            return_pct = ((float(close) / start) - 1.0) * 100.0
            slope = float(return_pct) / float(elapsed_bars)
            previous_slope = self._previous_slopes.get(int(bars))
            slope_velocity = (
                float(slope) - float(previous_slope)
                if previous_slope is not None
                else None
            )
            turn = None
            if previous_slope is not None:
                if previous_slope <= 0.0 < slope:
                    turn = "up"
                elif previous_slope >= 0.0 > slope:
                    turn = "down"
            previous_turn_age = self._turn_ages.get(int(bars))
            turn_age = (
                0
                if turn is not None
                else (
                    int(previous_turn_age) + 1
                    if previous_turn_age is not None
                    else None
                )
            )
            self._previous_slopes[int(bars)] = float(slope)
            self._turn_ages[int(bars)] = turn_age

            path = 0.0
            for idx in range(start_idx + 1, len(close_values)):
                left = float(close_values[idx - 1])
                right = float(close_values[idx])
                if left > 0.0:
                    path += abs(((right / left) - 1.0) * 100.0)
            efficiency = (
                max(-1.0, min(1.0, float(return_pct) / float(path)))
                if path > 0.0
                else 0.0
            )
            tr_mean = float(sum(tr_window) / len(tr_window))
            angle = math.degrees(
                math.atan(float(slope) / max(float(tr_mean), 1e-12))
            )
            rows.append(
                DirectionalImpulseHorizon(
                    bars=int(bars),
                    elapsed_minutes=elapsed_minutes,
                    observations=len(close_values) - start_idx,
                    anchor_lag_minutes=anchor_lag_minutes,
                    return_pct=float(return_pct),
                    slope_pct_per_bar=float(slope),
                    slope_velocity_pct_per_bar=(
                        float(slope_velocity)
                        if slope_velocity is not None
                        else None
                    ),
                    slope_angle_deg=float(angle),
                    efficiency=float(efficiency),
                    tr_mean_pct=float(tr_mean),
                    turn=turn,
                    turn_age_bars=turn_age,
                )
            )

        ready = len(rows) == len(self._horizons)
        direction_score: float | None = None
        coherence: float | None = None
        direction: str | None = None
        abstain_reason = "not_ready"
        if rows:
            contributions = [
                (
                    row.slope_pct_per_bar
                    / math.hypot(row.slope_pct_per_bar, row.tr_mean_pct)
                    if row.slope_pct_per_bar or row.tr_mean_pct
                    else 0.0
                )
                * (0.5 + (0.5 * abs(row.efficiency)))
                for row in rows
            ]
            if any(contributions):
                direction_score = float(sum(contributions) / len(contributions))
                raw_direction = "up" if direction_score > 0.0 else "down"
                matching = sum(
                    1
                    for row in rows
                    if (row.slope_pct_per_bar > 0.0) == (raw_direction == "up")
                )
                coherence = float(matching / len(rows))
                if ready and abs(direction_score) >= self._min_direction_score:
                    if coherence >= self._min_coherence:
                        direction = raw_direction
                        abstain_reason = None
                    else:
                        abstain_reason = "incoherent"
                elif ready:
                    abstain_reason = "weak_direction"
            elif ready:
                abstain_reason = "weak_direction"

        fast_bars = self._horizons[min(1, len(self._horizons) - 1)]
        slow_bars = self._horizons[-1]
        if self._bar_duration is None:
            atr_fast = self._mean_tail(self._tr_pct, fast_bars)
            atr_slow = self._mean_tail(self._tr_pct, slow_bars)
        else:
            tr_by_horizon = {row.bars: row.tr_mean_pct for row in rows}
            atr_fast = tr_by_horizon.get(fast_bars)
            atr_slow = tr_by_horizon.get(slow_bars)
        atr_ratio = (
            float(atr_fast) / float(atr_slow)
            if atr_fast is not None and atr_slow is not None and atr_slow > 0.0
            else None
        )
        atr_velocity = (
            float(atr_fast) - float(self._previous_atr_fast)
            if atr_fast is not None and self._previous_atr_fast is not None
            else None
        )
        atr_acceleration = (
            float(atr_velocity) - float(self._previous_atr_velocity)
            if atr_velocity is not None and self._previous_atr_velocity is not None
            else None
        )
        if atr_fast is not None:
            self._previous_atr_fast = float(atr_fast)
        if atr_velocity is not None:
            self._previous_atr_velocity = float(atr_velocity)

        conviction = None
        if direction_score is not None and coherence is not None:
            volatility_support = max(
                0.5,
                min(1.5, float(atr_ratio) if atr_ratio is not None else 1.0),
            )
            conviction = min(
                1.0,
                abs(float(direction_score))
                * float(coherence)
                * (0.75 + (0.25 * volatility_support)),
            )

        turn_sequence_direction = None
        turn_sequence_order = None
        turn_sequence_span = None
        if len(rows) >= 3:
            sentinels = (rows[0], rows[len(rows) // 2], rows[-1])
            signs = tuple(
                "up"
                if row.slope_pct_per_bar > 0.0
                else "down"
                if row.slope_pct_per_bar < 0.0
                else None
                for row in sentinels
            )
            ages = tuple(row.turn_age_bars for row in sentinels)
            if signs[0] is not None and len(set(signs)) == 1 and all(
                age is not None for age in ages
            ):
                short_age, middle_age, long_age = (
                    int(age) for age in ages if age is not None
                )
                turn_sequence_direction = signs[0]
                turn_sequence_span = max(ages) - min(ages)
                if short_age <= middle_age <= long_age:
                    turn_sequence_order = "long_to_short"
                elif short_age >= middle_age >= long_age:
                    turn_sequence_order = "short_to_long"
                else:
                    turn_sequence_order = "mixed"

        observed_horizons = len(rows)
        turn_ready = False
        turn_abstain_reason = None
        turn_event = None
        state_age = None
        retrace_atr = None
        policy = self._turn_policy
        if policy is not None:
            turn_abstain_reason = "not_ready"
            if ts is None:
                turn_abstain_reason = "timestamp_missing"
            elif self._bar_duration != policy.bar_duration:
                turn_abstain_reason = "unsupported_bar_size"
            else:
                current_et = _to_et(ts, naive_ts_mode="utc")
                in_window = policy.start_et <= current_et.time() <= policy.end_et
                if not in_window:
                    turn_abstain_reason = "outside_window"
                else:
                    self._turn_index += 1
                    raw_score = float(direction_score or 0.0)
                    self._turn_smooth = (
                        raw_score
                        if self._turn_smooth is None
                        else (
                            float(policy.smooth_alpha) * raw_score
                            + (1.0 - float(policy.smooth_alpha))
                            * float(self._turn_smooth)
                        )
                    )
                    turn_ready = (
                        observed_horizons >= int(policy.min_observed_horizons)
                    )
                    if not turn_ready:
                        turn_abstain_reason = "underwarmed"
                    if self._trend_state is None:
                        if abs(float(self._turn_smooth)) >= float(
                            policy.initial_score
                        ):
                            self._trend_state = (
                                1 if self._turn_smooth > 0.0 else -1
                            )
                            self._trend_state_index = self._turn_index
                            self._trend_peak = float(high)
                            self._trend_trough = float(low)
                        if turn_ready:
                            turn_abstain_reason = "initializing"
                    else:
                        self._trend_peak = max(
                            float(self._trend_peak or high),
                            float(high),
                        )
                        self._trend_trough = min(
                            float(self._trend_trough or low),
                            float(low),
                        )
                        state_age = self._turn_index - int(
                            self._trend_state_index or 0
                        )
                        retrace_pct = (
                            (
                                float(self._trend_peak) - float(close)
                            )
                            / float(close)
                            * 100.0
                            if self._trend_state == 1
                            else (
                                float(close) - float(self._trend_trough)
                            )
                            / float(close)
                            * 100.0
                        )
                        current_tr = (
                            float(rows[-1].tr_mean_pct)
                            if rows
                            else float(self._tr_pct[-1])
                        )
                        retrace_atr = retrace_pct / max(current_tr, 1e-9)
                        crossed = (
                            self._turn_smooth <= -float(policy.turn_score)
                            if self._trend_state == 1
                            else self._turn_smooth >= float(policy.turn_score)
                        )
                        cooldown_ready = (
                            self._last_turn_index is None
                            or self._turn_index - self._last_turn_index
                            >= int(policy.cooldown_bars)
                        )
                        if (
                            turn_ready
                            and crossed
                            and state_age >= int(policy.min_state_bars)
                            and retrace_atr >= float(policy.retrace_atr)
                            and cooldown_ready
                        ):
                            self._trend_state = -self._trend_state
                            self._trend_state_index = self._turn_index
                            self._last_turn_index = self._turn_index
                            self._trend_peak = float(high)
                            self._trend_trough = float(low)
                            state_age = 0
                            turn_event = (
                                "up" if self._trend_state == 1 else "down"
                            )
                            turn_abstain_reason = None
                        elif turn_ready:
                            turn_abstain_reason = "holding"

        return DirectionalImpulseSnapshot(
            ready=bool(ready),
            direction=direction,
            abstain_reason=abstain_reason,
            direction_score=direction_score,
            coherence=coherence,
            conviction=conviction,
            atr_fast_pct=atr_fast,
            atr_slow_pct=atr_slow,
            atr_ratio=atr_ratio,
            atr_velocity_pct=atr_velocity,
            atr_acceleration_pct=atr_acceleration,
            turn_sequence_direction=turn_sequence_direction,
            turn_sequence_order=turn_sequence_order,
            turn_sequence_span_bars=turn_sequence_span,
            observed_horizons=observed_horizons,
            required_turn_horizons=(
                policy.min_observed_horizons if policy is not None else None
            ),
            turn_ready=bool(turn_ready),
            turn_abstain_reason=turn_abstain_reason,
            smoothed_direction_score=self._turn_smooth,
            trend_state=(
                "up"
                if self._trend_state == 1
                else "down"
                if self._trend_state == -1
                else None
            ),
            state_age_bars=state_age,
            retrace_atr=retrace_atr,
            turn_event=turn_event,
            horizons=tuple(rows),
        )
