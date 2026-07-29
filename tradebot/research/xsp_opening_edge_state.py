"""Causal regime-and-incumbent state owner for Opening Edge v3."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, time, timedelta

from ..backtest.models import Bar, SpotTrade
from ..engines.signals import SupertrendEngine
from ..backtest.spot_tape import PreparedSpotEvaluatorTape
from ..engines.market import (
    is_early_close_day,
    is_trading_day,
    xsp_bar_session_label_et,
    xsp_bar_trading_date,
)
from ..time_utils import to_et
from ..spot.evaluator_common import SpotSignalSnapshot
from ..spot.lifecycle import SpotExcursionPolicy


XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS = 94
_HORIZONS = (5, 10, 21, 42, 63, 84)
_FAST_HORIZONS = (1, 3, 6)
_SLOW_HORIZONS = (12, 24)
_FIZZLE_POLICY = SpotExcursionPolicy(fizzle_bars=12, fizzle_mfe_atr=0.25)
_NO_EXCURSION_POLICY = SpotExcursionPolicy()


@dataclass(frozen=True)
class XspDailyBar:
    day: date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class _MaturationRule:
    name: str
    required_fast_votes: int
    require_source: bool
    scope: Callable[[SpotSignalSnapshot, Mapping[str, object]], str | None]


def validate_xsp_daily_bars(
    daily_bars: Sequence[XspDailyBar],
    *,
    minimum_sessions: int = XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS,
) -> tuple[XspDailyBar, ...]:
    """Validate one ordered, positive, non-overlapping XSP context tape."""

    rows = tuple(daily_bars)
    if len(rows) < minimum_sessions:
        raise ValueError("Opening Edge v3 daily context is underwarmed")
    for index, row in enumerate(rows):
        values = (row.open, row.high, row.low, row.close)
        if (
            not all(math.isfinite(value) and value > 0.0 for value in values)
            or row.low > min(row.open, row.close)
            or row.high < max(row.open, row.close)
            or row.low > row.high
        ):
            raise ValueError("Opening Edge v3 daily context has malformed OHLC")
        if index and row.day <= rows[index - 1].day:
            kind = "duplicate" if row.day == rows[index - 1].day else "nonmonotonic"
            raise ValueError(f"Opening Edge v3 daily context is {kind}")
    return rows


def xsp_daily_bars_from_intraday(
    bars: Sequence[Bar],
) -> tuple[XspDailyBar, ...]:
    """Aggregate only complete, exact close-stamped XSP RTH sessions."""

    ordered = tuple(bars)
    if any(
        ordered[index].ts <= ordered[index - 1].ts for index in range(1, len(ordered))
    ):
        raise ValueError("Opening Edge v3 intraday context is nonmonotonic")
    grouped: dict[date, list[Bar]] = {}
    for bar in ordered:
        if xsp_bar_session_label_et(bar.ts, naive_ts_mode="utc") != "RTH":
            continue
        day = xsp_bar_trading_date(bar.ts)
        if day is not None:
            grouped.setdefault(day, []).append(bar)
    complete: list[tuple[date, list[Bar]]] = []
    first_day = min(grouped, default=None)
    latest_day = max(grouped, default=None)
    for day, rows in sorted(grouped.items()):
        close = time(13, 0) if is_early_close_day(day) else time(16, 0)
        expected = []
        cursor = time(9, 35)
        while cursor <= close:
            expected.append(cursor)
            stamp = timedelta(
                hours=cursor.hour,
                minutes=cursor.minute,
            ) + timedelta(minutes=5)
            total_minutes = int(stamp.total_seconds() // 60)
            cursor = time(
                hour=(total_minutes // 60) % 24,
                minute=total_minutes % 60,
            )
        observed = [
            to_et(row.ts, naive_ts_mode="utc").time().replace(tzinfo=None)
            for row in rows
        ]
        if observed == expected:
            complete.append((day, rows))
            continue
        if day == first_day and observed and observed[0] > time(9, 35):
            continue
        if day == latest_day and observed and observed[-1] < close:
            continue
        raise ValueError(
            f"Opening Edge v3 XSP RTH session is incomplete: {day.isoformat()}"
        )
    return validate_xsp_daily_bars(
        tuple(
            XspDailyBar(
                day=day,
                open=float(rows[0].open),
                high=max(float(row.high) for row in rows),
                low=min(float(row.low) for row in rows),
                close=float(rows[-1].close),
            )
            for day, rows in complete
        ),
        minimum_sessions=0,
    )


def merge_xsp_daily_context(
    seed: Sequence[XspDailyBar],
    *,
    persisted: Sequence[XspDailyBar] = (),
    fresh: Sequence[XspDailyBar] = (),
) -> tuple[XspDailyBar, ...]:
    """Append exact completed sessions without replacing frozen history."""

    base = list(validate_xsp_daily_bars(seed))
    by_day = {row.day: row for row in base}
    last_seed_day = base[-1].day
    additions: dict[date, XspDailyBar] = {}
    for source in (
        validate_xsp_daily_bars(persisted, minimum_sessions=0),
        validate_xsp_daily_bars(fresh, minimum_sessions=0),
    ):
        for row in source:
            existing = by_day.get(row.day) or additions.get(row.day)
            if existing is not None:
                if existing != row:
                    raise ValueError("Opening Edge v3 daily context overlap drifted")
                continue
            if row.day < last_seed_day:
                raise ValueError("Opening Edge v3 daily context would replace history")
            additions[row.day] = row
    previous = last_seed_day
    for day in sorted(additions):
        expected = previous + timedelta(days=1)
        while not is_trading_day(expected):
            expected += timedelta(days=1)
        if day != expected:
            raise ValueError("Opening Edge v3 daily context has a known gap")
        base.append(additions[day])
        by_day[day] = additions[day]
        previous = day
    return validate_xsp_daily_bars(base)


def _window(
    rows: Sequence[XspDailyBar],
    end: int,
    length: int,
) -> dict[str, float]:
    segment = rows[end - length : end]
    if len(segment) != length:
        raise ValueError("Opening Edge v3 daily context is underwarmed")
    closes = [row.close for row in segment]
    returns = [
        closes[index] / closes[index - 1] - 1.0 for index in range(1, len(closes))
    ]
    true_ranges = [
        max(
            segment[index].high - segment[index].low,
            abs(segment[index].high - segment[index - 1].close),
            abs(segment[index].low - segment[index - 1].close),
        )
        / segment[index - 1].close
        for index in range(1, len(segment))
    ]
    total_return = closes[-1] / closes[0] - 1.0
    path = sum(abs(value) for value in returns)
    realized_volatility = (
        statistics.pstdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
    )
    peak = closes[0]
    maximum_drawdown = 0.0
    for close in closes:
        peak = max(peak, close)
        maximum_drawdown = max(maximum_drawdown, (peak - close) / peak)
    sigma = realized_volatility * math.sqrt(length / 252)
    return {
        "return": total_return,
        "return_sigma": total_return / sigma if sigma > 0.0 else 0.0,
        "max_drawdown": maximum_drawdown,
        "realized_volatility": realized_volatility,
        "atr_mean": statistics.mean(true_ranges) if true_ranges else 0.0,
        "efficiency": abs(total_return) / path if path > 0.0 else 0.0,
        "up_fraction": (
            sum(value > 0.0 for value in returns) / len(returns) if returns else 0.0
        ),
    }


def _daily_observations(
    rows: Sequence[XspDailyBar],
    end: int,
    length: int,
) -> tuple[dict[str, float], ...]:
    return tuple(
        {
            "return": rows[index].close / rows[index - 1].close - 1.0,
            "gap": rows[index].open / rows[index - 1].close - 1.0,
            "tr": max(
                rows[index].high - rows[index].low,
                abs(rows[index].high - rows[index - 1].close),
                abs(rows[index].low - rows[index - 1].close),
            )
            / rows[index - 1].close,
        }
        for index in range(end - length, end)
    )


def _drawdown(
    rows: Sequence[XspDailyBar],
    end: int,
    length: int = 84,
) -> float:
    closes = [row.close for row in rows[end - length : end]]
    return closes[-1] / max(closes) - 1.0


def _sign(value: float, *, zero: str = "flat") -> str:
    return "up" if value > 0.0 else "down" if value < 0.0 else zero


def _age_class(value: int) -> str:
    if value <= 1:
        return "fresh"
    if value <= 5:
        return "maturing"
    if value <= 20:
        return "established"
    return "entrenched"


def _daily_context(
    rows: Sequence[XspDailyBar],
    end: int,
) -> dict[str, object]:
    windows = {horizon: _window(rows, end, horizon) for horizon in _HORIZONS}
    prior5 = {horizon: _window(rows, end - 5, horizon) for horizon in _HORIZONS}
    prior10 = {horizon: _window(rows, end - 10, horizon) for horizon in _HORIZONS}
    directions = {
        horizon: _sign(float(windows[horizon]["return"]), zero="up")
        for horizon in _HORIZONS
    }
    median_tr = {
        horizon: statistics.median(
            row["tr"] for row in _daily_observations(rows, end, horizon)
        )
        for horizon in (5, 21, 84)
    }
    tr5_prior1 = statistics.median(
        row["tr"] for row in _daily_observations(rows, end - 1, 5)
    )
    tr5_prior2 = statistics.median(
        row["tr"] for row in _daily_observations(rows, end - 2, 5)
    )
    tr_velocity = median_tr[5] - tr5_prior1
    tr_acceleration = tr_velocity - (tr5_prior1 - tr5_prior2)
    recent = _daily_observations(rows, end, 5)
    negative_gaps = sum(row["gap"] < 0.0 for row in recent)
    positive_gaps = sum(row["gap"] > 0.0 for row in recent)
    drawdown = _drawdown(rows, end)
    drawdown5 = _drawdown(rows, end - 5)
    drawdown10 = _drawdown(rows, end - 10)
    drawdown_velocity = (drawdown - drawdown5) / 5.0
    prior_drawdown_velocity = (drawdown5 - drawdown10) / 5.0
    velocity = {
        horizon: (
            float(windows[horizon]["return_sigma"])
            - float(prior5[horizon]["return_sigma"])
        )
        / 5.0
        for horizon in _HORIZONS
    }
    prior_velocity = {
        horizon: (
            float(prior5[horizon]["return_sigma"])
            - float(prior10[horizon]["return_sigma"])
        )
        / 5.0
        for horizon in _HORIZONS
    }
    acceleration = {
        horizon: (velocity[horizon] - prior_velocity[horizon]) / 5.0
        for horizon in _HORIZONS
    }
    pattern = "/".join(directions[horizon] for horizon in (5, 21, 63, 84))
    return {
        "as_of_day": rows[end - 1].day.isoformat(),
        "directions": {str(horizon): value for horizon, value in directions.items()},
        "pattern": pattern,
        "soft_direction": directions[5],
        "fast_direction": directions[21],
        "mid_direction": directions[63],
        "hard_direction": directions[84],
        "transition": (
            f"transition_{directions[21]}"
            if directions[21] != directions[84]
            else f"aligned_{directions[21]}"
        ),
        "windows": {str(horizon): value for horizon, value in windows.items()},
        "return_velocity": {str(horizon): value for horizon, value in velocity.items()},
        "return_acceleration": {
            str(horizon): value for horizon, value in acceleration.items()
        },
        "tr5_median": median_tr[5],
        "tr21_median": median_tr[21],
        "tr84_median": median_tr[84],
        "tr_velocity": tr_velocity,
        "tr_acceleration": tr_acceleration,
        "tr_phase": (
            f"{'high' if median_tr[5] > median_tr[21] else 'low'}_{_sign(tr_velocity)}"
        ),
        "gap_bias": (
            "down"
            if negative_gaps > positive_gaps
            else "up"
            if positive_gaps > negative_gaps
            else "balanced"
        ),
        "drawdown_84": drawdown,
        "drawdown_velocity": drawdown_velocity,
        "drawdown_acceleration": (drawdown_velocity - prior_drawdown_velocity) / 5.0,
        "damage_phase": _sign(drawdown_velocity),
    }


def _contexts_by_end(
    rows: Sequence[XspDailyBar],
) -> dict[int, dict[str, object]]:
    output: dict[int, dict[str, object]] = {}
    prior: Mapping[str, object] | None = None
    ages = {
        "pattern_age": 0,
        "hard_age": 0,
        "fast_hard_age": 0,
        "tr_phase_age": 0,
    }
    comparisons = {
        "pattern_age": "pattern",
        "hard_age": "hard_direction",
        "fast_hard_age": "transition",
        "tr_phase_age": "tr_phase",
    }
    for end in range(XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS, len(rows) + 1):
        current = _daily_context(rows, end)
        for age_name, field in comparisons.items():
            ages[age_name] = (
                ages[age_name] + 1
                if prior is not None and prior[field] == current[field]
                else 1
            )
            current[age_name] = ages[age_name]
            current[f"{age_name}_class"] = _age_class(ages[age_name])
        output[end] = current
        prior = current
    return output


def _velocity_votes(
    snapshot: SpotSignalSnapshot,
    horizons: Sequence[int],
    direction: str,
) -> int:
    impulse = snapshot.directional_impulse
    if impulse is None:
        return 0
    by_horizon = {int(row.bars): row for row in impulse.horizons}
    return sum(
        horizon in by_horizon
        and by_horizon[horizon].slope_velocity_pct_per_bar is not None
        and (
            float(by_horizon[horizon].slope_velocity_pct_per_bar) > 0.0
            if direction == "up"
            else float(by_horizon[horizon].slope_velocity_pct_per_bar) < 0.0
        )
        for horizon in horizons
    )


def _project(
    snapshot: SpotSignalSnapshot,
    *,
    direction: str,
    mechanism: str,
    admitted: bool,
) -> SpotSignalSnapshot:
    marker = f"opening_edge_v3:{mechanism}:{'pass' if admitted else 'armed'}"
    return replace(
        snapshot,
        entry_dir=direction if admitted else None,
        entry_proposed_dir=direction,
        entry_blocked_by=None if admitted else marker,
        entry_controls=(*snapshot.entry_controls, marker),
    )


class XspOpeningEdgeV3StateOwner:
    """One causal owner for v3 admission maturation, financing, and surrender."""

    def __init__(self, daily_bars: Sequence[XspDailyBar]) -> None:
        rows = validate_xsp_daily_bars(daily_bars)
        self._daily = rows
        self._days = tuple(row.day for row in rows)
        self._contexts = _contexts_by_end(rows)
        self._flip_pending: dict[int, dict[str, object]] = {}
        self._counts: Counter[str] = Counter()
        self._events: list[dict[str, object]] = []
        payload = [{**asdict(row), "day": row.day.isoformat()} for row in rows]
        self.context_fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def context_for_day(self, day: date) -> Mapping[str, object] | None:
        return self._contexts.get(bisect_left(self._days, day))

    @staticmethod
    def _soft_wakeup(context: Mapping[str, object]) -> bool:
        windows = context["windows"]
        magnitudes = {
            horizon: abs(float(windows[str(horizon)]["return_sigma"]))
            for horizon in (5, 21, 84)
        }
        hottest = max(magnitudes, key=magnitudes.get)
        transition_heat = (
            f"{'aligned' if context['fast_direction'] == context['hard_direction'] else 'transition'}_"
            f"{'hot' if hottest in (5, 21) else 'cold'}"
        )
        return bool(
            context["soft_direction"] == "down"
            and context["fast_direction"] == "up"
            and context["hard_direction"] == "up"
            and transition_heat == "aligned_cold"
            and context["damage_phase"] == "down"
        )

    @staticmethod
    def _corroboration(
        snapshot: SpotSignalSnapshot,
        context: Mapping[str, object],
        direction: str,
    ) -> int:
        velocity = context["return_velocity"]
        flags = (
            _velocity_votes(snapshot, _FAST_HORIZONS, direction) >= 2,
            _velocity_votes(snapshot, _SLOW_HORIZONS, direction) >= 1,
            context["soft_direction"] == direction,
            context["hard_direction"] == direction,
            context["damage_phase"] == direction,
            sum(
                _sign(float(velocity[str(horizon)])) == direction
                for horizon in (5, 21, 84)
            )
            >= 2,
            context["gap_bias"] == direction,
        )
        return sum(flags)

    def _rules(self) -> tuple[_MaturationRule, ...]:
        def transition_up(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            return (
                "down"
                if snapshot.entry_dir == "down"
                and context["transition"] == "transition_up"
                else None
            )

        def recovery_up(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            impulse = snapshot.directional_impulse
            return (
                "up"
                if snapshot.entry_dir == "up"
                and context["damage_phase"] == "up"
                and impulse is not None
                and impulse.observed_horizons == 3
                and impulse.coherence is not None
                and 0.0 < float(impulse.coherence) < 1.0
                else None
            )

        def context_confidence(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            velocity = context["return_velocity"]
            impulse = snapshot.directional_impulse
            if (
                snapshot.entry_dir == "up"
                and context["hard_direction"] == "up"
                and float(context["drawdown_velocity"]) < 0.0
                and all(float(velocity[str(horizon)]) < 0.0 for horizon in (5, 21, 84))
            ):
                return "up"
            if (
                snapshot.entry_dir == "down"
                and context["tr_phase"] == "high_up"
                and impulse is not None
                and _velocity_votes(snapshot, _FAST_HORIZONS, "down") == 3
                and _velocity_votes(snapshot, _SLOW_HORIZONS, "down") == 1
                and impulse.atr_acceleration_pct is not None
                and float(impulse.atr_acceleration_pct) > 0.0
            ):
                return "down"
            return None

        def established_fast_short(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            impulse = snapshot.directional_impulse
            return (
                "down"
                if snapshot.entry_dir == "down"
                and context["fast_hard_age_class"] == "established"
                and _velocity_votes(snapshot, _FAST_HORIZONS, "down") == 3
                and _velocity_votes(snapshot, _SLOW_HORIZONS, "down") == 0
                and impulse is not None
                and impulse.coherence is not None
                and float(impulse.coherence) >= 1.0
                else None
            )

        def entrenched_hard_up_short(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            impulse = snapshot.directional_impulse
            return (
                "down"
                if snapshot.entry_dir == "down"
                and context["hard_direction"] == "up"
                and context["hard_age_class"] == "entrenched"
                and _velocity_votes(snapshot, _FAST_HORIZONS, "down") == 2
                and _velocity_votes(snapshot, _SLOW_HORIZONS, "down") == 0
                and impulse is not None
                and impulse.coherence is not None
                and float(impulse.coherence) < 1.0
                and not self._soft_wakeup(context)
                else None
            )

        def zero_corroboration(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            direction = snapshot.entry_dir
            return (
                str(direction)
                if direction in ("up", "down")
                and self._corroboration(snapshot, context, str(direction)) == 0
                else None
            )

        def fading_range_bear(
            snapshot: SpotSignalSnapshot,
            context: Mapping[str, object],
        ) -> str | None:
            return (
                "down"
                if snapshot.entry_dir == "down"
                and context["hard_direction"] == "up"
                and context["soft_direction"] == "down"
                and context["damage_phase"] == "down"
                and all(
                    float(context["return_velocity"][str(horizon)]) < 0.0
                    for horizon in (5, 21, 84)
                )
                and all(
                    float(context["return_acceleration"][str(horizon)]) < 0.0
                    for horizon in (5, 21, 84)
                )
                and float(context["tr_acceleration"]) < 0.0
                else None
            )

        return (
            _MaturationRule("transition_up_short", 3, False, transition_up),
            _MaturationRule("recovery_up", 3, True, recovery_up),
            _MaturationRule(
                "composite_context_confidence",
                3,
                True,
                context_confidence,
            ),
            _MaturationRule(
                "established_fast_only_short",
                3,
                True,
                established_fast_short,
            ),
            _MaturationRule(
                "entrenched_hard_up_short",
                3,
                True,
                entrenched_hard_up_short,
            ),
            _MaturationRule(
                "zero_corroboration",
                3,
                True,
                zero_corroboration,
            ),
            _MaturationRule(
                "fading_range_bear_wakeup",
                2,
                True,
                fading_range_bear,
            ),
        )

    def _project_rule(
        self,
        prepared: PreparedSpotEvaluatorTape,
        bars: Sequence[Bar | None],
        rule: _MaturationRule,
    ) -> PreparedSpotEvaluatorTape:
        output: list[SpotSignalSnapshot | None] = []
        pending: dict[str, object] | None = None
        current_day: date | None = None
        for index, (bar, snapshot) in enumerate(zip(bars, prepared.signals)):
            if snapshot is None:
                output.append(None)
                continue
            if bar is None:
                raise ValueError("Opening Edge v3 signal alignment is missing")
            day = xsp_bar_trading_date(bar.ts)
            if day != current_day:
                if pending is not None:
                    self._events.append({**pending, "result": "session_expired"})
                pending = None
                current_day = day
            raw = (
                str(snapshot.entry_dir)
                if snapshot.entry_dir in ("up", "down")
                else None
            )
            if pending is not None and raw is not None and raw != pending["direction"]:
                self._counts[f"{rule.name}:cancelled_by_opposite"] += 1
                self._events.append({**pending, "result": "cancelled_by_opposite"})
                pending = None
                output.append(snapshot)
                continue
            matured = False
            if pending is not None and index - int(pending["index"]) == 1:
                direction = str(pending["direction"])
                price_pass = (
                    float(bar.close) > float(pending["close"])
                    if direction == "up"
                    else float(bar.close) < float(pending["close"])
                )
                source_pass = (
                    not rule.require_source
                    or snapshot.lifecycle_inputs().get("signal_source_dir") == direction
                )
                votes = _velocity_votes(
                    snapshot,
                    _FAST_HORIZONS,
                    direction,
                )
                velocity_pass = votes >= rule.required_fast_votes
                if price_pass and source_pass and velocity_pass:
                    output.append(
                        _project(
                            snapshot,
                            direction=direction,
                            mechanism=rule.name,
                            admitted=True,
                        )
                    )
                    self._counts[f"{rule.name}:matured"] += 1
                    self._events.append(
                        {
                            **pending,
                            "confirmation_time": snapshot.bar_ts.isoformat(),
                            "fast_votes": votes,
                            "result": "matured",
                        }
                    )
                    matured = True
                else:
                    self._counts[f"{rule.name}:expired"] += 1
                    self._events.append(
                        {
                            **pending,
                            "confirmation_time": snapshot.bar_ts.isoformat(),
                            "fast_votes": votes,
                            "price_pass": price_pass,
                            "source_pass": source_pass,
                            "result": "expired",
                        }
                    )
                pending = None
            if matured:
                continue
            context = self.context_for_day(day) if day is not None else None
            direction = rule.scope(snapshot, context) if context is not None else None
            if direction is None:
                output.append(snapshot)
                continue
            pending = {
                "index": index,
                "close": float(bar.close),
                "proposal_time": snapshot.bar_ts.isoformat(),
                "trading_day": day.isoformat(),
                "direction": direction,
                "mechanism": rule.name,
            }
            output.append(
                _project(
                    snapshot,
                    direction=direction,
                    mechanism=rule.name,
                    admitted=False,
                )
            )
            self._counts[f"{rule.name}:armed"] += 1
        if pending is not None:
            self._events.append({**pending, "result": "tape_expired"})
        return replace(prepared, signals=tuple(output))

    def project_evaluator_tape(
        self,
        prepared: PreparedSpotEvaluatorTape,
        bars: Sequence[Bar],
        *,
        sig_idx_by_exec_idx: Sequence[int] | None = None,
    ) -> PreparedSpotEvaluatorTape:
        aligned_bars: tuple[Bar | None, ...] = (
            tuple(
                bars[index] if 0 <= int(index) < len(bars) else None
                for index in sig_idx_by_exec_idx
            )
            if sig_idx_by_exec_idx is not None
            else tuple(bars)
        )
        if len(prepared.signals) != len(aligned_bars):
            raise ValueError("Opening Edge v3 signal tape alignment mismatch")
        if any(bars[index].ts <= bars[index - 1].ts for index in range(1, len(bars))):
            raise ValueError("Opening Edge v3 requires ordered unique bars")
        self._flip_pending.clear()
        self._counts.clear()
        self._events.clear()
        phase_atr_engines = {
            phase: SupertrendEngine(
                atr_period=14,
                multiplier=1.0,
                source="hl2",
            )
            for phase in ("down", "flat")
        }
        phase_signals = []
        for bar, snapshot in zip(aligned_bars, prepared.signals):
            if bar is None:
                if snapshot is not None:
                    raise ValueError("Opening Edge v3 signal alignment is missing")
                phase_signals.append(None)
                continue
            day = xsp_bar_trading_date(bar.ts)
            context = self.context_for_day(day) if day is not None else None
            phase = str(context["damage_phase"]) if context is not None else None
            phase_atr = None
            if phase in phase_atr_engines:
                state = phase_atr_engines[phase].update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )
                if state.ready and state.atr is not None:
                    phase_atr = float(state.atr)
            phase_signals.append(
                replace(snapshot, atr=phase_atr) if snapshot is not None else None
            )
        projected = replace(prepared, signals=tuple(phase_signals))
        for rule in self._rules():
            projected = self._project_rule(projected, aligned_bars, rule)
        output = []
        for bar, snapshot in zip(aligned_bars, projected.signals):
            if bar is None:
                if snapshot is not None:
                    raise ValueError("Opening Edge v3 signal alignment is missing")
                output.append(None)
                continue
            day = xsp_bar_trading_date(bar.ts)
            context = self.context_for_day(day) if day is not None else None
            if (
                snapshot is not None
                and snapshot.entry_dir in ("up", "down")
                and (day is None or context is None)
            ):
                output.append(
                    _project(
                        snapshot,
                        direction=str(snapshot.entry_dir),
                        mechanism="context_underwarmed",
                        admitted=False,
                    )
                )
                self._counts["context_underwarmed:blocked"] += 1
            elif (
                snapshot is not None
                and snapshot.entry_dir in ("up", "down")
                and context is not None
                and context["damage_phase"] in {"down", "flat"}
                and snapshot.atr is None
            ):
                output.append(
                    _project(
                        snapshot,
                        direction=str(snapshot.entry_dir),
                        mechanism="short_financing_atr_underwarmed",
                        admitted=False,
                    )
                )
                self._counts["short_financing_atr_underwarmed:blocked"] += 1
            else:
                output.append(snapshot)
        return replace(projected, signals=tuple(output))

    def excursion_policy_for_trade(
        self,
        trade: SpotTrade,
    ) -> SpotExcursionPolicy | None:
        day = xsp_bar_trading_date(trade.entry_time)
        context = self.context_for_day(day) if day is not None else None
        if (
            int(trade.qty) < 0
            and context is not None
            and context["damage_phase"] in {"down", "flat"}
        ):
            self._counts["short_fizzle:enabled"] += 1
            return _FIZZLE_POLICY
        if context is not None and context["damage_phase"] in {"down", "flat"}:
            return _NO_EXCURSION_POLICY
        return None

    @staticmethod
    def _long_agrees(context: Mapping[str, object]) -> bool:
        windows = context["windows"]
        values = [float(windows[horizon]["return"]) for horizon in ("21", "84")]
        return all(value > 0.0 for value in values) or all(
            value < 0.0 for value in values
        )

    @staticmethod
    def _weak_surrender(
        snapshot: SpotSignalSnapshot,
        context: Mapping[str, object],
        incumbent: str,
    ) -> bool:
        impulse = snapshot.directional_impulse
        if impulse is None:
            return False
        sign = 1.0 if incumbent == "up" else -1.0
        horizons = {int(row.bars): row for row in impulse.horizons}

        def aligned_velocity(horizon: int) -> float | None:
            row = horizons.get(horizon)
            value = row.slope_velocity_pct_per_bar if row is not None else None
            return sign * float(value) if value is not None else None

        fast = [
            value
            for horizon in _FAST_HORIZONS
            if (value := aligned_velocity(horizon)) is not None
        ]
        slow = [
            value
            for horizon in _SLOW_HORIZONS
            if (value := aligned_velocity(horizon)) is not None
        ]
        long_votes = sum(
            sign * float(context["windows"][horizon]["return"]) > 0.0
            for horizon in ("21", "84")
        )
        atr_velocity = impulse.atr_velocity_pct
        return bool(
            long_votes >= 1
            and sum(value > 0.0 for value in slow) >= 1
            and sum(value < 0.0 for value in fast) <= 1
            and atr_velocity is not None
            and float(atr_velocity) > 0.0
        )

    def resolve_flip(
        self,
        *,
        trade: SpotTrade,
        bar: Bar,
        snapshot: SpotSignalSnapshot | None,
        hit: bool,
    ) -> bool:
        key = id(trade)
        for stale in tuple(self._flip_pending):
            if stale != key:
                self._flip_pending.pop(stale, None)
        if not hit:
            pending = self._flip_pending.pop(key, None)
            if pending is not None:
                self._counts[f"{pending['owner']}:cancelled"] += 1
            return False
        if key in self._flip_pending:
            pending = self._flip_pending[key]
            observed = int(pending["observed"]) + 1
            if observed >= int(pending["required"]):
                self._counts[f"{pending['owner']}:released"] += 1
                self._flip_pending.pop(key, None)
                return True
            pending["observed"] = observed
            self._counts[f"{pending['owner']}:held_confirmation"] += 1
            return False
        day = xsp_bar_trading_date(bar.ts)
        context = self.context_for_day(day) if day is not None else None
        incumbent = "up" if int(trade.qty) > 0 else "down"
        if (
            snapshot is not None
            and context is not None
            and self._weak_surrender(snapshot, context, incumbent)
        ):
            owner = "existing_weak_two_bar"
            self._flip_pending[key] = {
                "owner": owner,
                "observed": 0,
                "required": 2,
            }
            self._counts[f"{owner}:armed"] += 1
            return False
        if (
            context is not None
            and context["damage_phase"] in {"up", "flat"}
            and self._long_agrees(context)
        ):
            owner = "recovering_flat_long_context_one_bar"
            self._flip_pending[key] = {
                "owner": owner,
                "observed": 0,
                "required": 1,
            }
            self._counts[f"{owner}:armed"] += 1
            return False
        self._counts["ordinary_flip"] += 1
        return True

    def state_payload(self) -> dict[str, object]:
        return {
            "schema": "xsp.opening-edge-v3-state-owner.v1",
            "context_sessions": len(self._daily),
            "context_fingerprint": self.context_fingerprint,
            "projection_and_lifecycle_counts": dict(sorted(self._counts.items())),
            "projection_events": tuple(self._events),
            "pending_flip_count": len(self._flip_pending),
            "order_authority": "none",
        }
