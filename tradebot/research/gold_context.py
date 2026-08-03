"""Causal completed-bar context shared by gold research and runtime replay."""

from __future__ import annotations

import bisect
import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from ..engines.signals import EmaDecisionEngine


GOLD_MACRO_HORIZONS = (5, 21, 63)


def gold_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def gold_finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _field(row: object, name: str) -> object:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name)


def gold_bar_time(row: object) -> datetime:
    for name in ("end", "ts"):
        try:
            value = _field(row, name)
        except (AttributeError, KeyError):
            continue
        if value is not None:
            return gold_utc(value)
    raise ValueError("gold bars require an end or ts timestamp")


def gold_bar_value(row: object, name: str) -> float:
    value = gold_finite(_field(row, name))
    if value is None:
        raise ValueError(f"gold bar has invalid {name}")
    return value


def gold_complete_bars(
    rows: Sequence[object], *, as_of: datetime
) -> list[object]:
    cutoff = gold_utc(as_of)
    ordered = sorted(rows, key=gold_bar_time)
    if any(
        gold_bar_time(ordered[index]) <= gold_bar_time(ordered[index - 1])
        for index in range(1, len(ordered))
    ):
        raise ValueError("gold bars must have ordered unique timestamps")
    return [row for row in ordered if gold_bar_time(row) <= cutoff]


def gold_latest_index(times: Sequence[datetime], at: datetime) -> int:
    return bisect.bisect_right(times, gold_utc(at)) - 1


def gold_macro_timeline(
    uup_rows: Sequence[object],
    tip_rows: Sequence[object],
    *,
    as_of: datetime,
) -> list[dict[str, object]]:
    """Return completed common UUP/TIP states in gold-oriented coordinates."""

    sources = {
        "UUP": gold_complete_bars(uup_rows, as_of=as_of),
        "TIP": gold_complete_bars(tip_rows, as_of=as_of),
    }
    by_time = {
        symbol: {gold_bar_time(row): row for row in rows}
        for symbol, rows in sources.items()
    }
    common = sorted(set(by_time["UUP"]).intersection(by_time["TIP"]))
    closes = {
        symbol: [gold_bar_value(by_time[symbol][stamp], "close") for stamp in common]
        for symbol in sources
    }
    ages = {horizon: {"state": None, "age": 0} for horizon in GOLD_MACRO_HORIZONS}
    output: list[dict[str, object]] = []
    for index, stamp in enumerate(common):
        horizons: dict[str, object] = {}
        for horizon in GOLD_MACRO_HORIZONS:
            if index < horizon + 2:
                continue
            symbols: dict[str, object] = {}
            votes = {"direction": 0, "velocity": 0, "acceleration": 0}
            for symbol, values in closes.items():
                multiplier = -1.0 if symbol == "UUP" else 1.0
                displacement = multiplier * (
                    values[index] / values[index - horizon] - 1.0
                )
                prior_displacement = multiplier * (
                    values[index - 1] / values[index - horizon - 1] - 1.0
                )
                older_displacement = multiplier * (
                    values[index - 2] / values[index - horizon - 2] - 1.0
                )
                velocity = displacement - prior_displacement
                acceleration = velocity - (prior_displacement - older_displacement)
                symbols[symbol] = {
                    "gold_oriented_displacement": displacement,
                    "gold_oriented_velocity": velocity,
                    "gold_oriented_acceleration": acceleration,
                    "direction": (
                        "supportive"
                        if displacement > 0.0
                        else "adverse"
                        if displacement < 0.0
                        else "flat"
                    ),
                }
                votes["direction"] += displacement > 0.0
                votes["velocity"] += velocity > 0.0
                votes["acceleration"] += acceleration > 0.0

            def label(count: int) -> str:
                return (
                    "supportive"
                    if count == 2
                    else "adverse"
                    if count == 0
                    else "mixed"
                )

            direction = label(votes["direction"])
            state = ages[horizon]
            state["age"] = int(state["age"]) + 1 if state["state"] == direction else 1
            state["state"] = direction
            horizons[str(horizon)] = {
                "direction": direction,
                "velocity": label(votes["velocity"]),
                "acceleration": label(votes["acceleration"]),
                "state_age": state["age"],
                "symbols": symbols,
            }
        if len(horizons) == len(GOLD_MACRO_HORIZONS):
            output.append({"end": stamp, "horizons": horizons})
    return output


def gold_daily_timeline(
    rows: Sequence[object], *, as_of: datetime
) -> list[dict[str, object]]:
    """Return the completed D1 hard/soft state and volatility phase."""

    hard = EmaDecisionEngine(ema_preset="21/50", ema_entry_mode="trend")
    soft = EmaDecisionEngine(ema_preset="8/21", ema_entry_mode="trend")
    state = None
    age = 0
    prior_close = None
    prior_atr14 = None
    ranges: deque[float] = deque(maxlen=63)
    output: list[dict[str, object]] = []
    for bar in gold_complete_bars(rows, as_of=as_of):
        close = gold_bar_value(bar, "close")
        hard_snapshot = hard.update(close)
        soft_snapshot = soft.update(close)
        current = hard_snapshot.state if hard_snapshot.ema_ready else None
        age = 0 if current is None else age + 1 if current == state else 1
        state = current
        high, low = gold_bar_value(bar, "high"), gold_bar_value(bar, "low")
        true_range = high - low
        if prior_close is not None:
            true_range = max(
                true_range,
                abs(high - prior_close),
                abs(low - prior_close),
            )
        prior_close = close
        ranges.append(true_range)
        atr14 = statistics.fmean(list(ranges)[-14:]) if len(ranges) >= 14 else None
        atr63 = statistics.fmean(ranges) if len(ranges) >= 63 else None
        velocity = (
            (atr14 - prior_atr14) / close
            if atr14 is not None and prior_atr14 is not None
            else None
        )
        if atr14 is not None:
            prior_atr14 = atr14
        ratio = atr14 / atr63 if atr14 is not None and atr63 else None
        output.append(
            {
                "end": gold_bar_time(bar),
                "hard_direction": current,
                "hard_age": age if current is not None else None,
                "soft_direction": (
                    soft_snapshot.state if soft_snapshot.ema_ready else None
                ),
                "atr14": atr14,
                "atr_ratio_14_63": ratio,
                "atr_velocity": velocity,
                "high_contracting": bool(
                    ratio is not None
                    and ratio >= 1.0
                    and velocity is not None
                    and velocity <= 0.0
                ),
            }
        )
    return output


def gold_h4_timeline(
    rows: Sequence[object], *, as_of: datetime
) -> list[dict[str, object]]:
    """Return one exact H4 signal/curvature row for every completed bar."""

    engine = EmaDecisionEngine(
        ema_preset="8/21",
        ema_entry_mode="cross",
        entry_confirm_bars=1,
    )
    closes: list[float] = []
    states: list[str | None] = []
    ranges: deque[float] = deque(maxlen=63)
    prior_close = None
    prior_atr14 = None
    prior_fast_slope = None
    prior_spread = None
    prior_fast_slope_pct = None
    prior_spread_pct = None
    output: list[dict[str, object]] = []
    for bar in gold_complete_bars(rows, as_of=as_of):
        stamp = gold_bar_time(bar)
        close = gold_bar_value(bar, "close")
        snapshot = engine.update(close)
        state = snapshot.state if snapshot.ema_ready else None
        high, low = gold_bar_value(bar, "high"), gold_bar_value(bar, "low")
        true_range = high - low
        if prior_close is not None:
            true_range = max(
                true_range,
                abs(high - prior_close),
                abs(low - prior_close),
            )
        prior_close = close
        ranges.append(true_range)
        atr14 = statistics.fmean(list(ranges)[-14:]) if len(ranges) >= 14 else None
        atr63 = statistics.fmean(ranges) if len(ranges) >= 63 else None
        atr_velocity = (
            atr14 - prior_atr14
            if atr14 is not None and prior_atr14 is not None
            else None
        )
        if atr14 is not None:
            prior_atr14 = atr14
        fast_slope = (
            float(snapshot.ema_fast) - float(snapshot.prev_ema_fast)
            if snapshot.ema_fast is not None and snapshot.prev_ema_fast is not None
            else None
        )
        spread = (
            float(snapshot.ema_fast) - float(snapshot.ema_slow)
            if snapshot.ema_fast is not None and snapshot.ema_slow is not None
            else None
        )
        spread_velocity = (
            spread - prior_spread
            if spread is not None and prior_spread is not None
            else None
        )
        fast_acceleration = (
            fast_slope - prior_fast_slope
            if fast_slope is not None and prior_fast_slope is not None
            else None
        )
        if fast_slope is not None:
            prior_fast_slope = fast_slope
        if spread is not None:
            prior_spread = spread
        fast_slope_pct = (
            fast_slope / close * 100.0 if fast_slope is not None else None
        )
        spread_pct = spread / close * 100.0 if spread is not None else None
        spread_velocity_pct = (
            spread_pct - prior_spread_pct
            if spread_pct is not None and prior_spread_pct is not None
            else None
        )
        fast_acceleration_pct = (
            fast_slope_pct - prior_fast_slope_pct
            if fast_slope_pct is not None and prior_fast_slope_pct is not None
            else None
        )
        if fast_slope_pct is not None:
            prior_fast_slope_pct = fast_slope_pct
        if spread_pct is not None:
            prior_spread_pct = spread_pct
        closes.append(close)
        states.append(state)
        sign = 1.0 if state == "up" else -1.0 if state == "down" else None
        path: dict[str, object] = {}
        for horizon in (12, 30, 60):
            if sign is None or len(closes) <= horizon:
                continue
            window_closes = closes[-(horizon + 1) :]
            window_states = states[-horizon:]
            travelled = sum(
                abs(right / left - 1.0)
                for left, right in zip(
                    window_closes,
                    window_closes[1:],
                    strict=False,
                )
            )
            path[str(horizon)] = {
                "signed_return": sign * (close / window_closes[0] - 1.0),
                "efficiency": (
                    abs(close / window_closes[0] - 1.0) / travelled
                    if travelled > 0.0
                    else None
                ),
                "direction_occupancy": sum(value == state for value in window_states)
                / len(window_states),
                "flip_count": sum(
                    left in ("up", "down")
                    and right in ("up", "down")
                    and left != right
                    for left, right in zip(
                        window_states,
                        window_states[1:],
                        strict=False,
                    )
                ),
            }
        output.append(
            {
                "end": stamp,
                "close": close,
                "raw_direction": state,
                "raw_turn": (
                    "up"
                    if snapshot.cross_up
                    else "down"
                    if snapshot.cross_down
                    else None
                ),
                "proposed_direction": snapshot.entry_dir,
                "fast_slope_dollars": fast_slope,
                "spread_velocity_dollars": spread_velocity,
                "fast_acceleration_dollars": fast_acceleration,
                "fast_slope_pct": fast_slope_pct,
                "spread_pct": spread_pct,
                "spread_velocity_pct": spread_velocity_pct,
                "fast_acceleration_pct": fast_acceleration_pct,
                "atr14_dollars": atr14,
                "atr_ratio_14_63": (
                    atr14 / atr63
                    if atr14 is not None and atr63 is not None and atr63 > 0.0
                    else None
                ),
                "atr_velocity_dollars": atr_velocity,
                "path": path,
            }
        )
    return output
