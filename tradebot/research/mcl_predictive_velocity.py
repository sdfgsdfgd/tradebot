"""Outcome-blind Stage-89 velocity-jerk and CL-to-MCL handoff morphology."""

from __future__ import annotations

from collections.abc import Mapping
import math

from .mcl_predictive_onset import MCL_PREDICTIVE_ONSET_AUTHORITY


MCL_VELOCITY_JERK_VERSION = "mcl.predictive-velocity-jerk-handoff.v1"
MCL_VELOCITY_JERK_AUTHORITY = (
    "prospective_derived_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_VELOCITY_JERK_INTERVALS = (
    "closing_baseline_60_30s",
    "closing_acceleration_30_15s",
    "closing_commitment_15_5s",
    "closing_trigger_5_0s",
    "spark_0_5s",
    "acceptance_5_15s",
    "propagation_15_30s",
    "persistence_30_60s",
)


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("MCL velocity-jerk evidence must be finite")
    return result


def _plain_shape(value: float) -> str:
    return "UP" if value > 0.0 else "DOWN" if value < 0.0 else "FLAT"


def project_mcl_velocity_jerk_handoff(
    event: Mapping[str, object],
) -> dict[str, object]:
    """Derive the frozen Stage-89 cross-book ignition law without outcomes."""

    if event.get("authority") != MCL_PREDICTIVE_ONSET_AUTHORITY:
        raise ValueError("MCL velocity-jerk event authority drifted")
    if event.get("outcomes_exposed") is not False or event.get("submitted_orders") != 0:
        raise ValueError("MCL velocity-jerk input opened forbidden authority")
    windows = event.get("windows")
    if not isinstance(windows, Mapping) or set(MCL_VELOCITY_JERK_INTERVALS) - set(
        windows
    ):
        raise ValueError("MCL velocity-jerk interval contract drifted")

    books: dict[str, dict[str, object]] = {}
    alignment_index: dict[str, int | None] = {}
    alignment_status: dict[str, str] = {}
    for symbol in ("CL", "MCL"):
        intervals: dict[str, dict[str, object]] = {}
        previous: dict[str, float | None] | None = None
        for name in MCL_VELOCITY_JERK_INTERVALS:
            window = windows[name]
            if not isinstance(window, Mapping):
                raise ValueError("MCL velocity-jerk window is invalid")
            window_books = window.get("books")
            if not isinstance(window_books, Mapping) or not isinstance(
                window_books.get(symbol), Mapping
            ):
                raise ValueError("MCL velocity-jerk book evidence is missing")
            source = window_books[symbol]

            def metric(key: str) -> float | None:
                value = source.get(key)
                return None if value is None else _finite(value)

            current = {
                "directional_velocity": metric(
                    "directional_microprice_slope_velocity_ticks_per_second2"
                ),
                "tr_velocity": metric("microprice_tr_velocity_ticks"),
                "quote_acceleration": metric("quote_intensity_acceleration"),
                "spread_elasticity": metric("spread_last_minus_first_ticks"),
                "directional_displacement": metric(
                    "directional_microprice_displacement_ticks"
                ),
            }

            def change(key: str) -> float | None:
                if previous is None or previous[key] is None or current[key] is None:
                    return None
                return float(current[key]) - float(previous[key])

            velocity_jerk = change("directional_velocity")
            tr_acceleration = change("tr_velocity")
            quote_jerk = change("quote_acceleration")
            spread = current["spread_elasticity"]
            aligned = (
                None
                if current["directional_displacement"] is None
                or current["directional_velocity"] is None
                else current["directional_displacement"] > 0.0
                and current["directional_velocity"] > 0.0
            )
            intervals[name] = {
                **current,
                "velocity_jerk": velocity_jerk,
                "velocity_jerk_sign": (
                    _plain_shape(velocity_jerk)
                    if velocity_jerk is not None
                    else "UNRESOLVED"
                ),
                "tr_acceleration": tr_acceleration,
                "tr_acceleration_sign": (
                    _plain_shape(tr_acceleration)
                    if tr_acceleration is not None
                    else "UNRESOLVED"
                ),
                "quote_jerk": quote_jerk,
                "quote_jerk_sign": (
                    _plain_shape(quote_jerk)
                    if quote_jerk is not None
                    else "UNRESOLVED"
                ),
                "spread_failure": (
                    None
                    if spread is None or velocity_jerk is None
                    else spread > 0.0 and velocity_jerk <= 0.0
                ),
                "aligned": aligned,
            }
            previous = current

        first = None
        unresolved_before_alignment = False
        for index, name in enumerate(MCL_VELOCITY_JERK_INTERVALS[1:], start=1):
            aligned = intervals[name]["aligned"]
            if aligned is None:
                unresolved_before_alignment = True
            elif aligned:
                if not unresolved_before_alignment:
                    first = index
                break
        status = (
            "ALIGNED"
            if first is not None
            else "UNRESOLVED_EARLY_GAP"
            if unresolved_before_alignment
            else "NO_ALIGNMENT"
        )
        alignment_index[symbol] = first
        alignment_status[symbol] = status
        books[symbol] = {
            "first_alignment_interval": (
                MCL_VELOCITY_JERK_INTERVALS[first] if first is not None else None
            ),
            "alignment_status": status,
            "intervals": intervals,
            "sequence_shape": {
                "velocity_jerk": [
                    intervals[name]["velocity_jerk_sign"]
                    for name in MCL_VELOCITY_JERK_INTERVALS
                ],
                "tr_acceleration": [
                    intervals[name]["tr_acceleration_sign"]
                    for name in MCL_VELOCITY_JERK_INTERVALS
                ],
                "quote_jerk": [
                    intervals[name]["quote_jerk_sign"]
                    for name in MCL_VELOCITY_JERK_INTERVALS
                ],
            },
        }

    cl_index, mcl_index = alignment_index["CL"], alignment_index["MCL"]
    handoff = (
        "UNRESOLVED"
        if cl_index is None or mcl_index is None
        else "CL_LEADS"
        if cl_index < mcl_index
        else "MCL_LEADS"
        if mcl_index < cl_index
        else "SAME_INTERVAL"
    )
    trigger = MCL_VELOCITY_JERK_INTERVALS.index("closing_trigger_5_0s")
    acceptance = MCL_VELOCITY_JERK_INTERVALS.index("acceptance_5_15s")
    propagation = MCL_VELOCITY_JERK_INTERVALS.index("propagation_15_30s")

    def row(symbol: str, index: int) -> Mapping[str, object]:
        return books[symbol]["intervals"][MCL_VELOCITY_JERK_INTERVALS[index]]

    spread_checks = [
        row(symbol, index)["spread_failure"]
        for symbol in ("CL", "MCL")
        for index in range(trigger, acceptance + 1)
    ]
    no_spread_failure = (
        None
        if any(value is None for value in spread_checks)
        else not any(spread_checks)
    )
    continuity: dict[str, bool | None] = {}
    for symbol in ("CL", "MCL"):
        first = alignment_index[symbol]
        if first is None:
            continuity[symbol] = None
            continue
        values = [
            row(symbol, index)["directional_velocity"]
            for index in range(first, propagation + 1)
        ]
        continuity[symbol] = (
            None
            if any(value is None for value in values)
            else all(float(value) >= 0.0 for value in values)
        )
    no_reversal = (
        None
        if any(value is None for value in continuity.values())
        else all(continuity.values())
    )
    mcl_acceptance = row("MCL", acceptance)
    mcl_acceptance_jerk = mcl_acceptance["velocity_jerk"]
    mcl_acceptance_adverse = (
        None
        if mcl_acceptance["directional_displacement"] is None
        or mcl_acceptance["directional_velocity"] is None
        else float(mcl_acceptance["directional_displacement"]) < 0.0
        or float(mcl_acceptance["directional_velocity"]) < 0.0
    )
    ordered_operands = (
        handoff in {"CL_LEADS", "SAME_INTERVAL"},
        None if mcl_acceptance_jerk is None else float(mcl_acceptance_jerk) >= 0.0,
        no_spread_failure,
        no_reversal,
    )
    ordered = (
        None
        if any(value is None for value in ordered_operands)
        else all(ordered_operands)
    )
    refusal = (
        None
        if alignment_status["CL"] == "UNRESOLVED_EARLY_GAP"
        else False
        if alignment_status["CL"] != "ALIGNED"
        else True
        if alignment_status["MCL"] == "NO_ALIGNMENT"
        else None
        if alignment_status["MCL"] == "UNRESOLVED_EARLY_GAP"
        or mcl_acceptance_adverse is None
        else bool(mcl_acceptance_adverse)
    )
    whipsaw_by_book: dict[str, bool | None] = {}
    for symbol in ("CL", "MCL"):
        jerks = [row(symbol, index)["velocity_jerk"] for index in range(trigger, acceptance + 1)]
        whipsaw_by_book[symbol] = (
            None
            if any(value is None for value in jerks)
            else all(float(value) != 0.0 for value in jerks)
            and float(jerks[0]) * float(jerks[1]) < 0.0
            and float(jerks[1]) * float(jerks[2]) < 0.0
        )
    whipsaw = (
        None
        if all(value is None for value in whipsaw_by_book.values())
        else any(value is True for value in whipsaw_by_book.values())
    )
    exhaustion_by_book: dict[str, bool | None] = {}
    for symbol in ("CL", "MCL"):
        trigger_row = row(symbol, trigger)
        trigger_aligned = trigger_row["aligned"]
        exhausted: list[bool | None] = []
        for index in (trigger + 1, acceptance):
            current = row(symbol, index)
            exhausted.append(
                None
                if current["velocity_jerk"] is None
                or current["tr_acceleration"] is None
                or current["spread_elasticity"] is None
                else float(current["velocity_jerk"]) < 0.0
                and (
                    float(current["tr_acceleration"]) > 0.0
                    or float(current["spread_elasticity"]) > 0.0
                )
            )
        exhaustion_by_book[symbol] = (
            None
            if trigger_aligned is None or all(value is None for value in exhausted)
            else bool(trigger_aligned) and any(value is True for value in exhausted)
        )
    exhaustion = (
        None
        if all(value is None for value in exhaustion_by_book.values())
        else any(value is True for value in exhaustion_by_book.values())
    )

    return {
        "schema": MCL_VELOCITY_JERK_VERSION,
        "authority": MCL_VELOCITY_JERK_AUTHORITY,
        "raw_turn_at_utc": event["raw_turn_at_utc"],
        "raw_direction": event["raw_direction"],
        "generation_sha256": event["generation_sha256"],
        "frozen_interval_order": list(MCL_VELOCITY_JERK_INTERVALS),
        "books": books,
        "handoff": handoff,
        "hypotheses": {
            "ORDERED_IGNITION": {
                "matched": ordered,
                "mcl_acceptance_jerk_nonnegative": ordered_operands[1],
                "no_trigger_to_acceptance_spread_failure": no_spread_failure,
                "no_directional_velocity_reversal_through_propagation": no_reversal,
                "continuity_by_book": continuity,
            },
            "TRANSPORT_REFUSAL": {
                "matched": refusal,
                "mcl_acceptance_adverse": mcl_acceptance_adverse,
            },
            "TRANSPORT_NOISE": {"matched": handoff == "MCL_LEADS"},
            "WHIPSAW": {"matched": whipsaw, "by_book": whipsaw_by_book},
            "EXHAUSTION": {
                "matched": exhaustion,
                "by_book": exhaustion_by_book,
            },
        },
        "winner": None,
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
