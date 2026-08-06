"""Shared cutoff-bound accounting for one-contract futures canaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..live.capital import validate_live_capital_decision
from .live_graduation import evidence_sha256


RiskProjection = Callable[[Mapping[str, object]], Mapping[str, object]]
CoverageSignature = Callable[[Mapping[str, object]], object]
EvaluationSlots = Callable[[datetime, datetime, bool], tuple[datetime, ...]]
NaturalSlot = Callable[[datetime], bool]
HeldDirection = Callable[[float], object]
ExcludedSlots = Callable[[datetime, datetime], int]


FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA = (
    "live.futures-profitability-coverage-epoch.v1"
)


@dataclass(frozen=True)
class FuturesProfitabilitySpec:
    """Product identity, clock, and risk law for one futures canary."""

    receipt_schema: str
    authority: str
    strategy_id: str
    strategy_version: str
    capital_sleeve: str
    symbol: str
    max_drawdown_usd: float
    slot_tolerance_seconds: float
    evaluation_slots: EvaluationSlots
    natural_slot: NaturalSlot
    held_direction: HeldDirection
    risk_projection: RiskProjection = dict
    coverage_signature: CoverageSignature | None = None
    excluded_clock_field: str = "excluded_clock_slots"
    excluded_slots: ExcludedSlots = lambda _start, _end: 0


FUTURES_RISK_FIELDS = (
    "position_from_fills",
    "run_realized_gross_usd",
    "run_realized_cost_usd",
    "run_realized_net_usd",
    "open_mark_gross_usd",
    "open_mark_cost_usd",
    "open_mark_net_usd",
    "run_gross_usd",
    "run_cost_usd",
    "run_net_usd",
    "peak_run_net_usd",
    "drawdown_usd",
    "closed_trades",
    "gross_wins_usd",
    "top_five_gross_wins_usd",
    "fill_count",
)


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("futures profitability timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _gate(
    status: str,
    reasons: Sequence[str],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def selected_futures_rows(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    spec: FuturesProfitabilitySpec,
) -> tuple[dict[str, object], ...]:
    """Return one selected execution prefix without product-specific projection."""

    return tuple(
        dict(record)
        for record in records
        if record.get("kind") == "checkpoint"
        and record.get("strategy_id") == spec.strategy_id
        and record.get("strategy_version") == spec.strategy_version
        and isinstance(record.get("evidence"), Mapping)
        and record["evidence"].get("selection_id") == selection_id
    )


def _state_rows(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    spec: FuturesProfitabilitySpec,
) -> tuple[dict[str, object], ...]:
    return tuple(
        row
        for row in selected_futures_rows(
            records, selection_id=selection_id, spec=spec
        )
        if row.get("status") == "EVALUATED"
        and row["evidence"].get("phase") == "STATE"
    )


def _aligned_state(
    rows: Sequence[Mapping[str, object]],
    slot: datetime,
    *,
    tolerance_seconds: float,
    coverage_signature: CoverageSignature | None,
) -> tuple[Mapping[str, object] | None, bool]:
    candidates = []
    signatures = set()
    for row in rows:
        try:
            evaluated = _utc(str(row["evaluation_as_of_utc"]))
            recorded = _utc(str(row["recorded_at_utc"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            evaluated.replace(second=0, microsecond=0) == slot
            and abs((recorded - slot).total_seconds()) <= tolerance_seconds
        ):
            candidates.append((recorded, evaluated, row))
            signatures.add(
                evidence_sha256(
                    coverage_signature(row)
                    if coverage_signature is not None
                    else row.get("evidence")
                )
            )
    if len(signatures) > 1:
        return None, True
    return (
        min(candidates, key=lambda item: (item[0], item[1]))[2]
        if candidates
        else None,
        False,
    )


def _risk(
    row: Mapping[str, object] | None,
    spec: FuturesProfitabilitySpec,
) -> Mapping[str, object] | None:
    evidence = row.get("evidence") if isinstance(row, Mapping) else None
    raw = evidence.get("risk_state") if isinstance(evidence, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    projected = spec.risk_projection(raw)
    return projected if isinstance(projected, Mapping) else None


def single_contract_position(
    broker: Mapping[str, object], *, symbol: str, con_id: int
) -> float | None:
    positions = broker.get("positions")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        return None
    relevant = [
        row
        for row in positions
        if isinstance(row, Mapping)
        and (row.get("symbol") == symbol or int(row.get("con_id") or 0) == con_id)
    ]
    if len(relevant) > 1:
        return None
    try:
        return float(relevant[0]["quantity"]) if relevant else 0.0
    except (KeyError, TypeError, ValueError):
        return None


def _risk_numbers(risk: Mapping[str, object]) -> dict[str, float]:
    return {field: float(risk[field]) for field in FUTURES_RISK_FIELDS}


def _empty_receipt(
    *,
    selection_id: str,
    run_started_at: datetime | str,
    cutoff: datetime,
    spec: FuturesProfitabilitySpec,
    status: str,
    reasons: Sequence[str],
    coverage_started_at: datetime | None = None,
    coverage_epoch_id: str | None = None,
) -> dict[str, object]:
    epoch_identity = (
        {
            "coverage_epoch_id": coverage_epoch_id,
            "coverage_started_at_utc": coverage_started_at.isoformat(),
        }
        if coverage_epoch_id is not None and coverage_started_at is not None
        else {}
    )
    return {
        "schema": spec.receipt_schema,
        "authority": spec.authority,
        "as_of_utc": cutoff.isoformat(),
        "status": status,
        "policy": {
            "run_id": selection_id,
            "strategy_id": spec.strategy_id,
            "strategy_version": spec.strategy_version,
            "config_fingerprint": selection_id,
            "capital_sleeve": spec.capital_sleeve,
            "unit": "USD",
            "max_drawdown_usd": spec.max_drawdown_usd,
            "slot_tolerance_seconds": spec.slot_tolerance_seconds,
            **epoch_identity,
        },
        "clock": {
            "run_started_at_utc": _utc(run_started_at).isoformat(),
            "coverage_started_at_utc": (
                coverage_started_at.isoformat()
                if coverage_started_at is not None
                else None
            ),
            **(
                {"coverage_epoch_id": coverage_epoch_id}
                if coverage_epoch_id is not None
                else {}
            ),
            "elapsed_seconds": 0.0,
            "complete_sessions": 0,
            "coverage_broken": False,
        },
        "economics": None,
        "sessions": [],
        "milestones": {
            name: {"passed": False, "reasons": list(reasons)}
            for name in ("24h", "48h", "five_session_week")
        },
        "reasons": sorted(set(reasons)),
    }


def single_contract_profitability_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    run_started_at: datetime | str,
    con_id: int,
    spec: FuturesProfitabilitySpec,
    as_of: datetime | str,
    coverage_epoch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Reduce one reconciled futures prefix without recalculating broker fills."""

    cutoff = _utc(as_of)
    run_started = _utc(run_started_at)
    tolerance = spec.slot_tolerance_seconds
    coverage_epoch_id = None
    fixed_coverage_start = None
    terminal_values: dict[str, float] | None = None
    if coverage_epoch is not None:
        epoch_selection = coverage_epoch.get("selection")
        terminal = coverage_epoch.get("terminal_checkpoint")
        terminal_risk = (
            terminal.get("risk_state") if isinstance(terminal, Mapping) else None
        )
        if (
            coverage_epoch.get("schema")
            != FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA
            or not isinstance(epoch_selection, Mapping)
            or epoch_selection.get("selection_id") != selection_id
            or not isinstance(terminal_risk, Mapping)
        ):
            raise ValueError("coverage epoch does not own this futures run")
        try:
            coverage_epoch_id = str(coverage_epoch["epoch_id"])
            fixed_coverage_start = _utc(coverage_epoch["eligible_start_utc"])
            terminal_values = _risk_numbers(terminal_risk)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("futures coverage epoch evidence is invalid") from exc
        if (
            len(coverage_epoch_id) != 64
            or any(
                value not in "0123456789abcdef" for value in coverage_epoch_id
            )
            or fixed_coverage_start < run_started
            or fixed_coverage_start
            not in spec.evaluation_slots(
                fixed_coverage_start, fixed_coverage_start, True
            )
            or not all(math.isfinite(value) for value in terminal_values.values())
        ):
            raise ValueError("futures coverage epoch evidence is invalid")
    if (
        not math.isfinite(tolerance)
        or tolerance < 0
        or not math.isfinite(spec.max_drawdown_usd)
        or spec.max_drawdown_usd < 0
    ):
        raise ValueError("futures profitability policy is invalid")
    if fixed_coverage_start is not None and cutoff < (
        fixed_coverage_start + timedelta(seconds=tolerance)
    ):
        return _empty_receipt(
            selection_id=selection_id,
            run_started_at=run_started,
            cutoff=cutoff,
            spec=spec,
            status="NOT_STARTED",
            reasons=["coverage_epoch_not_started", "elapsed_time_incomplete"],
            coverage_started_at=fixed_coverage_start,
            coverage_epoch_id=coverage_epoch_id,
        )
    rows = _state_rows(records, selection_id=selection_id, spec=spec)
    aligned = []
    for row in rows:
        try:
            stamp = _utc(str(row["evaluation_as_of_utc"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            (fixed_coverage_start or run_started) <= stamp <= cutoff
            and spec.natural_slot(stamp)
        ):
            aligned.append((stamp, row))
    if not aligned:
        return _empty_receipt(
            selection_id=selection_id,
            run_started_at=run_started,
            cutoff=cutoff,
            spec=spec,
            status=(
                "INVALID_EVIDENCE"
                if fixed_coverage_start is not None
                else "NOT_STARTED"
            ),
            reasons=(
                [
                    "incomplete_session_coverage",
                    "invalid_coverage_epoch_baseline",
                ]
                if fixed_coverage_start is not None
                else ["no_selected_timer_checkpoint"]
            ),
            coverage_started_at=fixed_coverage_start,
            coverage_epoch_id=coverage_epoch_id,
        )
    coverage_start = fixed_coverage_start or min(
        aligned, key=lambda item: item[0]
    )[0].replace(second=0, microsecond=0)
    baseline, conflict = _aligned_state(
        rows,
        coverage_start,
        tolerance_seconds=tolerance,
        coverage_signature=spec.coverage_signature,
    )
    errors: set[str] = set()
    if baseline is None or conflict:
        errors.add(
            "invalid_coverage_epoch_baseline"
            if coverage_epoch is not None
            else "invalid_zero_baseline_checkpoint"
        )
    baseline_risk = _risk(baseline, spec)
    try:
        baseline_values = _risk_numbers(baseline_risk or {})
    except (KeyError, TypeError, ValueError):
        baseline_values = {}
        errors.add("invalid_zero_baseline_checkpoint")
    if coverage_epoch is None and baseline_values and any(
        abs(baseline_values[field]) > 1e-9
        for field in (
            "position_from_fills",
            "run_realized_gross_usd",
            "run_realized_cost_usd",
            "run_realized_net_usd",
            "open_mark_gross_usd",
            "open_mark_cost_usd",
            "open_mark_net_usd",
            "run_gross_usd",
            "run_cost_usd",
            "run_net_usd",
            "drawdown_usd",
            "closed_trades",
            "fill_count",
        )
    ):
        errors.add("nonzero_run_baseline")
    if coverage_epoch is not None and baseline_values:
        baseline_evidence = (
            baseline.get("evidence") if isinstance(baseline, Mapping) else None
        )
        baseline_broker = (
            baseline_evidence.get("broker_state")
            if isinstance(baseline_evidence, Mapping)
            else None
        )
        baseline_plan = (
            baseline_evidence.get("plan")
            if isinstance(baseline_evidence, Mapping)
            else None
        )
        baseline_risk = _risk(baseline, spec)
        try:
            broker_position = (
                single_contract_position(
                    baseline_broker, symbol=spec.symbol, con_id=con_id
                )
                if isinstance(baseline_broker, Mapping)
                else None
            )
        except (AttributeError, TypeError, ValueError):
            broker_position = None
        if (
            not isinstance(baseline_risk, Mapping)
            or baseline_risk.get("valid") is not True
            or baseline_risk.get("attribution_complete") is not True
            or baseline_risk.get("safety_breaches") != []
            or not isinstance(baseline_broker, Mapping)
            or baseline_broker.get("open_orders") != []
            or broker_position != 0
            or not isinstance(baseline_plan, Mapping)
            or baseline_plan.get("held_direction") is not None
            or baseline_evidence.get("submitted_orders") != 0
            or abs(baseline_values["position_from_fills"]) > 1e-9
            or any(
                abs(baseline_values[field]) > 1e-9
                for field in (
                    "open_mark_gross_usd",
                    "open_mark_cost_usd",
                    "open_mark_net_usd",
                )
            )
        ):
            errors.add("invalid_coverage_epoch_baseline")
        assert terminal_values is not None
        if any(
            baseline_values[field] < terminal_values[field] - 1e-9
            for field in (
                "run_cost_usd",
                "drawdown_usd",
                "closed_trades",
                "gross_wins_usd",
                "top_five_gross_wins_usd",
                "fill_count",
            )
        ):
            errors.add("coverage_epoch_economics_regressed")

    matured_to = cutoff - timedelta(seconds=tolerance)
    due_slots = spec.evaluation_slots(coverage_start, matured_to, False)
    slot_rows: dict[datetime, Mapping[str, object]] = {}
    missing = []
    conflicts = []
    for slot in due_slots:
        row, conflict = _aligned_state(
            rows,
            slot,
            tolerance_seconds=tolerance,
            coverage_signature=spec.coverage_signature,
        )
        if conflict:
            conflicts.append(slot.isoformat())
        elif row is None:
            missing.append(slot.isoformat())
        else:
            slot_rows[slot] = row
    if missing:
        errors.add("incomplete_session_coverage")
    if conflicts:
        errors.add("conflicting_session_coverage")

    ordered = [baseline, *(slot_rows[slot] for slot in due_slots if slot in slot_rows)]
    prior_fill = prior_cost = prior_trades = 0.0
    final_risk: Mapping[str, object] | None = None
    equity_path = [0.0]
    reported_drawdown = 0.0
    for row in (value for value in ordered if isinstance(value, Mapping)):
        evidence = row.get("evidence")
        risk = _risk(row, spec)
        broker = evidence.get("broker_state") if isinstance(evidence, Mapping) else None
        plan = evidence.get("plan") if isinstance(evidence, Mapping) else None
        if (
            not isinstance(risk, Mapping)
            or risk.get("valid") is not True
            or risk.get("attribution_complete") is not True
            or not isinstance(broker, Mapping)
            or not isinstance(plan, Mapping)
            or evidence.get("submitted_orders") != 0
        ):
            errors.add("selected_risk_projection_invalid")
            continue
        try:
            values = _risk_numbers(risk)
            broker_position = single_contract_position(
                broker, symbol=spec.symbol, con_id=con_id
            )
        except (KeyError, TypeError, ValueError):
            errors.add("selected_risk_economics_invalid")
            continue
        if (
            broker_position is None
            or abs(broker_position - values["position_from_fills"]) > 1e-9
            or values["position_from_fills"] not in {-1.0, 0.0, 1.0}
            or plan.get("held_direction")
            != spec.held_direction(values["position_from_fills"])
            or not all(math.isfinite(value) for value in values.values())
            or values["run_cost_usd"] < -1e-9
            or values["drawdown_usd"] < -1e-9
            or values["fill_count"] < prior_fill
            or values["run_cost_usd"] < prior_cost - 1e-9
            or values["closed_trades"] < prior_trades
            or not values["fill_count"].is_integer()
            or not values["closed_trades"].is_integer()
            or abs(
                values["run_net_usd"]
                - values["run_gross_usd"]
                + values["run_cost_usd"]
            )
            > 1e-7
            or abs(
                values["run_net_usd"]
                - values["run_realized_net_usd"]
                - values["open_mark_net_usd"]
            )
            > 1e-7
        ):
            errors.add("selected_risk_economics_invalid")
            continue
        breaches = risk.get("safety_breaches")
        if not isinstance(breaches, list):
            errors.add("selected_safety_evidence_invalid")
        elif breaches:
            errors.add("selected_safety_breach")
        if values["drawdown_usd"] > spec.max_drawdown_usd + 1e-9:
            errors.add("drawdown_limit_breached")
        prior_fill = values["fill_count"]
        prior_cost = values["run_cost_usd"]
        prior_trades = values["closed_trades"]
        reported_drawdown = max(reported_drawdown, values["drawdown_usd"])
        final_risk = risk
        equity_path.append(values["run_net_usd"])

    sessions = []
    complete_sessions = 0
    session_start = coverage_start
    prior_net = (
        baseline_values.get("run_net_usd", 0.0)
        if coverage_epoch is not None
        else 0.0
    )
    while session_start + timedelta(hours=24, seconds=tolerance) <= cutoff:
        session_end = session_start + timedelta(hours=24)
        expected = spec.evaluation_slots(session_start, session_end, False)
        end_risk = _risk(slot_rows.get(session_end), spec)
        complete = bool(expected) and all(slot in slot_rows for slot in expected)
        session_net = None
        if complete and isinstance(end_risk, Mapping):
            try:
                end_net = float(end_risk["run_net_usd"])
                session_net = end_net - prior_net
                prior_net = end_net
                complete_sessions += 1
            except (KeyError, TypeError, ValueError):
                complete = False
                errors.add("selected_risk_economics_invalid")
        sessions.append(
            {
                "session_start_utc": session_start.isoformat(),
                "session_end_utc": session_end.isoformat(),
                "expected_slots": len(expected),
                "evaluated_slots": sum(slot in slot_rows for slot in expected),
                "complete": complete,
                "net_usd": session_net,
            }
        )
        session_start = session_end

    final_values = (
        _risk_numbers(final_risk)
        if final_risk is not None
        else {field: 0.0 for field in FUTURES_RISK_FIELDS}
    )
    high = maximum_drawdown = 0.0
    for value in equity_path:
        high = max(high, value)
        maximum_drawdown = max(maximum_drawdown, high - value)
    if coverage_epoch is not None:
        maximum_drawdown = max(maximum_drawdown, reported_drawdown)
    gross_wins = final_values["gross_wins_usd"]
    top_five = final_values["top_five_gross_wins_usd"]
    economics = {
        "unit": "USD",
        "gross_usd": final_values["run_gross_usd"],
        "cost_usd": final_values["run_cost_usd"],
        "net_usd": final_values["run_net_usd"],
        "realized_net_usd": final_values["run_realized_net_usd"],
        "open_mark_usd": final_values["open_mark_net_usd"],
        "maximum_drawdown_usd": maximum_drawdown,
        "worst_session_usd": min(
            (float(row["net_usd"]) for row in sessions if row["net_usd"] is not None),
            default=0.0,
        ),
        "closed_trades": int(final_values["closed_trades"]),
        "fills": int(final_values["fill_count"]),
        "gross_wins_usd": gross_wins,
        "top_five_gross_wins_usd": top_five,
        "top_five_win_share": top_five / gross_wins if gross_wins > 0 else None,
    }
    elapsed = max(0.0, (cutoff - coverage_start).total_seconds())
    milestones = {}
    for name, seconds, required_sessions in (
        ("24h", 24 * 3600, 1),
        ("48h", 48 * 3600, 2),
        ("five_session_week", 7 * 24 * 3600, 5),
    ):
        evidence_at = coverage_start + timedelta(seconds=seconds + tolerance)
        reasons = []
        if cutoff < evidence_at:
            reasons.append("elapsed_time_incomplete")
        if complete_sessions < required_sessions:
            reasons.append("eligible_sessions_incomplete")
        boundary = coverage_start + timedelta(seconds=seconds)
        boundary_risk = _risk(slot_rows.get(boundary), spec)
        try:
            boundary_net = float(boundary_risk["run_net_usd"])
            boundary_fills = int(boundary_risk["fill_count"])
        except (KeyError, TypeError, ValueError):
            boundary_net = 0.0
            boundary_fills = 0
        if cutoff >= evidence_at and boundary_net <= 0:
            reasons.append("net_not_positive")
        if cutoff >= evidence_at and boundary_fills < 1:
            reasons.append("authentic_execution_missing")
        milestones[name] = {
            "passed": not reasons,
            "economic_window_end_utc": boundary.isoformat(),
            "evidence_as_of_utc": evidence_at.isoformat(),
            "required_elapsed_seconds": seconds,
            "elapsed_seconds": min(elapsed, float(seconds)),
            "required_complete_sessions": required_sessions,
            "complete_sessions": complete_sessions,
            "economics": (
                {"net_usd": boundary_net, "fills": boundary_fills}
                if cutoff >= evidence_at
                else None
            ),
            "reasons": sorted(set(reasons)),
        }
    return {
        "schema": spec.receipt_schema,
        "authority": spec.authority,
        "as_of_utc": cutoff.isoformat(),
        "status": (
            "INVALID_EVIDENCE"
            if errors
            else "PASSED"
            if all(row["passed"] for row in milestones.values())
            else "ACTIVE"
        ),
        "policy": {
            "run_id": selection_id,
            "strategy_id": spec.strategy_id,
            "strategy_version": spec.strategy_version,
            "config_fingerprint": selection_id,
            "capital_sleeve": spec.capital_sleeve,
            "unit": "USD",
            "max_drawdown_usd": spec.max_drawdown_usd,
            "slot_tolerance_seconds": tolerance,
            **(
                {
                    "coverage_epoch_id": coverage_epoch_id,
                    "coverage_started_at_utc": coverage_start.isoformat(),
                }
                if coverage_epoch_id is not None
                else {}
            ),
        },
        "clock": {
            "run_started_at_utc": run_started.isoformat(),
            "coverage_started_at_utc": coverage_start.isoformat(),
            **(
                {"coverage_epoch_id": coverage_epoch_id}
                if coverage_epoch_id is not None
                else {}
            ),
            "elapsed_seconds": elapsed,
            "complete_sessions": complete_sessions,
            "coverage_broken": bool(missing or conflicts),
            "due_slots": len(due_slots),
            "evaluated_slots": len(slot_rows),
            spec.excluded_clock_field: spec.excluded_slots(
                coverage_start, cutoff
            ),
        },
        "economics": economics,
        "sessions": sessions,
        "milestones": milestones,
        "reasons": sorted(errors),
    }


def single_contract_restart_gate(
    *,
    selected_at_utc: datetime | str,
    rows: Sequence[Mapping[str, object]],
    con_id: int,
    spec: FuturesProfitabilitySpec,
) -> dict[str, object]:
    states = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("phase") == "STATE"
    ]
    if not states:
        return _gate("HOLD", ["restart_checkpoint_missing"], {})
    reasons = []
    selected_at = _utc(selected_at_utc)
    first_at = _utc(str(states[0]["recorded_at_utc"]))
    if not selected_at <= first_at <= selected_at + timedelta(minutes=6, seconds=30):
        reasons.append("restart_checkpoint_outside_selection_boundary")
    for row in states[:2]:
        evidence = row["evidence"]
        risk = _risk(row, spec)
        broker = evidence.get("broker_state")
        plan = evidence.get("plan")
        try:
            values = _risk_numbers(risk or {})
            zero = bool(
                isinstance(broker, Mapping)
                and isinstance(plan, Mapping)
                and values["position_from_fills"] == 0
                and values["fill_count"] == 0
                and values["closed_trades"] == 0
                and values["run_net_usd"] == 0
                and single_contract_position(
                    broker, symbol=spec.symbol, con_id=con_id
                )
                == 0
                and broker.get("open_orders") == []
                and plan.get("held_direction") is None
                and evidence.get("submitted_orders") == 0
            )
        except (KeyError, TypeError, ValueError):
            zero = False
        if not zero:
            reasons.append("restart_zero_baseline_mismatch")
    status = "INVALID" if reasons else "PASS" if len(states) >= 2 else "HOLD"
    if status == "HOLD":
        reasons.append("first_natural_recurrence_missing")
    return _gate(
        status,
        reasons,
        {
            "restart_checkpoint_id": states[0].get("checkpoint_id"),
            "first_natural_checkpoint_id": (
                states[1].get("checkpoint_id") if len(states) >= 2 else None
            ),
        },
    )


def single_contract_risk_gates(
    *,
    rows: Sequence[Mapping[str, object]],
    con_id: int,
    spec: FuturesProfitabilitySpec,
) -> tuple[dict[str, object], dict[str, object]]:
    states = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("phase") == "STATE"
    ]
    if not states:
        return (
            _gate("HOLD", ["selected_risk_state_missing"], {}),
            _gate("HOLD", ["attribution_not_observed"], {}),
        )
    invalid: list[str] = []
    stops: list[str] = []
    attribution: list[str] = []
    prior_fills = prior_cost = prior_trades = 0.0
    latest: Mapping[str, object] = {}
    for row in states:
        evidence = row["evidence"]
        risk = _risk(row, spec)
        broker = evidence.get("broker_state")
        plan = evidence.get("plan")
        if not all(isinstance(value, Mapping) for value in (risk, broker, plan)):
            invalid.append("selected_risk_projection_invalid")
            continue
        assert isinstance(risk, Mapping) and isinstance(broker, Mapping)
        assert isinstance(plan, Mapping)
        latest = risk
        try:
            values = _risk_numbers(risk)
            broker_position = single_contract_position(
                broker, symbol=spec.symbol, con_id=con_id
            )
            finite = all(math.isfinite(value) for value in values.values())
        except (KeyError, TypeError, ValueError):
            invalid.append("selected_risk_economics_invalid")
            continue
        if (
            risk.get("valid") is not True
            or not finite
            or broker_position != values["position_from_fills"]
            or values["position_from_fills"] not in {-1.0, 0.0, 1.0}
            or plan.get("held_direction")
            != spec.held_direction(values["position_from_fills"])
            or values["fill_count"] < prior_fills
            or values["run_cost_usd"] < prior_cost - 1e-9
            or values["closed_trades"] < prior_trades
            or values["run_cost_usd"] < -1e-9
            or values["drawdown_usd"] < -1e-9
            or not values["fill_count"].is_integer()
            or not values["closed_trades"].is_integer()
        ):
            invalid.append("selected_risk_economics_invalid")
        prior_fills = values["fill_count"]
        prior_cost = values["run_cost_usd"]
        prior_trades = values["closed_trades"]
        if plan.get("leg") is not None:
            try:
                decision = validate_live_capital_decision(plan.get("capital_admission"))
                if decision.get("status") != "ALLOW":
                    raise ValueError
            except (TypeError, ValueError):
                invalid.append("capital_admission_invalid")
        breaches = risk.get("safety_breaches")
        if not isinstance(breaches, list):
            invalid.append("selected_safety_evidence_invalid")
        elif breaches:
            stops.extend(str(value) for value in breaches)
        if values["drawdown_usd"] > spec.max_drawdown_usd + 1e-9:
            stops.append("drawdown_limit_breached")
        if risk.get("attribution_complete") is not True:
            attribution.append("selected_attribution_incomplete")
    return (
        _gate(
            "STOP" if stops else "INVALID" if invalid else "PASS",
            [*stops, *invalid],
            {
                "state_rows": len(states),
                "latest_risk_fingerprint": evidence_sha256(latest),
                "fill_count": latest.get("fill_count"),
                "closed_trades": latest.get("closed_trades"),
                "drawdown_usd": latest.get("drawdown_usd"),
            },
        ),
        _gate(
            "INVALID" if attribution else "PASS",
            attribution,
            {"state_rows": len(states), "attribution_complete": not attribution},
        ),
    )
