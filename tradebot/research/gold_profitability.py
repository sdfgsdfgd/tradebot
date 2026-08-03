"""Cutoff-bound profitability and graduation evidence for selected 1OZ runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..live.capital import validate_live_capital_decision
from ..live.capital_stability import portfolio_capital_owner_stability_gate
from ..live.order_evidence import single_contract_execution_graduation_gate
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_EXECUTION_VERSION,
    GOLD_LIVE_MAX_COMMISSION_USD,
    GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
    GOLD_REGIME_HARMONY_VERSION,
    load_gold_live_selection_from_mapping,
)
from .live_graduation import evidence_sha256, live_calibration_logical_prefix


GOLD_LIVE_PROFITABILITY_SCHEMA = "gold.live-profitability.v1"
GOLD_RUNTIME_PARITY_SCHEMA = "gold.1oz-regime-harmony-runtime-parity.v1"
GOLD_TIMER_MINUTES = frozenset(range(2, 60, 5))
CHICAGO = ZoneInfo("America/Chicago")


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("gold profitability timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def gold_1oz_maintenance(at: datetime | str) -> bool:
    """Return CME's regular 1OZ maintenance state in America/Chicago time."""

    local = _utc(at).astimezone(CHICAGO)
    minute = local.hour * 60 + local.minute
    if local.weekday() == 5:
        return 2 * 60 <= minute < 4 * 60
    return local.weekday() < 5 and 16 * 60 <= minute < 16 * 60 + 2


def gold_1oz_evaluation_slots(
    start_utc: datetime | str,
    end_utc: datetime | str,
    *,
    include_start: bool = False,
) -> tuple[datetime, ...]:
    """Project the five-minute worker clock, excluding published maintenance."""

    start = _utc(start_utc)
    end = _utc(end_utc)
    if end < start:
        raise ValueError("gold evidence interval regressed")
    cursor = start.replace(second=0, microsecond=0)
    while cursor.minute not in GOLD_TIMER_MINUTES or cursor < start:
        cursor += timedelta(minutes=1)
    output = []
    while cursor <= end:
        if (include_start or cursor > start) and not gold_1oz_maintenance(cursor):
            output.append(cursor)
        cursor += timedelta(minutes=5)
    return tuple(output)


def _state_rows(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(record)
        for record in records
        if record.get("kind") == "checkpoint"
        and record.get("strategy_id") == GOLD_REGIME_HARMONY_VERSION
        and record.get("strategy_version") == GOLD_LIVE_EXECUTION_VERSION
        and record.get("status") == "EVALUATED"
        and isinstance(record.get("evidence"), Mapping)
        and record["evidence"].get("selection_id") == selection_id
        and record["evidence"].get("phase") == "STATE"
    )


def _selected_rows(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(record)
        for record in records
        if record.get("kind") == "checkpoint"
        and record.get("strategy_id") == GOLD_REGIME_HARMONY_VERSION
        and record.get("strategy_version") == GOLD_LIVE_EXECUTION_VERSION
        and isinstance(record.get("evidence"), Mapping)
        and record["evidence"].get("selection_id") == selection_id
    )


def _aligned_state(
    rows: Sequence[Mapping[str, object]],
    slot: datetime,
    *,
    tolerance_seconds: float,
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
            evidence = row.get("evidence")
            candidates.append((recorded, evaluated, row))
            signatures.add(evidence_sha256(evidence))
    if len(signatures) > 1:
        return None, True
    return (
        min(candidates, key=lambda item: (item[0], item[1]))[2]
        if candidates
        else None,
        False,
    )


def _risk(row: Mapping[str, object]) -> Mapping[str, object] | None:
    evidence = row.get("evidence")
    value = evidence.get("risk_state") if isinstance(evidence, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _gold_position(broker: Mapping[str, object], con_id: int) -> float | None:
    positions = broker.get("positions")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
        return None
    relevant = [
        row
        for row in positions
        if isinstance(row, Mapping)
        and (row.get("symbol") == "1OZ" or int(row.get("con_id") or 0) == con_id)
    ]
    if len(relevant) > 1:
        return None
    try:
        return float(relevant[0]["quantity"]) if relevant else 0.0
    except (KeyError, TypeError, ValueError):
        return None


def _risk_numbers(risk: Mapping[str, object]) -> dict[str, float]:
    fields = (
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
    return {field: float(risk[field]) for field in fields}


def _empty_receipt(
    selection: Mapping[str, object],
    *,
    cutoff: datetime,
    status: str,
    reasons: Sequence[str],
) -> dict[str, object]:
    run_id = str(selection.get("selection_id") or "")
    return {
        "schema": GOLD_LIVE_PROFITABILITY_SCHEMA,
        "authority": "selected_reconciled_gold_risk_state_only",
        "as_of_utc": cutoff.isoformat(),
        "status": status,
        "policy": {
            "run_id": run_id,
            "strategy_id": GOLD_REGIME_HARMONY_VERSION,
            "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
            "config_fingerprint": run_id,
            "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
            "unit": "USD",
            "max_drawdown_usd": GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
            "slot_tolerance_seconds": 90.0,
        },
        "clock": {
            "run_started_at_utc": selection.get("run_started_at_utc"),
            "coverage_started_at_utc": None,
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


def gold_live_profitability_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    selection: Mapping[str, object],
    as_of: datetime | str,
    slot_tolerance_seconds: float = 90.0,
) -> dict[str, object]:
    """Reduce selected gold state rows without recalculating broker fills."""

    selected = load_gold_live_selection_from_mapping(selection)
    cutoff = _utc(as_of)
    if not math.isfinite(slot_tolerance_seconds) or not 0 <= slot_tolerance_seconds < 150:
        raise ValueError("gold slot tolerance is invalid")
    run_started = _utc(str(selected["run_started_at_utc"]))
    rows = _state_rows(records, selection_id=str(selected["selection_id"]))
    aligned = []
    for row in rows:
        try:
            stamp = _utc(str(row["evaluation_as_of_utc"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            stamp >= run_started
            and stamp <= cutoff
            and stamp.minute in GOLD_TIMER_MINUTES
            and stamp.second <= slot_tolerance_seconds
        ):
            aligned.append((stamp, row))
    if not aligned:
        return _empty_receipt(
            selected,
            cutoff=cutoff,
            status="NOT_STARTED",
            reasons=["no_selected_timer_checkpoint"],
        )
    coverage_start = min(aligned, key=lambda item: item[0])[0].replace(
        second=0, microsecond=0
    )
    baseline, baseline_conflict = _aligned_state(
        rows,
        coverage_start,
        tolerance_seconds=slot_tolerance_seconds,
    )
    errors: set[str] = set()
    if baseline is None or baseline_conflict:
        errors.add("invalid_zero_baseline_checkpoint")
    baseline_risk = _risk(baseline) if baseline is not None else None
    try:
        baseline_values = _risk_numbers(baseline_risk or {})
    except (KeyError, TypeError, ValueError):
        baseline_values = {}
        errors.add("invalid_zero_baseline_checkpoint")
    if baseline_values and any(
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

    matured_to = cutoff - timedelta(seconds=slot_tolerance_seconds)
    due_slots = gold_1oz_evaluation_slots(
        coverage_start,
        matured_to,
        include_start=False,
    )
    slot_rows: dict[datetime, Mapping[str, object]] = {}
    missing = []
    conflicts = []
    for slot in due_slots:
        row, conflict = _aligned_state(
            rows,
            slot,
            tolerance_seconds=slot_tolerance_seconds,
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
    ordered = [row for row in ordered if isinstance(row, Mapping)]
    prior_fill = prior_cost = prior_trades = 0.0
    con_id = int(selected["contract"]["con_id"])
    final_risk: Mapping[str, object] | None = None
    equity_path = [0.0]
    for row in ordered:
        evidence = row.get("evidence")
        risk = _risk(row)
        broker = evidence.get("broker_state") if isinstance(evidence, Mapping) else None
        plan = evidence.get("plan") if isinstance(evidence, Mapping) else None
        if (
            not isinstance(risk, Mapping)
            or risk.get("valid") is not True
            or not isinstance(broker, Mapping)
            or not isinstance(plan, Mapping)
            or evidence.get("submitted_orders") != 0
            or risk.get("attribution_complete") is not True
        ):
            errors.add("selected_risk_projection_invalid")
            continue
        try:
            values = _risk_numbers(risk)
            broker_position = _gold_position(broker, con_id)
        except (KeyError, TypeError, ValueError):
            errors.add("selected_risk_economics_invalid")
            continue
        if (
            broker_position is None
            or abs(broker_position - values["position_from_fills"]) > 1e-9
            or values["position_from_fills"] not in {-1.0, 0.0, 1.0}
            or plan.get("held_direction")
            != (
                "up"
                if values["position_from_fills"] > 0
                else "down"
                if values["position_from_fills"] < 0
                else None
            )
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
        if values["drawdown_usd"] > GOLD_LIVE_MAX_RUN_DRAWDOWN_USD + 1e-9:
            errors.add("drawdown_limit_breached")
        prior_fill = values["fill_count"]
        prior_cost = values["run_cost_usd"]
        prior_trades = values["closed_trades"]
        final_risk = risk
        equity_path.append(values["run_net_usd"])

    sessions = []
    complete_sessions = 0
    session_start = coverage_start
    prior_net = 0.0
    while session_start + timedelta(hours=24, seconds=slot_tolerance_seconds) <= cutoff:
        session_end = session_start + timedelta(hours=24)
        expected = gold_1oz_evaluation_slots(session_start, session_end)
        end_row = slot_rows.get(session_end)
        complete = bool(expected) and all(slot in slot_rows for slot in expected)
        end_risk = _risk(end_row) if end_row is not None else None
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

    final_values = _risk_numbers(final_risk) if final_risk is not None else {
        field: 0.0
        for field in (
            "run_gross_usd",
            "run_cost_usd",
            "run_net_usd",
            "run_realized_net_usd",
            "open_mark_net_usd",
            "closed_trades",
            "gross_wins_usd",
            "top_five_gross_wins_usd",
            "fill_count",
        )
    }
    high = maximum_drawdown = 0.0
    for value in equity_path:
        high = max(high, value)
        maximum_drawdown = max(maximum_drawdown, high - value)
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
        evidence_at = coverage_start + timedelta(
            seconds=seconds + slot_tolerance_seconds
        )
        reasons = []
        if cutoff < evidence_at:
            reasons.append("elapsed_time_incomplete")
        if complete_sessions < required_sessions:
            reasons.append("eligible_sessions_incomplete")
        boundary = coverage_start + timedelta(seconds=seconds)
        boundary_row = slot_rows.get(boundary)
        boundary_risk = _risk(boundary_row) if boundary_row is not None else None
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
                {
                    "net_usd": boundary_net,
                    "fills": boundary_fills,
                }
                if cutoff >= evidence_at
                else None
            ),
            "reasons": sorted(set(reasons)),
        }

    return {
        "schema": GOLD_LIVE_PROFITABILITY_SCHEMA,
        "authority": "selected_reconciled_gold_risk_state_only",
        "as_of_utc": cutoff.isoformat(),
        "status": "INVALID_EVIDENCE" if errors else "PASSED" if all(
            row["passed"] for row in milestones.values()
        ) else "ACTIVE",
        "policy": {
            "run_id": selected["selection_id"],
            "strategy_id": GOLD_REGIME_HARMONY_VERSION,
            "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
            "config_fingerprint": selected["selection_id"],
            "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
            "unit": "USD",
            "max_drawdown_usd": GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
            "slot_tolerance_seconds": slot_tolerance_seconds,
        },
        "clock": {
            "run_started_at_utc": run_started.isoformat(),
            "coverage_started_at_utc": coverage_start.isoformat(),
            "elapsed_seconds": elapsed,
            "complete_sessions": complete_sessions,
            "coverage_broken": bool(missing or conflicts),
            "due_slots": len(due_slots),
            "evaluated_slots": len(slot_rows),
            "maintenance_slots_excluded": sum(
                gold_1oz_maintenance(
                    coverage_start + timedelta(minutes=5 * index)
                )
                for index in range(max(0, int(elapsed // 300) + 1))
            ),
        },
        "economics": economics,
        "sessions": sessions,
        "milestones": milestones,
        "reasons": sorted(errors),
    }


def gold_runtime_parity_graduation_gate(
    path: Path,
    *,
    repo_root: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable Stage-76 crown/runtime proof and current owners."""

    try:
        raw = path.read_bytes()
        proof = json.loads(raw)
        selected = load_gold_live_selection_from_mapping(selection)
        if not isinstance(proof, Mapping):
            raise ValueError("gold runtime proof must be an object")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _gate(
            "INVALID",
            ["runtime_parity_proof_unreadable"],
            {"path": str(path), "error": str(exc)},
        )
    reasons: list[str] = []
    digest = hashlib.sha256(raw).hexdigest()
    frozen = selected.get("evidence")
    frozen_runtime = frozen.get("runtime_parity") if isinstance(frozen, Mapping) else None
    crown = proof.get("crown")
    selected_crown = frozen.get("crown") if isinstance(frozen, Mapping) else None
    if (
        proof.get("schema") != GOLD_RUNTIME_PARITY_SCHEMA
        or proof.get("authority")
        != "immutable_signal_runtime_parity_only_no_selection_no_capital_no_orders"
        or not isinstance(frozen_runtime, Mapping)
        or frozen_runtime.get("sha256") != digest
        or not isinstance(crown, Mapping)
        or not isinstance(selected_crown, Mapping)
        or any(
            crown.get(field) != selected_crown.get(field)
            for field in (
                "strategy_version",
                "strategy_key",
                "declaration_path",
                "declaration_sha256",
                "artifact_path",
                "artifact_sha256",
            )
        )
    ):
        reasons.append("runtime_parity_identity_invalid")
    root = repo_root.resolve()
    owners = proof.get("owners")
    if not isinstance(owners, Mapping) or not owners:
        reasons.append("runtime_owner_surface_missing")
    else:
        for owner in owners.values():
            try:
                if not isinstance(owner, Mapping):
                    raise ValueError
                relative = Path(str(owner["path"]))
                current = (root / relative).resolve()
                if relative.is_absolute() or root not in current.parents:
                    raise ValueError
                if _sha256(current) != owner.get("sha256"):
                    reasons.append(f"runtime_owner_drift:{relative}")
            except (KeyError, OSError, TypeError, ValueError):
                reasons.append("runtime_owner_invalid")
    context = proof.get("context_parity")
    exact_context = {
        "daily_hard_direction_and_age_exact",
        "h4_fast_slope_spread_velocity_acceleration_and_atr_exact",
        "macro_5_21_63_direction_velocity_acceleration_exact",
    }
    if not isinstance(context, Mapping) or any(
        context.get(key) is not True for key in exact_context
    ):
        reasons.append("runtime_context_parity_invalid")
    historical = proof.get("historical_parity")
    for name in ("full_three_year", "full_ten_year"):
        window = historical.get(name) if isinstance(historical, Mapping) else None
        try:
            valid = bool(
                isinstance(window, Mapping)
                and window.get("converged") is True
                and int(window["trades"]) > 0
                and float(window["net_pnl"]) > 0
                and float(window["profit_factor"]) > 1
                and len(str(window["ledger_sha256"])) == 64
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            reasons.append(f"runtime_{name}_invalid")
    prefix = proof.get("prospective_prefix")
    if (
        not isinstance(prefix, Mapping)
        or prefix.get("cold_replay_equal") is not True
        or prefix.get("converged") is not True
        or prefix.get("synthetic_midcycle_entry_authority") != "none"
        or prefix.get("order_authority") != "none"
        or prefix.get("submitted_orders") != 0
    ):
        reasons.append("runtime_cold_restart_parity_invalid")
    gates = proof.get("gates")
    required = (
        "machine_crown_identity",
        "shared_context_math",
        "full_three_year_ledger",
        "full_ten_year_ledger",
        "cold_replay_and_restart_identity",
        "flat_current_prefix",
    )
    if not isinstance(gates, Mapping) or any(gates.get(key) != "PASS" for key in required):
        reasons.append("runtime_signal_gates_invalid")
    return _gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "path": str(path),
            "sha256": digest,
            "three_year_ledger_sha256": (
                historical.get("full_three_year", {}).get("ledger_sha256")
                if isinstance(historical, Mapping)
                else None
            ),
            "ten_year_ledger_sha256": (
                historical.get("full_ten_year", {}).get("ledger_sha256")
                if isinstance(historical, Mapping)
                else None
            ),
        },
    )


def _gold_restart_gate(
    selection: Mapping[str, object], rows: Sequence[Mapping[str, object]]
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
    selected_at = _utc(str(selection["selected_at_utc"]))
    first_at = _utc(str(states[0]["recorded_at_utc"]))
    if not selected_at <= first_at <= selected_at + timedelta(minutes=6, seconds=30):
        reasons.append("restart_checkpoint_outside_selection_boundary")
    for row in states[:2]:
        evidence = row["evidence"]
        risk = evidence.get("risk_state")
        broker = evidence.get("broker_state")
        plan = evidence.get("plan")
        try:
            zero = bool(
                isinstance(risk, Mapping)
                and isinstance(broker, Mapping)
                and isinstance(plan, Mapping)
                and float(risk["position_from_fills"]) == 0
                and float(risk["fill_count"]) == 0
                and float(risk["closed_trades"]) == 0
                and float(risk["run_net_usd"]) == 0
                and _gold_position(broker, int(selection["contract"]["con_id"])) == 0
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


def _gold_risk_gates(
    selection: Mapping[str, object], rows: Sequence[Mapping[str, object]]
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
    con_id = int(selection["contract"]["con_id"])
    for row in states:
        evidence = row["evidence"]
        risk = evidence.get("risk_state")
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
            broker_position = _gold_position(broker, con_id)
            finite = all(math.isfinite(value) for value in values.values())
        except (KeyError, TypeError, ValueError):
            invalid.append("selected_risk_economics_invalid")
            continue
        held = "up" if values["position_from_fills"] > 0 else "down" if values["position_from_fills"] < 0 else None
        if (
            risk.get("valid") is not True
            or not finite
            or broker_position != values["position_from_fills"]
            or values["position_from_fills"] not in {-1.0, 0.0, 1.0}
            or plan.get("held_direction") != held
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
        admission = plan.get("capital_admission")
        if plan.get("leg") is not None:
            try:
                decision = validate_live_capital_decision(admission)
                if decision.get("status") != "ALLOW":
                    raise ValueError
            except (TypeError, ValueError):
                invalid.append("capital_admission_invalid")
        breaches = risk.get("safety_breaches")
        if not isinstance(breaches, list):
            invalid.append("selected_safety_evidence_invalid")
        elif breaches:
            stops.extend(str(value) for value in breaches)
        if values["drawdown_usd"] > GOLD_LIVE_MAX_RUN_DRAWDOWN_USD + 1e-9:
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


def gold_live_graduation_inputs(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    cutoff_utc: datetime | str,
    profitability_receipt: Mapping[str, object],
    runtime_parity_path: Path,
    capital_owner_stability_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Project Gold raw truth into the shared live-graduation reducer."""

    selected = load_gold_live_selection_from_mapping(selection)
    cutoff = _utc(cutoff_utc)
    selection_sha = _sha256(selection_path)
    prefix, projected = live_calibration_logical_prefix(records, cutoff_utc=cutoff)
    rows = _selected_rows(projected, selection_id=str(selected["selection_id"]))
    restart = _gold_restart_gate(selected, rows)
    risk, attribution = _gold_risk_gates(selected, rows)
    broker = selected.get("broker_at_selection")
    account_id = str(broker.get("account_id") or "") if isinstance(broker, Mapping) else ""
    subject = {
        "strategy_id": GOLD_REGIME_HARMONY_VERSION,
        "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "signal_instrument": "XAUUSD/GC",
        "execution_sleeve": "1OZ",
        "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
        "selection_id": selected["selection_id"],
        "run_id": selected["selection_id"],
        "account_fingerprint": hashlib.sha256(account_id.encode()).hexdigest(),
    }
    return {
        "subject": subject,
        "selection": {
            "path": str(selection_path),
            "selection_id": selected["selection_id"],
            "run_id": selected["selection_id"],
            "selected_at_utc": selected["selected_at_utc"],
            "run_started_at_utc": selected["run_started_at_utc"],
            "signal_strategy_version": selected["strategy_version"],
            "execution_strategy_version": GOLD_LIVE_EXECUTION_VERSION,
            "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
            "selection_file_sha256": selection_sha,
        },
        "selection_file_sha256": selection_sha,
        "ledger_prefix": {
            **prefix,
            "gates": {
                "restart": restart,
                "cash_risk_safety": risk,
                "attribution": attribution,
                "execution": single_contract_execution_graduation_gate(
                    rows,
                    selection_id=str(selected["selection_id"]),
                    sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
                    symbol="1OZ",
                    con_id=int(selected["contract"]["con_id"]),
                    order_ref_prefix="GOLD76",
                    ladder_schema="gold.execution-ladder-transition.v1",
                    max_commission_usd=GOLD_LIVE_MAX_COMMISSION_USD,
                ),
            },
        },
        "profitability_receipt": dict(profitability_receipt),
        "runtime_parity_proof": gold_runtime_parity_graduation_gate(
            runtime_parity_path, repo_root=repo_root, selection=selected
        ),
        "capital_owner_stability_proof": portfolio_capital_owner_stability_gate(
            capital_owner_stability_path,
            repo_root=repo_root,
            sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
            selection_id=str(selected["selection_id"]),
            selection_file_sha256=selection_sha,
        ),
    }
