"""Selected-equity contracts for XSP live profitability receipts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ..engines.market import xsp_rth_cash_evaluation_slots
from ..live.capital import validate_live_capital_decision
from ..time_utils import ET_ZONE
from .live_graduation import (
    evidence_sha256,
    live_calibration_logical_prefix,
)
from .xsp_capital_stability import (
    XSP_CAPITAL_OWNER_STABILITY_SCHEMA as XSP_CAPITAL_OWNER_STABILITY_SCHEMA,
    xsp_capital_owner_stability_graduation_gate,
)
from .xsp_runtime_parity_contract import (
    XSP_P009_RUNTIME_PARITY_SCHEMA,
    XSP_RUNTIME_PARITY_SCHEMA,
    xsp_runtime_parity_owner_paths,
)


LIVE_PROFITABILITY_SCHEMA = "xsp.live-profitability.v1"
SELECTED_EQUITY_SCHEMA = "xsp.selected-equity.v1"
SELECTED_CASH_EQUITY_SCHEMA = "xsp.selected-cash-equity.v1"

_SELECTED_EQUITY_FIELDS = {
    "schema", "run_id", "run_started_at_utc", "config_fingerprint",
    "capital_sleeve", "unit", "cumulative_gross_points",
    "cumulative_cost_points", "cumulative_net_points",
    "cumulative_realized_net_points", "open_mark_points",
    "session_gross_points", "session_cost_points", "session_net_points",
    "closed_trades", "gross_wins_points", "top_five_gross_wins_points",
    "reconciled", "attribution_complete", "safety_breaches",
}
_SELECTED_CASH_EQUITY_FIELDS = {
    "schema", "run_id", "run_started_at_utc", "config_fingerprint",
    "capital_sleeve", "unit", "cumulative_gross_usd",
    "cumulative_cost_usd", "cumulative_net_usd",
    "cumulative_realized_net_usd", "open_mark_usd",
    "session_gross_usd", "session_cost_usd", "session_net_usd",
    "closed_trades", "gross_wins_usd", "top_five_gross_wins_usd",
    "reconciled", "attribution_complete", "safety_breaches",
}
_SELECTED_EQUITY_CONTRACTS = {
    SELECTED_EQUITY_SCHEMA: {
        "evidence_key": "selected_equity",
        "unit": "$1_per_XSP_point",
        "suffix": "points",
        "fields": _SELECTED_EQUITY_FIELDS,
    },
    SELECTED_CASH_EQUITY_SCHEMA: {
        "evidence_key": "selected_cash_equity",
        "unit": "USD",
        "suffix": "usd",
        "fields": _SELECTED_CASH_EQUITY_FIELDS,
    },
}

@dataclass(frozen=True)
class XspProfitabilityPolicy:
    """Frozen identity and risk limits for one selected XSP run."""

    run_id: str
    strategy_id: str
    strategy_version: str
    config_fingerprint: str
    capital_sleeve: str
    max_drawdown_points: float
    max_session_loss_points: float
    minimum_week_closed_trades: int
    maximum_top_five_win_share: float
    slot_tolerance_seconds: float = 90.0
    unit: str = "$1_per_XSP_point"
    equity_schema: str = SELECTED_EQUITY_SCHEMA
    coverage_epoch_id: str | None = None
    coverage_started_at_utc: str | None = None


def xsp_profitability_contract(
    policy: XspProfitabilityPolicy,
) -> tuple[Mapping[str, object] | None, list[str]]:
    """Resolve and validate one selected-equity evidence contract."""

    contract = _SELECTED_EQUITY_CONTRACTS.get(policy.equity_schema)
    errors = []
    if not isinstance(contract, Mapping) or contract.get("unit") != policy.unit:
        errors.append("invalid_equity_contract")
    if not policy.run_id.strip():
        errors.append("missing_run_id")
    if not policy.strategy_id.strip() or policy.strategy_id.strip().upper() == "NO_TRADE":
        errors.append("no_selected_strategy")
    if not policy.strategy_version.strip():
        errors.append("missing_strategy_version")
    if not policy.config_fingerprint.strip():
        errors.append("missing_config_fingerprint")
    if not policy.capital_sleeve.strip():
        errors.append("missing_capital_sleeve")
    if not math.isfinite(policy.max_drawdown_points) or policy.max_drawdown_points < 0:
        errors.append("invalid_max_drawdown")
    if not math.isfinite(policy.max_session_loss_points) or policy.max_session_loss_points < 0:
        errors.append("invalid_max_session_loss")
    if policy.minimum_week_closed_trades < 2:
        errors.append("weekly_trade_floor_below_two")
    if not 0 < policy.maximum_top_five_win_share <= 1:
        errors.append("invalid_win_concentration_limit")
    if (
        not math.isfinite(policy.slot_tolerance_seconds)
        or not 0 <= policy.slot_tolerance_seconds < 150
    ):
        errors.append("invalid_slot_tolerance")
    if (policy.coverage_epoch_id is None) != (
        policy.coverage_started_at_utc is None
    ):
        errors.append("incomplete_coverage_epoch_identity")
    elif policy.coverage_epoch_id is not None:
        try:
            coverage_start = _utc(policy.coverage_started_at_utc)
        except (TypeError, ValueError):
            errors.append("invalid_coverage_epoch_start")
        else:
            if len(policy.coverage_epoch_id) != 64 or any(
                value not in "0123456789abcdef"
                for value in policy.coverage_epoch_id
            ):
                errors.append("invalid_coverage_epoch_identity")
            slots = xsp_rth_cash_evaluation_slots(
                coverage_start.astimezone(ET_ZONE).date()
            )
            if coverage_start not in {
                slot.astimezone(timezone.utc) for slot in slots
            }:
                errors.append("coverage_epoch_not_on_cash_slot")
    return contract, errors


def xsp_profitability_amount_fields(
    contract: Mapping[str, object],
) -> dict[str, str]:
    """Map normalized receipt amounts to their schema-specific field names."""

    suffix = str(contract["suffix"])
    return {
        name: f"{name}_{suffix}" if name != "open_mark" else f"open_mark_{suffix}"
        for name in (
            "cumulative_gross", "cumulative_cost", "cumulative_net",
            "cumulative_realized_net", "open_mark", "session_gross",
            "session_cost", "session_net", "gross_wins",
            "top_five_gross_wins",
        )
    }


def empty_xsp_profitability_receipt(
    *,
    policy: XspProfitabilityPolicy,
    observed_at: datetime,
    status: str,
    reasons: Sequence[str],
) -> dict[str, object]:
    """Return the shared fail-closed shape before a selected run starts."""

    return {
        "schema": LIVE_PROFITABILITY_SCHEMA,
        "authority": "selected_reconciled_economics_only",
        "as_of_utc": observed_at.isoformat(),
        "status": status,
        "policy": {
            "run_id": policy.run_id,
            "strategy_id": policy.strategy_id,
            "strategy_version": policy.strategy_version,
            "config_fingerprint": policy.config_fingerprint,
            "capital_sleeve": policy.capital_sleeve,
            "unit": policy.unit,
            "equity_schema": policy.equity_schema,
            **(
                {
                    "coverage_epoch_id": policy.coverage_epoch_id,
                    "coverage_started_at_utc": policy.coverage_started_at_utc,
                }
                if policy.coverage_epoch_id is not None
                else {}
            ),
        },
        "clock": {
            "run_started_at_utc": None,
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
        "reasons": list(reasons),
    }


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("XSP graduation timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_proof(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: graduation proof must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _graduation_gate(
    status: str,
    reasons: Sequence[str],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def xsp_runtime_parity_graduation_gate(
    path: Path,
    *,
    repo_root: Path,
    strategy_id: str = "",
) -> dict[str, object]:
    """Validate the one historical runtime/crown parity proof class."""

    try:
        proof, fingerprint = _load_json_proof(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _graduation_gate(
            "INVALID",
            ["runtime_parity_proof_unreadable"],
            {"path": str(path), "error": str(exc)},
        )
    reasons = []
    checks = proof.get("checks")
    expected = proof.get("expected")
    actual = proof.get("actual")
    inputs = proof.get("inputs")
    safety = proof.get("safety")
    schema = proof.get("schema")
    p009 = schema == XSP_P009_RUNTIME_PARITY_SCHEMA
    authority = (
        "offline_exact_dual_clock_historical_and_runtime_parity_only"
        if p009
        else "offline_historical_rth_parity_only"
    )
    owner_paths = xsp_runtime_parity_owner_paths(schema)
    if (
        schema not in {XSP_RUNTIME_PARITY_SCHEMA, XSP_P009_RUNTIME_PARITY_SCHEMA}
        or proof.get("authority") != authority
        or proof.get("passes") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        reasons.append("runtime_parity_verdict_invalid")
    comparison_fields = (
        (
            "full_combined_trades",
            "full_rth_trades",
            "full_combined_ledger_sha256",
            "full_rth_ledger_sha256",
        )
        if p009
        else (
            "trades",
            "net_points",
            "profit_factor",
            "max_drawdown_points",
            "ordered_ledger_sha256",
        )
    )
    if (
        not isinstance(expected, Mapping)
        or not isinstance(actual, Mapping)
        or any(expected.get(field) != actual.get(field) for field in comparison_fields)
    ):
        reasons.append("runtime_parity_result_mismatch")
    if not isinstance(inputs, Mapping):
        reasons.append("runtime_parity_inputs_missing")
    else:
        for field, relative in owner_paths.items():
            owner = repo_root / relative
            try:
                current = _file_sha256(owner)
            except OSError:
                reasons.append(f"runtime_owner_missing:{relative}")
                continue
            if inputs.get(field) != current:
                reasons.append(f"runtime_owner_drift:{relative}")
    if p009 and (
        proof.get("strategy_id")
        != "xsp.opening-edge-v4-dual-clock-arbitration-p009.v1"
        or (
            strategy_id
            and strategy_id
            != "xsp.opening-edge-v4-dual-clock-arbitration-p009.v1"
        )
    ):
        reasons.append("runtime_parity_strategy_mismatch")
    if (
        not isinstance(safety, Mapping)
        or safety.get("broker_queried") is not False
        or safety.get("service_or_timer_mutated") is not False
        or safety.get("selection_mutated") is not False
        or safety.get("submitted_orders") != 0
        or safety.get("profitability_clock_mutated") is not False
    ):
        reasons.append("runtime_parity_safety_boundary_invalid")
    return _graduation_gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "path": str(path),
            "sha256": fingerprint,
            "observed_at_utc": proof.get("observed_at_utc"),
            "ordered_ledger_sha256": (
                actual.get(
                    "full_combined_ledger_sha256"
                    if p009
                    else "ordered_ledger_sha256"
                )
                if isinstance(actual, Mapping)
                else None
            ),
        },
    )


def _selected_execution_rows(
    records: Sequence[Mapping[str, object]],
    *,
    policy: XspProfitabilityPolicy,
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(record)
        for record in records
        if record.get("kind") == "checkpoint"
        and record.get("strategy_id") == policy.strategy_id
        and record.get("strategy_version") == policy.strategy_version
        and isinstance(record.get("evidence"), Mapping)
        and record["evidence"].get("selection_id") == policy.run_id
    )


def _xsp_restart_gate(
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    policy: XspProfitabilityPolicy,
) -> dict[str, object]:
    state_rows = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("phase") == "STATE"
    ]
    if not state_rows:
        return _graduation_gate("HOLD", ["restart_checkpoint_missing"], {})
    reasons = []
    first = state_rows[0]
    first_evidence = first["evidence"]
    assert isinstance(first_evidence, Mapping)
    selected_at = _utc(str(selection["selected_at_utc"]))
    first_at = _utc(str(first["recorded_at_utc"]))
    baseline = selection.get("broker_at_selection")
    baseline_positions = (
        baseline.get("positions") if isinstance(baseline, Mapping) else None
    )
    plan = first_evidence.get("plan")
    broker = first_evidence.get("broker_state")
    risk = first_evidence.get("risk_state")
    if first_at < selected_at:
        reasons.append("restart_checkpoint_precedes_selection")
    elif first_at - selected_at > timedelta(
        seconds=300 + policy.slot_tolerance_seconds
    ):
        reasons.append("restart_checkpoint_late")
    if (
        not isinstance(baseline_positions, Mapping)
        or not isinstance(plan, Mapping)
        or plan.get("holdings") != baseline_positions
        or not isinstance(broker, Mapping)
        or broker.get("positions") != baseline_positions
        or not isinstance(risk, Mapping)
        or risk.get("holdings_from_fills") != {
            symbol: float(quantity)
            for symbol, quantity in baseline_positions.items()
        }
        or first_evidence.get("submitted_orders") != 0
    ):
        reasons.append("restart_baseline_mismatch")
    first_index = rows.index(first)
    if any(
        isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("phase") in {"PREPARED", "SUBMITTED", "TERMINAL"}
        for row in rows[:first_index]
    ):
        reasons.append("order_precedes_restart_checkpoint")
    if reasons:
        status = "INVALID"
    elif len(state_rows) < 2:
        status = "HOLD"
        reasons.append("first_natural_recurrence_missing")
    else:
        status = "PASS"
    return _graduation_gate(
        status,
        reasons,
        {
            "restart_checkpoint_id": first.get("checkpoint_id"),
            "restart_recorded_at_utc": first.get("recorded_at_utc"),
            "first_natural_checkpoint_id": (
                state_rows[1].get("checkpoint_id") if len(state_rows) > 1 else None
            ),
        },
    )


def _equity_matches_risk(
    equity: Mapping[str, object],
    risk: Mapping[str, object],
    *,
    policy: XspProfitabilityPolicy,
) -> bool:
    fields = {
        "cumulative_gross_usd": "run_gross_usd",
        "cumulative_cost_usd": "run_cost_usd",
        "cumulative_net_usd": "run_net_usd",
        "cumulative_realized_net_usd": "run_realized_net_usd",
        "open_mark_usd": "open_mark_net_usd",
        "session_gross_usd": "session_gross_usd",
        "session_cost_usd": "session_cost_usd",
        "session_net_usd": "session_net_usd",
        "gross_wins_usd": "gross_wins_usd",
        "top_five_gross_wins_usd": "top_five_gross_wins_usd",
    }
    try:
        numeric_match = all(
            abs(float(equity[left]) - float(risk[right])) <= 1e-7
            for left, right in fields.items()
        )
        return bool(
            numeric_match
            and equity.get("run_id") == policy.run_id
            and equity.get("config_fingerprint") == policy.config_fingerprint
            and equity.get("capital_sleeve") == policy.capital_sleeve
            and int(equity["closed_trades"]) == int(risk["closed_trades"])
            and equity.get("reconciled") is True
            and equity.get("attribution_complete")
            is (risk.get("attribution_complete") is True)
            and equity.get("safety_breaches") == risk.get("safety_breaches")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _xsp_cash_risk_gates(
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    policy: XspProfitabilityPolicy,
) -> tuple[dict[str, object], dict[str, object]]:
    state_rows = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("phase") == "STATE"
    ]
    if not state_rows:
        hold = _graduation_gate("HOLD", ["selected_risk_state_missing"], {})
        return hold, _graduation_gate("HOLD", ["attribution_not_observed"], {})
    invalid = []
    stops = []
    attribution = []
    prior_fills = prior_cost = prior_trades = 0.0
    latest_risk: Mapping[str, object] = {}
    for row in state_rows:
        evidence = row["evidence"]
        assert isinstance(evidence, Mapping)
        risk = evidence.get("risk_state")
        broker = evidence.get("broker_state")
        equity = evidence.get("selected_cash_equity")
        plan = evidence.get("plan")
        if (
            not isinstance(risk, Mapping)
            or risk.get("valid") is not True
            or not isinstance(broker, Mapping)
            or not isinstance(equity, Mapping)
            or not isinstance(plan, Mapping)
            or not _equity_matches_risk(equity, risk, policy=policy)
        ):
            invalid.append("selected_risk_projection_invalid")
            continue
        latest_risk = risk
        try:
            fill_count = float(risk["fill_count"])
            cost = float(risk["run_cost_usd"])
            closed_trades = float(risk["closed_trades"])
            settled_cash = float(risk["settled_cash_usd"])
            drawdown = float(risk["drawdown_usd"])
            session_net = float(risk["session_net_usd"])
            holdings = risk["holdings_from_fills"]
            holding_values = (
                [float(value) for value in holdings.values()]
                if isinstance(holdings, Mapping)
                else []
            )
        except (KeyError, TypeError, ValueError):
            invalid.append("selected_risk_economics_invalid")
            continue
        if (
            not all(
                math.isfinite(value)
                for value in (
                    fill_count,
                    cost,
                    closed_trades,
                    settled_cash,
                    drawdown,
                    session_net,
                    *holding_values,
                )
            )
            or fill_count < 0
            or not fill_count.is_integer()
            or closed_trades < 0
            or not closed_trades.is_integer()
            or cost < -1e-9
            or drawdown < -1e-9
            or any(value < -1e-9 for value in holding_values)
        ):
            invalid.append("selected_risk_economics_invalid")
            continue
        if (
            fill_count < prior_fills
            or cost < prior_cost - 1e-9
            or closed_trades < prior_trades
        ):
            invalid.append("selected_risk_state_nonmonotonic")
        prior_fills, prior_cost, prior_trades = fill_count, cost, closed_trades
        if (
            not isinstance(holdings, Mapping)
            or broker.get("positions") != holdings
            or plan.get("holdings") != holdings
            or sum(value > 1e-9 for value in holding_values) > 1
        ):
            invalid.append("selected_holdings_do_not_reconcile")
        if settled_cash < -1e-7:
            stops.append("illegal_negative_settled_cash")
        breaches = risk.get("safety_breaches")
        if not isinstance(breaches, list):
            invalid.append("selected_safety_evidence_invalid")
        elif breaches:
            stops.extend(str(breach) for breach in breaches)
        if drawdown > policy.max_drawdown_points + 1e-9:
            stops.append("drawdown_limit_breached")
        if session_net < -policy.max_session_loss_points - 1e-9:
            stops.append("session_loss_limit_breached")
        if risk.get("attribution_complete") is not True:
            attribution.append("selected_attribution_incomplete")
    risk_status = "STOP" if stops else "INVALID" if invalid else "PASS"
    attribution_status = "INVALID" if attribution else "PASS"
    return (
        _graduation_gate(
            risk_status,
            [*stops, *invalid],
            {
                "state_rows": len(state_rows),
                "latest_risk_fingerprint": evidence_sha256(latest_risk),
                "fill_count": latest_risk.get("fill_count"),
                "closed_trades": latest_risk.get("closed_trades"),
                "settled_cash_usd": latest_risk.get("settled_cash_usd"),
                "drawdown_usd": latest_risk.get("drawdown_usd"),
                "session_net_usd": latest_risk.get("session_net_usd"),
            },
        ),
        _graduation_gate(
            attribution_status,
            attribution,
            {
                "state_rows": len(state_rows),
                "attribution_complete": not attribution,
            },
        ),
    )


def xsp_execution_graduation_gate(
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    order_rows = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and str(row["evidence"].get("order_ref") or "")
    ]
    if not order_rows:
        return _graduation_gate(
            "HOLD",
            ["execution_not_observed"],
            {"orders": 0, "terminal_orders": 0, "fills": 0},
        )
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in order_rows:
        evidence = row["evidence"]
        assert isinstance(evidence, Mapping)
        groups.setdefault(str(evidence["order_ref"]), []).append(evidence)
    invalid = []
    failures = []
    stops = []
    pending = []
    terminal_orders = fills = quote_ineligible_transitions = 0
    commission_total = price_improvement = 0.0
    seen_transitions: dict[str, str] = {}
    seen_exec_ids: set[str] = set()
    nominee = selection.get("nominee")
    commission_limits = (
        nominee.get("commission_limits_usd") if isinstance(nominee, Mapping) else None
    )
    contract_ids = nominee.get("contract_ids") if isinstance(nominee, Mapping) else None
    execution = selection.get("execution")
    policy_contract = (
        execution.get("policy_contract") if isinstance(execution, Mapping) else None
    )
    try:
        auto_timeout = (
            float(policy_contract.get("auto_timeout_seconds"))
            if isinstance(policy_contract, Mapping)
            else 0.0
        )
    except (TypeError, ValueError):
        auto_timeout = 0.0
    allowed_modes = {"OPT", "MID", "AGG", "CROSS", "RLT"}
    for order_ref, evidence_rows in groups.items():
        plans = [row.get("plan") for row in evidence_rows]
        if any(not isinstance(plan, Mapping) for plan in plans):
            invalid.append("order_plan_missing")
            continue
        plan = plans[0]
        assert isinstance(plan, Mapping)
        if any(evidence_sha256(value) != evidence_sha256(plan) for value in plans[1:]):
            invalid.append("order_plan_changed")
        transition = str(plan.get("transition_id") or "")
        expected_ref = f"XSPV3-{transition[:24]}"
        if len(transition) != 64 or order_ref != expected_ref:
            invalid.append("order_transition_identity_invalid")
        prior_ref = seen_transitions.setdefault(transition, order_ref)
        if prior_ref != order_ref:
            stops.append("duplicate_transition_submission")
        phases = [str(row.get("phase") or "") for row in evidence_rows]
        phase_rank = {"PREPARED": 0, "SUBMITTED": 1, "TERMINAL": 2}
        if (
            phases[0] != "PREPARED"
            or any(phase not in phase_rank for phase in phases)
            or any(
                phase_rank[current] < phase_rank[prior]
                for prior, current in zip(phases, phases[1:])
                if prior in phase_rank and current in phase_rank
            )
            or phases.count("PREPARED") != 1
            or phases.count("TERMINAL") > 1
        ):
            invalid.append("order_lifecycle_prefix_invalid")
        if any(int(row.get("submitted_orders") or 0) not in {0, 1} for row in evidence_rows):
            stops.append("duplicate_or_unauthorized_submission")
        leg = plan.get("leg")
        if not isinstance(leg, Mapping):
            invalid.append("order_leg_missing")
            continue
        symbol = str(leg.get("symbol") or "")
        action = str(leg.get("action") or "").upper()
        try:
            admission = validate_live_capital_decision(
                plan.get("capital_admission") or {}
            )
        except (TypeError, ValueError):
            invalid.append("capital_admission_invalid")
        else:
            expected_intents = (
                {"ENTER", "INCREASE", "ROTATE_IN"}
                if action == "BUY"
                else {"EXIT", "REDUCE", "ROTATE_OUT"}
            )
            if (
                admission.get("status") != "ALLOW"
                or admission.get("intent") not in expected_intents
                or admission.get("run_id") != selection.get("selection_id")
            ):
                invalid.append("capital_admission_not_allowed")
        try:
            quantity_value = float(leg["quantity"])
            quantity = int(quantity_value)
            commission_limit = float(commission_limits[symbol])
            contract_id = int(contract_ids[symbol])
        except (KeyError, TypeError, ValueError):
            invalid.append("order_contract_identity_invalid")
            continue
        try:
            bid = float(leg["bid"])
            ask = float(leg["ask"])
        except (KeyError, TypeError, ValueError):
            bid = ask = math.nan
        if (
            not math.isfinite(quantity_value)
            or quantity_value != quantity
            or quantity <= 0
            or not math.isfinite(commission_limit)
            or commission_limit < 0
            or contract_id <= 0
            or action not in {"BUY", "SELL"}
            or symbol not in {"UPRO", "SPXU"}
            or not math.isfinite(bid)
            or not math.isfinite(ask)
            or bid <= 0
            or ask < bid
            or leg.get("outside_rth") is not False
        ):
            invalid.append("order_contract_identity_invalid")
            continue
        previews = [row.get("what_if_preview") for row in evidence_rows if row.get("what_if_preview")]
        if not previews or any(not isinstance(preview, Mapping) for preview in previews):
            invalid.append("order_preview_missing")
        else:
            preview = previews[0]
            try:
                values = [
                    float(value)
                    for value in (
                        preview.get("commission"),
                        preview.get("min_commission"),
                        preview.get("max_commission"),
                    )
                    if value is not None
                ]
            except (TypeError, ValueError):
                values = []
            if (
                not values
                or not all(math.isfinite(value) and value >= 0 for value in values)
                or str(preview.get("commission_currency") or "").upper() != "USD"
                or max(values) > commission_limit + 0.01
            ):
                failures.append("broker_preview_commission_exceeded")
        terminal = next(
            (row for row in reversed(evidence_rows) if row.get("phase") == "TERMINAL"),
            None,
        )
        if terminal is None:
            pending.append("order_terminal_state_pending")
            continue
        terminal_orders += 1
        order = terminal.get("broker_order")
        if not isinstance(order, Mapping) or order.get("done") is not True:
            invalid.append("terminal_broker_order_incomplete")
            continue
        try:
            filled = float(order["filled"])
            remaining = float(order["remaining"])
            limit_price = float(order["limit_price"])
            broker_fills = order["fills"]
        except (KeyError, TypeError, ValueError):
            invalid.append("terminal_broker_order_invalid")
            continue
        if (
            order.get("order_ref") != order_ref
            or order.get("symbol") != symbol
            or int(order.get("con_id") or 0) != contract_id
            or str(order.get("action") or "").upper() != action
            or float(order.get("quantity") or 0) != quantity
            or filled < 0
            or remaining < 0
            or filled + remaining > quantity + 1e-9
            or not isinstance(broker_fills, list)
        ):
            invalid.append("terminal_order_identity_mismatch")
            continue
        fill_total = 0.0
        order_commission = 0.0
        for fill in broker_fills:
            try:
                shares = float(fill["shares"])
                price = float(fill["price"])
                commission = float(fill["commission"])
                side = str(fill["side"]).upper()
            except (KeyError, TypeError, ValueError):
                invalid.append("broker_fill_invalid")
                continue
            exec_id = str(fill.get("exec_id") or "")
            if (
                not exec_id
                or not str(fill.get("time_utc") or "")
                or fill.get("symbol") != symbol
                or side not in ({"BOT", "BUY"} if action == "BUY" else {"SLD", "SELL"})
                or shares <= 0
                or price <= 0
                or commission < 0
                or str(fill.get("commission_currency") or "").upper() != "USD"
            ):
                invalid.append("broker_fill_identity_mismatch")
            if exec_id in seen_exec_ids:
                stops.append("duplicate_broker_execution")
            seen_exec_ids.add(exec_id)
            if (action == "BUY" and price > limit_price + 1e-9) or (
                action == "SELL" and price < limit_price - 1e-9
            ):
                failures.append("fill_worse_than_submitted_limit")
            fill_total += shares
            order_commission += commission
            price_improvement += (
                (limit_price - price) if action == "BUY" else (price - limit_price)
            ) * shares
        if abs(fill_total - filled) > 1e-9:
            invalid.append("terminal_fill_quantity_mismatch")
        if order_commission > commission_limit + 0.01:
            failures.append("actual_commission_exceeded")
        fills += len(broker_fills)
        commission_total += order_commission
        transitions = [
            row.get("ladder_transition")
            for row in evidence_rows
            if row.get("ladder_transition") is not None
        ]
        for transition_row in transitions:
            try:
                elapsed = float(transition_row.get("elapsed_seconds") or 0.0)
            except (AttributeError, TypeError, ValueError):
                elapsed = math.nan
            if (
                not isinstance(transition_row, Mapping)
                or transition_row.get("schema") != "xsp.execution-ladder-transition.v1"
                or transition_row.get("event") != "ladder_mode_transition"
                or transition_row.get("active_mode") not in allowed_modes
                or not math.isfinite(elapsed)
                or elapsed < 0
                or elapsed > auto_timeout + 1e-9
                or str(transition_row.get("action") or "").upper() != action
            ):
                failures.append("execution_ladder_contract_breached")
            elif transition_row.get("quote_eligible") is not True:
                quote_ineligible_transitions += 1
    status = (
        "STOP"
        if stops
        else "INVALID"
        if invalid
        else "FAIL"
        if failures
        else "HOLD"
        if pending
        else "PASS"
    )
    return _graduation_gate(
        status,
        [*stops, *invalid, *failures, *pending],
        {
            "orders": len(groups),
            "terminal_orders": terminal_orders,
            "fills": fills,
            "commission_usd": commission_total,
            "limit_price_improvement_usd": price_improvement,
            "quote_ineligible_transitions": quote_ineligible_transitions,
        },
    )


def xsp_live_graduation_inputs(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    cutoff_utc: datetime | str,
    policy: XspProfitabilityPolicy,
    profitability_receipt: Mapping[str, object],
    runtime_parity_path: Path,
    capital_owner_stability_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Project XSP raw truth into the generic live-graduation contract."""

    cutoff = _utc(cutoff_utc)
    selection_sha = _file_sha256(selection_path)
    prefix, projected = live_calibration_logical_prefix(
        records,
        cutoff_utc=cutoff,
    )
    rows = _selected_execution_rows(projected, policy=policy)
    restart = _xsp_restart_gate(selection, rows, policy=policy)
    risk, attribution = _xsp_cash_risk_gates(selection, rows, policy=policy)
    execution = xsp_execution_graduation_gate(selection, rows)
    broker = selection.get("broker_at_selection")
    account_id = str(broker.get("account_id") or "") if isinstance(broker, Mapping) else ""
    subject = {
        "strategy_id": policy.strategy_id,
        "strategy_version": policy.strategy_version,
        "signal_instrument": "XSP",
        "execution_sleeve": "UPRO/SPXU",
        "capital_sleeve": policy.capital_sleeve,
        "selection_id": str(selection["selection_id"]),
        "run_id": policy.run_id,
        "account_fingerprint": hashlib.sha256(account_id.encode()).hexdigest(),
    }
    selection_identity = {
        "path": str(selection_path),
        "selection_id": str(selection["selection_id"]),
        "run_id": policy.run_id,
        "selected_at_utc": selection.get("selected_at_utc"),
        "run_started_at_utc": selection.get("run_started_at_utc"),
        "signal_strategy_version": selection.get("strategy_version"),
        "execution_strategy_version": policy.strategy_version,
        "capital_sleeve": policy.capital_sleeve,
        "selection_file_sha256": selection_sha,
        **(
            {
                "coverage_epoch_id": policy.coverage_epoch_id,
                "coverage_started_at_utc": policy.coverage_started_at_utc,
            }
            if policy.coverage_epoch_id is not None
            else {}
        ),
    }
    return {
        "subject": subject,
        "selection": selection_identity,
        "selection_file_sha256": selection_sha,
        "ledger_prefix": {
            **prefix,
            **(
                {"coverage_epoch_id": policy.coverage_epoch_id}
                if policy.coverage_epoch_id is not None
                else {}
            ),
            "gates": {
                "restart": restart,
                "cash_risk_safety": risk,
                "attribution": attribution,
                "execution": execution,
            },
        },
        "profitability_receipt": dict(profitability_receipt),
        "runtime_parity_proof": xsp_runtime_parity_graduation_gate(
            runtime_parity_path,
            repo_root=repo_root,
            strategy_id=policy.strategy_id,
        ),
        "capital_owner_stability_proof": (
            xsp_capital_owner_stability_graduation_gate(
                capital_owner_stability_path,
                repo_root=repo_root,
                selection_id=str(selection["selection_id"]),
                selection_file_sha256=selection_sha,
                records=records,
                strategy_id=policy.strategy_id,
                strategy_version=policy.strategy_version,
            )
        ),
    }
