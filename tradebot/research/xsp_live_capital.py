"""XSP adapter for the shared account capital-sleeve owner."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from ..live.capital import (
    admit_live_capital,
    build_live_capital_plan,
    validate_live_capital_decision,
)
from .xsp_live_transport import XSP_V3_TRANSPORT_CAPITAL_SLEEVE


XSP_LIVE_CAPITAL_SLEEVE_ID = XSP_V3_TRANSPORT_CAPITAL_SLEEVE


def build_xsp_live_capital_plan(
    selection: Mapping[str, object],
    *,
    selection_path: Path,
    created_at_utc: datetime | str,
    supersedes_plan_id: str | None = None,
) -> dict[str, object]:
    """Derive XSP's exact qualified pool without consuming spare cash."""

    broker = selection.get("broker_at_selection")
    nominee = selection.get("nominee")
    if not isinstance(broker, Mapping) or not isinstance(nominee, Mapping):
        raise ValueError("selected XSP capital evidence is incomplete")
    commissions = nominee.get("commission_limits_usd")
    if not isinstance(commissions, Mapping) or not commissions:
        raise ValueError("selected XSP commission ceiling is missing")
    notional = float(nominee.get("fixed_entry_notional_usd") or 0)
    maximum_commission = max(float(value) for value in commissions.values())
    minimum_cash = float(broker.get("minimum_settled_cash_usd") or 0)
    if (
        notional <= 0
        or maximum_commission < 0
        or abs(minimum_cash - notional - maximum_commission) > 1e-8
    ):
        raise ValueError("selected XSP capital and commission identity disagree")
    return build_live_capital_plan(
        account_id=str(broker.get("account_id") or ""),
        account_type=str(broker.get("account_type") or ""),
        currency="USD",
        observed_settled_cash_usd=broker.get("settled_cash_usd"),
        managed_capital_usd=minimum_cash,
        sleeves=[
            {
                "sleeve_id": XSP_LIVE_CAPITAL_SLEEVE_ID,
                "strategy_id": str(selection.get("strategy_version") or ""),
                "run_id": str(selection.get("selection_id") or ""),
                "selection_path": str(selection_path),
                "selection_file_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
                "capital_kind": "CASH_DEBIT",
                "weight_bps": 10_000,
            }
        ],
        reserve_reasons=[
            "cash_above_selected_authority_unallocated",
            "selected_fixed_notional_and_commission_ceiling",
            "unrelated_positions_excluded_from_selected_sleeve",
        ],
        created_at_utc=created_at_utc,
        supersedes_plan_id=supersedes_plan_id,
    )


def apply_xsp_live_capital(
    projected: Mapping[str, object],
    *,
    capital_plan: Mapping[str, object] | None,
    selection: Mapping[str, object],
    selection_file_sha256: str,
    available_cash_usd: float,
) -> dict[str, object]:
    """Attach one allocation receipt and fail a new BUY closed."""

    plan = dict(projected)
    if plan.get("status") != "ACTIONABLE":
        return plan
    leg = plan.get("leg")
    broker = selection.get("broker_at_selection")
    if not isinstance(leg, Mapping) or not isinstance(broker, Mapping):
        raise ValueError("actionable XSP plan has no capital identity")
    action = str(leg.get("action") or "").upper()
    if action == "BUY":
        required = leg.get("required_settled_cash_usd")
        if required is None:
            raise ValueError("actionable XSP BUY has no cash requirement")
        intent = "ROTATE_IN"
        projected_capital = cash_debit = required
    elif action == "SELL":
        intent = (
            "ROTATE_OUT"
            if plan.get("reason") == "sell_incumbent_before_target"
            else "EXIT"
        )
        projected_capital = cash_debit = 0
    else:
        raise ValueError("actionable XSP leg has an invalid capital action")
    decision = admit_live_capital(
        capital_plan,
        intent=intent,
        account_id=str(broker.get("account_id") or ""),
        account_type=str(broker.get("account_type") or "CASH"),
        currency="USD",
        sleeve_id=XSP_LIVE_CAPITAL_SLEEVE_ID,
        run_id=str(selection.get("selection_id") or ""),
        selection_file_sha256=selection_file_sha256,
        capital_kind="CASH_DEBIT",
        projected_capital_usd=projected_capital,
        cash_debit_usd=cash_debit,
        available_cash_usd=available_cash_usd,
    )
    admitted = {**plan, "capital_admission": decision}
    if decision["status"] == "ALLOW":
        return admitted
    return {
        **admitted,
        "status": "CAPITAL_HOLD",
        "reason": "capital_allocation_blocked",
        "blocked_leg": dict(leg),
        "leg": None,
    }


def xsp_pending_buy_has_capital_reservation(plan: Mapping[str, object]) -> bool:
    """Prove that a durable pending BUY was admitted before submission."""

    leg = plan.get("leg")
    decision = plan.get("capital_admission")
    if not isinstance(leg, Mapping) or str(leg.get("action") or "").upper() != "BUY":
        return True
    try:
        validated = validate_live_capital_decision(decision or {})
    except (TypeError, ValueError):
        return False
    return bool(validated["status"] == "ALLOW" and validated.get("plan_id"))
