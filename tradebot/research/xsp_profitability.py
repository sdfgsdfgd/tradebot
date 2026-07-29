"""Selected-equity contracts for XSP live profitability receipts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence


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
