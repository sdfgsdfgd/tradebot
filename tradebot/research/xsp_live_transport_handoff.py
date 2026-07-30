"""One-time position-preserving handoff for the v3 cash transport."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .live_calibration import calibration_fingerprint
from .xsp_execution_observer import xsp_v2_position_state
from .xsp_live_transport import (
    XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
    XSP_V3_ROTATION_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    _BROKER_SNAPSHOT_MAX_AGE_SECONDS,
    _SOURCE_MAX_AGE_SECONDS,
    _V3_DIRECTION_SYMBOL,
    _V3_SYMBOLS,
    _load,
    _number,
    _sha256,
    _utc,
)
from .xsp_live_transport_v3 import (
    _CONTINUITY_SCHEMA,
    _IMMEDIATE_PROCEEDS_SCHEMA,
    _IMMEDIATE_PROCEEDS_SHA256,
    _RESET_SCHEMA,
    _inherited_cash_and_holdings,
    _inherited_fills,
    _sha256_identity,
    load_xsp_v3_transport_selection_from_mapping,
    select_xsp_v3_transport,
)
from .xsp_opening_edge_v3 import XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA


def _immediate_proceeds_capability(path: Path) -> dict[str, object]:
    capability = _load(path)
    if (
        _sha256(path) != _IMMEDIATE_PROCEEDS_SHA256
        or capability.get("schema") != _IMMEDIATE_PROCEEDS_SCHEMA
        or capability.get("verdict")
        != "PASS_IMPLEMENTATION_AND_FRESH_POST_SELL_BROKER_RECONCILIATION_REQUIRED"
        or capability.get("order_authority") != "none"
        or capability.get("submitted_orders") != 0
        or capability.get("historical_replay", {}).get("all_gates_pass") is not True
        or capability.get("broker_preview", {}).get("matching_open_orders_after") != 0
    ):
        raise ValueError("immediate-proceeds capability receipt is invalid")
    return capability


def handoff_xsp_v3_immediate_proceeds(
    *,
    predecessor: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    source_receipt: Mapping[str, object],
    broker_snapshot: Mapping[str, object],
    immediate_proceeds_receipt_path: Path,
    selected_at: datetime,
) -> dict[str, object]:
    """Freeze a restart-safe live successor around one inherited v3 position."""

    prior = load_xsp_v3_transport_selection_from_mapping(predecessor)
    if prior["schema"] != XSP_V3_TRANSPORT_SELECTION_SCHEMA:
        raise ValueError("immediate-proceeds handoff requires the original v3 selection")
    capability = _immediate_proceeds_capability(immediate_proceeds_receipt_path)

    selected_utc = _utc(selected_at)
    source_at = _utc(source_receipt.get("recorded_at_utc"))
    broker_at = _utc(broker_snapshot.get("observed_at_utc"))
    cash_at = _utc(broker_snapshot.get("cash_observed_at_utc"))
    paired = source_receipt.get("paired_equity")
    if (
        not 0 <= (selected_utc - source_at).total_seconds() <= _SOURCE_MAX_AGE_SECONDS
        or source_receipt.get("evaluation_status") != "EVALUATED"
        or source_receipt.get("freshness_ok") is not True
        or source_receipt.get("session") != "RTH"
        or source_receipt.get("order_authority") != "none"
        or not _sha256_identity(source_receipt.get("checkpoint_id"))
        or not isinstance(paired, Mapping)
        or not 0
        <= (selected_utc - broker_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or not 0
        <= (broker_at - cash_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
    ):
        raise ValueError("immediate-proceeds handoff requires fresh source and broker state")

    _, source_target = xsp_v2_position_state(paired)
    if source_target is not None and (
        not isinstance(source_target, Mapping)
        or source_target.get("lane") != "rth"
        or str(source_target.get("direction") or "") not in _V3_DIRECTION_SYMBOL
    ):
        raise ValueError("handoff source target is not executable in RTH")
    positions = broker_snapshot.get("positions")
    open_orders = broker_snapshot.get("open_orders")
    unrelated = broker_snapshot.get("unrelated_positions")
    broker_cash = _number(
        broker_snapshot.get("settled_cash_usd"),
        name="handoff USD cash",
    )
    if (
        not isinstance(positions, Mapping)
        or not isinstance(open_orders, Sequence)
        or isinstance(open_orders, (str, bytes))
        or bool(open_orders)
        or not isinstance(unrelated, Sequence)
        or isinstance(unrelated, (str, bytes))
        or str(broker_snapshot.get("account_id") or "")
        != str(prior["broker_at_selection"]["account_id"])
        or str(broker_snapshot.get("account_type") or "").upper() != "CASH"
    ):
        raise ValueError("handoff does not own one reconciled RTH cash position")

    fills = _inherited_fills(records, selection_id=str(prior["selection_id"]))
    starting_cash = float(prior["broker_at_selection"]["settled_cash_usd"])
    replay_cash, replay_holdings = _inherited_cash_and_holdings(
        fills,
        starting_cash_usd=starting_cash,
    )
    broker_holdings = {
        candidate: int(
            _number(positions.get(candidate, 0), name=f"{candidate} position")
        )
        for candidate in _V3_SYMBOLS
    }
    held_symbols = [
        candidate
        for candidate, quantity in broker_holdings.items()
        if quantity > 0
    ]
    if (
        replay_holdings != broker_holdings
        or len(held_symbols) != 1
        or abs(replay_cash - broker_cash) > 0.02
    ):
        raise ValueError("broker position/cash does not match predecessor fills")
    held_symbol = held_symbols[0]
    held_direction = next(
        direction
        for direction, candidate in _V3_DIRECTION_SYMBOL.items()
        if candidate == held_symbol
    )

    context = paired.get("daily_context_state")
    context_state = context.get("state") if isinstance(context, Mapping) else None
    if (
        not isinstance(context, Mapping)
        or context.get("schema") != XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA
        or not isinstance(context_state, Mapping)
        or context.get("state_fingerprint")
        != calibration_fingerprint(context_state)
    ):
        raise ValueError("handoff source has no canonical daily context")
    nominee = json.loads(json.dumps(prior["nominee"]))
    nominee["capital_identity"] = {
        "starting_cash_identity_usd": 900.0,
        "fixed_entry_notional_usd": 900.0,
        "cash_slots": 1,
        "maximum_gross_purchase_notional_usd": 900.0,
        "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
        "unsettled_sale_proceeds_reused": True,
    }
    continuity = {
        "schema": _CONTINUITY_SCHEMA,
        "predecessor_schema": prior["schema"],
        "predecessor_selection_id": prior["selection_id"],
        "predecessor_run_started_at_utc": prior["run_started_at_utc"],
        "predecessor_starting_cash_usd": starting_cash,
        "inherited_holding_direction": held_direction,
        "source_target_state": (
            dict(source_target) if isinstance(source_target, Mapping) else None
        ),
        "inherited_fills": fills,
        "inherited_fill_ledger_fingerprint": calibration_fingerprint(fills),
    }
    evidence = {
        **json.loads(json.dumps(prior["evidence"])),
        "source_checkpoint_id": source_receipt["checkpoint_id"],
        "source_recorded_at_utc": source_at.isoformat(),
        "source_daily_context": {
            "schema": context["schema"],
            "trading_day": context["trading_day"],
            "context_as_of_day": context["context_as_of_day"],
            "state_fingerprint": context["state_fingerprint"],
        },
        "immediate_proceeds": {
            "path": str(immediate_proceeds_receipt_path),
            "sha256": _sha256(immediate_proceeds_receipt_path),
            "official_account_contract_url": capability[
                "official_account_contract"
            ]["url"],
            "broker_preview_sha256": capability["broker_preview"]["result_sha256"],
        },
    }
    body = {
        **{
            key: json.loads(json.dumps(value))
            for key, value in prior.items()
            if key not in {"selection_id", "schema", "selected_at_utc", "run_started_at_utc"}
        },
        "schema": XSP_V3_ROTATION_SELECTION_SCHEMA,
        "selected_at_utc": selected_utc.isoformat(),
        "run_started_at_utc": selected_utc.isoformat(),
        "nominee": nominee,
        "baseline_state": continuity["source_target_state"],
        "continuity": continuity,
        "broker_at_selection": {
            "observed_at_utc": broker_at.isoformat(),
            "cash_observed_at_utc": cash_at.isoformat(),
            "account_id": broker_snapshot["account_id"],
            "account_type": "CASH",
            "settled_cash_usd": broker_cash,
            "positions": broker_holdings,
            "unrelated_positions": [dict(row) for row in unrelated],
            "open_orders": [],
        },
        "risk": {
            "starting_cash_identity_usd": 900.0,
            "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        },
        "evidence": evidence,
    }
    selected = {**body, "selection_id": calibration_fingerprint(body)}
    return load_xsp_v3_transport_selection_from_mapping(selected)


def rebase_xsp_v3_immediate_proceeds(
    *,
    predecessor: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    source_receipt: Mapping[str, object],
    broker_snapshot: Mapping[str, object],
    cash_receipt_path: Path,
    preview_path: Path,
    immediate_proceeds_receipt_path: Path,
    selected_at: datetime,
    rth_scope_accepted: bool,
) -> dict[str, object]:
    """Freeze a zero-inheritance immediate-proceeds run from terminal flat state."""

    from .xsp_live_transport_risk import xsp_transport_risk_state

    prior = load_xsp_v3_transport_selection_from_mapping(predecessor)
    if prior["schema"] != XSP_V3_ROTATION_SELECTION_SCHEMA:
        raise ValueError("clean rebase requires an immediate-proceeds predecessor")
    capability = _immediate_proceeds_capability(immediate_proceeds_receipt_path)
    selected_utc = _utc(selected_at)
    terminal = xsp_transport_risk_state(
        selection=prior,
        records=tuple(records),
        observed_at=selected_utc,
        liquidation_bids={},
    )
    broker_cash = _number(
        broker_snapshot.get("settled_cash_usd"),
        name="rebase USD cash",
    )
    if (
        terminal["holdings_from_fills"] != {"UPRO": 0.0, "SPXU": 0.0}
        or terminal["closed_trades"] < 1
        or terminal["fill_count"] < 2
        or terminal["pending_settlement_usd"] != 0
        or terminal["safety_breaches"]
        or abs(float(terminal["settled_cash_usd"]) - broker_cash) > 0.02
        or str(broker_snapshot.get("account_id") or "")
        != str(prior["broker_at_selection"]["account_id"])
    ):
        raise ValueError("clean rebase predecessor is not terminal and flat")
    base = select_xsp_v3_transport(
        cash_receipt_path=cash_receipt_path,
        preview_path=preview_path,
        source_receipt=source_receipt,
        broker_snapshot=broker_snapshot,
        selected_at=selected_utc,
        rth_scope_accepted=rth_scope_accepted,
    )
    if base["baseline_state"] is not None:
        raise ValueError("clean rebase requires a flat v3 source")
    nominee = json.loads(json.dumps(base["nominee"]))
    nominee["capital_identity"] = {
        "starting_cash_identity_usd": 900.0,
        "fixed_entry_notional_usd": 900.0,
        "cash_slots": 1,
        "maximum_gross_purchase_notional_usd": 900.0,
        "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
        "unsettled_sale_proceeds_reused": True,
    }
    evidence = json.loads(json.dumps(base["evidence"]))
    evidence["immediate_proceeds"] = {
        "path": str(immediate_proceeds_receipt_path),
        "sha256": _sha256(immediate_proceeds_receipt_path),
        "official_account_contract_url": capability["official_account_contract"][
            "url"
        ],
        "broker_preview_sha256": capability["broker_preview"]["result_sha256"],
    }
    reset = {
        "schema": _RESET_SCHEMA,
        "predecessor_schema": prior["schema"],
        "predecessor_selection_id": prior["selection_id"],
        "predecessor_run_started_at_utc": prior["run_started_at_utc"],
        "predecessor_fill_ledger_fingerprint": terminal[
            "fill_ledger_fingerprint"
        ],
        "predecessor_risk_state_fingerprint": calibration_fingerprint(terminal),
        "predecessor_realized_net_usd": terminal["run_realized_net_usd"],
        "predecessor_closed_trades": terminal["closed_trades"],
        "source_target_state": None,
    }
    body = {
        **{
            key: json.loads(json.dumps(value))
            for key, value in base.items()
            if key not in {"selection_id", "schema", "nominee", "risk", "evidence"}
        },
        "schema": XSP_V3_ROTATION_SELECTION_SCHEMA,
        "nominee": nominee,
        "risk": {
            "starting_cash_identity_usd": 900.0,
            "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        },
        "evidence": evidence,
        "reset": reset,
    }
    selected = {**body, "selection_id": calibration_fingerprint(body)}
    return load_xsp_v3_transport_selection_from_mapping(selected)
