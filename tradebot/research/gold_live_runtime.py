"""Restart-safe one-contract execution for the selected Stage-76 gold run."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone

from ib_insync import Contract

from ..backtest.quotes import contract_from_ticker
from ..engines.execution import EXECUTION_POLICY, quote_health
from ..live.capital import admit_live_capital
from ..live.execution import LiveOrderExecution, order_ids
from ..live.order_evidence import (
    broker_trade_snapshot,
    execution_price_for_ticker,
    terminal_broker_snapshot_complete,
)
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_EXECUTION_SCHEMA,
    GOLD_LIVE_EXECUTION_VERSION,
    GOLD_LIVE_MAX_COMMISSION_USD,
    GOLD_LIVE_ORDER_AUTHORITY,
    GOLD_LIVE_PLAN_SCHEMA,
    GOLD_REGIME_HARMONY_SOURCE_VERSION,
    load_gold_live_selection_from_mapping,
)
from .gold_live_state import (
    GOLD_LIVE_ACTION_SOURCE_MAX_AGE_SECONDS,
    gold_transport_risk_state,
    project_gold_transport_plan,
)
from .gold_regime_harmony import GOLD_REGIME_HARMONY_VERSION
from .live_calibration import LiveCalibrationLedger


GOLD_LIVE_ORDER_REF_PREFIX = "GOLD76"


def _number(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("gold live timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def latest_gold_source_checkpoint(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for record in reversed(records):
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version")
            == GOLD_REGIME_HARMONY_SOURCE_VERSION
            and record.get("status") == "EVALUATED"
            and isinstance(evidence, Mapping)
            and evidence.get("schema")
            == "gold.1oz-regime-harmony-source-checkpoint.v1"
        ):
            return dict(record)
    raise ValueError("gold Stage-76 source has no executable checkpoint")


def gold_live_contract(selection: Mapping[str, object]) -> Contract:
    selected = load_gold_live_selection_from_mapping(selection)
    frozen = selected["contract"]
    assert isinstance(frozen, Mapping)
    contract = Contract(
        conId=int(frozen["con_id"]),
        symbol="1OZ",
        secType="FUT",
        lastTradeDateOrContractMonth=str(frozen["expiry"]),
        multiplier="1",
        exchange="COMEX",
        currency="USD",
        localSymbol=str(frozen["local_symbol"]),
    )
    setattr(contract, "minTick", float(frozen["min_tick"]))
    return contract


def gold_transport_order_ref(plan: Mapping[str, object]) -> str:
    transition_id = str(plan.get("transition_id") or "")
    if plan.get("schema") != GOLD_LIVE_PLAN_SCHEMA or len(transition_id) != 64:
        raise ValueError("gold transport plan has no transition identity")
    return f"{GOLD_LIVE_ORDER_REF_PREFIX}-{transition_id[:24]}"


async def gold_broker_snapshot(client) -> dict[str, object]:
    portfolio = await client.fetch_portfolio()
    account_id = str(client.account_id() or "").strip()
    account_type = str(client.account_text_value("TradingType-S") or "").upper()
    if not account_id or account_type != "STKCASH":
        raise ValueError("gold live transport requires one cash account")

    def account(tag: str, currency: str) -> float:
        value, actual, _updated = client.account_value(tag, currency=currency)
        if str(actual or "").upper() != currency:
            raise ValueError(f"fresh {tag} {currency} is unavailable")
        return _number(value, name=f"{tag} {currency}")

    positions = []
    for item in portfolio:
        contract = getattr(item, "contract", None)
        quantity = _number(
            getattr(item, "position", 0.0) or 0.0,
            name="broker position",
        )
        if abs(quantity) <= 1e-9:
            continue
        positions.append(
            {
                "symbol": str(getattr(contract, "symbol", "") or "").upper(),
                "local_symbol": str(
                    getattr(contract, "localSymbol", "") or ""
                ),
                "con_id": int(getattr(contract, "conId", 0) or 0),
                "sec_type": str(getattr(contract, "secType", "") or ""),
                "quantity": quantity,
            }
        )
    open_orders = []
    for trade in client.open_trades():
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        open_orders.append(
            {
                "symbol": str(getattr(contract, "symbol", "") or "").upper(),
                "con_id": int(getattr(contract, "conId", 0) or 0),
                "action": str(getattr(order, "action", "") or "").upper(),
                "quantity": _number(
                    getattr(order, "totalQuantity", 0.0) or 0.0,
                    name="broker order quantity",
                ),
                "order_ref": str(getattr(order, "orderRef", "") or ""),
                "status": str(getattr(status, "status", "") or ""),
            }
        )
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "account_type": "CASH",
        "base_currency": "AUD",
        "settled_cash_usd": account("CashBalance", "USD"),
        "equity_with_loan_aud": account("EquityWithLoanValue", "AUD"),
        "available_funds_aud": account("AvailableFunds", "AUD"),
        "excess_liquidity_aud": account("ExcessLiquidity", "AUD"),
        "initial_margin_aud": account("FullInitMarginReq", "AUD"),
        "maintenance_margin_aud": account("FullMaintMarginReq", "AUD"),
        "gross_position_value_aud": account("GrossPositionValue", "AUD"),
        "usd_to_aud": account("ExchangeRate", "USD"),
        "positions": positions,
        "open_orders": open_orders,
    }


def _latest_execution_by_ref(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
) -> dict[str, Mapping[str, object]]:
    output = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == GOLD_LIVE_EXECUTION_VERSION
            and isinstance(evidence, Mapping)
            and evidence.get("selection_id") == selection_id
            and str(evidence.get("order_ref") or "")
        ):
            output[str(evidence["order_ref"])] = record
    return output


def _checkpoint(
    ledger: LiveCalibrationLedger,
    *,
    selection_id: str,
    plan: Mapping[str, object],
    phase: str,
    observed_at: datetime,
    order_ref: str = "",
    preview: Mapping[str, object] | None = None,
    trade: object | None = None,
    submitted_orders: int = 0,
    risk_state: Mapping[str, object] | None = None,
    broker_state: Mapping[str, object] | None = None,
    quote: Mapping[str, object] | None = None,
    ladder_transition: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return ledger.checkpoint(
        evaluation_as_of=observed_at,
        strategy_id=GOLD_REGIME_HARMONY_VERSION,
        strategy_version=GOLD_LIVE_EXECUTION_VERSION,
        trading_date=observed_at.astimezone(timezone.utc).date().isoformat(),
        session="GOLD_GTH",
        status="EVALUATED",
        evidence={
            "schema": GOLD_LIVE_EXECUTION_SCHEMA,
            "selection_id": selection_id,
            "transition_id": plan["transition_id"],
            "source_checkpoint_id": plan.get("source_checkpoint_id"),
            "phase": phase,
            "order_ref": order_ref,
            "plan": dict(plan),
            "what_if_preview": dict(preview) if preview is not None else None,
            "broker_order": (
                broker_trade_snapshot(trade) if trade is not None else None
            ),
            "risk_state": dict(risk_state) if risk_state is not None else None,
            "broker_state": dict(broker_state) if broker_state is not None else None,
            "quote": dict(quote) if quote is not None else None,
            "ladder_transition": (
                dict(ladder_transition) if ladder_transition is not None else None
            ),
            "submitted_orders": int(submitted_orders),
            "order_authority": GOLD_LIVE_ORDER_AUTHORITY,
        },
        recorded_at=observed_at,
    )


def _commission_values(preview: object) -> list[float]:
    return [
        float(value)
        for value in (
            getattr(preview, "commission", None),
            getattr(preview, "min_commission", None),
            getattr(preview, "max_commission", None),
        )
        if value is not None and math.isfinite(float(value))
    ]


def _complete_one_contract_fill(snapshot: Mapping[str, object]) -> bool:
    try:
        return (
            terminal_broker_snapshot_complete(snapshot)
            and float(snapshot.get("filled") or 0.0) == 1.0
            and float(snapshot.get("remaining") or 0.0) == 0.0
        )
    except (TypeError, ValueError):
        return False


def _admit_gold_entry(
    plan: Mapping[str, object],
    *,
    preview: object,
    capital_plan: Mapping[str, object],
    selection: Mapping[str, object],
    selection_file_sha256: str,
    broker: Mapping[str, object],
) -> dict[str, object]:
    leg = plan.get("leg")
    if not isinstance(leg, Mapping):
        raise ValueError("gold capital admission requires one leg")
    held = plan.get("held_direction")
    reduction = held is not None
    if reduction:
        intent = "ROTATE_OUT" if plan.get("desired_after_close") else "EXIT"
        decision = admit_live_capital(
            capital_plan,
            intent=intent,
            account_id=str(broker["account_id"]),
            account_type="CASH",
            currency="USD",
            sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
            run_id=str(selection["selection_id"]),
            selection_file_sha256=selection_file_sha256,
            capital_kind="FUTURES_MARGIN",
            projected_capital_usd=0,
            cash_debit_usd=0,
            available_cash_usd=broker["settled_cash_usd"],
        )
    else:
        positions = broker["positions"]
        orders = broker["open_orders"]
        assert isinstance(positions, Sequence) and isinstance(orders, Sequence)
        decision = admit_live_capital(
            capital_plan,
            intent="ENTER",
            account_id=str(broker["account_id"]),
            account_type="CASH",
            currency="USD",
            sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
            run_id=str(selection["selection_id"]),
            selection_file_sha256=selection_file_sha256,
            capital_kind="FUTURES_MARGIN",
            projected_capital_usd=0,
            cash_debit_usd=0,
            available_cash_usd=broker["settled_cash_usd"],
            resource_state={
                "account_positions": [dict(row) for row in positions],
                "account_open_orders": [dict(row) for row in orders],
                "base_currency": "AUD",
                "quantity": 1,
                "initial_margin_change": preview.init_margin_change,
                "maintenance_margin_change": preview.maintenance_margin_change,
                "initial_margin_after": preview.init_margin_after,
                "maintenance_margin_after": preview.maintenance_margin_after,
                "equity_with_loan_after": preview.equity_with_loan_after,
                "available_funds_before": broker["available_funds_aud"],
                "unrelated_position_gross": broker["gross_position_value_aud"],
                "usd_to_base_rate": broker["usd_to_aud"],
            },
        )
    admitted = {**dict(plan), "capital_admission": decision}
    if decision["status"] == "ALLOW":
        return admitted
    return {
        **admitted,
        "status": "CAPITAL_HOLD",
        "blocked_leg": dict(leg),
        "leg": None,
        "reason": "capital_allocation_blocked",
    }


async def execute_gold_transport_plan(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    plan: Mapping[str, object],
    contract: Contract,
    ticker: object,
    observed_at: datetime,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    leg = plan.get("leg")
    if (
        plan.get("schema") != GOLD_LIVE_PLAN_SCHEMA
        or plan.get("selection_id") != selected["selection_id"]
        or plan.get("status") != "ACTIONABLE"
        or not isinstance(leg, Mapping)
        or plan.get("order_authority") != GOLD_LIVE_ORDER_AUTHORITY
    ):
        raise ValueError("only one admitted gold transport leg may execute")
    order_ref = gold_transport_order_ref(plan)
    records = tuple(ledger.records())
    latest = _latest_execution_by_ref(
        records, selection_id=str(selected["selection_id"])
    ).get(order_ref)
    prior = latest.get("evidence") if isinstance(latest, Mapping) else None
    matches = await client.reconcile_trades_for_order_ref(order_ref)
    if len(matches) > 1:
        raise ValueError("broker returned multiple gold orders for one transition")
    trade = matches[0] if matches else None
    if trade is not None and not isinstance(prior, Mapping):
        raise ValueError("gold broker order has no prepared local transition")
    if isinstance(prior, Mapping) and prior.get("phase") == "TERMINAL":
        snapshot = broker_trade_snapshot(trade) if trade is not None else {}
        if not _complete_one_contract_fill(snapshot):
            raise ValueError("gold terminal receipt disagrees with broker")
        return {
            "status": "TERMINAL",
            "order_ref": order_ref,
            "checkpoint_id": latest["checkpoint_id"],
            "submitted_orders": 0,
            "broker_order": snapshot,
        }
    if (
        trade is None
        and isinstance(prior, Mapping)
        and prior.get("phase") == "SUBMITTED"
    ):
        raise ValueError("submitted gold order disappeared from broker state")
    preview_payload = (
        dict(prior["what_if_preview"])
        if isinstance(prior, Mapping)
        and isinstance(prior.get("what_if_preview"), Mapping)
        else None
    )
    price_for_mode = execution_price_for_ticker(contract, ticker)
    action = str(leg["action"])
    initial_price = price_for_mode(str(leg["initial_mode"]), action)
    if initial_price is None or not math.isfinite(float(initial_price)):
        raise ValueError("gold transport has no executable initial price")
    submitted_orders = 0
    if trade is None:
        preview = await client.preview_limit_order(
            contract,
            action,
            1,
            float(initial_price),
            True,
            order_ref,
        )
        preview_payload = asdict(preview)
        commissions = _commission_values(preview)
        if (
            preview.status != "PreSubmitted"
            or not commissions
            or str(preview.commission_currency or "").upper() != "USD"
            or max(commissions) > GOLD_LIVE_MAX_COMMISSION_USD
            or str(preview.warning_text or "")
        ):
            raise ValueError("fresh gold what-if exceeds the selected boundary")
        if not isinstance(plan.get("capital_admission"), Mapping):
            raise ValueError("gold order lacks central capital admission")
        _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="PREPARED",
            observed_at=observed_at,
            order_ref=order_ref,
            preview=preview_payload,
        )
        trade = await client.place_limit_order(
            contract,
            action,
            1,
            float(initial_price),
            True,
            order_ref,
        )
        submitted_orders = 1
        _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            observed_at=observed_at,
            order_ref=order_ref,
            preview=preview_payload,
            trade=trade,
            submitted_orders=1,
        )

    order_id, perm_id = order_ids(trade)
    submitted_at = _utc(latest["recorded_at_utc"]) if latest is not None else observed_at
    elapsed = max(0.0, (_utc(observed_at) - submitted_at).total_seconds())

    def transition(payload: dict[str, object]) -> None:
        at = datetime.now(timezone.utc)
        _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            observed_at=at,
            order_ref=order_ref,
            preview=preview_payload,
            ladder_transition={
                "schema": "gold.execution-ladder-transition.v1",
                "observed_at_utc": at.isoformat(),
                "source_age_seconds": plan["source_age_seconds"],
                **payload,
            },
        )

    execution = LiveOrderExecution(
        client=client,
        state_by_order={},
        price_for_mode=price_for_mode,
        on_transition=transition,
    )
    if not bool(getattr(trade, "isDone", lambda: False)()):
        await execution.chase(
            trade,
            action,
            mode=str(leg["chase_mode"]),
            policy=EXECUTION_POLICY,
            elapsed_offset_sec=elapsed,
            require_fresh_top=True,
        )
    reconciled = await client.reconcile_order_state(
        order_id=order_id,
        perm_id=perm_id,
        force=True,
    )
    if isinstance(reconciled, Mapping) and reconciled.get("trade") is not None:
        trade = reconciled["trade"]
    broker_order = broker_trade_snapshot(trade)
    if broker_order.get("done") is True and not _complete_one_contract_fill(
        broker_order
    ):
        raise ValueError("gold order terminated without one complete fill")
    if not terminal_broker_snapshot_complete(broker_order):
        pending = _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            observed_at=datetime.now(timezone.utc),
            order_ref=order_ref,
            preview=preview_payload,
            trade=trade,
        )
        return {
            "status": "PENDING",
            "order_ref": order_ref,
            "checkpoint_id": pending["checkpoint_id"],
            "broker_order": broker_order,
            "submitted_orders": submitted_orders,
        }
    terminal = _checkpoint(
        ledger,
        selection_id=str(selected["selection_id"]),
        plan=plan,
        phase="TERMINAL",
        observed_at=datetime.now(timezone.utc),
        order_ref=order_ref,
        preview=preview_payload,
        trade=trade,
        submitted_orders=submitted_orders,
    )
    return {
        "status": "TERMINAL",
        "order_ref": order_ref,
        "checkpoint_id": terminal["checkpoint_id"],
        "broker_order": terminal["evidence"]["broker_order"],
        "submitted_orders": submitted_orders,
    }


async def advance_gold_live_transport(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    source_checkpoint: Mapping[str, object],
    capital_plan: Mapping[str, object],
    selection_file_sha256: str,
    observed_at: datetime,
    quote_wait_seconds: float = 3.0,
    _rotation_depth: int = 0,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    if quote_wait_seconds < 0 or _rotation_depth not in {0, 1}:
        raise ValueError("gold live observation inputs are invalid")
    broker = await gold_broker_snapshot(client)
    if broker["account_id"] != selected["broker_at_selection"]["account_id"]:
        raise ValueError("gold selected broker account changed")
    contract = gold_live_contract(selected)
    ticker = await client.ensure_ticker(contract, owner="gold-live-stage76")
    deadline = time.monotonic() + quote_wait_seconds
    health = {}
    quote = {}
    while True:
        captured = contract_from_ticker(
            getattr(ticker, "contract", None) or contract,
            ticker,
        )
        updated = getattr(ticker, "tbTopQuoteUpdatedMono", None)
        age = (
            max(0.0, time.monotonic() - float(updated))
            if updated is not None
            else None
        )
        health = quote_health(
            bid=captured.bid,
            ask=captured.ask,
            last=captured.last,
            close=captured.close,
            market_data_type=captured.market_data_type,
            age_sec=age,
            max_age_sec=10.0,
            require_live=True,
            require_nbbo=True,
            require_age=True,
        )
        quote = {
            "bid": captured.bid,
            "ask": captured.ask,
            "last": captured.last,
            "age_seconds": age,
            "market_data_type": captured.market_data_type,
            "health": health,
        }
        if health.get("eligible") is True or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)
    if health.get("eligible") is not True:
        raise ValueError("gold selected contract lacks fresh streaming NBBO")
    if int(getattr(contract, "conId", 0) or 0) != int(
        selected["contract"]["con_id"]
    ):
        raise ValueError("gold selected contract identity changed")

    records = tuple(ledger.records())
    latest = _latest_execution_by_ref(
        records, selection_id=str(selected["selection_id"])
    )
    pending = [
        record
        for record in latest.values()
        if isinstance(record.get("evidence"), Mapping)
        and record["evidence"].get("phase") in {"PREPARED", "SUBMITTED"}
    ]
    if len(pending) > 1:
        raise ValueError("gold selected run has multiple pending transitions")
    open_orders = broker["open_orders"]
    assert isinstance(open_orders, Sequence)
    if pending:
        evidence = pending[0]["evidence"]
        assert isinstance(evidence, Mapping)
        order_ref = str(evidence["order_ref"])
        if any(
            row.get("order_ref") != order_ref
            for row in open_orders
            if isinstance(row, Mapping)
            and (
                row.get("symbol") == "1OZ"
                or str(row.get("order_ref") or "").startswith(
                    f"{GOLD_LIVE_ORDER_REF_PREFIX}-"
                )
            )
        ):
            raise ValueError("unknown 1OZ order blocks reconciliation")
        pending_plan = evidence.get("plan")
        if not isinstance(pending_plan, Mapping):
            raise ValueError("pending gold transition has no durable plan")
        execution = await execute_gold_transport_plan(
            ledger,
            client=client,
            selection=selected,
            plan=pending_plan,
            contract=contract,
            ticker=ticker,
            observed_at=observed_at,
        )
        return {
            "status": "RECONCILED",
            "selection_id": selected["selection_id"],
            "execution": execution,
            "submitted_orders": execution["submitted_orders"],
        }

    gold_positions = [
        row
        for row in broker["positions"]
        if isinstance(row, Mapping) and row.get("symbol") == "1OZ"
    ]
    if len(gold_positions) > 1:
        raise ValueError("broker holds multiple 1OZ contract identities")
    broker_position = (
        float(gold_positions[0]["quantity"]) if gold_positions else 0.0
    )
    if gold_positions and int(gold_positions[0]["con_id"]) != int(
        selected["contract"]["con_id"]
    ):
        raise ValueError("broker holds an unselected 1OZ contract")
    liquidation = float(quote["bid"] if broker_position >= 0 else quote["ask"])
    risk_state = gold_transport_risk_state(
        selection=selected,
        records=records,
        observed_at=observed_at,
        liquidation_price=liquidation,
    )
    if abs(float(risk_state["position_from_fills"]) - broker_position) > 1e-9:
        raise ValueError("gold broker holding disagrees with selected-run fills")
    nonselected_orders = [
        row
        for row in open_orders
        if not str(row.get("order_ref") or "").startswith(
            f"{GOLD_LIVE_ORDER_REF_PREFIX}-"
        )
    ]
    if any(row.get("symbol") == "1OZ" for row in nonselected_orders):
        raise ValueError("unknown 1OZ order blocks selected-run authority")
    plan = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=source_checkpoint,
        broker_position=broker_position,
        open_orders=[],
        risk_state=risk_state,
        observed_at=observed_at,
    )
    preview = None
    if plan["status"] == "ACTIONABLE":
        leg = plan["leg"]
        assert isinstance(leg, Mapping)
        price_for_mode = execution_price_for_ticker(contract, ticker)
        price = price_for_mode(str(leg["initial_mode"]), str(leg["action"]))
        if price is None:
            raise ValueError("gold capital preview has no executable price")
        preview = await client.preview_limit_order(
            contract,
            str(leg["action"]),
            1,
            float(price),
            True,
            gold_transport_order_ref(plan),
        )
        commissions = _commission_values(preview)
        if (
            preview.status != "PreSubmitted"
            or not commissions
            or str(preview.commission_currency or "").upper() != "USD"
            or max(commissions) > GOLD_LIVE_MAX_COMMISSION_USD
            or str(preview.warning_text or "")
        ):
            raise ValueError("gold action preview exceeds selected boundaries")
        plan = _admit_gold_entry(
            plan,
            preview=preview,
            capital_plan=capital_plan,
            selection=selected,
            selection_file_sha256=selection_file_sha256,
            broker=broker,
        )
    state = _checkpoint(
        ledger,
        selection_id=str(selected["selection_id"]),
        plan=plan,
        phase="STATE",
        observed_at=observed_at,
        preview=asdict(preview) if preview is not None else None,
        risk_state=risk_state,
        broker_state=broker,
        quote=quote,
    )
    if plan["status"] != "ACTIONABLE":
        return {
            "status": plan["status"],
            "selection_id": selected["selection_id"],
            "checkpoint_id": state["checkpoint_id"],
            "plan": plan,
            "risk_state": risk_state,
            "submitted_orders": 0,
        }
    execution = await execute_gold_transport_plan(
        ledger,
        client=client,
        selection=selected,
        plan=plan,
        contract=contract,
        ticker=ticker,
        observed_at=observed_at,
    )
    result = {
        "status": "EXECUTED",
        "selection_id": selected["selection_id"],
        "checkpoint_id": state["checkpoint_id"],
        "plan": plan,
        "risk_state": risk_state,
        "execution": execution,
        "submitted_orders": execution["submitted_orders"],
    }
    close_for_reverse = bool(
        _rotation_depth == 0
        and plan.get("reason") == "close_before_reverse"
        and execution.get("status") == "TERMINAL"
        and isinstance(execution.get("broker_order"), Mapping)
        and float(execution["broker_order"].get("filled") or 0.0) == 1.0
    )
    if close_for_reverse:
        rotated_at = datetime.now(timezone.utc)
        if (
            rotated_at - _utc(source_checkpoint["recorded_at_utc"])
        ).total_seconds() <= GOLD_LIVE_ACTION_SOURCE_MAX_AGE_SECONDS:
            inverse = await advance_gold_live_transport(
                ledger,
                client=client,
                selection=selected,
                source_checkpoint=source_checkpoint,
                capital_plan=capital_plan,
                selection_file_sha256=selection_file_sha256,
                observed_at=rotated_at,
                quote_wait_seconds=quote_wait_seconds,
                _rotation_depth=1,
            )
            result["inverse_execution"] = inverse
            result["submitted_orders"] += int(inverse["submitted_orders"])
    return result
