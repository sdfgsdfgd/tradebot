"""Restart-safe broker execution for one selected XSP cash transport."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone

from ib_insync import Stock

from ..backtest.quotes import contract_from_ticker
from ..engines.execution import EXECUTION_POLICY, execution_price, quote_health
from ..engines.market import xsp_trading_date
from ..live.execution import LiveOrderExecution, order_ids
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .xsp_live_transport import (
    XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
    XSP_V2_TRANSPORT_EXECUTION_VERSION,
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V2_TRANSPORT_PLAN_SCHEMA,
    XSP_V2_TRANSPORT_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_EXECUTION_SCHEMA,
    XSP_V3_TRANSPORT_EXECUTION_VERSION,
    XSP_V3_TRANSPORT_PLAN_SCHEMA,
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    load_xsp_transport_selection_from_mapping,
    project_xsp_transport_plan,
    xsp_signal_utc,
    xsp_transport_contract,
)
from .xsp_live_transport_state import xsp_v2_broker_snapshot
from .xsp_live_transport_risk import (
    xsp_transport_cash_equity,
    xsp_transport_risk_state,
    xsp_v2_transport_risk_state as xsp_v2_transport_risk_state,
)
from .xsp_opening_edge_v2 import XSP_OPENING_EDGE_V2_VERSION
from .xsp_opening_edge_v3 import XSP_OPENING_EDGE_V3_VERSION


def xsp_transport_order_ref(plan: Mapping[str, object]) -> str:
    """Return the broker-visible identity for one content-addressed transition."""

    transition_id = str(plan.get("transition_id") or "")
    prefix = {
        XSP_V2_TRANSPORT_PLAN_SCHEMA: "XSPV2",
        XSP_V3_TRANSPORT_PLAN_SCHEMA: "XSPV3",
    }.get(plan.get("schema"))
    if prefix is None or len(transition_id) != 64:
        raise ValueError("transport plan has no valid transition identity")
    return f"{prefix}-{transition_id[:24]}"


def xsp_v2_transport_order_ref(plan: Mapping[str, object]) -> str:
    if plan.get("schema") != XSP_V2_TRANSPORT_PLAN_SCHEMA:
        raise ValueError("v2 order reference requires a v2 transport plan")
    return xsp_transport_order_ref(plan)


def _prior_execution(
    records: tuple[Mapping[str, object], ...],
    *,
    selection_id: str,
    order_ref: str,
    execution_version: str = XSP_V2_TRANSPORT_EXECUTION_VERSION,
) -> Mapping[str, object] | None:
    for record in reversed(records):
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == execution_version
            and isinstance(evidence, Mapping)
            and evidence.get("selection_id") == selection_id
            and evidence.get("order_ref") == order_ref
        ):
            return record
    return None


def _trade_snapshot(trade: object) -> dict[str, object]:
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    contract = getattr(trade, "contract", None)
    order_id, perm_id = order_ids(trade)
    symbol = str(getattr(contract, "symbol", "") or "")
    fills = []
    for fill in getattr(trade, "fills", ()) or ():
        execution = getattr(fill, "execution", None)
        commission = getattr(fill, "commissionReport", None)
        fill_time = getattr(fill, "time", None)
        fills.append(
            {
                "exec_id": str(getattr(execution, "execId", "") or ""),
                "time_utc": (
                    fill_time.astimezone(timezone.utc).isoformat()
                    if isinstance(fill_time, datetime) and fill_time.tzinfo is not None
                    else None
                ),
                "side": str(getattr(execution, "side", "") or "").upper(),
                "symbol": symbol,
                "shares": getattr(execution, "shares", None),
                "price": getattr(execution, "price", None),
                "commission": getattr(commission, "commission", None),
                "commission_currency": getattr(commission, "currency", None),
            }
        )
    return {
        "order_id": order_id,
        "perm_id": perm_id,
        "order_ref": str(getattr(order, "orderRef", "") or ""),
        "symbol": symbol,
        "con_id": int(getattr(contract, "conId", 0) or 0),
        "action": str(getattr(order, "action", "") or "").upper(),
        "quantity": getattr(order, "totalQuantity", None),
        "limit_price": getattr(order, "lmtPrice", None),
        "status": str(getattr(status, "status", "") or ""),
        "filled": getattr(status, "filled", None),
        "remaining": getattr(status, "remaining", None),
        "average_fill_price": getattr(status, "avgFillPrice", None),
        "done": bool(getattr(trade, "isDone", lambda: False)()),
        "fills": fills,
    }


def _terminal_snapshot_complete(snapshot: Mapping[str, object]) -> bool:
    """Require terminal broker state plus exact fill economics."""

    if snapshot.get("done") is not True:
        return False
    try:
        filled = float(snapshot.get("filled") or 0.0)
        fills = snapshot["fills"]
        fill_total = sum(float(fill["shares"]) for fill in fills)
        fills_complete = all(
            str(fill.get("exec_id") or "")
            and str(fill.get("time_utc") or "")
            and float(fill["price"]) > 0
            and float(fill["commission"]) >= 0
            and str(fill.get("commission_currency") or "").upper() == "USD"
            for fill in fills
        )
    except (KeyError, TypeError, ValueError):
        return False
    return filled >= 0 and abs(fill_total - filled) <= 1e-9 and fills_complete


def _checkpoint(
    ledger: LiveCalibrationLedger,
    *,
    selection_id: str,
    plan: Mapping[str, object],
    phase: str,
    order_ref: str,
    observed_at: datetime,
    preview: Mapping[str, object] | None,
    trade: object | None,
    submitted_orders: int,
    risk_state: Mapping[str, object] | None = None,
    broker_state: Mapping[str, object] | None = None,
    ladder_transition: Mapping[str, object] | None = None,
    selected_cash_equity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    runtime = {
        XSP_V2_TRANSPORT_PLAN_SCHEMA: (
            XSP_OPENING_EDGE_V2_VERSION,
            XSP_V2_TRANSPORT_EXECUTION_VERSION,
            XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
        ),
        XSP_V3_TRANSPORT_PLAN_SCHEMA: (
            XSP_OPENING_EDGE_V3_VERSION,
            XSP_V3_TRANSPORT_EXECUTION_VERSION,
            XSP_V3_TRANSPORT_EXECUTION_SCHEMA,
        ),
    }.get(plan.get("schema"))
    if runtime is None:
        raise ValueError("unsupported selected transport plan")
    strategy_id, execution_version, execution_schema = runtime
    trading_day = xsp_trading_date(observed_at)
    return ledger.checkpoint(
        evaluation_as_of=observed_at,
        strategy_id=strategy_id,
        strategy_version=execution_version,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=str(plan.get("source_session") or "RTH"),
        status="EVALUATED",
        evidence={
            "schema": execution_schema,
            "selection_id": selection_id,
            "transition_id": plan["transition_id"],
            "source_checkpoint_id": plan.get("source_checkpoint_id"),
            "phase": phase,
            "order_ref": order_ref,
            "plan": dict(plan),
            "what_if_preview": dict(preview) if preview is not None else None,
            "broker_order": _trade_snapshot(trade) if trade is not None else None,
            "risk_state": dict(risk_state) if risk_state is not None else None,
            "broker_state": (dict(broker_state) if broker_state is not None else None),
            "ladder_transition": (
                dict(ladder_transition) if ladder_transition is not None else None
            ),
            "submitted_orders": int(submitted_orders),
            "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
            **(
                {"selected_cash_equity": dict(selected_cash_equity)}
                if selected_cash_equity is not None
                else {}
            ),
        },
        recorded_at=observed_at,
    )


def _price_for_mode(contract: object, ticker: object):
    def price(
        mode: str,
        action: str,
        *,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        ticker: object | None = None,
        elapsed_sec: float = 0.0,
        quote_stale: bool = False,
        open_shock: bool = False,
        no_progress_reprices: int = 0,
        arrival_ref: float | None = None,
        delay_recoveries: int = 0,
        delay_anchor_price: float | None = None,
        delay_sweep_anchor_price: float | None = None,
        delay_locked_price_dir: float | None = None,
    ) -> float | None:
        active_ticker = ticker or ticker_ref
        return execution_price(
            contract,
            active_ticker,
            mode,
            action,
            bid=bid if bid is not None else getattr(active_ticker, "bid", None),
            ask=ask if ask is not None else getattr(active_ticker, "ask", None),
            last=last if last is not None else getattr(active_ticker, "last", None),
            fallback_price=getattr(active_ticker, "close", None),
            custom_price=None,
            policy=EXECUTION_POLICY,
            elapsed_sec=elapsed_sec,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=no_progress_reprices,
            arrival_ref=arrival_ref,
            delay_recoveries=delay_recoveries,
            delay_anchor_price=delay_anchor_price,
            delay_sweep_anchor_price=delay_sweep_anchor_price,
            delay_locked_price_dir=delay_locked_price_dir,
        )

    ticker_ref = ticker
    return price


async def execute_xsp_transport_plan(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    plan: Mapping[str, object],
    contract: object,
    ticker: object,
    observed_at: datetime,
) -> dict[str, object]:
    """Preview, submit, chase, and reconcile at most one selected transport leg."""

    selected = load_xsp_transport_selection_from_mapping(selection)
    transport = xsp_transport_contract(selected)
    symbols = tuple(transport["symbols"])
    execution_version = str(transport["execution_version"])
    if observed_at.tzinfo is None:
        raise ValueError("transport execution timestamp must be aware")
    if (
        plan.get("schema") != transport["plan_schema"]
        or plan.get("selection_id") != selected["selection_id"]
        or plan.get("order_authority") != XSP_V2_TRANSPORT_ORDER_AUTHORITY
        or plan.get("status") != "ACTIONABLE"
        or not isinstance(plan.get("leg"), Mapping)
    ):
        raise ValueError("only one selected actionable transport plan may execute")
    leg = plan["leg"]
    assert isinstance(leg, Mapping)
    signal_context = plan.get("signal_context")
    if (
        not isinstance(signal_context, Mapping)
        or signal_context.get("schema") != "xsp.execution-signal-context.v1"
        or not signal_context.get("decision_trace_fingerprint")
        or not isinstance(signal_context.get("directional_impulse"), Mapping)
        or not isinstance(signal_context.get("market_state"), Mapping)
    ):
        raise ValueError("actionable transport has no causal signal context")
    try:
        signal_at_utc = xsp_signal_utc(signal_context["signal_bar_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport signal timestamp is invalid") from exc
    symbol = str(leg.get("symbol") or "")
    action = str(leg.get("action") or "").upper()
    quantity = leg.get("quantity")
    outside_rth = leg.get("outside_rth")
    if (
        symbol not in symbols
        or action not in {"BUY", "SELL"}
        or not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity <= 0
        or not isinstance(outside_rth, bool)
        or (outside_rth and action != "SELL")
        or str(getattr(contract, "symbol", "") or "") != symbol
        or int(getattr(contract, "conId", 0) or 0) <= 0
    ):
        raise ValueError("transport leg and qualified contract do not agree")

    order_ref = xsp_transport_order_ref(plan)
    records = tuple(ledger.records())
    prior = _prior_execution(
        records,
        selection_id=str(selected["selection_id"]),
        order_ref=order_ref,
        execution_version=execution_version,
    )
    matches = await client.reconcile_trades_for_order_ref(order_ref)
    if len(matches) > 1:
        raise ValueError("broker returned multiple orders for one transition")
    trade = matches[0] if matches else None
    if trade is not None and prior is None:
        raise ValueError("broker order has no prepared local transition")
    if (
        prior is not None
        and isinstance(prior.get("evidence"), Mapping)
        and prior["evidence"].get("phase") == "TERMINAL"
    ):
        if trade is None or not bool(getattr(trade, "isDone", lambda: False)()):
            raise ValueError("terminal receipt disagrees with reconciled broker")
        return {
            "status": "TERMINAL",
            "order_ref": order_ref,
            "checkpoint_id": prior["checkpoint_id"],
            "submitted_orders": 0,
        }
    prior_evidence = (
        prior.get("evidence")
        if prior is not None and isinstance(prior.get("evidence"), Mapping)
        else None
    )
    preview_payload = (
        dict(prior_evidence["what_if_preview"])
        if isinstance(prior_evidence, Mapping)
        and isinstance(prior_evidence.get("what_if_preview"), Mapping)
        else None
    )
    submitted_orders = 0
    price_for_mode = _price_for_mode(contract, ticker)
    initial_mode = str(leg.get("initial_mode") or "")
    initial_price = price_for_mode(initial_mode, action)
    if initial_price is None or not math.isfinite(float(initial_price)):
        raise ValueError("selected transport has no executable initial price")

    if trade is None:
        if (
            prior is not None
            and isinstance(prior.get("evidence"), Mapping)
            and prior["evidence"].get("phase") == "SUBMITTED"
        ):
            raise ValueError("submitted order is absent from reconciled broker state")
        preview = await client.preview_limit_order(
            contract,
            action,
            quantity,
            float(initial_price),
            outside_rth,
            order_ref,
        )
        preview_payload = asdict(preview)
        commission_values = [
            value
            for value in (
                preview.commission,
                preview.min_commission,
                preview.max_commission,
            )
            if value is not None and math.isfinite(float(value))
        ]
        nominee = selected["nominee"]
        assert isinstance(nominee, Mapping)
        commission_limits = nominee.get("commission_limits_usd")
        if (
            not commission_values
            or str(preview.commission_currency or "").upper() != "USD"
            or not isinstance(commission_limits, Mapping)
            or max(float(value) for value in commission_values)
            > float(commission_limits[symbol]) + 0.01
        ):
            raise ValueError("fresh broker preview exceeds selected commission")
        if prior is None:
            _checkpoint(
                ledger,
                selection_id=str(selected["selection_id"]),
                plan=plan,
                phase="PREPARED",
                order_ref=order_ref,
                observed_at=observed_at,
                preview=preview_payload,
                trade=None,
                submitted_orders=0,
            )
        trade = await client.place_limit_order(
            contract,
            action,
            quantity,
            float(initial_price),
            outside_rth,
            order_ref,
        )
        submitted_orders = 1
        _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            order_ref=order_ref,
            observed_at=observed_at,
            preview=preview_payload,
            trade=trade,
            submitted_orders=1,
        )

    order_id, perm_id = order_ids(trade)
    submitted_at = observed_at
    first_submitted_at = next(
        (
            datetime.fromisoformat(
                str(record["recorded_at_utc"]).replace("Z", "+00:00")
            )
            for record in records
            if record.get("kind") == "checkpoint"
            and record.get("strategy_version")
            == execution_version
            and isinstance(record.get("evidence"), Mapping)
            and record["evidence"].get("selection_id") == selected["selection_id"]
            and record["evidence"].get("order_ref") == order_ref
            and record["evidence"].get("phase") == "SUBMITTED"
        ),
        None,
    )
    if first_submitted_at is not None:
        submitted_at = first_submitted_at
    elif prior is not None:
        submitted_at = datetime.fromisoformat(
            str(prior["recorded_at_utc"]).replace("Z", "+00:00")
        )
    elapsed = max(
        0.0,
        (
            observed_at.astimezone(timezone.utc) - submitted_at.astimezone(timezone.utc)
        ).total_seconds(),
    )
    def record_ladder_transition(payload: dict[str, object]) -> None:
        transition_at = datetime.now(timezone.utc)
        _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            order_ref=order_ref,
            observed_at=transition_at,
            preview=preview_payload,
            trade=None,
            submitted_orders=0,
            ladder_transition={
                "schema": "xsp.execution-ladder-transition.v1",
                "observed_at_utc": transition_at.isoformat(),
                "signal_age_seconds": max(
                    0.0, (transition_at - signal_at_utc).total_seconds()
                ),
                "signal_context_fingerprint": calibration_fingerprint(signal_context),
                **payload,
            },
        )

    execution = LiveOrderExecution(
        client=client,
        state_by_order={},
        price_for_mode=price_for_mode,
        on_transition=record_ladder_transition,
    )
    if not bool(getattr(trade, "isDone", lambda: False)()):
        await execution.chase(
            trade,
            action,
            mode=str(leg.get("chase_mode") or ""),
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
    broker_order = _trade_snapshot(trade)
    if not _terminal_snapshot_complete(broker_order):
        pending = _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            phase="SUBMITTED",
            order_ref=order_ref,
            observed_at=datetime.now(timezone.utc),
            preview=preview_payload,
            trade=trade,
            submitted_orders=submitted_orders,
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
        order_ref=order_ref,
        observed_at=datetime.now(timezone.utc),
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


async def execute_xsp_v2_transport_plan(
    ledger: LiveCalibrationLedger,
    **kwargs,
) -> dict[str, object]:
    selection = kwargs.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema") != XSP_V2_TRANSPORT_SELECTION_SCHEMA
    ):
        raise ValueError("v2 execution requires a v2 selected transport")
    return await execute_xsp_transport_plan(ledger, **kwargs)


async def advance_xsp_live_transport(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    source_receipt: Mapping[str, object],
    observed_at: datetime,
    quote_wait_seconds: float = 3.0,
) -> dict[str, object]:
    """Reconcile one selected sleeve, then execute at most one exact RTH leg."""

    selected = load_xsp_transport_selection_from_mapping(selection)
    transport = xsp_transport_contract(selected)
    symbols = tuple(transport["symbols"])
    execution_version = str(transport["execution_version"])
    order_ref_prefix = f"{transport['order_ref_prefix']}-"
    if observed_at.tzinfo is None or quote_wait_seconds < 0:
        raise ValueError("live transport observation inputs are invalid")
    broker_snapshot = await xsp_v2_broker_snapshot(client, symbols=symbols)
    account_id = str(broker_snapshot["account_id"])
    selected_broker = selected["broker_at_selection"]
    assert isinstance(selected_broker, Mapping)
    if account_id != selected_broker["account_id"]:
        raise ValueError("selected transport broker account identity changed")
    broker_cash = float(broker_snapshot["settled_cash_usd"])
    positions = dict(broker_snapshot["positions"])
    unrelated_positions = list(broker_snapshot["unrelated_positions"])
    open_rows = list(broker_snapshot["open_orders"])

    records = tuple(ledger.records())
    latest_execution_by_ref: dict[str, Mapping[str, object]] = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == execution_version
            and isinstance(evidence, Mapping)
            and evidence.get("selection_id") == selected["selection_id"]
            and evidence.get("phase") in {"PREPARED", "SUBMITTED", "TERMINAL"}
            and str(evidence.get("order_ref") or "")
        ):
            latest_execution_by_ref[str(evidence["order_ref"])] = evidence
    pending = [
        evidence
        for evidence in latest_execution_by_ref.values()
        if evidence.get("phase") in {"PREPARED", "SUBMITTED"}
    ]
    if len(pending) > 1:
        raise ValueError("selected transport has multiple pending transitions")

    nominee = selected["nominee"]
    assert isinstance(nominee, Mapping)
    contract_ids = nominee.get("contract_ids")
    if not isinstance(contract_ids, Mapping):
        raise ValueError("selected transport contract identity is missing")
    qualified = await client.qualify_proxy_contracts(
        *(Stock(symbol, "SMART", "USD") for symbol in symbols)
    )
    contracts = {
        str(getattr(contract, "symbol", "") or "").upper(): contract
        for contract in qualified
        if str(getattr(contract, "symbol", "") or "").upper() in symbols
    }
    if set(contracts) != set(symbols) or any(
        int(getattr(contracts[symbol], "conId", 0) or 0) != int(contract_ids[symbol])
        for symbol in symbols
    ):
        raise ValueError("selected transport contract identity changed")
    generic_ticks = transport["generic_ticks"]
    assert isinstance(generic_ticks, Mapping)
    tickers = {
        symbol: await client.ensure_ticker(
            contracts[symbol],
            owner=str(transport["ticker_owner"]),
            generic_ticks=str(generic_ticks[symbol]) if symbol in generic_ticks else "",
        )
        for symbol in symbols
    }

    deadline = time.monotonic() + quote_wait_seconds
    quotes: dict[str, dict[str, object]] = {}
    quote_health_by_symbol: dict[str, dict[str, object]] = {}
    while True:
        quotes = {}
        quote_health_by_symbol = {}
        for symbol, ticker in tickers.items():
            routed = getattr(ticker, "contract", None) or contracts[symbol]
            captured = contract_from_ticker(routed, ticker)
            updated_mono = getattr(ticker, "tbTopQuoteUpdatedMono", None)
            try:
                age = (
                    max(0.0, time.monotonic() - float(updated_mono))
                    if updated_mono is not None
                    else None
                )
            except (TypeError, ValueError):
                age = None
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
            quote_health_by_symbol[symbol] = health
            quotes[symbol] = {
                "bid": captured.bid,
                "ask": captured.ask,
                "age_seconds": age,
                "market_data_type": captured.market_data_type,
            }
        if (
            all(
                health.get("eligible") is True
                for health in quote_health_by_symbol.values()
            )
            or time.monotonic() >= deadline
        ):
            break
        await asyncio.sleep(0.1)

    nav_symbol = transport["nav_symbol"]
    nav_value = nav_at = None
    if isinstance(nav_symbol, str):
        nav_value, nav_at = client.generic_tick_value(tickers[nav_symbol], 96)
    nav_age = (
        max(
            0.0,
            (
                datetime.now(timezone.utc) - nav_at.astimezone(timezone.utc)
            ).total_seconds(),
        )
        if nav_at is not None
        else None
    )
    spyu_nav = (
        {"value": nav_value, "age_seconds": nav_age}
        if nav_value is not None and nav_age is not None
        else None
    )

    if pending:
        pending_evidence = pending[0]
        pending_ref = str(pending_evidence["order_ref"])
        if any(
            row["order_ref"] != pending_ref
            and (
                row["symbol"] in symbols
                or str(row["order_ref"]).startswith(order_ref_prefix)
            )
            for row in open_rows
        ):
            raise ValueError("unknown cash-pair order blocks reconciliation")
        pending_plan = pending_evidence.get("plan")
        if not isinstance(pending_plan, Mapping):
            raise ValueError("pending selected transition has no durable plan")
        leg = pending_plan.get("leg")
        symbol = str(leg.get("symbol") or "") if isinstance(leg, Mapping) else ""
        if (
            symbol not in contracts
            or quote_health_by_symbol[symbol].get("eligible") is not True
        ):
            raise ValueError("pending selected transition contract is invalid")
        execution = await execute_xsp_transport_plan(
            ledger,
            client=client,
            selection=selected,
            plan=pending_plan,
            contract=contracts[symbol],
            ticker=tickers[symbol],
            observed_at=observed_at,
        )
        return {
            "status": "RECONCILED",
            "selection_id": selected["selection_id"],
            "execution": execution,
            "submitted_orders": execution["submitted_orders"],
        }

    held_symbols = [symbol for symbol, quantity in positions.items() if quantity > 1e-9]
    if any(
        quote_health_by_symbol[symbol].get("eligible") is not True
        for symbol in held_symbols
    ):
        raise ValueError("held selected sleeve has no fresh liquidation NBBO")
    risk_state = xsp_transport_risk_state(
        selection=selected,
        records=records,
        observed_at=observed_at,
        liquidation_bids={
            symbol: float(quote["bid"])
            for symbol, quote in quotes.items()
            if quote["bid"] is not None
        },
    )
    fill_holdings = risk_state["holdings_from_fills"]
    assert isinstance(fill_holdings, Mapping)
    if any(
        abs(float(fill_holdings[symbol]) - positions[symbol]) > 1e-9
        for symbol in symbols
    ):
        raise ValueError("broker holdings disagree with selected-run fill ledger")
    selected_cash_equity = (
        xsp_transport_cash_equity(
            selection=selected,
            risk_state=risk_state,
            reconciled=True,
        )
        if selected["schema"] == XSP_V3_TRANSPORT_SELECTION_SCHEMA
        else None
    )
    plan = project_xsp_transport_plan(
        selection=selected,
        source_receipt=source_receipt,
        observed_at=observed_at,
        positions=positions,
        open_orders=open_rows,
        settled_cash_usd=min(
            broker_cash,
            float(risk_state["settled_cash_usd"]),
        ),
        quotes=quotes,
        spyu_nav=spyu_nav,
        session_net_usd=float(risk_state["session_net_usd"]),
        drawdown_usd=float(risk_state["drawdown_usd"]),
    )
    broker_state = {
        "account_id": account_id,
        "account_type": "STKCASH",
        "cash_balance_usd": broker_cash,
        "positions": positions,
        "unrelated_positions": unrelated_positions,
        "open_orders": open_rows,
        "quotes": quotes,
        "quote_health": quote_health_by_symbol,
        "spyu_nav": spyu_nav,
    }
    state_checkpoint = _checkpoint(
        ledger,
        selection_id=str(selected["selection_id"]),
        plan=plan,
        phase="STATE",
        order_ref="",
        observed_at=observed_at,
        preview=None,
        trade=None,
        submitted_orders=0,
        risk_state=risk_state,
        broker_state=broker_state,
        selected_cash_equity=selected_cash_equity,
    )
    if plan["status"] != "ACTIONABLE":
        return {
            "status": plan["status"],
            "selection_id": selected["selection_id"],
            "checkpoint_id": state_checkpoint["checkpoint_id"],
            "plan": plan,
            "risk_state": risk_state,
            "submitted_orders": 0,
        }
    leg = plan["leg"]
    assert isinstance(leg, Mapping)
    symbol = str(leg["symbol"])
    execution = await execute_xsp_transport_plan(
        ledger,
        client=client,
        selection=selected,
        plan=plan,
        contract=contracts[symbol],
        ticker=tickers[symbol],
        observed_at=observed_at,
    )
    return {
        "status": "EXECUTED",
        "selection_id": selected["selection_id"],
        "checkpoint_id": state_checkpoint["checkpoint_id"],
        "plan": plan,
        "risk_state": risk_state,
        "execution": execution,
        "submitted_orders": execution["submitted_orders"],
    }


async def advance_xsp_v2_live_transport(
    ledger: LiveCalibrationLedger,
    **kwargs,
) -> dict[str, object]:
    selection = kwargs.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema") != XSP_V2_TRANSPORT_SELECTION_SCHEMA
    ):
        raise ValueError("v2 runtime requires a v2 selected transport")
    return await advance_xsp_live_transport(ledger, **kwargs)
