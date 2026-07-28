"""Restart-safe broker execution for one selected XSP v2 cash transport."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from ib_insync import Stock

from ..backtest.quotes import contract_from_ticker
from ..engines.execution import EXECUTION_POLICY, execution_price, quote_health
from ..engines.market import is_trading_day, xsp_trading_date
from ..live.execution import LiveOrderExecution, order_ids
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .xsp_live_transport import (
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V2_TRANSPORT_PLAN_SCHEMA,
    load_xsp_v2_transport_selection_from_mapping,
    project_xsp_v2_transport_plan,
    xsp_v2_broker_snapshot,
)
from .xsp_opening_edge_v2 import XSP_OPENING_EDGE_V2_VERSION


XSP_V2_TRANSPORT_EXECUTION_VERSION = "xsp.opening-edge-v2-spyu-spxu-live-execution.v1"
XSP_V2_TRANSPORT_EXECUTION_SCHEMA = (
    "xsp.opening-edge-v2-spyu-spxu-execution-checkpoint.v1"
)


def xsp_v2_transport_order_ref(plan: Mapping[str, object]) -> str:
    """Return the broker-visible identity for one content-addressed transition."""

    transition_id = str(plan.get("transition_id") or "")
    if plan.get("schema") != XSP_V2_TRANSPORT_PLAN_SCHEMA or len(transition_id) != 64:
        raise ValueError("transport plan has no valid transition identity")
    return f"XSPV2-{transition_id[:24]}"


def _prior_execution(
    records: tuple[Mapping[str, object], ...],
    *,
    selection_id: str,
    order_ref: str,
) -> Mapping[str, object] | None:
    for record in reversed(records):
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == XSP_V2_TRANSPORT_EXECUTION_VERSION
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


def xsp_v2_transport_risk_state(
    *,
    selection: Mapping[str, object],
    records: tuple[Mapping[str, object], ...],
    observed_at: datetime,
    liquidation_bids: Mapping[str, float],
) -> dict[str, object]:
    """Reconstruct selected-sleeve equity from immutable broker fills."""

    selected = load_xsp_v2_transport_selection_from_mapping(selection)
    if observed_at.tzinfo is None:
        raise ValueError("risk observation timestamp must be aware")
    fills_by_id: dict[str, dict[str, object]] = {}
    prior_risks = []
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != XSP_V2_TRANSPORT_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selected["selection_id"]
        ):
            continue
        prior_risk = evidence.get("risk_state")
        if isinstance(prior_risk, Mapping):
            prior_risks.append((record, prior_risk))
        order = evidence.get("broker_order")
        if evidence.get("phase") != "TERMINAL" or not isinstance(order, Mapping):
            continue
        order_fills = order.get("fills")
        if not isinstance(order_fills, list):
            raise ValueError("terminal broker order has no fill ledger")
        try:
            filled_quantity = float(order.get("filled") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("terminal filled quantity is invalid") from exc
        if filled_quantity > 0 and not order_fills:
            raise ValueError("filled broker order has no execution details")
        for raw_fill in order_fills:
            if not isinstance(raw_fill, Mapping):
                raise ValueError("broker fill must be an object")
            fill = dict(raw_fill)
            exec_id = str(fill.get("exec_id") or "").strip()
            if not exec_id:
                raise ValueError("broker fill has no execution identity")
            prior = fills_by_id.get(exec_id)
            if prior is not None and prior != fill:
                raise ValueError("broker execution identity changed")
            fills_by_id[exec_id] = fill

    ordered = sorted(
        fills_by_id.values(),
        key=lambda fill: (
            str(fill.get("time_utc") or ""),
            str(fill.get("exec_id") or ""),
        ),
    )
    holdings = {"SPYU": 0.0, "SPXU": 0.0}
    open_cost = {"SPYU": 0.0, "SPXU": 0.0}
    realized = 0.0
    risk_identity = selected["risk"]
    assert isinstance(risk_identity, Mapping)
    selected_broker = selected["broker_at_selection"]
    assert isinstance(selected_broker, Mapping)
    settled_cash = float(selected_broker["settled_cash_usd"])
    pending_settlements: list[tuple[object, float]] = []
    canonical_fills = []

    def mature(trading_day) -> None:
        nonlocal settled_cash, pending_settlements
        settled_cash += sum(
            proceeds
            for settlement_day, proceeds in pending_settlements
            if settlement_day <= trading_day
        )
        pending_settlements = [
            (settlement_day, proceeds)
            for settlement_day, proceeds in pending_settlements
            if settlement_day > trading_day
        ]

    for fill in ordered:
        symbol = str(fill.get("symbol") or "")
        side = str(fill.get("side") or "").upper()
        try:
            shares = float(fill.get("shares"))
            price = float(fill.get("price"))
            commission = float(fill.get("commission"))
            fill_time = datetime.fromisoformat(
                str(fill.get("time_utc") or "").replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("broker fill economics are incomplete") from exc
        if (
            symbol not in {"SPYU", "SPXU"}
            or side not in {"BOT", "BUY", "SLD", "SELL"}
            or not all(math.isfinite(value) for value in (shares, price, commission))
            or shares <= 0
            or price <= 0
            or commission < 0
            or str(fill.get("commission_currency") or "").upper() != "USD"
            or fill_time.tzinfo is None
            or fill_time.astimezone(timezone.utc) > observed_at.astimezone(timezone.utc)
        ):
            raise ValueError("broker fill is not valid selected-sleeve evidence")
        action = "BUY" if side in {"BOT", "BUY"} else "SELL"
        fill_trading_day = xsp_trading_date(fill_time)
        if fill_trading_day is None or not is_trading_day(fill_trading_day):
            raise ValueError("selected-sleeve fill is outside its RTH trading day")
        mature(fill_trading_day)
        cash_amount = shares * price + commission
        settlement_day = None
        if action == "BUY":
            if any(
                quantity > 1e-9
                for other, quantity in holdings.items()
                if other != symbol
            ):
                raise ValueError("cash-pair fills overlap both symbols")
            if cash_amount > settled_cash + 1e-9:
                raise ValueError("selected-sleeve buy exceeds settled USD reserve")
            settled_cash -= cash_amount
            holdings[symbol] += shares
            open_cost[symbol] += cash_amount
        else:
            quantity = holdings[symbol]
            if shares > quantity + 1e-9:
                raise ValueError("cash-pair sell exceeds selected-run holdings")
            removed_cost = open_cost[symbol] * shares / quantity
            proceeds = shares * price - commission
            realized += proceeds - removed_cost
            holdings[symbol] -= shares
            open_cost[symbol] -= removed_cost
            settlement_day = fill_trading_day + timedelta(days=1)
            while not is_trading_day(settlement_day):
                settlement_day += timedelta(days=1)
            pending_settlements.append((settlement_day, proceeds))
        canonical_fills.append(
            {
                "exec_id": fill["exec_id"],
                "time_utc": fill_time.astimezone(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "price": price,
                "commission_usd": commission,
                "settled_cash_after_usd": settled_cash,
                "settlement_date": (
                    settlement_day.isoformat() if settlement_day is not None else None
                ),
            }
        )

    trading_day = xsp_trading_date(observed_at)
    if trading_day is None or not is_trading_day(trading_day):
        raise ValueError("risk observation is outside an XSP trading day")
    mature(trading_day)
    pending_by_day: dict[str, float] = {}
    for settlement_day, proceeds in pending_settlements:
        key = settlement_day.isoformat()
        pending_by_day[key] = pending_by_day.get(key, 0.0) + proceeds
    pending_receipt = [
        {"settlement_date": settlement_day, "proceeds_usd": proceeds}
        for settlement_day, proceeds in sorted(pending_by_day.items())
    ]
    open_mark = 0.0
    for symbol, quantity in holdings.items():
        if quantity <= 1e-9:
            continue
        try:
            bid = float(liquidation_bids[symbol])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("open selected sleeve has no liquidation bid") from exc
        if not math.isfinite(bid) or bid <= 0:
            raise ValueError("selected-sleeve liquidation bid is invalid")
        open_mark += quantity * bid - open_cost[symbol]
    run_net = realized + open_mark
    prior_peak = max(
        (float(risk.get("peak_run_net_usd") or 0.0) for _record, risk in prior_risks),
        default=0.0,
    )
    peak = max(0.0, prior_peak, run_net)
    trading_date = trading_day.isoformat() if trading_day else None
    same_day = [
        risk
        for record, risk in prior_risks
        if record.get("trading_date") == trading_date
    ]
    session_baseline = (
        float(same_day[0].get("session_baseline_run_net_usd") or 0.0)
        if same_day
        else run_net
    )
    return {
        "valid": True,
        "as_of_utc": observed_at.astimezone(timezone.utc).isoformat(),
        "trading_date": trading_date,
        "run_realized_net_usd": realized,
        "open_mark_net_usd": open_mark,
        "run_net_usd": run_net,
        "peak_run_net_usd": peak,
        "drawdown_usd": peak - run_net,
        "session_baseline_run_net_usd": session_baseline,
        "session_net_usd": run_net - session_baseline,
        "holdings_from_fills": holdings,
        "open_cost_usd": open_cost,
        "starting_settled_cash_usd": float(selected_broker["settled_cash_usd"]),
        "settled_cash_usd": settled_cash,
        "pending_settlement_usd": sum(row["proceeds_usd"] for row in pending_receipt),
        "pending_settlements": pending_receipt,
        "fill_count": len(canonical_fills),
        "fill_ledger_fingerprint": calibration_fingerprint(canonical_fills),
    }


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
) -> dict[str, object]:
    trading_day = xsp_trading_date(observed_at)
    return ledger.checkpoint(
        evaluation_as_of=observed_at,
        strategy_id=XSP_OPENING_EDGE_V2_VERSION,
        strategy_version=XSP_V2_TRANSPORT_EXECUTION_VERSION,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=str(plan.get("source_session") or "RTH"),
        status="EVALUATED",
        evidence={
            "schema": XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
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


async def execute_xsp_v2_transport_plan(
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

    selected = load_xsp_v2_transport_selection_from_mapping(selection)
    if observed_at.tzinfo is None:
        raise ValueError("transport execution timestamp must be aware")
    if (
        plan.get("schema") != XSP_V2_TRANSPORT_PLAN_SCHEMA
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
        signal_at = datetime.fromisoformat(
            str(signal_context["signal_bar_ts"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport signal timestamp is invalid") from exc
    if signal_at.tzinfo is None:
        raise ValueError("transport signal timestamp must be aware")
    signal_at_utc = signal_at.astimezone(timezone.utc)
    symbol = str(leg.get("symbol") or "")
    action = str(leg.get("action") or "").upper()
    quantity = leg.get("quantity")
    outside_rth = leg.get("outside_rth")
    if (
        symbol not in {"SPYU", "SPXU"}
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

    order_ref = xsp_v2_transport_order_ref(plan)
    records = tuple(ledger.records())
    prior = _prior_execution(
        records,
        selection_id=str(selected["selection_id"]),
        order_ref=order_ref,
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
            == XSP_V2_TRANSPORT_EXECUTION_VERSION
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


async def advance_xsp_v2_live_transport(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    source_receipt: Mapping[str, object],
    observed_at: datetime,
    quote_wait_seconds: float = 3.0,
) -> dict[str, object]:
    """Reconcile one selected sleeve, then execute at most one exact RTH leg."""

    selected = load_xsp_v2_transport_selection_from_mapping(selection)
    if observed_at.tzinfo is None or quote_wait_seconds < 0:
        raise ValueError("live transport observation inputs are invalid")
    broker_snapshot = await xsp_v2_broker_snapshot(client)
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
            and record.get("strategy_version") == XSP_V2_TRANSPORT_EXECUTION_VERSION
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
        Stock("SPYU", "SMART", "USD"),
        Stock("SPXU", "SMART", "USD"),
    )
    contracts = {
        str(getattr(contract, "symbol", "") or "").upper(): contract
        for contract in qualified
        if str(getattr(contract, "symbol", "") or "").upper() in {"SPYU", "SPXU"}
    }
    if set(contracts) != {"SPYU", "SPXU"} or any(
        int(getattr(contracts[symbol], "conId", 0) or 0) != int(contract_ids[symbol])
        for symbol in ("SPYU", "SPXU")
    ):
        raise ValueError("selected transport contract identity changed")
    tickers = {
        "SPYU": await client.ensure_ticker(
            contracts["SPYU"],
            owner="xsp-v2-live-transport",
            generic_ticks="577,614,623",
        ),
        "SPXU": await client.ensure_ticker(
            contracts["SPXU"],
            owner="xsp-v2-live-transport",
        ),
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

    nav_value, nav_at = client.generic_tick_value(tickers["SPYU"], 96)
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
                row["symbol"] in {"SPYU", "SPXU"}
                or str(row["order_ref"]).startswith("XSPV2-")
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
        execution = await execute_xsp_v2_transport_plan(
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
    risk_state = xsp_v2_transport_risk_state(
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
        for symbol in ("SPYU", "SPXU")
    ):
        raise ValueError("broker holdings disagree with selected-run fill ledger")
    plan = project_xsp_v2_transport_plan(
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
    execution = await execute_xsp_v2_transport_plan(
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
