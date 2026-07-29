"""Restart-safe cash and drawdown state for selected XSP transports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from ..engines.market import is_trading_day, xsp_trading_date
from .live_calibration import calibration_fingerprint
from .xsp_live_transport import (
    XSP_V2_TRANSPORT_SELECTION_SCHEMA,
    load_xsp_transport_selection_from_mapping,
    xsp_transport_contract,
)


def xsp_transport_risk_state(
    *,
    selection: Mapping[str, object],
    records: tuple[Mapping[str, object], ...],
    observed_at: datetime,
    liquidation_bids: Mapping[str, float],
) -> dict[str, object]:
    """Reconstruct selected-sleeve equity from immutable broker fills."""

    selected = load_xsp_transport_selection_from_mapping(selection)
    contract = xsp_transport_contract(selected)
    symbols = tuple(contract["symbols"])
    execution_version = str(contract["execution_version"])
    if observed_at.tzinfo is None:
        raise ValueError("risk observation timestamp must be aware")
    fills_by_id: dict[str, dict[str, object]] = {}
    prior_risks = []
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != execution_version
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
    holdings = {symbol: 0.0 for symbol in symbols}
    open_cost = {symbol: 0.0 for symbol in symbols}
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
            symbol not in symbols
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


def xsp_v2_transport_risk_state(**kwargs) -> dict[str, object]:
    selection = kwargs.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema") != XSP_V2_TRANSPORT_SELECTION_SCHEMA
    ):
        raise ValueError("v2 risk state requires a v2 selected transport")
    return xsp_transport_risk_state(**kwargs)
