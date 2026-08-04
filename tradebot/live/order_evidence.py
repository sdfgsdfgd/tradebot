"""Shared live-order evidence and execution-price projection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone

from ..engines.execution import (
    EXECUTION_POLICY,
    execution_policy_contract,
    execution_price,
)
from .capital import validate_live_capital_decision
from .execution import LiveOrderExecution, order_ids


def _evidence_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _finite_number(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


async def broker_account_snapshot(
    client,
    *,
    base_currency: str,
) -> dict[str, object]:
    """Capture one cash account and base-converted portfolio for live admission."""

    base = str(base_currency or "").strip().upper()
    portfolio = await client.fetch_portfolio()
    account_id = str(client.account_id() or "").strip()
    account_type = str(client.account_text_value("TradingType-S") or "").upper()
    if not base or not account_id or account_type != "STKCASH":
        raise ValueError("live admission requires one cash account and base currency")

    def account(tag: str, currency: str) -> float:
        value, actual, _updated = client.account_value(tag, currency=currency)
        if str(actual or "").upper() != currency:
            raise ValueError(f"fresh {tag} {currency} is unavailable")
        return _finite_number(value, name=f"{tag} {currency}")

    rates = {base: 1.0}

    def base_rate(currency: str) -> float:
        normalized = str(currency or "").strip().upper()
        if not normalized:
            raise ValueError("broker position currency is unavailable")
        if normalized not in rates:
            rates[normalized] = account("ExchangeRate", normalized)
        return rates[normalized]

    positions = []
    for item in portfolio:
        contract = getattr(item, "contract", None)
        quantity = _finite_number(
            getattr(item, "position", 0.0) or 0.0,
            name="broker position",
        )
        if abs(quantity) <= 1e-9:
            continue
        currency = str(getattr(contract, "currency", "") or "").upper()
        positions.append(
            {
                "symbol": str(getattr(contract, "symbol", "") or "").upper(),
                "local_symbol": str(getattr(contract, "localSymbol", "") or ""),
                "con_id": int(getattr(contract, "conId", 0) or 0),
                "sec_type": str(getattr(contract, "secType", "") or ""),
                "currency": currency,
                "quantity": quantity,
                "market_value_base_cents": math.ceil(
                    abs(
                        _finite_number(
                            getattr(item, "marketValue", 0.0) or 0.0,
                            name="broker position market value",
                        )
                    )
                    * base_rate(currency)
                    * 100
                ),
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
                "quantity": _finite_number(
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
        "base_currency": base,
        "settled_cash_usd": account("CashBalance", "USD"),
        "equity_with_loan_base": account("EquityWithLoanValue", base),
        "available_funds_base": account("AvailableFunds", base),
        "excess_liquidity_base": account("ExcessLiquidity", base),
        "initial_margin_base": account("FullInitMarginReq", base),
        "maintenance_margin_base": account("FullMaintMarginReq", base),
        "gross_position_value_base": account("GrossPositionValue", base),
        "usd_to_base_rate": base_rate("USD"),
        "positions": positions,
        "open_orders": open_orders,
    }


def _gate(
    status: str, reasons: Sequence[str], evidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def broker_trade_snapshot(trade: object) -> dict[str, object]:
    """Freeze the broker-visible order and exact fill economics."""

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
                    if isinstance(fill_time, datetime)
                    and fill_time.tzinfo is not None
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


def terminal_broker_snapshot_complete(snapshot: Mapping[str, object]) -> bool:
    """Require terminal broker state plus complete, USD-costed fills."""

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


async def execute_single_contract_limit_order(
    *,
    client,
    contract: object,
    ticker: object,
    action: str,
    order_ref: str,
    plan: Mapping[str, object],
    latest_checkpoint: Mapping[str, object] | None,
    observed_at: datetime,
    max_commission_usd: float,
    initial_mode: str,
    chase_mode: str,
    price_for_mode: Callable[..., float | None],
    checkpoint: Callable[..., Mapping[str, object]],
    ladder_schema: str,
    source_age_seconds: float,
) -> dict[str, object]:
    """Execute or reconcile one durable, capital-admitted LIMIT transition."""

    prior = (
        latest_checkpoint.get("evidence")
        if isinstance(latest_checkpoint, Mapping)
        else None
    )
    matches = await client.reconcile_trades_for_order_ref(order_ref)
    if len(matches) > 1:
        raise ValueError("broker returned multiple orders for one transition")
    trade = matches[0] if matches else None
    if trade is not None and not isinstance(prior, Mapping):
        raise ValueError("broker order has no prepared local transition")
    if isinstance(prior, Mapping) and prior.get("phase") == "TERMINAL":
        snapshot = broker_trade_snapshot(trade) if trade is not None else {}
        if not _complete_single_contract_fill(snapshot):
            raise ValueError("terminal receipt disagrees with broker")
        return {
            "status": "TERMINAL",
            "order_ref": order_ref,
            "checkpoint_id": latest_checkpoint["checkpoint_id"],
            "submitted_orders": 0,
            "broker_order": snapshot,
        }
    if trade is None and isinstance(prior, Mapping) and prior.get("phase") == "SUBMITTED":
        raise ValueError("submitted order disappeared from broker state")
    preview_payload = (
        dict(prior["what_if_preview"])
        if isinstance(prior, Mapping)
        and isinstance(prior.get("what_if_preview"), Mapping)
        else None
    )
    initial_price = price_for_mode(initial_mode, action)
    if initial_price is None or not math.isfinite(float(initial_price)):
        raise ValueError("transition has no executable initial LIMIT price")
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
        commission_values = [
            float(value)
            for value in (
                preview.commission,
                preview.min_commission,
                preview.max_commission,
            )
            if value is not None
        ]
        if (
            preview.status != "PreSubmitted"
            or not commission_values
            or any(not math.isfinite(value) or value < 0 for value in commission_values)
            or str(preview.commission_currency or "").upper() != "USD"
            or max(commission_values) > max_commission_usd
            or str(preview.warning_text or "")
            or not isinstance(plan.get("capital_admission"), Mapping)
            or plan["capital_admission"].get("status") != "ALLOW"
        ):
            raise ValueError("fresh LIMIT what-if or capital admission failed")
        checkpoint(
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
        checkpoint(
            phase="SUBMITTED",
            observed_at=observed_at,
            order_ref=order_ref,
            preview=preview_payload,
            trade=trade,
            submitted_orders=1,
        )

    order_id, perm_id = order_ids(trade)
    submitted_at = (
        datetime.fromisoformat(
            str(latest_checkpoint["recorded_at_utc"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        if latest_checkpoint is not None
        else observed_at.astimezone(timezone.utc)
    )
    elapsed = max(
        0.0,
        (observed_at.astimezone(timezone.utc) - submitted_at).total_seconds(),
    )

    def transition(payload: dict[str, object]) -> None:
        at = datetime.now(timezone.utc)
        checkpoint(
            phase="SUBMITTED",
            observed_at=at,
            order_ref=order_ref,
            preview=preview_payload,
            ladder_transition={
                "schema": ladder_schema,
                "observed_at_utc": at.isoformat(),
                "source_age_seconds": source_age_seconds,
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
            mode=chase_mode,
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
    if broker_order.get("done") is True and not _complete_single_contract_fill(
        broker_order
    ):
        raise ValueError("LIMIT order terminated without one complete fill")
    if not terminal_broker_snapshot_complete(broker_order):
        pending = checkpoint(
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
    terminal = checkpoint(
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


def _complete_single_contract_fill(snapshot: Mapping[str, object]) -> bool:
    try:
        return bool(
            terminal_broker_snapshot_complete(snapshot)
            and float(snapshot.get("filled") or 0.0) == 1.0
            and float(snapshot.get("remaining") or 0.0) == 0.0
            and float(snapshot.get("quantity") or 0.0) == 1.0
        )
    except (TypeError, ValueError):
        return False


def single_contract_execution_graduation_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    sleeve_id: str,
    symbol: str,
    con_id: int,
    order_ref_prefix: str,
    ladder_schema: str,
    max_commission_usd: float,
) -> dict[str, object]:
    """Grade one restart-safe, centrally admitted futures execution prefix."""

    order_rows = [
        row
        for row in rows
        if isinstance(row.get("evidence"), Mapping)
        and str(row["evidence"].get("order_ref") or "")
    ]
    if not order_rows:
        return _gate(
            "HOLD", ["execution_not_observed"], {"orders": 0, "fills": 0}
        )
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in order_rows:
        evidence = row["evidence"]
        assert isinstance(evidence, Mapping)
        groups.setdefault(str(evidence["order_ref"]), []).append(evidence)
    invalid: list[str] = []
    failures: list[str] = []
    pending: list[str] = []
    stops: list[str] = []
    terminal_orders = fills = 0
    commission = 0.0
    exec_ids: set[str] = set()
    timeout = float(execution_policy_contract()["auto_timeout_seconds"])
    for order_ref, evidence_rows in groups.items():
        plans = [row.get("plan") for row in evidence_rows]
        if not plans or not isinstance(plans[0], Mapping) or any(
            _evidence_sha256(plan) != _evidence_sha256(plans[0])
            for plan in plans[1:]
        ):
            invalid.append("order_plan_missing_or_changed")
            continue
        plan = plans[0]
        assert isinstance(plan, Mapping)
        transition = str(plan.get("transition_id") or "")
        phases = [str(row.get("phase") or "") for row in evidence_rows]
        phase_rank = {"PREPARED": 0, "SUBMITTED": 1, "TERMINAL": 2}
        if len(transition) != 64 or order_ref != f"{order_ref_prefix}-{transition[:24]}":
            invalid.append("order_transition_identity_invalid")
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
        submissions = [int(row.get("submitted_orders") or 0) for row in evidence_rows]
        if any(value not in {0, 1} for value in submissions) or 1 not in submissions:
            stops.append("duplicate_or_unauthorized_submission")
        leg = plan.get("leg")
        try:
            admission = validate_live_capital_decision(plan["capital_admission"])
            valid_leg = bool(
                isinstance(leg, Mapping)
                and leg.get("symbol") == symbol
                and str(leg.get("action") or "").upper() in {"BUY", "SELL"}
                and int(leg.get("quantity") or 0) == 1
                and admission.get("status") == "ALLOW"
                and admission.get("run_id") == selection_id
                and admission.get("sleeve_id") == sleeve_id
            )
        except (KeyError, TypeError, ValueError):
            valid_leg = False
        if not valid_leg:
            invalid.append("order_leg_or_capital_admission_invalid")
            continue
        previews = [row.get("what_if_preview") for row in evidence_rows]
        preview = previews[0]
        try:
            if not isinstance(preview, Mapping) or any(
                not isinstance(value, Mapping)
                or _evidence_sha256(value) != _evidence_sha256(preview)
                for value in previews[1:]
            ):
                raise TypeError
            preview_values = [
                float(preview[key])
                for key in ("commission", "min_commission", "max_commission")
                if preview.get(key) is not None
            ]
            preview_valid = bool(
                preview.get("status") == "PreSubmitted"
                and str(preview.get("commission_currency") or "").upper()
                == "USD"
                and preview_values
                and all(
                    math.isfinite(value) and value >= 0 for value in preview_values
                )
                and max(preview_values) <= max_commission_usd
                and not str(preview.get("warning_text") or "")
            )
        except (KeyError, TypeError, ValueError):
            preview_valid = False
        if not preview_valid:
            failures.append("fresh_preview_boundary_breached")
        for transition_row in (
            row.get("ladder_transition") for row in evidence_rows
        ):
            if transition_row is None:
                continue
            try:
                elapsed = float(transition_row["elapsed_seconds"])
            except (KeyError, TypeError, ValueError):
                elapsed = math.nan
            if (
                not isinstance(transition_row, Mapping)
                or transition_row.get("schema") != ladder_schema
                or transition_row.get("event") != "ladder_mode_transition"
                or transition_row.get("active_mode")
                not in {"OPT", "MID", "AGG", "CROSS", "RLT"}
                or str(transition_row.get("action") or "").upper()
                != str(leg.get("action") or "").upper()
                or not math.isfinite(elapsed)
                or not 0 <= elapsed <= timeout + 1e-9
            ):
                failures.append("execution_ladder_contract_breached")
        terminal = next(
            (row for row in evidence_rows if row.get("phase") == "TERMINAL"),
            None,
        )
        if terminal is None:
            pending.append("terminal_execution_pending")
            continue
        snapshot = terminal.get("broker_order")
        try:
            complete = bool(
                isinstance(snapshot, Mapping)
                and terminal_broker_snapshot_complete(snapshot)
                and snapshot.get("order_ref") == order_ref
                and snapshot.get("symbol") == symbol
                and int(snapshot.get("con_id") or 0) == con_id
                and float(snapshot.get("filled") or 0) == 1
                and float(snapshot.get("remaining") or 0) == 0
                and int(float(snapshot.get("quantity") or 0)) == 1
                and str(snapshot.get("action") or "").upper()
                == str(leg.get("action") or "").upper()
            )
        except (KeyError, TypeError, ValueError):
            complete = False
        if not complete:
            failures.append("terminal_fill_invalid")
            continue
        terminal_orders += 1
        action = str(leg.get("action") or "").upper()
        try:
            limit_price = float(snapshot["limit_price"])
        except (KeyError, TypeError, ValueError):
            limit_price = math.nan
        for fill in snapshot["fills"]:
            exec_id = str(fill.get("exec_id") or "")
            try:
                fill_price = float(fill["price"])
                fill_valid = bool(
                    fill.get("symbol") == symbol
                    and str(fill.get("side") or "").upper()
                    in ({"BOT", "BUY"} if action == "BUY" else {"SLD", "SELL"})
                    and math.isfinite(limit_price)
                    and (
                        fill_price <= limit_price + 1e-9
                        if action == "BUY"
                        else fill_price >= limit_price - 1e-9
                    )
                )
            except (KeyError, TypeError, ValueError):
                fill_valid = False
            if not fill_valid:
                failures.append("terminal_fill_execution_boundary_breached")
            if exec_id in exec_ids:
                stops.append("duplicate_fill_attribution")
            exec_ids.add(exec_id)
            fills += 1
            commission += float(fill["commission"])
        if commission > terminal_orders * max_commission_usd + 1e-9:
            failures.append("commission_limit_breached")
    return _gate(
        "STOP"
        if stops
        else "INVALID"
        if invalid
        else "FAIL"
        if failures
        else "HOLD"
        if pending
        else "PASS",
        [*stops, *invalid, *failures, *pending],
        {
            "orders": len(groups),
            "terminal_orders": terminal_orders,
            "fills": fills,
            "commission_usd": commission,
        },
    )


def execution_price_for_ticker(contract: object, ticker: object):
    """Bind the shared execution ladder to one live ticker."""

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
