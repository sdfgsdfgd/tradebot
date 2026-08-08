"""Pure Stage-76 fill-risk reconstruction and signal-to-order planning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from .gold_live_transport import (
    GOLD_LIVE_EXECUTION_VERSION,
    GOLD_LIVE_MAX_COMMISSION_USD,
    GOLD_LIVE_ORDER_AUTHORITY,
    GOLD_LIVE_PLAN_SCHEMA,
    GOLD_REGIME_HARMONY_SOURCE_VERSION,
    load_gold_live_selection_from_mapping,
)
from .live_calibration import calibration_fingerprint


GOLD_LIVE_ACTION_SOURCE_MAX_AGE_SECONDS = 15 * 60.0
GOLD_LIVE_HOLD_SOURCE_MAX_AGE_SECONDS = 4 * 60 * 60 + 20 * 60.0


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


def gold_transport_risk_state(
    *,
    selection: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    observed_at: datetime,
    liquidation_price: float,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    mark = _number(liquidation_price, name="gold liquidation price")
    fills_by_id: dict[str, dict[str, object]] = {}
    prior_risks = []
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != GOLD_LIVE_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selected["selection_id"]
        ):
            continue
        prior = evidence.get("risk_state")
        if isinstance(prior, Mapping):
            prior_risks.append(prior)
        order = evidence.get("broker_order")
        if evidence.get("phase") != "TERMINAL" or not isinstance(order, Mapping):
            continue
        order_fills = order.get("fills")
        if not isinstance(order_fills, Sequence) or isinstance(
            order_fills, (str, bytes)
        ):
            raise ValueError("terminal gold order has no fill ledger")
        for raw in order_fills:
            if not isinstance(raw, Mapping):
                raise ValueError("gold broker fill must be an object")
            fill = dict(raw)
            exec_id = str(fill.get("exec_id") or "")
            prior_fill = fills_by_id.get(exec_id)
            if not exec_id or (prior_fill is not None and prior_fill != fill):
                raise ValueError("gold broker execution identity changed")
            fills_by_id[exec_id] = fill

    position = 0.0
    entry_price = entry_commission = None
    realized_gross = realized_cost = 0.0
    closed_trade_gross: list[float] = []
    canonical = []
    for fill in sorted(
        fills_by_id.values(),
        key=lambda row: (str(row.get("time_utc") or ""), str(row["exec_id"])),
    ):
        side = str(fill.get("side") or "").upper()
        fill_time = _utc(fill.get("time_utc"))
        quantity = _number(fill.get("shares"), name="gold fill quantity")
        price = _number(fill.get("price"), name="gold fill price")
        commission = _number(fill.get("commission"), name="gold fill commission")
        signed = quantity if side in {"BOT", "BUY"} else -quantity
        if (
            side not in {"BOT", "BUY", "SLD", "SELL"}
            or fill_time > now
            or quantity != 1.0
            or price <= 0
            or commission < 0
            or str(fill.get("commission_currency") or "").upper() != "USD"
        ):
            raise ValueError("gold fill economics are invalid")
        if abs(position) <= 1e-9:
            position = signed
            entry_price = price
            entry_commission = commission
        elif signed == -position and entry_price is not None:
            gross = position * (price - entry_price)
            realized_gross += gross
            realized_cost += float(entry_commission or 0.0) + commission
            closed_trade_gross.append(gross)
            position = 0.0
            entry_price = entry_commission = None
        else:
            raise ValueError("gold fills overlap or exceed one selected contract")
        canonical.append(
            {
                "exec_id": fill["exec_id"],
                "time_utc": fill_time.isoformat(),
                "side": side,
                "price": price,
                "commission_usd": commission,
                "position_after": position,
            }
        )

    open_gross = (
        position * (mark - float(entry_price))
        if abs(position) > 1e-9 and entry_price is not None
        else 0.0
    )
    open_cost = (
        float(entry_commission or 0.0) + GOLD_LIVE_MAX_COMMISSION_USD
        if abs(position) > 1e-9
        else 0.0
    )
    run_gross = realized_gross + open_gross
    run_cost = realized_cost + open_cost
    run_net = run_gross - run_cost
    peak = max(
        0.0,
        run_net,
        *(float(row.get("peak_run_net_usd") or 0.0) for row in prior_risks),
    )
    drawdown = peak - run_net
    maximum = float(selected["risk"]["max_run_drawdown_usd"])
    wins = sorted((value for value in closed_trade_gross if value > 0), reverse=True)
    return {
        "valid": True,
        "as_of_utc": now.isoformat(),
        "position_from_fills": position,
        "entry_price": entry_price,
        "run_realized_gross_usd": realized_gross,
        "run_realized_cost_usd": realized_cost,
        "run_realized_net_usd": realized_gross - realized_cost,
        "open_mark_gross_usd": open_gross,
        "open_mark_cost_usd": open_cost,
        "open_mark_net_usd": open_gross - open_cost,
        "run_gross_usd": run_gross,
        "run_cost_usd": run_cost,
        "run_net_usd": run_net,
        "peak_run_net_usd": peak,
        "drawdown_usd": drawdown,
        "closed_trades": len(closed_trade_gross),
        "gross_wins_usd": sum(wins),
        "top_five_gross_wins_usd": sum(wins[:5]),
        "fill_count": len(canonical),
        "fill_ledger_fingerprint": calibration_fingerprint(canonical),
        "attribution_complete": True,
        "safety_breaches": (
            ["run_drawdown_limit_breached"] if drawdown > maximum else []
        ),
    }


def project_gold_transport_plan(
    *,
    selection: Mapping[str, object],
    source_checkpoint: Mapping[str, object],
    broker_position: float,
    open_orders: Sequence[Mapping[str, object]],
    risk_state: Mapping[str, object],
    observed_at: datetime,
    entry_market_data_eligible: bool = True,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    source_at = _utc(source_checkpoint.get("recorded_at_utc"))
    evidence = source_checkpoint.get("evidence")
    if (
        source_checkpoint.get("strategy_version")
        != GOLD_REGIME_HARMONY_SOURCE_VERSION
        or source_checkpoint.get("status") != "EVALUATED"
        or not isinstance(evidence, Mapping)
        or evidence.get("schema")
        != "gold.1oz-regime-harmony-source-checkpoint.v1"
        or evidence.get("source_usable") is not True
        or now < source_at
    ):
        raise ValueError("gold live source is invalid")
    if not isinstance(entry_market_data_eligible, bool):
        raise ValueError("gold entry market-data state is invalid")
    if open_orders:
        raise ValueError("gold plan requires a reconciled order-free broker state")
    held = _number(broker_position, name="gold broker position")
    if held not in {-1.0, 0.0, 1.0}:
        raise ValueError("gold broker position exceeds the one-contract canary")
    target = evidence.get("target")
    target_direction = None
    target_id = None
    target_time = None
    if target is not None:
        if not isinstance(target, Mapping):
            raise ValueError("gold source target is invalid")
        target_direction = str(target.get("direction") or "")
        target_id = str(target.get("target_id") or "")
        target_time = _utc(target.get("entry_time_utc"))
        if target_direction not in {"up", "down"} or len(target_id) != 64:
            raise ValueError("gold source target identity is invalid")
    held_direction = "up" if held > 0 else "down" if held < 0 else None
    source_age = (now - source_at).total_seconds()
    source_stale = source_age > GOLD_LIVE_HOLD_SOURCE_MAX_AGE_SECONDS
    breaches = risk_state.get("safety_breaches")
    if not isinstance(breaches, Sequence) or isinstance(breaches, (str, bytes)):
        raise ValueError("gold risk state is invalid")

    action = reason = None
    desired_after_close = target_direction
    if held_direction is not None and breaches:
        action = "SELL" if held > 0 else "BUY"
        reason = "risk_reduction"
        desired_after_close = None
    elif held_direction is not None and held_direction != target_direction:
        action = "SELL" if held > 0 else "BUY"
        reason = (
            "close_before_reverse" if target_direction is not None else "signal_flat_exit"
        )
    elif held_direction is None and not entry_market_data_eligible:
        reason = "entry_market_data_unavailable"
        desired_after_close = None
    elif held_direction is None and target_direction is not None:
        if target_time is None or target_time <= _utc(selected["selected_at_utc"]):
            reason = "preselection_target_not_adopted"
        elif source_age > GOLD_LIVE_ACTION_SOURCE_MAX_AGE_SECONDS:
            reason = "entry_source_stale"
        else:
            action = "BUY" if target_direction == "up" else "SELL"
            reason = "fresh_stage76_admission"
    else:
        reason = (
            "target_already_owned" if held_direction is not None else "flat_no_target"
        )
    body = {
        "schema": GOLD_LIVE_PLAN_SCHEMA,
        "selection_id": selected["selection_id"],
        "source_checkpoint_id": source_checkpoint["checkpoint_id"],
        "source_recorded_at_utc": source_checkpoint["recorded_at_utc"],
        "source_age_seconds": source_age,
        "source_stale": source_stale,
        "target_id": target_id,
        "target_direction": target_direction,
        "held_direction": held_direction,
        "desired_after_close": desired_after_close,
        "reason": reason,
        "status": "ACTIONABLE" if action is not None else "HOLD",
        "leg": (
            {
                "symbol": "1OZ",
                "action": action,
                "quantity": 1,
                "initial_mode": "OPTIMISTIC",
                "chase_mode": "AUTO",
                "outside_rth": True,
            }
            if action is not None
            else None
        ),
        "capital_admission": None,
        "execution_state_context": {
            "schema": "gold.1oz-execution-state-context.v1",
            "owner_state": evidence.get("owner_state"),
            "signal": evidence.get("signal_context"),
            "macro": evidence.get("macro_context"),
            "news": evidence.get("fundamental_pressure"),
            "contract_pair": evidence.get("contract_pair"),
        },
        "order_authority": GOLD_LIVE_ORDER_AUTHORITY,
    }
    transition = {
        key: body[key]
        for key in (
            "selection_id",
            "source_checkpoint_id",
            "target_id",
            "target_direction",
            "held_direction",
            "desired_after_close",
            "reason",
            "leg",
        )
    }
    return {**body, "transition_id": calibration_fingerprint(transition)}
