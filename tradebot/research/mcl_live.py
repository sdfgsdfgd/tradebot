"""Immutable selection and finalized source ownership for crowned MCL V18."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta

from ib_insync import Contract

from ..chart_data.series import OhlcvBar
from ..live.capital import admit_live_capital
from ..live.order_evidence import (
    broker_account_snapshot,
    broker_trade_snapshot,
    execute_single_contract_limit_order,
    execution_price_for_ticker,
)
from .live_calibration import LiveCalibrationLedger
from .mcl_live_transport import (
    MCL_LIVE_ADMISSION_MAX_AGE_SECONDS,
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_EXECUTION_SCHEMA,
    MCL_LIVE_EXECUTION_VERSION,
    MCL_LIVE_MAX_COMMISSION_USD,
    MCL_LIVE_MAX_RUN_DRAWDOWN_USD,
    MCL_LIVE_ORDER_AUTHORITY,
    MCL_LIVE_ORDER_REF_PREFIX,
    MCL_LIVE_PLAN_SCHEMA,
    MCL_LIVE_RAW_LOSS_CAP_USD,
    MCL_LIVE_SOURCE_SCHEMA,
    MCL_LIVE_SOURCE_VERSION,
    MCL_LIVE_WEEKLY_FLAT_ET,
    _ET,
    _bar_map,
    _identity,
    _is_sha,
    _live_quote,
    _number,
    _utc,
    load_mcl_live_selection_from_mapping,
    mcl_live_contracts,
)
from .mcl_live_reopen import (
    MCL_LIVE_SOURCE_AUTHORITY_FRESH,
    MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
    refresh_mcl_live_source,
)
from .mcl_two_speed_auction import (
    MCL_TWO_SPEED_AUCTION_MULTIPLIER,
)


def _latest_execution_by_ref(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> dict[str, Mapping[str, object]]:
    output = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == MCL_LIVE_EXECUTION_VERSION
            and isinstance(evidence, Mapping)
            and evidence.get("selection_id") == selection_id
            and str(evidence.get("order_ref") or "")
        ):
            output[str(evidence["order_ref"])] = record
    return output


def _consumed_admissions(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> set[str]:
    consumed = set()
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("strategy_version") != MCL_LIVE_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selection_id
        ):
            continue
        plan = evidence.get("plan")
        if not isinstance(plan, Mapping) or plan.get("reason") not in {
            "fresh_v18_admission",
            "fresh_source_admission",
        }:
            continue
        event_id = str(plan.get("admission_event_id") or "")
        if _is_sha(event_id):
            consumed.add(event_id)
    return consumed


def _weekly_flat_due(observed_at: datetime) -> bool:
    local = _utc(observed_at).astimezone(_ET)
    weekday, hour, minute = MCL_LIVE_WEEKLY_FLAT_ET
    return local.weekday() == weekday and (local.hour, local.minute) >= (
        hour,
        minute,
    )


def mcl_transport_risk_state(
    *,
    selection: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    observed_at: datetime,
    liquidation_price: float,
    completed_mcl_bars: Sequence[OhlcvBar] = (),
) -> dict[str, object]:
    """Reconstruct exact selected fills and actual one-contract safety state."""

    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    mark = _number(liquidation_price, name="MCL liquidation price")
    plans: dict[str, Mapping[str, object]] = {}
    fills: dict[str, dict[str, object]] = {}
    prior_risks = []
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("strategy_version") != MCL_LIVE_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selected["selection_id"]
        ):
            continue
        prior = evidence.get("risk_state")
        if isinstance(prior, Mapping):
            prior_risks.append(prior)
        order_ref = str(evidence.get("order_ref") or "")
        plan = evidence.get("plan")
        if order_ref and isinstance(plan, Mapping):
            prior_plan = plans.get(order_ref)
            if prior_plan is not None and dict(prior_plan) != dict(plan):
                raise ValueError("MCL durable order plan changed")
            plans[order_ref] = plan
        order = evidence.get("broker_order")
        if evidence.get("phase") != "TERMINAL" or not isinstance(order, Mapping):
            continue
        if int(order.get("con_id") or 0) != int(selected["contracts"]["MCL"]["con_id"]):
            raise ValueError("MCL terminal fill changed contract identity")
        order_fills = order.get("fills")
        if not isinstance(order_fills, Sequence) or isinstance(
            order_fills, (str, bytes)
        ):
            raise ValueError("MCL terminal order has no fill ledger")
        for raw in order_fills:
            if not isinstance(raw, Mapping):
                raise ValueError("MCL broker fill must be one object")
            fill = {**dict(raw), "order_ref": order_ref}
            exec_id = str(fill.get("exec_id") or "")
            previous = fills.get(exec_id)
            if not exec_id or (previous is not None and previous != fill):
                raise ValueError("MCL broker execution identity changed")
            fills[exec_id] = fill

    position = 0
    entry_price = entry_commission = entry_time = route = admission_id = owner = None
    open_exec_id = None
    realized_gross = realized_cost = 0.0
    closed_trade_gross: list[float] = []
    canonical = []
    for fill in sorted(
        fills.values(), key=lambda row: (str(row.get("time_utc") or ""), row["exec_id"])
    ):
        side = str(fill.get("side") or "").upper()
        signed = 1 if side in {"BOT", "BUY"} else -1 if side in {"SLD", "SELL"} else 0
        fill_time = _utc(fill.get("time_utc"))
        quantity = _number(fill.get("shares"), name="MCL fill quantity")
        price = _number(fill.get("price"), name="MCL fill price")
        commission = _number(fill.get("commission"), name="MCL fill commission")
        plan = plans.get(str(fill.get("order_ref") or ""))
        if (
            signed == 0
            or fill_time > now
            or quantity != 1.0
            or price <= 0
            or not 0 <= commission <= MCL_LIVE_MAX_COMMISSION_USD
            or str(fill.get("commission_currency") or "").upper() != "USD"
            or not isinstance(plan, Mapping)
        ):
            raise ValueError("MCL fill economics are invalid")
        canonical.append(fill)
        if position == 0:
            position = signed
            entry_price = price
            entry_commission = commission
            entry_time = fill_time
            route = plan.get("target_route")
            owner = plan.get("target_owner", "v18")
            admission_id = plan.get("admission_event_id")
            open_exec_id = fill["exec_id"]
            if (
                route
                not in {
                    "continuation",
                    "failed_auction",
                    "shock_continuation",
                    "shock_reacquisition",
                }
                or owner not in {"v18", "shock"}
                or not _is_sha(admission_id)
            ):
                raise ValueError("MCL opening fill lacks its source admission")
        elif position + signed == 0:
            assert entry_price is not None and entry_commission is not None
            gross = (
                position * (price - entry_price) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
            )
            realized_gross += gross
            realized_cost += entry_commission + commission
            closed_trade_gross.append(gross)
            position = 0
            entry_price = entry_commission = entry_time = route = admission_id = None
            owner = None
            open_exec_id = None
        else:
            raise ValueError("MCL fill ledger exceeded one contract or crossed zero")

    mfe = mae = 0.0
    marked_through = entry_time
    prior = next(
        (
            item
            for item in reversed(prior_risks)
            if item.get("open_exec_id") == open_exec_id and open_exec_id is not None
        ),
        None,
    )
    if prior is not None:
        mfe = _number(prior.get("mfe_usd"), name="MCL prior MFE")
        mae = _number(prior.get("mae_usd"), name="MCL prior MAE")
        marked_through = _utc(prior.get("marked_through_utc"))
    safety_breaches = []
    exit_triggers = []
    profit_memory_stop = None
    if position and entry_price is not None and entry_time is not None:
        for bar in sorted(completed_mcl_bars, key=lambda row: row.ts):
            bar_at = _utc(bar.ts)
            if bar_at <= max(entry_time, marked_through or entry_time) or bar_at > now:
                continue
            if (
                owner == "v18"
                and route == "failed_auction"
                and mfe >= entry_price * 0.5
            ):
                protected = 0.25 * mfe
                stop = entry_price + position * protected / MCL_TWO_SPEED_AUCTION_MULTIPLIER
                crossed = (
                    position > 0 and float(bar.low) <= stop
                ) or (position < 0 and float(bar.high) >= stop)
                if crossed:
                    profit_memory_stop = stop
                    exit_triggers.append("failed_auction_profit_memory")
                    marked_through = bar_at
                    break
            favorable = (
                (float(bar.high) - entry_price) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
                if position > 0
                else (entry_price - float(bar.low)) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
            )
            adverse = (
                (float(bar.low) - entry_price) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
                if position > 0
                else (entry_price - float(bar.high)) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
            )
            mfe = max(mfe, favorable)
            mae = min(mae, adverse)
            marked_through = bar_at
        unrealized = position * (mark - entry_price) * MCL_TWO_SPEED_AUCTION_MULTIPLIER
        if unrealized <= -MCL_LIVE_RAW_LOSS_CAP_USD:
            safety_breaches.append("raw_loss_cap")
        if _weekly_flat_due(now):
            exit_triggers.append("weekly_closure")
    else:
        unrealized = 0.0
        marked_through = None
    open_cost = (
        float(entry_commission or 0.0) + MCL_LIVE_MAX_COMMISSION_USD
        if position
        else 0.0
    )
    run_gross = realized_gross + unrealized
    run_cost = realized_cost + open_cost
    run_net = run_gross - run_cost
    peak = max(
        0.0,
        run_net,
        *(float(row.get("peak_run_net_usd") or 0.0) for row in prior_risks),
    )
    drawdown = peak - run_net
    if drawdown > MCL_LIVE_MAX_RUN_DRAWDOWN_USD:
        safety_breaches.append("run_drawdown_limit_breached")
    wins = sorted((value for value in closed_trade_gross if value > 0), reverse=True)
    return {
        "schema": "mcl.two-speed-auction-risk-state.v1",
        "valid": True,
        "attribution_complete": True,
        "selection_id": selected["selection_id"],
        "observed_at_utc": now.isoformat(),
        "as_of_utc": now.isoformat(),
        "position_from_fills": position,
        "open_exec_id": open_exec_id,
        "entry_time_utc": entry_time.isoformat() if entry_time is not None else None,
        "entry_price": entry_price,
        "entry_commission_usd": entry_commission,
        "route": route,
        "owner": owner,
        "admission_event_id": admission_id,
        "liquidation_price": mark,
        "unrealized_raw_usd": unrealized,
        "mfe_usd": mfe,
        "mae_usd": mae,
        "marked_through_utc": (
            marked_through.isoformat() if marked_through is not None else None
        ),
        "profit_memory_stop": profit_memory_stop,
        "run_realized_gross_usd": round(realized_gross, 8),
        "run_realized_cost_usd": round(realized_cost, 8),
        "run_realized_net_usd": round(realized_gross - realized_cost, 8),
        "open_mark_gross_usd": round(unrealized, 8),
        "open_mark_cost_usd": round(open_cost, 8),
        "open_mark_net_usd": round(unrealized - open_cost, 8),
        "run_gross_usd": round(run_gross, 8),
        "run_cost_usd": round(run_cost, 8),
        "run_net_usd": round(run_net, 8),
        "peak_run_net_usd": round(peak, 8),
        "drawdown_usd": round(drawdown, 8),
        "closed_trades": len(closed_trade_gross),
        "gross_wins_usd": round(sum(wins), 8),
        "top_five_gross_wins_usd": round(sum(wins[:5]), 8),
        "fill_count": len(canonical),
        "fill_ledger_fingerprint": _identity(canonical),
        "exit_triggers": sorted(set(exit_triggers)),
        "safety_breaches": sorted(set(safety_breaches)),
    }


def project_mcl_transport_plan(
    *,
    selection: Mapping[str, object],
    source_checkpoint: Mapping[str, object],
    source_authority: str,
    broker_position: float,
    risk_state: Mapping[str, object],
    consumed_admissions: set[str],
    observed_at: datetime,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    if source_authority not in {
        MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
    }:
        raise ValueError("MCL live source authority is invalid")
    evidence = source_checkpoint.get("evidence")
    if (
        source_checkpoint.get("strategy_version") != MCL_LIVE_SOURCE_VERSION
        or source_checkpoint.get("status") != "EVALUATED"
        or not isinstance(evidence, Mapping)
        or evidence.get("schema") != MCL_LIVE_SOURCE_SCHEMA
        or evidence.get("selection_id") != selected["selection_id"]
    ):
        raise ValueError("MCL live source checkpoint is invalid")
    held = _number(broker_position, name="MCL broker position")
    if held not in {-1.0, 0.0, 1.0}:
        raise ValueError("MCL broker position exceeds one contract")
    target = evidence.get("target")
    target_direction = target_route = target_owner = admission_id = None
    target_at = None
    if target is not None:
        if not isinstance(target, Mapping):
            raise ValueError("MCL source target is invalid")
        target_direction = int(target.get("direction") or 0)
        target_route = str(target.get("route") or "")
        target_owner = str(target.get("owner") or "v18")
        admission_id = str(target.get("event_id") or "")
        target_at = _utc(target.get("observed_at_utc"))
        if (
            target_direction not in {-1, 1}
            or target_route
            not in {
                "continuation",
                "failed_auction",
                "shock_continuation",
                "shock_reacquisition",
            }
            or target_owner not in {"v18", "shock"}
            or not _is_sha(admission_id)
        ):
            raise ValueError("MCL source target identity is invalid")
    held_direction = 1 if held > 0 else -1 if held < 0 else None
    breaches = risk_state.get("safety_breaches")
    exit_triggers = risk_state.get("exit_triggers", [])
    if not isinstance(breaches, Sequence) or isinstance(breaches, (str, bytes)):
        raise ValueError("MCL risk state is invalid")
    if not isinstance(exit_triggers, Sequence) or isinstance(
        exit_triggers, (str, bytes)
    ):
        raise ValueError("MCL risk exit triggers are invalid")
    breaches = [str(value) for value in breaches]
    exit_triggers = [str(value) for value in exit_triggers]
    held_admission_id = risk_state.get("admission_event_id")
    action = reason = None
    if held_direction is not None and (breaches or exit_triggers):
        action = "SELL" if held_direction > 0 else "BUY"
        reason = (breaches or exit_triggers)[0]
    elif held_direction is not None and held_direction != target_direction:
        action = "SELL" if held_direction > 0 else "BUY"
        reason = "raw_turn_or_source_flatten"
    elif (
        held_direction is not None
        and admission_id is not None
        and held_admission_id != admission_id
    ):
        action = "SELL" if held_direction > 0 else "BUY"
        reason = "source_admission_identity_changed"
    elif held_direction is None and breaches:
        reason = breaches[0]
    elif held_direction is None and _weekly_flat_due(now):
        reason = "weekly_closure_entry_lock"
    elif held_direction is None and source_authority == MCL_LIVE_SOURCE_AUTHORITY_REOPEN:
        reason = (
            "maintenance_reopen_entry_locked"
            if target_direction is not None
            else "maintenance_reopen_reconciliation_only"
        )
    elif held_direction is None and target_direction is not None:
        assert target_at is not None and admission_id is not None
        due_at = target_at + timedelta(minutes=1)
        age = (now - due_at).total_seconds()
        if target_at <= _utc(selected["selected_at_utc"]):
            reason = "preselection_target_not_adopted"
        elif admission_id in consumed_admissions:
            reason = "admission_already_consumed"
        elif age < 0:
            reason = "next_minute_entry_not_due"
        elif age > MCL_LIVE_ADMISSION_MAX_AGE_SECONDS:
            reason = "entry_source_stale"
        else:
            action = "BUY" if target_direction > 0 else "SELL"
            reason = "fresh_source_admission"
    else:
        reason = "target_already_owned" if held_direction else "flat_no_target"
    reduction = held_direction is not None and action is not None
    latest_source = evidence["source"]["latest_common_close_utc"]
    source_age = (now - _utc(latest_source)).total_seconds()
    body = {
        "schema": MCL_LIVE_PLAN_SCHEMA,
        "strategy_version": selected["strategy_version"],
        "selection_id": selected["selection_id"],
        "source_checkpoint_id": source_checkpoint["checkpoint_id"],
        "source_authority": source_authority,
        "source_recorded_at_utc": source_checkpoint["recorded_at_utc"],
        "source_age_seconds": source_age,
        "admission_event_id": admission_id,
        "target_direction": target_direction,
        "target_route": target_route,
        "target_owner": target_owner,
        "held_direction": held_direction,
        "reason": reason,
        "status": "ACTIONABLE" if action is not None else "HOLD",
        "leg": (
            {
                "symbol": "MCL",
                "action": action,
                "quantity": 1,
                "initial_mode": "CROSS" if reduction else "OPTIMISTIC",
                "chase_mode": "RELENTLESS" if reduction else "AUTO",
                "phase_speed_multiplier": (
                    2.0 if not reduction and target_owner == "shock" else 1.0
                ),
                "outside_rth": True,
            }
            if action is not None
            else None
        ),
        "capital_admission": None,
        "order_authority": MCL_LIVE_ORDER_AUTHORITY,
    }
    transition = {
        key: body[key]
        for key in (
            "selection_id",
            "strategy_version",
            "source_checkpoint_id",
            "source_authority",
            "admission_event_id",
            "target_direction",
            "target_route",
            "target_owner",
            "held_direction",
            "reason",
            "leg",
        )
    }
    return {**body, "transition_id": _identity(transition)}


def mcl_transport_order_ref(plan: Mapping[str, object]) -> str:
    transition_id = str(plan.get("transition_id") or "")
    if plan.get("schema") != MCL_LIVE_PLAN_SCHEMA or not _is_sha(transition_id):
        raise ValueError("MCL transport plan has no transition identity")
    return f"{MCL_LIVE_ORDER_REF_PREFIX}-{transition_id[:24]}"


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
    now = _utc(observed_at)
    return ledger.checkpoint(
        evaluation_as_of=now,
        strategy_id=str(plan["strategy_version"]),
        strategy_version=MCL_LIVE_EXECUTION_VERSION,
        trading_date=now.date().isoformat(),
        session="MCL_GTH",
        status="EVALUATED",
        evidence={
            "schema": MCL_LIVE_EXECUTION_SCHEMA,
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
            "order_authority": MCL_LIVE_ORDER_AUTHORITY,
        },
        recorded_at=now,
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


def _admit_mcl_plan(
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
        raise ValueError("MCL capital admission requires one leg")
    reduction = plan.get("held_direction") is not None
    if reduction:
        decision = admit_live_capital(
            capital_plan,
            intent="EXIT",
            account_id=str(broker["account_id"]),
            account_type="CASH",
            currency="USD",
            sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
            run_id=str(selection["selection_id"]),
            selection_file_sha256=selection_file_sha256,
            capital_kind="FUTURES_MARGIN",
            projected_capital_usd=0,
            cash_debit_usd=0,
            available_cash_usd=broker["settled_cash_usd"],
        )
    else:
        commission = max(_commission_values(preview))
        decision = admit_live_capital(
            capital_plan,
            intent="ENTER",
            account_id=str(broker["account_id"]),
            account_type="CASH",
            currency="USD",
            sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
            run_id=str(selection["selection_id"]),
            selection_file_sha256=selection_file_sha256,
            capital_kind="FUTURES_MARGIN",
            projected_capital_usd=commission,
            cash_debit_usd=commission,
            available_cash_usd=broker["settled_cash_usd"],
            resource_state={
                "account_positions": [dict(row) for row in broker["positions"]],
                "account_open_orders": [dict(row) for row in broker["open_orders"]],
                "base_currency": "AUD",
                "available_funds_base_cents": math.floor(
                    float(broker["available_funds_base"]) * 100
                ),
                "excess_liquidity_base_cents": math.floor(
                    float(broker["excess_liquidity_base"]) * 100
                ),
                "usd_to_base_rate_ppm": math.ceil(
                    float(broker["usd_to_base_rate"]) * 1_000_000
                ),
                "candidate_initial_margin_base_cents": math.ceil(
                    float(preview.init_margin_change) * 100
                ),
                "candidate_maintenance_margin_base_cents": math.ceil(
                    float(preview.maintenance_margin_change) * 100
                ),
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


async def execute_mcl_transport_plan(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    plan: Mapping[str, object],
    contract: Contract,
    ticker: object,
    observed_at: datetime,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    leg = plan.get("leg")
    if (
        plan.get("schema") != MCL_LIVE_PLAN_SCHEMA
        or plan.get("selection_id") != selected["selection_id"]
        or plan.get("status") != "ACTIONABLE"
        or not isinstance(leg, Mapping)
        or plan.get("order_authority") != MCL_LIVE_ORDER_AUTHORITY
    ):
        raise ValueError("only one admitted MCL LIMIT leg may execute")
    order_ref = mcl_transport_order_ref(plan)
    latest = _latest_execution_by_ref(
        tuple(ledger.records()), selection_id=str(selected["selection_id"])
    ).get(order_ref)
    price_for_mode = execution_price_for_ticker(contract, ticker)

    def checkpoint(**kwargs: object) -> Mapping[str, object]:
        return _checkpoint(
            ledger,
            selection_id=str(selected["selection_id"]),
            plan=plan,
            **kwargs,
        )

    return await execute_single_contract_limit_order(
        client=client,
        contract=contract,
        ticker=ticker,
        action=str(leg["action"]),
        order_ref=order_ref,
        plan=plan,
        latest_checkpoint=latest,
        observed_at=observed_at,
        max_commission_usd=MCL_LIVE_MAX_COMMISSION_USD,
        initial_mode=str(leg["initial_mode"]),
        chase_mode=str(leg["chase_mode"]),
        phase_speed_multiplier=float(leg.get("phase_speed_multiplier", 1.0)),
        price_for_mode=price_for_mode,
        checkpoint=checkpoint,
        ladder_schema="mcl.execution-ladder-transition.v1",
        source_age_seconds=float(plan["source_age_seconds"]),
    )


async def _mcl_risk_bars(
    client, contract: Contract, *, observed_at: datetime
) -> list[OhlcvBar]:
    raw = await client.historical_bars_ohlcv(
        contract,
        duration_str="1 W",
        bar_size="1 min",
        use_rth=False,
        what_to_show="TRADES",
        cache_ttl_sec=0,
    )
    cutoff = _utc(observed_at).replace(second=0, microsecond=0)
    return list(_bar_map(raw, cutoff=cutoff, name="MCL risk").values())


async def advance_mcl_live_transport(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    capital_plan: Mapping[str, object],
    selection_file_sha256: str,
    observed_at: datetime,
    observe_only: bool = False,
) -> dict[str, object]:
    """Refresh exact V18 state, reconcile, and execute only fresh selected intent."""

    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    broker = await broker_account_snapshot(client, base_currency="AUD")
    if broker["account_id"] != selected["broker_at_selection"]["account_id"]:
        raise ValueError("MCL selected broker account changed")
    source, source_authority = await refresh_mcl_live_source(
        ledger,
        client=client,
        selection=selected,
        observed_at=now,
    )
    _cl_contract, contract = mcl_live_contracts(selected)
    ticker, quote = await _live_quote(client, contract, owner="mcl-live-v18")
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
        raise ValueError("MCL selected run has multiple pending transitions")
    mcl_orders = [
        row
        for row in broker["open_orders"]
        if row.get("symbol") == "MCL"
        or str(row.get("order_ref") or "").startswith(
            f"{MCL_LIVE_ORDER_REF_PREFIX}-"
        )
    ]
    if pending:
        evidence = pending[0]["evidence"]
        assert isinstance(evidence, Mapping)
        order_ref = str(evidence["order_ref"])
        if any(row.get("order_ref") != order_ref for row in mcl_orders):
            raise ValueError("unknown MCL order blocks reconciliation")
        pending_plan = evidence.get("plan")
        if not isinstance(pending_plan, Mapping):
            raise ValueError("pending MCL transition has no durable plan")
        execution = await execute_mcl_transport_plan(
            ledger,
            client=client,
            selection=selected,
            plan=pending_plan,
            contract=contract,
            ticker=ticker,
            observed_at=now,
        )
        return {
            "status": "RECONCILED",
            "selection_id": selected["selection_id"],
            "execution": execution,
            "submitted_orders": execution["submitted_orders"],
        }
    if mcl_orders:
        raise ValueError("unowned MCL order blocks selected-run authority")
    positions = [row for row in broker["positions"] if row.get("symbol") == "MCL"]
    if len(positions) > 1:
        raise ValueError("broker holds multiple MCL contract identities")
    broker_position = float(positions[0]["quantity"]) if positions else 0.0
    if positions and int(positions[0]["con_id"]) != int(
        selected["contracts"]["MCL"]["con_id"]
    ):
        raise ValueError("broker holds an unselected MCL contract")
    liquidation = float(quote["bid"] if broker_position >= 0 else quote["ask"])
    risk_bars = (
        await _mcl_risk_bars(client, contract, observed_at=now)
        if broker_position
        else []
    )
    risk = mcl_transport_risk_state(
        selection=selected,
        records=records,
        observed_at=now,
        liquidation_price=liquidation,
        completed_mcl_bars=risk_bars,
    )
    if abs(float(risk["position_from_fills"]) - broker_position) > 1e-9:
        raise ValueError("MCL broker holding disagrees with selected-run fills")
    plan = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=source_authority,
        broker_position=broker_position,
        risk_state=risk,
        consumed_admissions=_consumed_admissions(
            records, selection_id=str(selected["selection_id"])
        ),
        observed_at=now,
    )
    preview = None
    if plan["status"] == "ACTIONABLE":
        leg = plan["leg"]
        assert isinstance(leg, Mapping)
        price = execution_price_for_ticker(contract, ticker)(
            str(leg["initial_mode"]), str(leg["action"])
        )
        if price is None:
            raise ValueError("MCL action preview has no executable LIMIT price")
        preview = await client.preview_limit_order(
            contract,
            str(leg["action"]),
            1,
            float(price),
            True,
            mcl_transport_order_ref(plan),
        )
        commissions = _commission_values(preview)
        if (
            preview.status != "PreSubmitted"
            or not commissions
            or str(preview.commission_currency or "").upper() != "USD"
            or max(commissions) > MCL_LIVE_MAX_COMMISSION_USD
            or str(preview.warning_text or "")
        ):
            raise ValueError("MCL action preview exceeds selected boundaries")
        plan = _admit_mcl_plan(
            plan,
            preview=preview,
            capital_plan=capital_plan,
            selection=selected,
            selection_file_sha256=selection_file_sha256,
            broker=broker,
        )
    if observe_only and plan["status"] == "ACTIONABLE":
        plan = {
            **plan,
            "status": "OBSERVE_ONLY_HOLD",
            "blocked_leg": plan["leg"],
            "leg": None,
            "reason": "commissioning_restart_proof",
        }
    state = _checkpoint(
        ledger,
        selection_id=str(selected["selection_id"]),
        plan=plan,
        phase="STATE",
        observed_at=now,
        preview=asdict(preview) if preview is not None else None,
        risk_state=risk,
        broker_state=broker,
        quote=quote,
    )
    if plan["status"] != "ACTIONABLE":
        return {
            "status": plan["status"],
            "selection_id": selected["selection_id"],
            "checkpoint_id": state["checkpoint_id"],
            "plan": plan,
            "risk_state": risk,
            "submitted_orders": 0,
        }
    execution = await execute_mcl_transport_plan(
        ledger,
        client=client,
        selection=selected,
        plan=plan,
        contract=contract,
        ticker=ticker,
        observed_at=now,
    )
    return {
        "status": "EXECUTED",
        "selection_id": selected["selection_id"],
        "checkpoint_id": state["checkpoint_id"],
        "plan": plan,
        "risk_state": risk,
        "execution": execution,
        "submitted_orders": execution["submitted_orders"],
    }
