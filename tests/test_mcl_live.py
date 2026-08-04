from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.chart_data.series import OhlcvBar
from tradebot.research.mcl_live import (
    mcl_transport_risk_state,
    project_mcl_transport_plan,
)
from tradebot.research.mcl_live_transport import (
    MCL_LIVE_EXECUTION_VERSION,
    MCL_LIVE_SOURCE_SCHEMA,
    MCL_LIVE_SOURCE_VERSION,
    build_mcl_live_selection,
    load_mcl_live_selection_from_mapping,
    mcl_source_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
AT = datetime(2026, 8, 4, 8, 31, 21, tzinfo=timezone.utc)


def _preview() -> dict[str, object]:
    old = json.loads(
        (ROOT / "backtests/mcl/mcl_v18_live_commissioning_preview_post_funding.json").read_text()
    )
    generation = json.loads(
        (ROOT / "backtests/mcl/mcl_turn_authenticity_microstructure_generation.json").read_text()
    )
    broker = old["broker"]
    return {
        "schema": "mcl.v18-live-commissioning-preview.v2",
        "observed_at_utc": AT.isoformat(),
        "authority": "fresh_nontransmitting_what_if_only",
        "broker": {
            "observed_at_utc": broker["observed_at_utc"],
            "account_id": broker["account_id"],
            "account_type": "CASH",
            "base_currency": "AUD",
            "settled_cash_usd": broker["settled_cash_usd"],
            "equity_with_loan_base": broker["equity_with_loan_aud"],
            "available_funds_base": broker["available_funds_aud"],
            "excess_liquidity_base": broker["excess_liquidity_aud"],
            "initial_margin_base": broker["initial_margin_aud"],
            "maintenance_margin_base": broker["maintenance_margin_aud"],
            "gross_position_value_base": broker["gross_position_value_aud"],
            "usd_to_base_rate": broker["usd_to_aud"],
            "positions": broker["positions"],
            "open_orders": [],
        },
        "contracts": generation["contracts"],
        "quote": old["quote"],
        "source": {"submitted_orders": 0},
        "what_if": old["what_if"],
        "submitted_orders": 0,
    }


def _selection():
    return build_mcl_live_selection(
        repository_root=ROOT,
        preview=_preview(),
        selected_at=AT + timedelta(seconds=1),
    )


def test_mcl_selection_binds_flat_limit_only_stage91_canary() -> None:
    selected = _selection()

    assert load_mcl_live_selection_from_mapping(selected) == selected
    assert selected["baseline"]["position"] == 0
    assert selected["baseline"]["inherited_target_authority"] == "none"
    assert selected["execution"]["order_type"] == "LMT"
    assert selected["execution"]["market_orders_allowed"] is False
    assert selected["risk"]["raw_loss_cap_usd"] == 300.0
    assert selected["risk"]["package_stressed_loss_usd"] == 305.52
    assert selected["allocation_successor"]["package_id"] == (
        "mcl-one-contract-stage91"
    )

    mutated = deepcopy(selected)
    mutated["execution"]["market_orders_allowed"] = True
    with pytest.raises(ValueError, match="selected-run contract is invalid"):
        load_mcl_live_selection_from_mapping(mutated)


def test_mcl_source_uses_completed_et_minutes_and_never_adopts_history() -> None:
    now = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)
    first = (now - timedelta(minutes=620)).astimezone(
        timezone(timedelta(hours=-4))
    ).replace(tzinfo=None)
    rows = []
    for index in range(620):
        ts = first + timedelta(minutes=index)
        close = 80 + index * 0.001
        rows.append(OhlcvBar(ts, close, close + 0.01, close - 0.01, close, 10))

    class Client:
        async def historical_bars_ohlcv(self, contract, **_kwargs):
            bump = 0.0 if contract.symbol == "CL" else 0.01
            return [
                OhlcvBar(
                    row.ts,
                    row.open + bump,
                    row.high + bump,
                    row.low + bump,
                    row.close + bump,
                    row.volume,
                )
                for row in rows
            ]

    selected = _selection()
    from tradebot.research.mcl_live_transport import mcl_live_contracts

    cl, mcl = mcl_live_contracts(selected)
    source = asyncio.run(
        mcl_source_snapshot(
            Client(),
            cl_contract=cl,
            mcl_contract=mcl,
            observed_at=now,
            selected_at=now,
        )
    )

    assert source["rows"]["common"] == 620
    assert source["latest_common_close_utc"] == now.isoformat()
    assert source["target"] is None
    assert source["synthetic_midcycle_entry_authority"] == "none"


def _source_checkpoint(selected, *, event_at: datetime, event_id: str):
    target = {
        "event_id": event_id,
        "observed_at_utc": event_at.isoformat(),
        "signal_at_utc": event_at.isoformat(),
        "direction": 1,
        "route": "failed_auction",
        "decision": {},
    }
    return {
        "checkpoint_id": "2" * 64,
        "strategy_version": MCL_LIVE_SOURCE_VERSION,
        "recorded_at_utc": (event_at + timedelta(seconds=5)).isoformat(),
        "status": "EVALUATED",
        "evidence": {
            "schema": MCL_LIVE_SOURCE_SCHEMA,
            "selection_id": selected["selection_id"],
            "target": target,
            "source": {
                "latest_common_close_utc": event_at.isoformat(),
            },
        },
    }


def test_mcl_plan_waits_for_next_minute_and_consumes_each_admission_once() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(minutes=5)
    event_id = "3" * 64
    source = _source_checkpoint(selected, event_at=event_at, event_id=event_id)
    risk = {"safety_breaches": []}

    early = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(seconds=30),
    )
    fresh = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    consumed = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=0,
        risk_state=risk,
        consumed_admissions={event_id},
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )

    assert early["reason"] == "next_minute_entry_not_due"
    assert fresh["status"] == "ACTIONABLE"
    assert fresh["leg"] == {
        "symbol": "MCL",
        "action": "BUY",
        "quantity": 1,
        "initial_mode": "OPTIMISTIC",
        "chase_mode": "AUTO",
        "outside_rth": True,
    }
    assert consumed["reason"] == "admission_already_consumed"


def test_mcl_actual_fill_risk_uses_failed_auction_memory_and_raw_cap() -> None:
    selected = _selection()
    event_id = "4" * 64
    entry_at = AT + timedelta(minutes=10)
    plan = {
        "reason": "fresh_v18_admission",
        "target_route": "failed_auction",
        "admission_event_id": event_id,
    }
    record = {
        "strategy_version": MCL_LIVE_EXECUTION_VERSION,
        "evidence": {
            "selection_id": selected["selection_id"],
            "phase": "TERMINAL",
            "order_ref": "MCLV18-test",
            "plan": plan,
            "broker_order": {
                "con_id": selected["contracts"]["MCL"]["con_id"],
                "fills": [
                    {
                        "exec_id": "entry",
                        "time_utc": entry_at.isoformat(),
                        "side": "BOT",
                        "shares": 1,
                        "price": 80.0,
                        "commission": 0.76,
                        "commission_currency": "USD",
                    }
                ],
            },
        },
    }
    bars = [
        OhlcvBar(
            entry_at + timedelta(minutes=1),
            80.0,
            80.5,
            79.9,
            80.4,
            10,
        ),
        OhlcvBar(
            entry_at + timedelta(minutes=2),
            80.4,
            80.45,
            80.1,
            80.2,
            10,
        ),
    ]
    risk = mcl_transport_risk_state(
        selection=selected,
        records=[record],
        observed_at=entry_at + timedelta(minutes=3),
        liquidation_price=76.9,
        completed_mcl_bars=bars,
    )

    assert risk["position_from_fills"] == 1
    assert risk["mfe_usd"] == 50.0
    assert risk["profit_memory_stop"] == 80.125
    assert set(risk["safety_breaches"]) == {
        "failed_auction_profit_memory",
        "raw_loss_cap",
    }


def test_mcl_live_worker_is_shared_locked_limit_only_and_maintenance_aware() -> None:
    service = (ROOT / "deploy/systemd/tradebot-mcl-live.service").read_text()
    timer = (ROOT / "deploy/systemd/tradebot-mcl-live.timer").read_text()
    runtime = (ROOT / "tradebot/research/mcl_live.py").read_text()

    assert "%t/tradebot-live-account.lock" in service
    assert "/usr/bin/flock --exclusive --wait 180" in service
    assert "Environment=IBKR_READONLY=0" in service
    assert "tradebot.research.mcl_live_cli" in service
    assert "Restart=on-failure" in service
    assert "Sun *-*-* 18..23:*:10 America/New_York" in timer
    assert "Mon..Thu *-*-* 00..16:*:10 America/New_York" in timer
    assert "Mon..Thu *-*-* 18..23:*:10 America/New_York" in timer
    assert "Fri *-*-* 00..16:*:10 America/New_York" in timer
    assert "MarketOrder" not in runtime
    assert "place_market" not in runtime


def test_mcl_live_binding_uses_the_one_durable_worker() -> None:
    from tradebot.live.strategies import LIVE_STRATEGY_BINDINGS

    binding = next(
        item for item in LIVE_STRATEGY_BINDINGS if item.champion_symbol == "MCL"
    )
    assert binding.strategy_id == "mcl.two-speed-auction-relay.v18"
    assert binding.execution_strategy_version == MCL_LIVE_EXECUTION_VERSION
    assert binding.timer_unit == "tradebot-mcl-live.timer"
    assert binding.service_unit == "tradebot-mcl-live.service"
    assert binding.champion_track == "HF"
