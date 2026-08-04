from __future__ import annotations

import asyncio
import hashlib
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
from tradebot.research.mcl_profitability import (
    mcl_live_evaluation_slots,
    mcl_live_profitability_receipt,
    mcl_market_open,
    mcl_runtime_parity_graduation_gate,
    normalize_mcl_risk,
)
from tradebot.research.mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION


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
        "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
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
        "source": {
            "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
            "submitted_orders": 0,
        },
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
    assert selected["strategy_version"] == MCL_TWO_SPEED_SHOCK_VERSION
    assert selected["execution"]["order_type"] == "LMT"
    assert selected["execution"]["market_orders_allowed"] is False
    assert selected["risk"]["raw_loss_cap_usd"] == 300.0
    assert selected["risk"]["package_stressed_loss_usd"] == 305.52
    assert selected["allocation_successor"]["package_id"] == (
        "mcl-one-contract-stage91"
    )
    assert selected["allocation_successor"]["initial_margin_base_cents"] == 268_670
    assert selected["allocation_successor"]["maintenance_margin_base_cents"] == 214_936

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
            strategy_version=selected["strategy_version"],
        )
    )

    assert source["rows"]["common"] == 620
    assert source["latest_common_close_utc"] == now.isoformat()
    assert source["target"] is None
    assert source["synthetic_midcycle_entry_authority"] == "none"


def _source_checkpoint(
    selected, *, event_at: datetime, event_id: str, owner: str = "v18"
):
    target = {
        "event_id": event_id,
        "observed_at_utc": event_at.isoformat(),
        "signal_at_utc": event_at.isoformat(),
        "direction": 1,
        "route": "failed_auction",
        "owner": owner,
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
        "phase_speed_multiplier": 1.0,
        "outside_rth": True,
    }
    assert consumed["reason"] == "admission_already_consumed"


def test_mcl_shock_entry_uses_accelerated_limit_ladder_and_friday_lock() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(minutes=5)
    source = _source_checkpoint(
        selected, event_at=event_at, event_id="5" * 64, owner="shock"
    )
    source["evidence"]["target"]["route"] = "shock_continuation"
    risk = {"safety_breaches": []}
    active = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    locked = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=datetime(2026, 8, 7, 20, 54, tzinfo=timezone.utc),
    )

    assert active["reason"] == "fresh_source_admission"
    assert active["leg"]["phase_speed_multiplier"] == 2.0
    assert active["leg"]["chase_mode"] == "AUTO"
    assert locked["reason"] == "weekly_closure_entry_lock"
    assert locked["leg"] is None


def test_mcl_same_direction_cannot_inherit_a_new_admission_across_restart() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(
        minutes=5
    )
    source = _source_checkpoint(
        selected, event_at=event_at, event_id="6" * 64, owner="shock"
    )
    source["evidence"]["target"]["route"] = "shock_continuation"
    retained = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=1,
        risk_state={"safety_breaches": [], "admission_event_id": "6" * 64},
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    replaced = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        broker_position=1,
        risk_state={"safety_breaches": [], "admission_event_id": "7" * 64},
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )

    assert retained["reason"] == "target_already_owned"
    assert retained["leg"] is None
    assert replaced["reason"] == "source_admission_identity_changed"
    assert replaced["leg"]["action"] == "SELL"
    assert replaced["leg"]["chase_mode"] == "RELENTLESS"


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
    assert risk["owner"] == "v18"
    assert risk["admission_event_id"] == event_id
    assert risk["mfe_usd"] == 50.0
    assert risk["profit_memory_stop"] == 80.125
    assert risk["exit_triggers"] == ["failed_auction_profit_memory"]
    assert set(risk["safety_breaches"]) == {
        "raw_loss_cap",
        "run_drawdown_limit_breached",
    }
    assert risk["run_gross_usd"] == -310.0
    assert risk["run_cost_usd"] == 1.52
    assert risk["run_net_usd"] == -311.52
    assert risk["drawdown_usd"] == 311.52
    assert risk["fill_count"] == 1
    assert risk["attribution_complete"] is True


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
    assert binding.strategy_id == MCL_TWO_SPEED_SHOCK_VERSION
    assert binding.execution_strategy_version == MCL_LIVE_EXECUTION_VERSION
    assert binding.timer_unit == "tradebot-mcl-live.timer"
    assert binding.service_unit == "tradebot-mcl-live.service"
    assert binding.champion_track == "HF"


def _profitability_risk(
    *, net: float = 0.0, fills: int = 0, trades: int = 0
) -> dict[str, object]:
    return {
        "valid": True,
        "attribution_complete": True,
        "position_from_fills": 0.0,
        "run_realized_gross_usd": net,
        "run_realized_cost_usd": 0.0,
        "run_realized_net_usd": net,
        "open_mark_gross_usd": 0.0,
        "open_mark_cost_usd": 0.0,
        "open_mark_net_usd": 0.0,
        "run_gross_usd": net,
        "run_cost_usd": 0.0,
        "run_net_usd": net,
        "peak_run_net_usd": max(0.0, net),
        "drawdown_usd": max(0.0, -net),
        "closed_trades": trades,
        "gross_wins_usd": max(0.0, net),
        "top_five_gross_wins_usd": max(0.0, net),
        "fill_count": fills,
        "exit_triggers": [],
        "safety_breaches": [],
    }


def _profitability_state(
    selected: dict[str, object],
    evaluated: datetime,
    *,
    net: float = 0.0,
    fills: int = 0,
    trades: int = 0,
) -> dict[str, object]:
    recorded = evaluated + timedelta(seconds=10)
    checkpoint = hashlib.sha256(evaluated.isoformat().encode()).hexdigest()
    return {
        "kind": "checkpoint",
        "checkpoint_id": checkpoint,
        "recorded_at_utc": recorded.isoformat(),
        "evaluation_as_of_utc": recorded.isoformat(),
        "strategy_id": selected["strategy_version"],
        "strategy_version": MCL_LIVE_EXECUTION_VERSION,
        "status": "EVALUATED",
        "evidence": {
            "selection_id": selected["selection_id"],
            "phase": "STATE",
            "submitted_orders": 0,
            "plan": {"held_direction": None, "leg": None},
            "broker_state": {"positions": [], "open_orders": []},
            "risk_state": _profitability_risk(
                net=net, fills=fills, trades=trades
            ),
        },
    }


def test_mcl_clock_owns_gth_and_excludes_daily_and_weekend_closures() -> None:
    assert mcl_market_open("2026-08-03T20:59:00+00:00")
    assert not mcl_market_open("2026-08-03T21:00:00+00:00")
    assert not mcl_market_open("2026-08-02T21:59:00+00:00")
    assert mcl_market_open("2026-08-02T22:00:00+00:00")
    slots = mcl_live_evaluation_slots(
        "2026-08-03T20:58:00+00:00", "2026-08-03T22:01:00+00:00"
    )
    assert [row.isoformat() for row in slots] == [
        "2026-08-03T20:59:00+00:00",
        "2026-08-03T22:00:00+00:00",
        "2026-08-03T22:01:00+00:00",
    ]


def test_mcl_legacy_zero_prefix_normalizes_but_nonzero_history_cannot() -> None:
    legacy = {
        "schema": "mcl.two-speed-auction-risk-state.v1",
        "position_from_fills": 0,
        "open_exec_id": None,
        "entry_time_utc": None,
        "entry_price": None,
        "run_realized_net_usd": 0.0,
        "closed_trades": 0,
        "unrealized_raw_usd": 0.0,
        "fill_ledger_fingerprint": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "safety_breaches": [],
    }
    normalized = normalize_mcl_risk(legacy)
    assert normalized["valid"] is True
    assert normalized["run_net_usd"] == 0.0
    assert normalized["fill_count"] == 0
    with pytest.raises(ValueError, match="cannot be normalized"):
        normalize_mcl_risk({**legacy, "position_from_fills": 1})


def test_mcl_positive_24h_requires_complete_minutes_and_authentic_fill() -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    end = baseline + timedelta(hours=24)
    rows = [_profitability_state(selected, baseline)]
    rows.extend(
        _profitability_state(selected, slot, net=4.0, fills=2, trades=1)
        for slot in mcl_live_evaluation_slots(baseline, end)
    )
    receipt = mcl_live_profitability_receipt(
        rows,
        selection=selected,
        as_of=end + timedelta(seconds=55),
    )
    assert receipt["status"] == "ACTIVE"
    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["milestones"]["48h"]["passed"] is False
    assert receipt["economics"]["net_usd"] == 4.0
    assert receipt["economics"]["fills"] == 2


def test_mcl_same_minute_receipt_noise_collapses_by_economic_state() -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    first = _profitability_state(selected, baseline + timedelta(minutes=1))
    duplicate = deepcopy(first)
    duplicate["checkpoint_id"] = "d" * 64
    duplicate["recorded_at_utc"] = (baseline + timedelta(minutes=1, seconds=30)).isoformat()
    duplicate["evaluation_as_of_utc"] = duplicate["recorded_at_utc"]
    duplicate["evidence"]["source_checkpoint_id"] = "e" * 64
    duplicate["evidence"]["quote"] = {"bid": 75.81, "ask": 75.82}
    duplicate["evidence"]["broker_state"]["observed_at_utc"] = duplicate[
        "recorded_at_utc"
    ]
    duplicate["evidence"]["risk_state"].update(
        {
            "as_of_utc": duplicate["recorded_at_utc"],
            "liquidation_price": 75.81,
            "observed_at_utc": duplicate["recorded_at_utc"],
        }
    )

    receipt = mcl_live_profitability_receipt(
        [
            _profitability_state(selected, baseline),
            first,
            duplicate,
        ],
        selection=selected,
        as_of=baseline + timedelta(minutes=2),
    )

    assert receipt["status"] == "ACTIVE"
    assert receipt["reasons"] == []
    assert receipt["clock"]["evaluated_slots"] == 1


@pytest.mark.parametrize(
    "change",
    ("fill", "position", "safety", "order"),
)
def test_mcl_same_minute_material_state_change_remains_a_conflict(
    change: str,
) -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    first = _profitability_state(selected, baseline + timedelta(minutes=1))
    changed = deepcopy(first)
    changed["checkpoint_id"] = "f" * 64
    changed["recorded_at_utc"] = (baseline + timedelta(minutes=1, seconds=30)).isoformat()
    changed["evaluation_as_of_utc"] = changed["recorded_at_utc"]
    if change == "fill":
        changed["evidence"]["risk_state"]["fill_count"] = 1
    elif change == "position":
        changed["evidence"]["broker_state"]["positions"] = [
            {"symbol": "MCL", "con_id": selected["contracts"]["MCL"]["con_id"], "quantity": 1.0}
        ]
    elif change == "safety":
        changed["evidence"]["risk_state"]["safety_breaches"] = [
            "raw_loss_cap"
        ]
    else:
        changed["evidence"]["broker_state"]["open_orders"] = [
            {
                "symbol": "MCL",
                "con_id": selected["contracts"]["MCL"]["con_id"],
                "order_ref": "MCLV18-test",
                "status": "Submitted",
            }
        ]

    receipt = mcl_live_profitability_receipt(
        [
            _profitability_state(selected, baseline),
            first,
            changed,
        ],
        selection=selected,
        as_of=baseline + timedelta(minutes=2),
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert "conflicting_session_coverage" in receipt["reasons"]


def test_mcl_runtime_gate_rehashes_the_selected_stage112_owners() -> None:
    passed = mcl_runtime_parity_graduation_gate(
        selection=_selection(), repo_root=ROOT
    )
    assert passed["status"] == "PASS"
    changed = _selection()
    changed["evidence"]["lifecycle_parity"]["sha256"] = "0" * 64
    rejected = mcl_runtime_parity_graduation_gate(
        selection=changed, repo_root=ROOT
    )
    assert rejected["status"] == "INVALID"


def test_mcl_cli_graduation_never_loads_broker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradebot.research import mcl_live_cli

    selected = _selection()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selected), encoding="utf-8")
    output = tmp_path / "graduation.json"
    monkeypatch.setattr(mcl_live_cli, "load_live_capital_plan", lambda _path: {})
    monkeypatch.setattr(
        mcl_live_cli,
        "load_allocated_live_selection",
        lambda *_args, **_kwargs: (selected, selection_path, "0" * 64),
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "LiveCalibrationLedger",
        lambda _path: type("Ledger", (), {"records": lambda self: ()})(),
    )
    monkeypatch.setattr(
        mcl_live_cli, "live_calibration_logical_prefix", lambda *_args, **_kwargs: ({}, ())
    )
    monkeypatch.setattr(
        mcl_live_cli, "mcl_live_profitability_receipt", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        mcl_live_cli, "mcl_live_graduation_inputs", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "reduce_live_graduation",
        lambda **_kwargs: {"verdict": "HOLD"},
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "publish_live_graduation_receipt",
        lambda path, receipt: path.write_text(json.dumps(receipt)),
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "load_config",
        lambda: pytest.fail("graduation queried broker configuration"),
    )
    code = asyncio.run(
        mcl_live_cli._main_async(
            [
                "--capital-plan",
                str(tmp_path / "capital.json"),
                "--graduation-target",
                "24h",
                "--graduation-cutoff",
                "2026-08-04T10:00:00+00:00",
                "--graduation-output",
                str(output),
            ]
        )
    )
    assert code == 0
    assert json.loads(output.read_text()) == {"verdict": "HOLD"}
