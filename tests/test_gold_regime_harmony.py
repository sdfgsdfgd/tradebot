from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradebot.backtest.models import BacktestResult, SpotTrade, SummaryStats
from tradebot.client import BrokerOrderPreview
from tradebot.live.capital import build_live_capital_plan
from tradebot.research import gold_onset_cli
from tradebot.research.gold_live_runtime import (
    _gold_price_for_ticker,
    advance_gold_live_transport,
    execute_gold_transport_plan,
    gold_broker_snapshot,
    gold_live_contract,
    gold_transport_order_ref,
)
from tradebot.research.gold_live_cli import _require_current_gold_runtime_parity
from tradebot.research.gold_onset_cli import (
    _gold_rollover_boundary,
    _unmanaged_stress,
)
from tradebot.research.gold_live_state import (
    gold_transport_risk_state,
    project_gold_transport_plan,
)
from tradebot.research.gold_live_transport import (
    GOLD_LIVE_EXECUTION_VERSION,
    GOLD_LIVE_PACKAGE_SELECTION_SCHEMA,
    GOLD_LIVE_SELECTION_SCHEMA,
    GOLD_RUNTIME_PARITY_PATH,
    build_gold_portfolio_capital_plan,
    load_gold_live_selection_from_mapping,
    publish_gold_live_selection,
    reallocate_gold_live_transport,
    select_gold_live_transport,
    advance_gold_regime_harmony_source,
)
from tradebot.research.gold_profitability import gold_live_profitability_receipt
from tradebot.research.gold_regime_harmony import (
    GOLD_REGIME_HARMONY_FULL10_LEDGER,
    GOLD_REGIME_HARMONY_FULL3_LEDGER,
    GoldHardRegimeOwner,
    GoldRegimeHarmonyOwner,
    gold_regime_harmony_config,
    load_gold_regime_harmony_crown,
)
from tradebot.research.live_calibration import LiveCalibrationLedger


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary() -> SummaryStats:
    return SummaryStats(
        trades=0,
        wins=0,
        losses=0,
        win_rate=0.0,
        total_pnl=0.0,
        roi=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        avg_hold_hours=0.0,
    )


def _daily_context() -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "end": start + timedelta(days=index),
            "hard_direction": "up",
            "hard_age": index + 1,
        }
        for index in range(8)
    ]


def test_gold_stage76_config_preserves_frozen_signal_and_cost_law() -> None:
    strategy = gold_regime_harmony_config(
        date(2023, 7, 1), date(2026, 6, 30)
    ).strategy

    assert strategy.ema_preset == "8/21"
    assert strategy.ema_entry_mode == "cross"
    assert strategy.entry_confirm_bars == 1
    assert strategy.regime2_ema_preset == "21/50"
    assert strategy.regime2_bar_size == "1 day"
    assert strategy.exit_on_signal_flip is True
    assert strategy.spot_controlled_flip is True
    assert strategy.spot_close_eod is False
    assert strategy.spot_profit_target_pct is None
    assert strategy.spot_stop_loss_pct is None
    assert strategy.spot_spread == 0.50
    assert strategy.spot_commission_per_share == 0.66
    assert strategy.spot_slippage_per_share == 0.25
    assert strategy.spot_min_qty == strategy.spot_max_qty == 1


def test_gold_owner_reconstructs_target_but_cannot_synthesize_entry() -> None:
    owner = GoldRegimeHarmonyOwner(
        _daily_context(),
        (),
        (),
        predecessors={},
        source_records={},
    )
    trade = SpotTrade(
        symbol="XAUUSD",
        qty=1,
        entry_time=datetime(2026, 1, 8),
        entry_price=4100.0,
    )
    state = owner.state_payload(BacktestResult([trade], [], _summary()))

    assert state["target_direction"] == "up"
    assert state["synthetic_midcycle_entry_authority"] == "none"
    assert state["order_authority"] == "none"
    assert state["submitted_orders"] == 0


def test_gold_live_source_consumes_canonical_exchange_parity(
    monkeypatch, tmp_path: Path
) -> None:
    class Replay:
        def __init__(self, _tape: object) -> None:
            pass

        def converged_window(self, *_args, **_kwargs):
            return (
                SimpleNamespace(trades=[]),
                SimpleNamespace(
                    state_payload=lambda _result: {
                        "state_sha256": "f" * 64,
                        "target_direction": None,
                    }
                ),
                {"stable": True},
                True,
            )

    monkeypatch.setattr(
        "tradebot.research.gold_live_transport.GoldRegimeHarmonyReplay",
        Replay,
    )
    now = datetime(2026, 8, 3, 10, tzinfo=UTC)
    decision_bar = now - timedelta(hours=2)
    tape = SimpleNamespace(
        as_of=now,
        h1=(),
        h4=(),
        daily=(),
        uup=(),
        tip=(),
    )
    pair = {"usable": True, "contract_month": "2026-12"}
    output = advance_gold_regime_harmony_source(
        LiveCalibrationLedger(tmp_path / "source.jsonl"),
        tape=tape,
        onset_context={
            "source_usable": True,
            "exchange_parity": pair,
            "signal": {
                "usable": True,
                "decision_bar_end_utc": decision_bar.isoformat(),
            },
            "macro": {"usable": True},
            "news": {"authority": "attribution_only"},
            "source_points": {},
        },
        observed_at=now,
    )

    assert output["checkpoint"]["status"] == "EVALUATED"
    assert output["checkpoint"]["evidence"]["contract_pair"] == pair
    assert (
        output["checkpoint"]["evidence"]["decision_bar_end_utc"]
        == decision_bar.isoformat()
    )

    rejected = advance_gold_regime_harmony_source(
        LiveCalibrationLedger(tmp_path / "rejected-source.jsonl"),
        tape=tape,
        onset_context={
            "source_usable": False,
            "exchange_parity": pair,
            "signal": {
                "usable": True,
                "decision_bar_end_utc": decision_bar.isoformat(),
            },
            "macro": {"usable": True},
            "news": {"authority": "attribution_only"},
            "source_points": {},
            "timing_parity": {"usable": False},
        },
        observed_at=now,
    )
    assert rejected["checkpoint"]["status"] == "NO_DATA"
    assert rejected["target"] is None


def test_gold_hard_state_identity_uses_completed_state_birth() -> None:
    owner = GoldHardRegimeOwner(_daily_context())

    assert owner.state_id(datetime(2026, 1, 8), "up") == (
        "up:2026-01-01T00:00:00+00:00"
    )


def test_gold_runtime_parity_receipt_binds_crown_owner_and_holds_live() -> None:
    path = ROOT / GOLD_RUNTIME_PARITY_PATH
    receipt = json.loads(path.read_text())
    crown = load_gold_regime_harmony_crown(root=ROOT)

    assert receipt["crown"]["artifact_sha256"] == crown["artifact_sha256"]
    assert receipt["historical_parity"]["full_three_year"]["ledger_sha256"] == (
        GOLD_REGIME_HARMONY_FULL3_LEDGER
    )
    assert receipt["historical_parity"]["full_ten_year"]["ledger_sha256"] == (
        GOLD_REGIME_HARMONY_FULL10_LEDGER
    )
    for owner in receipt["owners"].values():
        assert _sha256(ROOT / owner["path"]) == owner["sha256"]
    assert receipt["prospective_prefix"]["cold_replay_equal"] is True
    assert receipt["prospective_prefix"]["target_direction"] is None
    assert receipt["prospective_prefix"]["order_authority"] == "none"
    assert receipt["gates"]["native_1oz_margin_and_cash"] == "HOLD"
    assert receipt["gates"]["live_24h"] == "NOT_STARTED"
    assert receipt["verdict"] == "SIGNAL_RUNTIME_PARITY_PASS_LIVE_TRANSPORT_HOLD"


def test_gold_live_runtime_rejects_the_pre_fail_closed_selection() -> None:
    selected = _selection(datetime(2026, 8, 8, 12, tzinfo=UTC))

    assert _require_current_gold_runtime_parity(
        selected, repository_root=ROOT
    ) == selected["evidence"]["runtime_parity"]

    predecessor = json.loads(json.dumps(selected))
    predecessor["evidence"]["runtime_parity"] = {
        "path": (
            "backtests/gold/"
            "one_oz_regime_harmony_runtime_parity_fail_closed_20260808.json"
        ),
        "sha256": (
            "9d9a6ebda0c7c1b93fb89805a588ee60e617622db76855f1ea2783125bb9bdb3"
        ),
    }
    with pytest.raises(
        ValueError, match="requires current fail-closed parity binding"
    ):
        _require_current_gold_runtime_parity(
            predecessor, repository_root=ROOT
        )


def test_gold_signal_owner_has_no_broker_or_live_order_dependency() -> None:
    path = ROOT / "tradebot/research/gold_regime_harmony.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not any(
        value.startswith(("ib_insync", "tradebot.client", "tradebot.live"))
        for value in imports
    )
    assert not calls.intersection(
        {"placeOrder", "place_limit_order", "submit_order", "cancelOrder"}
    )


def _source(
    recorded_at: datetime,
    *,
    target: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": "live_calibration.v1",
        "kind": "checkpoint",
        "checkpoint_id": "e" * 64,
        "recorded_at_utc": recorded_at.isoformat(),
        "evaluation_as_of_utc": recorded_at.isoformat(),
        "strategy_id": "gold.1oz-regime-harmony-stage76.v1",
        "strategy_version": "gold.1oz-regime-harmony-source.v1",
        "trading_date": recorded_at.date().isoformat(),
        "session": "GOLD_24X5_SOURCE",
        "status": "EVALUATED",
        "evidence": {
            "schema": "gold.1oz-regime-harmony-source-checkpoint.v1",
            "target": target,
            "owner_state": {"state_sha256": "f" * 64},
            "synthetic_midcycle_entry_authority": "none",
            "signal_context": {"proposed_direction": None},
            "macro_context": {"usable": True},
            "fundamental_pressure": {"authority": "attribution_only"},
            "contract_pair": {"usable": True},
            "source_usable": True,
        },
    }


def _preview(observed_at: datetime) -> dict[str, object]:
    pair = {
        "usable": True,
        "contract_month": "2026-12",
        "gc": {
            "local_symbol": "GCZ6",
            "con_id": 111,
            "expiry": "20261229",
            "contract_month": "2026-12",
            "market_data_type": 1,
            "age_seconds": 0.1,
            "bid": 4109.2,
            "bid_size": 1,
            "ask": 4109.5,
            "ask_size": 1,
        },
        "one_oz": {
            "local_symbol": "1OZZ6",
            "con_id": 222,
            "expiry": "20261125",
            "contract_month": "2026-12",
            "market_data_type": 1,
            "age_seconds": 0.1,
            "bid": 4109.25,
            "bid_size": 1,
            "ask": 4109.5,
            "ask_size": 1,
        },
    }
    what_if = [
        {
            "action": action,
            "quantity": 1,
            "limit_price": price,
            "status": "PreSubmitted",
            "initial_margin_change_aud": 560.0,
            "initial_margin_after_aud": 590.0,
            "maintenance_margin_change_aud": 487.0,
            "maintenance_margin_after_aud": 516.0,
            "equity_with_loan_after_aud": 2106.0,
            "commission_usd": 0.66,
            "commission_currency": "USD",
            "warning_text": "",
        }
        for action, price in (("BUY", 4109.5), ("SELL", 4109.25))
    ]
    return {
        "schema": "gold.1oz-selection-preview.v1",
        "authority": "fresh_nontransmitting_what_if_only",
        "observed_at_utc": observed_at.isoformat(),
        "account_id": "U123",
        "account_type": "CASH",
        "base_currency": "AUD",
        "account_values": {
            "settled_cash_usd": 1318.05,
            "equity_with_loan_aud": 2106.0,
            "available_funds_aud": 2073.0,
            "excess_liquidity_aud": 2078.0,
            "initial_margin_aud": 33.0,
            "maintenance_margin_aud": 29.0,
            "gross_position_value_aud": 94.0,
            "usd_to_aud": 1.427,
        },
        "pair": pair,
        "contract": {
            "symbol": "1OZ",
            "local_symbol": "1OZZ6",
            "con_id": 222,
            "expiry": "20261125",
            "exchange": "COMEX",
            "currency": "USD",
            "multiplier": "1",
            "min_tick": 0.25,
        },
        "positions": [{"symbol": "TQQQ", "quantity": 1}],
        "open_orders": [],
        "what_if": what_if,
        "submitted_orders": 0,
    }


def _selection(selected_at: datetime) -> dict[str, object]:
    return select_gold_live_transport(
        source_checkpoint=_source(selected_at - timedelta(seconds=10)),
        preview=_preview(selected_at - timedelta(seconds=5)),
        selected_at_utc=selected_at,
        root=ROOT,
    )


def test_gold_canary_selection_is_flat_fresh_and_content_addressed() -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)

    assert selected["schema"] == GOLD_LIVE_SELECTION_SCHEMA
    assert selected["baseline_target"] is None
    assert selected["quantity"] == 1
    assert selected["execution"]["transport_hours"].startswith("1OZ_24x7")
    assert selected["execution"]["signal_hours"] == "XAUUSD_GC_24x5"
    assert selected["profitability_clock_started"] is True
    assert load_gold_live_selection_from_mapping(selected) == selected

    stale = _preview(selected_at - timedelta(minutes=2))
    with pytest.raises(ValueError, match="fresh flat source"):
        select_gold_live_transport(
            source_checkpoint=_source(selected_at - timedelta(seconds=10)),
            preview=stale,
            selected_at_utc=selected_at,
            root=ROOT,
        )


def test_gold_package_successor_removes_only_the_account_mutex() -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    predecessor = _selection(selected_at)
    successor_at = selected_at + timedelta(minutes=5)

    selected = reallocate_gold_live_transport(
        predecessor=predecessor,
        records=(),
        source_checkpoint=_source(successor_at - timedelta(seconds=10)),
        preview=_preview(successor_at - timedelta(seconds=5)),
        selected_at_utc=successor_at,
        stress_receipt_path=ROOT
        / "backtests/gold/one_oz_stage76_open_position_stress_20260803.json",
        root=ROOT,
    )

    assert selected["schema"] == GOLD_LIVE_PACKAGE_SELECTION_SCHEMA
    assert selected["quantity"] == 1
    assert "max_concurrent_directional_sleeves" not in selected["risk"]
    assert selected["risk"]["max_open_position_stress_usd"] == 256.16
    assert selected["risk"]["max_run_drawdown_usd"] == 700.0
    assert selected["allocation_successor"]["predecessor_selection_id"] == (
        predecessor["selection_id"]
    )
    assert selected["evidence"]["source_recorded_at_utc"] == (
        successor_at - timedelta(seconds=10)
    ).isoformat()
    assert selected["evidence"]["runtime_parity"]["path"] == (
        GOLD_RUNTIME_PARITY_PATH.as_posix()
    )
    assert load_gold_live_selection_from_mapping(selected) == selected

    selected["risk"]["max_open_position_stress_usd"] = 255.0
    with pytest.raises(ValueError, match="invalid"):
        load_gold_live_selection_from_mapping(selected)


def test_gold_rollover_stresses_only_unmanaged_positions() -> None:
    assert _unmanaged_stress(
        {
            "positions": [
                {"symbol": "SPXU", "market_value_base": 800.0},
                {"symbol": "MCL", "market_value_base": 77_000.0},
                {"symbol": "SPCX", "market_value_base": -92.74},
            ]
        }
    ) == 92.74


def test_gold_rollover_recovers_only_its_immediate_active_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor_id = "a" * 64
    successor_id = "b" * 64
    xsp_id = "c" * 64
    mcl_id = "d" * 64
    selections = {
        "gold-1oz-stage76-margin": {"selection_id": predecessor_id},
        "xsp-upro-spxu-rth-cash": {"selection_id": xsp_id},
        "mcl-two-speed-auction-margin": {"selection_id": mcl_id},
    }
    paths = {}
    for sleeve_id in selections:
        path = tmp_path / "db/calibration/selections" / f"{sleeve_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        paths[sleeve_id] = path
    plan = {
        "plan_id": "e" * 64,
        "sleeves": [
            {
                "sleeve_id": sleeve_id,
                "run_id": selection["selection_id"],
                "selection_file_sha256": index * 64,
            }
            for sleeve_id, selection, index in zip(
                selections,
                selections.values(),
                ("1", "2", "3"),
                strict=True,
            )
        ],
    }

    monkeypatch.setattr(gold_onset_cli, "load_live_capital_plan", lambda _path: plan)

    def allocated(_plan, *, sleeve_id, repository_root):
        del _plan, repository_root
        selection = selections[sleeve_id]
        sleeve = next(row for row in plan["sleeves"] if row["sleeve_id"] == sleeve_id)
        return selection, paths[sleeve_id], sleeve["selection_file_sha256"]

    monkeypatch.setattr(gold_onset_cli, "load_allocated_live_selection", allocated)
    intent_path = tmp_path / "state/intent.json"
    intent, recovered = _gold_rollover_boundary(
        capital_path=tmp_path / "capital.json",
        intent_path=intent_path,
        root=tmp_path,
    )
    assert recovered is None
    assert intent["predecessor_selection_id"] == predecessor_id
    assert intent_path.stat().st_mode & 0o777 == 0o600

    selections["gold-1oz-stage76-margin"] = {
        "selection_id": successor_id,
        "allocation_successor": {
            "predecessor_selection_id": predecessor_id,
        },
    }
    plan["plan_id"] = "f" * 64
    plan["sleeves"][0]["run_id"] = successor_id
    generation = (
        tmp_path
        / "db/calibration/portfolio_generations"
        / f"{plan['plan_id']}.json"
    )
    generation.parent.mkdir(parents=True, exist_ok=True)
    generation.write_text("generation\n")
    stability = tmp_path / "db/calibration/portfolio_capital_owner_stability.json"
    stability.write_text("stability\n")
    stability_sha = hashlib.sha256(stability.read_bytes()).hexdigest()
    archive = (
        tmp_path
        / "db/calibration/portfolio_capital_stability"
        / f"{stability_sha}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(stability.read_bytes())
    validated = []
    monkeypatch.setattr(
        gold_onset_cli,
        "load_gold_live_selection_from_mapping",
        lambda selection: validated.append(selection) or selection,
    )
    gates = []
    monkeypatch.setattr(
        gold_onset_cli,
        "portfolio_capital_owner_stability_gate",
        lambda _path, **kwargs: gates.append(kwargs["sleeve_id"])
        or {"status": "PASS"},
    )

    repeated_intent, recovered = _gold_rollover_boundary(
        capital_path=tmp_path / "capital.json",
        intent_path=intent_path,
        root=tmp_path,
    )

    assert repeated_intent == intent
    assert recovered["rollover"]["selection_id"] == successor_id
    assert recovered["rollover"]["predecessor_selection_id"] == predecessor_id
    assert recovered["rollover"]["recovered_after_interruption"] is True
    assert recovered["rollover"]["retained_xsp_selection_id"] == xsp_id
    assert recovered["rollover"]["retained_mcl_selection_id"] == mcl_id
    assert recovered["rollover"]["submitted_orders"] == 0
    assert validated == [selections["gold-1oz-stage76-margin"]]
    assert set(gates) == set(selections)


def test_gold_rollover_rejects_a_non_immediate_active_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "plan_id": "a" * 64,
        "sleeves": [
            {
                "sleeve_id": "gold-1oz-stage76-margin",
                "run_id": "b" * 64,
                "selection_file_sha256": "c" * 64,
            }
        ],
    }
    selection_path = tmp_path / "gold.json"
    selection_path.write_text("{}\n")
    selection = {"selection_id": "b" * 64}
    monkeypatch.setattr(gold_onset_cli, "load_live_capital_plan", lambda _path: plan)
    monkeypatch.setattr(
        gold_onset_cli,
        "load_allocated_live_selection",
        lambda *_args, **_kwargs: (selection, selection_path, "c" * 64),
    )
    intent_path = tmp_path / "intent.json"
    _gold_rollover_boundary(
        capital_path=tmp_path / "capital.json",
        intent_path=intent_path,
        root=tmp_path,
    )
    selection.update(
        {
            "selection_id": "d" * 64,
            "allocation_successor": {"predecessor_selection_id": "e" * 64},
        }
    )

    with pytest.raises(ValueError, match="crossed the rollover boundary"):
        _gold_rollover_boundary(
            capital_path=tmp_path / "capital.json",
            intent_path=intent_path,
            root=tmp_path,
        )


def test_gold_rollover_recovery_returns_before_constructing_ib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recovered = {
        "rollover": {
            "selection_id": "a" * 64,
            "submitted_orders": 0,
            "verdict": "FRESH_FAIL_CLOSED_GOLD_RUN_SELECTED_FLAT",
        }
    }
    monkeypatch.setattr(
        gold_onset_cli,
        "_gold_rollover_boundary",
        lambda **_kwargs: ({"intent_id": "b" * 64}, recovered),
    )

    class ForbiddenIB:
        def __init__(self) -> None:
            raise AssertionError("recovery must not construct a broker client")

    monkeypatch.setattr(gold_onset_cli, "IB", ForbiddenIB)
    gold_onset_cli.main(
        [
            "--rollover-canary",
            "--rollover-intent",
            str(tmp_path / "intent.json"),
        ]
    )

    assert json.loads(capsys.readouterr().out) == recovered


def test_gold_selection_extends_xsp_cash_plan_as_exclusive_margin_overlay(
    tmp_path: Path,
) -> None:
    selected = _selection(datetime(2026, 8, 3, 10, tzinfo=UTC))
    path = tmp_path / "gold.json"
    publish_gold_live_selection(path, selected)
    predecessor = build_live_capital_plan(
        account_id="U123",
        account_type="CASH",
        currency="USD",
        observed_settled_cash_usd=1318.05,
        managed_capital_usd=900.46,
        sleeves=[
            {
                "sleeve_id": "xsp-upro-spxu-rth-cash",
                "strategy_id": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
                "run_id": "a" * 64,
                "selection_path": "db/calibration/xsp_selected_live_transport.json",
                "selection_file_sha256": "b" * 64,
                "capital_kind": "CASH_DEBIT",
                "weight_bps": 10_000,
            }
        ],
        reserve_reasons=["cash_above_selected_authority_unallocated"],
        created_at_utc="2026-08-02T00:00:00+00:00",
    )

    plan = build_gold_portfolio_capital_plan(
        selected,
        selection_path=path,
        current_plan=predecessor,
    )

    assert plan["schema"] == "live.capital-plan.v2"
    assert plan["supersedes_plan_id"] == predecessor["plan_id"]
    assert plan["capital"]["managed_capital_cents"] == 90_046
    assert plan["capital"]["unallocated_reserve_cents"] == 41_759
    assert len(plan["sleeves"]) == 2
    margin = next(row for row in plan["sleeves"] if row["capital_kind"] == "FUTURES_MARGIN")
    assert margin["weight_bps"] == 0
    assert margin["margin"]["max_contracts"] == 1
    assert plan["constraints"]["max_concurrent_directional_sleeves"] == 1


def _target(entry_time: datetime, direction: str = "up") -> dict[str, object]:
    body = {
        "direction": direction,
        "signal_bar_utc": (entry_time - timedelta(hours=1)).isoformat(),
        "entry_time_utc": entry_time.isoformat(),
        "signal_entry_price": 4100.0,
        "mfe_usd": 0.0,
        "mae_usd": 0.0,
    }
    from tradebot.research.live_calibration import calibration_fingerprint

    return {**body, "target_id": calibration_fingerprint(body)}


def _risk(*, breaches: list[str] | None = None) -> dict[str, object]:
    return {"safety_breaches": list(breaches or ())}


def test_gold_plan_enters_only_a_fresh_post_selection_admission() -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    target = _target(selected_at + timedelta(minutes=5))
    source_at = selected_at + timedelta(minutes=6)

    plan = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(source_at, target=target),
        broker_position=0,
        open_orders=(),
        risk_state=_risk(),
        observed_at=source_at + timedelta(minutes=1),
    )
    assert plan["status"] == "ACTIONABLE"
    assert plan["reason"] == "fresh_stage76_admission"
    assert plan["leg"]["action"] == "BUY"

    stale = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(source_at, target=target),
        broker_position=0,
        open_orders=(),
        risk_state=_risk(),
        observed_at=source_at + timedelta(minutes=16),
    )
    assert stale["status"] == "HOLD"
    assert stale["reason"] == "entry_source_stale"

    inherited = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(source_at, target=_target(selected_at)),
        broker_position=0,
        open_orders=(),
        risk_state=_risk(),
        observed_at=source_at + timedelta(minutes=1),
    )
    assert inherited["status"] == "HOLD"
    assert inherited["reason"] == "preselection_target_not_adopted"


def test_gold_flat_wide_book_persists_accounting_but_cannot_enter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    observed_at = selected_at + timedelta(minutes=7)
    contract = gold_live_contract(selected)
    ticker = SimpleNamespace(
        contract=contract,
        bid=4109.25,
        bidSize=1,
        ask=4112.0,
        askSize=1,
        last=4109.25,
        close=4109.0,
        marketDataType=1,
        tbTopQuoteUpdatedMono=time.monotonic(),
    )

    class Client:
        async def ensure_ticker(self, _contract, *, owner: str):
            assert owner == "gold-live-stage76"
            return ticker

    async def broker(_client):
        return {
            "account_id": "U123",
            "account_type": "CASH",
            "base_currency": "AUD",
            "settled_cash_usd": 1318.05,
            "equity_with_loan_aud": 2106.0,
            "available_funds_aud": 2073.0,
            "excess_liquidity_aud": 2078.0,
            "initial_margin_aud": 33.0,
            "maintenance_margin_aud": 29.0,
            "gross_position_value_aud": 94.0,
            "usd_to_aud": 1.427,
            "positions": [],
            "open_orders": [],
        }

    monkeypatch.setattr(
        "tradebot.research.gold_live_runtime.gold_broker_snapshot", broker
    )
    ledger = LiveCalibrationLedger(tmp_path / "quote-outage.jsonl")
    output = asyncio.run(
        advance_gold_live_transport(
            ledger,
            client=Client(),
            selection=selected,
            source_checkpoint=_source(
                observed_at - timedelta(minutes=1),
                target=_target(observed_at - timedelta(minutes=2)),
            ),
            capital_plan={},
            selection_file_sha256="a" * 64,
            observed_at=observed_at,
            quote_wait_seconds=0,
        )
    )

    row = tuple(ledger.records())[-1]
    assert output["status"] == "HOLD"
    assert output["plan"]["reason"] == "entry_market_data_unavailable"
    assert output["plan"]["leg"] is None
    assert output["risk_state"]["position_from_fills"] == 0
    assert output["submitted_orders"] == 0
    assert row["evidence"]["phase"] == "STATE"
    assert row["evidence"]["quote"]["health"]["eligible"] is True
    assert row["evidence"]["quote"]["entry_health"]["eligible"] is False
    assert row["evidence"]["quote"]["entry_health"]["reasons"] == [
        "spread_above_maximum"
    ]
    assert row["evidence"]["submitted_orders"] == 0
    profitability = gold_live_profitability_receipt(
        tuple(ledger.records()),
        selection=selected,
        as_of=observed_at + timedelta(seconds=90),
    )
    assert profitability["status"] == "ACTIVE"
    assert profitability["reasons"] == []
    assert profitability["clock"]["coverage_broken"] is False


def test_gold_quote_outage_cannot_account_for_a_held_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    contract = gold_live_contract(selected)
    ticker = SimpleNamespace(
        contract=contract,
        bid=float("nan"),
        ask=float("nan"),
        last=4109.25,
        close=4109.0,
        marketDataType=1,
        tbTopQuoteUpdatedMono=None,
    )

    class Client:
        async def ensure_ticker(self, _contract, *, owner: str):
            return ticker

    async def broker(_client):
        return {
            "account_id": "U123",
            "positions": [
                {
                    "symbol": "1OZ",
                    "con_id": selected["contract"]["con_id"],
                    "quantity": 1.0,
                }
            ],
            "open_orders": [],
        }

    monkeypatch.setattr(
        "tradebot.research.gold_live_runtime.gold_broker_snapshot", broker
    )
    with pytest.raises(ValueError, match="lacks fresh streaming NBBO"):
        asyncio.run(
            advance_gold_live_transport(
                LiveCalibrationLedger(tmp_path / "held-outage.jsonl"),
                client=Client(),
                selection=selected,
                source_checkpoint=_source(selected_at),
                capital_plan={},
                selection_file_sha256="a" * 64,
                observed_at=selected_at + timedelta(minutes=1),
                quote_wait_seconds=0,
            )
        )


def test_gold_plan_closes_and_reverses_without_waiting_for_rth() -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    at = selected_at + timedelta(hours=8)
    down = _target(at - timedelta(minutes=1), "down")

    plan = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(at, target=down),
        broker_position=1,
        open_orders=(),
        risk_state=_risk(),
        observed_at=at + timedelta(minutes=1),
        entry_market_data_eligible=False,
    )

    assert plan["status"] == "ACTIONABLE"
    assert plan["reason"] == "close_before_reverse"
    assert plan["leg"]["action"] == "SELL"
    assert plan["desired_after_close"] == "down"
    assert plan["execution_state_context"]["news"]["authority"] == "attribution_only"


def test_gold_entry_pricing_rests_when_the_book_widens_but_exit_still_crosses() -> None:
    contract = SimpleNamespace(secType="FUT", minTick=0.25)
    ticker = SimpleNamespace(
        bid=4409.5,
        bidSize=1,
        ask=4412.0,
        askSize=1,
        last=4410.0,
        close=4410.0,
        marketDataType=1,
    )
    entry = _gold_price_for_ticker(
        contract,
        ticker,
        entry=True,
        resting_price=4410.25,
    )
    exit_price = _gold_price_for_ticker(contract, ticker, entry=False)

    assert entry("CROSS", "BUY") == 4410.25
    assert entry(
        "CROSS",
        "BUY",
        bid=ticker.bid,
        ask=ticker.ask,
        last=ticker.last,
        ticker=ticker,
    ) is None
    assert exit_price("CROSS", "SELL") == ticker.bid


def _fill_record(
    selection_id: str,
    *,
    exec_id: str,
    side: str,
    price: float,
    commission: float,
    at: datetime,
) -> dict[str, object]:
    return {
        "kind": "checkpoint",
        "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "evidence": {
            "selection_id": selection_id,
            "phase": "TERMINAL",
            "broker_order": {
                "fills": [
                    {
                        "exec_id": exec_id,
                        "time_utc": at.isoformat(),
                        "side": side,
                        "symbol": "1OZ",
                        "shares": 1,
                        "price": price,
                        "commission": commission,
                        "commission_currency": "USD",
                    }
                ]
            },
        },
    }


def test_gold_risk_reconstructs_long_short_fills_and_liquidation_cost() -> None:
    selected = _selection(datetime(2026, 8, 3, 10, tzinfo=UTC))
    opened = _fill_record(
        selected["selection_id"],
        exec_id="buy-1",
        side="BOT",
        price=4100.0,
        commission=0.66,
        at=datetime(2026, 8, 3, 11, tzinfo=UTC),
    )
    open_risk = gold_transport_risk_state(
        selection=selected,
        records=(opened,),
        observed_at=datetime(2026, 8, 3, 12, tzinfo=UTC),
        liquidation_price=4110.0,
    )
    assert open_risk["position_from_fills"] == 1
    assert open_risk["run_gross_usd"] == 10.0
    assert open_risk["run_cost_usd"] == 1.32
    assert open_risk["run_net_usd"] == pytest.approx(8.68)

    closed = _fill_record(
        selected["selection_id"],
        exec_id="sell-1",
        side="SLD",
        price=4112.0,
        commission=0.66,
        at=datetime(2026, 8, 3, 13, tzinfo=UTC),
    )
    flat = gold_transport_risk_state(
        selection=selected,
        records=(opened, closed),
        observed_at=datetime(2026, 8, 3, 14, tzinfo=UTC),
        liquidation_price=4112.0,
    )
    assert flat["position_from_fills"] == 0
    assert flat["run_realized_net_usd"] == pytest.approx(10.68)
    assert flat["closed_trades"] == 1


def _terminal_trade(
    contract: object,
    order_ref: str,
    *,
    filled: float = 1.0,
) -> object:
    at = datetime(2026, 8, 3, 11, tzinfo=UTC)
    fills = (
        [
            SimpleNamespace(
                time=at,
                execution=SimpleNamespace(
                    execId="gold-fill-1",
                    side="BOT",
                    shares=1.0,
                    price=4109.25,
                ),
                commissionReport=SimpleNamespace(
                    commission=0.66,
                    currency="USD",
                ),
            )
        ]
        if filled
        else []
    )
    return SimpleNamespace(
        contract=contract,
        order=SimpleNamespace(
            orderId=501,
            permId=901,
            orderRef=order_ref,
            action="BUY",
            totalQuantity=1,
            lmtPrice=4109.25,
        ),
        orderStatus=SimpleNamespace(
            status="Filled" if filled else "Cancelled",
            filled=filled,
            remaining=1.0 - filled,
            avgFillPrice=4109.25 if filled else 0.0,
        ),
        fills=fills,
        isDone=lambda: True,
    )


class _GoldExecutionClient:
    def __init__(self, contract: object, *, filled: float = 1.0) -> None:
        self.contract = contract
        self.filled = filled
        self.trade: object | None = None
        self.placed = 0

    async def reconcile_trades_for_order_ref(self, _order_ref: str):
        return [self.trade] if self.trade is not None else []

    async def preview_limit_order(self, *_args):
        return BrokerOrderPreview(
            status="PreSubmitted",
            init_margin_change=560.0,
            maintenance_margin_change=487.0,
            commission=0.66,
            min_commission=0.66,
            max_commission=0.66,
            commission_currency="USD",
            warning_text="",
        )

    async def place_limit_order(self, *_args):
        self.placed += 1
        self.trade = _terminal_trade(
            self.contract,
            str(_args[-1]),
            filled=self.filled,
        )
        return self.trade

    async def reconcile_order_state(self, **_kwargs):
        return {"trade": self.trade}


def test_gold_broker_snapshot_converts_portfolio_values_to_account_base() -> None:
    contract = SimpleNamespace(
        symbol="TQQQ",
        localSymbol="TQQQ",
        conId=72_539_702,
        secType="STK",
        currency="USD",
    )

    class Client:
        async def fetch_portfolio(self):
            return [
                SimpleNamespace(
                    contract=contract,
                    position=1.0,
                    marketValue=69.44,
                )
            ]

        def account_id(self):
            return "U123"

        def account_text_value(self, _tag):
            return "STKCASH"

        def account_value(self, tag, *, currency=None):
            values = {
                ("CashBalance", "USD"): 1_318.05,
                ("EquityWithLoanValue", "AUD"): 3_107.79,
                ("AvailableFunds", "AUD"): 3_072.19,
                ("ExcessLiquidity", "AUD"): 3_077.79,
                ("FullInitMarginReq", "AUD"): 35.60,
                ("FullMaintMarginReq", "AUD"): 30.00,
                ("GrossPositionValue", "AUD"): 98.88,
                ("ExchangeRate", "USD"): 1.4239442,
            }
            return values[(tag, currency)], currency, None

        def open_trades(self):
            return []

    snapshot = asyncio.run(gold_broker_snapshot(Client()))

    assert snapshot["positions"] == [
        {
            "symbol": "TQQQ",
            "local_symbol": "TQQQ",
            "con_id": 72_539_702,
            "sec_type": "STK",
            "currency": "USD",
            "quantity": 1.0,
            "market_value_base_cents": 9_888,
        }
    ]


def test_gold_execution_is_one_order_and_restart_adopts_terminal_fill(
    tmp_path: Path,
) -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    source_at = selected_at + timedelta(minutes=6)
    plan = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(
            source_at,
            target=_target(selected_at + timedelta(minutes=5)),
        ),
        broker_position=0,
        open_orders=(),
        risk_state=_risk(),
        observed_at=source_at + timedelta(minutes=1),
    )
    plan["capital_admission"] = {"status": "ALLOW"}
    contract = gold_live_contract(selected)
    ticker = SimpleNamespace(
        bid=4109.25,
        bidSize=1,
        ask=4109.5,
        askSize=1,
        last=4109.25,
        close=4109.0,
        marketDataType=1,
    )
    ledger = LiveCalibrationLedger(tmp_path / "gold.jsonl")
    client = _GoldExecutionClient(contract)

    first = asyncio.run(
        execute_gold_transport_plan(
            ledger,
            client=client,
            selection=selected,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=source_at + timedelta(minutes=1),
        )
    )
    restarted = asyncio.run(
        execute_gold_transport_plan(
            ledger,
            client=client,
            selection=selected,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=source_at + timedelta(minutes=2),
        )
    )

    assert first["status"] == "TERMINAL"
    assert first["submitted_orders"] == 1
    assert restarted["status"] == "TERMINAL"
    assert restarted["submitted_orders"] == 0
    assert restarted["checkpoint_id"] == first["checkpoint_id"]
    assert client.placed == 1
    assert gold_transport_order_ref(plan).startswith("GOLD76-")


def test_gold_execution_rejects_cancelled_zero_fill(tmp_path: Path) -> None:
    selected_at = datetime(2026, 8, 3, 10, tzinfo=UTC)
    selected = _selection(selected_at)
    source_at = selected_at + timedelta(minutes=6)
    plan = project_gold_transport_plan(
        selection=selected,
        source_checkpoint=_source(
            source_at,
            target=_target(selected_at + timedelta(minutes=5)),
        ),
        broker_position=0,
        open_orders=(),
        risk_state=_risk(),
        observed_at=source_at + timedelta(minutes=1),
    )
    plan["capital_admission"] = {"status": "ALLOW"}
    contract = gold_live_contract(selected)
    client = _GoldExecutionClient(contract, filled=0.0)

    with pytest.raises(ValueError, match="terminated without one complete fill"):
        asyncio.run(
            execute_gold_transport_plan(
                LiveCalibrationLedger(tmp_path / "cancelled.jsonl"),
                client=client,
                selection=selected,
                plan=plan,
                contract=contract,
                ticker=SimpleNamespace(
                    bid=4109.25,
                    bidSize=1,
                    ask=4109.5,
                    askSize=1,
                    last=4109.25,
                    close=4109.0,
                    marketDataType=1,
                ),
                observed_at=source_at + timedelta(minutes=1),
            )
        )


def test_gold_and_xsp_live_workers_share_one_account_execution_lock() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = (root / "deploy/systemd/tradebot-gold-live.service").read_text()
    xsp = (root / "deploy/systemd/tradebot-xsp-shadow.service").read_text()
    lock = "%t/tradebot-live-account.lock"

    assert lock in gold
    assert lock in xsp
    assert "/usr/bin/flock --exclusive --wait 180" in gold
    assert "/usr/bin/flock --exclusive --wait 180" in xsp
    assert "Environment=IBKR_READONLY=0" in gold
    assert "tradebot.research.gold_live_cli" in gold
    assert "StartLimitIntervalSec=60s" in gold
    assert "StartLimitBurst=3" in gold
    assert "Restart=on-failure" not in gold
    assert "RestartSec=" not in gold
