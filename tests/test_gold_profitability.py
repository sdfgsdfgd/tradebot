from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.live.capital import admit_live_capital, build_live_capital_plan_v3
from tradebot.live.capital_stability import (
    PORTFOLIO_CAPITAL_STABILITY_SCHEMA,
    portfolio_capital_owner_stability_gate,
)
from tradebot.live.order_evidence import single_contract_execution_graduation_gate
from tradebot.research import gold_profitability
from tradebot.research.gold_profitability import (
    GOLD_LIVE_EXECUTION_VERSION,
    gold_1oz_evaluation_slots,
    gold_1oz_maintenance,
    gold_live_graduation_inputs,
    gold_live_profitability_receipt,
    gold_runtime_parity_graduation_gate,
)
from tradebot.research.gold_regime_harmony import GOLD_REGIME_HARMONY_VERSION
from tradebot.research.live_graduation import reduce_live_graduation
from tradebot.research.xsp_capital_stability import (
    xsp_capital_owner_stability_graduation_gate,
)


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 8, 3, 12, 41, 13, tzinfo=UTC)
GOLD_ID = "a" * 64
XSP_ID = "b" * 64


def _selection() -> dict[str, object]:
    parity_path = ROOT / "backtests/gold/one_oz_regime_harmony_runtime_parity_20260803.json"
    parity = json.loads(parity_path.read_text())
    return {
        "selection_id": GOLD_ID,
        "selected_at_utc": START.isoformat(),
        "run_started_at_utc": START.isoformat(),
        "strategy_version": GOLD_REGIME_HARMONY_VERSION,
        "execution_strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "contract": {"con_id": 222, "symbol": "1OZ"},
        "broker_at_selection": {"account_id": "U123"},
        "allocation_successor": {"package_id": "gold-one-contract"},
        "evidence": {
            "runtime_parity": {
                "sha256": hashlib.sha256(parity_path.read_bytes()).hexdigest()
            },
            "crown": dict(parity["crown"]),
        },
    }


def _risk(*, net: float = 0.0, fills: int = 0, trades: int = 0) -> dict[str, object]:
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
        "safety_breaches": [],
    }


def _state(
    evaluated: datetime,
    *,
    recorded: datetime | None = None,
    net: float = 0.0,
    fills: int = 0,
    trades: int = 0,
) -> dict[str, object]:
    stamp = recorded or evaluated
    identity = hashlib.sha256(f"{evaluated.isoformat()}:{stamp.isoformat()}".encode()).hexdigest()
    return {
        "kind": "checkpoint",
        "checkpoint_id": identity,
        "recorded_at_utc": stamp.isoformat(),
        "evaluation_as_of_utc": evaluated.isoformat(),
        "strategy_id": GOLD_REGIME_HARMONY_VERSION,
        "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "status": "EVALUATED",
        "evidence": {
            "selection_id": GOLD_ID,
            "phase": "STATE",
            "submitted_orders": 0,
            "plan": {"held_direction": None, "leg": None, "capital_admission": None},
            "broker_state": {"positions": [], "open_orders": []},
            "risk_state": _risk(net=net, fills=fills, trades=trades),
        },
    }


def _patch_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gold_profitability,
        "load_gold_live_selection_from_mapping",
        lambda value: dict(value),
    )


def test_gold_clock_excludes_maintenance_and_owns_exact_evaluation_minutes() -> None:
    assert gold_1oz_maintenance("2026-08-03T21:00:00+00:00")
    assert gold_1oz_maintenance("2026-08-08T07:00:00+00:00")
    assert not gold_1oz_maintenance("2026-08-03T20:59:00+00:00")
    slots = gold_1oz_evaluation_slots(
        "2026-08-03T20:57:00+00:00", "2026-08-03T21:07:00+00:00"
    )
    assert [row.isoformat() for row in slots] == [
        "2026-08-03T21:02:00+00:00",
        "2026-08-03T21:07:00+00:00",
    ]


def test_manual_cold_checkpoint_cannot_conflict_with_natural_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selection(monkeypatch)
    rows = [
        _state(
            datetime(2026, 8, 3, 12, 41, 53, tzinfo=UTC),
            recorded=datetime(2026, 8, 3, 12, 41, 53, tzinfo=UTC),
        ),
        _state(
            datetime(2026, 8, 3, 12, 42, 3, tzinfo=UTC),
            recorded=datetime(2026, 8, 3, 12, 42, 3, tzinfo=UTC),
        ),
        _state(datetime(2026, 8, 3, 12, 47, tzinfo=UTC)),
    ]
    receipt = gold_live_profitability_receipt(
        rows, selection=_selection(), as_of="2026-08-03T12:49:00+00:00"
    )
    assert receipt["status"] == "ACTIVE"
    assert receipt["reasons"] == []
    assert receipt["clock"]["coverage_started_at_utc"] == "2026-08-03T12:42:00+00:00"
    assert receipt["clock"]["evaluated_slots"] == 1


def test_positive_24h_requires_a_complete_timer_prefix_and_authentic_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selection(monkeypatch)
    baseline = datetime(2026, 8, 3, 12, 42, tzinfo=UTC)
    end = baseline + timedelta(hours=24)
    rows = [_state(baseline)]
    for slot in gold_1oz_evaluation_slots(baseline, end):
        rows.append(_state(slot, net=4.0, fills=2, trades=1))
    receipt = gold_live_profitability_receipt(
        rows, selection=_selection(), as_of=end + timedelta(seconds=90)
    )
    assert receipt["status"] == "ACTIVE"
    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["milestones"]["48h"]["passed"] is False
    assert receipt["economics"]["net_usd"] == 4.0


def _package(
    package_id: str,
    *,
    cash: int,
    initial: int = 0,
    maintenance: int = 0,
    stress: int = 100,
) -> dict[str, object]:
    return {
        "package_id": package_id,
        "rank": 0,
        "cash_debit_usd_cents": cash,
        "initial_margin_base_cents": initial,
        "maintenance_margin_base_cents": maintenance,
        "stressed_loss_usd_cents": stress,
        "fx_stress_bps": 11_000,
    }


def _portfolio_manifest(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    selections = tmp_path / "db/calibration/selections"
    selections.mkdir(parents=True)
    selected = {
        "gold": {
            "selection_id": GOLD_ID,
            "strategy_version": GOLD_REGIME_HARMONY_VERSION,
            "allocation_successor": {"package_id": "gold-one-contract"},
        },
        "xsp": {
            "selection_id": XSP_ID,
            "strategy_version": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
            "allocation_successor": {"package_id": "xsp-minimum"},
        },
    }
    bindings = {}
    for key, value in selected.items():
        path = selections / f"{value['selection_id']}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        bindings[key] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "selection_id": value["selection_id"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    sleeves = [
        {
            "sleeve_id": "gold-1oz-stage76-margin",
            "strategy_id": selected["gold"]["strategy_version"],
            "run_id": GOLD_ID,
            "selection_path": bindings["gold"]["path"],
            "selection_file_sha256": bindings["gold"]["sha256"],
            "capital_kind": "FUTURES_MARGIN",
            "position_symbols": ["1OZ"],
            "residual_weight_bps": 0,
            "minimum_package_id": "gold-one-contract",
            "package_ladder": [
                _package(
                    "gold-one-contract",
                    cash=66,
                    initial=60_000,
                    maintenance=52_000,
                    stress=25_617,
                )
            ],
        },
        {
            "sleeve_id": "xsp-upro-spxu-rth-cash",
            "strategy_id": selected["xsp"]["strategy_version"],
            "run_id": XSP_ID,
            "selection_path": bindings["xsp"]["path"],
            "selection_file_sha256": bindings["xsp"]["sha256"],
            "capital_kind": "CASH_DEBIT",
            "position_symbols": ["SPXU", "UPRO"],
            "residual_weight_bps": 10_000,
            "minimum_package_id": "xsp-minimum",
            "package_ladder": [_package("xsp-minimum", cash=40_046, stress=1_761)],
        },
    ]
    plan = build_live_capital_plan_v3(
        account_id="U123",
        account_type="CASH",
        cash_currency="USD",
        base_currency="AUD",
        observed_settled_cash_usd="1318.05",
        observed_available_funds_base="2074.57",
        observed_excess_liquidity_base="2079.38",
        usd_to_base_rate="1.428756",
        minimum_post_reservation_base="300",
        unmanaged_position_stress_base="92.69",
        sleeves=sleeves,
        reserve_reasons=["minimum_packages_reserved_first"],
        created_at_utc=START,
    )
    generation = {
        "schema": "live.portfolio-package-generation.v1",
        "authority": "zero-transmission-successor-and-capital-switch",
        "plan": plan,
        "selections": bindings,
        "submitted_orders": 0,
    }
    generation_path = tmp_path / f"db/calibration/portfolio_generations/{plan['plan_id']}.json"
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    owner = tmp_path / "owner.py"
    owner.write_text("OWNER = 1\n", encoding="utf-8")
    manifest = {
        "schema": PORTFOLIO_CAPITAL_STABILITY_SCHEMA,
        "authority": "frozen_portfolio_package_generation",
        "observed_at_utc": START.isoformat(),
        "source_revision": "revision",
        "generation": {
            "path": generation_path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
            "plan_id": plan["plan_id"],
        },
        "selections": {
            sleeve["sleeve_id"]: {
                "selection_id": sleeve["run_id"],
                "selection_path": sleeve["selection_path"],
                "selection_file_sha256": sleeve["selection_file_sha256"],
                "allocated_package_id": sleeve["allocated_package_id"],
            }
            for sleeve in plan["sleeves"]
        },
        "capital_semantic_surface": {
            "owner.py": hashlib.sha256(owner.read_bytes()).hexdigest()
        },
        "checks": {"minimum_packages_reserved_first": True},
        "verdict": "PASS_CAPITAL_OWNER_STABLE",
        "boundaries": {
            "broker_queried": False,
            "service_or_timer_mutated": False,
            "selection_mutated": False,
            "submitted_orders": 0,
            "profitability_clock_mutated": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, selections / f"{GOLD_ID}.json", plan


def test_shared_portfolio_manifest_binds_gold_and_xsp(tmp_path: Path) -> None:
    manifest, gold_path, plan = _portfolio_manifest(tmp_path)
    gold_sha = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    gold = portfolio_capital_owner_stability_gate(
        manifest,
        repo_root=tmp_path,
        sleeve_id="gold-1oz-stage76-margin",
        selection_id=GOLD_ID,
        selection_file_sha256=gold_sha,
    )
    xsp_sleeve = next(
        row for row in plan["sleeves"] if row["sleeve_id"] == "xsp-upro-spxu-rth-cash"
    )
    xsp = xsp_capital_owner_stability_graduation_gate(
        manifest,
        repo_root=tmp_path,
        selection_id=XSP_ID,
        selection_file_sha256=xsp_sleeve["selection_file_sha256"],
    )
    assert gold["status"] == xsp["status"] == "PASS"
    changed = json.loads(gold_path.read_text())
    changed["strategy_version"] = "drifted"
    gold_path.write_text(json.dumps(changed), encoding="utf-8")
    rejected = portfolio_capital_owner_stability_gate(
        manifest,
        repo_root=tmp_path,
        sleeve_id="gold-1oz-stage76-margin",
        selection_id=GOLD_ID,
        selection_file_sha256=gold_sha,
    )
    assert rejected["status"] == "INVALID"
    assert "portfolio_selected_run_invalid:gold-1oz-stage76-margin" in rejected["reasons"]


def test_runtime_parity_rehashes_selected_crown_and_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selection(monkeypatch)
    path = ROOT / "backtests/gold/one_oz_regime_harmony_runtime_parity_20260803.json"
    passed = gold_runtime_parity_graduation_gate(
        path, repo_root=ROOT, selection=_selection()
    )
    assert passed["status"] == "PASS"
    changed = _selection()
    changed["evidence"]["crown"]["artifact_sha256"] = "0" * 64
    rejected = gold_runtime_parity_graduation_gate(
        path, repo_root=ROOT, selection=changed
    )
    assert rejected["status"] == "INVALID"


def test_single_contract_execution_gate_accepts_one_terminal_fill() -> None:
    decision = admit_live_capital(
        None,
        intent="EXIT",
        account_id="U123",
        account_type="CASH",
        currency="USD",
        sleeve_id="gold-1oz-stage76-margin",
        run_id=GOLD_ID,
        selection_file_sha256="c" * 64,
        capital_kind="FUTURES_MARGIN",
        projected_capital_usd=0,
        cash_debit_usd=0,
        available_cash_usd=0,
    )
    transition = "d" * 64
    plan = {
        "transition_id": transition,
        "leg": {"symbol": "1OZ", "action": "SELL", "quantity": 1},
        "capital_admission": decision,
    }
    order_ref = f"GOLD76-{transition[:24]}"
    preview = {
        "status": "PreSubmitted",
        "commission": 0.66,
        "min_commission": 0.66,
        "max_commission": 0.66,
        "commission_currency": "USD",
        "warning_text": "",
    }
    terminal = {
        "order_ref": order_ref,
        "symbol": "1OZ",
        "con_id": 222,
        "action": "SELL",
        "quantity": 1,
        "limit_price": 4100,
        "filled": 1,
        "remaining": 0,
        "done": True,
        "fills": [
            {
                "exec_id": "fill-1",
                "time_utc": START.isoformat(),
                "side": "SLD",
                "symbol": "1OZ",
                "shares": 1,
                "price": 4100,
                "commission": 0.66,
                "commission_currency": "USD",
            }
        ],
    }
    rows = [
        {"evidence": {"phase": "PREPARED", "order_ref": order_ref, "plan": plan, "what_if_preview": preview, "submitted_orders": 0}},
        {"evidence": {"phase": "SUBMITTED", "order_ref": order_ref, "plan": plan, "what_if_preview": preview, "submitted_orders": 1}},
        {"evidence": {"phase": "TERMINAL", "order_ref": order_ref, "plan": plan, "what_if_preview": preview, "broker_order": terminal, "submitted_orders": 1}},
    ]
    gate = single_contract_execution_graduation_gate(
        rows,
        selection_id=GOLD_ID,
        sleeve_id="gold-1oz-stage76-margin",
        symbol="1OZ",
        con_id=222,
        order_ref_prefix="GOLD76",
        ladder_schema="gold.execution-ladder-transition.v1",
        max_commission_usd=0.66,
    )
    assert gate["status"] == "PASS"


def test_gold_projection_reuses_shared_reducer_and_portfolio_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_selection(monkeypatch)
    manifest, selection_path, _plan = _portfolio_manifest(tmp_path)
    selected = _selection()
    selection_path.write_text(json.dumps(selected), encoding="utf-8")
    # This integration test uses the already validated manifest test for plan identity;
    # isolate Gold's adapter gates by substituting the pure capital gate result.
    monkeypatch.setattr(
        gold_profitability,
        "portfolio_capital_owner_stability_gate",
        lambda *_args, **_kwargs: {"status": "PASS", "reasons": [], "evidence": {}},
    )
    rows = [
        _state(datetime(2026, 8, 3, 12, 42, 3, tzinfo=UTC)),
        _state(datetime(2026, 8, 3, 12, 47, tzinfo=UTC)),
    ]
    cutoff = datetime(2026, 8, 3, 12, 49, tzinfo=UTC)
    profitability = gold_live_profitability_receipt(
        rows, selection=selected, as_of=cutoff
    )
    inputs = gold_live_graduation_inputs(
        selection=selected,
        selection_path=selection_path,
        records=rows,
        cutoff_utc=cutoff,
        profitability_receipt=profitability,
        runtime_parity_path=ROOT
        / "backtests/gold/one_oz_regime_harmony_runtime_parity_20260803.json",
        capital_owner_stability_path=manifest,
        repo_root=ROOT,
    )
    receipt = reduce_live_graduation(
        target_milestone="24h", cutoff_utc=cutoff, **inputs
    )
    assert receipt["gates"]["runtime_parity"]["status"] == "PASS"
    assert receipt["gates"]["restart"]["status"] == "PASS"
    assert receipt["gates"]["cash_risk_safety"]["status"] == "PASS"
    assert receipt["gates"]["attribution"]["status"] == "PASS"
    assert receipt["gates"]["execution"]["status"] == "HOLD"
    assert receipt["gates"]["profitability"]["status"] == "HOLD"
    assert receipt["verdict"] == "HOLD"


def test_gold_cli_graduation_branch_never_loads_broker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradebot.research import gold_live_cli

    selected = tmp_path / "selected.json"
    selected.write_text("{}", encoding="utf-8")
    output = tmp_path / "graduation.json"
    monkeypatch.setattr(gold_live_cli, "load_live_capital_plan", lambda _path: {"schema": "old"})
    monkeypatch.setattr(gold_live_cli, "load_gold_live_selection", lambda _path: {})
    monkeypatch.setattr(
        gold_live_cli,
        "LiveCalibrationLedger",
        lambda _path: type("Ledger", (), {"records": lambda self: ()})(),
    )
    monkeypatch.setattr(gold_live_cli, "gold_live_profitability_receipt", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(gold_live_cli, "gold_live_graduation_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(
        gold_live_cli,
        "reduce_live_graduation",
        lambda **_kwargs: {"verdict": "HOLD"},
    )
    monkeypatch.setattr(
        gold_live_cli,
        "publish_live_graduation_receipt",
        lambda path, receipt: path.write_text(json.dumps(receipt)),
    )
    monkeypatch.setattr(
        gold_live_cli,
        "load_config",
        lambda: pytest.fail("graduation queried broker configuration"),
    )
    code = asyncio.run(
        gold_live_cli._main_async(
            [
                "--selection",
                str(selected),
                "--capital-plan",
                str(tmp_path / "capital.json"),
                "--graduation-target",
                "24h",
                "--graduation-cutoff",
                "2026-08-03T13:00:00+00:00",
                "--graduation-output",
                str(output),
            ]
        )
    )
    assert code == 0
    assert json.loads(output.read_text()) == {"verdict": "HOLD"}
