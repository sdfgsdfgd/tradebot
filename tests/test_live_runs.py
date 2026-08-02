from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradebot.live.capital import build_live_capital_plan
from tradebot.live.runs import LiveRunBinding, LiveRunCockpit, read_systemd_user_unit
from tradebot.research.live_graduation import (
    LIVE_GRADUATION_PREFIX_SCHEMA,
    publish_live_graduation_receipt,
    reduce_live_graduation,
    validate_live_graduation_receipt,
)


RUN_ID = "a" * 64
STRATEGY_ID = "test.strategy.v1"
EXECUTION_VERSION = "test.strategy.execution.v1"
CUTOFF = datetime(2026, 8, 1, 20, 18, tzinfo=timezone.utc)


def _gate() -> dict[str, object]:
    return {"status": "PASS", "reasons": [], "evidence": {"proof": True}}


def _receipt() -> dict[str, object]:
    subject = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": EXECUTION_VERSION,
        "signal_instrument": "TEST",
        "execution_sleeve": "TEST/INVERSE",
        "capital_sleeve": "test-cash",
        "selection_id": RUN_ID,
        "run_id": RUN_ID,
        "account_fingerprint": "b" * 64,
    }
    return reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        subject=subject,
        selection={
            "selection_id": RUN_ID,
            "run_id": RUN_ID,
            "capital_sleeve": "test-cash",
            "selection_file_sha256": "c" * 64,
        },
        selection_file_sha256="c" * 64,
        ledger_prefix={
            "schema": LIVE_GRADUATION_PREFIX_SCHEMA,
            "cutoff_utc": CUTOFF.isoformat(),
            "candidate_records": 1,
            "included_records": 1,
            "excluded_for_dependency": 0,
            "clock_regressions": 0,
            "first_record_id": "d" * 64,
            "last_record_id": "d" * 64,
            "sha256": "e" * 64,
            "gates": {
                "restart": _gate(),
                "cash_risk_safety": _gate(),
                "attribution": _gate(),
                "execution": _gate(),
            },
        },
        profitability_receipt={
            "as_of_utc": CUTOFF.isoformat(),
            "status": "PASSED",
            "policy": {
                "run_id": RUN_ID,
                "strategy_id": STRATEGY_ID,
                "strategy_version": EXECUTION_VERSION,
                "capital_sleeve": "test-cash",
            },
            "milestones": {
                "24h": {
                    "passed": True,
                    "reasons": [],
                    "evidence_as_of_utc": CUTOFF.isoformat(),
                }
            },
            "reasons": [],
        },
        runtime_parity_proof=_gate(),
        capital_owner_stability_proof=_gate(),
    )


def _unit(unit: str, *, active: bool, service: bool = False) -> dict[str, object]:
    return {
        "unit": unit,
        "available": True,
        "load_state": "loaded",
        "active_state": "active" if active else "inactive",
        "sub_state": "running" if active and service else "waiting" if active else "dead",
        "unit_file_state": "static" if service else "enabled",
        "result": "success",
        "exec_main_status": "0",
        "error": None,
    }


def _fixture(
    tmp_path: Path,
    *,
    phase: str = "STATE",
) -> tuple[LiveRunCockpit, dict[str, dict[str, object]], list[tuple[str, ...]]]:
    selection = {
        "selection_id": RUN_ID,
        "strategy_version": STRATEGY_ID,
        "broker_at_selection": {
            "positions": {"TEST": 0, "INVERSE": 0},
            "open_orders": [],
            "settled_cash_usd": 1_000,
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    plan = build_live_capital_plan(
        account_id="DU123",
        account_type="CASH",
        currency="USD",
        observed_settled_cash_usd=1_000,
        managed_capital_usd=800,
        sleeves=[
            {
                "sleeve_id": "test-cash",
                "strategy_id": STRATEGY_ID,
                "run_id": RUN_ID,
                "selection_path": selection_path.name,
                "selection_file_sha256": selection_sha,
                "capital_kind": "CASH_DEBIT",
                "weight_bps": 10_000,
            }
        ],
        reserve_reasons=["operator_reserve"],
        created_at_utc=CUTOFF,
    )
    (tmp_path / "capital.json").write_text(json.dumps(plan), encoding="utf-8")
    evidence = {
        "selection_id": RUN_ID,
        "phase": phase,
        "order_ref": "ORDER-1" if phase != "STATE" else "",
        "plan": {"holdings": {"TEST": 0, "INVERSE": 0}},
        "broker_state": {
            "positions": {"TEST": 0, "INVERSE": 0},
            "open_orders": [],
            "cash_balance_usd": 1_000,
        },
        "risk_state": {
            "valid": True,
            "attribution_complete": True,
            "safety_breaches": [],
            "settled_cash_usd": 1_000,
            "run_net_usd": 12.5,
            "run_realized_net_usd": 10.0,
            "open_mark_net_usd": 2.5,
            "run_cost_usd": 0.8,
            "drawdown_usd": 3.0,
            "fill_count": 2,
            "closed_trades": 1,
        },
        "submitted_orders": 0,
    }
    record = {
        "kind": "checkpoint",
        "checkpoint_id": "f" * 64,
        "recorded_at_utc": CUTOFF.isoformat(),
        "strategy_id": STRATEGY_ID,
        "strategy_version": EXECUTION_VERSION,
        "evidence": evidence,
    }
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    graduation = tmp_path / "graduation"
    graduation.mkdir()
    publish_live_graduation_receipt(
        graduation / f"{RUN_ID}.24h.20260801T201800Z.json",
        _receipt(),
    )
    binding = LiveRunBinding(
        strategy_id=STRATEGY_ID,
        label="Test durable champion",
        execution_strategy_version=EXECUTION_VERSION,
        ledger_path="ledger.jsonl",
        timer_unit="test.timer",
        service_unit="test.service",
        selection_validator=lambda value: dict(value),
    )
    states = {
        "test.timer": _unit("test.timer", active=True),
        "test.service": _unit("test.service", active=False, service=True),
    }
    calls: list[tuple[str, ...]] = []

    def command(arguments) -> None:
        args = tuple(arguments)
        calls.append(args)
        if args[:2] == ("disable", "--now"):
            states["test.timer"] = _unit("test.timer", active=False)
        elif args[:2] == ("enable", "--now"):
            states["test.timer"] = _unit("test.timer", active=True)

    cockpit = LiveRunCockpit(
        repository_root=tmp_path,
        capital_plan_path=tmp_path / "capital.json",
        bindings=(binding,),
        graduation_directory=Path("graduation"),
        graduation_validator=validate_live_graduation_receipt,
        unit_reader=lambda unit: states[unit],
        command_runner=command,
    )
    return cockpit, states, calls


def test_cockpit_projects_one_official_selected_run_without_broker_authority(
    tmp_path: Path,
) -> None:
    cockpit, _states, _calls = _fixture(tmp_path)

    snapshot = cockpit.snapshot()
    run = snapshot["runs"][0]

    assert snapshot["status"] == "READY"
    assert snapshot["capital"]["unallocated_reserve_cents"] == 20_000
    assert run["state"] == "RUNNING"
    assert run["allocation"] == {"weight_bps": 10_000, "limit_cents": 80_000}
    assert run["positions"] == {"TEST": 0, "INVERSE": 0}
    assert run["economics"]["run_net_usd"] == 12.5
    assert run["graduation"]["verdict"] == "PROMOTE"
    assert run["controls"]["START"]["status"] == "NOOP"
    assert run["controls"]["STOP"]["status"] == "ALLOW"
    assert run["controls"]["REPLACE"]["status"] == "HOLD"
    assert run["controls"]["REBALANCE"]["status"] == "HOLD"


def test_selection_drift_quarantines_only_its_allocated_run(tmp_path: Path) -> None:
    cockpit, _states, _calls = _fixture(tmp_path)
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")

    snapshot = cockpit.snapshot()
    run = snapshot["runs"][0]

    assert snapshot["status"] == "QUARANTINED"
    assert run["valid"] is False
    assert run["state"] == "QUARANTINED"
    assert "selection file identity changed" in run["errors"][0]
    assert run["controls"]["START"]["status"] == "HOLD"


def test_pending_transition_prevents_schedule_pause(tmp_path: Path) -> None:
    cockpit, _states, calls = _fixture(tmp_path, phase="SUBMITTED")

    run = cockpit.snapshot()["runs"][0]

    assert run["pending_order_refs"] == ["ORDER-1"]
    assert run["controls"]["STOP"] == {
        "status": "HOLD",
        "reasons": ["unreconciled_transition_present"],
    }
    with pytest.raises(ValueError, match="unreconciled_transition"):
        cockpit.control("test-cash", "STOP")
    assert calls == []


def test_start_and_flat_stop_only_mutate_the_bound_timer(tmp_path: Path) -> None:
    cockpit, states, calls = _fixture(tmp_path)

    stopped = cockpit.control("test-cash", "STOP")
    assert stopped["after"]["runs"][0]["state"] == "PAUSED"
    assert calls == [("disable", "--now", "test.timer")]

    started = cockpit.control("test-cash", "START")
    assert started["after"]["runs"][0]["state"] == "RUNNING"
    assert calls[-1] == ("enable", "--now", "test.timer")
    assert states["test.service"]["active_state"] == "inactive"


def test_replace_and_rebalance_require_immutable_successor_artifacts(
    tmp_path: Path,
) -> None:
    cockpit, _states, calls = _fixture(tmp_path)

    with pytest.raises(ValueError, match="successor_selection"):
        cockpit.control("test-cash", "REPLACE")
    with pytest.raises(ValueError, match="successor_capital_plan"):
        cockpit.control("test-cash", "REBALANCE")
    assert calls == []


def test_systemd_read_failure_becomes_unavailable_truth(monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("systemctl", 5)

    monkeypatch.setattr(subprocess, "run", timeout)

    state = read_systemd_user_unit("test.timer")

    assert state["available"] is False
    assert state["active_state"] == "unknown"
    assert "timed out" in str(state["error"])


def test_unchanged_append_only_ledger_is_parsed_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cockpit, _states, _calls = _fixture(tmp_path)
    reads = 0
    original = Path.read_text

    def counted(path: Path, *args, **kwargs):
        nonlocal reads
        if path == tmp_path / "ledger.jsonl":
            reads += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)

    cockpit.snapshot()
    cockpit.snapshot()

    assert reads == 1
