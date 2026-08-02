from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tradebot.live.capital import build_live_capital_plan
from tradebot.live.portfolio import (
    LIVE_CONTROL_REQUEST_SCHEMA,
    LIVE_CONTROL_RECEIPT_SCHEMA,
    LivePortfolioCockpit,
)
from tradebot.live.portfolio_endpoint import LivePortfolioEndpoint
from tradebot.live.runs import LiveRunBinding


RUN_ID = "a" * 64
STRATEGY_ID = "test.strategy.v1"
EXECUTION_VERSION = "test.execution.v1"
CUTOFF = datetime(2026, 8, 2, 1, 2, tzinfo=timezone.utc)


def _unit(unit: str, *, active: bool) -> dict[str, object]:
    return {
        "unit": unit,
        "available": True,
        "load_state": "loaded",
        "active_state": "active" if active else "inactive",
        "sub_state": "waiting" if active else "dead",
        "unit_file_state": "enabled",
        "result": "success",
        "exec_main_status": "0",
        "error": None,
    }


def _write_crown(root: Path, *, symbol: str = "TEST") -> None:
    directory = root / "backtests" / symbol.lower()
    directory.mkdir(parents=True)
    artifact = {
        "groups": [{"_key": "test-crown", "name": "Test Crown", "entries": []}],
        "crown_metrics": {
            "full_three_year": {
                "net_pnl": 123.45,
                "profit_factor": 2.1,
                "max_drawdown": 9.87,
            }
        },
    }
    artifact_path = directory / "crown.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    declaration = {
        "schema": "tradebot.spot.champion.v1",
        "symbol": symbol,
        "track": "LF",
        "version": "1",
        "artifact": artifact_path.relative_to(root).as_posix(),
        "artifact_sha256": artifact_sha,
        "strategy_key": "test-crown",
        "promotion": {
            "eligible": True,
            "order_authority": "none",
        },
    }
    (directory / "current-lf.json").write_text(
        json.dumps(declaration),
        encoding="utf-8",
    )


def _portfolio(
    tmp_path: Path,
) -> tuple[LivePortfolioCockpit, list[tuple[str, ...]]]:
    _write_crown(tmp_path)
    legacy = tmp_path / "backtests" / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "crown.json").write_text(
        json.dumps({"groups": [{"name": "Legacy Crown", "entries": []}]}),
        encoding="utf-8",
    )
    (legacy / "readme-lf.md").write_text(
        "### CURRENT (v1)\n\n- Preset file: `backtests/legacy/crown.json`\n",
        encoding="utf-8",
    )
    selection = {
        "selection_id": RUN_ID,
        "strategy_version": STRATEGY_ID,
        "broker_at_selection": {
            "positions": {"TEST": 0},
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
    context = {
        "directional_impulse": {
            "atr_ratio": 0.7,
            "atr_velocity_pct": -0.01,
            "horizons": [{"elapsed_minutes": 5, "slope_angle_deg": -10.0}],
        },
        "fundamental_pressure": {
            "signed_pressure": -0.8,
            "pressure_delta": 0.1,
            "pressure_velocity_per_hour": 0.02,
        },
    }
    record = {
        "kind": "checkpoint",
        "checkpoint_id": "b" * 64,
        "recorded_at_utc": CUTOFF.isoformat(),
        "strategy_version": EXECUTION_VERSION,
        "evidence": {
            "selection_id": RUN_ID,
            "phase": "STATE",
            "order_ref": "TEST-ORDER",
            "ladder_transition": {
                "previous_mode": "OPT",
                "active_mode": "MID",
                "elapsed_seconds": 6.2,
                "limit_price": 10.01,
                "quote_age_seconds": 0.5,
                "quote_eligible": True,
                "no_progress_reprices": 1,
            },
            "broker_order": {
                "status": "Filled",
                "filled": 1,
                "quantity": 1,
                "average_fill_price": 10.01,
                "fills": [
                    {
                        "commission": 0.35,
                        "commission_currency": "USD",
                    }
                ],
            },
            "plan": {
                "status": "HOLD",
                "reason": "weak_direction",
                "target_direction": None,
                "holdings": {"TEST": 0},
                "execution_state_context": context,
            },
            "broker_state": {
                "positions": {"TEST": 0},
                "open_orders": [],
                "cash_balance_usd": 1_000,
            },
            "risk_state": {
                "valid": True,
                "attribution_complete": True,
                "safety_breaches": [],
                "settled_cash_usd": 1_000,
                "run_net_usd": 0,
                "drawdown_usd": 0,
                "fill_count": 0,
                "closed_trades": 0,
            },
        },
    }
    (tmp_path / "ledger.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    states = {
        "test.timer": _unit("test.timer", active=True),
        "test.service": _unit("test.service", active=False),
    }
    commands: list[tuple[str, ...]] = []

    def command(arguments) -> None:
        args = tuple(arguments)
        commands.append(args)
        if args[:2] == ("disable", "--now"):
            states["test.timer"] = _unit("test.timer", active=False)
        elif args[:2] == ("enable", "--now"):
            states["test.timer"] = _unit("test.timer", active=True)

    binding = LiveRunBinding(
        strategy_id=STRATEGY_ID,
        label="Test durable crown",
        execution_strategy_version=EXECUTION_VERSION,
        ledger_path="ledger.jsonl",
        timer_unit="test.timer",
        service_unit="test.service",
        selection_validator=lambda value: dict(value),
        champion_symbol="TEST",
        champion_track="LF",
    )
    owner = LivePortfolioCockpit(
        repository_root=tmp_path,
        capital_plan_path=tmp_path / "capital.json",
        bindings=(binding,),
        graduation_directory=Path("graduation"),
        graduation_validator=lambda value: dict(value),
        control_ledger_path=Path("control.jsonl"),
        unit_reader=lambda unit: states[unit],
        command_runner=command,
    )
    return owner, commands


def test_catalog_distinguishes_machine_crown_from_readme_provenance(
    tmp_path: Path,
) -> None:
    owner, _commands = _portfolio(tmp_path)

    candidates = {
        (row["symbol"], row["track"]): row for row in owner.snapshot()["candidates"]
    }

    assert candidates[("TEST", "LF")]["stage"] == "CANARY"
    assert candidates[("TEST", "LF")]["machine_authority"] is True
    assert candidates[("TEST", "LF")]["run_id"] == RUN_ID
    assert candidates[("TEST", "LF")]["historical"]["net_pnl"] == 123.45
    assert candidates[("LEGACY", "LF")]["stage"] == "RESEARCH_ONLY"
    assert candidates[("LEGACY", "LF")]["controls"]["COMMISSION"]["status"] == "HOLD"
    assert "legacy_readme_declaration_is_research_only" in candidates[("LEGACY", "LF")]["reasons"]


def test_candidate_commission_never_creates_a_second_order_owner(tmp_path: Path) -> None:
    owner, commands = _portfolio(tmp_path)
    candidate = next(
        row for row in owner.snapshot()["candidates"] if row["symbol"] == "TEST"
    )

    result = owner.commission(str(candidate["candidate_id"]))
    request = result["request"]
    receipt = result["receipt"]

    assert request["schema"] == LIVE_CONTROL_REQUEST_SCHEMA
    assert receipt["schema"] == LIVE_CONTROL_RECEIPT_SCHEMA
    assert receipt["request_id"] == request["request_id"]
    assert receipt["action"] == "COMMISSION"
    assert receipt["decision"] == {
        "status": "NOOP",
        "reasons": ["durable_run_already_active"],
    }
    assert receipt["boundaries"]["ui_broker_client_used"] is False
    assert receipt["boundaries"]["broker_order_submitted"] is False
    assert commands == []
    persisted = [
        json.loads(line) for line in (tmp_path / "control.jsonl").read_text().splitlines()
    ]
    assert persisted == [request, receipt]


def test_control_and_hawkeye_timeline_share_the_persisted_truth(tmp_path: Path) -> None:
    owner, commands = _portfolio(tmp_path)

    stopped = owner.request_control("test-cash", "STOP")
    view = owner.view()
    execution = next(row for row in view["timeline"] if row["kind"] == "EXECUTION")
    control = next(row for row in view["timeline"] if row["kind"] == "CONTROL")

    assert stopped["receipt"]["decision"]["status"] == "ALLOW"
    assert commands == [("disable", "--now", "test.timer")]
    assert execution["reason"] == "weak_direction"
    assert execution["execution_state_context"]["directional_impulse"]["atr_ratio"] == 0.7
    assert execution["execution_state_context"]["fundamental_pressure"]["signed_pressure"] == -0.8
    assert execution["execution_detail"]["order_ref"] == "TEST-ORDER"
    assert execution["execution_detail"]["ladder_transition"]["active_mode"] == "MID"
    assert execution["execution_detail"]["broker_order"]["status"] == "Filled"
    assert control["action"] == "STOP"
    assert control["status"] == "ALLOW"


def test_remote_endpoint_uses_q_without_falling_back_to_local_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def run(arguments, **_kwargs):
        calls.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps({"snapshot": {"status": "READY"}, "timeline": []}),
            stderr="",
        )

    monkeypatch.delenv("TRADEBOT_LIVE_HOST", raising=False)
    monkeypatch.setattr("tradebot.live.portfolio_endpoint.socket.gethostname", lambda: "mac.local")
    monkeypatch.setattr("tradebot.live.portfolio_endpoint.subprocess.run", run)

    endpoint = LivePortfolioEndpoint.default(tmp_path)
    result = endpoint.view(limit=12)

    assert endpoint.host == "q"
    assert result["snapshot"]["status"] == "READY"
    assert calls[0][0:2] == ["ssh", "q"]
    assert "tradebot.live.portfolio_endpoint view --limit 12" in calls[0][2]


def test_q_endpoint_uses_the_same_owner_in_process(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRADEBOT_LIVE_HOST", raising=False)
    monkeypatch.setattr("tradebot.live.portfolio_endpoint.socket.gethostname", lambda: "Q")

    endpoint = LivePortfolioEndpoint.default(tmp_path)

    assert endpoint.host is None
    assert endpoint._owner is not None
