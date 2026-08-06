from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.live.capital import build_live_capital_plan
from tradebot.live.portfolio import (
    LIVE_CONTROL_REQUEST_SCHEMA,
    LIVE_CONTROL_RECEIPT_SCHEMA,
    LivePortfolioCockpit,
)
from tradebot.live.portfolio_endpoint import LivePortfolioEndpoint, _RemoteTransportError
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
            "catalog": {
                "coverage": "complete_available_history",
                "net_pnl": 234.56,
                "profit_factor": 2.2,
                "max_drawdown": 8.76,
            },
            "full_three_year": {
                "net_pnl": 123.45,
                "profit_factor": 2.1,
                "max_drawdown": 9.87,
            }
        },
        "graduation_enrollment": {
            "lifecycle_state": "CROWNED",
            "runtime_parity": "HOLD",
            "native_margin_and_cash": "HOLD",
            "live_24h": "NOT_STARTED",
            "next_gate": "prove runtime",
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
    assert candidates[("TEST", "LF")]["historical"] == {
        "coverage": "complete_available_history",
        "net_pnl": 234.56,
        "profit_factor": 2.2,
        "max_drawdown": 8.76,
    }
    assert candidates[("TEST", "LF")]["qualification"]["next_gate"] == "prove runtime"
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
    trace = view["traces"][0]

    assert stopped["receipt"]["decision"]["status"] == "ALLOW"
    assert commands == [("disable", "--now", "test.timer")]
    assert execution["reason"] == "weak_direction"
    assert "execution_state_context" not in execution
    assert trace["volatility"]["atr_ratio"] == 0.7
    assert trace["news"]["signed_pressure"] == -0.8
    assert execution["execution_detail"]["order_ref"] == "TEST-ORDER"
    assert execution["execution_detail"]["ladder_transition"]["active_mode"] == "MID"
    assert execution["execution_detail"]["broker_order"]["status"] == "Filled"
    assert control["action"] == "STOP"
    assert control["status"] == "ALLOW"


def test_transaction_activity_deduplicates_heartbeats_while_trace_counts_them(
    tmp_path: Path,
) -> None:
    owner, _commands = _portfolio(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    first = json.loads(ledger.read_text())
    repeated = deepcopy(first)
    repeated["checkpoint_id"] = "c" * 64
    repeated["recorded_at_utc"] = (CUTOFF + timedelta(minutes=1)).isoformat()
    ledger.write_text(
        json.dumps(first) + "\n" + json.dumps(repeated) + "\n",
        encoding="utf-8",
    )

    view = owner.view()

    assert len([row for row in view["timeline"] if row["kind"] == "EXECUTION"]) == 1
    assert len(view["traces"]) == 1
    assert view["traces"][0]["sample_count"] == 2


def test_view_separates_timeline_depth_from_cached_trace_history(
    tmp_path: Path,
) -> None:
    owner, _commands = _portfolio(tmp_path)

    first = owner.view(limit=7, trace_limit=2_000)
    second = owner.view(limit=7, trace_limit=2_000)

    assert first["traces"] is second["traces"]
    ledger = tmp_path / "ledger.jsonl"
    repeated = json.loads(ledger.read_text())
    repeated["checkpoint_id"] = "c" * 64
    repeated["recorded_at_utc"] = (CUTOFF + timedelta(minutes=1)).isoformat()
    ledger.write_text(
        ledger.read_text() + json.dumps(repeated) + "\n",
        encoding="utf-8",
    )

    changed = owner.view(limit=7, trace_limit=2_000)

    assert changed["traces"] is not first["traces"]
    assert changed["traces"][0]["sample_count"] == 2


def test_remote_endpoint_uses_q_without_falling_back_to_local_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    requests: list[dict[str, object]] = []

    class _Stdin:
        def __init__(self, process) -> None:
            self.process = process

        def write(self, raw: str) -> None:
            request = json.loads(raw)
            requests.append(request)
            self.process.request = request

        def flush(self) -> None:
            return

        def close(self) -> None:
            self.process.returncode = 0

    class _Process:
        def __init__(self) -> None:
            self.request: dict[str, object] = {}
            self.returncode: int | None = None
            self.stdin = _Stdin(self)
            self.stdout = object()
            self.stderr = None

        def poll(self):
            return self.returncode

        def wait(self, **_kwargs):
            self.returncode = 0
            return 0

    def popen(arguments, **_kwargs):
        calls.append(list(arguments))
        return _Process()

    def read_line(process) -> str:
        request_id = process.request["request_id"]
        previous = process.request.get("previous_view_id")
        result = (
            {"view_id": "stable", "unchanged": True}
            if previous == "stable"
            else {
                "view_id": "stable",
                "unchanged": False,
                "snapshot": {"status": "READY"},
                "timeline": [],
            }
        )
        return json.dumps({"request_id": request_id, "ok": True, "result": result})

    monkeypatch.delenv("TRADEBOT_LIVE_HOST", raising=False)
    monkeypatch.setattr(
        "tradebot.live.portfolio_endpoint.socket.gethostname", lambda: "mac.local"
    )
    monkeypatch.setattr("tradebot.live.portfolio_endpoint.subprocess.Popen", popen)
    monkeypatch.setattr(
        LivePortfolioEndpoint, "_read_remote_line", staticmethod(read_line)
    )

    endpoint = LivePortfolioEndpoint.default(tmp_path)
    result = endpoint.view(limit=12)
    unchanged = endpoint.view(limit=12, previous_view_id=str(result["view_id"]))
    endpoint.close()

    assert endpoint.host == "q"
    assert endpoint._owner is None
    assert result["snapshot"]["status"] == "READY"
    assert unchanged == {"view_id": "stable", "unchanged": True}
    assert calls[0][0:2] == ["ssh", "q"]
    assert "tradebot.live.portfolio_endpoint serve" in calls[0][2]
    assert len(calls) == 1
    assert [request["operation"] for request in requests] == ["view", "view"]
    assert [request["limit"] for request in requests] == [12, 12]
    assert [request["trace_limit"] for request in requests] == [12, 12]


def test_endpoint_returns_component_deltas_from_one_persistent_owner(
    tmp_path: Path,
) -> None:
    class _Owner:
        def __init__(self) -> None:
            self.snapshot_id = "snapshot-a"
            self.trace_id = "trace-a"
            self.calls = 0

        def view(self, *, limit: int, trace_limit: int):
            self.calls += 1
            assert limit == 250
            assert trace_limit == 2_000
            return {
                "snapshot": {"snapshot_id": self.snapshot_id, "status": "READY"},
                "timeline": [{"event_id": "event-a", "kind": "EXECUTION"}],
                "traces": [{"trace_id": self.trace_id}],
            }

    endpoint = LivePortfolioEndpoint(repository_root=tmp_path, host="q")
    owner = _Owner()
    endpoint._owner = owner

    initial = endpoint.view(limit=250, trace_limit=2_000)
    unchanged = endpoint.view(
        limit=250,
        trace_limit=2_000,
        previous_view_id=str(initial["view_id"]),
    )
    owner.snapshot_id = "snapshot-b"
    snapshot_delta = endpoint.view(
        limit=250,
        trace_limit=2_000,
        previous_view_id=str(initial["view_id"]),
    )
    owner.trace_id = "trace-b"
    trace_delta = endpoint.view(
        limit=250,
        trace_limit=2_000,
        previous_view_id=str(snapshot_delta["view_id"]),
    )

    assert initial["snapshot"]["snapshot_id"] == "snapshot-a"
    assert initial["timeline"][0]["event_id"] == "event-a"
    assert initial["traces"] == [{"trace_id": "trace-a"}]
    assert unchanged["unchanged"] is True
    assert "snapshot" not in unchanged and "timeline" not in unchanged
    assert snapshot_delta["snapshot"]["snapshot_id"] == "snapshot-b"
    assert "timeline" not in snapshot_delta
    assert trace_delta["traces"] == [{"trace_id": "trace-b"}]
    assert "snapshot" not in trace_delta and "timeline" not in trace_delta
    assert owner.calls == 4


def test_remote_view_restarts_once_but_control_fails_closed(tmp_path: Path) -> None:
    endpoint = LivePortfolioEndpoint(repository_root=tmp_path, host="q")
    exchanges: list[str] = []
    stops: list[None] = []

    def exchange(request):
        operation = str(request["operation"])
        exchanges.append(operation)
        if len(exchanges) == 1 or operation == "control":
            raise _RemoteTransportError("transport lost")
        return {
            "view_id": "recovered",
            "unchanged": False,
            "snapshot": {"status": "READY"},
            "timeline": [],
        }

    endpoint._exchange_remote = exchange  # type: ignore[method-assign]
    endpoint._stop_remote = lambda: stops.append(None)  # type: ignore[method-assign]

    assert endpoint.view()["view_id"] == "recovered"
    with pytest.raises(RuntimeError, match="transport lost"):
        endpoint.request_control("test", "STOP")

    assert exchanges == ["view", "view", "control"]
    assert len(stops) == 2


def test_q_endpoint_uses_the_same_owner_in_process(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRADEBOT_LIVE_HOST", raising=False)
    monkeypatch.setattr(
        "tradebot.live.portfolio_endpoint.socket.gethostname", lambda: "Q"
    )

    endpoint = LivePortfolioEndpoint.default(tmp_path)

    assert endpoint.host is None
    assert endpoint._owner is not None
