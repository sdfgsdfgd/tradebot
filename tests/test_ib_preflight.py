from __future__ import annotations

import asyncio
import contextlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from ib_insync import Stock

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig
from tradebot.live.ib_preflight import (
    gate_actionable_plan,
    ib_sentinel,
    ib_preflight_decision,
    order_preflight_mode,
    publish_ib_preflight,
    reduce_ib_preflight,
    require_order_preflight,
    validate_ib_preflight,
)


NOW = datetime.now(timezone.utc)


def _facts() -> dict[str, object]:
    return {
        "gateway": {
            "port_accepting": True,
            "api_authenticated": True,
            "expected_account_returned": True,
        },
        "broker": {
            "positions_fresh": True,
            "open_orders_fresh": True,
            "reduction_quote_ready": True,
            "positions": [],
            "open_orders": [],
        },
        "capabilities": [
            {"label": "MCL", "con_id": 661016525, "healthy": True},
        ],
        "connectivity": {
            "losses_10m": 0,
            "restores_10m": 0,
            "unpaired_1100": False,
        },
        "runtime": {
            "expected_members": ["tradebot-mcl-live.timer"],
            "armed_members": ["tradebot-mcl-live.timer"],
            "missing_members": [],
        },
    }


def test_preflight_separates_entry_from_position_reduction() -> None:
    facts = _facts()
    facts["capabilities"][0]["healthy"] = False
    facts["connectivity"]["losses_10m"] = 3
    facts["runtime"]["missing_members"] = ["tradebot-mcl-live.timer"]

    receipt = reduce_ib_preflight(facts, checked_at_utc=NOW)

    assert receipt["verdict"]["entry_ready"] is False
    assert receipt["verdict"]["reduction_ready"] is True
    assert receipt["verdict"]["entry_reasons"] == [
        "connectivity_flap_storm",
        "required_capability_unhealthy:MCL",
        "runtime_member_not_armed:tradebot-mcl-live.timer",
    ]
    assert receipt["boundaries"]["broker_orders_submitted"] == 0
    assert validate_ib_preflight(receipt) == receipt


def test_preflight_fails_both_paths_without_fresh_account_truth() -> None:
    facts = _facts()
    facts["broker"]["open_orders_fresh"] = False
    facts["broker"]["reduction_quote_ready"] = False

    receipt = reduce_ib_preflight(facts, checked_at_utc=NOW)

    assert receipt["verdict"]["entry_ready"] is False
    assert receipt["verdict"]["reduction_ready"] is False
    assert receipt["verdict"]["reduction_reasons"] == [
        "broker_open_orders_not_fresh",
        "held_position_quote_not_ready",
    ]


def test_configured_preflight_is_fresh_tamper_evident_and_gates_only_entry(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "preflight.json"
    receipt = reduce_ib_preflight(_facts(), checked_at_utc=NOW)
    publish_ib_preflight(path, receipt)
    monkeypatch.setenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", str(path))
    monkeypatch.setenv("TRADEBOT_IB_PREFLIGHT_MAX_AGE_SEC", "3600")

    assert ib_preflight_decision("entry", now=NOW)["ready"] is True
    assert ib_preflight_decision("entry", now=NOW + timedelta(hours=2))["ready"] is False

    blocked = deepcopy(receipt)
    blocked["facts"]["runtime"]["missing_members"] = ["test.timer"]
    blocked = reduce_ib_preflight(blocked["facts"], checked_at_utc=NOW)
    publish_ib_preflight(path, blocked)
    plan = {"schema": "test.plan.v1", "status": "ACTIONABLE", "reason": "entry", "leg": {"action": "BUY"}}

    assert gate_actionable_plan(plan, reduction=False)["reason"] == "ib_preflight_entry_not_ready"
    assert gate_actionable_plan(plan, reduction=True) == plan

    tampered = json.loads(path.read_text())
    tampered["verdict"]["entry_ready"] = True
    path.write_text(json.dumps(tampered))
    assert ib_preflight_decision("entry", now=NOW)["ready"] is False


def test_order_gate_allows_only_bounded_position_reduction_when_entry_is_closed(
    tmp_path,
    monkeypatch,
) -> None:
    facts = _facts()
    facts["runtime"]["missing_members"] = ["tradebot-mcl-live.timer"]
    path = tmp_path / "preflight.json"
    publish_ib_preflight(path, reduce_ib_preflight(facts, checked_at_utc=NOW))
    monkeypatch.setenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", str(path))

    assert order_preflight_mode(position=2, action="SELL", quantity=2) == "reduction"
    assert order_preflight_mode(position=-2, action="BUY", quantity=1) == "reduction"
    assert order_preflight_mode(position=2, action="SELL", quantity=3) == "entry"
    assert require_order_preflight(position=2, action="SELL", quantity=1)["mode"] == "reduction"
    with pytest.raises(RuntimeError, match="blocked entry order"):
        require_order_preflight(position=0, action="BUY", quantity=1)


def test_broker_submission_boundary_enforces_the_configured_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    facts = _facts()
    facts["runtime"]["missing_members"] = ["tradebot-mcl-live.timer"]
    path = tmp_path / "preflight.json"
    publish_ib_preflight(path, reduce_ib_preflight(facts, checked_at_utc=NOW))
    monkeypatch.setenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", str(path))

    contract = Stock("SPXU", "SMART", "USD")
    contract.conId = 828937771

    class _IB:
        def __init__(self) -> None:
            self.submitted = []

        @staticmethod
        def positions(_account):
            return [SimpleNamespace(contract=contract, position=2.0)]

        def placeOrder(self, placed_contract, order):
            self.submitted.append((placed_contract, order))
            return SimpleNamespace(contract=placed_contract, order=order)

    client = IBKRClient(
        IBKRConfig(
            host="127.0.0.1",
            port=4001,
            client_id=991,
            proxy_client_id=992,
            account="U1",
            refresh_sec=0.25,
            detail_refresh_sec=0.5,
            reconnect_interval_sec=5,
            reconnect_timeout_sec=60,
            reconnect_slow_interval_sec=60,
            client_id_state_file="",
        )
    )
    broker = _IB()
    client._ib = broker

    async def _connected() -> None:
        return None

    client.connect = _connected
    asyncio.run(client.place_limit_order(contract, "SELL", 1, 20, False))
    with pytest.raises(RuntimeError, match="blocked entry order"):
        asyncio.run(client.place_limit_order(contract, "BUY", 1, 20, False))
    assert len(broker.submitted) == 1


def test_sentinel_ignores_retained_candidate_failure_outside_its_monitor_window(
    tmp_path,
    monkeypatch,
) -> None:
    import tradebot.live.ib_preflight as preflight

    binding = SimpleNamespace(
        strategy_id="mcl.two-speed-shock-arbiter.v112",
        champion_symbol="MCL",
        runtime_timer_units=("tradebot-mcl-live.timer",),
        runtime_service_units=("tradebot-mcl-live.service",),
    )
    monkeypatch.setattr(preflight, "load_live_capital_plan", lambda _path: {"sleeves": []})
    monkeypatch.setattr(preflight, "_selected_runtime_bindings", lambda _plan: [binding])
    monkeypatch.setattr(
        preflight,
        "_unit_liveness",
        lambda unit: {
            "unit": unit,
            "available": "loaded",
            "enabled": "enabled",
            "active": "active" if unit.endswith(".timer") else "failed",
            "result": "success" if unit.endswith(".timer") else "exit-code",
            "next": "Mon 2026-08-10 08:00:10 AEST",
        },
    )

    receipt = ib_sentinel(
        repository_root=tmp_path,
        capital_plan_path=tmp_path / "plan.json",
        receipt_path=tmp_path / "preflight.json",
        login_receipt_path=tmp_path / "login.json",
        state_path=tmp_path / "state.json",
        now=datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc),  # Saturday 10:00 ET
    )

    assert receipt["active_candidates"] == []
    assert receipt["failures"] == []


def test_sentinel_grants_a_continuous_thirty_minute_warmup_before_escalation(
    tmp_path,
    monkeypatch,
) -> None:
    import tradebot.live.ib_preflight as preflight

    binding = SimpleNamespace(
        strategy_id="mcl.two-speed-shock-arbiter.v112",
        champion_symbol="MCL",
        runtime_timer_units=("tradebot-mcl-live.timer",),
        runtime_service_units=("tradebot-mcl-live.service",),
    )
    monkeypatch.setattr(preflight, "load_live_capital_plan", lambda _path: {"sleeves": []})
    monkeypatch.setattr(preflight, "_selected_runtime_bindings", lambda _plan: [binding])

    def liveness(unit: str) -> dict[str, str]:
        if unit.endswith(".timer"):
            return {
                "unit": unit,
                "available": "loaded",
                "enabled": "enabled",
                "active": "active",
                "result": "success",
                "next": "Sun 2026-08-09 18:00:00 EDT",
            }
        return {
            "unit": unit,
            "available": "loaded",
            "enabled": "enabled",
            "active": "active" if unit == "tradebot-ib-gateway.service" else "inactive",
            "result": "success",
            "next": "n/a",
        }

    monkeypatch.setattr(preflight, "_unit_liveness", liveness)
    monkeypatch.setattr(preflight.socket, "create_connection", lambda *_args, **_kwargs: contextlib.nullcontext())
    monkeypatch.setattr(preflight, "_recent_decisive_ib_failure", lambda: False)

    state_path = tmp_path / "sentinel-state.json"
    arguments = {
        "repository_root": tmp_path,
        "capital_plan_path": tmp_path / "plan.json",
        "receipt_path": tmp_path / "missing-preflight.json",
        "login_receipt_path": tmp_path / "login.json",
        "state_path": state_path,
    }
    first = ib_sentinel(
        **arguments,
        now=datetime(2026, 8, 9, 21, 30, tzinfo=timezone.utc),  # Sunday 17:30 ET
    )
    assert first["active_candidates"] == ["mcl"]
    assert first["failures"] == []
    assert first["pending_warmups"] == [
        {
            "reason": "mcl-runtime-failed",
            "detail": "entry_authority_unready",
            "first_unhealthy_at_utc": "2026-08-09T21:30:00+00:00",
            "age_sec": 0.0,
            "grace_remaining_sec": 1800.0,
        }
    ]

    due = ib_sentinel(
        **arguments,
        now=datetime(2026, 8, 9, 22, 0, tzinfo=timezone.utc),  # Sunday 18:00 ET
    )
    assert due["pending_warmups"] == []
    assert due["failures"] == [
        {
            "reason": "mcl-runtime-failed",
            "detail": "entry_authority_unready",
            "first_unhealthy_at_utc": "2026-08-09T21:30:00+00:00",
            "age_sec": 1800.0,
        }
    ]
