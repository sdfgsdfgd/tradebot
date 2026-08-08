from __future__ import annotations

import asyncio
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
