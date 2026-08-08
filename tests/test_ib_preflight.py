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
            "reduction_quotes": {},
            "positions": [],
            "open_orders": [],
        },
        "capabilities": [
            {
                "label": "MCL",
                "con_id": 661016525,
                "healthy": True,
                "sleeve_id": "mcl-two-speed-auction-margin",
                "strategy_id": "mcl.two-speed-shock-arbiter.v112",
            },
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
            "members_by_sleeve": {
                "mcl-two-speed-auction-margin": ["tradebot-mcl-live.timer"],
            },
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


def test_reduction_readiness_is_scoped_to_requested_held_contracts(
    tmp_path,
) -> None:
    facts = _facts()
    facts["broker"].update(
        {
            "positions": [
                {"con_id": 661016525, "quantity": 1.0},
                {"con_id": 828937771, "quantity": 23.0},
                {"con_id": 898708983, "quantity": 1.0},
            ],
            "reduction_quote_ready": False,
            "reduction_quotes": {
                "661016525": True,
                "828937771": False,
                "898708983": False,
            },
        }
    )
    path = tmp_path / "preflight.json"
    publish_ib_preflight(path, reduce_ib_preflight(facts, checked_at_utc=NOW))

    assert ib_preflight_decision("reduction", path=path)["ready"] is False
    assert ib_preflight_decision(
        "reduction",
        path=path,
        con_ids=(661016525,),
    ) == {
        "configured": True,
        "ready": True,
        "reasons": [],
        "receipt_id": json.loads(path.read_text())["receipt_id"],
        "con_ids": [661016525],
    }
    blocked = ib_preflight_decision(
        "reduction",
        path=path,
        con_ids=(61228752, 828937771),
    )
    assert blocked["ready"] is False
    assert blocked["reasons"] == ["held_position_quote_not_ready:828937771"]
    assert ib_preflight_decision(
        "reduction",
        path=path,
        con_ids=(61228752,),
    )["ready"] is True


def test_entry_readiness_is_scoped_to_the_selected_contract_and_bundle(
    tmp_path,
) -> None:
    facts = _facts()
    facts["capabilities"].extend(
        [
            {
                "label": "1OZ",
                "con_id": 753716623,
                "healthy": True,
                "sleeve_id": "gold-1oz-stage76-margin",
                "strategy_id": "gold.1oz-regime-harmony-stage76.v1",
            },
        ]
    )
    facts["capabilities"][0]["healthy"] = False
    facts["runtime"].update(
        {
            "expected_members": [
                "tradebot-gold-live.timer",
                "tradebot-mcl-live.timer",
            ],
            "armed_members": ["tradebot-gold-live.timer"],
            "missing_members": ["tradebot-mcl-live.timer"],
            "members_by_sleeve": {
                "gold-1oz-stage76-margin": ["tradebot-gold-live.timer"],
                "mcl-two-speed-auction-margin": ["tradebot-mcl-live.timer"],
            },
        }
    )
    path = tmp_path / "preflight.json"
    publish_ib_preflight(path, reduce_ib_preflight(facts, checked_at_utc=NOW))

    assert ib_preflight_decision("entry", path=path)["ready"] is False
    assert ib_preflight_decision(
        "entry",
        path=path,
        con_ids=(753716623,),
    )["ready"] is True
    blocked = ib_preflight_decision(
        "entry",
        path=path,
        con_ids=(661016525,),
    )
    assert blocked["ready"] is False
    assert blocked["reasons"] == [
        "required_capability_unhealthy:MCL",
        "runtime_member_not_armed:tradebot-mcl-live.timer",
    ]
    unavailable = ib_preflight_decision(
        "entry",
        path=path,
        con_ids=(999999999,),
    )
    assert unavailable["ready"] is False
    assert unavailable["reasons"] == [
        "required_capability_scope_unavailable:999999999",
        "runtime_scope_unavailable",
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
    facts["broker"].update(
        {
            "positions": [
                {"con_id": 828937771, "quantity": 2.0},
                {"con_id": 661016525, "quantity": 1.0},
            ],
            "reduction_quote_ready": False,
            "reduction_quotes": {
                "828937771": True,
                "661016525": False,
            },
        }
    )
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


def test_gold_monitor_window_matches_the_official_1oz_maintenance_clock() -> None:
    import tradebot.live.ib_preflight as preflight

    binding = SimpleNamespace(
        strategy_id="gold.1oz-regime-harmony-stage76.v1",
    )

    assert preflight._candidate_monitor_window_open(
        binding,
        datetime(2026, 8, 8, 6, 20, tzinfo=timezone.utc),
    )
    assert not preflight._candidate_monitor_window_open(
        binding,
        datetime(2026, 8, 8, 7, 30, tzinfo=timezone.utc),
    )
    assert preflight._candidate_monitor_window_open(
        binding,
        datetime(2026, 8, 8, 9, 1, tzinfo=timezone.utc),
    )
    assert not preflight._candidate_monitor_window_open(
        binding,
        datetime(2026, 8, 10, 21, 1, tzinfo=timezone.utc),
    )
    assert preflight._candidate_monitor_window_open(
        binding,
        datetime(2026, 8, 10, 21, 2, tzinfo=timezone.utc),
    )


def test_sentinel_reports_planned_gold_recovery_without_false_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import tradebot.live.ib_preflight as preflight

    strategy_id = "gold.1oz-regime-harmony-stage76.v1"
    binding = SimpleNamespace(
        strategy_id=strategy_id,
        champion_symbol="1OZ",
        timer_unit="tradebot-gold-live.timer",
        runtime_timer_units=(
            "tradebot-gold-live.timer",
            "tradebot-gold-onset.timer",
        ),
        runtime_service_units=(
            "tradebot-gold-live.service",
            "tradebot-gold-onset.service",
        ),
        recovery_timer_unit="tradebot-gold-fail-closed-rollover.timer",
        recovery_managed_timer_units=("tradebot-gold-live.timer",),
    )
    plan = {"sleeves": [{"sleeve_id": "gold", "strategy_id": strategy_id}]}
    monkeypatch.setattr(preflight, "load_live_capital_plan", lambda _path: plan)
    monkeypatch.setattr(preflight, "_selected_runtime_bindings", lambda _plan: [binding])
    monkeypatch.setattr(
        preflight,
        "_contract_specs",
        lambda *_args, **_kwargs: [{"con_id": 753716608}],
    )

    def liveness(unit: str) -> dict[str, str]:
        timer = unit.endswith(".timer")
        disabled = unit == "tradebot-gold-live.timer"
        gateway = unit == "tradebot-ib-gateway.service"
        return {
            "unit": unit,
            "available": "loaded",
            "enabled": "disabled" if disabled else "enabled" if timer else "static",
            "active": "active" if gateway or timer and not disabled else "inactive",
            "result": "success",
            "next": "n/a" if disabled or not timer else "Mon 2026-08-10 10:12:00 AEST",
        }

    monkeypatch.setattr(preflight, "_unit_liveness", liveness)
    monkeypatch.setattr(
        preflight,
        "ib_preflight_decision",
        lambda *_args, **_kwargs: {
            "ready": False,
            "reasons": ["runtime_member_not_armed:tradebot-gold-live.timer"],
        },
    )
    monkeypatch.setattr(
        preflight.socket,
        "create_connection",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(preflight, "_recent_decisive_ib_failure", lambda: False)

    receipt = ib_sentinel(
        repository_root=tmp_path,
        capital_plan_path=tmp_path / "plan.json",
        receipt_path=tmp_path / "preflight.json",
        login_receipt_path=tmp_path / "login.json",
        state_path=tmp_path / "state.json",
        now=datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc),
    )

    assert receipt["active_candidates"] == ["1oz"]
    assert receipt["planned_recoveries"] == [
        {
            "candidate": "1oz",
            "timer": "tradebot-gold-fail-closed-rollover.timer",
            "next": "Mon 2026-08-10 10:12:00 AEST",
            "managed_timers": ["tradebot-gold-live.timer"],
        }
    ]
    assert receipt["failures"] == []
    assert receipt["pending_warmups"] == []


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
