from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.engines.execution import EXECUTION_POLICY
from tradebot.live.execution import LiveOrderExecution, order_ids


def test_order_ids_normalize_broker_identity() -> None:
    trade = SimpleNamespace(order=SimpleNamespace(orderId="42", permId=9001))
    assert order_ids(trade) == (42, 9001)


def test_chase_state_is_shared_and_cleared_across_broker_ids() -> None:
    states: dict[int, dict[str, object]] = {}
    registry = LiveOrderExecution(state_by_order=states)

    state = registry.update_state(
        order_id=42,
        perm_id=9001,
        updates={"active": "RELENTLESS"},
    )

    assert state is states[42] is states[9001]
    assert registry.state(order_id=0, perm_id=9001) is state
    registry.clear_state(order_id=42, perm_id=0)
    assert states == {}


def test_cancel_intent_expires_once_for_both_ids() -> None:
    registry = LiveOrderExecution(state_by_order={}, cancel_ttl_sec=10.0)
    registry.mark_cancel_requested(order_id=42, perm_id=9001, now=100.0)

    assert registry.cancel_requested(order_id=0, perm_id=9001, now=109.9)
    assert not registry.cancel_requested(order_id=42, perm_id=9001, now=110.1)


def test_terminal_trade_without_broker_status_reports_done_once() -> None:
    notices: list[str] = []

    class _Client:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str) -> None:
            return None

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str) -> None:
            return None

    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=11),
        order=SimpleNamespace(orderId=7, permId=0),
        orderStatus=SimpleNamespace(status=""),
        isDone=lambda: True,
    )
    execution = LiveOrderExecution(
        client=_Client(),
        price_for_mode=lambda *_args, **_kwargs: None,
        on_update=lambda _status, notice, _level: notices.append(notice) if notice else None,
        state_by_order={},
    )

    asyncio.run(execution.chase(trade, "BUY", mode="AUTO", policy=EXECUTION_POLICY))

    assert notices == ["#7 Done"]
