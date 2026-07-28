from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.engines.execution import EXECUTION_POLICY
from tradebot.engines.execution import _EXEC_AUTO_TIMEOUT_SEC
from tradebot.engines.execution import _EXEC_RELENTLESS_TIMEOUT_SEC
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


def test_resumed_chase_preserves_original_timeout_budget() -> None:
    cancelled: list[int] = []

    class _Client:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str) -> None:
            return None

        @staticmethod
        async def cancel_trade(trade) -> None:
            cancelled.append(int(trade.order.orderId))

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str) -> None:
            return None

    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=11),
        order=SimpleNamespace(orderId=7, permId=0),
        orderStatus=SimpleNamespace(status="Submitted", filled=0),
        isDone=lambda: False,
    )
    execution = LiveOrderExecution(
        client=_Client(),
        price_for_mode=lambda *_args, **_kwargs: None,
        state_by_order={},
    )

    asyncio.run(
        execution.chase(
            trade,
            "BUY",
            mode="AUTO",
            policy=EXECUTION_POLICY,
            elapsed_offset_sec=_EXEC_AUTO_TIMEOUT_SEC + 1,
        )
    )

    assert cancelled == [7]


def test_fresh_top_contract_pauses_stale_repricing_until_timeout() -> None:
    cancelled: list[int] = []
    modified: list[float] = []
    refreshed: list[int] = []
    transitions: list[dict[str, object]] = []
    ticker = SimpleNamespace(
        bid=30.48,
        ask=30.50,
        last=30.49,
        marketDataType=3,
        tbTopQuoteUpdatedMono=None,
    )

    class _Client:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str) -> None:
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return ticker

        @staticmethod
        async def refresh_live_snapshot_once(_contract) -> None:
            refreshed.append(1)

        @staticmethod
        async def modify_limit_order(trade, price: float):
            modified.append(price)
            return trade

        @staticmethod
        async def cancel_trade(trade) -> None:
            cancelled.append(int(trade.order.orderId))

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str) -> None:
            return None

    trade = SimpleNamespace(
        contract=SimpleNamespace(conId=11, secType="STK", minTick=0.01),
        order=SimpleNamespace(orderId=7, permId=0, lmtPrice=30.50),
        orderStatus=SimpleNamespace(status="Submitted", filled=0),
        isDone=lambda: False,
    )
    execution = LiveOrderExecution(
        client=_Client(),
        price_for_mode=lambda *_args, **_kwargs: 30.60,
        on_transition=transitions.append,
        state_by_order={},
    )

    asyncio.run(
        execution.chase(
            trade,
            "BUY",
            mode="RELENTLESS",
            policy=EXECUTION_POLICY,
            elapsed_offset_sec=_EXEC_RELENTLESS_TIMEOUT_SEC - 0.1,
            require_fresh_top=True,
        )
    )

    assert refreshed
    assert modified == []
    assert cancelled == [7]
    assert len(transitions) == 1
    assert transitions[0]["event"] == "ladder_mode_transition"
    assert transitions[0]["active_mode"] == "RLT"
    assert transitions[0]["previous_mode"] is None
    assert transitions[0]["quote_eligible"] is False
    assert transitions[0]["resumed"] is True
