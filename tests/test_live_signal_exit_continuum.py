from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
import types

from ib_insync import Stock

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig

_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

from tradebot.ui.bot_engine_runtime import BotEngineRuntimeMixin
from tradebot.ui.bot_models import _BotInstance, _BotOrder
from tradebot.ui.bot_signal_runtime import BotSignalRuntimeMixin


@dataclass(frozen=True)
class _RawBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _new_client() -> IBKRClient:
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=123,
        proxy_client_id=124,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
    )
    return IBKRClient(cfg)


def test_historical_full24_stitches_overnight_and_keeps_what_to_show_cache_separate() -> None:
    client = _new_client()
    calls: list[tuple[str, str, bool]] = []

    async def _fake_request(
        contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        calls.append((str(getattr(contract, "exchange", "") or ""), str(what_to_show), bool(use_rth)))
        stamp_overnight = datetime(2026, 2, 8, 21, 15)
        stamp_rth = datetime(2026, 2, 9, 10, 0)
        exchange = str(getattr(contract, "exchange", "") or "").upper()
        if exchange == "OVERNIGHT":
            if str(what_to_show).upper() == "MIDPOINT":
                return [_RawBar(stamp_overnight, 220.0, 220.0, 220.0, 220.0, 20.0)]
            return [_RawBar(stamp_overnight, 210.0, 210.0, 210.0, 210.0, 10.0)]
        if str(what_to_show).upper() == "MIDPOINT":
            return [_RawBar(stamp_rth, 120.0, 120.0, 120.0, 120.0, 2.0)]
        return [
            _RawBar(stamp_overnight, 101.0, 101.0, 101.0, 101.0, 1.0),
            _RawBar(stamp_rth, 111.0, 111.0, 111.0, 111.0, 1.0),
        ]

    client._request_historical_data = _fake_request  # type: ignore[method-assign]

    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    contract.conId = 1001

    trades_bars = asyncio.run(
        client.historical_bars_ohlcv(
            contract,
            duration_str="2 D",
            bar_size="5 mins",
            use_rth=False,
            what_to_show="TRADES",
            cache_ttl_sec=60.0,
        )
    )
    midpoint_bars = asyncio.run(
        client.historical_bars_ohlcv(
            contract,
            duration_str="2 D",
            bar_size="5 mins",
            use_rth=False,
            what_to_show="MIDPOINT",
            cache_ttl_sec=60.0,
        )
    )

    by_ts_trades = {bar.ts: bar for bar in trades_bars}
    by_ts_midpoint = {bar.ts: bar for bar in midpoint_bars}
    overnight_ts = datetime(2026, 2, 8, 21, 15)
    assert by_ts_trades[overnight_ts].close == 210.0
    assert by_ts_midpoint[overnight_ts].close == 220.0

    assert any(what == "TRADES" for _, what, _ in calls)
    assert any(what == "MIDPOINT" for _, what, _ in calls)
    assert any(exchange == "OVERNIGHT" and what == "TRADES" for exchange, what, _ in calls)
    assert any(exchange == "OVERNIGHT" and what == "MIDPOINT" for exchange, what, _ in calls)


class _ExitGateHarness(BotSignalRuntimeMixin):
    pass


def _new_instance(*, strategy: dict | None = None) -> _BotInstance:
    return _BotInstance(
        instance_id=1,
        group="g",
        symbol="SLV",
        strategy=dict(strategy or {}),
        filters=None,
    )


def test_exit_gate_blocks_same_bar_after_fill_marker_only() -> None:
    harness = _ExitGateHarness()
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.last_exit_bar_ts = bar_ts
    gates: list[str] = []

    async def _run() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates.append(str(status)),
        )

    assert asyncio.run(_run()) is False
    assert "BLOCKED_EXIT_SAME_BAR" in gates


def test_exit_gate_blocks_retry_limit_and_cooldown() -> None:
    harness = _ExitGateHarness()
    bar_ts = datetime(2026, 2, 9, 10, 0)

    instance_limit = _new_instance(strategy={"exit_retry_max_per_bar": 2})
    instance_limit.exit_retry_bar_ts = bar_ts
    instance_limit.exit_retry_count = 2
    gates_limit: list[str] = []

    async def _run_limit() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance_limit,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates_limit.append(str(status)),
        )

    assert asyncio.run(_run_limit()) is False
    assert "BLOCKED_EXIT_RETRY_LIMIT" in gates_limit

    instance_cd = _new_instance(strategy={"exit_retry_max_per_bar": 3})
    instance_cd.exit_retry_bar_ts = bar_ts
    instance_cd.exit_retry_count = 1
    instance_cd.exit_retry_cooldown_until = datetime(2026, 2, 9, 10, 5)
    gates_cd: list[str] = []

    async def _run_cd() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance_cd,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates_cd.append(str(status)),
        )

    assert asyncio.run(_run_cd()) is False
    assert "BLOCKED_EXIT_RETRY_COOLDOWN" in gates_cd


class _FakeTrade:
    def __init__(self, status: str) -> None:
        self.orderStatus = SimpleNamespace(status=status)
        self.fills = []

    def isDone(self) -> bool:
        return True


class _EngineHarness(BotEngineRuntimeMixin):
    def __init__(self, *, instance: _BotInstance, order: _BotOrder) -> None:
        self._orders = [order]
        self._instances = [instance]
        self._last_chase_ts = -1_000_000_000.0
        self._active_panel = "instances"
        self._order_rows = []
        self._events: list[tuple[str, dict]] = []

    async def _reprice_order(self, order: _BotOrder, *, mode: str) -> bool:
        return False

    def _journal_write(self, *, event: str, data: dict | None = None, **kwargs) -> None:
        self._events.append((str(event), dict(data or {})))

    def _refresh_orders_table(self) -> None:
        return None


def _new_order(*, status: str, signal_bar_ts: datetime) -> _BotOrder:
    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    return _BotOrder(
        instance_id=1,
        preset=None,
        underlying=contract,
        order_contract=contract,
        legs=[],
        action="SELL",
        quantity=1,
        limit_price=1.0,
        created_at=datetime(2026, 2, 9, 10, 0),
        status="WORKING",
        intent="exit",
        signal_bar_ts=signal_bar_ts,
        trade=_FakeTrade(status),
    )


def test_engine_inactive_exit_marks_retryable_and_sets_cooldown() -> None:
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance(strategy={"exit_retry_cooldown_sec": 2})
    order = _new_order(status="Inactive", signal_bar_ts=bar_ts)
    harness = _EngineHarness(instance=instance, order=order)

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "INACTIVE"
    assert instance.exit_retry_bar_ts == bar_ts
    assert instance.exit_retry_count == 1
    assert instance.exit_retry_cooldown_until is not None
    assert any(event == "ORDER_DONE" and data.get("retryable") is True for event, data in harness._events)


def test_engine_filled_exit_sets_same_bar_lock_and_clears_retry_state() -> None:
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.exit_retry_bar_ts = bar_ts
    instance.exit_retry_count = 2
    instance.exit_retry_cooldown_until = datetime(2026, 2, 9, 10, 5)
    order = _new_order(status="Filled", signal_bar_ts=bar_ts)
    harness = _EngineHarness(instance=instance, order=order)

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "FILLED"
    assert instance.last_exit_bar_ts == bar_ts
    assert instance.exit_retry_bar_ts is None
    assert instance.exit_retry_count == 0
    assert instance.exit_retry_cooldown_until is None
    assert any(event == "ORDER_FILLED" and data.get("exit_lock_bar_ts") for event, data in harness._events)
