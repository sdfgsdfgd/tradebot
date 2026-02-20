from __future__ import annotations

import asyncio
from types import SimpleNamespace
from time import monotonic

from ib_insync import Contract, PortfolioItem

from tradebot.ui.common import _exec_chase_mode
from tradebot.ui.positions import PositionDetailScreen, _DETAIL_CHASE_STATE_BY_ORDER


def _fut_item(contract: Contract, *, position: float = 0.0, market_price: float = 5000.0) -> PortfolioItem:
    return PortfolioItem(
        contract=contract,
        position=position,
        marketPrice=market_price,
        marketValue=0.0,
        averageCost=0.0,
        unrealizedPNL=0.0,
        realizedPNL=0.0,
        account="",
    )


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_render_execution_block_supports_fut_contracts() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    lines = screen._render_execution_block(panel_width=80)

    assert lines
    assert "Execution Ladder" in lines[0].plain


def test_exec_chase_mode_relentless_has_extended_timeout() -> None:
    assert _exec_chase_mode(100.0, selected_mode="RELENTLESS") == "RELENTLESS"
    assert _exec_chase_mode(2_000.0, selected_mode="RELENTLESS") is None


def test_render_execution_block_includes_custom_price_row() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    lines = screen._render_execution_block(panel_width=80)
    plain = "\n".join(line.plain for line in lines)

    assert "CUSTOM" in plain


def test_render_execution_block_includes_relentless_row() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    lines = screen._render_execution_block(panel_width=80)
    plain = "\n".join(line.plain for line in lines)

    assert "RLT Fill" in plain


def test_submit_order_does_not_reject_fut_as_unsupported() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]

    screen._submit_order("BUY")

    assert screen._exec_status != "Exec: unsupported contract"


def test_custom_price_input_sets_custom_exec_mode_price() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    screen._ticker = SimpleNamespace(
        contract=contract,
        bid=112.0,
        ask=113.0,
        last=112.5,
    )
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]
    screen._exec_selected = screen._exec_rows.index("custom")

    for char in "112.5":
        screen._handle_digit(char)

    assert screen._selected_exec_mode() == "CUSTOM"
    price = screen._initial_exec_price("BUY", mode="CUSTOM")
    assert price is not None
    assert abs(float(price) - 112.5) < 1e-9


def test_relentless_price_escalates_over_time() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=100.0, ask=101.0, last=100.5)

    p0 = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=100.0,
        ask=101.0,
        last=100.5,
        ticker=ticker,
        elapsed_sec=1.0,
        quote_stale=False,
        open_shock=False,
        no_progress_reprices=0,
        arrival_ref=100.5,
    )
    p1 = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=100.0,
        ask=101.0,
        last=100.5,
        ticker=ticker,
        elapsed_sec=10.0,
        quote_stale=False,
        open_shock=False,
        no_progress_reprices=0,
        arrival_ref=100.5,
    )
    p2 = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=100.0,
        ask=101.0,
        last=100.5,
        ticker=ticker,
        elapsed_sec=10.0,
        quote_stale=True,
        open_shock=True,
        no_progress_reprices=4,
        arrival_ref=100.5,
    )

    assert p0 is not None and p1 is not None and p2 is not None
    assert float(p1) > float(p0)
    assert float(p2) > float(p1)


def test_relentless_spread_pressure_boosts_target() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=100.0, ask=103.0, last=101.5)

    screen._spread_samples.clear()
    baseline = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=100.0,
        ask=103.0,
        last=101.5,
        ticker=ticker,
        elapsed_sec=6.0,
        quote_stale=False,
        open_shock=False,
        no_progress_reprices=0,
        arrival_ref=101.5,
    )
    screen._spread_samples.extend([0.25] * 24)
    pressure = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=100.0,
        ask=103.0,
        last=101.5,
        ticker=ticker,
        elapsed_sec=6.0,
        quote_stale=False,
        open_shock=False,
        no_progress_reprices=0,
        arrival_ref=101.5,
    )

    assert baseline is not None and pressure is not None
    assert float(pressure) > float(baseline)


def test_relentless_min_reprice_hyper_in_open_shock_when_no_progress() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    interval = screen._relentless_min_reprice_sec(
        quote_stale=False,
        open_shock=True,
        no_progress_reprices=3,
        spread_pressure=1.0,
    )

    assert interval <= float(screen._RELENTLESS_MIN_REPRICE_SEC_HYPER)


def test_align_front_future_rebinds_flat_search_item() -> None:
    _ensure_event_loop()
    start = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    start.lastTradeDateOrContractMonth = "202602"
    start.conId = 1001
    front = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    front.lastTradeDateOrContractMonth = "202604"
    front.conId = 1002

    class _Client:
        async def front_future(
            self,
            symbol: str,
            *,
            exchange: str = "CME",
            cache_ttl_sec: float = 3600.0,
        ) -> Contract | None:
            assert symbol == "GC"
            assert exchange in ("COMEX", "CME")
            assert cache_ttl_sec == 180.0
            return front

        def portfolio_item(self, con_id: int):
            assert con_id == 1002
            return None

    screen = PositionDetailScreen(_Client(), _fut_item(start, position=0.0, market_price=0.0), refresh_sec=0.25)
    asyncio.run(screen._maybe_align_front_future_contract())

    assert int(getattr(screen._item.contract, "conId", 0) or 0) == 1002


def test_orders_panel_renders_order_feed_notice() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    class _Client:
        @staticmethod
        def open_trades_for_conids(_con_ids: list[int]):
            return []

    class _RightPane:
        def __init__(self) -> None:
            self.size = SimpleNamespace(width=80, height=24)
            self.value = None

        def update(self, text) -> None:
            self.value = text

    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)
    screen._detail_right = _RightPane()  # type: ignore[attr-defined]
    screen._set_orders_notice("IB 201: bad tif", level="error")

    screen._render_orders_panel()

    rendered = screen._detail_right.value  # type: ignore[attr-defined]
    assert rendered is not None
    plain = rendered.plain
    assert "Order Feed" in plain
    assert "IB 201: bad tif" in plain


def test_orders_panel_notice_replaces_previous_and_expires() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    screen._set_orders_notice("old msg", level="warn")
    screen._set_orders_notice("new msg", level="error")
    line = screen._orders_notice_line()
    assert line is not None
    assert "new msg" in line.plain
    assert "old msg" not in line.plain

    screen._orders_notice = (
        monotonic() - float(screen._ORDER_PANEL_NOTICE_TTL_SEC) - 1.0,
        "error",
        "stale",
    )
    assert screen._orders_notice_line() is None
    assert screen._orders_notice is None


def test_chase_inactive_surfaces_ib_reject_in_order_feed() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    class _Client:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return None

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        @staticmethod
        def pop_order_error(order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 777
            assert float(max_age_sec) >= 120.0
            return {"code": 201, "message": "bad tif"}

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=777, permId=0),
        orderStatus=SimpleNamespace(status="Inactive", whyHeld=""),
        contract=contract,
        isDone=lambda: False,
    )
    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)

    asyncio.run(screen._chase_until_filled(trade, "SELL", mode="AUTO"))

    line = screen._orders_notice_line()
    assert line is not None
    assert "#777 Inactive: IB 201: bad tif" in line.plain


def test_chase_done_after_pending_surfaces_ib_reject_in_order_feed() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    class _Client:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return None

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        @staticmethod
        def pop_order_error(order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 888
            assert float(max_age_sec) >= 120.0
            return {"code": 201, "message": "bad tif"}

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=888, permId=0),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld=""),
        contract=contract,
        isDone=lambda: True,
    )
    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)

    asyncio.run(screen._chase_until_filled(trade, "SELL", mode="AUTO"))

    line = screen._orders_notice_line()
    assert line is not None
    assert "#888 Done (PendingSubmission): IB 201: bad tif" in line.plain


def test_chase_pending_submission_surfaces_ib_reject_without_terminal_status() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        def __init__(self) -> None:
            self._served = False

        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return None

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        def pop_order_error(self, order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 991
            assert float(max_age_sec) >= 120.0
            if self._served:
                return None
            self._served = True
            return {"code": 110, "message": "price does not conform to minimum increment"}

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=991, permId=0),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld=""),
        contract=contract,
        isDone=lambda: False,
    )
    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="AUTO"))

    line = screen._orders_notice_line()
    assert line is not None
    assert "#991 PendingSubmission: IB 110: price does not conform to minimum increment" in line.plain


def test_refresh_ticker_attempts_one_shot_live_probe_for_delayed_fop() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        def __init__(self) -> None:
            self.live_snapshot_calls = 0

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return SimpleNamespace(
                contract=contract,
                marketDataType=3,
                bid=None,
                ask=None,
                last=None,
                close=112.0,
                prevLast=112.0,
            )

        async def refresh_live_snapshot_once(self, _contract) -> str | None:
            self.live_snapshot_calls += 1
            return "live-snapshot"

    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]

    asyncio.run(screen.action_refresh_ticker())

    assert client.live_snapshot_calls == 1
    assert "1-shot live-snapshot" in str(screen._exec_status or "")


def test_chase_pending_submission_does_not_modify_before_accept() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        def __init__(self) -> None:
            self.modify_calls = 0

        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return None

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        async def modify_limit_order(self, trade, limit_price: float):
            self.modify_calls += 1
            return trade

    loop_count = {"n": 0}

    def _is_done() -> bool:
        loop_count["n"] += 1
        return loop_count["n"] >= 2

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=999, permId=0),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld=""),
        contract=contract,
        isDone=_is_done,
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="AUTO"))

    assert client.modify_calls == 0


def test_chase_pending_submission_reconciles_to_submitted_then_modifies() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    done = {"value": False}

    class _Client:
        def __init__(self) -> None:
            self.modify_calls = 0
            self.reconcile_calls = 0
            self._ticker = SimpleNamespace(contract=contract, bid=100.0, ask=101.0, last=100.5)

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        async def reconcile_order_state(self, *, order_id: int = 0, perm_id: int = 0, force: bool = False):
            self.reconcile_calls += 1
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": "Submitted",
                "is_terminal": False,
                "filled_qty": 0.0,
            }

        async def modify_limit_order(self, current_trade, limit_price: float):
            self.modify_calls += 1
            assert float(limit_price) > 0
            current_trade.orderStatus.status = "Submitted"
            done["value"] = True
            return current_trade

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=744, permId=100744, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.5),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=lambda: bool(done["value"]),
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)
    screen._CHASE_PENDING_ACK_SEC = 0.0
    screen._CHASE_RECONCILE_INTERVAL_SEC = 0.0

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="AUTO"))

    assert client.reconcile_calls >= 1
    assert client.modify_calls >= 1


def test_order_line_shows_effective_status_hint_when_diverged() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    trade = SimpleNamespace(
        contract=contract,
        order=SimpleNamespace(orderId=321, permId=0, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.0),
        orderStatus=SimpleNamespace(status="PendingSubmission", filled=0.0, remaining=1.0),
    )
    screen._set_chase_state(
        order_id=321,
        perm_id=0,
        updates={
            "selected": "AUTO",
            "active": "MID",
            "effective_status": "Submitted",
        },
    )

    line = screen._format_order_line(trade, width=160)

    assert "PendingSu" in line.plain
    assert "~Submitted" in line.plain
    screen._clear_chase_state(order_id=321, perm_id=0)


def test_order_line_uses_client_current_effective_status_hint() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        @staticmethod
        def current_order_state(*, order_id: int = 0, perm_id: int = 0):
            assert int(order_id) == 322
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "raw_status": "PendingSubmission",
                "effective_status": "Submitted",
                "is_terminal": False,
            }

    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)
    trade = SimpleNamespace(
        contract=contract,
        order=SimpleNamespace(orderId=322, permId=0, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.0),
        orderStatus=SimpleNamespace(status="PendingSubmission", filled=0.0, remaining=1.0),
    )

    line = screen._format_order_line(trade, width=160)

    assert "PendingSu" in line.plain
    assert "~Submitted" in line.plain


def test_cancel_order_surfaces_cancel_not_found_error() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        @staticmethod
        async def cancel_trade(_trade) -> None:
            return None

        @staticmethod
        def pop_order_error(order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 733
            assert float(max_age_sec) > 0
            return {"code": 10147, "message": "order to cancel not found"}

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=733, permId=0),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld=""),
        contract=contract,
        isDone=lambda: False,
    )
    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]

    asyncio.run(screen._cancel_order(trade))

    line = screen._orders_notice_line()
    assert line is not None
    assert "Cancel #733: IB 10147: order to cancel not found" in line.plain


def test_action_cancel_order_halts_relentless_chase_repricing_before_ack() -> None:
    _ensure_event_loop()
    _DETAIL_CHASE_STATE_BY_ORDER.clear()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    class _Client:
        def __init__(self, trade) -> None:
            self._trade = trade
            self.modify_calls = 0
            self.cancel_calls = 0
            self._tick = 0
            self._ticker = SimpleNamespace(contract=trade.contract, bid=100.0, ask=100.5, last=100.25)

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            self._tick += 1
            bid = 100.0 + float((self._tick % 5) * 0.25)
            self._ticker.bid = bid
            self._ticker.ask = bid + 0.5
            self._ticker.last = bid + 0.25
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        async def modify_limit_order(self, trade, _limit_price: float):
            self.modify_calls += 1
            trade.orderStatus.status = "Submitted"
            return trade

        async def cancel_trade(self, _trade) -> None:
            self.cancel_calls += 1

        @staticmethod
        def pop_order_error(order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 360
            assert float(max_age_sec) > 0
            return None

        def trade_for_order_ids(self, *, order_id: int = 0, perm_id: int = 0, include_closed: bool = True):
            _ = include_closed
            if int(order_id) == 360:
                return self._trade
            return None

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": str(self._trade.orderStatus.status),
                "filled_qty": 0.0,
                "is_terminal": False,
                "trade": self._trade,
            }

        async def reconcile_order_state(self, *, order_id: int = 0, perm_id: int = 0, force: bool = False):
            _ = force
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": str(self._trade.orderStatus.status),
                "filled_qty": 0.0,
                "is_terminal": False,
                "trade": self._trade,
            }

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=360, permId=0, action="SELL", totalQuantity=1, orderType="LMT", lmtPrice=0.88),
        orderStatus=SimpleNamespace(status="Submitted", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=lambda: False,
    )
    client = _Client(trade)
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]
    screen._render_details_if_mounted = lambda *args, **kwargs: None  # type: ignore[method-assign]
    screen._RELENTLESS_MIN_REPRICE_SEC = 0.1
    screen._RELENTLESS_MIN_REPRICE_SEC_SHOCK = 0.1
    screen._RELENTLESS_MIN_REPRICE_SEC_HYPER = 0.1
    screen._active_panel = "orders"
    screen._orders_rows = [trade]
    screen._orders_selected = 0

    async def _run() -> tuple[int, int]:
        chase = asyncio.create_task(screen._chase_until_filled(trade, "SELL", mode="RELENTLESS"))
        screen._chase_tasks.add(chase)
        screen._register_chase_task(chase, order_id=360, perm_id=0)
        await asyncio.sleep(0.9)
        before = int(client.modify_calls)
        screen.action_cancel_order()
        await asyncio.sleep(0.9)
        after = int(client.modify_calls)
        if not chase.done():
            chase.cancel()
        try:
            await chase
        except BaseException:
            pass
        return before, after

    before_cancel, after_cancel = asyncio.run(_run())

    assert client.cancel_calls >= 1
    assert after_cancel == before_cancel


def test_chase_state_survives_order_id_transition_from_perm_id() -> None:
    _ensure_event_loop()
    _DETAIL_CHASE_STATE_BY_ORDER.clear()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    screen._set_chase_state(
        order_id=0,
        perm_id=456001,
        updates={
            "selected": "AUTO",
            "active": "OPT",
            "target_price": 5022.0,
            "mods": 0,
        },
    )
    screen._set_chase_state(
        order_id=2619,
        perm_id=456001,
        updates={
            "active": "MID",
            "target_price": 5022.25,
        },
    )
    transitioned = SimpleNamespace(
        order=SimpleNamespace(orderId=2619, permId=456001),
        orderStatus=SimpleNamespace(status="Submitted"),
        contract=contract,
    )

    line = screen._active_chase_line([transitioned])

    assert "Chase #2619" in line.plain
    assert "AUTO->MID" in line.plain
    assert _DETAIL_CHASE_STATE_BY_ORDER.get(2619) is _DETAIL_CHASE_STATE_BY_ORDER.get(456001)
    screen._clear_chase_state(order_id=2619, perm_id=456001)


def test_place_order_surfaces_chase_task_exception_in_order_feed() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=314, permId=2718),
        orderStatus=SimpleNamespace(status="Submitted", whyHeld=""),
        contract=contract,
        isDone=lambda: False,
    )

    class _Client:
        @staticmethod
        async def place_limit_order(
            _contract,
            _action: str,
            _qty: int,
            _price: float,
            _outside_rth: bool,
        ):
            return trade

        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return None

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            raise RuntimeError("md boom")

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)

    async def _run() -> None:
        await screen._place_order("SELL", 1, 5022.0, False, "AUTO")
        await asyncio.sleep(0.05)

    asyncio.run(_run())

    line = screen._orders_notice_line()
    assert line is not None
    assert "#314 chase task error: md boom" in line.plain
