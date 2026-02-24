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


def test_market_hud_unreal_shows_official_and_estimate() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    item = PortfolioItem(
        contract=contract,
        position=2.0,
        marketPrice=90.0,
        marketValue=180.0,
        averageCost=0.0,
        unrealizedPNL=15.0,
        realizedPNL=-3.0,
        account="",
    )

    class _Client:
        @staticmethod
        def pnl_single_unrealized(_con_id: int) -> float | None:
            return 15.0

        @staticmethod
        def pnl_single_daily(_con_id: int) -> float | None:
            return -2.0

    screen = PositionDetailScreen(_Client(), item, refresh_sec=0.25)
    screen._ticker = SimpleNamespace(
        contract=contract,
        bid=99.0,
        ask=101.0,
        last=100.0,
        marketDataType=1,
    )
    lines = screen._render_market_hud(
        panel_width=80,
        bid=99.0,
        ask=101.0,
        last=100.0,
        price=100.0,
        mid=100.0,
        close=98.0,
        mark=100.0,
        spread=2.0,
    )

    tail_row = next(line.plain for line in lines if "✦ Unreal " in line.plain)
    assert "✦ Unreal 15.00 (35.00)" in tail_row
    assert "≈est" not in tail_row


def test_market_hud_unreal_shows_estimate_only_when_official_missing() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150194
    item = PortfolioItem(
        contract=contract,
        position=2.0,
        marketPrice=90.0,
        marketValue=0.0,
        averageCost=80.0,
        unrealizedPNL=None,
        realizedPNL=-3.0,
        account="",
    )

    class _Client:
        @staticmethod
        def pnl_single_unrealized(_con_id: int) -> float | None:
            return None

        @staticmethod
        def pnl_single_daily(_con_id: int) -> float | None:
            return None

    screen = PositionDetailScreen(_Client(), item, refresh_sec=0.25)
    screen._ticker = SimpleNamespace(
        contract=contract,
        bid=99.0,
        ask=101.0,
        last=100.0,
        marketDataType=1,
    )
    lines = screen._render_market_hud(
        panel_width=80,
        bid=99.0,
        ask=101.0,
        last=100.0,
        price=100.0,
        mid=100.0,
        close=98.0,
        mark=100.0,
        spread=2.0,
    )

    tail_row = next(line.plain for line in lines if "✦ Unreal " in line.plain)
    assert "✦ Unreal 40.00 ≈est" in tail_row


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


def test_exec_chase_mode_relentless_delay_has_extended_timeout() -> None:
    assert _exec_chase_mode(100.0, selected_mode="RELENTLESS_DELAY") == "RELENTLESS_DELAY"
    assert _exec_chase_mode(2_000.0, selected_mode="RELENTLESS_DELAY") is None


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

    assert "RLT       B/S" in plain


def test_render_execution_block_includes_relentless_delay_row() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)

    lines = screen._render_execution_block(panel_width=80)
    plain = "\n".join(line.plain for line in lines)

    assert "RLT ⚔ Delay" in plain


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


def test_qty_input_keeps_selected_exec_row_when_not_qty() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    screen._render_details = lambda *args, **kwargs: None  # type: ignore[method-assign]

    cross_idx = screen._exec_rows.index("cross")
    screen._exec_selected = cross_idx
    screen._exec_qty_input = ""

    screen._handle_digit("2")
    screen._handle_digit("5")
    assert screen._exec_selected == cross_idx
    assert screen._exec_qty == 25

    screen._handle_backspace()
    assert screen._exec_selected == cross_idx
    assert screen._exec_qty == 2


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


def test_relentless_delay_price_stays_conservative_on_delayed_wide_spread() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=106.0, ask=115.0, last=110.5, marketDataType=3, minTick=0.5)

    price = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=1,
        delay_sweep_anchor_price=110.5,
    )

    assert price is not None
    assert float(price) <= 111.0


def test_relentless_delay_price_sweeps_optimistic_then_pessimistic() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=106.0, ask=115.0, last=110.5, marketDataType=3, minTick=0.5)

    p1 = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=1,
        delay_sweep_anchor_price=110.5,
    )
    p2 = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=2,
        delay_sweep_anchor_price=110.5,
    )
    p3 = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=3,
        delay_sweep_anchor_price=110.5,
    )
    p4 = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=4,
        delay_sweep_anchor_price=110.5,
    )

    assert p1 is not None and p2 is not None and p3 is not None and p4 is not None
    assert float(p1) < 110.5
    assert float(p2) > 110.5
    assert float(p3) < float(p1)
    assert float(p4) > float(p2)


def test_relentless_delay_anchor_caps_pessimistic_sweep() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=106.0, ask=115.0, last=110.5, marketDataType=3, minTick=0.5)

    capped = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        delay_recoveries=2,
        delay_anchor_price=110.5,
        delay_sweep_anchor_price=110.5,
    )

    assert capped is not None
    assert abs(float(capped) - 110.5) < 1e-9


def test_relentless_delay_matches_relentless_before_any_202() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    ticker = SimpleNamespace(contract=contract, bid=106.0, ask=115.0, last=110.5, marketDataType=3, minTick=0.5)

    relentless = screen._exec_price_for_mode(
        "RELENTLESS",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        elapsed_sec=5.0,
        quote_stale=True,
        open_shock=True,
        no_progress_reprices=1,
        arrival_ref=110.5,
    )
    delay_idle = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        elapsed_sec=5.0,
        quote_stale=True,
        open_shock=True,
        no_progress_reprices=1,
        arrival_ref=110.5,
        delay_recoveries=0,
    )

    assert relentless is not None and delay_idle is not None
    assert abs(float(relentless) - float(delay_idle)) < 1e-9


def test_relentless_delay_locked_direction_biases_relentless_continuation() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    ticker = SimpleNamespace(
        contract=contract,
        bid=106.0,
        ask=115.0,
        last=110.5,
        marketDataType=3,
        minTick=0.5,
        tbTopQuoteUpdatedMono=monotonic() - 16.0,
    )

    unlocked = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        elapsed_sec=8.0,
        quote_stale=True,
        open_shock=True,
        no_progress_reprices=2,
        arrival_ref=110.5,
        delay_recoveries=0,
    )
    locked_down = screen._exec_price_for_mode(
        "RELENTLESS_DELAY",
        "BUY",
        bid=106.0,
        ask=115.0,
        last=110.5,
        ticker=ticker,
        elapsed_sec=8.0,
        quote_stale=True,
        open_shock=True,
        no_progress_reprices=2,
        arrival_ref=110.5,
        delay_recoveries=0,
        delay_locked_price_dir=-1.0,
    )

    assert unlocked is not None and locked_down is not None
    assert float(locked_down) < float(unlocked)


def test_relentless_delay_first_202_reanchors_to_rejected_price() -> None:
    _ensure_event_loop()
    _DETAIL_CHASE_STATE_BY_ORDER.clear()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    loops = {"n": 0}

    class _Client:
        def __init__(self) -> None:
            self.error_served = False
            self._ticker = SimpleNamespace(
                contract=contract,
                bid=106.0,
                ask=115.0,
                last=110.5,
                marketDataType=3,
                minTick=0.5,
                tbTopQuoteUpdatedMono=monotonic() - 16.0,
            )

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        def pop_order_error(self, order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 612
            assert float(max_age_sec) >= 120.0
            if self.error_served:
                return None
            self.error_served = True
            return {"code": 202, "message": "Order price is outside price limits"}

        @staticmethod
        async def modify_limit_order(trade, limit_price: float):
            trade.order.lmtPrice = float(limit_price)
            return trade

    def _is_done() -> bool:
        loops["n"] += 1
        return loops["n"] >= 2

    trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=612,
            permId=0,
            action="BUY",
            totalQuantity=1,
            orderType="LMT",
            lmtPrice=121.0,
        ),
        orderStatus=SimpleNamespace(status="Submitted", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=_is_done,
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    screen._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_SHOCK = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_HYPER = 0.0
    screen._set_chase_state(
        order_id=612,
        perm_id=0,
        updates={
            "delay_recoveries": 0,
            "delay_sweep_anchor_price": 110.5,
            "delay_locked_price_dir": 1.0,
        },
    )
    captured_updates: list[dict[str, object]] = []
    original_set_chase_state = screen._set_chase_state

    def _capture_set_chase_state(
        *,
        order_id: int,
        perm_id: int,
        updates: dict[str, object] | None = None,
    ):
        if updates:
            captured_updates.append(dict(updates))
        return original_set_chase_state(order_id=order_id, perm_id=perm_id, updates=updates)

    screen._set_chase_state = _capture_set_chase_state  # type: ignore[method-assign]
    try:
        asyncio.run(screen._chase_until_filled(trade, "BUY", mode="RELENTLESS_DELAY"))
    finally:
        _DETAIL_CHASE_STATE_BY_ORDER.clear()

    sweep_updates = [
        payload
        for payload in captured_updates
        if "delay_recoveries" in payload and "delay_sweep_anchor_price" in payload
    ]
    assert sweep_updates
    assert any(abs(float(payload["delay_sweep_anchor_price"]) - 121.0) < 1e-9 for payload in sweep_updates)
    assert any(payload.get("delay_locked_price_dir") is None for payload in sweep_updates)


def test_relentless_delay_settle_locks_winning_side_for_directional_relentless() -> None:
    _ensure_event_loop()
    _DETAIL_CHASE_STATE_BY_ORDER.clear()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    done = {"value": False}
    modified_prices: list[float] = []

    class _Client:
        def __init__(self) -> None:
            self._ticker = SimpleNamespace(
                contract=contract,
                bid=106.0,
                ask=115.0,
                last=110.5,
                marketDataType=3,
                minTick=0.5,
                tbTopQuoteUpdatedMono=monotonic() - 16.0,
            )

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        @staticmethod
        def pop_order_error(_order_id: int, *, max_age_sec: float = 120.0):
            assert float(max_age_sec) >= 120.0
            return None

        async def modify_limit_order(self, trade, limit_price: float):
            modified_prices.append(float(limit_price))
            trade.order.lmtPrice = float(limit_price)
            done["value"] = True
            return trade

    trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=613,
            permId=0,
            action="SELL",
            totalQuantity=1,
            orderType="LMT",
            lmtPrice=110.5,
        ),
        orderStatus=SimpleNamespace(status="Submitted", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=lambda: bool(done["value"]),
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    screen._CHASE_MODIFY_ERROR_BACKOFF_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_SHOCK = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_HYPER = 0.0
    screen._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC = 0.0
    screen._set_chase_state(
        order_id=613,
        perm_id=0,
        updates={
            "delay_recoveries": 4,
            "delay_last_202_ts": monotonic() - 8.0,
            "delay_last_leg_sign": 1.0,
            "delay_last_leg_name": "FAV",
            "delay_sweep_anchor_price": 110.5,
        },
    )
    captured_updates: list[dict[str, object]] = []
    original_set_chase_state = screen._set_chase_state

    def _capture_set_chase_state(
        *,
        order_id: int,
        perm_id: int,
        updates: dict[str, object] | None = None,
    ):
        if updates:
            captured_updates.append(dict(updates))
        return original_set_chase_state(order_id=order_id, perm_id=perm_id, updates=updates)

    screen._set_chase_state = _capture_set_chase_state  # type: ignore[method-assign]
    try:
        asyncio.run(screen._chase_until_filled(trade, "SELL", mode="RELENTLESS_DELAY"))
    finally:
        _DETAIL_CHASE_STATE_BY_ORDER.clear()

    assert modified_prices
    assert float(modified_prices[0]) > 106.0
    assert any(payload.get("delay_locked_price_dir") == 1.0 for payload in captured_updates)


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


def test_orders_panel_wraps_long_order_feed_notice() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    class _Client:
        @staticmethod
        def open_trades_for_conids(_con_ids: list[int]):
            return []

    class _RightPane:
        def __init__(self) -> None:
            self.size = SimpleNamespace(width=46, height=24)
            self.value = None

        def update(self, text) -> None:
            self.value = text

    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)
    screen._detail_right = _RightPane()  # type: ignore[attr-defined]
    screen._set_orders_notice(
        "IB 201: We are unable to accept your order because this contract is not available for trading",
        level="error",
    )

    screen._render_orders_panel()

    rendered = screen._detail_right.value  # type: ignore[attr-defined]
    assert rendered is not None
    plain = rendered.plain
    assert "Order Feed" in plain
    assert "ERR IB 201: We are unable to accept your" in plain
    wrapped_lines = [line for line in plain.splitlines() if "order because this contract is not" in line]
    assert wrapped_lines
    assert wrapped_lines[0].startswith("│    ")
    assert "available for trading" in plain


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


def test_relentless_delay_recovers_from_ib202_before_repricing() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5
    done = {"value": False}

    class _Client:
        def __init__(self) -> None:
            self.modify_calls = 0
            self.error_served = False
            self._ticker = SimpleNamespace(
                contract=contract,
                bid=106.0,
                ask=115.0,
                last=110.5,
                marketDataType=3,
                minTick=0.5,
                tbTopQuoteUpdatedMono=monotonic() - 16.0,
            )

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        def pop_order_error(self, order_id: int, *, max_age_sec: float = 120.0):
            assert int(order_id) == 611
            assert float(max_age_sec) >= 120.0
            if self.error_served:
                return None
            self.error_served = True
            return {"code": 202, "message": "Order price is outside price limits"}

        async def modify_limit_order(self, trade, limit_price: float):
            self.modify_calls += 1
            assert float(limit_price) <= 110.5
            trade.order.lmtPrice = float(limit_price)
            done["value"] = True
            return trade

    trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=611,
            permId=0,
            action="BUY",
            totalQuantity=1,
            orderType="LMT",
            lmtPrice=121.0,
        ),
        orderStatus=SimpleNamespace(
            status="Submitted",
            whyHeld="",
            filled=0.0,
            remaining=1.0,
            mktCapPrice=110.5,
        ),
        contract=contract,
        isDone=lambda: bool(done["value"]),
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    screen._CHASE_MODIFY_ERROR_BACKOFF_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_SHOCK = 0.0
    screen._RELENTLESS_MIN_REPRICE_SEC_HYPER = 0.0
    screen._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC = 0.0

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="RELENTLESS_DELAY"))

    assert client.modify_calls >= 1


def test_relentless_delay_wraps_202_sweep_instead_of_halting_at_attempt_cap() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    contract.minTick = 0.5

    class _Client:
        def __init__(self) -> None:
            self.error_budget = 7
            self.errors_served = 0
            self.place_calls = 0
            self._ticker = SimpleNamespace(
                contract=contract,
                bid=106.0,
                ask=115.0,
                last=110.5,
                marketDataType=3,
                minTick=0.5,
                tbTopQuoteUpdatedMono=monotonic() - 16.0,
            )

        async def ensure_ticker(self, _contract, *, owner: str = "details"):
            return self._ticker

        def ticker_for_con_id(self, _con_id: int):
            return self._ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        def pop_order_error(self, _order_id: int, *, max_age_sec: float = 120.0):
            assert float(max_age_sec) >= 120.0
            if self.errors_served >= self.error_budget:
                return None
            self.errors_served += 1
            return {"code": 202, "message": "Order price is outside price limits"}

        async def place_limit_order(
            self,
            order_contract,
            action: str,
            qty: float,
            price: float,
            _outside_rth: bool,
        ):
            self.place_calls += 1
            status = "Filled" if self.place_calls >= self.error_budget else "Cancelled"
            filled = float(qty) if status == "Filled" else 0.0
            remaining = 0.0 if status == "Filled" else float(qty)
            return SimpleNamespace(
                order=SimpleNamespace(
                    orderId=800 + self.place_calls,
                    permId=0,
                    action=action,
                    totalQuantity=float(qty),
                    orderType="LMT",
                    lmtPrice=float(price),
                ),
                orderStatus=SimpleNamespace(
                    status=status,
                    whyHeld="",
                    filled=filled,
                    remaining=remaining,
                    mktCapPrice=0.0,
                ),
                contract=order_contract,
                isDone=lambda resolved=status: resolved == "Filled",
            )

    trade = SimpleNamespace(
        order=SimpleNamespace(
            orderId=799,
            permId=0,
            action="BUY",
            totalQuantity=1.0,
            orderType="LMT",
            lmtPrice=110.5,
        ),
        orderStatus=SimpleNamespace(
            status="Cancelled",
            whyHeld="",
            filled=0.0,
            remaining=1.0,
            mktCapPrice=0.0,
        ),
        contract=contract,
        isDone=lambda: False,
    )
    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract, market_price=110.5), refresh_sec=0.25)
    screen._RELENTLESS_DELAY_RECOVER_ATTEMPTS = 4
    screen._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC = 0.0

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="RELENTLESS_DELAY"))

    assert client.place_calls > int(screen._RELENTLESS_DELAY_RECOVER_ATTEMPTS)
    notice = screen._orders_notice_line()
    if notice is not None:
        assert "Chase halted" not in notice.plain


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
    assert screen._md_probe_requested_type == 3
    assert float(screen._md_probe_started_mono) > 0.0


def test_refresh_ticker_attempts_one_shot_live_probe_for_live_frozen_fop() -> None:
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
                marketDataType=2,
                bid=112.0,
                ask=112.5,
                last=112.25,
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
    assert screen._md_probe_requested_type == 2
    assert float(screen._md_probe_started_mono) > 0.0


def test_market_data_probe_row_shows_transition_and_expires() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    screen = PositionDetailScreen(SimpleNamespace(), _fut_item(contract), refresh_sec=0.25)
    screen._ticker = SimpleNamespace(contract=contract, marketDataType=1)
    screen._md_probe_requested_type = 3
    screen._md_probe_started_mono = monotonic()

    row = screen._market_data_probe_row()
    assert row is not None
    assert "req Delayed (3) -> now Live (1)" in row.plain

    screen._md_probe_started_mono = monotonic() - (screen._MD_PROBE_BANNER_TTL_SEC + 0.5)
    assert screen._market_data_probe_row() is None


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


def test_capture_tick_mid_coalesces_stream_render_updates() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    ticker = SimpleNamespace(
        contract=contract,
        bid=100.0,
        ask=101.0,
        last=100.5,
        lastSize=1.0,
        bidSize=2.0,
        askSize=2.0,
        rtTradeVolume=None,
        rtVolume=None,
        volume=None,
    )

    class _Client:
        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return ticker

    screen = PositionDetailScreen(_Client(), _fut_item(contract), refresh_sec=0.25)
    screen._ticker = ticker
    screen._STREAM_RENDER_DEBOUNCE_SEC = 0.05
    calls = {"n": 0}
    screen._render_details_if_mounted = lambda *args, **kwargs: calls.__setitem__("n", calls["n"] + 1)  # type: ignore[method-assign]

    async def _run() -> int:
        screen._capture_tick_mid()
        screen._capture_tick_mid()
        screen._capture_tick_mid()
        await asyncio.sleep(0.08)
        return int(calls["n"])

    renders = asyncio.run(_run())
    assert renders == 1


def test_orders_panel_probes_effective_status_only_for_pending_rows() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    statuses = ["Submitted", "PendingSubmission", "Filled", "", "ApiPending"]
    trades = [
        SimpleNamespace(
            contract=contract,
            order=SimpleNamespace(orderId=1000 + idx, permId=0, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.0),
            orderStatus=SimpleNamespace(status=status, filled=0.0, remaining=1.0),
        )
        for idx, status in enumerate(statuses)
    ]

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def open_trades_for_conids(self, _con_ids: list[int]):
            return list(trades)

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            self.calls += 1
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": "Submitted",
                "is_terminal": False,
            }

    class _RightPane:
        def __init__(self) -> None:
            self.size = SimpleNamespace(width=110, height=30)

        @staticmethod
        def update(_text) -> None:
            return None

    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)
    screen._detail_right = _RightPane()  # type: ignore[attr-defined]

    screen._render_orders_panel()

    assert client.calls == 3


def test_chase_skips_noop_modify_when_limit_price_unchanged() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    ticker = SimpleNamespace(contract=contract, bid=100.0, ask=101.0, last=100.5)

    loop_count = {"n": 0}

    def _is_done() -> bool:
        loop_count["n"] += 1
        return loop_count["n"] >= 3

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=616, permId=0, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.5),
        orderStatus=SimpleNamespace(status="Submitted", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=_is_done,
    )

    class _Client:
        def __init__(self) -> None:
            self.modify_calls = 0

        @staticmethod
        async def ensure_ticker(_contract, *, owner: str = "details"):
            return ticker

        @staticmethod
        def ticker_for_con_id(_con_id: int):
            return ticker

        @staticmethod
        def release_ticker(_con_id: int, *, owner: str = "details") -> None:
            return None

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": "Submitted",
                "is_terminal": False,
                "filled_qty": 0.0,
            }

        async def modify_limit_order(self, current_trade, _limit_price: float):
            self.modify_calls += 1
            return current_trade

    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="MID"))

    assert client.modify_calls == 0


def test_chase_pending_reconcile_force_calls_are_rate_limited() -> None:
    _ensure_event_loop()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193

    loop_count = {"n": 0}

    def _is_done() -> bool:
        loop_count["n"] += 1
        return loop_count["n"] >= 10

    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=817, permId=0, action="BUY", totalQuantity=1, orderType="LMT", lmtPrice=100.5),
        orderStatus=SimpleNamespace(status="PendingSubmission", whyHeld="", filled=0.0, remaining=1.0),
        contract=contract,
        isDone=_is_done,
    )

    class _Client:
        def __init__(self) -> None:
            self.force_flags: list[bool] = []

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
        def trade_for_order_ids(*, order_id: int = 0, perm_id: int = 0, include_closed: bool = True):
            _ = include_closed
            if int(order_id) == 817:
                return trade
            return None

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": "PendingSubmission",
                "is_terminal": False,
                "filled_qty": 0.0,
                "trade": trade,
            }

        async def reconcile_order_state(self, *, order_id: int = 0, perm_id: int = 0, force: bool = False):
            self.force_flags.append(bool(force))
            return {
                "order_id": int(order_id),
                "perm_id": int(perm_id),
                "effective_status": "PendingSubmission",
                "is_terminal": False,
                "filled_qty": 0.0,
                "trade": trade,
            }

    client = _Client()
    screen = PositionDetailScreen(client, _fut_item(contract), refresh_sec=0.25)
    screen._CHASE_PENDING_ACK_SEC = 0.3
    screen._CHASE_RECONCILE_INTERVAL_SEC = 0.2
    screen._CHASE_FORCE_RECONCILE_INTERVAL_SEC = 0.8

    asyncio.run(screen._chase_until_filled(trade, "BUY", mode="AUTO"))

    assert client.force_flags
    assert any(flag is False for flag in client.force_flags)
    assert any(flag is True for flag in client.force_flags)
    assert sum(1 for flag in client.force_flags if flag) < len(client.force_flags)
