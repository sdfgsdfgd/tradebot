from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
import types

from ib_insync import Contract, PortfolioItem
from textual.app import App, ComposeResult

if "tradebot.ui.bot_runtime" not in sys.modules:
    bot_runtime_stub = types.ModuleType("tradebot.ui.bot_runtime")

    class _BotRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def install(self, *_args, **_kwargs) -> None:
            return None

    bot_runtime_stub.BotRuntime = _BotRuntime  # type: ignore[attr-defined]
    sys.modules["tradebot.ui.bot_runtime"] = bot_runtime_stub

from tradebot.ui.app import PositionsApp
from tradebot.ui.footer import TradebotFooter


def _item(contract: Contract) -> PortfolioItem:
    return PortfolioItem(
        contract=contract,
        position=1.0,
        marketPrice=1.0,
        marketValue=1.0,
        averageCost=1.0,
        unrealizedPNL=0.0,
        realizedPNL=0.0,
        account="DU123456",
    )


def test_portfolio_item_for_contract_prefers_matching_fut_expiry() -> None:
    feb = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    feb.lastTradeDateOrContractMonth = "202602"
    feb.conId = 2001
    apr = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    apr.lastTradeDateOrContractMonth = "202604"
    apr.conId = 2002
    target = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    target.lastTradeDateOrContractMonth = "202604"

    fake_self = SimpleNamespace(_snapshot=SimpleNamespace(items=[_item(feb), _item(apr)]))
    result = PositionsApp._portfolio_item_for_contract(fake_self, target)

    assert int(getattr(getattr(result, "contract", None), "conId", 0) or 0) == 2002


def test_portfolio_item_for_contract_prefers_matching_option_strike() -> None:
    low = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        right="C",
        strike=4950.0,
    )
    low.conId = 3001
    hi = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        right="C",
        strike=5000.0,
    )
    hi.conId = 3002
    target = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        right="C",
        strike=5000.0,
    )

    fake_self = SimpleNamespace(_snapshot=SimpleNamespace(items=[_item(low), _item(hi)]))
    result = PositionsApp._portfolio_item_for_contract(fake_self, target)

    assert int(getattr(getattr(result, "contract", None), "conId", 0) or 0) == 3002


def test_entry_now_holds_last_valid_values_through_sparse_portfolio_update() -> None:
    contract = Contract(secType="STK", symbol="TQQQ", exchange="SMART", currency="USD")
    contract.conId = 72539702
    item = SimpleNamespace(contract=contract, averageCost=70.2714, position=1.0)
    marks = iter((74.71, None))
    probe = SimpleNamespace(
        _position_entry_by_con_id={},
        _last_position_mark_by_con_id={},
        _POSITION_MARK_STICKY_SEC=20.0,
    )
    probe._float_or_none = PositionsApp._float_or_none
    probe._mark_price = lambda _item: (next(marks), False)

    initial = PositionsApp._entry_now_inputs(probe, item)
    item.averageCost = None
    sparse = PositionsApp._entry_now_inputs(probe, item)

    assert initial[:2] == sparse[:2] == (70.2714, 74.71)


def test_home_footer_exposes_only_high_value_product_actions() -> None:
    visible = [binding for binding in PositionsApp.BINDINGS if binding.show]
    hidden = [binding for binding in PositionsApp.BINDINGS if not binding.show]

    assert [binding.description for binding in visible] == [
        "Quit",
        "Refresh",
        "Details",
        "Favorites",
        "Search",
        "Bot",
    ]
    assert {binding.action for binding in hidden} == {"cursor_down", "cursor_up"}
    assert {binding.action: binding.key_display for binding in visible} == {
        "quit": None,
        "refresh": None,
        "open_details": None,
        "open_favorites": None,
        "toggle_search": "⌃F",
        "toggle_bot": "⌃T",
    }


def test_tradebot_footer_is_compact_and_omits_generic_palette() -> None:
    class FooterProofApp(App):
        CSS = PositionsApp.CSS
        BINDINGS = PositionsApp.BINDINGS

        def compose(self) -> ComposeResult:
            yield TradebotFooter()

        def action_refresh(self) -> None:
            return

        def action_cursor_down(self) -> None:
            return

        def action_cursor_up(self) -> None:
            return

        def action_open_details(self) -> None:
            return

        def action_open_favorites(self) -> None:
            return

        def action_toggle_search(self) -> None:
            return

        def action_toggle_bot(self) -> None:
            return

    async def exercise() -> None:
        app = FooterProofApp()
        async with app.run_test(size=(100, 4)) as pilot:
            await pilot.pause()
            footer = app.query_one(TradebotFooter)
            screenshot = app.export_screenshot()

            assert footer.compact is True
            assert footer.show_command_palette is False
            assert len(footer.query("FooterKey")) == 6
            assert "Details" in screenshot
            assert "Favorites" in screenshot
            assert "Down" not in screenshot
            assert "Up" not in screenshot
            assert "palette" not in screenshot.lower()

    asyncio.run(exercise())
