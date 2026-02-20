from __future__ import annotations

import sys
from types import SimpleNamespace
import types

from ib_insync import Contract, PortfolioItem

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
