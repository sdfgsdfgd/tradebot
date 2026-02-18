from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

from ib_insync import Contract


def _load_positions_app():
    # Python 3.9 test runners choke while importing tradebot.ui.bot (Pydantic-style
    # `type | None` in class bases). We only need PositionsApp formatting helpers,
    # so stub bot_runtime before importing app.py.
    bot_runtime_stub = types.ModuleType("tradebot.ui.bot_runtime")

    class _BotRuntime:
        def __init__(self, *_args, **_kwargs):
            pass

        def install(self, *_args, **_kwargs):
            pass

    bot_runtime_stub.BotRuntime = _BotRuntime
    sys.modules["tradebot.ui.bot_runtime"] = bot_runtime_stub

    sys.modules.pop("tradebot.ui.app", None)
    module = importlib.import_module("tradebot.ui.app")
    return module.PositionsApp


def test_search_contract_description_option_includes_contract_identity() -> None:
    positions_app = _load_positions_app()
    contract = Contract(
        secType="OPT",
        symbol="BITU",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=29.73,
        right="C",
        localSymbol="BITU  260220C00029730",
    )
    contract.conId = 123456

    text = positions_app._search_contract_description(contract)

    assert "BITU 20260220 CALL 29.73" in text
    assert "BITU  260220C00029730" in text
    assert "SMART" in text
    assert "conId 123456" in text


def test_search_contract_description_option_falls_back_without_local_symbol() -> None:
    positions_app = _load_positions_app()
    contract = Contract(
        secType="OPT",
        symbol="AAPL",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260320",
        strike=190.0,
        right="P",
    )

    text = positions_app._search_contract_description(contract)

    assert text.startswith("AAPL 20260320 PUT 190.00")
    assert "conId" not in text


def test_portfolio_row_key_distinguishes_call_and_put_without_conid() -> None:
    positions_app = _load_positions_app()
    call_contract = Contract(
        secType="OPT",
        symbol="SLV",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260320",
        strike=23.0,
        right="C",
    )
    put_contract = Contract(
        secType="OPT",
        symbol="SLV",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260320",
        strike=23.0,
        right="P",
    )

    call_key = positions_app._portfolio_row_key(SimpleNamespace(contract=call_contract))
    put_key = positions_app._portfolio_row_key(SimpleNamespace(contract=put_contract))

    assert call_key != put_key


def test_portfolio_row_key_prefers_conid_when_present() -> None:
    positions_app = _load_positions_app()
    contract = Contract(
        secType="OPT",
        symbol="SLV",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260320",
        strike=23.0,
        right="P",
    )
    contract.conId = 991122

    key = positions_app._portfolio_row_key(SimpleNamespace(contract=contract))

    assert key == "OPT:991122"


def test_opt_underlyer_description_prefers_human_label() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    app._search_opt_underlyers = ["BITU"]
    app._search_opt_underlyer_index = 0
    app._search_opt_underlyer_descriptions = {"BITU": "ProShares Ultra Bitcoin ETF"}

    assert app._current_opt_underlyer_description() == "ProShares Ultra Bitcoin ETF"


def test_opt_underlyer_description_hides_symbol_only_label() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    app._search_opt_underlyers = ["BITU"]
    app._search_opt_underlyer_index = 0
    app._search_opt_underlyer_descriptions = {"BITU": "BITU"}

    assert app._current_opt_underlyer_description() == ""
