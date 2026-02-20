from __future__ import annotations

import asyncio
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


def test_search_contract_description_option_includes_label_when_provided() -> None:
    positions_app = _load_positions_app()
    contract = Contract(
        secType="FOP",
        symbol="MCL",
        exchange="NYMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=66.0,
        right="P",
    )

    text = positions_app._search_contract_description(contract, label="Micro WTI Crude Oil")

    assert "MCL 20260220 PUT 66.00  •  Micro WTI Crude Oil" in text


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


def test_search_label_for_symbol_returns_mode_fallback_label() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    app._search_symbol_labels = {}
    app._search_opt_underlyer_descriptions = {}

    assert app._search_label_for_symbol("AAPL", sec_type="STK") == "Equity"
    assert app._search_label_for_symbol("MNQ", sec_type="FUT") == "Futures"
    assert app._search_label_for_symbol("AAPL", sec_type="OPT") == "Equity Option"
    assert app._search_label_for_symbol("MCL", sec_type="FOP") == "Futures Option"


def test_option_pair_rows_supports_fop_mode() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    app._search_mode_index = positions_app._SEARCH_MODES.index("FOP")
    app._search_opt_expiry_index = 0
    app._search_results = []
    app._search_selected = 0
    app._search_side = 0
    call_contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        strike=5000.0,
        right="C",
    )
    put_contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        strike=5000.0,
        right="P",
    )
    app._search_results = [call_contract, put_contract]

    rows = app._option_pair_rows()

    assert len(rows) == 1
    row_call, row_put = rows[0]
    assert row_call is not None
    assert row_put is not None
    assert str(getattr(row_call, "right", "") or "").upper() == "C"
    assert str(getattr(row_put, "right", "") or "").upper() == "P"
    assert app._search_row_count() == 1


def test_selected_search_contract_uses_side_in_fop_mode() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    app._search_mode_index = positions_app._SEARCH_MODES.index("FOP")
    app._search_opt_expiry_index = 0
    app._search_selected = 0
    app._search_side = 1
    call_contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        strike=5000.0,
        right="C",
    )
    put_contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260327",
        strike=5000.0,
        right="P",
    )
    app._search_results = [call_contract, put_contract]

    selected = app._selected_search_contract()

    assert selected is not None
    assert str(getattr(selected, "right", "") or "").upper() == "P"


def test_render_search_hides_no_option_rows_while_loading() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    rendered: list[object] = []
    app._search_active = True
    app._search = SimpleNamespace(display=False, update=lambda value: rendered.append(value))
    app._search_mode_index = positions_app._SEARCH_MODES.index("OPT")
    app._search_query = "NVDA"
    app._search_error = None
    app._search_loading = True
    app._search_results = []
    app._search_side = 0
    app._search_scroll = 0
    app._search_selected = 0
    app._search_opt_expiry_index = 0
    app._search_opt_underlyers = []
    app._search_opt_underlyer_index = 0
    app._search_opt_underlyer_descriptions = {}
    app._search_symbol_labels = {}
    app._search_timing = {"generation": 1}
    app._search_name_line = lambda: None
    app._search_timing_line = lambda: None
    app._sync_search_option_tickers = lambda: None
    app._clear_search_tickers = lambda: None

    app._render_search()

    assert rendered
    rendered_text = str(getattr(rendered[-1], "plain", rendered[-1]))
    assert "Searching..." in rendered_text
    assert "No option chain rows" not in rendered_text


def test_run_search_opt_initial_query_uses_single_pass_contract_lookup() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    generation = 14
    app._SEARCH_DEBOUNCE_SEC = 0.0
    app._search_generation = generation
    app._search_mode_index = positions_app._SEARCH_MODES.index("OPT")
    app._search_loading = True
    app._search_error = None
    app._search_results = []
    app._search_selected = 0
    app._search_scroll = 0
    app._search_side = 0
    app._search_opt_expiry_index = 0
    app._search_query = "NVDA"
    app._search_timing = {"generation": generation}
    app._search_opt_underlyers = []
    app._search_opt_underlyer_descriptions = {}
    app._search_symbol_labels = {}
    app._search_opt_underlyer_index = 0
    app._search_opt_chain_cache = {}
    app._render_search = lambda: None
    app._ensure_search_visible = lambda: None
    app._default_opt_row_index = lambda: 0

    call_contract = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="C",
    )
    call_contract.conId = 998001
    put_contract = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="P",
    )
    put_contract.conId = 998002
    seen_search_kwargs: dict[str, object] = {}

    async def _search_option_underlyers(
        _query: str,
        *,
        limit: int,
        timing: dict[str, object] | None = None,
    ):
        assert limit == positions_app._SEARCH_OPT_UNDERLYER_LIMIT
        if timing is not None:
            timing.update({"total_ms": 1.0, "source": "direct"})
        return [("NVDA", "Equity Option")]

    async def _search_contracts(
        _query: str,
        *,
        mode: str,
        limit: int,
        opt_underlyer_symbol: str | None = None,
        timing: dict[str, object] | None = None,
        **kwargs,
    ) -> list[Contract]:
        assert mode == "OPT"
        assert limit == positions_app._SEARCH_OPT_FETCH_LIMIT
        assert str(opt_underlyer_symbol or "").upper() == "NVDA"
        seen_search_kwargs.clear()
        seen_search_kwargs.update(kwargs)
        if timing is not None:
            timing.update(
                {
                    "candidate_count": 2,
                    "qualified_count": 2,
                    "stage": "done",
                    "reason": "ok",
                    "total_ms": 20.0,
                }
            )
        return [call_contract, put_contract]

    app._client = SimpleNamespace(
        search_option_underlyers=_search_option_underlyers,
        search_contracts=_search_contracts,
    )

    asyncio.run(
        app._run_search(
            generation,
            "NVDA",
            "OPT",
            fetch_limit=positions_app._SEARCH_OPT_FETCH_LIMIT,
        )
    )

    assert "opt_first_limit" not in seen_search_kwargs
    assert "opt_progress" not in seen_search_kwargs
    assert len(app._search_results) == 2
    assert len(app._search_opt_chain_cache.get("NVDA", [])) == 2
    assert app._search_loading is False


def test_run_search_opt_underlyer_uses_single_pass_contract_lookup() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    generation = 11
    app._search_generation = generation
    app._search_mode_index = positions_app._SEARCH_MODES.index("OPT")
    app._search_loading = True
    app._search_error = None
    app._search_results = []
    app._search_selected = 0
    app._search_scroll = 0
    app._search_side = 0
    app._search_opt_expiry_index = 0
    app._search_opt_chain_cache = {}
    app._search_timing = {"generation": generation}
    app._search_query = "NVDA"
    app._render_search = lambda: None
    app._ensure_search_visible = lambda: None
    app._default_opt_row_index = lambda: 0

    call_contract = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="C",
    )
    call_contract.conId = 991001
    put_contract = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="P",
    )
    put_contract.conId = 991002
    seen_search_kwargs: dict[str, object] = {}

    async def _search_contracts(
        _query: str,
        *,
        mode: str,
        limit: int,
        opt_underlyer_symbol: str | None = None,
        timing: dict[str, object] | None = None,
        **kwargs,
    ) -> list[Contract]:
        assert mode == "OPT"
        assert limit == positions_app._SEARCH_OPT_FETCH_LIMIT
        assert str(opt_underlyer_symbol or "").upper() == "NVDA"
        seen_search_kwargs.clear()
        seen_search_kwargs.update(kwargs)
        if timing is not None:
            timing.update(
                {
                    "candidate_count": 2,
                    "qualified_count": 2,
                    "stage": "done",
                    "reason": "ok",
                    "total_ms": 19.0,
                }
            )
        return [call_contract, put_contract]

    app._client = SimpleNamespace(search_contracts=_search_contracts)

    asyncio.run(
        app._run_search_opt_underlyer(
            generation,
            "NVDA",
            "NVDA",
            fetch_limit=positions_app._SEARCH_OPT_FETCH_LIMIT,
        )
    )

    assert "opt_first_limit" not in seen_search_kwargs
    assert "opt_progress" not in seen_search_kwargs
    assert len(app._search_opt_chain_cache.get("NVDA", [])) == 2
    assert len(app._search_results) == 2
    assert app._search_loading is False


def test_apply_opt_progress_rows_does_not_regress_full_rows() -> None:
    positions_app = _load_positions_app()
    app = positions_app.__new__(positions_app)
    generation = 21
    app._search_generation = generation
    app._search_loading = False
    app._search_error = None
    app._search_selected = 0
    app._search_scroll = 0
    app._search_side = 0
    app._search_opt_expiry_index = 0
    app._search_query = "NVDA"
    app._search_timing = {"generation": generation}
    app._render_search = lambda: None
    app._ensure_search_visible = lambda: None
    app._default_opt_row_index = lambda: 0

    full_call = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="C",
    )
    full_call.conId = 993001
    full_put = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="P",
    )
    full_put.conId = 993002
    app._search_results = [full_call, full_put]
    app._search_opt_chain_cache = {"NVDA": [full_call, full_put]}

    partial_call = Contract(
        secType="OPT",
        symbol="NVDA",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=190.0,
        right="C",
    )
    partial_call.conId = 993001

    app._apply_opt_progress_rows(
        generation=generation,
        symbol="NVDA",
        results=[partial_call],
        contract_timing={"stage": "qualify-first", "reason": "progress"},
        contracts_started=0.0,
    )

    assert len(app._search_results) == 2
    assert len(app._search_opt_chain_cache.get("NVDA", [])) == 2
