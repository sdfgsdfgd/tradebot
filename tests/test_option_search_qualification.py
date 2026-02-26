from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from ib_insync import Contract

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig


def _client() -> IBKRClient:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return IBKRClient(
        IBKRConfig(
            host="127.0.0.1",
            port=4001,
            client_id=101,
            proxy_client_id=102,
            account=None,
            refresh_sec=0.25,
            detail_refresh_sec=0.25,
            reconnect_interval_sec=1.0,
            reconnect_timeout_sec=5.0,
            reconnect_slow_interval_sec=5.0,
        )
    )


def test_qualify_proxy_contracts_falls_back_to_single_retries() -> None:
    client = _client()

    class _BurstyProxy:
        async def qualifyContractsAsync(self, *contracts):
            if len(contracts) > 1:
                raise RuntimeError("pacing burst")
            source = contracts[0]
            resolved = Contract(
                secType=str(getattr(source, "secType", "") or ""),
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            strike = int(round(float(getattr(source, "strike", 0.0) or 0.0)))
            resolved.conId = 700000 + strike
            return [resolved]

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _BurstyProxy()
    client.connect_proxy = _connect_proxy

    candidates = [
        Contract(
            secType="OPT",
            symbol="NVDA",
            exchange="SMART",
            currency="USD",
            lastTradeDateOrContractMonth="20260213",
            strike=184.0,
            right="C",
        ),
        Contract(
            secType="OPT",
            symbol="NVDA",
            exchange="SMART",
            currency="USD",
            lastTradeDateOrContractMonth="20260213",
            strike=185.0,
            right="P",
        ),
    ]

    qualified = asyncio.run(client.qualify_proxy_contracts(*candidates))

    assert len(qualified) == 2
    assert {int(getattr(contract, "conId", 0) or 0) for contract in qualified} == {
        700184,
        700185,
    }


def test_search_contracts_opt_drops_unqualified_rows() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD"),
                derivativeSecTypes=("OPT",),
            )
        ]

    async def _stock_option_chain(_symbol: str):
        underlying = Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD", conId=1)
        chain = SimpleNamespace(
            expirations=("20260213",),
            strikes=(184.0, 185.0),
            exchange="SMART",
            multiplier="100",
            tradingClass="NVDA",
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=184.2, ask=184.3, last=None, close=184.25)

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for contract in contracts:
            strike = float(getattr(contract, "strike", 0.0) or 0.0)
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            if (strike, right) != (184.0, "C"):
                continue
            resolved = Contract(
                secType="OPT",
                symbol="NVDA",
                exchange="SMART",
                currency="USD",
                lastTradeDateOrContractMonth="20260213",
                strike=184.0,
                right="C",
            )
            resolved.conId = 900001
            out.append(resolved)
        return out

    client._matching_symbols = _matching_symbols
    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    results = asyncio.run(client.search_contracts("NVDA", mode="OPT", limit=8))

    assert len(results) == 1
    assert int(getattr(results[0], "conId", 0) or 0) == 900001


def test_search_terms_opt_expands_bitcoin_aliases() -> None:
    terms = IBKRClient._search_terms("bitcoin", mode="OPT")

    assert "BITCOIN" in terms
    assert "BITU" in terms
    assert "IBIT" in terms


def test_search_terms_fop_expands_micro_bitcoin_aliases() -> None:
    terms = IBKRClient._search_terms("micro bitcoin", mode="FOP")

    assert "MICRO" in terms
    assert "BITCOIN" in terms
    assert "MBT" in terms


def test_search_terms_fop_maps_micro_crude_alias_to_mcl_without_mbt() -> None:
    terms = IBKRClient._search_terms("micro crude", mode="FOP")

    assert "MICRO" in terms
    assert "CRUDE" in terms
    assert "MCL" in terms
    assert "MBT" not in terms


def test_search_terms_fop_expands_1oz_alias_to_gc() -> None:
    terms = IBKRClient._search_terms("1oz", mode="FOP")

    assert "1OZ" in terms
    assert "GC" in terms


def test_search_terms_fop_collapses_spaced_1oz_alias_to_gc() -> None:
    terms = IBKRClient._search_terms("1 oz", mode="FOP")

    assert "1OZ" in terms
    assert "GC" in terms


def test_search_contract_labels_uses_hint_when_matching_symbols_missing_label() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="MCL", exchange="NYMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
                description=None,
                longName=None,
                companyName=None,
            )
        ]

    client._matching_symbols = _matching_symbols

    labels = asyncio.run(
        client.search_contract_labels("micro crude", mode="FOP", symbols=["MCL"])
    )

    assert labels.get("MCL") == "Micro WTI Crude Oil"


def test_search_option_underlyers_returns_ranked_symbols() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="STK", symbol="BITU", exchange="SMART", currency="USD"),
                derivativeSecTypes=("OPT",),
                description="PROSHARES ULTRA BITCOIN ETF",
            ),
            SimpleNamespace(
                contract=Contract(secType="STK", symbol="IBIT", exchange="SMART", currency="USD"),
                derivativeSecTypes=("OPT",),
                description="ISHARES BITCOIN TRUST ETF",
            ),
        ]

    client._matching_symbols = _matching_symbols

    underlyers = asyncio.run(client.search_option_underlyers("bitcoin", limit=4))

    assert [symbol for symbol, _desc in underlyers[:2]] == ["BITU", "IBIT"]
    assert "BITCOIN ETF" in underlyers[0][1].upper()


def test_search_option_underlyers_prefers_direct_symbol_fast_path() -> None:
    client = _client()
    calls = {"matching": 0, "chain": 0}

    async def _matching_symbols(*_args, **_kwargs):
        calls["matching"] += 1
        return [
            SimpleNamespace(
                contract=Contract(secType="STK", symbol="OTHER", exchange="SMART", currency="USD"),
                derivativeSecTypes=("OPT",),
            )
        ]

    async def _stock_option_chain(symbol: str):
        calls["chain"] += 1
        assert str(symbol or "").strip().upper() == "NVDA"
        underlying = Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD")
        chain = SimpleNamespace(
            exchange="SMART",
            tradingClass="NVDA",
            expirations=("20260220",),
            strikes=(180.0, 185.0),
            multiplier="100",
        )
        return underlying, chain

    client._matching_symbols = _matching_symbols
    client.stock_option_chain = _stock_option_chain

    underlyers = asyncio.run(client.search_option_underlyers("NVDA", limit=4))

    assert underlyers == [("NVDA", "")]
    assert calls["chain"] == 1
    assert calls["matching"] == 0


def test_search_option_underlyers_reports_direct_timing() -> None:
    client = _client()

    async def _stock_option_chain(symbol: str):
        assert str(symbol or "").strip().upper() == "NVDA"
        underlying = Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD")
        chain = SimpleNamespace(
            exchange="SMART",
            tradingClass="NVDA",
            expirations=("20260220",),
            strikes=(180.0, 185.0),
            multiplier="100",
        )
        return underlying, chain

    client.stock_option_chain = _stock_option_chain

    timing: dict[str, object] = {}
    underlyers = asyncio.run(client.search_option_underlyers("NVDA", limit=4, timing=timing))

    assert underlyers == [("NVDA", "")]
    assert str(timing.get("source", "")) == "direct"
    assert int(timing.get("result_count", 0) or 0) == 1
    assert float(timing.get("direct_ms", 0.0) or 0.0) >= 0.0
    assert float(timing.get("total_ms", 0.0) or 0.0) >= 0.0


def test_search_contracts_opt_honors_forced_underlyer_symbol() -> None:
    client = _client()
    requested: dict[str, str] = {}

    async def _matching_symbols(*_args, **_kwargs):
        raise AssertionError("_matching_symbols should be bypassed when opt_underlyer_symbol is set")

    async def _stock_option_chain(symbol: str):
        requested["symbol"] = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=requested["symbol"],
            exchange="SMART",
            currency="USD",
            conId=11,
        )
        chain = SimpleNamespace(
            expirations=("20260220",),
            strikes=(60.0,),
            exchange="SMART",
            multiplier="100",
            tradingClass=requested["symbol"],
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=60.0, ask=60.2, last=60.1, close=60.0)

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 810000 + idx
            out.append(resolved)
        return out

    client._matching_symbols = _matching_symbols
    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    results = asyncio.run(
        client.search_contracts(
            "bitcoin",
            mode="OPT",
            limit=4,
            opt_underlyer_symbol="AAPL",
        )
    )

    assert requested.get("symbol") == "AAPL"
    assert results
    assert all(str(getattr(contract, "symbol", "") or "").upper() == "AAPL" for contract in results)


def test_search_contracts_opt_reports_timing_metrics() -> None:
    client = _client()

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=11,
        )
        chain = SimpleNamespace(
            expirations=("20260220",),
            strikes=(60.0,),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=60.0, ask=60.2, last=60.1, close=60.0)

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 820000 + idx
            out.append(resolved)
        return out

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    timing: dict[str, object] = {}
    results = asyncio.run(
        client.search_contracts(
            "nvda",
            mode="OPT",
            limit=4,
            opt_underlyer_symbol="AAPL",
            timing=timing,
        )
    )

    assert results
    assert str(timing.get("source", "")) == "search_contracts_opt"
    assert str(timing.get("stage", "")) == "done"
    assert str(timing.get("reason", "")) == "ok"
    assert int(timing.get("candidate_count", 0) or 0) >= 2
    assert int(timing.get("qualified_count", 0) or 0) >= 2
    assert float(timing.get("chain_ms", 0.0) or 0.0) >= 0.0
    assert float(timing.get("ref_price_ms", 0.0) or 0.0) >= 0.0
    assert float(timing.get("ref_price", 0.0) or 0.0) > 0.0
    assert float(timing.get("qualify_ms", 0.0) or 0.0) >= 0.0
    assert float(timing.get("total_ms", 0.0) or 0.0) >= 0.0


def test_search_contracts_opt_frontloads_nearest_expiry_before_paging() -> None:
    client = _client()

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=77,
        )
        chain = SimpleNamespace(
            expirations=("20260220", "20260227", "20260306", "20260320", "20260417", "20260619"),
            strikes=tuple(float(strike) for strike in range(140, 261)),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=190.0, ask=190.2, last=190.1, close=190.0)

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 840000 + idx
            out.append(resolved)
        return out

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    timing: dict[str, object] = {}
    results = asyncio.run(
        client.search_contracts(
            "nvda",
            mode="OPT",
            limit=48,
            opt_underlyer_symbol="NVDA",
            timing=timing,
        )
    )

    assert results
    expiries = sorted(
        {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in results
        }
    )
    assert len(expiries) == 1
    assert int(timing.get("selected_expiry_count", 0) or 0) == 1
    assert int(timing.get("rows_per_expiry", 0) or 0) == 20
    assert int(timing.get("candidate_count", 0) or 0) == 40


def test_search_contracts_opt_supports_expiry_offset_paging() -> None:
    client = _client()

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=91,
        )
        chain = SimpleNamespace(
            expirations=("20260220", "20260227", "20260306", "20260320", "20260417", "20260619"),
            strikes=tuple(float(strike) for strike in range(140, 261)),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=190.0, ask=190.2, last=190.1, close=190.0)

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 845000 + idx
            out.append(resolved)
        return out

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    seen_expiries: set[str] = set()
    offset = 0
    first_page_count: int | None = None
    page_count = 0
    while True:
        timing_page: dict[str, object] = {}
        page = asyncio.run(
            client.search_contracts(
                "nvda",
                mode="OPT",
                limit=48,
                opt_underlyer_symbol="NVDA",
                timing=timing_page,
                expiry_offset=offset,
            )
        )
        page_expiries = sorted(
            {
                str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
                for contract in page
            }
        )
        assert page
        assert page_expiries
        if first_page_count is None:
            first_page_count = len(page_expiries)
        assert set(page_expiries).isdisjoint(seen_expiries)
        seen_expiries.update(page_expiries)
        page_count += 1

        has_more = bool(timing_page.get("has_more_expiries"))
        next_offset = int(timing_page.get("next_expiry_offset", 0) or 0)
        if not has_more:
            break
        assert next_offset > offset
        offset = next_offset

    assert first_page_count == 1
    assert page_count >= 2
    assert seen_expiries == {
        "20260220",
        "20260227",
        "20260306",
        "20260320",
        "20260417",
        "20260619",
    }


def test_search_contracts_opt_progress_split_qualifies_in_two_passes() -> None:
    client = _client()
    qualify_batch_sizes: list[int] = []
    progress_events: list[dict[str, object]] = []

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=42,
        )
        chain = SimpleNamespace(
            expirations=("20260220",),
            strikes=(60.0, 61.0, 62.0, 63.0),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=61.0, ask=61.2, last=61.1, close=61.0)

    async def _qualify_proxy_contracts(*contracts: Contract):
        qualify_batch_sizes.append(len(contracts))
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 830000 + len(out) + idx
            out.append(resolved)
        return out

    async def _on_progress(rows: list[Contract], timing: dict[str, object]) -> None:
        progress_events.append(
            {
                "rows": len(rows),
                "candidate_count": int(timing.get("candidate_count", 0) or 0),
                "qualified_count": int(timing.get("qualified_count", 0) or 0),
                "stage": str(timing.get("stage", "") or ""),
            }
        )

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    timing: dict[str, object] = {}
    results = asyncio.run(
        client.search_contracts(
            "nvda",
            mode="OPT",
            limit=8,
            opt_underlyer_symbol="NVDA",
            timing=timing,
            opt_first_limit=4,
            opt_progress=_on_progress,
        )
    )

    assert len(results) == 8
    assert qualify_batch_sizes == [4, 4]
    assert progress_events
    assert progress_events[0]["rows"] == 4
    assert progress_events[0]["candidate_count"] == 4
    assert progress_events[0]["stage"] == "qualify-first"
    assert bool(timing.get("split_active")) is True
    assert int(timing.get("first_limit", 0) or 0) == 4
    assert float(timing.get("qualify_ms_first", 0.0) or 0.0) >= 0.0
    assert float(timing.get("qualify_ms_rest", 0.0) or 0.0) >= 0.0


def test_search_contracts_opt_waits_briefly_for_ticker_reference_before_median() -> None:
    client = _client()

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=123,
        )
        chain = SimpleNamespace(
            expirations=("20260320",),
            strikes=tuple(float(strike) for strike in range(300, 501, 5)),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        ticker = SimpleNamespace(bid=None, ask=None, last=None, close=None)

        async def _delayed_fill() -> None:
            await asyncio.sleep(0.08)
            ticker.bid = 383.9
            ticker.ask = 384.1

        asyncio.create_task(_delayed_fill())
        return ticker

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 860000 + idx
            out.append(resolved)
        return out

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    timing: dict[str, object] = {}
    results = asyncio.run(
        client.search_contracts(
            "msft",
            mode="OPT",
            limit=8,
            opt_underlyer_symbol="MSFT",
            timing=timing,
        )
    )

    assert results
    assert str(timing.get("ref_price_source", "")) == "ticker"
    strikes = sorted({float(getattr(contract, "strike", 0.0) or 0.0) for contract in results})
    assert strikes == [375.0, 380.0, 385.0, 390.0]


def test_search_contracts_opt_uses_historical_reference_when_ticker_unavailable() -> None:
    client = _client()
    historical_calls: list[str] = []

    async def _stock_option_chain(symbol: str):
        normalized = str(symbol or "").strip().upper()
        underlying = Contract(
            secType="STK",
            symbol=normalized,
            exchange="SMART",
            currency="USD",
            conId=124,
        )
        chain = SimpleNamespace(
            expirations=("20260320",),
            strikes=tuple(float(strike) for strike in range(300, 501, 5)),
            exchange="SMART",
            multiplier="100",
            tradingClass=normalized,
        )
        return underlying, chain

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=None, ask=None, last=None, close=None)

    async def _historical_bars(
        _contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str = "TRADES",
        cache_ttl_sec: float = 30.0,
    ):
        historical_calls.append(str(what_to_show))
        assert duration_str == "10800 S"
        assert bar_size == "1 min"
        assert use_rth is False
        _ = cache_ttl_sec
        if str(what_to_show).upper() == "TRADES":
            return [(datetime(2026, 2, 24, 10, 0, 0), 384.0)]
        return []

    async def _qualify_proxy_contracts(*contracts: Contract):
        out: list[Contract] = []
        for idx, source in enumerate(contracts, start=1):
            resolved = Contract(
                secType="OPT",
                symbol=str(getattr(source, "symbol", "") or ""),
                exchange=str(getattr(source, "exchange", "") or ""),
                currency=str(getattr(source, "currency", "") or ""),
                lastTradeDateOrContractMonth=str(
                    getattr(source, "lastTradeDateOrContractMonth", "") or ""
                ),
                strike=float(getattr(source, "strike", 0.0) or 0.0),
                right=str(getattr(source, "right", "") or ""),
            )
            resolved.conId = 861000 + idx
            out.append(resolved)
        return out

    client.stock_option_chain = _stock_option_chain
    client.ensure_ticker = _ensure_ticker
    client.historical_bars = _historical_bars
    client.qualify_proxy_contracts = _qualify_proxy_contracts
    client.release_ticker = lambda *_args, **_kwargs: None

    timing: dict[str, object] = {}
    results = asyncio.run(
        client.search_contracts(
            "msft",
            mode="OPT",
            limit=8,
            opt_underlyer_symbol="MSFT",
            timing=timing,
        )
    )

    assert results
    assert str(timing.get("ref_price_source", "")) == "historical-trades"
    assert historical_calls and historical_calls[0] == "TRADES"
    strikes = sorted({float(getattr(contract, "strike", 0.0) or 0.0) for contract in results})
    assert strikes == [375.0, 380.0, 385.0, 390.0]


def test_matching_symbols_retries_transient_timeout() -> None:
    client = _client()

    class _FlakyProxy:
        def __init__(self) -> None:
            self.calls = 0

        async def reqMatchingSymbolsAsync(self, _term):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("simulated timeout")
            return [
                SimpleNamespace(
                    contract=Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD"),
                    derivativeSecTypes=("OPT",),
                )
            ]

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _FlakyProxy()
    client.connect_proxy = _connect_proxy

    rows = asyncio.run(
        client._matching_symbols("NVDA", use_proxy=True, mode="OPT", raise_on_error=True)
    )

    assert rows
    assert str(getattr(getattr(rows[0], "contract", None), "symbol", "") or "").upper() == "NVDA"


def test_matching_symbols_retries_timeout_like_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()

    monkeypatch.setattr("tradebot.client._MATCHING_SYMBOL_TIMEOUT_INITIAL_SEC", 0.05)
    monkeypatch.setattr("tradebot.client._MATCHING_SYMBOL_TIMEOUT_RETRY_SEC", 0.05)
    monkeypatch.setattr("tradebot.client._MATCHING_SYMBOL_RETRY_BASE_SEC", 0.0)

    class _BoundaryProxy:
        def __init__(self) -> None:
            self.calls = 0

        async def reqMatchingSymbolsAsync(self, _term):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.046)
                return []
            return [
                SimpleNamespace(
                    contract=Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD"),
                    derivativeSecTypes=("OPT",),
                )
            ]

    async def _connect_proxy() -> None:
        return None

    proxy = _BoundaryProxy()
    client._ib_proxy = proxy
    client.connect_proxy = _connect_proxy

    rows = asyncio.run(
        client._matching_symbols("NVDA", use_proxy=True, mode="OPT", raise_on_error=True)
    )

    assert rows
    assert proxy.calls >= 2
    assert str(getattr(getattr(rows[0], "contract", None), "symbol", "") or "").upper() == "NVDA"


def test_search_option_underlyers_falls_back_to_direct_symbol_when_lookup_unavailable() -> None:
    client = _client()
    calls = {"chain": 0}

    async def _matching_symbols(*_args, **_kwargs):
        raise RuntimeError("IBKR symbol lookup unavailable: simulated timeout")

    async def _stock_option_chain(symbol: str):
        calls["chain"] += 1
        assert str(symbol).upper() == "NVDA"
        underlying = Contract(secType="STK", symbol="NVDA", exchange="SMART", currency="USD")
        chain = SimpleNamespace(
            exchange="SMART",
            tradingClass="NVDA",
            expirations=("20260220",),
            strikes=(180.0, 185.0),
            multiplier="100",
        )
        return underlying, chain

    client._matching_symbols = _matching_symbols
    client.stock_option_chain = _stock_option_chain

    underlyers = asyncio.run(client.search_option_underlyers("NVDA", limit=4))

    assert underlyers == [("NVDA", "")]
    assert calls["chain"] == 1


def test_search_contracts_fop_can_fallback_from_alias_root() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        # Mirrors real IB behavior for "1OZ": no direct FOP deriv in the hit list.
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="1OZ", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FUT",),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG",
                expirations=("20260327",),
                strikes=(5000.0, 5025.0, 5050.0),
                multiplier="100",
            )
        ]

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        resolved = Contract(
            secType=str(getattr(contract, "secType", "") or ""),
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=str(
                getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            ),
            strike=float(getattr(contract, "strike", 0.0) or 0.0),
            right=str(getattr(contract, "right", "") or ""),
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        resolved.conId = 920001
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    results = asyncio.run(client.search_contracts("1oz", mode="FOP", limit=5))

    assert results
    first = results[0]
    assert str(getattr(first, "secType", "") or "").upper() == "FOP"
    assert str(getattr(first, "symbol", "") or "").upper() == "GC"
    assert str(getattr(first, "tradingClass", "") or "").upper() == "OG"


def test_search_contracts_fop_micro_crude_prefers_mcl_over_mbt() -> None:
    client = _client()
    requested_symbols: list[str] = []

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="MBT", exchange="CME", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            ),
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="MCL", exchange="NYMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            ),
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        normalized = str(symbol or "").strip().upper()
        requested_symbols.append(normalized)
        if normalized != "MCL":
            return None
        fut = Contract(secType="FUT", symbol="MCL", exchange="NYMEX", currency="USD")
        fut.conId = 693600001
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "MCL"
        assert sec_type == "FUT"
        assert con_id == 693600001
        assert fut_fop_exchange == "NYMEX"
        return [
            SimpleNamespace(
                exchange="NYMEX",
                tradingClass="MW3",
                expirations=("20260220",),
                strikes=(65.0, 65.25, 65.5),
                multiplier="100",
            )
        ]

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=65.2, ask=65.3, last=65.25, close=65.1)

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        if right not in ("C", "P"):
            return None
        resolved = Contract(
            secType="FOP",
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=str(
                getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            ),
            strike=strike,
            right=right,
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        base = 936000 if right == "C" else 937000
        resolved.conId = base + int(round(strike * 100))
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client.ensure_ticker = _ensure_ticker
    client.release_ticker = lambda *_args, **_kwargs: None
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    results = asyncio.run(client.search_contracts("micro crude", mode="FOP", limit=4))

    assert results
    assert requested_symbols == ["MCL"]
    assert all(str(getattr(contract, "symbol", "") or "").upper() == "MCL" for contract in results)


def test_search_contracts_fop_recovers_after_empty_primary_lookup() -> None:
    client = _client()
    lookup_terms: list[str] = []

    async def _matching_symbols(term: str, *_args, **_kwargs):
        normalized = str(term or "").strip().upper()
        lookup_terms.append(normalized)
        if normalized != "GC":
            return []
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="GC", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG",
                expirations=("20260327",),
                strikes=(5000.0, 5025.0, 5050.0),
                multiplier="100",
            )
        ]

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        resolved = Contract(
            secType=str(getattr(contract, "secType", "") or ""),
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=str(
                getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            ),
            strike=float(getattr(contract, "strike", 0.0) or 0.0),
            right=str(getattr(contract, "right", "") or ""),
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        resolved.conId = 920002
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    results = asyncio.run(client.search_contracts("1 oz", mode="FOP", limit=5))

    assert results
    first = results[0]
    assert str(getattr(first, "secType", "") or "").upper() == "FOP"
    assert str(getattr(first, "symbol", "") or "").upper() == "GC"
    assert lookup_terms[0] == "1 OZ"
    assert "GC" in lookup_terms


def test_search_contracts_fop_returns_call_and_put_rows() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="GC", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG",
                expirations=("20260327",),
                strikes=(5000.0, 5025.0),
                multiplier="100",
            )
        ]

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=4998.0, ask=5002.0, last=5000.0, close=4995.0)

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        if right not in ("C", "P"):
            return None
        resolved = Contract(
            secType="FOP",
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=str(
                getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            ),
            strike=strike,
            right=right,
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        base = 930000 if right == "C" else 931000
        resolved.conId = base + int(round(strike))
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client.ensure_ticker = _ensure_ticker
    client.release_ticker = lambda *_args, **_kwargs: None
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    results = asyncio.run(client.search_contracts("1oz", mode="FOP", limit=6))

    assert results
    rights = {str(getattr(contract, "right", "") or "").strip().upper()[:1] for contract in results}
    assert "C" in rights
    assert "P" in rights
    assert all(str(getattr(contract, "secType", "") or "").strip().upper() == "FOP" for contract in results)


def test_search_contracts_fop_merges_expiries_across_chain_classes() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="GC", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            # Mirrors IB behavior where the first class may only hold one weekly expiry.
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG3",
                expirations=("20260220",),
                strikes=(4995.0, 5000.0, 5005.0),
                multiplier="100",
            ),
            # Monthly class with additional expiries that should still be surfaced.
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG",
                expirations=("20260224", "20260326"),
                strikes=(4985.0, 4990.0, 4995.0, 5000.0, 5005.0, 5010.0),
                multiplier="100",
            ),
        ]

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=4998.0, ask=5002.0, last=5000.0, close=4995.0)

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if right not in ("C", "P"):
            return None
        resolved = Contract(
            secType="FOP",
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        base = 932000 if right == "C" else 933000
        resolved.conId = base + int(round(strike)) + int(expiry[-2:])
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client.ensure_ticker = _ensure_ticker
    client.release_ticker = lambda *_args, **_kwargs: None
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    timing_first: dict[str, object] = {}
    first_page = asyncio.run(
        client.search_contracts(
            "1oz",
            mode="FOP",
            limit=96,
            timing=timing_first,
            expiry_offset=0,
        )
    )
    first_expiries = sorted(
        {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in first_page
        }
    )
    assert first_page
    assert first_expiries == ["20260220"]
    assert int(timing_first.get("selected_expiry_count", 0) or 0) == 1
    assert bool(timing_first.get("has_more_expiries")) is True

    next_offset = int(timing_first.get("next_expiry_offset", 0) or 0)
    assert next_offset > 0

    second_page = asyncio.run(
        client.search_contracts(
            "1oz",
            mode="FOP",
            limit=96,
            expiry_offset=next_offset,
        )
    )
    # Later pages should still surface the monthly class expiries.
    assert any(
        str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip() == "20260224"
        and str(getattr(contract, "tradingClass", "") or "").strip().upper() == "OG"
        for contract in second_page
    )


def test_search_contracts_fop_supports_expiry_offset_paging() -> None:
    client = _client()

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="GC", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG",
                expirations=("20260220", "20260224", "20260326", "20260424", "20260521"),
                strikes=(4985.0, 4990.0, 4995.0, 5000.0, 5005.0, 5010.0),
                multiplier="100",
            )
        ]

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        return SimpleNamespace(bid=4998.0, ask=5002.0, last=5000.0, close=4995.0)

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if right not in ("C", "P"):
            return None
        resolved = Contract(
            secType="FOP",
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        base = 952000 if right == "C" else 953000
        resolved.conId = base + int(round(strike)) + int(expiry[-2:])
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client.ensure_ticker = _ensure_ticker
    client.release_ticker = lambda *_args, **_kwargs: None
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    timing_first: dict[str, object] = {}
    first_page = asyncio.run(
        client.search_contracts(
            "1oz",
            mode="FOP",
            limit=96,
            timing=timing_first,
            expiry_offset=0,
        )
    )
    first_expiries = sorted(
        {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in first_page
        }
    )
    assert first_page
    assert first_expiries
    assert int(timing_first.get("selected_expiry_count", 0) or 0) == 1
    assert int(timing_first.get("rows_per_expiry", 0) or 0) == 20
    assert bool(timing_first.get("has_more_expiries")) is True

    next_offset = int(timing_first.get("next_expiry_offset", 0) or 0)
    assert next_offset > 0

    timing_second: dict[str, object] = {}
    second_page = asyncio.run(
        client.search_contracts(
            "1oz",
            mode="FOP",
            limit=96,
            timing=timing_second,
            expiry_offset=next_offset,
        )
    )
    second_expiries = sorted(
        {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in second_page
        }
    )
    assert second_page
    assert second_expiries
    assert set(second_expiries).isdisjoint(set(first_expiries))


def test_search_contracts_fop_uses_snapshot_reference_for_atm_rows() -> None:
    client = _client()
    snapshot_calls = {"count": 0}

    async def _matching_symbols(*_args, **_kwargs):
        return [
            SimpleNamespace(
                contract=Contract(secType="IND", symbol="GC", exchange="COMEX", currency="USD"),
                derivativeSecTypes=("FOP", "FUT"),
            )
        ]

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        if str(symbol).strip().upper() != "GC":
            return None
        fut = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
        fut.conId = 693609539
        fut.multiplier = "100"
        return fut

    async def _connect() -> None:
        return None

    async def _req_secdef(symbol: str, fut_fop_exchange: str, sec_type: str, con_id: int):
        assert symbol == "GC"
        assert sec_type == "FUT"
        assert con_id == 693609539
        assert fut_fop_exchange == "COMEX"
        return [
            # Near expiry has broad strikes but includes ATM only near 5010.
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="OG3",
                expirations=("20260220",),
                strikes=(4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 5000.0, 5010.0, 5020.0),
                multiplier="100",
            ),
            # Earlier expiry intentionally far from ATM; should be deprioritized.
            SimpleNamespace(
                exchange="COMEX",
                tradingClass="G3R",
                expirations=("20260219",),
                strikes=(4300.0, 4400.0, 4500.0, 4510.0),
                multiplier="100",
            ),
        ]

    async def _ensure_ticker(_contract: Contract, *, owner: str = "default"):
        # Stream quote unavailable -> must use snapshot fallback.
        return SimpleNamespace(bid=None, ask=None, last=None, close=None)

    async def _attempt_snapshot(_contract: Contract, *, ticker):
        snapshot_calls["count"] += 1
        ticker.last = 5010.0
        ticker.close = 5008.0
        return True

    async def _attempt_historical(_contract: Contract, *, ticker):
        return False

    async def _qualify_contract(contract: Contract, use_proxy: bool):
        assert use_proxy is False
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike = float(getattr(contract, "strike", 0.0) or 0.0)
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if right not in ("C", "P"):
            return None
        resolved = Contract(
            secType="FOP",
            symbol=str(getattr(contract, "symbol", "") or ""),
            exchange=str(getattr(contract, "exchange", "") or ""),
            currency=str(getattr(contract, "currency", "") or ""),
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            tradingClass=str(getattr(contract, "tradingClass", "") or ""),
            multiplier=str(getattr(contract, "multiplier", "") or ""),
        )
        base = 934000 if right == "C" else 935000
        resolved.conId = base + int(round(strike))
        return resolved

    client._matching_symbols = _matching_symbols
    client.front_future = _front_future
    client.connect = _connect
    client.ensure_ticker = _ensure_ticker
    client.release_ticker = lambda *_args, **_kwargs: None
    client._attempt_main_contract_snapshot_quote = _attempt_snapshot
    client._attempt_main_contract_historical_quote = _attempt_historical
    client._ib = SimpleNamespace(reqSecDefOptParamsAsync=_req_secdef)
    client._qualify_contract = _qualify_contract

    results = asyncio.run(client.search_contracts("1oz", mode="FOP", limit=4))

    assert results
    assert snapshot_calls["count"] >= 1
    strikes = sorted({float(getattr(contract, "strike", 0.0) or 0.0) for contract in results})
    expiries = sorted(
        {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in results
        }
    )
    assert strikes == [5000.0, 5010.0]
    assert expiries == ["20260220"]


def test_matching_symbols_raises_on_transient_outage_when_requested() -> None:
    client = _client()

    class _DownProxy:
        async def reqMatchingSymbolsAsync(self, _term):
            raise TimeoutError("simulated timeout")

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _DownProxy()
    client.connect_proxy = _connect_proxy

    with pytest.raises(RuntimeError, match="symbol lookup unavailable"):
        asyncio.run(client._matching_symbols("NVDA", use_proxy=True, mode="OPT", raise_on_error=True))
