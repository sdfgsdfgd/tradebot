from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
