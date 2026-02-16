from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ib_insync import Stock

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig


def _new_client() -> IBKRClient:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=301,
        proxy_client_id=302,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
    )
    return IBKRClient(cfg)


class _FakeProxyIB:
    def __init__(self) -> None:
        self.market_data_types: list[int] = []
        self.requests: list[object] = []
        self.cancels: list[object] = []

    def reqMarketDataType(self, md_type: int) -> None:
        self.market_data_types.append(int(md_type))

    def reqMktData(self, contract):
        self.requests.append(contract)
        md_type = self.market_data_types[-1] if self.market_data_types else None
        return SimpleNamespace(
            contract=contract,
            marketDataType=md_type,
            bid=None,
            ask=None,
            last=None,
            close=600.0,
            prevLast=600.0,
        )

    def cancelMktData(self, contract) -> None:
        self.cancels.append(contract)


def test_ticker_has_data_requires_actionable_quote() -> None:
    close_only = SimpleNamespace(bid=None, ask=None, last=None, close=600.0, prevLast=600.0)
    assert IBKRClient._ticker_has_data(close_only) is False

    with_last = SimpleNamespace(bid=None, ask=None, last=600.5, close=None, prevLast=None)
    assert IBKRClient._ticker_has_data(with_last) is True

    with_nbbo = SimpleNamespace(bid=600.4, ask=600.6, last=None, close=None, prevLast=None)
    assert IBKRClient._ticker_has_data(with_nbbo) is True


def test_ensure_proxy_tickers_reloads_on_session_route_change(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    qqq = Stock(symbol="QQQ", exchange="SMART", currency="USD")
    tqqq = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    client._proxy_contracts = {"QQQ": qqq, "TQQQ": tqqq}

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))
    asyncio.run(client._ensure_proxy_tickers())

    first_pass = list(fake_ib.requests)
    assert len(first_pass) == 2
    assert all(str(getattr(contract, "exchange", "")).upper() == "OVERNIGHT" for contract in first_pass)

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (True, False))
    asyncio.run(client._ensure_proxy_tickers())

    second_pass = list(fake_ib.requests)[-2:]
    assert len(fake_ib.cancels) >= 2
    assert len(second_pass) == 2
    assert all(str(getattr(contract, "exchange", "")).upper() == "SMART" for contract in second_pass)


def test_ensure_ticker_overnight_delayed_prefers_overnight_route(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib
    client._proxy_force_delayed = True

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))

    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.primaryExchange = "ARCA"
    asyncio.run(client.ensure_ticker(contract, owner="test"))

    requested = fake_ib.requests[-1]
    assert str(getattr(requested, "exchange", "")).upper() == "OVERNIGHT"


def test_delayed_resubscribe_falls_back_to_primary_exchange(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))

    contract = Stock(symbol="TQQQ", exchange="OVERNIGHT", currency="USD")
    contract.primaryExchange = "ARCA"
    contract.conId = 900001
    asyncio.run(client._resubscribe_proxy_contract_delayed(contract))

    requested = fake_ib.requests[-1]
    assert str(getattr(requested, "exchange", "")).upper() == "ARCA"


def test_proxy_error_10167_is_scoped_to_contract(monkeypatch) -> None:
    client = _new_client()
    called: dict[str, int] = {"global": 0, "contract": 0}

    def _global() -> None:
        called["global"] += 1

    def _contract(_contract) -> None:
        called["contract"] += 1

    client._start_proxy_resubscribe = _global  # type: ignore[method-assign]
    client._start_proxy_contract_delayed_resubscribe = _contract  # type: ignore[method-assign]

    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.conId = 12345
    client._on_error_proxy(0, 10167, "No market data permissions", contract)

    assert called["contract"] == 1
    assert called["global"] == 0
    assert 12345 in client._proxy_contract_force_delayed
