from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ib_insync import Contract, Stock

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


class _FakePnLSingleIB:
    def __init__(self) -> None:
        self.req_calls: list[tuple[str, str, int]] = []
        self.cancel_calls: list[tuple[str, str, int]] = []

    @staticmethod
    def managedAccounts() -> list[str]:
        return ["DU123456"]

    def reqPnLSingle(self, account: str, model_code: str, con_id: int):
        self.req_calls.append((str(account), str(model_code), int(con_id)))
        return SimpleNamespace(
            account=str(account),
            modelCode=str(model_code),
            conId=int(con_id),
            unrealizedPnL=float("nan"),
        )

    def cancelPnLSingle(self, account: str, model_code: str, con_id: int) -> None:
        self.cancel_calls.append((str(account), str(model_code), int(con_id)))


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


def test_ensure_index_tickers_reloads_on_session_change(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    async def _qualify_index_contracts() -> dict[str, object]:
        return {
            "NQ": SimpleNamespace(
                symbol="QQQ",
                exchange="SMART",
                secType="STK",
                conId=8001,
                primaryExchange="NASDAQ",
            )
        }

    client.connect = _connect  # type: ignore[method-assign]
    client._qualify_index_contracts = _qualify_index_contracts  # type: ignore[method-assign]

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))
    asyncio.run(client._ensure_index_tickers())
    assert len(fake_ib.requests) == 1
    assert len(fake_ib.cancels) == 0
    assert str(getattr(fake_ib.requests[-1], "exchange", "")).upper() == "OVERNIGHT"

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, False))
    asyncio.run(client._ensure_index_tickers())
    assert len(fake_ib.cancels) >= 1
    assert len(fake_ib.requests) == 2
    assert str(getattr(fake_ib.requests[-1], "exchange", "")).upper() == "SMART"


def test_qualify_index_contracts_resolves_stock_proxies(monkeypatch) -> None:
    client = _new_client()
    monkeypatch.setattr("tradebot.client._INDEX_PROXY_SYMBOLS", {"NQ": "QQQ", "ES": "SPY"})
    seen: list[str] = []
    next_id = {"value": 91000}

    async def _qualify(candidate, use_proxy: bool):
        seen.append(str(getattr(candidate, "symbol", "")))
        next_id["value"] += 1
        return SimpleNamespace(
            symbol=str(getattr(candidate, "symbol", "")),
            secType="STK",
            exchange="SMART",
            conId=int(next_id["value"]),
        )

    client._qualify_contract = _qualify  # type: ignore[method-assign]
    qualified = asyncio.run(client._qualify_index_contracts())
    assert set(qualified.keys()) == {"NQ", "ES"}
    assert str(getattr(qualified["NQ"], "symbol", "")) == "QQQ"
    assert str(getattr(qualified["ES"], "symbol", "")) == "SPY"
    assert set(seen) == {"QQQ", "SPY"}


def test_on_error_main_index_stock_permission_forces_delayed() -> None:
    client = _new_client()
    client._index_contracts = {
        "NQ": SimpleNamespace(symbol="QQQ", secType="STK", conId=93001),
    }
    called: dict[str, object] = {"value": False, "requalify": False}

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        called["value"] = True
        called["requalify"] = bool(requalify)

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]

    contract = SimpleNamespace(symbol="QQQ", secType="STK", conId=93001)
    client._on_error_main(0, 354, "No market data subscription", contract)

    assert client._index_force_delayed is True
    assert called["value"] is True
    assert called["requalify"] is True


def test_probe_index_quotes_triggers_requalifying_resubscribe(monkeypatch) -> None:
    client = _new_client()
    client._index_tickers = {
        "NQ": SimpleNamespace(
            contract=SimpleNamespace(symbol="NQ", conId=9001),
            bid=None,
            ask=None,
            last=None,
            close=100.0,
            prevLast=100.0,
        )
    }
    called: dict[str, object] = {"value": False, "requalify": False}

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        called["value"] = True
        called["requalify"] = bool(requalify)

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())
    assert called["value"] is True
    assert called["requalify"] is False


def test_probe_index_quotes_degrades_to_delayed_before_requalify(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = False
    client._index_tickers = {
        "ES": SimpleNamespace(
            contract=SimpleNamespace(symbol="SPY", conId=9010),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        )
    }
    calls: list[bool] = []

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        calls.append(bool(requalify))

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())

    assert client._index_force_delayed is True
    assert calls == [False, False]


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


def test_probe_proxy_contract_quote_retries_live_without_forcing_delayed(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(
        secType="OPT",
        symbol="BITU",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=32.73,
        right="P",
    )
    contract.conId = 792492697
    ticker = SimpleNamespace(contract=contract, bid=None, ask=None, last=None)
    client._detail_tickers[int(contract.conId)] = (client._ib_proxy, ticker)

    calls: list[int] = []

    def _start_live_resubscribe(req_contract) -> None:
        calls.append(int(getattr(req_contract, "conId", 0) or 0))
        ticker.bid = 0.12
        ticker.ask = 0.14

    async def _sleep(_: float) -> None:
        return None

    client._start_proxy_contract_live_resubscribe = _start_live_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)

    asyncio.run(client._probe_proxy_contract_quote(contract))

    assert calls == [792492697]
    assert 792492697 not in client._proxy_contract_force_delayed


def test_proxy_contract_delayed_flags_clear_when_session_bucket_changes(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(
        secType="OPT",
        symbol="SLV",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20260320",
        strike=24.0,
        right="C",
    )
    contract.conId = 550011
    ticker = SimpleNamespace(contract=contract, bid=None, ask=None, last=None)
    client._detail_tickers[int(contract.conId)] = (client._ib_proxy, ticker)
    client._proxy_contract_force_delayed.add(int(contract.conId))
    client._proxy_session_bucket = "PRE"

    started_live: list[int] = []
    started_probe: list[int] = []

    def _start_live(req_contract) -> None:
        started_live.append(int(getattr(req_contract, "conId", 0) or 0))

    def _start_probe(req_contract) -> None:
        started_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    client._start_proxy_contract_live_resubscribe = _start_live  # type: ignore[method-assign]
    client._start_proxy_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_bucket", lambda _now: "RTH")

    client._maybe_reset_proxy_contract_delay_on_session_change()

    assert int(contract.conId) not in client._proxy_contract_force_delayed
    assert started_live == [550011]
    assert started_probe == [550011]


def test_sync_pnl_single_subscriptions_tracks_portfolio_con_ids() -> None:
    client = _new_client()
    fake_ib = _FakePnLSingleIB()
    client._ib = fake_ib

    item_1 = SimpleNamespace(contract=SimpleNamespace(conId=101))
    item_2 = SimpleNamespace(contract=SimpleNamespace(conId=202))
    client._sync_pnl_single_subscriptions([item_1, item_2], account="")

    assert sorted(fake_ib.req_calls) == [("DU123456", "", 101), ("DU123456", "", 202)]
    assert client._pnl_single_account == "DU123456"

    client._sync_pnl_single_subscriptions([item_2], account="")

    assert ("DU123456", "", 101) in fake_ib.cancel_calls
    assert ("DU123456", "", 202) not in fake_ib.cancel_calls


def test_pnl_single_unrealized_reads_live_value_and_ignores_nan() -> None:
    client = _new_client()
    client._pnl_single_by_con_id[101] = SimpleNamespace(unrealizedPnL=12.34)
    client._pnl_single_by_con_id[202] = SimpleNamespace(unrealizedPnL=float("nan"))

    assert client.pnl_single_unrealized(101) == 12.34
    assert client.pnl_single_unrealized(202) is None
    assert client.pnl_single_unrealized(303) is None
    assert client.has_pnl_single_subscription(101) is True
    assert client.has_pnl_single_subscription(202) is True
    assert client.has_pnl_single_subscription(303) is False
