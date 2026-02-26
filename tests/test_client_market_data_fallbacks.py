from __future__ import annotations

import asyncio
from datetime import datetime
import json
import time
from types import SimpleNamespace

from ib_insync import Contract, Stock

import tradebot.client as client_module
from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig


class _FakeConnectIB:
    def __init__(self, *, connected: bool = False) -> None:
        self.connected = bool(connected)

    def isConnected(self) -> bool:
        return bool(self.connected)

    def disconnect(self) -> None:
        self.connected = False


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _new_client() -> IBKRClient:
    _ensure_event_loop()
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


class _FakeMainIBForRules:
    def __init__(self) -> None:
        self.market_data_types: list[int] = []
        self.requests: list[object] = []
        self.contract_details_calls = 0
        self.market_rule_calls = 0

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
            close=None,
            prevLast=None,
            minTick=None,
        )

    async def reqContractDetailsAsync(self, contract):
        self.contract_details_calls += 1
        return [
            SimpleNamespace(
                contract=contract,
                minTick=0.05,
                marketRuleIds="85",
            )
        ]

    async def reqMarketRuleAsync(self, market_rule_id: int):
        self.market_rule_calls += 1
        assert int(market_rule_id) == 85
        return [
            SimpleNamespace(lowEdge=0.0, increment=0.05),
            SimpleNamespace(lowEdge=5.0, increment=0.25),
            SimpleNamespace(lowEdge=100.0, increment=0.5),
        ]


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


def test_futures_md_ladder_prefers_live_then_delayed() -> None:
    open_ladder = client_module._futures_md_ladder(datetime(2026, 2, 24, 11, 0, 0))
    closed_ladder = client_module._futures_md_ladder(datetime(2026, 2, 28, 11, 0, 0))
    assert open_ladder[:2] == (1, 2)
    assert open_ladder[2:] == (3, 4)
    assert closed_ladder[:2] == (1, 2)
    assert closed_ladder[2:] == (4, 3)


def test_ensure_ticker_primes_price_increments_for_fop() -> None:
    client = _new_client()
    fake_ib = _FakeMainIBForRules()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    client._start_main_contract_quote_watchdog = lambda _contract: None  # type: ignore[method-assign]
    client._start_main_contract_quote_probe = lambda _contract: None  # type: ignore[method-assign]

    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 853838839
    ticker = asyncio.run(client.ensure_ticker(contract, owner="test"))
    _ = asyncio.run(client.ensure_ticker(contract, owner="test2"))

    assert tuple(getattr(ticker, "tbPriceIncrements", ())) == (
        (0.0, 0.05),
        (5.0, 0.25),
        (100.0, 0.5),
    )
    assert tuple(getattr(contract, "tbPriceIncrements", ())) == (
        (0.0, 0.05),
        (5.0, 0.25),
        (100.0, 0.5),
    )
    assert fake_ib.contract_details_calls == 1
    assert fake_ib.market_rule_calls == 1


def test_client_id_conflict_error_classifier() -> None:
    assert IBKRClient._is_client_id_conflict_error(RuntimeError("Client id already in use")) is True
    assert IBKRClient._is_client_id_conflict_error(RuntimeError("Duplicate client id")) is True
    assert IBKRClient._is_client_id_conflict_error(RuntimeError("API connection failed")) is False


def test_api_session_init_error_classifier() -> None:
    assert IBKRClient._is_api_session_init_error(asyncio.TimeoutError()) is True
    assert IBKRClient._is_api_session_init_error(RuntimeError("API connection failed: TimeoutError()")) is True
    assert IBKRClient._is_api_session_init_error(RuntimeError("Socket connection broken while connecting")) is True
    assert IBKRClient._is_api_session_init_error(RuntimeError("Client id already in use")) is False


def test_connect_ib_uses_configured_timeout() -> None:
    _ensure_event_loop()
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=701,
        proxy_client_id=702,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        connect_timeout_sec=13.5,
    )
    client = IBKRClient(cfg)

    class _FakeIB:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int, float]] = []

        async def connectAsync(self, host: str, port: int, clientId: int, timeout: float) -> None:
            self.calls.append((str(host), int(port), int(clientId), float(timeout)))

    fake_ib = _FakeIB()
    asyncio.run(client._connect_ib(fake_ib, client_id=745))
    assert fake_ib.calls == [("127.0.0.1", 4001, 745, 13.5)]


def test_current_order_state_promotes_pending_to_submitted_when_open() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=455, permId=0),
        orderStatus=SimpleNamespace(
            status="PendingSubmission",
            filled=0.0,
            remaining=1.0,
        ),
        contract=contract,
        isDone=lambda: False,
    )

    class _StateIB:
        @staticmethod
        def isConnected() -> bool:
            return True

        @staticmethod
        def trades():
            return [trade]

        @staticmethod
        def openTrades():
            return [trade]

        @staticmethod
        def fills():
            return []

    client._ib = _StateIB()

    payload = client.current_order_state(order_id=455, perm_id=0)

    assert isinstance(payload, dict)
    assert str(payload.get("raw_status")) == "PendingSubmission"
    assert str(payload.get("effective_status")) == "Submitted"


def test_connect_rotates_client_id_on_conflict_and_persists_pair(tmp_path) -> None:
    _ensure_event_loop()
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        client_id_pool_start=500,
        client_id_pool_end=505,
        client_id_burst_attempts=4,
        client_id_backoff_initial_sec=0.5,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(tmp_path / "ids.json"),
    )
    client = IBKRClient(cfg)
    client._request_reconnect = lambda: None  # type: ignore[method-assign]
    client._ib = _FakeConnectIB()
    client._ib_proxy = _FakeConnectIB(connected=True)
    client._connected_proxy_client_id = 503
    attempted: list[int] = []

    async def _fake_connect_ib(ib, *, client_id: int) -> None:
        attempted.append(int(client_id))
        if int(client_id) == 500:
            raise RuntimeError("Client id already in use")
        ib.connected = True

    client._connect_ib = _fake_connect_ib  # type: ignore[method-assign]
    asyncio.run(client.connect())

    assert attempted[:2] == [500, 502]
    assert int(client._main_client_id) == 502
    assert int(client._proxy_client_id) == 503
    persisted = json.loads((tmp_path / "ids.json").read_text(encoding="utf-8"))
    assert int(persisted["main_client_id"]) == 502
    assert int(persisted["proxy_client_id"]) == 503


def test_connect_rotates_client_id_on_api_init_timeout_and_quarantines_pair(tmp_path) -> None:
    _ensure_event_loop()
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        client_id_pool_start=500,
        client_id_pool_end=507,
        client_id_burst_attempts=4,
        client_id_backoff_initial_sec=0.5,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_quarantine_sec=120.0,
        client_id_state_file=str(tmp_path / "ids.json"),
    )
    client = IBKRClient(cfg)
    client._request_reconnect = lambda: None  # type: ignore[method-assign]
    client._ib = _FakeConnectIB()
    client._ib_proxy = _FakeConnectIB(connected=True)
    client._connected_proxy_client_id = 503
    attempted: list[int] = []

    async def _fake_connect_ib(ib, *, client_id: int) -> None:
        attempted.append(int(client_id))
        if int(client_id) == 500:
            raise asyncio.TimeoutError()
        ib.connected = True

    client._connect_ib = _fake_connect_ib  # type: ignore[method-assign]
    asyncio.run(client.connect())

    assert attempted[:2] == [500, 502]
    assert int(client._main_client_id) == 502
    assert int(client._proxy_client_id) == 503
    assert client._is_pair_quarantined(500, 501) is True


def test_connect_pool_exhaustion_arms_backoff(tmp_path) -> None:
    _ensure_event_loop()
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        client_id_pool_start=500,
        client_id_pool_end=503,
        client_id_burst_attempts=2,
        client_id_backoff_initial_sec=2.0,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(tmp_path / "ids.json"),
    )
    client = IBKRClient(cfg)
    client._request_reconnect = lambda: None  # type: ignore[method-assign]
    client._ib = _FakeConnectIB()

    async def _always_conflict(_ib, *, client_id: int) -> None:
        raise RuntimeError(f"Client id already in use: {int(client_id)}")

    client._connect_ib = _always_conflict  # type: ignore[method-assign]
    try:
        asyncio.run(client.connect())
    except Exception as exc:
        assert "pool exhausted" in str(exc).lower()
    else:
        raise AssertionError("expected pool exhaustion to raise")

    assert client._client_id_backoff_remaining_sec() > 0
    try:
        asyncio.run(client.connect())
    except Exception as exc:
        assert "backoff active" in str(exc).lower()
    else:
        raise AssertionError("expected active backoff to raise")


def test_connect_pool_exhaustion_on_api_init_timeout_arms_backoff(tmp_path) -> None:
    _ensure_event_loop()
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        client_id_pool_start=500,
        client_id_pool_end=503,
        client_id_burst_attempts=2,
        client_id_backoff_initial_sec=2.0,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(tmp_path / "ids.json"),
    )
    client = IBKRClient(cfg)
    client._request_reconnect = lambda: None  # type: ignore[method-assign]
    client._ib = _FakeConnectIB()

    async def _always_timeout(_ib, *, client_id: int) -> None:
        raise asyncio.TimeoutError(f"timed out while connecting with {int(client_id)}")

    client._connect_ib = _always_timeout  # type: ignore[method-assign]
    try:
        asyncio.run(client.connect())
    except Exception as exc:
        assert "connect retries exhausted" in str(exc).lower()
    else:
        raise AssertionError("expected timeout retries to raise")

    assert client._client_id_backoff_remaining_sec() > 0
    assert client._is_pair_quarantined(500, 501) is True
    assert client._is_pair_quarantined(502, 503) is True


def test_client_id_state_loads_valid_pair(tmp_path) -> None:
    _ensure_event_loop()
    state_path = tmp_path / "ids.json"
    state_path.write_text(
        json.dumps({"main_client_id": 504, "proxy_client_id": 505}),
        encoding="utf-8",
    )
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
        client_id_pool_start=500,
        client_id_pool_end=505,
        client_id_burst_attempts=3,
        client_id_backoff_initial_sec=0.5,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(state_path),
    )
    client = IBKRClient(cfg)
    assert int(client._main_client_id) == 504
    assert int(client._proxy_client_id) == 505


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
                symbol="NQ",
                exchange="CME",
                secType="FUT",
                conId=8001,
            )
        }

    client.connect = _connect  # type: ignore[method-assign]
    client._qualify_index_contracts = _qualify_index_contracts  # type: ignore[method-assign]

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))
    asyncio.run(client._ensure_index_tickers())
    assert len(fake_ib.requests) == 1
    assert fake_ib.market_data_types[-1] == 1
    assert len(fake_ib.cancels) == 0
    assert str(getattr(fake_ib.requests[-1], "exchange", "")).upper() == "CME"

    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, False))
    asyncio.run(client._ensure_index_tickers())
    assert len(fake_ib.cancels) == 0
    assert len(fake_ib.requests) == 1
    assert str(getattr(fake_ib.requests[-1], "exchange", "")).upper() == "CME"


def test_ensure_index_tickers_forced_delayed_tracks_futures_session(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib = fake_ib
    client._index_force_delayed = True

    async def _connect() -> None:
        return None

    async def _qualify_index_contracts() -> dict[str, object]:
        return {
            "NQ": SimpleNamespace(symbol="NQ", exchange="CME", secType="FUT", conId=8201),
        }

    state = {"open": True}

    def _ladder(_now):
        return (1, 2, 3, 4) if state["open"] else (1, 2, 4, 3)

    client.connect = _connect  # type: ignore[method-assign]
    client._qualify_index_contracts = _qualify_index_contracts  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, False))
    monkeypatch.setattr("tradebot.client._futures_md_ladder", _ladder)

    asyncio.run(client._ensure_index_tickers())
    assert fake_ib.market_data_types[-1] == 3
    assert len(fake_ib.requests) == 1
    assert len(fake_ib.cancels) == 0

    state["open"] = False
    asyncio.run(client._ensure_index_tickers())
    assert fake_ib.market_data_types[-1] == 4
    assert len(fake_ib.requests) == 2
    assert len(fake_ib.cancels) >= 1


def test_index_delayed_strip_resubscribes_on_futures_session_transition(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = True
    client._index_futures_session_open = True
    client._index_tickers = {
        "NQ": SimpleNamespace(contract=SimpleNamespace(symbol="NQ", secType="FUT", conId=8801))
    }
    calls: list[bool] = []

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        calls.append(bool(requalify))

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_session_is_open", lambda _now: False)

    client._maybe_resubscribe_index_on_session_transition()

    assert client._index_futures_session_open is False
    assert calls == [False]


def test_index_live_strip_ignores_futures_session_transition(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = False
    client._index_futures_session_open = True
    client._index_tickers = {
        "NQ": SimpleNamespace(contract=SimpleNamespace(symbol="NQ", secType="FUT", conId=8802))
    }
    calls: list[bool] = []

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        calls.append(bool(requalify))

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_session_is_open", lambda _now: False)

    client._maybe_resubscribe_index_on_session_transition()

    assert client._index_futures_session_open is False
    assert calls == []


def test_qualify_index_contracts_resolves_front_futures(monkeypatch) -> None:
    client = _new_client()
    monkeypatch.setattr("tradebot.client._INDEX_STRIP_SYMBOLS", ("NQ", "ES"))
    seen: list[tuple[str, str]] = []
    next_id = {"value": 91000}

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        seen.append((str(symbol or "").upper(), str(exchange or "").upper()))
        if str(exchange or "").upper() != "CME":
            return None
        next_id["value"] += 1
        return SimpleNamespace(
            symbol=str(symbol or "").upper(),
            secType="FUT",
            exchange="CME",
            conId=int(next_id["value"]),
        )

    client.front_future = _front_future  # type: ignore[method-assign]
    qualified = asyncio.run(client._qualify_index_contracts())
    assert set(qualified.keys()) == {"NQ", "ES"}
    assert str(getattr(qualified["NQ"], "symbol", "")).upper() == "NQ"
    assert str(getattr(qualified["ES"], "symbol", "")).upper() == "ES"
    assert ("NQ", "CME") in seen
    assert ("ES", "CME") in seen


def test_on_error_main_index_permission_forces_delayed() -> None:
    client = _new_client()
    client._index_contracts = {
        "NQ": SimpleNamespace(symbol="NQ", secType="FUT", conId=93001),
    }
    called: dict[str, object] = {"value": False, "requalify": False}

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        called["value"] = True
        called["requalify"] = bool(requalify)

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]

    contract = SimpleNamespace(symbol="NQ", secType="FUT", conId=93001)
    client._on_error_main(0, 354, "No market data subscription", contract)

    assert client._index_force_delayed is False
    assert client._index_symbol_force_delayed == {"NQ"}
    assert called["value"] is True
    assert called["requalify"] is True


def test_on_error_main_future_permission_starts_main_probe() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(contract=contract, bid=None, ask=None, last=None)
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)
    seen: list[int] = []
    resubscribe_md: list[int | None] = []

    def _start_probe(req_contract) -> None:
        seen.append(int(getattr(req_contract, "conId", 0) or 0))

    def _resubscribe(_ticker, *, md_type_override: int | None = None):
        resubscribe_md.append(md_type_override)
        return _ticker

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._resubscribe_main_contract_stream = _resubscribe  # type: ignore[method-assign]

    client._on_error_main(0, 354, "No market data subscription", contract)

    assert seen == [753716628]
    assert resubscribe_md == [4]
    assert getattr(ticker, "tbQuoteErrorCode", None) == 354


def test_on_error_main_future_permission_skips_error_stamp_when_close_only_exists() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=5036.25,
        prevLast=5036.25,
        tbQuoteSource="stream-close-only",
        tbQuoteErrorCode=10090,
    )
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)
    seen_probe: list[int] = []
    seen_resubscribe: list[int | None] = []
    seen_watchdog: list[int] = []

    def _start_probe(req_contract) -> None:
        seen_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    def _resubscribe(_ticker, *, md_type_override: int | None = None):
        seen_resubscribe.append(md_type_override)
        return _ticker

    def _start_watchdog(req_contract) -> None:
        seen_watchdog.append(int(getattr(req_contract, "conId", 0) or 0))

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._resubscribe_main_contract_stream = _resubscribe  # type: ignore[method-assign]
    client._start_main_contract_quote_watchdog = _start_watchdog  # type: ignore[method-assign]

    client._on_error_main(0, 10090, "Part of requested market data is not subscribed", contract)

    assert seen_probe == []
    assert seen_resubscribe == []
    assert seen_watchdog == [753716628]
    assert getattr(ticker, "tbQuoteErrorCode", None) is None


def test_probe_index_quotes_degrades_to_delayed_when_strip_totally_dead(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = False
    client._index_futures_session_open = True
    client._index_tickers = {
        "NQ": SimpleNamespace(
            contract=SimpleNamespace(symbol="NQ", conId=9001),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
        "ES": SimpleNamespace(
            contract=SimpleNamespace(symbol="ES", conId=9002),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
        "YM": SimpleNamespace(
            contract=SimpleNamespace(symbol="YM", conId=9003),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
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
    assert client._index_symbol_force_delayed == set()
    assert calls == [False]


def test_probe_index_quotes_degrades_to_delayed_after_warmup_without_actionable(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = False
    client._index_futures_session_open = True
    client._index_tickers = {
        "NQ": SimpleNamespace(
            contract=SimpleNamespace(symbol="NQ", conId=9010),
            bid=None,
            ask=None,
            last=None,
            close=25_000.0,
            prevLast=25_000.0,
        ),
        "ES": SimpleNamespace(
            contract=SimpleNamespace(symbol="ES", conId=9011),
            bid=None,
            ask=None,
            last=None,
            close=6_000.0,
            prevLast=6_000.0,
        ),
        "YM": SimpleNamespace(
            contract=SimpleNamespace(symbol="YM", conId=9012),
            bid=None,
            ask=None,
            last=None,
            close=45_000.0,
            prevLast=45_000.0,
        ),
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
    assert client._index_symbol_force_delayed == set()
    assert calls == [False]


def test_probe_index_quotes_partial_strip_forces_missing_leg_delayed(monkeypatch) -> None:
    client = _new_client()
    client._index_force_delayed = False
    client._index_futures_session_open = True
    client._index_tickers = {
        "NQ": SimpleNamespace(
            contract=SimpleNamespace(symbol="NQ", conId=9020),
            bid=25_000.0,
            ask=25_000.25,
            last=None,
            close=None,
            prevLast=None,
        ),
        "ES": SimpleNamespace(
            contract=SimpleNamespace(symbol="ES", conId=9021),
            bid=6_000.0,
            ask=6_000.25,
            last=None,
            close=None,
            prevLast=None,
        ),
        "YM": SimpleNamespace(
            contract=SimpleNamespace(symbol="YM", conId=9022),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
    }
    calls: list[bool] = []

    def _start_index_resubscribe(*, requalify: bool = False) -> None:
        calls.append(bool(requalify))

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())

    assert client._index_force_delayed is False
    assert client._index_symbol_force_delayed == {"YM"}
    assert calls == [False]


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


def test_attempt_main_contract_snapshot_quote_populates_fallback(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=1,
    )

    class _MainIB:
        def __init__(self) -> None:
            self.calls = 0

        def reqMarketDataType(self, _md_type: int) -> None:
            return None

        def reqMktData(self, _contract, _generic: str = "", snapshot: bool = False, _reg: bool = False):
            self.calls += 1
            if snapshot and self.calls >= 2:
                return SimpleNamespace(
                    contract=contract,
                    marketDataType=3,
                    bid=5016.5,
                    ask=5017.25,
                    last=5017.0,
                    close=4906.0,
                    prevLast=5019.25,
                )
            return SimpleNamespace(
                contract=contract,
                marketDataType=3,
                bid=None,
                ask=None,
                last=None,
                close=None,
                prevLast=None,
            )

    client._ib = _MainIB()
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)
    ok = asyncio.run(client._attempt_main_contract_snapshot_quote(contract, ticker=ticker))

    assert ok is True
    assert float(ticker.last) == 5017.0
    assert float(ticker.bid) == 5016.5
    assert float(ticker.ask) == 5017.25
    assert str(getattr(ticker, "tbQuoteSource", "")) in {
        "delayed-snapshot",
        "delayed-frozen-snapshot",
    }


def test_refresh_live_snapshot_once_prefers_live_snapshot(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=3,
    )

    class _MainIB:
        def __init__(self) -> None:
            self.market_data_types: list[int] = []

        def reqMarketDataType(self, md_type: int) -> None:
            self.market_data_types.append(int(md_type))

        def reqMktData(
            self,
            _contract,
            _generic: str = "",
            snapshot: bool = False,
            _reg: bool = False,
        ):
            md_type = self.market_data_types[-1] if self.market_data_types else 0
            if snapshot and md_type == 1:
                return SimpleNamespace(
                    contract=contract,
                    marketDataType=1,
                    bid=116.0,
                    ask=116.5,
                    last=116.25,
                    close=114.0,
                    prevLast=114.0,
                )
            return SimpleNamespace(
                contract=contract,
                marketDataType=md_type,
                bid=None,
                ask=None,
                last=None,
                close=None,
                prevLast=None,
            )

    client._ib = _MainIB()
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]

    async def _connect() -> None:
        return None

    async def _sleep(_: float) -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    source = asyncio.run(client.refresh_live_snapshot_once(contract))

    assert source == "live-snapshot"
    assert float(ticker.last) == 116.25
    assert float(ticker.bid) == 116.0
    assert float(ticker.ask) == 116.5
    assert client._ib.market_data_types == [1]


def test_attempt_main_contract_historical_quote_populates_last(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=1,
    )
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))

    async def _historical_bars(
        _contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str,
        cache_ttl_sec: float,
    ):
        assert duration_str == "10800 S"
        assert bar_size == "1 min"
        assert use_rth is False
        assert cache_ttl_sec == 20.0
        if what_to_show == "TRADES":
            return [(datetime(2026, 2, 18, 10, 0, 0), 25011.5)]
        return []

    client.historical_bars = _historical_bars  # type: ignore[method-assign]
    ok = asyncio.run(client._attempt_main_contract_historical_quote(contract, ticker=ticker))

    assert ok is True
    assert float(ticker.last) == 25011.5
    assert float(ticker.close) == 25011.5
    assert str(getattr(ticker, "tbQuoteSource", "")) == "historical-trades"


def test_attempt_main_contract_historical_quote_marks_delayed_when_ladder_live_first(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=1,
    )
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (1, 2, 3, 4))
    monkeypatch.setattr("tradebot.client._futures_session_is_open", lambda _now: True)

    async def _historical_bars(
        _contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str,
        cache_ttl_sec: float,
    ):
        if what_to_show == "TRADES":
            return [(datetime(2026, 2, 18, 10, 0, 0), 25011.5)]
        return []

    client.historical_bars = _historical_bars  # type: ignore[method-assign]
    ok = asyncio.run(client._attempt_main_contract_historical_quote(contract, ticker=ticker))

    assert ok is True
    assert int(getattr(ticker, "marketDataType", 0) or 0) == 3


def test_attempt_main_contract_historical_quote_uses_daily_fallback_for_fop(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=5005.0,
        right="C",
    )
    contract.conId = 849222157
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=1,
    )
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    seen_requests: list[tuple[str, str, str, bool, float]] = []

    async def _historical_bars(
        _contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str,
        cache_ttl_sec: float,
    ):
        seen_requests.append((duration_str, bar_size, what_to_show, bool(use_rth), float(cache_ttl_sec)))
        if (duration_str, bar_size, what_to_show) == ("2 M", "1 day", "TRADES"):
            return [(datetime(2026, 2, 18, 0, 0, 0), 41.75)]
        return []

    client.historical_bars = _historical_bars  # type: ignore[method-assign]
    ok = asyncio.run(client._attempt_main_contract_historical_quote(contract, ticker=ticker))

    assert ok is True
    assert float(ticker.last) == 41.75
    assert float(ticker.close) == 41.75
    assert str(getattr(ticker, "tbQuoteSource", "")) == "historical-daily-trades"
    assert ("10800 S", "1 min", "TRADES", False, 20.0) in seen_requests
    assert ("2 M", "1 day", "TRADES", False, 120.0) in seen_requests


def test_attempt_main_contract_historical_quote_uses_daily_fallback_for_fut(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=1,
    )
    client._on_stream_update = lambda *args, **kwargs: None  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    seen_requests: list[tuple[str, str, str, bool, float]] = []

    async def _historical_bars(
        _contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str,
        cache_ttl_sec: float,
    ):
        seen_requests.append((duration_str, bar_size, what_to_show, bool(use_rth), float(cache_ttl_sec)))
        if (duration_str, bar_size, what_to_show) == ("2 M", "1 day", "TRADES"):
            return [(datetime(2026, 2, 18, 0, 0, 0), 5010.5)]
        return []

    client.historical_bars = _historical_bars  # type: ignore[method-assign]
    ok = asyncio.run(client._attempt_main_contract_historical_quote(contract, ticker=ticker))

    assert ok is True
    assert float(ticker.last) == 5010.5
    assert float(ticker.close) == 5010.5
    assert str(getattr(ticker, "tbQuoteSource", "")) == "historical-daily-trades"
    assert ("10800 S", "1 min", "TRADES", False, 20.0) in seen_requests
    assert ("2 M", "1 day", "TRADES", False, 120.0) in seen_requests


def test_tag_ticker_quote_meta_clears_stale_asof_when_unspecified() -> None:
    client = _new_client()
    ticker = SimpleNamespace(
        contract=SimpleNamespace(conId=1),
        bid=None,
        ask=None,
        last=None,
        tbQuoteAsOf="2026-02-19T12:00:00",
    )

    client._tag_ticker_quote_meta(ticker, source="stream")

    assert getattr(ticker, "tbQuoteAsOf", "missing") is None


def test_on_stream_update_updates_freshness_only_on_quote_signature_change(monkeypatch) -> None:
    client = _new_client()
    now = {"value": 10.0}
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: float(now["value"]))
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=5016.5,
        ask=5017.0,
        last=5016.75,
        close=None,
        prevLast=None,
        bidSize=1.0,
        askSize=1.0,
        lastSize=1.0,
        marketDataType=3,
        tbQuoteSource="historical-daily-trades",
        tbQuoteAsOf="2026-02-19T00:00:00",
        tbQuoteUpdatedMono=0.0,
    )

    client._on_stream_update([ticker])
    assert float(getattr(ticker, "tbQuoteUpdatedMono", 0.0)) == 10.0
    assert float(getattr(ticker, "tbTopQuoteUpdatedMono", 0.0)) == 10.0
    assert int(getattr(ticker, "tbTopQuoteMoveCount", 0) or 0) == 1
    assert str(getattr(ticker, "tbQuoteSource", "")) == "stream"
    assert getattr(ticker, "tbQuoteAsOf", "missing") is None

    now["value"] = 20.0
    client._on_stream_update([ticker])
    assert float(getattr(ticker, "tbQuoteUpdatedMono", 0.0)) == 10.0
    assert float(getattr(ticker, "tbTopQuoteUpdatedMono", 0.0)) == 10.0
    assert int(getattr(ticker, "tbTopQuoteMoveCount", 0) or 0) == 1

    ticker.last = 5017.25
    client._on_stream_update([ticker])
    assert float(getattr(ticker, "tbQuoteUpdatedMono", 0.0)) == 20.0
    assert float(getattr(ticker, "tbTopQuoteUpdatedMono", 0.0)) == 20.0
    assert int(getattr(ticker, "tbTopQuoteMoveCount", 0) or 0) == 2


def test_on_stream_update_size_only_change_keeps_top_change_timestamp(monkeypatch) -> None:
    client = _new_client()
    now = {"value": 10.0}
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: float(now["value"]))
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=5016.5,
        ask=5017.0,
        last=5016.75,
        close=None,
        prevLast=None,
        bidSize=1.0,
        askSize=1.0,
        lastSize=1.0,
        marketDataType=3,
        tbQuoteUpdatedMono=0.0,
    )

    client._on_stream_update([ticker])
    assert float(getattr(ticker, "tbTopQuoteUpdatedMono", 0.0)) == 10.0
    assert int(getattr(ticker, "tbTopQuoteMoveCount", 0) or 0) == 1

    now["value"] = 20.0
    ticker.bidSize = 2.0
    client._on_stream_update([ticker])
    assert float(getattr(ticker, "tbQuoteUpdatedMono", 0.0)) == 20.0
    assert float(getattr(ticker, "tbTopQuoteUpdatedMono", 0.0)) == 10.0
    assert int(getattr(ticker, "tbTopQuoteMoveCount", 0) or 0) == 1


def test_watch_main_contract_quote_reprobes_when_quote_is_stale(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=5017.0,
        close=5008.0,
        prevLast=5008.0,
        marketDataType=3,
        tbQuoteUpdatedMono=1.0,
    )
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)

    seen: list[int] = []

    def _start_probe(req_contract) -> None:
        seen.append(int(getattr(req_contract, "conId", 0) or 0))
        client._detail_tickers.pop(int(contract.conId), None)

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 200.0)

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._watch_main_contract_quote(contract))

    assert seen == [753716628]


def test_watch_main_contract_quote_promotes_md3_to_delayed_frozen_when_topline_stale(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=5017.0,
        ask=5017.5,
        last=5017.25,
        close=5008.0,
        prevLast=5008.0,
        marketDataType=3,
        tbQuoteUpdatedMono=190.0,
        tbTopQuoteUpdatedMono=100.0,
    )
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)

    seen_probe: list[int] = []
    seen_resubscribe: list[int | None] = []

    def _start_probe(req_contract) -> None:
        seen_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    def _resubscribe(_ticker, *, md_type_override: int | None = None):
        seen_resubscribe.append(md_type_override)
        client._detail_tickers.pop(int(contract.conId), None)
        return _ticker

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._resubscribe_main_contract_stream = _resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 200.0)

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._watch_main_contract_quote(contract))

    assert seen_probe == [753716628]
    assert seen_resubscribe == [4]


def test_watch_main_contract_quote_promotes_md2_to_live_when_session_open(monkeypatch) -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(
        contract=contract,
        bid=5017.0,
        ask=5017.5,
        last=5017.25,
        close=5008.0,
        prevLast=5008.0,
        marketDataType=2,
        tbQuoteUpdatedMono=198.0,
        tbTopQuoteUpdatedMono=198.0,
    )
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)

    seen_probe: list[int] = []
    seen_resubscribe: list[int | None] = []

    def _start_probe(req_contract) -> None:
        seen_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    def _resubscribe(_ticker, *, md_type_override: int | None = None):
        seen_resubscribe.append(md_type_override)
        client._detail_tickers.pop(int(contract.conId), None)
        return _ticker

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._resubscribe_main_contract_stream = _resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 200.0)
    monkeypatch.setattr("tradebot.client._futures_session_is_open", lambda _now: True)

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._watch_main_contract_quote(contract))

    assert seen_probe == []
    assert seen_resubscribe == [1]


def test_front_future_ignores_undated_cache_and_prefers_dated_contract() -> None:
    client = _new_client()
    stale = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    stale.conId = 1001
    stale.lastTradeDateOrContractMonth = ""
    client._front_future_cache[("1OZ", "COMEX")] = (stale, time.monotonic())

    dated = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    dated.conId = 753716628
    dated.lastTradeDateOrContractMonth = "20260327"
    dated.localSymbol = "1OZJ6"

    class _MainIB:
        async def reqContractDetailsAsync(self, _candidate):
            return [SimpleNamespace(contract=dated, realExpirationDate="20260327")]

        async def qualifyContractsAsync(self, contract):
            return [contract]

    async def _connect() -> None:
        return None

    client._ib = _MainIB()
    client.connect = _connect  # type: ignore[method-assign]

    resolved = asyncio.run(client.front_future("1OZ", exchange="COMEX", cache_ttl_sec=3600.0))

    assert resolved is not None
    assert str(getattr(resolved, "lastTradeDateOrContractMonth", "") or "") == "20260327"
    assert int(getattr(resolved, "conId", 0) or 0) == 753716628


def test_ensure_ticker_future_starts_main_probe_when_stream_empty(monkeypatch) -> None:
    client = _new_client()

    class _MainIB:
        def __init__(self) -> None:
            self.market_data_types: list[int] = []
            self.requests: list[object] = []

        def reqMarketDataType(self, md_type: int) -> None:
            self.market_data_types.append(int(md_type))

        def reqMktData(self, contract):
            self.requests.append(contract)
            return SimpleNamespace(
                contract=contract,
                marketDataType=self.market_data_types[-1] if self.market_data_types else None,
                bid=None,
                ask=None,
                last=None,
                close=None,
                prevLast=None,
            )

    fake_ib = _MainIB()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    probes: list[int] = []

    def _start_probe(req_contract) -> None:
        probes.append(int(getattr(req_contract, "conId", 0) or 0))

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    ticker = asyncio.run(client.ensure_ticker(contract, owner="test"))

    assert int(getattr(ticker, "marketDataType", 0) or 0) == 3
    assert probes == [753716628]


def test_ensure_ticker_future_defaults_exchange_when_missing(monkeypatch) -> None:
    client = _new_client()

    class _MainIB:
        def __init__(self) -> None:
            self.market_data_types: list[int] = []
            self.requests: list[object] = []

        def reqMarketDataType(self, md_type: int) -> None:
            self.market_data_types.append(int(md_type))

        def reqMktData(self, contract):
            self.requests.append(contract)
            return SimpleNamespace(
                contract=contract,
                marketDataType=self.market_data_types[-1] if self.market_data_types else None,
                bid=None,
                ask=None,
                last=None,
                close=None,
                prevLast=None,
            )

    fake_ib = _MainIB()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    contract = Contract(secType="FUT", symbol="1OZ", currency="USD")
    contract.conId = 753716628

    ticker = asyncio.run(client.ensure_ticker(contract, owner="test"))

    assert fake_ib.requests
    assert str(getattr(fake_ib.requests[-1], "exchange", "") or "").strip().upper() == "COMEX"
    assert str(getattr(ticker.contract, "exchange", "") or "").strip().upper() == "COMEX"


def test_ensure_ticker_future_replaces_cached_empty_exchange(monkeypatch) -> None:
    client = _new_client()

    class _MainIB:
        def __init__(self) -> None:
            self.market_data_types: list[int] = []
            self.requests: list[object] = []
            self.cancels: list[object] = []

        def reqMarketDataType(self, md_type: int) -> None:
            self.market_data_types.append(int(md_type))

        def reqMktData(self, contract):
            self.requests.append(contract)
            return SimpleNamespace(
                contract=contract,
                marketDataType=self.market_data_types[-1] if self.market_data_types else None,
                bid=None,
                ask=None,
                last=None,
                close=None,
                prevLast=None,
            )

        def cancelMktData(self, contract) -> None:
            self.cancels.append(contract)

    fake_ib = _MainIB()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    started_watchdog: list[str] = []
    started_probe: list[str] = []

    def _start_watchdog(req_contract) -> None:
        started_watchdog.append(str(getattr(req_contract, "exchange", "") or "").strip().upper())

    def _start_probe(req_contract) -> None:
        started_probe.append(str(getattr(req_contract, "exchange", "") or "").strip().upper())

    client._start_main_contract_quote_watchdog = _start_watchdog  # type: ignore[method-assign]
    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]

    contract = Contract(secType="FUT", symbol="1OZ", currency="USD")
    contract.conId = 753716628
    stale = Contract(secType="FUT", symbol="1OZ", currency="USD")
    stale.conId = int(contract.conId)
    client._detail_tickers[int(contract.conId)] = (
        fake_ib,
        SimpleNamespace(
            contract=stale,
            marketDataType=3,
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
    )

    ticker = asyncio.run(client.ensure_ticker(contract, owner="test"))

    assert fake_ib.cancels and fake_ib.cancels[-1] is stale
    assert fake_ib.requests
    assert str(getattr(fake_ib.requests[-1], "exchange", "") or "").strip().upper() == "COMEX"
    assert str(getattr(ticker.contract, "exchange", "") or "").strip().upper() == "COMEX"
    assert started_watchdog == ["COMEX"]
    assert started_probe == ["COMEX"]


def test_ensure_ticker_future_arms_watchdog_even_when_stream_has_data(monkeypatch) -> None:
    client = _new_client()

    class _MainIB:
        def __init__(self) -> None:
            self.market_data_types: list[int] = []

        def reqMarketDataType(self, md_type: int) -> None:
            self.market_data_types.append(int(md_type))

        def reqMktData(self, contract):
            return SimpleNamespace(
                contract=contract,
                marketDataType=self.market_data_types[-1] if self.market_data_types else None,
                bid=None,
                ask=None,
                last=5017.0,
                close=5008.0,
                prevLast=5008.0,
            )

    fake_ib = _MainIB()
    client._ib = fake_ib

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._futures_md_ladder", lambda _now: (3, 4))
    probes: list[int] = []
    watchdogs: list[int] = []

    def _start_probe(req_contract) -> None:
        probes.append(int(getattr(req_contract, "conId", 0) or 0))

    def _start_watchdog(req_contract) -> None:
        watchdogs.append(int(getattr(req_contract, "conId", 0) or 0))

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._start_main_contract_quote_watchdog = _start_watchdog  # type: ignore[method-assign]
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628

    asyncio.run(client.ensure_ticker(contract, owner="test"))

    assert probes == []
    assert watchdogs == [753716628]


def test_resolve_underlying_contract_fop_falls_back_to_front_future_when_under_con_id_missing() -> None:
    client = _new_client()
    contract = Contract(
        secType="FOP",
        symbol="GC",
        exchange="COMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20260220",
        strike=5005.0,
        right="C",
    )
    contract.conId = 849222157
    future = Contract(secType="FUT", symbol="GC", exchange="COMEX", currency="USD")
    future.conId = 693609539
    calls: list[tuple[str, str]] = []

    async def _front_future(symbol: str, *, exchange: str = "CME", cache_ttl_sec: float = 3600.0):
        calls.append((str(symbol or "").strip().upper(), str(exchange or "").strip().upper()))
        return future

    client.front_future = _front_future  # type: ignore[method-assign]
    resolved = asyncio.run(client.resolve_underlying_contract(contract))

    assert resolved is future
    assert calls
    assert calls[0][0] == "GC"
    assert calls[0][1] == "COMEX"


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


def test_proxy_top_row_resubscribes_when_overnight_route_flips(monkeypatch) -> None:
    client = _new_client()
    client._proxy_session_bucket = "POST"
    client._proxy_session_include_overnight = False
    client._proxy_tickers = {
        "QQQ": SimpleNamespace(contract=Stock(symbol="QQQ", exchange="SMART", currency="USD"))
    }
    calls: list[int] = []

    def _start_proxy_resubscribe() -> None:
        calls.append(1)

    client._start_proxy_resubscribe = _start_proxy_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_bucket", lambda _now: "OVERNIGHT")
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))

    client._maybe_reset_proxy_contract_delay_on_session_change()

    assert client._proxy_session_bucket == "OVERNIGHT"
    assert client._proxy_session_include_overnight is True
    assert calls == [1]


def test_qualify_proxy_contracts_resolves_all_proxy_symbols(monkeypatch) -> None:
    client = _new_client()
    monkeypatch.setattr("tradebot.client._PROXY_SYMBOLS", ("QQQ", "SPY", "DIA", "TQQQ"))

    class _ProxyIB:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def qualifyContractsAsync(self, contract):
            symbol = str(getattr(contract, "symbol", "")).upper()
            self.seen.append(symbol)
            return [
                SimpleNamespace(
                    symbol=symbol,
                    secType="STK",
                    exchange="SMART",
                    conId=1000 + len(self.seen),
                )
            ]

    proxy_ib = _ProxyIB()
    client._ib_proxy = proxy_ib

    qualified = asyncio.run(client._qualify_proxy_contracts())

    assert set(proxy_ib.seen) == {"QQQ", "SPY", "DIA", "TQQQ"}
    assert set(qualified.keys()) == {"QQQ", "SPY", "DIA", "TQQQ"}
    assert all(str(getattr(contract, "exchange", "")).upper() == "SMART" for contract in qualified.values())


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
    client._pnl_single_by_con_id[404] = SimpleNamespace(unrealizedPnL=float("inf"))
    client._pnl_single_by_con_id[505] = SimpleNamespace(unrealizedPnL=1.7976931348623157e308)

    assert client.pnl_single_unrealized(101) == 12.34
    assert client.pnl_single_unrealized(202) is None
    assert client.pnl_single_unrealized(303) is None
    assert client.pnl_single_unrealized(404) is None
    assert client.pnl_single_unrealized(505) is None
    assert client.has_pnl_single_subscription(101) is True
    assert client.has_pnl_single_subscription(202) is True
    assert client.has_pnl_single_subscription(303) is False


def test_pnl_single_daily_reads_live_value_and_ignores_nan() -> None:
    client = _new_client()
    client._pnl_single_by_con_id[101] = SimpleNamespace(dailyPnL=-2.75)
    client._pnl_single_by_con_id[202] = SimpleNamespace(dailyPnL=float("nan"))
    client._pnl_single_by_con_id[404] = SimpleNamespace(dailyPnL=float("-inf"))
    client._pnl_single_by_con_id[505] = SimpleNamespace(dailyPnL=1.7976931348623157e308)

    assert client.pnl_single_daily(101) == -2.75
    assert client.pnl_single_daily(202) is None
    assert client.pnl_single_daily(303) is None
    assert client.pnl_single_daily(404) is None
    assert client.pnl_single_daily(505) is None


def test_account_pnl_stream_fields_ignore_invalid_values() -> None:
    client = _new_client()
    client._pnl = SimpleNamespace(unrealizedPnL=18.5, realizedPnL=-4.25)
    assert client.pnl_unrealized() == 18.5
    assert client.pnl_realized() == -4.25

    client._pnl = SimpleNamespace(unrealizedPnL=float("inf"), realizedPnL=1.7976931348623157e308)
    assert client.pnl_unrealized() is None
    assert client.pnl_realized() is None


def test_account_value_ignores_non_finite_values() -> None:
    client = _new_client()

    class _FakeIB:
        @staticmethod
        def accountValues(_account: str):
            return [SimpleNamespace(tag="UnrealizedPnL", currency="BASE", value=float("inf"))]

    client._ib = _FakeIB()  # type: ignore[assignment]
    value, currency, _updated = client.account_value("UnrealizedPnL")
    assert value is None
    assert currency == "BASE"


def test_session_close_anchors_expose_prev1_close_and_keep_legacy_shape() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 993311

    async def _request_historical_data(
        _contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        assert duration_str == "2 W"
        assert bar_size == "1 day"
        assert what_to_show == "TRADES"
        assert use_rth is True
        return [
            SimpleNamespace(close=19_980.0),
            SimpleNamespace(close=20_010.0),
            SimpleNamespace(close=20_040.0),
            SimpleNamespace(close=20_070.0),
            SimpleNamespace(close=20_100.0),
        ]

    client._request_historical_data = _request_historical_data  # type: ignore[method-assign]

    prev_close, close_1ago, close_3ago = asyncio.run(client.session_close_anchors(contract))
    assert prev_close == 20_100.0
    assert close_1ago == 20_070.0
    assert close_3ago == 20_010.0

    legacy_prev, legacy_3ago = asyncio.run(client.session_closes(contract))
    assert legacy_prev == prev_close
    assert legacy_3ago == close_3ago


def test_request_historical_data_timeout_records_diagnostics(monkeypatch) -> None:
    client = _new_client()
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 889911
    calls: list[dict[str, object]] = []

    class _SlowIB:
        async def reqHistoricalDataAsync(self, *_args, **kwargs):
            calls.append(dict(kwargs))
            await asyncio.sleep(0.05)
            return []

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _SlowIB()  # type: ignore[assignment]
    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    monkeypatch.setattr(client_module, "_HISTORICAL_REQUEST_TIMEOUT_SEC", 0.01)

    out = asyncio.run(
        client._request_historical_data(
            contract,
            duration_str="2 W",
            bar_size="10 mins",
            what_to_show="TRADES",
            use_rth=False,
        )
    )

    assert out == []
    diag = client.last_historical_request(contract)
    assert isinstance(diag, dict)
    assert diag.get("status") == "timeout"
    assert diag.get("error_type") == "TimeoutError"
    request = diag.get("request")
    assert isinstance(request, dict)
    assert request.get("duration_str") == "2 W"
    assert request.get("bar_size") == "10 mins"
    assert request.get("what_to_show") == "TRADES"
    assert request.get("use_rth") is False
    assert request.get("use_proxy") is True
    assert len(calls) == 1
    assert float(calls[0].get("timeout", 0.0) or 0.0) == 0.01


def test_request_historical_data_applies_month_duration_timeout_overrides() -> None:
    client = _new_client()
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 889912
    calls: list[dict[str, object]] = []

    class _CaptureIB:
        async def reqHistoricalDataAsync(self, *_args, **kwargs):
            calls.append(dict(kwargs))
            return []

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _CaptureIB()  # type: ignore[assignment]
    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]

    for duration, expected_timeout in (("1 M", 80.0), ("2 M", 100.0), ("3 M", 120.0)):
        calls.clear()
        out = asyncio.run(
            client._request_historical_data(
                contract,
                duration_str=duration,
                bar_size="10 mins",
                what_to_show="TRADES",
                use_rth=False,
            )
        )
        assert out == []
        assert len(calls) == 1
        assert float(calls[0].get("timeout", 0.0) or 0.0) == float(expected_timeout)
        diag = client.last_historical_request(contract)
        assert isinstance(diag, dict)
        assert float(diag.get("timeout_sec", 0.0) or 0.0) == float(expected_timeout)


def test_request_historical_data_for_stream_rejects_incomplete_full24_stitch() -> None:
    client = _new_client()
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 889922

    def _raw_bar(ts: datetime):
        return SimpleNamespace(
            date=ts,
            open=70.0,
            high=70.0,
            low=70.0,
            close=70.0,
            volume=1.0,
        )

    overnight_rows = [
        _raw_bar(datetime(2026, 2, 10, 20, 0)),
        _raw_bar(datetime(2026, 2, 11, 3, 40)),
        _raw_bar(datetime(2026, 2, 11, 20, 0)),
        _raw_bar(datetime(2026, 2, 12, 3, 40)),
    ]

    class _LegIB:
        async def reqHistoricalDataAsync(self, req_contract, *_args, **_kwargs):
            exchange = str(getattr(req_contract, "exchange", "") or "").strip().upper()
            if exchange == "SMART":
                raise asyncio.TimeoutError()
            if exchange == "OVERNIGHT":
                return list(overnight_rows)
            return []

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _LegIB()  # type: ignore[assignment]
    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]

    out = asyncio.run(
        client._request_historical_data_for_stream(
            contract,
            duration_str="1 M",
            bar_size="10 mins",
            what_to_show="TRADES",
            use_rth=False,
        )
    )

    assert out == []
    diag = client.last_historical_request(contract)
    assert isinstance(diag, dict)
    assert str(diag.get("status")) == "timeout"
    assert "stitch incomplete" in str(diag.get("detail", "")).lower()
    request = diag.get("request")
    assert isinstance(request, dict)
    assert str(request.get("duration_str")) == "1 M"
    assert str(request.get("bar_size")) == "10 mins"
    assert str(request.get("what_to_show")) == "TRADES"
    assert bool(request.get("use_rth")) is False
    stream_legs = diag.get("stream_legs")
    assert isinstance(stream_legs, dict)
    assert int(stream_legs.get("smart_rows", -1)) == 0
    assert int(stream_legs.get("overnight_rows", -1)) == len(overnight_rows)
    assert str(stream_legs.get("smart_status")) == "timeout"
    assert str(stream_legs.get("overnight_status")) == "ok"
    stream_quality = diag.get("stream_quality")
    assert isinstance(stream_quality, dict)
    assert bool(stream_quality.get("complete")) is False
    assert int(stream_quality.get("missing_days", 0)) >= 1


def test_last_historical_request_tracks_per_contract_statuses() -> None:
    client = _new_client()
    contract_a = Stock("SLV", "SMART", "USD")
    contract_b = Stock("GLD", "SMART", "USD")
    contract_a.conId = 3311
    contract_b.conId = 7722
    attempts: dict[int, int] = {}

    class _FakeIB:
        async def reqHistoricalDataAsync(self, req_contract, *_args, **_kwargs):
            con_id = int(getattr(req_contract, "conId", 0) or 0)
            attempts[con_id] = int(attempts.get(con_id, 0)) + 1
            if con_id == 3311:
                return []
            return [SimpleNamespace(close=1.0)]

    async def _connect_proxy() -> None:
        return None

    client._ib_proxy = _FakeIB()  # type: ignore[assignment]
    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]

    out_a = asyncio.run(
        client._request_historical_data(
            contract_a,
            duration_str="2 W",
            bar_size="10 mins",
            what_to_show="TRADES",
            use_rth=False,
        )
    )
    out_b = asyncio.run(
        client._request_historical_data(
            contract_b,
            duration_str="2 W",
            bar_size="10 mins",
            what_to_show="TRADES",
            use_rth=False,
        )
    )

    assert out_a == []
    assert len(out_b) == 1
    diag_a = client.last_historical_request(contract_a)
    diag_b = client.last_historical_request(contract_b)
    assert isinstance(diag_a, dict)
    assert isinstance(diag_b, dict)
    assert diag_a.get("status") == "empty"
    assert diag_b.get("status") == "ok"
    assert int(diag_a.get("bars_count", -1)) == 0
    assert int(diag_b.get("bars_count", -1)) == 1
