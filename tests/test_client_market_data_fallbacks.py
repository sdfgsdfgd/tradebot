from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from ib_insync import Contract, Stock
import pytest

import tradebot.client as client_module
from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig, load_config


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
        client_id_state_file="",
    )
    return IBKRClient(cfg)


def test_connect_ib_uses_transport_only_handshake_for_auxiliary_roles() -> None:
    client = _new_client()

    class _WireClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int, float]] = []

        async def connectAsync(
            self,
            host: str,
            port: int,
            client_id: int,
            timeout: float,
        ) -> None:
            self.calls.append((host, int(port), int(client_id), float(timeout)))

    class _IB:
        def __init__(self) -> None:
            self.wrapper = SimpleNamespace(clientId=None)
            self.client = _WireClient()
            self.full_sync_calls: list[dict[str, object]] = []

        async def connectAsync(self, host: str, port: int, **kwargs) -> None:
            self.full_sync_calls.append({"host": host, "port": port, **kwargs})

    auxiliary = _IB()
    asyncio.run(client._connect_ib(auxiliary, client_id=302))  # type: ignore[arg-type]

    assert auxiliary.client.calls == [("127.0.0.1", 4001, 302, 15.0)]
    assert auxiliary.wrapper.clientId == 302
    assert auxiliary.full_sync_calls == []

    main = _IB()
    client._ib = main  # type: ignore[assignment]
    asyncio.run(client._connect_ib(main, client_id=301))  # type: ignore[arg-type]

    assert main.client.calls == []
    assert main.full_sync_calls == [
        {
            "host": "127.0.0.1",
            "port": 4001,
            "clientId": 301,
            "timeout": 15.0,
            "readonly": False,
        }
    ]


def test_connection_state_covers_only_requested_roles() -> None:
    client = _new_client()
    client._ib = _FakeConnectIB(connected=True)  # type: ignore[assignment]
    client._ib_proxy = _FakeConnectIB(connected=True)  # type: ignore[assignment]
    client._ib_index = _FakeConnectIB(connected=True)  # type: ignore[assignment]

    assert client.connection_state() == "connected"

    client._proxy_required = True
    client._index_required = True
    assert client.connection_state() == "connected"

    client._ib_index.connected = False
    assert client.connection_state() == "degraded"

    client._ib.connected = False
    client._ib_proxy.connected = False
    assert client.connection_state() == "disconnected"


def test_index_disconnect_and_recovery_use_shared_reconnect_loop() -> None:
    client = _new_client()
    client._ib = _FakeConnectIB(connected=True)  # type: ignore[assignment]
    client._ib_proxy = _FakeConnectIB(connected=True)  # type: ignore[assignment]
    client._ib_index = _FakeConnectIB(connected=False)  # type: ignore[assignment]
    client._proxy_required = True
    client._index_required = True
    reconnect_starts = 0

    def _start_reconnect_loop() -> None:
        nonlocal reconnect_starts
        reconnect_starts += 1

    client._start_reconnect_loop = _start_reconnect_loop  # type: ignore[method-assign]
    client._on_disconnected_index()

    assert reconnect_starts == 1
    assert client._reconnect_requested is True
    assert client._resubscribe_index_needed is True
    assert client.connection_state() == "degraded"

    index_resubscriptions = 0

    async def _connect_index() -> None:
        client._ib_index.connected = True

    async def _ensure_index_tickers() -> None:
        nonlocal index_resubscriptions
        index_resubscriptions += 1

    async def _reconcile_order_state(*, force: bool = False) -> None:
        assert force is True

    client.connect_index = _connect_index  # type: ignore[method-assign]
    client._ensure_index_tickers = _ensure_index_tickers  # type: ignore[method-assign]
    client._start_index_probe = lambda: None  # type: ignore[method-assign]
    client.reconcile_order_state = _reconcile_order_state  # type: ignore[method-assign]

    asyncio.run(client._reconnect_once())

    assert index_resubscriptions == 1
    assert client._resubscribe_index_needed is False
    assert client._reconnect_requested is False
    assert client.connection_state() == "connected"


def test_market_data_state_distinguishes_warming_degraded_and_ready() -> None:
    client = _new_client()
    client._index_required = True
    client._index_futures_session_open = True

    assert client.market_data_state() == "warming(index:NQ,ES)"

    client._index_error = "index connection lost"
    assert client.market_data_state() == "degraded(index:NQ,ES)"

    client._index_error = None
    client._index_tickers = {
        symbol: SimpleNamespace(bid=1.0, ask=2.0, last=None, close=None)
        for symbol in ("NQ", "ES")
    }
    assert client.market_data_state() == "ready"


def test_disconnect_cancels_owned_streams_before_closing_role_sockets() -> None:
    client = _new_client()
    events: list[str] = []

    class _Wire:
        def reqAccountUpdates(self, subscribe: bool, account: str) -> None:
            events.append(f"account:{subscribe}:{account}")

    class _IB:
        def __init__(self, role: str) -> None:
            self.role = role
            self.connected = True
            self.client = _Wire()

        def isConnected(self) -> bool:
            return self.connected

        def cancelMktData(self, contract) -> None:
            events.append(f"cancel:{self.role}:{contract.symbol}")

        def cancelPnL(self, account: str) -> None:
            events.append(f"pnl:{account}")

        def disconnect(self) -> None:
            events.append(f"disconnect:{self.role}")
            self.connected = False

    main = _IB("main")
    proxy = _IB("proxy")
    index = _IB("index")
    client._ib = main  # type: ignore[assignment]
    client._ib_proxy = proxy  # type: ignore[assignment]
    client._ib_index = index  # type: ignore[assignment]
    client._pnl_account = "DU123456"
    client._account_updates_started = True
    client._index_tickers = {
        "NQ": SimpleNamespace(contract=SimpleNamespace(symbol="NQ", secType="FUT", conId=1))
    }
    client._proxy_tickers = {
        "SPY": SimpleNamespace(contract=SimpleNamespace(symbol="SPY", secType="STK", conId=2))
    }
    client._detail_tickers = {
        3: (
            main,
            SimpleNamespace(contract=SimpleNamespace(symbol="MNQ", secType="FUT", conId=3)),
        )
    }

    asyncio.run(client.disconnect())

    assert events[:5] == [
        "cancel:index:NQ",
        "cancel:proxy:SPY",
        "cancel:main:MNQ",
        "pnl:DU123456",
        "account:False:DU123456",
    ]
    assert events[5:] == [
        "disconnect:index",
        "disconnect:proxy",
        "disconnect:main",
    ]


def test_broker_boundary_rejects_market_or_nonfinite_limit_orders() -> None:
    client = _new_client()

    with pytest.raises(ValueError, match="only finite LIMIT orders"):
        client._require_limit_order(
            SimpleNamespace(orderType="MKT", lmtPrice=73.11),
            context="test",
        )
    with pytest.raises(ValueError, match="only finite LIMIT orders"):
        client._require_limit_order(
            SimpleNamespace(orderType="LMT", lmtPrice=float("nan")),
            context="test",
        )


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


def test_ensure_ticker_preserves_and_accumulates_generic_ticks(
    monkeypatch,
) -> None:
    client = _new_client()

    class _GenericProxy(_FakeProxyIB):
        def __init__(self) -> None:
            super().__init__()
            self.generic_requests: list[str] = []

        def reqMktData(self, contract, generic_ticks: str = ""):
            self.generic_requests.append(str(generic_ticks))
            return super().reqMktData(contract)

    fake_ib = _GenericProxy()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    client._start_proxy_contract_quote_probe = lambda _contract: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        "tradebot.client._session_flags",
        lambda _now: (False, False),
    )
    contract = Stock("SPYU", "SMART", "USD")
    contract.conId = 669475151

    ticker = asyncio.run(
        client.ensure_ticker(
            contract,
            owner="xsp-live",
            generic_ticks="614,577,623",
        )
    )
    same = asyncio.run(
        client.ensure_ticker(
            contract,
            owner="xsp-live-2",
            generic_ticks="577,614",
        )
    )

    assert ticker is same
    assert fake_ib.generic_requests == ["577,614,623"]
    tick_time = datetime(2026, 7, 29, 13, 40, tzinfo=timezone.utc)
    ticker.ticks = [
        SimpleNamespace(tickType=96, price=30.77899933, time=tick_time)
    ]
    client._sync_generic_ticks_for_ticker(ticker)
    assert client.generic_tick_value(ticker, 96) == (
        30.77899933,
        tick_time,
    )


def test_tick_by_tick_subscription_uses_main_derivative_stream() -> None:
    client = _new_client()
    calls: list[tuple[object, str, int, bool]] = []
    cancels: list[tuple[object, str]] = []
    marker = SimpleNamespace(tickByTicks=[])

    class _MainIB:
        def reqTickByTickData(
            self,
            contract,
            tick_type: str,
            number_of_ticks: int,
            ignore_size: bool,
        ):
            calls.append((contract, tick_type, number_of_ticks, ignore_size))
            return marker

        def cancelTickByTickData(self, contract, tick_type: str) -> None:
            cancels.append((contract, tick_type))

    async def _connect() -> None:
        return None

    client._ib = _MainIB()
    client.connect = _connect  # type: ignore[method-assign]
    contract = Contract(secType="FUT", symbol="MCL", currency="USD", conId=42)

    ticker = asyncio.run(client.subscribe_tick_by_tick(contract, "bidask"))
    client.unsubscribe_tick_by_tick(calls[0][0], "BidAsk")

    assert ticker is marker
    assert calls[0][1:] == ("BidAsk", 0, False)
    assert calls[0][0].exchange == "NYMEX"
    assert cancels == [(calls[0][0], "BidAsk")]


def test_tick_by_tick_subscription_rejects_unknown_event_type() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MCL", exchange="NYMEX")

    try:
        asyncio.run(client.subscribe_tick_by_tick(contract, "Depth"))
    except ValueError as exc:
        assert "unsupported tick-by-tick type" in str(exc)
    else:
        raise AssertionError("unknown tick-by-tick type was accepted")


def test_limit_order_binds_a_restart_safe_order_ref(monkeypatch) -> None:
    client = _new_client()
    monkeypatch.setattr(
        "tradebot.client._session_flags",
        lambda _now: (False, False),
    )
    contract = Stock("SPYU", "SMART", "USD")

    prepared_contract, order = asyncio.run(
        client._prepare_limit_order(
            contract,
            "BUY",
            43,
            30.79,
            False,
            "XSPV2-0123456789abcdef",
        )
    )

    assert prepared_contract.symbol == "SPYU"
    assert order.orderRef == "XSPV2-0123456789abcdef"


def test_us_equity_session_clock_honors_early_close() -> None:
    before_close = datetime(2026, 11, 27, 12, 59)
    after_close = datetime(2026, 11, 27, 13, 1)

    assert client_module._session_flags(before_close) == (False, False)
    assert client_module._session_bucket(before_close) == "RTH"
    assert client_module._session_flags(after_close) == (True, False)
    assert client_module._session_bucket(after_close) == "POST"
    assert client_module._session_bucket(datetime(2026, 11, 30, 13, 1)) == "RTH"


def test_early_close_fallback_sell_is_marked_outside_rth(monkeypatch) -> None:
    client = _new_client()
    monkeypatch.setattr(
        client_module,
        "_now_et",
        lambda: datetime(2026, 11, 27, 13, 1),
    )

    prepared_contract, order = asyncio.run(
        client._prepare_limit_order(
            Stock("SPYU", "SMART", "USD"),
            "SELL",
            43,
            30.79,
            True,
            "XSPV2-early-close",
        )
    )

    assert prepared_contract.exchange == "SMART"
    assert order.outsideRth is True
    assert order.tif == "GTC"


def test_order_ref_reconciliation_refreshes_completed_broker_state() -> None:
    client = _new_client()
    expected = SimpleNamespace(
        order=SimpleNamespace(orderRef="XSPV2-0123456789abcdef")
    )

    class _TradesIB:
        @staticmethod
        def isConnected() -> bool:
            return True

        @staticmethod
        def trades():
            return [
                expected,
                SimpleNamespace(order=SimpleNamespace(orderRef="other")),
            ]

    refreshed: list[bool] = []

    async def _sync(*, include_completed: bool) -> bool:
        refreshed.append(include_completed)
        return True

    client._ib = _TradesIB()
    client._sync_order_snapshots = _sync  # type: ignore[method-assign]

    assert asyncio.run(
        client.reconcile_trades_for_order_ref(
            "XSPV2-0123456789abcdef"
        )
    ) == [expected]
    assert refreshed == [True]


def test_account_identity_and_type_are_available_without_numeric_coercion() -> None:
    client = _new_client()

    class _AccountIB:
        @staticmethod
        def managedAccounts() -> list[str]:
            return ["DU123456"]

        @staticmethod
        def accountValues(_account: str):
            return [
                SimpleNamespace(
                    tag="AccountType",
                    value="CASH",
                    currency="",
                )
            ]

    client._ib = _AccountIB()

    assert client.account_id() == "DU123456"
    assert client.account_text_value("AccountType") == "CASH"


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
            self.calls: list[tuple[str, int, int, float, bool]] = []

        async def connectAsync(
            self,
            host: str,
            port: int,
            clientId: int,
            timeout: float,
            readonly: bool,
        ) -> None:
            self.calls.append(
                (
                    str(host),
                    int(port),
                    int(clientId),
                    float(timeout),
                    bool(readonly),
                )
            )

    fake_ib = _FakeIB()
    asyncio.run(client._connect_ib(fake_ib, client_id=745))
    assert fake_ib.calls == [("127.0.0.1", 4001, 745, 13.5, False)]


def test_readonly_config_reaches_all_ib_connections(monkeypatch) -> None:
    monkeypatch.setenv("IBKR_READONLY", "true")
    config = load_config()
    assert config.readonly is True

    client = IBKRClient(config)

    class _FakeIB:
        def __init__(self) -> None:
            self.readonly: list[bool] = []

        async def connectAsync(self, _host, _port, **kwargs) -> None:
            self.readonly.append(bool(kwargs["readonly"]))

    fake_ib = _FakeIB()
    asyncio.run(client._connect_ib(fake_ib, client_id=745))
    assert fake_ib.readonly == [True]


def test_shadow_owns_an_on_demand_readonly_gateway_tunnel() -> None:
    root = Path(__file__).resolve().parents[1]
    shadow = (root / "deploy/systemd/tradebot-xsp-shadow.service").read_text()
    tunnel = (
        root / "deploy/systemd/tradebot-ib-gateway-tunnel.service"
    ).read_text()

    assert "Requires=tradebot-ib-gateway-tunnel.service" in shadow
    assert "Environment=IBKR_READONLY=1" in shadow
    assert "StopWhenUnneeded=yes" in tunnel
    assert "StartLimitIntervalSec=0" in tunnel
    assert "RestartSec=30s" in tunnel
    assert "ExitOnForwardFailure=yes" in tunnel
    assert "ServerAliveInterval=15" in tunnel
    assert "StrictHostKeyChecking=yes" in tunnel
    assert "-L 127.0.0.1:4001:127.0.0.1:4001" in tunnel


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


def test_reconcile_order_state_retries_immediately_when_all_snapshot_requests_fail() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 750150193
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=771, permId=88001),
        orderStatus=SimpleNamespace(status="Submitted", filled=0.0, remaining=1.0),
        contract=contract,
        fills=[],
        isDone=lambda: False,
    )

    class _FailingSnapshotIB:
        def __init__(self) -> None:
            self.calls: list[str] = []

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
        def completedTrades():
            return []

        @staticmethod
        def fills():
            return []

        async def _fail(self, name: str):
            self.calls.append(name)
            raise RuntimeError(f"{name} unavailable")

        async def reqAllOpenOrdersAsync(self):
            return await self._fail("all_open")

        async def reqOpenOrdersAsync(self):
            return await self._fail("open")

        async def reqExecutionsAsync(self, _execution_filter):
            return await self._fail("executions")

    fake_ib = _FailingSnapshotIB()
    client._ib = fake_ib
    client._last_order_reconcile_mono = 0.0

    async def _run():
        first = await client.reconcile_order_state(order_id=771, perm_id=88001)
        second = await client.reconcile_order_state(order_id=771, perm_id=88001)
        return first, second

    first, second = asyncio.run(_run())

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first.get("effective_status") == "Submitted"
    assert second.get("effective_status") == "Submitted"
    assert fake_ib.calls == [
        "all_open",
        "open",
        "executions",
        "all_open",
        "open",
        "executions",
    ]
    assert client._last_order_reconcile_mono == 0.0


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


def test_cold_market_sockets_share_one_settled_main_topology() -> None:
    client = _new_client()
    client._ib = _FakeConnectIB()
    client._ib_proxy = _FakeConnectIB()
    client._ib_index = _FakeConnectIB()
    client._request_reconnect = lambda: None  # type: ignore[method-assign]
    events: list[str] = []

    async def _main() -> None:
        events.append("main")
        await asyncio.sleep(0)
        client._ib.connected = True

    async def _proxy() -> None:
        assert client._ib.isConnected()
        events.append("proxy")
        await asyncio.sleep(0)
        client._ib_proxy.connected = True

    async def _index() -> None:
        assert client._ib.isConnected()
        events.append("index")
        await asyncio.sleep(0)
        client._ib_index.connected = True

    client._connect_main_with_client_id_pool = _main  # type: ignore[method-assign]
    client._connect_proxy_with_client_id_pool = _proxy  # type: ignore[method-assign]
    client._connect_index_with_client_id_pool = _index  # type: ignore[method-assign]

    async def _run() -> None:
        await asyncio.gather(client.connect_proxy(), client.connect_index())

    asyncio.run(_run())

    assert events[0] == "main"
    assert events.count("main") == 1
    assert set(events[1:]) == {"proxy", "index"}


def test_proxy_conflict_resets_settled_topology_without_mixing_pairs() -> None:
    client = _new_client()
    client._ib = _FakeConnectIB(connected=True)
    client._ib_proxy = _FakeConnectIB()
    client._ib_index = _FakeConnectIB(connected=True)
    client._connected_main_client_id = 301
    client._connected_index_client_id = 303
    attempted: list[int] = []

    async def _conflict(_ib, *, client_id: int) -> None:
        attempted.append(int(client_id))
        raise RuntimeError("Client id already in use")

    client._connect_ib = _conflict  # type: ignore[method-assign]
    client._request_reconnect = lambda: None  # type: ignore[method-assign]

    with pytest.raises(ConnectionError, match="topology reset"):
        asyncio.run(client.connect_proxy())

    assert attempted == [302]
    assert int(client._main_client_id) == 301
    assert int(client._proxy_client_id) == 302
    assert client._ib.isConnected() is False
    assert client._ib_index.isConnected() is False
    assert client._connected_main_client_id is None
    assert client._connected_index_client_id is None


def test_main_disconnect_drops_auxiliary_sockets_before_pair_rotation() -> None:
    client = _new_client()
    client._ib = _FakeConnectIB()
    client._ib_proxy = _FakeConnectIB(connected=True)
    client._ib_index = _FakeConnectIB(connected=True)
    client._connected_main_client_id = 301
    client._connected_proxy_client_id = 302
    client._connected_index_client_id = 303
    client._proxy_required = True
    client._index_required = True
    client._start_reconnect_loop = lambda: None  # type: ignore[method-assign]

    client._on_disconnected_main()

    assert client._ib_proxy.isConnected() is False
    assert client._ib_index.isConnected() is False
    assert client._connected_main_client_id is None
    assert client._connected_proxy_client_id is None
    assert client._connected_index_client_id is None
    assert client._resubscribe_main_needed is True
    assert client._resubscribe_proxy_needed is True
    assert client._resubscribe_index_needed is True


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


def test_proven_proxy_route_is_persisted_and_reused(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "ids.json"
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
        client_id_state_file=str(state_path),
    )
    client = IBKRClient(cfg)
    client._ib = _FakeConnectIB(connected=True)
    client._ib_proxy = _FakeConnectIB(connected=True)
    client._connected_main_client_id = 500
    client._connected_proxy_client_id = 501
    contract = Stock("TQQQ", "PEARL", "USD", primaryExchange="NASDAQ")
    contract.conId = 72539702
    ticker = SimpleNamespace(
        contract=contract,
        bid=72.04,
        ask=72.07,
        last=None,
        close=None,
        tbRequestedMdType=1,
    )
    client._proxy_tickers = {"TQQQ": ticker}

    client._remember_proxy_live_route(ticker)  # type: ignore[arg-type]

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["proxy_live_routes"]["TQQQ"]["route"] == "PEARL"

    reloaded = IBKRClient(cfg)
    now = datetime(2026, 8, 6, 7, 0)
    monkeypatch.setattr("tradebot.client._now_et", lambda: now)
    reloaded._proxy_phase_epoch = "2026-08-06:PRE"
    smart = Stock("TQQQ", "SMART", "USD", primaryExchange="NASDAQ")
    smart.conId = 72539702

    requested, md_type = reloaded._proxy_market_data_spec(
        smart,
        include_overnight=False,
    )

    assert md_type == 1
    assert requested.exchange == "PEARL"


def test_proven_route_falls_through_every_other_live_venue(monkeypatch) -> None:
    client = _new_client()
    now = datetime(2026, 8, 6, 7, 0)
    monkeypatch.setattr("tradebot.client._now_et", lambda: now)
    client._proxy_phase_epoch = "2026-08-06:PRE"
    client._proxy_live_route_preferences["TQQQ"] = ("PEARL", int(time.time()))
    failed = Stock("TQQQ", "PEARL", "USD", primaryExchange="NASDAQ")
    failed.conId = 72539702
    client._proxy_contract_live_routes[72539702] = "PEARL"
    attempts: list[int] = []
    client._start_proxy_contract_live_resubscribe = (  # type: ignore[method-assign]
        lambda _contract: attempts.append(1)
    )

    client._start_proxy_contract_market_data_recovery(failed)

    assert attempts == [1]
    assert client._proxy_contract_live_routes[72539702] == "SMART"
    assert client._proxy_live_routes_for_contract(failed, now=now) == (
        "PEARL",
        "SMART",
        "ARCA",
        "DRCTEDGE",
        "MEMX",
    )


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
    now = {"value": datetime(2026, 8, 9, 20, 0)}
    monkeypatch.setattr("tradebot.client._now_et", lambda: now["value"])
    client._proxy_phase_epoch = "2026-08-10:OVERNIGHT"
    client._start_proxy_resubscribe = lambda: None  # type: ignore[method-assign]

    asyncio.run(client._ensure_proxy_tickers())

    first_pass = list(fake_ib.requests)
    assert len(first_pass) == 2
    assert all(str(getattr(contract, "exchange", "")).upper() == "OVERNIGHT" for contract in first_pass)

    now["value"] = datetime(2026, 8, 10, 4, 0)
    asyncio.run(client._ensure_proxy_tickers())

    second_pass = list(fake_ib.requests)[-2:]
    assert len(fake_ib.cancels) >= 2
    assert len(second_pass) == 2
    assert all(str(getattr(contract, "exchange", "")).upper() == "SMART" for contract in second_pass)


def test_ensure_index_tickers_reloads_on_session_change(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_index = fake_ib

    async def _connect() -> None:
        return None

    async def _connect_index() -> None:
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
    client.connect_index = _connect_index  # type: ignore[method-assign]
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
    client._ib_index = fake_ib
    client._index_force_delayed = True

    async def _connect() -> None:
        return None

    async def _connect_index() -> None:
        return None

    async def _qualify_index_contracts() -> dict[str, object]:
        return {
            "NQ": SimpleNamespace(symbol="NQ", exchange="CME", secType="FUT", conId=8201),
        }

    state = {"open": True}

    def _ladder(_now):
        return (1, 2, 3, 4) if state["open"] else (1, 2, 4, 3)

    client.connect = _connect  # type: ignore[method-assign]
    client.connect_index = _connect_index  # type: ignore[method-assign]
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


def test_index_live_strip_reconciles_at_futures_session_transition(monkeypatch) -> None:
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
    assert client._index_force_delayed is False
    assert calls == [False]


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


def test_on_error_main_index_permission_does_not_mutate_index_role() -> None:
    client = _new_client()
    client._index_contracts = {
        "NQ": SimpleNamespace(symbol="NQ", secType="FUT", conId=93001),
    }
    called = {"value": False}

    def _start_index_resubscribe() -> None:
        called["value"] = True

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]

    contract = SimpleNamespace(symbol="NQ", secType="FUT", conId=93001)
    client._on_error_main(0, 354, "No market data subscription", contract)

    assert client._index_force_delayed is False
    assert called["value"] is False


def test_on_error_index_permission_transitions_to_delayed_once() -> None:
    client = _new_client()
    contract = SimpleNamespace(symbol="NQ", secType="FUT", conId=93001)
    client._index_contracts = {"NQ": contract}
    calls = 0

    def _start_index_resubscribe() -> None:
        nonlocal calls
        calls += 1

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]

    for _ in range(20):
        client._on_error_index(1, 354, "No market data subscription", contract)

    assert client._index_force_delayed is True
    assert calls == 1
    assert client.index_error() == "No market data subscription"


def test_index_entitlement_feedback_has_four_request_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(
        client_module,
        "_now_et",
        lambda: datetime(2026, 8, 6, 14, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_index = fake_ib
    client._index_required = True
    client._index_contracts = {
        symbol: Contract(
            secType="FUT",
            symbol=symbol,
            exchange="CME",
            currency="USD",
            conId=con_id,
        )
        for symbol, con_id in (("NQ", 93001), ("ES", 93002))
    }

    async def _connect() -> None:
        return None

    client.connect = _connect  # type: ignore[method-assign]
    client.connect_index = _connect  # type: ignore[method-assign]
    client._start_index_probe = lambda: None  # type: ignore[method-assign]

    async def _exercise() -> None:
        await client._ensure_index_tickers()
        assert len(fake_ib.requests) == 2

        client._on_error_index(
            1,
            354,
            "No market data subscription",
            client._index_contracts["NQ"],
        )
        assert client._index_task is not None
        await client._index_task

        for _ in range(20):
            client._on_error_index(
                2,
                354,
                "No market data subscription",
                client._index_contracts["ES"],
            )
            client.start_index_tickers()
            assert client._index_task is not None
            await client._index_task

    asyncio.run(_exercise())

    assert len(fake_ib.requests) == 4
    assert len(fake_ib.cancels) == 2
    assert fake_ib.market_data_types == [1, 1, 3, 3]


def test_index_resubscribe_queues_once_behind_active_load() -> None:
    client = _new_client()
    client._index_required = True
    reloads = 0

    async def _reload() -> None:
        nonlocal reloads
        reloads += 1

    client._reload_index_tickers = _reload  # type: ignore[method-assign]

    async def _exercise() -> None:
        active = asyncio.create_task(asyncio.sleep(0))
        client._index_task = active
        for _ in range(20):
            client._start_index_resubscribe()
            client.start_index_tickers()
        await active
        await asyncio.sleep(0)
        assert client._index_task is not None
        await client._index_task

    asyncio.run(_exercise())

    assert reloads == 1


def test_on_error_main_future_permission_arms_watchdog_without_starting_probe() -> None:
    client = _new_client()
    contract = Contract(secType="FUT", symbol="1OZ", exchange="COMEX", currency="USD")
    contract.conId = 753716628
    ticker = SimpleNamespace(contract=contract, bid=None, ask=None, last=None)
    client._detail_tickers[int(contract.conId)] = (client._ib, ticker)
    seen_probe: list[int] = []
    seen_watchdog: list[int] = []
    resubscribe_md: list[int | None] = []

    def _start_probe(req_contract) -> None:
        seen_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    def _start_watchdog(req_contract) -> None:
        seen_watchdog.append(int(getattr(req_contract, "conId", 0) or 0))

    def _resubscribe(_ticker, *, md_type_override: int | None = None):
        resubscribe_md.append(md_type_override)
        return _ticker

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    client._start_main_contract_quote_watchdog = _start_watchdog  # type: ignore[method-assign]
    client._resubscribe_main_contract_stream = _resubscribe  # type: ignore[method-assign]

    client._on_error_main(0, 354, "No market data subscription", contract)

    assert seen_probe == []
    assert seen_watchdog == [753716628]
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
    }
    calls = 0

    def _start_index_resubscribe() -> None:
        nonlocal calls
        calls += 1

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())

    assert client._index_force_delayed is True
    assert calls == 1


def test_probe_index_quotes_does_not_degrade_when_close_only_present(monkeypatch) -> None:
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
    }
    calls = 0

    def _start_index_resubscribe() -> None:
        nonlocal calls
        calls += 1

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())

    assert client._index_force_delayed is False
    assert calls == 0


def test_probe_index_quotes_partial_strip_preserves_healthy_subscription(monkeypatch) -> None:
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
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        ),
    }
    calls = 0

    def _start_index_resubscribe() -> None:
        nonlocal calls
        calls += 1

    async def _sleep(_: float) -> None:
        return None

    client._start_index_resubscribe = _start_index_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._probe_index_quotes())
    asyncio.run(client._probe_index_quotes())
    asyncio.run(client._probe_index_quotes())

    assert client._index_force_delayed is False
    assert calls == 0
    assert client.index_error() == "market data unavailable: ES"


def test_ensure_ticker_overnight_delayed_uses_primary_listing(monkeypatch) -> None:
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
    assert str(getattr(requested, "exchange", "")).upper() == "ARCA"


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


def test_ensure_ticker_reconciles_market_data_type_once_even_when_route_is_unchanged(
    monkeypatch,
) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (True, False))
    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.conId = 900002
    live = SimpleNamespace(
        contract=contract,
        marketDataType=1,
        tbRequestedMdType=1,
        tbGenericTicks="",
        bid=None,
        ask=None,
        last=None,
        close=72.84,
        prevLast=72.84,
    )
    client._detail_tickers[int(contract.conId)] = (fake_ib, live)
    client._proxy_contract_force_delayed.add(int(contract.conId))

    first = asyncio.run(client.ensure_ticker(contract, owner="test"))
    second = asyncio.run(client.ensure_ticker(contract, owner="test"))

    assert first is second
    assert first is not live
    assert int(first.tbRequestedMdType) == 3
    assert fake_ib.market_data_types == [3, 3]
    assert len(fake_ib.requests) == 1
    assert len(fake_ib.cancels) == 1


def test_proxy_error_10167_is_scoped_to_contract(monkeypatch) -> None:
    client = _new_client()
    called: dict[str, int] = {"global": 0, "contract": 0}

    def _global() -> None:
        called["global"] += 1

    def _contract(_contract) -> None:
        called["contract"] += 1

    client._start_proxy_resubscribe = _global  # type: ignore[method-assign]
    client._start_proxy_contract_market_data_recovery = _contract  # type: ignore[method-assign]

    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.conId = 12345
    client._on_error_proxy(0, 10167, "No market data permissions", contract)

    assert called["contract"] == 1
    assert called["global"] == 0
    assert client._proxy_contract_force_delayed == set()


def test_proxy_partial_entitlement_warning_waits_for_quote_probe() -> None:
    client = _new_client()
    client._proxy_probe_complete = True
    probes: list[int] = []
    recoveries: list[int] = []
    client._start_proxy_probe = lambda: probes.append(1)  # type: ignore[method-assign]
    client._start_proxy_contract_market_data_recovery = (  # type: ignore[method-assign]
        lambda contract: recoveries.append(int(getattr(contract, "conId", 0) or 0))
    )
    contract = Stock(symbol="TQQQ", exchange="ARCA", currency="USD")
    contract.conId = 12345

    client._on_error_proxy(0, 10091, "Partial market-data subscription", contract)

    assert client._proxy_probe_complete is False
    assert probes == [1]
    assert recoveries == []
    assert client._proxy_contract_force_delayed == set()


def test_proxy_live_route_ladders_are_session_appropriate() -> None:
    assert client_module._proxy_live_route_ladder(
        datetime(2026, 8, 6, 2, 0, tzinfo=timezone.utc)
    ) == ("OVERNIGHT",)
    expected_lit = ("SMART", "ARCA", "DRCTEDGE", "MEMX", "PEARL")
    assert client_module._proxy_live_route_ladder(
        datetime(2026, 8, 6, 8, 0, tzinfo=timezone.utc)
    ) == expected_lit
    assert client_module._proxy_live_route_ladder(
        datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    ) == expected_lit
    assert client_module._proxy_live_route_ladder(
        datetime(2026, 8, 6, 18, 0, tzinfo=timezone.utc)
    ) == expected_lit


def test_foreign_stock_never_inherits_us_overnight_or_lit_routes(monkeypatch) -> None:
    client = _new_client()
    now = datetime(2026, 8, 6, 21, 0)
    monkeypatch.setattr("tradebot.client._now_et", lambda: now)
    client._proxy_phase_epoch = "2026-08-07:OVERNIGHT"
    contract = Stock("ORC", "IBIS", "EUR", primaryExchange="IBIS")
    contract.conId = 11652126

    requested, md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=True,
    )

    assert md_type == 1
    assert requested.exchange == "IBIS"

    delayed: list[int] = []
    client._start_proxy_contract_delayed_resubscribe = (  # type: ignore[method-assign]
        lambda value: delayed.append(int(getattr(value, "conId", 0) or 0))
    )
    client._start_proxy_contract_market_data_recovery(contract)

    assert delayed == [11652126]
    assert 11652126 in client._proxy_contract_force_delayed
    assert 11652126 not in client._proxy_contract_live_routes


def test_proxy_stock_recovery_walks_live_routes_before_delayed(monkeypatch) -> None:
    client = _new_client()
    client._proxy_phase_epoch = "2026-08-06:PRE"
    con_id = 12345
    live_starts: list[int] = []
    delayed_starts: list[int] = []

    def _live(contract) -> None:
        live_starts.append(int(getattr(contract, "conId", 0) or 0))

    def _delayed(contract) -> None:
        delayed_starts.append(int(getattr(contract, "conId", 0) or 0))

    client._start_proxy_contract_live_resubscribe = _live  # type: ignore[method-assign]
    client._start_proxy_contract_delayed_resubscribe = _delayed  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 6, 5, 0))
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 100.0)

    for failed, expected in (
        ("SMART", "ARCA"),
        ("ARCA", "DRCTEDGE"),
        ("DRCTEDGE", "MEMX"),
        ("MEMX", "PEARL"),
    ):
        contract = Stock(symbol="TQQQ", exchange=failed, currency="USD")
        contract.conId = con_id
        client._start_proxy_contract_market_data_recovery(contract)
        assert client._proxy_contract_live_routes[con_id] == expected

    pearl = Stock(symbol="TQQQ", exchange="PEARL", currency="USD")
    pearl.conId = con_id
    client._start_proxy_contract_market_data_recovery(pearl)

    assert live_starts == [con_id, con_id, con_id, con_id]
    assert delayed_starts == [con_id]
    assert con_id in client._proxy_contract_force_delayed
    assert con_id not in client._proxy_contract_live_routes
    assert client._proxy_contract_live_retry_at_mono == {con_id: 400.0}


def test_top_strip_races_live_routes_and_keeps_only_first_actionable(monkeypatch) -> None:
    client = _new_client()
    client._ib = _FakeConnectIB(connected=True)
    now = datetime(2026, 8, 6, 5, 0)
    monkeypatch.setattr("tradebot.client._now_et", lambda: now)
    client._proxy_phase_epoch = "2026-08-06:PRE"

    class _RaceIB:
        def __init__(self) -> None:
            self.requests: list[object] = []
            self.cancels: list[object] = []

        @staticmethod
        def isConnected() -> bool:
            return True

        @staticmethod
        def reqMarketDataType(_md_type: int) -> None:
            return None

        def reqMktData(self, contract):
            self.requests.append(contract)
            live = str(getattr(contract, "exchange", "") or "") == "PEARL"
            return SimpleNamespace(
                contract=contract,
                bid=72.04 if live else None,
                ask=72.07 if live else None,
                last=None,
                close=None,
                marketDataType=1,
            )

        def cancelMktData(self, contract) -> None:
            self.cancels.append(contract)

    client._ib_proxy = _RaceIB()  # type: ignore[assignment]
    contract = Stock("TQQQ", "SMART", "USD", primaryExchange="NASDAQ")
    contract.conId = 72539702
    current = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        tbRequestedMdType=1,
    )
    client._proxy_tickers = {"TQQQ": current}

    async def _run() -> None:
        task = client._start_proxy_contract_market_data_recovery(contract)
        assert task is not None
        await task

    asyncio.run(_run())

    requested_routes = {
        str(getattr(request, "exchange", "") or "")
        for request in client._ib_proxy.requests
    }
    assert requested_routes == {"ARCA", "DRCTEDGE", "MEMX", "PEARL"}
    winner = client._proxy_tickers["TQQQ"]
    assert winner.contract.exchange == "PEARL"
    assert client._ticker_has_data(winner) is True
    assert client._proxy_contract_live_routes[72539702] == "PEARL"
    assert 72539702 not in client._proxy_contract_force_delayed
    cancelled_routes = {
        str(getattr(request, "exchange", "") or "")
        for request in client._ib_proxy.cancels
    }
    assert cancelled_routes == {"SMART", "ARCA", "DRCTEDGE", "MEMX"}


def test_searched_stock_races_live_routes_then_updates_detail_ticker(monkeypatch) -> None:
    client = _new_client()
    client._ib = _FakeConnectIB(connected=True)
    now = datetime(2026, 8, 6, 9, 35)
    monkeypatch.setattr("tradebot.client._now_et", lambda: now)
    client._proxy_phase_epoch = "2026-08-06:RTH"

    class _RaceIB:
        def __init__(self) -> None:
            self.requests: list[object] = []
            self.cancels: list[object] = []

        @staticmethod
        def isConnected() -> bool:
            return True

        @staticmethod
        def reqMarketDataType(_md_type: int) -> None:
            return None

        def reqMktData(self, contract):
            self.requests.append(contract)
            live = str(getattr(contract, "exchange", "") or "") == "PEARL"
            return SimpleNamespace(
                contract=contract,
                bid=141.52 if live else None,
                ask=141.72 if live else None,
                last=None,
                close=None,
                marketDataType=1,
            )

        def cancelMktData(self, contract) -> None:
            self.cancels.append(contract)

    fake_ib = _RaceIB()
    client._ib_proxy = fake_ib  # type: ignore[assignment]
    contract = Stock("ORCL", "SMART", "USD", primaryExchange="NYSE")
    contract.conId = 272800
    current = SimpleNamespace(
        contract=contract,
        bid=None,
        ask=None,
        last=None,
        close=None,
        tbRequestedMdType=1,
    )
    client._detail_tickers[int(contract.conId)] = (fake_ib, current)

    async def _run() -> None:
        task = client._start_proxy_contract_market_data_recovery(contract)
        assert task is not None
        await task

    asyncio.run(_run())

    winner = client.ticker_for_con_id(int(contract.conId))
    assert winner is not None
    assert winner.contract.exchange == "PEARL"
    assert client._ticker_has_data(winner) is True
    assert client._proxy_contract_live_routes[272800] == "PEARL"
    assert 272800 not in client._proxy_contract_force_delayed
    assert str(getattr(winner, "tbQuoteSource", "")) == "stream"
    requested_routes = {
        str(getattr(request, "exchange", "") or "")
        for request in fake_ib.requests
    }
    assert requested_routes == {"ARCA", "DRCTEDGE", "MEMX", "PEARL"}


def test_proxy_delayed_contracts_retry_live_after_cooldown(monkeypatch) -> None:
    client = _new_client()
    client._proxy_contract_force_delayed = {1, 2}
    client._proxy_contract_live_retry_at_mono = {1: 99.0, 2: 101.0}
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 100.0)

    assert client._release_due_proxy_live_retries() is True
    assert client._proxy_contract_force_delayed == {2}
    assert client._proxy_contract_live_retry_at_mono == {2: 101.0}
    assert client._release_due_proxy_live_retries() is False


def test_global_proxy_and_index_delayed_modes_retry_live_after_cooldown(monkeypatch) -> None:
    client = _new_client()
    clock = {"value": 100.0}
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: clock["value"])

    assert client._degrade_proxy_to_delayed() is True
    assert client._degrade_index_to_delayed() is True
    assert client._proxy_live_retry_at_mono == 400.0
    assert client._index_live_retry_at_mono == 400.0
    assert client._release_due_proxy_live_retries() is False
    assert client._release_due_index_live_retry() is False

    clock["value"] = 400.0
    assert client._release_due_proxy_live_retries() is True
    assert client._release_due_index_live_retry() is True
    assert client._proxy_force_delayed is False
    assert client._index_force_delayed is False


def test_proxy_stock_recovery_ignores_stale_route_errors(monkeypatch) -> None:
    client = _new_client()
    client._proxy_phase_epoch = "2026-08-06:PRE"
    con_id = 12345
    client._proxy_contract_live_routes[con_id] = "ARCA"
    starts: list[int] = []
    client._start_proxy_contract_live_resubscribe = (  # type: ignore[method-assign]
        lambda contract: starts.append(int(getattr(contract, "conId", 0) or 0))
    )
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 6, 5, 0))

    stale = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    stale.conId = con_id
    client._start_proxy_contract_market_data_recovery(stale)

    assert client._proxy_contract_live_routes[con_id] == "ARCA"
    assert starts == []
    assert client._proxy_contract_force_delayed == set()


def test_proxy_non_stock_entitlement_error_does_not_use_equity_routes() -> None:
    client = _new_client()
    contract = Contract(
        secType="OPT",
        symbol="TQQQ",
        exchange="SMART",
        currency="USD",
    )
    contract.conId = 54321
    delayed: list[int] = []
    client._start_proxy_contract_delayed_resubscribe = (  # type: ignore[method-assign]
        lambda req: delayed.append(int(getattr(req, "conId", 0) or 0))
    )

    client._start_proxy_contract_market_data_recovery(contract)

    assert delayed == [54321]
    assert client._proxy_contract_force_delayed == {54321}
    assert client._proxy_contract_live_routes == {}


def test_proxy_market_data_spec_caches_live_route_only_within_phase_epoch(monkeypatch) -> None:
    client = _new_client()
    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.primaryExchange = "NASDAQ"
    contract.conId = 12345
    now = {"value": datetime(2026, 8, 6, 5, 0)}
    monkeypatch.setattr("tradebot.client._now_et", lambda: now["value"])
    client._proxy_phase_epoch = "2026-08-06:PRE"
    client._proxy_contract_live_routes[12345] = "DRCTEDGE"

    live_contract, live_md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=False,
    )
    delayed_contract, delayed_md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=True,
        requested_md_type=3,
    )

    assert str(live_contract.exchange).upper() == "DRCTEDGE"
    assert live_md_type == 1
    assert str(delayed_contract.exchange).upper() == "NASDAQ"
    assert delayed_md_type == 3

    now["value"] = datetime(2026, 8, 6, 9, 30)
    client._reconcile_proxy_market_phase()
    next_session_contract, next_session_md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=False,
    )
    assert str(next_session_contract.exchange).upper() == "SMART"
    assert next_session_md_type == 1


def test_proxy_market_data_spec_never_reuses_stale_epoch_degradation(monkeypatch) -> None:
    client = _new_client()
    contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")
    contract.primaryExchange = "NASDAQ"
    contract.conId = 12345
    client._proxy_phase_epoch = "2026-08-06:PRE"
    client._proxy_force_delayed = True
    client._proxy_contract_force_delayed = {12345}
    client._proxy_contract_live_routes = {12345: "PEARL"}
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 7, 5, 0))

    request_contract, md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=False,
    )

    assert md_type == 1
    assert str(request_contract.exchange).upper() == "SMART"


def test_proxy_market_data_spec_prefers_overnight_only_overnight(monkeypatch) -> None:
    client = _new_client()
    contract = Stock(symbol="QQQ", exchange="SMART", currency="USD")
    contract.primaryExchange = "NASDAQ"
    contract.conId = 320227571
    monkeypatch.setattr("tradebot.client._session_bucket", lambda _now: "OVERNIGHT")

    live_contract, md_type = client._proxy_market_data_spec(
        contract,
        include_overnight=True,
    )

    assert str(live_contract.exchange).upper() == "OVERNIGHT"
    assert md_type == 1


def test_proxy_ensure_retains_healthy_directed_route_without_churn(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    contracts: dict[str, Contract] = {}
    for index, symbol in enumerate(("QQQ", "SPY", "DIA", "TQQQ"), 1):
        contract = Stock(symbol=symbol, exchange="SMART", currency="USD")
        contract.primaryExchange = "NASDAQ" if symbol in ("QQQ", "TQQQ") else "ARCA"
        contract.conId = index
        contracts[symbol] = contract
    client._proxy_contracts = contracts
    client._proxy_phase_epoch = "2026-08-06:PRE"
    client._proxy_contract_live_routes[1] = "ARCA"
    client._proxy_contract_live_routes[4] = "DRCTEDGE"
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 6, 5, 0))

    asyncio.run(client._ensure_proxy_tickers())
    retained = dict(client._proxy_tickers)
    request_count = len(fake_ib.requests)
    cancel_count = len(fake_ib.cancels)
    asyncio.run(client._ensure_proxy_tickers())

    assert len(fake_ib.requests) == request_count == 4
    assert len(fake_ib.cancels) == cancel_count == 0
    assert all(client._proxy_tickers[symbol] is ticker for symbol, ticker in retained.items())
    assert str(client._proxy_tickers["QQQ"].contract.exchange).upper() == "ARCA"
    assert str(client._proxy_tickers["TQQQ"].contract.exchange).upper() == "DRCTEDGE"


def test_proxy_strip_requires_every_symbol_instead_of_any_healthy_symbol() -> None:
    client = _new_client()
    client._proxy_tickers = {
        symbol: SimpleNamespace(
            contract=Stock(symbol=symbol, exchange="SMART", currency="USD"),
            bid=600.0 if symbol != "QQQ" else None,
            ask=600.1 if symbol != "QQQ" else None,
            last=None,
            close=None,
        )
        for symbol in ("QQQ", "SPY", "DIA", "TQQQ")
    }

    assert client._proxy_has_data() is False
    assert client._proxy_symbols_without_data() == ("QQQ",)

    client._proxy_tickers["QQQ"].bid = 500.0
    client._proxy_tickers["QQQ"].ask = 500.1
    assert client._proxy_has_data() is True


def test_proxy_strip_accepts_close_only_as_stable_display_data() -> None:
    client = _new_client()
    client._proxy_tickers = {
        symbol: SimpleNamespace(
            contract=Stock(symbol=symbol, exchange="NASDAQ", currency="USD"),
            bid=None,
            ask=None,
            last=None,
            close=500.0,
            prevLast=500.0,
        )
        for symbol in ("QQQ", "SPY", "DIA", "TQQQ")
    }

    assert client._proxy_symbols_without_data() == ()
    assert client._proxy_has_data() is True


def test_proxy_directed_live_quote_is_primed_with_historical_close() -> None:
    client = _new_client()
    contract = Stock(symbol="QQQ", exchange="SMART", currency="USD")
    contract.primaryExchange = "NASDAQ"
    contract.conId = 320227571
    ticker_contract = Stock(symbol="QQQ", exchange="ARCA", currency="USD")
    ticker_contract.primaryExchange = "NASDAQ"
    ticker_contract.conId = int(contract.conId)
    ticker = SimpleNamespace(
        contract=ticker_contract,
        bid=714.20,
        ask=714.28,
        last=None,
        close=None,
        prevLast=None,
    )
    client._proxy_contracts = {"QQQ": contract}
    client._proxy_tickers = {"QQQ": ticker}
    requested: list[str] = []
    updates: list[int] = []

    async def _anchors(req_contract):
        requested.append(str(getattr(req_contract, "exchange", "") or ""))
        return 717.30, 723.85, 687.99

    client.session_close_anchors = _anchors  # type: ignore[method-assign]
    client._on_stream_update = lambda *_args, **_kwargs: updates.append(1)  # type: ignore[method-assign]

    asyncio.run(client._prime_proxy_close_baselines())

    assert requested == ["SMART"]
    assert ticker.close == pytest.approx(717.30)
    assert ticker.tbCloseSource == "historical-daily"
    assert updates == [1]


def test_proxy_ensure_respects_per_symbol_delayed_routes_without_reloading_healthy_symbols(
    monkeypatch,
) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib

    async def _connect_proxy() -> None:
        return None

    client.connect_proxy = _connect_proxy  # type: ignore[method-assign]
    contracts: dict[str, Contract] = {}
    for index, symbol in enumerate(("QQQ", "SPY", "DIA", "TQQQ"), 1):
        contract = Stock(symbol=symbol, exchange="SMART", currency="USD")
        contract.primaryExchange = "NASDAQ" if symbol in ("QQQ", "TQQQ") else "ARCA"
        contract.conId = index
        contracts[symbol] = contract
    client._proxy_contracts = contracts
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (True, False))

    asyncio.run(client._ensure_proxy_tickers())
    spy_ticker = client._proxy_tickers["SPY"]
    dia_ticker = client._proxy_tickers["DIA"]

    for symbol in ("QQQ", "TQQQ"):
        client._proxy_contract_force_delayed.add(int(contracts[symbol].conId))
        asyncio.run(client._resubscribe_proxy_contract_delayed(contracts[symbol]))

    request_count = len(fake_ib.requests)
    cancel_count = len(fake_ib.cancels)
    qqq_ticker = client._proxy_tickers["QQQ"]
    tqqq_ticker = client._proxy_tickers["TQQQ"]

    asyncio.run(client._ensure_proxy_tickers())

    assert len(fake_ib.requests) == request_count
    assert len(fake_ib.cancels) == cancel_count
    assert client._proxy_tickers["SPY"] is spy_ticker
    assert client._proxy_tickers["DIA"] is dia_ticker
    assert client._proxy_tickers["QQQ"] is qqq_ticker
    assert client._proxy_tickers["TQQQ"] is tqqq_ticker
    assert str(qqq_ticker.contract.exchange).upper() == "NASDAQ"
    assert str(tqqq_ticker.contract.exchange).upper() == "NASDAQ"
    assert int(qqq_ticker.tbRequestedMdType) == 3
    assert int(tqqq_ticker.tbRequestedMdType) == 3


def test_missing_proxy_recovery_routes_only_the_empty_contract() -> None:
    client = _new_client()
    contracts: dict[str, Contract] = {}
    for index, symbol in enumerate(("QQQ", "SPY", "DIA", "TQQQ"), 1):
        contract = Stock(symbol=symbol, exchange="SMART", currency="USD")
        contract.conId = index
        contracts[symbol] = contract
    client._proxy_contracts = contracts
    recovered: list[int] = []

    def _recover(contract) -> None:
        recovered.append(int(getattr(contract, "conId", 0) or 0))

    client._start_proxy_contract_market_data_recovery = _recover  # type: ignore[method-assign]

    asyncio.run(
        client._resubscribe_missing_proxy_quotes(
            ("QQQ",),
            requalify=False,
        )
    )

    assert recovered == [1]
    assert client._proxy_contract_force_delayed == set()
    assert client._proxy_force_delayed is False


def test_proxy_probe_retries_empty_symbol_once_with_requalification(
    monkeypatch,
) -> None:
    client = _new_client()
    client._proxy_tickers = {
        symbol: SimpleNamespace(
            contract=Stock(symbol=symbol, exchange="SMART", currency="USD"),
            bid=600.0 if symbol != "QQQ" else None,
            ask=600.1 if symbol != "QQQ" else None,
            last=None,
            close=None,
        )
        for symbol in ("QQQ", "SPY", "DIA", "TQQQ")
    }
    calls: list[tuple[tuple[str, ...], bool]] = []
    prime_ready_states: list[bool] = []

    async def _sleep(_seconds: float) -> None:
        return

    async def _recover(symbols, *, requalify: bool) -> None:
        calls.append((tuple(symbols), requalify))
        if requalify:
            client._proxy_tickers["QQQ"].bid = 500.0
            client._proxy_tickers["QQQ"].ask = 500.1

    async def _prime() -> None:
        prime_ready_states.append(client._proxy_probe_complete)

    monkeypatch.setattr("asyncio.sleep", _sleep)
    client._resubscribe_missing_proxy_quotes = _recover  # type: ignore[method-assign]
    client._prime_proxy_close_baselines = _prime  # type: ignore[method-assign]

    asyncio.run(client._probe_proxy_quotes())

    assert calls == [(("QQQ",), False), (("QQQ",), True)]
    assert client._proxy_force_delayed is False
    assert client._proxy_probe_complete is True
    assert client._proxy_probe_failures == 0
    assert client._proxy_probe_retry_at_mono == 0.0
    assert prime_ready_states == [True]


def test_proxy_probe_backs_off_after_bounded_empty_recovery(
    monkeypatch,
) -> None:
    client = _new_client()
    client._proxy_tickers = {
        symbol: SimpleNamespace(
            contract=Stock(symbol=symbol, exchange="SMART", currency="USD"),
            bid=None,
            ask=None,
            last=None,
            close=None,
            prevLast=None,
        )
        for symbol in ("QQQ", "SPY", "DIA", "TQQQ")
    }
    calls: list[tuple[tuple[str, ...], bool]] = []

    async def _sleep(_seconds: float) -> None:
        return

    async def _recover(symbols, *, requalify: bool) -> None:
        calls.append((tuple(symbols), requalify))

    monkeypatch.setattr("asyncio.sleep", _sleep)
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("tradebot.client.random.uniform", lambda _low, _high: 0.0)
    monkeypatch.setattr(
        "tradebot.client._proxy_live_route_ladder",
        lambda _now: ("SMART", "ARCA", "DRCTEDGE", "MEMX", "PEARL"),
    )
    client._resubscribe_missing_proxy_quotes = _recover  # type: ignore[method-assign]

    asyncio.run(client._probe_proxy_quotes())

    expected = ("QQQ", "SPY", "DIA", "TQQQ")
    assert calls == [
        (expected, False),
        (expected, True),
        (expected, False),
        (expected, False),
        (expected, False),
    ]
    assert client._proxy_probe_complete is False
    assert client._proxy_probe_failures == 1
    assert client._proxy_probe_retry_at_mono == pytest.approx(115.0)


def test_proxy_probe_does_not_restart_after_session_is_settled() -> None:
    client = _new_client()
    client._proxy_probe_complete = True

    client._start_proxy_probe()

    assert client._proxy_probe_task is None


def test_probe_proxy_contract_quote_retries_live_without_forcing_delayed(
    monkeypatch,
) -> None:
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


def test_attempt_main_contract_snapshot_quote_is_disabled(monkeypatch) -> None:
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

    assert ok is False
    assert getattr(client._ib, "calls", 0) == 0


def test_refresh_live_snapshot_once_is_disabled(monkeypatch) -> None:
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

    assert source is None
    assert client._ib.market_data_types == []


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


def test_watch_main_contract_quote_does_not_start_probe_when_quote_is_stale(monkeypatch) -> None:
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

    client._start_main_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client.time.monotonic", lambda: 200.0)

    async def _sleep(_: float) -> None:
        client._detail_tickers.pop(int(contract.conId), None)
        return None

    monkeypatch.setattr("asyncio.sleep", _sleep)
    asyncio.run(client._watch_main_contract_quote(contract))

    assert seen == []


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

    assert seen_probe == []
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


def test_ensure_ticker_future_does_not_start_main_probe_when_stream_empty(monkeypatch) -> None:
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
    assert probes == []


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
    assert started_probe == []


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


def test_proxy_contract_delayed_flags_clear_when_phase_epoch_changes(monkeypatch) -> None:
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
    client._proxy_phase_epoch = "2026-08-06:PRE"

    started_live: list[int] = []
    started_probe: list[int] = []

    def _start_live(req_contract) -> None:
        started_live.append(int(getattr(req_contract, "conId", 0) or 0))

    def _start_probe(req_contract) -> None:
        started_probe.append(int(getattr(req_contract, "conId", 0) or 0))

    client._start_proxy_contract_live_resubscribe = _start_live  # type: ignore[method-assign]
    client._start_proxy_contract_quote_probe = _start_probe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 6, 9, 30))

    phase = client._reconcile_proxy_market_phase()

    assert phase.name == "RTH"
    assert int(contract.conId) not in client._proxy_contract_force_delayed
    assert started_live == [550011]
    assert started_probe == [550011]


def test_proxy_top_row_resubscribes_when_overnight_route_flips(monkeypatch) -> None:
    client = _new_client()
    client._proxy_phase_epoch = "2026-08-06:POST"
    client._proxy_tickers = {
        "QQQ": SimpleNamespace(contract=Stock(symbol="QQQ", exchange="SMART", currency="USD"))
    }
    calls: list[int] = []

    def _start_proxy_resubscribe() -> None:
        calls.append(1)

    client._start_proxy_resubscribe = _start_proxy_resubscribe  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 6, 20, 0))

    phase = client._reconcile_proxy_market_phase()

    assert phase.name == "OVERNIGHT"
    assert phase.include_overnight is True
    assert client._proxy_phase_epoch == "2026-08-07:OVERNIGHT"
    assert calls == [1]


def test_proxy_top_row_reconciles_live_routes_across_same_named_next_day_phase(
    monkeypatch,
) -> None:
    client = _new_client()
    client._proxy_phase_epoch = "2026-08-06:PRE"
    client._proxy_probe_complete = True
    client._proxy_force_delayed = True
    client._proxy_live_retry_at_mono = 999.0
    client._proxy_contract_force_delayed = {72539702}
    client._proxy_contract_live_routes = {72539702: "DRCTEDGE"}
    client._proxy_contract_live_retry_at_mono = {72539702: 999.0}
    client._proxy_tickers = {
        "TQQQ": SimpleNamespace(
            contract=Stock(symbol="TQQQ", exchange="DRCTEDGE", currency="USD")
        )
    }
    calls: list[int] = []
    client._start_proxy_resubscribe = lambda: calls.append(1)  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 7, 5, 0))

    phase = client._reconcile_proxy_market_phase()

    assert phase.name == "PRE"
    assert client._proxy_phase_epoch == "2026-08-07:PRE"
    assert client._proxy_probe_complete is False
    assert client._proxy_force_delayed is False
    assert client._proxy_live_retry_at_mono == 0.0
    assert client._proxy_contract_force_delayed == set()
    assert client._proxy_contract_live_routes == {}
    assert client._proxy_contract_live_retry_at_mono == {}
    assert calls == [1]


def test_proxy_top_row_stops_requests_during_closed_phase(monkeypatch) -> None:
    client = _new_client()
    fake_ib = _FakeProxyIB()
    client._ib_proxy = fake_ib
    client._proxy_phase_epoch = "2026-08-07:POST"
    client._proxy_tickers = {
        "TQQQ": SimpleNamespace(
            contract=Stock(symbol="TQQQ", exchange="SMART", currency="USD")
        )
    }
    calls: list[int] = []
    client._start_proxy_resubscribe = lambda: calls.append(1)  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._now_et", lambda: datetime(2026, 8, 7, 20, 0))

    phase = client._reconcile_proxy_market_phase()

    assert (phase.name, phase.tradable) == ("CLOSED", False)
    assert client._proxy_tickers == {}
    assert len(fake_ib.cancels) == 1
    assert calls == []


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
            duration_str="5 D",
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
    assert request.get("duration_str") == "5 D"
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


def test_historical_timeout_sec_normalizes_and_applies_overrides() -> None:
    # Router warmup floors and heal fallbacks can require larger intraday history windows,
    # which should not inherit the tiny base timeout.
    assert IBKRClient._historical_timeout_sec("1 D") == 25.0
    assert IBKRClient._historical_timeout_sec("1D") == 25.0
    assert IBKRClient._historical_timeout_sec("2 D") == 30.0
    assert IBKRClient._historical_timeout_sec("2D") == 30.0
    assert IBKRClient._historical_timeout_sec("1 W") == 45.0
    assert IBKRClient._historical_timeout_sec("1W") == 45.0
    assert IBKRClient._historical_timeout_sec("2 W") == 60.0
    assert IBKRClient._historical_timeout_sec("2W") == 60.0
    assert IBKRClient._historical_timeout_sec("6 M") == 180.0
    assert IBKRClient._historical_timeout_sec("6M") == 180.0
    assert IBKRClient._historical_timeout_sec("1 Y") == 240.0
    assert IBKRClient._historical_timeout_sec("1Y") == 240.0
    assert IBKRClient._historical_timeout_sec("2 Y") == 300.0
    assert IBKRClient._historical_timeout_sec("2Y") == 300.0


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
