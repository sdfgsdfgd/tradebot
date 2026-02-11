"""Thin async wrapper over ib_insync for snapshot-style position pulls."""
from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from ib_insync import (
    AccountValue,
    ContFuture,
    Contract,
    Forex,
    Future,
    IB,
    LimitOrder,
    PnL,
    PortfolioItem,
    Stock,
    Ticker,
    Trade,
    util,
)

from .config import IBKRConfig

# region Constants
_INDEX_CONTRACTS: dict[str, list[str]] = {
    "NQ": ["GLOBEX", "CME"],
    "ES": ["GLOBEX", "CME"],
    "YM": ["ECBOT", "CBOT", "GLOBEX"],
}
_PROXY_SYMBOLS = ("QQQ", "TQQQ")
_ET_ZONE = ZoneInfo("America/New_York")
_PREMARKET_START = dtime(4, 0)
_RTH_START = dtime(9, 30)
_RTH_END = dtime(16, 0)
_AFTER_END = dtime(20, 0)
_OVERNIGHT_END = dtime(3, 50)
# endregion


# region Models
@dataclass(frozen=True)
class OhlcvBar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
# endregion


# region Session Helpers
def _session_flags(now: datetime) -> tuple[bool, bool]:
    """Return (outside_rth, include_overnight) for US equity sessions."""
    current = now.time()
    outside_rth = (_PREMARKET_START <= current < _RTH_START) or (
        _RTH_END <= current < _AFTER_END
    )
    include_overnight = current >= _AFTER_END or current < _OVERNIGHT_END
    return outside_rth, include_overnight
# endregion


# region Client
class IBKRClient:
    async def _connect_ib(self, ib: IB, *, client_id: int) -> None:
        if hasattr(ib, "connectAsync"):
            await ib.connectAsync(
                self._config.host,
                self._config.port,
                clientId=int(client_id),
                timeout=5,
            )
            return
        await asyncio.to_thread(
            ib.connect,
            self._config.host,
            self._config.port,
            int(client_id),
            5,
        )

    @staticmethod
    def _safe_disconnect(ib: IB) -> None:
        if not ib.isConnected():
            return
        try:
            ib.disconnect()
        except OSError:
            # Avoid noisy shutdown if the socket is already closed.
            pass

    @staticmethod
    def _is_retryable_connect_error(exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionRefusedError, TimeoutError, asyncio.TimeoutError)):
            return True
        if isinstance(exc, OSError):
            err_no = int(getattr(exc, "errno", 0) or 0)
            if err_no in (61, 111, 10061):
                return True
        msg = str(exc)
        return "Connect call failed" in msg or "API connection failed" in msg

    def _request_reconnect(self) -> None:
        if self._shutdown:
            return
        if not self._reconnect_in_progress() or self._reconnect_fast_deadline is None:
            self._reconnect_fast_deadline = (
                time.monotonic() + float(self._config.reconnect_timeout_sec)
            )
        self._reconnect_requested = True
        self._start_reconnect_loop()

    def _reconnect_in_progress(self) -> bool:
        return bool(
            self._reconnect_requested
            and self._reconnect_task
            and not self._reconnect_task.done()
        )

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib = IB()
        self._ib_proxy = IB()
        self._shutdown = False
        self._connect_lock = asyncio.Lock()
        self._connect_proxy_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._proxy_lock = asyncio.Lock()
        self._account_updates_started = False
        self._index_contracts: dict[str, Contract] = {}
        self._index_tickers: dict[str, Ticker] = {}
        self._index_task: asyncio.Task | None = None
        self._index_error: str | None = None
        self._proxy_contracts: dict[str, Contract] = {}
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_task: asyncio.Task | None = None
        self._proxy_error: str | None = None
        self._proxy_force_delayed = False
        self._proxy_probe_task: asyncio.Task | None = None
        self._proxy_contract_force_delayed: set[int] = set()
        self._proxy_contract_probe_tasks: dict[int, asyncio.Task] = {}
        self._proxy_contract_delayed_tasks: dict[int, asyncio.Task] = {}
        self._detail_tickers: dict[int, tuple[IB, Ticker]] = {}
        self._ticker_owners: dict[int, set[str]] = {}
        self._historical_bar_cache: dict[
            tuple[str, int, str, str, bool, str, str],
            tuple[list[tuple[datetime, float]], float] | tuple[list[tuple[datetime, float]], float, float],
        ] = {}
        self._historical_bar_ohlcv_cache: dict[
            tuple[str, int, str, str, bool, str, str],
            tuple[list[OhlcvBar], float] | tuple[list[OhlcvBar], float, float],
        ] = {}
        self._front_future_cache: dict[tuple[str, str], tuple[Contract, float]] = {}
        self._update_callback: Callable[[], None] | None = None
        self._stream_listeners: set[Callable[[], None]] = set()
        self._pnl: PnL | None = None
        self._pnl_account: str | None = None
        self._account_value_cache: dict[tuple[str, str], tuple[float, datetime]] = {}
        self._session_close_cache: dict[int, tuple[float | None, float | None, float]] = {}
        self._order_error_cache: dict[int, tuple[float, int, str]] = {}
        self._fx_rate_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._farm_connectivity_lost = False
        self._reconnect_requested = False
        self._resubscribe_main_needed = False
        self._resubscribe_proxy_needed = False
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_fast_deadline: float | None = None
        self._ib.errorEvent += self._on_error_main
        self._ib.disconnectedEvent += self._on_disconnected_main
        self._ib.updatePortfolioEvent += self._on_stream_update
        self._ib.pendingTickersEvent += self._on_stream_update
        self._ib.pnlEvent += self._on_stream_update
        self._ib.accountValueEvent += self._on_account_value
        self._ib_proxy.errorEvent += self._on_error_proxy
        self._ib_proxy.disconnectedEvent += self._on_disconnected_proxy
        self._ib_proxy.pendingTickersEvent += self._on_stream_update

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    def reconnect_phase(self) -> str | None:
        if not self._reconnect_in_progress():
            return None
        deadline = self._reconnect_fast_deadline
        if deadline is None or time.monotonic() < deadline:
            return "fast"
        return "slow"

    def connection_state(self) -> str:
        phase = self.reconnect_phase()
        if phase == "fast":
            return "reconnecting-fast"
        if phase == "slow":
            return "reconnecting-slow"
        main_connected = self._ib.isConnected()
        proxy_connected = self._ib_proxy.isConnected()
        if main_connected and proxy_connected:
            return "connected"
        if main_connected or proxy_connected:
            return "degraded"
        return "disconnected"

    async def connect(self) -> None:
        self._shutdown = False
        if self._ib.isConnected():
            return
        if self._reconnect_in_progress() and asyncio.current_task() is not self._reconnect_task:
            raise ConnectionError("IBKR reconnect in progress")
        async with self._connect_lock:
            if self._ib.isConnected():
                return
            try:
                await self._connect_ib(self._ib, client_id=int(self._config.client_id))
            except Exception as exc:
                if self._is_retryable_connect_error(exc):
                    self._request_reconnect()
                raise

    async def connect_proxy(self) -> None:
        self._shutdown = False
        if self._ib_proxy.isConnected():
            return
        if self._reconnect_in_progress() and asyncio.current_task() is not self._reconnect_task:
            raise ConnectionError("IBKR reconnect in progress")
        async with self._connect_proxy_lock:
            if self._ib_proxy.isConnected():
                return
            try:
                await self._connect_ib(self._ib_proxy, client_id=int(self._config.proxy_client_id))
                self._proxy_error = None
            except Exception as exc:
                self._proxy_error = str(exc)
                if self._is_retryable_connect_error(exc):
                    self._request_reconnect()
                raise

    async def disconnect(self) -> None:
        self._shutdown = True
        self._stop_reconnect_loop()
        self._reconnect_requested = False
        self._reconnect_fast_deadline = None
        self._safe_disconnect(self._ib)
        self._safe_disconnect(self._ib_proxy)
        self._account_updates_started = False
        self._index_tickers = {}
        self._index_task = None
        self._index_error = None
        self._proxy_tickers = {}
        self._proxy_task = None
        self._proxy_error = None
        self._proxy_force_delayed = False
        self._proxy_probe_task = None
        self._proxy_contract_force_delayed = set()
        for task in self._proxy_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_delayed_tasks.values():
            if task and not task.done():
                task.cancel()
        self._proxy_contract_probe_tasks = {}
        self._proxy_contract_delayed_tasks = {}
        self._detail_tickers = {}
        self._ticker_owners = {}
        self._pnl = None
        self._pnl_account = None
        self._account_value_cache = {}
        self._session_close_cache = {}
        self._fx_rate_cache = {}
        self._resubscribe_main_needed = False
        self._resubscribe_proxy_needed = False

    async def fetch_portfolio(self) -> list[PortfolioItem]:
        """Fetch a snapshot of portfolio items (filtered by account if provided)."""
        async with self._lock:
            await self._ensure_account_updates()
            account = self._config.account or ""
            return list(self._ib.portfolio(account))

    async def fetch_index_tickers(self) -> dict[str, Ticker]:
        async with self._lock:
            await self._ensure_index_tickers()
            return dict(self._index_tickers)

    def start_index_tickers(self) -> None:
        if self._index_task and not self._index_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._index_task = loop.create_task(self._load_index_tickers())

    def index_tickers(self) -> dict[str, Ticker]:
        return dict(self._index_tickers)

    def index_error(self) -> str | None:
        return self._index_error

    def start_proxy_tickers(self) -> None:
        if self._proxy_task and not self._proxy_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_task = loop.create_task(self._load_proxy_tickers())

    def proxy_tickers(self) -> dict[str, Ticker]:
        return dict(self._proxy_tickers)

    def proxy_error(self) -> str | None:
        return self._proxy_error

    def set_update_callback(self, callback: Callable[[], None]) -> None:
        self._update_callback = callback

    def add_stream_listener(self, callback: Callable[[], None]) -> None:
        self._stream_listeners.add(callback)

    def remove_stream_listener(self, callback: Callable[[], None]) -> None:
        self._stream_listeners.discard(callback)

    def pnl(self) -> PnL | None:
        return self._pnl

    def portfolio_item(self, con_id: int) -> PortfolioItem | None:
        if not con_id or not self._ib.isConnected():
            return None
        account = self._config.account or ""
        for item in self._ib.portfolio(account):
            try:
                if int(item.contract.conId or 0) == con_id:
                    return item
            except (TypeError, ValueError):
                continue
        return None

    def ticker_for_con_id(self, con_id: int) -> Ticker | None:
        if not con_id:
            return None
        entry = self._detail_tickers.get(int(con_id))
        if not entry:
            return None
        _ib, ticker = entry
        return ticker

    async def ensure_ticker(self, contract: Contract, *, owner: str = "default") -> Ticker:
        con_id = int(contract.conId or 0)
        use_proxy = contract.secType in ("STK", "OPT")
        contract_force_delayed = bool(
            use_proxy and con_id and con_id in self._proxy_contract_force_delayed
        )
        if use_proxy:
            await self.connect_proxy()
            md_type = 3 if self._proxy_force_delayed or contract_force_delayed else 1
            self._ib_proxy.reqMarketDataType(md_type)
            ib = self._ib_proxy
        else:
            await self.connect()
            self._ib.reqMarketDataType(3)
            ib = self._ib
        req_contract = contract
        if contract.secType == "STK":
            _, include_overnight = _session_flags(datetime.now(tz=_ET_ZONE))
            if include_overnight:
                req_contract = copy.copy(contract)
                req_contract.exchange = "OVERNIGHT"
            else:
                primary_exchange = getattr(contract, "primaryExchange", "") or ""
                if (not contract.exchange or contract.exchange == "SMART") and primary_exchange:
                    req_contract = copy.copy(contract)
                    req_contract.exchange = primary_exchange
                elif not contract.exchange:
                    req_contract = copy.copy(contract)
                    req_contract.exchange = "SMART"
        elif contract.secType in ("OPT", "FOP"):
            if not contract.exchange:
                req_contract = copy.copy(contract)
                if contract.secType == "FOP":
                    primary_exchange = getattr(contract, "primaryExchange", "") or ""
                    req_contract.exchange = primary_exchange or "CME"
                else:
                    req_contract.exchange = "SMART"
        cached = self._detail_tickers.get(con_id) if con_id else None
        if cached:
            if con_id:
                self._ticker_owners.setdefault(con_id, set()).add(owner)
            cached_ib, cached_ticker = cached
            desired_exchange = getattr(req_contract, "exchange", "") or ""
            current_exchange = getattr(cached_ticker.contract, "exchange", "") or ""
            if contract.secType == "STK" and desired_exchange and desired_exchange != current_exchange:
                try:
                    cached_ib.cancelMktData(cached_ticker.contract)
                except Exception:
                    pass
                ticker = ib.reqMktData(req_contract)
                self._detail_tickers[con_id] = (ib, ticker)
                if use_proxy and con_id:
                    if con_id in self._proxy_contract_force_delayed:
                        if not self._ticker_has_data(ticker):
                            self._start_proxy_contract_delayed_resubscribe(req_contract)
                    else:
                        self._start_proxy_contract_quote_probe(req_contract)
                return ticker
            if use_proxy and con_id:
                if con_id in self._proxy_contract_force_delayed:
                    if not self._ticker_has_data(cached_ticker):
                        self._start_proxy_contract_delayed_resubscribe(req_contract)
                elif not self._ticker_has_data(cached_ticker):
                    self._start_proxy_contract_quote_probe(req_contract)
            return cached_ticker
        ticker = ib.reqMktData(req_contract)
        if con_id:
            self._detail_tickers[con_id] = (ib, ticker)
            self._ticker_owners.setdefault(con_id, set()).add(owner)
        if use_proxy and con_id:
            if con_id in self._proxy_contract_force_delayed:
                if not self._ticker_has_data(ticker):
                    self._start_proxy_contract_delayed_resubscribe(req_contract)
            else:
                self._start_proxy_contract_quote_probe(req_contract)
        return ticker

    async def place_limit_order(
        self,
        contract: Contract,
        action: str,
        quantity: float,
        limit_price: float,
        outside_rth: bool,
    ) -> Trade:
        await self.connect()
        order_contract = contract
        tif = "GTC"
        outside_session = False
        include_overnight = False
        if contract.secType == "STK":
            outside_session, include_overnight = _session_flags(datetime.now(tz=_ET_ZONE))
            if include_overnight:
                order_contract = copy.copy(contract)
                order_contract.exchange = "OVERNIGHT"
                # IBKR rejects STK OVERNIGHT orders with GTC; DAY is required.
                tif = "DAY"
        order = LimitOrder(action, quantity, limit_price, tif=tif)
        if contract.secType == "STK" and outside_rth and outside_session and not include_overnight:
            order.outsideRth = True
        order_contract = _normalize_order_contract(order_contract)
        return self._ib.placeOrder(order_contract, order)

    async def modify_limit_order(self, trade: Trade, limit_price: float) -> Trade:
        """Modify an existing LIMIT order's price in place."""
        await self.connect()
        order = trade.order
        if not hasattr(order, "lmtPrice"):
            raise ValueError("modify_limit_order: trade has no lmtPrice")
        order.lmtPrice = float(limit_price)
        return self._ib.placeOrder(trade.contract, order)

    def open_trades_for_conids(self, con_ids: Iterable[int]) -> list[Trade]:
        if not self._ib.isConnected():
            return []
        targets = {int(con_id) for con_id in con_ids if con_id}
        if not targets:
            return []
        trades: list[Trade] = []
        for trade in self._ib.openTrades():
            try:
                trade_con_id = int(getattr(trade.contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                continue
            if trade_con_id in targets:
                trades.append(trade)
        return trades

    async def cancel_trade(self, trade: Trade) -> None:
        await self.connect()
        self._ib.cancelOrder(trade.order)

    async def resolve_underlying_contract(self, contract: Contract) -> Contract | None:
        if contract.secType == "OPT":
            candidate = Stock(
                symbol=contract.symbol,
                exchange="SMART",
                currency=contract.currency or "USD",
            )
            qualified = await self._qualify_contract(candidate, use_proxy=True)
            return qualified or candidate
        if contract.secType == "FOP":
            under_con_id = int(getattr(contract, "underConId", 0) or 0)
            if not under_con_id:
                return None
            candidate = Contract(conId=under_con_id)
            qualified = await self._qualify_contract(candidate, use_proxy=False)
            return qualified or candidate
        return None

    @staticmethod
    def _historical_request_contract(contract: Contract, *, sec_type: str) -> Contract:
        req_contract = contract
        if sec_type in ("STK", "OPT") and not getattr(contract, "exchange", ""):
            req_contract = copy.copy(contract)
            req_contract.exchange = "SMART"
        return req_contract

    async def _request_historical_data(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        sec_type = str(getattr(contract, "secType", "") or "")
        use_proxy = sec_type in ("STK", "OPT")
        try:
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib
        except Exception as exc:
            if use_proxy:
                self._proxy_error = str(exc)
            return []
        req_contract = self._historical_request_contract(contract, sec_type=sec_type)
        try:
            return await ib.reqHistoricalDataAsync(
                req_contract,
                endDateTime="",
                durationStr=str(duration_str),
                barSizeSetting=str(bar_size),
                whatToShow=str(what_to_show),
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception:
            return []

    @staticmethod
    def _is_intraday_bar_size(bar_size: str) -> bool:
        label = str(bar_size or "").strip().lower()
        if not label:
            return True
        return not any(token in label for token in ("day", "week", "month"))

    @staticmethod
    def _bar_time_et(ts: datetime) -> dtime:
        if getattr(ts, "tzinfo", None) is None:
            return ts.time()
        return ts.astimezone(_ET_ZONE).timetz().replace(tzinfo=None)

    @classmethod
    def _is_overnight_bar(cls, ts: datetime) -> bool:
        current = cls._bar_time_et(ts)
        return current >= _AFTER_END or current < _PREMARKET_START

    @classmethod
    def _merge_full24_raw_bars(cls, *, smart: list, overnight: list) -> list:
        by_ts: dict[datetime, object] = {}
        for bar in smart or []:
            dt = cls._ib_bar_datetime(getattr(bar, "date", None))
            if dt is None:
                continue
            by_ts[dt] = bar
        for bar in overnight or []:
            dt = cls._ib_bar_datetime(getattr(bar, "date", None))
            if dt is None:
                continue
            if dt not in by_ts or cls._is_overnight_bar(dt):
                by_ts[dt] = bar
        return [by_ts[ts] for ts in sorted(by_ts.keys())]

    async def _request_historical_data_for_stream(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        sec_type = str(getattr(contract, "secType", "") or "")
        # For stocks, IBKR SMART full-session intraday misses OVERNIGHT. Stitch SMART+OVERNIGHT.
        if bool(use_rth) or sec_type != "STK" or not self._is_intraday_bar_size(str(bar_size)):
            return await self._request_historical_data(
                contract,
                duration_str=duration_str,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

        smart_contract = copy.copy(contract)
        smart_exchange = str(getattr(smart_contract, "exchange", "") or "").strip().upper()
        if not smart_exchange or smart_exchange == "OVERNIGHT":
            smart_contract.exchange = "SMART"

        smart = await self._request_historical_data(
            smart_contract,
            duration_str=duration_str,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=False,
        )
        overnight_contract = copy.copy(contract)
        overnight_contract.exchange = "OVERNIGHT"
        overnight = await self._request_historical_data(
            overnight_contract,
            duration_str=duration_str,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=False,
        )
        return self._merge_full24_raw_bars(smart=smart, overnight=overnight)

    @staticmethod
    def _ib_bar_datetime(value) -> datetime | None:
        dt = value
        if isinstance(dt, str):
            dt = util.parseIBDatetime(dt)
        if dt is None:
            return None
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, dtime(0, 0))
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    async def session_closes(
        self,
        contract: Contract,
        *,
        cache_ttl_sec: float = 900.0,
    ) -> tuple[float | None, float | None]:
        """Return (prev_close, close_3_sessions_ago) for the given contract.

        Uses 1-day bars with useRTH=True and a small in-memory TTL cache to
        avoid pacing issues.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id:
            return None, None
        cached = self._session_close_cache.get(con_id)
        if cached:
            prev_close, close_3ago, cached_at = cached
            if time.monotonic() - cached_at < cache_ttl_sec:
                return prev_close, close_3ago
        use_proxy = contract.secType in ("STK", "OPT")
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            cached = self._session_close_cache.get(con_id)
            if cached:
                prev_close, close_3ago, cached_at = cached
                if time.monotonic() - cached_at < cache_ttl_sec:
                    return prev_close, close_3ago
            bars = await self._request_historical_data(
                contract,
                duration_str="2 W",
                bar_size="1 day",
                what_to_show="TRADES",
                use_rth=True,
            )
            closes: list[float] = []
            for bar in bars or []:
                try:
                    value = float(getattr(bar, "close", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    closes.append(value)
            prev_close = closes[-1] if closes else None
            close_3ago = closes[-4] if len(closes) >= 4 else None
            self._session_close_cache[con_id] = (
                prev_close,
                close_3ago,
                time.monotonic(),
            )
            return prev_close, close_3ago

    async def historical_bars(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str = "TRADES",
        cache_ttl_sec: float = 30.0,
    ) -> list[tuple[datetime, float]]:
        """Return [(bar_ts, close), ...] for the given contract.

        Uses a small in-memory TTL cache to avoid pacing issues when the bot is running.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        sec_type = str(getattr(contract, "secType", "") or "")
        symbol = str(getattr(contract, "symbol", "") or "")
        key = (
            symbol,
            con_id,
            sec_type,
            str(bar_size),
            bool(use_rth),
            str(what_to_show),
            str(duration_str),
        )
        requested_ttl = max(0.0, float(cache_ttl_sec))
        empty_ttl_sec = 1.0

        def _cached_bars(
            cached_entry: tuple[list[tuple[datetime, float]], float] | tuple[list[tuple[datetime, float]], float, float] | None,
        ) -> list[tuple[datetime, float]] | None:
            if cached_entry is None:
                return None
            if len(cached_entry) >= 3:
                bars, cached_at, cached_ttl = cached_entry[0], cached_entry[1], cached_entry[2]
            else:
                bars, cached_at = cached_entry
                cached_ttl = requested_ttl
            ttl = max(0.0, float(cached_ttl))
            if time.monotonic() - float(cached_at) < ttl:
                return list(bars)
            return None

        cached_bars = _cached_bars(self._historical_bar_cache.get(key))
        if cached_bars is not None:
            return cached_bars

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            cached_bars = _cached_bars(self._historical_bar_cache.get(key))
            if cached_bars is not None:
                return cached_bars
            raw = await self._request_historical_data_for_stream(
                contract,
                duration_str=str(duration_str),
                bar_size=str(bar_size),
                what_to_show=str(what_to_show),
                use_rth=bool(use_rth),
            )

            bars: list[tuple[datetime, float]] = []
            for bar in raw or []:
                dt = self._ib_bar_datetime(getattr(bar, "date", None))
                if dt is None:
                    continue
                try:
                    close = float(getattr(bar, "close", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0:
                    continue
                bars.append((dt, close))

            cache_ttl = min(requested_ttl, empty_ttl_sec) if not bars else requested_ttl
            self._historical_bar_cache[key] = (bars, time.monotonic(), cache_ttl)
            return list(bars)

    async def historical_bars_ohlcv(
        self,
        contract: Contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str = "TRADES",
        cache_ttl_sec: float = 30.0,
    ) -> list[OhlcvBar]:
        """Return OHLCV bars for the given contract.

        Uses a small in-memory TTL cache to avoid pacing issues when the bot is running.
        """
        con_id = int(getattr(contract, "conId", 0) or 0)
        sec_type = str(getattr(contract, "secType", "") or "")
        symbol = str(getattr(contract, "symbol", "") or "")
        key = (
            symbol,
            con_id,
            sec_type,
            str(bar_size),
            bool(use_rth),
            str(what_to_show),
            str(duration_str),
        )
        requested_ttl = max(0.0, float(cache_ttl_sec))
        empty_ttl_sec = 1.0

        def _cached_bars(
            cached_entry: tuple[list[OhlcvBar], float] | tuple[list[OhlcvBar], float, float] | None,
        ) -> list[OhlcvBar] | None:
            if cached_entry is None:
                return None
            if len(cached_entry) >= 3:
                bars, cached_at, cached_ttl = cached_entry[0], cached_entry[1], cached_entry[2]
            else:
                bars, cached_at = cached_entry
                cached_ttl = requested_ttl
            ttl = max(0.0, float(cached_ttl))
            if time.monotonic() - float(cached_at) < ttl:
                return list(bars)
            return None

        cached_bars = _cached_bars(self._historical_bar_ohlcv_cache.get(key))
        if cached_bars is not None:
            return cached_bars

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            cached_bars = _cached_bars(self._historical_bar_ohlcv_cache.get(key))
            if cached_bars is not None:
                return cached_bars
            raw = await self._request_historical_data_for_stream(
                contract,
                duration_str=str(duration_str),
                bar_size=str(bar_size),
                what_to_show=str(what_to_show),
                use_rth=bool(use_rth),
            )

            bars: list[OhlcvBar] = []
            for bar in raw or []:
                dt = self._ib_bar_datetime(getattr(bar, "date", None))
                if dt is None:
                    continue
                try:
                    open_p = float(getattr(bar, "open", 0.0) or 0.0)
                    high = float(getattr(bar, "high", 0.0) or 0.0)
                    low = float(getattr(bar, "low", 0.0) or 0.0)
                    close = float(getattr(bar, "close", 0.0) or 0.0)
                    volume = float(getattr(bar, "volume", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0:
                    continue
                bars.append(
                    OhlcvBar(
                        ts=dt,
                        open=open_p,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                )

            cache_ttl = min(requested_ttl, empty_ttl_sec) if not bars else requested_ttl
            self._historical_bar_ohlcv_cache[key] = (bars, time.monotonic(), cache_ttl)
            return list(bars)

    async def stock_option_chain(self, symbol: str):
        """Return (qualified_underlying, chain) for an equity option underlyer."""
        candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
        underlying = await self._qualify_contract(candidate, use_proxy=True) or candidate
        try:
            await self.connect_proxy()
        except Exception:
            return None
        chains = self._ib_proxy.reqSecDefOptParams(
            underlying.symbol,
            "",
            underlying.secType,
            int(getattr(underlying, "conId", 0) or 0),
        )
        if not chains:
            return None
        chain = next((c for c in chains if getattr(c, "exchange", None) == "SMART"), chains[0])
        return underlying, chain

    async def qualify_proxy_contracts(self, *contracts: Contract) -> list[Contract]:
        try:
            await self.connect_proxy()
        except Exception:
            return []
        try:
            result = await self._ib_proxy.qualifyContractsAsync(*contracts)
        except Exception:
            return []
        return list(result or [])

    async def front_future(
        self,
        symbol: str,
        *,
        exchange: str = "CME",
        cache_ttl_sec: float = 3600.0,
    ) -> Contract | None:
        """Resolve a tradable front-month future contract.

        Uses a TTL cache to avoid repeated contract-details requests in the live UI.
        """
        sym = str(symbol or "").strip().upper()
        ex = str(exchange or "").strip().upper() or "CME"
        key = (sym, ex)
        cached = self._front_future_cache.get(key)
        if cached:
            contract, cached_at = cached
            if time.monotonic() - cached_at < float(cache_ttl_sec):
                return contract

        def _parse_expiry(raw: str | None) -> datetime | None:
            if not raw:
                return None
            cleaned = str(raw).strip()
            if len(cleaned) >= 8 and cleaned[:8].isdigit():
                try:
                    return datetime(int(cleaned[:4]), int(cleaned[4:6]), int(cleaned[6:8]))
                except ValueError:
                    return None
            if len(cleaned) >= 6 and cleaned[:6].isdigit():
                try:
                    return datetime(int(cleaned[:4]), int(cleaned[4:6]), 1)
                except ValueError:
                    return None
            return None

        async with self._lock:
            cached = self._front_future_cache.get(key)
            if cached:
                contract, cached_at = cached
                if time.monotonic() - cached_at < float(cache_ttl_sec):
                    return contract
            try:
                await self.connect()
            except Exception:
                return None
            candidate = Future(symbol=sym, lastTradeDateOrContractMonth="", exchange=ex, currency="USD")
            try:
                details = await self._ib.reqContractDetailsAsync(candidate)
            except Exception:
                details = []
            if not details:
                return None

            today = datetime.now(tz=_ET_ZONE).date()
            best = None
            best_dt = None
            for d in details:
                contract = getattr(d, "contract", None)
                if not contract:
                    continue
                if getattr(contract, "secType", "") != "FUT":
                    continue
                exp_raw = getattr(d, "realExpirationDate", None) or getattr(
                    contract, "lastTradeDateOrContractMonth", None
                )
                exp_dt = _parse_expiry(str(exp_raw) if exp_raw is not None else None)
                if exp_dt is None:
                    continue
                exp_date = exp_dt.date()
                if exp_date < today:
                    continue
                if best_dt is None or exp_date < best_dt:
                    best_dt = exp_date
                    best = contract

            if best is None:
                for d in details:
                    contract = getattr(d, "contract", None)
                    if contract and getattr(contract, "secType", "") == "FUT":
                        best = contract
                        break
            if best is None:
                return None

            try:
                qualified = await self._ib.qualifyContractsAsync(best)
            except Exception:
                qualified = []
            resolved = qualified[0] if qualified else best
            self._front_future_cache[key] = (resolved, time.monotonic())
            return resolved

    def release_ticker(self, con_id: int, *, owner: str = "default") -> None:
        if not con_id:
            return
        owners = self._ticker_owners.get(con_id)
        if owners is not None:
            owners.discard(owner)
            if owners:
                return
            self._ticker_owners.pop(con_id, None)

        entry = self._detail_tickers.pop(con_id, None)
        if entry:
            ib, ticker = entry
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass
        probe_task = self._proxy_contract_probe_tasks.pop(con_id, None)
        if probe_task and not probe_task.done():
            probe_task.cancel()
        delayed_task = self._proxy_contract_delayed_tasks.pop(con_id, None)
        if delayed_task and not delayed_task.done():
            delayed_task.cancel()

    def account_value(
        self,
        tag: str,
        *,
        currency: str | None = None,
    ) -> tuple[float | None, str | None, datetime | None]:
        desired_currency = str(currency or "").strip().upper() or None
        account = self._config.account or ""
        if desired_currency:
            cached_exact = self._account_value_cache.get((tag, desired_currency))
            if cached_exact:
                value, updated_at = cached_exact
                return value, desired_currency, updated_at
            cached = None
        else:
            cached = _pick_cached_value(self._account_value_cache, tag)
        if cached:
            value, currency, updated_at = cached
            return value, currency, updated_at
        values = [v for v in self._ib.accountValues(account) if v.tag == tag]
        if not values:
            return None, None, None
        if desired_currency:
            chosen = next(
                (
                    value
                    for value in values
                    if str(getattr(value, "currency", "") or "").strip().upper() == desired_currency
                ),
                None,
            )
        else:
            chosen = _pick_account_value(values)
        if not chosen:
            return None, None, None
        try:
            return float(chosen.value), chosen.currency, None
        except (TypeError, ValueError):
            return None, chosen.currency, None

    def account_exchange_rate(self, currency: str) -> float | None:
        target = str(currency or "").strip().upper()
        if not target:
            return None
        rate, _currency, _updated = self.account_value("ExchangeRate", currency=target)
        try:
            parsed = float(rate) if rate is not None else None
        except (TypeError, ValueError):
            parsed = None
        if parsed is None or parsed <= 0:
            return None
        return float(parsed)

    async def fx_rate(
        self,
        from_currency: str,
        to_currency: str,
        *,
        max_age_sec: float = 15.0,
    ) -> float | None:
        src = str(from_currency or "").strip().upper()
        dst = str(to_currency or "").strip().upper()
        if not src or not dst:
            return None
        if src == dst:
            return 1.0

        key = (src, dst)
        now = time.monotonic()
        cached = self._fx_rate_cache.get(key)
        if cached and (now - float(cached[1])) <= float(max_age_sec):
            return float(cached[0])

        await self.connect()
        contract = Forex(f"{src}{dst}")
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
        except Exception:
            qualified = []
        if qualified:
            contract = qualified[0]

        ticker = await self.ensure_ticker(contract, owner="fx")

        def _as_pos_float(value: object) -> float | None:
            try:
                parsed = float(value) if value is not None else None
            except (TypeError, ValueError):
                parsed = None
            if parsed is None or parsed <= 0:
                return None
            return parsed

        bid = _as_pos_float(getattr(ticker, "bid", None))
        ask = _as_pos_float(getattr(ticker, "ask", None))
        last = _as_pos_float(getattr(ticker, "last", None))
        close = _as_pos_float(getattr(ticker, "close", None))
        market_price = None
        market_price_fn = getattr(ticker, "marketPrice", None)
        if callable(market_price_fn):
            try:
                market_price = _as_pos_float(market_price_fn())
            except Exception:
                market_price = None

        rate = ((bid + ask) / 2.0) if bid is not None and ask is not None else (last or market_price or close)
        if rate is not None and rate > 0:
            self._fx_rate_cache[key] = (float(rate), now)

        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            self.release_ticker(con_id, owner="fx")
        return float(rate) if rate is not None and rate > 0 else None

    async def convert_currency_value(
        self,
        value: float,
        *,
        from_currency: str,
        to_currency: str,
    ) -> tuple[float | None, float | None]:
        src = str(from_currency or "").strip().upper()
        dst = str(to_currency or "").strip().upper()
        if not src or not dst:
            return None, None
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return None, None
        if src == dst:
            return amount, 1.0

        from_rate = self.account_exchange_rate(src)
        to_rate = self.account_exchange_rate(dst)
        if from_rate is not None and to_rate is not None and from_rate > 0 and to_rate > 0:
            rate = float(from_rate) / float(to_rate)
            return amount * float(rate), float(rate)

        direct = await self.fx_rate(src, dst)
        if direct is not None and direct > 0:
            return amount * float(direct), float(direct)

        inverse = await self.fx_rate(dst, src)
        if inverse is not None and inverse > 0:
            rate = 1.0 / float(inverse)
            return amount * float(rate), float(rate)

        return None, None

    async def hard_refresh(self) -> None:
        async with self._lock:
            await self.connect()
            account = self._config.account or ""
            try:
                self._ib.client.reqAccountUpdates(False, account)
            except Exception:
                pass
            if self._pnl_account:
                try:
                    self._ib.cancelPnL(self._pnl_account)
                except Exception:
                    pass
            self._pnl = None
            self._pnl_account = None
            self._account_updates_started = False
            self._session_close_cache = {}
            await self._ensure_account_updates()
            for ticker in self._index_tickers.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._index_tickers = {}
            self._index_task = None
            await self._ensure_index_tickers()
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception as exc:
                self._proxy_error = str(exc)
                self._proxy_tickers = {}
                self._proxy_task = None
                return
            self._proxy_contract_force_delayed = set()
            for task in self._proxy_contract_probe_tasks.values():
                if task and not task.done():
                    task.cancel()
            for task in self._proxy_contract_delayed_tasks.values():
                if task and not task.done():
                    task.cancel()
            self._proxy_contract_probe_tasks = {}
            self._proxy_contract_delayed_tasks = {}
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._proxy_tickers = {}
            self._proxy_task = None
            await self._ensure_proxy_tickers()

    async def _ensure_account_updates(self) -> None:
        if self._account_updates_started:
            return
        await self.connect()
        account = self._config.account or ""
        # Start streaming account/portfolio updates without blocking UI startup.
        self._ib.client.reqAccountUpdates(True, account)
        self._account_updates_started = True
        self._ensure_pnl(account)

    def _ensure_pnl(self, account: str) -> None:
        if self._pnl:
            return
        if not account:
            accounts = self._ib.managedAccounts()
            if accounts:
                account = accounts[0]
        if not account:
            return
        try:
            self._pnl = self._ib.reqPnL(account)
            self._pnl_account = account
        except Exception:
            self._pnl = None
            self._pnl_account = None

    async def _ensure_index_tickers(self) -> None:
        await self.connect()
        # Allow delayed market data fallback when real-time is unavailable.
        self._ib.reqMarketDataType(3)
        if not self._index_contracts:
            self._index_contracts = await self._qualify_index_contracts()
        if not self._index_tickers:
            for symbol, contract in self._index_contracts.items():
                self._index_tickers[symbol] = self._ib.reqMktData(contract)

    async def _ensure_proxy_tickers(self) -> None:
        await self.connect_proxy()
        md_type = 3 if self._proxy_force_delayed else 1
        self._ib_proxy.reqMarketDataType(md_type)
        if not self._proxy_contracts:
            self._proxy_contracts = await self._qualify_proxy_contracts()
        if not self._proxy_tickers:
            _, include_overnight = _session_flags(datetime.now(tz=_ET_ZONE))
            for symbol, contract in self._proxy_contracts.items():
                req_contract = contract
                if include_overnight and contract.secType == "STK":
                    if contract.exchange != "OVERNIGHT":
                        req_contract = copy.copy(contract)
                        req_contract.exchange = "OVERNIGHT"
                self._proxy_tickers[symbol] = self._ib_proxy.reqMktData(req_contract)

    async def _qualify_proxy_contracts(self) -> dict[str, Contract]:
        qualified: dict[str, Contract] = {}
        for symbol in _PROXY_SYMBOLS:
            candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
            try:
                result = await self._ib_proxy.qualifyContractsAsync(candidate)
            except Exception:
                continue
            if result:
                contract = result[0]
                primary_exchange = getattr(contract, "primaryExchange", "") or ""
                if contract.exchange == "SMART" and primary_exchange:
                    contract.exchange = primary_exchange
                qualified[symbol] = contract
        return qualified

    async def _qualify_contract(self, contract: Contract, use_proxy: bool) -> Contract | None:
        try:
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib
        except Exception:
            return None
        try:
            result = await ib.qualifyContractsAsync(contract)
        except Exception:
            return None
        return result[0] if result else None

    async def _qualify_index_contracts(self) -> dict[str, Contract]:
        qualified: dict[str, Contract] = {}
        for symbol, exchanges in _INDEX_CONTRACTS.items():
            contract = await self._qualify_cont_future(symbol, exchanges)
            if contract:
                qualified[symbol] = contract
        return qualified

    async def _qualify_cont_future(
        self, symbol: str, exchanges: Iterable[str]
    ) -> Contract | None:
        for exchange in exchanges:
            candidate = ContFuture(symbol=symbol, exchange=exchange, currency="USD")
            try:
                result = await self._ib.qualifyContractsAsync(candidate)
            except Exception:
                continue
            if result:
                return result[0]
        return None

    async def _load_index_tickers(self) -> None:
        try:
            async with self._lock:
                await self._ensure_index_tickers()
            self._index_error = None
        except Exception as exc:
            self._index_error = str(exc)

    async def _load_proxy_tickers(self) -> None:
        try:
            async with self._proxy_lock:
                await self._ensure_proxy_tickers()
            self._proxy_error = None
            self._start_proxy_probe()
        except Exception as exc:
            self._proxy_error = str(exc)

    def _on_error_main(self, reqId, errorCode, errorString, contract) -> None:
        self._remember_order_error(reqId, errorCode, errorString)
        self._handle_conn_error(errorCode)

    def _on_error_proxy(self, reqId, errorCode, errorString, contract) -> None:
        self._remember_order_error(reqId, errorCode, errorString)
        if errorCode == 10167 and not self._proxy_force_delayed:
            self._proxy_force_delayed = True
            self._start_proxy_resubscribe()
        if errorCode in (354, 10089, 10090, 10091, 10168) and contract:
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._proxy_contract_force_delayed.add(con_id)
                self._start_proxy_contract_delayed_resubscribe(contract)
        self._handle_conn_error(errorCode)

    def _remember_order_error(self, req_id, error_code, error_text) -> None:
        try:
            order_id = int(req_id or 0)
        except (TypeError, ValueError):
            return
        if order_id <= 0:
            return
        try:
            code = int(error_code or 0)
        except (TypeError, ValueError):
            code = 0
        message = str(error_text or "").strip()
        if not message:
            return
        # Keep order-related errors only; reqId is shared across other request types.
        if code not in (201, 202, 10147, 10148, 10149) and "order" not in message.lower():
            return
        self._order_error_cache[order_id] = (time.monotonic(), code, message)
        if len(self._order_error_cache) > 512:
            stale = sorted(
                self._order_error_cache.items(),
                key=lambda item: item[1][0],
            )[:64]
            for key, _value in stale:
                self._order_error_cache.pop(int(key), None)

    def pop_order_error(self, order_id: int, *, max_age_sec: float = 120.0) -> dict | None:
        try:
            key = int(order_id or 0)
        except (TypeError, ValueError):
            key = 0
        if key <= 0:
            return None
        payload = self._order_error_cache.pop(key, None)
        if payload is None:
            return None
        ts_mono, code, message = payload
        if (time.monotonic() - float(ts_mono)) > float(max_age_sec):
            return None
        return {"code": int(code), "message": str(message)}

    def _on_disconnected_main(self) -> None:
        if self._shutdown:
            return
        self._resubscribe_main_needed = True
        self._account_updates_started = False
        self._pnl = None
        self._pnl_account = None
        self._account_value_cache = {}
        self._fx_rate_cache = {}
        self._index_tickers = {}
        self._index_task = None
        self._reconnect_requested = True
        self._start_reconnect_loop()
        if self._update_callback:
            self._update_callback()

    def _on_disconnected_proxy(self) -> None:
        if self._shutdown:
            return
        self._resubscribe_proxy_needed = True
        self._proxy_task = None
        self._proxy_tickers = {}
        self._proxy_probe_task = None
        for task in self._proxy_contract_probe_tasks.values():
            if task and not task.done():
                task.cancel()
        for task in self._proxy_contract_delayed_tasks.values():
            if task and not task.done():
                task.cancel()
        self._proxy_contract_probe_tasks = {}
        self._proxy_contract_delayed_tasks = {}
        self._reconnect_requested = True
        self._start_reconnect_loop()
        if self._update_callback:
            self._update_callback()

    def _start_proxy_contract_delayed_resubscribe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        probe_task = self._proxy_contract_probe_tasks.pop(con_id, None)
        if probe_task and not probe_task.done():
            probe_task.cancel()
        existing = self._proxy_contract_delayed_tasks.get(con_id) if con_id else None
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._resubscribe_proxy_contract_delayed(contract))
        if con_id:
            self._proxy_contract_delayed_tasks[con_id] = task

            def _cleanup(done_task: asyncio.Task, key: int = con_id) -> None:
                current = self._proxy_contract_delayed_tasks.get(key)
                if current is done_task:
                    self._proxy_contract_delayed_tasks.pop(key, None)

            task.add_done_callback(_cleanup)

    @staticmethod
    def _ticker_has_data(ticker: Ticker | None) -> bool:
        if ticker is None:
            return False
        for attr in ("last", "close", "prevLast", "bid", "ask"):
            value = getattr(ticker, attr, None)
            if value is None:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(num) and num != 0:
                return True
        return False

    def _start_proxy_contract_quote_probe(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if (
            not con_id
            or self._proxy_force_delayed
            or con_id in self._proxy_contract_force_delayed
        ):
            return
        existing = self._proxy_contract_probe_tasks.get(con_id)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._probe_proxy_contract_quote(contract))
        self._proxy_contract_probe_tasks[con_id] = task

    async def _probe_proxy_contract_quote(self, contract: Contract) -> None:
        con_id = int(getattr(contract, "conId", 0) or 0)
        try:
            await asyncio.sleep(1.5)
            if (
                not con_id
                or self._proxy_force_delayed
                or con_id in self._proxy_contract_force_delayed
            ):
                return
            entry = self._detail_tickers.get(con_id)
            if not entry:
                return
            ib, ticker = entry
            if ib is not self._ib_proxy:
                return
            if self._ticker_has_data(ticker):
                return
            self._proxy_contract_force_delayed.add(con_id)
            self._start_proxy_contract_delayed_resubscribe(contract)
        finally:
            current = self._proxy_contract_probe_tasks.get(con_id)
            if current is asyncio.current_task():
                self._proxy_contract_probe_tasks.pop(con_id, None)

    async def _resubscribe_proxy_contract_delayed(self, contract: Contract) -> None:
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception:
                return
            self._ib_proxy.reqMarketDataType(3)
            req_contract = contract
            if contract.secType == "STK":
                _, include_overnight = _session_flags(datetime.now(tz=_ET_ZONE))
                if include_overnight:
                    req_contract = copy.copy(contract)
                    req_contract.exchange = "OVERNIGHT"
                else:
                    primary_exchange = getattr(contract, "primaryExchange", "") or ""
                    if (not contract.exchange or contract.exchange == "SMART") and primary_exchange:
                        req_contract = copy.copy(contract)
                        req_contract.exchange = primary_exchange
                    elif not contract.exchange:
                        req_contract = copy.copy(contract)
                        req_contract.exchange = "SMART"
            elif contract.secType in ("OPT", "FOP"):
                if not contract.exchange:
                    req_contract = copy.copy(contract)
                    if contract.secType == "FOP":
                        primary_exchange = getattr(contract, "primaryExchange", "") or ""
                        req_contract.exchange = primary_exchange or "CME"
                    else:
                        req_contract.exchange = "SMART"
            try:
                self._ib_proxy.cancelMktData(contract)
            except Exception:
                pass
            ticker = self._ib_proxy.reqMktData(req_contract)
            con_id = int(getattr(req_contract, "conId", 0) or 0)
            if con_id and con_id in self._detail_tickers:
                self._detail_tickers[con_id] = (self._ib_proxy, ticker)
            symbol = getattr(req_contract, "symbol", "") or ""
            if symbol and symbol in self._proxy_tickers:
                self._proxy_tickers[symbol] = ticker

    def _handle_conn_error(self, error_code: int) -> None:
        if error_code == 1100:
            self._farm_connectivity_lost = True
        elif error_code in (1101, 1102):
            self._farm_connectivity_lost = False

    def _start_proxy_resubscribe(self) -> None:
        if self._proxy_task and not self._proxy_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_task = loop.create_task(self._reload_proxy_tickers())

    def _start_proxy_probe(self) -> None:
        if self._proxy_force_delayed:
            return
        if self._proxy_probe_task and not self._proxy_probe_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._proxy_probe_task = loop.create_task(self._probe_proxy_quotes())

    async def _probe_proxy_quotes(self) -> None:
        await asyncio.sleep(2)
        if self._proxy_force_delayed or not self._proxy_tickers:
            return
        if self._proxy_has_data():
            return
        self._proxy_force_delayed = True
        self._start_proxy_resubscribe()

    def _proxy_has_data(self) -> bool:
        for ticker in self._proxy_tickers.values():
            if self._ticker_has_data(ticker):
                return True
        return False

    async def _reload_proxy_tickers(self) -> None:
        async with self._proxy_lock:
            try:
                await self.connect_proxy()
            except Exception as exc:
                self._proxy_error = str(exc)
                self._proxy_tickers = {}
                self._proxy_task = None
                return
            md_type = 3 if self._proxy_force_delayed else 1
            self._ib_proxy.reqMarketDataType(md_type)
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._proxy_tickers = {}
            for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                if ib is not self._ib_proxy:
                    continue
                req_contract = ticker.contract
                if req_contract.secType in ("OPT", "FOP") and not req_contract.exchange:
                    req_contract = copy.copy(req_contract)
                    if req_contract.secType == "FOP":
                        primary_exchange = getattr(req_contract, "primaryExchange", "") or ""
                        req_contract.exchange = primary_exchange or "CME"
                    else:
                        req_contract.exchange = "SMART"
                try:
                    self._ib_proxy.cancelMktData(ticker.contract)
                except Exception:
                    pass
                self._detail_tickers[con_id] = (
                    self._ib_proxy,
                    self._ib_proxy.reqMktData(req_contract),
                )
            await self._ensure_proxy_tickers()

    def _on_stream_update(self, *_, **__) -> None:
        if self._update_callback:
            self._update_callback()
        if not self._stream_listeners:
            return
        for callback in tuple(self._stream_listeners):
            try:
                callback()
            except Exception:
                continue

    def _on_account_value(self, value: AccountValue) -> None:
        if self._config.account and value.account != self._config.account:
            return
        try:
            parsed = float(value.value)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            key = (value.tag, value.currency)
            self._account_value_cache[key] = (parsed, datetime.now(timezone.utc))
        if self._update_callback:
            self._update_callback()

    def _start_reconnect_loop(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._reconnect_task = loop.create_task(self._reconnect_until_deadline())

    def _stop_reconnect_loop(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

    async def _reconnect_until_deadline(self) -> None:
        fast_deadline = self._reconnect_fast_deadline
        if fast_deadline is None:
            fast_deadline = time.monotonic() + float(self._config.reconnect_timeout_sec)
            self._reconnect_fast_deadline = fast_deadline
        fast_interval = max(0.5, float(self._config.reconnect_interval_sec))
        slow_interval = max(fast_interval, float(self._config.reconnect_slow_interval_sec))
        slow_notified = False
        while self._reconnect_requested:
            if not slow_notified and time.monotonic() >= fast_deadline:
                slow_notified = True
                if self._update_callback:
                    self._update_callback()
            await self._reconnect_once()
            if not self._reconnect_requested:
                break
            interval = fast_interval if time.monotonic() < fast_deadline else slow_interval
            await asyncio.sleep(interval)

    async def _reconnect_once(self) -> None:
        async with self._lock:
            if not self._ib.isConnected():
                try:
                    await self.connect()
                except Exception:
                    return
                self._resubscribe_main_needed = True
            if self._resubscribe_main_needed and self._ib.isConnected():
                self._account_updates_started = False
                await self._ensure_account_updates()
                for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                    if ib is not self._ib:
                        continue
                    try:
                        self._detail_tickers[con_id] = (
                            self._ib,
                            self._ib.reqMktData(ticker.contract),
                        )
                    except Exception:
                        continue
                self._index_tickers = {}
                self._index_task = None
                await self._ensure_index_tickers()
                self._resubscribe_main_needed = False
        async with self._proxy_lock:
            if not self._ib_proxy.isConnected():
                try:
                    await self.connect_proxy()
                except Exception as exc:
                    self._proxy_error = str(exc)
                    return
                self._resubscribe_proxy_needed = True
            if self._resubscribe_proxy_needed and self._ib_proxy.isConnected():
                md_type = 3 if self._proxy_force_delayed else 1
                self._ib_proxy.reqMarketDataType(md_type)
                for con_id, (ib, ticker) in list(self._detail_tickers.items()):
                    if ib is not self._ib_proxy:
                        continue
                    req_contract = ticker.contract
                    if req_contract.secType in ("OPT", "FOP") and not req_contract.exchange:
                        req_contract = copy.copy(req_contract)
                        if req_contract.secType == "FOP":
                            primary_exchange = getattr(req_contract, "primaryExchange", "") or ""
                            req_contract.exchange = primary_exchange or "CME"
                        else:
                            req_contract.exchange = "SMART"
                    try:
                        self._detail_tickers[con_id] = (
                            self._ib_proxy,
                            self._ib_proxy.reqMktData(req_contract),
                        )
                    except Exception:
                        continue
                self._proxy_tickers = {}
                self._proxy_task = None
                await self._ensure_proxy_tickers()
                self._proxy_probe_task = None
                self._start_proxy_probe()
                self._resubscribe_proxy_needed = False
        if (
            self._ib.isConnected()
            and self._ib_proxy.isConnected()
            and not self._resubscribe_main_needed
            and not self._resubscribe_proxy_needed
        ):
            self._reconnect_requested = False
            self._reconnect_fast_deadline = None
            if self._update_callback:
                self._update_callback()
# endregion


# region Helpers
def _pick_account_value(values: list[AccountValue]) -> AccountValue | None:
    for currency in ("BASE", "USD", "AUD"):
        for value in values:
            if value.currency == currency:
                return value
    return values[0] if values else None


def _pick_cached_value(
    cache: dict[tuple[str, str], tuple[float, datetime]], tag: str
) -> tuple[float, str, datetime] | None:
    for currency in ("BASE", "USD", "AUD"):
        key = (tag, currency)
        if key in cache:
            value, updated = cache[key]
            return value, currency, updated
    for (cached_tag, currency), (value, updated) in cache.items():
        if cached_tag == tag:
            return value, currency, updated
    return None


def _normalize_order_contract(contract: Contract) -> Contract:
    if contract.exchange:
        return contract
    normalized = copy.copy(contract)
    if contract.secType in ("STK", "OPT", "FUT", "FOP"):
        normalized.exchange = "SMART"
    return normalized
# endregion
