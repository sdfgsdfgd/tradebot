"""Thin async wrapper over ib_insync for snapshot-style position pulls."""
from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from ib_insync import (
    AccountValue,
    ContFuture,
    Contract,
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
        self._detail_tickers: dict[int, tuple[IB, Ticker]] = {}
        self._historical_bar_cache: dict[
            tuple[str, int, str, str, bool, str], tuple[list[tuple[datetime, float]], float]
        ] = {}
        self._historical_bar_ohlcv_cache: dict[
            tuple[str, int, str, str, bool, str], tuple[list[OhlcvBar], float]
        ] = {}
        self._front_future_cache: dict[tuple[str, str], tuple[Contract, float]] = {}
        self._update_callback: Callable[[], None] | None = None
        self._pnl: PnL | None = None
        self._pnl_account: str | None = None
        self._account_value_cache: dict[tuple[str, str], tuple[float, datetime]] = {}
        self._session_close_cache: dict[int, tuple[float | None, float | None, float]] = {}
        self._farm_connectivity_lost = False
        self._reconnect_requested = False
        self._resubscribe_main_needed = False
        self._resubscribe_proxy_needed = False
        self._reconnect_task: asyncio.Task | None = None
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

    async def connect(self) -> None:
        self._shutdown = False
        if self._ib.isConnected():
            return
        async with self._connect_lock:
            if self._ib.isConnected():
                return
            if hasattr(self._ib, "connectAsync"):
                await self._ib.connectAsync(
                    self._config.host,
                    self._config.port,
                    clientId=self._config.client_id,
                    timeout=5,
                )
            else:
                await asyncio.to_thread(
                    self._ib.connect,
                    self._config.host,
                    self._config.port,
                    self._config.client_id,
                    5,
                )

    async def connect_proxy(self) -> None:
        self._shutdown = False
        if self._ib_proxy.isConnected():
            return
        async with self._connect_proxy_lock:
            if self._ib_proxy.isConnected():
                return
            if hasattr(self._ib_proxy, "connectAsync"):
                await self._ib_proxy.connectAsync(
                    self._config.host,
                    self._config.port,
                    clientId=self._config.proxy_client_id,
                    timeout=5,
                )
            else:
                await asyncio.to_thread(
                    self._ib_proxy.connect,
                    self._config.host,
                    self._config.port,
                    self._config.proxy_client_id,
                    5,
                )

    async def disconnect(self) -> None:
        self._shutdown = True
        self._stop_reconnect_loop()
        self._reconnect_requested = False
        if self._ib.isConnected():
            try:
                self._ib.disconnect()
            except OSError:
                # Avoid noisy shutdown if the socket is already closed.
                pass
        if self._ib_proxy.isConnected():
            try:
                self._ib_proxy.disconnect()
            except OSError:
                pass
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
        self._detail_tickers = {}
        self._pnl = None
        self._pnl_account = None
        self._account_value_cache = {}
        self._session_close_cache = {}
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

    async def ensure_ticker(self, contract: Contract) -> Ticker:
        use_proxy = contract.secType in ("STK", "OPT")
        if use_proxy:
            await self.connect_proxy()
            md_type = 3 if self._proxy_force_delayed else 1
            self._ib_proxy.reqMarketDataType(md_type)
            ib = self._ib_proxy
        else:
            await self.connect()
            self._ib.reqMarketDataType(3)
            ib = self._ib
        con_id = int(contract.conId or 0)
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
                return ticker
            return cached_ticker
        ticker = ib.reqMktData(req_contract)
        if con_id:
            self._detail_tickers[con_id] = (ib, ticker)
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
        order = LimitOrder(action, quantity, limit_price, tif="GTC")
        if contract.secType == "STK":
            allow_outside = outside_rth
            outside_session, include_overnight = _session_flags(
                datetime.now(tz=_ET_ZONE)
            )
            if include_overnight:
                order_contract = copy.copy(contract)
                order_contract.exchange = "OVERNIGHT"
            if allow_outside and outside_session:
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
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib
            req_contract = contract
            if contract.secType in ("STK", "OPT") and (
                not getattr(contract, "exchange", "") or getattr(contract, "exchange", "") == "OVERNIGHT"
            ):
                req_contract = copy.copy(contract)
                req_contract.exchange = "SMART"
            bars = await ib.reqHistoricalDataAsync(
                req_contract,
                endDateTime="",
                durationStr="2 W",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
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
            str(duration_str),
        )
        cached = self._historical_bar_cache.get(key)
        if cached:
            bars, cached_at = cached
            if time.monotonic() - cached_at < float(cache_ttl_sec):
                return list(bars)

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            cached = self._historical_bar_cache.get(key)
            if cached:
                bars, cached_at = cached
                if time.monotonic() - cached_at < float(cache_ttl_sec):
                    return list(bars)
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib

            req_contract = contract
            if sec_type in ("STK", "OPT") and (
                not getattr(contract, "exchange", "") or getattr(contract, "exchange", "") == "OVERNIGHT"
            ):
                req_contract = copy.copy(contract)
                req_contract.exchange = "SMART"

            raw = await ib.reqHistoricalDataAsync(
                req_contract,
                endDateTime="",
                durationStr=str(duration_str),
                barSizeSetting=str(bar_size),
                whatToShow=str(what_to_show),
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )

            bars: list[tuple[datetime, float]] = []
            for bar in raw or []:
                dt = getattr(bar, "date", None)
                if isinstance(dt, str):
                    dt = util.parseIBDatetime(dt)
                if dt is None:
                    continue
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.replace(tzinfo=None)
                try:
                    close = float(getattr(bar, "close", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if close <= 0:
                    continue
                bars.append((dt, close))

            self._historical_bar_cache[key] = (bars, time.monotonic())
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
            str(duration_str),
        )
        cached = self._historical_bar_ohlcv_cache.get(key)
        if cached:
            bars, cached_at = cached
            if time.monotonic() - cached_at < float(cache_ttl_sec):
                return list(bars)

        use_proxy = sec_type in ("STK", "OPT")
        lock = self._proxy_lock if use_proxy else self._lock
        async with lock:
            cached = self._historical_bar_ohlcv_cache.get(key)
            if cached:
                bars, cached_at = cached
                if time.monotonic() - cached_at < float(cache_ttl_sec):
                    return list(bars)
            if use_proxy:
                await self.connect_proxy()
                ib = self._ib_proxy
            else:
                await self.connect()
                ib = self._ib

            req_contract = contract
            if sec_type in ("STK", "OPT") and (
                not getattr(contract, "exchange", "") or getattr(contract, "exchange", "") == "OVERNIGHT"
            ):
                req_contract = copy.copy(contract)
                req_contract.exchange = "SMART"

            raw = await ib.reqHistoricalDataAsync(
                req_contract,
                endDateTime="",
                durationStr=str(duration_str),
                barSizeSetting=str(bar_size),
                whatToShow=str(what_to_show),
                useRTH=1 if use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )

            bars: list[OhlcvBar] = []
            for bar in raw or []:
                dt = getattr(bar, "date", None)
                if isinstance(dt, str):
                    dt = util.parseIBDatetime(dt)
                if dt is None:
                    continue
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.replace(tzinfo=None)
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

            self._historical_bar_ohlcv_cache[key] = (bars, time.monotonic())
            return list(bars)

    async def stock_option_chain(self, symbol: str):
        """Return (qualified_underlying, chain) for an equity option underlyer."""
        candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
        underlying = await self._qualify_contract(candidate, use_proxy=True) or candidate
        await self.connect_proxy()
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
        await self.connect_proxy()
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
            await self.connect()
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

    def release_ticker(self, con_id: int) -> None:
        entry = self._detail_tickers.pop(con_id, None)
        if entry:
            ib, ticker = entry
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:
                pass

    def account_value(
        self, tag: str
    ) -> tuple[float | None, str | None, datetime | None]:
        account = self._config.account or ""
        cached = _pick_cached_value(self._account_value_cache, tag)
        if cached:
            value, currency, updated_at = cached
            return value, currency, updated_at
        values = [v for v in self._ib.accountValues(account) if v.tag == tag]
        if not values:
            return None, None, None
        chosen = _pick_account_value(values)
        if not chosen:
            return None, None, None
        try:
            return float(chosen.value), chosen.currency, None
        except (TypeError, ValueError):
            return None, chosen.currency, None

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
        if use_proxy:
            await self.connect_proxy()
            ib = self._ib_proxy
        else:
            await self.connect()
            ib = self._ib
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
        self._handle_conn_error(errorCode)

    def _on_error_proxy(self, reqId, errorCode, errorString, contract) -> None:
        if errorCode == 10167 and not self._proxy_force_delayed:
            self._proxy_force_delayed = True
            self._start_proxy_resubscribe()
        if errorCode in (10089, 10090, 10091, 10168) and contract:
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id and con_id not in self._proxy_contract_force_delayed:
                self._proxy_contract_force_delayed.add(con_id)
                self._start_proxy_contract_delayed_resubscribe(contract)
        self._handle_conn_error(errorCode)

    def _on_disconnected_main(self) -> None:
        if self._shutdown:
            return
        self._resubscribe_main_needed = True
        self._account_updates_started = False
        self._pnl = None
        self._pnl_account = None
        self._account_value_cache = {}
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
        self._reconnect_requested = True
        self._start_reconnect_loop()
        if self._update_callback:
            self._update_callback()

    def _start_proxy_contract_delayed_resubscribe(self, contract: Contract) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._resubscribe_proxy_contract_delayed(contract))

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
        deadline = time.monotonic() + self._config.reconnect_timeout_sec
        while self._reconnect_requested and time.monotonic() < deadline:
            await self._reconnect_once()
            if not self._reconnect_requested:
                break
            await asyncio.sleep(self._config.reconnect_interval_sec)

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
