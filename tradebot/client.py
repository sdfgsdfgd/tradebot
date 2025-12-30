"""Thin async wrapper over ib_insync for snapshot-style position pulls."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Iterable

from ib_insync import AccountValue, ContFuture, Contract, IB, PnL, PortfolioItem, Stock, Ticker

from .config import IBKRConfig

_INDEX_CONTRACTS: dict[str, list[str]] = {
    "NQ": ["GLOBEX", "CME"],
    "ES": ["GLOBEX", "CME"],
    "YM": ["ECBOT", "CBOT", "GLOBEX"],
}
_PROXY_SYMBOLS = ("QQQ", "TQQQ")


class IBKRClient:
    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib = IB()
        self._lock = asyncio.Lock()
        self._account_updates_started = False
        self._index_contracts: dict[str, Contract] = {}
        self._index_tickers: dict[str, Ticker] = {}
        self._index_task: asyncio.Task | None = None
        self._index_error: str | None = None
        self._proxy_contracts: dict[str, Contract] = {}
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_task: asyncio.Task | None = None
        self._proxy_error: str | None = None
        self._detail_tickers: dict[int, Ticker] = {}
        self._update_callback: Callable[[], None] | None = None
        self._pnl: PnL | None = None
        self._pnl_account: str | None = None
        self._account_value_cache: dict[tuple[str, str], tuple[float, datetime]] = {}
        self._connectivity_lost = False
        self._reconnect_task: asyncio.Task | None = None
        self._ib.errorEvent += self._on_error
        self._ib.updatePortfolioEvent += self._on_stream_update
        self._ib.pendingTickersEvent += self._on_stream_update
        self._ib.pnlEvent += self._on_stream_update
        self._ib.accountValueEvent += self._on_account_value

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    async def connect(self) -> None:
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

    async def disconnect(self) -> None:
        if not self._ib.isConnected():
            return
        try:
            self._ib.disconnect()
        except OSError:
            # Avoid noisy shutdown if the socket is already closed.
            pass
        self._account_updates_started = False
        self._index_tickers = {}
        self._index_task = None
        self._index_error = None
        self._proxy_tickers = {}
        self._proxy_task = None
        self._proxy_error = None
        self._detail_tickers = {}
        self._pnl = None
        self._pnl_account = None
        self._account_value_cache = {}

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

    async def ensure_ticker(self, contract: Contract) -> Ticker:
        await self.connect()
        # Allow delayed market data fallback when real-time is unavailable.
        self._ib.reqMarketDataType(3)
        con_id = int(contract.conId or 0)
        if con_id in self._detail_tickers:
            return self._detail_tickers[con_id]
        ticker = self._ib.reqMktData(contract)
        if con_id:
            self._detail_tickers[con_id] = ticker
        return ticker

    def release_ticker(self, con_id: int) -> None:
        ticker = self._detail_tickers.pop(con_id, None)
        if ticker:
            try:
                self._ib.cancelMktData(ticker.contract)
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
            await self._ensure_account_updates()
            for ticker in self._index_tickers.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            for ticker in self._proxy_tickers.values():
                try:
                    self._ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
            self._index_tickers = {}
            self._proxy_tickers = {}
            self._index_task = None
            self._proxy_task = None
            await self._ensure_index_tickers()
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
        await self.connect()
        # Allow delayed market data fallback when real-time is unavailable.
        self._ib.reqMarketDataType(3)
        if not self._proxy_contracts:
            self._proxy_contracts = await self._qualify_proxy_contracts()
        if not self._proxy_tickers:
            for symbol, contract in self._proxy_contracts.items():
                self._proxy_tickers[symbol] = self._ib.reqMktData(contract)

    async def _qualify_proxy_contracts(self) -> dict[str, Contract]:
        qualified: dict[str, Contract] = {}
        for symbol in _PROXY_SYMBOLS:
            candidate = Stock(symbol=symbol, exchange="SMART", currency="USD")
            try:
                result = await self._ib.qualifyContractsAsync(candidate)
            except Exception:
                continue
            if result:
                qualified[symbol] = result[0]
        return qualified

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
            async with self._lock:
                await self._ensure_proxy_tickers()
            self._proxy_error = None
        except Exception as exc:
            self._proxy_error = str(exc)

    def _on_error(self, reqId, errorCode, errorString, contract) -> None:
        if errorCode == 1100:
            self._connectivity_lost = True
            self._start_reconnect_loop()
        elif errorCode in (1101, 1102):
            self._connectivity_lost = False
            self._stop_reconnect_loop()

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
        while self._connectivity_lost and time.monotonic() < deadline:
            await self._reconnect_once()
            await asyncio.sleep(self._config.reconnect_interval_sec)

    async def _reconnect_once(self) -> None:
        async with self._lock:
            await self.disconnect()
            await self.connect()
            self._account_updates_started = False
            self._index_tickers = {}
            self._index_task = None
            self._proxy_tickers = {}
            self._proxy_task = None
            await self._ensure_account_updates()
            await self._ensure_index_tickers()
            await self._ensure_proxy_tickers()
