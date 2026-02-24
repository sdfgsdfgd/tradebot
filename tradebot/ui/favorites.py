"""Favorites watchlist screen."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import cast

from ib_insync import Contract, Crypto, Future, Index, PortfolioItem, Stock, Ticker
from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from .common import (
    _fmt_expiry,
    _pct24_72_from_price,
    _price_pct_dual_text,
    _safe_num,
    _ticker_price,
)
from .positions import PositionDetailScreen


@dataclass(frozen=True)
class _FavoriteSeed:
    symbol: str
    name: str
    sec_type: str
    exchange: str
    expiry: str = ""
    currency: str = "USD"


@dataclass(frozen=True)
class _FavoriteRow:
    seed: _FavoriteSeed
    contract: Contract


@dataclass(frozen=True)
class _SyntheticPortfolioItem:
    contract: Contract
    position: float = 0.0
    averageCost: float = 0.0
    marketPrice: float = 0.0
    marketValue: float = 0.0
    unrealizedPNL: float = 0.0
    realizedPNL: float = 0.0


_FAVORITE_SEEDS: tuple[_FavoriteSeed, ...] = (
    _FavoriteSeed("AGQ", "PROSHARES ULTRA SILVER", "STK", "SMART"),
    _FavoriteSeed("GDXU", "MICROSECTORS GOLD MINERS 3X", "STK", "SMART"),
    _FavoriteSeed("SLV", "ISHARES SILVER TRUST", "STK", "SMART"),
    _FavoriteSeed("BITU", "PROSHARES ULTRA BITCOIN ETF", "STK", "SMART"),
    _FavoriteSeed("UGL", "PROSHARES ULTRA GOLD", "STK", "SMART"),
    _FavoriteSeed("SI", "Silver Index", "FUT", "COMEX", expiry="202603"),
    _FavoriteSeed("BTC", "Bitcoin cryptocurrency", "CRYPTO", "PAXOS"),
    _FavoriteSeed("SILJ", "AMPLIFY JUNIOR SILVER MINERS", "STK", "SMART"),
    _FavoriteSeed("MBT", "Micro Bitcoin", "FUT", "CMECRYPTO", expiry="202602"),
    _FavoriteSeed("GLD", "SPDR GOLD SHARES", "STK", "SMART"),
    _FavoriteSeed("NVDA", "NVIDIA CORP", "STK", "SMART"),
    _FavoriteSeed("MNQ", "Micro E-Mini Nasdaq-100 Index", "FUT", "CME", expiry="202603"),
    _FavoriteSeed("MSFT", "MICROSOFT CORP", "STK", "SMART"),
    _FavoriteSeed("MCL", "Micro WTI Crude Oil", "FUT", "NYMEX", expiry="202603"),
    _FavoriteSeed("TLT", "ISHARES 20+ YEAR TREASURY BD", "STK", "SMART"),
    _FavoriteSeed("AAPL", "APPLE INC", "STK", "SMART"),
    _FavoriteSeed("BRK B", "BERKSHIRE HATHAWAY INC-CL B", "STK", "SMART"),
    _FavoriteSeed("TSLA", "TESLA INC", "STK", "SMART"),
    _FavoriteSeed("AMZN", "AMAZON.COM INC", "STK", "SMART"),
    _FavoriteSeed("TQQQ", "PROSHARES ULTRAPRO QQQ", "STK", "SMART"),
    _FavoriteSeed("META", "META PLATFORMS INC-CLASS A", "STK", "SMART"),
    _FavoriteSeed("ORCL", "ORACLE CORP", "STK", "SMART"),
    _FavoriteSeed("YINN", "DRX DLY FTSE CHINA BULL 3X", "STK", "SMART"),
    _FavoriteSeed("1OZ", "1 Ounce Gold Futures", "FUT", "COMEX", expiry="202602"),
    _FavoriteSeed("XSP", "Mini-SPX Index", "IND", "CBOE"),
    _FavoriteSeed("URNJ", "SPROTT JR URANIUM MINERS ETF", "STK", "SMART"),
)


class FavoritesScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("b", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("r", "reload", "Reload"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "open_details", "Details"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #favorites-table {
        height: 1fr;
        border: solid #1b3650;
    }

    #favorites-table:focus {
        border: solid #2c82c9;
    }

    #favorites-table > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #favorites-table > .datatable--header-cursor {
        background: #182230;
        color: #c6d4e1;
        text-style: bold;
    }

    #favorites-table > .datatable--header {
        background: #121820;
        color: #c6d4e1;
        text-style: bold;
    }

    #favorites-status {
        height: 1;
        padding: 0 1;
    }
    """

    _PX_COL_WIDTH = 32
    _TYPE_COL_WIDTH = 8
    _LAST_COL_WIDTH = 14
    _BA_COL_WIDTH = 21
    _TYPE_STYLE = {
        "STK": "bold #86dca9",
        "FUT": "bold #ffcc84",
        "CRYPTO": "bold #93d1ff",
        "IND": "bold #c9b2ff",
    }

    def __init__(self, client: IBKRClient, refresh_sec: float) -> None:
        super().__init__()
        self._client = client
        self._refresh_sec = max(float(refresh_sec), 0.25)
        self._favorites: list[_FavoriteRow] = []
        self._unresolved: list[str] = []
        self._row_keys: list[str] = []
        self._row_by_key: dict[str, _FavoriteRow] = {}
        self._ticker_con_ids: set[int] = set()
        self._ticker_loading: set[int] = set()
        self._closes_loading: set[int] = set()
        self._session_closes_by_con_id: dict[int, tuple[float | None, float | None]] = {}
        self._session_close_1ago_by_con_id: dict[int, float | None] = {}
        self._quote_signature_by_con_id: dict[
            int, tuple[float | None, float | None, float | None, float | None]
        ] = {}
        self._quote_updated_mono_by_con_id: dict[int, float] = {}
        self._refresh_timer = None
        self._reload_task: asyncio.Task | None = None
        self._reload_token = 0
        self._is_loading = False
        self._load_done = 0
        self._load_total = len(_FAVORITE_SEEDS)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(
            id="favorites-table",
            zebra_stripes=True,
            show_row_labels=False,
            cursor_foreground_priority="renderable",
            cursor_background_priority="css",
        )
        yield Static("Loading favorites...", id="favorites-status")
        yield Footer()

    async def on_mount(self) -> None:
        self._table = self.query_one("#favorites-table", DataTable)
        self._status = self.query_one("#favorites-status", Static)
        self._setup_columns()
        self._table.cursor_type = "row"
        self._table.focus()
        self._schedule_reload()
        self._refresh_timer = self.set_interval(self._refresh_sec, self._render_table)

    async def on_unmount(self) -> None:
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reload_task
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        for con_id in list(self._ticker_con_ids):
            self._client.release_ticker(con_id, owner="favorites")
        self._ticker_con_ids.clear()
        self._ticker_loading.clear()
        self._closes_loading.clear()
        self._session_close_1ago_by_con_id.clear()

    def _setup_columns(self) -> None:
        self._table.add_column("Symbol", width=26)
        self._table.add_column("Name", width=36)
        self._table.add_column("Type", width=self._TYPE_COL_WIDTH)
        self._table.add_column("Px 24-72".center(self._PX_COL_WIDTH), width=self._PX_COL_WIDTH)
        self._table.add_column("Last".center(self._LAST_COL_WIDTH), width=self._LAST_COL_WIDTH)
        self._table.add_column("Bid/Ask".center(self._BA_COL_WIDTH), width=self._BA_COL_WIDTH)

    def action_reload(self) -> None:
        self._schedule_reload()

    def action_cursor_down(self) -> None:
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        self._table.action_cursor_up()

    def action_open_details(self) -> None:
        row_index = self._table.cursor_coordinate.row
        if row_index < 0 or row_index >= len(self._row_keys):
            return
        row_key = self._row_keys[row_index]
        entry = self._row_by_key.get(row_key)
        if entry is None:
            return
        con_id = int(getattr(entry.contract, "conId", 0) or 0)
        item = self._client.portfolio_item(con_id) if con_id else None
        if item is None:
            item = cast(PortfolioItem, _SyntheticPortfolioItem(contract=entry.contract))
        self.app.push_screen(
            PositionDetailScreen(
                self._client,
                item,
                self._refresh_sec,
                session_closes=self._session_closes_by_con_id.get(con_id),
            )
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.control is self._table:
            self.action_open_details()

    def _schedule_reload(self) -> None:
        self._reload_token += 1
        token = self._reload_token
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._reload_task = loop.create_task(self._reload(token))

    async def _reload(self, token: int) -> None:
        if token != self._reload_token:
            return
        self._is_loading = True
        self._load_done = 0
        self._load_total = len(_FAVORITE_SEEDS)
        self._status.update("Loading favorites contracts...")
        for con_id in list(self._ticker_con_ids):
            self._client.release_ticker(con_id, owner="favorites")
        self._ticker_con_ids.clear()
        self._ticker_loading.clear()
        self._closes_loading.clear()
        self._session_closes_by_con_id.clear()
        self._session_close_1ago_by_con_id.clear()
        self._quote_signature_by_con_id.clear()
        self._quote_updated_mono_by_con_id.clear()
        self._favorites = []
        self._unresolved = []
        self._render_table()

        semaphore = asyncio.Semaphore(4)
        resolved: list[_FavoriteRow | None] = [None] * len(_FAVORITE_SEEDS)

        async def resolve_index(index: int, seed: _FavoriteSeed) -> tuple[int, _FavoriteRow | None]:
            async with semaphore:
                return index, await self._resolve_seed(seed)

        tasks = [
            asyncio.create_task(resolve_index(index, seed))
            for index, seed in enumerate(_FAVORITE_SEEDS)
        ]
        try:
            for finished in asyncio.as_completed(tasks):
                if token != self._reload_token:
                    return
                index, row = await finished
                self._load_done += 1
                resolved[index] = row
                self._favorites = [entry for entry in resolved if entry is not None]
                if row is not None:
                    con_id = int(getattr(row.contract, "conId", 0) or 0)
                    if con_id:
                        self._start_ticker_load(con_id, row.contract)
                        self._start_closes_load(con_id, row.contract)
                self._render_table()
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if token == self._reload_token:
                self._is_loading = False
        if token != self._reload_token:
            return
        self._favorites = [row for row in resolved if row is not None]
        self._unresolved = [seed.symbol for seed, row in zip(_FAVORITE_SEEDS, resolved) if row is None]
        self._render_table()

    async def _resolve_seed(self, seed: _FavoriteSeed) -> _FavoriteRow | None:
        if seed.sec_type == "FUT":
            for exchange in self._future_exchange_candidates(seed):
                try:
                    front = await self._client.front_future(seed.symbol, exchange=exchange)
                except Exception:
                    front = None
                if front is not None:
                    return _FavoriteRow(seed=seed, contract=front)
        candidate = self._seed_contract(seed)
        use_proxy = seed.sec_type in ("STK", "OPT")
        qualified = await self._client._qualify_contract(candidate, use_proxy=use_proxy)
        if qualified is not None:
            return _FavoriteRow(seed=seed, contract=qualified)
        fallback = await self._resolve_seed_fallback(seed)
        if fallback is not None:
            return _FavoriteRow(seed=seed, contract=fallback)
        return None

    @staticmethod
    def _future_exchange_candidates(seed: _FavoriteSeed) -> list[str]:
        out: list[str] = []
        for exchange in (
            seed.exchange,
            "CME",
            "COMEX",
            "NYMEX",
            "GLOBEX",
            "CMECRYPTO",
            "CBOT",
            "ECBOT",
        ):
            value = str(exchange or "").strip().upper()
            if not value or value == "SMART" or value in out:
                continue
            out.append(value)
        return out

    async def _resolve_seed_fallback(self, seed: _FavoriteSeed) -> Contract | None:
        mode = "STK" if seed.sec_type == "STK" else "FUT" if seed.sec_type == "FUT" else None
        if mode is None:
            return None
        queries = [seed.symbol]
        compact = seed.symbol.replace(" ", "")
        if compact and compact not in queries:
            queries.append(compact)
        if " " in seed.symbol:
            head = seed.symbol.split(" ")[0]
            if head and head not in queries:
                queries.append(head)
        for query in queries:
            try:
                results = await self._client.search_contracts(query, mode=mode, limit=48)
            except Exception:
                continue
            picked = self._pick_best(seed, results)
            if picked is not None:
                return picked
        return None

    @classmethod
    def _pick_best(cls, seed: _FavoriteSeed, candidates: list[Contract]) -> Contract | None:
        def norm(value: str) -> str:
            return "".join(ch for ch in value.upper() if ch.isalnum())

        target_symbol = norm(seed.symbol)
        best: tuple[int, Contract] | None = None
        for contract in candidates:
            sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
            symbol = norm(str(getattr(contract, "symbol", "") or ""))
            exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            score = 0
            if sec_type == seed.sec_type:
                score += 4
            if symbol == target_symbol:
                score += 4
            if seed.exchange and exchange == seed.exchange.upper():
                score += 2
            if seed.expiry and expiry.startswith(seed.expiry):
                score += 2
            if best is None or score > best[0]:
                best = (score, contract)
        return best[1] if best is not None else None

    @staticmethod
    def _seed_contract(seed: _FavoriteSeed) -> Contract:
        if seed.sec_type == "STK":
            return Stock(symbol=seed.symbol, exchange=seed.exchange, currency=seed.currency)
        if seed.sec_type == "FUT":
            return Future(
                symbol=seed.symbol,
                exchange=seed.exchange,
                currency=seed.currency,
                lastTradeDateOrContractMonth=seed.expiry,
            )
        if seed.sec_type == "CRYPTO":
            return Crypto(symbol=seed.symbol, exchange=seed.exchange, currency=seed.currency)
        if seed.sec_type == "IND":
            return Index(symbol=seed.symbol, exchange=seed.exchange, currency=seed.currency)
        return Contract(
            symbol=seed.symbol,
            secType=seed.sec_type,
            exchange=seed.exchange,
            currency=seed.currency,
        )

    def _prime_market_data(self) -> None:
        for entry in self._favorites:
            con_id = int(getattr(entry.contract, "conId", 0) or 0)
            if not con_id:
                continue
            self._start_ticker_load(con_id, entry.contract)
            self._start_closes_load(con_id, entry.contract)

    def _start_ticker_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._ticker_con_ids or con_id in self._ticker_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ticker_loading.add(con_id)
        loop.create_task(self._load_ticker(con_id, contract))

    async def _load_ticker(self, con_id: int, contract: Contract) -> None:
        try:
            await self._client.ensure_ticker(contract, owner="favorites")
            self._ticker_con_ids.add(con_id)
        except Exception:
            return
        finally:
            self._ticker_loading.discard(con_id)
        self._render_table()

    def _start_closes_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._session_closes_by_con_id or con_id in self._closes_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._closes_loading.add(con_id)
        loop.create_task(self._load_closes(con_id, contract))

    async def _load_closes(self, con_id: int, contract: Contract) -> None:
        try:
            prev_close, close_1ago, close_3ago = await self._client.session_close_anchors(contract)
            self._session_closes_by_con_id[con_id] = (prev_close, close_3ago)
            self._session_close_1ago_by_con_id[con_id] = close_1ago
        except Exception:
            pass
        finally:
            self._closes_loading.discard(con_id)
        self._render_table()

    def _render_table(self) -> None:
        if not hasattr(self, "_table"):
            return
        prev_coord = self._table.cursor_coordinate
        prev_key = None
        if 0 <= prev_coord.row < len(self._row_keys):
            prev_key = self._row_keys[prev_coord.row]

        self._table.clear()
        self._row_keys = []
        self._row_by_key = {}

        rows = self._sorted_favorites()
        for idx, entry in enumerate(rows):
            con_id = int(getattr(entry.contract, "conId", 0) or 0)
            row_key = f"fav:{con_id or idx}"
            self._table.add_row(
                self._symbol_cell(entry),
                Text(entry.seed.name, style="dim"),
                self._center_cell(
                    Text(entry.seed.sec_type, style=self._TYPE_STYLE.get(entry.seed.sec_type, "bold")),
                    self._TYPE_COL_WIDTH,
                ),
                self._price_cell(entry),
                self._last_cell(entry),
                self._bid_ask_cell(entry),
                key=row_key,
            )
            self._row_keys.append(row_key)
            self._row_by_key[row_key] = entry

        if self._row_keys:
            if prev_key and prev_key in self._row_keys:
                target_row = self._row_keys.index(prev_key)
            else:
                target_row = min(max(prev_coord.row, 0), len(self._row_keys) - 1)
            self._table.cursor_coordinate = (target_row, 0)
        self._status.update(self._status_text())

    def _status_text(self) -> str:
        conn = self._client.connection_state()
        base = f"IBKR {conn} | favorites: {len(self._favorites)}"
        if self._is_loading:
            return f"{base} | loading {self._load_done}/{self._load_total} | r reload | b/esc back"
        if self._unresolved:
            missing = ", ".join(self._unresolved[:6])
            more = f" +{len(self._unresolved) - 6}" if len(self._unresolved) > 6 else ""
            base = f"{base} | unresolved: {missing}{more}"
        return f"{base} | sorted by 72h% | r reload | l/enter details | b/esc back"

    def _sorted_favorites(self) -> list[_FavoriteRow]:
        def sort_key(entry: _FavoriteRow) -> tuple[int, float, str]:
            pct72 = self._pct72(entry)
            if pct72 is None:
                return (0, float("-inf"), entry.seed.symbol)
            return (1, float(pct72), entry.seed.symbol)

        return sorted(self._favorites, key=sort_key, reverse=True)

    def _pct72(self, entry: _FavoriteRow) -> float | None:
        con_id = int(getattr(entry.contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        ticker_price = _ticker_price(ticker) if ticker else None
        cached = self._session_closes_by_con_id.get(con_id)
        session_prev_close = cached[0] if cached else None
        session_prev_close_1ago = self._session_close_1ago_by_con_id.get(con_id)
        price = ticker_price if ticker_price is not None else session_prev_close
        close_3ago = cached[1] if cached else None
        _pct24, pct72 = _pct24_72_from_price(
            price=price,
            ticker=ticker,
            session_prev_close=session_prev_close,
            session_prev_close_1ago=session_prev_close_1ago,
            session_close_3ago=close_3ago,
        )
        return pct72

    def _symbol_cell(self, entry: _FavoriteRow) -> Text:
        contract = entry.contract
        symbol = Text(str(getattr(contract, "symbol", "") or entry.seed.symbol), style="bold")
        sec_type = str(getattr(contract, "secType", "") or entry.seed.sec_type).strip().upper()
        if sec_type == "FUT":
            expiry = _fmt_expiry(str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""))
            if expiry:
                symbol.append(" · ", style="grey35")
                symbol.append(expiry, style="grey58")
        exchange = str(getattr(contract, "exchange", "") or entry.seed.exchange).strip().upper()
        if exchange:
            symbol.append(" · ", style="grey35")
            symbol.append(exchange, style="dim")
        return symbol

    def _price_cell(self, entry: _FavoriteRow) -> Text:
        con_id = int(getattr(entry.contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        ticker_price = _ticker_price(ticker) if ticker else None
        cached = self._session_closes_by_con_id.get(con_id)
        session_prev_close = cached[0] if cached else None
        session_prev_close_1ago = self._session_close_1ago_by_con_id.get(con_id)
        price = ticker_price if ticker_price is not None else session_prev_close
        close_3ago = cached[1] if cached else None
        pct24, pct72 = _pct24_72_from_price(
            price=price,
            ticker=ticker,
            session_prev_close=session_prev_close,
            session_prev_close_1ago=session_prev_close_1ago,
            session_close_3ago=close_3ago,
        )
        text = _price_pct_dual_text(price, pct24, pct72, separator="·")
        glyph = self._price_direction_glyph(pct24, pct72)
        if glyph.plain:
            with_glyph = Text()
            with_glyph.append_text(glyph)
            if text.plain:
                with_glyph.append(" ")
            with_glyph.append_text(text)
            text = with_glyph
        if ticker_price is not None:
            age_sec = self._quote_age_seconds(con_id, ticker, price)
            ribbon = self._quote_age_ribbon(age_sec)
            if ribbon.plain:
                if text.plain:
                    text.append(" ", style="dim")
                text.append_text(ribbon)
        elif price is not None:
            text.append(" c", style="dim")
        centered = self._center_cell(text, self._PX_COL_WIDTH)
        return centered if isinstance(centered, Text) else Text(str(centered))

    def _last_cell(self, entry: _FavoriteRow) -> Text:
        con_id = int(getattr(entry.contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        ticker_price = _ticker_price(ticker) if ticker else None
        cached = self._session_closes_by_con_id.get(con_id)
        fallback_close = cached[0] if cached else None
        price = ticker_price if ticker_price is not None else fallback_close
        text = Text(
            f"{price:,.2f}" if price is not None else "n/a",
            style="" if ticker_price is not None else "dim",
        )
        if ticker_price is not None and ticker is not None:
            md = self._market_data_text(ticker)
            if md.plain:
                text.append(" ")
                text.append_text(md)
        elif fallback_close is not None:
            text.append(" [C]", style="dim")
        centered = self._center_cell(text, self._LAST_COL_WIDTH)
        return centered if isinstance(centered, Text) else Text(str(centered))

    def _bid_ask_cell(self, entry: _FavoriteRow) -> Text:
        con_id = int(getattr(entry.contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        bid = _safe_num(getattr(ticker, "bid", None)) if ticker else None
        ask = _safe_num(getattr(ticker, "ask", None)) if ticker else None
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
            text = Text(f"{bid:,.2f}/{ask:,.2f}")
        else:
            text = Text("n/a", style="dim")
        centered = self._center_cell(text, self._BA_COL_WIDTH)
        return centered if isinstance(centered, Text) else Text(str(centered))

    @staticmethod
    def _market_data_text(ticker: Ticker) -> Text:
        md_type = getattr(ticker, "marketDataType", None)
        source = str(getattr(ticker, "tbQuoteSource", "") or "").strip().lower()
        if source.startswith("stream") or source.startswith("live-"):
            return Text("[L]", style="bold #73d89e")
        if md_type in (1, 2):
            return Text("[L]", style="bold #73d89e")
        if md_type in (3, 4):
            return Text("[D]", style="bold #ffcc84")
        return Text("")

    @staticmethod
    def _price_direction_glyph(pct24: float | None, pct72: float | None) -> Text:
        ref = pct24 if pct24 is not None else pct72
        if ref is None:
            return Text("•", style="dim")
        if ref > 0:
            return Text("▲", style="bold green")
        if ref < 0:
            return Text("▼", style="bold red")
        return Text("•", style="dim")

    def _quote_age_seconds(
        self,
        con_id: int,
        ticker: Ticker | None,
        price: float | None,
    ) -> float | None:
        if not con_id or ticker is None:
            return None
        bid = _safe_num(getattr(ticker, "bid", None))
        ask = _safe_num(getattr(ticker, "ask", None))
        last = _safe_num(getattr(ticker, "last", None))
        signature = (bid, ask, last, price)
        if not any(value is not None for value in signature):
            return None
        now = time.monotonic()
        previous = self._quote_signature_by_con_id.get(con_id)
        if previous != signature:
            self._quote_signature_by_con_id[con_id] = signature
            self._quote_updated_mono_by_con_id[con_id] = now
            return 0.0
        updated = self._quote_updated_mono_by_con_id.get(con_id)
        if updated is None:
            self._quote_updated_mono_by_con_id[con_id] = now
            return 0.0
        return max(0.0, now - updated)

    @staticmethod
    def _quote_age_ribbon(age_sec: float | None) -> Text:
        if age_sec is None:
            return Text("")
        if age_sec < 1.5:
            return Text("▰▰▰▰", style="bold #73d89e")
        if age_sec < 3.5:
            return Text("▰▰▰▱", style="#6fc18f")
        if age_sec < 6.0:
            return Text("▰▰▱▱", style="#8fa2b3")
        if age_sec < 10.0:
            return Text("▰▱▱▱", style="#778797")
        return Text("▱▱▱▱", style="#5e6a74")

    @staticmethod
    def _center_cell(value: Text | str, width: int) -> Text | str:
        if width <= 0:
            return value
        if isinstance(value, Text):
            plain = value.plain
            pad = int(width) - len(plain)
            if pad <= 0:
                return value
            left = pad // 2
            right = pad - left
            centered = Text(" " * left)
            centered.append_text(value)
            centered.append(" " * right)
            return centered
        raw = str(value or "")
        pad = int(width) - len(raw)
        if pad <= 0:
            return raw
        left = pad // 2
        right = pad - left
        return f"{' ' * left}{raw}{' ' * right}"
