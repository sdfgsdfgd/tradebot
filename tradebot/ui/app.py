"""Portfolio TUI (positions) + bot hub entrypoint."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from ib_insync import Contract, PnL, PortfolioItem, Ticker
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from ..config import load_config
from .bot_runtime import BotRuntime
from .common import (
    _INDEX_LABELS,
    _INDEX_ORDER,
    _PROXY_LABELS,
    _PROXY_ORDER,
    _SECTION_ORDER,
    _SECTION_TYPES,
    _combined_value_pct,
    _fmt_money,
    _estimate_buying_power,
    _estimate_net_liq,
    _market_session_label,
    _option_display_price,
    _pct_change,
    _pnl_pct_value,
    _pnl_text,
    _pnl_value,
    _portfolio_row,
    _portfolio_sort_key,
    _price_pct_dual_text,
    _quote_status_line,
    _safe_num,
    _ticker_close,
    _ticker_line,
    _ticker_price,
    _unrealized_pnl_values,
)
from .favorites import FavoritesScreen
from .positions import PositionDetailScreen
from .store import PortfolioSnapshot


@dataclass
class _SyntheticPortfolioItem:
    contract: Contract
    position: float = 0.0
    averageCost: float = 0.0
    marketPrice: float = 0.0
    marketValue: float = 0.0
    unrealizedPNL: float = 0.0
    realizedPNL: float = 0.0


class _SearchDrawer(Static):
    can_focus = True


# region Positions UI
class PositionsApp(App):
    _PX_24_72_COL_WIDTH = 32
    _QTY_COL_WIDTH = 7
    _UNREAL_COL_WIDTH = 30
    _REALIZED_COL_WIDTH = 14
    _CLOSES_RETRY_SEC = 30.0
    _SEARCH_MODES = ("STK", "FUT", "OPT", "FOP")
    _SEARCH_LIMIT = 5
    _SEARCH_FETCH_LIMIT = 96
    _SEARCH_DEBOUNCE_SEC = 0.18

    _SECTION_HEADER_STYLE_BY_TYPE = {
        "OPT": "bold #8fbfff",
        "STK": "bold #86dca9",
        "FUT": "bold #ffcc84",
        "FOP": "bold #c9b2ff",
    }

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "open_details", "Details"),
        ("f", "open_favorites", "Favorites"),
        ("ctrl+f", "toggle_search", "Search"),
        ("ctrl+t", "toggle_bot", "Bot"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #ticker {
        height: 2;
        padding: 0 1;
    }

    #positions {
        height: 1fr;
        border: solid #1b3650;
    }

    #positions:focus {
        border: solid #2c82c9;
    }

    #positions > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #positions > .datatable--header-cursor {
        background: #182230;
        color: #c6d4e1;
        text-style: bold;
    }

    #positions > .datatable--header,
    #bot-presets > .datatable--header,
    #bot-instances > .datatable--header,
    #bot-orders > .datatable--header,
    #bot-logs > .datatable--header,
    #bot-config > .datatable--header {
        background: #121820;
        color: #c6d4e1;
        text-style: bold;
    }

    #status {
        height: 1;
        padding: 0 1;
    }

    #search {
        height: auto;
        min-height: 1;
        padding: 0 1;
        border-top: solid #1b3650;
        background: #0b1219;
    }

    #detail-body {
        height: 1fr;
        layout: horizontal;
    }

    #detail-left {
        width: 2fr;
        height: 1fr;
        padding: 0 1;
    }

    #detail-right {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    #detail-legend {
        height: 1;
        padding: 0 1;
        color: #8aa0b6;
    }

    #bot-body {
        height: 1fr;
        layout: vertical;
    }

    #bot-status {
        height: 5;
        padding: 0 1;
    }

    #bot-presets {
        height: 2fr;
        padding: 0 1;
        border: solid #1b3650;
    }

    #bot-instances {
        height: 10;
        padding: 0 1;
        border: solid #1b3650;
    }

    #bot-orders {
        height: 1fr;
        padding: 0 1;
        border: solid #1b3650;
    }

    #bot-logs {
        height: 1fr;
        padding: 0 1;
        border: solid #1b3650;
    }

    #bot-presets,
    #bot-instances,
    #bot-orders,
    #bot-logs,
    #bot-config {
        background-tint: #000000 12%;
    }

    #bot-presets:focus,
    #bot-instances:focus,
    #bot-orders:focus,
    #bot-logs:focus,
    #bot-config:focus {
        border: solid #2c82c9;
        background-tint: #000000 0%;
    }

    #bot-presets > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #bot-instances > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #bot-orders > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #bot-logs > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #bot-config > .datatable--cursor {
        background: #0d1117;
        text-style: none;
    }

    #bot-presets > .datatable--header-cursor,
    #bot-instances > .datatable--header-cursor,
    #bot-orders > .datatable--header-cursor,
    #bot-logs > .datatable--header-cursor,
    #bot-config > .datatable--header-cursor {
        background: #182230;
        color: #c6d4e1;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._client = IBKRClient(self._config)
        self._bot_runtime = BotRuntime(self._client, self._config.detail_refresh_sec)
        self._snapshot = PortfolioSnapshot()
        self._refresh_lock = asyncio.Lock()
        self._dirty = False
        self._dirty_task: asyncio.Task | None = None
        self._row_count = 0
        self._row_keys: list[str] = []
        self._row_item_by_key: dict[str, PortfolioItem] = {}
        self._column_count = 0
        self._index_tickers: dict[str, Ticker] = {}
        self._index_error: str | None = None
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_error: str | None = None
        self._pnl: PnL | None = None
        self._ticker_con_ids: set[int] = set()
        self._session_closes_by_con_id: dict[int, tuple[float | None, float | None]] = {}
        self._closes_retry_at_by_con_id: dict[int, float] = {}
        self._option_underlying_con_id: dict[int, int] = {}
        self._ticker_loading: set[int] = set()
        self._closes_loading: set[int] = set()
        self._underlying_loading: set[int] = set()
        self._md_session: str | None = None
        self._net_liq: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._net_liq_daily_anchor: float | None = None
        self._buying_power: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._buying_power_daily_anchor: float | None = None
        self._quote_signature_by_con_id: dict[
            int, tuple[float | None, float | None, float | None, float | None]
        ] = {}
        self._quote_updated_mono_by_con_id: dict[int, float] = {}
        self._search_active = False
        self._search_query = ""
        self._search_mode_index = 0
        self._search_results: list[Contract] = []
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0  # 0=CALL(left), 1=PUT(right) for OPT mode
        self._search_opt_expiry_index = 0
        self._search_loading = False
        self._search_error: str | None = None
        self._search_generation = 0
        self._search_ticker_con_ids: set[int] = set()
        self._search_ticker_loading: set[int] = set()
        self._search_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="ticker")
        yield DataTable(
            id="positions",
            zebra_stripes=True,
            show_row_labels=False,
            cursor_foreground_priority="renderable",
            cursor_background_priority="css",
        )
        yield Static("Starting...", id="status")
        yield _SearchDrawer("", id="search")
        yield Footer()

    async def on_mount(self) -> None:
        self._table = self.query_one(DataTable)
        self._ticker = self.query_one("#ticker", Static)
        self._status = self.query_one("#status", Static)
        self._search = self.query_one("#search", Static)
        self._search.display = False
        self._setup_columns()
        self._table.cursor_type = "row"
        self._table.focus()
        self._bot_runtime.install(self)
        self._client.set_update_callback(self._mark_dirty)
        await self.refresh_positions()

    async def on_unmount(self) -> None:
        self._cancel_search_task()
        self._clear_search_tickers()
        await self._client.disconnect()

    def _setup_columns(self) -> None:
        self._columns = [
            "Symbol",
            "Qty",
            "AvgCost",
            "Px 24-72",
            "Unreal (Pos)",
            "Realized (Pos)",
        ]
        self._column_count = len(self._columns)
        for label in self._columns:
            if label == "Px 24-72":
                self._table.add_column(
                    label.center(self._PX_24_72_COL_WIDTH),
                    width=self._PX_24_72_COL_WIDTH,
                )
            elif label == "Qty":
                self._table.add_column(
                    label.center(self._QTY_COL_WIDTH),
                    width=self._QTY_COL_WIDTH,
                )
            elif label.startswith("Unreal"):
                self._table.add_column(
                    label.center(self._UNREAL_COL_WIDTH),
                    width=self._UNREAL_COL_WIDTH,
                )
            elif label.startswith("Realized"):
                self._table.add_column(
                    label.center(self._REALIZED_COL_WIDTH),
                    width=self._REALIZED_COL_WIDTH,
                )
            else:
                self._table.add_column(label)

    def _center_cell(self, value: Text | str, width: int) -> Text | str:
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

    async def action_refresh(self) -> None:
        await self.refresh_positions(hard=True)

    def action_cursor_down(self) -> None:
        if self._search_active:
            self._move_search_selection(1)
            return
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        if self._search_active:
            self._move_search_selection(-1)
            return
        self._table.action_cursor_up()

    def action_open_details(self) -> None:
        if self._search_active:
            self._open_search_selection()
            return
        row_index = self._table.cursor_coordinate.row
        if row_index < 0 or row_index >= len(self._row_keys):
            return
        row_key = self._row_keys[row_index]
        item = self._row_item_by_key.get(row_key)
        if item:
            con_id = int(getattr(item.contract, "conId", 0) or 0)
            self.push_screen(
                PositionDetailScreen(
                    self._client,
                    item,
                    self._config.detail_refresh_sec,
                    session_closes=self._session_closes_by_con_id.get(con_id),
                )
            )

    def action_toggle_bot(self) -> None:
        if self._search_active:
            self._close_search()
        self._bot_runtime.toggle(self)

    def action_open_favorites(self) -> None:
        if self._search_active:
            self._close_search()
        self.push_screen(FavoritesScreen(self._client, self._config.detail_refresh_sec))

    def action_toggle_search(self) -> None:
        if self._search_active:
            self._close_search()
            return
        self._open_search()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._search_active:
            return
        item = self._row_item_by_key.get(event.row_key.value)
        if item:
            self.push_screen(
                PositionDetailScreen(
                    self._client,
                    item,
                    self._config.detail_refresh_sec,
                )
            )

    def on_key(self, event: events.Key) -> None:
        if self._search_active:
            if event.key == "escape":
                self._close_search()
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter":
                self._open_search_selection()
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                self._cycle_search_mode(1)
                event.prevent_default()
                event.stop()
                return
            if event.key in ("shift+tab", "backtab"):
                self._cycle_search_mode(-1)
                event.prevent_default()
                event.stop()
                return
            if event.key in ("up", "k"):
                self._move_search_selection(-1)
                event.prevent_default()
                event.stop()
                return
            if event.key in ("down", "j"):
                self._move_search_selection(1)
                event.prevent_default()
                event.stop()
                return
            if event.key in ("left", "h"):
                if self._search_mode() == "OPT":
                    self._search_side = 0
                    self._render_search()
                    event.prevent_default()
                    event.stop()
                    return
            if event.key in ("right", "l"):
                if self._search_mode() == "OPT":
                    self._search_side = 1
                    self._render_search()
                    event.prevent_default()
                    event.stop()
                    return
            if event.character in ("[", "]"):
                if self._search_mode() == "OPT":
                    self._cycle_search_expiry(-1 if event.character == "[" else 1)
                    event.prevent_default()
                    event.stop()
                    return
            if event.key == "backspace":
                if self._search_query:
                    self._search_query = self._search_query[:-1]
                    self._queue_search()
                else:
                    self._render_search()
                event.prevent_default()
                event.stop()
                return
            if event.character and event.character.isprintable():
                char = event.character
                if char.isalpha():
                    char = char.upper()
                self._search_query += char
                self._queue_search()
                event.prevent_default()
                event.stop()
                return
            event.prevent_default()
            event.stop()
            return

        if event.key in ("slash", "/") or event.character == "/":
            self._open_search()
            event.prevent_default()
            event.stop()

    def _open_search(self) -> None:
        self._search_active = True
        self._search_query = ""
        self._search_mode_index = 0
        self._search_results = []
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._search_loading = False
        self._search_error = None
        self._search_generation += 1
        self._cancel_search_task()
        self._search.display = True
        self._search.focus()
        self._render_search()

    def _close_search(self) -> None:
        self._search_active = False
        self._search_query = ""
        self._search_results = []
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._search_loading = False
        self._search_error = None
        self._search_generation += 1
        self._cancel_search_task()
        self._clear_search_tickers()
        self._search.display = False
        self._search.update("")
        self._table.focus()

    def _cancel_search_task(self) -> None:
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
        self._search_task = None

    def _cycle_search_mode(self, step: int) -> None:
        count = len(self._SEARCH_MODES)
        if count <= 0:
            return
        self._search_mode_index = (self._search_mode_index + int(step)) % count
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._queue_search()

    def _move_search_selection(self, delta: int) -> None:
        total = self._search_row_count()
        if total <= 0:
            self._render_search()
            return
        max_index = total - 1
        self._search_selected = min(max(self._search_selected + int(delta), 0), max_index)
        self._ensure_search_visible()
        self._render_search()

    def _search_mode(self) -> str:
        return self._SEARCH_MODES[self._search_mode_index]

    def _search_opt_expiries(self) -> list[str]:
        expiries = {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in self._search_results
            if str(getattr(contract, "secType", "") or "").strip().upper() == "OPT"
        }
        cleaned = [value for value in expiries if value]
        return sorted(cleaned)

    def _current_opt_expiry(self) -> str | None:
        expiries = self._search_opt_expiries()
        if not expiries:
            self._search_opt_expiry_index = 0
            return None
        self._search_opt_expiry_index = min(
            max(int(self._search_opt_expiry_index), 0),
            len(expiries) - 1,
        )
        return expiries[self._search_opt_expiry_index]

    def _cycle_search_expiry(self, step: int) -> None:
        if self._search_mode() != "OPT":
            return
        expiries = self._search_opt_expiries()
        if not expiries:
            return
        count = len(expiries)
        self._search_opt_expiry_index = (self._search_opt_expiry_index + int(step)) % count
        self._search_selected = self._default_opt_row_index()
        self._search_scroll = 0
        self._ensure_search_visible()
        self._render_search()

    def _option_pair_rows(self) -> list[tuple[Contract | None, Contract | None]]:
        target_expiry = self._current_opt_expiry()
        by_strike: dict[float, list[Contract | None]] = {}
        for contract in self._search_results:
            if str(getattr(contract, "secType", "") or "").strip().upper() != "OPT":
                continue
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            if target_expiry and expiry != target_expiry:
                continue
            try:
                strike = float(getattr(contract, "strike", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if strike not in by_strike:
                by_strike[strike] = [None, None]
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            if right == "P":
                by_strike[strike][1] = contract
            else:
                by_strike[strike][0] = contract
        rows: list[tuple[Contract | None, Contract | None]] = []
        for strike in sorted(by_strike.keys()):
            call, put = by_strike[strike]
            rows.append((call, put))
        return rows

    def _search_option_contracts(self) -> dict[int, Contract]:
        if not self._search_active or self._search_mode() != "OPT":
            return {}
        contracts: dict[int, Contract] = {}
        for call_contract, put_contract in self._option_pair_rows():
            for contract in (call_contract, put_contract):
                if contract is None:
                    continue
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id and con_id not in contracts:
                    contracts[con_id] = contract
        return contracts

    def _sync_search_option_tickers(self) -> None:
        wanted = self._search_option_contracts()
        wanted_ids = set(wanted.keys())
        for con_id in list(self._search_ticker_con_ids):
            if con_id in wanted_ids:
                continue
            self._client.release_ticker(con_id, owner="search")
            self._search_ticker_con_ids.discard(con_id)
            self._search_ticker_loading.discard(con_id)
        for con_id, contract in wanted.items():
            self._start_search_ticker_load(con_id, contract)

    def _clear_search_tickers(self) -> None:
        for con_id in list(self._search_ticker_con_ids):
            self._client.release_ticker(con_id, owner="search")
        self._search_ticker_con_ids.clear()
        self._search_ticker_loading.clear()

    def _start_search_ticker_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._search_ticker_con_ids or con_id in self._search_ticker_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._search_ticker_loading.add(con_id)
        loop.create_task(self._load_search_ticker(con_id, contract))

    async def _load_search_ticker(self, con_id: int, contract: Contract) -> None:
        try:
            await self._client.ensure_ticker(contract, owner="search")
        except Exception:
            return
        finally:
            self._search_ticker_loading.discard(con_id)
        wanted = self._search_option_contracts()
        if con_id not in wanted:
            self._client.release_ticker(con_id, owner="search")
            return
        self._search_ticker_con_ids.add(con_id)
        self._render_search()

    @staticmethod
    def _option_quote_value(ticker: Ticker | None) -> float | None:
        if ticker is None:
            return None
        bid = _safe_num(getattr(ticker, "bid", None))
        ask = _safe_num(getattr(ticker, "ask", None))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
            return (float(bid) + float(ask)) / 2.0
        for value in (
            _safe_num(getattr(ticker, "last", None)),
            _safe_num(getattr(ticker, "close", None)),
        ):
            if value is not None and value > 0:
                return float(value)
        return None

    def _search_option_cell_text(self, right: str, contract: Contract | None) -> str:
        if contract is None:
            return f"{right} --"
        con_id = int(getattr(contract, "conId", 0) or 0)
        if not con_id:
            return f"{right} --"
        ticker = self._client.ticker_for_con_id(con_id)
        quote = self._option_quote_value(ticker)
        if quote is not None:
            return f"{right} {quote:.2f}"
        if con_id in self._search_ticker_loading:
            return f"{right} ..."
        return f"{right} --"

    @staticmethod
    def _option_row_strike(
        call_contract: Contract | None,
        put_contract: Contract | None,
    ) -> float | None:
        source = call_contract if call_contract is not None else put_contract
        if source is None:
            return None
        try:
            strike = float(getattr(source, "strike", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        return strike if strike > 0 else None

    def _default_opt_row_index(self) -> int:
        rows = self._option_pair_rows()
        if not rows:
            return 0
        strikes: list[float] = []
        for call_contract, put_contract in rows:
            strike = self._option_row_strike(call_contract, put_contract)
            if strike is not None:
                strikes.append(float(strike))
        if not strikes:
            return 0
        target = strikes[len(strikes) // 2]
        best_idx = 0
        best_delta = float("inf")
        for idx, (call_contract, put_contract) in enumerate(rows):
            strike = self._option_row_strike(call_contract, put_contract)
            if strike is None:
                continue
            delta = abs(float(strike) - float(target))
            if delta < best_delta:
                best_delta = delta
                best_idx = idx
        return best_idx

    def _opt_underlyer_label(self) -> str:
        for contract in self._search_results:
            if str(getattr(contract, "secType", "") or "").strip().upper() != "OPT":
                continue
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
            if symbol and exchange:
                return f"{symbol} {exchange}"
            if symbol:
                return symbol
        return ""

    def _search_row_count(self) -> int:
        if self._search_mode() == "OPT":
            return len(self._option_pair_rows())
        return len(self._search_results)

    def _ensure_search_visible(self) -> None:
        total = self._search_row_count()
        if total <= 0:
            self._search_scroll = 0
            return
        max_scroll = max(0, total - self._SEARCH_LIMIT)
        if self._search_selected < self._search_scroll:
            self._search_scroll = self._search_selected
        elif self._search_selected >= (self._search_scroll + self._SEARCH_LIMIT):
            self._search_scroll = self._search_selected - self._SEARCH_LIMIT + 1
        self._search_scroll = min(max(self._search_scroll, 0), max_scroll)

    def _queue_search(self) -> None:
        self._search_error = None
        self._search_selected = 0
        self._search_scroll = 0
        self._search_opt_expiry_index = 0
        self._cancel_search_task()
        query = self._search_query.strip()
        if not query:
            self._search_results = []
            self._search_loading = False
            self._render_search()
            return
        self._search_loading = True
        self._render_search()
        self._search_generation += 1
        generation = self._search_generation
        mode = self._search_mode()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._search_loading = False
            self._render_search()
            return
        self._search_task = loop.create_task(
            self._run_search(
                generation,
                query,
                mode,
                fetch_limit=self._SEARCH_FETCH_LIMIT if mode == "OPT" else self._SEARCH_LIMIT,
            )
        )

    async def _run_search(self, generation: int, query: str, mode: str, *, fetch_limit: int) -> None:
        try:
            await asyncio.sleep(self._SEARCH_DEBOUNCE_SEC)
            results = await self._client.search_contracts(query, mode=mode, limit=fetch_limit)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if generation != self._search_generation:
                return
            self._search_loading = False
            self._search_results = []
            self._search_error = str(exc)
            self._render_search()
            return
        if generation != self._search_generation:
            return
        self._search_loading = False
        self._search_results = list(results)
        total = self._search_row_count()
        if mode == "OPT":
            self._search_selected = self._default_opt_row_index()
        else:
            self._search_selected = min(max(self._search_selected, 0), max(total - 1, 0))
        self._ensure_search_visible()
        self._render_search()

    def _render_search(self) -> None:
        if not hasattr(self, "_search"):
            return
        if not self._search_active:
            self._search.display = False
            return
        self._search.display = True
        line1 = Text("Search ", style="bold")
        for index, mode in enumerate(self._SEARCH_MODES):
            if index > 0:
                line1.append(" ", style="dim")
            if index == self._search_mode_index:
                line1.append(f"[{mode}]", style="bold #0d1117 on #7ab6ff")
            else:
                line1.append(mode, style="dim")
        line1.append("  > ", style="dim")
        if self._search_query:
            line1.append(self._search_query, style="bold white")
        else:
            line1.append("type symbol...", style="dim")
        lines: list[Text] = [line1]
        mode = self._search_mode()
        if mode == "OPT":
            self._sync_search_option_tickers()
        else:
            self._clear_search_tickers()
        if self._search_error:
            lines.append(Text(f"Error: {self._search_error}", style="red"))
        elif self._search_loading:
            lines.append(Text("Searching...", style="yellow"))
        elif not self._search_query.strip():
            lines.append(
                Text(
                    "Tab/Shift+Tab mode | Up/Down scroll | Enter details | Esc close",
                    style="dim",
                )
            )
        elif not self._search_results:
            lines.append(Text("No matches", style="dim"))
        elif mode == "OPT":
            expiries = self._search_opt_expiries()
            expiry_line = Text("Expiry ", style="dim")
            if expiries:
                self._search_opt_expiry_index = min(
                    max(self._search_opt_expiry_index, 0),
                    len(expiries) - 1,
                )
                for idx, expiry in enumerate(expiries):
                    if idx > 0:
                        expiry_line.append(" ", style="dim")
                    if idx == self._search_opt_expiry_index:
                        expiry_line.append(f"[{expiry}]", style="bold #0d1117 on #ffcc84")
                    else:
                        expiry_line.append(expiry, style="bold #ffcc84")
                expiry_line.append("  ([ / ])", style="dim")
            else:
                expiry_line.append("n/a", style="dim")
            lines.append(expiry_line)
            underlyer_label = self._opt_underlyer_label()
            if underlyer_label:
                lines.append(Text(f"Underlyer {underlyer_label}", style="dim"))
            side_line = Text("Side ", style="dim")
            if self._search_side == 0:
                side_line.append("[CALL]", style="bold #0d1117 on #8fbfff")
                side_line.append(" PUT", style="dim")
            else:
                side_line.append("CALL ", style="dim")
                side_line.append("[PUT]", style="bold #0d1117 on #8fbfff")
            side_line.append("  (left/right)", style="dim")
            lines.append(side_line)
            rows = self._option_pair_rows()
            total = len(rows)
            start = min(self._search_scroll, max(0, total - self._SEARCH_LIMIT))
            end = min(start + self._SEARCH_LIMIT, total)
            for idx in range(start, end):
                call_contract, put_contract = rows[idx]
                lines.append(
                    self._search_option_pair_line(
                        idx=idx,
                        call_contract=call_contract,
                        put_contract=put_contract,
                        active=idx == self._search_selected,
                    )
                )
            if total > self._SEARCH_LIMIT:
                lines.append(Text(f"Rows {start + 1}-{end}/{total}", style="dim"))
        else:
            total = len(self._search_results)
            start = min(self._search_scroll, max(0, total - self._SEARCH_LIMIT))
            end = min(start + self._SEARCH_LIMIT, total)
            for idx in range(start, end):
                lines.append(
                    self._search_result_line(
                        self._search_results[idx],
                        row=idx,
                        active=idx == self._search_selected,
                    )
                )
            if total > self._SEARCH_LIMIT:
                lines.append(Text(f"Rows {start + 1}-{end}/{total}", style="dim"))
        self._search.update(Text("\n").join(lines))

    def _search_result_line(self, contract: Contract, *, row: int, active: bool) -> Text:
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "?").strip().upper() or "?"
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, "bold white")
        line = Text(f"{row + 1}. ", style="dim")
        line.append(sec_type.ljust(3), style=style)
        line.append(" ")
        line.append(symbol, style="bold")
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if sec_type in ("OPT", "FOP"):
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            strike = getattr(contract, "strike", None)
            strike_text = ""
            if strike not in (None, ""):
                try:
                    strike_text = f"{float(strike):.2f}"
                except (TypeError, ValueError):
                    strike_text = str(strike)
            if expiry:
                line.append(f" {expiry}", style="dim")
            if right:
                line.append(f" {right}", style="bold")
            if strike_text:
                line.append(f" {strike_text}", style="bold")
        elif sec_type == "FUT" and expiry:
            line.append(f" {expiry}", style="dim")
        exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        if exchange:
            line.append(f"  {exchange}", style="dim")
        if active:
            line.stylize("bold on #1d2a38")
        return line

    @staticmethod
    def _clip_cell(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text.ljust(width)
        if width == 1:
            return "…"
        return text[: width - 1] + "…"

    def _search_option_pair_line(
        self,
        *,
        idx: int,
        call_contract: Contract | None,
        put_contract: Contract | None,
        active: bool,
    ) -> Text:
        strike = self._option_row_strike(call_contract, put_contract)
        strike_text = f"{strike:.2f}" if strike is not None else "--"
        left_style = "bold #8fbfff" if call_contract else "dim"
        right_style = "bold #ff9ac2" if put_contract else "dim"
        if active and self._search_side == 0:
            left_style = f"{left_style} on #1d2a38"
        if active and self._search_side == 1:
            right_style = f"{right_style} on #1d2a38"

        call_text = self._clip_cell(self._search_option_cell_text("C", call_contract), 10)
        put_text = self._clip_cell(self._search_option_cell_text("P", put_contract), 10)
        line = Text(f"{idx + 1}. ", style="dim")
        line.append(call_text, style=left_style)
        line.append("  |  ", style="dim")
        line.append(f"K={strike_text}", style="bold #ffcc84")
        line.append("  |  ", style="dim")
        line.append(put_text, style=right_style)
        return line

    def _open_search_selection(self) -> None:
        if not self._search_results:
            return
        contract: Contract | None = None
        if self._search_mode() == "OPT":
            rows = self._option_pair_rows()
            if not rows:
                return
            index = min(max(self._search_selected, 0), len(rows) - 1)
            call_contract, put_contract = rows[index]
            contract = call_contract if self._search_side == 0 else put_contract
            if contract is None:
                contract = put_contract if self._search_side == 0 else call_contract
        else:
            index = min(max(self._search_selected, 0), len(self._search_results) - 1)
            contract = self._search_results[index]
        if contract is None:
            return
        item = self._portfolio_item_for_contract(contract)
        self._close_search()
        self.push_screen(
            PositionDetailScreen(
                self._client,
                item,
                self._config.detail_refresh_sec,
            )
        )

    def _portfolio_item_for_contract(self, contract: Contract) -> PortfolioItem:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            for item in self._snapshot.items:
                if int(getattr(getattr(item, "contract", None), "conId", 0) or 0) == con_id:
                    return item
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        for item in self._snapshot.items:
            existing = getattr(item, "contract", None)
            if existing is None:
                continue
            if str(getattr(existing, "secType", "") or "").strip().upper() != sec_type:
                continue
            if str(getattr(existing, "symbol", "") or "").strip().upper() != symbol:
                continue
            return item
        return cast(PortfolioItem, _SyntheticPortfolioItem(contract=contract))

    async def refresh_positions(self, hard: bool = False) -> None:
        if self._refresh_lock.locked():
            self._dirty = True
            return
        async with self._refresh_lock:
            try:
                if hard:
                    await self._client.hard_refresh()
                items = await self._client.fetch_portfolio()
                self._snapshot.update(items, None)
            except Exception as exc:  # pragma: no cover - UI surface
                self._snapshot.update([], str(exc))
            self._sync_session_tickers()
            self._client.start_index_tickers()
            self._client.start_proxy_tickers()
            self._index_tickers = self._client.index_tickers()
            self._index_error = self._client.index_error()
            self._proxy_tickers = self._client.proxy_tickers()
            self._proxy_error = self._client.proxy_error()
            self._pnl = self._client.pnl()
            self._net_liq = self._client.account_value("NetLiquidation")
            self._maybe_update_netliq_anchor()
            self._buying_power = self._client.account_value("BuyingPower")
            self._maybe_update_buying_power_anchor()
            self._prime_change_data(self._snapshot.items)
            self._render_table()

    def _mark_dirty(self) -> None:
        self._dirty = True
        if self._dirty_task and not self._dirty_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._dirty_task = loop.create_task(self._flush_dirty())

    async def _flush_dirty(self) -> None:
        while self._dirty:
            self._dirty = False
            await self.refresh_positions()
            if not self._dirty:
                break
            await asyncio.sleep(self._config.refresh_sec)

    def _render_table(self) -> None:
        prev_coord = self._table.cursor_coordinate
        prev_row_key = None
        if 0 <= prev_coord.row < len(self._row_keys):
            prev_row_key = self._row_keys[prev_coord.row]
        prev_row_index = prev_coord.row
        prev_column = prev_coord.column

        self._table.clear()
        self._row_keys = []
        self._row_item_by_key = {}
        items = list(self._snapshot.items)
        section_rows_by_type = self._section_rows_by_type(items)
        self._row_count = sum(len(rows) for rows in section_rows_by_type.values())
        for title, sec_type in _SECTION_ORDER:
            rows = section_rows_by_type.get(sec_type, [])
            if not rows:
                continue
            self._add_section(title, sec_type, rows)
        if self._buying_power[0] is not None:
            spacer_key = "spacer:buying_power"
            self._table.add_row(
                *(Text("") for _ in range(self._column_count)),
                key=spacer_key,
            )
            self._row_keys.append(spacer_key)
            self._add_buying_power_row(self._buying_power, self._pnl)
        self._add_total_row(items)
        self._add_daily_row(self._pnl)
        self._add_net_liq_row(self._net_liq, self._pnl)
        self._render_ticker_bar()
        self._status.update(self._status_text())
        self._restore_cursor(prev_row_key, prev_row_index, prev_column)

    def _maybe_update_netliq_anchor(self) -> None:
        value, _currency, updated_at = self._net_liq
        if value is None:
            return
        daily = _pnl_value(self._pnl)
        if daily is None:
            return
        if self._net_liq_daily_anchor is None:
            self._net_liq_daily_anchor = daily
        if updated_at is None:
            return
        if not hasattr(self, "_net_liq_updated_at") or self._net_liq_updated_at != updated_at:
            self._net_liq_daily_anchor = daily
            self._net_liq_updated_at = updated_at

    def _maybe_update_buying_power_anchor(self) -> None:
        value, _currency, updated_at = self._buying_power
        if value is None:
            return
        daily = _pnl_value(self._pnl)
        if daily is None:
            return
        if self._buying_power_daily_anchor is None:
            self._buying_power_daily_anchor = daily
        if updated_at is None:
            return
        if (
            not hasattr(self, "_buying_power_updated_at")
            or self._buying_power_updated_at != updated_at
        ):
            self._buying_power_daily_anchor = daily
            self._buying_power_updated_at = updated_at

    def _sync_session_tickers(self) -> None:
        session = _market_session_label()
        if session == self._md_session:
            return
        self._md_session = session
        self._ticker_con_ids.clear()
        self._ticker_loading.clear()
        self._quote_signature_by_con_id.clear()
        self._quote_updated_mono_by_con_id.clear()

    def _status_text(self) -> str:
        conn = self._client.connection_state()
        updated = self._snapshot.updated_at
        if updated:
            ts = updated.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts = "n/a"
        session = _market_session_label()
        base = f"IBKR {conn} | last update: {ts} | rows: {self._row_count}"
        base = f"{base} | MKT: {session} | MD: [L]=Live [D]=Delayed"
        base = f"{base} | PnL rows=position(U/R), daily=account day move"
        if self._snapshot.error:
            return f"{base} | error: {self._snapshot.error}"
        return base

    def _section_rows_by_type(self, items: list[PortfolioItem]) -> dict[str, list[PortfolioItem]]:
        rows_by_type: dict[str, list[PortfolioItem]] = {sec_type: [] for _, sec_type in _SECTION_ORDER}
        for item in items:
            sec_type = str(getattr(item.contract, "secType", "") or "").strip().upper()
            if sec_type in rows_by_type:
                rows_by_type[sec_type].append(item)
        for rows in rows_by_type.values():
            rows.sort(key=_portfolio_sort_key, reverse=True)
        return rows_by_type

    def _add_section(self, title: str, sec_type: str, rows: list[PortfolioItem]) -> None:
        if not rows:
            return
        header_key = f"header:{sec_type}"
        self._table.add_row(
            *self._section_header_row(title, sec_type),
            key=header_key,
        )
        self._row_keys.append(header_key)
        for item in rows:
            row_key = f"{sec_type}:{item.contract.conId}"
            change_text = self._contract_change_text(item)
            unreal_text, unreal_pct_text = self._unreal_texts(item)
            row_values = _portfolio_row(
                item,
                change_text,
                unreal_text=unreal_text,
                unreal_pct_text=unreal_pct_text,
            )
            row_values[1] = self._center_cell(row_values[1], self._QTY_COL_WIDTH)
            row_values[4] = self._center_cell(row_values[4], self._UNREAL_COL_WIDTH)
            row_values[5] = self._center_cell(row_values[5], self._REALIZED_COL_WIDTH)
            self._table.add_row(
                *row_values,
                key=row_key,
            )
            self._row_keys.append(row_key)
            self._row_item_by_key[row_key] = item

    def _section_header_row(self, title: str, sec_type: str) -> list[Text]:
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, "bold white on #2b2b2b")
        row = [Text(f"— {title} —", style=style)]
        row.extend(Text("", style=style) for _ in range(self._column_count - 1))
        return row

    def _add_total_row(self, items: list[PortfolioItem]) -> None:
        total_unreal = 0.0
        total_real = 0.0
        total_cost = 0.0
        total_mkt = 0.0
        for item in items:
            if item.contract.secType not in _SECTION_TYPES:
                continue
            mark_price, _ = self._mark_price(item)
            unreal, _ = _unrealized_pnl_values(item, mark_price=mark_price)
            if unreal is not None:
                total_unreal += float(unreal)
            total_real += float(item.realizedPNL or 0.0)
            if item.averageCost:
                total_cost += abs(float(item.averageCost) * float(item.position))
            if item.marketValue:
                total_mkt += abs(float(item.marketValue))
        total_pnl = total_unreal + total_real
        denom = total_cost if total_cost > 0 else total_mkt
        pct = (total_pnl / denom * 100.0) if denom > 0 else None
        style = "bold white on #1f1f1f"
        label = Text("TOTAL (U+R)", style=style)
        blank = Text("")
        pnl_text = _pnl_text(total_pnl)
        pct_text = _pnl_pct_value(pct)
        unreal_text = _combined_value_pct(pnl_text, pct_text)
        unreal_text = self._center_cell(unreal_text, self._UNREAL_COL_WIDTH)
        realized_text = _pnl_text(total_real)
        realized_text = self._center_cell(realized_text, self._REALIZED_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            unreal_text,
            realized_text,
            key="total",
        )
        self._row_keys.append("total")

    def _add_daily_row(self, pnl: PnL | None) -> None:
        if not pnl:
            return
        daily = pnl.dailyPnL
        if daily is None or (isinstance(daily, float) and math.isnan(daily)):
            return
        style = "bold white on #1f1f1f"
        label = Text("DAILY P&L", style=style)
        blank = Text("")
        daily_text = _pnl_text(float(daily))
        daily_text = self._center_cell(daily_text, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            daily_text,
            blank,
            key="daily",
        )
        self._row_keys.append("daily")

    def _add_net_liq_row(
        self, net_liq: tuple[float | None, str | None, datetime | None], pnl: PnL | None
    ) -> None:
        value, _currency, _updated_at = net_liq
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text("NET LIQ", style=style)
        blank = Text("")
        est_value = _estimate_net_liq(value, pnl, self._net_liq_daily_anchor)
        shown_value = est_value if est_value is not None else value
        unreal_cell = Text(_fmt_money(shown_value))
        amount_text = self._center_cell(unreal_cell, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            amount_text,
            blank,
            key="netliq",
        )
        self._row_keys.append("netliq")

    def _add_buying_power_row(
        self,
        buying_power: tuple[float | None, str | None, datetime | None],
        pnl: PnL | None,
    ) -> None:
        value, _currency, _updated_at = buying_power
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text("BUYING POWER", style=style)
        blank = Text("")
        est_value = _estimate_buying_power(value, pnl, self._buying_power_daily_anchor)
        shown_value = est_value if est_value is not None else value
        unreal_cell = Text(_fmt_money(shown_value))
        amount_text = self._center_cell(unreal_cell, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            amount_text,
            blank,
            key="buying_power",
        )
        self._row_keys.append("buying_power")

    def _prime_change_data(self, items: list[PortfolioItem]) -> None:
        for item in items:
            if item.contract.secType not in _SECTION_TYPES:
                continue
            contract = item.contract
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._start_ticker_load(con_id, contract)
                self._start_closes_load(con_id, contract)
            if contract.secType in ("OPT", "FOP") and con_id:
                self._start_underlying_load(con_id, contract)

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
            await self._client.ensure_ticker(contract, owner="positions")
            self._ticker_con_ids.add(con_id)
        except Exception:
            return
        finally:
            self._ticker_loading.discard(con_id)
        self._mark_dirty()

    def _start_closes_load(self, con_id: int, contract: Contract) -> None:
        cached = self._session_closes_by_con_id.get(con_id)
        if cached is not None and cached[1] is not None:
            return
        if con_id in self._closes_loading:
            return
        if cached is not None:
            now = time.monotonic()
            retry_at = self._closes_retry_at_by_con_id.get(con_id, 0.0)
            if now < retry_at:
                return
            self._closes_retry_at_by_con_id[con_id] = now + self._CLOSES_RETRY_SEC
        else:
            self._closes_retry_at_by_con_id.pop(con_id, None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._closes_loading.add(con_id)
        loop.create_task(self._load_closes(con_id, contract))

    async def _load_closes(self, con_id: int, contract: Contract) -> None:
        try:
            prev_close, close_3ago = await self._client.session_closes(contract)
            self._session_closes_by_con_id[con_id] = (prev_close, close_3ago)
            if close_3ago is not None:
                self._closes_retry_at_by_con_id.pop(con_id, None)
            else:
                self._closes_retry_at_by_con_id[con_id] = (
                    time.monotonic() + self._CLOSES_RETRY_SEC
                )
        except Exception:
            self._closes_retry_at_by_con_id[con_id] = time.monotonic() + self._CLOSES_RETRY_SEC
            return
        finally:
            self._closes_loading.discard(con_id)
        self._mark_dirty()

    def _start_underlying_load(self, option_con_id: int, contract: Contract) -> None:
        if option_con_id in self._option_underlying_con_id or option_con_id in self._underlying_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._underlying_loading.add(option_con_id)
        loop.create_task(self._load_underlying(option_con_id, contract))

    async def _load_underlying(self, option_con_id: int, contract: Contract) -> None:
        try:
            underlying = await self._client.resolve_underlying_contract(contract)
            if not underlying:
                return
            under_con_id = int(getattr(underlying, "conId", 0) or 0)
            if not under_con_id:
                return
            self._option_underlying_con_id[option_con_id] = under_con_id
            self._start_ticker_load(under_con_id, underlying)
            self._start_closes_load(under_con_id, underlying)
        except Exception:
            return
        finally:
            self._underlying_loading.discard(option_con_id)
        self._mark_dirty()

    def _contract_change_text(self, item: PortfolioItem) -> Text:
        contract = item.contract
        con_id = int(getattr(contract, "conId", 0) or 0)
        return self._change_text_for_con_id(con_id, item=item)

    def _change_text_for_con_id(self, con_id: int, *, item: PortfolioItem | None = None) -> Text:
        if not con_id:
            return Text("")
        ticker = self._client.ticker_for_con_id(con_id)
        bid = _safe_num(getattr(ticker, "bid", None)) if ticker else None
        ask = _safe_num(getattr(ticker, "ask", None)) if ticker else None
        last = _safe_num(getattr(ticker, "last", None)) if ticker else None
        has_live_quote = bool(
            (bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask)
            or (last is not None and last > 0)
        )
        quote_price = _ticker_price(ticker) if ticker else None
        price = quote_price
        if item is not None:
            sec_type = str(getattr(item.contract, "secType", "") or "").strip().upper()
            if sec_type in ("OPT", "FOP"):
                display_price = _option_display_price(item, ticker)
                if display_price is not None:
                    price = float(display_price)
        cached = self._session_closes_by_con_id.get(con_id)
        ticker_close = _ticker_close(ticker) if ticker else None
        if (not has_live_quote) and cached and cached[0] is not None:
            prev_close = cached[0]
        else:
            prev_close = ticker_close if ticker_close is not None else (cached[0] if cached else None)
        close_3ago = cached[1] if cached else None
        pct24 = _pct_change(price, prev_close)
        pct72 = _pct_change(price, close_3ago)
        text = _price_pct_dual_text(
            price,
            pct24,
            pct72,
            separator="·",
        )
        glyph = self._price_direction_glyph(pct24, pct72)
        if glyph.plain:
            with_glyph = Text()
            with_glyph.append_text(glyph)
            if text.plain:
                with_glyph.append(" ")
            with_glyph.append_text(text)
            text = with_glyph
        age_sec = self._quote_age_seconds(con_id, ticker, price if price is not None else quote_price)
        ribbon = self._quote_age_ribbon(age_sec)
        if ribbon.plain:
            if text.plain:
                text.append(" ", style="dim")
            text.append_text(ribbon)
        centered = self._center_cell(text, self._PX_24_72_COL_WIDTH)
        return centered if isinstance(centered, Text) else Text(str(centered))

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
        if not con_id or not ticker:
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

    def _unreal_texts(self, item: PortfolioItem) -> tuple[Text, Text]:
        contract = item.contract
        if contract.secType not in _SECTION_TYPES:
            return Text(""), Text("")
        mark_price, is_estimate = self._mark_price(item)
        pnl, pct = _unrealized_pnl_values(item, mark_price=mark_price)
        raw_unreal = _safe_num(getattr(item, "unrealizedPNL", None))
        is_modeled = (
            raw_unreal is None
            and is_estimate
            and mark_price is not None
            and pnl is not None
        )
        prefix = "~" if is_modeled else ""
        return _pnl_text(pnl, prefix=prefix), _pnl_pct_value(pct)

    def _mark_price(self, item: PortfolioItem) -> tuple[float | None, bool]:
        contract = item.contract
        con_id = int(getattr(contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id)
        if contract.secType == "OPT" or contract.secType == "FOP":
            return self._option_estimated_mark(item, ticker)
        price = _ticker_price(ticker) if ticker else None
        if price is None:
            try:
                price = float(item.marketPrice)
            except (TypeError, ValueError):
                price = None
        return price, False

    def _option_estimated_mark(
        self, item: PortfolioItem, option_ticker: Ticker | None
    ) -> tuple[float | None, bool]:
        bid = _safe_num(getattr(option_ticker, "bid", None)) if option_ticker else None
        ask = _safe_num(getattr(option_ticker, "ask", None)) if option_ticker else None
        portfolio_mark = _safe_num(getattr(item, "marketPrice", None))
        direct_price = None
        if (
            bid is not None
            and ask is not None
            and bid > 0
            and ask > 0
            and bid <= ask
        ):
            direct_price = (float(bid) + float(ask)) / 2.0
        else:
            last = _safe_num(getattr(option_ticker, "last", None)) if option_ticker else None
            if last is not None and last > 0:
                direct_price = float(last)
        if direct_price is not None:
            return direct_price, False
        if item.contract.secType == "FOP":
            if portfolio_mark is not None and portfolio_mark > 0:
                return float(portfolio_mark), False
            return None, False

        option_con_id = int(getattr(item.contract, "conId", 0) or 0)
        under_con_id = self._option_underlying_con_id.get(option_con_id)
        under_ticker = self._client.ticker_for_con_id(under_con_id) if under_con_id else None
        under_price = _ticker_price(under_ticker) if under_ticker else None
        model = getattr(option_ticker, "modelGreeks", None) if option_ticker else None
        delta = _safe_num(getattr(model, "delta", None)) if model else None
        gamma = _safe_num(getattr(model, "gamma", None)) if model else None
        under_close = _ticker_close(under_ticker) if under_ticker else None
        if under_close is None and under_con_id:
            cached = self._session_closes_by_con_id.get(under_con_id)
            under_close = cached[0] if cached else None
        ref_price = _safe_num(getattr(option_ticker, "close", None)) if option_ticker else None
        if ref_price is None:
            ref_price = portfolio_mark
        if (
            under_price is not None
            and under_price > 0
            and delta is not None
            and under_close is not None
            and under_close > 0
            and ref_price is not None
        ):
            d_under = float(under_price) - float(under_close)
            estimated = float(ref_price) + (float(delta) * d_under)
            if gamma is not None:
                estimated += 0.5 * float(gamma) * (d_under**2)
            return max(estimated, 0.0), True
        model_price = _safe_num(getattr(model, "optPrice", None)) if model else None
        if model_price is not None and model_price > 0:
            return float(model_price), True
        if ref_price is not None and ref_price > 0:
            return float(ref_price), True
        return None, False
    def _render_ticker_bar(self) -> None:
        prefix = f"MKT:{_market_session_label()} | "
        line1 = _ticker_line(
            _INDEX_ORDER,
            _INDEX_LABELS,
            self._index_tickers,
            self._index_error,
            prefix,
        )
        line2 = _ticker_line(
            _PROXY_ORDER,
            _PROXY_LABELS,
            self._proxy_tickers,
            self._proxy_error,
            " " * len(prefix),
        )
        text = Text()
        text.append_text(line1)
        text.append("\n")
        text.append_text(line2)
        self._ticker.update(text)

    def _restore_cursor(self, row_key: str | None, row_index: int, column: int) -> None:
        if not self._row_keys:
            return
        if row_key and row_key in self._row_keys:
            target_row = self._row_keys.index(row_key)
        else:
            target_row = min(max(row_index, 0), len(self._row_keys) - 1)
        target_col = min(max(column, 0), max(self._column_count - 1, 0))
        self._table.cursor_coordinate = (target_row, target_col)
