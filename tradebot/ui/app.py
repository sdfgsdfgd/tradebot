"""Portfolio TUI (positions) + bot hub entrypoint."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from ib_insync import Contract, PnL, PortfolioItem, Ticker
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from ..config import load_config
from .bot_runtime import BotRuntime
from .favorites import FavoritesScreen
from .portfolio.market import PortfolioMarketValues
from .portfolio.search_runtime import PortfolioSearchRuntime
from .portfolio.search_state import PortfolioSearchState
from .portfolio.table import PortfolioTable
from .positions import PositionDetailScreen
from .store import PortfolioSnapshot


class _SearchDrawer(Static):
    can_focus = True


# region Positions UI
class PositionsApp(
    PortfolioSearchState,
    PortfolioSearchRuntime,
    PortfolioTable,
    PortfolioMarketValues,
    App,
):
    _PX_24_72_COL_WIDTH = 27
    _QTY_COL_WIDTH = 7
    _AVG_COL_WIDTH = 23
    _UNREAL_COL_WIDTH = 36
    _REALIZED_COL_WIDTH = 14
    _CLOSES_RETRY_SEC = 30.0
    _SEARCH_MODES = ("STK", "FUT", "OPT", "FOP")
    _SEARCH_LIMIT = 5
    _SEARCH_FETCH_LIMIT = 96
    _SEARCH_OPT_FETCH_LIMIT = 48
    _SEARCH_OPT_FIRST_PAINT_LIMIT = 12
    _SEARCH_OPT_UNDERLYER_LIMIT = 8
    _SEARCH_DEBOUNCE_SEC = 0.18
    _SEARCH_EXPIRY_IDLE_PREFETCH_SEC = 0.35
    _SNAPSHOT_THROTTLE_MIN_SEC = 1.0
    _DERIVATIVE_MARK_STICKY_SEC = 40.0

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
        height: 3;
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
        self._dirty_needs_snapshot = False
        self._dirty_task: asyncio.Task | None = None
        self._last_snapshot_fetch_mono: float = 0.0
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
        self._session_close_1ago_by_con_id: dict[int, float | None] = {}
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
        self._buying_power: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._buying_power_daily_anchor: float | None = None
        self._today_ibkr_day: date | None = None
        self._today_ibkr_last_value: float | None = None
        self._derivative_actionable_px_by_con_id: dict[int, tuple[float, float]] = {}
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
        self._search_side = 0  # 0=CALL(left), 1=PUT(right) for OPT/FOP modes
        self._search_opt_expiry_index = 0
        self._search_loading = False
        self._search_error: str | None = None
        self._search_generation = 0
        self._search_ticker_con_ids: set[int] = set()
        self._search_ticker_loading: set[int] = set()
        self._search_opt_underlyers: list[str] = []
        self._search_opt_underlyer_descriptions: dict[str, str] = {}
        self._search_symbol_labels: dict[str, str] = {}
        self._search_opt_underlyer_index = 0
        self._search_opt_chain_cache: dict[str, list[Contract]] = {}
        self._search_opt_chain_page_cache: dict[str, dict[str, object]] = {}
        self._search_expiry_has_more = False
        self._search_expiry_next_offset = 0
        self._search_expiry_total = 0
        self._search_expiry_loading_more = False
        self._search_expiry_auto_advance_from: int | None = None
        self._search_expiry_prefetch_task: asyncio.Task | None = None
        self._search_expiry_prefetch_generation = -1
        self._search_timing: dict[str, object] = {}
        self._search_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="ticker")
        yield DataTable(
            id="positions",
            zebra_stripes=True,
            show_row_labels=False,
            cell_padding=0,
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
        self._client.set_update_callback(self._mark_stream_dirty)
        self._mark_dirty(fetch_snapshot=True)

    async def on_unmount(self) -> None:
        if self._dirty_task and not self._dirty_task.done():
            self._dirty_task.cancel()
        self._cancel_search_task()
        self._clear_search_tickers()
        await self._client.disconnect()

    def _setup_columns(self) -> None:
        self._columns = [
            "Symbol",
            "Qty",
            "Entry¦Now",
            "Px 24-72",
            "PnL D|U (Pos, IBKR)",
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
            elif "¦" in label:
                self._table.add_column(
                    label.center(self._AVG_COL_WIDTH),
                    width=self._AVG_COL_WIDTH,
                )
            elif label.startswith("Unreal") or label.startswith("PnL D|U"):
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

    @staticmethod
    def _center_with_sep_bias(
        value: str,
        width: int,
        *,
        max_left_gap: int | None = None,
        max_right_gap: int | None = None,
    ) -> str:
        segment = value.center(width)
        if max_right_gap is not None:
            right_pad = len(segment) - len(segment.rstrip(" "))
            if right_pad > max_right_gap:
                shift = right_pad - max_right_gap
                segment = (" " * shift) + segment[:-shift]
        if max_left_gap is not None:
            left_pad = len(segment) - len(segment.lstrip(" "))
            if left_pad > max_left_gap:
                shift = left_pad - max_left_gap
                segment = segment[shift:] + (" " * shift)
        return segment

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
                if self._search_option_sec_type() is not None:
                    self._search_side = 0
                    self._render_search()
                    event.prevent_default()
                    event.stop()
                    return
            if event.key in ("right", "l"):
                if self._search_option_sec_type() is not None:
                    self._search_side = 1
                    self._render_search()
                    event.prevent_default()
                    event.stop()
                    return
            if event.key in ("ctrl+left",):
                if self._search_mode() == "OPT":
                    self._cycle_search_opt_underlyer(-1)
                    event.prevent_default()
                    event.stop()
                    return
            if event.key in ("ctrl+right",):
                if self._search_mode() == "OPT":
                    self._cycle_search_opt_underlyer(1)
                    event.prevent_default()
                    event.stop()
                    return
            if event.character in ("[", "]") or event.key in ("left_square_bracket", "right_square_bracket"):
                if self._search_option_sec_type() is not None:
                    bracket_char = event.character
                    if bracket_char not in ("[", "]"):
                        bracket_char = "[" if event.key == "left_square_bracket" else "]"
                    self._cycle_search_expiry(-1 if bracket_char == "[" else 1)
                    event.prevent_default()
                    event.stop()
                    return
            if (
                event.character in ("<", ">")
                or event.key in ("less_than", "greater_than")
                or (event.character is None and event.key in ("comma", "full_stop", "period"))
            ):
                if self._search_mode() == "OPT":
                    angle_char = event.character
                    if angle_char not in ("<", ">"):
                        if event.key in ("less_than", "comma"):
                            angle_char = "<"
                        else:
                            angle_char = ">"
                    self._cycle_search_opt_underlyer(-1 if angle_char == "<" else 1)
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
