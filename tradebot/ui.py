"""Minimal TUI for displaying current portfolio positions."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ib_insync import PnL, PortfolioItem, Ticker
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static

from .client import IBKRClient
from .config import load_config
from .store import PortfolioSnapshot


def _fmt_expiry(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    if len(raw) == 6 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}"
    return raw


def _fmt_qty(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


class PositionsApp(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
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
    }

    #status {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._client = IBKRClient(self._config)
        self._snapshot = PortfolioSnapshot()
        self._refresh_lock = asyncio.Lock()
        self._dirty = False
        self._dirty_task: asyncio.Task | None = None
        self._row_count = 0
        self._row_keys: list[str] = []
        self._column_count = 0
        self._index_tickers: dict[str, Ticker] = {}
        self._index_error: str | None = None
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_error: str | None = None
        self._pnl: PnL | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="ticker")
        yield DataTable(id="positions", zebra_stripes=True)
        yield Static("Starting...", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self._table = self.query_one(DataTable)
        self._ticker = self.query_one("#ticker", Static)
        self._status = self.query_one("#status", Static)
        self._setup_columns()
        self._table.focus()
        self._client.set_update_callback(self._mark_dirty)
        await self.refresh_positions()

    async def on_unmount(self) -> None:
        await self._client.disconnect()

    def _setup_columns(self) -> None:
        self._columns = [
            "Symbol",
            "Expiry",
            "Right",
            "Strike",
            "Qty",
            "AvgCost",
            "Unreal",
            "Unreal%",
            "Realized",
        ]
        self._column_count = len(self._columns)
        self._table.add_columns(*self._columns)

    async def action_refresh(self) -> None:
        await self.refresh_positions(hard=True)

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
            self._client.start_index_tickers()
            self._client.start_proxy_tickers()
            self._index_tickers = self._client.index_tickers()
            self._index_error = self._client.index_error()
            self._proxy_tickers = self._client.proxy_tickers()
            self._proxy_error = self._client.proxy_error()
            self._pnl = self._client.pnl()
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
        while True:
            await asyncio.sleep(self._config.refresh_sec)
            if not self._dirty:
                break
            self._dirty = False
            await self.refresh_positions()

    def _render_table(self) -> None:
        prev_coord = self._table.cursor_coordinate
        prev_row_key = None
        if 0 <= prev_coord.row < len(self._row_keys):
            prev_row_key = self._row_keys[prev_coord.row]
        prev_row_index = prev_coord.row
        prev_column = prev_coord.column

        self._table.clear()
        self._row_keys = []
        items = list(self._snapshot.items)
        self._row_count = sum(
            1 for item in items if item.contract.secType in _SECTION_TYPES
        )
        for title, sec_type in _SECTION_ORDER:
            self._add_section(title, sec_type, items)
        self._add_total_row(items)
        self._add_daily_row(self._pnl)
        self._render_ticker_bar()
        self._status.update(self._status_text())
        self._restore_cursor(prev_row_key, prev_row_index, prev_column)

    def _status_text(self) -> str:
        conn = "connected" if self._client.is_connected else "disconnected"
        updated = self._snapshot.updated_at
        if updated:
            ts = updated.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts = "n/a"
        session = _market_session_label()
        base = f"IBKR {conn} | last update: {ts} | rows: {self._row_count}"
        base = f"{base} | MKT: {session} | MD: [L]=Live [D]=Delayed"
        if self._snapshot.error:
            return f"{base} | error: {self._snapshot.error}"
        return base

    def _add_section(self, title: str, sec_type: str, items: list[PortfolioItem]) -> None:
        header_key = f"header:{sec_type}"
        self._table.add_row(*self._section_header_row(title), key=header_key)
        self._row_keys.append(header_key)
        rows = [item for item in items if item.contract.secType == sec_type]
        rows.sort(key=_portfolio_sort_key, reverse=True)
        for item in rows:
            row_key = f"{sec_type}:{item.contract.conId}"
            self._table.add_row(*_portfolio_row(item), key=row_key)
            self._row_keys.append(row_key)

    def _section_header_row(self, title: str) -> list[Text]:
        style = "bold white on #2b2b2b"
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
            total_unreal += float(item.unrealizedPNL or 0.0)
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
        realized_text = _pnl_text(total_real)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            pnl_text,
            pct_text,
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
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            daily_text,
            blank,
            blank,
            key="daily",
        )
        self._row_keys.append("daily")
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


def _portfolio_sort_key(item: PortfolioItem) -> float:
    unreal = float(item.unrealizedPNL or 0.0)
    realized = float(item.realizedPNL or 0.0)
    return unreal + realized


def _portfolio_row(item: PortfolioItem) -> list[Text | str]:
    contract = item.contract
    expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth or "")
    right = contract.right or ""
    strike = _fmt_money(contract.strike) if contract.strike else ""
    qty = _fmt_qty(float(item.position))
    avg_cost = _fmt_money(float(item.averageCost)) if item.averageCost else ""
    unreal = _pnl_text(item.unrealizedPNL)
    unreal_pct = _pnl_pct_text(item)
    realized = _pnl_text(item.realizedPNL)
    return [
        contract.symbol,
        expiry,
        right,
        strike,
        qty,
        avg_cost,
        unreal,
        unreal_pct,
        realized,
    ]


def _pnl_text(value: float | None) -> Text:
    if value is None:
        return Text("")
    text = _fmt_money(value)
    if value > 0:
        return Text(text, style="green")
    if value < 0:
        return Text(text, style="red")
    return Text(text)


def _pnl_pct_text(item: PortfolioItem) -> Text:
    value = item.unrealizedPNL
    if value is None:
        return Text("")
    cost_basis = 0.0
    if item.averageCost:
        cost_basis = float(item.averageCost) * float(item.position)
    denom = abs(cost_basis) if cost_basis else abs(float(item.marketValue or 0.0))
    if denom <= 0:
        return Text("")
    pct = (float(value) / denom) * 100.0
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


def _pnl_pct_value(pct: float | None) -> Text:
    if pct is None:
        return Text("")
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


_SECTION_ORDER = (
    ("OPTIONS", "OPT"),
    ("STOCKS", "STK"),
    ("FUTURES", "FUT"),
    ("FUTURES OPT", "FOP"),
)
_SECTION_TYPES = {sec_type for _, sec_type in _SECTION_ORDER}

_INDEX_ORDER = ("NQ", "ES", "YM")
_INDEX_LABELS = {
    "NQ": "NASDAQ",
    "ES": "S&P",
    "YM": "DOW",
}
_PROXY_ORDER = ("QQQ", "TQQQ")
_PROXY_LABELS = {
    "QQQ": "QQQ",
    "TQQQ": "TQQQ",
}
_TICKER_WIDTHS = {
    "label": 10,
    "price": 9,
    "change": 9,
    "pct": 10,
}


def _ticker_price(ticker: Ticker) -> float | None:
    try:
        value = float(ticker.marketPrice())
    except Exception:
        return None
    if not value or math.isnan(value):
        value = float(ticker.last or 0.0)
    if not value or math.isnan(value):
        return None
    return value


def _ticker_close(ticker: Ticker) -> float | None:
    for attr in ("close", "prevLast"):
        value = getattr(ticker, attr, None)
        if value:
            return float(value)
    return None


def _market_data_tag(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return " [L]"
    if md_type in (3, 4):
        return " [D]"
    return ""


def _market_session_label() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York")).time()
    if time(4, 0) <= now_et < time(9, 30):
        return "PRE"
    if time(9, 30) <= now_et < time(16, 0):
        return "MRKT"
    if time(16, 0) <= now_et < time(20, 0):
        return "POST"
    return "OVRNGHT"


def _ticker_line(
    order: tuple[str, ...],
    labels: dict[str, str],
    tickers: dict[str, Ticker],
    error: str | None,
    prefix: str,
) -> Text:
    if error:
        return Text(f"{prefix}Data error: {error}", style="red")
    text = Text()
    if prefix:
        text.append(prefix)
    for idx, symbol in enumerate(order):
        if idx:
            text.append(" | ", style="dim")
        label = labels[symbol]
        ticker = tickers.get(symbol)
        if not ticker:
            text.append_text(_ticker_missing(label))
            continue
        tag = _market_data_tag(ticker)
        price = _ticker_price(ticker)
        close = _ticker_close(ticker)
        if price is None or close is None or price <= 0 or close <= 0:
            if close and close > 0:
                text.append_text(_ticker_closed(label + tag, close))
            else:
                text.append_text(_ticker_missing(label + tag))
            continue
        change = price - close
        pct = (change / close) * 100.0
        style = "green" if change > 0 else "red" if change < 0 else ""
        text.append_text(_ticker_block(label + tag, price, change, pct, style))
    return text


def _ticker_block(label: str, price: float, change: float, pct: float, style: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    change_text = f"{change:+.2f}".rjust(_TICKER_WIDTHS["change"])
    pct_text = f"({pct:+.2f}%)".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(change_text, style=style)
    text.append(" ")
    text.append(pct_text, style=style)
    return text


def _ticker_missing(label: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "n/a".rjust(_TICKER_WIDTHS["price"])
    blank_change = "".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text, style="dim")
    text.append(" ", style="dim")
    text.append(price_text, style="dim")
    text.append(" ", style="dim")
    text.append(blank_change, style="dim")
    text.append(" ", style="dim")
    text.append(blank_pct, style="dim")
    return text


def _ticker_closed(label: str, last_price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "Closed".rjust(_TICKER_WIDTHS["price"])
    last_text = f"({last_price:,.2f})".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text, style="red")
    text.append(" ")
    text.append(last_text, style="red")
    text.append(" ")
    text.append(blank_pct, style="red")
    return text
