"""Minimal TUI for displaying current portfolio positions."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ib_insync import PnL, PortfolioItem, Ticker, Trade
from rich.text import Text
from textual.app import App, ComposeResult
from textual import events
from textual.containers import Horizontal
from textual.screen import Screen
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
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
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

    DataTable > .datatable--cursor {
        background: #2a2a2a;
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
        self._row_item_by_key: dict[str, PortfolioItem] = {}
        self._column_count = 0
        self._index_tickers: dict[str, Ticker] = {}
        self._index_error: str | None = None
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_error: str | None = None
        self._pnl: PnL | None = None
        self._net_liq: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._net_liq_daily_anchor: float | None = None

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
        self._table.cursor_type = "row"
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

    def action_cursor_down(self) -> None:
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        self._table.action_cursor_up()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        item = self._row_item_by_key.get(event.row_key.value)
        if item:
            self.push_screen(
                PositionDetailScreen(
                    self._client,
                    item,
                    self._config.detail_refresh_sec,
                )
            )

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
            self._net_liq = self._client.account_value("NetLiquidation")
            self._maybe_update_netliq_anchor()
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
        self._row_item_by_key = {}
        items = list(self._snapshot.items)
        self._row_count = sum(
            1 for item in items if item.contract.secType in _SECTION_TYPES
        )
        for title, sec_type in _SECTION_ORDER:
            self._add_section(title, sec_type, items)
        self._add_total_row(items)
        self._add_daily_row(self._pnl)
        self._add_net_liq_row(self._net_liq, self._pnl)
        self._render_ticker_bar()
        self._status.update(self._status_text())
        self._restore_cursor(prev_row_key, prev_row_index, prev_column)

    def _maybe_update_netliq_anchor(self) -> None:
        value, _currency, updated_at = self._net_liq
        if value is None or updated_at is None:
            return
        if not hasattr(self, "_net_liq_updated_at") or self._net_liq_updated_at != updated_at:
            daily = _pnl_value(self._pnl)
            if daily is not None:
                self._net_liq_daily_anchor = daily
            self._net_liq_updated_at = updated_at

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
            self._row_item_by_key[row_key] = item

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

    def _add_net_liq_row(
        self, net_liq: tuple[float | None, str | None, datetime | None], pnl: PnL | None
    ) -> None:
        value, currency, updated_at = net_liq
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text("NET LIQ", style=style)
        blank = Text("")
        amount = _fmt_money(value)
        if currency:
            amount = f"{amount} {currency}"
        amount_text = Text(amount)
        ts_text = Text("")
        if updated_at:
            local_ts = updated_at.astimezone().strftime("%H:%M:%S")
            ts_text = Text(f"@ {local_ts}", style="dim")
        est_text = Text("")
        est_value = _estimate_net_liq(value, pnl, self._net_liq_daily_anchor)
        if est_value is not None:
            est_amount = _fmt_money(est_value)
            if currency:
                est_amount = f"{est_amount} {currency}"
            est_text = Text(f"~{est_amount}", style="yellow")
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            amount_text,
            ts_text,
            est_text,
            key="netliq",
        )
        self._row_keys.append("netliq")
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


class PositionDetailScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("b", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("r", "refresh_ticker", "Refresh MD"),
        ("up", "exec_up", "Exec Up"),
        ("down", "exec_down", "Exec Down"),
        ("j", "exec_down", "Exec Down"),
        ("k", "exec_up", "Exec Up"),
        ("left", "exec_jump_left", "Exec Jump Left"),
        ("right", "exec_jump_right", "Exec Jump Right"),
        ("h", "exec_left", "Exec Left"),
        ("l", "exec_right", "Exec Right"),
        ("c", "cancel_order", "Cancel Order"),
    ]

    def __init__(
        self, client: IBKRClient, item: PortfolioItem, refresh_sec: float
    ) -> None:
        super().__init__()
        self._client = client
        self._item = item
        self._refresh_sec = max(refresh_sec, 0.1)
        self._ticker: Ticker | None = None
        self._underlying_ticker: Ticker | None = None
        self._underlying_con_id: int | None = None
        self._underlying_label: str | None = None
        self._exec_rows = ["mid", "optimistic", "aggressive", "custom", "qty"]
        self._exec_selected = 0
        self._exec_custom_input = ""
        self._exec_qty_input = ""
        self._exec_qty = _default_order_qty(item)
        self._exec_status: str | None = None
        self._active_panel = "exec"
        self._orders_selected = 0
        self._orders_scroll = 0
        self._orders_rows: list[Trade] = []
        self._refresh_task = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Static("", id="detail-left"),
            Static("", id="detail-right"),
            id="detail-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._detail_left = self.query_one("#detail-left", Static)
        self._detail_right = self.query_one("#detail-right", Static)
        self._ticker = await self._client.ensure_ticker(self._item.contract)
        await self._load_underlying()
        self._refresh_task = self.set_interval(self._refresh_sec, self._render_details)
        self._render_details()

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id)
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id)

    def on_key(self, event: events.Key) -> None:
        if event.key == "backspace":
            self._handle_backspace()
            event.stop()
            return
        if event.character:
            if event.character == "B":
                self._submit_order("BUY")
                event.stop()
                return
            if event.character in ("S", "s"):
                self._submit_order("SELL")
                event.stop()
                return
            if event.character in "0123456789.":
                self._handle_digit(event.character)
                event.stop()
                return

    def action_exec_up(self) -> None:
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = max(self._orders_selected - 1, 0)
        else:
            self._exec_selected = max(self._exec_selected - 1, 0)
        self._render_details()

    def action_exec_down(self) -> None:
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = min(
                    self._orders_selected + 1, len(self._orders_rows) - 1
                )
        else:
            self._exec_selected = min(self._exec_selected + 1, len(self._exec_rows) - 1)
        self._render_details()

    def action_exec_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
            self._render_details()
            return
        selected = self._exec_rows[self._exec_selected]
        if selected == "custom":
            self._adjust_custom_price(-1)
        elif selected == "qty":
            self._adjust_qty(-1)
        else:
            self._exec_selected = self._exec_rows.index("custom")
        self._render_details()

    def action_exec_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details()
            return
        selected = self._exec_rows[self._exec_selected]
        if selected == "custom":
            self._adjust_custom_price(1)
        elif selected == "qty":
            self._adjust_qty(1)
        else:
            self._active_panel = "orders"
        self._render_details()

    def action_exec_jump_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
        else:
            self._exec_selected = self._exec_rows.index("custom")
        self._render_details()

    def action_exec_jump_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details()
            return
        self._exec_selected = self._exec_rows.index("mid")
        self._render_details()

    def action_cancel_order(self) -> None:
        if self._active_panel != "orders":
            return
        trade = self._selected_order()
        if not trade:
            self._exec_status = "Cancel: no order"
            self._render_details()
            return
        order_id = trade.order.orderId or trade.order.permId or 0
        self._exec_status = f"Canceling #{order_id}"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Cancel: no loop"
            self._render_details()
            return
        loop.create_task(self._cancel_order(trade))

    async def action_refresh_ticker(self) -> None:
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id)
        self._ticker = await self._client.ensure_ticker(self._item.contract)
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id)
            self._underlying_con_id = None
            self._underlying_ticker = None
            self._underlying_label = None
            await self._load_underlying()
        self._exec_status = "MD refreshed"
        self._render_details()

    async def _load_underlying(self) -> None:
        contract = self._item.contract
        if contract.secType not in ("OPT", "FOP"):
            return
        underlying = await self._client.resolve_underlying_contract(contract)
        if not underlying:
            return
        self._underlying_ticker = await self._client.ensure_ticker(underlying)
        con_id = int(getattr(underlying, "conId", 0) or 0)
        if con_id:
            self._underlying_con_id = con_id
        symbol = getattr(underlying, "localSymbol", "") or getattr(underlying, "symbol", "")
        sec_type = getattr(underlying, "secType", "")
        if symbol or sec_type:
            self._underlying_label = " ".join(part for part in (symbol, sec_type) if part)

    def _render_details(self) -> None:
        try:
            con_id = int(self._item.contract.conId or 0)
            latest = self._client.portfolio_item(con_id) if con_id else None
            if latest:
                self._item = latest
            contract = self._item.contract
            lines: list[Text] = []
            lines.append(Text(f"{contract.symbol} {contract.secType}", style="bold"))
            lines.append(Text(f"ConId: {contract.conId}"))
            if contract.localSymbol:
                lines.append(Text(f"Local: {contract.localSymbol}"))
            if contract.exchange:
                lines.append(Text(f"Exchange: {contract.exchange}"))
            if contract.currency:
                lines.append(Text(f"Currency: {contract.currency}"))
            if contract.lastTradeDateOrContractMonth:
                lines.append(
                    Text(f"Expiry: {_fmt_expiry(contract.lastTradeDateOrContractMonth)}")
                )
            if contract.right:
                lines.append(Text(f"Right: {contract.right}"))
            if contract.strike:
                lines.append(Text(f"Strike: {_fmt_money(contract.strike)}"))
            if contract.multiplier:
                lines.append(Text(f"Contract Size: {contract.multiplier}"))
            lines.append(Text(""))
            lines.append(Text(f"Position: {_fmt_qty(float(self._item.position))}"))
            if self._item.averageCost:
                lines.append(Text(f"Avg Cost: {_fmt_money(float(self._item.averageCost))}"))
            if self._item.marketValue is not None:
                lines.append(Text(f"Market Value: {_fmt_money(float(self._item.marketValue))}"))
            if self._item.unrealizedPNL is not None:
                lines.append(
                    Text(f"Unrealized P&L: {_fmt_money(float(self._item.unrealizedPNL))}")
                )
            if self._item.realizedPNL is not None:
                lines.append(
                    Text(f"Realized P&L: {_fmt_money(float(self._item.realizedPNL))}")
                )
            lines.append(Text(""))
            if self._ticker:
                md_exchange = getattr(self._ticker.contract, "exchange", "") or ""
                md_label = _market_data_label(self._ticker)
                lines.append(
                    Text(
                        f"MD: {md_exchange or 'n/a'} ({md_label})",
                        style="dim",
                    )
                )
                md_local = getattr(self._ticker.contract, "localSymbol", "") or ""
                md_debug = f"MD Contract: {self._ticker.contract.secType}"
                if md_local:
                    md_debug += f" {md_local}"
                lines.append(Text(md_debug, style="dim"))
                bid = _safe_num(self._ticker.bid)
                ask = _safe_num(self._ticker.ask)
                last = _safe_num(self._ticker.last)
                price = _ticker_price(self._ticker)
                mid = _midpoint(bid, ask)
                close = _ticker_close(self._ticker)
                mark = _mark_price(self._item)
                lines.append(
                    Text(
                        f"Bid: {_fmt_quote(bid)}  Ask: {_fmt_quote(ask)}  Last: {_fmt_quote(last)}"
                    )
                )
                if price is None and close is not None:
                    lines.append(Text(f"Price: Closed ({_fmt_quote(close)})", style="red"))
                elif price is None and mark is not None:
                    lines.append(Text(f"Price: Mark ({_fmt_quote(mark)})", style="yellow"))
                else:
                    lines.append(Text(f"Price: {_fmt_quote(price)}"))
            if self._underlying_ticker:
                lines.append(Text(""))
                label = self._underlying_label or "Underlying"
                lines.append(Text(f"{label}", style="bold"))
                md_exchange = getattr(self._underlying_ticker.contract, "exchange", "") or ""
                md_label = _market_data_label(self._underlying_ticker)
                lines.append(
                    Text(
                        f"MD: {md_exchange or 'n/a'} ({md_label})",
                        style="dim",
                    )
                )
                md_local = getattr(self._underlying_ticker.contract, "localSymbol", "") or ""
                md_debug = f"MD Contract: {self._underlying_ticker.contract.secType}"
                if md_local:
                    md_debug += f" {md_local}"
                lines.append(Text(md_debug, style="dim"))
                bid = _safe_num(self._underlying_ticker.bid)
                ask = _safe_num(self._underlying_ticker.ask)
                last = _safe_num(self._underlying_ticker.last)
                lines.append(
                    Text(
                        f"Bid: {_fmt_quote(bid)}  Ask: {_fmt_quote(ask)}  Last: {_fmt_quote(last)}"
                    )
                )
            lines.extend(self._render_execution_block())
            self._detail_left.update(Text("\n").join(lines))
        except Exception as exc:
            self._detail_left.update(Text(f"Detail render error: {exc}", style="red"))
        try:
            self._render_orders_panel()
        except Exception as exc:
            self._detail_right.update(Text(f"Orders render error: {exc}", style="red"))

    def _render_execution_block(self) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT"):
            return []
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        tick = _tick_size(contract, self._ticker, last or mark)
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        optimistic_buy = _round_to_tick(_optimistic_price(bid, ask, mid, "BUY"), tick)
        optimistic_sell = _round_to_tick(_optimistic_price(bid, ask, mid, "SELL"), tick)
        aggressive_buy = _round_to_tick(_aggressive_price(bid, ask, mid, "BUY"), tick)
        aggressive_sell = _round_to_tick(_aggressive_price(bid, ask, mid, "SELL"), tick)
        custom = _parse_float(self._exec_custom_input)
        if custom is not None:
            custom = _round_to_tick(custom, tick)
        qty = self._exec_qty
        lines: list[Text] = []
        lines.append(Text(""))
        lines.append(Text("Execution", style="bold"))
        lines.append(Text(f"Tick: {tick:.{_tick_decimals(tick)}f}", style="dim"))
        lines.append(Text("B=Buy  S=Sell", style="dim"))
        if self._exec_status:
            lines.append(Text(self._exec_status, style="yellow"))
        rows = [
            ("mid", "Mid"),
            (
                "optimistic",
                f"Optimistic (B/S): {_fmt_quote(optimistic_buy)} / {_fmt_quote(optimistic_sell)}",
            ),
            (
                "aggressive",
                f"Aggressive (B/S): {_fmt_quote(aggressive_buy)} / {_fmt_quote(aggressive_sell)}",
            ),
            ("custom", f"Custom: {_fmt_quote(custom)}"),
            ("qty", f"Qty: {qty}"),
        ]
        for idx, (key, label) in enumerate(rows):
            style = (
                "bold on #2b2b2b"
                if self._active_panel == "exec" and idx == self._exec_selected
                else ""
            )
            if key == "mid":
                lines.append(Text(""))
                line = Text("Mid: ")
                line.append(_fmt_quote(mid), style="orange1")
            else:
                line = Text(label)
            if style:
                line.stylize(style)
            lines.append(line)
        return lines

    def _render_orders_panel(self) -> None:
        con_ids: list[int] = []
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            con_ids.append(con_id)
        if self._underlying_con_id and self._underlying_con_id != con_id:
            con_ids.append(self._underlying_con_id)
        trades = self._client.open_trades_for_conids(con_ids)
        trades.sort(key=_trade_sort_key, reverse=True)
        self._orders_rows = trades
        if not trades:
            self._orders_selected = 0
            self._orders_scroll = 0
            lines = [Text("Orders", style="bold"), Text("No open orders", style="dim")]
            self._detail_right.update(Text("\n").join(lines))
            return
        if self._orders_selected >= len(trades):
            self._orders_selected = len(trades) - 1
        header = Text("Label Status Side Qty Type@Price Fill/Rem Id", style="dim")
        lines: list[Text] = [Text("Orders", style="bold"), header]
        available = self._detail_right.size.height
        visible = len(trades)
        if available:
            visible = max(available - len(lines), 1)
        max_scroll = max(len(trades) - visible, 0)
        self._orders_scroll = min(max(self._orders_scroll, 0), max_scroll)
        if self._orders_selected < self._orders_scroll:
            self._orders_scroll = self._orders_selected
        elif self._orders_selected >= self._orders_scroll + visible:
            self._orders_scroll = self._orders_selected - visible + 1
        start = self._orders_scroll
        end = min(start + visible, len(trades))
        for idx in range(start, end):
            trade = trades[idx]
            line = self._format_order_line(trade)
            if self._active_panel == "orders" and idx == self._orders_selected:
                line.stylize("bold on #2b2b2b")
            lines.append(line)
        self._detail_right.update(Text("\n").join(lines))

    def _format_order_line(self, trade: Trade) -> Text:
        contract = trade.contract
        label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or "?"
        label = label[:10]
        status = trade.orderStatus.status or "n/a"
        status = status.replace("PreSubmitted", "PreSub")
        side = (trade.order.action or "?")[:1]
        qty = _fmt_qty(float(trade.order.totalQuantity or 0))
        order_type = trade.order.orderType or ""
        price = self._order_price(trade)
        if price is not None:
            type_label = f"{order_type}@{_fmt_quote(price)}"
        else:
            type_label = order_type
        filled = _fmt_qty(float(trade.orderStatus.filled or 0))
        remaining = _fmt_qty(float(trade.orderStatus.remaining or 0))
        order_id = trade.order.orderId or trade.order.permId or 0
        line = (
            f"{label:<10} {status:<10} {side:<1} {qty:>4} "
            f"{type_label:<12} {filled}/{remaining} #{order_id}"
        )
        return Text(line)

    def _order_price(self, trade: Trade) -> float | None:
        order = trade.order
        price = _safe_num(getattr(order, "lmtPrice", None))
        if price is not None:
            return price
        return _safe_num(getattr(order, "auxPrice", None))

    def _selected_order(self) -> Trade | None:
        if not self._orders_rows:
            return None
        idx = min(max(self._orders_selected, 0), len(self._orders_rows) - 1)
        return self._orders_rows[idx]

    async def _cancel_order(self, trade: Trade) -> None:
        try:
            await self._client.cancel_trade(trade)
            order_id = trade.order.orderId or trade.order.permId or 0
            self._exec_status = f"Cancel sent #{order_id}"
        except Exception as exc:
            self._exec_status = f"Cancel error: {exc}"
        self._render_details()

    def _handle_digit(self, char: str) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._exec_qty_input = _append_digit(self._exec_qty_input, char, allow_decimal=False)
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        else:
            if selected != "custom":
                self._exec_selected = self._exec_rows.index("custom")
            self._exec_custom_input = _append_digit(self._exec_custom_input, char, allow_decimal=True)
        self._render_details()

    def _handle_backspace(self) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._exec_qty_input = self._exec_qty_input[:-1]
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        else:
            self._exec_custom_input = self._exec_custom_input[:-1]
        self._render_details()

    def _adjust_custom_price(self, direction: int) -> None:
        tick = _tick_size(self._item.contract, self._ticker, _mark_price(self._item))
        current = _parse_float(self._exec_custom_input)
        if current is None:
            bid = _safe_num(self._ticker.bid) if self._ticker else None
            ask = _safe_num(self._ticker.ask) if self._ticker else None
            mid = _midpoint(bid, ask)
            current = _round_to_tick(mid, tick) or 0.0
        adjusted = _round_to_tick(current + (tick * direction), tick)
        if adjusted is None:
            return
        self._exec_custom_input = f"{adjusted:.{_tick_decimals(tick)}f}"

    def _adjust_qty(self, direction: int) -> None:
        next_qty = max(1, int(self._exec_qty) + direction)
        self._exec_qty = next_qty
        self._exec_qty_input = str(next_qty)

    def _submit_order(self, action: str) -> None:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT"):
            self._exec_status = "Exec: unsupported contract"
            self._render_details()
            return
        qty = int(self._exec_qty) if self._exec_qty else 0
        if qty <= 0:
            self._exec_status = "Exec: invalid qty"
            self._render_details()
            return
        price = self._selected_exec_price(action)
        if price is None:
            self._exec_status = "Exec: price n/a"
            self._render_details()
            return
        outside_rth = contract.secType == "STK"
        self._exec_status = f"Sending {action} {qty} @ {price:.2f}"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Exec: no loop"
            self._render_details()
            return
        loop.create_task(self._place_order(action, qty, price, outside_rth))

    async def _place_order(
        self, action: str, qty: int, price: float, outside_rth: bool
    ) -> None:
        try:
            await self._client.place_limit_order(
                self._item.contract, action, qty, price, outside_rth
            )
            self._exec_status = f"Sent {action} {qty} @ {price:.2f}"
        except Exception as exc:
            self._exec_status = f"Exec error: {exc}"
        self._render_details()

    def _selected_exec_price(self, action: str) -> float | None:
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        tick = _tick_size(self._item.contract, self._ticker, last or mark)
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        fallback = _round_to_tick(last or mark, tick)
        selected = self._exec_rows[self._exec_selected]
        if selected == "mid":
            return mid or fallback
        if selected == "optimistic":
            value = _optimistic_price(bid, ask, mid, action)
            return _round_to_tick(value, tick) or fallback
        if selected == "aggressive":
            value = _aggressive_price(bid, ask, mid, action)
            return _round_to_tick(value, tick) or fallback
        if selected == "custom":
            value = _parse_float(self._exec_custom_input)
            return _round_to_tick(value, tick) if value is not None else None
        return fallback


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


def _trade_sort_key(trade: Trade) -> int:
    order = trade.order
    order_id = getattr(order, "orderId", 0) or 0
    perm_id = getattr(order, "permId", 0) or 0
    return int(order_id or perm_id or 0)


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


def _pnl_value(pnl: PnL | None) -> float | None:
    if not pnl:
        return None
    value = pnl.dailyPnL
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _estimate_net_liq(
    net_liq: float, pnl: PnL | None, anchor: float | None
) -> float | None:
    daily = _pnl_value(pnl)
    if daily is None or anchor is None:
        return None
    return net_liq + (daily - anchor)


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
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or num == 0:
            continue
        return num
    return None


def _market_data_tag(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return " [L]"
    if md_type in (3, 4):
        return " [D]"
    return ""


def _market_data_label(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return "Live"
    if md_type in (3, 4):
        return "Delayed"
    return "n/a"


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
        if price is None or price <= 0:
            if close and close > 0:
                text.append_text(_ticker_closed(label + tag, close))
            else:
                text.append_text(_ticker_missing(label + tag))
            continue
        if close is None or close <= 0:
            text.append_text(_ticker_price_only(label + tag, price))
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


def _ticker_price_only(label: str, price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    blank_change = "n/a".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(blank_change, style="dim")
    text.append(" ")
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


def _safe_num(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or num == 0:
        return None
    return num


def _mark_price(item: PortfolioItem) -> float | None:
    value = _safe_num(getattr(item, "marketPrice", None))
    if value is not None:
        return value
    if item.marketValue is not None and item.position:
        try:
            return float(item.marketValue) / float(item.position)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _fmt_quote(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _default_order_qty(item: PortfolioItem) -> int:
    return 1


def _tick_size(contract, ticker: Ticker | None, ref_price: float | None) -> float:
    if ticker is not None:
        value = getattr(ticker, "minTick", None)
        if value:
            try:
                tick = float(value)
                if tick > 0:
                    return tick
            except (TypeError, ValueError):
                pass
    value = getattr(contract, "minTick", None)
    if value:
        try:
            tick = float(value)
            if tick > 0:
                return tick
        except (TypeError, ValueError):
            pass
    if contract.secType == "OPT":
        if ref_price is not None and ref_price >= 3:
            return 0.05
        return 0.01
    return 0.01


def _round_to_tick(value: float | None, tick: float) -> float | None:
    if value is None:
        return None
    if not tick:
        return value
    return round(value / tick) * tick


def _tick_decimals(tick: float) -> int:
    text = f"{tick:.10f}".rstrip("0").rstrip(".")
    if "." in text:
        return len(text.split(".")[1])
    return 0


def _optimistic_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    if mid is None:
        return bid if action == "BUY" else ask
    if action == "BUY":
        if bid is None:
            return mid
        return (mid + bid) / 2.0
    if ask is None:
        return mid
    return (mid + ask) / 2.0


def _aggressive_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    if action == "BUY":
        return ask or mid or bid
    return bid or mid or ask


def _append_digit(value: str, char: str, allow_decimal: bool) -> str:
    if char == "." and not allow_decimal:
        return value
    if char == "." and "." in value:
        return value
    if char == "." and not value:
        return "0."
    return value + char


def _parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: str) -> int | None:
    if not value:
        return None
    if not value.isdigit():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
