"""Position detail screen (execution + quotes)."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime

from ib_insync import Contract, PortfolioItem, Ticker, Trade
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..client import IBKRClient
from .common import (
    _aggressive_price,
    _append_digit,
    _default_order_qty,
    _fmt_quote,
    _market_data_label,
    _mark_price,
    _midpoint,
    _optimistic_price,
    _parse_float,
    _round_to_tick,
    _safe_num,
    _tick_decimals,
    _tick_size,
    _trade_sort_key,
    _quote_status_line,
    _ticker_price,
)

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
        if self._active_panel != "exec" or selected not in ("custom", "qty"):
            self.app.pop_screen()
            return
        if selected == "custom":
            self._adjust_custom_price(-1)
        else:
            self._adjust_qty(-1)
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
                lines.append(_quote_status_line(self._ticker))
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
                lines.append(_quote_status_line(self._underlying_ticker))
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


