"""Position detail screen (execution + quotes)."""

from __future__ import annotations

import asyncio
import math
from time import monotonic

from ib_insync import PortfolioItem, Ticker, Trade
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..client import IBKRClient
from ..engines.execution import (
    EXECUTION_POLICY,
    execution_mode_label,
    execution_price,
    _limit_price_for_mode,
    _midpoint,
    _quote_num_actionable,
    _round_to_tick,
    _tick_decimals,
    _tick_size,
)
from ..live.execution import LiveOrderExecution, order_ids
from .common import (
    _append_digit,
    _default_order_qty,
    _mark_price,
    _option_display_price,
    _parse_int,
    _safe_num,
    _ticker_close,
    _ticker_price,
    _unrealized_pnl_values,
)
from .position_detail.frame import pane_width
from .position_detail.market import PositionMarketView
from .position_detail.orders import PositionOrderView
from .time_compat import now_et as _now_et

class PositionDetailScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("b", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("r", "refresh_ticker", "Refresh MD"),
        ("a", "cycle_aurora_preset", "Aurora"),
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
    _ORDER_PANEL_NOTICE_TTL_SEC = 5 * 60.0
    _CHASE_PENDING_ACK_SEC = 0.9
    _CHASE_RECONCILE_INTERVAL_SEC = 0.9
    _CHASE_FORCE_RECONCILE_INTERVAL_SEC = 5.0
    _CHASE_MODIFY_ERROR_BACKOFF_SEC = 1.0
    _STREAM_RENDER_DEBOUNCE_SEC = 0.08
    def __init__(
        self,
        client: IBKRClient,
        item: PortfolioItem,
        refresh_sec: float,
        *,
        session_closes: tuple[float | None, float | None] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._item = item
        self._refresh_sec = max(refresh_sec, 0.1)
        self._ticker: Ticker | None = None
        self._underlying_ticker: Ticker | None = None
        self._underlying_con_id: int | None = None
        self._underlying_label: str | None = None
        self._refresh_task = None
        self._chase_tasks: set[asyncio.Task] = set()
        self._policy = EXECUTION_POLICY
        self._market = PositionMarketView(
            client,
            item,
            self._refresh_sec,
            session_prev_close=session_closes[0] if session_closes else None,
            request_render=lambda: self._request_stream_render(sample=False),
        )
        self._execution = LiveOrderExecution(
            client=client,
            error_max_age_sec=self._ORDER_PANEL_NOTICE_TTL_SEC,
            price_for_mode=self._exec_price_for_mode,
            recent_spreads=lambda: self._market.chart.spread_samples,
            on_update=self._on_chase_update,
        )
        self._orders = PositionOrderView(
            client,
            self._execution,
            lambda: self._policy,
            self._market.chart,
            item,
            _default_order_qty(item),
            initial_price=self._initial_exec_price,
            mode_price=self._exec_price_for_mode,
        )
        self._closes_task: asyncio.Task | None = None
        self._bootstrap_task: asyncio.Task | None = None
        self._stream_render_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Static("", id="detail-left"),
            Static("", id="detail-right"),
            id="detail-body",
        )
        yield Static("", id="detail-legend")
        yield Footer()

    async def on_mount(self) -> None:
        self._detail_left = self.query_one("#detail-left", Static)
        self._detail_right = self.query_one("#detail-right", Static)
        self._detail_legend = self.query_one("#detail-legend", Static)
        self._client.add_stream_listener(self._capture_tick_mid)
        self._refresh_task = self.set_interval(self._refresh_sec, self._render_details)
        self._render_details()
        try:
            loop = asyncio.get_running_loop()
            self._bootstrap_task = loop.create_task(self._bootstrap_screen_data())
        except RuntimeError:
            self._bootstrap_task = None
            await self._bootstrap_screen_data()

    async def on_unmount(self) -> None:
        self._client.remove_stream_listener(self._capture_tick_mid)
        if self._refresh_task:
            self._refresh_task.stop()
        if self._stream_render_task and not self._stream_render_task.done():
            self._stream_render_task.cancel()
        if self._bootstrap_task and not self._bootstrap_task.done():
            self._bootstrap_task.cancel()
        if self._closes_task and not self._closes_task.done():
            self._closes_task.cancel()
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id, owner="details")
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")

    async def _bootstrap_screen_data(self) -> None:
        try:
            await self._maybe_align_front_future_contract()
            self._ticker = await self._client.ensure_ticker(self._item.contract, owner="details")
            try:
                loop = asyncio.get_running_loop()
                self._closes_task = loop.create_task(self._load_session_closes())
            except RuntimeError:
                self._closes_task = None
            await self._load_underlying()
        except asyncio.CancelledError:
            raise
        except Exception:
            return
        self._render_details_if_mounted(sample=False)

    async def _maybe_align_front_future_contract(self) -> None:
        contract = self._item.contract
        if str(getattr(contract, "secType", "") or "").strip().upper() != "FUT":
            return
        try:
            qty = float(getattr(self._item, "position", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        # Search-opened synthetic rows are flat and account-less. Keep explicit live
        # positions pinned to their exact contract.
        if abs(qty) > 1e-12 or getattr(self._item, "account", None):
            return
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        if not symbol:
            return
        exchanges: list[str] = []
        for raw_exchange in (
            str(getattr(contract, "exchange", "") or "").strip().upper(),
            str(getattr(contract, "primaryExchange", "") or "").strip().upper(),
            "CME",
        ):
            if raw_exchange and raw_exchange not in exchanges:
                exchanges.append(raw_exchange)
        resolved = None
        for exchange in exchanges:
            try:
                resolved = await self._client.front_future(
                    symbol,
                    exchange=exchange,
                    cache_ttl_sec=180.0,
                )
            except Exception:
                resolved = None
            if resolved is not None:
                break
        if resolved is None:
            return
        current_con_id = int(getattr(contract, "conId", 0) or 0)
        resolved_con_id = int(getattr(resolved, "conId", 0) or 0)
        if current_con_id and resolved_con_id and current_con_id == resolved_con_id:
            return
        latest = self._client.portfolio_item(resolved_con_id) if resolved_con_id else None
        if latest is not None:
            self._item = latest
            return
        account = str(getattr(self._item, "account", "") or "")
        self._item = PortfolioItem(
            contract=resolved,
            position=0.0,
            marketPrice=0.0,
            marketValue=0.0,
            averageCost=0.0,
            unrealizedPNL=0.0,
            realizedPNL=0.0,
            account=account,
        )

    async def _load_session_closes(self) -> None:
        try:
            prev_close, _ = await self._client.session_closes(self._item.contract)
        except Exception:
            return
        self._market.set_session_close(prev_close)
        self._render_details_if_mounted(sample=False)

    @staticmethod
    def _flat_item_snapshot(item: PortfolioItem) -> PortfolioItem:
        realized = _safe_num(getattr(item, "realizedPNL", None)) or 0.0
        market_price = _safe_num(getattr(item, "marketPrice", None)) or 0.0
        account = str(getattr(item, "account", "") or "")
        return PortfolioItem(
            contract=item.contract,
            position=0.0,
            marketPrice=float(market_price),
            marketValue=0.0,
            averageCost=0.0,
            unrealizedPNL=0.0,
            realizedPNL=float(realized),
            account=account,
        )

    @staticmethod
    def _quote_num(value: float | None) -> float | None:
        return _quote_num_actionable(value)

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
        if self._orders.active_panel == "orders":
            if self._orders.rows:
                self._orders.selected = max(self._orders.selected - 1, 0)
        else:
            self._orders.exec_selected = max(self._orders.exec_selected - 1, 0)
        self._render_details(sample=False)

    def action_exec_down(self) -> None:
        if self._orders.active_panel == "orders":
            if self._orders.rows:
                self._orders.selected = min(
                    self._orders.selected + 1, len(self._orders.rows) - 1
                )
        else:
            self._orders.exec_selected = min(self._orders.exec_selected + 1, len(self._orders.exec_rows) - 1)
        self._render_details(sample=False)

    def action_exec_left(self) -> None:
        if self._orders.active_panel == "orders":
            self._orders.active_panel = "exec"
            self._render_details(sample=False)
            return
        selected = self._orders.exec_rows[self._orders.exec_selected]
        if self._orders.active_panel != "exec":
            self.app.pop_screen()
            return
        if selected == "qty":
            self._adjust_qty(-1)
            self._render_details(sample=False)
            return
        if selected == "custom":
            self._adjust_custom_price(-1)
            self._render_details(sample=False)
            return
        self.app.pop_screen()
        return

    def action_exec_right(self) -> None:
        if self._orders.active_panel == "orders":
            self._render_details(sample=False)
            return
        selected = self._orders.exec_rows[self._orders.exec_selected]
        if selected == "qty":
            self._adjust_qty(1)
        elif selected == "custom":
            self._adjust_custom_price(1)
        else:
            self._orders.active_panel = "orders"
        self._render_details(sample=False)

    def action_exec_jump_left(self) -> None:
        if self._orders.active_panel == "orders":
            self._orders.active_panel = "exec"
        else:
            self._orders.exec_selected = self._orders.exec_rows.index("qty")
        self._render_details(sample=False)

    def action_exec_jump_right(self) -> None:
        if self._orders.active_panel == "orders":
            self._render_details(sample=False)
            return
        self._orders.exec_selected = self._orders.exec_rows.index("ladder")
        self._render_details(sample=False)

    def action_cancel_order(self) -> None:
        if self._orders.active_panel != "orders":
            self._orders.status = "Cancel: focus orders panel"
            self._orders.set_notice(self._orders.status, level="warn")
            self._render_details(sample=False)
            return
        trade = self._orders.selected_order()
        if not trade:
            self._orders.status = "Cancel: no order"
            self._orders.set_notice(self._orders.status, level="warn")
            self._render_details(sample=False)
            return
        order_id_raw, perm_id_raw = order_ids(trade)
        self._execution.mark_cancel_requested(order_id=order_id_raw, perm_id=perm_id_raw)
        self._execution.cancel_task(order_id=order_id_raw, perm_id=perm_id_raw)
        order_id = order_id_raw or perm_id_raw or 0
        self._orders.status = f"Canceling #{order_id}"
        self._orders.set_notice(self._orders.status, level="warn")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._orders.status = "Cancel: no loop"
            self._orders.set_notice(self._orders.status, level="error")
            self._render_details(sample=False)
            return
        loop.create_task(self._cancel_order(trade))
        self._render_details(sample=False)

    def action_cycle_aurora_preset(self) -> None:
        self._orders.status = f"Aurora preset: {self._market.chart.cycle_aurora().upper()}"
        self._render_details(sample=False)

    async def action_refresh_ticker(self) -> None:
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id, owner="details")
        self._ticker = await self._client.ensure_ticker(self._item.contract, owner="details")
        self._market.start_probe(self._ticker)
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")
            self._underlying_con_id = None
            self._underlying_ticker = None
            self._underlying_label = None
            await self._load_underlying()
        self._orders.status = "MD refreshed"
        self._render_details()

    async def _load_underlying(self) -> None:
        contract = self._item.contract
        if contract.secType not in ("OPT", "FOP"):
            return
        underlying = await self._client.resolve_underlying_contract(contract)
        if not underlying:
            return
        self._underlying_ticker = await self._client.ensure_ticker(underlying, owner="details")
        con_id = int(getattr(underlying, "conId", 0) or 0)
        if con_id:
            self._underlying_con_id = con_id
        symbol = getattr(underlying, "localSymbol", "") or getattr(underlying, "symbol", "")
        sec_type = getattr(underlying, "secType", "")
        if symbol or sec_type:
            self._underlying_label = " ".join(part for part in (symbol, sec_type) if part)

    def _render_details_if_mounted(self, *, sample: bool = True) -> None:
        widget = getattr(self, "_detail_left", None)
        if widget is None or not bool(getattr(widget, "is_mounted", False)):
            return
        self._render_details(sample=sample)

    def _request_stream_render(self, *, sample: bool = False) -> None:
        task = self._stream_render_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._render_details_if_mounted(sample=sample)
            return

        async def _run() -> None:
            try:
                await asyncio.sleep(max(0.0, float(self._STREAM_RENDER_DEBOUNCE_SEC)))
                self._render_details_if_mounted(sample=sample)
            finally:
                self._stream_render_task = None

        self._stream_render_task = loop.create_task(_run())

    def _sync_bound_tickers(self) -> None:
        con_id = int(getattr(self._item.contract, "conId", 0) or 0)
        if con_id:
            latest = self._client.ticker_for_con_id(con_id)
            if latest is not None:
                self._ticker = latest
        if self._underlying_con_id:
            latest_underlying = self._client.ticker_for_con_id(int(self._underlying_con_id))
            if latest_underlying is not None:
                self._underlying_ticker = latest_underlying

    def _on_chase_update(
        self,
        status: str | None,
        notice: str | None,
        level: str,
    ) -> None:
        if status is not None:
            self._orders.status = status
        if notice is not None:
            self._orders.set_notice(notice, level=level)
        self._render_details_if_mounted(sample=False)

    def _capture_tick_mid(self) -> None:
        self._sync_bound_tickers()
        self._market.bind(
            self._item,
            self._ticker,
            underlying_ticker=self._underlying_ticker,
            underlying_label=self._underlying_label,
        )
        self._market.capture_tick()

    def _render_details(self, *, sample: bool = True) -> None:
        self._sync_bound_tickers()
        self._market.bind(
            self._item,
            self._ticker,
            underlying_ticker=self._underlying_ticker,
            underlying_label=self._underlying_label,
        )
        if hasattr(self, "_detail_legend"):
            self._detail_legend.update(self._market.chart.render_legend())
        try:
            con_id = int(self._item.contract.conId or 0)
            latest = self._client.portfolio_item(con_id) if con_id else None
            if latest is not None:
                self._item = latest
            elif con_id and float(getattr(self._item, "position", 0.0) or 0.0) != 0.0:
                # Filled-to-flat positions disappear from portfolio(); don't keep stale nonzero qty.
                self._item = self._flat_item_snapshot(self._item)
            self._market.bind(
                self._item,
                self._ticker,
                underlying_ticker=self._underlying_ticker,
                underlying_label=self._underlying_label,
            )
            self._orders.bind(
                self._item,
                self._ticker,
                underlying_con_id=self._underlying_con_id,
            )
            contract = self._item.contract
            bid = self._quote_num(self._ticker.bid) if self._ticker else None
            ask = self._quote_num(self._ticker.ask) if self._ticker else None
            last = self._quote_num(self._ticker.last) if self._ticker else None
            mid = _midpoint(bid, ask)
            if contract.secType in ("OPT", "FOP"):
                display_price = _option_display_price(self._item, self._ticker)
                actionable = display_price if contract.secType == "FOP" else (mid or last)
                price = self._market.resolve_derivative_price(
                    display_price,
                    actionable,
                    now_mono=monotonic(),
                )
            else:
                price = _ticker_price(self._ticker) if self._ticker else None
            close = _ticker_close(self._ticker) if self._ticker else None
            mark = _mark_price(self._item)
            live_mark = price or mark
            spread = (ask - bid) if bid is not None and ask is not None and ask >= bid else None
            size = _safe_num(getattr(self._ticker, "lastSize", None)) if self._ticker else None
            bid_size = _safe_num(getattr(self._ticker, "bidSize", None)) if self._ticker else None
            ask_size = _safe_num(getattr(self._ticker, "askSize", None)) if self._ticker else None
            broker_pnl, _ = _unrealized_pnl_values(self._item, mark_price=live_mark)
            fast_pnl = self._market.live_unrealized(live_mark)
            pnl_value = fast_pnl if fast_pnl is not None else broker_pnl
            ref_price = last or live_mark
            slip_proxy = abs(ref_price - mid) if ref_price is not None and mid is not None else None
            if sample:
                chart = self._market.chart
                previous_mid = chart.mid_samples[-1] if chart.mid_samples else None
                imbalance = None
                if bid_size is not None or ask_size is not None:
                    total_size = float((bid_size or 0.0) + (ask_size or 0.0))
                    imbalance = (
                        float((bid_size or 0.0) - (ask_size or 0.0)) / total_size
                        if total_size > 0
                        else 0.0
                    )
                self._market.record(
                    mid=mid or price or mark,
                    spread=spread,
                    size=size,
                    pnl=pnl_value,
                    slip_proxy=slip_proxy,
                    imbalance=imbalance,
                    vol_burst=(
                        abs(ref_price - previous_mid)
                        if ref_price is not None and previous_mid is not None
                        else 0.0
                    ),
                )

            width = pane_width(self._detail_left, floor=58)
            lines = self._market.render_hud(
                panel_width=width,
                bid=bid,
                ask=ask,
                last=last,
                price=price,
                mid=mid,
                close=close,
                mark=mark,
                spread=spread,
            )
            lines.append(Text(""))
            lines.extend(self._orders.render_execution(panel_width=width))
            self._detail_left.update(Text("\n").join(lines))
        except Exception as exc:
            self._detail_left.update(Text(f"Detail render error: {exc}", style="red"))
        try:
            self._render_orders_panel()
        except Exception as exc:
            self._detail_right.update(Text(f"Orders render error: {exc}", style="red"))

    def _render_orders_panel(self) -> None:
        self._orders.bind(
            self._item,
            self._ticker,
            underlying_con_id=self._underlying_con_id,
        )
        self._detail_right.update(
            self._orders.render_orders(
                panel_width=pane_width(self._detail_right, floor=44),
                available=int(getattr(self._detail_right.size, "height", 0) or 0),
            )
        )

    async def _cancel_order(self, trade: Trade) -> None:
        order_id, perm_id = order_ids(trade)
        order_ref = int(order_id or perm_id or 0)
        self._execution.mark_cancel_requested(order_id=order_id, perm_id=perm_id)
        try:
            await self._client.cancel_trade(trade)
        except Exception as exc:
            self._orders.status = f"Cancel error: {exc}"
            self._orders.set_notice(self._orders.status, level="error")
            self._execution.clear_cancel_requested(order_id=order_id, perm_id=perm_id)
            self._render_details(sample=False)
            return
        error_payload = (
            await self._execution.await_order_error(
                order_id,
                perm_id,
                attempts=4,
                interval_sec=0.1,
            )
            if order_ref
            else None
        )
        should_clear_cancel_intent = False
        if error_payload is not None:
            error_code, error_message = error_payload
            error_prefix = f"IB {error_code}: " if error_code else "IB: "
            level = "warn" if error_code in (10147, 10148, 10149) else "error"
            if order_ref:
                self._orders.status = f"Cancel #{order_ref}: {error_prefix}{error_message}"
            else:
                self._orders.status = f"Cancel: {error_prefix}{error_message}"
            self._orders.set_notice(self._orders.status, level=level)
            should_clear_cancel_intent = True
        else:
            ack_status = ""
            if order_ref:
                for attempt in range(5):
                    payload = self._execution.current_order_state(
                        order_id=order_id,
                        perm_id=perm_id,
                    )
                    if not isinstance(payload, dict) and attempt >= 2:
                        payload = await self._execution.reconcile_order_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            force=bool(attempt >= 4),
                        )
                    if isinstance(payload, dict):
                        ack_status = str(payload.get("effective_status") or "").strip()
                        if ack_status in (
                            "PendingCancel",
                            "Cancelled",
                            "ApiCancelled",
                            "Inactive",
                            "Filled",
                        ):
                            break
                    await asyncio.sleep(0.08)
            if order_ref:
                if ack_status:
                    self._orders.status = f"Cancel {ack_status} #{order_ref}"
                else:
                    self._orders.status = f"Cancel sent #{order_ref} (awaiting broker ack)"
            else:
                self._orders.status = "Cancel sent"
            self._orders.set_notice(self._orders.status, level="warn")
            should_clear_cancel_intent = ack_status in (
                "Cancelled",
                "ApiCancelled",
                "Inactive",
                "Filled",
            )
        if should_clear_cancel_intent:
            self._execution.clear_cancel_requested(order_id=order_id, perm_id=perm_id)
        self._render_details(sample=False)

    def _handle_digit(self, char: str) -> None:
        selected = self._orders.exec_rows[self._orders.exec_selected]
        if selected == "custom":
            if char not in "0123456789.":
                return
            self._orders.custom_input = _append_digit(self._orders.custom_input, char, allow_decimal=True)
            parsed = self._parse_custom_price(self._orders.custom_input)
            if parsed is not None:
                self._orders.custom_price = parsed
        else:
            if char not in "0123456789":
                return
            self._orders.qty_input = _append_digit(self._orders.qty_input, char, allow_decimal=False)
            parsed = _parse_int(self._orders.qty_input)
            if parsed:
                self._orders.qty = parsed
        self._render_details(sample=False)

    def _handle_backspace(self) -> None:
        selected = self._orders.exec_rows[self._orders.exec_selected]
        if selected == "custom":
            self._orders.custom_input = self._orders.custom_input[:-1]
            parsed = self._parse_custom_price(self._orders.custom_input)
            self._orders.custom_price = parsed
        else:
            self._orders.qty_input = self._orders.qty_input[:-1]
            parsed = _parse_int(self._orders.qty_input)
            if parsed:
                self._orders.qty = parsed
        self._render_details(sample=False)

    def _adjust_qty(self, direction: int) -> None:
        next_qty = max(1, int(self._orders.qty) + direction)
        self._orders.qty = next_qty
        self._orders.qty_input = str(next_qty)

    @staticmethod
    def _parse_custom_price(raw: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        if math.isnan(value) or value <= 0:
            return None
        return float(value)

    def _adjust_custom_price(self, direction: int) -> None:
        if direction == 0:
            return
        bid = self._quote_num(self._ticker.bid) if self._ticker else None
        ask = self._quote_num(self._ticker.ask) if self._ticker else None
        last = self._quote_num(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, self._ticker, last_ref)
        current = _round_to_tick(self._orders.custom_price, tick)
        if current is None:
            current = _round_to_tick(last_ref, tick)
        if current is None:
            return
        next_value = _round_to_tick(float(current) + (float(direction) * float(tick)), tick)
        if next_value is None or next_value <= 0:
            return
        self._orders.custom_price = float(next_value)
        self._orders.custom_input = f"{self._orders.custom_price:.{_tick_decimals(tick)}f}"

    def _submit_order(self, action: str) -> None:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP", "FUT"):
            self._orders.status = "Exec: unsupported contract"
            self._render_details(sample=False)
            return
        if contract.secType == "OPT":
            bid = self._quote_num(self._ticker.bid) if self._ticker else None
            ask = self._quote_num(self._ticker.ask) if self._ticker else None
            last = self._quote_num(self._ticker.last) if self._ticker else None
            has_actionable = bool(
                (bid is not None and ask is not None and bid <= ask)
                or (last is not None)
            )
            if not has_actionable:
                self._orders.status = "Exec locked: no actionable equity option quote yet (bid/ask/last n/a)"
                self._render_details(sample=False)
                return
        qty = int(self._orders.qty) if self._orders.qty else 0
        if qty <= 0:
            self._orders.status = "Exec: invalid qty"
            self._render_details(sample=False)
            return
        mode = self._orders.selected_mode()
        price = self._initial_exec_price(action, mode=mode)
        if price is None:
            if contract.secType == "OPT":
                self._orders.status = "Exec locked: no actionable option quote yet (bid/ask/last n/a)"
            else:
                self._orders.status = "Exec: price n/a"
            self._render_details(sample=False)
            return
        outside_rth = contract.secType == "STK"
        mode_label = execution_mode_label(mode)
        self._orders.status = f"Sending {action} {qty} @ {price:.2f} [{mode_label}]"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._orders.status = "Exec: no loop"
            self._render_details(sample=False)
            return
        loop.create_task(self._place_order(action, qty, price, outside_rth, mode))

    async def _place_order(
        self, action: str, qty: int, price: float, outside_rth: bool, mode: str
    ) -> None:
        try:
            trade = await self._client.place_limit_order(
                self._item.contract, action, qty, price, outside_rth
            )
            applied_price = _safe_num(getattr(getattr(trade, "order", None), "lmtPrice", None))
            if applied_price is None or applied_price <= 0:
                applied_price = float(price)
            mode_label = execution_mode_label(mode)
            self._orders.status = f"Sent {action} {qty} @ {float(applied_price):.2f} [{mode_label}]"
            order_id, perm_id = order_ids(trade)
            order_ref = order_id or perm_id
            if order_ref:
                self._orders.set_notice(
                    f"Submitted #{order_ref} {action} {qty} @ {float(applied_price):.2f} [{mode_label}]",
                    level="info",
                )
            else:
                self._orders.set_notice(
                    f"Submitted {action} {qty} @ {float(applied_price):.2f} [{mode_label}]",
                    level="info",
                )
            if order_ref:
                seeded = "OPTIMISTIC" if mode == "AUTO" else mode
                updates: dict[str, object] = {
                    "selected": execution_mode_label(mode),
                    "active": execution_mode_label(seeded),
                    "target_price": float(applied_price),
                    "mods": 0,
                }
                if str(mode).strip().upper() == "RELENTLESS_DELAY":
                    updates.update(
                        {
                            "delay_recoveries": 0,
                            "delay_anchor_price": None,
                            "delay_sweep_anchor_price": float(applied_price),
                            "delay_first_202_ts": None,
                            "delay_last_202_ts": None,
                            "delay_last_leg_sign": None,
                            "delay_last_leg_name": None,
                            "delay_locked_price_dir": None,
                        }
                    )
                self._execution.update_state(
                    order_id=order_id,
                    perm_id=perm_id,
                    updates=updates,
                )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(self._chase_until_filled(trade, action, mode=mode))
                self._chase_tasks.add(task)
                self._execution.register_task(task, order_id=order_id, perm_id=perm_id)

                def _on_chase_done(
                    done_task: asyncio.Task,
                    *,
                    seed_order_id: int = order_id,
                    seed_perm_id: int = perm_id,
                ) -> None:
                    self._chase_tasks.discard(done_task)
                    self._execution.unregister_task(done_task)
                    if done_task.cancelled():
                        return
                    try:
                        exc = done_task.exception()
                    except Exception:
                        return
                    if exc is None:
                        return
                    order_ref_local = int(seed_order_id or seed_perm_id or 0)
                    order_label = f"#{order_ref_local}" if order_ref_local else "order"
                    self._orders.status = f"Chase task error: {exc}"
                    self._orders.set_notice(
                        f"{order_label} chase task error: {exc}",
                        level="error",
                    )
                    self._render_details_if_mounted(sample=False)

                task.add_done_callback(_on_chase_done)
        except Exception as exc:
            self._orders.status = f"Exec error: {exc}"
            self._orders.set_notice(self._orders.status, level="error")
        self._render_details_if_mounted(sample=False)

    def _initial_exec_price(self, action: str, *, mode: str = "AUTO") -> float | None:
        bid = self._quote_num(self._ticker.bid) if self._ticker else None
        ask = self._quote_num(self._ticker.ask) if self._ticker else None
        last = self._quote_num(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, self._ticker, last_ref)
        if mode == "CUSTOM":
            custom = _round_to_tick(self._orders.custom_price, tick)
            if custom is not None:
                return custom
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        if mode in ("RELENTLESS", "RELENTLESS_DELAY"):
            quote_stale = self._policy.quote_is_stale(
                ticker=self._ticker,
                bid=bid,
                ask=ask,
                last=last,
            )
            return self._exec_price_for_mode(
                mode,
                action,
                bid=bid,
                ask=ask,
                last=last,
                ticker=self._ticker,
                elapsed_sec=0.0,
                quote_stale=quote_stale,
                open_shock=self._policy.in_open_shock(_now_et().time()),
                no_progress_reprices=0,
                arrival_ref=_midpoint(bid, ask) or last_ref,
            )
        selected_mode = "OPTIMISTIC" if mode == "AUTO" else mode
        value = (
            _limit_price_for_mode(bid, ask, last_ref, action=action, mode=selected_mode)
            if last_ref is not None or (bid is not None and ask is not None)
            else None
        )
        if value is None:
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        return _round_to_tick(float(value), tick)

    def _exec_price_for_mode(
        self,
        mode: str,
        action: str,
        *,
        bid: float | None = None,
        ask: float | None = None,
        last: float | None = None,
        ticker: Ticker | None = None,
        elapsed_sec: float = 0.0,
        quote_stale: bool = False,
        open_shock: bool = False,
        no_progress_reprices: int = 0,
        arrival_ref: float | None = None,
        delay_recoveries: int = 0,
        delay_anchor_price: float | None = None,
        delay_sweep_anchor_price: float | None = None,
        delay_locked_price_dir: float | None = None,
    ) -> float | None:
        bid = bid if bid is not None else (self._quote_num(self._ticker.bid) if self._ticker else None)
        ask = ask if ask is not None else (self._quote_num(self._ticker.ask) if self._ticker else None)
        last = last if last is not None else (self._quote_num(self._ticker.last) if self._ticker else None)
        ticker_ref = ticker or self._ticker
        mark = _option_display_price(self._item, ticker_ref) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        return execution_price(
            self._item.contract,
            ticker_ref,
            mode,
            action,
            bid=bid,
            ask=ask,
            last=last,
            fallback_price=mark,
            custom_price=self._orders.custom_price,
            policy=self._policy,
            elapsed_sec=elapsed_sec,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=no_progress_reprices,
            arrival_ref=arrival_ref,
            recent_spreads=self._market.chart.spread_samples,
            delay_recoveries=delay_recoveries,
            delay_anchor_price=delay_anchor_price,
            delay_sweep_anchor_price=delay_sweep_anchor_price,
            delay_locked_price_dir=delay_locked_price_dir,
        )

    async def _chase_until_filled(
        self,
        trade: Trade,
        action: str,
        *,
        mode: str = "AUTO",
    ) -> None:
        await self._execution.chase(
            trade,
            action,
            mode=mode,
            policy=self._policy,
            pending_ack_sec=self._CHASE_PENDING_ACK_SEC,
            reconcile_interval_sec=self._CHASE_RECONCILE_INTERVAL_SEC,
            force_reconcile_interval_sec=self._CHASE_FORCE_RECONCILE_INTERVAL_SEC,
            modify_error_backoff_sec=self._CHASE_MODIFY_ERROR_BACKOFF_SEC,
        )
