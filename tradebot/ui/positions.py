"""Position detail screen (execution + quotes)."""

from __future__ import annotations

import asyncio
from collections import deque
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
from .common import (
    _aggressive_price,
    _append_digit,
    _cost_basis,
    _default_order_qty,
    _exec_chase_mode,
    _exec_chase_quote_signature,
    _exec_chase_should_reprice,
    _EXEC_LADDER_TIMEOUT_SEC,
    _fmt_expiry,
    _fmt_money,
    _fmt_qty,
    _fmt_quote,
    _infer_multiplier,
    _limit_price_for_mode,
    _market_data_label,
    _mark_price,
    _midpoint,
    _option_display_price,
    _optimistic_price,
    _parse_int,
    _quote_num_actionable,
    _round_to_tick,
    _safe_num,
    _tick_decimals,
    _tick_size,
    _trade_sort_key,
    _quote_status_line,
    _ticker_close,
    _ticker_price,
    _unrealized_pnl_values,
)

_DETAIL_CHASE_STATE_BY_ORDER: dict[int, dict[str, object]] = {}

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
    _SPARK_CHARS = "▁▂▃▄▅▆▇█"
    _SPARK_LEVELS = " ▁▂▃▄▅▆▇█"
    _MOMENTUM_CHARS = "░▒▓█"
    _BAR_FILL = "█"
    _BAR_EMPTY = "░"
    _POSITION_PULSE_SEC = 1.6
    _TREND_WINDOW_SEC = 60.0
    _TREND_RETENTION_SEC = 180.0
    _TREND_BRAILLE_X = 2
    _TREND_BRAILLE_Y = 4
    _AURORA_TAPE_BIAS = 0.15
    _VOL_MAGENTA_STYLES = ("#4e1e57", "#6f2380", "#922aaa", "#b534d8", "#ff3dee")
    _TREND_ROWS = 5
    _AURORA_PRESET_ORDER = ("calm", "normal", "feral")
    _AURORA_PRESETS = {
        "calm": {"buy_soft": 0.28, "buy_strong": 0.56, "sell_soft": -0.28, "sell_strong": -0.56, "burst_gain": 0.80},
        "normal": {"buy_soft": 0.16, "buy_strong": 0.34, "sell_soft": -0.16, "sell_strong": -0.34, "burst_gain": 1.00},
        "feral": {"buy_soft": 0.08, "buy_strong": 0.22, "sell_soft": -0.08, "sell_strong": -0.22, "burst_gain": 1.30},
    }

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
        self._exec_rows = ["ladder", "optimistic", "mid", "aggressive", "cross", "qty"]
        self._exec_selected = 0
        self._exec_qty_input = ""
        self._exec_qty = _default_order_qty(item)
        self._exec_status: str | None = None
        self._active_panel = "exec"
        self._orders_selected = 0
        self._orders_scroll = 0
        self._orders_rows: list[Trade] = []
        self._refresh_task = None
        self._chase_tasks: set[asyncio.Task] = set()
        self._mid_samples: deque[float] = deque(maxlen=240)
        self._mid_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._spread_samples: deque[float] = deque(maxlen=96)
        self._size_samples: deque[float] = deque(maxlen=96)
        self._size_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._vol_flow_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._pnl_samples: deque[float] = deque(maxlen=96)
        self._slip_proxy_samples: deque[float] = deque(maxlen=96)
        self._imbalance_samples: deque[float] = deque(maxlen=240)
        self._vol_burst_samples: deque[float] = deque(maxlen=240)
        self._imbalance_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._vol_burst_tape: deque[tuple[float, float]] = deque(maxlen=4096)
        self._trend_bins: list[float] = []
        self._aurora_preset = "normal"
        self._pos_prev_qty: float | None = None
        self._pos_delta: float = 0.0
        self._pos_pulse_until: float = 0.0
        self._last_tick_signature: tuple[
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
        ] | None = None
        self._flow_cum_attr: str | None = None
        self._flow_cum_prev: float | None = None
        self._flow_prev_price: float | None = None
        self._flow_last_size: float | None = None
        self._flow_last_price: float | None = None
        self._session_prev_close: float | None = session_closes[0] if session_closes else None
        self._session_close_3ago: float | None = session_closes[1] if session_closes else None
        self._closes_task: asyncio.Task | None = None

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
        self._ticker = await self._client.ensure_ticker(self._item.contract, owner="details")
        self._client.add_stream_listener(self._capture_tick_mid)
        try:
            loop = asyncio.get_running_loop()
            self._closes_task = loop.create_task(self._load_session_closes())
        except RuntimeError:
            self._closes_task = None
        await self._load_underlying()
        self._refresh_task = self.set_interval(self._refresh_sec, self._render_details)
        self._render_details()

    async def on_unmount(self) -> None:
        self._client.remove_stream_listener(self._capture_tick_mid)
        if self._refresh_task:
            self._refresh_task.stop()
        if self._closes_task and not self._closes_task.done():
            self._closes_task.cancel()
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id, owner="details")
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")

    async def _load_session_closes(self) -> None:
        try:
            prev_close, close_3ago = await self._client.session_closes(self._item.contract)
        except Exception:
            return
        self._session_prev_close = prev_close
        self._session_close_3ago = close_3ago
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
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = max(self._orders_selected - 1, 0)
        else:
            self._exec_selected = max(self._exec_selected - 1, 0)
        self._render_details(sample=False)

    def action_exec_down(self) -> None:
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = min(
                    self._orders_selected + 1, len(self._orders_rows) - 1
                )
        else:
            self._exec_selected = min(self._exec_selected + 1, len(self._exec_rows) - 1)
        self._render_details(sample=False)

    def action_exec_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
            self._render_details(sample=False)
            return
        selected = self._exec_rows[self._exec_selected]
        if self._active_panel != "exec" or selected != "qty":
            self.app.pop_screen()
            return
        self._adjust_qty(-1)
        self._render_details(sample=False)

    def action_exec_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details(sample=False)
            return
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._adjust_qty(1)
        else:
            self._active_panel = "orders"
        self._render_details(sample=False)

    def action_exec_jump_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
        else:
            self._exec_selected = self._exec_rows.index("qty")
        self._render_details(sample=False)

    def action_exec_jump_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details(sample=False)
            return
        self._exec_selected = self._exec_rows.index("ladder")
        self._render_details(sample=False)

    def action_cancel_order(self) -> None:
        if self._active_panel != "orders":
            return
        trade = self._selected_order()
        if not trade:
            self._exec_status = "Cancel: no order"
            self._render_details(sample=False)
            return
        order_id = trade.order.orderId or trade.order.permId or 0
        self._exec_status = f"Canceling #{order_id}"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Cancel: no loop"
            self._render_details(sample=False)
            return
        loop.create_task(self._cancel_order(trade))

    def action_cycle_aurora_preset(self) -> None:
        try:
            idx = self._AURORA_PRESET_ORDER.index(self._aurora_preset)
        except ValueError:
            idx = 1
        self._aurora_preset = self._AURORA_PRESET_ORDER[(idx + 1) % len(self._AURORA_PRESET_ORDER)]
        self._exec_status = f"Aurora preset: {self._aurora_preset.upper()}"
        self._render_details(sample=False)

    async def action_refresh_ticker(self) -> None:
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id, owner="details")
        self._ticker = await self._client.ensure_ticker(self._item.contract, owner="details")
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")
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

    @staticmethod
    def _clip(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) > width:
            if width <= 1:
                return "…"
            return text[: width - 1] + "…"
        return text.ljust(width)

    @staticmethod
    def _pnl_style(value: float | None) -> str:
        if value is None:
            return "dim"
        return "green" if value >= 0 else "red"

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(num):
            return None
        return float(num)

    def _live_unrealized(self, mark_price: float | None) -> float | None:
        mark = self._float_or_none(mark_price)
        if mark is None:
            return None
        qty = self._float_or_none(getattr(self._item, "position", None))
        if qty is None:
            return None
        if abs(float(qty)) <= 1e-12:
            return 0.0
        multiplier = _infer_multiplier(self._item)
        cost_basis = _cost_basis(self._item)
        return (float(mark) * float(qty) * float(multiplier)) - float(cost_basis)

    @staticmethod
    def _direction_glyph(value: float | None) -> Text:
        if value is None:
            return Text("•", style="dim")
        if value > 0:
            return Text("▲", style="bold green")
        if value < 0:
            return Text("▼", style="bold red")
        return Text("•", style="dim")

    def _position_beacon_row(self, qty: float, notional: float | None) -> Text:
        now = monotonic()
        if self._pos_prev_qty is None:
            self._pos_prev_qty = float(qty)
        elif float(qty) != float(self._pos_prev_qty):
            self._pos_delta = float(qty) - float(self._pos_prev_qty)
            self._pos_prev_qty = float(qty)
            self._pos_pulse_until = now + self._POSITION_PULSE_SEC

        direction = "FLAT"
        direction_style = "dim"
        if qty > 0:
            direction = "LONG"
            direction_style = "bold green"
        elif qty < 0:
            direction = "SHORT"
            direction_style = "bold red"

        qty_label = _fmt_qty(float(qty))
        if qty > 0:
            qty_label = f"+{qty_label}"

        row = Text("POS ", style="bold")
        row.append(f"{qty_label} sh")
        if notional is not None:
            signed = float(notional)
            sign = "+" if signed > 0 else "-" if signed < 0 else ""
            row.append(" (", style="dim")
            row.append(f"{sign}${abs(signed):,.0f}")
            row.append(")", style="dim")
        row.append("   ")
        row.append(direction, style=direction_style)

        pulse_active = now < float(self._pos_pulse_until)
        if pulse_active and self._pos_delta:
            delta = self._pos_delta
            delta_label = _fmt_qty(abs(delta))
            sign = "+" if delta > 0 else "-"
            row.append("   Δ ")
            row.append(f"{sign}{delta_label}", style="bold")
            row.append(" sh")
        if pulse_active:
            row.stylize("bold on #2f2f2f")
        return row

    @staticmethod
    def _panel_width(widget: Static, floor: int) -> int:
        size = int(getattr(widget.size, "width", 0) or 0)
        if size <= 0:
            return floor
        # Detail panes are padded by one char on each side in app CSS.
        return max(size - 2, 24)

    @staticmethod
    def _drawdown_from(samples: deque[float], current: float | None) -> float:
        if current is None:
            return 0.0
        if not samples:
            return 0.0
        peak = max(samples)
        return max(peak - current, 0.0)

    def _trim_tapes(self, now: float) -> None:
        cutoff = now - max(float(self._TREND_RETENTION_SEC), float(self._TREND_WINDOW_SEC))
        for tape in (
            self._mid_tape,
            self._size_tape,
            self._vol_flow_tape,
            self._imbalance_tape,
            self._vol_burst_tape,
        ):
            while tape and tape[0][0] < cutoff:
                tape.popleft()

    def _record_mid_sample(self, mid: float, *, ts: float | None = None) -> None:
        now = monotonic() if ts is None else float(ts)
        mid_value = float(mid)
        if self._mid_tape:
            _last_ts, last_mid = self._mid_tape[-1]
            tick = _tick_size(self._item.contract, self._ticker, mid_value)
            eps = max(float(tick) * 0.01, 1e-9)
            if abs(mid_value - float(last_mid)) <= eps:
                self._mid_tape[-1] = (now, float(last_mid))
                self._trim_tapes(now)
                return
        self._mid_samples.append(mid_value)
        self._mid_tape.append((now, mid_value))
        self._trim_tapes(now)

    def _record_aurora_sample(
        self,
        *,
        imbalance: float | None,
        vol_burst: float | None,
        ts: float | None = None,
    ) -> None:
        now = monotonic() if ts is None else float(ts)
        if imbalance is not None:
            imbalance_value = max(min(float(imbalance), 1.0), -1.0)
            self._imbalance_samples.append(imbalance_value)
            self._imbalance_tape.append((now, imbalance_value))
        if vol_burst is not None and vol_burst >= 0:
            burst_value = float(vol_burst)
            self._vol_burst_samples.append(burst_value)
            self._vol_burst_tape.append((now, burst_value))
        self._trim_tapes(now)

    def _record_volume_size(self, size: float | None, *, ts: float | None = None) -> None:
        if size is None or size < 0:
            return
        now = monotonic() if ts is None else float(ts)
        size_value = float(size)
        self._size_samples.append(size_value)
        self._size_tape.append((now, size_value))
        self._trim_tapes(now)

    def _record_volume_flow(self, flow: float | None, *, ts: float | None = None) -> None:
        if flow is None:
            return
        flow_value = float(flow)
        if abs(flow_value) <= 1e-12:
            return
        now = monotonic() if ts is None else float(ts)
        self._vol_flow_tape.append((now, flow_value))
        self._trim_tapes(now)

    def _trade_volume_delta(self, ticker: Ticker) -> float | None:
        if self._flow_cum_attr:
            value = _safe_num(getattr(ticker, self._flow_cum_attr, None))
            if value is None or value < 0:
                self._flow_cum_attr = None
                self._flow_cum_prev = None
                return None
            prev = self._flow_cum_prev
            self._flow_cum_prev = float(value)
            if prev is None:
                return None
            delta = float(value) - float(prev)
            if delta < 0:
                return None
            return delta

        for attr in ("rtTradeVolume", "rtVolume", "volume"):
            value = _safe_num(getattr(ticker, attr, None))
            if value is None or value < 0:
                continue
            self._flow_cum_attr = attr
            self._flow_cum_prev = float(value)
            return None
        return None

    def _trade_volume_fallback(self, *, last_size: float | None, last_price: float | None) -> float | None:
        if last_size is None or last_size <= 0:
            return None
        prev_size = self._flow_last_size
        prev_price = self._flow_last_price
        self._flow_last_size = float(last_size)
        self._flow_last_price = float(last_price) if last_price is not None else None
        if prev_size is None:
            return None
        price_changed = (
            last_price is not None
            and prev_price is not None
            and abs(float(last_price) - float(prev_price)) > 1e-9
        )
        if abs(float(last_size) - float(prev_size)) <= 1e-9 and not price_changed:
            return None
        return float(last_size)

    def _flow_direction(self, *, price: float | None, imbalance: float | None) -> float:
        if price is not None:
            current = float(price)
            prev = self._flow_prev_price
            self._flow_prev_price = current
            if prev is not None:
                tick = _tick_size(self._item.contract, self._ticker, current)
                eps = max(float(tick) * 0.01, 1e-9)
                delta = current - float(prev)
                if delta > eps:
                    return 1.0
                if delta < -eps:
                    return -1.0
        if imbalance is not None and abs(float(imbalance)) >= 0.05:
            return 1.0 if float(imbalance) > 0 else -1.0
        if imbalance is not None:
            return 1.0 if float(imbalance) >= 0 else -1.0
        return 1.0

    def _capture_tick_mid(self) -> None:
        ticker = self._ticker
        if ticker is None:
            return
        bid = self._quote_num(getattr(ticker, "bid", None))
        ask = self._quote_num(getattr(ticker, "ask", None))
        last = self._quote_num(getattr(ticker, "last", None))
        last_size = _safe_num(getattr(ticker, "lastSize", None))
        bid_size = _safe_num(getattr(ticker, "bidSize", None))
        ask_size = _safe_num(getattr(ticker, "askSize", None))
        rt_trade_volume = _safe_num(getattr(ticker, "rtTradeVolume", None))
        rt_volume = _safe_num(getattr(ticker, "rtVolume", None))
        total_volume = _safe_num(getattr(ticker, "volume", None))
        signature = (
            bid,
            ask,
            last,
            last_size,
            bid_size,
            ask_size,
            rt_trade_volume,
            rt_volume,
            total_volume,
        )
        if signature == self._last_tick_signature:
            self._render_details_if_mounted(sample=False)
            return
        self._last_tick_signature = signature

        mid = _midpoint(bid, ask)
        if mid is None:
            mid = _ticker_price(ticker)
        if mid is None:
            mid = _mark_price(self._item)
        if mid is None:
            return
        now = monotonic()
        prev_mid = float(self._mid_tape[-1][1]) if self._mid_tape else None
        mid_value = float(mid)
        self._record_mid_sample(mid_value, ts=now)
        self._record_volume_size(last_size, ts=now)

        imbalance = None
        total_size = float((bid_size or 0.0) + (ask_size or 0.0))
        if bid_size is not None or ask_size is not None:
            imbalance = (
                float((bid_size or 0.0) - (ask_size or 0.0)) / total_size if total_size > 0 else 0.0
            )
        vol_burst = abs(mid_value - prev_mid) if prev_mid is not None else 0.0
        self._record_aurora_sample(imbalance=imbalance, vol_burst=vol_burst, ts=now)

        flow_qty = self._trade_volume_delta(ticker)
        if flow_qty is None:
            flow_qty = self._trade_volume_fallback(last_size=last_size, last_price=last)
        if flow_qty is not None and flow_qty > 0:
            direction = self._flow_direction(price=last or mid_value, imbalance=imbalance)
            self._record_volume_flow(flow_qty * direction, ts=now)
        self._render_details_if_mounted(sample=False)

    def _record_market_samples(
        self,
        *,
        mid: float | None,
        spread: float | None,
        size: float | None,
        pnl: float | None,
        slip_proxy: float | None,
        imbalance: float | None,
        vol_burst: float | None,
    ) -> None:
        now = monotonic()
        if mid is not None:
            self._record_mid_sample(float(mid), ts=now)
        if spread is not None and spread >= 0:
            self._spread_samples.append(float(spread))
        self._record_volume_size(size, ts=now)
        if pnl is not None:
            self._pnl_samples.append(float(pnl))
        if slip_proxy is not None and slip_proxy >= 0:
            self._slip_proxy_samples.append(float(slip_proxy))
        self._record_aurora_sample(imbalance=imbalance, vol_burst=vol_burst, ts=now)

    def _sparkline(self, values: deque[float], width: int) -> str:
        width = max(width, 1)
        points = list(values)[-width:]
        if not points:
            return " " * width
        lo = min(points)
        hi = max(points)
        span = hi - lo
        if span <= 1e-12:
            bar = self._SPARK_CHARS[len(self._SPARK_CHARS) // 2]
            return (bar * len(points)).rjust(width)
        out: list[str] = []
        scale = len(self._SPARK_CHARS) - 1
        for value in points:
            ratio = (value - lo) / span
            idx = max(0, min(scale, int(round(ratio * scale))))
            out.append(self._SPARK_CHARS[idx])
        return "".join(out).rjust(width)

    @staticmethod
    def _resample(points: list[float], width: int) -> list[float]:
        width = max(width, 1)
        if not points:
            return []
        if len(points) == 1 or width == 1:
            return [points[-1]] * width
        if len(points) == width:
            return list(points)
        step = float(len(points) - 1) / float(width - 1)
        out: list[float] = []
        for idx in range(width):
            pos = step * float(idx)
            lo = int(pos)
            hi = min(lo + 1, len(points) - 1)
            frac = pos - float(lo)
            out.append(points[lo] + (points[hi] - points[lo]) * frac)
        return out

    def _trend_window_bounds(self) -> tuple[float, float]:
        end = monotonic()
        start = end - float(self._TREND_WINDOW_SEC)
        return start, end

    @staticmethod
    def _tape_series(
        tape: deque[tuple[float, float]], *, width: int, start: float, end: float
    ) -> list[float]:
        width = max(width, 1)
        if not tape:
            return []
        points = list(tape)
        if not points:
            return []
        start_idx = 0
        while start_idx < len(points) and points[start_idx][0] < start:
            start_idx += 1
        if start_idx > 0:
            window = [points[start_idx - 1], *points[start_idx:]]
        else:
            window = points
        if not window:
            return []
        span = max(float(end - start), 1e-9)
        step = span / float(width)
        cursor = 0
        out: list[float] = []
        for idx in range(width):
            target = float(start) + (step * float(idx + 1))
            while cursor + 1 < len(window) and float(window[cursor + 1][0]) <= target:
                cursor += 1
            t0, v0 = window[cursor]
            if cursor + 1 < len(window):
                t1, v1 = window[cursor + 1]
                if t1 > t0 and t0 <= target <= t1:
                    frac = (target - t0) / (t1 - t0)
                    out.append(float(v0) + ((float(v1) - float(v0)) * frac))
                    continue
            out.append(float(v0))
        return out

    @staticmethod
    def _tape_bin_max(
        tape: deque[tuple[float, float]], *, width: int, start: float, end: float
    ) -> list[float]:
        width = max(width, 1)
        if not tape:
            return []
        span = max(float(end - start), 1e-9)
        out = [0.0] * width
        has_data = False
        for ts, value in tape:
            if ts < start or ts > end:
                continue
            ratio = (float(ts) - float(start)) / span
            idx = max(0, min(width - 1, int(ratio * float(width))))
            out[idx] = max(out[idx], float(value))
            has_data = True
        return out if has_data else []

    @staticmethod
    def _tape_bin_sum(
        tape: deque[tuple[float, float]], *, width: int, start: float, end: float
    ) -> list[float]:
        width = max(width, 1)
        if not tape:
            return []
        span = max(float(end - start), 1e-9)
        out = [0.0] * width
        has_data = False
        for ts, value in tape:
            if ts < start or ts > end:
                continue
            ratio = (float(ts) - float(start)) / span
            idx = max(0, min(width - 1, int(ratio * float(width))))
            out[idx] += float(value)
            has_data = True
        return out if has_data else []

    def _sparkline_smooth(self, values: deque[float], width: int) -> str:
        width = max(width, 1)
        points = self._resample(list(values), width)
        if not points:
            return " " * width
        lo = min(points)
        hi = max(points)
        span = hi - lo
        if span <= 1e-12:
            base = self._SPARK_LEVELS[4]
            return (base * len(points)).rjust(width)
        out: list[str] = []
        error = 0.0
        for value in points:
            ratio = (value - lo) / span
            level = ratio * 8.0 + error
            idx = int(level)
            if idx < 0:
                idx = 0
            elif idx > 8:
                idx = 8
            error = level - float(idx)
            out.append(self._SPARK_LEVELS[idx])
        return "".join(out).rjust(width)

    @staticmethod
    def _trend_braille_cell(left: int, right: int, *, top: bool) -> str:
        if left <= 0 and right <= 0:
            return " "
        # Braille gives two 4-dot columns per char (left/right), ideal for 2x supersampling.
        left_order = (64, 4, 2, 1) if top else (1, 2, 4, 64)
        right_order = (128, 32, 16, 8) if top else (8, 16, 32, 128)
        bits = 0
        for idx in range(max(0, min(4, left))):
            bits |= left_order[idx]
        for idx in range(max(0, min(4, right))):
            bits |= right_order[idx]
        return chr(0x2800 + bits) if bits else " "

    def _trend_continuity_rows(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> tuple[list[str], int]:
        width = max(width, 1)
        cols = width
        rows = max(3, int(self._TREND_ROWS))
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self._trend_window_bounds()
        )
        series = self._tape_series(self._mid_tape, width=width, start=start, end=end)
        if not series:
            approx = max(2, int(round(self._TREND_WINDOW_SEC / max(self._refresh_sec, 0.1))))
            fallback = list(self._mid_samples)[-approx:]
            series = self._resample([float(value) for value in fallback], width) if fallback else []
        if not series:
            blank = " " * width
            return ([blank for _ in range(rows)], rows // 2)
        self._trend_bins = list(series)
        lo = min(series)
        hi = max(series)
        span = hi - lo
        tick = _tick_size(self._item.contract, self._ticker, series[-1])
        min_span = max(float(tick) * 2.0, 1e-9)
        if span < min_span:
            center = (hi + lo) * 0.5
            lo = center - (min_span * 0.5)
            hi = center + (min_span * 0.5)
            span = hi - lo
        pad = span * 0.03
        lo -= pad
        hi += pad
        span = max(hi - lo, min_span)

        hi_rows = rows * int(self._TREND_BRAILLE_Y)
        hi_cols = cols * int(self._TREND_BRAILLE_X)
        raster: list[list[bool]] = [[False] * hi_cols for _ in range(hi_rows)]
        y_points: list[int] = []
        for value in series:
            ratio = (float(value) - lo) / span
            ratio = max(0.0, min(ratio, 1.0))
            y_points.append(int(round((1.0 - ratio) * float(hi_rows - 1))))

        def draw_segment(x0: int, y0: int, x1: int, y1: int) -> None:
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x = x0
            y = y0
            while True:
                if 0 <= y < hi_rows and 0 <= x < hi_cols:
                    raster[y][x] = True
                if x == x1 and y == y1:
                    break
                e2 = err * 2
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        for idx in range(1, cols):
            x0 = (idx - 1) * int(self._TREND_BRAILLE_X)
            x1 = idx * int(self._TREND_BRAILLE_X)
            draw_segment(x0, y_points[idx - 1], x1, y_points[idx])
        raster[y_points[-1]][hi_cols - 1] = True

        out_rows: list[str] = []
        for row_idx in range(rows):
            top = row_idx * int(self._TREND_BRAILLE_Y)
            chars: list[str] = []
            for col_idx in range(cols):
                left = col_idx * int(self._TREND_BRAILLE_X)
                bits = 0
                if raster[top + 0][left + 0]:
                    bits |= 0x01
                if raster[top + 1][left + 0]:
                    bits |= 0x02
                if raster[top + 2][left + 0]:
                    bits |= 0x04
                if raster[top + 3][left + 0]:
                    bits |= 0x40
                if raster[top + 0][left + 1]:
                    bits |= 0x08
                if raster[top + 1][left + 1]:
                    bits |= 0x10
                if raster[top + 2][left + 1]:
                    bits |= 0x20
                if raster[top + 3][left + 1]:
                    bits |= 0x80
                chars.append(chr(0x2800 + bits) if bits else " ")
            out_rows.append("".join(chars))

        now_row = max(0, min(rows - 1, int(y_points[-1] // int(self._TREND_BRAILLE_Y))))
        return (out_rows, now_row)

    @staticmethod
    def _trend_with_price_tag(row: Text, price: float | None, *, color: str) -> Text:
        if price is None:
            return row
        plain = row.plain
        width = len(plain)
        if width < 4:
            return row
        label = f" {_fmt_quote(price)}"
        usable = min(len(label), width - 1)
        start = width - 1 - usable
        out = row[:start]
        out.append(label[-usable:], style="bold #f8fbff")
        out.append("▐", style=f"bold {color}")
        return out

    def _vol_histogram_braille(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> Text:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self._trend_window_bounds()
        )
        flow_events = sum(1 for ts, _value in self._vol_flow_tape if start <= ts <= end)
        density = float(flow_events) / float(width)
        micro_factor = 2
        if density >= 1.2:
            micro_factor = 3
        if density >= 2.2:
            micro_factor = 4
        if density >= 3.5:
            micro_factor = 5
        micro_width = width * micro_factor
        flow_micro = self._tape_bin_sum(self._vol_flow_tape, width=micro_width, start=start, end=end)

        if not flow_micro:
            size_events = sum(1 for ts, _value in self._size_tape if start <= ts <= end)
            size_density = float(size_events) / float(width)
            if size_density >= 1.2:
                micro_factor = max(micro_factor, 3)
            if size_density >= 2.2:
                micro_factor = max(micro_factor, 4)
            if size_density >= 3.5:
                micro_factor = max(micro_factor, 5)
            micro_width = width * micro_factor
            size_micro = self._tape_series(self._size_tape, width=micro_width, start=start, end=end)
            if not size_micro:
                size_micro = self._resample(list(self._size_samples), micro_width)
            if not size_micro:
                return Text(" " * width, style="dim")
            imbalance_micro = self._tape_series(
                self._imbalance_tape, width=micro_width, start=start, end=end
            )
            if not imbalance_micro:
                imbalance_micro = [0.0] * micro_width
            flow_micro = []
            for size, imbalance in zip(size_micro, imbalance_micro):
                direction = 1.0 if float(imbalance) >= 0 else -1.0
                flow_micro.append(float(size) * direction)

        if len(flow_micro) != micro_width:
            flow_micro = self._resample([float(value) for value in flow_micro], micro_width)
        if len(flow_micro) > 2:
            smoothed = [float(flow_micro[0])]
            for idx in range(1, len(flow_micro) - 1):
                left = float(flow_micro[idx - 1])
                center = float(flow_micro[idx])
                right = float(flow_micro[idx + 1])
                blended = (left * 0.18) + (center * 0.64) + (right * 0.18)
                lo = min(left, center, right)
                hi = max(left, center, right)
                smoothed.append(max(lo, min(blended, hi)))
            smoothed.append(float(flow_micro[-1]))
            flow_micro = smoothed

        mags = sorted(abs(float(value)) for value in flow_micro if abs(float(value)) > 1e-12)
        if not mags:
            return Text(" " * width, style="dim")

        def quantile(ratio: float) -> float:
            ratio = max(0.0, min(float(ratio), 1.0))
            if len(mags) == 1:
                return mags[0]
            pos = ratio * float(len(mags) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(mags) - 1)
            frac = pos - float(lo)
            return mags[lo] + ((mags[hi] - mags[lo]) * frac)

        p25 = quantile(0.25)
        p90 = quantile(0.90)
        p98 = quantile(0.98)
        scale = max(p90, p98 * 0.72, mags[-1] * 0.26, 1e-9)
        floor_ratio = max(0.012, min((p25 / scale) * 0.60, 0.18))
        gamma = 0.84 if self._aurora_preset == "feral" else 0.90

        out = Text()
        base_split = max(1, micro_factor // 2)
        for idx in range(width):
            start_idx = idx * micro_factor
            chunk = flow_micro[start_idx : start_idx + micro_factor]
            if not chunk:
                out.append(" ", style="dim")
                continue
            if len(chunk) == 1:
                left_mag = abs(float(chunk[0]))
                right_mag = left_mag
            else:
                split = max(1, min(len(chunk) - 1, base_split))
                left_slice = chunk[:split]
                right_slice = chunk[split:]
                left_mag = sum(abs(float(value)) for value in left_slice) / float(len(left_slice))
                right_mag = sum(abs(float(value)) for value in right_slice) / float(len(right_slice))
            left_ratio = min(left_mag / scale, 1.0)
            right_ratio = min(right_mag / scale, 1.0)
            if left_ratio > 1e-12:
                left_ratio = max(left_ratio, floor_ratio)
            if right_ratio > 1e-12:
                right_ratio = max(right_ratio, floor_ratio)
            left_level = max(0, min(4, int(round(pow(left_ratio, gamma) * 4.0))))
            right_level = max(0, min(4, int(round(pow(right_ratio, gamma) * 4.0))))
            if left_level == 0 and left_ratio > 1e-12:
                left_level = 1
            if right_level == 0 and right_ratio > 1e-12:
                right_level = 1
            cell = self._trend_braille_cell(left_level, right_level, top=True)
            if cell == " ":
                out.append(" ", style="dim")
                continue
            intensity = max(left_level, right_level)
            style_idx = max(0, min(len(self._VOL_MAGENTA_STYLES) - 1, intensity))
            out.append(cell, style=self._VOL_MAGENTA_STYLES[style_idx])
        return out

    @staticmethod
    def _mark_now(text: Text, *, style: str = "bold #f8fbff") -> Text:
        plain = text.plain
        if not plain:
            return text
        marked = text[:-1]
        marked.append("▏", style=style)
        return marked

    def _aurora_config(self) -> dict[str, float]:
        return self._AURORA_PRESETS.get(self._aurora_preset, self._AURORA_PRESETS["normal"])

    @staticmethod
    def _aurora_pressure(imbalance: float) -> float:
        x = max(-1.0, min(float(imbalance), 1.0))
        mag = abs(x)
        if mag <= 1e-12:
            return 0.0
        # Lift mid-range imbalance so real pressure is easier to see in color.
        boosted = pow(mag, 0.72)
        return max(-1.0, min(boosted if x >= 0 else -boosted, 1.0))

    def _aurora_drift_series(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> list[float]:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self._trend_window_bounds()
        )
        mids = self._tape_series(self._mid_tape, width=width + 1, start=start, end=end)
        if len(mids) < 2:
            mids = self._resample(list(self._mid_samples), width + 1)
        if len(mids) < 2:
            return [0.0] * width
        deltas = [float(mids[idx] - mids[idx - 1]) for idx in range(1, len(mids))]
        mags = sorted(abs(delta) for delta in deltas if abs(delta) > 1e-12)
        if not mags:
            return [0.0] * width
        last = len(mags) - 1
        scale = max(mags[int(round(last * 0.85))], mags[-1] * 0.22, 1e-12)
        norm = [max(-1.0, min(delta / scale, 1.0)) for delta in deltas]
        alpha = 0.36
        smooth: list[float] = [norm[0]]
        for value in norm[1:]:
            prev = smooth[-1]
            smooth.append(prev + alpha * (value - prev))
        return self._resample(smooth, width) if smooth else [0.0] * width

    def _aurora_blended_imbalance(self, imbalance: float, drift: float) -> float:
        bias = max(0.0, min(self._AURORA_TAPE_BIAS, 1.0))
        return max(-1.0, min((float(imbalance) * (1.0 - bias)) + (float(drift) * bias), 1.0))

    def _aurora_style(self, imbalance: float, *, config: dict[str, float] | None = None) -> str:
        cfg = config or self._aurora_config()
        pressure = self._aurora_pressure(imbalance)
        buy_soft = float(cfg.get("buy_soft", 0.22))
        buy_strong = float(cfg.get("buy_strong", 0.48))
        sell_soft = float(cfg.get("sell_soft", -0.22))
        sell_strong = float(cfg.get("sell_strong", -0.48))
        if pressure >= buy_strong:
            return "#18a63f"  # lots buy pressure (dark green, brightened for terminal contrast)
        if pressure >= buy_soft:
            return "#e6d84e"  # slight buy pressure (yellow)
        if pressure <= sell_strong:
            return "red"  # lots sell pressure (red)
        if pressure <= sell_soft:
            return "#ffaf00"  # slight sell pressure (amber)
        return "#8aa0b6"

    def _aurora_now_style(self) -> str:
        if self._imbalance_tape:
            imbalance_now = float(self._imbalance_tape[-1][1])
        elif self._imbalance_samples:
            imbalance_now = float(self._imbalance_samples[-1])
        else:
            return "#8aa0b6"
        drift_now = self._aurora_drift_series(1)[0]
        blended = self._aurora_blended_imbalance(imbalance_now, drift_now)
        return self._aurora_style(blended)

    def _aurora_strip(
        self,
        width: int,
        *,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> Text:
        width = max(width, 1)
        start, end = (
            (float(window_start), float(window_end))
            if window_start is not None and window_end is not None
            else self._trend_window_bounds()
        )
        imbalances = self._tape_series(self._imbalance_tape, width=width, start=start, end=end)
        if not imbalances:
            imbalances = self._resample(list(self._imbalance_samples), width)
        if not imbalances:
            imbalances = [0.0] * width
        elif len(imbalances) != width:
            imbalances = self._resample(imbalances, width)

        drifts = self._aurora_drift_series(width, window_start=start, window_end=end)
        if not drifts:
            drifts = [0.0] * width
        elif len(drifts) != width:
            drifts = self._resample(drifts, width)

        bursts = self._tape_bin_max(self._vol_burst_tape, width=width, start=start, end=end)
        if not bursts:
            bursts = self._resample(list(self._vol_burst_samples), width)
        if not bursts:
            bursts = [0.0] * width
        elif len(bursts) != width:
            bursts = self._resample(bursts, width)

        config = self._aurora_config()
        top_burst = max(bursts) if bursts else 0.0
        if top_burst <= 1e-12:
            top_burst = 1.0
        strip = Text()
        for imbalance, drift, burst in zip(imbalances, drifts, bursts):
            gain = float(config.get("burst_gain", 1.0))
            ratio = max(0.0, min((float(burst) / top_burst) * gain, 1.0))
            char = self._SPARK_LEVELS[int(round(ratio * 8.0))]
            blended = self._aurora_blended_imbalance(float(imbalance), float(drift))
            style = self._aurora_style(blended, config=config)
            strip.append(char, style=style)
        return strip

    def _render_aurora_legend(self) -> Text:
        legend = Text(f"Aurora[{self._aurora_preset}]  ")
        legend.append("BUY+", style="#18a63f")
        legend.append("/", style="dim")
        legend.append("BUY", style="#e6d84e")
        legend.append(" -> ", style="dim")
        legend.append("NEUTRAL", style="#8aa0b6")
        legend.append(" -> ", style="dim")
        legend.append("SELL", style="#ffaf00")
        legend.append("/", style="dim")
        legend.append("SELL+", style="red")
        legend.append("  |  +tape 15%  |  height=vol burst  |  a preset", style="dim")
        return legend

    def _momentum_line(self, width: int) -> str:
        width = max(width, 1)
        mids = list(self._mid_samples)[-(width + 1) :]
        if len(mids) < 2:
            return self._MOMENTUM_CHARS[0] * width
        deltas = [mids[idx] - mids[idx - 1] for idx in range(1, len(mids))]
        scale = max(abs(delta) for delta in deltas) or 1.0
        levels = len(self._MOMENTUM_CHARS) - 1
        chars: list[str] = []
        for delta in deltas[-width:]:
            ratio = min(abs(delta) / scale, 1.0)
            idx = max(0, min(levels, int(round(ratio * levels))))
            chars.append(self._MOMENTUM_CHARS[idx])
        return "".join(chars).rjust(width, self._MOMENTUM_CHARS[0])

    def _meter(self, ratio: float | None, width: int) -> str:
        width = max(width, 1)
        if ratio is None:
            return self._BAR_EMPTY * width
        bounded = min(max(float(ratio), 0.0), 1.0)
        filled = max(0, min(width, int(round(bounded * width))))
        return (self._BAR_FILL * filled) + (self._BAR_EMPTY * (width - filled))

    @staticmethod
    def _exec_mode_label(mode: str) -> str:
        if mode == "AUTO":
            return "AUTO"
        if mode == "OPTIMISTIC":
            return "OPT"
        if mode == "AGGRESSIVE":
            return "AGG"
        return mode

    def _selected_exec_mode(self) -> str:
        selected = self._exec_rows[self._exec_selected]
        if selected == "optimistic":
            return "OPTIMISTIC"
        if selected == "mid":
            return "MID"
        if selected == "aggressive":
            return "AGGRESSIVE"
        if selected == "cross":
            return "CROSS"
        return "AUTO"

    def _box_top(self, title: str, inner_width: int, *, style: str) -> Text:
        label = f" {title} "
        if len(label) > inner_width:
            label = self._clip(label, inner_width)
        line = Text("┌", style=style)
        line.append(label, style="bold")
        line.append("─" * max(inner_width - len(label), 0), style=style)
        line.append("┐", style=style)
        return line

    def _box_rule(self, title: str, inner_width: int, *, style: str) -> Text:
        label = f" {title} "
        if len(label) > inner_width:
            label = self._clip(label, inner_width)
        line = Text("├", style=style)
        line.append(label, style="bold")
        line.append("─" * max(inner_width - len(label), 0), style=style)
        line.append("┤", style=style)
        return line

    def _box_row(self, content: Text | str, inner_width: int, *, style: str) -> Text:
        if isinstance(content, Text):
            row = content.copy()
        else:
            row = Text(str(content))
        plain = row.plain
        if len(plain) > inner_width:
            row = Text(self._clip(plain, inner_width))
            plain = row.plain
        padded = inner_width - len(plain)
        line = Text("│", style=style)
        line.append_text(row)
        if padded > 0:
            line.append(" " * padded)
        line.append("│", style=style)
        return line

    def _box_bottom(self, inner_width: int, *, style: str) -> Text:
        return Text("└" + ("─" * inner_width) + "┘", style=style)

    def _render_details(self, *, sample: bool = True) -> None:
        if hasattr(self, "_detail_legend"):
            self._detail_legend.update(self._render_aurora_legend())
        try:
            con_id = int(self._item.contract.conId or 0)
            latest = self._client.portfolio_item(con_id) if con_id else None
            if latest is not None:
                self._item = latest
            elif con_id and float(getattr(self._item, "position", 0.0) or 0.0) != 0.0:
                # Filled-to-flat positions disappear from portfolio(); don't keep stale nonzero qty.
                self._item = self._flat_item_snapshot(self._item)
            contract = self._item.contract
            bid = self._quote_num(self._ticker.bid) if self._ticker else None
            ask = self._quote_num(self._ticker.ask) if self._ticker else None
            last = self._quote_num(self._ticker.last) if self._ticker else None
            if contract.secType in ("OPT", "FOP"):
                price = _option_display_price(self._item, self._ticker)
            else:
                price = _ticker_price(self._ticker) if self._ticker else None
            mid = _midpoint(bid, ask)
            close = _ticker_close(self._ticker) if self._ticker else None
            mark = _mark_price(self._item)
            live_mark = price or mark
            spread = (ask - bid) if bid is not None and ask is not None and ask >= bid else None
            size = _safe_num(getattr(self._ticker, "lastSize", None)) if self._ticker else None
            bid_size = _safe_num(getattr(self._ticker, "bidSize", None)) if self._ticker else None
            ask_size = _safe_num(getattr(self._ticker, "askSize", None)) if self._ticker else None
            broker_pnl, _ = _unrealized_pnl_values(self._item, mark_price=live_mark)
            fast_pnl = self._live_unrealized(live_mark)
            pnl_value = fast_pnl if fast_pnl is not None else broker_pnl
            ref_price = last or live_mark
            slip_proxy = (
                abs(ref_price - mid)
                if ref_price is not None and mid is not None
                else None
            )
            if sample:
                prev_mid = self._mid_samples[-1] if self._mid_samples else None
                imbalance = None
                if bid_size is not None or ask_size is not None:
                    total_size = float((bid_size or 0.0) + (ask_size or 0.0))
                    if total_size > 0:
                        imbalance = float((bid_size or 0.0) - (ask_size or 0.0)) / total_size
                    else:
                        imbalance = 0.0
                vol_burst = (
                    abs(ref_price - prev_mid)
                    if ref_price is not None and prev_mid is not None
                    else 0.0
                )
                self._record_market_samples(
                    mid=mid or price or mark,
                    spread=spread,
                    size=size,
                    pnl=pnl_value,
                    slip_proxy=slip_proxy,
                    imbalance=imbalance,
                    vol_burst=vol_burst,
                )

            panel_width = self._panel_width(self._detail_left, floor=58)
            lines = self._render_market_hud(
                panel_width=panel_width,
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
            lines.extend(self._render_execution_block(panel_width=panel_width))
            self._detail_left.update(Text("\n").join(lines))
        except Exception as exc:
            self._detail_left.update(Text(f"Detail render error: {exc}", style="red"))
        try:
            self._render_orders_panel()
        except Exception as exc:
            self._detail_right.update(Text(f"Orders render error: {exc}", style="red"))

    def _render_market_hud(
        self,
        *,
        panel_width: int,
        bid: float | None,
        ask: float | None,
        last: float | None,
        price: float | None,
        mid: float | None,
        close: float | None,
        mark: float | None,
        spread: float | None,
    ) -> list[Text]:
        contract = self._item.contract
        inner = max(panel_width - 2, 24)
        spark_width = inner
        official_unreal = self._float_or_none(getattr(self._item, "unrealizedPNL", None))
        fast_unreal = self._live_unrealized(price or mark)
        pnl_value = official_unreal if official_unreal is not None else fast_unreal
        official_label = _fmt_money(official_unreal) if official_unreal is not None else "n/a"
        fast_label = _fmt_money(fast_unreal) if fast_unreal is not None else "n/a"
        position_qty = float(self._item.position or 0.0)
        avg_cost = (
            _fmt_money(float(self._item.averageCost))
            if self._item.averageCost is not None
            else "n/a"
        )
        market_value_raw = _safe_num(self._item.marketValue)
        market_value = (
            _fmt_money(float(market_value_raw))
            if market_value_raw is not None
            else "n/a"
        )
        realized = (
            _fmt_money(float(self._item.realizedPNL))
            if self._item.realizedPNL is not None
            else "n/a"
        )

        md_row = Text("MD: ")
        if self._ticker:
            md_exchange = getattr(self._ticker.contract, "exchange", "") or "n/a"
            md_label = _market_data_label(self._ticker)
            md_row.append(f"{md_exchange} ({md_label})", style="bright_cyan")
        else:
            md_row.append("n/a", style="dim")
        quote_status = (
            _quote_status_line(self._ticker)
            if self._ticker
            else Text("MD Quotes: n/a", style="dim")
        )
        has_live_quote = bool(
            (bid is not None and ask is not None and bid <= ask)
            or (last is not None)
        )
        close_only_badge_row: Text | None = None
        if self._ticker:
            md_type = getattr(self._ticker, "marketDataType", None)
            is_delayed = md_type in (3, 4)
            if is_delayed and (not has_live_quote) and close is not None and close > 0:
                close_only_badge_row = Text("CLOSE-ONLY DELAYED FEED", style="bold black on yellow")
        no_quote_badge_row: Text | None = None
        if contract.secType == "OPT" and not has_live_quote:
            no_quote_badge_row = Text("NO ACTIONABLE OPTION QUOTE YET", style="bold black on yellow")

        quote_row = Text("Bid ")
        quote_row.append(_fmt_quote(bid), style="green")
        quote_row.append("  Ask ")
        quote_row.append(_fmt_quote(ask), style="red")
        quote_row.append("  Last ")
        quote_row.append(_fmt_quote(last), style="bright_white")

        price_row = Text("Price ")
        if price is None and close is not None:
            price_row.append(f"Closed ({_fmt_quote(close)})", style="red")
        elif price is None and mark is not None:
            price_row.append(f"Mark ({_fmt_quote(mark)})", style="yellow")
        else:
            price_row.append(_fmt_quote(price), style="bright_white")

        headline = Text("MID ")
        headline.append(_fmt_quote(mid or price or mark), style="bright_cyan")
        headline.append("   SPRD ")
        headline.append(_fmt_quote(spread), style="cyan")
        position_row = self._position_beacon_row(position_qty, market_value_raw)

        trend_window_start, trend_window_end = self._trend_window_bounds()
        aurora_label_row = Text("Aurora", style="#8aa0b6")
        aurora_row = self._mark_now(
            self._aurora_strip(
                spark_width,
                window_start=trend_window_start,
                window_end=trend_window_end,
            )
        )
        trend_label_row = Text("1m Trend", style="cyan")
        trend_row_values, trend_now_row = self._trend_continuity_rows(
            spark_width,
            window_start=trend_window_start,
            window_end=trend_window_end,
        )
        comet_color = self._aurora_now_style()
        trend_rows = [Text(row, style="#63d9ff") for row in trend_row_values]
        trend_price = mid or price or mark
        trend_rows[trend_now_row] = self._trend_with_price_tag(
            trend_rows[trend_now_row], trend_price, color=comet_color
        )
        vol_label_row = Text("Vol Histogram", style="magenta")
        vol_row = self._mark_now(
            self._vol_histogram_braille(
                spark_width,
                window_start=trend_window_start,
                window_end=trend_window_end,
            )
        )
        momentum_label_row = Text("Momentum", style="yellow")
        momentum_row = self._mark_now(Text(self._momentum_line(spark_width), style="yellow"))

        detail_row = Text(f"Avg {avg_cost}   MktVal {market_value}")
        ref_price = mid or price or mark
        pct_baseline = close
        if (
            contract.secType in ("OPT", "FOP")
            and not has_live_quote
            and self._session_prev_close is not None
            and self._session_prev_close > 0
        ):
            pct_baseline = float(self._session_prev_close)
        pct24 = (
            ((float(ref_price) - float(pct_baseline)) / float(pct_baseline) * 100.0)
            if ref_price is not None and pct_baseline is not None and pct_baseline > 0
            else None
        )
        if pct24 is not None and position_qty < 0:
            pct24 *= -1.0
        pct24_prefix = self._direction_glyph(pct24)
        pct24_value = Text(" n/a", style="dim")
        if pct24 is not None:
            pct24_value = Text(f" {pct24:.2f}%")
            if pct24 > 0:
                pct24_value.stylize("green")
            elif pct24 < 0:
                pct24_value.stylize("red")
        pnl_prefix = "Unrealized "
        tail_row = Text("")
        tail_row.append_text(pct24_prefix)
        tail_row.append_text(pct24_value)
        tail_row.append("   ")
        tail_row.append(pnl_prefix)
        pnl_start = len(tail_row.plain)
        tail_row.append(official_label)
        pnl_end = len(tail_row.plain)
        tail_row.append(" (")
        est_start = len(tail_row.plain)
        tail_row.append(fast_label)
        est_end = len(tail_row.plain)
        tail_row.append(" est)")
        est_suffix_end = len(tail_row.plain)
        tail_row.append("   Realized ")
        tail_row.append(realized)
        tail_row.stylize(self._pnl_style(pnl_value), pnl_start, pnl_end)
        tail_row.stylize(self._pnl_style(fast_unreal), est_start, est_end)
        tail_row.stylize("dim", est_end, est_suffix_end)

        lines: list[Text] = [
            self._box_top(f"{contract.symbol} {contract.secType}", inner, style="#2d8fd5"),
            self._box_row(md_row, inner, style="#2d8fd5"),
            self._box_row(quote_status, inner, style="#2d8fd5"),
            self._box_row(headline, inner, style="#2d8fd5"),
            self._box_row(position_row, inner, style="#2d8fd5"),
            self._box_row(detail_row, inner, style="#2d8fd5"),
            self._box_row(tail_row, inner, style="#2d8fd5"),
            self._box_row(quote_row, inner, style="#2d8fd5"),
            self._box_row(price_row, inner, style="#2d8fd5"),
            self._box_row(aurora_label_row, inner, style="#2d8fd5"),
            self._box_row(aurora_row, inner, style="#2d8fd5"),
            self._box_row(trend_label_row, inner, style="#2d8fd5"),
            *[self._box_row(trend_row, inner, style="#2d8fd5") for trend_row in trend_rows],
            self._box_row(vol_label_row, inner, style="#2d8fd5"),
            self._box_row(vol_row, inner, style="#2d8fd5"),
            self._box_row(momentum_label_row, inner, style="#2d8fd5"),
            self._box_row(momentum_row, inner, style="#2d8fd5"),
        ]
        if no_quote_badge_row is not None:
            lines.insert(3, self._box_row(no_quote_badge_row, inner, style="#2d8fd5"))
        if close_only_badge_row is not None:
            lines.insert(4 if no_quote_badge_row is not None else 3, self._box_row(close_only_badge_row, inner, style="#2d8fd5"))
        if contract.lastTradeDateOrContractMonth:
            expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth)
            meta = Text(f"Expiry {expiry}")
            if contract.right:
                meta.append(f"  Right {contract.right}")
            if contract.strike:
                meta.append(f"  Strike {_fmt_money(contract.strike)}")
            lines.append(self._box_row(meta, inner, style="#2d8fd5"))
        if self._underlying_ticker:
            label = self._underlying_label or "Underlying"
            ubid = self._quote_num(self._underlying_ticker.bid)
            uask = self._quote_num(self._underlying_ticker.ask)
            ulast = self._quote_num(self._underlying_ticker.last)
            under_row = Text(f"{label}: ")
            under_row.append(f"{_fmt_quote(ubid)}/{_fmt_quote(uask)}/{_fmt_quote(ulast)}", style="cyan")
            lines.append(self._box_row(under_row, inner, style="#2d8fd5"))
        lines.append(self._box_bottom(inner, style="#2d8fd5"))
        return lines

    def _render_execution_block(self, *, panel_width: int) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP"):
            return []
        bid = self._quote_num(self._ticker.bid) if self._ticker else None
        ask = self._quote_num(self._ticker.ask) if self._ticker else None
        last = self._quote_num(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(contract, self._ticker, last_ref)
        mid_raw = _midpoint(bid, ask)
        fallback = _round_to_tick(last_ref, tick)
        mid = _round_to_tick(mid_raw, tick) or fallback
        optimistic_buy = _round_to_tick(_optimistic_price(bid, ask, mid_raw, "BUY"), tick) or mid
        optimistic_sell = _round_to_tick(_optimistic_price(bid, ask, mid_raw, "SELL"), tick) or mid
        aggressive_buy = _round_to_tick(_aggressive_price(bid, ask, mid_raw, "BUY"), tick) or mid
        aggressive_sell = _round_to_tick(_aggressive_price(bid, ask, mid_raw, "SELL"), tick) or mid
        cross_buy = _round_to_tick(ask, tick) if ask is not None else None
        cross_sell = _round_to_tick(bid, tick) if bid is not None else None
        ask_size = _safe_num(getattr(self._ticker, "askSize", None)) if self._ticker else None
        bid_size = _safe_num(getattr(self._ticker, "bidSize", None)) if self._ticker else None
        size_scale = max(1.0, ask_size or 0.0, bid_size or 0.0)
        qty = self._exec_qty
        inner = max(panel_width - 2, 24)
        depth_width = max(min(inner - 26, 18), 8)
        lines: list[Text] = [self._box_top("Execution Ladder", inner, style="#d4922f")]
        if self._exec_status:
            lines.append(self._box_row(Text(self._exec_status, style="yellow"), inner, style="#d4922f"))
        has_actionable_quote = bool(
            (bid is not None and ask is not None and bid <= ask)
            or (last is not None)
        )
        if contract.secType == "OPT" and not has_actionable_quote:
            lock_row = Text(
                "B/S locked: no actionable option quote yet (waiting for bid/ask/last)",
                style="bold black on yellow",
            )
            lines.append(self._box_row(lock_row, inner, style="#d4922f"))

        lines.append(self._box_rule("Depth", inner, style="#d4922f"))
        mid_size = ((ask_size or 0.0) + (bid_size or 0.0)) * 0.5
        ask_depth = self._meter((ask_size or 0.0) / size_scale, depth_width)
        mid_depth = self._meter(mid_size / size_scale, depth_width)
        bid_depth = self._meter((bid_size or 0.0) / size_scale, depth_width)
        lines.append(
            self._box_row(
                Text(
                    f"{_fmt_quote(cross_buy)} ask {_fmt_qty(ask_size or 0.0):>4} |{ask_depth}|",
                    style="red",
                ),
                inner,
                style="#d4922f",
            )
        )
        lines.append(
            self._box_row(
                Text(
                    f"{_fmt_quote(mid)} mid {_fmt_qty(mid_size):>4} |{mid_depth}|",
                    style="bright_white",
                ),
                inner,
                style="#d4922f",
            )
        )
        lines.append(
            self._box_row(
                Text(
                    f"{_fmt_quote(cross_sell)} bid {_fmt_qty(bid_size or 0.0):>4} |{bid_depth}|",
                    style="green",
                ),
                inner,
                style="#d4922f",
            )
        )

        lines.append(
            self._box_rule(
                f"Controls · tick {tick:.{_tick_decimals(tick)}f} · qty {qty}",
                inner,
                style="#d4922f",
            )
        )
        rows = [
            (
                "ladder",
                (
                    "Auto Ladder "
                    f"(OPT {_fmt_quote(optimistic_buy)}/{_fmt_quote(optimistic_sell)} "
                    f"-> MID {_fmt_quote(mid)} "
                    f"-> AGG {_fmt_quote(aggressive_buy)}/{_fmt_quote(aggressive_sell)} "
                    f"-> CROSS {_fmt_quote(cross_buy)}/{_fmt_quote(cross_sell)})"
                ),
            ),
            (
                "optimistic",
                f"OPT Only   B/S {_fmt_quote(optimistic_buy)} / {_fmt_quote(optimistic_sell)}",
            ),
            ("mid", f"MID Only   B/S {_fmt_quote(mid)} / {_fmt_quote(mid)}"),
            (
                "aggressive",
                f"AGG Only   B/S {_fmt_quote(aggressive_buy)} / {_fmt_quote(aggressive_sell)}",
            ),
            (
                "cross",
                f"CROSS Only B/S {_fmt_quote(cross_buy)} / {_fmt_quote(cross_sell)}",
            ),
            (
                "qty",
                f"Qty {qty}",
            ),
        ]
        for idx, (key, label) in enumerate(rows):
            row = Text(label)
            if self._active_panel == "exec" and idx == self._exec_selected:
                row.stylize("bold on #2b2b2b")
            lines.append(self._box_row(row, inner, style="#d4922f"))
        lines.append(self._box_bottom(inner, style="#d4922f"))
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
        panel_width = self._panel_width(self._detail_right, floor=44)
        inner = max(panel_width - 2, 24)
        available = int(getattr(self._detail_right.size, "height", 0) or 0)

        total_qty = sum(abs(float(trade.order.totalQuantity or 0.0)) for trade in trades)
        filled_qty = sum(abs(float(trade.orderStatus.filled or 0.0)) for trade in trades)
        fill_rate = (filled_qty / total_qty) if total_qty > 0 else 0.0
        cancel_like = 0
        replace_like = 0
        for trade in trades:
            status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").lower()
            if "cancel" in status:
                cancel_like += 1
            elif "pending" in status:
                replace_like += 1
        cancel_replace_rate = (
            float(cancel_like + replace_like) / float(len(trades))
            if trades
            else 0.0
        )
        fill_meter = self._meter(fill_rate, 8)
        cancel_meter = self._meter(cancel_replace_rate, 8)
        slip_spark = self._sparkline(self._slip_proxy_samples, max(min(inner - 21, 14), 8))

        lines: list[Text] = [self._box_top("Orders", inner, style="#2f78c4")]
        metrics_row = Text("Fill Rate ")
        metrics_row.append(fill_meter, style="green")
        metrics_row.append(f" {int(round(fill_rate * 100)):>3}%")
        metrics_row.append("   Cancel/Replace ")
        metrics_row.append(cancel_meter, style="yellow")
        metrics_row.append(f" {int(round(cancel_replace_rate * 100)):>3}%")
        lines.append(self._box_row(metrics_row, inner, style="#2f78c4"))
        slip_row = Text(f"Slippage Dist {slip_spark}")
        slip_row.stylize("magenta")
        lines.append(self._box_row(slip_row, inner, style="#2f78c4"))
        lines.append(self._box_row(self._armed_mode_line(), inner, style="#2f78c4"))
        lines.append(self._box_row(self._active_chase_line(trades), inner, style="#2f78c4"))

        if not trades:
            self._orders_selected = 0
            self._orders_scroll = 0
            lines.append(self._box_row(Text("No open orders", style="dim"), inner, style="#2f78c4"))
        else:
            if self._orders_selected >= len(trades):
                self._orders_selected = len(trades) - 1
            reserved_without_trades = 7  # top + 4 metrics + header + bottom
            visible = len(trades)
            if available:
                visible = max(available - reserved_without_trades, 1)
            max_scroll = max(len(trades) - visible, 0)
            self._orders_scroll = min(max(self._orders_scroll, 0), max_scroll)
            if self._orders_selected < self._orders_scroll:
                self._orders_scroll = self._orders_selected
            elif self._orders_selected >= self._orders_scroll + visible:
                self._orders_scroll = self._orders_selected - visible + 1
            header = Text("Label      Stat      Exec      S Qty Type@Price   Fill/Rem  Id", style="dim")
            lines.append(self._box_row(header, inner, style="#2f78c4"))
            start = self._orders_scroll
            end = min(start + visible, len(trades))
            for idx in range(start, end):
                trade = trades[idx]
                line = self._format_order_line(trade, width=inner)
                if self._active_panel == "orders" and idx == self._orders_selected:
                    line.stylize("bold on #2b2b2b")
                lines.append(self._box_row(line, inner, style="#2f78c4"))
            hidden = len(trades) - end
            if hidden > 0:
                lines.append(
                    self._box_row(
                        Text(f"... {hidden} more (j/k to scroll)", style="dim"),
                        inner,
                        style="#2f78c4",
                    )
                )

        lines.append(self._box_bottom(inner, style="#2f78c4"))
        self._detail_right.update(Text("\n").join(lines))

    def _trade_order_id(self, trade: Trade) -> int:
        order_id = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)
        if not order_id:
            order_id = int(getattr(getattr(trade, "order", None), "permId", 0) or 0)
        return order_id

    def _armed_mode_line(self) -> Text:
        selected = self._selected_exec_mode()
        selected_label = self._exec_mode_label(selected)
        buy = self._initial_exec_price("BUY", mode=selected)
        sell = self._initial_exec_price("SELL", mode=selected)
        line = Text("Armed ")
        line.append(selected_label, style="yellow")
        if selected == "AUTO":
            line.append(" (OPT->MID->AGG->CROSS)", style="dim")
        line.append("  B/S ")
        line.append(f"{_fmt_quote(buy)}/{_fmt_quote(sell)}", style="bright_white")
        return line

    def _active_chase_line(self, trades: list[Trade]) -> Text:
        for trade in trades:
            order_id = self._trade_order_id(trade)
            state = _DETAIL_CHASE_STATE_BY_ORDER.get(order_id)
            if not isinstance(state, dict):
                continue
            selected = str(state.get("selected") or "-")
            active = str(state.get("active") or "-")
            target = _safe_num(state.get("target_price"))
            try:
                mods = int(state.get("mods") or 0)
            except (TypeError, ValueError):
                mods = 0
            line = Text(f"Chase #{order_id} ")
            line.append(selected, style="yellow")
            if selected == "AUTO":
                line.append("->", style="dim")
                line.append(active, style="yellow")
            line.append(" @ ")
            line.append(_fmt_quote(target), style="bright_white")
            line.append(f"  mods {mods}", style="dim")
            return line
        return Text("Chase idle", style="dim")

    def _order_mode_for_trade(self, trade: Trade) -> str:
        order_id = self._trade_order_id(trade)
        state = _DETAIL_CHASE_STATE_BY_ORDER.get(order_id)
        if not isinstance(state, dict):
            return "-"
        selected = str(state.get("selected") or "-")
        active = str(state.get("active") or selected or "-")
        if selected == "AUTO":
            return f"A>{active}"[:9]
        if active and active != selected:
            return f"{selected}>{active}"[:9]
        return selected[:9]

    def _format_order_line(self, trade: Trade, *, width: int) -> Text:
        contract = trade.contract
        label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or "?"
        label = label[:10]
        status = trade.orderStatus.status or "n/a"
        status = status.replace("PreSubmitted", "PreSub")
        status = status[:9]
        mode = self._order_mode_for_trade(trade)
        side = (trade.order.action or "?")[:1].upper()
        qty = _fmt_qty(float(trade.order.totalQuantity or 0))
        order_type = trade.order.orderType or ""
        price = self._order_price(trade)
        if price is not None:
            type_label = f"{order_type}@{_fmt_quote(price)}"
        else:
            type_label = order_type
        type_label = type_label[:12]
        filled = _fmt_qty(float(trade.orderStatus.filled or 0))
        remaining = _fmt_qty(float(trade.orderStatus.remaining or 0))
        order_id = trade.order.orderId or trade.order.permId or 0
        line = (
            f"{label:<10} {status:<9} {mode:<9} {side:<1} {qty:>3} "
            f"{type_label:<12} {filled:>4}/{remaining:<4} #{order_id}"
        )
        return Text(self._clip(line, width))

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
        self._render_details(sample=False)

    def _handle_digit(self, char: str) -> None:
        if char not in "0123456789":
            return
        if self._exec_rows[self._exec_selected] != "qty":
            self._exec_selected = self._exec_rows.index("qty")
        self._exec_qty_input = _append_digit(self._exec_qty_input, char, allow_decimal=False)
        parsed = _parse_int(self._exec_qty_input)
        if parsed:
            self._exec_qty = parsed
        self._render_details(sample=False)

    def _handle_backspace(self) -> None:
        if self._exec_rows[self._exec_selected] != "qty":
            self._exec_selected = self._exec_rows.index("qty")
        self._exec_qty_input = self._exec_qty_input[:-1]
        parsed = _parse_int(self._exec_qty_input)
        if parsed:
            self._exec_qty = parsed
        self._render_details(sample=False)

    def _adjust_qty(self, direction: int) -> None:
        next_qty = max(1, int(self._exec_qty) + direction)
        self._exec_qty = next_qty
        self._exec_qty_input = str(next_qty)

    def _submit_order(self, action: str) -> None:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP"):
            self._exec_status = "Exec: unsupported contract"
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
                self._exec_status = "Exec locked: no actionable equity option quote yet (bid/ask/last n/a)"
                self._render_details(sample=False)
                return
        qty = int(self._exec_qty) if self._exec_qty else 0
        if qty <= 0:
            self._exec_status = "Exec: invalid qty"
            self._render_details(sample=False)
            return
        mode = self._selected_exec_mode()
        price = self._initial_exec_price(action, mode=mode)
        if price is None:
            if contract.secType == "OPT":
                self._exec_status = "Exec locked: no actionable option quote yet (bid/ask/last n/a)"
            else:
                self._exec_status = "Exec: price n/a"
            self._render_details(sample=False)
            return
        outside_rth = contract.secType == "STK"
        mode_label = self._exec_mode_label(mode)
        self._exec_status = f"Sending {action} {qty} @ {price:.2f} [{mode_label}]"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Exec: no loop"
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
            mode_label = self._exec_mode_label(mode)
            self._exec_status = f"Sent {action} {qty} @ {price:.2f} [{mode_label}]"
            order_id = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)
            if not order_id:
                order_id = int(getattr(getattr(trade, "order", None), "permId", 0) or 0)
            if order_id:
                seeded = "OPTIMISTIC" if mode == "AUTO" else mode
                _DETAIL_CHASE_STATE_BY_ORDER[order_id] = {
                    "selected": self._exec_mode_label(mode),
                    "active": self._exec_mode_label(seeded),
                    "target_price": float(price),
                    "mods": 0,
                }
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                task = loop.create_task(self._chase_until_filled(trade, action, mode=mode))
                self._chase_tasks.add(task)
                task.add_done_callback(lambda t: self._chase_tasks.discard(t))
        except Exception as exc:
            self._exec_status = f"Exec error: {exc}"
        self._render_details_if_mounted(sample=False)

    def _initial_exec_price(self, action: str, *, mode: str = "AUTO") -> float | None:
        bid = self._quote_num(self._ticker.bid) if self._ticker else None
        ask = self._quote_num(self._ticker.ask) if self._ticker else None
        last = self._quote_num(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, self._ticker, last_ref)
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
    ) -> float | None:
        bid = bid if bid is not None else (self._quote_num(self._ticker.bid) if self._ticker else None)
        ask = ask if ask is not None else (self._quote_num(self._ticker.ask) if self._ticker else None)
        last = last if last is not None else (self._quote_num(self._ticker.last) if self._ticker else None)
        ticker_ref = ticker or self._ticker
        mark = _option_display_price(self._item, ticker_ref) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, ticker_ref, last_ref)
        value = _limit_price_for_mode(bid, ask, last_ref, action=action, mode=mode)
        if value is None:
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        return _round_to_tick(float(value), tick)

    async def _chase_until_filled(self, trade: Trade, action: str, *, mode: str = "AUTO") -> None:
        order_id = int(getattr(getattr(trade, "order", None), "orderId", 0) or 0)
        if not order_id:
            order_id = int(getattr(getattr(trade, "order", None), "permId", 0) or 0)
        con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
        chase_owner = f"details-chase:{order_id or con_id or id(trade)}"
        try:
            await self._client.ensure_ticker(trade.contract, owner=chase_owner)
        except Exception:
            pass
        started = asyncio.get_running_loop().time()
        last_reprice_ts: float | None = None
        prev_mode: str | None = None
        prev_quote_sig: tuple[float | None, float | None, float | None] | None = None
        selected_label = self._exec_mode_label(mode)
        try:
            while True:
                try:
                    if trade.isDone():
                        if order_id:
                            _DETAIL_CHASE_STATE_BY_ORDER.pop(order_id, None)
                        return
                except Exception:
                    pass
                status_raw = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
                if status_raw in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                    if order_id:
                        _DETAIL_CHASE_STATE_BY_ORDER.pop(order_id, None)
                    return

                loop_now = asyncio.get_running_loop().time()
                elapsed = loop_now - started
                mode_now = _exec_chase_mode(elapsed, selected_mode=mode)
                if mode_now is None:
                    try:
                        await self._client.cancel_trade(trade)
                        order_id = trade.order.orderId or trade.order.permId or 0
                        self._exec_status = (
                            f"Timeout cancel sent #{order_id} (> {_EXEC_LADDER_TIMEOUT_SEC:.0f}s)"
                        )
                    except Exception as exc:
                        self._exec_status = f"Timeout cancel error: {exc}"
                    if order_id:
                        _DETAIL_CHASE_STATE_BY_ORDER.pop(order_id, None)
                    self._render_details_if_mounted(sample=False)
                    return

                ticker = self._client.ticker_for_con_id(con_id) if con_id else None
                bid = self._quote_num(getattr(ticker, "bid", None)) if ticker else None
                ask = self._quote_num(getattr(ticker, "ask", None)) if ticker else None
                last = self._quote_num(getattr(ticker, "last", None)) if ticker else None
                quote_sig = _exec_chase_quote_signature(bid, ask, last)
                should_reprice = _exec_chase_should_reprice(
                    now_sec=loop_now,
                    last_reprice_sec=last_reprice_ts,
                    mode_now=str(mode_now),
                    prev_mode=prev_mode,
                    quote_signature=quote_sig,
                    prev_quote_signature=prev_quote_sig,
                    min_interval_sec=5.0,
                )
                prev_mode = str(mode_now)
                prev_quote_sig = quote_sig
                if order_id:
                    state = _DETAIL_CHASE_STATE_BY_ORDER.get(order_id) or {}
                    state["selected"] = selected_label
                    state["active"] = self._exec_mode_label(str(mode_now))
                    _DETAIL_CHASE_STATE_BY_ORDER[order_id] = state

                if should_reprice:
                    price = self._exec_price_for_mode(
                        str(mode_now),
                        action,
                        bid=bid,
                        ask=ask,
                        last=last,
                        ticker=ticker,
                    )
                    if order_id and price is not None:
                        state = _DETAIL_CHASE_STATE_BY_ORDER.get(order_id) or {}
                        state["selected"] = selected_label
                        state["active"] = self._exec_mode_label(str(mode_now))
                        state["target_price"] = float(price)
                        _DETAIL_CHASE_STATE_BY_ORDER[order_id] = state
                else:
                    price = None
                if price is not None:
                    try:
                        trade = await self._client.modify_limit_order(trade, float(price))
                        order_id = trade.order.orderId or trade.order.permId or 0
                        mode_label = self._exec_mode_label(str(mode_now))
                        mods = 1
                        if order_id:
                            state = _DETAIL_CHASE_STATE_BY_ORDER.get(order_id) or {}
                            try:
                                mods = int(state.get("mods") or 0) + 1
                            except (TypeError, ValueError):
                                mods = 1
                            state["selected"] = selected_label
                            state["active"] = mode_label
                            state["target_price"] = float(price)
                            state["mods"] = mods
                            _DETAIL_CHASE_STATE_BY_ORDER[order_id] = state
                        mode_view = f"{selected_label}->{mode_label}" if selected_label == "AUTO" else mode_label
                        self._exec_status = f"Chasing #{order_id} [{mode_view}] @ {price:.2f} mod#{mods}"
                        last_reprice_ts = loop_now
                    except Exception as exc:
                        self._exec_status = f"Chase error: {exc}"
                    self._render_details_if_mounted(sample=False)

                await asyncio.sleep(0.25)
        finally:
            if order_id:
                _DETAIL_CHASE_STATE_BY_ORDER.pop(order_id, None)
            if con_id:
                try:
                    self._client.release_ticker(con_id, owner=chase_owner)
                except Exception:
                    pass
