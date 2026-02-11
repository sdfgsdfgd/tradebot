"""Position detail screen (execution + quotes)."""

from __future__ import annotations

import asyncio
from collections import deque
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
    _default_order_qty,
    _exec_chase_mode,
    _exec_chase_quote_signature,
    _exec_chase_should_reprice,
    _EXEC_LADDER_TIMEOUT_SEC,
    _fmt_expiry,
    _fmt_money,
    _fmt_qty,
    _fmt_quote,
    _limit_price_for_mode,
    _market_data_label,
    _mark_price,
    _midpoint,
    _optimistic_price,
    _parse_int,
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
    _AURORA_TAPE_BIAS = 0.15
    _VOL_MAGENTA_STYLES = ("#4e1e57", "#6f2380", "#922aaa", "#b534d8", "#ff3dee")
    _TREND_ROWS = 5
    _TREND_LINE_STYLE_BY_PRESET = {"calm": "square", "normal": "rounded", "feral": "heavy"}
    _AURORA_PRESET_ORDER = ("calm", "normal", "feral")
    _AURORA_PRESETS = {
        "calm": {"buy_soft": 0.28, "buy_strong": 0.56, "sell_soft": -0.28, "sell_strong": -0.56, "burst_gain": 0.80},
        "normal": {"buy_soft": 0.16, "buy_strong": 0.34, "sell_soft": -0.16, "sell_strong": -0.34, "burst_gain": 1.00},
        "feral": {"buy_soft": 0.08, "buy_strong": 0.22, "sell_soft": -0.08, "sell_strong": -0.22, "burst_gain": 1.30},
    }

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
        self._spread_samples: deque[float] = deque(maxlen=96)
        self._size_samples: deque[float] = deque(maxlen=96)
        self._pnl_samples: deque[float] = deque(maxlen=96)
        self._slip_proxy_samples: deque[float] = deque(maxlen=96)
        self._imbalance_samples: deque[float] = deque(maxlen=240)
        self._vol_burst_samples: deque[float] = deque(maxlen=240)
        self._trend_bins: list[float] = []
        self._aurora_preset = "normal"
        self._pos_prev_qty: float | None = None
        self._pos_delta: float = 0.0
        self._pos_pulse_until: float = 0.0

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
        await self._load_underlying()
        self._refresh_task = self.set_interval(self._refresh_sec, self._render_details)
        self._render_details()

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id, owner="details")
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")

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

    def _position_beacon_row(self, qty: float, notional: float | None) -> Text:
        now = monotonic()
        if self._pos_prev_qty is None:
            self._pos_prev_qty = float(qty)
        elif float(qty) != float(self._pos_prev_qty):
            self._pos_delta = float(qty) - float(self._pos_prev_qty)
            self._pos_prev_qty = float(qty)
            self._pos_pulse_until = now + self._POSITION_PULSE_SEC

        direction = "FLAT"
        style = "dim"
        if qty > 0:
            direction = "LONG"
            style = "bold green"
        elif qty < 0:
            direction = "SHORT"
            style = "bold red"

        qty_label = _fmt_qty(float(qty))
        if qty > 0:
            qty_label = f"+{qty_label}"

        row = Text("POS ", style="bold")
        row.append(f"{qty_label} sh", style=style)
        if notional is not None:
            signed = float(notional)
            sign = "+" if signed > 0 else "-" if signed < 0 else ""
            notional_style = "green" if signed > 0 else "red" if signed < 0 else "dim"
            row.append(" (", style="dim")
            row.append(f"{sign}${abs(signed):,.0f}", style=notional_style)
            row.append(")", style="dim")
        row.append("   ")
        row.append(direction, style=style)

        pulse_active = now < float(self._pos_pulse_until)
        if pulse_active and self._pos_delta:
            delta = self._pos_delta
            delta_style = "green" if delta > 0 else "red"
            delta_label = _fmt_qty(abs(delta))
            sign = "+" if delta > 0 else "-"
            row.append("   Δ ")
            row.append(f"{sign}{delta_label}", style=f"bold {delta_style}")
            row.append(" sh", style=delta_style)
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
        if mid is not None:
            self._mid_samples.append(float(mid))
        if spread is not None and spread >= 0:
            self._spread_samples.append(float(spread))
        if size is not None and size >= 0:
            self._size_samples.append(float(size))
        if pnl is not None:
            self._pnl_samples.append(float(pnl))
        if slip_proxy is not None and slip_proxy >= 0:
            self._slip_proxy_samples.append(float(slip_proxy))
        if imbalance is not None:
            self._imbalance_samples.append(max(min(float(imbalance), 1.0), -1.0))
        if vol_burst is not None and vol_burst >= 0:
            self._vol_burst_samples.append(float(vol_burst))

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
    def _trend_pair_value(older: float, newer: float) -> float:
        # Weight recency so the right edge tracks the latest tape more tightly.
        return older * 0.38 + newer * 0.62

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

    def _trend_continuity_rows(self, values: deque[float], width: int) -> tuple[list[str], int]:
        width = max(width, 1)
        cols = width
        rows = max(3, int(self._TREND_ROWS))
        raw = list(values)[-cols:]
        if not raw:
            blank = " " * width
            return ([blank for _ in range(rows)], rows // 2)
        if len(raw) < width:
            raw = ([raw[0]] * (width - len(raw))) + raw

        alpha = 0.22
        smoothed: list[float] = [float(raw[0])]
        for value in raw[1:]:
            prev = smoothed[-1]
            smoothed.append(prev + alpha * (float(value) - prev))

        latest = smoothed[-1]
        deltas = [value - latest for value in smoothed]
        mags = sorted(abs(delta) for delta in deltas if abs(delta) > 1e-12)
        if not mags:
            y_points = [((rows - 1) * 0.5)] * cols
        else:
            last = len(mags) - 1
            q80 = mags[int(round(last * 0.80))]
            q92 = mags[int(round(last * 0.92))]
            peak = mags[-1]
            scale = max(q80, q92 * 0.75, peak * 0.30, 1e-12)
            net = smoothed[-1] - smoothed[0]
            net_norm = max(-1.0, min(net / (scale * max(cols * 0.16, 1.0)), 1.0))
            if abs(net_norm) > 1e-12:
                net_norm = pow(abs(net_norm), 0.82) * (1.0 if net_norm >= 0 else -1.0)
            anchor = ((rows - 1) * 0.5) - (net_norm * 1.25)
            gain = 1.55 + (abs(net_norm) * 0.75)
            y_points = []
            for idx, delta in enumerate(deltas):
                progress = float(idx) / float(max(cols - 1, 1))
                history_push = (1.0 - progress) * net_norm * 1.20
                yy = anchor - ((delta / scale) * gain) + history_push
                y_points.append(min(rows - 1.0, max(0.0, yy)))
        self._trend_bins = self._resample(smoothed, width)

        y_int = [max(0, min(rows - 1, int(round(value)))) for value in y_points]
        links = [[0] * cols for _ in range(rows)]

        UP, DOWN, LEFT, RIGHT = 1, 2, 4, 8

        def add_link(r1: int, c1: int, r2: int, c2: int) -> None:
            if not (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols):
                return
            if r1 == r2 and abs(c1 - c2) == 1:
                if c2 > c1:
                    links[r1][c1] |= RIGHT
                    links[r2][c2] |= LEFT
                else:
                    links[r1][c1] |= LEFT
                    links[r2][c2] |= RIGHT
                return
            if c1 == c2 and abs(r1 - r2) == 1:
                if r2 > r1:
                    links[r1][c1] |= DOWN
                    links[r2][c2] |= UP
                else:
                    links[r1][c1] |= UP
                    links[r2][c2] |= DOWN

        for idx in range(1, cols):
            prev_row = y_int[idx - 1]
            curr_row = y_int[idx]
            add_link(prev_row, idx - 1, prev_row, idx)
            if curr_row != prev_row:
                step = 1 if curr_row > prev_row else -1
                row = prev_row
                while row != curr_row:
                    add_link(row, idx, row + step, idx)
                    row += step

        out_rows: list[list[str]] = [[" "] * width for _ in range(rows)]
        square_map = {
            LEFT | RIGHT: "─",
            UP | DOWN: "│",
            RIGHT | DOWN: "┌",
            LEFT | DOWN: "┐",
            RIGHT | UP: "└",
            LEFT | UP: "┘",
            UP | DOWN | RIGHT: "├",
            UP | DOWN | LEFT: "┤",
            LEFT | RIGHT | DOWN: "┬",
            LEFT | RIGHT | UP: "┴",
            UP | DOWN | LEFT | RIGHT: "┼",
            LEFT: "─",
            RIGHT: "─",
            UP: "│",
            DOWN: "│",
            0: " ",
        }
        rounded_map = {
            LEFT | RIGHT: "─",
            UP | DOWN: "│",
            RIGHT | DOWN: "╭",
            LEFT | DOWN: "╮",
            RIGHT | UP: "╰",
            LEFT | UP: "╯",
            UP | DOWN | RIGHT: "├",
            UP | DOWN | LEFT: "┤",
            LEFT | RIGHT | DOWN: "┬",
            LEFT | RIGHT | UP: "┴",
            UP | DOWN | LEFT | RIGHT: "┼",
            LEFT: "─",
            RIGHT: "─",
            UP: "│",
            DOWN: "│",
            0: " ",
        }
        heavy_map = {
            LEFT | RIGHT: "━",
            UP | DOWN: "┃",
            RIGHT | DOWN: "┏",
            LEFT | DOWN: "┓",
            RIGHT | UP: "┗",
            LEFT | UP: "┛",
            UP | DOWN | RIGHT: "┣",
            UP | DOWN | LEFT: "┫",
            LEFT | RIGHT | DOWN: "┳",
            LEFT | RIGHT | UP: "┻",
            UP | DOWN | LEFT | RIGHT: "╋",
            LEFT: "━",
            RIGHT: "━",
            UP: "┃",
            DOWN: "┃",
            0: " ",
        }
        line_style = self._TREND_LINE_STYLE_BY_PRESET.get(self._aurora_preset, "rounded")
        glyph_map = rounded_map
        if line_style == "square":
            glyph_map = square_map
        elif line_style == "heavy":
            glyph_map = heavy_map
        for row_idx in range(rows):
            for col_idx in range(cols):
                out_rows[row_idx][col_idx] = glyph_map.get(links[row_idx][col_idx], "·")

        now_row = max(0, min(rows - 1, y_int[-1]))
        return (["".join(row).rjust(width) for row in out_rows], now_row)

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

    def _vol_histogram_braille(self, values: deque[float], width: int) -> Text:
        width = max(width, 1)
        points = self._resample(list(values), width * 4)
        if not points:
            return Text(" " * width, style="dim")
        target = width * 4
        if len(points) < target:
            points.extend([points[-1]] * (target - len(points)))
        elif len(points) > target:
            points = points[-target:]
        global_max = max(points)
        if global_max <= 1e-12:
            return Text(" " * width, style="dim")
        global_lo = min(points)
        global_hi = max(points)
        global_span = global_hi - global_lo
        if global_span <= 1e-12:
            level = 4 if global_max > 1e-12 else 0
            row = Text()
            for _ in range(width):
                cell = self._trend_braille_cell(level, level, top=True)
                if cell == " ":
                    row.append(" ", style="dim")
                else:
                    row.append(cell, style=self._VOL_MAGENTA_STYLES[min(level, len(self._VOL_MAGENTA_STYLES) - 1)])
            return row
        window_chars = max(3, min(14, width // 6 + 2))
        floor_span = max(global_span * 0.08, 1e-12)
        lows: list[float] = []
        spans: list[float] = []
        for idx in range(width):
            lo_idx = max(0, (idx - window_chars) * 4)
            hi_idx = min(target, (idx + window_chars + 1) * 4)
            segment = points[lo_idx:hi_idx]
            local_lo = min(segment)
            local_hi = max(segment)
            lows.append(local_lo)
            spans.append(max(local_hi - local_lo, floor_span))
        alpha = 0.32
        for idx in range(1, width):
            lows[idx] = lows[idx - 1] + alpha * (lows[idx] - lows[idx - 1])
            spans[idx] = spans[idx - 1] + alpha * (spans[idx] - spans[idx - 1])
        for idx in range(width - 2, -1, -1):
            lows[idx] = lows[idx + 1] + alpha * (lows[idx] - lows[idx + 1])
            spans[idx] = max(floor_span, spans[idx + 1] + alpha * (spans[idx] - spans[idx + 1]))
        micro_gamma = 0.82 if self._aurora_preset == "feral" else 0.90
        out = Text()
        for char_idx in range(width):
            base = char_idx * 4
            left = self._trend_pair_value(float(points[base]), float(points[base + 1]))
            right = self._trend_pair_value(float(points[base + 2]), float(points[base + 3]))
            local_lo = lows[char_idx]
            local_span = spans[char_idx]
            left_dyn = min(max((left - local_lo) / local_span, 0.0), 1.0)
            right_dyn = min(max((right - local_lo) / local_span, 0.0), 1.0)
            left_abs = min(max(left / global_max, 0.0), 1.0)
            right_abs = min(max(right / global_max, 0.0), 1.0)
            left_ratio = max(left_dyn, left_abs * 0.52)
            right_ratio = max(right_dyn, right_abs * 0.52)
            left_level = max(0, min(4, int(round(pow(left_ratio, micro_gamma) * 4.0))))
            right_level = max(0, min(4, int(round(pow(right_ratio, micro_gamma) * 4.0))))
            if left_level == 0 and left_ratio > 1e-3:
                left_level = 1
            if right_level == 0 and right_ratio > 1e-3:
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

    def _aurora_drift_series(self, width: int) -> list[float]:
        width = max(width, 1)
        mids = list(self._mid_samples)
        if len(mids) < 3:
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
        return self._resample(smooth, width)

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
        if not self._imbalance_samples:
            return "#8aa0b6"
        drift_now = self._aurora_drift_series(1)[0]
        blended = self._aurora_blended_imbalance(float(self._imbalance_samples[-1]), drift_now)
        return self._aurora_style(blended)

    def _aurora_strip(self, width: int) -> Text:
        width = max(width, 1)
        imbalances = self._resample(list(self._imbalance_samples), width)
        drifts = self._aurora_drift_series(width)
        bursts = self._resample(list(self._vol_burst_samples), width)
        if not imbalances or not drifts or not bursts:
            return Text(" " * width, style="dim")
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
            if latest:
                self._item = latest
            contract = self._item.contract
            bid = _safe_num(self._ticker.bid) if self._ticker else None
            ask = _safe_num(self._ticker.ask) if self._ticker else None
            last = _safe_num(self._ticker.last) if self._ticker else None
            price = _ticker_price(self._ticker) if self._ticker else None
            mid = _midpoint(bid, ask)
            close = _ticker_close(self._ticker) if self._ticker else None
            mark = _mark_price(self._item)
            live_mark = price or mark
            spread = (ask - bid) if bid is not None and ask is not None and ask >= bid else None
            size = _safe_num(getattr(self._ticker, "lastSize", None)) if self._ticker else None
            bid_size = _safe_num(getattr(self._ticker, "bidSize", None)) if self._ticker else None
            ask_size = _safe_num(getattr(self._ticker, "askSize", None)) if self._ticker else None
            pnl_value, _ = _unrealized_pnl_values(self._item, mark_price=live_mark)
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
        pnl_value, _ = _unrealized_pnl_values(self._item, mark_price=price or mark)
        pnl_label = _fmt_money(pnl_value) if pnl_value is not None else "n/a"
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
        headline.append("   PnL ")
        headline.append(pnl_label, style=self._pnl_style(pnl_value))
        position_row = self._position_beacon_row(position_qty, market_value_raw)

        aurora_label_row = Text("Aurora", style="#8aa0b6")
        aurora_row = self._mark_now(self._aurora_strip(spark_width))
        trend_label_row = Text("1m Trend", style="cyan")
        trend_row_values, trend_now_row = self._trend_continuity_rows(
            self._mid_samples, spark_width
        )
        comet_color = self._aurora_now_style()
        trend_rows = [Text(row, style="#63d9ff") for row in trend_row_values]
        trend_price = mid or price or mark
        trend_rows[trend_now_row] = self._trend_with_price_tag(
            trend_rows[trend_now_row], trend_price, color=comet_color
        )
        vol_label_row = Text("Vol Histogram", style="magenta")
        vol_row = self._mark_now(self._vol_histogram_braille(self._size_samples, spark_width))
        momentum_label_row = Text("Momentum", style="yellow")
        momentum_row = self._mark_now(Text(self._momentum_line(spark_width), style="yellow"))

        detail_row = Text(f"Avg {avg_cost}   MktVal {market_value}")
        pnl_prefix = "Unrealized "
        tail_row = Text(f"{pnl_prefix}{pnl_label}   Realized {realized}")
        tail_row.stylize(
            self._pnl_style(pnl_value),
            len(pnl_prefix),
            len(pnl_prefix) + len(pnl_label),
        )

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
            ubid = _safe_num(self._underlying_ticker.bid)
            uask = _safe_num(self._underlying_ticker.ask)
            ulast = _safe_num(self._underlying_ticker.last)
            under_row = Text(f"{label}: ")
            under_row.append(f"{_fmt_quote(ubid)}/{_fmt_quote(uask)}/{_fmt_quote(ulast)}", style="cyan")
            lines.append(self._box_row(under_row, inner, style="#2d8fd5"))
        lines.append(self._box_bottom(inner, style="#2d8fd5"))
        return lines

    def _render_execution_block(self, *, panel_width: int) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT"):
            return []
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        last_ref = last or mark
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
        if contract.secType not in ("STK", "OPT"):
            self._exec_status = "Exec: unsupported contract"
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
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        last_ref = last or mark
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
        bid = bid if bid is not None else (_safe_num(self._ticker.bid) if self._ticker else None)
        ask = ask if ask is not None else (_safe_num(self._ticker.ask) if self._ticker else None)
        last = last if last is not None else (_safe_num(self._ticker.last) if self._ticker else None)
        mark = _mark_price(self._item)
        last_ref = last or mark
        tick = _tick_size(self._item.contract, ticker or self._ticker, last_ref)
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
                bid = _safe_num(getattr(ticker, "bid", None)) if ticker else None
                ask = _safe_num(getattr(ticker, "ask", None)) if ticker else None
                last = _safe_num(getattr(ticker, "last", None)) if ticker else None
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
