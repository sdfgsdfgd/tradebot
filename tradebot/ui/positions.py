"""Position detail screen (execution + quotes)."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import time as dtime
import math
import re
import textwrap
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
    _EXEC_RELENTLESS_TIMEOUT_SEC,
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
    _safe_float,
    _safe_num,
    _tick_decimals,
    _tick_size,
    _trade_sort_key,
    _quote_status_line,
    _ticker_close,
    _ticker_price,
    _unrealized_pnl_values,
)
from .time_compat import now_et as _now_et

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
    _ORDER_PANEL_NOTICE_TTL_SEC = 5 * 60.0
    _RELENTLESS_BASE_CROSS_SEC = 2.0
    _RELENTLESS_STEP_SEC = 3.0
    _RELENTLESS_STEP_SEC_SHOCK = 1.5
    _RELENTLESS_MIN_REPRICE_SEC = 0.75
    _RELENTLESS_MIN_REPRICE_SEC_SHOCK = 0.35
    _RELENTLESS_MIN_REPRICE_SEC_HYPER = 0.25
    _RELENTLESS_OPEN_SHOCK_WINDOW_SEC = 120.0
    _RELENTLESS_STALE_TOP_AGE_SEC = 2.0
    _RELENTLESS_SPREAD_PRESSURE_TRIGGER = 2.0
    _RELENTLESS_SPREAD_PRESSURE_HYPER = 3.5
    _RELENTLESS_MAX_EDGE_TICKS = 40
    _RELENTLESS_DELAY_RECOVER_ATTEMPTS = 24
    _RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC = 0.35
    _RELENTLESS_DELAY_RECOVER_WINDOW_SEC = 25.0
    _RELENTLESS_DELAY_RECOVER_SETTLE_SEC = 2.0
    _RELENTLESS_DELAY_FAVORABLE_PCT = 0.06
    _RELENTLESS_DELAY_ADVERSE_PCT = 0.12
    _RELENTLESS_DELAY_FAVORABLE_SPREAD_MULT = 0.8
    _RELENTLESS_DELAY_ADVERSE_SPREAD_MULT = 2.0
    _RELENTLESS_DELAY_FAVORABLE_MAX_TICKS = 20
    _RELENTLESS_DELAY_ADVERSE_MAX_TICKS = 60
    _RELENTLESS_DELAY_SHRINK_PER_REJECT = 0.75
    _RELENTLESS_DELAY_PRICE_HINT_RE = re.compile(r"(?<!\d)(\d[\d,]*\.\d+)")
    _CHASE_PENDING_ACK_SEC = 0.9
    _CHASE_RECONCILE_INTERVAL_SEC = 0.9
    _CHASE_FORCE_RECONCILE_INTERVAL_SEC = 5.0
    _CHASE_MODIFY_ERROR_BACKOFF_SEC = 1.0
    _CANCEL_REQUEST_TTL_SEC = 90.0
    _STREAM_RENDER_DEBOUNCE_SEC = 0.08
    _MD_PROBE_BANNER_TTL_SEC = 10.0
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
        self._exec_rows = [
            "ladder",
            "relentless",
            "relentless_delay",
            "optimistic",
            "mid",
            "aggressive",
            "cross",
            "custom",
            "qty",
        ]
        self._exec_selected = 0
        self._exec_custom_input = ""
        self._exec_custom_price: float | None = None
        self._exec_qty_input = ""
        self._exec_qty = _default_order_qty(item)
        self._exec_status: str | None = None
        self._active_panel = "exec"
        self._orders_selected = 0
        self._orders_scroll = 0
        self._orders_rows: list[Trade] = []
        self._orders_notice: tuple[float, str, str] | None = None
        self._refresh_task = None
        self._chase_tasks: set[asyncio.Task] = set()
        self._chase_task_by_order: dict[int, asyncio.Task] = {}
        self._cancel_requested_at_by_order: dict[int, float] = {}
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
        self._bootstrap_task: asyncio.Task | None = None
        self._stream_render_task: asyncio.Task | None = None
        self._md_probe_requested_type: int | None = None
        self._md_probe_started_mono: float = 0.0

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
        if self._active_panel != "exec":
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
        if self._active_panel == "orders":
            self._render_details(sample=False)
            return
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._adjust_qty(1)
        elif selected == "custom":
            self._adjust_custom_price(1)
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
            self._exec_status = "Cancel: focus orders panel"
            self._set_orders_notice(self._exec_status, level="warn")
            self._render_details(sample=False)
            return
        trade = self._selected_order()
        if not trade:
            self._exec_status = "Cancel: no order"
            self._set_orders_notice(self._exec_status, level="warn")
            self._render_details(sample=False)
            return
        order_id_raw, perm_id_raw = self._trade_order_ids(trade)
        self._mark_cancel_requested(order_id=order_id_raw, perm_id=perm_id_raw)
        self._cancel_chase_for_ids(order_id=order_id_raw, perm_id=perm_id_raw)
        order_id = order_id_raw or perm_id_raw or 0
        self._exec_status = f"Canceling #{order_id}"
        self._set_orders_notice(self._exec_status, level="warn")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Cancel: no loop"
            self._set_orders_notice(self._exec_status, level="error")
            self._render_details(sample=False)
            return
        loop.create_task(self._cancel_order(trade))
        self._render_details(sample=False)

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
        self._md_probe_requested_type = self._md_type_value(self._ticker)
        self._md_probe_started_mono = monotonic()
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id, owner="details")
            self._underlying_con_id = None
            self._underlying_ticker = None
            self._underlying_label = None
            await self._load_underlying()
        attempted_live_probe = False
        live_source: str | None = None
        sec_type = str(getattr(self._item.contract, "secType", "") or "").strip().upper()
        md_type_raw = getattr(self._ticker, "marketDataType", None) if self._ticker else None
        try:
            md_type = int(md_type_raw) if md_type_raw is not None else None
        except (TypeError, ValueError):
            md_type = None
        if sec_type in ("FUT", "FOP") and md_type in (2, 3, 4):
            attempted_live_probe = True
            try:
                live_source = await self._client.refresh_live_snapshot_once(self._item.contract)
            except Exception:
                live_source = None
        if live_source:
            self._exec_status = f"MD refreshed + 1-shot {live_source}"
        elif attempted_live_probe:
            self._exec_status = "MD refreshed (1-shot live unavailable)"
        else:
            self._exec_status = "MD refreshed"
        self._render_details()

    @staticmethod
    def _md_type_value(ticker: Ticker | None) -> int | None:
        if ticker is None:
            return None
        raw = getattr(ticker, "marketDataType", None)
        try:
            value = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            value = None
        if value in (1, 2, 3, 4):
            return value
        return None

    @staticmethod
    def _md_type_name(md_type: int | None) -> str:
        if md_type == 1:
            return "Live"
        if md_type == 2:
            return "Live-Frozen"
        if md_type == 3:
            return "Delayed"
        if md_type == 4:
            return "Delayed-Frozen"
        return "n/a"

    def _market_data_probe_row(self) -> Text | None:
        started_mono = float(self._md_probe_started_mono or 0.0)
        if started_mono <= 0:
            return None
        elapsed_sec = max(0.0, monotonic() - started_mono)
        if elapsed_sec > float(self._MD_PROBE_BANNER_TTL_SEC):
            return None
        req_type = self._md_probe_requested_type
        actual_type = self._md_type_value(self._ticker)
        remaining_sec = max(0.0, float(self._MD_PROBE_BANNER_TTL_SEC) - elapsed_sec)
        row = Text("MD Probe ", style="yellow")
        row.append("req ")
        row.append(self._md_type_name(req_type), style="bright_white")
        row.append(f" ({req_type if req_type is not None else 'n/a'})", style="dim")
        row.append(" -> now ")
        actual_style = "green" if req_type is not None and req_type == actual_type else "yellow"
        row.append(self._md_type_name(actual_type), style=actual_style)
        row.append(f" ({actual_type if actual_type is not None else 'n/a'})", style="dim")
        row.append(f"  {remaining_sec:.0f}s", style="dim")
        return row

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
        return _safe_float(value)

    def _official_unrealized(self) -> float | None:
        con_id = int(getattr(self._item.contract, "conId", 0) or 0)
        official = self._client.pnl_single_unrealized(con_id)
        if official is None:
            official = self._float_or_none(getattr(self._item, "unrealizedPNL", None))
        return official

    def _official_daily_contract_pnl(self) -> float | None:
        con_id = int(getattr(self._item.contract, "conId", 0) or 0)
        return self._client.pnl_single_daily(con_id)

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

    def _set_orders_notice(self, message: str, *, level: str = "info") -> None:
        text = str(message or "").strip()
        if not text:
            return
        cleaned_level = str(level or "info").strip().lower()
        if cleaned_level not in ("info", "warn", "error"):
            cleaned_level = "info"
        self._orders_notice = (monotonic(), cleaned_level, text)

    def _orders_notice_line(self) -> Text | None:
        payload = self._orders_notice
        if payload is None:
            return None
        ts_mono, level, message = payload
        if (monotonic() - float(ts_mono)) > float(self._ORDER_PANEL_NOTICE_TTL_SEC):
            self._orders_notice = None
            return None
        style = "bright_cyan"
        label = "INFO"
        if level == "warn":
            style = "yellow"
            label = "WARN"
        elif level == "error":
            style = "bold red"
            label = "ERR"
        line = Text(f"{label} ", style=style)
        line.append(str(message), style=style)
        return line

    def _consume_order_error(self, order_id: int, perm_id: int = 0) -> tuple[int, str] | None:
        candidate_ids: list[int] = []
        for raw_id in (order_id, perm_id):
            try:
                candidate = int(raw_id or 0)
            except (TypeError, ValueError):
                candidate = 0
            if candidate <= 0 or candidate in candidate_ids:
                continue
            candidate_ids.append(int(candidate))
        if not candidate_ids:
            return None
        pop_order_error = getattr(self._client, "pop_order_error", None)
        if not callable(pop_order_error):
            return None

        def _pop_error(order_ref: int):
            try:
                return pop_order_error(
                    int(order_ref),
                    max_age_sec=float(self._ORDER_PANEL_NOTICE_TTL_SEC),
                )
            except TypeError:
                try:
                    return pop_order_error(int(order_ref))
                except Exception:
                    return None
            except Exception:
                return None

        payload = None
        for candidate in candidate_ids:
            payload = _pop_error(int(candidate))
            if isinstance(payload, dict):
                break
        if not isinstance(payload, dict):
            return None
        try:
            code = int(payload.get("code") or 0)
        except (TypeError, ValueError):
            code = 0
        message = str(payload.get("message") or "").strip()
        if not message:
            return None
        return code, message

    async def _await_order_error(
        self,
        order_id: int,
        perm_id: int = 0,
        *,
        attempts: int = 4,
        interval_sec: float = 0.1,
    ) -> tuple[int, str] | None:
        payload = self._consume_order_error(order_id, perm_id)
        if payload is not None:
            return payload
        loops = max(0, int(attempts) - 1)
        pause = max(0.01, float(interval_sec))
        for _ in range(loops):
            await asyncio.sleep(pause)
            payload = self._consume_order_error(order_id, perm_id)
            if payload is not None:
                return payload
        return None

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
        self._sync_bound_tickers()
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
            self._request_stream_render(sample=False)
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
        self._request_stream_render(sample=False)

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
        if mode == "RELENTLESS":
            return "RLT"
        if mode == "RELENTLESS_DELAY":
            return "RLT⚔Delay"
        if mode == "CUSTOM":
            return "CUSTOM"
        if mode == "OPTIMISTIC":
            return "OPT"
        if mode == "AGGRESSIVE":
            return "AGG"
        return mode

    def _selected_exec_mode(self) -> str:
        selected = self._exec_rows[self._exec_selected]
        if selected == "relentless":
            return "RELENTLESS"
        if selected == "relentless_delay":
            return "RELENTLESS_DELAY"
        if selected == "optimistic":
            return "OPTIMISTIC"
        if selected == "mid":
            return "MID"
        if selected == "aggressive":
            return "AGGRESSIVE"
        if selected == "cross":
            return "CROSS"
        if selected == "custom":
            return "CUSTOM"
        return "AUTO"

    @staticmethod
    def _contract_header_title(contract: object) -> str:
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper() or "?"
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type == "STK":
            kind = "STOCK"
        elif sec_type == "FUT":
            kind = "FUTURES"
        elif sec_type == "OPT":
            kind = "OPTIONS"
        elif sec_type == "FOP":
            kind = "FOP"
        else:
            kind = sec_type or "UNKNOWN"
        side = ""
        if sec_type in ("OPT", "FOP"):
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            if right == "C":
                side = " CALLS"
            elif right == "P":
                side = " PUTS"
        return f"{symbol} {kind}{side}"

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
        self._sync_bound_tickers()
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
        official_unreal = self._official_unrealized()
        fast_unreal = self._live_unrealized(price or mark)
        day_pnl = self._official_daily_contract_pnl()
        day_label = _fmt_money(day_pnl) if day_pnl is not None else "n/a"
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
        realized_num = self._float_or_none(getattr(self._item, "realizedPNL", None))
        realized = _fmt_money(realized_num) if realized_num is not None else "n/a"

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
        if contract.secType in ("OPT", "FOP") and not has_live_quote:
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
        tail_row = Text("")
        tail_row.append_text(pct24_prefix)
        tail_row.append_text(pct24_value)
        tail_row.append("   ")
        tail_row.append("✦ Unreal ", style="#8fbfff")
        official_start = -1
        official_end = -1
        estimate_start = -1
        estimate_end = -1
        if official_unreal is not None:
            official_start = len(tail_row.plain)
            tail_row.append(_fmt_money(official_unreal))
            official_end = len(tail_row.plain)
            if fast_unreal is not None:
                tail_row.append(" (", style="dim")
                estimate_start = len(tail_row.plain)
                tail_row.append(_fmt_money(fast_unreal))
                estimate_end = len(tail_row.plain)
                tail_row.append(")", style="dim")
        elif fast_unreal is not None:
            estimate_start = len(tail_row.plain)
            tail_row.append(_fmt_money(fast_unreal))
            estimate_end = len(tail_row.plain)
            tail_row.append(" ≈est", style="dim")
        else:
            tail_row.append("n/a", style="dim")
        tail_row.append(" ", style="dim")
        tail_row.append("(", style="dim")
        tail_row.append("◷ Day ", style="#8aa0b6")
        day_start = len(tail_row.plain)
        tail_row.append(day_label)
        day_end = len(tail_row.plain)
        tail_row.append(")", style="dim")
        tail_row.append("   ")
        tail_row.append("Realized ", style="#8aa0b6")
        realized_start = len(tail_row.plain)
        tail_row.append(realized)
        realized_end = len(tail_row.plain)
        if official_start >= 0 and official_end > official_start:
            tail_row.stylize(self._pnl_style(official_unreal), official_start, official_end)
        if estimate_start >= 0 and estimate_end > estimate_start:
            tail_row.stylize(self._pnl_style(fast_unreal), estimate_start, estimate_end)
        tail_row.stylize(self._pnl_style(day_pnl), day_start, day_end)
        tail_row.stylize(self._pnl_style(realized_num), realized_start, realized_end)

        lines: list[Text] = [
            self._box_top(self._contract_header_title(contract), inner, style="#2d8fd5"),
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
        md_probe_row = self._market_data_probe_row()
        if md_probe_row is not None:
            lines.insert(3, self._box_row(md_probe_row, inner, style="#2d8fd5"))
        if no_quote_badge_row is not None:
            lines.insert(4 if md_probe_row is not None else 3, self._box_row(no_quote_badge_row, inner, style="#2d8fd5"))
        if close_only_badge_row is not None:
            close_insert_idx = 4 if no_quote_badge_row is not None else 3
            if md_probe_row is not None:
                close_insert_idx += 1
            lines.insert(close_insert_idx, self._box_row(close_only_badge_row, inner, style="#2d8fd5"))
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
            if ulast is None:
                ulast = _ticker_price(self._underlying_ticker) or _ticker_close(self._underlying_ticker)
            under_row = Text(f"{label}: ")
            under_row.append(f"{_fmt_quote(ubid)}/{_fmt_quote(uask)}/{_fmt_quote(ulast)}", style="cyan")
            lines.append(self._box_row(under_row, inner, style="#2d8fd5"))
        lines.append(self._box_bottom(inner, style="#2d8fd5"))
        return lines

    def _render_execution_block(self, *, panel_width: int) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP", "FUT"):
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
        custom = _round_to_tick(self._exec_custom_price, tick)
        if custom is None:
            custom = _round_to_tick(mid, tick)
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
        quote_stale = self._quote_is_stale_for_relentless(
            ticker=self._ticker,
            bid=bid,
            ask=ask,
            last=last,
        )
        open_shock = self._in_open_shock_window()
        relentless_buy = self._relentless_price(
            action="BUY",
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
        )
        relentless_sell = self._relentless_price(
            action="SELL",
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
        )
        relentless_delay_buy = self._exec_price_for_mode(
            "RELENTLESS_DELAY",
            "BUY",
            bid=bid,
            ask=ask,
            last=last,
            ticker=self._ticker,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
        )
        relentless_delay_sell = self._exec_price_for_mode(
            "RELENTLESS_DELAY",
            "SELL",
            bid=bid,
            ask=ask,
            last=last,
            ticker=self._ticker,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
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
                "relentless",
                f"RLT       B/S {_fmt_quote(relentless_buy)} / {_fmt_quote(relentless_sell)}",
            ),
            (
                "relentless_delay",
                f"RLT ⚔ Delay B/S {_fmt_quote(relentless_delay_buy)} / {_fmt_quote(relentless_delay_sell)}",
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
                "custom",
                f"CUSTOM     B/S {_fmt_quote(custom)} / {_fmt_quote(custom)}",
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
        notice_line = self._orders_notice_line()
        notice_reserved_rows = 0
        if notice_line is not None:
            lines.append(self._box_rule("Order Feed", inner, style="#2f78c4"))
            notice_plain = notice_line.plain
            prefix, sep, message = notice_plain.partition(" ")
            if sep:
                prefix_with_space = f"{prefix}{sep}"
                wrapped_notice = textwrap.wrap(
                    message,
                    width=max(inner, 1),
                    initial_indent=prefix_with_space,
                    subsequent_indent=" " * len(prefix_with_space),
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            else:
                wrapped_notice = textwrap.wrap(
                    notice_plain,
                    width=max(inner, 1),
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            if not wrapped_notice:
                wrapped_notice = [""]
            notice_style = notice_line.style
            for chunk in wrapped_notice:
                lines.append(self._box_row(Text(chunk, style=notice_style), inner, style="#2f78c4"))
            notice_reserved_rows = 1 + len(wrapped_notice)

        if not trades:
            self._orders_selected = 0
            self._orders_scroll = 0
            lines.append(self._box_row(Text("No open orders", style="dim"), inner, style="#2f78c4"))
        else:
            if self._orders_selected >= len(trades):
                self._orders_selected = len(trades) - 1
            reserved_without_trades = 7 + notice_reserved_rows
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
            visible_trades = list(trades[start:end])
            state_snapshot = self._order_state_snapshot_for_trades(visible_trades)
            for idx in range(start, end):
                trade = trades[idx]
                line = self._format_order_line(
                    trade,
                    width=inner,
                    state_snapshot=state_snapshot,
                )
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
        order_id, perm_id = self._trade_order_ids(trade)
        return order_id or perm_id

    @staticmethod
    def _trade_order_ids(trade: Trade) -> tuple[int, int]:
        order = getattr(trade, "order", None)
        try:
            order_id = int(getattr(order, "orderId", 0) or 0)
        except (TypeError, ValueError):
            order_id = 0
        try:
            perm_id = int(getattr(order, "permId", 0) or 0)
        except (TypeError, ValueError):
            perm_id = 0
        return max(0, order_id), max(0, perm_id)

    def _latest_trade_for_ids(
        self,
        *,
        order_id: int,
        perm_id: int,
        fallback: Trade,
    ) -> Trade:
        lookup = getattr(self._client, "trade_for_order_ids", None)
        if not callable(lookup):
            return fallback
        try:
            refreshed = lookup(
                order_id=int(order_id),
                perm_id=int(perm_id),
                include_closed=True,
            )
        except TypeError:
            try:
                refreshed = lookup(int(order_id), int(perm_id))
            except Exception:
                refreshed = None
        except Exception:
            refreshed = None
        if refreshed is None:
            return fallback
        return refreshed

    async def _client_reconcile_order_state(
        self,
        *,
        order_id: int,
        perm_id: int,
        force: bool = False,
    ) -> dict[str, object] | None:
        reconcile = getattr(self._client, "reconcile_order_state", None)
        if not callable(reconcile):
            return None
        try:
            payload = await reconcile(
                order_id=int(order_id),
                perm_id=int(perm_id),
                force=bool(force),
            )
        except TypeError:
            try:
                payload = await reconcile(int(order_id), int(perm_id))
            except Exception:
                payload = None
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return None
        return payload

    def _client_current_order_state(
        self,
        *,
        order_id: int,
        perm_id: int,
    ) -> dict[str, object] | None:
        current_state = getattr(self._client, "current_order_state", None)
        if not callable(current_state):
            return None
        try:
            payload = current_state(
                order_id=int(order_id),
                perm_id=int(perm_id),
            )
        except TypeError:
            try:
                payload = current_state(int(order_id), int(perm_id))
            except Exception:
                payload = None
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _status_compact(status: str) -> str:
        text = str(status or "").strip()
        if not text:
            return "n/a"
        return text.replace("PreSubmitted", "PreSub")[:9]

    @staticmethod
    def _state_order_keys(*, order_id: int, perm_id: int) -> list[int]:
        keys: list[int] = []
        if int(order_id or 0) > 0:
            keys.append(int(order_id))
        if int(perm_id or 0) > 0 and int(perm_id) not in keys:
            keys.append(int(perm_id))
        return keys

    def _chase_state_for_ids(self, *, order_id: int, perm_id: int) -> dict[str, object] | None:
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            state = _DETAIL_CHASE_STATE_BY_ORDER.get(int(key))
            if isinstance(state, dict):
                return state
        return None

    def _register_chase_task(self, task: asyncio.Task, *, order_id: int, perm_id: int) -> None:
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            self._chase_task_by_order[int(key)] = task

    def _unregister_chase_task(self, task: asyncio.Task) -> None:
        for key, value in list(self._chase_task_by_order.items()):
            if value is task:
                self._chase_task_by_order.pop(int(key), None)

    def _cancel_chase_for_ids(self, *, order_id: int, perm_id: int) -> None:
        tasks: set[asyncio.Task] = set()
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            task = self._chase_task_by_order.get(int(key))
            if task is not None:
                tasks.add(task)
        for task in tasks:
            if not task.done():
                task.cancel()

    def _mark_cancel_requested(self, *, order_id: int, perm_id: int) -> None:
        ts = monotonic()
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            self._cancel_requested_at_by_order[int(key)] = float(ts)

    def _clear_cancel_requested(self, *, order_id: int, perm_id: int) -> None:
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            self._cancel_requested_at_by_order.pop(int(key), None)

    def _cancel_requested_for_ids(self, *, order_id: int, perm_id: int) -> bool:
        now = monotonic()
        ttl = max(1.0, float(self._CANCEL_REQUEST_TTL_SEC))
        for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
            ts = self._cancel_requested_at_by_order.get(int(key))
            if ts is None:
                continue
            if (now - float(ts)) <= ttl:
                return True
            self._cancel_requested_at_by_order.pop(int(key), None)
        return False

    def _set_chase_state(
        self,
        *,
        order_id: int,
        perm_id: int,
        updates: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        keys = self._state_order_keys(order_id=order_id, perm_id=perm_id)
        if not keys:
            return None
        state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id)
        if state is None:
            state = {}
        if updates:
            state.update(dict(updates))
        for key in keys:
            _DETAIL_CHASE_STATE_BY_ORDER[int(key)] = state
        return state

    def _clear_chase_state(self, *, order_id: int, perm_id: int) -> None:
        keys = self._state_order_keys(order_id=order_id, perm_id=perm_id)
        state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id)
        if isinstance(state, dict):
            for key, value in list(_DETAIL_CHASE_STATE_BY_ORDER.items()):
                if value is state:
                    _DETAIL_CHASE_STATE_BY_ORDER.pop(int(key), None)
        for key in keys:
            _DETAIL_CHASE_STATE_BY_ORDER.pop(int(key), None)

    def _armed_mode_line(self) -> Text:
        selected = self._selected_exec_mode()
        selected_label = self._exec_mode_label(selected)
        buy = self._initial_exec_price("BUY", mode=selected)
        sell = self._initial_exec_price("SELL", mode=selected)
        line = Text("Armed ")
        line.append(selected_label, style="yellow")
        if selected == "AUTO":
            line.append(" (OPT->MID->AGG->CROSS)", style="dim")
        elif selected == "RELENTLESS":
            line.append(" (completion-first)", style="dim")
        elif selected == "RELENTLESS_DELAY":
            line.append(" (delay-aware offense)", style="dim")
        line.append("  B/S ")
        line.append(f"{_fmt_quote(buy)}/{_fmt_quote(sell)}", style="bright_white")
        return line

    def _active_chase_line(self, trades: list[Trade]) -> Text:
        for trade in trades:
            order_id, perm_id = self._trade_order_ids(trade)
            display_id = order_id or perm_id
            state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id)
            if not isinstance(state, dict):
                continue
            selected = str(state.get("selected") or "-")
            active = str(state.get("active") or "-")
            target = _safe_num(state.get("target_price"))
            try:
                mods = int(state.get("mods") or 0)
            except (TypeError, ValueError):
                mods = 0
            line = Text(f"Chase #{display_id} ")
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
        order_id, perm_id = self._trade_order_ids(trade)
        state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id)
        if not isinstance(state, dict):
            return "-"
        selected = str(state.get("selected") or "-")
        active = str(state.get("active") or selected or "-")
        if selected == "AUTO":
            return f"A>{active}"[:9]
        if active and active != selected:
            return f"{selected}>{active}"[:9]
        return selected[:9]

    @staticmethod
    def _should_probe_effective_status(raw_status: str) -> bool:
        status = str(raw_status or "").strip()
        return status in ("", "PendingSubmit", "PendingSubmission", "ApiPending")

    def _order_state_snapshot_for_trades(
        self,
        trades: list[Trade],
    ) -> dict[int, dict[str, object]]:
        snapshot: dict[int, dict[str, object]] = {}
        for trade in trades:
            order_id, perm_id = self._trade_order_ids(trade)
            keys = self._state_order_keys(order_id=order_id, perm_id=perm_id)
            if not keys:
                continue
            if all(int(key) in snapshot for key in keys):
                continue
            raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
            if not self._should_probe_effective_status(raw_status):
                continue
            payload = self._client_current_order_state(order_id=order_id, perm_id=perm_id)
            if not isinstance(payload, dict):
                continue
            for key in keys:
                snapshot[int(key)] = payload
        return snapshot

    def _effective_status_for_trade(
        self,
        trade: Trade,
        *,
        state_snapshot: dict[int, dict[str, object]] | None = None,
    ) -> str | None:
        order_id, perm_id = self._trade_order_ids(trade)
        raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
        state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id)
        if isinstance(state, dict):
            effective = str(state.get("effective_status") or "").strip()
            if effective and effective != raw_status:
                return effective
        payload = None
        if isinstance(state_snapshot, dict):
            for key in self._state_order_keys(order_id=order_id, perm_id=perm_id):
                candidate = state_snapshot.get(int(key))
                if isinstance(candidate, dict):
                    payload = candidate
                    break
            if payload is None and not self._should_probe_effective_status(raw_status):
                return None
        if payload is None:
            payload = self._client_current_order_state(order_id=order_id, perm_id=perm_id)
        if isinstance(payload, dict):
            effective = str(payload.get("effective_status") or "").strip()
            if effective and effective != raw_status:
                return effective
        return None

    def _format_order_line(
        self,
        trade: Trade,
        *,
        width: int,
        state_snapshot: dict[int, dict[str, object]] | None = None,
    ) -> Text:
        contract = trade.contract
        label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or "?"
        label = label[:10]
        raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
        status = self._status_compact(raw_status)
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
        effective_status = self._effective_status_for_trade(trade, state_snapshot=state_snapshot)
        effective_hint = ""
        if effective_status:
            effective_hint = f" ~{self._status_compact(effective_status)}"
        line = (
            f"{label:<10} {status:<9} {mode:<9} {side:<1} {qty:>3} "
            f"{type_label:<12} {filled:>4}/{remaining:<4} #{order_id}{effective_hint}"
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
        order_id, perm_id = self._trade_order_ids(trade)
        order_ref = int(order_id or perm_id or 0)
        self._mark_cancel_requested(order_id=order_id, perm_id=perm_id)
        try:
            await self._client.cancel_trade(trade)
        except Exception as exc:
            self._exec_status = f"Cancel error: {exc}"
            self._set_orders_notice(self._exec_status, level="error")
            self._clear_cancel_requested(order_id=order_id, perm_id=perm_id)
            self._render_details(sample=False)
            return
        error_payload = (
            await self._await_order_error(order_id, perm_id, attempts=4, interval_sec=0.1)
            if order_ref
            else None
        )
        should_clear_cancel_intent = False
        if error_payload is not None:
            error_code, error_message = error_payload
            error_prefix = f"IB {error_code}: " if error_code else "IB: "
            level = "warn" if error_code in (10147, 10148, 10149) else "error"
            if order_ref:
                self._exec_status = f"Cancel #{order_ref}: {error_prefix}{error_message}"
            else:
                self._exec_status = f"Cancel: {error_prefix}{error_message}"
            self._set_orders_notice(self._exec_status, level=level)
            should_clear_cancel_intent = True
        else:
            ack_status = ""
            if order_ref:
                for attempt in range(5):
                    payload = self._client_current_order_state(order_id=order_id, perm_id=perm_id)
                    if not isinstance(payload, dict) and attempt >= 2:
                        payload = await self._client_reconcile_order_state(
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
                    self._exec_status = f"Cancel {ack_status} #{order_ref}"
                else:
                    self._exec_status = f"Cancel sent #{order_ref} (awaiting broker ack)"
            else:
                self._exec_status = "Cancel sent"
            self._set_orders_notice(self._exec_status, level="warn")
            should_clear_cancel_intent = ack_status in (
                "Cancelled",
                "ApiCancelled",
                "Inactive",
                "Filled",
            )
        if should_clear_cancel_intent:
            self._clear_cancel_requested(order_id=order_id, perm_id=perm_id)
        self._render_details(sample=False)

    def _handle_digit(self, char: str) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "custom":
            if char not in "0123456789.":
                return
            self._exec_custom_input = _append_digit(self._exec_custom_input, char, allow_decimal=True)
            parsed = self._parse_custom_price(self._exec_custom_input)
            if parsed is not None:
                self._exec_custom_price = parsed
        else:
            if char not in "0123456789":
                return
            self._exec_qty_input = _append_digit(self._exec_qty_input, char, allow_decimal=False)
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        self._render_details(sample=False)

    def _handle_backspace(self) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "custom":
            self._exec_custom_input = self._exec_custom_input[:-1]
            parsed = self._parse_custom_price(self._exec_custom_input)
            self._exec_custom_price = parsed
        else:
            self._exec_qty_input = self._exec_qty_input[:-1]
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        self._render_details(sample=False)

    def _adjust_qty(self, direction: int) -> None:
        next_qty = max(1, int(self._exec_qty) + direction)
        self._exec_qty = next_qty
        self._exec_qty_input = str(next_qty)

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
        current = _round_to_tick(self._exec_custom_price, tick)
        if current is None:
            current = _round_to_tick(last_ref, tick)
        if current is None:
            return
        next_value = _round_to_tick(float(current) + (float(direction) * float(tick)), tick)
        if next_value is None or next_value <= 0:
            return
        self._exec_custom_price = float(next_value)
        self._exec_custom_input = f"{self._exec_custom_price:.{_tick_decimals(tick)}f}"

    @classmethod
    def _in_open_shock_window(cls) -> bool:
        now = _now_et().time()
        if now < dtime(9, 30) or now >= dtime(16, 0):
            return False
        open_sec = (9 * 60 * 60) + (30 * 60)
        now_sec = (now.hour * 60 * 60) + (now.minute * 60) + now.second
        elapsed = max(0, now_sec - open_sec)
        return elapsed <= int(cls._RELENTLESS_OPEN_SHOCK_WINDOW_SEC)

    def _quote_is_stale_for_relentless(
        self,
        *,
        ticker: Ticker | None,
        bid: float | None,
        ask: float | None,
        last: float | None,
    ) -> bool:
        has_actionable = bool(
            (bid is not None and ask is not None and bid <= ask)
            or (last is not None)
        )
        if not has_actionable:
            return True
        if ticker is None:
            return False
        top_updated_mono = getattr(ticker, "tbTopQuoteUpdatedMono", None)
        try:
            top_age_sec = (
                max(0.0, monotonic() - float(top_updated_mono))
                if top_updated_mono is not None
                else None
            )
        except (TypeError, ValueError):
            top_age_sec = None
        return bool(
            top_age_sec is not None
            and top_age_sec >= float(self._RELENTLESS_STALE_TOP_AGE_SEC)
        )

    def _relentless_spread_pressure(self, *, spread: float | None, tick: float) -> float:
        if spread is None or spread <= 0 or tick <= 0:
            return 1.0
        recent = [float(value) for value in list(self._spread_samples)[-24:] if float(value) > 0]
        if recent:
            recent.sort()
            baseline = float(recent[len(recent) // 2])
        else:
            baseline = float(spread)
        baseline = max(float(tick), baseline)
        return max(0.5, float(spread) / baseline)

    def _relentless_min_reprice_sec(
        self,
        *,
        quote_stale: bool,
        open_shock: bool,
        no_progress_reprices: int,
        spread_pressure: float,
    ) -> float:
        if open_shock and (
            no_progress_reprices >= 2
            or spread_pressure >= float(self._RELENTLESS_SPREAD_PRESSURE_TRIGGER)
        ):
            return float(self._RELENTLESS_MIN_REPRICE_SEC_HYPER)
        if quote_stale or open_shock:
            return float(self._RELENTLESS_MIN_REPRICE_SEC_SHOCK)
        if no_progress_reprices >= 4:
            return float(self._RELENTLESS_MIN_REPRICE_SEC_SHOCK)
        return float(self._RELENTLESS_MIN_REPRICE_SEC)

    def _relentless_price(
        self,
        *,
        action: str,
        bid: float | None,
        ask: float | None,
        last_ref: float | None,
        tick: float,
        elapsed_sec: float,
        quote_stale: bool,
        open_shock: bool,
        no_progress_reprices: int,
        arrival_ref: float | None,
        direction_sign_override: float | None = None,
    ) -> float | None:
        cross = _round_to_tick(ask if action == "BUY" else bid, tick)
        if cross is None:
            cross = _round_to_tick(last_ref, tick)
        if cross is None:
            return None
        spread = (
            float(ask) - float(bid)
            if bid is not None and ask is not None and ask >= bid
            else None
        )
        if spread is None or spread <= 0:
            spread = float(tick) * 2.0
        spread_pressure = self._relentless_spread_pressure(spread=spread, tick=tick)
        spread_cap_mult = 24.0 if open_shock else 12.0
        if spread_pressure >= float(self._RELENTLESS_SPREAD_PRESSURE_TRIGGER):
            spread_cap_mult = max(spread_cap_mult, 18.0)
        spread_cap = max(float(tick), float(tick) * spread_cap_mult)
        step = max(float(tick), min(float(spread) / 3.0, spread_cap))
        if spread_pressure >= float(self._RELENTLESS_SPREAD_PRESSURE_HYPER):
            step = max(step, float(tick) * 2.0)

        elapsed = max(0.0, float(elapsed_sec))
        if elapsed < float(self._RELENTLESS_BASE_CROSS_SEC):
            x = 0
        else:
            interval = (
                float(self._RELENTLESS_STEP_SEC_SHOCK)
                if open_shock
                else float(self._RELENTLESS_STEP_SEC)
            )
            if quote_stale:
                interval *= 0.5
            if int(no_progress_reprices) >= 2:
                interval *= 0.5
            x = 1 + int((elapsed - float(self._RELENTLESS_BASE_CROSS_SEC)) // max(interval, 0.5))

        if quote_stale:
            x += 2
        if open_shock:
            x += 1
        if int(no_progress_reprices) >= 1:
            x += min(8, int(no_progress_reprices))
        if spread_pressure >= float(self._RELENTLESS_SPREAD_PRESSURE_TRIGGER):
            x += min(4, int(spread_pressure))
        if spread_pressure >= float(self._RELENTLESS_SPREAD_PRESSURE_HYPER):
            x = max(x, 6)

        current_ref = _midpoint(bid, ask) or last_ref
        if arrival_ref is not None and current_ref is not None:
            adverse = (
                float(current_ref) - float(arrival_ref)
                if action == "BUY"
                else float(arrival_ref) - float(current_ref)
            )
            if adverse >= float(spread) * 0.75:
                x += 1
            if adverse >= float(spread) * 1.5:
                x += 2
            if adverse >= float(spread) * 2.5:
                x = max(x, 8)

        if open_shock and int(no_progress_reprices) >= 3:
            x = max(x, 6)

        x = max(0, min(int(self._RELENTLESS_MAX_EDGE_TICKS), int(x)))
        side_sign = 1.0 if action == "BUY" else -1.0
        if direction_sign_override is not None and math.isfinite(float(direction_sign_override)):
            side_sign = 1.0 if float(direction_sign_override) >= 0 else -1.0
        target = float(cross) + (side_sign * float(step) * float(x))
        return _round_to_tick(target, tick)

    def _relentless_delay_cap_ticks(
        self,
        *,
        anchor_price: float,
        spread: float | None,
        tick: float,
        favorable: bool,
        recoveries: int,
    ) -> int:
        if tick <= 0 or anchor_price <= 0:
            return 1
        pct_cap = (
            float(anchor_price) * float(self._RELENTLESS_DELAY_FAVORABLE_PCT)
            if favorable
            else float(anchor_price) * float(self._RELENTLESS_DELAY_ADVERSE_PCT)
        )
        spread_mult = (
            float(self._RELENTLESS_DELAY_FAVORABLE_SPREAD_MULT)
            if favorable
            else float(self._RELENTLESS_DELAY_ADVERSE_SPREAD_MULT)
        )
        hard_cap = (
            max(1, int(self._RELENTLESS_DELAY_FAVORABLE_MAX_TICKS))
            if favorable
            else max(1, int(self._RELENTLESS_DELAY_ADVERSE_MAX_TICKS))
        )
        spread_cap = (
            float(spread) * spread_mult
            if spread is not None and spread > 0
            else float(hard_cap) * float(tick)
        )
        cap_ticks = min(
            float(pct_cap) / float(tick),
            float(spread_cap) / float(tick),
            float(hard_cap),
        )
        shrink_base = min(0.95, max(0.25, float(self._RELENTLESS_DELAY_SHRINK_PER_REJECT)))
        shrink = float(shrink_base) ** max(0, int(recoveries) - 1)
        shrunk_ticks = max(1.0, float(cap_ticks) * float(shrink))
        return max(1, min(hard_cap, int(math.floor(shrunk_ticks))))

    @staticmethod
    def _cap_price_hint_from_trade(trade: Trade) -> float | None:
        cap = _safe_float(getattr(getattr(trade, "orderStatus", None), "mktCapPrice", None))
        if cap is None or cap <= 0:
            return None
        return float(cap)

    @classmethod
    def _price_hint_from_error_message(cls, message: str) -> float | None:
        text = str(message or "").strip()
        if not text:
            return None
        candidates: list[float] = []
        for match in cls._RELENTLESS_DELAY_PRICE_HINT_RE.finditer(text):
            token = str(match.group(1) or "").replace(",", "")
            try:
                value = float(token)
            except (TypeError, ValueError):
                continue
            if value <= 0 or not math.isfinite(value):
                continue
            candidates.append(float(value))
        if not candidates:
            return None
        return float(candidates[-1])

    def _relentless_delay_price(
        self,
        *,
        action: str,
        bid: float | None,
        ask: float | None,
        last_ref: float | None,
        tick: float,
        ticker: Ticker | None,
        elapsed_sec: float,
        no_progress_reprices: int,
        delay_recoveries: int = 0,
        delay_anchor_price: float | None = None,
        delay_sweep_anchor_price: float | None = None,
    ) -> float | None:
        cross = _round_to_tick(ask if action == "BUY" else bid, tick)
        if cross is None:
            cross = _round_to_tick(last_ref, tick)
        if cross is None:
            return None
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        last_rounded = _round_to_tick(last_ref, tick)
        base_ref = mid if mid is not None else (last_rounded if last_rounded is not None else cross)
        anchor = _round_to_tick(_safe_float(delay_sweep_anchor_price), tick)
        if anchor is None:
            anchor = base_ref
        if anchor is None or anchor <= 0:
            return None
        recoveries = max(1, int(delay_recoveries))
        spread = (
            float(ask) - float(bid)
            if (ask is not None and bid is not None and ask >= bid)
            else None
        )
        favorable_cap_ticks = self._relentless_delay_cap_ticks(
            anchor_price=float(anchor),
            spread=spread,
            tick=tick,
            favorable=True,
            recoveries=recoveries,
        )
        adverse_cap_ticks = self._relentless_delay_cap_ticks(
            anchor_price=float(anchor),
            spread=spread,
            tick=tick,
            favorable=False,
            recoveries=recoveries,
        )
        seq_step = 1 + ((recoveries - 1) // 2)
        favorable_leg, leg_sign = self._relentless_delay_leg(action, recoveries)
        leg_cap = favorable_cap_ticks if favorable_leg else adverse_cap_ticks
        step_ticks = max(1, min(int(leg_cap), int(seq_step)))
        target = float(anchor) + (leg_sign * float(tick) * float(step_ticks))
        cap_hint = _safe_float(delay_anchor_price)
        if cap_hint is not None and cap_hint > 0:
            if action == "BUY":
                target = min(float(target), float(cap_hint))
            else:
                target = max(float(target), float(cap_hint))
        rounded = _round_to_tick(target, tick)
        if rounded is None or rounded <= 0:
            return None
        return float(rounded)

    def _relentless_delay_sweep_span(self) -> int:
        span = max(2, int(self._RELENTLESS_DELAY_RECOVER_ATTEMPTS))
        # Keep span even so wrap points preserve the favorable/adverse alternation.
        if (span % 2) != 0:
            span += 1
        return int(span)

    def _relentless_delay_next_step(self, *, prior_recoveries: int) -> int:
        span = self._relentless_delay_sweep_span()
        base = max(0, int(prior_recoveries))
        return 1 + (base % span)

    @staticmethod
    def _relentless_delay_leg(action: str, recoveries: int) -> tuple[bool, float]:
        step = max(1, int(recoveries))
        favorable_leg = (step % 2) == 1
        side_sign = 1.0 if str(action or "").strip().upper() == "BUY" else -1.0
        leg_sign = (-1.0 * side_sign) if favorable_leg else side_sign
        return favorable_leg, float(leg_sign)

    def _submit_order(self, action: str) -> None:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP", "FUT"):
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
            applied_price = _safe_num(getattr(getattr(trade, "order", None), "lmtPrice", None))
            if applied_price is None or applied_price <= 0:
                applied_price = float(price)
            mode_label = self._exec_mode_label(mode)
            self._exec_status = f"Sent {action} {qty} @ {float(applied_price):.2f} [{mode_label}]"
            order_id, perm_id = self._trade_order_ids(trade)
            order_ref = order_id or perm_id
            if order_ref:
                self._set_orders_notice(
                    f"Submitted #{order_ref} {action} {qty} @ {float(applied_price):.2f} [{mode_label}]",
                    level="info",
                )
            else:
                self._set_orders_notice(
                    f"Submitted {action} {qty} @ {float(applied_price):.2f} [{mode_label}]",
                    level="info",
                )
            if order_ref:
                seeded = "OPTIMISTIC" if mode == "AUTO" else mode
                updates: dict[str, object] = {
                    "selected": self._exec_mode_label(mode),
                    "active": self._exec_mode_label(seeded),
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
                self._set_chase_state(
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
                self._register_chase_task(task, order_id=order_id, perm_id=perm_id)

                def _on_chase_done(
                    done_task: asyncio.Task,
                    *,
                    seed_order_id: int = order_id,
                    seed_perm_id: int = perm_id,
                ) -> None:
                    self._chase_tasks.discard(done_task)
                    self._unregister_chase_task(done_task)
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
                    self._exec_status = f"Chase task error: {exc}"
                    self._set_orders_notice(
                        f"{order_label} chase task error: {exc}",
                        level="error",
                    )
                    self._render_details_if_mounted(sample=False)

                task.add_done_callback(_on_chase_done)
        except Exception as exc:
            self._exec_status = f"Exec error: {exc}"
            self._set_orders_notice(self._exec_status, level="error")
        self._render_details_if_mounted(sample=False)

    def _initial_exec_price(self, action: str, *, mode: str = "AUTO") -> float | None:
        bid = self._quote_num(self._ticker.bid) if self._ticker else None
        ask = self._quote_num(self._ticker.ask) if self._ticker else None
        last = self._quote_num(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if self._item.contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, self._ticker, last_ref)
        if mode == "CUSTOM":
            custom = _round_to_tick(self._exec_custom_price, tick)
            if custom is not None:
                return custom
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        if mode in ("RELENTLESS", "RELENTLESS_DELAY"):
            quote_stale = self._quote_is_stale_for_relentless(
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
                open_shock=self._in_open_shock_window(),
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
        last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(self._item.contract, ticker_ref, last_ref)
        if mode == "CUSTOM":
            custom = _round_to_tick(self._exec_custom_price, tick)
            if custom is not None:
                return custom
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        if mode == "RELENTLESS":
            return self._relentless_price(
                action=action,
                bid=bid,
                ask=ask,
                last_ref=last_ref,
                tick=tick,
                elapsed_sec=elapsed_sec,
                quote_stale=bool(quote_stale),
                open_shock=bool(open_shock),
                no_progress_reprices=int(no_progress_reprices),
                arrival_ref=arrival_ref,
            )
        if mode == "RELENTLESS_DELAY":
            if int(delay_recoveries) <= 0:
                locked_sign = _safe_float(delay_locked_price_dir)
                if locked_sign is not None and math.isfinite(float(locked_sign)):
                    locked_sign = 1.0 if float(locked_sign) >= 0 else -1.0
                return self._relentless_price(
                    action=action,
                    bid=bid,
                    ask=ask,
                    last_ref=last_ref,
                    tick=tick,
                    elapsed_sec=elapsed_sec,
                    quote_stale=bool(quote_stale),
                    open_shock=bool(open_shock),
                    no_progress_reprices=int(no_progress_reprices),
                    arrival_ref=arrival_ref,
                    direction_sign_override=locked_sign,
                )
            return self._relentless_delay_price(
                action=action,
                bid=bid,
                ask=ask,
                last_ref=last_ref,
                tick=tick,
                ticker=ticker_ref,
                elapsed_sec=elapsed_sec,
                no_progress_reprices=int(no_progress_reprices),
                delay_recoveries=int(delay_recoveries),
                delay_anchor_price=delay_anchor_price,
                delay_sweep_anchor_price=delay_sweep_anchor_price,
            )
        value = _limit_price_for_mode(bid, ask, last_ref, action=action, mode=mode)
        if value is None:
            return _round_to_tick(last_ref, tick) if last_ref is not None else None
        return _round_to_tick(float(value), tick)

    async def _chase_until_filled(self, trade: Trade, action: str, *, mode: str = "AUTO") -> None:
        order_id, perm_id = self._trade_order_ids(trade)
        con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
        chase_owner = f"details-chase:{order_id or perm_id or con_id or id(trade)}"
        try:
            await self._client.ensure_ticker(trade.contract, owner=chase_owner)
        except Exception:
            pass
        started = asyncio.get_running_loop().time()
        last_reprice_ts: float | None = None
        prev_mode: str | None = None
        prev_quote_sig: tuple[float | None, float | None, float | None] | None = None
        selected_label = self._exec_mode_label(mode)
        selected_mode_clean = str(mode or "").strip().upper()
        selected_is_delay_mode = selected_mode_clean == "RELENTLESS_DELAY"
        arrival_ref: float | None = None
        no_progress_reprices = 0
        last_filled_qty = 0.0
        last_live_probe_ts: float | None = None
        last_reconcile_ts: float | None = None
        last_force_reconcile_ts: float | None = None
        last_modify_error_ts: float | None = None
        pending_since_ts: float | None = None
        try:
            while True:
                trade = self._latest_trade_for_ids(
                    order_id=order_id,
                    perm_id=perm_id,
                    fallback=trade,
                )
                live_order_id, live_perm_id = self._trade_order_ids(trade)
                if live_order_id > 0:
                    order_id = int(live_order_id)
                if live_perm_id > 0:
                    perm_id = int(live_perm_id)
                live_con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
                if live_con_id > 0:
                    con_id = int(live_con_id)
                current_task = asyncio.current_task()
                if current_task is not None:
                    self._register_chase_task(
                        current_task,
                        order_id=order_id,
                        perm_id=perm_id,
                    )
                loop_now = asyncio.get_running_loop().time()
                status_raw = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
                status_effective = status_raw
                live_state_payload = (
                    self._client_current_order_state(order_id=order_id, perm_id=perm_id)
                    if (order_id or perm_id)
                    else None
                )
                if isinstance(live_state_payload, dict):
                    live_trade = live_state_payload.get("trade")
                    if isinstance(live_trade, Trade):
                        trade = live_trade
                        live_order_id, live_perm_id = self._trade_order_ids(trade)
                        if live_order_id > 0:
                            order_id = int(live_order_id)
                        if live_perm_id > 0:
                            perm_id = int(live_perm_id)
                    effective_live = str(
                        live_state_payload.get("effective_status") or ""
                    ).strip()
                    if effective_live:
                        status_effective = effective_live
                terminal_statuses = ("Filled", "Cancelled", "ApiCancelled", "Inactive")
                repricable_statuses = ("PreSubmitted", "Submitted")
                pending_statuses = ("PendingSubmit", "PendingSubmission", "ApiPending")
                cancel_requested = self._cancel_requested_for_ids(
                    order_id=order_id,
                    perm_id=perm_id,
                )
                if status_raw in pending_statuses:
                    if pending_since_ts is None:
                        pending_since_ts = loop_now
                else:
                    pending_since_ts = None
                pending_age = (
                    max(0.0, loop_now - pending_since_ts)
                    if pending_since_ts is not None
                    else 0.0
                )
                reconcile_payload: dict[str, object] | None = None
                if order_id or perm_id:
                    should_reconcile = False
                    force_reconcile = False
                    if status_raw in pending_statuses and pending_age >= float(self._CHASE_PENDING_ACK_SEC):
                        if (
                            last_reconcile_ts is None
                            or (loop_now - last_reconcile_ts) >= float(self._CHASE_RECONCILE_INTERVAL_SEC)
                        ):
                            should_reconcile = True
                            force_candidate = bool(
                                pending_age >= (float(self._CHASE_PENDING_ACK_SEC) * 2.0)
                            )
                            if force_candidate:
                                force_reconcile = bool(
                                    last_force_reconcile_ts is None
                                    or (
                                        (loop_now - last_force_reconcile_ts)
                                        >= float(self._CHASE_FORCE_RECONCILE_INTERVAL_SEC)
                                    )
                                )
                    if should_reconcile:
                        reconcile_payload = await self._client_reconcile_order_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            force=bool(force_reconcile),
                        )
                        last_reconcile_ts = loop_now
                        if force_reconcile:
                            last_force_reconcile_ts = loop_now
                if reconcile_payload is not None:
                    reconciled_trade = reconcile_payload.get("trade")
                    if isinstance(reconciled_trade, Trade):
                        trade = reconciled_trade
                        live_order_id, live_perm_id = self._trade_order_ids(trade)
                        if live_order_id > 0:
                            order_id = int(live_order_id)
                        if live_perm_id > 0:
                            perm_id = int(live_perm_id)
                    status_effective = str(
                        reconcile_payload.get("effective_status") or status_effective
                    ).strip() or status_effective
                try:
                    filled_now = float(getattr(getattr(trade, "orderStatus", None), "filled", 0.0) or 0.0)
                except (TypeError, ValueError):
                    filled_now = 0.0
                if reconcile_payload is not None:
                    try:
                        reconciled_filled = float(reconcile_payload.get("filled_qty") or 0.0)
                    except (TypeError, ValueError):
                        reconciled_filled = 0.0
                    if reconciled_filled > filled_now:
                        filled_now = float(reconciled_filled)
                if isinstance(live_state_payload, dict):
                    try:
                        live_filled = float(live_state_payload.get("filled_qty") or 0.0)
                    except (TypeError, ValueError):
                        live_filled = 0.0
                    if live_filled > filled_now:
                        filled_now = float(live_filled)
                prev_filled_qty = float(last_filled_qty)
                fill_progress = bool(filled_now > (prev_filled_qty + 1e-9))
                if fill_progress:
                    last_filled_qty = float(filled_now)
                    no_progress_reprices = 0
                is_done = False
                try:
                    is_done = bool(trade.isDone())
                except Exception:
                    is_done = False
                reconcile_terminal = bool(reconcile_payload and reconcile_payload.get("is_terminal"))
                live_terminal = bool(live_state_payload and live_state_payload.get("is_terminal"))
                if (
                    status_effective in terminal_statuses
                    or status_raw in terminal_statuses
                    or is_done
                    or reconcile_terminal
                    or live_terminal
                ):
                    order_ref = int(order_id or perm_id or 0)
                    order_label = f"#{order_ref}" if order_ref else "order"
                    error_payload = (
                        self._consume_order_error(order_id, perm_id)
                        if order_ref
                        else None
                    )
                    status_label = status_effective or status_raw
                    if not status_label:
                        status_label = "Done"
                    elif is_done and status_label not in terminal_statuses:
                        status_label = f"Done ({status_label})"
                    if status_raw and status_effective and status_effective != status_raw:
                        status_label = f"{status_label} [{status_raw}]"
                    if error_payload is not None:
                        error_code, error_message = error_payload
                        if selected_is_delay_mode and error_code == 202 and not cancel_requested:
                            state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id) or {}
                            try:
                                prior_recoveries = int(state.get("delay_recoveries") or 0)
                            except (TypeError, ValueError):
                                prior_recoveries = 0
                            first_202_ts = _safe_float(state.get("delay_first_202_ts"))
                            recover_window_sec = max(
                                1.0,
                                float(self._RELENTLESS_DELAY_RECOVER_WINDOW_SEC),
                            )
                            if (
                                first_202_ts is None
                                or first_202_ts <= 0
                                or (float(loop_now) - float(first_202_ts)) > recover_window_sec
                            ):
                                first_202_ts = float(loop_now)
                            sweep_span = self._relentless_delay_sweep_span()
                            next_recoveries = self._relentless_delay_next_step(
                                prior_recoveries=prior_recoveries
                            )
                            favorable_leg, leg_sign = self._relentless_delay_leg(
                                action,
                                next_recoveries,
                            )
                            anchor_hint = (
                                self._cap_price_hint_from_trade(trade)
                                or self._price_hint_from_error_message(error_message)
                                or _safe_float(state.get("delay_anchor_price"))
                            )
                            ticker_retry = self._client.ticker_for_con_id(con_id) if con_id else None
                            bid_retry = (
                                self._quote_num(getattr(ticker_retry, "bid", None))
                                if ticker_retry
                                else None
                            )
                            ask_retry = (
                                self._quote_num(getattr(ticker_retry, "ask", None))
                                if ticker_retry
                                else None
                            )
                            last_retry = (
                                self._quote_num(getattr(ticker_retry, "last", None))
                                if ticker_retry
                                else None
                            )
                            order_price_now = _safe_num(
                                getattr(getattr(trade, "order", None), "lmtPrice", None)
                            )
                            sweep_anchor = _safe_float(state.get("delay_sweep_anchor_price"))
                            if (
                                prior_recoveries <= 0
                                and order_price_now is not None
                                and order_price_now > 0
                            ):
                                sweep_anchor = float(order_price_now)
                            if sweep_anchor is None or sweep_anchor <= 0:
                                sweep_anchor = order_price_now
                            if sweep_anchor is None or sweep_anchor <= 0:
                                last_ref_retry = (
                                    last_retry
                                    if last_retry is not None
                                    else (bid_retry if bid_retry is not None else ask_retry)
                                )
                                tick_retry = _tick_size(trade.contract, ticker_retry, last_ref_retry)
                                sweep_anchor = _round_to_tick(
                                    _midpoint(bid_retry, ask_retry) or last_ref_retry,
                                    tick_retry,
                                )
                            delay_updates_base: dict[str, object] = {
                                "selected": selected_label,
                                "active": self._exec_mode_label("RELENTLESS_DELAY"),
                                "delay_recoveries": next_recoveries,
                                "delay_first_202_ts": float(first_202_ts),
                                "delay_last_202_ts": float(loop_now),
                                "delay_last_leg_sign": float(leg_sign),
                                "delay_last_leg_name": "FAV" if favorable_leg else "ADV",
                                "delay_locked_price_dir": None,
                            }
                            if anchor_hint is not None and anchor_hint > 0:
                                delay_updates_base["delay_anchor_price"] = float(anchor_hint)
                            if sweep_anchor is not None and sweep_anchor > 0:
                                delay_updates_base["delay_sweep_anchor_price"] = float(sweep_anchor)
                            if order_id or perm_id:
                                self._set_chase_state(
                                    order_id=order_id,
                                    perm_id=perm_id,
                                    updates=delay_updates_base,
                                )
                            retry_price = self._exec_price_for_mode(
                                "RELENTLESS_DELAY",
                                action,
                                bid=bid_retry,
                                ask=ask_retry,
                                last=last_retry,
                                ticker=ticker_retry,
                                elapsed_sec=float(loop_now - started),
                                quote_stale=self._quote_is_stale_for_relentless(
                                    ticker=ticker_retry,
                                    bid=bid_retry,
                                    ask=ask_retry,
                                    last=last_retry,
                                ),
                                open_shock=self._in_open_shock_window(),
                                no_progress_reprices=int(no_progress_reprices),
                                arrival_ref=arrival_ref,
                                delay_recoveries=next_recoveries,
                                delay_anchor_price=anchor_hint,
                                delay_sweep_anchor_price=sweep_anchor,
                            )
                            if retry_price is None:
                                retry_price = _safe_num(
                                    getattr(getattr(trade, "order", None), "lmtPrice", None)
                                )
                            try:
                                qty_retry = float(
                                    getattr(getattr(trade, "order", None), "totalQuantity", 0.0) or 0.0
                                )
                            except (TypeError, ValueError):
                                qty_retry = 0.0
                            if retry_price is not None and qty_retry > 0:
                                try:
                                    replacement = await self._client.place_limit_order(
                                        trade.contract,
                                        action,
                                        qty_retry,
                                        float(retry_price),
                                        str(getattr(trade.contract, "secType", "") or "").strip().upper() == "STK",
                                    )
                                    trade = replacement
                                    live_order_id, live_perm_id = self._trade_order_ids(trade)
                                    if live_order_id > 0:
                                        order_id = int(live_order_id)
                                    if live_perm_id > 0:
                                        perm_id = int(live_perm_id)
                                    order_ref = int(order_id or perm_id or 0)
                                    updates = dict(delay_updates_base)
                                    updates["target_price"] = float(retry_price)
                                    self._set_chase_state(
                                        order_id=order_id,
                                        perm_id=perm_id,
                                        updates=updates,
                                    )
                                    hint_text = (
                                        f" cap {float(anchor_hint):.2f}"
                                        if anchor_hint is not None and anchor_hint > 0
                                        else ""
                                    )
                                    leg_text = "FAV" if favorable_leg else "ADV"
                                    self._exec_status = (
                                        f"RLT⚔Delay sweep #{order_ref} @ {float(retry_price):.2f}"
                                        f" {leg_text} step {next_recoveries}/{sweep_span}{hint_text}"
                                    )
                                    self._set_orders_notice(
                                        self._exec_status,
                                        level="warn",
                                    )
                                    self._render_details_if_mounted(sample=False)
                                    await asyncio.sleep(float(self._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC))
                                    continue
                                except Exception as exc:
                                    self._set_orders_notice(
                                        f"RLT⚔Delay retry failed #{order_ref}: {exc}",
                                        level="warn",
                                    )
                                    self._render_details_if_mounted(sample=False)
                                    await asyncio.sleep(float(self._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC))
                                    continue
                            self._set_orders_notice(
                                f"RLT⚔Delay no retryable qty/price for #{order_ref}",
                                level="warn",
                            )
                            self._render_details_if_mounted(sample=False)
                            await asyncio.sleep(float(self._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC))
                            continue
                        error_prefix = f"IB {error_code}: " if error_code else "IB: "
                        self._set_orders_notice(
                            f"{order_label} {status_label}: {error_prefix}{error_message}",
                            level="error",
                        )
                    elif status_raw in ("Cancelled", "ApiCancelled", "Inactive"):
                        why_held = str(
                            getattr(getattr(trade, "orderStatus", None), "whyHeld", "") or ""
                        ).strip()
                        if why_held:
                            self._set_orders_notice(
                                f"{order_label} {status_raw}: {why_held}",
                                level="warn",
                            )
                        else:
                            self._set_orders_notice(
                                f"{order_label} {status_raw}",
                                level="warn",
                            )
                    elif status_effective == "Filled" or status_raw == "Filled":
                        self._set_orders_notice(f"Filled {order_label}", level="info")
                    elif is_done:
                        self._set_orders_notice(f"{order_label} {status_label}", level="warn")
                    self._clear_cancel_requested(order_id=order_id, perm_id=perm_id)
                    self._clear_chase_state(order_id=order_id, perm_id=perm_id)
                    self._render_details_if_mounted(sample=False)
                    return

                order_ref = int(order_id or perm_id or 0)
                if order_ref:
                    error_payload = self._consume_order_error(order_id, perm_id)
                    if error_payload is not None:
                        error_code, error_message = error_payload
                        if error_code in (110, 201, 202, 10147, 10148, 10149):
                            if selected_is_delay_mode and error_code == 202:
                                state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id) or {}
                                try:
                                    prior_recoveries = int(state.get("delay_recoveries") or 0)
                                except (TypeError, ValueError):
                                    prior_recoveries = 0
                                first_202_ts = _safe_float(state.get("delay_first_202_ts"))
                                recover_window_sec = max(
                                    1.0,
                                    float(self._RELENTLESS_DELAY_RECOVER_WINDOW_SEC),
                                )
                                if (
                                    first_202_ts is None
                                    or first_202_ts <= 0
                                    or (float(loop_now) - float(first_202_ts)) > recover_window_sec
                                ):
                                    first_202_ts = float(loop_now)
                                sweep_span = self._relentless_delay_sweep_span()
                                next_recoveries = self._relentless_delay_next_step(
                                    prior_recoveries=prior_recoveries
                                )
                                favorable_leg, leg_sign = self._relentless_delay_leg(
                                    action,
                                    next_recoveries,
                                )
                                anchor_hint = (
                                    self._cap_price_hint_from_trade(trade)
                                    or self._price_hint_from_error_message(error_message)
                                    or _safe_float(state.get("delay_anchor_price"))
                                )
                                order_price_now = _safe_num(
                                    getattr(getattr(trade, "order", None), "lmtPrice", None)
                                )
                                sweep_anchor = _safe_float(state.get("delay_sweep_anchor_price"))
                                if (
                                    prior_recoveries <= 0
                                    and order_price_now is not None
                                    and order_price_now > 0
                                ):
                                    sweep_anchor = float(order_price_now)
                                if sweep_anchor is None or sweep_anchor <= 0:
                                    sweep_anchor = order_price_now
                                updates: dict[str, object] = {
                                    "selected": selected_label,
                                    "active": self._exec_mode_label("RELENTLESS_DELAY"),
                                    "delay_recoveries": next_recoveries,
                                    "delay_first_202_ts": float(first_202_ts),
                                    "delay_last_202_ts": float(loop_now),
                                    "delay_last_leg_sign": float(leg_sign),
                                    "delay_last_leg_name": "FAV" if favorable_leg else "ADV",
                                    "delay_locked_price_dir": None,
                                }
                                if anchor_hint is not None and anchor_hint > 0:
                                    updates["delay_anchor_price"] = float(anchor_hint)
                                if sweep_anchor is not None and sweep_anchor > 0:
                                    updates["delay_sweep_anchor_price"] = float(sweep_anchor)
                                self._set_chase_state(
                                    order_id=order_id,
                                    perm_id=perm_id,
                                    updates=updates,
                                )
                                hint_text = (
                                    f" cap {float(anchor_hint):.2f}"
                                    if anchor_hint is not None and anchor_hint > 0
                                    else ""
                                )
                                leg_text = "FAV" if favorable_leg else "ADV"
                                self._exec_status = (
                                    f"RLT⚔Delay sweep #{order_ref} "
                                    f"{leg_text} step {next_recoveries}/{sweep_span}{hint_text}"
                                )
                                self._set_orders_notice(self._exec_status, level="warn")
                                last_modify_error_ts = loop_now
                                self._render_details_if_mounted(sample=False)
                                await asyncio.sleep(float(self._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC))
                                continue
                            order_label = f"#{order_ref}"
                            status_label = status_raw or "Pending"
                            error_prefix = f"IB {error_code}: " if error_code else "IB: "
                            level = "warn" if error_code in (10147, 10148, 10149) else "error"
                            self._exec_status = (
                                f"Chase halted {order_label} {status_label}: {error_prefix}{error_message}"
                            )
                            self._set_orders_notice(
                                f"{order_label} {status_label}: {error_prefix}{error_message}",
                                level=level,
                            )
                            self._clear_chase_state(order_id=order_id, perm_id=perm_id)
                            self._render_details_if_mounted(sample=False)
                            return

                elapsed = loop_now - started
                mode_now = _exec_chase_mode(elapsed, selected_mode=mode)
                if mode_now is None:
                    try:
                        self._mark_cancel_requested(order_id=order_id, perm_id=perm_id)
                        await self._client.cancel_trade(trade)
                        live_order_id, live_perm_id = self._trade_order_ids(trade)
                        if live_order_id > 0:
                            order_id = int(live_order_id)
                        if live_perm_id > 0:
                            perm_id = int(live_perm_id)
                        order_ref = int(order_id or perm_id or 0)
                        timeout_sec = (
                            float(_EXEC_RELENTLESS_TIMEOUT_SEC)
                            if str(mode or "").strip().upper() in ("RELENTLESS", "RELENTLESS_DELAY")
                            else float(_EXEC_LADDER_TIMEOUT_SEC)
                        )
                        self._exec_status = (
                            f"Timeout cancel sent #{order_ref} (> {timeout_sec:.0f}s)"
                        )
                        self._set_orders_notice(self._exec_status, level="warn")
                        error_payload = (
                            await self._await_order_error(
                                int(order_id),
                                int(perm_id),
                                attempts=3,
                                interval_sec=0.1,
                            )
                            if order_ref
                            else None
                        )
                        if error_payload is not None:
                            error_code, error_message = error_payload
                            error_prefix = f"IB {error_code}: " if error_code else "IB: "
                            level = "warn" if error_code in (10147, 10148, 10149) else "error"
                            self._exec_status = (
                                f"Timeout cancel #{order_ref}: {error_prefix}{error_message}"
                            )
                            self._set_orders_notice(self._exec_status, level=level)
                    except Exception as exc:
                        self._exec_status = f"Timeout cancel error: {exc}"
                        self._set_orders_notice(self._exec_status, level="error")
                    self._clear_chase_state(order_id=order_id, perm_id=perm_id)
                    self._render_details_if_mounted(sample=False)
                    return

                ticker = self._client.ticker_for_con_id(con_id) if con_id else None
                bid = self._quote_num(getattr(ticker, "bid", None)) if ticker else None
                ask = self._quote_num(getattr(ticker, "ask", None)) if ticker else None
                last = self._quote_num(getattr(ticker, "last", None)) if ticker else None
                if arrival_ref is None:
                    arrival_ref = _midpoint(bid, ask) or last
                mode_now_clean = str(mode_now or "").strip().upper()
                is_relentless = mode_now_clean in ("RELENTLESS", "RELENTLESS_DELAY")
                is_relentless_delay = mode_now_clean == "RELENTLESS_DELAY"
                quote_stale = self._quote_is_stale_for_relentless(
                    ticker=ticker,
                    bid=bid,
                    ask=ask,
                    last=last,
                ) if is_relentless else False
                open_shock = self._in_open_shock_window() if is_relentless else False
                spread_now = (
                    float(ask) - float(bid)
                    if (is_relentless and bid is not None and ask is not None and ask >= bid)
                    else None
                )
                last_ref_now = last if last is not None else (bid if bid is not None else ask)
                tick_now = _tick_size(trade.contract, ticker, last_ref_now) if is_relentless else 0.0
                spread_pressure = (
                    self._relentless_spread_pressure(spread=spread_now, tick=tick_now)
                    if is_relentless
                    else 1.0
                )
                if is_relentless and quote_stale:
                    if last_live_probe_ts is None or (loop_now - last_live_probe_ts) >= 4.0:
                        try:
                            await self._client.refresh_live_snapshot_once(trade.contract)
                        except Exception:
                            pass
                        last_live_probe_ts = loop_now
                quote_sig = _exec_chase_quote_signature(bid, ask, last)
                min_interval_sec = (
                    self._relentless_min_reprice_sec(
                        quote_stale=bool(quote_stale),
                        open_shock=bool(open_shock),
                        no_progress_reprices=int(no_progress_reprices),
                        spread_pressure=float(spread_pressure),
                    )
                    if is_relentless
                    else 5.0
                )
                should_reprice = _exec_chase_should_reprice(
                    now_sec=loop_now,
                    last_reprice_sec=last_reprice_ts,
                    mode_now=str(mode_now),
                    prev_mode=prev_mode,
                    quote_signature=quote_sig,
                    prev_quote_signature=prev_quote_sig,
                    min_interval_sec=float(min_interval_sec),
                )
                working_status = status_effective or status_raw
                if cancel_requested:
                    should_reprice = False
                if should_reprice and working_status not in repricable_statuses:
                    should_reprice = False
                if (
                    should_reprice
                    and last_modify_error_ts is not None
                    and (loop_now - last_modify_error_ts) < float(self._CHASE_MODIFY_ERROR_BACKOFF_SEC)
                ):
                    should_reprice = False
                prev_mode = str(mode_now)
                prev_quote_sig = quote_sig
                if order_id or perm_id:
                    state_updates: dict[str, object] = {
                        "selected": selected_label,
                        "active": self._exec_mode_label(str(mode_now)),
                    }
                    if status_effective and status_effective != status_raw:
                        state_updates["effective_status"] = status_effective
                    self._set_chase_state(
                        order_id=order_id,
                        perm_id=perm_id,
                        updates=state_updates,
                    )

                if should_reprice:
                    chase_state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id) or {}
                    try:
                        delay_recoveries = int(chase_state.get("delay_recoveries") or 0)
                    except (TypeError, ValueError):
                        delay_recoveries = 0
                    delay_anchor_price = _safe_float(chase_state.get("delay_anchor_price"))
                    delay_sweep_anchor_price = _safe_float(chase_state.get("delay_sweep_anchor_price"))
                    delay_last_202_ts = _safe_float(chase_state.get("delay_last_202_ts"))
                    delay_locked_price_dir = _safe_float(chase_state.get("delay_locked_price_dir"))
                    if (
                        is_relentless_delay
                        and delay_recoveries > 0
                        and delay_last_202_ts is not None
                        and (float(loop_now) - float(delay_last_202_ts))
                        >= max(
                            float(self._RELENTLESS_DELAY_RECOVER_SETTLE_SEC),
                            float(self._RELENTLESS_DELAY_RECOVER_COOLDOWN_SEC),
                        )
                    ):
                        delay_last_leg_sign = _safe_float(chase_state.get("delay_last_leg_sign"))
                        lock_dir = (
                            (1.0 if float(delay_last_leg_sign) >= 0 else -1.0)
                            if delay_last_leg_sign is not None
                            else None
                        )
                        delay_recoveries = 0
                        delay_anchor_price = None
                        delay_sweep_anchor_price = None
                        delay_locked_price_dir = lock_dir
                        if order_id or perm_id:
                            updates: dict[str, object] = {
                                "delay_recoveries": 0,
                                "delay_anchor_price": None,
                                "delay_sweep_anchor_price": None,
                                "delay_first_202_ts": None,
                                "delay_last_202_ts": None,
                                "delay_last_leg_sign": None,
                                "delay_last_leg_name": None,
                            }
                            if lock_dir is None:
                                updates["delay_locked_price_dir"] = None
                            else:
                                updates["delay_locked_price_dir"] = float(lock_dir)
                            self._set_chase_state(
                                order_id=order_id,
                                perm_id=perm_id,
                                updates=updates,
                            )
                        if lock_dir is not None:
                            leg_text = "up" if float(lock_dir) > 0 else "down"
                            self._set_orders_notice(
                                f"RLT⚔Delay lock engaged ({leg_text} side) after 202 settle",
                                level="info",
                            )
                    price = self._exec_price_for_mode(
                        str(mode_now),
                        action,
                        bid=bid,
                        ask=ask,
                        last=last,
                        ticker=ticker,
                        elapsed_sec=float(elapsed),
                        quote_stale=bool(quote_stale),
                        open_shock=bool(open_shock),
                        no_progress_reprices=int(no_progress_reprices),
                        arrival_ref=arrival_ref,
                        delay_recoveries=int(delay_recoveries) if is_relentless_delay else 0,
                        delay_anchor_price=delay_anchor_price if is_relentless_delay else None,
                        delay_sweep_anchor_price=(
                            delay_sweep_anchor_price if is_relentless_delay else None
                        ),
                        delay_locked_price_dir=(
                            delay_locked_price_dir if is_relentless_delay else None
                        ),
                    )
                    if (order_id or perm_id) and price is not None:
                        self._set_chase_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            updates={
                                "selected": selected_label,
                                "active": self._exec_mode_label(str(mode_now)),
                                "target_price": float(price),
                            },
                        )
                else:
                    price = None
                if price is not None:
                    current_price = _safe_num(getattr(getattr(trade, "order", None), "lmtPrice", None))
                    compare_tick = _tick_size(trade.contract, ticker, price) or 0.01
                    if (
                        current_price is not None
                        and abs(float(price) - float(current_price))
                        <= max(float(compare_tick) * 0.5, 1e-9)
                    ):
                        last_reprice_ts = loop_now
                        price = None
                if price is not None:
                    try:
                        trade = await self._client.modify_limit_order(trade, float(price))
                        applied_price = _safe_num(getattr(getattr(trade, "order", None), "lmtPrice", None))
                        if applied_price is None or applied_price <= 0:
                            applied_price = float(price)
                        live_order_id, live_perm_id = self._trade_order_ids(trade)
                        if live_order_id > 0:
                            order_id = int(live_order_id)
                        if live_perm_id > 0:
                            perm_id = int(live_perm_id)
                        order_ref = int(order_id or perm_id or 0)
                        mode_label = self._exec_mode_label(str(mode_now))
                        mods = 1
                        if order_ref:
                            state = self._chase_state_for_ids(order_id=order_id, perm_id=perm_id) or {}
                            try:
                                mods = int(state.get("mods") or 0) + 1
                            except (TypeError, ValueError):
                                mods = 1
                            self._set_chase_state(
                                order_id=order_id,
                                perm_id=perm_id,
                                updates={
                                    "selected": selected_label,
                                    "active": mode_label,
                                    "target_price": float(applied_price),
                                    "mods": mods,
                                },
                            )
                        mode_view = f"{selected_label}->{mode_label}" if selected_label == "AUTO" else mode_label
                        order_label = f"#{order_ref}" if order_ref else "order"
                        self._exec_status = (
                            f"Chasing {order_label} [{mode_view}] @ {float(applied_price):.2f} mod#{mods}"
                        )
                        last_reprice_ts = loop_now
                        last_modify_error_ts = None
                        if not fill_progress:
                            no_progress_reprices += 1
                    except Exception as exc:
                        last_modify_error_ts = loop_now
                        self._exec_status = f"Chase error: {exc}"
                        self._set_orders_notice(self._exec_status, level="error")
                    self._render_details_if_mounted(sample=False)

                await asyncio.sleep(0.25)
        finally:
            current_task = asyncio.current_task()
            if current_task is not None:
                self._unregister_chase_task(current_task)
            self._clear_chase_state(order_id=order_id, perm_id=perm_id)
            if con_id:
                try:
                    self._client.release_ticker(con_id, owner=chase_owner)
                except Exception:
                    pass
