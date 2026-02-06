"""Bot hub screen (presets + orders + auto-trade)."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import Contract, PortfolioItem, Stock
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from ..engine import (
    flip_exit_hit,
    normalize_spot_entry_signal,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
)
from ..signals import (
    direction_from_action_right,
    ema_state_direction,
    flip_exit_mode,
    parse_bar_size,
)
from ..utils.date_utils import business_days_until
from .common import (
    _SECTION_TYPES,
    _fmt_quote,
    _limit_price_for_mode,
    _market_session_label,
    _midpoint,
    _optimistic_price,
    _aggressive_price,
    _parse_float,
    _parse_int,
    _portfolio_sort_key,
    _pnl_text,
    _quote_status_line,
    _round_to_tick,
    _safe_num,
    _tick_size,
    _ticker_price,
    _trade_sort_key,
)
from .bot_journal import BotJournal
from .bot_models import (
    _BotConfigField,
    _BotConfigResult,
    _BotInstance,
    _BotOrder,
    _BotPreset,
    _PresetHeader,
    _SignalSnapshot,
)
from .bot_engine_runtime import BotEngineRuntimeMixin
from .bot_order_builder import BotOrderBuilderMixin
from .bot_signal_runtime import BotSignalRuntimeMixin

# region Bot UI
class BotConfigScreen(Screen[_BotConfigResult | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("q", "cancel", "Cancel"),
        ("enter", "save", "Save"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("left", "cycle_left", "Prev"),
        ("right", "cycle_right", "Next"),
        ("space", "toggle_bool", "Toggle"),
        ("e", "toggle_edit", "Edit"),
    ]

    def __init__(
        self,
        *,
        mode: str,
        instance_id: int | None,
        group: str,
        symbol: str,
        strategy: dict,
        filters: dict | None,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._instance_id = instance_id
        self._group = group
        self._symbol = symbol
        self._strategy = copy.deepcopy(strategy)
        self._filters = copy.deepcopy(filters) if filters else None

        self._fields: list[_BotConfigField] = []
        self._editing: _BotConfigField | None = None
        self._edit_buffer = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="bot-config-header")
        yield DataTable(id="bot-config", zebra_stripes=True)
        yield Static(
            "Enter=Save & start  Esc=Cancel  Type=Edit  e=Edit w/ current  Space=Toggle  ←/→=Cycle enum",
            id="bot-config-help",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._header = self.query_one("#bot-config-header", Static)
        self._table = self.query_one("#bot-config", DataTable)
        self._table.cursor_type = "row"
        self._table.add_columns("Field", "Value")
        self._build_fields()
        self._refresh_table()
        self._table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; treat Enter as Save.
        self.action_save()

    def _build_fields(self) -> None:
        instrument_raw = self._strategy.get("instrument", "options")
        instrument = "spot" if str(instrument_raw or "").strip().lower() == "spot" else "options"

        self._strategy.setdefault("entry_signal", "ema")
        self._strategy.setdefault("entry_confirm_bars", 0)
        self._strategy.setdefault("orb_window_mins", 15)
        self._strategy.setdefault("orb_risk_reward", 2.0)
        self._strategy.setdefault("orb_target_mode", "rr")
        self._strategy.setdefault("orb_open_time_et", "")
        self._strategy.setdefault("spot_exit_mode", "pct")
        self._strategy.setdefault("spot_atr_period", 14)
        self._strategy.setdefault("spot_pt_atr_mult", 1.5)
        self._strategy.setdefault("spot_sl_atr_mult", 1.0)
        self._strategy.setdefault("spot_exit_time_et", "")
        self._strategy.setdefault("regime_ema_preset", "")
        self._strategy.setdefault("regime_bar_size", "")
        self._strategy.setdefault("regime_mode", "ema")
        self._strategy.setdefault("regime2_mode", "off")
        self._strategy.setdefault("regime2_ema_preset", "")
        self._strategy.setdefault("regime2_bar_size", "")
        self._strategy.setdefault("regime2_supertrend_atr_period", 10)
        self._strategy.setdefault("regime2_supertrend_multiplier", 3.0)
        self._strategy.setdefault("regime2_supertrend_source", "hl2")
        self._strategy.setdefault("supertrend_atr_period", 10)
        self._strategy.setdefault("supertrend_multiplier", 3.0)
        self._strategy.setdefault("supertrend_source", "hl2")

        if instrument == "spot":
            sym = str(self._symbol or self._strategy.get("symbol") or "").strip().upper()
            default_sec_type = (
                "FUT" if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"} else "STK"
            )
            self._strategy.setdefault("spot_sec_type", default_sec_type)
            self._strategy.setdefault("spot_exchange", "")
            self._strategy.setdefault("spot_close_eod", False)
            if not isinstance(self._strategy.get("directional_spot"), dict):
                self._strategy["directional_spot"] = {
                    "up": {"action": "BUY", "qty": 1},
                    "down": {"action": "SELL", "qty": 1},
                }

        self._fields = [
            _BotConfigField("Symbol", "text", "symbol"),
            _BotConfigField("Instrument", "enum", "instrument", options=("options", "spot")),
            _BotConfigField(
                "Signal bar size",
                "enum",
                "signal_bar_size",
                options=("1 min", "2 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour", "4 hours", "1 day"),
            ),
            _BotConfigField("Signal use RTH", "bool", "signal_use_rth"),
            _BotConfigField("Entry days", "text", "entry_days"),
            _BotConfigField("Max entries/day", "int", "max_entries_per_day"),
            _BotConfigField("Entry signal", "enum", "entry_signal", options=("ema", "orb")),
            _BotConfigField("EMA preset", "text", "ema_preset"),
            _BotConfigField("EMA entry mode", "enum", "ema_entry_mode", options=("trend", "cross")),
            _BotConfigField("Entry confirm bars", "int", "entry_confirm_bars"),
            _BotConfigField("ORB window mins", "int", "orb_window_mins"),
            _BotConfigField("ORB risk reward", "float", "orb_risk_reward"),
            _BotConfigField("ORB target mode", "enum", "orb_target_mode", options=("rr", "or_range")),
            _BotConfigField("ORB open time (ET)", "text", "orb_open_time_et"),
            _BotConfigField("Regime mode", "enum", "regime_mode", options=("ema", "supertrend")),
            _BotConfigField("Regime EMA preset", "text", "regime_ema_preset"),
            _BotConfigField(
                "Regime bar size",
                "enum",
                "regime_bar_size",
                options=("", "1 min", "2 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour", "4 hours", "1 day"),
            ),
            _BotConfigField("Supertrend ATR period", "int", "supertrend_atr_period"),
            _BotConfigField("Supertrend multiplier", "float", "supertrend_multiplier"),
            _BotConfigField(
                "Supertrend source",
                "enum",
                "supertrend_source",
                options=("hl2", "close"),
            ),
            _BotConfigField(
                "Regime2 mode",
                "enum",
                "regime2_mode",
                options=("off", "ema", "supertrend"),
            ),
            _BotConfigField(
                "Regime2 apply to",
                "enum",
                "regime2_apply_to",
                options=("both", "longs", "shorts"),
            ),
            _BotConfigField("Regime2 EMA preset", "text", "regime2_ema_preset"),
            _BotConfigField(
                "Regime2 bar size",
                "enum",
                "regime2_bar_size",
                options=("", "1 min", "2 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour", "4 hours", "1 day"),
            ),
            _BotConfigField("Regime2 Supertrend ATR period", "int", "regime2_supertrend_atr_period"),
            _BotConfigField("Regime2 Supertrend multiplier", "float", "regime2_supertrend_multiplier"),
            _BotConfigField(
                "Regime2 Supertrend source",
                "enum",
                "regime2_supertrend_source",
                options=("hl2", "close"),
            ),
            _BotConfigField("Entry start hour (ET)", "text", "filters.entry_start_hour_et"),
            _BotConfigField("Entry end hour (ET)", "text", "filters.entry_end_hour_et"),
            _BotConfigField("Entry start hour", "text", "filters.entry_start_hour"),
            _BotConfigField("Entry end hour", "text", "filters.entry_end_hour"),
            _BotConfigField("Volume ratio min", "float", "filters.volume_ratio_min"),
            _BotConfigField("Volume EMA period", "int", "filters.volume_ema_period"),
            _BotConfigField("Flip-exit", "bool", "exit_on_signal_flip"),
            _BotConfigField("Flip min hold bars", "int", "flip_exit_min_hold_bars"),
            _BotConfigField("Flip only if profit", "bool", "flip_exit_only_if_profit"),
        ]

        if instrument == "spot":
            self._fields.extend(
                [
                    _BotConfigField("Spot secType", "enum", "spot_sec_type", options=("STK", "FUT")),
                    _BotConfigField("Spot exchange", "text", "spot_exchange"),
                    _BotConfigField("Spot close EOD", "bool", "spot_close_eod"),
                    _BotConfigField("Spot exit time (ET)", "text", "spot_exit_time_et"),
                    _BotConfigField("Spot exit mode", "enum", "spot_exit_mode", options=("pct", "atr")),
                    _BotConfigField("Spot ATR period", "int", "spot_atr_period"),
                    _BotConfigField("Spot PT ATR mult", "float", "spot_pt_atr_mult"),
                    _BotConfigField("Spot SL ATR mult", "float", "spot_sl_atr_mult"),
                    _BotConfigField("Spot PT %", "float", "spot_profit_target_pct"),
                    _BotConfigField("Spot SL %", "float", "spot_stop_loss_pct"),
                    _BotConfigField("Spot up action", "enum", "directional_spot.up.action", options=("", "BUY")),
                    _BotConfigField("Spot up qty", "int", "directional_spot.up.qty"),
                    _BotConfigField("Spot down action", "enum", "directional_spot.down.action", options=("", "SELL")),
                    _BotConfigField("Spot down qty", "int", "directional_spot.down.qty"),
                ]
            )
            return

        self._fields.extend(
            [
                _BotConfigField("DTE", "int", "dte"),
                _BotConfigField("Profit target (PT)", "float", "profit_target"),
                _BotConfigField("Stop loss (SL)", "float", "stop_loss"),
                _BotConfigField("Exit DTE", "int", "exit_dte"),
                _BotConfigField("Stop loss basis", "enum", "stop_loss_basis", options=("max_loss", "credit")),
            ]
        )

        legs = self._strategy.get("legs", [])
        if not isinstance(legs, list):
            legs = []
            self._strategy["legs"] = legs
        for idx in range(len(legs)):
            prefix = f"legs.{idx}."
            self._fields.extend(
                [
                    _BotConfigField(f"Leg {idx+1} action", "enum", prefix + "action", options=("BUY", "SELL")),
                    _BotConfigField(f"Leg {idx+1} right", "enum", prefix + "right", options=("CALL", "PUT")),
                    _BotConfigField(f"Leg {idx+1} moneyness %", "float", prefix + "moneyness_pct"),
                    _BotConfigField(f"Leg {idx+1} delta", "float", prefix + "delta"),
                    _BotConfigField(f"Leg {idx+1} qty", "int", prefix + "qty"),
                ]
            )

    def _refresh_table(self) -> None:
        self._table.clear()
        mode = "Create" if self._mode == "create" else "Update"
        legs_desc = _legs_label(self._strategy.get("legs", []))
        dir_hint = _legs_direction_hint(
            self._strategy.get("legs", []),
            bool(self._strategy.get("ema_directional")),
        )
        lines = [
            Text(f"{mode}: {self._group}  ({self._symbol})", style="bold"),
            Text(f"Legs: {legs_desc}   Dir: {dir_hint}", style="dim"),
        ]
        self._header.update(Text("\n").join(lines))
        for field in self._fields:
            if self._editing and self._editing.path == field.path:
                self._table.add_row(field.label, self._edit_buffer)
                continue
            value = self._get_value(field)
            self._table.add_row(field.label, self._format_value(field, value))

    def _format_value(self, field: _BotConfigField, value: object) -> str:
        if field.kind == "bool":
            return "ON" if bool(value) else "OFF"
        if value is None:
            return ""
        if field.kind == "text" and isinstance(value, list):
            return ",".join(str(v) for v in value if v is not None)
        if field.kind == "float":
            try:
                return f"{float(value):.4g}"
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def _selected_field(self) -> _BotConfigField | None:
        row = self._table.cursor_coordinate.row
        if row < 0 or row >= len(self._fields):
            return None
        return self._fields[row]

    def _get_value(self, field: _BotConfigField) -> object:
        if field.path == "symbol":
            return self._symbol
        if field.path.startswith("filters."):
            if not self._filters:
                return None
            return _get_path(self._filters, field.path[len("filters.") :])
        return _get_path(self._strategy, field.path)

    def _set_value(self, field: _BotConfigField, value: object) -> None:
        if field.path == "symbol":
            self._symbol = str(value or "").strip().upper()
            return
        if field.path.startswith("filters."):
            if self._filters is None:
                self._filters = {}
            _set_path(self._filters, field.path[len("filters.") :], value)
            return
        _set_path(self._strategy, field.path, value)

    def action_cursor_down(self) -> None:
        if self._editing is not None:
            self._editing = None
            self._edit_buffer = ""
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        if self._editing is not None:
            self._editing = None
            self._edit_buffer = ""
        self._table.action_cursor_up()

    def action_toggle_edit(self) -> None:
        field = self._selected_field()
        if not field or field.kind not in ("int", "float", "text"):
            return
        if self._editing and self._editing.path == field.path:
            self._editing = None
            self._edit_buffer = ""
        else:
            self._editing = field
            self._edit_buffer = self._format_value(field, self._get_value(field))
        self._refresh_table()

    def action_toggle_bool(self) -> None:
        field = self._selected_field()
        if not field or field.kind != "bool":
            return
        current = bool(self._get_value(field))
        self._set_value(field, not current)
        self._refresh_table()

    def action_cycle_left(self) -> None:
        self._cycle_enum(-1)

    def action_cycle_right(self) -> None:
        self._cycle_enum(1)

    def _cycle_enum(self, direction: int) -> None:
        field = self._selected_field()
        if not field or field.kind != "enum" or not field.options:
            return
        current = self._get_value(field)
        current_str = "" if current is None else str(current)
        options = list(field.options)
        if current_str not in options:
            idx = 0
        else:
            idx = options.index(current_str)
        new_val = options[(idx + direction) % len(options)]
        if field.path == "ema_preset" and new_val == "":
            self._set_value(field, None)
        else:
            self._set_value(field, new_val)
        if field.path == "instrument":
            self._build_fields()
        self._refresh_table()

    def on_key(self, event: events.Key) -> None:
        if not self._editing:
            field = self._selected_field()
            if not field or not event.character:
                return
            char = event.character
            if field.kind == "int" and char.isdigit():
                self._editing = field
                self._edit_buffer = ""
            elif field.kind == "float" and (char in "0123456789." or char == "-"):
                self._editing = field
                self._edit_buffer = ""
            elif field.kind == "text":
                if event.key in (
                    "j",
                    "k",
                    "up",
                    "down",
                    "left",
                    "right",
                    "enter",
                    "escape",
                    "q",
                    "e",
                    "space",
                ):
                    return
                self._editing = field
                self._edit_buffer = ""
            else:
                return
        if event.key == "backspace":
            self._edit_buffer = self._edit_buffer[:-1]
            self._apply_edit_buffer()
            event.stop()
            return
        if not event.character:
            return
        char = event.character
        if self._editing.kind == "int":
            if char not in "0123456789":
                return
            self._edit_buffer = _append_digit(self._edit_buffer, char, allow_decimal=False)
            self._apply_edit_buffer()
            event.stop()
            return
        if self._editing.kind == "float":
            if char == "-" and not self._edit_buffer.startswith("-"):
                self._edit_buffer = "-" + self._edit_buffer
            else:
                if char not in "0123456789.":
                    return
                self._edit_buffer = _append_digit(self._edit_buffer, char, allow_decimal=True)
            self._apply_edit_buffer()
            event.stop()
            return
        if self._editing.kind == "text":
            if char:
                self._edit_buffer += char
                self._apply_edit_buffer()
                event.stop()

    def _apply_edit_buffer(self) -> None:
        field = self._editing
        if not field:
            return
        if field.kind == "text":
            if field.path == "entry_days":
                self._set_value(field, _parse_entry_days(self._edit_buffer))
                self._refresh_table()
                return
            if field.path.startswith("filters."):
                cleaned = self._edit_buffer.strip()
                value = None
                if cleaned.isdigit():
                    parsed = int(cleaned)
                    value = max(0, min(parsed, 23))
                self._set_value(field, value)
                self._refresh_table()
                return
            self._set_value(field, self._edit_buffer.strip())
            self._refresh_table()
            return
        if field.kind == "int":
            if not self._edit_buffer.isdigit():
                return
            parsed = int(self._edit_buffer)
            if field.path.endswith(".qty"):
                parsed = max(1, parsed)
            else:
                parsed = max(0, parsed)
            self._set_value(field, parsed)
            self._refresh_table()
            return
        if not self._edit_buffer.strip():
            self._set_value(field, None)
            self._refresh_table()
            return
        parsed = _parse_float(self._edit_buffer)
        if parsed is None:
            return
        self._set_value(field, float(parsed))
        self._refresh_table()

    def action_save(self) -> None:
        result = _BotConfigResult(
            mode=self._mode,
            instance_id=self._instance_id,
            group=self._group,
            symbol=self._symbol,
            strategy=self._strategy,
            filters=self._filters,
        )
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BotScreen(BotOrderBuilderMixin, BotSignalRuntimeMixin, BotEngineRuntimeMixin, Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("ctrl+t", "app.pop_screen", "Back"),
        ("ctrl+a", "toggle_presets", "Presets"),
        ("v", "toggle_scope", "Scope"),
        ("tab", "cycle_focus", "Focus"),
        ("h", "focus_prev", "Prev"),
        ("l", "focus_next", "Next"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("enter", "activate", "Select"),
        ("space", "context_space", "Run/Toggle"),
        ("s", "stop_bot", "Stop"),
        ("d", "delete_instance", "Del"),
        ("c", "cancel_order", "Cancel"),
        ("f", "cycle_dte_filter", "Filter"),
        ("w", "cycle_win_filter", "Win"),
        ("r", "reload", "Reload"),
    ]
    _PANEL_BY_TABLE_ID = {
        "bot-presets": "presets",
        "bot-instances": "instances",
        "bot-orders": "orders",
        "bot-logs": "logs",
    }
    _PANEL_ORDER = ("presets", "instances", "orders", "logs")
    _ACTIVATE_HANDLER_BY_PANEL = {
        "presets": "_activate_presets_panel",
        "instances": "_activate_instances_panel",
        "orders": "_submit_selected_order",
        "logs": "_activate_logs_panel",
    }
    _SPACE_HANDLER_BY_PANEL = {
        "presets": "action_activate",
        "instances": "action_toggle_instance",
        "orders": "action_toggle_instance",
        "logs": "action_toggle_instance",
    }

    def __init__(self, client: IBKRClient, refresh_sec: float) -> None:
        super().__init__()
        self._client = client
        self._refresh_sec = max(refresh_sec, 0.25)
        base = Path(__file__).resolve().parents[1]
        self._leaderboard_path = base / "backtest" / "leaderboard.json"
        self._spot_milestones_path = base / "backtest" / "spot_milestones.json"
        self._spot_champions_path = base / "backtest" / "spot_champions.json"
        self._backtest_readme_path = base / "backtest" / "README.md"
        self._group_eval_by_name: dict[str, dict] = {}
        self._payload: dict | None = None
        self._presets: list[_BotPreset] = []
        self._preset_rows: list[_BotPreset | _PresetHeader] = []
        self._preset_expanded: set[str] = set()
        self._preset_expand_initialized = False
        self._preset_known_contracts: set[str] = set()
        self._spot_champ_version: str | None = None
        self._presets_visible = True
        self._filter_dte: int | None = None
        self._filter_min_win_rate: float | None = None
        self._instances: list[_BotInstance] = []
        self._instance_rows: list[_BotInstance] = []
        self._next_instance_id = 1
        self._orders: list[_BotOrder] = []
        self._order_rows: list[_BotOrder] = []
        self._positions: list[PortfolioItem] = []
        self._log_events: list[dict] = []
        self._log_rows: list[dict] = []
        self._status: str | None = None
        self._refresh_task = None
        self._order_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._cancel_task: asyncio.Task | None = None
        self._tracked_conids: set[int] = set()
        self._active_panel = "presets"
        self._refresh_lock = asyncio.Lock()
        self._scope_all = False
        self._last_chase_ts = 0.0
        self._journal = BotJournal(Path(__file__).resolve().parent / "out")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="bot-status"),
            DataTable(id="bot-presets", zebra_stripes=True),
            DataTable(id="bot-instances", zebra_stripes=True),
            DataTable(id="bot-orders", zebra_stripes=True),
            DataTable(id="bot-logs", zebra_stripes=True),
            id="bot-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._presets_table = self.query_one("#bot-presets", DataTable)
        self._status_panel = self.query_one("#bot-status", Static)
        self._orders_table = self.query_one("#bot-orders", DataTable)
        self._instances_table = self.query_one("#bot-instances", DataTable)
        self._logs_table = self.query_one("#bot-logs", DataTable)
        self._presets_table.border_title = "Presets"
        self._instances_table.border_title = "Instances"
        self._orders_table.border_title = "Orders / Positions"
        self._logs_table.border_title = "Logs"
        self._presets_table.cursor_type = "row"
        self._orders_table.cursor_type = "row"
        self._instances_table.cursor_type = "row"
        self._logs_table.cursor_type = "row"
        self._setup_tables()
        self._load_leaderboard()
        self._presets_table.display = self._presets_visible
        await self._refresh_positions()
        self._refresh_instances_table()
        self._refresh_orders_table()
        self._refresh_logs_table()
        self._journal_write(event="BOOT", reason=None, data={"refresh_sec": self._refresh_sec})
        self._render_status()
        self._focus_panel("presets")
        self._refresh_task = self.set_interval(self._refresh_sec, self._on_refresh_tick)

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        for con_id in list(self._tracked_conids):
            self._client.release_ticker(con_id, owner="bot")
        self._tracked_conids.clear()
        self._journal_write(event="SHUTDOWN", reason=None, data=None)

    def _journal_write(
        self,
        *,
        event: str,
        instance: _BotInstance | None = None,
        order: _BotOrder | None = None,
        reason: str | None,
        data: dict | None,
    ) -> None:
        try:
            entry = self._journal.write(
                event=event,
                instance=instance,
                order=order,
                reason=reason,
                data=data,
                strategy_instrument=self._strategy_instrument,
            )
        except Exception:
            return
        self._log_events.append(entry)
        if len(self._log_events) > 500:
            self._log_events = self._log_events[-500:]
        if hasattr(self, "_logs_table"):
            self._refresh_logs_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; hook it so Enter arms/sends.
        table_id = str(getattr(event.control, "id", "") or "")
        panel = self._PANEL_BY_TABLE_ID.get(table_id)
        if panel:
            self._active_panel = str(panel)
        self.action_activate()

    def on_key(self, event: events.Key) -> None:
        if event.character == "X":
            self._submit_selected_order()
            event.stop()
            return
        if event.character == "S":
            self._kill_all()
            event.stop()
            return

    def action_cursor_down(self) -> None:
        self._cursor_move(1)

    def action_cursor_up(self) -> None:
        self._cursor_move(-1)

    def action_reload(self) -> None:
        self._load_leaderboard()
        self._set_status("Reloaded leaderboard")
        self._journal_write(event="RELOAD_LEADERBOARD", reason=None, data=None)

    def action_toggle_presets(self) -> None:
        self._presets_visible = not self._presets_visible
        self._presets_table.display = self._presets_visible
        if not self._presets_visible and self._active_panel == "presets":
            self._focus_panel("instances")
        elif self._presets_visible and self._active_panel != "presets":
            self._focus_panel("presets")
        self._set_status(f"Presets: {'ON' if self._presets_visible else 'OFF'}")
        self.refresh(layout=True)

    def action_toggle_scope(self) -> None:
        self._scope_all = not self._scope_all
        self._refresh_orders_table()
        self._refresh_logs_table()
        self._journal_write(
            event="SCOPE",
            reason=None,
            data={"scope": "ALL" if self._scope_all else "INSTANCE"},
        )
        self._set_status("Scope: ALL" if self._scope_all else "Scope: Instance")
        self.refresh(layout=True)

    def action_cycle_focus(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def _cycle_focus(self, direction: int) -> None:
        panels = list(self._PANEL_ORDER)
        if not self._presets_visible:
            panels.remove("presets")
        try:
            idx = panels.index(self._active_panel)
        except ValueError:
            idx = 0
        self._focus_panel(panels[(idx + direction) % len(panels)])

    def _panel_table(self, panel: str | None = None) -> DataTable:
        target = str(panel or self._active_panel or "logs")
        return {
            "presets": self._presets_table,
            "instances": self._instances_table,
            "orders": self._orders_table,
            "logs": self._logs_table,
        }.get(target, self._logs_table)

    def _focus_panel(self, panel: str) -> None:
        self._active_panel = panel
        self._panel_table(panel).focus()
        self._render_status()

    def _set_status(self, message: str, *, render_bot: bool = False) -> None:
        self._status = message
        if bool(render_bot):
            self._render_bot()
        else:
            self._render_status()

    def _dispatch_panel_handler(self, handlers: dict[str, str], *, default: str) -> None:
        fn_name = str(handlers.get(str(self._active_panel), default))
        fn = getattr(self, fn_name, None)
        if callable(fn):
            fn()

    def action_toggle_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._set_status("Run: select an instance")
            return
        instance.state = "PAUSED" if instance.state == "RUNNING" else "RUNNING"
        self._refresh_instances_table()
        self._set_status(f"Instance {instance.instance_id}: {instance.state}")

    def action_context_space(self) -> None:
        self._dispatch_panel_handler(self._SPACE_HANDLER_BY_PANEL, default="action_toggle_instance")

    def action_delete_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._set_status("Del: select an instance")
            return
        self._instances = [i for i in self._instances if i.instance_id != instance.instance_id]
        self._orders = [o for o in self._orders if o.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_orders_table()
        self._set_status(f"Deleted instance {instance.instance_id}")

    def action_cycle_dte_filter(self) -> None:
        dtes: list[int] = []
        if self._payload:
            raw = self._payload.get("grid", {}).get("dte", [])
            for value in raw if isinstance(raw, list) else []:
                try:
                    dtes.append(int(value))
                except (TypeError, ValueError):
                    continue
        dtes = sorted(set(dtes))
        cycle: list[int | None] = [None]
        if 0 not in dtes:
            cycle.append(0)
        cycle.extend(dtes)
        try:
            idx = cycle.index(self._filter_dte)
        except ValueError:
            idx = 0
        self._filter_dte = cycle[(idx + 1) % len(cycle)]
        self._rebuild_presets_table()
        self._set_status("Filter: DTE=ALL" if self._filter_dte is None else f"Filter: DTE={self._filter_dte}")
        self.refresh(layout=True)

    def action_cycle_win_filter(self) -> None:
        cycle: list[float | None] = [0.5, 0.6, 0.7, 0.8, None]
        try:
            idx = cycle.index(self._filter_min_win_rate)
        except ValueError:
            idx = 0
        self._filter_min_win_rate = cycle[(idx + 1) % len(cycle)]
        self._rebuild_presets_table()
        if self._filter_min_win_rate is None:
            self._set_status("Filter: Win=ALL")
        else:
            self._set_status(f"Filter: Win≥{int(self._filter_min_win_rate * 100)}%")
        self.refresh(layout=True)

    def action_stop_bot(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._set_status("Stop: select an instance")
            return
        instance.state = "PAUSED"
        self._orders = [o for o in self._orders if o.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_orders_table()
        self._journal_write(event="STOP_INSTANCE", instance=instance, reason=None, data=None)
        self._set_status(f"Stopped instance {instance.instance_id}: paused + cleared orders")

    def _kill_all(self) -> None:
        for instance in self._instances:
            instance.state = "PAUSED"
        self._orders.clear()
        self._refresh_instances_table()
        self._refresh_orders_table()
        self._journal_write(event="KILL_ALL", reason=None, data=None)
        self._set_status("KILL: paused all + cleared orders")

    def action_activate(self) -> None:
        self._dispatch_panel_handler(self._ACTIVATE_HANDLER_BY_PANEL, default="_activate_logs_panel")

    def _activate_presets_panel(self) -> None:
        row = self._selected_preset_row()
        if row is None:
            self._set_status("Preset: none selected")
            return
        if isinstance(row, _PresetHeader):
            self._toggle_preset_node(row.node_id)
            return
        self._open_config_for_preset(row)

    def _activate_instances_panel(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._set_status("Instance: none selected")
            return
        self._open_config_for_instance(instance)

    def _activate_logs_panel(self) -> None:
        self._set_status("Logs: no action")

    def action_cancel_order(self) -> None:
        if self._active_panel != "orders":
            self._set_status("Cancel: focus Orders")
            return
        order = self._selected_order()
        if not order:
            self._set_status("Cancel: no order selected")
            return
        if order.status == "STAGED":
            self._orders = [o for o in self._orders if o is not order]
            self._journal_write(event="ORDER_DROPPED", order=order, reason="staged", data=None)
            self._refresh_orders_table()
            self._set_status("Cancelled staged order", render_bot=True)
            return
        if order.status != "WORKING":
            self._set_status(f"Cancel: not working ({order.status})", render_bot=True)
            return
        if self._cancel_task and not self._cancel_task.done():
            self._set_status("Cancel: busy", render_bot=True)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("Cancel: no loop", render_bot=True)
            return
        self._cancel_task = loop.create_task(self._cancel_working_order(order))

    async def _cancel_working_order(self, order: _BotOrder) -> None:
        if order.status != "WORKING":
            return
        trade = order.trade
        if trade is None:
            order.error = "Cancel error: missing IB trade handle"
            self._set_status("Cancel error: missing IB trade handle")
            self._journal_write(event="CANCEL_ERROR", order=order, reason=None, data={"exc": "missing trade"})
            self._refresh_orders_table()
            self._render_bot()
            return
        order.status = "CANCELING"
        self._journal_write(event="CANCEL_REQUEST", order=order, reason=None, data=None)
        self._refresh_orders_table()
        self._set_status(f"Canceling #{order.order_id or 0}...", render_bot=True)
        try:
            await self._client.cancel_trade(trade)
            self._journal_write(event="CANCEL_SENT", order=order, reason=None, data=None)
            self._set_status(f"Cancel sent #{order.order_id or 0}")
        except Exception as exc:
            order.status = "WORKING"
            order.error = f"Cancel error: {exc}"
            self._set_status(f"Cancel error: {exc}")
            self._journal_write(event="CANCEL_ERROR", order=order, reason=None, data={"exc": str(exc)})
        self._refresh_orders_table()
        self._render_bot()

    def _selected_preset(self) -> _BotPreset | None:
        row = self._selected_preset_row()
        return row if isinstance(row, _BotPreset) else None

    def _selected_preset_row(self) -> _BotPreset | _PresetHeader | None:
        row = self._presets_table.cursor_coordinate.row
        if row < 0 or row >= len(self._preset_rows):
            return None
        return self._preset_rows[row]

    def _selected_instance(self) -> _BotInstance | None:
        row = self._instances_table.cursor_coordinate.row
        if row < 0 or row >= len(self._instance_rows):
            return None
        return self._instance_rows[row]

    def _selected_order(self) -> _BotOrder | None:
        row = self._orders_table.cursor_coordinate.row
        if row < 0 or row >= len(self._order_rows):
            return None
        return self._order_rows[row]

    def _scope_instance_id(self) -> int | None:
        if self._scope_all:
            return None
        instance = self._selected_instance()
        return instance.instance_id if instance else None

    def _open_config_for_preset(self, preset: _BotPreset) -> None:
        entry = preset.entry
        strategy = copy.deepcopy(entry.get("strategy", {}) or {})
        strategy.setdefault("instrument", "options")
        strategy.setdefault("max_entries_per_day", 1)
        strategy.setdefault("exit_dte", 0)
        strategy.setdefault("stop_loss_basis", "max_loss")
        strategy.setdefault("spot_sec_type", "STK")
        strategy.setdefault("spot_exchange", "")
        if "signal_bar_size" not in strategy:
            strategy["signal_bar_size"] = str(self._payload.get("bar_size", "1 hour") if self._payload else "1 hour")
        if "signal_use_rth" not in strategy:
            strategy["signal_use_rth"] = bool(self._payload.get("use_rth", False) if self._payload else False)
        strategy.setdefault("spot_close_eod", False)
        if "directional_spot" not in strategy:
            strategy["directional_spot"] = {"up": {"action": "BUY", "qty": 1}, "down": {"action": "SELL", "qty": 1}}
        filters = _filters_for_group(self._payload, preset.group) if self._payload else None
        symbol = str(
            entry.get("symbol")
            or strategy.get("symbol")
            or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()

        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._set_status("Config: cancelled")
                return
            instance = _BotInstance(
                instance_id=self._next_instance_id,
                group=result.group,
                symbol=result.symbol,
                strategy=result.strategy,
                filters=result.filters,
                metrics=entry.get("metrics"),
            )
            self._next_instance_id += 1
            self._instances.append(instance)
            self._refresh_instances_table()
            self._instances_table.cursor_coordinate = (max(len(self._instances) - 1, 0), 0)
            self._focus_panel("instances")
            self._set_status(f"Created instance {instance.instance_id}")
            self._journal_write(
                event="INSTANCE_CREATED",
                instance=instance,
                reason=None,
                data={"metrics": entry.get("metrics")},
            )

        self.app.push_screen(
            BotConfigScreen(
                mode="create",
                instance_id=None,
                group=preset.group,
                symbol=symbol,
                strategy=strategy,
                filters=filters,
            ),
            _on_done,
        )

    def _open_config_for_instance(self, instance: _BotInstance) -> None:
        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._set_status("Config: cancelled")
                return
            instance.group = result.group
            instance.symbol = result.symbol
            instance.strategy = result.strategy
            instance.filters = result.filters
            self._refresh_instances_table()
            self._set_status(f"Updated instance {instance.instance_id}")
            self._journal_write(event="INSTANCE_UPDATED", instance=instance, reason=None, data=None)

        self.app.push_screen(
            BotConfigScreen(
                mode="update",
                instance_id=instance.instance_id,
                group=instance.group,
                symbol=instance.symbol,
                strategy=instance.strategy,
                filters=instance.filters,
            ),
            _on_done,
        )

    def _load_leaderboard(self) -> None:
        """Load CURRENT champion presets (README-driven).

        Bot Hub presets are intentionally limited to promoted CURRENT champions documented in:
        - `tradebot/backtest/README.md` (TQQQ spot champ)
        - `backtests/slv/README.md` (SLV spot champ)
        """

        repo_root = Path(__file__).resolve().parents[2]

        def _read_text(path: Path) -> str | None:
            try:
                return path.read_text()
            except Exception:
                return None

        def _resolve_existing_json(rel_path: str) -> Path | None:
            candidate = repo_root / rel_path
            return candidate if candidate.exists() else None

        def _pick_group(payload: dict) -> dict | None:
            groups = payload.get("groups", [])
            if not isinstance(groups, list) or not groups:
                return None
            for group in groups:
                if not isinstance(group, dict):
                    continue
                name = str(group.get("name") or "")
                if "KINGMAKER #01" in name:
                    return group
            first = groups[0]
            return first if isinstance(first, dict) else None

        def _load_champion_group(*, symbol: str, version: str, path: Path) -> dict | None:
            try:
                payload = json.loads(path.read_text())
            except Exception:
                return None
            if not isinstance(payload, dict):
                return None
            group = _pick_group(payload)
            if not isinstance(group, dict):
                return None
            group = dict(group)
            group["_source"] = f"champion:{symbol}:v{version}"

            name = str(group.get("name") or "")
            spot_tag = f"Spot ({symbol})"
            if spot_tag in name:
                group["name"] = name.replace(spot_tag, f"{spot_tag} v{version}", 1)
            return group

        groups: list[dict] = []
        self._spot_champ_version = None

        # TQQQ CURRENT (from tradebot/backtest/README.md)
        backtest_readme = _read_text(self._backtest_readme_path)
        tqqq_ver: str | None = None
        tqqq_path: Path | None = None
        if backtest_readme:
            bullet = re.search(
                r"^- CURRENT \(v(?P<ver>\d+)\): `(?P<path>backtests/out/[^`]+\.json)`",
                backtest_readme,
                flags=re.MULTILINE,
            )
            if bullet:
                tqqq_ver = bullet.group("ver")
                tqqq_path = _resolve_existing_json(bullet.group("path"))

            if tqqq_ver is None:
                head = re.search(r"^#### CURRENT \(v(?P<ver>\d+)\)", backtest_readme, flags=re.MULTILINE)
                if head:
                    tqqq_ver = head.group("ver")
                    tail = backtest_readme[head.end() :]
                    next_head = re.search(r"^####\\s+", tail, flags=re.MULTILINE)
                    section = tail[: next_head.start()] if next_head else tail
                    match = re.search(r"`(?P<path>backtests/out/[^`]+\\.json)`", section)
                    if match:
                        tqqq_path = _resolve_existing_json(match.group("path"))

            if tqqq_ver and tqqq_path is None:
                out_dir = repo_root / "backtests" / "out"
                if out_dir.exists():
                    candidates = sorted(out_dir.glob(f"tqqq_exec5m_v{tqqq_ver}_*top*.json"))
                    if candidates:
                        tqqq_path = candidates[0]

        if tqqq_ver and tqqq_path:
            group = _load_champion_group(symbol="TQQQ", version=tqqq_ver, path=tqqq_path)
            if group is not None:
                self._spot_champ_version = tqqq_ver
                groups.append(group)

        # SLV CURRENT (from backtests/slv/README.md)
        slv_readme_path = repo_root / "backtests" / "slv" / "README.md"
        slv_readme = _read_text(slv_readme_path)
        slv_ver: str | None = None
        slv_path: Path | None = None
        if slv_readme:
            head = re.search(r"^### CURRENT \(v(?P<ver>\d+)\)", slv_readme, flags=re.MULTILINE)
            if head:
                slv_ver = head.group("ver")
                tail = slv_readme[head.end() :]
                next_head = re.search(r"^###\\s+", tail, flags=re.MULTILINE)
                section = tail[: next_head.start()] if next_head else tail
                match = re.search(r"`(?P<path>backtests/slv/[^`]+\\.json)`", section)
                if match:
                    slv_path = _resolve_existing_json(match.group("path"))

            if slv_ver and slv_path is None:
                slv_dir = repo_root / "backtests" / "slv"
                candidates = sorted(slv_dir.glob(f"slv_exec5m_v{slv_ver}_*top*.json"))
                if candidates:
                    slv_path = candidates[0]

        if slv_ver and slv_path:
            group = _load_champion_group(symbol="SLV", version=slv_ver, path=slv_path)
            if group is not None:
                groups.append(group)

        has_champions = bool(groups)
        if not has_champions:
            self._set_status("No CURRENT champions found. Check README paths.")

        # Debug / test preset (fast EMA crosses) to validate logging + auto-order behavior.
        groups.append(
            {
                "name": "Spot (SLV) DEBUG FAST 1m EMA 1/2",
                "_source": "debug:slv_fast_1m",
                "entries": [
                    {
                        "symbol": "SLV",
                        "metrics": {"pnl": 0.0, "win_rate": 0.0, "trades": 0},
                        "strategy": {
                            "instrument": "spot",
                            "symbol": "SLV",
                            "signal_bar_size": "1 min",
                            "signal_use_rth": False,
                            "entry_signal": "ema",
                            "ema_preset": "1/2",
                            "ema_entry_mode": "cross",
                            "entry_confirm_bars": 0,
                            "entry_days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                            "max_entries_per_day": 500,
                            "exit_on_signal_flip": True,
                            "flip_exit_min_hold_bars": 0,
                            "flip_exit_only_if_profit": False,
                            "spot_exit_mode": "pct",
                            "spot_profit_target_pct": 0.002,
                            "spot_stop_loss_pct": 0.002,
                            "directional_spot": {
                                "up": {"action": "BUY", "qty": 1},
                                "down": {"action": "SELL", "qty": 1},
                            },
                        },
                    }
                ],
            }
        )

        self._payload = {"groups": groups}
        self._group_eval_by_name = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or "")
            eval_payload = group.get("_eval")
            if name and isinstance(eval_payload, dict):
                self._group_eval_by_name[name] = eval_payload
        self._rebuild_presets_table()

    def _rebuild_presets_table(self) -> None:
        self._presets = []
        self._preset_rows = []
        self._presets_table.clear(columns=True)
        self._presets_table.add_column("Preset")
        self._presets_table.add_column("Hours", width=10)
        self._presets_table.add_column("TF/Exec", width=10)
        self._presets_table.add_column("TP", width=7)
        self._presets_table.add_column("SL", width=7)
        self._presets_table.add_column("EMA", width=6)
        self._presets_table.add_column("P/DD 10|2|1", width=14)
        self._presets_table.add_column("PnL 10|2|1", width=18)
        self._presets_table.add_column("DD 10|2|1", width=18)

        payload = self._payload or {}
        groups = payload.get("groups", [])
        if not isinstance(groups, list) or not groups:
            self._move_cursor_to_first_preset()
            self._render_status()
            self._presets_table.refresh(repaint=True)
            return

        def _get_symbol(group_name: str, entry: dict, strat: dict) -> str:
            raw = entry.get("symbol") or strat.get("symbol") or payload.get("symbol") or ""
            cleaned = str(raw or "").strip().upper()
            if cleaned:
                return cleaned
            text = str(group_name or "")
            if "(" in text and ")" in text:
                inside = text.split("(", 1)[1].split(")", 1)[0].strip()
                if inside and inside.replace("-", "").isalnum():
                    return inside.upper()
            return "UNKNOWN"

        def _get_dd(metrics: dict) -> float | None:
            for key in ("max_drawdown", "dd"):
                if metrics.get(key) is None:
                    continue
                try:
                    dd = float(metrics.get(key))
                except (TypeError, ValueError):
                    continue
                if dd >= 0:
                    return dd
            return None

        def _score(metrics: dict) -> float:
            try:
                value = float(metrics.get("pnl_over_dd"))
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                return value
            try:
                pnl = float(metrics.get("pnl") or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0
            dd = _get_dd(metrics)
            return pnl / dd if dd and dd > 0 else float("-inf")

        def _compact_bar_size(raw: str) -> str:
            text = str(raw or "").strip()
            if not text:
                return "?"
            parts = text.split()
            if len(parts) >= 2 and parts[0].isdigit():
                num = parts[0]
                unit = parts[1].lower()
                if unit.startswith("min"):
                    return f"{num}m"
                if unit.startswith("hour"):
                    return f"{num}h"
                if unit.startswith("day"):
                    return f"{num}d"
            return (
                text.replace(" mins", "m")
                .replace(" min", "m")
                .replace(" hours", "h")
                .replace(" hour", "h")
                .replace(" days", "d")
                .replace(" day", "d")
            )

        def _fmt_money_compact(value: float | None) -> str:
            if value is None:
                return "-"
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return "-"
            sign = "-" if parsed < 0 else ""
            parsed = abs(parsed)
            if parsed >= 1_000_000:
                return f"{sign}{parsed / 1_000_000:.1f}M"
            if parsed >= 10_000:
                return f"{sign}{parsed / 1_000:.1f}k"
            return f"{sign}{parsed:,.0f}"

        def _fmt_ratio_compact(value: float | None) -> str:
            if value is None:
                return "-"
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return "-"

        def _eval_windows(group_name: str) -> list[dict]:
            eval_payload = self._group_eval_by_name.get(group_name)
            if not isinstance(eval_payload, dict):
                return []
            windows = eval_payload.get("windows")
            if not isinstance(windows, list) or not windows:
                return []
            cleaned = [w for w in windows if isinstance(w, dict)]
            cleaned.sort(key=lambda w: str(w.get("start") or ""))
            return cleaned

        def _window_triplets(group_name: str, metrics: dict) -> tuple[str, Text, Text]:
            windows = _eval_windows(group_name)
            pnls: list[float | None] = []
            dds: list[float | None] = []
            ratios: list[float | None] = []

            def _as_float(value) -> float | None:
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            if windows:
                for w in windows:
                    pnl = _as_float(w.get("pnl"))
                    dd = _as_float(w.get("dd"))
                    if dd is None:
                        dd = _as_float(w.get("max_drawdown"))
                    ratio = _as_float(w.get("pnl_over_dd"))
                    if ratio is None and pnl is not None and dd is not None and dd > 0:
                        ratio = pnl / dd
                    pnls.append(pnl)
                    dds.append(dd)
                    ratios.append(ratio)
            else:
                pnl = _as_float(metrics.get("pnl"))
                dd = _get_dd(metrics)
                ratio = _as_float(metrics.get("pnl_over_dd"))
                if ratio is None and pnl is not None and dd is not None and dd > 0:
                    ratio = pnl / dd
                pnls = [pnl]
                dds = [dd]
                ratios = [ratio]

            ratio_s = "/".join(_fmt_ratio_compact(v) for v in ratios)
            pnl_s = "/".join(_fmt_money_compact(v) for v in pnls)
            dd_s = "/".join(_fmt_money_compact(v) for v in dds)

            pnl_style = "green" if any(v is not None for v in pnls) and all((v or 0) >= 0 for v in pnls) else ""
            pnl_cell = Text(pnl_s, style=pnl_style) if pnl_s else Text("")
            dd_cell = Text(dd_s, style="red") if dd_s else Text("")
            return ratio_s, pnl_cell, dd_cell

        def _spot_tp_sl(strat: dict) -> tuple[str, str]:
            exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
            if exit_mode == "atr":
                pt = strat.get("spot_pt_atr_mult")
                sl = strat.get("spot_sl_atr_mult")
                pt_s = "-"
                sl_s = "-"
                if pt is not None:
                    try:
                        pt_s = f"x{float(pt):.2f}"
                    except (TypeError, ValueError):
                        pt_s = "-"
                if sl is not None:
                    try:
                        sl_s = f"x{float(sl):.2f}"
                    except (TypeError, ValueError):
                        sl_s = "-"
                return pt_s, sl_s
            pt = strat.get("spot_profit_target_pct")
            sl = strat.get("spot_stop_loss_pct")
            pt_s = "-"
            sl_s = "-"
            if pt is not None:
                try:
                    pt_s = _fmt_pct(float(pt) * 100.0)
                except (TypeError, ValueError):
                    pt_s = "-"
            if sl is not None:
                try:
                    sl_s = _fmt_pct(float(sl) * 100.0)
                except (TypeError, ValueError):
                    sl_s = "-"
            return pt_s, sl_s

        def _options_tp_sl(strat: dict) -> tuple[str, str]:
            try:
                pt = float(strat.get("profit_target", 0.0)) * 100.0
            except (TypeError, ValueError):
                pt = 0.0
            try:
                sl = float(strat.get("stop_loss", 0.0)) * 100.0
            except (TypeError, ValueError):
                sl = 0.0
            return _fmt_pct(pt), _fmt_pct(sl)

        def _short_preset_name(*, group_name: str, symbol: str, instrument: str, source: str) -> str:
            base = _clean_group_label(group_name)
            if instrument == "spot" and symbol and f"Spot ({symbol})" in base:
                base = base.split(f"Spot ({symbol})", 1)[1].strip()
                while base[:1] in ("-", "—", ":", "|"):
                    base = base[1:].strip()
            return base or group_name

        contracts: dict[str, dict] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("name") or "")
            source = str(group.get("_source") or "").strip() or "unknown"
            entries = group.get("entries", [])
            if not isinstance(entries, list) or not entries:
                continue
            for entry_idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                strat = entry.get("strategy", {}) if isinstance(entry.get("strategy"), dict) else {}
                metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
                instrument = self._strategy_instrument(strat)

                legs = strat.get("legs", [])
                if instrument == "options" and (not isinstance(legs, list) or not legs):
                    continue

                try:
                    win = float(metrics.get("win_rate", 0.0))
                except (TypeError, ValueError):
                    win = 0.0
                if self._filter_min_win_rate is not None and win < self._filter_min_win_rate:
                    continue

                try:
                    dte = int(strat.get("dte", 0))
                except (TypeError, ValueError):
                    dte = 0
                if instrument == "options" and self._filter_dte is not None and dte != self._filter_dte:
                    continue

                symbol = _get_symbol(group_name, entry, strat)

                signal_bar = str(strat.get("signal_bar_size") or "").strip()
                if not signal_bar and instrument == "options":
                    signal_bar = str(payload.get("bar_size") or "").strip()
                tf = signal_bar or "?"

                preset = _BotPreset(group=group_name, entry=entry)
                name = _short_preset_name(group_name=group_name, symbol=symbol, instrument=instrument, source=source)
                row_id = f"preset:{source}:{group_name}:{entry_idx}"

                item = {
                    "row_id": row_id,
                    "preset": preset,
                    "name": name,
                    "symbol": symbol,
                    "instrument": instrument,
                    "source": source,
                    "tf": tf,
                    "dte": dte,
                    "metrics": metrics,
                    "strategy": strat,
                    "score": _score(metrics),
                }
                bucket = contracts.setdefault(symbol, {"spot": {}, "options": {}})
                if instrument == "spot":
                    bucket["spot"].setdefault(tf, []).append(item)
                else:
                    bucket["options"].setdefault(dte, []).append(item)

        symbols = sorted(contracts.keys(), key=lambda sym: (0 if sym == "TQQQ" else 1, sym))
        if not self._preset_expand_initialized:
            self._preset_expanded = {f"contract:{sym}" for sym in symbols}
            self._preset_expanded |= {f"contract:{sym}|spot" for sym in symbols}
            self._preset_expand_initialized = True
            self._preset_known_contracts = set(symbols)
        else:
            for sym in symbols:
                if sym not in self._preset_known_contracts:
                    self._preset_expanded.add(f"contract:{sym}")
                    self._preset_expanded.add(f"contract:{sym}|spot")
            self._preset_known_contracts.update(symbols)

        def _best(items: list[dict]) -> dict | None:
            return max(items, key=lambda it: float(it.get("score", float("-inf")))) if items else None

        def _best_version(best: dict | None) -> str | None:
            if best is None:
                return None
            raw = str(best.get("name") or "")
            match = re.search(r"\bv(?P<ver>\d+)\b", raw, flags=re.IGNORECASE)
            if match:
                return f"v{match.group('ver')}"
            preset = best.get("preset")
            group_name = getattr(preset, "group", "")
            match = re.search(r"\bv(?P<ver>\d+)\b", str(group_name or ""), flags=re.IGNORECASE)
            return f"v{match.group('ver')}" if match else None

        def _label_with_version(label: str, best: dict | None) -> str:
            ver = _best_version(best)
            if not ver:
                return label
            if re.search(rf"\\b{re.escape(ver)}\\b", label):
                return label
            return f"{label} {ver}"

        def _hours_label(*, strat: dict, filters: dict | None) -> str:
            use_rth_raw = strat.get("signal_use_rth")
            use_rth = None if use_rth_raw is None else bool(use_rth_raw)

            start = end = None
            if isinstance(filters, dict):
                raw_start_et = filters.get("entry_start_hour_et")
                raw_end_et = filters.get("entry_end_hour_et")
                if raw_start_et is not None and raw_end_et is not None:
                    try:
                        start = int(raw_start_et)
                        end = int(raw_end_et)
                    except (TypeError, ValueError):
                        start = None
                        end = None
                else:
                    raw_start = filters.get("entry_start_hour")
                    raw_end = filters.get("entry_end_hour")
                    if raw_start is not None and raw_end is not None:
                        try:
                            start = int(raw_start)
                            end = int(raw_end)
                        except (TypeError, ValueError):
                            start = None
                            end = None

            cutoff = None
            if isinstance(filters, dict) and filters.get("risk_entry_cutoff_hour_et") is not None:
                try:
                    cutoff = int(filters.get("risk_entry_cutoff_hour_et"))
                except (TypeError, ValueError):
                    cutoff = None

            if start is not None and end is not None:
                prefix = "R" if use_rth is True else ("F" if use_rth is False else "")
                label = f"{prefix}{start}-{end}"
                if cutoff is not None:
                    label = f"{label}c{cutoff}"
                return label

            if use_rth is True:
                return "RTH"
            if use_rth is False:
                return "24/5"
            return "-"

        def _add_leaf(item: dict, *, depth: int) -> None:
            preset = item["preset"]
            strat = item["strategy"]
            metrics = item["metrics"]
            instrument = item["instrument"]

            if instrument == "spot":
                exec_bar = str(strat.get("spot_exec_bar_size") or "").strip()
                if exec_bar:
                    tf_dte = f"{_compact_bar_size(item['tf'])}→{_compact_bar_size(exec_bar)}"
                else:
                    tf_dte = _compact_bar_size(item["tf"])
                tp_s, sl_s = _spot_tp_sl(strat)
            else:
                tf_dte = str(int(item["dte"]))
                tp_s, sl_s = _options_tp_sl(strat)

            filters = _filters_for_group(payload, preset.group) if self._payload else None
            hours_s = _hours_label(strat=strat, filters=filters)[:10]
            ema = str(strat.get("ema_preset", ""))[:6]

            ratio_s, pnl_trip, dd_trip = _window_triplets(preset.group, metrics)

            label = f"{'  ' * depth}{item['name']}".rstrip()
            self._presets.append(preset)
            self._preset_rows.append(preset)
            self._presets_table.add_row(
                label,
                hours_s,
                tf_dte,
                tp_s,
                sl_s,
                ema,
                ratio_s,
                pnl_trip,
                dd_trip,
                key=item["row_id"],
            )

        def _add_header(node_id: str, *, depth: int, label: str, best: dict | None) -> bool:
            expanded = node_id in self._preset_expanded
            caret = "▾" if expanded else "▸"
            left = f"{'  ' * depth}{caret} {label}".rstrip()
            legs_cell = ""
            tf_dte = ""
            tp_s = ""
            sl_s = ""
            ema = ""
            ratio_s = ""
            pnl_trip = Text("")
            dd_trip = Text("")

            if best is not None:
                filters = _filters_for_group(payload, best["preset"].group) if self._payload else None
                legs_cell = _hours_label(strat=best["strategy"], filters=filters)[:10]
                strat = best["strategy"]
                instrument = best["instrument"]
                if instrument == "spot":
                    exec_bar = str(strat.get("spot_exec_bar_size") or "").strip()
                    if exec_bar:
                        tf_dte = f"{_compact_bar_size(best['tf'])}→{_compact_bar_size(exec_bar)}"
                    else:
                        tf_dte = _compact_bar_size(best["tf"])
                    tp_s, sl_s = _spot_tp_sl(strat)
                else:
                    tf_dte = str(int(best["dte"]))
                    tp_s, sl_s = _options_tp_sl(strat)
                ema = str(strat.get("ema_preset", ""))[:6]
                metrics = best["metrics"]
                ratio_s, pnl_trip, dd_trip = _window_triplets(best["preset"].group, metrics)

            self._preset_rows.append(_PresetHeader(node_id=node_id, depth=depth, label=label))
            self._presets_table.add_row(
                Text(left, style="bold"),
                Text(legs_cell, style="dim") if legs_cell else Text(""),
                tf_dte,
                tp_s,
                sl_s,
                ema,
                ratio_s,
                pnl_trip,
                dd_trip,
                key=node_id,
            )
            return expanded

        for symbol in symbols:
            bucket = contracts[symbol]
            all_items: list[dict] = []
            for tf_items in bucket["spot"].values():
                all_items.extend(tf_items)
            for dte_items in bucket["options"].values():
                all_items.extend(dte_items)

            contract_node = f"contract:{symbol}"
            contract_best = _best(all_items)
            contract_expanded = _add_header(
                contract_node,
                depth=0,
                label=_label_with_version(symbol, contract_best),
                best=contract_best,
            )
            if not contract_expanded:
                continue

            spot_node = f"{contract_node}|spot"
            spot_items = [it for tf_items in bucket["spot"].values() for it in tf_items]
            spot_best = _best(spot_items)
            spot_expanded = _add_header(
                spot_node,
                depth=1,
                label=_label_with_version(f"{symbol} - Spot", spot_best),
                best=spot_best,
            )
            if spot_expanded:
                spot_tfs = sorted(bucket["spot"].keys())
                if len(spot_tfs) == 1:
                    tf_items = bucket["spot"][spot_tfs[0]]
                    ordered = sorted(tf_items, key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]))
                    for item in ordered:
                        _add_leaf(item, depth=2)
                else:
                    for tf in spot_tfs:
                        tf_items = bucket["spot"][tf]
                        tf_node = f"{spot_node}|tf:{tf}"
                        tf_expanded = _add_header(tf_node, depth=2, label=str(tf), best=_best(tf_items))
                        if tf_expanded:
                            ordered = sorted(
                                tf_items,
                                key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]),
                            )
                            for item in ordered:
                                _add_leaf(item, depth=3)

            opt_node = f"{contract_node}|options"
            opt_items = [it for items in bucket["options"].values() for it in items]
            if opt_items:
                opt_expanded = _add_header(opt_node, depth=1, label=f"{symbol} - Options", best=_best(opt_items))
                if opt_expanded:
                    for dte in sorted(bucket["options"].keys()):
                        dte_items = bucket["options"][dte]
                        dte_node = f"{opt_node}|dte:{dte}"
                        dte_expanded = _add_header(dte_node, depth=2, label=f"DTE {dte}", best=_best(dte_items))
                        if dte_expanded:
                            ordered = sorted(
                                dte_items,
                                key=lambda it: (-float(it.get("score", float("-inf"))), it["name"]),
                            )
                            for item in ordered:
                                _add_leaf(item, depth=3)

        self._move_cursor_to_first_preset()
        self._render_status()
        self._presets_table.refresh(repaint=True)

    def _move_cursor_to_first_preset(self) -> None:
        if not self._preset_rows:
            return
        self._presets_table.cursor_coordinate = (0, 0)

    def _toggle_preset_node(self, node_id: str) -> None:
        if not node_id:
            return
        if node_id in self._preset_expanded:
            self._preset_expanded.remove(node_id)
        else:
            self._preset_expanded.add(node_id)
        self._rebuild_presets_table()
        try:
            row = self._presets_table.get_row_index(node_id)
        except Exception:
            row = None
        if row is not None and self._presets_table.is_valid_row_index(row):
            self._presets_table.cursor_coordinate = (row, 0)

    def _setup_tables(self) -> None:
        self._instances_table.clear(columns=True)
        self._instances_table.add_columns("ID", "Strategy", "DTE", "State", "BT PnL", "Unreal", "Realized")

        self._orders_table.clear(columns=True)
        self._orders_table.add_columns("When", "Inst", "Side", "Qty", "Contract", "Lmt", "B/A", "Status", "Unreal", "Realized")

        self._logs_table.clear(columns=True)
        self._logs_table.add_columns("When", "Inst", "Sym", "Event", "Reason", "Msg")

    def _cursor_move(self, direction: int) -> None:
        table = self._panel_table()
        if direction > 0:
            table.action_cursor_down()
        else:
            table.action_cursor_up()
        if self._active_panel == "instances" and not self._scope_all:
            self._refresh_orders_table()
            self._refresh_logs_table()
        self._render_status()

    def _refresh_instances_table(self) -> None:
        prev_row = self._instances_table.cursor_coordinate.row
        prev_id = None
        if 0 <= prev_row < len(self._instance_rows):
            prev_id = self._instance_rows[prev_row].instance_id

        self._instances_table.clear()
        self._instance_rows = []
        for instance in self._instances:
            instrument = self._strategy_instrument(instance.strategy or {})
            if instrument == "spot":
                dte = "-"
            else:
                dte = instance.strategy.get("dte", "")
            bt_pnl = ""
            if instance.metrics:
                try:
                    bt_pnl = _pnl_text(float(instance.metrics.get("pnl", 0.0)))
                except (TypeError, ValueError):
                    bt_pnl = ""
            unreal_cell, realized_cell = _instance_pnl_cells(instance, self._positions)
            self._instances_table.add_row(
                str(instance.instance_id),
                instance.group[:24],
                str(dte),
                instance.state,
                bt_pnl,
                unreal_cell,
                realized_cell,
            )
            self._instance_rows.append(instance)

        if prev_id is not None:
            for idx, inst in enumerate(self._instance_rows):
                if inst.instance_id == prev_id:
                    self._instances_table.cursor_coordinate = (idx, 0)
                    break
        elif self._instance_rows:
            self._instances_table.cursor_coordinate = (0, 0)

        if not self._scope_all:
            self._refresh_orders_table()
            self._refresh_logs_table()

    async def _refresh_positions(self) -> None:
        try:
            items = await self._client.fetch_portfolio()
        except Exception as exc:  # pragma: no cover - UI surface
            self._set_status(f"Positions error: {exc}")
            return
        self._positions = [item for item in items if item.contract.secType in _SECTION_TYPES]
        self._positions.sort(key=_portfolio_sort_key, reverse=True)
        self._refresh_orders_table()

    def _refresh_logs_table(self) -> None:
        prev_count = len(self._log_rows)
        prev_row = self._logs_table.cursor_coordinate.row
        was_at_end = prev_count > 0 and prev_row >= (prev_count - 1)

        self._logs_table.clear()
        self._log_rows = []

        active_ids = {str(int(i.instance_id)) for i in self._instances}
        scope = self._scope_instance_id()
        tail = self._log_events[-200:]
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            inst_id = str(entry.get("instance_id") or "")
            if inst_id and inst_id not in active_ids:
                continue
            if scope is not None and not self._scope_all and inst_id and inst_id != str(int(scope)):
                continue
            ts = entry.get("ts_et")
            when = ts.astimezone().strftime("%H:%M:%S") if isinstance(ts, datetime) else ""
            self._logs_table.add_row(
                when,
                inst_id,
                str(entry.get("symbol") or ""),
                str(entry.get("event") or ""),
                str(entry.get("reason") or ""),
                str(entry.get("msg") or "")[:64],
            )
            self._log_rows.append(entry)

        if not self._log_rows:
            return
        if self._active_panel != "logs" or was_at_end:
            self._logs_table.cursor_coordinate = (len(self._log_rows) - 1, 0)
        elif 0 <= prev_row < len(self._log_rows):
            self._logs_table.cursor_coordinate = (prev_row, 0)

    def _render_status(self) -> None:
        self._render_bot()

    async def _reprice_order(self, order: _BotOrder, *, mode: str) -> bool:
        prev_mode = order.exec_mode
        order.exec_mode = str(mode or "").strip().upper() or None
        mode_changed = order.exec_mode != prev_mode
        legs = order.legs or []
        if not legs:
            return mode_changed

        if len(legs) == 1 and order.order_contract.secType != "BAG":
            leg = legs[0]
            ticker = await self._client.ensure_ticker(leg.contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _limit_price_for_mode(bid, ask, last, action=leg.action, mode=mode)
            if limit is None:
                return False
            tick = _tick_size(leg.contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            changed = not math.isclose(limit, order.limit_price, rel_tol=0, abs_tol=tick / 2.0)
            order.limit_price = float(limit)
            order.bid = bid
            order.ask = ask
            order.last = last
            return changed or mode_changed

        debit_mid = 0.0
        debit_bid = 0.0
        debit_ask = 0.0
        desired_debit = 0.0
        tick = None
        for leg in legs:
            ticker = await self._client.ensure_ticker(leg.contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            mid = _midpoint(bid, ask)
            leg_mid = mid or last
            if leg_mid is None:
                return False
            leg_bid = bid or mid or last
            leg_ask = ask or mid or last
            leg_desired = _limit_price_for_mode(bid, ask, last, action=leg.action, mode=mode)
            if leg_bid is None or leg_ask is None or leg_desired is None:
                return False
            leg_tick = _tick_size(leg.contract, ticker, leg_desired)
            tick = leg_tick if tick is None else min(tick, leg_tick)
            sign = 1.0 if leg.action == "BUY" else -1.0
            debit_mid += sign * float(leg_mid) * leg.ratio
            debit_bid += sign * float(leg_bid) * leg.ratio
            debit_ask += sign * float(leg_ask) * leg.ratio
            desired_debit += sign * float(leg_desired) * leg.ratio

        tick = tick or 0.01
        order.action = "BUY"
        new_limit = _round_to_tick(float(desired_debit), tick)
        if not new_limit:
            return False
        new_bid = float(debit_bid)
        new_ask = float(debit_ask)
        new_last = float(debit_mid)
        changed = not math.isclose(
            new_limit, order.limit_price, rel_tol=0, abs_tol=tick / 2.0
        )
        order.limit_price = float(new_limit)
        order.bid = new_bid
        order.ask = new_ask
        order.last = new_last
        return changed or mode_changed

    def _strategy_instrument(self, strategy: dict) -> str:
        value = strategy.get("instrument", "options")
        cleaned = str(value or "options").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    def _spot_sec_type(self, instance: _BotInstance, symbol: str) -> str:
        raw = (instance.strategy or {}).get("spot_sec_type")
        cleaned = str(raw or "").strip().upper()
        if cleaned in ("STK", "FUT"):
            return cleaned
        sym = str(symbol or "").strip().upper()
        if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
            return "FUT"
        return "STK"

    def _spot_exchange(self, instance: _BotInstance, symbol: str, *, sec_type: str) -> str:
        raw = (instance.strategy or {}).get("spot_exchange")
        cleaned = str(raw or "").strip().upper()
        if cleaned:
            return cleaned
        if sec_type == "FUT":
            sym = str(symbol or "").strip().upper()
            if sym in ("YM", "MYM"):
                return "CBOT"
            return "CME"
        return "SMART"

    async def _spot_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        sec_type = self._spot_sec_type(instance, symbol)
        exchange = self._spot_exchange(instance, symbol, sec_type=sec_type)
        if sec_type == "FUT":
            contract = await self._client.front_future(symbol, exchange=exchange, cache_ttl_sec=3600.0)
            if contract is None:
                return None
            return contract

        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    async def _signal_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        if self._strategy_instrument(instance.strategy) == "spot":
            return await self._spot_contract(instance, symbol)
        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    def _signal_bar_size(self, instance: _BotInstance) -> str:
        raw = instance.strategy.get("signal_bar_size")
        if raw:
            return str(raw)
        if self._payload:
            return str(self._payload.get("bar_size", "1 hour"))
        return "1 hour"

    def _signal_use_rth(self, instance: _BotInstance) -> bool:
        raw = instance.strategy.get("signal_use_rth")
        if raw is not None:
            return bool(raw)
        if self._payload:
            return bool(self._payload.get("use_rth", False))
        return False

    def _signal_snapshot_kwargs(
        self,
        instance: _BotInstance,
        *,
        strategy: dict | None = None,
        ema_preset_raw: str | None = None,
        entry_signal_raw: str | None = None,
        include_orb: bool = False,
        include_spot_exit: bool = False,
    ) -> dict[str, object]:
        strat = strategy if isinstance(strategy, dict) else (instance.strategy or {})
        kwargs: dict[str, object] = {
            "ema_preset_raw": ema_preset_raw,
            "bar_size": self._signal_bar_size(instance),
            "use_rth": self._signal_use_rth(instance),
            "entry_mode_raw": strat.get("ema_entry_mode"),
            "entry_confirm_bars": strat.get("entry_confirm_bars", 0),
            "regime_ema_preset_raw": strat.get("regime_ema_preset"),
            "regime_bar_size_raw": strat.get("regime_bar_size"),
            "regime_mode_raw": strat.get("regime_mode"),
            "supertrend_atr_period_raw": strat.get("supertrend_atr_period"),
            "supertrend_multiplier_raw": strat.get("supertrend_multiplier"),
            "supertrend_source_raw": strat.get("supertrend_source"),
            "regime2_ema_preset_raw": strat.get("regime2_ema_preset"),
            "regime2_bar_size_raw": strat.get("regime2_bar_size"),
            "regime2_mode_raw": strat.get("regime2_mode"),
            "regime2_supertrend_atr_period_raw": strat.get("regime2_supertrend_atr_period"),
            "regime2_supertrend_multiplier_raw": strat.get("regime2_supertrend_multiplier"),
            "regime2_supertrend_source_raw": strat.get("regime2_supertrend_source"),
            "filters": instance.filters if isinstance(instance.filters, dict) else None,
        }
        if entry_signal_raw is not None:
            kwargs["entry_signal_raw"] = entry_signal_raw
        if include_orb:
            kwargs["orb_window_mins_raw"] = strat.get("orb_window_mins")
            kwargs["orb_open_time_et_raw"] = strat.get("orb_open_time_et")
        if include_spot_exit:
            kwargs["spot_exit_mode_raw"] = strat.get("spot_exit_mode")
            kwargs["spot_atr_period_raw"] = strat.get("spot_atr_period")
        return kwargs

    def _signal_duration_str(self, bar_size: str, *, filters: dict | None = None) -> str:
        label = str(bar_size or "").strip().lower()

        def _rank(duration: str) -> int:
            order = ("1 D", "2 D", "1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
            cleaned = str(duration or "").strip()
            try:
                return order.index(cleaned)
            except ValueError:
                return 0

        def _max_duration(a: str, b: str) -> str:
            return a if _rank(a) >= _rank(b) else b

        base = "2 W"
        if label.startswith(("1 min", "2 mins")):
            base = "2 D"
        elif label.startswith(("5 mins", "10 mins", "15 mins", "30 mins")):
            base = "1 W"
        elif "hour" in label:
            base = "2 W"
        elif "day" in label:
            base = "1 Y"

        if not isinstance(filters, dict) or not filters:
            return base

        # Daily shock detectors need enough sessions to become ready; otherwise shock gating and
        # shock-based overlays never engage (even if configured in filters).
        from ..engine import normalize_shock_detector, normalize_shock_gate_mode

        shock_mode = normalize_shock_gate_mode(filters)
        if shock_mode == "off":
            return base
        detector = normalize_shock_detector(filters)
        if detector not in ("daily_atr_pct", "daily_drawdown"):
            return base

        days_needed = None
        if detector == "daily_atr_pct":
            raw = filters.get("shock_daily_atr_period", 14)
            try:
                days_needed = int(raw or 14)
            except (TypeError, ValueError):
                days_needed = 14
            days_needed = max(1, int(days_needed))
        else:
            raw = filters.get("shock_drawdown_lookback_days", 20)
            try:
                days_needed = int(raw or 20)
            except (TypeError, ValueError):
                days_needed = 20
            days_needed = max(2, int(days_needed))

        # Map required daily lookback into an IB duration string.
        if days_needed <= 20:
            needed = "2 M"
        elif days_needed <= 45:
            needed = "3 M"
        elif days_needed <= 90:
            needed = "6 M"
        else:
            needed = "1 Y"
        return _max_duration(base, needed)

    async def _signal_fetch_bars(
        self,
        *,
        contract: Contract,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        now_ref: datetime,
    ) -> list | None:
        from ..utils.bar_utils import trim_incomplete_last_bar

        bars = await self._client.historical_bars_ohlcv(
            contract,
            duration_str=duration_str,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_ttl_sec=30.0,
        )
        if not bars:
            return None
        bars = trim_incomplete_last_bar(bars, bar_size=bar_size, now_ref=now_ref)
        return bars if bars else None

    def _signal_regime_spec(
        self,
        *,
        regime_mode_raw: str | None,
        regime_ema_preset_raw: str | None,
        regime_bar_size_raw: str | None,
        bar_size: str,
    ) -> tuple[str, str | None, str, bool]:
        return resolve_spot_regime_spec(
            bar_size=bar_size,
            regime_mode_raw=regime_mode_raw,
            regime_ema_preset_raw=regime_ema_preset_raw,
            regime_bar_size_raw=regime_bar_size_raw,
        )

    def _signal_regime2_spec(
        self,
        *,
        regime2_mode_raw: str | None,
        regime2_ema_preset_raw: str | None,
        regime2_bar_size_raw: str | None,
        bar_size: str,
    ) -> tuple[str, str | None, str, bool]:
        return resolve_spot_regime2_spec(
            bar_size=bar_size,
            regime2_mode_raw=regime2_mode_raw,
            regime2_ema_preset_raw=regime2_ema_preset_raw,
            regime2_bar_size_raw=regime2_bar_size_raw,
        )

    def _signal_regime_duration(
        self,
        *,
        regime_duration: str,
        regime_bar_size: str,
        filters: dict | None,
    ) -> str:
        if not isinstance(filters, dict) or "hour" not in str(regime_bar_size).strip().lower():
            return regime_duration
        shock_gate_mode = str(filters.get("shock_gate_mode") or "off").strip().lower()
        if shock_gate_mode in ("", "0", "false", "none", "null"):
            shock_gate_mode = "off"
        if shock_gate_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
            shock_gate_mode = "off"
        if shock_gate_mode == "off":
            return regime_duration
        try:
            atr_slow = int(filters.get("shock_atr_slow_period", 50))
        except (TypeError, ValueError):
            atr_slow = 50
        if atr_slow <= 0:
            return regime_duration
        alt = "1 M" if atr_slow <= 60 else "2 M"
        order = ("1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
        try:
            return alt if order.index(str(alt)) > order.index(str(regime_duration)) else regime_duration
        except ValueError:
            return regime_duration

    def _signal_strategy_payload(
        self,
        *,
        entry_signal: str,
        ema_preset_raw: str | None,
        entry_mode_raw: str | None,
        entry_confirm_bars: int,
        orb_window_mins_raw: int | None,
        orb_open_time_et_raw: str | None,
        spot_exit_mode_raw: str | None,
        spot_atr_period_raw: int | None,
        regime_mode: str,
        regime_preset: str | None,
        supertrend_atr_period_raw: int | None,
        supertrend_multiplier_raw: float | None,
        supertrend_source_raw: str | None,
        regime2_mode: str,
        regime2_preset: str | None,
        regime2_supertrend_atr_period_raw: int | None,
        regime2_supertrend_multiplier_raw: float | None,
        regime2_supertrend_source_raw: str | None,
    ) -> dict:
        return {
            "entry_signal": entry_signal,
            "ema_preset": ema_preset_raw,
            "ema_entry_mode": entry_mode_raw,
            "entry_confirm_bars": entry_confirm_bars,
            "orb_window_mins": orb_window_mins_raw,
            "orb_open_time_et": orb_open_time_et_raw,
            "spot_exit_mode": spot_exit_mode_raw,
            "spot_atr_period": spot_atr_period_raw,
            "regime_mode": regime_mode,
            "regime_ema_preset": regime_preset,
            "supertrend_atr_period": supertrend_atr_period_raw,
            "supertrend_multiplier": supertrend_multiplier_raw,
            "supertrend_source": supertrend_source_raw,
            "regime2_mode": regime2_mode,
            "regime2_ema_preset": regime2_preset,
            "regime2_supertrend_atr_period": regime2_supertrend_atr_period_raw,
            "regime2_supertrend_multiplier": regime2_supertrend_multiplier_raw,
            "regime2_supertrend_source": regime2_supertrend_source_raw,
        }

    def _signal_eval_last_snapshot(self, *, evaluator: object, bars: list) -> object | None:
        last_snap = None
        for idx, bar in enumerate(bars):
            next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
            is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
            evaluator.update_exec_bar(bar, is_last_bar=bool(is_last_bar))
            snap = evaluator.update_signal_bar(bar)
            if snap is not None:
                last_snap = snap
        return last_snap

    def _signal_snapshot_from_eval(self, snap: object) -> _SignalSnapshot:
        return _SignalSnapshot(
            bar_ts=snap.bar_ts,
            close=float(snap.close),
            signal=snap.signal,
            bars_in_day=int(snap.bars_in_day),
            rv=float(snap.rv) if snap.rv is not None else None,
            volume=float(snap.volume) if snap.volume is not None else None,
            volume_ema=float(snap.volume_ema) if snap.volume_ema is not None else None,
            volume_ema_ready=bool(snap.volume_ema_ready),
            shock=snap.shock,
            shock_dir=snap.shock_dir,
            shock_atr_pct=float(snap.shock_atr_pct) if snap.shock_atr_pct is not None else None,
            risk=snap.risk,
            atr=float(snap.atr) if snap.atr is not None else None,
            or_high=float(snap.or_high) if snap.or_high is not None else None,
            or_low=float(snap.or_low) if snap.or_low is not None else None,
            or_ready=bool(snap.or_ready),
        )

    async def _signal_snapshot_for_contract(
        self,
        *,
        contract: Contract,
        ema_preset_raw: str | None,
        bar_size: str,
        use_rth: bool,
        entry_signal_raw: str | None = None,
        orb_window_mins_raw: int | None = None,
        orb_open_time_et_raw: str | None = None,
        entry_mode_raw: str | None = None,
        entry_confirm_bars: int = 0,
        spot_exit_mode_raw: str | None = None,
        spot_atr_period_raw: int | None = None,
        regime_ema_preset_raw: str | None = None,
        regime_bar_size_raw: str | None = None,
        regime_mode_raw: str | None = None,
        supertrend_atr_period_raw: int | None = None,
        supertrend_multiplier_raw: float | None = None,
        supertrend_source_raw: str | None = None,
        regime2_ema_preset_raw: str | None = None,
        regime2_bar_size_raw: str | None = None,
        regime2_mode_raw: str | None = None,
        regime2_supertrend_atr_period_raw: int | None = None,
        regime2_supertrend_multiplier_raw: float | None = None,
        regime2_supertrend_source_raw: str | None = None,
        filters: dict | None = None,
    ) -> _SignalSnapshot | None:
        from ..spot_engine import SpotSignalEvaluator

        entry_signal = normalize_spot_entry_signal(entry_signal_raw)

        # IB intraday bars are timestamped in ET wall-clock for this flow; use ET here so
        # trim_incomplete_last_bar drops the in-progress bar instead of treating it as complete.
        now_ref = datetime.now(tz=ZoneInfo("America/New_York")).replace(tzinfo=None)
        bars = await self._signal_fetch_bars(
            contract=contract,
            duration_str=self._signal_duration_str(bar_size, filters=filters),
            bar_size=bar_size,
            use_rth=use_rth,
            now_ref=now_ref,
        )
        if not bars:
            return None

        regime_mode, regime_preset, regime_bar_size, use_mtf_regime = self._signal_regime_spec(
            regime_mode_raw=regime_mode_raw,
            regime_ema_preset_raw=regime_ema_preset_raw,
            regime_bar_size_raw=regime_bar_size_raw,
            bar_size=bar_size,
        )

        regime_bars = None
        if use_mtf_regime:
            regime_duration = self._signal_regime_duration(
                regime_duration=self._signal_duration_str(regime_bar_size, filters=filters),
                regime_bar_size=regime_bar_size,
                filters=filters,
            )
            regime_bars = await self._signal_fetch_bars(
                contract=contract,
                duration_str=regime_duration,
                bar_size=regime_bar_size,
                use_rth=use_rth,
                now_ref=now_ref,
            )
            if not regime_bars:
                return None

        regime2_mode, regime2_preset, regime2_bar_size, use_mtf_regime2 = self._signal_regime2_spec(
            regime2_mode_raw=regime2_mode_raw,
            regime2_ema_preset_raw=regime2_ema_preset_raw,
            regime2_bar_size_raw=regime2_bar_size_raw,
            bar_size=bar_size,
        )

        regime2_bars = None
        if regime2_mode != "off" and use_mtf_regime2:
            regime2_bars = await self._signal_fetch_bars(
                contract=contract,
                duration_str=self._signal_duration_str(regime2_bar_size, filters=filters),
                bar_size=regime2_bar_size,
                use_rth=use_rth,
                now_ref=now_ref,
            )
            if not regime2_bars:
                return None
        strategy = self._signal_strategy_payload(
            entry_signal=entry_signal,
            ema_preset_raw=ema_preset_raw,
            entry_mode_raw=entry_mode_raw,
            entry_confirm_bars=entry_confirm_bars,
            orb_window_mins_raw=orb_window_mins_raw,
            orb_open_time_et_raw=orb_open_time_et_raw,
            spot_exit_mode_raw=spot_exit_mode_raw,
            spot_atr_period_raw=spot_atr_period_raw,
            regime_mode=regime_mode,
            regime_preset=regime_preset,
            supertrend_atr_period_raw=supertrend_atr_period_raw,
            supertrend_multiplier_raw=supertrend_multiplier_raw,
            supertrend_source_raw=supertrend_source_raw,
            regime2_mode=regime2_mode,
            regime2_preset=regime2_preset,
            regime2_supertrend_atr_period_raw=regime2_supertrend_atr_period_raw,
            regime2_supertrend_multiplier_raw=regime2_supertrend_multiplier_raw,
            regime2_supertrend_source_raw=regime2_supertrend_source_raw,
        )

        evaluator = SpotSignalEvaluator(
            strategy=strategy,
            filters=filters,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
        )

        last_snap = self._signal_eval_last_snapshot(evaluator=evaluator, bars=bars)
        if last_snap is None or not bool(last_snap.signal.ema_ready):
            return None

        return self._signal_snapshot_from_eval(last_snap)

    def _reset_daily_counters_if_needed(self, instance: _BotInstance) -> None:
        today = datetime.now(tz=ZoneInfo("America/New_York")).date()
        if instance.entries_today_date != today:
            instance.entries_today_date = today
            instance.entries_today = 0

    def _entry_limit_ok(self, instance: _BotInstance) -> bool:
        self._reset_daily_counters_if_needed(instance)
        raw = instance.strategy.get("max_entries_per_day", 1)
        try:
            max_entries = int(raw)
        except (TypeError, ValueError):
            max_entries = 1
        if max_entries <= 0:
            return True
        return instance.entries_today < max_entries

    def _spot_open_position(self, *, symbol: str, sec_type: str, con_id: int = 0) -> PortfolioItem | None:
        sym = str(symbol or "").strip().upper()
        stype = str(sec_type or "STK").strip().upper() or "STK"
        desired_con_id = int(con_id or 0)
        best = None
        best_abs = 0.0
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType != stype:
                continue
            if str(getattr(contract, "symbol", "") or "").strip().upper() != sym:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if not pos:
                continue
            if desired_con_id:
                try:
                    if int(getattr(contract, "conId", 0) or 0) == desired_con_id:
                        return item
                except (TypeError, ValueError):
                    pass
            abs_pos = abs(pos)
            if abs_pos > best_abs:
                best_abs = abs_pos
                best = item
        return best

    def _options_open_positions(self, instance: _BotInstance) -> list[PortfolioItem]:
        if not instance.touched_conids:
            return []
        open_items: list[PortfolioItem] = []
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType not in ("OPT", "FOP"):
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id not in instance.touched_conids:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if pos:
                open_items.append(item)
        return open_items

    def _resolve_open_positions(
        self,
        instance: _BotInstance,
        *,
        symbol: str,
        signal_contract: Contract | None = None,
    ) -> tuple[str, list[PortfolioItem], str | None]:
        instrument = self._strategy_instrument(instance.strategy)
        if instrument == "spot":
            sec_type = str(getattr(signal_contract, "secType", "") or "").strip().upper()
            if not sec_type:
                sec_type = self._spot_sec_type(instance, symbol)
            con_id = int(getattr(signal_contract, "conId", 0) or 0) if signal_contract else 0
            item = self._spot_open_position(symbol=symbol, sec_type=sec_type, con_id=con_id)
            items = [item] if item is not None else []
            if not items:
                return instrument, items, None
            try:
                pos = float(getattr(items[0], "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            direction = "up" if pos > 0 else "down" if pos < 0 else None
            return instrument, items, direction

        items = self._options_open_positions(instance)
        direction = instance.open_direction or self._open_direction_from_positions(items)
        return instrument, items, direction

    def _options_position_values(self, items: list[PortfolioItem]) -> tuple[float | None, float | None]:
        if not items:
            return None, None
        cost_basis = 0.0
        market_value = 0.0
        for item in items:
            cost_basis += float(_cost_basis(item))
            mv = _safe_num(getattr(item, "marketValue", None))
            if mv is None:
                try:
                    mv = float(getattr(item, "position", 0.0) or 0.0) * float(getattr(item, "marketPrice", 0.0) or 0.0)
                    mv *= _infer_multiplier(item)
                except (TypeError, ValueError):
                    mv = 0.0
            market_value += float(mv)

        # Convert into the backtest sign convention: SELL credit = positive, BUY debit = negative.
        entry_value = -float(cost_basis)
        current_value = -float(market_value)
        return entry_value, current_value

    def _options_max_loss_estimate(self, items: list[PortfolioItem], *, spot: float) -> float | None:
        if not items:
            return None
        entry_value, _ = self._options_position_values(items)
        if entry_value is None:
            return None
        strikes: list[float] = []
        legs: list[tuple[str, str, float, int, float]] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            raw_right = str(getattr(contract, "right", "") or "").upper()
            right = "CALL" if raw_right in ("C", "CALL") else "PUT" if raw_right in ("P", "PUT") else ""
            if not right:
                continue
            try:
                strike = float(getattr(contract, "strike", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            qty = int(abs(pos))
            if qty <= 0:
                continue
            action = "BUY" if pos > 0 else "SELL"
            mult = _infer_multiplier(item)
            strikes.append(strike)
            legs.append((action, right, strike, qty, mult))
        if not strikes or not legs:
            return None
        strikes = sorted(set(strikes))
        high = max(float(spot), strikes[-1]) * 5.0
        candidates = [0.0] + strikes + [high]

        def _payoff(price: float) -> float:
            payoff = 0.0
            for action, right, strike, qty, mult in legs:
                if right == "CALL":
                    intrinsic = max(price - strike, 0.0)
                else:
                    intrinsic = max(strike - price, 0.0)
                sign = 1.0 if action == "BUY" else -1.0
                payoff += sign * intrinsic * float(qty) * float(mult)
            return payoff

        min_pnl = None
        for price in candidates:
            pnl = float(entry_value) + _payoff(float(price))
            if min_pnl is None or pnl < min_pnl:
                min_pnl = pnl
        if min_pnl is None:
            return None
        return max(0.0, -float(min_pnl))

    def _should_exit_on_dte(self, instance: _BotInstance, items: list[PortfolioItem], today: date) -> bool:
        raw_exit = instance.strategy.get("exit_dte", 0)
        try:
            exit_dte = int(raw_exit or 0)
        except (TypeError, ValueError):
            exit_dte = 0
        if exit_dte <= 0:
            return False
        raw_entry = instance.strategy.get("dte", 0)
        try:
            entry_dte = int(raw_entry or 0)
        except (TypeError, ValueError):
            entry_dte = 0
        if entry_dte > 0 and exit_dte >= entry_dte:
            return False

        expiries: list[date] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            exp = _contract_expiry_date(getattr(contract, "lastTradeDateOrContractMonth", None))
            if exp is not None:
                expiries.append(exp)
        if not expiries:
            return False
        remaining = min(business_days_until(today, exp) for exp in expiries)
        return remaining <= exit_dte

    def _open_direction_from_positions(self, items: list[PortfolioItem]) -> str | None:
        if not items:
            return None
        biggest_any = None
        biggest_any_abs = 0.0
        biggest_short = None
        biggest_short_abs = 0.0
        for item in items:
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            abs_pos = abs(pos)
            if abs_pos > biggest_any_abs:
                biggest_any_abs = abs_pos
                biggest_any = item
            if pos < 0 and abs_pos > biggest_short_abs:
                biggest_short_abs = abs_pos
                biggest_short = item

        chosen = biggest_short or biggest_any
        if chosen is None:
            return None
        contract = getattr(chosen, "contract", None)
        if not contract:
            return None
        right_char = str(getattr(contract, "right", "") or "").upper()
        right = "CALL" if right_char in ("C", "CALL") else "PUT" if right_char in ("P", "PUT") else ""
        try:
            pos = float(getattr(chosen, "position", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        action = "BUY" if pos > 0 else "SELL" if pos < 0 else ""
        return direction_from_action_right(action, right)

    def _should_exit_on_flip(
        self,
        instance: _BotInstance,
        snap: _SignalSnapshot,
        open_dir: str | None,
        open_items: list[PortfolioItem],
    ) -> bool:
        if not flip_exit_hit(
            exit_on_signal_flip=bool(instance.strategy.get("exit_on_signal_flip")),
            open_dir=open_dir,
            signal=snap.signal,
            flip_exit_mode_raw=instance.strategy.get("flip_exit_mode"),
            ema_entry_mode_raw=instance.strategy.get("ema_entry_mode"),
        ):
            return False

        hold_bars_raw = instance.strategy.get("flip_exit_min_hold_bars", 0)
        try:
            hold_bars = int(hold_bars_raw)
        except (TypeError, ValueError):
            hold_bars = 0
        if hold_bars > 0 and instance.last_entry_bar_ts is not None:
            bar_def = parse_bar_size(self._signal_bar_size(instance))
            if bar_def is not None:
                if (snap.bar_ts - instance.last_entry_bar_ts) < (bar_def.duration * hold_bars):
                    return False

        if bool(instance.strategy.get("flip_exit_only_if_profit")):
            pnl = 0.0
            for item in open_items:
                try:
                    pnl += float(getattr(item, "unrealizedPNL", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
            if pnl <= 0:
                return False
        return True

    def _entry_direction_for_instance(self, instance: _BotInstance, snap: _SignalSnapshot) -> str | None:
        entry_dir = snap.signal.entry_dir
        return str(entry_dir) if entry_dir in ("up", "down") else None

    def _allowed_entry_directions(self, instance: _BotInstance) -> set[str]:
        strategy = instance.strategy or {}
        instrument = self._strategy_instrument(strategy)
        if instrument == "spot":
            mapping = strategy.get("directional_spot") if isinstance(strategy.get("directional_spot"), dict) else None
            if mapping:
                allowed = set()
                for key in ("up", "down"):
                    leg = mapping.get(key)
                    if not isinstance(leg, dict):
                        continue
                    action = str(leg.get("action", "")).strip().upper()
                    if action in ("BUY", "SELL"):
                        allowed.add(key)
                return allowed
            return {"up"}

        if isinstance(strategy.get("directional_legs"), dict):
            allowed = {k for k in ("up", "down") if strategy["directional_legs"].get(k)}
            return allowed or {"up", "down"}

        legs = strategy.get("legs", [])
        if isinstance(legs, list) and legs:
            first = legs[0] if isinstance(legs[0], dict) else None
            if isinstance(first, dict):
                bias = direction_from_action_right(first.get("action", ""), first.get("right", ""))
                if bias in ("up", "down"):
                    return {bias}
        return {"up", "down"}

    def _submit_order(self) -> None:
        self._submit_selected_order()

    def _submit_selected_order(self) -> None:
        order = self._selected_order()
        if not order:
            self._set_status("Send: no order selected", render_bot=True)
            return
        if order.status != "STAGED":
            self._set_status(f"Send: already {order.status}", render_bot=True)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("Send: no loop", render_bot=True)
            return
        self._set_status("Sending order...", render_bot=True)
        self._send_task = loop.create_task(self._send_order(order))

    async def _send_order(self, order: _BotOrder) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        try:
            self._journal_write(event="SENDING", order=order, reason=order.reason, data=None)
            trade = await self._client.place_limit_order(
                order.order_contract,
                order.action,
                order.quantity,
                order.limit_price,
                outside_rth=order.order_contract.secType == "STK",
            )
            order_id = trade.order.orderId or trade.order.permId or 0
            order.status = "WORKING"
            order.order_id = int(order_id or 0) or None
            order.trade = trade
            order.sent_at = loop.time() if loop is not None else None
            self._set_status(f"Sent #{order_id} {order.action} {order.quantity} @ {order.limit_price:.2f}")
            self._journal_write(event="SENT", order=order, reason=order.reason, data=None)
        except Exception as exc:
            order.status = "ERROR"
            order.error = str(exc)
            self._set_status(f"Send error: {exc}")
            self._journal_write(event="SEND_ERROR", order=order, reason=order.reason, data={"exc": str(exc)})
        self._refresh_orders_table()
        self._render_bot()

    def _render_bot(self) -> None:
        lines: list[Text] = [Text("Bot Hub", style="bold")]
        if self._payload:
            symbol = self._payload.get("symbol", "?")
            start = self._payload.get("start", "?")
            end = self._payload.get("end", "?")
            bar_size = self._payload.get("bar_size", "?")
            lines.append(Text(f"Leaderboard: {symbol} {start}→{end} ({bar_size})", style="dim"))
        else:
            lines.append(Text("No leaderboard loaded", style="red"))

        lines.append(
            Text(
                "Enter=Config/Send  Ctrl+A=Presets  f=FilterDTE  w=FilterWin  v=Scope  Tab/h/l=Focus  c=Cancel  Space=Run/Toggle  s=Stop  S=Kill  d=Del  X=Send",
                style="dim",
            )
        )
        dte_label = "ALL" if self._filter_dte is None else str(self._filter_dte)
        win_label = (
            "ALL"
            if self._filter_min_win_rate is None
            else f"≥{int(self._filter_min_win_rate * 100)}%"
        )
        lines.append(Text(f"Filter: DTE={dte_label} (f)  Win={win_label} (w)", style="dim"))
        lines.append(
            Text(
                f"Focus: {self._active_panel}  Presets: {'ON' if self._presets_visible else 'OFF'}  "
                f"Scope: {'ALL' if self._scope_all else 'Instance'}  "
                f"Instances: {len(self._instances)}  Orders: {len(self._order_rows)}",
                style="dim",
            )
        )

        if self._active_panel == "presets":
            selected = self._selected_preset()
            if selected:
                lines.append(Text(""))
                lines.append(Text("Selected preset", style="bold"))
                lines.extend(_preset_lines(selected))
                eval_payload = self._group_eval_by_name.get(selected.group)
                if isinstance(eval_payload, dict):
                    windows = eval_payload.get("windows")
                    if isinstance(windows, list) and windows:
                        lines.append(Text(""))
                        lines.append(Text("Multiwindow (stability)", style="bold"))
                        for w in windows:
                            if not isinstance(w, dict):
                                continue
                            start = str(w.get("start") or "").strip()
                            end = str(w.get("end") or "").strip()
                            label = f"{start}→{end}" if start and end else (start or end or "window")
                            try:
                                roi = float(w.get("roi", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                roi = 0.0
                            try:
                                dd_pct = float(w.get("dd_pct", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                dd_pct = 0.0
                            try:
                                pnl = float(w.get("pnl", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                pnl = 0.0
                            pnl_mo: float | None = None
                            if start and end:
                                try:
                                    start_d = date.fromisoformat(start)
                                    end_d = date.fromisoformat(end)
                                except ValueError:
                                    start_d = None
                                    end_d = None
                                if start_d and end_d and end_d > start_d:
                                    months = max((end_d - start_d).days / 30.0, 1.0)
                                    pnl_mo = pnl / months
                            try:
                                trades = int(w.get("trades", 0) or 0)
                            except (TypeError, ValueError):
                                trades = 0
                            roi_text = Text(f"roi={roi*100:.1f}%", style="green" if roi > 0 else "")
                            dd_text = Text(f"dd={dd_pct*100:.1f}%", style="red" if dd_pct > 0 else "")
                            lines.append(
                                Text(f"{label}  ", style="dim")
                                + roi_text
                                + Text("  ", style="dim")
                                + dd_text
                                + Text(
                                    f"  tr={trades}  pnl={pnl:,.1f}"
                                    + (f"  pnl/mo={pnl_mo:,.0f}" if pnl_mo is not None else ""),
                                    style="dim",
                                )
                            )
        elif self._active_panel == "instances":
            instance = self._selected_instance()
            if instance:
                instrument = self._strategy_instrument(instance.strategy or {})
                if instrument == "spot":
                    sec_type = str((instance.strategy or {}).get("spot_sec_type") or "").strip().upper()
                    legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT-STK"
                    dte = "-"
                else:
                    legs_desc = _legs_label(instance.strategy.get("legs", []))
                    dte = instance.strategy.get("dte", "?")
                lines.append(Text(""))
                lines.append(Text(f"Selected instance #{instance.instance_id}", style="bold"))
                lines.append(Text(f"{instance.group}  DTE={dte}  Legs={legs_desc}", style="dim"))
                lines.append(Text(f"State={instance.state}", style="dim"))
        elif self._active_panel == "orders":
            order = self._selected_order()
            if order:
                lines.append(Text(""))
                lines.append(Text(f"Selected order (inst {order.instance_id})", style="bold"))
                lines.extend(_order_lines(order))

        if self._status:
            lines.append(Text(""))
            lines.append(Text(self._status, style="yellow"))

        self._status_panel.update(Text("\n").join(lines))

    def _add_order(self, order: _BotOrder) -> None:
        self._orders.append(order)
        instance = next((i for i in self._instances if i.instance_id == order.instance_id), None)
        self._journal_write(event="ORDER_STAGED", instance=instance, order=order, reason=order.reason, data=None)
        self._refresh_orders_table()
        if self._active_panel == "orders":
            self._orders_table.cursor_coordinate = (max(len(self._order_rows) - 1, 0), 0)

    def _refresh_orders_table(self) -> None:
        self._orders_table.clear()
        self._order_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            return
        for order in self._orders:
            if scope is not None and order.instance_id != scope:
                continue
            self._orders_table.add_row(*_order_row(order))
            self._order_rows.append(order)

        # Unify positions into the Orders table (so the bottom pane can be Logs).
        if self._scope_all:
            con_ids = set().union(*(i.touched_conids for i in self._instances))
        else:
            instance = next((i for i in self._instances if i.instance_id == scope), None)
            con_ids = set(instance.touched_conids) if instance else set()
        if not con_ids:
            return
        self._orders_table.add_row("", "", "", "", Text("POSITIONS", style="bold"), "", "", "", "", "")
        for item in self._positions:
            try:
                con_id = int(getattr(item.contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            if con_id not in con_ids:
                continue
            self._orders_table.add_row(*_position_as_order_row(item, scope=scope))


# endregion


# region UI Helpers
# region View/Text Helpers
def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


def _clean_group_label(raw: str) -> str:
    """Shorten leaderboard group names for table display.

    Many generated groups include full metrics in the name. Keep the identifier part so the table
    stays readable, and rely on the numeric columns + the Selected preset panel for details.
    """
    value = str(raw or "").strip()
    if not value:
        return value
    for token in (" roi/dd=", " pnl/dd=", " roi=", " pnl="):
        if token in value:
            return value.split(token, 1)[0].strip()
    return value


def _legs_label(legs: list[dict]) -> str:
    if not legs:
        return "-"
    parts = []
    for leg in legs:
        action = str(leg.get("action", "?"))[:1].upper()
        right = str(leg.get("right", "?"))[:1].upper()
        m = leg.get("moneyness_pct", 0.0)
        try:
            m = float(m)
        except (TypeError, ValueError):
            m = 0.0
        parts.append(f"{action}{right} {m:+.1f}%")
    return " + ".join(parts)


def _legs_direction_hint(legs: list[dict], ema_directional: bool) -> str:
    if ema_directional:
        return "EMA (up/down)"
    if not legs:
        return "ANY"
    first = legs[0] if isinstance(legs, list) else None
    if not isinstance(first, dict):
        return "ANY"
    action = str(first.get("action", "")).upper()
    right = str(first.get("right", "")).upper()
    if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "UP"
    if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "DOWN"
    return "ANY"


def _preset_lines(preset: _BotPreset) -> list[Text]:
    entry = preset.entry
    metrics = entry.get("metrics", {})
    strat = entry.get("strategy", {})
    instrument = str(strat.get("instrument", "options") or "options").strip().lower()

    legs_label = _legs_label(strat.get("legs", []))
    if instrument == "spot":
        mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
        if mapping and mapping.get("up") and mapping.get("down"):
            legs_label = "SPOT (up/down)"
        else:
            legs_label = "SPOT"

        signal_bar = str(strat.get("signal_bar_size") or "").strip()
        entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
        confirm = strat.get("entry_confirm_bars", 0)
        regime_mode = str(strat.get("regime_mode") or "ema").strip().lower()
        regime = str(strat.get("regime_ema_preset") or "").strip()
        regime_bar = str(strat.get("regime_bar_size") or "").strip()
        regime2_mode = str(strat.get("regime2_mode") or "off").strip().lower()
        regime2 = str(strat.get("regime2_ema_preset") or "").strip()
        regime2_bar = str(strat.get("regime2_bar_size") or "").strip()
        mode_parts: list[str] = []
        if entry_signal == "orb":
            window = strat.get("orb_window_mins", "?")
            rr = strat.get("orb_risk_reward", "?")
            tgt_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
            if tgt_mode not in ("rr", "or_range"):
                tgt_mode = "rr"
            mode_parts.append(f"ORB: {window}m {tgt_mode} rr={rr}")
        else:
            mode_parts.append(f"EMA: {strat.get('ema_preset', '')} cross c{confirm}")
        if regime_mode == "supertrend":
            atr_p = strat.get("supertrend_atr_period", "?")
            mult = strat.get("supertrend_multiplier", "?")
            src = str(strat.get("supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime: ST({atr_p},{mult},{src}) @ {regime_bar or '?'}")
        elif regime:
            mode_parts.append(f"Regime: {regime} @ {regime_bar or '?'}")
        if regime2_mode == "supertrend":
            atr_p = strat.get("regime2_supertrend_atr_period", "?")
            mult = strat.get("regime2_supertrend_multiplier", "?")
            src = str(strat.get("regime2_supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime2: ST({atr_p},{mult},{src}) @ {regime2_bar or '?'}")
        elif regime2_mode == "ema" and regime2:
            mode_parts.append(f"Regime2: {regime2} @ {regime2_bar or '?'}")
        exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
        if exit_mode == "atr":
            atr_p = strat.get("spot_atr_period", "?")
            pt_mult = strat.get("spot_pt_atr_mult", "?")
            sl_mult = strat.get("spot_sl_atr_mult", "?")
            mode_parts.append(f"Exit: ATR({atr_p}) PTx{pt_mult} SLx{sl_mult}")
        if signal_bar:
            mode_parts.append(f"Bar: {signal_bar}")
        mode = "  ".join(mode_parts)
    else:
        pt = float(strat.get("profit_target", 0.0)) * 100.0
        sl = float(strat.get("stop_loss", 0.0)) * 100.0
        mode = (
            f"DTE: {strat.get('dte', '?')}  "
            f"PT: {_fmt_pct(pt)}  SL: {_fmt_pct(sl)}  "
            f"EMA: {strat.get('ema_preset', '')}"
        )

    pnl = float(metrics.get("pnl", 0.0))
    try:
        dd = float(metrics.get("max_drawdown")) if metrics.get("max_drawdown") is not None else None
    except (TypeError, ValueError):
        dd = None
    pnl_over_dd = None
    try:
        pnl_over_dd = (
            float(metrics.get("pnl_over_dd"))
            if metrics.get("pnl_over_dd") is not None
            else (pnl / dd if dd and dd > 0 else None)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        pnl_over_dd = None
    lines = [
        Text(preset.group),
        Text(f"Legs: {legs_label}", style="dim"),
        Text(mode, style="dim"),
        Text(
            f"PnL: {pnl:.2f}  "
            f"Win: {float(metrics.get('win_rate', 0.0)) * 100.0:.1f}%  "
            f"Trades: {int(metrics.get('trades', 0))}",
            style="dim",
        ),
    ]
    if dd is not None:
        extra = f"DD: {dd:.2f}"
        if pnl_over_dd is not None:
            extra += f"  PnL/DD: {pnl_over_dd:.2f}"
        lines.append(Text(extra, style="dim"))
    return lines


def _order_lines(order: _BotOrder) -> list[Text]:
    legs = order.legs or []
    legs_line: str | None = None
    if len(legs) == 1 and order.order_contract.secType != "BAG":
        contract = legs[0].contract
        local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
        sec_type = getattr(contract, "secType", "") or ""
        if sec_type == "STK":
            header = f"{local} STK".strip()
        elif sec_type == "FUT":
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            header = f"{local} {expiry} FUT".strip()
        else:
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or "?"
            right = getattr(contract, "right", "") or "?"
            strike = getattr(contract, "strike", None)
            header = f"{local} {expiry}{right} {strike}"
    else:
        symbol = getattr(order.order_contract, "symbol", "") or getattr(
            order.underlying, "symbol", ""
        )
        header = f"{symbol} BAG ({len(legs)} legs)"
        legs_desc: list[str] = []
        for leg in legs[:4]:
            contract = leg.contract
            local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
            legs_desc.append(f"{leg.action[:1]}{local}".strip())
        if legs_desc:
            legs_line = "  ".join(legs_desc) + ("  …" if len(legs) > 4 else "")

    parts: list[Text] = [Text(header, style="dim")]
    if legs_line:
        parts.append(Text(f"Legs: {legs_line}", style="dim"))
    parts.append(Text(f"Side: {order.action}  Qty: {order.quantity}", style="dim"))
    parts.append(Text(f"Limit: {_fmt_quote(order.limit_price)}", style="dim"))
    parts.append(
        Text(
            f"Bid: {_fmt_quote(order.bid)}  Ask: {_fmt_quote(order.ask)}  "
            f"Last: {_fmt_quote(order.last)}",
            style="dim",
        )
    )
    return parts


# endregion


# region Config/Path Helpers
def _parse_entry_days(raw: str) -> list[str]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return []
    normalized = (
        cleaned.replace(";", ",")
        .replace("|", ",")
        .replace("/", ",")
        .replace("\\", ",")
        .replace(" ", ",")
    )
    out: list[str] = []
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        key = token.upper()[:3]
        if key in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"):
            out.append(key.title())
    return out


def _filters_for_group(payload: dict | None, group_name: str) -> dict | None:
    if not payload:
        return None
    for group in payload.get("groups", []):
        if str(group.get("name")) == group_name:
            return group.get("filters") or None
    return None


def _contract_expiry_date(raw: object) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        except ValueError:
            return None
    if len(text) >= 6 and text[:6].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), 1)
        except ValueError:
            return None
    return None


def _get_path(root: object, path: str) -> object:
    current: object = root
    for part in str(path).split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _set_path(root: object, path: str, value: object) -> None:
    if not isinstance(path, str) or not path:
        return
    parts = path.split(".")
    current = root
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        next_part = parts[idx + 1] if not last else ""

        if isinstance(current, dict):
            if last:
                current[part] = value
                return
            nxt = current.get(part)
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[part] = nxt
            current = nxt
            continue

        if isinstance(current, list):
            try:
                list_idx = int(part)
            except (TypeError, ValueError):
                return
            while len(current) <= list_idx:
                current.append({})
            if last:
                current[list_idx] = value
                return
            nxt = current[list_idx]
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[list_idx] = nxt
            current = nxt
            continue
        return


# endregion


# region Table Row Helpers
def _order_row(order: _BotOrder) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    ts = order.created_at.astimezone().strftime("%H:%M:%S")
    inst = str(order.instance_id)
    contract = order.order_contract
    if contract.secType == "BAG" or len(order.legs) > 1:
        symbol = getattr(contract, "symbol", "") or getattr(order.underlying, "symbol", "") or "?"
        local = f"{symbol} BAG {len(order.legs)}L"
    else:
        leg_contract = order.legs[0].contract if order.legs else contract
        local = getattr(leg_contract, "localSymbol", "") or getattr(leg_contract, "symbol", "") or "?"
    local = str(local)[:12]
    side = order.action[:1]
    qty = str(int(order.quantity))
    limit = _fmt_quote(order.limit_price)
    bid = _fmt_quote(order.bid)
    ask = _fmt_quote(order.ask)
    bid_ask = f"{bid}/{ask}"
    status = order.status
    if order.exec_mode and order.status in ("STAGED", "WORKING", "CANCELING"):
        status = f"{status} {order.exec_mode}"
    if order.order_id:
        status = f"{status} #{order.order_id}"
    if order.error and order.status == "ERROR":
        status = f"ERROR {order.error}"[:32]
    elif order.error and order.status == "WORKING":
        status = f"{status} !{order.error}"[:32]
    return (ts, inst, side, qty, local, limit, bid_ask, status, "", "")


def _position_as_order_row(
    item: PortfolioItem, *, scope: int | None
) -> tuple[str, str, str, str, str, str, str, str, Text, Text]:
    contract = getattr(item, "contract", None)
    sec_type = str(getattr(contract, "secType", "") or "") if contract is not None else ""
    symbol = str(getattr(contract, "symbol", "") or "") if contract is not None else ""

    local = ""
    if contract is not None:
        if sec_type == "STK":
            local = symbol
        elif sec_type == "FUT":
            local = str(getattr(contract, "localSymbol", "") or symbol or "?")
        elif sec_type in ("OPT", "FOP"):
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
            right = str(getattr(contract, "right", "") or "")
            strike = getattr(contract, "strike", None)
            strike_s = ""
            if strike is not None:
                try:
                    strike_s = f"{float(strike):.1f}"
                except (TypeError, ValueError):
                    strike_s = ""
            local = f"{symbol} {expiry}{right[:1]} {strike_s}".strip()
        else:
            local = str(getattr(contract, "localSymbol", "") or symbol or "?")
    local = str(local or "?")[:12]

    try:
        pos = float(getattr(item, "position", 0.0) or 0.0)
    except (TypeError, ValueError):
        pos = 0.0
    side = "L" if pos > 0 else "S" if pos < 0 else ""
    abs_pos = abs(pos)
    qty = f"{abs_pos:.2f}"
    if abs_pos.is_integer():
        qty = str(int(abs_pos))

    avg = _safe_num(getattr(item, "averageCost", None))
    mkt = _safe_num(getattr(item, "marketPrice", None))
    unreal = _safe_num(getattr(item, "unrealizedPNL", None))
    realized = _safe_num(getattr(item, "realizedPNL", None))

    unreal_cell = _pnl_text(unreal) if unreal is not None else Text("")
    realized_cell = _pnl_text(realized) if realized is not None else Text("")

    return (
        "POS",
        str(int(scope)) if scope is not None else "",
        side,
        qty,
        local,
        _fmt_quote(avg),
        _fmt_quote(mkt),
        "",
        unreal_cell,
        realized_cell,
    )


def _instance_pnl_cells(instance: _BotInstance, positions: list[PortfolioItem]) -> tuple[Text | str, Text | str]:
    con_ids = set(instance.touched_conids)
    if not con_ids:
        return "", ""
    has_match = False
    unreal_total = 0.0
    realized_total = 0.0
    for item in positions:
        try:
            con_id = int(getattr(item.contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            con_id = 0
        if con_id not in con_ids:
            continue
        has_match = True
        unreal = _safe_num(getattr(item, "unrealizedPNL", None))
        realized = _safe_num(getattr(item, "realizedPNL", None))
        if unreal is not None:
            unreal_total += float(unreal)
        if realized is not None:
            realized_total += float(realized)
    if not has_match:
        return "", ""
    return _pnl_text(unreal_total), _pnl_text(realized_total)


# endregion


# endregion
