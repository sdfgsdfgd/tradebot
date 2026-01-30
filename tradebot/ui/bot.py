"""Bot hub screen (presets + proposals + auto-trade)."""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import math
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import Bag, ComboLeg, Contract, Option, PortfolioItem, Stock, Ticker, Trade
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from ..engine import (
    EmaDecisionSnapshot,
    RiskOverlaySnapshot,
    cooldown_ok_by_time,
    flip_exit_hit,
    parse_time_hhmm,
    signal_filters_ok,
)
from ..signals import (
    direction_from_action_right,
    ema_state_direction,
    flip_exit_mode,
    parse_bar_size,
)
from .common import (
    _SECTION_TYPES,
    _market_session_label,
    _midpoint,
    _optimistic_price,
    _aggressive_price,
    _parse_float,
    _parse_int,
    _portfolio_row,
    _portfolio_sort_key,
    _pnl_text,
    _quote_status_line,
    _round_to_tick,
    _safe_num,
    _tick_size,
    _ticker_price,
    _trade_sort_key,
)

# region Bot Journal
_BOT_JOURNAL_FIELDS = (
    "ts_et",
    "ts_utc",
    "event",
    "instance_id",
    "group",
    "symbol",
    "instrument",
    "action",
    "qty",
    "limit_price",
    "order_id",
    "status",
    "reason",
    "data_json",
)
# endregion

# region Bot UI
@dataclass(frozen=True)
class _BotPreset:
    group: str
    entry: dict


@dataclass(frozen=True)
class _PresetHeader:
    node_id: str
    depth: int
    label: str


@dataclass
class _BotInstance:
    instance_id: int
    group: str
    symbol: str
    strategy: dict
    filters: dict | None
    metrics: dict | None = None
    auto_trade: bool = False
    state: str = "RUNNING"
    last_propose_date: date | None = None
    open_direction: str | None = None
    last_entry_bar_ts: datetime | None = None
    last_exit_bar_ts: datetime | None = None
    entries_today: int = 0
    entries_today_date: date | None = None
    error: str | None = None
    spot_profit_target_price: float | None = None
    spot_stop_loss_price: float | None = None
    touched_conids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _BotConfigResult:
    mode: str  # "create" or "update"
    instance_id: int | None
    group: str
    symbol: str
    strategy: dict
    filters: dict | None
    auto_trade: bool


@dataclass(frozen=True)
class _BotConfigField:
    label: str
    kind: str  # "int" | "float" | "bool" | "enum" | "text"
    path: str
    options: tuple[str, ...] = ()


@dataclass
class _BotLegOrder:
    contract: Contract
    action: str  # BUY/SELL
    ratio: int


@dataclass
class _BotProposal:
    instance_id: int
    preset: _BotPreset | None
    underlying: Contract
    order_contract: Contract
    legs: list[_BotLegOrder]
    action: str  # BUY/SELL (order action for order_contract)
    quantity: int  # combo quantity (BAG) or contracts (single-leg)
    limit_price: float
    created_at: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    status: str = "PROPOSED"
    order_id: int | None = None
    error: str | None = None
    intent: str | None = None
    direction: str | None = None
    reason: str | None = None
    signal_bar_ts: datetime | None = None
    journal: dict | None = None


@dataclass(frozen=True)
class _SignalSnapshot:
    bar_ts: datetime
    close: float
    signal: EmaDecisionSnapshot
    bars_in_day: int
    rv: float | None
    volume: float | None = None
    volume_ema: float | None = None
    volume_ema_ready: bool = True
    shock: bool | None = None
    shock_dir: str | None = None
    shock_atr_pct: float | None = None
    risk: RiskOverlaySnapshot | None = None
    atr: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    or_ready: bool = False


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
        auto_trade: bool,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._instance_id = instance_id
        self._group = group
        self._symbol = symbol
        self._strategy = copy.deepcopy(strategy)
        self._filters = copy.deepcopy(filters) if filters else None
        self._auto_trade = bool(auto_trade)

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
            _BotConfigField("Auto trade", "bool", "auto_trade"),
            _BotConfigField("Instrument", "enum", "instrument", options=("options", "spot")),
            _BotConfigField(
                "Signal bar size",
                "enum",
                "signal_bar_size",
                options=("1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
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
                options=("", "1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
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
                options=("", "1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
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
                _BotConfigField(
                    "Price mode",
                    "enum",
                    "price_mode",
                    options=("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"),
                ),
                _BotConfigField("Chase proposals", "bool", "chase_proposals"),
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
        if field.path == "auto_trade":
            return self._auto_trade
        if field.path == "symbol":
            return self._symbol
        if field.path.startswith("filters."):
            if not self._filters:
                return None
            return _get_path(self._filters, field.path[len("filters.") :])
        return _get_path(self._strategy, field.path)

    def _set_value(self, field: _BotConfigField, value: object) -> None:
        if field.path == "auto_trade":
            self._auto_trade = bool(value)
            return
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
            auto_trade=self._auto_trade,
        )
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BotScreen(Screen):
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
        ("a", "toggle_auto_trade", "Auto"),
        ("space", "context_space", "Run/Toggle"),
        ("s", "stop_bot", "Stop"),
        ("d", "delete_instance", "Del"),
        ("p", "propose", "Propose"),
        ("f", "cycle_dte_filter", "Filter"),
        ("w", "cycle_win_filter", "Win"),
        ("r", "reload", "Reload"),
    ]

    def __init__(self, client: IBKRClient, refresh_sec: float) -> None:
        super().__init__()
        self._client = client
        self._refresh_sec = max(refresh_sec, 0.25)
        base = Path(__file__).resolve().parents[1]
        self._leaderboard_path = base / "backtest" / "leaderboard.json"
        self._spot_milestones_path = base / "backtest" / "spot_milestones.json"
        self._spot_champions_path = base / "backtest" / "spot_champions.json"
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
        self._proposals: list[_BotProposal] = []
        self._proposal_rows: list[_BotProposal] = []
        self._positions: list[PortfolioItem] = []
        self._position_rows: list[PortfolioItem] = []
        self._status: str | None = None
        self._refresh_task = None
        self._proposal_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._tracked_conids: set[int] = set()
        self._active_panel = "presets"
        self._refresh_lock = asyncio.Lock()
        self._scope_all = False
        self._last_chase_ts = 0.0
        self._journal_lock = threading.Lock()
        self._journal_path = self._init_journal_path()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="bot-status"),
            DataTable(id="bot-presets", zebra_stripes=True),
            DataTable(id="bot-instances", zebra_stripes=True),
            DataTable(id="bot-proposals", zebra_stripes=True),
            DataTable(id="bot-positions", zebra_stripes=True),
            id="bot-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._presets_table = self.query_one("#bot-presets", DataTable)
        self._status_panel = self.query_one("#bot-status", Static)
        self._proposals_table = self.query_one("#bot-proposals", DataTable)
        self._instances_table = self.query_one("#bot-instances", DataTable)
        self._positions_table = self.query_one("#bot-positions", DataTable)
        self._presets_table.border_title = "Presets"
        self._instances_table.border_title = "Instances"
        self._proposals_table.border_title = "Orders / Proposals"
        self._positions_table.border_title = "Bot Positions"
        self._presets_table.cursor_type = "row"
        self._proposals_table.cursor_type = "row"
        self._instances_table.cursor_type = "row"
        self._positions_table.cursor_type = "row"
        self._setup_tables()
        self._load_leaderboard()
        self._presets_table.display = self._presets_visible
        await self._refresh_positions()
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._journal_write(event="BOOT", reason=None, data={"refresh_sec": self._refresh_sec})
        self._render_status()
        self._focus_panel("presets")
        self._refresh_task = self.set_interval(self._refresh_sec, self._on_refresh_tick)

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        for con_id in list(self._tracked_conids):
            self._client.release_ticker(con_id)
        self._tracked_conids.clear()
        self._journal_write(event="SHUTDOWN", reason=None, data=None)

    # region Journal
    def _init_journal_path(self) -> Path | None:
        out_dir = Path(__file__).resolve().parent / "out"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        started_at = datetime.now(tz=ZoneInfo("America/New_York"))
        return out_dir / f"bot_journal_{started_at:%Y%m%d_%H%M%S_ET}.csv"

    def _journal_write(
        self,
        *,
        event: str,
        instance: _BotInstance | None = None,
        proposal: _BotProposal | None = None,
        reason: str | None,
        data: dict | None,
    ) -> None:
        path = self._journal_path
        if path is None:
            return

        try:
            now_et = datetime.now(tz=ZoneInfo("America/New_York"))
            now_utc = datetime.now(tz=timezone.utc)

            row: dict[str, object] = {k: "" for k in _BOT_JOURNAL_FIELDS}
            row["ts_et"] = now_et.isoformat()
            row["ts_utc"] = now_utc.isoformat()
            row["event"] = str(event or "")
            row["reason"] = str(reason) if reason else ""

            if instance is not None:
                row["instance_id"] = str(int(instance.instance_id))
                row["group"] = str(instance.group or "")
                row["symbol"] = str(instance.symbol or "")
                try:
                    row["instrument"] = str(self._strategy_instrument(instance.strategy or {}))
                except Exception:
                    row["instrument"] = ""
            if proposal is not None:
                row["instance_id"] = str(int(proposal.instance_id))
                row["action"] = str(proposal.action or "")
                row["qty"] = str(int(proposal.quantity or 0))
                row["limit_price"] = f"{float(proposal.limit_price):.6f}"
                row["status"] = str(proposal.status or "")
                row["order_id"] = str(int(proposal.order_id)) if proposal.order_id else ""

            extra: dict[str, object] = {}
            if instance is not None:
                extra["strategy"] = instance.strategy
                extra["filters"] = instance.filters
            if proposal is not None:
                extra["intent"] = proposal.intent
                extra["direction"] = proposal.direction
                extra["signal_bar_ts"] = proposal.signal_bar_ts.isoformat() if proposal.signal_bar_ts else None
                extra["proposal_journal"] = proposal.journal
                extra["error"] = proposal.error
            if isinstance(data, dict) and data:
                extra.update(data)
            row["data_json"] = json.dumps(extra, sort_keys=True, default=str) if extra else ""

            try:
                is_new = (not path.exists()) or path.stat().st_size == 0
            except Exception:
                is_new = True
            with self._journal_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=_BOT_JOURNAL_FIELDS)
                    if is_new:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception:
            return

    # endregion

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; hook it so Enter arms/sends.
        table_id = getattr(event.control, "id", None)
        if table_id == "bot-presets":
            self._active_panel = "presets"
        elif table_id == "bot-instances":
            self._active_panel = "instances"
        elif table_id == "bot-proposals":
            self._active_panel = "proposals"
        elif table_id == "bot-positions":
            self._active_panel = "positions"
        self.action_activate()

    def on_key(self, event: events.Key) -> None:
        if event.character == "X":
            self._submit_selected_proposal()
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
        self._status = "Reloaded leaderboard"
        self._journal_write(event="RELOAD_LEADERBOARD", reason=None, data=None)
        self._render_status()

    def action_toggle_presets(self) -> None:
        self._presets_visible = not self._presets_visible
        self._presets_table.display = self._presets_visible
        if not self._presets_visible and self._active_panel == "presets":
            self._focus_panel("instances")
        elif self._presets_visible and self._active_panel != "presets":
            self._focus_panel("presets")
        self._status = f"Presets: {'ON' if self._presets_visible else 'OFF'}"
        self._render_status()
        self.refresh(layout=True)

    def action_toggle_scope(self) -> None:
        self._scope_all = not self._scope_all
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = "Scope: ALL" if self._scope_all else "Scope: Instance"
        self._journal_write(
            event="SCOPE",
            reason=None,
            data={"scope": "ALL" if self._scope_all else "INSTANCE"},
        )
        self._render_status()
        self.refresh(layout=True)

    def action_cycle_focus(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def _cycle_focus(self, direction: int) -> None:
        panels = ["presets", "instances", "proposals", "positions"]
        if not self._presets_visible:
            panels.remove("presets")
        try:
            idx = panels.index(self._active_panel)
        except ValueError:
            idx = 0
        self._focus_panel(panels[(idx + direction) % len(panels)])

    def _focus_panel(self, panel: str) -> None:
        self._active_panel = panel
        if panel == "presets":
            self._presets_table.focus()
        elif panel == "instances":
            self._instances_table.focus()
        elif panel == "proposals":
            self._proposals_table.focus()
        else:
            self._positions_table.focus()
        self._render_status()

    def action_toggle_auto_trade(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Auto: select an instance"
            self._render_status()
            return
        instance.auto_trade = not instance.auto_trade
        self._refresh_instances_table()
        self._status = f"Instance {instance.instance_id}: auto trade {'ON' if instance.auto_trade else 'OFF'}"
        self._journal_write(
            event="AUTO_TOGGLE",
            instance=instance,
            reason=None,
            data={"auto_trade": bool(instance.auto_trade)},
        )
        self._render_status()

    def action_toggle_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Run: select an instance"
            self._render_status()
            return
        instance.state = "PAUSED" if instance.state == "RUNNING" else "RUNNING"
        self._refresh_instances_table()
        self._status = f"Instance {instance.instance_id}: {instance.state}"
        self._render_status()

    def action_context_space(self) -> None:
        if self._active_panel == "presets":
            self.action_activate()
            return
        self.action_toggle_instance()

    def action_delete_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Del: select an instance"
            self._render_status()
            return
        self._instances = [i for i in self._instances if i.instance_id != instance.instance_id]
        self._proposals = [p for p in self._proposals if p.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._status = f"Deleted instance {instance.instance_id}"
        self._render_status()

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
        self._status = (
            "Filter: DTE=ALL" if self._filter_dte is None else f"Filter: DTE={self._filter_dte}"
        )
        self._render_status()
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
            self._status = "Filter: Win=ALL"
        else:
            self._status = f"Filter: Win≥{int(self._filter_min_win_rate * 100)}%"
        self._render_status()
        self.refresh(layout=True)

    def action_stop_bot(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Stop: select an instance"
            self._render_status()
            return
        instance.state = "PAUSED"
        instance.auto_trade = False
        self._proposals = [p for p in self._proposals if p.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = f"Stopped instance {instance.instance_id}: paused + auto OFF + cleared proposals"
        self._journal_write(event="STOP_INSTANCE", instance=instance, reason=None, data=None)
        self._render_status()

    def _kill_all(self) -> None:
        for instance in self._instances:
            instance.state = "PAUSED"
            instance.auto_trade = False
        self._proposals.clear()
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = "KILL: paused all + auto OFF + cleared proposals"
        self._journal_write(event="KILL_ALL", reason=None, data=None)
        self._render_status()

    def action_activate(self) -> None:
        if self._active_panel == "presets":
            row = self._selected_preset_row()
            if row is None:
                self._status = "Preset: none selected"
                self._render_status()
                return
            if isinstance(row, _PresetHeader):
                self._toggle_preset_node(row.node_id)
                return
            self._open_config_for_preset(row)
            return
        if self._active_panel == "instances":
            instance = self._selected_instance()
            if not instance:
                self._status = "Instance: none selected"
                self._render_status()
                return
            self._open_config_for_instance(instance)
            return
        if self._active_panel == "proposals":
            self._submit_selected_proposal()
            return
        self._status = "Positions: no action"
        self._render_status()

    def action_propose(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Propose: select an instance"
            self._render_status()
            return
        self._queue_proposal(instance, intent="enter", direction=None, signal_bar_ts=None)

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

    def _selected_proposal(self) -> _BotProposal | None:
        row = self._proposals_table.cursor_coordinate.row
        if row < 0 or row >= len(self._proposal_rows):
            return None
        return self._proposal_rows[row]

    def _scope_instance_id(self) -> int | None:
        if self._scope_all:
            return None
        instance = self._selected_instance()
        return instance.instance_id if instance else None

    def _open_config_for_preset(self, preset: _BotPreset) -> None:
        entry = preset.entry
        strategy = copy.deepcopy(entry.get("strategy", {}) or {})
        strategy.setdefault("instrument", "options")
        strategy.setdefault("price_mode", "MID")
        strategy.setdefault("chase_proposals", True)
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
                self._status = "Config: cancelled"
                self._render_status()
                return
            instance = _BotInstance(
                instance_id=self._next_instance_id,
                group=result.group,
                symbol=result.symbol,
                strategy=result.strategy,
                filters=result.filters,
                metrics=entry.get("metrics"),
                auto_trade=result.auto_trade,
            )
            self._next_instance_id += 1
            self._instances.append(instance)
            self._refresh_instances_table()
            self._instances_table.cursor_coordinate = (max(len(self._instances) - 1, 0), 0)
            self._focus_panel("instances")
            self._status = f"Created instance {instance.instance_id}"
            self._journal_write(
                event="INSTANCE_CREATED",
                instance=instance,
                reason=None,
                data={"metrics": entry.get("metrics")},
            )
            self._render_status()

        self.app.push_screen(
            BotConfigScreen(
                mode="create",
                instance_id=None,
                group=preset.group,
                symbol=symbol,
                strategy=strategy,
                filters=filters,
                auto_trade=False,
            ),
            _on_done,
        )

    def _open_config_for_instance(self, instance: _BotInstance) -> None:
        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._status = "Config: cancelled"
                self._render_status()
                return
            instance.group = result.group
            instance.symbol = result.symbol
            instance.strategy = result.strategy
            instance.filters = result.filters
            instance.auto_trade = result.auto_trade
            self._refresh_instances_table()
            self._status = f"Updated instance {instance.instance_id}"
            self._journal_write(event="INSTANCE_UPDATED", instance=instance, reason=None, data=None)
            self._render_status()

        self.app.push_screen(
            BotConfigScreen(
                mode="update",
                instance_id=instance.instance_id,
                group=instance.group,
                symbol=instance.symbol,
                strategy=instance.strategy,
                filters=instance.filters,
                auto_trade=instance.auto_trade,
            ),
            _on_done,
        )

    def _load_leaderboard(self) -> None:
        payload: dict = {}
        try:
            payload = json.loads(self._leaderboard_path.read_text())
        except Exception as exc:
            self._status = f"Leaderboard load failed: {exc}"
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        groups: list[dict] = []
        base_groups = payload.get("groups", [])
        if isinstance(base_groups, list):
            for group in base_groups:
                if not isinstance(group, dict):
                    continue
                group["_source"] = "options_leaderboard"
                groups.append(group)

        self._spot_champ_version = None
        if self._spot_champions_path.exists():
            try:
                champ_payload = json.loads(self._spot_champions_path.read_text())
            except Exception:
                champ_payload = None
            if isinstance(champ_payload, dict):
                source = str(champ_payload.get("source") or "").strip()
                if source:
                    token = source.split("v", 1)[1] if "v" in source else ""
                    v = "".join(ch for ch in token if ch.isdigit())
                    self._spot_champ_version = v or None

                champ_groups = champ_payload.get("groups", [])
                if isinstance(champ_groups, list) and champ_groups:
                    ranked: list[tuple[float, dict]] = []
                    for group in champ_groups:
                        if not isinstance(group, dict):
                            continue
                        entries = group.get("entries") if isinstance(group.get("entries"), list) else []
                        entry = entries[0] if entries else None
                        if not isinstance(entry, dict):
                            continue
                        metrics = entry.get("metrics", {})
                        try:
                            score = float(metrics.get("pnl_over_dd"))
                        except (TypeError, ValueError):
                            score = float("-inf")
                        ranked.append((score, group))
                    ranked.sort(key=lambda pair: pair[0], reverse=True)
                    for _, group in ranked[:10]:
                        group["_source"] = "spot_champions"
                        groups.append(group)

        if self._spot_milestones_path.exists():
            try:
                spot_payload = json.loads(self._spot_milestones_path.read_text())
            except Exception:
                spot_payload = None
            if isinstance(spot_payload, dict):
                spot_groups = spot_payload.get("groups", [])
                if isinstance(spot_groups, list) and spot_groups:
                    for group in spot_groups:
                        if not isinstance(group, dict):
                            continue
                        group["_source"] = "spot_milestones"
                        groups.append(group)

        slv_dir = Path(__file__).resolve().parents[2] / "backtests" / "slv"
        if slv_dir.exists() and slv_dir.is_dir():
            for path in sorted(slv_dir.glob("*.json")):
                try:
                    slv_payload = json.loads(path.read_text())
                except Exception:
                    continue
                if not isinstance(slv_payload, dict):
                    continue
                slv_groups = slv_payload.get("groups", [])
                if not isinstance(slv_groups, list) or not slv_groups:
                    continue
                source = f"slv_research:{path.name}"
                for group in slv_groups:
                    if not isinstance(group, dict):
                        continue
                    group["_source"] = source
                    groups.append(group)

        payload["groups"] = groups
        self._payload = payload
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
        self._presets_table.add_column("Legs", width=18)
        self._presets_table.add_column("TF/DTE", width=8)
        self._presets_table.add_column("TP", width=7)
        self._presets_table.add_column("SL", width=7)
        self._presets_table.add_column("EMA", width=6)
        self._presets_table.add_column("PnL", width=12)
        self._presets_table.add_column("DD", width=12)
        self._presets_table.add_column("P/DD", width=6)
        self._presets_table.add_column("Win/Tr", width=9)

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

        def _fmt_pnl(value: float | None) -> Text:
            return _pnl_text(value)

        def _fmt_dd(value: float | None) -> Text:
            if value is None:
                return Text("")
            return Text(f"{value:,.2f}", style="red")

        def _fmt_ratio(value: float | None) -> str:
            if value is None:
                return ""
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return ""

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
            if source == "spot_champions" and self._spot_champ_version:
                base = f"v{self._spot_champ_version} {base}".strip()
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

        def _add_leaf(item: dict, *, depth: int) -> None:
            preset = item["preset"]
            strat = item["strategy"]
            metrics = item["metrics"]
            instrument = item["instrument"]

            if instrument == "spot":
                sec_type = str(strat.get("spot_sec_type") or "").strip().upper()
                legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT"
                tf_dte = str(item["tf"])
                tp_s, sl_s = _spot_tp_sl(strat)
            else:
                legs_desc = _legs_label(strat.get("legs", []))
                tf_dte = str(int(item["dte"]))
                tp_s, sl_s = _options_tp_sl(strat)

            legs_desc = legs_desc[:18]
            ema = str(strat.get("ema_preset", ""))[:6]

            pnl = None
            try:
                pnl = float(metrics.get("pnl")) if metrics.get("pnl") is not None else None
            except (TypeError, ValueError):
                pnl = None
            dd = _get_dd(metrics)
            p_dd = None
            try:
                p_dd = float(metrics.get("pnl_over_dd")) if metrics.get("pnl_over_dd") is not None else None
            except (TypeError, ValueError):
                p_dd = None
            if p_dd is None and pnl is not None and dd and dd > 0:
                p_dd = pnl / dd

            try:
                trades = int(metrics.get("trades", 0) or 0)
            except (TypeError, ValueError):
                trades = 0
            try:
                win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                win_rate = 0.0
            win_tr = f"{int(win_rate * 100):d}%/{trades}"

            label = f"{'  ' * depth}{item['name']}".rstrip()
            self._presets.append(preset)
            self._preset_rows.append(preset)
            self._presets_table.add_row(
                label,
                legs_desc,
                tf_dte,
                tp_s,
                sl_s,
                ema,
                _fmt_pnl(pnl),
                _fmt_dd(dd),
                _fmt_ratio(p_dd),
                win_tr,
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
            pnl = None
            dd = None
            p_dd = None
            win_tr = ""

            if best is not None:
                legs_cell = f"best: {best['name']}"[:18]
                strat = best["strategy"]
                instrument = best["instrument"]
                if instrument == "spot":
                    tf_dte = str(best["tf"])
                    tp_s, sl_s = _spot_tp_sl(strat)
                else:
                    tf_dte = str(int(best["dte"]))
                    tp_s, sl_s = _options_tp_sl(strat)
                ema = str(strat.get("ema_preset", ""))[:6]
                metrics = best["metrics"]
                try:
                    pnl = float(metrics.get("pnl")) if metrics.get("pnl") is not None else None
                except (TypeError, ValueError):
                    pnl = None
                dd = _get_dd(metrics)
                try:
                    p_dd = float(metrics.get("pnl_over_dd")) if metrics.get("pnl_over_dd") is not None else None
                except (TypeError, ValueError):
                    p_dd = None
                if p_dd is None and pnl is not None and dd and dd > 0:
                    p_dd = pnl / dd
                try:
                    trades = int(metrics.get("trades", 0) or 0)
                except (TypeError, ValueError):
                    trades = 0
                try:
                    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
                except (TypeError, ValueError):
                    win_rate = 0.0
                win_tr = f"{int(win_rate * 100):d}%/{trades}"

            self._preset_rows.append(_PresetHeader(node_id=node_id, depth=depth, label=label))
            self._presets_table.add_row(
                Text(left, style="bold"),
                Text(legs_cell, style="dim") if legs_cell else Text(""),
                tf_dte,
                tp_s,
                sl_s,
                ema,
                _fmt_pnl(pnl),
                _fmt_dd(dd),
                _fmt_ratio(p_dd),
                win_tr,
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
            contract_expanded = _add_header(contract_node, depth=0, label=symbol, best=_best(all_items))
            if not contract_expanded:
                continue

            spot_node = f"{contract_node}|spot"
            spot_items = [it for tf_items in bucket["spot"].values() for it in tf_items]
            spot_expanded = _add_header(spot_node, depth=1, label=f"{symbol} - Spot", best=_best(spot_items))
            if spot_expanded:
                for tf in sorted(bucket["spot"].keys()):
                    tf_items = bucket["spot"][tf]
                    tf_node = f"{spot_node}|tf:{tf}"
                    tf_expanded = _add_header(tf_node, depth=2, label=str(tf), best=_best(tf_items))
                    if tf_expanded:
                        ordered = sorted(
                            tf_items,
                            key=lambda it: (
                                0 if it["source"] == "spot_champions" else 1,
                                -float(it.get("score", float("-inf"))),
                                it["name"],
                            ),
                        )
                        for item in ordered:
                            _add_leaf(item, depth=3)

            opt_node = f"{contract_node}|options"
            opt_items = [it for items in bucket["options"].values() for it in items]
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
        self._instances_table.add_columns("ID", "Strategy", "Legs", "DTE", "Auto", "State", "BT PnL")

        self._proposals_table.clear(columns=True)
        self._proposals_table.add_columns("When", "Inst", "Side", "Qty", "Contract", "Lmt", "B/A", "Status")

        self._positions_table.clear(columns=True)
        self._positions_table.add_columns(
            "Symbol",
            "Expiry",
            "Right",
            "Strike",
            "Qty",
            "AvgCost",
            "Px 24-72",
            "Unreal",
            "Realized",
            "U 24-72",
        )

    def _cursor_move(self, direction: int) -> None:
        if self._active_panel == "presets":
            if direction > 0:
                self._presets_table.action_cursor_down()
            else:
                self._presets_table.action_cursor_up()
        elif self._active_panel == "instances":
            if direction > 0:
                self._instances_table.action_cursor_down()
            else:
                self._instances_table.action_cursor_up()
        elif self._active_panel == "proposals":
            if direction > 0:
                self._proposals_table.action_cursor_down()
            else:
                self._proposals_table.action_cursor_up()
        else:
            if direction > 0:
                self._positions_table.action_cursor_down()
            else:
                self._positions_table.action_cursor_up()
        if self._active_panel == "instances" and not self._scope_all:
            self._refresh_proposals_table()
            self._refresh_positions_table()
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
                sec_type = str((instance.strategy or {}).get("spot_sec_type") or "").strip().upper()
                legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT-STK"
                dte = "-"
            else:
                legs_desc = _legs_label(instance.strategy.get("legs", []))
                dte = instance.strategy.get("dte", "")
            auto = "ON" if instance.auto_trade else "OFF"
            bt_pnl = ""
            if instance.metrics:
                try:
                    bt_pnl = f"{float(instance.metrics.get('pnl', 0.0)):.0f}"
                except (TypeError, ValueError):
                    bt_pnl = ""
            self._instances_table.add_row(
                str(instance.instance_id),
                instance.group[:24],
                legs_desc[:24],
                str(dte),
                auto,
                instance.state,
                bt_pnl,
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
            self._refresh_proposals_table()
            self._refresh_positions_table()

    async def _refresh_positions(self) -> None:
        try:
            items = await self._client.fetch_portfolio()
        except Exception as exc:  # pragma: no cover - UI surface
            self._status = f"Positions error: {exc}"
            return
        self._positions = [item for item in items if item.contract.secType in _SECTION_TYPES]
        self._positions.sort(key=_portfolio_sort_key, reverse=True)
        self._refresh_positions_table()

    def _refresh_positions_table(self) -> None:
        self._positions_table.clear()
        self._position_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            return
        if self._scope_all:
            con_ids = set().union(*(i.touched_conids for i in self._instances))
        else:
            instance = next((i for i in self._instances if i.instance_id == scope), None)
            con_ids = set(instance.touched_conids) if instance else set()
        if not con_ids:
            return
        for item in self._positions:
            con_id = int(getattr(item.contract, "conId", 0) or 0)
            if con_id not in con_ids:
                continue
            self._positions_table.add_row(*_portfolio_row(item, Text(""), Text("")))
            self._position_rows.append(item)

    def _render_status(self) -> None:
        self._render_bot()

    async def _on_refresh_tick(self) -> None:
        if self._refresh_lock.locked():
            return
        async with self._refresh_lock:
            await self._refresh_positions()
            await self._auto_propose_tick()
            await self._chase_proposals_tick()
            self._auto_send_tick()
            self._render_status()

    async def _auto_propose_tick(self) -> None:
        if self._proposal_task and not self._proposal_task.done():
            return
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))

        for instance in self._instances:
            if instance.state != "RUNNING":
                continue
            if not self._can_propose_now(instance):
                continue

            pending = any(
                p.status == "PROPOSED" and p.instance_id == instance.instance_id for p in self._proposals
            )
            if pending:
                continue

            symbol = str(
                instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
            ).strip().upper()

            entry_signal = str(instance.strategy.get("entry_signal") or "ema").strip().lower()
            if entry_signal not in ("ema", "orb"):
                entry_signal = "ema"
            ema_preset = instance.strategy.get("ema_preset")
            if entry_signal == "ema" and not ema_preset:
                continue

            signal_contract = await self._signal_contract(instance, symbol)
            if signal_contract is None:
                continue

            snap = await self._signal_snapshot_for_contract(
                contract=signal_contract,
                ema_preset_raw=str(ema_preset) if ema_preset else None,
                bar_size=self._signal_bar_size(instance),
                use_rth=self._signal_use_rth(instance),
                entry_signal_raw=entry_signal,
                entry_mode_raw=instance.strategy.get("ema_entry_mode"),
                entry_confirm_bars=instance.strategy.get("entry_confirm_bars", 0),
                orb_window_mins_raw=instance.strategy.get("orb_window_mins"),
                orb_open_time_et_raw=instance.strategy.get("orb_open_time_et"),
                regime_ema_preset_raw=instance.strategy.get("regime_ema_preset"),
                regime_bar_size_raw=instance.strategy.get("regime_bar_size"),
                regime_mode_raw=instance.strategy.get("regime_mode"),
                supertrend_atr_period_raw=instance.strategy.get("supertrend_atr_period"),
                supertrend_multiplier_raw=instance.strategy.get("supertrend_multiplier"),
                supertrend_source_raw=instance.strategy.get("supertrend_source"),
                regime2_ema_preset_raw=instance.strategy.get("regime2_ema_preset"),
                regime2_bar_size_raw=instance.strategy.get("regime2_bar_size"),
                regime2_mode_raw=instance.strategy.get("regime2_mode"),
                regime2_supertrend_atr_period_raw=instance.strategy.get("regime2_supertrend_atr_period"),
                regime2_supertrend_multiplier_raw=instance.strategy.get("regime2_supertrend_multiplier"),
                regime2_supertrend_source_raw=instance.strategy.get("regime2_supertrend_source"),
                filters=instance.filters if isinstance(instance.filters, dict) else None,
                spot_exit_mode_raw=instance.strategy.get("spot_exit_mode"),
                spot_atr_period_raw=instance.strategy.get("spot_atr_period"),
            )
            if snap is None:
                continue

            entry_days = instance.strategy.get("entry_days", [])
            if entry_days:
                allowed_days = {_weekday_num(day) for day in entry_days}
            else:
                allowed_days = {0, 1, 2, 3, 4}
            if snap.bar_ts.weekday() not in allowed_days:
                continue

            cooldown_bars = 0
            if isinstance(instance.filters, dict):
                raw = instance.filters.get("cooldown_bars", 0)
                try:
                    cooldown_bars = int(raw or 0)
                except (TypeError, ValueError):
                    cooldown_bars = 0
            cooldown_ok = cooldown_ok_by_time(
                current_bar_ts=snap.bar_ts,
                last_entry_bar_ts=instance.last_entry_bar_ts,
                bar_size=self._signal_bar_size(instance),
                cooldown_bars=cooldown_bars,
            )
            if not signal_filters_ok(
                instance.filters,
                bar_ts=snap.bar_ts,
                bars_in_day=snap.bars_in_day,
                close=float(snap.close),
                volume=snap.volume,
                volume_ema=snap.volume_ema,
                volume_ema_ready=snap.volume_ema_ready,
                rv=snap.rv,
                signal=snap.signal,
                cooldown_ok=cooldown_ok,
                shock=snap.shock,
                shock_dir=snap.shock_dir,
            ):
                continue

            instrument = self._strategy_instrument(instance.strategy)
            open_items: list[PortfolioItem] = []
            open_dir: str | None = None
            if instrument == "spot":
                open_item = self._spot_open_position(
                    symbol=symbol,
                    sec_type=str(getattr(signal_contract, "secType", "") or "STK"),
                    con_id=int(getattr(signal_contract, "conId", 0) or 0),
                )
                if open_item is not None:
                    open_items = [open_item]
                    try:
                        pos = float(getattr(open_item, "position", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pos = 0.0
                    open_dir = "up" if pos > 0 else "down" if pos < 0 else None
            else:
                open_items = self._options_open_positions(instance)
                open_dir = instance.open_direction or self._open_direction_from_positions(open_items)

            if not open_items and instance.open_direction is not None:
                instance.open_direction = None
                instance.spot_profit_target_price = None
                instance.spot_stop_loss_price = None

            if open_items:
                if instance.last_exit_bar_ts is not None and instance.last_exit_bar_ts == snap.bar_ts:
                    continue

                if instrument == "spot":
                    open_item = open_items[0]
                    try:
                        pos = float(getattr(open_item, "position", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pos = 0.0
                    avg_cost = _safe_num(getattr(open_item, "averageCost", None))
                    market_price = _safe_num(getattr(open_item, "marketPrice", None))
                    if market_price is None:
                        ticker = await self._client.ensure_ticker(open_item.contract)
                        market_price = _ticker_price(ticker)

                    target_price = instance.spot_profit_target_price
                    stop_price = instance.spot_stop_loss_price
                    if (
                        pos
                        and market_price is not None
                        and market_price > 0
                        and (target_price is not None or stop_price is not None)
                    ):
                        try:
                            mp = float(market_price)
                        except (TypeError, ValueError):
                            mp = None
                        if mp is not None:
                            if target_price is not None:
                                try:
                                    target = float(target_price)
                                except (TypeError, ValueError):
                                    target = None
                                if target is not None and target > 0:
                                    if (pos > 0 and mp >= target) or (pos < 0 and mp <= target):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                            if stop_price is not None:
                                try:
                                    stop = float(stop_price)
                                except (TypeError, ValueError):
                                    stop = None
                                if stop is not None and stop > 0:
                                    if (pos > 0 and mp <= stop) or (pos < 0 and mp >= stop):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                    move = None
                    if (
                        target_price is None
                        and stop_price is None
                        and avg_cost is not None
                        and avg_cost > 0
                        and market_price is not None
                        and market_price > 0
                        and pos
                    ):
                        move = (market_price - avg_cost) / avg_cost
                        if pos < 0:
                            move = -move

                    try:
                        pt = (
                            float(instance.strategy.get("spot_profit_target_pct"))
                            if instance.strategy.get("spot_profit_target_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        pt = None
                    try:
                        sl = (
                            float(instance.strategy.get("spot_stop_loss_pct"))
                            if instance.strategy.get("spot_stop_loss_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        sl = None

                    # Dynamic shock SL/PT: mirror backtest semantics for pct-based exits.
                    if bool(snap.shock) and isinstance(instance.filters, dict):
                        try:
                            sl_mult = float(instance.filters.get("shock_stop_loss_pct_mult", 1.0) or 1.0)
                        except (TypeError, ValueError):
                            sl_mult = 1.0
                        try:
                            pt_mult = float(instance.filters.get("shock_profit_target_pct_mult", 1.0) or 1.0)
                        except (TypeError, ValueError):
                            pt_mult = 1.0
                        if sl_mult <= 0:
                            sl_mult = 1.0
                        if pt_mult <= 0:
                            pt_mult = 1.0
                        if sl is not None and float(sl) > 0:
                            sl = min(float(sl) * float(sl_mult), 0.99)
                        if pt is not None and float(pt) > 0:
                            pt = min(float(pt) * float(pt_mult), 0.99)

                    if move is not None and pt is not None and move >= pt:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break
                    if move is not None and sl is not None and move <= -sl:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    exit_time = parse_time_hhmm(instance.strategy.get("spot_exit_time_et"))
                    if exit_time is not None and now_et.time() >= exit_time:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    if bool(instance.strategy.get("spot_close_eod")) and (
                        now_et.hour > 15 or now_et.hour == 15 and now_et.minute >= 55
                    ):
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                if instrument != "spot":
                    if self._should_exit_on_dte(instance, open_items, now_et.date()):
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    entry_value, current_value = self._options_position_values(open_items)
                    if entry_value is not None and current_value is not None:
                        profit = float(entry_value) - float(current_value)
                        try:
                            profit_target = float(instance.strategy.get("profit_target", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            profit_target = 0.0
                        if profit_target > 0 and abs(entry_value) > 0:
                            if profit >= abs(entry_value) * profit_target:
                                self._queue_proposal(
                                    instance,
                                    intent="exit",
                                    direction=open_dir,
                                    signal_bar_ts=snap.bar_ts,
                                )
                                break

                        try:
                            stop_loss = float(instance.strategy.get("stop_loss", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            stop_loss = 0.0
                        if stop_loss > 0:
                            loss = max(0.0, float(current_value) - float(entry_value))
                            basis = str(instance.strategy.get("stop_loss_basis") or "max_loss").strip().lower()
                            if basis == "credit":
                                if entry_value >= 0:
                                    if current_value >= entry_value * (1.0 + stop_loss):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                                elif loss >= abs(entry_value) * stop_loss:
                                    self._queue_proposal(
                                        instance,
                                        intent="exit",
                                        direction=open_dir,
                                        signal_bar_ts=snap.bar_ts,
                                    )
                                    break
                            else:
                                max_loss = self._options_max_loss_estimate(open_items, spot=float(snap.close))
                                if max_loss is None or max_loss <= 0:
                                    max_loss = abs(entry_value)
                                if max_loss and loss >= float(max_loss) * stop_loss:
                                    self._queue_proposal(
                                        instance,
                                        intent="exit",
                                        direction=open_dir,
                                        signal_bar_ts=snap.bar_ts,
                                    )
                                    break

                if self._should_exit_on_flip(instance, snap, open_dir, open_items):
                    self._queue_proposal(instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts)
                    break
                continue

            if not self._entry_limit_ok(instance):
                continue
            if instance.last_entry_bar_ts is not None and instance.last_entry_bar_ts == snap.bar_ts:
                continue

            instrument = self._strategy_instrument(instance.strategy)
            if instrument == "spot":
                exit_mode = str(instance.strategy.get("spot_exit_mode") or "pct").strip().lower()
                if exit_mode == "atr":
                    atr = float(snap.atr or 0.0) if snap.atr is not None else 0.0
                    if atr <= 0:
                        continue

            direction = self._entry_direction_for_instance(instance, snap)
            if direction is None:
                continue
            if direction not in self._allowed_entry_directions(instance):
                continue

            self._queue_proposal(
                instance,
                intent="enter",
                direction=direction,
                signal_bar_ts=snap.bar_ts,
            )
            break

    def _auto_send_tick(self) -> None:
        if self._send_task and not self._send_task.done():
            return
        proposal = next((p for p in self._proposals if p.status == "PROPOSED"), None)
        if proposal is None:
            return
        instance = next((i for i in self._instances if i.instance_id == proposal.instance_id), None)
        if not instance or not instance.auto_trade:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Auto: no loop"
            return
        self._send_task = loop.create_task(self._send_order(proposal))

    async def _chase_proposals_tick(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        now = loop.time()
        if now - self._last_chase_ts < 0.5:
            return
        self._last_chase_ts = now

        updated = False
        for proposal in self._proposals:
            if proposal.status != "PROPOSED":
                continue
            instance = next(
                (i for i in self._instances if i.instance_id == proposal.instance_id), None
            )
            if not instance:
                continue
            if instance.strategy.get("chase_proposals") is False:
                continue
            changed = await self._reprice_proposal(proposal, instance)
            updated = updated or changed
        if updated:
            self._refresh_proposals_table()
            if self._active_panel == "proposals" and self._proposal_rows:
                row = min(
                    self._proposals_table.cursor_coordinate.row, len(self._proposal_rows) - 1
                )
                self._proposals_table.cursor_coordinate = (max(row, 0), 0)

    async def _reprice_proposal(self, proposal: _BotProposal, instance: _BotInstance) -> bool:
        raw_mode = str(instance.strategy.get("price_mode") or "MID").strip().upper()
        if raw_mode not in ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"):
            raw_mode = "MID"
        price_mode = raw_mode

        def _leg_price(
            bid: float | None, ask: float | None, last: float | None, action: str
        ) -> float | None:
            mid = _midpoint(bid, ask)
            if price_mode == "CROSS":
                return ask if action == "BUY" else bid
            if price_mode == "MID":
                return mid or last
            if price_mode == "OPTIMISTIC":
                return _optimistic_price(bid, ask, mid, action) or mid or last
            if price_mode == "AGGRESSIVE":
                return _aggressive_price(bid, ask, mid, action) or mid or last
            return mid or last

        legs = proposal.legs or []
        if not legs:
            return False

        if len(legs) == 1 and proposal.order_contract.secType != "BAG":
            leg = legs[0]
            ticker = await self._client.ensure_ticker(leg.contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _leg_price(bid, ask, last, proposal.action)
            if limit is None:
                return False
            tick = _tick_size(leg.contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            changed = not math.isclose(limit, proposal.limit_price, rel_tol=0, abs_tol=tick / 2.0)
            proposal.limit_price = float(limit)
            proposal.bid = bid
            proposal.ask = ask
            proposal.last = last
            return changed

        debit_mid = 0.0
        debit_bid = 0.0
        debit_ask = 0.0
        desired_debit = 0.0
        tick = None
        for leg in legs:
            ticker = await self._client.ensure_ticker(leg.contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            mid = _midpoint(bid, ask)
            leg_mid = mid or last
            if leg_mid is None:
                return False
            leg_bid = bid or mid or last
            leg_ask = ask or mid or last
            leg_desired = _leg_price(bid, ask, last, leg.action)
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
        proposal.action = "BUY"
        new_limit = _round_to_tick(float(desired_debit), tick)
        if not new_limit:
            return False
        new_bid = float(debit_bid)
        new_ask = float(debit_ask)
        new_last = float(debit_mid)
        changed = not math.isclose(
            new_limit, proposal.limit_price, rel_tol=0, abs_tol=tick / 2.0
        )
        proposal.limit_price = float(new_limit)
        proposal.bid = new_bid
        proposal.ask = new_ask
        proposal.last = new_last
        return changed

    def _queue_proposal(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        if self._proposal_task and not self._proposal_task.done():
            self._status = "Propose: busy"
            self._render_status()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Propose: no loop"
            self._render_status()
            return
        action = "Exiting" if intent == "exit" else "Proposing"
        dir_note = f" ({direction})" if direction else ""
        self._status = f"{action} for instance {instance.instance_id}{dir_note}..."
        self._render_status()
        self._proposal_task = loop.create_task(
            self._propose_for_instance(
                instance,
                intent=str(intent),
                direction=direction,
                signal_bar_ts=signal_bar_ts,
            )
        )

    def _can_propose_now(self, instance: _BotInstance) -> bool:
        entry_days = instance.strategy.get("entry_days", [])
        if entry_days:
            allowed = {_weekday_num(day) for day in entry_days}
        else:
            allowed = {0, 1, 2, 3, 4}
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        if now.weekday() not in allowed:
            return False
        return True

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

    def _signal_duration_str(self, bar_size: str, *, filters: dict | None = None) -> str:
        label = str(bar_size or "").strip().lower()

        def _rank(duration: str) -> int:
            order = ("1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
            cleaned = str(duration or "").strip()
            try:
                return order.index(cleaned)
            except ValueError:
                return 0

        def _max_duration(a: str, b: str) -> str:
            return a if _rank(a) >= _rank(b) else b

        base = "2 W"
        if label.startswith(("5 mins", "15 mins", "30 mins")):
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
        from ..utils.bar_utils import trim_incomplete_last_bar
        from ..spot_engine import SpotSignalEvaluator

        entry_signal = str(entry_signal_raw or "ema").strip().lower()
        if entry_signal not in ("ema", "orb"):
            entry_signal = "ema"

        bars = await self._client.historical_bars_ohlcv(
            contract,
            duration_str=self._signal_duration_str(bar_size, filters=filters),
            bar_size=bar_size,
            use_rth=use_rth,
            cache_ttl_sec=30.0,
        )
        if not bars:
            return None
        now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        bars = trim_incomplete_last_bar(bars, bar_size=bar_size, now_ref=now_ref)
        if not bars:
            return None

        regime_mode = str(regime_mode_raw or "ema").strip().lower()
        if regime_mode not in ("ema", "supertrend"):
            regime_mode = "ema"

        regime_preset = str(regime_ema_preset_raw or "").strip() or None
        regime_bar_size = str(regime_bar_size_raw or "").strip()
        if not regime_bar_size or regime_bar_size.lower() in ("same", "default"):
            regime_bar_size = str(bar_size)
        if regime_mode == "supertrend":
            use_mtf_regime = str(regime_bar_size) != str(bar_size)
        else:
            use_mtf_regime = bool(regime_preset) and (str(regime_bar_size) != str(bar_size))

        regime_bars = None
        if use_mtf_regime:
            regime_duration = self._signal_duration_str(regime_bar_size, filters=filters)
            if isinstance(filters, dict) and "hour" in str(regime_bar_size).strip().lower():
                shock_gate_mode = str(filters.get("shock_gate_mode") or "off").strip().lower()
                if shock_gate_mode in ("", "0", "false", "none", "null"):
                    shock_gate_mode = "off"
                if shock_gate_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
                    shock_gate_mode = "off"
                if shock_gate_mode != "off":
                    try:
                        atr_slow = int(filters.get("shock_atr_slow_period", 50))
                    except (TypeError, ValueError):
                        atr_slow = 50
                    if atr_slow > 0:
                        # Avoid shrinking the base duration (daily detectors may require longer).
                        alt = "1 M" if atr_slow <= 60 else "2 M"
                        order = ("1 W", "2 W", "1 M", "2 M", "3 M", "6 M", "1 Y", "2 Y")
                        try:
                            if order.index(str(alt)) > order.index(str(regime_duration)):
                                regime_duration = alt
                        except ValueError:
                            pass
            regime_bars = await self._client.historical_bars_ohlcv(
                contract,
                duration_str=regime_duration,
                bar_size=regime_bar_size,
                use_rth=use_rth,
                cache_ttl_sec=30.0,
            )
            if not regime_bars:
                return None
            regime_bars = trim_incomplete_last_bar(
                regime_bars, bar_size=regime_bar_size, now_ref=now_ref
            )

        regime2_mode = str(regime2_mode_raw or "off").strip().lower()
        if regime2_mode not in ("off", "ema", "supertrend"):
            regime2_mode = "off"
        regime2_preset = str(regime2_ema_preset_raw or "").strip() or None
        if regime2_mode == "ema" and not regime2_preset:
            regime2_mode = "off"
        regime2_bar_size = str(regime2_bar_size_raw or "").strip()
        if not regime2_bar_size or regime2_bar_size.lower() in ("same", "default"):
            regime2_bar_size = str(bar_size)
        if regime2_mode == "supertrend":
            use_mtf_regime2 = str(regime2_bar_size) != str(bar_size)
        else:
            use_mtf_regime2 = bool(regime2_preset) and (str(regime2_bar_size) != str(bar_size))

        regime2_bars = None
        if regime2_mode != "off" and use_mtf_regime2:
            regime2_bars = await self._client.historical_bars_ohlcv(
                contract,
                duration_str=self._signal_duration_str(regime2_bar_size, filters=filters),
                bar_size=regime2_bar_size,
                use_rth=use_rth,
                cache_ttl_sec=30.0,
            )
            if not regime2_bars:
                return None
            regime2_bars = trim_incomplete_last_bar(
                regime2_bars, bar_size=regime2_bar_size, now_ref=now_ref
            )

        strategy = {
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

        try:
            evaluator = SpotSignalEvaluator(
                strategy=strategy,
                filters=filters,
                bar_size=str(bar_size),
                use_rth=bool(use_rth),
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
            )
        except ValueError:
            return None

        last_snap = None
        for idx, bar in enumerate(bars):
            next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
            is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
            evaluator.update_exec_bar(bar, is_last_bar=bool(is_last_bar))
            snap = evaluator.update_signal_bar(bar)
            if snap is not None:
                last_snap = snap
        if last_snap is None or not bool(last_snap.signal.ema_ready):
            return None

        return _SignalSnapshot(
            bar_ts=last_snap.bar_ts,
            close=float(last_snap.close),
            signal=last_snap.signal,
            bars_in_day=int(last_snap.bars_in_day),
            rv=float(last_snap.rv) if last_snap.rv is not None else None,
            volume=float(last_snap.volume) if last_snap.volume is not None else None,
            volume_ema=float(last_snap.volume_ema) if last_snap.volume_ema is not None else None,
            volume_ema_ready=bool(last_snap.volume_ema_ready),
            shock=last_snap.shock,
            shock_dir=last_snap.shock_dir,
            shock_atr_pct=float(last_snap.shock_atr_pct) if last_snap.shock_atr_pct is not None else None,
            risk=last_snap.risk,
            atr=float(last_snap.atr) if last_snap.atr is not None else None,
            or_high=float(last_snap.or_high) if last_snap.or_high is not None else None,
            or_low=float(last_snap.or_low) if last_snap.or_low is not None else None,
            or_ready=bool(last_snap.or_ready),
        )

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

    async def _strike_by_delta(
        self,
        *,
        symbol: str,
        expiry: str,
        right_char: str,
        strikes: list[float],
        trading_class: str | None,
        near_strike: float,
        target_delta: float,
    ) -> float | None:
        try:
            target = abs(float(target_delta))
        except (TypeError, ValueError):
            return None
        if target <= 0 or target > 1:
            return None
        try:
            strike_values = sorted(float(s) for s in strikes)
        except (TypeError, ValueError):
            return None
        if not strike_values:
            return None
        center_idx = min(
            range(len(strike_values)), key=lambda idx: abs(strike_values[idx] - near_strike)
        )
        window = strike_values[max(center_idx - 10, 0) : center_idx + 11]
        if not window:
            return None

        candidates = [
            Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=float(strike),
                right=right_char,
                exchange="SMART",
                currency="USD",
                tradingClass=trading_class,
            )
            for strike in window
        ]
        qualified = await self._client.qualify_proxy_contracts(*candidates)
        if qualified and len(qualified) == len(candidates):
            contracts: list[Contract] = list(qualified)
        else:
            contracts = list(candidates)

        best_strike: float | None = None
        best_diff: float | None = None
        for contract, strike in zip(contracts, window):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            delta = None
            for _ in range(6):
                for attr in ("modelGreeks", "bidGreeks", "askGreeks", "lastGreeks"):
                    greeks = getattr(ticker, attr, None)
                    if greeks is not None:
                        raw = getattr(greeks, "delta", None)
                        if raw is not None:
                            try:
                                delta = float(raw)
                            except (TypeError, ValueError):
                                delta = None
                            break
                if delta is not None:
                    break
                await asyncio.sleep(0.05)
            if delta is None:
                continue
            diff = abs(abs(delta) - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_strike = float(strike)

        # Avoid keeping a large number of quote subscriptions just to pick strike.
        for contract, strike in zip(contracts, window):
            if best_strike is not None and float(strike) == best_strike:
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._client.release_ticker(con_id)
                self._tracked_conids.discard(con_id)
        return best_strike

    async def _propose_for_instance(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        strat = instance.strategy or {}
        instrument = self._strategy_instrument(strat)
        intent_clean = str(intent or "enter").strip().lower()
        intent_clean = "exit" if intent_clean == "exit" else "enter"
        symbol = str(
            instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()

        raw_mode = str(strat.get("price_mode") or "MID").strip().upper()
        if raw_mode not in ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"):
            raw_mode = "MID"
        price_mode = raw_mode

        def _leg_price(bid: float | None, ask: float | None, last: float | None, action: str) -> float | None:
            mid = _midpoint(bid, ask)
            if price_mode == "CROSS":
                return ask if action == "BUY" else bid
            if price_mode == "MID":
                return mid or last
            if price_mode == "OPTIMISTIC":
                return _optimistic_price(bid, ask, mid, action) or mid or last
            if price_mode == "AGGRESSIVE":
                return _aggressive_price(bid, ask, mid, action) or mid or last
            return mid or last

        def _bump_entry_counters() -> None:
            self._reset_daily_counters_if_needed(instance)
            instance.entries_today += 1

        def _finalize_leg_orders(
            *,
            underlying: Contract,
            leg_orders: list[_BotLegOrder],
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]],
        ) -> None:
            if not leg_orders:
                self._status = "Propose: no legs configured"
                self._render_status()
                return

            # Compute net price in "debit units": BUY adds, SELL subtracts.
            debit_mid = 0.0
            debit_bid = 0.0
            debit_ask = 0.0
            desired_debit = 0.0
            tick = None
            for leg_order, (bid, ask, last, ticker) in zip(leg_orders, leg_quotes):
                mid = _midpoint(bid, ask)
                leg_mid = mid or last
                if leg_mid is None:
                    self._status = "Quote: missing mid/last (cannot price)"
                    self._render_status()
                    return
                leg_bid = bid or mid or last
                leg_ask = ask or mid or last
                if leg_bid is None or leg_ask is None:
                    self._status = "Quote: missing bid/ask (cannot price)"
                    self._render_status()
                    return
                leg_desired = _leg_price(bid, ask, last, leg_order.action)
                if leg_desired is None:
                    self._status = "Quote: missing bid/ask/last (cannot price)"
                    self._render_status()
                    return
                leg_tick = _tick_size(leg_order.contract, ticker, leg_desired)
                tick = leg_tick if tick is None else min(tick, leg_tick)
                sign = 1.0 if leg_order.action == "BUY" else -1.0
                debit_mid += sign * float(leg_mid) * leg_order.ratio
                debit_bid += sign * float(leg_bid) * leg_order.ratio
                debit_ask += sign * float(leg_ask) * leg_order.ratio
                desired_debit += sign * float(leg_desired) * leg_order.ratio

            tick = tick or 0.01

            if len(leg_orders) == 1:
                single = leg_orders[0]
                (bid, ask, last, ticker) = leg_quotes[0]
                limit = _leg_price(bid, ask, last, single.action)
                if limit is None:
                    self._status = "Quote: no bid/ask/last (cannot price)"
                    self._render_status()
                    return
                limit = _round_to_tick(float(limit), tick)
                proposal = _BotProposal(
                    instance_id=instance.instance_id,
                    preset=None,
                    underlying=underlying,
                    order_contract=single.contract,
                    legs=leg_orders,
                    action=single.action,
                    quantity=single.ratio,
                    limit_price=float(limit),
                    created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                    bid=bid,
                    ask=ask,
                    last=last,
                    intent=intent_clean,
                    direction=direction,
                    reason=intent_clean,
                    signal_bar_ts=signal_bar_ts,
                )
                con_id = int(getattr(single.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
                self._add_proposal(proposal)
                self._status = f"Proposed {single.action} {single.ratio} {symbol} @ {limit:.2f}"
                if intent_clean == "enter":
                    if direction in ("up", "down"):
                        instance.open_direction = str(direction)
                    if signal_bar_ts is not None:
                        instance.last_entry_bar_ts = signal_bar_ts
                    _bump_entry_counters()
                elif signal_bar_ts is not None:
                    instance.last_exit_bar_ts = signal_bar_ts
                self._render_status()
                return

            # Multi-leg combo: use IBKR's native encoding (can be negative for credits).
            order_action = "BUY"
            order_bid = debit_bid
            order_ask = debit_ask
            order_last = debit_mid
            order_limit = _round_to_tick(float(desired_debit), tick)
            if not order_limit:
                self._status = "Quote: combo price is 0 (cannot price)"
                self._render_status()
                return
            combo_legs: list[ComboLeg] = []
            for leg_order, (_, _, _, ticker) in zip(leg_orders, leg_quotes):
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if not con_id:
                    self._status = "Contract: missing conId for combo leg"
                    self._render_status()
                    return
                leg_exchange = (
                    getattr(getattr(ticker, "contract", None), "exchange", "") or ""
                ).strip()
                if not leg_exchange:
                    leg_exchange = (getattr(leg_order.contract, "exchange", "") or "").strip()
                if not leg_exchange:
                    leg_sec_type = str(getattr(leg_order.contract, "secType", "") or "").strip()
                    leg_exchange = "CME" if leg_sec_type == "FOP" else "SMART"
                combo_legs.append(
                    ComboLeg(
                        conId=con_id,
                        ratio=leg_order.ratio,
                        action=leg_order.action,
                        exchange=leg_exchange,
                    )
                )
            bag = Bag(symbol=symbol, exchange="SMART", currency="USD", comboLegs=combo_legs)

            proposal = _BotProposal(
                instance_id=instance.instance_id,
                preset=None,
                underlying=underlying,
                order_contract=bag,
                legs=leg_orders,
                action=order_action,
                quantity=1,
                limit_price=float(order_limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=order_bid,
                ask=order_ask,
                last=order_last,
                intent=intent_clean,
                direction=direction,
                reason=intent_clean,
                signal_bar_ts=signal_bar_ts,
            )
            for leg_order in leg_orders:
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
            self._add_proposal(proposal)
            self._status = (
                f"Proposed {order_action} BAG {symbol} @ {order_limit:.2f} ({len(leg_orders)} legs)"
            )
            if intent_clean == "enter":
                if direction in ("up", "down"):
                    instance.open_direction = str(direction)
                if signal_bar_ts is not None:
                    instance.last_entry_bar_ts = signal_bar_ts
                _bump_entry_counters()
            elif signal_bar_ts is not None:
                instance.last_exit_bar_ts = signal_bar_ts
            self._render_status()

        if intent_clean == "exit":
            if instrument == "spot":
                sec_type = self._spot_sec_type(instance, symbol)
                open_item = self._spot_open_position(symbol=symbol, sec_type=sec_type, con_id=0)
                if open_item is None:
                    self._status = f"Exit: no spot position for {symbol}"
                    self._render_status()
                    return
                try:
                    pos = float(getattr(open_item, "position", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pos = 0.0
                if not pos:
                    self._status = f"Exit: no spot position for {symbol}"
                    self._render_status()
                    return

                action = "SELL" if pos > 0 else "BUY"
                qty = int(abs(pos))
                if qty <= 0:
                    self._status = f"Exit: invalid position size for {symbol}"
                    self._render_status()
                    return

                contract = open_item.contract
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    self._tracked_conids.add(con_id)
                ticker = await self._client.ensure_ticker(contract)
                bid = _safe_num(getattr(ticker, "bid", None))
                ask = _safe_num(getattr(ticker, "ask", None))
                last = _safe_num(getattr(ticker, "last", None))
                limit = _leg_price(bid, ask, last, action)
                if limit is None:
                    self._status = "Quote: no bid/ask/last (cannot price)"
                    self._render_status()
                    return
                tick = _tick_size(contract, ticker, limit) or 0.01
                limit = _round_to_tick(float(limit), tick)
                proposal = _BotProposal(
                    instance_id=instance.instance_id,
                    preset=None,
                    underlying=contract,
                    order_contract=contract,
                    legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                    action=action,
                    quantity=qty,
                    limit_price=float(limit),
                    created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                    bid=bid,
                    ask=ask,
                    last=last,
                    intent=intent_clean,
                    direction=direction,
                    reason="exit",
                    signal_bar_ts=signal_bar_ts,
                )
                if con_id:
                    instance.touched_conids.add(con_id)
                self._add_proposal(proposal)
                if signal_bar_ts is not None:
                    instance.last_exit_bar_ts = signal_bar_ts
                self._status = f"Proposed EXIT {action} {qty} {symbol} @ {limit:.2f}"
                self._render_status()
                return

            open_items = self._options_open_positions(instance)
            if not open_items:
                self._status = f"Exit: no option positions for instance {instance.instance_id}"
                self._render_status()
                return
            underlying = Stock(symbol=symbol, exchange="SMART", currency="USD")
            qualified = await self._client.qualify_proxy_contracts(underlying)
            if qualified:
                underlying = qualified[0]

            leg_orders: list[_BotLegOrder] = []
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
            for item in open_items:
                contract = item.contract
                try:
                    pos = float(getattr(item, "position", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if not pos:
                    continue
                ratio = int(abs(pos))
                if ratio <= 0:
                    continue
                action = "SELL" if pos > 0 else "BUY"
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    self._tracked_conids.add(con_id)
                ticker = await self._client.ensure_ticker(contract)
                bid = _safe_num(getattr(ticker, "bid", None))
                ask = _safe_num(getattr(ticker, "ask", None))
                last = _safe_num(getattr(ticker, "last", None))
                leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
                leg_quotes.append((bid, ask, last, ticker))

            if not leg_orders:
                self._status = f"Exit: no option positions for instance {instance.instance_id}"
                self._render_status()
                return
            _finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)
            return

        if instrument == "spot":
            entry_signal = str(strat.get("entry_signal") or "ema").strip().lower()
            if entry_signal not in ("ema", "orb"):
                entry_signal = "ema"
            exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
            if exit_mode not in ("pct", "atr"):
                exit_mode = "pct"

            signal_contract = await self._signal_contract(instance, symbol)
            snap = (
                await self._signal_snapshot_for_contract(
                    contract=signal_contract,
                    ema_preset_raw=str(strat.get("ema_preset")) if strat.get("ema_preset") else None,
                    bar_size=self._signal_bar_size(instance),
                    use_rth=self._signal_use_rth(instance),
                    entry_signal_raw=entry_signal,
                    orb_window_mins_raw=strat.get("orb_window_mins"),
                    orb_open_time_et_raw=strat.get("orb_open_time_et"),
                    entry_mode_raw=strat.get("ema_entry_mode"),
                    entry_confirm_bars=strat.get("entry_confirm_bars", 0),
                    spot_exit_mode_raw=strat.get("spot_exit_mode"),
                    spot_atr_period_raw=strat.get("spot_atr_period"),
                    regime_ema_preset_raw=strat.get("regime_ema_preset"),
                    regime_bar_size_raw=strat.get("regime_bar_size"),
                    regime_mode_raw=strat.get("regime_mode"),
                    supertrend_atr_period_raw=strat.get("supertrend_atr_period"),
                    supertrend_multiplier_raw=strat.get("supertrend_multiplier"),
                    supertrend_source_raw=strat.get("supertrend_source"),
                    regime2_ema_preset_raw=strat.get("regime2_ema_preset"),
                    regime2_bar_size_raw=strat.get("regime2_bar_size"),
                    regime2_mode_raw=strat.get("regime2_mode"),
                    regime2_supertrend_atr_period_raw=strat.get("regime2_supertrend_atr_period"),
                    regime2_supertrend_multiplier_raw=strat.get("regime2_supertrend_multiplier"),
                    regime2_supertrend_source_raw=strat.get("regime2_supertrend_source"),
                    filters=instance.filters if isinstance(instance.filters, dict) else None,
                )
                if signal_contract is not None
                else None
            )
            if snap is None:
                self._status = f"Signal: no snapshot for {symbol}"
                self._render_status()
                return

            if direction not in ("up", "down") and snap is not None:
                direction = self._entry_direction_for_instance(instance, snap) or (
                    str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                )
            direction = direction if direction in ("up", "down") else "up"

            mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
            chosen = mapping.get(direction) if mapping else None
            if not isinstance(chosen, dict):
                if direction == "up":
                    chosen = {"action": "BUY", "qty": 1}
                else:
                    chosen = {"action": "SELL", "qty": 1}
            action = str(chosen.get("action", "")).strip().upper()
            if action not in ("BUY", "SELL"):
                self._status = f"Propose: invalid spot action for {direction}"
                self._render_status()
                return
            try:
                qty = int(chosen.get("qty", 1) or 1)
            except (TypeError, ValueError):
                qty = 1
            qty = max(1, abs(qty))

            contract = await self._spot_contract(instance, symbol)
            if contract is None:
                self._status = f"Contract: not found for {symbol}"
                self._render_status()
                return

            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _leg_price(bid, ask, last, action)
            if limit is None:
                self._status = "Quote: no bid/ask/last (cannot price)"
                self._render_status()
                return
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)

            instance.spot_profit_target_price = None
            instance.spot_stop_loss_price = None
            if snap is not None and entry_signal == "orb":
                try:
                    rr = float(strat.get("orb_risk_reward", 2.0) or 2.0)
                except (TypeError, ValueError):
                    rr = 2.0
                target_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
                if target_mode not in ("rr", "or_range"):
                    target_mode = "rr"
                or_high = snap.or_high
                or_low = snap.or_low
                if (
                    rr > 0
                    and bool(snap.or_ready)
                    and or_high is not None
                    and or_low is not None
                    and float(or_high) > 0
                    and float(or_low) > 0
                ):
                    stop = float(or_low) if direction == "up" else float(or_high)
                    if target_mode == "or_range":
                        rng = float(or_high) - float(or_low)
                        if rng > 0:
                            target = (
                                float(or_high) + (rr * rng)
                                if direction == "up"
                                else float(or_low) - (rr * rng)
                            )
                            if (
                                (direction == "up" and float(target) <= float(limit))
                                or (direction == "down" and float(target) >= float(limit))
                            ):
                                self._status = "Propose: ORB target already hit (skip)"
                                self._render_status()
                                return
                            instance.spot_profit_target_price = float(target)
                            instance.spot_stop_loss_price = float(stop)
                    else:
                        risk = abs(float(limit) - stop)
                        if risk > 0:
                            target = float(limit) + (rr * risk) if direction == "up" else float(limit) - (rr * risk)
                            instance.spot_profit_target_price = float(target)
                            instance.spot_stop_loss_price = float(stop)
            elif exit_mode == "atr":
                atr = float(snap.atr) if snap is not None and snap.atr is not None else 0.0
                if atr <= 0:
                    self._status = "Propose: ATR not ready (spot_exit_mode=atr)"
                    self._render_status()
                    return
                try:
                    pt_mult = float(strat.get("spot_pt_atr_mult", 1.5) or 1.5)
                except (TypeError, ValueError):
                    pt_mult = 1.5
                try:
                    sl_mult = float(strat.get("spot_sl_atr_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    sl_mult = 1.0
                if direction == "up":
                    instance.spot_profit_target_price = float(limit) + (pt_mult * atr)
                    instance.spot_stop_loss_price = float(limit) - (sl_mult * atr)
                else:
                    instance.spot_profit_target_price = float(limit) - (pt_mult * atr)
                    instance.spot_stop_loss_price = float(limit) + (sl_mult * atr)

            # Spot sizing: mirror backtest semantics (fixed / notional_pct / risk_pct), with optional
            # shock/risk overlays applied via the filters snapshot.
            from ..engine import spot_calc_signed_qty

            filters = instance.filters if isinstance(instance.filters, dict) else None
            stop_loss_pct = None
            try:
                stop_loss_pct = (
                    float(strat.get("spot_stop_loss_pct"))
                    if strat.get("spot_stop_loss_pct") is not None
                    else None
                )
            except (TypeError, ValueError):
                stop_loss_pct = None
            if stop_loss_pct is not None and float(stop_loss_pct) <= 0:
                stop_loss_pct = None

            stop_price = instance.spot_stop_loss_price
            if stop_price is not None:
                try:
                    stop_price = float(stop_price)
                except (TypeError, ValueError):
                    stop_price = None
            if stop_price is not None and float(stop_price) <= 0:
                stop_price = None

            shock_now = bool(snap.shock) if snap.shock is not None else False
            if shock_now and filters is not None:
                try:
                    sl_mult = float(filters.get("shock_stop_loss_pct_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    sl_mult = 1.0
                if sl_mult > 0 and stop_loss_pct is not None and float(stop_loss_pct) > 0:
                    stop_loss_pct = min(float(stop_loss_pct) * float(sl_mult), 0.99)

            net_liq_val, _currency, _updated = self._client.account_value("NetLiquidation")
            buying_power_val, _bp_currency, _bp_updated = self._client.account_value("BuyingPower")
            try:
                equity_ref = float(net_liq_val) if net_liq_val is not None else 0.0
            except (TypeError, ValueError):
                equity_ref = 0.0
            try:
                cash_ref = float(buying_power_val) if buying_power_val is not None else None
            except (TypeError, ValueError):
                cash_ref = None

            riskoff = bool(snap.risk.riskoff) if snap.risk is not None else False
            riskpanic = bool(snap.risk.riskpanic) if snap.risk is not None else False

            signed_qty = spot_calc_signed_qty(
                strategy=strat,
                filters=filters,
                action=str(action),
                lot=int(qty),
                entry_price=float(limit),
                stop_price=stop_price,
                stop_loss_pct=stop_loss_pct,
                shock=snap.shock,
                shock_dir=snap.shock_dir,
                shock_atr_pct=snap.shock_atr_pct,
                riskoff=riskoff,
                risk_dir=snap.shock_dir,
                riskpanic=riskpanic,
                equity_ref=float(equity_ref),
                cash_ref=cash_ref,
            )
            if signed_qty == 0:
                self._status = "Propose: spot sizing returned 0 qty"
                self._render_status()
                return
            action = "BUY" if int(signed_qty) > 0 else "SELL"
            qty = int(abs(int(signed_qty)))

            journal = {
                "intent": intent_clean,
                "direction": direction,
                "bar_ts": snap.bar_ts.isoformat() if snap is not None else None,
                "close": float(snap.close) if snap is not None else None,
                "signal": {
                    "state": getattr(getattr(snap, "signal", None), "state", None),
                    "entry_dir": getattr(getattr(snap, "signal", None), "entry_dir", None),
                    "regime_dir": getattr(getattr(snap, "signal", None), "regime_dir", None),
                    "ema_ready": bool(getattr(getattr(snap, "signal", None), "ema_ready", False)),
                },
                "bars_in_day": int(snap.bars_in_day) if snap is not None else None,
                "rv": float(snap.rv) if snap is not None and snap.rv is not None else None,
                "volume": float(snap.volume) if snap is not None and snap.volume is not None else None,
                "shock": bool(snap.shock) if snap is not None and snap.shock is not None else None,
                "shock_dir": snap.shock_dir if snap is not None else None,
                "shock_atr_pct": float(snap.shock_atr_pct)
                if snap is not None and snap.shock_atr_pct is not None
                else None,
                "riskoff": bool(snap.risk.riskoff) if snap is not None and snap.risk is not None else None,
                "riskpanic": bool(snap.risk.riskpanic) if snap is not None and snap.risk is not None else None,
                "atr": float(snap.atr) if snap is not None and snap.atr is not None else None,
                "or_high": float(snap.or_high) if snap is not None and snap.or_high is not None else None,
                "or_low": float(snap.or_low) if snap is not None and snap.or_low is not None else None,
                "or_ready": bool(snap.or_ready) if snap is not None else None,
                "exit_mode": exit_mode,
                "stop_loss_pct": float(stop_loss_pct) if stop_loss_pct is not None else None,
                "stop_price": float(stop_price) if stop_price is not None else None,
                "target_price": float(instance.spot_profit_target_price)
                if instance.spot_profit_target_price is not None
                else None,
                "net_liq": float(equity_ref) if equity_ref is not None else None,
                "buying_power": float(cash_ref) if cash_ref is not None else None,
                "price_mode": price_mode,
                "chase_proposals": bool(strat.get("chase_proposals", True)),
            }

            proposal = _BotProposal(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=bid,
                ask=ask,
                last=last,
                intent=intent_clean,
                direction=direction,
                reason="enter",
                signal_bar_ts=snap.bar_ts if snap is not None else signal_bar_ts,
                journal=journal,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            instance.open_direction = str(direction)
            if signal_bar_ts is not None:
                instance.last_entry_bar_ts = signal_bar_ts
            _bump_entry_counters()
            self._add_proposal(proposal)
            self._status = f"Proposed {action} {qty} {symbol} @ {limit:.2f} ({direction})"
            self._render_status()
            return

        legs_raw: list[dict] | None = None
        if isinstance(strat.get("directional_legs"), dict):
            dmap = strat.get("directional_legs") or {}
            if direction not in ("up", "down"):
                ema_preset = strat.get("ema_preset")
                if ema_preset:
                    signal_contract = await self._signal_contract(instance, symbol)
                    snap = (
                        await self._signal_snapshot_for_contract(
                            contract=signal_contract,
                            ema_preset_raw=str(ema_preset),
                            bar_size=self._signal_bar_size(instance),
                            use_rth=self._signal_use_rth(instance),
                            entry_mode_raw=strat.get("ema_entry_mode"),
                            entry_confirm_bars=strat.get("entry_confirm_bars", 0),
                            regime_ema_preset_raw=strat.get("regime_ema_preset"),
                            regime_bar_size_raw=strat.get("regime_bar_size"),
                            regime_mode_raw=strat.get("regime_mode"),
                            supertrend_atr_period_raw=strat.get("supertrend_atr_period"),
                            supertrend_multiplier_raw=strat.get("supertrend_multiplier"),
                            supertrend_source_raw=strat.get("supertrend_source"),
                            regime2_ema_preset_raw=strat.get("regime2_ema_preset"),
                            regime2_bar_size_raw=strat.get("regime2_bar_size"),
                            regime2_mode_raw=strat.get("regime2_mode"),
                            regime2_supertrend_atr_period_raw=strat.get("regime2_supertrend_atr_period"),
                            regime2_supertrend_multiplier_raw=strat.get("regime2_supertrend_multiplier"),
                            regime2_supertrend_source_raw=strat.get("regime2_supertrend_source"),
                            filters=instance.filters if isinstance(instance.filters, dict) else None,
                        )
                            if signal_contract is not None
                            else None
                    )
                    if snap is not None:
                        direction = self._entry_direction_for_instance(instance, snap) or (
                            str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                        )
            if direction in ("up", "down") and dmap.get(direction):
                legs_raw = dmap.get(direction)
            else:
                for key in ("up", "down"):
                    if dmap.get(key):
                        legs_raw = dmap.get(key)
                        direction = key
                        break

        if legs_raw is None:
            raw = strat.get("legs", []) or []
            legs_raw = raw if isinstance(raw, list) else []

        if not isinstance(legs_raw, list) or not legs_raw:
            self._status = "Propose: no legs configured"
            self._render_status()
            return

        dte_raw = strat.get("dte", 0)
        try:
            dte = int(dte_raw or 0)
        except (TypeError, ValueError):
            dte = 0

        chain_info = await self._client.stock_option_chain(symbol)
        if not chain_info:
            self._status = f"Chain: not found for {symbol}"
            self._render_status()
            return
        underlying, chain = chain_info
        underlying_ticker = await self._client.ensure_ticker(underlying)
        under_con_id = int(getattr(underlying, "conId", 0) or 0)
        if under_con_id:
            self._tracked_conids.add(under_con_id)
        spot = _ticker_price(underlying_ticker)
        if spot is None:
            self._status = f"Spot: n/a for {symbol}"
            self._render_status()
            return

        expiry = _pick_chain_expiry(date.today(), dte, getattr(chain, "expirations", []))
        if not expiry:
            self._status = f"Expiry: none for {symbol}"
            self._render_status()
            return

        # Build and qualify option legs.
        strikes = getattr(chain, "strikes", [])
        trading_class = getattr(chain, "tradingClass", None)
        option_candidates: list[Option] = []
        leg_specs: list[tuple[str, str, int, float, float | None]] = []
        for leg_raw in legs_raw:
            if not isinstance(leg_raw, dict):
                self._status = "Propose: invalid leg config"
                self._render_status()
                return
            action = str(leg_raw.get("action", "")).upper()
            right = str(leg_raw.get("right", "")).upper()
            if action not in ("BUY", "SELL") or right not in ("PUT", "CALL"):
                self._status = "Propose: invalid leg config"
                self._render_status()
                return
            try:
                ratio = int(leg_raw.get("qty", 1) or 1)
            except (TypeError, ValueError):
                ratio = 1
            ratio = max(1, abs(ratio))
            try:
                moneyness = float(leg_raw.get("moneyness_pct", 0.0) or 0.0)
            except (TypeError, ValueError):
                moneyness = 0.0
            delta_target = leg_raw.get("delta")
            try:
                delta_target = float(delta_target) if delta_target is not None else None
            except (TypeError, ValueError):
                delta_target = None

            target_strike = _strike_from_moneyness(spot, right, moneyness)
            right_char = "P" if right == "PUT" else "C"
            strike = None
            if delta_target is not None and strikes:
                strike = await self._strike_by_delta(
                    symbol=symbol,
                    expiry=expiry,
                    right_char=right_char,
                    strikes=list(strikes),
                    trading_class=trading_class,
                    near_strike=target_strike,
                    target_delta=delta_target,
                )
            if strike is None:
                strike = _nearest_strike(strikes, target_strike)
            if strike is None:
                self._status = f"Strike: none for {symbol}"
                self._render_status()
                return
            option_candidates.append(
                Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(strike),
                    right=right_char,
                    exchange="SMART",
                    currency="USD",
                    tradingClass=trading_class,
                )
            )
            leg_specs.append((action, right, ratio, moneyness, delta_target))

        qualified = await self._client.qualify_proxy_contracts(*option_candidates)
        if qualified and len(qualified) == len(option_candidates):
            option_contracts: list[Contract] = list(qualified)
        else:
            option_contracts = list(option_candidates)

        leg_orders: list[_BotLegOrder] = []
        leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
        for contract, (action, _, ratio, _, _) in zip(option_contracts, leg_specs):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
            leg_quotes.append((bid, ask, last, ticker))

        _finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)

    def _submit_proposal(self) -> None:
        self._submit_selected_proposal()

    def _submit_selected_proposal(self) -> None:
        proposal = self._selected_proposal()
        if not proposal:
            self._status = "Send: no proposal selected"
            self._render_bot()
            return
        if proposal.status != "PROPOSED":
            self._status = f"Send: already {proposal.status}"
            self._render_bot()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Send: no loop"
            self._render_bot()
            return
        self._status = "Sending order..."
        self._render_bot()
        self._send_task = loop.create_task(self._send_order(proposal))

    async def _send_order(self, proposal: _BotProposal) -> None:
        try:
            self._journal_write(event="SENDING", proposal=proposal, reason=proposal.reason, data=None)
            trade = await self._client.place_limit_order(
                proposal.order_contract,
                proposal.action,
                proposal.quantity,
                proposal.limit_price,
                outside_rth=proposal.order_contract.secType == "STK",
            )
            order_id = trade.order.orderId or trade.order.permId or 0
            proposal.status = "SENT"
            proposal.order_id = int(order_id or 0) or None
            self._status = f"Sent #{order_id} {proposal.action} {proposal.quantity} @ {proposal.limit_price:.2f}"
            self._journal_write(event="SENT", proposal=proposal, reason=proposal.reason, data=None)
        except Exception as exc:
            proposal.status = "ERROR"
            proposal.error = str(exc)
            self._status = f"Send error: {exc}"
            self._journal_write(event="SEND_ERROR", proposal=proposal, reason=proposal.reason, data={"exc": str(exc)})
        self._refresh_proposals_table()
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
                "Enter=Config/Send  Ctrl+A=Presets  f=FilterDTE  w=FilterWin  v=Scope  Tab/h/l=Focus  p=Propose  a=Auto  Space=Run/Toggle  s=Stop  S=Kill  d=Del  X=Send",
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
                f"Instances: {len(self._instances)}  Proposals: {len(self._proposal_rows)}",
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
                auto = "ON" if instance.auto_trade else "OFF"
                lines.append(Text(""))
                lines.append(Text(f"Selected instance #{instance.instance_id}", style="bold"))
                lines.append(Text(f"{instance.group}  DTE={dte}  Legs={legs_desc}", style="dim"))
                lines.append(Text(f"State={instance.state}  Auto trade={auto}", style="dim"))
        elif self._active_panel == "proposals":
            proposal = self._selected_proposal()
            if proposal:
                lines.append(Text(""))
                lines.append(Text(f"Selected proposal (inst {proposal.instance_id})", style="bold"))
                lines.extend(_proposal_lines(proposal))

        if self._status:
            lines.append(Text(""))
            lines.append(Text(self._status, style="yellow"))

        self._status_panel.update(Text("\n").join(lines))

    def _add_proposal(self, proposal: _BotProposal) -> None:
        self._proposals.append(proposal)
        instance = next((i for i in self._instances if i.instance_id == proposal.instance_id), None)
        self._journal_write(event="PROPOSED", instance=instance, proposal=proposal, reason=proposal.reason, data=None)
        self._refresh_proposals_table()
        if self._active_panel == "proposals":
            self._proposals_table.cursor_coordinate = (max(len(self._proposal_rows) - 1, 0), 0)

    def _refresh_proposals_table(self) -> None:
        self._proposals_table.clear()
        self._proposal_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            return
        for proposal in self._proposals:
            if scope is not None and proposal.instance_id != scope:
                continue
            self._proposals_table.add_row(*_proposal_row(proposal))
            self._proposal_rows.append(proposal)


# endregion


# region UI Helpers
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
        entry_signal = str(strat.get("entry_signal") or "ema").strip().lower()
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


def _proposal_lines(proposal: _BotProposal) -> list[Text]:
    legs = proposal.legs or []
    legs_line: str | None = None
    if len(legs) == 1 and proposal.order_contract.secType != "BAG":
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
        symbol = getattr(proposal.order_contract, "symbol", "") or getattr(
            proposal.underlying, "symbol", ""
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
    parts.append(Text(f"Side: {proposal.action}  Qty: {proposal.quantity}", style="dim"))
    parts.append(Text(f"Limit: {_fmt_quote(proposal.limit_price)}", style="dim"))
    parts.append(
        Text(
            f"Bid: {_fmt_quote(proposal.bid)}  Ask: {_fmt_quote(proposal.ask)}  "
            f"Last: {_fmt_quote(proposal.last)}",
            style="dim",
        )
    )
    return parts


def _weekday_num(label: str) -> int:
    key = label.strip().upper()[:3]
    mapping = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    return mapping.get(key, 0)


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


def _pick_chain_expiry(today: date, dte: int, expirations: list[str]) -> str | None:
    if not expirations:
        return None
    target = add_business_days(today, dte)
    parsed: list[tuple[date, str]] = []
    for exp in expirations:
        dt = _parse_chain_date(exp)
        if dt:
            parsed.append((dt, exp))
    if not parsed:
        return None
    future = [(dt, exp) for dt, exp in parsed if dt >= target]
    candidates = future or parsed
    candidates.sort(key=lambda pair: abs((pair[0] - target).days))
    return candidates[0][1]


def _parse_chain_date(raw: str) -> date | None:
    raw = str(raw).strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


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


def _strike_from_moneyness(spot: float, right: str, moneyness_pct: float) -> float:
    # Negative moneyness means ITM (e.g., -1 = 1% ITM).
    if right == "PUT":
        return spot * (1 - moneyness_pct / 100.0)
    return spot * (1 + moneyness_pct / 100.0)


def _nearest_strike(strikes: list[float], target: float) -> float | None:
    if not strikes:
        return None
    try:
        return min((float(s) for s in strikes), key=lambda s: abs(s - target))
    except (TypeError, ValueError):
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


def _proposal_row(proposal: _BotProposal) -> tuple[str, str, str, str, str, str, str, str]:
    ts = proposal.created_at.astimezone().strftime("%H:%M:%S")
    inst = str(proposal.instance_id)
    contract = proposal.order_contract
    if contract.secType == "BAG" or len(proposal.legs) > 1:
        symbol = getattr(contract, "symbol", "") or getattr(proposal.underlying, "symbol", "") or "?"
        local = f"{symbol} BAG {len(proposal.legs)}L"
    else:
        leg_contract = proposal.legs[0].contract if proposal.legs else contract
        local = getattr(leg_contract, "localSymbol", "") or getattr(leg_contract, "symbol", "") or "?"
    local = str(local)[:12]
    side = proposal.action[:1]
    qty = str(int(proposal.quantity))
    limit = _fmt_quote(proposal.limit_price)
    bid = _fmt_quote(proposal.bid)
    ask = _fmt_quote(proposal.ask)
    bid_ask = f"{bid}/{ask}"
    status = proposal.status
    if proposal.order_id:
        status = f"{status} #{proposal.order_id}"
    if proposal.error and status == "ERROR":
        status = f"ERROR {proposal.error}"[:32]
    return (ts, inst, side, qty, local, limit, bid_ask, status)
