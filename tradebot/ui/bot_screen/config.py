"""Interactive bot preset and instance configuration screen."""

from __future__ import annotations

import copy
from typing import Optional

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ...contract_identity import is_future_symbol
from ..bot_models import _BotConfigField, _BotConfigResult
from ..common import _parse_float
from .formatting import (
    _get_path,
    _legs_direction_hint,
    _legs_label,
    _parse_entry_days,
    _set_path,
)

class BotConfigScreen(Screen[Optional[_BotConfigResult]]):
    _HOUR_FILTER_PATHS = {
        "filters.entry_start_hour_et",
        "filters.entry_end_hour_et",
        "filters.entry_start_hour",
        "filters.entry_end_hour",
    }

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
        yield DataTable(
            id="bot-config",
            zebra_stripes=True,
            show_row_labels=False,
            cursor_foreground_priority="renderable",
            cursor_background_priority="css",
        )
        yield Static(
            "Enter=Save & start  Esc=Cancel  Type=Edit  e=Edit w/ current  Space=Toggle  ←/→=Cycle enum  Hours: blank=off",
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
        self._sync_row_marker(force=True)
        self._table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; treat Enter as Save.
        self.action_save()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.control is self._table:
            self._sync_row_marker()

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
        self._strategy.setdefault("order_stage_timeout_sec", 20)

        if instrument == "spot":
            sym = str(self._symbol or self._strategy.get("symbol") or "").strip().upper()
            default_sec_type = "FUT" if is_future_symbol(sym) else "STK"
            self._strategy.setdefault("spot_sec_type", default_sec_type)
            self._strategy.setdefault("spot_exchange", "")
            self._strategy.setdefault("spot_close_eod", False)
            self._strategy.setdefault("spot_next_open_session", "auto")
            self._strategy.setdefault("spot_exec_feed_mode", "ticks_side")
            if not isinstance(self._strategy.get("directional_spot"), dict):
                self._strategy["directional_spot"] = {
                    "up": {"action": "BUY", "qty": 1},
                    "down": {"action": "SELL", "qty": 1},
                }

        if isinstance(self._filters, dict):
            start_et = self._filters.get("entry_start_hour_et")
            end_et = self._filters.get("entry_end_hour_et")
            start_legacy = self._filters.get("entry_start_hour")
            end_legacy = self._filters.get("entry_end_hour")
            if start_et is None and start_legacy is not None:
                self._filters["entry_start_hour_et"] = start_legacy
            elif start_legacy is None and start_et is not None:
                self._filters["entry_start_hour"] = start_et
            if end_et is None and end_legacy is not None:
                self._filters["entry_end_hour_et"] = end_legacy
            elif end_legacy is None and end_et is not None:
                self._filters["entry_end_hour"] = end_et

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
            _BotConfigField("Order stage timeout sec", "int", "order_stage_timeout_sec"),
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
                    _BotConfigField(
                        "Spot next-open session",
                        "enum",
                        "spot_next_open_session",
                        options=("auto", "rth", "extended", "tradable_24x5", "always"),
                    ),
                    _BotConfigField(
                        "Spot exec feed",
                        "enum",
                        "spot_exec_feed_mode",
                        options=("ticks_side", "ticks_quote", "poll"),
                    ),
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
        prev_row = 0
        prev_col = 0
        selected_path = None
        if hasattr(self, "_table"):
            try:
                coord = self._table.cursor_coordinate
                prev_row = int(getattr(coord, "row", 0) or 0)
                prev_col = int(getattr(coord, "column", 0) or 0)
            except Exception:
                prev_row = 0
                prev_col = 0
            field = self._selected_field()
            if field is not None:
                selected_path = field.path
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
                self._table.add_row(
                    Text(field.label, style="bold #ffd166"),
                    Text(self._edit_buffer, style="bold #ffd166"),
                )
                continue
            value = self._get_value(field)
            self._table.add_row(field.label, self._format_value(field, value))
        if self._fields:
            target_row = prev_row
            if selected_path:
                for idx, field in enumerate(self._fields):
                    if field.path == selected_path:
                        target_row = idx
                        break
            target_row = max(0, min(target_row, len(self._fields) - 1))
            target_col = max(0, min(prev_col, 1))
            self._table.cursor_coordinate = (target_row, target_col)
        self._sync_row_marker(force=True)

    def _sync_row_marker(self, *, force: bool = False) -> None:
        # DataTable row cursor highlight is sufficient; avoid label churn/flicker.
        return

    def _set_row_marker(self, row_index: int, *, active: bool) -> None:
        return

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

    @classmethod
    def _is_hour_filter_field(cls, field: _BotConfigField | None) -> bool:
        return bool(field and field.path in cls._HOUR_FILTER_PATHS)

    def _set_hour_filter_value(self, field_path: str, value: int | None) -> None:
        if self._filters is None:
            self._filters = {}
        if field_path in ("filters.entry_start_hour_et", "filters.entry_start_hour"):
            _set_path(self._filters, "entry_start_hour_et", value)
            _set_path(self._filters, "entry_start_hour", value)
            return
        if field_path in ("filters.entry_end_hour_et", "filters.entry_end_hour"):
            _set_path(self._filters, "entry_end_hour_et", value)
            _set_path(self._filters, "entry_end_hour", value)

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
        if self._editing and event.key == "enter":
            self.action_save()
            event.stop()
            return
        if self._editing and self._is_hour_filter_field(self._editing):
            if event.key in ("escape", "e"):
                self._editing = None
                self._edit_buffer = ""
                self._refresh_table()
                event.stop()
                return
            if event.key in ("up", "k"):
                self.action_cursor_up()
                event.stop()
                return
            if event.key in ("down", "j"):
                self.action_cursor_down()
                event.stop()
                return
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
            if self._editing and self._is_hour_filter_field(self._editing) and not self._edit_buffer:
                self._editing = None
                self._refresh_table()
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
                if self._is_hour_filter_field(self._editing) and char not in "0123456789":
                    return
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
            if self._is_hour_filter_field(field):
                cleaned = self._edit_buffer.strip()
                value = None
                if cleaned.isdigit():
                    parsed = int(cleaned)
                    value = max(0, min(parsed, 23))
                self._set_hour_filter_value(field.path, value)
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
        if self._editing is not None:
            self._editing = None
            self._edit_buffer = ""
            self._refresh_table()
            return
        self.dismiss(None)
