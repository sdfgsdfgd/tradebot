"""Bot hub screen shell and capability composition."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import fields
from pathlib import Path

from ib_insync import PortfolioItem
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..chart_data.series import BarSeries
from ..client import IBKRClient
from ..spot.codec import effective_filters_payload as _effective_filters_payload
from .bot_engine_runtime import BotEngineRuntimeMixin
from .bot_journal import BotJournal
from .bot_models import (
    _BotConfigResult,
    _BotInstance,
    _BotOrder,
    _BotPreset,
    _PresetHeader,
)
from .bot_order_builder import BotOrderBuilderMixin
from .bot_screen.config import BotConfigScreen
from .bot_screen.formatting import (
    _InstancePnlState,
    _center_table_cell,
    _filters_for_group,
    _missing_signal_transport_keys,
)
from .bot_screen.logs import BotLogsMixin
from .bot_screen.orders import BotOrdersMixin
from .bot_screen.positions import BotPositionsMixin
from .bot_screen.presets import BotPresetsMixin
from .bot_screen.signal_data import BotSignalDataMixin
from .bot_screen.signal_snapshot import BotSignalSnapshotMixin
from .bot_signal_runtime import BotSignalRuntimeMixin

class BotScreen(
    BotOrderBuilderMixin,
    BotSignalRuntimeMixin,
    BotEngineRuntimeMixin,
    BotPresetsMixin,
    BotPositionsMixin,
    BotLogsMixin,
    BotSignalDataMixin,
    BotSignalSnapshotMixin,
    BotOrdersMixin,
    Screen,
):
    _LOG_CAP = 99_999
    _UNREAL_STICKY_SEC = 20.0
    _CLOSES_RETRY_SEC = 30.0
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("ctrl+t", "app.pop_screen", "Back"),
        ("ctrl+a", "toggle_presets", "Presets"),
        ("p", "toggle_presets", "Presets"),
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
    _SIGNAL_HEAL_BACKOFF_BASE_SEC = 8.0
    _SIGNAL_HEAL_BACKOFF_MAX_SEC = 180.0
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
        self._log_row_keys: list[str] = []
        self._log_seq = 0
        self._log_max_rows = int(self._LOG_CAP)
        self._logs_follow_tail = True
        self._status: str | None = None
        self._refresh_task = None
        self._order_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._cancel_task: asyncio.Task | None = None
        self._tracked_conids: set[int] = set()
        self._session_closes_by_con_id: dict[int, tuple[float | None, float | None]] = {}
        self._session_close_1ago_by_con_id: dict[int, float | None] = {}
        self._closes_loading: set[int] = set()
        self._closes_retry_at_by_con_id: dict[int, float] = {}
        self._instance_pnl_state_by_id: dict[int, _InstancePnlState] = {}
        self._instance_live_total_by_id: dict[int, float | None] = {}
        self._last_unreal_by_conid: dict[int, tuple[float, float]] = {}
        self._stream_refresh_task: asyncio.Task | None = None
        self._stream_dirty = False
        self._active_panel = "presets"
        self._refresh_lock = asyncio.Lock()
        self._scope_all = False
        self._journal = BotJournal(Path(__file__).resolve().parent / "out")
        self._signal_heal_backoff: dict[tuple[int, str, str, bool], dict[str, float | int]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="bot-status"),
            DataTable(
                id="bot-presets",
                zebra_stripes=True,
                show_row_labels=False,
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            ),
            DataTable(
                id="bot-instances",
                zebra_stripes=True,
                show_row_labels=False,
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            ),
            DataTable(
                id="bot-orders",
                zebra_stripes=True,
                show_row_labels=False,
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            ),
            DataTable(
                id="bot-logs",
                zebra_stripes=True,
                show_row_labels=False,
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            ),
            id="bot-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._presets_table = self.query_one("#bot-presets", DataTable)
        self._status_panel = self.query_one("#bot-status", Static)
        self._orders_table = self.query_one("#bot-orders", DataTable)
        self._instances_table = self.query_one("#bot-instances", DataTable)
        self._logs_table = self.query_one("#bot-logs", DataTable)
        self._render_panel_titles()
        self._presets_table.cursor_type = "row"
        self._orders_table.cursor_type = "row"
        self._instances_table.cursor_type = "row"
        self._logs_table.cursor_type = "row"
        self._setup_tables()
        self._load_leaderboard()
        self._apply_presets_layout()
        await self._refresh_positions()
        self._refresh_instances_table()
        self._refresh_orders_table()
        self._refresh_logs_table()
        self._journal_write(event="BOOT", reason=None, data={"refresh_sec": self._refresh_sec})
        self._render_status()
        self._focus_panel("presets")
        self._sync_panel_markers(force=True)
        self._client.add_stream_listener(self._on_client_stream_update)
        self._refresh_task = self.set_interval(self._refresh_sec, self._on_refresh_tick)

    async def on_unmount(self) -> None:
        self._client.remove_stream_listener(self._on_client_stream_update)
        if self._refresh_task:
            self._refresh_task.stop()
        if self._stream_refresh_task and not self._stream_refresh_task.done():
            self._stream_refresh_task.cancel()
        for con_id in list(self._tracked_conids):
            self._client.release_ticker(con_id, owner="bot")
        self._tracked_conids.clear()
        self._journal_write(event="SHUTDOWN", reason=None, data=None)
        try:
            self._journal.close()
        except Exception:
            pass

    async def on_screen_resume(self) -> None:
        if self._refresh_lock.locked():
            return
        await self._refresh_positions()
        self._render_status()

    def _on_client_stream_update(self) -> None:
        self._stream_dirty = True
        if self._stream_refresh_task and not self._stream_refresh_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._stream_refresh_task = loop.create_task(self._flush_stream_refresh())

    async def _flush_stream_refresh(self) -> None:
        while self._stream_dirty:
            self._stream_dirty = False
            if self._refresh_lock.locked():
                await asyncio.sleep(min(0.1, self._refresh_sec))
                continue
            # Stream updates are frequent; keep logs stable by avoiding dependent
            # table rebuilds (orders/logs) from instance refresh.
            self._refresh_instances_table(refresh_dependents=False)
            self._refresh_orders_table()
            self._render_status()
            await asyncio.sleep(0)

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
        if len(self._log_events) > int(self._LOG_CAP):
            self._log_events = self._log_events[-int(self._LOG_CAP) :]
        if hasattr(self, "_logs_table"):
            self._append_log_entry(entry)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; hook it so Enter arms/sends.
        table_id = str(getattr(event.control, "id", "") or "")
        panel = self._PANEL_BY_TABLE_ID.get(table_id)
        if panel:
            self._active_panel = str(panel)
        self.action_activate()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = event.control
        table_id = str(getattr(table, "id", "") or "")
        if table_id in self._PANEL_BY_TABLE_ID:
            self._sync_row_marker(table)

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

    def _apply_presets_layout(self) -> None:
        self._presets_table.display = self._presets_visible
        self._orders_table.styles.height = "1fr"
        self._logs_table.styles.height = "1fr" if self._presets_visible else "3fr"

    def action_toggle_presets(self) -> None:
        self._presets_visible = not self._presets_visible
        self._apply_presets_layout()
        if not self._presets_visible and self._active_panel == "presets":
            self._focus_panel("instances")
        elif self._presets_visible and self._active_panel != "presets":
            self._focus_panel("presets")
        self._render_panel_titles()
        self._set_status(f"Presets: {'ON' if self._presets_visible else 'OFF'}")
        self._sync_panel_markers(force=True)
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

    def _sync_panel_markers(self, *, force: bool = False) -> None:
        return

    def _sync_row_marker(self, table: DataTable, *, force: bool = False) -> None:
        # DataTable row cursor highlight is sufficient; avoid label churn/flicker.
        return

    @staticmethod
    def _set_row_marker(table: DataTable, row_index: int, *, active: bool) -> None:
        return

    def _render_panel_titles(self) -> None:
        labels = {
            "presets": "Presets",
            "instances": "Instances",
            "orders": "Orders / Positions",
            "logs": "Logs",
        }
        for panel in self._PANEL_ORDER:
            table = self._panel_table(panel)
            prefix = "▶ " if panel == self._active_panel else "  "
            if panel != "logs":
                table.border_title = f"{prefix}{labels[panel]}"
                continue
            title = Text(prefix)
            title.append("Logs ", style="bold")
            title.append("[", style="dim")
            if self._logs_follow_tail:
                title.append("FOLLOW", style="bold #73d89e")
            else:
                title.append("PAUSED", style="bold #f2b36f")
            title.append("]", style="dim")
            table.border_title = title

    def _set_logs_follow_tail(self, follow_tail: bool) -> None:
        next_value = bool(follow_tail)
        if self._logs_follow_tail == next_value:
            return
        self._logs_follow_tail = next_value
        self._render_panel_titles()

    def _event_targets_logs_table(self, event: events.Event) -> bool:
        if not getattr(self, "_logs_table", None):
            return False
        target = getattr(event, "control", None) or getattr(event, "widget", None)
        while target is not None:
            if target is self._logs_table:
                return True
            if str(getattr(target, "id", "") or "") == "bot-logs":
                return True
            target = getattr(target, "parent", None)
        return False

    def _sync_logs_follow_tail_from_viewport(self) -> None:
        if not getattr(self, "_logs_table", None):
            return
        self._set_logs_follow_tail(self._logs_at_tail())

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._event_targets_logs_table(event):
            self.call_after_refresh(self._sync_logs_follow_tail_from_viewport)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._event_targets_logs_table(event):
            self.call_after_refresh(self._sync_logs_follow_tail_from_viewport)

    def _focus_panel(self, panel: str) -> None:
        self._active_panel = panel
        self._render_panel_titles()
        table = self._panel_table(panel)
        table.focus()
        self._sync_row_marker(table, force=True)
        if panel == "logs":
            self._set_logs_follow_tail(self._logs_at_tail())
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
        self._rebuild_presets_table()
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
        self._rebuild_presets_table()
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
        self._rebuild_presets_table()
        self._refresh_orders_table()
        self._journal_write(event="STOP_INSTANCE", instance=instance, reason=None, data=None)
        self._set_status(f"Stopped instance {instance.instance_id}: paused + cleared orders")

    def _kill_all(self) -> None:
        for instance in self._instances:
            instance.state = "PAUSED"
        self._orders.clear()
        self._refresh_instances_table()
        self._rebuild_presets_table()
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        order.status = "CANCELING"
        order.cancel_requested_at = loop.time() if loop is not None else None
        self._journal_write(event="CANCEL_REQUEST", order=order, reason=None, data=None)
        self._refresh_orders_table()
        self._set_status(f"Canceling #{order.order_id or 0}...", render_bot=True)
        try:
            await self._client.cancel_trade(trade)
            self._journal_write(event="CANCEL_SENT", order=order, reason=None, data=None)
            self._set_status(f"Cancel sent #{order.order_id or 0}")
        except Exception as exc:
            order.status = "WORKING"
            order.cancel_requested_at = None
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

    def _heal_strategy_filters_payload(self, *, strategy: dict, base_filters: dict | None) -> dict | None:
        """Normalize milestone-style payloads into live UI shape.

        Live UI stores filters separately from strategy. Milestone payloads may embed filters
        under `strategy.filters` and (historically) may accidentally place filter-shaped keys at
        the strategy root (e.g. `ratsv_*`). This method:
        - resolves the effective filters payload (group + strategy + hoisted keys)
        - removes embedded filter payload from the strategy dict so UI edits are unambiguous
        """
        effective = _effective_filters_payload(
            group_filters=base_filters if isinstance(base_filters, dict) else None,
            strategy=strategy if isinstance(strategy, dict) else None,
        )
        if not isinstance(strategy, dict):
            return effective

        nested = strategy.get("filters")
        if isinstance(nested, dict):
            strategy.pop("filters", None)

        try:
            from ..knobs.models import FiltersConfig as _FiltersConfig
        except Exception:
            _FiltersConfig = None

        if _FiltersConfig is None:
            return effective

        filter_keys = {field.name for field in fields(_FiltersConfig)}
        for key in list(strategy.keys()):
            if key in filter_keys:
                strategy.pop(key, None)
        return effective

    def _heal_instance_effective_filters(self, instance: _BotInstance) -> None:
        strategy = instance.strategy
        if not isinstance(strategy, dict):
            return
        base_filters = instance.filters if isinstance(instance.filters, dict) else None

        try:
            from ..knobs.models import FiltersConfig as _FiltersConfig
        except Exception:
            _FiltersConfig = None

        root_filter_keys: list[str] = []
        if _FiltersConfig is not None:
            filter_keys = {field.name for field in fields(_FiltersConfig)}
            root_filter_keys = sorted([key for key in strategy.keys() if key in filter_keys])

        embedded_filters = strategy.get("filters") if isinstance(strategy.get("filters"), dict) else None
        if embedded_filters is None and not root_filter_keys:
            return

        effective = self._heal_strategy_filters_payload(strategy=strategy, base_filters=base_filters)
        instance.filters = effective
        self._journal_write(
            event="INSTANCE_HEAL",
            instance=instance,
            reason="EFFECTIVE_FILTERS",
            data={
                "migrated_strategy_filters": bool(embedded_filters is not None),
                "hoisted_strategy_filter_keys": root_filter_keys or None,
            },
        )

    def _open_config_for_preset(self, preset: _BotPreset) -> None:
        entry = preset.entry
        strategy = copy.deepcopy(entry.get("strategy", {}) or {})
        if self._strategy_instrument(strategy) == "spot":
            missing_transport = _missing_signal_transport_keys(strategy)
            if missing_transport:
                missing_text = ", ".join(missing_transport)
                self._set_status(
                    f"Preset artifact missing {missing_text}; refusing silent signal transport defaults"
                )
                return
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
        group_filters = _filters_for_group(self._payload, preset.group) if self._payload else None
        filters = self._heal_strategy_filters_payload(strategy=strategy, base_filters=group_filters)
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
                preset_key=preset.key,
                metrics=entry.get("metrics"),
            )
            self._next_instance_id += 1
            self._instances.append(instance)
            self._refresh_instances_table()
            self._rebuild_presets_table()
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
        self._heal_instance_effective_filters(instance)

        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._set_status("Config: cancelled")
                return
            instance.group = result.group
            instance.symbol = result.symbol
            instance.strategy = result.strategy
            instance.filters = result.filters
            self._refresh_instances_table()
            self._rebuild_presets_table()
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

    def _setup_tables(self) -> None:
        self._instances_table.clear(columns=True)
        self._instances_table.add_column(_center_table_cell("ID"), width=4)
        self._instances_table.add_column(_center_table_cell("Strategy"), width=26)
        self._instances_table.add_column(_center_table_cell("DTE"), width=5)
        self._instances_table.add_column(_center_table_cell("State"), width=9)
        self._instances_table.add_column(_center_table_cell("BT PnL"), width=12)
        self._instances_table.add_column(_center_table_cell("Unreal"), width=16)
        self._instances_table.add_column(_center_table_cell("Realized"), width=12)
        self._instances_table.add_column(_center_table_cell("Total"), width=12)

        self._orders_table.clear(columns=True)
        self._orders_table.add_columns(
            *(
                _center_table_cell(label)
                for label in (
                    "When ET",
                    "Inst",
                    "Side",
                    "Qty",
                    "Contract",
                    "Lmt",
                    "B/A",
                    "Unreal",
                    "Realized",
                )
            )
        )

        self._logs_table.clear(columns=True)
        self._logs_table.add_columns("When ET", "Inst", "Sym", "Event", "Reason", "Msg")

    def _cursor_move(self, direction: int) -> None:
        table = self._panel_table()
        if direction > 0:
            table.action_cursor_down()
        else:
            table.action_cursor_up()
        if self._active_panel == "logs":
            self._set_logs_follow_tail(self._logs_at_tail())
        if self._active_panel == "instances" and not self._scope_all:
            self._refresh_orders_table()
            self._refresh_logs_table()
        self._render_status()

    @staticmethod
    def _signal_series_list(series: BarSeries | list | None) -> list:
        if series is None:
            return []
        if isinstance(series, BarSeries):
            return series.as_list()
        return list(series)

    @staticmethod
    def _signal_series_signature(series: BarSeries | list | None) -> tuple[object, ...]:
        bars = BotScreen._signal_series_list(series)
        if not bars:
            return (0, None, None, None, None)
        first = bars[0]
        last = bars[-1]
        return (
            int(len(bars)),
            first.ts.isoformat(),
            last.ts.isoformat(),
            round(float(getattr(first, "open", 0.0) or 0.0), 8),
            round(float(getattr(last, "close", 0.0) or 0.0), 8),
        )
