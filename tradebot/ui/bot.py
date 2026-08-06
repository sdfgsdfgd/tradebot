"""Official durable champion commissioning and portfolio cockpit."""

from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Header, Static, TabbedContent, TabPane

from ..live.strategies import LIVE_STRATEGY_BINDINGS
from .footer import TradebotFooter
from .bot_screen.live_runs import BotLiveRunsMixin
from .bot_screen.portfolio import BotPortfolioMixin
from .bot_screen.traces import BotTraceMixin


_TRACE_TABS = (
    ("ALL", "ALL TRANSITIONS"),
    *(
        (
            binding.champion_symbol.strip().upper(),
            "GOLD · 1OZ"
            if binding.champion_symbol.strip().upper() == "1OZ"
            else binding.champion_symbol.strip().upper(),
        )
        for binding in LIVE_STRATEGY_BINDINGS
        if binding.champion_symbol.strip()
    ),
)
_TRACE_TABLE_IDS = {key: f"bot-trace-{key.lower()}" for key, _label in _TRACE_TABS}
_TRACE_PANE_IDS = {key: f"bot-trace-pane-{key.lower()}" for key, _label in _TRACE_TABS}
_TRACE_KEY_BY_PANE = {value: key for key, value in _TRACE_PANE_IDS.items()}


class BotScreen(BotPortfolioMixin, BotLiveRunsMixin, BotTraceMixin, Screen):
    """Read and control q-owned runs; never evaluate signals or send orders."""

    AUTO_FOCUS = "#bot-trace-all"

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", key_display="esc"),
        Binding("q", "app.pop_screen", show=False),
        Binding("ctrl+t", "app.pop_screen", show=False),
        Binding("p", "toggle_candidates", "Candidates"),
        Binding("ctrl+a", "toggle_candidates", show=False),
        Binding("tab", "cycle_focus", "Panel"),
        Binding("h", "focus_prev", show=False),
        Binding("l", "focus_next", show=False),
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
        Binding("left", "trace_prev", "Contract", key_display="←/→", priority=True),
        Binding("right", "trace_next", show=False, priority=True),
        Binding("[", "trace_prev", show=False),
        Binding("]", "trace_next", show=False),
        Binding("enter", "activate", show=False),
        Binding("space", "context_space", "Act"),
        Binding("s", "stop_bot", "Safe stop"),
        Binding("r", "reload", "Refresh"),
        Binding("x", "replace_live_run", show=False),
        Binding("b", "rebalance_live_runs", show=False),
        Binding("f", "app.open_favorites", show=False),
        Binding("ctrl+f", "app.toggle_search", show=False),
    ]
    _PANEL_BY_TABLE_ID = {
        "bot-candidates": "candidates",
        "bot-live-runs": "live_runs",
        "bot-activity": "activity",
        **{table_id: "traces" for table_id in _TRACE_TABLE_IDS.values()},
    }
    _PANEL_ORDER = ("candidates", "live_runs", "activity", "traces")

    def __init__(self, client: object | None = None, refresh_sec: float = 5.0) -> None:
        super().__init__()
        # Kept only for the application constructor contract. The official screen
        # deliberately retains no broker client or local execution runtime.
        del client
        self._refresh_sec = max(float(refresh_sec), 1.0)
        self._active_panel = "traces"
        self._candidates_visible = True
        self._status: str | None = None
        self._refresh_task = None
        self._init_live_runs(Path(__file__).resolve().parents[2])
        self._init_portfolio_tables()
        self._init_strategy_traces()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="bot-body"):
            yield Static("", id="bot-status")
            yield DataTable(
                id="bot-candidates",
                zebra_stripes=True,
                cursor_type="row",
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            )
            yield DataTable(
                id="bot-live-runs",
                zebra_stripes=True,
                cursor_type="row",
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            )
            yield DataTable(
                id="bot-activity",
                zebra_stripes=True,
                cursor_type="row",
                cursor_foreground_priority="renderable",
                cursor_background_priority="css",
            )
            with TabbedContent(id="bot-traces"):
                for key, label in _TRACE_TABS:
                    with TabPane(label, id=_TRACE_PANE_IDS[key]):
                        yield DataTable(
                            id=_TRACE_TABLE_IDS[key],
                            classes="bot-trace-table",
                            zebra_stripes=True,
                            cursor_type="row",
                            cursor_foreground_priority="renderable",
                            cursor_background_priority="css",
                        )
        yield TradebotFooter()

    async def on_mount(self) -> None:
        self._status_panel = self.query_one("#bot-status", Static)
        self._candidates_table = self.query_one("#bot-candidates", DataTable)
        self._live_runs_table = self.query_one("#bot-live-runs", DataTable)
        self._activity_table = self.query_one("#bot-activity", DataTable)
        self._trace_tabs = self.query_one("#bot-traces", TabbedContent)
        self._trace_tables = {
            key: self.query_one(f"#{table_id}", DataTable)
            for key, table_id in _TRACE_TABLE_IDS.items()
        }
        self._setup_candidates_table()
        self._setup_live_runs_table()
        self._setup_timeline_tables()
        self._setup_trace_tables()
        self._candidates_table.display = self._candidates_visible
        self._render_panel_titles()
        await self._refresh_live_runs()
        self._focus_panel("traces")
        self._refresh_task = self.set_interval(
            max(self._refresh_sec, self._LIVE_RUN_REFRESH_SEC),
            self._refresh_live_runs,
        )

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        if self._live_run_control_task and not self._live_run_control_task.done():
            self._live_run_control_task.cancel()
        if self._candidate_control_task and not self._candidate_control_task.done():
            self._candidate_control_task.cancel()
        await self._live_runs_owner.aclose()

    async def on_screen_resume(self) -> None:
        await self._refresh_live_runs()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        panel = self._PANEL_BY_TABLE_ID.get(str(event.control.id or ""))
        if panel:
            self._active_panel = panel
        self.action_activate()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        panel = self._PANEL_BY_TABLE_ID.get(str(event.control.id or ""))
        # Reconciliation moves cursors in every table. Those synthetic highlight
        # events must never steal the inspector from the table the operator owns.
        if panel is None or not event.control.has_focus:
            return
        if panel == "candidates" and not self._candidates_visible:
            return
        self._active_panel = panel
        self._render_panel_titles()
        self._render_status()

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        if str(event.tabbed_content.id or "") != "bot-traces":
            return
        tabs_have_focus = event.tabbed_content.query_one("ContentTabs").has_focus
        trace_has_focus = any(table.has_focus for table in self._trace_tables.values())
        if not tabs_have_focus and not trace_has_focus:
            return
        self._active_panel = "traces"
        self._render_panel_titles()
        self._render_status()

    def action_reload(self) -> None:
        self._request_live_runs_refresh()
        self._set_status("Reloading q-owned candidates, runs, evidence, and graduation")

    def action_cycle_focus(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def action_toggle_candidates(self) -> None:
        self._candidates_visible = not self._candidates_visible
        self._candidates_table.display = self._candidates_visible
        if not self._candidates_visible and self._active_panel == "candidates":
            self._focus_panel("traces")
        self._render_panel_titles()
        self._set_status(
            f"Champions / Candidates: {'ON' if self._candidates_visible else 'OFF'}"
        )
        self.refresh(layout=True)

    def _cycle_focus(self, direction: int) -> None:
        panels = tuple(
            panel
            for panel in self._PANEL_ORDER
            if panel != "candidates" or self._candidates_visible
        )
        try:
            index = panels.index(self._active_panel)
        except ValueError:
            index = 0
        self._focus_panel(panels[(index + direction) % len(panels)])

    def _panel_table(self, panel: str | None = None) -> DataTable:
        selected = str(panel or self._active_panel)
        if selected == "traces":
            return self._trace_tables[self._active_trace_key()]
        return {
            "candidates": self._candidates_table,
            "live_runs": self._live_runs_table,
            "activity": self._activity_table,
        }[selected]

    def _active_trace_key(self) -> str:
        tabs = getattr(self, "_trace_tabs", None)
        return _TRACE_KEY_BY_PANE.get(str(getattr(tabs, "active", "")), "ALL")

    def _focus_panel(self, panel: str) -> None:
        self._active_panel = panel
        self._render_panel_titles()
        self._panel_table(panel).focus()
        self._render_status()

    def _render_panel_titles(self) -> None:
        if not hasattr(self, "_candidates_table"):
            return
        labels = {
            "candidates": "Champions / Candidates",
            "live_runs": "Official Durable Runs",
            "activity": "Orders / Fills / Reconciliation",
        }
        for panel, label in labels.items():
            prefix = "▶ " if panel == self._active_panel else "  "
            self._panel_table(panel).border_title = prefix + label
        self._trace_tabs.border_title = (
            "▶ " if self._active_panel == "traces" else "  "
        ) + "Strategy Traces · semantic episodes + observation deltas"
        self._trace_tabs.border_subtitle = self._trace_legend()

    def _set_status(self, message: str, *, render_bot: bool = False) -> None:
        del render_bot
        self._status = message
        self._render_status()

    def _render_status(self) -> None:
        if not hasattr(self, "_status_panel"):
            return
        summary = self._status or (
            "One q-owned chain: crown → cash/preview → selection → capital → run → graduation"
        )
        lines = [Text(summary, style="bold")]
        if self._active_panel == "candidates":
            candidate = self._selected_candidate()
            if candidate is not None:
                lines.extend(self._candidate_detail_lines(candidate))
        elif self._active_panel == "live_runs":
            run = self._selected_live_run()
            if run is not None:
                lines.extend(self._live_run_detail_lines(run))
        elif self._active_panel == "activity":
            event = self._selected_timeline_event()
            if event is not None:
                lines.extend(self._timeline_detail_lines(event))
        else:
            trace = self._selected_trace()
            if trace is not None:
                lines = self._trace_context_lines(trace)
                if self._status:
                    lines[0].append("  │  ", style="#34495a")
                    lines[0].append(self._status, style="bold")
        height = 4 if self._active_panel == "traces" else max(3, min(len(lines), 12))
        self._status_panel.styles.height = height
        self._status_panel.update(Group(*lines))

    def action_trace_prev(self) -> None:
        self._cycle_trace(-1)

    def action_trace_next(self) -> None:
        self._cycle_trace(1)

    def _cycle_trace(self, direction: int) -> None:
        if self._active_panel != "traces":
            self._active_panel = "traces"
        keys = [key for key, _label in _TRACE_TABS]
        current = self._active_trace_key()
        index = keys.index(current) if current in keys else 0
        target = keys[(index + direction) % len(keys)]
        self._trace_tabs.active = _TRACE_PANE_IDS[target]
        self._trace_tables[target].focus()
        self._render_panel_titles()
        self._render_status()

    def action_cursor_down(self) -> None:
        self._panel_table().action_cursor_down()
        self._render_status()

    def action_cursor_up(self) -> None:
        self._panel_table().action_cursor_up()
        self._render_status()

    def action_activate(self) -> None:
        if self._active_panel == "candidates":
            self._activate_candidates_panel()
        elif self._active_panel == "live_runs":
            self._activate_live_runs_panel()
        else:
            self._render_status()

    def action_context_space(self) -> None:
        if self._active_panel == "candidates":
            self._activate_candidates_panel()
        elif self._active_panel == "live_runs":
            self.action_toggle_live_run()
        else:
            self._render_status()

    def action_stop_bot(self) -> None:
        if self._active_panel != "live_runs":
            self._set_status("Safe Stop: focus an official durable run")
            return
        self._stop_live_run()

    def _sync_row_marker(self, *_args, **_kwargs) -> None:
        return
