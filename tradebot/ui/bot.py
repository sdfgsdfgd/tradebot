"""Official durable champion commissioning and portfolio cockpit."""

from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from .bot_screen.live_runs import BotLiveRunsMixin
from .bot_screen.portfolio import BotPortfolioMixin


class BotScreen(BotPortfolioMixin, BotLiveRunsMixin, Screen):
    """Read and control q-owned runs; never evaluate signals or send orders."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("ctrl+t", "app.pop_screen", "Back"),
        ("tab", "cycle_focus", "Focus"),
        ("h", "focus_prev", "Prev"),
        ("l", "focus_next", "Next"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("enter", "activate", "Commission/Inspect"),
        ("space", "context_space", "Commission/Toggle"),
        ("s", "stop_bot", "Safe Stop"),
        ("r", "reload", "Reload"),
        ("x", "replace_live_run", "Replace"),
        ("b", "rebalance_live_runs", "Rebalance"),
    ]
    _PANEL_BY_TABLE_ID = {
        "bot-candidates": "candidates",
        "bot-live-runs": "live_runs",
        "bot-activity": "activity",
        "bot-logs": "logs",
    }
    _PANEL_ORDER = ("candidates", "live_runs", "activity", "logs")

    def __init__(self, client: object | None = None, refresh_sec: float = 5.0) -> None:
        super().__init__()
        # Kept only for the application constructor contract. The official screen
        # deliberately retains no broker client or local execution runtime.
        del client
        self._refresh_sec = max(float(refresh_sec), 1.0)
        self._active_panel = "candidates"
        self._status: str | None = None
        self._refresh_task = None
        self._init_live_runs(Path(__file__).resolve().parents[2])
        self._init_portfolio_tables()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="bot-status"),
            DataTable(id="bot-candidates", zebra_stripes=True, cursor_type="row"),
            DataTable(id="bot-live-runs", zebra_stripes=True, cursor_type="row"),
            DataTable(id="bot-activity", zebra_stripes=True, cursor_type="row"),
            DataTable(id="bot-logs", zebra_stripes=True, cursor_type="row"),
            id="bot-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._status_panel = self.query_one("#bot-status", Static)
        self._candidates_table = self.query_one("#bot-candidates", DataTable)
        self._live_runs_table = self.query_one("#bot-live-runs", DataTable)
        self._activity_table = self.query_one("#bot-activity", DataTable)
        self._logs_table = self.query_one("#bot-logs", DataTable)
        self._setup_candidates_table()
        self._setup_live_runs_table()
        self._setup_timeline_tables()
        self._render_panel_titles()
        await self._refresh_live_runs()
        self._focus_panel("candidates")
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

    async def on_screen_resume(self) -> None:
        await self._refresh_live_runs()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        panel = self._PANEL_BY_TABLE_ID.get(str(event.control.id or ""))
        if panel:
            self._active_panel = panel
        self.action_activate()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        panel = self._PANEL_BY_TABLE_ID.get(str(event.control.id or ""))
        if panel:
            self._active_panel = panel
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

    def _cycle_focus(self, direction: int) -> None:
        try:
            index = self._PANEL_ORDER.index(self._active_panel)
        except ValueError:
            index = 0
        self._focus_panel(self._PANEL_ORDER[(index + direction) % len(self._PANEL_ORDER)])

    def _panel_table(self, panel: str | None = None) -> DataTable:
        return {
            "candidates": self._candidates_table,
            "live_runs": self._live_runs_table,
            "activity": self._activity_table,
            "logs": self._logs_table,
        }[str(panel or self._active_panel)]

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
            "logs": "Persisted Strategy / Execution / Control Log",
        }
        for panel, label in labels.items():
            prefix = "▶ " if panel == self._active_panel else "  "
            self._panel_table(panel).border_title = prefix + label

    def _set_status(self, message: str, *, render_bot: bool = False) -> None:
        del render_bot
        self._status = message
        self._render_status()

    def _render_status(self) -> None:
        if not hasattr(self, "_status_panel"):
            return
        lines = [
            Text(
                self._status
                or "One q-owned chain: crown → cash/preview → selection → capital → run → graduation",
                style="bold",
            )
        ]
        if self._active_panel == "candidates":
            candidate = self._selected_candidate()
            if candidate is not None:
                lines.extend(self._candidate_detail_lines(candidate))
        elif self._active_panel == "live_runs":
            run = self._selected_live_run()
            if run is not None:
                lines.extend(self._live_run_detail_lines(run))
        else:
            event = self._selected_timeline_event()
            if event is not None:
                lines.extend(self._timeline_detail_lines(event))
        self._status_panel.update(Group(*lines))

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
