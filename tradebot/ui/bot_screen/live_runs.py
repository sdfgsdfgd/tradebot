"""Bot Trade cockpit for persistent selected strategy runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from rich.text import Text

from ...live.runs import LiveRunCockpit
from ...research.live_graduation import validate_live_graduation_receipt
from ...research.xsp_live_capital import XSP_LIVE_RUN_BINDING
from ..common import _pnl_text
from .formatting import _center_table_row


class BotLiveRunsMixin:
    _LIVE_RUN_REFRESH_SEC = 5.0

    def _init_live_runs(self, repository_root: Path) -> None:
        self._live_runs_owner = LiveRunCockpit(
            repository_root=repository_root,
            capital_plan_path=repository_root / "db/calibration/live_capital_plan.json",
            bindings=(XSP_LIVE_RUN_BINDING,),
            graduation_directory=Path("db/calibration/live_graduation"),
            graduation_validator=validate_live_graduation_receipt,
        )
        self._live_runs_snapshot: dict[str, object] | None = None
        self._live_run_rows: list[dict[str, object]] = []
        self._live_runs_refreshing = False
        self._live_runs_refresh_task = None
        self._live_run_control_task: asyncio.Task | None = None

    def _setup_live_runs_table(self) -> None:
        self._live_runs_table.clear(columns=True)
        for label, width in (
            ("Champion / execution", 36),
            ("Owner", 13),
            ("Capital", 16),
            ("Position", 18),
            ("Net", 12),
            ("DD", 10),
            ("Fills/Tr", 9),
            ("Graduation", 14),
            ("Safety", 11),
            ("Controls", 22),
        ):
            self._live_runs_table.add_column(label, width=width)

    @staticmethod
    def _live_run_state_cell(run: Mapping[str, object]) -> Text:
        state = str(run.get("state") or "UNKNOWN")
        style = {
            "RUNNING": "bold #73d89e",
            "BUSY": "bold #62b0ff",
            "PAUSED": "bold #b8c0cb",
            "UNSAFE_PAUSED": "bold #f2b36f",
            "BROKEN": "bold #ff7070",
            "QUARANTINED": "bold #ff7070",
        }.get(state, "bold #d6a56f")
        return Text(state, style=style)

    @staticmethod
    def _live_run_position(run: Mapping[str, object]) -> str:
        positions = run.get("positions")
        if not isinstance(positions, Mapping):
            return "unknown"
        active = []
        for symbol, raw in sorted(positions.items()):
            try:
                quantity = float(raw or 0)
            except (TypeError, ValueError):
                return "invalid"
            if abs(quantity) >= 1e-9:
                shown = str(int(quantity)) if quantity.is_integer() else f"{quantity:.2f}"
                active.append(f"{shown} {symbol}")
        return "flat" if not active else " + ".join(active)

    @staticmethod
    def _live_run_graduation_cell(run: Mapping[str, object]) -> Text:
        graduation = run.get("graduation")
        if not isinstance(graduation, Mapping):
            return Text("UNKNOWN", style="bold #d6a56f")
        verdict = str(graduation.get("verdict") or "UNKNOWN")
        target = str(graduation.get("target") or "")
        label = verdict + (f" {target}" if target else "")
        style = {
            "PROMOTE": "bold #73d89e",
            "HOLD": "bold #f2b36f",
            "PENDING": "#b8c0cb",
            "REVISE": "bold #f2b36f",
            "QUARANTINE": "bold #ff7070",
            "STOP": "bold #ff7070",
        }.get(verdict, "bold #d6a56f")
        return Text(label, style=style)

    @staticmethod
    def _live_run_controls(run: Mapping[str, object]) -> str:
        controls = run.get("controls")
        if not isinstance(controls, Mapping):
            return "locked"
        timer = run.get("timer")
        running = isinstance(timer, Mapping) and timer.get("active_state") == "active"
        primary = "Space:Pause" if running else "Space:Start"
        replace = controls.get("REPLACE")
        rebalance = controls.get("REBALANCE")
        suffix = []
        suffix.append("x:R" if isinstance(replace, Mapping) and replace.get("status") == "ALLOW" else "x:gate")
        suffix.append("b:B" if isinstance(rebalance, Mapping) and rebalance.get("status") == "ALLOW" else "b:gate")
        return f"{primary} {' '.join(suffix)}"

    def _render_live_runs_table(self, snapshot: Mapping[str, object]) -> None:
        prior_row = self._live_runs_table.cursor_coordinate.row
        prior_id = None
        if 0 <= prior_row < len(self._live_run_rows):
            prior_id = self._live_run_rows[prior_row].get("sleeve_id")
        self._live_runs_table.clear()
        self._live_run_rows = []
        runs = snapshot.get("runs")
        for run in runs if isinstance(runs, list) else ():
            if not isinstance(run, Mapping):
                continue
            allocation = run.get("allocation")
            economics = run.get("economics")
            safety = run.get("safety")
            allocation = allocation if isinstance(allocation, Mapping) else {}
            economics = economics if isinstance(economics, Mapping) else {}
            safety = safety if isinstance(safety, Mapping) else {}
            limit = float(allocation.get("limit_cents") or 0) / 100
            weight = float(allocation.get("weight_bps") or 0) / 100
            fill_count = economics.get("fill_count")
            closed_trades = economics.get("closed_trades")
            breaches = list(safety.get("breaches") or ())
            safe = safety.get("valid") is True and not breaches
            safety_cell = Text(
                "PASS" if safe else "CHECK",
                style="bold #73d89e" if safe else "bold #ff7070",
            )
            net = economics.get("run_net_usd")
            drawdown = economics.get("drawdown_usd")
            self._live_runs_table.add_row(
                *_center_table_row(
                    str(run.get("label") or run.get("strategy_id") or "unknown"),
                    self._live_run_state_cell(run),
                    f"${limit:,.2f} · {weight:.0f}%",
                    self._live_run_position(run),
                    _pnl_text(float(net)) if net is not None else Text("-", style="dim"),
                    f"${float(drawdown):,.2f}" if drawdown is not None else "-",
                    f"{fill_count if fill_count is not None else '-'}/{closed_trades if closed_trades is not None else '-'}",
                    self._live_run_graduation_cell(run),
                    safety_cell,
                    self._live_run_controls(run),
                ),
                key=f"live:{run.get('sleeve_id')}",
            )
            self._live_run_rows.append(dict(run))
        if prior_id is not None:
            for index, run in enumerate(self._live_run_rows):
                if run.get("sleeve_id") == prior_id:
                    self._live_runs_table.cursor_coordinate = (index, 0)
                    break
        elif self._live_run_rows:
            self._live_runs_table.cursor_coordinate = (0, 0)
        self._sync_row_marker(self._live_runs_table, force=True)

    async def _refresh_live_runs(self) -> None:
        if self._live_runs_refreshing:
            return
        self._live_runs_refreshing = True
        try:
            snapshot = await asyncio.to_thread(self._live_runs_owner.snapshot)
            self._live_runs_snapshot = snapshot
            self._render_live_runs_table(snapshot)
            if self._active_panel == "live_runs":
                self._render_status()
        except Exception as exc:  # pragma: no cover - operator surface
            self._set_status(f"Live Runs: {exc}")
        finally:
            self._live_runs_refreshing = False

    def _request_live_runs_refresh(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._refresh_live_runs())

    def _selected_live_run(self) -> dict[str, object] | None:
        row = self._live_runs_table.cursor_coordinate.row
        if row < 0 or row >= len(self._live_run_rows):
            return None
        return self._live_run_rows[row]

    def _activate_live_runs_panel(self) -> None:
        run = self._selected_live_run()
        if run is None:
            self._set_status("Live Runs: no commissioned run selected")
            return
        self._set_status(
            f"{run.get('label')}: Space=Start/Pause; x=Replace gate; b=Rebalance gate"
        )

    def action_toggle_live_run(self) -> None:
        run = self._selected_live_run()
        if run is None:
            self._set_status("Live Runs: no commissioned run selected")
            return
        timer = run.get("timer")
        running = isinstance(timer, Mapping) and timer.get("active_state") == "active"
        self._request_live_run_control("STOP" if running else "START")

    def action_replace_live_run(self) -> None:
        if self._active_panel != "live_runs":
            self._set_status("Replace: focus Live Runs")
            return
        self._request_live_run_control("REPLACE")

    def action_rebalance_live_runs(self) -> None:
        if self._active_panel != "live_runs":
            self._set_status("Rebalance: focus Live Runs")
            return
        self._request_live_run_control("REBALANCE")

    def _stop_live_run(self) -> None:
        self._request_live_run_control("STOP")

    def _request_live_run_control(self, action: str) -> None:
        run = self._selected_live_run()
        if run is None:
            self._set_status(f"{action.title()}: no commissioned run selected")
            return
        if self._live_run_control_task and not self._live_run_control_task.done():
            self._set_status("Live Runs: another control transaction is active")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("Live Runs: no event loop")
            return
        sleeve_id = str(run.get("sleeve_id") or "")
        self._live_run_control_task = loop.create_task(
            self._control_live_run(sleeve_id, action)
        )

    async def _control_live_run(self, sleeve_id: str, action: str) -> None:
        try:
            receipt = await asyncio.to_thread(
                self._live_runs_owner.control,
                sleeve_id,
                action,
            )
            decision = receipt.get("decision")
            status = (
                str(decision.get("status") or "")
                if isinstance(decision, Mapping)
                else ""
            )
            self._journal_write(
                event="DURABLE_RUN_CONTROL",
                reason=action,
                data={"sleeve_id": sleeve_id, "status": status},
            )
            self._set_status(f"{action.title()} {sleeve_id}: {status}")
        except Exception as exc:
            self._set_status(f"{action.title()} blocked: {exc}")
        await self._refresh_live_runs()

    def _live_run_detail_lines(self, run: Mapping[str, object]) -> list[Text]:
        allocation = run.get("allocation")
        economics = run.get("economics")
        safety = run.get("safety")
        graduation = run.get("graduation")
        allocation = allocation if isinstance(allocation, Mapping) else {}
        economics = economics if isinstance(economics, Mapping) else {}
        safety = safety if isinstance(safety, Mapping) else {}
        graduation = graduation if isinstance(graduation, Mapping) else {}
        run_id = str(run.get("run_id") or "")
        lines = [
            Text(""),
            Text("Official commissioned run", style="bold"),
            Text(
                f"{run.get('label')}  id={run_id[:12]}…  owner={run.get('state')}",
                style="dim",
            ),
            Text(
                f"Capital=${float(allocation.get('limit_cents') or 0) / 100:,.2f} "
                f"({float(allocation.get('weight_bps') or 0) / 100:.0f}%)  "
                f"Position={self._live_run_position(run)}  "
                f"Cash=${float(run.get('settled_cash_usd') or 0):,.2f}",
                style="dim",
            ),
            Text(
                f"Net=${float(economics.get('run_net_usd') or 0):,.2f}  "
                f"Realized=${float(economics.get('run_realized_net_usd') or 0):,.2f}  "
                f"Mark=${float(economics.get('open_mark_net_usd') or 0):,.2f}  "
                f"Costs=${float(economics.get('run_cost_usd') or 0):,.2f}  "
                f"DD=${float(economics.get('drawdown_usd') or 0):,.2f}  "
                f"Fills/Trades={economics.get('fill_count', '-')}/{economics.get('closed_trades', '-')}",
                style="dim",
            ),
            Text(
                f"Graduation={graduation.get('verdict', 'UNKNOWN')} "
                f"{graduation.get('target') or ''}  "
                f"Safety={'PASS' if safety.get('valid') is True and not safety.get('breaches') else 'CHECK'}",
                style="dim",
            ),
        ]
        lines.extend(self._live_run_hawkeye_lines(run))
        lines.append(
            Text(
                "Space resumes/pauses q's timer only. Stop refuses non-flat, open, pending, or busy runs. "
                "Replace/Rebalance require immutable successor artifacts.",
                style="dim",
            )
        )
        reasons = list(graduation.get("reasons") or ())
        if reasons:
            lines.append(Text("Graduation: " + "; ".join(str(value) for value in reasons), style="#f2b36f"))
        errors = list(run.get("errors") or ())
        if errors:
            lines.append(Text("Run errors: " + "; ".join(str(value) for value in errors), style="bold #ff7070"))
        return lines

    @staticmethod
    def _live_run_hawkeye_lines(run: Mapping[str, object]) -> list[Text]:
        context = run.get("execution_state_context")
        if not isinstance(context, Mapping) or not context:
            return []
        impulse = context.get("directional_impulse")
        daily_context = context.get("daily_context_state")
        pressure = context.get("fundamental_pressure")
        entry_control = context.get("entry_control")
        impulse = impulse if isinstance(impulse, Mapping) else {}
        daily_context = daily_context if isinstance(daily_context, Mapping) else {}
        pressure = pressure if isinstance(pressure, Mapping) else {}
        entry_control = entry_control if isinstance(entry_control, Mapping) else {}
        daily = daily_context.get("state")
        daily = daily if isinstance(daily, Mapping) else {}
        horizons = impulse.get("horizons")
        horizons = [value for value in horizons if isinstance(value, Mapping)] if isinstance(horizons, list) else []

        def compact(value: object, digits: int) -> str:
            try:
                return f"{float(value):+.{digits}f}"
            except (TypeError, ValueError):
                return "-"

        angles = " ".join(
            f"{int(float(value.get('elapsed_minutes') or 0))}m:{compact(value.get('slope_angle_deg'), 1)}°"
            for value in horizons
        )
        velocities = " ".join(
            f"{int(float(value.get('elapsed_minutes') or 0))}m:{compact(value.get('slope_velocity_pct_per_bar'), 4)}"
            for value in horizons
        )
        decision = run.get("latest_decision")
        decision = decision if isinstance(decision, Mapping) else {}
        lines = [
            Text(
                "Hawkeye: "
                f"{decision.get('status') or '-'} / {decision.get('reason') or impulse.get('abstain_reason') or '-'}  "
                f"trend={impulse.get('trend_state') or '-'} direction={impulse.get('direction') or '-'} "
                f"coherence={compact(impulse.get('coherence'), 2)}",
                style="#8fbfff",
            )
        ]
        controls = entry_control.get("controls")
        controls = controls if isinstance(controls, list) else []
        lines.append(
            Text(
                "Gate: "
                f"source={entry_control.get('source') or '-'} "
                f"proposed={entry_control.get('proposed_direction') or '-'} "
                f"blocked={entry_control.get('blocked_by') or '-'} "
                f"controls={','.join(str(value) for value in controls) or '-'}  "
                f"signal={context.get('signal_session') or '-'}/"
                f"{context.get('signal_bar_ts') or '-'} "
                f"age={context.get('signal_snapshot_age_bars', '-')} bars",
                style="dim",
            )
        )
        if angles:
            lines.append(Text("Slope angles  " + angles, style="dim"))
        if velocities:
            lines.append(Text("Slope velocity  " + velocities, style="dim"))
        lines.append(
            Text(
                "ATR "
                f"ratio={compact(impulse.get('atr_ratio'), 3)} "
                f"velocity={compact(impulse.get('atr_velocity_pct'), 5)} "
                f"acceleration={compact(impulse.get('atr_acceleration_pct'), 5)}",
                style="dim",
            )
        )
        directions = daily.get("directions")
        directions = directions if isinstance(directions, Mapping) else {}
        if directions:
            direction_text = " ".join(
                f"{name}:{directions.get(name, '-')}"
                for name in ("5", "10", "21", "42", "63", "84")
            )
            lines.append(
                Text(
                    "Long context  "
                    f"{direction_text}  transition={daily.get('transition') or '-'} "
                    f"hard={daily.get('hard_direction') or '-'} soft={daily.get('soft_direction') or '-'} "
                    f"TR={daily.get('tr_phase') or '-'}",
                    style="dim",
                )
            )
        if pressure:
            lines.append(
                Text(
                    "News attribution  "
                    f"pressure={compact(pressure.get('signed_pressure'), 4)} "
                    f"delta={compact(pressure.get('pressure_delta'), 4)} "
                    f"velocity/h={compact(pressure.get('pressure_velocity_per_hour'), 6)} "
                    f"confidence={compact(pressure.get('confidence'), 2)}",
                    style="dim",
                )
            )
        return lines
