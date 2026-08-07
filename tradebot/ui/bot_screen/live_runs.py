"""Bot Trade cockpit for persistent selected strategy runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text

from ...live.portfolio_endpoint import LivePortfolioEndpoint
from ..common import _pnl_text
from .formatting import _center_table_row


class BotLiveRunsMixin:
    _LIVE_RUN_REFRESH_SEC = 5.0
    _LIVE_TIMELINE_LIMIT = 250
    _LIVE_TRACE_LIMIT_PER_STRATEGY = 2_000

    def _init_live_runs(self, repository_root: Path) -> None:
        self._live_runs_owner = LivePortfolioEndpoint.default(repository_root)
        self._live_runs_snapshot: dict[str, object] | None = None
        self._live_run_rows: list[dict[str, object]] = []
        self._live_runs_refreshing = False
        self._live_runs_refresh_task = None
        self._live_run_control_task: asyncio.Task | None = None
        self._live_runs_view_id: str | None = None

    def _setup_live_runs_table(self) -> None:
        self._live_runs_table.clear(columns=True)
        for label, width in (
            ("Champion / execution", 36),
            ("Owner", 13),
            ("Capital", 16),
            ("Position", 18),
            ("Run / campaign", 18),
            ("DD", 10),
            ("Fills/Tr", 9),
            ("Graduation", 24),
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
    def _graduation_age_hours(
        graduation: Mapping[str, object],
        *,
        now: datetime | None = None,
    ) -> float | None:
        value = graduation.get("cutoff_utc")
        if not value:
            return None
        try:
            cutoff = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        current = now or datetime.now(tz=timezone.utc)
        return max(0.0, (current.astimezone(timezone.utc) - cutoff).total_seconds() / 3600)

    @classmethod
    def _live_run_graduation_cell(cls, run: Mapping[str, object]) -> Text:
        graduation = run.get("graduation")
        if not isinstance(graduation, Mapping):
            return Text("UNKNOWN", style="bold #d6a56f")
        verdict = str(graduation.get("verdict") or "UNKNOWN")
        target = str(graduation.get("target") or "")
        if verdict == "PENDING" and not graduation.get("receipt_id"):
            return Text("NO RECEIPT", style="bold #b8c0cb")
        age = cls._graduation_age_hours(graduation)
        label = f"{target} {verdict}".strip()
        if age is not None:
            label += f" · {age:.0f}h old"
        style = {
            "PROMOTE": "bold #73d89e",
            "HOLD": "bold #f2b36f",
            "PENDING": "#b8c0cb",
            "REVISE": "bold #f2b36f",
            "QUARANTINE": "bold #ff7070",
            "STOP": "bold #ff7070",
        }.get(verdict, "bold #d6a56f")
        if verdict == "HOLD" and age is not None and age >= 6:
            style = "bold #ff7070"
        return Text(label, style=style)

    @staticmethod
    def _graduation_ladder(graduation: Mapping[str, object]) -> str:
        target = str(graduation.get("target") or "")
        verdict = str(graduation.get("verdict") or "PENDING")
        order = ("24h", "48h", "five_session_week")
        labels = {"24h": "24h", "48h": "48h", "five_session_week": "5 sessions"}
        if target not in order:
            return "24h ○ → 48h locked → 5 sessions locked"
        index = order.index(target)
        cells = []
        for position, milestone in enumerate(order):
            state = "✓" if position < index else verdict if position == index else "locked"
            cells.append(f"{labels[milestone]} {state}")
        return " → ".join(cells)

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

    @staticmethod
    def _live_run_capital(allocation: Mapping[str, object]) -> str:
        package_id = str(allocation.get("package_id") or "")
        if package_id:
            cash = float(allocation.get("cash_debit_cents") or 0) / 100
            initial = float(allocation.get("initial_margin_base_cents") or 0) / 100
            if allocation.get("capital_kind") == "FUTURES_MARGIN":
                return f"{package_id} · margin {initial:,.0f}"
            return f"${cash:,.2f} · {package_id}"
        limit = float(allocation.get("limit_cents") or 0) / 100
        weight = float(allocation.get("weight_bps") or 0) / 100
        return f"${limit:,.2f} · {weight:.0f}%"

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
            campaign = run.get("campaign_economics")
            safety = run.get("safety")
            allocation = allocation if isinstance(allocation, Mapping) else {}
            economics = economics if isinstance(economics, Mapping) else {}
            campaign = campaign if isinstance(campaign, Mapping) else {}
            safety = safety if isinstance(safety, Mapping) else {}
            fill_count = economics.get("fill_count")
            closed_trades = economics.get("closed_trades")
            breaches = list(safety.get("breaches") or ())
            safe = safety.get("valid") is True and not breaches
            safety_cell = Text(
                "PASS" if safe else "CHECK",
                style="bold #73d89e" if safe else "bold #ff7070",
            )
            net = economics.get("run_net_usd")
            campaign_net = campaign.get("known_net_usd")
            drawdown = economics.get("drawdown_usd")
            net_cell = Text()
            net_cell.append_text(
                _pnl_text(float(net)) if net is not None else Text("-", style="dim")
            )
            net_cell.append(" / ", style="dim")
            net_cell.append_text(
                _pnl_text(float(campaign_net))
                if campaign_net is not None
                else Text("-", style="dim")
            )
            self._live_runs_table.add_row(
                *_center_table_row(
                    str(run.get("label") or run.get("strategy_id") or "unknown"),
                    self._live_run_state_cell(run),
                    self._live_run_capital(allocation),
                    self._live_run_position(run),
                    net_cell,
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
            view = await asyncio.to_thread(
                self._live_runs_owner.view,
                limit=self._LIVE_TIMELINE_LIMIT,
                trace_limit=self._LIVE_TRACE_LIMIT_PER_STRATEGY,
                previous_view_id=self._live_runs_view_id,
            )
            view_id = str(view.get("view_id") or "") or None
            if view.get("unchanged") is True:
                self._live_runs_view_id = view_id
                return
            changed = False
            if "snapshot" in view:
                snapshot = view["snapshot"]
                if not isinstance(snapshot, Mapping):
                    raise RuntimeError(
                        "q portfolio endpoint returned an invalid snapshot"
                    )
                self._live_runs_snapshot = dict(snapshot)
                self._render_live_runs_table(snapshot)
                if hasattr(self, "_render_candidates_table"):
                    self._render_candidates_table(snapshot)
                changed = True
            if "timeline" in view:
                timeline = view["timeline"]
                if not isinstance(timeline, list) or any(
                    not isinstance(event, Mapping) for event in timeline
                ):
                    raise RuntimeError(
                        "q portfolio endpoint returned an invalid timeline"
                    )
                if hasattr(self, "_render_timeline_tables"):
                    self._render_timeline_tables([dict(event) for event in timeline])
                changed = True
            if "traces" in view:
                traces = view["traces"]
                if not isinstance(traces, list) or any(
                    not isinstance(trace, Mapping) for trace in traces
                ):
                    raise RuntimeError(
                        "q portfolio endpoint returned invalid strategy traces"
                    )
                if hasattr(self, "_render_strategy_traces"):
                    self._render_strategy_traces([dict(trace) for trace in traces])
                changed = True
            if not changed:
                raise RuntimeError("q portfolio endpoint returned an empty delta")
            self._live_runs_view_id = view_id
            if self._active_panel in {"candidates", "live_runs", "activity", "traces"}:
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
            result = await asyncio.to_thread(
                self._live_runs_owner.request_control,
                sleeve_id,
                action,
            )
            receipt = result.get("receipt")
            decision = receipt.get("decision") if isinstance(receipt, Mapping) else None
            status = (
                str(decision.get("status") or "")
                if isinstance(decision, Mapping)
                else ""
            )
            self._set_status(f"{action.title()} {sleeve_id}: {status}")
        except Exception as exc:
            self._set_status(f"{action.title()} blocked: {exc}")
        await self._refresh_live_runs()

    def _live_run_detail_lines(self, run: Mapping[str, object]) -> list[Text]:
        allocation = run.get("allocation")
        economics = run.get("economics")
        campaign = run.get("campaign_economics")
        safety = run.get("safety")
        graduation = run.get("graduation")
        allocation = allocation if isinstance(allocation, Mapping) else {}
        economics = economics if isinstance(economics, Mapping) else {}
        campaign = campaign if isinstance(campaign, Mapping) else {}
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
                f"Capital={self._live_run_capital(allocation)}  "
                f"Position={self._live_run_position(run)}  "
                f"Cash=${float(run.get('settled_cash_usd') or 0):,.2f}",
                style="dim",
            ),
            Text(
                f"Active run  Net=${float(economics.get('run_net_usd') or 0):,.2f}  "
                f"Realized=${float(economics.get('run_realized_net_usd') or 0):,.2f}  "
                f"Mark=${float(economics.get('open_mark_net_usd') or 0):,.2f}  "
                f"Costs=${float(economics.get('run_cost_usd') or 0):,.2f}  "
                f"DD=${float(economics.get('drawdown_usd') or 0):,.2f}  "
                f"Fills/Trades={economics.get('fill_count', '-')}/{economics.get('closed_trades', '-')}",
                style="dim",
            ),
            Text(
                "Known live campaign  "
                f"Net=${float(campaign.get('known_net_usd') or 0):,.2f}  "
                f"Realized=${float(campaign.get('known_realized_net_usd') or 0):,.2f}  "
                f"Active mark=${float(campaign.get('active_open_mark_net_usd') or 0):,.2f}  "
                f"Trades={campaign.get('closed_trades', '-')}  "
                f"Selections={campaign.get('accounted_selection_runs', '-')}/{campaign.get('selection_runs', '-')}  "
                f"Attribution={'COMPLETE' if campaign.get('attribution_complete') is True else 'PARTIAL'}",
                style=(
                    "dim"
                    if campaign.get("attribution_complete") is True
                    else "bold #f2b36f"
                ),
            ),
            Text(
                f"Graduation={graduation.get('verdict', 'UNKNOWN')} "
                f"{graduation.get('target') or ''}  "
                f"cutoff={graduation.get('cutoff_utc') or 'none'}  "
                f"Safety={'PASS' if safety.get('valid') is True and not safety.get('breaches') else 'CHECK'}",
                style="dim",
            ),
            Text("Gate ladder: " + self._graduation_ladder(graduation), style="#8fbfff"),
        ]
        reasons = list(graduation.get("reasons") or ())
        if reasons:
            lines.append(Text("Graduation: " + "; ".join(str(value) for value in reasons), style="#f2b36f"))
        trace = (
            self._latest_trace_for_sleeve(run.get("sleeve_id"))
            if hasattr(self, "_latest_trace_for_sleeve")
            else None
        )
        if trace is not None and hasattr(self, "_trace_detail_lines"):
            lines.extend(self._trace_detail_lines(trace))
        else:
            lines.extend(self._live_run_hawkeye_lines(run))
        lines.append(
            Text(
                "Space resumes/pauses q's timer only. Stop refuses non-flat, open, pending, or busy runs. "
                "Replace/Rebalance require immutable successor artifacts.",
                style="dim",
            )
        )
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
