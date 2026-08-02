"""Immutable champion candidates and persistent portfolio evidence surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from rich.text import Text

from ..common import _pnl_text
from .formatting import _center_table_row


class BotPortfolioMixin:
    """Project q-owned candidates and ledger events without local execution."""

    def _init_portfolio_tables(self) -> None:
        self._candidate_rows: list[dict[str, object]] = []
        self._timeline_rows: list[dict[str, object]] = []
        self._candidate_control_task: asyncio.Task | None = None

    def _setup_candidates_table(self) -> None:
        self._candidates_table.clear(columns=True)
        for label, width in (
            ("Champion candidate", 34),
            ("Lane", 9),
            ("Identity", 14),
            ("Stage", 18),
            ("3Y net/PF/DD", 24),
            ("Durable run", 18),
            ("Graduation", 16),
            ("Control", 18),
        ):
            self._candidates_table.add_column(label, width=width)

    def _setup_timeline_tables(self) -> None:
        self._activity_table.clear(columns=True)
        self._activity_table.add_columns(
            "When UTC",
            "Champion",
            "Phase",
            "Target",
            "Action",
            "Qty",
            "Symbol",
            "Net",
            "DD",
        )
        self._logs_table.clear(columns=True)
        self._logs_table.add_columns(
            "When UTC",
            "Champion",
            "Kind",
            "Phase",
            "Status",
            "Reason",
            "Message",
        )

    @staticmethod
    def _candidate_stage(value: Mapping[str, object]) -> Text:
        stage = str(value.get("stage") or "UNKNOWN")
        style = {
            "PROMOTED": "bold #73d89e",
            "CANARY": "bold #62b0ff",
            "CROWNED": "bold #8fbfff",
            "SELECTION_REQUIRED": "bold #f2b36f",
            "RESEARCH_ONLY": "#b8c0cb",
            "HOLD": "bold #f2b36f",
            "REVISE": "bold #f2b36f",
            "QUARANTINED": "bold #ff7070",
            "STOP": "bold #ff7070",
        }.get(stage, "bold #d6a56f")
        return Text(stage, style=style)

    @staticmethod
    def _candidate_historical(value: Mapping[str, object]) -> str:
        metrics = value.get("historical")
        if not isinstance(metrics, Mapping) or not metrics:
            return "-"
        net = metrics.get("net_pnl")
        factor = metrics.get("profit_factor")
        drawdown = metrics.get("max_drawdown")
        try:
            return f"{float(net):+.2f} / {float(factor):.3f} / {float(drawdown):.2f}"
        except (TypeError, ValueError):
            return "invalid"

    @staticmethod
    def _candidate_graduation(value: Mapping[str, object]) -> str:
        graduation = value.get("graduation")
        if not isinstance(graduation, Mapping):
            return "not started"
        verdict = str(graduation.get("verdict") or "UNKNOWN")
        target = str(graduation.get("target") or "")
        return f"{verdict} {target}".strip()

    @staticmethod
    def _candidate_control(value: Mapping[str, object]) -> str:
        controls = value.get("controls")
        commission = controls.get("COMMISSION") if isinstance(controls, Mapping) else None
        status = (
            str(commission.get("status") or "HOLD")
            if isinstance(commission, Mapping)
            else "HOLD"
        )
        return {
            "ALLOW": "Enter:Commission",
            "NOOP": "Enter:Focus run",
        }.get(status, "Enter:Explain HOLD")

    def _render_candidates_table(self, snapshot: Mapping[str, object]) -> None:
        prior = self._selected_candidate()
        prior_id = prior.get("candidate_id") if prior is not None else None
        self._candidates_table.clear()
        self._candidate_rows = []
        candidates = snapshot.get("candidates")
        for candidate in candidates if isinstance(candidates, list) else ():
            if not isinstance(candidate, Mapping):
                continue
            candidate = dict(candidate)
            artifact_sha = str(candidate.get("artifact_sha256") or "")
            run_id = str(candidate.get("run_id") or "")
            self._candidates_table.add_row(
                *_center_table_row(
                    str(candidate.get("label") or candidate.get("symbol") or "unknown"),
                    f"{candidate.get('symbol')}/{candidate.get('track')}",
                    f"{artifact_sha[:12]}…" if artifact_sha else "-",
                    self._candidate_stage(candidate),
                    self._candidate_historical(candidate),
                    f"{candidate.get('run_state')} {run_id[:8]}…" if run_id else "not selected",
                    self._candidate_graduation(candidate),
                    self._candidate_control(candidate),
                ),
                key=f"candidate:{candidate.get('candidate_id')}",
            )
            self._candidate_rows.append(candidate)
        if prior_id is not None:
            for index, candidate in enumerate(self._candidate_rows):
                if candidate.get("candidate_id") == prior_id:
                    self._candidates_table.cursor_coordinate = (index, 0)
                    break
        elif self._candidate_rows:
            self._candidates_table.cursor_coordinate = (0, 0)

    def _render_timeline_tables(self, events: list[dict[str, object]]) -> None:
        self._activity_table.clear()
        self._logs_table.clear()
        self._timeline_rows = events
        for event in events:
            when = str(event.get("recorded_at_utc") or "")
            label = str(event.get("label") or event.get("sleeve_id") or "portfolio")
            net = event.get("run_net_usd")
            drawdown = event.get("drawdown_usd")
            self._activity_table.add_row(
                when,
                label,
                str(event.get("phase") or "-"),
                str(event.get("target_direction") or "-"),
                str(event.get("action") or "-"),
                str(event.get("quantity") if event.get("quantity") is not None else "-"),
                str(event.get("symbol") or "-"),
                _pnl_text(float(net)) if net is not None else Text("-", style="dim"),
                f"${float(drawdown):,.2f}" if drawdown is not None else "-",
                key=f"activity:{event.get('event_id')}",
            )
            self._logs_table.add_row(
                when,
                label,
                str(event.get("kind") or "-"),
                str(event.get("phase") or "-"),
                str(event.get("status") or "-"),
                str(event.get("reason") or "-"),
                str(event.get("message") or "-"),
                key=f"log:{event.get('event_id')}",
            )
        if events:
            last = len(events) - 1
            self._activity_table.cursor_coordinate = (last, 0)
            self._logs_table.cursor_coordinate = (last, 0)

    def _selected_candidate(self) -> dict[str, object] | None:
        table = getattr(self, "_candidates_table", None)
        if table is None:
            return None
        row = table.cursor_coordinate.row
        if row < 0 or row >= len(self._candidate_rows):
            return None
        return self._candidate_rows[row]

    def _selected_timeline_event(self) -> dict[str, object] | None:
        table = self._activity_table if self._active_panel == "activity" else self._logs_table
        row = table.cursor_coordinate.row
        if row < 0 or row >= len(self._timeline_rows):
            return None
        return self._timeline_rows[row]

    def _activate_candidates_panel(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            self._set_status("Champions: no candidate selected")
            return
        if self._candidate_control_task and not self._candidate_control_task.done():
            self._set_status("Champions: commissioning transaction already active")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("Champions: no event loop")
            return
        self._candidate_control_task = loop.create_task(
            self._commission_candidate(str(candidate.get("candidate_id") or ""))
        )

    async def _commission_candidate(self, candidate_id: str) -> None:
        try:
            result = await asyncio.to_thread(self._live_runs_owner.commission, candidate_id)
            receipt = result.get("receipt")
            decision = receipt.get("decision") if isinstance(receipt, Mapping) else None
            status = (
                str(decision.get("status") or "HOLD")
                if isinstance(decision, Mapping)
                else "HOLD"
            )
            reasons = list(decision.get("reasons") or ()) if isinstance(decision, Mapping) else []
            suffix = f": {', '.join(str(reason) for reason in reasons)}" if reasons else ""
            self._set_status(f"Commission {status}{suffix}")
        except Exception as exc:  # pragma: no cover - operator surface
            self._set_status(f"Commission blocked: {exc}")
        await self._refresh_live_runs()

    def _candidate_detail_lines(self, candidate: Mapping[str, object]) -> list[Text]:
        lines = [
            Text(""),
            Text("Immutable champion candidate", style="bold"),
            Text(
                f"{candidate.get('label')}  lane={candidate.get('symbol')}/{candidate.get('track')}  "
                f"stage={candidate.get('stage')}",
                style="dim",
            ),
            Text(
                f"Declaration={str(candidate.get('declaration_sha256') or '')[:16]}…  "
                f"artifact={str(candidate.get('artifact_sha256') or '')[:16]}…  "
                f"run={str(candidate.get('run_id') or 'not selected')[:16]}",
                style="dim",
            ),
        ]
        reasons = list(candidate.get("reasons") or ())
        if reasons:
            lines.append(Text("Gates: " + "; ".join(str(value) for value in reasons), style="#f2b36f"))
        if candidate.get("machine_authority") is not True:
            lines.append(
                Text(
                    "README provenance is research-only; live authority requires a machine crown, "
                    "registered worker, immutable selection, and capital sleeve.",
                    style="bold #ff7070",
                )
            )
        return lines

    def _timeline_detail_lines(self, event: Mapping[str, object]) -> list[Text]:
        lines = [
            Text(""),
            Text("Persisted portfolio evidence", style="bold"),
            Text(
                f"{event.get('recorded_at_utc')}  {event.get('kind')} / {event.get('phase')}  "
                f"status={event.get('status') or '-'} reason={event.get('reason') or '-'}",
                style="dim",
            ),
            Text(str(event.get("message") or ""), style="dim"),
        ]
        execution = event.get("execution_detail")
        execution = execution if isinstance(execution, Mapping) else {}
        ladder = execution.get("ladder_transition")
        ladder = ladder if isinstance(ladder, Mapping) else {}
        broker = execution.get("broker_order")
        broker = broker if isinstance(broker, Mapping) else {}
        preview = execution.get("what_if_preview")
        preview = preview if isinstance(preview, Mapping) else {}
        if ladder:
            lines.append(
                Text(
                    "Execution ladder  "
                    f"{ladder.get('previous_mode') or 'START'}→{ladder.get('active_mode') or '-'} "
                    f"elapsed={float(ladder.get('elapsed_seconds') or 0):.2f}s "
                    f"limit={ladder.get('limit_price') or '-'} "
                    f"quote_age={float(ladder.get('quote_age_seconds') or 0):.2f}s "
                    f"eligible={ladder.get('quote_eligible')} "
                    f"reprices={ladder.get('no_progress_reprices', '-')}",
                    style="#8fbfff",
                )
            )
        if broker:
            fills = broker.get("fills")
            fills = fills if isinstance(fills, list) else []
            commissions = [
                float(fill.get("commission") or 0)
                for fill in fills
                if isinstance(fill, Mapping)
            ]
            lines.append(
                Text(
                    "Broker  "
                    f"ref={execution.get('order_ref') or '-'} "
                    f"status={broker.get('status') or '-'} "
                    f"filled={broker.get('filled', '-')}/{broker.get('quantity', '-')} "
                    f"avg={broker.get('average_fill_price', '-')} "
                    f"commission=${sum(commissions):.4f}",
                    style="dim",
                )
            )
        if preview:
            lines.append(
                Text(
                    "Broker preview  "
                    f"status={preview.get('status') or '-'} "
                    f"commission={preview.get('commission') or preview.get('max_commission') or '-'}",
                    style="dim",
                )
            )
        context = event.get("execution_state_context")
        if isinstance(context, Mapping) and context:
            lines.extend(
                self._live_run_hawkeye_lines(
                    {
                        "execution_state_context": context,
                        "latest_decision": event.get("latest_decision") or {},
                    }
                )
            )
        return lines
