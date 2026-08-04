"""Durable champion catalog, commissioning controls, and evidence timeline."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..spot.champions import discover_current_champions, load_champion_group
from .runs import (
    CommandRunner,
    GraduationValidator,
    LiveRunBinding,
    LiveRunCockpit,
    UnitReader,
    _canonical_json,
    _identity,
    _ledger_records,
    _matching_execution_records,
    _read_json,
    _repo_path,
    _run_systemctl,
    read_systemd_user_unit,
)


LIVE_PORTFOLIO_SCHEMA = "live.portfolio-cockpit.v1"
LIVE_CONTROL_REQUEST_SCHEMA = "live.control-request.v1"
LIVE_CONTROL_RECEIPT_SCHEMA = "live.control-receipt.v1"


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class LivePortfolioCockpit(LiveRunCockpit):
    """Expose one q-owned crown -> run -> evidence -> graduation chain."""

    def __init__(
        self,
        *,
        repository_root: Path,
        capital_plan_path: Path,
        bindings: Sequence[LiveRunBinding],
        graduation_directory: Path,
        graduation_validator: GraduationValidator,
        control_ledger_path: Path | None = None,
        unit_reader: UnitReader = read_systemd_user_unit,
        command_runner: CommandRunner = _run_systemctl,
    ) -> None:
        super().__init__(
            repository_root=repository_root,
            capital_plan_path=capital_plan_path,
            bindings=bindings,
            graduation_directory=graduation_directory,
            graduation_validator=graduation_validator,
            unit_reader=unit_reader,
            command_runner=command_runner,
        )
        configured = control_ledger_path or Path("db/calibration/live_control.jsonl")
        self.control_ledger_path = (
            configured if configured.is_absolute() else self.repository_root / configured
        )
        self.candidate_bindings = {
            (
                binding.champion_symbol.strip().upper(),
                binding.champion_track.strip().upper(),
            ): binding
            for binding in bindings
            if binding.champion_symbol.strip() and binding.champion_track.strip()
        }
        expected = sum(
            bool(binding.champion_symbol.strip() and binding.champion_track.strip())
            for binding in bindings
        )
        if len(self.candidate_bindings) != expected:
            raise ValueError("durable-run champion bindings are duplicated")

    def _project_candidates(
        self,
        runs: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        run_by_strategy = {
            str(run.get("strategy_id") or ""): run
            for run in runs
            if isinstance(run, Mapping)
        }
        projected: list[dict[str, object]] = []
        for ref in discover_current_champions(root=self.repository_root):
            lane = (ref.symbol.strip().upper(), ref.track.strip().upper())
            binding = self.candidate_bindings.get(lane)
            declaration_sha = hashlib.sha256(ref.declaration_path.read_bytes()).hexdigest()
            artifact_sha = hashlib.sha256(ref.artifact_path.read_bytes()).hexdigest()
            try:
                artifact = _read_json(ref.artifact_path)
            except (OSError, TypeError, ValueError):
                artifact = {}
            machine = ref.declaration_path.suffix.lower() == ".json"
            reasons: list[str] = []
            declaration: Mapping[str, object] | None = None
            if machine:
                try:
                    declaration = _read_json(ref.declaration_path)
                except (OSError, TypeError, ValueError) as exc:
                    reasons.append(f"invalid_machine_declaration:{exc}")
                promotion = (
                    declaration.get("promotion")
                    if isinstance(declaration, Mapping)
                    else None
                )
                if (
                    not isinstance(declaration, Mapping)
                    or declaration.get("schema") != "tradebot.spot.champion.v1"
                    or declaration.get("symbol") != lane[0]
                    or declaration.get("track") != lane[1]
                    or declaration.get("artifact_sha256") != artifact_sha
                    or not isinstance(promotion, Mapping)
                    or promotion.get("eligible") is not True
                    or promotion.get("order_authority") != "none"
                ):
                    reasons.append("machine_crown_identity_or_promotion_invalid")
            else:
                reasons.append("legacy_readme_declaration_is_research_only")
            if binding is None:
                reasons.append("durable_execution_binding_missing")
            run = run_by_strategy.get(binding.strategy_id) if binding is not None else None
            if machine and not reasons and run is None:
                reasons.append("immutable_selection_and_capital_sleeve_missing")

            graduation = run.get("graduation") if isinstance(run, Mapping) else None
            verdict = (
                str(graduation.get("verdict") or "")
                if isinstance(graduation, Mapping)
                else ""
            )
            if not machine:
                stage = "RESEARCH_ONLY"
            elif any(reason.startswith("invalid_") or "identity" in reason for reason in reasons):
                stage = "QUARANTINED"
            elif binding is None:
                stage = "CROWNED"
            elif run is None:
                stage = "SELECTION_REQUIRED"
            elif run.get("valid") is not True:
                stage = "QUARANTINED"
            elif verdict == "PROMOTE":
                stage = "PROMOTED"
            elif verdict in {"REVISE", "QUARANTINE", "STOP"}:
                stage = "QUARANTINED" if verdict == "QUARANTINE" else verdict
            else:
                stage = "CANARY"

            if isinstance(run, Mapping) and run.get("valid") is True:
                timer = run.get("timer")
                running = (
                    isinstance(timer, Mapping)
                    and timer.get("active_state") == "active"
                )
                controls = run.get("controls")
                start = controls.get("START") if isinstance(controls, Mapping) else None
                if running:
                    commission = {"status": "NOOP", "reasons": ["durable_run_already_active"]}
                elif isinstance(start, Mapping):
                    commission = dict(start)
                else:
                    commission = {"status": "HOLD", "reasons": ["durable_run_start_state_missing"]}
            else:
                commission = {"status": "HOLD", "reasons": sorted(set(reasons))}

            group = load_champion_group(ref)
            crown_metrics = artifact.get("crown_metrics")
            qualification = artifact.get("graduation_enrollment")
            full_metrics = (
                crown_metrics.get("catalog", crown_metrics.get("full_three_year"))
                if isinstance(crown_metrics, Mapping)
                else None
            )
            identity = {
                "symbol": lane[0],
                "track": lane[1],
                "version": ref.version,
                "declaration_sha256": declaration_sha,
                "artifact_sha256": artifact_sha,
                "strategy_id": binding.strategy_id if binding is not None else None,
            }
            projected.append(
                {
                    "candidate_id": _identity(identity),
                    "symbol": lane[0],
                    "track": lane[1],
                    "version": ref.version,
                    "label": (
                        str(group.get("name") or "")
                        if isinstance(group, Mapping)
                        else f"{lane[0]} {lane[1]}"
                    ),
                    "machine_authority": machine,
                    "declaration_path": str(ref.declaration_path.relative_to(self.repository_root)),
                    "declaration_sha256": declaration_sha,
                    "artifact_path": str(ref.artifact_path.relative_to(self.repository_root)),
                    "artifact_sha256": artifact_sha,
                    "strategy_id": binding.strategy_id if binding is not None else None,
                    "stage": stage,
                    "reasons": sorted(set(reasons)),
                    "historical": dict(full_metrics) if isinstance(full_metrics, Mapping) else {},
                    "qualification": (
                        dict(qualification) if isinstance(qualification, Mapping) else {}
                    ),
                    "run_sleeve_id": run.get("sleeve_id") if isinstance(run, Mapping) else None,
                    "run_id": run.get("run_id") if isinstance(run, Mapping) else None,
                    "run_state": run.get("state") if isinstance(run, Mapping) else None,
                    "graduation": dict(graduation) if isinstance(graduation, Mapping) else None,
                    "controls": {"COMMISSION": commission},
                }
            )
        return projected

    def snapshot(self) -> dict[str, object]:
        selected = super().snapshot()
        body = {
            **{key: value for key, value in selected.items() if key != "snapshot_id"},
            "schema": LIVE_PORTFOLIO_SCHEMA,
            "candidates": self._project_candidates(selected.get("runs", ())),
        }
        return {**body, "snapshot_id": _identity(body)}

    def _publish_control(
        self,
        *,
        request_id: str,
        action: str,
        target: Mapping[str, object],
        decision: Mapping[str, object],
        before: Mapping[str, object],
        after: Mapping[str, object],
    ) -> dict[str, object]:
        body = {
            "schema": LIVE_CONTROL_RECEIPT_SCHEMA,
            "recorded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "request_id": request_id,
            "action": action.upper(),
            "target": dict(target),
            "decision": dict(decision),
            "before_snapshot_id": before.get("snapshot_id"),
            "after_snapshot_id": after.get("snapshot_id"),
            "boundaries": {
                "ui_broker_client_used": False,
                "broker_order_submitted": False,
                "selection_mutated": False,
                "capital_plan_mutated": False,
            },
        }
        receipt = {**body, "receipt_id": _identity(body)}
        _append_jsonl(self.control_ledger_path, receipt)
        return receipt

    def _publish_request(
        self,
        *,
        action: str,
        target: Mapping[str, object],
        before: Mapping[str, object],
    ) -> dict[str, object]:
        body = {
            "schema": LIVE_CONTROL_REQUEST_SCHEMA,
            "recorded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "action": action.upper(),
            "target": dict(target),
            "before_snapshot_id": before.get("snapshot_id"),
            "authority": "schedule_or_commission_request_only",
        }
        request = {**body, "request_id": _identity(body)}
        _append_jsonl(self.control_ledger_path, request)
        return request

    def request_control(self, sleeve_id: str, action: str) -> dict[str, object]:
        before = self.snapshot()
        target = {"kind": "run", "sleeve_id": sleeve_id}
        request = self._publish_request(action=action, target=target, before=before)
        try:
            result = self.control(sleeve_id, action)
            decision = result["decision"]
            after = result["after"]
        except (RuntimeError, ValueError) as exc:
            decision = {"status": "HOLD", "reasons": [str(exc)]}
            after = self.snapshot()
        receipt = self._publish_control(
            request_id=str(request["request_id"]),
            action=action,
            target=target,
            decision=decision,
            before=before,
            after=after,
        )
        return {"request": request, "receipt": receipt, "before": before, "after": after}

    def commission(self, candidate_id: str) -> dict[str, object]:
        before = self.snapshot()
        matches = [
            candidate
            for candidate in before.get("candidates", ())
            if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError("live candidate is missing or duplicated")
        candidate = matches[0]
        controls = candidate.get("controls")
        commission = controls.get("COMMISSION") if isinstance(controls, Mapping) else None
        decision = (
            dict(commission)
            if isinstance(commission, Mapping)
            else {"status": "HOLD", "reasons": ["candidate_commission_state_missing"]}
        )
        target = {
            "kind": "candidate",
            "candidate_id": candidate_id,
            "symbol": candidate.get("symbol"),
            "track": candidate.get("track"),
            "strategy_id": candidate.get("strategy_id"),
            "sleeve_id": candidate.get("run_sleeve_id"),
        }
        request = self._publish_request(
            action="COMMISSION",
            target=target,
            before=before,
        )
        sleeve_id = str(candidate.get("run_sleeve_id") or "")
        after = before
        if decision.get("status") == "ALLOW" and sleeve_id:
            try:
                result = self.control(sleeve_id, "START")
                decision = dict(result["decision"])
                after = result["after"]
            except (RuntimeError, ValueError) as exc:
                decision = {"status": "HOLD", "reasons": [str(exc)]}
                after = self.snapshot()
        receipt = self._publish_control(
            request_id=str(request["request_id"]),
            action="COMMISSION",
            target=target,
            decision=decision,
            before=before,
            after=after,
        )
        return {"request": request, "receipt": receipt, "before": before, "after": after}

    def _timeline(
        self,
        snapshot: Mapping[str, object],
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for run in snapshot.get("runs", ()):
            if not isinstance(run, Mapping) or run.get("valid") is not True:
                continue
            binding = self.bindings.get(str(run.get("strategy_id") or ""))
            if binding is None:
                continue
            records = _matching_execution_records(
                self._read_ledger(_repo_path(self.repository_root, binding.ledger_path)),
                binding=binding,
                run_id=str(run.get("run_id") or ""),
            )
            for record in records:
                evidence = record.get("evidence")
                evidence = evidence if isinstance(evidence, Mapping) else {}
                plan = evidence.get("plan")
                plan = plan if isinstance(plan, Mapping) else {}
                leg = plan.get("leg")
                leg = leg if isinstance(leg, Mapping) else {}
                broker = evidence.get("broker_order")
                broker = broker if isinstance(broker, Mapping) else {}
                ladder = evidence.get("ladder_transition")
                ladder = ladder if isinstance(ladder, Mapping) else {}
                preview = evidence.get("what_if_preview")
                preview = preview if isinstance(preview, Mapping) else {}
                risk = evidence.get("risk_state")
                risk = risk if isinstance(risk, Mapping) else {}
                context = plan.get("execution_state_context")
                context = context if isinstance(context, Mapping) else {}
                action = str(leg.get("action") or broker.get("action") or "")
                symbol = str(leg.get("symbol") or broker.get("symbol") or "")
                quantity = leg.get("quantity", broker.get("quantity"))
                target = str(plan.get("target_direction") or "flat")
                events.append(
                    {
                        "event_id": record.get("checkpoint_id"),
                        "recorded_at_utc": record.get("recorded_at_utc"),
                        "kind": "EXECUTION",
                        "sleeve_id": run.get("sleeve_id"),
                        "run_id": run.get("run_id"),
                        "label": run.get("label"),
                        "phase": evidence.get("phase") or "STATE",
                        "reason": plan.get("reason"),
                        "status": plan.get("status"),
                        "target_direction": plan.get("target_direction"),
                        "target_symbol": plan.get("target_symbol"),
                        "symbol": symbol or None,
                        "action": action or None,
                        "quantity": quantity,
                        "holdings": dict(plan.get("holdings") or {}),
                        "run_net_usd": risk.get("run_net_usd"),
                        "drawdown_usd": risk.get("drawdown_usd"),
                        "message": (
                            f"target={target}"
                            + (f" {action} {quantity} {symbol}" if action and symbol else "")
                        ),
                        "latest_decision": {
                            "status": plan.get("status"),
                            "reason": plan.get("reason"),
                        },
                        "execution_state_context": dict(context),
                        "execution_detail": {
                            "order_ref": evidence.get("order_ref"),
                            "ladder_transition": dict(ladder),
                            "broker_order": dict(broker),
                            "what_if_preview": dict(preview),
                        },
                    }
                )
        if self.control_ledger_path.exists():
            for receipt in _ledger_records(self.control_ledger_path):
                if receipt.get("schema") not in {
                    LIVE_CONTROL_REQUEST_SCHEMA,
                    LIVE_CONTROL_RECEIPT_SCHEMA,
                }:
                    continue
                target = receipt.get("target")
                decision = receipt.get("decision")
                target = target if isinstance(target, Mapping) else {}
                decision = decision if isinstance(decision, Mapping) else {
                    "status": "REQUESTED",
                    "reasons": [],
                }
                events.append(
                    {
                        "event_id": receipt.get("receipt_id") or receipt.get("request_id"),
                        "recorded_at_utc": receipt.get("recorded_at_utc"),
                        "kind": (
                            "CONTROL_REQUEST"
                            if receipt.get("schema") == LIVE_CONTROL_REQUEST_SCHEMA
                            else "CONTROL"
                        ),
                        "sleeve_id": target.get("sleeve_id"),
                        "run_id": None,
                        "label": target.get("strategy_id") or target.get("symbol") or "portfolio",
                        "phase": receipt.get("action"),
                        "reason": ",".join(str(value) for value in decision.get("reasons", ())),
                        "status": decision.get("status"),
                        "target_direction": None,
                        "target_symbol": None,
                        "symbol": target.get("symbol"),
                        "action": receipt.get("action"),
                        "quantity": None,
                        "holdings": {},
                        "run_net_usd": None,
                        "drawdown_usd": None,
                        "message": f"{receipt.get('action')} {decision.get('status')}",
                        "latest_decision": {},
                        "execution_state_context": {},
                    }
                )
        events.sort(key=lambda event: str(event.get("recorded_at_utc") or ""))
        return events[-max(1, min(int(limit), 10_000)) :]

    def view(self, *, limit: int = 1_000) -> dict[str, object]:
        """Read one internally consistent cockpit snapshot and its timeline."""

        snapshot = self.snapshot()
        return {
            "snapshot": snapshot,
            "timeline": self._timeline(snapshot, limit=limit),
        }

    def timeline(self, *, limit: int = 1_000) -> list[dict[str, object]]:
        return list(self.view(limit=limit)["timeline"])
