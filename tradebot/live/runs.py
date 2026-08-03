"""Durable selected-run projection and systemd schedule control."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .capital import load_live_capital_plan


LIVE_RUN_COCKPIT_SCHEMA = "live.run-cockpit.v1"
LIVE_RUN_CONTROL_ACTIONS = {"START", "STOP", "REPLACE", "REBALANCE"}

SelectionValidator = Callable[[Mapping[str, object]], dict[str, object]]
GraduationValidator = Callable[[Mapping[str, object]], dict[str, object]]
UnitReader = Callable[[str], Mapping[str, object]]
CommandRunner = Callable[[Sequence[str]], object]


@dataclass(frozen=True)
class LiveRunBinding:
    """Bind one selected strategy identity to its durable q owner."""

    strategy_id: str
    label: str
    execution_strategy_version: str
    ledger_path: str
    timer_unit: str
    service_unit: str
    selection_validator: SelectionValidator
    champion_symbol: str = ""
    champion_track: str = ""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected one JSON object")
    return dict(value)


def _repo_path(root: Path, raw: object) -> Path:
    path = Path(str(raw or ""))
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("durable-run path escapes the repository root")
    return resolved


def _ledger_records(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: expected one JSON object")
        records.append(dict(value))
    return tuple(records)


def read_systemd_user_unit(unit: str) -> dict[str, object]:
    """Read one user unit without changing it."""

    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=Id",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=UnitFileState",
                "--property=Result",
                "--property=ExecMainStatus",
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "unit": unit,
            "available": False,
            "load_state": "unknown",
            "active_state": "unknown",
            "sub_state": "unknown",
            "unit_file_state": "unknown",
            "result": "unknown",
            "exec_main_status": None,
            "error": str(exc),
        }
    values = {
        key: value
        for raw in completed.stdout.splitlines()
        if "=" in raw
        for key, value in (raw.split("=", 1),)
    }
    return {
        "unit": unit,
        "available": completed.returncode == 0 and values.get("LoadState") == "loaded",
        "load_state": values.get("LoadState", "not-found"),
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "unit_file_state": values.get("UnitFileState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus"),
        "error": completed.stderr.strip() or None,
    }


def _run_systemctl(arguments: Sequence[str]) -> None:
    completed = subprocess.run(
        ["systemctl", "--user", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"systemctl {' '.join(arguments)} failed: {detail}")


def _matching_execution_records(
    records: Sequence[Mapping[str, object]],
    *,
    binding: LiveRunBinding,
    run_id: str,
) -> tuple[dict[str, object], ...]:
    selected = []
    identities: set[str] = set()
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version")
            != binding.execution_strategy_version
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != run_id
        ):
            continue
        checkpoint_id = str(record.get("checkpoint_id") or "")
        if not checkpoint_id or checkpoint_id in identities:
            raise ValueError("selected execution ledger identities are missing or duplicated")
        identities.add(checkpoint_id)
        selected.append(dict(record))
    return tuple(selected)


def _latest_mapping(
    records: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object] | None:
    for record in reversed(records):
        evidence = record.get("evidence")
        value = evidence.get(field) if isinstance(evidence, Mapping) else None
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _pending_order_refs(records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    latest: dict[str, str] = {}
    for record in records:
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        order_ref = str(evidence.get("order_ref") or "")
        phase = str(evidence.get("phase") or "")
        if order_ref and phase in {"PREPARED", "SUBMITTED", "TERMINAL"}:
            latest[order_ref] = phase
    return tuple(
        sorted(
            order_ref
            for order_ref, phase in latest.items()
            if phase in {"PREPARED", "SUBMITTED"}
        )
    )


def _latest_graduation(
    directory: Path,
    *,
    run_id: str,
    strategy_id: str,
    execution_strategy_version: str,
    validator: GraduationValidator,
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    invalid: list[str] = []
    if directory.exists():
        for path in sorted(directory.glob(f"{run_id}.*.json")):
            try:
                receipt = validator(_read_json(path))
            except (OSError, TypeError, ValueError):
                invalid.append(path.name)
                continue
            subject = receipt.get("subject")
            if (
                isinstance(subject, Mapping)
                and subject.get("run_id") == run_id
                and subject.get("strategy_id") == strategy_id
                and subject.get("strategy_version") == execution_strategy_version
            ):
                matches.append(receipt)
    if invalid:
        return {
            "verdict": "QUARANTINE",
            "target": None,
            "cutoff_utc": None,
            "receipt_id": None,
            "reasons": [f"invalid_graduation_receipt:{name}" for name in invalid],
        }
    if not matches:
        return {
            "verdict": "PENDING",
            "target": None,
            "cutoff_utc": None,
            "receipt_id": None,
            "reasons": ["no_cutoff_bound_graduation_receipt"],
        }
    receipt = max(
        matches,
        key=lambda value: str(
            value.get("target", {}).get("cutoff_utc", "")
            if isinstance(value.get("target"), Mapping)
            else ""
        ),
    )
    target = receipt["target"]
    assert isinstance(target, Mapping)
    return {
        "verdict": receipt["verdict"],
        "target": target.get("milestone"),
        "cutoff_utc": target.get("cutoff_utc"),
        "receipt_id": receipt.get("receipt_id"),
        "reasons": list(receipt.get("reasons") or ()),
    }


def _positions_flat(positions: Mapping[str, object]) -> bool:
    try:
        return all(abs(float(value or 0)) < 1e-9 for value in positions.values())
    except (TypeError, ValueError):
        return False


def _position_mapping(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(quantity or 0) for key, quantity in value.items()}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    positions: dict[str, float] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("durable broker position row is invalid")
        symbol = str(row.get("symbol") or "").upper()
        quantity = float(row.get("quantity") or 0)
        if symbol:
            positions[symbol] = positions.get(symbol, 0.0) + quantity
    return positions


def _control_status(run: Mapping[str, object], action: str) -> dict[str, object]:
    normalized = str(action or "").upper()
    if normalized not in LIVE_RUN_CONTROL_ACTIONS:
        raise ValueError(f"unsupported durable-run control: {action!r}")
    if normalized in {"REPLACE", "REBALANCE"}:
        reason = (
            "validated_successor_selection_and_capital_plan_required"
            if normalized == "REPLACE"
            else "immutable_successor_capital_plan_required"
        )
        return {"status": "HOLD", "reasons": [reason]}
    if run.get("valid") is not True:
        return {"status": "HOLD", "reasons": ["durable_run_invalid"]}
    service = run.get("service")
    timer = run.get("timer")
    if not isinstance(service, Mapping) or not isinstance(timer, Mapping):
        return {"status": "HOLD", "reasons": ["durable_owner_state_missing"]}
    if service.get("available") is not True or timer.get("available") is not True:
        return {"status": "HOLD", "reasons": ["durable_owner_unit_unavailable"]}
    timer_active = timer.get("active_state") == "active"
    if normalized == "START":
        if timer_active:
            return {"status": "NOOP", "reasons": ["schedule_already_active"]}
        positions = run.get("positions")
        pending = run.get("pending_order_refs")
        open_orders = run.get("open_orders")
        needs_reconciliation = (
            isinstance(positions, Mapping)
            and not _positions_flat(positions)
        ) or bool(pending) or bool(open_orders)
        graduation = run.get("graduation")
        verdict = (
            str(graduation.get("verdict") or "")
            if isinstance(graduation, Mapping)
            else ""
        )
        if verdict in {"STOP", "QUARANTINE", "REVISE"} and not needs_reconciliation:
            return {
                "status": "HOLD",
                "reasons": [f"graduation_{verdict.lower()}"],
            }
        if run.get("ledger_rows", 0) == 0 and not needs_reconciliation:
            return {"status": "HOLD", "reasons": ["run_not_observed"]}
        return {
            "status": "ALLOW",
            "reasons": (
                ["reconciliation_owner_must_resume"] if needs_reconciliation else []
            ),
        }
    if not timer_active:
        return {"status": "NOOP", "reasons": ["schedule_already_inactive"]}
    reasons = []
    positions = run.get("positions")
    if not isinstance(positions, Mapping) or not _positions_flat(positions):
        reasons.append("selected_position_not_flat")
    if run.get("open_orders"):
        reasons.append("broker_open_orders_present")
    if run.get("pending_order_refs"):
        reasons.append("unreconciled_transition_present")
    if service.get("active_state") != "inactive":
        reasons.append("durable_owner_invocation_active")
    return {
        "status": "HOLD" if reasons else "ALLOW",
        "reasons": reasons,
    }


class LiveRunCockpit:
    """Project and control durable selected runs without owning broker orders."""

    def __init__(
        self,
        *,
        repository_root: Path,
        capital_plan_path: Path,
        bindings: Sequence[LiveRunBinding],
        graduation_directory: Path,
        graduation_validator: GraduationValidator,
        unit_reader: UnitReader = read_systemd_user_unit,
        command_runner: CommandRunner = _run_systemctl,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.capital_plan_path = capital_plan_path
        self.graduation_directory = graduation_directory
        self._graduation_validator = graduation_validator
        self.bindings = {binding.strategy_id: binding for binding in bindings}
        if len(self.bindings) != len(bindings):
            raise ValueError("durable-run strategy bindings are duplicated")
        self._unit_reader = unit_reader
        self._command_runner = command_runner
        self._ledger_cache: dict[
            Path,
            tuple[int, int, tuple[dict[str, object], ...]],
        ] = {}

    def _read_ledger(self, path: Path) -> tuple[dict[str, object], ...]:
        stat = path.stat()
        cached = self._ledger_cache.get(path)
        identity = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == identity:
            return cached[2]
        records = _ledger_records(path)
        self._ledger_cache[path] = (*identity, records)
        return records

    def _invalid_run(
        self,
        sleeve: Mapping[str, object],
        *,
        error: Exception | str,
        binding: LiveRunBinding | None,
    ) -> dict[str, object]:
        message = str(error)
        timer = (
            dict(self._unit_reader(binding.timer_unit)) if binding is not None else None
        )
        service = (
            dict(self._unit_reader(binding.service_unit)) if binding is not None else None
        )
        run = {
            "sleeve_id": sleeve.get("sleeve_id"),
            "strategy_id": sleeve.get("strategy_id"),
            "run_id": sleeve.get("run_id"),
            "label": binding.label if binding is not None else str(sleeve.get("strategy_id")),
            "valid": False,
            "state": "QUARANTINED",
            "errors": [message],
            "timer": timer,
            "service": service,
            "positions": {},
            "open_orders": [],
            "pending_order_refs": [],
            "ledger_rows": 0,
            "graduation": {
                "verdict": "QUARANTINE",
                "reasons": [message],
            },
        }
        run["controls"] = {
            action: _control_status(run, action)
            for action in sorted(LIVE_RUN_CONTROL_ACTIONS)
        }
        return run

    def _project_run(
        self,
        sleeve: Mapping[str, object],
        *,
        managed_capital_cents: int,
    ) -> dict[str, object]:
        strategy_id = str(sleeve.get("strategy_id") or "")
        binding = self.bindings.get(strategy_id)
        if binding is None:
            return self._invalid_run(
                sleeve,
                error="no durable owner binding for allocated strategy",
                binding=None,
            )
        try:
            selection_path = _repo_path(
                self.repository_root,
                sleeve.get("selection_path"),
            )
            payload = selection_path.read_bytes()
            selection_sha = hashlib.sha256(payload).hexdigest()
            if selection_sha != sleeve.get("selection_file_sha256"):
                raise ValueError("allocated selection file identity changed")
            raw_selection = json.loads(payload)
            if not isinstance(raw_selection, Mapping):
                raise ValueError("allocated selection is not one JSON object")
            selection = binding.selection_validator(raw_selection)
            run_id = str(sleeve.get("run_id") or "")
            if (
                selection.get("selection_id") != run_id
                or selection.get("strategy_version") != strategy_id
            ):
                raise ValueError("allocated selection and durable run identity disagree")
            ledger_path = _repo_path(self.repository_root, binding.ledger_path)
            records = _matching_execution_records(
                self._read_ledger(ledger_path),
                binding=binding,
                run_id=run_id,
            )
            latest = records[-1] if records else None
            latest_evidence = (
                latest.get("evidence")
                if isinstance(latest, Mapping)
                and isinstance(latest.get("evidence"), Mapping)
                else {}
            )
            latest_plan = latest_evidence.get("plan")
            latest_plan = dict(latest_plan) if isinstance(latest_plan, Mapping) else {}
            execution_context = latest_plan.get("execution_state_context")
            execution_context = (
                dict(execution_context)
                if isinstance(execution_context, Mapping)
                else {}
            )
            broker = _latest_mapping(records, "broker_state")
            risk = _latest_mapping(records, "risk_state")
            selected_broker = selection.get("broker_at_selection")
            if broker is None and isinstance(selected_broker, Mapping):
                broker = dict(selected_broker)
            broker = broker or {}
            selected_positions = broker.get("positions")
            positions = _position_mapping(selected_positions)
            open_orders = broker.get("open_orders")
            open_orders = list(open_orders) if isinstance(open_orders, list) else []
            timer = dict(self._unit_reader(binding.timer_unit))
            service = dict(self._unit_reader(binding.service_unit))
            pending = list(_pending_order_refs(records))
            unit_failure = (
                timer.get("available") is not True
                or service.get("available") is not True
                or timer.get("active_state") == "failed"
                or service.get("active_state") == "failed"
                or timer.get("result") == "failed"
                or service.get("result") == "failed"
            )
            timer_active = timer.get("active_state") == "active"
            service_active = service.get("active_state") == "active"
            flat = _positions_flat(positions)
            state = (
                "BROKEN"
                if unit_failure
                else "UNSAFE_PAUSED"
                if not timer_active and (not flat or pending or open_orders)
                else "BUSY"
                if service_active
                else "RUNNING"
                if timer_active
                else "PAUSED"
            )
            weight_bps = int(sleeve["weight_bps"])
            allocation_cents = managed_capital_cents * weight_bps // 10_000
            allocation = {
                "weight_bps": weight_bps,
                "limit_cents": allocation_cents,
            }
            margin = sleeve.get("margin")
            if isinstance(margin, Mapping):
                allocation = {
                    **allocation,
                    "capital_kind": sleeve.get("capital_kind"),
                    "limit_cents": None,
                    "margin": dict(margin),
                }
            graduation = _latest_graduation(
                _repo_path(self.repository_root, self.graduation_directory),
                run_id=run_id,
                strategy_id=strategy_id,
                execution_strategy_version=binding.execution_strategy_version,
                validator=self._graduation_validator,
            )
            risk = risk or {}
            safety_breaches = list(risk.get("safety_breaches") or ())
            run = {
                "sleeve_id": sleeve["sleeve_id"],
                "strategy_id": strategy_id,
                "execution_strategy_version": binding.execution_strategy_version,
                "run_id": run_id,
                "label": binding.label,
                "valid": True,
                "state": state,
                "errors": [],
                "selection_path": str(selection_path.relative_to(self.repository_root)),
                "selection_file_sha256": selection_sha,
                "allocation": allocation,
                "timer": timer,
                "service": service,
                "ledger_path": str(ledger_path.relative_to(self.repository_root)),
                "ledger_rows": len(records),
                "latest_recorded_at_utc": (
                    latest.get("recorded_at_utc") if isinstance(latest, Mapping) else None
                ),
                "latest_checkpoint_id": (
                    latest.get("checkpoint_id") if isinstance(latest, Mapping) else None
                ),
                "latest_phase": latest_evidence.get("phase"),
                "latest_decision": {
                    "status": latest_plan.get("status"),
                    "reason": latest_plan.get("reason"),
                    "target_direction": latest_plan.get("target_direction"),
                    "target_symbol": latest_plan.get("target_symbol"),
                    "entry_window_open": latest_plan.get("entry_window_open"),
                },
                "execution_state_context": execution_context,
                "pending_order_refs": pending,
                "positions": positions,
                "open_orders": open_orders,
                "settled_cash_usd": (
                    risk.get(
                        "settled_cash_usd",
                        broker.get("settled_cash_usd", broker.get("cash_balance_usd")),
                    )
                ),
                "economics": {
                    "run_net_usd": risk.get("run_net_usd"),
                    "run_realized_net_usd": risk.get("run_realized_net_usd"),
                    "open_mark_net_usd": risk.get("open_mark_net_usd"),
                    "run_cost_usd": risk.get("run_cost_usd"),
                    "drawdown_usd": risk.get("drawdown_usd"),
                    "fill_count": risk.get("fill_count"),
                    "closed_trades": risk.get("closed_trades"),
                },
                "safety": {
                    "valid": risk.get("valid"),
                    "attribution_complete": risk.get("attribution_complete"),
                    "breaches": safety_breaches,
                },
                "graduation": graduation,
            }
            run["controls"] = {
                action: _control_status(run, action)
                for action in sorted(LIVE_RUN_CONTROL_ACTIONS)
            }
            return run
        except (OSError, TypeError, ValueError) as exc:
            return self._invalid_run(sleeve, error=exc, binding=binding)

    def snapshot(self) -> dict[str, object]:
        """Reduce the capital plan, selections, ledgers, proofs, and unit states."""

        try:
            plan = load_live_capital_plan(self.capital_plan_path)
        except (OSError, TypeError, ValueError) as exc:
            body = {
                "schema": LIVE_RUN_COCKPIT_SCHEMA,
                "status": "QUARANTINED",
                "capital_plan_id": None,
                "account": None,
                "capital": None,
                "runs": [],
                "errors": [str(exc)],
            }
            return {**body, "snapshot_id": _identity(body)}
        capital = plan["capital"]
        assert isinstance(capital, Mapping)
        runs = [
            self._project_run(
                sleeve,
                managed_capital_cents=int(capital["managed_capital_cents"]),
            )
            for sleeve in plan["sleeves"]
            if isinstance(sleeve, Mapping)
        ]
        body = {
            "schema": LIVE_RUN_COCKPIT_SCHEMA,
            "status": (
                "READY" if all(run.get("valid") is True for run in runs) else "QUARANTINED"
            ),
            "capital_plan_id": plan["plan_id"],
            "account": dict(plan["account"]),
            "capital": dict(capital),
            "runs": runs,
            "errors": [],
        }
        return {**body, "snapshot_id": _identity(body)}

    def control(self, sleeve_id: str, action: str) -> dict[str, object]:
        """Start or flat-safely pause one schedule; never submit a broker order."""

        normalized = str(action or "").upper()
        before = self.snapshot()
        matches = [
            run
            for run in before["runs"]
            if isinstance(run, Mapping) and run.get("sleeve_id") == sleeve_id
        ]
        if len(matches) != 1:
            raise ValueError("durable-run control target is missing or duplicated")
        run = matches[0]
        decision = _control_status(run, normalized)
        if decision["status"] == "HOLD":
            raise ValueError(", ".join(decision["reasons"]))
        if decision["status"] == "NOOP":
            return {"action": normalized, "decision": decision, "before": before, "after": before}
        timer = run["timer"]
        assert isinstance(timer, Mapping)
        timer_unit = str(timer["unit"])
        if normalized == "START":
            self._command_runner(("enable", "--now", timer_unit))
        elif normalized == "STOP":
            self._command_runner(("disable", "--now", timer_unit))
        else:
            raise ValueError(", ".join(decision["reasons"]))
        after = self.snapshot()
        updated = next(
            run
            for run in after["runs"]
            if isinstance(run, Mapping) and run.get("sleeve_id") == sleeve_id
        )
        updated_timer = updated.get("timer")
        expected_active = normalized == "START"
        actual_active = (
            isinstance(updated_timer, Mapping)
            and updated_timer.get("active_state") == "active"
        )
        if normalized == "STOP":
            updated_positions = updated.get("positions")
            updated_service = updated.get("service")
            unsafe = (
                not isinstance(updated_positions, Mapping)
                or not _positions_flat(updated_positions)
                or bool(updated.get("open_orders"))
                or bool(updated.get("pending_order_refs"))
                or not isinstance(updated_service, Mapping)
                or updated_service.get("active_state") != "inactive"
            )
            if unsafe:
                self._command_runner(("enable", "--now", timer_unit))
                raise RuntimeError("durable run changed during pause; schedule restored")
        if actual_active != expected_active:
            raise RuntimeError("durable-run schedule state did not change as requested")
        return {"action": normalized, "decision": decision, "before": before, "after": after}
