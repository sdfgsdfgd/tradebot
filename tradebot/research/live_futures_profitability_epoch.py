"""Content-addressed prospective coverage clocks for futures profitability."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .live_futures_profitability import (
    FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA,
    FUTURES_RISK_FIELDS,
    FuturesProfitabilitySpec,
    selected_futures_rows,
    single_contract_position,
)
from .live_graduation import (
    evidence_sha256,
    validate_live_graduation_receipt,
)


_CHECKPOINT_FIELDS = (
    "evaluation_as_of_utc",
    "strategy_id",
    "strategy_version",
    "trading_date",
    "session",
    "status",
    "evidence",
    "recorded_at_utc",
)


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("futures coverage timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _bound_path(path: Path, root: Path, *, label: str) -> tuple[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"futures coverage {label} escaped repository") from exc
    return relative, hashlib.sha256(resolved.read_bytes()).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"futures coverage {label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"futures coverage {label} must be an object")
    return value


def _checkpoint_addressed(row: Mapping[str, object]) -> bool:
    try:
        full = {field: row[field] for field in _CHECKPOINT_FIELDS}
    except KeyError:
        return False
    without_recorded = {
        field: value for field, value in full.items() if field != "recorded_at_utc"
    }
    return row.get("checkpoint_id") in {
        evidence_sha256(full),
        evidence_sha256(without_recorded),
    }


def _risk(
    row: Mapping[str, object], spec: FuturesProfitabilitySpec
) -> Mapping[str, object] | None:
    evidence = row.get("evidence")
    raw = evidence.get("risk_state") if isinstance(evidence, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    projected = spec.risk_projection(raw)
    return projected if isinstance(projected, Mapping) else None


def _risk_numbers(risk: Mapping[str, object]) -> dict[str, float]:
    return {field: float(risk[field]) for field in FUTURES_RISK_FIELDS}


def build_futures_profitability_coverage_epoch(
    *,
    selection_id: str,
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    spec: FuturesProfitabilitySpec,
    con_id: int,
    predecessor_receipt_paths: Sequence[Path],
    preregistration_path: Path,
    registered_at_utc: datetime | str,
    eligible_start_utc: datetime | str,
    repo_root: Path,
) -> dict[str, object]:
    """Freeze one future futures coverage clock without resetting economics."""

    registered = _utc(registered_at_utc)
    eligible = _utc(eligible_start_utc)
    root = repo_root.resolve()
    selection_relative, selection_sha = _bound_path(
        selection_path, root, label="selection"
    )
    prereg_relative, prereg_sha = _bound_path(
        preregistration_path, root, label="preregistration"
    )
    preregistration = _load_object(
        preregistration_path, label="preregistration"
    )
    prereg_selection = preregistration.get("selection")
    if (
        len(selection_id) != 64
        or any(value not in "0123456789abcdef" for value in selection_id)
        or registered >= eligible
        or eligible not in spec.evaluation_slots(eligible, eligible, True)
        or preregistration.get("registered_at_utc") != registered.isoformat()
        or preregistration.get("eligible_start_utc") != eligible.isoformat()
        or not isinstance(prereg_selection, Mapping)
        or prereg_selection.get("selection_id") != selection_id
        or prereg_selection.get("path") != selection_relative
        or prereg_selection.get("sha256") != selection_sha
    ):
        raise ValueError("futures coverage epoch identity is invalid")

    candidates = []
    for row in selected_futures_rows(
        records, selection_id=selection_id, spec=spec
    ):
        evidence = row.get("evidence")
        try:
            recorded = _utc(str(row["recorded_at_utc"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            row.get("status") == "EVALUATED"
            and isinstance(evidence, Mapping)
            and evidence.get("phase") == "STATE"
            and recorded <= registered
            and _checkpoint_addressed(row)
        ):
            candidates.append((recorded, row, evidence))
    if not candidates:
        raise ValueError("futures coverage epoch has no terminal checkpoint")
    _recorded, terminal_row, terminal_evidence = max(
        candidates,
        key=lambda item: (item[0], str(item[1].get("checkpoint_id") or "")),
    )
    terminal_risk = _risk(terminal_row, spec)
    terminal_broker = terminal_evidence.get("broker_state")
    terminal_plan = terminal_evidence.get("plan")
    if not isinstance(terminal_broker, Mapping):
        raise ValueError("futures coverage terminal broker state is invalid")
    try:
        terminal_values = _risk_numbers(terminal_risk or {})
        broker_position = single_contract_position(
            terminal_broker, symbol=spec.symbol, con_id=con_id
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("futures coverage terminal risk is invalid") from exc
    if (
        not isinstance(terminal_risk, Mapping)
        or terminal_risk.get("valid") is not True
        or terminal_risk.get("attribution_complete") is not True
        or terminal_risk.get("safety_breaches") != []
        or terminal_broker.get("open_orders") != []
        or not isinstance(terminal_plan, Mapping)
        or terminal_plan.get("held_direction") is not None
        or terminal_evidence.get("submitted_orders") != 0
        or broker_position != 0
        or abs(terminal_values["position_from_fills"]) > 1e-9
        or any(
            abs(terminal_values[field]) > 1e-9
            for field in (
                "open_mark_gross_usd",
                "open_mark_cost_usd",
                "open_mark_net_usd",
            )
        )
        or not all(math.isfinite(value) for value in terminal_values.values())
        or terminal_values["run_cost_usd"] < -1e-9
        or terminal_values["drawdown_usd"] < -1e-9
    ):
        raise ValueError("futures coverage epoch requires terminal flat truth")

    predecessors = []
    for receipt_path in predecessor_receipt_paths:
        relative, fingerprint = _bound_path(
            receipt_path, root, label="predecessor receipt"
        )
        receipt = _load_object(receipt_path, label="predecessor receipt")
        validated = validate_live_graduation_receipt(receipt)
        subject = receipt.get("subject")
        target = receipt.get("target")
        try:
            cutoff = _utc(target["cutoff_utc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "futures coverage predecessor cutoff is invalid"
            ) from exc
        if (
            not isinstance(subject, Mapping)
            or subject.get("selection_id") != selection_id
            or not isinstance(target, Mapping)
            or cutoff > registered
        ):
            raise ValueError("futures coverage predecessor is invalid")
        predecessors.append(
            {
                "path": relative,
                "sha256": fingerprint,
                "receipt_id": validated["receipt_id"],
                "milestone": target["milestone"],
                "cutoff_utc": target["cutoff_utc"],
                "verdict": receipt["verdict"],
            }
        )
    predecessors.sort(key=lambda row: (str(row["cutoff_utc"]), str(row["path"])))
    if (
        not predecessors
        or len({row["path"] for row in predecessors}) != len(predecessors)
        or len({row["receipt_id"] for row in predecessors}) != len(predecessors)
    ):
        raise ValueError("futures coverage predecessors are incomplete")

    body = {
        "schema": FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA,
        "authority": (
            "prospective_coverage_clock_only_cumulative_economics_unchanged"
        ),
        "registered_at_utc": registered.isoformat(),
        "eligible_start_utc": eligible.isoformat(),
        "selection": {
            "selection_id": selection_id,
            "path": selection_relative,
            "sha256": selection_sha,
        },
        "preregistration": {"path": prereg_relative, "sha256": prereg_sha},
        "terminal_checkpoint": {
            "checkpoint_id": terminal_row["checkpoint_id"],
            "recorded_at_utc": terminal_row["recorded_at_utc"],
            "evaluation_as_of_utc": terminal_row["evaluation_as_of_utc"],
            "evidence_sha256": evidence_sha256(terminal_evidence),
            "risk_state": {
                **{
                    field: terminal_values[field]
                    for field in FUTURES_RISK_FIELDS
                },
                "valid": True,
                "attribution_complete": True,
                "safety_breaches": [],
            },
            "broker_position": 0.0,
            "broker_open_orders": [],
            "submitted_orders": 0,
        },
        "predecessor_receipts": predecessors,
        "invariants": {
            "strategy_selection_changed": False,
            "runtime_risk_reset": False,
            "capital_plan_changed": False,
            "pre_epoch_economics_inherited": True,
            "missing_slots_backfilled": False,
            "profitability_threshold_changed": False,
            "broker_or_timer_mutated": False,
        },
        "submitted_orders": 0,
    }
    return {**body, "epoch_id": evidence_sha256(body)}


def load_futures_profitability_coverage_epoch(
    path: Path,
    *,
    selection_id: str,
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    spec: FuturesProfitabilitySpec,
    con_id: int,
    repo_root: Path,
) -> dict[str, object]:
    """Rebuild and validate one content-addressed futures coverage epoch."""

    epoch = _load_object(path, label="epoch")
    predecessors = epoch.get("predecessor_receipts")
    preregistration = epoch.get("preregistration")
    if (
        epoch.get("schema") != FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA
        or not isinstance(predecessors, Sequence)
        or isinstance(predecessors, (str, bytes))
        or not isinstance(preregistration, Mapping)
    ):
        raise ValueError("invalid futures profitability coverage epoch")
    try:
        rebuilt = build_futures_profitability_coverage_epoch(
            selection_id=selection_id,
            selection_path=selection_path,
            records=records,
            spec=spec,
            con_id=con_id,
            predecessor_receipt_paths=[
                repo_root / str(row["path"])
                for row in predecessors
                if isinstance(row, Mapping)
            ],
            preregistration_path=repo_root / str(preregistration["path"]),
            registered_at_utc=epoch["registered_at_utc"],
            eligible_start_utc=epoch["eligible_start_utc"],
            repo_root=repo_root,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid futures profitability coverage epoch") from exc
    if rebuilt != epoch:
        raise ValueError("futures profitability coverage epoch identity drift")
    return epoch
