"""Pure, cutoff-bound graduation for immutable live-strategy evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


LIVE_GRADUATION_SCHEMA = "live.strategy-graduation.v1"
LIVE_GRADUATION_PREFIX_SCHEMA = "live.calibration-logical-prefix.v1"
LIVE_GRADUATION_VERDICTS = {"PROMOTE", "HOLD", "REVISE", "QUARANTINE", "STOP"}
LIVE_GRADUATION_TARGETS = {
    "24h": (),
    "48h": ("24h",),
    "five_session_week": ("24h", "48h"),
}

_GATE_ORDER = (
    "identity",
    "runtime_parity",
    "capital_owner_stability",
    "restart",
    "cash_risk_safety",
    "attribution",
    "execution",
    "profitability",
)
_GATE_STATUSES = {"PASS", "HOLD", "FAIL", "INVALID", "STOP"}
_IMMATURE_PROFITABILITY_REASONS = {
    "elapsed_time_incomplete",
    "eligible_sessions_incomplete",
}
_STOP_PROFITABILITY_REASONS = {
    "drawdown_limit_breached",
    "session_loss_limit_breached",
    "selected_safety_breach",
}
_SUBJECT_FIELDS = {
    "strategy_id",
    "strategy_version",
    "signal_instrument",
    "execution_sleeve",
    "capital_sleeve",
    "selection_id",
    "run_id",
    "account_fingerprint",
}


def canonical_json_bytes(value: object) -> bytes:
    """Return the shared compact canonical JSON representation."""

    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def evidence_sha256(value: object) -> str:
    """Content-address one evidence value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_identity(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _aware_utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("graduation timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _record_id(record: Mapping[str, object]) -> str:
    key = {
        "forecast": "forecast_id",
        "result": "result_id",
        "checkpoint": "checkpoint_id",
    }.get(str(record.get("kind")))
    return str(record.get(key) or "") if key else ""


def _record_time(record: Mapping[str, object]) -> datetime:
    key = "settled_at_utc" if record.get("kind") == "result" else "recorded_at_utc"
    if key not in record:
        raise ValueError("calibration record has no authoritative evidence time")
    return _aware_utc(str(record[key]))


def _dependencies(record: Mapping[str, object]) -> tuple[str, ...]:
    if record.get("kind") == "result":
        return (str(record.get("forecast_id") or ""),)
    evidence = record.get("evidence")
    if record.get("kind") == "checkpoint" and isinstance(evidence, Mapping):
        source_id = str(evidence.get("source_checkpoint_id") or "")
        if source_id and evidence.get("selection_id"):
            return (source_id,)
    return ()


def live_calibration_logical_prefix(
    records: Sequence[Mapping[str, object]],
    *,
    cutoff_utc: datetime | str,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Project one dependency-closed cutoff in immutable JSONL order."""

    cutoff = _aware_utc(cutoff_utc)
    frozen = tuple(dict(record) for record in records)
    ids = [_record_id(record) for record in frozen]
    if any(not identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError("calibration record identities are missing or duplicated")
    times = [_record_time(record) for record in frozen]
    candidates = [
        (identity, record, stamp)
        for identity, record, stamp in zip(ids, frozen, times)
        if stamp <= cutoff
    ]
    candidate_times = [stamp for _, _, stamp in candidates]
    clock_regressions = sum(
        current < prior
        for prior, current in zip(candidate_times, candidate_times[1:])
    )
    included = {identity for identity, _, _ in candidates}
    excluded_dependencies: set[str] = set()
    changed = True
    while changed:
        changed = False
        for identity, record in zip(ids, frozen):
            if identity not in included:
                continue
            dependencies = _dependencies(record)
            if any(not dependency or dependency not in included for dependency in dependencies):
                included.remove(identity)
                excluded_dependencies.add(identity)
                changed = True
    projected = tuple(
        record for identity, record in zip(ids, frozen) if identity in included
    )
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in projected)
    projected_ids = [_record_id(record) for record in projected]
    descriptor = {
        "schema": LIVE_GRADUATION_PREFIX_SCHEMA,
        "cutoff_utc": cutoff.isoformat(),
        "candidate_records": len(candidates),
        "included_records": len(projected),
        "excluded_for_dependency": len(excluded_dependencies),
        "clock_regressions": clock_regressions,
        "first_record_id": projected_ids[0] if projected_ids else None,
        "last_record_id": projected_ids[-1] if projected_ids else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return descriptor, projected


def _gate(
    value: Mapping[str, object] | None,
    *,
    missing_reason: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"status": "INVALID", "reasons": [missing_reason], "evidence": None}
    status = str(value.get("status") or "").upper()
    reasons = value.get("reasons")
    evidence = value.get("evidence")
    if (
        status not in _GATE_STATUSES
        or not isinstance(reasons, Sequence)
        or isinstance(reasons, (str, bytes))
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or not isinstance(evidence, Mapping)
    ):
        return {
            "status": "INVALID",
            "reasons": [f"invalid_{missing_reason}"],
            "evidence": None,
        }
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def _identity_gate(
    *,
    cutoff: datetime,
    subject: Mapping[str, object],
    selection: Mapping[str, object],
    selection_file_sha256: str,
    ledger_prefix: Mapping[str, object],
    profitability_receipt: Mapping[str, object],
) -> dict[str, object]:
    reasons = []
    if _SUBJECT_FIELDS - set(subject) or any(
        not str(subject.get(field) or "").strip() for field in _SUBJECT_FIELDS
    ):
        reasons.append("invalid_subject_identity")
    if any(
        not _sha256_identity(subject.get(field))
        for field in ("selection_id", "run_id", "account_fingerprint")
    ):
        reasons.append("invalid_subject_content_identity")
    if not _sha256_identity(selection_file_sha256):
        reasons.append("invalid_selection_file_sha256")
    for field in ("selection_id", "run_id", "capital_sleeve"):
        if str(selection.get(field) or "") != str(subject.get(field) or ""):
            reasons.append(f"selection_{field}_mismatch")
    if selection.get("selection_file_sha256") != selection_file_sha256:
        reasons.append("selection_file_identity_mismatch")
    if (
        ledger_prefix.get("schema") != LIVE_GRADUATION_PREFIX_SCHEMA
        or ledger_prefix.get("cutoff_utc") != cutoff.isoformat()
        or not _sha256_identity(ledger_prefix.get("sha256"))
    ):
        reasons.append("invalid_ledger_prefix_identity")
    policy = profitability_receipt.get("policy")
    if not isinstance(policy, Mapping):
        reasons.append("missing_profitability_policy")
    else:
        expected = {
            "run_id": subject.get("run_id"),
            "strategy_id": subject.get("strategy_id"),
            "strategy_version": subject.get("strategy_version"),
            "capital_sleeve": subject.get("capital_sleeve"),
        }
        if any(str(policy.get(key) or "") != str(value or "") for key, value in expected.items()):
            reasons.append("profitability_policy_identity_mismatch")
        coverage_epoch_id = policy.get("coverage_epoch_id")
        coverage_started_at = policy.get("coverage_started_at_utc")
        if coverage_epoch_id is not None and (
            selection.get("coverage_epoch_id") != coverage_epoch_id
            or selection.get("coverage_started_at_utc") != coverage_started_at
            or ledger_prefix.get("coverage_epoch_id") != coverage_epoch_id
        ):
            reasons.append("profitability_coverage_epoch_mismatch")
    if profitability_receipt.get("as_of_utc") != cutoff.isoformat():
        reasons.append("profitability_cutoff_mismatch")
    return {
        "status": "INVALID" if reasons else "PASS",
        "reasons": sorted(set(reasons)),
        "evidence": {
            "selection_id": selection.get("selection_id"),
            "selection_file_sha256": selection_file_sha256,
            "ledger_prefix_sha256": ledger_prefix.get("sha256"),
            "ledger_prefix_records": ledger_prefix.get("included_records"),
        },
    }


def _profitability_gate(
    receipt: Mapping[str, object],
    *,
    target: str,
) -> dict[str, object]:
    reasons = set(str(reason) for reason in receipt.get("reasons", ()) if reason)
    milestones = receipt.get("milestones")
    required = (*LIVE_GRADUATION_TARGETS[target], target)
    evidence = {"target": target, "required_milestones": list(required)}
    if reasons & _STOP_PROFITABILITY_REASONS:
        return {"status": "STOP", "reasons": sorted(reasons), "evidence": evidence}
    if receipt.get("status") == "INVALID_EVIDENCE" or not isinstance(milestones, Mapping):
        return {
            "status": "INVALID",
            "reasons": sorted(reasons or {"invalid_profitability_evidence"}),
            "evidence": evidence,
        }
    entries = []
    for name in required:
        milestone = milestones.get(name)
        if not isinstance(milestone, Mapping):
            return {
                "status": "INVALID",
                "reasons": [f"missing_{name}_milestone"],
                "evidence": evidence,
            }
        entries.append(milestone)
        reasons.update(str(reason) for reason in milestone.get("reasons", ()) if reason)
    evidence["milestones"] = {
        name: {
            "passed": milestone.get("passed") is True,
            "evidence_as_of_utc": milestone.get("evidence_as_of_utc"),
        }
        for name, milestone in zip(required, entries)
    }
    if all(milestone.get("passed") is True for milestone in entries):
        return {"status": "PASS", "reasons": [], "evidence": evidence}
    if reasons & _IMMATURE_PROFITABILITY_REASONS:
        status = "HOLD"
    else:
        status = "FAIL"
    return {
        "status": status,
        "reasons": sorted(reasons or {"profitability_milestone_failed"}),
        "evidence": evidence,
    }


def reduce_live_graduation(
    *,
    target_milestone: str,
    cutoff_utc: datetime | str,
    subject: Mapping[str, object],
    selection: Mapping[str, object],
    selection_file_sha256: str,
    ledger_prefix: Mapping[str, object],
    profitability_receipt: Mapping[str, object],
    runtime_parity_proof: Mapping[str, object],
    capital_owner_stability_proof: Mapping[str, object],
) -> dict[str, object]:
    """Reduce one immutable strategy/run/cutoff into one advisory verdict."""

    target = str(target_milestone).strip().lower().replace("-", "_")
    if target == "five_session":
        target = "five_session_week"
    if target not in LIVE_GRADUATION_TARGETS:
        raise ValueError(f"unsupported graduation target: {target_milestone!r}")
    cutoff = _aware_utc(cutoff_utc)
    direct = ledger_prefix.get("gates")
    direct = direct if isinstance(direct, Mapping) else {}
    gates = {
        "identity": _identity_gate(
            cutoff=cutoff,
            subject=subject,
            selection=selection,
            selection_file_sha256=selection_file_sha256,
            ledger_prefix=ledger_prefix,
            profitability_receipt=profitability_receipt,
        ),
        "runtime_parity": _gate(
            runtime_parity_proof,
            missing_reason="runtime_parity_proof",
        ),
        "capital_owner_stability": _gate(
            capital_owner_stability_proof,
            missing_reason="capital_owner_stability_proof",
        ),
        "restart": _gate(direct.get("restart"), missing_reason="restart_evidence"),
        "cash_risk_safety": _gate(
            direct.get("cash_risk_safety"),
            missing_reason="cash_risk_safety_evidence",
        ),
        "attribution": _gate(
            direct.get("attribution"),
            missing_reason="attribution_evidence",
        ),
        "execution": _gate(
            direct.get("execution"),
            missing_reason="execution_evidence",
        ),
        "profitability": _profitability_gate(
            profitability_receipt,
            target=target,
        ),
    }
    statuses = {gate["status"] for gate in gates.values()}
    verdict = (
        "STOP"
        if "STOP" in statuses
        else "QUARANTINE"
        if "INVALID" in statuses
        else "REVISE"
        if "FAIL" in statuses
        else "HOLD"
        if "HOLD" in statuses
        else "PROMOTE"
    )
    reasons = [
        f"{name}:{reason}"
        for name in _GATE_ORDER
        for reason in gates[name]["reasons"]
    ]
    body = {
        "schema": LIVE_GRADUATION_SCHEMA,
        "authority": "cutoff_bound_evidence_reduction_only",
        "subject": dict(subject),
        "target": {
            "milestone": target,
            "cutoff_utc": cutoff.isoformat(),
            "required_predecessors": list(LIVE_GRADUATION_TARGETS[target]),
        },
        "identity": {
            "selection": dict(selection),
            "selection_file_sha256": selection_file_sha256,
            "ledger_prefix": {
                key: value
                for key, value in ledger_prefix.items()
                if key != "gates"
            },
        },
        "gates": gates,
        "verdict": verdict,
        "reasons": reasons,
        "remaining_requirements": (
            reasons if verdict in {"HOLD", "REVISE", "QUARANTINE"} else []
        ),
        "boundaries": {
            "broker_queried": False,
            "service_or_timer_mutated": False,
            "selection_mutated": False,
            "order_authority_mutated": False,
            "submitted_orders": 0,
            "profitability_clock_mutated": False,
        },
    }
    return {**body, "receipt_id": evidence_sha256(body)}


def validate_live_graduation_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate one content-addressed graduation receipt."""

    frozen = dict(receipt)
    receipt_id = str(frozen.pop("receipt_id", ""))
    boundaries = frozen.get("boundaries")
    target = frozen.get("target")
    gates = frozen.get("gates")
    statuses = (
        {gate.get("status") for gate in gates.values()}
        if isinstance(gates, Mapping)
        and set(gates) == set(_GATE_ORDER)
        and all(isinstance(gate, Mapping) for gate in gates.values())
        else set()
    )
    expected_verdict = (
        "STOP"
        if "STOP" in statuses
        else "QUARANTINE"
        if "INVALID" in statuses
        else "REVISE"
        if "FAIL" in statuses
        else "HOLD"
        if "HOLD" in statuses
        else "PROMOTE"
        if statuses == {"PASS"}
        else None
    )
    gate_shape_valid = bool(
        isinstance(gates, Mapping)
        and set(gates) == set(_GATE_ORDER)
        and all(
            isinstance(gate, Mapping)
            and gate.get("status") in _GATE_STATUSES
            and isinstance(gate.get("reasons"), list)
            and all(isinstance(reason, str) and reason for reason in gate["reasons"])
            and isinstance(gate.get("evidence"), Mapping)
            for gate in gates.values()
        )
    )
    expected_reasons = (
        [
            f"{name}:{reason}"
            for name in _GATE_ORDER
            for reason in gates[name]["reasons"]
        ]
        if gate_shape_valid
        else None
    )
    if (
        frozen.get("schema") != LIVE_GRADUATION_SCHEMA
        or frozen.get("authority") != "cutoff_bound_evidence_reduction_only"
        or not isinstance(frozen.get("subject"), Mapping)
        or not isinstance(frozen.get("identity"), Mapping)
        or not isinstance(target, Mapping)
        or target.get("milestone") not in LIVE_GRADUATION_TARGETS
        or target.get("required_predecessors")
        != list(LIVE_GRADUATION_TARGETS.get(str(target.get("milestone")), ()))
        or not gate_shape_valid
        or frozen.get("verdict") not in LIVE_GRADUATION_VERDICTS
        or frozen.get("verdict") != expected_verdict
        or frozen.get("reasons") != expected_reasons
        or frozen.get("remaining_requirements")
        != (
            expected_reasons
            if expected_verdict in {"HOLD", "REVISE", "QUARANTINE"}
            else []
        )
        or receipt_id != evidence_sha256(frozen)
        or boundaries
        != {
            "broker_queried": False,
            "service_or_timer_mutated": False,
            "selection_mutated": False,
            "order_authority_mutated": False,
            "submitted_orders": 0,
            "profitability_clock_mutated": False,
        }
    ):
        raise ValueError("invalid live graduation receipt")
    return dict(receipt)


def publish_live_graduation_receipt(
    path: Path,
    receipt: Mapping[str, object],
) -> bool:
    """Create one immutable receipt; identical reruns are idempotent."""

    frozen = validate_live_graduation_receipt(receipt)
    payload = (json.dumps(frozen, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = validate_live_graduation_receipt(json.loads(path.read_text()))
        if existing != frozen:
            raise ValueError("graduation receipt identity conflicts with immutable evidence")
        return False
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = validate_live_graduation_receipt(json.loads(path.read_text()))
                if existing != frozen:
                    raise ValueError(
                        "graduation receipt identity conflicts with immutable evidence"
                    )
                return False
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return True
        finally:
            temporary.unlink(missing_ok=True)
