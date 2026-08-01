"""Graduation proof for XSP capital-owner generations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .live_graduation import live_calibration_logical_prefix


XSP_CAPITAL_OWNER_STABILITY_SCHEMA = (
    "xsp.opening-edge-v3-capital-owner-stability-manifest.v1"
)
XSP_CAPITAL_OWNER_GENERATION_SCHEMA = (
    "xsp.opening-edge-v3-capital-owner-stability-manifest.v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(
    status: str,
    reasons: Sequence[str],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def xsp_zero_capital_migration_evidence(
    records: Sequence[Mapping[str, object]],
    *,
    cutoff_utc: str,
    strategy_id: str,
    strategy_version: str,
    run_id: str,
) -> dict[str, object]:
    """Prove a selected prefix observed state but never exercised capital."""

    prefix, projected = live_calibration_logical_prefix(
        records,
        cutoff_utc=cutoff_utc,
    )
    rows = [
        row
        for row in projected
        if row.get("kind") == "checkpoint"
        and row.get("strategy_id") == strategy_id
        and row.get("strategy_version") == strategy_version
        and isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("selection_id") == run_id
    ]
    reasons = []
    submitted_orders = fill_count = closed_trades = 0.0
    maximum_absolute_net = maximum_cost = maximum_drawdown = 0.0
    order_refs: set[str] = set()
    nonflat_risk_rows = nonflat_broker_rows = broker_open_order_rows = 0
    for row in rows:
        evidence = row["evidence"]
        assert isinstance(evidence, Mapping)
        risk = evidence.get("risk_state")
        broker = evidence.get("broker_state")
        plan = evidence.get("plan")
        if (
            evidence.get("phase") != "STATE"
            or not isinstance(risk, Mapping)
            or not isinstance(broker, Mapping)
            or not isinstance(plan, Mapping)
        ):
            reasons.append("pre_migration_state_invalid")
            continue
        try:
            submitted_orders += float(evidence.get("submitted_orders") or 0)
            fill_count = max(fill_count, float(risk["fill_count"]))
            closed_trades = max(closed_trades, float(risk["closed_trades"]))
            maximum_absolute_net = max(
                maximum_absolute_net,
                abs(float(risk["run_net_usd"])),
                abs(float(risk["run_realized_net_usd"])),
                abs(float(risk["open_mark_net_usd"])),
            )
            maximum_cost = max(maximum_cost, abs(float(risk["run_cost_usd"])))
            maximum_drawdown = max(maximum_drawdown, float(risk["drawdown_usd"]))
            risk_holdings = risk["holdings_from_fills"]
            broker_positions = broker["positions"]
        except (KeyError, TypeError, ValueError):
            reasons.append("pre_migration_economics_invalid")
            continue
        order_ref = str(evidence.get("order_ref") or "")
        if order_ref:
            order_refs.add(order_ref)
        if not isinstance(risk_holdings, Mapping) or any(
            abs(float(value)) > 1e-9 for value in risk_holdings.values()
        ):
            nonflat_risk_rows += 1
        if not isinstance(broker_positions, Mapping) or any(
            abs(float(value)) > 1e-9 for value in broker_positions.values()
        ):
            nonflat_broker_rows += 1
        if broker.get("open_orders") != []:
            broker_open_order_rows += 1
        if plan.get("leg") is not None:
            reasons.append("pre_migration_actionable_leg_present")
    if not rows:
        reasons.append("pre_migration_selected_prefix_missing")
    numeric = (
        submitted_orders,
        fill_count,
        closed_trades,
        maximum_absolute_net,
        maximum_cost,
        maximum_drawdown,
    )
    if not all(math.isfinite(value) and abs(value) <= 1e-9 for value in numeric):
        reasons.append("pre_migration_capital_event_present")
    if order_refs or nonflat_risk_rows or nonflat_broker_rows or broker_open_order_rows:
        reasons.append("pre_migration_broker_event_present")
    return {
        "status": "INVALID" if reasons else "PASS",
        "reasons": sorted(set(reasons)),
        "ledger_prefix": prefix,
        "selected_execution_rows": len(rows),
        "submitted_orders": submitted_orders,
        "order_refs": sorted(order_refs),
        "max_fill_count": fill_count,
        "max_closed_trades": closed_trades,
        "max_absolute_net_usd": maximum_absolute_net,
        "max_cost_usd": maximum_cost,
        "max_drawdown_usd": maximum_drawdown,
        "nonflat_risk_rows": nonflat_risk_rows,
        "nonflat_broker_rows": nonflat_broker_rows,
        "broker_open_order_rows": broker_open_order_rows,
    }


def xsp_capital_owner_stability_graduation_gate(
    path: Path,
    *,
    repo_root: Path,
    selection_id: str,
    selection_file_sha256: str,
    records: Sequence[Mapping[str, object]] = (),
    strategy_id: str = "",
    strategy_version: str = "",
) -> dict[str, object]:
    """Rehash one frozen owner generation and validate any migration prefix."""

    try:
        raw = path.read_bytes()
        proof = json.loads(raw)
        if not isinstance(proof, Mapping):
            raise ValueError("capital owner proof must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _gate(
            "INVALID",
            ["capital_owner_manifest_unreadable"],
            {"path": str(path), "error": str(exc)},
        )
    reasons = []
    root = repo_root.resolve()
    schema = proof.get("schema")
    selection = proof.get("selection")
    owners = proof.get("capital_semantic_surface")
    checks = proof.get("checks")
    boundaries = proof.get("boundaries")
    if (
        schema not in {XSP_CAPITAL_OWNER_STABILITY_SCHEMA, XSP_CAPITAL_OWNER_GENERATION_SCHEMA}
        or proof.get("authority") != "frozen_post_selection_capital_owner_manifest"
        or proof.get("verdict") != "PASS_CAPITAL_OWNER_STABLE"
    ):
        reasons.append("capital_owner_manifest_verdict_invalid")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_id") != selection_id
        or selection.get("selection_file_sha256") != selection_file_sha256
    ):
        reasons.append("capital_owner_selection_mismatch")
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        reasons.append("capital_owner_checks_invalid")
    if not isinstance(owners, Mapping) or not owners:
        reasons.append("capital_owner_surface_missing")
    else:
        for relative, expected in sorted(owners.items()):
            owner = (root / str(relative)).resolve()
            if root not in owner.parents:
                reasons.append(f"capital_owner_path_invalid:{relative}")
                continue
            try:
                current = _sha256(owner)
            except OSError:
                reasons.append(f"capital_owner_missing:{relative}")
                continue
            if current != expected:
                reasons.append(f"capital_owner_drift:{relative}")
    migration_evidence = None
    if schema == XSP_CAPITAL_OWNER_GENERATION_SCHEMA:
        migration = proof.get("migration")
        if not isinstance(migration, Mapping):
            reasons.append("capital_owner_migration_missing")
        else:
            predecessor = (root / str(migration.get("predecessor_path") or "")).resolve()
            try:
                predecessor_proof = json.loads(predecessor.read_bytes())
                predecessor_valid = (
                    root in predecessor.parents
                    and _sha256(predecessor) == migration.get("predecessor_sha256")
                    and predecessor_proof.get("schema") == XSP_CAPITAL_OWNER_STABILITY_SCHEMA
                    and predecessor_proof.get("selection") == selection
                )
            except (AttributeError, OSError, json.JSONDecodeError):
                predecessor_valid = False
            if not predecessor_valid:
                reasons.append("capital_owner_predecessor_invalid")
            try:
                migration_evidence = xsp_zero_capital_migration_evidence(
                    records,
                    cutoff_utc=str(migration.get("effective_at_utc") or ""),
                    strategy_id=strategy_id,
                    strategy_version=strategy_version,
                    run_id=selection_id,
                )
            except (TypeError, ValueError):
                migration_evidence = None
            if (
                migration_evidence is None
                or migration_evidence.get("status") != "PASS"
                or migration.get("zero_capital_prefix") != migration_evidence
            ):
                reasons.append("capital_owner_migration_prefix_invalid")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("broker_queried") is not False
        or boundaries.get("service_or_timer_mutated") is not False
        or boundaries.get("selection_mutated") is not False
        or boundaries.get("submitted_orders") != 0
        or boundaries.get("profitability_clock_mutated") is not False
    ):
        reasons.append("capital_owner_safety_boundary_invalid")
    return _gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "observed_at_utc": proof.get("observed_at_utc"),
            "source_revision": proof.get("source_revision"),
            "owner_count": len(owners) if isinstance(owners, Mapping) else 0,
            "migration_prefix_sha256": (
                migration_evidence["ledger_prefix"]["sha256"]
                if isinstance(migration_evidence, Mapping)
                else None
            ),
        },
    )
