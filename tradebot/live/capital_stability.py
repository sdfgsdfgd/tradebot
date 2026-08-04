"""Immutable proof that one portfolio generation retains its capital owners."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .capital import validate_live_capital_plan
from .capital_packages import load_allocated_live_selection


PORTFOLIO_CAPITAL_STABILITY_SCHEMA = (
    "live.portfolio-capital-owner-stability-manifest.v1"
)
PORTFOLIO_PACKAGE_GENERATION_SCHEMA = "live.portfolio-package-generation.v1"
PORTFOLIO_PACKAGE_GENERATION_DIRECTORY = Path(
    "db/calibration/portfolio_generations"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(root: Path, value: object) -> Path:
    relative = Path(str(value or ""))
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents:
        raise ValueError("portfolio proof path escaped the repository")
    return path


def _gate(
    status: str, reasons: Sequence[str], evidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def publish_portfolio_package_generation(
    repository_root: Path, plan: Mapping[str, object]
) -> tuple[str, str]:
    """Publish the plan and every immutable selected run as one generation."""

    validated = validate_live_capital_plan(plan)
    if validated.get("schema") != "live.capital-plan.v3":
        raise ValueError("portfolio package generation requires a v3 plan")
    root = repository_root.resolve()
    selections = {}
    for sleeve in validated["sleeves"]:
        sleeve_id = str(sleeve["sleeve_id"])
        selected, path, digest = load_allocated_live_selection(
            validated, sleeve_id=sleeve_id, repository_root=root
        )
        successor = selected.get("allocation_successor")
        if (
            not isinstance(successor, Mapping)
            or successor.get("package_id") != sleeve["allocated_package_id"]
        ):
            raise ValueError("selected run and allocated package disagree")
        selections[sleeve_id] = {
            "selection_id": selected["selection_id"],
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }
    generation = {
        "schema": PORTFOLIO_PACKAGE_GENERATION_SCHEMA,
        "authority": "zero-transmission-successor-and-capital-switch",
        "plan": validated,
        "selections": selections,
        "submitted_orders": 0,
    }
    payload = json.dumps(
        generation, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    relative = PORTFOLIO_PACKAGE_GENERATION_DIRECTORY / (
        f"{validated['plan_id']}.json"
    )
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("portfolio package generation escaped the repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("immutable portfolio package generation changed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return relative.as_posix(), hashlib.sha256(payload).hexdigest()


def portfolio_capital_owner_stability_gate(
    path: Path,
    *,
    repo_root: Path,
    sleeve_id: str,
    selection_id: str,
    selection_file_sha256: str,
) -> dict[str, object]:
    """Rehash one complete package generation and all of its selected sleeves."""

    try:
        raw = path.read_bytes()
        proof = json.loads(raw)
        if not isinstance(proof, Mapping):
            raise ValueError("portfolio capital proof must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _gate(
            "INVALID",
            ["portfolio_capital_manifest_unreadable"],
            {"path": str(path), "error": str(exc)},
        )

    reasons: list[str] = []
    root = repo_root.resolve()
    generation_spec = proof.get("generation")
    bound = proof.get("selections")
    owners = proof.get("capital_semantic_surface")
    checks = proof.get("checks")
    boundaries = proof.get("boundaries")
    generation: Mapping[str, object] = {}
    plan: Mapping[str, object] = {}
    generation_path: Path | None = None
    if (
        proof.get("schema") != PORTFOLIO_CAPITAL_STABILITY_SCHEMA
        or proof.get("authority") != "frozen_portfolio_package_generation"
        or proof.get("verdict") != "PASS_CAPITAL_OWNER_STABLE"
    ):
        reasons.append("portfolio_capital_manifest_verdict_invalid")
    try:
        if not isinstance(generation_spec, Mapping):
            raise ValueError
        generation_path = _resolved(root, generation_spec.get("path"))
        generation_raw = generation_path.read_bytes()
        if hashlib.sha256(generation_raw).hexdigest() != generation_spec.get("sha256"):
            raise ValueError
        parsed = json.loads(generation_raw)
        if not isinstance(parsed, Mapping):
            raise ValueError
        generation = parsed
        if (
            generation.get("schema") != PORTFOLIO_PACKAGE_GENERATION_SCHEMA
            or generation.get("authority")
            != "zero-transmission-successor-and-capital-switch"
            or generation.get("submitted_orders") != 0
            or not isinstance(generation.get("plan"), Mapping)
        ):
            raise ValueError
        plan = validate_live_capital_plan(generation["plan"])
        if (
            plan.get("schema") != "live.capital-plan.v3"
            or plan.get("plan_id") != generation_spec.get("plan_id")
        ):
            raise ValueError
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        reasons.append("portfolio_package_generation_invalid")

    sleeves = plan.get("sleeves") if isinstance(plan, Mapping) else None
    generation_selections = generation.get("selections")
    if (
        not isinstance(bound, Mapping)
        or not isinstance(sleeves, Sequence)
        or isinstance(sleeves, (str, bytes))
        or not isinstance(generation_selections, Mapping)
    ):
        reasons.append("portfolio_selection_bindings_invalid")
        sleeves = ()
    requested = None
    observed_bindings: list[tuple[str, str, str]] = []
    sleeve_ids: set[str] = set()
    for raw_sleeve in sleeves:
        if not isinstance(raw_sleeve, Mapping):
            reasons.append("portfolio_selection_bindings_invalid")
            continue
        current_sleeve = str(raw_sleeve.get("sleeve_id") or "")
        sleeve_ids.add(current_sleeve)
        binding = bound.get(current_sleeve) if isinstance(bound, Mapping) else None
        if not isinstance(binding, Mapping):
            reasons.append(f"portfolio_selection_binding_missing:{current_sleeve}")
            continue
        expected = {
            "selection_id": raw_sleeve.get("run_id"),
            "selection_path": raw_sleeve.get("selection_path"),
            "selection_file_sha256": raw_sleeve.get("selection_file_sha256"),
            "allocated_package_id": raw_sleeve.get("allocated_package_id"),
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            reasons.append(f"portfolio_selection_binding_mismatch:{current_sleeve}")
            continue
        try:
            selected_path = _resolved(root, binding["selection_path"])
            selected_raw = selected_path.read_bytes()
            selected = json.loads(selected_raw)
            selected_valid = (
                hashlib.sha256(selected_raw).hexdigest()
                == binding["selection_file_sha256"]
                and isinstance(selected, Mapping)
                and selected.get("selection_id") == binding["selection_id"]
                and selected.get("strategy_version") == raw_sleeve.get("strategy_id")
                and isinstance(selected.get("allocation_successor"), Mapping)
                and selected["allocation_successor"].get("package_id")
                == binding["allocated_package_id"]
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            selected_valid = False
        if not selected_valid:
            reasons.append(f"portfolio_selected_run_invalid:{current_sleeve}")
        observed_bindings.append(
            (
                str(binding.get("selection_id") or ""),
                str(binding.get("selection_path") or ""),
                str(binding.get("selection_file_sha256") or ""),
            )
        )
        if current_sleeve == sleeve_id:
            requested = binding
    generation_bindings = sorted(
        (
            str(value.get("selection_id") or ""),
            str(value.get("path") or ""),
            str(value.get("sha256") or ""),
        )
        for value in generation_selections.values()
        if isinstance(value, Mapping)
    ) if isinstance(generation_selections, Mapping) else []
    if sorted(observed_bindings) != generation_bindings:
        reasons.append("portfolio_generation_selection_set_mismatch")
    if isinstance(bound, Mapping) and set(bound) != sleeve_ids:
        reasons.append("portfolio_manifest_selection_set_mismatch")
    if (
        not isinstance(requested, Mapping)
        or requested.get("selection_id") != selection_id
        or requested.get("selection_file_sha256") != selection_file_sha256
    ):
        reasons.append("portfolio_requested_selection_mismatch")

    if not isinstance(owners, Mapping) or not owners:
        reasons.append("portfolio_capital_surface_missing")
    else:
        for relative, expected in sorted(owners.items()):
            try:
                current = _sha256(_resolved(root, relative))
            except (OSError, ValueError):
                reasons.append(f"portfolio_capital_owner_missing:{relative}")
                continue
            if current != expected:
                reasons.append(f"portfolio_capital_owner_drift:{relative}")
    if not isinstance(checks, Mapping) or not checks or any(
        value is not True for value in checks.values()
    ):
        reasons.append("portfolio_capital_checks_invalid")
    if (
        not isinstance(boundaries, Mapping)
        or boundaries.get("broker_queried") is not False
        or boundaries.get("service_or_timer_mutated") is not False
        or boundaries.get("selection_mutated") is not False
        or boundaries.get("submitted_orders") != 0
        or boundaries.get("profitability_clock_mutated") is not False
    ):
        reasons.append("portfolio_capital_safety_boundary_invalid")
    return _gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generation_path": str(generation_path) if generation_path else None,
            "generation_sha256": (
                generation_spec.get("sha256")
                if isinstance(generation_spec, Mapping)
                else None
            ),
            "plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
            "sleeves": len(sleeves),
            "source_revision": proof.get("source_revision"),
        },
    )
