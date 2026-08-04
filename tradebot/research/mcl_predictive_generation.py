"""Immutable generation ownership for prospective MCL onset treatments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION


MCL_PREDICTIVE_GENERATION_SCHEMA_V1 = (
    "mcl.predictive-onset-accumulator-generation.v1"
)
MCL_PREDICTIVE_GENERATION_SCHEMA_V2 = (
    "mcl.predictive-onset-accumulator-generation.v2"
)
MCL_PREDICTIVE_GENERATION_AUTHORITY = (
    "prospective_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_PREDICTIVE_PREDECESSOR_GENERATION_PATH = Path(
    "backtests/mcl/mcl_predictive_onset_accumulator_generation.json"
)
MCL_PREDICTIVE_RUNTIME_GENERATION_PATH = Path(
    "db/calibration/mcl_predictive_onset_generation.json"
)
MCL_PREDICTIVE_GENERATION_DIRECTORY = Path(
    "db/calibration/predictive_generations"
)

_CURRENT_ARTIFACTS = {
    "accumulator_owner": "tradebot/research/mcl_predictive_accumulator.py",
    "selected_live_source_owner": "tradebot/research/mcl_live_transport.py",
    "generation_owner": "tradebot/research/mcl_predictive_generation.py",
    "v18_owner": "tradebot/research/mcl_two_speed_auction.py",
    "shock_arbiter_owner": "tradebot/research/mcl_shock_arbiter.py",
    "minute_shock_owner": "tradebot/research/mcl_minute_shock.py",
    "seconds_shock_owner": "tradebot/research/mcl_shock_crest.py",
    "news_contract_owner": "tradebot/news/contract.py",
    "completed_bar_event_owner": "tradebot/research/mcl_predictive_onset.py",
    "velocity_jerk_owner": "tradebot/research/mcl_predictive_velocity.py",
}
_PRESERVED_ARTIFACTS = (
    "turn_tape_generation",
    "stage88_seed",
    "stage89_seed",
    "stage90_preregistration",
    "stage90_seed",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("MCL predictive generation timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _artifact(root: Path, relative: object) -> dict[str, str]:
    value = Path(str(relative or ""))
    path = (root / value).resolve()
    if value.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("MCL predictive generation artifact escaped or is missing")
    return {"path": value.as_posix(), "sha256": _sha256(path)}


def validate_mcl_predictive_generation(
    value: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    generation = dict(value)
    schema = generation.get("schema")
    if (
        schema
        not in {
            MCL_PREDICTIVE_GENERATION_SCHEMA_V1,
            MCL_PREDICTIVE_GENERATION_SCHEMA_V2,
        }
        or generation.get("authority") != MCL_PREDICTIVE_GENERATION_AUTHORITY
        or generation.get("outcomes_exposed") is not False
        or generation.get("submitted_orders") != 0
    ):
        raise ValueError("MCL predictive accumulator generation is invalid")
    if schema == MCL_PREDICTIVE_GENERATION_SCHEMA_V2:
        body = dict(generation)
        generation_id = str(body.pop("generation_id", ""))
        inherited = generation.get("inherited_prefix")
        treatment_ids = (
            inherited.get("treatment_ids") if isinstance(inherited, Mapping) else None
        )
        if (
            not _is_sha(generation_id)
            or _identity(body) != generation_id
            or generation.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
            or not isinstance(treatment_ids, Sequence)
            or isinstance(treatment_ids, (str, bytes))
            or any(not _is_sha(item) for item in treatment_ids)
            or len(set(treatment_ids)) != len(treatment_ids)
            or inherited.get("treatment_count") != len(treatment_ids)
            or inherited.get("treatment_ids_sha256") != _identity(treatment_ids)
        ):
            raise ValueError("MCL predictive successor identity drifted")
    artifacts = generation.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("MCL predictive accumulator generation has no artifacts")
    root = repository_root.resolve()
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or not _is_sha(item.get("sha256")):
            raise ValueError(f"MCL predictive artifact {name} is invalid")
        artifact = root / str(item.get("path") or "")
        if not artifact.is_file() or _sha256(artifact) != item["sha256"]:
            raise ValueError(f"MCL predictive artifact {name} drifted")
    gate = generation.get("cohort_gate")
    if not isinstance(gate, Mapping) or set(gate) != {
        "complete_turns",
        "each_raw_direction",
        "admitted_turns",
        "each_admitted_route",
        "resolved_handoffs",
    }:
        raise ValueError("MCL predictive cohort gate drifted")
    return generation


def load_mcl_predictive_generation(
    path: Path = MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL predictive generation is not an object")
    root = repository_root or Path(__file__).resolve().parents[2]
    return validate_mcl_predictive_generation(value, repository_root=root)


def build_mcl_predictive_successor_generation(
    *,
    repository_root: Path,
    selected: Mapping[str, object],
    inherited_treatment_ids: Sequence[str],
    generated_at: datetime,
    predecessor_path: Path = MCL_PREDICTIVE_PREDECESSOR_GENERATION_PATH,
) -> dict[str, object]:
    root = repository_root.resolve()
    predecessor_file = (root / predecessor_path).resolve()
    predecessor = json.loads(predecessor_file.read_text())
    if (
        root not in predecessor_file.parents
        or not isinstance(predecessor, Mapping)
        or predecessor.get("schema") != MCL_PREDICTIVE_GENERATION_SCHEMA_V1
        or predecessor.get("authority") != MCL_PREDICTIVE_GENERATION_AUTHORITY
        or selected.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or selected.get("authority") != "selected_live_bounded_canary"
        or not _is_sha(selected.get("selection_id"))
    ):
        raise ValueError("MCL predictive predecessor or successor selection is invalid")
    inherited_ids = list(inherited_treatment_ids)
    if any(not _is_sha(item) for item in inherited_ids) or len(
        set(inherited_ids)
    ) != len(inherited_ids):
        raise ValueError("MCL predictive inherited treatment identity is invalid")
    previous_artifacts = predecessor.get("artifacts")
    if not isinstance(previous_artifacts, Mapping):
        raise ValueError("MCL predictive predecessor artifacts are missing")
    artifacts = {
        name: _artifact(root, path) for name, path in _CURRENT_ARTIFACTS.items()
    }
    for name in _PRESERVED_ARTIFACTS:
        item = previous_artifacts.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"MCL predictive predecessor {name} is missing")
        observed = _artifact(root, item.get("path"))
        if observed["sha256"] != item.get("sha256"):
            raise ValueError(f"MCL predictive predecessor {name} drifted")
        artifacts[name] = observed
    at = generated_at.astimezone(timezone.utc)
    body = {
        "schema": MCL_PREDICTIVE_GENERATION_SCHEMA_V2,
        "generated_at_utc": at.isoformat(),
        "authority": MCL_PREDICTIVE_GENERATION_AUTHORITY,
        "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
        "selection_id": selected["selection_id"],
        "eligible_start_utc": _utc(selected["selected_at_utc"]).isoformat(),
        "turn_tape_generation_sha256": predecessor[
            "turn_tape_generation_sha256"
        ],
        "historical_manifest_sha256": predecessor["historical_manifest_sha256"],
        "predecessor": {
            "path": predecessor_path.as_posix(),
            "sha256": _sha256(predecessor_file),
            "schema": predecessor["schema"],
            "selection_id": predecessor["selection_id"],
        },
        "inherited_prefix": {
            "treatment_count": len(inherited_ids),
            "treatment_ids": inherited_ids,
            "treatment_ids_sha256": _identity(inherited_ids),
        },
        "artifacts": artifacts,
        "cohort_gate": dict(predecessor["cohort_gate"]),
        "invariants": {
            **dict(predecessor["invariants"]),
            "direction_authority": "V18_only",
            "stage112_shock_source_used_as_direction": False,
            "predecessor_treatments_preserved": True,
            "selection_switch_is_not_a_new_cohort": True,
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
    return {**body, "generation_id": _identity(body)}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_mcl_predictive_generation(
    repository_root: Path,
    generation: Mapping[str, object],
    *,
    current_path: Path = MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
) -> tuple[str, str]:
    root = repository_root.resolve()
    validated = validate_mcl_predictive_generation(
        generation, repository_root=root
    )
    payload = json.dumps(
        validated, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    relative = MCL_PREDICTIVE_GENERATION_DIRECTORY / (
        f"{validated['generation_id']}.json"
    )
    immutable = root / relative
    immutable.parent.mkdir(parents=True, exist_ok=True)
    if immutable.exists() and immutable.read_bytes() != payload:
        raise ValueError("immutable MCL predictive generation changed")
    if not immutable.exists():
        _atomic_write(immutable, payload)
    current = (root / current_path).resolve()
    if root not in current.parents:
        raise ValueError("MCL predictive current generation escaped repository")
    _atomic_write(current, payload)
    return relative.as_posix(), digest
