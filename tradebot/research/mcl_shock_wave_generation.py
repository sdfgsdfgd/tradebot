"""Immutable runtime succession for observation-only MCL shock waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .mcl_live_transport import _identity, _utc
from .mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION
from .mcl_shock_wave_accumulator import (
    MCL_SHOCK_WAVE_GENERATION_PATH,
    validate_mcl_shock_wave_generation,
)


MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH = Path(
    "db/calibration/mcl_shock_wave_generation.json"
)
MCL_SHOCK_WAVE_GENERATION_DIRECTORY = Path(
    "db/calibration/shock_wave_generations"
)
MCL_SHOCK_WAVE_RUNTIME_SERVICE_PATH = Path(
    "deploy/systemd/tradebot-mcl-predictive-onset-runtime.service"
)
MCL_SHOCK_WAVE_RUNTIME_TIMER_PATH = Path(
    "deploy/systemd/tradebot-mcl-predictive-onset-runtime.timer"
)
_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _artifact(root: Path, relative: object) -> dict[str, str]:
    value = Path(str(relative or ""))
    path = (root / value).resolve()
    if value.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("MCL shock-wave successor artifact escaped or is missing")
    return {"path": value.as_posix(), "sha256": _sha256(path)}


def _bound_artifact(root: Path, relative: object, digest: object) -> dict[str, str]:
    artifact = _artifact(root, relative)
    if not _is_sha(digest) or artifact["sha256"] != digest:
        raise ValueError("MCL shock-wave successor artifact hash drifted")
    return artifact


def _predecessor_path(root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    current = root / MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH
    if current.is_file():
        generation = json.loads(current.read_text())
        generation_id = str(generation.get("generation_id") or "")
        if not _is_sha(generation_id):
            raise ValueError("current MCL shock-wave generation identity drifted")
        return MCL_SHOCK_WAVE_GENERATION_DIRECTORY / f"{generation_id}.json"
    return MCL_SHOCK_WAVE_GENERATION_PATH


def _load_immutable_predecessor(path: Path, root: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL shock-wave predecessor must be one object")
    generation = dict(value)
    body = dict(generation)
    generation_id = str(body.pop("generation_id", ""))
    artifacts = generation.get("artifacts")
    if (
        not _is_sha(generation_id)
        or _identity(body) != generation_id
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError("MCL shock-wave predecessor identity drifted")
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or not _is_sha(item.get("sha256")):
            raise ValueError(f"MCL shock-wave predecessor artifact {name} is invalid")
        relative = Path(str(item.get("path") or ""))
        artifact = (root / relative).resolve()
        if relative.is_absolute() or root not in artifact.parents:
            raise ValueError(f"MCL shock-wave predecessor artifact {name} escaped")
    return generation


def build_mcl_shock_wave_successor_generation(
    *,
    repository_root: Path,
    selected: Mapping[str, object],
    selection_path: str,
    selection_file_sha256: str,
    capital_plan_id: str,
    portfolio_generation_path: str,
    portfolio_generation_sha256: str,
    predictive: Mapping[str, object],
    predictive_path: str,
    predictive_file_sha256: str,
    inherited_episode_ids: Sequence[str],
    generated_at: datetime,
    predecessor_path: Path | None = None,
    successor_reason: str = "fresh Stage-112 maintenance-reopen runtime selection",
) -> dict[str, object]:
    root = repository_root.resolve()
    predecessor_path = _predecessor_path(root, predecessor_path)
    predecessor_file = (root / predecessor_path).resolve()
    if root not in predecessor_file.parents:
        raise ValueError("MCL shock-wave predecessor escaped repository")
    predecessor = _load_immutable_predecessor(predecessor_file, root)
    selection_id = str(selected.get("selection_id") or "")
    predictive_id = str(predictive.get("generation_id") or "")
    inherited_ids = list(inherited_episode_ids)
    successor_reason = successor_reason.strip()
    if (
        selected.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or selected.get("authority") != "selected_live_bounded_canary"
        or not _is_sha(selection_id)
        or predictive.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or predictive.get("selection_id") != selection_id
        or not _is_sha(predictive_id)
        or not _is_sha(capital_plan_id)
        or any(not _is_sha(value) for value in inherited_ids)
        or len(set(inherited_ids)) != len(inherited_ids)
        or not successor_reason
    ):
        raise ValueError("MCL shock-wave successor identity is invalid")
    artifacts = dict(predecessor["artifacts"])
    for name in (
        "predictive_successor_generation",
        "successor_predictive_service",
        "successor_predictive_timer",
    ):
        artifacts.pop(name, None)
    artifacts.update(
        {
            "predecessor_generation": _artifact(root, predecessor_path),
            "selected_live_selection": _bound_artifact(
                root, selection_path, selection_file_sha256
            ),
            "portfolio_package_generation": _bound_artifact(
                root, portfolio_generation_path, portfolio_generation_sha256
            ),
            "predictive_successor_generation": _bound_artifact(
                root, predictive_path, predictive_file_sha256
            ),
            "successor_generation_owner": _artifact(
                root, "tradebot/research/mcl_shock_wave_generation.py"
            ),
            "wave_accumulator_owner": _artifact(
                root, "tradebot/research/mcl_shock_wave_accumulator.py"
            ),
            "stage131_guard_owner": _artifact(
                root, "tradebot/research/mcl_stage131.py"
            ),
            "successor_predictive_service": _artifact(
                root, MCL_SHOCK_WAVE_RUNTIME_SERVICE_PATH
            ),
            "successor_predictive_timer": _artifact(
                root, MCL_SHOCK_WAVE_RUNTIME_TIMER_PATH
            ),
        }
    )
    at = _utc(generated_at)
    body = {
        **{
            key: value
            for key, value in predecessor.items()
            if key not in {"generation_id", "registered_at_utc", "eligible_start_utc"}
        },
        "registered_at_utc": at.isoformat(),
        "eligible_start_utc": at.isoformat(),
        "selection_id": selection_id,
        "selection_file_sha256": selection_file_sha256,
        "capital_plan_id": capital_plan_id,
        "predictive_generation_id": predictive_id,
        "predictive_generation_file_sha256": predictive_file_sha256,
        "pre_outcome_basis": {
            "predecessor_generation_id": predecessor["generation_id"],
            "predecessor_generation_sha256": _sha256(predecessor_file),
            "selection_successor_reason": successor_reason,
            "exposure": (
                "Inherited morphology identities only; no return, MFE, MAE, PnL, "
                "matched-control label, order, or capital authority opened."
            ),
        },
        "inherited_prefix": {
            "episode_count": len(inherited_ids),
            "episode_ids": inherited_ids,
            "episode_ids_sha256": _identity(inherited_ids),
        },
        "artifacts": artifacts,
        "invariants": {
            **dict(predecessor["invariants"]),
            "predecessor_generation_rewritten": False,
            "predecessor_complete_episodes_preserved": True,
            "selection_switch_is_not_a_new_treatment": True,
            "selection_boundary_terminates_incomplete_wave": True,
            "runtime_generation_pointer_is_only_mutable_binding": True,
        },
    }
    return validate_mcl_shock_wave_successor_generation(
        {**body, "generation_id": _identity(body)}, repository_root=root
    )


def validate_mcl_shock_wave_successor_generation(
    value: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    generation = validate_mcl_shock_wave_generation(
        value, repository_root=repository_root
    )
    inherited = generation.get("inherited_prefix")
    artifacts = generation.get("artifacts")
    episode_ids = inherited.get("episode_ids") if isinstance(inherited, Mapping) else None
    required = {
        "predecessor_generation",
        "selected_live_selection",
        "portfolio_package_generation",
        "predictive_successor_generation",
        "successor_generation_owner",
        "successor_predictive_service",
        "successor_predictive_timer",
        "stage131_guard_owner",
    }
    if (
        not isinstance(episode_ids, Sequence)
        or isinstance(episode_ids, (str, bytes))
        or any(not _is_sha(value) for value in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
        or inherited.get("episode_count") != len(episode_ids)
        or inherited.get("episode_ids_sha256") != _identity(episode_ids)
        or not isinstance(artifacts, Mapping)
        or not required.issubset(artifacts)
        or generation.get("registered_at_utc") != generation.get("eligible_start_utc")
    ):
        raise ValueError("MCL shock-wave successor generation is invalid")
    return generation


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


def publish_mcl_shock_wave_generation(
    repository_root: Path,
    generation: Mapping[str, object],
    *,
    current_path: Path = MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH,
) -> tuple[str, str]:
    root = repository_root.resolve()
    validated = validate_mcl_shock_wave_successor_generation(
        generation, repository_root=root
    )
    payload = json.dumps(
        validated, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    relative = MCL_SHOCK_WAVE_GENERATION_DIRECTORY / (
        f"{validated['generation_id']}.json"
    )
    immutable = root / relative
    if immutable.exists() and immutable.read_bytes() != payload:
        raise ValueError("immutable MCL shock-wave generation changed")
    if not immutable.exists():
        _atomic_write(immutable, payload)
    current = (root / current_path).resolve()
    if root not in current.parents:
        raise ValueError("MCL shock-wave current generation escaped repository")
    _atomic_write(current, payload)
    return relative.as_posix(), digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", type=Path, default=MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH
    )
    args = parser.parse_args(argv)
    generation = json.loads(args.generation.read_text())
    validated = validate_mcl_shock_wave_successor_generation(
        generation, repository_root=_ROOT
    )
    print(
        json.dumps(
            {
                "generation_id": validated["generation_id"],
                "selection_id": validated["selection_id"],
                "inherited_episodes": validated["inherited_prefix"]["episode_count"],
                "submitted_orders": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
