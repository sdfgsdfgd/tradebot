"""Exact Stage-114 coverage guard for one opposite V18 maturation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from ..chart_data.series import OhlcvBar
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .mcl_live_transport import (
    _identity,
    _is_sha,
    _utc,
    load_mcl_live_selection_from_mapping,
)
from .mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION


MCL_STAGE131_COVERAGE_SCHEMA = "mcl.stage114-observer-coverage.v1"
MCL_STAGE131_DECISION_SCHEMA = "mcl.stage131-observer-coverage-decision.v1"
MCL_STAGE131_COVERAGE_PATH = (
    Path.home() / ".local/state/tradebot/research/mcl_shock_wave_coverage.json"
)
MCL_STAGE131_BINDING_PATH = Path(
    "backtests/mcl/mcl_stage114_v18_epoch_veto_stage131_runtime_binding.json"
)
MCL_STAGE131_BINDING_KEY = "stage131_epoch_veto"
MCL_STAGE131_BLOCKING_ACTIONS = {
    "DEFER_COVERAGE_MISSING",
    "DEFER_COVERAGE_LAG",
    "DEFER_COVERAGE_IDENTITY_MISMATCH",
    "VETO_OPPOSITE_EXACT_EPOCH",
}
MCL_STAGE131_HOLD_REASONS = {
    "DEFER_COVERAGE_MISSING": "stage131_coverage_missing",
    "DEFER_COVERAGE_LAG": "stage131_coverage_lag",
    "DEFER_COVERAGE_IDENTITY_MISMATCH": "stage131_coverage_identity_mismatch",
    "VETO_OPPOSITE_EXACT_EPOCH": "stage131_veto_opposite_exact_epoch",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def validate_mcl_stage131_coverage(
    value: Mapping[str, object],
) -> dict[str, object]:
    body = dict(value)
    state_id = str(body.pop("state_id", ""))
    episode_ids = body.get("complete_episode_ids")
    open_id = body.get("open_episode_state_id")
    if (
        body.get("schema") != MCL_STAGE131_COVERAGE_SCHEMA
        or body.get("authority")
        != "coverage_only_no_direction_no_orders_no_capital"
        or body.get("submitted_orders") != 0
        or not _is_sha(body.get("generation_id"))
        or not _is_sha(body.get("selection_id"))
        or len(str(body.get("contract_key") or "")) != 6
        or not isinstance(episode_ids, list)
        or any(not _is_sha(item) for item in episode_ids)
        or sorted(set(episode_ids)) != episode_ids
        or body.get("complete_episode_set_sha256")
        != calibration_fingerprint(episode_ids)
        or (open_id is not None and not _is_sha(open_id))
        or not _is_sha(state_id)
        or calibration_fingerprint(body) != state_id
    ):
        raise ValueError("MCL Stage-131 coverage identity drifted")
    for key in (
        "evaluated_completed_bar_through_utc",
        "evaluated_turn_tape_through_utc",
        "recorded_at_utc",
    ):
        _utc(body[key])
    return dict(value)


def build_mcl_stage131_coverage(
    *,
    generation: Mapping[str, object],
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    complete_episodes: Sequence[Mapping[str, object]],
    open_episode: Mapping[str, object] | None,
    recorded_at: datetime,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    generation_id = str(generation.get("generation_id") or "")
    selection_id = str(selected["selection_id"])
    contract_key = str(selected["contracts"]["MCL"]["expiry"])[:6]
    if (
        generation.get("selection_id") != selection_id
        or generation.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or not _is_sha(generation_id)
        or not rows
        or any(not bars.get(symbol) for symbol in ("CL", "MCL"))
    ):
        raise ValueError("MCL Stage-131 coverage source is incomplete")
    episode_ids = []
    for episode in complete_episodes:
        identity = episode.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("MCL Stage-131 episode has no identity")
        if (
            identity.get("generation_id") != generation_id
            or identity.get("selection_id") != selection_id
            or identity.get("contract_key") != contract_key
            or not _is_sha(episode.get("episode_id"))
        ):
            raise ValueError("MCL Stage-131 episode escaped its generation")
        episode_ids.append(str(episode["episode_id"]))
    body = {
        "schema": MCL_STAGE131_COVERAGE_SCHEMA,
        "generation_id": generation_id,
        "selection_id": selection_id,
        "contract_key": contract_key,
        "evaluated_completed_bar_through_utc": min(
            max(bars[symbol]) for symbol in ("CL", "MCL")
        ).isoformat(),
        "evaluated_turn_tape_through_utc": max(
            _utc(row["_time"]) for row in rows
        ).isoformat(),
        "complete_episode_ids": sorted(set(episode_ids)),
        "complete_episode_set_sha256": calibration_fingerprint(
            sorted(set(episode_ids))
        ),
        "open_episode_state_id": (
            calibration_fingerprint(dict(open_episode))
            if open_episode is not None
            else None
        ),
        "recorded_at_utc": _utc(recorded_at).isoformat(),
        "authority": "coverage_only_no_direction_no_orders_no_capital",
        "submitted_orders": 0,
    }
    return validate_mcl_stage131_coverage(
        {**body, "state_id": calibration_fingerprint(body)}
    )


def publish_mcl_stage131_coverage(
    path: Path, coverage: Mapping[str, object]
) -> dict[str, object]:
    validated = validate_mcl_stage131_coverage(coverage)
    payload = json.dumps(
        validated, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    _atomic_write(path.expanduser(), payload)
    return validated


def load_mcl_stage131_coverage(path: Path) -> dict[str, object] | None:
    source = path.expanduser()
    if not source.is_file():
        return None
    value = json.loads(source.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL Stage-131 coverage must be one object")
    return validate_mcl_stage131_coverage(value)


def validate_mcl_stage131_binding(repository_root: Path) -> dict[str, object]:
    root = repository_root.resolve()
    path = root / MCL_STAGE131_BINDING_PATH
    value = json.loads(path.read_text())
    owners = value.get("owners") if isinstance(value, Mapping) else None
    evidence = value.get("evidence") if isinstance(value, Mapping) else None
    expected = {
        "guard": "tradebot/research/mcl_stage131.py",
        "live": "tradebot/research/mcl_live.py",
        "cli": "tradebot/research/mcl_live_cli.py",
        "wave_accumulator": "tradebot/research/mcl_shock_wave_accumulator.py",
        "wave_generation": "tradebot/research/mcl_shock_wave_generation.py",
        "profitability": "tradebot/research/mcl_profitability.py",
    }
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "mcl.stage131-runtime-binding.v1"
        or value.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or value.get("verdict") != "PASS"
        or value.get("submitted_orders") != 0
        or not isinstance(owners, Mapping)
        or not isinstance(evidence, Mapping)
        or not all(value.get("gates", {}).values())
    ):
        raise ValueError("MCL Stage-131 runtime binding is invalid")
    for name, relative in expected.items():
        owner = owners.get(name)
        if (
            not isinstance(owner, Mapping)
            or owner.get("path") != relative
            or owner.get("sha256") != _sha256(root / relative)
        ):
            raise ValueError(f"MCL Stage-131 {name} owner drifted")
    for name, item in evidence.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"MCL Stage-131 {name} evidence is invalid")
        relative = Path(str(item.get("path") or ""))
        artifact = (root / relative).resolve()
        if (
            relative.is_absolute()
            or root not in artifact.parents
            or not artifact.is_file()
            or item.get("sha256") != _sha256(artifact)
        ):
            raise ValueError(f"MCL Stage-131 {name} evidence drifted")
    return dict(value)


def bind_mcl_stage131_selection(
    selection: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    root = repository_root.resolve()
    validate_mcl_stage131_binding(root)
    body = dict(selected)
    body.pop("selection_id")
    body["evidence"] = {
        **dict(selected["evidence"]),
        MCL_STAGE131_BINDING_KEY: {
            "path": MCL_STAGE131_BINDING_PATH.as_posix(),
            "sha256": _sha256(root / MCL_STAGE131_BINDING_PATH),
        },
    }
    return load_mcl_live_selection_from_mapping(
        {**body, "selection_id": _identity(body)}
    )


def load_mcl_stage131_context(
    selection: Mapping[str, object],
    *,
    repository_root: Path,
    generation_path: Path,
    coverage_path: Path = MCL_STAGE131_COVERAGE_PATH,
    wave_ledger_path: Path,
) -> dict[str, object] | None:
    selected = load_mcl_live_selection_from_mapping(selection)
    binding = selected["evidence"].get(MCL_STAGE131_BINDING_KEY)
    if binding is None:
        return None
    root = repository_root.resolve()
    if (
        not isinstance(binding, Mapping)
        or binding.get("path") != MCL_STAGE131_BINDING_PATH.as_posix()
        or binding.get("sha256") != _sha256(root / MCL_STAGE131_BINDING_PATH)
    ):
        raise ValueError("MCL Stage-131 selection binding drifted")
    validate_mcl_stage131_binding(root)
    from .mcl_shock_wave_accumulator import (  # avoid an accumulator import cycle
        load_mcl_shock_wave_generation,
        mcl_shock_wave_episodes,
    )

    generation = load_mcl_shock_wave_generation(
        generation_path, repository_root=root
    )
    if generation.get("selection_id") != selected["selection_id"]:
        raise ValueError("MCL Stage-131 generation selection drifted")
    return {
        "generation_id": generation["generation_id"],
        "selection_id": selected["selection_id"],
        "coverage": load_mcl_stage131_coverage(coverage_path),
        "episodes": mcl_shock_wave_episodes(
            tuple(LiveCalibrationLedger(wave_ledger_path.expanduser()).records())
        ),
    }


def mcl_stage131_veto_identity(
    episode: Mapping[str, object],
    source: Mapping[str, object],
    *,
    selection_id: str,
) -> dict[str, object] | None:
    identity = episode.get("identity")
    terminal = episode.get("terminal")
    raw = source.get("last_raw_turn")
    target = source.get("target")
    raw_decision = raw.get("decision") if isinstance(raw, Mapping) else None
    target_decision = target.get("decision") if isinstance(target, Mapping) else None
    waves = episode.get("authority_waves")
    direction = episode.get("terminal_authority_direction")
    raw_at = raw.get("observed_at_utc") if isinstance(raw, Mapping) else None
    target_direction = target.get("direction") if isinstance(target, Mapping) else None
    if (
        not isinstance(identity, Mapping)
        or identity.get("selection_id") != selection_id
        or identity.get("contract_key") != source.get("contract_month")
        or not isinstance(terminal, Mapping)
        or terminal.get("reasons") != ["stage112_v18_raw_turn"]
        or episode.get("terminal_at_utc") != raw_at
        or direction not in (-1, 1)
        or not isinstance(waves, Sequence)
        or isinstance(waves, (str, bytes))
        or not waves
        or not isinstance(waves[-1], Mapping)
        or waves[-1].get("direction") != direction
        or not isinstance(raw, Mapping)
        or raw.get("owner") != "v18"
        or raw.get("signal_at_utc") != raw_at
        or not isinstance(raw_decision, Mapping)
        or raw_decision.get("phase") != "RAW_TURN"
        or raw_decision.get("raw_direction") != direction
        or not isinstance(target, Mapping)
        or target.get("owner") != "v18"
        or target.get("route") != "failed_auction"
        or target.get("signal_at_utc") != raw_at
        or target_direction != -int(direction)
        or not isinstance(target_decision, Mapping)
        or target_decision.get("phase") != "MATURATION"
        or target_decision.get("raw_direction") != direction
        or target_decision.get("admitted_direction") != target_direction
    ):
        return None
    body = {
        "selection_id": selection_id,
        "contract_key": identity["contract_key"],
        "episode_id": episode["episode_id"],
        "terminal_at_utc": episode["terminal_at_utc"],
        "terminal_direction": direction,
        "raw_event_id": raw["event_id"],
        "maturation_event_id": target["event_id"],
    }
    return {**body, "veto_id": calibration_fingerprint(body)}


def project_mcl_stage131_entry_guard(
    source_checkpoint: Mapping[str, object],
    *,
    context: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if context is None:
        return None
    evidence = source_checkpoint.get("evidence")
    target = evidence.get("target") if isinstance(evidence, Mapping) else None
    raw = evidence.get("last_raw_turn") if isinstance(evidence, Mapping) else None
    source = evidence.get("source") if isinstance(evidence, Mapping) else None
    selection_id = str(context.get("selection_id") or "")
    generation_id = str(context.get("generation_id") or "")
    eligible = (
        isinstance(target, Mapping)
        and target.get("owner") == "v18"
        and target.get("route") == "failed_auction"
        and isinstance(raw, Mapping)
        and isinstance(source, Mapping)
    )
    action = "ALLOW_UNSCOPED"
    coverage_id = veto_id = None
    if eligible:
        coverage = context.get("coverage")
        if coverage is None:
            action = "DEFER_COVERAGE_MISSING"
        elif not isinstance(coverage, Mapping):
            raise ValueError("MCL Stage-131 context coverage is invalid")
        else:
            checked = validate_mcl_stage131_coverage(coverage)
            coverage_id = checked["state_id"]
            if (
                checked.get("generation_id") != generation_id
                or checked.get("selection_id") != selection_id
                or checked.get("contract_key") != source.get("contract_month")
            ):
                action = "DEFER_COVERAGE_IDENTITY_MISMATCH"
            elif min(
                _utc(checked["evaluated_completed_bar_through_utc"]),
                _utc(checked["evaluated_turn_tape_through_utc"]),
            ) < _utc(raw["observed_at_utc"]):
                action = "DEFER_COVERAGE_LAG"
            else:
                covered_ids = set(checked["complete_episode_ids"])
                episodes = [
                    row
                    for row in context.get("episodes", [])
                    if isinstance(row, Mapping) and row.get("episode_id") in covered_ids
                ]
                if {row.get("episode_id") for row in episodes} != covered_ids:
                    raise ValueError(
                        "MCL Stage-131 coverage references an unavailable episode"
                    )
                matches = [
                    match
                    for episode in episodes
                    if (
                        match := mcl_stage131_veto_identity(
                            episode, source, selection_id=selection_id
                        )
                    )
                    is not None
                ]
                if len(matches) > 1:
                    raise ValueError(
                        "multiple Stage-114 waves cover one V18 maturation"
                    )
                if matches:
                    action = "VETO_OPPOSITE_EXACT_EPOCH"
                    veto_id = matches[0]["veto_id"]
                else:
                    action = "ALLOW_COVERAGE_PROVEN"
    body = {
        "schema": MCL_STAGE131_DECISION_SCHEMA,
        "selection_id": selection_id,
        "source_checkpoint_id": source_checkpoint.get("checkpoint_id"),
        "admission_event_id": (
            target.get("event_id") if isinstance(target, Mapping) else None
        ),
        "coverage_state_id": coverage_id,
        "action": action,
        "veto_id": veto_id,
        "direction_authority": "none",
        "entry_authority": "none",
        "submitted_orders": 0,
    }
    return {**body, "decision_id": calibration_fingerprint(body)}


def validate_mcl_stage131_entry_guard(
    value: Mapping[str, object],
    *,
    selection_id: str,
    source_checkpoint_id: str,
    admission_event_id: str | None,
) -> dict[str, object]:
    body = dict(value)
    decision_id = str(body.pop("decision_id", ""))
    if (
        body.get("schema") != MCL_STAGE131_DECISION_SCHEMA
        or body.get("selection_id") != selection_id
        or body.get("source_checkpoint_id") != source_checkpoint_id
        or body.get("admission_event_id") != admission_event_id
        or body.get("action")
        not in {
            "ALLOW_UNSCOPED",
            "ALLOW_COVERAGE_PROVEN",
            *MCL_STAGE131_BLOCKING_ACTIONS,
        }
        or body.get("direction_authority") != "none"
        or body.get("entry_authority") != "none"
        or body.get("submitted_orders") != 0
        or not _is_sha(decision_id)
        or calibration_fingerprint(body) != decision_id
    ):
        raise ValueError("MCL Stage-131 entry guard identity drifted")
    return dict(value)
