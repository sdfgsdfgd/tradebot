"""Join fresh crowned XSP targets to the outcome-blind seconds pressure atlas."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .live_calibration import LiveCalibrationLedger
from .xsp_context import xsp_execution_signal_context
from .xsp_live_transport import xsp_signal_utc
from .xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
    XSP_OPENING_EDGE_V3_VERSION,
)
from .xsp_pressure_atlas import (
    XSP_PRESSURE_ATLAS_AUTHORITY,
    project_xsp_pressure_atlas,
)
from .xsp_pressure_tape import XSP_PRESSURE_TAPE_STATE_DIR


XSP_PRESSURE_ACCUMULATOR_VERSION = "xsp.pressure-atlas-accumulator.v1"
XSP_PRESSURE_TREATMENT_SCHEMA = "xsp.pressure-crowned-target-treatment.v1"
XSP_PRESSURE_ACCUMULATOR_AUTHORITY = XSP_PRESSURE_ATLAS_AUTHORITY
XSP_PRESSURE_ACCUMULATOR_GENERATION_SCHEMA = (
    "xsp.pressure-atlas-accumulation-generation.v1"
)
XSP_PRESSURE_ACCUMULATOR_GENERATION_PATH = Path(
    "backtests/xsp/opening_edge_v3_pressure_atlas_accumulation_generation.json"
)
XSP_PRESSURE_ACCUMULATOR_LEDGER_PATH = (
    Path.home() / ".local/state/tradebot/research/xsp_pressure_atlas.jsonl"
)
_ROOT = Path(__file__).resolve().parents[2]
_FAMILY_NAMES = (
    "seconds_energy",
    "transport_handoff",
    "volatility_liquidity",
    "cross_scale",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("XSP pressure-accumulator timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _is_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def load_xsp_pressure_accumulator_generation(
    path: Path = XSP_PRESSURE_ACCUMULATOR_GENERATION_PATH,
    *,
    root: Path | None = None,
    owner_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Rehash the prospective join and every immutable predecessor it names."""

    source = Path(path)
    generation = json.loads(source.read_text())
    if not isinstance(generation, dict):
        raise ValueError("XSP pressure-accumulator generation must be an object")
    body = dict(generation)
    generation_id = str(body.pop("generation_id", ""))
    gate = generation.get("cohort_gate")
    if (
        generation.get("schema") != XSP_PRESSURE_ACCUMULATOR_GENERATION_SCHEMA
        or generation.get("authority") != XSP_PRESSURE_ATLAS_AUTHORITY
        or generation.get("incumbent_strategy_version")
        != XSP_OPENING_EDGE_V3_VERSION
        or generation.get("source_strategy_version")
        != XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        or not _is_sha(generation_id)
        or _identity(body) != generation_id
        or not _is_sha(generation.get("selection_id"))
        or not isinstance(gate, Mapping)
        or gate
        != {
            "minimum_complete_crowned_targets": 30,
            "minimum_each_crowned_direction": 10,
            "minimum_repeated_morphologies_each_candidate_family": 5,
        }
        or generation.get("outcomes_open") is not False
        or generation.get("classifier_open") is not False
        or generation.get("permission_open") is not False
        or generation.get("order_authority") != "none"
        or generation.get("submitted_orders") != 0
    ):
        raise ValueError("XSP pressure-accumulator generation drifted")
    if _utc(generation["eligible_start_utc"]) < _utc(
        generation["registered_at_utc"]
    ):
        raise ValueError("XSP pressure-accumulator eligibility predates registration")

    repository = root or _ROOT
    artifacts = generation.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("XSP pressure-accumulator artifacts are missing")
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or not _is_sha(item.get("sha256")):
            raise ValueError(f"XSP pressure-accumulator artifact {name} is invalid")
        artifact = repository / str(item.get("path") or "")
        if not artifact.is_file() or _sha256(artifact) != item["sha256"]:
            raise ValueError(f"XSP pressure-accumulator artifact {name} drifted")
    owner = owner_path or Path(__file__).resolve()
    if _sha256(owner) != generation.get("owner_sha256"):
        raise ValueError("XSP pressure-accumulator owner drifted")
    return generation, _sha256(source)


def _target_candidates(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    eligible_start: datetime,
) -> list[dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}
    keys: dict[str, str] = {}
    ordered = sorted(records, key=lambda row: str(row.get("recorded_at_utc") or ""))
    for record in ordered:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version")
            != XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
            or record.get("session") != "RTH"
            or record.get("status") != "EVALUATED"
            or not isinstance(evidence, Mapping)
            or evidence.get("rth_provenance_fresh") is not True
            or evidence.get("order_authority") != "none"
        ):
            continue
        paired = evidence.get("paired_equity")
        if not isinstance(paired, Mapping):
            raise ValueError("XSP pressure source has no paired crown state")
        signal = xsp_execution_signal_context(paired)
        if signal is None or signal.get("lane") != "rth":
            continue
        signal_at = xsp_signal_utc(signal["signal_bar_ts"])
        entry_at = xsp_signal_utc(signal["entry_time_utc"])
        if signal_at < eligible_start:
            continue
        daily = paired.get("daily_context_state")
        daily_state = daily.get("state") if isinstance(daily, Mapping) else None
        if not isinstance(daily_state, Mapping):
            raise ValueError("XSP pressure source daily context is missing")
        target_key_body = {
            "selection_id": selection_id,
            "direction": signal["direction"],
            "signal_bar_at_utc": signal_at.isoformat(),
            "entry_at_utc": entry_at.isoformat(),
        }
        target_key = _identity(target_key_body)
        target_id = _identity(
            {
                **target_key_body,
                "decision_trace_fingerprint": signal[
                    "decision_trace_fingerprint"
                ],
            }
        )
        prior = keys.get(target_key)
        if prior is not None and prior != target_id:
            raise ValueError("XSP pressure source target identity conflicts")
        keys[target_key] = target_id
        if target_id not in candidates:
            candidates[target_id] = {
                "target_id": target_id,
                "target_key": target_key,
                "selection_id": selection_id,
                "direction": signal["direction"],
                "signal_bar_at_utc": signal_at.isoformat(),
                "entry_at_utc": entry_at.isoformat(),
                "decision_trace_fingerprint": signal[
                    "decision_trace_fingerprint"
                ],
                "source_checkpoint_id": record.get("checkpoint_id"),
                "source_recorded_at_utc": record.get("recorded_at_utc"),
                "xsp_impulse": signal["directional_impulse"],
                "daily_context": dict(daily_state),
            }
    return sorted(candidates.values(), key=lambda row: row["signal_bar_at_utc"])


def _tape_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    output = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid pressure JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: pressure row is not an object")
        output.append(row)
    return output


def _morphology_signatures(atlas: Mapping[str, object]) -> dict[str, str]:
    seconds = atlas["seconds"]
    horizons = seconds["horizons"]
    assert isinstance(horizons, Mapping)
    rows = [horizons[str(value)] for value in (5, 15, 30, 45)]
    assert all(isinstance(row, Mapping) for row in rows)
    slower = atlas["slower_context"]
    assert isinstance(slower, Mapping)
    signatures = {
        "seconds_energy": [
            (
                row["path_consensus"]["target_alignment"],
                row["velocity_consensus"]["target_alignment"],
            )
            for row in rows
        ],
        "transport_handoff": [
            (
                row["morphology"]["ignition"],
                row["morphology"]["transport"],
                row["basis"]["state"],
                row["lead"]["dominant_leader"],
            )
            for row in rows
        ],
        "volatility_liquidity": [
            (
                row["morphology"]["volatility_flow"],
                row["morphology"]["liquidity"],
            )
            for row in rows
        ],
        "cross_scale": {
            "seconds": slower["seconds_state"],
            "xsp": slower["xsp_impulse"]["path_state"],
            "xsp_velocity": slower["xsp_impulse"]["velocity_state"],
            "spy": slower["spy_impulse"]["path_state"],
            "daily": slower["daily_5_10_21_42_63_84"]["path_state"],
            "joint": slower["cross_scale_state"],
        },
    }
    return {name: _identity(value) for name, value in signatures.items()}


def _treatment(
    candidate: Mapping[str, object],
    atlas: Mapping[str, object],
    *,
    generation_sha256: str,
) -> dict[str, object]:
    body = {
        "schema": XSP_PRESSURE_TREATMENT_SCHEMA,
        "authority": XSP_PRESSURE_ATLAS_AUTHORITY,
        "generation_sha256": generation_sha256,
        "incumbent_strategy_version": XSP_OPENING_EDGE_V3_VERSION,
        "target": {
            key: candidate[key]
            for key in (
                "target_id",
                "target_key",
                "selection_id",
                "direction",
                "signal_bar_at_utc",
                "entry_at_utc",
                "decision_trace_fingerprint",
                "source_checkpoint_id",
                "source_recorded_at_utc",
            )
        },
        "atlas": dict(atlas),
        "morphology_signatures": _morphology_signatures(atlas),
        "slow_spy_status": "UNDERWARMED_NO_SAME_CLOCK_RTH_SPY_IMPULSE",
        "classifier": "none",
        "permission": "none",
        "outcomes": None,
        "order_authority": "none",
        "capital_authority": "none",
        "submitted_orders": 0,
    }
    return {**body, "treatment_id": _identity(body)}


def _validated_treatment(value: Mapping[str, object]) -> dict[str, object]:
    treatment = dict(value)
    body = dict(treatment)
    treatment_id = str(body.pop("treatment_id", ""))
    target = treatment.get("target")
    signatures = treatment.get("morphology_signatures")
    if (
        treatment.get("schema") != XSP_PRESSURE_TREATMENT_SCHEMA
        or treatment.get("authority") != XSP_PRESSURE_ATLAS_AUTHORITY
        or not _is_sha(treatment_id)
        or _identity(body) != treatment_id
        or not isinstance(target, Mapping)
        or not _is_sha(target.get("target_id"))
        or not _is_sha(target.get("target_key"))
        or target.get("direction") not in {"up", "down"}
        or not isinstance(signatures, Mapping)
        or set(signatures) != set(_FAMILY_NAMES)
        or not all(_is_sha(item) for item in signatures.values())
        or treatment.get("classifier") != "none"
        or treatment.get("permission") != "none"
        or treatment.get("outcomes") is not None
        or treatment.get("order_authority") != "none"
        or treatment.get("capital_authority") != "none"
        or treatment.get("submitted_orders") != 0
    ):
        raise ValueError("XSP pressure treatment identity drifted")
    return treatment


def xsp_pressure_treatments(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    targets: dict[str, str] = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != XSP_PRESSURE_ACCUMULATOR_VERSION
            or not isinstance(evidence, Mapping)
        ):
            continue
        treatment = _validated_treatment(evidence)
        target = treatment["target"]
        target_key = str(target["target_key"])
        treatment_id = str(treatment["treatment_id"])
        prior = targets.get(target_key)
        if prior is not None and prior != treatment_id:
            raise ValueError("XSP pressure ledger target conflicts")
        if prior is not None:
            raise ValueError("XSP pressure ledger repeats one target")
        targets[target_key] = treatment_id
        output.append(treatment)
    return sorted(output, key=lambda row: row["target"]["signal_bar_at_utc"])


def xsp_pressure_cohort(
    treatments: Sequence[Mapping[str, object]],
    *,
    gate: Mapping[str, object],
) -> dict[str, object]:
    directions = Counter(str(row["target"]["direction"]) for row in treatments)
    families = {}
    for name in _FAMILY_NAMES:
        counts = Counter(str(row["morphology_signatures"][name]) for row in treatments)
        families[name] = {
            "unique_shapes": len(counts),
            "maximum_repetition": max(counts.values(), default=0),
            "repeated_shapes": sum(
                count
                >= int(gate["minimum_repeated_morphologies_each_candidate_family"])
                for count in counts.values()
            ),
        }
    gates = {
        "complete_crowned_targets": len(treatments)
        >= int(gate["minimum_complete_crowned_targets"]),
        "each_crowned_direction": all(
            directions[direction]
            >= int(gate["minimum_each_crowned_direction"])
            for direction in ("up", "down")
        ),
        "repeated_morphologies_each_family": all(
            row["repeated_shapes"] > 0 for row in families.values()
        ),
    }
    passed = all(gates.values())
    return {
        "complete_targets": len(treatments),
        "directions": dict(sorted(directions.items())),
        "families": families,
        "gates": gates,
        "verdict": (
            "MORPHOLOGY_COHORT_READY_OUTCOMES_SEALED"
            if passed
            else "FROZEN_ACCUMULATE"
        ),
    }


def accumulate_xsp_pressure_atlas(
    source_records: Sequence[Mapping[str, object]],
    *,
    observed_at: datetime,
    generation_path: Path = XSP_PRESSURE_ACCUMULATOR_GENERATION_PATH,
    tape_dir: Path = XSP_PRESSURE_TAPE_STATE_DIR,
    ledger_path: Path = XSP_PRESSURE_ACCUMULATOR_LEDGER_PATH,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Append every complete unseen crowned target once; never project permission."""

    generation, generation_sha = load_xsp_pressure_accumulator_generation(
        generation_path,
        root=repository_root,
    )
    eligible_start = _utc(generation["eligible_start_utc"])
    candidates = _target_candidates(
        source_records,
        selection_id=str(generation["selection_id"]),
        eligible_start=eligible_start,
    )
    ledger = LiveCalibrationLedger(ledger_path)
    existing = {
        str(row["target"]["target_key"]): row
        for row in xsp_pressure_treatments(tuple(ledger.records()))
    }
    appended = 0
    incomplete = []
    for candidate in candidates:
        signal_at = _utc(candidate["signal_bar_at_utc"])
        rows = _tape_records(tape_dir / f"{signal_at.date().isoformat()}.jsonl")
        try:
            atlas = project_xsp_pressure_atlas(
                rows,
                as_of_utc=signal_at,
                target_direction=str(candidate["direction"]),
                xsp_impulse=candidate["xsp_impulse"],
                spy_impulse=None,
                daily_context=candidate["daily_context"],
            )
            if (
                atlas["source"]["pressure_tape_generation_sha256"]
                != generation["pressure_tape_generation_sha256"]
            ):
                raise ValueError("XSP pressure treatment crossed tape generations")
            treatment = _treatment(
                candidate,
                atlas,
                generation_sha256=generation_sha,
            )
        except ValueError as exc:
            if str(candidate["target_key"]) in existing:
                raise
            incomplete.append(
                {
                    "target_id": candidate["target_id"],
                    "signal_bar_at_utc": candidate["signal_bar_at_utc"],
                    "reason": str(exc),
                }
            )
            continue
        prior = existing.get(str(candidate["target_key"]))
        if prior is not None:
            if prior["treatment_id"] != treatment["treatment_id"]:
                raise ValueError("XSP pressure treatment conflicts with immutable target")
            continue
        ledger.checkpoint(
            evaluation_as_of=signal_at,
            strategy_id=XSP_OPENING_EDGE_V3_VERSION,
            strategy_version=XSP_PRESSURE_ACCUMULATOR_VERSION,
            trading_date=str(atlas["source"]["trading_date"]),
            session="XSP_PRESSURE_ATLAS",
            status="EVALUATED",
            evidence=treatment,
            recorded_at=max(_utc(observed_at), signal_at),
        )
        existing[str(candidate["target_key"])] = treatment
        appended += 1
    treatments = xsp_pressure_treatments(tuple(ledger.records()))
    return {
        "schema": "xsp.pressure-atlas-accumulation-status.v1",
        "authority": XSP_PRESSURE_ATLAS_AUTHORITY,
        "generation_sha256": generation_sha,
        "source_candidates": len(candidates),
        "appended": appended,
        "incomplete": incomplete,
        "cohort": xsp_pressure_cohort(treatments, gate=generation["cohort_gate"]),
        "classifier": "none",
        "permission": "none",
        "outcomes": None,
        "order_authority": "none",
        "capital_authority": "none",
        "submitted_orders": 0,
    }
