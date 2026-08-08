"""Content-addressed Core membership, separate from runtime activation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


CORE_PROFILE_SCHEMA = "live.core-strategy-profile.v1"
CORE_POOL_SCHEMA = "live.core-strategy-pool.v1"
CORE_AUTHORITY = "proven_strategy_identity_only_no_runtime_activation"
CORE_BOUNDARIES = {
    "capital_authority": "none",
    "order_authority": "none",
    "runtime_activation": "none",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: object) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("Core identity must be one lowercase sha256")
    return text


def build_core_profile(
    candidate: Mapping[str, object],
    graduation: Mapping[str, object],
) -> dict[str, object]:
    """Bind one machine crown to its final five-session profitability proof."""

    if candidate.get("machine_authority") is not True:
        raise ValueError("Core membership requires one machine-authoritative crown")
    if graduation.get("verdict") != "PROMOTE" or graduation.get("target") != "five_session_week":
        raise ValueError("Core membership requires a five-session PROMOTE receipt")
    identity = {
        key: candidate.get(key)
        for key in (
            "candidate_id",
            "symbol",
            "track",
            "version",
            "declaration_sha256",
            "artifact_sha256",
            "strategy_id",
            "run_id",
        )
    }
    for field in ("candidate_id", "declaration_sha256", "artifact_sha256", "run_id"):
        identity[field] = _digest(identity[field])
    if any(not str(identity.get(field) or "").strip() for field in ("symbol", "track", "version", "strategy_id")):
        raise ValueError("Core candidate identity is incomplete")
    body = {
        "schema": CORE_PROFILE_SCHEMA,
        "authority": CORE_AUTHORITY,
        "candidate": identity,
        "graduation": {
            "receipt_id": _digest(graduation.get("receipt_id")),
            "target": "five_session_week",
            "cutoff_utc": graduation.get("cutoff_utc"),
        },
        "boundaries": dict(CORE_BOUNDARIES),
    }
    return {**body, "profile_id": _identity(body)}


def validate_core_profile(value: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(value)
    profile_id = str(frozen.pop("profile_id", ""))
    candidate = frozen.get("candidate")
    graduation = frozen.get("graduation")
    if (
        frozen.get("schema") != CORE_PROFILE_SCHEMA
        or frozen.get("authority") != CORE_AUTHORITY
        or frozen.get("boundaries") != CORE_BOUNDARIES
        or not isinstance(candidate, Mapping)
        or not isinstance(graduation, Mapping)
        or graduation.get("target") != "five_session_week"
        or profile_id != _identity(frozen)
    ):
        raise ValueError("invalid Core strategy profile")
    for field in ("candidate_id", "declaration_sha256", "artifact_sha256", "run_id"):
        _digest(candidate.get(field))
    _digest(graduation.get("receipt_id"))
    return dict(value)


def build_core_pool(profiles: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    members = sorted(
        (validate_core_profile(profile) for profile in profiles),
        key=lambda profile: str(profile["profile_id"]),
    )
    candidate_ids = [str(profile["candidate"]["candidate_id"]) for profile in members]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Core pool candidate identities are duplicated")
    body = {
        "schema": CORE_POOL_SCHEMA,
        "authority": CORE_AUTHORITY,
        "members": members,
        "boundaries": dict(CORE_BOUNDARIES),
    }
    return {**body, "pool_id": _identity(body)}


def validate_core_pool(value: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(value)
    pool_id = str(frozen.pop("pool_id", ""))
    members = frozen.get("members")
    if (
        frozen.get("schema") != CORE_POOL_SCHEMA
        or frozen.get("authority") != CORE_AUTHORITY
        or frozen.get("boundaries") != CORE_BOUNDARIES
        or not isinstance(members, Sequence)
        or isinstance(members, (str, bytes))
        or any(not isinstance(member, Mapping) for member in members)
        or pool_id != _identity(frozen)
    ):
        raise ValueError("invalid Core strategy pool")
    rebuilt = build_core_pool(members)
    if rebuilt != value:
        raise ValueError("Core strategy pool is not canonical")
    return dict(value)


def load_core_pool(path: Path) -> dict[str, object]:
    if not path.exists():
        return build_core_pool()
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("Core strategy pool must be one JSON object")
    return validate_core_pool(value)


def publish_core_pool(path: Path, profiles: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Atomically publish membership only; never arm capital or a runtime."""

    pool = build_core_pool(profiles)
    payload = json.dumps(pool, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return pool
