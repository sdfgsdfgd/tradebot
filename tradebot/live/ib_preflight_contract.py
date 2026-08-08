"""Tamper-evident IB readiness receipt and final order gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


IB_PREFLIGHT_SCHEMA = "live.ib-preflight.v1"
IB_PREFLIGHT_AUTHORITY = "read_only_broker_and_runtime_readiness"
IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC = 30 * 60 * 60
IB_PREFLIGHT_BOUNDARIES = {
    "broker_orders_submitted": 0,
    "broker_orders_cancelled": 0,
    "gateway_restarted": False,
    "runtime_units_mutated": False,
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


def _aware_utc(value: datetime | str) -> datetime:
    parsed = (
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None:
        raise ValueError("IB preflight timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def reduce_ib_preflight(
    facts: Mapping[str, object],
    *,
    checked_at_utc: datetime | str,
) -> dict[str, object]:
    """Reduce read-only facts into distinct entry and reduction readiness."""

    checked = _aware_utc(checked_at_utc)
    gateway = facts.get("gateway")
    broker = facts.get("broker")
    connectivity = facts.get("connectivity")
    runtime = facts.get("runtime")
    capabilities = facts.get("capabilities")
    if (
        not isinstance(gateway, Mapping)
        or not isinstance(broker, Mapping)
        or not isinstance(connectivity, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(capabilities, Sequence)
        or isinstance(capabilities, (str, bytes))
        or any(not isinstance(row, Mapping) for row in capabilities)
    ):
        raise ValueError("IB preflight facts are incomplete")

    shared: list[str] = []
    if gateway.get("port_accepting") is not True:
        shared.append("gateway_port_unavailable")
    if gateway.get("api_authenticated") is not True:
        shared.append("broker_api_not_authenticated")
    if gateway.get("expected_account_returned") is not True:
        shared.append("expected_account_not_returned")
    if broker.get("positions_fresh") is not True:
        shared.append("broker_positions_not_fresh")
    if broker.get("open_orders_fresh") is not True:
        shared.append("broker_open_orders_not_fresh")

    reduction_reasons = list(shared)
    if broker.get("reduction_quote_ready") is not True:
        reduction_reasons.append("held_position_quote_not_ready")

    entry_reasons = list(shared)
    if not capabilities:
        entry_reasons.append("required_capabilities_missing")
    entry_reasons.extend(
        f"required_capability_unhealthy:{row.get('label')}"
        for row in capabilities
        if row.get("healthy") is not True
    )
    if connectivity.get("unpaired_1100") is True:
        entry_reasons.append("connectivity_loss_unpaired")
    if connectivity.get("losses_10m", 0) >= 3:
        entry_reasons.append("connectivity_flap_storm")
    missing_members = runtime.get("missing_members")
    if not isinstance(missing_members, Sequence) or isinstance(
        missing_members, (str, bytes)
    ):
        entry_reasons.append("runtime_membership_unknown")
    else:
        entry_reasons.extend(
            f"runtime_member_not_armed:{unit}" for unit in missing_members
        )

    entry_reasons = sorted(set(entry_reasons))
    reduction_reasons = sorted(set(reduction_reasons))
    body = {
        "schema": IB_PREFLIGHT_SCHEMA,
        "authority": IB_PREFLIGHT_AUTHORITY,
        "checked_at_utc": checked.isoformat(),
        "facts": dict(facts),
        "verdict": {
            "entry_ready": not entry_reasons,
            "reduction_ready": not reduction_reasons,
            "entry_reasons": entry_reasons,
            "reduction_reasons": reduction_reasons,
        },
        "boundaries": dict(IB_PREFLIGHT_BOUNDARIES),
    }
    return {**body, "receipt_id": _identity(body)}


def validate_ib_preflight(value: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(value)
    receipt_id = str(frozen.pop("receipt_id", ""))
    if (
        frozen.get("schema") != IB_PREFLIGHT_SCHEMA
        or frozen.get("authority") != IB_PREFLIGHT_AUTHORITY
        or frozen.get("boundaries") != IB_PREFLIGHT_BOUNDARIES
        or not isinstance(frozen.get("facts"), Mapping)
        or not isinstance(frozen.get("verdict"), Mapping)
        or receipt_id != _identity(frozen)
    ):
        raise ValueError("invalid IB preflight receipt")
    rebuilt = reduce_ib_preflight(
        frozen["facts"],
        checked_at_utc=str(frozen["checked_at_utc"]),
    )
    if rebuilt != value:
        raise ValueError("IB preflight receipt is not canonical")
    return dict(value)


def publish_ib_preflight(path: Path, receipt: Mapping[str, object]) -> None:
    frozen = validate_ib_preflight(receipt)
    payload = (
        json.dumps(frozen, allow_nan=False, indent=2, sort_keys=True).encode()
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_ib_preflight(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_sec: float = IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("IB preflight receipt must be one JSON object")
    receipt = validate_ib_preflight(value)
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (observed - _aware_utc(str(receipt["checked_at_utc"]))).total_seconds()
    if age < -60 or age > max(1.0, float(max_age_sec)):
        raise ValueError("IB preflight receipt is stale")
    return receipt


def _configured_receipt_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return path
    raw = os.getenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", "").strip()
    return Path(raw).expanduser() if raw else None


def ib_preflight_configured(path: Path | None = None) -> bool:
    return _configured_receipt_path(path) is not None


def _scoped_reduction_reasons(
    reasons: list[str], facts: object, requested_ids: Sequence[int]
) -> list[str]:
    reasons = [
        reason for reason in reasons if reason != "held_position_quote_not_ready"
    ]
    broker = facts.get("broker") if isinstance(facts, Mapping) else None
    positions = broker.get("positions") if isinstance(broker, Mapping) else None
    readiness = (
        broker.get("reduction_quotes") if isinstance(broker, Mapping) else None
    )
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or not isinstance(readiness, Mapping)
    ):
        return [*reasons, "held_position_quote_scope_unavailable"]
    held_ids = {
        int(position.get("con_id", 0) or 0)
        for position in positions
        if isinstance(position, Mapping)
        and abs(float(position.get("quantity", 0.0) or 0.0)) > 1e-9
    }
    return [
        *reasons,
        *(
            f"held_position_quote_not_ready:{con_id}"
            for con_id in requested_ids
            if con_id in held_ids and readiness.get(str(con_id)) is not True
        ),
    ]


def _scoped_entry_reasons(
    reasons: list[str], facts: object, requested_ids: Sequence[int]
) -> list[str]:
    reasons = [
        reason
        for reason in reasons
        if reason != "required_capabilities_missing"
        and not reason.startswith("required_capability_unhealthy:")
        and not reason.startswith("runtime_member_not_armed:")
    ]
    capabilities = facts.get("capabilities") if isinstance(facts, Mapping) else None
    runtime = facts.get("runtime") if isinstance(facts, Mapping) else None
    by_id = (
        {
            int(row.get("con_id", 0) or 0): row
            for row in capabilities
            if isinstance(row, Mapping)
            and int(row.get("con_id", 0) or 0) > 0
        }
        if isinstance(capabilities, Sequence)
        and not isinstance(capabilities, (str, bytes))
        else {}
    )
    selected = [by_id.get(con_id) for con_id in requested_ids]
    for con_id, capability in zip(requested_ids, selected, strict=True):
        if not isinstance(capability, Mapping):
            reasons.append(f"required_capability_scope_unavailable:{con_id}")
        elif capability.get("healthy") is not True:
            reasons.append(
                f"required_capability_unhealthy:{capability.get('label') or con_id}"
            )
    sleeve_ids = {
        str(capability.get("sleeve_id") or "")
        for capability in selected
        if isinstance(capability, Mapping)
    }
    members_by_sleeve = (
        runtime.get("members_by_sleeve") if isinstance(runtime, Mapping) else None
    )
    armed = runtime.get("armed_members") if isinstance(runtime, Mapping) else None
    if (
        not sleeve_ids
        or "" in sleeve_ids
        or not isinstance(members_by_sleeve, Mapping)
        or not isinstance(armed, Sequence)
        or isinstance(armed, (str, bytes))
    ):
        return [*reasons, "runtime_scope_unavailable"]
    armed_set = {str(unit) for unit in armed}
    for sleeve_id in sorted(sleeve_ids):
        members = members_by_sleeve.get(sleeve_id)
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
            reasons.append(f"runtime_scope_unavailable:{sleeve_id}")
            continue
        reasons.extend(
            f"runtime_member_not_armed:{unit}"
            for unit in members
            if str(unit) not in armed_set
        )
    return reasons


def ib_preflight_decision(
    mode: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
    con_ids: Sequence[int] | None = None,
) -> dict[str, object]:
    normalized = str(mode).strip().lower()
    if normalized not in {"entry", "reduction"}:
        raise ValueError("IB preflight mode must be entry or reduction")
    requested_ids = sorted({int(value) for value in con_ids or ()})
    if any(value <= 0 for value in requested_ids):
        raise ValueError("IB preflight contract IDs must be positive")
    configured = _configured_receipt_path(path)
    if configured is None:
        return {
            "configured": False,
            "ready": True,
            "reasons": [],
            **({"con_ids": requested_ids} if requested_ids else {}),
        }
    try:
        max_age = float(
            os.getenv(
                "TRADEBOT_IB_PREFLIGHT_MAX_AGE_SEC",
                str(IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC),
            )
        )
        receipt = load_ib_preflight(configured, now=now, max_age_sec=max_age)
        verdict = receipt["verdict"]
        assert isinstance(verdict, Mapping)
        reasons = list(verdict[f"{normalized}_reasons"])
        if requested_ids:
            facts = receipt.get("facts")
            reasons = (
                _scoped_reduction_reasons(reasons, facts, requested_ids)
                if normalized == "reduction"
                else _scoped_entry_reasons(reasons, facts, requested_ids)
            )
            reasons = sorted(set(reasons))
        return {
            "configured": True,
            "ready": not reasons,
            "reasons": reasons,
            "receipt_id": receipt["receipt_id"],
            **({"con_ids": requested_ids} if requested_ids else {}),
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return {
            "configured": True,
            "ready": False,
            "reasons": [f"ib_preflight_unavailable:{exc}"],
        }


def gate_actionable_plan(
    plan: Mapping[str, object],
    *,
    reduction: bool,
    path: Path | None = None,
) -> dict[str, object]:
    if plan.get("status") != "ACTIONABLE":
        return dict(plan)
    mode = "reduction" if reduction else "entry"
    decision = ib_preflight_decision(mode, path=path)
    if decision["ready"] is True:
        return dict(plan)
    return {
        **dict(plan),
        "status": "HOLD",
        "reason": f"ib_preflight_{mode}_not_ready",
        "leg": None,
        "ib_preflight": decision,
    }


def require_reduction_preflight(
    path: Path | None = None,
    *,
    con_ids: Sequence[int] | None = None,
) -> None:
    decision = ib_preflight_decision("reduction", path=path, con_ids=con_ids)
    if decision["ready"] is not True:
        raise RuntimeError(", ".join(str(reason) for reason in decision["reasons"]))


def order_preflight_mode(*, position: float, action: str, quantity: float) -> str:
    """Classify an exact single-contract order against current broker position."""

    side = str(action or "").strip().upper()
    try:
        held = float(position)
        size = float(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError("order preflight position and quantity must be numeric") from exc
    if not math.isfinite(held) or not math.isfinite(size) or size <= 0:
        raise ValueError("order preflight position and quantity must be finite")
    reduction = (
        (held > 0 and side == "SELL" and size <= held + 1e-9)
        or (held < 0 and side == "BUY" and size <= abs(held) + 1e-9)
    )
    return "reduction" if reduction else "entry"


def require_order_preflight(
    *,
    position: float,
    action: str,
    quantity: float,
    path: Path | None = None,
    con_id: int | None = None,
) -> dict[str, object]:
    """Require the appropriate receipt immediately before broker submission."""

    mode = order_preflight_mode(
        position=position,
        action=action,
        quantity=quantity,
    )
    decision = ib_preflight_decision(
        mode,
        path=path,
        con_ids=(int(con_id),) if con_id is not None else None,
    )
    if decision["ready"] is not True:
        reasons = ", ".join(str(reason) for reason in decision["reasons"])
        raise RuntimeError(f"IB preflight blocked {mode} order: {reasons}")
    return {**decision, "mode": mode}
