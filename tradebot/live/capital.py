"""Immutable account-level capital sleeves for selected live strategies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path


LIVE_CAPITAL_PLAN_SCHEMA = "live.capital-plan.v1"
LIVE_CAPITAL_DECISION_SCHEMA = "live.capital-admission.v1"
LIVE_CAPITAL_ENTRY_INTENTS = {"ENTER", "INCREASE", "ROTATE_IN"}
LIVE_CAPITAL_REDUCTION_INTENTS = {"EXIT", "REDUCE", "ROTATE_OUT"}
LIVE_CAPITAL_KINDS = {"CASH_DEBIT", "DEFINED_RISK_DEBIT"}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_identity(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _aware_utc(value: datetime | str) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("capital-plan timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _usd_cents(value: object, *, rounding: str) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("USD capital must be a finite nonnegative number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD capital must be a finite nonnegative number")
    return int((amount * 100).to_integral_value(rounding=rounding))


def usd_to_cents(value: object) -> int:
    """Conservatively round a USD requirement up to whole cents."""

    return _usd_cents(value, rounding=ROUND_CEILING)


def _validate_sleeve(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "sleeve_id",
        "strategy_id",
        "run_id",
        "selection_path",
        "selection_file_sha256",
        "capital_kind",
        "weight_bps",
    }
    if required - set(value):
        raise ValueError("capital sleeve identity is incomplete")
    sleeve = {key: value[key] for key in required}
    for field in ("sleeve_id", "strategy_id", "selection_path"):
        if not str(sleeve[field] or "").strip():
            raise ValueError(f"capital sleeve {field} is empty")
        sleeve[field] = str(sleeve[field])
    if not _sha256_identity(sleeve["run_id"]):
        raise ValueError("capital sleeve run identity is invalid")
    if not _sha256_identity(sleeve["selection_file_sha256"]):
        raise ValueError("capital sleeve selection identity is invalid")
    sleeve["run_id"] = str(sleeve["run_id"])
    sleeve["selection_file_sha256"] = str(sleeve["selection_file_sha256"])
    sleeve["capital_kind"] = str(sleeve["capital_kind"] or "").upper()
    if sleeve["capital_kind"] not in LIVE_CAPITAL_KINDS:
        raise ValueError("capital sleeve kind is unsupported")
    try:
        sleeve["weight_bps"] = int(sleeve["weight_bps"])
    except (TypeError, ValueError) as exc:
        raise ValueError("capital sleeve weight is invalid") from exc
    if not 1 <= sleeve["weight_bps"] <= 10_000:
        raise ValueError("capital sleeve weight must be 1..10000 bps")
    return sleeve


def build_live_capital_plan(
    *,
    account_id: str,
    account_type: str,
    currency: str,
    observed_settled_cash_usd: object,
    managed_capital_usd: object,
    sleeves: Sequence[Mapping[str, object]],
    reserve_reasons: Sequence[str],
    created_at_utc: datetime | str,
    supersedes_plan_id: str | None = None,
) -> dict[str, object]:
    """Build one content-addressed allocation over explicit managed capital."""

    observed_cents = _usd_cents(observed_settled_cash_usd, rounding=ROUND_FLOOR)
    managed_cents = usd_to_cents(managed_capital_usd)
    if not account_id.strip() or account_type.upper() != "CASH":
        raise ValueError("v1 capital plans require one explicit cash account")
    if currency.upper() != "USD":
        raise ValueError("v1 capital plans support USD sleeves only")
    if managed_cents <= 0 or managed_cents > observed_cents:
        raise ValueError("managed capital must fit positive observed settled cash")
    normalized_sleeves = sorted(
        (_validate_sleeve(value) for value in sleeves),
        key=lambda value: str(value["sleeve_id"]),
    )
    sleeve_ids = [str(value["sleeve_id"]) for value in normalized_sleeves]
    if not sleeve_ids or len(set(sleeve_ids)) != len(sleeve_ids):
        raise ValueError("capital sleeve identities are empty or duplicated")
    if sum(int(value["weight_bps"]) for value in normalized_sleeves) != 10_000:
        raise ValueError("capital sleeve weights must allocate exactly 10000 bps")
    reasons = sorted(set(str(reason).strip() for reason in reserve_reasons if str(reason).strip()))
    reserve_cents = observed_cents - managed_cents
    if reserve_cents and not reasons:
        raise ValueError("unallocated cash requires an explicit reserve reason")
    if supersedes_plan_id is not None and not _sha256_identity(supersedes_plan_id):
        raise ValueError("superseded capital-plan identity is invalid")
    body: dict[str, object] = {
        "schema": LIVE_CAPITAL_PLAN_SCHEMA,
        "created_at_utc": _aware_utc(created_at_utc),
        "supersedes_plan_id": supersedes_plan_id,
        "authority": "entry_ceiling_only",
        "account": {
            "account_id": account_id,
            "account_type": "CASH",
            "currency": "USD",
        },
        "capital": {
            "observed_settled_cash_cents": observed_cents,
            "managed_capital_cents": managed_cents,
            "unallocated_reserve_cents": reserve_cents,
            "reserve_reasons": reasons,
        },
        "sleeves": normalized_sleeves,
        "constraints": {
            "weights_apply_to": "managed_capital_cents",
            "unallocated_reserve_is_entry_authority": False,
            "automatic_borrowing_or_reallocation": False,
            "risk_reduction_requires_plan": False,
        },
    }
    return {**body, "plan_id": _identity(body)}


def validate_live_capital_plan(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one immutable capital-plan generation."""

    plan = dict(value)
    plan_id = str(plan.pop("plan_id", ""))
    if plan.get("schema") != LIVE_CAPITAL_PLAN_SCHEMA or plan_id != _identity(plan):
        raise ValueError("capital-plan content identity is invalid")
    account = plan.get("account")
    capital = plan.get("capital")
    sleeves = plan.get("sleeves")
    constraints = plan.get("constraints")
    if not all(isinstance(item, Mapping) for item in (account, capital, constraints)):
        raise ValueError("capital-plan account or capital contract is invalid")
    if not isinstance(sleeves, Sequence) or isinstance(sleeves, (str, bytes)):
        raise ValueError("capital-plan sleeves are invalid")
    rebuilt = build_live_capital_plan(
        account_id=str(account.get("account_id") or ""),
        account_type=str(account.get("account_type") or ""),
        currency=str(account.get("currency") or ""),
        observed_settled_cash_usd=Decimal(
            int(capital.get("observed_settled_cash_cents", -1))
        )
        / 100,
        managed_capital_usd=Decimal(int(capital.get("managed_capital_cents", -1)))
        / 100,
        sleeves=[dict(item) for item in sleeves if isinstance(item, Mapping)],
        reserve_reasons=list(capital.get("reserve_reasons") or ()),
        created_at_utc=str(plan.get("created_at_utc") or ""),
        supersedes_plan_id=(
            str(plan["supersedes_plan_id"])
            if plan.get("supersedes_plan_id") is not None
            else None
        ),
    )
    if rebuilt != value:
        raise ValueError("capital-plan normalized contract changed")
    return rebuilt


def load_live_capital_plan(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("capital-plan document must be an object")
    return validate_live_capital_plan(value)


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
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish_live_capital_plan(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    """Atomically publish one generation while retaining its predecessor."""

    plan = validate_live_capital_plan(value)
    payload = json.dumps(plan, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        previous = load_live_capital_plan(path)
        if previous["plan_id"] == plan["plan_id"]:
            return plan
        if plan.get("supersedes_plan_id") != previous["plan_id"]:
            raise ValueError("capital-plan replacement does not bind its predecessor")
        archive = path.with_name(f"{path.stem}.{previous['plan_id']}{path.suffix}")
        previous_payload = path.read_bytes()
        if archive.exists() and archive.read_bytes() != previous_payload:
            raise ValueError("capital-plan predecessor archive identity changed")
        if not archive.exists():
            _atomic_write(archive, previous_payload)
    _atomic_write(path, payload)
    return plan


def _decision(body: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(body)
    return {**frozen, "decision_id": _identity(frozen)}


def validate_live_capital_decision(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate one content-addressed admission recorded with an order plan."""

    decision = dict(value)
    decision_id = str(decision.pop("decision_id", ""))
    intent = str(decision.get("intent") or "")
    status = str(decision.get("status") or "")
    reasons = decision.get("reasons")
    plan_id = decision.get("plan_id")
    if (
        decision.get("schema") != LIVE_CAPITAL_DECISION_SCHEMA
        or decision_id != _identity(decision)
        or intent not in LIVE_CAPITAL_ENTRY_INTENTS | LIVE_CAPITAL_REDUCTION_INTENTS
        or status not in {"ALLOW", "HOLD"}
        or not str(decision.get("sleeve_id") or "")
        or not _sha256_identity(decision.get("run_id"))
        or not isinstance(reasons, Sequence)
        or isinstance(reasons, (str, bytes))
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or (plan_id is not None and not _sha256_identity(plan_id))
    ):
        raise ValueError("capital admission identity or shape is invalid")
    if intent in LIVE_CAPITAL_ENTRY_INTENTS:
        if (
            (status == "ALLOW" and (reasons or plan_id is None))
            or (status == "HOLD" and not reasons)
            or (plan_id is None) != (decision.get("allocation") is None)
        ):
            raise ValueError("capital entry admission contract is invalid")
    elif (
        status != "ALLOW"
        or reasons != ["risk_reduction_always_allowed"]
        or decision.get("allocation") is not None
    ):
        raise ValueError("capital reduction admission contract is invalid")
    return dict(value)


def admit_live_capital(
    plan: Mapping[str, object] | None,
    *,
    intent: str,
    account_id: str,
    account_type: str,
    currency: str,
    sleeve_id: str,
    run_id: str,
    selection_file_sha256: str,
    capital_kind: str,
    projected_capital_usd: object,
    cash_debit_usd: object,
    available_cash_usd: object,
) -> dict[str, object]:
    """Admit one entry ceiling or unconditionally preserve risk reduction."""

    normalized_intent = str(intent or "").upper()
    base = {
        "schema": LIVE_CAPITAL_DECISION_SCHEMA,
        "intent": normalized_intent,
        "sleeve_id": sleeve_id,
        "run_id": run_id,
    }
    if normalized_intent in LIVE_CAPITAL_REDUCTION_INTENTS:
        return _decision(
            {
                **base,
                "status": "ALLOW",
                "reasons": ["risk_reduction_always_allowed"],
                "plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
                "allocation": None,
            }
        )
    if normalized_intent not in LIVE_CAPITAL_ENTRY_INTENTS:
        raise ValueError("capital intent is invalid")
    try:
        validated = validate_live_capital_plan(plan or {})
    except (TypeError, ValueError):
        return _decision(
            {
                **base,
                "status": "HOLD",
                "reasons": ["invalid_or_missing_capital_plan"],
                "plan_id": None,
                "allocation": None,
            }
        )
    account = validated["account"]
    sleeves = validated["sleeves"]
    capital = validated["capital"]
    assert isinstance(account, Mapping) and isinstance(capital, Mapping)
    matches = [
        item
        for item in sleeves
        if isinstance(item, Mapping) and item.get("sleeve_id") == sleeve_id
    ]
    reasons: list[str] = []
    if len(matches) != 1:
        reasons.append("capital_sleeve_not_allocated")
        sleeve = None
    else:
        sleeve = matches[0]
    if (
        account.get("account_id") != account_id
        or account.get("account_type") != account_type.upper()
        or account.get("currency") != currency.upper()
    ):
        reasons.append("capital_account_identity_mismatch")
    if sleeve is not None:
        if sleeve.get("run_id") != run_id:
            reasons.append("capital_run_identity_mismatch")
        if sleeve.get("selection_file_sha256") != selection_file_sha256:
            reasons.append("capital_selection_identity_mismatch")
        if sleeve.get("capital_kind") != capital_kind.upper():
            reasons.append("capital_kind_mismatch")
    projected_cents = usd_to_cents(projected_capital_usd)
    debit_cents = usd_to_cents(cash_debit_usd)
    available_cents = _usd_cents(available_cash_usd, rounding=ROUND_FLOOR)
    weight_bps = int(sleeve["weight_bps"]) if sleeve is not None else 0
    managed_cents = int(capital["managed_capital_cents"])
    sleeve_limit_cents = managed_cents * weight_bps // 10_000
    if projected_cents > sleeve_limit_cents:
        reasons.append("capital_sleeve_limit_exceeded")
    if debit_cents > available_cents:
        reasons.append("insufficient_live_cash")
    allocation = {
        "managed_capital_cents": managed_cents,
        "weight_bps": weight_bps,
        "sleeve_limit_cents": sleeve_limit_cents,
        "projected_capital_cents": projected_cents,
        "cash_debit_cents": debit_cents,
        "available_cash_cents": available_cents,
        "unallocated_reserve_cents": int(capital["unallocated_reserve_cents"]),
    }
    return _decision(
        {
            **base,
            "status": "HOLD" if reasons else "ALLOW",
            "reasons": sorted(set(reasons)),
            "plan_id": validated["plan_id"],
            "allocation": allocation,
        }
    )
