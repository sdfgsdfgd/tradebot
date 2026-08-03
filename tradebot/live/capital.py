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

from .capital_packages import allocate_live_packages, live_package_entry_capacity


LIVE_CAPITAL_PLAN_SCHEMA = "live.capital-plan.v1"
LIVE_CAPITAL_PLAN_V2_SCHEMA = "live.capital-plan.v2"
LIVE_CAPITAL_PLAN_V3_SCHEMA = "live.capital-plan.v3"
LIVE_CAPITAL_PLAN_SCHEMAS = {
    LIVE_CAPITAL_PLAN_SCHEMA,
    LIVE_CAPITAL_PLAN_V2_SCHEMA,
    LIVE_CAPITAL_PLAN_V3_SCHEMA,
}
LIVE_CAPITAL_DECISION_SCHEMA = "live.capital-admission.v1"
LIVE_CAPITAL_ENTRY_INTENTS = {"ENTER", "INCREASE", "ROTATE_IN"}
LIVE_CAPITAL_REDUCTION_INTENTS = {"EXIT", "REDUCE", "ROTATE_OUT"}
LIVE_CAPITAL_KINDS = {"CASH_DEBIT", "DEFINED_RISK_DEBIT", "FUTURES_MARGIN"}


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
    if sleeve["capital_kind"] == "FUTURES_MARGIN":
        valid_weight = sleeve["weight_bps"] == 0
    else:
        valid_weight = 1 <= sleeve["weight_bps"] <= 10_000
    if not valid_weight:
        raise ValueError("capital sleeve weight must be 1..10000 bps")
    return sleeve


def _validate_v2_sleeve(value: Mapping[str, object]) -> dict[str, object]:
    sleeve = _validate_sleeve(value)
    symbols = value.get("position_symbols")
    if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
        raise ValueError("capital sleeve position symbols are invalid")
    position_symbols = sorted(
        set(str(symbol or "").strip().upper() for symbol in symbols)
    )
    if not position_symbols or any(not symbol for symbol in position_symbols):
        raise ValueError("capital sleeve position symbols are empty")
    sleeve["position_symbols"] = position_symbols
    if sleeve["capital_kind"] != "FUTURES_MARGIN":
        if value.get("margin") is not None:
            raise ValueError("cash capital sleeve cannot own a margin contract")
        return sleeve

    if sleeve["weight_bps"] != 0:
        raise ValueError("futures-margin sleeves cannot consume the cash pool")
    margin = value.get("margin")
    required = {
        "base_currency",
        "max_contracts",
        "max_initial_margin_change_cents",
        "max_maintenance_margin_change_cents",
        "max_stressed_loss_usd_cents",
        "fx_stress_bps",
        "minimum_post_stress_excess_liquidity_cents",
    }
    if not isinstance(margin, Mapping) or required - set(margin):
        raise ValueError("futures-margin sleeve contract is incomplete")
    normalized_margin = {key: margin[key] for key in required}
    normalized_margin["base_currency"] = str(
        normalized_margin["base_currency"] or ""
    ).upper()
    if not normalized_margin["base_currency"]:
        raise ValueError("futures-margin base currency is empty")
    for field in required - {"base_currency"}:
        try:
            normalized_margin[field] = int(normalized_margin[field])
        except (TypeError, ValueError) as exc:
            raise ValueError("futures-margin limits must be integers") from exc
    if (
        normalized_margin["max_contracts"] <= 0
        or normalized_margin["max_initial_margin_change_cents"] <= 0
        or normalized_margin["max_maintenance_margin_change_cents"] <= 0
        or normalized_margin["max_stressed_loss_usd_cents"] <= 0
        or normalized_margin["fx_stress_bps"] < 10_000
        or normalized_margin["minimum_post_stress_excess_liquidity_cents"] < 0
    ):
        raise ValueError("futures-margin limits are invalid")
    sleeve["margin"] = normalized_margin
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
    if any(value["capital_kind"] == "FUTURES_MARGIN" for value in normalized_sleeves):
        raise ValueError("v1 capital plans cannot allocate futures margin")
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


def build_live_capital_plan_v2(
    *,
    account_id: str,
    account_type: str,
    cash_currency: str,
    base_currency: str,
    observed_settled_cash_usd: object,
    managed_capital_usd: object,
    sleeves: Sequence[Mapping[str, object]],
    reserve_reasons: Sequence[str],
    created_at_utc: datetime | str,
    max_concurrent_directional_sleeves: int = 1,
    supersedes_plan_id: str | None = None,
) -> dict[str, object]:
    """Build one cash pool plus indivisible base-currency margin overlays."""

    observed_cents = _usd_cents(observed_settled_cash_usd, rounding=ROUND_FLOOR)
    managed_cents = usd_to_cents(managed_capital_usd)
    normalized_base = str(base_currency or "").upper()
    if (
        not account_id.strip()
        or account_type.upper() != "CASH"
        or cash_currency.upper() != "USD"
        or not normalized_base
    ):
        raise ValueError("v2 capital plans require one USD cash account and base currency")
    if managed_cents <= 0 or managed_cents > observed_cents:
        raise ValueError("managed capital must fit positive observed settled cash")
    normalized_sleeves = sorted(
        (_validate_v2_sleeve(value) for value in sleeves),
        key=lambda value: str(value["sleeve_id"]),
    )
    sleeve_ids = [str(value["sleeve_id"]) for value in normalized_sleeves]
    cash_sleeves = [
        value for value in normalized_sleeves
        if value["capital_kind"] != "FUTURES_MARGIN"
    ]
    margin_sleeves = [
        value for value in normalized_sleeves
        if value["capital_kind"] == "FUTURES_MARGIN"
    ]
    if (
        not sleeve_ids
        or len(set(sleeve_ids)) != len(sleeve_ids)
        or not cash_sleeves
        or not margin_sleeves
        or sum(int(value["weight_bps"]) for value in cash_sleeves) != 10_000
        or any(value["margin"]["base_currency"] != normalized_base for value in margin_sleeves)
    ):
        raise ValueError("v2 cash and futures-margin sleeves are inconsistent")
    try:
        maximum_open = int(max_concurrent_directional_sleeves)
    except (TypeError, ValueError) as exc:
        raise ValueError("concurrent directional sleeve limit is invalid") from exc
    if maximum_open != 1:
        raise ValueError("v2 currently requires one account-level directional exposure")
    reasons = sorted(set(str(reason).strip() for reason in reserve_reasons if str(reason).strip()))
    reserve_cents = observed_cents - managed_cents
    if reserve_cents and not reasons:
        raise ValueError("unallocated cash requires an explicit reserve reason")
    if supersedes_plan_id is not None and not _sha256_identity(supersedes_plan_id):
        raise ValueError("superseded capital-plan identity is invalid")
    body: dict[str, object] = {
        "schema": LIVE_CAPITAL_PLAN_V2_SCHEMA,
        "created_at_utc": _aware_utc(created_at_utc),
        "supersedes_plan_id": supersedes_plan_id,
        "authority": "entry_ceiling_and_cross_sleeve_risk",
        "account": {
            "account_id": account_id,
            "account_type": "CASH",
            "cash_currency": "USD",
            "base_currency": normalized_base,
        },
        "capital": {
            "observed_settled_cash_cents": observed_cents,
            "managed_capital_cents": managed_cents,
            "unallocated_reserve_cents": reserve_cents,
            "reserve_reasons": reasons,
        },
        "sleeves": normalized_sleeves,
        "constraints": {
            "cash_weights_apply_to": "managed_capital_cents",
            "futures_margin_is_not_cash_debit_authority": True,
            "unallocated_reserve_is_entry_authority": False,
            "automatic_borrowing_or_reallocation": False,
            "risk_reduction_requires_plan": False,
            "max_concurrent_directional_sleeves": maximum_open,
        },
    }
    return {**body, "plan_id": _identity(body)}


def build_live_capital_plan_v3(
    *,
    account_id: str,
    account_type: str,
    cash_currency: str,
    base_currency: str,
    observed_settled_cash_usd: object,
    observed_available_funds_base: object,
    observed_excess_liquidity_base: object,
    usd_to_base_rate: object,
    minimum_post_reservation_base: object,
    unmanaged_position_stress_base: object,
    sleeves: Sequence[Mapping[str, object]],
    reserve_reasons: Sequence[str],
    created_at_utc: datetime | str,
    supersedes_plan_id: str | None = None,
) -> dict[str, object]:
    """Build one minimum-first allocation across cash, margin, and risk."""

    if (
        not account_id.strip()
        or account_type.upper() != "CASH"
        or cash_currency.upper() != "USD"
        or not str(base_currency or "").strip()
    ):
        raise ValueError("v3 capital plans require one USD cash account and base currency")
    try:
        rate = Decimal(str(usd_to_base_rate))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("USD/base conversion rate is invalid") from exc
    if not rate.is_finite() or rate <= 0:
        raise ValueError("USD/base conversion rate is invalid")
    resources = {
        "observed_settled_cash_usd_cents": _usd_cents(
            observed_settled_cash_usd, rounding=ROUND_FLOOR
        ),
        "observed_available_funds_base_cents": _usd_cents(
            observed_available_funds_base, rounding=ROUND_FLOOR
        ),
        "observed_excess_liquidity_base_cents": _usd_cents(
            observed_excess_liquidity_base, rounding=ROUND_FLOOR
        ),
        "usd_to_base_rate_ppm": int(
            (rate * 1_000_000).to_integral_value(rounding=ROUND_CEILING)
        ),
        "minimum_buffer_base_cents": usd_to_cents(
            minimum_post_reservation_base
        ),
        "unmanaged_position_stress_base_cents": usd_to_cents(
            unmanaged_position_stress_base
        ),
    }
    if any(
        not _sha256_identity(sleeve.get(field))
        for sleeve in sleeves
        for field in ("run_id", "selection_file_sha256")
    ):
        raise ValueError("package sleeve run or selection identity is invalid")
    allocated, allocation = allocate_live_packages(
        sleeves,
        settled_cash_usd_cents=resources["observed_settled_cash_usd_cents"],
        available_funds_base_cents=resources[
            "observed_available_funds_base_cents"
        ],
        excess_liquidity_base_cents=resources[
            "observed_excess_liquidity_base_cents"
        ],
        usd_to_base_rate_ppm=resources["usd_to_base_rate_ppm"],
        minimum_buffer_base_cents=resources["minimum_buffer_base_cents"],
        unmanaged_position_stress_base_cents=resources[
            "unmanaged_position_stress_base_cents"
        ],
    )
    reasons = sorted(
        {str(reason).strip() for reason in reserve_reasons if str(reason).strip()}
    )
    managed_cents = int(allocation["capacity"]["cash_debit_usd_cents"])
    reserve_cents = resources["observed_settled_cash_usd_cents"] - managed_cents
    if reserve_cents and not reasons:
        raise ValueError("unallocated cash requires an explicit reserve reason")
    if supersedes_plan_id is not None and not _sha256_identity(supersedes_plan_id):
        raise ValueError("superseded capital-plan identity is invalid")
    body: dict[str, object] = {
        "schema": LIVE_CAPITAL_PLAN_V3_SCHEMA,
        "created_at_utc": _aware_utc(created_at_utc),
        "supersedes_plan_id": supersedes_plan_id,
        "authority": "minimum_packages_and_portfolio_resource_reservation",
        "account": {
            "account_id": account_id,
            "account_type": "CASH",
            "cash_currency": "USD",
            "base_currency": str(base_currency).upper(),
        },
        "capital": {
            "observed_settled_cash_cents": resources[
                "observed_settled_cash_usd_cents"
            ],
            "managed_capital_cents": managed_cents,
            "unallocated_reserve_cents": reserve_cents,
            "reserve_reasons": reasons,
        },
        "resources": resources,
        "sleeves": allocated,
        "allocation": allocation,
        "constraints": {
            "minimum_executable_packages_reserved_first": True,
            "residual_allocation": "minimum_first_weighted_residual.v1",
            "flat_sleeves_retain_allocated_package_reservation": True,
            "unmanaged_positions_receive_full_gross_stress": True,
            "automatic_borrowing_or_unproved_reallocation": False,
            "risk_reduction_requires_plan": False,
        },
    }
    return {**body, "plan_id": _identity(body)}


def _validate_live_capital_plan_v1(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one immutable v1 cash-plan generation."""

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


def _validate_live_capital_plan_v2(value: Mapping[str, object]) -> dict[str, object]:
    plan = dict(value)
    plan_id = str(plan.pop("plan_id", ""))
    if plan.get("schema") != LIVE_CAPITAL_PLAN_V2_SCHEMA or plan_id != _identity(plan):
        raise ValueError("capital-plan content identity is invalid")
    account = plan.get("account")
    capital = plan.get("capital")
    sleeves = plan.get("sleeves")
    constraints = plan.get("constraints")
    if not all(isinstance(item, Mapping) for item in (account, capital, constraints)):
        raise ValueError("capital-plan account or capital contract is invalid")
    if not isinstance(sleeves, Sequence) or isinstance(sleeves, (str, bytes)):
        raise ValueError("capital-plan sleeves are invalid")
    rebuilt = build_live_capital_plan_v2(
        account_id=str(account.get("account_id") or ""),
        account_type=str(account.get("account_type") or ""),
        cash_currency=str(account.get("cash_currency") or ""),
        base_currency=str(account.get("base_currency") or ""),
        observed_settled_cash_usd=Decimal(
            int(capital.get("observed_settled_cash_cents", -1))
        ) / 100,
        managed_capital_usd=Decimal(
            int(capital.get("managed_capital_cents", -1))
        ) / 100,
        sleeves=[dict(item) for item in sleeves if isinstance(item, Mapping)],
        reserve_reasons=list(capital.get("reserve_reasons") or ()),
        created_at_utc=str(plan.get("created_at_utc") or ""),
        max_concurrent_directional_sleeves=int(
            constraints.get("max_concurrent_directional_sleeves", 0)
        ),
        supersedes_plan_id=(
            str(plan["supersedes_plan_id"])
            if plan.get("supersedes_plan_id") is not None
            else None
        ),
    )
    if rebuilt != value:
        raise ValueError("capital-plan normalized contract changed")
    return rebuilt


def _validate_live_capital_plan_v3(value: Mapping[str, object]) -> dict[str, object]:
    plan = dict(value)
    plan_id = str(plan.pop("plan_id", ""))
    if plan.get("schema") != LIVE_CAPITAL_PLAN_V3_SCHEMA or plan_id != _identity(plan):
        raise ValueError("capital-plan content identity is invalid")
    account = plan.get("account")
    capital = plan.get("capital")
    resources = plan.get("resources")
    sleeves = plan.get("sleeves")
    if (
        not all(isinstance(item, Mapping) for item in (account, capital, resources))
        or not isinstance(sleeves, Sequence)
        or isinstance(sleeves, (str, bytes))
    ):
        raise ValueError("v3 capital-plan resources are invalid")
    rebuilt = build_live_capital_plan_v3(
        account_id=str(account.get("account_id") or ""),
        account_type=str(account.get("account_type") or ""),
        cash_currency=str(account.get("cash_currency") or ""),
        base_currency=str(account.get("base_currency") or ""),
        observed_settled_cash_usd=Decimal(
            int(resources.get("observed_settled_cash_usd_cents", -1))
        ) / 100,
        observed_available_funds_base=Decimal(
            int(resources.get("observed_available_funds_base_cents", -1))
        ) / 100,
        observed_excess_liquidity_base=Decimal(
            int(resources.get("observed_excess_liquidity_base_cents", -1))
        ) / 100,
        usd_to_base_rate=Decimal(int(resources.get("usd_to_base_rate_ppm", -1)))
        / 1_000_000,
        minimum_post_reservation_base=Decimal(
            int(resources.get("minimum_buffer_base_cents", -1))
        ) / 100,
        unmanaged_position_stress_base=Decimal(
            int(resources.get("unmanaged_position_stress_base_cents", -1))
        ) / 100,
        sleeves=[
            {key: item for key, item in sleeve.items() if key != "allocated_package_id"}
            for sleeve in sleeves
            if isinstance(sleeve, Mapping)
        ],
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


def validate_live_capital_plan(value: Mapping[str, object]) -> dict[str, object]:
    """Validate either immutable account capital-plan generation."""

    schema = value.get("schema")
    if schema == LIVE_CAPITAL_PLAN_SCHEMA:
        return _validate_live_capital_plan_v1(value)
    if schema == LIVE_CAPITAL_PLAN_V2_SCHEMA:
        return _validate_live_capital_plan_v2(value)
    if schema == LIVE_CAPITAL_PLAN_V3_SCHEMA:
        return _validate_live_capital_plan_v3(value)
    raise ValueError("capital-plan schema is unsupported")


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


def _resource_cents(value: object, *, rounding: str) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("capital resource must be finite and nonnegative") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("capital resource must be finite and nonnegative")
    return int((amount * 100).to_integral_value(rounding=rounding))


def _v2_active_sleeves(
    plan: Mapping[str, object],
    *,
    current_sleeve_id: str,
    resource_state: Mapping[str, object] | None,
) -> tuple[list[str], list[str]]:
    if not isinstance(resource_state, Mapping):
        return [], ["account_exposure_state_missing"]
    positions = resource_state.get("account_positions")
    open_orders = resource_state.get("account_open_orders")
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or not isinstance(open_orders, Sequence)
        or isinstance(open_orders, (str, bytes))
    ):
        return [], ["account_exposure_state_missing"]
    active_symbols: set[str] = set()
    try:
        for row in (*positions, *open_orders):
            if not isinstance(row, Mapping):
                raise ValueError
            symbol = str(row.get("symbol") or "").strip().upper()
            quantity = float(row.get("quantity") or 0.0)
            if symbol and abs(quantity) > 1e-9:
                active_symbols.add(symbol)
    except (TypeError, ValueError):
        return [], ["account_exposure_state_invalid"]
    active = sorted(
        str(sleeve["sleeve_id"])
        for sleeve in plan["sleeves"]
        if isinstance(sleeve, Mapping)
        and sleeve.get("sleeve_id") != current_sleeve_id
        and active_symbols.intersection(sleeve.get("position_symbols") or ())
    )
    return active, (["concurrent_directional_sleeve_active"] if active else [])


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
    resource_state: Mapping[str, object] | None = None,
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
    cash_currency = (
        account.get("cash_currency")
        if validated["schema"]
        in {LIVE_CAPITAL_PLAN_V2_SCHEMA, LIVE_CAPITAL_PLAN_V3_SCHEMA}
        else account.get("currency")
    )
    if (
        account.get("account_id") != account_id
        or account.get("account_type") != account_type.upper()
        or cash_currency != currency.upper()
    ):
        reasons.append("capital_account_identity_mismatch")
    if sleeve is not None:
        if sleeve.get("run_id") != run_id:
            reasons.append("capital_run_identity_mismatch")
        if sleeve.get("selection_file_sha256") != selection_file_sha256:
            reasons.append("capital_selection_identity_mismatch")
        if sleeve.get("capital_kind") != capital_kind.upper():
            reasons.append("capital_kind_mismatch")
    if validated["schema"] == LIVE_CAPITAL_PLAN_V3_SCHEMA and sleeve is not None:
        allocation, capacity_reasons = live_package_entry_capacity(
            validated,
            sleeve_id=sleeve_id,
            resource_state=resource_state,
            available_cash_usd_cents=_usd_cents(
                available_cash_usd, rounding=ROUND_FLOOR
            ),
            candidate_cash_debit_usd_cents=usd_to_cents(cash_debit_usd),
        )
        reasons.extend(capacity_reasons)
        return _decision(
            {
                **base,
                "status": "HOLD" if reasons else "ALLOW",
                "reasons": sorted(set(reasons)),
                "plan_id": validated["plan_id"],
                "allocation": allocation,
            }
        )
    active_sleeves: list[str] = []
    if validated["schema"] == LIVE_CAPITAL_PLAN_V2_SCHEMA:
        active_sleeves, exposure_reasons = _v2_active_sleeves(
            validated,
            current_sleeve_id=sleeve_id,
            resource_state=resource_state,
        )
        reasons.extend(exposure_reasons)

    if sleeve is not None and sleeve.get("capital_kind") == "FUTURES_MARGIN":
        margin = sleeve.get("margin")
        allocation: dict[str, object] = {
            "capital_kind": "FUTURES_MARGIN",
            "active_other_sleeves": active_sleeves,
        }
        try:
            if not isinstance(margin, Mapping) or not isinstance(resource_state, Mapping):
                raise ValueError
            base_currency = str(resource_state.get("base_currency") or "").upper()
            quantity = int(resource_state["quantity"])
            initial_change = _resource_cents(
                resource_state["initial_margin_change"], rounding=ROUND_CEILING
            )
            maintenance_change = _resource_cents(
                resource_state["maintenance_margin_change"], rounding=ROUND_CEILING
            )
            initial_after = _resource_cents(
                resource_state["initial_margin_after"], rounding=ROUND_CEILING
            )
            maintenance_after = _resource_cents(
                resource_state["maintenance_margin_after"], rounding=ROUND_CEILING
            )
            equity_after = _resource_cents(
                resource_state["equity_with_loan_after"], rounding=ROUND_FLOOR
            )
            available_before = _resource_cents(
                resource_state["available_funds_before"], rounding=ROUND_FLOOR
            )
            unrelated_gross = _resource_cents(
                resource_state["unrelated_position_gross"], rounding=ROUND_CEILING
            )
            fx_rate = Decimal(str(resource_state["usd_to_base_rate"]))
            if (
                base_currency != margin["base_currency"]
                or quantity <= 0
                or not fx_rate.is_finite()
                or fx_rate <= 0
            ):
                raise ValueError
            stressed_loss = int(
                (
                    Decimal(int(margin["max_stressed_loss_usd_cents"]))
                    * fx_rate
                    * Decimal(int(margin["fx_stress_bps"]))
                    / Decimal(10_000)
                ).to_integral_value(rounding=ROUND_CEILING)
            )
            minimum_buffer = int(
                margin["minimum_post_stress_excess_liquidity_cents"]
            )
            post_entry_available = available_before - initial_change
            post_stress_excess = (
                equity_after - maintenance_after - unrelated_gross - stressed_loss
            )
            if quantity > int(margin["max_contracts"]):
                reasons.append("futures_contract_limit_exceeded")
            if initial_change > int(margin["max_initial_margin_change_cents"]):
                reasons.append("initial_margin_limit_exceeded")
            if maintenance_change > int(
                margin["max_maintenance_margin_change_cents"]
            ):
                reasons.append("maintenance_margin_limit_exceeded")
            if post_entry_available < minimum_buffer:
                reasons.append("post_entry_available_funds_below_floor")
            if post_stress_excess < minimum_buffer:
                reasons.append("post_stress_excess_liquidity_below_floor")
            allocation.update(
                {
                    "base_currency": base_currency,
                    "quantity": quantity,
                    "initial_margin_change_cents": initial_change,
                    "maintenance_margin_change_cents": maintenance_change,
                    "initial_margin_after_cents": initial_after,
                    "maintenance_margin_after_cents": maintenance_after,
                    "equity_with_loan_after_cents": equity_after,
                    "available_funds_before_cents": available_before,
                    "unrelated_position_gross_cents": unrelated_gross,
                    "stressed_loss_base_cents": stressed_loss,
                    "post_entry_available_funds_cents": post_entry_available,
                    "post_stress_excess_liquidity_cents": post_stress_excess,
                    "minimum_buffer_cents": minimum_buffer,
                }
            )
        except (InvalidOperation, KeyError, TypeError, ValueError):
            reasons.append("invalid_futures_margin_state")
        return _decision(
            {
                **base,
                "status": "HOLD" if reasons else "ALLOW",
                "reasons": sorted(set(reasons)),
                "plan_id": validated["plan_id"],
                "allocation": allocation,
            }
        )

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
    if validated["schema"] == LIVE_CAPITAL_PLAN_V2_SCHEMA:
        allocation = {
            "capital_kind": str(capital_kind).upper(),
            "active_other_sleeves": active_sleeves,
            **allocation,
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
