"""Deterministic minimum-first resource allocation for live strategy packages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


PACKAGE_RESOURCE_FIELDS = (
    "cash_debit_usd_cents",
    "initial_margin_base_cents",
    "maintenance_margin_base_cents",
    "stressed_loss_usd_cents",
)
PACKAGE_ALLOCATION_METHOD = "minimum_first_weighted_residual.v1"
PACKAGE_FIRST_ADMITTER_METHOD = "first_admitter_just_in_time.v1"
IMMUTABLE_SELECTION_DIRECTORY = Path("db/calibration/selections")


def _nonnegative_int(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if isinstance(value, bool) or parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


def _positive_int(value: object, *, name: str) -> int:
    parsed = _nonnegative_int(value, name=name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _resource_package(value: Mapping[str, object]) -> dict[str, object]:
    required = {"package_id", "rank", "fx_stress_bps", *PACKAGE_RESOURCE_FIELDS}
    if set(value) != required:
        raise ValueError("executable package resource contract is incomplete")
    package_id = str(value.get("package_id") or "").strip()
    if not package_id:
        raise ValueError("executable package identity is empty")
    package = {
        "package_id": package_id,
        "rank": _nonnegative_int(value["rank"], name="package rank"),
        **{
            field: _nonnegative_int(value[field], name=field)
            for field in PACKAGE_RESOURCE_FIELDS
        },
        "fx_stress_bps": _positive_int(
            value["fx_stress_bps"], name="package FX stress"
        ),
    }
    if package["fx_stress_bps"] < 10_000:
        raise ValueError("package FX stress cannot be below spot")
    if not any(package[field] for field in PACKAGE_RESOURCE_FIELDS):
        raise ValueError("executable package consumes no account resource")
    return package


def normalize_package_sleeve(value: Mapping[str, object]) -> dict[str, object]:
    """Validate one strategy's ordered, indivisible executable-package ladder."""

    required = {
        "sleeve_id",
        "strategy_id",
        "run_id",
        "selection_path",
        "selection_file_sha256",
        "capital_kind",
        "position_symbols",
        "residual_weight_bps",
        "minimum_package_id",
        "package_ladder",
    }
    if set(value) != required:
        raise ValueError("package capital sleeve contract is incomplete")
    normalized = {
        key: str(value.get(key) or "").strip()
        for key in (
            "sleeve_id",
            "strategy_id",
            "run_id",
            "selection_path",
            "selection_file_sha256",
            "capital_kind",
            "minimum_package_id",
        )
    }
    if any(not item for item in normalized.values()):
        raise ValueError("package capital sleeve identity is empty")
    normalized["capital_kind"] = normalized["capital_kind"].upper()
    symbols = value.get("position_symbols")
    ladder = value.get("package_ladder")
    if (
        not isinstance(symbols, Sequence)
        or isinstance(symbols, (str, bytes))
        or not isinstance(ladder, Sequence)
        or isinstance(ladder, (str, bytes))
    ):
        raise ValueError("package sleeve symbols or ladder are invalid")
    normalized_symbols = sorted(
        {str(symbol or "").strip().upper() for symbol in symbols}
    )
    packages = sorted(
        (
            _resource_package(package)
            for package in ladder
            if isinstance(package, Mapping)
        ),
        key=lambda package: int(package["rank"]),
    )
    if (
        not normalized_symbols
        or any(not symbol for symbol in normalized_symbols)
        or len(packages) != len(ladder)
        or not packages
        or [package["rank"] for package in packages] != list(range(len(packages)))
        or len({package["package_id"] for package in packages}) != len(packages)
        or normalized["minimum_package_id"]
        not in {package["package_id"] for package in packages}
    ):
        raise ValueError("package sleeve ladder is empty, duplicated, or discontinuous")
    for previous, current in zip(packages, packages[1:], strict=False):
        if any(
            int(current[field]) < int(previous[field])
            for field in PACKAGE_RESOURCE_FIELDS
        ) or not any(
            int(current[field]) > int(previous[field])
            for field in PACKAGE_RESOURCE_FIELDS
        ):
            raise ValueError("package resources must increase monotonically")
    weight = _nonnegative_int(
        value["residual_weight_bps"], name="residual package weight"
    )
    minimum_index = next(
        index
        for index, package in enumerate(packages)
        if package["package_id"] == normalized["minimum_package_id"]
    )
    if minimum_index < len(packages) - 1 and weight == 0:
        packages = packages[: minimum_index + 1]
    return {
        **normalized,
        "position_symbols": normalized_symbols,
        "residual_weight_bps": weight,
        "package_ladder": packages,
    }


def _ceil_ratio(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _capacity(
    sleeves: Sequence[Mapping[str, object]],
    indexes: Mapping[str, int],
    *,
    settled_cash_usd_cents: int,
    available_funds_base_cents: int,
    excess_liquidity_base_cents: int,
    usd_to_base_rate_ppm: int,
    minimum_buffer_base_cents: int,
    unmanaged_position_stress_base_cents: int,
) -> dict[str, int]:
    totals = {field: 0 for field in PACKAGE_RESOURCE_FIELDS}
    cash_base = stressed_base = 0
    for sleeve in sleeves:
        package = sleeve["package_ladder"][indexes[str(sleeve["sleeve_id"])]]
        for field in PACKAGE_RESOURCE_FIELDS:
            totals[field] += int(package[field])
        cash_base += _ceil_ratio(
            int(package["cash_debit_usd_cents"]) * usd_to_base_rate_ppm,
            1_000_000,
        )
        stressed_base += _ceil_ratio(
            int(package["stressed_loss_usd_cents"])
            * usd_to_base_rate_ppm
            * int(package["fx_stress_bps"]),
            1_000_000 * 10_000,
        )
    return {
        **totals,
        "cash_debit_base_cents": cash_base,
        "stressed_loss_base_cents": stressed_base,
        "settled_cash_remaining_cents": (
            settled_cash_usd_cents - totals["cash_debit_usd_cents"]
        ),
        "post_reservation_available_funds_base_cents": (
            available_funds_base_cents
            - cash_base
            - totals["initial_margin_base_cents"]
        ),
        "post_stress_excess_liquidity_base_cents": (
            excess_liquidity_base_cents
            - unmanaged_position_stress_base_cents
            - totals["maintenance_margin_base_cents"]
            - stressed_base
        ),
        "minimum_buffer_base_cents": minimum_buffer_base_cents,
    }


def _fits(capacity: Mapping[str, int]) -> bool:
    floor = int(capacity["minimum_buffer_base_cents"])
    return bool(
        int(capacity["settled_cash_remaining_cents"]) >= 0
        and int(capacity["post_reservation_available_funds_base_cents"]) >= floor
        and int(capacity["post_stress_excess_liquidity_base_cents"]) >= floor
    )


def allocate_live_packages(
    sleeves: Sequence[Mapping[str, object]],
    *,
    settled_cash_usd_cents: int,
    available_funds_base_cents: int,
    excess_liquidity_base_cents: int,
    usd_to_base_rate_ppm: int,
    minimum_buffer_base_cents: int,
    unmanaged_position_stress_base_cents: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reserve every minimum, then distribute only resource-safe residual packages."""

    normalized = sorted(
        (normalize_package_sleeve(sleeve) for sleeve in sleeves),
        key=lambda sleeve: str(sleeve["sleeve_id"]),
    )
    if not normalized or len({row["sleeve_id"] for row in normalized}) != len(normalized):
        raise ValueError("package sleeve identities are empty or duplicated")
    numeric = {
        "settled_cash_usd_cents": _positive_int(
            settled_cash_usd_cents, name="settled cash"
        ),
        "available_funds_base_cents": _positive_int(
            available_funds_base_cents, name="available funds"
        ),
        "excess_liquidity_base_cents": _positive_int(
            excess_liquidity_base_cents, name="excess liquidity"
        ),
        "usd_to_base_rate_ppm": _positive_int(
            usd_to_base_rate_ppm, name="USD/base rate"
        ),
        "minimum_buffer_base_cents": _nonnegative_int(
            minimum_buffer_base_cents, name="minimum account buffer"
        ),
        "unmanaged_position_stress_base_cents": _nonnegative_int(
            unmanaged_position_stress_base_cents,
            name="unmanaged-position stress",
        ),
    }
    indexes = {
        str(sleeve["sleeve_id"]): next(
            index
            for index, package in enumerate(sleeve["package_ladder"])
            if package["package_id"] == sleeve["minimum_package_id"]
        )
        for sleeve in normalized
    }
    minimum_capacity = _capacity(normalized, indexes, **numeric)
    if not _fits(minimum_capacity):
        raise ValueError("minimum executable packages exceed account capacity")

    upgrades = {str(sleeve["sleeve_id"]): 0 for sleeve in normalized}
    while True:
        candidates = []
        for sleeve in normalized:
            sleeve_id = str(sleeve["sleeve_id"])
            next_index = indexes[sleeve_id] + 1
            weight = int(sleeve["residual_weight_bps"])
            if next_index >= len(sleeve["package_ladder"]) or weight <= 0:
                continue
            proposed = {**indexes, sleeve_id: next_index}
            capacity = _capacity(normalized, proposed, **numeric)
            if _fits(capacity):
                candidates.append(
                    (Fraction(upgrades[sleeve_id], weight), sleeve_id, capacity)
                )
        if not candidates:
            break
        _score, sleeve_id, _capacity_after = min(candidates)
        indexes[sleeve_id] += 1
        upgrades[sleeve_id] += 1

    capacity = _capacity(normalized, indexes, **numeric)
    allocated = []
    for sleeve in normalized:
        sleeve_id = str(sleeve["sleeve_id"])
        allocated.append(
            {
                **sleeve,
                "allocated_package_id": sleeve["package_ladder"][indexes[sleeve_id]][
                    "package_id"
                ],
            }
        )
    return allocated, {
        "method": PACKAGE_ALLOCATION_METHOD,
        "minimum_packages_fit": True,
        "upgrades_by_sleeve": upgrades,
        "capacity": capacity,
    }


def allocate_first_admitter_packages(
    sleeves: Sequence[Mapping[str, object]],
    *,
    settled_cash_usd_cents: int,
    available_funds_base_cents: int,
    excess_liquidity_base_cents: int,
    usd_to_base_rate_ppm: int,
    minimum_buffer_base_cents: int,
    unmanaged_position_stress_base_cents: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Bind each selected minimum independently; reserve only at admission."""

    normalized = sorted(
        (normalize_package_sleeve(sleeve) for sleeve in sleeves),
        key=lambda sleeve: str(sleeve["sleeve_id"]),
    )
    if not normalized or len({row["sleeve_id"] for row in normalized}) != len(
        normalized
    ):
        raise ValueError("package sleeve identities are empty or duplicated")
    numeric = {
        "settled_cash_usd_cents": _positive_int(
            settled_cash_usd_cents, name="settled cash"
        ),
        "available_funds_base_cents": _positive_int(
            available_funds_base_cents, name="available funds"
        ),
        "excess_liquidity_base_cents": _positive_int(
            excess_liquidity_base_cents, name="excess liquidity"
        ),
        "usd_to_base_rate_ppm": _positive_int(
            usd_to_base_rate_ppm, name="USD/base rate"
        ),
        "minimum_buffer_base_cents": _nonnegative_int(
            minimum_buffer_base_cents, name="minimum account buffer"
        ),
        "unmanaged_position_stress_base_cents": _nonnegative_int(
            unmanaged_position_stress_base_cents,
            name="unmanaged-position stress",
        ),
    }
    capacities = {}
    allocated = []
    for sleeve in normalized:
        sleeve_id = str(sleeve["sleeve_id"])
        index = next(
            index
            for index, package in enumerate(sleeve["package_ladder"])
            if package["package_id"] == sleeve["minimum_package_id"]
        )
        capacity = _capacity([sleeve], {sleeve_id: index}, **numeric)
        if not _fits(capacity):
            raise ValueError(
                f"minimum executable package exceeds account capacity: {sleeve_id}"
            )
        capacities[sleeve_id] = capacity
        allocated.append(
            {
                **sleeve,
                "allocated_package_id": sleeve["minimum_package_id"],
            }
        )
    return allocated, {
        "method": PACKAGE_FIRST_ADMITTER_METHOD,
        "individual_minimum_packages_fit": True,
        "aggregate_minimum_packages_promised": False,
        "individual_capacity": capacities,
    }


def package_for_sleeve(
    sleeve: Mapping[str, object], *, allocated: bool
) -> Mapping[str, object]:
    package_id = sleeve[
        "allocated_package_id" if allocated else "minimum_package_id"
    ]
    return next(
        package
        for package in sleeve["package_ladder"]
        if package["package_id"] == package_id
    )


def publish_immutable_live_selection(
    repository_root: Path,
    selection: Mapping[str, object],
) -> tuple[str, str]:
    """Publish one content-addressed run before an atomic capital-plan switch."""

    run_id = str(selection.get("selection_id") or "")
    strategy_id = str(selection.get("strategy_version") or "")
    if (
        len(run_id) != 64
        or any(character not in "0123456789abcdef" for character in run_id)
        or not strategy_id
    ):
        raise ValueError("immutable selected-run identity is invalid")
    relative = IMMUTABLE_SELECTION_DIRECTORY / f"{run_id}.json"
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("immutable selected-run path escaped the repository")
    payload = json.dumps(
        dict(selection), allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable selected-run content changed")
    else:
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
    return relative.as_posix(), hashlib.sha256(payload).hexdigest()


def load_allocated_live_selection(
    plan: Mapping[str, object],
    *,
    sleeve_id: str,
    repository_root: Path,
) -> tuple[dict[str, object], Path, str]:
    """Resolve one active immutable run exclusively through the account plan."""

    if plan.get("schema") != "live.capital-plan.v3":
        raise ValueError("allocated selection requires a v3 capital plan")
    sleeves = plan.get("sleeves")
    matches = [
        sleeve
        for sleeve in sleeves
        if isinstance(sleeve, Mapping) and sleeve.get("sleeve_id") == sleeve_id
    ] if isinstance(sleeves, Sequence) and not isinstance(sleeves, (str, bytes)) else []
    if len(matches) != 1:
        raise ValueError("allocated capital sleeve is missing or duplicated")
    sleeve = matches[0]
    relative = Path(str(sleeve.get("selection_path") or ""))
    root = repository_root.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("allocated selected-run path is invalid")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sleeve.get("selection_file_sha256"):
        raise ValueError("allocated selected-run file identity changed")
    value = json.loads(payload)
    if (
        not isinstance(value, Mapping)
        or value.get("selection_id") != sleeve.get("run_id")
        or value.get("strategy_version") != sleeve.get("strategy_id")
    ):
        raise ValueError("allocated selected-run semantic identity changed")
    return dict(value), path, digest


def live_package_entry_capacity(
    plan: Mapping[str, object],
    *,
    sleeve_id: str,
    resource_state: Mapping[str, object] | None,
    available_cash_usd_cents: int,
    candidate_cash_debit_usd_cents: int,
) -> tuple[dict[str, object], list[str]]:
    """Re-evaluate all promised packages from one fresh broker account snapshot."""

    reasons: list[str] = []
    if not isinstance(resource_state, Mapping):
        return {}, ["account_resource_state_missing"]
    sleeves = plan["sleeves"]
    candidate = next(row for row in sleeves if row["sleeve_id"] == sleeve_id)
    allocated = package_for_sleeve(candidate, allocated=True)
    first_admitter = (
        plan.get("constraints", {}).get("entry_capacity_policy")
        == PACKAGE_FIRST_ADMITTER_METHOD
    )
    positions = resource_state.get("account_positions")
    orders = resource_state.get("account_open_orders")
    try:
        if (
            not isinstance(positions, Sequence)
            or isinstance(positions, (str, bytes))
            or not isinstance(orders, Sequence)
            or isinstance(orders, (str, bytes))
        ):
            raise ValueError
        base_currency = str(resource_state["base_currency"] or "").upper()
        if base_currency != plan["account"]["base_currency"]:
            reasons.append("capital_base_currency_mismatch")
        available = _nonnegative_int(
            resource_state["available_funds_base_cents"],
            name="available funds",
        )
        excess = _nonnegative_int(
            resource_state["excess_liquidity_base_cents"],
            name="excess liquidity",
        )
        fx_ppm = _positive_int(
            resource_state["usd_to_base_rate_ppm"], name="USD/base rate"
        )
        initial_change = _nonnegative_int(
            resource_state.get("candidate_initial_margin_base_cents", 0),
            name="candidate initial margin",
        )
        maintenance_change = _nonnegative_int(
            resource_state.get("candidate_maintenance_margin_base_cents", 0),
            name="candidate maintenance margin",
        )
        if candidate_cash_debit_usd_cents > int(allocated["cash_debit_usd_cents"]):
            reasons.append("allocated_package_cash_exceeded")
        if initial_change > int(allocated["initial_margin_base_cents"]):
            reasons.append("allocated_package_initial_margin_exceeded")
        if maintenance_change > int(allocated["maintenance_margin_base_cents"]):
            reasons.append("allocated_package_maintenance_margin_exceeded")

        active: set[str] = set()
        unmanaged_stress = 0
        for row in positions:
            if not isinstance(row, Mapping):
                raise ValueError
            symbol = str(row.get("symbol") or "").strip().upper()
            quantity = Decimal(str(row.get("quantity") or 0))
            if not symbol or not quantity.is_finite():
                raise ValueError
            if abs(quantity) <= Decimal("1e-9"):
                continue
            owners = [
                str(sleeve["sleeve_id"])
                for sleeve in sleeves
                if symbol in sleeve["position_symbols"]
            ]
            if len(owners) > 1:
                reasons.append("position_symbol_has_multiple_capital_owners")
            elif owners:
                active.add(owners[0])
            else:
                unmanaged_stress += _nonnegative_int(
                    row.get("market_value_base_cents"),
                    name="unmanaged position market value",
                )
        pending: set[str] = set()
        if not first_admitter and any(
            isinstance(row, Mapping)
            and abs(Decimal(str(row.get("quantity") or 0))) > Decimal("1e-9")
            for row in orders
        ):
            reasons.append("open_order_blocks_portfolio_capacity_proof")
        elif first_admitter:
            for row in orders:
                if not isinstance(row, Mapping):
                    raise ValueError
                symbol = str(row.get("symbol") or "").strip().upper()
                quantity = Decimal(str(row.get("quantity") or 0))
                if not symbol or not quantity.is_finite():
                    raise ValueError
                if abs(quantity) <= Decimal("1e-9"):
                    continue
                owners = [
                    str(sleeve["sleeve_id"])
                    for sleeve in sleeves
                    if symbol in sleeve["position_symbols"]
                ]
                if len(owners) != 1:
                    reasons.append("open_order_has_no_unique_capital_owner")
                elif owners[0] in active or owners[0] == sleeve_id:
                    reasons.append("open_order_blocks_portfolio_capacity_proof")
                else:
                    pending.add(owners[0])
        if sleeve_id in active:
            reasons.append("candidate_sleeve_already_active")

        cash_reserved = initial_reserved = maintenance_reserved = stress_reserved = 0
        reserved = (
            {sleeve_id, *active, *pending}
            if first_admitter
            else {str(sleeve["sleeve_id"]) for sleeve in sleeves}
        )
        for sleeve in sleeves:
            current = str(sleeve["sleeve_id"])
            if current not in reserved:
                continue
            package = package_for_sleeve(sleeve, allocated=True)
            stress_reserved += _ceil_ratio(
                int(package["stressed_loss_usd_cents"])
                * fx_ppm
                * int(package["fx_stress_bps"]),
                1_000_000 * 10_000,
            )
            if current in active:
                continue
            cash_reserved += int(package["cash_debit_usd_cents"])
            initial_reserved += int(package["initial_margin_base_cents"])
            maintenance_reserved += int(package["maintenance_margin_base_cents"])
        cash_base = _ceil_ratio(cash_reserved * fx_ppm, 1_000_000)
        floor = int(plan["resources"]["minimum_buffer_base_cents"])
        cash_remaining = available_cash_usd_cents - cash_reserved
        post_available = available - cash_base - initial_reserved
        post_stress = excess - maintenance_reserved - stress_reserved - unmanaged_stress
        if cash_remaining < 0:
            reasons.append("insufficient_live_cash_for_promised_packages")
        if post_available < floor:
            reasons.append("post_reservation_available_funds_below_floor")
        if post_stress < floor:
            reasons.append("post_stress_excess_liquidity_below_floor")
        allocation = {
            "capital_kind": candidate["capital_kind"],
            "allocated_package_id": candidate["allocated_package_id"],
            "active_sleeves": sorted(active),
            "pending_sleeves": sorted(pending),
            "entry_capacity_policy": (
                PACKAGE_FIRST_ADMITTER_METHOD
                if first_admitter
                else PACKAGE_ALLOCATION_METHOD
            ),
            "cash_reserved_usd_cents": cash_reserved,
            "initial_margin_reserved_base_cents": initial_reserved,
            "maintenance_margin_reserved_base_cents": maintenance_reserved,
            "stressed_loss_reserved_base_cents": stress_reserved,
            "unmanaged_position_stress_base_cents": unmanaged_stress,
            "post_reservation_cash_usd_cents": cash_remaining,
            "post_reservation_available_funds_base_cents": post_available,
            "post_stress_excess_liquidity_base_cents": post_stress,
            "minimum_buffer_base_cents": floor,
        }
    except (InvalidOperation, KeyError, StopIteration, TypeError, ValueError):
        return {}, sorted(set([*reasons, "invalid_portfolio_resource_state"]))
    return allocation, sorted(set(reasons))
