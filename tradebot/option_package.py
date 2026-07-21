from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class OptionPackageRisk:
    structure: str
    right: str
    expiry: str
    width: float
    debit_value: float
    multiplier: float
    quantity: int
    max_loss: float

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric != float(parsed) or parsed <= 0:
        return None
    return parsed


def _option_right(value: object) -> str | None:
    cleaned = str(value or "").strip().upper()
    if cleaned in {"P", "PUT"}:
        return "PUT"
    if cleaned in {"C", "CALL"}:
        return "CALL"
    return None


def option_package_debit_value(
    rows: Iterable[tuple[str, int, float | None]],
) -> float | None:
    """Aggregate option-leg values in debit units: BUY positive, SELL negative."""
    total = 0.0
    for action, ratio, value in rows:
        if value is None:
            return None

        cleaned_action = str(action or "").strip().upper()
        if cleaned_action == "BUY":
            sign = 1.0
        elif cleaned_action == "SELL":
            sign = -1.0
        else:
            raise ValueError(f"unsupported option leg action: {action!r}")

        total += sign * float(value) * int(ratio)

    return float(total)


def option_package_risk(
    rows: Iterable[tuple[str, str, float, int, str, float | None]],
    *,
    debit_value: float,
    quantity: int,
) -> OptionPackageRisk | None:
    """Return defined-risk vertical economics in account-currency units."""
    try:
        materialized = list(rows)
    except TypeError:
        return None
    if len(materialized) != 2:
        return None

    parsed_rows: list[tuple[str, str, float, int, str, float]] = []
    for row in materialized:
        try:
            action_raw, right_raw, strike_raw, ratio_raw, expiry_raw, multiplier_raw = row
        except (TypeError, ValueError):
            return None

        action = str(action_raw or "").strip().upper()
        right = _option_right(right_raw)
        strike = _finite_float(strike_raw)
        ratio = _positive_int(ratio_raw)
        expiry = str(expiry_raw or "").strip()
        multiplier = _finite_float(multiplier_raw)
        if (
            action not in {"BUY", "SELL"}
            or right is None
            or strike is None
            or strike <= 0
            or ratio is None
            or not expiry
            or multiplier is None
            or multiplier <= 0
        ):
            return None
        parsed_rows.append((action, right, strike, ratio, expiry, multiplier))

    first, second = parsed_rows
    if {first[0], second[0]} != {"BUY", "SELL"}:
        return None
    if first[1] != second[1] or first[3] != second[3] or first[4] != second[4]:
        return None
    if not math.isclose(first[5], second[5], rel_tol=0.0, abs_tol=1e-12):
        return None

    width = abs(first[2] - second[2])
    debit = _finite_float(debit_value)
    package_quantity = _positive_int(quantity)
    if width <= 0 or debit is None or debit == 0 or package_quantity is None:
        return None

    ratio = first[3]
    multiplier = first[5]
    if debit > 0:
        structure = "vertical_debit"
        max_loss_units = debit
    else:
        structure = "vertical_credit"
        max_loss_units = (width * ratio) + debit
        if max_loss_units <= 0:
            return None

    max_loss = max_loss_units * multiplier * package_quantity
    if not math.isfinite(max_loss) or max_loss <= 0:
        return None

    return OptionPackageRisk(
        structure=structure,
        right=first[1],
        expiry=first[4],
        width=float(width),
        debit_value=float(debit),
        multiplier=float(multiplier),
        quantity=int(package_quantity),
        max_loss=float(max_loss),
    )
