from __future__ import annotations

from collections.abc import Iterable


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
