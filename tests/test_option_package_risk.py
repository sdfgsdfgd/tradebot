from __future__ import annotations

import pytest

import tradebot.option_package as option_package_module


def _package_risk_owners():
    risk_type = getattr(option_package_module, "OptionPackageRisk", None)
    assert risk_type is not None, "tradebot.option_package.OptionPackageRisk is missing"
    evaluate = getattr(option_package_module, "option_package_risk", None)
    assert callable(evaluate), "tradebot.option_package.option_package_risk is missing"
    return risk_type, evaluate


def test_option_package_risk_prices_credit_and_debit_verticals_in_dollars() -> None:
    risk_type, evaluate = _package_risk_owners()

    credit = evaluate(
        [
            ("SELL", "P", 100.0, 1, "20260209", 100.0),
            ("BUY", "PUT", 95.0, 1, "20260209", 100.0),
        ],
        debit_value=-1.00,
        quantity=1,
    )
    assert isinstance(credit, risk_type)
    assert credit.as_payload() == {
        "structure": "vertical_credit",
        "right": "PUT",
        "expiry": "20260209",
        "width": 5.0,
        "debit_value": -1.0,
        "multiplier": 100.0,
        "quantity": 1,
        "max_loss": 400.0,
    }

    debit = evaluate(
        [
            ("BUY", "CALL", 95.0, 1, "20260209", 100.0),
            ("SELL", "C", 100.0, 1, "20260209", 100.0),
        ],
        debit_value=1.25,
        quantity=2,
    )
    assert isinstance(debit, risk_type)
    assert debit.as_payload() == {
        "structure": "vertical_debit",
        "right": "CALL",
        "expiry": "20260209",
        "width": 5.0,
        "debit_value": 1.25,
        "multiplier": 100.0,
        "quantity": 2,
        "max_loss": 250.0,
    }


@pytest.mark.parametrize(
    ("rows", "debit_value", "quantity"),
    [
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", None),
                ("BUY", "PUT", 95.0, 1, "20260209", None),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("BUY", "PUT", 95.0, 1, "20260210", 100.0),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("BUY", "CALL", 95.0, 1, "20260209", 100.0),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 2, "20260209", 100.0),
                ("BUY", "PUT", 95.0, 1, "20260209", 100.0),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("SELL", "PUT", 95.0, 1, "20260209", 100.0),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("BUY", "PUT", 100.0, 1, "20260209", 100.0),
            ],
            -1.0,
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("BUY", "PUT", 95.0, 1, "20260209", 100.0),
            ],
            float("nan"),
            1,
        ),
        (
            [
                ("SELL", "PUT", 100.0, 1, "20260209", 100.0),
                ("BUY", "PUT", 95.0, 1, "20260209", 100.0),
            ],
            -1.0,
            0,
        ),
    ],
)
def test_option_package_risk_rejects_incomplete_or_non_vertical_identity(
    rows: list[tuple[str, str, float, int, str, float | None]],
    debit_value: float,
    quantity: int,
) -> None:
    _risk_type, evaluate = _package_risk_owners()
    assert evaluate(rows, debit_value=debit_value, quantity=quantity) is None
