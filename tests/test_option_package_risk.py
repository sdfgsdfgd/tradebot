from __future__ import annotations

from types import SimpleNamespace

import pytest

import tradebot.option_package as option_package_module
from tradebot.option_package import (
    OptionPackage,
    ResolvedOptionLeg,
    option_package_risk,
    option_product_facts,
)


def _package_risk_owners():
    risk_type = getattr(option_package_module, "OptionPackageRisk", None)
    assert risk_type is not None, "tradebot.option_package.OptionPackageRisk is missing"
    evaluate = getattr(option_package_module, "option_package_risk", None)
    assert callable(evaluate), "tradebot.option_package.option_package_risk is missing"
    return risk_type, evaluate


def _package(
    rows: list[tuple[str, str, float, int, str]],
    *,
    debit_value: float,
    quantity: int = 1,
    multiplier: float = 100.0,
) -> OptionPackage:
    return OptionPackage(
        product=option_product_facts("XSP", multiplier=multiplier),
        legs=tuple(ResolvedOptionLeg(*row) for row in rows),
        quantity=quantity,
        debit_value=debit_value,
    )


def test_option_package_risk_prices_credit_and_debit_verticals_in_dollars() -> None:
    risk_type, evaluate = _package_risk_owners()

    credit = evaluate(
        _package(
            [
                ("SELL", "P", 100.0, 1, "20260209"),
                ("BUY", "PUT", 95.0, 1, "20260209"),
            ],
            debit_value=-1.00,
        )
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
        "max_profit": 100.0,
    }

    debit = evaluate(
        _package(
            [
                ("BUY", "CALL", 95.0, 1, "20260209"),
                ("SELL", "C", 100.0, 1, "20260209"),
            ],
            debit_value=1.25,
            quantity=2,
        )
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
        "max_profit": 750.0,
    }


def test_option_package_risk_prices_defined_risk_multi_leg_packages() -> None:
    risk_type, evaluate = _package_risk_owners()
    condor = evaluate(
        _package(
            [
                ("BUY", "PUT", 90.0, 1, "20260209"),
                ("SELL", "PUT", 95.0, 1, "20260209"),
                ("SELL", "CALL", 105.0, 1, "20260209"),
                ("BUY", "CALL", 110.0, 1, "20260209"),
            ],
            debit_value=-2.0,
            quantity=2,
        )
    )
    assert isinstance(condor, risk_type)
    assert condor.as_payload() == {
        "structure": "iron_condor_credit",
        "right": "MIXED",
        "expiry": "20260209",
        "width": 5.0,
        "debit_value": -2.0,
        "multiplier": 100.0,
        "quantity": 2,
        "max_loss": 600.0,
        "max_profit": 400.0,
    }

    butterfly = evaluate(
        _package(
            [
                ("BUY", "CALL", 90.0, 1, "20260209"),
                ("SELL", "CALL", 100.0, 2, "20260209"),
                ("BUY", "CALL", 110.0, 1, "20260209"),
            ],
            debit_value=2.0,
        )
    )
    assert isinstance(butterfly, risk_type)
    assert butterfly.structure == "butterfly_debit"
    assert butterfly.max_loss == pytest.approx(200.0)
    assert butterfly.max_profit == pytest.approx(800.0)


def test_option_package_risk_rejects_mixed_expiry_and_unbounded_loss() -> None:
    _risk_type, evaluate = _package_risk_owners()
    assert evaluate(
        _package(
            [
                ("SELL", "PUT", 100.0, 1, "20260209"),
                ("BUY", "PUT", 95.0, 1, "20260210"),
            ],
            debit_value=-1.0,
        )
    ) is None
    assert evaluate(
        _package(
            [("SELL", "CALL", 100.0, 1, "20260209")],
            debit_value=-2.0,
        )
    ) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("security_type", "FUT"),
        ("multiplier", 0),
        ("pricing_model", "guess"),
    ],
)
def test_option_product_facts_reject_incomplete_economics(field: str, value: object) -> None:
    kwargs = {
        "underlying_symbol": "XSP",
        "security_type": "OPT",
        "exchange": "CBOE",
        "currency": "USD",
        "multiplier": 100,
        "pricing_model": "black_scholes",
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        option_package_module.OptionProductFacts(**kwargs)


def test_option_product_facts_encode_verified_offline_units() -> None:
    assert option_product_facts("XSP").multiplier == 100.0
    assert option_product_facts("MNQ").multiplier == 2.0
    assert option_product_facts("MNQ").pricing_model == "black_76"
    assert option_product_facts("MCL").exchange == "NYMEX"
    assert option_product_facts("MCL").multiplier == 100.0


@pytest.mark.parametrize(
    ("symbol", "exchange", "multiplier"),
    [
        ("XSP", "CBOE", 100.0),
        ("MNQ", "CME", 2.0),
        ("MCL", "NYMEX", 100.0),
    ],
)
def test_offline_backtest_meta_uses_canonical_option_product_facts(
    symbol: str,
    exchange: str,
    multiplier: float,
) -> None:
    from tradebot.backtest.engine import _resolve_backtest_contract_meta

    cfg = SimpleNamespace(
        strategy=SimpleNamespace(
            symbol=symbol,
            instrument="options",
        ),
        backtest=SimpleNamespace(offline=True),
    )
    meta = _resolve_backtest_contract_meta(data=object(), cfg=cfg)
    assert (meta.exchange, meta.multiplier) == (exchange, multiplier)


def test_nonstandard_ratio_spread_is_not_mislabeled_as_vertical() -> None:
    risk = option_package_risk(
        _package(
            [
                ("SELL", "PUT", 100.0, 1, "20260209"),
                ("BUY", "PUT", 95.0, 2, "20260209"),
            ],
            debit_value=1.0,
        )
    )
    assert risk is not None
    assert risk.structure == "defined_risk_combo"
