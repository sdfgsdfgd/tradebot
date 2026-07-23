from __future__ import annotations

import importlib

import pytest


def _policy_module():
    try:
        return importlib.import_module("tradebot.order_admission")
    except ModuleNotFoundError as exc:
        if exc.name == "tradebot.order_admission":
            pytest.fail("tradebot.order_admission is missing")
        raise


def _require_owner(module, name: str):
    owner = getattr(module, name, None)
    assert owner is not None, f"tradebot.order_admission.{name} is missing"
    return owner


def _complete_xsp_request(module):
    leg_type = _require_owner(module, "OrderAdmissionLeg")
    request_type = _require_owner(module, "OrderAdmissionRequest")
    return request_type(
        account="DU2200",
        intent="enter",
        product_domain="XSP",
        structure="vertical_credit",
        sec_type="BAG",
        symbol="XSP",
        currency="USD",
        exchange="SMART",
        action="BUY",
        quantity=1,
        limit_price=-1.00,
        max_loss=400.00,
        legs=(
            leg_type(con_id=1001, ratio=1, action="SELL", exchange="SMART"),
            leg_type(con_id=1002, ratio=1, action="BUY", exchange="SMART"),
        ),
    )


def _complete_preview_facts(module, **overrides):
    facts_type = _require_owner(module, "OrderAdmissionFacts")
    values = {
        "status": "PreSubmitted",
        "init_margin_before": 2200.00,
        "init_margin_change": 500.00,
        "init_margin_after": 1700.00,
        "maintenance_margin_before": 1800.00,
        "maintenance_margin_change": 400.00,
        "maintenance_margin_after": 1400.00,
        "equity_with_loan_before": 2200.00,
        "equity_with_loan_change": -2.50,
        "equity_with_loan_after": 2197.50,
        "commission": 2.50,
        "min_commission": 1.00,
        "max_commission": 3.00,
        "commission_currency": "USD",
        "warning_text": None,
    }
    values.update(overrides)
    return facts_type(**values)


def test_complete_xsp_credit_bag_is_admitted_from_empirical_preview_without_static_floor() -> None:
    module = _policy_module()
    decision_type = _require_owner(module, "OrderAdmissionDecision")
    evaluate = _require_owner(module, "evaluate_order_admission")
    assert callable(evaluate)

    request = _complete_xsp_request(module)
    facts = _complete_preview_facts(module)
    decision = evaluate(request, facts)

    assert isinstance(decision, decision_type)
    assert decision.allow is True
    assert decision.reason == "broker_preview_admitted"
    assert not hasattr(decision, "trade")

    payload = decision.as_payload()
    assert payload["allow"] is True
    assert payload["reason"] == "broker_preview_admitted"
    assert payload["trace"] == {
        "account": "DU2200",
        "intent": "enter",
        "product_domain": "XSP",
        "structure": "vertical_credit",
        "sec_type": "BAG",
        "symbol": "XSP",
        "currency": "USD",
        "exchange": "SMART",
        "action": "BUY",
        "quantity": 1,
        "limit_price": -1.00,
        "max_loss": 400.00,
        "status": "PreSubmitted",
        "init_margin_change": 500.00,
        "init_margin_after": 1700.00,
        "equity_with_loan_after": 2197.50,
        "commission": 2.50,
        "commission_currency": "USD",
        "warning_text": "",
    }


def test_complete_xsp_vertical_debit_exit_is_admitted_from_empirical_preview() -> None:
    module = _policy_module()
    leg_type = _require_owner(module, "OrderAdmissionLeg")
    request_type = _require_owner(module, "OrderAdmissionRequest")
    evaluate = _require_owner(module, "evaluate_order_admission")

    decision = evaluate(
        request_type(
            account="DU2200",
            intent="exit",
            product_domain="XSP",
            structure="vertical_debit",
            sec_type="BAG",
            symbol="XSP",
            currency="USD",
            exchange="SMART",
            action="BUY",
            quantity=1,
            limit_price=0.90,
            max_loss=90.00,
            legs=(
                leg_type(con_id=1001, ratio=1, action="BUY", exchange="SMART"),
                leg_type(con_id=1002, ratio=1, action="SELL", exchange="SMART"),
            ),
        ),
        _complete_preview_facts(
            module,
            init_margin_change=-400.00,
            maintenance_margin_change=-300.00,
        ),
    )

    assert decision.allow is True
    assert decision.reason == "broker_preview_admitted"
    assert decision.as_payload()["trace"]["intent"] == "exit"
    assert decision.as_payload()["trace"]["structure"] == "vertical_debit"


def test_complete_xsp_iron_condor_resize_is_admitted_from_empirical_preview() -> None:
    module = _policy_module()
    leg_type = _require_owner(module, "OrderAdmissionLeg")
    request_type = _require_owner(module, "OrderAdmissionRequest")
    evaluate = _require_owner(module, "evaluate_order_admission")

    decision = evaluate(
        request_type(
            account="DU2200",
            intent="resize",
            product_domain="XSP",
            structure="iron_condor_credit",
            sec_type="BAG",
            symbol="XSP",
            currency="USD",
            exchange="SMART",
            action="BUY",
            quantity=2,
            limit_price=-1.20,
            max_loss=760.00,
            legs=tuple(
                leg_type(con_id=con_id, ratio=1, action=action, exchange="SMART")
                for con_id, action in (
                    (1001, "BUY"),
                    (1002, "SELL"),
                    (1003, "SELL"),
                    (1004, "BUY"),
                )
            ),
        ),
        _complete_preview_facts(module),
    )

    assert decision.allow is True
    assert decision.reason == "broker_preview_admitted"
    assert decision.as_payload()["trace"]["intent"] == "resize"
    assert decision.as_payload()["trace"]["structure"] == "iron_condor_credit"


@pytest.mark.parametrize(
    ("request_changes", "fact_changes", "expected_reason"),
    [
        ({"max_loss": None}, {}, "max_loss_unknown"),
        ({}, {"status": None}, "preview_incomplete"),
        ({}, {"status": "Inactive"}, "broker_status_blocked"),
        ({}, {"status": "PendingSubmit"}, "broker_status_unproven"),
        ({}, {"commission_currency": "CAD"}, "currency_mismatch"),
        ({}, {"equity_with_loan_after": -1.0}, "broker_capacity_exceeded"),
        ({"product_domain": "SPX"}, {}, "identity_mismatch"),
        ({"product_domain": "MCL", "symbol": "MCL", "sec_type": "BAG"}, {}, "product_policy_unavailable"),
    ],
)
def test_order_admission_fails_closed_with_explicit_evidence_reason(
    request_changes: dict[str, object],
    fact_changes: dict[str, object],
    expected_reason: str,
) -> None:
    module = _policy_module()
    request_type = _require_owner(module, "OrderAdmissionRequest")
    evaluate = _require_owner(module, "evaluate_order_admission")

    base_request = _complete_xsp_request(module)
    request_values = {
        "account": base_request.account,
        "intent": base_request.intent,
        "product_domain": base_request.product_domain,
        "structure": base_request.structure,
        "sec_type": base_request.sec_type,
        "symbol": base_request.symbol,
        "currency": base_request.currency,
        "exchange": base_request.exchange,
        "action": base_request.action,
        "quantity": base_request.quantity,
        "limit_price": base_request.limit_price,
        "max_loss": base_request.max_loss,
        "legs": base_request.legs,
    }
    request_values.update(request_changes)

    decision = evaluate(
        request_type(**request_values),
        _complete_preview_facts(module, **fact_changes),
    )

    assert decision.allow is False
    assert decision.reason == expected_reason
    assert decision.as_payload()["trace"]["product_domain"] == request_values["product_domain"]


def test_sparse_ibkr_preview_with_advisory_warning_preserves_bounded_risk_admission() -> None:
    module = _policy_module()
    evaluate = _require_owner(module, "evaluate_order_admission")
    warning = (
        "If your order is not immediately executable, our systems may reject "
        "the order based on its limit price."
    )

    decision = evaluate(
        _complete_xsp_request(module),
        _complete_preview_facts(
            module,
            init_margin_before=None,
            init_margin_change=None,
            init_margin_after=None,
            maintenance_margin_before=None,
            maintenance_margin_change=None,
            maintenance_margin_after=None,
            equity_with_loan_before=None,
            equity_with_loan_change=None,
            equity_with_loan_after=None,
            commission=0.0,
            min_commission=None,
            max_commission=None,
            commission_currency=None,
            warning_text=warning,
        ),
    )

    assert decision.allow is True
    assert decision.reason == "broker_preview_admitted"
    assert decision.trace["status"] == "PreSubmitted"
    assert decision.trace["warning_text"] == warning
