from __future__ import annotations

import importlib
import math

import pytest


def _reservation_owners():
    try:
        module = importlib.import_module("tradebot.order_reservation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"tradebot.order_reservation is missing: {exc}")

    reservation_type = getattr(module, "OrderReservation", None)
    summary_type = getattr(module, "OrderReservationSummary", None)
    summarize = getattr(module, "summarize_order_reservations", None)

    assert isinstance(
        reservation_type,
        type,
    ), "tradebot.order_reservation.OrderReservation is missing"
    assert isinstance(
        summary_type,
        type,
    ), "tradebot.order_reservation.OrderReservationSummary is missing"
    assert callable(
        summarize,
    ), "tradebot.order_reservation.summarize_order_reservations is missing"
    return reservation_type, summary_type, summarize


def test_xsp_reservation_sums_only_active_local_defined_risk_packages() -> None:
    reservation_type, summary_type, summarize = _reservation_owners()

    reservations = [
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="STAGED",
            max_loss=400.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="xsp",
            sec_type="bag",
            structure="VERTICAL_CREDIT",
            status="working",
            max_loss=250.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="CANCELING",
            max_loss=300.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="FILLED",
            max_loss=999.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="SPX",
            sec_type="BAG",
            structure="vertical_credit",
            status="STAGED",
            max_loss=999.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="OPT",
            structure="vertical_credit",
            status="STAGED",
            max_loss=999.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_debit",
            status="STAGED",
            max_loss=999.0,
        ),
        reservation_type(
            account="OTHER",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="STAGED",
            max_loss=999.0,
        ),
    ]

    summary = summarize(reservations, account="DU2200")

    assert isinstance(summary, summary_type)
    assert summary.as_payload() == {
        "account": "DU2200",
        "active_count": 3,
        "unknown_active_count": 0,
        "reserved_max_loss": 950.0,
        "complete": True,
        "reason": "reservation_complete",
    }


def test_xsp_reservation_fails_closed_when_active_package_risk_is_unknown() -> None:
    reservation_type, summary_type, summarize = _reservation_owners()

    reservations = [
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="STAGED",
            max_loss=400.0,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="WORKING",
            max_loss=None,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="CANCELING",
            max_loss=math.nan,
        ),
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status="FILLED",
            max_loss=None,
        ),
        reservation_type(
            account="DU2200",
            product_domain="SPX",
            sec_type="BAG",
            structure="vertical_credit",
            status="STAGED",
            max_loss=None,
        ),
    ]

    summary = summarize(reservations, account="DU2200")

    assert isinstance(summary, summary_type)
    assert summary.as_payload() == {
        "account": "DU2200",
        "active_count": 3,
        "unknown_active_count": 2,
        "reserved_max_loss": None,
        "complete": False,
        "reason": "active_risk_unknown",
    }


def test_xsp_reservation_releases_all_local_terminal_states() -> None:
    reservation_type, summary_type, summarize = _reservation_owners()

    reservations = [
        reservation_type(
            account="DU2200",
            product_domain="XSP",
            sec_type="BAG",
            structure="vertical_credit",
            status=status,
            max_loss=400.0,
        )
        for status in (
            "BLOCKED",
            "ERROR",
            "FILLED",
            "CANCELLED",
            "INACTIVE",
            "DONE",
        )
    ]

    summary = summarize(reservations, account="DU2200")

    assert isinstance(summary, summary_type)
    assert summary.as_payload() == {
        "account": "DU2200",
        "active_count": 0,
        "unknown_active_count": 0,
        "reserved_max_loss": 0.0,
        "complete": True,
        "reason": "reservation_complete",
    }


def _capacity_owners():
    module = importlib.import_module("tradebot.order_reservation")
    request_type = getattr(module, "OrderReservationCapacityRequest", None)
    decision_type = getattr(module, "OrderReservationCapacityDecision", None)
    evaluate = getattr(module, "evaluate_order_reservation_capacity", None)

    assert isinstance(
        request_type,
        type,
    ), "tradebot.order_reservation.OrderReservationCapacityRequest is missing"
    assert isinstance(
        decision_type,
        type,
    ), "tradebot.order_reservation.OrderReservationCapacityDecision is missing"
    assert callable(
        evaluate,
    ), "tradebot.order_reservation.evaluate_order_reservation_capacity is missing"
    return request_type, decision_type, evaluate


def _complete_reservation_summary(*, reserved_max_loss: float, account: str = "DU2200"):
    _reservation_type, summary_type, _summarize = _reservation_owners()
    return summary_type(
        account=account,
        active_count=2,
        unknown_active_count=0,
        reserved_max_loss=reserved_max_loss,
        complete=True,
        reason="reservation_complete",
    )


def test_xsp_reservation_capacity_allows_candidate_within_explicit_capacity() -> None:
    request_type, decision_type, evaluate = _capacity_owners()
    summary = _complete_reservation_summary(reserved_max_loss=650.0)
    request = request_type(
        account="DU2200",
        product_domain="XSP",
        sec_type="BAG",
        structure="vertical_credit",
        candidate_max_loss=400.0,
        available_capacity=2200.0,
    )

    decision = evaluate(request, summary)

    assert isinstance(decision, decision_type)
    assert decision.as_payload() == {
        "allow": True,
        "reason": "capacity_available",
        "account": "DU2200",
        "product_domain": "XSP",
        "sec_type": "BAG",
        "structure": "vertical_credit",
        "reserved_max_loss": 650.0,
        "candidate_max_loss": 400.0,
        "total_max_loss": 1050.0,
        "available_capacity": 2200.0,
        "remaining_capacity": 1150.0,
    }


def test_xsp_reservation_capacity_blocks_candidate_that_exceeds_explicit_capacity() -> None:
    request_type, decision_type, evaluate = _capacity_owners()
    summary = _complete_reservation_summary(reserved_max_loss=1900.0)
    request = request_type(
        account="DU2200",
        product_domain="XSP",
        sec_type="BAG",
        structure="vertical_credit",
        candidate_max_loss=400.0,
        available_capacity=2200.0,
    )

    decision = evaluate(request, summary)

    assert isinstance(decision, decision_type)
    assert decision.as_payload() == {
        "allow": False,
        "reason": "capacity_exceeded",
        "account": "DU2200",
        "product_domain": "XSP",
        "sec_type": "BAG",
        "structure": "vertical_credit",
        "reserved_max_loss": 1900.0,
        "candidate_max_loss": 400.0,
        "total_max_loss": 2300.0,
        "available_capacity": 2200.0,
        "remaining_capacity": -100.0,
    }


def test_xsp_reservation_capacity_fails_closed_for_incomplete_or_mismatched_facts() -> None:
    request_type, decision_type, evaluate = _capacity_owners()
    _reservation_type, summary_type, _summarize = _reservation_owners()
    complete = _complete_reservation_summary(reserved_max_loss=650.0)
    incomplete = summary_type(
        account="DU2200",
        active_count=2,
        unknown_active_count=1,
        reserved_max_loss=None,
        complete=False,
        reason="active_risk_unknown",
    )

    cases = [
        (
            request_type(
                account="DU2200",
                product_domain="XSP",
                sec_type="BAG",
                structure="vertical_credit",
                candidate_max_loss=400.0,
                available_capacity=2200.0,
            ),
            incomplete,
            "reservation_incomplete",
        ),
        (
            request_type(
                account="DU2200",
                product_domain="XSP",
                sec_type="BAG",
                structure="vertical_credit",
                candidate_max_loss=None,
                available_capacity=2200.0,
            ),
            complete,
            "candidate_risk_unknown",
        ),
        (
            request_type(
                account="DU2200",
                product_domain="XSP",
                sec_type="BAG",
                structure="vertical_credit",
                candidate_max_loss=400.0,
                available_capacity=None,
            ),
            complete,
            "capacity_unavailable",
        ),
        (
            request_type(
                account="DU2200",
                product_domain="XSP",
                sec_type="BAG",
                structure="vertical_credit",
                candidate_max_loss=400.0,
                available_capacity=2200.0,
            ),
            _complete_reservation_summary(
                reserved_max_loss=650.0,
                account="OTHER",
            ),
            "account_mismatch",
        ),
        (
            request_type(
                account="DU2200",
                product_domain="SPX",
                sec_type="BAG",
                structure="vertical_credit",
                candidate_max_loss=400.0,
                available_capacity=2200.0,
            ),
            complete,
            "product_policy_unavailable",
        ),
    ]

    for request, summary, expected_reason in cases:
        decision = evaluate(request, summary)
        assert isinstance(decision, decision_type)
        assert decision.allow is False
        assert decision.reason == expected_reason
