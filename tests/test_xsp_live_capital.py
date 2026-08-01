from __future__ import annotations

from tradebot.live.capital import build_live_capital_plan
from tradebot.research.xsp_live_capital import (
    apply_xsp_live_capital,
    build_xsp_live_capital_plan,
    xsp_pending_buy_has_capital_reservation,
)


RUN_ID = "a" * 64
SELECTION_SHA = "b" * 64


def _selection() -> dict[str, object]:
    return {
        "selection_id": RUN_ID,
        "broker_at_selection": {
            "account_id": "U123",
            "account_type": "CASH",
        },
    }


def _capital_plan() -> dict[str, object]:
    return build_live_capital_plan(
        account_id="U123",
        account_type="CASH",
        currency="USD",
        observed_settled_cash_usd="1318.05",
        managed_capital_usd="900.45034925",
        sleeves=[
            {
                "sleeve_id": "xsp-upro-spxu-rth-cash",
                "strategy_id": "xsp-v3",
                "run_id": RUN_ID,
                "selection_path": "selection.json",
                "selection_file_sha256": SELECTION_SHA,
                "capital_kind": "CASH_DEBIT",
                "weight_bps": 10_000,
            }
        ],
        reserve_reasons=["outside_selected_authority"],
        created_at_utc="2026-08-01T15:00:00+00:00",
    )


def _projected(action: str) -> dict[str, object]:
    return {
        "status": "ACTIONABLE",
        "reason": (
            "buy_post_selection_target"
            if action == "BUY"
            else "sell_incumbent_before_target"
        ),
        "leg": {
            "action": action,
            "symbol": "UPRO" if action == "BUY" else "SPXU",
            "quantity": 6 if action == "BUY" else 23,
            **(
                {"required_settled_cash_usd": 810.75}
                if action == "BUY"
                else {}
            ),
        },
    }


def test_xsp_buy_requires_exact_shared_capital_plan() -> None:
    blocked = apply_xsp_live_capital(
        _projected("BUY"),
        capital_plan=None,
        selection=_selection(),
        selection_file_sha256=SELECTION_SHA,
        available_cash_usd=1_318.05,
    )
    admitted = apply_xsp_live_capital(
        _projected("BUY"),
        capital_plan=_capital_plan(),
        selection=_selection(),
        selection_file_sha256=SELECTION_SHA,
        available_cash_usd=1_318.05,
    )

    assert blocked["status"] == "CAPITAL_HOLD"
    assert blocked["leg"] is None
    assert blocked["blocked_leg"]["action"] == "BUY"
    assert admitted["status"] == "ACTIONABLE"
    assert admitted["capital_admission"]["status"] == "ALLOW"
    assert xsp_pending_buy_has_capital_reservation(admitted)


def test_xsp_sell_remains_allowed_without_a_capital_plan() -> None:
    admitted = apply_xsp_live_capital(
        _projected("SELL"),
        capital_plan=None,
        selection=_selection(),
        selection_file_sha256="",
        available_cash_usd=0,
    )

    assert admitted["status"] == "ACTIONABLE"
    assert admitted["capital_admission"]["reasons"] == [
        "risk_reduction_always_allowed"
    ]


def test_pending_buy_without_a_durable_reservation_is_rejected() -> None:
    assert not xsp_pending_buy_has_capital_reservation(_projected("BUY"))


def test_xsp_plan_derives_only_the_selected_notional_plus_fee(
    tmp_path,
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    selection = {
        **_selection(),
        "strategy_version": "xsp-v3",
        "nominee": {
            "fixed_entry_notional_usd": 900.0,
            "commission_limits_usd": {"UPRO": 0.45034925, "SPXU": 0.45034925},
        },
        "broker_at_selection": {
            **_selection()["broker_at_selection"],
            "minimum_settled_cash_usd": 900.45034925,
            "settled_cash_usd": 1_318.05,
        },
    }

    plan = build_xsp_live_capital_plan(
        selection,
        selection_path=selection_path,
        created_at_utc="2026-08-01T15:00:00+00:00",
    )

    assert plan["capital"]["managed_capital_cents"] == 90_046
    assert plan["capital"]["unallocated_reserve_cents"] == 41_759
    assert plan["sleeves"][0]["weight_bps"] == 10_000
