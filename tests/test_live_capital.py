from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tradebot.live.capital import (
    admit_live_capital,
    build_live_capital_plan,
    load_live_capital_plan,
    publish_live_capital_plan,
    usd_to_cents,
    validate_live_capital_decision,
)


RUN_ID = "a" * 64
SELECTION_SHA = "b" * 64


def _sleeve(*, weight_bps: int = 10_000) -> dict[str, object]:
    return {
        "sleeve_id": "xsp-upro-spxu-rth-cash",
        "strategy_id": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
        "run_id": RUN_ID,
        "selection_path": "db/calibration/xsp_selected_live_transport.json",
        "selection_file_sha256": SELECTION_SHA,
        "capital_kind": "CASH_DEBIT",
        "weight_bps": weight_bps,
    }


def _plan(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "account_id": "U123",
        "account_type": "CASH",
        "currency": "USD",
        "observed_settled_cash_usd": "1318.05",
        "managed_capital_usd": "900.45034925",
        "sleeves": [_sleeve()],
        "reserve_reasons": [
            "cash_above_selected_authority_unallocated",
            "selected_fixed_notional_and_commission_ceiling",
        ],
        "created_at_utc": "2026-08-01T15:00:00+00:00",
    }
    values.update(changes)
    return build_live_capital_plan(**values)


def _admit(plan: dict[str, object] | None, **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "intent": "ENTER",
        "account_id": "U123",
        "account_type": "CASH",
        "currency": "USD",
        "sleeve_id": "xsp-upro-spxu-rth-cash",
        "run_id": RUN_ID,
        "selection_file_sha256": SELECTION_SHA,
        "capital_kind": "CASH_DEBIT",
        "projected_capital_usd": "900.45034925",
        "cash_debit_usd": "900.45034925",
        "available_cash_usd": "1318.05",
    }
    values.update(changes)
    return admit_live_capital(plan, **values)


def test_usd_requirements_round_up_to_conservative_cents() -> None:
    assert usd_to_cents("900.45034925") == 90_046
    assert usd_to_cents("0") == 0
    with pytest.raises(ValueError):
        usd_to_cents("nan")


def test_available_cash_rounds_down_at_the_admission_boundary() -> None:
    decision = _admit(_plan(), available_cash_usd="900.4501")

    assert decision["status"] == "HOLD"
    assert "insufficient_live_cash" in decision["reasons"]


def test_plan_preserves_explicit_managed_pool_and_unallocated_cash() -> None:
    plan = _plan()

    assert plan["capital"] == {
        "observed_settled_cash_cents": 131_805,
        "managed_capital_cents": 90_046,
        "unallocated_reserve_cents": 41_759,
        "reserve_reasons": [
            "cash_above_selected_authority_unallocated",
            "selected_fixed_notional_and_commission_ceiling",
        ],
    }
    assert plan["sleeves"][0]["weight_bps"] == 10_000
    assert plan["constraints"]["unallocated_reserve_is_entry_authority"] is False


def test_weights_split_only_managed_capital_and_cannot_borrow_reserve() -> None:
    plan = _plan(
        sleeves=[
            {**_sleeve(weight_bps=7_500)},
            {
                **_sleeve(weight_bps=2_500),
                "sleeve_id": "mcl-defined-risk",
                "strategy_id": "mcl.future-selected.v1",
                "run_id": "c" * 64,
                "selection_file_sha256": "d" * 64,
                "capital_kind": "DEFINED_RISK_DEBIT",
                "selection_path": "db/calibration/mcl_selected_live_transport.json",
            },
        ]
    )

    decision = _admit(plan)

    assert decision["status"] == "HOLD"
    assert decision["reasons"] == ["capital_sleeve_limit_exceeded"]
    assert decision["allocation"]["sleeve_limit_cents"] == 67_534
    assert decision["allocation"]["unallocated_reserve_cents"] == 41_759


def test_entry_requires_exact_account_run_selection_kind_cap_and_cash() -> None:
    plan = _plan()
    admitted = _admit(plan)
    assert validate_live_capital_decision(admitted) == admitted

    cases = {
        "capital_account_identity_mismatch": {"account_id": "OTHER"},
        "capital_run_identity_mismatch": {"run_id": "f" * 64},
        "capital_selection_identity_mismatch": {
            "selection_file_sha256": "e" * 64
        },
        "capital_kind_mismatch": {"capital_kind": "DEFINED_RISK_DEBIT"},
        "capital_sleeve_limit_exceeded": {"projected_capital_usd": "900.47"},
        "insufficient_live_cash": {"available_cash_usd": "900.45"},
    }
    for reason, changes in cases.items():
        decision = _admit(plan, **changes)
        assert decision["status"] == "HOLD"
        assert reason in decision["reasons"]


def test_entries_fail_closed_but_reductions_never_depend_on_plan() -> None:
    assert _admit(None)["reasons"] == ["invalid_or_missing_capital_plan"]

    decision = _admit(
        None,
        intent="ROTATE_OUT",
        projected_capital_usd=0,
        cash_debit_usd=0,
        available_cash_usd=0,
    )

    assert decision["status"] == "ALLOW"
    assert decision["reasons"] == ["risk_reduction_always_allowed"]


def test_plan_tampering_is_rejected() -> None:
    plan = _plan()
    tampered = deepcopy(plan)
    tampered["capital"]["managed_capital_cents"] += 1

    assert _admit(tampered)["status"] == "HOLD"

    decision = _admit(plan)
    decision["allocation"]["sleeve_limit_cents"] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_live_capital_decision(decision)


def test_publish_is_idempotent_and_archives_exact_predecessor(tmp_path: Path) -> None:
    path = tmp_path / "live_capital_plan.json"
    first = _plan()
    assert publish_live_capital_plan(path, first) == first
    assert publish_live_capital_plan(path, first) == first

    second = _plan(
        managed_capital_usd="850",
        created_at_utc="2026-08-02T15:00:00+00:00",
        supersedes_plan_id=first["plan_id"],
    )
    publish_live_capital_plan(path, second)

    archive = path.with_name(f"live_capital_plan.{first['plan_id']}.json")
    assert load_live_capital_plan(path) == second
    assert load_live_capital_plan(archive) == first


def test_replacement_must_bind_current_generation(tmp_path: Path) -> None:
    path = tmp_path / "live_capital_plan.json"
    publish_live_capital_plan(path, _plan())

    with pytest.raises(ValueError, match="does not bind"):
        publish_live_capital_plan(
            path,
            _plan(
                managed_capital_usd="850",
                created_at_utc="2026-08-02T15:00:00+00:00",
            ),
        )
