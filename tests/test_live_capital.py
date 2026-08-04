from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tradebot.live.capital import (
    admit_live_capital,
    build_live_capital_plan,
    build_live_capital_plan_v2,
    build_live_capital_plan_v3,
    load_live_capital_plan,
    publish_live_capital_plan,
    usd_to_cents,
    validate_live_capital_decision,
)
from tradebot.live.capital_packages import (
    PACKAGE_FIRST_ADMITTER_METHOD,
    load_allocated_live_selection,
    publish_immutable_live_selection,
)
from tradebot.live.capital_stability import publish_portfolio_package_generation


RUN_ID = "a" * 64
SELECTION_SHA = "b" * 64
GOLD_RUN_ID = "c" * 64
GOLD_SELECTION_SHA = "d" * 64
MCL_RUN_ID = "f" * 64
MCL_SELECTION_SHA = "1" * 64


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


def _portfolio_plan() -> dict[str, object]:
    return build_live_capital_plan_v2(
        account_id="U123",
        account_type="CASH",
        cash_currency="USD",
        base_currency="AUD",
        observed_settled_cash_usd="1318.05",
        managed_capital_usd="900.45034925",
        sleeves=[
            {
                **_sleeve(),
                "position_symbols": ["UPRO", "SPXU"],
            },
            {
                "sleeve_id": "gold-1oz-stage76-margin",
                "strategy_id": "gold.1oz-regime-harmony-stage76.v1",
                "run_id": GOLD_RUN_ID,
                "selection_path": "db/calibration/gold_selected_live_transport.json",
                "selection_file_sha256": GOLD_SELECTION_SHA,
                "capital_kind": "FUTURES_MARGIN",
                "weight_bps": 0,
                "position_symbols": ["1OZ"],
                "margin": {
                    "base_currency": "AUD",
                    "max_contracts": 1,
                    "max_initial_margin_change_cents": 60_000,
                    "max_maintenance_margin_change_cents": 52_000,
                    "max_stressed_loss_usd_cents": 70_000,
                    "fx_stress_bps": 11_000,
                    "minimum_post_stress_excess_liquidity_cents": 30_000,
                },
            },
        ],
        reserve_reasons=["outside_selected_cash_authority"],
        created_at_utc="2026-08-03T09:40:00+00:00",
        supersedes_plan_id="e" * 64,
    )


def _margin_state(**changes: object) -> dict[str, object]:
    state: dict[str, object] = {
        "base_currency": "AUD",
        "quantity": 1,
        "initial_margin_change": 560.27,
        "maintenance_margin_change": 487.19,
        "initial_margin_after": 593.98,
        "maintenance_margin_after": 516.04,
        "equity_with_loan_after": 2105.62,
        "available_funds_before": 2072.91,
        "unrelated_position_gross": 93.65,
        "usd_to_base_rate": 1.43,
        "account_positions": [
            {"symbol": "TQQQ", "quantity": 1},
        ],
        "account_open_orders": [],
    }
    state.update(changes)
    return state


def _admit_gold(plan: dict[str, object], **changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "intent": "ENTER",
        "account_id": "U123",
        "account_type": "CASH",
        "currency": "USD",
        "sleeve_id": "gold-1oz-stage76-margin",
        "run_id": GOLD_RUN_ID,
        "selection_file_sha256": GOLD_SELECTION_SHA,
        "capital_kind": "FUTURES_MARGIN",
        "projected_capital_usd": 0,
        "cash_debit_usd": 0,
        "available_cash_usd": 1318.05,
        "resource_state": _margin_state(),
    }
    values.update(changes)
    return admit_live_capital(plan, **values)


def _package(
    package_id: str,
    rank: int,
    *,
    cash: int = 0,
    initial: int = 0,
    maintenance: int = 0,
    stress: int = 0,
) -> dict[str, object]:
    return {
        "package_id": package_id,
        "rank": rank,
        "cash_debit_usd_cents": cash,
        "initial_margin_base_cents": initial,
        "maintenance_margin_base_cents": maintenance,
        "stressed_loss_usd_cents": stress,
        "fx_stress_bps": 11_000,
    }


def _package_sleeves() -> list[dict[str, object]]:
    maes = [1_761, 2_043, 2_325, 2_607, 2_888, 2_888, 3_170, 3_452, 3_734, 4_015, 4_297]
    return [
        {
            "sleeve_id": "xsp-upro-spxu-rth-cash",
            "strategy_id": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
            "run_id": RUN_ID,
            "selection_path": "db/calibration/xsp_selected_live_transport.json",
            "selection_file_sha256": SELECTION_SHA,
            "capital_kind": "CASH_DEBIT",
            "position_symbols": ["UPRO", "SPXU"],
            "residual_weight_bps": 10_000,
            "minimum_package_id": "xsp-usd-400",
            "package_ladder": [
                _package(
                    f"xsp-usd-{notional}",
                    rank,
                    cash=notional * 100 + 46,
                    stress=maes[rank],
                )
                for rank, notional in enumerate(range(400, 901, 50))
            ],
        },
        {
            "sleeve_id": "gold-1oz-stage76-margin",
            "strategy_id": "gold.1oz-regime-harmony-stage76.v1",
            "run_id": GOLD_RUN_ID,
            "selection_path": "db/calibration/gold_selected_live_transport.json",
            "selection_file_sha256": GOLD_SELECTION_SHA,
            "capital_kind": "FUTURES_MARGIN",
            "position_symbols": ["1OZ"],
            "residual_weight_bps": 0,
            "minimum_package_id": "gold-one-contract",
            "package_ladder": [
                _package(
                    "gold-one-contract",
                    0,
                    cash=66,
                    initial=60_000,
                    maintenance=52_000,
                    stress=25_616,
                )
            ],
        },
    ]


def _package_plan(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "account_id": "U123",
        "account_type": "CASH",
        "cash_currency": "USD",
        "base_currency": "AUD",
        "observed_settled_cash_usd": "1318.05",
        "observed_available_funds_base": "2073.52",
        "observed_excess_liquidity_base": "2078.33",
        "usd_to_base_rate": "1.4279269",
        "minimum_post_reservation_base": "300",
        "unmanaged_position_stress_base": "92.74",
        "sleeves": _package_sleeves(),
        "reserve_reasons": ["cash_outside_allocated_executable_packages"],
        "created_at_utc": "2026-08-03T11:45:26+00:00",
        "supersedes_plan_id": "e" * 64,
    }
    values.update(changes)
    return build_live_capital_plan_v3(**values)


def _first_admitter_plan(**changes: object) -> dict[str, object]:
    sleeves = deepcopy(_package_sleeves())
    sleeves[0]["minimum_package_id"] = "xsp-usd-800"
    sleeves[0]["residual_weight_bps"] = 0
    sleeves.append(
        {
            "sleeve_id": "mcl-two-speed-auction-margin",
            "strategy_id": "mcl.two-speed-auction-relay.v18",
            "run_id": MCL_RUN_ID,
            "selection_path": "db/calibration/mcl_selected_live_transport.json",
            "selection_file_sha256": MCL_SELECTION_SHA,
            "capital_kind": "FUTURES_MARGIN",
            "position_symbols": ["MCL"],
            "residual_weight_bps": 0,
            "minimum_package_id": "mcl-one-contract-stage91",
            "package_ladder": [
                _package(
                    "mcl-one-contract-stage91",
                    0,
                    cash=76,
                    initial=268_670,
                    maintenance=214_936,
                    stress=30_552,
                )
            ],
        }
    )
    values: dict[str, object] = {
        "account_id": "U123",
        "account_type": "CASH",
        "cash_currency": "USD",
        "base_currency": "AUD",
        "observed_settled_cash_usd": "1318.05",
        "observed_available_funds_base": "3072.19",
        "observed_excess_liquidity_base": "3077.79",
        "usd_to_base_rate": "1.4239442",
        "minimum_post_reservation_base": "300",
        "unmanaged_position_stress_base": "98.88",
        "sleeves": sleeves,
        "reserve_reasons": ["cash_outside_individually_selected_packages"],
        "created_at_utc": "2026-08-04T08:31:20+00:00",
        "supersedes_plan_id": "e" * 64,
        "entry_capacity_policy": PACKAGE_FIRST_ADMITTER_METHOD,
    }
    values.update(changes)
    return build_live_capital_plan_v3(**values)


def _package_state(
    *,
    positions: list[dict[str, object]] | None = None,
    available: int = 207_352,
    excess: int = 207_833,
    initial: int = 0,
    maintenance: int = 0,
) -> dict[str, object]:
    return {
        "account_positions": positions
        if positions is not None
        else [
            {
                "symbol": "TQQQ",
                "quantity": 1,
                "market_value_base_cents": 9_274,
            }
        ],
        "account_open_orders": [],
        "base_currency": "AUD",
        "available_funds_base_cents": available,
        "excess_liquidity_base_cents": excess,
        "usd_to_base_rate_ppm": 1_427_927,
        "candidate_initial_margin_base_cents": initial,
        "candidate_maintenance_margin_base_cents": maintenance,
    }


def _admit_first(
    plan: dict[str, object],
    sleeve_id: str,
    *,
    resource_state: dict[str, object],
    available_cash: object = "1318.05",
) -> dict[str, object]:
    sleeve = next(row for row in plan["sleeves"] if row["sleeve_id"] == sleeve_id)
    package = next(
        row
        for row in sleeve["package_ladder"]
        if row["package_id"] == sleeve["allocated_package_id"]
    )
    return admit_live_capital(
        plan,
        intent="ENTER",
        account_id="U123",
        account_type="CASH",
        currency="USD",
        sleeve_id=sleeve_id,
        run_id=str(sleeve["run_id"]),
        selection_file_sha256=str(sleeve["selection_file_sha256"]),
        capital_kind=str(sleeve["capital_kind"]),
        projected_capital_usd=int(package["cash_debit_usd_cents"]) / 100,
        cash_debit_usd=int(package["cash_debit_usd_cents"]) / 100,
        available_cash_usd=available_cash,
        resource_state=resource_state,
    )


def _admit_package(
    plan: dict[str, object],
    *,
    gold: bool,
    resource_state: dict[str, object],
    available_cash: object = "1318.05",
) -> dict[str, object]:
    return admit_live_capital(
        plan,
        intent="ENTER",
        account_id="U123",
        account_type="CASH",
        currency="USD",
        sleeve_id=(
            "gold-1oz-stage76-margin" if gold else "xsp-upro-spxu-rth-cash"
        ),
        run_id=GOLD_RUN_ID if gold else RUN_ID,
        selection_file_sha256=GOLD_SELECTION_SHA if gold else SELECTION_SHA,
        capital_kind="FUTURES_MARGIN" if gold else "CASH_DEBIT",
        projected_capital_usd="0.66" if gold else "800.46",
        cash_debit_usd="0.66" if gold else "800.46",
        available_cash_usd=available_cash,
        resource_state=resource_state,
    )


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


def test_v2_keeps_cash_and_futures_margin_as_distinct_resources() -> None:
    plan = _portfolio_plan()
    decision = _admit_gold(plan)

    assert plan["schema"] == "live.capital-plan.v2"
    assert plan["capital"]["managed_capital_cents"] == 90_046
    assert plan["capital"]["unallocated_reserve_cents"] == 41_759
    assert [row["weight_bps"] for row in plan["sleeves"]] == [0, 10_000]
    assert decision["status"] == "ALLOW"
    assert decision["allocation"]["post_stress_excess_liquidity_cents"] == 39_483


def test_v2_blocks_overlap_and_margin_or_stress_boundary_failures() -> None:
    plan = _portfolio_plan()
    occupied = _admit_gold(
        plan,
        resource_state=_margin_state(
            account_positions=[{"symbol": "UPRO", "quantity": 6}]
        ),
    )
    oversized = _admit_gold(
        plan,
        resource_state=_margin_state(initial_margin_change=600.01),
    )
    stressed = _admit_gold(
        plan,
        resource_state=_margin_state(equity_with_loan_after=1900.0),
    )

    assert "concurrent_directional_sleeve_active" in occupied["reasons"]
    assert "initial_margin_limit_exceeded" in oversized["reasons"]
    assert "post_stress_excess_liquidity_below_floor" in stressed["reasons"]


def test_v3_reserves_all_minima_then_assigns_the_largest_safe_residual_package() -> None:
    plan = _package_plan()
    by_id = {row["sleeve_id"]: row for row in plan["sleeves"]}

    assert plan["schema"] == "live.capital-plan.v3"
    assert by_id["gold-1oz-stage76-margin"]["allocated_package_id"] == (
        "gold-one-contract"
    )
    assert by_id["xsp-upro-spxu-rth-cash"]["allocated_package_id"] == (
        "xsp-usd-800"
    )
    assert plan["allocation"]["capacity"][
        "post_reservation_available_funds_base_cents"
    ] >= 30_000
    assert plan["constraints"]["minimum_executable_packages_reserved_first"] is True


def test_v3_rejects_a_portfolio_whose_minimum_packages_cannot_coexist() -> None:
    sleeves = _package_sleeves()
    sleeves[1]["package_ladder"][0]["initial_margin_base_cents"] = 180_000

    with pytest.raises(ValueError, match="minimum executable packages"):
        _package_plan(sleeves=sleeves)


def test_v3_package_allocation_is_product_agnostic() -> None:
    sleeves = deepcopy(_package_sleeves())
    debit = sleeves[1]
    debit.update(
        {
            "sleeve_id": "mcl-defined-risk-debit",
            "strategy_id": "mcl.future-selected.v1",
            "run_id": "f" * 64,
            "selection_path": "db/calibration/mcl_selected_live_transport.json",
            "selection_file_sha256": "1" * 64,
            "capital_kind": "DEFINED_RISK_DEBIT",
            "position_symbols": ["MCL"],
            "minimum_package_id": "mcl-debit-one",
            "package_ladder": [
                _package(
                    "mcl-debit-one",
                    0,
                    cash=10_066,
                    stress=10_000,
                )
            ],
        }
    )

    plan = _package_plan(sleeves=sleeves)
    by_id = {row["sleeve_id"]: row for row in plan["sleeves"]}

    assert by_id["mcl-defined-risk-debit"]["allocated_package_id"] == (
        "mcl-debit-one"
    )
    assert by_id["xsp-upro-spxu-rth-cash"]["allocated_package_id"] == (
        "xsp-usd-900"
    )


def test_v3_plan_is_the_atomic_pointer_to_immutable_selected_runs(
    tmp_path: Path,
) -> None:
    selection = {
        "selection_id": RUN_ID,
        "strategy_version": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
        "authority": "selected_live_cash_transport",
    }
    relative, digest = publish_immutable_live_selection(tmp_path, selection)
    sleeves = deepcopy(_package_sleeves())
    sleeves[0]["selection_path"] = relative
    sleeves[0]["selection_file_sha256"] = digest
    plan = _package_plan(sleeves=sleeves)

    loaded, path, loaded_digest = load_allocated_live_selection(
        plan,
        sleeve_id="xsp-upro-spxu-rth-cash",
        repository_root=tmp_path,
    )

    assert loaded == selection
    assert path == tmp_path / relative
    assert loaded_digest == digest
    assert publish_immutable_live_selection(tmp_path, selection) == (relative, digest)

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file identity changed"):
        load_allocated_live_selection(
            plan,
            sleeve_id="xsp-upro-spxu-rth-cash",
            repository_root=tmp_path,
        )


def test_portfolio_generation_immutably_binds_every_selected_sleeve(
    tmp_path: Path,
) -> None:
    sleeves = deepcopy(_package_sleeves())
    preview = _package_plan(sleeves=sleeves)
    packages = {
        sleeve["sleeve_id"]: sleeve["allocated_package_id"]
        for sleeve in preview["sleeves"]
    }
    for sleeve in sleeves:
        selection = {
            "selection_id": sleeve["run_id"],
            "strategy_version": sleeve["strategy_id"],
            "allocation_successor": {
                "package_id": packages[sleeve["sleeve_id"]]
            },
        }
        relative, digest = publish_immutable_live_selection(tmp_path, selection)
        sleeve["selection_path"] = relative
        sleeve["selection_file_sha256"] = digest
    plan = _package_plan(sleeves=sleeves)

    relative, digest = publish_portfolio_package_generation(tmp_path, plan)
    path = tmp_path / relative
    generation = json.loads(path.read_text())

    assert publish_portfolio_package_generation(tmp_path, plan) == (
        relative,
        digest,
    )
    assert generation["plan"] == plan
    assert set(generation["selections"]) == {
        "gold-1oz-stage76-margin",
        "xsp-upro-spxu-rth-cash",
    }
    assert generation["submitted_orders"] == 0

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation changed"):
        publish_portfolio_package_generation(tmp_path, plan)


def test_v3_allows_both_entry_orderings_without_reusing_promised_resources() -> None:
    plan = _package_plan()
    both_flat_xsp = _admit_package(plan, gold=False, resource_state=_package_state())
    both_flat_gold = _admit_package(
        plan,
        gold=True,
        resource_state=_package_state(initial=56_044, maintenance=48_735),
    )
    gold_first_then_xsp = _admit_package(
        plan,
        gold=False,
        resource_state=_package_state(
            positions=[
                {"symbol": "1OZ", "quantity": 1},
                {
                    "symbol": "TQQQ",
                    "quantity": 1,
                    "market_value_base_cents": 9_274,
                },
            ],
            available=151_086,
            excess=158_877,
        ),
    )
    xsp_first_then_gold = _admit_package(
        plan,
        gold=True,
        resource_state=_package_state(
            positions=[
                {"symbol": "UPRO", "quantity": 5},
                {
                    "symbol": "TQQQ",
                    "quantity": 1,
                    "market_value_base_cents": 9_274,
                },
            ],
            available=93_032,
            excess=190_000,
            initial=56_044,
            maintenance=48_735,
        ),
        available_cash="517.59",
    )

    assert both_flat_xsp["status"] == "ALLOW"
    assert both_flat_gold["status"] == "ALLOW"
    assert gold_first_then_xsp["status"] == "ALLOW"
    assert xsp_first_then_gold["status"] == "ALLOW"
    assert gold_first_then_xsp["allocation"]["active_sleeves"] == [
        "gold-1oz-stage76-margin"
    ]
    assert xsp_first_then_gold["allocation"]["active_sleeves"] == [
        "xsp-upro-spxu-rth-cash"
    ]


def test_v3_holds_unknown_orders_and_full_stresses_unmanaged_positions() -> None:
    plan = _package_plan()
    open_order = _package_state()
    open_order["account_open_orders"] = [{"symbol": "MCL", "quantity": 1}]
    oversized_unknown = _package_state(
        positions=[
            {
                "symbol": "UNKNOWN",
                "quantity": 1,
                "market_value_base_cents": 80_000,
            }
        ]
    )

    assert "open_order_blocks_portfolio_capacity_proof" in _admit_package(
        plan, gold=False, resource_state=open_order
    )["reasons"]
    assert "post_stress_excess_liquidity_below_floor" in _admit_package(
        plan, gold=False, resource_state=oversized_unknown
    )["reasons"]


def test_v3_first_admitter_binds_each_minimum_without_promising_the_sum() -> None:
    plan = _first_admitter_plan()

    assert plan["authority"] == "active_plus_candidate_just_in_time_admission"
    assert plan["allocation"]["aggregate_minimum_packages_promised"] is False
    assert plan["constraints"] == {
        "minimum_executable_packages_reserved_first": False,
        "residual_allocation": PACKAGE_FIRST_ADMITTER_METHOD,
        "flat_sleeves_retain_allocated_package_reservation": False,
        "unmanaged_positions_receive_full_gross_stress": True,
        "automatic_borrowing_or_unproved_reallocation": False,
        "risk_reduction_requires_plan": False,
        "entry_capacity_policy": PACKAGE_FIRST_ADMITTER_METHOD,
    }
    assert {
        row["sleeve_id"]: row["allocated_package_id"] for row in plan["sleeves"]
    } == {
        "gold-1oz-stage76-margin": "gold-one-contract",
        "mcl-two-speed-auction-margin": "mcl-one-contract-stage91",
        "xsp-upro-spxu-rth-cash": "xsp-usd-800",
    }


def test_v3_first_admitter_allows_any_individually_funded_flat_candidate() -> None:
    plan = _first_admitter_plan()
    state = _package_state(
        available=307_219,
        excess=307_779,
        initial=268_670,
        maintenance=214_936,
    )

    mcl = _admit_first(plan, "mcl-two-speed-auction-margin", resource_state=state)

    assert mcl["status"] == "ALLOW"
    assert mcl["allocation"]["active_sleeves"] == []
    assert mcl["allocation"]["cash_reserved_usd_cents"] == 76
    assert mcl["allocation"]["initial_margin_reserved_base_cents"] == 268_670
    assert mcl["allocation"]["post_stress_excess_liquidity_base_cents"] >= 30_000


def test_v3_first_admitter_later_candidate_uses_only_fresh_remaining_resources() -> None:
    plan = _first_admitter_plan()
    xsp_active = _package_state(
        positions=[
            {"symbol": "UPRO", "quantity": 5, "market_value_base_cents": 113_900},
            {"symbol": "TQQQ", "quantity": 1, "market_value_base_cents": 9_888},
        ],
        available=193_200,
        excess=295_000,
        initial=268_670,
        maintenance=214_936,
    )
    mcl_active = _package_state(
        positions=[
            {"symbol": "MCL", "quantity": 1, "market_value_base_cents": 0},
            {"symbol": "TQQQ", "quantity": 1, "market_value_base_cents": 9_888},
        ],
        available=38_400,
        excess=92_800,
    )

    after_xsp = _admit_first(
        plan, "mcl-two-speed-auction-margin", resource_state=xsp_active
    )
    after_mcl = _admit_first(
        plan, "xsp-upro-spxu-rth-cash", resource_state=mcl_active
    )

    assert after_xsp["status"] == "HOLD"
    assert "post_reservation_available_funds_below_floor" in after_xsp["reasons"]
    assert after_mcl["status"] == "HOLD"
    assert "post_reservation_available_funds_below_floor" in after_mcl["reasons"]
    assert after_xsp["allocation"]["active_sleeves"] == [
        "xsp-upro-spxu-rth-cash"
    ]
    assert after_mcl["allocation"]["active_sleeves"] == [
        "mcl-two-speed-auction-margin"
    ]


def test_v3_first_admitter_reserves_pending_owner_and_fails_unknown_orders_closed() -> None:
    plan = _first_admitter_plan()
    pending = _package_state(available=307_219, excess=307_779)
    pending["account_open_orders"] = [{"symbol": "MCL", "quantity": 1}]
    unknown = _package_state(available=307_219, excess=307_779)
    unknown["account_open_orders"] = [{"symbol": "OTHER", "quantity": 1}]

    gold = _admit_first(
        plan,
        "gold-1oz-stage76-margin",
        resource_state={**pending, "candidate_initial_margin_base_cents": 60_000,
                        "candidate_maintenance_margin_base_cents": 52_000},
    )
    invalid = _admit_first(
        plan, "xsp-upro-spxu-rth-cash", resource_state=unknown
    )

    assert gold["status"] == "HOLD"
    assert gold["allocation"]["pending_sleeves"] == [
        "mcl-two-speed-auction-margin"
    ]
    assert "post_reservation_available_funds_below_floor" in gold["reasons"]
    assert invalid["status"] == "HOLD"
    assert "open_order_has_no_unique_capital_owner" in invalid["reasons"]


def test_v3_first_admitter_remains_product_agnostic_and_reductions_bypass_capacity() -> None:
    sleeves = deepcopy(_first_admitter_plan()["sleeves"])
    for index, sleeve in enumerate(sleeves):
        sleeve.pop("allocated_package_id")
        sleeve["sleeve_id"] = f"generic-{index}"
        sleeve["strategy_id"] = f"generic.strategy.{index}"
        sleeve["position_symbols"] = [f"GEN{index}"]
    plan = _first_admitter_plan(sleeves=sleeves)
    candidate = plan["sleeves"][0]

    decision = admit_live_capital(
        plan,
        intent="EXIT",
        account_id="wrong",
        account_type="wrong",
        currency="wrong",
        sleeve_id=str(candidate["sleeve_id"]),
        run_id="wrong",
        selection_file_sha256="wrong",
        capital_kind="wrong",
        projected_capital_usd=0,
        cash_debit_usd=0,
        available_cash_usd=0,
        resource_state=None,
    )

    assert decision["status"] == "ALLOW"
    assert decision["reasons"] == ["risk_reduction_always_allowed"]


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
