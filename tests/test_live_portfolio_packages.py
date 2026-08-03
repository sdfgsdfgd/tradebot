from __future__ import annotations

from pathlib import Path

import pytest

from tradebot.research import live_portfolio_packages as packages


def _selections() -> tuple[dict[str, object], dict[str, object]]:
    xsp = {
        "schema": "xsp.opening-edge-v3-upro-spxu-selected-run.v3",
        "selection_id": "a" * 64,
        "strategy_version": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
        "allocation_successor": {"package_id": "xsp-usd-800"},
    }
    gold = {
        "schema": "gold.1oz-regime-harmony-selected-run.v2",
        "selection_id": "b" * 64,
        "strategy_version": "gold.1oz-regime-harmony-stage76.v1",
        "allocation_successor": {"package_id": "gold-one-contract"},
        "risk": {
            "max_initial_margin_change_aud": 600.0,
            "max_maintenance_margin_change_aud": 520.0,
            "max_open_position_stress_usd": 256.16,
            "fx_stress_bps": 11_000,
        },
    }
    return xsp, gold


def _resources(**changes: object) -> dict[str, object]:
    value = {
        "account_id": "U123",
        "account_type": "CASH",
        "base_currency": "AUD",
        "settled_cash_usd": 1318.05,
        "available_funds_base": 2073.52,
        "excess_liquidity_base": 2078.33,
        "usd_to_base_rate": 1.4279269,
        "unmanaged_position_stress_base": 92.74,
    }
    value.update(changes)
    return value


def _build(monkeypatch, **resource_changes: object) -> dict[str, object]:
    xsp, gold = _selections()
    monkeypatch.setattr(
        packages, "load_xsp_v3_package_selection_from_mapping", lambda value: dict(value)
    )
    monkeypatch.setattr(
        packages, "load_gold_live_selection_from_mapping", lambda value: dict(value)
    )
    return packages.build_xsp_gold_portfolio_package_plan(
        xsp_selection=xsp,
        gold_selection=gold,
        xsp_selection_path="db/calibration/xsp_selected_live_transport.json",
        xsp_selection_file_sha256="c" * 64,
        gold_selection_path="db/calibration/gold_selected_live_transport.json",
        gold_selection_file_sha256="d" * 64,
        account_resources=_resources(**resource_changes),
        repository_root=Path(__file__).resolve().parents[1],
        created_at_utc="2026-08-03T12:00:00+00:00",
        supersedes_plan_id="e" * 64,
    )


def test_portfolio_composer_binds_both_selected_packages(monkeypatch) -> None:
    plan = _build(monkeypatch)
    allocated = {
        sleeve["sleeve_id"]: sleeve["allocated_package_id"]
        for sleeve in plan["sleeves"]
    }

    assert allocated == {
        "gold-1oz-stage76-margin": "gold-one-contract",
        "xsp-upro-spxu-rth-cash": "xsp-usd-800",
    }
    assert plan["capital"]["managed_capital_cents"] == 80_112
    assert plan["capital"]["unallocated_reserve_cents"] == 51_693


def test_portfolio_composer_rejects_selection_allocation_drift(monkeypatch) -> None:
    with pytest.raises(ValueError, match="allocated executable packages"):
        _build(monkeypatch, available_funds_base=2000.0)
