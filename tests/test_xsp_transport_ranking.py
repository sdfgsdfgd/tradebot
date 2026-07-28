from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradebot.research.xsp_transport_ranking import rank_spyu_transport


NOW = datetime(2026, 7, 28, 23, 15, tzinfo=timezone.utc)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _period(net: float) -> dict[str, float]:
    return {
        "net_usd": net,
        "up_net_usd": net / 2,
        "down_net_usd": net / 2,
    }


def _profile(*, net: float, drawdown: float, trades: float = 200.0):
    periods = {
        "recent": _period(net / 10),
        "prior_complete_year": _period(net / 4),
        "latest_complete_year": _period(net / 3),
        "current_partial": _period(net / 8),
        "full_available_history": {
            **_period(net),
            "profit_factor": 1.2,
            "worst_session_net_usd": -10,
        },
    }
    return {
        "research_pass": True,
        "operationally_eligible": True,
        "current": {
            "periods": periods,
            "intrabar_max_drawdown_usd": drawdown,
            "trades_per_year": trades,
            "qty_ranges": {
                "SPYU": [20, 24],
                "SPXU": [50, 57],
            },
        },
        "scheduled": {
            "periods": {
                "full_available_history": _period(net / 2),
            }
        },
    }


def _ranking() -> dict[str, object]:
    return {
        "schema": "xsp.opening-edge-v2-spyu-selection-ranking.v2",
        "authority": "preregistered_research_ranking_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "gate_sha256": {
            "notional": "notional-gate",
            "five_slot": "five-gate",
            "two_slot": "two-gate",
        },
        "cohort_floor_members": [
            "recent.net_usd",
            "prior_complete_year.net_usd",
            "latest_complete_year.net_usd",
            "current_partial.net_usd",
            "full_available_history.up_net_usd",
            "full_available_history.down_net_usd",
            "scheduled_exit_next_open.full_available_history.net_usd",
        ],
        "ranking": {
            "order": "lexicographic",
            "descending": [
                "minimum_cohort_net_usd / max(intrabar_max_drawdown_usd, 1e-12)",
                "full_available_history.net_usd / max(intrabar_max_drawdown_usd, 1e-12)",
                "minimum_cohort_net_usd / fixed_entry_notional_usd",
                "full_available_history.net_usd / fixed_entry_notional_usd",
                "trades_per_year",
            ],
            "ascending": [
                "intrabar_max_drawdown_usd / starting_settled_cash_usd",
                "fixed_entry_notional_usd",
            ],
            "stable_family_tiebreak": [
                "two_slot",
                "five_slot",
                "notional",
            ],
            "final_tiebreak": "profile_id lexical ascending",
        },
        "selection_boundary": {
            "ranking_does_not_select": True,
            "usd_1350_cash_identity_required": True,
        },
    }


def _results(tmp_path: Path):
    notional = {
        "schema": "xsp.opening-edge-v2-spyu-spxu-replay.v1",
        "authority": "research_and_broker_preview_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "gate_sha256": "notional-gate",
        "operationally_eligible": ["notional=1200:fixed_measured"],
        "profiles": {
            "notional=1200:fixed_measured": _profile(
                net=100,
                drawdown=20,
            )
        },
    }
    five_slot = {
        "schema": "xsp.opening-edge-v2-spyu-spxu-cash-partition.v1",
        "authority": "research_and_broker_preview_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "gate_sha256": "five-gate",
        "capital_identity": {
            "starting_settled_cash_usd": 1350,
            "fixed_entry_notional_usd": 260,
        },
        "operationally_eligible": ["fixed_measured"],
        "profiles": {
            "fixed_measured": _profile(net=80, drawdown=10),
        },
    }
    two_slot = {
        "schema": "xsp.opening-edge-v2-spyu-spxu-two-slot.v1",
        "authority": "research_and_broker_preview_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "gate_sha256": "two-gate",
        "capital_identity": {
            "starting_settled_cash_usd": 1350,
            "fixed_entry_notional_usd": 650,
        },
        "operationally_eligible": [],
        "profiles": {},
    }
    paths = {
        "notional": _write(tmp_path / "notional.json", notional),
        "five_slot": _write(tmp_path / "five.json", five_slot),
        "two_slot": _write(tmp_path / "two.json", two_slot),
    }
    nominees = [
        "five_slot:fixed_measured",
        "notional:notional=1200:fixed_measured",
    ]
    latency = {
        "schema": "xsp.opening-edge-v2-spyu-entry-latency-stress.v1",
        "authority": "execution_stress_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "base_nominees": nominees,
        "profiles": {name: {"stress_pass": True} for name in nominees},
        "base_results": {
            family: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "operationally_eligible": json.loads(
                    path.read_text(encoding="utf-8")
                )["operationally_eligible"],
            }
            for family, path in paths.items()
        },
        "all_base_nominees_survive": True,
        "verdict": "LATENCY_STRESS_PASS_SELECTION_STILL_HOLD",
    }
    return paths, _write(tmp_path / "latency.json", latency)


def test_ranking_names_one_nominee_without_selecting(tmp_path: Path) -> None:
    paths, latency = _results(tmp_path)
    receipt = rank_spyu_transport(
        ranking_path=_write(tmp_path / "ranking.json", _ranking()),
        result_paths=paths,
        latency_path=latency,
        observed_at=NOW,
    )

    assert receipt["verdict"] == "NOMINEE_STILL_HOLD"
    assert receipt["nominee"]["family"] == "five_slot"
    assert receipt["nominee"]["fixed_entry_notional_usd"] == 260
    assert receipt["nominee"]["quantity_rule"].startswith("floor(")
    assert receipt["nominee"]["historical_quantity_ranges"] == {
        "SPYU": [20, 24],
        "SPXU": [50, 57],
    }
    assert receipt["nominee"]["frozen_max_quantities"] == {
        "SPYU": 24,
        "SPXU": 57,
    }
    assert receipt["selected_shadow_created"] is False
    assert receipt["order_authority"] == "none"
    assert "historical_full_quantity_symbol_dwell_validation" in receipt[
        "selection_blockers"
    ]
    assert len(receipt["ranked_candidates"]) == 2


def test_any_latency_failure_preserves_hold(tmp_path: Path) -> None:
    paths, latency_path = _results(tmp_path)
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    latency["profiles"]["five_slot:fixed_measured"]["stress_pass"] = False
    latency["all_base_nominees_survive"] = False
    latency["verdict"] = "HOLD"
    _write(latency_path, latency)

    receipt = rank_spyu_transport(
        ranking_path=_write(tmp_path / "ranking.json", _ranking()),
        result_paths=paths,
        latency_path=latency_path,
        observed_at=NOW,
    )

    assert receipt["verdict"] == "HOLD"
    assert receipt["reason"] == "latency_stress_failed"
    assert receipt["nominee"] is None
    assert receipt["ranked_candidates"] == []


def test_ranking_rejects_latency_candidate_drift(tmp_path: Path) -> None:
    paths, latency_path = _results(tmp_path)
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    latency["base_nominees"].pop()
    _write(latency_path, latency)

    with pytest.raises(
        ValueError,
        match="latency receipt does not bind every base nominee",
    ):
        rank_spyu_transport(
            ranking_path=_write(tmp_path / "ranking.json", _ranking()),
            result_paths=paths,
            latency_path=latency_path,
            observed_at=NOW,
        )


def test_ranking_rejects_invalid_quantity_range(tmp_path: Path) -> None:
    paths, latency_path = _results(tmp_path)
    result = json.loads(paths["five_slot"].read_text(encoding="utf-8"))
    result["profiles"]["fixed_measured"]["current"]["qty_ranges"]["SPYU"] = [
        24,
        20,
    ]
    _write(paths["five_slot"], result)
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    latency["base_results"]["five_slot"]["sha256"] = hashlib.sha256(
        paths["five_slot"].read_bytes()
    ).hexdigest()
    _write(latency_path, latency)

    with pytest.raises(ValueError, match="SPYU quantity range is invalid"):
        rank_spyu_transport(
            ranking_path=_write(tmp_path / "ranking.json", _ranking()),
            result_paths=paths,
            latency_path=latency_path,
            observed_at=NOW,
        )
