from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import pytest

from tradebot.research.xsp_pressure_atlas import (
    XSP_PRESSURE_ATLAS_AUTHORITY,
    XSP_PRESSURE_ATLAS_HORIZONS_SECONDS,
    XSP_PRESSURE_ATLAS_VERSION,
    load_xsp_pressure_atlas_generation,
    project_xsp_pressure_atlas,
)
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
)
from tradebot.research.xsp_pressure_accumulator import (
    XSP_PRESSURE_ACCUMULATOR_AUTHORITY,
    accumulate_xsp_pressure_atlas,
    load_xsp_pressure_accumulator_generation,
    xsp_pressure_treatments,
)
from tradebot.research.xsp_pressure_tape import XspPressureTapeRecorder


START = datetime(2026, 8, 6, 13, 30, tzinfo=timezone.utc)


def _contracts() -> dict[str, dict[str, object]]:
    return {
        symbol: {
            "symbol": symbol,
            "sec_type": "STK",
            "con_id": con_id,
            "local_symbol": symbol,
            "exchange": "SMART",
            "primary_exchange": "ARCA",
            "currency": "USD",
            "tick_size": 0.01,
        }
        for symbol, con_id in (
            ("SPY", 756733),
            ("UPRO", 61228752),
            ("SPXU", 828937771),
        )
    }


def _records(
    tmp_path: Path,
    *,
    cresting: bool = False,
    generation_sha256: str = "a" * 64,
) -> list[dict[str, object]]:
    recorder = XspPressureTapeRecorder(
        generation_sha256=generation_sha256,
        contracts=_contracts(),
        output_dir=tmp_path,
        eligible_start_utc=START,
    )
    rows: list[dict[str, object]] = []
    for index in range(60):
        second = START + timedelta(seconds=index)
        market_move = (
            0.00008 * index - 0.0000005 * index * index
            if cresting
            else 0.00001 * index + 0.0000005 * index * index
        )
        prices = {
            "SPY": 100.0 * math.exp(market_move),
            "UPRO": 150.0 * math.exp(3.0 * market_move),
            "SPXU": 30.0 * math.exp(-3.0 * market_move),
        }
        for offset, symbol in enumerate(("SPY", "UPRO", "SPXU"), start=1):
            mid = prices[symbol]
            bid_size, ask_size = ((5.0, 1.0) if symbol != "SPXU" else (1.0, 5.0))
            recorder.ingest(
                symbol,
                [
                    mid - 0.005,
                    mid + 0.005,
                    bid_size,
                    ask_size,
                    mid,
                    1.0,
                    100.0 + index * index,
                ],
                received_at=second + timedelta(microseconds=offset * 100_000),
            )
        rows.extend(
            recorder.drain(
                now=second + timedelta(seconds=1),
                market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
                force=True,
            )
        )
    return rows


def _impulse() -> dict[str, object]:
    return {
        "atr_velocity_pct": 0.02,
        "atr_acceleration_pct": 0.01,
        "horizons": [
            {
                "bars": bars,
                "elapsed_minutes": bars * 5.0,
                "return_pct": 0.1 * bars,
                "slope_pct_per_bar": 0.02,
                "slope_velocity_pct_per_bar": 0.005,
            }
            for bars in (1, 3, 6, 12, 24)
        ],
    }


def _daily() -> dict[str, object]:
    return {
        "windows": {
            str(horizon): {"return": 0.01 * horizon}
            for horizon in (5, 10, 21, 42, 63, 84)
        },
        "return_velocity": {
            str(horizon): 0.001 for horizon in (5, 10, 21, 42, 63, 84)
        },
        "return_acceleration": {
            str(horizon): 0.0001 for horizon in (5, 10, 21, 42, 63, 84)
        },
        "tr_velocity": 0.01,
        "tr_acceleration": 0.001,
    }


def _source(
    *,
    signal_at: datetime,
    direction: str = "up",
    decision_trace: str = "d" * 64,
) -> dict[str, object]:
    entry_at = signal_at + timedelta(minutes=5)
    entry = {
        "signal_bar_ts": signal_at.replace(tzinfo=None).isoformat(),
        "directional_impulse": _impulse(),
        "market_state": {"hard_dir": direction},
        "control": {
            "source": "directional_impulse",
            "direction": direction,
            "proposed_direction": direction,
        },
        "local_extrema": None,
    }
    position = {
        "lane": "rth",
        "direction": direction,
        "entry_time": entry_at.replace(tzinfo=None).isoformat(),
        "attribution": {
            "decision_trace_fingerprint": decision_trace,
            "entry": entry,
        },
    }
    profile = {"latest_position": position}
    return {
        "kind": "checkpoint",
        "checkpoint_id": "c" * 64,
        "recorded_at_utc": entry_at.isoformat(),
        "strategy_version": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "session": "RTH",
        "status": "EVALUATED",
        "evidence": {
            "rth_provenance_fresh": True,
            "order_authority": "none",
            "paired_equity": {
                "daily_context_state": {"state": _daily()},
                "profiles": {
                    "research": profile,
                    "broker": profile,
                },
            },
        },
    }


def test_pressure_atlas_projects_ordered_multiscale_morphology(
    tmp_path: Path,
) -> None:
    atlas = project_xsp_pressure_atlas(
        _records(tmp_path),
        as_of_utc=START + timedelta(seconds=60),
        target_direction="up",
        xsp_impulse=_impulse(),
        spy_impulse=_impulse(),
        daily_context=_daily(),
    )

    assert atlas["schema"] == XSP_PRESSURE_ATLAS_VERSION
    assert atlas["authority"] == XSP_PRESSURE_ATLAS_AUTHORITY
    assert atlas["seconds"]["horizons_seconds"] == list(
        XSP_PRESSURE_ATLAS_HORIZONS_SECONDS
    )
    for horizon in (5, 15, 30, 45):
        row = atlas["seconds"]["horizons"][str(horizon)]
        assert row["path_consensus"]["target_alignment"] == "ALL_WITH_TARGET"
        assert row["velocity_consensus"]["target_alignment"] == "ALL_WITH_TARGET"
        assert row["morphology"]["transport"] == "FULL_ACCEPTANCE"
        assert row["morphology"]["ignition"] == "ORDERED_CASH_IGNITION"
        assert row["basis"]["transport_scale"].startswith("UPRO_and_inverse_SPXU")
    assert atlas["slower_context"]["cross_scale_state"] == (
        "ALL_CLOCKS_WITH_TARGET"
    )
    assert atlas["direction_authority"] == "opening_edge_v3_crown_only"
    assert atlas["classifier"] == "none"
    assert atlas["permission"] == "none"
    assert atlas["outcomes"] is None
    assert atlas["submitted_orders"] == 0
    assert len(atlas["projection_id"]) == 64


def test_pressure_atlas_normalizes_the_same_path_against_a_down_target(
    tmp_path: Path,
) -> None:
    atlas = project_xsp_pressure_atlas(
        _records(tmp_path),
        as_of_utc=START + timedelta(seconds=60),
        target_direction="down",
    )

    fast = atlas["seconds"]["horizons"]["5"]
    assert fast["path_consensus"]["target_alignment"] == "ALL_AGAINST_TARGET"
    assert "INVERSE_SPXU" in fast["books"]
    assert atlas["slower_context"]["cross_scale_state"] == "UNDERWARMED"


def test_pressure_atlas_distinguishes_a_velocity_crest_from_ignition(
    tmp_path: Path,
) -> None:
    atlas = project_xsp_pressure_atlas(
        _records(tmp_path, cresting=True),
        as_of_utc=START + timedelta(seconds=60),
        target_direction="up",
    )

    fast = atlas["seconds"]["horizons"]["5"]
    assert fast["path_consensus"]["target_alignment"] == "ALL_WITH_TARGET"
    assert fast["velocity_consensus"]["target_alignment"] == (
        "ALL_AGAINST_TARGET"
    )
    assert fast["morphology"]["ignition"] == (
        "ALIGNED_PATH_WITHOUT_ALIGNED_ACCELERATION"
    )
    assert all(
        row["mid"]["target_energy_state"] == "TARGET_CRESTING"
        for row in fast["books"].values()
    )


def test_pressure_atlas_rejects_gaps_and_invalid_evidence(tmp_path: Path) -> None:
    rows = _records(tmp_path)
    with pytest.raises(ValueError, match="60 contiguous"):
        project_xsp_pressure_atlas(
            rows[:-1],
            as_of_utc=START + timedelta(seconds=60),
            target_direction="up",
        )

    rows[-1]["valid_evidence"] = False
    with pytest.raises(ValueError, match="ineligible or invalid"):
        project_xsp_pressure_atlas(
            rows,
            as_of_utc=START + timedelta(seconds=60),
            target_direction="up",
        )

    rows = _records(tmp_path / "rehash")
    rows[-1]["cross_book"]["full_alignment_direction"] = "down"
    with pytest.raises(ValueError, match="content hash drifted"):
        project_xsp_pressure_atlas(
            rows,
            as_of_utc=START + timedelta(seconds=60),
            target_direction="up",
        )


def test_committed_atlas_generation_rehashes_all_owners() -> None:
    root = Path(__file__).resolve().parents[1]
    generation, generation_sha = load_xsp_pressure_atlas_generation(
        root / "backtests/xsp/opening_edge_v3_pressure_atlas_generation.json",
        root=root,
    )

    assert len(generation_sha) == 64
    assert generation["authority"] == XSP_PRESSURE_ATLAS_AUTHORITY
    assert generation["formation_seconds"] == 60
    assert generation["horizons_seconds"] == [1, 3, 5, 10, 15, 30, 45]
    assert generation["outcomes_open"] is False
    assert generation["permission_open"] is False
    assert generation["order_authority"] == "none"


def test_pressure_accumulator_generation_rehashes_every_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    generation, generation_sha = load_xsp_pressure_accumulator_generation(
        root
        / "backtests/xsp/opening_edge_v3_pressure_atlas_accumulation_generation.json",
        root=root,
    )

    assert len(generation_sha) == 64
    assert generation["authority"] == XSP_PRESSURE_ACCUMULATOR_AUTHORITY
    assert generation["cohort_gate"] == {
        "minimum_complete_crowned_targets": 30,
        "minimum_each_crowned_direction": 10,
        "minimum_repeated_morphologies_each_candidate_family": 5,
    }
    assert generation["outcomes_open"] is False
    assert generation["permission_open"] is False
    assert generation["submitted_orders"] == 0


def test_pressure_accumulator_appends_one_target_exactly_once(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generation, generation_sha = load_xsp_pressure_accumulator_generation(
        root
        / "backtests/xsp/opening_edge_v3_pressure_atlas_accumulation_generation.json",
        root=root,
    )
    tape = tmp_path / "tape"
    _records(
        tape,
        generation_sha256=str(generation["pressure_tape_generation_sha256"]),
    )
    signal_at = START + timedelta(seconds=60)
    source = _source(signal_at=signal_at)
    ledger = tmp_path / "atlas.jsonl"

    first = accumulate_xsp_pressure_atlas(
        (source,),
        observed_at=signal_at + timedelta(minutes=6),
        tape_dir=tape,
        ledger_path=ledger,
        repository_root=root,
    )
    before = ledger.read_bytes()
    second = accumulate_xsp_pressure_atlas(
        (source,),
        observed_at=signal_at + timedelta(minutes=7),
        tape_dir=tape,
        ledger_path=ledger,
        repository_root=root,
    )

    assert first["generation_sha256"] == generation_sha
    assert first["appended"] == 1
    assert second["appended"] == 0
    assert ledger.read_bytes() == before
    assert first["cohort"]["complete_targets"] == 1
    assert first["cohort"]["directions"] == {"up": 1}
    assert first["cohort"]["verdict"] == "FROZEN_ACCUMULATE"
    assert not any(first["cohort"]["gates"].values())
    treatments = xsp_pressure_treatments(
        tuple(LiveCalibrationLedger(ledger).records())
    )
    assert len(treatments) == 1
    assert treatments[0]["atlas"]["target_direction"] == "up"
    assert treatments[0]["slow_spy_status"].startswith("UNDERWARMED")
    assert treatments[0]["outcomes"] is None
    assert treatments[0]["permission"] == "none"
    assert treatments[0]["submitted_orders"] == 0


def test_pressure_accumulator_leaves_an_incomplete_window_unwritten(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    signal_at = START + timedelta(seconds=60)
    ledger = tmp_path / "atlas.jsonl"
    result = accumulate_xsp_pressure_atlas(
        (_source(signal_at=signal_at),),
        observed_at=signal_at + timedelta(minutes=6),
        tape_dir=tmp_path / "missing-tape",
        ledger_path=ledger,
        repository_root=root,
    )

    assert result["source_candidates"] == 1
    assert result["appended"] == 0
    assert result["incomplete"][0]["reason"].endswith(
        "requires 60 contiguous source seconds"
    )
    assert not ledger.exists()
    assert result["cohort"]["complete_targets"] == 0
    assert result["permission"] == "none"
    assert result["submitted_orders"] == 0


def test_pressure_accumulator_rejects_conflicting_target_provenance(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    signal_at = START + timedelta(seconds=60)
    with pytest.raises(ValueError, match="target identity conflicts"):
        accumulate_xsp_pressure_atlas(
            (
                _source(signal_at=signal_at, decision_trace="1" * 64),
                _source(signal_at=signal_at, decision_trace="2" * 64),
            ),
            observed_at=signal_at + timedelta(minutes=6),
            tape_dir=tmp_path / "tape",
            ledger_path=tmp_path / "atlas.jsonl",
            repository_root=root,
        )
