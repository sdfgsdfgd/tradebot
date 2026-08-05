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
) -> list[dict[str, object]]:
    recorder = XspPressureTapeRecorder(
        generation_sha256="a" * 64,
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
