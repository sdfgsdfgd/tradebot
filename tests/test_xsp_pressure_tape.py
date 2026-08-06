from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from tradebot.research.xsp_pressure_tape import (
    XSP_PRESSURE_TAPE_AUTHORITY,
    XSP_PRESSURE_TAPE_SCHEMA,
    XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS,
    XspPressureTapeRecorder,
    _canonical,
    append_xsp_pressure_record,
    load_xsp_pressure_tape_generation,
)


START = datetime(2026, 8, 6, 13, 29, 59, tzinfo=timezone.utc)


def _contracts() -> dict[str, dict[str, object]]:
    return {
        "SPY": {
            "symbol": "SPY",
            "sec_type": "STK",
            "con_id": 756733,
            "local_symbol": "SPY",
            "exchange": "SMART",
            "primary_exchange": "ARCA",
            "currency": "USD",
            "tick_size": 0.01,
        },
        "UPRO": {
            "symbol": "UPRO",
            "sec_type": "STK",
            "con_id": 61228752,
            "local_symbol": "UPRO",
            "exchange": "SMART",
            "primary_exchange": "ARCA",
            "currency": "USD",
            "tick_size": 0.01,
        },
        "SPXU": {
            "symbol": "SPXU",
            "sec_type": "STK",
            "con_id": 828937771,
            "local_symbol": "SPXU",
            "exchange": "SMART",
            "primary_exchange": "ARCA",
            "currency": "USD",
            "tick_size": 0.01,
        },
    }


def _snapshot(
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
    *,
    last: float,
    last_size: float = 1.0,
    volume: float = 100.0,
) -> list[float]:
    return [bid, ask, bid_size, ask_size, last, last_size, volume]


def test_committed_generation_binds_exact_readonly_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    generation, generation_sha = load_xsp_pressure_tape_generation(
        root / "backtests/xsp/opening_edge_v3_pressure_tape_generation.json",
        root=root,
    )

    assert len(generation_sha) == 64
    assert generation["authority"] == XSP_PRESSURE_TAPE_AUTHORITY
    assert generation["timestamp_semantics"] == XSP_PRESSURE_TAPE_TIMESTAMP_SEMANTICS
    assert generation["tick_by_tick_subscriptions"] == 0
    assert generation["max_snapshot_age_seconds"] == 5.0
    assert generation["eligible_start_utc"] == "2026-08-06T13:20:00Z"
    assert {
        symbol: generation["contracts"][symbol]["con_id"]
        for symbol in ("SPY", "UPRO", "SPXU")
    } == {"SPY": 756733, "UPRO": 61228752, "SPXU": 828937771}


def test_second_record_preserves_changed_books_and_aligned_transport(
    tmp_path: Path,
) -> None:
    recorder = XspPressureTapeRecorder(
        generation_sha256="a" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    initial = {
        "SPY": _snapshot(100.00, 100.02, 2, 2, last=100.01),
        "UPRO": _snapshot(150.00, 150.02, 3, 2, last=150.01),
        "SPXU": _snapshot(30.00, 30.01, 2, 3, last=30.00),
    }
    for index, symbol in enumerate(("SPY", "UPRO", "SPXU"), start=1):
        recorder.ingest(
            symbol,
            initial[symbol],
            received_at=START + timedelta(microseconds=index * 100_000),
        )
    recorder.drain(
        now=START + timedelta(microseconds=900_000),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
        force=True,
    )

    second = START + timedelta(seconds=1)
    updates = {
        "SPY": _snapshot(100.01, 100.03, 4, 1, last=100.03, volume=104),
        "UPRO": _snapshot(150.03, 150.05, 5, 1, last=150.05, volume=106),
        "SPXU": _snapshot(29.99, 30.00, 1, 5, last=29.99, volume=108),
    }
    for index, symbol in enumerate(("SPY", "UPRO", "SPXU"), start=1):
        recorder.ingest(
            symbol,
            updates[symbol],
            received_at=second + timedelta(microseconds=index * 100_000),
        )
    row = recorder.drain(
        now=second + timedelta(seconds=1),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
        force=True,
    )[0]

    assert row["schema"] == XSP_PRESSURE_TAPE_SCHEMA
    assert row["session"] == "RTH"
    assert row["valid_evidence"] is True
    assert row["cross_book"]["full_alignment_direction"] == "up"
    assert row["cross_book"]["alignment_votes"] == {
        "up": 3,
        "down": 0,
        "flat": 0,
    }
    assert row["cross_book"]["first_mid_move_leaders"] == ["SPY"]
    assert row["books"]["UPRO"]["summary"]["cumulative_volume_delta"] == 6
    assert row["books"]["SPXU"]["summary"]["same_price_size_proxy"] == {
        "bid_add": 0.0,
        "bid_remove": 0.0,
        "ask_add": 0.0,
        "ask_remove": 0.0,
    }
    content = dict(row)
    record_id = content.pop("record_id")
    assert record_id == hashlib.sha256(_canonical(content)).hexdigest()


def test_carried_book_expires_instead_of_becoming_fake_live_evidence(
    tmp_path: Path,
) -> None:
    recorder = XspPressureTapeRecorder(
        generation_sha256="b" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    for symbol, price in (("SPY", 100.0), ("UPRO", 150.0), ("SPXU", 30.0)):
        recorder.ingest(
            symbol,
            _snapshot(price, price + 0.01, 1, 1, last=price),
            received_at=START + timedelta(microseconds=100_000),
        )
    for offset in range(8):
        recorder.heartbeat(START + timedelta(seconds=offset))
    rows = recorder.drain(
        now=START + timedelta(seconds=8),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
        force=True,
    )

    assert rows[0]["valid_evidence"] is True
    assert rows[-1]["valid_evidence"] is False
    assert rows[-1]["books"]["SPY"]["summary"][
        "snapshot_age_seconds_at_close"
    ] > 5.0


def test_preeligible_second_is_preserved_but_cannot_enter_cohort(
    tmp_path: Path,
) -> None:
    recorder = XspPressureTapeRecorder(
        generation_sha256="e" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
        eligible_start_utc=START + timedelta(seconds=1),
    )
    for symbol, price in (("SPY", 100.0), ("UPRO", 150.0), ("SPXU", 30.0)):
        recorder.ingest(
            symbol,
            _snapshot(price, price + 0.01, 1, 1, last=price),
            received_at=START + timedelta(microseconds=100_000),
        )
    row = recorder.drain(
        now=START + timedelta(seconds=1),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
        force=True,
    )[0]

    assert row["eligible_treatment"] is False
    assert row["valid_evidence"] is False
    assert row["eligible_start_utc"] == (START + timedelta(seconds=1)).isoformat()


def test_late_update_is_preserved_but_never_valid(tmp_path: Path) -> None:
    recorder = XspPressureTapeRecorder(
        generation_sha256="c" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    recorder.heartbeat(START)
    recorder.drain(
        now=START + timedelta(seconds=1),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
        force=True,
    )
    recorder.ingest(
        "SPY",
        _snapshot(100.0, 100.01, 1, 1, last=100.0),
        received_at=START + timedelta(microseconds=500_000),
    )
    late = recorder.drain(
        now=START + timedelta(seconds=2),
        market_data_types={"SPY": 1, "UPRO": 1, "SPXU": 1},
    )[0]

    assert late["kind"] == "late_events"
    assert late["valid_evidence"] is False
    assert late["events"][0]["symbol"] == "SPY"


def test_append_repairs_interrupted_jsonl_tail(tmp_path: Path) -> None:
    path = tmp_path / "tape.jsonl"
    path.write_bytes(b'{"old":1}\n{"partial"')

    append_xsp_pressure_record(path, {"new": 2})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"old": 1},
        {"new": 2},
    ]


def test_pressure_service_is_bounded_readonly_and_uses_no_tick_by_tick() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tradebot/research/xsp_pressure_tape.py").read_text()
    service = (root / "deploy/systemd/tradebot-xsp-pressure-tape.service").read_text()
    timer = (root / "deploy/systemd/tradebot-xsp-pressure-tape.timer").read_text()

    assert "IBKR_READONLY=1" in service
    assert "IBKR_CLIENT_ID_POOL_START=3217" in service
    assert "IBKR_CLIENT_ID_POOL_END=3219" in service
    assert "Restart=no" in service
    assert "RestartForceExitStatus=1" in service
    assert "RuntimeMaxSec=6h50m" in service
    assert "-m tradebot.research.xsp_pressure_tape" in service
    assert "09:20:00 America/New_York" in timer
    assert "Persistent=true" in timer
    assert "reqTickByTickData" not in source
    assert "place_limit_order" not in source
    assert "preview_limit_order" not in source
