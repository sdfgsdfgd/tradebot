from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from ib_insync import (
    TickAttribBidAsk,
    TickAttribLast,
    TickByTickAllLast,
    TickByTickBidAsk,
)
import pytest

from tradebot.research.mcl_turn_tape import (
    MCL_TURN_TAPE_AUTHORITY,
    MCL_TURN_TAPE_GENERATION_SCHEMA,
    MCL_TURN_TAPE_TIMESTAMP_SEMANTICS,
    MclTurnTapeRecorder,
    _canonical,
    append_turn_tape_record,
    load_turn_tape_generation,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contracts() -> dict[str, dict[str, object]]:
    return {
        "CL": {
            "symbol": "CL",
            "con_id": 1,
            "local_symbol": "CLU6",
            "expiry": "20260820",
            "exchange": "NYMEX",
            "currency": "USD",
            "multiplier": "1000",
        },
        "MCL": {
            "symbol": "MCL",
            "con_id": 2,
            "local_symbol": "MCLU6",
            "expiry": "20260819",
            "exchange": "NYMEX",
            "currency": "USD",
            "multiplier": "100",
        },
    }


def _bid_ask(
    ts: datetime,
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
) -> TickByTickBidAsk:
    return TickByTickBidAsk(
        ts,
        bid,
        ask,
        bid_size,
        ask_size,
        TickAttribBidAsk(),
    )


def _trade(ts: datetime, price: float, size: float) -> TickByTickAllLast:
    return TickByTickAllLast(
        2,
        ts,
        price,
        size,
        TickAttribLast(),
        "",
        "",
    )


def test_generation_binds_preregistration_and_recorder(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    preregistration = root / "backtests/mcl/prereg.json"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text('{"schema":"prereg"}\n')
    recorder = root / "tradebot/research/recorder.py"
    recorder.parent.mkdir(parents=True)
    recorder.write_text("# frozen recorder\n")
    generation = {
        "schema": MCL_TURN_TAPE_GENERATION_SCHEMA,
        "authority": MCL_TURN_TAPE_AUTHORITY,
        "order_authority": "none",
        "timestamp_semantics": MCL_TURN_TAPE_TIMESTAMP_SEMANTICS,
        "preregistration_path": "backtests/mcl/prereg.json",
        "preregistration_sha256": _sha(preregistration),
        "recorder_sha256": _sha(recorder),
        "contracts": _contracts(),
        "flush_lag_seconds": 3,
        "submitted_orders": 0,
    }
    path = root / "generation.json"
    path.write_text(json.dumps(generation))

    loaded, generation_sha = load_turn_tape_generation(
        path,
        root=root,
        recorder_path=recorder,
    )

    assert loaded == generation
    assert generation_sha == _sha(path)
    preregistration.write_text('{"schema":"tampered"}\n')
    with pytest.raises(ValueError, match="preregistration drifted"):
        load_turn_tape_generation(path, root=root, recorder_path=recorder)


def test_recorder_preserves_raw_events_and_deterministic_pressure_summary(
    tmp_path: Path,
) -> None:
    recorder = MclTurnTapeRecorder(
        generation_sha256="a" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    start = datetime(2026, 8, 4, 4, 30, tzinfo=timezone.utc)
    recorder.ingest("CL", _bid_ask(start + timedelta(microseconds=100_000), 80.00, 80.02, 2, 2))
    recorder.ingest("MCL", _bid_ask(start + timedelta(microseconds=150_000), 80.00, 80.02, 1, 3))
    recorder.ingest("CL", _trade(start + timedelta(microseconds=200_000), 80.02, 2))
    recorder.ingest("CL", _bid_ask(start + timedelta(microseconds=300_000), 80.01, 80.03, 3, 1))
    recorder.ingest("MCL", _bid_ask(start + timedelta(microseconds=500_000), 80.01, 80.03, 4, 2))
    recorder.ingest("MCL", _trade(start + timedelta(microseconds=600_000), 80.01, 3))

    rows = recorder.drain(
        now=start + timedelta(seconds=2),
        market_data_types={"CL": 1, "MCL": 1},
        force=True,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["valid_evidence"] is True
    assert row["timestamp_semantics"] == MCL_TURN_TAPE_TIMESTAMP_SEMANTICS
    assert row["books"]["CL"]["summary"]["bid_ask_events"] == 2
    assert row["books"]["CL"]["summary"]["trade_volume"] == 2
    assert row["books"]["CL"]["summary"]["signed_trade_volume_proxy"] == 2
    assert row["books"]["MCL"]["summary"]["signed_trade_volume_proxy"] == -3
    assert row["books"]["CL"]["summary"]["first_mid_move_offset_us"] == 300_000
    assert row["books"]["MCL"]["summary"]["first_mid_move_offset_us"] == 500_000
    assert row["cross_book"]["first_mid_move_leader"] == "CL"
    assert row["cross_book"]["mcl_minus_cl_first_mid_move_us"] == 200_000
    assert row["books"]["CL"]["summary"]["microprice_ohlc"][-1] == pytest.approx(80.025)
    assert len(row["books"]["CL"]["bid_ask"]) == 2
    content = dict(row)
    record_id = content.pop("record_id")
    assert record_id == hashlib.sha256(_canonical(content)).hexdigest()
    saved = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert saved == row


def test_append_repairs_interrupted_jsonl_tail(tmp_path: Path) -> None:
    path = tmp_path / "tape.jsonl"
    path.write_bytes(b'{"old":1}\n{"partial"')

    append_turn_tape_record(path, {"new": 2})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"old": 1},
        {"new": 2},
    ]


def test_same_receipt_time_preserves_quote_trade_order_and_avoids_false_basis(
    tmp_path: Path,
) -> None:
    recorder = MclTurnTapeRecorder(
        generation_sha256="c" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    start = datetime(2026, 8, 4, 4, 30, tzinfo=timezone.utc)
    shared = start + timedelta(microseconds=250_000)
    recorder.ingest("CL", _bid_ask(start, 80.00, 80.02, 2, 2))
    recorder.ingest("MCL", _bid_ask(start, 80.00, 80.02, 2, 2))
    recorder.ingest("CL", _trade(shared, 80.02, 1))
    recorder.ingest("CL", _bid_ask(shared, 80.02, 80.04, 2, 2))
    recorder.ingest("MCL", _bid_ask(shared, 80.02, 80.04, 2, 2))

    row = recorder.drain(
        now=start + timedelta(seconds=2),
        market_data_types={"CL": 1, "MCL": 1},
        force=True,
    )[0]

    assert row["books"]["CL"]["summary"]["signed_trade_volume_proxy"] == 1
    assert row["cross_book"]["basis_ticks_ohlc"] == [0.0, 0.0, 0.0, 0.0]
    assert row["cross_book"]["first_mid_move_leader"] == "SIMULTANEOUS"


def test_initial_one_sided_book_is_not_valid_matched_evidence(tmp_path: Path) -> None:
    recorder = MclTurnTapeRecorder(
        generation_sha256="d" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    start = datetime(2026, 8, 4, 4, 30, tzinfo=timezone.utc)
    recorder.ingest("CL", _bid_ask(start, 80.00, 80.02, 1, 1))

    row = recorder.drain(
        now=start + timedelta(seconds=2),
        market_data_types={"CL": 1, "MCL": 1},
        force=True,
    )[0]

    assert row["valid_evidence"] is False
    assert row["cross_book"]["basis_ticks_ohlc"] is None


def test_late_events_are_persisted_but_never_valid(tmp_path: Path) -> None:
    recorder = MclTurnTapeRecorder(
        generation_sha256="b" * 64,
        contracts=_contracts(),
        output_dir=tmp_path,
    )
    start = datetime(2026, 8, 4, 4, 30, tzinfo=timezone.utc)
    recorder.last_flushed_second = int(start.timestamp())
    recorder.ingest("CL", _bid_ask(start, 80.00, 80.02, 1, 1))

    rows = recorder.drain(
        now=start + timedelta(seconds=5),
        market_data_types={"CL": 1, "MCL": 1},
    )

    assert len(rows) == 1
    assert rows[0]["kind"] == "late_events"
    assert rows[0]["valid_evidence"] is False
    assert rows[0]["events"][0]["receipt_time_utc"] == start.isoformat()


def test_committed_generation_and_service_are_exact_and_non_submitting() -> None:
    root = Path(__file__).resolve().parents[1]
    generation_path = (
        root
        / "backtests/mcl/mcl_turn_authenticity_microstructure_generation.json"
    )

    generation, generation_sha = load_turn_tape_generation(
        generation_path,
        root=root,
    )

    assert len(generation_sha) == 64
    assert generation["contracts"]["CL"]["con_id"] == 304037484
    assert generation["contracts"]["MCL"]["con_id"] == 661016525
    assert generation["timestamp_semantics"] == MCL_TURN_TAPE_TIMESTAMP_SEMANTICS
    service = (root / "deploy/systemd/tradebot-mcl-turn-tape.service").read_text()
    assert "IBKR_READONLY=1" in service
    assert "IBKR_CLIENT_ID_POOL_START=3211" in service
    assert "IBKR_CLIENT_ID_POOL_END=3213" in service
    assert "-m tradebot.research.mcl_turn_tape" in service
    source = (root / "tradebot/research/mcl_turn_tape.py").read_text()
    assert "place_limit_order" not in source
    assert "preview_limit_order" not in source
