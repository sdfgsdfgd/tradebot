from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tradebot.news.contract import publication_id
from tradebot.chart_data.series import OhlcvBar
from tradebot.engines.directional_impulse import (
    DirectionalImpulseHorizon,
    DirectionalImpulseSnapshot,
)
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.mcl_predictive_accumulator import (
    MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY,
    MCL_PREDICTIVE_ACCUMULATOR_SCHEMA,
    MCL_PREDICTIVE_ACCUMULATOR_VERSION,
    _append_treatment,
    _cohort,
    _identity,
    _news_context,
    _read_event_window,
    _seed_treatments,
    _source_candidates,
    load_mcl_predictive_generation,
    mcl_predictive_treatments,
)
from tradebot.research.mcl_predictive_onset import (
    MCL_PREDICTIVE_ONSET_AUTHORITY,
    MCL_PREDICTIVE_ONSET_FAMILIES,
    MclOnsetNewsContext,
    MclWeeklyPrior,
    combine_mcl_predictive_onset_atlas,
    project_mcl_completed_bar_onset,
    project_mcl_event_onset,
)
from tradebot.research.mcl_predictive_velocity import (
    MCL_VELOCITY_JERK_AUTHORITY,
    MCL_VELOCITY_JERK_INTERVALS,
    project_mcl_velocity_jerk_handoff,
)
from tradebot.research.mcl_turn_tape import MCL_TURN_TAPE_SCHEMA
from tradebot.research.mcl_two_speed_auction import (
    MclAuctionBar,
    MclAuctionDecision,
)


TURN = datetime(2026, 8, 4, 6, 0, tzinfo=timezone.utc)
GENERATION = "b" * 64


def _horizon(velocity: float, tr: float) -> DirectionalImpulseHorizon:
    return DirectionalImpulseHorizon(
        bars=48,
        elapsed_minutes=240.0,
        observations=49,
        anchor_lag_minutes=0.0,
        return_pct=1.0,
        slope_pct_per_bar=0.02,
        slope_velocity_pct_per_bar=velocity,
        slope_angle_deg=10.0,
        efficiency=0.7,
        tr_mean_pct=tr,
        turn=None,
        turn_age_bars=2,
    )


def _snapshot(
    velocity: float,
    tr: float,
    *,
    raw: bool = False,
    age: int = 30,
    h4_ready: bool = True,
) -> DirectionalImpulseSnapshot:
    return DirectionalImpulseSnapshot(
        ready=True,
        direction="up",
        abstain_reason=None,
        direction_score=0.4,
        coherence=0.8,
        conviction=0.6,
        atr_fast_pct=0.2,
        atr_slow_pct=0.15,
        atr_ratio=1.3,
        atr_velocity_pct=0.01,
        atr_acceleration_pct=0.002,
        turn_sequence_direction="up" if raw else None,
        turn_sequence_order="fast_to_slow" if raw else None,
        turn_sequence_span_bars=2 if raw else None,
        observed_horizons=5,
        required_turn_horizons=3,
        turn_ready=True,
        turn_abstain_reason=None,
        smoothed_direction_score=0.2,
        trend_state="up",
        state_age_bars=age,
        retrace_atr=2.1,
        turn_event="up" if raw else None,
        horizons=(_horizon(velocity, tr),) if h4_ready else (),
    )


def _decision(
    minutes: int,
    velocity: float,
    tr: float,
    *,
    raw: bool = False,
    h4_ready: bool = True,
) -> MclAuctionDecision:
    ts = TURN + timedelta(minutes=minutes)
    return MclAuctionDecision(
        observed_at_utc=ts,
        contract_key="202608",
        phase="RAW_TURN" if raw else "STATE",
        signal_at_utc=ts if raw else None,
        raw_direction=1 if raw else None,
        proposed_direction=1 if raw else None,
        admitted_direction=None,
        route=None,
        risk_reduction=raw,
        contract_reset=False,
        cl_move=0.04,
        mcl_move=0.03,
        velocity_aligned=True if raw else None,
        velocity_breadth=4 if raw else None,
        parity_aligned=True if raw else None,
        retained=None,
        raw_parity_ticks=3 if raw else None,
        basis_velocity_ticks=1 if raw else None,
        snapshot=_snapshot(
            velocity,
            tr,
            raw=raw,
            age=0 if raw else 30,
            h4_ready=h4_ready,
        ),
    )


def _bars() -> tuple[MclAuctionBar, ...]:
    rows = []
    for index in range(10):
        ts = TURN - timedelta(minutes=(9 - index) * 5)
        cl_close = 80.0 + (index * index * 0.002)
        mcl_close = cl_close - 0.02 + (index * index * 0.0002)

        def bar(close: float) -> OhlcvBar:
            return OhlcvBar(
                ts,
                close - 0.01,
                close + 0.03,
                close - 0.02,
                close,
                10.0 + index,
            )

        rows.append(MclAuctionBar("202608", bar(cl_close), bar(mcl_close)))
    return tuple(rows)


def _event_record(
    ts: datetime,
    *,
    cl_micro: tuple[float, float],
    mcl_micro: tuple[float, float],
    spread: float,
    leader: str,
) -> dict[str, object]:
    books = {}
    for symbol, (opening, closing) in (
        ("CL", cl_micro),
        ("MCL", mcl_micro),
    ):
        books[symbol] = {
            "summary": {
                "microprice_ohlc": [
                    opening,
                    max(opening, closing) + 0.01,
                    min(opening, closing) - 0.01,
                    closing,
                ],
                "bid_ask_events": 4 if symbol == "CL" else 3,
                "spread_ticks_min_max_last": [spread, spread, spread],
                "same_price_size_proxy": {
                    "bid_add": 3.0,
                    "bid_remove": 1.0,
                    "ask_add": 1.0,
                    "ask_remove": 2.0,
                },
                "signed_trade_volume_proxy": 2.0,
            }
        }
    record: dict[str, object] = {
        "schema": MCL_TURN_TAPE_SCHEMA,
        "kind": "second",
        "authority": "prospective_observation_only_no_signal_no_orders_no_capital",
        "timestamp_semantics": (
            "ib_insync_tcp_packet_receipt_utc_not_exchange_or_broker_event_time"
        ),
        "generation_sha256": GENERATION,
        "bucket_start_utc": ts.isoformat(),
        "recorded_at_utc": (ts + timedelta(seconds=3)).isoformat(),
        "market_data_types": {"CL": 1, "MCL": 1},
        "valid_evidence": True,
        "books": books,
        "cross_book": {
            "basis_ticks_ohlc": [-2.0, -1.0, -3.0, -1.0],
            "first_mid_move_leader": leader,
            "mcl_minus_cl_first_mid_move_us": 100_000,
        },
        "submitted_orders": 0,
    }
    record["record_id"] = hashlib.sha256(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return record


def _events() -> tuple[dict[str, object], ...]:
    offsets = (-59, -40, -20, -2, 1, 20, 40, 58, 61, 120, 180, 240, 299)
    rows = []
    for index, offset in enumerate(offsets):
        phase = 1 if offset >= 0 else -1
        rows.append(
            _event_record(
                TURN + timedelta(seconds=offset),
                cl_micro=(80.00 + index * 0.01, 80.01 + index * 0.012 * phase),
                mcl_micro=(79.98 + index * 0.01, 79.99 + index * 0.010 * phase),
                spread=1.0 if offset < 60 else 2.0,
                leader="CL" if index % 2 == 0 else "MCL",
            )
        )
    return tuple(rows)


def _velocity_event(
    cl: tuple[float, ...],
    mcl: tuple[float, ...],
) -> dict[str, object]:
    event = deepcopy(
        project_mcl_event_onset(
            _events(),
            raw_turn_at_utc=TURN,
            raw_direction=1,
            prefix_start_utc=TURN - timedelta(seconds=60),
            prefix_end_utc=TURN + timedelta(minutes=5),
        )
    )
    assert len(cl) == len(mcl) == len(MCL_VELOCITY_JERK_INTERVALS)
    for symbol, velocities in (("CL", cl), ("MCL", mcl)):
        for index, name in enumerate(MCL_VELOCITY_JERK_INTERVALS):
            book = event["windows"][name]["books"][symbol]
            book["directional_microprice_displacement_ticks"] = (
                1.0 if velocities[index] > 0.0 else -1.0
            )
            book["directional_microprice_slope_velocity_ticks_per_second2"] = (
                velocities[index]
            )
            book["microprice_tr_velocity_ticks"] = float(index)
            book["quote_intensity_acceleration"] = float(index * 2)
            book["spread_last_minus_first_ticks"] = 0.0
    return event


def test_completed_bar_atlas_freezes_eclectic_clocks_without_outcomes() -> None:
    payload = project_mcl_completed_bar_onset(
        (
            _decision(-10, 0.01, 0.10),
            _decision(-5, 0.02, 0.12),
            _decision(0, 0.05, 0.16, raw=True),
        ),
        _bars(),
        weekly_prior=MclWeeklyPrior(
            as_of_utc=TURN - timedelta(hours=8),
            return_pct=1.2,
            return_velocity_pct=0.2,
            tr_velocity_pct=-0.01,
            state_age_sessions=3,
        ),
        news=MclOnsetNewsContext(
            published_at_utc=TURN - timedelta(hours=1),
            horizon_hours=4.0,
            signed_pressure=-0.7,
            pressure_delta=0.1,
            pressure_velocity_per_hour=0.02,
            impact=0.8,
            confidence=0.9,
        ),
    )

    assert payload["authority"] == MCL_PREDICTIVE_ONSET_AUTHORITY
    assert payload["raw_direction"] == 1
    assert payload["v18"]["risk_reduction"] is True
    assert payload["v18"]["raw_state_age_bars"] == 0
    assert payload["v18"]["incumbent_state_age_bars"] == 30
    assert payload["weekly_prior"]["return_shape"] == "WITH_RAW_DIRECTION"
    assert payload["news"]["fresh"] is True
    assert (
        payload["bar_family_shapes"]["five_minute_to_four_hour_volatility_phase"][
            "phase"
        ]
        == "JOINT_EXPANSION"
    )
    assert payload["outcomes_exposed"] is False
    assert payload["submitted_orders"] == 0


def test_sparse_four_hour_clock_uses_only_finalized_observations_and_real_age() -> None:
    decisions = (
        _decision(-20, 0.01, 0.10),
        _decision(-15, 0.01, 0.10, h4_ready=False),
        _decision(-10, 0.03, 0.12),
        _decision(-5, 0.03, 0.12, h4_ready=False),
        _decision(0, 0.06, 0.17, raw=True),
    )
    with pytest.raises(ValueError, match="adjacent"):
        project_mcl_completed_bar_onset(decisions, _bars())

    payload = project_mcl_completed_bar_onset(
        decisions,
        _bars(),
        four_hour_clock="finalized_sparse",
    )

    features = payload["completed_bar_features"]
    assert features["four_hour_clock"] == "finalized_sparse"
    assert features["four_hour_observation_gaps_minutes"] == [10.0, 10.0]
    assert features["four_hour_measure_units"] == (
        "finalized_observation_delta_per_actual_elapsed_hour"
    )


def test_event_atlas_uses_complete_identity_clean_window_and_combines_all_families() -> None:
    completed = project_mcl_completed_bar_onset(
        (
            _decision(-10, 0.01, 0.10),
            _decision(-5, 0.02, 0.12),
            _decision(0, 0.05, 0.16, raw=True),
        ),
        _bars(),
    )
    event = project_mcl_event_onset(
        _events(),
        raw_turn_at_utc=TURN,
        raw_direction=1,
        prefix_start_utc=TURN - timedelta(seconds=60),
        prefix_end_utc=TURN + timedelta(minutes=5),
    )
    atlas = combine_mcl_predictive_onset_atlas(completed, event)

    assert tuple(atlas["families"]) == MCL_PREDICTIVE_ONSET_FAMILIES
    assert event["generation_sha256"] == GENERATION
    assert event["windows"]["pre_turn_60s"]["active_seconds"] == 4
    assert event["windows"]["turn_response_60s"]["active_seconds"] == 4
    assert event["windows"]["maturation_4m"]["active_seconds"] == 5
    assert event["windows"]["spark_0_5s"]["active_seconds"] == 1
    assert event["windows"]["acceptance_5_15s"]["active_seconds"] == 0
    assert event["windows"]["propagation_15_30s"]["active_seconds"] == 1
    assert event["windows"]["persistence_30_60s"]["active_seconds"] == 2
    ladder = event["event_shapes"]["velocity_ignition_ladder"]
    assert tuple(ladder) == (
        "closing_baseline_60_30s",
        "closing_acceleration_30_15s",
        "closing_commitment_15_5s",
        "closing_trigger_5_0s",
        "spark_0_5s",
        "acceptance_5_15s",
        "propagation_15_30s",
        "persistence_30_60s",
    )
    assert [
        event["windows"][name]["active_seconds"] for name in tuple(ladder)[:4]
    ] == [2, 1, 0, 1]
    assert ladder["spark_0_5s"]["books"]["CL"]["top_size_pressure"] == (
        "WITH_RAW_DIRECTION"
    )
    assert atlas["outcomes_exposed"] is False
    assert atlas["submitted_orders"] == 0
    serialized = json.dumps(atlas, sort_keys=True).lower()
    for forbidden in ("forward_return", '"mfe"', '"mae"', '"pnl"'):
        assert forbidden not in serialized


def test_event_atlas_fails_closed_on_incomplete_window_or_hash_drift() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        project_mcl_event_onset(
            _events(),
            raw_turn_at_utc=TURN,
            raw_direction=1,
            prefix_start_utc=TURN,
            prefix_end_utc=TURN + timedelta(minutes=5),
        )

    corrupted = [dict(row) for row in _events()]
    corrupted[0]["valid_evidence"] = False
    with pytest.raises(ValueError, match="invalid evidence"):
        project_mcl_event_onset(
            corrupted,
            raw_turn_at_utc=TURN,
            raw_direction=1,
            prefix_start_utc=TURN - timedelta(seconds=60),
            prefix_end_utc=TURN + timedelta(minutes=5),
        )


def test_slow_context_cannot_use_future_week_or_cross_roll() -> None:
    decisions = (
        _decision(-10, 0.01, 0.10),
        _decision(-5, 0.02, 0.12),
        _decision(0, 0.05, 0.16, raw=True),
    )
    with pytest.raises(ValueError, match="strictly causal"):
        project_mcl_completed_bar_onset(
            decisions,
            _bars(),
            weekly_prior=MclWeeklyPrior(
                as_of_utc=TURN,
                return_pct=0.0,
                return_velocity_pct=0.0,
                tr_velocity_pct=0.0,
                state_age_sessions=1,
            ),
        )

    bars = list(_bars())
    last = bars[-1]
    bars[-1] = MclAuctionBar("202609", last.cl, last.mcl)
    with pytest.raises(ValueError, match="contract roll"):
        project_mcl_completed_bar_onset(decisions, bars)


def test_velocity_jerk_handoff_preserves_frozen_sequence_without_a_winner() -> None:
    payload = project_mcl_velocity_jerk_handoff(
        _velocity_event(
            (-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
            (-2.0, -2.0, -0.5, 0.2, 0.8, 2.0, 3.0, 4.0),
        )
    )

    assert payload["authority"] == MCL_VELOCITY_JERK_AUTHORITY
    assert payload["handoff"] == "CL_LEADS"
    assert payload["books"]["CL"]["first_alignment_interval"] == (
        "closing_commitment_15_5s"
    )
    assert payload["books"]["MCL"]["first_alignment_interval"] == (
        "closing_trigger_5_0s"
    )
    assert payload["hypotheses"]["ORDERED_IGNITION"]["matched"] is True
    assert payload["hypotheses"]["TRANSPORT_REFUSAL"]["matched"] is False
    assert payload["hypotheses"]["TRANSPORT_NOISE"]["matched"] is False
    assert payload["winner"] is None
    assert payload["outcomes_exposed"] is False
    assert payload["submitted_orders"] == 0


def test_velocity_jerk_hypotheses_remain_independent_and_sign_only() -> None:
    noise = project_mcl_velocity_jerk_handoff(
        _velocity_event(
            (-2.0, -1.0, -0.5, 0.2, 0.8, 2.0, 3.0, 4.0),
            (-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
        )
    )
    exhaustion = project_mcl_velocity_jerk_handoff(
        _velocity_event(
            (-2.0, -1.0, 0.5, 1.0, -0.5, -1.0, 0.5, 1.0),
            (-2.0, -1.0, 0.5, 1.0, -0.5, -1.0, 0.5, 1.0),
        )
    )

    assert noise["handoff"] == "MCL_LEADS"
    assert noise["hypotheses"]["TRANSPORT_NOISE"]["matched"] is True
    assert exhaustion["hypotheses"]["EXHAUSTION"]["matched"] is True
    assert exhaustion["winner"] is None


def test_velocity_jerk_projection_rejects_authority_or_interval_drift() -> None:
    event = _velocity_event(
        (-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
        (-2.0, -2.0, -0.5, 0.2, 0.8, 2.0, 3.0, 4.0),
    )
    event["authority"] = "signal"
    with pytest.raises(ValueError, match="authority drifted"):
        project_mcl_velocity_jerk_handoff(event)

    event["authority"] = MCL_PREDICTIVE_ONSET_AUTHORITY
    del event["windows"]["spark_0_5s"]
    with pytest.raises(ValueError, match="interval contract drifted"):
        project_mcl_velocity_jerk_handoff(event)


def test_predictive_accumulator_seeds_exact_stage90_once(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generation = load_mcl_predictive_generation(repository_root=root)
    seeds = _seed_treatments(generation, repository_root=root)

    assert [row["raw_turn_at_utc"] for row in seeds] == [
        "2026-08-04T07:05:00+00:00",
        "2026-08-04T09:05:00+00:00",
    ]
    assert {row["raw_direction"] for row in seeds} == {-1, 1}
    assert all(row["schema"] == MCL_PREDICTIVE_ACCUMULATOR_SCHEMA for row in seeds)
    assert all(row["authority"] == MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY for row in seeds)
    assert all(row["outcomes_exposed"] is False for row in seeds)
    assert all(row["submitted_orders"] == 0 for row in seeds)

    ledger = LiveCalibrationLedger(tmp_path / "onset.jsonl")
    now = datetime(2026, 8, 4, 11, tzinfo=timezone.utc)
    assert [_append_treatment(ledger, row, recorded_at=now) for row in seeds] == [
        True,
        True,
    ]
    before = ledger.path.read_bytes()
    assert [_append_treatment(ledger, row, recorded_at=now) for row in seeds] == [
        False,
        False,
    ]
    assert ledger.path.read_bytes() == before
    restored = mcl_predictive_treatments(tuple(ledger.records()))
    assert [row["treatment_id"] for row in restored] == [
        row["treatment_id"] for row in seeds
    ]
    assert all(
        record["strategy_version"] == MCL_PREDICTIVE_ACCUMULATOR_VERSION
        for record in ledger.records()
    )


def test_predictive_accumulator_accepts_only_exact_live_raw_events() -> None:
    decision = {
        "observed_at_utc": TURN.isoformat(),
        "phase": "RAW_TURN",
        "raw_direction": 1,
    }
    event_id = _identity(decision)
    record = {
        "kind": "checkpoint",
        "checkpoint_id": "c" * 64,
        "strategy_version": "mcl.two-speed-auction-live-source.v1",
        "evidence": {
            "schema": "mcl.two-speed-auction-source-checkpoint.v1",
            "selection_id": "selection",
            "last_raw_turn": {
                "event_id": event_id,
                "observed_at_utc": TURN.isoformat(),
                "decision": decision,
            },
        },
    }

    values = _source_candidates(
        (record,),
        selection_id="selection",
        eligible_start=TURN - timedelta(minutes=1),
    )
    assert len(values) == 1
    assert values[0]["event_id"] == event_id
    assert values[0]["source_checkpoint_id"] == "c" * 64

    broken = deepcopy(record)
    broken["evidence"]["last_raw_turn"]["event_id"] = "d" * 64
    with pytest.raises(ValueError, match="source identity drifted"):
        _source_candidates(
            (broken,), selection_id="selection", eligible_start=TURN
        )


def test_predictive_news_context_reuses_v3_v4_contract_and_legacy_identity() -> None:
    def snapshot(
        *, schema: str, at: str, direction: int, impact: int, change: str
    ) -> dict[str, object]:
        return {
            "schema": schema,
            "score_version": "causal-impact-100.v2",
            "run_status": "published",
            "signal_as_of_utc": at,
            "snapshot_as_of_utc": at,
            "analysis": {
                "assets": {
                    "MCL": {
                        "direction": direction,
                        "impact": impact,
                        "confidence": 0.9,
                        "horizon_hours": 4,
                        "change": change,
                        "drivers": [],
                    }
                }
            },
        }

    legacy = snapshot(
        schema="tradebot.news-signal.v3",
        at="2026-08-04T06:00:00Z",
        direction=-1,
        impact=20,
        change="new",
    )
    current = snapshot(
        schema="tradebot.news-signal.v4",
        at="2026-08-04T07:00:00Z",
        direction=1,
        impact=50,
        change="strengthening",
    )
    current["publication_id"] = publication_id(current)

    context, source = _news_context(
        (legacy, current),
        treatment_at=datetime(2026, 8, 4, 7, 5, tzinfo=timezone.utc),
    )

    assert context is not None and source is not None
    assert context.signed_pressure == 0.5
    assert context.pressure_delta == 0.7
    assert context.pressure_velocity_per_hour == 0.7
    assert source["publication_id"] == current["publication_id"]
    assert source["prior_publication_id"] == publication_id(legacy)
    assert source["usable"] is True


def test_predictive_event_window_waits_for_both_bookends(tmp_path: Path) -> None:
    start = TURN - timedelta(seconds=60)
    end = TURN + timedelta(minutes=5)

    def row(stamp: datetime) -> dict[str, object]:
        body = {
            "schema": MCL_TURN_TAPE_SCHEMA,
            "generation_sha256": GENERATION,
            "bucket_start_utc": stamp.isoformat(),
            "valid_evidence": True,
        }
        return {**body, "record_id": _identity(body)}

    path = tmp_path / f"{TURN.date().isoformat()}.jsonl"
    path.write_text(json.dumps(row(start)) + "\n")
    assert (
        _read_event_window(
            tmp_path, turn=TURN, expected_generation_sha256=GENERATION
        )
        is None
    )
    path.write_text(
        "\n".join(json.dumps(value) for value in (row(start), row(end - timedelta(seconds=1))))
        + "\n"
    )
    records, evidence = _read_event_window(
        tmp_path, turn=TURN, expected_generation_sha256=GENERATION
    )
    assert len(records) == 2
    assert evidence["first_bucket_utc"] == start.isoformat()
    assert evidence["prefix_end_exclusive_utc"] == end.isoformat()


def test_predictive_cohort_never_opens_outcomes_implicitly() -> None:
    treatments = [
        {
            "raw_direction": 1 if index % 2 else -1,
            "admitted": index < 20,
            "route": "continuation" if index % 2 else "failed_auction",
            "velocity_jerk": {
                "handoff": "SAME_INTERVAL" if index < 20 else "UNRESOLVED"
            },
        }
        for index in range(30)
    ]
    result = _cohort(
        treatments,
        {
            "complete_turns": 30,
            "each_raw_direction": 10,
            "admitted_turns": 20,
            "each_admitted_route": 5,
            "resolved_handoffs": 20,
        },
    )
    assert all(result["gates"].values())
    assert result["verdict"] == "ELIGIBLE_FOR_MATCHED_CONTROL_PREREGISTRATION"
    assert "outcome" not in json.dumps(result).lower()


def test_predictive_accumulator_service_is_separate_read_only_and_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "tradebot/research/mcl_predictive_accumulator.py"
    live_source = (root / "tradebot/research/mcl_live_transport.py").read_text()
    tree = ast.parse(source.read_text(), filename=str(source))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    service = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset.service"
    ).read_text()
    timer = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset.timer"
    ).read_text()

    assert not calls & {
        "placeOrder",
        "place_limit_order",
        "preview_limit_order",
        "submit_order",
    }
    assert "historical_bars_ohlcv" not in source.read_text()
    assert "mcl_finalized_minute_source" in source.read_text()
    assert "mcl_finalized_minute_source" in live_source
    assert "Environment=IBKR_READONLY=1" in service
    assert "mcl_predictive_accumulator" in service
    assert "tradebot-mcl-live.service" not in service
    assert "tradebot-mcl-turn-tape.service" in service
    assert "NoNewPrivileges=true" in service
    assert timer.count("OnCalendar=") == 4
    assert all(":0/5:45 America/New_York" in row for row in timer.splitlines() if row.startswith("OnCalendar="))
    assert "Persistent=false" in timer
