from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradebot.chart_data.series import OhlcvBar
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.mcl_shock_accumulator import (
    MCL_SHOCK_ACCUMULATOR_AUTHORITY,
    MCL_SHOCK_ACCUMULATOR_VERSION,
    MCL_SHOCK_GENERATION_PATH,
    _minute_resets,
    load_mcl_shock_generation,
    mcl_shock_cohort,
    mcl_shock_episodes,
    replay_mcl_shock_episodes,
)
from tradebot.research.mcl_shock_crest import (
    MCL_SHOCK_CREST_AUTHORITY,
    MCL_SHOCK_CREST_VERSION,
    MclShockBookEvidence,
    MclShockCrestEngine,
    MclShockCrestPolicy,
    MclShockObservation,
)
from tradebot.research.mcl_shock_evidence import (
    _slow_context,
    build_mcl_shock_observations,
)
from tradebot.research.mcl_shock_waves import (
    MCL_SHOCK_WAVE_AUTHORITY,
    MCL_SHOCK_WAVE_VERSION,
    MclAuthorityBoundShockWaveEngine,
    mcl_shock_full_alignment,
)
from tradebot.research.mcl_shock_wave_accumulator import (
    MCL_SHOCK_WAVE_GENERATION_PATH,
    _episode_replay_identity,
    mcl_shock_wave_cohort,
    mcl_shock_wave_episodes,
    replay_mcl_shock_wave_episodes,
)
from tradebot.research.mcl_shock_wave_generation import (
    _load_immutable_predecessor,
    _predecessor_path,
    build_mcl_shock_wave_successor_generation,
    validate_mcl_shock_wave_successor_generation,
)
from tradebot.research.mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION
from tradebot.research.mcl_minute_shock import MclMinuteShockEngine, MclShockMinute


START = datetime(2026, 8, 4, 10, 44, 20, tzinfo=timezone.utc)


def _book(
    velocity: float,
    *,
    slow: float | None = None,
    flow: float | None = None,
    volume_velocity: float = 1.0,
) -> MclShockBookEvidence:
    slow_value = velocity if slow is None else slow
    return MclShockBookEvidence(
        velocity_5s=velocity,
        velocity_15s=velocity,
        velocity_60s=velocity,
        slope_15m=slow_value,
        velocity_15m=slow_value,
        acceleration_15m=slow_value,
        signed_flow_15s=velocity if flow is None else flow,
        volume_velocity_5s=volume_velocity,
    )


def _observation(
    second: int,
    *,
    multiple: float,
    velocity: float,
    price: float,
    slow: float | None = None,
    flow: float | None = None,
    volume_velocity: float = 1.0,
    contract: str = "202608",
) -> MclShockObservation:
    book = _book(
        velocity,
        slow=slow,
        flow=flow,
        volume_velocity=volume_velocity,
    )
    return MclShockObservation(
        observed_at_utc=START + timedelta(seconds=second),
        contract_key=contract,
        mcl_microprice=price,
        volume_multiple=multiple,
        cl=book,
        mcl=book,
    )


def test_policy_freezes_levelled_volume_and_urgency_law() -> None:
    policy = MclShockCrestPolicy()

    assert policy.level(4.99) == "NORMAL_UNDER_5X"
    assert policy.level(5.0) == "ELEVATED_5_TO_10X"
    assert policy.level(10.0) == "MAJOR_PROTECT_10_TO_12X"
    assert policy.level(12.0) == "TRADEABLE_SHOCK_12_TO_20X"
    assert policy.level(20.0) == "REGIME_20X_PLUS"
    assert policy.tradeable_phase_speed_multiplier == 1.5
    assert policy.regime_phase_speed_multiplier == 2.0
    assert policy.major_exit_patience_multiplier == 2.0
    assert policy.tradeable_exit_patience_multiplier == 3.0
    assert policy.regime_exit_patience_multiplier == 4.0
    assert MclShockCrestPolicy(
        major_multiple=10.0,
        tradeable_multiple=10.0,
    ).level(10.0) == "TRADEABLE_SHOCK_12_TO_20X"


def test_regime_shock_waits_for_crest_then_admits_reacceleration() -> None:
    engine = MclShockCrestEngine()
    rows = [
        _observation(0, multiple=5.1, velocity=-1.0, price=81.70),
        _observation(1, multiple=10.1, velocity=-5.0, price=81.50),
        _observation(2, multiple=12.1, velocity=-8.0, price=81.40),
        _observation(3, multiple=20.1, velocity=-10.0, price=81.35),
        _observation(4, multiple=21.0, velocity=-8.0, price=81.34),
        _observation(5, multiple=22.0, velocity=-6.0, price=81.36),
        _observation(6, multiple=23.0, velocity=-4.0, price=81.42),
        _observation(7, multiple=24.0, velocity=-2.0, price=81.43),
        _observation(8, multiple=24.5, velocity=-3.0, price=81.38),
    ]

    decisions = [engine.update(row) for row in rows]

    assert decisions[3].latched_level == "REGIME_20X_PLUS"
    assert decisions[3].shock_direction == -1
    assert decisions[3].opposing_position_must_flatten is True
    assert decisions[3].phase_speed_multiplier == 2.0
    assert decisions[3].exit_patience_multiplier == 4.0
    assert decisions[3].fast_rotation_flatten_authority is False
    assert not any(row.continuation_direction for row in decisions[:7])
    assert decisions[6].phase == "CREST_CONFIRMED"
    assert decisions[8].phase == "CONTINUATION"
    assert decisions[8].continuation_direction == -1
    assert decisions[8].countertrend_inversion_eligible is False
    assert decisions[8].as_payload()["authority"] == MCL_SHOCK_CREST_AUTHORITY
    assert decisions[8].as_payload()["schema"] == MCL_SHOCK_CREST_VERSION
    assert decisions[8].as_payload()["submitted_orders"] == 0


def test_volume_reset_cannot_unlatch_a_major_shock() -> None:
    engine = MclShockCrestEngine()
    engine.update(_observation(0, multiple=10.1, velocity=-2.0, price=80.0))

    reset_minute = engine.update(
        _observation(1, multiple=0.1, velocity=-1.0, price=79.9)
    )

    assert reset_minute.current_level == "NORMAL_UNDER_5X"
    assert reset_minute.latched_level == "MAJOR_PROTECT_10_TO_12X"
    assert reset_minute.opposing_position_must_flatten is True
    assert reset_minute.countertrend_inversion_eligible is False


def test_seconds_volume_baseline_excludes_later_finalized_active_minute() -> None:
    active = START.replace(second=0)
    bars: dict[str, dict[datetime, OhlcvBar]] = {"CL": {}, "MCL": {}}
    for offset in range(-18, 2):
        closed_at = active + timedelta(minutes=offset)
        volume = (
            float(offset + 10)
            if -9 <= offset <= 0
            else 10_000.0
            if offset > 0
            else 1.0
        )
        bar = OhlcvBar(closed_at, 100.0, 100.1, 99.9, 100.0, volume)
        bars["CL"][closed_at] = bar
        bars["MCL"][closed_at] = bar

    rows = []
    for index in range(20):
        price = 100.0 - index / 100.0
        summary = {
            "microprice_ohlc": [price, price, price, price],
            "trade_volume": 1.0,
            "signed_trade_volume_proxy": -1.0,
        }
        rows.append(
            {
                "_time": active + timedelta(seconds=index * 3),
                "books": {
                    "CL": {"summary": summary},
                    "MCL": {"summary": summary},
                },
                "market_data_types": {"CL": 1, "MCL": 1},
            }
        )

    observations = build_mcl_shock_observations(
        rows,
        bars,
        contract_key="202608",
        eligible_start=active,
    )

    assert observations
    assert observations[-1][1].volume_multiple == pytest.approx(20.0 / 5.5)


def test_regime_rotation_arms_protection_before_reversal_without_flat_authority() -> None:
    policy = MclShockCrestPolicy(crest_lower_observations=2, crest_min_seconds=2.0)
    engine = MclShockCrestEngine(policy)
    setup = [
        _observation(0, multiple=20.0, velocity=-5.0, price=80.0),
        _observation(1, multiple=21.0, velocity=-3.0, price=79.9),
        _observation(2, multiple=22.0, velocity=-2.0, price=79.95),
        _observation(3, multiple=23.0, velocity=-3.0, price=79.8),
    ]
    assert engine.update(setup[0]).continuation_direction is None
    assert engine.update(setup[1]).continuation_direction is None
    assert engine.update(setup[2]).phase == "CREST_CONFIRMED"
    assert engine.update(setup[3]).phase == "CONTINUATION"

    decisions = []
    for second in range(4, 31):
        decisions.append(
            engine.update(
                _observation(
                    second,
                    multiple=1.0,
                    velocity=1.0,
                    slow=1.0,
                    flow=1.0,
                    volume_velocity=-1.0,
                    price=79.9 + second / 1000.0,
                )
            )
        )

    rotation = next(row for row in decisions if row.phase == "ROTATION_ARMED")
    reversal = next(row for row in decisions if row.phase == "REVERSAL_ELIGIBLE")
    assert rotation.observed_at_utc < reversal.observed_at_utc
    assert rotation.fast_rotation_flatten_authority is False
    assert rotation.countertrend_inversion_eligible is False
    assert reversal.countertrend_inversion_eligible is True


def test_disagreement_cannot_manufacture_a_shock_direction() -> None:
    down = _book(-2.0)
    up = _book(2.0)
    engine = MclShockCrestEngine()

    decision = engine.update(
        MclShockObservation(
            observed_at_utc=START,
            contract_key="202608",
            mcl_microprice=80.0,
            volume_multiple=25.0,
            cl=down,
            mcl=up,
        )
    )

    assert decision.latched_level == "REGIME_20X_PLUS"
    assert decision.shock_direction is None
    assert decision.continuation_direction is None
    assert decision.opposing_position_must_flatten is False


def test_time_must_increase_and_contract_roll_resets_state() -> None:
    engine = MclShockCrestEngine()
    row = _observation(0, multiple=20.0, velocity=-2.0, price=80.0)
    engine.update(row)
    with pytest.raises(ValueError, match="must increase"):
        engine.update(row)

    rolled = engine.update(
        _observation(
            1,
            multiple=0.0,
            velocity=0.0,
            price=80.0,
            contract="202609",
        )
    )
    assert rolled.latched_level == "NORMAL_UNDER_5X"
    assert rolled.shock_direction is None


def test_stage114_attention_is_directionless_until_full_major_alignment() -> None:
    engine = MclAuthorityBoundShockWaveEngine()

    attention = engine.update(
        _observation(0, multiple=6.1, velocity=2.0, price=80.0)
    )
    bound = engine.update(
        _observation(1, multiple=10.1, velocity=-3.0, price=79.9)
    )

    assert attention.event == "ATTENTION_OPENED"
    assert attention.authority_direction is None
    assert attention.wave_sequence == 0
    assert bound.event == "AUTHORITY_BOUND"
    assert bound.authority_direction == -1
    assert bound.authority_level == "MAJOR_PROTECT_10_TO_12X"
    assert bound.wave_sequence == 1
    assert bound.crest is not None
    assert bound.crest.shock_direction == -1
    assert bound.as_payload()["schema"] == MCL_SHOCK_WAVE_VERSION
    assert bound.as_payload()["authority"] == MCL_SHOCK_WAVE_AUTHORITY
    assert bound.as_payload()["submitted_orders"] == 0


def test_stage114_only_higher_level_opposite_alignment_hands_off() -> None:
    engine = MclAuthorityBoundShockWaveEngine()
    bound = engine.update(
        _observation(0, multiple=10.1, velocity=-5.0, price=80.0)
    )
    same_level_opposite = engine.update(
        _observation(1, multiple=11.0, velocity=4.0, price=80.1)
    )
    handoff = engine.update(
        _observation(2, multiple=12.1, velocity=3.0, price=80.2)
    )

    assert bound.authority_direction == -1
    assert same_level_opposite.authority_direction == -1
    assert same_level_opposite.event != "AUTHORITY_HANDOFF"
    assert handoff.event == "AUTHORITY_HANDOFF"
    assert handoff.handoff_from_direction == -1
    assert handoff.authority_direction == 1
    assert handoff.authority_level == "TRADEABLE_SHOCK_12_TO_20X"
    assert handoff.wave_sequence == 2
    assert handoff.crest is not None
    assert handoff.crest.shock_direction == 1
    assert handoff.crest.crest_at_utc is None


def test_stage114_requires_fresh_complete_alignment_for_every_binding() -> None:
    down = _book(-2.0)
    up = _book(2.0)
    mixed = MclShockObservation(
        observed_at_utc=START,
        contract_key="202608",
        mcl_microprice=80.0,
        volume_multiple=10.1,
        cl=down,
        mcl=up,
    )
    stale = MclShockObservation(
        observed_at_utc=START + timedelta(seconds=1),
        contract_key="202608",
        mcl_microprice=79.9,
        volume_multiple=10.2,
        cl=down,
        mcl=down,
        fresh_top=False,
    )
    engine = MclAuthorityBoundShockWaveEngine()

    assert mcl_shock_full_alignment(mixed) is None
    assert mcl_shock_full_alignment(stale) is None
    assert engine.update(mixed).event == "AUTHORITY_PENDING"
    pending = engine.update(stale)
    assert pending.authority_direction is None
    assert pending.wave_sequence == 0


def test_stage114_same_direction_escalation_keeps_one_wave() -> None:
    engine = MclAuthorityBoundShockWaveEngine()
    engine.update(_observation(0, multiple=10.1, velocity=-2.0, price=80.0))

    escalated = engine.update(
        _observation(1, multiple=20.1, velocity=-4.0, price=79.8)
    )

    assert escalated.event == "AUTHORITY_ESCALATED"
    assert escalated.authority_direction == -1
    assert escalated.authority_level == "REGIME_20X_PLUS"
    assert escalated.wave_sequence == 1


def test_stage114_episode_preserves_each_authoritative_wave() -> None:
    observations = [
        _observation(0, multiple=6.1, velocity=2.0, price=80.0),
        _observation(1, multiple=10.1, velocity=-4.0, price=79.8),
        _observation(2, multiple=12.1, velocity=3.0, price=80.0),
        _observation(4, multiple=1.0, velocity=0.0, price=80.0),
    ]
    source = [_episode_row(index) for index in (0, 1, 2, 4)]
    complete, opened = replay_mcl_shock_wave_episodes(
        list(zip(source, observations, strict=True)),
        resets=[{"at_utc": START + timedelta(seconds=3), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )

    assert opened is None
    assert len(complete) == 1
    episode = complete[0]
    assert episode["initial_authority_direction"] == -1
    assert episode["terminal_authority_direction"] == 1
    assert episode["reached_tradeable_12x"] is True
    assert [wave["event"] for wave in episode["authority_waves"]] == [
        "AUTHORITY_BOUND",
        "AUTHORITY_HANDOFF",
    ]
    assert [wave["direction"] for wave in episode["authority_waves"]] == [-1, 1]
    assert episode["authority_waves"][1]["handoff_from_direction"] == -1
    assert episode["outcomes_exposed"] is False
    assert episode["submitted_orders"] == 0


def test_stage114_replay_identity_ignores_only_the_sliding_bar_prefix() -> None:
    source = [_episode_row(index) for index in (0, 1, 3)]
    episodes, _opened = replay_mcl_shock_wave_episodes(
        list(
            zip(
                source,
                (
                    _observation(0, multiple=10.1, velocity=-4.0, price=79.8),
                    _observation(1, multiple=12.1, velocity=-3.0, price=79.7),
                    _observation(3, multiple=1.0, velocity=0.0, price=79.7),
                ),
                strict=True,
            )
        ),
        resets=[{"at_utc": START + timedelta(seconds=2), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )
    original = episodes[0]
    replayed = json.loads(json.dumps(original))
    replayed["bar_prefix"]["common_rows"] += 1
    replayed["bar_prefix"]["sha256"] = "9" * 64
    replayed["episode_sha256"] = "8" * 64

    assert _episode_replay_identity(replayed) == _episode_replay_identity(original)

    replayed["authority_waves"][0]["direction"] = 1
    assert _episode_replay_identity(replayed) != _episode_replay_identity(original)


def test_stage114_stored_episode_validation_keeps_full_bar_prefix_hash() -> None:
    source = [_episode_row(index) for index in (0, 1, 3)]
    episodes, _opened = replay_mcl_shock_wave_episodes(
        list(
            zip(
                source,
                (
                    _observation(0, multiple=10.1, velocity=-4.0, price=79.8),
                    _observation(1, multiple=12.1, velocity=-3.0, price=79.7),
                    _observation(3, multiple=1.0, velocity=0.0, price=79.7),
                ),
                strict=True,
            )
        ),
        resets=[{"at_utc": START + timedelta(seconds=2), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )
    episode = episodes[0]
    assert mcl_shock_wave_episodes([{"kind": "checkpoint", "evidence": episode}]) == [
        episode
    ]

    changed = json.loads(json.dumps(episode))
    changed["bar_prefix"]["common_rows"] += 1
    with pytest.raises(ValueError, match="episode identity drifted"):
        mcl_shock_wave_episodes([{"kind": "checkpoint", "evidence": changed}])


def test_stage114_wave_cohort_counts_handoffs_without_opening_outcomes() -> None:
    observations = [
        _observation(0, multiple=10.1, velocity=-4.0, price=79.8),
        _observation(1, multiple=12.1, velocity=3.0, price=80.0),
        _observation(3, multiple=1.0, velocity=0.0, price=80.0),
    ]
    source = [_episode_row(index) for index in (0, 1, 3)]
    episodes, _opened = replay_mcl_shock_wave_episodes(
        list(zip(source, observations, strict=True)),
        resets=[{"at_utc": START + timedelta(seconds=2), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )
    gate = {
        "complete_episodes": 1,
        "authority_bound_episodes": 1,
        "each_authority_direction": 1,
        "tradeable_episodes": 1,
        "regime_episodes": 0,
        "causal_crests": 0,
        "continuations": 0,
        "each_continuation_direction": 0,
        "authority_handoffs": 1,
        "each_handoff_direction": 0,
    }

    cohort = mcl_shock_wave_cohort(episodes, gate)

    assert cohort["authority_directions"] == {"down": 1, "up": 1}
    assert cohort["authority_handoffs"] == 1
    assert cohort["handoff_directions"] == {"up": 1}
    assert cohort["verdict"] == "COHORT_READY_FOR_PREREGISTERED_MATCHED_CONTROLS"


def _minute(index: int, close: float, volume: float, width: float) -> MclShockMinute:
    observed = START.replace(second=0) + timedelta(minutes=index)
    bar = OhlcvBar(
        observed,
        close + width / 4,
        close + width / 2,
        close - width / 2,
        close,
        volume,
    )
    return MclShockMinute("202608", bar, bar)


def test_minute_plateau_schedules_not_delays_the_stage106_entry() -> None:
    engine = MclMinuteShockEngine()
    rows = []
    close = 100.0
    for index in range(19):
        close -= 0.001 * (index + 1) ** 2
        rows.append(_minute(index, close, 10.0, 0.02))
    for index, (move, volume) in enumerate(
        ((-0.5, 150.0), (-0.8, 50.0), (-1.1, 50.0)), start=19
    ):
        close += move
        rows.append(_minute(index, close, volume, 0.6))

    decisions = [engine.update(row) for row in rows]
    confirmed = decisions[-1]
    assert confirmed.scheduled_entry_direction == -1
    assert confirmed.scheduled_entry_signal_at_utc == rows[-1].ts
    assert confirmed.entry_direction is None

    due = engine.update(_minute(22, close - 0.2, 20.0, 0.2))
    assert due.entry_direction == -1
    assert due.entry_signal_at_utc == rows[-1].ts
    assert due.active_direction_at_open == -1


def _episode_row(index: int) -> dict[str, object]:
    when = START + timedelta(seconds=index)
    summary = {
        "microprice_ohlc": [80.0, 80.0, 80.0, 80.0],
        "trade_volume": 1.0,
        "signed_trade_volume_proxy": -1.0,
    }
    return {
        "record_id": f"{index:064x}",
        "_time": when,
        "_recorded": when + timedelta(seconds=3),
        "books": {
            "CL": {"summary": summary},
            "MCL": {"summary": summary},
        },
    }


def _episode_bars() -> dict[str, dict[datetime, OhlcvBar]]:
    output = {"CL": {}, "MCL": {}}
    first = START.replace(second=0) - timedelta(minutes=800)
    for index in range(805):
        stamp = first + timedelta(minutes=index)
        close = 80.0 - index / 10_000.0
        bar = OhlcvBar(stamp, close, close + 0.01, close - 0.01, close, 10.0)
        output["CL"][stamp] = bar
        output["MCL"][stamp] = bar
    return output


def _episode_generation() -> dict[str, object]:
    return {
        "generation_id": "a" * 64,
        "selection_id": "b" * 64,
        "cohort_gate": {
            "complete_episodes": 30,
            "each_resolved_direction": 10,
            "tradeable_episodes": 10,
            "regime_episodes": 5,
            "causal_crests": 20,
            "continuations": 10,
            "each_continuation_direction": 5,
        },
    }


def test_prospective_regime_episode_records_crest_continuation_and_reset() -> None:
    rows = [
        _observation(0, multiple=5.1, velocity=-1.0, price=81.70),
        _observation(1, multiple=10.1, velocity=-5.0, price=81.50),
        _observation(2, multiple=12.1, velocity=-8.0, price=81.40),
        _observation(3, multiple=20.1, velocity=-10.0, price=81.35),
        _observation(4, multiple=21.0, velocity=-8.0, price=81.34),
        _observation(5, multiple=22.0, velocity=-6.0, price=81.36),
        _observation(6, multiple=23.0, velocity=-4.0, price=81.42),
        _observation(7, multiple=24.0, velocity=-2.0, price=81.43),
        _observation(8, multiple=24.5, velocity=-3.0, price=81.38),
        _observation(10, multiple=1.0, velocity=0.0, price=81.38),
    ]
    source = [_episode_row(index) for index in (*range(9), 10)]
    complete, opened = replay_mcl_shock_episodes(
        list(zip(source, rows, strict=True)),
        resets=[
            {
                "at_utc": START + timedelta(seconds=9),
                "reasons": ["stage112_minute_release:quiet_spine_loss"],
            }
        ],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )

    assert opened is None
    assert len(complete) == 1
    episode = complete[0]
    phases = [row["decision"]["phase"] for row in episode["transitions"]]
    assert episode["maximum_level"] == "REGIME_20X_PLUS"
    assert episode["shock_direction"] == -1
    assert episode["reached_tradeable_12x"] is True
    assert episode["reached_regime_20x"] is True
    assert "CREST_CONFIRMED" in phases
    assert "CONTINUATION" in phases
    assert episode["terminal"]["reasons"] == [
        "stage112_minute_release:quiet_spine_loss"
    ]
    assert episode["outcomes_exposed"] is False
    assert episode["submitted_orders"] == 0


def test_attention_only_episode_never_enters_the_ten_x_cohort() -> None:
    observations = [
        _observation(0, multiple=5.1, velocity=-1.0, price=80.0),
        _observation(2, multiple=1.0, velocity=0.0, price=80.0),
    ]
    source = [_episode_row(index) for index in (0, 2)]
    complete, _opened = replay_mcl_shock_episodes(
        list(zip(source, observations, strict=True)),
        resets=[{"at_utc": START + timedelta(seconds=1), "reasons": ["gap"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )

    assert complete == []


def test_external_reset_precedes_same_second_fresh_shock() -> None:
    first = [_observation(0, multiple=10.1, velocity=-2.0, price=80.0)]
    second = [
        _observation(20, multiple=20.1, velocity=2.0, price=80.2),
        _observation(21, multiple=21.0, velocity=3.0, price=80.3),
    ]
    observations = [*first, *second]
    source = [_episode_row(index) for index in (0, 20, 21)]
    complete, opened = replay_mcl_shock_episodes(
        list(zip(source, observations, strict=True)),
        resets=[{"at_utc": START + timedelta(seconds=20), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )

    assert len(complete) == 1
    assert complete[0]["shock_direction"] == -1
    assert opened is not None
    assert opened["started_at_utc"] == (START + timedelta(seconds=20)).isoformat()
    assert opened["shock_direction"] == 1
    assert opened["maximum_level"] == "REGIME_20X_PLUS"


def test_shock_episode_ledger_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    observations = [
        _observation(0, multiple=10.1, velocity=-2.0, price=80.0),
        _observation(2, multiple=1.0, velocity=0.0, price=79.9),
    ]
    source = [_episode_row(index) for index in (0, 2)]
    episodes, _opened = replay_mcl_shock_episodes(
        list(zip(source, observations, strict=True)),
        resets=[{"at_utc": START + timedelta(seconds=1), "reasons": ["release"]}],
        generation=_episode_generation(),
        rows=source,
        bars=_episode_bars(),
    )
    ledger = LiveCalibrationLedger(tmp_path / "shock.jsonl")
    episode = episodes[0]
    for _ in range(2):
        ledger.checkpoint(
            evaluation_as_of=episode["terminal_at_utc"],
            strategy_id=MCL_SHOCK_CREST_VERSION,
            strategy_version=MCL_SHOCK_ACCUMULATOR_VERSION,
            trading_date="2026-08-04",
            session="MCL_SHOCK",
            status="EVALUATED",
            evidence=episode,
            recorded_at=START + timedelta(seconds=10),
        )

    frozen = mcl_shock_episodes(tuple(ledger.records()))
    assert len(frozen) == 1
    assert frozen[0]["authority"] == MCL_SHOCK_ACCUMULATOR_AUTHORITY
    cohort = mcl_shock_cohort(frozen, _episode_generation()["cohort_gate"])
    assert cohort["complete_episodes"] == 1
    assert cohort["verdict"] == "ACCUMULATE"


def test_stage113_generation_and_service_remain_immutable() -> None:
    generation = load_mcl_shock_generation(MCL_SHOCK_GENERATION_PATH)
    root = Path(__file__).resolve().parents[1]
    service = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset.service"
    ).read_text()
    timer = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset.timer"
    ).read_text()
    accumulator = (
        root / "tradebot/research/mcl_shock_accumulator.py"
    ).read_text()

    assert generation["generation_id"] == (
        "fd2dd70f9238a0ea423d9e7fe4c7317ca80aab93ba79488ab163223cfe6b3756"
    )
    assert generation["frozen_levels"] == {
        "attention": 5.0,
        "major": 10.0,
        "tradeable": 12.0,
        "regime": 20.0,
    }
    assert service.count("ExecStart=") == 2
    assert "-m tradebot.research.mcl_predictive_accumulator" in service
    assert "-m tradebot.research.mcl_shock_accumulator" in service
    assert "MCL_SHOCK_LEDGER=" in service
    assert "Unit=tradebot-mcl-predictive-onset.service" in timer
    assert "MCL_LIVE_SELECTION_PATH" not in accumulator


def test_stage114_generation_owns_one_successor_observer_service() -> None:
    root = Path(__file__).resolve().parents[1]
    generation = _load_immutable_predecessor(
        MCL_SHOCK_WAVE_GENERATION_PATH, root
    )
    service = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset-stage114.service"
    ).read_text()
    timer = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset-stage114.timer"
    ).read_text()

    assert generation["generation_id"] == (
        "39d5d311561ffce71a3f88551a53a61fc905ceff108dae51e38346165ba5537d"
    )
    assert generation["state_law"]["attention_directionless_below_multiple"] == 10.0
    assert generation["state_law"]["handoff_resets_inner_crest_state"] is True
    assert service.count("ExecStart=") == 2
    assert "-m tradebot.research.mcl_predictive_accumulator" in service
    assert "-m tradebot.research.mcl_shock_wave_accumulator" in service
    assert "-m tradebot.research.mcl_shock_accumulator" not in service
    assert "MCL_SHOCK_WAVE_LEDGER=" in service
    assert "Unit=tradebot-mcl-predictive-onset-stage114.service" in timer


def test_stage114_runtime_successor_preserves_law_and_inherits_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = "backtests/mcl/mcl_v18_shock_stage112_live_source_shadow.json"
    digest = hashlib.sha256((root / artifact).read_bytes()).hexdigest()
    selected = {
        "selection_id": "1" * 64,
        "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
        "authority": "selected_live_bounded_canary",
        "selected_at_utc": "2026-08-05T10:00:00+00:00",
    }
    predictive = {
        "generation_id": "2" * 64,
        "selection_id": selected["selection_id"],
        "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
    }
    generation = build_mcl_shock_wave_successor_generation(
        repository_root=root,
        selected=selected,
        selection_path=artifact,
        selection_file_sha256=digest,
        capital_plan_id="3" * 64,
        portfolio_generation_path=artifact,
        portfolio_generation_sha256=digest,
        predictive=predictive,
        predictive_path=artifact,
        predictive_file_sha256=digest,
        inherited_episode_ids=["4" * 64, "5" * 64],
        generated_at=datetime(2026, 8, 5, 10, 1, tzinfo=timezone.utc),
        successor_reason="sliding broker bar-prefix replay normalization",
    )

    validated = validate_mcl_shock_wave_successor_generation(
        generation, repository_root=root
    )
    predecessor = _load_immutable_predecessor(
        MCL_SHOCK_WAVE_GENERATION_PATH, root
    )

    assert validated["state_law"] == predecessor["state_law"]
    assert validated["episode_law"] == predecessor["episode_law"]
    assert validated["cohort_gate"] == predecessor["cohort_gate"]
    assert validated["selection_id"] == selected["selection_id"]
    assert validated["predictive_generation_id"] == predictive["generation_id"]
    assert validated["inherited_prefix"]["episode_count"] == 2
    assert validated["registered_at_utc"] == validated["eligible_start_utc"]
    assert validated["outcomes_exposed"] is False
    assert validated["submitted_orders"] == 0
    assert validated["pre_outcome_basis"]["selection_successor_reason"] == (
        "sliding broker bar-prefix replay normalization"
    )
    accumulator = root / "tradebot/research/mcl_shock_wave_accumulator.py"
    assert validated["artifacts"]["wave_accumulator_owner"]["sha256"] == (
        hashlib.sha256(accumulator.read_bytes()).hexdigest()
    )


def test_stage114_runtime_service_uses_one_validated_generation_pointer() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset-runtime.service"
    ).read_text()
    timer = (
        root / "deploy/systemd/tradebot-mcl-predictive-onset-runtime.timer"
    ).read_text()

    assert service.count("ExecStart=") == 2
    assert service.count("ExecStartPre=") == 1
    assert service.count("db/calibration/mcl_shock_wave_generation.json") == 3
    assert "mcl_authority_bound_shock_waves_stage114_preregistration" not in service
    assert "-m tradebot.research.mcl_predictive_accumulator" in service
    assert "-m tradebot.research.mcl_shock_wave_accumulator" in service
    assert "MCL_STAGE131_COVERAGE=" in service
    assert "IBKR_READONLY=1" in service
    assert "Unit=tradebot-mcl-predictive-onset-runtime.service" in timer


def test_stage114_successor_uses_the_current_immutable_generation(
    tmp_path: Path,
) -> None:
    generation_id = "6" * 64
    immutable = (
        tmp_path
        / "db/calibration/shock_wave_generations"
        / f"{generation_id}.json"
    )
    current = tmp_path / "db/calibration/mcl_shock_wave_generation.json"
    immutable.parent.mkdir(parents=True)
    immutable.write_text("{}")
    current.write_text(f'{{"generation_id":"{generation_id}"}}')

    assert _predecessor_path(tmp_path, None) == immutable.relative_to(tmp_path)


def test_stage114_historical_predecessor_does_not_rehash_current_owner(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    generation = _load_immutable_predecessor(
        MCL_SHOCK_WAVE_GENERATION_PATH, root
    )
    generation["artifacts"]["wave_state_owner"]["sha256"] = "0" * 64
    body = {key: value for key, value in generation.items() if key != "generation_id"}
    generation["generation_id"] = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(json.dumps(generation))
    assert _load_immutable_predecessor(predecessor, tmp_path)["generation_id"] == (
        generation["generation_id"]
    )


def _reset_bars(stamps: tuple[datetime, ...]) -> dict[str, dict[datetime, OhlcvBar]]:
    return {
        symbol: {
            stamp: OhlcvBar(stamp, 80.0, 80.01, 79.99, 80.0, 10.0)
            for stamp in stamps
        }
        for symbol in ("CL", "MCL")
    }


def test_stage113_maintenance_resets_before_reopen_not_at_first_returning_bar() -> None:
    maintenance = datetime(2026, 8, 4, 21, 0, tzinfo=timezone.utc)
    resets = _minute_resets(
        _reset_bars(
            (
                maintenance - timedelta(minutes=1),
                maintenance,
                maintenance + timedelta(hours=1, minutes=1),
            )
        ),
        contract_key="202608",
    )

    assert resets == [
        {"at_utc": maintenance, "reasons": ["stage112_maintenance"]}
    ]


def test_stage113_unscheduled_gap_resets_at_first_missing_minute() -> None:
    first = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)
    resets = _minute_resets(
        _reset_bars((first, first + timedelta(minutes=5))),
        contract_key="202608",
    )

    assert resets == [
        {
            "at_utc": first + timedelta(minutes=1),
            "reasons": ["unscheduled_minute_gap"],
        }
    ]


def test_stage113_slow_clock_never_bridges_maintenance() -> None:
    maintenance = datetime(2026, 8, 4, 21, 0, tzinfo=timezone.utc)
    stamps = tuple(
        maintenance - timedelta(minutes=index) for index in range(17, -1, -1)
    ) + (maintenance + timedelta(hours=1, minutes=1),)
    bars = _reset_bars(stamps)

    context = _slow_context(
        bars,
        {symbol: sorted(bars[symbol]) for symbol in ("CL", "MCL")},
        symbol="MCL",
        when=maintenance + timedelta(hours=1, minutes=1, seconds=15),
        price=80.0,
    )

    assert context is None


def test_stage113_unconditional_v18_raw_turn_is_an_episode_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawTurnLifecycle:
        def __init__(self) -> None:
            self.count = 0

        def update(self, _minute: object) -> SimpleNamespace:
            self.count += 1
            decision = SimpleNamespace(phase="RAW_TURN") if self.count == 2 else None
            return SimpleNamespace(decision=decision)

    monkeypatch.setattr(
        "tradebot.research.mcl_shock_accumulator.MclTwoSpeedAuctionLifecycle",
        RawTurnLifecycle,
    )
    first = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)

    assert _minute_resets(
        _reset_bars((first, first + timedelta(minutes=1))),
        contract_key="202608",
    ) == [
        {
            "at_utc": first + timedelta(minutes=1),
            "reasons": ["stage112_v18_raw_turn"],
        }
    ]
