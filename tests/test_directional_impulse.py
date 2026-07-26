from datetime import datetime, timedelta

import pytest

from tradebot.backtest.models import Bar
from tradebot.backtest.spot_context import spot_signal_warmup_days_from_strategy
from tradebot.engines.directional_impulse import (
    DIRECTIONAL_IMPULSE_WARMUP_BARS,
    DirectionalImpulseAdmissionPolicy,
    DirectionalImpulseEngine,
    DirectionalTurnPolicy,
)
from tradebot.research.evidence import (
    XSP_DIRECTIONAL_TURN_SCHEMA,
    xsp_directional_turn_census,
)
from tradebot.spot.entry_control import SpotEntryControlPlan
from tradebot.spot.lifecycle import SpotExcursionPolicy, SpotExcursionState
from tradebot.spot_engine import SpotSignalEvaluator


def _bar(index: int, close: float, *, spread: float = 0.2) -> Bar:
    ts = datetime(2026, 7, 20, 13, 30) + timedelta(minutes=5 * index)
    return Bar(ts, close, close + spread, close - spread, close, 1_000.0)


def test_directional_impulse_is_causal_multihorizon_and_symmetric() -> None:
    up = DirectionalImpulseEngine(horizons=(1, 2, 3))
    up_rows = [
        up.update(
            high=bar.high,
            low=bar.low,
            close=bar.close,
            session_key=bar.ts.date(),
        )
        for bar in (_bar(i, 100.0 + i) for i in range(4))
    ]
    assert up_rows[-1].ready
    assert up_rows[-1].direction == "up"
    assert up_rows[-1].coherence == 1.0
    assert all(row.slope_pct_per_bar > 0.0 for row in up_rows[-1].horizons)
    frozen = up_rows[-1].as_payload()

    down = DirectionalImpulseEngine(horizons=(1, 2, 3))
    down_row = None
    for bar in (_bar(i, 103.0 - i) for i in range(4)):
        down_row = down.update(
            high=bar.high,
            low=bar.low,
            close=bar.close,
            session_key=bar.ts.date(),
        )
    assert down_row is not None
    assert down_row.direction == "down"
    assert down_row.coherence == 1.0
    assert all(row.slope_pct_per_bar < 0.0 for row in down_row.horizons)

    up.update(high=105.0, low=95.0, close=100.0, session_key=datetime(2026, 7, 20).date())
    assert up_rows[-1].as_payload() == frozen


def test_elapsed_horizons_respect_scheduled_overnight_break() -> None:
    start = datetime(2026, 7, 20, 3, 30)
    timestamps = [
        start + timedelta(minutes=5 * index)
        for index in range(73)
        if (start + timedelta(minutes=5 * index)).time()
        not in (datetime(2026, 7, 20, 3, 50).time(), datetime(2026, 7, 20, 3, 55).time())
    ]
    elapsed = DirectionalImpulseEngine(
        horizons=(24, 48, 72),
        bar_duration=timedelta(minutes=5),
    )
    counted = DirectionalImpulseEngine(horizons=(24, 48, 72))
    for index, ts in enumerate(timestamps):
        close = 100.0 + index
        elapsed_snapshot = elapsed.update(
            ts=ts,
            high=close + 0.1,
            low=close - 0.1,
            close=close,
        )
        counted_snapshot = counted.update(
            high=close + 0.1,
            low=close - 0.1,
            close=close,
        )

    assert len(timestamps) == 71
    assert elapsed_snapshot.ready
    assert not counted_snapshot.ready
    longest = elapsed_snapshot.horizons[-1]
    assert longest.bars == 72
    assert longest.elapsed_minutes == 360.0
    assert longest.observations == 71
    assert longest.anchor_lag_minutes == 0.0
    assert longest.slope_pct_per_bar == pytest.approx(longest.return_pct / 72.0)


def test_observational_impulse_reserves_its_longest_horizon() -> None:
    engine = DirectionalImpulseEngine()
    assert engine.warmup_bars == DIRECTIONAL_IMPULSE_WARMUP_BARS == 25
    assert (
        spot_signal_warmup_days_from_strategy(
            strategy={
                "entry_signal": "orb",
                "signal_bar_size": "5 mins",
                "signal_use_rth": True,
                "regime_mode": "off",
            },
            default_signal_bar_size="5 mins",
            default_signal_use_rth=True,
        )
        >= 2
    )


def test_directional_turn_census_replays_the_production_sensor() -> None:
    result = xsp_directional_turn_census(
        [_bar(index, 100.0 + abs(12 - index) * 0.1) for index in range(28)],
        source_fingerprint="fixture",
        include_session_ledger=True,
    )

    assert result["schema"] == XSP_DIRECTIONAL_TURN_SCHEMA
    assert result["contract"]["authority"] == "observation_only"
    assert result["contract"]["direction_owner"] == "xsp_native"
    assert result["coverage"]["complete_turn_window_sessions"] == 1
    assert result["coverage"]["events_below_required_horizons"] == 0
    assert len(result["sessions"]) == 1
    event = result["sessions"][0]["events"][0]
    assert event["forward_paths"]["1"]["observations"] == 1
    assert event["forward_paths"]["1"]["directed_mfe_points"] >= 0.0
    assert event["forward_paths"]["1"]["directed_mae_points"] >= 0.0


def test_opening_edge_policy_uses_causal_bar_close_clock() -> None:
    turn = DirectionalTurnPolicy().as_payload()
    admission = DirectionalImpulseAdmissionPolicy().as_payload()

    assert turn["start_et"] == "09:35"
    assert turn["end_et"] == "11:50"
    assert admission["start_minute_et"] == 9 * 60 + 35
    assert admission["core_end_minute_et"] == 11 * 60 + 20
    assert admission["late_up_end_minute_et"] == 11 * 60 + 30


def test_entry_control_plan_centralizes_source_permissions_and_order() -> None:
    plan = SpotEntryControlPlan.from_sources(
        strategy={
            "entry_signal": "ema",
            "regime_mode": "off",
            "regime_ema_preset": "8/21",
            "regime2_mode": "supertrend",
            "regime2_bar_size": "30 mins",
            "directional_spot": {
                "up": {"action": "BUY", "qty": 1},
                "down": {"action": "", "qty": 1},
            },
            "spot_entry_policy": "slope_tr_guard",
            "spot_dual_branch_enabled": True,
            "spot_branch_a_min_signed_slope_pct": 0.01,
            "regime2_apply_to": "longs",
            "regime2_bear_entry_mode": "supertrend",
            "regime2_bear_takeover_mode": "riskoff",
            "regime2_crash_block_longs": True,
        },
        filters={
            "entry_start_hour_et": 9,
            "entry_end_hour_et": 12,
            "shock_gate_mode": "detect",
            "ema_spread_min_pct": 0.1,
            "ratsv_enabled": True,
        },
        bar_size="5 mins",
    )
    assert plan.source == "ema"
    assert plan.source_gates == ("dual_branch", "branch_slope", "ratsv")
    assert plan.primary_regime == "off"
    assert plan.confirmation_regime == "supertrend"
    assert plan.confirmation_scope == "longs"
    assert plan.bear_takeover == "supertrend"
    assert plan.bear_takeover_scope == "riskoff"
    assert plan.shock_gate == "detect"
    assert plan.signal_filters == ("time", "permission")
    assert plan.allowed_directions == ("up",)
    assert plan.graph_entry_policy == "slope_tr_guard"
    assert plan.directional_impulse == "observe"
    assert plan.regime_gates.active_gates() == ("crash",)
    assert plan.as_payload()["observation_modes"] == {
        "directional_impulse": "observe",
        "fundamental_pressure": "off",
    }
    assert plan.as_payload()["order"] == [
        "source",
        "source_gates",
        "primary_regime",
        "regime2",
        "bear_takeover",
        "regime_entry_gates",
        "signal_filters",
        "direction_mapping",
        "lifecycle",
        "graph_entry_policy",
    ]


def test_fundamental_pressure_is_explicit_observation_without_entry_authority() -> None:
    plan = SpotEntryControlPlan.from_sources(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "2/3",
            "fundamental_pressure_mode": "observe",
        },
        filters=None,
        bar_size="5 mins",
    )

    assert plan.source == "ema"
    assert plan.fundamental_pressure == "observe"
    assert plan.observations == (
        "directional_impulse",
        "fundamental_pressure",
    )
    assert "fundamental_pressure" not in plan.source_gates


def test_explicit_primary_regime_off_cannot_override_entry_source() -> None:
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "2/3",
            "ema_entry_mode": "trend",
            "regime_mode": "off",
            "regime_ema_preset": "2/3",
            "regime_bar_size": "30 mins",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
        regime_bars=[_bar(index, 200.0 - index) for index in range(30)],
    )
    snap = None
    for index in range(30):
        snap = evaluator.update_signal_bar(_bar(index, 100.0 + index))
    assert snap is not None
    assert snap.entry_dir == "up"
    assert snap.entry_control_trace()["plan"]["confirmations"][
        "primary_regime"
    ] == "off"
    assert not any(
        control.startswith("primary_regime:")
        for control in snap.entry_controls
    )


def test_volatility_changes_conviction_but_cannot_choose_direction() -> None:
    engine = DirectionalImpulseEngine(horizons=(1, 2, 3))
    snap = None
    for index, spread in enumerate((0.1, 0.5, 1.0, 2.0)):
        bar = _bar(index, 100.0, spread=spread)
        snap = engine.update(
            high=bar.high,
            low=bar.low,
            close=bar.close,
            session_key=bar.ts.date(),
        )
    assert snap is not None and snap.ready
    assert snap.direction is None
    assert snap.abstain_reason == "weak_direction"
    assert snap.atr_ratio is not None and snap.atr_ratio > 1.0


def test_rth_session_boundary_resets_impulse_horizons() -> None:
    engine = DirectionalImpulseEngine(horizons=(1, 2))
    first_day = datetime(2026, 7, 20).date()
    second_day = datetime(2026, 7, 21).date()
    for close in (100.0, 101.0, 102.0):
        snap = engine.update(
            high=close + 0.1,
            low=close - 0.1,
            close=close,
            session_key=first_day,
        )
    assert snap.ready
    reset = engine.update(
        high=110.1,
        low=109.9,
        close=110.0,
        session_key=second_day,
    )
    assert not reset.ready
    assert reset.horizons == ()


def test_shared_evaluator_exposes_identical_impulse_in_lifecycle_trace() -> None:
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "2/3",
            "ema_entry_mode": "trend",
            "regime_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
        naive_ts_mode="utc",
    )
    snap = None
    for index in range(30):
        snap = evaluator.update_signal_bar(_bar(index, 100.0 + (0.1 * index)))
    assert snap is not None
    assert snap.directional_impulse is not None
    assert snap.directional_impulse.direction == "up"
    assert snap.entry_control_trace()["plan"]["source"] == "ema"
    assert snap.entry_control_trace()["plan"]["observations"] == [
        "directional_impulse"
    ]
    assert (
        snap.lifecycle_trace()["directional_impulse"]
        == snap.directional_impulse.as_payload()
    )
    assert (
        snap.entry_context()["directional_impulse"]
        == snap.directional_impulse.as_payload()
    )
    assert snap.entry_context()["signal_bar_ts"] == snap.bar_ts.isoformat()
    assert snap.signal.entry_dir == "up"
    assert snap.entry_dir == "up"
    assert snap.directional_impulse.conviction == pytest.approx(
        snap.lifecycle_trace()["directional_impulse"]["conviction"]
    )


def test_directional_turn_requires_three_horizons_and_reports_full_warmup() -> None:
    engine = DirectionalImpulseEngine(
        horizons=(1, 3, 6, 12, 24),
        bar_duration=timedelta(minutes=5),
        turn_policy=DirectionalTurnPolicy(),
    )
    snapshots = []
    for index, close in enumerate(
        (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 104.0, 102.0, 100.0)
    ):
        bar = _bar(index, close, spread=0.1)
        snapshots.append(
            engine.update(
                ts=bar.ts,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                session_key=bar.ts.date(),
            )
        )

    assert all(
        snap.turn_event is None
        for snap in snapshots
        if snap.observed_horizons < 3
    )
    first_ready = next(snap for snap in snapshots if snap.turn_ready)
    assert first_ready.observed_horizons == 3
    assert first_ready.required_turn_horizons == 3
    assert not first_ready.ready
    assert any(snap.turn_event == "down" for snap in snapshots)
    assert not snapshots[-1].ready


def test_directional_turn_observer_can_be_disabled_centrally() -> None:
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "2/3",
            "ema_entry_mode": "trend",
            "regime_mode": "off",
            "directional_impulse_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
    )
    snap = evaluator.update_signal_bar(_bar(0, 100.0))
    assert snap is not None
    assert snap.directional_impulse is None
    assert snap.entry_control_trace()["plan"]["observations"] == []


def test_directional_turn_can_own_the_normal_entry_contract() -> None:
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "directional_impulse",
            "regime_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
    )
    snapshots = [
        evaluator.update_signal_bar(_bar(index, close, spread=0.1))
        for index, close in enumerate(
            (
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                104.0,
                102.0,
                100.0,
            )
        )
    ]
    turn = next(
        snap
        for snap in snapshots
        if snap is not None and snap.entry_dir in ("up", "down")
    )

    assert turn.entry_dir == turn.directional_impulse.turn_event
    assert turn.signal.entry_dir == turn.entry_dir
    assert turn.entry_source == "directional_impulse"
    assert turn.entry_proposed_dir == turn.entry_dir
    assert turn.entry_controls[0] == "directional_impulse:turn"
    assert turn.entry_control_trace()["plan"]["observations"] == []
    quiet = snapshots[-1]
    assert quiet is not None
    assert quiet.entry_proposed_dir is None
    assert quiet.lifecycle_inputs()["signal_source_dir"] == "down"


def test_directional_admission_preserves_raw_turn_and_central_trace() -> None:
    strategy = {
        "entry_signal": "directional_impulse",
        "regime_mode": "off",
        "directional_impulse_admission": {
            "mode": "opening_edge",
            "atr_velocity_max": 0.1,
            "down_retrace_min": 1.0,
        },
    }
    evaluator = SpotSignalEvaluator(
        strategy=strategy,
        filters=None,
        bar_size="5 mins",
        use_rth=True,
    )
    snap = None
    for index, close in enumerate(
        (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 104.0)
    ):
        snap = evaluator.update_signal_bar(_bar(index, close, spread=0.1))

    assert snap is not None
    assert snap.entry_proposed_dir == "down"
    assert snap.entry_dir is None
    assert snap.entry_blocked_by == "directional_impulse_admission"
    assert snap.lifecycle_inputs()["signal_source_dir"] == "down"
    assert "directional_impulse_admission:block:atr_velocity" in snap.entry_controls

    policy = DirectionalImpulseAdmissionPolicy.from_mapping(
        {
            "mode": "opening_edge",
            "atr_velocity_max": 1.0,
            "down_retrace_min": 1.0,
        }
    )
    assert policy is not None
    assert policy.allows(
        direction="down",
        minute_et=605,
        atr_velocity=snap.directional_impulse.atr_velocity_pct,
        retrace_atr=snap.directional_impulse.retrace_atr,
        coherence=snap.directional_impulse.coherence,
    ) == (True, "core")
    assert (
        SpotEntryControlPlan.from_sources(
            strategy={
                **strategy,
                "directional_impulse_admission": policy.as_payload(),
            },
            filters=None,
            bar_size="5 mins",
        ).source_gates
        == ("directional_impulse_admission",)
    )


def test_directional_turn_uses_same_timeframe_ema_confirmation() -> None:
    closes = (100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 104.0)
    raw = SpotSignalEvaluator(
        strategy={
            "entry_signal": "directional_impulse",
            "regime_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
    )
    confirmed = SpotSignalEvaluator(
        strategy={
            "entry_signal": "directional_impulse",
            "regime_mode": "ema",
            "regime_ema_preset": "2/4",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
    )

    raw_snap = confirmed_snap = None
    for index, close in enumerate(closes):
        bar = _bar(index, close, spread=0.1)
        raw_snap = raw.update_signal_bar(bar)
        confirmed_snap = confirmed.update_signal_bar(bar)

    assert raw_snap is not None and raw_snap.entry_dir == "down"
    assert confirmed_snap is not None
    assert confirmed_snap.entry_proposed_dir == "down"
    assert confirmed_snap.signal.regime_dir == "up"
    assert confirmed_snap.entry_dir is None
    assert confirmed_snap.entry_blocked_by == "primary_regime"
    assert "primary_regime:block" in confirmed_snap.entry_controls
    assert confirmed_snap.lifecycle_inputs()["signal_source_dir"] == "down"
    assert confirmed_snap.lifecycle_inputs()["signal_entry_dir"] is None


@pytest.mark.parametrize(
    ("direction", "high", "low", "expected_stop"),
    (("up", 101.2, 99.5, 100.7), ("down", 100.5, 98.8, 99.3)),
)
def test_excursion_ratchet_is_symmetric_and_cannot_reprice_its_source_bar(
    direction: str,
    high: float,
    low: float,
    expected_stop: float,
) -> None:
    policy = SpotExcursionPolicy(
        initial_stop_atr=1.0,
        trail_activate_atr=0.5,
        trail_distance_atr=0.25,
        breakeven_atr=0.5,
    )
    opened = SpotExcursionState.open(
        policy=policy,
        direction=direction,
        entry_price=100.0,
        entry_atr=2.0,
    )
    advanced, reason = opened.advance(policy=policy, high=high, low=low)

    assert opened.stop_price == (98.0 if direction == "up" else 102.0)
    assert advanced.stop_price == pytest.approx(expected_stop)
    assert advanced.stop_reason == "trail_stop"
    assert advanced.bars_held == 1
    assert reason is None


def test_excursion_profit_lock_does_not_require_an_initial_stop() -> None:
    policy = SpotExcursionPolicy(
        trail_activate_atr=2.0,
        trail_distance_atr=1.0,
    )
    opened = SpotExcursionState.open(
        policy=policy,
        direction="up",
        entry_price=100.0,
        entry_atr=2.0,
    )
    assert policy.enabled
    assert opened.stop_price is None
    assert opened.stop_reason is None

    warming, reason = opened.advance(policy=policy, high=103.0, low=99.0)
    assert reason is None
    assert warming.stop_price is None

    locked, reason = warming.advance(policy=policy, high=105.0, low=102.0)
    assert reason is None
    assert locked.stop_price == pytest.approx(103.0)
    assert locked.stop_reason == "trail_stop"


def test_excursion_fizzle_and_max_hold_are_completed_bar_decisions() -> None:
    fizzle = SpotExcursionPolicy(
        initial_stop_atr=1.0,
        fizzle_bars=2,
        fizzle_mfe_atr=0.5,
        max_hold_bars=3,
    )
    state = SpotExcursionState.open(
        policy=fizzle,
        direction="up",
        entry_price=100.0,
        entry_atr=2.0,
    )
    state, reason = state.advance(policy=fizzle, high=100.2, low=99.8)
    assert reason is None
    state, reason = state.advance(policy=fizzle, high=100.3, low=99.9)
    assert reason == "fizzle"

    max_hold = SpotExcursionPolicy(initial_stop_atr=1.0, max_hold_bars=2)
    state = SpotExcursionState.open(
        policy=max_hold,
        direction="down",
        entry_price=100.0,
        entry_atr=2.0,
    )
    state, reason = state.advance(policy=max_hold, high=100.1, low=99.5)
    assert reason is None
    state, reason = state.advance(policy=max_hold, high=100.0, low=99.0)
    assert reason == "max_hold"
