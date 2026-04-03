from __future__ import annotations

from tradebot.climate_router import ClimateDecision, DailyBar, DailyRegimeRouterEngine, RegimeRouterConfig, RollingClimateState, YearFeatures, bull_sovereign_entry_ok, classify_climate_v2, classify_climate_v3, classify_climate_v4, classify_rolling_climate_v4, classify_rolling_climate_v5, damage_positive_lock_hit, host_policy, regime_router_dwell_days, rolling_climate_states


def test_classify_climate_v2_bull_grind_low_vol() -> None:
    features = YearFeatures(
        year=2017,
        ret=1.10,
        maxdd=0.15,
        rv=0.31,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.60,
        efficiency=0.33,
        dd_frac_ge_10pct=0.04,
    )
    out = classify_climate_v2(features)
    assert out == ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")


def test_classify_climate_v2_positive_high_stress_transition() -> None:
    features = YearFeatures(
        year=2025,
        ret=0.34,
        maxdd=0.57,
        rv=0.70,
        atr_med=0.04,
        atr_mean=0.05,
        up_frac=0.57,
        efficiency=0.05,
        dd_frac_ge_10pct=0.45,
    )
    out = classify_climate_v2(features)
    assert out == ClimateDecision(climate="positive_high_stress_transition", chosen_host="hf_host")


def test_classify_climate_v2_negative_extreme_bear() -> None:
    features = YearFeatures(
        year=2022,
        ret=-0.80,
        maxdd=0.81,
        rv=0.97,
        atr_med=0.07,
        atr_mean=0.08,
        up_frac=0.44,
        efficiency=0.07,
        dd_frac_ge_10pct=0.99,
    )
    out = classify_climate_v2(features)
    assert out == ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")


def test_classify_climate_v2_negative_transition_bear() -> None:
    features = YearFeatures(
        year=2018,
        ret=-0.24,
        maxdd=0.58,
        rv=0.68,
        atr_med=0.04,
        atr_mean=0.05,
        up_frac=0.54,
        efficiency=0.03,
        dd_frac_ge_10pct=0.52,
    )
    out = classify_climate_v2(features)
    assert out == ClimateDecision(climate="negative_transition_bear", chosen_host="sma200")


def test_classify_climate_v3_negative_transition_bear_uses_defensive_host() -> None:
    features = YearFeatures(
        year=2018,
        ret=-0.24,
        maxdd=0.58,
        rv=0.68,
        atr_med=0.04,
        atr_mean=0.05,
        up_frac=0.54,
        efficiency=0.03,
        dd_frac_ge_10pct=0.52,
    )
    out = classify_climate_v3(features)
    assert out == ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v1")


def test_classify_climate_v4_negative_transition_bear_uses_drawdown_kill_host() -> None:
    features = YearFeatures(
        year=2018,
        ret=-0.24,
        maxdd=0.58,
        rv=0.68,
        atr_med=0.04,
        atr_mean=0.05,
        up_frac=0.54,
        efficiency=0.03,
        dd_frac_ge_10pct=0.52,
    )
    out = classify_climate_v4(features)
    assert out == ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v2")


def test_rolling_climate_states_respects_dwell_before_switch() -> None:
    days: list[DailyBar] = []
    # 126 calm positive days -> buyhold
    for i in range(126):
        px = 100.0 + (0.2 * i)
        days.append(DailyBar(ts=f"2024-01-{(i % 28) + 1:02d}", open=px, high=px * 1.01, low=px * 0.99, close=px))
    # 24 extreme negative days -> should only switch after dwell is satisfied
    base = days[-1].close
    for j in range(24):
        close = base * (0.90 ** (j + 1))
        days.append(DailyBar(ts=f"2024-07-{j + 1:02d}", open=close, high=close * 1.02, low=close * 0.98, close=close))

    states = rolling_climate_states(days, fast_window_days=21, slow_window_days=63, min_dwell_days=3)
    assert states
    assert states[0].active.chosen_host == "buyhold"
    assert states[-1].proposed.chosen_host == "hf_host"
    assert states[-1].active.chosen_host == "hf_host"


def test_host_policy_marks_hf_vs_host_managed() -> None:
    assert host_policy("hf_host").host_managed is False
    assert host_policy("buyhold").host_managed is True
    assert host_policy("bull_ma200_v1").host_managed is True


def test_daily_regime_router_emits_host_managed_for_buyhold() -> None:
    router = DailyRegimeRouterEngine(
        config=RegimeRouterConfig(enabled=True, fast_window_days=2, slow_window_days=4, min_dwell_days=1)
    )
    snap = None
    for idx, close in enumerate((100.0, 101.0, 102.0, 103.0, 101.0), start=1):
        snap = router.update_bar(
            ts=f"2025-01-0{idx}T23:55:00",
            open=close,
            high=close,
            low=close,
            close=close,
            hf_entry_dir="down",
        )
    assert snap is not None
    assert snap.ready is True
    assert snap.chosen_host == "buyhold"
    assert snap.host_managed is True
    assert snap.bull_sovereign_ok is False
    assert snap.effective_entry_dir == "up"


def test_classify_rolling_climate_v4_uses_fast_bull_recovery_when_hf_active() -> None:
    fast = YearFeatures(
        year=1,
        ret=0.25,
        maxdd=0.10,
        rv=0.25,
        atr_med=0.01,
        atr_mean=0.02,
        up_frac=0.60,
        efficiency=0.30,
        dd_frac_ge_10pct=0.0,
    )
    slow = YearFeatures(
        year=1,
        ret=0.15,
        maxdd=0.55,
        rv=0.65,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.55,
        efficiency=0.08,
        dd_frac_ge_10pct=0.50,
    )
    out = classify_rolling_climate_v4(
        fast_features=fast,
        slow_features=slow,
        active=ClimateDecision(climate="positive_high_stress_transition", chosen_host="hf_host"),
    )
    assert out == ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")


def test_classify_rolling_climate_v4_uses_fast_crash_override() -> None:
    fast = YearFeatures(
        year=1,
        ret=-0.25,
        maxdd=0.75,
        rv=0.90,
        atr_med=0.05,
        atr_mean=0.06,
        up_frac=0.30,
        efficiency=0.05,
        dd_frac_ge_10pct=0.90,
    )
    slow = YearFeatures(
        year=1,
        ret=0.10,
        maxdd=0.20,
        rv=0.30,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.22,
        dd_frac_ge_10pct=0.10,
    )
    out = classify_rolling_climate_v4(fast_features=fast, slow_features=slow, active=None)
    assert out == ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")


def test_classify_rolling_climate_v5_narrows_positive_stress_false_positive() -> None:
    crash = YearFeatures(
        year=1,
        ret=0.10,
        maxdd=0.05,
        rv=0.30,
        atr_med=0.01,
        atr_mean=0.02,
        up_frac=0.55,
        efficiency=0.20,
        dd_frac_ge_10pct=0.0,
    )
    fast = YearFeatures(
        year=1,
        ret=0.22,
        maxdd=0.21,
        rv=0.46,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.56,
        efficiency=0.16,
        dd_frac_ge_10pct=0.21,
    )
    slow = YearFeatures(
        year=1,
        ret=0.51,
        maxdd=0.31,
        rv=0.61,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.56,
        efficiency=0.14,
        dd_frac_ge_10pct=0.34,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")


def test_classify_rolling_climate_v5_uses_crash_sentinel() -> None:
    crash = YearFeatures(
        year=1,
        ret=-0.25,
        maxdd=0.35,
        rv=0.90,
        atr_med=0.05,
        atr_mean=0.06,
        up_frac=0.30,
        efficiency=0.10,
        dd_frac_ge_10pct=0.30,
    )
    fast = YearFeatures(
        year=1,
        ret=0.02,
        maxdd=0.18,
        rv=0.45,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.51,
        efficiency=0.02,
        dd_frac_ge_10pct=0.10,
    )
    slow = YearFeatures(
        year=1,
        ret=0.15,
        maxdd=0.22,
        rv=0.50,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.53,
        efficiency=0.08,
        dd_frac_ge_10pct=0.12,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="negative_extreme_bear", chosen_host="lf_defensive_long_v1")


def test_classify_rolling_climate_v5_keeps_hf_host_for_crash_when_slow_tape_is_bearish() -> None:
    crash = YearFeatures(
        year=1,
        ret=-0.25,
        maxdd=0.35,
        rv=0.90,
        atr_med=0.05,
        atr_mean=0.06,
        up_frac=0.30,
        efficiency=0.10,
        dd_frac_ge_10pct=0.30,
    )
    fast = YearFeatures(
        year=1,
        ret=0.02,
        maxdd=0.18,
        rv=0.45,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.51,
        efficiency=0.02,
        dd_frac_ge_10pct=0.10,
    )
    slow = YearFeatures(
        year=1,
        ret=-0.50,
        maxdd=0.65,
        rv=0.80,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.35,
        efficiency=0.10,
        dd_frac_ge_10pct=0.65,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")


def test_classify_rolling_climate_v5_uses_episode_crash_takeover() -> None:
    crash = YearFeatures(
        year=1,
        ret=-0.10,
        maxdd=0.22,
        rv=0.50,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.40,
        efficiency=0.08,
        dd_frac_ge_10pct=0.20,
    )
    fast = YearFeatures(
        year=1,
        ret=0.08,
        maxdd=0.16,
        rv=0.45,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.12,
        dd_frac_ge_10pct=0.12,
    )
    slow = YearFeatures(
        year=1,
        ret=0.20,
        maxdd=0.18,
        rv=0.50,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.15,
        dd_frac_ge_10pct=0.12,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="episode_crash_takeover", chosen_host="bull_ma200_v1")


def test_classify_rolling_climate_v5_uses_learned_pre_bear_handoff() -> None:
    crash = YearFeatures(
        year=1,
        ret=-0.05,
        maxdd=0.20,
        rv=0.55,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.48,
        efficiency=0.10,
        dd_frac_ge_10pct=0.12,
    )
    fast = YearFeatures(
        year=1,
        ret=-0.12,
        maxdd=0.23,
        rv=0.60,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.46,
        efficiency=0.09,
        dd_frac_ge_10pct=0.22,
    )
    slow = YearFeatures(
        year=1,
        ret=0.08,
        maxdd=0.22,
        rv=0.55,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.04,
        dd_frac_ge_10pct=0.18,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="pre_bear_handoff", chosen_host="hf_host")


def test_classify_rolling_climate_v5_uses_defensive_host_on_pre_bear_rebound() -> None:
    crash = YearFeatures(
        year=1,
        ret=0.25,
        maxdd=0.18,
        rv=0.95,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.30,
        dd_frac_ge_10pct=0.15,
    )
    fast = YearFeatures(
        year=1,
        ret=-0.15,
        maxdd=0.25,
        rv=0.75,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.52,
        efficiency=0.05,
        dd_frac_ge_10pct=0.45,
    )
    slow = YearFeatures(
        year=1,
        ret=0.05,
        maxdd=0.25,
        rv=0.60,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.52,
        efficiency=0.04,
        dd_frac_ge_10pct=0.45,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="pre_bear_handoff", chosen_host="lf_defensive_long_v1")


def test_classify_rolling_climate_v5_uses_damage_positive_lock() -> None:
    crash = YearFeatures(
        year=1,
        ret=0.05,
        maxdd=0.05,
        rv=0.30,
        atr_med=0.01,
        atr_mean=0.02,
        up_frac=0.55,
        efficiency=0.20,
        dd_frac_ge_10pct=0.0,
    )
    fast = YearFeatures(
        year=1,
        ret=0.08,
        maxdd=0.16,
        rv=0.45,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.55,
        efficiency=0.12,
        dd_frac_ge_10pct=0.12,
    )
    slow = YearFeatures(
        year=1,
        ret=0.18,
        maxdd=0.25,
        rv=0.55,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.54,
        efficiency=0.10,
        dd_frac_ge_10pct=0.35,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="damage_positive_lock", chosen_host="lf_defensive_long_v1")


def test_classify_rolling_climate_v5_uses_v1_for_negative_transition_bear() -> None:
    crash = YearFeatures(
        year=1,
        ret=-0.05,
        maxdd=0.10,
        rv=0.40,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.48,
        efficiency=0.08,
        dd_frac_ge_10pct=0.10,
    )
    fast = YearFeatures(
        year=1,
        ret=-0.08,
        maxdd=0.26,
        rv=0.60,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.46,
        efficiency=0.06,
        dd_frac_ge_10pct=0.30,
    )
    slow = YearFeatures(
        year=1,
        ret=-0.14,
        maxdd=0.34,
        rv=0.62,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.47,
        efficiency=0.07,
        dd_frac_ge_10pct=0.40,
    )
    out = classify_rolling_climate_v5(
        crash_features=crash,
        fast_features=fast,
        slow_features=slow,
        active=None,
    )
    assert out == ClimateDecision(climate="negative_transition_bear", chosen_host="lf_defensive_long_v1")




def test_regime_router_dwell_days_is_asymmetric() -> None:
    active = ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")
    crash = ClimateDecision(climate="negative_extreme_bear", chosen_host="hf_host")
    recovery = ClimateDecision(climate="bull_grind_low_vol", chosen_host="buyhold")
    assert regime_router_dwell_days(active=active, proposed=crash, base_dwell_days=10) == 1
    assert regime_router_dwell_days(active=crash, proposed=recovery, base_dwell_days=10) == 5


def test_bull_sovereign_entry_ok_requires_confident_bull_path() -> None:
    fast = YearFeatures(
        year=1,
        ret=0.22,
        maxdd=0.15,
        rv=0.50,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.56,
        efficiency=0.11,
        dd_frac_ge_10pct=0.20,
    )
    slow = YearFeatures(
        year=1,
        ret=0.30,
        maxdd=0.30,
        rv=0.60,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.54,
        efficiency=0.05,
        dd_frac_ge_10pct=0.35,
    )
    assert bull_sovereign_entry_ok(
        climate="bull_grind_low_vol",
        chosen_host="buyhold",
        fast_features=fast,
        slow_features=slow,
    )
    weak_slow = YearFeatures(
        year=1,
        ret=0.20,
        maxdd=0.27,
        rv=0.61,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.58,
        efficiency=0.06,
        dd_frac_ge_10pct=0.31,
    )
    weak_fast = YearFeatures(
        year=1,
        ret=0.17,
        maxdd=0.10,
        rv=0.55,
        atr_med=0.02,
        atr_mean=0.03,
        up_frac=0.53,
        efficiency=0.10,
        dd_frac_ge_10pct=0.25,
    )
    assert not bull_sovereign_entry_ok(
        climate="bull_grind_low_vol",
        chosen_host="buyhold",
        fast_features=weak_fast,
        slow_features=weak_slow,
    )


def test_damage_positive_lock_hit_requires_damaged_positive_tape() -> None:
    cfg = RegimeRouterConfig(
        enabled=True,
        fast_window_days=63,
        slow_window_days=84,
        min_dwell_days=10,
        damage_positive_lock_maxdd_min=0.24,
        damage_positive_lock_ret_max=0.20,
        damage_positive_lock_eff_max=0.10,
    )
    slow_bad = YearFeatures(
        year=1,
        ret=0.19,
        maxdd=0.25,
        rv=0.55,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.54,
        efficiency=0.10,
        dd_frac_ge_10pct=0.35,
    )
    slow_good = YearFeatures(
        year=1,
        ret=0.24,
        maxdd=0.18,
        rv=0.50,
        atr_med=0.03,
        atr_mean=0.04,
        up_frac=0.56,
        efficiency=0.10,
        dd_frac_ge_10pct=0.22,
    )
    assert damage_positive_lock_hit(slow_features=slow_bad, config=cfg)
    assert not damage_positive_lock_hit(slow_features=slow_good, config=cfg)
