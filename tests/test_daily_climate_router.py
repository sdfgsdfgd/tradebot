from __future__ import annotations

from tradebot.climate_router import ClimateDecision, DailyBar, RollingClimateState, YearFeatures, classify_climate_v2, classify_climate_v3, classify_climate_v4, rolling_climate_states


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
