from __future__ import annotations

from collections import deque

from tradebot.chart_data.realtime import (
    RealtimeChartData,
    resample,
    tape_bin_max,
    tape_bin_sum,
    tape_series,
)


def test_realtime_chart_data_coalesces_and_trims_mid_tape() -> None:
    data = RealtimeChartData(window_sec=10.0, retention_sec=20.0)
    data.record_mid(100.0, epsilon=0.01, ts=0.0)
    data.record_mid(100.005, epsilon=0.01, ts=1.0)
    data.record_mid(101.0, epsilon=0.01, ts=25.0)

    assert list(data.mid_samples) == [100.0, 101.0]
    assert list(data.mid_tape) == [(25.0, 101.0)]


def test_realtime_chart_data_tracks_volume_sources_and_fallbacks() -> None:
    data = RealtimeChartData(window_sec=10.0, retention_sec=20.0)
    assert data.cumulative_volume_delta({"rtVolume": 100.0}) is None
    assert data.cumulative_volume_delta({"rtVolume": 107.0}) == 7.0
    assert data.fallback_volume_delta(last_size=3.0, last_price=100.0) is None
    assert data.fallback_volume_delta(last_size=3.0, last_price=100.0) is None
    assert data.fallback_volume_delta(last_size=3.0, last_price=100.25) == 3.0


def test_realtime_chart_data_infers_flow_direction_from_price_then_imbalance() -> None:
    data = RealtimeChartData(window_sec=10.0, retention_sec=20.0)
    assert data.flow_direction(price=100.0, imbalance=-0.5, epsilon=0.01) == -1.0
    assert data.flow_direction(price=100.25, imbalance=-0.5, epsilon=0.01) == 1.0
    assert data.flow_direction(price=100.0, imbalance=0.5, epsilon=0.01) == -1.0
    assert data.flow_direction(price=None, imbalance=0.5, epsilon=0.01) == 1.0


def test_realtime_chart_resampling_and_bins_are_deterministic() -> None:
    tape = deque([(0.0, 10.0), (5.0, 20.0), (10.0, 30.0)])
    assert resample([0.0, 10.0], 3) == [0.0, 5.0, 10.0]
    assert tape_series(tape, width=2, start=0.0, end=10.0) == [20.0, 30.0]
    assert tape_bin_sum(tape, width=2, start=0.0, end=10.0) == [10.0, 50.0]
    assert tape_bin_max(tape, width=2, start=0.0, end=10.0) == [10.0, 30.0]
