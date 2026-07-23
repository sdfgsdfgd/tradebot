from __future__ import annotations

from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from tradebot.backtest.cache_ops.cli import main_resample
from tradebot.backtest.cli_utils import expected_cache_path
from tradebot.chart_data.history import (
    cache_data_revision,
    cache_path,
    ensure_offline_cached_window,
    read_cache,
    write_cache,
)
from tradebot.backtest.data import IBKRHistoricalData
from tradebot.backtest.models import Bar
from tradebot.research.spot_sweeps.market import SweepMarketData
from tradebot.research.spot_sweeps.cli import parse_spot_sweep_args
from tradebot.research.spot_sweeps.runtime import SpotSweepRuntime
from tradebot.research.spot_sweeps.support import _require_offline_cache_or_die


def _bar(ts: datetime, px: float) -> Bar:
    return Bar(ts=ts, open=px, high=px + 0.1, low=px - 0.1, close=px, volume=100.0)


def _touch_overlap_segments(tmp_path) -> None:
    seg1 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 1), time(0, 0)),
        datetime.combine(date(2025, 1, 1), time(23, 59)),
        "1 min",
        False,
    )
    seg2 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 2), time(0, 0)),
        datetime.combine(date(2025, 1, 2), time(23, 59)),
        "1 min",
        False,
    )
    seg1.parent.mkdir(parents=True, exist_ok=True)
    write_cache(seg1, [_bar(datetime(2025, 1, 1, 12, 0), 10.0)])
    write_cache(seg2, [_bar(datetime(2025, 1, 2, 12, 0), 11.0)])


def _touch_overlap_segments_dense_1min(tmp_path) -> None:
    seg1 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 1), time(0, 0)),
        datetime.combine(date(2025, 1, 1), time(23, 59)),
        "1 min",
        False,
    )
    seg2 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 2), time(0, 0)),
        datetime.combine(date(2025, 1, 2), time(23, 59)),
        "1 min",
        False,
    )
    seg1.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        seg1,
        [
            _bar(datetime(2025, 1, 1, 0, 0), 10.0),
            _bar(datetime(2025, 1, 1, 0, 1), 10.1),
            _bar(datetime(2025, 1, 1, 0, 2), 10.2),
            _bar(datetime(2025, 1, 1, 0, 3), 10.3),
        ],
    )
    write_cache(
        seg2,
        [
            _bar(datetime(2025, 1, 2, 0, 0), 11.0),
            _bar(datetime(2025, 1, 2, 0, 1), 11.1),
            _bar(datetime(2025, 1, 2, 0, 2), 11.2),
            _bar(datetime(2025, 1, 2, 0, 3), 11.3),
        ],
    )


def _require_sweeps_offline_cache(
    *,
    cache_dir,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str = "1 min",
    use_rth: bool = False,
    cache_policy: str = "strict",
) -> None:
    data = IBKRHistoricalData()
    try:
        _require_offline_cache_or_die(
            data=data,
            cache_dir=cache_dir,
            symbol="SLV",
            exchange=None,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=bar_size,
            use_rth=bool(use_rth),
            cache_policy=cache_policy,
        )
    finally:
        try:
            data.disconnect()
        except Exception:
            pass


def test_cache_data_revision_changes_with_market_tape(tmp_path) -> None:
    before = cache_data_revision(tmp_path)
    _touch_overlap_segments(tmp_path)
    populated = cache_data_revision(tmp_path)
    path = next((tmp_path / "SLV").glob("*.csv"))
    write_cache(path, [_bar(datetime(2025, 1, 1, 12, 0), 1234.0)])

    assert populated != before
    assert cache_data_revision(tmp_path) != populated


def test_sweeps_offline_preflight_accepts_overlap_coverage(tmp_path) -> None:
    _touch_overlap_segments(tmp_path)
    _require_sweeps_offline_cache(
        cache_dir=tmp_path,
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
    )


def test_rth_four_hour_coverage_allows_early_close_without_full_bar(tmp_path) -> None:
    start = datetime(2025, 11, 26)
    end = datetime(2025, 11, 28, 23, 59)
    path = cache_path(tmp_path, "TQQQ", start, end, "4 hours", True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(path, [_bar(datetime(2025, 11, 26, 16, 0), 10.0)])

    ok, _expected, resolved, missing, error = ensure_offline_cached_window(
        cache_dir=tmp_path,
        symbol="TQQQ",
        start=start,
        end=end,
        bar_size="4 hours",
        use_rth=True,
    )

    assert ok is True
    assert resolved == path
    assert missing == []
    assert error is None


def test_combo_parallel_worker_forces_strict_cache_policy(tmp_path) -> None:
    day = date(2025, 1, 2)
    path = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(day, time(0, 0)),
        datetime.combine(day, time(23, 59)),
        "1 day",
        False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(path, [_bar(datetime.combine(day, time(0, 0)), 10.0)])
    args = parse_spot_sweep_args(
        [
            "--symbol",
            "SLV",
            "--start",
            day.isoformat(),
            "--end",
            day.isoformat(),
            "--cache-dir",
            str(tmp_path),
            "--offline",
            "--cache-policy",
            "auto",
            "--combo-full-cartesian-worker",
            "0",
        ]
    )

    runtime = SpotSweepRuntime(args)
    try:
        assert runtime.cache_policy == "strict"
    finally:
        runtime.data.disconnect()
        if runtime.run_cfg_persistent_conn is not None:
            runtime.run_cfg_persistent_conn.close()


def test_sweeps_offline_preflight_auto_resamples_from_finer_cache(tmp_path) -> None:
    _touch_overlap_segments_dense_1min(tmp_path)
    _require_sweeps_offline_cache(
        cache_dir=tmp_path,
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
        bar_size="2 mins",
        cache_policy="auto",
    )
    dst = expected_cache_path(
        cache_dir=tmp_path,
        symbol="SLV",
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
        bar_size="2 mins",
        use_rth=False,
    )
    assert dst.exists()
    assert len(read_cache(dst)) > 0


def test_sweep_market_data_applies_auto_policy_to_every_requested_timeframe(
    tmp_path,
) -> None:
    _touch_overlap_segments_dense_1min(tmp_path)
    data = IBKRHistoricalData()
    runtime = SimpleNamespace(
        offline=True,
        data=data,
        cache_dir=tmp_path,
        cache_policy="auto",
        symbol="SLV",
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
        use_rth=False,
    )
    try:
        bars = SweepMarketData._bars(runtime, "2 mins")
    finally:
        data.disconnect()

    assert bars
    assert expected_cache_path(
        cache_dir=tmp_path,
        symbol="SLV",
        start_dt=runtime.start_dt,
        end_dt=runtime.end_dt,
        bar_size="2 mins",
        use_rth=False,
    ).exists()


def test_sweeps_offline_preflight_reports_missing_ranges(tmp_path) -> None:
    _touch_overlap_segments(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _require_sweeps_offline_cache(
            cache_dir=tmp_path,
            start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
            end_dt=datetime.combine(date(2025, 1, 3), time(23, 59)),
        )
    msg = str(exc.value)
    assert "missing=" in msg
    assert "2025-01-03" in msg


def test_sweeps_offline_preflight_auto_fetches_when_cache_missing(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    def _fake_load_or_fetch(
        self,
        symbol: str,
        exchange,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
        cache_dir,
    ):
        calls.append((str(symbol), str(bar_size), bool(use_rth)))
        path = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_cache(path, [_bar(datetime.combine(start.date(), time(12, 0)), 10.0)])
        return object()

    monkeypatch.setattr(IBKRHistoricalData, "load_or_fetch_bar_series", _fake_load_or_fetch)
    _require_sweeps_offline_cache(
        cache_dir=tmp_path,
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 1), time(23, 59)),
        bar_size="5 mins",
        cache_policy="auto",
    )
    assert calls == [("SLV", "5 mins", False)]
    expected = expected_cache_path(
        cache_dir=tmp_path,
        symbol="SLV",
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 1), time(23, 59)),
        bar_size="5 mins",
        use_rth=False,
    )
    assert expected.exists()


def test_cache_ops_resample_uses_overlap_stitching(tmp_path) -> None:
    seg1 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 1), time(0, 0)),
        datetime.combine(date(2025, 1, 1), time(23, 59)),
        "1 min",
        False,
    )
    seg2 = cache_path(
        tmp_path,
        "SLV",
        datetime.combine(date(2025, 1, 2), time(0, 0)),
        datetime.combine(date(2025, 1, 2), time(23, 59)),
        "1 min",
        False,
    )
    seg1.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        seg1,
        [
            _bar(datetime(2025, 1, 1, 0, 0), 10.0),
            _bar(datetime(2025, 1, 1, 0, 1), 10.1),
            _bar(datetime(2025, 1, 1, 0, 2), 10.2),
            _bar(datetime(2025, 1, 1, 0, 3), 10.3),
        ],
    )
    write_cache(
        seg2,
        [
            _bar(datetime(2025, 1, 2, 0, 0), 11.0),
            _bar(datetime(2025, 1, 2, 0, 1), 11.1),
            _bar(datetime(2025, 1, 2, 0, 2), 11.2),
            _bar(datetime(2025, 1, 2, 0, 3), 11.3),
        ],
    )

    stitched_src = expected_cache_path(
        cache_dir=tmp_path,
        symbol="SLV",
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
        bar_size="1 min",
        use_rth=False,
    )
    assert not stitched_src.exists()

    main_resample(
        [
            "--symbol",
            "SLV",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
            "--src-bar-size",
            "1 min",
            "--dst-bar-size",
            "2 mins",
            "--cache-dir",
            str(tmp_path),
        ]
    )

    assert not stitched_src.exists()
    dst = expected_cache_path(
        cache_dir=tmp_path,
        symbol="SLV",
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
        bar_size="2 mins",
        use_rth=False,
    )
    assert dst.exists()
    assert len(read_cache(dst)) > 0
