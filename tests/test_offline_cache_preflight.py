from __future__ import annotations

from datetime import date, datetime, time

import pytest

from tradebot.backtest.cache_ops_lib import main_resample
from tradebot.backtest.cli_utils import expected_cache_path
from tradebot.backtest.data import IBKRHistoricalData, cache_path, read_cache, write_cache
from tradebot.backtest.models import Bar
from tradebot.backtest.multiwindow_helpers import preflight_offline_cache_or_die
from tradebot.backtest.run_backtests_spot_sweeps import _require_offline_cache_or_die


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


def _require_sweeps_offline_cache(*, cache_dir, start_dt: datetime, end_dt: datetime) -> None:
    data = IBKRHistoricalData()
    try:
        _require_offline_cache_or_die(
            data=data,
            cache_dir=cache_dir,
            symbol="SLV",
            exchange=None,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size="1 min",
            use_rth=False,
        )
    finally:
        try:
            data.disconnect()
        except Exception:
            pass


def test_sweeps_offline_preflight_accepts_overlap_coverage(tmp_path) -> None:
    _touch_overlap_segments(tmp_path)
    _require_sweeps_offline_cache(
        cache_dir=tmp_path,
        start_dt=datetime.combine(date(2025, 1, 1), time(0, 0)),
        end_dt=datetime.combine(date(2025, 1, 2), time(23, 59)),
    )


def test_multiwindow_offline_preflight_accepts_overlap_coverage(tmp_path) -> None:
    _touch_overlap_segments(tmp_path)
    preflight_offline_cache_or_die(
        symbol="SLV",
        candidates=[
            {
                "strategy": {
                    "instrument": "spot",
                    "symbol": "SLV",
                    "signal_bar_size": "1 min",
                    "signal_use_rth": False,
                }
            }
        ],
        windows=[(date(2025, 1, 1), date(2025, 1, 2))],
        signal_bar_size="1 min",
        use_rth=False,
        cache_dir=tmp_path,
    )


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

    assert stitched_src.exists()
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
