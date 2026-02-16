from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from tradebot.backtest.data import IBKRHistoricalData, cache_path, write_cache
from tradebot.backtest.models import Bar
from tradebot.series import BarSeries, BarSeriesMeta
from tradebot.series_cache import SeriesCacheService
from tradebot.spot_engine import SpotSignalEvaluator


def _bars(start: datetime, *, n: int, minutes: int) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        ts = start + timedelta(minutes=minutes * i)
        px = 25.0 + (0.05 * i)
        out.append(
            Bar(
                ts=ts,
                open=px,
                high=px + 0.1,
                low=px - 0.1,
                close=px + 0.02,
                volume=1000 + i,
            )
        )
    return out


def test_load_cached_bar_series_returns_metadata() -> None:
    start = datetime(2025, 1, 6, 14, 30)
    end = datetime(2025, 1, 6, 15, 30)
    bars = _bars(start, n=5, minutes=15)
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = cache_path(root, "SLV", start, end, "15 mins", use_rth=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_cache(path, bars)

        data = IBKRHistoricalData()
        series = data.load_cached_bar_series(
            symbol="SLV",
            exchange=None,
            start=start,
            end=end,
            bar_size="15 mins",
            use_rth=False,
            cache_dir=root,
        )

    assert isinstance(series, BarSeries)
    assert len(series) > 0
    assert series.meta.symbol == "SLV"
    assert series.meta.bar_size == "15 mins"
    assert series.meta.tz_mode == "utc_naive"
    assert series.meta.session_mode == "full24"
    assert series.meta.source in {"cache", "cache-covering"}


def test_spot_signal_evaluator_accepts_bar_series_inputs() -> None:
    start = datetime(2025, 1, 6, 14, 30)
    regime = _bars(start, n=40, minutes=30)
    signal = _bars(start + timedelta(days=1), n=30, minutes=30)
    regime_series = BarSeries(
        bars=tuple(regime),
        meta=BarSeriesMeta(symbol="SLV", bar_size="30 mins", tz_mode="utc_naive", session_mode="rth"),
    )

    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "4/9",
            "ema_entry_mode": "trend",
            "regime_mode": "ema",
            "regime_ema_preset": "5/13",
        },
        filters=None,
        bar_size="30 mins",
        use_rth=True,
        naive_ts_mode="utc",
        regime_bars=regime_series,
    )

    last_snap = None
    for i, bar in enumerate(signal):
        next_bar = signal[i + 1] if i + 1 < len(signal) else None
        is_last_bar = bool(next_bar is None or next_bar.ts.date() != bar.ts.date())
        evaluator.update_exec_bar(bar, is_last_bar=is_last_bar)
        snap = evaluator.update_signal_bar(bar)
        if snap is not None:
            last_snap = snap

    assert last_snap is not None
    assert last_snap.signal is not None


def test_series_cache_service_roundtrip_memory_and_persistent() -> None:
    cache = SeriesCacheService()
    key = ("k", 1)
    value = {"x": 1}
    cache.set(namespace="ns", key=key, value=value)
    assert cache.get(namespace="ns", key=key) == value

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "series_cache.sqlite3"
        cache.set_persistent(
            db_path=db_path,
            namespace="ns",
            key_hash="abc",
            value=value,
        )
        loaded = cache.get_persistent(
            db_path=db_path,
            namespace="ns",
            key_hash="abc",
            validator=lambda obj: isinstance(obj, dict),
        )
    assert loaded == value
