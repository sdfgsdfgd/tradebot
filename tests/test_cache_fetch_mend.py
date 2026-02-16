from __future__ import annotations

from datetime import date, datetime

from tradebot.backtest.cache_ops_lib import _CacheFetchRequest, _build_fetch_batches, _fetch_single_request
from tradebot.backtest.data import _read_cache_cached, cache_path, read_cache, write_cache
from tradebot.backtest.models import Bar


class _DummyProvider:
    def __init__(self, client_id_offset: int = 0) -> None:
        self.client_id_offset = int(client_id_offset)

    def disconnect(self) -> None:
        return None

    def load_or_fetch_bars(self, **kwargs):
        raise AssertionError("unexpected full-window fetch in cache-hit mend test")


def _bar(ts: datetime) -> Bar:
    return Bar(ts=ts, open=10.0, high=11.0, low=9.0, close=10.5, volume=100.0)


def test_fetch_single_request_auto_mends_existing_1min_gap(monkeypatch, tmp_path) -> None:
    req = _CacheFetchRequest(
        symbol="SLV",
        start=date(2025, 1, 15),
        end=date(2025, 1, 15),
        bar_size="1 min",
        use_rth=True,
        source="test",
    )
    start_dt = datetime(2025, 1, 15, 0, 0)
    end_dt = datetime(2025, 1, 15, 23, 59)
    path = cache_path(tmp_path, req.symbol, start_dt, end_dt, req.bar_size, req.use_rth)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        path,
        [
            _bar(datetime(2025, 1, 15, 14, 30)),
            _bar(datetime(2025, 1, 15, 14, 31)),
            _bar(datetime(2025, 1, 15, 14, 33)),
        ],
    )

    monkeypatch.setattr("tradebot.backtest.cache_ops_lib.IBKRHistoricalData", _DummyProvider)

    def _fake_overlay_adaptive(**kwargs):
        by_ts = kwargs["by_ts"]
        by_ts[datetime(2025, 1, 15, 14, 32)] = _bar(datetime(2025, 1, 15, 14, 32))
        return {
            "days_considered": 1,
            "threads": 1,
            "adaptive_threads_enabled": False,
            "thread_plan": [1],
            "pass_stats": [],
            "retries": 1,
            "timeout_sec": 1.0,
            "fetched_ok": 1,
            "fetched_fail": 0,
            "days_replaced": 1,
            "fetched": [],
            "replaced": [],
        }

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops_lib._ibkr_overlay_adaptive",
        _fake_overlay_adaptive,
    )

    out = _fetch_single_request(
        req=req,
        cache_dir=tmp_path,
        force_refresh=False,
        timeout_sec=1.0,
        client_id_offset=1,
        mend_threads=1,
        mend_retries=1,
        mend_adaptive_threads=False,
    )

    assert out.ok is True
    assert out.from_cache is False
    assert out.rows == 4
    _read_cache_cached.cache_clear()
    rows = read_cache(path)
    assert any(row.ts == datetime(2025, 1, 15, 14, 32) for row in rows)


def test_fetch_single_request_keeps_clean_cache_hit(monkeypatch, tmp_path) -> None:
    req = _CacheFetchRequest(
        symbol="SLV",
        start=date(2025, 1, 15),
        end=date(2025, 1, 15),
        bar_size="1 min",
        use_rth=True,
        source="test",
    )
    start_dt = datetime(2025, 1, 15, 0, 0)
    end_dt = datetime(2025, 1, 15, 23, 59)
    path = cache_path(tmp_path, req.symbol, start_dt, end_dt, req.bar_size, req.use_rth)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        path,
        [
            _bar(datetime(2025, 1, 15, 14, 30)),
            _bar(datetime(2025, 1, 15, 14, 31)),
            _bar(datetime(2025, 1, 15, 14, 32)),
        ],
    )

    monkeypatch.setattr("tradebot.backtest.cache_ops_lib.IBKRHistoricalData", _DummyProvider)

    def _overlay_should_not_run(**kwargs):
        raise AssertionError("unexpected overlay for clean cache-hit")

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops_lib._ibkr_overlay_adaptive",
        _overlay_should_not_run,
    )

    out = _fetch_single_request(
        req=req,
        cache_dir=tmp_path,
        force_refresh=False,
        timeout_sec=1.0,
        client_id_offset=1,
        mend_threads=1,
        mend_retries=1,
        mend_adaptive_threads=False,
    )

    assert out.ok is True
    assert out.from_cache is True
    assert out.rows == 3


def test_build_fetch_batches_merges_without_sharding() -> None:
    reqs = [
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            bar_size="1 min",
            use_rth=False,
            source="a",
        ),
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 3, 1),
            end=date(2025, 3, 31),
            bar_size="1 min",
            use_rth=False,
            source="b",
        ),
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 6, 1),
            end=date(2025, 6, 30),
            bar_size="1 min",
            use_rth=False,
            source="c",
        ),
    ]

    batches = _build_fetch_batches(reqs)

    assert len(batches) == 1
    primary = batches[0].primary
    assert primary.start == date(2025, 1, 1)
    assert primary.end == date(2025, 6, 30)
    assert len(batches[0].targets) == 3


def test_build_fetch_batches_shards_by_max_span_days() -> None:
    reqs = [
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            bar_size="1 min",
            use_rth=False,
            source="a",
        ),
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 3, 1),
            end=date(2025, 3, 31),
            bar_size="1 min",
            use_rth=False,
            source="b",
        ),
        _CacheFetchRequest(
            symbol="SLV",
            start=date(2025, 6, 1),
            end=date(2025, 6, 30),
            bar_size="1 min",
            use_rth=False,
            source="c",
        ),
    ]

    batches = _build_fetch_batches(reqs, max_primary_span_days=120)

    assert len(batches) == 2
    assert batches[0].primary.start == date(2025, 1, 1)
    assert batches[0].primary.end == date(2025, 3, 31)
    assert len(batches[0].targets) == 2
    assert "shard1/2" in batches[0].primary.source

    assert batches[1].primary.start == date(2025, 6, 1)
    assert batches[1].primary.end == date(2025, 6, 30)
    assert len(batches[1].targets) == 1
    assert "shard2/2" in batches[1].primary.source
