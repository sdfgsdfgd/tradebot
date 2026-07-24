from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from tradebot.backtest.cache_ops.cli import main_fetch
from tradebot.backtest.cache_ops.sync import (
    _CacheFetchOutcome,
    _CacheFetchRequest,
    _build_fetch_batches,
    _fetch_single_request,
    _repo_root,
)
from tradebot.backtest.cache_ops.coverage import _intra_session_gap_days
from tradebot.backtest.cache_ops.repair import (
    _adaptive_thread_plan,
    _fetch_day_from_ibkr,
    _is_retryable_ibkr_error,
)
from tradebot.chart_data.history import _read_cache_cached, cache_path, read_cache, write_cache
from tradebot.backtest.models import Bar


def test_cache_ops_repo_root_resolves_checkout() -> None:
    assert _repo_root() == Path(__file__).resolve().parents[1]


def test_adaptive_thread_plan_only_reduces_concurrency() -> None:
    assert _adaptive_thread_plan(1) == [1]
    assert _adaptive_thread_plan(2) == [2, 1]
    assert _adaptive_thread_plan(6) == [6, 3, 1]
    assert _adaptive_thread_plan(10) == [10, 5, 3, 1]


def test_retryable_ibkr_errors_include_ambiguous_and_incomplete_fetches() -> None:
    assert _is_retryable_ibkr_error("historical_fetch_exhausted: head=unknown")
    assert _is_retryable_ibkr_error("historical_no_data_observed: head=unknown")
    assert _is_retryable_ibkr_error("incomplete_after_fetch: dates=2025-01-16")
    assert _is_retryable_ibkr_error("connection reset by peer")
    assert not _is_retryable_ibkr_error("historical_unavailable_before_head: earliest=2025-01-17")


def test_day_repair_uses_canonical_index_without_overnight(monkeypatch) -> None:
    class _Contract:
        secType = "IND"

    class _Provider:
        resolve_calls: list[tuple[str, str | None]] = []

        def __init__(self, *, client_id_offset: int) -> None:
            assert client_id_offset == 17

        def resolve_contract(self, symbol: str, exchange: str | None):
            self.resolve_calls.append((symbol, exchange))
            return _Contract(), object()

        def _fetch_bars(self, *_args, **_kwargs):
            return [_bar(datetime(2025, 1, 15, 14, 30))]

        def disconnect(self) -> None:
            return

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.repair.IBKRHistoricalData",
        _Provider,
    )

    out = _fetch_day_from_ibkr(
        symbol="VIX",
        bar_size="5 mins",
        day=date(2025, 1, 15),
        use_rth=False,
        timeout_sec=1.0,
        retries=1,
        client_id_offset=17,
    )

    assert out.ok is True
    assert _Provider.resolve_calls == [("VIX", None)]
    assert out.overnight_rows == 0


def test_day_repair_does_not_retry_proven_unavailability(monkeypatch) -> None:
    class _Provider:
        instances = 0

        def __init__(self, *, client_id_offset: int) -> None:
            type(self).instances += 1

        def resolve_contract(self, symbol: str, exchange: str | None):
            return object(), object()

        def _fetch_bars(self, *_args, **_kwargs):
            raise RuntimeError(
                "historical_unavailable_before_head: "
                "requested_end=2025-01-15 earliest=2025-01-16"
            )

        def disconnect(self) -> None:
            return

    sleeps: list[float] = []
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.repair.IBKRHistoricalData",
        _Provider,
    )
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.repair.sleep",
        sleeps.append,
    )

    out = _fetch_day_from_ibkr(
        symbol="VIX",
        bar_size="5 mins",
        day=date(2025, 1, 15),
        use_rth=True,
        timeout_sec=1.0,
        retries=3,
        client_id_offset=17,
    )

    assert out.ok is False
    assert out.attempts == 1
    assert _Provider.instances == 1
    assert sleeps == []


def test_day_repair_backs_off_ambiguous_failures(monkeypatch) -> None:
    class _Provider:
        instances = 0

        def __init__(self, *, client_id_offset: int) -> None:
            type(self).instances += 1

        def resolve_contract(self, symbol: str, exchange: str | None):
            return object(), object()

        def _fetch_bars(self, *_args, **_kwargs):
            if type(self).instances < 3:
                raise RuntimeError("historical_fetch_exhausted: head=unknown")
            return [_bar(datetime(2025, 1, 15, 14, 30))]

        def disconnect(self) -> None:
            return

    sleeps: list[float] = []
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.repair.IBKRHistoricalData",
        _Provider,
    )
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.repair.sleep",
        sleeps.append,
    )

    out = _fetch_day_from_ibkr(
        symbol="VIX",
        bar_size="5 mins",
        day=date(2025, 1, 15),
        use_rth=True,
        timeout_sec=1.0,
        retries=3,
        client_id_offset=17,
    )

    assert out.ok is True
    assert out.attempts == 3
    assert sleeps == [1.5, 3.1]


class _DummyProvider:
    def __init__(self, client_id_offset: int = 0) -> None:
        self.client_id_offset = int(client_id_offset)

    def disconnect(self) -> None:
        return None

    def load_or_fetch_bars(self, **kwargs):
        raise AssertionError("unexpected full-window fetch in cache-hit mend test")


def _bar(ts: datetime) -> Bar:
    return Bar(ts=ts, open=10.0, high=11.0, low=9.0, close=10.5, volume=100.0)


def _rth_day(day: date) -> list[Bar]:
    start = datetime.combine(day, datetime.min.time()).replace(hour=14, minute=30)
    return [_bar(start + timedelta(minutes=offset)) for offset in range(390)]


def test_one_minute_rth_rejects_contiguous_partial_session() -> None:
    start = datetime(2026, 2, 25, 14, 30)
    rows = [_bar(start + timedelta(minutes=offset)) for offset in range(75)]

    assert _intra_session_gap_days(
        rows,
        start_utc_date=date(2026, 2, 25),
        end_utc_date=date(2026, 2, 25),
        session_mode="rth",
        bar_size="1 min",
    ) == {"2026-02-25": ["RTH"]}


def test_one_minute_rth_accepts_complete_early_close() -> None:
    start = datetime(2025, 11, 28, 14, 30)
    rows = [_bar(start + timedelta(minutes=offset)) for offset in range(210)]

    assert _intra_session_gap_days(
        rows,
        start_utc_date=date(2025, 11, 28),
        end_utc_date=date(2025, 11, 28),
        session_mode="rth",
        bar_size="1 min",
    ) == {}


def test_five_minute_rth_rejects_post_early_close_index_bars() -> None:
    start = datetime(2025, 11, 28, 14, 30)
    rows = [_bar(start + timedelta(minutes=5 * offset)) for offset in range(45)]

    assert _intra_session_gap_days(
        rows,
        start_utc_date=date(2025, 11, 28),
        end_utc_date=date(2025, 11, 28),
        session_mode="rth",
        bar_size="5 mins",
    ) == {"2025-11-28": ["RTH"]}


def test_five_minute_rth_accepts_complete_early_close() -> None:
    start = datetime(2025, 11, 28, 14, 30)
    rows = [_bar(start + timedelta(minutes=5 * offset)) for offset in range(42)]

    assert _intra_session_gap_days(
        rows,
        start_utc_date=date(2025, 11, 28),
        end_utc_date=date(2025, 11, 28),
        session_mode="rth",
        bar_size="5 mins",
    ) == {}


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
            row
            for offset, row in enumerate(_rth_day(date(2025, 1, 15)))
            if offset != 2
        ],
    )

    class _SparseProvider(_DummyProvider):
        calls = 0

        def load_or_fetch_bars(self, **_kwargs):
            type(self).calls += 1
            write_cache(path, _rth_day(date(2025, 1, 15)))

    monkeypatch.setattr("tradebot.backtest.cache_ops.sync.IBKRHistoricalData", _SparseProvider)
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync._ibkr_overlay_adaptive",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("day overlay should not run after sparse hydration")
        ),
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
    assert out.rows == 390
    assert _SparseProvider.calls == 1
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
        _rth_day(date(2025, 1, 15)),
    )

    monkeypatch.setattr("tradebot.backtest.cache_ops.sync.IBKRHistoricalData", _DummyProvider)

    def _overlay_should_not_run(**kwargs):
        raise AssertionError("unexpected overlay for clean cache-hit")

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync._ibkr_overlay_adaptive",
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
    assert out.rows == 390


def test_fetch_single_request_rejects_incomplete_new_cache(
    monkeypatch, tmp_path
) -> None:
    req = _CacheFetchRequest(
        symbol="SLV",
        start=date(2025, 1, 15),
        end=date(2025, 1, 16),
        bar_size="1 min",
        use_rth=True,
        source="test",
    )
    expected = cache_path(
        tmp_path,
        req.symbol,
        datetime(2025, 1, 15),
        datetime(2025, 1, 16, 23, 59),
        req.bar_size,
        req.use_rth,
    )

    class _IncompleteProvider(_DummyProvider):
        def load_or_fetch_bars(self, **kwargs):
            write_cache(
                cache_path(
                    kwargs["cache_dir"],
                    kwargs["symbol"],
                    kwargs["start"],
                    kwargs["end"],
                    kwargs["bar_size"],
                    kwargs["use_rth"],
                ),
                _rth_day(date(2025, 1, 15)),
            )

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync.IBKRHistoricalData",
        _IncompleteProvider,
    )
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync._ibkr_overlay_adaptive",
        lambda **kwargs: None,
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

    assert out.ok is False
    assert (
        out.error
        == "incomplete_after_fetch: session_days=1 gap_days=0 dates=2025-01-16"
    )
    assert not expected.exists()


def test_fetch_single_request_mends_incomplete_new_cache(
    monkeypatch, tmp_path
) -> None:
    req = _CacheFetchRequest(
        symbol="SLV",
        start=date(2025, 1, 15),
        end=date(2025, 1, 16),
        bar_size="1 min",
        use_rth=True,
        source="test",
    )
    expected = cache_path(
        tmp_path,
        req.symbol,
        datetime(2025, 1, 15),
        datetime(2025, 1, 16, 23, 59),
        req.bar_size,
        req.use_rth,
    )

    class _IncompleteProvider(_DummyProvider):
        def load_or_fetch_bars(self, **kwargs):
            write_cache(
                expected,
                _rth_day(date(2025, 1, 15)),
            )

    def _repair_missing_day(**kwargs):
        by_ts = kwargs["by_ts"]
        for row in _rth_day(date(2025, 1, 16)):
            by_ts[row.ts] = row

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync.IBKRHistoricalData",
        _IncompleteProvider,
    )
    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.sync._ibkr_overlay_adaptive",
        _repair_missing_day,
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
    assert out.healed is True
    assert out.healed_days == 1
    assert out.rows == 780
    assert expected.exists()
    _read_cache_cached.cache_clear()
    assert any(
        row.ts.date() == date(2025, 1, 16)
        for row in read_cache(expected)
    )


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


def test_cache_sync_keeps_target_windows_as_views(monkeypatch, tmp_path) -> None:
    primary_path = cache_path(
        tmp_path,
        "SLV",
        datetime(2025, 1, 1),
        datetime(2025, 3, 31, 23, 59),
        "5 mins",
        True,
    )
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(
        primary_path,
        [
            _bar(datetime(2025, 1, 15, 14, 30)),
            _bar(datetime(2025, 3, 15, 14, 30)),
        ],
    )

    run_kwargs: list[dict[str, object]] = []

    def _fake_run(*, batches, **kwargs):
        run_kwargs.append(kwargs)
        req = batches[0].primary
        return [
            _CacheFetchOutcome(
                request=req,
                ok=True,
                from_cache=True,
                rows=2,
                cache_path=str(primary_path),
                error=None,
            )
        ]

    monkeypatch.setattr(
        "tradebot.backtest.cache_ops.cli._run_primary_batches",
        _fake_run,
    )
    report_path = tmp_path / "report.json"
    main_fetch(
        [
            "--request",
            "SLV|2025-01-01|2025-01-31|5 mins|rth|a",
            "--request",
            "SLV|2025-03-01|2025-03-31|5 mins|rth|b",
            "--cache-dir",
            str(tmp_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert run_kwargs[0]["threads"] == 2
    assert run_kwargs[0]["mend_threads"] == 1
    assert primary_path.exists()
    assert not cache_path(
        tmp_path,
        "SLV",
        datetime(2025, 1, 1),
        datetime(2025, 1, 31, 23, 59),
        "5 mins",
        True,
    ).exists()
    assert not cache_path(
        tmp_path,
        "SLV",
        datetime(2025, 3, 1),
        datetime(2025, 3, 31, 23, 59),
        "5 mins",
        True,
    ).exists()
