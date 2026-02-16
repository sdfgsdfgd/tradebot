import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from tradebot.backtest.data import (
    ContractMeta,
    IBKRHistoricalData,
    _as_utc_aware,
    _convert_bar,
    _duration_for_bar_size,
    _duration_to_timedelta,
    cache_path,
    find_covering_cache_path,
    parse_cache_filename,
    write_cache,
)
from tradebot.backtest.models import Bar


class _RawBar:
    def __init__(self, *, ts, open_: float = 1.0, high: float = 2.0, low: float = 0.5, close: float = 1.5, volume: float = 10.0) -> None:
        self.date = ts
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _day_bars(start_day: date, end_day: date, *, hour: int = 14, minute: int = 30) -> list[Bar]:
    out: list[Bar] = []
    cur = start_day
    while cur <= end_day:
        ts = datetime(cur.year, cur.month, cur.day, hour, minute)
        out.append(Bar(ts=ts, open=1.0, high=1.2, low=0.8, close=1.1, volume=100.0))
        cur += timedelta(days=1)
    return out


class _StubHistoricalData(IBKRHistoricalData):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_calls: list[tuple[str, datetime, datetime, str, bool]] = []

    def connect(self) -> None:
        return

    def resolve_contract(self, symbol: str, exchange) -> tuple[object, ContractMeta]:
        class _Contract:
            def __init__(self, sym: str, ex: str) -> None:
                self.symbol = sym
                self.exchange = ex
                self.secType = "STK"

        ex = str(exchange or "SMART")
        return (
            _Contract(symbol, ex),
            ContractMeta(symbol=symbol, exchange=ex, multiplier=1.0, min_tick=0.01),
        )

    def _fetch_bars(
        self,
        contract: object,
        start: datetime,
        end: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> list[Bar]:
        self.fetch_calls.append((str(getattr(contract, "exchange", "?")), start, end, str(bar_size), bool(use_rth)))
        return _day_bars(start.date(), end.date())


class BacktestDataTimezoneTests(unittest.TestCase):
    def test_as_utc_aware_marks_naive_as_utc(self) -> None:
        ts = _as_utc_aware(datetime(2025, 1, 15, 9, 30))
        self.assertEqual(ts.tzinfo, timezone.utc)
        self.assertEqual(ts.hour, 9)
        self.assertEqual(ts.minute, 30)

    def test_as_utc_aware_converts_aware_to_utc(self) -> None:
        ts = _as_utc_aware(datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc))
        self.assertEqual(ts, datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc))

    def test_duration_for_1min_uses_smaller_ibkr_chunk(self) -> None:
        self.assertEqual(_duration_for_bar_size("1 min"), "1 W")

    def test_duration_to_timedelta_week(self) -> None:
        self.assertEqual(_duration_to_timedelta("1 W"), timedelta(days=7))

    def test_convert_bar_intraday_naive_et_is_canonicalized_to_utc_naive_standard_time(self) -> None:
        raw = _RawBar(ts=datetime(2025, 1, 15, 9, 30))
        out = _convert_bar(raw)
        self.assertEqual(out.ts, datetime(2025, 1, 15, 14, 30))

    def test_convert_bar_intraday_naive_et_is_canonicalized_to_utc_naive_dst(self) -> None:
        raw = _RawBar(ts=datetime(2025, 7, 15, 9, 30))
        out = _convert_bar(raw)
        self.assertEqual(out.ts, datetime(2025, 7, 15, 13, 30))

    def test_convert_bar_intraday_aware_datetime_is_normalized_to_utc_naive(self) -> None:
        raw = _RawBar(ts=datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc))
        out = _convert_bar(raw)
        self.assertEqual(out.ts, datetime(2025, 1, 15, 14, 30))

    def test_convert_bar_daily_date_remains_midnight_date_marker(self) -> None:
        raw = _RawBar(ts=date(2025, 1, 15))
        out = _convert_bar(raw)
        self.assertEqual(out.ts, datetime(2025, 1, 15, 0, 0))

    def test_parse_cache_filename_rejects_unapproved_bar_token(self) -> None:
        bad = parse_cache_filename(Path("SLV_2025-01-01_2026-01-01_1minute_full24.csv"))
        self.assertIsNone(bad)
        good = parse_cache_filename(Path("SLV_2025-01-01_2026-01-01_1min_full24.csv"))
        self.assertIsNotNone(good)

    def test_find_covering_cache_path_ignores_invalid_cache_filenames(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            folder = cache_dir / "SLV"
            folder.mkdir(parents=True, exist_ok=True)
            good = folder / "SLV_2025-01-01_2026-01-01_1min_full24.csv"
            good.write_text("ts,open,high,low,close,volume\n")
            (folder / "SLV_2025-01-01_2026-01-01_1minute_full24.csv").write_text("x")
            (folder / "SLV_2025-01-01_2026-01-01_1min_full24.csv.tmp").write_text("x")
            (folder / "SLV_2025-01-01_2026-01-01_1min_full24.bak").write_text("x")

            out = find_covering_cache_path(
                cache_dir=cache_dir,
                symbol="SLV",
                start=datetime(2025, 2, 1, 0, 0),
                end=datetime(2025, 2, 7, 0, 0),
                bar_size="1 min",
                use_rth=False,
            )
            self.assertEqual(out, good)

    def test_find_covering_cache_path_allows_known_alias_token(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            folder = cache_dir / "SLV"
            folder.mkdir(parents=True, exist_ok=True)
            legacy = folder / "SLV_2025-01-01_2026-01-01_10min_full24.csv"
            legacy.write_text("ts,open,high,low,close,volume\n")

            out = find_covering_cache_path(
                cache_dir=cache_dir,
                symbol="SLV",
                start=datetime(2025, 3, 1, 0, 0),
                end=datetime(2025, 3, 5, 0, 0),
                bar_size="10 min",
                use_rth=False,
            )
            self.assertEqual(out, legacy)


    def test_load_or_fetch_stitches_contiguous_cache_shards_without_refetch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            req_start = datetime(2025, 1, 1, 0, 0)
            req_end = datetime(2025, 1, 10, 23, 59)

            shard_a = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 5, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_b = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 1, 6, 0, 0),
                datetime(2025, 1, 10, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_a.parent.mkdir(parents=True, exist_ok=True)
            write_cache(shard_a, _day_bars(date(2025, 1, 1), date(2025, 1, 5)))
            write_cache(shard_b, _day_bars(date(2025, 1, 6), date(2025, 1, 10)))

            data = _StubHistoricalData()
            series = data.load_or_fetch_bar_series(
                symbol="SLV",
                exchange="SMART",
                start=req_start,
                end=req_end,
                bar_size="1 min",
                use_rth=False,
                cache_dir=cache_dir,
            )

            self.assertEqual(len(data.fetch_calls), 0)
            self.assertEqual(series.meta.source, "cache-stitched")
            exact = cache_path(cache_dir, "SLV", req_start, req_end, "1 min", use_rth=False)
            self.assertTrue(exact.exists())

    def test_load_or_fetch_fetches_only_uncovered_gap_days(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            req_start = datetime(2025, 1, 1, 0, 0)
            req_end = datetime(2025, 1, 10, 23, 59)

            shard_a = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 5, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_b = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 1, 7, 0, 0),
                datetime(2025, 1, 10, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_a.parent.mkdir(parents=True, exist_ok=True)
            write_cache(shard_a, _day_bars(date(2025, 1, 1), date(2025, 1, 5)))
            write_cache(shard_b, _day_bars(date(2025, 1, 7), date(2025, 1, 10)))

            data = _StubHistoricalData()
            series = data.load_or_fetch_bar_series(
                symbol="SLV",
                exchange="SMART",
                start=req_start,
                end=req_end,
                bar_size="1 min",
                use_rth=False,
                cache_dir=cache_dir,
            )

            self.assertGreaterEqual(len(data.fetch_calls), 2)
            for _, fetch_start, fetch_end, _, _ in data.fetch_calls:
                self.assertEqual(fetch_start.date(), date(2025, 1, 6))
                self.assertEqual(fetch_end.date(), date(2025, 1, 6))
            exchanges = {x[0] for x in data.fetch_calls}
            self.assertIn("SMART", exchanges)
            self.assertIn("OVERNIGHT", exchanges)
            self.assertEqual(series.meta.source, "cache+ibkr")


    def test_load_cached_stitches_contiguous_cache_shards_and_persists_exact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            req_start = datetime(2025, 2, 1, 0, 0)
            req_end = datetime(2025, 2, 10, 23, 59)

            shard_a = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 2, 1, 0, 0),
                datetime(2025, 2, 5, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_b = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 2, 6, 0, 0),
                datetime(2025, 2, 10, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_a.parent.mkdir(parents=True, exist_ok=True)
            write_cache(shard_a, _day_bars(date(2025, 2, 1), date(2025, 2, 5)))
            write_cache(shard_b, _day_bars(date(2025, 2, 6), date(2025, 2, 10)))

            data = IBKRHistoricalData()
            series = data.load_cached_bar_series(
                symbol="SLV",
                exchange="SMART",
                start=req_start,
                end=req_end,
                bar_size="1 min",
                use_rth=False,
                cache_dir=cache_dir,
            )

            self.assertEqual(series.meta.source, "cache-stitched")
            exact = cache_path(cache_dir, "SLV", req_start, req_end, "1 min", use_rth=False)
            self.assertTrue(exact.exists())

    def test_load_cached_raises_when_date_coverage_has_gaps(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            req_start = datetime(2025, 3, 1, 0, 0)
            req_end = datetime(2025, 3, 10, 23, 59)

            shard_a = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 3, 1, 0, 0),
                datetime(2025, 3, 5, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_b = cache_path(
                cache_dir,
                "SLV",
                datetime(2025, 3, 7, 0, 0),
                datetime(2025, 3, 10, 23, 59),
                "1 min",
                use_rth=False,
            )
            shard_a.parent.mkdir(parents=True, exist_ok=True)
            write_cache(shard_a, _day_bars(date(2025, 3, 1), date(2025, 3, 5)))
            write_cache(shard_b, _day_bars(date(2025, 3, 7), date(2025, 3, 10)))

            data = IBKRHistoricalData()
            with self.assertRaises(FileNotFoundError) as ctx:
                data.load_cached_bar_series(
                    symbol="SLV",
                    exchange="SMART",
                    start=req_start,
                    end=req_end,
                    bar_size="1 min",
                    use_rth=False,
                    cache_dir=cache_dir,
                )
            self.assertIn("missing date ranges", str(ctx.exception))
            self.assertIn("2025-03-06", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
