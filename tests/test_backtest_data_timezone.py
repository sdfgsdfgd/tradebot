import asyncio
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tradebot.chart_data.history import (
    cache_path,
    find_covering_cache_path,
    parse_cache_filename,
    write_cache,
)
from tradebot.backtest.data import (
    ContractMeta,
    IBKRHistoricalData,
    _HISTORICAL_HEAD_CACHE,
    _as_utc_aware,
    _convert_bar,
    _duration_for_bar_size,
    _duration_to_timedelta,
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


class _HistoricalIB:
    def __init__(self, *responses: list[_RawBar]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def isConnected(self) -> bool:
        return True

    def sleep(self, _seconds: float) -> None:
        return

    def reqHistoricalData(self, _contract, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0) if self.responses else []


class _ErrorEvent:
    def __init__(self) -> None:
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def emit(self, code: int, message: str) -> None:
        for handler in tuple(self.handlers):
            handler(1, code, message, None)


class _HistoricalErrorIB(_HistoricalIB):
    def __init__(self, code: int, message: str, *responses: list[_RawBar]) -> None:
        super().__init__(*responses)
        self.code = int(code)
        self.message = str(message)
        self.errorEvent = _ErrorEvent()
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))

    def reqHistoricalData(self, contract, **kwargs):
        out = super().reqHistoricalData(contract, **kwargs)
        if not out:
            self.errorEvent.emit(self.code, self.message)
        return out


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
    def test_xsp_resolves_as_cboe_index(self) -> None:
        data = IBKRHistoricalData()
        data.connect = lambda: None
        data._ib.qualifyContracts = lambda contract: [contract]

        contract, meta = data.resolve_contract("xsp", None)

        self.assertEqual(contract.secType, "IND")
        self.assertEqual(contract.symbol, "XSP")
        self.assertEqual(contract.exchange, "CBOE")
        self.assertEqual(meta.exchange, "CBOE")

    def test_vix_resolves_as_cboe_index(self) -> None:
        data = IBKRHistoricalData()
        data.connect = lambda: None
        data._ib.qualifyContracts = lambda contract: [contract]

        contract, meta = data.resolve_contract("vix", None)

        self.assertEqual(contract.secType, "IND")
        self.assertEqual(contract.exchange, "CBOE")
        self.assertEqual(meta.exchange, "CBOE")

    def test_one_day_fetch_uses_one_day_window_and_retries_empty_cursor(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalIB(
            [],
            [_RawBar(ts=datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc))],
        )
        bars = data._fetch_bars(
            SimpleNamespace(symbol="VIX", exchange="CBOE", secType="IND"),
            datetime(2025, 1, 15, 14, 30),
            datetime(2025, 1, 15, 21, 0),
            "5 mins",
            use_rth=True,
        )

        self.assertEqual(len(bars), 1)
        self.assertEqual(len(data._ib.calls), 2)
        self.assertEqual(
            [call["durationStr"] for call in data._ib.calls],
            ["1 D", "1 D"],
        )

    def test_fetch_never_steps_past_unexplained_empty_history(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalIB([], [], [])

        with self.assertRaisesRegex(RuntimeError, "historical_fetch_exhausted"):
            data._fetch_bars(
                SimpleNamespace(conId=13455763, symbol="VIX", exchange="CBOE", secType="IND"),
                datetime(2025, 1, 15, 14, 30),
                datetime(2025, 1, 15, 21, 0),
                "5 mins",
                use_rth=True,
            )

        self.assertEqual(len(data._ib.calls), 3)
        self.assertEqual(
            [call["durationStr"] for call in data._ib.calls],
            ["1 D", "1 D", "1 D"],
        )

    def test_no_data_message_remains_unresolved_without_head_proof(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalErrorIB(
            165,
            "Historical Market Data Service query message: No data",
            [],
            [],
            [],
        )

        with self.assertRaisesRegex(RuntimeError, "historical_no_data_observed"):
            data._fetch_bars(
                SimpleNamespace(conId=13455763, symbol="VIX", exchange="CBOE", secType="IND"),
                datetime(2025, 1, 15, 14, 30),
                datetime(2025, 1, 15, 21, 0),
                "5 mins",
                use_rth=True,
            )

    def test_pacing_violation_uses_fifteen_second_retry_floor(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalErrorIB(
            162,
            "Historical Market Data Service error message: pacing violation",
            [],
            [_RawBar(ts=datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc))],
        )

        bars = data._fetch_bars(
            SimpleNamespace(conId=3, symbol="VIX", exchange="CBOE", secType="IND"),
            datetime(2025, 1, 15, 14, 30),
            datetime(2025, 1, 15, 21, 0),
            "5 mins",
            use_rth=True,
        )

        self.assertEqual(len(bars), 1)
        self.assertEqual(data._ib.sleeps, [15.0])

    def test_head_timestamp_reuses_recent_broker_proof(self) -> None:
        data = IBKRHistoricalData()
        contract = SimpleNamespace(
            conId=13455763,
            symbol="VIX",
            exchange="CBOE",
            secType="IND",
        )
        key = (13455763, "IND", "VIX", "CBOE", True)
        _HISTORICAL_HEAD_CACHE.pop(key, None)
        loop = asyncio.new_event_loop()

        class _Wrapper:
            future = None

            def startReq(self, _req_id, _contract):
                self.future = loop.create_future()
                return self.future

        class _Client:
            requests = 0
            cancels = 0

            def getReqId(self):
                return 1

            def reqHeadTimeStamp(self, *_args):
                self.requests += 1
                wrapper.future.set_result(datetime(2005, 10, 3, 13, 30, tzinfo=timezone.utc))

            def cancelHeadTimeStamp(self, _req_id):
                self.cancels += 1

        wrapper = _Wrapper()
        client = _Client()
        data._ib = SimpleNamespace(
            client=client,
            wrapper=wrapper,
            _run=loop.run_until_complete,
        )
        try:
            first = data._historical_head_timestamp(contract, use_rth=True, timeout_sec=1.0)
            second = data._historical_head_timestamp(contract, use_rth=True, timeout_sec=1.0)
        finally:
            loop.close()
            _HISTORICAL_HEAD_CACHE.pop(key, None)

        self.assertEqual(first, second)
        self.assertEqual(client.requests, 1)
        self.assertEqual(client.cancels, 1)

    def test_one_day_rth_fetch_does_not_request_the_prior_session(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalIB(
            [_RawBar(ts=datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc))]
        )
        bars = data._fetch_bars(
            SimpleNamespace(conId=756733, symbol="SPY", exchange="SMART", secType="STK"),
            datetime(2025, 1, 15, 4, 0),
            datetime(2025, 1, 16, 3, 59),
            "5 mins",
            use_rth=True,
        )

        self.assertEqual(len(bars), 1)
        self.assertEqual(len(data._ib.calls), 1)

    def test_month_timeout_shrinks_later_chunks(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalIB(
            [],
            [_RawBar(ts=datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc))],
        )
        bars = data._fetch_bars(
            SimpleNamespace(conId=756733, symbol="SPY", exchange="SMART", secType="STK"),
            datetime(2025, 1, 1, 14, 30),
            datetime(2025, 3, 1, 21, 0),
            "5 mins",
            use_rth=True,
        )

        self.assertEqual(len(bars), 1)
        self.assertEqual(
            [call["durationStr"] for call in data._ib.calls],
            ["1 M", "2 W"],
        )

    def test_index_rth_fetch_discards_ibkr_extended_index_bars(self) -> None:
        data = IBKRHistoricalData()
        data._ib = _HistoricalIB(
            [
                _RawBar(ts=datetime(2025, 7, 24, 8, 0, tzinfo=timezone.utc)),
                _RawBar(ts=datetime(2025, 7, 24, 13, 30, tzinfo=timezone.utc)),
                _RawBar(ts=datetime(2025, 7, 24, 20, 55, tzinfo=timezone.utc)),
            ]
        )
        bars = data._fetch_bars(
            SimpleNamespace(conId=13455763, symbol="VIX", exchange="CBOE", secType="IND"),
            datetime(2025, 7, 24, 4, 0),
            datetime(2025, 7, 25, 3, 59),
            "5 mins",
            use_rth=True,
        )

        self.assertEqual([bar.ts for bar in bars], [datetime(2025, 7, 24, 13, 30)])

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
            self.assertFalse(exact.exists())

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

    def test_load_or_fetch_detects_missing_day_inside_covering_cache(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            req_start = datetime(2025, 1, 1, 0, 0)
            req_end = datetime(2025, 1, 10, 23, 59)
            covering = cache_path(
                cache_dir,
                "SLV",
                req_start,
                req_end,
                "1 min",
                use_rth=False,
            )
            covering.parent.mkdir(parents=True, exist_ok=True)
            write_cache(
                covering,
                [
                    bar
                    for bar in _day_bars(req_start.date(), req_end.date())
                    if bar.ts.date() != date(2025, 1, 6)
                ],
            )

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

            assert {start.date() for _, start, _, _, _ in data.fetch_calls} == {date(2025, 1, 6)}
            assert {end.date() for _, _, end, _, _ in data.fetch_calls} == {date(2025, 1, 6)}
            assert series.meta.source == "cache+ibkr"


    def test_load_cached_stitches_contiguous_cache_shards_without_copying_exact(self) -> None:
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
            self.assertFalse(exact.exists())

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
