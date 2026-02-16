import unittest
from datetime import datetime

from tradebot.backtest.data import _normalize_bars
from tradebot.backtest.models import Bar


class BacktestTimeAlignmentTests(unittest.TestCase):
    def test_intraday_bars_shift_to_close_timestamp(self) -> None:
        raw = [
            Bar(ts=datetime(2025, 1, 1, 0, 0), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 1, 1, 0), open=1, high=1, low=1, close=1, volume=0),
        ]
        out = _normalize_bars(raw, symbol="MNQ", bar_size="1 hour", use_rth=False)
        self.assertEqual(out[0].ts, datetime(2025, 1, 1, 1, 0))
        self.assertEqual(out[1].ts, datetime(2025, 1, 1, 2, 0))

    def test_daily_bars_align_to_session_close_for_futures(self) -> None:
        # For MNQ full-session daily bars, we align to the CME daily break at 17:00 ET.
        # 2025-01-10 is in standard time (ET = UTC-5), so 17:00 ET == 22:00 UTC.
        raw = [Bar(ts=datetime(2025, 1, 10, 0, 0), open=1, high=1, low=1, close=1, volume=0)]
        out = _normalize_bars(raw, symbol="MNQ", bar_size="1 day", use_rth=False)
        self.assertEqual(out[0].ts, datetime(2025, 1, 10, 22, 0))

    def test_mtf_regime_bar_not_available_until_complete(self) -> None:
        # Regression test for multi-timeframe lookahead:
        # a 4-hour regime bar starting at 00:00 should not be usable at the 1-hour close at 01:00.
        signal_raw = [
            Bar(ts=datetime(2025, 1, 1, 0, 0), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 1, 1, 0), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 1, 2, 0), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 1, 3, 0), open=1, high=1, low=1, close=1, volume=0),
        ]
        regime_raw = [Bar(ts=datetime(2025, 1, 1, 0, 0), open=1, high=1, low=1, close=1, volume=0)]

        signal = _normalize_bars(signal_raw, symbol="MNQ", bar_size="1 hour", use_rth=False)
        regime = _normalize_bars(regime_raw, symbol="MNQ", bar_size="4 hours", use_rth=False)

        regime_idx = 0
        seen = {}
        for bar in signal:
            while regime_idx < len(regime) and regime[regime_idx].ts <= bar.ts:
                regime_idx += 1
            seen[bar.ts] = regime_idx

        self.assertEqual(seen.get(datetime(2025, 1, 1, 1, 0)), 0)
        self.assertEqual(seen.get(datetime(2025, 1, 1, 4, 0)), 1)

    def test_rth_fragmented_4hour_bars_align_to_true_closes(self) -> None:
        # IBKR can return fragmented 4-hour RTH starts for equities:
        # 09:30 ET, 11:00 ET, 15:00 ET (UTC: 14:30, 16:00, 20:00 in winter).
        # Close alignment should be 11:00 ET, 15:00 ET, 16:00 ET.
        raw = [
            Bar(ts=datetime(2025, 1, 2, 14, 30), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 2, 16, 0), open=1, high=1, low=1, close=1, volume=0),
            Bar(ts=datetime(2025, 1, 2, 20, 0), open=1, high=1, low=1, close=1, volume=0),
        ]
        out = _normalize_bars(raw, symbol="TQQQ", bar_size="4 hours", use_rth=True)
        self.assertEqual(out[0].ts, datetime(2025, 1, 2, 16, 0))
        self.assertEqual(out[1].ts, datetime(2025, 1, 2, 20, 0))
        self.assertEqual(out[2].ts, datetime(2025, 1, 2, 21, 0))


if __name__ == "__main__":
    unittest.main()
