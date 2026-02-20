import unittest
from datetime import date, datetime, time

from tradebot.backtest.trading_calendar import (
    SESSION_ORDER,
    expected_sessions,
    et_day_from_utc_naive,
    full24_post_close_time_et,
    is_early_close_day,
    is_trading_day,
    session_label_et,
    utc_bounds_for_et_day,
)


class BacktestTradingCalendarTests(unittest.TestCase):
    def test_session_label_and_maintenance_gap(self) -> None:
        self.assertEqual(session_label_et(time(3, 49)), "OVERNIGHT_EARLY")
        self.assertIsNone(session_label_et(time(3, 50)))
        self.assertEqual(session_label_et(time(4, 0)), "PRE")
        self.assertEqual(session_label_et(time(9, 30)), "RTH")
        self.assertEqual(session_label_et(time(16, 0)), "POST")
        self.assertEqual(session_label_et(time(20, 0)), "OVERNIGHT_LATE")

    def test_special_closed_day_not_trading(self) -> None:
        self.assertFalse(is_trading_day(date(2025, 1, 9)))
        self.assertEqual(expected_sessions(date(2025, 1, 9), session_mode="full24"), set())

    def test_expected_sessions_include_overnight_late_when_next_day_trades(self) -> None:
        sessions = expected_sessions(date(2025, 1, 13), session_mode="full24")
        self.assertEqual(tuple(sorted(sessions, key=lambda x: SESSION_ORDER.index(x))), SESSION_ORDER)

    def test_expected_sessions_skip_overnight_late_before_weekend(self) -> None:
        sessions = expected_sessions(date(2025, 1, 10), session_mode="full24")
        self.assertNotIn("OVERNIGHT_LATE", sessions)

    def test_et_utc_day_bounds_and_roundtrip_trade_day(self) -> None:
        start_utc, end_utc = utc_bounds_for_et_day(date(2025, 1, 15))
        self.assertEqual(start_utc, datetime(2025, 1, 15, 5, 0))
        self.assertEqual(end_utc, datetime(2025, 1, 16, 4, 59))
        self.assertEqual(et_day_from_utc_naive(datetime(2025, 1, 16, 0, 30)), date(2025, 1, 15))

    def test_early_close_day_rules(self) -> None:
        self.assertTrue(is_early_close_day(date(2025, 11, 28)))
        self.assertTrue(is_early_close_day(date(2025, 12, 24)))
        self.assertFalse(is_early_close_day(date(2025, 12, 26)))

    def test_full24_post_close_cutoff_on_early_close_day(self) -> None:
        self.assertEqual(full24_post_close_time_et(date(2025, 11, 28)), time(17, 0))
        self.assertEqual(full24_post_close_time_et(date(2025, 12, 24)), time(17, 0))
        self.assertEqual(full24_post_close_time_et(date(2025, 12, 26)), time(20, 0))


if __name__ == "__main__":
    unittest.main()
