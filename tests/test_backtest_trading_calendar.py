import unittest
from datetime import date, datetime, time, timedelta, timezone

from tradebot.engines.market import (
    SESSION_ORDER,
    expected_sessions,
    et_day_from_utc_naive,
    full24_post_close_time_et,
    is_early_close_day,
    is_trading_day,
    market_breadth_observation,
    session_label_et,
    utc_bounds_for_et_day,
    xsp_capture_window_date,
    xsp_session_label_et,
    xsp_trading_date,
)


class BacktestTradingCalendarTests(unittest.TestCase):
    def test_session_label_and_maintenance_gap(self) -> None:
        self.assertEqual(session_label_et(time(3, 49)), "OVERNIGHT_EARLY")
        self.assertIsNone(session_label_et(time(3, 50)))
        self.assertEqual(session_label_et(time(4, 0)), "PRE")
        self.assertEqual(session_label_et(time(9, 30)), "RTH")
        self.assertEqual(session_label_et(time(16, 0)), "POST")
        self.assertEqual(session_label_et(time(20, 0)), "OVERNIGHT_LATE")

    def test_xsp_weekly_sessions_are_gth_rth_and_curb(self) -> None:
        self.assertEqual(xsp_session_label_et(datetime(2026, 7, 24, 8, 34)), "GTH")
        self.assertIsNone(xsp_session_label_et(datetime(2026, 7, 24, 9, 25)))
        self.assertEqual(xsp_session_label_et(datetime(2026, 7, 24, 9, 30)), "RTH")
        self.assertEqual(xsp_session_label_et(datetime(2026, 7, 24, 16, 15)), "CURB")
        self.assertIsNone(xsp_session_label_et(datetime(2026, 7, 24, 17, 0)))
        self.assertIsNone(xsp_session_label_et(datetime(2026, 7, 24, 20, 15)))
        self.assertEqual(xsp_session_label_et(datetime(2026, 7, 26, 20, 15)), "GTH")

    def test_xsp_trading_date_carries_evening_gth_into_the_next_day(self) -> None:
        self.assertEqual(
            xsp_trading_date(datetime(2026, 7, 26, 20, 15)),
            date(2026, 7, 27),
        )
        self.assertEqual(
            xsp_trading_date(datetime(2026, 7, 27, 8, 34)),
            date(2026, 7, 27),
        )
        self.assertEqual(
            xsp_trading_date(datetime(2026, 7, 27, 16, 30)),
            date(2026, 7, 27),
        )
        self.assertEqual(
            xsp_trading_date(datetime(2026, 7, 27, 0, 30, tzinfo=timezone.utc)),
            date(2026, 7, 27),
        )
        self.assertIsNone(xsp_trading_date(datetime(2026, 7, 25, 10, 0)))

    def test_xsp_capture_window_spans_transition_gap_and_stops_at_close(self) -> None:
        self.assertEqual(
            xsp_capture_window_date(datetime(2026, 7, 26, 20, 15)),
            date(2026, 7, 27),
        )
        self.assertEqual(
            xsp_capture_window_date(datetime(2026, 7, 27, 9, 27)),
            date(2026, 7, 27),
        )
        self.assertEqual(
            xsp_capture_window_date(datetime(2026, 7, 27, 16, 59)),
            date(2026, 7, 27),
        )
        self.assertIsNone(
            xsp_capture_window_date(datetime(2026, 7, 27, 17, 0))
        )
        self.assertIsNone(
            xsp_capture_window_date(datetime(2026, 7, 31, 20, 15))
        )

    def test_special_closed_day_not_trading(self) -> None:
        self.assertFalse(is_trading_day(date(2025, 1, 9)))
        self.assertEqual(
            expected_sessions(date(2025, 1, 9), session_mode="full24"), set()
        )

    def test_expected_sessions_include_overnight_late_when_next_day_trades(
        self,
    ) -> None:
        sessions = expected_sessions(date(2025, 1, 13), session_mode="full24")
        self.assertEqual(
            tuple(sorted(sessions, key=lambda x: SESSION_ORDER.index(x))), SESSION_ORDER
        )

    def test_expected_sessions_skip_overnight_late_before_weekend(self) -> None:
        sessions = expected_sessions(date(2025, 1, 10), session_mode="full24")
        self.assertNotIn("OVERNIGHT_LATE", sessions)

    def test_et_utc_day_bounds_and_roundtrip_trade_day(self) -> None:
        start_utc, end_utc = utc_bounds_for_et_day(date(2025, 1, 15))
        self.assertEqual(start_utc, datetime(2025, 1, 15, 5, 0))
        self.assertEqual(end_utc, datetime(2025, 1, 16, 4, 59))
        self.assertEqual(
            et_day_from_utc_naive(datetime(2025, 1, 16, 0, 30)), date(2025, 1, 15)
        )

    def test_early_close_day_rules(self) -> None:
        self.assertTrue(is_early_close_day(date(2025, 11, 28)))
        self.assertTrue(is_early_close_day(date(2025, 12, 24)))
        self.assertFalse(is_early_close_day(date(2025, 12, 26)))

    def test_full24_post_close_cutoff_on_early_close_day(self) -> None:
        self.assertEqual(full24_post_close_time_et(date(2025, 11, 28)), time(17, 0))
        self.assertEqual(full24_post_close_time_et(date(2025, 12, 24)), time(17, 0))
        self.assertEqual(full24_post_close_time_et(date(2025, 12, 26)), time(20, 0))

    def test_market_breadth_is_causal_signed_observation_not_direction(self) -> None:
        observed_at = datetime(2026, 7, 24, 14, 0)
        samples = [
            (datetime(2026, 7, 23, 19, 55), 900.0),
            *(
                (datetime(2026, 7, 24, 13, 35) + timedelta(minutes=5 * index), value)
                for index, value in enumerate((0.0, 50.0, 100.0, 200.0, 150.0, 100.0))
            ),
            (datetime(2026, 7, 24, 14, 5), -999.0),
        ]
        observation = market_breadth_observation(
            samples,
            observed_at=observed_at,
            provider="IBKR",
            symbol="TICK-NASD",
            exchange="NASDAQ",
            proxy_for="XSP market breadth",
        )

        self.assertTrue(observation.ready)
        self.assertFalse(observation.stale)
        self.assertEqual(observation.sample_count, 6)
        self.assertEqual(observation.current, 100.0)
        self.assertEqual(observation.fast3, 150.0)
        self.assertEqual(observation.slow6, 100.0)
        self.assertEqual(observation.session_cumulative, 600.0)
        self.assertEqual(
            observation.relative_to("up"),
            {
                "alignment": 150.0,
                "transition_delta": 50.0,
                "transition": "improving",
            },
        )
        self.assertEqual(
            observation.relative_to("down")["transition"],
            "deteriorating",
        )
        payload = observation.as_payload("up")
        self.assertEqual(payload["authority"], "observation_only")
        self.assertEqual(payload["observed_at_utc"], "2026-07-24T14:00:00+00:00")
        self.assertEqual(payload["source_at_utc"], "2026-07-24T14:00:00+00:00")
        self.assertEqual(payload["reasons"], ())
        self.assertTrue(payload["usable"])


if __name__ == "__main__":
    unittest.main()
