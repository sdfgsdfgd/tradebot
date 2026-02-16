import unittest
from datetime import datetime

from tradebot.engine import _trade_date, _trade_hour_et, _trade_weekday


class EngineTimeSemanticsTests(unittest.TestCase):
    def test_trade_date_utc_naive_default(self) -> None:
        # 2025-01-16 00:30 UTC == 2025-01-15 19:30 ET.
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(_trade_date(ts).isoformat(), "2025-01-15")

    def test_trade_date_et_naive_mode(self) -> None:
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(_trade_date(ts, naive_ts_mode="et").isoformat(), "2025-01-16")

    def test_trade_hour_et_with_modes(self) -> None:
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(_trade_hour_et(ts), 19)
        self.assertEqual(_trade_hour_et(ts, naive_ts_mode="et"), 0)

    def test_trade_weekday_with_modes(self) -> None:
        # UTC-naive interpret: ET Wednesday (2). ET-naive interpret: Thursday (3).
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(_trade_weekday(ts), 2)
        self.assertEqual(_trade_weekday(ts, naive_ts_mode="et"), 3)


if __name__ == "__main__":
    unittest.main()
