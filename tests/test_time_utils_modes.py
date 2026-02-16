import unittest
from datetime import datetime

from tradebot.time_utils import NaiveTsMode, NaiveTsSourceMode, normalize_naive_ts_mode, to_et, to_utc_naive


class TimeUtilsModeTests(unittest.TestCase):
    def test_normalize_mode_accepts_enum_and_alias(self) -> None:
        self.assertEqual(normalize_naive_ts_mode(NaiveTsMode.ET), NaiveTsMode.ET)
        self.assertEqual(normalize_naive_ts_mode(NaiveTsSourceMode.ET_NAIVE), NaiveTsMode.ET)
        self.assertEqual(normalize_naive_ts_mode("america/new_york"), NaiveTsMode.ET)

    def test_to_et_with_enum_mode(self) -> None:
        ts = datetime(2025, 1, 16, 0, 30)
        out = to_et(ts, naive_ts_mode=NaiveTsMode.UTC)
        self.assertEqual(out.hour, 19)
        self.assertEqual(out.date().isoformat(), "2025-01-15")

    def test_to_utc_naive_source_modes(self) -> None:
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(to_utc_naive(ts, naive_ts_mode=NaiveTsSourceMode.UTC_NAIVE), ts)
        self.assertEqual(to_utc_naive(ts, naive_ts_mode=NaiveTsSourceMode.ET_NAIVE), datetime(2025, 1, 16, 5, 30))


if __name__ == "__main__":
    unittest.main()
