import unittest
from datetime import datetime

from tradebot.backtest.models import Bar
from tradebot.spot_engine import SpotSignalEvaluator


def _strategy() -> dict[str, object]:
    return {
        "entry_signal": "ema",
        "ema_preset": "1/2",
        "ema_entry_mode": "trend",
        "entry_confirm_bars": 0,
        "regime_mode": "ema",
        "regime_ema_preset": None,
        "spot_exit_mode": "pct",
    }


def _bar(ts: datetime, close: float) -> Bar:
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


def _risk_filters() -> dict[str, object]:
    return {
        "riskoff_tr5_med_pct": 0.01,
        "riskoff_tr5_lookback_days": 1,
    }


class SpotSignalTimeModeTests(unittest.TestCase):
    def test_utc_mode_keeps_same_et_trade_date_across_utc_midnight(self) -> None:
        evaluator = SpotSignalEvaluator(
            strategy=_strategy(),
            filters=None,
            bar_size="5 mins",
            use_rth=False,
            naive_ts_mode="utc",
        )
        first = evaluator.update_signal_bar(_bar(datetime(2025, 1, 15, 23, 55), 100.0))
        second = evaluator.update_signal_bar(_bar(datetime(2025, 1, 16, 0, 0), 101.0))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(int(first.bars_in_day), 1)
        self.assertEqual(int(second.bars_in_day), 2)

    def test_et_mode_resets_day_at_naive_midnight(self) -> None:
        evaluator = SpotSignalEvaluator(
            strategy=_strategy(),
            filters=None,
            bar_size="5 mins",
            use_rth=False,
            naive_ts_mode="et",
        )
        first = evaluator.update_signal_bar(_bar(datetime(2025, 1, 15, 23, 55), 100.0))
        second = evaluator.update_signal_bar(_bar(datetime(2025, 1, 16, 0, 0), 101.0))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(int(first.bars_in_day), 1)
        self.assertEqual(int(second.bars_in_day), 1)

    def test_utc_mode_risk_overlay_keeps_same_trade_day_across_utc_midnight(self) -> None:
        evaluator = SpotSignalEvaluator(
            strategy=_strategy(),
            filters=_risk_filters(),
            bar_size="1 min",
            use_rth=False,
            naive_ts_mode="utc",
        )
        evaluator.update_exec_bar(_bar(datetime(2025, 1, 15, 23, 55), 100.0), is_last_bar=False)
        self.assertIsNotNone(evaluator._risk_overlay)
        self.assertEqual(evaluator._risk_overlay._cur_day.isoformat(), "2025-01-15")

        evaluator.update_exec_bar(_bar(datetime(2025, 1, 16, 0, 5), 101.0), is_last_bar=False)
        self.assertEqual(evaluator._risk_overlay._cur_day.isoformat(), "2025-01-15")

    def test_et_mode_risk_overlay_resets_trade_day_at_naive_midnight(self) -> None:
        evaluator = SpotSignalEvaluator(
            strategy=_strategy(),
            filters=_risk_filters(),
            bar_size="1 min",
            use_rth=False,
            naive_ts_mode="et",
        )
        evaluator.update_exec_bar(_bar(datetime(2025, 1, 15, 23, 55), 100.0), is_last_bar=False)
        self.assertIsNotNone(evaluator._risk_overlay)
        self.assertEqual(evaluator._risk_overlay._cur_day.isoformat(), "2025-01-15")

        evaluator.update_exec_bar(_bar(datetime(2025, 1, 16, 0, 5), 101.0), is_last_bar=False)
        self.assertEqual(evaluator._risk_overlay._cur_day.isoformat(), "2025-01-16")


if __name__ == "__main__":
    unittest.main()
