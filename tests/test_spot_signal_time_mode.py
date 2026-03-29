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
    def test_regime_router_overrides_entry_dir_and_emits_metadata(self) -> None:
        strategy = {
            **_strategy(),
            "regime_router": True,
            "regime_router_fast_window_days": 2,
            "regime_router_slow_window_days": 4,
            "regime_router_min_dwell_days": 1,
        }
        evaluator = SpotSignalEvaluator(
            strategy=strategy,
            filters=None,
            bar_size="5 mins",
            use_rth=False,
            naive_ts_mode="utc",
        )
        bars = [
            _bar(datetime(2025, 1, 1, 23, 55), 100.0),
            _bar(datetime(2025, 1, 2, 23, 55), 101.0),
            _bar(datetime(2025, 1, 3, 23, 55), 102.0),
            _bar(datetime(2025, 1, 4, 23, 55), 103.0),
            _bar(datetime(2025, 1, 5, 23, 55), 101.0),
        ]
        snap = None
        for bar in bars:
            snap = evaluator.update_signal_bar(bar)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertTrue(bool(snap.regime_router_ready))
        self.assertEqual(str(snap.regime_router_host), "buyhold")
        self.assertEqual(str(snap.entry_dir), "up")

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
