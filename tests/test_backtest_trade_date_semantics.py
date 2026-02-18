import unittest
from dataclasses import dataclass
from datetime import date, datetime

from tradebot.backtest.engine import (
    _spot_fill_due_ts,
    _spot_next_open_due_ts,
    _spot_next_open_entry_allowed,
    _spot_pending_entry_should_cancel,
    _trade_date,
)
from tradebot.backtest.strategy import CreditSpreadStrategy


@dataclass(frozen=True)
class _StrategyCfg:
    entry_days: tuple[int, ...]
    dte: int
    legs: tuple = ()
    quantity: int = 1
    otm_pct: float = 0.0
    right: str = "PUT"
    width_pct: float = 0.0


class BacktestTradeDateSemanticsTests(unittest.TestCase):
    def test_trade_date_uses_et_for_utc_naive_timestamp(self) -> None:
        # 2025-01-16 00:30 UTC is 2025-01-15 19:30 ET.
        ts = datetime(2025, 1, 16, 0, 30)
        self.assertEqual(_trade_date(ts), date(2025, 1, 15))

    def test_next_open_riskoff_day_check_uses_et_date(self) -> None:
        # Same ET day, different UTC dates (18:55 ET -> 19:05 ET).
        signal_ts = datetime(2025, 1, 15, 23, 55)
        next_ts = datetime(2025, 1, 16, 0, 5)
        ok = _spot_next_open_entry_allowed(
            signal_ts=signal_ts,
            next_ts=next_ts,
            riskoff_today=True,
            riskoff_end_hour=None,
            exit_mode="pct",
            atr_value=None,
        )
        self.assertTrue(ok)

    def test_pending_entry_cancel_day_check_uses_et_date(self) -> None:
        # Pending date is ET trade date; exec ts remains same ET day despite UTC rollover.
        should_cancel = _spot_pending_entry_should_cancel(
            pending_dir="up",
            pending_set_date=date(2025, 1, 15),
            exec_ts=datetime(2025, 1, 16, 0, 30),  # 19:30 ET on Jan 15
            risk_overlay_enabled=True,
            riskoff_today=True,
            riskpanic_today=False,
            riskpop_today=False,
            riskoff_mode="hygiene",
            shock_dir_now=None,
            riskoff_end_hour=None,
        )
        self.assertFalse(should_cancel)

    def test_pending_entry_cancel_cutoff_hour_uses_et_hour(self) -> None:
        # 00:30 UTC == 19:30 ET, so cutoff at 19 should cancel.
        should_cancel = _spot_pending_entry_should_cancel(
            pending_dir="up",
            pending_set_date=date(2025, 1, 15),
            exec_ts=datetime(2025, 1, 16, 0, 30),
            risk_overlay_enabled=True,
            riskoff_today=True,
            riskpanic_today=False,
            riskpop_today=False,
            riskoff_mode="hygiene",
            shock_dir_now=None,
            riskoff_end_hour=19,
        )
        self.assertTrue(should_cancel)

    def test_next_open_due_defaults_to_tradable_24x5_when_signal_use_rth_false(self) -> None:
        strategy = {"spot_sec_type": "STK", "signal_use_rth": False}
        # 2026-02-10 02:20 UTC == 2026-02-09 21:20 ET (overnight tradable window).
        due = _spot_next_open_due_ts(
            strategy=strategy,
            signal_close_ts=datetime(2026, 2, 10, 2, 20),
            exec_bar_size="5 mins",
        )
        self.assertEqual(due, datetime(2026, 2, 10, 2, 20))

    def test_next_open_due_extended_mode_waits_until_0400_et(self) -> None:
        strategy = {
            "spot_sec_type": "STK",
            "signal_use_rth": False,
            "spot_next_open_session": "extended",
        }
        # 2026-02-10 02:20 UTC == 2026-02-09 21:20 ET; extended next open is 04:00 ET (09:00 UTC).
        due = _spot_next_open_due_ts(
            strategy=strategy,
            signal_close_ts=datetime(2026, 2, 10, 2, 20),
            exec_bar_size="5 mins",
        )
        self.assertEqual(due, datetime(2026, 2, 10, 9, 0))

    def test_fill_due_next_bar_aligns_to_next_exec_boundary(self) -> None:
        due = _spot_fill_due_ts(
            strategy={"spot_sec_type": "STK", "signal_use_rth": False},
            fill_mode="next_bar",
            signal_close_ts=datetime(2026, 2, 10, 2, 21),
            exec_bar_size="5 mins",
        )
        self.assertEqual(due, datetime(2026, 2, 10, 2, 25))

    def test_fill_due_legacy_next_open_alias_matches_next_tradable_bar(self) -> None:
        strategy = {
            "spot_sec_type": "STK",
            "signal_use_rth": False,
            "spot_next_open_session": "extended",
        }
        due_legacy = _spot_fill_due_ts(
            strategy=strategy,
            fill_mode="next_open",
            signal_close_ts=datetime(2026, 2, 10, 2, 20),
            exec_bar_size="5 mins",
        )
        due_canonical = _spot_fill_due_ts(
            strategy=strategy,
            fill_mode="next_tradable_bar",
            signal_close_ts=datetime(2026, 2, 10, 2, 20),
            exec_bar_size="5 mins",
        )
        self.assertEqual(due_legacy, due_canonical)

    def test_strategy_should_enter_uses_et_weekday(self) -> None:
        # 2025-01-16 00:30 UTC is Wednesday ET (weekday=2).
        strat = CreditSpreadStrategy(_StrategyCfg(entry_days=(2,), dte=0))
        self.assertTrue(strat.should_enter(datetime(2025, 1, 16, 0, 30)))

    def test_strategy_expiry_anchor_uses_et_trade_date(self) -> None:
        # dte=1 from ET Jan 15 should expire on ET Jan 16.
        strat = CreditSpreadStrategy(_StrategyCfg(entry_days=(2,), dte=1))
        spec = strat.build_spec(datetime(2025, 1, 16, 0, 30), spot=25.0)
        self.assertEqual(spec.expiry, date(2025, 1, 16))


if __name__ == "__main__":
    unittest.main()
