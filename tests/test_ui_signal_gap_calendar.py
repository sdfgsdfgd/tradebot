from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

from tradebot.ui.bot import BotScreen


@dataclass(frozen=True)
class _TsBar:
    ts: datetime


def _screen() -> BotScreen:
    return BotScreen(client=SimpleNamespace(), refresh_sec=1.0)


def test_expected_live_rth_is_false_on_holiday_for_stk() -> None:
    screen = _screen()
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2026, 1, 1, 12, 0),
        use_rth=True,
        sec_type="STK",
    ) is False
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2026, 1, 2, 12, 0),
        use_rth=True,
        sec_type="STK",
    ) is True


def test_expected_live_full24_uses_session_day_for_overnight_late() -> None:
    screen = _screen()
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2026, 1, 4, 20, 30),
        use_rth=False,
        sec_type="STK",
    ) is True
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2026, 9, 6, 20, 30),
        use_rth=False,
        sec_type="STK",
    ) is False


def test_daily_holiday_gap_is_not_flagged() -> None:
    screen = _screen()
    stats = screen._signal_gap_stats(
        bars=[_TsBar(datetime(2025, 12, 24, 0, 0)), _TsBar(datetime(2025, 12, 26, 0, 0))],
        bar_size="1 day",
        use_rth=False,
        sec_type="STK",
        strict_zero_gap=True,
    )
    assert stats["gap_detected"] is False
    assert stats["gap_count"] == 0


def test_daily_missing_trading_day_is_flagged_even_with_weekend_holiday_mix() -> None:
    screen = _screen()
    stats = screen._signal_gap_stats(
        bars=[_TsBar(datetime(2026, 2, 12, 0, 0)), _TsBar(datetime(2026, 2, 17, 0, 0))],
        bar_size="1 day",
        use_rth=False,
        sec_type="STK",
        strict_zero_gap=True,
    )
    assert stats["gap_detected"] is True
    assert stats["gap_count"] == 1
    assert stats["max_gap_bars"] == 2.0
