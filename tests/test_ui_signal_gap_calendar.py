from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

from tradebot.ui.bot import BotScreen


@dataclass(frozen=True)
class _TsBar:
    ts: datetime


def _screen() -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
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


def test_expected_live_full24_respects_early_close_post_cutoff() -> None:
    screen = _screen()
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2025, 12, 24, 16, 50),
        use_rth=False,
        sec_type="STK",
    ) is True
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2025, 12, 24, 17, 0),
        use_rth=False,
        sec_type="STK",
    ) is False
    assert screen._signal_expected_live_bars(
        now_ref=datetime(2025, 12, 24, 19, 50),
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


def test_intraday_halfday_tail_gap_is_not_flagged() -> None:
    screen = _screen()
    stats = screen._signal_gap_stats(
        bars=[_TsBar(datetime(2025, 12, 24, 16, 50)), _TsBar(datetime(2025, 12, 25, 20, 0))],
        bar_size="10 mins",
        use_rth=False,
        sec_type="STK",
        strict_zero_gap=True,
    )
    assert stats["gap_detected"] is False
    assert stats["gap_count"] == 0


def test_historical_only_gap_does_not_trigger_live_block() -> None:
    screen = _screen()
    start = datetime(2026, 2, 20, 0, 0)
    bars = []
    ts = start
    for idx in range(160):
        if idx == 20:
            ts += timedelta(minutes=10)
            continue
        bars.append(_TsBar(ts))
        ts += timedelta(minutes=10)
    stats = screen._signal_gap_stats(
        bars=bars,
        bar_size="10 mins",
        use_rth=False,
        sec_type="STK",
        strict_zero_gap=True,
    )
    assert stats["gap_count"] == 1
    assert stats["recent_gap_count"] == 0
    assert stats["gap_detected_any"] is True
    assert stats["gap_detected"] is False


def test_zero_gap_mode_is_not_strategy_configurable() -> None:
    screen = _screen()
    assert screen._signal_zero_gap_enabled(None) is True
    assert screen._signal_zero_gap_enabled({"signal_zero_gap_mode": "off"}) is True
    assert screen._signal_zero_gap_enabled({"zero_gap_mode": "relaxed"}) is True
