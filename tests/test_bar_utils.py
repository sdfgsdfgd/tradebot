from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from tradebot.utils.bar_utils import trim_incomplete_last_bar


@dataclass(frozen=True)
class _Bar:
    ts: datetime | date


def test_trim_incomplete_last_bar_handles_date_timestamps() -> None:
    bars = [_Bar(ts=date(2026, 2, 5)), _Bar(ts=date(2026, 2, 6))]

    out = trim_incomplete_last_bar(
        bars,
        bar_size="1 day",
        now_ref=datetime(2026, 2, 6, 7, 45, 12),
    )

    assert out == [_Bar(ts=date(2026, 2, 5))]


def test_trim_incomplete_last_bar_keeps_complete_daily_bar() -> None:
    bars = [_Bar(ts=date(2026, 2, 5)), _Bar(ts=date(2026, 2, 6))]

    out = trim_incomplete_last_bar(
        bars,
        bar_size="1 day",
        now_ref=datetime(2026, 2, 7, 0, 1, 0),
    )

    assert out == bars
