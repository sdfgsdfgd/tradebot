from __future__ import annotations

from datetime import datetime

import pytest

from tradebot.research.mcl_turn_tape_session import session_seconds


@pytest.mark.parametrize(
    ("now", "remaining"),
    (
        ("2026-08-08T16:00:00-04:00", 0),
        ("2026-08-09T17:54:00-04:00", 0),
        ("2026-08-09T17:55:00-04:00", 23 * 3600 + 5 * 60),
        ("2026-08-10T10:00:00-04:00", 7 * 3600),
        ("2026-08-10T17:30:00-04:00", 0),
        ("2026-08-14T16:59:00-04:00", 60),
        ("2026-08-14T17:00:00-04:00", 0),
    ),
)
def test_turn_tape_is_bounded_to_each_tradable_session(now: str, remaining: int) -> None:
    assert session_seconds(datetime.fromisoformat(now)) == remaining
