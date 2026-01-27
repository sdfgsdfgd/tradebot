"""Small shared date helpers (no market calendars).

These helpers intentionally implement a simple weekday-based "business day" definition:
- Monday..Friday are business days
- Saturday/Sunday are not

No holiday calendar is applied.
"""

from __future__ import annotations

from datetime import date, timedelta


def add_business_days(anchor: date, days: int) -> date:
    """Return `anchor` plus N weekday business days.

    - Counting starts *after* `anchor` (i.e. add_business_days(d, 0) == d).
    - Only Mon..Fri count; weekends are skipped.
    """
    current = anchor
    remaining = max(int(days), 0)
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def business_days_until(start: date, end: date) -> int:
    """Return the number of weekday business days in (start, end].

    Examples:
    - start=Mon, end=Tue -> 1
    - start=Mon, end=Mon -> 0
    - start=Fri, end=Mon -> 1 (Mon)
    """
    if end <= start:
        return 0
    days = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days
