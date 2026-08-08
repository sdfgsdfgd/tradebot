"""Session-bounded launcher for the immutable MCL turn-tape recorder."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .mcl_turn_tape import main as run_turn_tape


_ET = ZoneInfo("America/New_York")
_PREOPEN = time(17, 55)
_CLOSE = time(17, 0)


def session_seconds(now: datetime) -> float:
    if now.tzinfo is None:
        raise ValueError("MCL turn-tape session clock must be timezone-aware")
    current = now.astimezone(_ET)
    weekday = current.weekday()
    clock = current.timetz().replace(tzinfo=None)
    if weekday == 5 or (weekday == 6 and clock < _PREOPEN):
        return 0.0
    if (weekday == 4 and clock >= _CLOSE) or _CLOSE <= clock < _PREOPEN:
        return 0.0
    close_day = current.date() + (timedelta(days=1) if clock >= _PREOPEN else timedelta())
    close = datetime.combine(close_day, _CLOSE, tzinfo=_ET)
    return max(0.0, (close.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds())


def main() -> int:
    remaining = session_seconds(datetime.now(timezone.utc))
    if remaining <= 0:
        return 0
    return run_turn_tape(("--duration-sec", str(remaining)))


if __name__ == "__main__":
    raise SystemExit(main())
