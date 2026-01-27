"""Small time-series helpers shared across UI/backtests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, TypeVar

from .signals import parse_bar_size


class _BarTsLike(Protocol):
    ts: datetime


TBar = TypeVar("TBar", bound=_BarTsLike)


def trim_incomplete_last_bar(
    bars: list[TBar],
    *,
    bar_size: str,
    now_ref: datetime | None = None,
) -> list[TBar]:
    """Drop the last bar if it appears to still be in-progress."""
    if not bars or len(bars) < 2:
        return bars
    bar_def = parse_bar_size(bar_size)
    if bar_def is None:
        return bars
    now = now_ref
    if now is None:
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    if getattr(now, "tzinfo", None) is not None:
        now = now.replace(tzinfo=None)
    last_ts = bars[-1].ts
    if getattr(last_ts, "tzinfo", None) is not None:
        last_ts = last_ts.replace(tzinfo=None)
    return bars[:-1] if (last_ts + bar_def.duration) > now else bars

