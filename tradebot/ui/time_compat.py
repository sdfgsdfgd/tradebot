"""UI time compatibility shim.

Keeps standalone-module test imports resilient while preferring shared time utils.
"""

from __future__ import annotations

from datetime import datetime

try:  # pragma: no cover - normal runtime path
    from tradebot.time_utils import ET_ZONE, now_et, now_et_naive
except Exception:  # pragma: no cover - standalone import fallback
    from zoneinfo import ZoneInfo

    ET_ZONE = ZoneInfo("America/New_York")

    def now_et() -> datetime:
        return datetime.now(tz=ET_ZONE)

    def now_et_naive() -> datetime:
        return now_et().replace(tzinfo=None)


__all__ = ["ET_ZONE", "now_et", "now_et_naive"]
