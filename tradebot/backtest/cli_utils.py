"""Shared helpers for backtest CLI entrypoints.

Intentionally tiny: keep parsing consistent across the various scripts without
pulling in heavier dependencies.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from .data import cache_path as _cache_path_for_window


def parse_date(value: str) -> date:
    year_s, month_s, day_s = str(value).strip().split("-")
    return date(int(year_s), int(month_s), int(day_s))


def parse_window(value: str) -> tuple[date, date]:
    raw = str(value).strip()
    if ":" not in raw:
        raise ValueError("Window must be formatted like YYYY-MM-DD:YYYY-MM-DD")
    start_s, end_s = raw.split(":", 1)
    return parse_date(start_s), parse_date(end_s)


def expected_cache_path(
    *,
    cache_dir: Path,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
) -> Path:
    return _cache_path_for_window(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start_dt,
        end=end_dt,
        bar=bar_size,
        use_rth=use_rth,
    )
