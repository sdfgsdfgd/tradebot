from __future__ import annotations

from enum import Enum
from datetime import date, datetime, timezone
from typing import Union
from zoneinfo import ZoneInfo

ET_ZONE = ZoneInfo("America/New_York")
UTC = timezone.utc


class NaiveTsMode(str, Enum):
    UTC = "utc"
    ET = "et"


class NaiveTsSourceMode(str, Enum):
    UTC_NAIVE = "utc_naive"
    ET_NAIVE = "et_naive"


NaiveTsModeInput = Union[str, NaiveTsMode, NaiveTsSourceMode, None]


def _mode_token(value: NaiveTsModeInput, *, default: str = "utc") -> str:
    if isinstance(value, Enum):
        return str(value.value).strip().lower()
    return str(value or default).strip().lower()


def normalize_naive_ts_mode(naive_ts_mode: NaiveTsModeInput, *, default: str = "utc") -> NaiveTsMode:
    mode = _mode_token(naive_ts_mode, default=default)
    if mode in {"et", "et_naive", "america/new_york"}:
        return NaiveTsMode.ET
    return NaiveTsMode.UTC


def to_et(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None, default_naive_ts_mode: str = "utc") -> datetime:
    mode = normalize_naive_ts_mode(naive_ts_mode, default=default_naive_ts_mode)
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=ET_ZONE if mode == NaiveTsMode.ET else UTC)
    return ts.astimezone(ET_ZONE)


def to_utc_naive(
    ts: datetime,
    *,
    naive_ts_mode: NaiveTsModeInput = None,
    default_naive_ts_mode: str = "utc",
) -> datetime:
    mode = normalize_naive_ts_mode(naive_ts_mode, default=default_naive_ts_mode)
    if getattr(ts, "tzinfo", None) is None:
        if mode == NaiveTsMode.ET:
            return ts.replace(tzinfo=ET_ZONE).astimezone(UTC).replace(tzinfo=None)
        return ts
    return ts.astimezone(UTC).replace(tzinfo=None)


def trade_date(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None, default_naive_ts_mode: str = "utc") -> date:
    return to_et(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode=default_naive_ts_mode).date()


def trade_hour_et(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None, default_naive_ts_mode: str = "utc") -> int:
    return int(to_et(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode=default_naive_ts_mode).hour)


def trade_weekday(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None, default_naive_ts_mode: str = "utc") -> int:
    return int(to_et(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode=default_naive_ts_mode).weekday())


def now_et() -> datetime:
    return datetime.now(tz=ET_ZONE)


def now_et_naive() -> datetime:
    return now_et().replace(tzinfo=None)
