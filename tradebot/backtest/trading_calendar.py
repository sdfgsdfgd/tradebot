from __future__ import annotations

from datetime import date, datetime, time, timedelta

from ..time_utils import ET_ZONE, to_utc_naive, trade_date

SESSION_ORDER = (
    "OVERNIGHT_EARLY",
    "PRE",
    "RTH",
    "POST",
    "OVERNIGHT_LATE",
)

SESSION_WEIGHTS = {
    "OVERNIGHT_EARLY": 20,
    "PRE": 50,
    "RTH": 100,
    "POST": 60,
    "OVERNIGHT_LATE": 20,
}

# Known one-off NYSE full-closure dates (in addition to recurring holiday rules).
NYSE_SPECIAL_CLOSED_DAYS = {
    date(2018, 12, 5),  # National Day of Mourning (George H. W. Bush)
    date(2025, 1, 9),   # National Day of Mourning (Jimmy Carter)
}

_HOLIDAY_CACHE: dict[int, set[date]] = {}
_EARLY_CLOSE_CACHE: dict[int, set[date]] = {}


def is_maintenance_gap(t: time) -> bool:
    return time(3, 50) <= t <= time(3, 59)


def session_label_et(t: time) -> str | None:
    if is_maintenance_gap(t):
        return None
    if time(0, 0) <= t <= time(3, 49):
        return "OVERNIGHT_EARLY"
    if time(4, 0) <= t <= time(9, 29):
        return "PRE"
    if time(9, 30) <= t <= time(15, 59):
        return "RTH"
    if time(16, 0) <= t <= time(19, 59):
        return "POST"
    return "OVERNIGHT_LATE"


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def _observed_holiday(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _easter_sunday(year: int) -> date:
    # Meeus/Jones/Butcher algorithm (Gregorian calendar).
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    out: set[date] = set()
    out.add(_observed_holiday(date(year, 1, 1)))
    observed_next_new_year = _observed_holiday(date(year + 1, 1, 1))
    if observed_next_new_year.year == year:
        out.add(observed_next_new_year)
    out.add(_nth_weekday(year, 1, 0, 3))
    out.add(_nth_weekday(year, 2, 0, 3))
    out.add(_easter_sunday(year) - timedelta(days=2))
    out.add(_last_weekday(year, 5, 0))
    out.add(_observed_holiday(date(year, 6, 19)))
    out.add(_observed_holiday(date(year, 7, 4)))
    out.add(_nth_weekday(year, 9, 0, 1))
    out.add(_nth_weekday(year, 11, 3, 4))
    out.add(_observed_holiday(date(year, 12, 25)))
    return out


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d in NYSE_SPECIAL_CLOSED_DAYS:
        return False
    years = {d.year - 1, d.year, d.year + 1}
    holidays: set[date] = set()
    for y in years:
        cached = _HOLIDAY_CACHE.get(y)
        if cached is None:
            cached = _nyse_holidays(y)
            _HOLIDAY_CACHE[y] = cached
        for hd in cached:
            if hd.year == d.year:
                holidays.add(hd)
    return d not in holidays


def _nyse_early_closes(year: int) -> set[date]:
    out: set[date] = set()
    # Day after Thanksgiving (Black Friday).
    black_friday = _nth_weekday(year, 11, 3, 4) + timedelta(days=1)
    if is_trading_day(black_friday):
        out.add(black_friday)
    # Christmas Eve.
    christmas_eve = date(year, 12, 24)
    if is_trading_day(christmas_eve):
        out.add(christmas_eve)
    # Independence Day eve when it remains a trading day.
    july_3 = date(year, 7, 3)
    if is_trading_day(july_3):
        out.add(july_3)
    return out


def is_early_close_day(d: date) -> bool:
    if not is_trading_day(d):
        return False
    cached = _EARLY_CLOSE_CACHE.get(d.year)
    if cached is None:
        cached = _nyse_early_closes(d.year)
        _EARLY_CLOSE_CACHE[d.year] = cached
    return d in cached


def full24_post_close_time_et(day: date) -> time:
    # IBKR full-session equity bars typically stop at 17:00 ET on NYSE half-days.
    return time(17, 0) if is_early_close_day(day) else time(20, 0)


def expected_sessions(day: date, *, session_mode: str) -> set[str]:
    if not is_trading_day(day):
        return set()
    if session_mode == "rth":
        return {"RTH"}
    if session_mode == "smart_ext":
        return {"PRE", "RTH", "POST"}
    out = {"OVERNIGHT_EARLY", "PRE", "RTH", "POST"}
    if is_trading_day(day + timedelta(days=1)):
        out.add("OVERNIGHT_LATE")
    return out


def utc_bounds_for_et_day(day: date) -> tuple[datetime, datetime]:
    start_et = datetime.combine(day, time(0, 0), tzinfo=ET_ZONE)
    end_et = datetime.combine(day, time(23, 59), tzinfo=ET_ZONE)
    return to_utc_naive(start_et), to_utc_naive(end_et)


def et_day_from_utc_naive(ts: datetime) -> date:
    return trade_date(ts, naive_ts_mode="utc")
