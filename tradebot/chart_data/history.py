"""Canonical raw market-history persistence, discovery, and coverage."""

from __future__ import annotations

import csv
import hashlib
import math
import mmap
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..contract_identity import is_future_symbol
from ..engines.market import is_early_close_day, is_trading_day
from ..signals import parse_bar_size
from ..time_utils import ET_ZONE, NaiveTsMode, NaiveTsModeInput, now_et_naive, to_et, to_utc_naive
from .series import OhlcvBar, OhlcvBar as Bar


_APPROVED_CACHE_BAR_TOKENS = frozenset(
    {
        "1min",
        "2min",
        "2mins",
        "5min",
        "5mins",
        "10min",
        "10mins",
        "15min",
        "15mins",
        "30min",
        "30mins",
        "1hour",
        "4hour",
        "4hours",
        "1day",
    }
)
_CACHE_NAME_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9.\-]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?P<bar>[a-z0-9]+)_(?P<tag>rth|full24|full)\.csv$"
)


@dataclass(frozen=True)
class CacheFileMeta:
    symbol: str
    start_date: date
    end_date: date
    bar_token: str
    tag: str


@dataclass(frozen=True)
class HistoryWindow:
    bars: tuple[OhlcvBar, ...]
    missing_ranges: tuple[tuple[date, date], ...]
    source_paths: tuple[Path, ...]


def duration_window_et(
    duration_str: str,
    *,
    end: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve IBKR duration syntax into an ET-naive request window."""
    end_et = to_et(end, naive_ts_mode=NaiveTsMode.ET).replace(tzinfo=None) if end else now_et_naive()
    match = re.fullmatch(r"\s*(\d+)\s*([SDWMY])\s*", str(duration_str or "").upper())
    if match is None:
        raise ValueError(f"Unsupported IBKR duration: {duration_str!r}")
    count = max(1, int(match.group(1)))
    unit = match.group(2)
    span = {
        "S": timedelta(seconds=count),
        "D": timedelta(days=count),
        "W": timedelta(weeks=count),
        "M": timedelta(days=31 * count),
        "Y": timedelta(days=366 * count),
    }[unit]
    return end_et - span, end_et


def normalize_bars_to_close(
    bars: Iterable[OhlcvBar],
    *,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    naive_ts_mode: NaiveTsModeInput = NaiveTsMode.UTC,
) -> list[OhlcvBar]:
    """Convert IBKR bar-start timestamps to causal bar-close timestamps."""
    ordered = sorted(bars, key=lambda bar: bar.ts)
    if not ordered:
        return []
    close_time_et = _daily_close_time_et(symbol=symbol, use_rth=use_rth)
    if str(bar_size or "").strip().lower().startswith("1 day"):
        out: list[OhlcvBar] = []
        for bar in ordered:
            close_et = datetime.combine(bar.ts.date(), close_time_et, tzinfo=ET_ZONE)
            close_ts = (
                close_et.replace(tzinfo=None)
                if str(getattr(naive_ts_mode, "value", naive_ts_mode)).lower() in {"et", "et_naive"}
                else to_utc_naive(close_et)
            )
            out.append(_bar_at(bar, close_ts))
        return out

    bar_def = parse_bar_size(str(bar_size))
    if bar_def is None or bar_def.duration <= timedelta(0):
        return ordered
    duration = bar_def.duration
    out = []
    for index, bar in enumerate(ordered):
        close_ts = bar.ts + duration
        if index + 1 < len(ordered) and bar.ts < ordered[index + 1].ts < close_ts:
            close_ts = ordered[index + 1].ts
        if use_rth:
            start_et = to_et(bar.ts, naive_ts_mode=naive_ts_mode)
            session_close_et = datetime.combine(start_et.date(), close_time_et, tzinfo=ET_ZONE)
            session_close = (
                session_close_et.replace(tzinfo=None)
                if str(getattr(naive_ts_mode, "value", naive_ts_mode)).lower() in {"et", "et_naive"}
                else to_utc_naive(session_close_et)
            )
            close_ts = min(close_ts, session_close)
        out.append(_bar_at(bar, close_ts))
    return out


def _daily_close_time_et(*, symbol: str, use_rth: bool) -> time:
    if is_future_symbol(str(symbol or "").strip().upper()):
        return time(16, 0) if use_rth else time(17, 0)
    return time(16, 0) if use_rth else time(20, 0)


def _bar_at(bar: OhlcvBar, ts: datetime) -> OhlcvBar:
    return OhlcvBar(ts, bar.open, bar.high, bar.low, bar.close, bar.volume)


def load_history_window(
    *,
    cache_dir: Path,
    symbol: str,
    start_et: datetime,
    end_et: datetime,
    bar_size: str,
    use_rth: bool,
    naive_ts_mode: NaiveTsModeInput = NaiveTsMode.ET,
) -> HistoryWindow:
    """Read one canonical history view and identify absent trading days."""
    paths = find_overlapping_cache_paths(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start_et,
        end=end_et,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    mode = str(getattr(naive_ts_mode, "value", naive_ts_mode) or "").strip().lower()
    window_is_et = mode in {"et", "et_naive"}
    start_utc = (
        to_utc_naive(start_et, naive_ts_mode=NaiveTsMode.ET)
        if window_is_et
        else start_et
    )
    end_utc = (
        to_utc_naive(end_et, naive_ts_mode=NaiveTsMode.ET)
        if window_is_et
        else end_et
    )
    by_ts: dict[datetime, OhlcvBar] = {}
    for path in paths:
        for bar in read_cache(path, start=start_utc, end=end_utc):
            by_ts[bar.ts] = bar

    bars = tuple(by_ts[ts] for ts in sorted(by_ts))
    bar_def = parse_bar_size(str(bar_size))
    daily = bool(bar_def is not None and bar_def.duration >= timedelta(days=1))
    present_days = {
        (
            bar.ts.date()
            if daily or not window_is_et
            else to_et(bar.ts, naive_ts_mode=NaiveTsMode.UTC).date()
        )
        for bar in bars
    }
    missing_days = {
        day
        for day in _date_range(start_et.date(), end_et.date())
        if _day_intersects_window(day, start_et=start_et, end_et=end_et, use_rth=use_rth)
        and day not in present_days
    }
    missing_days.update(
        _incomplete_rth_days(
            bars,
            start_et=start_et,
            end_et=end_et,
            bar_size=bar_size,
            use_rth=use_rth,
            partial_window=window_is_et,
        )
    )
    return HistoryWindow(
        bars=bars,
        missing_ranges=tuple(_coalesce_days(missing_days)),
        source_paths=tuple(paths),
    )


def write_history_chunk(
    *,
    cache_dir: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    bar_size: str,
    use_rth: bool,
    bars: Iterable[OhlcvBar],
) -> Path | None:
    """Merge one fetched span into its canonical UTC-naive history shard."""
    rows = list(bars)
    if not rows:
        return None
    start = datetime.combine(start_date, datetime.min.time())
    end = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))
    path = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = [*read_cache(path), *rows]
    write_cache(path, rows)
    return path


def _date_range(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _day_intersects_window(
    day: date,
    *,
    start_et: datetime,
    end_et: datetime,
    use_rth: bool,
) -> bool:
    if not is_trading_day(day):
        return False
    session_start = time(9, 30) if use_rth else time(0, 0)
    if use_rth:
        session_end = time(12, 59, 59) if is_early_close_day(day) else time(15, 59, 59)
    else:
        session_end = time(23, 59, 59)
    return (
        datetime.combine(day, session_end) >= start_et
        and datetime.combine(day, session_start) <= end_et
    )


def _incomplete_rth_days(
    bars: Iterable[OhlcvBar],
    *,
    start_et: datetime,
    end_et: datetime,
    bar_size: str,
    use_rth: bool,
    partial_window: bool = False,
) -> set[date]:
    bar_def = parse_bar_size(str(bar_size))
    if (
        not use_rth
        or bar_def is None
        or bar_def.duration < timedelta(minutes=1)
        or bar_def.duration > timedelta(minutes=30)
    ):
        return set()
    bar_minutes = int(bar_def.duration.total_seconds() // 60)
    if bar_minutes <= 0:
        return set()
    timestamps: dict[date, set[datetime]] = defaultdict(set)
    for bar in bars:
        stamp = to_et(bar.ts, naive_ts_mode=NaiveTsMode.UTC).replace(tzinfo=None)
        timestamps[stamp.date()].add(stamp)

    incomplete: set[date] = set()
    for day in _date_range(start_et.date(), end_et.date()):
        if not is_trading_day(day):
            continue
        stamps = timestamps.get(day, set())
        close = time(13, 0) if is_early_close_day(day) else time(16, 0)
        session_start = datetime.combine(day, time(9, 30))
        session_end = datetime.combine(day, close)
        if partial_window:
            requested_start = max(session_start, start_et)
            requested_end = min(session_end, end_et)
            if requested_end <= requested_start:
                continue
            elapsed = max(0.0, (requested_start - session_start).total_seconds())
            steps = int(math.ceil(elapsed / bar_def.duration.total_seconds()))
            cursor = session_start + (bar_def.duration * steps)
            expected_stamps: set[datetime] = set()
            while cursor + bar_def.duration <= requested_end:
                expected_stamps.add(cursor)
                cursor += bar_def.duration
            if expected_stamps and not expected_stamps.issubset(stamps):
                incomplete.add(day)
            continue
        if start_et > session_start or end_et < session_end:
            continue
        expected = int((session_end - session_start).total_seconds() // bar_def.duration.total_seconds())
        ordered = sorted(stamps)
        if (
            not ordered
            or len(ordered) != expected
            or ordered[0] != session_start
            or ordered[-1] != session_end - bar_def.duration
            or any(current - previous != bar_def.duration for previous, current in zip(ordered, ordered[1:]))
        ):
            incomplete.add(day)
    return incomplete


def _coalesce_days(days: Iterable[date]) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    for day in sorted(set(days)):
        if ranges:
            cursor = ranges[-1][1] + timedelta(days=1)
            intervening_trading_day = False
            while cursor < day:
                intervening_trading_day = intervening_trading_day or is_trading_day(cursor)
                cursor += timedelta(days=1)
            if not intervening_trading_day:
                ranges[-1] = (ranges[-1][0], day)
                continue
        ranges.append((day, day))
    return ranges


def cache_data_revision(cache_dir: Path) -> str:
    """Fingerprint cached market-data files so derived results cannot outlive their tape."""
    root = Path(cache_dir)
    digest = hashlib.blake2b(digest_size=16)
    if not root.exists():
        return f"cache-v1:{digest.hexdigest()}"
    for path in sorted(root.rglob("*.csv")):
        if parse_cache_filename(path) is None:
            continue
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_ctime_ns)).encode("ascii"))
        digest.update(b"\n")
    return f"cache-v1:{digest.hexdigest()}"


def _canonical_bar_token(bar_size: str) -> str:
    parsed = parse_bar_size(str(bar_size))
    if parsed is not None:
        return parsed.label.replace(" ", "")
    return str(bar_size or "").strip().replace(" ", "").lower()


def _bar_token_variants(bar_size: str) -> tuple[str, ...]:
    canonical = _canonical_bar_token(bar_size)
    raw = str(bar_size or "").strip().replace(" ", "").lower()
    out: list[str] = []
    for token in (canonical, raw):
        if token and token not in out:
            out.append(token)
    return tuple(out)


def _intraday_resample_preferred(bar_size: str) -> bool:
    parsed = parse_bar_size(str(bar_size))
    if parsed is None:
        return False
    return timedelta(minutes=1) < parsed.duration < timedelta(days=1)


def parse_cache_filename(path: Path) -> CacheFileMeta | None:
    m = _CACHE_NAME_RE.match(path.name)
    if not m:
        return None
    bar_token = str(m.group("bar")).lower()
    if bar_token not in _APPROVED_CACHE_BAR_TOKENS:
        return None
    try:
        start_date = date.fromisoformat(str(m.group("start")))
        end_date = date.fromisoformat(str(m.group("end")))
    except ValueError:
        return None
    if end_date < start_date:
        return None
    return CacheFileMeta(
        symbol=str(m.group("symbol")),
        start_date=start_date,
        end_date=end_date,
        bar_token=bar_token,
        tag=str(m.group("tag")).lower(),
    )


def cache_path(cache_dir: Path, symbol: str, start: datetime, end: datetime, bar: str, use_rth: bool) -> Path:
    # Naming: `rth` for RTH-only. For non-RTH we prefer `full24` (explicitly meaning 24/5 for STK,
    # and "extended/full session" for everything else).
    tag = "rth" if use_rth else "full24"
    safe_bar = _canonical_bar_token(bar)
    return cache_dir / symbol / f"{symbol}_{start.date()}_{end.date()}_{safe_bar}_{tag}.csv"


def read_cache(
    path: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Bar]:
    stat = path.stat()
    return list(
        _read_cache_cached(
            str(path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            int(stat.st_ino),
            start.isoformat() if start is not None else "",
            end.isoformat() if end is not None else "",
        )
    )


@lru_cache(maxsize=32)
def _read_cache_cached(
    path: str,
    _mtime_ns: int = 0,
    _size: int = 0,
    _inode: int = 0,
    _start_iso: str = "",
    _end_iso: str = "",
) -> tuple[Bar, ...]:
    start = datetime.fromisoformat(_start_iso) if _start_iso else None
    end = datetime.fromisoformat(_end_iso) if _end_iso else None
    cache_path = Path(path)
    if (start is not None or end is not None) and parse_cache_filename(cache_path) is not None:
        sorted_window = _read_sorted_cache_window(cache_path, start=start, end=end)
        if sorted_window is not None:
            return sorted_window

    rows: list[Bar] = []
    with cache_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = datetime.fromisoformat(row["ts"])
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            rows.append(
                Bar(
                    ts=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    needs_fix = False
    prev_ts = None
    for bar in rows:
        if prev_ts is not None and bar.ts <= prev_ts:
            needs_fix = True
            break
        prev_ts = bar.ts
    if needs_fix:
        rows.sort(key=lambda b: b.ts)
        deduped: list[Bar] = []
        last_ts = None
        for bar in rows:
            if last_ts is not None and bar.ts == last_ts:
                deduped[-1] = bar
            else:
                deduped.append(bar)
                last_ts = bar.ts
        rows = deduped
    return tuple(rows)


def _read_sorted_cache_window(
    path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
) -> tuple[Bar, ...] | None:
    """Seek canonical sorted tapes; return None to retain the repair-capable reader."""
    if path.stat().st_size == 0:
        return ()
    rows: list[Bar] = []
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as tape:
        header = tape.readline().rstrip(b"\r\n")
        if header != b"ts,open,high,low,close,volume":
            return None
        data_start = tape.tell()

        if start is not None:
            lo, hi = data_start, len(tape)
            while lo < hi:
                mid = (lo + hi) // 2
                prior_break = tape.rfind(b"\n", data_start, mid)
                row_start = data_start if prior_break < 0 else prior_break + 1
                row_end = tape.find(b"\n", row_start)
                if row_end < 0:
                    row_end = len(tape)
                comma = tape.find(b",", row_start, row_end)
                if comma < 0:
                    return None
                ts = datetime.fromisoformat(tape[row_start:comma].decode("ascii"))
                if ts < start:
                    lo = min(len(tape), row_end + 1)
                else:
                    hi = row_start
            tape.seek(lo)
        else:
            tape.seek(data_start)

        for line in iter(tape.readline, b""):
            fields = line.rstrip(b"\r\n").split(b",")
            if len(fields) != 6:
                return None
            ts = datetime.fromisoformat(fields[0].decode("ascii"))
            if end is not None and ts > end:
                break
            rows.append(
                Bar(
                    ts=ts,
                    open=float(fields[1]),
                    high=float(fields[2]),
                    low=float(fields[3]),
                    close=float(fields[4]),
                    volume=float(fields[5]),
                )
            )
    return tuple(rows)


def write_cache(path: Path, bars: Iterable[Bar]) -> None:
    rows = list(bars)
    if not rows:
        return
    rows.sort(key=lambda b: b.ts)
    deduped: list[Bar] = []
    last_ts = None
    for bar in rows:
        if last_ts is not None and bar.ts == last_ts:
            deduped[-1] = bar
        else:
            deduped.append(bar)
            last_ts = bar.ts

    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in deduped:
            writer.writerow(
                {
                    "ts": bar.ts.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )
    os.replace(tmp, path)


def find_covering_cache_path(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> Path | None:
    folder = cache_dir / symbol
    if not folder.exists():
        return None

    bar_tokens = set(_bar_token_variants(bar_size))
    tags = ("rth",) if use_rth else ("full24", "full")
    tag_rank = {tag: rank for rank, tag in enumerate(tags)}
    symbol_upper = str(symbol).upper()
    start_d = start.date()
    end_d = end.date()
    candidates: list[tuple[int, int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        meta = parse_cache_filename(path)
        if meta is None:
            continue
        if str(meta.symbol).upper() != symbol_upper:
            continue
        if meta.tag not in tag_rank:
            continue
        if meta.bar_token not in bar_tokens:
            continue
        if meta.start_date <= start_d and meta.end_date >= end_d:
            span_days = (meta.end_date - meta.start_date).days
            candidates.append((tag_rank[meta.tag], span_days, path))
    if candidates:
        # Prefer the broadest covering cache. Narrower window-specific caches can
        # diverge from the canonical wide-range cache (IBKR replay drift), which
        # breaks deterministic backtest parity.
        candidates.sort(key=lambda t: (t[0], -t[1]))
        return candidates[0][2]
    return None


def find_overlapping_cache_paths(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> list[Path]:
    folder = cache_dir / symbol
    if not folder.exists():
        return []

    bar_tokens = set(_bar_token_variants(bar_size))
    tags = ("rth",) if use_rth else ("full24", "full")
    tag_rank = {tag: rank for rank, tag in enumerate(tags)}
    symbol_upper = str(symbol).upper()
    start_d = start.date()
    end_d = end.date()
    candidates: list[tuple[int, date, date, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        meta = parse_cache_filename(path)
        if meta is None:
            continue
        if str(meta.symbol).upper() != symbol_upper:
            continue
        if meta.tag not in tag_rank:
            continue
        if meta.bar_token not in bar_tokens:
            continue
        if meta.start_date > end_d or meta.end_date < start_d:
            continue
        candidates.append((tag_rank[meta.tag], meta.start_date, meta.end_date, path))
    if not candidates:
        return []
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return [c[3] for c in candidates]


def cache_covers_window(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> tuple[bool, Path | None, list[tuple[date, date]]]:
    """Return content-aware coverage for one canonical UTC-naive window."""
    history = load_history_window(
        cache_dir=cache_dir,
        symbol=symbol,
        start_et=start,
        end_et=end,
        bar_size=bar_size,
        use_rth=use_rth,
        naive_ts_mode=NaiveTsMode.UTC,
    )
    missing_ranges = list(history.missing_ranges)
    if not history.bars and not missing_ranges:
        missing_ranges = [(start.date(), end.date())]
    covering = history.source_paths[0] if len(history.source_paths) == 1 else None
    return bool(history.bars and not missing_ranges), covering, missing_ranges


def ensure_offline_cached_window(
    *,
    cache_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> tuple[bool, Path, Path | None, list[tuple[date, date]], str | None]:
    """Validate whether canonical files cover an offline request.

    Returns:
      (ok, expected_path, resolved_path, missing_ranges, error_text)
    """
    expected = cache_path(cache_dir, symbol, start, end, bar_size, use_rth)
    cache_ok, covering, missing_ranges = cache_covers_window(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    if not cache_ok:
        return False, expected, covering, missing_ranges, None

    resolved = covering if covering is not None else (expected if expected.exists() else None)
    return True, expected, resolved, [], None


def _uncovered_date_ranges(
    *,
    request_start: date,
    request_end: date,
    covered_ranges: Iterable[tuple[date, date]],
) -> list[tuple[date, date]]:
    merged: list[tuple[date, date]] = []
    for raw_start, raw_end in sorted(covered_ranges):
        cur_start = max(raw_start, request_start)
        cur_end = min(raw_end, request_end)
        if cur_end < cur_start:
            continue
        if not merged:
            merged.append((cur_start, cur_end))
            continue
        prev_start, prev_end = merged[-1]
        if cur_start <= (prev_end + timedelta(days=1)):
            merged[-1] = (prev_start, max(prev_end, cur_end))
            continue
        merged.append((cur_start, cur_end))

    if not merged:
        return [(request_start, request_end)]

    out: list[tuple[date, date]] = []
    cursor = request_start
    for cov_start, cov_end in merged:
        if cov_start > cursor:
            out.append((cursor, cov_start - timedelta(days=1)))
        cursor = max(cursor, cov_end + timedelta(days=1))
        if cursor > request_end:
            break
    if cursor <= request_end:
        out.append((cursor, request_end))
    return out
