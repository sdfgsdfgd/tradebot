"""Canonical backtest cache persistence, discovery, and coverage."""

from __future__ import annotations

import csv
import hashlib
import mmap
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..signals import parse_bar_size
from .models import Bar

if TYPE_CHECKING:
    from .data import IBKRHistoricalData


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
    """Return whether cache files fully cover a requested window.

    Coverage is true if either:
    - one covering cache file exists, or
    - overlapping cache files can fully span the requested date range.

    Returns:
      (covers, covering_path, missing_ranges)
    where `covering_path` is set only when a single covering file exists, and
    `missing_ranges` contains uncovered date spans when coverage is incomplete.
    """
    covering = find_covering_cache_path(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if covering is not None:
        return True, covering, []

    overlap_paths = find_overlapping_cache_paths(
        cache_dir=cache_dir,
        symbol=symbol,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if not overlap_paths:
        return False, None, [(start.date(), end.date())]

    covered_ranges: list[tuple[date, date]] = []
    for overlap in overlap_paths:
        meta = parse_cache_filename(overlap)
        if meta is not None:
            covered_ranges.append((meta.start_date, meta.end_date))

    missing_ranges = _uncovered_date_ranges(
        request_start=start.date(),
        request_end=end.date(),
        covered_ranges=covered_ranges,
    )
    if missing_ranges:
        return False, None, missing_ranges
    return True, None, []


def ensure_offline_cached_window(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
    hydrate_overlap: bool = True,
) -> tuple[bool, Path, Path | None, list[tuple[date, date]], str | None]:
    """Validate offline cache availability using runtime-equivalent loader logic.

    Returns:
      (ok, expected_path, resolved_path, missing_ranges, error_text)

    Behavior:
    - Fast path checks filename coverage metadata first.
    - If coverage is overlap-only and no exact file exists, optionally hydrate by
      invoking `load_cached_bar_series`, which performs deterministic stitch and
      persists the exact cache window.
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
    if not bool(hydrate_overlap):
        return True, expected, resolved, [], None
    if resolved is not None:
        return True, expected, resolved, [], None

    # Coverage is metadata-complete but exact file is missing. Reuse the same
    # loader path as runtime so stitched windows are materialized consistently.
    try:
        series = data.load_cached_bar_series(
            symbol=str(symbol),
            exchange=exchange,
            start=start,
            end=end,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            cache_dir=cache_dir,
        )
    except FileNotFoundError as exc:
        cache_ok2, covering2, missing_ranges2 = cache_covers_window(
            cache_dir=cache_dir,
            symbol=str(symbol),
            start=start,
            end=end,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
        )
        missing_eff = missing_ranges2 if missing_ranges2 else missing_ranges
        resolved_eff = covering2 if cache_ok2 else None
        return False, expected, resolved_eff, missing_eff, str(exc)

    source_path = str(getattr(series.meta, "source_path", "") or "").strip()
    if source_path:
        try:
            resolved = Path(source_path)
        except Exception:
            resolved = expected if expected.exists() else None
    elif expected.exists():
        resolved = expected
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
