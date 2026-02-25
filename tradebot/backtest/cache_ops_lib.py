"""Unified cache maintenance CLI for sync, repair, and resample.

Subcommands:
- sync: threaded cache retrieval with built-in audit/heal and timezone canonicalization.
- repair: ET session completeness + timezone canonicalization + targeted healing for existing caches.
- resample: deterministic larger-bar derivation from an existing cache (no IBKR refetch).

Examples:
  python -m tradebot.backtest.tools.cache_ops repair \
    --cache-file db/SLV/SLV_2025-02-14_2026-02-14_1min_full24.csv \
    --cache-file db/SLV/SLV_2024-02-14_2026-02-14_1min_full24.csv \
    --archive-file db/SLV/SLV_2024-01-08_2026-01-08_1min_full24.csv \
    --heal --aggressive --threads 10 --retries 3 --timeout-sec 45
  python -m tradebot.backtest.tools.cache_ops sync --champion-current --aggressive
  python -m tradebot.backtest.tools.cache_ops resample --symbol SLV --start 2025-01-08 --end 2026-01-08 --src-bar-size "5 mins" --dst-bar-size "10 mins"
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

from .data import (
    CacheFileMeta,
    IBKRHistoricalData,
    cache_path,
    ensure_offline_cached_window,
    parse_cache_filename,
    read_cache,
    write_cache,
)
from .spot_context import SpotBarRequirement, spot_bar_requirements_from_strategy
from .models import Bar
from .trading_calendar import (
    SESSION_ORDER as _SESSION_ORDER,
    SESSION_WEIGHTS as _SESSION_WEIGHTS,
    et_day_from_utc_naive as _et_day_from_utc_naive,
    expected_sessions as _expected_sessions,
    is_maintenance_gap as _is_maintenance_gap,
    session_label_et as _session_label_et,
    utc_bounds_for_et_day as _utc_bounds_for_et_day,
)
from ..signals import parse_bar_size
from ..time_utils import (
    UTC as _UTC,
    NaiveTsMode,
    NaiveTsSourceMode,
    to_et as _to_et_shared,
    to_utc_naive as _to_utc_naive_shared,
)

_FETCH_SAFE_THREAD_MAX = 2

def _require_cache_meta(path: Path) -> CacheFileMeta:
    meta = parse_cache_filename(path)
    if meta is not None:
        return meta
    raise SystemExit(
        f"Unable to parse symbol/start/end from cache filename: {path.name}. "
        "Expected pattern: SYMBOL_YYYY-MM-DD_YYYY-MM-DD_<approved_bar>_<rth|full24|full>.csv"
    )


@dataclass(frozen=True)
class _FetchResult:
    day: date
    ok: bool
    attempts: int
    smart_rows: int
    overnight_rows: int
    merged_rows: int
    error: str | None
    bars: tuple[Bar, ...]


@dataclass(frozen=True)
class _ResampleStats:
    kept: int
    dropped_incomplete: int


@dataclass(frozen=True)
class CacheResampleOutcome:
    ok: bool
    dst_path: Path
    src_bar_size: str | None = None
    src_path: Path | None = None
    src_rows: int = 0
    dst_rows: int = 0
    dropped_incomplete: int = 0
    error: str | None = None


@dataclass(frozen=True)
class _CacheFetchRequest:
    symbol: str
    start: date
    end: date
    bar_size: str
    use_rth: bool
    source: str


@dataclass(frozen=True)
class _CacheFetchOutcome:
    request: _CacheFetchRequest
    ok: bool
    from_cache: bool
    rows: int
    cache_path: str
    error: str | None
    status: str = "ok"  # ok | warn | fail
    healed: bool = False
    healed_days: int = 0
    remaining_days: int = 0
    session_missing_days_before: int = 0
    session_missing_days_after: int = 0
    gap_days_before: int = 0
    gap_days_after: int = 0
    timezone_canonicalized: bool = False
    anomaly_fingerprint: str | None = None


@dataclass(frozen=True)
class _CacheFetchBatch:
    primary: _CacheFetchRequest
    targets: tuple[_CacheFetchRequest, ...]


def _parse_date(value: str) -> date:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise SystemExit("Missing date")
    try:
        return date.fromisoformat(cleaned)
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {value!r}") from exc


def _infer_timestamp_mode(bars: list[Bar]) -> tuple[str, dict[str, int]]:
    if not bars:
        return "utc_naive", {"sample_rows": 0, "utc_naive_gap_hits": 0, "et_naive_gap_hits": 0}

    step = max(1, len(bars) // 250000)
    sample = bars[::step]
    utc_gap = 0
    et_gap = 0
    utc_rth = 0
    et_rth = 0
    for bar in sample:
        t_utc_mode = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsSourceMode.UTC_NAIVE).timetz().replace(tzinfo=None)
        t_et_mode = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsSourceMode.ET_NAIVE).timetz().replace(tzinfo=None)
        if _is_maintenance_gap(t_utc_mode):
            utc_gap += 1
        if _is_maintenance_gap(t_et_mode):
            et_gap += 1
        if time(9, 30) <= t_utc_mode <= time(15, 59):
            utc_rth += 1
        if time(9, 30) <= t_et_mode <= time(15, 59):
            et_rth += 1

    # Correct interpretation should keep maintenance-gap hits near zero.
    # Tie-break: choose mode with stronger RTH presence.
    utc_score = (utc_gap, -utc_rth)
    et_score = (et_gap, -et_rth)
    mode = (
        NaiveTsSourceMode.UTC_NAIVE.value
        if utc_score <= et_score
        else NaiveTsSourceMode.ET_NAIVE.value
    )
    diag = {
        "sample_rows": len(sample),
        "utc_naive_gap_hits": int(utc_gap),
        "et_naive_gap_hits": int(et_gap),
        "utc_naive_rth_hits": int(utc_rth),
        "et_naive_rth_hits": int(et_rth),
    }
    return mode, diag


def _day_state_for_bars(
    bars: Iterable[Bar],
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[dict[date, set[str]], dict[date, int]]:
    sessions_by_day: dict[date, set[str]] = defaultdict(set)
    rows_by_day: dict[date, int] = defaultdict(int)
    for bar in bars:
        ts = bar.ts
        if ts < start_utc or ts > end_utc:
            continue
        ts_et = _to_et_shared(ts, naive_ts_mode=NaiveTsMode.UTC)
        session = _session_label_et(ts_et.timetz().replace(tzinfo=None))
        if session is None:
            continue
        day = ts_et.date()
        sessions_by_day[day].add(session)
        rows_by_day[day] += 1
    return sessions_by_day, rows_by_day


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _sort_sessions(labels: Iterable[str]) -> list[str]:
    order = {k: i for i, k in enumerate(_SESSION_ORDER)}
    return sorted(labels, key=lambda x: order.get(x, 999))


def _audit_rows(
    bars: Iterable[Bar],
    *,
    start_utc_date: date,
    end_utc_date: date,
    session_mode: str,
) -> dict[str, object]:
    start_utc = datetime.combine(start_utc_date, time(0, 0))
    end_utc = datetime.combine(end_utc_date, time(23, 59))
    start_et = _to_et_shared(start_utc, naive_ts_mode=NaiveTsMode.UTC)
    end_et = _to_et_shared(end_utc, naive_ts_mode=NaiveTsMode.UTC)

    et_day_start = start_et.date()
    et_day_end = end_et.date()
    boundary_partial_days: set[date] = set()
    if start_et.timetz().replace(tzinfo=None) != time(0, 0):
        boundary_partial_days.add(et_day_start)
    if end_et.timetz().replace(tzinfo=None) != time(23, 59):
        boundary_partial_days.add(et_day_end)

    sessions_by_day, rows_by_day = _day_state_for_bars(bars, start_utc=start_utc, end_utc=end_utc)
    anomaly_days_all: dict[str, list[str]] = {}
    anomaly_days_effective: dict[str, list[str]] = {}
    missing_counts_all = {k: 0 for k in _SESSION_ORDER}
    missing_counts_effective = {k: 0 for k in _SESSION_ORDER}

    for day in _daterange(et_day_start, et_day_end):
        expected = _expected_sessions(day, session_mode=session_mode)
        if not expected:
            continue
        have = sessions_by_day.get(day, set())
        missing = _sort_sessions(expected - have)
        if not missing:
            continue
        key = day.isoformat()
        anomaly_days_all[key] = missing
        for label in missing:
            missing_counts_all[label] += 1
        if day in boundary_partial_days:
            continue
        anomaly_days_effective[key] = missing
        for label in missing:
            missing_counts_effective[label] += 1

    return {
        "row_count": sum(1 for _ in bars),
        "start_utc_date": start_utc_date.isoformat(),
        "end_utc_date": end_utc_date.isoformat(),
        "start_et": start_et.isoformat(),
        "end_et": end_et.isoformat(),
        "boundary_partial_days_et": sorted(d.isoformat() for d in boundary_partial_days),
        "anomaly_day_count_all": len(anomaly_days_all),
        "anomaly_day_count_effective": len(anomaly_days_effective),
        "anomaly_days_all": anomaly_days_all,
        "anomaly_days_effective": anomaly_days_effective,
        "missing_session_day_counts_all": missing_counts_all,
        "missing_session_day_counts_effective": missing_counts_effective,
        "rows_by_day_et": {d.isoformat(): int(v) for d, v in sorted(rows_by_day.items())},
    }


def _supports_session_audit(bar_size: str) -> bool:
    bar_def = parse_bar_size(str(bar_size))
    if bar_def is None:
        return "day" not in str(bar_size or "").strip().lower()
    return bar_def.duration < timedelta(days=1)


def _is_one_minute_bar_size(bar_size: str) -> bool:
    bar_def = parse_bar_size(str(bar_size))
    if bar_def is None:
        token = str(bar_size or "").strip().lower().replace(" ", "")
        return token in {"1m", "1min", "1mins", "1minute", "1minutes"}
    return bar_def.duration == timedelta(minutes=1)


def _intra_session_gap_days(
    bars: Iterable[Bar],
    *,
    start_utc_date: date,
    end_utc_date: date,
    session_mode: str,
    bar_size: str,
) -> dict[str, list[str]]:
    if not _is_one_minute_bar_size(bar_size):
        return {}
    start_utc = datetime.combine(start_utc_date, time(0, 0))
    end_utc = datetime.combine(end_utc_date, time(23, 59))
    by_day_session: dict[tuple[date, str], list[datetime]] = defaultdict(list)
    for bar in bars:
        ts = bar.ts
        if ts < start_utc or ts > end_utc:
            continue
        ts_et = _to_et_shared(ts, naive_ts_mode=NaiveTsMode.UTC)
        sess = _session_label_et(ts_et.timetz().replace(tzinfo=None))
        if sess is None:
            continue
        day = ts_et.date()
        if sess not in _expected_sessions(day, session_mode=session_mode):
            continue
        by_day_session[(day, sess)].append(ts)

    out: dict[str, list[str]] = defaultdict(list)
    for (day, sess), values in by_day_session.items():
        if len(values) < 2:
            continue
        seq = sorted(set(values))
        prev = seq[0]
        has_gap = False
        for cur in seq[1:]:
            if (cur - prev).total_seconds() > 60.0:
                has_gap = True
                break
            prev = cur
        if has_gap:
            out[day.isoformat()].append(sess)
    return {k: _sort_sessions(v) for k, v in out.items()}


def _coverage_fingerprint(audit: dict[str, object], gaps: dict[str, list[str]]) -> str | None:
    anomalies = audit.get("anomaly_days_effective", {})
    parts: list[str] = []
    if isinstance(anomalies, dict):
        for day in sorted(anomalies.keys()):
            missing = anomalies.get(day)
            if isinstance(missing, list) and missing:
                parts.append(f"s:{day}:{','.join(sorted(str(x) for x in missing))}")
    for day in sorted(gaps.keys()):
        labels = gaps.get(day) or []
        if labels:
            parts.append(f"g:{day}:{','.join(sorted(str(x) for x in labels))}")
    if not parts:
        return None
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def _dedupe_sort_bars(rows: Iterable[Bar]) -> list[Bar]:
    by_ts: dict[datetime, Bar] = {}
    for row in rows:
        by_ts[row.ts] = row
    return [by_ts[ts] for ts in sorted(by_ts.keys())]


def _canonicalize_rows(rows: list[Bar], *, mode: str) -> list[Bar]:
    out: list[Bar] = []
    for bar in rows:
        out.append(
            Bar(
                ts=_to_utc_naive_shared(bar.ts, naive_ts_mode=mode),
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
        )
    return _dedupe_sort_bars(out)


def _build_indices(rows: list[Bar]) -> tuple[dict[datetime, Bar], dict[date, set[datetime]]]:
    by_ts: dict[datetime, Bar] = {bar.ts: bar for bar in rows}
    day_to_ts: dict[date, set[datetime]] = defaultdict(set)
    for ts in by_ts:
        day_to_ts[_et_day_from_utc_naive(ts)].add(ts)
    return by_ts, day_to_ts


def _bars_for_day(by_ts: dict[datetime, Bar], day_to_ts: dict[date, set[datetime]], day: date) -> list[Bar]:
    stamps = sorted(day_to_ts.get(day, set()))
    return [by_ts[ts] for ts in stamps]


def _day_quality(day: date, day_rows: list[Bar], *, session_mode: str) -> tuple[tuple[int, int, int], dict[str, object]]:
    expected = _expected_sessions(day, session_mode=session_mode)
    sessions: set[str] = set()
    for bar in day_rows:
        t_et = _to_et_shared(bar.ts, naive_ts_mode=NaiveTsMode.UTC).timetz().replace(tzinfo=None)
        label = _session_label_et(t_et)
        if label is None:
            continue
        sessions.add(label)

    covered = expected & sessions
    missing = _sort_sessions(expected - sessions)
    weighted = sum(_SESSION_WEIGHTS.get(s, 0) for s in covered)
    score = (weighted, len(covered), len(day_rows))
    meta = {
        "expected_sessions": _sort_sessions(expected),
        "present_sessions": _sort_sessions(sessions),
        "missing_sessions": missing,
        "covered_count": len(covered),
        "row_count": len(day_rows),
        "weighted_score": weighted,
    }
    return score, meta


def _replace_day(
    by_ts: dict[datetime, Bar],
    day_to_ts: dict[date, set[datetime]],
    *,
    day: date,
    new_rows: list[Bar],
) -> None:
    for ts in day_to_ts.get(day, set()):
        by_ts.pop(ts, None)
    day_to_ts[day] = set()

    for row in new_rows:
        by_ts[row.ts] = row
    for row in new_rows:
        day_to_ts[_et_day_from_utc_naive(row.ts)].add(row.ts)


def _is_overnight_bar_utc_naive(ts: datetime) -> bool:
    t_et = _to_et_shared(ts, naive_ts_mode=NaiveTsMode.UTC).timetz().replace(tzinfo=None)
    return (t_et >= time(20, 0)) or (t_et < time(4, 0))


def _merge_smart_overnight(smart: list[Bar], overnight: list[Bar]) -> list[Bar]:
    by_ts: dict[datetime, Bar] = {}
    for bar in smart:
        by_ts[bar.ts] = bar
    for bar in overnight:
        if _is_overnight_bar_utc_naive(bar.ts) or bar.ts not in by_ts:
            by_ts[bar.ts] = bar
    return [by_ts[k] for k in sorted(by_ts.keys())]


def _fetch_day_from_ibkr(
    *,
    symbol: str,
    bar_size: str,
    day: date,
    use_rth: bool,
    timeout_sec: float,
    retries: int,
    client_id_offset: int,
) -> _FetchResult:
    os.environ["TRADEBOT_IBKR_HIST_TIMEOUT_SEC"] = str(max(1.0, float(timeout_sec)))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    attempts = 0
    smart_rows = 0
    overnight_rows = 0
    merged_rows = 0
    last_error: str | None = None
    start_utc, end_utc = _utc_bounds_for_et_day(day)
    try:
        for _ in range(max(1, int(retries))):
            attempts += 1
            provider = IBKRHistoricalData(client_id_offset=client_id_offset)
            try:
                contract_smart, _ = provider.resolve_contract(symbol, exchange="SMART")
                if bool(use_rth):
                    smart = provider._fetch_bars(contract_smart, start_utc, end_utc, bar_size, use_rth=True)
                    overnight = []
                    merged = list(smart)
                else:
                    contract_overnight, _ = provider.resolve_contract(symbol, exchange="OVERNIGHT")
                    smart = provider._fetch_bars(contract_smart, start_utc, end_utc, bar_size, use_rth=False)
                    overnight = provider._fetch_bars(contract_overnight, start_utc, end_utc, bar_size, use_rth=False)
                    merged = _merge_smart_overnight(smart, overnight)
                smart_rows = len(smart)
                overnight_rows = len(overnight)
                merged_rows = len(merged)
                if merged:
                    return _FetchResult(
                        day=day,
                        ok=True,
                        attempts=attempts,
                        smart_rows=smart_rows,
                        overnight_rows=overnight_rows,
                        merged_rows=merged_rows,
                        error=None,
                        bars=tuple(_dedupe_sort_bars(merged)),
                    )
                last_error = "empty_merge"
            except Exception as exc:  # pragma: no cover - network/IBKR-dependent
                last_error = str(exc)
            finally:
                try:
                    provider.disconnect()
                except Exception:
                    pass

        return _FetchResult(
            day=day,
            ok=False,
            attempts=attempts,
            smart_rows=smart_rows,
            overnight_rows=overnight_rows,
            merged_rows=merged_rows,
            error=last_error or "unknown",
            bars=tuple(),
        )
    finally:
        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


def _archive_overlay(
    *,
    by_ts: dict[datetime, Bar],
    day_to_ts: dict[date, set[datetime]],
    candidate_days: list[date],
    archive_paths: list[Path],
    session_mode: str,
) -> dict[str, object]:
    if not archive_paths:
        return {
            "archive_files": [],
            "days_considered": len(candidate_days),
            "days_replaced": 0,
            "replaced": [],
            "skipped": [],
        }

    replaced: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for archive_path in archive_paths:
        if not archive_path.exists():
            skipped.append({"archive": str(archive_path), "reason": "missing_file"})
            continue

        raw_rows = list(read_cache(archive_path))
        mode, diag = _infer_timestamp_mode(raw_rows)
        rows = _canonicalize_rows(raw_rows, mode=mode)
        archive_by_ts, archive_day_to_ts = _build_indices(rows)

        for day in candidate_days:
            new_rows = _bars_for_day(archive_by_ts, archive_day_to_ts, day)
            if not new_rows:
                continue
            cur_rows = _bars_for_day(by_ts, day_to_ts, day)
            cur_score, cur_meta = _day_quality(day, cur_rows, session_mode=session_mode)
            new_score, new_meta = _day_quality(day, new_rows, session_mode=session_mode)
            if new_score > cur_score:
                _replace_day(by_ts, day_to_ts, day=day, new_rows=new_rows)
                replaced.append(
                    {
                        "day": day.isoformat(),
                        "source": str(archive_path),
                        "source_mode": mode,
                        "source_mode_diag": diag,
                        "old_score": list(cur_score),
                        "new_score": list(new_score),
                        "old_missing": cur_meta["missing_sessions"],
                        "new_missing": new_meta["missing_sessions"],
                        "old_rows": cur_meta["row_count"],
                        "new_rows": new_meta["row_count"],
                    }
                )

    return {
        "archive_files": [str(p) for p in archive_paths],
        "days_considered": len(candidate_days),
        "days_replaced": len(replaced),
        "replaced": replaced,
        "skipped": skipped,
    }


def _ibkr_overlay(
    *,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    by_ts: dict[datetime, Bar],
    day_to_ts: dict[date, set[datetime]],
    candidate_days: list[date],
    threads: int,
    retries: int,
    timeout_sec: float,
    client_id_base: int,
    session_mode: str,
) -> dict[str, object]:
    if not candidate_days:
        return {
            "days_considered": 0,
            "threads": int(threads),
            "retries": int(retries),
            "timeout_sec": float(timeout_sec),
            "fetched_ok": 0,
            "fetched_fail": 0,
            "days_replaced": 0,
            "fetched": [],
            "replaced": [],
        }

    fetched: list[dict[str, object]] = []
    replaced: list[dict[str, object]] = []
    threads = max(1, int(threads))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {}
        for idx, day in enumerate(candidate_days):
            fut = pool.submit(
                _fetch_day_from_ibkr,
                symbol=symbol,
                bar_size=bar_size,
                day=day,
                use_rth=bool(use_rth),
                timeout_sec=timeout_sec,
                retries=retries,
                client_id_offset=client_id_base + idx,
            )
            futures[fut] = day

        for fut in as_completed(futures):
            res = fut.result()
            fetched.append(
                {
                    "day": res.day.isoformat(),
                    "ok": bool(res.ok),
                    "attempts": int(res.attempts),
                    "smart_rows": int(res.smart_rows),
                    "overnight_rows": int(res.overnight_rows),
                    "merged_rows": int(res.merged_rows),
                    "error": res.error,
                }
            )
            if not res.ok or not res.bars:
                continue

            new_rows = list(res.bars)
            cur_rows = _bars_for_day(by_ts, day_to_ts, res.day)
            cur_score, cur_meta = _day_quality(res.day, cur_rows, session_mode=session_mode)
            new_score, new_meta = _day_quality(res.day, new_rows, session_mode=session_mode)
            if new_score > cur_score:
                _replace_day(by_ts, day_to_ts, day=res.day, new_rows=new_rows)
                replaced.append(
                    {
                        "day": res.day.isoformat(),
                        "old_score": list(cur_score),
                        "new_score": list(new_score),
                        "old_missing": cur_meta["missing_sessions"],
                        "new_missing": new_meta["missing_sessions"],
                        "old_rows": cur_meta["row_count"],
                        "new_rows": new_meta["row_count"],
                    }
                )

    fetched_ok = sum(1 for row in fetched if row["ok"])
    fetched_fail = sum(1 for row in fetched if not row["ok"])
    fetched.sort(key=lambda x: x["day"])
    replaced.sort(key=lambda x: x["day"])
    return {
        "days_considered": len(candidate_days),
        "threads": int(threads),
        "retries": int(retries),
        "timeout_sec": float(timeout_sec),
        "fetched_ok": int(fetched_ok),
        "fetched_fail": int(fetched_fail),
        "days_replaced": len(replaced),
        "fetched": fetched,
        "replaced": replaced,
    }


def _ibkr_overlay_adaptive(
    *,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    by_ts: dict[datetime, Bar],
    day_to_ts: dict[date, set[datetime]],
    candidate_days: list[date],
    threads: int,
    retries: int,
    timeout_sec: float,
    client_id_base: int,
    session_mode: str,
    adaptive_threads: bool,
) -> dict[str, object]:
    if not candidate_days:
        return {
            "days_considered": 0,
            "threads": int(threads),
            "adaptive_threads_enabled": bool(adaptive_threads),
            "thread_plan": [int(max(1, int(threads)))],
            "pass_stats": [],
            "retries": int(retries),
            "timeout_sec": float(timeout_sec),
            "fetched_ok": 0,
            "fetched_fail": 0,
            "days_replaced": 0,
            "fetched": [],
            "replaced": [],
        }

    base_threads = max(1, int(threads))
    thread_plan = [base_threads] if not bool(adaptive_threads) else _adaptive_thread_plan(base_threads)
    pending = sorted(set(candidate_days))
    fetched_all: list[dict[str, object]] = []
    replaced_by_day: dict[str, dict[str, object]] = {}
    pass_stats: list[dict[str, object]] = []

    for pass_idx, pass_threads in enumerate(thread_plan):
        if not pending:
            break
        report = _ibkr_overlay(
            symbol=symbol,
            bar_size=bar_size,
            use_rth=bool(use_rth),
            by_ts=by_ts,
            day_to_ts=day_to_ts,
            candidate_days=pending,
            threads=pass_threads,
            retries=retries,
            timeout_sec=timeout_sec,
            client_id_base=int(client_id_base) + (pass_idx * 1000),
            session_mode=session_mode,
        )
        fetched_rows = list(report.get("fetched", []))
        replaced_rows = list(report.get("replaced", []))
        for row in fetched_rows:
            if not isinstance(row, dict):
                continue
            tagged = dict(row)
            tagged["pass"] = int(pass_idx + 1)
            tagged["threads"] = int(pass_threads)
            fetched_all.append(tagged)
        for row in replaced_rows:
            if not isinstance(row, dict):
                continue
            day_key = str(row.get("day") or "").strip()
            if day_key:
                replaced_by_day[day_key] = row

        rows_by_day: dict[str, dict[str, object]] = {}
        for row in fetched_rows:
            if not isinstance(row, dict):
                continue
            day_key = str(row.get("day") or "").strip()
            if day_key:
                rows_by_day[day_key] = row

        retry_next: list[date] = []
        ok_n = 0
        retryable_n = 0
        terminal_n = 0
        for day in pending:
            key = day.isoformat()
            row = rows_by_day.get(key)
            if row is None:
                terminal_n += 1
                continue
            if bool(row.get("ok")):
                ok_n += 1
                continue
            if pass_idx < (len(thread_plan) - 1) and _is_retryable_ibkr_error(row.get("error")):
                retry_next.append(day)
                retryable_n += 1
                continue
            terminal_n += 1
        pass_stats.append(
            {
                "pass": int(pass_idx + 1),
                "threads": int(pass_threads),
                "attempted_days": int(len(pending)),
                "ok_days": int(ok_n),
                "retryable_fail_days": int(retryable_n),
                "terminal_fail_days": int(terminal_n),
            }
        )
        pending = sorted(set(retry_next))

    final_by_day: dict[str, dict[str, object]] = {}
    for row in fetched_all:
        day_key = str(row.get("day") or "").strip()
        if day_key:
            final_by_day[day_key] = row
    fetched_ok = sum(1 for row in final_by_day.values() if bool(row.get("ok")))
    fetched_fail = len(candidate_days) - fetched_ok
    fetched_sorted = sorted(
        fetched_all,
        key=lambda row: (
            str(row.get("day") or ""),
            int(row.get("pass") or 0),
        ),
    )
    replaced_sorted = sorted(replaced_by_day.values(), key=lambda row: str(row.get("day") or ""))
    return {
        "days_considered": len(candidate_days),
        "threads": int(base_threads),
        "adaptive_threads_enabled": bool(adaptive_threads),
        "thread_plan": [int(x) for x in thread_plan],
        "max_primary_span_days": int(max_primary_span_days),
        "pass_stats": pass_stats,
        "retries": int(retries),
        "timeout_sec": float(timeout_sec),
        "fetched_ok": int(fetched_ok),
        "fetched_fail": int(fetched_fail),
        "days_replaced": len(replaced_sorted),
        "fetched": fetched_sorted,
        "replaced": replaced_sorted,
    }


def _remaining_refetch_plan(audit_after: dict[str, object], *, session_mode: str) -> list[dict[str, object]]:
    boundary = set(audit_after.get("boundary_partial_days_et", []))
    remaining = audit_after.get("anomaly_days_effective", {})
    out: list[dict[str, object]] = []
    for key, missing in sorted(remaining.items()):
        action = "ibkr_targeted_refetch"
        note = "Retry with IBKR targeted day refetch."
        d = date.fromisoformat(key)
        if key in boundary:
            action = "boundary_artifact"
            note = "UTC window boundary creates ET partial day."
        elif d.year <= 2024 and session_mode != "rth":
            action = "archive_legacy_window"
            note = "Likely older-data retention edge; archive/remove legacy window from active champion sets."
        out.append({"et_day": key, "missing": list(missing), "action": action, "note": note})
    return out


def _process_cache_file(
    *,
    cache_file: Path,
    archive_files: list[Path],
    heal: bool,
    threads: int,
    retries: int,
    timeout_sec: float,
    client_id_base: int,
    backup: bool,
    adaptive_threads: bool,
    ibkr_min_et_day: date | None,
    ibkr_max_et_day: date | None,
) -> dict[str, object]:
    if not cache_file.exists():
        raise SystemExit(f"Cache file not found: {cache_file}")

    meta = _require_cache_meta(cache_file)
    symbol = str(meta.symbol).upper()
    session_mode = "rth" if meta.tag == "rth" else "full24"
    raw_rows = list(read_cache(cache_file))
    mode_before, mode_diag_before = _infer_timestamp_mode(raw_rows)
    canonical_rows = _canonicalize_rows(raw_rows, mode=mode_before)

    audit_before = _audit_rows(
        canonical_rows,
        start_utc_date=meta.start_date,
        end_utc_date=meta.end_date,
        session_mode=session_mode,
    )

    by_ts, day_to_ts = _build_indices(canonical_rows)
    anomaly_days_before = sorted(date.fromisoformat(k) for k in audit_before["anomaly_days_effective"].keys())

    archive_report: dict[str, object] = {
        "archive_files": [],
        "days_considered": 0,
        "days_replaced": 0,
        "replaced": [],
        "skipped": [],
    }
    ibkr_report: dict[str, object] = {
        "days_considered": 0,
        "threads": int(threads),
        "adaptive_threads_enabled": bool(adaptive_threads),
        "thread_plan": [int(max(1, int(threads)))],
        "pass_stats": [],
        "retries": int(retries),
        "timeout_sec": float(timeout_sec),
        "legacy_skipped_days": [],
        "fetched_ok": 0,
        "fetched_fail": 0,
        "days_replaced": 0,
        "fetched": [],
        "replaced": [],
    }

    if heal and anomaly_days_before:
        archive_report = _archive_overlay(
            by_ts=by_ts,
            day_to_ts=day_to_ts,
            candidate_days=anomaly_days_before,
            archive_paths=archive_files,
            session_mode=session_mode,
        )
        interim_rows = _dedupe_sort_bars(by_ts.values())
        interim_audit = _audit_rows(
            interim_rows,
            start_utc_date=meta.start_date,
            end_utc_date=meta.end_date,
            session_mode=session_mode,
        )
        remaining_days = sorted(date.fromisoformat(k) for k in interim_audit["anomaly_days_effective"].keys())
        if ibkr_min_et_day is not None:
            remaining_days = [d for d in remaining_days if d >= ibkr_min_et_day]
        if ibkr_max_et_day is not None:
            remaining_days = [d for d in remaining_days if d <= ibkr_max_et_day]
        legacy_skipped_days: list[date] = []
        if session_mode != "rth" and ibkr_min_et_day is None and ibkr_max_et_day is None:
            legacy_skipped_days = [d for d in remaining_days if d.year <= 2024]
            remaining_days = [d for d in remaining_days if d.year >= 2025]
        ibkr_report["legacy_skipped_days"] = [d.isoformat() for d in legacy_skipped_days]
        if remaining_days:
            ibkr_report = _ibkr_overlay_adaptive(
                symbol=symbol,
                bar_size="1 min",
                use_rth=bool(session_mode == "rth"),
                by_ts=by_ts,
                day_to_ts=day_to_ts,
                candidate_days=remaining_days,
                threads=threads,
                retries=retries,
                timeout_sec=timeout_sec,
                client_id_base=client_id_base,
                session_mode=session_mode,
                adaptive_threads=bool(adaptive_threads),
            )
            if legacy_skipped_days:
                ibkr_report["legacy_skipped_days"] = [d.isoformat() for d in legacy_skipped_days]

    final_rows = _dedupe_sort_bars(by_ts.values())
    mode_after, mode_diag_after = _infer_timestamp_mode(final_rows)
    audit_after = _audit_rows(
        final_rows,
        start_utc_date=meta.start_date,
        end_utc_date=meta.end_date,
        session_mode=session_mode,
    )

    raw_needs_canonicalization = mode_before != "utc_naive"
    changed = len(final_rows) != len(canonical_rows) or any(a.ts != b.ts for a, b in zip(canonical_rows, final_rows))
    if not changed and len(final_rows) == len(canonical_rows):
        # If same timestamps, compare OHLCV in case values changed for same ts.
        changed = any(a != b for a, b in zip(canonical_rows, final_rows))
    # Even without overlay replacements, persist canonical UTC-naive timestamps when source cache
    # is still ET-naive (or otherwise non-canonical).
    changed = bool(changed or raw_needs_canonicalization)

    backup_path = None
    if heal and changed:
        if backup:
            stamp = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
            backup_path = cache_file.with_name(cache_file.name + f".bak_before_audit_heal_{stamp}")
            if not backup_path.exists():
                cache_file.replace(backup_path)
                write_cache(cache_file, final_rows)
            else:
                write_cache(cache_file, final_rows)
        else:
            write_cache(cache_file, final_rows)

    plan = _remaining_refetch_plan(audit_after, session_mode=session_mode)
    return {
        "cache_file": str(cache_file),
        "symbol": symbol,
        "session_mode": session_mode,
        "start_utc_date": meta.start_date.isoformat(),
        "end_utc_date": meta.end_date.isoformat(),
        "rows_before": len(raw_rows),
        "rows_after": len(final_rows),
        "detected_mode_before": mode_before,
        "detected_mode_after": mode_after,
        "mode_diag_before": mode_diag_before,
        "mode_diag_after": mode_diag_after,
        "audit_before": audit_before,
        "archive_overlay": archive_report,
        "ibkr_overlay": ibkr_report,
        "audit_after": audit_after,
        "remaining_refetch_plan": plan,
        "updated_cache": bool(heal and changed),
        "canonicalized_timezone": bool(heal and raw_needs_canonicalization),
        "backup_path": str(backup_path) if backup_path else None,
    }


_EPOCH = datetime(1970, 1, 1)


def _floor_bucket_start(ts: datetime, *, bucket: timedelta) -> datetime:
    if bucket <= timedelta(0):
        return ts
    delta = ts - _EPOCH
    micros = (delta.days * 86400 * 1_000_000) + (delta.seconds * 1_000_000) + delta.microseconds
    bucket_micros = int(bucket.total_seconds() * 1_000_000)
    bucket_idx = micros // bucket_micros
    start_micros = bucket_idx * bucket_micros
    return _EPOCH + timedelta(microseconds=int(start_micros))


def _resample_intraday_ohlcv(
    bars: list[Bar],
    *,
    src_bar_size: str,
    dst_bar_size: str,
    allow_day_from_partial: bool = False,
) -> tuple[list[Bar], _ResampleStats]:
    src = parse_bar_size(src_bar_size)
    dst = parse_bar_size(dst_bar_size)
    if src is None:
        raise SystemExit(f"Invalid src bar size: {src_bar_size!r}")
    if dst is None:
        raise SystemExit(f"Invalid dst bar size: {dst_bar_size!r}")
    if src.duration <= timedelta(0) or dst.duration <= timedelta(0):
        raise SystemExit(f"Invalid bar durations: src={src.duration} dst={dst.duration}")
    if dst.duration < src.duration:
        raise SystemExit("dst_bar_size must be >= src_bar_size")
    if (dst.duration.total_seconds() % src.duration.total_seconds()) != 0:
        raise SystemExit(f"Non-integer resample ratio: {src_bar_size!r} -> {dst_bar_size!r}")

    if not bars:
        return [], _ResampleStats(kept=0, dropped_incomplete=0)

    bars = sorted(bars, key=lambda b: b.ts)

    # RTH streams do not contain 24h coverage, so 1-day bars cannot satisfy strict
    # ratio-based chunk completeness (390 != 1440). Allow deterministic day grouping
    # when explicitly requested by caller.
    if allow_day_from_partial and src.duration < timedelta(days=1) and dst.duration >= timedelta(days=1):
        by_day: dict[date, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_day[bar.ts.date()].append(bar)
        out: list[Bar] = []
        for day in sorted(by_day.keys()):
            chunk = by_day[day]
            if not chunk:
                continue
            first = chunk[0]
            last = chunk[-1]
            out.append(
                Bar(
                    ts=datetime.combine(day, time(0, 0)),
                    open=first.open,
                    high=max(b.high for b in chunk),
                    low=min(b.low for b in chunk),
                    close=last.close,
                    volume=sum(float(b.volume or 0.0) for b in chunk),
                )
            )
        return out, _ResampleStats(kept=len(out), dropped_incomplete=0)

    factor = max(1, int(dst.duration.total_seconds() // src.duration.total_seconds()))
    out: list[Bar] = []
    cur_bucket = None
    cur: list[Bar] = []
    dropped = 0

    def _flush(bucket_start: datetime | None, chunk: list[Bar]) -> None:
        nonlocal dropped
        if bucket_start is None or not chunk:
            return
        if len(chunk) != factor:
            dropped += 1
            return
        first = chunk[0]
        last = chunk[-1]
        out.append(
            Bar(
                ts=bucket_start,
                open=first.open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=last.close,
                volume=sum(float(b.volume or 0.0) for b in chunk),
            )
        )

    for bar in bars:
        bucket_start = _floor_bucket_start(bar.ts, bucket=dst.duration)
        if cur_bucket is None:
            cur_bucket = bucket_start
        if bucket_start != cur_bucket:
            _flush(cur_bucket, cur)
            cur_bucket = bucket_start
            cur = []
        cur.append(bar)
    _flush(cur_bucket, cur)

    return out, _ResampleStats(kept=len(out), dropped_incomplete=int(dropped))


def _cache_resample_source_candidates(dst_bar_size: str) -> tuple[str, ...]:
    dst = parse_bar_size(str(dst_bar_size))
    if dst is None:
        return tuple()
    base = ("1 min", "2 mins", "5 mins", "10 mins", "15 mins", "30 mins", "1 hour", "4 hours", "1 day")
    out: list[tuple[float, str]] = []
    for src_label in base:
        src = parse_bar_size(src_label)
        if src is None:
            continue
        if src.duration >= dst.duration:
            continue
        if (dst.duration.total_seconds() % src.duration.total_seconds()) != 0:
            continue
        out.append((float(src.duration.total_seconds()), src.label))
    out.sort(key=lambda row: row[0])
    return tuple(label for _seconds, label in out)


def resample_cached_window(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    dst_bar_size: str,
    use_rth: bool,
    src_bar_size: str | None = None,
) -> CacheResampleOutcome:
    dst_path = cache_path(cache_dir, symbol, start, end, dst_bar_size, use_rth)
    src_candidates = (
        (str(src_bar_size).strip(),)
        if str(src_bar_size or "").strip()
        else _cache_resample_source_candidates(dst_bar_size)
    )
    if not src_candidates:
        return CacheResampleOutcome(
            ok=False,
            dst_path=dst_path,
            error=f"no_resample_source_candidates for {dst_bar_size!r}",
        )

    last_err: str | None = None
    for src_label in src_candidates:
        try:
            src_series = data.load_cached_bar_series(
                symbol=symbol,
                exchange=exchange,
                start=start,
                end=end,
                bar_size=str(src_label),
                use_rth=use_rth,
                cache_dir=cache_dir,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            last_err = f"source_load_error({src_label}): {exc}"
            continue

        source_path_raw = str(getattr(src_series.meta, "source_path", "") or "").strip()
        source_path = Path(source_path_raw) if source_path_raw else None
        src_bars = [bar for bar in src_series.as_list() if start <= bar.ts <= end]
        if not src_bars:
            last_err = f"source_empty_after_slice({src_label})"
            continue

        try:
            dst_bars, stats = _resample_intraday_ohlcv(
                src_bars,
                src_bar_size=str(src_label),
                dst_bar_size=str(dst_bar_size),
                allow_day_from_partial=bool(use_rth),
            )
        except SystemExit as exc:
            last_err = f"resample_error({src_label}): {exc}"
            continue

        if not dst_bars:
            last_err = f"resample_empty({src_label})"
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        write_cache(dst_path, dst_bars)
        return CacheResampleOutcome(
            ok=True,
            dst_path=dst_path,
            src_bar_size=str(src_label),
            src_path=source_path,
            src_rows=len(src_bars),
            dst_rows=len(dst_bars),
            dropped_incomplete=int(stats.dropped_incomplete),
            error=None,
        )

    return CacheResampleOutcome(ok=False, dst_path=dst_path, error=last_err)


def ensure_cached_window_with_policy(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
    cache_policy: str = "strict",
) -> tuple[bool, Path, Path | None, list[tuple[date, date]], str | None]:
    policy = str(cache_policy or "strict").strip().lower() or "strict"
    if policy not in {"strict", "auto"}:
        raise ValueError(f"Unsupported cache_policy: {cache_policy!r}")

    ok, expected, resolved, missing_ranges, err = ensure_offline_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if ok or policy == "strict":
        return ok, expected, resolved, missing_ranges, err

    attempt_notes: list[str] = []
    rs_out = resample_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        dst_bar_size=bar_size,
        use_rth=use_rth,
        src_bar_size=None,
    )
    if rs_out.ok:
        src_note = (
            f"{rs_out.src_bar_size} ({rs_out.src_path})"
            if rs_out.src_path is not None
            else str(rs_out.src_bar_size)
        )
        attempt_notes.append(f"auto_resample_ok:{src_note}")
        ok2, expected2, resolved2, missing2, err2 = ensure_offline_cached_window(
            data=data,
            cache_dir=cache_dir,
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
        if ok2:
            return ok2, expected2, resolved2, missing2, err2
        err = err2
        expected = expected2
        resolved = resolved2
        missing_ranges = missing2
    elif rs_out.error:
        attempt_notes.append(str(rs_out.error))

    try:
        data.load_or_fetch_bar_series(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
        attempt_notes.append("auto_fetch_attempted")
    except Exception as exc:
        attempt_notes.append(f"auto_fetch_error:{exc}")

    ok3, expected3, resolved3, missing3, err3 = ensure_offline_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    if ok3:
        return ok3, expected3, resolved3, missing3, err3

    details = [str(err3 or err or "").strip(), *[str(x).strip() for x in attempt_notes if str(x).strip()]]
    err_out = "; ".join([x for x in details if x])
    return ok3, expected3, resolved3, missing3, (err_out or None)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_session_mode(raw: str) -> bool:
    token = str(raw or "").strip().lower()
    if token in {"rth", "true", "1", "yes"}:
        return True
    if token in {"full24", "full", "false", "0", "no"}:
        return False
    raise SystemExit(f"Invalid session mode: {raw!r} (expected rth or full24)")


def _parse_fetch_request(raw: str) -> _CacheFetchRequest:
    txt = str(raw or "").strip()
    if not txt:
        raise SystemExit("Empty --request")
    sep = "|" if "|" in txt else ","
    parts = [p.strip() for p in txt.split(sep)]
    if len(parts) < 5:
        raise SystemExit(
            "Invalid --request format. Expected SYMBOL|YYYY-MM-DD|YYYY-MM-DD|BAR_SIZE|rth|[source]"
        )
    symbol = str(parts[0]).upper()
    start = _parse_date(parts[1])
    end = _parse_date(parts[2])
    if end < start:
        raise SystemExit(f"Invalid request date range: {start}..{end}")
    bar_size = str(parts[3]).strip()
    if not bar_size:
        raise SystemExit("Missing request bar size")
    if parse_bar_size(bar_size) is None:
        raise SystemExit(f"Unsupported request bar size: {bar_size!r}")
    use_rth = _parse_session_mode(parts[4])
    source = str(parts[5]).strip() if len(parts) > 5 and str(parts[5]).strip() else "cli"
    return _CacheFetchRequest(
        symbol=symbol,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
        source=source,
    )


def _extract_current_json_from_readme(readme_path: Path) -> Path | None:
    if not readme_path.exists():
        return None
    text = readme_path.read_text(encoding="utf-8")
    head = re.search(r"^###\s+CURRENT(?:\s+\(v[^)]+\))?", text, flags=re.MULTILINE | re.IGNORECASE)
    if not head:
        return None
    tail = text[head.end() :]
    next_head = re.search(r"^###\s+", tail, flags=re.MULTILINE)
    section = tail[: next_head.start()] if next_head else tail
    match = re.search(r"`(?P<path>(?:tradebot|backtests)/[^`]+\.json)`", section)
    if match is None:
        match = re.search(r"(?P<path>(?:tradebot|backtests)/[^\s)]+\.json)", section)
    if match is None:
        return None
    path = (_repo_root() / str(match.group("path")).strip()).resolve()
    return path if path.exists() else None


def _discover_current_champion_jsons() -> list[Path]:
    repo_root = _repo_root()
    out: list[Path] = []
    slv_lf = _extract_current_json_from_readme(repo_root / "backtests" / "slv" / "readme-lf.md")
    slv_hf = _extract_current_json_from_readme(repo_root / "backtests" / "slv" / "readme-hf.md")
    tqqq = (repo_root / "tradebot" / "backtest" / "spot_champions.json").resolve()
    if slv_lf is not None:
        out.append(slv_lf)
    if slv_hf is not None:
        out.append(slv_hf)
    if tqqq.exists():
        out.append(tqqq)
    if not out:
        raise SystemExit("Unable to resolve current champion JSON paths from README files.")
    return out


def _collect_windows_from_payload(payload: dict[str, object]) -> list[tuple[date, date]]:
    out: set[tuple[date, date]] = set()
    top_windows = payload.get("windows")
    if isinstance(top_windows, list):
        for row in top_windows:
            if not isinstance(row, dict):
                continue
            s = row.get("start")
            e = row.get("end")
            if isinstance(s, str) and isinstance(e, str):
                try:
                    out.add((_parse_date(s), _parse_date(e)))
                except SystemExit:
                    continue

    groups = payload.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            ev = group.get("_eval")
            if not isinstance(ev, dict):
                continue
            windows = ev.get("windows")
            if not isinstance(windows, list):
                continue
            for row in windows:
                if not isinstance(row, dict):
                    continue
                s = row.get("start")
                e = row.get("end")
                if isinstance(s, str) and isinstance(e, str):
                    try:
                        out.add((_parse_date(s), _parse_date(e)))
                    except SystemExit:
                        continue
    return sorted(out, key=lambda x: (x[0], x[1]))


def _extract_bar_requirements(
    *,
    strategy: dict[str, object],
    default_symbol: str,
    default_exchange: str | None,
) -> tuple[SpotBarRequirement, ...]:
    if str(strategy.get("instrument") or "spot").strip().lower() != "spot":
        return tuple()
    reqs = spot_bar_requirements_from_strategy(
        strategy=strategy,
        default_symbol=str(default_symbol),
        default_exchange=default_exchange,
        default_signal_bar_size=str(strategy.get("signal_bar_size") or "1 min"),
        default_signal_use_rth=bool(strategy.get("signal_use_rth")),
        include_signal=True,
    )
    out: list[SpotBarRequirement] = []
    for req in reqs:
        if parse_bar_size(str(req.bar_size)) is None:
            continue
        out.append(req)
    return tuple(out)


def _requests_from_champion_json(path: Path) -> list[_CacheFetchRequest]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"Champion file must be a JSON object: {path}")
    windows = _collect_windows_from_payload(raw)
    if not windows:
        raise SystemExit(f"No windows found in champion payload: {path}")
    groups = raw.get("groups")
    if not isinstance(groups, list):
        raise SystemExit(f"No groups[] found in champion payload: {path}")

    requests: list[_CacheFetchRequest] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            strategy = entry.get("strategy")
            if not isinstance(strategy, dict):
                continue
            symbol = str(entry.get("symbol") or strategy.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            exchange_raw = strategy.get("exchange")
            exchange = str(exchange_raw).strip() if exchange_raw not in (None, "") else None
            reqs = _extract_bar_requirements(
                strategy=strategy,
                default_symbol=symbol,
                default_exchange=exchange,
            )
            for start, end in windows:
                for req in reqs:
                    req_start = start - timedelta(days=max(0, int(req.warmup_days)))
                    requests.append(
                        _CacheFetchRequest(
                            symbol=str(req.symbol).strip().upper() or symbol,
                            start=req_start,
                            end=end,
                            bar_size=str(req.bar_size),
                            use_rth=bool(req.use_rth),
                            source=str(path),
                        )
                    )
    return requests


def _dedupe_fetch_requests(requests: list[_CacheFetchRequest]) -> list[_CacheFetchRequest]:
    by_key: dict[tuple[str, date, date, str, bool], _CacheFetchRequest] = {}
    for req in requests:
        key = (req.symbol, req.start, req.end, req.bar_size, req.use_rth)
        by_key[key] = req
    return [by_key[k] for k in sorted(by_key.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))]


def _build_fetch_batches(
    requests: list[_CacheFetchRequest],
    *,
    max_primary_span_days: int = 0,
) -> list[_CacheFetchBatch]:
    grouped: dict[tuple[str, str, bool], list[_CacheFetchRequest]] = defaultdict(list)
    for req in requests:
        grouped[(req.symbol, req.bar_size, req.use_rth)].append(req)

    span_limit = max(0, int(max_primary_span_days))
    out: list[_CacheFetchBatch] = []
    for (symbol, bar_size, use_rth), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: (r.start, r.end))
        shards: list[tuple[_CacheFetchRequest, ...]] = []
        if span_limit <= 0:
            shards = [tuple(rows_sorted)]
        else:
            cur: list[_CacheFetchRequest] = []
            cur_start: date | None = None
            cur_end: date | None = None
            for req in rows_sorted:
                if not cur:
                    cur = [req]
                    cur_start = req.start
                    cur_end = req.end
                    continue
                assert cur_start is not None and cur_end is not None
                next_end = max(cur_end, req.end)
                next_span_days = int((next_end - cur_start).days) + 1
                if next_span_days > span_limit:
                    shards.append(tuple(cur))
                    cur = [req]
                    cur_start = req.start
                    cur_end = req.end
                else:
                    cur.append(req)
                    cur_end = next_end
            if cur:
                shards.append(tuple(cur))

        shard_total = len(shards)
        for shard_idx, shard_rows in enumerate(shards, start=1):
            start = min(r.start for r in shard_rows)
            end = max(r.end for r in shard_rows)
            srcs = sorted({r.source for r in shard_rows if r.source})
            source = f"merged:{len(shard_rows)}"
            if span_limit > 0 and shard_total > 1:
                source = f"{source}:shard{shard_idx}/{shard_total}"
            if srcs:
                source = f"{source}:{srcs[0]}"
                if len(srcs) > 1:
                    source = f"{source}:+{len(srcs)-1}"
            primary = _CacheFetchRequest(
                symbol=symbol,
                start=start,
                end=end,
                bar_size=bar_size,
                use_rth=use_rth,
                source=source,
            )
            out.append(_CacheFetchBatch(primary=primary, targets=tuple(shard_rows)))
    out.sort(
        key=lambda b: (
            b.primary.symbol,
            b.primary.start,
            b.primary.end,
            b.primary.bar_size,
            b.primary.use_rth,
        )
    )
    return out


def _adaptive_thread_plan(base_threads: int) -> list[int]:
    base = max(1, int(base_threads))
    plan = [base]
    plan.append(max(1, min(8, base // 2)))
    plan.append(3)
    plan.append(1)
    out: list[int] = []
    for t in plan:
        if t not in out:
            out.append(t)
    return out


def _is_retryable_ibkr_error(err: str | None) -> bool:
    txt = str(err or "").strip().lower()
    if not txt:
        return False
    return (
        "timeout" in txt
        or "historical market data service error message:api historical data query cancelled" in txt
        or "pacing" in txt
        or "socket" in txt
    )


def _fetch_single_request(
    *,
    req: _CacheFetchRequest,
    cache_dir: Path,
    force_refresh: bool,
    timeout_sec: float,
    client_id_offset: int,
    mend_threads: int,
    mend_retries: int,
    mend_adaptive_threads: bool,
) -> _CacheFetchOutcome:
    os.environ["TRADEBOT_IBKR_HIST_TIMEOUT_SEC"] = str(max(1.0, float(timeout_sec)))
    start_dt = datetime.combine(req.start, time(0, 0))
    end_dt = datetime.combine(req.end, time(23, 59))
    cache_file = cache_path(cache_dir, req.symbol, start_dt, end_dt, req.bar_size, req.use_rth)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    session_mode = "rth" if bool(req.use_rth) else "full24"
    supports_audit = _supports_session_audit(req.bar_size)
    session_before = 0
    session_after = 0
    gap_before = 0
    gap_after = 0
    candidate_days_count = 0
    healed_days = 0
    tz_canon = False
    anomaly_fingerprint = None

    def _rows_in_window(rows: list[Bar]) -> list[Bar]:
        return [bar for bar in rows if start_dt <= bar.ts <= end_dt]

    def _coverage_days(
        rows: list[Bar],
    ) -> tuple[dict[str, object], dict[str, list[str]], list[date], int, int]:
        if not supports_audit:
            return {}, {}, [], 0, 0
        effective_session_mode = session_mode
        if session_mode == "full24":
            has_overnight = False
            for bar in rows:
                ts = bar.ts
                if ts < start_dt or ts > end_dt:
                    continue
                ts_et = _to_et_shared(ts, naive_ts_mode=NaiveTsMode.UTC).timetz().replace(tzinfo=None)
                sess = _session_label_et(ts_et)
                if sess in {"OVERNIGHT_EARLY", "OVERNIGHT_LATE"}:
                    has_overnight = True
                    break
            if not has_overnight:
                # Some historical windows have no IBKR OVERNIGHT stream for this symbol.
                # Fall back to SMART extended-session completeness expectations.
                effective_session_mode = "smart_ext"

        audit = _audit_rows(
            rows,
            start_utc_date=req.start,
            end_utc_date=req.end,
            session_mode=effective_session_mode,
        )
        session_days = {date.fromisoformat(k) for k in audit.get("anomaly_days_effective", {}).keys()}
        session_count = int(audit.get("anomaly_day_count_effective", 0) or 0)
        gap_map = _intra_session_gap_days(
            rows,
            start_utc_date=req.start,
            end_utc_date=req.end,
            session_mode=effective_session_mode,
            bar_size=req.bar_size,
        )
        gap_days = {date.fromisoformat(k) for k in gap_map.keys()}
        combined = sorted(session_days | gap_days)
        return audit, gap_map, combined, session_count, len(gap_map)

    def _severity(*, ok: bool, healed: bool, tz_canonicalized: bool, remaining: int) -> str:
        if (not ok) or int(remaining) > 0:
            return "fail"
        if healed or tz_canonicalized:
            return "warn"
        return "ok"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    provider = IBKRHistoricalData(client_id_offset=client_id_offset)
    try:
        if force_refresh and cache_file.exists():
            cache_file.unlink()

        # Default behavior: if exact cache exists, detect internal coverage anomalies and
        # heal in-place before accepting cache-hit.
        if not force_refresh and cache_file.exists():
            raw_rows = list(read_cache(cache_file))
            if raw_rows:
                mode_before, _ = _infer_timestamp_mode(raw_rows)
                canonical_rows = _canonicalize_rows(raw_rows, mode=mode_before)
                _, _, candidate_days, session_before, gap_before = _coverage_days(canonical_rows)
                candidate_days_count = int(len(candidate_days))
                tz_canon = mode_before != NaiveTsSourceMode.UTC_NAIVE.value

                final_rows = canonical_rows
                if candidate_days:
                    by_ts, day_to_ts = _build_indices(canonical_rows)
                    _ibkr_overlay_adaptive(
                        symbol=req.symbol,
                        bar_size=req.bar_size,
                        use_rth=bool(req.use_rth),
                        by_ts=by_ts,
                        day_to_ts=day_to_ts,
                        candidate_days=candidate_days,
                        threads=max(1, int(mend_threads)),
                        retries=max(1, int(mend_retries)),
                        timeout_sec=timeout_sec,
                        client_id_base=(int(client_id_offset) * 1000),
                        session_mode=session_mode,
                        adaptive_threads=bool(mend_adaptive_threads),
                    )
                    final_rows = _dedupe_sort_bars(by_ts.values())

                changed = bool(tz_canon)
                if len(final_rows) != len(canonical_rows):
                    changed = True
                elif not changed:
                    changed = any(a != b for a, b in zip(canonical_rows, final_rows))

                if changed:
                    write_cache(cache_file, final_rows)

                rows_in_window = _rows_in_window(final_rows)
                if supports_audit:
                    audit_after, gaps_after, remaining_days, session_after, gap_after = _coverage_days(final_rows)
                    healed_days = max(0, candidate_days_count - len(remaining_days))
                    anomaly_fingerprint = _coverage_fingerprint(audit_after, gaps_after)
                    if not remaining_days and final_rows:
                        ok = bool(rows_in_window)
                        return _CacheFetchOutcome(
                            request=req,
                            ok=ok,
                            from_cache=not changed and not bool(candidate_days),
                            rows=len(rows_in_window),
                            cache_path=str(cache_file),
                            error=None if ok else "empty_rows_in_window",
                            status=_severity(
                                ok=ok,
                                healed=bool(healed_days > 0),
                                tz_canonicalized=bool(tz_canon),
                                remaining=0,
                            ),
                            healed=bool(healed_days > 0),
                            healed_days=int(healed_days),
                            remaining_days=0,
                            session_missing_days_before=int(session_before),
                            session_missing_days_after=int(session_after),
                            gap_days_before=int(gap_before),
                            gap_days_after=int(gap_after),
                            timezone_canonicalized=bool(tz_canon),
                            anomaly_fingerprint=None,
                        )
                    # Coverage still incomplete: force one full-window refresh attempt.
                    if cache_file.exists():
                        cache_file.unlink()
                else:
                    ok = bool(rows_in_window)
                    return _CacheFetchOutcome(
                        request=req,
                        ok=ok,
                        from_cache=not changed,
                        rows=len(rows_in_window),
                        cache_path=str(cache_file),
                        error=None if ok else "empty_rows_in_window",
                        status=_severity(
                            ok=ok,
                            healed=bool(healed_days > 0),
                            tz_canonicalized=bool(tz_canon),
                            remaining=0,
                        ),
                        healed=bool(healed_days > 0),
                        healed_days=int(healed_days),
                        remaining_days=0,
                        session_missing_days_before=0,
                        session_missing_days_after=0,
                        gap_days_before=0,
                        gap_days_after=0,
                        timezone_canonicalized=bool(tz_canon),
                        anomaly_fingerprint=None,
                    )

        provider.load_or_fetch_bars(
            symbol=req.symbol,
            exchange=None,
            start=start_dt,
            end=end_dt,
            bar_size=req.bar_size,
            use_rth=req.use_rth,
            cache_dir=cache_dir,
        )
        raw_after = list(read_cache(cache_file)) if cache_file.exists() else []
        if not raw_after:
            return _CacheFetchOutcome(
                request=req,
                ok=False,
                from_cache=False,
                rows=0,
                cache_path=str(cache_file),
                error="empty_cache_after_fetch",
                status="fail",
                healed=bool(healed_days > 0),
                healed_days=int(healed_days),
                remaining_days=max(0, int(candidate_days_count) - int(healed_days)),
                session_missing_days_before=int(session_before),
                session_missing_days_after=int(session_after),
                gap_days_before=int(gap_before),
                gap_days_after=int(gap_after),
                timezone_canonicalized=bool(tz_canon),
                anomaly_fingerprint=anomaly_fingerprint,
            )

        mode_after, _ = _infer_timestamp_mode(raw_after)
        canonical_after = _canonicalize_rows(raw_after, mode=mode_after)
        if mode_after != NaiveTsSourceMode.UTC_NAIVE.value:
            write_cache(cache_file, canonical_after)
        tz_canon = bool(tz_canon or (mode_after != NaiveTsSourceMode.UTC_NAIVE.value))

        rows_in_window = _rows_in_window(canonical_after)
        if not supports_audit:
            ok = bool(rows_in_window)
            return _CacheFetchOutcome(
                request=req,
                ok=ok,
                from_cache=False,
                rows=len(rows_in_window),
                cache_path=str(cache_file),
                error=None if rows_in_window else "empty_cache_after_fetch",
                status=_severity(
                    ok=ok,
                    healed=bool(healed_days > 0),
                    tz_canonicalized=bool(tz_canon),
                    remaining=0,
                ),
                healed=bool(healed_days > 0),
                healed_days=int(healed_days),
                remaining_days=0,
                session_missing_days_before=0,
                session_missing_days_after=0,
                gap_days_before=0,
                gap_days_after=0,
                timezone_canonicalized=bool(tz_canon),
                anomaly_fingerprint=None,
            )

        audit_after, gaps_after, remaining_days, session_after, gap_after = _coverage_days(canonical_after)
        healed_days = max(int(healed_days), max(0, int(candidate_days_count) - len(remaining_days)))
        anomaly_fingerprint = _coverage_fingerprint(audit_after, gaps_after)
        ok_complete = bool(rows_in_window) and not remaining_days
        error = None
        if not ok_complete:
            error = (
                "incomplete_after_fetch:"
                f" session_days={int(audit_after.get('anomaly_day_count_effective', 0) or 0)}"
                f" gap_days={len(gaps_after)}"
            )
        return _CacheFetchOutcome(
            request=req,
            ok=ok_complete,
            from_cache=False,
            rows=len(rows_in_window),
            cache_path=str(cache_file),
            error=error,
            status=_severity(
                ok=ok_complete,
                healed=bool(healed_days > 0),
                tz_canonicalized=bool(tz_canon),
                remaining=len(remaining_days),
            ),
            healed=bool(healed_days > 0),
            healed_days=int(healed_days),
            remaining_days=int(len(remaining_days)),
            session_missing_days_before=int(session_before),
            session_missing_days_after=int(session_after),
            gap_days_before=int(gap_before),
            gap_days_after=int(gap_after),
            timezone_canonicalized=bool(tz_canon),
            anomaly_fingerprint=anomaly_fingerprint,
        )
    except Exception as exc:  # pragma: no cover - network/IBKR dependent
        return _CacheFetchOutcome(
            request=req,
            ok=False,
            from_cache=False,
            rows=0,
            cache_path=str(cache_file),
            error=str(exc),
            status="fail",
            healed=bool(healed_days > 0),
            healed_days=int(healed_days),
            remaining_days=max(0, int(candidate_days_count) - int(healed_days)),
            session_missing_days_before=int(session_before),
            session_missing_days_after=int(session_after),
            gap_days_before=int(gap_before),
            gap_days_after=int(gap_after),
            timezone_canonicalized=bool(tz_canon),
            anomaly_fingerprint=anomaly_fingerprint,
        )
    finally:
        try:
            provider.disconnect()
        except Exception:
            pass
        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


def _run_primary_batches(
    *,
    batches: list[_CacheFetchBatch],
    cache_dir: Path,
    force_refresh: bool,
    timeout_sec: float,
    threads: int,
    client_id_base: int,
    mend_threads: int,
    mend_retries: int,
    mend_adaptive_threads: bool,
) -> list[_CacheFetchOutcome]:
    if not batches:
        return []
    threads = max(1, int(threads))
    out: list[_CacheFetchOutcome | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        fut_to_idx = {}
        for idx, batch in enumerate(batches):
            fut = pool.submit(
                _fetch_single_request,
                req=batch.primary,
                cache_dir=cache_dir,
                force_refresh=bool(force_refresh),
                timeout_sec=timeout_sec,
                client_id_offset=int(client_id_base) + idx,
                mend_threads=max(1, int(mend_threads)),
                mend_retries=max(1, int(mend_retries)),
                mend_adaptive_threads=bool(mend_adaptive_threads),
            )
            fut_to_idx[fut] = idx
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            out[idx] = fut.result()
    return [row for row in out if isinstance(row, _CacheFetchOutcome)]


def main_repair(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Repair ET session completeness for cached intraday bars.")
    ap.add_argument("--cache-file", action="append", required=True, help="Path to target cache CSV (repeatable).")
    ap.add_argument(
        "--archive-file",
        action="append",
        default=[],
        help="Archive cache CSV candidate(s) for overlay before IBKR refetch.",
    )
    ap.add_argument("--heal", action="store_true", help="Apply quality-gated replacements to the cache file.")
    ap.add_argument("--aggressive", action="store_true", help="Aggressive defaults for threads/retries/timeout.")
    ap.add_argument("--threads", type=int, default=6, help="IBKR day-fetch worker threads.")
    ap.add_argument("--retries", type=int, default=2, help="IBKR retries per ET day.")
    ap.add_argument("--timeout-sec", type=float, default=40.0, help="IBKR reqHistoricalData timeout seconds.")
    ap.add_argument(
        "--client-id-base",
        type=int,
        default=500,
        help="Base IBKR client-id offset for threaded workers.",
    )
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak before modifying cache.")
    ap.add_argument(
        "--no-adaptive-shards",
        action="store_true",
        help="Disable adaptive worker-thread step-down retries for retryable IBKR errors.",
    )
    ap.add_argument(
        "--ibkr-min-et-day",
        default=None,
        help="Only IBKR-refetch ET days on/after this date (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--ibkr-max-et-day",
        default=None,
        help="Only IBKR-refetch ET days on/before this date (YYYY-MM-DD).",
    )
    ap.add_argument("--report-path", default=None, help="Output JSON report path.")
    args = ap.parse_args(argv)

    threads = int(args.threads)
    retries = int(args.retries)
    timeout_sec = float(args.timeout_sec)
    if args.aggressive:
        threads = max(threads, 10)
        retries = max(retries, 3)
        timeout_sec = max(timeout_sec, 45.0)

    cache_files = [Path(p) for p in args.cache_file]
    archive_files = [Path(p) for p in args.archive_file]
    ibkr_min_et_day = _parse_date(args.ibkr_min_et_day) if args.ibkr_min_et_day else None
    ibkr_max_et_day = _parse_date(args.ibkr_max_et_day) if args.ibkr_max_et_day else None

    reports: list[dict[str, object]] = []
    for cache_file in cache_files:
        report = _process_cache_file(
            cache_file=cache_file,
            archive_files=archive_files,
            heal=bool(args.heal),
            threads=threads,
            retries=retries,
            timeout_sec=timeout_sec,
            client_id_base=int(args.client_id_base),
            backup=not bool(args.no_backup),
            adaptive_threads=not bool(args.no_adaptive_shards),
            ibkr_min_et_day=ibkr_min_et_day,
            ibkr_max_et_day=ibkr_max_et_day,
        )
        reports.append(report)

    out = {
        "generated_at_utc": datetime.now(tz=_UTC).isoformat(),
        "heal": bool(args.heal),
        "aggressive": bool(args.aggressive),
        "threads": threads,
        "adaptive_threads_enabled": not bool(args.no_adaptive_shards),
        "retries": retries,
        "timeout_sec": timeout_sec,
        "reports": reports,
    }

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        stamp = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
        base_dir = cache_files[0].parent if cache_files else Path("db")
        report_path = base_dir / f"cache_repair_report_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("")
    print("=== cache repair ===")
    print(f"- report: {report_path}")
    for rep in reports:
        before = rep["audit_before"]["anomaly_day_count_effective"]
        after = rep["audit_after"]["anomaly_day_count_effective"]
        print(f"- {rep['cache_file']}")
        print(f"  mode(before->after): {rep['detected_mode_before']} -> {rep['detected_mode_after']}")
        print(f"  anomalies(before->after): {before} -> {after}")
        print(f"  updated: {rep['updated_cache']}")
    print("")


def main_resample(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Resample cached OHLCV bars (deterministic, no IBKR fetch).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--src-bar-size", required=True, help='e.g. "5 mins"')
    ap.add_argument("--dst-bar-size", required=True, help='e.g. "10 mins"')
    ap.add_argument("--cache-dir", default="db")
    ap.add_argument("--use-rth", action="store_true", help="Use RTH-only cache tag (default: FULL24).")
    args = ap.parse_args(argv)

    symbol = str(args.symbol).strip().upper()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit(f"Invalid date range: {start}..{end}")
    start_dt = datetime.combine(start, time(0, 0))
    end_dt = datetime.combine(end, time(23, 59))
    cache_dir = Path(args.cache_dir)
    use_rth = bool(args.use_rth)
    src_bar_size = str(args.src_bar_size).strip()
    dst_bar_size = str(args.dst_bar_size).strip()
    provider = IBKRHistoricalData()
    try:
        out = resample_cached_window(
            data=provider,
            cache_dir=cache_dir,
            symbol=symbol,
            exchange=None,
            start=start_dt,
            end=end_dt,
            dst_bar_size=dst_bar_size,
            use_rth=use_rth,
            src_bar_size=src_bar_size,
        )
    finally:
        try:
            provider.disconnect()
        except Exception:
            pass
    if not out.ok:
        raise SystemExit(f"Resample failed: {out.error or 'unknown_error'}")
    if not out.src_path:
        raise SystemExit("Resample failed to resolve source cache path.")

    print("")
    print("=== cache resample ===")
    print(f"- symbol={symbol} use_rth={use_rth}")
    print(f"- src={src_bar_size} rows={int(out.src_rows)} path={out.src_path}")
    print(f"- dst={dst_bar_size} rows={int(out.dst_rows)} path={out.dst_path}")
    print(f"- dropped_incomplete={int(out.dropped_incomplete)}")
    dst_bars = read_cache(out.dst_path)
    if dst_bars:
        print(f"- first={dst_bars[0].ts} last={dst_bars[-1].ts}")
    print("")


def main_fetch(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Threaded cache sync (direct requests and/or champion-derived requirements)."
    )
    ap.add_argument(
        "--request",
        action="append",
        default=[],
        help="SYMBOL|YYYY-MM-DD|YYYY-MM-DD|BAR_SIZE|rth|[source] (repeatable)",
    )
    ap.add_argument(
        "--champion-json",
        action="append",
        default=[],
        help="Champion JSON payload to derive required caches from (repeatable).",
    )
    ap.add_argument(
        "--champion-current",
        action="store_true",
        help="Auto-include current champion payloads from README + spot_champions.json.",
    )
    ap.add_argument("--cache-dir", default="db")
    ap.add_argument("--threads", type=int, default=8, help="Threaded cache requests (effective cap defaults to 2).")
    ap.add_argument("--timeout-sec", type=float, default=45.0, help="IBKR reqHistoricalData timeout.")
    ap.add_argument("--client-id-base", type=int, default=900, help="Base client-id offset for worker sessions.")
    ap.add_argument(
        "--max-primary-span-days",
        type=int,
        default=0,
        help=(
            "Split large merged primary windows into shards no larger than N days "
            "(0 disables sharding)."
        ),
    )
    ap.add_argument("--aggressive", action="store_true", help="Raise threads and timeout defaults.")
    ap.add_argument("--force-refresh", action="store_true", help="Delete target cache file before fetching.")
    ap.add_argument(
        "--no-adaptive-shards",
        action="store_true",
        help="Disable adaptive shard step-down retries for retryable IBKR errors.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print deduped requests and exit.")
    ap.add_argument("--report-path", default=None, help="Output JSON report path.")
    args = ap.parse_args(argv)

    requests: list[_CacheFetchRequest] = []
    for raw in args.request:
        requests.append(_parse_fetch_request(str(raw)))

    champion_paths = [Path(p) for p in args.champion_json]
    if args.champion_current:
        champion_paths.extend(_discover_current_champion_jsons())
    for path in champion_paths:
        resolved = path if path.is_absolute() else (_repo_root() / path).resolve()
        if not resolved.exists():
            raise SystemExit(f"Champion JSON not found: {resolved}")
        requests.extend(_requests_from_champion_json(resolved))

    requests = _dedupe_fetch_requests(requests)
    if not requests:
        raise SystemExit("No fetch requests found. Use --request and/or --champion-json/--champion-current.")
    max_primary_span_days = int(args.max_primary_span_days)
    if max_primary_span_days < 0:
        raise SystemExit("--max-primary-span-days must be >= 0")
    batches = _build_fetch_batches(requests, max_primary_span_days=max_primary_span_days)

    cache_dir = Path(args.cache_dir)
    requested_threads = max(1, int(args.threads))
    threads = min(requested_threads, int(_FETCH_SAFE_THREAD_MAX))
    thread_cap_applied = threads < requested_threads
    timeout_sec = max(1.0, float(args.timeout_sec))
    mend_retries = 3 if bool(args.aggressive) else 2
    if args.aggressive:
        timeout_sec = max(timeout_sec, 55.0)

    if args.dry_run:
        print("")
        print("=== cache sync dry-run ===")
        for batch in batches:
            req = batch.primary
            tag = "rth" if req.use_rth else "full24"
            print(
                f"- PRIMARY {req.symbol} {req.start}..{req.end} {req.bar_size} {tag} "
                f"targets={len(batch.targets)} source={req.source}"
            )
        print(f"- primary_batches={len(batches)} total_requests={len(requests)}")
        if max_primary_span_days > 0:
            print(f"- max_primary_span_days={max_primary_span_days}")
        print("")
        return

    thread_plan = [threads] if bool(args.no_adaptive_shards) else _adaptive_thread_plan(threads)
    pending: list[int] = list(range(len(batches)))
    primary_outcomes: dict[int, _CacheFetchOutcome] = {}
    last_retry: dict[int, _CacheFetchOutcome] = {}
    pass_stats: list[dict[str, object]] = []
    force_refresh_this_pass = bool(args.force_refresh)

    for pass_idx, pass_threads in enumerate(thread_plan):
        if not pending:
            break
        pass_indices = sorted(pending)
        pass_batches = [batches[i] for i in pass_indices]
        pass_outcomes = _run_primary_batches(
            batches=pass_batches,
            cache_dir=cache_dir,
            force_refresh=force_refresh_this_pass,
            timeout_sec=timeout_sec,
            threads=pass_threads,
            client_id_base=int(args.client_id_base) + (pass_idx * 1000),
            mend_threads=max(1, min(4, 8 // max(1, int(pass_threads)))),
            mend_retries=max(1, int(mend_retries)),
            mend_adaptive_threads=not bool(args.no_adaptive_shards),
        )
        force_refresh_this_pass = False

        retry_next: list[int] = []
        ok_n = 0
        fail_n = 0
        retryable_n = 0
        for local_idx, outcome in enumerate(pass_outcomes):
            global_idx = pass_indices[local_idx]
            if outcome.ok:
                primary_outcomes[global_idx] = outcome
                ok_n += 1
                continue
            if pass_idx < (len(thread_plan) - 1) and _is_retryable_ibkr_error(outcome.error):
                retry_next.append(global_idx)
                last_retry[global_idx] = outcome
                retryable_n += 1
                continue
            primary_outcomes[global_idx] = outcome
            fail_n += 1

        pending = retry_next
        pass_stats.append(
            {
                "pass": int(pass_idx + 1),
                "threads": int(pass_threads),
                "attempted_batches": len(pass_indices),
                "ok_batches": int(ok_n),
                "retryable_fail_batches": int(retryable_n),
                "terminal_fail_batches": int(fail_n),
            }
        )

    if pending:
        for idx in pending:
            prior = last_retry.get(idx)
            if prior is not None:
                primary_outcomes[idx] = prior
                continue
            req = batches[idx].primary
            fallback_path = cache_path(
                cache_dir,
                req.symbol,
                datetime.combine(req.start, time(0, 0)),
                datetime.combine(req.end, time(23, 59)),
                req.bar_size,
                req.use_rth,
            )
            primary_outcomes[idx] = _CacheFetchOutcome(
                request=req,
                ok=False,
                from_cache=False,
                rows=0,
                cache_path=str(fallback_path),
                error="retry_exhausted",
                status="fail",
            )

    primary_records: list[dict[str, object]] = []
    request_records: list[dict[str, object]] = []
    for idx, batch in enumerate(batches):
        primary = batch.primary
        primary_out = primary_outcomes[idx]
        primary_records.append(
            {
                "symbol": primary.symbol,
                "start": primary.start.isoformat(),
                "end": primary.end.isoformat(),
                "bar_size": primary.bar_size,
                "use_rth": bool(primary.use_rth),
                "targets": len(batch.targets),
                "ok": bool(primary_out.ok),
                "rows": int(primary_out.rows),
                "cache_path": str(primary_out.cache_path),
                "error": primary_out.error,
                "status": str(primary_out.status or ("ok" if primary_out.ok else "fail")),
                "healed": bool(primary_out.healed),
                "healed_days": int(primary_out.healed_days),
                "remaining_days": int(primary_out.remaining_days),
                "session_missing_days_before": int(primary_out.session_missing_days_before),
                "session_missing_days_after": int(primary_out.session_missing_days_after),
                "gap_days_before": int(primary_out.gap_days_before),
                "gap_days_after": int(primary_out.gap_days_after),
                "timezone_canonicalized": bool(primary_out.timezone_canonicalized),
                "anomaly_fingerprint": primary_out.anomaly_fingerprint,
            }
        )

        if not primary_out.ok:
            for target in batch.targets:
                target_path = cache_path(
                    cache_dir,
                    target.symbol,
                    datetime.combine(target.start, time(0, 0)),
                    datetime.combine(target.end, time(23, 59)),
                    target.bar_size,
                    target.use_rth,
                )
                request_records.append(
                    {
                        "symbol": target.symbol,
                        "start": target.start.isoformat(),
                        "end": target.end.isoformat(),
                        "bar_size": target.bar_size,
                        "use_rth": bool(target.use_rth),
                        "source": target.source,
                        "ok": False,
                        "from_cache": False,
                        "rows": 0,
                        "cache_path": str(target_path),
                        "error": primary_out.error,
                        "status": "fail",
                        "healed": bool(primary_out.healed),
                        "healed_days": int(primary_out.healed_days),
                        "remaining_days": int(primary_out.remaining_days),
                        "session_missing_days_before": int(primary_out.session_missing_days_before),
                        "session_missing_days_after": int(primary_out.session_missing_days_after),
                        "gap_days_before": int(primary_out.gap_days_before),
                        "gap_days_after": int(primary_out.gap_days_after),
                        "timezone_canonicalized": bool(primary_out.timezone_canonicalized),
                        "anomaly_fingerprint": primary_out.anomaly_fingerprint,
                        "primary_window": {
                            "start": primary.start.isoformat(),
                            "end": primary.end.isoformat(),
                        },
                    }
                )
            continue

        primary_path = Path(str(primary_out.cache_path))
        try:
            primary_rows = list(read_cache(primary_path))
        except Exception as exc:  # pragma: no cover - filesystem dependent
            msg = f"primary_cache_read_error: {exc}"
            for target in batch.targets:
                target_path = cache_path(
                    cache_dir,
                    target.symbol,
                    datetime.combine(target.start, time(0, 0)),
                    datetime.combine(target.end, time(23, 59)),
                    target.bar_size,
                    target.use_rth,
                )
                request_records.append(
                    {
                        "symbol": target.symbol,
                        "start": target.start.isoformat(),
                        "end": target.end.isoformat(),
                        "bar_size": target.bar_size,
                        "use_rth": bool(target.use_rth),
                        "source": target.source,
                        "ok": False,
                        "from_cache": False,
                        "rows": 0,
                        "cache_path": str(target_path),
                        "error": msg,
                        "status": "fail",
                        "healed": bool(primary_out.healed),
                        "healed_days": int(primary_out.healed_days),
                        "remaining_days": int(primary_out.remaining_days),
                        "session_missing_days_before": int(primary_out.session_missing_days_before),
                        "session_missing_days_after": int(primary_out.session_missing_days_after),
                        "gap_days_before": int(primary_out.gap_days_before),
                        "gap_days_after": int(primary_out.gap_days_after),
                        "timezone_canonicalized": bool(primary_out.timezone_canonicalized),
                        "anomaly_fingerprint": primary_out.anomaly_fingerprint,
                        "primary_window": {
                            "start": primary.start.isoformat(),
                            "end": primary.end.isoformat(),
                        },
                    }
                )
            continue

        for target in batch.targets:
            target_path = cache_path(
                cache_dir,
                target.symbol,
                datetime.combine(target.start, time(0, 0)),
                datetime.combine(target.end, time(23, 59)),
                target.bar_size,
                target.use_rth,
            )
            is_primary_target = (target.start == primary.start) and (target.end == primary.end)
            if is_primary_target:
                rows = len(primary_rows)
                ok = rows > 0
                from_cache = bool(primary_out.from_cache)
                err = None if ok else "empty_primary_cache"
                status = str(primary_out.status or ("ok" if ok else "fail"))
                healed = bool(primary_out.healed)
                healed_days = int(primary_out.healed_days)
                remaining_days = int(primary_out.remaining_days)
                session_before = int(primary_out.session_missing_days_before)
                session_after = int(primary_out.session_missing_days_after)
                gap_before = int(primary_out.gap_days_before)
                gap_after = int(primary_out.gap_days_after)
                tz_canonicalized = bool(primary_out.timezone_canonicalized)
                anomaly_fp = primary_out.anomaly_fingerprint
            else:
                target_start_dt = datetime.combine(target.start, time(0, 0))
                target_end_dt = datetime.combine(target.end, time(23, 59))
                try:
                    sliced = [bar for bar in primary_rows if target_start_dt <= bar.ts <= target_end_dt]
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if bool(args.force_refresh) and target_path.exists():
                        target_path.unlink()
                    write_cache(target_path, sliced)
                    rows = len(sliced)
                    ok = rows > 0
                    from_cache = False
                    err = None if ok else "empty_slice_from_primary"
                    status = "ok" if ok else "fail"
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    rows = 0
                    ok = False
                    from_cache = False
                    err = f"slice_write_error: {exc}"
                    status = "fail"
                healed = False
                healed_days = 0
                remaining_days = 0
                session_before = 0
                session_after = 0
                gap_before = 0
                gap_after = 0
                tz_canonicalized = False
                anomaly_fp = None

            request_records.append(
                {
                    "symbol": target.symbol,
                    "start": target.start.isoformat(),
                    "end": target.end.isoformat(),
                    "bar_size": target.bar_size,
                    "use_rth": bool(target.use_rth),
                    "source": target.source,
                    "ok": bool(ok),
                    "from_cache": bool(from_cache),
                    "rows": int(rows),
                    "cache_path": str(target_path),
                    "error": err,
                    "status": status,
                    "healed": bool(healed),
                    "healed_days": int(healed_days),
                    "remaining_days": int(remaining_days),
                    "session_missing_days_before": int(session_before),
                    "session_missing_days_after": int(session_after),
                    "gap_days_before": int(gap_before),
                    "gap_days_after": int(gap_after),
                    "timezone_canonicalized": bool(tz_canonicalized),
                    "anomaly_fingerprint": anomaly_fp,
                    "primary_window": {
                        "start": primary.start.isoformat(),
                        "end": primary.end.isoformat(),
                    },
                }
            )

    request_records.sort(
        key=lambda r: (
            str(r.get("symbol") or ""),
            str(r.get("start") or ""),
            str(r.get("end") or ""),
            str(r.get("bar_size") or ""),
            bool(r.get("use_rth")),
        )
    )
    ok_requests = [r for r in request_records if bool(r.get("ok"))]
    fail_requests = [r for r in request_records if not bool(r.get("ok"))]
    cache_hits = sum(1 for r in ok_requests if bool(r.get("from_cache")))
    primary_ok = sum(1 for r in primary_records if bool(r.get("ok")))
    primary_fail = len(primary_records) - primary_ok
    status_order = {"ok": 0, "warn": 1, "fail": 2}
    primary_ok_status = sum(1 for r in primary_records if str(r.get("status") or "ok") == "ok")
    primary_warn_status = sum(1 for r in primary_records if str(r.get("status") or "ok") == "warn")
    primary_fail_status = sum(1 for r in primary_records if str(r.get("status") or "ok") == "fail")
    healed_primary_count = sum(1 for r in primary_records if bool(r.get("healed")))
    tz_canon_primary_count = sum(1 for r in primary_records if bool(r.get("timezone_canonicalized")))
    days_healed_total = sum(int(r.get("healed_days") or 0) for r in primary_records)
    remaining_days_total = sum(int(r.get("remaining_days") or 0) for r in primary_records)
    session_days_fixed_total = sum(
        max(
            0,
            int(r.get("session_missing_days_before") or 0) - int(r.get("session_missing_days_after") or 0),
        )
        for r in primary_records
    )
    gap_days_fixed_total = sum(
        max(0, int(r.get("gap_days_before") or 0) - int(r.get("gap_days_after") or 0))
        for r in primary_records
    )

    report = {
        "generated_at_utc": datetime.now(tz=_UTC).isoformat(),
        "threads": threads,
        "requested_threads": int(requested_threads),
        "thread_cap": int(_FETCH_SAFE_THREAD_MAX),
        "thread_cap_applied": bool(thread_cap_applied),
        "adaptive_threads_enabled": not bool(args.no_adaptive_shards),
        "thread_plan": [int(x) for x in thread_plan],
        "max_primary_span_days": int(max_primary_span_days),
        "pass_stats": pass_stats,
        "timeout_sec": timeout_sec,
        "force_refresh": bool(args.force_refresh),
        "request_count": len(request_records),
        "ok_count": len(ok_requests),
        "fail_count": len(fail_requests),
        "cache_hit_count": int(cache_hits),
        "primary_batch_count": len(primary_records),
        "primary_ok_count": int(primary_ok),
        "primary_fail_count": int(primary_fail),
        "health_rollup": {
            "primary_status_ok_count": int(primary_ok_status),
            "primary_status_warn_count": int(primary_warn_status),
            "primary_status_fail_count": int(primary_fail_status),
            "healed_primary_count": int(healed_primary_count),
            "timezone_canonicalized_primary_count": int(tz_canon_primary_count),
            "days_healed_total": int(days_healed_total),
            "remaining_days_total": int(remaining_days_total),
            "session_days_fixed_total": int(session_days_fixed_total),
            "gap_days_fixed_total": int(gap_days_fixed_total),
        },
        "primary_batches": primary_records,
        "requests": request_records,
    }

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        stamp = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
        report_path = cache_dir / f"cache_sync_report_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print("=== cache sync ===")
    print(f"- report: {report_path}")
    print(
        f"- primary_batches: {len(primary_records)} "
        f"ok={primary_ok} fail={primary_fail} thread_plan={thread_plan}"
    )
    if thread_cap_applied:
        print(
            f"- thread_cap_applied: requested={requested_threads} "
            f"effective={threads} cap={int(_FETCH_SAFE_THREAD_MAX)}"
        )
    if max_primary_span_days > 0:
        print(f"- max_primary_span_days={max_primary_span_days}")
    print(
        f"- health(primary): ok={primary_ok_status} warn={primary_warn_status} fail={primary_fail_status} "
        f"healed={healed_primary_count} tz_canon={tz_canon_primary_count} "
        f"days_healed={days_healed_total} remaining_days={remaining_days_total}"
    )
    print(
        f"- heal_breakdown: session_days_fixed={session_days_fixed_total} "
        f"gap_days_fixed={gap_days_fixed_total}"
    )
    print(
        f"- requests: {len(request_records)} "
        f"ok={len(ok_requests)} fail={len(fail_requests)} cache_hits={cache_hits}"
    )
    flagged_primary = [
        row for row in primary_records if str(row.get("status") or "ok") in {"warn", "fail"}
    ]
    flagged_primary.sort(
        key=lambda row: (
            status_order.get(str(row.get("status") or "ok"), 0),
            int(row.get("remaining_days") or 0),
            int(row.get("healed_days") or 0),
        ),
        reverse=True,
    )
    max_flagged_lines = 12
    for row in flagged_primary[:max_flagged_lines]:
        tag = "rth" if bool(row.get("use_rth")) else "full24"
        fp = str(row.get("anomaly_fingerprint") or "-")
        print(
            f"  {str(row.get('status') or 'ok').upper()} {row.get('symbol')} "
            f"{row.get('start')}..{row.get('end')} {row.get('bar_size')} {tag} "
            f"healed={int(row.get('healed_days') or 0)} "
            f"remaining={int(row.get('remaining_days') or 0)} "
            f"gap={int(row.get('gap_days_before') or 0)}->{int(row.get('gap_days_after') or 0)} "
            f"sess={int(row.get('session_missing_days_before') or 0)}->{int(row.get('session_missing_days_after') or 0)} "
            f"tz={1 if bool(row.get('timezone_canonicalized')) else 0} fp={fp}"
        )
    if len(flagged_primary) > max_flagged_lines:
        print(f"  ... (+{len(flagged_primary) - max_flagged_lines} more flagged primary requests)")

    max_fail_lines = 10
    for row in fail_requests[:max_fail_lines]:
        tag = "rth" if bool(row.get("use_rth")) else "full24"
        print(
            f"  FAIL {row.get('symbol')} {row.get('start')}..{row.get('end')} "
            f"{row.get('bar_size')} {tag}: {row.get('error')}"
        )
    if len(fail_requests) > max_fail_lines:
        print(f"  ... (+{len(fail_requests) - max_fail_lines} more failed request slices; see report)")
    print("")


def main_audit_heal(argv: list[str]) -> None:
    """Backward-compatible alias for repair command."""
    main_repair(argv)


def main_sync(argv: list[str]) -> None:
    """Canonical alias for sync command."""
    main_fetch(argv)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Usage: cache_ops <sync|repair|resample> ...")
    cmd = str(args[0]).strip().lower()
    if cmd in {"-h", "--help", "help"}:
        print("Usage: cache_ops <sync|repair|resample> ...")
        print("Subcommands:")
        print("  sync         Threaded cache retrieval with built-in mend and timezone canonicalization.")
        print("  repair       Targeted audit/heal of existing cache windows.")
        print("  resample     Deterministic cache resampling from existing source bars.")
        print("Aliases:")
        print("  fetch -> sync")
        print("  audit-heal/audit/heal -> repair")
        return 0
    if cmd in {"repair", "audit-heal", "audit", "heal"}:
        main_repair(args[1:])
        return 0
    if cmd in {"resample"}:
        main_resample(args[1:])
        return 0
    if cmd in {"sync", "fetch"}:
        main_sync(args[1:])
        return 0
    raise SystemExit(f"Unknown subcommand: {args[0]!r}. Use sync, repair, or resample.")


if __name__ == "__main__":
    raise SystemExit(main())
