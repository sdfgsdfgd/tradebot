"""Cache timestamp, session-coverage, and bar-quality truth."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

from ...signals import parse_bar_size
from ...time_utils import (
    NaiveTsMode,
    NaiveTsSourceMode,
    to_et as _to_et_shared,
    to_utc_naive as _to_utc_naive_shared,
)
from ..cache import CacheFileMeta, parse_cache_filename
from ..models import Bar
from ..trading_calendar import (
    SESSION_ORDER as _SESSION_ORDER,
    SESSION_WEIGHTS as _SESSION_WEIGHTS,
    et_day_from_utc_naive as _et_day_from_utc_naive,
    expected_sessions as _expected_sessions,
    is_maintenance_gap as _is_maintenance_gap,
    session_label_et as _session_label_et,
)

def _require_cache_meta(path: Path) -> CacheFileMeta:
    meta = parse_cache_filename(path)
    if meta is not None:
        return meta
    raise SystemExit(
        f"Unable to parse symbol/start/end from cache filename: {path.name}. "
        "Expected pattern: SYMBOL_YYYY-MM-DD_YYYY-MM-DD_<approved_bar>_<rth|full24|full>.csv"
    )


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
