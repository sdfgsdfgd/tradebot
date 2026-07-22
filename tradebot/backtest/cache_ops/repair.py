"""Cache auditing, targeted repair, and broker-backed healing."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ...time_utils import UTC as _UTC
from ..cache import read_cache, write_cache
from ..data import IBKRHistoricalData
from ..models import Bar
from ..trading_calendar import utc_bounds_for_et_day as _utc_bounds_for_et_day
from .coverage import (
    _audit_rows,
    _bars_for_day,
    _build_indices,
    _canonicalize_rows,
    _day_quality,
    _dedupe_sort_bars,
    _infer_timestamp_mode,
    _merge_smart_overnight,
    _replace_day,
    _require_cache_meta,
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
