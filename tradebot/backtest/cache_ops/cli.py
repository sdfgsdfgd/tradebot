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
import json
import sys
from datetime import datetime, time
from pathlib import Path

from ...time_utils import UTC as _UTC
from ..cache import cache_path, read_cache, write_cache
from ..data import IBKRHistoricalData
from .repair import (
    _adaptive_thread_plan,
    _is_retryable_ibkr_error,
    _process_cache_file,
)
from .resample import resample_cached_window
from .sync import (
    _CacheFetchOutcome,
    _CacheFetchRequest,
    _build_fetch_batches,
    _dedupe_fetch_requests,
    _discover_current_champion_jsons,
    _parse_date,
    _parse_fetch_request,
    _repo_root,
    _requests_from_champion_json,
    _run_primary_batches,
)


_FETCH_SAFE_THREAD_MAX = 2

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
