"""Cache request discovery, batching, retrieval, and mending."""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from ...signals import parse_bar_size
from ...time_utils import (
    NaiveTsMode,
    NaiveTsSourceMode,
    to_et as _to_et_shared,
)
from ..cache import cache_path, read_cache, write_cache
from ..data import IBKRHistoricalData
from ..models import Bar
from ..spot_context import SpotBarRequirement, spot_bar_requirements_from_strategy
from ...engines.market import session_label_et as _session_label_et
from ...spot.champions import discover_current_champions
from .coverage import (
    _audit_rows,
    _build_indices,
    _canonicalize_rows,
    _coverage_fingerprint,
    _dedupe_sort_bars,
    _infer_timestamp_mode,
    _intra_session_gap_days,
    _supports_session_audit,
)
from .repair import _ibkr_overlay_adaptive

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


def _discover_current_champion_jsons() -> list[Path]:
    out = [ref.artifact_path for ref in discover_current_champions()]
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

    def _overlay_days(rows: list[Bar], days: list[date]) -> list[Bar]:
        by_ts, day_to_ts = _build_indices(rows)
        _ibkr_overlay_adaptive(
            symbol=req.symbol,
            bar_size=req.bar_size,
            use_rth=bool(req.use_rth),
            by_ts=by_ts,
            day_to_ts=day_to_ts,
            candidate_days=days,
            threads=max(1, int(mend_threads)),
            retries=max(1, int(mend_retries)),
            timeout_sec=timeout_sec,
            client_id_base=(int(client_id_offset) * 1000),
            session_mode=session_mode,
            adaptive_threads=bool(mend_adaptive_threads),
        )
        return _dedupe_sort_bars(by_ts.values())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    provider = IBKRHistoricalData(client_id_offset=client_id_offset)
    backup_path: Path | None = None
    fetch_attempted = False
    fetch_accepted = False

    def _stage_existing() -> None:
        nonlocal backup_path
        if backup_path is not None or not cache_file.exists():
            return
        backup_path = cache_file.with_name(
            f".{cache_file.name}.pre_fetch.{os.getpid()}.{os.urandom(4).hex()}"
        )
        os.replace(cache_file, backup_path)

    try:
        if force_refresh:
            _stage_existing()

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
                    final_rows = _overlay_days(canonical_rows, candidate_days)

                changed = bool(tz_canon)
                if len(final_rows) != len(canonical_rows):
                    changed = True
                elif not changed:
                    changed = any(a != b for a, b in zip(canonical_rows, final_rows))

                rows_in_window = _rows_in_window(final_rows)
                if supports_audit:
                    audit_after, gaps_after, remaining_days, session_after, gap_after = _coverage_days(final_rows)
                    healed_days = max(0, candidate_days_count - len(remaining_days))
                    anomaly_fingerprint = _coverage_fingerprint(audit_after, gaps_after)
                    if not remaining_days and final_rows:
                        ok = bool(rows_in_window)
                        if changed:
                            write_cache(cache_file, final_rows)
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
                    _stage_existing()
                else:
                    ok = bool(rows_in_window)
                    if changed:
                        write_cache(cache_file, final_rows)
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

        fetch_attempted = True
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
        tz_canon = bool(tz_canon or (mode_after != NaiveTsSourceMode.UTC_NAIVE.value))

        rows_in_window = _rows_in_window(canonical_after)
        if not supports_audit:
            ok = bool(rows_in_window)
            if ok:
                if mode_after != NaiveTsSourceMode.UTC_NAIVE.value:
                    write_cache(cache_file, canonical_after)
                fetch_accepted = True
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
        if remaining_days:
            candidate_days_count = max(candidate_days_count, len(remaining_days))
            canonical_after = _overlay_days(canonical_after, remaining_days)
            audit_after, gaps_after, remaining_days, session_after, gap_after = _coverage_days(
                canonical_after
            )
            rows_in_window = _rows_in_window(canonical_after)
        healed_days = max(int(healed_days), max(0, int(candidate_days_count) - len(remaining_days)))
        anomaly_fingerprint = _coverage_fingerprint(audit_after, gaps_after)
        ok_complete = bool(rows_in_window) and not remaining_days
        error = None
        if not ok_complete:
            missing_dates = ",".join(day.isoformat() for day in remaining_days[:8])
            error = (
                "incomplete_after_fetch:"
                f" session_days={int(audit_after.get('anomaly_day_count_effective', 0) or 0)}"
                f" gap_days={len(gaps_after)}"
                f" dates={missing_dates or '-'}"
            )
        else:
            if mode_after != NaiveTsSourceMode.UTC_NAIVE.value or healed_days > 0:
                write_cache(cache_file, canonical_after)
            fetch_accepted = True
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
            if backup_path is not None:
                if fetch_accepted:
                    backup_path.unlink(missing_ok=True)
                else:
                    cache_file.unlink(missing_ok=True)
                    os.replace(backup_path, cache_file)
            elif fetch_attempted and not fetch_accepted:
                cache_file.unlink(missing_ok=True)
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
