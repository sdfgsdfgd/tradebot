"""Spot multi-window stability eval (kingmaker) entrypoint.

This module now owns the physical implementation for the `spot_multitimeframe`
workflow. A thin compatibility wrapper remains in
`run_backtests_spot_sweeps.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time as pytime
from datetime import date, datetime, time
from pathlib import Path

from .cli_utils import parse_date as _parse_date, parse_window as _parse_window
from .config import ConfigBundle
from .multiwindow_helpers import (
    die_empty_bars as _mw_die_empty_bars,
    load_bars as _mw_load_bars,
    preflight_offline_cache_or_die as _mw_preflight_offline_cache_or_die,
)
from .spot_codec import (
    filters_from_payload as _codec_filters_from_payload,
    make_bundle as _codec_make_bundle,
    metrics_from_summary as _codec_metrics_from_summary,
    strategy_from_payload as _codec_strategy_from_payload,
)
from .spot_context import SpotBarRequirement, load_spot_context_bars
from .data import ContractMeta, IBKRHistoricalData
from .engine import _run_spot_backtest_summary, _spot_multiplier
from .sweep_fingerprint import _strategy_fingerprint
from .sweep_parallel import (
    _collect_parallel_payload_records,
    _parse_worker_shard,
    _progress_line,
    _run_parallel_stage_kernel,
    _strip_flags,
)
from .sweeps import (
    utc_now_iso_z,
    write_json,
)
from ..series_cache import series_cache_service
from ..time_utils import now_et as _now_et

_SERIES_CACHE = series_cache_service()
_SWEEP_MULTIWINDOW_BARS_NAMESPACE = "spot.sweeps.multiwindow.bars"
# Bump whenever evaluation semantics change so stale cached rows don't mask runtime fixes.
_MULTIWINDOW_CACHE_ENGINE_VERSION = "spot_multiwindow_v8"

def _score_key(item: dict) -> tuple:
    return (
        float(item.get("stability_min_roi_dd") or item.get("stability_min_pnl_dd") or float("-inf")),
        float(item.get("stability_min_roi") or item.get("stability_min_pnl") or float("-inf")),
        float(item.get("full_roi_over_dd_pct") or item.get("full_pnl_over_dd") or float("-inf")),
        float(item.get("full_roi") or item.get("full_pnl") or float("-inf")),
        float(item.get("full_win") or 0.0),
        int(item.get("full_trades") or 0),
    )


def _strategy_key(strategy: dict, *, filters: dict | None) -> str:
    return _strategy_fingerprint(strategy, filters=filters)


# endregion


# region CLI
def spot_multitimeframe_main() -> None:
    ap = argparse.ArgumentParser(prog="tradebot.backtest.multitimeframe")
    ap.add_argument("--milestones", required=True, help="Input spot milestones JSON to evaluate.")
    ap.add_argument("--symbol", default="TQQQ", help="Symbol to filter (default: TQQQ).")
    ap.add_argument("--bar-size", default="1 hour", help="Signal bar size filter (default: 1 hour).")
    ap.add_argument("--use-rth", action="store_true", help="Filter to RTH-only strategies.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use cached bars at evaluation time. "
            "With --cache-policy=auto, preflight may hydrate missing caches before run."
        ),
    )
    ap.add_argument(
        "--cache-policy",
        default="auto",
        choices=("auto", "strict"),
        help=(
            "Offline cache preflight policy. "
            "auto = hydrate via cache manager (resample-from-cache or fetch) before evaluating; "
            "strict = fail on any missing cache."
        ),
    )
    ap.add_argument("--cache-dir", default="db", help="Bars cache dir (default: db).")
    ap.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto). Requires --offline for >1.")
    ap.add_argument("--top", type=int, default=200, help="How many candidates to evaluate (after sorting).")
    ap.add_argument("--min-trades", type=int, default=200, help="Min trades per window.")
    ap.add_argument(
        "--min-trades-per-year",
        type=float,
        default=None,
        help=(
            "Min trades per year per window (e.g. 500 => 1y>=500, 2y>=1000, 10y>=5000). "
            "Enforced as ceil(window_years * min_trades_per_year)."
        ),
    )
    ap.add_argument("--min-win", type=float, default=0.0, help="Min win rate per window (0..1).")
    ap.add_argument(
        "--allow-unlimited-stacking",
        action="store_true",
        default=False,
        help="Deprecated no-op for spot; kept only for CLI compatibility.",
    )
    ap.add_argument(
        "--require-close-eod",
        action="store_true",
        default=False,
        help="Require spot_close_eod=true (forces strategies to close at end of day).",
    )
    ap.add_argument(
        "--require-positive-pnl",
        action="store_true",
        default=False,
        help="Require pnl>0 in every evaluation window.",
    )
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="Evaluation window formatted YYYY-MM-DD:YYYY-MM-DD. Repeatable.",
    )
    ap.add_argument(
        "--include-full",
        action="store_true",
        help="Also evaluate the full window from the milestones payload notes (best-effort).",
    )
    ap.add_argument(
        "--write-top",
        type=int,
        default=0,
        help="Write a small milestones JSON of the top K stability winners (0 disables).",
    )
    ap.add_argument(
        "--out",
        default="backtests/out/multitimeframe_top.json",
        help="Output file for --write-top (default: backtests/out/multitimeframe_top.json).",
    )
    # Internal flags (used by parallel worker sharding).
    ap.add_argument("--multitimeframe-worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-workers", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-out", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()
    if bool(args.allow_unlimited_stacking):
        print("[compat] --allow-unlimited-stacking is deprecated and ignored for spot multitimeframe eval.", flush=True)
    try:
        min_trades_per_year = float(args.min_trades_per_year) if args.min_trades_per_year is not None else None
    except (TypeError, ValueError):
        min_trades_per_year = None
    if min_trades_per_year is not None and min_trades_per_year < 0:
        raise SystemExit("--min-trades-per-year must be >= 0")

    def _default_jobs() -> int:
        detected = os.cpu_count()
        if detected is None:
            return 1
        try:
            detected_i = int(detected)
        except (TypeError, ValueError):
            return 1
        return max(1, detected_i)

    try:
        jobs = int(args.jobs) if args.jobs is not None else 0
    except (TypeError, ValueError):
        jobs = 0
    jobs_eff = _default_jobs() if jobs <= 0 else min(int(jobs), _default_jobs())
    jobs_eff = max(1, int(jobs_eff))

    milestones_path = Path(args.milestones)
    payload = json.loads(milestones_path.read_text())
    groups = payload.get("groups") or []
    symbol = str(args.symbol).strip().upper()
    bar_size = str(args.bar_size).strip().lower()
    use_rth = bool(args.use_rth)

    candidates: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        filters_payload = group.get("filters")
        entries = group.get("entries") or []
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strat = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strat, dict) or not isinstance(metrics, dict):
            continue
        if str(strat.get("instrument", "spot") or "spot").strip().lower() != "spot":
            continue
        if str(strat.get("symbol") or "").strip().upper() != symbol:
            continue
        if str(strat.get("signal_bar_size") or "").strip().lower() != bar_size:
            continue
        if bool(strat.get("signal_use_rth")) != use_rth:
            continue
        candidates.append(
            {
                "group_name": str(group.get("name") or ""),
                "filters": filters_payload,
                "strategy": strat,
                "metrics": metrics,
            }
        )

    if not candidates:
        raise SystemExit(f"No candidates found for {symbol} bar={bar_size} rth={use_rth} in {milestones_path}")

    def _sort_key_seed(item: dict) -> tuple:
        m = item.get("metrics") or {}
        return (
            float(m.get("pnl_over_dd") or float("-inf")),
            float(m.get("pnl") or float("-inf")),
            float(m.get("win_rate") or 0.0),
            int(m.get("trades") or 0),
        )

    # Stage-local semantic dedupe: keep the strongest seed row per canonical strategy key.
    best_candidates: dict[str, dict] = {}
    for cand in candidates:
        strategy_payload = cand.get("strategy") if isinstance(cand.get("strategy"), dict) else {}
        filters_payload = cand.get("filters") if isinstance(cand.get("filters"), dict) else None
        cand_key = _strategy_key(strategy_payload, filters=filters_payload)
        prev = best_candidates.get(cand_key)
        if prev is None or _sort_key_seed(cand) > _sort_key_seed(prev):
            best_candidates[cand_key] = cand

    candidates = sorted(best_candidates.values(), key=_sort_key_seed, reverse=True)[: max(1, int(args.top))]
    jobs_eff = max(1, min(int(jobs_eff), len(candidates)))
    print(
        "multitimeframe prep "
        f"candidates={len(candidates)} jobs={int(jobs_eff)} "
        f"offline={bool(args.offline)} cache_policy={str(args.cache_policy).strip().lower()} cache_dir={args.cache_dir}",
        flush=True,
    )

    windows: list[tuple[date, date]] = []
    for raw in args.window or []:
        windows.append(_parse_window(raw))
    if not windows:
        windows = [
            (_parse_date("2023-01-01"), _parse_date("2024-01-01")),
            (_parse_date("2024-01-01"), _parse_date("2025-01-01")),
            (_parse_date("2025-01-01"), _now_et().date()),
        ]
    print(
        "multitimeframe windows "
        + ", ".join([f"{a.isoformat()}->{b.isoformat()}" for a, b in windows]),
        flush=True,
    )

    cache_dir = Path(args.cache_dir)
    offline = bool(args.offline)
    multiwindow_cache_path = cache_dir / "spot_multiwindow_eval_cache.sqlite3"
    multiwindow_cache_conn: sqlite3.Connection | None = None
    multiwindow_cache_enabled = True
    multiwindow_cache_lock = threading.Lock()
    multiwindow_cache_hits = 0
    multiwindow_cache_writes = 0
    _MULTIWINDOW_CACHE_MISS = object()

    def _multiwindow_cache_conn() -> sqlite3.Connection | None:
        nonlocal multiwindow_cache_conn, multiwindow_cache_enabled
        if not bool(multiwindow_cache_enabled):
            return None
        if multiwindow_cache_conn is not None:
            return multiwindow_cache_conn
        try:
            conn = sqlite3.connect(
                str(multiwindow_cache_path),
                timeout=15.0,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS multiwindow_eval_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            multiwindow_cache_conn = conn
            return conn
        except Exception:
            multiwindow_cache_enabled = False
            multiwindow_cache_conn = None
            return None

    def _multiwindow_windows_signature() -> tuple[tuple[str, str], ...]:
        return tuple((a.isoformat(), b.isoformat()) for a, b in windows)

    def _multiwindow_cache_key(*, strategy_payload: dict, filters_payload: dict | None) -> str:
        raw = {
            "version": str(_MULTIWINDOW_CACHE_ENGINE_VERSION),
            "strategy_key": _strategy_key(strategy_payload, filters=filters_payload),
            "windows": _multiwindow_windows_signature(),
            "min_trades": int(args.min_trades),
            "min_trades_per_year": float(min_trades_per_year) if min_trades_per_year is not None else None,
            "min_win": float(args.min_win),
            "require_close_eod": bool(args.require_close_eod),
            "require_positive_pnl": bool(args.require_positive_pnl),
            "offline": bool(offline),
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _multiwindow_cache_get(*, cache_key: str) -> dict | None | object:
        conn = _multiwindow_cache_conn()
        if conn is None:
            return _MULTIWINDOW_CACHE_MISS
        try:
            with multiwindow_cache_lock:
                row = conn.execute(
                    "SELECT payload_json FROM multiwindow_eval_cache WHERE cache_key=?",
                    (str(cache_key),),
                ).fetchone()
        except Exception:
            return _MULTIWINDOW_CACHE_MISS
        if row is None:
            return _MULTIWINDOW_CACHE_MISS
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            return _MULTIWINDOW_CACHE_MISS
        if payload is None:
            return None
        return dict(payload) if isinstance(payload, dict) else _MULTIWINDOW_CACHE_MISS

    def _multiwindow_cache_set(*, cache_key: str, payload: dict | None) -> None:
        conn = _multiwindow_cache_conn()
        if conn is None:
            return
        try:
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            with multiwindow_cache_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO multiwindow_eval_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (str(cache_key), payload_json, float(pytime.time())),
                )
        except Exception:
            return

    def _required_trades_for_window(wstart: date, wend: date) -> int:
        required = int(args.min_trades)
        if min_trades_per_year is None:
            return required
        days = int((wend - wstart).days) + 1
        years = max(0.0, float(days) / 365.25)
        req_by_year = int(math.ceil(years * float(min_trades_per_year)))
        return max(required, req_by_year)

    def _make_bars_loader(data: IBKRHistoricalData):
        def _load_bars_cached(
            *,
            symbol: str,
            exchange: str | None,
            start_dt: datetime,
            end_dt: datetime,
            bar_size: str,
            use_rth: bool,
            offline: bool,
        ) -> list:
            key = (
                str(symbol),
                str(exchange) if exchange is not None else None,
                start_dt.isoformat(),
                end_dt.isoformat(),
                str(bar_size),
                bool(use_rth),
                bool(offline),
                str(cache_dir),
            )
            cached = _SERIES_CACHE.get(namespace=_SWEEP_MULTIWINDOW_BARS_NAMESPACE, key=key)
            if isinstance(cached, list):
                return cached
            bars = _mw_load_bars(
                data,
                symbol=symbol,
                exchange=exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bar_size,
                use_rth=use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )
            _SERIES_CACHE.set(namespace=_SWEEP_MULTIWINDOW_BARS_NAMESPACE, key=key, value=bars)
            return bars

        return _load_bars_cached

    meta_cache: dict[tuple[str, str | None, bool], ContractMeta] = {}

    def _resolve_meta(bundle: ConfigBundle, *, data: IBKRHistoricalData | None) -> ContractMeta:
        key = (str(bundle.strategy.symbol), bundle.strategy.exchange, bool(offline))
        cached = meta_cache.get(key)
        if cached is not None:
            return cached
        is_future = bundle.strategy.symbol in ("MNQ", "MBT")
        if offline or data is None:
            exchange = "CME" if is_future else "SMART"
            meta = ContractMeta(
                symbol=bundle.strategy.symbol,
                exchange=exchange,
                multiplier=_spot_multiplier(bundle.strategy.symbol, is_future),
                min_tick=0.01,
            )
        else:
            _, resolved = data.resolve_contract(bundle.strategy.symbol, bundle.strategy.exchange)
            meta = ContractMeta(
                symbol=resolved.symbol,
                exchange=resolved.exchange,
                multiplier=_spot_multiplier(bundle.strategy.symbol, is_future, default=resolved.multiplier),
                min_tick=resolved.min_tick,
            )
        meta_cache[key] = meta
        return meta

    def _load_window_context_bars(
        *,
        bundle: ConfigBundle,
        start_dt: datetime,
        end_dt: datetime,
        load_bars_cached,
    ) -> tuple[list | None, list | None, list | None, list | None]:
        def _load_requirement(req: SpotBarRequirement, req_start: datetime, req_end: datetime):
            return load_bars_cached(
                symbol=req.symbol,
                exchange=req.exchange,
                start_dt=req_start,
                end_dt=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
                offline=bundle.backtest.offline,
            )

        def _on_missing(req: SpotBarRequirement, req_start: datetime, req_end: datetime) -> None:
            kind = str(req.kind)
            if kind == "tick":
                kind = "tick_gate"
            _mw_die_empty_bars(
                kind=kind,
                cache_dir=cache_dir,
                symbol=req.symbol,
                exchange=req.exchange,
                start_dt=req_start,
                end_dt=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
                offline=bundle.backtest.offline,
            )

        context = load_spot_context_bars(
            strategy=bundle.strategy,
            default_symbol=str(bundle.strategy.symbol),
            default_exchange=bundle.strategy.exchange,
            default_signal_bar_size=str(bundle.backtest.bar_size),
            default_signal_use_rth=bool(bundle.backtest.use_rth),
            start_dt=start_dt,
            end_dt=end_dt,
            load_requirement=_load_requirement,
            on_missing=_on_missing,
        )
        return context.regime_bars, context.regime2_bars, context.tick_bars, context.exec_bars

    def _evaluate_candidate_multiwindow(
        cand: dict,
        *,
        load_bars_cached,
        data: IBKRHistoricalData | None,
        progress_callback=None,
    ) -> dict | None:
        nonlocal multiwindow_cache_hits, multiwindow_cache_writes
        def _emit_candidate_progress(event: dict | None = None, **payload: object) -> None:
            if not callable(progress_callback):
                return
            merged: dict[str, object] = {}
            if isinstance(event, dict):
                merged.update(event)
            merged.update(payload)
            try:
                progress_callback(dict(merged))
            except Exception:
                return

        filters_payload = cand.get("filters")
        strategy_payload = cand["strategy"]
        cache_key = _multiwindow_cache_key(
            strategy_payload=strategy_payload if isinstance(strategy_payload, dict) else {},
            filters_payload=filters_payload if isinstance(filters_payload, dict) else None,
        )
        cached = _multiwindow_cache_get(cache_key=cache_key)
        if cached is not _MULTIWINDOW_CACHE_MISS:
            multiwindow_cache_hits += 1
            _emit_candidate_progress(phase="candidate.cache_hit", cached=True, kept=bool(cached))
            return dict(cached) if isinstance(cached, dict) else None

        def _cache_and_return(payload: dict | None) -> dict | None:
            nonlocal multiwindow_cache_writes
            _multiwindow_cache_set(cache_key=cache_key, payload=payload if isinstance(payload, dict) else None)
            multiwindow_cache_writes += 1
            _emit_candidate_progress(phase="candidate.done", cached=False, kept=bool(payload))
            return dict(payload) if isinstance(payload, dict) else None

        filters = _codec_filters_from_payload(filters_payload)
        strat_cfg = _codec_strategy_from_payload(strategy_payload, filters=filters)
        if bool(args.require_close_eod) and not bool(getattr(strat_cfg, "spot_close_eod", False)):
            return _cache_and_return(None)

        sig_bar_size = str(strategy_payload.get("signal_bar_size") or args.bar_size)
        sig_use_rth = (
            use_rth if strategy_payload.get("signal_use_rth") is None else bool(strategy_payload.get("signal_use_rth"))
        )

        window_plan = sorted(
            enumerate(windows),
            key=lambda row: (
                -int(_required_trades_for_window(row[1][0], row[1][1])),
                int((row[1][1] - row[1][0]).days),
                int(row[0]),
            ),
        )
        window_total = int(len(window_plan))

        per_window_staged: list[tuple[int, dict]] = []
        for window_pos, (window_idx, (wstart, wend)) in enumerate(window_plan, start=1):
            _emit_candidate_progress(
                phase="window.start",
                window_idx=int(window_pos),
                window_total=int(window_total),
                window_start=wstart.isoformat(),
                window_end=wend.isoformat(),
            )
            bundle = _codec_make_bundle(
                strategy=strat_cfg,
                start=wstart,
                end=wend,
                bar_size=sig_bar_size,
                use_rth=sig_use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )

            start_dt = datetime.combine(bundle.backtest.start, time(0, 0))
            end_dt = datetime.combine(bundle.backtest.end, time(23, 59))
            bars_sig = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bundle.backtest.bar_size,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not bars_sig:
                _mw_die_empty_bars(
                    kind="signal",
                    cache_dir=cache_dir,
                    symbol=bundle.strategy.symbol,
                    exchange=bundle.strategy.exchange,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=str(bundle.backtest.bar_size),
                    use_rth=bundle.backtest.use_rth,
                    offline=bundle.backtest.offline,
                )

            regime_bars, regime2_bars, tick_bars, exec_bars = _load_window_context_bars(
                bundle=bundle,
                start_dt=start_dt,
                end_dt=end_dt,
                load_bars_cached=load_bars_cached,
            )
            _emit_candidate_progress(
                phase="window.context_ready",
                window_idx=int(window_pos),
                window_total=int(window_total),
                window_start=wstart.isoformat(),
                window_end=wend.isoformat(),
                signal_total=int(len(bars_sig)),
                regime_total=int(len(regime_bars) if isinstance(regime_bars, list) else 0),
                regime2_total=int(len(regime2_bars) if isinstance(regime2_bars, list) else 0),
                tick_total=int(len(tick_bars) if isinstance(tick_bars, list) else 0),
                exec_total=int(len(exec_bars) if isinstance(exec_bars, list) else int(len(bars_sig))),
            )

            def _engine_progress(event: dict | None) -> None:
                if not isinstance(event, dict):
                    return
                _emit_candidate_progress(
                    event,
                    window_idx=int(window_pos),
                    window_total=int(window_total),
                    window_start=wstart.isoformat(),
                    window_end=wend.isoformat(),
                )

            summary = _run_spot_backtest_summary(
                bundle,
                bars_sig,
                _resolve_meta(bundle, data=data),
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
                tick_bars=tick_bars,
                exec_bars=exec_bars,
                progress_callback=_engine_progress,
            )
            m = _codec_metrics_from_summary(summary)
            _emit_candidate_progress(
                phase="window.done",
                window_idx=int(window_pos),
                window_total=int(window_total),
                window_start=wstart.isoformat(),
                window_end=wend.isoformat(),
                trades=int(m.get("trades") or 0),
                pnl=float(m.get("pnl") or 0.0),
                win_rate=float(m.get("win_rate") or 0.0),
            )
            if bool(args.require_positive_pnl) and float(m["pnl"]) <= 0:
                _emit_candidate_progress(
                    phase="window.rejected",
                    reason="non_positive_pnl",
                    window_idx=int(window_pos),
                    window_total=int(window_total),
                )
                return _cache_and_return(None)
            req_trades = _required_trades_for_window(wstart, wend)
            if m["trades"] < int(req_trades) or m["win_rate"] < float(args.min_win):
                _emit_candidate_progress(
                    phase="window.rejected",
                    reason="min_trade_or_win",
                    window_idx=int(window_pos),
                    window_total=int(window_total),
                )
                return _cache_and_return(None)
            per_window_staged.append(
                (
                    int(window_idx),
                    {
                    "start": wstart.isoformat(),
                    "end": wend.isoformat(),
                    **m,
                    },
                )
            )

        if not per_window_staged:
            return _cache_and_return(None)
        # Keep output semantics stable: primary window remains the original first window.
        per_window = [row for _, row in sorted(per_window_staged, key=lambda item: item[0])]
        min_pnl_dd = min(float(x["pnl_over_dd"]) for x in per_window)
        min_pnl = min(float(x["pnl"]) for x in per_window)
        min_roi_dd = min(float(x.get("roi_over_dd_pct") or 0.0) for x in per_window)
        min_roi = min(float(x.get("roi") or 0.0) for x in per_window)
        primary = per_window[0] if per_window else {}
        return _cache_and_return(
            {
            "key": _strategy_key(strategy_payload, filters=filters_payload),
            "strategy": strategy_payload,
            "filters": filters_payload,
            "seed_group_name": cand.get("group_name"),
            "full_trades": int(primary.get("trades") or 0),
            "full_win": float(primary.get("win_rate") or 0.0),
            "full_pnl": float(primary.get("pnl") or 0.0),
            "full_dd": float(primary.get("dd") or 0.0),
            "full_pnl_over_dd": float(primary.get("pnl_over_dd") or 0.0),
            "full_roi": float(primary.get("roi") or 0.0),
            "full_dd_pct": float(primary.get("dd_pct") or 0.0),
            "full_roi_over_dd_pct": float(primary.get("roi_over_dd_pct") or 0.0),
            "stability_min_pnl_dd": min_pnl_dd,
            "stability_min_pnl": min_pnl,
            "stability_min_roi_dd": min_roi_dd,
            "stability_min_roi": min_roi,
            "windows": per_window,
            }
        )

    def _evaluate_candidate_multiwindow_shard(
        *,
        load_bars_cached,
        data: IBKRHistoricalData | None,
        worker_id: int,
        workers: int,
        progress_mode: str,
    ) -> tuple[int, list[dict]]:
        out_rows: list[dict] = []
        tested = 0
        started = pytime.perf_counter()
        report_every = 10
        heartbeat_sec = 15.0
        last_report = float(started)

        workers_n = max(1, int(workers))
        worker_id_n = int(worker_id)
        progress_label = (
            f"multitimeframe worker {worker_id+1}/{workers}"
            if progress_mode == "worker"
            else "multitimeframe serial"
        )

        def _candidate_cost(cand: dict) -> float:
            strat = cand.get("strategy") if isinstance(cand.get("strategy"), dict) else {}
            sig_bar = str(strat.get("signal_bar_size") or args.bar_size)
            cost = 1.0 + (0.15 * float(len(windows)))

            regime_mode = str(strat.get("regime_mode", "ema") or "ema").strip().lower()
            regime_bar = str(strat.get("regime_bar_size") or "").strip() or sig_bar
            if regime_bar != sig_bar and (
                regime_mode == "supertrend" or bool(strat.get("regime_ema_preset"))
            ):
                cost += 0.5

            regime2_mode = str(strat.get("regime2_mode", "off") or "off").strip().lower()
            regime2_bar = str(strat.get("regime2_bar_size") or "").strip() or sig_bar
            if regime2_mode != "off" and regime2_bar != sig_bar:
                cost += 0.5

            tick_mode = str(strat.get("tick_gate_mode", "off") or "off").strip().lower()
            if tick_mode != "off":
                cost += 0.75

            exec_size = str(strat.get("spot_exec_bar_size") or "").strip()
            if exec_size and exec_size != sig_bar:
                cost += 0.75
            return float(cost)

        if workers_n == 1:
            worker_indices = list(range(len(candidates)))
        else:
            weighted = sorted(
                ((idx, _candidate_cost(cand)) for idx, cand in enumerate(candidates)),
                key=lambda row: (-float(row[1]), int(row[0])),
            )
            loads = [0.0] * workers_n
            buckets: list[list[int]] = [[] for _ in range(workers_n)]
            for idx, est_cost in weighted:
                target = min(range(workers_n), key=lambda wid: (loads[wid], wid))
                buckets[target].append(int(idx))
                loads[target] += float(est_cost)
            for bucket in buckets:
                bucket.sort()
            worker_indices = buckets[worker_id_n]

        worker_total = len(worker_indices)
        if int(worker_total) > 0:
            print(
                _progress_line(
                    label=str(progress_label),
                    tested=0,
                    total=int(worker_total),
                    kept=0,
                    started_at=started,
                    rate_unit="cands/s",
                ),
                flush=True,
            )
        for cand_idx in worker_indices:
            now_before = float(pytime.perf_counter())
            if int(tested) > 0 and (now_before - float(last_report)) >= float(heartbeat_sec):
                print(
                    _progress_line(
                        label=str(progress_label),
                        tested=int(tested),
                        total=int(worker_total),
                        kept=len(out_rows),
                        started_at=started,
                        rate_unit="cands/s",
                    ),
                    flush=True,
                )
                last_report = float(now_before)
            cand = candidates[int(cand_idx)]
            tested += 1
            candidate_started = float(pytime.perf_counter())
            candidate_stop = threading.Event()
            candidate_hb: threading.Thread | None = None
            candidate_phase_state: dict[str, object] = {}
            candidate_phase_lock = threading.Lock()

            def _update_candidate_phase(event: dict | None) -> None:
                if not isinstance(event, dict):
                    return
                with candidate_phase_lock:
                    for key in (
                        "phase",
                        "path",
                        "window_idx",
                        "window_total",
                        "window_start",
                        "window_end",
                        "signal_total",
                        "regime_total",
                        "regime2_total",
                        "tick_total",
                        "exec_total",
                        "sig_idx",
                        "sig_total",
                        "exec_idx",
                        "open_count",
                        "trades",
                        "cached",
                        "kept",
                    ):
                        if key in event:
                            candidate_phase_state[str(key)] = event.get(key)

            def _candidate_heartbeat() -> None:
                cadence = max(5.0, float(heartbeat_sec))
                while not candidate_stop.wait(cadence):
                    line = _progress_line(
                        label=str(progress_label),
                        tested=int(tested),
                        total=int(worker_total),
                        kept=len(out_rows),
                        started_at=started,
                        rate_unit="cands/s",
                    )
                    line += f" inflight_candidate={max(0.0, float(pytime.perf_counter()) - float(candidate_started)):0.1f}s"
                    with candidate_phase_lock:
                        phase_snap = dict(candidate_phase_state)
                    phase_name = str(phase_snap.get("phase") or "").strip()
                    if phase_name:
                        line += f" stage={phase_name}"
                    path_name = str(phase_snap.get("path") or "").strip()
                    if path_name:
                        line += f" path={path_name}"
                    window_idx = phase_snap.get("window_idx")
                    window_total = phase_snap.get("window_total")
                    if isinstance(window_idx, int) and isinstance(window_total, int) and int(window_total) > 0:
                        line += f" window={int(window_idx)}/{int(window_total)}"
                    exec_idx = phase_snap.get("exec_idx")
                    exec_total = phase_snap.get("exec_total")
                    if isinstance(exec_idx, int) and isinstance(exec_total, int) and int(exec_total) > 0:
                        line += f" exec={int(exec_idx)}/{int(exec_total)}"
                    sig_idx = phase_snap.get("sig_idx")
                    sig_total = phase_snap.get("sig_total")
                    if isinstance(sig_idx, int) and isinstance(sig_total, int) and int(sig_total) > 0:
                        line += f" sig={int(sig_idx)}/{int(sig_total)}"
                    trades_live = phase_snap.get("trades")
                    if isinstance(trades_live, int):
                        line += f" trades={int(trades_live)}"
                    print(line, flush=True)

            if float(heartbeat_sec) > 0.0:
                candidate_hb = threading.Thread(target=_candidate_heartbeat, daemon=True)
                candidate_hb.start()
            try:
                row = _evaluate_candidate_multiwindow(
                    cand,
                    load_bars_cached=load_bars_cached,
                    data=data,
                    progress_callback=_update_candidate_phase,
                )
            finally:
                candidate_stop.set()
                if candidate_hb is not None:
                    try:
                        candidate_hb.join(timeout=1.0)
                    except Exception:
                        pass
            if row is not None:
                out_rows.append(row)

            now_after = float(pytime.perf_counter())
            hit_report_every = int(report_every) > 0 and (int(tested) % int(report_every) == 0)
            hit_total = int(tested) >= int(worker_total)
            hit_heartbeat = (now_after - float(last_report)) >= float(heartbeat_sec)
            if hit_report_every or hit_total or hit_heartbeat:
                line = _progress_line(
                    label=str(progress_label),
                    tested=int(tested),
                    total=int(worker_total),
                    kept=len(out_rows),
                    started_at=started,
                    rate_unit="cands/s",
                )
                print(line, flush=True)
                last_report = float(now_after)
        return tested, out_rows

    def _emit_multitimeframe_results(*, out_rows: list[dict], tested_total: int | None = None, workers: int | None = None) -> None:
        out_rows = sorted(out_rows, key=_score_key, reverse=True)
        print("")
        print(f"Multiwindow results: {len(out_rows)} candidates passed filters.")
        print(f"- symbol={symbol} bar={args.bar_size} rth={use_rth} offline={offline}")
        print(f"- windows={', '.join([f'{a.isoformat()}→{b.isoformat()}' for a,b in windows])}")
        extra = f" min_trades_per_year={float(min_trades_per_year):g}" if min_trades_per_year is not None else ""
        print(f"- min_trades={int(args.min_trades)} min_win={float(args.min_win):0.2f}{extra}")
        if tested_total is not None and workers is not None:
            print(f"- workers={int(workers)} tested_total={int(tested_total)}")
        if bool(multiwindow_cache_enabled):
            print(
                f"- eval_cache={multiwindow_cache_path} hits={int(multiwindow_cache_hits)} writes={int(multiwindow_cache_writes)}",
                flush=True,
            )
        print("")

        show = min(20, len(out_rows))
        for rank, item in enumerate(out_rows[:show], start=1):
            st = item["strategy"]
            print(
                f"{rank:2d}. stability(min roi/dd)={item.get('stability_min_roi_dd', 0.0):.2f} "
                f"full roi/dd={item.get('full_roi_over_dd_pct', 0.0):.2f} "
                f"roi={item.get('full_roi', 0.0)*100:.1f}% dd%={item.get('full_dd_pct', 0.0)*100:.1f}% "
                f"pnl={item['full_pnl']:.1f} "
                f"win={item['full_win']*100:.1f}% tr={item['full_trades']} "
                f"ema={st.get('ema_preset')} {st.get('ema_entry_mode')} "
                f"regime={st.get('regime_mode')} rbar={st.get('regime_bar_size')}"
            )

        if int(args.write_top or 0) <= 0:
            return
        top_k = max(1, int(args.write_top))
        now = utc_now_iso_z()
        groups_out: list[dict] = []
        for idx, item in enumerate(out_rows[:top_k], start=1):
            strategy = item["strategy"]
            filters_payload = item.get("filters")
            key = _strategy_key(strategy, filters=filters_payload)
            metrics = {
                "pnl": float(item.get("full_pnl") or 0.0),
                "roi": float(item.get("full_roi") or 0.0),
                "win_rate": float(item.get("full_win") or 0.0),
                "trades": int(item.get("full_trades") or 0),
                "max_drawdown": float(item.get("full_dd") or 0.0),
                "max_drawdown_pct": float(item.get("full_dd_pct") or 0.0),
                "pnl_over_dd": float(item.get("full_pnl_over_dd") or 0.0),
                "roi_over_dd_pct": float(item.get("full_roi_over_dd_pct") or 0.0),
            }
            groups_out.append(
                {
                    "name": f"Spot ({symbol}) KINGMAKER #{idx:02d} roi/dd={metrics['roi_over_dd_pct']:.2f} "
                    f"roi={metrics['roi']*100:.1f}% dd%={metrics['max_drawdown_pct']*100:.1f}% "
                    f"win={metrics['win_rate']*100:.1f}% tr={metrics['trades']} pnl={metrics['pnl']:.1f}",
                    "filters": filters_payload,
                    "entries": [{"symbol": symbol, "metrics": metrics, "strategy": strategy}],
                    "_eval": {
                        "stability_min_pnl_dd": float(item.get("stability_min_pnl_dd") or 0.0),
                        "stability_min_pnl": float(item.get("stability_min_pnl") or 0.0),
                        "stability_min_roi_dd": float(item.get("stability_min_roi_dd") or 0.0),
                        "stability_min_roi": float(item.get("stability_min_roi") or 0.0),
                        "windows": item.get("windows") or [],
                    },
                    "_key": key,
                }
            )
        out_payload = {
            "name": "multitimeframe_top",
            "generated_at": now,
            "source": str(milestones_path),
            "windows": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in windows],
            "groups": groups_out,
        }
        out_path = Path(args.out)
        write_json(out_path, out_payload, sort_keys=False)
        print(f"\nWrote {out_path} (top={top_k}).")

    def _collect_multitimeframe_rows_from_payloads(*, payloads: dict[int, dict]) -> tuple[int, list[dict]]:
        out_rows: list[dict] = []

        def _decode_row(rec: dict) -> dict | None:
            return dict(rec) if isinstance(rec, dict) else None

        def _row_key(row: dict) -> str:
            strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
            filters_payload = row.get("filters") if isinstance(row.get("filters"), dict) else None
            return _strategy_key(strategy, filters=filters_payload)

        tested_total = _collect_parallel_payload_records(
            payloads=payloads,
            records_key="rows",
            tested_key="tested",
            decode_record=_decode_row,
            on_record=lambda row: out_rows.append(dict(row)),
            dedupe_key=_row_key,
        )
        return int(tested_total), out_rows

    if args.multitimeframe_worker is not None:
        if not offline:
            raise SystemExit("multitimeframe worker mode requires --offline (avoid parallel IBKR sessions).")
        out_path_raw = str(args.multitimeframe_out or "").strip()
        if not out_path_raw:
            raise SystemExit("--multitimeframe-out is required for multitimeframe worker mode.")
        out_path = Path(out_path_raw)

        worker_id, workers = _parse_worker_shard(
            args.multitimeframe_worker,
            args.multitimeframe_workers,
            label="multitimeframe",
        )

        _mw_preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
            cache_policy=str(args.cache_policy),
        )

        data = IBKRHistoricalData()
        _load_bars_cached = _make_bars_loader(data)

        tested, out_rows = _evaluate_candidate_multiwindow_shard(
            load_bars_cached=_load_bars_cached,
            data=data,
            worker_id=int(worker_id),
            workers=int(workers),
            progress_mode="worker",
        )

        out_payload = {"tested": tested, "kept": len(out_rows), "rows": out_rows}
        write_json(out_path, out_payload, sort_keys=False)
        print(f"multitimeframe worker done tested={tested} kept={len(out_rows)} out={out_path}", flush=True)
        return

    if jobs_eff > 1:
        if not offline:
            raise SystemExit("--jobs>1 for multitimeframe requires --offline (avoid parallel IBKR sessions).")

        _mw_preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
            cache_policy=str(args.cache_policy),
        )

        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags_with_values=("--jobs", "--multitimeframe-worker", "--multitimeframe-workers", "--multitimeframe-out"),
        )

        jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label="multitimeframe",
            jobs=int(jobs_eff),
            total=len(candidates),
            default_jobs=int(jobs_eff),
            offline=bool(offline),
            offline_error="--jobs>1 for multitimeframe requires --offline (avoid parallel IBKR sessions).",
            tmp_prefix="tradebot_multitimeframe_",
            worker_tag="mt",
            out_prefix="multitimeframe_out",
            build_cmd=lambda worker_id, workers_n, out_path: [
                sys.executable,
                "-u",
                "-m",
                "tradebot.backtest",
                "spot_multitimeframe",
                *base_cli,
                "--jobs",
                "1",
                "--multitimeframe-worker",
                str(worker_id),
                "--multitimeframe-workers",
                str(workers_n),
                "--multitimeframe-out",
                str(out_path),
            ],
            capture_error="Failed to capture multitimeframe worker stdout.",
            failure_label="multitimeframe worker",
            missing_label="multitimeframe",
            invalid_label="multitimeframe",
            status_heartbeat_sec=10.0,
        )

        tested_total, out_rows = _collect_multitimeframe_rows_from_payloads(payloads=payloads)

        _emit_multitimeframe_results(out_rows=out_rows, tested_total=tested_total, workers=jobs_eff)

        return

    data = IBKRHistoricalData()
    _load_bars_cached = _make_bars_loader(data)

    if not offline:
        try:
            data.connect()
        except Exception as exc:
            raise SystemExit(
                "IBKR API connection failed. Start IB Gateway / TWS (or run with --offline after prefetching cached bars)."
            ) from exc

    if offline:
        _mw_preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
            cache_policy=str(args.cache_policy),
        )

    _tested_serial, out_rows = _evaluate_candidate_multiwindow_shard(
        load_bars_cached=_load_bars_cached,
        data=data,
        worker_id=0,
        workers=1,
        progress_mode="serial",
    )

    if not offline:
        data.disconnect()

    _emit_multitimeframe_results(out_rows=out_rows)

multitimeframe_main = spot_multitimeframe_main


if __name__ == "__main__":
    spot_multitimeframe_main()
