"""Spot multi-window stability eval (kingmaker) entrypoint.

This module owns evaluation, cache, sharding, and data orchestration for the
`spot_multitimeframe` workflow.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import threading
import time as pytime
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .cli_utils import parse_date as _parse_date, parse_window as _parse_window
from .cache import cache_data_revision
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
from .spot_context import (
    SpotBarRequirement,
    SpotContextBars,
    load_spot_context_bars,
    spot_signal_warmup_days_from_strategy,
)
from .data import ContractMeta, IBKRHistoricalData
from .engine import _run_spot_backtest_summary, _spot_multiplier
from .sweep_parallel import (
    _parse_worker_shard,
    _progress_line,
    _run_parallel_stage_kernel,
    _strip_flags,
)
from .sweeps import normalize_jobs, write_json
from ..spot.champions import load_current_champion_groups
from ..research.multiwindow import (
    MultiwindowReport,
    candidate_shortlist,
    collect_multiwindow_rows,
    emit_multiwindow_results,
    multiwindow_cache_key,
    parse_multiwindow_args,
    strategy_key,
)
from ..chart_data.cache import series_cache_service
from ..time_utils import now_et as _now_et

_SERIES_CACHE = series_cache_service()
_SWEEP_MULTIWINDOW_BARS_NAMESPACE = "spot.sweeps.multiwindow.bars"
# Bump whenever evaluation semantics change so stale cached rows don't mask runtime fixes.
_MULTIWINDOW_CACHE_ENGINE_VERSION = "spot_multiwindow_v15"


def spot_multitimeframe_main() -> None:
    args = parse_multiwindow_args()
    try:
        min_trades_per_year = float(args.min_trades_per_year) if args.min_trades_per_year is not None else None
    except (TypeError, ValueError):
        min_trades_per_year = None
    if min_trades_per_year is not None and min_trades_per_year < 0:
        raise SystemExit("--min-trades-per-year must be >= 0")

    try:
        jobs = int(args.jobs) if args.jobs is not None else 0
    except (TypeError, ValueError):
        jobs = 0
    jobs_eff = normalize_jobs(jobs)

    milestones_path = Path(args.milestones)
    payload = json.loads(milestones_path.read_text())
    symbol = str(args.symbol).strip().upper()
    bar_size = str(args.bar_size).strip().lower()
    use_rth = bool(args.use_rth)
    candidates = candidate_shortlist(
        payload,
        symbol=symbol,
        bar_size=bar_size,
        use_rth=use_rth,
        limit=int(args.top),
        track=str(args.track),
    )

    if not candidates:
        raise SystemExit(f"No candidates found for {symbol} bar={bar_size} rth={use_rth} in {milestones_path}")

    requested_track = str(args.track).strip().upper()
    candidate_tracks = {str(cand.get("track") or "").strip().upper() for cand in candidates}
    candidate_tracks.discard("")
    report_track = (
        requested_track
        if requested_track != "AUTO"
        else next(iter(candidate_tracks))
        if len(candidate_tracks) == 1
        else None
    )
    incumbent_source: str | None = None
    incumbent_windows: tuple[dict, ...] = ()
    if report_track:
        incumbent_groups, incumbent_warnings = load_current_champion_groups(
            symbols=(symbol,),
            tracks=(report_track,),
        )
        for warning in incumbent_warnings:
            print(f"incumbent warning: {warning}", flush=True)
        if incumbent_groups:
            incumbent = incumbent_groups[0]
            evaluation = incumbent.get("_eval") if isinstance(incumbent.get("_eval"), dict) else {}
            raw_windows = evaluation.get("windows")
            incumbent_windows = tuple(
                row for row in raw_windows if isinstance(row, dict)
            ) if isinstance(raw_windows, list) else ()
            if incumbent_windows:
                incumbent_source = str(incumbent.get("_source") or "") or None
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
    if not windows and incumbent_windows:
        windows = [
            (_parse_date(str(row["start"])), _parse_date(str(row["end"])))
            for row in incumbent_windows
        ]
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
    data_revision = cache_data_revision(cache_dir)
    multiwindow_cache_path = cache_dir / "spot_multiwindow_eval_cache.sqlite3"
    multiwindow_cache_conn: sqlite3.Connection | None = None
    multiwindow_cache_enabled = True
    multiwindow_cache_lock = threading.Lock()
    multiwindow_cache_hits = 0
    multiwindow_cache_writes = 0
    _MULTIWINDOW_CACHE_MISS = object()
    report = MultiwindowReport(
        symbol=symbol,
        bar_size=str(args.bar_size),
        use_rth=use_rth,
        offline=offline,
        windows=tuple(windows),
        min_trades=int(args.min_trades),
        min_win=float(args.min_win),
        min_trades_per_year=min_trades_per_year,
        milestones_path=milestones_path,
        write_top=int(args.write_top or 0),
        out_path=Path(args.out),
        cache_path=multiwindow_cache_path,
        track=report_track,
        incumbent_source=incumbent_source,
        incumbent_windows=incumbent_windows,
    )

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

    def _multiwindow_cache_key(*, strategy_payload: dict, filters_payload: dict | None) -> str:
        return multiwindow_cache_key(
            engine_version=_MULTIWINDOW_CACHE_ENGINE_VERSION,
            data_revision=data_revision,
            strategy=strategy_payload,
            filters=filters_payload,
            windows=tuple(windows),
            min_trades=int(args.min_trades),
            min_trades_per_year=min_trades_per_year,
            min_win=float(args.min_win),
            require_close_eod=bool(args.require_close_eod),
            require_positive_pnl=bool(args.require_positive_pnl),
            offline=offline,
        )

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
        signal_bars: list,
        start_dt: datetime,
        end_dt: datetime,
        load_bars_cached,
    ) -> SpotContextBars:
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
            signal_bars=signal_bars,
            default_symbol=str(bundle.strategy.symbol),
            default_exchange=bundle.strategy.exchange,
            default_signal_bar_size=str(bundle.backtest.bar_size),
            default_signal_use_rth=bool(bundle.backtest.use_rth),
            start_dt=start_dt,
            end_dt=end_dt,
            load_requirement=_load_requirement,
            on_missing=_on_missing,
        )
        return context

    def _evaluate_candidate_multiwindow(
        cand: dict,
        *,
        load_bars_cached,
        data: IBKRHistoricalData | None,
        progress_callback=None,
    ) -> dict | None:
        nonlocal data_revision, multiwindow_cache_hits, multiwindow_cache_writes
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
            nonlocal data_revision, multiwindow_cache_writes
            final_cache_key = cache_key
            if not offline:
                data_revision = cache_data_revision(cache_dir)
                final_cache_key = _multiwindow_cache_key(
                    strategy_payload=strategy_payload if isinstance(strategy_payload, dict) else {},
                    filters_payload=filters_payload if isinstance(filters_payload, dict) else None,
                )
            _multiwindow_cache_set(
                cache_key=final_cache_key,
                payload=payload if isinstance(payload, dict) else None,
            )
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
            signal_start_dt = start_dt - timedelta(
                days=max(
                    0,
                    int(
                        spot_signal_warmup_days_from_strategy(
                            strategy=bundle.strategy,
                            default_signal_bar_size=str(bundle.backtest.bar_size),
                            default_signal_use_rth=bool(bundle.backtest.use_rth),
                        )
                    ),
                )
            )
            bars_sig = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=signal_start_dt,
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

            context = _load_window_context_bars(
                bundle=bundle,
                signal_bars=bars_sig,
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
                regime_total=int(len(context.regime_bars) if isinstance(context.regime_bars, list) else 0),
                regime2_total=int(len(context.regime2_bars) if isinstance(context.regime2_bars, list) else 0),
                regime2_bear_hard_total=int(
                    len(context.regime2_bear_hard_bars)
                    if isinstance(context.regime2_bear_hard_bars, list)
                    else 0
                ),
                tick_total=int(len(context.tick_bars) if isinstance(context.tick_bars, list) else 0),
                exec_total=int(
                    len(context.exec_bars)
                    if isinstance(context.exec_bars, list)
                    else len(context.signal_bars)
                ),
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
                context.signal_bars,
                _resolve_meta(bundle, data=data),
                regime_bars=context.regime_bars,
                regime2_bars=context.regime2_bars,
                regime2_bear_hard_bars=context.regime2_bear_hard_bars,
                tick_bars=context.tick_bars,
                exec_bars=context.exec_bars,
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
        # The first requested window is the primary comparison; all windows define stability.
        per_window = [row for _, row in sorted(per_window_staged, key=lambda item: item[0])]
        min_pnl_dd = min(float(x["pnl_over_dd"]) for x in per_window)
        min_pnl = min(float(x["pnl"]) for x in per_window)
        min_roi_dd = min(float(x.get("roi_over_dd_pct") or 0.0) for x in per_window)
        min_roi = min(float(x.get("roi") or 0.0) for x in per_window)
        primary = per_window[0] if per_window else {}
        return _cache_and_return(
            {
            "key": strategy_key(strategy_payload, filters=filters_payload),
            "strategy": strategy_payload,
            "filters": filters_payload,
            "seed_group_name": cand.get("group_name"),
            "primary": primary,
            "stability": {
                "min_pnl_over_dd": min_pnl_dd,
                "min_pnl": min_pnl,
                "min_roi_over_dd": min_roi_dd,
                "min_roi": min_roi,
            },
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
        data_revision = cache_data_revision(cache_dir)

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
        data_revision = cache_data_revision(cache_dir)

        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags_with_values=("--jobs", "--multitimeframe-worker", "--multitimeframe-workers", "--multitimeframe-out"),
        )

        jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label="multitimeframe",
            jobs=int(jobs_eff),
            total=len(candidates),
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

        tested_total, out_rows = collect_multiwindow_rows(payloads=payloads)

        emit_multiwindow_results(
            report=report,
            out_rows=out_rows,
            tested_total=tested_total,
            workers=jobs_eff,
            cache_enabled=multiwindow_cache_enabled,
            cache_hits=multiwindow_cache_hits,
            cache_writes=multiwindow_cache_writes,
        )

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
        data_revision = cache_data_revision(cache_dir)

    _tested_serial, out_rows = _evaluate_candidate_multiwindow_shard(
        load_bars_cached=_load_bars_cached,
        data=data,
        worker_id=0,
        workers=1,
        progress_mode="serial",
    )

    if not offline:
        data.disconnect()

    emit_multiwindow_results(
        report=report,
        out_rows=out_rows,
        cache_enabled=multiwindow_cache_enabled,
        cache_hits=multiwindow_cache_hits,
        cache_writes=multiwindow_cache_writes,
    )

multitimeframe_main = spot_multitimeframe_main


if __name__ == "__main__":
    spot_multitimeframe_main()
