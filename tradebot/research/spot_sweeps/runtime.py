"""Composition root for one canonical spot research run."""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time as pytime
from datetime import datetime, time
from pathlib import Path
from ...backtest.cli_utils import parse_date as _parse_date
from ...backtest.config import (
    ConfigBundle,
)
from ...backtest.data import ContractMeta, IBKRHistoricalData
from ...backtest.sweeps import normalize_jobs
from ...chart_data.series import BarSeriesSignature
from ...time_utils import now_et as _now_et
from ...signals import parse_bar_size
from .catalog import (
    _axis_mode_plan,
    build_axis_registry,
    _write_spot_sweep_coverage_map,
)
from .cli import parse_spot_sweep_args, resolve_run_min_trades as _resolve_run_min_trades
from .milestones import (
    _collect_milestone_items_from_rows,
    _merge_and_write_milestones,
    _milestone_entry_for,
    load_current_champion_milestones,
)
from .support import (
    _registry_float,
    _runtime_policy,
)

from .store_cache import SweepCacheStore
from .store_status import SweepStatusStore
from .store_frontiers import SweepFrontierStore
from .store_planner import SweepPlannerStore
from .market import SweepMarketData
from .evaluation import SweepEvaluation
from .planning import SweepPlanning
from .workers import SweepWorkers
from .stages import SweepStages
from .parallel_runtime import SweepParallelRuntime
from .axes_core import SweepCoreAxes
from .axes_hf import SweepHighFrequencyAxes
from .axes_regime import SweepRegimeAxes
from .combo import SweepCartesian


class SpotSweepRuntime(
    SweepCacheStore,
    SweepStatusStore,
    SweepFrontierStore,
    SweepPlannerStore,
    SweepMarketData,
    SweepEvaluation,
    SweepPlanning,
    SweepWorkers,
    SweepStages,
    SweepParallelRuntime,
    SweepCoreAxes,
    SweepHighFrequencyAxes,
    SweepRegimeAxes,
    SweepCartesian,
):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        try:
            jobs_raw = int(self.args.jobs) if self.args.jobs is not None else 0
        except (TypeError, ValueError):
            jobs_raw = 0
        self.jobs = normalize_jobs(jobs_raw)

        self.symbol = str(self.args.symbol).strip().upper()
        self.start = _parse_date(self.args.start)
        self.end = _parse_date(self.args.end)
        self.use_rth = bool(self.args.use_rth)
        self.offline = bool(self.args.offline)
        self.cache_policy = str(self.args.cache_policy).strip().lower() or "strict"
        self.cache_dir = Path(self.args.cache_dir)
        self.start_dt = datetime.combine(self.start, time(0, 0))
        self.end_dt = datetime.combine(self.end, time(23, 59))
        self.signal_bar_size = str(self.args.bar_size).strip() or "1 hour"
        self.spot_exec_bar_size = str(self.args.spot_exec_bar_size).strip() if self.args.spot_exec_bar_size else None
        if self.spot_exec_bar_size and parse_bar_size(self.spot_exec_bar_size) is None:
            raise SystemExit(f"Invalid --spot-exec-bar-size: {self.spot_exec_bar_size!r}")
        self.close_eod = bool(self.args.close_eod)
        self.long_only = bool(self.args.long_only)
        self.realism2 = bool(self.args.realism2)
        self.spot_spread = float(self.args.spot_spread) if self.args.spot_spread is not None else (0.01 if self.realism2 else 0.0)
        self.spot_commission = float(self.args.spot_commission) if self.args.spot_commission is not None else (0.005 if self.realism2 else 0.0)
        self.spot_commission_min = float(self.args.spot_commission_min) if self.args.spot_commission_min is not None else (1.0 if self.realism2 else 0.0)
        self.spot_slippage = float(self.args.spot_slippage) if self.args.spot_slippage is not None else 0.0

        self.sizing_mode_arg_explicit = self.args.spot_sizing_mode is not None
        self.sizing_mode = str(self.args.spot_sizing_mode).strip().lower() if self.sizing_mode_arg_explicit else ("risk_pct" if self.realism2 else "fixed")
        if self.sizing_mode not in ("fixed", "notional_pct", "risk_pct"):
            self.sizing_mode = "fixed"
        self.spot_risk_pct_arg_explicit = self.args.spot_risk_pct is not None
        self.spot_risk_pct = float(self.args.spot_risk_pct) if self.spot_risk_pct_arg_explicit else (0.01 if self.realism2 else 0.0)
        self.spot_notional_pct_arg_explicit = self.args.spot_notional_pct is not None
        self.spot_notional_pct = float(self.args.spot_notional_pct) if self.spot_notional_pct_arg_explicit else 0.0
        self.spot_max_notional_pct_arg_explicit = self.args.spot_max_notional_pct is not None
        self.spot_max_notional_pct = float(self.args.spot_max_notional_pct) if self.spot_max_notional_pct_arg_explicit else (0.50 if self.realism2 else 1.0)
        self.spot_min_qty_arg_explicit = self.args.spot_min_qty is not None
        self.spot_max_qty_arg_explicit = self.args.spot_max_qty is not None
        self.spot_min_qty = int(self.args.spot_min_qty) if self.spot_min_qty_arg_explicit else 1
        self.spot_max_qty = int(self.args.spot_max_qty) if self.spot_max_qty_arg_explicit else 0

        self.run_min_trades = _resolve_run_min_trades(self.args)
        if bool(self.args.write_milestones):
            self.run_min_trades = min(self.run_min_trades, int(self.args.milestone_min_trades))
        self.data = IBKRHistoricalData()

        if self.offline:
            is_future = self.symbol in ("MNQ", "MBT")
            exchange = "CME" if is_future else "SMART"
            multiplier = 1.0
            if is_future:
                multiplier = {"MNQ": 2.0, "MBT": 0.1}.get(self.symbol, 1.0)
            self.meta = ContractMeta(
                symbol=self.symbol,
                exchange=exchange,
                multiplier=multiplier,
                min_tick=0.01,
            )
        else:
            try:
                _, self.meta = self.data.resolve_contract(self.symbol, exchange=None)
            except Exception as exc:
                raise SystemExit("IBKR API connection failed. Start IB Gateway / TWS (or run with --offline after prefetching cached bars).") from exc

        self.milestones: dict | None = None
        self.champion_track: str | None = None
        base_name = str(self.args.base).strip().lower()
        if self.args.seed_milestones and base_name in ("champion", "champion_pnl"):
            seed_path = Path(str(self.args.seed_milestones))
            try:
                seed_payload = json.loads(seed_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Unable to load --seed-milestones {seed_path}: {exc}") from exc
            has_match = _milestone_entry_for(
                seed_payload if isinstance(seed_payload, dict) else None,
                symbol=self.symbol,
                signal_bar_size=str(self.signal_bar_size),
                use_rth=self.use_rth,
                sort_by="pnl" if base_name == "champion_pnl" else "pnl_dd",
                prefer_realism=self.realism2,
            )
            if has_match is None:
                raise SystemExit(
                    f"Seed milestones have no matching champion for {self.symbol} "
                    f"bar={self.signal_bar_size!r} rth={self.use_rth}"
                )
            self.milestones = seed_payload
            self.champion_track = "SEED"
        elif base_name in ("champion", "champion_pnl"):
            try:
                self.milestones, self.champion_track, warnings = (
                    load_current_champion_milestones(
                        symbol=self.symbol,
                        signal_bar_size=str(self.signal_bar_size),
                        use_rth=self.use_rth,
                        track=str(self.args.track),
                        prefer_realism=self.realism2,
                    )
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            for warning in warnings:
                print(f"champion source warning: {warning}", flush=True)
            print(
                f"champion source: symbol={self.symbol} track={self.champion_track} "
                f"bar={self.signal_bar_size} rth={self.use_rth}",
                flush=True,
            )

        self.run_calls_total = 0
        self.run_cfg_cache: dict[tuple, dict | None] = {}
        self.run_cfg_cache_hits = 0
        self.run_cfg_fingerprint_hits = 0
        self.run_cfg_persistent_hits = 0
        self.run_cfg_persistent_writes = 0
        self.run_cfg_dim_index_writes = 0
        self.worker_plan_cache_hits = 0
        self.worker_plan_cache_writes = 0
        self.stage_cell_status_reads = 0
        self.stage_cell_status_writes = 0
        self.cartesian_manifest_reads = 0
        self.cartesian_manifest_writes = 0
        self.cartesian_manifest_hits = 0
        self.cartesian_rank_manifest_reads = 0
        self.cartesian_rank_manifest_writes = 0
        self.cartesian_rank_manifest_hits = 0
        self.cartesian_rank_manifest_compactions = 0
        self.cartesian_rank_manifest_pending_ttl_prunes = 0
        self.stage_unresolved_summary_reads = 0
        self.stage_unresolved_summary_writes = 0
        self.stage_unresolved_summary_hits = 0
        self.rank_dominance_stamp_reads = 0
        self.rank_dominance_stamp_writes = 0
        self.rank_dominance_stamp_hits = 0
        self.rank_dominance_manifest_applies = 0
        self.rank_dominance_stamp_compactions = 0
        self.rank_dominance_stamp_ttl_prunes = 0
        self.rank_bin_runtime_reads = 0
        self.rank_bin_runtime_writes = 0
        self.stage_frontier_reads = 0
        self.stage_frontier_writes = 0
        self.stage_frontier_hits = 0
        self.dimension_utility_reads = 0
        self.dimension_utility_writes = 0
        self.dimension_utility_hint_hits = 0
        self.dimension_upper_bound_reads = 0
        self.dimension_upper_bound_writes = 0
        self.dimension_upper_bound_deferred = 0
        self.planner_heartbeat_reads = 0
        self.planner_heartbeat_writes = 0
        self.planner_heartbeat_stale_candidates = 0
        self.run_cfg_fingerprint_cache: dict[
            str,
            tuple[
                tuple[tuple[str, BarSeriesSignature], ...],
                dict | None,
            ],
        ] = {}
        self.run_cfg_axis_fp_cache: dict[str, str] = {}
        self.run_cfg_dim_index_seen: set[str] = set()
        self.run_cfg_dim_index_loaded: dict[str, float] = {}
        self.run_cfg_dim_index_loaded_once = False
        self.run_cfg_window_sig_cache: dict[
            tuple[tuple[str, BarSeriesSignature], ...],
            str,
        ] = {}
        self.cartesian_rank_manifest_compact_seen: dict[tuple[str, str], float] = {}
        self.rank_dominance_stamp_compact_seen: dict[tuple[str, str], float] = {}
        self.rank_dominance_manifest_applied_seen: set[tuple[str, str]] = set()
        self._RUN_CFG_CACHE_MISS = object()
        self._RUN_CFG_CACHE_UNSET = object()
        self.run_cfg_persistent_payload_cache: dict[str, dict | None | object] = {}
        self.run_cfg_persistent_path = self.cache_dir / "spot_sweeps_run_cfg_cache.sqlite3"
        self.run_cfg_persistent_conn: sqlite3.Connection | None = None
        self.run_cfg_persistent_enabled = True
        self.run_cfg_persistent_lock = threading.Lock()
        self.run_cfg_persistent_pending: dict[str, tuple[str, dict | None]] = {}
        self.run_cfg_persistent_last_flush_ts = float(pytime.perf_counter())
        run_cfg_persistent_cfg = _runtime_policy("run_cfg_persistent")
        self.run_cfg_persistent_batch_write_size = max(
            1,
            int(_registry_float(run_cfg_persistent_cfg.get("batch_write_size"), 256.0)),
        )
        self.run_cfg_persistent_batch_write_interval_sec = max(
            0.0,
            float(_registry_float(run_cfg_persistent_cfg.get("batch_write_interval_sec"), 2.0)),
        )
        run_cfg_stage_worker_mode = bool(getattr(self.args, "combo_full_cartesian_stage", None) or getattr(self.args, "cfg_stage", None))
        self.run_cfg_persistent_ram_first_enabled = bool(
            _registry_float(run_cfg_persistent_cfg.get("ram_first_worker"), 1.0) > 0.0 and bool(self.offline) and bool(run_cfg_stage_worker_mode)
        )
        self._STAGE_CELL_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated"))
        self._CARTESIAN_CELL_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated", "dominated"))
        self._CARTESIAN_RANK_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated", "dominated"))

        regime_bars_1d = self._bars_cached("1 day")
        if not regime_bars_1d:
            raise SystemExit("No 1 day regime bars returned (IBKR).")

        self.axis_progress_state: dict[str, object] = {
            "active": False,
            "axis_key": "",
            "label": "",
            "start_calls": 0,
            "tested": 0,
            "kept": 0,
            "total": None,
            "started_at": 0.0,
            "last_report": 0.0,
            "last_reported_tested": 0,
            "report_every": 0,
            "heartbeat_sec": 20.0,
            "suppress": False,
        }
        self.axis_progress_history_path = self.cache_dir / "spot_axis_total_hints.json"
        self.axis_progress_history: dict[str, int] = {}
        try:
            raw_hist = json.loads(self.axis_progress_history_path.read_text())
            if isinstance(raw_hist, dict):
                for key, val in raw_hist.items():
                    try:
                        iv = int(val)
                    except (TypeError, ValueError):
                        continue
                    if iv > 0:
                        self.axis_progress_history[str(key).strip().lower()] = int(iv)
        except Exception:
            self.axis_progress_history = {}

        self.milestone_rows: list[tuple[ConfigBundle, dict, str]] = []
        self.milestones_written = False

        # Populated once after sweep functions are declared; used by axis dispatchers.
        self.axis_registry: dict[str, object] = {}

    def run(self) -> None:
        axis = str(self.args.axis).strip().lower()
        print(
            f"{self.symbol} spot evolve sweep ({self.start.isoformat()} -> {self.end.isoformat()}, use_rth={self.use_rth}, "
            f"bar_size={self.signal_bar_size}, offline={self.offline}, cache_policy={self.cache_policy}, base={self.args.base}, axis={axis}, "
            f"jobs={self.jobs}, "
            f"long_only={self.long_only} realism={'v2' if self.realism2 else 'off'} "
            f"spread={self.spot_spread:g} comm={self.spot_commission:g} comm_min={self.spot_commission_min:g} "
            f"slip={self.spot_slippage:g} sizing={self.sizing_mode} risk={self.spot_risk_pct:g} max_notional={self.spot_max_notional_pct:g})"
        )

        # Generic config workers are transport, not axes; dispatch them before
        # the requested axis so every research pipeline reuses one worker path.
        if self.args.cfg_stage:
            try:
                cfg_stage_payload = json.loads(Path(str(self.args.cfg_stage)).read_text())
                cfg_stage_axis = str(cfg_stage_payload.get("axis_tag") or "cfg_stage")
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                raise SystemExit(f"Invalid cfg_stage payload: {self.args.cfg_stage}") from exc
            self._run_cfg_pairs_grid(axis_tag=cfg_stage_axis, cfg_pairs=[], rows=[])
            return

        if axis == "all" and self.jobs > 1:
            axis_plan = _axis_mode_plan(mode="axis_all")
            self._run_axis_plan_parallel_if_requested(
                axis_plan=list(axis_plan),
                jobs_req=int(self.jobs),
                label="axis=all parallel",
                tmp_prefix="tradebot_axis_all_",
                offline_error="--jobs>1 for --axis all requires --offline (avoid parallel IBKR sessions).",
            )
            return

        self.axis_registry = build_axis_registry(self)

        if axis == "all":
            self._run_axis_plan_serial(list(_axis_mode_plan(mode="axis_all")), timed=False)
        else:
            fn_obj = self.axis_registry.get(str(axis))
            fn = fn_obj if callable(fn_obj) else None
            if fn is not None:
                self._run_axis_plan_serial([(str(axis), "single", False)], timed=False)

        if int(self.run_cfg_cache_hits) > 0 and int(self.run_calls_total) > 0:
            hit_rate = float(self.run_cfg_cache_hits) / float(self.run_calls_total)
            print(
                f"run_cfg cache (strategy+axis_dims+window): entries={len(self.run_cfg_cache)} hits={self.run_cfg_cache_hits}/{self.run_calls_total} "
                f"({hit_rate * 100.0:0.1f}%) fp_hits={int(self.run_cfg_fingerprint_hits)}",
                flush=True,
            )
        if bool(self.run_cfg_persistent_enabled):
            print(
                f"run_cfg persistent cache (strategy+axis_dims+window): path={self.run_cfg_persistent_path} "
                f"hits={int(self.run_cfg_persistent_hits)} writes={int(self.run_cfg_persistent_writes)} "
                f"dim_index_writes={int(self.run_cfg_dim_index_writes)} "
                f"worker_plan_hits={int(self.worker_plan_cache_hits)} worker_plan_writes={int(self.worker_plan_cache_writes)} "
                f"stage_status_reads={int(self.stage_cell_status_reads)} stage_status_writes={int(self.stage_cell_status_writes)} "
                f"cartesian_manifest_reads={int(self.cartesian_manifest_reads)} "
                f"cartesian_manifest_writes={int(self.cartesian_manifest_writes)} "
                f"cartesian_manifest_hits={int(self.cartesian_manifest_hits)} "
                f"cartesian_rank_manifest_reads={int(self.cartesian_rank_manifest_reads)} "
                f"cartesian_rank_manifest_writes={int(self.cartesian_rank_manifest_writes)} "
                f"cartesian_rank_manifest_hits={int(self.cartesian_rank_manifest_hits)} "
                f"cartesian_rank_manifest_compactions={int(self.cartesian_rank_manifest_compactions)} "
                f"cartesian_rank_manifest_pending_ttl_prunes={int(self.cartesian_rank_manifest_pending_ttl_prunes)} "
                f"stage_unresolved_summary_reads={int(self.stage_unresolved_summary_reads)} "
                f"stage_unresolved_summary_writes={int(self.stage_unresolved_summary_writes)} "
                f"stage_unresolved_summary_hits={int(self.stage_unresolved_summary_hits)} "
                f"rank_dominance_stamp_reads={int(self.rank_dominance_stamp_reads)} "
                f"rank_dominance_stamp_writes={int(self.rank_dominance_stamp_writes)} "
                f"rank_dominance_stamp_hits={int(self.rank_dominance_stamp_hits)} "
                f"rank_dominance_manifest_applies={int(self.rank_dominance_manifest_applies)} "
                f"rank_dominance_stamp_compactions={int(self.rank_dominance_stamp_compactions)} "
                f"rank_dominance_stamp_ttl_prunes={int(self.rank_dominance_stamp_ttl_prunes)} "
                f"rank_bin_runtime_reads={int(self.rank_bin_runtime_reads)} "
                f"rank_bin_runtime_writes={int(self.rank_bin_runtime_writes)} "
                f"stage_frontier_reads={int(self.stage_frontier_reads)} stage_frontier_writes={int(self.stage_frontier_writes)} "
                f"stage_frontier_hits={int(self.stage_frontier_hits)} "
                f"dimension_utility_reads={int(self.dimension_utility_reads)} "
                f"dimension_utility_writes={int(self.dimension_utility_writes)} "
                f"dimension_utility_hint_hits={int(self.dimension_utility_hint_hits)} "
                f"dimension_upper_bound_reads={int(self.dimension_upper_bound_reads)} "
                f"dimension_upper_bound_writes={int(self.dimension_upper_bound_writes)} "
                f"dimension_upper_bound_deferred={int(self.dimension_upper_bound_deferred)} "
                f"planner_heartbeat_reads={int(self.planner_heartbeat_reads)} "
                f"planner_heartbeat_writes={int(self.planner_heartbeat_writes)} "
                f"planner_heartbeat_stale_candidates={int(self.planner_heartbeat_stale_candidates)}",
                flush=True,
            )

        if bool(self.args.write_milestones) and not bool(self.milestones_written):
            eligible_new = _collect_milestone_items_from_rows(
                self.milestone_rows,
                meta=self.meta,
                min_win=float(self.args.milestone_min_win),
                min_trades=int(self.args.milestone_min_trades),
                min_pnl_dd=float(self.args.milestone_min_pnl_dd),
            )
            out_path = Path(self.args.milestones_out)
            total = _merge_and_write_milestones(
                out_path=out_path,
                eligible_new=eligible_new,
                merge_existing=bool(self.args.merge_milestones),
                add_top_pnl_dd=int(self.args.milestone_add_top_pnl_dd or 0),
                add_top_pnl=int(self.args.milestone_add_top_pnl or 0),
                symbol=self.symbol,
                start=self.start,
                end=self.end,
                signal_bar_size=self.signal_bar_size,
                use_rth=self.use_rth,
                milestone_min_win=float(self.args.milestone_min_win),
                milestone_min_trades=int(self.args.milestone_min_trades),
                milestone_min_pnl_dd=float(self.args.milestone_min_pnl_dd),
            )
            print(f"Wrote {out_path} ({total} eligible presets).")

        if not self.offline:
            self.data.disconnect()


def main() -> None:
    args = parse_spot_sweep_args()
    if bool(args.sync_axis_docs):
        out_path = Path(str(args.axis_docs_out or "tradebot/backtest/spot_sweep_coverage_map.md"))
        _write_spot_sweep_coverage_map(out_path, generated_on=_now_et().date())
        print(f"Wrote {out_path}")
        return
    SpotSweepRuntime(args).run()


if __name__ == "__main__":
    main()
