"""SweepEvaluation capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import threading
import time as pytime
from dataclasses import dataclass

from ...backtest.config import ConfigBundle
from ...backtest.engine import _run_spot_backtest_summary, _spot_prepare_summary_series_pack
from ...backtest.spot_context import SpotContextBars
from ...backtest.sweep_parallel import _progress_line
from ...chart_data.series import BarSeriesSignature
from .fingerprints import _axis_dimension_fingerprint, _window_signature
from .milestones import _milestone_key
from .support import _registry_float, _runtime_policy

_BarSignature = BarSeriesSignature
_ContextSignature = tuple[tuple[str, _BarSignature], ...]
_CacheKey = tuple[str, str, str]
_PlanItem = tuple[ConfigBundle, str, dict | None]
_CachedHit = tuple[_CacheKey, ConfigBundle, dict | None, str, dict | None]


@dataclass(frozen=True)
class _PreparedContext:
    bars: SpotContextBars
    series_pack: object | None


class SweepEvaluation:
    def _run_cfg_cache_coords(
        self,
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
        context: SpotContextBars | None = None,
        update_dim_index: bool = True,
    ) -> tuple[_ContextSignature, _CacheKey, str, str]:
        context_eff = context or self._context_bars_for_cfg(cfg=cfg, bars=bars)
        cfg_key = _milestone_key(cfg)
        ctx_sig = tuple(
            (kind, self._bars_signature(series))
            for kind, series in context_eff.items()
        )

        axis_dim_fp = self.run_cfg_axis_fp_cache.get(cfg_key)
        if axis_dim_fp is None:
            axis_dim_fp = _axis_dimension_fingerprint(cfg)
            self.run_cfg_axis_fp_cache[cfg_key] = str(axis_dim_fp)
        if bool(update_dim_index) and axis_dim_fp not in self.run_cfg_dim_index_seen:
            self.run_cfg_dim_index_seen.add(str(axis_dim_fp))
            est_cost = 1.0
            try:
                est_cost = float(self._cfg_eval_cost_hint(cfg))
            except Exception:
                est_cost = 1.0
            self._run_cfg_dimension_index_set(
                fingerprint=str(axis_dim_fp),
                payload_json=str(axis_dim_fp),
                est_cost=float(est_cost),
            )

        window_sig = self.run_cfg_window_sig_cache.get(ctx_sig)
        if window_sig is None:
            window_sig = _window_signature(context_sig=ctx_sig)
            self.run_cfg_window_sig_cache[ctx_sig] = str(window_sig)

        cache_key = (
            str(cfg_key),
            str(axis_dim_fp),
            str(window_sig),
        )
        persistent_key = self._run_cfg_persistent_key(
            strategy_fingerprint=str(cfg_key),
            axis_dimension_fingerprint=str(axis_dim_fp),
            window_signature=str(window_sig),
        )
        return ctx_sig, cache_key, str(axis_dim_fp), str(persistent_key)

    def _materialize_cache_hit(
        self,
        cached: object,
        *,
        ctx_sig: _ContextSignature,
        cache_key: _CacheKey,
        source: str,
    ) -> dict | None:
        """Centralize cache-tier accounting and promotion into the hot in-memory tiers."""
        payload = cached if isinstance(cached, dict) else None
        cfg_key = str(cache_key[0])
        self.run_cfg_cache_hits += 1
        if source == "fingerprint":
            self.run_cfg_fingerprint_hits += 1
        else:
            self.run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, payload)
        if source == "persistent":
            self.run_cfg_persistent_hits += 1
            self.run_cfg_cache[cache_key] = payload
        if not isinstance(payload, dict):
            return None
        try:
            eligible = int(payload.get("trades") or 0) >= int(self.run_min_trades)
        except (TypeError, ValueError):
            eligible = False
        return dict(payload) if eligible else None

    def _stage_partition_plan_by_cache(
        self,
        *,
        stage_label: str,
        plan_all,
        bars: list | None,
    ) -> tuple[list[_PlanItem], list[_CachedHit], dict[int, _CacheKey], int]:
        pending_plan: list[_PlanItem] = []
        pending_cell_map: dict[int, _CacheKey] = {}
        cached_hits: list[_CachedHit] = []
        prefetched_tested = 0
        status_updates: list[tuple[str, str, str, str]] = []
        manifest_updates: list[tuple[str, str, str, str]] = []
        planner_started = float(pytime.perf_counter())
        planner_last = float(planner_started)
        planner_heartbeat_sec = 15.0
        try:
            planner_total = int(len(plan_all))
        except Exception:
            planner_total = None

        def _planner_progress(*, phase: str, processed: int, force: bool = False) -> None:
            nonlocal planner_last
            now = float(pytime.perf_counter())
            if not bool(force) and (now - planner_last) < float(planner_heartbeat_sec):
                return
            total_s = str(planner_total) if isinstance(planner_total, int) and planner_total > 0 else "?"
            print(
                f"{stage_label} planner[{phase}] processed={int(processed)}/{total_s} pending={len(pending_plan)} unresolved={len(unresolved)} cached={len(cached_hits)} elapsed={now - planner_started:0.1f}s",
                flush=True,
            )
            planner_last = float(now)

        unresolved: list[tuple[ConfigBundle, str, dict | None, _ContextSignature, _CacheKey, str, _CacheKey]] = []
        persistent_keys: list[str] = []
        cell_keys_for_status: list[tuple[str, str, str]] = []
        stage_cell_hasher = hashlib.sha1()
        stage_cell_window_set: set[str] = set()
        stage_cell_window_hasher = hashlib.sha1()

        def _append_resolution(
            row: dict | None,
            *,
            cfg: ConfigBundle,
            note: str,
            meta: dict | None,
            cell_key: _CacheKey,
            manifest_status: str,
            strategy_fp: str | None = None,
        ) -> None:
            nonlocal prefetched_tested
            prefetched_tested += 1
            self.run_calls_total += 1
            self._axis_progress_record(kept=bool(row))
            status_updates.append((*cell_key, "cached_hit"))
            manifest_updates.append((cell_key[1], cell_key[2], strategy_fp or cell_key[0], manifest_status))
            if isinstance(row, dict):
                row["note"] = note
            cached_hits.append((cell_key, cfg, row, note, meta))

        def _append_cached(
            cached: object,
            *,
            source: str,
            cfg: ConfigBundle,
            note: str,
            meta: dict | None,
            ctx_sig: _ContextSignature,
            cache_key: _CacheKey,
            cell_key: _CacheKey,
        ) -> None:
            row = self._materialize_cache_hit(cached, ctx_sig=ctx_sig, cache_key=cache_key, source=source)
            _append_resolution(row, cfg=cfg, note=note, meta=meta, cell_key=cell_key, manifest_status="cached_hit")

        for idx, item in enumerate(plan_all, start=1):
            if not (isinstance(item, tuple) and len(item) >= 2):
                _planner_progress(phase="scan", processed=int(idx))
                continue
            cfg = item[0] if isinstance(item[0], ConfigBundle) else None
            if cfg is None:
                _planner_progress(phase="scan", processed=int(idx))
                continue
            note_s = str(item[1] or "")
            meta_item = item[2] if len(item) >= 3 and isinstance(item[2], dict) else None
            ctx_sig, cache_key, axis_dim_fp, persistent_key = self._run_cfg_cache_coords(
                cfg=cfg,
                bars=bars,
                update_dim_index=True,
            )
            cfg_key = str(cache_key[0])
            cell_key = (str(cache_key[0]), str(axis_dim_fp), str(cache_key[2]))
            cell_keys_for_status.append(cell_key)
            stage_cell_hasher.update(str(cell_key[0]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1f")
            stage_cell_hasher.update(str(cell_key[1]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1f")
            stage_cell_hasher.update(str(cell_key[2]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1e")
            window_sig = str(cell_key[2]).strip()
            if window_sig:
                stage_cell_window_set.add(str(window_sig))
                stage_cell_window_hasher.update(str(window_sig).encode("utf-8"))
                stage_cell_window_hasher.update(b"\x1f")

            fp_cached = self.run_cfg_fingerprint_cache.get(cfg_key)
            if fp_cached is not None and fp_cached[0] == ctx_sig:
                _append_cached(
                    fp_cached[1],
                    source="fingerprint",
                    cfg=cfg,
                    note=note_s,
                    meta=meta_item,
                    ctx_sig=ctx_sig,
                    cache_key=cache_key,
                    cell_key=cell_key,
                )
                _planner_progress(phase="scan", processed=int(idx))
                continue

            cached = self.run_cfg_cache.get(cache_key, self._RUN_CFG_CACHE_MISS)
            if cached is not self._RUN_CFG_CACHE_MISS:
                _append_cached(cached, source="memory", cfg=cfg, note=note_s, meta=meta_item, ctx_sig=ctx_sig, cache_key=cache_key, cell_key=cell_key)
                _planner_progress(phase="scan", processed=int(idx))
                continue

            unresolved.append(
                (
                    cfg,
                    note_s,
                    meta_item,
                    ctx_sig,
                    cache_key,
                    str(persistent_key),
                    cell_key,
                )
            )
            persistent_keys.append(str(persistent_key))
            _planner_progress(phase="scan", processed=int(idx))

        _planner_progress(
            phase="scan",
            processed=(int(planner_total) if isinstance(planner_total, int) and planner_total > 0 else len(unresolved)),
            force=True,
        )

        stage_cell_total = int(len(cell_keys_for_status))
        stage_cell_plan_signature = str(stage_cell_hasher.hexdigest()) if int(stage_cell_total) > 0 else ""
        if int(stage_cell_total) <= 0:
            stage_cell_window_signature = ""
        elif len(stage_cell_window_set) == 1:
            stage_cell_window_signature = next(iter(stage_cell_window_set))
        else:
            stage_cell_window_signature = "multi:" + str(stage_cell_window_hasher.hexdigest())
        stage_cell_summary_all_resolved = False
        if int(stage_cell_total) > 0 and stage_cell_plan_signature and stage_cell_window_signature:
            stage_summary = self._stage_unresolved_summary_get(
                manifest_name="stage_cell",
                stage_label=str(stage_label),
                plan_signature=str(stage_cell_plan_signature),
                window_signature=str(stage_cell_window_signature),
                total=int(stage_cell_total),
            )
            if isinstance(stage_summary, tuple) and len(stage_summary) == 2:
                try:
                    stage_cell_summary_all_resolved = int(stage_summary[0]) <= 0
                except (TypeError, ValueError):
                    stage_cell_summary_all_resolved = False

        persisted_by_key = self._run_cfg_persistent_get_many(cache_keys=persistent_keys) if persistent_keys else {}
        frontier_by_dim_window: dict[tuple[str, str], dict[str, object]] = {}
        upper_bound_by_dim_window: dict[tuple[str, str], dict[str, object]] = {}
        manifest_by_dim_window: dict[tuple[str, str], tuple[str, str]] = {}
        if not bool(stage_cell_summary_all_resolved):
            frontier_by_dim_window = self._stage_frontier_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
            upper_bound_by_dim_window = self._dimension_upper_bound_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
            manifest_by_dim_window = self._cartesian_cell_manifest_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
        prior_status = (
            {(strategy_fp, axis_dim_fp, window_sig): "cached_hit" for strategy_fp, axis_dim_fp, window_sig in cell_keys_for_status}
            if bool(stage_cell_summary_all_resolved)
            else self._stage_cell_status_get_many(stage_label=str(stage_label), cells=cell_keys_for_status)
        )
        rank_dominance_stamp_rows_by_window: dict[str, list[tuple[str, int, int]]] = {}

        unresolved_total = len(unresolved)
        for idx_u, (
            cfg,
            note_s,
            meta_item,
            ctx_sig,
            cache_key,
            persistent_key,
            cell_key,
        ) in enumerate(unresolved, start=1):
            persisted = persisted_by_key.get(str(persistent_key), self._RUN_CFG_CACHE_MISS)
            cfg_key = str(cache_key[0])
            if persisted is not self._RUN_CFG_CACHE_MISS:
                _append_cached(
                    persisted,
                    source="persistent",
                    cfg=cfg,
                    note=note_s,
                    meta=meta_item,
                    ctx_sig=ctx_sig,
                    cache_key=cache_key,
                    cell_key=cell_key,
                )
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            prev_cell_status = str(prior_status.get(cell_key) or "").strip().lower()
            if prev_cell_status in ("cached_hit", "evaluated"):
                _append_resolution(None, cfg=cfg, note=note_s, meta=meta_item, cell_key=cell_key, manifest_status="cached_hit")
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            frontier_key = (str(cell_key[1]), str(cell_key[2]))
            upper_bound_row = upper_bound_by_dim_window.get(frontier_key)
            upper_bound_sig = self._upper_bound_dominance_signature(upper_bound_row)
            if upper_bound_sig:
                if isinstance(meta_item, dict):
                    try:
                        rank_i = int(meta_item.get("_mr_rank"))
                    except (TypeError, ValueError):
                        rank_i = None
                    if rank_i is not None and int(rank_i) >= 0:
                        window_key = str(cell_key[2]).strip()
                        if window_key:
                            rank_dominance_stamp_rows_by_window.setdefault(str(window_key), []).append((str(upper_bound_sig), int(rank_i), int(rank_i)))
                _append_resolution(None, cfg=cfg, note=note_s, meta=meta_item, cell_key=cell_key, manifest_status="dominated")
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            frontier_row = frontier_by_dim_window.get(frontier_key)
            if self._stage_frontier_is_dominated(frontier_row):
                self.stage_frontier_hits += 1
                _append_resolution(None, cfg=cfg, note=note_s, meta=meta_item, cell_key=cell_key, manifest_status="dominated")
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            manifest_state = manifest_by_dim_window.get(frontier_key)
            manifest_status = str(manifest_state[0]).strip().lower() if isinstance(manifest_state, tuple) else ""
            manifest_strategy_fp = str(manifest_state[1]).strip() if isinstance(manifest_state, tuple) and len(manifest_state) >= 2 else ""
            if manifest_status in ("cached_hit", "evaluated", "dominated"):
                self.cartesian_manifest_hits += 1
                resolved_row: dict | None = None
                if manifest_status in ("cached_hit", "evaluated"):
                    probe_strategy_fp = manifest_strategy_fp or str(cell_key[0])
                    probe_key = self._run_cfg_persistent_key(
                        strategy_fingerprint=str(probe_strategy_fp),
                        axis_dimension_fingerprint=str(cell_key[1]),
                        window_signature=str(cell_key[2]),
                    )
                    probe_cached = self._run_cfg_persistent_get(cache_key=str(probe_key))
                    if probe_cached is not self._RUN_CFG_CACHE_MISS:
                        resolved_row = self._materialize_cache_hit(
                            probe_cached,
                            ctx_sig=ctx_sig,
                            cache_key=cache_key,
                            source="persistent",
                        )

                _append_resolution(
                    resolved_row,
                    cfg=cfg,
                    note=note_s,
                    meta=meta_item,
                    cell_key=cell_key,
                    manifest_status="dominated" if manifest_status == "dominated" else "cached_hit",
                    strategy_fp=manifest_strategy_fp,
                )
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            pending_idx = len(pending_plan)
            pending_plan.append((cfg, note_s, meta_item))
            pending_cell_map[pending_idx] = cell_key
            status_updates.append((cell_key[0], cell_key[1], cell_key[2], "pending"))
            manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "pending"))
            _planner_progress(phase="resolve", processed=int(idx_u))

        if unresolved_total > 0:
            _planner_progress(phase="resolve", processed=int(unresolved_total), force=True)

        status_updates_filtered: list[tuple[str, str, str, str]] = []
        for strategy_fp, axis_dim_fp, window_sig, status in status_updates:
            prev = prior_status.get((strategy_fp, axis_dim_fp, window_sig))
            if str(prev or "") == str(status):
                continue
            status_updates_filtered.append((strategy_fp, axis_dim_fp, window_sig, status))
        if status_updates_filtered:
            self._stage_cell_status_set_many(stage_label=str(stage_label), rows=status_updates_filtered)

        manifest_updates_filtered: list[tuple[str, str, str, str]] = []
        manifest_priority = {
            "pending": 0,
            "cached_hit": 1,
            "evaluated": 2,
            "dominated": 3,
        }
        for axis_dim_fp, window_sig, strategy_fp, status in manifest_updates:
            prev_state = manifest_by_dim_window.get((str(axis_dim_fp), str(window_sig)))
            prev_status = str(prev_state[0]).strip().lower() if isinstance(prev_state, tuple) else ""
            if prev_status == str(status):
                continue
            if manifest_priority.get(str(status), -1) < manifest_priority.get(str(prev_status), -1):
                continue
            manifest_updates_filtered.append((axis_dim_fp, window_sig, strategy_fp, status))
        if manifest_updates_filtered:
            self._cartesian_cell_manifest_set_many(stage_label=str(stage_label), rows=manifest_updates_filtered)

        if rank_dominance_stamp_rows_by_window:
            for window_sig, stamp_rows in rank_dominance_stamp_rows_by_window.items():
                self._rank_dominance_stamp_set_many(
                    stage_label=str(stage_label),
                    window_signature=str(window_sig),
                    rows=list(stamp_rows),
                )

        if int(stage_cell_total) > 0 and stage_cell_plan_signature and stage_cell_window_signature:
            unresolved_count = int(len(pending_plan))
            self._stage_unresolved_summary_set(
                manifest_name="stage_cell",
                stage_label=str(stage_label),
                plan_signature=str(stage_cell_plan_signature),
                window_signature=str(stage_cell_window_signature),
                total=int(stage_cell_total),
                unresolved_count=int(unresolved_count),
                resolved_count=int(max(0, int(stage_cell_total) - int(unresolved_count))),
            )

        return pending_plan, cached_hits, pending_cell_map, int(prefetched_tested)

    def _run_cfg(
        self,
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
        prepared_context: _PreparedContext | None = None,
        progress_callback=None,
    ) -> dict | None:
        self.run_calls_total += 1

        def _emit_cfg_progress(**payload: object) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(dict(payload))
            except Exception:
                return

        if isinstance(prepared_context, _PreparedContext):
            context = prepared_context.bars
            prepared_pack = prepared_context.series_pack
        else:
            context = self._context_bars_for_cfg(cfg=cfg, bars=bars)
            prepared_pack = None
        ctx_sig, cache_key, axis_dim_fp, persistent_key = self._run_cfg_cache_coords(
            cfg=cfg,
            context=context,
            update_dim_index=True,
        )
        cfg_key = str(cache_key[0])
        fp_cached = self.run_cfg_fingerprint_cache.get(cfg_key)
        if fp_cached is not None and fp_cached[0] == ctx_sig:
            row = self._materialize_cache_hit(fp_cached[1], ctx_sig=ctx_sig, cache_key=cache_key, source="fingerprint")
            self._axis_progress_record(kept=bool(row))
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            return row
        cached = self.run_cfg_cache.get(cache_key, self._RUN_CFG_CACHE_MISS)
        if cached is not self._RUN_CFG_CACHE_MISS:
            row = self._materialize_cache_hit(cached, ctx_sig=ctx_sig, cache_key=cache_key, source="memory")
            self._axis_progress_record(kept=bool(row))
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            return row
        persisted = self._run_cfg_persistent_get(cache_key=str(persistent_key))
        if persisted is not self._RUN_CFG_CACHE_MISS:
            row = self._materialize_cache_hit(persisted, ctx_sig=ctx_sig, cache_key=cache_key, source="persistent")
            self._axis_progress_record(kept=bool(row))
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            return row
        _emit_cfg_progress(
            phase="cfg.context_ready",
            cached=False,
            signal_total=int(len(context.signal_bars) if isinstance(context.signal_bars, list) else 0),
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
                if isinstance(context.signal_bars, list)
                else 0
            ),
        )
        eval_started = pytime.perf_counter()
        s = _run_spot_backtest_summary(
            cfg,
            context.signal_bars,
            self.meta,
            regime_bars=context.regime_bars,
            regime2_bars=context.regime2_bars,
            regime2_bear_hard_bars=context.regime2_bear_hard_bars,
            tick_bars=context.tick_bars,
            exec_bars=context.exec_bars,
            prepared_series_pack=prepared_pack,
            progress_callback=progress_callback,
        )
        eval_elapsed = max(1e-6, float(pytime.perf_counter()) - float(eval_started))
        _emit_cfg_progress(
            phase="cfg.engine_done",
            elapsed_sec=float(eval_elapsed),
            trades=int(getattr(s, "trades", 0) or 0),
        )
        self._run_cfg_dimension_index_set(
            fingerprint=str(axis_dim_fp),
            payload_json=str(axis_dim_fp),
            est_cost=float(eval_elapsed),
        )
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        roi = float(getattr(s, "roi", 0.0) or 0.0)
        dd_pct = float(getattr(s, "max_drawdown_pct", 0.0) or 0.0)
        out = {
            "trades": int(s.trades),
            "win_rate": float(s.win_rate),
            "pnl": pnl,
            "dd": dd,
            "roi": roi,
            "dd_pct": dd_pct,
            "pnl_over_dd": (pnl / dd) if dd > 0 else None,
        }
        self.run_cfg_cache[cache_key] = out
        self.run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, out)
        self._run_cfg_persistent_set(cache_key=str(persistent_key), payload=out)
        self.run_cfg_persistent_writes += 1
        eligible = int(s.trades) >= int(self.run_min_trades)
        self._axis_progress_record(kept=bool(eligible))
        return dict(out) if eligible else None

    def _run_sweep(
        self,
        *,
        plan,
        bars: list,
        total: int | None = None,
        progress_label: str | None = None,
        report_every: int = 0,
        heartbeat_sec: float = 0.0,
        record_milestones: bool = True,
        frontier_stage_label: str | None = None,
        progress_callback=None,
    ) -> tuple[int, list[tuple[ConfigBundle, dict, str, dict | None]]]:
        tested = 0
        kept: list[tuple[ConfigBundle, dict, str, dict | None]] = []
        frontier_updates: list[tuple[str, str, dict | None]] = []
        dimension_upper_bound_updates: list[tuple[str, str, dict | None]] = []
        rank_runtime_updates: dict[tuple[str, int], tuple[int, float, int]] = {}
        dimension_utility_updates: dict[tuple[str, str, str], tuple[int, int, int, float]] = {}
        t0 = pytime.perf_counter()
        last = float(t0)
        total_i = int(total) if total is not None else None
        suppress_prev = bool(self.axis_progress_state.get("suppress"))
        heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 0.0
        heartbeat_eff = max(5.0, float(heartbeat_eff)) if float(heartbeat_eff) > 0.0 else 0.0
        eval_inflight_started_at = 0.0
        eval_inflight_active = False
        eval_phase_state: dict[str, object] = {}
        eval_phase_lock = threading.Lock()
        sweep_heartbeat_stop = threading.Event()
        sweep_heartbeat_thread: threading.Thread | None = None
        dim_util_cfg = _runtime_policy("dimension_value_utility")
        dim_upper_cfg = _runtime_policy("dimension_upper_bound")
        dim_util_write_min_total = max(0, int(_registry_float(dim_util_cfg.get("write_min_total"), 128.0)))
        dim_upper_write_min_total = max(0, int(_registry_float(dim_upper_cfg.get("write_min_total"), 96.0)))
        dim_util_write_sample_mod = max(1, int(_registry_float(dim_util_cfg.get("write_sample_mod"), 1.0)))
        dim_upper_write_sample_mod = max(1, int(_registry_float(dim_upper_cfg.get("write_sample_mod"), 1.0)))
        allow_dim_util_writes = bool(frontier_stage_label and (total_i is None or int(total_i) >= int(dim_util_write_min_total)))
        allow_dim_upper_writes = bool(frontier_stage_label and (total_i is None or int(total_i) >= int(dim_upper_write_min_total)))
        series_pack_prewarm_cfg = _runtime_policy("series_pack_prewarm")
        series_pack_prewarm_enabled = bool(_registry_float(series_pack_prewarm_cfg.get("enabled"), 1.0) > 0.0)
        series_pack_prewarm_min_total = max(0, int(_registry_float(series_pack_prewarm_cfg.get("min_total"), 32.0)))
        series_pack_prewarm_max_unique = max(1, int(_registry_float(series_pack_prewarm_cfg.get("max_unique"), 4096.0)))
        use_prepared_context = bool(series_pack_prewarm_enabled and (total_i is None or int(total_i) >= int(series_pack_prewarm_min_total)))
        prepared_context_by_cache_key: dict[_CacheKey, _PreparedContext] = {}
        prepared_series_pack_by_hash: dict[str, object | None] = {}
        if progress_label:
            self.axis_progress_state["suppress"] = True

        def _emit_progress(done: bool = False) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(
                    tested=int(tested),
                    total=(int(total_i) if total_i is not None else None),
                    kept=int(len(kept)),
                    elapsed=max(0.0, float(pytime.perf_counter()) - float(t0)),
                    done=bool(done),
                )
            except Exception:
                return

        def _emit_sweep_heartbeat() -> None:
            nonlocal last
            if not progress_label and not callable(progress_callback):
                return
            now = float(pytime.perf_counter())
            if progress_label:
                line = _progress_line(
                    label=str(progress_label),
                    tested=int(tested),
                    total=total_i,
                    kept=len(kept),
                    started_at=t0,
                    rate_unit="s",
                )
                if bool(eval_inflight_active) and float(eval_inflight_started_at) > 0.0:
                    line += f" inflight={max(0.0, now - float(eval_inflight_started_at)):0.1f}s"
                with eval_phase_lock:
                    phase_snap = dict(eval_phase_state)
                phase_name = str(phase_snap.get("phase") or "").strip()
                if phase_name:
                    line += f" stage={phase_name}"
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
                window_idx = phase_snap.get("window_idx")
                window_total = phase_snap.get("window_total")
                if isinstance(window_idx, int) and isinstance(window_total, int) and int(window_total) > 0:
                    line += f" window={int(window_idx)}/{int(window_total)}"
                print(line, flush=True)
            last = float(now)
            _emit_progress(done=False)

        def _update_eval_phase(event: dict | None) -> None:
            if not isinstance(event, dict):
                return
            with eval_phase_lock:
                for key in (
                    "phase",
                    "path",
                    "window_idx",
                    "window_total",
                    "signal_total",
                    "regime_total",
                    "regime2_total",
                    "tick_total",
                    "exec_total",
                    "sig_idx",
                    "exec_idx",
                    "open_count",
                    "trades",
                    "cached",
                    "kept",
                ):
                    if key in event:
                        eval_phase_state[str(key)] = event.get(key)

        def _sweep_heartbeat_worker() -> None:
            if float(heartbeat_eff) <= 0.0:
                return
            while not sweep_heartbeat_stop.wait(float(heartbeat_eff)):
                if bool(eval_inflight_active):
                    _emit_sweep_heartbeat()

        if float(heartbeat_eff) > 0.0 and (progress_label or callable(progress_callback)):
            sweep_heartbeat_thread = threading.Thread(target=_sweep_heartbeat_worker, daemon=True)
            sweep_heartbeat_thread.start()

        def _flush_rank_runtime_updates() -> None:
            if not frontier_stage_label or not rank_runtime_updates:
                return
            rows_to_set = [
                (
                    str(window_sig),
                    int(rank_bin),
                    int(rec[0]),
                    float(rec[1]),
                    int(rec[2]),
                )
                for (window_sig, rank_bin), rec in rank_runtime_updates.items()
                if str(window_sig) and int(rec[0]) > 0
            ]
            rank_runtime_updates.clear()
            if rows_to_set:
                self._rank_bin_runtime_set_many(stage_label=str(frontier_stage_label), rows=rows_to_set)

        def _flush_dimension_utility_updates() -> None:
            if not frontier_stage_label or not dimension_utility_updates:
                return
            rows_to_set = [
                (
                    str(window_sig),
                    str(dim_key),
                    str(dim_value),
                    int(rec[0]),
                    int(rec[1]),
                    int(rec[2]),
                    float(rec[3]),
                )
                for (
                    window_sig,
                    dim_key,
                    dim_value,
                ), rec in dimension_utility_updates.items()
                if str(window_sig) and str(dim_key) and int(rec[0]) > 0
            ]
            dimension_utility_updates.clear()
            if rows_to_set:
                self._dimension_value_utility_set_many(stage_label=str(frontier_stage_label), rows=rows_to_set)

        try:
            for cfg, note, meta_item in plan:
                prepared_context = None
                if bool(use_prepared_context):
                    context_pc = self._context_bars_for_cfg(cfg=cfg, bars=bars)
                    _ctx_sig_pc, cache_key_pc, _axis_dim_fp_pc, _persistent_key_pc = self._run_cfg_cache_coords(
                        cfg=cfg,
                        context=context_pc,
                        update_dim_index=False,
                    )
                    prepared_context = prepared_context_by_cache_key.get(cache_key_pc)
                    if prepared_context is None and len(prepared_context_by_cache_key) < int(series_pack_prewarm_max_unique):
                        pack_hash, prepared_pack = _spot_prepare_summary_series_pack(
                            cfg=cfg,
                            signal_bars=context_pc.signal_bars,
                            tick_bars=context_pc.tick_bars,
                            exec_bars=context_pc.exec_bars,
                        )
                        if pack_hash:
                            if pack_hash in prepared_series_pack_by_hash:
                                prepared_pack = prepared_series_pack_by_hash.get(pack_hash)
                            elif len(prepared_series_pack_by_hash) < int(series_pack_prewarm_max_unique):
                                prepared_series_pack_by_hash[str(pack_hash)] = prepared_pack
                        prepared_context = _PreparedContext(
                            bars=context_pc,
                            series_pack=prepared_pack,
                        )
                        prepared_context_by_cache_key[cache_key_pc] = prepared_context

                cache_hits_before = int(self.run_cfg_cache_hits)
                eval_started = pytime.perf_counter()
                eval_inflight_started_at = float(eval_started)
                eval_inflight_active = True
                try:
                    row = self._run_cfg(
                        cfg=cfg,
                        bars=bars,
                        prepared_context=prepared_context,
                        progress_callback=_update_eval_phase,
                    )
                finally:
                    eval_inflight_active = False
                    eval_inflight_started_at = 0.0
                tested += 1
                eval_elapsed = max(1e-6, float(pytime.perf_counter()) - float(eval_started))
                cache_hit_eval = 1 if int(self.run_cfg_cache_hits) > int(cache_hits_before) else 0
                if frontier_stage_label:
                    _ctx_sig, cache_key, axis_dim_fp, _persistent_key = self._run_cfg_cache_coords(
                        cfg=cfg,
                        bars=bars,
                        update_dim_index=False,
                    )
                    window_sig = str(cache_key[2])
                    frontier_updates.append(
                        (
                            str(axis_dim_fp),
                            str(window_sig),
                            row if isinstance(row, dict) else None,
                        )
                    )
                    if len(frontier_updates) >= 200:
                        self._stage_frontier_upsert_many(
                            stage_label=str(frontier_stage_label),
                            rows=frontier_updates,
                        )
                        frontier_updates.clear()
                    if bool(allow_dim_upper_writes) and (int(dim_upper_write_sample_mod) <= 1 or (int(tested) % int(dim_upper_write_sample_mod) == 0)):
                        dimension_upper_bound_updates.append(
                            (
                                str(axis_dim_fp),
                                str(window_sig),
                                row if isinstance(row, dict) else None,
                            )
                        )
                        if len(dimension_upper_bound_updates) >= 200:
                            self._dimension_upper_bound_upsert_many(
                                stage_label=str(frontier_stage_label),
                                rows=dimension_upper_bound_updates,
                            )
                            dimension_upper_bound_updates.clear()
                    rank_raw = meta_item.get("_mr_rank") if isinstance(meta_item, dict) else None
                    try:
                        rank_i = int(rank_raw)
                    except (TypeError, ValueError):
                        rank_i = -1
                    if rank_i >= 0 and window_sig:
                        rank_bin = int(self._rank_bin_from_rank(rank_i))
                        cell = (str(window_sig), int(rank_bin))
                        prev = rank_runtime_updates.get(cell)
                        if prev is None:
                            rank_runtime_updates[cell] = (
                                1,
                                float(eval_elapsed),
                                int(cache_hit_eval),
                            )
                        else:
                            rank_runtime_updates[cell] = (
                                int(prev[0]) + 1,
                                float(prev[1]) + float(eval_elapsed),
                                int(prev[2]) + int(cache_hit_eval),
                            )
                        if len(rank_runtime_updates) >= 256:
                            _flush_rank_runtime_updates()
                    if (
                        window_sig
                        and isinstance(meta_item, dict)
                        and bool(allow_dim_util_writes)
                        and (int(dim_util_write_sample_mod) <= 1 or (int(tested) % int(dim_util_write_sample_mod) == 0))
                    ):
                        keep_i = 1 if isinstance(row, dict) else 0
                        for raw_dim_key, raw_dim_value in meta_item.items():
                            dim_key = str(raw_dim_key or "").strip()
                            if not dim_key or dim_key.startswith("_"):
                                continue
                            dim_value = str(raw_dim_value)
                            cell = (str(window_sig), str(dim_key), str(dim_value))
                            prev = dimension_utility_updates.get(cell)
                            if prev is None:
                                dimension_utility_updates[cell] = (
                                    1,
                                    int(keep_i),
                                    int(cache_hit_eval),
                                    float(eval_elapsed),
                                )
                            else:
                                dimension_utility_updates[cell] = (
                                    int(prev[0]) + 1,
                                    int(prev[1]) + int(keep_i),
                                    int(prev[2]) + int(cache_hit_eval),
                                    float(prev[3]) + float(eval_elapsed),
                                )
                        if len(dimension_utility_updates) >= 768:
                            _flush_dimension_utility_updates()
                if row:
                    note_s = str(note or "")
                    row = dict(row)
                    if note_s:
                        row["note"] = note_s
                        if bool(record_milestones):
                            self._record_milestone(cfg, row, note_s)
                    kept.append((cfg, row, note_s, meta_item))

                if progress_label:
                    now = pytime.perf_counter()
                    hit_report_every = int(report_every) > 0 and (tested % int(report_every) == 0)
                    hit_total = total_i is not None and tested == int(total_i)
                    hit_heartbeat = float(heartbeat_sec) > 0 and (now - last) >= float(heartbeat_sec)
                    if hit_report_every or hit_total or hit_heartbeat:
                        print(
                            _progress_line(
                                label=str(progress_label),
                                tested=int(tested),
                                total=total_i,
                                kept=len(kept),
                                started_at=t0,
                                rate_unit="s",
                            ),
                            flush=True,
                        )
                        last = float(now)
                        _emit_progress(done=False)
        finally:
            sweep_heartbeat_stop.set()
            if sweep_heartbeat_thread is not None:
                try:
                    sweep_heartbeat_thread.join(timeout=1.0)
                except Exception:
                    pass
            if frontier_updates and frontier_stage_label:
                self._stage_frontier_upsert_many(
                    stage_label=str(frontier_stage_label),
                    rows=frontier_updates,
                )
            if dimension_upper_bound_updates and frontier_stage_label:
                self._dimension_upper_bound_upsert_many(
                    stage_label=str(frontier_stage_label),
                    rows=dimension_upper_bound_updates,
                )
            _flush_rank_runtime_updates()
            _flush_dimension_utility_updates()
            self._run_cfg_persistent_flush_pending(force=True)
            _emit_progress(done=True)
            self.axis_progress_state["suppress"] = suppress_prev

        return tested, kept
