"""SweepWorkers capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import math
import time as pytime
from pathlib import Path
from ...backtest.config import (
    ConfigBundle,
)
from ...backtest.sweep_parallel import (
    _collect_parallel_payload_records,
    _parse_worker_shard,
)
from ...backtest.sweeps import (
    write_json,
)
from .milestones import (
    _milestone_key,
)
from .support import (
    _registry_float,
    _runtime_policy,
)


class SweepWorkers:
    def _run_sharded_stage_worker_kernel(
        self,
        *,
        stage_label: str,
        worker_raw,
        workers_raw,
        out_path_raw: str,
        out_flag_name: str,
        plan_all,
        bars: list,
        report_every: int,
        heartbeat_sec: float = 0.0,
        plan_total: int | None = None,
        plan_item_from_rank=None,
        rank_manifest_window_signature: str = "",
        rank_batch_size: int = 384,
    ) -> None:
        def _run_sharded_stage_worker_lazy_rank(
            *,
            total_ranks: int,
            item_from_rank,
            manifest_window_signature: str,
            batch_size: int,
        ) -> None:
            if not self.offline:
                raise SystemExit(f"{stage_label} worker mode requires --offline (avoid parallel IBKR sessions).")
            out_path_str = str(out_path_raw or "").strip()
            if not out_path_str:
                raise SystemExit(f"--{out_flag_name} is required for {stage_label} worker mode.")
            out_path = Path(out_path_str)
            worker_id, workers = _parse_worker_shard(worker_raw, workers_raw, label=str(stage_label))
            use_rank_manifest = bool(str(manifest_window_signature).strip())
            unresolved_ranges = (
                self._cartesian_rank_manifest_unresolved_ranges(
                    stage_label=str(stage_label),
                    window_signature=str(manifest_window_signature),
                    total=int(total_ranks),
                )
                if use_rank_manifest
                else ((0, int(total_ranks) - 1),)
            )
            unresolved_total = sum(max(0, int(rank_hi) - int(rank_lo) + 1) for rank_lo, rank_hi in unresolved_ranges)
            dynamic_claim_mode = bool(
                use_rank_manifest and self._run_cfg_persistent_conn() is not None
            )
            worker_ranges: list[tuple[int, int]] = []
            local_total = 0
            if not bool(dynamic_claim_mode):
                range_buckets = self._partition_rank_ranges_for_workers(
                    ranges=tuple(unresolved_ranges),
                    workers=int(workers),
                )
                if int(worker_id) >= len(range_buckets):
                    raise SystemExit(f"Invalid {stage_label} worker shard: worker={worker_id} workers={workers}.")
                worker_ranges = list(range_buckets[int(worker_id)])
                local_total = sum(max(0, int(rank_hi) - int(rank_lo) + 1) for rank_lo, rank_hi in worker_ranges)
            if unresolved_ranges:
                preview = ", ".join(f"{int(rank_lo)}-{int(rank_hi)}" for rank_lo, rank_hi in tuple(unresolved_ranges[:8]))
                if preview:
                    print(
                        f"{stage_label} unresolved rank ranges total={int(unresolved_total)}/{int(total_ranks)} {preview}",
                        flush=True,
                    )
            if worker_ranges:
                assigned_preview = ", ".join(f"{int(rank_lo)}-{int(rank_hi)}" for rank_lo, rank_hi in tuple(worker_ranges[:8]))
                if assigned_preview:
                    print(
                        f"{stage_label} worker {int(worker_id) + 1}/{int(workers)} ranges {assigned_preview}",
                        flush=True,
                    )
            self._planner_heartbeat_set(
                stage_label=str(stage_label),
                worker_id=int(worker_id),
                tested=0,
                cached_hits=0,
                total=(0 if bool(dynamic_claim_mode) else int(local_total)),
                eta_sec=(0.0 if int(unresolved_total) <= 0 else None),
                status=("done" if int(unresolved_total) <= 0 else "starting"),
            )
            if int(unresolved_total) <= 0:
                write_json(out_path, {"tested": 0, "kept": 0, "records": []}, sort_keys=False)
                print(
                    f"{stage_label} worker done tested=0 kept=0 out={out_path} (no unresolved ranks)",
                    flush=True,
                )
                return
            if not bool(dynamic_claim_mode) and int(local_total) <= 0:
                write_json(out_path, {"tested": 0, "kept": 0, "records": []}, sort_keys=False)
                print(
                    f"{stage_label} worker done tested=0 kept=0 out={out_path} (no shard assignment)",
                    flush=True,
                )
                return

            worker_started_at = float(pytime.perf_counter())
            tested_eval_total = 0
            cached_hits_total = 0
            processed_ranks = 0
            claimed_ranks = 0
            records: list[dict] = []
            batch_size_i = max(1, int(batch_size))
            small_total_threshold = int(max(1, int(workers) * int(batch_size_i) * 2))
            if int(total_ranks) <= int(small_total_threshold):
                claim_span_default = max(
                    4,
                    int(math.ceil(float(max(1, int(total_ranks))) / float(max(1, int(workers) * 3)))),
                )
                claim_span_default = min(int(batch_size_i), int(claim_span_default))
            else:
                claim_span_default = max(
                    int(batch_size_i),
                    min(
                        int(batch_size_i * 8),
                        int(math.ceil(float(max(1, int(total_ranks))) / float(max(1, int(workers) * 64)))),
                    ),
                )
            claim_span_i = int(claim_span_default)
            claim_cfg = _runtime_policy("claim_span_tuner")
            if bool(_registry_float(claim_cfg.get("enabled"), 1.0) > 0.0):
                target_claims = max(
                    2,
                    int(_registry_float(claim_cfg.get("target_claims_per_worker"), 24.0)),
                )
                min_claim_span = max(1, int(_registry_float(claim_cfg.get("min_claim_span"), 32.0)))
                max_claim_span = max(
                    int(min_claim_span),
                    int(_registry_float(claim_cfg.get("max_claim_span"), 2048.0)),
                )
                max_batch_multiple = max(1, int(_registry_float(claim_cfg.get("max_batch_multiple"), 8.0)))
                tuned_span = int(math.ceil(float(max(1, int(total_ranks))) / float(max(1, int(workers) * int(target_claims)))))
                tuned_span = max(int(min_claim_span), min(int(max_claim_span), int(tuned_span)))
                tuned_span = min(
                    int(tuned_span),
                    int(max(1, int(batch_size_i)) * int(max_batch_multiple)),
                )
                claim_span_i = max(1, int(tuned_span))
            claim_span_i = max(1, int(claim_span_i))
            if bool(dynamic_claim_mode):
                print(
                    f"{stage_label} worker {int(worker_id) + 1}/{int(workers)} dynamic-claim enabled "
                    f"(claim_span={int(claim_span_i)}, batch={int(batch_size_i)})",
                    flush=True,
                )
            heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 30.0
            last_progress_ts = float(worker_started_at)

            def _emit_worker_heartbeat(*, done: bool = False, tested_override: int | None = None) -> None:
                nonlocal claimed_ranks
                elapsed = max(0.0, float(pytime.perf_counter()) - float(worker_started_at))
                processed = int(max(0, int(tested_override))) if tested_override is not None else int(max(0, int(processed_ranks)))
                total_i = int(max(0, int(claimed_ranks))) if bool(dynamic_claim_mode) else int(max(0, int(local_total)))
                if total_i <= 0:
                    eta_f: float | None = 0.0
                else:
                    remaining = max(0, int(total_i) - int(processed))
                    rate = (float(processed) / elapsed) if elapsed > 0 else 0.0
                    eta_f = float(remaining) / float(rate) if rate > 0.0 else None
                    if done:
                        eta_f = 0.0
                self._planner_heartbeat_set(
                    stage_label=str(stage_label),
                    worker_id=int(worker_id),
                    tested=int(processed),
                    cached_hits=int(cached_hits_total),
                    total=int(total_i),
                    eta_sec=eta_f,
                    status=("done" if bool(done) else "running"),
                )

            _emit_worker_heartbeat(done=False)

            def _process_rank_batch(batch_ranks: list[int]) -> None:
                nonlocal tested_eval_total, cached_hits_total, processed_ranks, records, last_progress_ts
                if not batch_ranks:
                    return
                plan_batch: list[tuple[ConfigBundle, str, dict | None]] = []
                for rank in batch_ranks:
                    item = item_from_rank(int(rank))
                    if not (isinstance(item, tuple) and len(item) >= 2):
                        raise SystemExit(f"{stage_label} rank decoder returned invalid item for rank={int(rank)}")
                    cfg = item[0] if isinstance(item[0], ConfigBundle) else None
                    note = str(item[1] or "")
                    meta_item = item[2] if len(item) >= 3 and isinstance(item[2], dict) else None
                    if cfg is None:
                        raise SystemExit(f"{stage_label} rank decoder returned invalid cfg for rank={int(rank)}")
                    if not isinstance(meta_item, dict):
                        meta_item = {}
                    if "_mr_rank" not in meta_item:
                        meta_item = dict(meta_item)
                        meta_item["_mr_rank"] = int(rank)
                    plan_batch.append((cfg, note, meta_item))
                pending_plan, cached_hits, pending_cell_map, _prefetched = self._stage_partition_plan_by_cache(
                    stage_label=str(stage_label),
                    plan_all=plan_batch,
                    bars=bars,
                )
                if pending_plan:
                    ordered_indices = self._ordered_plan_indices_by_dimension_utility(
                        stage_label=str(stage_label),
                        plan_all=pending_plan,
                        bars=bars,
                    )
                    if ordered_indices and ordered_indices != list(range(len(pending_plan))):
                        pending_plan = [pending_plan[int(i)] for i in ordered_indices]
                        if pending_cell_map:
                            pending_cell_map = {
                                int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                                for new_idx, old_idx in enumerate(ordered_indices)
                                if isinstance(pending_cell_map.get(int(old_idx)), tuple) and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                            }
                    bound_indices, deferred_count = self._ordered_plan_indices_by_upper_bound(
                        stage_label=str(stage_label),
                        plan_all=pending_plan,
                        bars=bars,
                    )
                    if bound_indices and bound_indices != list(range(len(pending_plan))):
                        pending_plan = [pending_plan[int(i)] for i in bound_indices]
                        if pending_cell_map:
                            pending_cell_map = {
                                int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                                for new_idx, old_idx in enumerate(bound_indices)
                                if isinstance(pending_cell_map.get(int(old_idx)), tuple) and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                            }
                    if int(deferred_count) > 0:
                        print(
                            f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                            flush=True,
                        )
                tested_batch = 0
                kept_batch: list[tuple[ConfigBundle, dict, str, dict | None]] = []
                if pending_plan:
                    tested_batch, kept_batch = self._run_sweep(
                        plan=(pending_plan[idx] for idx in range(len(pending_plan))),
                        bars=bars,
                        total=len(pending_plan),
                        progress_label=f"{stage_label} worker {int(worker_id) + 1}/{int(workers)}",
                        report_every=0,
                        heartbeat_sec=float(heartbeat_eff),
                        record_milestones=False,
                        frontier_stage_label=str(stage_label),
                        progress_callback=lambda tested, _total, _kept, _elapsed, done: _emit_worker_heartbeat(
                            done=bool(done),
                            tested_override=int(processed_ranks) + int(tested),
                        ),
                    )
                if pending_cell_map:
                    evaluated_rows = [
                        (strategy_fp, axis_dim_fp, window_sig, "evaluated")
                        for idx, (
                            strategy_fp,
                            axis_dim_fp,
                            window_sig,
                        ) in pending_cell_map.items()
                        if int(idx) < len(pending_plan)
                    ]
                    if evaluated_rows:
                        self._stage_cell_status_set_many(stage_label=str(stage_label), rows=evaluated_rows)
                        self._cartesian_cell_manifest_set_many(
                            stage_label=str(stage_label),
                            rows=[(axis_dim_fp, window_sig, strategy_fp, "evaluated") for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows],
                        )
                tested_eval_total += int(tested_batch)
                cached_hits_total += int(len(cached_hits))
                processed_ranks += int(len(batch_ranks))
                for rec in self._worker_records_from_kept(kept_batch):
                    records.append(rec)
                for (
                    _cell_key,
                    cfg_cached,
                    row_cached,
                    note_cached,
                    _meta_cached,
                ) in cached_hits:
                    if isinstance(row_cached, dict):
                        records.append(self._encode_cfg_payload(cfg_cached, note=note_cached, extra={"row": row_cached}))

                rank_status: dict[int, str] = {int(rank): "pending" for rank in batch_ranks}
                cached_manifest_cells: list[tuple[str, str, int]] = []
                for cell_key, _cfg, _row, _note, meta_item in cached_hits:
                    rank_i = None
                    if isinstance(meta_item, dict):
                        try:
                            rank_i = int(meta_item.get("_mr_rank"))
                        except (TypeError, ValueError):
                            rank_i = None
                    if rank_i is None or rank_i not in rank_status:
                        continue
                    rank_status[int(rank_i)] = "cached_hit"
                    if isinstance(cell_key, tuple) and len(cell_key) == 3:
                        axis_dim_fp = str(cell_key[1] or "").strip()
                        window_sig = str(cell_key[2] or "").strip()
                        if axis_dim_fp and window_sig:
                            cached_manifest_cells.append((axis_dim_fp, window_sig, int(rank_i)))
                if cached_manifest_cells:
                    manifest_lookup = self._cartesian_cell_manifest_get_many(
                        stage_label=str(stage_label),
                        cells=[(axis_dim_fp, window_sig) for axis_dim_fp, window_sig, _rank in cached_manifest_cells],
                    )
                    for axis_dim_fp, window_sig, rank_i in cached_manifest_cells:
                        manifest_state = manifest_lookup.get((str(axis_dim_fp), str(window_sig)))
                        manifest_status = str(manifest_state[0]).strip().lower() if isinstance(manifest_state, tuple) and len(manifest_state) >= 1 else ""
                        if manifest_status == "dominated":
                            rank_status[int(rank_i)] = "dominated"
                for item in pending_plan:
                    meta_item = item[2] if isinstance(item, tuple) and len(item) >= 3 and isinstance(item[2], dict) else None
                    if not isinstance(meta_item, dict):
                        continue
                    try:
                        rank_i = int(meta_item.get("_mr_rank"))
                    except (TypeError, ValueError):
                        continue
                    if rank_i in rank_status:
                        rank_status[int(rank_i)] = "evaluated"
                rank_rows = self._compress_rank_status_rows(rank_status)
                if use_rank_manifest and rank_rows:
                    self._cartesian_rank_manifest_set_many(
                        stage_label=str(stage_label),
                        window_signature=str(manifest_window_signature),
                        rows=rank_rows,
                    )
                now = float(pytime.perf_counter())
                if (now - float(last_progress_ts)) >= float(max(5.0, heartbeat_eff)):
                    progress_total = int(max(0, int(claimed_ranks))) if bool(dynamic_claim_mode) else int(local_total)
                    print(
                        f"{stage_label} worker {int(worker_id) + 1}/{int(workers)} "
                        f"processed={int(processed_ranks)}/{int(progress_total)} "
                        f"eval={int(tested_eval_total)} cached={int(cached_hits_total)} kept={len(records)}",
                        flush=True,
                    )
                    last_progress_ts = float(now)
                _emit_worker_heartbeat(done=False)

            if bool(dynamic_claim_mode):
                claim_count = 0
                while True:
                    claimed = self._cartesian_rank_manifest_claim_next_range(
                        stage_label=str(stage_label),
                        window_signature=str(manifest_window_signature),
                        total=int(total_ranks),
                        max_span=int(claim_span_i),
                    )
                    if not (isinstance(claimed, tuple) and len(claimed) == 2):
                        break
                    claim_lo = int(claimed[0])
                    claim_hi = int(claimed[1])
                    if int(claim_hi) < int(claim_lo):
                        continue
                    claim_count += 1
                    claimed_ranks += max(0, int(claim_hi) - int(claim_lo) + 1)
                    if int(claim_count) <= 6:
                        print(
                            f"{stage_label} worker {int(worker_id) + 1}/{int(workers)} claim {int(claim_lo)}-{int(claim_hi)}",
                            flush=True,
                        )
                    rank_cursor = int(claim_lo)
                    batch_span_eff = max(int(batch_size_i), int(claim_span_i))
                    while int(rank_cursor) <= int(claim_hi):
                        batch_hi = min(int(claim_hi), int(rank_cursor) + int(batch_span_eff) - 1)
                        _process_rank_batch(list(range(int(rank_cursor), int(batch_hi) + 1)))
                        rank_cursor = int(batch_hi) + 1
            else:
                for rank_lo, rank_hi in worker_ranges:
                    lo_i = int(rank_lo)
                    hi_i = int(rank_hi)
                    rank_cursor = int(lo_i)
                    while int(rank_cursor) <= int(hi_i):
                        batch_hi = min(int(hi_i), int(rank_cursor) + int(batch_size_i) - 1)
                        _process_rank_batch(list(range(int(rank_cursor), int(batch_hi) + 1)))
                        rank_cursor = int(batch_hi) + 1

            _emit_worker_heartbeat(done=True)
            tested_total = int(tested_eval_total) + int(cached_hits_total)
            write_json(
                out_path,
                {"tested": int(tested_total), "kept": len(records), "records": records},
                sort_keys=False,
            )
            print(
                f"{stage_label} worker done tested={int(tested_total)} kept={len(records)} out={out_path}",
                flush=True,
            )

        if callable(plan_item_from_rank):
            total_i = int(plan_total or 0)
            window_sig = str(rank_manifest_window_signature or "").strip()
            if total_i <= 0:
                raise SystemExit(f"{stage_label} lazy-rank worker requires positive total rank count.")
            _run_sharded_stage_worker_lazy_rank(
                total_ranks=int(total_i),
                item_from_rank=plan_item_from_rank,
                manifest_window_signature=str(window_sig),
                batch_size=int(rank_batch_size),
            )
            return

        if plan_all is None:
            raise SystemExit(f"{stage_label} worker mode requires plan_all when lazy-rank mode is not active.")

        if not self.offline:
            raise SystemExit(f"{stage_label} worker mode requires --offline (avoid parallel IBKR sessions).")
        out_path_str = str(out_path_raw or "").strip()
        if not out_path_str:
            raise SystemExit(f"--{out_flag_name} is required for {stage_label} worker mode.")
        out_path = Path(out_path_str)

        worker_id, workers = _parse_worker_shard(worker_raw, workers_raw, label=str(stage_label))
        self._planner_heartbeat_set(
            stage_label=str(stage_label),
            worker_id=int(worker_id),
            tested=0,
            cached_hits=0,
            total=0,
            eta_sec=None,
            status="starting",
        )
        pending_plan, cached_hits, pending_cell_map, _prefetched_total = self._stage_partition_plan_by_cache(
            stage_label=str(stage_label),
            plan_all=plan_all,
            bars=bars,
        )
        if pending_plan:
            bound_ordered_indices, deferred_count = self._ordered_plan_indices_by_upper_bound(
                stage_label=str(stage_label),
                plan_all=pending_plan,
                bars=bars,
            )
            if bound_ordered_indices and bound_ordered_indices != list(range(len(pending_plan))):
                pending_plan = [pending_plan[int(i)] for i in bound_ordered_indices]
                if pending_cell_map:
                    pending_cell_map = {
                        int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                        for new_idx, old_idx in enumerate(bound_ordered_indices)
                        if isinstance(pending_cell_map.get(int(old_idx)), tuple) and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                    }
            if int(deferred_count) > 0:
                print(
                    f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                    flush=True,
                )
        total = len(pending_plan)
        warm_ranges = self._mixed_radix_warm_ranges(plan_all=pending_plan)
        if warm_ranges:
            preview = ", ".join(f"{int(lo)}-{int(hi)}:{int(count)}" for lo, hi, count in tuple(warm_ranges[:6]))
            if preview:
                print(f"{stage_label} warm unresolved rank bins {preview}", flush=True)
        cache_key = self._worker_plan_cache_key(stage_label=str(stage_label), workers=int(workers), plan_all=pending_plan)
        worker_buckets = self._worker_plan_cache_get(cache_key=str(cache_key))

        def _is_valid_worker_buckets(raw_buckets) -> bool:
            if not isinstance(raw_buckets, list):
                return False
            if len(raw_buckets) != int(workers):
                return False
            flat: list[int] = []
            for bucket in raw_buckets:
                if not isinstance(bucket, list):
                    return False
                for idx in bucket:
                    try:
                        i = int(idx)
                    except (TypeError, ValueError):
                        return False
                    if i < 0 or i >= int(total):
                        return False
                    flat.append(i)
            if len(flat) != int(total):
                return False
            return len(set(flat)) == int(total)

        if not _is_valid_worker_buckets(worker_buckets):
            worker_buckets = self._worker_bucketed_indices(
                plan_all=pending_plan,
                workers=int(workers),
                bars=bars,
                stage_label=str(stage_label),
                warm_ranges=warm_ranges,
            )
            if _is_valid_worker_buckets(worker_buckets):
                self._worker_plan_cache_set(cache_key=str(cache_key), buckets=worker_buckets)
        else:
            self.worker_plan_cache_hits += 1
        if int(worker_id) >= len(worker_buckets):
            raise SystemExit(f"Invalid {stage_label} worker shard: worker={worker_id} workers={workers}.")

        def _cached_owner(cell_key: tuple[str, str, str], workers_n: int) -> int:
            raw = "\x1f".join((str(cell_key[0]), str(cell_key[1]), str(cell_key[2])))
            h = hashlib.sha1(raw.encode("utf-8")).digest()
            return int.from_bytes(h[:4], byteorder="big", signed=False) % max(1, int(workers_n))

        cached_hits_local = [
            (cell_key, cfg, row, note, meta_item)
            for cell_key, cfg, row, note, meta_item in cached_hits
            if int(_cached_owner(cell_key, int(workers))) == int(worker_id)
        ]
        heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 30.0
        worker_indices = worker_buckets[int(worker_id)]
        local_total = len(worker_indices)
        cached_hits_total = int(len(cached_hits_local))
        worker_started_at = float(pytime.perf_counter())

        def _emit_worker_heartbeat(*, tested_live: int, done: bool = False) -> None:
            elapsed = max(0.0, float(pytime.perf_counter()) - float(worker_started_at))
            tested_i = int(max(0, int(tested_live)))
            total_i = int(max(0, int(local_total)))
            if total_i <= 0:
                eta_f: float | None = 0.0
            else:
                remaining = max(0, int(total_i) - int(tested_i))
                rate = (float(tested_i) / elapsed) if elapsed > 0 else 0.0
                eta_f = float(remaining) / float(rate) if rate > 0.0 else None
                if done:
                    eta_f = 0.0
            self._planner_heartbeat_set(
                stage_label=str(stage_label),
                worker_id=int(worker_id),
                tested=int(tested_i),
                cached_hits=int(cached_hits_total),
                total=int(total_i),
                eta_sec=eta_f,
                status=("done" if bool(done) else "running"),
            )

        _emit_worker_heartbeat(tested_live=0, done=False)
        shard_plan = (pending_plan[idx] for idx in worker_indices)
        tested, kept = self._run_sweep(
            plan=shard_plan,
            bars=bars,
            total=local_total,
            progress_label=f"{stage_label} worker {worker_id + 1}/{workers}",
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_eff),
            record_milestones=False,
            frontier_stage_label=str(stage_label),
            progress_callback=lambda tested, total, kept, elapsed, done: _emit_worker_heartbeat(
                tested_live=int(tested),
                done=bool(done),
            ),
        )
        _emit_worker_heartbeat(tested_live=int(tested), done=True)
        if pending_cell_map:
            evaluated_rows = [
                (strategy_fp, axis_dim_fp, window_sig, "evaluated")
                for idx in worker_indices
                for strategy_fp, axis_dim_fp, window_sig in ((pending_cell_map.get(int(idx)) or ("", "", "")),)
                if strategy_fp and axis_dim_fp and window_sig
            ]
            if evaluated_rows:
                self._stage_cell_status_set_many(stage_label=str(stage_label), rows=evaluated_rows)
                self._cartesian_cell_manifest_set_many(
                    stage_label=str(stage_label),
                    rows=[(axis_dim_fp, window_sig, strategy_fp, "evaluated") for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows],
                )
        records = self._worker_records_from_kept(kept)
        for _cell_key, cfg, row, note, _meta in cached_hits_local:
            if isinstance(row, dict):
                records.append(self._encode_cfg_payload(cfg, note=note, extra={"row": row}))
        tested_total = int(tested) + len(cached_hits_local)
        out_payload = {
            "tested": int(tested_total),
            "kept": len(records),
            "records": records,
        }
        write_json(out_path, out_payload, sort_keys=False)
        print(
            f"{stage_label} worker done tested={tested_total} kept={len(records)} out={out_path}",
            flush=True,
        )

    def _run_sharded_stage_worker(
        self,
        *,
        stage_label: str,
        worker_raw,
        workers_raw,
        out_path_raw: str,
        out_flag_name: str,
        plan_all,
        bars: list,
        report_every: int,
        heartbeat_sec: float = 0.0,
        plan_total: int | None = None,
        plan_item_from_rank=None,
        rank_manifest_window_signature: str = "",
        rank_batch_size: int = 384,
    ) -> None:
        self._run_sharded_stage_worker_kernel(
            stage_label=str(stage_label),
            worker_raw=worker_raw,
            workers_raw=workers_raw,
            out_path_raw=str(out_path_raw or ""),
            out_flag_name=str(out_flag_name),
            plan_all=plan_all,
            bars=bars,
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            plan_total=(int(plan_total) if plan_total is not None else None),
            plan_item_from_rank=plan_item_from_rank,
            rank_manifest_window_signature=str(rank_manifest_window_signature or ""),
            rank_batch_size=int(rank_batch_size),
        )

    def _decode_payload_schema_cfg_row(
        self,
        rec: dict,
        *,
        default_note: str,
        context: dict | None = None,
    ) -> tuple[ConfigBundle, dict, str] | None:
        decoded = self._decode_cfg_payload(rec, note_key="note", default_note=str(default_note))
        if decoded is None:
            return None
        cfg, note = decoded
        row = rec.get("row")
        if not isinstance(row, dict):
            return None
        row_out = dict(row)
        row_out["note"] = note
        return cfg, row_out, str(note)

    def _collect_stage_cfg_payload_rows(
        self,
        *,
        payloads: dict[int, dict],
        default_note: str,
        on_item=None,
    ) -> int:
        def _decode_record(rec: dict):
            return self._decode_payload_schema_cfg_row(rec, default_note=str(default_note), context=None)

        return _collect_parallel_payload_records(
            payloads=payloads,
            records_key="records",
            tested_key="tested",
            decode_record=_decode_record,
            on_record=on_item,
        )

    def _collect_stage_rows_from_payloads(
        self,
        *,
        payloads: dict[int, dict],
        default_note: str,
        on_row,
        dedupe_by_milestone_key: bool = False,
    ) -> int:
        seen_keys: set[str] | None = set() if bool(dedupe_by_milestone_key) else None

        def _on_item(item: tuple[ConfigBundle, dict, str] | None) -> None:
            if item is None:
                return
            cfg, row, note = item
            if seen_keys is not None:
                cfg_key = _milestone_key(cfg)
                if cfg_key in seen_keys:
                    return
                seen_keys.add(cfg_key)
            on_row(cfg, row, note)

        return self._collect_stage_cfg_payload_rows(
            payloads=payloads,
            default_note=str(default_note),
            on_item=_on_item,
        )

    def _run_stage_serial(
        self,
        *,
        stage_label: str,
        plan,
        bars: list,
        total: int,
        report_every: int,
        heartbeat_sec: float = 0.0,
        record_milestones: bool = True,
    ) -> tuple[int, list[tuple[ConfigBundle, dict, str]]]:
        tested, kept = self._run_sweep(
            plan=plan,
            bars=bars,
            total=total,
            progress_label=str(stage_label),
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=bool(record_milestones),
            frontier_stage_label=str(stage_label),
        )
        return int(tested), self._rows_from_kept(kept)

    def _prune_pending_plan_by_manifest(
        self,
        *,
        stage_label: str,
        pending_plan: list[tuple[ConfigBundle, str, dict | None]],
        pending_cell_map: dict[int, tuple[str, str, str]],
    ) -> tuple[
        list[tuple[ConfigBundle, str, dict | None]],
        dict[int, tuple[str, str, str]],
        int,
    ]:
        if not pending_plan or not pending_cell_map:
            return pending_plan, pending_cell_map, 0

        cells: list[tuple[str, str, str]] = []
        for idx in range(len(pending_plan)):
            cell_key = pending_cell_map.get(int(idx))
            if isinstance(cell_key, tuple) and len(cell_key) == 3:
                cells.append((str(cell_key[0]), str(cell_key[1]), str(cell_key[2])))
        if not cells:
            return pending_plan, pending_cell_map, 0

        prior_status = self._stage_cell_status_get_many(stage_label=str(stage_label), cells=cells)
        frontier_by_dim_window = self._stage_frontier_get_many(
            stage_label=str(stage_label),
            cells=[(axis_dim_fp, window_sig) for _strategy_fp, axis_dim_fp, window_sig in cells],
        )
        manifest_by_dim_window = self._cartesian_cell_manifest_get_many(
            stage_label=str(stage_label),
            cells=[(axis_dim_fp, window_sig) for _strategy_fp, axis_dim_fp, window_sig in cells],
        )

        kept_plan: list[tuple[ConfigBundle, str, dict | None]] = []
        kept_cell_map: dict[int, tuple[str, str, str]] = {}
        status_updates: list[tuple[str, str, str, str]] = []
        manifest_updates: list[tuple[str, str, str, str]] = []
        skipped = 0

        for idx, item in enumerate(pending_plan):
            cell_key = pending_cell_map.get(int(idx))
            if not (isinstance(cell_key, tuple) and len(cell_key) == 3):
                kept_plan.append(item)
                continue

            strategy_fp = str(cell_key[0])
            axis_dim_fp = str(cell_key[1])
            window_sig = str(cell_key[2])
            dim_window_key = (axis_dim_fp, window_sig)
            prune_manifest_status = ""
            prune_strategy_fp = strategy_fp

            prev_cell_status = str(prior_status.get((strategy_fp, axis_dim_fp, window_sig)) or "").strip().lower()
            if prev_cell_status in ("cached_hit", "evaluated"):
                prune_manifest_status = "cached_hit"
            else:
                frontier_row = frontier_by_dim_window.get(dim_window_key)
                if self._stage_frontier_is_dominated(frontier_row):
                    prune_manifest_status = "dominated"
                else:
                    manifest_state = manifest_by_dim_window.get(dim_window_key)
                    manifest_status = str(manifest_state[0]).strip().lower() if isinstance(manifest_state, tuple) and len(manifest_state) >= 1 else ""
                    manifest_strategy_fp = str(manifest_state[1]).strip() if isinstance(manifest_state, tuple) and len(manifest_state) >= 2 else ""
                    if manifest_status in ("dominated", "cached_hit", "evaluated"):
                        prune_manifest_status = "dominated" if manifest_status == "dominated" else "cached_hit"
                        if manifest_strategy_fp:
                            prune_strategy_fp = str(manifest_strategy_fp)

            if prune_manifest_status:
                skipped += 1
                self.run_calls_total += 1
                self._axis_progress_record(kept=False)
                status_updates.append((strategy_fp, axis_dim_fp, window_sig, "cached_hit"))
                manifest_updates.append((axis_dim_fp, window_sig, prune_strategy_fp, prune_manifest_status))
                continue

            new_idx = len(kept_plan)
            kept_plan.append(item)
            kept_cell_map[new_idx] = (strategy_fp, axis_dim_fp, window_sig)

        if status_updates:
            self._stage_cell_status_set_many(stage_label=str(stage_label), rows=status_updates)

        if manifest_updates:
            manifest_priority = {
                "pending": 0,
                "cached_hit": 1,
                "evaluated": 2,
                "dominated": 3,
            }
            filtered_updates: list[tuple[str, str, str, str]] = []
            for axis_dim_fp, window_sig, strategy_fp, status in manifest_updates:
                prev_state = manifest_by_dim_window.get((str(axis_dim_fp), str(window_sig)))
                prev_status = str(prev_state[0]).strip().lower() if isinstance(prev_state, tuple) else ""
                if prev_status == str(status):
                    continue
                if manifest_priority.get(str(status), -1) < manifest_priority.get(str(prev_status), -1):
                    continue
                filtered_updates.append((axis_dim_fp, window_sig, strategy_fp, status))
            if filtered_updates:
                self._cartesian_cell_manifest_set_many(stage_label=str(stage_label), rows=filtered_updates)

        if skipped > 0:
            print(
                f"{stage_label} pre-shard prune skipped={int(skipped)} remaining={len(kept_plan)}",
                flush=True,
            )
        return kept_plan, kept_cell_map, int(skipped)
