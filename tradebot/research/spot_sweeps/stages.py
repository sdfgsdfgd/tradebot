"""SweepStages capability slice for the canonical spot research runtime."""

from __future__ import annotations

import json
from pathlib import Path
from ...backtest.config import (
    ConfigBundle,
)
from .milestones import _milestone_key
from .support import (
    _claim_first_serial_force_worker_enabled,
    _claim_first_stage_enabled,
)


class SweepStages:
    def _run_stage_cfg_rows(
        self,
        *,
        stage_label: str,
        total: int,
        jobs_req: int,
        bars: list,
        report_every: int,
        on_row,
        serial_plan=None,
        serial_plan_builder=None,
        heartbeat_sec: float = 0.0,
        parallel_payloads_builder=None,
        parallel_payload_supports_plan: bool = False,
        parallel_default_note: str = "",
        parallel_dedupe_by_milestone_key: bool = True,
        record_milestones: bool = True,
    ) -> int:
        def _on_row_local(cfg: ConfigBundle, row: dict, note: str) -> None:
            if bool(record_milestones):
                self._record_milestone(cfg, row, note)
            on_row(cfg, row, note)

        claim_first_serial_worker = bool(
            int(jobs_req) <= 1
            and bool(self.offline)
            and callable(parallel_payloads_builder)
            and _claim_first_serial_force_worker_enabled()
            and _claim_first_stage_enabled(
                stage_label=str(stage_label), total=int(total)
            )
        )
        if bool(claim_first_serial_worker):
            print(
                f"{stage_label} claim-first planner: using worker-claim path for serial run (total={int(total)})",
                flush=True,
            )

        needs_serial_plan = (
            (int(jobs_req) <= 1 and not bool(claim_first_serial_worker))
            or bool(parallel_payload_supports_plan)
            or not callable(parallel_payloads_builder)
        )
        serial_plan_eff_raw = (
            (serial_plan_builder() if callable(serial_plan_builder) else serial_plan)
            if bool(needs_serial_plan)
            else None
        )
        serial_plan_eff = list(serial_plan_eff_raw or ()) if bool(needs_serial_plan) else []

        prefilter_here = bool(
            (int(jobs_req) <= 1 and not bool(claim_first_serial_worker))
            or bool(parallel_payload_supports_plan)
        )
        prefetched_tested = 0
        pending_plan = serial_plan_eff
        pending_cell_map: dict[int, tuple[str, str, str]] = {}

        if prefilter_here and serial_plan_eff:
            pending_plan, cached_hits, pending_cell_map, prefetched_cache_tested = (
                self._stage_partition_plan_by_cache(
                    stage_label=str(stage_label),
                    plan_all=serial_plan_eff,
                    bars=bars,
                )
            )
            prefetched_tested += int(prefetched_cache_tested)
            for _cell_key, cfg, row, note, _meta_item in cached_hits:
                if isinstance(row, dict):
                    _on_row_local(cfg, row, note)
            pending_plan, pending_cell_map, pruned_here = (
                self._prune_pending_plan_by_manifest(
                    stage_label=str(stage_label),
                    pending_plan=list(pending_plan),
                    pending_cell_map=dict(pending_cell_map),
                )
            )
            prefetched_tested += int(pruned_here)
            if pending_plan:
                ordered_indices = self._ordered_plan_indices_by_dimension_utility(
                    stage_label=str(stage_label),
                    plan_all=pending_plan,
                    bars=bars,
                )
                if ordered_indices and ordered_indices != list(
                    range(len(pending_plan))
                ):
                    pending_plan = [pending_plan[int(i)] for i in ordered_indices]
                    if pending_cell_map:
                        pending_cell_map = {
                            int(new_idx): pending_cell_map.get(
                                int(old_idx), ("", "", "")
                            )
                            for new_idx, old_idx in enumerate(ordered_indices)
                            if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                            and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                        }
            if pending_plan:
                bound_ordered_indices, deferred_count = (
                    self._ordered_plan_indices_by_upper_bound(
                        stage_label=str(stage_label),
                        plan_all=pending_plan,
                        bars=bars,
                    )
                )
                if bound_ordered_indices and bound_ordered_indices != list(
                    range(len(pending_plan))
                ):
                    pending_plan = [pending_plan[int(i)] for i in bound_ordered_indices]
                    if pending_cell_map:
                        pending_cell_map = {
                            int(new_idx): pending_cell_map.get(
                                int(old_idx), ("", "", "")
                            )
                            for new_idx, old_idx in enumerate(bound_ordered_indices)
                            if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                            and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                        }
                if int(deferred_count) > 0:
                    print(
                        f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                        flush=True,
                    )

        parallel_total = (
            len(pending_plan)
            if bool(parallel_payload_supports_plan)
            else (len(serial_plan_eff) if bool(needs_serial_plan) else int(total))
        )
        if (
            (int(jobs_req) > 1 or bool(claim_first_serial_worker))
            and int(parallel_total) > 0
            and callable(parallel_payloads_builder)
        ):
            if bool(parallel_payload_supports_plan):
                payloads = parallel_payloads_builder(pending_plan)
            else:
                payloads = parallel_payloads_builder()
            tested_parallel = self._collect_stage_rows_from_payloads(
                payloads=payloads,
                default_note=str(parallel_default_note or stage_label),
                on_row=_on_row_local,
                dedupe_by_milestone_key=bool(parallel_dedupe_by_milestone_key),
            )
            self.run_calls_total += int(tested_parallel)
            return int(prefetched_tested) + int(tested_parallel)

        tested, serial_rows = self._run_stage_serial(
            stage_label=str(stage_label),
            plan=pending_plan,
            bars=bars,
            total=len(pending_plan),
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=bool(record_milestones),
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
                self._stage_cell_status_set_many(
                    stage_label=str(stage_label), rows=evaluated_rows
                )
                self._cartesian_cell_manifest_set_many(
                    stage_label=str(stage_label),
                    rows=[
                        (axis_dim_fp, window_sig, strategy_fp, "evaluated")
                        for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows
                    ],
                )
        for cfg, row, note in serial_rows:
            on_row(cfg, row, note)
        return int(prefetched_tested) + int(tested)

    def _run_cfg_pairs_grid(
        self,
        *,
        axis_tag: str,
        cfg_pairs: list[tuple[ConfigBundle, str]],
        rows: list[dict],
        on_row=None,
        report_every: int = 0,
        heartbeat_sec: float = 0.0,
    ) -> int:
        unique_pairs: dict[str, tuple[ConfigBundle, str]] = {}
        for cfg, note in cfg_pairs:
            unique_pairs.setdefault(_milestone_key(cfg), (cfg, str(note)))
        cfg_pairs = list(unique_pairs.values())
        bar_sizes = {str(cfg.backtest.bar_size) for cfg, _note in cfg_pairs}
        if len(bar_sizes) > 1:
            raise SystemExit(f"{axis_tag} cfg grid mixes bar sizes: {', '.join(sorted(bar_sizes))}")
        plan_bar_size = next(iter(bar_sizes), str(self.signal_bar_size))

        def _capture(cfg: ConfigBundle, row: dict, note: str) -> None:
            rows.append(row)
            if callable(on_row):
                on_row(cfg, row, note)

        if (
            not self.args.cfg_stage
            and bool(self.axis_progress_state.get("active"))
            and str(self.axis_progress_state.get("axis_key") or "")
            == str(axis_tag).strip().lower()
        ):
            self.axis_progress_state["total"] = len(cfg_pairs)

        if self.args.cfg_stage:
            payload_path = Path(str(self.args.cfg_stage))
            try:
                payload_raw = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid cfg_stage payload JSON: {payload_path}"
                ) from exc
            if not isinstance(payload_raw, dict):
                raise SystemExit(
                    f"Invalid cfg_stage payload: expected object ({payload_path})"
                )
            payload_axis = str(payload_raw.get("axis_tag") or "").strip().lower()
            if payload_axis and payload_axis != str(axis_tag).strip().lower():
                raise SystemExit(
                    f"cfg_pairs worker payload axis mismatch: expected {axis_tag} got {payload_axis} ({payload_path})"
                )
            cfg_records_raw = payload_raw.get("cfgs")
            if not isinstance(cfg_records_raw, list):
                raise SystemExit(
                    f"cfg_stage payload missing 'cfgs' list: {payload_path}"
                )
            cfg_records = [
                dict(rec) for rec in cfg_records_raw if isinstance(rec, dict)
            ]
            cfg_catalog = self._cfg_catalog_from_payload(payload_raw)

            def _cfg_item_from_rank(rank: int) -> tuple[ConfigBundle, str, dict]:
                rank_i = int(rank)
                if rank_i < 0 or rank_i >= len(cfg_records):
                    raise SystemExit(
                        f"cfg_stage rank out of bounds: rank={rank_i} total={len(cfg_records)}"
                    )
                decoded = self._decode_cfg_payload(
                    cfg_records[rank_i],
                    note_key="note",
                    cfg_catalog=cfg_catalog,
                )
                if decoded is None:
                    raise SystemExit(f"cfg_stage rank decode failed: rank={rank_i}")
                cfg_obj, note = decoded
                return cfg_obj, str(note), {"_mr_rank": int(rank_i)}

            worker_bar_size = (
                str(_cfg_item_from_rank(0)[0].backtest.bar_size)
                if cfg_records
                else str(self.signal_bar_size)
            )
            self._run_sharded_stage_worker(
                stage_label=str(axis_tag),
                worker_raw=self.args.cfg_worker,
                workers_raw=self.args.cfg_workers,
                out_path_raw=str(self.args.cfg_out or ""),
                out_flag_name="cfg-out",
                plan_all=None,
                bars=self._bars_cached(worker_bar_size),
                report_every=max(1, int(report_every)),
                heartbeat_sec=float(heartbeat_sec),
                plan_total=len(cfg_records),
                plan_item_from_rank=_cfg_item_from_rank,
                rank_batch_size=256,
            )
            return -1

        bars_stage = self._bars_cached(plan_bar_size)
        plan_all: list[tuple[ConfigBundle, str, None]] = [
            (cfg, str(note), None) for cfg, note in cfg_pairs
        ]

        def _cfg_pairs_parallel_payloads(pending_plan_items) -> dict[int, dict]:
            pending_cfgs = list(pending_plan_items or ())
            return self._run_parallel_stage_with_payload(
                axis_name=str(self.args.axis),
                stage_label=str(axis_tag),
                total=len(pending_cfgs),
                jobs=int(self.jobs),
                payload={
                    "axis_tag": str(axis_tag),
                    "cfgs": [
                        self._encode_cfg_payload(cfg, note=note)
                        for cfg, note, _meta in pending_cfgs
                    ],
                },
                payload_filename="cfg_pairs_payload.json",
                temp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_",
                worker_tmp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_worker_",
                worker_tag=f"cfgpairs:{str(axis_tag)}",
                out_prefix="cfg_pairs_out",
                stage_flag="--cfg-stage",
                worker_flag="--cfg-worker",
                workers_flag="--cfg-workers",
                out_flag="--cfg-out",
                strip_flags_with_values=(
                    "--start",
                    "--end",
                    "--base",
                    "--seed-milestones",
                    "--stability-window",
                    "--stability-top",
                    "--stability-write-top",
                    "--stability-min-trades-per-year",
                    "--stability-out",
                    "--promotion-objective",
                    "--promotion-version",
                    "--cfg-stage",
                    "--cfg-worker",
                    "--cfg-workers",
                    "--cfg-out",
                    "--combo-full-cartesian-run-min-trades",
                ),
                strip_flags=("--promote",),
                stage_args=(
                    "--start",
                    self.start.isoformat(),
                    "--end",
                    self.end.isoformat(),
                    "--base",
                    "default",
                ),
                run_min_trades_flag="--combo-full-cartesian-run-min-trades",
                run_min_trades=int(self.run_min_trades),
                capture_error=f"Failed to capture {axis_tag} worker stdout.",
                failure_label=f"{axis_tag} worker",
                missing_label=str(axis_tag),
                invalid_label=str(axis_tag),
            )

        tested_total = self._run_stage_cfg_rows(
            stage_label=str(axis_tag),
            total=len(plan_all),
            jobs_req=int(self.jobs),
            bars=bars_stage,
            report_every=max(1, int(report_every)),
            heartbeat_sec=float(heartbeat_sec),
            on_row=_capture,
            serial_plan=plan_all,
            parallel_payloads_builder=_cfg_pairs_parallel_payloads,
            parallel_payload_supports_plan=True,
            parallel_default_note=str(axis_tag),
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        return int(tested_total)
