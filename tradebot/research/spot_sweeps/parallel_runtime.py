"""SweepParallelRuntime capability slice for the canonical spot research runtime."""

from __future__ import annotations

import sys
import tempfile
import threading
import time as pytime
from pathlib import Path
from ...backtest.sweep_fingerprint import _strategy_fingerprint
from ...backtest.sweep_parallel import (
    _progress_line,
    _run_parallel_stage_kernel,
    _strip_flags,
)
from ...backtest.sweeps import (
    write_json,
)
from .cli import (
    default_jobs as _default_jobs,
)
from .milestones import (
    _collect_milestone_items_from_payload,
    _merge_and_write_milestones,
)
from .parallel import _run_axis_subprocess_plan
from .support import (
    _cache_config,
    _registry_float,
    _tuned_parallel_jobs,
)


class SweepParallelRuntime:
    def _stage_parallel_base_cli(self, *, flags_with_values: tuple[str, ...]) -> list[str]:
        return _strip_flags(
            list(sys.argv[1:]),
            flags=("--write-milestones", "--merge-milestones"),
            flags_with_values=(
                "--axis",
                "--jobs",
                "--milestones-out",
                *flags_with_values,
            ),
        )

    def _compact_parallel_payload_cfg_refs(self, payload: dict) -> dict:
        catalog: dict[str, dict[str, object]] = {}

        def _compact(node):
            if isinstance(node, list):
                return [_compact(item) for item in node]
            if not isinstance(node, dict):
                return node
            strategy_payload = node.get("strategy")
            filters_payload = node.get("filters")
            if isinstance(strategy_payload, dict):
                filters_payload_norm = dict(filters_payload) if isinstance(filters_payload, dict) else None
                backtest_payload = node.get("backtest")
                if not isinstance(backtest_payload, dict):
                    backtest_payload = {}
                cfg_ref = _strategy_fingerprint(
                    strategy_payload,
                    filters=filters_payload_norm,
                    signal_bar_size=str(
                        backtest_payload.get("bar_size") or self.signal_bar_size
                    ),
                    signal_use_rth=bool(
                        backtest_payload.get("use_rth", self.use_rth)
                    ),
                )
                if cfg_ref not in catalog:
                    catalog[cfg_ref] = {
                        "cfg_ref": str(cfg_ref),
                        "strategy": dict(strategy_payload),
                        "filters": dict(filters_payload_norm) if isinstance(filters_payload_norm, dict) else None,
                    }
                out_item = {k: _compact(v) for k, v in node.items() if k not in ("strategy", "filters", "cfg_ref")}
                out_item["cfg_ref"] = str(cfg_ref)
                return out_item
            return {k: _compact(v) for k, v in node.items()}

        compact_payload = _compact(dict(payload))
        if catalog:
            compact_payload["_cfg_catalog"] = list(catalog.values())
        return compact_payload

    def _worker_name_to_worker_id(self, worker_name: str) -> int | None:
        raw = str(worker_name or "").strip()
        if not raw:
            return None
        token = raw.rsplit(":", 1)[-1]
        try:
            out = int(token)
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return int(out)

    def _planner_parallel_status_probe(
        self,
        *,
        stage_label: str,
        stage_total: int,
    ):
        cfg = _cache_config("planner_heartbeat")
        stale_after_sec = max(30.0, float(_registry_float(cfg.get("stale_after_sec"), 180.0)))
        bootstrap_grace_sec = max(
            30.0,
            float(_registry_float(cfg.get("bootstrap_grace_sec"), stale_after_sec)),
        )
        stage_started_at = float(pytime.time())

        def _probe(running_workers, pending_count: int) -> dict[str, object]:
            worker_ids: list[int] = []
            worker_id_by_name: dict[str, int] = {}
            elapsed_by_name: dict[str, float] = {}
            for item in list(running_workers or ()):
                if not (isinstance(item, tuple) and len(item) >= 2):
                    continue
                worker_name = str(item[0] or "").strip()
                if not worker_name:
                    continue
                try:
                    elapsed_sec = float(item[1] or 0.0)
                except (TypeError, ValueError):
                    elapsed_sec = 0.0
                worker_id = self._worker_name_to_worker_id(worker_name)
                if worker_id is None:
                    continue
                worker_ids.append(int(worker_id))
                worker_id_by_name[str(worker_name)] = int(worker_id)
                elapsed_by_name[str(worker_name)] = float(max(0.0, elapsed_sec))

            hb_rows = self._planner_heartbeat_get_many(stage_label=str(stage_label), worker_ids=worker_ids)
            tested_sum = 0
            cached_sum = 0
            eta_vals: list[float] = []
            stale_names: list[str] = []
            now_ts = float(pytime.time())
            for worker_name, worker_id in worker_id_by_name.items():
                row = hb_rows.get(int(worker_id))
                if isinstance(row, dict):
                    tested_sum += int(row.get("tested") or 0)
                    cached_sum += int(row.get("cached_hits") or 0)
                    eta_raw = row.get("eta_sec")
                    if eta_raw is not None:
                        try:
                            eta_f = float(eta_raw)
                        except (TypeError, ValueError):
                            eta_f = -1.0
                        if eta_f >= 0.0:
                            eta_vals.append(float(eta_f))
                    status_s = str(row.get("status") or "").strip().lower()
                    try:
                        last_seen = float(row.get("last_seen") or 0.0)
                    except (TypeError, ValueError):
                        last_seen = 0.0
                    age = max(0.0, float(now_ts - float(last_seen)))
                    if status_s != "done" and age >= float(stale_after_sec):
                        stale_names.append(str(worker_name))
                    continue

                elapsed_sec = float(elapsed_by_name.get(str(worker_name), 0.0))
                if elapsed_sec >= float(bootstrap_grace_sec) or (now_ts - stage_started_at) >= float(bootstrap_grace_sec):
                    stale_names.append(str(worker_name))

            if stale_names:
                self.planner_heartbeat_stale_candidates += int(len(stale_names))
            total_eff = max(0, int(stage_total))
            eta_max = max(eta_vals) if eta_vals else 0.0
            line = (
                f"{stage_label} planner heartbeat running={len(worker_id_by_name)} "
                f"pending={int(pending_count)} hb_rows={len(hb_rows)} "
                f"tested={int(tested_sum)}/{int(total_eff)} cached_hits={int(cached_sum)} "
                f"eta~{float(eta_max) / 60.0:0.1f}m stale={len(stale_names)}"
            )
            return {"line": line, "stale": tuple(stale_names)}

        return _probe

    def _run_parallel_stage_with_payload(
        self,
        *,
        axis_name: str,
        stage_label: str,
        total: int,
        jobs: int,
        payload: dict,
        payload_filename: str,
        temp_prefix: str,
        worker_tmp_prefix: str,
        worker_tag: str,
        out_prefix: str,
        stage_flag: str,
        worker_flag: str,
        workers_flag: str,
        out_flag: str,
        strip_flags_with_values: tuple[str, ...],
        run_min_trades_flag: str | None,
        run_min_trades: int | None,
        capture_error: str,
        failure_label: str,
        missing_label: str,
        invalid_label: str,
        planner_stage_label: str | None = None,
        prefetched_tested_if_empty: int = 0,
    ) -> dict[int, dict]:
        base_cli = self._stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        planner_stage_key = str(planner_stage_label or stage_label).strip().lower()
        planner_cfg = _cache_config("planner_heartbeat")
        monitor_interval_sec = max(5.0, float(_registry_float(planner_cfg.get("monitor_interval_sec"), 30.0)))
        max_stale_retries = max(0, int(_registry_float(planner_cfg.get("max_stale_retries"), 1.0)))
        probe = self._planner_parallel_status_probe(
            stage_label=str(planner_stage_key),
            stage_total=int(total),
        )
        self._planner_heartbeat_clear_stage(stage_label=str(planner_stage_key))
        if int(total) <= 0:
            prefetched_i = max(0, int(prefetched_tested_if_empty))
            print(
                f"{stage_label} parallel: unresolved=0 skip worker launch" + (f" prefetched={int(prefetched_i)}" if int(prefetched_i) > 0 else ""),
                flush=True,
            )
            if int(prefetched_i) <= 0:
                return {}
            return {0: {"tested": int(prefetched_i), "kept": 0, "records": []}}
        default_jobs_i = int(_default_jobs())
        jobs_tuned = _tuned_parallel_jobs(
            stage_label=str(stage_label),
            jobs_requested=int(jobs),
            total=int(total),
            default_jobs=int(default_jobs_i),
        )
        if int(jobs_tuned) != int(max(1, int(jobs))):
            print(
                f"{stage_label} jobs tuner: requested={int(max(1, int(jobs)))} tuned={int(jobs_tuned)} total={int(total)}",
                flush=True,
            )
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmpdir:
            payload_path = Path(tmpdir) / str(payload_filename)
            payload_compact = self._compact_parallel_payload_cfg_refs(payload)
            write_json(payload_path, payload_compact, sort_keys=False)
            _jobs_eff, payloads = _run_parallel_stage_kernel(
                stage_label=str(stage_label),
                jobs=int(jobs_tuned),
                total=int(total),
                default_jobs=int(default_jobs_i),
                offline=bool(self.offline),
                offline_error=f"--jobs>1 for {axis_name} requires --offline (avoid parallel IBKR sessions).",
                tmp_prefix=str(worker_tmp_prefix),
                worker_tag=str(worker_tag),
                out_prefix=str(out_prefix),
                build_cmd=lambda worker_id, workers_n, out_path: [
                    sys.executable,
                    "-u",
                    "-m",
                    "tradebot.backtest",
                    "spot",
                    *base_cli,
                    "--axis",
                    str(axis_name),
                    "--jobs",
                    "1",
                    str(stage_flag),
                    str(payload_path),
                    str(worker_flag),
                    str(worker_id),
                    str(workers_flag),
                    str(workers_n),
                    str(out_flag),
                    str(out_path),
                    *([str(run_min_trades_flag), str(int(run_min_trades))] if (run_min_trades_flag and run_min_trades is not None) else []),
                ],
                capture_error=str(capture_error),
                failure_label=str(failure_label),
                missing_label=str(missing_label),
                invalid_label=str(invalid_label),
                status_heartbeat_sec=float(monitor_interval_sec),
                worker_status_probe=probe,
                max_stale_retries=int(max_stale_retries),
            )
            return payloads

    def _run_parallel_stage(
        self,
        *,
        axis_name: str | None,
        stage_label: str,
        total: int,
        jobs: int,
        worker_tmp_prefix: str,
        worker_tag: str,
        out_prefix: str,
        worker_flag: str,
        workers_flag: str,
        out_flag: str,
        strip_flags_with_values: tuple[str, ...],
        capture_error: str,
        failure_label: str,
        missing_label: str,
        invalid_label: str,
        run_min_trades_flag: str | None = None,
        run_min_trades: int | None = None,
        stage_flag: str | None = None,
        stage_value: str | None = None,
        stage_args: tuple[str, ...] = (),
        entrypoint: tuple[str, ...] = ("-m", "tradebot.backtest", "spot"),
        planner_stage_label: str | None = None,
        prefetched_tested_if_empty: int = 0,
    ) -> dict[int, dict]:
        base_cli = self._stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        planner_stage_key = str(planner_stage_label or stage_label).strip().lower()
        planner_cfg = _cache_config("planner_heartbeat")
        monitor_interval_sec = max(5.0, float(_registry_float(planner_cfg.get("monitor_interval_sec"), 30.0)))
        max_stale_retries = max(0, int(_registry_float(planner_cfg.get("max_stale_retries"), 1.0)))
        probe = self._planner_parallel_status_probe(
            stage_label=str(planner_stage_key),
            stage_total=int(total),
        )
        self._planner_heartbeat_clear_stage(stage_label=str(planner_stage_key))
        if int(total) <= 0:
            prefetched_i = max(0, int(prefetched_tested_if_empty))
            print(
                f"{stage_label} parallel: unresolved=0 skip worker launch" + (f" prefetched={int(prefetched_i)}" if int(prefetched_i) > 0 else ""),
                flush=True,
            )
            if int(prefetched_i) <= 0:
                return {}
            return {0: {"tested": int(prefetched_i), "kept": 0, "records": []}}
        default_jobs_i = int(_default_jobs())
        jobs_tuned = _tuned_parallel_jobs(
            stage_label=str(stage_label),
            jobs_requested=int(jobs),
            total=int(total),
            default_jobs=int(default_jobs_i),
        )
        if int(jobs_tuned) != int(max(1, int(jobs))):
            print(
                f"{stage_label} jobs tuner: requested={int(max(1, int(jobs)))} tuned={int(jobs_tuned)} total={int(total)}",
                flush=True,
            )
        _jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label=str(stage_label),
            jobs=int(jobs_tuned),
            total=int(total),
            default_jobs=int(default_jobs_i),
            offline=bool(self.offline),
            offline_error=f"--jobs>1 for {axis_name or stage_label} requires --offline (avoid parallel IBKR sessions).",
            tmp_prefix=str(worker_tmp_prefix),
            worker_tag=str(worker_tag),
            out_prefix=str(out_prefix),
            build_cmd=lambda worker_id, workers_n, out_path: [
                sys.executable,
                "-u",
                *[str(p) for p in entrypoint],
                *base_cli,
                *(["--axis", str(axis_name)] if axis_name else []),
                "--jobs",
                "1",
                *([str(stage_flag), str(stage_value)] if (stage_flag and stage_value is not None) else ([str(stage_flag)] if stage_flag else [])),
                *[str(arg) for arg in stage_args],
                str(worker_flag),
                str(worker_id),
                str(workers_flag),
                str(workers_n),
                str(out_flag),
                str(out_path),
                *([str(run_min_trades_flag), str(int(run_min_trades))] if (run_min_trades_flag and run_min_trades is not None) else []),
            ],
            capture_error=str(capture_error),
            failure_label=str(failure_label),
            missing_label=str(missing_label),
            invalid_label=str(invalid_label),
            status_heartbeat_sec=float(monitor_interval_sec),
            worker_status_probe=probe,
            max_stale_retries=int(max_stale_retries),
        )
        return payloads

    def _collect_axis_milestone_items(
        self,
        *,
        milestone_payloads: dict[str, dict],
        milestone_axes: tuple[str, ...],
    ) -> list[dict]:
        eligible_new: list[dict] = []
        for axis_name in milestone_axes:
            payload = milestone_payloads.get(axis_name)
            if isinstance(payload, dict):
                eligible_new.extend(_collect_milestone_items_from_payload(payload, symbol=self.symbol))
        return eligible_new

    def _merge_axis_parallel_milestones(
        self,
        *,
        milestone_payloads: dict[str, dict],
        milestone_axes: tuple[str, ...],
    ) -> int | None:
        if not bool(self.args.write_milestones):
            return None
        eligible_new = self._collect_axis_milestone_items(
            milestone_payloads=milestone_payloads,
            milestone_axes=milestone_axes,
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
        print(f"Wrote {out_path} ({total} eligible presets).", flush=True)
        return int(total)

    def _run_parallel_axis_stage(
        self,
        *,
        label: str,
        axes: tuple[str, ...],
        jobs_req: int,
        axis_jobs_resolver,
        tmp_prefix: str,
        offline_error: str,
    ) -> dict[str, dict]:
        if not self.offline:
            raise SystemExit(str(offline_error))
        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags=("--merge-milestones",),
            flags_with_values=("--axis", "--jobs", "--milestones-out"),
        )
        return _run_axis_subprocess_plan(
            label=str(label),
            axes=axes,
            jobs=int(jobs_req),
            base_cli=base_cli,
            axis_jobs_resolver=axis_jobs_resolver,
            write_milestones=bool(self.args.write_milestones),
            tmp_prefix=str(tmp_prefix),
        )

    def _axis_plan_parts(self, axis_plan: list[tuple[str, str, bool]]) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
        axes = tuple(axis_name for axis_name, _profile, _emit in axis_plan)
        axis_profile_by_name = {axis_name: str(profile) for axis_name, profile, _emit in axis_plan}
        milestone_axes = tuple(axis_name for axis_name, _profile, emit in axis_plan if bool(emit))
        return axes, axis_profile_by_name, milestone_axes

    def _run_axis_plan_parallel_if_requested(
        self,
        *,
        axis_plan: list[tuple[str, str, bool]],
        jobs_req: int,
        label: str,
        tmp_prefix: str,
        offline_error: str,
    ) -> bool:
        if int(jobs_req) <= 1:
            return False
        axes, axis_profile_by_name, milestone_axes = self._axis_plan_parts(axis_plan)
        milestone_payloads = self._run_parallel_axis_stage(
            label=str(label),
            axes=axes,
            jobs_req=int(jobs_req),
            axis_jobs_resolver=lambda axis_name: min(int(jobs_req), int(_default_jobs()))
            if axis_profile_by_name.get(str(axis_name), "single") == "scaled"
            else 1,
            tmp_prefix=str(tmp_prefix),
            offline_error=str(offline_error),
        )
        if bool(self.args.write_milestones):
            self._merge_axis_parallel_milestones(
                milestone_payloads=milestone_payloads,
                milestone_axes=milestone_axes,
            )
            self.milestones_written = True
        return True

    def _run_axis_plan_serial(
        self,
        axis_plan: list[tuple[str, str, bool]],
        *,
        timed: bool = False,
    ) -> None:
        worker_stage_mode = bool(self.args.cfg_worker is not None or self.args.combo_full_cartesian_worker is not None)
        if bool(worker_stage_mode):
            print(
                "worker-stage mode: axis-level progress disabled; using worker/stage heartbeats.",
                flush=True,
            )

        def _run_axis_callable(axis_name: str, fn, *, timed_local: bool) -> None:
            before_calls = int(self.run_calls_total)
            t0 = pytime.perf_counter()
            total_hint = self._axis_total_hint(str(axis_name))
            total_hint_s = str(total_hint) if total_hint is not None else "?"
            axis_watchdog_stop = threading.Event()
            axis_watchdog_thread: threading.Thread | None = None
            axis_watchdog_sec = 30.0
            if bool(timed_local):
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            else:
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            axis_progress_enabled = not bool(worker_stage_mode)

            def _axis_watchdog() -> None:
                cadence = max(5.0, float(axis_watchdog_sec))
                while not axis_watchdog_stop.wait(cadence):
                    now = float(pytime.perf_counter())
                    if bool(axis_progress_enabled):
                        last_report = float(self.axis_progress_state.get("last_report") or 0.0)
                        # If inner axis progress already reported recently, suppress watchdog noise.
                        if float(last_report) > 0.0 and (now - float(last_report)) < float(cadence * 0.8):
                            continue
                    tested_live = max(0, int(self.run_calls_total) - int(before_calls))
                    if bool(axis_progress_enabled):
                        kept_live = int(self.axis_progress_state.get("kept") or 0)
                        total_live = (
                            int(self.axis_progress_state.get("total"))
                            if isinstance(self.axis_progress_state.get("total"), int)
                            else (int(total_hint) if isinstance(total_hint, int) else None)
                        )
                    else:
                        kept_live = 0
                        total_live = int(total_hint) if isinstance(total_hint, int) else None
                    line = _progress_line(
                        label=f"{axis_name} watchdog",
                        tested=int(tested_live),
                        total=total_live,
                        kept=int(kept_live),
                        started_at=float(t0),
                        rate_unit="cfg/s",
                    )
                    line += " heartbeat=axis"
                    print(line, flush=True)

            axis_watchdog_thread = threading.Thread(target=_axis_watchdog, daemon=True)
            axis_watchdog_thread.start()
            if bool(axis_progress_enabled):
                self._axis_progress_begin(axis_name=str(axis_name))
            try:
                fn()
            finally:
                axis_watchdog_stop.set()
                if axis_watchdog_thread is not None:
                    try:
                        axis_watchdog_thread.join(timeout=1.0)
                    except Exception:
                        pass
                if bool(axis_progress_enabled):
                    self._axis_progress_end()
            elapsed = pytime.perf_counter() - t0
            tested = int(self.run_calls_total) - int(before_calls)
            print(f"DONE  {axis_name} tested={tested} elapsed={elapsed:0.1f}s", flush=True)
            print("", flush=True)

        for axis_name, _profile, _emit in axis_plan:
            fn_obj = self.axis_registry.get(str(axis_name))
            fn = fn_obj if callable(fn_obj) else None
            if fn is None:
                continue
            _run_axis_callable(str(axis_name), fn, timed_local=bool(timed))
