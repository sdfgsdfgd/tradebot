"""SweepCartesian capability slice for the canonical spot research runtime."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from .combo_presets import ComboPresetContext
from .catalog import (
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
    _combo_full_preset_axes,
    _combo_full_preset_key,
    _combo_full_preset_tier,
)
from .dimensions import _AXIS_DIMENSION_REGISTRY
from .fingerprints import (
    _RUN_CFG_CACHE_ENGINE_VERSION,
)
from .milestones import (
    _print_leaderboards,
)
from .support import (
    _require_offline_cache_or_die,
)


class SweepCartesian:
    def _combo_full_context(self, preset: str | None = None) -> ComboPresetContext:
        """Build the single search-space authority shared by progress and execution."""

        dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
        if not isinstance(dims, dict):
            raise SystemExit("combo_full dimension registry missing: combo_full_cartesian_tight")
        requested_preset = getattr(self.args, "combo_full_preset", "") if preset is None else preset
        combo_full_preset = _combo_full_preset_key(str(requested_preset or ""))
        valid_combo_presets = set(_combo_full_preset_axes())
        if combo_full_preset and combo_full_preset not in valid_combo_presets:
            raise SystemExit(f"Unknown combo_full preset: {combo_full_preset!r}")

        def _pairs_from_registry(
            *,
            dim_name: str,
            variants_key: str,
        ) -> list[tuple[str, dict[str, object]]]:
            out: list[tuple[str, dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    out.append((label, dict(payload)))
            if not out:
                raise SystemExit(f"combo_full requires at least one {dim_name} variant.")
            return out

        def _timing_profiles_from_registry(
            *,
            variants_key: str,
        ) -> list[tuple[str, dict[str, object], dict[str, object]]]:
            out: list[tuple[str, dict[str, object], dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    strat_over = payload.get("strategy_overrides")
                    filt_over = payload.get("filter_overrides")
                    strat_dict = dict(strat_over) if isinstance(strat_over, dict) else {}
                    filt_dict = dict(filt_over) if isinstance(filt_over, dict) else {}
                    out.append((str(label), strat_dict, filt_dict))
            return out

        pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]] = {
            str(dim_name): _pairs_from_registry(
                dim_name=str(dim_name),
                variants_key=str(variants_key),
            )
            for dim_name, variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }
        if not bool(getattr(self.args, "combo_full_include_tick", False)):
            tick_rows = [
                (str(label), dict(payload))
                for label, payload in tuple(pair_variants_by_dim.get("tick") or ())
                if str((payload or {}).get("tick_gate_mode") or "off").strip().lower() in ("off", "", "none", "false", "0")
            ]
            if not tick_rows:
                tick_rows = [("tick=off", {"tick_gate_mode": "off"})]
            pair_variants_by_dim["tick"] = tick_rows
        confirm_bars = [int(v) for v in tuple(dims.get("confirm_bars") or ())]
        timing_profile_variants = _timing_profiles_from_registry(variants_key="timing_profile_variants")
        if not timing_profile_variants:
            timing_profile_variants = [("timing=base", {}, {})]
        short_mults = [float(v) for v in tuple(dims.get("short_mults") or ())]
        if not confirm_bars or not short_mults:
            raise SystemExit("combo_full requires non-empty confirm_bars and short_mults.")

        dim_state: dict[str, list[object]] = {
            "timing_profile": list(timing_profile_variants),
            "confirm": list(confirm_bars),
            **{str(dim_name): list(rows) for dim_name, rows in pair_variants_by_dim.items()},
            "short_mult": list(short_mults),
        }
        preset_context = ComboPresetContext(self, dims, dim_state, _timing_profiles_from_registry)

        preset_context.apply(combo_full_preset)
        return preset_context

    def _sweep_combo_full(self) -> None:
        """Unified tight Cartesian sweep over centralized combo dimensions."""

        bars_sig = self._bars_cached(self.signal_bar_size)
        combo_full_preset = _combo_full_preset_key(str(getattr(self.args, "combo_full_preset", "") or ""))
        preset_context = self._combo_full_context(combo_full_preset)
        pair_variants_by_dim = {
            dim_name: list(preset_context.rows[dim_name])
            for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }

        requires_tick_daily = any(
            str((payload or {}).get("tick_gate_mode") or "off").strip().lower() != "off" for _label, payload in pair_variants_by_dim["tick"]
        )
        if self.offline and requires_tick_daily:
            tick_warm_start = self.start_dt - timedelta(days=400)
            tick_ok = False
            for tick_sym in ("TICK-AMEX", "TICK-NYSE"):
                try:
                    _require_offline_cache_or_die(
                        data=self.data,
                        cache_dir=self.cache_dir,
                        symbol=tick_sym,
                        exchange=None,
                        start_dt=tick_warm_start,
                        end_dt=self.end_dt,
                        bar_size="1 day",
                        use_rth=True,
                        cache_policy=self.cache_policy,
                    )
                    tick_ok = True
                    break
                except SystemExit:
                    continue
            if not tick_ok:
                raise SystemExit(
                    "combo_full requires cached daily $TICK bars when running with --offline "
                    "(expected under db/TICK-AMEX or db/TICK-NYSE). Run once without --offline to fetch."
                )

        size_by_dim = preset_context.size_by_dim
        total = preset_context.total
        if total <= 0:
            raise SystemExit("combo_full has empty Cartesian dimensions.")
        base = preset_context.base
        ordered_dims = preset_context.ordered_dims
        combo_dim_space_sig = preset_context.dimension_signature

        def _combo_full_worker_stage_window_signature() -> str:
            raw = {
                "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
                "stage": "combo_full_cartesian",
                "symbol": str(self.symbol),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "signal_bar_size": str(self.signal_bar_size),
                "use_rth": bool(self.use_rth),
                "run_min_trades": int(self.run_min_trades),
                "preset": str(combo_full_preset or ""),
                "ordered_dims": tuple(str(v) for v in ordered_dims),
                "size_by_dim": tuple((str(k), int(v)) for k, v in size_by_dim.items()),
                "dim_space_sig": str(combo_dim_space_sig),
                "bars_sig": self._bars_signature(bars_sig),
            }
            return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        if self.args.combo_full_cartesian_stage:
            self._run_sharded_stage_worker(
                stage_label="combo_full_cartesian",
                worker_raw=self.args.combo_full_cartesian_worker,
                workers_raw=self.args.combo_full_cartesian_workers,
                out_path_raw=str(self.args.combo_full_cartesian_out or ""),
                out_flag_name="combo-full-cartesian-out",
                plan_all=None,
                bars=bars_sig,
                report_every=0,
                heartbeat_sec=30.0,
                plan_total=int(total),
                plan_item_from_rank=preset_context.plan_item_from_rank,
                rank_manifest_window_signature=_combo_full_worker_stage_window_signature(),
                rank_batch_size=2048,
            )
            return

        print("")
        print("=== combo_full: unified tight Cartesian core ===")
        if combo_full_preset:
            print(
                f"combo_full preset active: {combo_full_preset} (tier={_combo_full_preset_tier(str(combo_full_preset))})",
                flush=True,
            )
        print(
            "combo_full dimensions: total="
            f"{int(total)} " + " ".join(f"{str(dim_name)}={int(size_by_dim.get(str(dim_name), 0) or 0)}" for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER),
            flush=True,
        )
        print(
            f"combo_full sharding order: {','.join(ordered_dims)}",
            flush=True,
        )
        print("")

        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        rows: list[dict] = []
        combo_stage_args = tuple(("--combo-full-preset", str(combo_full_preset)) if combo_full_preset else ())
        combo_manifest_window_sig = _combo_full_worker_stage_window_signature()

        def _combo_full_parallel_totals() -> tuple[int, int]:
            try:
                unresolved_ranges = self._cartesian_rank_manifest_unresolved_ranges(
                    stage_label="combo_full_cartesian",
                    window_signature=str(combo_manifest_window_sig),
                    total=int(total),
                )
                unresolved_total = sum(max(0, int(rank_hi) - int(rank_lo) + 1) for rank_lo, rank_hi in tuple(unresolved_ranges))
            except Exception:
                unresolved_total = int(total)
            unresolved_i = max(0, int(unresolved_total))
            prefetched_i = max(0, int(total) - int(unresolved_i))
            return int(unresolved_i), int(prefetched_i)

        tested_total = self._run_stage_cfg_rows(
            stage_label="combo_full_cartesian",
            total=int(total),
            jobs_req=int(self.jobs),
            bars=bars_sig,
            report_every=200,
            heartbeat_sec=30.0,
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan_builder=preset_context.iter_plan,
            parallel_payloads_builder=lambda: (
                lambda unresolved_i, prefetched_i: self._run_parallel_stage(
                    axis_name="combo_full",
                    stage_label="combo_full Cartesian",
                    total=int(unresolved_i),
                    jobs=int(self.jobs),
                    worker_tmp_prefix="tradebot_combo_full_cartesian_",
                    worker_tag="cfc",
                    out_prefix="combo_full_cartesian_out",
                    stage_flag="--combo-full-cartesian-stage",
                    stage_value="1",
                    worker_flag="--combo-full-cartesian-worker",
                    workers_flag="--combo-full-cartesian-workers",
                    out_flag="--combo-full-cartesian-out",
                    strip_flags_with_values=(
                        "--combo-full-cartesian-stage",
                        "--combo-full-cartesian-worker",
                        "--combo-full-cartesian-workers",
                        "--combo-full-cartesian-out",
                        "--combo-full-cartesian-run-min-trades",
                        "--combo-full-preset",
                    ),
                    run_min_trades_flag="--combo-full-cartesian-run-min-trades",
                    run_min_trades=int(self.run_min_trades),
                    stage_args=combo_stage_args,
                    capture_error="Failed to capture combo_full Cartesian worker stdout.",
                    failure_label="combo_full Cartesian worker",
                    missing_label="combo_full Cartesian",
                    invalid_label="combo_full Cartesian",
                    planner_stage_label="combo_full_cartesian",
                    prefetched_tested_if_empty=int(prefetched_i),
                )
            )(*_combo_full_parallel_totals()),
            parallel_default_note="combo_full Cartesian",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        if base_row:
            rows.append(base_row)
        print(
            f"combo_full Cartesian tested={int(tested_total)} kept={len(rows)} min_trades={int(self.run_min_trades)}",
            flush=True,
        )
        _print_leaderboards(
            rows,
            title="combo_full sweep (unified tight Cartesian)",
            top_n=int(self.args.top),
        )
