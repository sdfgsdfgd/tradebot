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
from .profiles import (
    _PERM_JOINT_PROFILE,
)
from .milestones import (
    _print_leaderboards,
)
from .risk import (
    _risk_pack_riskoff,
    _risk_pack_riskpanic,
    _risk_pack_riskpop,
)
from .support import (
    _require_offline_cache_or_die,
)


class SweepCartesian:
    def _sweep_frontier(self) -> None:
        """Summarize the current milestones set as a multi-objective frontier."""
        groups = self.milestones.get("groups", []) if isinstance(self.milestones, dict) else []
        rows: list[dict] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            entries = group.get("entries") or []
            if not entries or not isinstance(entries, list):
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
            if str(strat.get("signal_bar_size") or "").strip().lower() != str(self.signal_bar_size).strip().lower():
                continue
            if bool(strat.get("signal_use_rth")) != bool(self.use_rth):
                continue
            if str(entry.get("symbol") or "").strip().upper() != str(self.symbol).strip().upper():
                continue
            try:
                trades = int(metrics.get("trades") or 0)
                win = float(metrics.get("win_rate") or 0.0)
                pnl = float(metrics.get("pnl") or 0.0)
                dd = float(metrics.get("max_drawdown") or 0.0)
                pnl_dd = metrics.get("pnl_over_dd")
                pnl_over_dd = float(pnl_dd) if pnl_dd is not None else (pnl / dd if dd > 0 else None)
            except (TypeError, ValueError):
                continue
            note = str(group.get("name") or "").strip() or "milestone"
            rows.append(
                {
                    "trades": trades,
                    "win_rate": win,
                    "pnl": pnl,
                    "dd": dd,
                    "pnl_over_dd": pnl_over_dd,
                    "note": note,
                }
            )

        if not rows:
            print("No matching spot milestones found for this bar_size/symbol.")
            return

        _print_leaderboards(
            rows,
            title="Milestones frontier (current presets)",
            top_n=int(self.args.top),
        )

        print("")
        print("Frontier by win-rate constraint (best pnl):")
        for thr in (0.55, 0.58, 0.60, 0.62, 0.65):
            eligible = [r for r in rows if int(r.get("trades") or 0) >= int(self.run_min_trades) and float(r.get("win_rate") or 0.0) >= thr]
            if not eligible:
                continue
            best = max(eligible, key=lambda r: float(r.get("pnl") or float("-inf")))
            print(
                f"- win>={thr:.2f}: pnl={best['pnl']:.1f} pnl/dd={(best['pnl_over_dd'] or 0):.2f} "
                f"win={best['win_rate'] * 100:.1f}% tr={best['trades']} note={best.get('note')}"
            )

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

        direction_catalog_default: dict[str, dict[str, object]] = {
            "ema=2/4 cross": {
                "entry_signal": "ema",
                "ema_preset": "2/4",
                "ema_entry_mode": "cross",
            },
            "ema=4/9 cross": {
                "entry_signal": "ema",
                "ema_preset": "4/9",
                "ema_entry_mode": "cross",
            },
        }
        regime_catalog_default: dict[str, dict[str, object]] = {
            "regime=ST(4h:7,0.5,hl2)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "4 hours",
                "supertrend_atr_period": 7,
                "supertrend_multiplier": 0.5,
                "supertrend_source": "hl2",
            },
            "regime=ST(1d:14,1.0,hl2)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "1 day",
                "supertrend_atr_period": 14,
                "supertrend_multiplier": 1.0,
                "supertrend_source": "hl2",
            },
            "regime=ST(4h:10,0.8,close)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "4 hours",
                "supertrend_atr_period": 10,
                "supertrend_multiplier": 0.8,
                "supertrend_source": "close",
            },
        }
        regime2_catalog_default: dict[str, dict[str, object]] = {
            "r2=off": {"regime2_mode": "off", "regime2_bar_size": None},
            "r2=ST(4h:3,0.25,close)": {
                "regime2_mode": "supertrend",
                "regime2_bar_size": "4 hours",
                "regime2_supertrend_atr_period": 3,
                "regime2_supertrend_multiplier": 0.25,
                "regime2_supertrend_source": "close",
            },
        }
        exit_catalog_default: dict[str, dict[str, object]] = {
            "exit=pct(0.015,0.03)": {
                "spot_exit_mode": "pct",
                "spot_profit_target_pct": 0.015,
                "spot_stop_loss_pct": 0.03,
                "spot_pt_atr_mult": None,
                "spot_sl_atr_mult": None,
            },
            "exit=stop_only(0.03)": {
                "spot_exit_mode": "pct",
                "spot_profit_target_pct": None,
                "spot_stop_loss_pct": 0.03,
                "spot_pt_atr_mult": None,
                "spot_sl_atr_mult": None,
            },
            "exit=atr(14,0.8,1.6)": {
                "spot_exit_mode": "atr",
                "spot_profit_target_pct": None,
                "spot_stop_loss_pct": None,
                "spot_atr_period": 14,
                "spot_pt_atr_mult": 0.8,
                "spot_sl_atr_mult": 1.6,
            },
        }
        tick_catalog_default: dict[str, dict[str, object]] = {
            "tick=off": {"tick_gate_mode": "off"},
            "tick=raschke": {
                "tick_gate_mode": "raschke",
                "tick_gate_symbol": "TICK-AMEX",
                "tick_gate_exchange": "AMEX",
                "tick_neutral_policy": "allow",
                "tick_direction_policy": "both",
                "tick_band_ma_period": 10,
                "tick_width_z_lookback": 252,
                "tick_width_z_enter": 1.0,
                "tick_width_z_exit": 0.5,
                "tick_width_slope_lookback": 3,
            },
        }
        shock_catalog_default: dict[str, dict[str, object]] = {
            "shock=off": {"shock_gate_mode": "off"},
            "shock=surf_daily": {
                "shock_gate_mode": "surf",
                "shock_detector": "daily_atr_pct",
                "shock_daily_atr_period": 14,
                "shock_daily_on_atr_pct": 13.5,
                "shock_daily_off_atr_pct": 13.0,
                "shock_daily_on_tr_pct": 9.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
            "shock=surf_atr_ratio": {
                "shock_gate_mode": "surf",
                "shock_detector": "atr_ratio",
                "shock_atr_fast_period": 7,
                "shock_atr_slow_period": 50,
                "shock_on_ratio": 1.55,
                "shock_off_ratio": 1.30,
                "shock_min_atr_pct": 7.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
            "shock=surf_tr_ratio": {
                "shock_gate_mode": "surf",
                "shock_detector": "tr_ratio",
                "shock_atr_fast_period": 7,
                "shock_atr_slow_period": 50,
                "shock_on_ratio": 1.55,
                "shock_off_ratio": 1.30,
                "shock_min_atr_pct": 7.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
        }
        slope_catalog_default: dict[str, dict[str, object]] = {
            "slope=off": {},
            "slope>=0.01": {"ema_slope_min_pct": 0.01},
        }
        risk_catalog_default: dict[str, dict[str, object]] = {
            "risk=off": {},
            "risk=riskoff9": _risk_pack_riskoff(tr_med=9.0, lookback_days=5, mode="hygiene", cutoff_hour_et=15),
            "risk=riskpanic9": _risk_pack_riskpanic(
                tr_med=9.0,
                neg_gap_ratio=0.6,
                lookback_days=5,
                short_factor=0.5,
                cutoff_hour_et=15,
            ),
            "risk=riskpop9": _risk_pack_riskpop(
                tr_med=9.0,
                pos_gap_ratio=0.6,
                lookback_days=5,
                long_factor=1.2,
                short_factor=0.5,
                cutoff_hour_et=15,
            ),
        }

        def _pairs_from_registry(
            *,
            dim_name: str,
            variants_key: str,
            fallback_catalog: dict[str, dict[str, object]],
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
            if out:
                return out
            fallback_rows = [
                (str(label), dict(payload)) for label, payload in tuple(fallback_catalog.items()) if isinstance(label, str) and isinstance(payload, dict)
            ]
            if not fallback_rows:
                raise SystemExit(f"combo_full requires at least one {dim_name} variant.")
            return fallback_rows

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

        perm_catalog_default = {
            str(note): dict(over) for note, over in tuple(_PERM_JOINT_PROFILE.get("perm_variants") or ()) if isinstance(note, str) and isinstance(over, dict)
        }
        tod_catalog_default = {
            str(note): dict(over)
            for _start_h, _end_h, note, over in tuple(_PERM_JOINT_PROFILE.get("tod_windows") or ())
            if isinstance(note, str) and isinstance(over, dict)
        }
        vol_catalog_default = {
            str(note): dict(over) for note, over in tuple(_PERM_JOINT_PROFILE.get("vol_variants") or ()) if isinstance(note, str) and isinstance(over, dict)
        }
        cadence_catalog_default = {
            str(note): dict(over) for note, over in tuple(_PERM_JOINT_PROFILE.get("cadence_variants") or ()) if isinstance(note, str) and isinstance(over, dict)
        }
        pair_catalog_by_dim: dict[str, dict[str, dict[str, object]]] = {
            "direction": direction_catalog_default,
            "perm": perm_catalog_default,
            "tod": tod_catalog_default,
            "vol": vol_catalog_default,
            "cadence": cadence_catalog_default,
            "regime": regime_catalog_default,
            "regime2": regime2_catalog_default,
            "exit": exit_catalog_default,
            "tick": tick_catalog_default,
            "shock": shock_catalog_default,
            "slope": slope_catalog_default,
            "risk": risk_catalog_default,
        }
        pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]] = {
            str(dim_name): _pairs_from_registry(
                dim_name=str(dim_name),
                variants_key=str(variants_key),
                fallback_catalog=pair_catalog_by_dim[str(dim_name)],
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
