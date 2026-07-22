"""SweepCoreAxes capability slice for the canonical spot research runtime."""

from __future__ import annotations

from dataclasses import replace
from ...backtest.config import (
    ConfigBundle,
    FiltersConfig,
    SpotLegConfig,
)
from ...backtest.config_filters import _parse_filters
from ...spot.fill_modes import SPOT_FILL_MODE_NEXT_TRADABLE_BAR
from .profiles import (
    _ATR_EXIT_PROFILE_REGISTRY,
)
from .milestones import (
    _apply_milestone_base,
    _filters_payload,
    _milestone_entry_for,
    _print_leaderboards,
)
from .support import (
    _bundle_base,
    _mk_filters,
)


class SweepCoreAxes:
    def _base_bundle(self, *, bar_size: str, filters: FiltersConfig | None) -> ConfigBundle:
        cfg = _bundle_base(
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            bar_size=bar_size,
            use_rth=self.use_rth,
            cache_dir=self.cache_dir,
            offline=self.offline,
            filters=filters,
            spot_close_eod=self.close_eod,
        )
        if self.spot_exec_bar_size:
            cfg = replace(
                cfg,
                strategy=replace(cfg.strategy, spot_exec_bar_size=self.spot_exec_bar_size),
            )
        base_name = str(self.args.base).strip().lower()
        if base_name in ("champion", "champion_pnl"):
            sort_by = "pnl" if base_name == "champion_pnl" else "pnl_dd"
            selected = _milestone_entry_for(
                self.milestones,
                symbol=self.symbol,
                signal_bar_size=str(bar_size),
                use_rth=self.use_rth,
                sort_by=sort_by,
                prefer_realism=self.realism2,
            )
            if selected is not None:
                base_strategy, base_filters, _ = selected
                cfg = _apply_milestone_base(cfg, strategy=base_strategy, filters=base_filters)
            # Allow sweeps to layer additional filters on top of the milestone baseline
            # (e.g., keep the champion's TOD window and add volume/spread/cooldown filters).
            if filters is not None:
                base_payload = _filters_payload(cfg.strategy.filters) or {}
                over_payload = _filters_payload(filters) or {}
                merged = dict(base_payload)
                merged.update(over_payload)
                merged_filters = _parse_filters(merged)
                if _filters_payload(merged_filters) is None:
                    merged_filters = None
                cfg = replace(cfg, strategy=replace(cfg.strategy, filters=merged_filters))
        elif base_name == "dual_regime":
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    regime2_mode="supertrend",
                    regime2_bar_size="4 hours",
                    regime2_supertrend_atr_period=2,
                    regime2_supertrend_multiplier=0.3,
                    regime2_supertrend_source="close",
                ),
            )

        if self.long_only:
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    directional_spot={"up": SpotLegConfig(action="BUY", qty=1)},
                ),
            )

        # Realism overrides (backtest only).
        if self.realism2:
            champion_like = base_name in ("champion", "champion_pnl")
            sizing_mode_eff = (
                str(self.sizing_mode)
                if (not champion_like) or self.sizing_mode_arg_explicit
                else str(getattr(cfg.strategy, "spot_sizing_mode", self.sizing_mode) or self.sizing_mode)
            )
            spot_notional_pct_eff = (
                float(self.spot_notional_pct)
                if (not champion_like) or self.spot_notional_pct_arg_explicit
                else float(getattr(cfg.strategy, "spot_notional_pct", self.spot_notional_pct) or 0.0)
            )
            spot_risk_pct_eff = (
                float(self.spot_risk_pct)
                if (not champion_like) or self.spot_risk_pct_arg_explicit
                else float(getattr(cfg.strategy, "spot_risk_pct", self.spot_risk_pct) or 0.0)
            )
            spot_max_notional_pct_eff = (
                float(self.spot_max_notional_pct)
                if (not champion_like) or self.spot_max_notional_pct_arg_explicit
                else float(
                    getattr(
                        cfg.strategy,
                        "spot_max_notional_pct",
                        self.spot_max_notional_pct,
                    )
                    or 1.0
                )
            )
            spot_min_qty_eff = (
                int(self.spot_min_qty)
                if (not champion_like) or self.spot_min_qty_arg_explicit
                else int(getattr(cfg.strategy, "spot_min_qty", self.spot_min_qty) or 1)
            )
            spot_max_qty_eff = (
                int(self.spot_max_qty)
                if (not champion_like) or self.spot_max_qty_arg_explicit
                else int(getattr(cfg.strategy, "spot_max_qty", self.spot_max_qty) or 0)
            )
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    spot_entry_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
                    spot_flip_exit_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
                    spot_intrabar_exits=True,
                    spot_spread=float(self.spot_spread),
                    spot_commission_per_share=float(self.spot_commission),
                    spot_commission_min=float(self.spot_commission_min),
                    spot_slippage_per_share=float(self.spot_slippage),
                    spot_mark_to_market="liquidation",
                    spot_drawdown_mode="intrabar",
                    spot_sizing_mode=str(sizing_mode_eff),
                    spot_notional_pct=float(spot_notional_pct_eff),
                    spot_risk_pct=float(spot_risk_pct_eff),
                    spot_max_notional_pct=float(spot_max_notional_pct_eff),
                    spot_min_qty=int(spot_min_qty_eff),
                    spot_max_qty=int(spot_max_qty_eff),
                ),
            )
        return cfg

    def _record_milestone(self, cfg: ConfigBundle, row: dict, note: str) -> None:
        if not bool(self.args.write_milestones):
            return
        self.milestone_rows.append((cfg, row, str(note)))

    def _sweep_volume(self) -> None:
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        ratios = [None, 1.0, 1.1, 1.2, 1.5]
        periods = [10, 20, 30]
        rows: list[dict] = []
        for ratio in ratios:
            if ratio is None:
                variants = [(None, None)]
            else:
                variants = [(ratio, p) for p in periods]
            for ratio_min, ema_p in variants:
                f = _mk_filters(volume_ratio_min=ratio_min, volume_ema_period=ema_p)
                cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
                row = self._run_cfg(cfg=cfg, bars=bars_sig)
                if not row:
                    continue
                note = "-" if ratio_min is None else f"vol>={ratio_min}@{ema_p}"
                row["note"] = note
                self._record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="A) Volume gate sweep", top_n=int(self.args.top))

    def _sweep_rv(self) -> None:
        """Orthogonal gate: annualized realized-vol (EWMA) band."""
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        rv_mins = [None, 0.25, 0.3, 0.35, 0.4, 0.45]
        rv_maxs = [None, 0.7, 0.8, 0.9, 1.0]
        rows: list[dict] = []
        for rv_min in rv_mins:
            for rv_max in rv_maxs:
                if rv_min is None and rv_max is None:
                    continue
                f = _mk_filters(rv_min=rv_min, rv_max=rv_max)
                cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
                row = self._run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"rv_min={rv_min} rv_max={rv_max}"
                row["note"] = note
                self._record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="RV gate sweep (annualized EWMA vol)", top_n=int(self.args.top))

    def _sweep_ema(self) -> None:
        bars_sig = self._bars_cached(self.signal_bar_size)
        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21"]
        rows: list[dict] = []
        for preset in presets:
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, ema_preset=str(preset)))
            row = self._run_cfg(cfg=cfg, bars=bars_sig)
            if not row:
                continue
            note = f"ema={preset}"
            row["note"] = note
            self._record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="0) Timing sweep (EMA preset)", top_n=int(self.args.top))

    def _sweep_entry_mode(self) -> None:
        """Timing semantics: cross vs trend entries (+ small confirm grid)."""
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        rows: list[dict] = []
        for mode in ("cross", "trend"):
            for confirm in (0, 1, 2):
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        ema_entry_mode=str(mode),
                        entry_confirm_bars=int(confirm),
                    ),
                )
                row = self._run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"entry_mode={mode} confirm={confirm}"
                row["note"] = note
                self._record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Entry mode sweep (cross vs trend)", top_n=int(self.args.top))

    def _sweep_tod(self) -> None:
        bars_sig = self._bars_cached(self.signal_bar_size)
        windows = [
            (None, None, "base"),
            (9, 16, "RTH 9–16 ET"),
            (10, 15, "10–15 ET"),
            (11, 16, "11–16 ET"),
        ]
        # Overnight micro-grid (wraps midnight in ET): this has been a high-leverage permission layer
        # post-lookahead-fix, and is cheap to explore.
        for start_h in (16, 17, 18, 19, 20):
            for end_h in (2, 3, 4, 5, 6):
                windows.append((start_h, end_h, f"{start_h:02d}–{end_h:02d} ET"))
        rows: list[dict] = []
        for start_h, end_h, label in windows:
            f = _mk_filters(entry_start_hour_et=start_h, entry_end_hour_et=end_h)
            cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=f)
            row = self._run_cfg(cfg=cfg, bars=bars_sig)
            if not row:
                continue
            row["note"] = label
            self._record_milestone(cfg, row, label)
            rows.append(row)
        _print_leaderboards(rows, title="B) Time-of-day gate sweep (ET)", top_n=int(self.args.top))

    def _sweep_chop_joint(self) -> None:
        """Joint chop filter stack: slope × cooldown × skip-open (keeps everything else fixed)."""
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        slope_vals = [None, 0.005, 0.01, 0.02, 0.03]
        cooldown_vals = [0, 1, 2, 3, 4, 6]
        skip_vals = [0, 1, 2, 3]

        rows: list[dict] = []
        for slope in slope_vals:
            for cooldown in cooldown_vals:
                for skip in skip_vals:
                    overrides: dict[str, object] = {
                        "ema_slope_min_pct": float(slope) if slope is not None else None,
                        "cooldown_bars": int(cooldown),
                        "skip_first_bars": int(skip),
                    }
                    f = self._merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    row = self._run_cfg(cfg=cfg)
                    if not row:
                        continue
                    slope_note = "-" if slope is None else f"slope>={float(slope):g}"
                    note = f"{slope_note} | cooldown={cooldown} | skip={skip}"
                    row["note"] = note
                    self._record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(
            rows,
            title="Chop joint sweep (slope × cooldown × skip-open)",
            top_n=int(self.args.top),
        )

    def _sweep_weekdays(self) -> None:
        """Gate exploration: which UTC weekdays contribute to the edge."""
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        day_sets: list[tuple[tuple[int, ...], str]] = [
            ((0, 1, 2, 3, 4), "Mon-Fri"),
            ((0, 1, 2, 3), "Mon-Thu"),
            ((1, 2, 3, 4), "Tue-Fri"),
            ((1, 2, 3), "Tue-Thu"),
            ((2, 3, 4), "Wed-Fri"),
            ((0, 1, 2), "Mon-Wed"),
            ((0, 1, 2, 3, 4, 5, 6), "All days"),
        ]

        rows: list[dict] = []
        for days, label in day_sets:
            cfg = replace(base, strategy=replace(base.strategy, entry_days=tuple(days)))
            row = self._run_cfg(cfg=cfg)
            if not row:
                continue
            note = f"days={label}"
            row["note"] = note
            self._record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Weekday sweep (UTC weekday gating)", top_n=int(self.args.top))

    def _sweep_exit_time(self) -> None:
        """Session-aware exit experiment: force a daily time-based flatten (ET)."""
        bars_sig = self._bars_cached(self.signal_bar_size)
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_row = self._run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            self._record_milestone(base, base_row, "base")

        times = [
            None,
            "04:00",
            "09:30",
            "10:00",
            "11:00",
            "16:00",
            "17:00",
        ]
        rows: list[dict] = []
        for t in times:
            cfg = replace(base, strategy=replace(base.strategy, spot_exit_time_et=t))
            row = self._run_cfg(cfg=cfg)
            if not row:
                continue
            note = "-" if t is None else f"exit_time={t} ET"
            row["note"] = note
            self._record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Exit-time sweep (ET flatten)", top_n=int(self.args.top))

    def _fmt_sweep_float(self, value: float, decimals: int | None) -> str:
        if decimals is None:
            return f"{float(value):g}"
        return f"{float(value):.{int(decimals)}f}"

    def _run_atr_exit_profile(self, profile_name: str) -> None:
        profile = _ATR_EXIT_PROFILE_REGISTRY.get(str(profile_name))
        if not isinstance(profile, dict):
            raise SystemExit(f"Unknown ATR sweep profile: {profile_name!r}")
        axis_tag = str(profile_name).strip().lower()
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        atr_periods = tuple(profile.get("atr_periods") or ())
        pt_mults = tuple(profile.get("pt_mults") or ())
        sl_mults = tuple(profile.get("sl_mults") or ())
        decimals_raw = profile.get("decimals")
        decimals = int(decimals_raw) if isinstance(decimals_raw, int) else None
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            spot_exit_mode="atr",
                            spot_atr_period=int(atr_p),
                            spot_pt_atr_mult=float(pt_m),
                            spot_sl_atr_mult=float(sl_m),
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=None,
                        ),
                    )
                    note = f"ATR({int(atr_p)}) PTx{self._fmt_sweep_float(float(pt_m), decimals)} SLx{self._fmt_sweep_float(float(sl_m), decimals)}"
                    cfg_pairs.append((cfg, note))
        tested_total = self._run_cfg_pairs_grid(
            axis_tag=str(axis_tag),
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=50,
            heartbeat_sec=10.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(
            rows,
            title=str(profile.get("title") or "ATR exits sweep"),
            top_n=int(self.args.top),
        )

    def _sweep_atr_exits(self) -> None:
        self._run_atr_exit_profile("atr")

    def _sweep_atr_exits_fine(self) -> None:
        """Fine-grained ATR exit sweep around the current champion neighborhood."""
        self._run_atr_exit_profile("atr_fine")

    def _sweep_atr_exits_ultra(self) -> None:
        """Ultra-fine ATR exit sweep around the current best PT neighborhood."""
        self._run_atr_exit_profile("atr_ultra")

    def _sweep_ptsl(self) -> None:
        bars_sig = self._bars_cached(self.signal_bar_size)
        base_cfg = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base_strat = base_cfg.strategy
        base_mode = str(getattr(self.args, "base", "") or "").strip().lower()
        seeded_local = bool(str(getattr(self.args, "seed_milestones", "") or "").strip()) or base_mode in (
            "champion",
            "champion_pnl",
        )
        # Seeded runs should mutate around the current champion neighborhood (tight, high-value).
        # Unseeded runs keep the broad legacy PT/SL surface.
        if seeded_local:
            base_pt_raw = getattr(base_strat, "spot_profit_target_pct", None)
            base_pt = float(base_pt_raw) if base_pt_raw is not None and float(base_pt_raw) > 0 else None
            base_sl_raw = getattr(base_strat, "spot_stop_loss_pct", None)
            base_sl = float(base_sl_raw) if base_sl_raw is not None and float(base_sl_raw) > 0 else 0.01
            base_only_profit = bool(getattr(base_strat, "flip_exit_only_if_profit", False))
            base_close_eod = bool(getattr(base_strat, "spot_close_eod", False))
            base_hold = max(0, int(getattr(base_strat, "flip_exit_min_hold_bars", 0) or 0))
            base_gate = str(getattr(base_strat, "flip_exit_gate_mode", "regime_or_permission") or "regime_or_permission")
            base_flip_mode = str(getattr(base_strat, "flip_exit_mode", "entry") or "entry")

            pt_vals: list[float | None]
            if base_pt is None:
                pt_vals = [None, 0.0015, 0.0025]
            else:
                pt_vals = sorted(
                    {
                        round(max(0.0005, min(0.05, float(base_pt) * 0.8)), 6),
                        round(max(0.0005, min(0.05, float(base_pt))), 6),
                        round(max(0.0005, min(0.05, float(base_pt) * 1.2)), 6),
                    }
                )
            sl_vals = sorted(
                {
                    round(max(0.001, min(0.08, float(base_sl) * 0.8)), 6),
                    round(max(0.001, min(0.08, float(base_sl))), 6),
                    round(max(0.001, min(0.08, float(base_sl) * 1.2)), 6),
                }
            )
            only_profit_vals = list(dict.fromkeys([base_only_profit, not base_only_profit]))
            close_eod_vals = list(dict.fromkeys([base_close_eod, not base_close_eod]))
            hold_vals = sorted({int(base_hold), int(max(0, base_hold + 2))})
            alt_gate = "off" if str(base_gate).strip().lower() != "off" else "regime_or_permission"
            gate_modes = list(dict.fromkeys([str(base_gate), str(alt_gate)]))
            flip_modes = [str(base_flip_mode)]
            run_title = "PT/SL sweep (seeded-local mutation)"
        else:
            # Unified fixed-percent exit pocket:
            # combines PT/SL and exit-pivot neighborhoods in one core sweep.
            pt_vals = [
                None,
                0.0015,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.01,
                0.015,
                0.02,
            ]
            sl_vals = [0.003, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03]
            only_profit_vals = [False, True]
            close_eod_vals = [False, True]
            hold_vals = [0, 2]
            gate_modes = ["off", "regime_or_permission"]
            flip_modes = ["entry"]
            run_title = "PT/SL sweep (fixed pct exits + flip/close_eod semantics)"
        plan = []
        for pt in pt_vals:
            for sl in sl_vals:
                for only_profit in only_profit_vals:
                    for close_eod in close_eod_vals:
                        for hold in hold_vals:
                            for gate_mode in gate_modes:
                                for flip_mode in flip_modes:
                                    cfg = replace(
                                        base_cfg,
                                        strategy=replace(
                                            base_cfg.strategy,
                                            spot_profit_target_pct=(float(pt) if pt is not None else None),
                                            spot_stop_loss_pct=float(sl),
                                            spot_exit_mode="pct",
                                            exit_on_signal_flip=True,
                                            flip_exit_mode=str(flip_mode),
                                            flip_exit_only_if_profit=bool(only_profit),
                                            flip_exit_min_hold_bars=int(hold),
                                            flip_exit_gate_mode=str(gate_mode),
                                            spot_close_eod=bool(close_eod),
                                        ),
                                    )
                                    pt_note = "None" if pt is None else f"{float(pt):.4f}"
                                    note = (
                                        f"PT={pt_note} SL={float(sl):.4f} "
                                        f"flip={str(flip_mode)} hold={int(hold)} only_profit={int(only_profit)} "
                                        f"gate={gate_mode} close_eod={int(close_eod)}"
                                    )
                                    plan.append((cfg, note, None))
        _tested, kept = self._run_sweep(
            plan=plan,
            bars=bars_sig,
            total=len(plan),
            progress_label="ptsl axis",
            report_every=50,
            heartbeat_sec=15.0,
        )
        rows = [row for _, row, _note, _meta in kept]
        _print_leaderboards(rows, title=run_title, top_n=int(self.args.top))
