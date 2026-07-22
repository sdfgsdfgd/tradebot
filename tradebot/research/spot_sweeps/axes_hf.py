"""SweepHighFrequencyAxes capability slice for the canonical spot research runtime."""

from __future__ import annotations

from dataclasses import replace
from ...backtest.config import (
    ConfigBundle,
)
from ...backtest.config_filters import _parse_filters
from .milestones import (
    _filters_payload,
    _print_leaderboards,
    _print_top,
    _rank_cfg_rows,
    _score_row_pnl,
    _score_row_pnl_dd,
)
from .support import (
    _mk_filters,
)


class SweepHighFrequencyAxes:
    def _sweep_hf_scalp(self) -> None:
        """High-frequency spot axis (stacked stop+flip + cadence knobs + stability overlays).

        Designed to discover "many trades/day" shapes under realism2 without requiring a seeded champion.

        Stage 1: stacked stop-loss + flip-profit (fast-runner-friendly baseline).
        Stage 2: sweep cadence knobs around the best stage-1 candidates (TOD, cooldown, skip-open, confirm).
        Stage 3: apply a small set of TQQQ v34-inspired stability overlays (shock/permission/regime interactions).
        Stage 4: expand slower knobs (spot_close_eod) on a tiny shortlist.
        """
        bars_sig = self._bars_cached(self.signal_bar_size)

        def _shortlist(
            items: list[tuple[ConfigBundle, dict, str]],
            *,
            top_pnl_dd: int,
            top_pnl: int,
            top_trades: int = 0,
        ) -> list[tuple[ConfigBundle, dict, str]]:
            return _rank_cfg_rows(
                items,
                scorers=[
                    (_score_row_pnl_dd, int(top_pnl_dd)),
                    (_score_row_pnl, int(top_pnl)),
                    (
                        lambda row: (
                            int(row.get("trades") or 0),
                            float(row.get("pnl_over_dd") or float("-inf")),
                            float(row.get("pnl") or float("-inf")),
                        ),
                        int(top_trades),
                    ),
                ],
            )

        def _print_top_trades(rows: list[dict], *, title: str, top_n: int) -> None:
            _print_top(
                rows,
                title=f"{title} — Top by trades",
                top_n=int(top_n),
                sort_key=lambda row: (
                    int(row.get("trades") or 0),
                    float(row.get("pnl_over_dd") or float("-inf")),
                    float(row.get("pnl") or float("-inf")),
                ),
            )

        # Stage 1: stop-loss + flip-profit baseline (keep it fast-runner-friendly).
        #
        # Note: v1 used very tight stops (sub-0.6%), which produced many 1y winners but was negative over 10y for
        # every candidate. v2 widens the stop grid + EMA presets to reduce whipsaw and improve decade stability.
        ema_presets = ["3/7", "4/9", "5/13", "8/21", "9/21", "21/50"]
        stop_only_vals = [0.0060, 0.0080, 0.0100, 0.0120, 0.0150, 0.0200]
        flip_hold_vals = [0, 2, 4]
        # Keep stages 1-3 on the fast summary runner path (single-position, close_eod=False),
        # then expand close_eod on a tiny shortlist at the end.
        stage_fast_close_eod = False
        expand_close_eod_vals = [False, True]

        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                ema_entry_mode="trend",
                exit_on_signal_flip=False,
                flip_exit_only_if_profit=True,
                flip_exit_min_hold_bars=0,
            ),
        )

        stage1: list[tuple[ConfigBundle, dict, str]] = []
        rows: list[dict] = []

        # Stage 1 session baseline: wide RTH window, no cooldown/skip, no confirm.
        # Keep a small permission grid; permission gating is a proven stabilizer in this codebase.
        perm_variants_stage1: list[tuple[dict[str, object], str]] = [
            ({}, "perm=off"),
            (
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                },
                "perm=v34",
            ),
        ]

        regime_variants_stage1: list[tuple[dict[str, object], str]] = [
            (
                {
                    "regime_mode": "ema",
                    "regime_ema_preset": None,
                    "regime_bar_size": str(self.signal_bar_size),
                },
                "regime=off",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
                "regime=ST(7,0.5,hl2)@4h",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                },
                "regime=ST(14,0.6,hl2)@1d",
            ),
        ]
        for ema_preset in ema_presets:
            for regime_patch, regime_note in regime_variants_stage1:
                for perm_patch, perm_note in perm_variants_stage1:
                    f = _mk_filters(
                        entry_start_hour_et=9,
                        entry_end_hour_et=16,
                        cooldown_bars=0,
                        skip_first_bars=0,
                        overrides=perm_patch,
                    )
                    for sl in stop_only_vals:
                        for hold in flip_hold_vals:
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    ema_preset=str(ema_preset),
                                    entry_confirm_bars=0,
                                    spot_exit_mode="pct",
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=float(sl),
                                    exit_on_signal_flip=True,
                                    flip_exit_only_if_profit=True,
                                    flip_exit_min_hold_bars=int(hold),
                                    flip_exit_gate_mode="off",
                                    spot_close_eod=bool(stage_fast_close_eod),
                                    spot_short_risk_mult=0.01,
                                    filters=f,
                                    **regime_patch,
                                ),
                            )
                            row = self._run_cfg(cfg=cfg, bars=bars_sig)
                            if not row:
                                continue
                            note = (
                                f"stop+flip | EMA={ema_preset} confirm=0 | {regime_note} | {perm_note} | "
                                f"tod=9-16 ET skip=0 cd=0 close_eod={int(stage_fast_close_eod)} | "
                                f"SL={sl:.4f} hold={hold}"
                            )
                            row["note"] = note
                            self._record_milestone(cfg, row, note)
                            stage1.append((cfg, row, note))
                            rows.append(row)

        _print_leaderboards(rows, title="HF scalper: stage1 (stop+flip)", top_n=int(self.args.top))
        _print_top_trades(rows, title="HF scalper: stage1 (stop+flip)", top_n=int(self.args.top))

        if not stage1:
            print("HF scalper: stage1 produced 0 results; nothing to refine.", flush=True)
            return

        # Stage 2: sweep cadence knobs around the best stage1 candidates.
        target_trades = max(0, int(self.args.milestone_min_trades or 0))
        stage1_hi = [t for t in stage1 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        shortlist_pool = stage1_hi if stage1_hi else stage1
        shortlisted = _shortlist(shortlist_pool, top_pnl_dd=10, top_pnl=10, top_trades=10)
        print("")
        print(
            f"HF scalper: stage2 seeds={len(shortlisted)} (pool={len(shortlist_pool)} target_trades={target_trades})",
            flush=True,
        )

        confirm_vals = [0, 1]
        tod_variants = [
            (9, 11, "tod=9-11 ET"),
            (9, 16, "tod=9-16 ET"),
            (10, 15, "tod=10-15 ET"),
            (11, 16, "tod=11-16 ET"),
        ]
        cooldown_vals = [0, 2]
        skip_open_vals = [0, 1, 2]
        close_eod_vals = [False]

        stage2: list[tuple[ConfigBundle, dict, str]] = []
        rows2: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted:
            for confirm in confirm_vals:
                for entry_s, entry_e, tod_note in tod_variants:
                    for cooldown in cooldown_vals:
                        for skip_open in skip_open_vals:
                            for close_eod in close_eod_vals:
                                base_payload = _filters_payload(seed_cfg.strategy.filters) or {}
                                raw = dict(base_payload)
                                raw["entry_start_hour_et"] = int(entry_s)
                                raw["entry_end_hour_et"] = int(entry_e)
                                raw["cooldown_bars"] = int(cooldown)
                                raw["skip_first_bars"] = int(skip_open)
                                f = _parse_filters(raw)
                                if _filters_payload(f) is None:
                                    f = None
                                cfg = replace(
                                    seed_cfg,
                                    strategy=replace(
                                        seed_cfg.strategy,
                                        entry_confirm_bars=int(confirm),
                                        spot_close_eod=bool(close_eod),
                                        filters=f,
                                    ),
                                )
                                row = self._run_cfg(cfg=cfg, bars=bars_sig)
                                if not row:
                                    continue
                                note = f"{seed_note} | {tod_note} skip={skip_open} cd={cooldown} close_eod={int(close_eod)} confirm={confirm}"
                                row["note"] = note
                                self._record_milestone(cfg, row, note)
                                stage2.append((cfg, row, note))
                                rows2.append(row)

        _print_leaderboards(rows2, title="HF scalper: stage2 (cadence knobs)", top_n=int(self.args.top))
        _print_top_trades(rows2, title="HF scalper: stage2 (cadence knobs)", top_n=int(self.args.top))

        if not stage2:
            print("HF scalper: stage2 produced 0 results; skipping overlays.", flush=True)
            return

        # Stage 3: apply a small overlay grid (v34-inspired) to the best stage2 candidates.
        stage2_hi = [t for t in stage2 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        overlay_pool = stage2_hi if stage2_hi else stage2
        shortlisted2 = _shortlist(overlay_pool, top_pnl_dd=8, top_pnl=8, top_trades=8)
        print("")
        print(
            f"HF scalper: stage3 seeds={len(shortlisted2)} (pool={len(overlay_pool)})",
            flush=True,
        )

        # Overlays:
        # - Regime: off vs 4h supertrend (v34-like)
        regime_variants: list[tuple[dict[str, object], str]] = [
            (
                {
                    "regime_mode": "ema",
                    "regime_ema_preset": None,
                    "regime_bar_size": str(self.signal_bar_size),
                },
                "regime=off",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
                "regime=ST(7,0.5,hl2)@4h",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                },
                "regime=ST(14,0.6,hl2)@1d",
            ),
        ]

        # - Permission: off vs v34-like thresholds (kept small; SLV needs its own calibration later).
        perm_variants: list[tuple[dict[str, object] | None, str]] = [
            (None, "perm=seed"),
            (
                {
                    "ema_spread_min_pct": None,
                    "ema_slope_min_pct": None,
                    "ema_spread_min_pct_down": None,
                },
                "perm=off",
            ),
            (
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                },
                "perm=v34",
            ),
        ]

        # - Shock: off vs detect(tr_ratio) with SLV-scaled min_atr_pct and a couple ratio thresholds.
        shock_variants: list[tuple[dict[str, object] | None, str]] = [
            (None, "shock=seed"),
            ({"shock_gate_mode": "off"}, "shock=off"),
            (
                {
                    "shock_gate_mode": "block",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 4.5,
                    "shock_daily_off_atr_pct": 4.0,
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                },
                "shock=block daily_atr% 4.5/4.0",
            ),
            (
                {
                    "shock_gate_mode": "detect",
                    "shock_detector": "tr_ratio",
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_on_ratio": 1.30,
                    "shock_off_ratio": 1.20,
                    "shock_min_atr_pct": 1.5,
                    "shock_risk_scale_target_atr_pct": 3.5,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_stop_loss_pct_mult": 1.0,
                    "shock_profit_target_pct_mult": 1.0,
                },
                "shock=detect tr_ratio(3/21) 1.30/1.20 min_atr%=1.5",
            ),
        ]

        # - Short sizing asymmetry: mimic v34's "shorts can be toxic" behavior.
        short_mult_vals = [1.0, 0.2, 0.01, 0.0]

        flip_variants: list[tuple[dict[str, object], str]] = [
            ({"exit_on_signal_flip": False}, "flip=off"),
            (
                {
                    "exit_on_signal_flip": True,
                    "flip_exit_only_if_profit": True,
                    "flip_exit_min_hold_bars": 2,
                    "flip_exit_gate_mode": "off",
                },
                "flip=profit hold=2",
            ),
        ]

        stage3: list[tuple[ConfigBundle, dict, str]] = []
        rows3: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted2:
            seed_filters = seed_cfg.strategy.filters
            entry_s = getattr(seed_filters, "entry_start_hour_et", None) if seed_filters is not None else None
            entry_e = getattr(seed_filters, "entry_end_hour_et", None) if seed_filters is not None else None
            cooldown = int(getattr(seed_filters, "cooldown_bars", 0) or 0) if seed_filters is not None else 0
            skip_open = int(getattr(seed_filters, "skip_first_bars", 0) or 0) if seed_filters is not None else 0

            for regime_patch, regime_note in regime_variants:
                for perm_patch, perm_note in perm_variants:
                    for shock_patch, shock_note in shock_variants:
                        base_payload = _filters_payload(seed_cfg.strategy.filters) or {}
                        raw = dict(base_payload)
                        if entry_s is not None and entry_e is not None:
                            raw["entry_start_hour_et"] = int(entry_s)
                            raw["entry_end_hour_et"] = int(entry_e)
                        raw["cooldown_bars"] = int(cooldown)
                        raw["skip_first_bars"] = int(skip_open)
                        if perm_patch is not None:
                            raw.update(perm_patch)
                        if shock_patch is not None:
                            raw.update(shock_patch)
                        f2 = _parse_filters(raw)
                        if _filters_payload(f2) is None:
                            f2 = None

                        for short_mult in short_mult_vals:
                            for flip_patch, flip_note in flip_variants:
                                cfg = seed_cfg
                                cfg = replace(
                                    cfg,
                                    strategy=replace(
                                        cfg.strategy,
                                        filters=f2,
                                        spot_short_risk_mult=float(short_mult),
                                        **regime_patch,
                                        **flip_patch,
                                    ),
                                )
                                row = self._run_cfg(cfg=cfg, bars=bars_sig)
                                if not row:
                                    continue
                                note = f"{seed_note} | {regime_note} | {perm_note} | {shock_note} | short_mult={short_mult:g} | {flip_note}"
                                row["note"] = note
                                self._record_milestone(cfg, row, note)
                                stage3.append((cfg, row, note))
                                rows3.append(row)

        _print_leaderboards(
            rows3,
            title="HF scalper: stage3 (v34-inspired overlays)",
            top_n=int(self.args.top),
        )
        _print_top_trades(
            rows3,
            title="HF scalper: stage3 (v34-inspired overlays)",
            top_n=int(self.args.top),
        )

        # Stage 4: expand close_eod on a tiny shortlist.
        if not stage3:
            return
        stage3_hi = [t for t in stage3 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        expand_pool = stage3_hi if stage3_hi else stage3
        shortlisted3 = _shortlist(expand_pool, top_pnl_dd=6, top_pnl=6, top_trades=6)
        print("")
        print(
            f"HF scalper: expand seeds={len(shortlisted3)} (pool={len(expand_pool)})",
            flush=True,
        )

        rows4: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted3:
            for close_eod in expand_close_eod_vals:
                cfg = replace(
                    seed_cfg,
                    strategy=replace(
                        seed_cfg.strategy,
                        spot_close_eod=bool(close_eod),
                    ),
                )
                row = self._run_cfg(cfg=cfg, bars=bars_sig)
                if not row:
                    continue
                note = f"{seed_note} | expand close_eod={int(close_eod)}"
                row["note"] = note
                self._record_milestone(cfg, row, note)
                rows4.append(row)

        _print_leaderboards(rows4, title="HF scalper: expansion (close_eod)", top_n=int(self.args.top))
        _print_top_trades(rows4, title="HF scalper: expansion (close_eod)", top_n=int(self.args.top))

    def _sweep_hold(self) -> None:
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for hold in (0, 1, 2, 3, 4, 6, 8):
            cfg = replace(base, strategy=replace(base.strategy, flip_exit_min_hold_bars=int(hold)))
            note = f"hold={hold}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="hold", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Flip-exit min hold sweep", top_n=int(self.args.top))

    def _sweep_spot_short_risk_mult(self) -> None:
        """Sweep the short sizing multiplier (only affects spot_sizing_mode=risk_pct)."""
        base = self._base_bundle(bar_size=self.signal_bar_size, filters=None)
        vals = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0]
        cfg_pairs = [(base, "base")]
        for mult in vals:
            cfg = replace(base, strategy=replace(base.strategy, spot_short_risk_mult=float(mult)))
            note = f"spot_short_risk_mult={mult:g}"
            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(axis_tag="spot_short_risk_mult", cfg_pairs=cfg_pairs, rows=rows) < 0:
            return
        _print_leaderboards(rows, title="Spot short risk multiplier sweep", top_n=int(self.args.top))

    def _sweep_orb(self) -> None:
        base = self._base_bundle(bar_size="15 mins", filters=None)
        cfg_pairs: list[tuple[ConfigBundle, str]] = [(base, "base")]
        rr_vals = [0.618, 0.707, 0.786, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override (not merge) filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    entry_signal="orb",
                                    ema_preset=None,
                                    entry_confirm_bars=0,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
                                    orb_risk_reward=float(rr),
                                    orb_target_mode=str(target_mode),
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=None,
                                ),
                            )
                            vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                            note = f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} tod={start_h:02d}-{end_h:02d} ET {vol_note}"
                            cfg_pairs.append((cfg, note))
        rows: list[dict] = []
        if self._run_cfg_pairs_grid(
            axis_tag="orb", cfg_pairs=cfg_pairs, rows=rows, report_every=50, heartbeat_sec=20.0
        ) < 0:
            return
        _print_leaderboards(rows, title="D) ORB sweep (open-time + window)", top_n=int(self.args.top))

    def _sweep_orb_joint(self) -> None:
        """Joint ORB exploration: ORB params × (regime bias) × (optional tick bias).

        Note: ORB uses its own stop/target derived from the opening range, so EMA-based
        quality gates (spread/slope) aren't applicable here unless we compute EMA in
        parallel. We stick to regime/tick/volume/TOD gates that remain well-defined.
        """
        bars_15m = self._bars_cached("15 mins")

        # Start from the selected base shape, but neutralize regime/tick so stage1 can
        # shortlist ORB mechanics without hidden gating.
        base = self._base_bundle(bar_size="15 mins", filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                entry_signal="orb",
                ema_preset=None,
                entry_confirm_bars=0,
                regime_mode="ema",
                regime_bar_size="15 mins",
                regime_ema_preset=None,
                regime2_mode="off",
                regime2_bar_size=None,
                tick_gate_mode="off",
            ),
        )
        base_row = self._run_cfg(
            cfg=base,
            bars=bars_15m,
        )
        if base_row:
            base_row["note"] = "base (orb, no regime/tick)"
            self._record_milestone(base, base_row, str(base_row["note"]))

        rr_vals = [0.618, 0.707, 0.786, 0.8, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]

        # Stage 1: find the best ORB mechanics without regime/tick overlays.
        best_by_orb: dict[tuple, dict] = {}
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
                                    orb_risk_reward=float(rr),
                                    orb_target_mode=str(target_mode),
                                ),
                            )
                            row = self._run_cfg(
                                cfg=cfg,
                                bars=bars_15m,
                            )
                            if not row:
                                continue
                            orb_key = (
                                str(open_time),
                                int(window_mins),
                                str(target_mode),
                                float(rr),
                                vol_min,
                            )
                            best_by_orb[orb_key] = {"row": row}

        shortlisted = self._ranked_keys_by_row_scores(best_by_orb, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible ORB candidates (try lowering --min-trades).")
            return
        print("")
        print(f"ORB×(regime/tick): stage1 shortlisted orb={len(shortlisted)} (from {len(best_by_orb)})")

        # Stage 2: apply a small curated set of regime overlays + tick "wide-only" bias.
        regime_variants: list[tuple[str, dict[str, object]]] = [
            (
                "regime=off",
                {
                    "regime_mode": "ema",
                    "regime_bar_size": "15 mins",
                    "regime_ema_preset": None,
                },
            ),
        ]
        for atr_p, mult, src in (
            (3, 0.4, "hl2"),
            (6, 0.6, "hl2"),
            (7, 0.6, "hl2"),
            (14, 0.6, "hl2"),
            (21, 0.5, "close"),
            (21, 0.6, "hl2"),
        ):
            regime_variants.append(
                (
                    f"ST({atr_p},{mult:g},{src})@4h",
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "4 hours",
                        "supertrend_atr_period": int(atr_p),
                        "supertrend_multiplier": float(mult),
                        "supertrend_source": str(src),
                    },
                )
            )

        tick_variants: list[tuple[str, dict[str, object]]] = [
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=wide_only allow (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "allow",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
            (
                "tick=wide_only block (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
        ]

        rows: list[dict] = []
        for open_time, window_mins, target_mode, rr, vol_min in shortlisted:
            start_h, end_h = 9, 16
            if str(open_time) == "18:00":
                start_h, end_h = 18, 4
            f = _mk_filters(
                entry_start_hour_et=int(start_h),
                entry_end_hour_et=int(end_h),
                volume_ratio_min=vol_min,
                volume_ema_period=20 if vol_min is not None else None,
            )

            for regime_note, reg_over in regime_variants:
                for tick_note, tick_over in tick_variants:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            filters=f,
                            orb_open_time_et=str(open_time),
                            orb_window_mins=int(window_mins),
                            orb_risk_reward=float(rr),
                            orb_target_mode=str(target_mode),
                            regime_mode=str(reg_over.get("regime_mode") or "ema"),
                            regime_bar_size=str(reg_over.get("regime_bar_size") or "15 mins"),
                            regime_ema_preset=reg_over.get("regime_ema_preset"),
                            supertrend_atr_period=int(reg_over.get("supertrend_atr_period") or 10),
                            supertrend_multiplier=float(reg_over.get("supertrend_multiplier") or 3.0),
                            supertrend_source=str(reg_over.get("supertrend_source") or "hl2"),
                            tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                            tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                            tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                            tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                            tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                            tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                            tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                            tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                            tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                            tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
                        ),
                    )
                    row = self._run_cfg(
                        cfg=cfg,
                        bars=bars_15m,
                    )
                    if not row:
                        continue
                    vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                    note = (
                        f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} tod={start_h:02d}-{end_h:02d} ET {vol_note} | {regime_note} | {tick_note}"
                    )
                    row["note"] = note
                    self._record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(
            rows,
            title="ORB joint sweep (ORB × regime × tick)",
            top_n=int(self.args.top),
        )
