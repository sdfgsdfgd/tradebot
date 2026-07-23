"""Canonical spot-sweep search dimensions."""

from __future__ import annotations

_EMA_SIGNAL_PRESET_LANES = (
    ("2/4", ("core", "combo", "tight")),
    ("3/7", ("core", "combo")),
    ("4/9", ("core", "combo", "tight")),
    ("5/10", ("core", "combo")),
    ("5/13", ("combo",)),
    ("8/21", ("core", "combo")),
    ("9/21", ("core", "combo")),
    ("21/50", ("combo",)),
)


def _ema_signal_presets(lane: str) -> tuple[str, ...]:
    key = str(lane).strip().lower()
    return tuple(preset for preset, lanes in _EMA_SIGNAL_PRESET_LANES if key in lanes)


def _ema_signal_variants(
    lane: str, *, entry_mode: str = "cross"
) -> tuple[tuple[str, dict[str, object]], ...]:
    mode = str(entry_mode).strip().lower()
    return tuple(
        (
            f"ema={preset} {mode}",
            {
                "entry_signal": "ema",
                "ema_preset": preset,
                "ema_entry_mode": mode,
            },
        )
        for preset in _ema_signal_presets(lane)
    )


_PERMISSION_TOD_HOURS = ((9, 16), (10, 15), (11, 16), (17, 3), (17, 4), (17, 5), (18, 3), (18, 4), (18, 5), (19, 3), (19, 4), (19, 5))
_PERMISSION_TOD_WINDOWS = (
    (None, None, "tod=base", {}), (None, None, "tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
    *((start, end, f"tod={start:02d}-{end:02d} ET", {"entry_start_hour_et": start, "entry_end_hour_et": end}) for start, end in _PERMISSION_TOD_HOURS),
)

_AXIS_DIMENSION_REGISTRY: dict[str, dict[str, object]] = {
    "perm_joint": {
        "tod_windows": _PERMISSION_TOD_WINDOWS,
        "perm_variants": (
            ("perm=base", {}),
            (
                "perm=off",
                {
                    "ema_spread_min_pct": None,
                    "ema_slope_min_pct": None,
                    "ema_spread_min_pct_down": None,
                    "ema_slope_signed_min_pct_up": None,
                    "ema_slope_signed_min_pct_down": None,
                },
            ),
            (
                "perm=loose 0.0015/0.01/0.02",
                {
                    "ema_spread_min_pct": 0.0015,
                    "ema_slope_min_pct": 0.01,
                    "ema_spread_min_pct_down": 0.02,
                },
            ),
            (
                "perm=loose_slope 0.0015/0.03/0.02",
                {
                    "ema_spread_min_pct": 0.0015,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.02,
                },
            ),
            (
                "perm=mid 0.003/0.03/0.04",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                },
            ),
            (
                "perm=spready 0.006/0.03/0.08",
                {
                    "ema_spread_min_pct": 0.006,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.08,
                },
            ),
            (
                "perm=tight_slope 0.003/0.06/0.08",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.06,
                    "ema_spread_min_pct_down": 0.08,
                },
            ),
            (
                "perm=tight 0.006/0.06/0.08",
                {
                    "ema_spread_min_pct": 0.006,
                    "ema_slope_min_pct": 0.06,
                    "ema_spread_min_pct_down": 0.08,
                },
            ),
            (
                "perm=signed_down",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                    "ema_slope_signed_min_pct_down": 0.005,
                },
            ),
            (
                "perm=signed_both",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                    "ema_slope_signed_min_pct_up": 0.005,
                    "ema_slope_signed_min_pct_down": 0.005,
                },
            ),
            ("perm=fine spread>=0.002", {"ema_spread_min_pct": 0.002}),
            ("perm=fine spread>=0.004", {"ema_spread_min_pct": 0.004}),
            ("perm=fine spread>=0.007", {"ema_spread_min_pct": 0.007}),
        ),
        "vol_variants": (
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.0@20", {"volume_ratio_min": 1.0, "volume_ema_period": 20}),
            ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
            ("vol>=1.5@20", {"volume_ratio_min": 1.5, "volume_ema_period": 20}),
        ),
        "cadence_variants": (
            ("cad=base", {}),
            ("cad=skip1 cd2", {"skip_first_bars": 1, "cooldown_bars": 2}),
            ("cad=skip2 cd2", {"skip_first_bars": 2, "cooldown_bars": 2}),
        ),
        "cost_hints": {"tod_windows": 1.2, "perm_variants": 1.0, "vol_variants": 0.6, "cadence_variants": 0.4},
    },
    "gate_matrix": {
        "perm_variants": (
            ("perm=off", {}),
            (
                "perm=core",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.01,
                    "ema_spread_min_pct_down": 0.03,
                    "ema_slope_signed_min_pct_down": 0.005,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.2,
                    "volume_ema_period": 20,
                },
            ),
            (
                "perm=loose",
                {
                    "ema_spread_min_pct": 0.0015,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.02,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.0,
                    "volume_ema_period": 20,
                },
            ),
            (
                "perm=tight",
                {
                    "ema_spread_min_pct": 0.006,
                    "ema_slope_min_pct": 0.06,
                    "ema_spread_min_pct_down": 0.08,
                    "ema_slope_signed_min_pct_up": 0.005,
                    "ema_slope_signed_min_pct_down": 0.005,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.2,
                    "volume_ema_period": 20,
                },
            ),
        ),
        "tod_variants": (
            ("tod=off", None, None),
            ("tod=09-11 ET", 9, 11),
            ("tod=09-16 ET", 9, 16),
            ("tod=10-15 ET", 10, 15),
            ("tod=18-04 ET", 18, 4),
        ),
        "short_mults": (1.0, 0.2, 0.05, 0.02, 0.01, 0.0),
        "cost_hints": {"perm_variants": 1.0, "tod_variants": 0.8, "short_mults": 0.4},
    },
    "shock": {
        "modes": ("detect", "block", "block_longs", "block_shorts", "surf"),
        "dir_variants": (("regime", 2, "dir=regime@2"), ("signal", 1, "dir=signal@1")),
        "sl_mults": (1.0, 0.75),
        "pt_mults": (1.0, 0.75),
        "short_risk_factors": (1.0, 0.5),
        "ratio_rows": (
            ("atr_ratio", 5, 30, 1.35, 1.20, 6.0),
            ("atr_ratio", 7, 50, 1.55, 1.30, 7.0),
            ("atr_ratio", 10, 80, 1.45, 1.25, 7.0),
            ("atr_ratio", 14, 120, 1.35, 1.20, 9.0),
            ("atr_ratio", 7, 30, 1.70, 1.40, 7.0),
            ("tr_ratio", 5, 30, 1.35, 1.20, 6.0),
            ("tr_ratio", 7, 50, 1.55, 1.30, 7.0),
            ("tr_ratio", 10, 80, 1.45, 1.25, 7.0),
            ("tr_ratio", 14, 120, 1.35, 1.20, 9.0),
            ("tr_ratio", 7, 30, 1.70, 1.40, 7.0),
        ),
        "daily_atr_rows": (
            (14, 13.0, 11.0, None),
            (14, 13.5, 13.0, None),
            (14, 14.0, 13.0, None),
            (14, 14.0, 13.0, 9.0),
            (10, 13.0, 11.0, 9.0),
            (21, 14.0, 13.0, 10.0),
        ),
        "drawdown_rows": (
            (10, -15.0, -8.0),
            (20, -20.0, -10.0),
            (20, -25.0, -15.0),
            (30, -25.0, -15.0),
            (60, -30.0, -20.0),
        ),
        "advanced_modes": ("detect", "surf", "block_longs"),
        "advanced_detectors": (
            (
                {
                    "shock_detector": "tr_ratio",
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_on_ratio": 1.30,
                    "shock_off_ratio": 1.20,
                    "shock_min_atr_pct": 5.0,
                },
                "det=tr_ratio 3/21 on=1.30 off=1.20 min=5",
            ),
            (
                {
                    "shock_detector": "tr_ratio",
                    "shock_atr_fast_period": 5,
                    "shock_atr_slow_period": 50,
                    "shock_on_ratio": 1.35,
                    "shock_off_ratio": 1.25,
                    "shock_min_atr_pct": 7.0,
                },
                "det=tr_ratio 5/50 on=1.35 off=1.25 min=7",
            ),
            (
                {
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 13.5,
                    "shock_daily_off_atr_pct": 13.0,
                    "shock_daily_on_tr_pct": 9.0,
                },
                "det=daily_atr p14 on=13.5 off=13 tr_on=9",
            ),
        ),
        "advanced_short_risk_factors": (1.0, 2.0, 5.0, 12.0),
        "advanced_long_down_factors": (1.0, 0.7, 0.4, 0.0),
        "advanced_scales": (
            ({}, "scale=off"),
            (
                {
                    "shock_risk_scale_target_atr_pct": 12.0,
                    "shock_risk_scale_min_mult": 0.2,
                },
                "scale=atr12 min=0.2",
            ),
            (
                {
                    "shock_scale_detector": "tr_ratio",
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_risk_scale_target_atr_pct": 0.45,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_risk_scale_apply_to": "both",
                },
                "scale=tr_ratio 3/21 target=0.45 min=0.2 both",
            ),
            (
                {
                    "shock_scale_detector": "daily_drawdown",
                    "shock_drawdown_lookback_days": 20,
                    "shock_risk_scale_target_atr_pct": 8.0,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_risk_scale_apply_to": "both",
                },
                "scale=daily_drawdown lb=20 target=8 min=0.2 both",
            ),
        ),
        "cost_hints": {"modes": 1.0, "detectors": 1.2, "advanced": 1.4},
    },
    "risk_overlays": {
        "riskoff_trs": (6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        "riskoff_lbs": (3, 5, 7, 10),
        "riskoff_modes": ("hygiene", "directional"),
        "riskoff_cutoffs_et": (None, 15, 16),
        "panic_trs": (2.75, 3.0, 3.25, 8.0, 9.0, 10.0, 12.0),
        "panic_neg_ratios": (0.5, 0.6, 0.8),
        "panic_lbs": (5, 10),
        "panic_short_factors": (1.0, 0.5, 0.2, 0.0),
        "panic_long_factors": (None, 1.0, 0.4, 0.0),
        "panic_cutoffs_et": (None, 15, 16),
        "panic_neg_gap_abs_pcts": (None, 0.005, 0.01, 0.02),
        "panic_tr_delta_variants": (
            (None, 1, "trΔ=off"),
            (0.25, 1, "trΔ>=0.25@1d"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (0.75, 1, "trΔ>=0.75@1d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ),
        "pop_trs": (7.0, 8.0, 9.0, 10.0, 12.0),
        "pop_pos_ratios": (0.5, 0.6, 0.8),
        "pop_lbs": (5, 10),
        "pop_long_factors": (0.6, 0.8, 1.0, 1.2, 1.5),
        "pop_short_factors": (1.0, 0.5, 0.2, 0.0),
        "pop_cutoffs_et": (None, 15),
        "pop_modes": ("hygiene", "directional"),
        "pop_pos_gap_abs_pcts": (None, 0.01, 0.02),
        "pop_tr_delta_variants": (
            (None, 1, "trΔ=off"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ),
        "cost_hints": {"riskoff": 1.0, "riskpanic": 1.3, "riskpop": 1.2},
    },
    "combo_full_cartesian_tight": {
        # Unified tight Cartesian for combo_full: all dimensions are crossed in one core.
        "direction_variants": _ema_signal_variants("tight"),
        "confirm_bars": (0, 1),
        "perm_variants": (
            (
                "perm=off",
                {
                    "ema_spread_min_pct": None,
                    "ema_slope_min_pct": None,
                    "ema_spread_min_pct_down": None,
                    "ema_slope_signed_min_pct_up": None,
                    "ema_slope_signed_min_pct_down": None,
                },
            ),
            (
                "perm=mid 0.003/0.03/0.04",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                },
            ),
            (
                "perm=tight 0.006/0.06/0.08",
                {
                    "ema_spread_min_pct": 0.006,
                    "ema_slope_min_pct": 0.06,
                    "ema_spread_min_pct_down": 0.08,
                },
            ),
        ),
        "tod_variants": (
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
        ),
        "vol_variants": (
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
        ),
        "cadence_variants": (
            ("cad=base", {}),
            ("cad=skip1 cd2", {"skip_first_bars": 1, "cooldown_bars": 2}),
        ),
        "regime_variants": (
            (
                "regime=ST(4h:7,0.5,hl2)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
            ),
            (
                "regime=ST(1d:14,1.0,hl2)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 1.0,
                    "supertrend_source": "hl2",
                },
            ),
            (
                "regime=ST(4h:10,0.8,close)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 10,
                    "supertrend_multiplier": 0.8,
                    "supertrend_source": "close",
                },
            ),
        ),
        "regime2_variants": (
            ("r2=off", {"regime2_mode": "off", "regime2_bar_size": None}),
            (
                "r2=ST(4h:3,0.25,close)",
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
            ),
        ),
        "exit_variants": (
            (
                "exit=pct(0.015,0.03)",
                {
                    "spot_exit_mode": "pct",
                    "spot_profit_target_pct": 0.015,
                    "spot_stop_loss_pct": 0.03,
                    "spot_pt_atr_mult": None,
                    "spot_sl_atr_mult": None,
                },
            ),
            (
                "exit=stop_only(0.03)",
                {
                    "spot_exit_mode": "pct",
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": 0.03,
                    "spot_pt_atr_mult": None,
                    "spot_sl_atr_mult": None,
                },
            ),
            (
                "exit=atr(14,0.8,1.6)",
                {
                    "spot_exit_mode": "atr",
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.8,
                    "spot_sl_atr_mult": 1.6,
                },
            ),
        ),
        "tick_variants": (
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=raschke",
                {
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
            ),
        ),
        "shock_variants": (
            ("shock=off", {"shock_gate_mode": "off"}),
            (
                "shock=surf_daily",
                {
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
            ),
            (
                "shock=surf_atr_ratio",
                {
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
            ),
            (
                "shock=surf_tr_ratio",
                {
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
            ),
        ),
        "slope_variants": (
            ("slope=off", {}),
            ("slope>=0.01", {"ema_slope_min_pct": 0.01}),
        ),
        "risk_variants": (
            ("risk=off", {}),
            (
                "risk=riskoff9",
                {
                    "riskoff_tr5_med_pct": 9.0,
                    "riskoff_lookback_days": 5,
                    "riskoff_mode": "hygiene",
                    "risk_entry_cutoff_hour_et": 15,
                    "riskpanic_tr5_med_pct": None,
                    "riskpanic_neg_gap_ratio_min": None,
                },
            ),
            (
                "risk=riskpanic9",
                {
                    "riskoff_tr5_med_pct": None,
                    "riskpanic_tr5_med_pct": 9.0,
                    "riskpanic_neg_gap_ratio_min": 0.6,
                    "riskpanic_lookback_days": 5,
                    "riskpanic_short_risk_mult_factor": 0.5,
                    "risk_entry_cutoff_hour_et": 15,
                },
            ),
            (
                "risk=riskpop9",
                {
                    "riskoff_tr5_med_pct": None,
                    "riskpanic_tr5_med_pct": None,
                    "riskpanic_neg_gap_ratio_min": None,
                    "riskpop_tr5_med_pct": 9.0,
                    "riskpop_pos_gap_ratio_min": 0.6,
                    "riskpop_lookback_days": 5,
                    "riskpop_long_risk_mult_factor": 1.2,
                    "riskpop_short_risk_mult_factor": 0.5,
                    "risk_entry_cutoff_hour_et": 15,
                    "riskoff_mode": "hygiene",
                },
            ),
        ),
        # Timing/rats profiles: base by default; HF presets can expand this dimension.
        "timing_profile_variants": (
            (
                "timing=base",
                {
                    "strategy_overrides": {},
                    "filter_overrides": {},
                },
            ),
        ),
        "hf_profile_variants": (
            (
                "timing=hf_symm_v10",
                {
                    "strategy_overrides": {
                        "ema_preset": "3/7",
                        "ema_entry_mode": "trend",
                        "entry_confirm_bars": 0,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": 0.026,
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 7,
                        "supertrend_multiplier": 0.4,
                        "supertrend_source": "close",
                        "tick_gate_mode": "off",
                        "spot_exec_bar_size": "1 min",
                    },
                    "filter_overrides": {
                        "ema_spread_min_pct": 0.00075,
                        "ema_spread_min_pct_down": 0.014,
                        "ema_slope_min_pct": 0.004,
                        "shock_gate_mode": "detect",
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_short_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor_down": 1.0,
                        "shock_stop_loss_pct_mult": 1.0,
                        "shock_profit_target_pct_mult": 1.0,
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 4.0,
                        "shock_daily_off_atr_pct": 4.0,
                        "shock_daily_on_tr_pct": 4.0,
                        "shock_on_drawdown_pct": -20.0,
                        "shock_off_drawdown_pct": -10.0,
                        "shock_scale_detector": "tr_ratio",
                        "shock_drawdown_lookback_days": 10,
                        "shock_risk_scale_target_atr_pct": 12.0,
                        "shock_risk_scale_min_mult": 0.2,
                        "shock_risk_scale_apply_to": "both",
                        "shock_atr_fast_period": 5,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.30,
                        "shock_off_ratio": 1.25,
                        "shock_min_atr_pct": 4.0,
                        "riskpanic_tr5_med_pct": 2.25,
                        "riskpanic_neg_gap_ratio_min": 0.65,
                        "riskpanic_neg_gap_abs_pct_min": 0.005,
                        "riskpanic_lookback_days": 6,
                        "riskpanic_tr5_med_delta_min_pct": 0.5,
                        "riskpanic_tr5_med_delta_lookback_days": 1,
                        "riskpanic_long_risk_mult_factor": 0.0,
                        "riskpanic_short_risk_mult_factor": 1.0,
                        "riskpanic_long_scale_mode": "linear",
                        "riskpanic_long_scale_tr_delta_max_pct": None,
                        "entry_permission_mode": "off",
                        "entry_time_gate_mode": "off",
                        "ratsv_enabled": True,
                        "ratsv_slope_window_bars": 5,
                        "ratsv_tr_fast_bars": 5,
                        "ratsv_tr_slow_bars": 20,
                        "ratsv_branch_a_rank_min": 0.0240,
                        "ratsv_branch_a_tr_ratio_min": 0.96,
                        "ratsv_branch_a_slope_med_min_pct": 0.000024,
                        "ratsv_branch_a_slope_vel_min_pct": 0.000012,
                        "ratsv_branch_a_cross_age_max_bars": 4,
                        "ratsv_branch_b_rank_min": 0.22,
                        "ratsv_branch_b_tr_ratio_min": 1.12,
                        "ratsv_branch_b_slope_med_min_pct": 0.00085,
                        "ratsv_branch_b_slope_vel_min_pct": 0.00006,
                        "ratsv_branch_b_cross_age_max_bars": 4,
                        "ratsv_probe_cancel_max_bars": 5,
                        "ratsv_probe_cancel_slope_adverse_min_pct": 0.00029,
                        "ratsv_probe_cancel_tr_ratio_min": 0.95,
                        "ratsv_adverse_release_min_hold_bars": 1,
                        "ratsv_adverse_release_slope_adverse_min_pct": 0.0007,
                        "ratsv_adverse_release_tr_ratio_min": 1.0325,
                    },
                },
            ),
            (
                "timing=hf_symm_v9",
                {
                    "strategy_overrides": {
                        "ema_preset": "3/7",
                        "ema_entry_mode": "trend",
                        "entry_confirm_bars": 0,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": 0.026,
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 7,
                        "supertrend_multiplier": 0.4,
                        "supertrend_source": "close",
                        "tick_gate_mode": "off",
                        "spot_exec_bar_size": "1 min",
                    },
                    "filter_overrides": {
                        "ema_spread_min_pct": 0.00075,
                        "ema_spread_min_pct_down": 0.014,
                        "ema_slope_min_pct": 0.004,
                        "shock_gate_mode": "detect",
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_short_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor_down": 1.0,
                        "shock_stop_loss_pct_mult": 1.0,
                        "shock_profit_target_pct_mult": 1.0,
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 4.0,
                        "shock_daily_off_atr_pct": 4.0,
                        "shock_daily_on_tr_pct": 4.0,
                        "shock_on_drawdown_pct": -20.0,
                        "shock_off_drawdown_pct": -10.0,
                        "shock_scale_detector": "tr_ratio",
                        "shock_drawdown_lookback_days": 10,
                        "shock_risk_scale_target_atr_pct": 12.0,
                        "shock_risk_scale_min_mult": 0.2,
                        "shock_risk_scale_apply_to": "both",
                        "shock_atr_fast_period": 5,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.30,
                        "shock_off_ratio": 1.25,
                        "shock_min_atr_pct": 4.0,
                        "riskpanic_tr5_med_pct": 2.25,
                        "riskpanic_neg_gap_ratio_min": 0.65,
                        "riskpanic_neg_gap_abs_pct_min": 0.005,
                        "riskpanic_lookback_days": 6,
                        "riskpanic_tr5_med_delta_min_pct": 0.5,
                        "riskpanic_tr5_med_delta_lookback_days": 1,
                        "riskpanic_long_risk_mult_factor": 0.0,
                        "riskpanic_short_risk_mult_factor": 1.0,
                        "riskpanic_long_scale_mode": "linear",
                        "riskpanic_long_scale_tr_delta_max_pct": None,
                        "entry_permission_mode": "off",
                        "entry_time_gate_mode": "off",
                        "ratsv_enabled": True,
                        "ratsv_slope_window_bars": 5,
                        "ratsv_tr_fast_bars": 5,
                        "ratsv_tr_slow_bars": 20,
                        "ratsv_branch_a_rank_min": 0.0245,
                        "ratsv_branch_a_tr_ratio_min": 0.96,
                        "ratsv_branch_a_slope_med_min_pct": 0.000024,
                        "ratsv_branch_a_slope_vel_min_pct": 0.000012,
                        "ratsv_branch_a_cross_age_max_bars": 5,
                        "ratsv_branch_b_rank_min": 0.22,
                        "ratsv_branch_b_tr_ratio_min": 1.12,
                        "ratsv_branch_b_slope_med_min_pct": 0.00085,
                        "ratsv_branch_b_slope_vel_min_pct": 0.00006,
                        "ratsv_branch_b_cross_age_max_bars": 4,
                        "ratsv_probe_cancel_max_bars": 5,
                        "ratsv_probe_cancel_slope_adverse_min_pct": 0.00029,
                        "ratsv_probe_cancel_tr_ratio_min": 0.95,
                        "ratsv_adverse_release_min_hold_bars": 1,
                        "ratsv_adverse_release_slope_adverse_min_pct": 0.0007,
                        "ratsv_adverse_release_tr_ratio_min": 1.0325,
                    },
                },
            ),
        ),
        "short_mults": (1.0, 0.2),
        # Dominant ordering used by mixed-radix shard ranges.
        "dominant_dims": (
            "timing_profile",
            "tick",
            "regime",
            "regime2",
            "risk",
            "shock",
            "exit",
            "slope",
        ),
        "cost_hints": {
            "timing_profile": 1.8,
            "tick": 1.4,
            "regime": 1.3,
            "regime2": 1.2,
            "risk": 1.5,
            "shock": 1.3,
            "exit": 1.2,
            "direction": 1.0,
            "perm": 1.1,
            "tod": 1.0,
            "vol": 0.8,
            "cadence": 0.7,
            "confirm": 0.8,
            "slope": 0.9,
            "short_mult": 0.7,
        },
    },
}

_SWEEP_COST_MODEL: dict[str, float] = {
        "base": 1.0,
        "regime_cross_tf": 0.5,
        "regime2_cross_tf": 0.5,
        "tick_gate_on": 0.75,
        "exec_cross_tf": 0.75,
        "perm_gate_on": 0.15,
        "tod_gate_on": 0.12,
        "volume_gate_on": 0.08,
        "cadence_gate_on": 0.05,
        "shock_gate_on": 0.4,
        "riskoff_overlay_on": 0.2,
        "riskpanic_overlay_on": 0.35,
        "riskpop_overlay_on": 0.3,
}

_SWEEP_RUNTIME_POLICY: dict[str, dict[str, object]] = {
        "stage_frontier": {
            # Conservative dominance rule: require repeated misses before pruning.
            "min_eval_count": 3,
            "max_keep_count": 0,
            "max_best_pnl": 0.0,
            "max_best_pnl_over_dd": 0.0,
        },
        "dimension_value_utility": {
            # Minimum sample count before treating a dimension-value row as predictive.
            "min_eval_count": 6,
            # Weights for utility scoring used by cartesian planning/sharding.
            "weight_keep_rate": 0.70,
            "weight_hit_rate": 0.20,
            "weight_confidence": 0.10,
            # Confidence normalizer in eval-count units.
            "confidence_eval_scale": 24.0,
            # Floor for eval seconds in utility denominator.
            "eval_sec_floor": 0.01,
            # Skip low-cardinality stage writes; utility hints are noisy there.
            "write_min_total": 128,
            # Sampling factor for very large stages (1 = write every eval).
            "write_sample_mod": 1,
        },
        "dimension_upper_bound": {
            # Minimum observations before an upper-bound row can influence ordering.
            "min_eval_count": 6,
            # Low-ceiling frontier (<= both) gets deferred toward tail.
            "low_ceiling_max_keep_count": 0,
            "low_ceiling_max_best_pnl": 0.0,
            "low_ceiling_max_best_pnl_over_dd": 0.0,
            # Confidence normalization for scoring.
            "confidence_eval_scale": 24.0,
            # Skip low-cardinality stage writes; upper-bound stats are low-signal there.
            "write_min_total": 96,
            # Sampling factor for very large stages (1 = write every eval).
            "write_sample_mod": 1,
        },
        "run_cfg_persistent": {
            # Enable RAM-first batching in worker-stage processes.
            "ram_first_worker": 1,
            # Flush when pending batch reaches this size.
            "batch_write_size": 256,
            # Flush at least this often (seconds), even under low throughput.
            "batch_write_interval_sec": 2.0,
        },
        "series_pack_prewarm": {
            # Build/reuse summary series packs once per unique key in worker loops.
            "enabled": 1,
            # Only activate prewarm logic for meaningful stage sizes.
            "min_total": 128,
            # Guardrail for unique per-stage pack contexts.
            "max_unique": 4096,
        },
        "planner_heartbeat": {
            # Parent monitor cadence for worker-heartbeat aggregation output.
            "monitor_interval_sec": 30.0,
            # Mark worker stale if heartbeat age exceeds this threshold.
            "stale_after_sec": 180.0,
            # Grace period for workers that have not written their first heartbeat row.
            "bootstrap_grace_sec": 180.0,
            # How many stale recycle attempts to allow before failing stage.
            "max_stale_retries": 1,
        },
        "claim_first_planner": {
            # Force serial runs of large stages into worker-claim mode.
            "enabled": 1,
            "serial_force_worker": 1,
            "min_total": 512,
            "stage_labels": ("combo_full_cartesian", "shock"),
        },
        "jobs_tuner": {
            # Auto-downshift worker fanout for tiny stages to reduce spawn overhead.
            "enabled": 1,
            "min_items_per_worker": 8,
            # Full-combo cells are multi-second causal simulations, so parallelize
            # even a tiny cold slice; cache-complete slices launch no workers.
            "min_items_per_worker_by_stage": {"combo_full_cartesian": 1},
            # 0 means "no explicit max", only clamp by detected CPUs and stage size.
            "max_workers": 0,
            # Soft caps discovered from local benchmarks:
            # <=256 items scales best around 3 workers; <=4096 around 8 workers.
            # Larger stages are unconstrained (except by CPU count / explicit max_workers).
            "soft_max_workers_by_total": ((256, 3), (4096, 8)),
        },
        "claim_span_tuner": {
            # Adaptive claim span for dynamic-claim workers (balance overhead vs stragglers).
            "enabled": 1,
            "target_claims_per_worker": 24,
            "min_claim_span": 1,
            "max_claim_span": 2048,
            "max_batch_multiple": 8,
        },
        "stage_result_snapshot": {
            # Retain complete bounded spaces so reporting-only changes need no workers.
            "complete_max_total": 2048,
        },
        "cartesian_rank_manifest": {
            # Trigger compaction only when row volume is meaningful.
            "compact_min_rows": 1024,
            # Avoid repeated compaction churn on the same stage/window key.
            "compact_min_interval_sec": 120.0,
        },
        "stage_unresolved_summary": {
            # Recompute unresolved spans when summary row is older than this TTL.
            "ttl_sec": 21600.0,
        },
        "rank_dominance_stamp": {
            # Trigger stamp compaction only when signature/range rows are meaningful.
            "compact_min_rows": 512,
            # Avoid repeated stamp compaction churn on the same stage/window key.
            "compact_min_interval_sec": 120.0,
            # Discard stale stamp rows older than this TTL window.
            "ttl_sec": 1209600.0,
        },
}
