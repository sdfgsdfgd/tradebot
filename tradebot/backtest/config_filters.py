"""Canonical parser for the backtest filter configuration payload."""

from __future__ import annotations

from ..knobs.models import FiltersConfig


def _parse_filters(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object")
    def _f(val):
        return None if val is None else float(val)
    def _i(val):
        return None if val is None else int(val)
    def _pos_float_or_none(val):
        out = _f(val)
        if out is None or out <= 0:
            return None
        return float(out)
    def _ratio01_or_none(val):
        out = _f(val)
        if out is None:
            return None
        return float(max(0.0, min(1.0, out)))
    def _nonneg_int_default(val, default):
        out = _i(val)
        if out is None:
            return int(default)
        return max(0, int(out))
    volume_period = _i(raw.get("volume_ema_period"))
    if volume_period is not None and volume_period <= 0:
        volume_period = None
    start_et = _i(raw.get("entry_start_hour_et"))
    end_et = _i(raw.get("entry_end_hour_et"))
    if start_et is not None and not (0 <= int(start_et) <= 23):
        start_et = None
    if end_et is not None and not (0 <= int(end_et) <= 23):
        end_et = None

    slope_signed_up = _f(raw.get("ema_slope_signed_min_pct_up"))
    if slope_signed_up is not None and slope_signed_up <= 0:
        slope_signed_up = None
    slope_signed_down = _f(raw.get("ema_slope_signed_min_pct_down"))
    if slope_signed_down is not None and slope_signed_down <= 0:
        slope_signed_down = None

    risk_cutoff_et = _i(raw.get("risk_entry_cutoff_hour_et"))
    if risk_cutoff_et is not None and not (0 <= int(risk_cutoff_et) <= 23):
        risk_cutoff_et = None

    shock_gate_mode = raw.get("shock_gate_mode")
    if shock_gate_mode is None:
        shock_gate_mode = raw.get("shock_mode")
    if isinstance(shock_gate_mode, bool):
        shock_gate_mode = "block" if shock_gate_mode else "off"
    shock_gate_mode = str(shock_gate_mode or "off").strip().lower()
    if shock_gate_mode in ("", "0", "false", "none", "null"):
        shock_gate_mode = "off"
    if shock_gate_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        shock_gate_mode = "off"

    shock_detector = str(raw.get("shock_detector") or "atr_ratio").strip().lower()
    if shock_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
        shock_detector = "daily_atr_pct"
    elif shock_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
        shock_detector = "daily_drawdown"
    elif shock_detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
        shock_detector = "tr_ratio"
    elif shock_detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
        shock_detector = "atr_ratio"
    else:
        shock_detector = "atr_ratio"

    shock_scale_detector = raw.get("shock_scale_detector")
    if shock_scale_detector is not None:
        shock_scale_detector = str(shock_scale_detector).strip().lower()
    if shock_scale_detector in ("", "0", "false", "none", "null", "off"):
        shock_scale_detector = None
    elif shock_scale_detector is not None:
        if shock_scale_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
            shock_scale_detector = "daily_atr_pct"
        elif shock_scale_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
            shock_scale_detector = "daily_drawdown"
        elif shock_scale_detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
            shock_scale_detector = "tr_ratio"
        elif shock_scale_detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
            shock_scale_detector = "atr_ratio"
        else:
            shock_scale_detector = None

    shock_fast = _i(raw.get("shock_atr_fast_period"))
    if shock_fast is None or shock_fast <= 0:
        shock_fast = 7
    shock_slow = _i(raw.get("shock_atr_slow_period"))
    if shock_slow is None or shock_slow <= 0:
        shock_slow = 50
    shock_on = _f(raw.get("shock_on_ratio"))
    shock_off = _f(raw.get("shock_off_ratio"))
    shock_min_atr = _f(raw.get("shock_min_atr_pct"))
    shock_short_mult = _f(raw.get("shock_short_risk_mult_factor"))
    if shock_short_mult is None or shock_short_mult < 0:
        shock_short_mult = 1.0
    shock_short_boost_min_down_streak = _i(raw.get("shock_short_boost_min_down_streak_bars"))
    if shock_short_boost_min_down_streak is None or shock_short_boost_min_down_streak <= 0:
        shock_short_boost_min_down_streak = 1
    shock_short_boost_require_regime_down = bool(raw.get("shock_short_boost_require_regime_down"))
    shock_short_boost_require_entry_down = bool(raw.get("shock_short_boost_require_entry_down"))
    shock_short_boost_min_dist_on_pp = _f(raw.get("shock_short_boost_min_dist_on_pp"))
    if shock_short_boost_min_dist_on_pp is None or shock_short_boost_min_dist_on_pp < 0:
        shock_short_boost_min_dist_on_pp = 0.0
    shock_short_boost_max_dist_on_pp = _f(raw.get("shock_short_boost_max_dist_on_pp"))
    if shock_short_boost_max_dist_on_pp is None or shock_short_boost_max_dist_on_pp < 0:
        shock_short_boost_max_dist_on_pp = 0.0
    shock_short_entry_max_dist_on_pp = _f(raw.get("shock_short_entry_max_dist_on_pp"))
    if shock_short_entry_max_dist_on_pp is None or shock_short_entry_max_dist_on_pp < 0:
        shock_short_entry_max_dist_on_pp = 0.0
    shock_prearm_dist_on_max_pp = _f(raw.get("shock_prearm_dist_on_max_pp"))
    if shock_prearm_dist_on_max_pp is None or shock_prearm_dist_on_max_pp < 0:
        shock_prearm_dist_on_max_pp = 0.0
    shock_prearm_min_drawdown_pct = _f(raw.get("shock_prearm_min_drawdown_pct"))
    if shock_prearm_min_drawdown_pct is None:
        shock_prearm_min_drawdown_pct = 0.0
    shock_prearm_min_dist_on_vel_pp = _f(raw.get("shock_prearm_min_dist_on_vel_pp"))
    if shock_prearm_min_dist_on_vel_pp is None or shock_prearm_min_dist_on_vel_pp < 0:
        shock_prearm_min_dist_on_vel_pp = 0.0
    shock_prearm_min_dist_on_accel_pp = _f(raw.get("shock_prearm_min_dist_on_accel_pp"))
    if shock_prearm_min_dist_on_accel_pp is None or shock_prearm_min_dist_on_accel_pp < 0:
        shock_prearm_min_dist_on_accel_pp = 0.0
    shock_prearm_min_streak_bars = _i(raw.get("shock_prearm_min_streak_bars"))
    if shock_prearm_min_streak_bars is None or shock_prearm_min_streak_bars < 0:
        shock_prearm_min_streak_bars = 0
    shock_prearm_short_mult = _f(raw.get("shock_prearm_short_risk_mult_factor"))
    if shock_prearm_short_mult is None or shock_prearm_short_mult < 0:
        shock_prearm_short_mult = 1.0
    shock_prearm_require_regime_down = bool(raw.get("shock_prearm_require_regime_down", True))
    shock_prearm_require_entry_down = bool(raw.get("shock_prearm_require_entry_down", True))
    shock_long_mult = _f(raw.get("shock_long_risk_mult_factor"))
    if shock_long_mult is None or shock_long_mult < 0:
        shock_long_mult = 1.0
    shock_long_mult_down = _f(raw.get("shock_long_risk_mult_factor_down"))
    if shock_long_mult_down is None or shock_long_mult_down < 0:
        shock_long_mult_down = 1.0
    shock_long_boost_require_regime_up = bool(raw.get("shock_long_boost_require_regime_up"))
    shock_long_boost_require_entry_up = bool(raw.get("shock_long_boost_require_entry_up"))
    shock_long_boost_max_dist_off_pp = _f(raw.get("shock_long_boost_max_dist_off_pp"))
    if shock_long_boost_max_dist_off_pp is None or shock_long_boost_max_dist_off_pp < 0:
        shock_long_boost_max_dist_off_pp = 0.0
    shock_sl_mult = _f(raw.get("shock_stop_loss_pct_mult"))
    if shock_sl_mult is None or shock_sl_mult <= 0:
        shock_sl_mult = 1.0
    shock_pt_mult = _f(raw.get("shock_profit_target_pct_mult"))
    if shock_pt_mult is None or shock_pt_mult <= 0:
        shock_pt_mult = 1.0

    daily_atr_period = _i(raw.get("shock_daily_atr_period"))
    if daily_atr_period is None or daily_atr_period <= 0:
        daily_atr_period = 14
    daily_on = _f(raw.get("shock_daily_on_atr_pct"))
    daily_off = _f(raw.get("shock_daily_off_atr_pct"))
    daily_tr_on = _f(raw.get("shock_daily_on_tr_pct"))
    if daily_tr_on is not None and daily_tr_on <= 0:
        daily_tr_on = None
    if daily_on is None:
        daily_on = 13.0
    if daily_off is None:
        daily_off = 11.0
    if float(daily_off) > float(daily_on):
        daily_off = float(daily_on)

    dd_lb = _i(raw.get("shock_drawdown_lookback_days"))
    if dd_lb is None or dd_lb <= 1:
        dd_lb = 20
    dd_on = _f(raw.get("shock_on_drawdown_pct"))
    if dd_on is None:
        dd_on = -20.0
    dd_off = _f(raw.get("shock_off_drawdown_pct"))
    if dd_off is None:
        dd_off = -10.0
    if float(dd_off) < float(dd_on):
        dd_off = float(dd_on)
    shock_dir_lb = _i(raw.get("shock_direction_lookback"))
    if shock_dir_lb is None or shock_dir_lb <= 0:
        shock_dir_lb = 2
    shock_dir_source = str(raw.get("shock_direction_source") or "regime").strip().lower()
    if shock_dir_source not in ("regime", "signal"):
        shock_dir_source = "regime"
    shock_regime_override_dir = bool(raw.get("shock_regime_override_dir"))

    shock_regime_st_mult = _f(raw.get("shock_regime_supertrend_multiplier"))
    if shock_regime_st_mult is not None and shock_regime_st_mult <= 0:
        shock_regime_st_mult = None
    shock_cool_st_mult = _f(raw.get("shock_cooling_regime_supertrend_multiplier"))
    if shock_cool_st_mult is not None and shock_cool_st_mult <= 0:
        shock_cool_st_mult = None
    shock_daily_cool_atr = _f(raw.get("shock_daily_cooling_atr_pct"))
    if shock_daily_cool_atr is not None and shock_daily_cool_atr <= 0:
        shock_daily_cool_atr = None

    shock_scale_target = _f(raw.get("shock_risk_scale_target_atr_pct"))
    if shock_scale_target is not None and shock_scale_target <= 0:
        shock_scale_target = None
    shock_scale_min = _f(raw.get("shock_risk_scale_min_mult"))
    if shock_scale_min is None:
        shock_scale_min = 0.2
    shock_scale_min = float(max(0.0, min(1.0, shock_scale_min)))

    shock_scale_apply_to = raw.get("shock_risk_scale_apply_to")
    if shock_scale_apply_to is not None:
        shock_scale_apply_to = str(shock_scale_apply_to).strip().lower()
    if shock_scale_apply_to in ("", "0", "false", "none", "null"):
        shock_scale_apply_to = None
    if shock_scale_apply_to in ("cap", "notional_cap", "max_notional", "cap_only"):
        shock_scale_apply_to = "cap"
    elif shock_scale_apply_to in ("both", "cap_and_risk", "risk_and_cap", "all"):
        shock_scale_apply_to = "both"
    else:
        shock_scale_apply_to = "risk"
    shock_ramp_enable = bool(raw.get("shock_ramp_enable", False))
    shock_ramp_apply_to = raw.get("shock_ramp_apply_to")
    if shock_ramp_apply_to is not None:
        shock_ramp_apply_to = str(shock_ramp_apply_to).strip().lower()
    shock_ramp_apply_to = str(shock_ramp_apply_to or "down").strip().lower()
    if shock_ramp_apply_to not in ("down", "up", "both"):
        shock_ramp_apply_to = "down"
    shock_ramp_max_risk_mult = _f(raw.get("shock_ramp_max_risk_mult"))
    if shock_ramp_max_risk_mult is None or shock_ramp_max_risk_mult < 1.0:
        shock_ramp_max_risk_mult = 1.0
    shock_ramp_max_cap_floor_frac = _f(raw.get("shock_ramp_max_cap_floor_frac"))
    if shock_ramp_max_cap_floor_frac is None:
        shock_ramp_max_cap_floor_frac = 0.0
    shock_ramp_max_cap_floor_frac = float(max(0.0, min(1.0, shock_ramp_max_cap_floor_frac)))
    shock_ramp_min_slope_streak_bars = _i(raw.get("shock_ramp_min_slope_streak_bars"))
    if shock_ramp_min_slope_streak_bars is None or shock_ramp_min_slope_streak_bars < 0:
        shock_ramp_min_slope_streak_bars = 0
    shock_ramp_min_slope_abs_pct = _f(raw.get("shock_ramp_min_slope_abs_pct"))
    if shock_ramp_min_slope_abs_pct is None or shock_ramp_min_slope_abs_pct < 0.0:
        shock_ramp_min_slope_abs_pct = 0.0
    liq_boost_enable = bool(raw.get("liq_boost_enable", False))
    liq_boost_score_min = _f(raw.get("liq_boost_score_min"))
    if liq_boost_score_min is None:
        liq_boost_score_min = 2.0
    liq_boost_score_span = _f(raw.get("liq_boost_score_span"))
    if liq_boost_score_span is None or liq_boost_score_span <= 0:
        liq_boost_score_span = 2.0
    liq_boost_max_risk_mult = _f(raw.get("liq_boost_max_risk_mult"))
    if liq_boost_max_risk_mult is None or liq_boost_max_risk_mult < 1.0:
        liq_boost_max_risk_mult = 1.0
    liq_boost_cap_floor_frac = _f(raw.get("liq_boost_cap_floor_frac"))
    if liq_boost_cap_floor_frac is None:
        liq_boost_cap_floor_frac = 0.0
    liq_boost_cap_floor_frac = float(max(0.0, min(1.0, liq_boost_cap_floor_frac)))
    liq_boost_require_alignment = bool(raw.get("liq_boost_require_alignment", True))
    liq_boost_require_shock = bool(raw.get("liq_boost_require_shock", False))
    riskoff_tr5_med = _f(raw.get("riskoff_tr5_med_pct"))
    if riskoff_tr5_med is not None and riskoff_tr5_med <= 0:
        riskoff_tr5_med = None
    riskoff_tr5_lb = _i(raw.get("riskoff_tr5_lookback_days"))
    if riskoff_tr5_lb is None or riskoff_tr5_lb <= 0:
        riskoff_tr5_lb = 5

    riskoff_mode = raw.get("riskoff_mode")
    if isinstance(riskoff_mode, str):
        riskoff_mode = riskoff_mode.strip().lower()
    else:
        riskoff_mode = "hygiene"
    if riskoff_mode not in ("hygiene", "directional"):
        riskoff_mode = "hygiene"
    riskoff_short_factor = _f(raw.get("riskoff_short_risk_mult_factor"))
    if riskoff_short_factor is None:
        riskoff_short_factor = 1.0
    if riskoff_short_factor < 0:
        riskoff_short_factor = 1.0
    riskoff_long_factor = _f(raw.get("riskoff_long_risk_mult_factor"))
    if riskoff_long_factor is None:
        riskoff_long_factor = 1.0
    if riskoff_long_factor < 0:
        riskoff_long_factor = 1.0

    riskpanic_tr5_med = _f(raw.get("riskpanic_tr5_med_pct"))
    if riskpanic_tr5_med is not None and riskpanic_tr5_med <= 0:
        riskpanic_tr5_med = None
    riskpanic_neg_gap = _f(raw.get("riskpanic_neg_gap_ratio_min"))
    if riskpanic_neg_gap is not None:
        riskpanic_neg_gap = float(max(0.0, min(1.0, riskpanic_neg_gap)))
    riskpanic_neg_gap_abs = _f(raw.get("riskpanic_neg_gap_abs_pct_min"))
    if riskpanic_neg_gap_abs is not None:
        riskpanic_neg_gap_abs = float(max(0.0, min(1.0, riskpanic_neg_gap_abs)))
        if riskpanic_neg_gap_abs <= 0:
            riskpanic_neg_gap_abs = None
    riskpanic_lb = _i(raw.get("riskpanic_lookback_days"))
    if riskpanic_lb is None or riskpanic_lb <= 0:
        riskpanic_lb = 5
    riskpanic_tr5_delta_min = _f(raw.get("riskpanic_tr5_med_delta_min_pct"))
    riskpanic_tr5_delta_lb = _i(raw.get("riskpanic_tr5_med_delta_lookback_days"))
    if riskpanic_tr5_delta_lb is None or riskpanic_tr5_delta_lb <= 0:
        riskpanic_tr5_delta_lb = 1
    riskpanic_long_factor = _f(raw.get("riskpanic_long_risk_mult_factor"))
    if riskpanic_long_factor is None:
        riskpanic_long_factor = 1.0
    if riskpanic_long_factor < 0:
        riskpanic_long_factor = 1.0
    riskpanic_long_scale_mode = raw.get("riskpanic_long_scale_mode")
    if isinstance(riskpanic_long_scale_mode, str):
        riskpanic_long_scale_mode = riskpanic_long_scale_mode.strip().lower()
    else:
        riskpanic_long_scale_mode = "off"
    if riskpanic_long_scale_mode in ("linear", "lin", "delta", "linear_delta", "linear_tr_delta"):
        riskpanic_long_scale_mode = "linear"
    elif riskpanic_long_scale_mode in ("", "0", "false", "none", "null", "off"):
        riskpanic_long_scale_mode = "off"
    else:
        riskpanic_long_scale_mode = "off"
    riskpanic_long_scale_delta_max = _f(raw.get("riskpanic_long_scale_tr_delta_max_pct"))
    if riskpanic_long_scale_delta_max is not None and riskpanic_long_scale_delta_max <= 0:
        riskpanic_long_scale_delta_max = None
    riskpanic_short_factor = _f(raw.get("riskpanic_short_risk_mult_factor"))
    if riskpanic_short_factor is None:
        riskpanic_short_factor = 1.0
    if riskpanic_short_factor < 0:
        riskpanic_short_factor = 1.0
    riskpop_tr5_med = _f(raw.get("riskpop_tr5_med_pct"))
    if riskpop_tr5_med is not None and riskpop_tr5_med <= 0:
        riskpop_tr5_med = None
    riskpop_pos_gap = _f(raw.get("riskpop_pos_gap_ratio_min"))
    if riskpop_pos_gap is not None:
        riskpop_pos_gap = float(max(0.0, min(1.0, riskpop_pos_gap)))
    riskpop_pos_gap_abs = _f(raw.get("riskpop_pos_gap_abs_pct_min"))
    if riskpop_pos_gap_abs is not None:
        riskpop_pos_gap_abs = float(max(0.0, min(1.0, riskpop_pos_gap_abs)))
        if riskpop_pos_gap_abs <= 0:
            riskpop_pos_gap_abs = None
    riskpop_lb = _i(raw.get("riskpop_lookback_days"))
    if riskpop_lb is None or riskpop_lb <= 0:
        riskpop_lb = 5
    riskpop_tr5_delta_min = _f(raw.get("riskpop_tr5_med_delta_min_pct"))
    riskpop_tr5_delta_lb = _i(raw.get("riskpop_tr5_med_delta_lookback_days"))
    if riskpop_tr5_delta_lb is None or riskpop_tr5_delta_lb <= 0:
        riskpop_tr5_delta_lb = 1
    riskpop_long_factor = _f(raw.get("riskpop_long_risk_mult_factor"))
    if riskpop_long_factor is None:
        riskpop_long_factor = 1.0
    if riskpop_long_factor < 0:
        riskpop_long_factor = 1.0
    riskpop_short_factor = _f(raw.get("riskpop_short_risk_mult_factor"))
    if riskpop_short_factor is None:
        riskpop_short_factor = 1.0
    if riskpop_short_factor < 0:
        riskpop_short_factor = 1.0

    ratsv_enabled = bool(raw.get("ratsv_enabled", False))
    ratsv_slope_window = _nonneg_int_default(raw.get("ratsv_slope_window_bars"), 5)
    if ratsv_slope_window <= 0:
        ratsv_slope_window = 5
    ratsv_slope_slow_window = _i(raw.get("ratsv_slope_slow_window_bars"))
    if ratsv_slope_slow_window is not None and ratsv_slope_slow_window <= 0:
        ratsv_slope_slow_window = None
    ratsv_tr_fast = _nonneg_int_default(raw.get("ratsv_tr_fast_bars"), 5)
    if ratsv_tr_fast <= 0:
        ratsv_tr_fast = 5
    ratsv_tr_slow = _nonneg_int_default(raw.get("ratsv_tr_slow_bars"), 20)
    if ratsv_tr_slow <= 0:
        ratsv_tr_slow = 20
    if ratsv_tr_slow < ratsv_tr_fast:
        ratsv_tr_slow = int(ratsv_tr_fast)

    ratsv_rank_min = _ratio01_or_none(raw.get("ratsv_rank_min"))
    ratsv_tr_ratio_min = _pos_float_or_none(raw.get("ratsv_tr_ratio_min"))
    ratsv_slope_med_min_pct = _pos_float_or_none(raw.get("ratsv_slope_med_min_pct"))
    ratsv_slope_vel_min_pct = _pos_float_or_none(raw.get("ratsv_slope_vel_min_pct"))
    ratsv_slope_med_slow_min_pct = _pos_float_or_none(raw.get("ratsv_slope_med_slow_min_pct"))
    ratsv_slope_vel_slow_min_pct = _pos_float_or_none(raw.get("ratsv_slope_vel_slow_min_pct"))
    ratsv_slope_vel_consistency_bars = _nonneg_int_default(raw.get("ratsv_slope_vel_consistency_bars"), 0)
    ratsv_slope_vel_consistency_min = _ratio01_or_none(raw.get("ratsv_slope_vel_consistency_min"))
    ratsv_cross_age_max = _i(raw.get("ratsv_cross_age_max_bars"))
    if ratsv_cross_age_max is not None and ratsv_cross_age_max < 0:
        ratsv_cross_age_max = None

    ratsv_branch_a_rank_min = _ratio01_or_none(raw.get("ratsv_branch_a_rank_min"))
    ratsv_branch_a_tr_ratio_min = _pos_float_or_none(raw.get("ratsv_branch_a_tr_ratio_min"))
    ratsv_branch_a_slope_med_min_pct = _pos_float_or_none(raw.get("ratsv_branch_a_slope_med_min_pct"))
    ratsv_branch_a_slope_vel_min_pct = _pos_float_or_none(raw.get("ratsv_branch_a_slope_vel_min_pct"))
    ratsv_branch_a_slope_med_slow_min_pct = _pos_float_or_none(raw.get("ratsv_branch_a_slope_med_slow_min_pct"))
    ratsv_branch_a_slope_vel_slow_min_pct = _pos_float_or_none(raw.get("ratsv_branch_a_slope_vel_slow_min_pct"))
    regime2_soft_bear_branch_a_slope_med_slow_min_pct = _pos_float_or_none(
        raw.get("regime2_soft_bear_branch_a_slope_med_slow_min_pct")
    )
    regime2_soft_bear_branch_a_slope_vel_slow_min_pct = _pos_float_or_none(
        raw.get("regime2_soft_bear_branch_a_slope_vel_slow_min_pct")
    )
    ratsv_branch_a_slope_vel_consistency_bars = _i(raw.get("ratsv_branch_a_slope_vel_consistency_bars"))
    if ratsv_branch_a_slope_vel_consistency_bars is not None and ratsv_branch_a_slope_vel_consistency_bars < 0:
        ratsv_branch_a_slope_vel_consistency_bars = None
    ratsv_branch_a_slope_vel_consistency_min = _ratio01_or_none(raw.get("ratsv_branch_a_slope_vel_consistency_min"))
    ratsv_branch_a_cross_age_max = _i(raw.get("ratsv_branch_a_cross_age_max_bars"))
    if ratsv_branch_a_cross_age_max is not None and ratsv_branch_a_cross_age_max < 0:
        ratsv_branch_a_cross_age_max = None

    ratsv_branch_b_rank_min = _ratio01_or_none(raw.get("ratsv_branch_b_rank_min"))
    ratsv_branch_b_tr_ratio_min = _pos_float_or_none(raw.get("ratsv_branch_b_tr_ratio_min"))
    ratsv_branch_b_slope_med_min_pct = _pos_float_or_none(raw.get("ratsv_branch_b_slope_med_min_pct"))
    ratsv_branch_b_slope_vel_min_pct = _pos_float_or_none(raw.get("ratsv_branch_b_slope_vel_min_pct"))
    ratsv_branch_b_slope_med_slow_min_pct = _pos_float_or_none(raw.get("ratsv_branch_b_slope_med_slow_min_pct"))
    ratsv_branch_b_slope_vel_slow_min_pct = _pos_float_or_none(raw.get("ratsv_branch_b_slope_vel_slow_min_pct"))
    ratsv_branch_b_slope_vel_consistency_bars = _i(raw.get("ratsv_branch_b_slope_vel_consistency_bars"))
    if ratsv_branch_b_slope_vel_consistency_bars is not None and ratsv_branch_b_slope_vel_consistency_bars < 0:
        ratsv_branch_b_slope_vel_consistency_bars = None
    ratsv_branch_b_slope_vel_consistency_min = _ratio01_or_none(raw.get("ratsv_branch_b_slope_vel_consistency_min"))
    ratsv_branch_b_cross_age_max = _i(raw.get("ratsv_branch_b_cross_age_max_bars"))
    if ratsv_branch_b_cross_age_max is not None and ratsv_branch_b_cross_age_max < 0:
        ratsv_branch_b_cross_age_max = None

    ratsv_probe_cancel_max_bars = _nonneg_int_default(raw.get("ratsv_probe_cancel_max_bars"), 0)
    ratsv_probe_cancel_slope_adverse_min_pct = _pos_float_or_none(raw.get("ratsv_probe_cancel_slope_adverse_min_pct"))
    ratsv_probe_cancel_tr_ratio_min = _pos_float_or_none(raw.get("ratsv_probe_cancel_tr_ratio_min"))

    ratsv_adverse_release_min_hold_bars = _nonneg_int_default(raw.get("ratsv_adverse_release_min_hold_bars"), 0)
    ratsv_adverse_release_slope_adverse_min_pct = _pos_float_or_none(raw.get("ratsv_adverse_release_slope_adverse_min_pct"))
    ratsv_adverse_release_tr_ratio_min = _pos_float_or_none(raw.get("ratsv_adverse_release_tr_ratio_min"))

    return FiltersConfig(
        rv_min=_f(raw.get("rv_min")),
        rv_max=_f(raw.get("rv_max")),
        ema_spread_min_pct=_f(raw.get("ema_spread_min_pct")),
        ema_spread_min_pct_down=_f(raw.get("ema_spread_min_pct_down")),
        ema_slope_min_pct=_f(raw.get("ema_slope_min_pct")),
        ema_slope_signed_min_pct_up=slope_signed_up,
        ema_slope_signed_min_pct_down=slope_signed_down,
        entry_start_hour=_i(raw.get("entry_start_hour")),
        entry_end_hour=_i(raw.get("entry_end_hour")),
        entry_start_hour_et=start_et,
        entry_end_hour_et=end_et,
        skip_first_bars=int(raw.get("skip_first_bars", 0) or 0),
        cooldown_bars=int(raw.get("cooldown_bars", 0) or 0),
        volume_ema_period=volume_period,
        volume_ratio_min=_f(raw.get("volume_ratio_min")),
        shock_gate_mode=shock_gate_mode,
        shock_detector=shock_detector,
        shock_scale_detector=str(shock_scale_detector) if shock_scale_detector is not None else None,
        shock_atr_fast_period=int(shock_fast),
        shock_atr_slow_period=int(shock_slow),
        shock_on_ratio=float(shock_on) if shock_on is not None else 1.55,
        shock_off_ratio=float(shock_off) if shock_off is not None else 1.30,
        shock_min_atr_pct=float(shock_min_atr) if shock_min_atr is not None else 7.0,
        shock_daily_atr_period=int(daily_atr_period),
        shock_daily_on_atr_pct=float(daily_on),
        shock_daily_off_atr_pct=float(daily_off),
        shock_daily_on_tr_pct=float(daily_tr_on) if daily_tr_on is not None else None,
        shock_drawdown_lookback_days=int(dd_lb),
        shock_on_drawdown_pct=float(dd_on),
        shock_off_drawdown_pct=float(dd_off),
        shock_short_risk_mult_factor=float(shock_short_mult),
        shock_short_boost_min_down_streak_bars=int(shock_short_boost_min_down_streak),
        shock_short_boost_require_regime_down=bool(shock_short_boost_require_regime_down),
        shock_short_boost_require_entry_down=bool(shock_short_boost_require_entry_down),
        shock_short_boost_min_dist_on_pp=float(shock_short_boost_min_dist_on_pp),
        shock_short_boost_max_dist_on_pp=float(shock_short_boost_max_dist_on_pp),
        shock_short_entry_max_dist_on_pp=float(shock_short_entry_max_dist_on_pp),
        shock_prearm_dist_on_max_pp=float(shock_prearm_dist_on_max_pp),
        shock_prearm_min_drawdown_pct=float(shock_prearm_min_drawdown_pct),
        shock_prearm_min_dist_on_vel_pp=float(shock_prearm_min_dist_on_vel_pp),
        shock_prearm_min_dist_on_accel_pp=float(shock_prearm_min_dist_on_accel_pp),
        shock_prearm_min_streak_bars=int(shock_prearm_min_streak_bars),
        shock_prearm_short_risk_mult_factor=float(shock_prearm_short_mult),
        shock_prearm_require_regime_down=bool(shock_prearm_require_regime_down),
        shock_prearm_require_entry_down=bool(shock_prearm_require_entry_down),
        shock_long_risk_mult_factor=float(shock_long_mult),
        shock_long_risk_mult_factor_down=float(shock_long_mult_down),
        shock_long_boost_require_regime_up=bool(shock_long_boost_require_regime_up),
        shock_long_boost_require_entry_up=bool(shock_long_boost_require_entry_up),
        shock_long_boost_max_dist_off_pp=float(shock_long_boost_max_dist_off_pp),
        shock_stop_loss_pct_mult=float(shock_sl_mult),
        shock_profit_target_pct_mult=float(shock_pt_mult),
        shock_direction_lookback=int(shock_dir_lb),
        shock_direction_source=shock_dir_source,
        shock_regime_override_dir=shock_regime_override_dir,
        shock_regime_supertrend_multiplier=shock_regime_st_mult,
        shock_cooling_regime_supertrend_multiplier=shock_cool_st_mult,
        shock_daily_cooling_atr_pct=shock_daily_cool_atr,
        shock_risk_scale_target_atr_pct=shock_scale_target,
        shock_risk_scale_min_mult=shock_scale_min,
        shock_risk_scale_apply_to=str(shock_scale_apply_to),
        shock_ramp_enable=bool(shock_ramp_enable),
        shock_ramp_apply_to=str(shock_ramp_apply_to),
        shock_ramp_max_risk_mult=float(shock_ramp_max_risk_mult),
        shock_ramp_max_cap_floor_frac=float(shock_ramp_max_cap_floor_frac),
        shock_ramp_min_slope_streak_bars=int(shock_ramp_min_slope_streak_bars),
        shock_ramp_min_slope_abs_pct=float(shock_ramp_min_slope_abs_pct),
        liq_boost_enable=bool(liq_boost_enable),
        liq_boost_score_min=float(liq_boost_score_min),
        liq_boost_score_span=float(liq_boost_score_span),
        liq_boost_max_risk_mult=float(liq_boost_max_risk_mult),
        liq_boost_cap_floor_frac=float(liq_boost_cap_floor_frac),
        liq_boost_require_alignment=bool(liq_boost_require_alignment),
        liq_boost_require_shock=bool(liq_boost_require_shock),
        risk_entry_cutoff_hour_et=risk_cutoff_et,
        riskoff_tr5_med_pct=riskoff_tr5_med,
        riskoff_tr5_lookback_days=int(riskoff_tr5_lb),
        riskoff_mode=str(riskoff_mode),
        riskoff_short_risk_mult_factor=float(riskoff_short_factor),
        riskoff_long_risk_mult_factor=float(riskoff_long_factor),
        riskpanic_tr5_med_pct=riskpanic_tr5_med,
        riskpanic_neg_gap_ratio_min=riskpanic_neg_gap,
        riskpanic_neg_gap_abs_pct_min=riskpanic_neg_gap_abs,
        riskpanic_lookback_days=int(riskpanic_lb),
        riskpanic_tr5_med_delta_min_pct=riskpanic_tr5_delta_min,
        riskpanic_tr5_med_delta_lookback_days=int(riskpanic_tr5_delta_lb),
        riskpanic_long_risk_mult_factor=float(riskpanic_long_factor),
        riskpanic_long_scale_mode=str(riskpanic_long_scale_mode),
        riskpanic_long_scale_tr_delta_max_pct=riskpanic_long_scale_delta_max,
        riskpanic_short_risk_mult_factor=float(riskpanic_short_factor),
        riskpop_tr5_med_pct=riskpop_tr5_med,
        riskpop_pos_gap_ratio_min=riskpop_pos_gap,
        riskpop_pos_gap_abs_pct_min=riskpop_pos_gap_abs,
        riskpop_lookback_days=int(riskpop_lb),
        riskpop_tr5_med_delta_min_pct=riskpop_tr5_delta_min,
        riskpop_tr5_med_delta_lookback_days=int(riskpop_tr5_delta_lb),
        riskpop_long_risk_mult_factor=float(riskpop_long_factor),
        riskpop_short_risk_mult_factor=float(riskpop_short_factor),
        ratsv_enabled=bool(ratsv_enabled),
        ratsv_slope_window_bars=int(ratsv_slope_window),
        ratsv_slope_slow_window_bars=int(ratsv_slope_slow_window) if ratsv_slope_slow_window is not None else None,
        ratsv_tr_fast_bars=int(ratsv_tr_fast),
        ratsv_tr_slow_bars=int(ratsv_tr_slow),
        ratsv_rank_min=ratsv_rank_min,
        ratsv_tr_ratio_min=ratsv_tr_ratio_min,
        ratsv_slope_med_min_pct=ratsv_slope_med_min_pct,
        ratsv_slope_vel_min_pct=ratsv_slope_vel_min_pct,
        ratsv_slope_med_slow_min_pct=ratsv_slope_med_slow_min_pct,
        ratsv_slope_vel_slow_min_pct=ratsv_slope_vel_slow_min_pct,
        ratsv_slope_vel_consistency_bars=int(ratsv_slope_vel_consistency_bars),
        ratsv_slope_vel_consistency_min=ratsv_slope_vel_consistency_min,
        ratsv_cross_age_max_bars=int(ratsv_cross_age_max) if ratsv_cross_age_max is not None else None,
        ratsv_branch_a_rank_min=ratsv_branch_a_rank_min,
        ratsv_branch_a_tr_ratio_min=ratsv_branch_a_tr_ratio_min,
        ratsv_branch_a_slope_med_min_pct=ratsv_branch_a_slope_med_min_pct,
        ratsv_branch_a_slope_vel_min_pct=ratsv_branch_a_slope_vel_min_pct,
        ratsv_branch_a_slope_med_slow_min_pct=ratsv_branch_a_slope_med_slow_min_pct,
        ratsv_branch_a_slope_vel_slow_min_pct=ratsv_branch_a_slope_vel_slow_min_pct,
        regime2_soft_bear_branch_a_slope_med_slow_min_pct=regime2_soft_bear_branch_a_slope_med_slow_min_pct,
        regime2_soft_bear_branch_a_slope_vel_slow_min_pct=regime2_soft_bear_branch_a_slope_vel_slow_min_pct,
        ratsv_branch_a_slope_vel_consistency_bars=(
            int(ratsv_branch_a_slope_vel_consistency_bars)
            if ratsv_branch_a_slope_vel_consistency_bars is not None
            else None
        ),
        ratsv_branch_a_slope_vel_consistency_min=ratsv_branch_a_slope_vel_consistency_min,
        ratsv_branch_a_cross_age_max_bars=int(ratsv_branch_a_cross_age_max) if ratsv_branch_a_cross_age_max is not None else None,
        ratsv_branch_b_rank_min=ratsv_branch_b_rank_min,
        ratsv_branch_b_tr_ratio_min=ratsv_branch_b_tr_ratio_min,
        ratsv_branch_b_slope_med_min_pct=ratsv_branch_b_slope_med_min_pct,
        ratsv_branch_b_slope_vel_min_pct=ratsv_branch_b_slope_vel_min_pct,
        ratsv_branch_b_slope_med_slow_min_pct=ratsv_branch_b_slope_med_slow_min_pct,
        ratsv_branch_b_slope_vel_slow_min_pct=ratsv_branch_b_slope_vel_slow_min_pct,
        ratsv_branch_b_slope_vel_consistency_bars=(
            int(ratsv_branch_b_slope_vel_consistency_bars)
            if ratsv_branch_b_slope_vel_consistency_bars is not None
            else None
        ),
        ratsv_branch_b_slope_vel_consistency_min=ratsv_branch_b_slope_vel_consistency_min,
        ratsv_branch_b_cross_age_max_bars=int(ratsv_branch_b_cross_age_max) if ratsv_branch_b_cross_age_max is not None else None,
        ratsv_probe_cancel_max_bars=int(ratsv_probe_cancel_max_bars),
        ratsv_probe_cancel_slope_adverse_min_pct=ratsv_probe_cancel_slope_adverse_min_pct,
        ratsv_probe_cancel_tr_ratio_min=ratsv_probe_cancel_tr_ratio_min,
        ratsv_adverse_release_min_hold_bars=int(ratsv_adverse_release_min_hold_bars),
        ratsv_adverse_release_slope_adverse_min_pct=ratsv_adverse_release_slope_adverse_min_pct,
        ratsv_adverse_release_tr_ratio_min=ratsv_adverse_release_tr_ratio_min,
    )
