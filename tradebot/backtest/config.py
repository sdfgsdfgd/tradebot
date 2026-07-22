"""Config loading for synthetic backtests."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..knobs.models import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    OptionsStrategyConfig,
    SpotLegConfig,
    SpotStrategyConfig,
    StrategyConfig,
    SyntheticConfig,
)
from .config_filters import _parse_filters
from .config_values import (
    _parse_bar_size,
    _parse_date,
    _parse_direction_source,
    _parse_directional_legs,
    _parse_directional_spot,
    _parse_ema_entry_mode,
    _parse_ema_preset,
    _parse_entry_signal,
    _parse_flip_exit_gate_mode,
    _parse_flip_exit_mode,
    _parse_instrument,
    _parse_legs,
    _parse_non_negative_float,
    _parse_non_negative_float_or_default,
    _parse_non_negative_int,
    _parse_optional_float,
    _parse_orb_target_mode,
    _parse_positive_float,
    _parse_positive_int,
    _parse_regime2_apply_to,
    _parse_regime2_bear_entry_mode,
    _parse_regime2_bear_takeover_mode,
    _parse_regime2_clean_host_takeover_state,
    _parse_regime2_crash_prearm_apply_to,
    _parse_regime_gate_mode,
    _parse_regime_mode,
    _parse_spot_drawdown_mode,
    _parse_spot_dual_branch_priority,
    _parse_spot_exit_mode,
    _parse_spot_fill_mode,
    _parse_spot_mark_to_market,
    _parse_spot_next_open_session_mode,
    _parse_spot_resize_adaptive_mode,
    _parse_spot_resize_mode,
    _parse_spot_sizing_mode,
    _parse_supertrend_source,
    _parse_tick_direction_policy,
    _parse_tick_gate_mode,
    _parse_tick_neutral_policy,
    _parse_weekdays,
)

__all__ = [
    "BacktestConfig",
    "ConfigBundle",
    "FiltersConfig",
    "LegConfig",
    "OptionsStrategyConfig",
    "SpotLegConfig",
    "SpotStrategyConfig",
    "StrategyConfig",
    "SyntheticConfig",
    "load_config",
]


_MISSING = object()


@dataclass(frozen=True)
class _FieldSpec:
    parser: Callable[[object], object]
    default: object = _MISSING


def _field(parser: Callable[[object], object], default: object = _MISSING) -> _FieldSpec:
    return _FieldSpec(parser=parser, default=default)


def _identity(value):
    return value


# region Public API
def load_config(path: str | Path) -> ConfigBundle:
    raw = json.loads(Path(path).read_text())
    backtest_raw = raw.get("backtest", {}) if isinstance(raw, dict) else {}
    strategy_raw = raw.get("strategy", {}) if isinstance(raw, dict) else {}
    synthetic_raw = raw.get("synthetic", {}) if isinstance(raw, dict) else {}
    if not isinstance(backtest_raw, dict):
        backtest_raw = {}
    if not isinstance(strategy_raw, dict):
        strategy_raw = {}
    if not isinstance(synthetic_raw, dict):
        synthetic_raw = {}

    backtest = BacktestConfig(**_parse_from_schema(backtest_raw, _backtest_schema()))
    instrument = _parse_instrument(strategy_raw.get("instrument"))

    # Keep strategy session intent aligned across live + backtest:
    # if a spot-STK strategy is full24 (signal_use_rth=false or backtest.use_rth=false)
    # but spot_next_open_session is missing, force tradable_24x5 so deferred fills
    # (next_open/next_tradable_bar) schedule correctly.
    if instrument == "spot":
        mode_raw = str(strategy_raw.get("spot_next_open_session") or "").strip()
        sec_type = str(strategy_raw.get("spot_sec_type") or "STK").strip().upper()
        use_rth_raw = strategy_raw.get("signal_use_rth")
        if isinstance(use_rth_raw, str):
            signal_use_rth = use_rth_raw.strip().lower() in ("1", "true", "yes", "on")
        elif use_rth_raw is None:
            signal_use_rth = bool(backtest.use_rth)
        else:
            signal_use_rth = bool(use_rth_raw)
        if sec_type == "STK" and not bool(signal_use_rth) and not mode_raw:
            strategy_raw = dict(strategy_raw)
            strategy_raw["spot_next_open_session"] = "tradable_24x5"

    strategy_input = _translate_legacy_strategy_payload(strategy_raw, instrument=instrument)
    if instrument == "spot":
        strategy_payload = _parse_from_schema(strategy_input, _spot_strategy_schema())
        strategy = SpotStrategyConfig(**strategy_payload)
    else:
        strategy_payload = _parse_from_schema(strategy_input, _options_strategy_schema())
        strategy = OptionsStrategyConfig(**strategy_payload)
    synthetic = SyntheticConfig(**_parse_from_schema(synthetic_raw, _synthetic_schema()))

    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def _parse_from_schema(
    raw: dict,
    schema: dict[str, Callable[[dict], object] | _FieldSpec],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for field, parser in schema.items():
        if isinstance(parser, _FieldSpec):
            if parser.default is _MISSING:
                value = raw.get(field)
            else:
                value = raw.get(field, parser.default)
            out[field] = parser.parser(value)
        else:
            out[field] = parser(raw)
    return out


def _translate_legacy_strategy_payload(raw: dict, *, instrument: str) -> dict:
    _ = instrument
    return dict(raw)


def _backtest_schema() -> dict[str, _FieldSpec]:
    return {
        "start": _field(_parse_date),
        "end": _field(_parse_date),
        "bar_size": _field(_identity, "1 hour"),
        "use_rth": _field(bool, False),
        "starting_cash": _field(float, 100_000.0),
        "risk_free_rate": _field(float, 0.02),
        "cache_dir": _field(Path, "db"),
        "calibration_dir": _field(Path, "db/calibration"),
        "output_dir": _field(Path, "backtests/out"),
        "calibrate": _field(bool, False),
        "offline": _field(bool, False),
    }


def _strategy_schema_common() -> dict[str, _FieldSpec]:
    return {
        "name": _field(_identity, "credit_spread"),
        "instrument": _field(_parse_instrument, "options"),
        "symbol": _field(_identity, "MNQ"),
        "exchange": _field(_identity, None),
        "right": _field(lambda value: str(value or "PUT").upper(), "PUT"),
        "entry_days": _field(_parse_weekdays, []),
        "max_entries_per_day": _field(lambda value: _parse_non_negative_int(value, default=1), 1),
        "dte": _field(int, 0),
        "otm_pct": _field(float, 2.5),
        "width_pct": _field(float, 1.0),
        "profit_target": _field(float, 0.5),
        "stop_loss": _field(float, 0.35),
        "exit_dte": _field(int, 0),
        "quantity": _field(int, 1),
        "stop_loss_basis": _field(_identity, "max_loss"),
        "min_credit": _field(_parse_optional_float, None),
        "ema_preset": _field(_parse_ema_preset, None),
        "ema_entry_mode": _field(_parse_ema_entry_mode, None),
        "entry_confirm_bars": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "regime_ema_preset": _field(_parse_ema_preset, None),
        "regime_bar_size": _field(_parse_bar_size, None),
        "ema_directional": _field(bool, False),
        "exit_on_signal_flip": _field(bool, False),
        "flip_exit_mode": _field(_parse_flip_exit_mode, None),
        "flip_exit_gate_mode": _field(_parse_flip_exit_gate_mode, None),
        "flip_exit_min_hold_bars": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "flip_exit_only_if_profit": _field(bool, False),
        "spot_controlled_flip": _field(bool, False),
        "direction_source": _field(_parse_direction_source, None),
        "directional_legs": _field(_parse_directional_legs, None),
        "directional_spot": _field(_parse_directional_spot, None),
        "legs": _field(_parse_legs, None),
        "filters": _field(_parse_filters, None),
        "spot_profit_target_pct": _field(_parse_optional_float, None),
        "spot_stop_loss_pct": _field(_parse_optional_float, None),
        "spot_close_eod": _field(bool, False),
        "entry_signal": _field(_parse_entry_signal, None),
        "orb_window_mins": _field(lambda value: _parse_positive_int(value, default=15), 15),
        "orb_risk_reward": _field(lambda value: _parse_positive_float(value, default=2.0), 2.0),
        "orb_target_mode": _field(_parse_orb_target_mode, None),
        "orb_open_time_et": _field(_identity, None),
        "spot_exit_mode": _field(_parse_spot_exit_mode, None),
        "spot_atr_period": _field(lambda value: _parse_positive_int(value, default=14), 14),
        "spot_pt_atr_mult": _field(lambda value: _parse_non_negative_float_or_default(value, default=1.5), 1.5),
        "spot_sl_atr_mult": _field(lambda value: _parse_non_negative_float_or_default(value, default=1.0), 1.0),
        "spot_exit_time_et": _field(_identity, None),
        "spot_exec_bar_size": _field(_parse_bar_size, None),
        "regime_mode": _field(_parse_regime_mode, None),
        "regime2_mode": _field(_parse_regime_gate_mode, None),
        "regime2_apply_to": _field(_parse_regime2_apply_to, None),
        "regime2_ema_preset": _field(_parse_ema_preset, None),
        "regime2_bar_size": _field(_parse_bar_size, None),
        "regime2_supertrend_atr_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "regime2_supertrend_multiplier": _field(lambda value: _parse_positive_float(value, default=3.0), 3.0),
        "regime2_supertrend_source": _field(_parse_supertrend_source, None),
        "regime2_bear_entry_mode": _field(_parse_regime2_bear_entry_mode, None),
        "regime2_bear_allow_long_recovery": _field(bool, True),
        "regime2_bear_supertrend_atr_period": _field(
            lambda value: None if value is None else _parse_positive_int(value, default=10),
            None,
        ),
        "regime2_bear_supertrend_multiplier": _field(
            lambda value: None if value is None else _parse_positive_float(value, default=3.0),
            None,
        ),
        "regime2_bear_supertrend_source": _field(_parse_supertrend_source, None),
        "regime2_bear_takeover_mode": _field(_parse_regime2_bear_takeover_mode, None),
        "regime2_crash_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_crash_block_longs": _field(bool, False),
        "regime2_crash_prearm_apply_to": _field(_parse_regime2_crash_prearm_apply_to, None),
        "regime2_crash_prearm_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_crash_prearm_shock_dir_ret_sum_pct_max": _field(_parse_optional_float, None),
        "regime2_crash_prearm_branch_a_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max": _field(_parse_optional_float, None),
        "regime2_repair_block_branch_b_longs": _field(bool, False),
        "regime2_repair_branch_b_long_max_shock_atr_pct": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_repair_branch_b_long_block_after_hour_et": _field(
            lambda value: None if value is None else max(0, min(23, _parse_non_negative_int(value, default=0))),
            None,
        ),
        "regime2_transition_hot_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_transition_hot_release_max_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_upcorridor_branch_a_long_fresh_release_age_max_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_upcorridor_branch_a_long_stale_release_age_min_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_upcorridor_branch_b_long_stale_release_age_min_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_release_age_min_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_release_age_max_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_ddv_min_pp": _field(_parse_optional_float, None),
        "regime2_trenddown_branch_b_long_hard_up_ddv_max_pp": _field(_parse_optional_float, None),
        "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp": _field(_parse_optional_float, None),
        "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp": _field(_parse_optional_float, None),
        "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_continuation_confidence_branch_a_transition_release_age_max_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max": _field(
            lambda value: None if value is None else _parse_non_negative_float_or_default(value, default=0.0),
            None,
        ),
        "regime2_continuation_confidence_branch_a_transition_ddv_max_pp": _field(_parse_optional_float, None),
        "regime2_clean_host_enable": _field(bool, False),
        "regime2_clean_host_takeover_state": _field(_parse_regime2_clean_host_takeover_state, None),
        "regime2_clean_host_supertrend_multiplier": _field(
            lambda value: None if value is None else _parse_positive_float(value, default=3.0),
            None,
        ),
        "regime2_clean_host_bear_supertrend_multiplier": _field(
            lambda value: None if value is None else _parse_positive_float(value, default=3.0),
            None,
        ),
        "regime2_clean_host_bear_hard_supertrend_multiplier": _field(
            lambda value: None if value is None else _parse_positive_float(value, default=3.0),
            None,
        ),
        "regime2_bear_hard_mode": _field(_parse_regime2_bear_entry_mode, None),
        "regime2_bear_hard_bar_size": _field(_parse_bar_size, None),
        "regime2_bear_hard_supertrend_atr_period": _field(
            lambda value: None if value is None else _parse_positive_int(value, default=10),
            None,
        ),
        "regime2_bear_hard_supertrend_multiplier": _field(
            lambda value: None if value is None else _parse_positive_float(value, default=3.0),
            None,
        ),
        "regime2_bear_hard_supertrend_source": _field(_parse_supertrend_source, None),
        "supertrend_atr_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "supertrend_multiplier": _field(lambda value: _parse_positive_float(value, default=3.0), 3.0),
        "supertrend_source": _field(_parse_supertrend_source, None),
        "tick_gate_mode": _field(_parse_tick_gate_mode, None),
        "tick_gate_symbol": _field(lambda value: str(value or "TICK-NYSE").strip(), "TICK-NYSE"),
        "tick_gate_exchange": _field(lambda value: str(value or "NYSE").strip(), "NYSE"),
        "tick_band_ma_period": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "tick_width_z_lookback": _field(lambda value: _parse_positive_int(value, default=252), 252),
        "tick_width_z_enter": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "tick_width_z_exit": _field(lambda value: _parse_positive_float(value, default=0.5), 0.5),
        "tick_width_slope_lookback": _field(lambda value: _parse_positive_int(value, default=3), 3),
        "tick_neutral_policy": _field(_parse_tick_neutral_policy, None),
        "tick_direction_policy": _field(_parse_tick_direction_policy, None),
        "spot_entry_fill_mode": _field(lambda value: _parse_spot_fill_mode(value, default="close"), "close"),
        "spot_flip_exit_fill_mode": _field(lambda value: _parse_spot_fill_mode(value, default="close"), "close"),
        "spot_next_open_session": _field(_parse_spot_next_open_session_mode, "auto"),
        "spot_intrabar_exits": _field(bool, False),
        "spot_spread": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_commission_per_share": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_commission_min": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_slippage_per_share": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_mark_to_market": _field(_parse_spot_mark_to_market, None),
        "spot_drawdown_mode": _field(_parse_spot_drawdown_mode, None),
        "spot_sizing_mode": _field(_parse_spot_sizing_mode, None),
        "spot_notional_pct": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_risk_pct": _field(lambda value: _parse_non_negative_float(value, default=0.0), 0.0),
        "spot_short_risk_mult": _field(lambda value: _parse_non_negative_float(value, default=1.0), 1.0),
        "spot_max_notional_pct": _field(lambda value: _parse_non_negative_float(value, default=1.0), 1.0),
        "spot_min_qty": _field(lambda value: _parse_positive_int(value, default=1), 1),
        "spot_max_qty": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "spot_dual_branch_enabled": _field(bool, False),
        "spot_dual_branch_priority": _field(_parse_spot_dual_branch_priority, None),
        "spot_branch_a_ema_preset": _field(_parse_ema_preset, None),
        "spot_branch_a_entry_confirm_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "spot_branch_a_min_signed_slope_pct": _field(_parse_optional_float, None),
        "spot_branch_a_max_signed_slope_pct": _field(_parse_optional_float, None),
        "spot_branch_a_size_mult": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "spot_branch_b_ema_preset": _field(_parse_ema_preset, None),
        "spot_branch_b_entry_confirm_bars": _field(
            lambda value: None if value is None else _parse_non_negative_int(value, default=0),
            None,
        ),
        "spot_branch_b_min_signed_slope_pct": _field(_parse_optional_float, None),
        "spot_branch_b_max_signed_slope_pct": _field(_parse_optional_float, None),
        "spot_branch_b_size_mult": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "spot_policy_pack": _field(_identity, None),
        "spot_policy_graph": _field(_identity, None),
        "spot_graph_profile": _field(_identity, None),
        "regime_router": _field(bool, False),
        "regime_router_fast_window_days": _field(lambda value: _parse_positive_int(value, default=63), 63),
        "regime_router_slow_window_days": _field(lambda value: _parse_positive_int(value, default=84), 84),
        "regime_router_min_dwell_days": _field(lambda value: _parse_positive_int(value, default=10), 10),
        "regime_router_crash_hf_slow_ret_max": _field(_parse_optional_float, -0.25),
        "regime_router_hf_takeover_crash_ret_max": _field(_parse_optional_float, -0.08),
        "regime_router_hf_takeover_crash_maxdd_min": _field(_parse_optional_float, 0.20),
        "regime_router_hf_takeover_crash_rv_max": _field(_parse_optional_float, 0.55),
        "regime_router_damage_positive_lock_maxdd_min": _field(_parse_optional_float, 0.24),
        "regime_router_damage_positive_lock_ret_max": _field(_parse_optional_float, 0.20),
        "regime_router_damage_positive_lock_eff_max": _field(_parse_optional_float, 0.10),
        "regime_router_bull_sovereign_on_confirm_days": _field(lambda value: _parse_positive_int(value, default=1), 1),
        "regime_router_bull_sovereign_off_confirm_days": _field(lambda value: _parse_positive_int(value, default=7), 7),
        "regime_router_bull_overextended_hf_fast_ret_min": _field(_parse_optional_float, 999.0),
        "regime_router_bull_overextended_hf_slow_ret_min": _field(_parse_optional_float, 0.0),
        "spot_entry_policy": _field(_identity, None),
        "spot_exit_policy": _field(_identity, None),
        "spot_resize_policy": _field(_identity, None),
        "spot_risk_overlay_policy": _field(_identity, None),
        "spot_resize_mode": _field(_parse_spot_resize_mode, None),
        "spot_resize_min_delta_qty": _field(lambda value: _parse_positive_int(value, default=1), 1),
        "spot_resize_max_step_qty": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "spot_resize_allow_scale_in": _field(bool, True),
        "spot_resize_allow_scale_out": _field(bool, True),
        "spot_resize_cooldown_bars": _field(lambda value: _parse_non_negative_int(value, default=0), 0),
        "spot_resize_adaptive_mode": _field(_parse_spot_resize_adaptive_mode, None),
        "spot_resize_adaptive_min_mult": _field(lambda value: _parse_positive_float(value, default=0.5), 0.5),
        "spot_resize_adaptive_max_mult": _field(lambda value: _parse_positive_float(value, default=1.75), 1.75),
        "spot_resize_adaptive_atr_target_pct": _field(_parse_optional_float, None),
        "spot_resize_adaptive_atr_vel_ref_pct": _field(lambda value: _parse_positive_float(value, default=0.40), 0.40),
        "spot_resize_adaptive_slope_ref_pct": _field(lambda value: _parse_positive_float(value, default=0.10), 0.10),
        "spot_resize_adaptive_vel_ref_pct": _field(lambda value: _parse_positive_float(value, default=0.08), 0.08),
        "spot_resize_adaptive_tr_ratio_ref": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "spot_entry_tr_ratio_min": _field(_parse_optional_float, None),
        "spot_entry_slope_med_abs_min_pct": _field(_parse_optional_float, None),
        "spot_entry_slope_vel_abs_min_pct": _field(_parse_optional_float, None),
        "spot_entry_slow_slope_med_abs_min_pct": _field(_parse_optional_float, None),
        "spot_entry_slow_slope_vel_abs_min_pct": _field(_parse_optional_float, None),
        "spot_entry_shock_atr_max_pct": _field(_parse_optional_float, None),
        "spot_entry_atr_vel_min_pct": _field(_parse_optional_float, None),
        "spot_entry_atr_accel_min_pct": _field(_parse_optional_float, None),
        "spot_entry_context_confidence_mode": _field(_identity, None),
        "spot_guard_threshold_scale_mode": _field(_identity, None),
        "spot_guard_threshold_scale_min_mult": _field(lambda value: _parse_positive_float(value, default=0.70), 0.70),
        "spot_guard_threshold_scale_max_mult": _field(lambda value: _parse_positive_float(value, default=1.80), 1.80),
        "spot_guard_threshold_scale_tr_ref": _field(_parse_optional_float, None),
        "spot_guard_threshold_scale_atr_vel_ref_pct": _field(_parse_optional_float, None),
        "spot_guard_threshold_scale_tr_median_ref_pct": _field(_parse_optional_float, None),
        "spot_exit_flip_hold_slope_min_pct": _field(_parse_optional_float, None),
        "spot_exit_flip_hold_tr_ratio_min": _field(_parse_optional_float, None),
        "spot_exit_flip_hold_slow_slope_min_pct": _field(_parse_optional_float, None),
        "spot_exit_flip_hold_slope_vel_min_pct": _field(_parse_optional_float, None),
        "spot_exit_flip_hold_slow_slope_vel_min_pct": _field(_parse_optional_float, None),
        "spot_flip_hold_dynamic_mode": _field(_identity, None),
        "spot_flip_hold_dynamic_min_mult": _field(lambda value: _parse_positive_float(value, default=0.50), 0.50),
        "spot_flip_hold_dynamic_max_mult": _field(lambda value: _parse_positive_float(value, default=2.50), 2.50),
        "spot_flip_hold_dynamic_tr_ref": _field(_parse_optional_float, None),
        "spot_flip_hold_dynamic_atr_vel_ref_pct": _field(_parse_optional_float, None),
        "spot_flip_hold_dynamic_tr_median_ref_pct": _field(_parse_optional_float, None),
        "spot_graph_overlay_atr_hi_pct": _field(_parse_optional_float, None),
        "spot_graph_overlay_atr_hi_min_mult": _field(lambda value: _parse_positive_float(value, default=0.5), 0.5),
        "spot_graph_overlay_atr_vel_ref_pct": _field(lambda value: _parse_positive_float(value, default=0.40), 0.40),
        "spot_graph_overlay_tr_ratio_ref": _field(lambda value: _parse_positive_float(value, default=1.0), 1.0),
        "spot_graph_overlay_slope_ref_pct": _field(lambda value: _parse_positive_float(value, default=0.08), 0.08),
        "spot_graph_overlay_trend_boost_max": _field(lambda value: _parse_positive_float(value, default=1.35), 1.35),
        "spot_graph_overlay_trend_floor_mult": _field(lambda value: _parse_positive_float(value, default=0.65), 0.65),
    }


def _options_strategy_schema() -> dict[str, _FieldSpec]:
    return _strategy_schema_common()


def _spot_strategy_schema() -> dict[str, _FieldSpec]:
    return _strategy_schema_common()


def _synthetic_schema() -> dict[str, _FieldSpec]:
    return {
        "rv_lookback": _field(int, 60),
        "rv_ewma_lambda": _field(float, 0.94),
        "iv_risk_premium": _field(float, 1.2),
        "iv_floor": _field(float, 0.05),
        "term_slope": _field(float, 0.02),
        "skew": _field(float, -0.25),
        "min_spread_pct": _field(float, 0.1),
    }


# endregion


# region Basic Parsing


# endregion


# region Filters Parsing


# endregion


# region Misc Normalizers


# endregion
