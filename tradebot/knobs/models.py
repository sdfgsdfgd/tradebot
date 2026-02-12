"""Dataclass definitions for strategy/backtest configuration.

These are intentionally dumb containers: no IO, no IBKR, no backtest logic.

Canonical import path: `tradebot.knobs.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    bar_size: str
    use_rth: bool
    starting_cash: float
    risk_free_rate: float
    cache_dir: Path
    calibration_dir: Path
    output_dir: Path
    calibrate: bool
    offline: bool


@dataclass(frozen=True)
class SyntheticConfig:
    rv_lookback: int
    rv_ewma_lambda: float
    iv_risk_premium: float
    iv_floor: float
    term_slope: float
    skew: float
    min_spread_pct: float


@dataclass(frozen=True)
class LegConfig:
    action: str
    right: str
    moneyness_pct: float
    qty: int


@dataclass(frozen=True)
class SpotLegConfig:
    action: str
    qty: int


@dataclass(frozen=True)
class FiltersConfig:
    rv_min: float | None
    rv_max: float | None
    ema_spread_min_pct: float | None
    ema_slope_min_pct: float | None
    entry_start_hour: int | None
    entry_end_hour: int | None
    skip_first_bars: int
    cooldown_bars: int
    entry_start_hour_et: int | None = None
    entry_end_hour_et: int | None = None
    volume_ema_period: int | None = None
    volume_ratio_min: float | None = None
    ema_spread_min_pct_down: float | None = None
    ema_slope_signed_min_pct_up: float | None = None
    ema_slope_signed_min_pct_down: float | None = None
    shock_gate_mode: str = "off"
    shock_detector: str = "atr_ratio"  # "atr_ratio" | "tr_ratio" | "daily_atr_pct" | "daily_drawdown"
    shock_scale_detector: str | None = None  # Optional: compute shock_risk_scale_* off a separate detector.
    shock_atr_fast_period: int = 7
    shock_atr_slow_period: int = 50
    shock_on_ratio: float = 1.55
    shock_off_ratio: float = 1.30
    shock_min_atr_pct: float = 7.0
    shock_daily_atr_period: int = 14
    shock_daily_on_atr_pct: float = 13.0
    shock_daily_off_atr_pct: float = 11.0
    shock_daily_on_tr_pct: float | None = None
    shock_drawdown_lookback_days: int = 20
    shock_on_drawdown_pct: float = -20.0
    shock_off_drawdown_pct: float = -10.0
    shock_short_risk_mult_factor: float = 1.0
    shock_long_risk_mult_factor: float = 1.0
    shock_long_risk_mult_factor_down: float = 1.0
    shock_stop_loss_pct_mult: float = 1.0
    shock_profit_target_pct_mult: float = 1.0
    shock_direction_lookback: int = 2
    shock_direction_source: str = "regime"  # "regime" | "signal"
    shock_regime_override_dir: bool = False
    shock_regime_supertrend_multiplier: float | None = None
    shock_cooling_regime_supertrend_multiplier: float | None = None
    shock_daily_cooling_atr_pct: float | None = None
    shock_risk_scale_target_atr_pct: float | None = None
    shock_risk_scale_min_mult: float = 0.2
    shock_risk_scale_apply_to: str = "risk"  # "risk" | "cap" | "both"
    risk_entry_cutoff_hour_et: int | None = None
    riskoff_tr5_med_pct: float | None = None
    riskoff_tr5_lookback_days: int = 5
    riskoff_mode: str = "hygiene"  # "hygiene" | "directional"
    riskoff_short_risk_mult_factor: float = 1.0
    riskoff_long_risk_mult_factor: float = 1.0
    riskpanic_tr5_med_pct: float | None = None
    riskpanic_neg_gap_ratio_min: float | None = None
    riskpanic_neg_gap_abs_pct_min: float | None = None
    riskpanic_lookback_days: int = 5
    riskpanic_tr5_med_delta_min_pct: float | None = None
    riskpanic_tr5_med_delta_lookback_days: int = 1
    riskpanic_long_risk_mult_factor: float = 1.0
    riskpanic_long_scale_mode: str = "off"  # "off" | "linear"
    riskpanic_long_scale_tr_delta_max_pct: float | None = None
    riskpanic_short_risk_mult_factor: float = 1.0
    riskpop_tr5_med_pct: float | None = None
    riskpop_pos_gap_ratio_min: float | None = None
    riskpop_pos_gap_abs_pct_min: float | None = None
    riskpop_lookback_days: int = 5
    riskpop_tr5_med_delta_min_pct: float | None = None
    riskpop_tr5_med_delta_lookback_days: int = 1
    riskpop_long_risk_mult_factor: float = 1.0
    riskpop_short_risk_mult_factor: float = 1.0
    # RATS-V (Reversal-Aware Two-Stage + anti-trap release) knobs; all default-off.
    ratsv_enabled: bool = False
    ratsv_slope_window_bars: int = 5
    ratsv_tr_fast_bars: int = 5
    ratsv_tr_slow_bars: int = 20
    ratsv_rank_min: float | None = None
    ratsv_tr_ratio_min: float | None = None
    ratsv_slope_med_min_pct: float | None = None
    ratsv_slope_vel_min_pct: float | None = None
    ratsv_cross_age_max_bars: int | None = None
    ratsv_branch_a_rank_min: float | None = None
    ratsv_branch_a_tr_ratio_min: float | None = None
    ratsv_branch_a_slope_med_min_pct: float | None = None
    ratsv_branch_a_slope_vel_min_pct: float | None = None
    ratsv_branch_a_cross_age_max_bars: int | None = None
    ratsv_branch_b_rank_min: float | None = None
    ratsv_branch_b_tr_ratio_min: float | None = None
    ratsv_branch_b_slope_med_min_pct: float | None = None
    ratsv_branch_b_slope_vel_min_pct: float | None = None
    ratsv_branch_b_cross_age_max_bars: int | None = None
    ratsv_probe_cancel_max_bars: int = 0
    ratsv_probe_cancel_slope_adverse_min_pct: float | None = None
    ratsv_probe_cancel_tr_ratio_min: float | None = None
    ratsv_adverse_release_min_hold_bars: int = 0
    ratsv_adverse_release_slope_adverse_min_pct: float | None = None
    ratsv_adverse_release_tr_ratio_min: float | None = None


@dataclass(frozen=True)
class StrategyConfigBase:
    name: str
    instrument: str
    symbol: str
    exchange: str | None
    right: str
    entry_days: tuple[int, ...]
    max_entries_per_day: int
    dte: int
    otm_pct: float
    width_pct: float
    profit_target: float
    stop_loss: float
    exit_dte: int
    quantity: int
    stop_loss_basis: str
    min_credit: float | None
    ema_preset: str | None
    ema_entry_mode: str
    entry_confirm_bars: int
    regime_ema_preset: str | None
    regime_bar_size: str | None
    ema_directional: bool
    exit_on_signal_flip: bool
    flip_exit_mode: str
    flip_exit_gate_mode: str
    flip_exit_min_hold_bars: int
    flip_exit_only_if_profit: bool
    direction_source: str
    directional_legs: dict[str, tuple[LegConfig, ...]] | None
    directional_spot: dict[str, SpotLegConfig] | None
    legs: tuple[LegConfig, ...] | None
    filters: FiltersConfig | None
    spot_profit_target_pct: float | None
    spot_stop_loss_pct: float | None
    spot_close_eod: bool
    entry_signal: str = "ema"
    orb_window_mins: int = 15
    orb_risk_reward: float = 2.0
    orb_target_mode: str = "rr"
    orb_open_time_et: str | None = None
    spot_exit_mode: str = "pct"
    spot_atr_period: int = 14
    spot_pt_atr_mult: float = 1.5
    spot_sl_atr_mult: float = 1.0
    spot_exit_time_et: str | None = None
    spot_exec_bar_size: str | None = None
    regime_mode: str = "ema"
    regime2_mode: str = "off"
    regime2_apply_to: str = "both"
    regime2_ema_preset: str | None = None
    regime2_bar_size: str | None = None
    regime2_supertrend_atr_period: int = 10
    regime2_supertrend_multiplier: float = 3.0
    regime2_supertrend_source: str = "hl2"
    supertrend_atr_period: int = 10
    supertrend_multiplier: float = 3.0
    supertrend_source: str = "hl2"
    tick_gate_mode: str = "off"
    tick_gate_symbol: str = "TICK-NYSE"
    tick_gate_exchange: str = "NYSE"
    tick_band_ma_period: int = 10
    tick_width_z_lookback: int = 252
    tick_width_z_enter: float = 1.0
    tick_width_z_exit: float = 0.5
    tick_width_slope_lookback: int = 3
    tick_neutral_policy: str = "allow"
    tick_direction_policy: str = "both"
    spot_entry_fill_mode: str = "close"
    spot_flip_exit_fill_mode: str = "close"
    spot_intrabar_exits: bool = False
    spot_spread: float = 0.0
    spot_commission_per_share: float = 0.0
    spot_commission_min: float = 0.0
    spot_slippage_per_share: float = 0.0
    spot_mark_to_market: str = "close"
    spot_drawdown_mode: str = "close"
    spot_sizing_mode: str = "fixed"
    spot_notional_pct: float = 0.0
    spot_risk_pct: float = 0.0
    spot_short_risk_mult: float = 1.0
    spot_max_notional_pct: float = 1.0
    spot_min_qty: int = 1
    spot_max_qty: int = 0
    spot_dual_branch_enabled: bool = False
    spot_dual_branch_priority: str = "b_then_a"  # "a_then_b" | "b_then_a"
    spot_branch_a_ema_preset: str | None = None
    spot_branch_a_entry_confirm_bars: int | None = None
    spot_branch_a_min_signed_slope_pct: float | None = None
    spot_branch_a_max_signed_slope_pct: float | None = None
    spot_branch_a_size_mult: float = 1.0
    spot_branch_b_ema_preset: str | None = None
    spot_branch_b_entry_confirm_bars: int | None = None
    spot_branch_b_min_signed_slope_pct: float | None = None
    spot_branch_b_max_signed_slope_pct: float | None = None
    spot_branch_b_size_mult: float = 1.0


@dataclass(frozen=True)
class OptionsStrategyConfig(StrategyConfigBase):
    max_open_trades: int = 1

    def __post_init__(self) -> None:
        if str(self.instrument or "").strip().lower() != "options":
            raise ValueError("OptionsStrategyConfig requires instrument='options'")


@dataclass(frozen=True)
class SpotStrategyConfig(StrategyConfigBase):
    def __post_init__(self) -> None:
        if str(self.instrument or "").strip().lower() != "spot":
            raise ValueError("SpotStrategyConfig requires instrument='spot'")


# Backward-compatible alias for older options-only call sites.
StrategyConfig = OptionsStrategyConfig
AnyStrategyConfig = Union[OptionsStrategyConfig, SpotStrategyConfig]


@dataclass(frozen=True)
class ConfigBundle:
    backtest: BacktestConfig
    strategy: AnyStrategyConfig
    synthetic: SyntheticConfig
