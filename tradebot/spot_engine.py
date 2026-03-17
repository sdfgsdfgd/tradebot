"""Shared spot signal evaluation pipeline (UI + backtests).

This module centralizes "signal bar" feature computation:
- entry signal (EMA / ORB)
- optional regime gating (EMA regime or Supertrend, incl. shock/cooling supertrend variants)
- optional shock detector updates (ATR ratio / TR ratio / daily ATR% / daily drawdown)
- rolling realized vol (EWMA of log returns, annualized)
- rolling volume EMA for volume ratio filters
- optional ATR engine used for ATR-based exits

It intentionally does *not* simulate trades, fills, costs, or portfolio state.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from collections.abc import Mapping
from typing import Protocol

from .series import BarSeries, bars_list
from .engine import (
    EmaDecisionEngine,
    EmaDecisionSnapshot,
    OrbDecisionEngine,
    RiskOverlaySnapshot,
    SupertrendEngine,
    _trade_date as _trade_date_shared,
    _trade_hour_et as _trade_hour_et_shared,
    annualized_ewma_vol,
    build_shock_engine,
    build_tr_pct_risk_overlay_engine,
    normalize_spot_entry_signal,
    normalize_spot_regime_mode,
    normalize_shock_detector,
    normalize_shock_direction_source,
    parse_time_hhmm,
    resolve_spot_regime2_spec,
    spot_regime_apply_matches_direction,
)
from .spot.lifecycle import apply_regime_gate
from .signals import ema_next, ema_periods
from .time_utils import NaiveTsModeInput, normalize_naive_ts_mode


# region Protocols / Helpers
class BarLike(Protocol):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _bars_input_list(values: list[BarLike] | BarSeries[BarLike] | None) -> list[BarLike]:
    if values is None:
        return []
    return bars_list(values)

def _get(obj: Mapping[str, object] | object | None, key: str, default: object = None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)
# endregion


# region Models
@dataclass(frozen=True)
class SpotSignalSnapshot:
    bar_ts: datetime
    close: float
    signal: EmaDecisionSnapshot
    bars_in_day: int
    rv: float | None
    volume: float | None
    volume_ema: float | None
    volume_ema_ready: bool
    shock: bool | None
    shock_dir: str | None
    shock_detector: str | None
    shock_direction_source_effective: str | None
    shock_scale_detector: str | None
    shock_dir_ret_sum_pct: float | None
    shock_atr_pct: float | None
    shock_drawdown_pct: float | None
    shock_drawdown_on_pct: float | None
    shock_drawdown_off_pct: float | None
    shock_drawdown_dist_on_pct: float | None
    shock_drawdown_dist_on_vel_pp: float | None
    shock_drawdown_dist_on_accel_pp: float | None
    shock_prearm_down_streak_bars: int | None
    shock_drawdown_dist_off_pct: float | None
    shock_scale_drawdown_pct: float | None
    shock_peak_close: float | None
    shock_dir_down_streak_bars: int | None
    shock_dir_up_streak_bars: int | None
    risk: RiskOverlaySnapshot | None
    atr: float | None
    regime2_dir: str | None
    regime2_ready: bool
    regime2_bear_hard_dir: str | None
    regime2_bear_hard_ready: bool
    regime2_bear_hard_release_age_bars: int | None
    regime4_state: str | None
    regime4_transition_hot: bool
    regime4_owner: str | None
    or_high: float | None
    or_low: float | None
    or_ready: bool
    entry_dir: str | None = None
    entry_branch: str | None = None
    ratsv_side_rank: float | None = None
    ratsv_tr_ratio: float | None = None
    ratsv_fast_slope_pct: float | None = None
    ratsv_fast_slope_med_pct: float | None = None
    ratsv_fast_slope_vel_pct: float | None = None
    ratsv_slow_slope_med_pct: float | None = None
    ratsv_slow_slope_vel_pct: float | None = None
    ratsv_slope_vel_consistency: float | None = None
    ratsv_cross_age_bars: int | None = None
    shock_atr_vel_pct: float | None = None
    shock_atr_accel_pct: float | None = None
    shock_ramp: dict[str, object] | None = None
# endregion


# region Evaluator
class SpotSignalEvaluator:
    """Stateful evaluator for spot signals across sequential bars."""

    def __init__(
        self,
        *,
        strategy: Mapping[str, object] | object,
        filters: Mapping[str, object] | object | None,
        bar_size: str,
        use_rth: bool,
        naive_ts_mode: NaiveTsModeInput = "utc",
        rv_lookback: int = 60,
        rv_ewma_lambda: float = 0.94,
        regime_bars: list[BarLike] | BarSeries[BarLike] | None = None,
        regime2_bars: list[BarLike] | BarSeries[BarLike] | None = None,
        regime2_bear_hard_bars: list[BarLike] | BarSeries[BarLike] | None = None,
    ) -> None:
        self._strategy = strategy
        self._filters = filters
        self._bar_size = str(bar_size)
        self._use_rth = bool(use_rth)
        self._naive_ts_mode = normalize_naive_ts_mode(naive_ts_mode, default="utc").value

        self._rv_lookback = max(1, int(rv_lookback))
        self._rv_lam = float(rv_ewma_lambda)
        self._returns: deque[float] = deque(maxlen=self._rv_lookback)
        self._prev_sig_close: float | None = None
        self._rv_enabled = bool(
            filters is not None and (_get(filters, "rv_min", None) is not None or _get(filters, "rv_max", None) is not None)
        )

        self._sig_last_date = None
        self._sig_bars_in_day = 0

        # Entry signal
        entry_signal = normalize_spot_entry_signal(_get(strategy, "entry_signal", "ema"))
        self.entry_signal = entry_signal

        # Regime mode (primary)
        regime_mode = normalize_spot_regime_mode(_get(strategy, "regime_mode", "ema"))
        self._regime_mode = regime_mode
        regime_preset = str(_get(strategy, "regime_ema_preset", "") or "").strip() or None

        # Multi-timeframe regime: if provided, caller already fetched the right bars.
        self._use_mtf_regime = bool(regime_bars)
        self._regime_bars = _bars_input_list(regime_bars)
        self._regime_idx = 0

        # Volume EMA (only when volume_ratio_min is enabled)
        self._volume_period: int | None = None
        if filters is not None and _get(filters, "volume_ratio_min", None) is not None:
            raw_period = _get(filters, "volume_ema_period", None)
            try:
                self._volume_period = int(raw_period) if raw_period is not None else 20
            except (TypeError, ValueError):
                self._volume_period = 20
            self._volume_period = max(1, int(self._volume_period))
        self._volume_ema: float | None = None
        self._volume_count = 0

        # Optional exit ATR engine (spot_exit_mode=atr)
        exit_mode = str(_get(strategy, "spot_exit_mode", "pct") or "pct").strip().lower()
        if exit_mode not in ("pct", "atr"):
            exit_mode = "pct"
        self._exit_atr_engine: SupertrendEngine | None = None
        self._last_exit_atr = None
        if exit_mode == "atr":
            raw_atr = _get(strategy, "spot_atr_period", None)
            try:
                atr_p = int(raw_atr) if raw_atr is not None else 14
            except (TypeError, ValueError):
                atr_p = 14
            atr_p = max(1, int(atr_p))
            self._exit_atr_engine = SupertrendEngine(atr_period=atr_p, multiplier=1.0, source="hl2")

        # Signal engines
        self._signal_engine: EmaDecisionEngine | None = None
        self._signal_engine_a: EmaDecisionEngine | None = None
        self._signal_engine_b: EmaDecisionEngine | None = None
        self._orb_engine: OrbDecisionEngine | None = None
        self._dual_branch_enabled = False
        self._dual_branch_priority = "b_then_a"
        self._branch_a_min_signed_slope_pct: float | None = None
        self._branch_a_max_signed_slope_pct: float | None = None
        self._branch_b_min_signed_slope_pct: float | None = None
        self._branch_b_max_signed_slope_pct: float | None = None

        # RATS-V runtime state (default-off; only active when filters.ratsv_enabled=true).
        self._ratsv_enabled = bool(_get(filters, "ratsv_enabled", False)) if filters is not None else False
        self._ratsv_slope_window = max(1, int(_get(filters, "ratsv_slope_window_bars", 5) or 5)) if filters is not None else 5
        raw_slow_window = _get(filters, "ratsv_slope_slow_window_bars", None) if filters is not None else None
        try:
            slow_window = int(raw_slow_window) if raw_slow_window is not None else (int(self._ratsv_slope_window) * 3)
        except (TypeError, ValueError):
            slow_window = int(self._ratsv_slope_window) * 3
        self._ratsv_slope_slow_window = max(int(self._ratsv_slope_window) + 1, int(slow_window))
        self._ratsv_tr_fast = max(1, int(_get(filters, "ratsv_tr_fast_bars", 5) or 5)) if filters is not None else 5
        self._ratsv_tr_slow = max(self._ratsv_tr_fast, int(_get(filters, "ratsv_tr_slow_bars", 20) or 20)) if filters is not None else 20
        self._ratsv_tr_pct_hist: deque[float] = deque(maxlen=max(256, self._ratsv_tr_slow + 8))
        self._ratsv_prev_tr_close: float | None = None
        self._ratsv_branch_cross_age: dict[str, int | None] = {"single": None, "a": None, "b": None}
        slope_hist_maxlen = max(16, max(int(self._ratsv_slope_window), int(self._ratsv_slope_slow_window)) * 4)
        self._ratsv_branch_slope_hist: dict[str, deque[float]] = {
            "single": deque(maxlen=int(slope_hist_maxlen)),
            "a": deque(maxlen=int(slope_hist_maxlen)),
            "b": deque(maxlen=int(slope_hist_maxlen)),
        }
        self._ratsv_branch_slope_vel_hist: dict[str, deque[float]] = {
            "single": deque(maxlen=int(slope_hist_maxlen)),
            "a": deque(maxlen=int(slope_hist_maxlen)),
            "b": deque(maxlen=int(slope_hist_maxlen)),
        }
        self._ratsv_branch_last_slope_med: dict[str, float | None] = {"single": None, "a": None, "b": None}
        self._ratsv_branch_last_slope_med_slow: dict[str, float | None] = {"single": None, "a": None, "b": None}
        self._ratsv_last_candidate_metrics: dict[str, dict[str, float | int | None] | None] = {"single": None, "a": None, "b": None}

        def _ratsv_pos_float(raw) -> float | None:
            if raw is None:
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            if v <= 0:
                return None
            return float(v)

        def _ratsv_ratio(raw) -> float | None:
            if raw is None:
                return None
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return None
            return float(max(0.0, min(1.0, v)))

        def _ratsv_cross_age(raw) -> int | None:
            if raw is None:
                return None
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return None
            if v < 0:
                return None
            return int(v)

        self._ratsv_rank_min = _ratsv_ratio(_get(filters, "ratsv_rank_min", None)) if filters is not None else None
        self._ratsv_tr_ratio_min = _ratsv_pos_float(_get(filters, "ratsv_tr_ratio_min", None)) if filters is not None else None
        self._ratsv_slope_med_min_pct = _ratsv_pos_float(_get(filters, "ratsv_slope_med_min_pct", None)) if filters is not None else None
        self._ratsv_slope_vel_min_pct = _ratsv_pos_float(_get(filters, "ratsv_slope_vel_min_pct", None)) if filters is not None else None
        self._ratsv_slope_med_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_slope_med_slow_min_pct", None)) if filters is not None else None
        )
        self._ratsv_slope_vel_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_slope_vel_slow_min_pct", None)) if filters is not None else None
        )
        raw_consistency_bars = _get(filters, "ratsv_slope_vel_consistency_bars", 0) if filters is not None else 0
        try:
            consistency_bars = int(raw_consistency_bars)
        except (TypeError, ValueError):
            consistency_bars = 0
        self._ratsv_slope_vel_consistency_bars = max(0, int(consistency_bars))
        self._ratsv_slope_vel_consistency_min = (
            _ratsv_ratio(_get(filters, "ratsv_slope_vel_consistency_min", None)) if filters is not None else None
        )
        self._ratsv_cross_age_max = _ratsv_cross_age(_get(filters, "ratsv_cross_age_max_bars", None)) if filters is not None else None

        self._ratsv_branch_a_rank_min = _ratsv_ratio(_get(filters, "ratsv_branch_a_rank_min", None)) if filters is not None else None
        self._ratsv_branch_a_tr_ratio_min = _ratsv_pos_float(_get(filters, "ratsv_branch_a_tr_ratio_min", None)) if filters is not None else None
        self._ratsv_branch_a_slope_med_min_pct = _ratsv_pos_float(_get(filters, "ratsv_branch_a_slope_med_min_pct", None)) if filters is not None else None
        self._ratsv_branch_a_slope_vel_min_pct = _ratsv_pos_float(_get(filters, "ratsv_branch_a_slope_vel_min_pct", None)) if filters is not None else None
        self._ratsv_branch_a_slope_med_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_branch_a_slope_med_slow_min_pct", None)) if filters is not None else None
        )
        self._ratsv_branch_a_slope_vel_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_branch_a_slope_vel_slow_min_pct", None)) if filters is not None else None
        )
        self._regime2_soft_bear_branch_a_slope_med_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "regime2_soft_bear_branch_a_slope_med_slow_min_pct", None))
            if filters is not None
            else None
        )
        self._regime2_soft_bear_branch_a_slope_vel_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "regime2_soft_bear_branch_a_slope_vel_slow_min_pct", None))
            if filters is not None
            else None
        )
        self._ratsv_branch_a_slope_vel_consistency_min = (
            _ratsv_ratio(_get(filters, "ratsv_branch_a_slope_vel_consistency_min", None)) if filters is not None else None
        )
        raw_a_consistency_bars = _get(filters, "ratsv_branch_a_slope_vel_consistency_bars", None) if filters is not None else None
        try:
            consistency_a_bars = int(raw_a_consistency_bars) if raw_a_consistency_bars is not None else None
        except (TypeError, ValueError):
            consistency_a_bars = None
        self._ratsv_branch_a_slope_vel_consistency_bars = (
            max(0, int(consistency_a_bars)) if consistency_a_bars is not None else None
        )
        self._ratsv_branch_a_cross_age_max = _ratsv_cross_age(_get(filters, "ratsv_branch_a_cross_age_max_bars", None)) if filters is not None else None

        self._ratsv_branch_b_rank_min = _ratsv_ratio(_get(filters, "ratsv_branch_b_rank_min", None)) if filters is not None else None
        self._ratsv_branch_b_tr_ratio_min = _ratsv_pos_float(_get(filters, "ratsv_branch_b_tr_ratio_min", None)) if filters is not None else None
        self._ratsv_branch_b_slope_med_min_pct = _ratsv_pos_float(_get(filters, "ratsv_branch_b_slope_med_min_pct", None)) if filters is not None else None
        self._ratsv_branch_b_slope_vel_min_pct = _ratsv_pos_float(_get(filters, "ratsv_branch_b_slope_vel_min_pct", None)) if filters is not None else None
        self._ratsv_branch_b_slope_med_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_branch_b_slope_med_slow_min_pct", None)) if filters is not None else None
        )
        self._ratsv_branch_b_slope_vel_slow_min_pct = (
            _ratsv_pos_float(_get(filters, "ratsv_branch_b_slope_vel_slow_min_pct", None)) if filters is not None else None
        )
        self._ratsv_branch_b_slope_vel_consistency_min = (
            _ratsv_ratio(_get(filters, "ratsv_branch_b_slope_vel_consistency_min", None)) if filters is not None else None
        )
        raw_b_consistency_bars = _get(filters, "ratsv_branch_b_slope_vel_consistency_bars", None) if filters is not None else None
        try:
            consistency_b_bars = int(raw_b_consistency_bars) if raw_b_consistency_bars is not None else None
        except (TypeError, ValueError):
            consistency_b_bars = None
        self._ratsv_branch_b_slope_vel_consistency_bars = (
            max(0, int(consistency_b_bars)) if consistency_b_bars is not None else None
        )
        self._ratsv_branch_b_cross_age_max = _ratsv_cross_age(_get(filters, "ratsv_branch_b_cross_age_max_bars", None)) if filters is not None else None

        if entry_signal == "ema":
            ema_preset = str(_get(strategy, "ema_preset", "") or "").strip()
            if not ema_preset:
                raise ValueError("EMA entry requires ema_preset")
            # Mirror backtest semantics: only embed same-timeframe EMA regime inside the EMA engine.
            embedded_regime = None
            if (not self._use_mtf_regime) and self._regime_mode != "supertrend":
                embedded_regime = regime_preset
            dual_enabled = bool(_get(strategy, "spot_dual_branch_enabled", False))
            self._dual_branch_enabled = bool(dual_enabled)

            raw_priority = str(_get(strategy, "spot_dual_branch_priority", "b_then_a") or "b_then_a").strip().lower()
            if raw_priority in ("a_then_b", "a", "a_first", "a-first"):
                self._dual_branch_priority = "a_then_b"
            else:
                self._dual_branch_priority = "b_then_a"

            def _opt_slope_threshold(raw) -> float | None:
                if raw is None:
                    return None
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    return None
                if v <= 0:
                    return None
                return float(v)

            self._branch_a_min_signed_slope_pct = _opt_slope_threshold(
                _get(strategy, "spot_branch_a_min_signed_slope_pct", None)
            )
            self._branch_a_max_signed_slope_pct = _opt_slope_threshold(
                _get(strategy, "spot_branch_a_max_signed_slope_pct", None)
            )
            self._branch_b_min_signed_slope_pct = _opt_slope_threshold(
                _get(strategy, "spot_branch_b_min_signed_slope_pct", None)
            )
            self._branch_b_max_signed_slope_pct = _opt_slope_threshold(
                _get(strategy, "spot_branch_b_max_signed_slope_pct", None)
            )

            if self._dual_branch_enabled:
                branch_a_preset = str(_get(strategy, "spot_branch_a_ema_preset", ema_preset) or ema_preset).strip()
                branch_b_preset = str(_get(strategy, "spot_branch_b_ema_preset", ema_preset) or ema_preset).strip()
                if not branch_a_preset:
                    branch_a_preset = ema_preset
                if not branch_b_preset:
                    branch_b_preset = ema_preset

                base_confirm = int(_get(strategy, "entry_confirm_bars", 0) or 0)
                raw_a_confirm = _get(strategy, "spot_branch_a_entry_confirm_bars", None)
                raw_b_confirm = _get(strategy, "spot_branch_b_entry_confirm_bars", None)
                try:
                    branch_a_confirm = int(raw_a_confirm) if raw_a_confirm is not None else int(base_confirm)
                except (TypeError, ValueError):
                    branch_a_confirm = int(base_confirm)
                try:
                    branch_b_confirm = int(raw_b_confirm) if raw_b_confirm is not None else int(base_confirm)
                except (TypeError, ValueError):
                    branch_b_confirm = int(base_confirm)
                branch_a_confirm = max(0, int(branch_a_confirm))
                branch_b_confirm = max(0, int(branch_b_confirm))

                self._signal_engine_a = EmaDecisionEngine(
                    ema_preset=str(branch_a_preset),
                    ema_entry_mode=_get(strategy, "ema_entry_mode", None),
                    entry_confirm_bars=int(branch_a_confirm),
                    regime_ema_preset=embedded_regime,
                )
                self._signal_engine_b = EmaDecisionEngine(
                    ema_preset=str(branch_b_preset),
                    ema_entry_mode=_get(strategy, "ema_entry_mode", None),
                    entry_confirm_bars=int(branch_b_confirm),
                    regime_ema_preset=embedded_regime,
                )
            else:
                self._signal_engine = EmaDecisionEngine(
                    ema_preset=ema_preset,
                    ema_entry_mode=_get(strategy, "ema_entry_mode", None),
                    entry_confirm_bars=int(_get(strategy, "entry_confirm_bars", 0) or 0),
                    regime_ema_preset=embedded_regime,
                )
        else:
            raw_window = _get(strategy, "orb_window_mins", None)
            try:
                window = int(raw_window) if raw_window is not None else 15
            except (TypeError, ValueError):
                window = 15
            window = max(1, int(window))
            orb_open = parse_time_hhmm(_get(strategy, "orb_open_time_et", None), default=time(9, 30)) or time(9, 30)
            self._orb_engine = OrbDecisionEngine(window_mins=window, open_time_et=orb_open)

        # Primary regime engines
        self._regime_engine: EmaDecisionEngine | None = None
        if self._regime_mode == "ema" and regime_preset:
            self._regime_engine = EmaDecisionEngine(
                ema_preset=str(regime_preset),
                ema_entry_mode="trend",
                entry_confirm_bars=0,
                regime_ema_preset=None,
            )
        self._supertrend_engine: SupertrendEngine | None = None
        self._supertrend_shock_engine: SupertrendEngine | None = None
        self._supertrend_cooling_engine: SupertrendEngine | None = None
        self._last_supertrend = None
        self._last_supertrend_shock = None
        self._last_supertrend_cooling = None
        if self._regime_mode == "supertrend":
            try:
                st_atr_p = int(_get(strategy, "supertrend_atr_period", 10) or 10)
            except (TypeError, ValueError):
                st_atr_p = 10
            try:
                st_mult = float(_get(strategy, "supertrend_multiplier", 3.0) or 3.0)
            except (TypeError, ValueError):
                st_mult = 3.0
            st_src = str(_get(strategy, "supertrend_source", "hl2") or "hl2").strip() or "hl2"
            self._supertrend_engine = SupertrendEngine(atr_period=int(st_atr_p), multiplier=float(st_mult), source=st_src)

            if filters is not None:
                shock_st_mult = _get(filters, "shock_regime_supertrend_multiplier", None)
                if shock_st_mult is not None and float(shock_st_mult) > 0:
                    self._supertrend_shock_engine = SupertrendEngine(
                        atr_period=int(st_atr_p),
                        multiplier=float(shock_st_mult),
                        source=st_src,
                    )
                cooling_st_mult = _get(filters, "shock_cooling_regime_supertrend_multiplier", None)
                if cooling_st_mult is not None and float(cooling_st_mult) > 0:
                    self._supertrend_cooling_engine = SupertrendEngine(
                        atr_period=int(st_atr_p),
                        multiplier=float(cooling_st_mult),
                        source=st_src,
                    )

        # Shock engine
        self._shock_detector = normalize_shock_detector(filters)
        self._shock_dir_source = normalize_shock_direction_source(filters)
        st_src_for_shock = str(_get(strategy, "supertrend_source", "hl2") or "hl2").strip() or "hl2"
        self._shock_engine = build_shock_engine(filters, source=st_src_for_shock)
        self._last_shock = None
        self._shock_scale_detector: str | None = None
        self._shock_scale_engine = None
        self._last_shock_scale = None
        if filters is not None:
            raw_scale = _get(filters, "shock_scale_detector", None)
            cleaned = str(raw_scale).strip().lower() if raw_scale is not None else ""
            if cleaned and cleaned not in ("0", "false", "none", "null", "off"):
                scale_detector = normalize_shock_detector({"shock_detector": cleaned})
                if scale_detector in ("atr_ratio", "tr_ratio", "daily_atr_pct", "daily_drawdown"):
                    self._shock_scale_detector = str(scale_detector)
                    scale_filters: dict[str, object] = {
                        # Scale-only detector: always detect-only (no gating) by construction.
                        "shock_gate_mode": "detect",
                        "shock_detector": str(scale_detector),
                        # Reuse the existing shock knob family for the scale detector.
                        "shock_atr_fast_period": _get(filters, "shock_atr_fast_period", 7),
                        "shock_atr_slow_period": _get(filters, "shock_atr_slow_period", 50),
                        "shock_on_ratio": _get(filters, "shock_on_ratio", 1.55),
                        "shock_off_ratio": _get(filters, "shock_off_ratio", 1.30),
                        "shock_min_atr_pct": _get(filters, "shock_min_atr_pct", 7.0),
                        "shock_daily_atr_period": _get(filters, "shock_daily_atr_period", 14),
                        "shock_daily_on_atr_pct": _get(filters, "shock_daily_on_atr_pct", 13.0),
                        "shock_daily_off_atr_pct": _get(filters, "shock_daily_off_atr_pct", 11.0),
                        "shock_daily_on_tr_pct": _get(filters, "shock_daily_on_tr_pct", None),
                        "shock_drawdown_lookback_days": _get(filters, "shock_drawdown_lookback_days", 20),
                        "shock_on_drawdown_pct": _get(filters, "shock_on_drawdown_pct", -20.0),
                        "shock_off_drawdown_pct": _get(filters, "shock_off_drawdown_pct", -10.0),
                        "shock_direction_source": _get(filters, "shock_direction_source", "regime"),
                        "shock_direction_lookback": _get(filters, "shock_direction_lookback", 2),
                    }
                    self._shock_scale_engine = build_shock_engine(scale_filters, source=st_src_for_shock)

        # Optional: auxiliary daily-drawdown engine.
        #
        # Motivation: dd-based gates (prearm / depth bands / rebound boost / ramp) operate on
        # `shock_drawdown_*` telemetry, but the main shock detector is often ATR/TR ratio for
        # HF. In slow downturn regimes the ratio shock can stay off, and the dd gates never
        # get their inputs. We keep shock detector semantics unchanged, and only compute the
        # dd telemetry when a dd gate is configured.
        self._aux_drawdown_engine = None
        self._last_aux_drawdown = None
        if filters is not None and str(self._shock_detector) != "daily_drawdown":
            dd_needed = bool(_get(filters, "shock_ramp_enable", False))
            if not dd_needed:
                for k in (
                    "shock_prearm_dist_on_max_pp",
                    "shock_short_boost_max_dist_on_pp",
                    "shock_short_entry_max_dist_on_pp",
                    "shock_long_boost_max_dist_off_pp",
                ):
                    raw = _get(filters, k, 0.0)
                    try:
                        v = float(raw) if raw is not None else 0.0
                    except (TypeError, ValueError):
                        v = 0.0
                    if v > 0:
                        dd_needed = True
                        break
            if dd_needed:
                dd_filters: dict[str, object] = {
                    "shock_gate_mode": "detect",
                    "shock_detector": "daily_drawdown",
                    "shock_drawdown_lookback_days": _get(filters, "shock_drawdown_lookback_days", 20),
                    "shock_on_drawdown_pct": _get(filters, "shock_on_drawdown_pct", -20.0),
                    "shock_off_drawdown_pct": _get(filters, "shock_off_drawdown_pct", -10.0),
                    "shock_direction_lookback": _get(filters, "shock_direction_lookback", 2),
                }
                self._aux_drawdown_engine = build_shock_engine(dd_filters, source=st_src_for_shock)

        # Risk overlay (daily TR% heuristics)
        self._risk_overlay = build_tr_pct_risk_overlay_engine(filters)
        self._last_risk: RiskOverlaySnapshot | None = None

        # Regime2 gating (secondary)
        regime2_mode, regime2_preset, _regime2_bar_size, _use_mtf_regime2_cfg = resolve_spot_regime2_spec(
            bar_size=self._bar_size,
            regime2_mode_raw=_get(strategy, "regime2_mode", "off"),
            regime2_ema_preset_raw=_get(strategy, "regime2_ema_preset", ""),
            regime2_bar_size_raw=_get(strategy, "regime2_bar_size", ""),
        )
        self._regime2_mode = regime2_mode

        self._use_mtf_regime2 = bool(regime2_bars)
        self._regime2_bars = _bars_input_list(regime2_bars)
        self._regime2_idx = 0

        self._regime2_engine: EmaDecisionEngine | None = None
        if regime2_mode == "ema" and regime2_preset:
            self._regime2_engine = EmaDecisionEngine(
                ema_preset=str(regime2_preset),
                ema_entry_mode="trend",
                entry_confirm_bars=0,
                regime_ema_preset=None,
            )
        self._supertrend2_engine: SupertrendEngine | None = None
        if regime2_mode == "supertrend":
            try:
                atr_p = int(_get(strategy, "regime2_supertrend_atr_period", 10) or 10)
            except (TypeError, ValueError):
                atr_p = 10
            try:
                mult = float(_get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0)
            except (TypeError, ValueError):
                mult = 3.0
            src = str(_get(strategy, "regime2_supertrend_source", "hl2") or "hl2").strip() or "hl2"
            self._supertrend2_engine = SupertrendEngine(atr_period=int(atr_p), multiplier=float(mult), source=src)
        self._last_regime2 = None
        self._last_supertrend2 = None
        self._active_regime2_dir: str | None = None
        self._active_regime2_ready: bool = False
        self._regime2_clean_host_enable = bool(_get(strategy, "regime2_clean_host_enable", False))
        self._regime2_clean_host_takeover_state = str(
            _get(strategy, "regime2_clean_host_takeover_state", "trend_up_clean") or "trend_up_clean"
        ).strip().lower()
        if self._regime2_clean_host_takeover_state not in (
            "trend_up_clean",
            "crash_down",
            "transition_up_hot",
            "crash_or_transition_up_hot",
        ):
            self._regime2_clean_host_takeover_state = "trend_up_clean"
        self._clean_supertrend2_engine: SupertrendEngine | None = None
        self._last_clean_supertrend2 = None
        self._clean_regime2_idx = 0
        self._active_clean_regime2_dir: str | None = None
        self._active_clean_regime2_ready: bool = False
        if self._regime2_clean_host_enable and regime2_mode == "supertrend":
            raw_clean_mult = _get(strategy, "regime2_clean_host_supertrend_multiplier", None)
            try:
                clean_mult = (
                    float(raw_clean_mult)
                    if raw_clean_mult is not None
                    else float(_get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0)
                )
            except (TypeError, ValueError):
                clean_mult = float(_get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0)
            self._clean_supertrend2_engine = SupertrendEngine(
                atr_period=int(atr_p),
                multiplier=max(0.01, float(clean_mult)),
                source=src,
            )
        self._regime2_bear_hard_mode = str(_get(strategy, "regime2_bear_hard_mode", "off") or "off").strip().lower()
        if self._regime2_bear_hard_mode not in ("off", "supertrend"):
            self._regime2_bear_hard_mode = "off"
        self._use_mtf_regime2_bear_hard = bool(regime2_bear_hard_bars)
        self._regime2_bear_hard_bars = _bars_input_list(regime2_bear_hard_bars)
        self._regime2_bear_hard_idx = 0
        self._bear_hard_supertrend_engine: SupertrendEngine | None = None
        self._last_bear_hard_supertrend = None
        self._active_regime2_bear_hard_dir: str | None = None
        self._active_regime2_bear_hard_ready: bool = False
        self._clean_bear_hard_supertrend_engine: SupertrendEngine | None = None
        self._last_clean_bear_hard_supertrend = None
        self._clean_regime2_bear_hard_idx = 0
        self._active_clean_regime2_bear_hard_dir: str | None = None
        self._active_clean_regime2_bear_hard_ready: bool = False
        if self._regime2_bear_hard_mode == "supertrend":
            raw_hard_atr = _get(strategy, "regime2_bear_hard_supertrend_atr_period", None)
            raw_hard_mult = _get(strategy, "regime2_bear_hard_supertrend_multiplier", None)
            raw_hard_src = _get(strategy, "regime2_bear_hard_supertrend_source", None)
            try:
                hard_atr = int(raw_hard_atr) if raw_hard_atr is not None else int(_get(strategy, "regime2_supertrend_atr_period", 10) or 10)
            except (TypeError, ValueError):
                hard_atr = int(_get(strategy, "regime2_supertrend_atr_period", 10) or 10)
            try:
                hard_mult = (
                    float(raw_hard_mult)
                    if raw_hard_mult is not None
                    else float(_get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0)
                )
            except (TypeError, ValueError):
                hard_mult = float(_get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0)
            hard_src = str(raw_hard_src or _get(strategy, "regime2_supertrend_source", "hl2") or "hl2").strip() or "hl2"
            self._bear_hard_supertrend_engine = SupertrendEngine(
                atr_period=max(1, int(hard_atr)),
                multiplier=max(0.01, float(hard_mult)),
                source=hard_src,
            )
            if self._regime2_clean_host_enable:
                raw_clean_hard_mult = _get(strategy, "regime2_clean_host_bear_hard_supertrend_multiplier", None)
                try:
                    clean_hard_mult = (
                        float(raw_clean_hard_mult)
                        if raw_clean_hard_mult is not None
                        else float(_get(strategy, "regime2_bear_hard_supertrend_multiplier", hard_mult) or hard_mult)
                    )
                except (TypeError, ValueError):
                    clean_hard_mult = float(_get(strategy, "regime2_bear_hard_supertrend_multiplier", hard_mult) or hard_mult)
                self._clean_bear_hard_supertrend_engine = SupertrendEngine(
                    atr_period=max(1, int(hard_atr)),
                    multiplier=max(0.01, float(clean_hard_mult)),
                    source=hard_src,
                )
        self._regime2_bear_entry_mode = str(_get(strategy, "regime2_bear_entry_mode", "off") or "off").strip().lower()
        if self._regime2_bear_entry_mode not in ("off", "supertrend"):
            self._regime2_bear_entry_mode = "off"
        self._regime2_bear_allow_long_recovery = bool(_get(strategy, "regime2_bear_allow_long_recovery", True))
        self._regime2_bear_takeover_mode = str(
            _get(strategy, "regime2_bear_takeover_mode", "always") or "always"
        ).strip().lower()
        if self._regime2_bear_takeover_mode not in (
            "always",
            "hostile",
            "riskoff",
            "riskpanic",
            "shockdown",
            "hostile_or_shockdown",
        ):
            self._regime2_bear_takeover_mode = "always"
        raw_crash_atr_min = _get(strategy, "regime2_crash_atr_pct_min", None)
        try:
            self._regime2_crash_atr_pct_min = (
                float(raw_crash_atr_min) if raw_crash_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_crash_atr_pct_min = None
        if self._regime2_crash_atr_pct_min is not None and self._regime2_crash_atr_pct_min < 0:
            self._regime2_crash_atr_pct_min = None
        self._regime2_crash_block_longs = bool(_get(strategy, "regime2_crash_block_longs", False))
        self._regime2_crash_prearm_apply_to = (
            str(_get(strategy, "regime2_crash_prearm_apply_to", "off") or "off").strip().lower()
        )
        if self._regime2_crash_prearm_apply_to not in ("off", "branch_b_longs", "all_longs"):
            self._regime2_crash_prearm_apply_to = "off"
        raw_crash_prearm_atr_min = _get(strategy, "regime2_crash_prearm_shock_atr_pct_min", None)
        try:
            self._regime2_crash_prearm_shock_atr_pct_min = (
                float(raw_crash_prearm_atr_min) if raw_crash_prearm_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_crash_prearm_shock_atr_pct_min = None
        if (
            self._regime2_crash_prearm_shock_atr_pct_min is not None
            and self._regime2_crash_prearm_shock_atr_pct_min < 0
        ):
            self._regime2_crash_prearm_shock_atr_pct_min = None
        raw_crash_prearm_ret_sum_max = _get(strategy, "regime2_crash_prearm_shock_dir_ret_sum_pct_max", None)
        try:
            self._regime2_crash_prearm_shock_dir_ret_sum_pct_max = (
                float(raw_crash_prearm_ret_sum_max) if raw_crash_prearm_ret_sum_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_crash_prearm_shock_dir_ret_sum_pct_max = None
        raw_crash_prearm_branch_a_atr_min = _get(strategy, "regime2_crash_prearm_branch_a_shock_atr_pct_min", None)
        try:
            self._regime2_crash_prearm_branch_a_shock_atr_pct_min = (
                float(raw_crash_prearm_branch_a_atr_min) if raw_crash_prearm_branch_a_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_crash_prearm_branch_a_shock_atr_pct_min = None
        if (
            self._regime2_crash_prearm_branch_a_shock_atr_pct_min is not None
            and self._regime2_crash_prearm_branch_a_shock_atr_pct_min < 0
        ):
            self._regime2_crash_prearm_branch_a_shock_atr_pct_min = None
        raw_crash_prearm_branch_a_ret_sum_max = _get(
            strategy,
            "regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max",
            None,
        )
        try:
            self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max = (
                float(raw_crash_prearm_branch_a_ret_sum_max)
                if raw_crash_prearm_branch_a_ret_sum_max is not None
                else None
            )
        except (TypeError, ValueError):
            self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max = None
        self._regime2_repair_block_branch_b_longs = bool(
            _get(strategy, "regime2_repair_block_branch_b_longs", False)
        )
        raw_repair_b_atr_max = _get(strategy, "regime2_repair_branch_b_long_max_shock_atr_pct", None)
        try:
            self._regime2_repair_branch_b_long_max_shock_atr_pct = (
                float(raw_repair_b_atr_max) if raw_repair_b_atr_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_repair_branch_b_long_max_shock_atr_pct = None
        if (
            self._regime2_repair_branch_b_long_max_shock_atr_pct is not None
            and self._regime2_repair_branch_b_long_max_shock_atr_pct < 0
        ):
            self._regime2_repair_branch_b_long_max_shock_atr_pct = None
        raw_repair_b_after_hour = _get(strategy, "regime2_repair_branch_b_long_block_after_hour_et", None)
        try:
            self._regime2_repair_branch_b_long_block_after_hour_et = (
                int(raw_repair_b_after_hour) if raw_repair_b_after_hour is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_repair_branch_b_long_block_after_hour_et = None
        if self._regime2_repair_branch_b_long_block_after_hour_et is not None:
            self._regime2_repair_branch_b_long_block_after_hour_et = max(
                0,
                min(23, int(self._regime2_repair_branch_b_long_block_after_hour_et)),
            )
        raw_transition_hot_shock_atr = _get(strategy, "regime2_transition_hot_shock_atr_pct_min", None)
        try:
            self._regime2_transition_hot_shock_atr_pct_min = (
                float(raw_transition_hot_shock_atr) if raw_transition_hot_shock_atr is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_transition_hot_shock_atr_pct_min = None
        if (
            self._regime2_transition_hot_shock_atr_pct_min is not None
            and self._regime2_transition_hot_shock_atr_pct_min < 0
        ):
            self._regime2_transition_hot_shock_atr_pct_min = None
        raw_transition_hot_release = _get(strategy, "regime2_transition_hot_release_max_bars", None)
        try:
            self._regime2_transition_hot_release_max_bars = (
                int(raw_transition_hot_release) if raw_transition_hot_release is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_transition_hot_release_max_bars = None
        if self._regime2_transition_hot_release_max_bars is not None:
            self._regime2_transition_hot_release_max_bars = max(0, int(self._regime2_transition_hot_release_max_bars))
        raw_upcorridor_mid_min = _get(strategy, "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min", None)
        try:
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min = (
                float(raw_upcorridor_mid_min) if raw_upcorridor_mid_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min = None
        if (
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min < 0
        ):
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min = None
        raw_upcorridor_mid_max = _get(strategy, "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max", None)
        try:
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max = (
                float(raw_upcorridor_mid_max) if raw_upcorridor_mid_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max = None
        if (
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max < 0
        ):
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max = None
        if (
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max
            < self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min
        ):
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max = (
                self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min
            )
        raw_upcorridor_extreme_min = _get(strategy, "regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min", None)
        try:
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min = (
                float(raw_upcorridor_extreme_min) if raw_upcorridor_extreme_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min = None
        if (
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min is not None
            and self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min < 0
        ):
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min = None
        raw_upcorridor_fresh_max = _get(
            strategy,
            "regime2_upcorridor_branch_a_long_fresh_release_age_max_bars",
            None,
        )
        try:
            self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars = (
                int(raw_upcorridor_fresh_max) if raw_upcorridor_fresh_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars = None
        if self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars is not None:
            self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars = max(
                0,
                int(self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars),
            )
        raw_upcorridor_stale_min = _get(
            strategy,
            "regime2_upcorridor_branch_a_long_stale_release_age_min_bars",
            None,
        )
        try:
            self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars = (
                int(raw_upcorridor_stale_min) if raw_upcorridor_stale_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars = None
        if self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars is not None:
            self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars = max(
                0,
                int(self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars),
            )
        raw_upcorridor_b_stale_min = _get(
            strategy,
            "regime2_upcorridor_branch_b_long_stale_release_age_min_bars",
            None,
        )
        try:
            self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars = (
                int(raw_upcorridor_b_stale_min) if raw_upcorridor_b_stale_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars = None
        if self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars is not None:
            self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars = max(
                0,
                int(self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars),
            )
        raw_upcorridor_b_flat_low_atr_max = _get(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max",
            None,
        )
        try:
            self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max = (
                float(raw_upcorridor_b_flat_low_atr_max)
                if raw_upcorridor_b_flat_low_atr_max is not None
                else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max = None
        if (
            self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max is not None
            and self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max < 0
        ):
            self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max = None
        raw_upcorridor_b_flat_low_stale_min = _get(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars",
            None,
        )
        try:
            self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars = (
                int(raw_upcorridor_b_flat_low_stale_min)
                if raw_upcorridor_b_flat_low_stale_min is not None
                else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars = None
        if self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars is not None:
            self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars = max(
                0,
                int(self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars),
            )
        raw_upcorridor_b_flat_atr_max = _get(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max",
            None,
        )
        try:
            self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max = (
                float(raw_upcorridor_b_flat_atr_max) if raw_upcorridor_b_flat_atr_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max = None
        if (
            self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max is not None
            and self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max < 0
        ):
            self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max = None
        raw_upcorridor_b_flat_ddv_abs_max = _get(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp",
            None,
        )
        try:
            self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp = (
                float(raw_upcorridor_b_flat_ddv_abs_max) if raw_upcorridor_b_flat_ddv_abs_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp = None
        if (
            self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp is not None
            and self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp < 0
        ):
            self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp = None
        raw_trenddown_b_hard_up_age_min = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_release_age_min_bars",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars = (
                int(raw_trenddown_b_hard_up_age_min) if raw_trenddown_b_hard_up_age_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars = None
        if self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars is not None:
            self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars = max(
                0,
                int(self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars),
            )
        raw_trenddown_b_hard_up_age_max = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_release_age_max_bars",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars = (
                int(raw_trenddown_b_hard_up_age_max) if raw_trenddown_b_hard_up_age_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars = None
        if self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars is not None:
            self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars = max(
                0,
                int(self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars),
            )
        if (
            self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars is not None
            and self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars is not None
            and self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars
            < self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars
        ):
            self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars = (
                self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars
            )
        raw_trenddown_b_hard_up_atr_min = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min = (
                float(raw_trenddown_b_hard_up_atr_min) if raw_trenddown_b_hard_up_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min is not None
            and self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min < 0
        ):
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min = None
        raw_trenddown_b_hard_up_atr_max = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max = (
                float(raw_trenddown_b_hard_up_atr_max) if raw_trenddown_b_hard_up_atr_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max is not None
            and self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max < 0
        ):
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min is not None
            and self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max is not None
            and self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max
            < self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min
        ):
            self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max = (
                self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min
            )
        raw_trenddown_b_hard_up_ddv_min = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_ddv_min_pp",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp = (
                float(raw_trenddown_b_hard_up_ddv_min) if raw_trenddown_b_hard_up_ddv_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp = None
        raw_trenddown_b_hard_up_ddv_max = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_ddv_max_pp",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp = (
                float(raw_trenddown_b_hard_up_ddv_max) if raw_trenddown_b_hard_up_ddv_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp is not None
            and self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp is not None
            and self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp
            < self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp
        ):
            self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp = (
                self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp
            )
        raw_trenddown_recovery_atr_min = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min = (
                float(raw_trenddown_recovery_atr_min) if raw_trenddown_recovery_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min < 0
        ):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min = None
        raw_trenddown_recovery_atr_max = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max = (
                float(raw_trenddown_recovery_atr_max) if raw_trenddown_recovery_atr_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max < 0
        ):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max
            < self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min
        ):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max = (
                self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min
            )
        raw_trenddown_recovery_ddv_min = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp = (
                float(raw_trenddown_recovery_ddv_min) if raw_trenddown_recovery_ddv_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp = None
        raw_trenddown_recovery_ddv_max = _get(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp",
            None,
        )
        try:
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp = (
                float(raw_trenddown_recovery_ddv_max) if raw_trenddown_recovery_ddv_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp = None
        if (
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp is not None
            and self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp
            < self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp
        ):
            self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp = (
                self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp
            )
        raw_continuation_conf_age_min = _get(
            strategy,
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars = (
                int(raw_continuation_conf_age_min) if raw_continuation_conf_age_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars = None
        if self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars is not None:
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars = max(
                0,
                int(self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars),
            )
        raw_continuation_conf_age_max = _get(
            strategy,
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars = (
                int(raw_continuation_conf_age_max) if raw_continuation_conf_age_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars = None
        if self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars is not None:
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars = max(
                0,
                int(self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars),
            )
        if (
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars is not None
            and self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars is not None
            and self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars
            < self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars
        ):
            self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars = (
                self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars
            )
        self._bear_supertrend_engine: SupertrendEngine | None = None
        self._last_bear_supertrend = None
        self._bear_prev_dir: str | None = None
        self._clean_bear_supertrend_engine: SupertrendEngine | None = None
        self._last_clean_bear_supertrend = None
        self._clean_bear_prev_dir: str | None = None
        if self._regime2_bear_entry_mode == "supertrend":
            raw_bear_atr = _get(strategy, "regime2_bear_supertrend_atr_period", None)
            raw_bear_mult = _get(strategy, "regime2_bear_supertrend_multiplier", None)
            raw_bear_src = _get(strategy, "regime2_bear_supertrend_source", None)
            try:
                bear_atr = int(raw_bear_atr) if raw_bear_atr is not None else int(_get(strategy, "supertrend_atr_period", 10) or 10)
            except (TypeError, ValueError):
                bear_atr = int(_get(strategy, "supertrend_atr_period", 10) or 10)
            try:
                bear_mult = (
                    float(raw_bear_mult)
                    if raw_bear_mult is not None
                    else float(_get(strategy, "supertrend_multiplier", 3.0) or 3.0)
                )
            except (TypeError, ValueError):
                bear_mult = float(_get(strategy, "supertrend_multiplier", 3.0) or 3.0)
            bear_src = str(raw_bear_src or _get(strategy, "supertrend_source", "hl2") or "hl2").strip() or "hl2"
            self._bear_supertrend_engine = SupertrendEngine(
                atr_period=max(1, int(bear_atr)),
                multiplier=max(0.01, float(bear_mult)),
                source=bear_src,
            )
            if self._regime2_clean_host_enable:
                raw_clean_bear_mult = _get(strategy, "regime2_clean_host_bear_supertrend_multiplier", None)
                try:
                    clean_bear_mult = (
                        float(raw_clean_bear_mult)
                        if raw_clean_bear_mult is not None
                        else float(_get(strategy, "regime2_bear_supertrend_multiplier", bear_mult) or bear_mult)
                    )
                except (TypeError, ValueError):
                    clean_bear_mult = float(_get(strategy, "regime2_bear_supertrend_multiplier", bear_mult) or bear_mult)
                self._clean_bear_supertrend_engine = SupertrendEngine(
                    atr_period=max(1, int(bear_atr)),
                    multiplier=max(0.01, float(clean_bear_mult)),
                    source=bear_src,
                )

        self._last_signal: EmaDecisionSnapshot | None = None
        self._last_snapshot: SpotSignalSnapshot | None = None
        self._prev_shock_atr_pct: float | None = None
        self._prev_shock_atr_vel_pct: float | None = None
        self._prev_shock_drawdown_dist_on_pct: float | None = None
        self._prev_shock_drawdown_dist_on_vel_pp: float | None = None
        self._shock_prearm_down_streak_bars: int = 0
        self._prev_ema_slope_pct: float | None = None
        self._ema_slope_up_streak_bars: int = 0
        self._ema_slope_down_streak_bars: int = 0
        self._shock_dir_down_streak_bars: int = 0
        self._shock_dir_up_streak_bars: int = 0
        self._regime2_bear_hard_prev_was_down: bool = False
        self._regime2_bear_hard_release_age_bars: int | None = None
        self._clean_regime2_bear_hard_prev_was_down: bool = False
        self._clean_regime2_bear_hard_release_age_bars: int | None = None
        self._active_regime2_bear_hard_release_age_bars: int | None = None
        self._regime4_transition_hot: bool = False
        self._regime4_owner: str | None = None

        # Validate EMA presets early for UI ergonomics.
        if entry_signal == "ema":
            if ema_periods(str(_get(strategy, "ema_preset", "") or "")) is None:
                raise ValueError("Invalid ema_preset")
        if self._regime_mode == "ema" and regime_preset:
            if ema_periods(str(regime_preset)) is None:
                raise ValueError("Invalid regime_ema_preset")
        if self._regime2_mode == "ema" and regime2_preset:
            if ema_periods(str(regime2_preset)) is None:
                raise ValueError("Invalid regime2_ema_preset")

    @property
    def last_snapshot(self) -> SpotSignalSnapshot | None:
        return self._last_snapshot

    @property
    def shock_enabled(self) -> bool:
        return self._shock_engine is not None

    @property
    def shock_view(self) -> tuple[bool | None, str | None, float | None]:
        return self._shock_view()

    @property
    def risk_overlay_enabled(self) -> bool:
        return self._risk_overlay is not None

    @property
    def last_risk(self) -> RiskOverlaySnapshot | None:
        return self._last_risk

    @property
    def orb_engine(self) -> OrbDecisionEngine | None:
        return self._orb_engine

    @staticmethod
    def _signed_fast_slope_pct(signal: EmaDecisionSnapshot, close: float) -> float | None:
        if signal.ema_fast is None or signal.prev_ema_fast is None:
            return None
        if close <= 0:
            return None
        return float(signal.ema_fast - signal.prev_ema_fast) / float(close)

    @staticmethod
    def _median(values: deque[float] | list[float]) -> float | None:
        if not values:
            return None
        try:
            return float(median(values))
        except Exception:
            return None

    def _trade_date(self, ts: datetime) -> date:
        return _trade_date_shared(ts, naive_ts_mode=self._naive_ts_mode)

    def _ratsv_tr_fast_slow(self) -> tuple[float | None, float | None]:
        tr_hist = list(self._ratsv_tr_pct_hist)
        if not tr_hist:
            return None, None
        fast = self._median(tr_hist[-int(self._ratsv_tr_fast) :])
        slow = self._median(tr_hist[-int(self._ratsv_tr_slow) :])
        return fast, slow

    def _ratsv_update_bar_metrics(self, *, high: float, low: float, close: float) -> None:
        prev_close = self._ratsv_prev_tr_close
        self._ratsv_prev_tr_close = float(close)
        if close <= 0:
            return
        if prev_close is None or prev_close <= 0:
            tr = max(0.0, float(high) - float(low))
        else:
            tr = max(
                0.0,
                float(high) - float(low),
                abs(float(high) - float(prev_close)),
                abs(float(low) - float(prev_close)),
            )
        tr_pct = float(tr) / max(float(close), 1e-9)
        self._ratsv_tr_pct_hist.append(float(tr_pct))

    def _ratsv_side_rank(self, *, signal: EmaDecisionSnapshot, entry_dir: str, close: float) -> float | None:
        if close <= 0:
            return None
        if signal.ema_fast is None or signal.ema_slow is None:
            return None
        tr_fast, _tr_slow = self._ratsv_tr_fast_slow()
        if tr_fast is None or tr_fast <= 0:
            return None
        spread = (float(signal.ema_fast) - float(signal.ema_slow)) / float(close)
        aligned = float(spread) if str(entry_dir) == "up" else -float(spread)
        if aligned <= 0:
            return 0.0
        rank = float(aligned) / (float(aligned) + float(tr_fast))
        return float(max(0.0, min(1.0, rank)))

    def _ratsv_thresholds_for_branch(
        self,
        *,
        branch_key: str,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        int,
        float | None,
        int | None,
    ]:
        rank_min = self._ratsv_rank_min
        tr_ratio_min = self._ratsv_tr_ratio_min
        slope_med_min = self._ratsv_slope_med_min_pct
        slope_vel_min = self._ratsv_slope_vel_min_pct
        slope_med_slow_min = self._ratsv_slope_med_slow_min_pct
        slope_vel_slow_min = self._ratsv_slope_vel_slow_min_pct
        slope_vel_consistency_bars = int(self._ratsv_slope_vel_consistency_bars)
        slope_vel_consistency_min = self._ratsv_slope_vel_consistency_min
        cross_age_max = self._ratsv_cross_age_max
        if branch_key == "a":
            rank_min = self._ratsv_branch_a_rank_min if self._ratsv_branch_a_rank_min is not None else rank_min
            tr_ratio_min = (
                self._ratsv_branch_a_tr_ratio_min if self._ratsv_branch_a_tr_ratio_min is not None else tr_ratio_min
            )
            slope_med_min = (
                self._ratsv_branch_a_slope_med_min_pct
                if self._ratsv_branch_a_slope_med_min_pct is not None
                else slope_med_min
            )
            slope_vel_min = (
                self._ratsv_branch_a_slope_vel_min_pct
                if self._ratsv_branch_a_slope_vel_min_pct is not None
                else slope_vel_min
            )
            slope_med_slow_min = (
                self._ratsv_branch_a_slope_med_slow_min_pct
                if self._ratsv_branch_a_slope_med_slow_min_pct is not None
                else slope_med_slow_min
            )
            slope_vel_slow_min = (
                self._ratsv_branch_a_slope_vel_slow_min_pct
                if self._ratsv_branch_a_slope_vel_slow_min_pct is not None
                else slope_vel_slow_min
            )
            if self._active_regime2_ready and self._active_regime2_dir == "down":
                if self._regime2_soft_bear_branch_a_slope_med_slow_min_pct is not None:
                    slope_med_slow_min = max(
                        float(slope_med_slow_min or 0.0),
                        float(self._regime2_soft_bear_branch_a_slope_med_slow_min_pct),
                    )
                if self._regime2_soft_bear_branch_a_slope_vel_slow_min_pct is not None:
                    slope_vel_slow_min = max(
                        float(slope_vel_slow_min or 0.0),
                        float(self._regime2_soft_bear_branch_a_slope_vel_slow_min_pct),
                    )
            if self._ratsv_branch_a_slope_vel_consistency_bars is not None:
                slope_vel_consistency_bars = max(0, int(self._ratsv_branch_a_slope_vel_consistency_bars))
            if self._ratsv_branch_a_slope_vel_consistency_min is not None:
                slope_vel_consistency_min = self._ratsv_branch_a_slope_vel_consistency_min
            cross_age_max = (
                self._ratsv_branch_a_cross_age_max
                if self._ratsv_branch_a_cross_age_max is not None
                else cross_age_max
            )
        elif branch_key == "b":
            rank_min = self._ratsv_branch_b_rank_min if self._ratsv_branch_b_rank_min is not None else rank_min
            tr_ratio_min = (
                self._ratsv_branch_b_tr_ratio_min if self._ratsv_branch_b_tr_ratio_min is not None else tr_ratio_min
            )
            slope_med_min = (
                self._ratsv_branch_b_slope_med_min_pct
                if self._ratsv_branch_b_slope_med_min_pct is not None
                else slope_med_min
            )
            slope_vel_min = (
                self._ratsv_branch_b_slope_vel_min_pct
                if self._ratsv_branch_b_slope_vel_min_pct is not None
                else slope_vel_min
            )
            slope_med_slow_min = (
                self._ratsv_branch_b_slope_med_slow_min_pct
                if self._ratsv_branch_b_slope_med_slow_min_pct is not None
                else slope_med_slow_min
            )
            slope_vel_slow_min = (
                self._ratsv_branch_b_slope_vel_slow_min_pct
                if self._ratsv_branch_b_slope_vel_slow_min_pct is not None
                else slope_vel_slow_min
            )
            if self._ratsv_branch_b_slope_vel_consistency_bars is not None:
                slope_vel_consistency_bars = max(0, int(self._ratsv_branch_b_slope_vel_consistency_bars))
            if self._ratsv_branch_b_slope_vel_consistency_min is not None:
                slope_vel_consistency_min = self._ratsv_branch_b_slope_vel_consistency_min
            cross_age_max = (
                self._ratsv_branch_b_cross_age_max
                if self._ratsv_branch_b_cross_age_max is not None
                else cross_age_max
            )
        return (
            rank_min,
            tr_ratio_min,
            slope_med_min,
            slope_vel_min,
            slope_med_slow_min,
            slope_vel_slow_min,
            int(slope_vel_consistency_bars),
            slope_vel_consistency_min,
            cross_age_max,
        )

    def _ratsv_branch_metrics(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        entry_dir: str | None,
    ) -> dict[str, float | int | None] | None:
        if signal is None:
            self._ratsv_last_candidate_metrics[branch_key] = None
            return None

        cross_age = self._ratsv_branch_cross_age.get(branch_key)
        if bool(signal.cross_up) or bool(signal.cross_down):
            cross_age = 0
        elif cross_age is None:
            cross_age = 1
        else:
            cross_age = int(cross_age) + 1
        self._ratsv_branch_cross_age[branch_key] = int(cross_age)

        slope_now = self._signed_fast_slope_pct(signal, float(close))
        if slope_now is not None:
            self._ratsv_branch_slope_hist[branch_key].append(float(slope_now))
        window = list(self._ratsv_branch_slope_hist[branch_key])[-int(self._ratsv_slope_window) :]
        slope_med = self._median(window)
        window_slow = list(self._ratsv_branch_slope_hist[branch_key])[-int(self._ratsv_slope_slow_window) :]
        slope_med_slow = self._median(window_slow)
        prev_med = self._ratsv_branch_last_slope_med.get(branch_key)
        prev_med_slow = self._ratsv_branch_last_slope_med_slow.get(branch_key)
        slope_vel = None
        if slope_med is not None and prev_med is not None:
            slope_vel = float(slope_med) - float(prev_med)
        slope_vel_slow = None
        if slope_med_slow is not None and prev_med_slow is not None:
            slope_vel_slow = float(slope_med_slow) - float(prev_med_slow)
        if slope_med is not None:
            self._ratsv_branch_last_slope_med[branch_key] = float(slope_med)
        if slope_med_slow is not None:
            self._ratsv_branch_last_slope_med_slow[branch_key] = float(slope_med_slow)
        if slope_vel is not None:
            self._ratsv_branch_slope_vel_hist[branch_key].append(float(slope_vel))

        tr_fast, tr_slow = self._ratsv_tr_fast_slow()
        tr_ratio = None
        if tr_fast is not None and tr_slow is not None and tr_slow > 0:
            tr_ratio = float(tr_fast) / float(tr_slow)

        side_rank = None
        if entry_dir in ("up", "down") and signal.ema_ready:
            side_rank = self._ratsv_side_rank(signal=signal, entry_dir=str(entry_dir), close=float(close))

        (
            _rank_min,
            _tr_ratio_min,
            _slope_med_min,
            _slope_vel_min,
            _slope_med_slow_min,
            _slope_vel_slow_min,
            slope_vel_consistency_bars,
            _slope_vel_consistency_min,
            _cross_age_max,
        ) = self._ratsv_thresholds_for_branch(branch_key=branch_key)
        slope_vel_consistency = None
        if entry_dir in ("up", "down") and int(slope_vel_consistency_bars) > 0:
            vel_hist = list(self._ratsv_branch_slope_vel_hist[branch_key])
            if vel_hist:
                n = min(len(vel_hist), int(slope_vel_consistency_bars))
                tail = vel_hist[-int(n) :]
                if tail:
                    if str(entry_dir) == "up":
                        aligned = sum(1 for vel in tail if float(vel) >= 0.0)
                    else:
                        aligned = sum(1 for vel in tail if float(vel) <= 0.0)
                    slope_vel_consistency = float(aligned) / float(len(tail))

        metrics = {
            "side_rank": float(side_rank) if side_rank is not None else None,
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "slope_now": float(slope_now) if slope_now is not None else None,
            "slope_med": float(slope_med) if slope_med is not None else None,
            "slope_vel": float(slope_vel) if slope_vel is not None else None,
            "slope_med_slow": float(slope_med_slow) if slope_med_slow is not None else None,
            "slope_vel_slow": float(slope_vel_slow) if slope_vel_slow is not None else None,
            "slope_vel_consistency": float(slope_vel_consistency) if slope_vel_consistency is not None else None,
            "cross_age": int(cross_age) if cross_age is not None else None,
        }
        self._ratsv_last_candidate_metrics[branch_key] = metrics
        return metrics

    def _ratsv_entry_ok(
        self,
        *,
        branch_key: str,
        entry_dir: str | None,
        metrics: dict[str, float | int | None] | None,
    ) -> bool:
        if not bool(self._ratsv_enabled):
            return True
        if entry_dir not in ("up", "down"):
            return False
        if not isinstance(metrics, dict):
            return False

        (
            rank_min,
            tr_ratio_min,
            slope_med_min,
            slope_vel_min,
            slope_med_slow_min,
            slope_vel_slow_min,
            _slope_vel_consistency_bars,
            slope_vel_consistency_min,
            cross_age_max,
        ) = self._ratsv_thresholds_for_branch(branch_key=branch_key)

        side_rank = metrics.get("side_rank")
        tr_ratio = metrics.get("tr_ratio")
        slope_med = metrics.get("slope_med")
        slope_vel = metrics.get("slope_vel")
        slope_med_slow = metrics.get("slope_med_slow")
        slope_vel_slow = metrics.get("slope_vel_slow")
        slope_vel_consistency = metrics.get("slope_vel_consistency")
        cross_age = metrics.get("cross_age")

        if rank_min is not None:
            if side_rank is None or float(side_rank) < float(rank_min):
                return False
        if tr_ratio_min is not None:
            if tr_ratio is None or float(tr_ratio) < float(tr_ratio_min):
                return False

        signed_med = None
        if slope_med is not None:
            signed_med = float(slope_med) if str(entry_dir) == "up" else -float(slope_med)
        if slope_med_min is not None:
            if signed_med is None or float(signed_med) < float(slope_med_min):
                return False

        signed_vel = None
        if slope_vel is not None:
            signed_vel = float(slope_vel) if str(entry_dir) == "up" else -float(slope_vel)
        if slope_vel_min is not None:
            if signed_vel is None or float(signed_vel) < float(slope_vel_min):
                return False

        signed_med_slow = None
        if slope_med_slow is not None:
            signed_med_slow = float(slope_med_slow) if str(entry_dir) == "up" else -float(slope_med_slow)
        if slope_med_slow_min is not None:
            if signed_med_slow is None or float(signed_med_slow) < float(slope_med_slow_min):
                return False

        signed_vel_slow = None
        if slope_vel_slow is not None:
            signed_vel_slow = float(slope_vel_slow) if str(entry_dir) == "up" else -float(slope_vel_slow)
        if slope_vel_slow_min is not None:
            if signed_vel_slow is None or float(signed_vel_slow) < float(slope_vel_slow_min):
                return False

        if slope_vel_consistency_min is not None:
            if slope_vel_consistency is None or float(slope_vel_consistency) < float(slope_vel_consistency_min):
                return False

        if cross_age_max is not None:
            if cross_age is None or int(cross_age) > int(cross_age_max):
                return False
        return True

    def _branch_signed_slope_thresholds(self, *, branch_key: str) -> tuple[float | None, float | None]:
        if branch_key == "a":
            return self._branch_a_min_signed_slope_pct, self._branch_a_max_signed_slope_pct
        if branch_key == "b":
            return self._branch_b_min_signed_slope_pct, self._branch_b_max_signed_slope_pct
        return None, None

    def _candidate_entry_dir(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        candidate_dir: str | None,
        min_signed_slope_pct: float | None,
        max_signed_slope_pct: float | None,
    ) -> str | None:
        if signal is None or not bool(signal.ema_ready):
            self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=None)
            return None
        entry_dir = candidate_dir
        if entry_dir not in ("up", "down"):
            self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=None)
            return None

        slope = self._signed_fast_slope_pct(signal, float(close))
        if min_signed_slope_pct is not None or max_signed_slope_pct is not None:
            if slope is None:
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None
            signed = float(slope) if entry_dir == "up" else -float(slope)
            if min_signed_slope_pct is not None and signed < float(min_signed_slope_pct):
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None
            if max_signed_slope_pct is not None and signed > float(max_signed_slope_pct):
                self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=str(entry_dir))
                return None

        metrics = self._ratsv_branch_metrics(
            branch_key=branch_key,
            signal=signal,
            close=float(close),
            entry_dir=str(entry_dir),
        )
        if not self._ratsv_entry_ok(branch_key=branch_key, entry_dir=str(entry_dir), metrics=metrics):
            return None
        return str(entry_dir)

    def _branch_entry_dir(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        min_signed_slope_pct: float | None,
        max_signed_slope_pct: float | None,
    ) -> str | None:
        return self._candidate_entry_dir(
            branch_key=branch_key,
            signal=signal,
            close=float(close),
            candidate_dir=getattr(signal, "entry_dir", None),
            min_signed_slope_pct=min_signed_slope_pct,
            max_signed_slope_pct=max_signed_slope_pct,
        )

    def _apply_regime2_bear_primary(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        bar: BarLike,
        close: float,
    ) -> tuple[EmaDecisionSnapshot | None, str | None]:
        if (
            signal is None
            or not bool(signal.ema_ready)
            or self._bear_supertrend_engine is None
            or not self._active_regime2_ready
            or self._active_regime2_dir != "down"
        ):
            return signal, None
        if self._regime2_bear_hard_mode != "off":
            if not self._active_regime2_bear_hard_ready or self._active_regime2_bear_hard_dir != "down":
                return signal, None
        if not self._regime2_bear_takeover_allowed():
            return signal, None

        use_clean_host = bool(self._regime4_owner == "clean_host" and self._clean_bear_supertrend_engine is not None)
        bear_engine = self._clean_bear_supertrend_engine if use_clean_host else self._bear_supertrend_engine
        if bear_engine is None:
            return signal, None

        last_bear_supertrend = self._clean_bear_supertrend_engine.update(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        ) if use_clean_host else self._bear_supertrend_engine.update(
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        if use_clean_host:
            self._last_clean_bear_supertrend = last_bear_supertrend
        else:
            self._last_bear_supertrend = last_bear_supertrend

        bear_ready = bool(last_bear_supertrend and last_bear_supertrend.ready)
        bear_dir = last_bear_supertrend.direction if last_bear_supertrend is not None else None
        bear_dir = str(bear_dir) if bear_dir in ("up", "down") else None
        prev_dir = self._clean_bear_prev_dir if use_clean_host else self._bear_prev_dir
        cross_up = bool(bear_ready and bear_dir == "up" and prev_dir == "down")
        cross_down = bool(bear_ready and bear_dir == "down" and prev_dir == "up")
        if bear_ready:
            if use_clean_host:
                self._clean_bear_prev_dir = bear_dir
            else:
                self._bear_prev_dir = bear_dir

        min_signed_slope_pct, max_signed_slope_pct = self._branch_signed_slope_thresholds(branch_key=branch_key)
        candidate_dir: str | None = None
        if bear_dir == "down":
            candidate_dir = self._candidate_entry_dir(
                branch_key=branch_key,
                signal=signal,
                close=float(close),
                candidate_dir="down",
                min_signed_slope_pct=min_signed_slope_pct,
                max_signed_slope_pct=max_signed_slope_pct,
            )
        elif bear_dir == "up" and bool(self._regime2_bear_allow_long_recovery) and bool(cross_up):
            candidate_dir = self._candidate_entry_dir(
                branch_key=branch_key,
                signal=signal,
                close=float(close),
                candidate_dir="up",
                min_signed_slope_pct=min_signed_slope_pct,
                max_signed_slope_pct=max_signed_slope_pct,
            )

        return (
            replace(
                signal,
                cross_up=bool(cross_up),
                cross_down=bool(cross_down),
                state=bear_dir,
                entry_dir=candidate_dir,
                regime_dir=bear_dir,
                regime_ready=bool(bear_ready),
            ),
            candidate_dir,
        )

    def _regime2_bear_takeover_allowed(self) -> bool:
        mode = str(self._regime2_bear_takeover_mode or "always").strip().lower()
        if mode == "always":
            return True
        risk = self._last_risk
        riskoff = bool(getattr(risk, "riskoff", False))
        riskpanic = bool(getattr(risk, "riskpanic", False))
        hostile = bool(riskoff or riskpanic)
        shock, shock_dir, _shock_atr_pct = self._shock_view()
        shockdown = bool(shock) and shock_dir == "down"
        if mode == "hostile":
            return hostile
        if mode == "riskoff":
            return riskoff
        if mode == "riskpanic":
            return riskpanic
        if mode == "shockdown":
            return shockdown
        if mode == "hostile_or_shockdown":
            return bool(hostile or shockdown)
        return True

    def _classify_regime4_state(
        self,
        *,
        shock_atr_pct: float | None,
        fast_dir: str | None,
        fast_ready: bool,
        hard_dir: str | None,
        hard_ready: bool,
        hard_release_age_bars: int | None,
    ) -> tuple[str | None, bool]:
        fast_dir = str(fast_dir) if bool(fast_ready) and fast_dir in ("up", "down") else None
        hard_dir = str(hard_dir) if bool(hard_ready) and hard_dir in ("up", "down") else None
        if hard_dir == "down":
            if fast_dir == "up":
                return "transition_up_hot", True
            if fast_dir == "down" or fast_dir is None:
                if (
                    self._regime2_crash_atr_pct_min is not None
                    and shock_atr_pct is not None
                    and float(shock_atr_pct) >= float(self._regime2_crash_atr_pct_min)
                ):
                    return "crash_down", False
                return "trend_down", False
        if fast_dir == "up":
            transition_hot = False
            if (
                self._regime2_transition_hot_shock_atr_pct_min is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct) >= float(self._regime2_transition_hot_shock_atr_pct_min)
            ):
                transition_hot = True
            if (
                not transition_hot
                and self._regime2_transition_hot_release_max_bars is not None
                and hard_release_age_bars is not None
                and int(hard_release_age_bars) <= int(self._regime2_transition_hot_release_max_bars)
            ):
                transition_hot = True
            return ("transition_up_hot" if transition_hot else "trend_up_clean"), bool(transition_hot)
        if fast_dir == "down":
            return "trend_down", False
        return None, False

    def _regime2_crash_blocks_long(self, *, regime4_state: str | None, entry_dir: str | None) -> bool:
        return bool(self._regime2_crash_block_longs and regime4_state == "crash_down" and entry_dir == "up")

    def _regime2_crash_prearm_blocks_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_ret_sum_pct: float | None,
    ) -> bool:
        apply_to = str(self._regime2_crash_prearm_apply_to or "off")
        if apply_to == "off":
            return False
        if regime4_state != "trend_down" or entry_dir != "up" or shock_dir != "down":
            return False
        branch_key = str(entry_branch or "")
        atr_pct_min = self._regime2_crash_prearm_shock_atr_pct_min
        ret_sum_pct_max = self._regime2_crash_prearm_shock_dir_ret_sum_pct_max
        if branch_key == "a":
            if self._regime2_crash_prearm_branch_a_shock_atr_pct_min is not None:
                atr_pct_min = self._regime2_crash_prearm_branch_a_shock_atr_pct_min
            if self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max is not None:
                ret_sum_pct_max = self._regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max
        if (
            atr_pct_min is not None
            and (shock_atr_pct is None or float(shock_atr_pct) < float(atr_pct_min))
        ):
            return False
        if (
            ret_sum_pct_max is not None
            and (
                shock_dir_ret_sum_pct is None
                or float(shock_dir_ret_sum_pct) > float(ret_sum_pct_max)
            )
        ):
            return False
        if apply_to == "branch_b_longs":
            return branch_key == "b"
        return True

    def _regime2_blocks_branch_b_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_drawdown_dist_on_vel_pp: float | None,
        bar_ts: datetime,
    ) -> bool:
        if not (self._dual_branch_enabled and entry_branch == "b" and entry_dir == "up"):
            return False
        if regime4_state == "transition_up_hot":
            if self._regime2_repair_block_branch_b_longs:
                return True
            if (
                self._regime2_repair_branch_b_long_max_shock_atr_pct is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct) >= float(self._regime2_repair_branch_b_long_max_shock_atr_pct)
            ):
                return True
            if self._regime2_repair_branch_b_long_block_after_hour_et is not None:
                hour_et = _trade_hour_et_shared(bar_ts, naive_ts_mode=self._naive_ts_mode)
                return int(hour_et) >= int(self._regime2_repair_branch_b_long_block_after_hour_et)
            return False

        if shock_atr_pct is None or shock_drawdown_dist_on_vel_pp is None:
            return False
        atr_value = float(shock_atr_pct)
        ddv_value = float(shock_drawdown_dist_on_vel_pp)
        release_age = self._active_regime2_bear_hard_release_age_bars
        release_age_value = int(release_age) if release_age is not None else None
        if regime4_state == "trend_down":
            age_min = self._regime2_trenddown_branch_b_long_hard_up_release_age_min_bars
            age_max = self._regime2_trenddown_branch_b_long_hard_up_release_age_max_bars
            atr_min = self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min
            atr_max = self._regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max
            ddv_min = self._regime2_trenddown_branch_b_long_hard_up_ddv_min_pp
            ddv_max = self._regime2_trenddown_branch_b_long_hard_up_ddv_max_pp
            in_primary_band = bool(
                shock_dir == "down"
                and self._active_regime2_bear_hard_dir == "up"
                and release_age_value is not None
                and age_min is not None
                and age_max is not None
                and int(age_min) <= release_age_value < int(age_max)
                and atr_min is not None
                and atr_max is not None
                and float(atr_min) <= atr_value < float(atr_max)
                and ddv_min is not None
                and ddv_max is not None
                and float(ddv_min) <= ddv_value < float(ddv_max)
            )
            recovery_atr_min = self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min
            recovery_atr_max = self._regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max
            recovery_ddv_min = self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp
            recovery_ddv_max = self._regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp
            in_recovery_band = bool(
                shock_dir == "up"
                and self._active_regime2_bear_hard_dir == "up"
                and release_age_value is not None
                and age_min is not None
                and age_max is not None
                and int(age_min) <= release_age_value < int(age_max)
                and recovery_atr_min is not None
                and recovery_atr_max is not None
                and float(recovery_atr_min) <= atr_value < float(recovery_atr_max)
                and recovery_ddv_min is not None
                and recovery_ddv_max is not None
                and float(recovery_ddv_min) <= ddv_value < float(recovery_ddv_max)
            )
            return bool(in_primary_band or in_recovery_band)
        if regime4_state != "trend_up_clean":
            return False
        stale_min = self._regime2_upcorridor_branch_b_long_stale_release_age_min_bars
        if not (
            shock_dir == "up"
            and self._active_regime2_bear_hard_dir == "up"
            and release_age_value is not None
        ):
            return False
        flat_ddv_abs_max = self._regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp
        flat_low_atr_max = self._regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max
        flat_low_stale_min = self._regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars
        flat_high_atr_max = self._regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max
        return bool(
            flat_ddv_abs_max is not None
            and abs(ddv_value) < float(flat_ddv_abs_max)
            and (
                (
                    flat_low_atr_max is not None
                    and flat_low_stale_min is not None
                    and atr_value < float(flat_low_atr_max)
                    and release_age_value >= int(flat_low_stale_min)
                )
                or (
                    flat_high_atr_max is not None
                    and stale_min is not None
                    and atr_value < float(flat_high_atr_max)
                    and (flat_low_atr_max is None or atr_value >= float(flat_low_atr_max))
                    and release_age_value >= int(stale_min)
                )
            )
        )

    def _regime2_upcorridor_blocks_branch_a_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_atr_pct: float | None,
    ) -> bool:
        if not (
            self._dual_branch_enabled
            and regime4_state in ("transition_up_hot", "trend_up_clean")
            and entry_branch == "a"
            and entry_dir == "up"
        ):
            return False
        if shock_atr_pct is None:
            return False
        atr_value = float(shock_atr_pct)
        in_mid_band = False
        if (
            self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min is not None
            and self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max is not None
        ):
            in_mid_band = (
                float(self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min)
                <= atr_value
                < float(self._regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max)
            )
        in_extreme_band = (
            self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min is not None
            and atr_value >= float(self._regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min)
        )
        if not (in_mid_band or in_extreme_band):
            return False
        release_age = self._active_regime2_bear_hard_release_age_bars
        if release_age is None:
            return False
        if regime4_state == "transition_up_hot":
            fresh_max = self._regime2_upcorridor_branch_a_long_fresh_release_age_max_bars
            return fresh_max is not None and int(release_age) <= int(fresh_max)
        stale_min = self._regime2_upcorridor_branch_a_long_stale_release_age_min_bars
        return stale_min is not None and int(release_age) >= int(stale_min)

    def _continuation_confidence_blocks_long(
        self,
        *,
        regime4_state: str | None,
        entry_dir: str | None,
        entry_branch: str | None,
        shock_dir: str | None,
    ) -> bool:
        if not (
            entry_dir == "up"
            and entry_branch == "b"
            and regime4_state == "trend_up_clean"
            and shock_dir == "up"
            and self._active_regime2_bear_hard_dir == "up"
        ):
            return False
        release_age = self._active_regime2_bear_hard_release_age_bars
        age_min = self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars
        age_max = self._regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars
        return bool(
            release_age is not None
            and age_min is not None
            and age_max is not None
            and int(age_min) <= int(release_age) < int(age_max)
        )

    def _select_dual_signal(
        self,
        *,
        close: float,
        signal_a: EmaDecisionSnapshot,
        signal_b: EmaDecisionSnapshot,
    ) -> tuple[EmaDecisionSnapshot, str | None, str | None]:
        """Return (signal_for_gate/flip, entry_dir_for_entries, entry_branch)."""
        dir_a = self._branch_entry_dir(
            branch_key="a",
            signal=signal_a,
            close=float(close),
            min_signed_slope_pct=self._branch_a_min_signed_slope_pct,
            max_signed_slope_pct=self._branch_a_max_signed_slope_pct,
        )
        dir_b = self._branch_entry_dir(
            branch_key="b",
            signal=signal_b,
            close=float(close),
            min_signed_slope_pct=self._branch_b_min_signed_slope_pct,
            max_signed_slope_pct=self._branch_b_max_signed_slope_pct,
        )

        if self._dual_branch_priority == "a_then_b":
            if dir_a in ("up", "down"):
                return signal_a, str(dir_a), "a"
            if dir_b in ("up", "down"):
                return signal_b, str(dir_b), "b"
            return signal_a, None, None

        # default: b_then_a
        if dir_b in ("up", "down"):
            return signal_b, str(dir_b), "b"
        if dir_a in ("up", "down"):
            return signal_a, str(dir_a), "a"
        return signal_b, None, None

    def update_exec_bar(self, bar: BarLike, *, is_last_bar: bool = False) -> None:
        """Update any exec-bar-driven detectors (daily engines + risk overlays)."""
        if self._risk_overlay is not None:
            self._last_risk = self._risk_overlay.update(
                ts=bar.ts,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                is_last_bar=bool(is_last_bar),
                trade_day=self._trade_date(bar.ts),
            )

        if self._shock_engine is not None and self._shock_detector in ("daily_atr_pct", "daily_drawdown"):
            self._last_shock = self._shock_engine.update(
                day=self._trade_date(bar.ts),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=(self._shock_dir_source != "signal"),
            )

        if self._shock_scale_engine is not None and self._shock_scale_detector in ("daily_atr_pct", "daily_drawdown"):
            self._last_shock_scale = self._shock_scale_engine.update(
                day=self._trade_date(bar.ts),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=False,
            )

        if self._aux_drawdown_engine is not None:
            self._last_aux_drawdown = self._aux_drawdown_engine.update(
                day=self._trade_date(bar.ts),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=False,
            )

    def _shock_view(self) -> tuple[bool | None, str | None, float | None]:
        def _atr_pct_from(snap: object | None) -> float | None:
            if snap is None:
                return None
            # Daily drawdown snapshots use a negative percent (e.g. -12.0 for -12%).
            # For scaling we treat it as a positive "magnitude" comparable to ATR%.
            dd_pct = getattr(snap, "drawdown_pct", None)
            if dd_pct is not None:
                try:
                    v = float(dd_pct)
                except (TypeError, ValueError):
                    v = None
                if v is not None:
                    return max(0.0, -float(v))
            atr_pct = getattr(snap, "atr_pct", None)
            if atr_pct is None:
                atr_pct = getattr(snap, "atr_fast_pct", None)
            if atr_pct is None:
                atr_pct = getattr(snap, "tr_fast_pct", None)
            try:
                return float(atr_pct) if atr_pct is not None else None
            except (TypeError, ValueError):
                return None

        shock = None
        shock_dir = None
        if self._shock_engine is not None and self._last_shock is not None:
            gate_ready_ok = (
                self._shock_detector in ("daily_atr_pct", "daily_drawdown")
                or bool(getattr(self._last_shock, "ready", False))
            )
            if gate_ready_ok:
                shock = bool(getattr(self._last_shock, "shock", False))
                if bool(getattr(self._last_shock, "direction_ready", False)) and getattr(self._last_shock, "direction", None) in ("up", "down"):
                    shock_dir = str(getattr(self._last_shock, "direction"))

        # Optional: override the ATR% stream used for risk scaling with a separate detector.
        # When configured, we return the scale detector's ATR% stream (or None until ready).
        if self._shock_scale_engine is not None:
            if self._last_shock_scale is None:
                return shock, shock_dir, None
            scale_ready_ok = (
                self._shock_scale_detector in ("daily_atr_pct", "daily_drawdown")
                or bool(getattr(self._last_shock_scale, "ready", False))
            )
            if not scale_ready_ok:
                return shock, shock_dir, None
            return shock, shock_dir, _atr_pct_from(self._last_shock_scale)

        return shock, shock_dir, _atr_pct_from(self._last_shock)

    def _advance_supertrend_state(
        self,
        *,
        bar: BarLike,
        engine: SupertrendEngine | None,
        use_mtf: bool,
        bars: list[BarLike],
        idx: int,
        last_snapshot,
    ) -> tuple[object | None, int, str | None, bool]:
        if engine is None:
            return last_snapshot, idx, None, False
        if use_mtf and bars:
            while idx < len(bars) and bars[idx].ts <= bar.ts:
                reg_bar = bars[idx]
                last_snapshot = engine.update(
                    high=float(reg_bar.high),
                    low=float(reg_bar.low),
                    close=float(reg_bar.close),
                )
                idx += 1
        else:
            last_snapshot = engine.update(
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )
        direction = last_snapshot.direction if last_snapshot is not None else None
        ready = bool(last_snapshot and last_snapshot.ready)
        direction = str(direction) if direction in ("up", "down") else None
        return last_snapshot, int(idx), direction, bool(ready)

    def _advance_regime2_state(self, *, bar: BarLike) -> tuple[str | None, bool]:
        regime2_dir: str | None = None
        regime2_ready = False
        if self._supertrend2_engine is not None:
            self._last_supertrend2, self._regime2_idx, regime2_dir, regime2_ready = self._advance_supertrend_state(
                bar=bar,
                engine=self._supertrend2_engine,
                use_mtf=bool(self._use_mtf_regime2),
                bars=self._regime2_bars,
                idx=int(self._regime2_idx),
                last_snapshot=self._last_supertrend2,
            )
        elif self._regime2_engine is not None:
            if self._use_mtf_regime2 and self._regime2_bars:
                while self._regime2_idx < len(self._regime2_bars) and self._regime2_bars[self._regime2_idx].ts <= bar.ts:
                    reg_bar = self._regime2_bars[self._regime2_idx]
                    if float(reg_bar.close) > 0:
                        self._last_regime2 = self._regime2_engine.update(float(reg_bar.close))
                    self._regime2_idx += 1
            else:
                self._last_regime2 = self._regime2_engine.update(float(bar.close))
            regime2_dir = self._last_regime2.state if self._last_regime2 is not None else None
            regime2_ready = bool(self._last_regime2 and self._last_regime2.ema_ready)
        self._active_regime2_dir = str(regime2_dir) if regime2_dir in ("up", "down") else None
        self._active_regime2_ready = bool(regime2_ready)
        return self._active_regime2_dir, bool(self._active_regime2_ready)

    def _advance_regime2_bear_hard_state(self, *, bar: BarLike) -> tuple[str | None, bool]:
        hard_dir: str | None = None
        hard_ready = False
        if self._bear_hard_supertrend_engine is not None:
            (
                self._last_bear_hard_supertrend,
                self._regime2_bear_hard_idx,
                hard_dir,
                hard_ready,
            ) = self._advance_supertrend_state(
                bar=bar,
                engine=self._bear_hard_supertrend_engine,
                use_mtf=bool(self._use_mtf_regime2_bear_hard),
                bars=self._regime2_bear_hard_bars,
                idx=int(self._regime2_bear_hard_idx),
                last_snapshot=self._last_bear_hard_supertrend,
            )
        self._active_regime2_bear_hard_dir = str(hard_dir) if hard_dir in ("up", "down") else None
        self._active_regime2_bear_hard_ready = bool(hard_ready)
        return self._active_regime2_bear_hard_dir, bool(self._active_regime2_bear_hard_ready)

    def _advance_clean_regime2_state(self, *, bar: BarLike) -> tuple[str | None, bool]:
        clean_dir: str | None = None
        clean_ready = False
        if self._clean_supertrend2_engine is not None:
            (
                self._last_clean_supertrend2,
                self._clean_regime2_idx,
                clean_dir,
                clean_ready,
            ) = self._advance_supertrend_state(
                bar=bar,
                engine=self._clean_supertrend2_engine,
                use_mtf=bool(self._use_mtf_regime2),
                bars=self._regime2_bars,
                idx=int(self._clean_regime2_idx),
                last_snapshot=self._last_clean_supertrend2,
            )
        self._active_clean_regime2_dir = str(clean_dir) if clean_dir in ("up", "down") else None
        self._active_clean_regime2_ready = bool(clean_ready)
        return self._active_clean_regime2_dir, bool(self._active_clean_regime2_ready)

    def _advance_clean_regime2_bear_hard_state(self, *, bar: BarLike) -> tuple[str | None, bool]:
        clean_hard_dir: str | None = None
        clean_hard_ready = False
        if self._clean_bear_hard_supertrend_engine is not None:
            (
                self._last_clean_bear_hard_supertrend,
                self._clean_regime2_bear_hard_idx,
                clean_hard_dir,
                clean_hard_ready,
            ) = self._advance_supertrend_state(
                bar=bar,
                engine=self._clean_bear_hard_supertrend_engine,
                use_mtf=bool(self._use_mtf_regime2_bear_hard),
                bars=self._regime2_bear_hard_bars,
                idx=int(self._clean_regime2_bear_hard_idx),
                last_snapshot=self._last_clean_bear_hard_supertrend,
            )
        self._active_clean_regime2_bear_hard_dir = (
            str(clean_hard_dir) if clean_hard_dir in ("up", "down") else None
        )
        self._active_clean_regime2_bear_hard_ready = bool(clean_hard_ready)
        return self._active_clean_regime2_bear_hard_dir, bool(self._active_clean_regime2_bear_hard_ready)

    @staticmethod
    def _next_regime2_bear_hard_release_age(
        *,
        hard_dir: str | None,
        hard_ready: bool,
        prev_was_down: bool,
        release_age_bars: int | None,
    ) -> tuple[int | None, bool]:
        is_down = bool(hard_ready and hard_dir == "down")
        if is_down:
            release_age_bars = 0
        elif prev_was_down:
            release_age_bars = 1
        elif release_age_bars is not None:
            release_age_bars = int(release_age_bars) + 1
        return release_age_bars, bool(is_down)

    def _update_regime2_bear_hard_release_age(self, *, hard_dir: str | None, hard_ready: bool) -> int | None:
        (
            self._regime2_bear_hard_release_age_bars,
            self._regime2_bear_hard_prev_was_down,
        ) = self._next_regime2_bear_hard_release_age(
            hard_dir=hard_dir,
            hard_ready=bool(hard_ready),
            prev_was_down=bool(self._regime2_bear_hard_prev_was_down),
            release_age_bars=self._regime2_bear_hard_release_age_bars,
        )
        return self._regime2_bear_hard_release_age_bars

    def update_signal_bar(self, bar: BarLike) -> SpotSignalSnapshot | None:
        """Update the evaluator for a single signal bar close."""
        close = float(bar.close)
        if close <= 0:
            return None

        if self._sig_last_date != self._trade_date(bar.ts):
            self._sig_last_date = self._trade_date(bar.ts)
            self._sig_bars_in_day = 0
        self._sig_bars_in_day += 1

        self._ratsv_update_bar_metrics(high=float(bar.high), low=float(bar.low), close=float(close))

        rv = None
        if self._rv_enabled:
            prev_close = self._prev_sig_close
            self._prev_sig_close = float(close)
            if prev_close is not None and float(prev_close) > 0 and float(close) > 0:
                self._returns.append(math.log(float(close) / float(prev_close)))
            rv = annualized_ewma_vol(
                self._returns,
                lam=float(self._rv_lam),
                bar_size=self._bar_size,
                use_rth=self._use_rth,
            )

        if self._volume_period is not None:
            self._volume_ema = ema_next(self._volume_ema, float(bar.volume), int(self._volume_period))
            self._volume_count += 1

        if self._exit_atr_engine is not None:
            self._last_exit_atr = self._exit_atr_engine.update(
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )

        regime2_dir, regime2_ready = self._advance_regime2_state(bar=bar)
        regime2_bear_hard_dir, regime2_bear_hard_ready = self._advance_regime2_bear_hard_state(bar=bar)
        self._update_regime2_bear_hard_release_age(
            hard_dir=regime2_bear_hard_dir,
            hard_ready=bool(regime2_bear_hard_ready),
        )
        clean_regime2_dir: str | None = None
        clean_regime2_ready = False
        clean_regime2_bear_hard_dir: str | None = None
        clean_regime2_bear_hard_ready = False
        if self._regime2_clean_host_enable:
            clean_regime2_dir, clean_regime2_ready = self._advance_clean_regime2_state(bar=bar)
            clean_regime2_bear_hard_dir, clean_regime2_bear_hard_ready = self._advance_clean_regime2_bear_hard_state(
                bar=bar
            )
            (
                self._clean_regime2_bear_hard_release_age_bars,
                self._clean_regime2_bear_hard_prev_was_down,
            ) = self._next_regime2_bear_hard_release_age(
                hard_dir=clean_regime2_bear_hard_dir,
                hard_ready=bool(clean_regime2_bear_hard_ready),
                prev_was_down=bool(self._clean_regime2_bear_hard_prev_was_down),
                release_age_bars=self._clean_regime2_bear_hard_release_age_bars,
            )

        signal = None
        entry_dir_for_entries: str | None = None
        entry_branch: str | None = None
        ratsv_branch_key: str | None = None
        if self._signal_engine is not None:
            signal = self._signal_engine.update(close)
            entry_dir_for_entries = self._branch_entry_dir(
                branch_key="single",
                signal=signal,
                close=float(close),
                min_signed_slope_pct=None,
                max_signed_slope_pct=None,
            )
            ratsv_branch_key = "single"
        elif self._signal_engine_a is not None and self._signal_engine_b is not None and bool(self._dual_branch_enabled):
            signal_a = self._signal_engine_a.update(close)
            signal_b = self._signal_engine_b.update(close)
            signal, entry_dir_for_entries, entry_branch = self._select_dual_signal(
                close=float(close),
                signal_a=signal_a,
                signal_b=signal_b,
            )
            if signal is signal_a:
                ratsv_branch_key = "a"
            elif signal is signal_b:
                ratsv_branch_key = "b"
        elif self._orb_engine is not None:
            signal = self._orb_engine.update(
                ts=bar.ts,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )
            entry_dir_for_entries = signal.entry_dir if signal is not None else None

        ema_slope_pct = None
        if signal is not None and close:
            ema_fast = getattr(signal, "ema_fast", None)
            prev_ema_fast = getattr(signal, "prev_ema_fast", None)
            if ema_fast is not None and prev_ema_fast is not None:
                try:
                    ema_slope_pct = ((float(ema_fast) - float(prev_ema_fast)) / float(close)) * 100.0
                except (TypeError, ValueError, ZeroDivisionError):
                    ema_slope_pct = None
        ema_slope_vel_pct = None
        if ema_slope_pct is not None and self._prev_ema_slope_pct is not None:
            ema_slope_vel_pct = float(ema_slope_pct) - float(self._prev_ema_slope_pct)
        self._prev_ema_slope_pct = float(ema_slope_pct) if ema_slope_pct is not None else None
        min_slope_abs = float(_get(self._filters, "shock_ramp_min_slope_abs_pct", 0.0) or 0.0)
        if ema_slope_pct is None or abs(float(ema_slope_pct)) < float(min_slope_abs):
            self._ema_slope_up_streak_bars = 0
            self._ema_slope_down_streak_bars = 0
        elif float(ema_slope_pct) > 0:
            self._ema_slope_up_streak_bars += 1
            self._ema_slope_down_streak_bars = 0
        elif float(ema_slope_pct) < 0:
            self._ema_slope_down_streak_bars += 1
            self._ema_slope_up_streak_bars = 0
        else:
            self._ema_slope_up_streak_bars = 0
            self._ema_slope_down_streak_bars = 0

        # Primary regime gating + shock updates.
        if self._supertrend_engine is not None:
            if self._use_mtf_regime and self._regime_bars:
                while self._regime_idx < len(self._regime_bars) and self._regime_bars[self._regime_idx].ts <= bar.ts:
                    reg_bar = self._regime_bars[self._regime_idx]
                    self._last_supertrend = self._supertrend_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                    )
                    if self._supertrend_shock_engine is not None:
                        self._last_supertrend_shock = self._supertrend_shock_engine.update(
                            high=float(reg_bar.high),
                            low=float(reg_bar.low),
                            close=float(reg_bar.close),
                        )
                    if self._supertrend_cooling_engine is not None:
                        self._last_supertrend_cooling = self._supertrend_cooling_engine.update(
                            high=float(reg_bar.high),
                            low=float(reg_bar.low),
                            close=float(reg_bar.close),
                        )
                    if self._shock_engine is not None and self._shock_detector not in ("daily_atr_pct", "daily_drawdown"):
                        self._last_shock = self._shock_engine.update(
                            high=float(reg_bar.high),
                            low=float(reg_bar.low),
                            close=float(reg_bar.close),
                            update_direction=(self._shock_dir_source != "signal"),
                        )
                    self._regime_idx += 1
            else:
                self._last_supertrend = self._supertrend_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )
                if self._supertrend_shock_engine is not None:
                    self._last_supertrend_shock = self._supertrend_shock_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
                if self._supertrend_cooling_engine is not None:
                    self._last_supertrend_cooling = self._supertrend_cooling_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
                if self._shock_engine is not None and self._shock_detector not in ("daily_atr_pct", "daily_drawdown"):
                    self._last_shock = self._shock_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )

            st_for_gate = self._last_supertrend
            if (
                self._shock_engine is not None
                and self._last_shock is not None
                and (self._supertrend_shock_engine is not None or self._supertrend_cooling_engine is not None)
            ):
                shock_ready = bool(
                    self._shock_detector in ("daily_atr_pct", "daily_drawdown")
                    or bool(getattr(self._last_shock, "ready", False))
                )
                shock_now = bool(getattr(self._last_shock, "shock", False)) if shock_ready else False

                cooling_now = False
                cooling_atr = (
                    float(_get(self._filters, "shock_daily_cooling_atr_pct", 0.0) or 0.0)
                    if (_get(self._filters, "shock_daily_cooling_atr_pct", None) is not None)
                    else None
                )
                atr_pct = getattr(self._last_shock, "atr_pct", None)
                if (
                    (not bool(shock_now))
                    and cooling_atr is not None
                    and atr_pct is not None
                    and self._shock_detector == "daily_atr_pct"
                    and shock_ready
                    and float(atr_pct) >= float(cooling_atr)
                ):
                    cooling_now = True

                if shock_now and self._last_supertrend_shock is not None:
                    st_for_gate = self._last_supertrend_shock
                elif cooling_now and self._last_supertrend_cooling is not None:
                    st_for_gate = self._last_supertrend_cooling

            regime_dir = st_for_gate.direction if st_for_gate is not None else None
            regime_ready = bool(st_for_gate and st_for_gate.ready)

            if (
                bool(_get(self._filters, "shock_regime_override_dir", False))
                and self._shock_engine is not None
                and self._last_shock is not None
            ):
                shock_ready = bool(
                    self._shock_detector in ("daily_atr_pct", "daily_drawdown")
                    or bool(getattr(self._last_shock, "ready", False))
                )
                if shock_ready and bool(getattr(self._last_shock, "shock", False)):
                    if bool(getattr(self._last_shock, "direction_ready", False)) and getattr(
                        self._last_shock, "direction", None
                    ) in ("up", "down"):
                        regime_dir = str(getattr(self._last_shock, "direction"))
                        regime_ready = True

            signal = apply_regime_gate(signal, regime_dir=regime_dir, regime_ready=regime_ready)

        elif self._use_mtf_regime and self._regime_engine is not None and self._regime_bars:
            while self._regime_idx < len(self._regime_bars) and self._regime_bars[self._regime_idx].ts <= bar.ts:
                reg_bar = self._regime_bars[self._regime_idx]
                if float(reg_bar.close) > 0:
                    self._last_regime = self._regime_engine.update(float(reg_bar.close))
                if self._shock_engine is not None and self._shock_detector not in ("daily_atr_pct", "daily_drawdown"):
                    self._last_shock = self._shock_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                        update_direction=(self._shock_dir_source != "signal"),
                    )
                self._regime_idx += 1
            signal = apply_regime_gate(
                signal,
                regime_dir=self._last_regime.state if self._last_regime is not None else None,
                regime_ready=bool(self._last_regime and self._last_regime.ema_ready),
            )
        elif (
            self._shock_engine is not None
            and self._shock_detector not in ("daily_atr_pct", "daily_drawdown")
            and (not self._use_mtf_regime)
        ):
            self._last_shock = self._shock_engine.update(
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )

        if (
            self._shock_engine is not None
            and self._shock_detector == "atr_ratio"
            and self._use_mtf_regime
            and self._shock_dir_source == "signal"
        ):
            self._last_shock = self._shock_engine.update_direction(close=float(bar.close))

        if (
            self._shock_engine is not None
            and self._shock_detector in ("daily_atr_pct", "daily_drawdown")
            and self._shock_dir_source == "signal"
        ):
            # Daily shock engines are already advanced by exec bars (for intraday high/low/close),
            # but the *direction* can optionally be driven by signal-bar closes. Avoid a redundant
            # full daily update here.
            if hasattr(self._shock_engine, "update_direction"):
                self._last_shock = self._shock_engine.update_direction(close=float(bar.close))
            else:
                self._last_shock = self._shock_engine.update(
                    day=self._trade_date(bar.ts),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    update_direction=True,
                )

        if (
            self._shock_scale_engine is not None
            and self._shock_scale_detector not in ("daily_atr_pct", "daily_drawdown")
        ):
            self._last_shock_scale = self._shock_scale_engine.update(
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=False,
            )

        shock, shock_dir, shock_atr_pct = self._shock_view()
        regime4_state, regime4_transition_hot = self._classify_regime4_state(
            shock_atr_pct=shock_atr_pct,
            fast_dir=regime2_dir,
            fast_ready=bool(regime2_ready),
            hard_dir=regime2_bear_hard_dir,
            hard_ready=bool(regime2_bear_hard_ready),
            hard_release_age_bars=self._regime2_bear_hard_release_age_bars,
        )
        regime4_owner = "primary"
        active_regime2_dir = str(regime2_dir) if regime2_dir in ("up", "down") else None
        active_regime2_ready = bool(regime2_ready)
        active_regime2_bear_hard_dir = (
            str(regime2_bear_hard_dir) if regime2_bear_hard_dir in ("up", "down") else None
        )
        active_regime2_bear_hard_ready = bool(regime2_bear_hard_ready)
        active_regime2_bear_hard_release_age_bars = self._regime2_bear_hard_release_age_bars
        if self._regime2_clean_host_enable:
            clean_regime4_state, clean_regime4_transition_hot = self._classify_regime4_state(
                shock_atr_pct=shock_atr_pct,
                fast_dir=clean_regime2_dir,
                fast_ready=bool(clean_regime2_ready),
                hard_dir=clean_regime2_bear_hard_dir,
                hard_ready=bool(clean_regime2_bear_hard_ready),
                hard_release_age_bars=self._clean_regime2_bear_hard_release_age_bars,
            )
            clean_takeover = False
            if self._regime2_clean_host_takeover_state == "trend_up_clean":
                clean_takeover = bool(
                    regime4_state == "trend_up_clean" and clean_regime4_state == "trend_up_clean"
                )
            elif self._regime2_clean_host_takeover_state == "crash_down":
                clean_takeover = bool(clean_regime4_state == "crash_down")
            elif self._regime2_clean_host_takeover_state == "transition_up_hot":
                clean_takeover = bool(clean_regime4_state == "transition_up_hot")
            elif self._regime2_clean_host_takeover_state == "crash_or_transition_up_hot":
                clean_takeover = bool(clean_regime4_state in ("crash_down", "transition_up_hot"))
            if clean_takeover:
                regime4_owner = "clean_host"
                regime4_state = clean_regime4_state
                regime4_transition_hot = bool(clean_regime4_transition_hot)
                active_regime2_dir = str(clean_regime2_dir) if clean_regime2_dir in ("up", "down") else None
                active_regime2_ready = bool(clean_regime2_ready)
                active_regime2_bear_hard_dir = (
                    str(clean_regime2_bear_hard_dir) if clean_regime2_bear_hard_dir in ("up", "down") else None
                )
                active_regime2_bear_hard_ready = bool(clean_regime2_bear_hard_ready)
                active_regime2_bear_hard_release_age_bars = self._clean_regime2_bear_hard_release_age_bars
        self._active_regime2_dir = active_regime2_dir
        self._active_regime2_ready = bool(active_regime2_ready)
        self._active_regime2_bear_hard_dir = active_regime2_bear_hard_dir
        self._active_regime2_bear_hard_ready = bool(active_regime2_bear_hard_ready)
        self._active_regime2_bear_hard_release_age_bars = active_regime2_bear_hard_release_age_bars
        self._regime4_transition_hot = bool(regime4_transition_hot)
        self._regime4_owner = str(regime4_owner)

        # Secondary regime2 gating.
        if self._supertrend2_engine is not None:
            if spot_regime_apply_matches_direction(
                apply_to_raw=_get(self._strategy, "regime2_apply_to", "both"),
                entry_dir=getattr(signal, "entry_dir", None),
            ):
                signal = apply_regime_gate(
                    signal,
                    regime_dir=active_regime2_dir,
                    regime_ready=bool(active_regime2_ready),
                )
        elif self._regime2_engine is not None:
            if spot_regime_apply_matches_direction(
                apply_to_raw=_get(self._strategy, "regime2_apply_to", "both"),
                entry_dir=getattr(signal, "entry_dir", None),
            ):
                signal = apply_regime_gate(
                    signal,
                    regime_dir=active_regime2_dir,
                    regime_ready=bool(active_regime2_ready),
                )

        if signal is not None and self._regime2_bear_entry_mode == "supertrend":
            signal, bear_entry_dir = self._apply_regime2_bear_primary(
                branch_key=str(ratsv_branch_key or "single"),
                signal=signal,
                bar=bar,
                close=float(close),
            )
            if bear_entry_dir in ("up", "down"):
                if self._dual_branch_enabled and entry_branch not in ("a", "b"):
                    entry_dir_for_entries = None
                    entry_branch = None
                else:
                    entry_dir_for_entries = str(bear_entry_dir)
                    if entry_branch not in ("a", "b") or getattr(signal, "entry_dir", None) != entry_dir_for_entries:
                        entry_branch = None

        if signal is None:
            return None

        # Branch-selected entry direction is further constrained by regime gates above.
        gated_dir = signal.entry_dir if getattr(signal, "entry_dir", None) in ("up", "down") else None
        if self._dual_branch_enabled:
            if entry_dir_for_entries not in ("up", "down"):
                entry_dir_for_entries = None
                entry_branch = None
            elif gated_dir != entry_dir_for_entries:
                entry_dir_for_entries = None
                entry_branch = None
        else:
            if entry_dir_for_entries not in ("up", "down"):
                entry_dir_for_entries = None
            elif gated_dir != entry_dir_for_entries:
                entry_dir_for_entries = None
            entry_branch = None

        shock_atr_vel_pct = None
        shock_atr_accel_pct = None
        if shock_atr_pct is not None:
            cur_atr_pct = float(shock_atr_pct)
            if self._prev_shock_atr_pct is not None:
                shock_atr_vel_pct = float(cur_atr_pct) - float(self._prev_shock_atr_pct)
                if self._prev_shock_atr_vel_pct is not None:
                    shock_atr_accel_pct = float(shock_atr_vel_pct) - float(self._prev_shock_atr_vel_pct)
            self._prev_shock_atr_pct = float(cur_atr_pct)
            self._prev_shock_atr_vel_pct = float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None
        else:
            self._prev_shock_atr_pct = None
            self._prev_shock_atr_vel_pct = None

        def _float_or_none(raw: object) -> float | None:
            try:
                return float(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None

        shock_peak_close = _float_or_none(getattr(self._last_shock, "peak_close", None))
        shock_dir_ret_sum_pct = _float_or_none(getattr(self._last_shock, "direction_ret_sum_pct", None))

        dd_snap = self._last_shock if str(self._shock_detector) == "daily_drawdown" else None
        if dd_snap is None:
            dd_snap = self._last_aux_drawdown

        shock_drawdown_pct = _float_or_none(getattr(dd_snap, "drawdown_pct", None))
        shock_scale_drawdown_pct = _float_or_none(getattr(self._last_shock_scale, "drawdown_pct", None))
        if shock_peak_close is None:
            shock_peak_close = _float_or_none(getattr(dd_snap, "peak_close", None))
        if shock_dir_ret_sum_pct is None:
            shock_dir_ret_sum_pct = _float_or_none(getattr(dd_snap, "direction_ret_sum_pct", None))
        shock_on_drawdown_pct = None
        shock_off_drawdown_pct = None
        shock_drawdown_dist_on_pct = None
        shock_drawdown_dist_on_vel_pp = None
        shock_drawdown_dist_on_accel_pp = None
        shock_drawdown_dist_off_pct = None
        if dd_snap is not None:
            shock_on_drawdown_pct = _float_or_none(_get(self._filters, "shock_on_drawdown_pct", -20.0))
            shock_off_drawdown_pct = _float_or_none(_get(self._filters, "shock_off_drawdown_pct", -10.0))
            if shock_on_drawdown_pct is not None and shock_off_drawdown_pct is not None and shock_off_drawdown_pct < shock_on_drawdown_pct:
                shock_off_drawdown_pct = float(shock_on_drawdown_pct)
            if (
                shock_drawdown_pct is not None
                and shock_on_drawdown_pct is not None
                and shock_off_drawdown_pct is not None
            ):
                # +dist_on means drawdown has crossed ON threshold by that many pp.
                # +dist_off means drawdown has recovered past OFF threshold by that many pp.
                shock_drawdown_dist_on_pct = float(shock_on_drawdown_pct) - float(shock_drawdown_pct)
                shock_drawdown_dist_off_pct = float(shock_drawdown_pct) - float(shock_off_drawdown_pct)
                if self._prev_shock_drawdown_dist_on_pct is not None:
                    shock_drawdown_dist_on_vel_pp = (
                        float(shock_drawdown_dist_on_pct) - float(self._prev_shock_drawdown_dist_on_pct)
                    )
                    if self._prev_shock_drawdown_dist_on_vel_pp is not None:
                        shock_drawdown_dist_on_accel_pp = (
                            float(shock_drawdown_dist_on_vel_pp) - float(self._prev_shock_drawdown_dist_on_vel_pp)
                        )
                self._prev_shock_drawdown_dist_on_pct = float(shock_drawdown_dist_on_pct)
                self._prev_shock_drawdown_dist_on_vel_pp = (
                    float(shock_drawdown_dist_on_vel_pp) if shock_drawdown_dist_on_vel_pp is not None else None
                )
            else:
                self._prev_shock_drawdown_dist_on_pct = None
                self._prev_shock_drawdown_dist_on_vel_pp = None
        else:
            self._prev_shock_drawdown_dist_on_pct = None
            self._prev_shock_drawdown_dist_on_vel_pp = None

        if self._regime2_crash_blocks_long(regime4_state=regime4_state, entry_dir=entry_dir_for_entries):
            entry_dir_for_entries = None
            entry_branch = None

        if self._regime2_crash_prearm_blocks_long(
            regime4_state=regime4_state,
            entry_dir=entry_dir_for_entries,
            entry_branch=entry_branch,
            shock_dir=shock_dir,
            shock_atr_pct=shock_atr_pct,
            shock_dir_ret_sum_pct=shock_dir_ret_sum_pct,
        ):
            entry_dir_for_entries = None
            entry_branch = None

        if self._regime2_blocks_branch_b_long(
            regime4_state=regime4_state,
            entry_dir=entry_dir_for_entries,
            entry_branch=entry_branch,
            shock_dir=shock_dir,
            shock_atr_pct=shock_atr_pct,
            shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
            bar_ts=bar.ts,
        ):
            entry_dir_for_entries = None
            entry_branch = None

        if self._regime2_upcorridor_blocks_branch_a_long(
            regime4_state=regime4_state,
            entry_dir=entry_dir_for_entries,
            entry_branch=entry_branch,
            shock_atr_pct=shock_atr_pct,
        ):
            entry_dir_for_entries = None
            entry_branch = None

        if self._continuation_confidence_blocks_long(
            regime4_state=regime4_state,
            entry_dir=entry_dir_for_entries,
            entry_branch=entry_branch,
            shock_dir=shock_dir,
        ):
            entry_dir_for_entries = None
            entry_branch = None

        if bool(shock) and shock_dir == "down":
            self._shock_dir_down_streak_bars += 1
        else:
            self._shock_dir_down_streak_bars = 0
        if bool(shock) and shock_dir == "up":
            self._shock_dir_up_streak_bars += 1
        else:
            self._shock_dir_up_streak_bars = 0

        prearm_streak = 0
        if (
            dd_snap is not None
            and (not bool(shock))
            and shock_drawdown_dist_on_pct is not None
        ):
            dd_band = float(_get(self._filters, "shock_prearm_dist_on_max_pp", 0.0) or 0.0)
            dd_band = float(max(0.0, float(dd_band)))
            min_dd = float(_get(self._filters, "shock_prearm_min_drawdown_pct", 0.0) or 0.0)
            # Convenience: allow users to specify +6 to mean ">= 6% drawdown".
            if min_dd > 0:
                min_dd = -float(min_dd)
            ddv_min = float(_get(self._filters, "shock_prearm_min_dist_on_vel_pp", 0.0) or 0.0)
            ddv_min = float(max(0.0, float(ddv_min)))
            dda_min = float(_get(self._filters, "shock_prearm_min_dist_on_accel_pp", 0.0) or 0.0)
            dda_min = float(max(0.0, float(dda_min)))
            latch_min_streak = int(_get(self._filters, "shock_prearm_min_streak_bars", 0) or 0)
            latch_enabled = latch_min_streak > 0

            dist_on = float(shock_drawdown_dist_on_pct)
            dist_on_vel = float(shock_drawdown_dist_on_vel_pp) if shock_drawdown_dist_on_vel_pp is not None else None
            dist_on_accel = (
                float(shock_drawdown_dist_on_accel_pp) if shock_drawdown_dist_on_accel_pp is not None else None
            )
            dist_off = float(shock_drawdown_dist_off_pct) if shock_drawdown_dist_off_pct is not None else None

            arm_ok = True
            if dd_band <= 0:
                arm_ok = False
            if bool(arm_ok) and min_dd < 0:
                # Depth gate: avoid arming prearm in mild pullbacks near ATH noise.
                # Note: drawdown is negative (e.g., -7.2). We require dd <= min_dd (e.g., -6.0).
                if shock_drawdown_pct is None:
                    arm_ok = False
                elif float(shock_drawdown_pct) > float(min_dd):
                    arm_ok = False
            if bool(arm_ok) and not (-float(dd_band) <= float(dist_on) < 0.0):
                arm_ok = False
            if bool(arm_ok) and dist_on_vel is None:
                arm_ok = False
            if bool(arm_ok) and float(dist_on_vel) < float(ddv_min):
                arm_ok = False
            if bool(arm_ok) and dda_min > 0:
                if dist_on_accel is None:
                    arm_ok = False
                elif float(dist_on_accel) < float(dda_min):
                    arm_ok = False

            if not bool(latch_enabled):
                # Legacy telemetry: 1-bar arm indicator (no persistence).
                self._shock_prearm_down_streak_bars = 1 if bool(arm_ok) else 0
            else:
                latched = bool(self._shock_prearm_down_streak_bars > 0)
                release = False
                if dist_off is not None and float(dist_off) >= 0.0:
                    release = True
                elif dd_band > 0 and float(dist_on) < -float(dd_band):
                    release = True
                elif dist_on_vel is not None and float(dist_on) < 0.0 and float(dist_on_vel) < 0.0:
                    release = True

                if latched:
                    if release:
                        self._shock_prearm_down_streak_bars = 0
                    else:
                        self._shock_prearm_down_streak_bars += 1
                else:
                    self._shock_prearm_down_streak_bars = 1 if bool(arm_ok) else 0

            prearm_streak = int(self._shock_prearm_down_streak_bars)
        else:
            self._shock_prearm_down_streak_bars = 0

        atr = (
            float(self._last_exit_atr.atr)
            if self._last_exit_atr is not None and bool(self._last_exit_atr.ready) and self._last_exit_atr.atr is not None
            else None
        )
        ratsv_metrics = self._ratsv_last_candidate_metrics.get(str(ratsv_branch_key)) if ratsv_branch_key else None

        shock_ramp = None
        if bool(_get(self._filters, "shock_ramp_enable", False)):
            apply_to = str(_get(self._filters, "shock_ramp_apply_to", "down") or "down").strip().lower()
            if apply_to not in ("down", "up", "both"):
                apply_to = "down"

            def _clamp01(x: float) -> float:
                return float(max(0.0, min(1.0, float(x))))

            def _node(dir_: str) -> dict[str, object]:
                direction = str(dir_).strip().lower()
                if direction not in ("up", "down"):
                    direction = "down"
                allow = bool(apply_to == "both" or apply_to == direction)
                if not bool(allow):
                    return {
                        "phase": "off",
                        "intensity": 0.0,
                        "risk_mult": 1.0,
                        "cap_floor_frac": 0.0,
                        "align_ok": False,
                        "streak_bars": 0,
                        "min_streak_bars": 0,
                        "ema_slope_pct": float(ema_slope_pct) if ema_slope_pct is not None else None,
                        "ema_slope_vel_pct": float(ema_slope_vel_pct) if ema_slope_vel_pct is not None else None,
                        "dd_prog": None,
                        "ddv_strength": None,
                        "reason": "disabled",
                    }
                entry_dir = str(getattr(signal, "entry_dir", "") or "")
                regime_dir = str(getattr(signal, "regime_dir", "") or "")
                align_ok = entry_dir == direction and regime_dir == direction

                streak = int(self._ema_slope_up_streak_bars) if direction == "up" else int(self._ema_slope_down_streak_bars)
                min_streak = int(_get(self._filters, "shock_ramp_min_slope_streak_bars", 0) or 0)
                min_streak = max(0, int(min_streak))

                slope_signed = 0.0
                if ema_slope_pct is not None:
                    slope_signed = float(ema_slope_pct) if direction == "up" else -float(ema_slope_pct)
                slope_vel_signed = 0.0
                if ema_slope_vel_pct is not None:
                    slope_vel_signed = float(ema_slope_vel_pct) if direction == "up" else -float(ema_slope_vel_pct)

                try:
                    slope_ref = float(_get(self._strategy, "spot_graph_overlay_slope_ref_pct", 0.08) or 0.08)
                except (TypeError, ValueError):
                    slope_ref = 0.08
                slope_ref = float(max(1e-9, float(slope_ref)))

                slope_strength = _clamp01(float(slope_signed) / float(slope_ref))
                intensity = float(slope_strength) if direction == "up" else 0.0
                reason = "slope" if direction == "up" else "dd"
                dd_prog = 0.0
                ddv_strength = 0.0
                if direction == "down" and shock_drawdown_dist_on_pct is not None:
                    dd_band = float(_get(self._filters, "shock_prearm_dist_on_max_pp", 0.0) or 0.0)
                    dd_band = float(max(0.0, float(dd_band)))
                    dist_on = float(shock_drawdown_dist_on_pct)
                    if dist_on >= 0:
                        dd_prog = 1.0
                    elif dd_band > 0 and dist_on >= -float(dd_band):
                        dd_prog = 1.0 - (abs(float(dist_on)) / float(dd_band))
                    else:
                        dd_prog = 0.0
                    dd_prog = _clamp01(float(dd_prog))

                    if bool(shock) and shock_dir == "down":
                        ddv_strength = 1.0
                    elif shock_drawdown_dist_on_vel_pp is not None:
                        ddv = float(shock_drawdown_dist_on_vel_pp)
                        ddv_min = float(_get(self._filters, "shock_prearm_min_dist_on_vel_pp", 0.0) or 0.0)
                        ddv_min = float(max(0.0, float(ddv_min)))
                        ddv_ref = float(max(0.5, 2.0 * float(ddv_min))) if ddv_min > 0 else 0.5
                        ddv_strength = _clamp01(max(0.0, float(ddv)) / float(ddv_ref))
                    dd_comp = _clamp01(float(dd_prog) * float(ddv_strength))
                    intensity = float(dd_comp)

                if slope_vel_signed < 0:
                    intensity *= 0.60
                    reason = f"{reason}+vel<0"
                if slope_signed <= 0:
                    intensity = 0.0
                    reason = "slope_opposite"

                if min_streak > 0:
                    if streak <= 0:
                        intensity = 0.0
                        reason = "streak=0"
                    else:
                        intensity *= _clamp01(float(streak) / float(min_streak))
                        if streak < min_streak:
                            reason = "streak_ramp"

                if not bool(align_ok):
                    intensity = 0.0
                    reason = "align_fail"

                try:
                    max_mult = float(_get(self._filters, "shock_ramp_max_risk_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    max_mult = 1.0
                max_mult = float(max(1.0, float(max_mult)))
                risk_mult = 1.0 + (float(max_mult) - 1.0) * float(_clamp01(float(intensity)))

                try:
                    max_floor = float(_get(self._filters, "shock_ramp_max_cap_floor_frac", 0.0) or 0.0)
                except (TypeError, ValueError):
                    max_floor = 0.0
                max_floor = float(max(0.0, min(1.0, float(max_floor))))
                floor_frac = float(max_floor) * float(_clamp01(float(intensity)))

                phase = "off"
                if float(intensity) > 1e-9:
                    if direction == "down" and bool(shock) and shock_dir == "down":
                        phase = "active"
                    elif direction == "down" and dd_prog > 0 and dd_prog < 1:
                        phase = "approach"
                    else:
                        phase = "trend"

                return {
                    "phase": str(phase),
                    "intensity": float(_clamp01(float(intensity))),
                    "risk_mult": float(risk_mult),
                    "cap_floor_frac": float(floor_frac),
                    "align_ok": bool(align_ok),
                    "streak_bars": int(streak),
                    "min_streak_bars": int(min_streak),
                    "ema_slope_pct": float(ema_slope_pct) if ema_slope_pct is not None else None,
                    "ema_slope_vel_pct": float(ema_slope_vel_pct) if ema_slope_vel_pct is not None else None,
                    "dd_prog": float(dd_prog) if direction == "down" else None,
                    "ddv_strength": float(ddv_strength) if direction == "down" else None,
                    "reason": str(reason),
                }

            shock_ramp = {"up": _node("up"), "down": _node("down")}

        snap = SpotSignalSnapshot(
            bar_ts=bar.ts,
            close=float(close),
            signal=signal,
            bars_in_day=int(self._sig_bars_in_day),
            rv=float(rv) if rv is not None else None,
            volume=float(bar.volume) if bar.volume is not None else None,
            volume_ema=float(self._volume_ema) if self._volume_ema is not None else None,
            volume_ema_ready=bool(self._volume_count >= int(self._volume_period)) if self._volume_period else True,
            shock=shock,
            shock_dir=shock_dir,
            shock_detector=str(self._shock_detector) if self._shock_engine is not None else None,
            shock_direction_source_effective=str(self._shock_dir_source) if self._shock_engine is not None else None,
            shock_scale_detector=str(self._shock_scale_detector) if self._shock_scale_engine is not None else None,
            shock_dir_ret_sum_pct=shock_dir_ret_sum_pct,
            shock_atr_pct=shock_atr_pct,
            shock_drawdown_pct=shock_drawdown_pct,
            shock_drawdown_on_pct=shock_on_drawdown_pct,
            shock_drawdown_off_pct=shock_off_drawdown_pct,
            shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct,
            shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
            shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp,
            shock_prearm_down_streak_bars=int(prearm_streak),
            shock_drawdown_dist_off_pct=shock_drawdown_dist_off_pct,
            shock_scale_drawdown_pct=shock_scale_drawdown_pct,
            shock_peak_close=shock_peak_close,
            shock_dir_down_streak_bars=int(self._shock_dir_down_streak_bars),
            shock_dir_up_streak_bars=int(self._shock_dir_up_streak_bars),
            risk=self._last_risk,
            atr=atr,
            regime2_dir=str(self._active_regime2_dir) if self._active_regime2_dir in ("up", "down") else None,
            regime2_ready=bool(self._active_regime2_ready),
            regime2_bear_hard_dir=(
                str(self._active_regime2_bear_hard_dir)
                if self._active_regime2_bear_hard_dir in ("up", "down")
                else None
            ),
            regime2_bear_hard_ready=bool(self._active_regime2_bear_hard_ready),
            regime2_bear_hard_release_age_bars=(
                int(self._active_regime2_bear_hard_release_age_bars)
                if self._active_regime2_bear_hard_release_age_bars is not None
                else None
            ),
            regime4_state=str(regime4_state) if regime4_state is not None else None,
            regime4_transition_hot=bool(self._regime4_transition_hot),
            regime4_owner=str(self._regime4_owner) if self._regime4_owner is not None else None,
            or_high=self._orb_engine.or_high if self._orb_engine is not None else None,
            or_low=self._orb_engine.or_low if self._orb_engine is not None else None,
            or_ready=bool(self._orb_engine and self._orb_engine.or_ready),
            entry_dir=str(entry_dir_for_entries) if entry_dir_for_entries in ("up", "down") else None,
            entry_branch=str(entry_branch) if entry_branch in ("a", "b") else None,
            ratsv_side_rank=(
                float(ratsv_metrics.get("side_rank"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("side_rank") is not None
                else None
            ),
            ratsv_tr_ratio=(
                float(ratsv_metrics.get("tr_ratio"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("tr_ratio") is not None
                else None
            ),
            ratsv_fast_slope_pct=(
                float(ratsv_metrics.get("slope_now"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_now") is not None
                else None
            ),
            ratsv_fast_slope_med_pct=(
                float(ratsv_metrics.get("slope_med"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_med") is not None
                else None
            ),
            ratsv_fast_slope_vel_pct=(
                float(ratsv_metrics.get("slope_vel"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_vel") is not None
                else None
            ),
            ratsv_slow_slope_med_pct=(
                float(ratsv_metrics.get("slope_med_slow"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_med_slow") is not None
                else None
            ),
            ratsv_slow_slope_vel_pct=(
                float(ratsv_metrics.get("slope_vel_slow"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_vel_slow") is not None
                else None
            ),
            ratsv_slope_vel_consistency=(
                float(ratsv_metrics.get("slope_vel_consistency"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("slope_vel_consistency") is not None
                else None
            ),
            ratsv_cross_age_bars=(
                int(ratsv_metrics.get("cross_age"))
                if isinstance(ratsv_metrics, dict) and ratsv_metrics.get("cross_age") is not None
                else None
            ),
            shock_atr_vel_pct=float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
            shock_atr_accel_pct=float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
            shock_ramp=shock_ramp,
        )
        self._last_signal = signal
        self._last_snapshot = snap
        return snap
# endregion
