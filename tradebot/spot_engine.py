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
from dataclasses import dataclass
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
    shock_atr_pct: float | None
    risk: RiskOverlaySnapshot | None
    atr: float | None
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

        self._last_signal: EmaDecisionSnapshot | None = None
        self._last_snapshot: SpotSignalSnapshot | None = None
        self._prev_shock_atr_pct: float | None = None
        self._prev_shock_atr_vel_pct: float | None = None

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

    def _branch_entry_dir(
        self,
        *,
        branch_key: str,
        signal: EmaDecisionSnapshot | None,
        close: float,
        min_signed_slope_pct: float | None,
        max_signed_slope_pct: float | None,
    ) -> str | None:
        if signal is None or not bool(signal.ema_ready):
            self._ratsv_branch_metrics(branch_key=branch_key, signal=signal, close=float(close), entry_dir=None)
            return None
        entry_dir = signal.entry_dir
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

        # Secondary regime2 gating.
        if self._supertrend2_engine is not None:
            if self._use_mtf_regime2 and self._regime2_bars:
                while self._regime2_idx < len(self._regime2_bars) and self._regime2_bars[self._regime2_idx].ts <= bar.ts:
                    reg_bar = self._regime2_bars[self._regime2_idx]
                    self._last_supertrend2 = self._supertrend2_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                    )
                    self._regime2_idx += 1
            else:
                self._last_supertrend2 = self._supertrend2_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )

            if spot_regime_apply_matches_direction(
                apply_to_raw=_get(self._strategy, "regime2_apply_to", "both"),
                entry_dir=getattr(signal, "entry_dir", None),
            ):
                signal = apply_regime_gate(
                    signal,
                    regime_dir=self._last_supertrend2.direction if self._last_supertrend2 is not None else None,
                    regime_ready=bool(self._last_supertrend2 and self._last_supertrend2.ready),
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

            if spot_regime_apply_matches_direction(
                apply_to_raw=_get(self._strategy, "regime2_apply_to", "both"),
                entry_dir=getattr(signal, "entry_dir", None),
            ):
                signal = apply_regime_gate(
                    signal,
                    regime_dir=self._last_regime2.state if self._last_regime2 is not None else None,
                    regime_ready=bool(self._last_regime2 and self._last_regime2.ema_ready),
                )

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

        shock, shock_dir, shock_atr_pct = self._shock_view()
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
        atr = (
            float(self._last_exit_atr.atr)
            if self._last_exit_atr is not None and bool(self._last_exit_atr.ready) and self._last_exit_atr.atr is not None
            else None
        )
        ratsv_metrics = self._ratsv_last_candidate_metrics.get(str(ratsv_branch_key)) if ratsv_branch_key else None

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
            shock_atr_pct=shock_atr_pct,
            risk=self._last_risk,
            atr=atr,
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
        )
        self._last_signal = signal
        self._last_snapshot = snap
        return snap
# endregion
