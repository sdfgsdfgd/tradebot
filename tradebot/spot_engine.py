"""Shared spot signal evaluation pipeline (UI + backtests).

This module owns the concrete evaluator while focused spot modules own setup,
policy, regime advancement, and sequential bar evaluation. It intentionally
does not simulate trades, fills, costs, or portfolio state.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import time

from .chart_data.series import BarSeries
from .climate_router import DailyRegimeRouterEngine, regime_router_config
from .engine import normalize_spot_entry_signal, normalize_spot_regime_mode, parse_time_hhmm
from .engines.risk import RiskOverlaySnapshot, build_tr_pct_risk_overlay_engine
from .engines.shock import build_shock_engine, normalize_shock_detector, normalize_shock_direction_source
from .engines.signals import EmaDecisionEngine, EmaDecisionSnapshot, OrbDecisionEngine, SupertrendEngine
from .signals import ema_periods
from .spot.evaluator_common import BarLike, SpotSignalSnapshot, _bars_input_list
from .spot.policy_contract import source_value as _get
from .spot.evaluator_policy import SpotSignalPolicyMixin
from .spot.evaluator_regime import SpotSignalRegimeMixin
from .spot.evaluator_runtime import SpotSignalRuntimeMixin
from .spot.evaluator_setup import SpotSignalSetupMixin
from .time_utils import NaiveTsModeInput, normalize_naive_ts_mode


# region Evaluator
class SpotSignalEvaluator(
    SpotSignalRuntimeMixin,
    SpotSignalRegimeMixin,
    SpotSignalPolicyMixin,
    SpotSignalSetupMixin,
):
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
        regime_router_seed_days: list[BarLike] | BarSeries[BarLike] | None = None,
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
        self._regime_router_cfg = regime_router_config(strategy)
        self._regime_router = DailyRegimeRouterEngine(config=self._regime_router_cfg)
        if bool(self._regime_router_cfg.enabled) and regime_router_seed_days:
            self._regime_router.seed_completed_days(_bars_input_list(regime_router_seed_days))

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
        self._ratsv_tr_fast_hist: deque[float] = deque(maxlen=self._ratsv_tr_fast)
        self._ratsv_tr_slow_hist: deque[float] = deque(maxlen=self._ratsv_tr_slow)
        self._ratsv_prev_tr_close: float | None = None
        self._ratsv_branch_cross_age: dict[str, int | None] = {"single": None, "a": None, "b": None}
        self._ratsv_branch_slope_fast_hist: dict[str, deque[float]] = {
            key: deque(maxlen=self._ratsv_slope_window)
            for key in ("single", "a", "b")
        }
        self._ratsv_branch_slope_slow_hist: dict[str, deque[float]] = {
            key: deque(maxlen=self._ratsv_slope_slow_window)
            for key in ("single", "a", "b")
        }
        slope_hist_maxlen = max(16, self._ratsv_slope_slow_window * 4)
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

        regime2_preset = self._configure_secondary_regime(
            strategy=strategy,
            regime2_bars=regime2_bars,
            regime2_bear_hard_bars=regime2_bear_hard_bars,
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
# endregion
