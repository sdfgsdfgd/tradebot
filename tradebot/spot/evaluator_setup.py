"""Secondary-regime construction for the spot-signal evaluator."""
from __future__ import annotations

from ..chart_data.series import BarSeries
from ..engine import resolve_spot_regime2_spec
from ..engines.signals import EmaDecisionEngine, SupertrendEngine
from .evaluator_common import BarLike, _bars_input_list, _get


class SpotSignalSetupMixin:
    def _configure_secondary_regime(
        self,
        *,
        strategy: object,
        regime2_bars: list[BarLike] | BarSeries[BarLike] | None,
        regime2_bear_hard_bars: list[BarLike] | BarSeries[BarLike] | None,
    ) -> str | None:
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
        raw_continuation_conf_a_age_max = _get(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_release_age_max_bars",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars = (
                int(raw_continuation_conf_a_age_max) if raw_continuation_conf_a_age_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars = None
        if self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars is not None:
            self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars = max(
                0,
                int(self._regime2_continuation_confidence_branch_a_transition_release_age_max_bars),
            )
        raw_continuation_conf_a_atr_min = _get(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min = (
                float(raw_continuation_conf_a_atr_min) if raw_continuation_conf_a_atr_min is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min = None
        if (
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min < 0
        ):
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min = None
        raw_continuation_conf_a_atr_max = _get(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max = (
                float(raw_continuation_conf_a_atr_max) if raw_continuation_conf_a_atr_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max = None
        if (
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max < 0
        ):
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max = None
        if (
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max is not None
            and self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max
            < self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min
        ):
            self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max = (
                self._regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min
            )
        raw_continuation_conf_a_ddv_max = _get(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_ddv_max_pp",
            None,
        )
        try:
            self._regime2_continuation_confidence_branch_a_transition_ddv_max_pp = (
                float(raw_continuation_conf_a_ddv_max) if raw_continuation_conf_a_ddv_max is not None else None
            )
        except (TypeError, ValueError):
            self._regime2_continuation_confidence_branch_a_transition_ddv_max_pp = None
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
        return regime2_preset
