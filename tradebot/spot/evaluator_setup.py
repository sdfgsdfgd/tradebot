"""Construction and legacy-key decoding for the canonical spot regime model."""

from __future__ import annotations

from ..chart_data.series import BarSeries
from ..engine import resolve_spot_regime2_spec
from ..engines.signals import EmaDecisionEngine, SupertrendEngine
from .evaluator_common import (
    BarLike,
    SpotGateBand,
    SpotRegimeGatePolicy,
    _bars_input_list,
)
from .policy_contract import source_value as _get


def _optional_float(source: object, key: str, *, nonnegative: bool = False) -> float | None:
    raw = _get(source, key, None)
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return None if value is not None and nonnegative and value < 0 else value


def _optional_int(
    source: object,
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    raw = _get(source, key, None)
    try:
        value = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if value is None:
        return None
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _gate_band(
    source: object,
    minimum_key: str,
    maximum_key: str,
    *,
    integer: bool = False,
    nonnegative: bool = False,
) -> SpotGateBand:
    if integer:
        minimum = _optional_int(source, minimum_key, minimum=0 if nonnegative else None)
        maximum = _optional_int(source, maximum_key, minimum=0 if nonnegative else None)
    else:
        minimum = _optional_float(source, minimum_key, nonnegative=nonnegative)
        maximum = _optional_float(source, maximum_key, nonnegative=nonnegative)
    if minimum is not None and maximum is not None and maximum < minimum:
        maximum = minimum
    return SpotGateBand(minimum, maximum)


def _regime_gate_policy(strategy: object) -> SpotRegimeGatePolicy:
    """Decode historical `regime2_*` keys once; core policy stays semantic."""
    prearm_scope = str(
        _get(strategy, "regime2_crash_prearm_apply_to", "off") or "off"
    ).strip().lower()
    if prearm_scope not in ("off", "branch_b_longs", "all_longs"):
        prearm_scope = "off"
    return SpotRegimeGatePolicy(
        crash_atr_min=_optional_float(
            strategy,
            "regime2_crash_atr_pct_min",
            nonnegative=True,
        ),
        crash_block_longs=bool(
            _get(strategy, "regime2_crash_block_longs", False)
        ),
        transition_hot_atr_min=_optional_float(
            strategy,
            "regime2_transition_hot_shock_atr_pct_min",
            nonnegative=True,
        ),
        transition_hot_release_age_max=_optional_int(
            strategy,
            "regime2_transition_hot_release_max_bars",
            minimum=0,
        ),
        crash_prearm_scope=prearm_scope,
        crash_prearm_atr_min=_optional_float(
            strategy,
            "regime2_crash_prearm_shock_atr_pct_min",
            nonnegative=True,
        ),
        crash_prearm_ret_max=_optional_float(
            strategy,
            "regime2_crash_prearm_shock_dir_ret_sum_pct_max",
        ),
        crash_prearm_branch_a_atr_min=_optional_float(
            strategy,
            "regime2_crash_prearm_branch_a_shock_atr_pct_min",
            nonnegative=True,
        ),
        crash_prearm_branch_a_ret_max=_optional_float(
            strategy,
            "regime2_crash_prearm_branch_a_shock_dir_ret_sum_pct_max",
        ),
        repair_branch_b_block=bool(
            _get(strategy, "regime2_repair_block_branch_b_longs", False)
        ),
        repair_branch_b_atr_max=_optional_float(
            strategy,
            "regime2_repair_branch_b_long_max_shock_atr_pct",
            nonnegative=True,
        ),
        repair_branch_b_after_hour=_optional_int(
            strategy,
            "regime2_repair_branch_b_long_block_after_hour_et",
            minimum=0,
            maximum=23,
        ),
        upcorridor_branch_a_mid_atr=_gate_band(
            strategy,
            "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min",
            "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_a_extreme_atr_min=_optional_float(
            strategy,
            "regime2_upcorridor_branch_a_long_extreme_shock_atr_pct_min",
            nonnegative=True,
        ),
        upcorridor_branch_a_fresh_age_max=_optional_int(
            strategy,
            "regime2_upcorridor_branch_a_long_fresh_release_age_max_bars",
            minimum=0,
        ),
        upcorridor_branch_a_stale_age_min=_optional_int(
            strategy,
            "regime2_upcorridor_branch_a_long_stale_release_age_min_bars",
            minimum=0,
        ),
        upcorridor_branch_b_stale_age_min=_optional_int(
            strategy,
            "regime2_upcorridor_branch_b_long_stale_release_age_min_bars",
            minimum=0,
        ),
        upcorridor_branch_b_flat_low_atr_max=_optional_float(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_low_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_low_stale_age_min=_optional_int(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_low_stale_release_age_min_bars",
            minimum=0,
        ),
        upcorridor_branch_b_flat_atr_max=_optional_float(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_shock_atr_pct_max",
            nonnegative=True,
        ),
        upcorridor_branch_b_flat_ddv_abs_max=_optional_float(
            strategy,
            "regime2_upcorridor_branch_b_long_flat_ddv_abs_max_pp",
            nonnegative=True,
        ),
        trenddown_branch_b_release_age=_gate_band(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_release_age_min_bars",
            "regime2_trenddown_branch_b_long_hard_up_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        trenddown_branch_b_atr=_gate_band(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_min",
            "regime2_trenddown_branch_b_long_hard_up_shock_atr_pct_max",
            nonnegative=True,
        ),
        trenddown_branch_b_ddv=_gate_band(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_ddv_min_pp",
            "regime2_trenddown_branch_b_long_hard_up_ddv_max_pp",
        ),
        trenddown_branch_b_recovery_atr=_gate_band(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_min",
            "regime2_trenddown_branch_b_long_hard_up_recovery_shock_atr_pct_max",
            nonnegative=True,
        ),
        trenddown_branch_b_recovery_ddv=_gate_band(
            strategy,
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_min_pp",
            "regime2_trenddown_branch_b_long_hard_up_recovery_ddv_max_pp",
        ),
        continuation_branch_b_release_age=_gate_band(
            strategy,
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_min_bars",
            "regime2_continuation_confidence_branch_b_trend_up_clean_release_age_max_bars",
            integer=True,
            nonnegative=True,
        ),
        continuation_branch_a_release_age_max=_optional_int(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_release_age_max_bars",
            minimum=0,
        ),
        continuation_branch_a_atr=_gate_band(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_min",
            "regime2_continuation_confidence_branch_a_transition_shock_atr_pct_max",
            nonnegative=True,
        ),
        continuation_branch_a_ddv_max=_optional_float(
            strategy,
            "regime2_continuation_confidence_branch_a_transition_ddv_max_pp",
        ),
    )


def _coerce(raw: object, cast, default):
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


class SpotSignalSetupMixin:
    def _configure_secondary_regime(
        self,
        *,
        strategy: object,
        regime2_bars: list[BarLike] | BarSeries[BarLike] | None,
        regime2_bear_hard_bars: list[BarLike] | BarSeries[BarLike] | None,
    ) -> str | None:
        regime2_mode, regime2_preset, _, _ = resolve_spot_regime2_spec(
            bar_size=self._bar_size,
            regime2_mode_raw=_get(strategy, "regime2_mode", "off"),
            regime2_ema_preset_raw=_get(strategy, "regime2_ema_preset", ""),
            regime2_bar_size_raw=_get(strategy, "regime2_bar_size", ""),
        )
        self._regime2_mode = regime2_mode
        self._use_mtf_regime2 = bool(regime2_bars)
        self._regime2_bars = _bars_input_list(regime2_bars)
        self._regime2_idx = 0

        fast_atr = _coerce(
            _get(strategy, "regime2_supertrend_atr_period", 10) or 10,
            int,
            10,
        )
        fast_mult = _coerce(
            _get(strategy, "regime2_supertrend_multiplier", 3.0) or 3.0,
            float,
            3.0,
        )
        fast_source = str(
            _get(strategy, "regime2_supertrend_source", "hl2") or "hl2"
        ).strip() or "hl2"
        self._regime2_engine = (
            EmaDecisionEngine(
                ema_preset=str(regime2_preset),
                ema_entry_mode="trend",
                entry_confirm_bars=0,
                regime_ema_preset=None,
            )
            if regime2_mode == "ema" and regime2_preset
            else None
        )
        self._supertrend2_engine = (
            SupertrendEngine(
                atr_period=fast_atr,
                multiplier=fast_mult,
                source=fast_source,
            )
            if regime2_mode == "supertrend"
            else None
        )
        self._last_regime2 = None
        self._last_supertrend2 = None
        self._fast_regime_dir: str | None = None
        self._fast_regime_ready = False

        self._regime2_clean_host_enable = bool(
            _get(strategy, "regime2_clean_host_enable", False)
        )
        self._regime2_clean_host_takeover_state = str(
            _get(strategy, "regime2_clean_host_takeover_state", "trend_up_clean")
            or "trend_up_clean"
        ).strip().lower()
        if self._regime2_clean_host_takeover_state not in (
            "trend_up_clean",
            "crash_down",
            "transition_up_hot",
            "crash_or_transition_up_hot",
        ):
            self._regime2_clean_host_takeover_state = "trend_up_clean"
        clean_mult = _coerce(
            _get(strategy, "regime2_clean_host_supertrend_multiplier", None),
            float,
            fast_mult,
        )
        self._clean_supertrend2_engine = (
            SupertrendEngine(
                atr_period=fast_atr,
                multiplier=max(0.01, clean_mult),
                source=fast_source,
            )
            if self._regime2_clean_host_enable and regime2_mode == "supertrend"
            else None
        )
        self._last_clean_supertrend2 = None
        self._clean_regime2_idx = 0

        self._regime2_bear_hard_mode = str(
            _get(strategy, "regime2_bear_hard_mode", "off") or "off"
        ).strip().lower()
        if self._regime2_bear_hard_mode not in ("off", "supertrend"):
            self._regime2_bear_hard_mode = "off"
        self._use_mtf_regime2_bear_hard = bool(regime2_bear_hard_bars)
        self._regime2_bear_hard_bars = _bars_input_list(regime2_bear_hard_bars)
        self._regime2_bear_hard_idx = 0
        hard_atr = _coerce(
            _get(strategy, "regime2_bear_hard_supertrend_atr_period", None),
            int,
            fast_atr,
        )
        hard_mult = _coerce(
            _get(strategy, "regime2_bear_hard_supertrend_multiplier", None),
            float,
            fast_mult,
        )
        hard_source = str(
            _get(strategy, "regime2_bear_hard_supertrend_source", None)
            or fast_source
        ).strip() or fast_source
        hard_enabled = self._regime2_bear_hard_mode == "supertrend"
        self._bear_hard_supertrend_engine = (
            SupertrendEngine(
                atr_period=max(1, hard_atr),
                multiplier=max(0.01, hard_mult),
                source=hard_source,
            )
            if hard_enabled
            else None
        )
        clean_hard_mult = _coerce(
            _get(
                strategy,
                "regime2_clean_host_bear_hard_supertrend_multiplier",
                None,
            ),
            float,
            hard_mult,
        )
        self._clean_bear_hard_supertrend_engine = (
            SupertrendEngine(
                atr_period=max(1, hard_atr),
                multiplier=max(0.01, clean_hard_mult),
                source=hard_source,
            )
            if hard_enabled and self._regime2_clean_host_enable
            else None
        )
        self._last_bear_hard_supertrend = None
        self._last_clean_bear_hard_supertrend = None
        self._clean_regime2_bear_hard_idx = 0

        self._regime2_bear_entry_mode = str(
            _get(strategy, "regime2_bear_entry_mode", "off") or "off"
        ).strip().lower()
        if self._regime2_bear_entry_mode not in ("off", "supertrend"):
            self._regime2_bear_entry_mode = "off"
        self._regime2_bear_allow_long_recovery = bool(
            _get(strategy, "regime2_bear_allow_long_recovery", True)
        )
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
        self._regime_gates = _regime_gate_policy(strategy)

        bear_atr = _coerce(
            _get(strategy, "regime2_bear_supertrend_atr_period", None),
            int,
            _coerce(_get(strategy, "supertrend_atr_period", 10) or 10, int, 10),
        )
        bear_mult = _coerce(
            _get(strategy, "regime2_bear_supertrend_multiplier", None),
            float,
            _coerce(
                _get(strategy, "supertrend_multiplier", 3.0) or 3.0,
                float,
                3.0,
            ),
        )
        bear_source = str(
            _get(strategy, "regime2_bear_supertrend_source", None)
            or _get(strategy, "supertrend_source", "hl2")
            or "hl2"
        ).strip() or "hl2"
        bear_enabled = self._regime2_bear_entry_mode == "supertrend"
        self._bear_supertrend_engine = (
            SupertrendEngine(
                atr_period=max(1, bear_atr),
                multiplier=max(0.01, bear_mult),
                source=bear_source,
            )
            if bear_enabled
            else None
        )
        clean_bear_mult = _coerce(
            _get(strategy, "regime2_clean_host_bear_supertrend_multiplier", None),
            float,
            bear_mult,
        )
        self._clean_bear_supertrend_engine = (
            SupertrendEngine(
                atr_period=max(1, bear_atr),
                multiplier=max(0.01, clean_bear_mult),
                source=bear_source,
            )
            if bear_enabled and self._regime2_clean_host_enable
            else None
        )
        self._last_bear_supertrend = None
        self._bear_prev_dir: str | None = None
        self._last_clean_bear_supertrend = None
        self._clean_bear_prev_dir: str | None = None
        return regime2_preset
