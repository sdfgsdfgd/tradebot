"""Construction and legacy-key decoding for the canonical spot regime model."""

from __future__ import annotations

from ..chart_data.series import BarSeries
from ..engines.signals import EmaDecisionEngine, SupertrendEngine
from .evaluator_common import (
    BarLike,
    _bars_input_list,
)
from .policy_contract import source_value as _get


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
        regime2_mode = self._entry_control_plan.confirmation_regime
        regime2_preset = self._entry_control_plan.confirmation_regime_preset
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

        self._regime2_bear_entry_mode = self._entry_control_plan.bear_takeover
        self._regime2_bear_allow_long_recovery = bool(
            _get(strategy, "regime2_bear_allow_long_recovery", True)
        )
        self._regime2_bear_takeover_mode = (
            self._entry_control_plan.bear_takeover_scope
        )
        self._regime_gates = self._entry_control_plan.regime_gates

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
