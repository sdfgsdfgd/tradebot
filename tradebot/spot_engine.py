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
from dataclasses import dataclass
from datetime import datetime, time
from typing import Mapping, Protocol

from .engine import (
    EmaDecisionEngine,
    EmaDecisionSnapshot,
    OrbDecisionEngine,
    RiskOverlaySnapshot,
    SupertrendEngine,
    annualized_ewma_vol,
    apply_regime_gate,
    build_shock_engine,
    build_tr_pct_risk_overlay_engine,
    normalize_shock_detector,
    normalize_shock_direction_source,
    parse_time_hhmm,
)
from .signals import ema_next, ema_periods


# region Protocols / Helpers
class BarLike(Protocol):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


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
        rv_lookback: int = 60,
        rv_ewma_lambda: float = 0.94,
        regime_bars: list[BarLike] | None = None,
        regime2_bars: list[BarLike] | None = None,
    ) -> None:
        self._strategy = strategy
        self._filters = filters
        self._bar_size = str(bar_size)
        self._use_rth = bool(use_rth)

        self._rv_lookback = max(1, int(rv_lookback))
        self._rv_lam = float(rv_ewma_lambda)
        self._returns: deque[float] = deque(maxlen=self._rv_lookback)
        self._prev_sig_close: float | None = None

        self._sig_last_date = None
        self._sig_bars_in_day = 0

        # Entry signal
        entry_signal = str(_get(strategy, "entry_signal", "ema") or "ema").strip().lower()
        if entry_signal not in ("ema", "orb"):
            entry_signal = "ema"
        self.entry_signal = entry_signal

        # Regime mode (primary)
        regime_mode = str(_get(strategy, "regime_mode", "ema") or "ema").strip().lower()
        if regime_mode not in ("ema", "supertrend"):
            regime_mode = "ema"
        self._regime_mode = regime_mode
        regime_preset = str(_get(strategy, "regime_ema_preset", "") or "").strip() or None

        # Multi-timeframe regime: if provided, caller already fetched the right bars.
        self._use_mtf_regime = bool(regime_bars)
        self._regime_bars = list(regime_bars) if regime_bars else []
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
        self._orb_engine: OrbDecisionEngine | None = None
        if entry_signal == "ema":
            ema_preset = str(_get(strategy, "ema_preset", "") or "").strip()
            if not ema_preset:
                raise ValueError("EMA entry requires ema_preset")
            # Mirror backtest semantics: only embed same-timeframe EMA regime inside the EMA engine.
            embedded_regime = None
            if (not self._use_mtf_regime) and self._regime_mode != "supertrend":
                embedded_regime = regime_preset
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

        # Risk overlay (daily TR% heuristics)
        self._risk_overlay = build_tr_pct_risk_overlay_engine(filters)
        self._last_risk: RiskOverlaySnapshot | None = None

        # Regime2 gating (secondary)
        regime2_mode = str(_get(strategy, "regime2_mode", "off") or "off").strip().lower()
        if regime2_mode not in ("off", "ema", "supertrend"):
            regime2_mode = "off"
        self._regime2_mode = regime2_mode

        self._use_mtf_regime2 = bool(regime2_bars)
        self._regime2_bars = list(regime2_bars) if regime2_bars else []
        self._regime2_idx = 0

        regime2_preset = str(_get(strategy, "regime2_ema_preset", "") or "").strip() or None
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
            )

        if self._shock_engine is None:
            return
        if self._shock_detector not in ("daily_atr_pct", "daily_drawdown"):
            return
        self._last_shock = self._shock_engine.update(
            day=bar.ts.date(),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            update_direction=(self._shock_dir_source != "signal"),
        )

    def _shock_view(self) -> tuple[bool | None, str | None, float | None]:
        if self._shock_engine is None:
            return None, None, None
        if self._last_shock is None:
            return None, None, None
        ready_ok = (
            self._shock_detector in ("daily_atr_pct", "daily_drawdown")
            or bool(getattr(self._last_shock, "ready", False))
        )
        if not ready_ok:
            return None, None, None

        shock = bool(getattr(self._last_shock, "shock", False))
        shock_dir = (
            str(getattr(self._last_shock, "direction"))
            if bool(getattr(self._last_shock, "direction_ready", False))
            and getattr(self._last_shock, "direction", None) in ("up", "down")
            else None
        )
        atr_pct = getattr(self._last_shock, "atr_pct", None)
        if atr_pct is None:
            atr_pct = getattr(self._last_shock, "atr_fast_pct", None)
        shock_atr_pct = float(atr_pct) if atr_pct is not None else None
        return shock, shock_dir, shock_atr_pct

    def update_signal_bar(self, bar: BarLike) -> SpotSignalSnapshot | None:
        """Update the evaluator for a single signal bar close."""
        close = float(bar.close)
        if close <= 0:
            return None

        if self._sig_last_date != bar.ts.date():
            self._sig_last_date = bar.ts.date()
            self._sig_bars_in_day = 0
        self._sig_bars_in_day += 1

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
        if self._signal_engine is not None:
            signal = self._signal_engine.update(close)
        elif self._orb_engine is not None:
            signal = self._orb_engine.update(
                ts=bar.ts,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )

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
            self._last_shock = self._shock_engine.update(
                day=bar.ts.date(),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=True,
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

            regime2_apply_to = str(_get(self._strategy, "regime2_apply_to", "both") or "both").strip().lower()
            apply_regime2 = True
            if regime2_apply_to == "longs":
                apply_regime2 = bool(signal is not None and signal.entry_dir == "up")
            elif regime2_apply_to == "shorts":
                apply_regime2 = bool(signal is not None and signal.entry_dir == "down")
            if apply_regime2:
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

            regime2_apply_to = str(_get(self._strategy, "regime2_apply_to", "both") or "both").strip().lower()
            apply_regime2 = True
            if regime2_apply_to == "longs":
                apply_regime2 = bool(signal is not None and signal.entry_dir == "up")
            elif regime2_apply_to == "shorts":
                apply_regime2 = bool(signal is not None and signal.entry_dir == "down")
            if apply_regime2:
                signal = apply_regime_gate(
                    signal,
                    regime_dir=self._last_regime2.state if self._last_regime2 is not None else None,
                    regime_ready=bool(self._last_regime2 and self._last_regime2.ema_ready),
                )

        if signal is None:
            return None

        shock, shock_dir, shock_atr_pct = self._shock_view()
        atr = (
            float(self._last_exit_atr.atr)
            if self._last_exit_atr is not None and bool(self._last_exit_atr.ready) and self._last_exit_atr.atr is not None
            else None
        )

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
        )
        self._last_signal = signal
        self._last_snapshot = snap
        return snap
# endregion
