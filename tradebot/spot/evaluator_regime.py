"""Regime and shock-state advancement for spot signals."""
from __future__ import annotations

from ..engine import spot_regime_apply_matches_direction
from ..engines.signals import EmaDecisionSnapshot, SupertrendEngine
from .evaluator_common import (
    BarLike,
    SpotEntryCandidate,
    SpotRegimeState,
    SpotSignalSelection,
    _get,
)
from .gates import apply_regime_gate


class SpotSignalRegimeMixin:
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

    def _advance_primary_regime_and_shock(
        self,
        *,
        bar: BarLike,
        signal: EmaDecisionSnapshot | None,
    ) -> EmaDecisionSnapshot | None:
        """Advance primary regime/shock engines and apply the primary direction gate."""
        if self._supertrend_engine is not None:
            intraday_shock = self._shock_engine is not None and self._shock_detector not in (
                "daily_atr_pct",
                "daily_drawdown",
            )

            def _advance(regime_bar: BarLike, *, mtf: bool) -> None:
                values = {
                    "high": float(regime_bar.high),
                    "low": float(regime_bar.low),
                    "close": float(regime_bar.close),
                }
                self._last_supertrend = self._supertrend_engine.update(**values)
                if self._supertrend_shock_engine is not None:
                    self._last_supertrend_shock = self._supertrend_shock_engine.update(**values)
                if self._supertrend_cooling_engine is not None:
                    self._last_supertrend_cooling = self._supertrend_cooling_engine.update(**values)
                if intraday_shock:
                    direction = (
                        {"update_direction": self._shock_dir_source != "signal"}
                        if mtf
                        else {}
                    )
                    self._last_shock = self._shock_engine.update(**values, **direction)

            if self._use_mtf_regime and self._regime_bars:
                while (
                    self._regime_idx < len(self._regime_bars)
                    and self._regime_bars[self._regime_idx].ts <= bar.ts
                ):
                    _advance(self._regime_bars[self._regime_idx], mtf=True)
                    self._regime_idx += 1
            else:
                _advance(bar, mtf=False)

            gate = self._last_supertrend
            shock_ready = bool(
                self._last_shock is not None
                and (
                    self._shock_detector in ("daily_atr_pct", "daily_drawdown")
                    or bool(getattr(self._last_shock, "ready", False))
                )
            )
            shock_now = bool(
                shock_ready and getattr(self._last_shock, "shock", False)
            )
            if (
                self._shock_engine is not None
                and self._last_shock is not None
                and (
                    self._supertrend_shock_engine is not None
                    or self._supertrend_cooling_engine is not None
                )
            ):
                cooling_raw = _get(
                    self._filters,
                    "shock_daily_cooling_atr_pct",
                    None,
                )
                cooling_atr = (
                    float(cooling_raw or 0.0) if cooling_raw is not None else None
                )
                atr_pct = getattr(self._last_shock, "atr_pct", None)
                cooling_now = bool(
                    not shock_now
                    and cooling_atr is not None
                    and atr_pct is not None
                    and self._shock_detector == "daily_atr_pct"
                    and shock_ready
                    and float(atr_pct) >= float(cooling_atr)
                )
                if shock_now and self._last_supertrend_shock is not None:
                    gate = self._last_supertrend_shock
                elif cooling_now and self._last_supertrend_cooling is not None:
                    gate = self._last_supertrend_cooling

            regime_dir = gate.direction if gate is not None else None
            regime_ready = bool(gate and gate.ready)
            if (
                bool(_get(self._filters, "shock_regime_override_dir", False))
                and self._shock_engine is not None
                and self._last_shock is not None
            ):
                if (
                    shock_now
                    and bool(getattr(self._last_shock, "direction_ready", False))
                    and getattr(self._last_shock, "direction", None) in ("up", "down")
                ):
                    regime_dir = str(getattr(self._last_shock, "direction"))
                    regime_ready = True

            signal = apply_regime_gate(
                signal,
                regime_dir=regime_dir,
                regime_ready=regime_ready,
            )
        elif self._use_mtf_regime and self._regime_engine is not None and self._regime_bars:
            while (
                self._regime_idx < len(self._regime_bars)
                and self._regime_bars[self._regime_idx].ts <= bar.ts
            ):
                reg_bar = self._regime_bars[self._regime_idx]
                if float(reg_bar.close) > 0:
                    self._last_regime = self._regime_engine.update(float(reg_bar.close))
                if self._shock_engine is not None and self._shock_detector not in (
                    "daily_atr_pct",
                    "daily_drawdown",
                ):
                    self._last_shock = self._shock_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                        update_direction=(self._shock_dir_source != "signal"),
                    )
                self._regime_idx += 1
            signal = apply_regime_gate(
                signal,
                regime_dir=(
                    self._last_regime.state
                    if self._last_regime is not None
                    else None
                ),
                regime_ready=bool(self._last_regime and self._last_regime.ema_ready),
            )
        elif (
            self._shock_engine is not None
            and self._shock_detector not in ("daily_atr_pct", "daily_drawdown")
            and not self._use_mtf_regime
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
            self._last_shock = self._shock_engine.update_direction(
                close=float(bar.close)
            )
        if (
            self._shock_engine is not None
            and self._shock_detector in ("daily_atr_pct", "daily_drawdown")
            and self._shock_dir_source == "signal"
        ):
            # Exec bars advance daily shock magnitude; signal bars only advance direction.
            if hasattr(self._shock_engine, "update_direction"):
                self._last_shock = self._shock_engine.update_direction(
                    close=float(bar.close)
                )
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
        return signal

    def _apply_confirmation_regime(
        self,
        *,
        selection: SpotSignalSelection,
        bar: BarLike,
        close: float,
        regime: SpotRegimeState,
    ) -> SpotSignalSelection:
        """Apply the configured confirmation/bear regime to one normalized signal."""
        signal = selection.signal
        candidate = selection.candidate
        if (self._supertrend2_engine is not None or self._regime2_engine is not None) and (
            spot_regime_apply_matches_direction(
                apply_to_raw=_get(self._strategy, "regime2_apply_to", "both"),
                entry_dir=getattr(signal, "entry_dir", None),
            )
        ):
            signal = apply_regime_gate(
                signal,
                regime_dir=regime.fast_dir,
                regime_ready=regime.fast_ready,
            )

        if signal is not None and self._regime2_bear_entry_mode == "supertrend":
            signal, bear_direction = self._apply_regime2_bear_primary(
                branch_key=str(selection.branch_key or "single"),
                signal=signal,
                bar=bar,
                close=float(close),
                regime=regime,
            )
            if bear_direction in ("up", "down"):
                if self._dual_branch_enabled and candidate.branch not in ("a", "b"):
                    candidate = SpotEntryCandidate(None)
                else:
                    branch = candidate.branch
                    if branch not in ("a", "b") or getattr(signal, "entry_dir", None) != bear_direction:
                        branch = None
                    candidate = SpotEntryCandidate(str(bear_direction), branch)

        if signal is None:
            return SpotSignalSelection(None, candidate, selection.branch_key)

        gated_direction = signal.entry_dir if signal.entry_dir in ("up", "down") else None
        direction = candidate.direction if candidate.direction in ("up", "down") else None
        branch = candidate.branch
        if direction is None or direction != gated_direction:
            direction = None
            branch = None
        elif not self._dual_branch_enabled:
            branch = None
        return SpotSignalSelection(
            signal=signal,
            candidate=SpotEntryCandidate(direction, branch),
            branch_key=selection.branch_key,
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

    def _classify_regime_state(
        self,
        *,
        shock_atr_pct: float | None,
        fast_dir: str | None,
        fast_ready: bool,
        hard_dir: str | None,
        hard_ready: bool,
        hard_release_age_bars: int | None,
    ) -> tuple[str | None, bool]:
        fast_dir = str(fast_dir) if fast_ready and fast_dir in ("up", "down") else None
        hard_dir = str(hard_dir) if hard_ready and hard_dir in ("up", "down") else None
        if hard_dir == "down":
            if fast_dir == "up":
                return "transition_up_hot", True
            if fast_dir in ("down", None):
                if (
                    self._regime_gates.crash_atr_min is not None
                    and shock_atr_pct is not None
                    and float(shock_atr_pct) >= self._regime_gates.crash_atr_min
                ):
                    return "crash_down", False
                return "trend_down", False
        if fast_dir == "up":
            transition_hot = bool(
                self._regime_gates.transition_hot_atr_min is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct)
                >= self._regime_gates.transition_hot_atr_min
            )
            if (
                not transition_hot
                and self._regime_gates.transition_hot_release_age_max is not None
                and hard_release_age_bars is not None
                and int(hard_release_age_bars)
                <= self._regime_gates.transition_hot_release_age_max
            ):
                transition_hot = True
            return ("transition_up_hot" if transition_hot else "trend_up_clean"), transition_hot
        if fast_dir == "down":
            return "trend_down", False
        return None, False

    def _resolve_regime_state(
        self,
        *,
        shock_atr_pct: float | None,
        fast_dir: str | None,
        fast_ready: bool,
        hard_dir: str | None,
        hard_ready: bool,
        hard_release_age_bars: int | None,
        clean_fast_dir: str | None = None,
        clean_fast_ready: bool = False,
        clean_hard_dir: str | None = None,
        clean_hard_ready: bool = False,
        clean_hard_release_age_bars: int | None = None,
    ) -> SpotRegimeState:
        label, transition_hot = self._classify_regime_state(
            shock_atr_pct=shock_atr_pct,
            fast_dir=fast_dir,
            fast_ready=fast_ready,
            hard_dir=hard_dir,
            hard_ready=hard_ready,
            hard_release_age_bars=hard_release_age_bars,
        )
        state = SpotRegimeState(
            label=label,
            transition_hot=transition_hot,
            fast_dir=str(fast_dir) if fast_dir in ("up", "down") else None,
            fast_ready=bool(fast_ready),
            hard_dir=str(hard_dir) if hard_dir in ("up", "down") else None,
            hard_ready=bool(hard_ready),
            hard_release_age_bars=hard_release_age_bars,
        )
        if self._regime2_clean_host_enable:
            clean_label, clean_hot = self._classify_regime_state(
                shock_atr_pct=shock_atr_pct,
                fast_dir=clean_fast_dir,
                fast_ready=clean_fast_ready,
                hard_dir=clean_hard_dir,
                hard_ready=clean_hard_ready,
                hard_release_age_bars=clean_hard_release_age_bars,
            )
            takeover_mode = self._regime2_clean_host_takeover_state
            takeover_labels = {
                "trend_up_clean": ("trend_up_clean",),
                "crash_down": ("crash_down",),
                "transition_up_hot": ("transition_up_hot",),
                "crash_or_transition_up_hot": (
                    "crash_down",
                    "transition_up_hot",
                ),
            }
            clean_takeover = bool(
                clean_label in takeover_labels.get(takeover_mode, ())
                and (takeover_mode != "trend_up_clean" or label == clean_label)
            )
            if clean_takeover:
                state = SpotRegimeState(
                    label=clean_label,
                    owner="clean_host",
                    transition_hot=clean_hot,
                    fast_dir=str(clean_fast_dir) if clean_fast_dir in ("up", "down") else None,
                    fast_ready=bool(clean_fast_ready),
                    hard_dir=str(clean_hard_dir) if clean_hard_dir in ("up", "down") else None,
                    hard_ready=bool(clean_hard_ready),
                    hard_release_age_bars=clean_hard_release_age_bars,
                )
        return state

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
        self._fast_regime_dir = str(regime2_dir) if regime2_dir in ("up", "down") else None
        self._fast_regime_ready = bool(regime2_ready)
        return self._fast_regime_dir, self._fast_regime_ready

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
        hard_dir = str(hard_dir) if hard_dir in ("up", "down") else None
        return hard_dir, bool(hard_ready)

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
        clean_dir = str(clean_dir) if clean_dir in ("up", "down") else None
        return clean_dir, bool(clean_ready)

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
        clean_hard_dir = (
            str(clean_hard_dir) if clean_hard_dir in ("up", "down") else None
        )
        return clean_hard_dir, bool(clean_hard_ready)

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
