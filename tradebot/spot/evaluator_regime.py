"""Regime and shock-state advancement for spot signals."""
from __future__ import annotations

from ..engines.signals import SupertrendEngine
from .evaluator_common import BarLike


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
