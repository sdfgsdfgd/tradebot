"""Sequential signal-bar evaluation for the shared spot engine."""
from __future__ import annotations

import math

from ..engine import annualized_ewma_vol
from ..signals import ema_next
from .evaluator_common import (
    BarLike,
    SpotEntryCandidate,
    SpotEntryGateContext,
    SpotSignalSnapshot,
    _get,
)


class SpotSignalRuntimeMixin:
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

        selection = self._advance_entry_signal(bar=bar, close=close)
        signal = selection.signal
        entry_dir_for_entries = selection.candidate.direction
        entry_branch = selection.candidate.branch
        ratsv_branch_key = selection.branch_key

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

        signal = self._advance_primary_regime_and_shock(bar=bar, signal=signal)

        shock, shock_dir, shock_atr_pct = self._shock_view()
        regime = self._resolve_regime_state(
            shock_atr_pct=shock_atr_pct,
            fast_dir=regime2_dir,
            fast_ready=bool(regime2_ready),
            hard_dir=regime2_bear_hard_dir,
            hard_ready=bool(regime2_bear_hard_ready),
            hard_release_age_bars=self._regime2_bear_hard_release_age_bars,
            clean_fast_dir=clean_regime2_dir,
            clean_fast_ready=clean_regime2_ready,
            clean_hard_dir=clean_regime2_bear_hard_dir,
            clean_hard_ready=clean_regime2_bear_hard_ready,
            clean_hard_release_age_bars=self._clean_regime2_bear_hard_release_age_bars,
        )

        selection = self._apply_confirmation_regime(
            selection=selection,
            bar=bar,
            close=close,
            regime=regime,
        )
        signal = selection.signal
        if signal is None:
            return None
        entry_dir_for_entries = selection.candidate.direction
        entry_branch = selection.candidate.branch

        router_snap = self._regime_router.update_bar(
            ts=str(bar.ts.isoformat()),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            hf_entry_dir=str(entry_dir_for_entries) if entry_dir_for_entries in ("up", "down") else None,
        )
        if bool(self._regime_router_cfg.enabled):
            entry_dir_for_entries = (
                str(router_snap.effective_entry_dir) if router_snap.effective_entry_dir in ("up", "down") else None
            )
            if str(router_snap.chosen_host or "") != "hf_host":
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

        candidate = self._apply_entry_gates(
            SpotEntryCandidate(entry_dir_for_entries, entry_branch),
            SpotEntryGateContext(
                bar_ts=bar.ts,
                regime=regime,
                shock_dir=shock_dir,
                shock_atr_pct=shock_atr_pct,
                shock_dir_ret_sum_pct=shock_dir_ret_sum_pct,
                shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
                router=router_snap,
            ),
        )
        entry_dir_for_entries, entry_branch = candidate.direction, candidate.branch

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
            regime2_dir=regime.fast_dir,
            regime2_ready=regime.fast_ready,
            regime2_bear_hard_dir=regime.hard_dir,
            regime2_bear_hard_ready=regime.hard_ready,
            regime2_bear_hard_release_age_bars=(
                int(regime.hard_release_age_bars)
                if regime.hard_release_age_bars is not None
                else None
            ),
            regime4_state=regime.label,
            regime4_transition_hot=regime.transition_hot,
            regime4_owner=regime.owner,
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
            regime_router_ready=bool(router_snap.ready),
            regime_router_climate=str(router_snap.climate) if router_snap.climate else None,
            regime_router_host=str(router_snap.chosen_host) if router_snap.chosen_host else None,
            regime_router_entry_dir=(
                str(router_snap.effective_entry_dir) if router_snap.effective_entry_dir in ("up", "down") else None
            ),
            regime_router_host_managed=bool(router_snap.host_managed),
            regime_router_bull_sovereign_ok=bool(router_snap.bull_sovereign_ok),
            regime_router_dwell_days=int(getattr(router_snap, "dwell_days", 0) or 0),
            regime_router_crash_ret=(
                float(getattr(router_snap, "crash_ret", None)) if getattr(router_snap, "crash_ret", None) is not None else None
            ),
            regime_router_crash_maxdd=(
                float(getattr(router_snap, "crash_maxdd", None))
                if getattr(router_snap, "crash_maxdd", None) is not None
                else None
            ),
            regime_router_crash_rv=(
                float(getattr(router_snap, "crash_rv", None)) if getattr(router_snap, "crash_rv", None) is not None else None
            ),
            regime_router_fast_ret=(
                float(getattr(router_snap, "fast_ret", None)) if getattr(router_snap, "fast_ret", None) is not None else None
            ),
            regime_router_slow_ret=(
                float(getattr(router_snap, "slow_ret", None)) if getattr(router_snap, "slow_ret", None) is not None else None
            ),
            regime=regime,
        )
        self._last_signal = signal
        self._last_snapshot = snap
        return snap
