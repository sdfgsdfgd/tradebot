"""Spot position sizing policy shared by backtests and live execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .graph import SpotPolicyGraph
from .policy_contract import SpotDecisionTrace, SpotPolicyConfigView


class SpotSizingPolicy:
    """Sizing capability composed by the canonical :class:`SpotPolicy`."""

    @classmethod
    def calc_signed_qty_with_trace(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        filters: Mapping[str, object] | object | None,
        action: str,
        lot: int,
        entry_price: float,
        stop_price: float | None,
        stop_loss_pct: float | None,
        shock: bool | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_down_streak_bars: int | None = None,
        shock_drawdown_dist_on_pct: float | None = None,
        shock_drawdown_dist_on_vel_pp: float | None = None,
        shock_drawdown_dist_on_accel_pp: float | None = None,
        shock_prearm_down_streak_bars: int | None = None,
        shock_ramp: Mapping[str, object] | object | None = None,
        riskoff: bool = False,
        risk_dir: str | None = None,
        riskpanic: bool = False,
        riskpop: bool = False,
        risk: object | None = None,
        signal_entry_dir: str | None = None,
        signal_regime_dir: str | None = None,
        regime2_dir: str | None = None,
        regime2_ready: bool = False,
        equity_ref: float = 0.0,
        cash_ref: float | None = None,
        policy_graph: SpotPolicyGraph | None = None,
        policy_config: SpotPolicyConfigView | None = None,
    ) -> tuple[int, SpotDecisionTrace]:
        cfg = policy_config or cls.policy_config(strategy=strategy, filters=filters)

        lot_i = max(1, int(lot or 1))
        raw_action = str(action or "BUY").strip().upper()
        if raw_action not in ("BUY", "SELL"):
            raw_action = "BUY"
        base_signed_qty = int(lot_i) * int(cfg.quantity_mult)
        if raw_action != "BUY":
            base_signed_qty = -int(base_signed_qty)

        try:
            entry_price_f = float(entry_price)
        except (TypeError, ValueError):
            entry_price_f = 0.0

        stop_price_clean = None
        if stop_price is not None:
            try:
                stop_price_f = float(stop_price)
            except (TypeError, ValueError):
                stop_price_f = 0.0
            if stop_price_f > 0:
                stop_price_clean = float(stop_price_f)

        stop_loss_pct_clean = None
        if stop_loss_pct is not None:
            try:
                sl_pct = float(stop_loss_pct)
            except (TypeError, ValueError):
                sl_pct = 0.0
            if sl_pct > 0:
                stop_loss_pct_clean = float(sl_pct)

        try:
            equity_ref_f = float(equity_ref)
        except (TypeError, ValueError):
            equity_ref_f = 0.0

        cash_ref_f = None
        if cash_ref is not None:
            try:
                cash_ref_f = float(cash_ref)
            except (TypeError, ValueError):
                cash_ref_f = None

        shock_dir_clean = str(shock_dir) if shock_dir in ("up", "down") else None
        risk_dir_clean = str(risk_dir) if risk_dir in ("up", "down") else None
        signal_entry_dir_clean = str(signal_entry_dir) if signal_entry_dir in ("up", "down") else None
        signal_regime_dir_clean = str(signal_regime_dir) if signal_regime_dir in ("up", "down") else None
        regime2_dir_clean = str(regime2_dir) if regime2_dir in ("up", "down") else None
        regime2_ready_clean = bool(regime2_ready)
        shock_dir_down_streak_clean: int | None = None
        if shock_dir_down_streak_bars is not None:
            try:
                shock_dir_down_streak_clean = max(0, int(shock_dir_down_streak_bars))
            except (TypeError, ValueError):
                shock_dir_down_streak_clean = None
        shock_drawdown_dist_on_clean: float | None = None
        if shock_drawdown_dist_on_pct is not None:
            try:
                shock_drawdown_dist_on_clean = float(shock_drawdown_dist_on_pct)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_clean = None
        shock_drawdown_dist_on_vel_clean: float | None = None
        if shock_drawdown_dist_on_vel_pp is not None:
            try:
                shock_drawdown_dist_on_vel_clean = float(shock_drawdown_dist_on_vel_pp)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_vel_clean = None
        shock_drawdown_dist_on_accel_clean: float | None = None
        if shock_drawdown_dist_on_accel_pp is not None:
            try:
                shock_drawdown_dist_on_accel_clean = float(shock_drawdown_dist_on_accel_pp)
            except (TypeError, ValueError):
                shock_drawdown_dist_on_accel_clean = None
        shock_prearm_down_streak_clean: int | None = None
        if shock_prearm_down_streak_bars is not None:
            try:
                shock_prearm_down_streak_clean = max(0, int(shock_prearm_down_streak_bars))
            except (TypeError, ValueError):
                shock_prearm_down_streak_clean = None

        shock_drawdown_dist_off_clean: float | None = None
        if shock_drawdown_dist_on_clean is not None:
            try:
                dd_on = float(cfg.shock_on_drawdown_pct)
                dd_off = float(cfg.shock_off_drawdown_pct)
                dd_dist_on = float(shock_drawdown_dist_on_clean)
                dd_pct = float(dd_on) - float(dd_dist_on)
                shock_drawdown_dist_off_clean = float(dd_pct) - float(dd_off)
            except (TypeError, ValueError):
                shock_drawdown_dist_off_clean = None

        shock_atr_pct_clean = None
        if shock_atr_pct is not None:
            try:
                shock_atr_pct_f = float(shock_atr_pct)
            except (TypeError, ValueError):
                shock_atr_pct_f = 0.0
            if shock_atr_pct_f > 0:
                shock_atr_pct_clean = float(shock_atr_pct_f)

        graph = policy_graph or SpotPolicyGraph.from_sources(strategy=strategy, filters=filters)
        risk_tr_ratio = None
        risk_slope_med = None
        risk_slope_vel = None
        if risk is not None:
            raw_tr_ratio = getattr(risk, "tr_ratio", None)
            if raw_tr_ratio is not None:
                try:
                    risk_tr_ratio = float(raw_tr_ratio)
                except (TypeError, ValueError):
                    risk_tr_ratio = None
            raw_slope_med = getattr(risk, "tr_median_delta_pct", None)
            if raw_slope_med is not None:
                try:
                    risk_slope_med = float(raw_slope_med)
                except (TypeError, ValueError):
                    risk_slope_med = None
            raw_slope_vel = getattr(risk, "tr_slope_vel_pct", None)
            if raw_slope_vel is not None:
                try:
                    risk_slope_vel = float(raw_slope_vel)
                except (TypeError, ValueError):
                    risk_slope_vel = None
        graph_overlay = graph.resolve_risk_overlay_adjustments(
            strategy=strategy,
            filters=filters,
            action=str(raw_action),
            shock_atr_pct=shock_atr_pct_clean,
            tr_ratio=risk_tr_ratio,
            slope_med_pct=risk_slope_med,
            slope_vel_pct=risk_slope_vel,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir_clean,
        )
        cap_pct_base = float(cfg.spot_max_notional_pct)
        cap_pct = float(cap_pct_base) * float(graph_overlay.cap_mult)

        trace = SpotDecisionTrace(
            action=str(raw_action),
            sizing_mode=str(cfg.sizing_mode),
            lot=int(lot_i),
            quantity_mult=int(cfg.quantity_mult),
            base_signed_qty=int(base_signed_qty),
            entry_price=float(entry_price_f),
            stop_price=float(stop_price_clean) if stop_price_clean is not None else None,
            stop_loss_pct=float(stop_loss_pct_clean) if stop_loss_pct_clean is not None else None,
            equity_ref=float(equity_ref_f),
            cash_ref=float(cash_ref_f) if cash_ref_f is not None else None,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir_clean,
            shock_dir_down_streak_bars=int(shock_dir_down_streak_clean)
            if shock_dir_down_streak_clean is not None
            else None,
            shock_atr_pct=float(shock_atr_pct_clean) if shock_atr_pct_clean is not None else None,
            shock_drawdown_dist_on_pct=(
                float(shock_drawdown_dist_on_clean) if shock_drawdown_dist_on_clean is not None else None
            ),
            shock_drawdown_dist_on_vel_pp=(
                float(shock_drawdown_dist_on_vel_clean)
                if shock_drawdown_dist_on_vel_clean is not None
                else None
            ),
            shock_drawdown_dist_on_accel_pp=(
                float(shock_drawdown_dist_on_accel_clean)
                if shock_drawdown_dist_on_accel_clean is not None
                else None
            ),
            shock_drawdown_dist_off_pct=(
                float(shock_drawdown_dist_off_clean) if shock_drawdown_dist_off_clean is not None else None
            ),
            shock_prearm_down_streak_bars=(
                int(shock_prearm_down_streak_clean) if shock_prearm_down_streak_clean is not None else None
            ),
            riskoff_mode=str(cfg.riskoff_mode),
            risk_dir=risk_dir_clean,
            signal_entry_dir=signal_entry_dir_clean,
            signal_regime_dir=signal_regime_dir_clean,
            regime2_dir=regime2_dir_clean,
            regime2_ready=bool(regime2_ready_clean),
            riskoff_long_factor=float(cfg.riskoff_long_factor),
            riskoff_short_factor=float(cfg.riskoff_short_factor),
            riskpanic_long_factor=float(cfg.riskpanic_long_factor),
            riskpanic_short_factor=float(cfg.riskpanic_short_factor),
            riskpop_long_factor=float(cfg.riskpop_long_factor),
            riskpop_short_factor=float(cfg.riskpop_short_factor),
            graph_profile=str(graph.profile_name),
            graph_entry_policy=str(graph.entry_policy),
            graph_exit_policy=str(graph.exit_policy),
            graph_resize_policy=str(graph.resize_policy),
            graph_risk_overlay_policy=str(graph.risk_overlay_policy),
            graph_overlay_trace=graph_overlay.as_payload(),
            cap_pct_base=float(cap_pct_base),
            cap_pct_final=float(cap_pct),
        )

        def _update_trace(**kwargs: object) -> None:
            nonlocal trace
            trace = replace(trace, **kwargs)

        if str(cfg.sizing_mode) == "fixed":
            abs_qty = abs(int(base_signed_qty))
            _update_trace(
                desired_qty_pre_caps=int(abs_qty),
                desired_qty_post_caps=int(abs_qty),
                lot_rounded_qty=int(abs_qty),
                min_effective_qty=max(int(cfg.spot_min_qty), int(lot_i)),
                signed_qty_final=int(base_signed_qty),
            )
            return int(base_signed_qty), trace

        if float(entry_price_f) <= 0:
            _update_trace(zero_reason="invalid_entry_price", signed_qty_final=0)
            return 0, trace

        desired_qty = 0

        if str(cfg.sizing_mode) == "notional_pct":
            if float(cfg.spot_notional_pct) > 0 and float(equity_ref_f) > 0:
                desired_qty = int((float(equity_ref_f) * float(cfg.spot_notional_pct)) / float(entry_price_f))
        else:
            stop_level = None
            if stop_price_clean is not None:
                stop_level = float(stop_price_clean)
            elif stop_loss_pct_clean is not None:
                stop_level = cls.stop_level(
                    entry_price=float(entry_price_f),
                    qty=1 if raw_action == "BUY" else -1,
                    stop_loss_pct=float(stop_loss_pct_clean),
                )

            if stop_level is not None and float(cfg.spot_risk_pct) > 0 and float(equity_ref_f) > 0:
                per_share_risk = abs(float(entry_price_f) - float(stop_level))
                risk_dollars = float(equity_ref_f) * float(cfg.spot_risk_pct)
                _update_trace(
                    per_share_risk=float(per_share_risk),
                    risk_dollars_base=float(risk_dollars),
                    risk_dollars_final=float(risk_dollars),
                )

                if raw_action == "BUY":
                    if bool(riskoff):
                        # In `directional` mode we only apply the long factor when the risk direction is explicitly up.
                        # In `hygiene` mode we apply the long factor unconditionally on riskoff days.
                        # (Default factor=1.0 keeps legacy behavior unchanged.)
                        if str(cfg.riskoff_mode) == "directional":
                            if str(risk_dir_clean) == "up":
                                if float(cfg.riskoff_long_factor) == 0:
                                    _update_trace(zero_reason="riskoff_long_factor_zero", signed_qty_final=0)
                                    return 0, trace
                                risk_dollars *= float(cfg.riskoff_long_factor)
                        else:
                            if float(cfg.riskoff_long_factor) == 0:
                                _update_trace(zero_reason="riskoff_long_factor_zero", signed_qty_final=0)
                                return 0, trace
                            risk_dollars *= float(cfg.riskoff_long_factor)

                    if bool(riskpanic):
                        if float(cfg.riskpanic_long_factor) == 0:
                            _update_trace(zero_reason="riskpanic_long_factor_zero", signed_qty_final=0)
                            return 0, trace
                        risk_dollars *= float(cfg.riskpanic_long_factor)
                    elif risk is not None and str(cfg.riskpanic_long_scale_mode) == "linear":
                        tr_delta = getattr(risk, "tr_median_delta_pct", None)
                        gap_ratio = getattr(risk, "neg_gap_ratio", None)
                        if tr_delta is not None and gap_ratio is not None:
                            gap_min = float(cfg.riskpanic_neg_gap_ratio_min)
                            try:
                                gap_ratio_f = float(gap_ratio)
                            except (TypeError, ValueError):
                                gap_ratio_f = 0.0
                            gap_ratio_f = float(max(0.0, min(1.0, gap_ratio_f)))
                            gap_strength = 0.0
                            if gap_min < 1.0:
                                gap_strength = (gap_ratio_f - gap_min) / (1.0 - gap_min)
                            gap_strength = float(max(0.0, min(1.0, gap_strength)))

                            try:
                                delta_f = float(tr_delta)
                            except (TypeError, ValueError):
                                delta_f = 0.0
                            delta_strength = float(
                                max(0.0, min(1.0, max(0.0, delta_f) / float(cfg.riskpanic_long_scale_tr_delta_max_pct)))
                            )

                            strength = float(gap_strength) * float(delta_strength)
                            if strength > 0:
                                base_factor = float(max(0.0, min(1.0, float(cfg.riskpanic_long_factor))))
                                eff = 1.0 - (float(strength) * (1.0 - float(base_factor)))
                                eff = float(max(0.0, min(1.0, eff)))
                                if eff <= 0:
                                    _update_trace(zero_reason="riskpanic_linear_zero", signed_qty_final=0)
                                    return 0, trace
                                risk_dollars *= float(eff)

                    if bool(riskpop):
                        if float(cfg.riskpop_long_factor) == 0:
                            _update_trace(zero_reason="riskpop_long_factor_zero", signed_qty_final=0)
                            return 0, trace
                        risk_dollars *= float(cfg.riskpop_long_factor)

                    if bool(shock) and shock_dir_clean in ("up", "down"):
                        shock_long_mult = (
                            float(cfg.shock_long_risk_mult_factor)
                            if shock_dir_clean == "up"
                            else float(cfg.shock_long_risk_mult_factor_down)
                        )

                        # Optional "rebound-aware" long boost gate: only apply the >1.0 boost
                        # when drawdown is close to the OFF threshold (still shock-on but recovering).
                        if shock_dir_clean == "up" and float(shock_long_mult) > 1.0:
                            boost_ok = True
                            gate_reason = "ok"
                            max_dist = float(cfg.shock_long_boost_max_dist_off_pp)
                            if max_dist > 0:
                                dist_off = shock_drawdown_dist_off_clean
                                if dist_off is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_off_missing"
                                else:
                                    try:
                                        dist_off_f = float(dist_off)
                                    except (TypeError, ValueError):
                                        dist_off_f = None
                                    if dist_off_f is None:
                                        boost_ok = False
                                        gate_reason = "dd_dist_off_invalid"
                                    elif dist_off_f > 0:
                                        boost_ok = False
                                        gate_reason = "dd_dist_off_gt_0"
                                    else:
                                        dist_off_abs = abs(float(dist_off_f))
                                        if dist_off_abs > float(max_dist):
                                            boost_ok = False
                                            gate_reason = f"dd_dist_off_abs_gt_{float(max_dist):g}pp"

                            if boost_ok and bool(cfg.shock_long_boost_require_regime_up) and signal_regime_dir_clean != "up":
                                boost_ok = False
                                gate_reason = "regime_not_up"
                            if boost_ok and bool(cfg.shock_long_boost_require_entry_up) and signal_entry_dir_clean != "up":
                                boost_ok = False
                                gate_reason = "entry_not_up"

                            if boost_ok:
                                _update_trace(
                                    shock_long_factor=float(shock_long_mult),
                                    shock_long_boost_applied=True,
                                    shock_long_boost_gate_reason=gate_reason,
                                )
                                risk_dollars *= float(shock_long_mult)
                            else:
                                _update_trace(
                                    shock_long_factor=1.0,
                                    shock_long_boost_applied=False,
                                    shock_long_boost_gate_reason=gate_reason,
                                )
                        else:
                            _update_trace(shock_long_factor=float(shock_long_mult))
                            if shock_long_mult == 0:
                                _update_trace(zero_reason="shock_long_factor_zero", signed_qty_final=0)
                                return 0, trace
                            risk_dollars *= float(shock_long_mult)

                    risk_dollars *= float(graph_overlay.long_risk_mult)
                    risk_dollars *= float(graph_overlay.risk_mult)
                    if risk_dollars <= 0:
                        _update_trace(zero_reason="graph_overlay_nonpositive", signed_qty_final=0)
                        return 0, trace

                else:
                    short_mult = float(cfg.spot_short_risk_mult)
                    _update_trace(short_mult_base=float(short_mult), short_mult_final=float(short_mult))

                    # Drawdown depth gate for shorts:
                    # Historically we only applied this when the main shock detector was ON and the direction was DOWN.
                    # For HF chop defense, the same drawdown telemetry can be present even when shock stays OFF
                    # (ex: atr_ratio is muted), so we apply the same gate whenever it's configured.
                    #
                    # Semantics: gate around the ON threshold (dd->on distance in percentage points).
                    # - dist_on == 0: at ON threshold (ex: -20% dd if on=-20)
                    # - dist_on < 0: above ON threshold (milder drawdown / near ATH)
                    # - dist_on > 0: below ON threshold (deeper crash)
                    # We allow short entries only when dist_on is within [-max_dist, +max_dist].
                    max_dist = float(cfg.shock_short_entry_max_dist_on_pp)
                    if max_dist > 0:
                        gate_ok = True
                        gate_reason = "ok"
                        dist_on = shock_drawdown_dist_on_clean
                        if dist_on is None:
                            gate_reason = "dd_dist_on_missing"
                        else:
                            try:
                                dist_on_f = float(dist_on)
                            except (TypeError, ValueError):
                                dist_on_f = None
                            if dist_on_f is None:
                                gate_ok = False
                                gate_reason = "dd_dist_on_invalid"
                            elif dist_on_f < -float(max_dist):
                                gate_ok = False
                                gate_reason = f"dd_dist_on_lt_-{float(max_dist):g}pp"
                            elif dist_on_f > float(max_dist):
                                gate_ok = False
                                gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"
                        _update_trace(
                            shock_short_entry_blocked=(not bool(gate_ok)),
                            shock_short_entry_gate_reason=str(gate_reason),
                        )
                        if not bool(gate_ok):
                            _update_trace(zero_reason="shock_short_entry_depth_gate", signed_qty_final=0)
                            return 0, trace

                    if bool(riskoff):
                        # In `directional` mode we only apply the short factor when the risk direction is explicitly down.
                        # In `hygiene` mode we apply the short factor unconditionally on riskoff days.
                        if str(cfg.riskoff_mode) == "directional":
                            if str(risk_dir_clean) == "down":
                                short_mult *= float(cfg.riskoff_short_factor)
                        else:
                            short_mult *= float(cfg.riskoff_short_factor)
                    if bool(riskpanic):
                        short_mult *= float(cfg.riskpanic_short_factor)
                    if bool(riskpop):
                        short_mult *= float(cfg.riskpop_short_factor)
                    if bool(shock) and str(shock_dir_clean) == "down":
                        boost_ok = True
                        gate_reason = "ok"
                        min_streak = max(1, int(cfg.shock_short_boost_min_down_streak_bars))
                        streak = int(shock_dir_down_streak_clean or 0)
                        if streak < min_streak:
                            boost_ok = False
                            gate_reason = f"down_streak_lt_{min_streak}"

                        max_dist = float(cfg.shock_short_boost_max_dist_on_pp)
                        if boost_ok and max_dist > 0:
                            min_dist = float(cfg.shock_short_boost_min_dist_on_pp)
                            dist_on = shock_drawdown_dist_on_clean
                            if dist_on is None:
                                boost_ok = False
                                gate_reason = "dd_dist_on_missing"
                            else:
                                try:
                                    dist_on_f = float(dist_on)
                                except (TypeError, ValueError):
                                    dist_on_f = None
                                if dist_on_f is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_invalid"
                                elif dist_on_f < float(min_dist):
                                    boost_ok = False
                                    gate_reason = (
                                        "dd_dist_on_lt_0" if float(min_dist) <= 0 else f"dd_dist_on_lt_{float(min_dist):g}pp"
                                    )
                                elif dist_on_f > float(max_dist):
                                    boost_ok = False
                                    gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"

                        if boost_ok and bool(cfg.shock_short_boost_require_regime_down) and signal_regime_dir_clean != "down":
                            boost_ok = False
                            gate_reason = "regime_not_down"
                        if boost_ok and bool(cfg.shock_short_boost_require_entry_down) and signal_entry_dir_clean != "down":
                            boost_ok = False
                            gate_reason = "entry_not_down"

                        if boost_ok:
                            shock_short_mult = float(cfg.shock_short_risk_mult_factor)
                            _update_trace(
                                shock_short_factor=float(shock_short_mult),
                                shock_short_boost_applied=True,
                                shock_short_boost_gate_reason=gate_reason,
                            )
                            short_mult *= float(shock_short_mult)
                        else:
                            _update_trace(
                                shock_short_boost_applied=False,
                                shock_short_boost_gate_reason=gate_reason,
                            )
                    elif bool(shock):
                        _update_trace(
                            shock_short_boost_applied=False,
                            shock_short_boost_gate_reason="shock_not_down",
                        )
                    else:
                        dd_boost_applied = False
                        dd_boost_reason = "shock_off"

                        # Drawdown-based short boost (shock-off):
                        # When configured, allow the same short boost to fire off drawdown
                        # telemetry even if the main shock detector (ATR/TR ratio) stays off.
                        #
                        # Guarded by `shock_short_boost_max_dist_on_pp > 0` so existing configs
                        # that only rely on factor != 1.0 do not change behavior.
                        if str(shock_dir_clean) == "down":
                            boost_ok = True
                            gate_reason = "ok"
                            max_dist = float(cfg.shock_short_boost_max_dist_on_pp)
                            if max_dist <= 0:
                                boost_ok = False
                                gate_reason = "dd_boost_disabled"
                            else:
                                min_dist = float(cfg.shock_short_boost_min_dist_on_pp)
                                dist_on = shock_drawdown_dist_on_clean
                                if dist_on is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_missing"
                                else:
                                    try:
                                        dist_on_f = float(dist_on)
                                    except (TypeError, ValueError):
                                        dist_on_f = None
                                    if dist_on_f is None:
                                        boost_ok = False
                                        gate_reason = "dd_dist_on_invalid"
                                    elif dist_on_f < float(min_dist):
                                        boost_ok = False
                                        gate_reason = (
                                            "dd_dist_on_lt_0" if float(min_dist) <= 0 else f"dd_dist_on_lt_{float(min_dist):g}pp"
                                        )
                                    elif dist_on_f > float(max_dist):
                                        boost_ok = False
                                        gate_reason = f"dd_dist_on_gt_{float(max_dist):g}pp"

                            # Optional persistence gates (reuse prearm vel/accel knobs).
                            if boost_ok and float(cfg.shock_prearm_min_dist_on_vel_pp) > 0:
                                if shock_drawdown_dist_on_vel_clean is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_vel_missing"
                                elif float(shock_drawdown_dist_on_vel_clean) < float(cfg.shock_prearm_min_dist_on_vel_pp):
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_vel_low"
                            if boost_ok and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0:
                                if shock_drawdown_dist_on_accel_clean is None:
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_accel_missing"
                                elif float(shock_drawdown_dist_on_accel_clean) < float(cfg.shock_prearm_min_dist_on_accel_pp):
                                    boost_ok = False
                                    gate_reason = "dd_dist_on_accel_low"

                            if boost_ok and bool(cfg.shock_short_boost_require_regime_down) and signal_regime_dir_clean != "down":
                                boost_ok = False
                                gate_reason = "regime_not_down"
                            if boost_ok and bool(cfg.shock_short_boost_require_entry_down) and signal_entry_dir_clean != "down":
                                boost_ok = False
                                gate_reason = "entry_not_down"

                            if boost_ok:
                                shock_short_mult = float(cfg.shock_short_risk_mult_factor)
                                short_mult *= float(shock_short_mult)
                                dd_boost_applied = True
                                dd_boost_reason = str(gate_reason)
                                _update_trace(
                                    shock_short_factor=float(shock_short_mult),
                                    shock_short_boost_applied=True,
                                    shock_short_boost_gate_reason=f"dd:{gate_reason}",
                                )
                            else:
                                dd_boost_reason = str(gate_reason)

                        if not bool(dd_boost_applied):
                            _update_trace(
                                shock_short_boost_applied=False,
                                shock_short_boost_gate_reason=str(dd_boost_reason),
                            )

                        prearm_applied = False
                        prearm_reason = "off"
                        prearm_factor = 1.0
                        dist_on = shock_drawdown_dist_on_clean
                        dist_on_vel = shock_drawdown_dist_on_vel_clean
                        dist_on_accel = shock_drawdown_dist_on_accel_clean
                        prearm_streak = shock_prearm_down_streak_clean
                        near_band = float(cfg.shock_prearm_dist_on_max_pp)
                        latch_enabled = int(cfg.shock_prearm_min_streak_bars) > 0
                        if near_band <= 0:
                            prearm_reason = "disabled"
                        elif dist_on is None:
                            prearm_reason = "dist_on_missing"
                        elif bool(latch_enabled) and float(dist_on) < -float(near_band):
                            prearm_reason = "outside_band"
                        elif (not bool(latch_enabled)) and not (-float(near_band) <= float(dist_on) < 0.0):
                            prearm_reason = "outside_band"
                        elif (not bool(latch_enabled)) and dist_on_vel is None:
                            prearm_reason = "dist_vel_missing"
                        elif (not bool(latch_enabled)) and float(dist_on_vel) < float(cfg.shock_prearm_min_dist_on_vel_pp):
                            prearm_reason = "dist_vel_low"
                        elif (not bool(latch_enabled)) and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0 and dist_on_accel is None:
                            prearm_reason = "dist_accel_missing"
                        elif (not bool(latch_enabled)) and float(cfg.shock_prearm_min_dist_on_accel_pp) > 0 and float(dist_on_accel) < float(
                            cfg.shock_prearm_min_dist_on_accel_pp
                        ):
                            prearm_reason = "dist_accel_low"
                        elif int(cfg.shock_prearm_min_streak_bars) > 0 and prearm_streak is None:
                            prearm_reason = "streak_missing"
                        elif int(cfg.shock_prearm_min_streak_bars) > 0 and int(prearm_streak) < int(
                            cfg.shock_prearm_min_streak_bars
                        ):
                            prearm_reason = f"streak_lt_{int(cfg.shock_prearm_min_streak_bars)}"
                        elif bool(cfg.shock_prearm_require_regime_down) and signal_regime_dir_clean != "down":
                            prearm_reason = "regime_not_down"
                        elif bool(cfg.shock_prearm_require_entry_down) and signal_entry_dir_clean != "down":
                            prearm_reason = "entry_not_down"
                        else:
                            prearm_factor = float(cfg.shock_prearm_short_risk_mult_factor)
                            if prearm_factor > 0:
                                short_mult *= float(prearm_factor)
                                prearm_applied = True
                                prearm_reason = "ok"
                            else:
                                prearm_reason = "factor_nonpositive"
                        _update_trace(
                            shock_prearm_applied=bool(prearm_applied),
                            shock_prearm_factor=float(prearm_factor) if prearm_applied else None,
                            shock_prearm_reason=str(prearm_reason),
                        )
                    short_mult *= float(graph_overlay.short_risk_mult)
                    _update_trace(short_mult_final=float(short_mult))
                    if short_mult <= 0:
                        _update_trace(zero_reason="short_mult_nonpositive", signed_qty_final=0)
                        return 0, trace
                    risk_dollars *= float(short_mult)
                    risk_dollars *= float(graph_overlay.risk_mult)
                    if risk_dollars <= 0:
                        _update_trace(zero_reason="graph_overlay_nonpositive", signed_qty_final=0)
                        return 0, trace

                if (
                    cfg.shock_risk_scale_target_atr_pct is not None
                    and shock_atr_pct_clean is not None
                    and float(shock_atr_pct_clean) > 0
                ):
                    target = float(cfg.shock_risk_scale_target_atr_pct)
                    min_mult = float(cfg.shock_risk_scale_min_mult)
                    scale = min(1.0, float(target) / float(shock_atr_pct_clean))
                    scale = float(max(min_mult, min(1.0, scale)))
                    apply_to = str(cfg.shock_risk_scale_apply_to)
                    _update_trace(
                        shock_scale_target_atr_pct=float(target),
                        shock_scale_mult=float(scale),
                        shock_scale_apply_to=str(apply_to),
                    )
                    if apply_to in ("risk", "both"):
                        risk_dollars *= float(scale)
                    if apply_to in ("cap", "both"):
                        cap_pct *= float(scale)

                _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                liq_boost_applied = False
                liq_boost_reason = "off"
                liq_boost_mult = 1.0
                liq_boost_score = None
                if bool(cfg.liq_boost_enable):
                    overlay_trace = graph_overlay.as_payload() if hasattr(graph_overlay, "as_payload") else {}
                    raw_score = overlay_trace.get("trace", {}).get("score") if isinstance(overlay_trace, dict) else None
                    if raw_score is None and isinstance(trace.graph_overlay_trace, dict):
                        raw_score = trace.graph_overlay_trace.get("trace", {}).get("score")
                    try:
                        liq_boost_score = float(raw_score) if raw_score is not None else None
                    except (TypeError, ValueError):
                        liq_boost_score = None
                    if liq_boost_score is None:
                        # Fallback score keeps boost logic usable with non-trend-bias overlay policies.
                        score = 0.0
                        if risk_tr_ratio is not None:
                            score += max(0.0, float(risk_tr_ratio) - 1.0)
                        if risk_slope_med is not None:
                            slope_signed = float(risk_slope_med) if raw_action == "BUY" else -float(risk_slope_med)
                            score += max(0.0, slope_signed / 0.35)
                        if risk_slope_vel is not None:
                            vel_signed = float(risk_slope_vel) if raw_action == "BUY" else -float(risk_slope_vel)
                            score += max(0.0, vel_signed / 0.20)
                        liq_boost_score = float(score)
                    action_dir = "up" if raw_action == "BUY" else "down"
                    align_ok = True
                    if bool(cfg.liq_boost_require_alignment):
                        align_ok = (
                            signal_entry_dir_clean == action_dir and signal_regime_dir_clean == action_dir
                        )
                    if bool(cfg.liq_boost_require_shock) and not bool(shock):
                        align_ok = False
                        liq_boost_reason = "shock_required"
                    elif not align_ok:
                        liq_boost_reason = "align_fail"
                    elif liq_boost_score is None:
                        liq_boost_reason = "score_missing"
                    elif liq_boost_score < float(cfg.liq_boost_score_min):
                        liq_boost_reason = "score_low"
                    else:
                        score_over = float(liq_boost_score) - float(cfg.liq_boost_score_min)
                        intensity = min(1.0, max(0.0, score_over / float(cfg.liq_boost_score_span)))
                        liq_boost_mult = 1.0 + (float(cfg.liq_boost_max_risk_mult) - 1.0) * float(intensity)
                        if liq_boost_mult > 1.0:
                            risk_dollars *= float(liq_boost_mult)
                            liq_boost_applied = True
                            liq_boost_reason = "ok"
                        else:
                            liq_boost_reason = "mult_unity"
                    _update_trace(
                        liq_boost_applied=bool(liq_boost_applied),
                        liq_boost_score=float(liq_boost_score) if liq_boost_score is not None else None,
                        liq_boost_mult=float(liq_boost_mult),
                        liq_boost_reason=str(liq_boost_reason),
                        liq_boost_cap_floor_frac=float(cfg.liq_boost_cap_floor_frac),
                    )
                    if liq_boost_applied:
                        _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                if shock_ramp is not None:
                    ramp_applied = False
                    ramp_reason = "off"
                    ramp_phase = None
                    ramp_dir = "up" if raw_action == "BUY" else "down"
                    ramp_intensity = None
                    ramp_mult = 1.0
                    ramp_floor_frac = 0.0
                    ramp_node = None
                    if isinstance(shock_ramp, Mapping):
                        ramp_node = shock_ramp.get(str(ramp_dir))
                    else:
                        ramp_node = getattr(shock_ramp, str(ramp_dir), None)
                    if isinstance(ramp_node, Mapping):
                        raw_mult = ramp_node.get("risk_mult")
                        raw_floor = ramp_node.get("cap_floor_frac")
                        try:
                            ramp_mult = float(raw_mult) if raw_mult is not None else 1.0
                        except (TypeError, ValueError):
                            ramp_mult = 1.0
                        try:
                            ramp_floor_frac = float(raw_floor) if raw_floor is not None else 0.0
                        except (TypeError, ValueError):
                            ramp_floor_frac = 0.0
                        ramp_floor_frac = float(max(0.0, min(1.0, ramp_floor_frac)))
                        ramp_phase_raw = ramp_node.get("phase")
                        ramp_phase = str(ramp_phase_raw) if ramp_phase_raw is not None else None
                        try:
                            ramp_intensity = float(ramp_node.get("intensity")) if ramp_node.get("intensity") is not None else None
                        except (TypeError, ValueError):
                            ramp_intensity = None
                        ramp_reason_raw = ramp_node.get("reason")
                        ramp_reason = str(ramp_reason_raw) if ramp_reason_raw is not None else "ok"
                        if ramp_mult > 1.0 and risk_dollars > 0:
                            risk_dollars *= float(ramp_mult)
                            ramp_applied = True
                        else:
                            ramp_reason = "mult_unity"
                    elif ramp_node is not None:
                        ramp_reason = "invalid_node"
                    else:
                        ramp_reason = "missing_dir"
                    _update_trace(
                        shock_ramp_applied=bool(ramp_applied),
                        shock_ramp_dir=str(ramp_dir),
                        shock_ramp_phase=str(ramp_phase) if ramp_phase is not None else None,
                        shock_ramp_intensity=float(ramp_intensity) if ramp_intensity is not None else None,
                        shock_ramp_risk_mult=float(ramp_mult),
                        shock_ramp_cap_floor_frac=float(ramp_floor_frac),
                        shock_ramp_reason=str(ramp_reason),
                    )
                    if ramp_applied:
                        _update_trace(risk_dollars_final=float(risk_dollars), cap_pct_final=float(cap_pct))

                if per_share_risk > 1e-9 and risk_dollars > 0:
                    desired_qty = int(risk_dollars / per_share_risk)

        fallback_to_lot = False
        if desired_qty <= 0:
            desired_qty = int(lot_i) * int(cfg.quantity_mult)
            fallback_to_lot = True

        desired_qty_pre_caps = int(desired_qty)
        cap_qty = None
        if cap_pct > 0 and float(equity_ref_f) > 0:
            cap_qty = int((float(equity_ref_f) * float(cap_pct)) / float(entry_price_f))
            desired_qty = min(int(desired_qty), max(0, int(cap_qty)))
            if (
                bool(getattr(trace, "liq_boost_applied", False))
                and float(getattr(cfg, "liq_boost_cap_floor_frac", 0.0)) > 0
                and cap_qty is not None
                and cap_qty > 0
            ):
                floor_qty = int(float(cap_qty) * float(cfg.liq_boost_cap_floor_frac))
                if floor_qty > 0:
                    desired_qty = max(int(desired_qty), int(floor_qty))
                    _update_trace(liq_boost_cap_floor_qty=int(floor_qty))
            if (
                float(getattr(trace, "shock_ramp_cap_floor_frac", 0.0) or 0.0) > 0
                and cap_qty is not None
                and cap_qty > 0
            ):
                floor_qty = int(float(cap_qty) * float(getattr(trace, "shock_ramp_cap_floor_frac", 0.0) or 0.0))
                if floor_qty > 0:
                    desired_qty = max(int(desired_qty), int(floor_qty))
                    _update_trace(shock_ramp_cap_floor_qty=int(floor_qty))

        afford_qty = None
        if raw_action == "BUY" and cash_ref_f is not None and float(cash_ref_f) > 0:
            afford_qty = int(float(cash_ref_f) / float(entry_price_f))
            desired_qty = min(int(desired_qty), max(0, int(afford_qty)))

        if int(cfg.spot_max_qty) > 0:
            desired_qty = min(int(desired_qty), int(cfg.spot_max_qty))

        desired_qty_post_caps = int(desired_qty)
        lot_rounded_qty = (int(desired_qty) // int(lot_i)) * int(lot_i)
        min_effective = max(int(cfg.spot_min_qty), int(lot_i))

        if lot_rounded_qty < min_effective:
            _update_trace(
                desired_qty_pre_caps=int(desired_qty_pre_caps),
                desired_qty_post_caps=int(desired_qty_post_caps),
                cap_qty=int(cap_qty) if cap_qty is not None else None,
                afford_qty=int(afford_qty) if afford_qty is not None else None,
                fallback_to_lot=bool(fallback_to_lot),
                lot_rounded_qty=int(lot_rounded_qty),
                min_effective_qty=int(min_effective),
                min_qty_blocked=True,
                signed_qty_final=0,
                zero_reason="min_qty_gate",
                cap_pct_final=float(cap_pct),
            )
            return 0, trace

        signed_qty = int(lot_rounded_qty) if raw_action == "BUY" else -int(lot_rounded_qty)
        _update_trace(
            desired_qty_pre_caps=int(desired_qty_pre_caps),
            desired_qty_post_caps=int(desired_qty_post_caps),
            cap_qty=int(cap_qty) if cap_qty is not None else None,
            afford_qty=int(afford_qty) if afford_qty is not None else None,
            fallback_to_lot=bool(fallback_to_lot),
            lot_rounded_qty=int(lot_rounded_qty),
            min_effective_qty=int(min_effective),
            min_qty_blocked=False,
            signed_qty_final=int(signed_qty),
            zero_reason=None,
            cap_pct_final=float(cap_pct),
        )
        return int(signed_qty), trace

    @classmethod
    def calc_signed_qty(
        cls,
        *,
        strategy: Mapping[str, object] | object,
        filters: Mapping[str, object] | object | None,
        action: str,
        lot: int,
        entry_price: float,
        stop_price: float | None,
        stop_loss_pct: float | None,
        shock: bool | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        shock_dir_down_streak_bars: int | None = None,
        shock_drawdown_dist_on_pct: float | None = None,
        shock_drawdown_dist_on_vel_pp: float | None = None,
        shock_drawdown_dist_on_accel_pp: float | None = None,
        shock_prearm_down_streak_bars: int | None = None,
        shock_ramp: Mapping[str, object] | object | None = None,
        riskoff: bool = False,
        risk_dir: str | None = None,
        riskpanic: bool = False,
        riskpop: bool = False,
        risk: object | None = None,
        signal_entry_dir: str | None = None,
        signal_regime_dir: str | None = None,
        equity_ref: float = 0.0,
        cash_ref: float | None = None,
    ) -> int:
        qty, _trace = cls.calc_signed_qty_with_trace(
            strategy=strategy,
            filters=filters,
            action=action,
            lot=lot,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_loss_pct=stop_loss_pct,
            shock=shock,
            shock_dir=shock_dir,
            shock_atr_pct=shock_atr_pct,
            shock_dir_down_streak_bars=shock_dir_down_streak_bars,
            shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct,
            shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
            shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp,
            shock_prearm_down_streak_bars=shock_prearm_down_streak_bars,
            shock_ramp=shock_ramp,
            riskoff=riskoff,
            risk_dir=risk_dir,
            riskpanic=riskpanic,
            riskpop=riskpop,
            risk=risk,
            signal_entry_dir=signal_entry_dir,
            signal_regime_dir=signal_regime_dir,
            equity_ref=equity_ref,
            cash_ref=cash_ref,
        )
        return int(qty)
