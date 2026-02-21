import unittest
from datetime import date, datetime
from types import SimpleNamespace

from tradebot.spot.policy import SpotPolicy, SpotPolicyConfigView


class SpotPolicyKernelTests(unittest.TestCase):
    def test_resolve_entry_action_qty_directional_and_fallback(self) -> None:
        strategy = {
            "directional_spot": {
                "up": {"action": "BUY", "qty": 2},
                "down": {"action": "SELL", "qty": 3},
            }
        }
        self.assertEqual(
            SpotPolicy.resolve_entry_action_qty(
                strategy=strategy,
                entry_dir="up",
                needs_direction=True,
                fallback_short_sell=False,
            ),
            ("BUY", 2),
        )
        self.assertEqual(
            SpotPolicy.resolve_entry_action_qty(
                strategy={},
                entry_dir="down",
                needs_direction=False,
                fallback_short_sell=True,
            ),
            ("SELL", 1),
        )
        self.assertIsNone(
            SpotPolicy.resolve_entry_action_qty(
                strategy={},
                entry_dir="down",
                needs_direction=True,
                fallback_short_sell=False,
            )
        )

    def test_pending_entry_cancel_uses_utc_mode_by_default(self) -> None:
        # 00:30 UTC is previous ET trade day; pending date should still match Jan 15.
        should_cancel = SpotPolicy.pending_entry_should_cancel(
            pending_dir="up",
            pending_set_date=date(2025, 1, 15),
            exec_ts=datetime(2025, 1, 16, 0, 30),
            risk_overlay_enabled=True,
            riskoff_today=True,
            riskpanic_today=False,
            riskpop_today=False,
            riskoff_mode="hygiene",
            shock_dir_now=None,
            riskoff_end_hour=None,
        )
        self.assertFalse(should_cancel)

    def test_pending_entry_cancel_supports_et_naive_mode(self) -> None:
        # In ET naive mode, this is Jan 16 ET and should cancel against Jan 15 pending date.
        should_cancel = SpotPolicy.pending_entry_should_cancel(
            pending_dir="up",
            pending_set_date=date(2025, 1, 15),
            exec_ts=datetime(2025, 1, 16, 0, 30),
            risk_overlay_enabled=True,
            riskoff_today=True,
            riskpanic_today=False,
            riskpop_today=False,
            riskoff_mode="hygiene",
            shock_dir_now=None,
            riskoff_end_hour=None,
            naive_ts_mode="et",
        )
        self.assertTrue(should_cancel)

    def test_risk_entry_cutoff_hour_parsing(self) -> None:
        self.assertEqual(SpotPolicy.risk_entry_cutoff_hour_et({"risk_entry_cutoff_hour_et": "15"}), 15)
        self.assertEqual(SpotPolicy.risk_entry_cutoff_hour_et({"entry_end_hour_et": "14"}), 14)
        self.assertEqual(SpotPolicy.risk_entry_cutoff_hour_et({"entry_end_hour": "13"}), 13)
        self.assertIsNone(SpotPolicy.risk_entry_cutoff_hour_et({"entry_start_hour_et": 9, "entry_end_hour_et": 16}))

    def test_apply_branch_size_mult_clamps_min_and_max(self) -> None:
        self.assertEqual(
            SpotPolicy.apply_branch_size_mult(
                signed_qty=3,
                size_mult=0.5,
                spot_min_qty=1,
                spot_max_qty=0,
            ),
            1,
        )
        self.assertEqual(
            SpotPolicy.apply_branch_size_mult(
                signed_qty=3,
                size_mult=2.0,
                spot_min_qty=1,
                spot_max_qty=5,
            ),
            5,
        )
        self.assertEqual(
            SpotPolicy.apply_branch_size_mult(
                signed_qty=-2,
                size_mult=2.0,
                spot_min_qty=1,
                spot_max_qty=0,
            ),
            -4,
        )

    def test_policy_config_view_sanitizes_defaults(self) -> None:
        cfg = SpotPolicyConfigView.from_sources(
            strategy={
                "spot_sizing_mode": "unknown_mode",
                "spot_branch_a_size_mult": 0,
            },
            filters={
                "riskoff_mode": "invalid",
                "riskpanic_long_scale_mode": "lin",
                "riskpanic_tr5_med_delta_min_pct": 0,
                "shock_risk_scale_apply_to": "cap-only",
                "shock_stop_loss_pct_mult": 0,
            },
        )
        self.assertEqual(cfg.sizing_mode, "fixed")
        self.assertEqual(cfg.riskoff_mode, "hygiene")
        self.assertEqual(cfg.riskpanic_long_scale_mode, "linear")
        self.assertEqual(cfg.riskpanic_long_scale_tr_delta_max_pct, 1.0)
        self.assertEqual(cfg.shock_risk_scale_apply_to, "cap")
        self.assertEqual(cfg.shock_stop_loss_pct_mult, 1.0)
        self.assertEqual(cfg.spot_branch_a_size_mult, 1.0)

    def test_calc_signed_qty_with_trace_captures_risk_pipeline(self) -> None:
        qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.01,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters={
                "riskoff_mode": "directional",
                "riskoff_long_risk_mult_factor": 0.5,
                "shock_risk_scale_target_atr_pct": 10.0,
                "shock_risk_scale_min_mult": 0.2,
                "shock_risk_scale_apply_to": "both",
            },
            action="BUY",
            lot=1,
            entry_price=100.0,
            stop_price=99.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="up",
            shock_atr_pct=20.0,
            riskoff=True,
            risk_dir="up",
            riskpanic=False,
            riskpop=False,
            risk=None,
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )

        self.assertEqual(qty, 250)
        self.assertAlmostEqual(float(trace.risk_dollars_base or 0.0), 1000.0, places=6)
        self.assertAlmostEqual(float(trace.risk_dollars_final or 0.0), 250.0, places=6)
        self.assertEqual(trace.shock_scale_apply_to, "both")
        self.assertEqual(trace.signed_qty_final, 250)

    def test_short_shock_boost_respects_down_streak_gate(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.01,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters={
                "shock_short_risk_mult_factor": 3.0,
                "shock_short_boost_min_down_streak_bars": 4,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="down",
            shock_atr_pct=12.0,
            shock_dir_down_streak_bars=2,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertFalse(bool(trace.shock_short_boost_applied))
        self.assertEqual(trace.shock_short_boost_gate_reason, "down_streak_lt_4")
        self.assertAlmostEqual(float(trace.short_mult_final or 0.0), 1.0, places=6)

    def test_short_shock_boost_requires_regime_and_entry_when_enabled(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.01,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters={
                "shock_short_risk_mult_factor": 2.5,
                "shock_short_boost_min_down_streak_bars": 2,
                "shock_short_boost_require_regime_down": True,
                "shock_short_boost_require_entry_down": True,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="down",
            shock_atr_pct=12.0,
            shock_dir_down_streak_bars=3,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertTrue(bool(trace.shock_short_boost_applied))
        self.assertEqual(trace.shock_short_boost_gate_reason, "ok")
        self.assertAlmostEqual(float(trace.short_mult_final or 0.0), 2.5, places=6)

    def test_short_shock_boost_respects_drawdown_depth_gate(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.01,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters={
                "shock_short_risk_mult_factor": 2.5,
                "shock_short_boost_min_down_streak_bars": 1,
                "shock_short_boost_max_dist_on_pp": 2.0,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="down",
            shock_atr_pct=12.0,
            shock_dir_down_streak_bars=1,
            shock_drawdown_dist_on_pct=3.0,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertFalse(bool(trace.shock_short_boost_applied))
        self.assertEqual(trace.shock_short_boost_gate_reason, "dd_dist_on_gt_2pp")
        self.assertAlmostEqual(float(trace.short_mult_final or 0.0), 1.0, places=6)

    def test_short_prearm_applies_before_shock_when_dd_is_near_and_accelerating(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.01,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters={
                "shock_prearm_dist_on_max_pp": 2.0,
                "shock_prearm_min_dist_on_vel_pp": 0.2,
                "shock_prearm_short_risk_mult_factor": 2.0,
                "shock_prearm_require_regime_down": True,
                "shock_prearm_require_entry_down": True,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=False,
            shock_dir="down",
            shock_atr_pct=8.0,
            shock_drawdown_dist_on_pct=-0.7,
            shock_drawdown_dist_on_vel_pp=0.4,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertTrue(bool(trace.shock_prearm_applied))
        self.assertEqual(trace.shock_prearm_reason, "ok")
        self.assertAlmostEqual(float(trace.short_mult_final or 0.0), 2.0, places=6)

    def test_liq_boost_can_floor_toward_cap_when_confidence_is_high(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.002,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
                "spot_risk_overlay_policy": "trend_bias",
            },
            filters={
                "liq_boost_enable": True,
                "liq_boost_score_min": 0.2,
                "liq_boost_score_span": 0.8,
                "liq_boost_max_risk_mult": 4.0,
                "liq_boost_cap_floor_frac": 0.6,
                "liq_boost_require_alignment": True,
            },
            action="BUY",
            lot=1,
            entry_price=100.0,
            stop_price=99.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="up",
            shock_atr_pct=8.0,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=SimpleNamespace(
                tr_ratio=2.0,
                tr_median_delta_pct=1.0,
                tr_slope_vel_pct=1.0,
            ),
            signal_entry_dir="up",
            signal_regime_dir="up",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertTrue(bool(trace.liq_boost_applied))
        self.assertEqual(trace.liq_boost_reason, "ok")
        self.assertEqual(trace.cap_qty, 1000)
        self.assertEqual(trace.liq_boost_cap_floor_qty, 600)
        self.assertGreaterEqual(int(trace.signed_qty_final), 600)

    def test_shock_ramp_mult_scales_qty_when_enabled(self) -> None:
        base_qty, base_trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.002,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters=None,
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=False,
            shock_dir="down",
            shock_atr_pct=8.0,
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.002,
                "spot_short_risk_mult": 1.0,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters=None,
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=False,
            shock_dir="down",
            shock_atr_pct=8.0,
            shock_ramp={
                "down": {
                    "risk_mult": 3.0,
                    "cap_floor_frac": 0.0,
                    "phase": "approach",
                    "intensity": 0.7,
                    "reason": "dd",
                },
                "up": {"risk_mult": 1.0, "cap_floor_frac": 0.0, "phase": "off", "intensity": 0.0},
            },
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="down",
            signal_regime_dir="down",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertIsNone(base_trace.shock_ramp_applied)
        self.assertTrue(bool(trace.shock_ramp_applied))
        self.assertAlmostEqual(float(trace.shock_ramp_risk_mult or 0.0), 3.0, places=6)
        self.assertGreater(abs(int(qty)), abs(int(base_qty)))

    def test_shock_ramp_can_floor_toward_cap(self) -> None:
        _qty, trace = SpotPolicy.calc_signed_qty_with_trace(
            strategy={
                "quantity": 1,
                "spot_sizing_mode": "risk_pct",
                "spot_risk_pct": 0.0004,
                "spot_max_notional_pct": 1.0,
                "spot_min_qty": 1,
                "spot_max_qty": 0,
            },
            filters=None,
            action="BUY",
            lot=1,
            entry_price=100.0,
            stop_price=99.0,
            stop_loss_pct=None,
            shock=False,
            shock_dir="up",
            shock_atr_pct=8.0,
            shock_ramp={
                "down": {"risk_mult": 1.0, "cap_floor_frac": 0.0, "phase": "off", "intensity": 0.0},
                "up": {
                    "risk_mult": 1.0,
                    "cap_floor_frac": 0.60,
                    "phase": "trend",
                    "intensity": 1.0,
                    "reason": "slope",
                },
            },
            riskoff=False,
            risk_dir=None,
            riskpanic=False,
            riskpop=False,
            risk=None,
            signal_entry_dir="up",
            signal_regime_dir="up",
            equity_ref=100_000.0,
            cash_ref=100_000.0,
        )
        self.assertEqual(trace.cap_qty, 1000)
        self.assertEqual(trace.shock_ramp_cap_floor_qty, 600)
        self.assertGreaterEqual(int(trace.signed_qty_final), 600)

    def test_resolve_position_intent_enter_from_flat(self) -> None:
        decision = SpotPolicy.resolve_position_intent(
            strategy={"spot_resize_mode": "target"},
            current_qty=0,
            target_qty=12,
        )
        self.assertEqual(decision.intent, "enter")
        self.assertEqual(decision.order_action, "BUY")
        self.assertEqual(decision.order_qty, 12)
        self.assertEqual(decision.delta_qty, 12)

    def test_resolve_position_intent_resize_and_clamp(self) -> None:
        decision = SpotPolicy.resolve_position_intent(
            strategy={
                "spot_resize_mode": "target",
                "spot_resize_max_step_qty": 3,
            },
            current_qty=5,
            target_qty=12,
        )
        self.assertEqual(decision.intent, "resize")
        self.assertEqual(decision.resize_kind, "scale_in")
        self.assertTrue(decision.clamped)
        self.assertEqual(decision.order_action, "BUY")
        self.assertEqual(decision.order_qty, 3)
        self.assertEqual(decision.delta_qty, 3)

    def test_resolve_position_intent_holds_when_resize_off(self) -> None:
        decision = SpotPolicy.resolve_position_intent(
            strategy={"spot_resize_mode": "off"},
            current_qty=5,
            target_qty=7,
        )
        self.assertEqual(decision.intent, "hold")
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "resize_mode_off")

    def test_resolve_position_intent_min_delta_gate(self) -> None:
        decision = SpotPolicy.resolve_position_intent(
            strategy={
                "spot_resize_mode": "target",
                "spot_resize_min_delta_qty": 4,
            },
            current_qty=10,
            target_qty=12,
        )
        self.assertEqual(decision.intent, "hold")
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.reason, "min_delta_gate")


if __name__ == "__main__":
    unittest.main()
