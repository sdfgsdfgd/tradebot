import unittest
from datetime import date, datetime
from types import SimpleNamespace

from tradebot.spot.evaluator_common import (
    SpotEntryCandidate,
    SpotEntryGateContext,
    SpotRegimeState,
    SpotSignalSelection,
)
from tradebot.spot.evaluator_policy import SpotSignalPolicyMixin
from tradebot.spot.evaluator_regime import SpotSignalRegimeMixin
from tradebot.spot.evaluator_setup import _regime_gate_policy
from tradebot.spot.gates import flip_exit_allowed
from tradebot.spot.policy import SpotPolicy, SpotPolicyConfigView


class _GateRecorder(SpotSignalPolicyMixin):
    def __init__(self, blocked_gate: str) -> None:
        self.blocked_gate = blocked_gate
        self.visited: list[str] = []

    def _record(self, name: str) -> bool:
        self.visited.append(name)
        return name == self.blocked_gate

    def _crash_gate_blocks(self, *_args) -> bool:
        return self._record("crash")

    def _router_damage_gate_blocks(self, *_args) -> bool:
        return self._record("router_damage")

    def _crash_prearm_gate_blocks(self, *_args) -> bool:
        return self._record("crash_prearm")

    def _branch_b_regime_gate_blocks(self, *_args) -> bool:
        return self._record("branch_b_regime")

    def _branch_a_upcorridor_gate_blocks(self, *_args) -> bool:
        return self._record("branch_a_upcorridor")

    def _continuation_confidence_gate_blocks(self, *_args) -> bool:
        return self._record("continuation_confidence")


class _ConfirmationRegimeHarness(SpotSignalRegimeMixin):
    _supertrend2_engine = None
    _regime2_engine = None
    _regime2_bear_entry_mode = "off"
    _dual_branch_enabled = True


class SpotPolicyKernelTests(unittest.TestCase):
    def test_canonical_regime_state_owns_legacy_snapshot_names(self) -> None:
        canonical = SpotRegimeState(
            label="transition_up_hot",
            owner="clean",
            transition_hot=True,
            fast_dir="up",
            fast_ready=True,
            hard_dir="down",
            hard_ready=True,
            hard_release_age_bars=3,
        )
        snapshot = SimpleNamespace(
            regime=canonical,
            regime4_state="stale_legacy_value",
            regime2_dir="down",
        )

        self.assertIs(SpotRegimeState.from_snapshot(snapshot), canonical)
        self.assertEqual(
            SpotRegimeState.from_snapshot(
                SimpleNamespace(
                    regime4_state="trend_down",
                    regime4_owner="primary",
                    regime2_dir="down",
                    regime2_ready=True,
                )
            ),
            SpotRegimeState(
                label="trend_down",
                owner="primary",
                fast_dir="down",
                fast_ready=True,
            ),
        )

    def test_legacy_regime_keys_decode_into_semantic_gate_policy(self) -> None:
        policy = _regime_gate_policy(
            {
                "regime2_crash_atr_pct_min": -1,
                "regime2_crash_prearm_apply_to": "invalid",
                "regime2_repair_branch_b_long_block_after_hour_et": 99,
                "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_min": 3,
                "regime2_upcorridor_branch_a_long_mid_shock_atr_pct_max": 2,
                "regime2_trenddown_branch_b_long_hard_up_ddv_min_pp": -2,
                "regime2_trenddown_branch_b_long_hard_up_ddv_max_pp": -3,
            }
        )

        self.assertIsNone(policy.crash_atr_min)
        self.assertEqual(policy.crash_prearm_scope, "off")
        self.assertEqual(policy.repair_branch_b_after_hour, 23)
        self.assertEqual(policy.upcorridor_branch_a_mid_atr.minimum, 3)
        self.assertEqual(policy.upcorridor_branch_a_mid_atr.maximum, 3)
        self.assertEqual(policy.trenddown_branch_b_ddv.minimum, -2)
        self.assertEqual(policy.trenddown_branch_b_ddv.maximum, -2)

    def test_entry_gate_pipeline_is_ordered_and_short_circuits(self) -> None:
        evaluator = _GateRecorder("branch_b_regime")
        result = evaluator._apply_entry_gates(
            SpotEntryCandidate("up", "b"),
            SpotEntryGateContext(
                bar_ts=datetime(2026, 7, 22),
                regime=SpotRegimeState(label="trend_down"),
            ),
        )

        self.assertEqual(
            evaluator.visited,
            ["crash", "router_damage", "crash_prearm", "branch_b_regime"],
        )
        self.assertEqual(result, SpotEntryCandidate(None, None, "branch_b_regime"))

    def test_confirmation_regime_preserves_only_aligned_branch_selection(self) -> None:
        signal = SimpleNamespace(entry_dir="up")
        harness = _ConfirmationRegimeHarness()
        bar = SimpleNamespace(ts=datetime(2026, 7, 22), open=1, high=1, low=1, close=1, volume=1)

        aligned = harness._apply_confirmation_regime(
            selection=SpotSignalSelection(signal, SpotEntryCandidate("up", "a"), "a"),
            bar=bar,
            close=1.0,
            regime=SpotRegimeState(),
        )
        blocked = harness._apply_confirmation_regime(
            selection=SpotSignalSelection(signal, SpotEntryCandidate(None, "a"), "a"),
            bar=bar,
            close=1.0,
            regime=SpotRegimeState(),
        )

        self.assertEqual(aligned.candidate, SpotEntryCandidate("up", "a"))
        self.assertEqual(blocked.candidate, SpotEntryCandidate(None))

    def test_flip_exit_allowed_enforces_canonical_hold_boundary(self) -> None:
        strategy = {
            "direction_source": "ema",
            "exit_on_signal_flip": True,
            "ema_entry_mode": "trend",
            "flip_exit_mode": "state",
            "flip_exit_min_hold_bars": 6,
        }
        signal = SimpleNamespace(
            ema_ready=True,
            ema_fast=99.0,
            ema_slow=100.0,
            state="down",
            cross_up=False,
            cross_down=False,
        )
        entry = datetime(2026, 7, 22, 10, 0)
        self.assertFalse(
            flip_exit_allowed(
                strategy=strategy,
                open_dir="up",
                entry_time=entry,
                current_time=datetime(2026, 7, 22, 10, 25),
                bar_size="5 mins",
                signal=signal,
            )
        )
        self.assertTrue(
            flip_exit_allowed(
                strategy=strategy,
                open_dir="up",
                entry_time=entry,
                current_time=datetime(2026, 7, 22, 10, 30),
                bar_size="5 mins",
                signal=signal,
            )
        )

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

    def test_policy_sources_share_explicit_pack_and_default_precedence(self) -> None:
        from tradebot.spot.policy import SpotRuntimeSpec

        packed = SpotPolicyConfigView.from_sources(
            strategy={"spot_policy_pack": "defensive"},
        )
        self.assertEqual(packed.spot_policy_pack, "defensive")
        self.assertEqual(packed.spot_resize_mode, "target")
        self.assertFalse(packed.spot_resize_allow_scale_in)
        self.assertEqual(packed.riskoff_mode, "directional")

        explicit = SpotPolicyConfigView.from_sources(
            strategy={
                "spot_policy_pack": "defensive",
                "spot_resize_mode": None,
            },
            filters={"riskoff_mode": None},
        )
        self.assertEqual(explicit.spot_resize_mode, "off")
        self.assertEqual(explicit.riskoff_mode, "hygiene")

        filter_selected = SpotPolicyConfigView.from_sources(
            strategy={},
            filters={"spot_policy_pack": "defensive"},
        )
        self.assertEqual(filter_selected.spot_policy_pack, "defensive")
        self.assertEqual(filter_selected.spot_resize_min_delta_qty, 2)

        runtime = SpotRuntimeSpec.from_sources(
            strategy={
                "spot_policy_pack": "defensive",
                "spot_spread": "malformed",
                "spot_close_eod": "yes",
            },
        )
        self.assertEqual(runtime.spot_policy_pack, "defensive")
        self.assertEqual(runtime.spread, 0.0)
        self.assertTrue(runtime.close_eod)
        self.assertIsNone(
            SpotRuntimeSpec.from_sources(
                strategy={"spot_policy_pack": "unknown"},
            ).spot_policy_pack
        )

    def test_engine_policy_views_use_typed_factories(self) -> None:
        from tradebot.engine import spot_policy_config_view, spot_riskoff_end_hour, spot_runtime_spec_view
        from tradebot.spot.policy import SpotRuntimeSpec

        strategy = {"spot_spread": "0.25", "spot_resize_cooldown_bars": "3"}
        filters = {"risk_entry_cutoff_hour_et": "15"}
        self.assertEqual(
            spot_policy_config_view(strategy=strategy, filters=filters),
            SpotPolicyConfigView.from_sources(strategy=strategy, filters=filters),
        )
        self.assertEqual(
            spot_runtime_spec_view(strategy=strategy, filters=filters),
            SpotRuntimeSpec.from_sources(strategy=strategy, filters=filters),
        )
        self.assertEqual(spot_riskoff_end_hour(filters), 15)

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

    def test_short_entry_depth_gate_blocks_late_shock_down_entries(self) -> None:
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
                "shock_short_entry_max_dist_on_pp": 1.25,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="down",
            shock_atr_pct=12.0,
            shock_dir_down_streak_bars=5,
            shock_drawdown_dist_on_pct=2.0,
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
        self.assertEqual(int(qty), 0)
        self.assertEqual(trace.zero_reason, "shock_short_entry_depth_gate")
        self.assertTrue(bool(trace.shock_short_entry_blocked))
        self.assertEqual(trace.shock_short_entry_gate_reason, "dd_dist_on_gt_1.25pp")

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
                "shock_short_entry_max_dist_on_pp": 1.25,
            },
            action="SELL",
            lot=1,
            entry_price=100.0,
            stop_price=101.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="down",
            shock_atr_pct=12.0,
            shock_dir_down_streak_bars=5,
            shock_drawdown_dist_on_pct=1.0,
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
        self.assertNotEqual(int(qty), 0)
        self.assertFalse(bool(trace.shock_short_entry_blocked))
        self.assertEqual(trace.shock_short_entry_gate_reason, "ok")

    def test_long_shock_boost_respects_drawdown_recovery_gate(self) -> None:
        # ON=-10, OFF=-5. Dist-on=-4.5 => dd=-5.5 => dd→off=-0.5 (within 1pp of OFF).
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
                "shock_on_drawdown_pct": -10.0,
                "shock_off_drawdown_pct": -5.0,
                "shock_long_risk_mult_factor": 1.5,
                "shock_long_boost_max_dist_off_pp": 1.0,
                "shock_long_boost_require_regime_up": True,
                "shock_long_boost_require_entry_up": True,
            },
            action="BUY",
            lot=1,
            entry_price=100.0,
            stop_price=99.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="up",
            shock_atr_pct=12.0,
            shock_drawdown_dist_on_pct=-4.5,
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
        self.assertTrue(bool(trace.shock_long_boost_applied))
        self.assertEqual(trace.shock_long_boost_gate_reason, "ok")
        self.assertAlmostEqual(float(trace.shock_long_factor or 0.0), 1.5, places=6)

        # Dist-on=-2.0 => dd=-8.0 => dd→off=-3.0 (too far from OFF for 1pp max).
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
                "shock_on_drawdown_pct": -10.0,
                "shock_off_drawdown_pct": -5.0,
                "shock_long_risk_mult_factor": 1.5,
                "shock_long_boost_max_dist_off_pp": 1.0,
                "shock_long_boost_require_regime_up": True,
                "shock_long_boost_require_entry_up": True,
            },
            action="BUY",
            lot=1,
            entry_price=100.0,
            stop_price=99.0,
            stop_loss_pct=None,
            shock=True,
            shock_dir="up",
            shock_atr_pct=12.0,
            shock_drawdown_dist_on_pct=-2.0,
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
        self.assertFalse(bool(trace.shock_long_boost_applied))
        self.assertEqual(trace.shock_long_boost_gate_reason, "dd_dist_off_abs_gt_1pp")
        self.assertAlmostEqual(float(trace.shock_long_factor or 0.0), 1.0, places=6)

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


def test_spot_sizing_input_contract_is_typed_and_complete() -> None:
    from dataclasses import fields, is_dataclass

    import tradebot.engine as engine_module
    from tradebot.spot import policy_contract as contract_module

    expected_fields = (
        "strategy",
        "filters",
        "action",
        "lot",
        "entry_price",
        "stop_price",
        "stop_loss_pct",
        "shock",
        "shock_dir",
        "shock_atr_pct",
        "shock_dir_down_streak_bars",
        "shock_drawdown_dist_on_pct",
        "shock_drawdown_dist_on_vel_pp",
        "shock_drawdown_dist_on_accel_pp",
        "shock_prearm_down_streak_bars",
        "shock_ramp",
        "riskoff",
        "risk_dir",
        "riskpanic",
        "riskpop",
        "risk",
        "signal_entry_dir",
        "signal_regime_dir",
        "regime2_dir",
        "regime2_ready",
        "equity_ref",
        "cash_ref",
        "policy_graph",
        "policy_config",
    )

    contract_type = getattr(contract_module, "SpotSizingInput", None)
    assert contract_type is not None, "SpotSizingInput contract is missing"
    assert is_dataclass(contract_type), "SpotSizingInput must be a dataclass"
    assert tuple(field.name for field in fields(contract_type)) == expected_fields

    factory = getattr(engine_module, "spot_sizing_input", None)
    assert callable(factory), "tradebot.engine.spot_sizing_input factory is missing"

    strategy = object()
    filters = object()
    risk = object()
    policy_graph = object()
    policy_config = object()
    values = {
        "strategy": strategy,
        "filters": filters,
        "action": "SELL",
        "lot": 3,
        "entry_price": 101.25,
        "stop_price": 103.0,
        "stop_loss_pct": 0.02,
        "shock": True,
        "shock_dir": "down",
        "shock_atr_pct": 8.5,
        "shock_dir_down_streak_bars": 4,
        "shock_drawdown_dist_on_pct": 1.5,
        "shock_drawdown_dist_on_vel_pp": 0.4,
        "shock_drawdown_dist_on_accel_pp": 0.1,
        "shock_prearm_down_streak_bars": 2,
        "shock_ramp": {"down": {"risk_mult": 1.5}},
        "riskoff": True,
        "risk_dir": "down",
        "riskpanic": False,
        "riskpop": False,
        "risk": risk,
        "signal_entry_dir": "down",
        "signal_regime_dir": "down",
        "regime2_dir": "down",
        "regime2_ready": True,
        "equity_ref": 2_200.0,
        "cash_ref": 1_800.0,
        "policy_graph": policy_graph,
        "policy_config": policy_config,
    }
    payload = factory(**values)
    assert isinstance(payload, contract_type)
    for name, expected in values.items():
        assert getattr(payload, name) == expected
