from __future__ import annotations

from datetime import datetime

from tradebot.spot.graph import SpotPolicyGraph, spot_dynamic_flip_hold_bars
from tradebot.spot.lifecycle import decide_flat_position_intent
from tradebot.spot.policy import SpotPolicy


def test_graph_profile_resolution_and_overrides() -> None:
    graph = SpotPolicyGraph.from_sources(strategy={"spot_policy_pack": "defensive"}, filters=None)
    assert str(graph.profile_name) == "defensive"
    assert str(graph.entry_policy) == "slope_tr_guard"
    assert str(graph.resize_policy) == "adaptive_atr_defensive"

    graph_override = SpotPolicyGraph.from_sources(
        strategy={"spot_policy_pack": "defensive", "spot_exit_policy": "priority"},
        filters=None,
    )
    assert str(graph_override.exit_policy) == "priority"


def test_graph_entry_policy_blocks_on_tr_ratio() -> None:
    graph = SpotPolicyGraph.from_sources(
        strategy={"spot_entry_policy": "slope_tr_guard", "spot_entry_tr_ratio_min": 1.2},
        filters=None,
    )
    decision = graph.evaluate_entry_gate(
        strategy={"spot_entry_policy": "slope_tr_guard", "spot_entry_tr_ratio_min": 1.2},
        bar_ts=datetime(2026, 2, 14, 10, 0, 0),
        entry_dir="up",
        tr_ratio=1.0,
    )
    assert bool(decision.allow) is False
    assert str(decision.gate) == "BLOCKED_GRAPH_ENTRY_TR_RATIO"


def test_graph_exit_policy_can_suppress_flip() -> None:
    strategy = {
        "spot_exit_policy": "slope_flip_guard",
        "spot_exit_flip_hold_slope_min_pct": 0.05,
    }
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    out = graph.resolve_exit_reason(
        strategy=strategy,
        open_dir="up",
        signal_entry_dir="down",
        exit_candidates={"flip": True},
        slope_med_pct=0.10,
        tr_ratio=1.5,
    )
    assert out.reason is None
    assert bool(out.trace.get("flip_suppressed")) is True


def test_lifecycle_flat_entry_uses_graph_gate() -> None:
    strategy = {
        "spot_entry_policy": "slope_tr_guard",
        "spot_entry_slope_med_abs_min_pct": 0.10,
        "spot_entry_fill_mode": "close",
    }
    decision = decide_flat_position_intent(
        strategy=strategy,
        bar_ts=datetime(2026, 2, 14, 10, 0, 0),
        entry_dir="up",
        allowed_directions=("up", "down"),
        can_order_now=True,
        preflight_ok=True,
        filters_ok=True,
        entry_capacity=True,
        atr_ready=True,
        next_open_allowed=True,
        slope_med_pct=0.02,
    )
    assert str(decision.intent) == "hold"
    assert str(decision.gate) == "BLOCKED_GRAPH_ENTRY_SLOPE"


def test_graph_risk_overlay_compresses_qty() -> None:
    base_strategy = {
        "quantity": 1,
        "spot_sizing_mode": "risk_pct",
        "spot_risk_pct": 0.02,
        "spot_min_qty": 1,
        "spot_max_qty": 0,
        "spot_max_notional_pct": 1.0,
        "spot_risk_overlay_policy": "legacy",
    }
    compressed_strategy = {
        **base_strategy,
        "spot_risk_overlay_policy": "atr_compress",
        "spot_graph_overlay_atr_hi_pct": 1.0,
        "spot_graph_overlay_atr_hi_min_mult": 0.25,
    }
    qty_base, _trace_base = SpotPolicy.calc_signed_qty_with_trace(
        strategy=base_strategy,
        filters=None,
        action="BUY",
        lot=1,
        entry_price=100.0,
        stop_price=None,
        stop_loss_pct=0.02,
        shock=True,
        shock_dir="up",
        shock_atr_pct=4.0,
        equity_ref=100000.0,
        cash_ref=100000.0,
    )
    qty_compressed, trace_compressed = SpotPolicy.calc_signed_qty_with_trace(
        strategy=compressed_strategy,
        filters=None,
        action="BUY",
        lot=1,
        entry_price=100.0,
        stop_price=None,
        stop_loss_pct=0.02,
        shock=True,
        shock_dir="up",
        shock_atr_pct=4.0,
        equity_ref=100000.0,
        cash_ref=100000.0,
    )
    assert int(qty_base) > int(qty_compressed)
    payload = trace_compressed.as_payload()
    overlay = payload.get("graph_overlay_trace")
    assert isinstance(overlay, dict)
    assert str(overlay.get("trace", {}).get("policy")) == "atr_compress"


def test_graph_entry_guard_dynamic_threshold_scale_tightens_in_calm_regime() -> None:
    strategy = {
        "spot_entry_policy": "slope_tr_guard",
        "spot_entry_slope_med_abs_min_pct": 0.05,
        "spot_guard_threshold_scale_mode": "tr",
        "spot_guard_threshold_scale_min_mult": 0.5,
        "spot_guard_threshold_scale_max_mult": 2.0,
        "spot_guard_threshold_scale_tr_ref": 1.0,
    }
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    decision = graph.evaluate_entry_gate(
        strategy=strategy,
        bar_ts=datetime(2026, 2, 14, 10, 0, 0),
        entry_dir="up",
        tr_ratio=0.5,
        slope_med_pct=0.07,
    )
    assert bool(decision.allow) is False
    assert str(decision.gate) == "BLOCKED_GRAPH_ENTRY_SLOPE"
    trace = dict(decision.trace)
    assert abs(float(trace.get("min_abs", 0.0)) - 0.10) < 1e-12


def test_graph_exit_guard_dynamic_threshold_scale_can_preserve_flip() -> None:
    strategy = {
        "spot_exit_policy": "slope_flip_guard",
        "spot_exit_flip_hold_slope_min_pct": 0.05,
        "spot_guard_threshold_scale_mode": "tr",
        "spot_guard_threshold_scale_min_mult": 0.5,
        "spot_guard_threshold_scale_max_mult": 2.0,
        "spot_guard_threshold_scale_tr_ref": 1.0,
    }
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    out = graph.resolve_exit_reason(
        strategy=strategy,
        open_dir="up",
        signal_entry_dir="down",
        exit_candidates={"flip": True},
        tr_ratio=0.5,
        slope_med_pct=0.07,
    )
    assert str(out.reason) == "flip"
    assert bool(out.trace.get("flip_suppressed")) is False


def test_spot_dynamic_flip_hold_bars_scales_by_tr_ratio() -> None:
    strategy = {
        "flip_exit_min_hold_bars": 6,
        "spot_flip_hold_dynamic_mode": "tr",
        "spot_flip_hold_dynamic_min_mult": 0.5,
        "spot_flip_hold_dynamic_max_mult": 2.0,
        "spot_flip_hold_dynamic_tr_ref": 1.0,
    }
    hold_hi, _trace_hi = spot_dynamic_flip_hold_bars(strategy=strategy, tr_ratio=2.0, shock_atr_vel_pct=None)
    hold_lo, _trace_lo = spot_dynamic_flip_hold_bars(strategy=strategy, tr_ratio=0.5, shock_atr_vel_pct=None)
    assert int(hold_hi) == 3
    assert int(hold_lo) == 12


def test_graph_entry_guard_dynamic_threshold_scale_supports_tr_median_mode() -> None:
    strategy = {
        "spot_entry_policy": "slope_tr_guard",
        "spot_entry_slope_med_abs_min_pct": 0.05,
        "spot_guard_threshold_scale_mode": "tr_median",
        "spot_guard_threshold_scale_min_mult": 0.5,
        "spot_guard_threshold_scale_max_mult": 2.0,
        "spot_guard_threshold_scale_tr_median_ref_pct": 0.06,
    }
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    decision = graph.evaluate_entry_gate(
        strategy=strategy,
        bar_ts=datetime(2026, 2, 14, 10, 0, 0),
        entry_dir="up",
        tr_median_pct=0.03,
        slope_med_pct=0.07,
    )
    assert bool(decision.allow) is False
    assert str(decision.gate) == "BLOCKED_GRAPH_ENTRY_SLOPE"
    trace = dict(decision.trace)
    assert abs(float(trace.get("min_abs", 0.0)) - 0.10) < 1e-12


def test_spot_dynamic_flip_hold_bars_scales_by_tr_median() -> None:
    strategy = {
        "flip_exit_min_hold_bars": 6,
        "spot_flip_hold_dynamic_mode": "tr_median",
        "spot_flip_hold_dynamic_min_mult": 0.5,
        "spot_flip_hold_dynamic_max_mult": 2.0,
        "spot_flip_hold_dynamic_tr_median_ref_pct": 0.06,
    }
    hold_hi, _trace_hi = spot_dynamic_flip_hold_bars(
        strategy=strategy,
        tr_ratio=None,
        shock_atr_vel_pct=None,
        tr_median_pct=0.12,
    )
    hold_lo, _trace_lo = spot_dynamic_flip_hold_bars(
        strategy=strategy,
        tr_ratio=None,
        shock_atr_vel_pct=None,
        tr_median_pct=0.03,
    )
    assert int(hold_hi) == 3
    assert int(hold_lo) == 12
