"""Spot policy-graph risk-overlay algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .graph_core import SpotRiskOverlayAdjustments, _get, _parse_float, _pick_value

if TYPE_CHECKING:
    from .graph import SpotPolicyGraph


def _risk_overlay_policy_legacy(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (
        graph,
        strategy,
        filters,
        action,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
        shock_dir,
    )
    return SpotRiskOverlayAdjustments(trace={"policy": "legacy"})


def _risk_overlay_policy_atr_compress(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (
        graph,
        action,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
        shock_dir,
    )
    hi_raw = _pick_value(
        strategy=strategy,
        filters=filters,
        keys=("spot_graph_overlay_atr_hi_pct", "spot_overlay_atr_hi_pct"),
    )
    min_raw = _pick_value(
        strategy=strategy,
        filters=filters,
        keys=("spot_graph_overlay_atr_hi_min_mult", "spot_overlay_atr_hi_min_mult"),
    )
    atr_hi = max(0.0, _parse_float(hi_raw, default=2.5))
    floor_mult = max(0.05, _parse_float(min_raw, default=0.5))
    scale = 1.0
    if atr_hi > 0.0 and shock_atr_pct is not None and float(shock_atr_pct) > float(atr_hi):
        scale = float(atr_hi) / float(shock_atr_pct)
        scale = float(max(floor_mult, min(1.0, scale)))
    return SpotRiskOverlayAdjustments(
        risk_mult=float(scale),
        cap_mult=float(scale),
        trace={
            "policy": "atr_compress",
            "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
            "atr_hi_pct": float(atr_hi),
            "min_mult": float(floor_mult),
            "scale": float(scale),
        },
    )

def _risk_overlay_policy_trend_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    _ = (graph, filters, shock_atr_pct, riskoff, riskpanic, riskpop, shock, shock_dir)
    action_u = str(action or "BUY").strip().upper()
    tr_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_tr_ratio_ref"), default=1.0))
    slope_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_slope_ref_pct"), default=0.08))
    boost_max = max(1.0, _parse_float(_get(strategy, "spot_graph_overlay_trend_boost_max"), default=1.35))
    floor_mult = max(0.05, _parse_float(_get(strategy, "spot_graph_overlay_trend_floor_mult"), default=0.65))
    atr_vel_ref = max(1e-9, _parse_float(_get(strategy, "spot_graph_overlay_atr_vel_ref_pct"), default=0.40))

    slope = float(slope_med_pct) if slope_med_pct is not None else 0.0
    vel = float(slope_vel_pct) if slope_vel_pct is not None else 0.0
    slope_slow = float(slope_med_slow_pct) if slope_med_slow_pct is not None else 0.0
    vel_slow = float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else 0.0
    atr_vel = float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else 0.0
    atr_accel = float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else 0.0
    tr_bonus = max(0.0, (float(tr_ratio) - float(tr_ref))) if tr_ratio is not None else 0.0

    if action_u == "SELL":
        slope = -float(slope)
        vel = -float(vel)
        slope_slow = -float(slope_slow)
        vel_slow = -float(vel_slow)

    score = (
        (float(slope) / float(slope_ref))
        + 0.5 * (float(vel) / float(slope_ref))
        + 0.35 * (float(slope_slow) / float(slope_ref))
        + 0.25 * (float(vel_slow) / float(slope_ref))
        + 0.20 * (float(atr_vel) / float(atr_vel_ref))
        + 0.10 * (float(atr_accel) / float(atr_vel_ref))
        + float(tr_bonus)
    )
    if score >= 0:
        mult = min(float(boost_max), 1.0 + 0.20 * float(score))
    else:
        mult = max(float(floor_mult), 1.0 + 0.20 * float(score))

    long_mult = float(mult) if action_u == "BUY" else 1.0
    short_mult = float(mult) if action_u == "SELL" else 1.0
    return SpotRiskOverlayAdjustments(
        risk_mult=1.0,
        long_risk_mult=float(long_mult),
        short_risk_mult=float(short_mult),
        cap_mult=1.0,
        trace={
            "policy": "trend_bias",
            "action": str(action_u),
            "score": float(score),
            "mult": float(mult),
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "slope_med_pct": float(slope_med_pct) if slope_med_pct is not None else None,
            "slope_vel_pct": float(slope_vel_pct) if slope_vel_pct is not None else None,
            "slope_med_slow_pct": float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
            "slope_vel_slow_pct": float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
            "shock_atr_vel_pct": float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
            "shock_atr_accel_pct": float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
            "tr_ref": float(tr_ref),
            "slope_ref_pct": float(slope_ref),
        },
    )


def _risk_overlay_policy_shock_dir_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    # Direction-only overlay: bias long/short risk based on the smoothed shock direction
    # (sign of recent returns). This is explicitly not tied to the volatility ramp.
    _ = (
        graph,
        filters,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
    )
    action_u = str(action or "BUY").strip().upper()
    shock_dir_u = str(shock_dir) if shock_dir in ("up", "down") else None

    boost_max = max(1.0, _parse_float(_get(strategy, "spot_graph_overlay_trend_boost_max"), default=1.15))
    floor_mult = max(0.05, _parse_float(_get(strategy, "spot_graph_overlay_trend_floor_mult"), default=0.65))

    align = None
    if shock_dir_u is not None:
        if action_u == "BUY":
            align = shock_dir_u == "up"
        elif action_u == "SELL":
            align = shock_dir_u == "down"

    if align is None:
        mult = 1.0
        reason = "dir_missing"
    elif align:
        mult = float(boost_max)
        reason = "align"
    else:
        mult = float(floor_mult)
        reason = "misalign"

    long_mult = float(mult) if action_u == "BUY" else 1.0
    short_mult = float(mult) if action_u == "SELL" else 1.0
    return SpotRiskOverlayAdjustments(
        risk_mult=1.0,
        long_risk_mult=float(long_mult),
        short_risk_mult=float(short_mult),
        cap_mult=1.0,
        trace={
            "policy": "shock_dir_bias",
            "action": str(action_u),
            "shock_dir": shock_dir_u,
            "align": align,
            "reason": str(reason),
            "mult": float(mult),
            "floor_mult": float(floor_mult),
            "boost_max": float(boost_max),
        },
    )


def _risk_overlay_policy_shock_dir_shortboost(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    # Downshift-only on longs, boost-enabled on shorts:
    # - BUY: align => 1.0, misalign => floor_mult
    # - SELL: align => boost_max, misalign => floor_mult
    _ = (
        graph,
        filters,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        riskoff,
        riskpanic,
        riskpop,
        shock,
    )
    action_u = str(action or "BUY").strip().upper()
    shock_dir_u = str(shock_dir) if shock_dir in ("up", "down") else None

    boost_max = max(1.0, _parse_float(_get(strategy, "spot_graph_overlay_trend_boost_max"), default=1.15))
    floor_mult = max(0.05, _parse_float(_get(strategy, "spot_graph_overlay_trend_floor_mult"), default=0.65))

    align = None
    if shock_dir_u is not None:
        if action_u == "BUY":
            align = shock_dir_u == "up"
        elif action_u == "SELL":
            align = shock_dir_u == "down"

    if align is None:
        mult = 1.0
        reason = "dir_missing"
    elif action_u == "BUY":
        if align:
            mult = 1.0
            reason = "align_no_boost"
        else:
            mult = float(floor_mult)
            reason = "misalign"
    else:
        if align:
            mult = float(boost_max)
            reason = "align_boost"
        else:
            mult = float(floor_mult)
            reason = "misalign"

    long_mult = float(mult) if action_u == "BUY" else 1.0
    short_mult = float(mult) if action_u == "SELL" else 1.0
    return SpotRiskOverlayAdjustments(
        risk_mult=1.0,
        long_risk_mult=float(long_mult),
        short_risk_mult=float(short_mult),
        cap_mult=1.0,
        trace={
            "policy": "shock_dir_shortboost",
            "action": str(action_u),
            "shock_dir": shock_dir_u,
            "align": align,
            "reason": str(reason),
            "mult": float(mult),
            "floor_mult": float(floor_mult),
            "boost_max": float(boost_max),
        },
    )


def _risk_overlay_policy_atr_compress_trend_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    # Keep ATR compression (crash hygiene) while also biasing long/short risk
    # based on signed trend strength (drift-down detection, chop defense).
    atr = _risk_overlay_policy_atr_compress(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    trend = _risk_overlay_policy_trend_bias(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    return SpotRiskOverlayAdjustments(
        risk_mult=float(atr.risk_mult) * float(trend.risk_mult),
        cap_mult=float(atr.cap_mult) * float(trend.cap_mult),
        long_risk_mult=float(trend.long_risk_mult),
        short_risk_mult=float(trend.short_risk_mult),
        trace={
            "policy": "atr_compress_trend_bias",
            "atr_compress": dict(atr.trace or {}),
            "trend_bias": dict(trend.trace or {}),
        },
    )


def _risk_overlay_policy_atr_compress_shock_dir_bias(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    atr = _risk_overlay_policy_atr_compress(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    bias = _risk_overlay_policy_shock_dir_bias(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    return SpotRiskOverlayAdjustments(
        risk_mult=float(atr.risk_mult) * float(bias.risk_mult),
        cap_mult=float(atr.cap_mult) * float(bias.cap_mult),
        long_risk_mult=float(bias.long_risk_mult),
        short_risk_mult=float(bias.short_risk_mult),
        trace={
            "policy": "atr_compress_shock_dir_bias",
            "atr_compress": dict(atr.trace or {}),
            "shock_dir_bias": dict(bias.trace or {}),
        },
    )


def _risk_overlay_policy_atr_compress_shock_dir_shortboost(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    filters: Mapping[str, object] | object | None,
    action: str,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    riskoff: bool,
    riskpanic: bool,
    riskpop: bool,
    shock: bool,
    shock_dir: str | None,
) -> SpotRiskOverlayAdjustments:
    atr = _risk_overlay_policy_atr_compress(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    bias = _risk_overlay_policy_shock_dir_shortboost(
        graph,
        strategy=strategy,
        filters=filters,
        action=action,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        riskoff=riskoff,
        riskpanic=riskpanic,
        riskpop=riskpop,
        shock=shock,
        shock_dir=shock_dir,
    )
    return SpotRiskOverlayAdjustments(
        risk_mult=float(atr.risk_mult) * float(bias.risk_mult),
        cap_mult=float(atr.cap_mult) * float(bias.cap_mult),
        long_risk_mult=float(bias.long_risk_mult),
        short_risk_mult=float(bias.short_risk_mult),
        trace={
            "policy": "atr_compress_shock_dir_shortboost",
            "atr_compress": dict(atr.trace or {}),
            "shock_dir_shortboost": dict(bias.trace or {}),
        },
    )
