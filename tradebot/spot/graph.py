"""Composable spot policy graph.

This module provides a central registry for spot lifecycle/sizing policy nodes:
- entry gate policies
- exit arbitration policies
- resize target policies
- risk overlay policies

Profiles map to a coherent set of node names and can be overridden per-node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from .graph_core import (
    SpotEntryGateResult,
    SpotExitReasonResult,
    SpotResizeTargetResult,
    SpotRiskOverlayAdjustments,
    _GRAPH_PROFILES,
    _adaptive_target,
    _get,
    _node_name,
    _parse_float,
    _resolve_graph_profile_name,
    pick_exit_reason,
    spot_guard_threshold_scale,
)
from .graph_risk import (
    _risk_overlay_policy_atr_compress,
    _risk_overlay_policy_atr_compress_shock_dir_bias,
    _risk_overlay_policy_atr_compress_shock_dir_shortboost,
    _risk_overlay_policy_atr_compress_trend_bias,
    _risk_overlay_policy_legacy,
    _risk_overlay_policy_shock_dir_bias,
    _risk_overlay_policy_shock_dir_shortboost,
    _risk_overlay_policy_trend_bias,
)


def _entry_regime_context(
    entry_context: Mapping[str, object] | None,
) -> tuple[str, str, int | None]:
    """Decode legacy transport names once at the policy-graph boundary."""
    context = entry_context if isinstance(entry_context, Mapping) else {}
    state = str(context.get("regime4_state") or "").strip().lower()
    hard_dir = str(context.get("hard_dir") or "").strip().lower()
    try:
        release_age = (
            int(context["release_age_bars"])
            if context.get("release_age_bars") is not None
            else None
        )
    except (TypeError, ValueError):
        release_age = None
    return state, hard_dir, release_age


def _entry_policy_default(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    entry_context: Mapping[str, object] | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotEntryGateResult:
    _ = (
        graph,
        strategy,
        bar_ts,
        entry_dir,
        entry_context,
        shock_atr_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
        tr_ratio,
        tr_median_pct,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
    )
    return SpotEntryGateResult(
        allow=True,
        reason="entry_allowed",
        gate="TRIGGER_ENTRY",
        trace={"policy": "default"},
    )


def _entry_context_continuation_v1(
    *,
    entry_dir: str | None,
    entry_context: Mapping[str, object] | None,
) -> tuple[int, dict[str, object]]:
    context = entry_context if isinstance(entry_context, Mapping) else {}
    state, hard_dir, release_age = _entry_regime_context(context)
    branch = str(context.get("branch") or "").strip().lower()
    shock_dir = str(context.get("shock_dir") or "").strip().lower()
    flags = {
        "long_entry": str(entry_dir or "").strip().lower() == "up",
        "clean_downshock_mid_age": (
            state == "trend_up_clean"
            and hard_dir == "up"
            and shock_dir == "down"
            and release_age is not None
            and 500 <= int(release_age) < 1600
        ),
        "trend_down_branch_b_mid_age": (
            state == "trend_down"
            and branch == "b"
            and hard_dir == "up"
            and shock_dir == "down"
            and release_age is not None
            and 500 <= int(release_age) < 1600
        ),
        "transition_hot_stale_branch_a": (
            state == "transition_up_hot"
            and branch == "a"
            and hard_dir == "up"
            and shock_dir == "up"
            and release_age is not None
            and int(release_age) >= 1600
        ),
    }
    score = (
        int(bool(flags["clean_downshock_mid_age"]))
        + int(bool(flags["trend_down_branch_b_mid_age"]))
        + int(bool(flags["transition_hot_stale_branch_a"]))
    )
    return score, {
        "mode": "continuation_v1",
        "state": state or None,
        "branch": branch or None,
        "shock_dir": shock_dir or None,
        "hard_dir": hard_dir or None,
        "release_age_bars": int(release_age) if release_age is not None else None,
        "flags": flags,
        "score": int(score),
        "block_min_score": 1,
    }


def _entry_policy_slope_tr_guard(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    entry_context: Mapping[str, object] | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotEntryGateResult:
    _ = (graph, bar_ts)
    tr_min = max(0.0, _parse_float(_get(strategy, "spot_entry_tr_ratio_min"), default=0.0))
    slope_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slope_med_abs_min_pct"), default=0.0))
    slope_vel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slope_vel_abs_min_pct"), default=0.0))
    slope_slow_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slow_slope_med_abs_min_pct"), default=0.0))
    slope_vel_slow_min = max(0.0, _parse_float(_get(strategy, "spot_entry_slow_slope_vel_abs_min_pct"), default=0.0))
    threshold_scale, threshold_trace = spot_guard_threshold_scale(
        strategy=strategy,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    )
    tr_min *= float(threshold_scale)
    slope_min *= float(threshold_scale)
    slope_vel_min *= float(threshold_scale)
    slope_slow_min *= float(threshold_scale)
    slope_vel_slow_min *= float(threshold_scale)
    context_mode = str(_get(strategy, "spot_entry_context_confidence_mode", "off") or "off").strip().lower()
    context_trace: dict[str, object] | None = None
    if context_mode == "continuation_v1":
        context_score, context_trace = _entry_context_continuation_v1(
            entry_dir=entry_dir,
            entry_context=entry_context,
        )
        if bool(context_trace.get("flags", {}).get("long_entry")) and int(context_score) >= 1:
            return SpotEntryGateResult(
                allow=False,
                reason="graph_entry_context_confidence",
                gate="BLOCKED_GRAPH_ENTRY_CONTEXT_CONFIDENCE",
                trace={
                    "policy": "slope_tr_guard",
                    "context_confidence_mode": str(context_mode),
                    "context_confidence": dict(context_trace),
                },
            )
    atr_max = max(0.0, _parse_float(_get(strategy, "spot_entry_shock_atr_max_pct"), default=0.0))
    atr_vel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_atr_vel_min_pct"), default=0.0))
    atr_accel_min = max(0.0, _parse_float(_get(strategy, "spot_entry_atr_accel_min_pct"), default=0.0))
    if atr_max > 0.0 and shock_atr_pct is not None and float(shock_atr_pct) > float(atr_max):
        return SpotEntryGateResult(
            allow=False,
            reason="graph_entry_shock_atr",
            gate="BLOCKED_GRAPH_ENTRY_SHOCK",
            trace={"policy": "slope_tr_guard", "shock_atr_pct": float(shock_atr_pct), "max": float(atr_max)},
        )
    if atr_vel_min > 0.0:
        if shock_atr_vel_pct is None or float(shock_atr_vel_pct) < float(atr_vel_min):
            return SpotEntryGateResult(
                allow=False,
                reason="graph_entry_atr_vel",
                gate="BLOCKED_GRAPH_ENTRY_ATR_VEL",
                trace={"policy": "slope_tr_guard", "shock_atr_vel_pct": shock_atr_vel_pct, "min": float(atr_vel_min)},
            )
    if atr_accel_min > 0.0:
        if shock_atr_accel_pct is None or float(shock_atr_accel_pct) < float(atr_accel_min):
            return SpotEntryGateResult(
                allow=False,
                reason="graph_entry_atr_accel",
                gate="BLOCKED_GRAPH_ENTRY_ATR_ACCEL",
                trace={
                    "policy": "slope_tr_guard",
                    "shock_atr_accel_pct": shock_atr_accel_pct,
                    "min": float(atr_accel_min),
                },
            )
    if tr_min > 0.0 and (tr_ratio is None or float(tr_ratio) < float(tr_min)):
        return SpotEntryGateResult(
            allow=False,
            reason="graph_entry_tr_ratio",
            gate="BLOCKED_GRAPH_ENTRY_TR_RATIO",
            trace={"policy": "slope_tr_guard", "tr_ratio": tr_ratio, "min": float(tr_min)},
        )
    if entry_dir in ("up", "down"):
        direction = str(entry_dir)
        if slope_min > 0.0:
            if slope_med_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE",
                    trace={"policy": "slope_tr_guard", "slope_med_pct": None, "min_abs": float(slope_min)},
                )
            slope_ok = (direction == "up" and float(slope_med_pct) >= float(slope_min)) or (
                direction == "down" and float(slope_med_pct) <= -float(slope_min)
            )
            if not bool(slope_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_med_pct": float(slope_med_pct),
                        "min_abs": float(slope_min),
                    },
                )
        if slope_vel_min > 0.0:
            if slope_vel_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_vel_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE_VEL",
                    trace={"policy": "slope_tr_guard", "slope_vel_pct": None, "min_abs": float(slope_vel_min)},
                )
            slope_vel_ok = (direction == "up" and float(slope_vel_pct) >= float(slope_vel_min)) or (
                direction == "down" and float(slope_vel_pct) <= -float(slope_vel_min)
            )
            if not bool(slope_vel_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slope_vel",
                    gate="BLOCKED_GRAPH_ENTRY_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_vel_pct": float(slope_vel_pct),
                        "min_abs": float(slope_vel_min),
                    },
                )
        if slope_slow_min > 0.0:
            if slope_med_slow_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE",
                    trace={"policy": "slope_tr_guard", "slope_med_slow_pct": None, "min_abs": float(slope_slow_min)},
                )
            slope_slow_ok = (direction == "up" and float(slope_med_slow_pct) >= float(slope_slow_min)) or (
                direction == "down" and float(slope_med_slow_pct) <= -float(slope_slow_min)
            )
            if not bool(slope_slow_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_med_slow_pct": float(slope_med_slow_pct),
                        "min_abs": float(slope_slow_min),
                    },
                )
        if slope_vel_slow_min > 0.0:
            if slope_vel_slow_pct is None:
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_vel_missing",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "slope_vel_slow_pct": None,
                        "min_abs": float(slope_vel_slow_min),
                    },
                )
            slope_vel_slow_ok = (direction == "up" and float(slope_vel_slow_pct) >= float(slope_vel_slow_min)) or (
                direction == "down" and float(slope_vel_slow_pct) <= -float(slope_vel_slow_min)
            )
            if not bool(slope_vel_slow_ok):
                return SpotEntryGateResult(
                    allow=False,
                    reason="graph_entry_slow_slope_vel",
                    gate="BLOCKED_GRAPH_ENTRY_SLOW_SLOPE_VEL",
                    trace={
                        "policy": "slope_tr_guard",
                        "entry_dir": str(direction),
                        "slope_vel_slow_pct": float(slope_vel_slow_pct),
                        "min_abs": float(slope_vel_slow_min),
                    },
                )
    return SpotEntryGateResult(
        allow=True,
        reason="entry_allowed",
        gate="TRIGGER_ENTRY",
        trace={
            "policy": "slope_tr_guard",
            "guard_threshold_scale": float(threshold_scale),
            "guard_threshold_trace": dict(threshold_trace),
            "context_confidence_mode": str(context_mode),
            "context_confidence": dict(context_trace) if isinstance(context_trace, dict) else None,
        },
    )


def _exit_policy_priority(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    open_dir: str | None,
    signal_entry_dir: str | None,
    exit_candidates: Mapping[str, bool] | None,
    exit_priority: Sequence[str] | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
) -> SpotExitReasonResult:
    _ = (
        graph,
        strategy,
        open_dir,
        signal_entry_dir,
        tr_ratio,
        tr_median_pct,
        slope_med_pct,
        slope_vel_pct,
        slope_med_slow_pct,
        slope_vel_slow_pct,
        shock_atr_vel_pct,
        shock_atr_accel_pct,
    )
    reason = pick_exit_reason(exit_candidates, priority=exit_priority)
    return SpotExitReasonResult(reason=reason, trace={"policy": "priority"})


def _exit_policy_slope_flip_guard(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    open_dir: str | None,
    signal_entry_dir: str | None,
    exit_candidates: Mapping[str, bool] | None,
    exit_priority: Sequence[str] | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
) -> SpotExitReasonResult:
    _ = (graph, signal_entry_dir, shock_atr_vel_pct, shock_atr_accel_pct)
    candidates = {str(k): bool(v) for k, v in (exit_candidates or {}).items()}
    flip_suppressed = False
    slope_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slope_min_pct"), default=0.0))
    tr_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_tr_ratio_min"), default=0.0))
    slope_slow_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slow_slope_min_pct"), default=0.0))
    slope_vel_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slope_vel_min_pct"), default=0.0))
    slope_vel_slow_min = max(0.0, _parse_float(_get(strategy, "spot_exit_flip_hold_slow_slope_vel_min_pct"), default=0.0))
    threshold_scale, threshold_trace = spot_guard_threshold_scale(
        strategy=strategy,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    )
    slope_min *= float(threshold_scale)
    tr_min *= float(threshold_scale)
    slope_slow_min *= float(threshold_scale)
    slope_vel_min *= float(threshold_scale)
    slope_vel_slow_min *= float(threshold_scale)
    if bool(candidates.get("flip")) and open_dir in ("up", "down"):
        directional_ok = slope_min <= 0.0 or (
            slope_med_pct is not None
            and ((open_dir == "up" and float(slope_med_pct) >= float(slope_min)) or (open_dir == "down" and float(slope_med_pct) <= -float(slope_min)))
        )
        directional_slow_ok = slope_slow_min <= 0.0 or (
            slope_med_slow_pct is not None
            and (
                (open_dir == "up" and float(slope_med_slow_pct) >= float(slope_slow_min))
                or (open_dir == "down" and float(slope_med_slow_pct) <= -float(slope_slow_min))
            )
        )
        vel_ok = slope_vel_min <= 0.0 or (
            slope_vel_pct is not None
            and ((open_dir == "up" and float(slope_vel_pct) >= float(slope_vel_min)) or (open_dir == "down" and float(slope_vel_pct) <= -float(slope_vel_min)))
        )
        vel_slow_ok = slope_vel_slow_min <= 0.0 or (
            slope_vel_slow_pct is not None
            and (
                (open_dir == "up" and float(slope_vel_slow_pct) >= float(slope_vel_slow_min))
                or (open_dir == "down" and float(slope_vel_slow_pct) <= -float(slope_vel_slow_min))
            )
        )
        tr_ok = tr_min <= 0.0 or (tr_ratio is not None and float(tr_ratio) >= float(tr_min))
        if bool(directional_ok) and bool(directional_slow_ok) and bool(vel_ok) and bool(vel_slow_ok) and bool(tr_ok):
            candidates["flip"] = False
            flip_suppressed = True
    reason = pick_exit_reason(candidates, priority=exit_priority)
    return SpotExitReasonResult(
        reason=reason,
        trace={
            "policy": "slope_flip_guard",
            "flip_suppressed": bool(flip_suppressed),
            "guard_threshold_scale": float(threshold_scale),
            "guard_threshold_trace": dict(threshold_trace),
            "slope_med_pct": float(slope_med_pct) if slope_med_pct is not None else None,
            "slope_vel_pct": float(slope_vel_pct) if slope_vel_pct is not None else None,
            "slope_med_slow_pct": float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
            "slope_vel_slow_pct": float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
            "tr_ratio": float(tr_ratio) if tr_ratio is not None else None,
            "tr_median_pct": float(tr_median_pct) if tr_median_pct is not None else None,
        },
    )


def _resize_policy_adaptive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="off",
        default_min_mult=0.5,
        default_max_mult=1.75,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive", **out.trace})


def _resize_policy_adaptive_atr_defensive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="atr",
        default_min_mult=0.35,
        default_max_mult=1.0,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive_atr_defensive", **out.trace})


def _resize_policy_adaptive_hybrid_aggressive(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="hybrid",
        default_min_mult=0.5,
        default_max_mult=2.25,
    )
    return SpotResizeTargetResult(
        target_qty=int(out.target_qty),
        trace={"policy": "adaptive_hybrid_aggressive", **out.trace},
    )


def _resize_policy_adaptive_slope_probe(
    graph: SpotPolicyGraph,
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
) -> SpotResizeTargetResult:
    _ = graph
    out = _adaptive_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        default_mode="slope",
        default_min_mult=0.4,
        default_max_mult=1.5,
    )
    return SpotResizeTargetResult(target_qty=int(out.target_qty), trace={"policy": "adaptive_slope_probe", **out.trace})




class SpotPolicyGraph:
    def __init__(
        self,
        *,
        profile_name: str,
        entry_policy: str,
        exit_policy: str,
        resize_policy: str,
        risk_overlay_policy: str,
        block_trend_down_longs: bool = False,
    ) -> None:
        self.profile_name = str(profile_name)
        self.entry_policy = str(entry_policy)
        self.exit_policy = str(exit_policy)
        self.resize_policy = str(resize_policy)
        self.risk_overlay_policy = str(risk_overlay_policy)
        self.block_trend_down_longs = bool(block_trend_down_longs)

    @classmethod
    def from_sources(
        cls,
        *,
        strategy: Mapping[str, object] | object | None,
        filters: Mapping[str, object] | object | None,
    ) -> SpotPolicyGraph:
        profile_name = _resolve_graph_profile_name(strategy=strategy, filters=filters)
        profile = _GRAPH_PROFILES.get(str(profile_name), _GRAPH_PROFILES["neutral"])
        entry_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_entry_policy", "spot_graph_entry_policy"),
            default=str(profile.entry_policy),
        )
        exit_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_exit_policy", "spot_graph_exit_policy"),
            default=str(profile.exit_policy),
        )
        resize_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_resize_policy", "spot_graph_resize_policy"),
            default=str(profile.resize_policy),
        )
        risk_overlay_policy = _node_name(
            strategy=strategy,
            filters=filters,
            keys=("spot_risk_overlay_policy", "spot_graph_risk_overlay_policy"),
            default=str(profile.risk_overlay_policy),
        )
        return cls(
            profile_name=str(profile.name),
            entry_policy=str(entry_policy),
            exit_policy=str(exit_policy),
            resize_policy=str(resize_policy),
            risk_overlay_policy=str(risk_overlay_policy),
            block_trend_down_longs=bool(
                _get(strategy, "regime4_trenddown_block_longs", False)
            ),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "profile": str(self.profile_name),
            "entry_policy": str(self.entry_policy),
            "exit_policy": str(self.exit_policy),
            "resize_policy": str(self.resize_policy),
            "risk_overlay_policy": str(self.risk_overlay_policy),
            "block_trend_down_longs": bool(self.block_trend_down_longs),
        }

    def evaluate_entry_gate(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        bar_ts: datetime,
        entry_dir: str | None,
        entry_context: Mapping[str, object] | None = None,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        tr_median_pct: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
        entry_gate_bypass: bool = False,
    ) -> SpotEntryGateResult:
        regime_state, hard_dir, _release_age = _entry_regime_context(entry_context)
        if (
            self.block_trend_down_longs
            and entry_dir == "up"
            and regime_state == "trend_down"
            and hard_dir == "down"
        ):
            out = SpotEntryGateResult(
                allow=False,
                reason="regime4_trend_down",
                gate="BLOCKED_REGIME4_TREND_DOWN",
                trace={
                    "policy": "regime_guard",
                    "regime4_state": regime_state,
                    "hard_dir": hard_dir,
                },
            )
        elif entry_gate_bypass:
            out = SpotEntryGateResult(
                allow=True,
                reason="entry_gate_bypass",
                gate="ENTRY_GATE_BYPASS",
                trace={"policy": "bypass"},
            )
        else:
            fn = _ENTRY_POLICY_REGISTRY.get(str(self.entry_policy), _entry_policy_default)
            out = fn(
                self,
                strategy=strategy,
                bar_ts=bar_ts,
                entry_dir=entry_dir,
                entry_context=entry_context,
                shock_atr_pct=shock_atr_pct,
                shock_atr_vel_pct=shock_atr_vel_pct,
                shock_atr_accel_pct=shock_atr_accel_pct,
                tr_ratio=tr_ratio,
                tr_median_pct=tr_median_pct,
                slope_med_pct=slope_med_pct,
                slope_vel_pct=slope_vel_pct,
                slope_med_slow_pct=slope_med_slow_pct,
                slope_vel_slow_pct=slope_vel_slow_pct,
            )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotEntryGateResult(allow=bool(out.allow), reason=str(out.reason), gate=str(out.gate), trace=trace)

    def resolve_exit_reason(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        open_dir: str | None,
        signal_entry_dir: str | None,
        exit_candidates: Mapping[str, bool] | None,
        exit_priority: Sequence[str] | None = None,
        tr_ratio: float | None = None,
        tr_median_pct: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
    ) -> SpotExitReasonResult:
        fn = _EXIT_POLICY_REGISTRY.get(str(self.exit_policy), _exit_policy_priority)
        out = fn(
            self,
            strategy=strategy,
            open_dir=open_dir,
            signal_entry_dir=signal_entry_dir,
            exit_candidates=exit_candidates,
            exit_priority=exit_priority,
            tr_ratio=tr_ratio,
            tr_median_pct=tr_median_pct,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
        )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotExitReasonResult(reason=out.reason, trace=trace)

    def resolve_resize_target(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        current_qty: int,
        base_target_qty: int,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
    ) -> SpotResizeTargetResult:
        fn = _RESIZE_POLICY_REGISTRY.get(str(self.resize_policy), _resize_policy_adaptive)
        out = fn(
            self,
            strategy=strategy,
            current_qty=int(current_qty),
            base_target_qty=int(base_target_qty),
            shock_atr_pct=shock_atr_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
            tr_ratio=tr_ratio,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
        )
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotResizeTargetResult(target_qty=int(out.target_qty), trace=trace)

    def resolve_risk_overlay_adjustments(
        self,
        *,
        strategy: Mapping[str, object] | object | None,
        filters: Mapping[str, object] | object | None,
        action: str,
        shock_atr_pct: float | None = None,
        shock_atr_vel_pct: float | None = None,
        shock_atr_accel_pct: float | None = None,
        tr_ratio: float | None = None,
        slope_med_pct: float | None = None,
        slope_vel_pct: float | None = None,
        slope_med_slow_pct: float | None = None,
        slope_vel_slow_pct: float | None = None,
        riskoff: bool = False,
        riskpanic: bool = False,
        riskpop: bool = False,
        shock: bool = False,
        shock_dir: str | None = None,
    ) -> SpotRiskOverlayAdjustments:
        fn = _RISK_OVERLAY_POLICY_REGISTRY.get(str(self.risk_overlay_policy), _risk_overlay_policy_legacy)
        out = fn(
            self,
            strategy=strategy,
            filters=filters,
            action=str(action),
            shock_atr_pct=shock_atr_pct,
            shock_atr_vel_pct=shock_atr_vel_pct,
            shock_atr_accel_pct=shock_atr_accel_pct,
            tr_ratio=tr_ratio,
            slope_med_pct=slope_med_pct,
            slope_vel_pct=slope_vel_pct,
            slope_med_slow_pct=slope_med_slow_pct,
            slope_vel_slow_pct=slope_vel_slow_pct,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            shock=bool(shock),
            shock_dir=shock_dir,
        )
        risk_mult = max(0.0, float(out.risk_mult))
        long_mult = max(0.0, float(out.long_risk_mult))
        short_mult = max(0.0, float(out.short_risk_mult))
        cap_mult = max(0.0, float(out.cap_mult))
        trace = dict(out.trace) if isinstance(out.trace, dict) else {}
        trace["graph"] = self.as_payload()
        return SpotRiskOverlayAdjustments(
            risk_mult=float(risk_mult),
            long_risk_mult=float(long_mult),
            short_risk_mult=float(short_mult),
            cap_mult=float(cap_mult),
            trace=trace,
        )


_ENTRY_POLICY_REGISTRY: dict[str, Callable[..., SpotEntryGateResult]] = {
    "default": _entry_policy_default,
    "slope_tr_guard": _entry_policy_slope_tr_guard,
}

_EXIT_POLICY_REGISTRY: dict[str, Callable[..., SpotExitReasonResult]] = {
    "priority": _exit_policy_priority,
    "slope_flip_guard": _exit_policy_slope_flip_guard,
}

_RESIZE_POLICY_REGISTRY: dict[str, Callable[..., SpotResizeTargetResult]] = {
    "adaptive": _resize_policy_adaptive,
    "adaptive_atr_defensive": _resize_policy_adaptive_atr_defensive,
    "adaptive_hybrid_aggressive": _resize_policy_adaptive_hybrid_aggressive,
    "adaptive_slope_probe": _resize_policy_adaptive_slope_probe,
}

_RISK_OVERLAY_POLICY_REGISTRY: dict[str, Callable[..., SpotRiskOverlayAdjustments]] = {
    "legacy": _risk_overlay_policy_legacy,
    "atr_compress": _risk_overlay_policy_atr_compress,
    "atr_compress_shock_dir_bias": _risk_overlay_policy_atr_compress_shock_dir_bias,
    "atr_compress_shock_dir_shortboost": _risk_overlay_policy_atr_compress_shock_dir_shortboost,
    "atr_compress_trend_bias": _risk_overlay_policy_atr_compress_trend_bias,
    "shock_dir_bias": _risk_overlay_policy_shock_dir_bias,
    "shock_dir_shortboost": _risk_overlay_policy_shock_dir_shortboost,
    "trend_bias": _risk_overlay_policy_trend_bias,
}
