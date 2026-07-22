"""Shared spot lifecycle decision kernel.

This module centralizes spot lifecycle decisions used by both:
- live UI runtime (`tradebot/ui/bot_signal_runtime.py`)
- backtest runtime (`tradebot/backtest/engine.py`)

Kernel outputs one typed decision:
- `hold`
- `enter`
- `exit`
- `resize`

with a normalized gate/result payload for consistent diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from ..time_utils import NaiveTsModeInput
from .fill_modes import (
    SPOT_FILL_MODE_CLOSE,
    SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    normalize_spot_fill_mode,
    spot_fill_mode_is_deferred,
    spot_fill_mode_is_next_tradable,
)
from .gates import _bars_elapsed, _get, _normalize_fill_mode
from .graph import SpotPolicyGraph
from .graph_core import canonical_exit_reason as graph_canonical_exit_reason
from .graph_core import pick_exit_reason as graph_pick_exit_reason
from .policy import SpotPolicy
from .policy_contract import SpotIntentDecision, SpotPolicyConfigView


@dataclass(frozen=True)
class SpotLifecycleDecision:
    intent: str
    reason: str
    gate: str
    direction: str | None = None
    fill_mode: str = SPOT_FILL_MODE_CLOSE
    blocked: bool = False
    pending_clear_entry: bool = False
    pending_clear_exit: bool = False
    queue_reentry_dir: str | None = None
    spot_intent: SpotIntentDecision | None = None
    spot_decision: dict[str, object] | None = None
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        if self.spot_intent is not None:
            payload["spot_intent"] = self.spot_intent.as_payload()
        return payload


def canonical_exit_reason(reason: str | None) -> str:
    return graph_canonical_exit_reason(reason)


def pick_exit_reason(
    exit_candidates: Mapping[str, bool] | None,
    *,
    priority: Sequence[str] | None = None,
) -> str | None:
    return graph_pick_exit_reason(exit_candidates, priority=priority)




def adaptive_resize_target_qty(
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
) -> tuple[int, dict[str, object]]:
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    out = graph.resolve_resize_target(
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
    return int(out.target_qty), dict(out.trace)


def decide_pending_next_open(
    *,
    now_ts: datetime,
    has_open: bool,
    open_dir: str | None,
    pending_entry_dir: str | None,
    pending_entry_set_date: date | None,
    pending_entry_due_ts: datetime | None,
    pending_exit_reason: str | None,
    pending_exit_due_ts: datetime | None,
    risk_overlay_enabled: bool,
    riskoff_today: bool,
    riskpanic_today: bool,
    riskpop_today: bool,
    riskoff_mode: str,
    shock_dir_now: str | None,
    riskoff_end_hour: int | None,
    pending_entry_fill_mode: str = SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    pending_exit_fill_mode: str = SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    naive_ts_mode: NaiveTsModeInput = None,
) -> SpotLifecycleDecision:
    entry_fill_mode = normalize_spot_fill_mode(pending_entry_fill_mode, default=SPOT_FILL_MODE_NEXT_TRADABLE_BAR)
    exit_fill_mode = normalize_spot_fill_mode(pending_exit_fill_mode, default=SPOT_FILL_MODE_NEXT_TRADABLE_BAR)

    if pending_exit_due_ts is not None and now_ts >= pending_exit_due_ts:
        if bool(has_open):
            reason = canonical_exit_reason(pending_exit_reason or "flip")
            return SpotLifecycleDecision(
                intent="exit",
                reason=reason or "flip",
                gate="TRIGGER_EXIT",
                direction=str(open_dir) if open_dir in ("up", "down") else None,
                fill_mode=exit_fill_mode,
                pending_clear_exit=True,
                trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
            )
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_exit_empty",
            gate="CLEAR_PENDING_EXIT",
            pending_clear_exit=True,
            trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
        )

    if pending_entry_dir in ("up", "down") and pending_entry_due_ts is not None:
        should_cancel = SpotPolicy.pending_entry_should_cancel(
            pending_dir=str(pending_entry_dir),
            pending_set_date=pending_entry_set_date,
            exec_ts=now_ts,
            risk_overlay_enabled=bool(risk_overlay_enabled),
            riskoff_today=bool(riskoff_today),
            riskpanic_today=bool(riskpanic_today),
            riskpop_today=bool(riskpop_today),
            riskoff_mode=str(riskoff_mode),
            shock_dir_now=shock_dir_now if shock_dir_now in ("up", "down") else None,
            riskoff_end_hour=riskoff_end_hour,
            naive_ts_mode=naive_ts_mode,
        )
        if bool(should_cancel):
            return SpotLifecycleDecision(
                intent="hold",
                reason="pending_entry_risk_overlay_cancel",
                gate="CANCEL_PENDING_ENTRY_RISK_OVERLAY",
                pending_clear_entry=True,
                trace={
                    "stage": "pending",
                    "pending_kind": "entry",
                    "direction": str(pending_entry_dir),
                    "due": pending_entry_due_ts.isoformat(),
                },
            )
        if now_ts >= pending_entry_due_ts:
            if not bool(has_open):
                return SpotLifecycleDecision(
                    intent="enter",
                    reason="next_open",
                    gate="TRIGGER_ENTRY",
                    direction=str(pending_entry_dir),
                    fill_mode=entry_fill_mode,
                    pending_clear_entry=True,
                    trace={
                        "stage": "pending",
                        "pending_kind": "entry",
                        "direction": str(pending_entry_dir),
                        "due": pending_entry_due_ts.isoformat(),
                    },
                )
            return SpotLifecycleDecision(
                intent="hold",
                reason="pending_entry_blocked_open",
                gate="CLEAR_PENDING_ENTRY",
                pending_clear_entry=True,
                trace={
                    "stage": "pending",
                    "pending_kind": "entry",
                    "direction": str(pending_entry_dir),
                    "due": pending_entry_due_ts.isoformat(),
                },
            )
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_entry_wait",
            gate="PENDING_ENTRY_NEXT_OPEN",
            direction=str(pending_entry_dir),
            fill_mode=entry_fill_mode,
            trace={
                "stage": "pending",
                "pending_kind": "entry",
                "direction": str(pending_entry_dir),
                "due": pending_entry_due_ts.isoformat(),
            },
        )

    if pending_exit_due_ts is not None:
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_exit_wait",
            gate="PENDING_EXIT_NEXT_OPEN",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            fill_mode=exit_fill_mode,
            trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
        )
    return SpotLifecycleDecision(intent="hold", reason="no_pending", gate="HOLDING", trace={"stage": "pending"})


def decide_open_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    bar_size: str,
    open_dir: str | None,
    current_qty: int,
    exit_candidates: Mapping[str, bool] | None = None,
    exit_priority: Sequence[str] | None = None,
    target_qty: int | None = None,
    spot_decision: dict[str, object] | None = None,
    last_resize_bar_ts: datetime | None = None,
    signal_entry_dir: str | None = None,
    shock_atr_pct: float | None = None,
    shock_atr_vel_pct: float | None = None,
    shock_atr_accel_pct: float | None = None,
    tr_ratio: float | None = None,
    tr_median_pct: float | None = None,
    slope_med_pct: float | None = None,
    slope_vel_pct: float | None = None,
    slope_med_slow_pct: float | None = None,
    slope_vel_slow_pct: float | None = None,
    policy_graph: SpotPolicyGraph | None = None,
    policy_config: SpotPolicyConfigView | None = None,
) -> SpotLifecycleDecision:
    graph = policy_graph or SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    exit_pick = graph.resolve_exit_reason(
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
    exit_reason = exit_pick.reason
    if exit_reason:
        flip_fill = _normalize_fill_mode(_get(strategy, "spot_flip_exit_fill_mode", "close"), default="close")
        fill_mode = (
            str(flip_fill)
            if exit_reason == "flip" and spot_fill_mode_is_deferred(flip_fill)
            else SPOT_FILL_MODE_CLOSE
        )

        queue_reentry_dir = None
        if bool(_get(strategy, "spot_controlled_flip", False)) and exit_reason == "flip":
            desired = str(signal_entry_dir) if signal_entry_dir in ("up", "down") else None
            if desired is not None and desired != str(open_dir):
                queue_reentry_dir = desired

        return SpotLifecycleDecision(
            intent="exit",
            reason=str(exit_reason),
            gate="TRIGGER_EXIT",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            fill_mode=fill_mode,
            queue_reentry_dir=queue_reentry_dir,
            trace={
                "stage": "open",
                "path": "exit",
                "exit_reason": str(exit_reason),
                "fill_mode": fill_mode,
                "controlled_flip": bool(queue_reentry_dir is not None),
                "exit_policy": exit_pick.as_payload(),
            },
        )

    if target_qty is None:
        return SpotLifecycleDecision(
            intent="hold",
            reason="holding_no_resize_target",
            gate="HOLDING",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            trace={"stage": "open", "path": "hold"},
        )

    resize_target = graph.resolve_resize_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
    )
    effective_target = int(resize_target.target_qty)
    adaptive = dict(resize_target.trace)
    spot_intent = SpotPolicy.resolve_position_intent(
        strategy=strategy,
        current_qty=int(current_qty),
        target_qty=int(effective_target),
        policy_config=policy_config,
    )
    if str(spot_intent.intent) == "resize":
        cfg = policy_config or SpotPolicyConfigView.from_sources(strategy=strategy, filters=None)
        cooldown = max(0, int(cfg.spot_resize_cooldown_bars))
        if cooldown > 0:
            elapsed = _bars_elapsed(last_resize_bar_ts, bar_ts, bar_size=str(bar_size))
            if elapsed < int(cooldown):
                return SpotLifecycleDecision(
                    intent="hold",
                    reason="resize_cooldown",
                    gate="BLOCKED_RESIZE_COOLDOWN",
                    direction=str(open_dir) if open_dir in ("up", "down") else None,
                    blocked=True,
                    spot_intent=spot_intent,
                    spot_decision=spot_decision,
                    trace={
                        "stage": "open",
                        "path": "resize",
                        "cooldown_bars": int(cooldown),
                        "elapsed_bars": int(elapsed),
                        "resize_policy": adaptive,
                    },
                )
        return SpotLifecycleDecision(
            intent="resize",
            reason=str(spot_intent.reason or "target_delta"),
            gate="TRIGGER_RESIZE",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "resize", "resize_policy": adaptive},
        )

    if str(spot_intent.intent) == "enter":
        target_dir = "up" if int(spot_intent.target_qty) > 0 else "down" if int(spot_intent.target_qty) < 0 else None
        return SpotLifecycleDecision(
            intent="enter",
            reason=str(spot_intent.reason or "from_flat"),
            gate="TRIGGER_ENTRY",
            direction=target_dir,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "enter", "resize_policy": adaptive},
        )

    if str(spot_intent.intent) == "exit":
        return SpotLifecycleDecision(
            intent="exit",
            reason=str(spot_intent.reason or "target_zero"),
            gate="TRIGGER_EXIT",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "exit_from_intent", "resize_policy": adaptive},
        )

    blocked_gate = "HOLDING" if not bool(spot_intent.blocked) else "BLOCKED_RESIZE"
    return SpotLifecycleDecision(
        intent="hold",
        reason=str(spot_intent.reason or "holding"),
        gate=blocked_gate,
        direction=str(open_dir) if open_dir in ("up", "down") else None,
        blocked=bool(spot_intent.blocked),
        spot_intent=spot_intent,
        spot_decision=spot_decision,
        trace={"stage": "open", "path": "hold", "resize_policy": adaptive},
    )


def decide_flat_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    entry_context: Mapping[str, object] | None = None,
    allowed_directions: Sequence[str],
    can_order_now: bool,
    preflight_ok: bool,
    filters_ok: bool,
    entry_capacity: bool,
    stale_signal: bool = False,
    gap_signal: bool = False,
    pending_exists: bool = False,
    atr_ready: bool = True,
    next_open_allowed: bool = True,
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
    policy_graph: SpotPolicyGraph | None = None,
) -> SpotLifecycleDecision:
    if bool(stale_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="stale_signal",
            gate="BLOCKED_STALE_SIGNAL",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if bool(gap_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="data_gap",
            gate="WAITING_DATA_GAP",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(preflight_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="preflight",
            gate="WAITING_PREFLIGHT_BARS",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if bool(pending_exists):
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_order",
            gate="PENDING_ORDER",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(can_order_now):
        return SpotLifecycleDecision(
            intent="hold",
            reason="weekday_gate",
            gate="BLOCKED_WEEKDAY_NOW",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(entry_capacity):
        return SpotLifecycleDecision(
            intent="hold",
            reason="entry_limit",
            gate="BLOCKED_ENTRY_LIMIT",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if entry_dir not in ("up", "down"):
        return SpotLifecycleDecision(
            intent="hold",
            reason="waiting_signal",
            gate="WAITING_SIGNAL",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if str(entry_dir) not in {str(d) for d in allowed_directions}:
        return SpotLifecycleDecision(
            intent="hold",
            reason="direction_blocked",
            gate="BLOCKED_DIRECTION",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(filters_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="filters",
            gate="BLOCKED_FILTERS",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(atr_ready):
        return SpotLifecycleDecision(
            intent="hold",
            reason="atr_not_ready",
            gate="BLOCKED_ATR_NOT_READY",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )

    fill_mode = _normalize_fill_mode(_get(strategy, "spot_entry_fill_mode", "close"), default="close")
    if spot_fill_mode_is_next_tradable(fill_mode) and not bool(next_open_allowed):
        return SpotLifecycleDecision(
            intent="hold",
            reason="next_open_not_allowed",
            gate="BLOCKED_NEXT_OPEN",
            direction=str(entry_dir),
            blocked=True,
            fill_mode=str(fill_mode),
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )

    graph = policy_graph or SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    entry_gate = graph.evaluate_entry_gate(
        strategy=strategy,
        bar_ts=bar_ts,
        entry_dir=str(entry_dir) if entry_dir in ("up", "down") else None,
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
        entry_gate_bypass=bool(entry_gate_bypass),
    )
    graph_payload = entry_gate.as_payload()
    if not bool(entry_gate.allow):
        return SpotLifecycleDecision(
            intent="hold",
            reason=str(entry_gate.reason or "graph_entry_gate"),
            gate=str(entry_gate.gate or "BLOCKED_GRAPH_ENTRY"),
            direction=str(entry_dir) if entry_dir in ("up", "down") else None,
            fill_mode=str(fill_mode),
            blocked=True,
            trace={
                "stage": "flat",
                "bar_ts": bar_ts.isoformat(),
                "fill_mode": str(fill_mode),
                "graph_entry": graph_payload,
            },
        )
    return SpotLifecycleDecision(
        intent="enter",
        reason="entry",
        gate="TRIGGER_ENTRY",
        direction=str(entry_dir),
        fill_mode=str(fill_mode),
        blocked=False,
        trace={
            "stage": "flat",
            "bar_ts": bar_ts.isoformat(),
            "fill_mode": str(fill_mode),
            "graph_entry": graph_payload,
        },
    )
