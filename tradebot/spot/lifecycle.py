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

import math

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


@dataclass(frozen=True)
class SpotEntryBasisState:
    quantity: float
    basis_price: float | None
    source: str


@dataclass(frozen=True)
class SpotExcursionPolicy:
    """Causal stop/trail/fizzle policy kernel."""

    initial_stop_atr: float = 0.0
    trail_activate_atr: float = 0.0
    trail_distance_atr: float = 0.0
    breakeven_atr: float = 0.0
    fizzle_bars: int = 0
    fizzle_mfe_atr: float = 0.0
    max_hold_bars: int = 0

    @property
    def enabled(self) -> bool:
        return bool(
            self.initial_stop_atr > 0.0
            or (
                self.trail_activate_atr > 0.0
                and self.trail_distance_atr > 0.0
            )
            or self.breakeven_atr > 0.0
            or self.fizzle_bars > 0
            or self.max_hold_bars > 0
        )

    @classmethod
    def from_strategy(
        cls,
        strategy: Mapping[str, object] | object | None,
    ) -> "SpotExcursionPolicy":
        raw = _get(strategy, "spot_excursion_exit", None)
        if not isinstance(raw, Mapping) or not bool(raw.get("enabled", True)):
            return cls()

        def _float(key: str) -> float:
            try:
                return max(0.0, float(raw.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _int(key: str) -> int:
            try:
                return max(0, int(raw.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            initial_stop_atr=_float("initial_stop_atr"),
            trail_activate_atr=_float("trail_activate_atr"),
            trail_distance_atr=_float("trail_distance_atr"),
            breakeven_atr=_float("breakeven_atr"),
            fizzle_bars=_int("fizzle_bars"),
            fizzle_mfe_atr=_float("fizzle_mfe_atr"),
            max_hold_bars=_int("max_hold_bars"),
        )

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpotExcursionState:
    """Completed-bar excursion state; a ratchet never affects its source bar."""

    direction: str
    entry_price: float
    entry_atr: float
    bars_held: int
    best_price: float
    worst_price: float
    stop_price: float | None
    stop_reason: str | None = None

    @classmethod
    def open(
        cls,
        *,
        policy: SpotExcursionPolicy,
        direction: str,
        entry_price: float,
        entry_atr: float,
    ) -> "SpotExcursionState":
        if not policy.enabled or direction not in ("up", "down"):
            raise ValueError("excursion state requires an enabled policy and direction")
        entry = float(entry_price)
        atr = float(entry_atr)
        if entry <= 0.0 or atr <= 0.0:
            raise ValueError("excursion state requires positive entry price and ATR")
        stop = None
        stop_reason = None
        if policy.initial_stop_atr > 0.0:
            stop = entry - (policy.initial_stop_atr * atr)
            if direction == "down":
                stop = entry + (policy.initial_stop_atr * atr)
            stop_reason = "initial_stop"
        return cls(
            direction=str(direction),
            entry_price=entry,
            entry_atr=atr,
            bars_held=0,
            best_price=entry,
            worst_price=entry,
            stop_price=float(stop) if stop is not None else None,
            stop_reason=stop_reason,
        )

    @property
    def mfe_points(self) -> float:
        if self.direction == "up":
            return max(0.0, self.best_price - self.entry_price)
        return max(0.0, self.entry_price - self.best_price)

    @property
    def mae_points(self) -> float:
        if self.direction == "up":
            return max(0.0, self.entry_price - self.worst_price)
        return max(0.0, self.worst_price - self.entry_price)

    def advance(
        self,
        *,
        policy: SpotExcursionPolicy,
        high: float,
        low: float,
    ) -> tuple["SpotExcursionState", str | None]:
        if self.direction == "up":
            best = max(self.best_price, float(high))
            worst = min(self.worst_price, float(low))
        else:
            best = min(self.best_price, float(low))
            worst = max(self.worst_price, float(high))

        bars = self.bars_held + 1
        mfe = (
            max(0.0, best - self.entry_price)
            if self.direction == "up"
            else max(0.0, self.entry_price - best)
        )
        stop = self.stop_price
        stop_reason = self.stop_reason

        if (
            policy.breakeven_atr > 0.0
            and mfe >= policy.breakeven_atr * self.entry_atr
        ):
            candidate = self.entry_price
            tighter = stop is None or (
                candidate > stop if self.direction == "up" else candidate < stop
            )
            if tighter:
                stop, stop_reason = candidate, "breakeven_stop"

        if (
            policy.trail_activate_atr > 0.0
            and policy.trail_distance_atr > 0.0
            and mfe >= policy.trail_activate_atr * self.entry_atr
        ):
            distance = policy.trail_distance_atr * self.entry_atr
            candidate = best - distance if self.direction == "up" else best + distance
            tighter = stop is None or (
                candidate > stop if self.direction == "up" else candidate < stop
            )
            if tighter:
                stop, stop_reason = candidate, "trail_stop"

        state = SpotExcursionState(
            direction=self.direction,
            entry_price=self.entry_price,
            entry_atr=self.entry_atr,
            bars_held=bars,
            best_price=best,
            worst_price=worst,
            stop_price=float(stop) if stop is not None else None,
            stop_reason=stop_reason,
        )
        if (
            policy.fizzle_bars > 0
            and bars >= policy.fizzle_bars
            and mfe < policy.fizzle_mfe_atr * self.entry_atr
        ):
            return state, "fizzle"
        if policy.max_hold_bars > 0 and bars >= policy.max_hold_bars:
            return state, "max_hold"
        return state, None

    def as_payload(self) -> dict[str, object]:
        return {
            **asdict(self),
            "mfe_points": self.mfe_points,
            "mae_points": self.mae_points,
        }


def reconcile_spot_entry_basis(
    *,
    previous_qty: float,
    previous_basis_price: float | None,
    fill_delta_qty: float = 0.0,
    fill_price: float | None = None,
    broker_qty: float | None = None,
    broker_average_cost: float | None = None,
) -> SpotEntryBasisState:
    epsilon = 1e-12

    def _number(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    previous_quantity = _number(previous_qty) or 0.0
    previous_basis = _number(previous_basis_price)
    if previous_basis is not None and previous_basis <= 0.0:
        previous_basis = None

    authoritative_quantity = _number(broker_qty)
    authoritative_basis = _number(broker_average_cost)
    if authoritative_basis is not None and authoritative_basis <= 0.0:
        authoritative_basis = None
    if authoritative_quantity is not None:
        if abs(authoritative_quantity) <= epsilon:
            return SpotEntryBasisState(quantity=0.0, basis_price=None, source="flat")
        if authoritative_basis is not None:
            return SpotEntryBasisState(
                quantity=float(authoritative_quantity),
                basis_price=float(authoritative_basis),
                source="broker_average_cost",
            )
        return SpotEntryBasisState(
            quantity=float(authoritative_quantity),
            basis_price=float(previous_basis) if previous_basis is not None else None,
            source="fill" if previous_basis is not None else "broker_position",
        )

    delta = _number(fill_delta_qty) or 0.0
    execution_basis = _number(fill_price)
    if execution_basis is not None and execution_basis <= 0.0:
        execution_basis = None
    next_quantity = float(previous_quantity + delta)

    if abs(delta) <= epsilon:
        if abs(previous_quantity) <= epsilon:
            return SpotEntryBasisState(quantity=0.0, basis_price=None, source="flat")
        return SpotEntryBasisState(
            quantity=float(previous_quantity),
            basis_price=float(previous_basis) if previous_basis is not None else None,
            source="fill" if previous_basis is not None else "unknown",
        )
    if abs(next_quantity) <= epsilon:
        return SpotEntryBasisState(quantity=0.0, basis_price=None, source="flat")
    if abs(previous_quantity) <= epsilon:
        return SpotEntryBasisState(
            quantity=float(next_quantity),
            basis_price=float(execution_basis) if execution_basis is not None else None,
            source="fill",
        )

    same_direction = (previous_quantity > 0.0 and next_quantity > 0.0) or (
        previous_quantity < 0.0 and next_quantity < 0.0
    )
    if not same_direction:
        return SpotEntryBasisState(
            quantity=float(next_quantity),
            basis_price=float(execution_basis) if execution_basis is not None else None,
            source="fill",
        )

    previous_abs = abs(float(previous_quantity))
    next_abs = abs(float(next_quantity))
    if next_abs > previous_abs:
        added_abs = float(next_abs - previous_abs)
        if previous_basis is not None and execution_basis is not None:
            weighted_basis = (
                (float(previous_basis) * previous_abs)
                + (float(execution_basis) * added_abs)
            ) / next_abs
        elif execution_basis is not None:
            weighted_basis = float(execution_basis)
        else:
            weighted_basis = previous_basis
        return SpotEntryBasisState(
            quantity=float(next_quantity),
            basis_price=float(weighted_basis) if weighted_basis is not None else None,
            source="fill",
        )

    return SpotEntryBasisState(
        quantity=float(next_quantity),
        basis_price=float(previous_basis) if previous_basis is not None else None,
        source="fill",
    )


@dataclass(frozen=True)
class SpotPendingMutationPlan:
    clear_entry: bool = False
    clear_exit: bool = False
    queue_intent: str | None = None
    queue_direction: str | None = None
    queue_reason: str | None = None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def plan_pending_mutation(
    decision: SpotLifecycleDecision,
    *,
    pending_entry_direction: str | None,
    pending_exit_reason: str | None,
    open_dir: str | None,
) -> SpotPendingMutationPlan:
    intent = str(decision.intent or "").strip().lower()
    if intent == "enter":
        queue_intent = "enter"
        queue_direction = (
            str(decision.direction)
            if decision.direction in ("up", "down")
            else str(pending_entry_direction)
            if pending_entry_direction in ("up", "down")
            else None
        )
        queue_reason = str(decision.reason or "next_open")
    elif intent == "exit":
        queue_intent = "exit"
        queue_direction = (
            str(decision.direction)
            if decision.direction in ("up", "down")
            else str(open_dir)
            if open_dir in ("up", "down")
            else None
        )
        queue_reason = str(decision.reason or pending_exit_reason or "flip")
    else:
        queue_intent = None
        queue_direction = None
        queue_reason = None

    return SpotPendingMutationPlan(
        clear_entry=bool(decision.pending_clear_entry),
        clear_exit=bool(decision.pending_clear_exit),
        queue_intent=queue_intent,
        queue_direction=queue_direction,
        queue_reason=queue_reason,
    )


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
    signal_source_dir: str | None = None,
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
    capture_trace: bool = True,
) -> SpotLifecycleDecision:
    graph = policy_graph or SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    exit_pick = graph.resolve_exit_reason(
        strategy=strategy,
        open_dir=open_dir,
        signal_entry_dir=(
            signal_source_dir
            if signal_source_dir in ("up", "down")
            else signal_entry_dir
        ),
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
            else SPOT_FILL_MODE_NEXT_TRADABLE_BAR
            if exit_reason in ("fizzle", "max_hold")
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
            trace=(
                {
                    "stage": "open",
                    "path": "exit",
                    "exit_reason": str(exit_reason),
                    "fill_mode": fill_mode,
                    "controlled_flip": bool(queue_reentry_dir is not None),
                    "source_reversal": signal_source_dir,
                    "admitted_reentry": queue_reentry_dir,
                    "exit_policy": exit_pick.as_payload(),
                }
                if capture_trace
                else {}
            ),
        )

    if target_qty is None:
        return SpotLifecycleDecision(
            intent="hold",
            reason="holding_no_resize_target",
            gate="HOLDING",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            trace={"stage": "open", "path": "hold"} if capture_trace else {},
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
    adaptive = dict(resize_target.trace) if capture_trace else {}
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
                    trace=(
                        {
                            "stage": "open",
                            "path": "resize",
                            "cooldown_bars": int(cooldown),
                            "elapsed_bars": int(elapsed),
                            "resize_policy": adaptive,
                        }
                        if capture_trace
                        else {}
                    ),
                )
        return SpotLifecycleDecision(
            intent="resize",
            reason=str(spot_intent.reason or "target_delta"),
            gate="TRIGGER_RESIZE",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace=(
                {"stage": "open", "path": "resize", "resize_policy": adaptive}
                if capture_trace
                else {}
            ),
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
            trace=(
                {"stage": "open", "path": "enter", "resize_policy": adaptive}
                if capture_trace
                else {}
            ),
        )

    if str(spot_intent.intent) == "exit":
        return SpotLifecycleDecision(
            intent="exit",
            reason=str(spot_intent.reason or "target_zero"),
            gate="TRIGGER_EXIT",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace=(
                {
                    "stage": "open",
                    "path": "exit_from_intent",
                    "resize_policy": adaptive,
                }
                if capture_trace
                else {}
            ),
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
        trace=(
            {"stage": "open", "path": "hold", "resize_policy": adaptive}
            if capture_trace
            else {}
        ),
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
    entry_day_ok: bool = True,
    filter_checks: Mapping[str, bool] | None = None,
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
    capture_trace: bool = True,
) -> SpotLifecycleDecision:
    flat_trace: dict[str, object] = {}
    if capture_trace:
        flat_trace = {"stage": "flat", "bar_ts": bar_ts.isoformat()}
        if isinstance(entry_context, Mapping) and isinstance(
            entry_context.get("entry_control"), Mapping
        ):
            flat_trace["entry_control"] = dict(entry_context["entry_control"])
        if filter_checks is not None:
            checks = {
                str(name): bool(passed)
                for name, passed in filter_checks.items()
            }
            flat_trace["filter_checks"] = checks
            flat_trace["failed_filters"] = [
                name for name, passed in checks.items() if not passed
            ]
    if bool(stale_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="stale_signal",
            gate="BLOCKED_STALE_SIGNAL",
            blocked=True,
            trace=flat_trace,
        )
    if bool(gap_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="data_gap",
            gate="WAITING_DATA_GAP",
            blocked=True,
            trace=flat_trace,
        )
    if not bool(preflight_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="preflight",
            gate="WAITING_PREFLIGHT_BARS",
            blocked=True,
            trace=flat_trace,
        )
    if bool(pending_exists):
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_order",
            gate="PENDING_ORDER",
            blocked=True,
            trace=flat_trace,
        )
    if not bool(can_order_now):
        return SpotLifecycleDecision(
            intent="hold",
            reason="weekday_gate",
            gate="BLOCKED_WEEKDAY_NOW",
            blocked=True,
            trace=flat_trace,
        )
    if not bool(entry_day_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="entry_day",
            gate="BLOCKED_ENTRY_DAY",
            blocked=True,
            trace=flat_trace,
        )
    if not bool(entry_capacity):
        return SpotLifecycleDecision(
            intent="hold",
            reason="entry_limit",
            gate="BLOCKED_ENTRY_LIMIT",
            blocked=True,
            trace=flat_trace,
        )
    if entry_dir not in ("up", "down"):
        return SpotLifecycleDecision(
            intent="hold",
            reason="waiting_signal",
            gate="WAITING_SIGNAL",
            blocked=True,
            trace=flat_trace,
        )
    if str(entry_dir) not in {str(d) for d in allowed_directions}:
        return SpotLifecycleDecision(
            intent="hold",
            reason="direction_blocked",
            gate="BLOCKED_DIRECTION",
            direction=str(entry_dir),
            blocked=True,
            trace=flat_trace,
        )
    if not bool(filters_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="filters",
            gate="BLOCKED_FILTERS",
            direction=str(entry_dir),
            blocked=True,
            trace=flat_trace,
        )
    if not bool(atr_ready):
        return SpotLifecycleDecision(
            intent="hold",
            reason="atr_not_ready",
            gate="BLOCKED_ATR_NOT_READY",
            direction=str(entry_dir),
            blocked=True,
            trace=flat_trace,
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
            trace=flat_trace,
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
    graph_payload = entry_gate.as_payload() if capture_trace else None
    if not bool(entry_gate.allow):
        return SpotLifecycleDecision(
            intent="hold",
            reason=str(entry_gate.reason or "graph_entry_gate"),
            gate=str(entry_gate.gate or "BLOCKED_GRAPH_ENTRY"),
            direction=str(entry_dir) if entry_dir in ("up", "down") else None,
            fill_mode=str(fill_mode),
            blocked=True,
            trace=(
                {
                    **flat_trace,
                    "fill_mode": str(fill_mode),
                    "graph_entry": graph_payload,
                }
                if capture_trace
                else {}
            ),
        )
    return SpotLifecycleDecision(
        intent="enter",
        reason="entry",
        gate="TRIGGER_ENTRY",
        direction=str(entry_dir),
        fill_mode=str(fill_mode),
        blocked=False,
        trace=(
            {
                **flat_trace,
                "fill_mode": str(fill_mode),
                "graph_entry": graph_payload,
            }
            if capture_trace
            else {}
        ),
    )
