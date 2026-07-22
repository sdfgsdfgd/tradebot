"""Shared spot policy kernel for live UI and backtests."""

from .gates import (
    SpotDeferredEntryPlan,
    apply_regime_gate,
    deferred_entry_plan,
    fill_due_ts,
    flip_exit_allowed,
    flip_exit_hit,
    entry_capacity_ok,
    flip_exit_gate_blocked,
    next_open_entry_allowed,
    permission_gate_status,
    signal_filter_checks,
    signal_filters_ok,
)
from .lifecycle import (
    SpotLifecycleDecision,
    adaptive_resize_target_qty,
    decide_flat_position_intent,
    decide_open_position_intent,
    decide_pending_next_open,
    pick_exit_reason,
)
from .graph import SpotPolicyGraph
from .graph_core import (
    SpotGraphProfile,
    canonical_exit_reason,
    all_graph_profiles,
)
from .policy import SpotPolicy
from .policy_contract import SpotDecisionTrace, SpotIntentDecision, SpotPolicyConfigView, SpotRuntimeSpec
from .scenario import lifecycle_trace_row, why_not_exit_resize_report, write_rows_csv

__all__ = [
    "SpotPolicy",
    "SpotPolicyConfigView",
    "SpotRuntimeSpec",
    "SpotDecisionTrace",
    "SpotIntentDecision",
    "SpotDeferredEntryPlan",
    "SpotLifecycleDecision",
    "SpotGraphProfile",
    "SpotPolicyGraph",
    "canonical_exit_reason",
    "all_graph_profiles",
    "apply_regime_gate",
    "entry_capacity_ok",
    "flip_exit_allowed",
    "flip_exit_hit",
    "flip_exit_gate_blocked",
    "next_open_entry_allowed",
    "permission_gate_status",
    "pick_exit_reason",
    "signal_filter_checks",
    "signal_filters_ok",
    "adaptive_resize_target_qty",
    "fill_due_ts",
    "deferred_entry_plan",
    "decide_pending_next_open",
    "decide_open_position_intent",
    "decide_flat_position_intent",
    "lifecycle_trace_row",
    "why_not_exit_resize_report",
    "write_rows_csv",
]
