"""Shared spot policy kernel for live UI and backtests."""

from .lifecycle import (
    SpotLifecycleDecision,
    adaptive_resize_target_qty,
    apply_regime_gate,
    decide_flat_position_intent,
    decide_open_position_intent,
    decide_pending_next_open,
    flip_exit_hit,
    entry_capacity_ok,
    flip_exit_gate_blocked,
    next_open_entry_allowed,
    permission_gate_status,
    pick_exit_reason,
    signal_filter_checks,
    signal_filters_ok,
)
from .graph import (
    SpotGraphProfile,
    SpotPolicyGraph,
    canonical_exit_reason,
    all_graph_profiles,
)
from .policy import SpotDecisionTrace, SpotIntentDecision, SpotPolicy, SpotPolicyConfigView, SpotRuntimeSpec
from .scenario import lifecycle_trace_row, why_not_exit_resize_report, write_rows_csv

__all__ = [
    "SpotPolicy",
    "SpotPolicyConfigView",
    "SpotRuntimeSpec",
    "SpotDecisionTrace",
    "SpotIntentDecision",
    "SpotLifecycleDecision",
    "SpotGraphProfile",
    "SpotPolicyGraph",
    "canonical_exit_reason",
    "all_graph_profiles",
    "apply_regime_gate",
    "entry_capacity_ok",
    "flip_exit_hit",
    "flip_exit_gate_blocked",
    "next_open_entry_allowed",
    "permission_gate_status",
    "pick_exit_reason",
    "signal_filter_checks",
    "signal_filters_ok",
    "adaptive_resize_target_qty",
    "decide_pending_next_open",
    "decide_open_position_intent",
    "decide_flat_position_intent",
    "lifecycle_trace_row",
    "why_not_exit_resize_report",
    "write_rows_csv",
]
