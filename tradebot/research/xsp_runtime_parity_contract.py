"""Content-addressed owner surfaces for XSP runtime parity proofs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


XSP_RUNTIME_PARITY_SCHEMA = "xsp.opening-edge-v3-current-runtime-parity-audit.v1"
XSP_P009_RUNTIME_PARITY_SCHEMA = (
    "xsp.opening-edge-v4-dual-clock-p009-current-runtime-parity-audit.v1"
)
XSP_P009_RUNTIME_PARITY_PATH = Path(
    "backtests/xsp/opening_edge_v4_dual_clock_arbitration_p009_"
    "current_runtime_parity_q_native_gateway_20260809.json"
)
_XSP_RUNTIME_OWNER_PATHS = {
    "frozen_crown_artifact_sha256": (
        "backtests/xsp/opening_edge_v3_regime_harmony_24x5.json"
    ),
    "production_state_owner_sha256": (
        "tradebot/research/xsp_opening_edge_state.py"
    ),
    "production_spec_owner_sha256": "tradebot/research/xsp_opening_edge_v3.py",
    "production_v2_lane_owner_sha256": "tradebot/research/xsp_opening_edge_v2.py",
}
_XSP_P009_RUNTIME_OWNER_PATHS = {
    "frozen_crown_artifact_sha256": (
        "backtests/xsp/opening_edge_v4_dual_clock_arbitration_p009_crown.json"
    ),
    "durable_owner_sha256": "tradebot/research/xsp_dual_clock.py",
    "production_source_owner_sha256": "tradebot/research/xsp_opening_edge_v3.py",
    "production_source_adapter_sha256": "tradebot/research/xsp_opening_edge_v2.py",
    "production_minute_context_sha256": (
        "tradebot/research/xsp_opening_edge_minute_context.py"
    ),
    "production_run_start_owner_sha256": (
        "tradebot/research/xsp_opening_edge_run_start.py"
    ),
    "production_state_owner_sha256": "tradebot/research/xsp_opening_edge_state.py",
    "production_tape_owner_sha256": "tradebot/research/xsp_opening_edge_tapes.py",
    "execution_context_owner_sha256": "tradebot/research/xsp_context.py",
    "execution_observer_owner_sha256": (
        "tradebot/research/xsp_execution_observer.py"
    ),
    "transport_plan_owner_sha256": "tradebot/research/xsp_live_transport.py",
    "transport_runtime_owner_sha256": (
        "tradebot/research/xsp_live_transport_runtime.py"
    ),
    "transport_selection_owner_sha256": (
        "tradebot/research/xsp_live_transport_v3.py"
    ),
    "transport_allocation_owner_sha256": (
        "tradebot/research/xsp_live_transport_allocation.py"
    ),
    "transport_risk_owner_sha256": (
        "tradebot/research/xsp_live_transport_risk.py"
    ),
    "transport_state_owner_sha256": (
        "tradebot/research/xsp_live_transport_state.py"
    ),
    "live_cli_owner_sha256": "tradebot/research/xsp_shadow_cli.py",
    "pressure_accumulator_owner_sha256": (
        "tradebot/research/xsp_pressure_accumulator.py"
    ),
    "profitability_adapter_sha256": "tradebot/research/xsp_profitability.py",
    "runtime_parity_contract_sha256": (
        "tradebot/research/xsp_runtime_parity_contract.py"
    ),
    "live_registry_owner_sha256": "tradebot/live/strategies.py",
    "shared_execution_policy_sha256": "tradebot/engines/execution.py",
    "shared_live_execution_sha256": "tradebot/live/execution.py",
    "service_unit_sha256": "deploy/systemd/tradebot-xsp-shadow.service",
    "timer_unit_sha256": "deploy/systemd/tradebot-xsp-shadow.timer",
}


def xsp_runtime_parity_owner_paths(schema: object) -> Mapping[str, str]:
    if schema == XSP_P009_RUNTIME_PARITY_SCHEMA:
        return _XSP_P009_RUNTIME_OWNER_PATHS
    return _XSP_RUNTIME_OWNER_PATHS
