"""Content-addressed owner surfaces for XSP runtime parity proofs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


XSP_RUNTIME_PARITY_SCHEMA = "xsp.opening-edge-v3-current-runtime-parity-audit.v1"
XSP_P009_RUNTIME_PARITY_SCHEMA = (
    "xsp.opening-edge-v4-dual-clock-p009-current-runtime-parity-audit.v1"
)
XSP_P009_RUNTIME_PARITY_PATH = Path(
    "backtests/xsp/opening_edge_v4_dual_clock_arbitration_p009_"
    "current_runtime_parity_recovery_control_plane_20260809.json"
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


def xsp_p009_result_provenance_valid(
    *,
    repository_root: Path,
    actual: object,
    inputs: object,
) -> bool:
    """Bind reported P-009 economics to its crown and durable replay."""
    try:
        root = repository_root.resolve()
        crown = json.loads(
            (root / _XSP_P009_RUNTIME_OWNER_PATHS["frozen_crown_artifact_sha256"])
            .read_text()
        )
        candidate = crown["candidate"]
        durable_ref = crown["proofs"]["durable_owner_parity"]
        durable_relative = Path(str(durable_ref["path"]))
        durable_path = (root / durable_relative).resolve()
        if durable_relative.is_absolute() or root not in durable_path.parents:
            return False
        durable_raw = durable_path.read_bytes()
        durable = json.loads(durable_raw)
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return bool(
        isinstance(actual, Mapping)
        and isinstance(inputs, Mapping)
        and isinstance(candidate, Mapping)
        and isinstance(durable_ref, Mapping)
        and isinstance(durable, Mapping)
        and hashlib.sha256(durable_raw).hexdigest() == durable_ref.get("sha256")
        and durable.get("exact_parity") is True
        and durable.get("submitted_orders") == 0
        and durable.get("durable_owner_sha256")
        == inputs.get("durable_owner_sha256")
        and candidate.get("trades") == actual.get("full_combined_trades")
        and candidate.get("full_rth_trades") == actual.get("full_rth_trades")
        and candidate.get("full_combined_ledger_sha256")
        == actual.get("full_combined_ledger_sha256")
        and candidate.get("full_rth_ledger_sha256")
        == actual.get("full_rth_ledger_sha256")
        and durable.get("full_combined_trade_count")
        == actual.get("full_combined_trades")
        and durable.get("full_rth_trade_count") == actual.get("full_rth_trades")
        and durable.get("full_combined_ledger_sha256")
        == actual.get("full_combined_ledger_sha256")
        and durable.get("full_rth_ledger_sha256")
        == actual.get("full_rth_ledger_sha256")
    )
