"""Canonical durable live-strategy bindings.

Product workers remain responsible for signal, selection, execution, and
reconciliation.  This registry only tells the shared portfolio cockpit how to
validate and observe those persistent owners.
"""

from __future__ import annotations

from pathlib import Path

from ..research.gold_live_transport import (
    GOLD_LIVE_EXECUTION_VERSION,
    load_gold_live_selection_from_mapping,
)
from ..research.gold_regime_harmony import GOLD_REGIME_HARMONY_VERSION
from ..research.xsp_live_transport import XSP_V3_TRANSPORT_EXECUTION_VERSION
from ..research.xsp_live_transport_v3 import (
    load_xsp_v3_transport_selection_from_mapping,
)
from ..research.xsp_dual_clock import XSP_DUAL_CLOCK_VERSION
from ..research.live_graduation import validate_live_graduation_receipt
from ..research.mcl_live_transport import (
    MCL_LIVE_EXECUTION_VERSION,
    load_mcl_live_selection_from_mapping,
)
from ..research.mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION
from .portfolio import LivePortfolioCockpit
from .runs import LiveRunBinding


LIVE_STRATEGY_BINDINGS = (
    LiveRunBinding(
        strategy_id=XSP_DUAL_CLOCK_VERSION,
        label="XSP P-009 Dual Clock · UPRO/SPXU RTH",
        execution_strategy_version=XSP_V3_TRANSPORT_EXECUTION_VERSION,
        ledger_path="db/calibration/xsp_live_calibration.jsonl",
        timer_unit="tradebot-xsp-shadow.timer",
        service_unit="tradebot-xsp-shadow.service",
        selection_validator=load_xsp_v3_transport_selection_from_mapping,
        champion_symbol="XSP",
        champion_track="LF",
    ),
    LiveRunBinding(
        strategy_id=GOLD_REGIME_HARMONY_VERSION,
        label="1OZ Stage 76 Regime Harmony · GTH/24x7 transport",
        execution_strategy_version=GOLD_LIVE_EXECUTION_VERSION,
        ledger_path="db/calibration/gold_live_calibration.jsonl",
        timer_unit="tradebot-gold-live.timer",
        service_unit="tradebot-gold-live.service",
        selection_validator=load_gold_live_selection_from_mapping,
        champion_symbol="1OZ",
        champion_track="LF",
    ),
    LiveRunBinding(
        strategy_id=MCL_TWO_SPEED_SHOCK_VERSION,
        label="MCL Stage 112 Two-Speed Shock Arbiter · CL/MCL GTH",
        execution_strategy_version=MCL_LIVE_EXECUTION_VERSION,
        ledger_path="db/calibration/mcl_live_calibration.jsonl",
        timer_unit="tradebot-mcl-live.timer",
        service_unit="tradebot-mcl-live.service",
        selection_validator=load_mcl_live_selection_from_mapping,
        champion_symbol="MCL",
        champion_track="HF",
    ),
)


def build_live_portfolio_cockpit(repository_root: Path) -> LivePortfolioCockpit:
    """Build the one account portfolio owner used by q and remote operators."""

    root = repository_root.resolve()
    return LivePortfolioCockpit(
        repository_root=root,
        capital_plan_path=root / "db/calibration/live_capital_plan.json",
        bindings=LIVE_STRATEGY_BINDINGS,
        graduation_directory=Path("db/calibration/live_graduation"),
        graduation_validator=validate_live_graduation_receipt,
        control_ledger_path=Path("db/calibration/live_control.jsonl"),
    )
