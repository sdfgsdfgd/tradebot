"""Canonical durable live-strategy bindings.

Product workers remain responsible for signal, selection, execution, and
reconciliation.  This registry only tells the shared portfolio cockpit how to
validate and observe those persistent owners.
"""

from __future__ import annotations

from pathlib import Path

from ..research.xsp_live_transport import XSP_V3_TRANSPORT_EXECUTION_VERSION
from ..research.xsp_live_transport_v3 import (
    load_xsp_v3_transport_selection_from_mapping,
)
from ..research.xsp_opening_edge_v3 import XSP_OPENING_EDGE_V3_VERSION
from ..research.live_graduation import validate_live_graduation_receipt
from .portfolio import LivePortfolioCockpit
from .runs import LiveRunBinding


LIVE_STRATEGY_BINDINGS = (
    LiveRunBinding(
        strategy_id=XSP_OPENING_EDGE_V3_VERSION,
        label="XSP v3 Regime Harmony · UPRO/SPXU RTH",
        execution_strategy_version=XSP_V3_TRANSPORT_EXECUTION_VERSION,
        ledger_path="db/calibration/xsp_live_calibration.jsonl",
        timer_unit="tradebot-xsp-shadow.timer",
        service_unit="tradebot-xsp-shadow.service",
        selection_validator=load_xsp_v3_transport_selection_from_mapping,
        champion_symbol="XSP",
        champion_track="LF",
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
