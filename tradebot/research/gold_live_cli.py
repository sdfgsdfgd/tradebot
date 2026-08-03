"""Advance the sole durable broker owner for a selected Stage-76 gold run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..client import IBKRClient
from ..config import auxiliary_client_config, load_config
from ..live.capital import load_live_capital_plan
from ..live.capital_packages import load_allocated_live_selection
from .gold_live_runtime import (
    advance_gold_live_transport,
    latest_gold_source_checkpoint,
)
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_LEDGER_PATH,
    GOLD_LIVE_SELECTION_PATH,
    load_gold_live_selection,
)
from .live_calibration import LiveCalibrationLedger


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile one selected one-contract Stage-76 canary."
    )
    parser.add_argument("--ledger", default=str(GOLD_LIVE_LEDGER_PATH))
    parser.add_argument("--selection", default=str(GOLD_LIVE_SELECTION_PATH))
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    args = parser.parse_args(argv)
    selection_path = Path(args.selection).expanduser()
    capital_path = Path(args.capital_plan).expanduser()
    capital_plan = load_live_capital_plan(capital_path)
    if capital_plan.get("schema") == "live.capital-plan.v3":
        _, selection_path, _ = load_allocated_live_selection(
            capital_plan,
            sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
            repository_root=Path(__file__).resolve().parents[2],
        )
    ledger = LiveCalibrationLedger(Path(args.ledger).expanduser())
    selection = load_gold_live_selection(selection_path)
    records = tuple(ledger.records())
    source = latest_gold_source_checkpoint(records)
    config = auxiliary_client_config(load_config(), 92)
    if config.readonly:
        raise ValueError("selected gold worker requires explicit broker authority")
    client = IBKRClient(config)
    try:
        output = await advance_gold_live_transport(
            ledger,
            client=client,
            selection=selection,
            source_checkpoint=source,
            capital_plan=capital_plan,
            selection_file_sha256=hashlib.sha256(
                selection_path.read_bytes()
            ).hexdigest(),
            observed_at=datetime.now(timezone.utc),
        )
    finally:
        await client.disconnect()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
