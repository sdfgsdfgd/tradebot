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
from ..live.capital_stability import PORTFOLIO_CAPITAL_STABILITY_PATH
from .gold_live_runtime import (
    advance_gold_live_transport,
    latest_gold_source_checkpoint,
)
from .gold_profitability import (
    gold_live_graduation_inputs,
    gold_live_profitability_receipt,
)
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_LEDGER_PATH,
    GOLD_LIVE_SELECTION_PATH,
    load_gold_live_selection,
)
from .live_calibration import LiveCalibrationLedger
from .live_graduation import (
    live_calibration_logical_prefix,
    publish_live_graduation_receipt,
    reduce_live_graduation,
)


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile one selected one-contract Stage-76 canary."
    )
    parser.add_argument("--ledger", default=str(GOLD_LIVE_LEDGER_PATH))
    parser.add_argument("--selection", default=str(GOLD_LIVE_SELECTION_PATH))
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    parser.add_argument(
        "--graduation-target", choices=("24h", "48h", "five-session")
    )
    parser.add_argument("--graduation-cutoff")
    parser.add_argument(
        "--graduation-runtime-parity",
        default=(
            "backtests/gold/"
            "one_oz_regime_harmony_runtime_parity_20260803.json"
        ),
    )
    parser.add_argument(
        "--graduation-capital-stability",
        default=PORTFOLIO_CAPITAL_STABILITY_PATH.as_posix(),
    )
    parser.add_argument("--graduation-output")
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
    graduation_requested = any(
        (args.graduation_target, args.graduation_cutoff, args.graduation_output)
    )
    if graduation_requested:
        if not all(
            (args.graduation_target, args.graduation_cutoff, args.graduation_output)
        ):
            raise ValueError("graduation requires target, cutoff, and output")
        cutoff = datetime.fromisoformat(
            str(args.graduation_cutoff).replace("Z", "+00:00")
        )
        if cutoff.tzinfo is None:
            raise ValueError("graduation cutoff must be timezone-aware")
        _, graduation_records = live_calibration_logical_prefix(
            records, cutoff_utc=cutoff
        )
        profitability = gold_live_profitability_receipt(
            graduation_records, selection=selection, as_of=cutoff
        )
        inputs = gold_live_graduation_inputs(
            selection=selection,
            selection_path=selection_path,
            records=records,
            cutoff_utc=cutoff,
            profitability_receipt=profitability,
            runtime_parity_path=Path(args.graduation_runtime_parity).expanduser(),
            capital_owner_stability_path=Path(
                args.graduation_capital_stability
            ).expanduser(),
            repo_root=Path(__file__).resolve().parents[2],
        )
        receipt = reduce_live_graduation(
            target_milestone=args.graduation_target,
            cutoff_utc=cutoff,
            **inputs,
        )
        publish_live_graduation_receipt(
            Path(args.graduation_output).expanduser(), receipt
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 2 if receipt["verdict"] in {"QUARANTINE", "STOP"} else 0
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
