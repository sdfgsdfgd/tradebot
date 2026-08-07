"""Prospective MCL profitability epochs without mutating the live owner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from ..live.capital import load_live_capital_plan
from ..live.capital_packages import load_allocated_live_selection
from ..live.capital_stability import PORTFOLIO_CAPITAL_STABILITY_PATH
from .live_calibration import LiveCalibrationLedger
from .live_futures_profitability import single_contract_profitability_receipt
from .live_futures_profitability_epoch import (
    build_futures_profitability_coverage_epoch,
    load_futures_profitability_coverage_epoch,
)
from .live_graduation import (
    live_calibration_logical_prefix,
    publish_live_graduation_receipt,
    reduce_live_graduation,
)
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_LEDGER_PATH,
    load_mcl_live_selection_from_mapping,
)
from .mcl_profitability import _spec, mcl_live_graduation_inputs


def _identity(
    selection: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    selected = load_mcl_live_selection_from_mapping(selection)
    return selected, int(selected["contracts"]["MCL"]["con_id"])


def mcl_profitability_receipt_with_coverage_epoch(
    records: Sequence[Mapping[str, object]],
    *,
    selection: Mapping[str, object],
    as_of: datetime | str,
    coverage_epoch: Mapping[str, object],
) -> dict[str, object]:
    """Reduce cumulative MCL economics from one immutable fresh clock."""

    selected, con_id = _identity(selection)
    return single_contract_profitability_receipt(
        records,
        selection_id=str(selected["selection_id"]),
        run_started_at=str(selected["run_started_at_utc"]),
        con_id=con_id,
        spec=_spec(str(selected["strategy_version"]), con_id),
        as_of=as_of,
        coverage_epoch=coverage_epoch,
    )


def build_mcl_profitability_coverage_epoch(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    predecessor_receipt_paths: Sequence[Path],
    preregistration_path: Path,
    registered_at_utc: datetime | str,
    eligible_start_utc: datetime | str,
    repo_root: Path,
) -> dict[str, object]:
    selected, con_id = _identity(selection)
    return build_futures_profitability_coverage_epoch(
        selection_id=str(selected["selection_id"]),
        selection_path=selection_path,
        records=records,
        spec=_spec(str(selected["strategy_version"]), con_id),
        con_id=con_id,
        predecessor_receipt_paths=predecessor_receipt_paths,
        preregistration_path=preregistration_path,
        registered_at_utc=registered_at_utc,
        eligible_start_utc=eligible_start_utc,
        repo_root=repo_root,
    )


def load_mcl_profitability_coverage_epoch(
    path: Path,
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    selected, con_id = _identity(selection)
    return load_futures_profitability_coverage_epoch(
        path,
        selection_id=str(selected["selection_id"]),
        selection_path=selection_path,
        records=records,
        spec=_spec(str(selected["strategy_version"]), con_id),
        con_id=con_id,
        repo_root=repo_root,
    )


def mcl_graduation_inputs_with_coverage_epoch(
    *,
    coverage_epoch: Mapping[str, object],
    **kwargs: object,
) -> dict[str, object]:
    """Bind the epoch identity to the unchanged MCL graduation projection."""

    inputs = mcl_live_graduation_inputs(**kwargs)
    epoch_identity = {
        "coverage_epoch_id": coverage_epoch["epoch_id"],
        "coverage_started_at_utc": coverage_epoch["eligible_start_utc"],
    }
    return {
        **inputs,
        "selection": {**inputs["selection"], **epoch_identity},
        "ledger_prefix": {**inputs["ledger_prefix"], **epoch_identity},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce one immutable MCL profitability coverage epoch."
    )
    parser.add_argument("--ledger", default=str(MCL_LIVE_LEDGER_PATH))
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    parser.add_argument(
        "--graduation-target",
        required=True,
        choices=("24h", "48h", "five-session"),
    )
    parser.add_argument("--graduation-cutoff", required=True)
    parser.add_argument("--graduation-coverage-epoch", required=True)
    parser.add_argument(
        "--graduation-capital-stability",
        default=PORTFOLIO_CAPITAL_STABILITY_PATH.as_posix(),
    )
    parser.add_argument("--graduation-output", required=True)
    args = parser.parse_args(argv)

    cutoff = datetime.fromisoformat(
        str(args.graduation_cutoff).replace("Z", "+00:00")
    )
    if cutoff.tzinfo is None:
        raise ValueError("graduation cutoff must be timezone-aware")
    root = Path(__file__).resolve().parents[2]
    records = tuple(
        LiveCalibrationLedger(Path(args.ledger).expanduser()).records()
    )
    selection, selection_path, _ = load_allocated_live_selection(
        load_live_capital_plan(Path(args.capital_plan).expanduser().resolve()),
        sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
        repository_root=root,
    )
    selected = load_mcl_live_selection_from_mapping(selection)
    epoch = load_mcl_profitability_coverage_epoch(
        Path(args.graduation_coverage_epoch).expanduser(),
        selection=selected,
        selection_path=selection_path,
        records=records,
        repo_root=root,
    )
    _, projected = live_calibration_logical_prefix(records, cutoff_utc=cutoff)
    profitability = mcl_profitability_receipt_with_coverage_epoch(
        projected,
        selection=selected,
        as_of=cutoff,
        coverage_epoch=epoch,
    )
    inputs = mcl_graduation_inputs_with_coverage_epoch(
        coverage_epoch=epoch,
        selection=selected,
        selection_path=selection_path,
        records=records,
        cutoff_utc=cutoff,
        profitability_receipt=profitability,
        capital_owner_stability_path=Path(
            args.graduation_capital_stability
        ).expanduser(),
        repo_root=root,
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


if __name__ == "__main__":
    raise SystemExit(main())
