"""Commission or advance the sole selected MCL bounded canary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..client import IBKRClient
from ..config import auxiliary_client_config, load_config
from ..live.capital import load_live_capital_plan, publish_live_capital_plan
from ..live.capital_packages import (
    load_allocated_live_selection,
    publish_immutable_live_selection,
)
from ..live.capital_stability import (
    PORTFOLIO_CAPITAL_STABILITY_PATH,
    publish_portfolio_capital_owner_stability,
    publish_portfolio_package_generation,
)
from .live_calibration import LiveCalibrationLedger
from .live_graduation import (
    live_calibration_logical_prefix,
    publish_live_graduation_receipt,
    reduce_live_graduation,
)
from .live_portfolio_packages import build_xsp_gold_mcl_portfolio_package_plan
from .mcl_live import (
    advance_mcl_live_transport,
)
from .mcl_live_reopen import bind_mcl_maintenance_reopen_selection
from .mcl_stage131 import (
    MCL_STAGE131_BINDING_KEY,
    MCL_STAGE131_COVERAGE_PATH,
    bind_mcl_stage131_selection,
    load_mcl_stage131_context,
)
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_EXECUTION_VERSION,
    MCL_LIVE_LEDGER_PATH,
    build_mcl_live_selection,
    capture_mcl_commissioning_preview,
    load_mcl_live_selection_from_mapping,
)
from .mcl_predictive_accumulator import (
    MCL_PREDICTIVE_LEDGER_PATH,
    mcl_predictive_treatments,
)
from .mcl_predictive_generation import (
    build_mcl_predictive_successor_generation,
    publish_mcl_predictive_generation,
)
from .mcl_profitability import (
    mcl_live_graduation_inputs,
    mcl_live_profitability_receipt,
)
from .mcl_shock_wave_accumulator import (
    MCL_SHOCK_WAVE_LEDGER_PATH,
    mcl_shock_wave_episodes,
)
from .mcl_shock_wave_generation import (
    MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH,
    build_mcl_shock_wave_successor_generation,
    publish_mcl_shock_wave_generation,
)
from .gold_live_transport import GOLD_LIVE_CAPITAL_SLEEVE
from .xsp_live_transport import XSP_V3_TRANSPORT_CAPITAL_SLEEVE


def _unmanaged_stress(
    broker: dict[str, object], *, owned_symbols: set[str]
) -> float:
    return sum(
        int(row["market_value_base_cents"]) / 100
        for row in broker["positions"]
        if str(row.get("symbol") or "").upper() not in owned_symbols
    )


async def _commission(
    *,
    client: IBKRClient,
    ledger: LiveCalibrationLedger,
    capital_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    predecessor = load_live_capital_plan(capital_path)
    if predecessor.get("schema") != "live.capital-plan.v3":
        raise ValueError("MCL commissioning requires the current v3 account plan")
    now = datetime.now(timezone.utc)
    preview = await capture_mcl_commissioning_preview(
        client, repository_root=repository_root, observed_at=now
    )
    selected = bind_mcl_stage131_selection(
        bind_mcl_maintenance_reopen_selection(
            build_mcl_live_selection(
                repository_root=repository_root,
                preview=preview,
                selected_at=datetime.now(timezone.utc),
            ),
            repository_root=repository_root,
        ),
        repository_root=repository_root,
    )
    mcl_path, mcl_sha = publish_immutable_live_selection(repository_root, selected)
    predictive_records = tuple(
        LiveCalibrationLedger(MCL_PREDICTIVE_LEDGER_PATH).records()
    )
    predictive = build_mcl_predictive_successor_generation(
        repository_root=repository_root,
        selected=selected,
        inherited_treatment_ids=[
            str(row["treatment_id"])
            for row in mcl_predictive_treatments(predictive_records)
        ],
        generated_at=datetime.now(timezone.utc),
    )
    predictive_path, predictive_sha = publish_mcl_predictive_generation(
        repository_root, predictive
    )
    xsp, xsp_path, xsp_sha = load_allocated_live_selection(
        predecessor,
        sleeve_id=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
        repository_root=repository_root,
    )
    gold, gold_path, gold_sha = load_allocated_live_selection(
        predecessor,
        sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
        repository_root=repository_root,
    )
    broker = preview["broker"]
    assert isinstance(broker, dict)
    plan = build_xsp_gold_mcl_portfolio_package_plan(
        xsp_selection=xsp,
        gold_selection=gold,
        mcl_selection=selected,
        xsp_selection_path=xsp_path.relative_to(repository_root).as_posix(),
        xsp_selection_file_sha256=xsp_sha,
        gold_selection_path=gold_path.relative_to(repository_root).as_posix(),
        gold_selection_file_sha256=gold_sha,
        mcl_selection_path=mcl_path,
        mcl_selection_file_sha256=mcl_sha,
        account_resources={
            "account_id": broker["account_id"],
            "account_type": broker["account_type"],
            "base_currency": broker["base_currency"],
            "settled_cash_usd": broker["settled_cash_usd"],
            "available_funds_base": broker["available_funds_base"],
            "excess_liquidity_base": broker["excess_liquidity_base"],
            "usd_to_base_rate": broker["usd_to_base_rate"],
            "unmanaged_position_stress_base": _unmanaged_stress(
                broker,
                owned_symbols={"UPRO", "SPXU", "1OZ", "MCL"},
            ),
        },
        repository_root=repository_root,
        created_at_utc=selected["selected_at_utc"],
        supersedes_plan_id=str(predecessor["plan_id"]),
    )
    generation_path, generation_sha = publish_portfolio_package_generation(
        repository_root, plan
    )
    wave_episodes = mcl_shock_wave_episodes(
        tuple(LiveCalibrationLedger(MCL_SHOCK_WAVE_LEDGER_PATH).records())
    )
    wave = build_mcl_shock_wave_successor_generation(
        repository_root=repository_root,
        selected=selected,
        selection_path=mcl_path,
        selection_file_sha256=mcl_sha,
        capital_plan_id=str(plan["plan_id"]),
        portfolio_generation_path=generation_path,
        portfolio_generation_sha256=generation_sha,
        predictive=predictive,
        predictive_path=predictive_path,
        predictive_file_sha256=predictive_sha,
        inherited_episode_ids=[str(row["episode_id"]) for row in wave_episodes],
        generated_at=datetime.now(timezone.utc),
    )
    wave_path, wave_sha = publish_mcl_shock_wave_generation(
        repository_root, wave
    )
    stability_path, stability_sha = publish_portfolio_capital_owner_stability(
        repository_root,
        generation_path=generation_path,
        generation_sha256=generation_sha,
        observed_at_utc=selected["selected_at_utc"],
    )
    publish_live_capital_plan(capital_path, plan)
    receipt = {
        "schema": "mcl.two-speed-auction-live-commissioning.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_id": selected["selection_id"],
        "selection_path": mcl_path,
        "selection_file_sha256": mcl_sha,
        "capital_plan_id": plan["plan_id"],
        "portfolio_generation_path": generation_path,
        "portfolio_generation_sha256": generation_sha,
        "capital_stability_path": stability_path,
        "capital_stability_current_path": (
            PORTFOLIO_CAPITAL_STABILITY_PATH.as_posix()
        ),
        "capital_stability_sha256": stability_sha,
        "predictive_generation_id": predictive["generation_id"],
        "predictive_generation_path": predictive_path,
        "predictive_generation_sha256": predictive_sha,
        "predictive_inherited_treatments": predictive["inherited_prefix"][
            "treatment_count"
        ],
        "shock_wave_generation_id": wave["generation_id"],
        "shock_wave_generation_path": wave_path,
        "shock_wave_generation_sha256": wave_sha,
        "shock_wave_inherited_episodes": wave["inherited_prefix"][
            "episode_count"
        ],
        "stage131_runtime_binding_sha256": selected["evidence"][
            MCL_STAGE131_BINDING_KEY
        ]["sha256"],
        "supersedes_plan_id": predecessor["plan_id"],
        "preview_fingerprint": selected["allocation_successor"][
            "broker_preview_fingerprint"
        ],
        "order_authority": "armed_only_after_observe_only_restart_proof",
        "submitted_orders": 0,
    }
    ledger.checkpoint(
        evaluation_as_of=datetime.now(timezone.utc),
        strategy_id=selected["strategy_version"],
        strategy_version=MCL_LIVE_EXECUTION_VERSION,
        trading_date=datetime.now(timezone.utc).date().isoformat(),
        session="MCL_COMMISSIONING",
        status="EVALUATED",
        evidence=receipt,
        recorded_at=datetime.now(timezone.utc),
    )
    return receipt


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Commission or reconcile one selected MCL bounded canary."
    )
    parser.add_argument("--ledger", default=str(MCL_LIVE_LEDGER_PATH))
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    parser.add_argument("--commission", action="store_true")
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument(
        "--stage131-coverage",
        type=Path,
        default=Path(
            os.environ.get("MCL_STAGE131_COVERAGE", MCL_STAGE131_COVERAGE_PATH)
        ),
    )
    parser.add_argument(
        "--shock-wave-generation",
        type=Path,
        default=MCL_SHOCK_WAVE_RUNTIME_GENERATION_PATH,
    )
    parser.add_argument(
        "--shock-wave-ledger",
        type=Path,
        default=Path(
            os.environ.get("MCL_SHOCK_WAVE_LEDGER", MCL_SHOCK_WAVE_LEDGER_PATH)
        ),
    )
    parser.add_argument(
        "--graduation-target", choices=("24h", "48h", "five-session")
    )
    parser.add_argument("--graduation-cutoff")
    parser.add_argument(
        "--graduation-capital-stability",
        default=PORTFOLIO_CAPITAL_STABILITY_PATH.as_posix(),
    )
    parser.add_argument("--graduation-output")
    args = parser.parse_args(argv)
    if args.commission and args.observe_only:
        raise ValueError("commission and observe-only are separate phases")
    root = Path(__file__).resolve().parents[2]
    capital_path = Path(args.capital_plan).expanduser().resolve()
    ledger = LiveCalibrationLedger(Path(args.ledger).expanduser())
    graduation_requested = any(
        (args.graduation_target, args.graduation_cutoff, args.graduation_output)
    )
    if graduation_requested:
        if args.commission or args.observe_only or not all(
            (args.graduation_target, args.graduation_cutoff, args.graduation_output)
        ):
            raise ValueError(
                "graduation requires target, cutoff, and output as a read-only phase"
            )
        cutoff = datetime.fromisoformat(
            str(args.graduation_cutoff).replace("Z", "+00:00")
        )
        if cutoff.tzinfo is None:
            raise ValueError("graduation cutoff must be timezone-aware")
        plan = load_live_capital_plan(capital_path)
        selection, selection_path, _ = load_allocated_live_selection(
            plan,
            sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
            repository_root=root,
        )
        selected = load_mcl_live_selection_from_mapping(selection)
        records = tuple(ledger.records())
        _, graduation_records = live_calibration_logical_prefix(
            records, cutoff_utc=cutoff
        )
        profitability = mcl_live_profitability_receipt(
            graduation_records, selection=selected, as_of=cutoff
        )
        inputs = mcl_live_graduation_inputs(
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
        output = reduce_live_graduation(
            target_milestone=args.graduation_target,
            cutoff_utc=cutoff,
            **inputs,
        )
        publish_live_graduation_receipt(
            Path(args.graduation_output).expanduser(), output
        )
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 2 if output["verdict"] in {"QUARANTINE", "STOP"} else 0
    config = auxiliary_client_config(load_config(), 94)
    if config.readonly and not args.observe_only:
        raise ValueError("MCL commissioning/live worker requires broker what-if authority")
    client = IBKRClient(config)
    try:
        await client.connect()
        if args.commission:
            output = await _commission(
                client=client,
                ledger=ledger,
                capital_path=capital_path,
                repository_root=root,
            )
        else:
            plan = load_live_capital_plan(capital_path)
            selection, selection_path, selection_sha = load_allocated_live_selection(
                plan,
                sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
                repository_root=root,
            )
            selected = load_mcl_live_selection_from_mapping(selection)
            output = await advance_mcl_live_transport(
                ledger,
                client=client,
                selection=selected,
                capital_plan=plan,
                selection_file_sha256=selection_sha,
                observed_at=datetime.now(timezone.utc),
                observe_only=args.observe_only,
                stage131_context=load_mcl_stage131_context(
                    selected,
                    repository_root=root,
                    generation_path=args.shock_wave_generation,
                    coverage_path=args.stage131_coverage,
                    wave_ledger_path=args.shock_wave_ledger,
                ),
            )
    finally:
        await client.disconnect()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
