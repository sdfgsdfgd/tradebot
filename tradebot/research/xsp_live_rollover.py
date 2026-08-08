"""Atomically rebind a terminal-flat XSP P-009 transport generation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..client import IBKRClient
from ..config import auxiliary_client_config, load_config
from ..live.capital import load_live_capital_plan, publish_live_capital_plan
from ..live.capital_packages import (
    load_allocated_live_selection,
    publish_immutable_live_selection,
)
from ..live.capital_stability import (
    publish_portfolio_capital_owner_stability,
    publish_portfolio_package_generation,
)
from .gold_live_transport import GOLD_LIVE_CAPITAL_SLEEVE
from .live_calibration import LiveCalibrationLedger
from .live_portfolio_packages import build_xsp_gold_mcl_portfolio_package_plan
from .mcl_live_transport import MCL_LIVE_CAPITAL_SLEEVE
from .xsp_dual_clock import XSP_DUAL_CLOCK_SOURCE_VERSION, XSP_DUAL_CLOCK_VERSION
from .xsp_live_transport import XSP_V3_TRANSPORT_CAPITAL_SLEEVE
from .xsp_live_transport_allocation import (
    XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH,
    _pending_order_refs,
    reallocate_xsp_v3_transport,
    xsp_p009_crown_binding,
    xsp_portfolio_package_preview,
)
from .xsp_live_transport_risk import xsp_transport_risk_state
from .xsp_pressure_accumulator import (
    XSP_PRESSURE_ACCUMULATOR_GENERATION_PATH,
    XSP_PRESSURE_ACCUMULATOR_LEDGER_PATH,
    load_xsp_pressure_accumulator_generation,
    xsp_pressure_treatments,
)


XSP_P009_ROLLOVER_SCHEMA = "xsp.p009-flat-run-rollover.v1"
XSP_P009_PRESSURE_GENERATION_DIRECTORY = Path(
    "db/calibration/xsp_pressure_generations"
)
_OWNED_SYMBOLS = {"UPRO", "SPXU", "1OZ", "MCL"}
_ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("XSP rollover timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable XSP rollover artifact changed")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("immutable XSP rollover artifact changed")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_xsp_pressure_generation_successor(
    *,
    current: Mapping[str, object],
    current_sha256: str,
    current_archive_path: str,
    selection: Mapping[str, object],
    inherited_treatment_ids: Sequence[str],
    registered_at_utc: datetime,
) -> dict[str, object]:
    """Rebind the outcome-blind atlas without inheriting transport outcomes."""

    selection_id = str(selection.get("selection_id") or "")
    treatment_ids = sorted(set(str(value) for value in inherited_treatment_ids))
    if (
        current.get("schema")
        != "xsp.pressure-atlas-accumulation-generation.v1"
        or selection.get("strategy_version") != XSP_DUAL_CLOCK_VERSION
        or selection.get("source_strategy_version")
        != XSP_DUAL_CLOCK_SOURCE_VERSION
        or len(selection_id) != 64
        or len(str(current.get("generation_id") or "")) != 64
        or len(current_sha256) != 64
        or not current_archive_path
        or Path(current_archive_path).is_absolute()
        or ".." in Path(current_archive_path).parts
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in treatment_ids
        )
    ):
        raise ValueError("XSP pressure-generation successor inputs are invalid")
    artifacts = json.loads(json.dumps(current.get("artifacts")))
    if not isinstance(artifacts, dict):
        raise ValueError("XSP pressure-generation artifacts are invalid")
    artifacts["predecessor_generation"] = {
        "path": current_archive_path,
        "sha256": current_sha256,
    }
    registered = _utc(registered_at_utc)
    body = {
        **{
            key: json.loads(json.dumps(value))
            for key, value in current.items()
            if key
            not in {
                "generation_id",
                "registered_at_utc",
                "eligible_start_utc",
                "selection_id",
                "artifacts",
                "predecessor_generation_id",
                "inherited_treatment_ids",
            }
        },
        "registered_at_utc": registered.isoformat(),
        "eligible_start_utc": registered.isoformat(),
        "selection_id": selection_id,
        "artifacts": artifacts,
        "predecessor_generation_id": current["generation_id"],
        "inherited_treatment_ids": treatment_ids,
    }
    return {**body, "generation_id": _identity(body)}


def _publish_pressure_generation(
    *,
    repository_root: Path,
    selection: Mapping[str, object],
    treatment_ids: Sequence[str],
    registered_at_utc: datetime,
    current_path: Path = XSP_PRESSURE_ACCUMULATOR_GENERATION_PATH,
) -> tuple[dict[str, object], str, str]:
    root = repository_root.resolve()
    current_file = (root / current_path).resolve()
    current, current_sha = load_xsp_pressure_accumulator_generation(
        current_file, root=root
    )
    if current.get("selection_id") == selection.get("selection_id"):
        relative = (
            XSP_P009_PRESSURE_GENERATION_DIRECTORY
            / f"{current['generation_id']}.json"
        )
        _publish_immutable(root / relative, current_file.read_bytes())
        return current, relative.as_posix(), current_sha

    predecessor_relative = (
        XSP_P009_PRESSURE_GENERATION_DIRECTORY
        / f"{current['generation_id']}.json"
    )
    predecessor_payload = current_file.read_bytes()
    _publish_immutable(root / predecessor_relative, predecessor_payload)
    successor = build_xsp_pressure_generation_successor(
        current=current,
        current_sha256=current_sha,
        current_archive_path=predecessor_relative.as_posix(),
        selection=selection,
        inherited_treatment_ids=treatment_ids,
        registered_at_utc=registered_at_utc,
    )
    payload = json.dumps(
        successor, allow_nan=False, indent=2, sort_keys=True
    ).encode() + b"\n"
    successor_relative = (
        XSP_P009_PRESSURE_GENERATION_DIRECTORY
        / f"{successor['generation_id']}.json"
    )
    _publish_immutable(root / successor_relative, payload)
    temporary = current_file.with_name(f".{current_file.name}.validate")
    _atomic_write(temporary, payload)
    try:
        validated, digest = load_xsp_pressure_accumulator_generation(
            temporary, root=root
        )
    finally:
        temporary.unlink(missing_ok=True)
    if validated != successor:
        raise ValueError("XSP pressure-generation successor failed validation")
    _atomic_write(current_file, payload)
    return successor, successor_relative.as_posix(), digest


def _account_resources(broker: Mapping[str, object]) -> dict[str, object]:
    resources = broker.get("account_resources")
    positions = broker.get("account_positions")
    if (
        not isinstance(resources, Mapping)
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
    ):
        raise ValueError("XSP rollover account resources are unavailable")
    unmanaged_cents = sum(
        int(row["market_value_base_cents"])
        for row in positions
        if isinstance(row, Mapping)
        and abs(float(row.get("quantity") or 0)) > 1e-9
        and str(row.get("symbol") or "").upper() not in _OWNED_SYMBOLS
    )
    return {
        "account_id": broker["account_id"],
        "account_type": broker["account_type"],
        "base_currency": resources["base_currency"],
        "settled_cash_usd": broker["settled_cash_usd"],
        "available_funds_base": int(
            resources["available_funds_base_cents"]
        )
        / 100,
        "excess_liquidity_base": int(
            resources["excess_liquidity_base_cents"]
        )
        / 100,
        "usd_to_base_rate": int(resources["usd_to_base_rate_ppm"]) / 1_000_000,
        "unmanaged_position_stress_base": unmanaged_cents / 100,
    }


def _require_fresh_flat_successor(
    *,
    selection: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    broker: Mapping[str, object],
    observed_at: datetime,
) -> None:
    risk = xsp_transport_risk_state(
        selection=selection,
        records=records,
        observed_at=observed_at,
        liquidation_bids={},
    )
    if (
        broker.get("positions") != {"UPRO": 0.0, "SPXU": 0.0}
        or broker.get("open_orders") != []
        or risk["holdings_from_fills"] != {"UPRO": 0.0, "SPXU": 0.0}
        or risk["pending_settlement_usd"] != 0
        or risk["fill_count"] != 0
        or risk["closed_trades"] != 0
        or risk["safety_breaches"]
    ):
        raise ValueError("active XSP rollover successor is not fresh and flat")


def inspect_xsp_p009_rollover_boundary(
    *,
    selection: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    broker_snapshot: Mapping[str, object],
    observed_at: datetime,
    expected_predecessor_selection_id: str,
) -> dict[str, object]:
    """Classify whether the interrupted owner may remain active or must stop."""

    now = _utc(observed_at)
    selection_id = str(selection.get("selection_id") or "")
    if selection_id != expected_predecessor_selection_id:
        successor = selection.get("allocation_successor")
        if (
            selection.get("strategy_version") != XSP_DUAL_CLOCK_VERSION
            or selection.get("source_strategy_version")
            != XSP_DUAL_CLOCK_SOURCE_VERSION
            or not isinstance(successor, Mapping)
            or successor.get("predecessor_selection_id")
            != expected_predecessor_selection_id
        ):
            raise ValueError("active XSP selection crossed the rollover boundary")
        _require_fresh_flat_successor(
            selection=selection,
            records=records,
            broker=broker_snapshot,
            observed_at=now,
        )
        return {
            "schema": XSP_P009_ROLLOVER_SCHEMA,
            "status": "ALREADY_ROLLED_OVER",
            "selection_id": selection_id,
            "broker_flat": True,
            "ledger_flat": True,
            "incumbent_timer_may_remain_active": False,
            "submitted_orders": 0,
        }

    positions = broker_snapshot.get("positions")
    orders = broker_snapshot.get("open_orders")
    if (
        not isinstance(positions, Mapping)
        or set(positions) != {"UPRO", "SPXU"}
        or not isinstance(orders, Sequence)
        or isinstance(orders, (str, bytes))
    ):
        raise ValueError("XSP rollover preflight broker state is invalid")
    quantities = {symbol: float(positions[symbol]) for symbol in ("UPRO", "SPXU")}
    if any(value < 0 for value in quantities.values()) or sum(
        value > 1e-9 for value in quantities.values()
    ) > 1:
        raise ValueError("XSP rollover preflight position is ambiguous")
    broker_flat = all(abs(value) <= 1e-9 for value in quantities.values())
    relevant_orders = [
        row
        for row in orders
        if isinstance(row, Mapping)
        and str(row.get("symbol") or "").upper() in {"UPRO", "SPXU"}
    ]
    if not broker_flat:
        return {
            "schema": XSP_P009_ROLLOVER_SCHEMA,
            "status": (
                "INCUMBENT_REDUCTION_ACTIVE"
                if relevant_orders
                else "INCUMBENT_HELD"
            ),
            "selection_id": selection_id,
            "broker_flat": False,
            "ledger_flat": None,
            "incumbent_timer_may_remain_active": True,
            "submitted_orders": 0,
        }
    if orders:
        return {
            "schema": XSP_P009_ROLLOVER_SCHEMA,
            "status": "TERMINAL_FLAT_ORDER_RECONCILIATION_REQUIRED",
            "selection_id": selection_id,
            "broker_flat": True,
            "ledger_flat": None,
            "incumbent_timer_may_remain_active": False,
            "submitted_orders": 0,
        }

    # The bids are diagnostic placeholders used only to reconstruct fill-owned
    # quantities. Economics are consulted only after those quantities are flat.
    risk = xsp_transport_risk_state(
        selection=selection,
        records=records,
        observed_at=now,
        liquidation_bids={"UPRO": 1.0, "SPXU": 1.0},
    )
    ledger_flat = risk["holdings_from_fills"] == {"UPRO": 0.0, "SPXU": 0.0}
    pending_refs = _pending_order_refs(records, selection_id=selection_id)
    if not ledger_flat or pending_refs:
        status = "TERMINAL_FLAT_LEDGER_RECONCILIATION_REQUIRED"
    elif risk["pending_settlement_usd"] != 0 or risk["safety_breaches"]:
        status = "TERMINAL_FLAT_BLOCKED"
    else:
        status = "TERMINAL_FLAT_READY"
    return {
        "schema": XSP_P009_ROLLOVER_SCHEMA,
        "status": status,
        "selection_id": selection_id,
        "broker_flat": True,
        "ledger_flat": ledger_flat,
        "pending_order_refs": pending_refs,
        "fill_count": risk["fill_count"],
        "closed_trades": risk["closed_trades"],
        "incumbent_timer_may_remain_active": False,
        "submitted_orders": 0,
    }


def publish_xsp_p009_rollover(
    *,
    repository_root: Path,
    capital_plan_path: Path,
    ledger_path: Path,
    pressure_ledger_path: Path,
    preview: Mapping[str, object],
    broker_snapshot: Mapping[str, object],
    selected_at_utc: datetime,
    expected_predecessor_selection_id: str,
) -> dict[str, object]:
    """Publish one flat-only P-009 selection and its dependent generations."""

    root = repository_root.resolve()
    selected_at = _utc(selected_at_utc)
    predecessor_plan = load_live_capital_plan(capital_plan_path)
    predecessor, _predecessor_path, _predecessor_sha = (
        load_allocated_live_selection(
            predecessor_plan,
            sleeve_id=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
            repository_root=root,
        )
    )
    records = tuple(LiveCalibrationLedger(ledger_path).records())
    if predecessor.get("selection_id") != expected_predecessor_selection_id:
        successor = predecessor.get("allocation_successor")
        if (
            predecessor.get("strategy_version") != XSP_DUAL_CLOCK_VERSION
            or predecessor.get("source_strategy_version")
            != XSP_DUAL_CLOCK_SOURCE_VERSION
            or not isinstance(successor, Mapping)
            or successor.get("predecessor_selection_id")
            != expected_predecessor_selection_id
        ):
            raise ValueError("active XSP selection crossed the rollover boundary")
        _require_fresh_flat_successor(
            selection=predecessor,
            records=records,
            broker=broker_snapshot,
            observed_at=selected_at,
        )
        generation_path = (
            Path("db/calibration/portfolio_generations")
            / f"{predecessor_plan['plan_id']}.json"
        )
        generation_sha = _sha256(root / generation_path)
        stability_path, stability_sha = publish_portfolio_capital_owner_stability(
            root,
            generation_path=generation_path.as_posix(),
            generation_sha256=generation_sha,
            observed_at_utc=selected_at,
        )
        treatments = xsp_pressure_treatments(
            tuple(LiveCalibrationLedger(pressure_ledger_path).records())
        )
        pressure, pressure_path, pressure_sha = _publish_pressure_generation(
            repository_root=root,
            selection=predecessor,
            treatment_ids=[str(row["treatment_id"]) for row in treatments],
            registered_at_utc=selected_at,
        )
        return {
            "schema": XSP_P009_ROLLOVER_SCHEMA,
            "status": "ALREADY_ROLLED_OVER",
            "predecessor_selection_id": expected_predecessor_selection_id,
            "selection_id": predecessor["selection_id"],
            "capital_plan_id": predecessor_plan["plan_id"],
            "capital_stability_path": stability_path,
            "capital_stability_sha256": stability_sha,
            "pressure_generation_id": pressure["generation_id"],
            "pressure_generation_path": pressure_path,
            "pressure_generation_sha256": pressure_sha,
            "submitted_orders": 0,
        }

    package_id = str(predecessor["allocation_successor"]["package_id"])
    selection = reallocate_xsp_v3_transport(
        predecessor=predecessor,
        records=records,
        broker_snapshot=broker_snapshot,
        preview=preview,
        package_receipt_path=root / XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH,
        package_id=package_id,
        selected_at=selected_at,
        strategy_version=XSP_DUAL_CLOCK_VERSION,
        source_strategy_version=XSP_DUAL_CLOCK_SOURCE_VERSION,
        crown_evidence=xsp_p009_crown_binding(root),
    )
    selection_path, selection_sha = publish_immutable_live_selection(
        root, selection
    )
    gold, gold_path, gold_sha = load_allocated_live_selection(
        predecessor_plan,
        sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
        repository_root=root,
    )
    mcl, mcl_path, mcl_sha = load_allocated_live_selection(
        predecessor_plan,
        sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
        repository_root=root,
    )
    plan = build_xsp_gold_mcl_portfolio_package_plan(
        xsp_selection=selection,
        gold_selection=gold,
        mcl_selection=mcl,
        xsp_selection_path=selection_path,
        xsp_selection_file_sha256=selection_sha,
        gold_selection_path=gold_path.relative_to(root).as_posix(),
        gold_selection_file_sha256=gold_sha,
        mcl_selection_path=mcl_path.relative_to(root).as_posix(),
        mcl_selection_file_sha256=mcl_sha,
        account_resources=_account_resources(broker_snapshot),
        repository_root=root,
        created_at_utc=selected_at,
        supersedes_plan_id=str(predecessor_plan["plan_id"]),
    )
    generation_path, generation_sha = publish_portfolio_package_generation(
        root, plan
    )
    treatments = xsp_pressure_treatments(
        tuple(LiveCalibrationLedger(pressure_ledger_path).records())
    )
    pressure, pressure_path, pressure_sha = _publish_pressure_generation(
        repository_root=root,
        selection=selection,
        treatment_ids=[str(row["treatment_id"]) for row in treatments],
        registered_at_utc=selected_at,
    )
    publish_live_capital_plan(capital_plan_path, plan)
    stability_path, stability_sha = publish_portfolio_capital_owner_stability(
        root,
        generation_path=generation_path,
        generation_sha256=generation_sha,
        observed_at_utc=selected_at,
    )
    return {
        "schema": XSP_P009_ROLLOVER_SCHEMA,
        "status": "ROLLED_OVER",
        "predecessor_selection_id": predecessor["selection_id"],
        "selection_id": selection["selection_id"],
        "selection_path": selection_path,
        "selection_file_sha256": selection_sha,
        "retained_gold_selection_id": gold["selection_id"],
        "retained_mcl_selection_id": mcl["selection_id"],
        "capital_plan_id": plan["plan_id"],
        "portfolio_generation_path": generation_path,
        "portfolio_generation_sha256": generation_sha,
        "capital_stability_path": stability_path,
        "capital_stability_sha256": stability_sha,
        "pressure_generation_id": pressure["generation_id"],
        "pressure_generation_path": pressure_path,
        "pressure_generation_sha256": pressure_sha,
        "starting_cash_identity_usd": selection["risk"][
            "starting_cash_identity_usd"
        ],
        "submitted_orders": 0,
    }


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebind P-009 only after its selected cash run is terminal flat."
    )
    parser.add_argument("--expected-predecessor-selection-id", required=True)
    parser.add_argument(
        "--capital-plan", default="db/calibration/live_capital_plan.json"
    )
    parser.add_argument(
        "--ledger", default="db/calibration/xsp_live_calibration.jsonl"
    )
    parser.add_argument(
        "--pressure-ledger", default=str(XSP_PRESSURE_ACCUMULATOR_LEDGER_PATH)
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    root = _ROOT
    capital_path = Path(args.capital_plan).expanduser()
    plan = load_live_capital_plan(capital_path)
    current, _path, _sha = load_allocated_live_selection(
        plan,
        sleeve_id=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
        repository_root=root,
    )
    config = auxiliary_client_config(load_config(), 86)
    if config.readonly and not args.preflight_only:
        raise ValueError("XSP rollover requires nontransmitting what-if authority")
    client = IBKRClient(config)
    try:
        await client.connect()
        now = datetime.now(timezone.utc)
        from .xsp_live_transport_state import xsp_v2_broker_snapshot

        broker = await xsp_v2_broker_snapshot(
            client,
            symbols=("UPRO", "SPXU"),
            resource_base_currency=(None if args.preflight_only else "AUD"),
        )
        records = tuple(
            LiveCalibrationLedger(Path(args.ledger).expanduser()).records()
        )
        preflight = inspect_xsp_p009_rollover_boundary(
            selection=current,
            records=records,
            broker_snapshot=broker,
            observed_at=now,
            expected_predecessor_selection_id=(
                args.expected_predecessor_selection_id
            ),
        )
        if args.preflight_only:
            print(json.dumps(preflight, allow_nan=False, indent=2, sort_keys=True))
            return 0
        if current["selection_id"] == args.expected_predecessor_selection_id:
            if preflight["status"] != "TERMINAL_FLAT_READY":
                raise ValueError(
                    f"XSP rollover is not terminal-flat ready: {preflight['status']}"
                )
            package_id = str(current["allocation_successor"]["package_id"])
            preview, broker = await xsp_portfolio_package_preview(
                client,
                notional_usd=int(package_id.removeprefix("xsp-usd-")),
                observed_at=now,
            )
        else:
            preview = {}
        output = publish_xsp_p009_rollover(
            repository_root=root,
            capital_plan_path=capital_path,
            ledger_path=Path(args.ledger).expanduser(),
            pressure_ledger_path=Path(args.pressure_ledger).expanduser(),
            preview=preview,
            broker_snapshot=broker,
            selected_at_utc=datetime.now(timezone.utc),
            expected_predecessor_selection_id=(
                args.expected_predecessor_selection_id
            ),
        )
    finally:
        await client.disconnect()
    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
