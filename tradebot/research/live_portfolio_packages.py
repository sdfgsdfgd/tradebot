"""Product adapters for the shared minimum-first live capital owner."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from ..live.capital import build_live_capital_plan_v3
from ..live.capital_packages import PACKAGE_FIRST_ADMITTER_METHOD
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_PACKAGE_SELECTION_SCHEMA,
    GOLD_LIVE_MIN_STRESS_BUFFER_AUD,
    load_gold_live_selection_from_mapping,
)
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_PACKAGE_ID,
    load_mcl_live_selection_from_mapping,
)
from .xsp_live_transport import (
    XSP_V3_PACKAGE_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
)
from .xsp_live_transport_allocation import (
    XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH,
    _package_cell,
    load_xsp_v3_package_selection_from_mapping,
    xsp_package_cash_debit_usd_cents,
)


def _identity(value: object, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} is not a SHA-256 identity")
    return text


def _selection_binding(
    selection: Mapping[str, object], *, path: str, sha256: str
) -> dict[str, object]:
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("selected-run path must be repository-relative")
    return {
        "run_id": _identity(selection.get("selection_id"), name="selected run"),
        "selection_path": path,
        "selection_file_sha256": _identity(sha256, name="selection file"),
    }


def xsp_portfolio_package_sleeve(
    selection: Mapping[str, object],
    *,
    selection_path: str,
    selection_file_sha256: str,
    repository_root: Path,
) -> dict[str, object]:
    selected = load_xsp_v3_package_selection_from_mapping(selection)
    if selected["schema"] != XSP_V3_PACKAGE_SELECTION_SCHEMA:
        raise ValueError("XSP package sleeve requires the package-sized selection")
    receipt_path = repository_root / XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH
    receipt = json.loads(receipt_path.read_text())
    cells = receipt.get("package_cells") if isinstance(receipt, Mapping) else None
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("XSP package curve is unavailable")
    ladder = []
    for rank, raw in enumerate(sorted(cells, key=lambda row: int(row["notional_usd"]))):
        package_id = f"xsp-usd-{int(raw['notional_usd'])}"
        cell = _package_cell(receipt, package_id)
        ladder.append(
            {
                "package_id": package_id,
                "rank": rank,
                "cash_debit_usd_cents": xsp_package_cash_debit_usd_cents(
                    int(cell["notional_usd"])
                ),
                "initial_margin_base_cents": 0,
                "maintenance_margin_base_cents": 0,
                "stressed_loss_usd_cents": math.ceil(
                    float(cell["max_single_position_mae_usd"]) * 100
                ),
                "fx_stress_bps": 11_000,
            }
        )
    return {
        "sleeve_id": XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
        "strategy_id": str(selected["strategy_version"]),
        **_selection_binding(
            selected, path=selection_path, sha256=selection_file_sha256
        ),
        "capital_kind": "CASH_DEBIT",
        "position_symbols": ["UPRO", "SPXU"],
        "residual_weight_bps": 10_000,
        "minimum_package_id": ladder[0]["package_id"],
        "package_ladder": ladder,
    }


def gold_portfolio_package_sleeve(
    selection: Mapping[str, object],
    *,
    selection_path: str,
    selection_file_sha256: str,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    if selected["schema"] != GOLD_LIVE_PACKAGE_SELECTION_SCHEMA:
        raise ValueError("Gold package sleeve requires the package-sized selection")
    risk = selected["risk"]
    assert isinstance(risk, Mapping)
    return {
        "sleeve_id": GOLD_LIVE_CAPITAL_SLEEVE,
        "strategy_id": str(selected["strategy_version"]),
        **_selection_binding(
            selected, path=selection_path, sha256=selection_file_sha256
        ),
        "capital_kind": "FUTURES_MARGIN",
        "position_symbols": ["1OZ"],
        "residual_weight_bps": 0,
        "minimum_package_id": "gold-one-contract",
        "package_ladder": [
            {
                "package_id": "gold-one-contract",
                "rank": 0,
                "cash_debit_usd_cents": 66,
                "initial_margin_base_cents": math.ceil(
                    float(risk["max_initial_margin_change_aud"]) * 100
                ),
                "maintenance_margin_base_cents": math.ceil(
                    float(risk["max_maintenance_margin_change_aud"]) * 100
                ),
                "stressed_loss_usd_cents": math.ceil(
                    float(risk["max_open_position_stress_usd"]) * 100
                ),
                "fx_stress_bps": int(risk["fx_stress_bps"]),
            }
        ],
    }


def mcl_portfolio_package_sleeve(
    selection: Mapping[str, object],
    *,
    selection_path: str,
    selection_file_sha256: str,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    risk = selected["risk"]
    assert isinstance(risk, Mapping)
    return {
        "sleeve_id": MCL_LIVE_CAPITAL_SLEEVE,
        "strategy_id": str(selected["strategy_version"]),
        **_selection_binding(
            selected, path=selection_path, sha256=selection_file_sha256
        ),
        "capital_kind": "FUTURES_MARGIN",
        "position_symbols": ["MCL"],
        "residual_weight_bps": 0,
        "minimum_package_id": MCL_LIVE_PACKAGE_ID,
        "package_ladder": [
            {
                "package_id": MCL_LIVE_PACKAGE_ID,
                "rank": 0,
                "cash_debit_usd_cents": 76,
                "initial_margin_base_cents": math.ceil(
                    float(risk["max_initial_margin_change_aud"]) * 100
                ),
                "maintenance_margin_base_cents": math.ceil(
                    float(risk["max_maintenance_margin_change_aud"]) * 100
                ),
                "stressed_loss_usd_cents": math.ceil(
                    float(risk["package_stressed_loss_usd"]) * 100
                ),
                "fx_stress_bps": int(risk["fx_stress_bps"]),
            }
        ],
    }


def build_xsp_gold_portfolio_package_plan(
    *,
    xsp_selection: Mapping[str, object],
    gold_selection: Mapping[str, object],
    xsp_selection_path: str,
    xsp_selection_file_sha256: str,
    gold_selection_path: str,
    gold_selection_file_sha256: str,
    account_resources: Mapping[str, object],
    repository_root: Path,
    created_at_utc: datetime | str,
    supersedes_plan_id: str,
) -> dict[str, object]:
    """Bind current product proofs to one general minimum-first account plan."""

    sleeves = [
        xsp_portfolio_package_sleeve(
            xsp_selection,
            selection_path=xsp_selection_path,
            selection_file_sha256=xsp_selection_file_sha256,
            repository_root=repository_root,
        ),
        gold_portfolio_package_sleeve(
            gold_selection,
            selection_path=gold_selection_path,
            selection_file_sha256=gold_selection_file_sha256,
        ),
    ]
    plan = build_live_capital_plan_v3(
        account_id=str(account_resources.get("account_id") or ""),
        account_type=str(account_resources.get("account_type") or ""),
        cash_currency="USD",
        base_currency=str(account_resources.get("base_currency") or ""),
        observed_settled_cash_usd=account_resources.get("settled_cash_usd"),
        observed_available_funds_base=account_resources.get(
            "available_funds_base"
        ),
        observed_excess_liquidity_base=account_resources.get(
            "excess_liquidity_base"
        ),
        usd_to_base_rate=account_resources.get("usd_to_base_rate"),
        minimum_post_reservation_base=GOLD_LIVE_MIN_STRESS_BUFFER_AUD,
        unmanaged_position_stress_base=account_resources.get(
            "unmanaged_position_stress_base"
        ),
        sleeves=sleeves,
        reserve_reasons=[
            "cash_outside_allocated_executable_packages",
            "minimum_packages_reserved_before_weighted_residual",
            "unmanaged_positions_receive_full_gross_stress",
        ],
        created_at_utc=created_at_utc,
        supersedes_plan_id=supersedes_plan_id,
    )
    expected = {
        XSP_V3_TRANSPORT_CAPITAL_SLEEVE: xsp_selection[
            "allocation_successor"
        ]["package_id"],
        GOLD_LIVE_CAPITAL_SLEEVE: gold_selection["allocation_successor"][
            "package_id"
        ],
    }
    actual = {
        str(sleeve["sleeve_id"]): str(sleeve["allocated_package_id"])
        for sleeve in plan["sleeves"]
    }
    if actual != expected:
        raise ValueError("selected runs and allocated executable packages disagree")
    return plan


def build_xsp_gold_mcl_portfolio_package_plan(
    *,
    xsp_selection: Mapping[str, object],
    gold_selection: Mapping[str, object],
    mcl_selection: Mapping[str, object],
    xsp_selection_path: str,
    xsp_selection_file_sha256: str,
    gold_selection_path: str,
    gold_selection_file_sha256: str,
    mcl_selection_path: str,
    mcl_selection_file_sha256: str,
    account_resources: Mapping[str, object],
    repository_root: Path,
    created_at_utc: datetime | str,
    supersedes_plan_id: str,
) -> dict[str, object]:
    """Bind three immutable runs while reserving resources only at admission."""

    xsp = xsp_portfolio_package_sleeve(
        xsp_selection,
        selection_path=xsp_selection_path,
        selection_file_sha256=xsp_selection_file_sha256,
        repository_root=repository_root,
    )
    selected_xsp_package = str(
        xsp_selection["allocation_successor"]["package_id"]
    )
    xsp.update(
        minimum_package_id=selected_xsp_package,
        residual_weight_bps=0,
    )
    sleeves = [
        xsp,
        gold_portfolio_package_sleeve(
            gold_selection,
            selection_path=gold_selection_path,
            selection_file_sha256=gold_selection_file_sha256,
        ),
        mcl_portfolio_package_sleeve(
            mcl_selection,
            selection_path=mcl_selection_path,
            selection_file_sha256=mcl_selection_file_sha256,
        ),
    ]
    plan = build_live_capital_plan_v3(
        account_id=str(account_resources.get("account_id") or ""),
        account_type=str(account_resources.get("account_type") or ""),
        cash_currency="USD",
        base_currency=str(account_resources.get("base_currency") or ""),
        observed_settled_cash_usd=account_resources.get("settled_cash_usd"),
        observed_available_funds_base=account_resources.get(
            "available_funds_base"
        ),
        observed_excess_liquidity_base=account_resources.get(
            "excess_liquidity_base"
        ),
        usd_to_base_rate=account_resources.get("usd_to_base_rate"),
        minimum_post_reservation_base=GOLD_LIVE_MIN_STRESS_BUFFER_AUD,
        unmanaged_position_stress_base=account_resources.get(
            "unmanaged_position_stress_base"
        ),
        sleeves=sleeves,
        reserve_reasons=[
            "cash_outside_individually_executable_packages",
            "first_lawful_admitter_reserves_its_indivisible_minimum",
            "later_entries_use_only_fresh_remaining_resources",
            "unmanaged_positions_receive_full_gross_stress",
        ],
        created_at_utc=created_at_utc,
        supersedes_plan_id=supersedes_plan_id,
        entry_capacity_policy=PACKAGE_FIRST_ADMITTER_METHOD,
    )
    expected = {
        XSP_V3_TRANSPORT_CAPITAL_SLEEVE: selected_xsp_package,
        GOLD_LIVE_CAPITAL_SLEEVE: str(
            gold_selection["allocation_successor"]["package_id"]
        ),
        MCL_LIVE_CAPITAL_SLEEVE: MCL_LIVE_PACKAGE_ID,
    }
    actual = {
        str(sleeve["sleeve_id"]): str(sleeve["allocated_package_id"])
        for sleeve in plan["sleeves"]
    }
    if actual != expected:
        raise ValueError("three selected runs changed executable package identity")
    return plan
