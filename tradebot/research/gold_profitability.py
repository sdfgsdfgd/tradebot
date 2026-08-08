"""Cutoff-bound profitability and graduation evidence for selected 1OZ runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..live.capital_stability import portfolio_capital_owner_stability_gate
from ..live.order_evidence import single_contract_execution_graduation_gate
from .gold_live_transport import (
    GOLD_LIVE_CAPITAL_SLEEVE,
    GOLD_LIVE_EXECUTION_VERSION,
    GOLD_LIVE_MAX_COMMISSION_USD,
    GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
    GOLD_REGIME_HARMONY_VERSION,
    load_gold_live_selection_from_mapping,
)
from .gold_runtime_parity_contract import (
    GOLD_RUNTIME_PARITY_AUTHORITY,
    GOLD_RUNTIME_PARITY_REQUIRED_GATES,
    GOLD_RUNTIME_PARITY_SCHEMA,
)
from .live_graduation import live_calibration_logical_prefix
from .live_futures_profitability import (
    FuturesProfitabilitySpec,
    selected_futures_rows,
    single_contract_profitability_receipt,
    single_contract_restart_gate,
    single_contract_risk_gates,
)
from .live_futures_profitability_epoch import (
    build_futures_profitability_coverage_epoch,
    load_futures_profitability_coverage_epoch,
)


GOLD_LIVE_PROFITABILITY_SCHEMA = "gold.live-profitability.v1"
GOLD_TIMER_MINUTES = frozenset(range(2, 60, 5))
CHICAGO = ZoneInfo("America/Chicago")


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("gold profitability timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(
    status: str,
    reasons: Sequence[str],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def gold_1oz_maintenance(at: datetime | str) -> bool:
    """Return CME's regular 1OZ maintenance state in America/Chicago time."""

    local = _utc(at).astimezone(CHICAGO)
    minute = local.hour * 60 + local.minute
    if local.weekday() == 5:
        return 2 * 60 <= minute < 4 * 60
    return local.weekday() < 5 and 16 * 60 <= minute < 16 * 60 + 2


def gold_1oz_evaluation_slots(
    start_utc: datetime | str,
    end_utc: datetime | str,
    *,
    include_start: bool = False,
) -> tuple[datetime, ...]:
    """Project the five-minute worker clock, excluding published maintenance."""

    start = _utc(start_utc)
    end = _utc(end_utc)
    if end < start:
        raise ValueError("gold evidence interval regressed")
    cursor = start.replace(second=0, microsecond=0)
    while cursor.minute not in GOLD_TIMER_MINUTES or cursor < start:
        cursor += timedelta(minutes=1)
    output = []
    while cursor <= end:
        if (include_start or cursor > start) and not gold_1oz_maintenance(cursor):
            output.append(cursor)
        cursor += timedelta(minutes=5)
    return tuple(output)


def _gold_slots(
    start: datetime, end: datetime, include_start: bool
) -> tuple[datetime, ...]:
    return gold_1oz_evaluation_slots(start, end, include_start=include_start)


def _gold_excluded_slots(start: datetime, end: datetime) -> int:
    elapsed = max(0.0, (end - start).total_seconds())
    return sum(
        gold_1oz_maintenance(start + timedelta(minutes=5 * index))
        for index in range(max(0, int(elapsed // 300) + 1))
    )


def _gold_direction(position: float) -> str | None:
    return "up" if position > 0 else "down" if position < 0 else None


def _gold_spec(slot_tolerance_seconds: float = 90.0) -> FuturesProfitabilitySpec:
    return FuturesProfitabilitySpec(
        receipt_schema=GOLD_LIVE_PROFITABILITY_SCHEMA,
        authority="selected_reconciled_gold_risk_state_only",
        strategy_id=GOLD_REGIME_HARMONY_VERSION,
        strategy_version=GOLD_LIVE_EXECUTION_VERSION,
        capital_sleeve=GOLD_LIVE_CAPITAL_SLEEVE,
        symbol="1OZ",
        max_drawdown_usd=GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
        slot_tolerance_seconds=slot_tolerance_seconds,
        evaluation_slots=_gold_slots,
        natural_slot=lambda stamp: stamp.minute in GOLD_TIMER_MINUTES,
        held_direction=_gold_direction,
        excluded_clock_field="maintenance_slots_excluded",
        excluded_slots=_gold_excluded_slots,
    )


def gold_live_profitability_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    selection: Mapping[str, object],
    as_of: datetime | str,
    slot_tolerance_seconds: float = 90.0,
    coverage_epoch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    return single_contract_profitability_receipt(
        records,
        selection_id=str(selected["selection_id"]),
        run_started_at=str(selected["run_started_at_utc"]),
        con_id=int(selected["contract"]["con_id"]),
        spec=_gold_spec(slot_tolerance_seconds),
        as_of=as_of,
        coverage_epoch=coverage_epoch,
    )


def build_gold_profitability_coverage_epoch(
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
    selected = load_gold_live_selection_from_mapping(selection)
    return build_futures_profitability_coverage_epoch(
        selection_id=str(selected["selection_id"]),
        selection_path=selection_path,
        records=records,
        spec=_gold_spec(),
        con_id=int(selected["contract"]["con_id"]),
        predecessor_receipt_paths=predecessor_receipt_paths,
        preregistration_path=preregistration_path,
        registered_at_utc=registered_at_utc,
        eligible_start_utc=eligible_start_utc,
        repo_root=repo_root,
    )


def load_gold_profitability_coverage_epoch(
    path: Path,
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    return load_futures_profitability_coverage_epoch(
        path,
        selection_id=str(selected["selection_id"]),
        selection_path=selection_path,
        records=records,
        spec=_gold_spec(),
        con_id=int(selected["contract"]["con_id"]),
        repo_root=repo_root,
    )


def gold_runtime_parity_graduation_gate(
    path: Path,
    *,
    repo_root: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable Stage-76 crown/runtime proof and current owners."""

    try:
        raw = path.read_bytes()
        proof = json.loads(raw)
        selected = load_gold_live_selection_from_mapping(selection)
        if not isinstance(proof, Mapping):
            raise ValueError("gold runtime proof must be an object")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _gate(
            "INVALID",
            ["runtime_parity_proof_unreadable"],
            {"path": str(path), "error": str(exc)},
        )
    reasons: list[str] = []
    digest = hashlib.sha256(raw).hexdigest()
    frozen = selected.get("evidence")
    frozen_runtime = frozen.get("runtime_parity") if isinstance(frozen, Mapping) else None
    crown = proof.get("crown")
    selected_crown = frozen.get("crown") if isinstance(frozen, Mapping) else None
    if (
        proof.get("schema") != GOLD_RUNTIME_PARITY_SCHEMA
        or proof.get("authority") != GOLD_RUNTIME_PARITY_AUTHORITY
        or not isinstance(frozen_runtime, Mapping)
        or frozen_runtime.get("sha256") != digest
        or not isinstance(crown, Mapping)
        or not isinstance(selected_crown, Mapping)
        or any(
            crown.get(field) != selected_crown.get(field)
            for field in (
                "strategy_version",
                "strategy_key",
                "declaration_path",
                "declaration_sha256",
                "artifact_path",
                "artifact_sha256",
            )
        )
    ):
        reasons.append("runtime_parity_identity_invalid")
    root = repo_root.resolve()
    owners = proof.get("owners")
    if not isinstance(owners, Mapping) or not owners:
        reasons.append("runtime_owner_surface_missing")
    else:
        for owner in owners.values():
            try:
                if not isinstance(owner, Mapping):
                    raise ValueError
                relative = Path(str(owner["path"]))
                current = (root / relative).resolve()
                if relative.is_absolute() or root not in current.parents:
                    raise ValueError
                if _sha256(current) != owner.get("sha256"):
                    reasons.append(f"runtime_owner_drift:{relative}")
            except (KeyError, OSError, TypeError, ValueError):
                reasons.append("runtime_owner_invalid")
    context = proof.get("context_parity")
    exact_context = {
        "daily_hard_direction_and_age_exact",
        "h4_fast_slope_spread_velocity_acceleration_and_atr_exact",
        "macro_5_21_63_direction_velocity_acceleration_exact",
    }
    if not isinstance(context, Mapping) or any(
        context.get(key) is not True for key in exact_context
    ):
        reasons.append("runtime_context_parity_invalid")
    historical = proof.get("historical_parity")
    for name in ("full_three_year", "full_ten_year"):
        window = historical.get(name) if isinstance(historical, Mapping) else None
        try:
            valid = bool(
                isinstance(window, Mapping)
                and window.get("converged") is True
                and int(window["trades"]) > 0
                and float(window["net_pnl"]) > 0
                and float(window["profit_factor"]) > 1
                and len(str(window["ledger_sha256"])) == 64
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            reasons.append(f"runtime_{name}_invalid")
    prefix = proof.get("prospective_prefix")
    if (
        not isinstance(prefix, Mapping)
        or prefix.get("cold_replay_equal") is not True
        or prefix.get("converged") is not True
        or prefix.get("synthetic_midcycle_entry_authority") != "none"
        or prefix.get("order_authority") != "none"
        or prefix.get("submitted_orders") != 0
    ):
        reasons.append("runtime_cold_restart_parity_invalid")
    gates = proof.get("gates")
    if not isinstance(gates, Mapping) or any(
        gates.get(key) != "PASS" for key in GOLD_RUNTIME_PARITY_REQUIRED_GATES
    ):
        reasons.append("runtime_signal_gates_invalid")
    return _gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "path": str(path),
            "sha256": digest,
            "three_year_ledger_sha256": (
                historical.get("full_three_year", {}).get("ledger_sha256")
                if isinstance(historical, Mapping)
                else None
            ),
            "ten_year_ledger_sha256": (
                historical.get("full_ten_year", {}).get("ledger_sha256")
                if isinstance(historical, Mapping)
                else None
            ),
        },
    )


def _gold_restart_gate(
    selection: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return single_contract_restart_gate(
        selected_at_utc=str(selection["selected_at_utc"]),
        rows=rows,
        con_id=int(selection["contract"]["con_id"]),
        spec=_gold_spec(),
    )


def _gold_risk_gates(
    selection: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    return single_contract_risk_gates(
        rows=rows,
        con_id=int(selection["contract"]["con_id"]),
        spec=_gold_spec(),
    )


def gold_live_graduation_inputs(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    cutoff_utc: datetime | str,
    profitability_receipt: Mapping[str, object],
    runtime_parity_path: Path,
    capital_owner_stability_path: Path,
    repo_root: Path,
    coverage_epoch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project Gold raw truth into the shared live-graduation reducer."""

    selected = load_gold_live_selection_from_mapping(selection)
    cutoff = _utc(cutoff_utc)
    selection_sha = _sha256(selection_path)
    prefix, projected = live_calibration_logical_prefix(records, cutoff_utc=cutoff)
    rows = selected_futures_rows(
        projected,
        selection_id=str(selected["selection_id"]),
        spec=_gold_spec(),
    )
    restart = _gold_restart_gate(selected, rows)
    risk, attribution = _gold_risk_gates(selected, rows)
    broker = selected.get("broker_at_selection")
    account_id = str(broker.get("account_id") or "") if isinstance(broker, Mapping) else ""
    subject = {
        "strategy_id": GOLD_REGIME_HARMONY_VERSION,
        "strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "signal_instrument": "XAUUSD/GC",
        "execution_sleeve": "1OZ",
        "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
        "selection_id": selected["selection_id"],
        "run_id": selected["selection_id"],
        "account_fingerprint": hashlib.sha256(account_id.encode()).hexdigest(),
    }
    epoch_identity = (
        {
            "coverage_epoch_id": coverage_epoch["epoch_id"],
            "coverage_started_at_utc": coverage_epoch["eligible_start_utc"],
        }
        if coverage_epoch is not None
        else {}
    )
    return {
        "subject": subject,
        "selection": {
            "path": str(selection_path),
            "selection_id": selected["selection_id"],
            "run_id": selected["selection_id"],
            "selected_at_utc": selected["selected_at_utc"],
            "run_started_at_utc": selected["run_started_at_utc"],
            "signal_strategy_version": selected["strategy_version"],
            "execution_strategy_version": GOLD_LIVE_EXECUTION_VERSION,
            "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
            "selection_file_sha256": selection_sha,
            **epoch_identity,
        },
        "selection_file_sha256": selection_sha,
        "ledger_prefix": {
            **prefix,
            **epoch_identity,
            "gates": {
                "restart": restart,
                "cash_risk_safety": risk,
                "attribution": attribution,
                "execution": single_contract_execution_graduation_gate(
                    rows,
                    selection_id=str(selected["selection_id"]),
                    sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
                    symbol="1OZ",
                    con_id=int(selected["contract"]["con_id"]),
                    order_ref_prefix="GOLD76",
                    ladder_schema="gold.execution-ladder-transition.v1",
                    max_commission_usd=GOLD_LIVE_MAX_COMMISSION_USD,
                ),
            },
        },
        "profitability_receipt": dict(profitability_receipt),
        "runtime_parity_proof": gold_runtime_parity_graduation_gate(
            runtime_parity_path, repo_root=repo_root, selection=selected
        ),
        "capital_owner_stability_proof": portfolio_capital_owner_stability_gate(
            capital_owner_stability_path,
            repo_root=repo_root,
            sleeve_id=GOLD_LIVE_CAPITAL_SLEEVE,
            selection_id=str(selected["selection_id"]),
            selection_file_sha256=selection_sha,
        ),
    }
