"""Cutoff-bound profitability and graduation evidence for selected MCL V18."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..live.capital_stability import portfolio_capital_owner_stability_gate
from ..live.order_evidence import single_contract_execution_graduation_gate
from .live_futures_profitability import (
    FuturesProfitabilitySpec,
    selected_futures_rows,
    single_contract_profitability_receipt,
    single_contract_restart_gate,
    single_contract_risk_gates,
)
from .live_graduation import live_calibration_logical_prefix
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_EXECUTION_VERSION,
    MCL_LIVE_MAX_COMMISSION_USD,
    MCL_LIVE_MAX_RUN_DRAWDOWN_USD,
    MCL_LIVE_ORDER_REF_PREFIX,
    load_mcl_live_selection_from_mapping,
)
from .mcl_two_speed_auction import MCL_TWO_SPEED_AUCTION_VERSION


MCL_LIVE_PROFITABILITY_SCHEMA = "mcl.live-profitability.v1"
MCL_LIVE_SLOT_TOLERANCE_SECONDS = 55.0
_ET = ZoneInfo("America/New_York")
_EMPTY_FILLS_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("MCL profitability timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(
    status: str, reasons: Sequence[str], evidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "evidence": dict(evidence),
    }


def mcl_market_open(at: datetime | str) -> bool:
    """Return the exact timer-owned NYMEX minute session."""

    local = _utc(at).astimezone(_ET)
    weekday, hour = local.weekday(), local.hour
    return bool(
        (weekday == 6 and hour >= 18)
        or (weekday < 4 and hour != 17)
        or (weekday == 4 and hour <= 16)
    )


def mcl_live_evaluation_slots(
    start_utc: datetime | str,
    end_utc: datetime | str,
    include_start: bool = False,
) -> tuple[datetime, ...]:
    start, end = _utc(start_utc), _utc(end_utc)
    if end < start:
        raise ValueError("MCL evidence interval regressed")
    cursor = start.replace(second=0, microsecond=0)
    if cursor < start:
        cursor += timedelta(minutes=1)
    output = []
    while cursor <= end:
        if (include_start or cursor > start) and mcl_market_open(cursor):
            output.append(cursor)
        cursor += timedelta(minutes=1)
    return tuple(output)


def _excluded_slots(start: datetime, end: datetime) -> int:
    cursor = start.replace(second=0, microsecond=0)
    excluded = 0
    while cursor <= end:
        excluded += not mcl_market_open(cursor)
        cursor += timedelta(minutes=1)
    return excluded


def _direction(position: float) -> int | None:
    return 1 if position > 0 else -1 if position < 0 else None


def normalize_mcl_risk(risk: Mapping[str, object]) -> Mapping[str, object]:
    """Project only the pre-accounting flat prefix into exact zero economics."""

    if risk.get("valid") is True:
        return dict(risk)
    zero_legacy = bool(
        risk.get("schema") == "mcl.two-speed-auction-risk-state.v1"
        and float(risk.get("position_from_fills") or 0.0) == 0
        and risk.get("open_exec_id") is None
        and risk.get("entry_time_utc") is None
        and risk.get("entry_price") is None
        and float(risk.get("run_realized_net_usd") or 0.0) == 0
        and int(risk.get("closed_trades") or 0) == 0
        and float(risk.get("unrealized_raw_usd") or 0.0) == 0
        and risk.get("fill_ledger_fingerprint") == _EMPTY_FILLS_SHA256
        and risk.get("safety_breaches") == []
    )
    if not zero_legacy:
        raise ValueError("legacy MCL risk cannot be normalized")
    return {
        **dict(risk),
        "valid": True,
        "attribution_complete": True,
        "run_realized_gross_usd": 0.0,
        "run_realized_cost_usd": 0.0,
        "run_realized_net_usd": 0.0,
        "open_mark_gross_usd": 0.0,
        "open_mark_cost_usd": 0.0,
        "open_mark_net_usd": 0.0,
        "run_gross_usd": 0.0,
        "run_cost_usd": 0.0,
        "run_net_usd": 0.0,
        "peak_run_net_usd": 0.0,
        "drawdown_usd": 0.0,
        "gross_wins_usd": 0.0,
        "top_five_gross_wins_usd": 0.0,
        "fill_count": 0,
        "exit_triggers": [],
    }


def _spec() -> FuturesProfitabilitySpec:
    return FuturesProfitabilitySpec(
        receipt_schema=MCL_LIVE_PROFITABILITY_SCHEMA,
        authority="selected_reconciled_mcl_risk_state_only",
        strategy_id=MCL_TWO_SPEED_AUCTION_VERSION,
        strategy_version=MCL_LIVE_EXECUTION_VERSION,
        capital_sleeve=MCL_LIVE_CAPITAL_SLEEVE,
        symbol="MCL",
        max_drawdown_usd=MCL_LIVE_MAX_RUN_DRAWDOWN_USD,
        slot_tolerance_seconds=MCL_LIVE_SLOT_TOLERANCE_SECONDS,
        evaluation_slots=mcl_live_evaluation_slots,
        natural_slot=lambda stamp: (
            mcl_market_open(stamp)
            and stamp.second <= MCL_LIVE_SLOT_TOLERANCE_SECONDS
        ),
        held_direction=_direction,
        risk_projection=normalize_mcl_risk,
        excluded_clock_field="closed_minutes_excluded",
        excluded_slots=_excluded_slots,
    )


def mcl_live_profitability_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    selection: Mapping[str, object],
    as_of: datetime | str,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    return single_contract_profitability_receipt(
        records,
        selection_id=str(selected["selection_id"]),
        run_started_at=str(selected["run_started_at_utc"]),
        con_id=int(selected["contracts"]["MCL"]["con_id"]),
        spec=_spec(),
        as_of=as_of,
    )


def mcl_runtime_parity_graduation_gate(
    *, selection: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    """Rehash every selected V18 proof and the sole signal/lifecycle owner."""

    try:
        selected = load_mcl_live_selection_from_mapping(selection)
        root = repo_root.resolve()
        evidence = selected["evidence"]
        artifacts = {}
        for name, binding in evidence.items():
            if not isinstance(binding, Mapping):
                raise ValueError("MCL evidence binding is invalid")
            relative = Path(str(binding.get("path") or ""))
            path = (root / relative).resolve()
            if relative.is_absolute() or root not in path.parents:
                raise ValueError("MCL evidence path escaped repository")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != binding.get("sha256"):
                raise ValueError(f"MCL evidence drifted: {name}")
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError(f"MCL evidence is invalid: {name}")
            artifacts[name] = value
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _gate(
            "INVALID",
            ["runtime_parity_proof_invalid"],
            {"error": str(exc)},
        )
    reasons = []
    lifecycle = artifacts.get("lifecycle_parity", {})
    signal = artifacts.get("signal_parity", {})
    signal_events = signal.get("signal_events") if isinstance(signal, Mapping) else None
    source = artifacts.get("source_shadow", {})
    stage91 = artifacts.get("stage91_result", {})
    owner_path = root / "tradebot/research/mcl_two_speed_auction.py"
    if (
        lifecycle.get("exact_trade_parity") is not True
        or lifecycle.get("expected_trades") != lifecycle.get("actual_trades")
        or lifecycle.get("actual_trades") != 338
        or lifecycle.get("expected_sha256") != lifecycle.get("actual_sha256")
        or lifecycle.get("owner_sha256") != _sha256(owner_path)
    ):
        reasons.append("runtime_lifecycle_parity_invalid")
    if (
        not isinstance(signal_events, Mapping)
        or signal_events.get("exact_event_parity") is not True
        or signal_events.get("expected") != signal_events.get("actual")
        or signal_events.get("actual") != 2247
        or signal_events.get("expected_sha256")
        != signal_events.get("actual_sha256")
    ):
        reasons.append("runtime_signal_parity_invalid")
    shadow = source.get("current_shadow") if isinstance(source, Mapping) else None
    boundary = source.get("selection_boundary") if isinstance(source, Mapping) else None
    if (
        source.get("verdict") != "LIVE_SOURCE_SHADOW_PASS"
        or not isinstance(shadow, Mapping)
        or not isinstance(boundary, Mapping)
        or boundary.get("fresh_post_selection_admission_required") is not True
        or boundary.get("counterfactual_position_is_evidence_only") is not True
        or source.get("submitted_orders") != 0
    ):
        reasons.append("runtime_source_shadow_invalid")
    if (
        stage91.get("verdict") != "REJECT_NO_POST_OUTCOME_GATE_CHANGE"
        or stage91.get("submitted_orders") != 0
    ):
        reasons.append("runtime_stage91_boundary_invalid")
    return _gate(
        "INVALID" if reasons else "PASS",
        reasons,
        {
            "lifecycle_sha256": evidence["lifecycle_parity"]["sha256"],
            "signal_sha256": evidence["signal_parity"]["sha256"],
            "source_shadow_sha256": evidence["source_shadow"]["sha256"],
            "owner_sha256": _sha256(owner_path),
            "trades": lifecycle.get("actual_trades"),
            "signal_events": (
                signal_events.get("actual")
                if isinstance(signal_events, Mapping)
                else None
            ),
        },
    )


def mcl_live_graduation_inputs(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    cutoff_utc: datetime | str,
    profitability_receipt: Mapping[str, object],
    capital_owner_stability_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    cutoff = _utc(cutoff_utc)
    selection_sha = _sha256(selection_path)
    prefix, projected = live_calibration_logical_prefix(records, cutoff_utc=cutoff)
    rows = selected_futures_rows(
        projected, selection_id=str(selected["selection_id"]), spec=_spec()
    )
    con_id = int(selected["contracts"]["MCL"]["con_id"])
    restart = single_contract_restart_gate(
        selected_at_utc=str(selected["selected_at_utc"]),
        rows=rows,
        con_id=con_id,
        spec=_spec(),
    )
    risk, attribution = single_contract_risk_gates(
        rows=rows, con_id=con_id, spec=_spec()
    )
    broker = selected.get("broker_at_selection")
    account_id = str(broker.get("account_id") or "") if isinstance(broker, Mapping) else ""
    return {
        "subject": {
            "strategy_id": MCL_TWO_SPEED_AUCTION_VERSION,
            "strategy_version": MCL_LIVE_EXECUTION_VERSION,
            "signal_instrument": "CL",
            "execution_sleeve": "MCL",
            "capital_sleeve": MCL_LIVE_CAPITAL_SLEEVE,
            "selection_id": selected["selection_id"],
            "run_id": selected["selection_id"],
            "account_fingerprint": hashlib.sha256(account_id.encode()).hexdigest(),
        },
        "selection": {
            "path": str(selection_path),
            "selection_id": selected["selection_id"],
            "run_id": selected["selection_id"],
            "selected_at_utc": selected["selected_at_utc"],
            "run_started_at_utc": selected["run_started_at_utc"],
            "signal_strategy_version": selected["strategy_version"],
            "execution_strategy_version": MCL_LIVE_EXECUTION_VERSION,
            "capital_sleeve": MCL_LIVE_CAPITAL_SLEEVE,
            "selection_file_sha256": selection_sha,
        },
        "selection_file_sha256": selection_sha,
        "ledger_prefix": {
            **prefix,
            "gates": {
                "restart": restart,
                "cash_risk_safety": risk,
                "attribution": attribution,
                "execution": single_contract_execution_graduation_gate(
                    rows,
                    selection_id=str(selected["selection_id"]),
                    sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
                    symbol="MCL",
                    con_id=con_id,
                    order_ref_prefix=MCL_LIVE_ORDER_REF_PREFIX,
                    ladder_schema="mcl.execution-ladder-transition.v1",
                    max_commission_usd=MCL_LIVE_MAX_COMMISSION_USD,
                ),
            },
        },
        "profitability_receipt": dict(profitability_receipt),
        "runtime_parity_proof": mcl_runtime_parity_graduation_gate(
            selection=selected, repo_root=repo_root
        ),
        "capital_owner_stability_proof": portfolio_capital_owner_stability_gate(
            capital_owner_stability_path,
            repo_root=repo_root,
            sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
            selection_id=str(selected["selection_id"]),
            selection_file_sha256=selection_sha,
        ),
    }
