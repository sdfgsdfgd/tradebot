"""Immutable selection and restart-safe live transport for crowned MCL V18."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import Contract, Future

from ..backtest.quotes import contract_from_ticker
from ..chart_data.series import OhlcvBar
from ..engines.execution import quote_health
from ..live.order_evidence import broker_account_snapshot
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .mcl_two_speed_auction import (
    MCL_TWO_SPEED_AUCTION_VERSION,
    MclAuctionMinute,
    MclTwoSpeedAuctionLifecycle,
)


MCL_LIVE_SELECTION_SCHEMA = "mcl.two-speed-auction-selected-run.v1"
MCL_LIVE_SOURCE_VERSION = "mcl.two-speed-auction-live-source.v1"
MCL_LIVE_SOURCE_SCHEMA = "mcl.two-speed-auction-source-checkpoint.v1"
MCL_LIVE_EXECUTION_VERSION = "mcl.two-speed-auction-live-execution.v1"
MCL_LIVE_EXECUTION_SCHEMA = "mcl.two-speed-auction-execution-checkpoint.v1"
MCL_LIVE_PLAN_SCHEMA = "mcl.two-speed-auction-live-plan.v1"
MCL_LIVE_CAPITAL_SLEEVE = "mcl-two-speed-auction-margin"
MCL_LIVE_ORDER_AUTHORITY = "one_mcl_limit_canary"
MCL_LIVE_ORDER_REF_PREFIX = "MCLV18"
MCL_LIVE_SELECTION_PATH = Path("db/calibration/mcl_selected_live_transport.json")
MCL_LIVE_LEDGER_PATH = Path("db/calibration/mcl_live_calibration.jsonl")
MCL_LIVE_MAX_COMMISSION_USD = 0.76
MCL_LIVE_RAW_LOSS_CAP_USD = 300.0
MCL_LIVE_STRESSED_LOSS_USD = 305.52
MCL_LIVE_MAX_INITIAL_MARGIN_AUD = 2_770.0
MCL_LIVE_MAX_MAINTENANCE_MARGIN_AUD = 2_200.0
MCL_LIVE_MIN_STRESS_BUFFER_AUD = 300.0
MCL_LIVE_FX_STRESS_BPS = 11_000
MCL_LIVE_ADMISSION_MAX_AGE_SECONDS = 120.0
MCL_LIVE_SOURCE_MAX_AGE_SECONDS = 8 * 60.0
MCL_LIVE_QUOTE_MAX_AGE_SECONDS = 10.0
MCL_LIVE_WEEKLY_FLAT_ET = (4, 16, 53)
MCL_LIVE_PACKAGE_ID = "mcl-one-contract-stage91"

_ET = ZoneInfo("America/New_York")
_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = {
    "declaration": Path("backtests/mcl/current-hf.json"),
    "crown": Path("backtests/mcl/mcl_two_speed_auction_v18_crown.json"),
    "generation": Path(
        "backtests/mcl/mcl_turn_authenticity_microstructure_generation.json"
    ),
    "signal_parity": Path(
        "backtests/mcl/mcl_two_speed_auction_v18_signal_owner_parity.json"
    ),
    "lifecycle_parity": Path(
        "backtests/mcl/mcl_two_speed_auction_v18_lifecycle_owner_parity.json"
    ),
    "source_shadow": Path("backtests/mcl/mcl_v18_live_source_shadow_receipt.json"),
    "stage91_preregistration": Path(
        "backtests/mcl/mcl_v18_weekly_closure_canary_stage91_preregistration.json"
    ),
    "stage91_result": Path(
        "backtests/mcl/mcl_v18_weekly_closure_canary_stage91_rejection.json"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("MCL live timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _identity(value: object) -> str:
    return calibration_fingerprint(value)


def _is_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _artifact(root: Path, key: str) -> tuple[dict[str, object], str]:
    path = root / _ARTIFACTS[key]
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"MCL {key} artifact is invalid")
    return dict(value), _sha256(path)


def _contract(identity: Mapping[str, object]) -> Future:
    contract = Future(
        conId=int(identity["con_id"]),
        symbol=str(identity["symbol"]),
        lastTradeDateOrContractMonth=str(identity["expiry"]),
        exchange="NYMEX",
        currency="USD",
        localSymbol=str(identity["local_symbol"]),
        multiplier=str(identity["multiplier"]),
    )
    setattr(contract, "minTick", 0.01)
    return contract


def mcl_live_contracts(selection: Mapping[str, object]) -> tuple[Future, Future]:
    selected = load_mcl_live_selection_from_mapping(selection)
    return _contract(selected["contracts"]["CL"]), _contract(
        selected["contracts"]["MCL"]
    )


def load_mcl_live_selection_from_mapping(
    value: Mapping[str, object],
) -> dict[str, object]:
    selection = dict(value)
    selection_id = str(selection.pop("selection_id", ""))
    contracts = selection.get("contracts")
    risk = selection.get("risk")
    execution = selection.get("execution")
    broker = selection.get("broker_at_selection")
    evidence = selection.get("evidence")
    successor = selection.get("allocation_successor")
    baseline = selection.get("baseline")
    if (
        selection.get("schema") != MCL_LIVE_SELECTION_SCHEMA
        or selection_id != _identity(selection)
        or selection.get("strategy_version") != MCL_TWO_SPEED_AUCTION_VERSION
        or selection.get("authority") != "selected_live_bounded_canary"
        or not all(
            isinstance(item, Mapping)
            for item in (contracts, risk, execution, broker, evidence, successor, baseline)
        )
        or set(contracts) != {"CL", "MCL"}
        or {str(contracts[symbol].get("symbol") or "") for symbol in contracts}
        != {"CL", "MCL"}
        or len(
            {
                str(contracts[symbol].get("expiry") or "")[:6]
                for symbol in contracts
            }
        )
        != 1
        or any(int(contracts[symbol].get("con_id") or 0) <= 0 for symbol in contracts)
        or risk.get("max_contracts") != 1
        or risk.get("raw_loss_cap_usd") != MCL_LIVE_RAW_LOSS_CAP_USD
        or risk.get("package_stressed_loss_usd") != MCL_LIVE_STRESSED_LOSS_USD
        or risk.get("max_commission_usd_per_order") != MCL_LIVE_MAX_COMMISSION_USD
        or risk.get("fx_stress_bps") != MCL_LIVE_FX_STRESS_BPS
        or risk.get("minimum_post_stress_excess_liquidity_aud")
        != MCL_LIVE_MIN_STRESS_BUFFER_AUD
        or execution.get("order_type") != "LMT"
        or execution.get("entry_chase_mode") != "AUTO"
        or execution.get("risk_reduction_chase_mode") != "RELENTLESS"
        or execution.get("market_orders_allowed") is not False
        or broker.get("account_type") != "CASH"
        or broker.get("base_currency") != "AUD"
        or broker.get("open_orders") != []
        or baseline.get("position") != 0
        or baseline.get("inherited_target_authority") != "none"
        or successor.get("package_id") != MCL_LIVE_PACKAGE_ID
        or successor.get("package_cash_debit_usd_cents") != 76
        or not 0 < int(successor.get("initial_margin_base_cents") or 0)
        <= math.ceil(MCL_LIVE_MAX_INITIAL_MARGIN_AUD * 100)
        or not 0 < int(successor.get("maintenance_margin_base_cents") or 0)
        <= math.ceil(MCL_LIVE_MAX_MAINTENANCE_MARGIN_AUD * 100)
        or any(not _is_sha(item.get("sha256")) for item in evidence.values())
    ):
        raise ValueError("MCL selected-run contract is invalid")
    return {**selection, "selection_id": selection_id}


def load_mcl_live_selection(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL selected run must be one JSON object")
    return load_mcl_live_selection_from_mapping(value)


def _completed_bar(raw: OhlcvBar) -> OhlcvBar:
    timestamp = raw.ts
    start = (
        timestamp.replace(tzinfo=_ET)
        if timestamp.tzinfo is None
        else timestamp.astimezone(timezone.utc)
    )
    close = start.astimezone(timezone.utc) + timedelta(minutes=1)
    return OhlcvBar(
        close,
        float(raw.open),
        float(raw.high),
        float(raw.low),
        float(raw.close),
        float(raw.volume),
    )


def _bar_map(
    rows: Sequence[OhlcvBar], *, cutoff: datetime, name: str
) -> dict[datetime, OhlcvBar]:
    output: dict[datetime, OhlcvBar] = {}
    for raw in rows:
        bar = _completed_bar(raw)
        if bar.ts > cutoff:
            continue
        if bar.ts in output and output[bar.ts] != bar:
            raise ValueError(f"MCL {name} history changed at one timestamp")
        output[bar.ts] = bar
    if not output:
        raise ValueError(f"MCL {name} history is empty")
    return output


async def mcl_source_snapshot(
    client,
    *,
    cl_contract: Contract,
    mcl_contract: Contract,
    observed_at: datetime,
    selected_at: datetime | None = None,
) -> dict[str, object]:
    """Replay the exact finalized CL/MCL minute source without adopting history."""

    now = _utc(observed_at)
    raw_cl, raw_mcl = await asyncio.gather(
        client.historical_bars_ohlcv(
            cl_contract,
            duration_str="3 D",
            bar_size="1 min",
            use_rth=False,
            what_to_show="TRADES",
            cache_ttl_sec=0,
        ),
        client.historical_bars_ohlcv(
            mcl_contract,
            duration_str="3 D",
            bar_size="1 min",
            use_rth=False,
            what_to_show="TRADES",
            cache_ttl_sec=0,
        ),
    )
    cutoff = now.replace(second=0, microsecond=0)
    cl = _bar_map(raw_cl, cutoff=cutoff, name="CL")
    mcl = _bar_map(raw_mcl, cutoff=cutoff, name="MCL")
    common = sorted(set(cl).intersection(mcl))
    if len(common) < 500:
        raise ValueError("MCL source warmup is incomplete")
    latest = common[-1]
    if not 0 <= (now - latest).total_seconds() <= MCL_LIVE_SOURCE_MAX_AGE_SECONDS:
        raise ValueError("MCL finalized source is stale")
    contract_key = str(getattr(mcl_contract, "lastTradeDateOrContractMonth", ""))[:6]
    lifecycle = MclTwoSpeedAuctionLifecycle()
    decisions = []
    for stamp in common:
        step = lifecycle.update(
            MclAuctionMinute(contract_key, cl[stamp], mcl[stamp])
        )
        if step.decision is not None:
            decisions.append(step.decision)
    if not decisions:
        raise ValueError("MCL source produced no V18 state")
    start = _utc(selected_at) if selected_at is not None else now
    target = None
    last_raw_turn = None
    for decision in decisions:
        if decision.observed_at_utc <= start:
            continue
        payload = decision.as_payload()
        event = {
            "event_id": _identity(payload),
            "observed_at_utc": decision.observed_at_utc.isoformat(),
            "signal_at_utc": (
                decision.signal_at_utc.isoformat()
                if decision.signal_at_utc is not None
                else None
            ),
            "direction": decision.admitted_direction,
            "route": decision.route,
            "decision": payload,
        }
        if decision.phase == "RAW_TURN":
            target = None
            last_raw_turn = event
        elif decision.phase == "MATURATION" and decision.admitted_direction in {
            -1,
            1,
        }:
            target = event
    latest_decision = decisions[-1]
    return {
        "schema": "mcl.two-speed-auction-finalized-source-snapshot.v1",
        "observed_at_utc": now.isoformat(),
        "authority": "finalized_source_only_no_orders_no_capital",
        "contract_month": contract_key,
        "contracts": {
            "CL": {
                "con_id": int(getattr(cl_contract, "conId", 0) or 0),
                "local_symbol": str(getattr(cl_contract, "localSymbol", "") or ""),
            },
            "MCL": {
                "con_id": int(getattr(mcl_contract, "conId", 0) or 0),
                "local_symbol": str(
                    getattr(mcl_contract, "localSymbol", "") or ""
                ),
            },
        },
        "rows": {"CL": len(cl), "MCL": len(mcl), "common": len(common)},
        "first_common_close_utc": common[0].isoformat(),
        "latest_common_close_utc": latest.isoformat(),
        "latest_decision": latest_decision.as_payload(),
        "target": target,
        "last_raw_turn": last_raw_turn,
        "counterfactual_position": lifecycle.position,
        "synthetic_midcycle_entry_authority": "none",
        "submitted_orders": 0,
    }


def _preview_rows(preview: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = preview.get("what_if")
    return (
        {
            str(row.get("action") or "").upper(): row
            for row in rows
            if isinstance(row, Mapping)
        }
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
        else {}
    )


def build_mcl_live_selection(
    *,
    repository_root: Path,
    preview: Mapping[str, object],
    selected_at: datetime,
) -> dict[str, object]:
    """Freeze one flat, funded V18 canary from exact repository and broker truth."""

    root = repository_root.resolve()
    at = _utc(selected_at)
    observed = _utc(preview.get("observed_at_utc"))
    artifacts = {key: _artifact(root, key) for key in _ARTIFACTS}
    declaration = artifacts["declaration"][0]
    crown = artifacts["crown"][0]
    generation = artifacts["generation"][0]
    signal = artifacts["signal_parity"][0]
    lifecycle = artifacts["lifecycle_parity"][0]
    source = artifacts["source_shadow"][0]
    stage91 = artifacts["stage91_result"][0]
    broker = preview.get("broker")
    contracts = preview.get("contracts")
    source_preview = preview.get("source")
    what_if = _preview_rows(preview)
    positions = broker.get("positions") if isinstance(broker, Mapping) else None
    if (
        preview.get("schema") != "mcl.v18-live-commissioning-preview.v2"
        or preview.get("authority") != "fresh_nontransmitting_what_if_only"
        or preview.get("submitted_orders") != 0
        or not 0 <= (at - observed).total_seconds() <= 90
        or not isinstance(broker, Mapping)
        or broker.get("account_type") != "CASH"
        or broker.get("base_currency") != "AUD"
        or broker.get("open_orders") != []
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or any(
            isinstance(row, Mapping)
            and str(row.get("symbol") or "").upper() == "MCL"
            and abs(_number(row.get("quantity"), name="MCL position")) > 1e-9
            for row in positions
        )
        or not isinstance(contracts, Mapping)
        or set(contracts) != {"CL", "MCL"}
        or set(what_if) != {"BUY", "SELL"}
        or any(
            row.get("status") != "PreSubmitted"
            or str(row.get("commission_currency") or "").upper() != "USD"
            or _number(row.get("commission"), name="MCL commission")
            > MCL_LIVE_MAX_COMMISSION_USD
            or _number(row.get("init_margin_change"), name="MCL initial margin")
            > MCL_LIVE_MAX_INITIAL_MARGIN_AUD
            or _number(
                row.get("maintenance_margin_change"),
                name="MCL maintenance margin",
            )
            > MCL_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            or str(row.get("warning_text") or "")
            for row in what_if.values()
        )
        or not isinstance(source_preview, Mapping)
        or source_preview.get("submitted_orders") != 0
        or declaration.get("version") != "18"
        or declaration.get("artifact_sha256") != artifacts["crown"][1]
        or not isinstance(crown.get("signal"), Mapping)
        or crown["signal"].get("strategy_version")
        != MCL_TWO_SPEED_AUCTION_VERSION
        or not isinstance(signal.get("signal_events"), Mapping)
        or signal["signal_events"].get("exact_event_parity") is not True
        or lifecycle.get("exact_trade_parity") is not True
        or lifecycle.get("actual_trades") != 338
        or source.get("verdict") != "LIVE_SOURCE_SHADOW_PASS"
        or stage91.get("primary_300", {}).get("maximum_observed_loss_usd")
        != MCL_LIVE_RAW_LOSS_CAP_USD
    ):
        raise ValueError("MCL canary selection evidence is incomplete")
    frozen_contracts = generation.get("contracts")
    if not isinstance(frozen_contracts, Mapping) or any(
        dict(contracts[symbol]) != dict(frozen_contracts[symbol])
        for symbol in ("CL", "MCL")
    ):
        raise ValueError("MCL selected contract generation changed")
    selected_initial_margin_cents = math.ceil(
        max(
            _number(row["init_margin_change"], name="MCL selected initial margin")
            for row in what_if.values()
        )
        * 100
    )
    selected_maintenance_margin_cents = math.ceil(
        max(
            _number(
                row["maintenance_margin_change"],
                name="MCL selected maintenance margin",
            )
            for row in what_if.values()
        )
        * 100
    )
    body = {
        "schema": MCL_LIVE_SELECTION_SCHEMA,
        "strategy_version": MCL_TWO_SPEED_AUCTION_VERSION,
        "authority": "selected_live_bounded_canary",
        "selected_at_utc": at.isoformat(),
        "run_started_at_utc": at.isoformat(),
        "contracts": {symbol: dict(contracts[symbol]) for symbol in ("CL", "MCL")},
        "baseline": {
            "position": 0,
            "inherited_target_authority": "none",
            "first_eligible_event": "fresh_post_selection_v18_admission",
            "source_snapshot_sha256": _identity(source_preview),
        },
        "broker_at_selection": dict(broker),
        "risk": {
            "max_contracts": 1,
            "max_commission_usd_per_order": MCL_LIVE_MAX_COMMISSION_USD,
            "raw_loss_cap_usd": MCL_LIVE_RAW_LOSS_CAP_USD,
            "package_stressed_loss_usd": MCL_LIVE_STRESSED_LOSS_USD,
            "max_initial_margin_change_aud": MCL_LIVE_MAX_INITIAL_MARGIN_AUD,
            "max_maintenance_margin_change_aud": (
                MCL_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            ),
            "minimum_post_stress_excess_liquidity_aud": (
                MCL_LIVE_MIN_STRESS_BUFFER_AUD
            ),
            "fx_stress_bps": MCL_LIVE_FX_STRESS_BPS,
            "weekly_flat_et": "Friday 16:53",
            "post_safety_exit_reentry": "next_original_v18_admission_only",
        },
        "execution": {
            "order_type": "LMT",
            "time_in_force": "GTC",
            "outside_rth": True,
            "entry_initial_mode": "OPTIMISTIC",
            "entry_chase_mode": "AUTO",
            "risk_reduction_initial_mode": "CROSS",
            "risk_reduction_chase_mode": "RELENTLESS",
            "market_orders_allowed": False,
        },
        "evidence": {
            key: {"path": _ARTIFACTS[key].as_posix(), "sha256": digest}
            for key, (_value, digest) in artifacts.items()
        },
        "exception": {
            "authority": "user_explicit_gold_style_bounded_live_canary",
            "waived": "prospective_stage87_89_cohort_before_unchanged_v18_canary",
            "not_waived": [
                "finalized_source_identity",
                "fresh_two_sided_limit_preview",
                "one_contract_margin_and_first_admitter_capital",
                "flat_immutable_selection",
                "restart_reconciliation_and_risk_reduction",
            ],
        },
        "allocation_successor": {
            "schema": "mcl.two-speed-auction-portfolio-package.v1",
            "package_id": MCL_LIVE_PACKAGE_ID,
            "package_cash_debit_usd_cents": 76,
            "initial_margin_base_cents": selected_initial_margin_cents,
            "maintenance_margin_base_cents": selected_maintenance_margin_cents,
            "broker_preview_fingerprint": _identity(preview),
        },
    }
    selected = {**body, "selection_id": _identity(body)}
    return load_mcl_live_selection_from_mapping(selected)


async def _live_quote(
    client,
    contract: Contract,
    *,
    owner: str,
    wait_seconds: float = 3.0,
) -> tuple[object, dict[str, object]]:
    ticker = await client.ensure_ticker(contract, owner=owner)
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        captured = contract_from_ticker(
            getattr(ticker, "contract", None) or contract,
            ticker,
        )
        updated = getattr(ticker, "tbTopQuoteUpdatedMono", None)
        age = (
            max(0.0, time.monotonic() - float(updated))
            if updated is not None
            else None
        )
        health = quote_health(
            bid=captured.bid,
            ask=captured.ask,
            last=captured.last,
            close=captured.close,
            market_data_type=captured.market_data_type,
            age_sec=age,
            max_age_sec=MCL_LIVE_QUOTE_MAX_AGE_SECONDS,
            require_live=True,
            require_nbbo=True,
            require_age=True,
        )
        quote = {
            "bid": captured.bid,
            "ask": captured.ask,
            "last": captured.last,
            "close": captured.close,
            "age_seconds": age,
            "market_data_type": captured.market_data_type,
            "health": health,
        }
        if health.get("eligible") is True:
            return ticker, quote
        if time.monotonic() >= deadline:
            raise ValueError("selected MCL contract lacks fresh streaming L1 NBBO")
        await asyncio.sleep(0.1)


async def capture_mcl_commissioning_preview(
    client,
    *,
    repository_root: Path = _ROOT,
    observed_at: datetime,
) -> dict[str, object]:
    """Capture the exact funded, nontransmitting MCL commissioning boundary."""

    generation, _digest = _artifact(repository_root.resolve(), "generation")
    identities = generation.get("contracts")
    if not isinstance(identities, Mapping) or set(identities) != {"CL", "MCL"}:
        raise ValueError("MCL commissioning generation has no exact contract pair")
    contracts = {symbol: _contract(identities[symbol]) for symbol in ("CL", "MCL")}
    broker = await broker_account_snapshot(client, base_currency="AUD")
    if any(
        row.get("symbol") == "MCL" and abs(float(row.get("quantity") or 0.0)) > 1e-9
        for row in broker["positions"]
    ) or any(row.get("symbol") == "MCL" for row in broker["open_orders"]):
        raise ValueError("MCL commissioning requires a flat, order-free MCL baseline")
    ticker, quote = await _live_quote(
        client, contracts["MCL"], owner="mcl-v18-commissioning"
    )
    source = await mcl_source_snapshot(
        client,
        cl_contract=contracts["CL"],
        mcl_contract=contracts["MCL"],
        observed_at=observed_at,
    )
    rows = []
    for action, price in (("BUY", quote["ask"]), ("SELL", quote["bid"])):
        preview = await client.preview_limit_order(
            contracts["MCL"],
            action,
            1,
            float(price),
            True,
            f"{MCL_LIVE_ORDER_REF_PREFIX}-PREVIEW-{action}",
        )
        row = {"action": action, "limit_price": float(price), **asdict(preview)}
        commission = _number(row.get("commission"), name="MCL preview commission")
        if (
            row.get("status") != "PreSubmitted"
            or str(row.get("commission_currency") or "").upper() != "USD"
            or commission > MCL_LIVE_MAX_COMMISSION_USD
            or str(row.get("warning_text") or "")
        ):
            raise ValueError("MCL nontransmitting LIMIT preview failed")
        rows.append(row)
    client.release_ticker(
        int(getattr(contracts["MCL"], "conId", 0) or 0),
        owner="mcl-v18-commissioning",
    )
    now = _utc(observed_at)
    return {
        "schema": "mcl.v18-live-commissioning-preview.v2",
        "observed_at_utc": now.isoformat(),
        "authority": "fresh_nontransmitting_what_if_only",
        "broker": broker,
        "contracts": {
            symbol: dict(identities[symbol]) for symbol in ("CL", "MCL")
        },
        "quote": quote,
        "source": source,
        "what_if": rows,
        "submitted_orders": 0,
    }


def persist_mcl_source_checkpoint(
    ledger: LiveCalibrationLedger,
    *,
    selection: Mapping[str, object],
    source: Mapping[str, object],
    observed_at: datetime,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    if (
        source.get("schema")
        != "mcl.two-speed-auction-finalized-source-snapshot.v1"
        or source.get("submitted_orders") != 0
        or source.get("contract_month")
        != str(selected["contracts"]["MCL"]["expiry"])[:6]
    ):
        raise ValueError("MCL finalized source snapshot changed identity")
    evidence = {
        "schema": MCL_LIVE_SOURCE_SCHEMA,
        "selection_id": selected["selection_id"],
        "source": dict(source),
        "target": source.get("target"),
        "last_raw_turn": source.get("last_raw_turn"),
        "synthetic_midcycle_entry_authority": "none",
        "order_authority": "none",
        "submitted_orders": 0,
    }
    return ledger.checkpoint(
        evaluation_as_of=now,
        strategy_id=MCL_TWO_SPEED_AUCTION_VERSION,
        strategy_version=MCL_LIVE_SOURCE_VERSION,
        trading_date=now.date().isoformat(),
        session="MCL_GTH_SOURCE",
        status="EVALUATED",
        evidence=evidence,
        recorded_at=now,
    )


def latest_mcl_source_checkpoint(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> dict[str, object] | None:
    for record in reversed(records):
        evidence = record.get("evidence")
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == MCL_LIVE_SOURCE_VERSION
            and record.get("status") == "EVALUATED"
            and isinstance(evidence, Mapping)
            and evidence.get("schema") == MCL_LIVE_SOURCE_SCHEMA
            and evidence.get("selection_id") == selection_id
        ):
            return dict(record)
    return None


async def refresh_mcl_source_if_due(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    observed_at: datetime,
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    records = tuple(ledger.records())
    latest = latest_mcl_source_checkpoint(
        records, selection_id=str(selected["selection_id"])
    )
    due = now.replace(second=0, microsecond=0) - timedelta(
        minutes=now.minute % 5
    )
    latest_decision = None
    if latest is not None:
        evidence = latest["evidence"]
        assert isinstance(evidence, Mapping)
        source = evidence.get("source")
        if isinstance(source, Mapping):
            decision = source.get("latest_decision")
            if isinstance(decision, Mapping):
                latest_decision = _utc(decision.get("observed_at_utc"))
    if latest is not None and latest_decision is not None and latest_decision >= due:
        return latest
    cl_contract, mcl_contract = mcl_live_contracts(selected)
    source = await mcl_source_snapshot(
        client,
        cl_contract=cl_contract,
        mcl_contract=mcl_contract,
        observed_at=now,
        selected_at=_utc(selected["selected_at_utc"]),
    )
    decision_at = _utc(source["latest_decision"]["observed_at_utc"])
    if decision_at < due:
        raise ValueError("MCL broker history has not finalized the due V18 boundary")
    return persist_mcl_source_checkpoint(
        ledger, selection=selected, source=source, observed_at=now
    )
