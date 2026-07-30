"""State-change-only execution evidence for the XSP v2 ETF transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import Stock

from ..backtest.quotes import contract_from_ticker
from ..engines.execution import execution_price, quote_health
from ..engines.market import xsp_session_label_et, xsp_trading_date
from ..spot.champions import repo_root
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .xsp_opening_edge_v2 import XSP_OPENING_EDGE_V2_VERSION


XSP_V2_ETF_EXECUTION_OBSERVER_VERSION = (
    "xsp.opening-edge-v2-upro-spxu-execution-observer.v1"
)
XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA = (
    "xsp.opening-edge-v2-upro-spxu-execution-observation.v1"
)
XSP_V2_ETF_QUOTE_MAX_AGE_SECONDS = 10.0
XSP_V2_ETF_QUOTE_WAIT_SECONDS = 3.0
_LADDER_MODES = ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS")
_GATE_RELATIVE_PATH = Path(
    "backtests/xsp/opening_edge_v2_upro_spxu_preregistered_gate.json"
)
_RECEIPT_RELATIVE_PATH = Path(
    "backtests/xsp/opening_edge_v2_upro_spxu_acceptance_receipt.json"
)
_STATE_IDENTITY_FIELDS = (
    "lane",
    "direction",
    "entry_time",
    "trading_date",
)
_STATE_FIELDS = (
    *_STATE_IDENTITY_FIELDS,
    "entry_price",
)


@dataclass(frozen=True)
class XspV2EtfTransport:
    candidate: Mapping[str, object]
    gate_sha256: str
    receipt_sha256: str
    verdict: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_xsp_v2_etf_transport(
    *,
    root: Path | None = None,
) -> XspV2EtfTransport:
    """Load the rejected transport without weakening its frozen HOLD boundary."""

    resolved_root = (root or repo_root()).resolve()
    gate_path = resolved_root / _GATE_RELATIVE_PATH
    receipt_path = resolved_root / _RECEIPT_RELATIVE_PATH
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    gate_sha256 = _sha256(gate_path)
    receipt_candidate = receipt.get("candidate")
    if (
        gate.get("schema")
        != "xsp.opening-edge-v2-upro-spxu-preregistered-gate.v1"
        or gate.get("authority") != "research_and_broker_preview_only"
        or gate.get("order_authority") != "none"
        or gate.get("profitability_clock_started") is not False
        or receipt.get("schema")
        != "xsp.opening-edge-v2-upro-spxu-acceptance-receipt.v1"
        or receipt.get("authority") != "research_and_broker_preview_only"
        or receipt.get("order_authority") != "none"
        or receipt.get("profitability_clock_started") is not False
        or receipt.get("verdict") != "HOLD"
        or receipt.get("selected_shadow_created") is not False
        or receipt_candidate != gate.get("candidate")
        or (receipt.get("evidence") or {}).get("preregistered_gate_sha256")
        != gate_sha256
    ):
        raise ValueError("invalid frozen UPRO/SPXU HOLD transport")
    candidate = receipt_candidate
    if not isinstance(candidate, Mapping):
        raise ValueError("UPRO/SPXU transport candidate is missing")
    for direction, symbol, quantity in (("up", "UPRO", 9), ("down", "SPXU", 31)):
        leg = candidate.get(direction)
        if (
            not isinstance(leg, Mapping)
            or leg.get("symbol") != symbol
            or leg.get("quantity") != quantity
        ):
            raise ValueError("UPRO/SPXU transport identity drift")
    return XspV2EtfTransport(
        candidate=dict(candidate),
        gate_sha256=gate_sha256,
        receipt_sha256=_sha256(receipt_path),
        verdict="HOLD",
    )


def xsp_v2_position_state(
    paired_equity: Mapping[str, object],
) -> tuple[str, dict[str, object] | None]:
    profiles = paired_equity.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("v2 execution observation requires paired profiles")
    states = []
    identities = []
    starts = []
    for name in ("research", "broker"):
        profile = profiles.get(name)
        if not isinstance(profile, Mapping):
            raise ValueError("v2 execution observation requires both profiles")
        position = profile.get("latest_position")
        starts.append(str(profile.get("run_started_at_utc") or ""))
        if position is None:
            states.append(None)
            identities.append(None)
            continue
        if (
            not isinstance(position, Mapping)
            or position.get("exit_reason") != "end"
            or position.get("direction") not in {"up", "down"}
            or not position.get("entry_time")
        ):
            raise ValueError("invalid v2 open-position state")
        state = {field: position.get(field) for field in _STATE_FIELDS}
        states.append(state)
        identities.append(
            {field: state.get(field) for field in _STATE_IDENTITY_FIELDS}
        )
    if identities[0] != identities[1] or len(set(starts)) != 1 or not starts[0]:
        raise ValueError("v2 research/broker position state drift")
    run_key = calibration_fingerprint(
        {
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_VERSION,
            "crown_config_fingerprint": paired_equity.get(
                "crown_config_fingerprint"
            ),
            "run_started_at_utc": starts[0],
        }
    )
    return run_key, states[0]


def _prior_state(
    records: Sequence[Mapping[str, object]],
    *,
    run_key: str,
) -> tuple[bool, dict[str, object] | None]:
    for row in reversed(records):
        evidence = row.get("evidence")
        if (
            row.get("kind") == "checkpoint"
            and row.get("strategy_version")
            == XSP_V2_ETF_EXECUTION_OBSERVER_VERSION
            and isinstance(evidence, Mapping)
            and evidence.get("run_key") == run_key
        ):
            state = evidence.get("current_state")
            return (
                True,
                (
                    {field: state.get(field) for field in _STATE_FIELDS}
                    if isinstance(state, Mapping)
                    else None
                ),
            )
    return False, None


def _transport_state(
    state: Mapping[str, object] | None,
    transport: XspV2EtfTransport,
) -> dict[str, object] | None:
    if state is None:
        return None
    direction = str(state["direction"])
    leg = transport.candidate[direction]
    if not isinstance(leg, Mapping):
        raise ValueError("invalid frozen transport leg")
    return {
        **dict(state),
        "symbol": str(leg["symbol"]),
        "quantity": int(leg["quantity"]),
    }


def _transition_legs(
    prior: Mapping[str, object] | None,
    current: Mapping[str, object] | None,
) -> tuple[dict[str, object], ...]:
    legs = []
    if prior is not None:
        legs.append(
            {
                "action": "SELL",
                "symbol": str(prior["symbol"]),
                "quantity": int(prior["quantity"]),
            }
        )
    if current is not None:
        legs.append(
            {
                "action": "BUY",
                "symbol": str(current["symbol"]),
                "quantity": int(current["quantity"]),
            }
        )
    return tuple(legs)


def _top_quote_age(ticker: object) -> float | None:
    updated = getattr(ticker, "tbTopQuoteUpdatedMono", None)
    try:
        return max(0.0, time.monotonic() - float(updated))
    except (TypeError, ValueError):
        return None


async def _observe_leg(
    client,
    *,
    leg: Mapping[str, object],
    observed_at: datetime,
) -> dict[str, object]:
    symbol = str(leg["symbol"])
    qualified = await client.qualify_proxy_contracts(
        Stock(symbol, "SMART", "USD")
    )
    contract = next(
        (
            row
            for row in qualified
            if int(getattr(row, "conId", 0) or 0) > 0
            and str(getattr(row, "symbol", "") or "").upper() == symbol
            and str(getattr(row, "secType", "") or "").upper() == "STK"
        ),
        None,
    )
    if contract is None:
        return {
            **dict(leg),
            "status": "qualification_unavailable",
            "quote_eligible": False,
            "execution_eligible": False,
            "order_authority": "none",
        }
    ticker = await client.ensure_ticker(
        contract,
        owner="xsp-v2-etf-execution-observer",
    )
    deadline = time.monotonic() + XSP_V2_ETF_QUOTE_WAIT_SECONDS
    captured = None
    health = None
    while True:
        routed_contract = getattr(ticker, "contract", None) or contract
        captured = contract_from_ticker(routed_contract, ticker)
        health = quote_health(
            bid=captured.bid,
            ask=captured.ask,
            last=captured.last,
            close=captured.close,
            market_data_type=captured.market_data_type,
            age_sec=_top_quote_age(ticker),
            max_age_sec=XSP_V2_ETF_QUOTE_MAX_AGE_SECONDS,
            require_live=True,
            require_nbbo=True,
            require_age=True,
        )
        if health["eligible"] or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)
    assert captured is not None and health is not None
    action = str(leg["action"])
    ladder = {
        mode: execution_price(
            getattr(ticker, "contract", None) or contract,
            ticker,
            mode,
            action,
            bid=captured.bid,
            ask=captured.ask,
            last=captured.last,
            fallback_price=captured.close,
            custom_price=None,
        )
        for mode in _LADDER_MODES
    }
    preview = None
    preview_error = None
    if health["eligible"] and ladder["CROSS"] is not None:
        try:
            preview = asdict(
                await client.preview_limit_order(
                    contract,
                    action,
                    float(leg["quantity"]),
                    float(ladder["CROSS"]),
                    xsp_session_label_et(observed_at) != "RTH",
                )
            )
        except Exception as exc:
            preview_error = f"{type(exc).__name__}: {exc}"
    return {
        **dict(leg),
        "status": "evaluated" if health["eligible"] else "quote_ineligible",
        "quote_eligible": bool(health["eligible"]),
        "execution_eligible": False,
        "contract": asdict(captured),
        "quote_source": getattr(ticker, "tbQuoteSource", None),
        "quote_as_of": getattr(ticker, "tbQuoteAsOf", None)
        or captured.quote_time,
        "quote_health": dict(health),
        "ladder": ladder,
        "what_if_preview": preview,
        "preview_error": preview_error,
        "submitted_orders": 0,
        "order_authority": "none",
    }


async def advance_xsp_v2_etf_execution_observer(
    ledger: LiveCalibrationLedger,
    *,
    client,
    source_receipt: Mapping[str, object],
    observed_at: datetime,
    recorded_at: datetime | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Record one broker-preview observation only when v2 position state changes."""

    if (
        source_receipt.get("evaluation_status") != "EVALUATED"
        or not isinstance(source_receipt.get("paired_equity"), Mapping)
    ):
        return {
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
            "status": "source_not_evaluated",
            "checkpoint_id": None,
            "order_authority": "none",
        }
    paired = source_receipt["paired_equity"]
    assert isinstance(paired, Mapping)
    run_key, raw_current = xsp_v2_position_state(paired)
    records = tuple(ledger.records())
    prior_exists, raw_prior = _prior_state(records, run_key=run_key)
    if (not prior_exists and raw_current is None) or (
        prior_exists and raw_prior == raw_current
    ):
        return {
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
            "status": "unchanged",
            "checkpoint_id": None,
            "order_authority": "none",
        }
    transport = load_xsp_v2_etf_transport(root=root)
    prior = _transport_state(raw_prior, transport)
    current = _transport_state(raw_current, transport)
    transition_id = calibration_fingerprint(
        {
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
            "run_key": run_key,
            "prior_state": prior,
            "current_state": current,
        }
    )
    if any(
        row.get("kind") == "checkpoint"
        and isinstance(row.get("evidence"), Mapping)
        and row["evidence"].get("transition_id") == transition_id
        for row in records
    ):
        return {
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
            "status": "unchanged",
            "transition_id": transition_id,
            "checkpoint_id": None,
            "order_authority": "none",
        }
    observations = [
        await _observe_leg(client, leg=leg, observed_at=observed_at)
        for leg in _transition_legs(prior, current)
    ]
    evaluated_at = recorded_at or datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None or observed_at.tzinfo is None:
        raise ValueError("v2 execution observation timestamps must be aware")
    status = (
        "EVALUATED"
        if observations and all(row["quote_eligible"] for row in observations)
        else "NO_DATA"
    )
    trading_day = xsp_trading_date(observed_at)
    checkpoint = ledger.checkpoint(
        evaluation_as_of=evaluated_at,
        strategy_id=XSP_OPENING_EDGE_V2_VERSION,
        strategy_version=XSP_V2_ETF_EXECUTION_OBSERVER_VERSION,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=str(source_receipt.get("session") or "CLOSED"),
        status=status,
        evidence={
            "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
            "transition_id": transition_id,
            "run_key": run_key,
            "source_checkpoint_id": source_receipt.get("checkpoint_id"),
            "source_observed_at_utc": observed_at.astimezone(
                timezone.utc
            ).isoformat(),
            "prior_state": prior,
            "current_state": current,
            "legs": observations,
            "frozen_transport": {
                "gate_sha256": transport.gate_sha256,
                "receipt_sha256": transport.receipt_sha256,
                "verdict": transport.verdict,
                "selected": False,
            },
            "execution_eligibility": "HOLD",
            "profitability_clock_started": False,
            "submitted_orders": 0,
            "order_authority": "none",
        },
        recorded_at=evaluated_at,
    )
    return {
        "schema": XSP_V2_ETF_EXECUTION_OBSERVER_SCHEMA,
        "status": status,
        "transition_id": transition_id,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "legs": observations,
        "order_authority": "none",
    }
