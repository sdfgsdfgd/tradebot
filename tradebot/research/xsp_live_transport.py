"""Selected-run boundary and pure execution plan for XSP Opening Edge."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from ..engines.execution import execution_policy_contract
from ..engines.market import equity_rth_close_time_et, xsp_session_label_et
from ..live.order_evidence import tiered_us_stock_commission_ceiling
from ..time_utils import ET_ZONE
from .live_calibration import calibration_fingerprint
from .xsp_execution_observer import xsp_v2_position_state
from .xsp_context import xsp_execution_signal_context, xsp_execution_state_context
from .xsp_opening_edge_v2 import XSP_OPENING_EDGE_V2_VERSION

XSP_V2_TRANSPORT_SELECTION_SCHEMA = "xsp.opening-edge-v2-spyu-spxu-selected-run.v1"
XSP_V2_TRANSPORT_PLAN_SCHEMA = "xsp.opening-edge-v2-spyu-spxu-transport-plan.v1"
XSP_V3_TRANSPORT_SELECTION_SCHEMA = "xsp.opening-edge-v3-upro-spxu-selected-run.v1"
XSP_V3_ROTATION_SELECTION_SCHEMA = "xsp.opening-edge-v3-upro-spxu-selected-run.v2"
XSP_V3_PACKAGE_SELECTION_SCHEMA = "xsp.opening-edge-v3-upro-spxu-selected-run.v3"
XSP_V3_TRANSPORT_PLAN_SCHEMA = "xsp.opening-edge-v3-upro-spxu-transport-plan.v1"
XSP_V2_TRANSPORT_ORDER_AUTHORITY = "rth_cash_pair_limit_only"
XSP_V2_TRANSPORT_EXECUTION_VERSION = "xsp.opening-edge-v2-spyu-spxu-live-execution.v1"
XSP_V2_TRANSPORT_EXECUTION_SCHEMA = "xsp.opening-edge-v2-spyu-spxu-execution-checkpoint.v1"
XSP_V3_TRANSPORT_EXECUTION_VERSION = "xsp.opening-edge-v3-upro-spxu-live-execution.v1"
XSP_V3_TRANSPORT_EXECUTION_SCHEMA = (
    "xsp.opening-edge-v3-upro-spxu-execution-checkpoint.v1"
)
XSP_V3_TRANSPORT_CAPITAL_SLEEVE = "xsp-upro-spxu-rth-cash"
XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT = "ibkr_australia_trading_immediate_stock_sale_proceeds"
XSP_V3_TRANSPORT_SELECTION_SCHEMAS = frozenset({
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    XSP_V3_ROTATION_SELECTION_SCHEMA,
    XSP_V3_PACKAGE_SELECTION_SCHEMA,
})
_RANKING_SCHEMA = "xsp.opening-edge-v2-spyu-selection-ranking-result.v1"
_DWELL_SCHEMA = "xsp.network-b-symbol-dwell-validation-result.v1"
_PREVIEW_SCHEMA = "xsp.opening-edge-v2-ranked-nominee-preview.v1"
_SYMBOLS = ("SPYU", "SPXU")
_DIRECTION_SYMBOL = {"up": "SPYU", "down": "SPXU"}
_V3_SYMBOLS = ("UPRO", "SPXU")
_V3_DIRECTION_SYMBOL = {"up": "UPRO", "down": "SPXU"}
_SELECTION_MAX_AGE_SECONDS = 10 * 60.0
_SOURCE_MAX_AGE_SECONDS = 90.0
_BROKER_SNAPSHOT_MAX_AGE_SECONDS = 90.0
_QUOTE_MAX_AGE_SECONDS = 10.0
_NAV_MAX_AGE_SECONDS = 30.0
_STARTING_CASH_IDENTITY_USD = 1_350.0
_FIXED_NOTIONALS_USD = frozenset({1_050.0, 1_150.0, 1_200.0})
_POSITION_STATE_FIELDS = ("lane", "direction", "entry_time", "trading_date", "entry_price")


def _execution_contract() -> dict[str, object]:
    return {
        "SPYU_BUY": {"initial_mode": "CROSS", "chase_mode": "RELENTLESS"},
        "SPXU_BUY": {"initial_mode": "OPTIMISTIC", "chase_mode": "AUTO"},
        "SELL": {"initial_mode": "OPTIMISTIC", "chase_mode": "AUTO"},
        "sell_before_buy": True,
        "partial_buy": "hold_filled_quantity_without_top_up",
        "partial_sell": "no_new_buy_until_flat",
        "stale_or_ambiguous_state": "HOLD",
        "fresh_streaming_nbbo_required": True,
        "stale_top_action": "pause_repricing_until_fresh_or_timeout",
        "policy_contract": execution_policy_contract(),
    }


def _v3_execution_contract() -> dict[str, object]:
    return {
        "UPRO_BUY": {"initial_mode": "OPTIMISTIC", "chase_mode": "AUTO"},
        "SPXU_BUY": {"initial_mode": "OPTIMISTIC", "chase_mode": "AUTO"},
        "SELL": {"initial_mode": "OPTIMISTIC", "chase_mode": "AUTO"},
        "sell_before_buy": True,
        "partial_buy": "hold_filled_quantity_without_top_up",
        "partial_sell": "no_new_buy_until_flat",
        "stale_or_ambiguous_state": "HOLD",
        "fresh_streaming_nbbo_required": True,
        "stale_top_action": "pause_repricing_until_fresh_or_timeout",
        "policy_contract": execution_policy_contract(),
    }


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("transport timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def xsp_signal_utc(value: object) -> datetime:
    """Read the crown's canonical UTC-naive bar timestamp."""

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def _number(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _identity_valid(payload: Mapping[str, object]) -> bool:
    identity = payload.get("identity_sha256")
    return isinstance(identity, str) and identity == calibration_fingerprint(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    )


def _capital_identity(nominee: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(nominee, Mapping):
        raise ValueError("ranked cash-partition identity is invalid")
    family = str(nominee.get("family") or "")
    profile_id = str(nominee.get("profile_id") or "")
    notional = _number(
        nominee.get("fixed_entry_notional_usd"),
        name="fixed entry notional",
    )
    if family == "five_slot":
        valid = profile_id == "fixed_measured" and notional == 260.0
        cash_slots = 5
        maximum_gross = 1_300.0
    elif family == "two_slot":
        valid = profile_id == "fixed_measured" and notional == 650.0
        cash_slots = 2
        maximum_gross = 1_300.0
    elif family == "notional":
        valid = (
            profile_id == f"notional={int(notional)}:fixed_measured"
            and notional in _FIXED_NOTIONALS_USD
        )
        cash_slots = 1
        maximum_gross = notional
    else:
        valid = False
        cash_slots = 0
        maximum_gross = 0.0
    if not valid:
        raise ValueError("ranked cash-partition identity is invalid")
    return {
        "starting_cash_identity_usd": _STARTING_CASH_IDENTITY_USD,
        "fixed_entry_notional_usd": notional,
        "cash_slots": cash_slots,
        "maximum_gross_purchase_notional_usd": maximum_gross,
        "settlement": "strict_T_plus_1_settled_cash_only",
        "unsettled_sale_proceeds_reused": False,
    }


def _minimum_settled_cash_usd(nominee: Mapping[str, object]) -> float:
    """Return the exact cash partition plus worst modeled BUY commissions."""

    capital = nominee.get("capital_identity")
    commissions = nominee.get("commission_limits_usd")
    if not isinstance(capital, Mapping) or not isinstance(commissions, Mapping):
        raise ValueError("ranked cash reserve identity is incomplete")
    slots = int(
        _number(capital.get("cash_slots"), name="cash slots")
    )
    maximum_gross = _number(
        capital.get("maximum_gross_purchase_notional_usd"),
        name="maximum gross purchase notional",
    )
    commission = max(
        _number(commissions.get(symbol), name=f"{symbol} commission limit")
        for symbol in _SYMBOLS
    )
    if slots <= 0 or maximum_gross <= 0:
        raise ValueError("ranked cash reserve identity is invalid")
    return maximum_gross + slots * commission


def _validated_nominee(
    ranking: Mapping[str, object],
    dwell: Mapping[str, object],
    preview: Mapping[str, object],
) -> dict[str, object]:
    nominee = ranking.get("nominee")
    broker = preview.get("broker")
    rows = broker.get("rows") if isinstance(broker, Mapping) else None
    inputs = preview.get("inputs")
    if (
        ranking.get("schema") != _RANKING_SCHEMA
        or ranking.get("authority") != "research_ranking_only"
        or ranking.get("order_authority") != "none"
        or ranking.get("profitability_clock_started") is not False
        or ranking.get("selected_shadow_created") is not False
        or ranking.get("verdict") != "NOMINEE_STILL_HOLD"
        or not isinstance(nominee, Mapping)
        or dwell.get("schema") != _DWELL_SCHEMA
        or dwell.get("authority") != "historical_execution_validation_only"
        or dwell.get("order_authority") != "none"
        or dwell.get("submitted_orders") != 0
        or dwell.get("profitability_clock_started") is not False
        or dwell.get("selected_shadow_created") is not False
        or dwell.get("verdict") != "DWELL_VALIDATION_PASS_SELECTION_STILL_HOLD"
        or dwell.get("nominee_id") != nominee.get("nominee_id")
        or not _identity_valid(dwell)
        or preview.get("schema") != _PREVIEW_SCHEMA
        or preview.get("authority") != "fresh_broker_preview_only"
        or preview.get("order_authority") != "none"
        or preview.get("submitted_orders") != 0
        or preview.get("profitability_clock_started") is not False
        or preview.get("selected_shadow_created") is not False
        or preview.get("verdict") != "PREVIEW_PASS_SELECTION_STILL_HOLD"
        or preview.get("nominee") != nominee
        or not _identity_valid(preview)
        or not isinstance(inputs, Mapping)
        or not isinstance(broker, Mapping)
        or broker.get("submitted_orders") != 0
        or broker.get("all_previews_pass") is not True
        or broker.get("open_trades_before") != broker.get("open_trades_after")
        or not isinstance(rows, list)
        or len(rows) != len(_SYMBOLS)
    ):
        raise ValueError("transport evidence did not produce one previewed nominee")
    keyed = {str(row.get("symbol")): row for row in rows if isinstance(row, Mapping)}
    if set(keyed) != set(_SYMBOLS):
        raise ValueError("transport preview must bind both cash-pair symbols")
    ranges = nominee.get("historical_quantity_ranges")
    notional = _number(
        nominee.get("fixed_entry_notional_usd"),
        name="fixed entry notional",
    )
    if (
        notional <= 0
        or notional > _STARTING_CASH_IDENTITY_USD
        or not isinstance(ranges, Mapping)
        or set(ranges) != set(_SYMBOLS)
    ):
        raise ValueError("ranked cash identity is invalid")
    capital_identity = _capital_identity(nominee)
    commission_limits: dict[str, float] = {}
    preview_quantities: dict[str, int] = {}
    contract_ids: dict[str, int] = {}
    for symbol in _SYMBOLS:
        row = keyed[symbol]
        bounds = ranges[symbol]
        quantity = row.get("quote_derived_quantity")
        commission_limit = _number(
            row.get("commission_limit_usd"),
            name=f"{symbol} commission limit",
        )
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(value, int) for value in bounds)
            or int(bounds[0]) <= 0
            or int(bounds[1]) < int(bounds[0])
            or not isinstance(quantity, int)
            or not int(bounds[0]) <= quantity <= int(bounds[1])
            or row.get("fixed_entry_notional_usd") != notional
            or row.get("historical_quantity_range") != bounds
            or row.get("quantity_in_historical_range") is not True
            or row.get("preview_pass") is not True
            or not isinstance(row.get("contract"), Mapping)
            or int(row["contract"].get("con_id") or 0) <= 0
            or not isinstance(row.get("order"), Mapping)
            or row["order"].get("action") != "BUY"
            or row["order"].get("what_if") is not True
            or row["order"].get("transmit") is not False
            or commission_limit <= 0
        ):
            raise ValueError(f"{symbol}: exact preview identity is invalid")
        preview_quantities[symbol] = quantity
        commission_limits[symbol] = commission_limit
        contract_ids[symbol] = int(row["contract"]["con_id"])
    return {
        **dict(nominee),
        "fixed_entry_notional_usd": notional,
        "capital_identity": capital_identity,
        "preview_quantities": preview_quantities,
        "commission_limits_usd": commission_limits,
        "contract_ids": contract_ids,
    }


def select_xsp_v2_transport(
    *,
    ranking_path: Path,
    dwell_path: Path,
    preview_path: Path,
    source_receipt: Mapping[str, object],
    broker_snapshot: Mapping[str, object],
    selected_at: datetime,
) -> dict[str, object]:
    """Freeze one strictly post-selection RTH cash-pair run."""

    selected_utc = _utc(selected_at)
    ranking = _load(ranking_path)
    dwell = _load(dwell_path)
    preview = _load(preview_path)
    nominee = _validated_nominee(ranking, dwell, preview)
    minimum_settled_cash = _minimum_settled_cash_usd(nominee)
    preview_inputs = preview["inputs"]
    assert isinstance(preview_inputs, Mapping)
    preview_ranking = preview_inputs.get("ranking")
    preview_dwell = preview_inputs.get("dwell")
    if (
        not isinstance(preview_ranking, Mapping)
        or preview_ranking.get("sha256") != _sha256(ranking_path)
        or not isinstance(preview_dwell, Mapping)
        or preview_dwell.get("sha256") != _sha256(dwell_path)
    ):
        raise ValueError("preview does not bind the supplied ranking and dwell")
    preview_at = _utc(preview.get("observed_at_utc"))
    source_at = _utc(source_receipt.get("recorded_at_utc"))
    broker_at = _utc(broker_snapshot.get("observed_at_utc"))
    cash_at = _utc(broker_snapshot.get("cash_observed_at_utc"))
    if (
        selected_utc < preview_at
        or (selected_utc - preview_at).total_seconds() > _SELECTION_MAX_AGE_SECONDS
        or selected_utc < source_at
        or (selected_utc - source_at).total_seconds() > _SOURCE_MAX_AGE_SECONDS
        or source_receipt.get("evaluation_status") != "EVALUATED"
        or source_receipt.get("freshness_ok") is not True
        or source_receipt.get("session") != "RTH"
        or source_receipt.get("order_authority") != "none"
        or not source_receipt.get("checkpoint_id")
        or not isinstance(source_receipt.get("paired_equity"), Mapping)
    ):
        raise ValueError("selection requires one fresh evaluated RTH checkpoint")
    if (
        selected_utc < broker_at
        or (selected_utc - broker_at).total_seconds()
        > _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or broker_at < cash_at
        or (broker_at - cash_at).total_seconds()
        > _BROKER_SNAPSHOT_MAX_AGE_SECONDS
    ):
        raise ValueError("selection requires one fresh broker account snapshot")
    _, baseline_state = xsp_v2_position_state(source_receipt["paired_equity"])
    positions = broker_snapshot.get("positions")
    unrelated = broker_snapshot.get("unrelated_positions")
    open_orders = broker_snapshot.get("open_orders")
    account_id = str(broker_snapshot.get("account_id") or "").strip()
    settled_cash = _number(
        broker_snapshot.get("settled_cash_usd"),
        name="settled USD cash",
    )
    unrelated_positions = (
        [dict(row) for row in unrelated if isinstance(row, Mapping)]
        if isinstance(unrelated, Sequence)
        and not isinstance(unrelated, (str, bytes))
        else []
    )
    if (
        not account_id
        or str(broker_snapshot.get("account_type") or "").upper() != "CASH"
        or not isinstance(positions, Mapping)
        or not isinstance(unrelated, Sequence)
        or isinstance(unrelated, (str, bytes))
        or len(unrelated_positions) != len(unrelated)
        or any(str(row.get("symbol") or "") in _SYMBOLS for row in unrelated_positions)
        or any(
            abs(_number(positions.get(symbol, 0), name=f"{symbol} position")) > 1e-9
            for symbol in _SYMBOLS
        )
        or not isinstance(open_orders, Sequence)
        or isinstance(open_orders, (str, bytes))
        or bool(open_orders)
        or settled_cash < minimum_settled_cash
    ):
        raise ValueError(
            "selection requires a flat cash-pair sleeve and its settled USD reserve"
        )
    evidence = {
        "ranking": {"path": str(ranking_path), "sha256": _sha256(ranking_path)},
        "dwell": {"path": str(dwell_path), "sha256": _sha256(dwell_path)},
        "preview": {"path": str(preview_path), "sha256": _sha256(preview_path)},
        "source_checkpoint_id": source_receipt["checkpoint_id"],
        "source_recorded_at_utc": source_at.isoformat(),
    }
    body = {
        "schema": XSP_V2_TRANSPORT_SELECTION_SCHEMA,
        "selected_at_utc": selected_utc.isoformat(),
        "run_started_at_utc": selected_utc.isoformat(),
        "strategy_version": XSP_OPENING_EDGE_V2_VERSION,
        "authority": "selected_live_cash_transport",
        "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
        "profitability_clock_started": True,
        "execution_session": "RTH",
        "direction_symbols": dict(_DIRECTION_SYMBOL),
        "nominee": nominee,
        "baseline_state": baseline_state,
        "broker_at_selection": {
            "observed_at_utc": broker_at.isoformat(),
            "cash_observed_at_utc": cash_at.isoformat(),
            "account_id": account_id,
            "account_type": "CASH",
            "settled_cash_usd": settled_cash,
            "minimum_settled_cash_usd": minimum_settled_cash,
            "positions": {symbol: 0 for symbol in _SYMBOLS},
            "unrelated_positions": unrelated_positions,
            "open_orders": [],
        },
        "risk": {
            "starting_cash_identity_usd": _STARTING_CASH_IDENTITY_USD,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        },
        "execution": _execution_contract(),
        "evidence": evidence,
    }
    return {
        **body,
        "selection_id": calibration_fingerprint(body),
    }


def load_xsp_v2_transport_selection(path: Path) -> dict[str, object]:
    """Load one content-addressed selected transport or fail closed."""

    return load_xsp_v2_transport_selection_from_mapping(_load(path))


def write_xsp_v2_transport_selection(
    path: Path,
    selection: Mapping[str, object],
) -> None:
    """Atomically publish one already-validated selected transport."""

    load_xsp_v2_transport_selection_from_mapping(selection)
    payload = (
        json.dumps(selection, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_xsp_v2_transport_selection_from_mapping(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate an in-memory selected transport."""

    nominee = selection.get("nominee")
    ranges = (
        nominee.get("historical_quantity_ranges")
        if isinstance(
            nominee,
            Mapping,
        )
        else None
    )
    commissions = (
        nominee.get("commission_limits_usd")
        if isinstance(
            nominee,
            Mapping,
        )
        else None
    )
    contract_ids = (
        nominee.get("contract_ids")
        if isinstance(
            nominee,
            Mapping,
        )
        else None
    )
    risk = selection.get("risk")
    execution = selection.get("execution")
    broker = selection.get("broker_at_selection")
    baseline = selection.get("baseline_state")
    evidence = selection.get("evidence")
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    try:
        selected_at = _utc(selection.get("selected_at_utc"))
        run_started_at = _utc(selection.get("run_started_at_utc"))
        broker_at = _utc(
            broker.get("observed_at_utc") if isinstance(broker, Mapping) else None
        )
        cash_at = _utc(
            broker.get("cash_observed_at_utc")
            if isinstance(broker, Mapping)
            else None
        )
        notional = _number(
            nominee.get("fixed_entry_notional_usd")
            if isinstance(nominee, Mapping)
            else None,
            name="fixed entry notional",
        )
        settled_cash = _number(
            broker.get("settled_cash_usd") if isinstance(broker, Mapping) else None,
            name="selected settled cash",
        )
        capital_identity = _capital_identity(nominee)
        minimum_settled_cash = _minimum_settled_cash_usd(nominee)
        baseline_valid = baseline is None or bool(
            isinstance(baseline, Mapping)
            and set(baseline) == set(_POSITION_STATE_FIELDS)
            and baseline.get("lane") in {"rth", "gth"}
            and baseline.get("direction") in {"up", "down"}
            and (
                baseline.get("lane") == "rth"
                or baseline.get("direction") == "down"
            )
            and bool(str(baseline.get("trading_date") or ""))
            and xsp_signal_utc(baseline.get("entry_time"))
            and _number(baseline.get("entry_price"), name="baseline entry price") > 0
        )
        semantic_numbers_valid = bool(
            0 < notional <= _STARTING_CASH_IDENTITY_USD
            and settled_cash >= minimum_settled_cash
            and 0
            <= (selected_at - broker_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and 0
            <= (broker_at - cash_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and baseline_valid
            and nominee.get("capital_identity") == capital_identity
            and all(
                isinstance(ranges[symbol], list)
                and len(ranges[symbol]) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in ranges[symbol]
                )
                and 0 < int(ranges[symbol][0]) <= int(ranges[symbol][1])
                for symbol in _SYMBOLS
            )
        )
    except (KeyError, TypeError, ValueError):
        semantic_numbers_valid = False
        selected_at = run_started_at = broker_at = cash_at = (
            datetime.min.replace(tzinfo=timezone.utc)
        )
    if (
        selection.get("schema") != XSP_V2_TRANSPORT_SELECTION_SCHEMA
        or selection.get("strategy_version") != XSP_OPENING_EDGE_V2_VERSION
        or selection.get("authority") != "selected_live_cash_transport"
        or selection.get("order_authority") != XSP_V2_TRANSPORT_ORDER_AUTHORITY
        or selection.get("profitability_clock_started") is not True
        or selection.get("execution_session") != "RTH"
        or selection.get("direction_symbols") != _DIRECTION_SYMBOL
        or selected_at != run_started_at
        or not semantic_numbers_valid
        or not isinstance(ranges, Mapping)
        or set(ranges) != set(_SYMBOLS)
        or not isinstance(commissions, Mapping)
        or set(commissions) != set(_SYMBOLS)
        or not isinstance(contract_ids, Mapping)
        or set(contract_ids) != set(_SYMBOLS)
        or any(
            not isinstance(contract_ids[symbol], int)
            or isinstance(contract_ids[symbol], bool)
            or int(contract_ids[symbol]) <= 0
            for symbol in _SYMBOLS
        )
        or risk
        != {
            "starting_cash_identity_usd": _STARTING_CASH_IDENTITY_USD,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        }
        or execution != _execution_contract()
        or not isinstance(broker, Mapping)
        or not str(broker.get("account_id") or "").strip()
        or broker.get("account_type") != "CASH"
        or broker.get("minimum_settled_cash_usd") != minimum_settled_cash
        or broker.get("positions") != {"SPYU": 0, "SPXU": 0}
        or not isinstance(broker.get("unrelated_positions"), list)
        or any(
            not isinstance(row, Mapping) or str(row.get("symbol") or "") in _SYMBOLS
            for row in broker["unrelated_positions"]
        )
        or broker.get("open_orders") != []
        or not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "ranking",
            "dwell",
            "preview",
            "source_checkpoint_id",
            "source_recorded_at_utc",
        }
        or selection.get("selection_id") != calibration_fingerprint(body)
    ):
        raise ValueError("invalid selected XSP v2 cash transport")
    return dict(selection)


def load_xsp_transport_selection_from_mapping(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate either supported selected cash transport."""

    if selection.get("schema") == XSP_V2_TRANSPORT_SELECTION_SCHEMA:
        return load_xsp_v2_transport_selection_from_mapping(selection)
    if selection.get("schema") in XSP_V3_TRANSPORT_SELECTION_SCHEMAS:
        from .xsp_live_transport_v3 import (
            load_xsp_v3_transport_selection_from_mapping,
        )

        return load_xsp_v3_transport_selection_from_mapping(selection)
    raise ValueError("unsupported selected XSP cash transport")


def load_xsp_transport_selection(path: Path) -> dict[str, object]:
    return load_xsp_transport_selection_from_mapping(_load(path))


def xsp_transport_contract(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact runtime identity carried by a validated selection."""

    selected = load_xsp_transport_selection_from_mapping(selection)
    if selected["schema"] == XSP_V2_TRANSPORT_SELECTION_SCHEMA:
        return {
            "symbols": _SYMBOLS,
            "direction_symbols": dict(_DIRECTION_SYMBOL),
            "plan_schema": XSP_V2_TRANSPORT_PLAN_SCHEMA,
            "execution_version": XSP_V2_TRANSPORT_EXECUTION_VERSION,
            "execution_schema": XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
            "order_ref_prefix": "XSPV2",
            "ticker_owner": "xsp-v2-live-transport",
            "generic_ticks": {"SPYU": "577,614,623"},
            "nav_symbol": "SPYU",
        }
    if selected["schema"] not in XSP_V3_TRANSPORT_SELECTION_SCHEMAS:
        raise ValueError("unsupported selected XSP cash transport")
    return {
        "symbols": _V3_SYMBOLS,
        "direction_symbols": dict(_V3_DIRECTION_SYMBOL),
        "plan_schema": XSP_V3_TRANSPORT_PLAN_SCHEMA,
        "execution_version": XSP_V3_TRANSPORT_EXECUTION_VERSION,
        "execution_schema": XSP_V3_TRANSPORT_EXECUTION_SCHEMA,
        "order_ref_prefix": "XSPV3",
        "ticker_owner": "xsp-v3-live-transport",
        "generic_ticks": {},
        "nav_symbol": None,
    }


def _fresh_quote(
    quote: Mapping[str, object] | None,
    *,
    symbol: str,
) -> tuple[float, float]:
    if not isinstance(quote, Mapping):
        raise ValueError(f"{symbol}: fresh executable quote is missing")
    bid = _number(quote.get("bid"), name=f"{symbol} bid")
    ask = _number(quote.get("ask"), name=f"{symbol} ask")
    age = _number(quote.get("age_seconds"), name=f"{symbol} quote age")
    if (
        quote.get("market_data_type") != 1
        or bid <= 0
        or ask < bid
        or age < 0
        or age > _QUOTE_MAX_AGE_SECONDS
    ):
        raise ValueError(f"{symbol}: quote is not fresh and executable")
    return bid, ask


def _post_selection_target(
    selection: Mapping[str, object],
    source_receipt: Mapping[str, object],
    *,
    observed_at: datetime,
) -> tuple[str | None, str | None, bool]:
    contract = xsp_transport_contract(selection)
    direction_symbols = contract["direction_symbols"]
    assert isinstance(direction_symbols, Mapping)
    selected_at = _utc(selection["selected_at_utc"])
    source_at = _utc(source_receipt.get("recorded_at_utc"))
    source_session = str(source_receipt.get("session") or "")
    observed_session = xsp_session_label_et(observed_at)
    if (
        observed_at <= selected_at
        or source_at <= selected_at
        or (observed_at - source_at).total_seconds() > _SOURCE_MAX_AGE_SECONDS
        or source_session not in {"GTH", "RTH", "CURB"}
        or source_session != observed_session
        or source_receipt.get("order_authority") != "none"
        or not isinstance(source_receipt.get("paired_equity"), Mapping)
    ):
        raise ValueError("live transport requires a fresh post-selection source")
    if (source_receipt.get("evaluation_status"), source_receipt.get("freshness_ok")) != ("EVALUATED", True):
        return None, None, False
    _, raw_target = xsp_v2_position_state(source_receipt["paired_equity"])
    if source_session != "RTH":
        return None, None, True
    if raw_target is None:
        return None, None, True
    if raw_target == selection.get("baseline_state"):
        continuity = selection.get("continuity")
        if (
            selection.get("schema") != XSP_V3_ROTATION_SELECTION_SCHEMA
            or not isinstance(continuity, Mapping)
            or raw_target != continuity.get("source_target_state")
        ):
            return None, None, True
        direction = str(raw_target.get("direction") or "")
        return direction, direction_symbols.get(direction), True
    if raw_target.get("lane") != "rth":
        raise ValueError("GTH execution is forbidden")
    direction = str(raw_target.get("direction")) if xsp_signal_utc(raw_target.get("entry_time")) > selected_at else None
    return direction, direction_symbols.get(direction), True


def _cash_pair_holdings(
    positions: Mapping[str, object],
    *,
    symbols: Sequence[str],
) -> dict[str, int]:
    holdings: dict[str, int] = {}
    for symbol in symbols:
        quantity = _number(positions.get(symbol, 0), name=f"{symbol} position")
        if quantity < 0 or abs(quantity - round(quantity)) > 1e-9:
            raise ValueError("cash-pair positions must be nonnegative whole shares")
        holdings[symbol] = int(round(quantity))
    if sum(quantity > 0 for quantity in holdings.values()) > 1:
        raise ValueError("both cash-pair symbols are held")
    return holdings


def project_xsp_transport_plan(
    *,
    selection: Mapping[str, object],
    source_receipt: Mapping[str, object],
    observed_at: datetime,
    positions: Mapping[str, object],
    open_orders: Sequence[Mapping[str, object]],
    settled_cash_usd: float,
    quotes: Mapping[str, Mapping[str, object]],
    spyu_nav: Mapping[str, object] | None = None,
    session_net_usd: float = 0.0,
    drawdown_usd: float = 0.0,
) -> dict[str, object]:
    """Project at most one sell-or-buy leg; never submit an order."""

    selected = load_xsp_transport_selection_from_mapping(selection)
    contract = xsp_transport_contract(selected)
    symbols = contract["symbols"]
    assert isinstance(symbols, Sequence)
    plan_schema = str(contract["plan_schema"])
    observed_utc = _utc(observed_at)
    target_direction, target_symbol, source_executable = _post_selection_target(
        selected, source_receipt, observed_at=observed_utc,
    )
    source_session = str(source_receipt["session"])
    signal_context = xsp_execution_signal_context(source_receipt["paired_equity"])
    execution_state_context = None
    if plan_schema == XSP_V3_TRANSPORT_PLAN_SCHEMA:
        execution_state_context = xsp_execution_state_context(source_receipt)
    observed_et = observed_utc.astimezone(ET_ZONE)
    rth_close = equity_rth_close_time_et(observed_et.date())
    rth_liquidation_at = datetime.combine(
        observed_et.date(), rth_close, tzinfo=ET_ZONE
    ) - timedelta(minutes=3)
    equity_rth_open = time(9, 30) <= observed_et.time().replace(tzinfo=None) < rth_close
    entry_window_open = equity_rth_open and observed_et < rth_liquidation_at
    holdings = _cash_pair_holdings(positions, symbols=symbols)
    held = [symbol for symbol, quantity in holdings.items() if quantity > 0]
    relevant_orders = [dict(row) for row in open_orders if isinstance(row, Mapping)]
    if len(relevant_orders) != len(open_orders):
        raise ValueError("open-order snapshot is invalid")
    cash_pair_orders = [
        row for row in relevant_orders if str(row.get("symbol") or "") in symbols
    ]
    transition = {
        "selection_id": selected["selection_id"],
        "source_checkpoint_id": source_receipt.get("checkpoint_id"),
        "source_session": source_session,
        "source_evaluation_status": source_receipt.get("evaluation_status"),
        "source_freshness_ok": source_receipt.get("freshness_ok"),
        "target_direction": target_direction,
        "target_symbol": target_symbol,
        "entry_window_open": entry_window_open,
        "holdings": holdings,
        "signal_context": signal_context,
        "execution_state_context": execution_state_context,
    }
    base = {
        "schema": plan_schema,
        "observed_at_utc": observed_utc.isoformat(),
        "transition_id": calibration_fingerprint(transition),
        **transition,
        "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
        "submitted_orders": 0,
    }
    if cash_pair_orders:
        return {
            **base,
            "status": "RECONCILE_REQUIRED",
            "reason": "cash_pair_open_order_exists",
            "leg": None,
            "open_orders": cash_pair_orders,
        }
    if held and (held[0] != target_symbol or not entry_window_open):
        if signal_context is None and target_symbol is not None:
            raise ValueError("actionable transport has no causal signal context")
        symbol = held[0]
        bid, ask = _fresh_quote(quotes.get(symbol), symbol=symbol)
        leg = {
            "action": "SELL", "symbol": symbol, "quantity": holdings[symbol],
            "initial_mode": "OPTIMISTIC", "chase_mode": "AUTO",
            "outside_rth": not equity_rth_open, "bid": bid, "ask": ask,
        }
        if selected["schema"] == XSP_V3_PACKAGE_SELECTION_SCHEMA:
            leg["commission_limit_usd"] = tiered_us_stock_commission_ceiling(
                holdings[symbol]
            )
        return {
            **base,
            "status": "ACTIONABLE",
            "reason": (
                "rth_end_liquidation"
                if not entry_window_open
                else "sell_incumbent_before_target"
            ),
            "leg": leg,
        }
    if held or target_symbol is None:
        return {
            **base,
            "status": "UNCHANGED",
            "reason": (
                "target_already_owned" if held
                else "flat_target" if source_executable
                else "source_not_executable"
            ),
            "leg": None,
        }
    if not entry_window_open:
        return {
            **base,
            "status": "UNCHANGED",
            "reason": "rth_entry_cutoff",
            "leg": None,
        }
    if relevant_orders:
        return {
            **base,
            "status": "RECONCILE_REQUIRED",
            "reason": "unrelated_open_order_blocks_buy",
            "leg": None,
            "open_orders": relevant_orders,
        }

    risk = selected.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("selected transport risk identity is missing")
    session_net = _number(session_net_usd, name="session net")
    drawdown = _number(drawdown_usd, name="run drawdown")
    if session_net <= -_number(
        risk.get("max_session_loss_usd"),
        name="maximum session loss",
    ) or drawdown >= _number(
        risk.get("max_drawdown_usd"),
        name="maximum drawdown",
    ):
        return {
            **base,
            "status": "RISK_HALTED",
            "reason": "loss_limit_reached",
            "leg": None,
            "risk_state": {
                "session_net_usd": session_net,
                "drawdown_usd": drawdown,
            },
        }

    bid, ask = _fresh_quote(quotes.get(target_symbol), symbol=target_symbol)
    if signal_context is None:
        raise ValueError("actionable transport has no causal signal context")
    nominee = selected["nominee"]
    assert isinstance(nominee, Mapping)
    notional = _number(
        nominee.get("fixed_entry_notional_usd"),
        name="fixed entry notional",
    )
    quantity = math.floor(notional / ask)
    ranges = nominee.get("historical_quantity_ranges")
    commissions = nominee.get("commission_limits_usd")
    if not isinstance(ranges, Mapping) or not isinstance(commissions, Mapping):
        raise ValueError("selected quantity or commission identity is missing")
    bounds = ranges[target_symbol]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("selected quantity identity is invalid")
    package = selected["schema"] == XSP_V3_PACKAGE_SELECTION_SCHEMA
    if not package and not int(bounds[0]) <= quantity <= int(bounds[1]):
        raise ValueError("quote-derived quantity left its historical range")
    commission = (
        tiered_us_stock_commission_ceiling(quantity)
        if package
        else _number(commissions[target_symbol], name=f"{target_symbol} commission limit")
    )
    required_cash = quantity * ask + commission
    if _number(settled_cash_usd, name="settled cash") < required_cash:
        raise ValueError("insufficient settled USD cash")
    nav_symbol = contract["nav_symbol"]
    if target_symbol == nav_symbol:
        if not isinstance(spyu_nav, Mapping):
            raise ValueError(f"{nav_symbol} entry requires fresh indicative value")
        nav = _number(spyu_nav.get("value"), name="SPYU NAV")
        nav_age = _number(spyu_nav.get("age_seconds"), name="SPYU NAV age")
        midpoint = (bid + ask) / 2.0
        spread_fraction = (ask - bid) / midpoint
        divergence = abs(midpoint - nav) / nav
        if (
            nav <= 0
            or nav_age < 0
            or nav_age > _NAV_MAX_AGE_SECONDS
            or divergence > max(0.005, spread_fraction)
        ):
            raise ValueError(f"{nav_symbol} indicative value is stale or divergent")
    else:
        divergence = None
    execution = selected["execution"]
    assert isinstance(execution, Mapping)
    buy_policy = execution.get(f"{target_symbol}_BUY")
    if not isinstance(buy_policy, Mapping):
        raise ValueError("selected transport has no BUY execution policy")
    return {
        **base,
        "status": "ACTIONABLE",
        "reason": "buy_post_selection_target",
        "leg": {
            "action": "BUY",
            "symbol": target_symbol,
            "quantity": quantity,
            "initial_mode": buy_policy["initial_mode"],
            "chase_mode": buy_policy["chase_mode"],
            "outside_rth": False,
            "bid": bid,
            "ask": ask,
            "fixed_entry_notional_usd": notional,
            "commission_limit_usd": commission,
            "required_settled_cash_usd": required_cash,
            "spyu_nav_divergence": divergence,
        },
    }


def project_xsp_v2_transport_plan(
    **kwargs,
) -> dict[str, object]:
    """Backward-compatible v2 projection wrapper."""

    selection = kwargs.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema") != XSP_V2_TRANSPORT_SELECTION_SCHEMA
    ):
        raise ValueError("v2 projection requires a v2 selected transport")
    return project_xsp_transport_plan(**kwargs)
