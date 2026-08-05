"""Selection proof for the Opening Edge v3 UPRO/SPXU cash transport."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from ..engines.market import xsp_trading_date
from ..live.order_evidence import tiered_us_stock_commission_ceiling
from .live_calibration import (
    SELECTED_CASH_EQUITY_SCHEMA,
    XspProfitabilityPolicy,
    calibration_fingerprint,
)
from .xsp_execution_observer import xsp_v2_position_state
from .xsp_live_transport import (
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
    XSP_V3_PACKAGE_SELECTION_SCHEMA,
    XSP_V3_ROTATION_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
    XSP_V3_TRANSPORT_EXECUTION_VERSION,
    _BROKER_SNAPSHOT_MAX_AGE_SECONDS,
    _POSITION_STATE_FIELDS,
    _SELECTION_MAX_AGE_SECONDS,
    _SOURCE_MAX_AGE_SECONDS,
    _V3_DIRECTION_SYMBOL,
    _V3_SYMBOLS,
    _load,
    _number,
    _sha256,
    _utc,
    _v3_execution_contract,
    xsp_signal_utc,
)
from .xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA,
    XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
    XSP_OPENING_EDGE_V3_VERSION,
)


_PREVIEW_SCHEMA = "xsp.opening-edge-v3-upro-spxu-preview.v1"
_CASH_SCHEMA = "xsp.opening-edge-v3-regime-harmony-cash-qualification.v1"
_CASH_RECEIPT_SHA256 = (
    "e41e44db270ea872679746cb2b83b2aa73987e523863236472f6e8ec0434c8dc"
)
_CROWN_SHA256 = "d47eb39cef3d2ca575d779d6b5b87e3b88e08606fd09a8801b8cb55c350208db"
_RTH_LEDGER_SHA256 = (
    "40708a28476fad504d29b1222e52b88481b1c0da8313ed4667076e2b6ed3f157"
)
_CONTEXT_HORIZONS = frozenset({"5", "10", "21", "42", "63", "84"})
_IMMEDIATE_PROCEEDS_SCHEMA = (
    "xsp.opening-edge-v3-immediate-proceeds-receipt.v1"
)
_IMMEDIATE_PROCEEDS_SHA256 = (
    "bbe079289882d06f3bf7f7a1351595ff9d2392d80f3959fff34362629cf4fd71"
)
_CONTINUITY_SCHEMA = "xsp.opening-edge-v3-selection-continuity.v1"
_RESET_SCHEMA = "xsp.opening-edge-v3-flat-reset.v1"


def _sha256_identity(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validated_nominee(
    cash_receipt: Mapping[str, object],
    preview: Mapping[str, object],
) -> dict[str, object]:
    profile = (
        cash_receipt.get("profiles", {}).get("tiered_conservative_full_cross")
        if isinstance(cash_receipt.get("profiles"), Mapping)
        else None
    )
    gate = cash_receipt.get("selection_gate")
    source = preview.get("source")
    rows = preview.get("rows")
    if (
        cash_receipt.get("schema") != _CASH_SCHEMA
        or cash_receipt.get("authority") != "historical_cash_qualification_only"
        or cash_receipt.get("scope") != "RTH UPRO/SPXU whole-share transport only"
        or cash_receipt.get("signal_rth_ledger_sha256") != _RTH_LEDGER_SHA256
        or cash_receipt.get("submitted_orders") != 0
        or not isinstance(profile, Mapping)
        or profile.get("verdict")
        != "economically_preferred_but_effective_commission_unproved"
        or not isinstance(gate, Mapping)
        or gate.get("selected") is not False
        or gate.get("order_authority") != "none"
        or preview.get("schema") != _PREVIEW_SCHEMA
        or preview.get("authority") != "broker_preview_only"
        or preview.get("cash_receipt_sha256") != _CASH_RECEIPT_SHA256
        or preview.get("notional_usd") != 900.0
        or preview.get("books_pass") is not True
        or preview.get("quantity_and_cash_pass") is not True
        or preview.get("effective_tiered_commission_pass") is not True
        or preview.get("selection_created") is not False
        or preview.get("profitability_clock_started") is not False
        or preview.get("order_authority") != "none"
        or preview.get("submitted_orders") != 0
        or preview.get("verdict") != "PREVIEW_PASS_STILL_HOLD"
        or preview.get("open_trades_before") != preview.get("open_trades_after")
        or preview.get("relevant_positions") != []
        or preview.get("relevant_open_orders") != []
        or not isinstance(source, Mapping)
        or source.get("crown_artifact_sha256") != _CROWN_SHA256
        or not _sha256_identity(source.get("state_owner_sha256"))
        or not _sha256_identity(source.get("daily_context_fingerprint"))
        or not isinstance(rows, list)
        or len(rows) != len(_V3_SYMBOLS)
    ):
        raise ValueError("v3 cash evidence did not produce one previewed nominee")
    preview_at = _utc(preview.get("observed_at_utc"))
    ranges = profile.get("quantity_ranges")
    keyed = {str(row.get("symbol")): row for row in rows if isinstance(row, Mapping)}
    if (
        not isinstance(ranges, Mapping)
        or ranges != {"UPRO": [6, 24], "SPXU": [3, 24]}
        or set(keyed) != set(_V3_SYMBOLS)
    ):
        raise ValueError("v3 cash quantity identity is invalid")
    preview_quantities: dict[str, int] = {}
    contract_ids: dict[str, int] = {}
    for symbol in _V3_SYMBOLS:
        row = keyed[symbol]
        contract = row.get("contract")
        order = row.get("order")
        what_if = row.get("preview")
        books = row.get("books")
        quantity = order.get("quantity") if isinstance(order, Mapping) else None
        modeled = _number(
            row.get("tiered_conservative_buy_fee_usd"),
            name=f"{symbol} modeled Tiered commission",
        )
        commission_values = (
            [
                _number(value, name=f"{symbol} preview {key}")
                for key in ("commission", "min_commission", "max_commission")
                if (value := what_if.get(key)) is not None
            ]
            if isinstance(what_if, Mapping)
            else []
        )
        bounds = ranges[symbol]
        if (
            not isinstance(contract, Mapping)
            or int(contract.get("con_id") or 0) <= 0
            or contract.get("exchange") != "SMART"
            or contract.get("primary_exchange") != "ARCA"
            or contract.get("currency") != "USD"
            or not isinstance(order, Mapping)
            or order.get("action") != "BUY"
            or not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or not int(bounds[0]) <= quantity <= int(bounds[1])
            or order.get("what_if") is not True
            or order.get("transmit") is not False
            or order.get("tif") != "DAY"
            or not isinstance(what_if, Mapping)
            or what_if.get("commission_currency") != "USD"
            or not commission_values
            or min(commission_values) <= 0
            or max(commission_values) > modeled + 0.01
            or row.get("effective_tiered_commission") is not True
            or row.get("cash_fit") is not True
            or not isinstance(books, Mapping)
        ):
            raise ValueError(f"{symbol}: exact v3 preview identity is invalid")
        for venue in ("smart", "direct_arca"):
            book = books.get(venue)
            bid = _number(book.get("bid"), name=f"{symbol} {venue} bid") if isinstance(
                book, Mapping
            ) else 0.0
            ask = _number(book.get("ask"), name=f"{symbol} {venue} ask") if isinstance(
                book, Mapping
            ) else 0.0
            book_at = (
                _utc(book.get("observed_at_utc"))
                if isinstance(book, Mapping)
                else datetime.min.replace(tzinfo=preview_at.tzinfo)
            )
            if (
                not isinstance(book, Mapping)
                or book.get("market_data_type") != 1
                or bid <= 0
                or ask < bid
                or not 0 <= (preview_at - book_at).total_seconds() <= 30.0
            ):
                raise ValueError(f"{symbol}: fresh {venue} book is invalid")
        preview_quantities[symbol] = quantity
        contract_ids[symbol] = int(contract["con_id"])
    commission_limits = {
        symbol: tiered_us_stock_commission_ceiling(int(ranges[symbol][1]))
        for symbol in _V3_SYMBOLS
    }
    return {
        "profile_id": "tiered_conservative_full_cross",
        "pricing_plan": "Tiered",
        "fixed_entry_notional_usd": 900.0,
        "historical_quantity_ranges": dict(ranges),
        "preview_quantities": preview_quantities,
        "commission_limits_usd": commission_limits,
        "contract_ids": contract_ids,
        "capital_identity": {
            "starting_cash_identity_usd": 900.0,
            "fixed_entry_notional_usd": 900.0,
            "cash_slots": 1,
            "maximum_gross_purchase_notional_usd": 900.0,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "unsettled_sale_proceeds_reused": False,
        },
    }


def select_xsp_v3_transport(
    *,
    cash_receipt_path: Path,
    preview_path: Path,
    source_receipt: Mapping[str, object],
    broker_snapshot: Mapping[str, object],
    selected_at: datetime,
    rth_scope_accepted: bool,
) -> dict[str, object]:
    """Freeze one v3 RTH-only UPRO/SPXU run after every external gate passes."""

    if not rth_scope_accepted:
        raise ValueError("v3 cash selection requires explicit RTH-only scope acceptance")
    selected_utc = _utc(selected_at)
    cash_receipt = _load(cash_receipt_path)
    preview = _load(preview_path)
    if _sha256(cash_receipt_path) != _CASH_RECEIPT_SHA256:
        raise ValueError("v3 cash receipt identity changed")
    nominee = _validated_nominee(cash_receipt, preview)
    preview_at = _utc(preview.get("observed_at_utc"))
    source_at = _utc(source_receipt.get("recorded_at_utc"))
    broker_at = _utc(broker_snapshot.get("observed_at_utc"))
    cash_at = _utc(broker_snapshot.get("cash_observed_at_utc"))
    paired = source_receipt.get("paired_equity")
    context = (
        paired.get("daily_context_state")
        if isinstance(paired, Mapping)
        else None
    )
    context_state = context.get("state") if isinstance(context, Mapping) else None
    try:
        context_day = date.fromisoformat(str(context.get("trading_day")))
        context_as_of = date.fromisoformat(str(context.get("context_as_of_day")))
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError(
            "v3 selection requires canonical causal daily context"
        ) from None
    if (
        not isinstance(paired, Mapping)
        or not _sha256_identity(paired.get("state_owner_sha256"))
        or not _sha256_identity(paired.get("daily_context_fingerprint"))
        or not isinstance(context, Mapping)
        or context.get("schema") != XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA
        or not isinstance(context_state, Mapping)
        or context.get("context_as_of_day") != context_state.get("as_of_day")
        or context.get("state_fingerprint")
        != calibration_fingerprint(context_state)
        or context_as_of >= context_day
        or context_day != xsp_trading_date(source_at)
        or any(
            not isinstance(context_state.get(field), Mapping)
            or set(context_state[field]) != _CONTEXT_HORIZONS
            for field in ("windows", "return_velocity", "return_acceleration")
        )
    ):
        raise ValueError("v3 selection requires canonical causal daily context")
    if (
        not 0
        <= (selected_utc - preview_at).total_seconds()
        <= _SELECTION_MAX_AGE_SECONDS
        or not 0
        <= (selected_utc - source_at).total_seconds()
        <= _SOURCE_MAX_AGE_SECONDS
        or source_receipt.get("evaluation_status") != "EVALUATED"
        or source_receipt.get("freshness_ok") is not True
        or source_receipt.get("session") != "RTH"
        or source_receipt.get("order_authority") != "none"
        or not _sha256_identity(source_receipt.get("checkpoint_id"))
        or not isinstance(source_receipt.get("paired_equity"), Mapping)
        or not 0
        <= (selected_utc - broker_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or not 0
        <= (broker_at - cash_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
    ):
        raise ValueError("v3 selection requires fresh preview, source, and broker state")
    _, baseline_state = xsp_v2_position_state(source_receipt["paired_equity"])
    positions = broker_snapshot.get("positions")
    unrelated = broker_snapshot.get("unrelated_positions")
    open_orders = broker_snapshot.get("open_orders")
    settled_cash = _number(
        broker_snapshot.get("settled_cash_usd"),
        name="settled USD cash",
    )
    minimum_settled_cash = 900.0 + max(
        float(value) for value in nominee["commission_limits_usd"].values()
    )
    if (
        not str(broker_snapshot.get("account_id") or "").strip()
        or str(broker_snapshot.get("account_type") or "").upper() != "CASH"
        or not isinstance(positions, Mapping)
        or any(
            abs(_number(positions.get(symbol, 0), name=f"{symbol} position")) > 1e-9
            for symbol in _V3_SYMBOLS
        )
        or not isinstance(unrelated, Sequence)
        or isinstance(unrelated, (str, bytes))
        or any(
            not isinstance(row, Mapping)
            or str(row.get("symbol") or "") in _V3_SYMBOLS
            for row in unrelated
        )
        or not isinstance(open_orders, Sequence)
        or isinstance(open_orders, (str, bytes))
        or bool(open_orders)
        or settled_cash < minimum_settled_cash
    ):
        raise ValueError("v3 selection requires a flat funded cash-pair sleeve")
    evidence = {
        "cash_receipt": {
            "path": str(cash_receipt_path),
            "sha256": _sha256(cash_receipt_path),
        },
        "preview": {"path": str(preview_path), "sha256": _sha256(preview_path)},
        "source_checkpoint_id": source_receipt["checkpoint_id"],
        "source_recorded_at_utc": source_at.isoformat(),
        "source_daily_context": {
            "schema": context["schema"],
            "trading_day": context_day.isoformat(),
            "context_as_of_day": context_as_of.isoformat(),
            "state_fingerprint": context["state_fingerprint"],
        },
        "rth_scope": {
            "accepted": True,
            "historical_trades": 423,
            "trades_per_year": 141.0,
            "gth_execution_allowed": False,
        },
    }
    body = {
        "schema": XSP_V3_TRANSPORT_SELECTION_SCHEMA,
        "selected_at_utc": selected_utc.isoformat(),
        "run_started_at_utc": selected_utc.isoformat(),
        "strategy_version": XSP_OPENING_EDGE_V3_VERSION,
        "source_strategy_version": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "authority": "selected_live_cash_transport",
        "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
        "profitability_clock_started": True,
        "execution_session": "RTH",
        "direction_symbols": dict(_V3_DIRECTION_SYMBOL),
        "nominee": nominee,
        "baseline_state": baseline_state,
        "broker_at_selection": {
            "observed_at_utc": broker_at.isoformat(),
            "cash_observed_at_utc": cash_at.isoformat(),
            "account_id": broker_snapshot["account_id"],
            "account_type": "CASH",
            "settled_cash_usd": settled_cash,
            "minimum_settled_cash_usd": minimum_settled_cash,
            "positions": {symbol: 0 for symbol in _V3_SYMBOLS},
            "unrelated_positions": [dict(row) for row in unrelated],
            "open_orders": [],
        },
        "risk": {
            "starting_cash_identity_usd": 900.0,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        },
        "execution": _v3_execution_contract(),
        "evidence": evidence,
    }
    return {**body, "selection_id": calibration_fingerprint(body)}


def _inherited_fills(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
) -> list[dict[str, object]]:
    fills: dict[str, dict[str, object]] = {}
    for record in records:
        evidence = record.get("evidence")
        order = (
            evidence.get("broker_order")
            if isinstance(evidence, Mapping)
            and evidence.get("selection_id") == selection_id
            and evidence.get("phase") == "TERMINAL"
            else None
        )
        if not isinstance(order, Mapping):
            continue
        rows = order.get("fills")
        if not isinstance(rows, list):
            raise ValueError("predecessor terminal order has no fills")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("predecessor fill is invalid")
            row = dict(raw)
            exec_id = str(row.get("exec_id") or "")
            if not exec_id or (exec_id in fills and fills[exec_id] != row):
                raise ValueError("predecessor execution identity is invalid")
            fills[exec_id] = row
    return sorted(
        fills.values(),
        key=lambda row: (
            str(row.get("time_utc") or ""),
            str(row.get("exec_id") or ""),
        ),
    )


def _inherited_cash_and_holdings(
    fills: Sequence[Mapping[str, object]],
    *,
    starting_cash_usd: float,
) -> tuple[float, dict[str, int]]:
    cash = float(starting_cash_usd)
    holdings = {symbol: 0 for symbol in _V3_SYMBOLS}
    seen = set()
    for fill in fills:
        try:
            exec_id = str(fill["exec_id"])
            symbol = str(fill["symbol"])
            side = str(fill["side"]).upper()
            shares = float(fill["shares"])
            price = float(fill["price"])
            commission = float(fill["commission"])
            fill_at = _utc(fill["time_utc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("inherited fill economics are incomplete") from exc
        if (
            not exec_id
            or exec_id in seen
            or symbol not in _V3_SYMBOLS
            or side not in {"BOT", "BUY"}
            or shares <= 0
            or shares != int(shares)
            or price <= 0
            or commission < 0
            or str(fill.get("commission_currency") or "").upper() != "USD"
            or fill_at > datetime.now(fill_at.tzinfo)
        ):
            raise ValueError("only one open whole-share predecessor BUY is inheritable")
        seen.add(exec_id)
        holdings[symbol] += int(shares)
        cash -= shares * price + commission
    if (
        not fills
        or cash < -1e-7
        or sum(quantity > 0 for quantity in holdings.values()) != 1
    ):
        raise ValueError("predecessor fills do not define one funded open position")
    return cash, holdings


def xsp_v3_transport_profitability_policy(
    selection: Mapping[str, object],
) -> XspProfitabilityPolicy:
    """Bind the shared milestone verifier to one immutable v3 cash selection."""

    selected = load_xsp_v3_transport_selection_from_mapping(selection)
    risk = selected["risk"]
    assert isinstance(risk, Mapping)
    return XspProfitabilityPolicy(
        run_id=str(selected["selection_id"]),
        strategy_id=XSP_OPENING_EDGE_V3_VERSION,
        strategy_version=XSP_V3_TRANSPORT_EXECUTION_VERSION,
        config_fingerprint=str(selected["selection_id"]),
        capital_sleeve=XSP_V3_TRANSPORT_CAPITAL_SLEEVE,
        max_drawdown_points=float(risk["max_drawdown_usd"]),
        max_session_loss_points=float(risk["max_session_loss_usd"]),
        minimum_week_closed_trades=2,
        maximum_top_five_win_share=0.5,
        slot_tolerance_seconds=90.0,
        unit="USD",
        equity_schema=SELECTED_CASH_EQUITY_SCHEMA,
    )


def load_xsp_v3_transport_selection(path: Path) -> dict[str, object]:
    """Load one content-addressed v3 selection or fail closed."""

    return load_xsp_v3_transport_selection_from_mapping(_load(path))


def write_xsp_v3_transport_selection(
    path: Path,
    selection: Mapping[str, object],
) -> None:
    """Atomically persist one fully validated v3 selection."""

    load_xsp_v3_transport_selection_from_mapping(selection)
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


def _load_xsp_v3_initial_selection_from_mapping(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate the original flat-start v3 UPRO/SPXU selection."""

    nominee = selection.get("nominee")
    broker = selection.get("broker_at_selection")
    evidence = selection.get("evidence")
    source_context = (
        evidence.get("source_daily_context")
        if isinstance(evidence, Mapping)
        else None
    )
    baseline = selection.get("baseline_state")
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    try:
        selected_at = _utc(selection.get("selected_at_utc"))
        source_at = _utc(evidence["source_recorded_at_utc"])
        broker_at = _utc(
            broker.get("observed_at_utc") if isinstance(broker, Mapping) else None
        )
        cash_at = _utc(
            broker.get("cash_observed_at_utc") if isinstance(broker, Mapping) else None
        )
        ranges = nominee["historical_quantity_ranges"]
        commissions = nominee["commission_limits_usd"]
        contract_ids = nominee["contract_ids"]
        preview_quantities = nominee["preview_quantities"]
        minimum_cash = 900.0 + max(float(value) for value in commissions.values())
        source_context_day = date.fromisoformat(
            str(source_context["trading_day"])
        )
        source_context_as_of = date.fromisoformat(
            str(source_context["context_as_of_day"])
        )
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
        semantic_valid = bool(
            selection.get("run_started_at_utc") == selected_at.isoformat()
            and 0
            <= (selected_at - source_at).total_seconds()
            <= _SOURCE_MAX_AGE_SECONDS
            and 0
            <= (selected_at - broker_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and 0
            <= (broker_at - cash_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and ranges == {"UPRO": [6, 24], "SPXU": [3, 24]}
            and commissions
            == {
                symbol: tiered_us_stock_commission_ceiling(24)
                for symbol in _V3_SYMBOLS
            }
            and set(contract_ids) == set(_V3_SYMBOLS)
            and all(
                isinstance(contract_ids[symbol], int)
                and not isinstance(contract_ids[symbol], bool)
                and contract_ids[symbol] > 0
                for symbol in _V3_SYMBOLS
            )
            and set(preview_quantities) == set(_V3_SYMBOLS)
            and all(
                isinstance(preview_quantities[symbol], int)
                and not isinstance(preview_quantities[symbol], bool)
                and int(ranges[symbol][0])
                <= preview_quantities[symbol]
                <= int(ranges[symbol][1])
                for symbol in _V3_SYMBOLS
            )
            and nominee.get("profile_id") == "tiered_conservative_full_cross"
            and nominee.get("pricing_plan") == "Tiered"
            and nominee.get("fixed_entry_notional_usd") == 900.0
            and nominee.get("capital_identity")
            == {
                "starting_cash_identity_usd": 900.0,
                "fixed_entry_notional_usd": 900.0,
                "cash_slots": 1,
                "maximum_gross_purchase_notional_usd": 900.0,
                "settlement": "strict_T_plus_1_settled_cash_only",
                "unsettled_sale_proceeds_reused": False,
            }
            and broker.get("settled_cash_usd") >= minimum_cash
            and broker.get("minimum_settled_cash_usd") == minimum_cash
            and source_context.get("schema")
            == XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA
            and source_context_as_of < source_context_day
            and source_context_day == xsp_trading_date(source_at)
            and _sha256_identity(source_context.get("state_fingerprint"))
            and baseline_valid
        )
    except (KeyError, TypeError, ValueError):
        semantic_valid = False
    if (
        selection.get("schema") != XSP_V3_TRANSPORT_SELECTION_SCHEMA
        or selection.get("strategy_version") != XSP_OPENING_EDGE_V3_VERSION
        or selection.get("source_strategy_version")
        != XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        or selection.get("authority") != "selected_live_cash_transport"
        or selection.get("order_authority") != XSP_V2_TRANSPORT_ORDER_AUTHORITY
        or selection.get("profitability_clock_started") is not True
        or selection.get("execution_session") != "RTH"
        or selection.get("direction_symbols") != _V3_DIRECTION_SYMBOL
        or selection.get("risk")
        != {
            "starting_cash_identity_usd": 900.0,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        }
        or selection.get("execution") != _v3_execution_contract()
        or not isinstance(nominee, Mapping)
        or not isinstance(broker, Mapping)
        or not str(broker.get("account_id") or "").strip()
        or broker.get("account_type") != "CASH"
        or broker.get("positions") != {"UPRO": 0, "SPXU": 0}
        or broker.get("open_orders") != []
        or not isinstance(broker.get("unrelated_positions"), list)
        or any(
            not isinstance(row, Mapping)
            or str(row.get("symbol") or "") in _V3_SYMBOLS
            for row in broker["unrelated_positions"]
        )
        or not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "cash_receipt",
            "preview",
            "source_checkpoint_id",
            "source_recorded_at_utc",
            "source_daily_context",
            "rth_scope",
        }
        or not isinstance(evidence.get("cash_receipt"), Mapping)
        or evidence["cash_receipt"].get("sha256") != _CASH_RECEIPT_SHA256
        or not isinstance(evidence.get("preview"), Mapping)
        or not _sha256_identity(evidence["preview"].get("sha256"))
        or not _sha256_identity(evidence.get("source_checkpoint_id"))
        or not str(evidence.get("source_recorded_at_utc") or "")
        or not isinstance(source_context, Mapping)
        or evidence.get("rth_scope")
        != {
            "accepted": True,
            "historical_trades": 423,
            "trades_per_year": 141.0,
            "gth_execution_allowed": False,
        }
        or not semantic_valid
        or selection.get("selection_id") != calibration_fingerprint(body)
    ):
        raise ValueError("invalid selected XSP v3 cash transport")
    return dict(selection)


def _load_xsp_v3_rotation_selection_from_mapping(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate an immediate-proceeds continuity or clean-reset selection."""

    nominee = selection.get("nominee")
    broker = selection.get("broker_at_selection")
    evidence = selection.get("evidence")
    continuity = selection.get("continuity")
    reset = selection.get("reset")
    baseline = selection.get("baseline_state")
    source_context = (
        evidence.get("source_daily_context")
        if isinstance(evidence, Mapping)
        else None
    )
    continuity_valid = bool(
        isinstance(continuity, Mapping)
        and set(continuity)
        == {
            "schema",
            "predecessor_schema",
            "predecessor_selection_id",
            "predecessor_run_started_at_utc",
            "predecessor_starting_cash_usd",
            "inherited_holding_direction",
            "source_target_state",
            "inherited_fills",
            "inherited_fill_ledger_fingerprint",
        }
        and continuity.get("schema") == _CONTINUITY_SCHEMA
        and continuity.get("predecessor_schema")
        == XSP_V3_TRANSPORT_SELECTION_SCHEMA
        and _sha256_identity(continuity.get("predecessor_selection_id"))
        and bool(str(continuity.get("predecessor_run_started_at_utc") or ""))
    )
    reset_valid = bool(
        isinstance(reset, Mapping)
        and set(reset)
        == {
            "schema",
            "predecessor_schema",
            "predecessor_selection_id",
            "predecessor_run_started_at_utc",
            "predecessor_fill_ledger_fingerprint",
            "predecessor_risk_state_fingerprint",
            "predecessor_realized_net_usd",
            "predecessor_closed_trades",
            "source_target_state",
        }
        and reset.get("schema") == _RESET_SCHEMA
        and reset.get("predecessor_schema") == XSP_V3_ROTATION_SELECTION_SCHEMA
        and _sha256_identity(reset.get("predecessor_selection_id"))
        and bool(str(reset.get("predecessor_run_started_at_utc") or ""))
        and _sha256_identity(reset.get("predecessor_fill_ledger_fingerprint"))
        and _sha256_identity(reset.get("predecessor_risk_state_fingerprint"))
        and reset.get("source_target_state") is None
    )
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    try:
        selected_at = _utc(selection["selected_at_utc"])
        source_at = _utc(evidence["source_recorded_at_utc"])
        broker_at = _utc(broker["observed_at_utc"])
        cash_at = _utc(broker["cash_observed_at_utc"])
        ranges = nominee["historical_quantity_ranges"]
        commissions = nominee["commission_limits_usd"]
        contract_ids = nominee["contract_ids"]
        preview_quantities = nominee["preview_quantities"]
        broker_positions = {
            symbol: int(
                _number(broker["positions"][symbol], name=f"{symbol} position")
            )
            for symbol in _V3_SYMBOLS
        }
        if continuity_valid:
            fills = continuity["inherited_fills"]
            replay_cash, replay_holdings = _inherited_cash_and_holdings(
                fills,
                starting_cash_usd=float(
                    continuity["predecessor_starting_cash_usd"]
                ),
            )
            held_symbol = next(
                symbol
                for symbol, quantity in broker_positions.items()
                if quantity > 0
            )
            held_direction = next(
                direction
                for direction, symbol in _V3_DIRECTION_SYMBOL.items()
                if symbol == held_symbol
            )
            source_target = continuity["source_target_state"]
            lifecycle_valid = bool(
                replay_holdings == broker_positions
                and abs(replay_cash - float(broker["settled_cash_usd"])) <= 0.02
                and sum(
                    quantity > 0 for quantity in broker_positions.values()
                )
                == 1
                and continuity["inherited_holding_direction"] == held_direction
                and baseline == source_target
                and (
                    source_target is None
                    or (
                        isinstance(source_target, Mapping)
                        and source_target["lane"] == "rth"
                        and str(source_target["direction"])
                        in _V3_DIRECTION_SYMBOL
                    )
                )
                and continuity["inherited_fill_ledger_fingerprint"]
                == calibration_fingerprint(fills)
            )
        else:
            source_target = reset["source_target_state"]
            lifecycle_valid = bool(
                reset_valid
                and broker_positions == {"UPRO": 0, "SPXU": 0}
                and baseline is None
                and source_target is None
                and broker["settled_cash_usd"]
                >= 900.0 + max(float(value) for value in commissions.values())
                and broker["minimum_settled_cash_usd"]
                == 900.0 + max(float(value) for value in commissions.values())
                and isinstance(reset["predecessor_closed_trades"], int)
                and not isinstance(reset["predecessor_closed_trades"], bool)
                and reset["predecessor_closed_trades"] >= 0
                and _number(
                    reset["predecessor_realized_net_usd"],
                    name="predecessor realized net",
                )
                == float(reset["predecessor_realized_net_usd"])
                and (
                    reset["predecessor_closed_trades"] > 0
                    or abs(float(reset["predecessor_realized_net_usd"]))
                    <= 1e-9
                )
            )
        context_day = date.fromisoformat(str(source_context["trading_day"]))
        context_as_of = date.fromisoformat(
            str(source_context["context_as_of_day"])
        )
        semantic_valid = bool(
            selection["run_started_at_utc"] == selected_at.isoformat()
            and 0
            <= (selected_at - source_at).total_seconds()
            <= _SOURCE_MAX_AGE_SECONDS
            and 0
            <= (selected_at - broker_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and 0
            <= (broker_at - cash_at).total_seconds()
            <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
            and ranges == {"UPRO": [6, 24], "SPXU": [3, 24]}
            and commissions
            == {
                symbol: tiered_us_stock_commission_ceiling(24)
                for symbol in _V3_SYMBOLS
            }
            and set(contract_ids) == set(_V3_SYMBOLS)
            and all(
                isinstance(contract_ids[symbol], int)
                and not isinstance(contract_ids[symbol], bool)
                and contract_ids[symbol] > 0
                for symbol in _V3_SYMBOLS
            )
            and set(preview_quantities) == set(_V3_SYMBOLS)
            and all(
                isinstance(preview_quantities[symbol], int)
                and not isinstance(preview_quantities[symbol], bool)
                and int(ranges[symbol][0])
                <= preview_quantities[symbol]
                <= int(ranges[symbol][1])
                for symbol in _V3_SYMBOLS
            )
            and nominee["profile_id"] == "tiered_conservative_full_cross"
            and nominee["pricing_plan"] == "Tiered"
            and nominee["fixed_entry_notional_usd"] == 900.0
            and nominee["capital_identity"]
            == {
                "starting_cash_identity_usd": 900.0,
                "fixed_entry_notional_usd": 900.0,
                "cash_slots": 1,
                "maximum_gross_purchase_notional_usd": 900.0,
                "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
                "unsettled_sale_proceeds_reused": True,
            }
            and lifecycle_valid
            and source_context["schema"]
            == XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA
            and context_as_of < context_day
            and context_day == xsp_trading_date(source_at)
            and _sha256_identity(source_context["state_fingerprint"])
        )
    except (KeyError, StopIteration, TypeError, ValueError):
        semantic_valid = False
    immediate = (
        evidence.get("immediate_proceeds")
        if isinstance(evidence, Mapping)
        else None
    )
    if (
        selection.get("schema") != XSP_V3_ROTATION_SELECTION_SCHEMA
        or selection.get("strategy_version") != XSP_OPENING_EDGE_V3_VERSION
        or selection.get("source_strategy_version")
        != XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        or selection.get("authority") != "selected_live_cash_transport"
        or selection.get("order_authority") != XSP_V2_TRANSPORT_ORDER_AUTHORITY
        or selection.get("profitability_clock_started") is not True
        or selection.get("execution_session") != "RTH"
        or selection.get("direction_symbols") != _V3_DIRECTION_SYMBOL
        or selection.get("risk")
        != {
            "starting_cash_identity_usd": 900.0,
            "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        }
        or selection.get("execution") != _v3_execution_contract()
        or not isinstance(nominee, Mapping)
        or not isinstance(broker, Mapping)
        or not str(broker.get("account_id") or "")
        or broker.get("account_type") != "CASH"
        or broker.get("open_orders") != []
        or not isinstance(broker.get("unrelated_positions"), list)
        or any(
            not isinstance(row, Mapping)
            or str(row.get("symbol") or "") in _V3_SYMBOLS
            for row in broker["unrelated_positions"]
        )
        or continuity_valid == reset_valid
        or not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "cash_receipt",
            "preview",
            "source_checkpoint_id",
            "source_recorded_at_utc",
            "source_daily_context",
            "rth_scope",
            "immediate_proceeds",
        }
        or not _sha256_identity(evidence.get("source_checkpoint_id"))
        or evidence.get("rth_scope")
        != {
            "accepted": True,
            "historical_trades": 423,
            "trades_per_year": 141.0,
            "gth_execution_allowed": False,
        }
        or not isinstance(immediate, Mapping)
        or immediate.get("sha256") != _IMMEDIATE_PROCEEDS_SHA256
        or immediate.get("official_account_contract_url")
        != "https://www.interactivebrokers.com.au/en/accounts/configuring-your-account.php"
        or not _sha256_identity(immediate.get("broker_preview_sha256"))
        or not semantic_valid
        or selection.get("selection_id") != calibration_fingerprint(body)
    ):
        raise ValueError("invalid immediate-proceeds XSP v3 cash transport")
    return dict(selection)


def load_xsp_v3_transport_selection_from_mapping(
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Validate either immutable v3 cash-selection generation."""

    if selection.get("schema") == XSP_V3_TRANSPORT_SELECTION_SCHEMA:
        return _load_xsp_v3_initial_selection_from_mapping(selection)
    if selection.get("schema") == XSP_V3_ROTATION_SELECTION_SCHEMA:
        return _load_xsp_v3_rotation_selection_from_mapping(selection)
    if selection.get("schema") == XSP_V3_PACKAGE_SELECTION_SCHEMA:
        from .xsp_live_transport_allocation import (
            load_xsp_v3_package_selection_from_mapping,
        )

        return load_xsp_v3_package_selection_from_mapping(selection)
    raise ValueError("unsupported selected XSP v3 cash transport")
