"""Immutable minimum-package successor for the selected XSP cash transport."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ib_insync import Stock

from ..backtest.quotes import contract_from_ticker
from ..live.order_evidence import tiered_us_stock_commission_ceiling
from .live_calibration import calibration_fingerprint
from .xsp_live_transport import (
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
    XSP_V3_PACKAGE_SELECTION_SCHEMA,
    XSP_V3_ROTATION_SELECTION_SCHEMA,
    XSP_V3_TRANSPORT_EXECUTION_VERSION,
    XSP_V3_TRANSPORT_SELECTION_SCHEMAS,
    _BROKER_SNAPSHOT_MAX_AGE_SECONDS,
    _V3_DIRECTION_SYMBOL,
    _V3_SYMBOLS,
    _number,
    _sha256,
    _utc,
    _v3_execution_contract,
)
from .xsp_live_transport_risk import xsp_transport_risk_state
from .xsp_live_transport_state import xsp_v2_broker_snapshot
from .xsp_live_transport_v3 import (
    _IMMEDIATE_PROCEEDS_SHA256,
    _sha256_identity,
    load_xsp_v3_transport_selection_from_mapping,
)
from .xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
    XSP_OPENING_EDGE_V3_VERSION,
)


XSP_PORTFOLIO_PACKAGE_RECEIPT_SCHEMA = (
    "xsp.opening-edge-v3-portfolio-package-curve.v1"
)
XSP_PORTFOLIO_PACKAGE_PREVIEW_SCHEMA = (
    "xsp.opening-edge-v3-portfolio-package-preview.v1"
)
XSP_PORTFOLIO_PACKAGE_SUCCESSOR_SCHEMA = (
    "xsp.opening-edge-v3-portfolio-package-successor.v1"
)
XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH = Path(
    "backtests/xsp/opening_edge_v3_portfolio_package_curve_20260803.json"
)
XSP_PORTFOLIO_SIGNAL_LEDGER_SHA256 = (
    "40708a28476fad504d29b1222e52b88481b1c0da8313ed4667076e2b6ed3f157"
)


def _package_cell(
    receipt: Mapping[str, object], package_id: str
) -> dict[str, object]:
    try:
        notional = int(package_id.removeprefix("xsp-usd-"))
    except ValueError as exc:
        raise ValueError("XSP package identity is invalid") from exc
    cells = receipt.get("package_cells")
    matches = [
        dict(cell)
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("notional_usd") == notional
    ] if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)) else []
    proof = receipt.get("proof")
    if (
        receipt.get("schema") != XSP_PORTFOLIO_PACKAGE_RECEIPT_SCHEMA
        or receipt.get("authority")
        != "historical_cash_package_and_open_position_resource_qualification_only"
        or receipt.get("frozen_signal_ledger_sha256")
        != XSP_PORTFOLIO_SIGNAL_LEDGER_SHA256
        or receipt.get("submitted_orders") != 0
        or not isinstance(proof, Mapping)
        or proof.get("blocked_entries_per_cell") != 0
        or proof.get("executed_trades_per_cell") != 426
        or proof.get("every_annual_slice_positive") is not True
        or len(matches) != 1
    ):
        raise ValueError("XSP package receipt is invalid")
    cell = matches[0]
    ranges = cell.get("quantity_ranges")
    annual = cell.get("annual_net_usd")
    if (
        not isinstance(ranges, Mapping)
        or set(ranges) != set(_V3_SYMBOLS)
        or not isinstance(annual, Sequence)
        or isinstance(annual, (str, bytes))
        or len(annual) != 3
        or any(_number(value, name="annual package net") <= 0 for value in annual)
        or _number(cell.get("net_usd"), name="package net") <= 0
        or _number(cell.get("profit_factor"), name="package profit factor") <= 1
        or _number(cell.get("intrabar_drawdown_usd"), name="package drawdown")
        >= notional * 0.15
    ):
        raise ValueError("XSP package did not pass its frozen economics")
    return cell


def xsp_package_cash_debit_usd_cents(notional_usd: int) -> int:
    """Reserve fixed purchase notional plus the crown's global fee ceiling."""

    commission_cents = math.ceil(tiered_us_stock_commission_ceiling(24) * 100)
    return notional_usd * 100 + commission_cents


async def xsp_portfolio_package_preview(
    client,
    *,
    notional_usd: int,
    observed_at: datetime,
    quote_wait_seconds: float = 3.0,
    preview_outside_rth: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Capture one live-book, zero-transmission preview for an XSP package."""

    now = _utc(observed_at)
    if notional_usd <= 0 or quote_wait_seconds < 0:
        raise ValueError("XSP package preview inputs are invalid")
    broker = await xsp_v2_broker_snapshot(
        client,
        symbols=_V3_SYMBOLS,
        resource_base_currency="AUD",
    )
    relevant_positions = [
        {"symbol": symbol, "quantity": quantity}
        for symbol, quantity in broker["positions"].items()
        if abs(float(quantity)) > 1e-9
    ]
    relevant_orders = [
        dict(row)
        for row in broker["open_orders"]
        if str(row.get("symbol") or "").upper() in _V3_SYMBOLS
    ]
    qualified = await client.qualify_proxy_contracts(
        *(Stock(symbol, "SMART", "USD") for symbol in _V3_SYMBOLS)
    )
    contracts = {
        str(getattr(contract, "symbol", "") or "").upper(): contract
        for contract in qualified
        if str(getattr(contract, "symbol", "") or "").upper() in _V3_SYMBOLS
    }
    if set(contracts) != set(_V3_SYMBOLS):
        raise ValueError("XSP package contracts are unavailable")
    tickers = {
        symbol: await client.ensure_ticker(
            contracts[symbol], owner="xsp-portfolio-package-preview"
        )
        for symbol in _V3_SYMBOLS
    }
    deadline = time.monotonic() + quote_wait_seconds
    books: dict[str, object] = {}
    while True:
        books = {
            symbol: contract_from_ticker(contracts[symbol], tickers[symbol])
            for symbol in _V3_SYMBOLS
        }
        if all(
            row.market_data_type == 1
            and row.bid is not None
            and row.ask is not None
            and row.bid > 0
            and row.ask >= row.bid
            for row in books.values()
        ) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)
    before = len(client.open_trades())
    rows = []
    for symbol in _V3_SYMBOLS:
        contract = contracts[symbol]
        book = books[symbol]
        if (
            book.market_data_type != 1
            or book.bid is None
            or book.ask is None
            or book.bid <= 0
            or book.ask < book.bid
        ):
            raise ValueError(f"{symbol}: live XSP package book is unavailable")
        quantity = math.floor(notional_usd / book.ask)
        preview = await client.preview_limit_order(
            contract,
            "BUY",
            quantity,
            book.ask,
            preview_outside_rth,
            f"XSPPKG-{notional_usd}-{symbol}",
        )
        rows.append(
            {
                "symbol": symbol,
                "contract": {
                    "con_id": int(getattr(contract, "conId", 0) or 0),
                    "exchange": "SMART",
                    "primary_exchange": str(
                        getattr(contract, "primaryExchange", "") or ""
                    ),
                    "currency": str(getattr(contract, "currency", "") or ""),
                },
                "book": {
                    "bid": book.bid,
                    "ask": book.ask,
                    "bid_size": book.bid_size,
                    "ask_size": book.ask_size,
                    "market_data_type": book.market_data_type,
                },
                "order": {
                    "action": "BUY",
                    "quantity": quantity,
                    "limit_price": book.ask,
                    "what_if": True,
                    "transmit": False,
                    "outside_rth": preview_outside_rth,
                },
                "what_if": asdict(preview),
            }
        )
    after = len(client.open_trades())
    return {
        "schema": XSP_PORTFOLIO_PACKAGE_PREVIEW_SCHEMA,
        "authority": "fresh_nontransmitting_what_if_only",
        "observed_at_utc": now.isoformat(),
        "notional_usd": float(notional_usd),
        "rows": rows,
        "relevant_positions": relevant_positions,
        "relevant_open_orders": relevant_orders,
        "open_trades_before": before,
        "open_trades_after": after,
        "order_authority": "none",
        "submitted_orders": 0,
    }, broker


def _pending_order_refs(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> list[str]:
    latest: dict[str, str] = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("strategy_version") != XSP_V3_TRANSPORT_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selection_id
        ):
            continue
        order_ref = str(evidence.get("order_ref") or "")
        phase = str(evidence.get("phase") or "")
        if order_ref and phase in {"PREPARED", "SUBMITTED", "TERMINAL"}:
            latest[order_ref] = phase
    return sorted(ref for ref, phase in latest.items() if phase != "TERMINAL")


def _preview_nominee(
    preview: Mapping[str, object],
    *,
    selected_at: datetime,
    cell: Mapping[str, object],
) -> dict[str, object]:
    notional = int(cell["notional_usd"])
    ranges = cell["quantity_ranges"]
    rows = preview.get("rows")
    observed_at = _utc(preview.get("observed_at_utc"))
    keyed = {
        str(row.get("symbol") or "").upper(): row
        for row in rows
        if isinstance(row, Mapping)
    } if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else {}
    if (
        preview.get("schema") != XSP_PORTFOLIO_PACKAGE_PREVIEW_SCHEMA
        or preview.get("authority") != "fresh_nontransmitting_what_if_only"
        or preview.get("notional_usd") != float(notional)
        or preview.get("order_authority") != "none"
        or preview.get("submitted_orders") != 0
        or preview.get("relevant_positions") != []
        or preview.get("relevant_open_orders") != []
        or preview.get("open_trades_before") != preview.get("open_trades_after")
        or not 0
        <= (selected_at - observed_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or set(keyed) != set(_V3_SYMBOLS)
    ):
        raise ValueError("XSP package preview is invalid")
    quantities: dict[str, int] = {}
    contracts: dict[str, int] = {}
    commissions: dict[str, float] = {}
    for symbol in _V3_SYMBOLS:
        row = keyed[symbol]
        contract = row.get("contract")
        book = row.get("book")
        order = row.get("order")
        what_if = row.get("what_if")
        bounds = ranges[symbol]
        bid = _number(book.get("bid"), name=f"{symbol} bid") if isinstance(book, Mapping) else 0
        ask = _number(book.get("ask"), name=f"{symbol} ask") if isinstance(book, Mapping) else 0
        quantity = order.get("quantity") if isinstance(order, Mapping) else None
        limit = tiered_us_stock_commission_ceiling(int(bounds[1]))
        observed_commission = max(
            (
                _number(value, name=f"{symbol} commission")
                for key in ("commission", "min_commission", "max_commission")
                if isinstance(what_if, Mapping)
                and (value := what_if.get(key)) is not None
            ),
            default=0,
        )
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 2
            or not isinstance(contract, Mapping)
            or contract.get("exchange") != "SMART"
            or contract.get("primary_exchange") != "ARCA"
            or contract.get("currency") != "USD"
            or int(contract.get("con_id") or 0) <= 0
            or not isinstance(book, Mapping)
            or book.get("market_data_type") != 1
            or bid <= 0
            or ask < bid
            or not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or quantity != math.floor(notional / ask)
            or not int(bounds[0]) <= quantity <= int(bounds[1])
            or order.get("action") != "BUY"
            or order.get("what_if") is not True
            or order.get("transmit") is not False
            or not isinstance(what_if, Mapping)
            or what_if.get("status") != "PreSubmitted"
            or what_if.get("commission_currency") != "USD"
            or observed_commission <= 0
            or observed_commission > limit + 0.01
            or str(what_if.get("warning_text") or "")
        ):
            raise ValueError(f"{symbol}: XSP package preview failed")
        quantities[symbol] = quantity
        contracts[symbol] = int(contract["con_id"])
        commissions[symbol] = limit
    return {
        "profile_id": "tiered_conservative_full_cross",
        "pricing_plan": "Tiered",
        "fixed_entry_notional_usd": float(notional),
        "historical_quantity_ranges": {
            symbol: [int(value) for value in ranges[symbol]] for symbol in _V3_SYMBOLS
        },
        "preview_quantities": quantities,
        "commission_limits_usd": commissions,
        "contract_ids": contracts,
        "capital_identity": {
            "starting_cash_identity_usd": float(notional),
            "fixed_entry_notional_usd": float(notional),
            "cash_slots": 1,
            "maximum_gross_purchase_notional_usd": float(notional),
            "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
            "unsettled_sale_proceeds_reused": True,
        },
    }


def reallocate_xsp_v3_transport(
    *,
    predecessor: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    broker_snapshot: Mapping[str, object],
    preview: Mapping[str, object],
    package_receipt_path: Path,
    package_id: str,
    selected_at: datetime,
) -> dict[str, object]:
    """Freeze a clean selected run at one historically qualified cash package."""

    prior = load_xsp_v3_transport_selection_from_mapping(predecessor)
    if prior["schema"] != XSP_V3_ROTATION_SELECTION_SCHEMA:
        raise ValueError("XSP package successor requires immediate-proceeds ownership")
    now = _utc(selected_at)
    receipt = json.loads(package_receipt_path.read_text())
    if not isinstance(receipt, Mapping):
        raise ValueError("XSP package receipt must be one object")
    cell = _package_cell(receipt, package_id)
    nominee = _preview_nominee(preview, selected_at=now, cell=cell)
    risk_state = xsp_transport_risk_state(
        selection=prior,
        records=records,
        observed_at=now,
        liquidation_bids={},
    )
    broker_at = _utc(broker_snapshot.get("observed_at_utc"))
    cash_at = _utc(broker_snapshot.get("cash_observed_at_utc"))
    positions = broker_snapshot.get("positions")
    open_orders = broker_snapshot.get("open_orders")
    unrelated = broker_snapshot.get("unrelated_positions")
    minimum_cash_cents = xsp_package_cash_debit_usd_cents(int(cell["notional_usd"]))
    settled_cash = _number(
        broker_snapshot.get("settled_cash_usd"), name="settled USD cash"
    )
    if (
        _pending_order_refs(records, selection_id=str(prior["selection_id"]))
        or risk_state["holdings_from_fills"] != {"UPRO": 0.0, "SPXU": 0.0}
        or risk_state["pending_settlement_usd"] != 0
        or risk_state["safety_breaches"]
        or abs(float(risk_state["settled_cash_usd"]) - settled_cash) > 0.02
        or not 0
        <= (now - broker_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or not 0
        <= (broker_at - cash_at).total_seconds()
        <= _BROKER_SNAPSHOT_MAX_AGE_SECONDS
        or str(broker_snapshot.get("account_id") or "")
        != str(prior["broker_at_selection"]["account_id"])
        or str(broker_snapshot.get("account_type") or "").upper() != "CASH"
        or positions != {"UPRO": 0, "SPXU": 0}
        or not isinstance(open_orders, Sequence)
        or isinstance(open_orders, (str, bytes))
        or bool(open_orders)
        or not isinstance(unrelated, Sequence)
        or isinstance(unrelated, (str, bytes))
        or settled_cash * 100 + 1e-7 < minimum_cash_cents
    ):
        raise ValueError("XSP package successor requires terminal flat broker truth")
    successor = {
        "schema": XSP_PORTFOLIO_PACKAGE_SUCCESSOR_SCHEMA,
        "predecessor_schema": prior["schema"],
        "predecessor_selection_id": prior["selection_id"],
        "predecessor_run_started_at_utc": prior["run_started_at_utc"],
        "predecessor_fill_ledger_fingerprint": risk_state[
            "fill_ledger_fingerprint"
        ],
        "predecessor_risk_state_fingerprint": calibration_fingerprint(risk_state),
        "predecessor_realized_net_usd": risk_state["run_realized_net_usd"],
        "predecessor_closed_trades": risk_state["closed_trades"],
        "package_id": package_id,
        "package_cash_debit_usd_cents": minimum_cash_cents,
        "broker_preview_fingerprint": calibration_fingerprint(preview),
    }
    evidence = json.loads(json.dumps(prior["evidence"]))
    evidence["portfolio_package"] = {
        "path": XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH.as_posix(),
        "sha256": _sha256(package_receipt_path),
        "package_id": package_id,
        "signal_ledger_sha256": XSP_PORTFOLIO_SIGNAL_LEDGER_SHA256,
    }
    notional = int(cell["notional_usd"])
    body = {
        **{
            key: json.loads(json.dumps(value))
            for key, value in prior.items()
            if key
            not in {
                "selection_id",
                "schema",
                "selected_at_utc",
                "run_started_at_utc",
                "nominee",
                "baseline_state",
                "broker_at_selection",
                "risk",
                "evidence",
                "continuity",
                "reset",
            }
        },
        "schema": XSP_V3_PACKAGE_SELECTION_SCHEMA,
        "selected_at_utc": now.isoformat(),
        "run_started_at_utc": now.isoformat(),
        "nominee": nominee,
        "baseline_state": None,
        "broker_at_selection": {
            "observed_at_utc": broker_at.isoformat(),
            "cash_observed_at_utc": cash_at.isoformat(),
            "account_id": broker_snapshot["account_id"],
            "account_type": "CASH",
            "settled_cash_usd": settled_cash,
            "minimum_settled_cash_usd": minimum_cash_cents / 100,
            "positions": {"UPRO": 0, "SPXU": 0},
            "unrelated_positions": [dict(row) for row in unrelated],
            "open_orders": [],
        },
        "risk": {
            "starting_cash_identity_usd": float(notional),
            "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
            "max_drawdown_usd": float(notional) * 0.15,
            "max_session_loss_usd": float(notional) * 0.075,
            "gth_execution_allowed": False,
        },
        "evidence": evidence,
        "allocation_successor": successor,
    }
    selected = {**body, "selection_id": calibration_fingerprint(body)}
    return load_xsp_v3_package_selection_from_mapping(selected)


def load_xsp_v3_package_selection_from_mapping(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate one immutable package-sized XSP selection generation."""

    selection = dict(value)
    body = {key: item for key, item in selection.items() if key != "selection_id"}
    nominee = selection.get("nominee")
    risk = selection.get("risk")
    broker = selection.get("broker_at_selection")
    evidence = selection.get("evidence")
    successor = selection.get("allocation_successor")
    try:
        notional = int(nominee["fixed_entry_notional_usd"])
        ranges = nominee["historical_quantity_ranges"]
        commissions = nominee["commission_limits_usd"]
        package_id = f"xsp-usd-{notional}"
        minimum_cash_cents = xsp_package_cash_debit_usd_cents(notional)
        semantic = bool(
            selection["run_started_at_utc"] == selection["selected_at_utc"]
            and selection["baseline_state"] is None
            and set(ranges) == set(_V3_SYMBOLS)
            and all(
                isinstance(ranges[symbol], list)
                and len(ranges[symbol]) == 2
                and 0 < int(ranges[symbol][0]) <= int(ranges[symbol][1])
                for symbol in _V3_SYMBOLS
            )
            and commissions
            == {
                symbol: tiered_us_stock_commission_ceiling(int(ranges[symbol][1]))
                for symbol in _V3_SYMBOLS
            }
            and all(
                int(ranges[symbol][0])
                <= int(nominee["preview_quantities"][symbol])
                <= int(ranges[symbol][1])
                and int(nominee["contract_ids"][symbol]) > 0
                for symbol in _V3_SYMBOLS
            )
            and nominee["profile_id"] == "tiered_conservative_full_cross"
            and nominee["pricing_plan"] == "Tiered"
            and nominee["capital_identity"]
            == {
                "starting_cash_identity_usd": float(notional),
                "fixed_entry_notional_usd": float(notional),
                "cash_slots": 1,
                "maximum_gross_purchase_notional_usd": float(notional),
                "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
                "unsettled_sale_proceeds_reused": True,
            }
            and risk
            == {
                "starting_cash_identity_usd": float(notional),
                "settlement": XSP_V3_IMMEDIATE_PROCEEDS_SETTLEMENT,
                "max_drawdown_usd": float(notional) * 0.15,
                "max_session_loss_usd": float(notional) * 0.075,
                "gth_execution_allowed": False,
            }
            and broker["account_type"] == "CASH"
            and broker["positions"] == {"UPRO": 0, "SPXU": 0}
            and broker["open_orders"] == []
            and broker["minimum_settled_cash_usd"] == minimum_cash_cents / 100
            and broker["settled_cash_usd"] >= minimum_cash_cents / 100
            and successor["schema"] == XSP_PORTFOLIO_PACKAGE_SUCCESSOR_SCHEMA
            and successor["predecessor_schema"]
            in XSP_V3_TRANSPORT_SELECTION_SCHEMAS
            and _sha256_identity(successor["predecessor_selection_id"])
            and _sha256_identity(successor["predecessor_fill_ledger_fingerprint"])
            and _sha256_identity(successor["predecessor_risk_state_fingerprint"])
            and _sha256_identity(successor["broker_preview_fingerprint"])
            and successor["package_id"] == package_id
            and successor["package_cash_debit_usd_cents"] == minimum_cash_cents
            and evidence["portfolio_package"]
            == {
                "path": XSP_PORTFOLIO_PACKAGE_RECEIPT_PATH.as_posix(),
                "sha256": evidence["portfolio_package"]["sha256"],
                "package_id": package_id,
                "signal_ledger_sha256": XSP_PORTFOLIO_SIGNAL_LEDGER_SHA256,
            }
            and _sha256_identity(evidence["portfolio_package"]["sha256"])
            and evidence["immediate_proceeds"]["sha256"]
            == _IMMEDIATE_PROCEEDS_SHA256
            and evidence["rth_scope"]["accepted"] is True
            and evidence["rth_scope"]["gth_execution_allowed"] is False
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        semantic = False
    if (
        selection.get("schema") != XSP_V3_PACKAGE_SELECTION_SCHEMA
        or selection.get("strategy_version") != XSP_OPENING_EDGE_V3_VERSION
        or selection.get("source_strategy_version")
        != XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        or selection.get("authority") != "selected_live_cash_transport"
        or selection.get("order_authority") != XSP_V2_TRANSPORT_ORDER_AUTHORITY
        or selection.get("profitability_clock_started") is not True
        or selection.get("execution_session") != "RTH"
        or selection.get("direction_symbols") != _V3_DIRECTION_SYMBOL
        or selection.get("execution") != _v3_execution_contract()
        or not all(isinstance(row, Mapping) for row in (nominee, risk, broker, evidence, successor))
        or not semantic
        or selection.get("selection_id") != calibration_fingerprint(body)
    ):
        raise ValueError("invalid package-sized XSP v3 cash transport")
    return dict(value)
