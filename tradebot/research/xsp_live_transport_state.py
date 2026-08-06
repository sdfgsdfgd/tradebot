"""Fresh observer and broker state for selected XSP cash transports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from .xsp_opening_edge_v2 import XSP_OPENING_EDGE_V2_TRANSPORT_VERSION
from .xsp_dual_clock import XSP_DUAL_CLOCK_SOURCE_VERSION
from .xsp_opening_edge_v3 import XSP_OPENING_EDGE_V3_TRANSPORT_VERSION


_BROKER_SNAPSHOT_MAX_AGE_SECONDS = 90.0
_V2_SYMBOLS = ("SPYU", "SPXU")


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
        raise ValueError("transport timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _source_receipt_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    transport_version: str,
) -> dict[str, object]:
    evidence = checkpoint.get("evidence")
    if (
        checkpoint.get("kind") != "checkpoint"
        or checkpoint.get("strategy_version") != transport_version
        or not isinstance(checkpoint.get("checkpoint_id"), str)
        or not isinstance(evidence, Mapping)
        or not isinstance(evidence.get("paired_equity"), Mapping)
    ):
        raise ValueError("invalid Opening Edge source checkpoint")
    status = str(checkpoint.get("status") or "")
    return {
        "evaluation_status": status,
        "freshness_ok": (
            status == "EVALUATED" and evidence.get("rth_provenance_fresh") is True
        ),
        "session": str(checkpoint.get("session") or ""),
        "order_authority": evidence.get("order_authority"),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "recorded_at_utc": checkpoint.get("recorded_at_utc"),
        "fundamental_pressure": evidence.get("fundamental_pressure"),
        "paired_equity": evidence["paired_equity"],
    }


def xsp_v2_source_receipt_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    return _source_receipt_from_checkpoint(
        checkpoint,
        transport_version=XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
    )


def xsp_v3_source_receipt_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    return _source_receipt_from_checkpoint(
        checkpoint,
        transport_version=XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
    )


def xsp_p009_source_receipt_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    return _source_receipt_from_checkpoint(
        checkpoint,
        transport_version=XSP_DUAL_CLOCK_SOURCE_VERSION,
    )


def latest_xsp_v2_source_receipt(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for record in reversed(records):
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == XSP_OPENING_EDGE_V2_TRANSPORT_VERSION
        ):
            return xsp_v2_source_receipt_from_checkpoint(record)
    raise ValueError("Opening Edge v2 has no source checkpoint")


def latest_xsp_v3_source_receipt(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for record in reversed(records):
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        ):
            return xsp_v3_source_receipt_from_checkpoint(record)
    raise ValueError("Opening Edge v3 has no source checkpoint")


def latest_xsp_p009_source_receipt(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for record in reversed(records):
        if (
            record.get("kind") == "checkpoint"
            and record.get("strategy_version") == XSP_DUAL_CLOCK_SOURCE_VERSION
        ):
            return xsp_p009_source_receipt_from_checkpoint(record)
    raise ValueError("Opening Edge P-009 has no source checkpoint")


async def xsp_v2_broker_snapshot(
    client,
    *,
    symbols: Sequence[str] = _V2_SYMBOLS,
    resource_base_currency: str | None = None,
) -> dict[str, object]:
    """Capture the one broker account state used by selection and execution."""

    portfolio = await client.fetch_portfolio()
    account_id = str(client.account_id() or "").strip()
    account_type = str(client.account_text_value("TradingType-S") or "").upper()
    cash_value, cash_currency, cash_at_raw = client.account_value(
        "CashBalance",
        currency="USD",
    )
    cash = _number(cash_value, name="broker USD cash balance")
    observed_at = datetime.now(timezone.utc)
    cash_at = _utc(cash_at_raw)
    if (
        not account_id
        or account_type != "STKCASH"
        or str(cash_currency or "").upper() != "USD"
        or cash < 0
        or observed_at < cash_at
        or (observed_at - cash_at).total_seconds()
        > _BROKER_SNAPSHOT_MAX_AGE_SECONDS
    ):
        raise ValueError("fresh broker cash-account state is unavailable")

    normalized_symbols = tuple(str(symbol).upper() for symbol in symbols)
    if (
        len(normalized_symbols) != 2
        or len(set(normalized_symbols)) != 2
        or any(not symbol for symbol in normalized_symbols)
    ):
        raise ValueError("cash-pair symbols are invalid")
    base = str(resource_base_currency or "").upper()
    rates: dict[str, float] = {base: 1.0} if base else {}

    def base_rate(currency: str) -> float:
        normalized = str(currency or "").upper()
        if not base or not normalized:
            raise ValueError("portfolio position currency is unavailable")
        if normalized not in rates:
            value, actual, _updated = client.account_value(
                "ExchangeRate", currency=normalized
            )
            if str(actual or "").upper() != normalized:
                raise ValueError(f"fresh {normalized}/{base} exchange rate is unavailable")
            rates[normalized] = _number(
                value, name=f"{normalized}/{base} exchange rate"
            )
        return rates[normalized]

    positions = {symbol: 0.0 for symbol in normalized_symbols}
    account_positions = []
    unrelated_positions = []
    for item in portfolio:
        contract = getattr(item, "contract", None)
        symbol = str(getattr(contract, "symbol", "") or "").upper()
        quantity = _number(
            getattr(item, "position", 0.0) or 0.0,
            name="broker portfolio quantity",
        )
        row = {
            "symbol": symbol,
            "con_id": int(getattr(contract, "conId", 0) or 0),
            "sec_type": str(getattr(contract, "secType", "") or ""),
            "quantity": quantity,
        }
        if resource_base_currency is not None:
            currency = str(getattr(contract, "currency", "") or "").upper()
            row["currency"] = currency
            row["market_value_base_cents"] = math.ceil(
                abs(
                    _number(
                        getattr(item, "marketValue", 0.0) or 0.0,
                        name="broker portfolio market value",
                    )
                )
                * base_rate(currency)
                * 100
            )
            account_positions.append(dict(row))
        if symbol in positions:
            positions[symbol] += quantity
        elif abs(quantity) > 1e-9:
            unrelated_positions.append(row)

    open_orders = []
    for trade in client.open_trades():
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        open_orders.append(
            {
                "symbol": str(getattr(contract, "symbol", "") or "").upper(),
                "con_id": int(getattr(contract, "conId", 0) or 0),
                "action": str(getattr(order, "action", "") or "").upper(),
                "quantity": _number(
                    getattr(order, "totalQuantity", 0.0) or 0.0,
                    name="broker open-order quantity",
                ),
                "order_ref": str(getattr(order, "orderRef", "") or ""),
                "status": str(getattr(status, "status", "") or ""),
            }
        )
    snapshot = {
        "observed_at_utc": observed_at.isoformat(),
        "cash_observed_at_utc": cash_at.isoformat(),
        "account_id": account_id,
        "account_type": "CASH",
        "settled_cash_usd": cash,
        "positions": positions,
        "unrelated_positions": unrelated_positions,
        "open_orders": open_orders,
    }
    if resource_base_currency is not None:
        if not base:
            raise ValueError("capital resource base currency is empty")

        def account_cents(tag: str) -> int:
            value, actual, _updated = client.account_value(tag, currency=base)
            if str(actual or "").upper() != base:
                raise ValueError(f"fresh {tag} {base} is unavailable")
            return math.floor(_number(value, name=f"{tag} {base}") * 100)

        rate = base_rate("USD")
        snapshot["account_positions"] = account_positions
        snapshot["account_resources"] = {
            "base_currency": base,
            "available_funds_base_cents": account_cents("AvailableFunds"),
            "excess_liquidity_base_cents": account_cents("ExcessLiquidity"),
            "usd_to_base_rate_ppm": math.ceil(
                rate * 1_000_000
            ),
        }
    return snapshot
