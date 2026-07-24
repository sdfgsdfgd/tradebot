"""Quote snapshot schema + helpers.

These snapshots are meant to be:
- human-readable (JSONL),
- append-only,
- reusable for both backtesting validation and live bot monitoring.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ib_insync import IB, ContFuture, Index, Stock

from ..contract_identity import (
    future_exchange_for_symbol,
    index_exchange_for_symbol,
    select_option_chain,
)


def _none_if_nan(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _nonnegative_or_none(value) -> float | None:
    number = _none_if_nan(value)
    return number if number is not None and number >= 0 else None


def resolve_option_chain(
    ib: IB,
    symbol: str,
    exchange: str | None,
) -> tuple[object, float | None, object | None, bool]:
    """Resolve one qualified underlyer and its most complete option chain."""
    symbol = str(symbol).strip().upper()
    future_exchange = future_exchange_for_symbol(symbol)
    index_exchange = index_exchange_for_symbol(symbol)
    if future_exchange is not None:
        underlying = ContFuture(symbol, exchange or future_exchange, "USD")
        is_future = True
    elif index_exchange is not None:
        underlying = Index(symbol, exchange or index_exchange, "USD")
        is_future = False
    else:
        underlying = Stock(symbol, exchange or "SMART", "USD")
        is_future = False

    qualified = ib.qualifyContracts(underlying)
    underlying = qualified[0] if qualified else underlying
    ticker = ib.reqTickers(underlying)[0]
    spot = None
    for value in (ticker.marketPrice(), ticker.last, ticker.close):
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0:
            spot = candidate
            break
    chains = ib.reqSecDefOptParams(
        underlying.symbol,
        str(getattr(underlying, "exchange", "") or "") if is_future else "",
        "FUT" if is_future else underlying.secType,
        underlying.conId,
    )
    return (
        underlying,
        spot,
        select_option_chain(chains, symbol, prefer_smart=not is_future),
        is_future,
    )


@dataclass(frozen=True)
class QuoteContract:
    con_id: int | None
    sec_type: str | None
    symbol: str | None
    local_symbol: str | None
    exchange: str | None
    currency: str | None

    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    trading_class: str | None = None
    multiplier: str | None = None

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    volume: float | None = None

    model_iv: float | None = None
    model_delta: float | None = None
    model_gamma: float | None = None
    model_vega: float | None = None
    model_theta: float | None = None
    model_under_price: float | None = None

    market_data_type: int | None = None
    quote_time: str | None = None


@dataclass(frozen=True)
class QuoteError:
    req_id: int | None
    code: int
    message: str
    con_id: int | None = None
    symbol: str | None = None
    local_symbol: str | None = None
    sec_type: str | None = None
    exchange: str | None = None


@dataclass(frozen=True)
class QuoteSnapshot:
    ts: str
    md_type: int
    symbol: str
    underlying: QuoteContract
    options: list[QuoteContract]
    errors: list[QuoteError]


def contract_from_ticker(contract, ticker) -> QuoteContract:
    model = getattr(ticker, "modelGreeks", None)
    return QuoteContract(
        con_id=_safe_int(getattr(contract, "conId", None)),
        sec_type=getattr(contract, "secType", None),
        symbol=getattr(contract, "symbol", None),
        local_symbol=getattr(contract, "localSymbol", None),
        exchange=getattr(contract, "exchange", None),
        currency=getattr(contract, "currency", None),
        expiry=getattr(contract, "lastTradeDateOrContractMonth", None),
        strike=_none_if_nan(getattr(contract, "strike", None)),
        right=getattr(contract, "right", None),
        trading_class=getattr(contract, "tradingClass", None),
        multiplier=getattr(contract, "multiplier", None),
        bid=_nonnegative_or_none(getattr(ticker, "bid", None)),
        ask=_nonnegative_or_none(getattr(ticker, "ask", None)),
        last=_nonnegative_or_none(getattr(ticker, "last", None)),
        close=_nonnegative_or_none(getattr(ticker, "close", None)),
        bid_size=_nonnegative_or_none(getattr(ticker, "bidSize", None)),
        ask_size=_nonnegative_or_none(getattr(ticker, "askSize", None)),
        last_size=_nonnegative_or_none(getattr(ticker, "lastSize", None)),
        volume=_nonnegative_or_none(getattr(ticker, "volume", None)),
        model_iv=_none_if_nan(getattr(model, "impliedVol", None)),
        model_delta=_none_if_nan(getattr(model, "delta", None)),
        model_gamma=_none_if_nan(getattr(model, "gamma", None)),
        model_vega=_none_if_nan(getattr(model, "vega", None)),
        model_theta=_none_if_nan(getattr(model, "theta", None)),
        model_under_price=_none_if_nan(getattr(model, "undPrice", None)),
        market_data_type=_safe_int(getattr(ticker, "marketDataType", None)),
        quote_time=_iso_or_none(getattr(ticker, "time", None)),
    )


def make_snapshot(
    symbol: str,
    md_type: int,
    underlying_contract,
    underlying_ticker,
    option_contracts: list[Any],
    option_tickers: list[Any],
    errors: list[QuoteError] | None = None,
    ts: datetime | None = None,
) -> QuoteSnapshot:
    now = ts or datetime.now(timezone.utc)
    snap_errors = list(errors or [])
    underlying = contract_from_ticker(underlying_contract, underlying_ticker)
    options: list[QuoteContract] = []
    for contract, ticker in zip(option_contracts, option_tickers):
        options.append(contract_from_ticker(contract, ticker))
    return QuoteSnapshot(
        ts=now.isoformat(),
        md_type=int(md_type),
        symbol=symbol,
        underlying=underlying,
        options=options,
        errors=snap_errors,
    )


def append_snapshot(path: Path, snapshot: QuoteSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(snapshot)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso_or_none(value) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else None
