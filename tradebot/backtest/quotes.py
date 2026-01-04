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
        bid=_none_if_nan(getattr(ticker, "bid", None)),
        ask=_none_if_nan(getattr(ticker, "ask", None)),
        last=_none_if_nan(getattr(ticker, "last", None)),
        close=_none_if_nan(getattr(ticker, "close", None)),
        bid_size=_none_if_nan(getattr(ticker, "bidSize", None)),
        ask_size=_none_if_nan(getattr(ticker, "askSize", None)),
        last_size=_none_if_nan(getattr(ticker, "lastSize", None)),
        volume=_none_if_nan(getattr(ticker, "volume", None)),
        model_iv=_none_if_nan(getattr(model, "impliedVol", None)),
        model_delta=_none_if_nan(getattr(model, "delta", None)),
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

