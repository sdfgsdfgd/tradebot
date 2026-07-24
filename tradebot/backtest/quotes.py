"""Quote snapshot schema + helpers.

These snapshots are meant to be:
- human-readable (JSONL),
- append-only,
- reusable for both backtesting validation and live bot monitoring.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ib_insync import IB, ContFuture, Index, Stock

from ..contract_identity import (
    future_exchange_for_symbol,
    index_exchange_for_symbol,
    select_option_chain,
)
from ..engines.execution import (
    OptionPackageQuote,
    _midpoint,
    _tick_size,
    quote_health,
    quote_option_package,
)
from ..engines.market import xsp_session_label_et
from ..option_package import (
    OptionPackage,
    OptionPackageRisk,
    ResolvedOptionLeg,
    option_package_risk,
    option_product_facts,
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
    min_tick: float | None = None


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
class OptionChainManifest:
    """Broker-reported expiry/strike sets; qualification proves exact pairs."""

    underlying_con_id: int | None
    underlying_sec_type: str | None
    symbol: str
    underlying_exchange: str | None
    currency: str | None
    option_exchange: str | None
    trading_class: str | None
    multiplier: str | None
    expirations: tuple[str, ...]
    strikes: tuple[float, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class QuoteSnapshot:
    ts: str
    md_type: int
    symbol: str
    underlying: QuoteContract
    options: list[QuoteContract]
    errors: list[QuoteError]
    chain_fingerprint: str | None = None
    target_expiry: str | None = None
    session: str | None = None
    schema_version: int = 4


@dataclass(frozen=True)
class CapturedPackageQuote:
    package: OptionPackage
    risk: OptionPackageRisk
    quote: OptionPackageQuote


def contract_from_ticker(contract, ticker) -> QuoteContract:
    model = getattr(ticker, "modelGreeks", None)
    bid = _nonnegative_or_none(getattr(ticker, "bid", None))
    ask = _nonnegative_or_none(getattr(ticker, "ask", None))
    last = _nonnegative_or_none(getattr(ticker, "last", None))
    reference = _midpoint(bid, ask) or last
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
        bid=bid,
        ask=ask,
        last=last,
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
        min_tick=_tick_size(contract, ticker, reference),
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
    chain_fingerprint: str | None = None,
    target_expiry: str | None = None,
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
        chain_fingerprint=chain_fingerprint,
        target_expiry=target_expiry,
        session=(
            xsp_session_label_et(now)
            if str(symbol).strip().upper() == "XSP"
            else None
        ),
    )


def make_chain_manifest(underlying: object, chain: object) -> OptionChainManifest:
    return OptionChainManifest(
        underlying_con_id=_safe_int(getattr(underlying, "conId", None)),
        underlying_sec_type=getattr(underlying, "secType", None),
        symbol=str(getattr(underlying, "symbol", "") or "").strip().upper(),
        underlying_exchange=getattr(underlying, "exchange", None),
        currency=getattr(underlying, "currency", None),
        option_exchange=getattr(chain, "exchange", None),
        trading_class=getattr(chain, "tradingClass", None),
        multiplier=getattr(chain, "multiplier", None),
        expirations=tuple(
            sorted(str(value) for value in (getattr(chain, "expirations", ()) or ()))
        ),
        strikes=tuple(
            sorted(float(value) for value in (getattr(chain, "strikes", ()) or ()))
        ),
    )


def persist_chain_manifest(directory: Path, manifest: OptionChainManifest) -> str:
    payload = json.dumps(
        asdict(manifest),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    fingerprint = hashlib.sha256(payload).hexdigest()
    path = directory / "chains" / f"{fingerprint}.json"
    if path.exists():
        return fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return fingerprint


def append_snapshot(path: Path, snapshot: QuoteSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(asdict(snapshot), separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _repair_jsonl_tail(handle)
        handle.seek(0, os.SEEK_END)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def iter_snapshot_payloads(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid snapshot JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: snapshot must be an object")
            yield payload


def iter_snapshots(path: Path) -> Iterator[QuoteSnapshot]:
    for payload in iter_snapshot_payloads(path):
        yield QuoteSnapshot(
            **{
                **payload,
                "underlying": QuoteContract(**payload["underlying"]),
                "options": [
                    QuoteContract(**option)
                    for option in payload.get("options", [])
                ],
                "errors": [
                    QuoteError(**error)
                    for error in payload.get("errors", [])
                ],
            }
        )


def quote_captured_option_package(
    snapshot: QuoteSnapshot,
    legs: tuple[ResolvedOptionLeg, ...],
    *,
    quantity: int = 1,
    intent: str = "enter",
    mode: str = "MID",
    max_age_sec: float | None = None,
    require_live: bool = False,
    require_provenance: bool = True,
) -> CapturedPackageQuote | None:
    """Project exact captured legs through the canonical live-intended quote kernel."""

    if require_provenance and (
        not _snapshot_has_provenance(snapshot)
        or any(leg.expiry != snapshot.target_expiry for leg in legs)
    ):
        return None
    observed_at = _parse_datetime(snapshot.ts)
    selected: list[QuoteContract] = []
    for leg in legs:
        right = "P" if leg.right == "PUT" else "C"
        matches = [
            option
            for option in snapshot.options
            if option.con_id is not None
            and option.con_id > 0
            and option.expiry == leg.expiry
            and str(option.right or "").strip().upper()[:1] == right
            and option.strike is not None
            and math.isclose(option.strike, leg.strike, abs_tol=1e-9)
        ]
        if len(matches) != 1:
            return None
        option = matches[0]
        quote_at = _parse_datetime(option.quote_time)
        age_sec = (
            max(0.0, (observed_at - quote_at).total_seconds())
            if observed_at is not None and quote_at is not None
            else None
        )
        if not quote_health(
            bid=option.bid,
            ask=option.ask,
            last=option.last,
            close=option.close,
            market_data_type=option.market_data_type,
            age_sec=age_sec,
            max_age_sec=max_age_sec,
            require_live=require_live,
            require_nbbo=True,
            require_age=max_age_sec is not None,
        )["eligible"]:
            return None
        selected.append(option)

    if len({option.con_id for option in selected}) != len(selected):
        return None
    identities = {
        (
            option.symbol,
            option.sec_type,
            option.exchange,
            option.currency,
            option.multiplier,
            option.trading_class,
        )
        for option in selected
    }
    if len(identities) != 1:
        return None
    symbol, sec_type, exchange, currency, multiplier, trading_class = identities.pop()
    quote = quote_option_package(
        (
            (leg.action, leg.ratio, option.bid, option.ask, option.last)
            for leg, option in zip(legs, selected)
        ),
        mode=mode,
        tick=min(
            (
                _tick_size(
                    option,
                    None,
                    _midpoint(option.bid, option.ask) or option.last,
                )
                for option in selected
            ),
            default=0.01,
        ),
    )
    if quote is None:
        return None
    try:
        package = OptionPackage(
            product=option_product_facts(
                str(symbol or snapshot.symbol),
                security_type=str(sec_type or "OPT"),
                exchange=str(exchange or "SMART"),
                currency=str(currency or "USD"),
                multiplier=float(multiplier or 0),
                trading_class=str(trading_class or "") or None,
                source="captured",
            ),
            legs=legs,
            quantity=quantity,
            debit_value=quote.limit_value,
            intent=intent,
        )
    except (TypeError, ValueError):
        return None
    risk = option_package_risk(package)
    return (
        CapturedPackageQuote(package=package, risk=risk, quote=quote)
        if risk is not None
        else None
    )


def snapshot_quality(
    snapshot: QuoteSnapshot,
    *,
    max_age_sec: float | None = None,
    require_live: bool = False,
    require_provenance: bool = False,
    require_all_options: bool = False,
    require_greeks: bool = False,
) -> dict[str, object]:
    observed_at = _parse_datetime(snapshot.ts)
    rows = []
    for option in snapshot.options:
        quote_at = _parse_datetime(option.quote_time)
        age_sec = (
            max(0.0, (observed_at - quote_at).total_seconds())
            if observed_at is not None and quote_at is not None
            else None
        )
        rows.append(
            quote_health(
                bid=option.bid,
                ask=option.ask,
                last=option.last,
                close=option.close,
                market_data_type=option.market_data_type,
                age_sec=age_sec,
                max_age_sec=max_age_sec,
                require_live=require_live,
                require_nbbo=True,
                require_age=max_age_sec is not None,
            )
        )
    qualified = [
        option.con_id is not None and option.con_id > 0
        for option in snapshot.options
    ]
    qualified_count = sum(qualified)
    eligible_count = sum(
        is_qualified and bool(row["eligible"])
        for is_qualified, row in zip(qualified, rows)
    )
    full_greek_count = sum(
        is_qualified
        and all(
            value is not None
            for value in (
                option.model_iv,
                option.model_delta,
                option.model_gamma,
                option.model_vega,
                option.model_theta,
                option.model_under_price,
            )
        )
        for option, is_qualified in zip(snapshot.options, qualified)
    )
    total = len(snapshot.options)
    reasons: list[str] = []
    if eligible_count <= 0:
        reasons.append("no_eligible_options")
    if require_provenance and not _snapshot_has_provenance(snapshot):
        reasons.append("provenance_incomplete")
    if require_all_options and qualified_count != total:
        reasons.append("unqualified_options")
    if require_all_options and eligible_count != total:
        reasons.append("ineligible_options")
    if require_greeks and full_greek_count != total:
        reasons.append("greeks_incomplete")
    return {
        "requirements": {
            "require_nbbo": True,
            "require_streaming_live": require_live,
            "max_age_sec": max_age_sec,
            "require_provenance": require_provenance,
            "require_all_options": require_all_options,
            "require_greeks": require_greeks,
        },
        "complete": not reasons,
        "reasons": tuple(reasons),
        "provenance_complete": _snapshot_has_provenance(snapshot),
        "total_options": total,
        "qualified_options": qualified_count,
        "invalid_options": total - qualified_count,
        "timestamped_options": sum(
            is_qualified and option.quote_time is not None
            for option, is_qualified in zip(snapshot.options, qualified)
        ),
        "nbbo_options": sum(
            is_qualified and bool(row["has_nbbo"])
            for is_qualified, row in zip(qualified, rows)
        ),
        "eligible_options": eligible_count,
        "live_options": sum(
            is_qualified and bool(row["live"])
            for is_qualified, row in zip(qualified, rows)
        ),
        "streaming_options": sum(
            is_qualified and bool(row["streaming"])
            for is_qualified, row in zip(qualified, rows)
        ),
        "delayed_options": sum(
            is_qualified and bool(row["delayed"])
            for is_qualified, row in zip(qualified, rows)
        ),
        "full_greek_options": full_greek_count,
        "errors": len(snapshot.errors),
    }


def _snapshot_has_provenance(snapshot: QuoteSnapshot) -> bool:
    fingerprint = str(snapshot.chain_fingerprint or "").strip().lower()
    expiry = str(snapshot.target_expiry or "").strip()
    underlying_id = snapshot.underlying.con_id
    return (
        len(fingerprint) == 64
        and all(char in "0123456789abcdef" for char in fingerprint)
        and len(expiry) == 8
        and expiry.isdigit()
        and underlying_id is not None
        and underlying_id > 0
        and str(snapshot.underlying.symbol or "").strip().upper()
        == str(snapshot.symbol or "").strip().upper()
    )


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


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _repair_jsonl_tail(handle) -> None:
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end <= 0:
        return
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        return
    start = end
    while start > 0:
        size = min(8192, start)
        start -= size
        handle.seek(start)
        chunk = handle.read(size)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            start += newline + 1
            break
    handle.seek(start)
    tail = handle.read(end - start)
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        handle.truncate(start)
    else:
        handle.seek(0, os.SEEK_END)
        handle.write(b"\n")
