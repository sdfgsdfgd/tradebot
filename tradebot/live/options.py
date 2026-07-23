"""Qualified option-package projection and broker admission."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math

from ib_insync import Bag, ComboLeg, Contract, Option

from ..engines.execution import (
    _limit_price_for_mode,
    _midpoint,
    _round_to_tick,
    _sanitize_nbbo,
    _tick_size,
    _ticker_price,
)
from ..option_package import (
    OptionPackage,
    OptionPackageEntryIntent,
    OptionPackageRisk,
    ResolvedOptionLeg,
    option_package_debit_value,
    option_package_risk,
    option_product_facts,
)
from ..order_admission import (
    OrderAdmissionDecision,
    OrderAdmissionFacts,
    OrderAdmissionLeg,
    OrderAdmissionRequest,
    evaluate_order_admission,
)


@dataclass(frozen=True)
class QualifiedOptionLeg:
    contract: Contract
    action: str
    ratio: int

    def __post_init__(self) -> None:
        action = str(self.action or "").strip().upper()
        try:
            ratio = int(self.ratio)
        except (TypeError, ValueError):
            ratio = 0
        if action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "ratio", ratio)


@dataclass(frozen=True)
class LiveOptionPackageDraft:
    """Immutable qualified-leg selection shared by option-capable UIs."""

    legs: tuple[QualifiedOptionLeg, ...] = ()

    @staticmethod
    def _con_id(contract: Contract) -> int:
        return int(getattr(contract, "conId", 0) or 0)

    @staticmethod
    def _identity(contract: Contract) -> tuple[str, ...]:
        return tuple(
            str(getattr(contract, field, "") or "").strip().upper()
            for field in (
                "symbol",
                "secType",
                "lastTradeDateOrContractMonth",
                "currency",
                "exchange",
                "tradingClass",
                "multiplier",
            )
        )

    def leg_for(self, contract: Contract | None) -> QualifiedOptionLeg | None:
        if contract is None:
            return None
        con_id = self._con_id(contract)
        return next(
            (leg for leg in self.legs if self._con_id(leg.contract) == con_id),
            None,
        )

    def cycle(self, contract: Contract, *, max_legs: int = 8) -> LiveOptionPackageDraft:
        """Cycle one row through unselected -> BUY -> SELL -> unselected."""

        con_id = self._con_id(contract)
        if con_id <= 0:
            raise ValueError("option leg is not qualified")
        current = self.leg_for(contract)
        if current is not None:
            remaining = tuple(
                leg for leg in self.legs if self._con_id(leg.contract) != con_id
            )
            return (
                LiveOptionPackageDraft(
                    (*remaining, QualifiedOptionLeg(contract, "SELL", current.ratio))
                )
                if current.action == "BUY"
                else LiveOptionPackageDraft(remaining)
            )

        if len(self.legs) >= max(2, int(max_legs)):
            raise ValueError(f"option package is limited to {max_legs} legs")
        if self.legs and self._identity(contract) != self._identity(self.legs[0].contract):
            raise ValueError("all package legs must share product, expiry, venue, and multiplier")
        return LiveOptionPackageDraft((*self.legs, QualifiedOptionLeg(contract, "BUY", 1)))

    def adjust_ratio(self, contract: Contract, delta: int) -> LiveOptionPackageDraft:
        current = self.leg_for(contract)
        if current is None:
            raise ValueError("select the option leg before changing its ratio")
        ratio = max(1, current.ratio + int(delta))
        return LiveOptionPackageDraft(
            tuple(
                QualifiedOptionLeg(leg.contract, leg.action, ratio)
                if self._con_id(leg.contract) == self._con_id(contract)
                else leg
                for leg in self.legs
            )
        )


@dataclass(frozen=True)
class LiveOptionPackage:
    package: OptionPackage
    risk: OptionPackageRisk
    contract: Bag
    legs: tuple[QualifiedOptionLeg, ...]


@dataclass(frozen=True)
class LiveOptionPackageQuote:
    """One executable package quote with canonical payoff and BAG identity."""

    live: LiveOptionPackage
    bid_value: float
    ask_value: float
    mid_value: float
    limit_value: float
    tick: float


@dataclass(frozen=True)
class LiveOptionOrderQuote:
    """Executable single-leg or native BAG order projection."""

    contract: Contract
    legs: tuple[QualifiedOptionLeg, ...]
    action: str
    quantity: int
    limit_value: float
    bid_value: float | None
    ask_value: float | None
    last_value: float | None
    tick: float
    package: OptionPackage | None = None
    risk: OptionPackageRisk | None = None


@dataclass(frozen=True)
class ResolvedLiveOptionEntry:
    """Qualified contracts and quote streams for one canonical entry intent."""

    underlying: Contract
    legs: tuple[QualifiedOptionLeg, ...]
    tickers: tuple[object, ...]


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _chain_expiry(target: date, expirations: Sequence[object]) -> str | None:
    parsed: list[tuple[date, str]] = []
    for raw in expirations:
        value = str(raw or "").strip()
        try:
            expiry = datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            continue
        parsed.append((expiry, value))
    if not parsed:
        return None
    future = [candidate for candidate in parsed if candidate[0] >= target]
    return min(
        future or parsed,
        key=lambda candidate: abs((candidate[0] - target).days),
    )[1]


def _nearest_strike(strikes: Sequence[object], target: float) -> float | None:
    try:
        return min(
            (float(strike) for strike in strikes),
            key=lambda strike: abs(strike - target),
        )
    except (TypeError, ValueError):
        return None


async def _strike_by_delta(
    client: object,
    *,
    symbol: str,
    expiry: str,
    right: str,
    strikes: Sequence[object],
    trading_class: str | None,
    target_strike: float,
    target_delta: float,
    owner: str,
) -> float | None:
    try:
        target = abs(float(target_delta))
        strike_values = sorted(float(strike) for strike in strikes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(target) or not 0 < target <= 1 or not strike_values:
        return None

    center = min(
        range(len(strike_values)),
        key=lambda index: abs(strike_values[index] - target_strike),
    )
    window = strike_values[max(center - 10, 0) : center + 11]
    candidates = [
        Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange="SMART",
            currency="USD",
            tradingClass=trading_class,
        )
        for strike in window
    ]
    qualified = await client.qualify_proxy_contracts(*candidates)
    contracts = (
        tuple(qualified)
        if qualified and len(qualified) == len(candidates)
        else tuple(candidates)
    )

    selected: tuple[float, Contract] | None = None
    try:
        for contract, strike in zip(contracts, window):
            ticker = await client.ensure_ticker(contract, owner=owner)
            delta = None
            for _ in range(6):
                for attr in ("modelGreeks", "bidGreeks", "askGreeks", "lastGreeks"):
                    greeks = getattr(ticker, attr, None)
                    raw = getattr(greeks, "delta", None) if greeks is not None else None
                    try:
                        delta = float(raw) if raw is not None else None
                    except (TypeError, ValueError):
                        delta = None
                    if delta is not None:
                        break
                if delta is not None:
                    break
                await asyncio.sleep(0.05)
            if delta is None or not math.isfinite(delta):
                continue
            difference = abs(abs(delta) - target)
            if selected is None or difference < selected[0]:
                selected = difference, contract
    finally:
        release_ticker = getattr(client, "release_ticker", None)
        if callable(release_ticker):
            for contract in contracts:
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    release_ticker(con_id, owner=owner)
    return float(getattr(selected[1], "strike", 0.0)) if selected else None


async def resolve_live_option_entry(
    client: object,
    *,
    symbol: str,
    intent: OptionPackageEntryIntent,
    anchor: date,
    owner: str,
) -> ResolvedLiveOptionEntry:
    """Resolve one shared entry intent against the broker's current chain."""

    chain_info = await client.stock_option_chain(symbol)
    if not chain_info:
        raise ValueError(f"Chain: not found for {symbol}")
    underlying, chain = chain_info
    acquired: set[int] = set()
    try:
        ticker = await client.ensure_ticker(underlying, owner=owner)
        underlying_con_id = int(getattr(underlying, "conId", 0) or 0)
        if underlying_con_id:
            acquired.add(underlying_con_id)
        spot = _ticker_price(ticker)
        if spot is None:
            raise ValueError(f"Spot: n/a for {symbol}")

        expiry = _chain_expiry(
            intent.target_expiry(anchor),
            getattr(chain, "expirations", ()),
        )
        if expiry is None:
            raise ValueError(f"Expiry: none for {symbol}")

        strikes = tuple(getattr(chain, "strikes", ()) or ())
        trading_class = getattr(chain, "tradingClass", None)
        candidates: list[Option] = []
        for leg in intent.legs:
            target_strike = intent.target_strike(leg, spot)
            right = "P" if leg.right == "PUT" else "C"
            strike = (
                await _strike_by_delta(
                    client,
                    symbol=symbol,
                    expiry=expiry,
                    right=right,
                    strikes=strikes,
                    trading_class=trading_class,
                    target_strike=target_strike,
                    target_delta=leg.delta,
                    owner=owner,
                )
                if leg.delta is not None
                else None
            )
            strike = strike if strike is not None else _nearest_strike(strikes, target_strike)
            if strike is None:
                raise ValueError(f"Strike: none for {symbol}")
            candidates.append(
                Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange="SMART",
                    currency="USD",
                    tradingClass=trading_class,
                )
            )

        qualified = await client.qualify_proxy_contracts(*candidates)
        contracts = (
            tuple(qualified)
            if qualified and len(qualified) == len(candidates)
            else tuple(candidates)
        )
        legs = tuple(
            QualifiedOptionLeg(contract=contract, action=leg.action, ratio=leg.qty)
            for contract, leg in zip(contracts, intent.legs)
        )
        tickers = []
        for leg in legs:
            ticker = await client.ensure_ticker(leg.contract, owner=owner)
            tickers.append(ticker)
            con_id = int(getattr(leg.contract, "conId", 0) or 0)
            if con_id:
                acquired.add(con_id)
        return ResolvedLiveOptionEntry(
            underlying=underlying,
            legs=legs,
            tickers=tuple(tickers),
        )
    except Exception:
        release_ticker = getattr(client, "release_ticker", None)
        if callable(release_ticker):
            for con_id in acquired:
                release_ticker(con_id, owner=owner)
        raise


def normalize_option_position_close(
    positions: Sequence[tuple[Contract, object]],
) -> tuple[tuple[QualifiedOptionLeg, ...], int]:
    """Invert complete option positions into minimal BAG ratios and quantity."""

    resolved: list[tuple[Contract, int]] = []
    for contract, raw_position in positions:
        try:
            position = float(raw_position)
        except (TypeError, ValueError):
            raise ValueError("option position must be an integer") from None
        if not math.isfinite(position) or not position.is_integer():
            raise ValueError("option position must be an integer")
        signed = int(position)
        if signed:
            resolved.append((contract, signed))
    if not resolved:
        raise ValueError("option package has no open positions")

    quantity = math.gcd(*(abs(position) for _contract, position in resolved))
    return (
        tuple(
            QualifiedOptionLeg(
                contract=contract,
                action="SELL" if position > 0 else "BUY",
                ratio=abs(position) // quantity,
            )
            for contract, position in resolved
        ),
        quantity,
    )


def resolve_live_option_package(
    *,
    symbol: str,
    legs: Sequence[QualifiedOptionLeg],
    quantity: int,
    debit_value: float,
    intent: str,
) -> LiveOptionPackage | None:
    """Project qualified contracts into one canonical package and native BAG."""

    normalized_symbol = str(symbol or "").strip().upper()
    materialized = tuple(legs)
    if not normalized_symbol or len(materialized) < 2:
        return None

    contracts = tuple(leg.contract for leg in materialized)
    symbols = {str(getattr(contract, "symbol", "") or "").strip().upper() for contract in contracts}
    security_types = {
        str(getattr(contract, "secType", "") or "").strip().upper()
        for contract in contracts
    }
    exchanges = {
        str(getattr(contract, "exchange", "") or "").strip().upper()
        for contract in contracts
    }
    currencies = {
        str(getattr(contract, "currency", "") or "").strip().upper()
        for contract in contracts
    }
    multipliers = tuple(
        _positive_float(getattr(contract, "multiplier", None))
        for contract in contracts
    )
    trading_classes = {
        str(getattr(contract, "tradingClass", "") or "").strip().upper()
        for contract in contracts
    }
    con_ids = {int(getattr(contract, "conId", 0) or 0) for contract in contracts}
    if (
        symbols != {normalized_symbol}
        or len(security_types) != 1
        or next(iter(security_types)) not in {"OPT", "FOP"}
        or len(exchanges) != 1
        or not next(iter(exchanges))
        or len(currencies) != 1
        or not next(iter(currencies))
        or any(multiplier is None for multiplier in multipliers)
        or len(set(multipliers)) != 1
        or len(trading_classes) != 1
        or len(con_ids) != len(contracts)
        or any(con_id <= 0 for con_id in con_ids)
    ):
        return None

    security_type = next(iter(security_types))
    exchange = next(iter(exchanges))
    currency = next(iter(currencies))
    multiplier = multipliers[0]
    assert multiplier is not None
    trading_class = next(iter(trading_classes)) or None
    try:
        package = OptionPackage(
            product=option_product_facts(
                normalized_symbol,
                security_type=security_type,
                exchange=exchange,
                currency=currency,
                multiplier=multiplier,
                trading_class=trading_class,
                source="broker",
            ),
            legs=tuple(
                ResolvedOptionLeg(
                    action=leg.action,
                    right=getattr(leg.contract, "right", ""),
                    strike=getattr(leg.contract, "strike", None),
                    ratio=leg.ratio,
                    expiry=str(
                        getattr(
                            leg.contract,
                            "lastTradeDateOrContractMonth",
                            "",
                        )
                        or ""
                    ),
                )
                for leg in materialized
            ),
            quantity=quantity,
            debit_value=debit_value,
            intent=intent,
        )
    except (TypeError, ValueError):
        return None

    risk = option_package_risk(package)
    if risk is None:
        return None
    bag = Bag(
        symbol=normalized_symbol,
        exchange=exchange,
        currency=currency,
        comboLegs=[
            ComboLeg(
                conId=int(leg.contract.conId),
                ratio=leg.ratio,
                action=leg.action,
                exchange=exchange,
            )
            for leg in materialized
        ],
    )
    return LiveOptionPackage(
        package=package,
        risk=risk,
        contract=bag,
        legs=materialized,
    )


def quote_live_option_package(
    *,
    symbol: str,
    legs: Sequence[QualifiedOptionLeg],
    tickers: Sequence[object],
    quantity: int,
    intent: str,
    mode: str,
) -> LiveOptionPackageQuote | None:
    """Price qualified legs once, then project the exact quote into canonical risk and BAG."""

    materialized = tuple(legs)
    quote_sources = tuple(tickers)
    if len(materialized) < 2 or len(materialized) != len(quote_sources):
        return None

    bid_rows: list[tuple[str, int, float]] = []
    ask_rows: list[tuple[str, int, float]] = []
    mid_rows: list[tuple[str, int, float]] = []
    limit_rows: list[tuple[str, int, float]] = []
    tick: float | None = None
    for leg, ticker in zip(materialized, quote_sources):
        bid, ask, last = _sanitize_nbbo(
            getattr(ticker, "bid", None),
            getattr(ticker, "ask", None),
            getattr(ticker, "last", None),
        )
        mid = _midpoint(bid, ask)
        reference = mid or last
        if reference is None:
            return None
        leg_bid = bid or reference
        leg_ask = ask or reference
        desired = _limit_price_for_mode(
            bid,
            ask,
            last,
            action=leg.action,
            mode=mode,
        )
        if desired is None:
            return None

        leg_tick = _tick_size(leg.contract, ticker, desired)
        tick = leg_tick if tick is None else min(tick, leg_tick)
        bid_rows.append(
            (leg.action, leg.ratio, leg_bid if leg.action == "BUY" else leg_ask)
        )
        ask_rows.append(
            (leg.action, leg.ratio, leg_ask if leg.action == "BUY" else leg_bid)
        )
        mid_rows.append((leg.action, leg.ratio, reference))
        limit_rows.append((leg.action, leg.ratio, desired))

    values = tuple(
        option_package_debit_value(rows)
        for rows in (bid_rows, ask_rows, mid_rows, limit_rows)
    )
    if any(value is None for value in values):
        return None
    bid_value, ask_value, mid_value, limit_value = (
        float(value) for value in values if value is not None
    )
    resolved_tick = tick or 0.01
    rounded_limit = _round_to_tick(limit_value, resolved_tick)
    if rounded_limit is None or math.isclose(rounded_limit, 0.0, abs_tol=resolved_tick / 2):
        return None
    live = resolve_live_option_package(
        symbol=symbol,
        legs=materialized,
        quantity=quantity,
        debit_value=float(rounded_limit),
        intent=intent,
    )
    if live is None:
        return None
    return LiveOptionPackageQuote(
        live=live,
        bid_value=min(bid_value, ask_value),
        ask_value=max(bid_value, ask_value),
        mid_value=mid_value,
        limit_value=float(rounded_limit),
        tick=float(resolved_tick),
    )


def quote_live_option_order(
    *,
    symbol: str,
    legs: Sequence[QualifiedOptionLeg],
    tickers: Sequence[object],
    quantity: int,
    intent: str,
    mode: str,
) -> LiveOptionOrderQuote | None:
    """Quote the exact contract submitted by both staging and chase repricing."""

    materialized = tuple(legs)
    quote_sources = tuple(tickers)
    try:
        order_quantity = int(quantity)
    except (TypeError, ValueError):
        return None
    if (
        not materialized
        or len(materialized) != len(quote_sources)
        or order_quantity <= 0
    ):
        return None

    if len(materialized) == 1:
        leg = materialized[0]
        ticker = quote_sources[0]
        bid, ask, last = _sanitize_nbbo(
            getattr(ticker, "bid", None),
            getattr(ticker, "ask", None),
            getattr(ticker, "last", None),
        )
        limit = _limit_price_for_mode(
            bid,
            ask,
            last,
            action=leg.action,
            mode=mode,
        )
        if limit is None:
            return None
        tick = _tick_size(leg.contract, ticker, limit) or 0.01
        rounded = _round_to_tick(limit, tick)
        if rounded is None:
            return None
        return LiveOptionOrderQuote(
            contract=leg.contract,
            legs=materialized,
            action=leg.action,
            quantity=order_quantity,
            limit_value=float(rounded),
            bid_value=bid,
            ask_value=ask,
            last_value=last,
            tick=float(tick),
        )

    package_quote = quote_live_option_package(
        symbol=symbol,
        legs=materialized,
        tickers=quote_sources,
        quantity=order_quantity,
        intent=intent,
        mode=mode,
    )
    if package_quote is None:
        return None
    live = package_quote.live
    return LiveOptionOrderQuote(
        contract=live.contract,
        legs=live.legs,
        action="BUY",
        quantity=order_quantity,
        limit_value=package_quote.limit_value,
        bid_value=package_quote.bid_value,
        ask_value=package_quote.ask_value,
        last_value=package_quote.mid_value,
        tick=package_quote.tick,
        package=live.package,
        risk=live.risk,
    )


def _admission_request(
    *,
    account: str,
    package: OptionPackage | None,
    risk: OptionPackageRisk | None,
    contract: Contract,
    legs: Sequence[QualifiedOptionLeg],
    action: str,
    quantity: int,
    limit_price: float,
    intent: str,
) -> OrderAdmissionRequest:
    symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
    return OrderAdmissionRequest(
        account=str(account or "").strip(),
        intent=str(intent or "").strip().lower(),
        product_domain=(
            package.product.underlying_symbol
            if package is not None
            else symbol
        ),
        structure=risk.structure if risk is not None else "",
        sec_type=str(getattr(contract, "secType", "") or "").strip().upper(),
        symbol=symbol,
        currency=str(getattr(contract, "currency", "") or "").strip().upper(),
        exchange=str(getattr(contract, "exchange", "") or "").strip().upper(),
        action=str(action or "").strip().upper(),
        quantity=int(quantity),
        limit_price=float(limit_price),
        max_loss=risk.max_loss if risk is not None else None,
        legs=tuple(
            OrderAdmissionLeg(
                con_id=int(getattr(leg.contract, "conId", 0) or 0),
                ratio=leg.ratio,
                action=leg.action,
                exchange=str(
                    getattr(leg.contract, "exchange", "") or ""
                ).strip().upper(),
            )
            for leg in legs
        ),
    )


def _preview_facts(preview: object) -> OrderAdmissionFacts:
    return OrderAdmissionFacts(
        status=getattr(preview, "status", None),
        init_margin_before=getattr(preview, "init_margin_before", None),
        init_margin_change=getattr(preview, "init_margin_change", None),
        init_margin_after=getattr(preview, "init_margin_after", None),
        maintenance_margin_before=getattr(preview, "maintenance_margin_before", None),
        maintenance_margin_change=getattr(preview, "maintenance_margin_change", None),
        maintenance_margin_after=getattr(preview, "maintenance_margin_after", None),
        equity_with_loan_before=getattr(preview, "equity_with_loan_before", None),
        equity_with_loan_change=getattr(preview, "equity_with_loan_change", None),
        equity_with_loan_after=getattr(preview, "equity_with_loan_after", None),
        commission=getattr(preview, "commission", None),
        min_commission=getattr(preview, "min_commission", None),
        max_commission=getattr(preview, "max_commission", None),
        commission_currency=getattr(preview, "commission_currency", None),
        warning_text=getattr(preview, "warning_text", None),
    )


async def preview_and_admit_option_order(
    client: object,
    *,
    account: str,
    package: OptionPackage | None,
    risk: OptionPackageRisk | None,
    contract: Contract,
    legs: Sequence[QualifiedOptionLeg],
    action: str,
    quantity: int,
    limit_price: float,
    intent: str,
) -> OrderAdmissionDecision:
    """Fail malformed/unsupported BAGs before a broker preview; preview all viable ones."""

    request = _admission_request(
        account=account,
        package=package,
        risk=risk,
        contract=contract,
        legs=legs,
        action=action,
        quantity=quantity,
        limit_price=limit_price,
        intent=intent,
    )
    preflight = evaluate_order_admission(request, OrderAdmissionFacts())
    if preflight.reason != "preview_incomplete":
        return preflight

    preview = await client.preview_limit_order(
        contract,
        action,
        quantity,
        limit_price,
        outside_rth=False,
    )
    return evaluate_order_admission(request, _preview_facts(preview))
