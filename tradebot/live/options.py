"""Qualified option-package projection and broker admission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from ib_insync import Bag, ComboLeg, Contract

from ..option_package import (
    OptionPackage,
    OptionPackageRisk,
    ResolvedOptionLeg,
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
class LiveOptionPackage:
    package: OptionPackage
    risk: OptionPackageRisk
    contract: Bag
    legs: tuple[QualifiedOptionLeg, ...]


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


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
