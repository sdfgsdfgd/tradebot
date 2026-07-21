from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class OrderAdmissionLeg:
    con_id: int
    ratio: int
    action: str
    exchange: str


@dataclass(frozen=True)
class OrderAdmissionRequest:
    account: str
    product_domain: str
    structure: str
    sec_type: str
    symbol: str
    currency: str
    exchange: str
    action: str
    quantity: int
    limit_price: float
    max_loss: float | None
    legs: tuple[OrderAdmissionLeg, ...]


@dataclass(frozen=True)
class OrderAdmissionFacts:
    status: str | None = None
    init_margin_before: float | None = None
    init_margin_change: float | None = None
    init_margin_after: float | None = None
    maintenance_margin_before: float | None = None
    maintenance_margin_change: float | None = None
    maintenance_margin_after: float | None = None
    equity_with_loan_before: float | None = None
    equity_with_loan_change: float | None = None
    equity_with_loan_after: float | None = None
    commission: float | None = None
    min_commission: float | None = None
    max_commission: float | None = None
    commission_currency: str | None = None
    warning_text: str | None = None


@dataclass(frozen=True)
class OrderAdmissionDecision:
    allow: bool
    reason: str
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _token(value: object) -> str:
    return str(value or "").strip().upper()


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _trace(
    request: OrderAdmissionRequest,
    facts: OrderAdmissionFacts,
) -> dict[str, object]:
    return {
        "account": str(request.account or "").strip(),
        "product_domain": _token(request.product_domain),
        "structure": str(request.structure or "").strip().lower(),
        "sec_type": _token(request.sec_type),
        "symbol": _token(request.symbol),
        "currency": _token(request.currency),
        "exchange": _token(request.exchange),
        "action": _token(request.action),
        "quantity": int(request.quantity),
        "limit_price": _finite(request.limit_price),
        "max_loss": _finite(request.max_loss),
        "init_margin_change": _finite(facts.init_margin_change),
        "init_margin_after": _finite(facts.init_margin_after),
        "equity_with_loan_after": _finite(facts.equity_with_loan_after),
        "commission": _finite(facts.commission),
        "commission_currency": _token(facts.commission_currency),
    }


def _decision(
    request: OrderAdmissionRequest,
    facts: OrderAdmissionFacts,
    *,
    allow: bool,
    reason: str,
) -> OrderAdmissionDecision:
    return OrderAdmissionDecision(
        allow=bool(allow),
        reason=str(reason),
        trace=_trace(request, facts),
    )


def _xsp_credit_bag_identity_valid(request: OrderAdmissionRequest) -> bool:
    if str(request.structure or "").strip().lower() != "vertical_credit":
        return False
    if _token(request.sec_type) != "BAG":
        return False
    if _token(request.currency) != "USD":
        return False
    if _token(request.exchange) != "SMART":
        return False
    if _token(request.action) != "BUY":
        return False
    if int(request.quantity) <= 0:
        return False
    limit_price = _finite(request.limit_price)
    if limit_price is None or limit_price >= 0:
        return False
    if len(request.legs) != 2:
        return False

    actions: list[str] = []
    con_ids: set[int] = set()
    for leg in request.legs:
        con_id = int(leg.con_id)
        ratio = int(leg.ratio)
        action = _token(leg.action)
        exchange = _token(leg.exchange)
        if con_id <= 0 or ratio <= 0 or action not in {"BUY", "SELL"} or not exchange:
            return False
        actions.append(action)
        con_ids.add(con_id)

    return sorted(actions) == ["BUY", "SELL"] and len(con_ids) == 2


def evaluate_order_admission(
    request: OrderAdmissionRequest,
    facts: OrderAdmissionFacts,
) -> OrderAdmissionDecision:
    product_domain = _token(request.product_domain)
    symbol = _token(request.symbol)

    if not product_domain or not symbol or product_domain != symbol:
        return _decision(request, facts, allow=False, reason="identity_mismatch")

    if product_domain != "XSP":
        return _decision(
            request,
            facts,
            allow=False,
            reason="product_policy_unavailable",
        )

    if not _xsp_credit_bag_identity_valid(request):
        return _decision(request, facts, allow=False, reason="structure_invalid")

    max_loss = _finite(request.max_loss)
    if max_loss is None:
        return _decision(request, facts, allow=False, reason="max_loss_unknown")
    if max_loss <= 0:
        return _decision(request, facts, allow=False, reason="max_loss_invalid")

    warning_text = str(facts.warning_text or "").strip()
    if warning_text:
        return _decision(request, facts, allow=False, reason="broker_warning")

    required_numeric = (
        facts.init_margin_before,
        facts.init_margin_change,
        facts.init_margin_after,
        facts.maintenance_margin_before,
        facts.maintenance_margin_change,
        facts.maintenance_margin_after,
        facts.equity_with_loan_before,
        facts.equity_with_loan_change,
        facts.equity_with_loan_after,
        facts.commission,
        facts.min_commission,
        facts.max_commission,
    )
    if (
        not str(facts.status or "").strip()
        or not _token(facts.commission_currency)
        or any(_finite(value) is None for value in required_numeric)
    ):
        return _decision(request, facts, allow=False, reason="preview_incomplete")

    if _token(facts.commission_currency) != _token(request.currency):
        return _decision(request, facts, allow=False, reason="currency_mismatch")

    status = _token(facts.status)
    if status in {"INACTIVE", "REJECTED", "CANCELLED", "APICANCELLED"}:
        return _decision(request, facts, allow=False, reason="broker_status_blocked")

    init_margin_after = _finite(facts.init_margin_after)
    equity_with_loan_after = _finite(facts.equity_with_loan_after)
    if (
        init_margin_after is None
        or equity_with_loan_after is None
        or init_margin_after < 0
        or equity_with_loan_after < 0
    ):
        return _decision(request, facts, allow=False, reason="broker_capacity_exceeded")

    return _decision(
        request,
        facts,
        allow=True,
        reason="broker_preview_admitted",
    )
