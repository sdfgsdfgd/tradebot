from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import math


_ACTIVE_STATUSES = frozenset({"STAGED", "WORKING", "CANCELING"})


@dataclass(frozen=True)
class OrderReservation:
    account: str
    product_domain: str
    sec_type: str
    structure: str
    status: str
    max_loss: float | None


@dataclass(frozen=True)
class OrderReservationSummary:
    account: str
    active_count: int
    unknown_active_count: int
    reserved_max_loss: float | None
    complete: bool
    reason: str

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrderReservationCapacityRequest:
    account: str
    product_domain: str
    sec_type: str
    structure: str
    candidate_max_loss: float | None
    available_capacity: float | None


@dataclass(frozen=True)
class OrderReservationCapacityDecision:
    allow: bool
    reason: str
    account: str
    product_domain: str
    sec_type: str
    structure: str
    reserved_max_loss: float | None
    candidate_max_loss: float | None
    total_max_loss: float | None
    available_capacity: float | None
    remaining_capacity: float | None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _token(value: object) -> str:
    return str(value or "").strip().upper()


def _positive_finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _is_target_reservation(
    reservation: OrderReservation,
    *,
    account: str,
) -> bool:
    return bool(
        str(reservation.account or "").strip() == account
        and _token(reservation.product_domain) == "XSP"
        and _token(reservation.sec_type) == "BAG"
        and _token(reservation.structure) == "VERTICAL_CREDIT"
        and _token(reservation.status) in _ACTIVE_STATUSES
    )


def summarize_order_reservations(
    reservations: Iterable[OrderReservation],
    *,
    account: str,
) -> OrderReservationSummary:
    account_clean = str(account or "").strip()
    active_count = 0
    unknown_active_count = 0
    reserved_max_loss = 0.0

    for reservation in reservations:
        if not _is_target_reservation(reservation, account=account_clean):
            continue

        active_count += 1
        max_loss = _positive_finite(reservation.max_loss)
        if max_loss is None:
            unknown_active_count += 1
            continue

        reserved_max_loss += max_loss
        if not math.isfinite(reserved_max_loss):
            unknown_active_count += 1

    if unknown_active_count:
        return OrderReservationSummary(
            account=account_clean,
            active_count=active_count,
            unknown_active_count=unknown_active_count,
            reserved_max_loss=None,
            complete=False,
            reason="active_risk_unknown",
        )

    return OrderReservationSummary(
        account=account_clean,
        active_count=active_count,
        unknown_active_count=0,
        reserved_max_loss=float(reserved_max_loss),
        complete=True,
        reason="reservation_complete",
    )


def _nonnegative_finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def evaluate_order_reservation_capacity(
    request: OrderReservationCapacityRequest,
    summary: OrderReservationSummary,
) -> OrderReservationCapacityDecision:
    account = str(request.account or "").strip()
    product_domain = _token(request.product_domain)
    sec_type = _token(request.sec_type)
    structure = str(request.structure or "").strip().lower()

    reserved_max_loss = _nonnegative_finite(summary.reserved_max_loss)
    candidate_max_loss = _positive_finite(request.candidate_max_loss)
    available_capacity = _positive_finite(request.available_capacity)

    def _decision(
        *,
        allow: bool,
        reason: str,
        total_max_loss: float | None = None,
        remaining_capacity: float | None = None,
    ) -> OrderReservationCapacityDecision:
        return OrderReservationCapacityDecision(
            allow=allow,
            reason=reason,
            account=account,
            product_domain=product_domain,
            sec_type=sec_type,
            structure=structure,
            reserved_max_loss=reserved_max_loss,
            candidate_max_loss=candidate_max_loss,
            total_max_loss=total_max_loss,
            available_capacity=available_capacity,
            remaining_capacity=remaining_capacity,
        )

    summary_account = str(summary.account or "").strip()
    if not account or not summary_account or account != summary_account:
        return _decision(allow=False, reason="account_mismatch")

    if product_domain != "XSP":
        return _decision(allow=False, reason="product_policy_unavailable")

    if sec_type != "BAG" or structure != "vertical_credit":
        return _decision(allow=False, reason="structure_invalid")

    if (
        not summary.complete
        or int(summary.unknown_active_count) != 0
        or reserved_max_loss is None
    ):
        return _decision(allow=False, reason="reservation_incomplete")

    if candidate_max_loss is None:
        return _decision(allow=False, reason="candidate_risk_unknown")

    if available_capacity is None:
        return _decision(allow=False, reason="capacity_unavailable")

    total_max_loss = reserved_max_loss + candidate_max_loss
    if not math.isfinite(total_max_loss):
        return _decision(allow=False, reason="candidate_risk_unknown")

    remaining_capacity = available_capacity - total_max_loss
    allow = total_max_loss <= available_capacity
    return _decision(
        allow=allow,
        reason="capacity_available" if allow else "capacity_exceeded",
        total_max_loss=float(total_max_loss),
        remaining_capacity=float(remaining_capacity),
    )
