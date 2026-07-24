from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
import math

from .contract_identity import (
    future_exchange_for_symbol,
    future_multiplier_for_symbol,
    is_future_symbol,
)
from .utils.date_utils import add_business_days


@dataclass(frozen=True)
class LegConfig:
    action: str
    right: str
    moneyness_pct: float
    qty: int
    delta: float | None = None
    otm_offset_points: float = 0.0


OptionLegSpec = LegConfig


def _option_leg_finite_float(value: object, *, path: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{path} must be a finite number") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be a finite number")
    return parsed


def _option_leg_positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{path} must be a positive integer")
        parsed = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or not cleaned.lstrip("+-").isdigit():
            raise ValueError(f"{path} must be a positive integer")
        parsed = int(cleaned)
    else:
        raise ValueError(f"{path} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return parsed


def normalize_option_leg(raw: object, *, path: str = "leg") -> LegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object")

    action = str(raw.get("action") or "").strip().upper()
    if action not in {"BUY", "SELL"}:
        raise ValueError(f"{path}.action must be BUY or SELL")

    right = str(raw.get("right") or "").strip().upper()
    if right not in {"PUT", "CALL"}:
        raise ValueError(f"{path}.right must be PUT or CALL")

    if "moneyness_pct" not in raw:
        raise ValueError(f"{path}.moneyness_pct is required")
    moneyness_pct = _option_leg_finite_float(
        raw.get("moneyness_pct"),
        path=f"{path}.moneyness_pct",
    )

    qty = 1
    if "qty" in raw:
        qty = _option_leg_positive_int(raw.get("qty"), path=f"{path}.qty")

    delta = None
    if raw.get("delta") is not None:
        delta = _option_leg_finite_float(raw.get("delta"), path=f"{path}.delta")
        if not 0 < abs(delta) <= 1:
            raise ValueError(f"{path}.delta must satisfy 0 < abs(delta) <= 1")

    otm_offset_points = _option_leg_finite_float(
        raw.get("otm_offset_points", 0.0),
        path=f"{path}.otm_offset_points",
    )

    return LegConfig(
        action=action,
        right=right,
        moneyness_pct=moneyness_pct,
        qty=qty,
        delta=delta,
        otm_offset_points=otm_offset_points,
    )


def normalize_option_legs(raw: object, *, path: str = "legs") -> tuple[LegConfig, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list")
    return tuple(
        normalize_option_leg(leg, path=f"{path}[{index}]")
        for index, leg in enumerate(raw, start=1)
    )


_MISSING_OPTION_ENTRY_FIELD = object()


def _option_entry_source_value(
    source: object,
    field: str,
    default: object,
) -> object:
    if isinstance(source, Mapping):
        return source[field] if field in source else default
    return getattr(source, field, default)


def _option_entry_non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{path} must be a non-negative integer")
        parsed = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or not cleaned.lstrip("+-").isdigit():
            raise ValueError(f"{path} must be a non-negative integer")
        parsed = int(cleaned)
    else:
        raise ValueError(f"{path} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return parsed


def _option_entry_min_credit(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite non-negative number")
    parsed = _option_leg_finite_float(value, path=path)
    if parsed < 0:
        raise ValueError(f"{path} must be a finite non-negative number")
    return parsed


def _option_entry_legs(raw: object, *, path: str) -> tuple[LegConfig, ...]:
    if isinstance(raw, tuple) and all(isinstance(leg, LegConfig) for leg in raw):
        normalized = raw
    elif isinstance(raw, list) and all(isinstance(leg, LegConfig) for leg in raw):
        normalized = tuple(raw)
    elif isinstance(raw, list):
        normalized = normalize_option_legs(raw, path=path)
    else:
        raise ValueError(f"{path} must be a list or tuple of option legs")
    if not normalized:
        raise ValueError(f"{path} must contain at least one leg")
    return normalized


@dataclass(frozen=True)
class OptionPackageEntryIntent:
    legs: tuple[LegConfig, ...]
    dte: int
    quantity: int
    min_credit: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.legs, tuple)
            or not self.legs
            or any(not isinstance(leg, LegConfig) for leg in self.legs)
        ):
            raise ValueError("legs must be a non-empty tuple of LegConfig")
        object.__setattr__(
            self,
            "dte",
            _option_entry_non_negative_int(self.dte, path="dte"),
        )
        object.__setattr__(
            self,
            "quantity",
            _option_leg_positive_int(self.quantity, path="quantity"),
        )
        object.__setattr__(
            self,
            "min_credit",
            _option_entry_min_credit(self.min_credit, path="min_credit"),
        )

    def target_expiry(self, anchor: date) -> date:
        """Resolve the shared weekday-DTE target used by replay and live chain selection."""
        return add_business_days(anchor, self.dte)

    @staticmethod
    def target_strike(leg: LegConfig, spot: float) -> float:
        """Resolve moneyness into the broker-independent target strike."""
        price = _option_leg_finite_float(spot, path="spot")
        if price <= 0:
            raise ValueError("spot must be positive")
        percent_target = price * (
            1 - leg.moneyness_pct / 100.0
            if leg.right == "PUT"
            else 1 + leg.moneyness_pct / 100.0
        )
        return percent_target + (
            -leg.otm_offset_points
            if leg.right == "PUT"
            else leg.otm_offset_points
        )

    def resolved_legs(
        self,
        *,
        spot: float,
        expiry: date | str,
    ) -> tuple[ResolvedOptionLeg, ...]:
        """Resolve deterministic replay legs without introducing adapter semantics."""
        expiry_text = expiry.strftime("%Y%m%d") if isinstance(expiry, date) else str(expiry)
        return tuple(
            ResolvedOptionLeg(
                action=leg.action,
                right=leg.right,
                strike=self.target_strike(leg, spot),
                ratio=leg.qty,
                expiry=expiry_text,
            )
            for leg in self.legs
        )

    def required_credit(self, tick: float) -> float:
        resolved_tick = _option_leg_finite_float(tick, path="tick")
        if resolved_tick <= 0:
            raise ValueError("tick must be positive")
        return resolved_tick if self.min_credit is None else self.min_credit

    def admits_debit_value(self, debit_value: float, *, tick: float) -> bool:
        """Apply the same minimum-credit contract to signed live/backtest values."""
        value = _option_leg_finite_float(debit_value, path="debit_value")
        return value >= 0 or -value >= self.required_credit(tick)


def option_package_entry_intent(
    source: object,
    *,
    legs: object = _MISSING_OPTION_ENTRY_FIELD,
    path: str = "legs",
) -> OptionPackageEntryIntent:
    raw_legs = (
        _option_entry_source_value(
            source,
            "legs",
            _MISSING_OPTION_ENTRY_FIELD,
        )
        if legs is _MISSING_OPTION_ENTRY_FIELD
        else legs
    )
    if raw_legs is _MISSING_OPTION_ENTRY_FIELD:
        raise ValueError(f"{path} is required")

    return OptionPackageEntryIntent(
        legs=_option_entry_legs(raw_legs, path=path),
        dte=_option_entry_non_negative_int(
            _option_entry_source_value(source, "dte", 0),
            path="dte",
        ),
        quantity=_option_leg_positive_int(
            _option_entry_source_value(source, "quantity", 1),
            path="quantity",
        ),
        min_credit=_option_entry_min_credit(
            _option_entry_source_value(source, "min_credit", None),
            path="min_credit",
        ),
    )


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric != float(parsed) or parsed <= 0:
        return None
    return parsed


def _option_right(value: object) -> str | None:
    cleaned = str(value or "").strip().upper()
    if cleaned in {"P", "PUT"}:
        return "PUT"
    if cleaned in {"C", "CALL"}:
        return "CALL"
    return None


@dataclass(frozen=True)
class OptionProductFacts:
    """Economic identity shared by live option packages and deterministic replay."""

    underlying_symbol: str
    security_type: str
    exchange: str
    currency: str
    multiplier: float
    pricing_model: str
    trading_class: str | None = None
    settlement: str = "unknown"
    source: str = "configured"

    def __post_init__(self) -> None:
        symbol = str(self.underlying_symbol or "").strip().upper()
        security_type = str(self.security_type or "").strip().upper()
        exchange = str(self.exchange or "").strip().upper()
        currency = str(self.currency or "").strip().upper()
        multiplier = _finite_float(self.multiplier)
        pricing_model = str(self.pricing_model or "").strip().lower()
        trading_class = str(self.trading_class or "").strip().upper() or None
        settlement = str(self.settlement or "unknown").strip().lower()
        source = str(self.source or "configured").strip().lower()
        if not symbol:
            raise ValueError("underlying_symbol is required")
        if security_type not in {"OPT", "FOP"}:
            raise ValueError("security_type must be OPT or FOP")
        if not exchange:
            raise ValueError("exchange is required")
        if not currency:
            raise ValueError("currency is required")
        if multiplier is None or multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if pricing_model not in {"black_scholes", "black_76"}:
            raise ValueError("pricing_model must be black_scholes or black_76")
        object.__setattr__(self, "underlying_symbol", symbol)
        object.__setattr__(self, "security_type", security_type)
        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "pricing_model", pricing_model)
        object.__setattr__(self, "trading_class", trading_class)
        object.__setattr__(self, "settlement", settlement)
        object.__setattr__(self, "source", source)


_OPTION_PRODUCT_DEFAULTS = {
    "XSP": ("OPT", "CBOE", 100.0, "black_scholes", "cash"),
    "SPX": ("OPT", "CBOE", 100.0, "black_scholes", "cash"),
}


def option_product_facts(
    underlying_symbol: object,
    *,
    security_type: object | None = None,
    exchange: object | None = None,
    currency: object = "USD",
    multiplier: object | None = None,
    pricing_model: object | None = None,
    trading_class: object | None = None,
    settlement: object | None = None,
    source: object = "configured",
) -> OptionProductFacts:
    """Resolve one product identity; explicit broker/tape facts override safe defaults."""

    symbol = str(underlying_symbol or "").strip().upper()
    default = _OPTION_PRODUCT_DEFAULTS.get(symbol)
    if default is None and is_future_symbol(symbol):
        default = (
            "FOP",
            future_exchange_for_symbol(symbol) or "CME",
            future_multiplier_for_symbol(symbol),
            "black_76",
            "financial" if symbol == "MCL" else "future",
        )
    if default is None:
        default = ("OPT", "SMART", 100.0, "black_scholes", "unknown")
    resolved_security_type = (
        str(security_type).strip().upper() if security_type is not None else default[0]
    )
    return OptionProductFacts(
        underlying_symbol=symbol,
        security_type=resolved_security_type,
        exchange=str(exchange).strip().upper() if exchange is not None else default[1],
        currency=str(currency or "USD"),
        multiplier=default[2] if multiplier is None else float(multiplier),
        pricing_model=(
            str(pricing_model).strip().lower()
            if pricing_model is not None
            else ("black_76" if resolved_security_type == "FOP" else default[3])
        ),
        trading_class=(
            str(trading_class).strip().upper() if trading_class is not None else None
        ),
        settlement=str(settlement).strip().lower() if settlement is not None else default[4],
        source=str(source or "configured"),
    )


@dataclass(frozen=True)
class ResolvedOptionLeg:
    """Broker-independent resolved leg; quantity is a package ratio."""

    action: str
    right: str
    strike: float
    ratio: int
    expiry: str

    def __post_init__(self) -> None:
        action = str(self.action or "").strip().upper()
        right = _option_right(self.right)
        strike = _finite_float(self.strike)
        ratio = _positive_int(self.ratio)
        expiry = str(self.expiry or "").strip()
        if action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        if right is None:
            raise ValueError("right must be PUT or CALL")
        if strike is None or strike <= 0:
            raise ValueError("strike must be positive")
        if ratio is None:
            raise ValueError("ratio must be a positive integer")
        if not expiry:
            raise ValueError("expiry is required")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "expiry", expiry)


@dataclass(frozen=True)
class OptionPackage:
    """Canonical signed package economics; debit is positive and credit negative."""

    product: OptionProductFacts
    legs: tuple[ResolvedOptionLeg, ...]
    quantity: int
    debit_value: float
    intent: str = "enter"

    def __post_init__(self) -> None:
        quantity = _positive_int(self.quantity)
        debit_value = _finite_float(self.debit_value)
        intent = str(self.intent or "").strip().lower()
        if not isinstance(self.product, OptionProductFacts):
            raise ValueError("product must be OptionProductFacts")
        if not isinstance(self.legs, tuple) or not self.legs:
            raise ValueError("legs must be a non-empty tuple")
        if any(not isinstance(leg, ResolvedOptionLeg) for leg in self.legs):
            raise ValueError("legs must contain ResolvedOptionLeg values")
        if quantity is None:
            raise ValueError("quantity must be a positive integer")
        if debit_value is None:
            raise ValueError("debit_value must be finite")
        if intent not in {"enter", "exit", "resize"}:
            raise ValueError("intent must be enter, exit, or resize")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "debit_value", debit_value)
        object.__setattr__(self, "intent", intent)


@dataclass(frozen=True)
class OptionPackageRisk:
    structure: str
    right: str
    expiry: str
    width: float
    debit_value: float
    multiplier: float
    quantity: int
    max_loss: float
    max_profit: float | None = None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


DEFINED_RISK_OPTION_STRUCTURES = frozenset(
    {
        "vertical_credit",
        "vertical_debit",
        "butterfly_credit",
        "butterfly_debit",
        "iron_butterfly_credit",
        "iron_butterfly_debit",
        "iron_condor_credit",
        "iron_condor_debit",
        "defined_risk_combo",
    }
)


def option_package_debit_value(
    rows: Iterable[tuple[str, int, float | None]],
) -> float | None:
    """Aggregate option-leg values in debit units: BUY positive, SELL negative."""
    total = 0.0
    for action, ratio, value in rows:
        if value is None:
            return None

        cleaned_action = str(action or "").strip().upper()
        if cleaned_action == "BUY":
            sign = 1.0
        elif cleaned_action == "SELL":
            sign = -1.0
        else:
            raise ValueError(f"unsupported option leg action: {action!r}")

        total += sign * float(value) * int(ratio)

    return float(total)


def option_profit_target_hit(
    *,
    entry_value: float,
    current_value: float,
    profit_target: float,
) -> bool:
    """Return whether enabled package economics reached the profit target."""
    if not profit_target > 0:
        return False
    return (float(entry_value) - float(current_value)) >= (
        abs(float(entry_value)) * float(profit_target)
    )


def option_stop_loss_hit(
    *,
    entry_value: float,
    current_value: float,
    stop_loss: float,
    basis: str,
    max_loss: float | None,
) -> bool:
    """Return whether enabled package economics reached the stop loss."""
    if not stop_loss > 0:
        return False

    entry = float(entry_value)
    current = float(current_value)
    threshold = float(stop_loss)
    loss = max(0.0, current - entry)
    if basis == "credit":
        if entry >= 0:
            return current >= entry * (1.0 + threshold)
        return loss >= abs(entry) * threshold

    resolved_max_loss = abs(entry) if max_loss is None else float(max_loss)
    return loss >= resolved_max_loss * threshold


def option_package_risk(
    package: OptionPackage,
) -> OptionPackageRisk | None:
    """Return exact same-expiry defined-risk economics in account-currency units."""

    if not isinstance(package, OptionPackage):
        return None
    expiries = {leg.expiry for leg in package.legs}
    if len(expiries) != 1:
        return None

    def _terminal_profit(spot: float) -> float:
        payoff = 0.0
        for leg in package.legs:
            intrinsic = (
                max(0.0, spot - leg.strike)
                if leg.right == "CALL"
                else max(0.0, leg.strike - spot)
            )
            payoff += (1.0 if leg.action == "BUY" else -1.0) * leg.ratio * intrinsic
        return payoff - package.debit_value

    upper_slope = sum(
        (1.0 if leg.action == "BUY" else -1.0) * leg.ratio
        for leg in package.legs
        if leg.right == "CALL"
    )
    if upper_slope < 0:
        return None

    strikes = sorted({leg.strike for leg in package.legs})
    terminal_profits = [_terminal_profit(spot) for spot in [0.0, *strikes]]
    max_loss_units = max(0.0, -min(terminal_profits))
    max_loss = max_loss_units * package.product.multiplier * package.quantity
    if not math.isfinite(max_loss) or max_loss <= 0:
        return None
    max_profit = (
        None
        if upper_slope > 0
        else max(0.0, max(terminal_profits))
        * package.product.multiplier
        * package.quantity
    )

    rights = {leg.right for leg in package.legs}
    right = next(iter(rights)) if len(rights) == 1 else "MIXED"
    structure = _option_package_structure(package)
    if structure.startswith(("iron_condor_", "iron_butterfly_")):
        width = max(
            max(leg.strike for leg in package.legs if leg.right == right)
            - min(leg.strike for leg in package.legs if leg.right == right)
            for right in {"PUT", "CALL"}
        )
    else:
        width = max(strikes) - min(strikes)
    return OptionPackageRisk(
        structure=structure,
        right=right,
        expiry=next(iter(expiries)),
        width=float(width),
        debit_value=package.debit_value,
        multiplier=package.product.multiplier,
        quantity=package.quantity,
        max_loss=float(max_loss),
        max_profit=float(max_profit) if max_profit is not None else None,
    )


def _option_package_structure(package: OptionPackage) -> str:
    legs = package.legs
    credit_or_debit = "credit" if package.debit_value < 0 else "debit"
    rights = {leg.right for leg in legs}
    actions_by_right = {
        right: {leg.action for leg in legs if leg.right == right}
        for right in rights
    }
    if (
        len(legs) == 2
        and len(rights) == 1
        and next(iter(actions_by_right.values())) == {"BUY", "SELL"}
        and legs[0].ratio == legs[1].ratio
    ):
        return f"vertical_{credit_or_debit}"
    if (
        len(legs) == 3
        and len(rights) == 1
        and next(iter(actions_by_right.values())) == {"BUY", "SELL"}
    ):
        ordered = sorted(legs, key=lambda leg: leg.strike)
        if (
            ordered[0].action == ordered[2].action
            and ordered[1].action != ordered[0].action
            and ordered[0].ratio == ordered[2].ratio
            and ordered[1].ratio == 2 * ordered[0].ratio
        ):
            return f"butterfly_{credit_or_debit}"
    if (
        len(legs) == 4
        and rights == {"PUT", "CALL"}
        and all(actions == {"BUY", "SELL"} for actions in actions_by_right.values())
        and len({leg.ratio for leg in legs}) == 1
    ):
        middle = sorted(leg.strike for leg in legs)[1:3]
        name = "iron_butterfly" if math.isclose(*middle) else "iron_condor"
        return f"{name}_{credit_or_debit}"
    return "defined_risk_combo"
