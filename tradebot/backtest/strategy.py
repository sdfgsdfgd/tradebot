"""Strategy definitions for synthetic backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import StrategyConfig, LegConfig
from .models import OptionLeg


@dataclass(frozen=True)
class TradeSpec:
    expiry: date
    legs: list[OptionLeg]


class CreditSpreadStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self._cfg = cfg

    def should_enter(self, ts: datetime) -> bool:
        return ts.weekday() in self._cfg.entry_days

    def build_spec(self, ts: datetime, spot: float, right_override: str | None = None) -> TradeSpec:
        expiry = _expiry_from_dte(ts.date(), self._cfg.dte)
        if self._cfg.legs:
            legs = _build_legs(self._cfg.legs, spot, self._cfg.quantity)
        else:
            legs = _build_default_legs(self._cfg, spot, right_override)
        return TradeSpec(expiry=expiry, legs=legs)


def _expiry_from_dte(anchor: date, dte: int) -> date:
    current = anchor
    days = max(dte, 0)
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current


def _build_legs(legs: tuple[LegConfig, ...], spot: float, quantity: int) -> list[OptionLeg]:
    built: list[OptionLeg] = []
    for leg in legs:
        strike = _strike_from_moneyness(spot, leg.right, leg.moneyness_pct)
        built.append(
            OptionLeg(
                action=leg.action,
                right=leg.right,
                strike=strike,
                qty=leg.qty * quantity,
            )
        )
    return built


def _build_default_legs(cfg: StrategyConfig, spot: float, right_override: str | None) -> list[OptionLeg]:
    width = spot * (cfg.width_pct / 100.0)
    right = (right_override or cfg.right).upper()
    # Negative otm_pct means ITM (e.g., -1 = 1% ITM).
    short_strike = _strike_from_moneyness(spot, right, cfg.otm_pct)
    if right == "PUT":
        long_strike = short_strike - width
    else:
        long_strike = short_strike + width
    return [
        OptionLeg(action="SELL", right=right, strike=short_strike, qty=cfg.quantity),
        OptionLeg(action="BUY", right=right, strike=long_strike, qty=cfg.quantity),
    ]


def _strike_from_moneyness(spot: float, right: str, moneyness_pct: float) -> float:
    if right == "PUT":
        return spot * (1 - moneyness_pct / 100.0)
    return spot * (1 + moneyness_pct / 100.0)
