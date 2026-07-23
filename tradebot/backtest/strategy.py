"""Strategy definitions for synthetic backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import StrategyConfig, LegConfig
from ..engine import _trade_date, _trade_weekday
from ..option_package import ResolvedOptionLeg, option_package_entry_intent


@dataclass(frozen=True)
class TradeSpec:
    expiry: date
    legs: tuple[ResolvedOptionLeg, ...]
    quantity: int


class OptionPackageStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self._cfg = cfg

    def should_enter(self, ts: datetime) -> bool:
        return _trade_weekday(ts) in self._cfg.entry_days

    def build_spec(
        self,
        ts: datetime,
        spot: float,
        *,
        right_override: str | None = None,
        legs_override: tuple[LegConfig, ...] | None = None,
    ) -> TradeSpec:
        if legs_override:
            selected_legs = legs_override
            legs_path = "legs_override"
        elif self._cfg.legs:
            selected_legs = self._cfg.legs
            legs_path = "legs"
        else:
            right = (right_override or self._cfg.right).upper()
            selected_legs = (
                LegConfig(
                    action="SELL",
                    right=right,
                    moneyness_pct=self._cfg.otm_pct,
                    qty=1,
                ),
                LegConfig(
                    action="BUY",
                    right=right,
                    moneyness_pct=self._cfg.otm_pct + self._cfg.width_pct,
                    qty=1,
                ),
            )
            legs_path = "default_legs"

        entry_intent = option_package_entry_intent(
            self._cfg,
            legs=selected_legs,
            path=legs_path,
        )
        expiry = _expiry_from_dte(_trade_date(ts), entry_intent.dte)
        legs = _build_legs(
            entry_intent.legs,
            spot,
            expiry,
        )
        return TradeSpec(
            expiry=expiry,
            legs=legs,
            quantity=entry_intent.quantity,
        )


def _expiry_from_dte(anchor: date, dte: int) -> date:
    current = anchor
    days = max(dte, 0)
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current


def _build_legs(
    legs: tuple[LegConfig, ...],
    spot: float,
    expiry: date,
) -> tuple[ResolvedOptionLeg, ...]:
    built: list[ResolvedOptionLeg] = []
    for leg in legs:
        strike = _strike_from_moneyness(spot, leg.right, leg.moneyness_pct)
        built.append(
            ResolvedOptionLeg(
                action=leg.action,
                right=leg.right,
                strike=strike,
                ratio=leg.qty,
                expiry=expiry.strftime("%Y%m%d"),
            )
        )
    return tuple(built)



def _strike_from_moneyness(spot: float, right: str, moneyness_pct: float) -> float:
    if right == "PUT":
        return spot * (1 - moneyness_pct / 100.0)
    return spot * (1 + moneyness_pct / 100.0)
