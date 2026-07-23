"""Strategy definitions for synthetic backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

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
        expiry = entry_intent.target_expiry(_trade_date(ts))
        return TradeSpec(
            expiry=expiry,
            legs=entry_intent.resolved_legs(spot=spot, expiry=expiry),
            quantity=entry_intent.quantity,
        )
