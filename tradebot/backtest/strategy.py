"""Strategy definitions for synthetic backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import StrategyConfig


@dataclass(frozen=True)
class SpreadSpec:
    right: str
    short_strike: float
    long_strike: float
    expiry: date
    qty: int


class CreditSpreadStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self._cfg = cfg

    def should_enter(self, ts: datetime) -> bool:
        return ts.weekday() in self._cfg.entry_days

    def build_spec(self, ts: datetime, spot: float) -> SpreadSpec:
        expiry = _expiry_from_dte(ts.date(), self._cfg.dte)
        width = spot * (self._cfg.width_pct / 100.0)
        if self._cfg.right == "PUT":
            # Negative otm_pct means ITM (e.g., -1 = 1% ITM).
            short_strike = spot * (1 - self._cfg.otm_pct / 100.0)
            long_strike = short_strike - width
        else:
            # Negative otm_pct means ITM (e.g., -1 = 1% ITM).
            short_strike = spot * (1 + self._cfg.otm_pct / 100.0)
            long_strike = short_strike + width
        return SpreadSpec(
            right=self._cfg.right,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            qty=self._cfg.quantity,
        )


def _expiry_from_dte(anchor: date, dte: int) -> date:
    current = anchor
    days = max(dte, 0)
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current
