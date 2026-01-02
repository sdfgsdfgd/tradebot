"""Data models for synthetic backtests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OptionLeg:
    action: str
    right: str
    strike: float
    qty: int


@dataclass
class OptionTrade:
    symbol: str
    legs: list[OptionLeg]
    entry_time: datetime
    expiry: date
    entry_credit: float
    stop_loss: float
    profit_target: float
    exit_time: datetime | None = None
    exit_debit: float | None = None
    exit_reason: str | None = None

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl(self, multiplier: float) -> float:
        if self.exit_debit is None:
            return 0.0
        return (self.entry_credit - self.exit_debit) * multiplier


@dataclass(frozen=True)
class EquityPoint:
    ts: datetime
    equity: float


@dataclass(frozen=True)
class SummaryStats:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    avg_hold_hours: float
