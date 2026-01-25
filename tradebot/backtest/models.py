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
    entry_price: float
    stop_loss: float
    profit_target: float
    margin_required: float = 0.0
    max_loss: float | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl(self, multiplier: float) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.entry_price - self.exit_price) * multiplier


@dataclass
class SpotTrade:
    symbol: str
    qty: int
    entry_time: datetime
    entry_price: float
    base_profit_target_pct: float | None = None
    base_stop_loss_pct: float | None = None
    profit_target_pct: float | None = None
    stop_loss_pct: float | None = None
    profit_target_price: float | None = None
    stop_loss_price: float | None = None
    margin_required: float = 0.0
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl(self, multiplier: float) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty * multiplier


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
    roi: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_hold_hours: float
