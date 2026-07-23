"""Canonical backtest records and outcome aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date

from ..option_package import OptionPackage, OptionPackageRisk, ResolvedOptionLeg


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OptionTrade:
    package: OptionPackage
    risk: OptionPackageRisk
    entry_time: datetime
    stop_loss: float
    profit_target: float
    margin_required: float = 0.0
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    @property
    def symbol(self) -> str:
        return self.package.product.underlying_symbol

    @property
    def legs(self) -> tuple[ResolvedOptionLeg, ...]:
        return self.package.legs

    @property
    def expiry(self) -> date:
        raw = self.package.legs[0].expiry
        return (
            date.fromisoformat(raw)
            if "-" in raw
            else datetime.strptime(raw[:8], "%Y%m%d").date()
        )

    @property
    def entry_price(self) -> float:
        """Signed package credit units: credit positive, debit negative."""
        return -self.package.debit_value

    @property
    def max_loss(self) -> float:
        scale = self.package.product.multiplier * self.package.quantity
        return self.risk.max_loss / scale

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl(self, multiplier: float) -> float:
        if self.exit_price is None:
            return 0.0
        return (
            (self.entry_price - self.exit_price)
            * self.package.product.multiplier
            * self.package.quantity
        )


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
    entry_branch: str | None = None
    decision_trace: dict[str, object] | None = None
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


@dataclass(frozen=True)
class BacktestResult:
    trades: list[OptionTrade | SpotTrade]
    equity: list[EquityPoint]
    summary: SummaryStats
    lifecycle_trace: list[dict[str, object]] | None = None


def summarize_with_max_drawdown(
    trades: list[OptionTrade | SpotTrade],
    *,
    starting_cash: float,
    max_drawdown: float,
    multiplier: float,
) -> SummaryStats:
    pnls = [trade.pnl(multiplier) for trade in trades]
    wins = [pnl for pnl in pnls if pnl >= 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    hold_hours = [
        (trade.exit_time - trade.entry_time).total_seconds() / 3600.0
        for trade in trades
        if trade.exit_time is not None
    ]
    total_pnl = sum(pnls)
    total = len(pnls)
    return SummaryStats(
        trades=total,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / total if total else 0.0,
        total_pnl=total_pnl,
        roi=total_pnl / starting_cash if starting_cash > 0 else 0.0,
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown / starting_cash if starting_cash > 0 else 0.0,
        avg_hold_hours=sum(hold_hours) / len(hold_hours) if hold_hours else 0.0,
    )


def summarize(
    trades: list[OptionTrade | SpotTrade],
    starting_cash: float,
    equity_curve: list[EquityPoint],
    multiplier: float,
) -> SummaryStats:
    peak = equity_curve[0].equity if equity_curve else starting_cash
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        max_drawdown = max(max_drawdown, peak - point.equity)
    return summarize_with_max_drawdown(
        trades,
        starting_cash=starting_cash,
        max_drawdown=max_drawdown,
        multiplier=multiplier,
    )
