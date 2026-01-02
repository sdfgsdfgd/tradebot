"""Reporting helpers for backtests."""
from __future__ import annotations

import csv
from pathlib import Path

from .engine import BacktestResult


def write_reports(result: BacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_trades(output_dir / "trades.csv", result)
    _write_equity(output_dir / "equity_curve.csv", result)
    _write_summary(output_dir / "summary.txt", result)


def print_summary(result: BacktestResult) -> None:
    summary = result.summary
    print("Backtest Summary")
    print(f"Trades: {summary.trades}")
    print(f"Win rate: {summary.win_rate:.1%}")
    print(f"Total PnL: {summary.total_pnl:,.2f}")
    print(f"Avg win: {summary.avg_win:,.2f}")
    print(f"Avg loss: {summary.avg_loss:,.2f}")
    print(f"Max drawdown: {summary.max_drawdown:,.2f}")
    print(f"Avg hold (hours): {summary.avg_hold_hours:.2f}")


def _write_trades(path: Path, result: BacktestResult) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol",
                "entry_time",
                "exit_time",
                "expiry",
                "legs",
                "entry_price",
                "exit_price",
                "exit_reason",
            ]
        )
        for trade in result.trades:
            legs = "; ".join(
                f"{leg.action} {leg.right} {leg.strike:.2f} x{leg.qty}" for leg in trade.legs
            )
            writer.writerow(
                [
                    trade.symbol,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else "",
                    trade.expiry.isoformat(),
                    legs,
                    f"{trade.entry_price:.4f}",
                    f"{trade.exit_price:.4f}" if trade.exit_price is not None else "",
                    trade.exit_reason or "",
                ]
            )


def _write_equity(path: Path, result: BacktestResult) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "equity"])
        for point in result.equity:
            writer.writerow([point.ts.isoformat(), f"{point.equity:.2f}"])


def _write_summary(path: Path, result: BacktestResult) -> None:
    summary = result.summary
    lines = [
        "Backtest Summary",
        f"Trades: {summary.trades}",
        f"Win rate: {summary.win_rate:.1%}",
        f"Total PnL: {summary.total_pnl:,.2f}",
        f"Avg win: {summary.avg_win:,.2f}",
        f"Avg loss: {summary.avg_loss:,.2f}",
        f"Max drawdown: {summary.max_drawdown:,.2f}",
        f"Avg hold (hours): {summary.avg_hold_hours:.2f}",
    ]
    path.write_text("\n".join(lines))
