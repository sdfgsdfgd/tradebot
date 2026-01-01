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
                "right",
                "entry_time",
                "exit_time",
                "expiry",
                "short_strike",
                "long_strike",
                "qty",
                "entry_credit",
                "exit_debit",
                "exit_reason",
            ]
        )
        for trade in result.trades:
            writer.writerow(
                [
                    trade.symbol,
                    trade.right,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else "",
                    trade.expiry.isoformat(),
                    f"{trade.short_strike:.2f}",
                    f"{trade.long_strike:.2f}",
                    trade.qty,
                    f"{trade.entry_credit:.4f}",
                    f"{trade.exit_debit:.4f}" if trade.exit_debit is not None else "",
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
