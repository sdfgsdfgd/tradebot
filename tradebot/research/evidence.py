"""Cost-adjusted research evidence shared by backtests and evaluators."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from ..backtest.models import BacktestResult


SCORE_VERSION = "research.daily.v1"


def backtest_evidence(
    result: BacktestResult,
    *,
    starting_cash: float,
    multiplier: float,
) -> dict[str, float | int | bool | None | str]:
    """Summarize causal daily equity and closed-trade outcomes without promotion claims."""

    trade_pnls = [float(trade.pnl(multiplier)) for trade in result.trades]
    end_of_day: dict[object, float] = {}
    for point in result.equity:
        end_of_day[point.ts.date()] = float(point.equity)

    previous = float(starting_cash)
    daily_pnls: list[float] = []
    for day in sorted(end_of_day):
        equity = end_of_day[day]
        daily_pnls.append(equity - previous)
        previous = equity

    sessions = len(daily_pnls)
    mean_daily = statistics.fmean(daily_pnls) if daily_pnls else 0.0
    daily_std = statistics.stdev(daily_pnls) if sessions > 1 else 0.0
    # Two-sided 95% normal bound is deliberately more conservative than a
    # one-sided discovery bound. Walk-forward/bootstrap gates remain separate.
    daily_lcb95 = (
        mean_daily - 1.96 * daily_std / math.sqrt(sessions)
        if sessions > 1
        else mean_daily
    )
    tail_count = max(1, math.ceil(sessions * 0.05)) if sessions else 0
    daily_cvar95 = (
        statistics.fmean(sorted(daily_pnls)[:tail_count])
        if tail_count
        else 0.0
    )

    wins = [pnl for pnl in trade_pnls if pnl > 0.0]
    losses = [pnl for pnl in trade_pnls if pnl < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    max_drawdown = float(result.summary.max_drawdown)
    total_pnl = float(result.summary.total_pnl)

    return {
        "version": SCORE_VERSION,
        "sessions": sessions,
        "active_sessions": sum(pnl != 0.0 for pnl in daily_pnls),
        "mean_daily_pnl": mean_daily,
        "daily_pnl_std": daily_std,
        "daily_pnl_lcb95": daily_lcb95,
        "daily_cvar95": daily_cvar95,
        "worst_daily_pnl": min(daily_pnls, default=0.0),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "payoff_ratio": (
            statistics.fmean(wins) / abs(statistics.fmean(losses))
            if wins and losses
            else None
        ),
        "pnl_over_max_drawdown": (
            total_pnl / max_drawdown if max_drawdown > 0.0 else None
        ),
        "top_5_win_share": (
            sum(sorted(wins, reverse=True)[:5]) / gross_profit
            if gross_profit > 0.0
            else None
        ),
        "sample_gate": sessions >= 60 and len(trade_pnls) >= 30,
        "positive_lcb": daily_lcb95 > 0.0,
    }


def research_rank_key(row: dict) -> tuple:
    """Exploration ordering only; walk-forward and authentic evidence promote."""

    metrics = row.get("metrics") or {}
    evidence = row.get("evidence") or {}

    def number(value: object, default: float = 0.0) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    lcb = number(evidence.get("daily_pnl_lcb95"), float("-inf"))
    pnl_dd = number(evidence.get("pnl_over_max_drawdown"), float("-inf"))
    profit_factor = min(number(evidence.get("profit_factor")), 10.0)
    concentration = number(evidence.get("top_5_win_share"), 1.0)
    return (
        bool(evidence.get("sample_gate")) and bool(evidence.get("positive_lcb")),
        lcb,
        pnl_dd,
        profit_factor,
        -concentration,
        number(metrics.get("pnl"), float("-inf")),
        int(number(metrics.get("trades"))),
    )

