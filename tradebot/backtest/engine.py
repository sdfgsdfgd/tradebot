"""Backtest runner for synthetic options strategies."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time

from .config import ConfigBundle
from .data import IBKRHistoricalData
from .models import Bar, EquityPoint, SpreadTrade, SummaryStats
from .strategy import CreditSpreadStrategy
from .synth import IVSurfaceParams, black_76, black_scholes, ewma_vol, iv_atm, iv_for_strike, mid_edge_quote


@dataclass(frozen=True)
class BacktestResult:
    trades: list[SpreadTrade]
    equity: list[EquityPoint]
    summary: SummaryStats


def run_backtest(cfg: ConfigBundle) -> BacktestResult:
    data = IBKRHistoricalData()
    start_dt = datetime.combine(cfg.backtest.start, time(0, 0))
    end_dt = datetime.combine(cfg.backtest.end, time(23, 59))
    bars = data.load_or_fetch_bars(
        symbol=cfg.strategy.symbol,
        exchange=cfg.strategy.exchange,
        start=start_dt,
        end=end_dt,
        bar_size=cfg.backtest.bar_size,
        use_rth=cfg.backtest.use_rth,
        cache_dir=cfg.backtest.cache_dir,
    )
    if not bars:
        raise RuntimeError("No bars loaded for backtest")
    contract, meta = data.resolve_contract(cfg.strategy.symbol, cfg.strategy.exchange)
    is_future = meta.exchange != "SMART"

    surface_params = IVSurfaceParams(
        rv_lookback=cfg.synthetic.rv_lookback,
        rv_ewma_lambda=cfg.synthetic.rv_ewma_lambda,
        iv_risk_premium=cfg.synthetic.iv_risk_premium,
        iv_floor=cfg.synthetic.iv_floor,
        term_slope=cfg.synthetic.term_slope,
        skew=cfg.synthetic.skew,
    )

    strategy = CreditSpreadStrategy(cfg.strategy)
    returns = deque(maxlen=surface_params.rv_lookback)
    equity = cfg.backtest.starting_cash
    equity_curve: list[EquityPoint] = []
    trades: list[SpreadTrade] = []
    open_trade: SpreadTrade | None = None
    prev_bar: Bar | None = None
    last_entry_date = None

    for bar in bars:
        if prev_bar is not None:
            if prev_bar.close > 0:
                returns.append(math.log(bar.close / prev_bar.close))
        rv = ewma_vol(returns, surface_params.rv_ewma_lambda)
        rv *= math.sqrt(_annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))

        if open_trade and prev_bar and bar.ts.date() > open_trade.expiry:
            exit_debit = _spread_value_from_spec(
                open_trade, prev_bar, rv, cfg, surface_params, meta.min_tick, is_future, mode="exit"
            )
            _close_trade(open_trade, prev_bar.ts, exit_debit, "expiry", trades)
            equity += open_trade.pnl(meta.multiplier)
            open_trade = None

        if open_trade:
            current_value = _spread_value(open_trade, bar, rv, cfg, surface_params, meta.min_tick, is_future)
            if _hit_profit(open_trade, current_value):
                exit_debit = _spread_value_from_spec(
                    open_trade, bar, rv, cfg, surface_params, meta.min_tick, is_future, mode="exit"
                )
                _close_trade(open_trade, bar.ts, exit_debit, "profit", trades)
                equity += open_trade.pnl(meta.multiplier)
                open_trade = None
            elif _hit_stop(open_trade, current_value, cfg.strategy.stop_loss_basis):
                exit_debit = _spread_value_from_spec(
                    open_trade, bar, rv, cfg, surface_params, meta.min_tick, is_future, mode="exit"
                )
                _close_trade(open_trade, bar.ts, exit_debit, "stop", trades)
                equity += open_trade.pnl(meta.multiplier)
                open_trade = None

        # TODO: add regime gating hook before entry decisions.
        if open_trade is None and strategy.should_enter(bar.ts):
            if last_entry_date != bar.ts.date():
                spec = strategy.build_spec(bar.ts, bar.close)
                entry_credit = _spread_value_from_spec(
                    spec, bar, rv, cfg, surface_params, meta.min_tick, is_future, mode="entry"
                )
                open_trade = SpreadTrade(
                    symbol=cfg.strategy.symbol,
                    right=spec.right,
                    entry_time=bar.ts,
                    expiry=spec.expiry,
                    short_strike=spec.short_strike,
                    long_strike=spec.long_strike,
                    qty=spec.qty,
                    entry_credit=entry_credit,
                    stop_loss=cfg.strategy.stop_loss,
                    profit_target=cfg.strategy.profit_target,
                )
                last_entry_date = bar.ts.date()

        unrealized = 0.0
        if open_trade:
            mark_value = _spread_value(open_trade, bar, rv, cfg, surface_params, meta.min_tick, is_future)
            unrealized = (open_trade.entry_credit - mark_value) * meta.multiplier * open_trade.qty
        equity_curve.append(EquityPoint(ts=bar.ts, equity=equity + unrealized))
        prev_bar = bar

    if open_trade and prev_bar:
        exit_debit = _spread_value_from_spec(
            open_trade, prev_bar, rv, cfg, surface_params, meta.min_tick, is_future, mode="exit"
        )
        _close_trade(open_trade, prev_bar.ts, exit_debit, "end", trades)
        equity += open_trade.pnl(meta.multiplier)

    summary = _summarize(trades, cfg.backtest.starting_cash, equity_curve, meta.multiplier)
    data.disconnect()
    return BacktestResult(trades=trades, equity=equity_curve, summary=summary)


def _spread_value(
    trade: SpreadTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
) -> float:
    spec = trade
    return _spread_value_from_spec(spec, bar, rv, cfg, surface_params, min_tick, is_future, mode="mark")


def _spread_value_from_spec(
    spec,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
    mode: str,
) -> float:
    dte_days = max((spec.expiry - bar.ts.date()).days, 0)
    atm_iv = iv_atm(rv, dte_days, surface_params)
    forward = bar.close
    t = max(dte_days / 365.0, 1e-6)
    short_iv = iv_for_strike(atm_iv, forward, spec.short_strike, surface_params)
    long_iv = iv_for_strike(atm_iv, forward, spec.long_strike, surface_params)
    if is_future:
        short_mid = black_76(forward, spec.short_strike, t, cfg.backtest.risk_free_rate, short_iv, spec.right)
        long_mid = black_76(forward, spec.long_strike, t, cfg.backtest.risk_free_rate, long_iv, spec.right)
    else:
        short_mid = black_scholes(forward, spec.short_strike, t, cfg.backtest.risk_free_rate, short_iv, spec.right)
        long_mid = black_scholes(forward, spec.long_strike, t, cfg.backtest.risk_free_rate, long_iv, spec.right)
    short_quote = mid_edge_quote(short_mid, cfg.synthetic.min_spread_pct, min_tick)
    long_quote = mid_edge_quote(long_mid, cfg.synthetic.min_spread_pct, min_tick)
    if mode == "entry":
        return max(0.0, short_quote.bid - long_quote.ask)
    if mode == "exit":
        return max(0.0, short_quote.ask - long_quote.bid)
    return max(0.0, short_quote.mid - long_quote.mid)


def _hit_profit(trade: SpreadTrade, current_value: float) -> bool:
    target = trade.entry_credit * trade.profit_target
    return (trade.entry_credit - current_value) >= target


def _hit_stop(trade: SpreadTrade, current_value: float, basis: str) -> bool:
    if basis == "credit":
        return current_value >= trade.entry_credit * (1 + trade.stop_loss)
    max_loss = max(trade.long_strike - trade.short_strike, trade.short_strike - trade.long_strike) - trade.entry_credit
    loss = max(0.0, current_value - trade.entry_credit)
    return loss >= max_loss * trade.stop_loss


def _close_trade(trade: SpreadTrade, ts: datetime, debit: float, reason: str, trades: list[SpreadTrade]) -> None:
    trade.exit_time = ts
    trade.exit_debit = debit
    trade.exit_reason = reason
    trades.append(trade)


def _summarize(
    trades: list[SpreadTrade],
    starting_cash: float,
    equity_curve: list[EquityPoint],
    multiplier: float,
) -> SummaryStats:
    wins = 0
    losses = 0
    total_pnl = 0.0
    win_pnls: list[float] = []
    loss_pnls: list[float] = []
    hold_hours: list[float] = []
    peak = starting_cash
    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0].equity
    for point in equity_curve:
        if point.equity > peak:
            peak = point.equity
        dd = peak - point.equity
        if dd > max_dd:
            max_dd = dd
    for trade in trades:
        pnl = trade.pnl(multiplier)
        total_pnl += pnl
        if pnl >= 0:
            wins += 1
            win_pnls.append(pnl)
        else:
            losses += 1
            loss_pnls.append(pnl)
        if trade.exit_time:
            hold_hours.append((trade.exit_time - trade.entry_time).total_seconds() / 3600.0)
    total = wins + losses
    win_rate = wins / total if total else 0.0
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0
    return SummaryStats(
        trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        avg_hold_hours=avg_hold,
    )


def _annualization_factor(bar_size: str, use_rth: bool) -> float:
    label = bar_size.lower()
    if \"hour\" in label:
        return 252 * (6.5 if use_rth else 24)
    if \"day\" in label:
        return 252
    return 252 * (6.5 if use_rth else 24)
