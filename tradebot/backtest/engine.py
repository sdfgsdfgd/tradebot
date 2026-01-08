"""Backtest runner for synthetic options strategies."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .config import ConfigBundle, SpotLegConfig
from .calibration import ensure_calibration, load_calibration
from .data import IBKRHistoricalData, ContractMeta
from .models import Bar, EquityPoint, OptionLeg, OptionTrade, SpotTrade, SummaryStats
from .strategy import CreditSpreadStrategy, TradeSpec
from .synth import IVSurfaceParams, black_76, black_scholes, ewma_vol, iv_atm, iv_for_strike, mid_edge_quote
from ..signals import (
    ema_next as _ema_next_shared,
    ema_periods as _ema_periods_shared,
    flip_exit_mode as _flip_exit_mode_shared,
)


@dataclass(frozen=True)
class BacktestResult:
    trades: list[OptionTrade | SpotTrade]
    equity: list[EquityPoint]
    summary: SummaryStats


def run_backtest(cfg: ConfigBundle) -> BacktestResult:
    data = IBKRHistoricalData()
    start_dt = datetime.combine(cfg.backtest.start, time(0, 0))
    end_dt = datetime.combine(cfg.backtest.end, time(23, 59))
    if cfg.backtest.offline:
        bars = data.load_cached_bars(
            symbol=cfg.strategy.symbol,
            exchange=cfg.strategy.exchange,
            start=start_dt,
            end=end_dt,
            bar_size=cfg.backtest.bar_size,
            use_rth=cfg.backtest.use_rth,
            cache_dir=cfg.backtest.cache_dir,
        )
    else:
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
    is_future = cfg.strategy.symbol in ("MNQ", "MBT")
    if cfg.backtest.offline:
        exchange = "CME" if is_future else "SMART"
        if cfg.strategy.instrument == "spot":
            multiplier = _spot_multiplier(cfg.strategy.symbol, is_future)
        else:
            multiplier = 1.0 if is_future else 100.0
        meta = ContractMeta(symbol=cfg.strategy.symbol, exchange=exchange, multiplier=multiplier, min_tick=0.01)
    else:
        _, meta = data.resolve_contract(cfg.strategy.symbol, cfg.strategy.exchange)
        if cfg.strategy.instrument == "spot":
            meta = ContractMeta(
                symbol=meta.symbol,
                exchange=meta.exchange,
                multiplier=_spot_multiplier(cfg.strategy.symbol, is_future, default=meta.multiplier),
                min_tick=meta.min_tick,
            )
        elif not is_future and meta.exchange == "SMART":
            meta = ContractMeta(
                symbol=meta.symbol,
                exchange=meta.exchange,
                multiplier=100.0,
                min_tick=meta.min_tick,
            )

    if cfg.strategy.instrument == "spot":
        result = _run_spot_backtest(cfg, bars, meta)
        data.disconnect()
        return result

    surface_params = IVSurfaceParams(
        rv_lookback=cfg.synthetic.rv_lookback,
        rv_ewma_lambda=cfg.synthetic.rv_ewma_lambda,
        iv_risk_premium=cfg.synthetic.iv_risk_premium,
        iv_floor=cfg.synthetic.iv_floor,
        term_slope=cfg.synthetic.term_slope,
        skew=cfg.synthetic.skew,
    )
    calibration = None
    if cfg.backtest.calibrate:
        calibration = ensure_calibration(cfg, rv_override=_rv_from_bars(bars, cfg))
    else:
        calibration = load_calibration(cfg.backtest.calibration_dir, cfg.strategy.symbol)

    strategy = CreditSpreadStrategy(cfg.strategy)
    returns = deque(maxlen=surface_params.rv_lookback)
    cash = cfg.backtest.starting_cash
    margin_used = 0.0
    equity_curve: list[EquityPoint] = []
    trades: list[OptionTrade] = []
    open_trades: list[OptionTrade] = []
    prev_bar: Bar | None = None
    entries_today = 0
    filters = cfg.strategy.filters
    ema_periods = _ema_periods(cfg.strategy.ema_preset)
    needs_direction = cfg.strategy.directional_legs is not None
    if needs_direction and cfg.strategy.direction_source == "ema" and ema_periods is None:
        raise ValueError("directional_legs requires ema_preset when direction_source='ema'")
    ema_needed = ema_periods is not None or (
        filters
        and (filters.ema_spread_min_pct is not None or filters.ema_slope_min_pct is not None)
    )
    if needs_direction:
        ema_needed = True
    ema_fast = None
    ema_slow = None
    prev_ema_fast = None
    prev_ema_slow = None
    ema_count = 0
    bars_in_day = 0
    last_entry_idx = None
    last_date = None
    for idx, bar in enumerate(bars):
        next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
        is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
        if prev_bar is not None:
            if prev_bar.close > 0:
                returns.append(math.log(bar.close / prev_bar.close))
        rv = ewma_vol(returns, surface_params.rv_ewma_lambda)
        rv *= math.sqrt(_annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
        if last_date != bar.ts.date():
            bars_in_day = 0
            last_date = bar.ts.date()
            entries_today = 0
        bars_in_day += 1
        if ema_periods and bar.close > 0:
            prev_ema_fast = ema_fast
            prev_ema_slow = ema_slow
            ema_fast = _ema_next(ema_fast, bar.close, ema_periods[0])
            ema_slow = _ema_next(ema_slow, bar.close, ema_periods[1])
            ema_count += 1
        ema_ready = (
            ema_periods is not None
            and ema_count >= ema_periods[1]
            and ema_fast is not None
            and ema_slow is not None
        )

        if open_trades and prev_bar:
            still_open: list[OptionTrade] = []
            for trade in open_trades:
                if bar.ts.date() > trade.expiry:
                    exit_debit = _trade_value_from_spec(
                        trade,
                        prev_bar,
                        rv,
                        cfg,
                        surface_params,
                        meta.min_tick,
                        is_future,
                        mode="exit",
                        calibration=calibration,
                    )
                    _close_trade(trade, prev_bar.ts, exit_debit, "expiry", trades)
                    cash += (-exit_debit) * meta.multiplier
                    margin_used = max(0.0, margin_used - trade.margin_required)
                else:
                    still_open.append(trade)
            open_trades = still_open

        liquidation = 0.0
        if open_trades:
            exit_mode = _flip_exit_mode(cfg)
            cross_up = False
            cross_down = False
            if (
                cfg.strategy.exit_on_signal_flip
                and ema_ready
                and exit_mode == "cross"
                and prev_ema_fast is not None
                and prev_ema_slow is not None
            ):
                cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

            still_open = []
            for trade in open_trades:
                current_value = _trade_value(
                    trade, bar, rv, cfg, surface_params, meta.min_tick, is_future, calibration
                )
                should_close = False
                reason = ""

                if _hit_profit(trade, current_value):
                    should_close = True
                    reason = "profit"
                elif _hit_stop(trade, current_value, cfg.strategy.stop_loss_basis, bar.close):
                    should_close = True
                    reason = "stop"
                elif _hit_exit_dte(cfg, trade, bar.ts.date()):
                    should_close = True
                    reason = "exit_dte"
                elif _hit_flip_exit(
                    cfg,
                    trade,
                    bar,
                    current_value,
                    ema_ready,
                    ema_fast,
                    ema_slow,
                    cross_up,
                    cross_down,
                ):
                    should_close = True
                    reason = "flip"
                elif cfg.strategy.dte == 0 and is_last_bar:
                    should_close = True
                    reason = "eod"

                if should_close:
                    exit_debit = _trade_value_from_spec(
                        trade,
                        bar,
                        rv,
                        cfg,
                        surface_params,
                        meta.min_tick,
                        is_future,
                        mode="exit",
                        calibration=calibration,
                    )
                    _close_trade(trade, bar.ts, exit_debit, reason, trades)
                    cash += (-exit_debit) * meta.multiplier
                    margin_used = max(0.0, margin_used - trade.margin_required)
                else:
                    still_open.append(trade)
                    liquidation += (-current_value) * meta.multiplier
            open_trades = still_open

        # TODO: add regime gating hook before entry decisions.
        ema_gate_ok = True
        ema_right_override = None
        direction = None
        if ema_needed and not ema_ready:
            ema_gate_ok = False
        elif ema_ready:
            if needs_direction:
                if cfg.strategy.direction_source != "ema":
                    ema_gate_ok = False
                elif cfg.strategy.ema_entry_mode == "cross":
                    if prev_ema_fast is None or prev_ema_slow is None:
                        ema_gate_ok = False
                    else:
                        cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                        cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
                        if cross_up:
                            direction = "up"
                        elif cross_down:
                            direction = "down"
                        else:
                            direction = None
                        ema_gate_ok = (
                            direction is not None
                            and cfg.strategy.directional_legs is not None
                            and direction in cfg.strategy.directional_legs
                        )
                else:
                    if ema_fast > ema_slow:
                        direction = "up"
                    elif ema_fast < ema_slow:
                        direction = "down"
                    else:
                        direction = None
                    ema_gate_ok = (
                        direction is not None
                        and cfg.strategy.directional_legs is not None
                        and direction in cfg.strategy.directional_legs
                    )
            elif cfg.strategy.ema_entry_mode == "cross":
                if prev_ema_fast is None or prev_ema_slow is None:
                    ema_gate_ok = False
                else:
                    cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                    cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
                    bias = _ema_bias(cfg)
                    if bias == "up":
                        ema_gate_ok = cross_up
                    elif bias == "down":
                        ema_gate_ok = cross_down
                    else:
                        ema_gate_ok = cross_up or cross_down
                    if cfg.strategy.ema_directional:
                        if cross_up:
                            ema_right_override = "CALL"
                        elif cross_down:
                            ema_right_override = "PUT"
            else:
                if cfg.strategy.ema_directional:
                    if ema_fast > ema_slow:
                        ema_right_override = "CALL"
                    elif ema_fast < ema_slow:
                        ema_right_override = "PUT"
                    else:
                        ema_gate_ok = False
                else:
                    ema_gate_ok = ema_fast > ema_slow

        filters_ok = True
        if filters:
            if filters.rv_min is not None and rv < filters.rv_min:
                filters_ok = False
            if filters.rv_max is not None and rv > filters.rv_max:
                filters_ok = False
            if filters.entry_start_hour is not None and filters.entry_end_hour is not None:
                hour = bar.ts.hour
                start = filters.entry_start_hour
                end = filters.entry_end_hour
                if start <= end:
                    if not (start <= hour < end):
                        filters_ok = False
                else:
                    if not (hour >= start or hour < end):
                        filters_ok = False
            if filters.skip_first_bars and bars_in_day <= filters.skip_first_bars:
                filters_ok = False
            if filters.cooldown_bars and last_entry_idx is not None:
                if (idx - last_entry_idx) < filters.cooldown_bars:
                    filters_ok = False
            if filters.ema_spread_min_pct is not None:
                if not ema_ready:
                    filters_ok = False
                else:
                    spread_pct = abs(ema_fast - ema_slow) / max(bar.close, 1e-9) * 100.0
                    if spread_pct < filters.ema_spread_min_pct:
                        filters_ok = False
            if filters.ema_slope_min_pct is not None:
                if not ema_ready or prev_ema_fast is None:
                    filters_ok = False
                else:
                    slope_pct = abs(ema_fast - prev_ema_fast) / max(bar.close, 1e-9) * 100.0
                    if slope_pct < filters.ema_slope_min_pct:
                        filters_ok = False

        open_slots_ok = cfg.strategy.max_open_trades == 0 or len(open_trades) < cfg.strategy.max_open_trades
        entries_ok = cfg.strategy.max_entries_per_day == 0 or entries_today < cfg.strategy.max_entries_per_day
        if open_slots_ok and entries_ok and strategy.should_enter(bar.ts) and ema_gate_ok and filters_ok:
            legs_override = None
            if needs_direction and direction and cfg.strategy.directional_legs:
                legs_override = cfg.strategy.directional_legs.get(direction)
            spec = strategy.build_spec(
                bar.ts,
                bar.close,
                right_override=ema_right_override,
                legs_override=legs_override,
            )
            entry_price = _trade_value_from_spec(
                spec,
                bar,
                rv,
                cfg,
                surface_params,
                meta.min_tick,
                is_future,
                mode="entry",
                calibration=calibration,
            )
            min_credit = cfg.strategy.min_credit if cfg.strategy.min_credit is not None else meta.min_tick
            if entry_price >= 0 and entry_price < min_credit:
                pass
            else:
                mark_price = _trade_value_from_spec(
                    spec,
                    bar,
                    rv,
                    cfg,
                    surface_params,
                    meta.min_tick,
                    is_future,
                    mode="mark",
                    calibration=calibration,
                )
                mark_liquidation = (-mark_price) * meta.multiplier
                candidate = OptionTrade(
                    symbol=cfg.strategy.symbol,
                    legs=spec.legs,
                    entry_time=bar.ts,
                    expiry=spec.expiry,
                    entry_price=entry_price,
                    stop_loss=cfg.strategy.stop_loss,
                    profit_target=cfg.strategy.profit_target,
                )
                candidate.max_loss = _max_loss(candidate)
                if candidate.max_loss is None:
                    candidate.max_loss = _max_loss_estimate(candidate, bar.close)
                candidate.margin_required = _margin_required(candidate, bar.close, meta.multiplier)
                cash_after = cash + (entry_price * meta.multiplier)
                margin_after = margin_used + candidate.margin_required
                equity_after = cash_after + liquidation + mark_liquidation
                if cash_after >= 0 and equity_after >= margin_after:
                    open_trades.append(candidate)
                    cash = cash_after
                    margin_used = margin_after
                    entries_today += 1
                    last_entry_idx = idx
                    liquidation += mark_liquidation
        equity_curve.append(EquityPoint(ts=bar.ts, equity=cash + liquidation))
        prev_bar = bar

    if open_trades and prev_bar:
        for trade in open_trades:
            exit_debit = _trade_value_from_spec(
                trade,
                prev_bar,
                rv,
                cfg,
                surface_params,
                meta.min_tick,
                is_future,
                mode="exit",
                calibration=calibration,
            )
            _close_trade(trade, prev_bar.ts, exit_debit, "end", trades)
            cash += (-exit_debit) * meta.multiplier
            margin_used = max(0.0, margin_used - trade.margin_required)

    summary = _summarize(trades, cfg.backtest.starting_cash, equity_curve, meta.multiplier)
    data.disconnect()
    return BacktestResult(trades=trades, equity=equity_curve, summary=summary)


def _spot_multiplier(symbol: str, is_future: bool, default: float = 1.0) -> float:
    if not is_future:
        return 1.0
    overrides = {
        "MNQ": 2.0,  # Micro E-mini Nasdaq-100
        "MBT": 0.1,  # Micro Bitcoin (0.1 BTC)
    }
    return overrides.get(symbol, default if default > 0 else 1.0)


def _run_spot_backtest(cfg: ConfigBundle, bars: list[Bar], meta: ContractMeta) -> BacktestResult:
    returns = deque(maxlen=cfg.synthetic.rv_lookback)
    cash = cfg.backtest.starting_cash
    margin_used = 0.0
    equity_curve: list[EquityPoint] = []
    trades: list[SpotTrade] = []
    open_trades: list[SpotTrade] = []
    prev_bar: Bar | None = None
    entries_today = 0
    filters = cfg.strategy.filters
    ema_periods = _ema_periods(cfg.strategy.ema_preset)
    needs_direction = cfg.strategy.directional_spot is not None
    if ema_periods is None:
        raise ValueError("spot backtests require ema_preset")
    ema_needed = True

    ema_fast = None
    ema_slow = None
    prev_ema_fast = None
    prev_ema_slow = None
    ema_count = 0
    bars_in_day = 0
    last_entry_idx = None
    last_date = None
    for idx, bar in enumerate(bars):
        next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
        is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
        if prev_bar is not None:
            if prev_bar.close > 0:
                returns.append(math.log(bar.close / prev_bar.close))
        rv = ewma_vol(returns, cfg.synthetic.rv_ewma_lambda)
        rv *= math.sqrt(_annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
        if last_date != bar.ts.date():
            bars_in_day = 0
            last_date = bar.ts.date()
            entries_today = 0
        bars_in_day += 1

        prev_ema_fast = ema_fast
        prev_ema_slow = ema_slow
        if bar.close > 0:
            ema_fast = _ema_next(ema_fast, bar.close, ema_periods[0])
            ema_slow = _ema_next(ema_slow, bar.close, ema_periods[1])
            ema_count += 1
        ema_ready = ema_needed and ema_count >= ema_periods[1] and ema_fast is not None and ema_slow is not None

        liquidation = 0.0
        if open_trades:
            exit_mode = _flip_exit_mode(cfg)
            cross_up = False
            cross_down = False
            if (
                cfg.strategy.exit_on_signal_flip
                and ema_ready
                and exit_mode == "cross"
                and prev_ema_fast is not None
                and prev_ema_slow is not None
            ):
                cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

            still_open = []
            for trade in open_trades:
                current_price = bar.close
                current_value = (trade.qty * current_price) * meta.multiplier
                should_close = False
                reason = ""

                if _spot_hit_profit(trade, current_price):
                    should_close = True
                    reason = "profit"
                elif _spot_hit_stop(trade, current_price):
                    should_close = True
                    reason = "stop"
                elif _spot_hit_flip_exit(
                    cfg,
                    trade,
                    bar,
                    ema_ready,
                    ema_fast,
                    ema_slow,
                    cross_up,
                    cross_down,
                ):
                    should_close = True
                    reason = "flip"
                elif cfg.strategy.spot_close_eod and is_last_bar:
                    should_close = True
                    reason = "eod"

                if should_close:
                    exit_price = bar.close
                    _close_spot_trade(trade, bar.ts, exit_price, reason, trades)
                    cash += (trade.qty * exit_price) * meta.multiplier
                    margin_used = max(0.0, margin_used - trade.margin_required)
                else:
                    still_open.append(trade)
                    liquidation += current_value
            open_trades = still_open

        ema_gate_ok = True
        direction = None
        if ema_needed and not ema_ready:
            ema_gate_ok = False
        elif ema_ready:
            if cfg.strategy.ema_entry_mode == "cross":
                if prev_ema_fast is None or prev_ema_slow is None:
                    ema_gate_ok = False
                else:
                    cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
                    cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow
                    if cross_up:
                        direction = "up"
                    elif cross_down:
                        direction = "down"
                    else:
                        direction = None
            else:
                if ema_fast > ema_slow:
                    direction = "up"
                elif ema_fast < ema_slow:
                    direction = "down"
                else:
                    direction = None

            if needs_direction:
                ema_gate_ok = (
                    direction is not None
                    and cfg.strategy.directional_spot is not None
                    and direction in cfg.strategy.directional_spot
                )
            else:
                ema_gate_ok = direction == "up"

        filters_ok = True
        if filters:
            if filters.rv_min is not None and rv < filters.rv_min:
                filters_ok = False
            if filters.rv_max is not None and rv > filters.rv_max:
                filters_ok = False
            if filters.entry_start_hour is not None and filters.entry_end_hour is not None:
                hour = bar.ts.hour
                start = filters.entry_start_hour
                end = filters.entry_end_hour
                if start <= end:
                    if not (start <= hour < end):
                        filters_ok = False
                else:
                    if not (hour >= start or hour < end):
                        filters_ok = False
            if filters.skip_first_bars and bars_in_day <= filters.skip_first_bars:
                filters_ok = False
            if filters.cooldown_bars and last_entry_idx is not None:
                if (idx - last_entry_idx) < filters.cooldown_bars:
                    filters_ok = False
            if filters.ema_spread_min_pct is not None:
                if not ema_ready:
                    filters_ok = False
                else:
                    spread_pct = abs(ema_fast - ema_slow) / max(bar.close, 1e-9) * 100.0
                    if spread_pct < filters.ema_spread_min_pct:
                        filters_ok = False
            if filters.ema_slope_min_pct is not None:
                if not ema_ready or prev_ema_fast is None:
                    filters_ok = False
                else:
                    slope_pct = abs(ema_fast - prev_ema_fast) / max(bar.close, 1e-9) * 100.0
                    if slope_pct < filters.ema_slope_min_pct:
                        filters_ok = False

        open_slots_ok = cfg.strategy.max_open_trades == 0 or len(open_trades) < cfg.strategy.max_open_trades
        entries_ok = cfg.strategy.max_entries_per_day == 0 or entries_today < cfg.strategy.max_entries_per_day
        if (
            open_slots_ok
            and entries_ok
            and (bar.ts.weekday() in cfg.strategy.entry_days)
            and ema_gate_ok
            and filters_ok
        ):
            spot_leg = None
            if needs_direction and direction and cfg.strategy.directional_spot:
                spot_leg = cfg.strategy.directional_spot.get(direction)
            elif direction == "up":
                spot_leg = None

            if spot_leg is None and not needs_direction and direction == "up":
                spot_leg = SpotLegConfig(action="BUY", qty=1)

            if spot_leg is not None:
                qty = int(spot_leg.qty) * int(cfg.strategy.quantity)
                signed_qty = qty if spot_leg.action.upper() == "BUY" else -qty
                entry_price = bar.close
                candidate = SpotTrade(
                    symbol=cfg.strategy.symbol,
                    qty=signed_qty,
                    entry_time=bar.ts,
                    entry_price=entry_price,
                    profit_target_pct=cfg.strategy.spot_profit_target_pct,
                    stop_loss_pct=cfg.strategy.spot_stop_loss_pct,
                )
                candidate.margin_required = abs(signed_qty * entry_price) * meta.multiplier
                cash_after = cash - (signed_qty * entry_price) * meta.multiplier
                margin_after = margin_used + candidate.margin_required
                mark_liquidation = (signed_qty * entry_price) * meta.multiplier
                equity_after = cash_after + liquidation + mark_liquidation
                if cash_after >= 0 and equity_after >= margin_after:
                    open_trades.append(candidate)
                    cash = cash_after
                    margin_used = margin_after
                    entries_today += 1
                    last_entry_idx = idx
                    liquidation += mark_liquidation

        equity_curve.append(EquityPoint(ts=bar.ts, equity=cash + liquidation))
        prev_bar = bar

    if open_trades and prev_bar:
        for trade in open_trades:
            exit_price = prev_bar.close
            _close_spot_trade(trade, prev_bar.ts, exit_price, "end", trades)
            cash += (trade.qty * exit_price) * meta.multiplier
            margin_used = max(0.0, margin_used - trade.margin_required)

    summary = _summarize(trades, cfg.backtest.starting_cash, equity_curve, meta.multiplier)
    return BacktestResult(trades=trades, equity=equity_curve, summary=summary)


def _spot_hit_profit(trade: SpotTrade, price: float) -> bool:
    if trade.profit_target_pct is None:
        return False
    entry = trade.entry_price
    if entry <= 0:
        return False
    move = (price - entry) / entry
    if trade.qty < 0:
        move = -move
    return move >= trade.profit_target_pct


def _spot_hit_stop(trade: SpotTrade, price: float) -> bool:
    if trade.stop_loss_pct is None:
        return False
    entry = trade.entry_price
    if entry <= 0:
        return False
    move = (price - entry) / entry
    if trade.qty < 0:
        move = -move
    return move <= -trade.stop_loss_pct


def _spot_hit_flip_exit(
    cfg: ConfigBundle,
    trade: SpotTrade,
    bar: Bar,
    ema_ready: bool,
    ema_fast: float | None,
    ema_slow: float | None,
    cross_up: bool,
    cross_down: bool,
) -> bool:
    if not cfg.strategy.exit_on_signal_flip:
        return False
    if cfg.strategy.direction_source != "ema":
        return False
    if not ema_ready or ema_fast is None or ema_slow is None:
        return False
    if cfg.strategy.flip_exit_min_hold_bars:
        held = _bars_held(cfg.backtest.bar_size, trade.entry_time, bar.ts)
        if held < cfg.strategy.flip_exit_min_hold_bars:
            return False
    if cfg.strategy.flip_exit_only_if_profit:
        pnl = (price := bar.close) - trade.entry_price
        if trade.qty < 0:
            pnl = -pnl
        if pnl <= 0:
            return False

    trade_dir = "up" if trade.qty > 0 else "down" if trade.qty < 0 else None
    if trade_dir is None:
        return False

    mode = _flip_exit_mode(cfg)
    if mode == "cross":
        if trade_dir == "up":
            return cross_down
        if trade_dir == "down":
            return cross_up
        return False

    if trade_dir == "up":
        return ema_fast < ema_slow
    if trade_dir == "down":
        return ema_fast > ema_slow
    return False


def _close_spot_trade(trade: SpotTrade, ts: datetime, price: float, reason: str, trades: list[SpotTrade]) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


def _trade_value(
    trade: OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
    calibration,
) -> float:
    return _trade_value_from_spec(
        trade,
        bar,
        rv,
        cfg,
        surface_params,
        min_tick,
        is_future,
        mode="mark",
        calibration=calibration,
    )


def _rv_from_bars(bars: list[Bar], cfg: ConfigBundle) -> float:
    closes = [bar.close for bar in bars if bar.close]
    returns = []
    for idx in range(1, len(closes)):
        if closes[idx - 1] > 0:
            returns.append(math.log(closes[idx] / closes[idx - 1]))
    if not returns:
        return cfg.synthetic.iv_floor
    rv = ewma_vol(returns[-cfg.synthetic.rv_lookback :], cfg.synthetic.rv_ewma_lambda)
    rv *= math.sqrt(_annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
    return rv


def _ema_periods(preset: str | None) -> tuple[int, int] | None:
    return _ema_periods_shared(preset)


def _ema_bias(cfg: ConfigBundle) -> str:
    if cfg.strategy.ema_directional:
        return "any"
    legs = cfg.strategy.legs
    if legs:
        first = legs[0]
        action = first.action.upper()
        right = first.right.upper()
        if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
            return "up"
        if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
            return "down"
        return "any"
    right = cfg.strategy.right.upper()
    if right == "PUT":
        return "up"
    if right == "CALL":
        return "down"
    return "any"


def _direction_from_legs(legs: list[OptionLeg]) -> str | None:
    if not legs:
        return None
    first = legs[0]
    action = first.action.upper()
    right = first.right.upper()
    if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "up"
    if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "down"
    return None


def _ema_next(current: float | None, price: float, period: int) -> float:
    return _ema_next_shared(current, price, period)


def _trade_value_from_spec(
    spec: TradeSpec | OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
    mode: str,
    calibration,
) -> float:
    dte_days = max((spec.expiry - bar.ts.date()).days, 0)
    if calibration:
        surface_params = calibration.surface_params_asof(
            dte_days,
            bar.ts.date().isoformat(),
            surface_params,
        )
    atm_iv = iv_atm(rv, dte_days, surface_params)
    forward = bar.close
    if dte_days == 0:
        t = max(_session_hours(cfg.backtest.use_rth) / (24.0 * 365.0), _min_time(cfg.backtest.bar_size))
    else:
        t = max(dte_days / 365.0, _min_time(cfg.backtest.bar_size))
    legs = spec.legs
    if len(legs) <= 1:
        net = 0.0
        for leg in legs:
            leg_iv = iv_for_strike(atm_iv, forward, leg.strike, surface_params)
            if is_future:
                mid = black_76(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
            else:
                mid = black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
            quote = mid_edge_quote(mid, cfg.synthetic.min_spread_pct, min_tick)
            if mode == "entry":
                price = quote.bid if leg.action == "SELL" else quote.ask
            elif mode == "exit":
                price = quote.ask if leg.action == "SELL" else quote.bid
            else:
                price = quote.mid
            sign = 1 if leg.action == "SELL" else -1
            net += sign * price * leg.qty
        return net

    # Multi-leg combos: apply a single bid/ask edge to the net mid instead of legging each spread.
    net_mid = 0.0
    for leg in legs:
        leg_iv = iv_for_strike(atm_iv, forward, leg.strike, surface_params)
        if is_future:
            mid = black_76(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        else:
            mid = black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        sign = 1 if leg.action == "SELL" else -1
        net_mid += sign * mid * leg.qty

    abs_mid = abs(net_mid)
    quote = mid_edge_quote(abs_mid, cfg.synthetic.min_spread_pct, min_tick)
    mid_signed = quote.mid if net_mid >= 0 else -quote.mid
    bid_signed = quote.bid if net_mid >= 0 else -quote.bid
    ask_signed = quote.ask if net_mid >= 0 else -quote.ask

    if mode == "mark":
        return mid_signed
    if mode == "entry":
        return bid_signed if net_mid >= 0 else ask_signed
    # mode == "exit"
    return ask_signed if net_mid >= 0 else bid_signed


def _hit_profit(trade: OptionTrade, current_value: float) -> bool:
    target = abs(trade.entry_price) * trade.profit_target
    return (trade.entry_price - current_value) >= target


def _hit_stop(trade: OptionTrade, current_value: float, basis: str, spot: float) -> bool:
    loss = max(0.0, current_value - trade.entry_price)
    if basis == "credit":
        if trade.entry_price >= 0:
            return current_value >= trade.entry_price * (1 + trade.stop_loss)
        return loss >= abs(trade.entry_price) * trade.stop_loss
    max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
    if max_loss is None:
        max_loss = _max_loss_estimate(trade, spot)
    if max_loss is None:
        max_loss = abs(trade.entry_price)
    return loss >= max_loss * trade.stop_loss


def _flip_exit_mode(cfg: ConfigBundle) -> str:
    return _flip_exit_mode_shared(cfg.strategy.flip_exit_mode, cfg.strategy.ema_entry_mode)


def _hit_exit_dte(cfg: ConfigBundle, trade: OptionTrade, today: date) -> bool:
    if cfg.strategy.exit_dte <= 0:
        return False
    entry_dte = _business_days_until(trade.entry_time.date(), trade.expiry)
    if cfg.strategy.exit_dte >= entry_dte:
        return False
    remaining = _business_days_until(today, trade.expiry)
    return remaining <= cfg.strategy.exit_dte


def _hit_flip_exit(
    cfg: ConfigBundle,
    trade: OptionTrade,
    bar: Bar,
    current_value: float,
    ema_ready: bool,
    ema_fast: float | None,
    ema_slow: float | None,
    cross_up: bool,
    cross_down: bool,
) -> bool:
    if not cfg.strategy.exit_on_signal_flip:
        return False
    if cfg.strategy.direction_source != "ema":
        return False
    if not ema_ready or ema_fast is None or ema_slow is None:
        return False
    if cfg.strategy.flip_exit_min_hold_bars:
        held = _bars_held(cfg.backtest.bar_size, trade.entry_time, bar.ts)
        if held < cfg.strategy.flip_exit_min_hold_bars:
            return False
    if cfg.strategy.flip_exit_only_if_profit:
        if (trade.entry_price - current_value) <= 0:
            return False
    trade_dir = _direction_from_legs(trade.legs)
    if trade_dir is None:
        return False

    mode = _flip_exit_mode(cfg)
    if mode == "cross":
        if trade_dir == "up":
            return cross_down
        if trade_dir == "down":
            return cross_up
        return False

    # state mode
    if trade_dir == "up":
        return ema_fast < ema_slow
    if trade_dir == "down":
        return ema_fast > ema_slow
    return False


def _bars_held(bar_size: str, start: datetime, end: datetime) -> int:
    hours = _bar_hours(bar_size)
    if hours <= 0:
        return 0
    return int((end - start).total_seconds() / 3600.0 / hours)


def _bar_hours(bar_size: str) -> float:
    label = bar_size.lower().strip()
    if "hour" in label:
        try:
            prefix = label.split("hour")[0].strip()
            return float(prefix) if prefix else 1.0
        except ValueError:
            return 1.0
    if "min" in label:
        try:
            prefix = label.split("min")[0].strip()
            mins = float(prefix) if prefix else 30.0
            return mins / 60.0
        except ValueError:
            return 0.5
    if "day" in label:
        try:
            prefix = label.split("day")[0].strip()
            days = float(prefix) if prefix else 1.0
            return days * 24.0
        except ValueError:
            return 24.0
    return 1.0


def _business_days_until(start: date, end: date) -> int:
    if end <= start:
        return 0
    days = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


def _close_trade(trade: OptionTrade, ts: datetime, price: float, reason: str, trades: list[OptionTrade]) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


def _summarize(
    trades: list[OptionTrade | SpotTrade],
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


def _max_loss(trade: OptionTrade) -> float | None:
    legs = trade.legs
    if len(legs) != 2:
        return None
    a, b = legs
    if a.right != b.right:
        return None
    if a.qty != b.qty:
        return None
    if {a.action, b.action} != {"BUY", "SELL"}:
        return None
    width = abs(a.strike - b.strike)
    if trade.entry_price >= 0:
        return max(0.0, width - trade.entry_price)
    return abs(trade.entry_price)


def _annualization_factor(bar_size: str, use_rth: bool) -> float:
    label = bar_size.lower()
    if "hour" in label:
        return 252 * (6.5 if use_rth else 24)
    if "day" in label:
        return 252
    return 252 * (6.5 if use_rth else 24)


def _session_hours(use_rth: bool) -> float:
    return 6.5 if use_rth else 24.0


def _min_time(bar_size: str) -> float:
    label = bar_size.lower()
    if "hour" in label:
        hours = 1.0
    elif "min" in label:
        try:
            hours = float(label.split("min")[0].strip()) / 60.0
        except (ValueError, IndexError):
            hours = 0.5
    elif "day" in label:
        hours = 24.0
    else:
        hours = 1.0
    return hours / (24.0 * 365.0)


def _margin_required(trade: OptionTrade, spot: float, multiplier: float) -> float:
    if trade.entry_price <= 0:
        return 0.0
    max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
    if max_loss is None:
        max_loss = _max_loss_estimate(trade, spot)
    if max_loss is None:
        return 0.0
    return max(0.0, max_loss) * multiplier


def _max_loss_estimate(trade: OptionTrade, spot: float) -> float | None:
    strikes = sorted({leg.strike for leg in trade.legs})
    if not strikes:
        return None
    high = max(spot, strikes[-1]) * 5.0
    candidates = [0.0] + strikes + [high]
    min_pnl = None
    for price in candidates:
        pnl = trade.entry_price + _payoff_at_expiry(trade.legs, price)
        if min_pnl is None or pnl < min_pnl:
            min_pnl = pnl
    if min_pnl is None:
        return None
    return max(0.0, -min_pnl)


def _payoff_at_expiry(legs: list[OptionLeg], spot: float) -> float:
    payoff = 0.0
    for leg in legs:
        right = leg.right.upper()
        if right == "CALL":
            intrinsic = max(spot - leg.strike, 0.0)
        else:
            intrinsic = max(leg.strike - spot, 0.0)
        sign = 1.0 if leg.action.upper() == "BUY" else -1.0
        payoff += sign * intrinsic * leg.qty
    return payoff


def _equity_after_entry(
    cash_after: float,
    open_trades: list[OptionTrade],
    candidate: OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    meta: ContractMeta,
    is_future: bool,
    calibration,
) -> float:
    liquidation = 0.0
    for trade in open_trades:
        mark_value = _trade_value(trade, bar, rv, cfg, surface_params, meta.min_tick, is_future, calibration)
        liquidation += (-mark_value) * meta.multiplier
    candidate_mark = _trade_value(candidate, bar, rv, cfg, surface_params, meta.min_tick, is_future, calibration)
    liquidation += (-candidate_mark) * meta.multiplier
    return cash_after + liquidation
