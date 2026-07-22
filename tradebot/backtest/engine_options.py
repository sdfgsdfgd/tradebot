"""Options-only backtest runner.

This module keeps the options execution path separate from spot execution logic.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import date, datetime

from .calibration import ensure_calibration, load_calibration
from .config import ConfigBundle, OptionsStrategyConfig
from .data import ContractMeta, IBKRHistoricalData, load_backtest_series
from .models import BacktestResult, Bar, EquityPoint, OptionLeg, OptionTrade, summarize
from .strategy import CreditSpreadStrategy, TradeSpec
from .synth import IVSurfaceParams, black_76, black_scholes, iv_atm, iv_for_strike, mid_edge_quote
from ..engines.signals import EmaDecisionEngine, EmaDecisionSnapshot, SupertrendEngine
from ..option_package import option_package_debit_value, option_profit_target_hit, option_stop_loss_hit
from ..spot.gates import apply_regime_gate, flip_exit_allowed, signal_filters_ok
from ..engine import (
    _trade_date,
    annualized_ewma_vol,
    cooldown_ok_by_index,
    realized_vol_from_closes,
)
from ..signals import ema_next, ema_periods, parse_bar_size
from ..utils.date_utils import business_days_until

def run_options_backtest(
    *,
    cfg: ConfigBundle,
    bars: list[Bar],
    meta: ContractMeta,
    data: IBKRHistoricalData,
    start_dt: datetime,
    end_dt: datetime,
) -> BacktestResult:
    if not isinstance(cfg.strategy, OptionsStrategyConfig):
        raise ValueError("run_options_backtest requires an options strategy config")

    is_future = cfg.strategy.symbol in ("MNQ", "MBT")
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
    periods = ema_periods(cfg.strategy.ema_preset)
    needs_direction = cfg.strategy.directional_legs is not None
    if needs_direction and cfg.strategy.direction_source == "ema" and periods is None:
        raise ValueError("directional_legs requires ema_preset when direction_source='ema'")
    ema_needed = periods is not None or (
        filters
        and (filters.ema_spread_min_pct is not None or filters.ema_slope_min_pct is not None)
    )
    if needs_direction:
        ema_needed = True

    regime_mode = str(getattr(cfg.strategy, "regime_mode", "ema") or "ema").strip().lower()
    if regime_mode not in ("ema", "supertrend"):
        regime_mode = "ema"
    regime_preset = cfg.strategy.regime_ema_preset
    regime_bar = cfg.strategy.regime_bar_size or cfg.backtest.bar_size
    if regime_mode == "supertrend":
        use_mtf_regime = str(regime_bar) != str(cfg.backtest.bar_size)
    else:
        use_mtf_regime = bool(regime_preset) and str(regime_bar) != str(cfg.backtest.bar_size)
    regime_bars = None
    if use_mtf_regime:
        regime_bars = load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=cfg.strategy.symbol,
            exchange=cfg.strategy.exchange,
            start=start_dt,
            end=end_dt,
            bar_size=str(regime_bar),
            use_rth=cfg.backtest.use_rth,
        ).as_list()

    signal_engine: EmaDecisionEngine | None = None
    if periods is not None:
        signal_engine = EmaDecisionEngine(
            ema_preset=str(cfg.strategy.ema_preset),
            ema_entry_mode=cfg.strategy.ema_entry_mode,
            entry_confirm_bars=cfg.strategy.entry_confirm_bars,
            regime_ema_preset=(
                None if (use_mtf_regime or regime_mode == "supertrend") else cfg.strategy.regime_ema_preset
            ),
        )
    regime_engine = (
        EmaDecisionEngine(
            ema_preset=str(regime_preset),
            ema_entry_mode="trend",
            entry_confirm_bars=0,
            regime_ema_preset=None,
        )
        if use_mtf_regime and regime_mode == "ema"
        else None
    )
    supertrend_engine = (
        SupertrendEngine(
            atr_period=int(getattr(cfg.strategy, "supertrend_atr_period", 10) or 10),
            multiplier=float(getattr(cfg.strategy, "supertrend_multiplier", 3.0) or 3.0),
            source=str(getattr(cfg.strategy, "supertrend_source", "hl2") or "hl2"),
        )
        if regime_mode == "supertrend"
        else None
    )
    regime_idx = 0
    last_regime = None
    last_supertrend = None

    volume_period = None
    if filters is not None and getattr(filters, "volume_ratio_min", None) is not None:
        raw_period = getattr(filters, "volume_ema_period", None)
        try:
            volume_period = int(raw_period) if raw_period is not None else 20
        except (TypeError, ValueError):
            volume_period = 20
        volume_period = max(1, volume_period)
    volume_ema = None
    volume_count = 0

    bars_in_day = 0
    last_entry_idx = None
    last_date = None
    for idx, bar in enumerate(bars):
        next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
        is_last_bar = next_bar is None or _trade_date(next_bar.ts) != _trade_date(bar.ts)
        if prev_bar is not None and prev_bar.close > 0:
            returns.append(math.log(bar.close / prev_bar.close))
        rv = annualized_ewma_vol(
            returns,
            lam=float(surface_params.rv_ewma_lambda),
            bar_size=str(cfg.backtest.bar_size),
            use_rth=bool(cfg.backtest.use_rth),
        )
        if last_date != _trade_date(bar.ts):
            bars_in_day = 0
            last_date = _trade_date(bar.ts)
            entries_today = 0
        bars_in_day += 1

        if volume_period is not None:
            volume_ema = ema_next(volume_ema, float(bar.volume), volume_period)
            volume_count += 1
        signal = signal_engine.update(bar.close) if signal_engine is not None else None
        if supertrend_engine is not None:
            if use_mtf_regime and regime_bars is not None:
                while regime_idx < len(regime_bars) and regime_bars[regime_idx].ts <= bar.ts:
                    reg_bar = regime_bars[regime_idx]
                    last_supertrend = supertrend_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                    )
                    regime_idx += 1
            else:
                last_supertrend = supertrend_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )
            signal = apply_regime_gate(
                signal,
                regime_dir=last_supertrend.direction if last_supertrend is not None else None,
                regime_ready=bool(last_supertrend and last_supertrend.ready),
            )
        elif use_mtf_regime and signal is not None and regime_engine is not None and regime_bars is not None:
            while regime_idx < len(regime_bars) and regime_bars[regime_idx].ts <= bar.ts:
                last_regime = regime_engine.update(regime_bars[regime_idx].close)
                regime_idx += 1
            signal = apply_regime_gate(
                signal,
                regime_dir=last_regime.state if last_regime is not None else None,
                regime_ready=bool(last_regime and last_regime.ema_ready),
            )
        ema_ready = bool(signal and signal.ema_ready)

        if open_trades and prev_bar:
            still_open: list[OptionTrade] = []
            for trade in open_trades:
                if _trade_date(bar.ts) > trade.expiry:
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
            still_open = []
            for trade in open_trades:
                current_value = _trade_value(
                    trade,
                    bar,
                    rv,
                    cfg,
                    surface_params,
                    meta.min_tick,
                    is_future,
                    calibration,
                )
                should_close = False
                reason = ""

                if _hit_profit(trade, current_value):
                    should_close = True
                    reason = "profit"
                elif _hit_stop(trade, current_value, cfg.strategy.stop_loss_basis, bar.close):
                    should_close = True
                    reason = "stop"
                elif _hit_exit_dte(cfg, trade, _trade_date(bar.ts)):
                    should_close = True
                    reason = "exit_dte"
                elif _hit_flip_exit(cfg, trade, bar, current_value, signal):
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

        entry_signal_dir = signal.entry_dir if signal is not None else None

        ema_gate_ok = True
        ema_right_override = None
        direction = None
        if ema_needed and not ema_ready:
            ema_gate_ok = False
        elif ema_ready:
            if needs_direction:
                if cfg.strategy.direction_source != "ema":
                    ema_gate_ok = False
                else:
                    direction = entry_signal_dir
                    ema_gate_ok = (
                        direction is not None
                        and cfg.strategy.directional_legs is not None
                        and direction in cfg.strategy.directional_legs
                    )
            elif cfg.strategy.ema_entry_mode == "cross":
                bias = _ema_bias(cfg)
                if bias == "up":
                    ema_gate_ok = entry_signal_dir == "up"
                elif bias == "down":
                    ema_gate_ok = entry_signal_dir == "down"
                else:
                    ema_gate_ok = entry_signal_dir in ("up", "down")
                if cfg.strategy.ema_directional:
                    if entry_signal_dir == "up":
                        ema_right_override = "CALL"
                    elif entry_signal_dir == "down":
                        ema_right_override = "PUT"
            else:
                if cfg.strategy.ema_directional:
                    if entry_signal_dir == "up":
                        ema_right_override = "CALL"
                    elif entry_signal_dir == "down":
                        ema_right_override = "PUT"
                    else:
                        ema_gate_ok = False
                else:
                    ema_gate_ok = entry_signal_dir == "up"

        cooldown_ok = cooldown_ok_by_index(
            current_idx=idx,
            last_entry_idx=last_entry_idx,
            cooldown_bars=filters.cooldown_bars if filters else 0,
        )
        filters_ok = signal_filters_ok(
            filters,
            bar_ts=bar.ts,
            bars_in_day=bars_in_day,
            close=float(bar.close),
            volume=float(bar.volume),
            volume_ema=float(volume_ema) if volume_ema is not None else None,
            volume_ema_ready=bool(volume_count >= volume_period) if volume_period else True,
            rv=float(rv),
            signal=signal,
            cooldown_ok=cooldown_ok,
        )

        open_slots_ok = len(open_trades) < 1
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

    summary = summarize(trades, cfg.backtest.starting_cash, equity_curve, meta.multiplier)
    return BacktestResult(trades=trades, equity=equity_curve, summary=summary)


def _rv_from_bars(bars: list[Bar], cfg: ConfigBundle) -> float:
    closes = [float(bar.close) for bar in bars if bar.close and float(bar.close) > 0]
    rv = realized_vol_from_closes(
        closes,
        lookback=int(cfg.synthetic.rv_lookback),
        lam=float(cfg.synthetic.rv_ewma_lambda),
        bar_size=str(cfg.backtest.bar_size),
        use_rth=bool(cfg.backtest.use_rth),
    )
    return float(cfg.synthetic.iv_floor) if rv is None else float(rv)


def _ema_bias(cfg: ConfigBundle) -> str:
    if cfg.strategy.ema_directional:
        return "any"
    if cfg.strategy.legs:
        first = cfg.strategy.legs[0]
        pair = (first.action.upper(), first.right.upper())
        if pair in (("BUY", "CALL"), ("SELL", "PUT")):
            return "up"
        if pair in (("BUY", "PUT"), ("SELL", "CALL")):
            return "down"
        return "any"
    return {"PUT": "up", "CALL": "down"}.get(cfg.strategy.right.upper(), "any")


def _direction_from_legs(legs: list[OptionLeg]) -> str | None:
    if not legs:
        return None
    pair = (legs[0].action.upper(), legs[0].right.upper())
    if pair in (("BUY", "CALL"), ("SELL", "PUT")):
        return "up"
    if pair in (("BUY", "PUT"), ("SELL", "CALL")):
        return "down"
    return None


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
    trade_day = _trade_date(bar.ts)
    dte_days = max((spec.expiry - trade_day).days, 0)
    if calibration:
        surface_params = calibration.surface_params_asof(dte_days, trade_day.isoformat(), surface_params)
    atm_iv = iv_atm(rv, dte_days, surface_params)
    forward = bar.close
    bar_def = parse_bar_size(cfg.backtest.bar_size)
    bar_hours = bar_def.duration.total_seconds() / 3600.0 if bar_def is not None else 1.0
    min_time = bar_hours / (24.0 * 365.0)
    t = max(
        (6.5 if cfg.backtest.use_rth else 24.0) / (24.0 * 365.0) if dte_days == 0 else dte_days / 365.0,
        min_time,
    )

    mid_rows: list[tuple[str, int, float]] = []
    for leg in spec.legs:
        leg_iv = iv_for_strike(atm_iv, forward, leg.strike, surface_params)
        price = (
            black_76(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
            if is_future
            else black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        )
        mid_rows.append(("SELL" if leg.action == "SELL" else "BUY", leg.qty, price))

    if len(mid_rows) <= 1:
        rows: list[tuple[str, int, float]] = []
        for action, qty, mid in mid_rows:
            quote = mid_edge_quote(mid, cfg.synthetic.min_spread_pct, min_tick)
            if mode == "entry":
                price = quote.bid if action == "SELL" else quote.ask
            elif mode == "exit":
                price = quote.ask if action == "SELL" else quote.bid
            else:
                price = quote.mid
            rows.append((action, qty, price))
        debit_value = option_package_debit_value(rows)
        assert debit_value is not None
        return -float(debit_value)

    debit_mid = option_package_debit_value(mid_rows)
    assert debit_mid is not None
    net_mid = -float(debit_mid)
    quote = mid_edge_quote(abs(net_mid), cfg.synthetic.min_spread_pct, min_tick)
    mid_signed = quote.mid if net_mid >= 0 else -quote.mid
    bid_signed = quote.bid if net_mid >= 0 else -quote.bid
    ask_signed = quote.ask if net_mid >= 0 else -quote.ask
    if mode == "mark":
        return mid_signed
    if mode == "entry":
        return bid_signed if net_mid >= 0 else ask_signed
    return ask_signed if net_mid >= 0 else bid_signed


def _hit_profit(trade: OptionTrade, current_value: float) -> bool:
    return option_profit_target_hit(
        entry_value=trade.entry_price,
        current_value=current_value,
        profit_target=trade.profit_target,
    )


def _hit_stop(trade: OptionTrade, current_value: float, basis: str, spot: float) -> bool:
    max_loss = None
    if basis != "credit":
        max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
        if max_loss is None:
            max_loss = _max_loss_estimate(trade, spot)
        if max_loss is None:
            max_loss = abs(trade.entry_price)
    return option_stop_loss_hit(
        entry_value=trade.entry_price,
        current_value=current_value,
        stop_loss=trade.stop_loss,
        basis=basis,
        max_loss=max_loss,
    )


def _hit_exit_dte(cfg: ConfigBundle, trade: OptionTrade, today: date) -> bool:
    if cfg.strategy.exit_dte <= 0:
        return False
    entry_dte = business_days_until(_trade_date(trade.entry_time), trade.expiry)
    if cfg.strategy.exit_dte >= entry_dte:
        return False
    return business_days_until(today, trade.expiry) <= cfg.strategy.exit_dte


def _hit_flip_exit(
    cfg: ConfigBundle,
    trade: OptionTrade,
    bar: Bar,
    current_value: float,
    signal: EmaDecisionSnapshot | None,
) -> bool:
    if not flip_exit_allowed(
        strategy=cfg.strategy,
        open_dir=_direction_from_legs(trade.legs),
        entry_time=trade.entry_time,
        current_time=bar.ts,
        bar_size=str(cfg.backtest.bar_size),
        signal=signal,
    ):
        return False
    return not cfg.strategy.flip_exit_only_if_profit or trade.entry_price - current_value > 0


def _close_trade(trade: OptionTrade, ts: datetime, price: float, reason: str, trades: list[OptionTrade]) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


def _max_loss(trade: OptionTrade) -> float | None:
    if len(trade.legs) != 2:
        return None
    a, b = trade.legs
    if a.right != b.right or a.qty != b.qty or {a.action, b.action} != {"BUY", "SELL"}:
        return None
    if trade.entry_price < 0:
        return abs(trade.entry_price)
    return max(0.0, abs(a.strike - b.strike) * a.qty - trade.entry_price)


def _margin_required(trade: OptionTrade, spot: float, multiplier: float) -> float:
    if trade.entry_price <= 0:
        return 0.0
    max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
    if max_loss is None:
        max_loss = _max_loss_estimate(trade, spot)
    return max(0.0, max_loss) * multiplier if max_loss is not None else 0.0


def _max_loss_estimate(trade: OptionTrade, spot: float) -> float | None:
    strikes = sorted({leg.strike for leg in trade.legs})
    if not strikes:
        return None
    candidates = [0.0, *strikes, max(spot, strikes[-1]) * 5.0]
    pnl = [trade.entry_price + _payoff_at_expiry(trade.legs, price) for price in candidates]
    return max(0.0, -min(pnl))


def _payoff_at_expiry(legs: list[OptionLeg], spot: float) -> float:
    rows = [
        (
            "BUY" if leg.action.upper() == "BUY" else "SELL",
            leg.qty,
            max(spot - leg.strike, 0.0) if leg.right.upper() == "CALL" else max(leg.strike - spot, 0.0),
        )
        for leg in legs
    ]
    payoff = option_package_debit_value(rows)
    assert payoff is not None
    return float(payoff)
