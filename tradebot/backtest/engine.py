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
from ..decision_core import (
    EmaDecisionEngine,
    EmaDecisionSnapshot,
    OrbDecisionEngine,
    SupertrendEngine,
    apply_regime_gate,
    annualization_factor,
    cooldown_ok_by_index,
    flip_exit_hit,
    signal_filters_ok,
)
from ..signals import ema_next, ema_periods as _ema_periods_shared


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
            if cfg.backtest.offline:
                regime_bars = data.load_cached_bars(
                    symbol=cfg.strategy.symbol,
                    exchange=cfg.strategy.exchange,
                    start=start_dt,
                    end=end_dt,
                    bar_size=str(regime_bar),
                    use_rth=cfg.backtest.use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )
            else:
                regime_bars = data.load_or_fetch_bars(
                    symbol=cfg.strategy.symbol,
                    exchange=cfg.strategy.exchange,
                    start=start_dt,
                    end=end_dt,
                    bar_size=str(regime_bar),
                    use_rth=cfg.backtest.use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )
        regime2_mode = str(getattr(cfg.strategy, "regime2_mode", "off") or "off").strip().lower()
        if regime2_mode not in ("off", "ema", "supertrend"):
            regime2_mode = "off"
        regime2_preset = getattr(cfg.strategy, "regime2_ema_preset", None)
        regime2_bar = getattr(cfg.strategy, "regime2_bar_size", None) or cfg.backtest.bar_size
        if regime2_mode == "supertrend":
            use_mtf_regime2 = str(regime2_bar) != str(cfg.backtest.bar_size)
        else:
            use_mtf_regime2 = bool(regime2_preset) and str(regime2_bar) != str(cfg.backtest.bar_size)
        regime2_bars = None
        if use_mtf_regime2:
            if cfg.backtest.offline:
                regime2_bars = data.load_cached_bars(
                    symbol=cfg.strategy.symbol,
                    exchange=cfg.strategy.exchange,
                    start=start_dt,
                    end=end_dt,
                    bar_size=str(regime2_bar),
                    use_rth=cfg.backtest.use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )
            else:
                regime2_bars = data.load_or_fetch_bars(
                    symbol=cfg.strategy.symbol,
                    exchange=cfg.strategy.exchange,
                    start=start_dt,
                    end=end_dt,
                    bar_size=str(regime2_bar),
                    use_rth=cfg.backtest.use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )

        tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
        if tick_mode not in ("off", "raschke"):
            tick_mode = "off"
        tick_bars = None
        if tick_mode != "off":
            tick_symbol = str(getattr(cfg.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
            tick_exchange = str(getattr(cfg.strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip()
            try:
                z_lookback = int(getattr(cfg.strategy, "tick_width_z_lookback", 252) or 252)
            except (TypeError, ValueError):
                z_lookback = 252
            try:
                ma_period = int(getattr(cfg.strategy, "tick_band_ma_period", 10) or 10)
            except (TypeError, ValueError):
                ma_period = 10
            try:
                slope_lb = int(getattr(cfg.strategy, "tick_width_slope_lookback", 3) or 3)
            except (TypeError, ValueError):
                slope_lb = 3
            tick_warm_days = max(60, z_lookback + ma_period + slope_lb + 5)
            tick_start_dt = start_dt - timedelta(days=tick_warm_days)
            # $TICK is defined for RTH only (NYSE hours).
            tick_use_rth = True
            if cfg.backtest.offline:
                tick_bars = data.load_cached_bars(
                    symbol=tick_symbol,
                    exchange=tick_exchange,
                    start=tick_start_dt,
                    end=end_dt,
                    bar_size="1 day",
                    use_rth=tick_use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )
            else:
                tick_bars = data.load_or_fetch_bars(
                    symbol=tick_symbol,
                    exchange=tick_exchange,
                    start=tick_start_dt,
                    end=end_dt,
                    bar_size="1 day",
                    use_rth=tick_use_rth,
                    cache_dir=cfg.backtest.cache_dir,
                )

        result = _run_spot_backtest(
            cfg,
            bars,
            meta,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
        )
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
        if cfg.backtest.offline:
            regime_bars = data.load_cached_bars(
                symbol=cfg.strategy.symbol,
                exchange=cfg.strategy.exchange,
                start=start_dt,
                end=end_dt,
                bar_size=str(regime_bar),
                use_rth=cfg.backtest.use_rth,
                cache_dir=cfg.backtest.cache_dir,
            )
        else:
            regime_bars = data.load_or_fetch_bars(
                symbol=cfg.strategy.symbol,
                exchange=cfg.strategy.exchange,
                start=start_dt,
                end=end_dt,
                bar_size=str(regime_bar),
                use_rth=cfg.backtest.use_rth,
                cache_dir=cfg.backtest.cache_dir,
            )

    signal_engine: EmaDecisionEngine | None = None
    if ema_periods is not None:
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
        is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
        if prev_bar is not None:
            if prev_bar.close > 0:
                returns.append(math.log(bar.close / prev_bar.close))
        rv = ewma_vol(returns, surface_params.rv_ewma_lambda)
        rv *= math.sqrt(annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
        if last_date != bar.ts.date():
            bars_in_day = 0
            last_date = bar.ts.date()
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
                    signal,
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


def _run_spot_backtest(
    cfg: ConfigBundle,
    bars: list[Bar],
    meta: ContractMeta,
    *,
    regime_bars: list[Bar] | None = None,
    regime2_bars: list[Bar] | None = None,
    tick_bars: list[Bar] | None = None,
) -> BacktestResult:
    returns = deque(maxlen=cfg.synthetic.rv_lookback)
    cash = cfg.backtest.starting_cash
    margin_used = 0.0
    equity_curve: list[EquityPoint] = []
    trades: list[SpotTrade] = []
    open_trades: list[SpotTrade] = []
    prev_bar: Bar | None = None
    entries_today = 0
    filters = cfg.strategy.filters
    entry_signal = str(getattr(cfg.strategy, "entry_signal", "ema") or "ema").strip().lower()
    if entry_signal not in ("ema", "orb"):
        entry_signal = "ema"

    ema_periods = _ema_periods(cfg.strategy.ema_preset) if entry_signal == "ema" else None
    needs_direction = cfg.strategy.directional_spot is not None
    if entry_signal == "ema" and ema_periods is None:
        raise ValueError("spot backtests require ema_preset")
    ema_needed = entry_signal == "ema"

    regime_mode = str(getattr(cfg.strategy, "regime_mode", "ema") or "ema").strip().lower()
    if regime_mode not in ("ema", "supertrend"):
        regime_mode = "ema"
    use_mtf_regime = bool(regime_bars) and (
        regime_mode == "supertrend" or bool(cfg.strategy.regime_ema_preset)
    )
    regime2_mode = str(getattr(cfg.strategy, "regime2_mode", "off") or "off").strip().lower()
    if regime2_mode not in ("off", "ema", "supertrend"):
        regime2_mode = "off"
    regime2_preset = str(getattr(cfg.strategy, "regime2_ema_preset", "") or "").strip()
    if regime2_mode == "ema" and not regime2_preset:
        regime2_mode = "off"
    regime2_bar = str(getattr(cfg.strategy, "regime2_bar_size", "") or "").strip() or str(cfg.backtest.bar_size)
    if regime2_mode != "off" and str(regime2_bar) != str(cfg.backtest.bar_size) and not regime2_bars:
        raise ValueError("regime2_mode enabled but regime2_bars was not provided for multi-timeframe regime2")
    use_mtf_regime2 = bool(regime2_bars) and (
        regime2_mode == "supertrend" or bool(regime2_preset)
    )
    signal_engine = (
        EmaDecisionEngine(
            ema_preset=str(cfg.strategy.ema_preset),
            ema_entry_mode=cfg.strategy.ema_entry_mode,
            entry_confirm_bars=cfg.strategy.entry_confirm_bars,
            regime_ema_preset=(
                None
                if (use_mtf_regime or regime_mode == "supertrend")
                else cfg.strategy.regime_ema_preset
            ),
        )
        if entry_signal == "ema"
        else None
    )
    orb_engine = (
        OrbDecisionEngine(
            window_mins=int(getattr(cfg.strategy, "orb_window_mins", 15) or 15),
        )
        if entry_signal == "orb"
        else None
    )
    regime_engine = (
        EmaDecisionEngine(
            ema_preset=str(cfg.strategy.regime_ema_preset),
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
    regime2_engine = (
        EmaDecisionEngine(
            ema_preset=str(regime2_preset),
            ema_entry_mode="trend",
            entry_confirm_bars=0,
            regime_ema_preset=None,
        )
        if regime2_mode == "ema" and regime2_preset
        else None
    )
    supertrend2_engine = (
        SupertrendEngine(
            atr_period=int(getattr(cfg.strategy, "regime2_supertrend_atr_period", 10) or 10),
            multiplier=float(getattr(cfg.strategy, "regime2_supertrend_multiplier", 3.0) or 3.0),
            source=str(getattr(cfg.strategy, "regime2_supertrend_source", "hl2") or "hl2"),
        )
        if regime2_mode == "supertrend"
        else None
    )
    regime_idx = 0
    last_regime = None
    last_supertrend = None
    regime2_idx = 0
    last_regime2 = None
    last_supertrend2 = None

    tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
    if tick_mode not in ("off", "raschke"):
        tick_mode = "off"
    tick_neutral_policy = str(getattr(cfg.strategy, "tick_neutral_policy", "allow") or "allow").strip().lower()
    if tick_neutral_policy not in ("allow", "block"):
        tick_neutral_policy = "allow"
    tick_direction_policy = str(
        getattr(cfg.strategy, "tick_direction_policy", "both") or "both"
    ).strip().lower()
    if tick_direction_policy not in ("both", "wide_only"):
        tick_direction_policy = "both"

    tick_ma_period = max(1, int(getattr(cfg.strategy, "tick_band_ma_period", 10) or 10))
    tick_z_lookback = max(5, int(getattr(cfg.strategy, "tick_width_z_lookback", 252) or 252))
    tick_z_enter = float(getattr(cfg.strategy, "tick_width_z_enter", 1.0) or 1.0)
    tick_z_exit = max(0.0, float(getattr(cfg.strategy, "tick_width_z_exit", 0.5) or 0.5))
    tick_slope_lookback = max(
        1, int(getattr(cfg.strategy, "tick_width_slope_lookback", 3) or 3)
    )

    tick_idx = 0
    tick_state = "neutral"  # "neutral" | "wide" | "narrow"
    tick_dir: str | None = None
    tick_ready = False
    tick_highs: deque[float] = deque(maxlen=tick_ma_period)
    tick_lows: deque[float] = deque(maxlen=tick_ma_period)
    tick_high_sum = 0.0
    tick_low_sum = 0.0
    tick_widths: deque[float] = deque(maxlen=tick_z_lookback)
    tick_width_hist: list[float] = []

    exit_mode = str(getattr(cfg.strategy, "spot_exit_mode", "pct") or "pct").strip().lower()
    if exit_mode not in ("pct", "atr"):
        exit_mode = "pct"
    exit_atr_engine = (
        SupertrendEngine(
            atr_period=int(getattr(cfg.strategy, "spot_atr_period", 14) or 14),
            multiplier=1.0,
            source="hl2",
        )
        if exit_mode == "atr"
        else None
    )
    last_exit_atr = None

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
        is_last_bar = next_bar is None or next_bar.ts.date() != bar.ts.date()
        if prev_bar is not None:
            if prev_bar.close > 0:
                returns.append(math.log(bar.close / prev_bar.close))
        rv = ewma_vol(returns, cfg.synthetic.rv_ewma_lambda)
        rv *= math.sqrt(annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
        if last_date != bar.ts.date():
            bars_in_day = 0
            last_date = bar.ts.date()
            entries_today = 0
        bars_in_day += 1
        if volume_period is not None:
            volume_ema = ema_next(volume_ema, float(bar.volume), volume_period)
            volume_count += 1
        if exit_atr_engine is not None:
            last_exit_atr = exit_atr_engine.update(
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )
        if signal_engine is not None:
            signal = signal_engine.update(bar.close)
        elif orb_engine is not None:
            signal = orb_engine.update(
                ts=bar.ts,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
            )
        else:
            signal = None
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
        elif use_mtf_regime and regime_engine is not None and regime_bars is not None:
            while regime_idx < len(regime_bars) and regime_bars[regime_idx].ts <= bar.ts:
                last_regime = regime_engine.update(regime_bars[regime_idx].close)
                regime_idx += 1
            signal = apply_regime_gate(
                signal,
                regime_dir=last_regime.state if last_regime is not None else None,
                regime_ready=bool(last_regime and last_regime.ema_ready),
            )

        if supertrend2_engine is not None:
            if use_mtf_regime2 and regime2_bars is not None:
                while regime2_idx < len(regime2_bars) and regime2_bars[regime2_idx].ts <= bar.ts:
                    reg_bar = regime2_bars[regime2_idx]
                    last_supertrend2 = supertrend2_engine.update(
                        high=float(reg_bar.high),
                        low=float(reg_bar.low),
                        close=float(reg_bar.close),
                    )
                    regime2_idx += 1
            else:
                last_supertrend2 = supertrend2_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )
            signal = apply_regime_gate(
                signal,
                regime_dir=last_supertrend2.direction if last_supertrend2 is not None else None,
                regime_ready=bool(last_supertrend2 and last_supertrend2.ready),
            )
        elif regime2_engine is not None:
            if use_mtf_regime2 and regime2_bars is not None:
                while regime2_idx < len(regime2_bars) and regime2_bars[regime2_idx].ts <= bar.ts:
                    last_regime2 = regime2_engine.update(regime2_bars[regime2_idx].close)
                    regime2_idx += 1
            else:
                last_regime2 = regime2_engine.update(bar.close)
            signal = apply_regime_gate(
                signal,
                regime_dir=last_regime2.state if last_regime2 is not None else None,
                regime_ready=bool(last_regime2 and last_regime2.ema_ready),
            )
        ema_ready = bool(ema_needed and signal is not None and signal.ema_ready)

        if tick_mode != "off" and tick_bars is not None:
            while tick_idx < len(tick_bars) and tick_bars[tick_idx].ts <= bar.ts:
                tbar = tick_bars[tick_idx]
                high_v = float(tbar.high)
                low_v = float(tbar.low)

                if len(tick_highs) == tick_highs.maxlen:
                    tick_high_sum -= tick_highs[0]
                if len(tick_lows) == tick_lows.maxlen:
                    tick_low_sum -= tick_lows[0]
                tick_highs.append(high_v)
                tick_lows.append(low_v)
                tick_high_sum += high_v
                tick_low_sum += low_v

                tick_ready = False
                tick_dir = None
                if len(tick_highs) >= tick_ma_period and len(tick_lows) >= tick_ma_period:
                    upper = tick_high_sum / float(tick_ma_period)
                    lower = tick_low_sum / float(tick_ma_period)
                    width = float(upper) - float(lower)
                    tick_widths.append(width)
                    tick_width_hist.append(width)

                    # Use a bounded window for normalization, but don't require a full 252 sessions.
                    min_z = min(tick_z_lookback, 30)
                    if len(tick_widths) >= max(5, min_z) and len(tick_width_hist) >= (tick_slope_lookback + 1):
                        w_list = list(tick_widths)
                        mean = sum(w_list) / float(len(w_list))
                        var = sum((w - mean) ** 2 for w in w_list) / float(len(w_list))
                        std = math.sqrt(var)
                        z = (width - mean) / std if std > 1e-9 else 0.0
                        delta = width - tick_width_hist[-1 - tick_slope_lookback]

                        if tick_state == "neutral":
                            if z >= tick_z_enter and delta > 0:
                                tick_state = "wide"
                            elif z <= (-tick_z_enter) and delta < 0:
                                tick_state = "narrow"
                        elif tick_state == "wide":
                            if z < tick_z_exit:
                                tick_state = "neutral"
                        elif tick_state == "narrow":
                            if z > (-tick_z_exit):
                                tick_state = "neutral"

                        if tick_state == "wide":
                            tick_dir = "up"
                        elif tick_state == "narrow":
                            tick_dir = "down" if tick_direction_policy == "both" else None
                        else:
                            tick_dir = None
                        tick_ready = True

                tick_idx += 1

        liquidation = 0.0
        if open_trades:
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
                    signal,
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

        entry_signal_dir = signal.entry_dir if signal is not None else None
        if tick_mode != "off":
            if not tick_ready:
                if tick_neutral_policy == "block":
                    entry_signal_dir = None
            elif tick_dir is None:
                if tick_neutral_policy == "block":
                    entry_signal_dir = None
            elif entry_signal_dir is not None and entry_signal_dir != tick_dir:
                entry_signal_dir = None

        entry_ok = True
        direction = entry_signal_dir
        if ema_needed and not ema_ready:
            entry_ok = False
            direction = None
        if needs_direction:
            entry_ok = (
                entry_ok
                and direction is not None
                and cfg.strategy.directional_spot is not None
                and direction in cfg.strategy.directional_spot
            )
        else:
            entry_ok = entry_ok and direction == "up"

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

        open_slots_ok = cfg.strategy.max_open_trades == 0 or len(open_trades) < cfg.strategy.max_open_trades
        entries_ok = cfg.strategy.max_entries_per_day == 0 or entries_today < cfg.strategy.max_entries_per_day
        if (
            open_slots_ok
            and entries_ok
            and (bar.ts.weekday() in cfg.strategy.entry_days)
            and entry_ok
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
                can_open = True
                qty = int(spot_leg.qty) * int(cfg.strategy.quantity)
                signed_qty = qty if spot_leg.action.upper() == "BUY" else -qty
                entry_price = bar.close
                target_price = None
                stop_price = None
                profit_target_pct = cfg.strategy.spot_profit_target_pct
                stop_loss_pct = cfg.strategy.spot_stop_loss_pct
                if entry_signal == "orb" and orb_engine is not None and direction in ("up", "down"):
                    orb_high = orb_engine.or_high
                    orb_low = orb_engine.or_low
                    if orb_high is not None and orb_low is not None and orb_high > 0 and orb_low > 0:
                        stop_price = float(orb_low) if direction == "up" else float(orb_high)
                        rr = float(getattr(cfg.strategy, "orb_risk_reward", 2.0) or 2.0)
                        target_mode = str(getattr(cfg.strategy, "orb_target_mode", "rr") or "rr").strip().lower()
                        if target_mode not in ("rr", "or_range"):
                            target_mode = "rr"

                        if rr <= 0:
                            can_open = False
                        elif target_mode == "or_range":
                            rng = float(orb_high) - float(orb_low)
                            if rng <= 0:
                                can_open = False
                            else:
                                target_price = (
                                    float(orb_high) + (rr * rng)
                                    if direction == "up"
                                    else float(orb_low) - (rr * rng)
                                )
                                if (
                                    (direction == "up" and float(target_price) <= float(entry_price))
                                    or (direction == "down" and float(target_price) >= float(entry_price))
                                ):
                                    can_open = False
                        else:
                            risk = abs(float(entry_price) - float(stop_price))
                            if risk <= 0:
                                can_open = False
                            else:
                                target_price = (
                                    float(entry_price) + (rr * risk)
                                    if direction == "up"
                                    else float(entry_price) - (rr * risk)
                                )
                    profit_target_pct = None
                    stop_loss_pct = None
                elif exit_mode == "atr":
                    atr = float(getattr(last_exit_atr, "atr", 0.0) or 0.0)
                    if atr > 0 and direction in ("up", "down"):
                        pt_mult = float(getattr(cfg.strategy, "spot_pt_atr_mult", 1.5) or 1.5)
                        sl_mult = float(getattr(cfg.strategy, "spot_sl_atr_mult", 1.0) or 1.0)
                        if signed_qty > 0:
                            target_price = float(entry_price) + (pt_mult * atr)
                            stop_price = float(entry_price) - (sl_mult * atr)
                        else:
                            target_price = float(entry_price) - (pt_mult * atr)
                            stop_price = float(entry_price) + (sl_mult * atr)
                        profit_target_pct = None
                        stop_loss_pct = None
                    else:
                        # Wait until ATR is ready.
                        can_open = False
                if not can_open:
                    pass
                else:
                    candidate = SpotTrade(
                        symbol=cfg.strategy.symbol,
                        qty=signed_qty,
                        entry_time=bar.ts,
                        entry_price=entry_price,
                        profit_target_pct=profit_target_pct,
                        stop_loss_pct=stop_loss_pct,
                        profit_target_price=target_price,
                        stop_loss_price=stop_price,
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
    if trade.profit_target_price is not None:
        target = float(trade.profit_target_price)
        if target <= 0:
            return False
        if trade.qty > 0:
            return float(price) >= target
        if trade.qty < 0:
            return float(price) <= target
        return False
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
    if trade.stop_loss_price is not None:
        stop = float(trade.stop_loss_price)
        if stop <= 0:
            return False
        if trade.qty > 0:
            return float(price) <= stop
        if trade.qty < 0:
            return float(price) >= stop
        return False
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
    signal: EmaDecisionSnapshot | None,
) -> bool:
    if cfg.strategy.direction_source != "ema":
        return False
    trade_dir = "up" if trade.qty > 0 else "down" if trade.qty < 0 else None
    if trade_dir is None:
        return False
    if not flip_exit_hit(
        exit_on_signal_flip=bool(cfg.strategy.exit_on_signal_flip),
        open_dir=trade_dir,
        signal=signal,
        flip_exit_mode_raw=cfg.strategy.flip_exit_mode,
        ema_entry_mode_raw=cfg.strategy.ema_entry_mode,
    ):
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
    return True


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
    rv *= math.sqrt(annualization_factor(cfg.backtest.bar_size, cfg.backtest.use_rth))
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
    signal: EmaDecisionSnapshot | None,
) -> bool:
    if cfg.strategy.direction_source != "ema":
        return False
    trade_dir = _direction_from_legs(trade.legs)
    if trade_dir is None:
        return False
    if not flip_exit_hit(
        exit_on_signal_flip=bool(cfg.strategy.exit_on_signal_flip),
        open_dir=trade_dir,
        signal=signal,
        flip_exit_mode_raw=cfg.strategy.flip_exit_mode,
        ema_entry_mode_raw=cfg.strategy.ema_entry_mode,
    ):
        return False

    if cfg.strategy.flip_exit_min_hold_bars:
        held = _bars_held(cfg.backtest.bar_size, trade.entry_time, bar.ts)
        if held < cfg.strategy.flip_exit_min_hold_bars:
            return False

    if cfg.strategy.flip_exit_only_if_profit:
        if (trade.entry_price - current_value) <= 0:
            return False
    return True


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
