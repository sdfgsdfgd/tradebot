"""Options-only backtest runner.

This module keeps the options execution path separate from spot execution logic.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime

from .calibration import ensure_calibration, load_calibration
from .config import ConfigBundle, OptionsStrategyConfig
from .data import ContractMeta, IBKRHistoricalData, load_backtest_series
from .models import BacktestResult, Bar, EquityPoint, OptionTrade, summarize
from .strategy import OptionPackageStrategy, TradeSpec
from .synth import IVSurfaceParams, black_76, black_scholes, iv_atm, iv_for_strike, mid_edge_quote
from ..chart_data.cache import series_cache_service
from ..chart_data.series import BarSeriesSignature
from ..engines.signals import EmaDecisionEngine, EmaDecisionSnapshot, SupertrendEngine
from ..option_package import (
    OptionPackage,
    OptionProductFacts,
    ResolvedOptionLeg,
    option_package_debit_value,
    option_package_risk,
    option_product_facts,
    option_profit_target_hit,
    option_stop_loss_hit,
)
from ..spot.gates import apply_regime_gate, flip_exit_allowed, signal_filters_ok
from ..engine import (
    _trade_date,
    annualized_ewma_vol,
    cooldown_ok_by_index,
    realized_vol_from_closes,
)
from ..signals import bar_sizes_equal, ema_next, ema_periods, parse_bar_size
from ..utils.date_utils import business_days_until


_OPTIONS_MARKET_TAPE_NAMESPACE = "options.market_tape.v1"
_OPTIONS_SIGNAL_TAPE_NAMESPACE = "options.signal_tape.v1"
_OPTIONS_TAPE_NAMESPACE = "options.prepared_tape.v2"
_OPTIONS_TAPE_CACHE = series_cache_service()


@dataclass(frozen=True)
class _OptionsMarketTape:
    revision: BarSeriesSignature
    bars: tuple[Bar, ...]
    trade_dates: tuple[date, ...]
    bars_in_day: tuple[int, ...]
    is_last_bar: tuple[bool, ...]
    realized_vol: tuple[float, ...]
    volume_ema: tuple[float | None, ...]
    volume_ema_ready: tuple[bool, ...]


@dataclass(frozen=True)
class _OptionsSignalTape:
    signals: tuple[EmaDecisionSnapshot | None, ...]
    ema_needed: bool


@dataclass(frozen=True)
class PreparedOptionsTape:
    revision: BarSeriesSignature
    bars: tuple[Bar, ...]
    trade_dates: tuple[date, ...]
    bars_in_day: tuple[int, ...]
    is_last_bar: tuple[bool, ...]
    realized_vol: tuple[float, ...]
    volume_ema: tuple[float | None, ...]
    volume_ema_ready: tuple[bool, ...]
    signals: tuple[EmaDecisionSnapshot | None, ...]
    ema_needed: bool


def prepare_options_tape(
    *,
    cfg: ConfigBundle,
    bars: list[Bar] | tuple[Bar, ...],
    data: IBKRHistoricalData,
    start_dt: datetime,
    end_dt: datetime,
) -> PreparedOptionsTape:
    """Reuse immutable market facts and signal state across option combinations."""

    if not isinstance(cfg.strategy, OptionsStrategyConfig):
        raise ValueError("prepare_options_tape requires an options strategy config")

    filters = cfg.strategy.filters
    periods = ema_periods(cfg.strategy.ema_preset)
    needs_direction = cfg.strategy.directional_legs is not None
    ema_needed = bool(
        periods is not None
        or needs_direction
        or (
            filters
            and (
                filters.ema_spread_min_pct is not None
                or filters.ema_slope_min_pct is not None
            )
        )
    )
    ema_entry_mode = getattr(cfg.strategy, "ema_entry_mode", None)
    entry_confirm_bars = int(getattr(cfg.strategy, "entry_confirm_bars", 0) or 0)
    volume_period = None
    if filters is not None and filters.volume_ratio_min is not None:
        try:
            volume_period = int(filters.volume_ema_period or 20)
        except (TypeError, ValueError):
            volume_period = 20
        volume_period = max(1, volume_period)

    revision = _OPTIONS_TAPE_CACHE.revision(bars)
    market_key = (
        revision,
        int(cfg.synthetic.rv_lookback),
        float(cfg.synthetic.rv_ewma_lambda),
        str(cfg.backtest.bar_size),
        bool(cfg.backtest.use_rth),
        volume_period,
    )
    market = _OPTIONS_TAPE_CACHE.get(
        namespace=_OPTIONS_MARKET_TAPE_NAMESPACE,
        key=market_key,
    )
    if not isinstance(market, _OptionsMarketTape):
        rows = tuple(bars)
        trade_dates = tuple(_trade_date(bar.ts) for bar in rows)
        bars_in_day: list[int] = []
        current_count = 0
        prior_day = None
        for trade_day in trade_dates:
            current_count = current_count + 1 if trade_day == prior_day else 1
            bars_in_day.append(current_count)
            prior_day = trade_day

        returns = deque(maxlen=max(0, int(cfg.synthetic.rv_lookback)))
        realized_vol: list[float] = []
        volume_values: list[float | None] = []
        volume_ready: list[bool] = []
        volume_ema = None
        volume_count = 0
        previous = None
        for bar in rows:
            if previous is not None and previous.close > 0:
                returns.append(math.log(bar.close / previous.close))
            realized_vol.append(
                annualized_ewma_vol(
                    returns,
                    lam=float(cfg.synthetic.rv_ewma_lambda),
                    bar_size=str(cfg.backtest.bar_size),
                    use_rth=bool(cfg.backtest.use_rth),
                )
            )
            if volume_period is not None:
                volume_ema = ema_next(volume_ema, float(bar.volume), volume_period)
                volume_count += 1
            volume_values.append(volume_ema)
            volume_ready.append(
                volume_period is None or volume_count >= volume_period
            )
            previous = bar

        market = _OptionsMarketTape(
            revision=revision,
            bars=rows,
            trade_dates=trade_dates,
            bars_in_day=tuple(bars_in_day),
            is_last_bar=tuple(
                idx + 1 == len(rows) or trade_dates[idx + 1] != trade_day
                for idx, trade_day in enumerate(trade_dates)
            ),
            realized_vol=tuple(realized_vol),
            volume_ema=tuple(volume_values),
            volume_ema_ready=tuple(volume_ready),
        )
        _OPTIONS_TAPE_CACHE.set(
            namespace=_OPTIONS_MARKET_TAPE_NAMESPACE,
            key=market_key,
            value=market,
        )

    regime_mode = str(
        getattr(cfg.strategy, "regime_mode", "ema") or "ema"
    ).strip().lower()
    if regime_mode not in {"ema", "supertrend"}:
        regime_mode = "ema"
    regime_preset = cfg.strategy.regime_ema_preset
    regime_bar = cfg.strategy.regime_bar_size or cfg.backtest.bar_size
    use_mtf_regime = (
        not bar_sizes_equal(regime_bar, cfg.backtest.bar_size)
        if regime_mode == "supertrend"
        else bool(regime_preset)
        and not bar_sizes_equal(regime_bar, cfg.backtest.bar_size)
    )
    regime_bars = (
        load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=cfg.strategy.symbol,
            exchange=cfg.strategy.exchange,
            start=start_dt,
            end=end_dt,
            bar_size=str(regime_bar),
            use_rth=cfg.backtest.use_rth,
        ).as_list()
        if use_mtf_regime
        else None
    )

    regime_revision = (
        _OPTIONS_TAPE_CACHE.revision(regime_bars)
        if regime_bars is not None
        else None
    )
    signal_key = (
        revision,
        regime_revision,
        cfg.strategy.ema_preset,
        ema_entry_mode,
        entry_confirm_bars,
        regime_mode,
        regime_preset,
        str(regime_bar),
        int(getattr(cfg.strategy, "supertrend_atr_period", 10) or 10),
        float(getattr(cfg.strategy, "supertrend_multiplier", 3.0) or 3.0),
        str(getattr(cfg.strategy, "supertrend_source", "hl2") or "hl2"),
        ema_needed,
    )
    signal_tape = _OPTIONS_TAPE_CACHE.get(
        namespace=_OPTIONS_SIGNAL_TAPE_NAMESPACE,
        key=signal_key,
    )
    if not isinstance(signal_tape, _OptionsSignalTape):
        signal_engine = (
            EmaDecisionEngine(
                ema_preset=str(cfg.strategy.ema_preset),
                ema_entry_mode=ema_entry_mode,
                entry_confirm_bars=entry_confirm_bars,
                regime_ema_preset=(
                    None
                    if (use_mtf_regime or regime_mode == "supertrend")
                    else cfg.strategy.regime_ema_preset
                ),
            )
            if periods is not None
            else None
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
                atr_period=int(
                    getattr(cfg.strategy, "supertrend_atr_period", 10) or 10
                ),
                multiplier=float(
                    getattr(cfg.strategy, "supertrend_multiplier", 3.0) or 3.0
                ),
                source=str(
                    getattr(cfg.strategy, "supertrend_source", "hl2") or "hl2"
                ),
            )
            if regime_mode == "supertrend"
            else None
        )
        signals: list[EmaDecisionSnapshot | None] = []
        regime_idx = 0
        last_regime = None
        last_supertrend = None
        for bar in market.bars:
            signal = (
                signal_engine.update(bar.close)
                if signal_engine is not None
                else None
            )
            if supertrend_engine is not None:
                if use_mtf_regime and regime_bars is not None:
                    while (
                        regime_idx < len(regime_bars)
                        and regime_bars[regime_idx].ts <= bar.ts
                    ):
                        regime_bar_row = regime_bars[regime_idx]
                        last_supertrend = supertrend_engine.update(
                            high=float(regime_bar_row.high),
                            low=float(regime_bar_row.low),
                            close=float(regime_bar_row.close),
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
                    regime_dir=(
                        last_supertrend.direction
                        if last_supertrend is not None
                        else None
                    ),
                    regime_ready=bool(last_supertrend and last_supertrend.ready),
                )
            elif (
                use_mtf_regime
                and signal is not None
                and regime_engine is not None
                and regime_bars is not None
            ):
                while (
                    regime_idx < len(regime_bars)
                    and regime_bars[regime_idx].ts <= bar.ts
                ):
                    last_regime = regime_engine.update(
                        regime_bars[regime_idx].close
                    )
                    regime_idx += 1
                signal = apply_regime_gate(
                    signal,
                    regime_dir=(
                        last_regime.state if last_regime is not None else None
                    ),
                    regime_ready=bool(last_regime and last_regime.ema_ready),
                )
            signals.append(signal)
        signal_tape = _OptionsSignalTape(
            signals=tuple(signals),
            ema_needed=ema_needed,
        )
        _OPTIONS_TAPE_CACHE.set(
            namespace=_OPTIONS_SIGNAL_TAPE_NAMESPACE,
            key=signal_key,
            value=signal_tape,
        )

    key = (market_key, signal_key)
    cached = _OPTIONS_TAPE_CACHE.get(namespace=_OPTIONS_TAPE_NAMESPACE, key=key)
    if isinstance(cached, PreparedOptionsTape):
        return cached
    prepared = PreparedOptionsTape(
        revision=market.revision,
        bars=market.bars,
        trade_dates=market.trade_dates,
        bars_in_day=market.bars_in_day,
        is_last_bar=market.is_last_bar,
        realized_vol=market.realized_vol,
        volume_ema=market.volume_ema,
        volume_ema_ready=market.volume_ema_ready,
        signals=signal_tape.signals,
        ema_needed=signal_tape.ema_needed,
    )
    return _OPTIONS_TAPE_CACHE.set(
        namespace=_OPTIONS_TAPE_NAMESPACE,
        key=key,
        value=prepared,
    )


def run_options_backtest(
    *,
    cfg: ConfigBundle,
    bars: list[Bar] | tuple[Bar, ...],
    meta: ContractMeta,
    data: IBKRHistoricalData,
    start_dt: datetime,
    end_dt: datetime,
) -> BacktestResult:
    if not isinstance(cfg.strategy, OptionsStrategyConfig):
        raise ValueError("run_options_backtest requires an options strategy config")

    product = option_product_facts(
        cfg.strategy.symbol,
        exchange=meta.exchange,
        multiplier=meta.multiplier,
        source="backtest_contract_meta",
    )
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

    strategy = OptionPackageStrategy(cfg.strategy)
    tape = prepare_options_tape(
        cfg=cfg,
        bars=bars,
        data=data,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    bars = tape.bars
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
    last_entry_idx = None
    for idx, bar in enumerate(bars):
        trade_day = tape.trade_dates[idx]
        if idx == 0 or tape.trade_dates[idx - 1] != trade_day:
            entries_today = 0
        rv = tape.realized_vol[idx]
        signal = tape.signals[idx]
        ema_ready = bool(signal and signal.ema_ready)

        if open_trades and prev_bar:
            still_open: list[OptionTrade] = []
            for trade in open_trades:
                if trade_day > trade.expiry:
                    exit_debit = _trade_value_from_spec(
                        trade,
                        prev_bar,
                        rv,
                        cfg,
                        surface_params,
                        meta.min_tick,
                        product,
                        mode="exit",
                        calibration=calibration,
                        trade_day=tape.trade_dates[idx - 1],
                    )
                    _close_trade(trade, prev_bar.ts, exit_debit, "expiry", trades)
                    cash += (
                        -exit_debit
                        * product.multiplier
                        * trade.package.quantity
                    ) - trade.exit_commission
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
                    product,
                    calibration,
                    trade_day=trade_day,
                )
                should_close = False
                reason = ""

                if _hit_profit(trade, current_value):
                    should_close = True
                    reason = "profit"
                elif _hit_stop(trade, current_value, cfg.strategy.stop_loss_basis):
                    should_close = True
                    reason = "stop"
                elif _hit_exit_dte(cfg, trade, trade_day):
                    should_close = True
                    reason = "exit_dte"
                elif _hit_flip_exit(cfg, trade, bar, current_value, signal):
                    should_close = True
                    reason = "flip"
                elif cfg.strategy.dte == 0 and tape.is_last_bar[idx]:
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
                        product,
                        mode="exit",
                        calibration=calibration,
                        trade_day=trade_day,
                    )
                    _close_trade(trade, bar.ts, exit_debit, reason, trades)
                    cash += (
                        -exit_debit
                        * product.multiplier
                        * trade.package.quantity
                    ) - trade.exit_commission
                    margin_used = max(0.0, margin_used - trade.margin_required)
                else:
                    still_open.append(trade)
                    liquidation += (
                        -current_value
                        * product.multiplier
                        * trade.package.quantity
                    ) - trade.exit_commission
            open_trades = still_open

        entry_signal_dir = signal.entry_dir if signal is not None else None

        ema_gate_ok = True
        ema_right_override = None
        direction = None
        if tape.ema_needed and not ema_ready:
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
            bars_in_day=tape.bars_in_day[idx],
            close=float(bar.close),
            volume=float(bar.volume),
            volume_ema=tape.volume_ema[idx],
            volume_ema_ready=tape.volume_ema_ready[idx],
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
                product,
                mode="entry",
                calibration=calibration,
                trade_day=trade_day,
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
                    product,
                    mode="mark",
                    calibration=calibration,
                    trade_day=trade_day,
                )
                package = OptionPackage(
                    product=product,
                    legs=spec.legs,
                    quantity=spec.quantity,
                    debit_value=-entry_price,
                )
                package_risk = option_package_risk(package)
                if package_risk is None:
                    continue
                scale = product.multiplier * package.quantity
                mark_liquidation = -mark_price * scale
                entry_commission = _option_commission(spec, cfg)
                exit_commission = _option_commission(spec, cfg)
                candidate = OptionTrade(
                    package=package,
                    risk=package_risk,
                    entry_time=bar.ts,
                    stop_loss=cfg.strategy.stop_loss,
                    profit_target=cfg.strategy.profit_target,
                    margin_required=(
                        package_risk.max_loss
                        + entry_commission
                        + exit_commission
                        if entry_price > 0
                        else 0.0
                    ),
                    entry_commission=entry_commission,
                    exit_commission=exit_commission,
                )
                cash_after = (
                    cash
                    + (entry_price * scale)
                    - candidate.entry_commission
                )
                margin_after = margin_used + candidate.margin_required
                equity_after = (
                    cash_after
                    + liquidation
                    + mark_liquidation
                    - candidate.exit_commission
                )
                if cash_after >= 0 and equity_after >= margin_after:
                    open_trades.append(candidate)
                    cash = cash_after
                    margin_used = margin_after
                    entries_today += 1
                    last_entry_idx = idx
                    liquidation += mark_liquidation - candidate.exit_commission
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
                product,
                mode="exit",
                calibration=calibration,
                trade_day=tape.trade_dates[-1],
            )
            _close_trade(trade, prev_bar.ts, exit_debit, "end", trades)
            cash += (
                -exit_debit
                * product.multiplier
                * trade.package.quantity
            ) - trade.exit_commission
            margin_used = max(0.0, margin_used - trade.margin_required)

    summary = summarize(
        trades,
        cfg.backtest.starting_cash,
        equity_curve,
        product.multiplier,
    )
    return BacktestResult(trades=trades, equity=equity_curve, summary=summary)


def _rv_from_bars(
    bars: list[Bar] | tuple[Bar, ...],
    cfg: ConfigBundle,
) -> float:
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


def _direction_from_legs(legs: tuple[ResolvedOptionLeg, ...]) -> str | None:
    if not legs:
        return None
    pair = (legs[0].action.upper(), legs[0].right.upper())
    if pair in (("BUY", "CALL"), ("SELL", "PUT")):
        return "up"
    if pair in (("BUY", "PUT"), ("SELL", "CALL")):
        return "down"
    return None


def _option_commission(
    spec: TradeSpec | OptionTrade,
    cfg: ConfigBundle,
) -> float:
    quantity = (
        spec.quantity
        if isinstance(spec, TradeSpec)
        else spec.package.quantity
    )
    return (
        float(getattr(cfg.synthetic, "commission_per_contract", 0.0))
        * int(quantity)
        * sum(int(leg.ratio) for leg in spec.legs)
    )


def _option_fill_slippage(
    value: float,
    *,
    mode: str,
    cfg: ConfigBundle,
    min_tick: float,
) -> float:
    if mode == "mark":
        return float(value)
    slippage = (
        float(getattr(cfg.synthetic, "slippage_ticks", 0.0))
        * float(min_tick)
    )
    return float(value) - slippage if mode == "entry" else float(value) + slippage


def _trade_value(
    trade: OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    product: OptionProductFacts,
    calibration,
    *,
    trade_day: date | None = None,
) -> float:
    return _trade_value_from_spec(
        trade,
        bar,
        rv,
        cfg,
        surface_params,
        min_tick,
        product,
        mode="mark",
        calibration=calibration,
        trade_day=trade_day,
    )


def _trade_value_from_spec(
    spec: TradeSpec | OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    product: OptionProductFacts,
    mode: str,
    calibration,
    trade_day: date | None = None,
) -> float:
    trade_day = trade_day or _trade_date(bar.ts)
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
            if product.pricing_model == "black_76"
            else black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        )
        mid_rows.append(("SELL" if leg.action == "SELL" else "BUY", leg.ratio, price))

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
        return _option_fill_slippage(
            -float(debit_value),
            mode=mode,
            cfg=cfg,
            min_tick=min_tick,
        )

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
        value = bid_signed if net_mid >= 0 else ask_signed
    else:
        value = ask_signed if net_mid >= 0 else bid_signed
    return _option_fill_slippage(
        value,
        mode=mode,
        cfg=cfg,
        min_tick=min_tick,
    )


def _hit_profit(trade: OptionTrade, current_value: float) -> bool:
    return option_profit_target_hit(
        entry_value=trade.entry_price,
        current_value=current_value,
        profit_target=trade.profit_target,
    )


def _hit_stop(trade: OptionTrade, current_value: float, basis: str) -> bool:
    max_loss = trade.max_loss if basis != "credit" else None
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
