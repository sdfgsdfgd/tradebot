"""Synthetic IV and option pricing helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time

# Re-export canonical EWMA volatility implementation (shared with UI + engine).
from ..engine import _trade_date, _ts_to_et, ewma_vol
from ..engines.market import is_early_close_day
from ..signals import parse_bar_size


@dataclass(frozen=True)
class IVSurfaceParams:
    rv_lookback: int
    rv_ewma_lambda: float
    iv_risk_premium: float
    iv_floor: float
    term_slope: float
    skew: float


@dataclass(frozen=True)
class Quote:
    mid: float
    bid: float
    ask: float


def iv_atm(rv: float, dte_days: int, params: IVSurfaceParams) -> float:
    base = max(params.iv_floor, rv * params.iv_risk_premium)
    if dte_days <= 0:
        return base
    t = dte_days / 365.0
    return base * (1 + params.term_slope * math.sqrt(t))


def iv_for_strike(atm_iv: float, forward: float, strike: float, params: IVSurfaceParams) -> float:
    if forward <= 0 or strike <= 0:
        return atm_iv
    moneyness = math.log(strike / forward)
    return max(params.iv_floor, atm_iv * (1 + params.skew * moneyness))


def black_scholes(spot: float, strike: float, t: float, rate: float, vol: float, right: str) -> float:
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike) if right == "CALL" else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if right == "CALL":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    return strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def black_76(forward: float, strike: float, t: float, rate: float, vol: float, right: str) -> float:
    if t <= 0 or vol <= 0 or forward <= 0 or strike <= 0:
        return max(0.0, forward - strike) if right == "CALL" else max(0.0, strike - forward)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    df = math.exp(-rate * t)
    if right == "CALL":
        return df * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return df * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def mid_edge_quote(mid: float, min_spread_pct: float, min_tick: float) -> Quote:
    spread = max(min_tick, abs(mid) * min_spread_pct)
    bid = max(0.0, mid - spread / 2)
    ask = mid + spread / 2
    return Quote(mid=mid, bid=bid, ask=ask)


def _option_commission(spec: object, cfg: object) -> float:
    package = getattr(spec, "package", None)
    quantity = getattr(spec, "quantity", None)
    legs = getattr(spec, "legs", None)
    if package is not None:
        quantity = package.quantity
        legs = package.legs
    return (
        float(getattr(cfg.synthetic, "commission_per_contract", 0.0))
        * int(quantity)
        * sum(int(leg.ratio) for leg in legs)
    )


def _option_fill_slippage(
    value: float,
    *,
    mode: str,
    cfg: object,
    min_tick: float,
) -> float:
    if mode == "mark":
        return float(value)
    slippage = float(getattr(cfg.synthetic, "slippage_ticks", 0.0)) * float(min_tick)
    return float(value) - slippage if mode == "entry" else float(value) + slippage


def _option_time_to_expiry_years(
    *,
    expiry: date,
    bar: object,
    cfg: object,
    product: object,
) -> float:
    bar_def = parse_bar_size(cfg.backtest.bar_size)
    bar_hours = bar_def.duration.total_seconds() / 3600.0 if bar_def is not None else 1.0
    min_time = bar_hours / (24.0 * 365.0)
    if product.security_type != "OPT":
        dte_days = max((expiry - _trade_date(bar.ts)).days, 0)
        session_hours = 6.5 if cfg.backtest.use_rth else 24.0
        return max(
            session_hours / (24.0 * 365.0)
            if dte_days == 0
            else dte_days / 365.0,
            min_time,
        )

    bar_et = _ts_to_et(bar.ts)
    close_et = time(13, 0) if is_early_close_day(expiry) else time(16, 0)
    expiry_et = datetime.combine(expiry, close_et, tzinfo=bar_et.tzinfo)
    seconds = max(0.0, (expiry_et - bar_et).total_seconds())
    return max(seconds / (365.0 * 24.0 * 3600.0), min_time)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
