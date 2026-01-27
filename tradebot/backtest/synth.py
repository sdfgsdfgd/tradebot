"""Synthetic IV and option pricing helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass

# Re-export canonical EWMA volatility implementation (shared with UI + engine).
from ..engine import ewma_vol


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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
