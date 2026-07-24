"""Cost-adjusted research evidence shared by backtests and evaluators."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time

from ..backtest.models import BacktestResult
from ..chart_data.series import OhlcvBar
from ..time_utils import to_et


SCORE_VERSION = "research.daily.v1"
XSP_CREDIT_BARRIER_SCHEMA = "xsp.credit-barrier-census.v1"

_XSP_BARRIER_TIMES = (time(10), time(10, 30), time(11), time(11, 30))
_XSP_BARRIER_OFFSETS = (0.0025, 0.005, 0.0075, 0.01)
_XSP_BARRIER_HORIZONS = (0, 1, 3, 5)
_XSP_BARRIER_SIDES = ("put_credit", "call_credit")
_XSP_BARRIER_FRICTION = 10.0
_XSP_MULTIPLIER = 100.0


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


def _wilson_upper(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 1.0
    rate = successes / total
    scale = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    )
    return min(1.0, (center + radius) / scale)


def xsp_credit_barrier_census(
    bars: Iterable[OhlcvBar],
    *,
    source_fingerprint: str,
) -> dict[str, object]:
    """Measure preregistered XSP short-strike barriers; never infer option PnL."""

    values = tuple(bars)
    if not values:
        raise ValueError("XSP credit-barrier census requires an admitted bar tape")
    if not str(source_fingerprint).strip():
        raise ValueError("XSP credit-barrier census requires a source fingerprint")

    by_day: dict[date, list[tuple[datetime, OhlcvBar]]] = defaultdict(list)
    for bar in values:
        et = to_et(bar.ts)
        by_day[et.date()].append((et, bar))
    days = sorted(by_day)
    for rows in by_day.values():
        rows.sort(key=lambda row: row[0])

    # Path extrema depend only on boundary and horizon, not strike distance or
    # side. Compute them once so the fixed 128-cell census stays cheap.
    paths: dict[
        tuple[time, int],
        list[tuple[str, float, float, float, float]],
    ] = {}
    for boundary in _XSP_BARRIER_TIMES:
        for horizon in _XSP_BARRIER_HORIZONS:
            samples: list[tuple[str, float, float, float, float]] = []
            for index, day in enumerate(days):
                expiry_index = index + horizon
                if expiry_index >= len(days):
                    continue
                decision = next(
                    (
                        bar
                        for et, bar in by_day[day]
                        if et.timetz().replace(tzinfo=None) == boundary
                    ),
                    None,
                )
                if decision is None:
                    continue
                path = [
                    bar
                    for path_index in range(index, expiry_index + 1)
                    for et, bar in by_day[days[path_index]]
                    if path_index != index
                    or et.timetz().replace(tzinfo=None) > boundary
                ]
                if not path:
                    continue
                samples.append(
                    (
                        str(day.year),
                        float(decision.close),
                        float(by_day[days[expiry_index]][-1][1].close),
                        min(float(bar.low) for bar in path),
                        max(float(bar.high) for bar in path),
                    )
                )
            if not samples:
                raise ValueError(
                    f"XSP tape has no eligible {boundary:%H:%M}/+{horizon} session paths"
                )
            paths[(boundary, horizon)] = samples

    cells: list[dict[str, object]] = []
    for boundary in _XSP_BARRIER_TIMES:
        for offset in _XSP_BARRIER_OFFSETS:
            for horizon in _XSP_BARRIER_HORIZONS:
                samples = paths[(boundary, horizon)]
                for side in _XSP_BARRIER_SIDES:
                    touches = breaches = 0
                    actual_offsets: list[float] = []
                    worst_beyond = 0.0
                    annual_counts: dict[str, list[int]] = defaultdict(
                        lambda: [0, 0]
                    )
                    for year, spot, expiry_close, path_low, path_high in samples:
                        target = spot * (
                            1.0 - offset if side == "put_credit" else 1.0 + offset
                        )
                        short_strike = float(
                            math.ceil(target)
                            if side == "put_credit"
                            else math.floor(target)
                        )
                        if side == "put_credit":
                            touched = path_low <= short_strike
                            breached = expiry_close <= short_strike
                            beyond = max(0.0, short_strike - path_low)
                            actual_offset = (spot - short_strike) / spot
                        else:
                            touched = path_high >= short_strike
                            breached = expiry_close >= short_strike
                            beyond = max(0.0, path_high - short_strike)
                            actual_offset = (short_strike - spot) / spot
                        touches += touched
                        breaches += breached
                        annual_counts[year][0] += 1
                        annual_counts[year][1] += breached
                        actual_offsets.append(actual_offset)
                        worst_beyond = max(worst_beyond, beyond)

                    total = len(samples)
                    breach_upper = _wilson_upper(breaches, total)
                    annual = {}
                    for year in sorted(annual_counts):
                        year_total, year_breaches = annual_counts[year]
                        annual[year] = {
                            "observations": year_total,
                            "breaches": year_breaches,
                            "breach_rate": year_breaches / year_total,
                            "breach_rate_upper95": _wilson_upper(
                                year_breaches, year_total
                            ),
                        }
                    cells.append(
                        {
                            "decision_time_et": boundary.strftime("%H:%M"),
                            "offset_pct": offset * 100.0,
                            "horizon_sessions": horizon,
                            "side": side,
                            "observations": total,
                            "touches": touches,
                            "touch_rate": touches / total,
                            "touch_rate_upper95": _wilson_upper(touches, total),
                            "expiration_breaches": breaches,
                            "expiration_breach_rate": breaches / total,
                            "expiration_breach_rate_upper95": breach_upper,
                            "required_credit_price": (
                                breach_upper
                                + _XSP_BARRIER_FRICTION / _XSP_MULTIPLIER
                            ),
                            "mean_actual_offset_pct": (
                                statistics.fmean(actual_offsets) * 100.0
                            ),
                            "worst_beyond_short_points": worst_beyond,
                            "annual": annual,
                        }
                    )

    return {
        "schema": XSP_CREDIT_BARRIER_SCHEMA,
        "source": {
            "symbol": "XSP",
            "bar_size": "5 mins",
            "use_rth": True,
            "start": days[0].isoformat(),
            "end": days[-1].isoformat(),
            "bars": len(values),
            "sessions": len(days),
            "stitched_source_manifest_sha256": source_fingerprint,
        },
        "contract": {
            "decision_times_et": [
                value.strftime("%H:%M") for value in _XSP_BARRIER_TIMES
            ],
            "offset_pct": [value * 100.0 for value in _XSP_BARRIER_OFFSETS],
            "horizon_sessions": list(_XSP_BARRIER_HORIZONS),
            "sides": list(_XSP_BARRIER_SIDES),
            "strike_rounding": "toward_spot_whole_xsp_point",
            "width_points": 1.0,
            "round_trip_friction_usd": _XSP_BARRIER_FRICTION,
            "required_credit_formula": "wilson95_expiration_breach_rate + 0.10",
            "authority": "underlying_risk_screen_only",
        },
        "cells": cells,
    }
