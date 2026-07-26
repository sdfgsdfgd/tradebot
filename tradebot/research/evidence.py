"""Cost-adjusted research evidence shared by backtests and evaluators."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from ..backtest.models import BacktestResult
from ..chart_data.series import OhlcvBar
from ..engines.directional_impulse import (
    DirectionalImpulseEngine,
    DirectionalTurnPolicy,
)
from ..time_utils import to_et


SCORE_VERSION = "research.daily.v1"
XSP_CREDIT_BARRIER_SCHEMA = "xsp.credit-barrier-census.v1"
XSP_DIRECTIONAL_TURN_SCHEMA = "xsp.directional-turn-census.v1"

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


def xsp_directional_turn_census(
    bars: Iterable[OhlcvBar],
    *,
    source_fingerprint: str,
    include_session_ledger: bool = False,
) -> dict[str, object]:
    """Score the production XSP turn sensor against hindsight labels.

    Material extrema are research-only labels. Direction and event timing come
    exclusively from the causal engine used by live and backtest.
    """

    values = tuple(bars)
    if not values:
        raise ValueError("XSP turn census requires an admitted bar tape")
    if not str(source_fingerprint).strip():
        raise ValueError("XSP turn census requires a source fingerprint")

    policy = DirectionalTurnPolicy()
    by_day: dict[date, list[tuple[datetime, OhlcvBar]]] = defaultdict(list)
    for bar in values:
        et = to_et(bar.ts, naive_ts_mode="utc")
        by_day[et.date()].append((et, bar))
    for rows in by_day.values():
        rows.sort(key=lambda row: row[0])

    totals = {"labels": 0, "events": 0, "matches": 0}
    lags: list[int] = []
    absolute = {"labels": 0, "matches": 0, "boundary_censored": 0}
    coverage = {
        "complete_turn_window_sessions": 0,
        "incomplete_turn_window_sessions": 0,
        "events_below_required_horizons": 0,
    }
    event_horizons: dict[int, int] = defaultdict(int)
    ledger: list[dict[str, object]] = []

    for day, day_bars in sorted(by_day.items()):
        turn_window = [
            et
            for et, _bar in day_bars
            if policy.start_et <= et.time() <= policy.end_et
        ]
        complete_turn_window = bool(
            len(turn_window) == 28
            and turn_window[0].time() == policy.start_et
            and turn_window[-1].time() == policy.end_et
            and all(
                right - left == policy.bar_duration
                for left, right in zip(turn_window, turn_window[1:])
            )
        )
        coverage[
            (
                "complete_turn_window_sessions"
                if complete_turn_window
                else "incomplete_turn_window_sessions"
            )
        ] += 1
        engine = DirectionalImpulseEngine(
            horizons=(1, 3, 6, 12, 24),
            bar_duration=timedelta(minutes=5),
            turn_policy=policy,
        )
        rows = []
        events: list[dict[str, object]] = []
        for index, (et, bar) in enumerate(day_bars):
            snapshot = engine.update(
                ts=et,
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                session_key=day,
            )
            tr_pct = (
                float(snapshot.horizons[-1].tr_mean_pct)
                if snapshot.horizons
                else (
                    max(0.0, float(bar.high) - float(bar.low))
                    / max(float(bar.close), 1e-9)
                    * 100.0
                )
            )
            rows.append((index, et, bar, snapshot, tr_pct))
            if snapshot.turn_event in ("up", "down"):
                event_horizons[snapshot.observed_horizons] += 1
                coverage["events_below_required_horizons"] += (
                    snapshot.observed_horizons
                    < policy.min_observed_horizons
                )
                events.append(
                    {
                        "direction": snapshot.turn_event,
                        "index": index,
                        "time_et": et.strftime("%H:%M"),
                        "matched": False,
                        "evidence": {
                            "direction_score": snapshot.direction_score,
                            "smoothed_direction_score": (
                                snapshot.smoothed_direction_score
                            ),
                            "coherence": snapshot.coherence,
                            "conviction": snapshot.conviction,
                            "observed_horizons": snapshot.observed_horizons,
                            "atr_ratio": snapshot.atr_ratio,
                            "atr_velocity_pct": snapshot.atr_velocity_pct,
                            "atr_acceleration_pct": (
                                snapshot.atr_acceleration_pct
                            ),
                            "retrace_atr": snapshot.retrace_atr,
                            "horizons": [
                                row.as_payload() for row in snapshot.horizons
                            ],
                        },
                    }
                )

        for event in events:
            index = int(event["index"])
            direction = str(event["direction"])
            sign = 1.0 if direction == "up" else -1.0
            base = float(rows[index][2].close)
            paths: dict[str, object] = {}
            for horizon in (1, 3, 6, 12, 24):
                future = rows[index + 1 : index + horizon + 1]
                if not future:
                    continue
                closes = [float(row[2].close) for row in future]
                highs = [float(row[2].high) for row in future]
                lows = [float(row[2].low) for row in future]
                favorable = (
                    max(highs) - base
                    if direction == "up"
                    else base - min(lows)
                )
                adverse = (
                    base - min(lows)
                    if direction == "up"
                    else max(highs) - base
                )
                paths[str(horizon)] = {
                    "observations": len(future),
                    "directed_close_points": sign * (closes[-1] - base),
                    "directed_mfe_points": max(0.0, favorable),
                    "directed_mae_points": max(0.0, adverse),
                }
            event["forward_paths"] = paths

        labels: list[dict[str, object]] = []
        for index, et, bar, _snapshot, tr_pct in rows:
            if not time(9, 45) <= et.time() <= time(11, 15):
                continue
            left = max(0, index - 3)
            right = min(len(rows), index + 4)
            neighborhood = rows[left:right]
            future = rows[index + 1 : right]
            if not future:
                continue
            up_excursion = (
                max(float(row[2].high) for row in future) - float(bar.low)
            ) / float(bar.close) * 100.0
            down_excursion = (
                float(bar.high) - min(float(row[2].low) for row in future)
            ) / float(bar.close) * 100.0
            threshold = max(float(tr_pct), 0.03)
            if (
                float(bar.low)
                <= min(float(row[2].low) for row in neighborhood)
                and up_excursion >= threshold
            ):
                labels.append(
                    {
                        "direction": "up",
                        "index": index,
                        "time_et": et.strftime("%H:%M"),
                        "excursion_pct": up_excursion,
                    }
                )
            if (
                float(bar.high)
                >= max(float(row[2].high) for row in neighborhood)
                and down_excursion >= threshold
            ):
                labels.append(
                    {
                        "direction": "down",
                        "index": index,
                        "time_et": et.strftime("%H:%M"),
                        "excursion_pct": down_excursion,
                    }
                )

        deduplicated: list[dict[str, object]] = []
        for label in labels:
            if (
                deduplicated
                and deduplicated[-1]["direction"] == label["direction"]
                and int(label["index"]) - int(deduplicated[-1]["index"]) <= 3
            ):
                old = deduplicated[-1]
                old_bar = rows[int(old["index"])][2]
                new_bar = rows[int(label["index"])][2]
                more_extreme = (
                    float(new_bar.low) < float(old_bar.low)
                    if label["direction"] == "up"
                    else float(new_bar.high) > float(old_bar.high)
                )
                if more_extreme:
                    deduplicated[-1] = label
            else:
                deduplicated.append(label)

        used_events: set[int] = set()
        for label in deduplicated:
            candidates = [
                (event_index, event)
                for event_index, event in enumerate(events)
                if event_index not in used_events
                and event["direction"] == label["direction"]
                and -1
                <= int(event["index"]) - int(label["index"])
                <= 3
            ]
            if not candidates:
                label["matched_event_time_et"] = None
                label["lag_bars"] = None
                continue
            event_index, event = min(
                candidates,
                key=lambda item: abs(
                    int(item[1]["index"]) - int(label["index"])
                ),
            )
            lag = int(event["index"]) - int(label["index"])
            used_events.add(event_index)
            event["matched"] = True
            label["matched_event_time_et"] = event["time_et"]
            label["lag_bars"] = lag
            lags.append(lag)

        early = [
            row
            for row in rows
            if time(9, 30) <= row[1].time() <= time(11, 30)
        ]
        absolute_rows = []
        if early:
            for direction, row in (
                ("up", min(early, key=lambda item: float(item[2].low))),
                ("down", max(early, key=lambda item: float(item[2].high))),
            ):
                index, et, _bar, _snapshot, _tr_pct = row
                candidates = [
                    event
                    for event in events
                    if event["direction"] == direction
                    and -1 <= int(event["index"]) - index <= 3
                ]
                match = (
                    min(
                        candidates,
                        key=lambda event: abs(int(event["index"]) - index),
                    )
                    if candidates
                    else None
                )
                boundary_censored = (
                    et.time() <= time(9, 35) or et.time() >= time(11, 25)
                )
                absolute_rows.append(
                    {
                        "direction": direction,
                        "time_et": et.strftime("%H:%M"),
                        "matched_event_time_et": (
                            match["time_et"] if match is not None else None
                        ),
                        "boundary_censored": boundary_censored,
                    }
                )
                absolute["labels"] += 1
                absolute["matches"] += match is not None
                absolute["boundary_censored"] += boundary_censored

        totals["labels"] += len(deduplicated)
        totals["events"] += len(events)
        totals["matches"] += len(used_events)
        if include_session_ledger:
            ledger.append(
                {
                    "date": day.isoformat(),
                    "material_extrema": deduplicated,
                    "absolute_extrema": absolute_rows,
                    "events": events,
                }
            )

    precision = (
        totals["matches"] / totals["events"] if totals["events"] else 0.0
    )
    recall = totals["matches"] / totals["labels"] if totals["labels"] else 0.0
    return {
        "schema": XSP_DIRECTIONAL_TURN_SCHEMA,
        "source": {
            "symbol": "XSP",
            "bar_size": "5 mins",
            "use_rth": True,
            "start": min(by_day).isoformat(),
            "end": max(by_day).isoformat(),
            "bars": len(values),
            "sessions": len(by_day),
            "stitched_source_manifest_sha256": source_fingerprint,
        },
        "contract": {
            "authority": "observation_only",
            "direction_owner": "xsp_native",
            "spy_role": "diagnostic_only",
            "horizons_minutes": [5, 15, 30, 60, 120],
            "turn_window_et": [
                policy.start_et.strftime("%H:%M"),
                policy.end_et.strftime("%H:%M"),
            ],
            "label_window_et": ["09:45", "11:15"],
            "match_lag_bars": [-1, 3],
            "labeling": "hindsight_material_local_extrema_scoring_only",
            "forward_path_horizons_bars": [1, 3, 6, 12, 24],
            "forward_path_authority": "hindsight_diagnostics_only",
            "policy": {
                "smooth_alpha": policy.smooth_alpha,
                "initial_score": policy.initial_score,
                "turn_score": policy.turn_score,
                "retrace_atr": policy.retrace_atr,
                "min_state_bars": policy.min_state_bars,
                "cooldown_bars": policy.cooldown_bars,
                "min_observed_horizons": policy.min_observed_horizons,
            },
        },
        "material_extrema": {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "median_lag_bars": statistics.median(lags) if lags else None,
        },
        "absolute_extrema": {
            **absolute,
            "recall": (
                absolute["matches"] / absolute["labels"]
                if absolute["labels"]
                else 0.0
            ),
        },
        "coverage": {
            **coverage,
            "event_horizon_counts": {
                str(horizons): count
                for horizons, count in sorted(event_horizons.items())
            },
        },
        "sessions": ledger,
    }
