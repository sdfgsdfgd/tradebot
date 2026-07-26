"""Falsify dense XSP directional admission through causal scale interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

import tradebot.backtest.engine as backtest_engine
from tradebot.backtest.data import ContractMeta
from tradebot.backtest.engine import (
    _run_spot_backtest_exec_loop,
    _spot_prepare_summary_series_pack,
)
from tradebot.backtest.spot_tape import (
    PreparedSpotEvaluatorTape,
    prepare_spot_evaluator_tape,
)
from tradebot.chart_data.history import (
    load_history_window,
    normalize_bars_to_close,
    read_cache,
)
from tradebot.engine import _trade_date
from tradebot.research.xsp_candidate import (
    XSP_OPENING_EDGE_ADMISSION,
    XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
    XSP_OPENING_EDGE_POLICY,
    xsp_opening_edge_bundle,
)
from tradebot.research.evidence import xsp_directional_turn_census
from tradebot.research.spot_sweeps.support import _bundle_base, _mk_filters
from tradebot.spot.entry_control import SpotEntryControlPlan
from tradebot.time_utils import to_et


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "db/XSP/XSP_2021-07-26_2024-07-23_5mins_rth.csv"
OUTPUT = ROOT / "backtests/out/xsp/xsp_directional_interaction_admission_20260726.json"
META = ContractMeta(symbol="XSP", exchange="CBOE", multiplier=1.0, min_tick=0.01)
MIN_DENSITY = 234 / 363
TARGET_ADMISSIONS = (245, 300, 360, 450, 600)
REGULARIZATIONS = (1.0, 10.0, 40.0)
OBJECTIVES = ("positive_trade", "trail_capture", "net_points")
LIFECYCLE = {
    "initial_stop_atr": 0.75,
    "trail_activate_atr": 1.0,
    "trail_distance_atr": 0.5,
    "breakeven_atr": 0.0,
    "fizzle": "12x0.50",
    "fizzle_bars": 12,
    "fizzle_mfe_atr": 0.5,
    "max_hold_bars": 24,
    "cooldown_bars": 0,
}
PARTITIONS = {
    "train": (date(2021, 7, 26), date(2022, 12, 30)),
    "audit": (date(2023, 1, 1), date(2023, 12, 31)),
    "holdout": (date(2024, 1, 1), date(2024, 7, 23)),
}
BASE_FEATURES = (
    "direction",
    "minute",
    "ordinal",
    "observed",
    "direction_aligned",
    "smoothed_aligned",
    "coherence",
    "conviction",
    "retrace_atr",
    "atr_ratio",
    "atr_ratio_present",
    "atr_velocity",
    "atr_acceleration",
    "trend_aligned",
    "state_age",
) + tuple(
    name
    for horizon in (1, 3, 6, 12, 24)
    for name in (
        f"h{horizon}_present",
        f"h{horizon}_slope_tr",
        f"h{horizon}_velocity_tr",
        f"h{horizon}_velocity_present",
        f"h{horizon}_efficiency",
        f"h{horizon}_tr",
    )
)
INTERACTION_FEATURES = (
    "slow_permission",
    "slow_permission_floor",
    "fast_slope",
    "fast_velocity",
    "slow_disorder",
    "fast_disorder",
    "atr_compression",
    "atr_reacceleration",
    "atr_deceleration",
    "cascade_match",
    "cascade_short_to_long",
    "cascade_long_to_short",
    "slow_x_fast_slope",
    "slow_x_fast_velocity",
    "slow_x_reacceleration",
    "fast_slope_x_reacceleration",
    "fast_velocity_x_reacceleration",
    "compression_x_reacceleration",
    "deceleration_x_reacceleration",
    "slow_x_compression_x_reacceleration",
    "fast_velocity_x_compression_x_reacceleration",
)
FEATURE_NAMES = BASE_FEATURES + INTERACTION_FEATURES


@dataclass(frozen=True)
class Part:
    name: str
    bars: tuple
    dates: tuple[date, ...]
    pack: Any
    tape: PreparedSpotEvaluatorTape


@dataclass(frozen=True)
class Model:
    objective: str
    regularization: float
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @property
    def identity(self) -> str:
        return f"{self.objective}:l2={self.regularization:g}"

    def score(self, values: np.ndarray) -> float:
        normalized = (values - self.mean) / self.scale
        raw = float(self.weights[0] + normalized @ self.weights[1:])
        if self.objective == "net_points":
            return raw
        return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, raw))))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(
    *,
    lifecycle: dict[str, object] | None = LIFECYCLE,
    flip_hold_bars: int = 0,
    primary_ema: str | None = None,
    max_entries_per_day: int = 4,
    directional_impulse_admission: dict[str, object] | None = None,
):
    base = _bundle_base(
        symbol="XSP",
        start=PARTITIONS["train"][0],
        end=PARTITIONS["holdout"][1],
        bar_size="5 mins",
        use_rth=True,
        cache_dir=Path("/private/tmp/xsp-directional-interaction-cache"),
        offline=True,
        filters=_mk_filters(cooldown_bars=0),
        entry_signal="directional_impulse",
        spot_profit_target_pct=None,
        spot_stop_loss_pct=None,
        spot_close_eod=True,
    )
    return replace(
        base,
        strategy=replace(
            base.strategy,
            regime_mode="ema" if primary_ema else "off",
            regime_ema_preset=primary_ema,
            directional_impulse_admission=directional_impulse_admission,
            regime2_mode="off",
            max_entries_per_day=int(max_entries_per_day),
            spot_entry_fill_mode="next_open",
            spot_flip_exit_fill_mode="next_open",
            spot_controlled_flip=True,
            exit_on_signal_flip=True,
            flip_exit_min_hold_bars=int(flip_hold_bars),
            flip_exit_only_if_profit=False,
            spot_intrabar_exits=True,
            spot_spread=0.0,
            spot_commission_per_share=0.05,
            spot_commission_min=0.0,
            spot_slippage_per_share=0.0,
            spot_mark_to_market="close",
            spot_drawdown_mode="intrabar",
            spot_sizing_mode="fixed",
            spot_excursion_exit=lifecycle,
        ),
    )


CFG = config()
ALL_BARS = tuple(read_cache(SOURCE))


def build_range(
    name: str,
    start: date,
    end: date,
    source_bars: tuple = ALL_BARS,
    *,
    cfg=CFG,
) -> Part:
    bars = tuple(
        normalize_bars_to_close(
            (
                bar
                for bar in source_bars
                if start <= _trade_date(bar.ts) <= end
            ),
            symbol="XSP",
            bar_size="5 mins",
            use_rth=True,
        )
    )
    dates = tuple(sorted({_trade_date(bar.ts) for bar in bars}))
    _, pack = _spot_prepare_summary_series_pack(
        cfg=cfg,
        signal_bars=bars,
        exec_bars=bars,
    )
    tape = prepare_spot_evaluator_tape(
        cfg=cfg,
        signal_bars=bars,
        exec_bars=bars,
        sig_idx_by_exec_idx=pack.align.sig_idx_by_exec_idx,
        exec_dates=pack.exec_dates,
    )
    return Part(name, bars, dates, pack, tape)


def build_part(name: str) -> Part:
    return build_range(name, *PARTITIONS[name])


def candidate_events(part: Part) -> tuple[tuple[Any, int], ...]:
    ordinals: dict[date, int] = defaultdict(int)
    rows = []
    for snapshot in part.tape.signals:
        impulse = getattr(snapshot, "directional_impulse", None)
        if (
            snapshot is None
            or snapshot.entry_dir not in ("up", "down")
            or getattr(impulse, "turn_event", None) not in ("up", "down")
        ):
            continue
        day = _trade_date(snapshot.bar_ts)
        ordinals[day] += 1
        rows.append((snapshot, ordinals[day]))
    return tuple(rows)


def feature_values(snapshot, ordinal: int) -> np.ndarray:
    impulse = snapshot.directional_impulse
    sign = 1.0 if snapshot.entry_dir == "up" else -1.0
    current = to_et(snapshot.bar_ts, naive_ts_mode="utc")
    minute = (current.hour * 60 + current.minute) - (9 * 60 + 30)
    trend = (
        1.0
        if impulse.trend_state == snapshot.entry_dir
        else -1.0
        if impulse.trend_state in ("up", "down")
        else 0.0
    )
    values = [
        sign,
        float(minute),
        float(ordinal),
        float(impulse.observed_horizons),
        sign * float(impulse.direction_score or 0.0),
        sign * float(impulse.smoothed_direction_score or 0.0),
        float(impulse.coherence or 0.0),
        float(impulse.conviction or 0.0),
        float(impulse.retrace_atr or 0.0),
        float(impulse.atr_ratio or 0.0),
        float(impulse.atr_ratio is not None),
        float(impulse.atr_velocity_pct or 0.0),
        float(impulse.atr_acceleration_pct or 0.0),
        trend,
        float(impulse.state_age_bars or 0.0),
    ]
    horizons = {int(row.bars): row for row in impulse.horizons}
    normalized: dict[int, tuple[float, float, float]] = {}
    for horizon in (1, 3, 6, 12, 24):
        row = horizons.get(horizon)
        if row is None:
            values.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            normalized[horizon] = (0.0, 0.0, 0.0)
            continue
        true_range = max(float(row.tr_mean_pct), 1e-12)
        slope = sign * float(row.slope_pct_per_bar) / true_range
        velocity = (
            sign * float(row.slope_velocity_pct_per_bar) / true_range
            if row.slope_velocity_pct_per_bar is not None
            else 0.0
        )
        efficiency = sign * float(row.efficiency)
        values.extend(
            (
                1.0,
                slope,
                velocity,
                float(row.slope_velocity_pct_per_bar is not None),
                efficiency,
                true_range,
            )
        )
        normalized[horizon] = (slope, velocity, efficiency)

    slow_slopes = tuple(normalized[horizon][0] for horizon in (12, 24))
    fast_slopes = tuple(normalized[horizon][0] for horizon in (1, 3, 6))
    fast_velocities = tuple(normalized[horizon][1] for horizon in (1, 3, 6))
    slow_efficiencies = tuple(abs(normalized[horizon][2]) for horizon in (12, 24))
    fast_efficiencies = tuple(abs(normalized[horizon][2]) for horizon in (1, 3, 6))
    slow_permission = statistics.fmean(slow_slopes)
    slow_floor = min(slow_slopes)
    fast_slope = statistics.fmean(fast_slopes)
    fast_velocity = statistics.fmean(fast_velocities)
    slow_disorder = 1.0 - statistics.fmean(slow_efficiencies)
    fast_disorder = 1.0 - statistics.fmean(fast_efficiencies)
    ratio = float(impulse.atr_ratio or 1.0)
    acceleration = float(impulse.atr_acceleration_pct or 0.0)
    velocity = float(impulse.atr_velocity_pct or 0.0)
    compression = max(0.0, 1.0 - ratio)
    reacceleration = max(0.0, acceleration)
    deceleration = max(0.0, -velocity)
    cascade_match = float(impulse.turn_sequence_direction == snapshot.entry_dir)
    short_to_long = float(impulse.turn_sequence_order == "short_to_long")
    long_to_short = float(impulse.turn_sequence_order == "long_to_short")
    values.extend(
        (
            slow_permission,
            slow_floor,
            fast_slope,
            fast_velocity,
            slow_disorder,
            fast_disorder,
            compression,
            reacceleration,
            deceleration,
            cascade_match,
            short_to_long,
            long_to_short,
            slow_permission * fast_slope,
            slow_permission * fast_velocity,
            slow_permission * reacceleration,
            fast_slope * reacceleration,
            fast_velocity * reacceleration,
            compression * reacceleration,
            deceleration * reacceleration,
            slow_permission * compression * reacceleration,
            fast_velocity * compression * reacceleration,
        )
    )
    result = np.asarray(values, dtype=float)
    if len(result) != len(FEATURE_NAMES):
        raise AssertionError((len(result), len(FEATURE_NAMES)))
    return result


def run(
    part: Part,
    tape: PreparedSpotEvaluatorTape,
    *,
    cfg=CFG,
    capture_equity: bool = False,
):
    original = backtest_engine.prepare_spot_evaluator_tape
    backtest_engine.prepare_spot_evaluator_tape = lambda **_kwargs: tape
    try:
        return _run_spot_backtest_exec_loop(
            cfg,
            signal_bars=part.bars,
            exec_bars=part.bars,
            meta=META,
            prepared_series_pack=part.pack,
            capture_equity=bool(capture_equity),
        )
    finally:
        backtest_engine.prepare_spot_evaluator_tape = original


def metrics(part: Part, result) -> dict[str, Any]:
    daily = {day: 0.0 for day in part.dates}
    sides = {"up": 0.0, "down": 0.0}
    side_counts = {"up": 0, "down": 0}
    exits = Counter()
    pnls = []
    for trade in result.trades:
        pnl = float(trade.pnl(1.0))
        pnls.append(pnl)
        if trade.exit_time is not None:
            daily[_trade_date(trade.exit_time)] += pnl
        side = "up" if trade.qty > 0 else "down"
        sides[side] += pnl
        side_counts[side] += 1
        exits[str(trade.exit_reason or "unknown")] += 1
    daily_values = list(daily.values())
    mean = statistics.fmean(daily_values) if daily_values else 0.0
    std = statistics.stdev(daily_values) if len(daily_values) > 1 else 0.0
    lower_bound = (
        mean - 1.96 * std / math.sqrt(len(daily_values))
        if len(daily_values) > 1
        else mean
    )
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl < 0.0]
    positive_days = sorted((pnl for pnl in daily_values if pnl > 0.0), reverse=True)
    gross_positive_days = sum(positive_days)
    return {
        "sessions": len(part.dates),
        "trades": len(pnls),
        "trades_per_session": len(pnls) / len(part.dates),
        "net_pnl": sum(pnls),
        "daily_pnl_lcb95": lower_bound,
        "profit_factor": (
            sum(wins) / abs(sum(losses))
            if losses
            else None
        ),
        "max_drawdown": float(result.summary.max_drawdown),
        "top_five_positive_day_concentration": (
            sum(positive_days[:5]) / gross_positive_days
            if gross_positive_days
            else 1.0
        ),
        "direction_pnl": sides,
        "direction_count": side_counts,
        "exit_count": dict(sorted(exits.items())),
        "win_rate": sum(pnl > 0.0 for pnl in pnls) / len(pnls) if pnls else 0.0,
    }


def passes(values: dict[str, Any]) -> bool:
    profit_factor = values["profit_factor"]
    return bool(
        values["trades_per_session"] >= MIN_DENSITY
        and values["net_pnl"] > 0.0
        and values["daily_pnl_lcb95"] > 0.0
        and profit_factor is not None
        and profit_factor >= 1.10
        and values["max_drawdown"] <= 25.0
        and values["top_five_positive_day_concentration"] < 0.50
        and min(values["direction_pnl"].values()) >= 0.0
        and min(values["direction_count"].values()) > 0
    )


def labeled_training(part: Part, baseline) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    events = {
        snapshot.bar_ts + timedelta(minutes=5): (snapshot, ordinal)
        for snapshot, ordinal in candidate_events(part)
    }
    rows = []
    labels: dict[str, list[float]] = {
        "positive_trade": [],
        "trail_capture": [],
        "net_points": [],
    }
    for trade in baseline.trades:
        pair = events.get(trade.entry_time)
        if pair is None:
            continue
        pnl = float(trade.pnl(1.0))
        rows.append(feature_values(*pair))
        labels["positive_trade"].append(float(pnl > 0.0))
        labels["trail_capture"].append(float(trade.exit_reason == "trail_stop"))
        labels["net_points"].append(pnl)
    return np.vstack(rows), {
        key: np.asarray(values, dtype=float) for key, values in labels.items()
    }


def fit_model(
    objective: str,
    regularization: float,
    matrix: np.ndarray,
    target: np.ndarray,
) -> Model:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(normalized)), normalized))
    if objective == "net_points":
        penalty = np.eye(design.shape[1]) * regularization
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    else:
        weights = np.zeros(design.shape[1])
        for index in range(3000):
            raw = np.clip(design @ weights, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-raw))
            gradient = (design.T @ (probability - target)) / len(target)
            gradient[1:] += regularization * weights[1:] / len(target)
            learning_rate = 0.25 / (1.0 + index / 1200)
            weights -= learning_rate * gradient
    return Model(objective, regularization, mean, scale, weights)


def threshold_for(model: Model, events, target: int) -> float:
    scores = sorted(
        (model.score(feature_values(snapshot, ordinal)) for snapshot, ordinal in events),
        reverse=True,
    )
    index = min(max(1, target), len(scores)) - 1
    if index + 1 >= len(scores):
        return float(scores[index])
    return float((scores[index] + scores[index + 1]) / 2.0)


def project(
    part: Part,
    model: Model,
    threshold: float,
) -> tuple[PreparedSpotEvaluatorTape, list[dict[str, Any]]]:
    event_map = {id(snapshot): ordinal for snapshot, ordinal in candidate_events(part)}
    signals = []
    decisions = []
    for snapshot in part.tape.signals:
        ordinal = event_map.get(id(snapshot))
        if ordinal is None:
            signals.append(snapshot)
            continue
        score = model.score(feature_values(snapshot, ordinal))
        admitted = score >= threshold
        decisions.append(
            {
                "bar_ts": snapshot.bar_ts.isoformat(),
                "direction": snapshot.entry_dir,
                "ordinal": ordinal,
                "score": score,
                "admitted": admitted,
            }
        )
        if admitted:
            signals.append(snapshot)
        else:
            signals.append(
                replace(
                    snapshot,
                    entry_dir=None,
                    entry_proposed_dir=None,
                    entry_blocked_by="directional_interaction_admission",
                    entry_controls=(
                        *snapshot.entry_controls,
                        "directional_impulse:interaction_admission_block",
                    ),
                )
            )
    return replace(part.tape, signals=tuple(signals)), decisions


def model_payload(model: Model) -> dict[str, Any]:
    return {
        "objective": model.objective,
        "regularization": model.regularization,
        "features": FEATURE_NAMES,
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "weights": model.weights.tolist(),
    }


def trade_ledger(result) -> list[dict[str, Any]]:
    return [
        {
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": (
                trade.exit_time.isoformat() if trade.exit_time is not None else None
            ),
            "qty": float(trade.qty),
            "entry_price": float(trade.entry_price),
            "exit_price": (
                float(trade.exit_price) if trade.exit_price is not None else None
            ),
            "pnl": float(trade.pnl(1.0)),
            "exit_reason": trade.exit_reason,
            "bars_held": int(trade.bars_held),
            "mfe_points": float(trade.max_favorable_excursion),
            "mae_points": float(trade.max_adverse_excursion),
            "entry_atr": (
                float(trade.entry_atr) if trade.entry_atr is not None else None
            ),
        }
        for trade in result.trades
    ]


def static_main() -> None:
    started = time.perf_counter()
    print(
        "XSP directional interaction campaign: 45 dense discovery cells; "
        "ETA 1–2 minutes; 10-minute hard cap; 2023/2024 sealed unless gates pass.",
        flush=True,
    )
    train = build_part("train")
    events = candidate_events(train)
    baseline = run(train, train.tape)
    matrix, targets = labeled_training(train, baseline)
    print(
        f"preflight bars={len(train.bars)} sessions={len(train.dates)} "
        f"events={len(events)} labeled_trades={len(matrix)} "
        f"baseline_net={metrics(train, baseline)['net_pnl']:.4f}",
        flush=True,
    )
    models = [
        fit_model(objective, regularization, matrix, targets[objective])
        for objective in OBJECTIVES
        for regularization in REGULARIZATIONS
    ]
    rows = []
    outcomes = {}
    total = len(models) * len(TARGET_ADMISSIONS)
    for completed, (model, target_count) in enumerate(
        (
            (model, target_count)
            for model in models
            for target_count in TARGET_ADMISSIONS
        ),
        start=1,
    ):
        if time.perf_counter() - started > 600:
            raise TimeoutError("ten-minute campaign hard cap reached")
        threshold = threshold_for(model, events, target_count)
        projected, decisions = project(train, model, threshold)
        result = run(train, projected)
        values = metrics(train, result)
        key = f"{model.identity}:target={target_count}"
        outcomes[key] = (result, decisions)
        rows.append(
            {
                "key": key,
                "model": model.identity,
                "target_admitted_events": target_count,
                "threshold": threshold,
                "admitted_events": sum(row["admitted"] for row in decisions),
                "metrics": values,
                "pass": passes(values),
            }
        )
        if completed % 5 == 0 or completed == total:
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (total - completed)
            print(
                f"progress={completed}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["pass"]),
            float(row["metrics"]["daily_pnl_lcb95"]),
            float(row["metrics"]["net_pnl"]),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked and ranked[0]["pass"] else None
    challenges: dict[str, Any] = {}
    if selected is not None:
        selected_model = next(
            model for model in models if model.identity == selected["model"]
        )
        for name in ("audit", "holdout"):
            if name == "holdout" and not challenges.get("audit", {}).get("pass"):
                break
            part = build_part(name)
            projected, decisions = project(
                part,
                selected_model,
                float(selected["threshold"]),
            )
            result = run(part, projected)
            values = metrics(part, result)
            challenges[name] = {
                "metrics": values,
                "pass": passes(values),
                "decisions": decisions,
                "trades": trade_ledger(result),
            }
            if not challenges[name]["pass"]:
                break

    best = selected or ranked[0]
    best_result, best_decisions = outcomes[best["key"]]
    payload = {
        "schema": "xsp.directional-interaction-admission.study.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "research_only",
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "start": PARTITIONS["train"][0].isoformat(),
            "end": PARTITIONS["holdout"][1].isoformat(),
            "directional_impulse_sha256": sha256(
                ROOT / "tradebot/engines/directional_impulse.py"
            ),
            "engine_sha256": sha256(ROOT / "tradebot/backtest/engine.py"),
            "lifecycle_sha256": sha256(ROOT / "tradebot/spot/lifecycle.py"),
            "campaign_sha256": sha256(Path(__file__)),
            "git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
        },
        "contract": {
            "train": "2021-07-26..2022-12-30",
            "audit": "2023-01-01..2023-12-31",
            "holdout": "2024-01-01..2024-07-23",
            "entry_signal": "directional_impulse",
            "entry_fill": "next_5m_open",
            "lifecycle": LIFECYCLE,
            "unit": "$1_per_XSP_point",
            "round_trip_friction": 0.10,
            "minimum_discovery_trades": 234,
            "minimum_trades_per_session": MIN_DENSITY,
            "objectives": OBJECTIVES,
            "regularizations": REGULARIZATIONS,
            "target_admissions": TARGET_ADMISSIONS,
            "pass_gates": {
                "net_pnl": ">0",
                "daily_pnl_lcb95": ">0",
                "profit_factor": ">=1.10",
                "max_drawdown": "<=25",
                "top_five_positive_day_concentration": "<0.50",
                "both_direction_pnl": ">=0",
                "trades_per_session": f">={MIN_DENSITY:.8f}",
            },
            "forbidden": (
                "detector/lifecycle retuning, hidden regimes, volume, future bars, "
                "news backfill, options, outcome-derived exits"
            ),
        },
        "training": {
            "bars": len(train.bars),
            "sessions": len(train.dates),
            "candidate_events": len(events),
            "labeled_trades": len(matrix),
            "baseline": metrics(train, baseline),
        },
        "models": {model.identity: model_payload(model) for model in models},
        "cells": sorted(rows, key=lambda row: row["key"]),
        "leader": best,
        "leader_decisions": best_decisions,
        "leader_trades": trade_ledger(best_result),
        "challenges": challenges,
        "verdict": {
            "discovery_passing_cells": sum(row["pass"] for row in rows),
            "audit_opened": "audit" in challenges,
            "holdout_opened": "holdout" in challenges,
            "promotion": (
                "holdout_pass_research_only"
                if challenges.get("holdout", {}).get("pass")
                else "rejected"
            ),
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"artifact={OUTPUT} sha256={sha256(OUTPUT)} "
        f"passes={payload['verdict']['discovery_passing_cells']} "
        f"audit_opened={payload['verdict']['audit_opened']} "
        f"holdout_opened={payload['verdict']['holdout_opened']} "
        f"elapsed={payload['runtime']['elapsed_seconds']:.1f}s",
        flush=True,
    )
    print("leader=" + json.dumps(best, sort_keys=True), flush=True)
    if challenges:
        print(
            "challenges="
            + json.dumps(
                {
                    name: {
                        "pass": row["pass"],
                        "metrics": row["metrics"],
                    }
                    for name, row in challenges.items()
                },
                sort_keys=True,
            ),
            flush=True,
        )


WALK_OUTPUT = (
    ROOT / "backtests/out/xsp/xsp_directional_walk_forward_admission_20260726.json"
)
WALK_WINDOWS = (126, 252, 0)
WALK_REGULARIZATIONS = (10.0, 40.0, 100.0)
WALK_ADMISSION_RATES = (0.18, 0.25, 0.35, 0.45)
EXPANSION_OUTPUT = (
    ROOT / "backtests/out/xsp/xsp_directional_expansion_source_20260726.json"
)
EXPANSION_SCORE_FLOORS = (0.25, 0.35, 0.45)
EXPANSION_COHERENCE = (2.0 / 3.0, 1.0)
EXPANSION_VELOCITY_FLOORS = (0.0, 0.2, 0.4)
EXPANSION_ATR_MODES = ("off", "velocity", "acceleration")
EXPANSION_END_MINUTES = (60, 90, 135)
EXPANSION_COOLDOWNS = (6, 12)
EXPANSION_SLOW_MODES = ("off", "aligned_if_present")
LIFECYCLE_OUTPUT = (
    ROOT / "backtests/out/xsp/xsp_directional_lifecycle_anatomy_20260726.json"
)
LIFECYCLE_WINDOWS = {
    "recent": (date(2026, 6, 29), date(2026, 7, 24)),
    "one_year": (date(2025, 7, 25), date(2026, 7, 24)),
    "five_year": (date(2021, 7, 26), date(2026, 7, 24)),
}
LIFECYCLE_GATES = (
    "raw",
    "opening_edge",
    "early_atr",
    "early_atr_slope2",
    "early_atr_velocity2",
    "early_atr_coherence",
    "early_atr_retrace",
    "atr_slope2",
)
LIFECYCLE_EMA_CONFIRMATIONS = (None, "2/4", "3/7")
LIFECYCLE_FLIP_HOLDS = (0, 2, 3, 5, 6, 8, 10, 12)


def full_history() -> tuple[tuple, tuple[Path, ...]]:
    window = load_history_window(
        cache_dir=ROOT / "db",
        symbol="XSP",
        start_et=datetime(2021, 7, 26, 9, 30),
        end_et=datetime(2026, 7, 24, 16, 0),
        bar_size="5 mins",
        use_rth=True,
    )
    if window.missing_ranges:
        raise RuntimeError(f"incomplete XSP history: {window.missing_ranges}")
    return tuple(window.bars), window.source_paths


def score_threshold(scores: list[float], rate: float) -> float:
    ordered = sorted(scores, reverse=True)
    target = min(max(1, round(len(ordered) * rate)), len(ordered))
    index = target - 1
    if index + 1 >= len(ordered):
        return float(ordered[index])
    return float((ordered[index] + ordered[index + 1]) / 2.0)


def walk_forward_main() -> None:
    started = time.perf_counter()
    print(
        "XSP causal walk-forward campaign: 36 algorithm identities; "
        "ETA 2–5 minutes; 10-minute hard cap; 2024+ sealed unless 2023 passes.",
        flush=True,
    )
    history_bars, source_paths = full_history()
    history = build_range(
        "history",
        date(2021, 7, 26),
        date(2026, 7, 24),
        history_bars,
    )
    baseline = run(history, history.tape)
    events = candidate_events(history)
    event_by_entry = {
        snapshot.bar_ts + timedelta(minutes=5): (snapshot, ordinal)
        for snapshot, ordinal in events
    }
    event_rows = tuple(
        {
            "day": _trade_date(snapshot.bar_ts),
            "values": feature_values(snapshot, ordinal),
        }
        for snapshot, ordinal in events
    )
    outcome_rows = []
    for trade in baseline.trades:
        pair = event_by_entry.get(trade.entry_time)
        if pair is None or trade.exit_time is None:
            continue
        outcome_rows.append(
            {
                "day": _trade_date(trade.exit_time),
                "values": feature_values(*pair),
                "pnl": float(trade.pnl(1.0)),
            }
        )
    session_dates = tuple(history.dates)
    model_cache: dict[tuple[int, int, int, float], tuple[Model, list[float], date]] = {}

    def monthly_model(
        month: date,
        window_sessions: int,
        regularization: float,
    ) -> tuple[Model, list[float], date]:
        key = (month.year, month.month, window_sessions, regularization)
        cached = model_cache.get(key)
        if cached is not None:
            return cached
        prior_sessions = tuple(day for day in session_dates if day < month)
        if not prior_sessions:
            raise RuntimeError(f"no prior sessions before {month}")
        cutoff = (
            prior_sessions[max(0, len(prior_sessions) - window_sessions)]
            if window_sessions
            else prior_sessions[0]
        )
        training = tuple(
            row
            for row in outcome_rows
            if cutoff <= row["day"] < month
        )
        prior_events = tuple(
            row
            for row in event_rows
            if cutoff <= row["day"] < month
        )
        if len(training) < 100 or not prior_events:
            raise RuntimeError(
                f"insufficient walk-forward history before {month}: "
                f"{len(training)} outcomes/{len(prior_events)} events"
            )
        matrix = np.vstack([row["values"] for row in training])
        target = np.asarray([row["pnl"] for row in training], dtype=float)
        model = fit_model("net_points", regularization, matrix, target)
        scores = [model.score(row["values"]) for row in prior_events]
        model_cache[key] = (model, scores, cutoff)
        return model_cache[key]

    def projected_year(
        part: Part,
        window_sessions: int,
        regularization: float,
        admission_rate: float,
    ) -> tuple[PreparedSpotEvaluatorTape, list[dict[str, Any]], dict[str, Any]]:
        event_map = {
            id(snapshot): ordinal for snapshot, ordinal in candidate_events(part)
        }
        signals = []
        decisions = []
        monthly: dict[str, Any] = {}
        for snapshot in part.tape.signals:
            ordinal = event_map.get(id(snapshot))
            if ordinal is None:
                signals.append(snapshot)
                continue
            day = _trade_date(snapshot.bar_ts)
            month = date(day.year, day.month, 1)
            model, prior_scores, cutoff = monthly_model(
                month,
                window_sessions,
                regularization,
            )
            threshold = score_threshold(prior_scores, admission_rate)
            score = model.score(feature_values(snapshot, ordinal))
            admitted = score >= threshold
            decisions.append(
                {
                    "bar_ts": snapshot.bar_ts.isoformat(),
                    "direction": snapshot.entry_dir,
                    "ordinal": ordinal,
                    "score": score,
                    "threshold": threshold,
                    "admitted": admitted,
                    "model_month": month.isoformat(),
                }
            )
            month_key = month.isoformat()
            if month_key not in monthly:
                monthly[month_key] = {
                    **model_payload(model),
                    "training_cutoff": cutoff.isoformat(),
                    "threshold": threshold,
                    "prior_score_count": len(prior_scores),
                }
            if admitted:
                signals.append(snapshot)
            else:
                signals.append(
                    replace(
                        snapshot,
                        entry_dir=None,
                        entry_proposed_dir=None,
                        entry_blocked_by="directional_walk_forward_admission",
                        entry_controls=(
                            *snapshot.entry_controls,
                            "directional_impulse:walk_forward_admission_block",
                        ),
                    )
                )
        return replace(part.tape, signals=tuple(signals)), decisions, monthly

    discovery = build_range(
        "walk-discovery-2023",
        date(2023, 1, 1),
        date(2023, 12, 31),
        history_bars,
    )
    rows = []
    outcomes = {}
    identities = tuple(
        (window_sessions, regularization, admission_rate)
        for window_sessions in WALK_WINDOWS
        for regularization in WALK_REGULARIZATIONS
        for admission_rate in WALK_ADMISSION_RATES
    )
    for completed, identity in enumerate(identities, start=1):
        if time.perf_counter() - started > 600:
            raise TimeoutError("ten-minute walk-forward hard cap reached")
        window_sessions, regularization, admission_rate = identity
        tape, decisions, monthly = projected_year(discovery, *identity)
        result = run(discovery, tape)
        values = metrics(discovery, result)
        key = (
            f"window={'expanding' if not window_sessions else window_sessions}:"
            f"l2={regularization:g}:rate={admission_rate:g}"
        )
        outcomes[key] = (result, decisions, monthly)
        rows.append(
            {
                "key": key,
                "window_sessions": window_sessions or "expanding",
                "regularization": regularization,
                "admission_rate": admission_rate,
                "admitted_events": sum(row["admitted"] for row in decisions),
                "metrics": values,
                "pass": passes(values),
            }
        )
        if completed % 4 == 0 or completed == len(identities):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(identities) - completed)
            print(
                f"progress={completed}/{len(identities)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["pass"]),
            float(row["metrics"]["daily_pnl_lcb95"]),
            float(row["metrics"]["net_pnl"]),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked and ranked[0]["pass"] else None
    challenges: dict[str, Any] = {}
    if selected is not None:
        identity = (
            int(selected["window_sessions"])
            if selected["window_sessions"] != "expanding"
            else 0,
            float(selected["regularization"]),
            float(selected["admission_rate"]),
        )
        for name, start, end in (
            ("2024", date(2024, 1, 1), date(2024, 12, 31)),
            ("2025", date(2025, 1, 1), date(2025, 12, 31)),
            ("2026_partial", date(2026, 1, 1), date(2026, 7, 24)),
        ):
            part = build_range(name, start, end, history_bars)
            tape, decisions, monthly = projected_year(part, *identity)
            result = run(part, tape)
            values = metrics(part, result)
            challenges[name] = {
                "metrics": values,
                "pass": passes(values),
                "decisions": decisions,
                "monthly_models": monthly,
                "trades": trade_ledger(result),
            }
            if not challenges[name]["pass"]:
                break

    best = selected or ranked[0]
    best_result, best_decisions, best_monthly = outcomes[best["key"]]
    payload = {
        "schema": "xsp.directional-walk-forward-admission.study.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "research_only",
        "source": {
            "paths": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
                for path in source_paths
            ],
            "bars": len(history.bars),
            "sessions": len(history.dates),
            "start": history.dates[0].isoformat(),
            "end": history.dates[-1].isoformat(),
            "directional_impulse_sha256": sha256(
                ROOT / "tradebot/engines/directional_impulse.py"
            ),
            "engine_sha256": sha256(ROOT / "tradebot/backtest/engine.py"),
            "lifecycle_sha256": sha256(ROOT / "tradebot/spot/lifecycle.py"),
            "campaign_sha256": sha256(Path(__file__)),
            "git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
        },
        "contract": {
            "algorithm_discovery": "2023-01-01..2023-12-31",
            "challenges": (
                "2024-01-01..2024-12-31",
                "2025-01-01..2025-12-31",
                "2026-01-01..2026-07-24",
            ),
            "objective": "net_points",
            "windows": WALK_WINDOWS,
            "regularizations": WALK_REGULARIZATIONS,
            "admission_rates": WALK_ADMISSION_RATES,
            "recalibration": "calendar-month boundary",
            "training_evidence": (
                "only raw-turn shadow trades exited before the model month"
            ),
            "threshold_evidence": (
                "same-window prior event-score distribution only"
            ),
            "feature_names": FEATURE_NAMES,
            "lifecycle": LIFECYCLE,
            "unit": "$1_per_XSP_point",
            "round_trip_friction": 0.10,
            "minimum_trades_per_session": MIN_DENSITY,
            "pass_gates": {
                "net_pnl": ">0",
                "daily_pnl_lcb95": ">0",
                "profit_factor": ">=1.10",
                "max_drawdown": "<=25",
                "top_five_positive_day_concentration": "<0.50",
                "both_direction_pnl": ">=0",
                "trades_per_session": f">={MIN_DENSITY:.8f}",
            },
            "order_authority": "none",
        },
        "baseline_shadow": {
            "trades": len(baseline.trades),
            "matured_outcomes": len(outcome_rows),
            "candidate_events": len(events),
        },
        "cells": sorted(rows, key=lambda row: row["key"]),
        "leader": best,
        "leader_monthly_models": best_monthly,
        "leader_decisions": best_decisions,
        "leader_trades": trade_ledger(best_result),
        "challenges": challenges,
        "verdict": {
            "discovery_passing_cells": sum(row["pass"] for row in rows),
            "selected": selected is not None,
            "opened_challenges": tuple(challenges),
            "promotion": (
                "all_historical_challenges_pass_research_only"
                if challenges.get("2026_partial", {}).get("pass")
                else "rejected"
            ),
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "model_cache_entries": len(model_cache),
        },
    }
    WALK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    WALK_OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"artifact={WALK_OUTPUT} sha256={sha256(WALK_OUTPUT)} "
        f"passes={payload['verdict']['discovery_passing_cells']} "
        f"challenges={list(challenges)} "
        f"elapsed={payload['runtime']['elapsed_seconds']:.1f}s",
        flush=True,
    )
    print("leader=" + json.dumps(best, sort_keys=True), flush=True)
    if challenges:
        print(
            "challenges="
            + json.dumps(
                {
                    name: {
                        "pass": row["pass"],
                        "metrics": row["metrics"],
                    }
                    for name, row in challenges.items()
                },
                sort_keys=True,
            ),
            flush=True,
        )


def expansion_evidence(snapshot) -> dict[str, Any] | None:
    impulse = getattr(snapshot, "directional_impulse", None)
    if snapshot is None or impulse is None:
        return None
    horizons = {int(row.bars): row for row in impulse.horizons}
    fast = tuple(horizons[horizon] for horizon in (1, 3, 6) if horizon in horizons)
    if len(fast) < 2:
        return None
    contributions = tuple(
        float(row.slope_pct_per_bar)
        / max(float(row.tr_mean_pct), 1e-12)
        * (0.5 + 0.5 * abs(float(row.efficiency)))
        for row in fast
    )
    score = statistics.fmean(contributions)
    direction = "up" if score > 0.0 else "down" if score < 0.0 else None
    if direction is None:
        return None
    sign = 1.0 if direction == "up" else -1.0
    coherence = sum(
        (float(row.slope_pct_per_bar) > 0.0) == (direction == "up")
        for row in fast
    ) / len(fast)
    velocities = tuple(
        sign
        * float(row.slope_velocity_pct_per_bar)
        / max(float(row.tr_mean_pct), 1e-12)
        for row in fast
        if row.slope_velocity_pct_per_bar is not None
    )
    slow = tuple(horizons[horizon] for horizon in (12, 24) if horizon in horizons)
    return {
        "direction": direction,
        "score": abs(score),
        "coherence": coherence,
        "velocity": statistics.fmean(velocities) if velocities else 0.0,
        "atr_velocity": float(impulse.atr_velocity_pct or 0.0),
        "atr_acceleration": float(impulse.atr_acceleration_pct or 0.0),
        "slow_aligned": all(
            sign * float(row.slope_pct_per_bar) >= 0.0 for row in slow
        ),
        "slow_observed": len(slow),
    }


def expansion_context(
    part: Part,
) -> tuple[tuple[date | None, int, datetime | None, dict[str, Any] | None], ...]:
    rows = []
    for snapshot in part.tape.signals:
        if snapshot is None:
            rows.append((None, -1, None, None))
            continue
        current = to_et(snapshot.bar_ts, naive_ts_mode="utc")
        rows.append(
            (
                _trade_date(snapshot.bar_ts),
                (current.hour * 60 + current.minute) - (9 * 60 + 30),
                snapshot.bar_ts,
                expansion_evidence(snapshot),
            )
        )
    return tuple(rows)


def expansion_plan(
    context: tuple[
        tuple[date | None, int, datetime | None, dict[str, Any] | None], ...
    ],
    cell: tuple[float, float, float, str, int, int, str],
    *,
    capture_decisions: bool = True,
) -> tuple[tuple[tuple[int, str], ...], list[dict[str, Any]]]:
    score_floor, coherence_floor, velocity_floor, atr_mode, end_minute, cooldown, slow_mode = cell
    events = []
    decisions = []
    current_day = None
    bar_index = 0
    last_event_index = -10_000
    active_direction = None
    for index, (day, minute, bar_ts, evidence) in enumerate(context):
        if day is None:
            continue
        if day != current_day:
            current_day = day
            bar_index = 0
            last_event_index = -10_000
            active_direction = None
        bar_index += 1
        qualified = bool(
            evidence is not None
            and 10 <= minute <= end_minute
            and float(evidence["score"]) >= score_floor
            and float(evidence["coherence"]) >= coherence_floor
            and float(evidence["velocity"]) >= velocity_floor
            and (
                atr_mode == "off"
                or atr_mode == "velocity"
                and float(evidence["atr_velocity"]) > 0.0
                or atr_mode == "acceleration"
                and float(evidence["atr_acceleration"]) > 0.0
            )
            and (
                slow_mode == "off"
                or not int(evidence["slow_observed"])
                or bool(evidence["slow_aligned"])
            )
        )
        direction = str(evidence["direction"]) if evidence is not None else None
        fire = bool(
            qualified
            and direction != active_direction
            and bar_index - last_event_index >= cooldown
        )
        if fire:
            active_direction = direction
            last_event_index = bar_index
            events.append((index, direction))
        elif not qualified:
            active_direction = None
        if capture_decisions and evidence is not None and (fire or qualified):
            decisions.append(
                {
                    "bar_ts": bar_ts.isoformat(),
                    **evidence,
                    "qualified": qualified,
                    "event": fire,
                }
            )
    return tuple(events), decisions


def expansion_tape(
    part: Part,
    events: tuple[tuple[int, str], ...],
    abstentions: tuple | None = None,
) -> PreparedSpotEvaluatorTape:
    if abstentions is None:
        abstentions = tuple(
            replace(
                snapshot,
                entry_dir=None,
                entry_proposed_dir=None,
                entry_source="directional_expansion",
                entry_blocked_by="directional_expansion_abstain",
                entry_controls=("directional_expansion:abstain",),
            )
            if snapshot is not None
            else None
            for snapshot in part.tape.signals
        )
    signals = list(abstentions)
    for index, direction in events:
        signals[index] = replace(
            part.tape.signals[index],
            entry_dir=direction,
            entry_proposed_dir=direction,
            entry_source="directional_expansion",
            entry_blocked_by=None,
            entry_controls=("directional_expansion:new_transition",),
        )
    return replace(part.tape, signals=tuple(signals))


def expansion_main() -> None:
    started = time.perf_counter()
    cells = tuple(
        (
            score,
            coherence,
            velocity,
            atr,
            end,
            cooldown,
            slow,
        )
        for score in EXPANSION_SCORE_FLOORS
        for coherence in EXPANSION_COHERENCE
        for velocity in EXPANSION_VELOCITY_FLOORS
        for atr in EXPANSION_ATR_MODES
        for end in EXPANSION_END_MINUTES
        for cooldown in EXPANSION_COOLDOWNS
        for slow in EXPANSION_SLOW_MODES
    )
    if len(cells) != 648:
        raise AssertionError(len(cells))
    print(
        "XSP directional expansion source: 648 identities; ETA 2–5 minutes; "
        "10-minute hard cap; 2023/2024 sealed unless discovery passes.",
        flush=True,
    )
    train = build_part("train")
    context = expansion_context(train)
    abstentions = expansion_tape(train, ()).signals
    plans = {}
    unique_events = {}
    for cell in cells:
        events, _ = expansion_plan(context, cell, capture_decisions=False)
        plans[cell] = events
        unique_events.setdefault(events, None)
    print(
        f"projected={len(cells)} unique_entry_tapes={len(unique_events)}",
        flush=True,
    )

    economics = {}
    for completed, events in enumerate(unique_events, start=1):
        if time.perf_counter() - started > 600:
            raise TimeoutError("ten-minute expansion hard cap reached")
        event_count = len(events)
        if event_count >= 234:
            result = run(train, expansion_tape(train, events, abstentions))
            values = metrics(train, result)
            ledger = trade_ledger(result)
        else:
            values = {
                "sessions": len(train.dates),
                "trades": 0,
                "trades_per_session": 0.0,
                "net_pnl": 0.0,
                "daily_pnl_lcb95": 0.0,
                "profit_factor": None,
                "max_drawdown": 0.0,
                "top_five_positive_day_concentration": 1.0,
                "direction_pnl": {"up": 0.0, "down": 0.0},
                "direction_count": {"up": 0, "down": 0},
                "exit_count": {},
                "win_rate": 0.0,
            }
            ledger = []
        economics[events] = (values, ledger)
        if completed % 50 == 0 or completed == len(unique_events):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(unique_events) - completed)
            print(
                f"unique_progress={completed}/{len(unique_events)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    rows = []
    for cell in cells:
        events = plans[cell]
        values, _ = economics[events]
        key = ":".join(str(value) for value in cell)
        rows.append(
            {
                "key": key,
                "params": {
                    "score_floor": cell[0],
                    "coherence_floor": cell[1],
                    "velocity_floor": cell[2],
                    "atr_mode": cell[3],
                    "end_minute_after_open": cell[4],
                    "cooldown_bars": cell[5],
                    "slow_mode": cell[6],
                },
                "events": len(events),
                "metrics": values,
                "pass": len(events) >= 234 and passes(values),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["pass"]),
            int(row["events"]) >= 234,
            float(row["metrics"]["daily_pnl_lcb95"]),
            float(row["metrics"]["net_pnl"]),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked and ranked[0]["pass"] else None
    challenges = {}
    if selected is not None:
        params = selected["params"]
        cell = (
            float(params["score_floor"]),
            float(params["coherence_floor"]),
            float(params["velocity_floor"]),
            str(params["atr_mode"]),
            int(params["end_minute_after_open"]),
            int(params["cooldown_bars"]),
            str(params["slow_mode"]),
        )
        for name in ("audit", "holdout"):
            if name == "holdout" and not challenges.get("audit", {}).get("pass"):
                break
            part = build_part(name)
            events, decisions = expansion_plan(expansion_context(part), cell)
            result = run(part, expansion_tape(part, events))
            values = metrics(part, result)
            challenges[name] = {
                "events": len(events),
                "metrics": values,
                "pass": passes(values),
                "decisions": decisions,
                "trades": trade_ledger(result),
            }
            if not challenges[name]["pass"]:
                break

    best = selected or ranked[0]
    best_cell = (
        float(best["params"]["score_floor"]),
        float(best["params"]["coherence_floor"]),
        float(best["params"]["velocity_floor"]),
        str(best["params"]["atr_mode"]),
        int(best["params"]["end_minute_after_open"]),
        int(best["params"]["cooldown_bars"]),
        str(best["params"]["slow_mode"]),
    )
    _, best_decisions = expansion_plan(context, best_cell)
    best_trades = economics[plans[best_cell]][1]
    payload = {
        "schema": "xsp.directional-expansion-source.study.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "research_only",
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "directional_impulse_sha256": sha256(
                ROOT / "tradebot/engines/directional_impulse.py"
            ),
            "engine_sha256": sha256(ROOT / "tradebot/backtest/engine.py"),
            "lifecycle_sha256": sha256(ROOT / "tradebot/spot/lifecycle.py"),
            "campaign_sha256": sha256(Path(__file__)),
            "git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
        },
        "contract": {
            "discovery": "2021-07-26..2022-12-30",
            "audit": "2023-01-01..2023-12-31",
            "holdout": "2024-01-01..2024-07-23",
            "cells": len(cells),
            "score_floors": EXPANSION_SCORE_FLOORS,
            "coherence": EXPANSION_COHERENCE,
            "velocity_floors": EXPANSION_VELOCITY_FLOORS,
            "atr_modes": EXPANSION_ATR_MODES,
            "end_minutes": EXPANSION_END_MINUTES,
            "cooldowns": EXPANSION_COOLDOWNS,
            "slow_modes": EXPANSION_SLOW_MODES,
            "event": (
                "new or flipped qualified 5/15/30m slope-velocity-coherence "
                "transition; rearmed only after qualification drops"
            ),
            "lifecycle": LIFECYCLE,
            "unit": "$1_per_XSP_point",
            "round_trip_friction": 0.10,
            "minimum_discovery_trades": 234,
            "minimum_trades_per_session": MIN_DENSITY,
            "order_authority": "none",
        },
        "cells": sorted(rows, key=lambda row: row["key"]),
        "leader": best,
        "leader_decisions": best_decisions,
        "leader_trades": best_trades,
        "challenges": challenges,
        "verdict": {
            "discovery_passing_cells": sum(row["pass"] for row in rows),
            "audit_opened": "audit" in challenges,
            "holdout_opened": "holdout" in challenges,
            "promotion": (
                "holdout_pass_research_only"
                if challenges.get("holdout", {}).get("pass")
                else "rejected"
            ),
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
    }
    EXPANSION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EXPANSION_OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"artifact={EXPANSION_OUTPUT} sha256={sha256(EXPANSION_OUTPUT)} "
        f"passes={payload['verdict']['discovery_passing_cells']} "
        f"audit_opened={payload['verdict']['audit_opened']} "
        f"holdout_opened={payload['verdict']['holdout_opened']} "
        f"elapsed={payload['runtime']['elapsed_seconds']:.1f}s",
        flush=True,
    )
    print("leader=" + json.dumps(best, sort_keys=True), flush=True)
    if challenges:
        print(
            "challenges="
            + json.dumps(
                {
                    name: {
                        "pass": row["pass"],
                        "metrics": row["metrics"],
                    }
                    for name, row in challenges.items()
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _gate_allows(
    *,
    gate: str,
    direction: str,
    minute_et: int,
    atr_velocity: float | None,
    coherence: float | None,
    retrace_atr: float | None,
    horizons: dict[int, object],
) -> bool:
    if gate == "raw":
        return True
    if gate == "opening_edge":
        allowed, _ = XSP_OPENING_EDGE_POLICY.allows(
            direction=direction,
            minute_et=minute_et,
            atr_velocity=atr_velocity,
            retrace_atr=retrace_atr,
            coherence=coherence,
        )
        return allowed
    sign = 1.0 if direction == "up" else -1.0

    def value(row: object, name: str) -> float:
        raw = row.get(name) if isinstance(row, dict) else getattr(row, name, 0.0)
        return float(raw or 0.0)

    fast_slopes = [
        sign * value(horizons[horizon], "slope_pct_per_bar")
        for horizon in (1, 3, 6)
        if horizon in horizons
    ]
    fast_velocities = [
        sign * value(horizons[horizon], "slope_velocity_pct_per_bar")
        for horizon in (1, 3, 6)
        if horizon in horizons
    ]
    early = minute_et <= (11 * 60 + 15)
    expanding = float(atr_velocity or 0.0) > 0.0
    slope2 = sum(row > 0.0 for row in fast_slopes) >= 2
    velocity2 = sum(row > 0.0 for row in fast_velocities) >= 2
    return {
        "early_atr": early and expanding,
        "early_atr_slope2": early and expanding and slope2,
        "early_atr_velocity2": early and expanding and velocity2,
        "early_atr_coherence": (
            early and expanding and float(coherence or 0.0) >= (2.0 / 3.0)
        ),
        "early_atr_retrace": (
            early and expanding and float(retrace_atr or 0.0) >= 1.0
        ),
        "atr_slope2": expanding and slope2,
    }[gate]


def lifecycle_gate_tape(
    part: Part,
    gate: str,
) -> tuple[PreparedSpotEvaluatorTape, dict[str, int]]:
    signals = []
    counts = Counter()
    for snapshot in part.tape.signals:
        impulse = getattr(snapshot, "directional_impulse", None)
        direction = getattr(impulse, "turn_event", None)
        if direction not in ("up", "down"):
            signals.append(snapshot)
            continue
        counts["source_events"] += 1
        if snapshot.entry_dir not in ("up", "down"):
            counts["upstream_blocked"] += 1
            signals.append(snapshot)
            continue
        counts["upstream_passed"] += 1
        current = to_et(snapshot.bar_ts, naive_ts_mode="utc")
        allowed = _gate_allows(
            gate=gate,
            direction=direction,
            minute_et=current.hour * 60 + current.minute,
            atr_velocity=impulse.atr_velocity_pct,
            coherence=impulse.coherence,
            retrace_atr=impulse.retrace_atr,
            horizons={int(row.bars): row for row in impulse.horizons},
        )
        control = f"directional_impulse:{gate}"
        if allowed:
            counts["gate_passed"] += 1
            signals.append(
                replace(
                    snapshot,
                    entry_controls=(*snapshot.entry_controls, f"{control}:pass"),
                )
            )
        else:
            counts["gate_blocked"] += 1
            signals.append(
                replace(
                    snapshot,
                    entry_dir=None,
                    entry_blocked_by=control,
                    entry_controls=(*snapshot.entry_controls, f"{control}:block"),
                )
            )
    return replace(part.tape, signals=tuple(signals)), dict(sorted(counts.items()))


def _candidate_metrics(values: dict[str, Any]) -> dict[str, object]:
    sessions = max(1, int(values["sessions"]))
    annualized_trades = float(values["trades"]) / sessions * 252.0
    profit_factor = values["profit_factor"]
    both_directions = min(values["direction_pnl"].values()) >= 0.0
    candidate = bool(
        annualized_trades > 200.0
        and values["net_pnl"] > 0.0
        and profit_factor is not None
        and profit_factor > 1.0
        and both_directions
    )
    return {
        "annualized_trades": annualized_trades,
        "candidate": candidate,
        "reliable": bool(
            candidate
            and values["daily_pnl_lcb95"] > 0.0
            and profit_factor >= 1.10
            and values["max_drawdown"] <= 25.0
            and values["top_five_positive_day_concentration"] < 0.50
        ),
    }


def _gate_path_summary(
    census: dict[str, object],
    gate: str,
) -> dict[str, object]:
    events = [
        event
        for session in census["sessions"]
        for event in session["events"]
    ]
    selected = []
    for event in events:
        evidence = event["evidence"]
        if _gate_allows(
            gate=gate,
            direction=str(event["direction"]),
            minute_et=int(event["time_et"][:2]) * 60 + int(event["time_et"][3:]),
            atr_velocity=evidence["atr_velocity_pct"],
            coherence=evidence["coherence"],
            retrace_atr=evidence["retrace_atr"],
            horizons={
                int(row["bars"]): row for row in evidence["horizons"]
            },
        ):
            selected.append(event)
    paths = {}
    for horizon in ("6", "12", "24"):
        rows = [
            event["forward_paths"][horizon]
            for event in selected
            if horizon in event["forward_paths"]
            and event["forward_paths"][horizon]["observations"] == int(horizon)
        ]
        paths[horizon] = {
            "observations": len(rows),
            "mean_directed_close_points": (
                statistics.fmean(row["directed_close_points"] for row in rows)
                if rows
                else None
            ),
            "mean_directed_mfe_points": (
                statistics.fmean(row["directed_mfe_points"] for row in rows)
                if rows
                else None
            ),
            "mean_directed_mae_points": (
                statistics.fmean(row["directed_mae_points"] for row in rows)
                if rows
                else None
            ),
        }
    return {
        "events": len(selected),
        "extrema_match_precision": (
            sum(bool(event["matched"]) for event in selected) / len(selected)
            if selected
            else 0.0
        ),
        "forward_paths": paths,
    }


def lifecycle_main() -> None:
    started = time.perf_counter()
    history_bars, source_paths = full_history()
    manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for path in source_paths
    ]
    source_fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    identities = [
        {
            "gate": gate,
            "primary_ema": primary_ema,
            "flip_hold_bars": hold,
            "id": (
                f"gate={gate}:ema={primary_ema or 'off'}:hold={hold}"
            ),
        }
        for gate in LIFECYCLE_GATES
        for primary_ema in LIFECYCLE_EMA_CONFIRMATIONS
        for hold in LIFECYCLE_FLIP_HOLDS
    ]
    print(
        f"XSP directional lifecycle anatomy: {len(identities)} compact cells; "
        "ETA 2–8 minutes; 20-minute hard cap; >200 annualized trades required.",
        flush=True,
    )

    window_rows: dict[str, list[dict[str, object]]] = {}
    anatomy: dict[str, object] = {}
    active = identities
    recent_candidates: list[str] = []
    freeze_sha = ""
    for window_index, (window_name, (start, end)) in enumerate(
        LIFECYCLE_WINDOWS.items()
    ):
        if window_index:
            previous_name = tuple(LIFECYCLE_WINDOWS)[window_index - 1]
            eligible = {
                str(row["id"])
                for row in window_rows[previous_name]
                if row["candidate"]
            }
            active = [
                identity
                for identity in active
                if str(identity["id"]) in eligible
            ]
            if not active:
                break
        rows = []
        census_bars = tuple(
            bar
            for bar in history_bars
            if start <= _trade_date(bar.ts) <= end
        )
        census = xsp_directional_turn_census(
            census_bars,
            source_fingerprint=source_fingerprint,
            include_session_ledger=True,
        )
        anatomy[window_name] = {
            "turn_census": {
                key: value
                for key, value in census.items()
                if key != "sessions"
            },
            "gate_paths": {
                gate: _gate_path_summary(census, gate)
                for gate in LIFECYCLE_GATES
            },
            "recent_session_ledger": (
                census["sessions"] if window_name == "recent" else None
            ),
        }
        for primary_ema in LIFECYCLE_EMA_CONFIRMATIONS:
            matching = [
                identity
                for identity in active
                if identity["primary_ema"] == primary_ema
            ]
            if not matching:
                continue
            base_cfg = config(
                lifecycle=None,
                primary_ema=primary_ema,
                max_entries_per_day=5,
            )
            part = build_range(
                window_name,
                start,
                end,
                history_bars,
                cfg=base_cfg,
            )
            admission_part = None
            tapes = {
                gate: lifecycle_gate_tape(part, gate)
                for gate in {
                    str(identity["gate"]) for identity in matching
                }
                if gate != "opening_edge"
            }
            if any(identity["gate"] == "opening_edge" for identity in matching):
                admission_cfg = xsp_opening_edge_bundle(
                    start=start,
                    end=end,
                    flip_hold_bars=0,
                    primary_ema=primary_ema,
                )
                admission_part = build_range(
                    window_name,
                    start,
                    end,
                    history_bars,
                    cfg=admission_cfg,
                )
                controls = Counter()
                for snapshot in admission_part.tape.signals:
                    if snapshot.entry_proposed_dir in ("up", "down"):
                        controls["source_events"] += 1
                    for control in snapshot.entry_controls:
                        if control.startswith("directional_impulse_admission:"):
                            controls[control] += 1
                tapes["opening_edge"] = (
                    admission_part.tape,
                    dict(sorted(controls.items())),
                )
            for identity in matching:
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("20-minute lifecycle campaign cap reached")
                cfg = (
                    xsp_opening_edge_bundle(
                        start=start,
                        end=end,
                        flip_hold_bars=int(identity["flip_hold_bars"]),
                        primary_ema=primary_ema,
                    )
                    if identity["gate"] == "opening_edge"
                    else config(
                        lifecycle=None,
                        flip_hold_bars=int(identity["flip_hold_bars"]),
                        primary_ema=primary_ema,
                        max_entries_per_day=5,
                    )
                )
                tape, gate_counts = tapes[str(identity["gate"])]
                active_part = (
                    admission_part
                    if identity["gate"] == "opening_edge"
                    else part
                )
                assert active_part is not None
                values = metrics(
                    active_part,
                    run(active_part, tape, cfg=cfg),
                )
                candidacy = _candidate_metrics(values)
                rows.append(
                    {
                        **identity,
                        "metrics": values,
                        "gate_counts": gate_counts,
                        **candidacy,
                    }
                )
        rows.sort(key=lambda row: str(row["id"]))
        window_rows[window_name] = rows
        best = max(
            rows,
            key=lambda row: (
                bool(row["reliable"]),
                bool(row["candidate"]),
                float(row["metrics"]["daily_pnl_lcb95"]),
                float(row["metrics"]["net_pnl"]),
            ),
        )
        print(
            f"{window_name}: tested={len(rows)} "
            f"candidates={sum(bool(row['candidate']) for row in rows)} "
            f"reliable={sum(bool(row['reliable']) for row in rows)} "
            f"best={best['id']} net={best['metrics']['net_pnl']:.4f} "
            f"trades={best['metrics']['trades']} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if window_name == "recent":
            recent_candidates = [
                str(row["id"]) for row in rows if row["candidate"]
            ]
            freeze_sha = hashlib.sha256(
                json.dumps(recent_candidates, sort_keys=True).encode()
            ).hexdigest()
            print(
                f"recent_freeze={freeze_sha} "
                f"identities={len(recent_candidates)}",
                flush=True,
            )

    controls = {}
    for primary_ema in LIFECYCLE_EMA_CONFIRMATIONS:
        cfg = config(
            lifecycle=None,
            primary_ema=primary_ema,
            max_entries_per_day=5,
        )
        controls[primary_ema or "off"] = SpotEntryControlPlan.from_sources(
            strategy=cfg.strategy,
            filters=cfg.strategy.filters,
            bar_size=str(cfg.backtest.bar_size),
        ).as_payload()
    edge_cfg = xsp_opening_edge_bundle(
        start=LIFECYCLE_WINDOWS["five_year"][0],
        end=LIFECYCLE_WINDOWS["five_year"][1],
    )
    controls["opening_edge"] = SpotEntryControlPlan.from_sources(
        strategy=edge_cfg.strategy,
        filters=edge_cfg.strategy.filters,
        bar_size=str(edge_cfg.backtest.bar_size),
    ).as_payload()
    payload = {
        "schema": "xsp.directional-lifecycle-anatomy.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "research_only",
        "source": {
            "manifest": manifest,
            "manifest_sha256": source_fingerprint,
            "git_head": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "campaign_sha256": sha256(Path(__file__)),
            "candidate_sha256": sha256(
                ROOT / "tradebot/research/xsp_candidate.py"
            ),
            "directional_impulse_sha256": sha256(
                ROOT / "tradebot/engines/directional_impulse.py"
            ),
            "engine_sha256": sha256(ROOT / "tradebot/backtest/engine.py"),
        },
        "contract": {
            "windows": {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in LIFECYCLE_WINDOWS.items()
            },
            "direction_source": "directional_impulse",
            "gates": list(LIFECYCLE_GATES),
            "primary_ema_confirmations": [
                value or "off" for value in LIFECYCLE_EMA_CONFIRMATIONS
            ],
            "flip_hold_bars": list(LIFECYCLE_FLIP_HOLDS),
            "lifecycle": "inverse_source_flip_or_eod_only",
            "opening_edge_admission": XSP_OPENING_EDGE_ADMISSION,
            "opening_edge_config_fingerprint": (
                XSP_OPENING_EDGE_CONFIG_FINGERPRINT
            ),
            "max_entries_per_session": 5,
            "round_trip_friction_points": 0.10,
            "annualized_trade_floor": ">200",
            "selection": "recent candidate identities frozen before challenges",
            "order_authority": "none",
        },
        "entry_control_plans": controls,
        "recent_candidate_freeze": {
            "identities": recent_candidates,
            "sha256": freeze_sha,
        },
        "anatomy": anatomy,
        "windows": window_rows,
        "verdict": {
            name: {
                "tested": len(rows),
                "candidates": sum(bool(row["candidate"]) for row in rows),
                "reliable": sum(bool(row["reliable"]) for row in rows),
            }
            for name, rows in window_rows.items()
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
    }
    LIFECYCLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LIFECYCLE_OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"artifact={LIFECYCLE_OUTPUT} sha256={sha256(LIFECYCLE_OUTPUT)} "
        f"freeze={freeze_sha} elapsed={payload['runtime']['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("static", "walk-forward", "expansion", "lifecycle"),
        default="static",
    )
    args = parser.parse_args()
    {
        "static": static_main,
        "walk-forward": walk_forward_main,
        "expansion": expansion_main,
        "lifecycle": lifecycle_main,
    }[args.mode]()
