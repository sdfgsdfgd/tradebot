"""Spot evolution sweeps + multiwindow stability eval (canonical module).

Designed to start from the current MNQ spot 12m champion (or a selected base)
and explore incremental improvements without confounding:
  0) timing (EMA preset)
  A) volume gating
  B) time-of-day (ET) gating
  C) ATR-scaled exits
  D) ORB + Fibonacci target variants (15m)
  E) Supertrend regime sensitivity squeeze
  F) Dual regime gating (regime2)
  G) Chop-killer quality filters (spread/slope/cooldown/skip-open)
  H) $TICK width gate (Raschke-style)
  I) Joint sweeps (interaction hunts)

All knobs are opt-in; default bot behavior is unchanged.

NOTE: This file was renamed from `run_backtest_spot.py` to keep spot backtest entrypoints
clustered together (`run_backtests_spot_sweeps.py`, `run_backtests_spot_multiwindow.py`).
Use `python -m tradebot.backtest spot ...` / `spot_multitimeframe ...` rather than importing
this file directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time as pytime
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .cli_utils import (
    expected_cache_path as _expected_cache_path,
    parse_date as _parse_date,
    parse_window as _parse_window,
)
from .config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    SpotStrategyConfig,
    SpotLegConfig,
    SyntheticConfig,
    _parse_filters,
)
from .data import ContractMeta, IBKRHistoricalData, _find_covering_cache_path
from .engine import _run_spot_backtest_summary, _spot_multiplier
from .sweeps import utc_now_iso_z, write_json
from ..signals import parse_bar_size


# NOTE (worker orchestration): many axes in this file spawn sharded subprocess workers via CLI args
# and need to strip/override flags from `sys.argv[1:]`. Keep the helpers centralized so new axes
# can reuse them without copy/paste.
def _strip_flag(argv: list[str], flag: str) -> list[str]:
    return [arg for arg in argv if arg != flag]


def _strip_flag_with_value(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == flag:
            idx += 2
            continue
        if arg.startswith(flag + "="):
            idx += 1
            continue
        out.append(arg)
        idx += 1
    return out


def _strip_flags(
    argv: list[str],
    *,
    flags: tuple[str, ...] = (),
    flags_with_values: tuple[str, ...] = (),
) -> list[str]:
    out = list(argv)
    for flag in flags:
        out = _strip_flag(out, str(flag))
    for flag in flags_with_values:
        out = _strip_flag_with_value(out, str(flag))
    return out


def _parse_worker_shard(raw_worker: object, raw_workers: object, *, label: str) -> tuple[int, int]:
    try:
        worker_id = int(raw_worker) if raw_worker is not None else 0
    except (TypeError, ValueError):
        worker_id = 0
    try:
        workers = int(raw_workers) if raw_workers is not None else 1
    except (TypeError, ValueError):
        workers = 1
    workers = max(1, int(workers))
    worker_id = max(0, int(worker_id))
    if worker_id >= workers:
        raise SystemExit(f"Invalid {label} worker shard: worker={worker_id} workers={workers} (worker must be < workers).")
    return worker_id, workers


# region Cache Helpers
def _require_offline_cache_or_die(
    *,
    cache_dir: Path,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
) -> None:
    covering = _find_covering_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    if covering is not None:
        return
    expected = _expected_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start_dt=start_dt,
        end_dt=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    tag = "rth" if use_rth else "full24"
    raise SystemExit(
        f"--offline was requested, but cached bars are missing for {symbol} {bar_size} {tag} "
        f"{start_dt.date().isoformat()}→{end_dt.date().isoformat()} (expected: {expected}). "
        "Re-run without --offline to fetch via IBKR (or prefetch the cache first)."
    )
# endregion


# region Bundle Builders
def _bundle_base(
    *,
    symbol: str,
    start: date,
    end: date,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
    filters: FiltersConfig | None,
    entry_signal: str = "ema",
    ema_preset: str | None = "2/4",
    entry_confirm_bars: int = 0,
    spot_exit_mode: str = "pct",
    spot_atr_period: int = 14,
    spot_pt_atr_mult: float = 1.5,
    spot_sl_atr_mult: float = 1.0,
    orb_window_mins: int = 15,
    orb_risk_reward: float = 2.0,
    orb_target_mode: str = "rr",
    spot_profit_target_pct: float | None = 0.015,
    spot_stop_loss_pct: float | None = 0.03,
    flip_exit_min_hold_bars: int = 4,
    spot_close_eod: bool = False,
) -> ConfigBundle:
    backtest = BacktestConfig(
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        starting_cash=100_000.0,
        risk_free_rate=0.02,
        cache_dir=Path(cache_dir),
        calibration_dir=Path(cache_dir) / "calibration",
        output_dir=Path("backtests/out"),
        calibrate=False,
        offline=bool(offline),
    )

    strategy = SpotStrategyConfig(
        name="spot_evolve",
        instrument="spot",
        symbol=str(symbol).strip().upper(),
        exchange=None,
        right="PUT",
        entry_days=(0, 1, 2, 3, 4),
        max_entries_per_day=0,
        dte=0,
        otm_pct=0.0,
        width_pct=0.0,
        profit_target=0.0,
        stop_loss=0.0,
        exit_dte=0,
        quantity=1,
        stop_loss_basis="max_loss",
        min_credit=None,
        ema_preset=ema_preset,
        ema_entry_mode="cross",
        entry_confirm_bars=int(entry_confirm_bars),
        regime_ema_preset=None,
        regime_bar_size="4 hours",
        ema_directional=False,
        exit_on_signal_flip=True,
        flip_exit_mode="entry",
        flip_exit_gate_mode="off",
        flip_exit_min_hold_bars=int(flip_exit_min_hold_bars),
        flip_exit_only_if_profit=False,
        direction_source="ema",
        directional_legs=None,
        directional_spot={
            "up": SpotLegConfig(action="BUY", qty=1),
            "down": SpotLegConfig(action="SELL", qty=1),
        },
        legs=None,
        filters=filters,
        spot_profit_target_pct=spot_profit_target_pct,
        spot_stop_loss_pct=spot_stop_loss_pct,
        spot_close_eod=bool(spot_close_eod),
        entry_signal=str(entry_signal),
        orb_window_mins=int(orb_window_mins),
        orb_risk_reward=float(orb_risk_reward),
        orb_target_mode=str(orb_target_mode),
        spot_exit_mode=str(spot_exit_mode),
        spot_atr_period=int(spot_atr_period),
        spot_pt_atr_mult=float(spot_pt_atr_mult),
        spot_sl_atr_mult=float(spot_sl_atr_mult),
        regime_mode="supertrend",
        supertrend_atr_period=5,
        supertrend_multiplier=0.4,
        supertrend_source="hl2",
    )

    synthetic = SyntheticConfig(
        rv_lookback=60,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.2,
        iv_floor=0.05,
        term_slope=0.02,
        skew=-0.25,
        min_spread_pct=0.1,
    )
    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def _mk_filters(
    *,
    rv_min: float | None = None,
    rv_max: float | None = None,
    ema_spread_min_pct: float | None = None,
    ema_spread_min_pct_down: float | None = None,
    ema_slope_min_pct: float | None = None,
    cooldown_bars: int = 0,
    skip_first_bars: int = 0,
    volume_ratio_min: float | None = None,
    volume_ema_period: int | None = None,
    entry_start_hour_et: int | None = None,
    entry_end_hour_et: int | None = None,
    overrides: dict[str, object] | None = None,
) -> FiltersConfig | None:
    raw: dict[str, object] = {
        "rv_min": rv_min,
        "rv_max": rv_max,
        "ema_spread_min_pct": ema_spread_min_pct,
        "ema_spread_min_pct_down": ema_spread_min_pct_down,
        "ema_slope_min_pct": ema_slope_min_pct,
        "entry_start_hour": None,
        "entry_end_hour": None,
        "skip_first_bars": int(skip_first_bars),
        "cooldown_bars": int(cooldown_bars),
        "entry_start_hour_et": entry_start_hour_et,
        "entry_end_hour_et": entry_end_hour_et,
        "volume_ratio_min": volume_ratio_min,
        "volume_ema_period": volume_ema_period,
    }
    if overrides:
        raw.update(overrides)
    f = _parse_filters(raw)
    if _filters_payload(f) is None:
        return None
    return f


_WDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_AXIS_SPECS: tuple[tuple[str, bool, bool], ...] = (
    # (axis_name, include_in_axis_all, include_in_combo_full)
    ("ema", True, True),
    ("entry_mode", False, True),
    ("combo_fast", False, True),
    ("combo_full", False, False),
    ("squeeze", False, True),
    ("volume", True, True),
    ("rv", True, True),
    ("tod", True, True),
    ("tod_interaction", False, True),
    ("perm_joint", False, True),
    ("weekday", False, True),
    ("exit_time", False, True),
    ("atr", True, True),
    ("atr_fine", False, True),
    ("atr_ultra", False, True),
    ("r2_atr", False, True),
    ("r2_tod", False, True),
    ("ema_perm_joint", False, True),
    ("tick_perm_joint", False, True),
    ("regime_atr", False, True),
    ("ema_regime", False, True),
    ("chop_joint", False, True),
    ("ema_atr", False, True),
    ("tick_ema", False, True),
    ("ptsl", True, True),
    ("hf_scalp", False, True),
    ("hold", True, True),
    ("spot_short_risk_mult", True, True),
    ("orb", True, True),
    ("orb_joint", False, True),
    ("frontier", False, True),
    ("regime", True, True),
    ("regime2", True, True),
    ("regime2_ema", False, True),
    ("joint", True, True),
    ("micro_st", False, True),
    ("flip_exit", True, True),
    ("confirm", True, True),
    ("spread", True, True),
    ("spread_fine", False, True),
    ("spread_down", False, True),
    ("slope", True, True),
    ("slope_signed", True, True),
    ("cooldown", True, True),
    ("skip_open", True, True),
    ("shock", True, True),
    ("risk_overlays", True, True),
    ("loosen", True, True),
    ("loosen_atr", False, True),
    ("tick", True, True),
    ("gate_matrix", False, True),
    ("seeded_refine", False, False),
    ("champ_refine", False, False),
    ("st37_refine", False, False),
    ("shock_alpha_refine", False, False),
    ("shock_velocity_refine", False, False),
    ("shock_velocity_refine_wide", False, False),
    ("shock_throttle_refine", False, False),
    ("shock_throttle_tr_ratio", False, False),
    ("shock_throttle_drawdown", False, False),
    ("riskpanic_micro", False, False),
    ("overlay_family", False, False),
    ("exit_pivot", False, False),
)
_AXIS_ALL_PLAN = tuple(name for name, in_all, _in_combo in _AXIS_SPECS if bool(in_all))
_COMBO_FULL_PLAN = tuple(name for name, _in_all, in_combo in _AXIS_SPECS if bool(in_combo))
_AXIS_CHOICES = tuple(name for name, _in_all, _in_combo in _AXIS_SPECS)
_AXIS_PLAN_BY_MODE: dict[str, tuple[str, ...]] = {
    "axis_all": _AXIS_ALL_PLAN,
    "combo_full": _COMBO_FULL_PLAN,
}
_AXIS_PARALLEL_PROFILE_BY_MODE: dict[str, dict[str, str]] = {
    "axis_all": {
        "risk_overlays": "scaled",
    },
    "combo_full": {
        "risk_overlays": "scaled",
        "combo_fast": "scaled",
        "gate_matrix": "scaled",
        "seeded_refine": "scaled",
    },
}
_COMBO_FULL_SEEDED_AXES: tuple[str, ...] = (
    "seeded_refine",
)
_SEEDED_REFINEMENT_MEMBER_AXES: tuple[str, ...] = (
    "champ_refine",
    "overlay_family",
    "st37_refine",
)


def _axis_mode_plan(
    *,
    mode: str,
    include_seeded: bool = False,
) -> tuple[tuple[str, str, bool], ...]:
    mode_key = str(mode).strip().lower()
    axes_base = _AXIS_PLAN_BY_MODE.get(mode_key)
    if axes_base is None:
        raise ValueError(f"Unknown axis mode: {mode!r}")
    axes = list(axes_base)
    if mode_key == "combo_full" and bool(include_seeded):
        axes.extend(_COMBO_FULL_SEEDED_AXES)
    profile_map = _AXIS_PARALLEL_PROFILE_BY_MODE.get(mode_key, {})
    return tuple((axis_name, str(profile_map.get(axis_name, "single")), True) for axis_name in axes)
_ATR_EXIT_PROFILE_REGISTRY: dict[str, dict[str, object]] = {
    "atr": {
        "atr_periods": (7, 10, 14, 21),
        "pt_mults": (0.6, 0.8, 0.9, 1.0, 1.5, 2.0),
        "sl_mults": (1.0, 1.5, 2.0),
        "title": "C) ATR exits sweep (1h timing + 1d Supertrend)",
        "decimals": None,
    },
    "atr_fine": {
        "atr_periods": (7, 10, 14, 21),
        "pt_mults": (0.8, 0.9, 1.0, 1.1, 1.2),
        "sl_mults": (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8),
        "title": "ATR exits fine sweep (PT/SL multipliers)",
        "decimals": 2,
    },
    "atr_ultra": {
        "atr_periods": (7,),
        "pt_mults": (1.05, 1.08, 1.10, 1.12, 1.15),
        "sl_mults": (1.35, 1.40, 1.45, 1.50, 1.55),
        "title": "ATR exits ultra-fine sweep (PT/SL micro-grid)",
        "decimals": 2,
    },
}
_SPREAD_PROFILE_REGISTRY: dict[str, dict[str, object]] = {
    "spread": {
        "field": "ema_spread_min_pct",
        "values": (None, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1),
        "note_prefix": "spread",
        "title": "EMA spread sweep (quality gate)",
        "decimals": None,
    },
    "spread_fine": {
        "field": "ema_spread_min_pct",
        "values": (None, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008),
        "note_prefix": "spread",
        "title": "EMA spread fine sweep (quality gate)",
        "decimals": 4,
    },
    "spread_down": {
        "field": "ema_spread_min_pct_down",
        "values": (None, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.02, 0.03, 0.05),
        "note_prefix": "spread_down",
        "title": "EMA spread DOWN sweep (directional permission)",
        "decimals": 4,
    },
}
_PERM_JOINT_PROFILE: dict[str, tuple] = {
    "tod_windows": (
        (None, None, "tod=base", {}),
        (None, None, "tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
        (9, 16, "tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
        (10, 15, "tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
        (11, 16, "tod=11-16 ET", {"entry_start_hour_et": 11, "entry_end_hour_et": 16}),
        (17, 3, "tod=17-03 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 3}),
        (17, 4, "tod=17-04 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 4}),
        (17, 5, "tod=17-05 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 5}),
        (18, 3, "tod=18-03 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 3}),
        (18, 4, "tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
        (18, 5, "tod=18-05 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 5}),
        (19, 3, "tod=19-03 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 3}),
        (19, 4, "tod=19-04 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 4}),
        (19, 5, "tod=19-05 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 5}),
    ),
    "spread_variants": (
        ("spread=base", {}),
        ("spread=off", {"ema_spread_min_pct": None}),
        ("spread>=0.0020", {"ema_spread_min_pct": 0.002}),
        ("spread>=0.0030", {"ema_spread_min_pct": 0.003}),
        ("spread>=0.0040", {"ema_spread_min_pct": 0.004}),
        ("spread>=0.0050", {"ema_spread_min_pct": 0.005}),
        ("spread>=0.0060", {"ema_spread_min_pct": 0.006}),
        ("spread>=0.0070", {"ema_spread_min_pct": 0.007}),
        ("spread>=0.0080", {"ema_spread_min_pct": 0.008}),
        ("spread>=0.0100", {"ema_spread_min_pct": 0.01}),
    ),
    "vol_variants": (
        ("vol=base", {}),
        ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
        ("vol>=1.0@10", {"volume_ratio_min": 1.0, "volume_ema_period": 10}),
        ("vol>=1.0@20", {"volume_ratio_min": 1.0, "volume_ema_period": 20}),
        ("vol>=1.1@10", {"volume_ratio_min": 1.1, "volume_ema_period": 10}),
        ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
        ("vol>=1.2@10", {"volume_ratio_min": 1.2, "volume_ema_period": 10}),
        ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
        ("vol>=1.5@10", {"volume_ratio_min": 1.5, "volume_ema_period": 10}),
        ("vol>=1.5@20", {"volume_ratio_min": 1.5, "volume_ema_period": 20}),
    ),
}
_REGIME_ST_PROFILE: dict[str, tuple] = {
    "bars": ("4 hours", "1 day"),
    "atr_periods": (2, 3, 4, 5, 6, 7, 10, 11, 14, 21),
    "multipliers": (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0),
    "sources": ("close", "hl2"),
}
_REGIME2_ST_PROFILE: dict[str, tuple] = {
    "atr_periods": (2, 3, 4, 5, 6, 7, 10, 11, 14, 21),
    "multipliers": (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0),
    "sources": ("close", "hl2"),
}
_SHOCK_SWEEP_PROFILE: dict[str, tuple] = {
    "modes": ("detect", "block", "block_longs", "block_shorts", "surf"),
    "dir_variants": (("regime", 2, "dir=regime@2"), ("signal", 1, "dir=signal@1")),
    "sl_mults": (1.0, 0.75),
    "pt_mults": (1.0, 0.75),
    "short_risk_factors": (1.0, 0.5),
    "ratio_rows": (
        ("atr_ratio", 5, 30, 1.35, 1.20, 6.0),
        ("atr_ratio", 7, 50, 1.55, 1.30, 7.0),
        ("atr_ratio", 10, 80, 1.45, 1.25, 7.0),
        ("atr_ratio", 14, 120, 1.35, 1.20, 9.0),
        ("atr_ratio", 7, 30, 1.70, 1.40, 7.0),
        ("tr_ratio", 5, 30, 1.35, 1.20, 6.0),
        ("tr_ratio", 7, 50, 1.55, 1.30, 7.0),
        ("tr_ratio", 10, 80, 1.45, 1.25, 7.0),
        ("tr_ratio", 14, 120, 1.35, 1.20, 9.0),
        ("tr_ratio", 7, 30, 1.70, 1.40, 7.0),
    ),
    "daily_atr_rows": (
        (14, 13.0, 11.0, None),
        (14, 13.5, 13.0, None),
        (14, 14.0, 13.0, None),
        (14, 14.0, 13.0, 9.0),
        (10, 13.0, 11.0, 9.0),
        (21, 14.0, 13.0, 10.0),
    ),
    "drawdown_rows": (
        (10, -15.0, -8.0),
        (20, -20.0, -10.0),
        (20, -25.0, -15.0),
        (30, -25.0, -15.0),
        (60, -30.0, -20.0),
    ),
}
_SHOCK_THROTTLE_TR_RATIO_PROFILE: dict[str, tuple] = {
    "periods": ((2, 50), (3, 50), (5, 50), (3, 21)),
    "targets": (0.25, 0.35, 0.45, 0.55, 0.7, 0.9, 1.1),
    "min_mults": (0.05, 0.1, 0.2),
    "apply_tos": ("cap", "both"),
}
_SHOCK_THROTTLE_DRAWDOWN_PROFILE: dict[str, tuple] = {
    "lookbacks": (10, 20, 40),
    "targets": (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    "min_mults": (0.05, 0.1, 0.2, 0.3),
    "apply_tos": ("cap", "both"),
}
_RISKPANIC_MICRO_PROFILE: dict[str, tuple] = {
    "cutoffs_et": (None, 15),
    "panic_tr_meds": (2.75, 3.0, 3.25),
    "neg_gap_ratios": (0.5, 0.6),
    "neg_gap_abs_pcts": (None, 0.005),
    "tr_delta_mins": (None, 0.25, 0.5, 0.75),
    "long_factors": (1.0, 0.4, 0.0),
    "scale_modes": (None, "linear"),
}
_RUN_CFG_CACHE_ENGINE_VERSION = "spot_summary_v1"
_MULTIWINDOW_CACHE_ENGINE_VERSION = "spot_multiwindow_v1"


def _strategy_fingerprint(
    strategy: dict,
    *,
    filters: dict | None,
    signal_bar_size: str | None = None,
    signal_use_rth: bool | None = None,
) -> str:
    raw = dict(strategy)
    raw["filters"] = filters
    if signal_bar_size is not None:
        raw["signal_bar_size"] = str(signal_bar_size)
    if signal_use_rth is not None:
        raw["signal_use_rth"] = bool(signal_use_rth)
    return json.dumps(raw, sort_keys=True, default=str)


def _milestone_metrics_from_row(row: dict) -> dict:
    return {
        "pnl": float(row.get("pnl") or 0.0),
        "roi": float(row.get("roi") or 0.0),
        "win_rate": float(row.get("win_rate") or 0.0),
        "trades": int(row.get("trades") or 0),
        "max_drawdown": float(row.get("dd") or row.get("max_drawdown") or 0.0),
        "max_drawdown_pct": float(row.get("dd_pct") or row.get("max_drawdown_pct") or 0.0),
        "pnl_over_dd": row.get("pnl_over_dd"),
    }


def _milestone_item(
    *,
    strategy: dict,
    filters: dict | None,
    note: str | None,
    metrics: dict,
) -> dict:
    return {
        "key": _strategy_fingerprint(strategy, filters=filters),
        "strategy": strategy,
        "filters": filters,
        "note": note,
        "metrics": {
            "pnl": float(metrics.get("pnl") or 0.0),
            "roi": float(metrics.get("roi") or 0.0),
            "win_rate": float(metrics.get("win_rate") or 0.0),
            "trades": int(metrics.get("trades") or 0),
            "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
            "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
            "pnl_over_dd": metrics.get("pnl_over_dd"),
        },
    }


def _note_from_group_name(raw_name: str) -> str | None:
    text = str(raw_name or "")
    if text.endswith("]") and "[" in text:
        try:
            return text[text.rfind("[") + 1 : -1].strip() or None
        except Exception:
            return None
    return None


def _collect_milestone_items_from_rows(
    rows: list[tuple[ConfigBundle, dict, str]],
    *,
    meta: ContractMeta,
    min_win: float,
    min_trades: int,
    min_pnl_dd: float,
) -> list[dict]:
    out: list[dict] = []
    for cfg, row, note in rows:
        try:
            win = float(row.get("win_rate") or 0.0)
        except (TypeError, ValueError):
            win = 0.0
        try:
            trades = int(row.get("trades") or 0)
        except (TypeError, ValueError):
            trades = 0
        pnl_dd_raw = row.get("pnl_over_dd")
        try:
            pnl_dd = float(pnl_dd_raw) if pnl_dd_raw is not None else None
        except (TypeError, ValueError):
            pnl_dd = None
        if win < float(min_win) or trades < int(min_trades) or pnl_dd is None or pnl_dd < float(min_pnl_dd):
            continue
        strategy = _spot_strategy_payload(cfg, meta=meta)
        out.append(
            _milestone_item(
                strategy=strategy,
                filters=_filters_payload(cfg.strategy.filters),
                note=str(note),
                metrics=_milestone_metrics_from_row(row),
            )
        )
    return out


def _collect_milestone_items_from_payload(payload: dict, *, symbol: str) -> list[dict]:
    out: list[dict] = []
    if not isinstance(payload, dict):
        return out
    symbol_key = str(symbol).strip().upper()
    for group in payload.get("groups") or []:
        if not isinstance(group, dict):
            continue
        filters = group.get("filters") if isinstance(group.get("filters"), dict) else None
        note = _note_from_group_name(str(group.get("name") or ""))
        for entry in group.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            strategy = entry.get("strategy") or {}
            metrics = entry.get("metrics") or {}
            if not isinstance(strategy, dict) or not isinstance(metrics, dict):
                continue
            entry_symbol = str(entry.get("symbol") or symbol_key).strip().upper()
            if entry_symbol != symbol_key:
                continue
            out.append(
                _milestone_item(
                    strategy=dict(strategy),
                    filters=filters,
                    note=note,
                    metrics=metrics,
                )
            )
    return out


def _milestone_sort_key(item: dict) -> tuple:
    m = item.get("metrics") or {}
    return (
        float(m.get("pnl_over_dd") or float("-inf")),
        float(m.get("pnl") or 0.0),
        float(m.get("win_rate") or 0.0),
        int(m.get("trades") or 0),
    )


def _milestone_sort_key_pnl(item: dict) -> tuple:
    m = item.get("metrics") or {}
    return (
        float(m.get("pnl") or float("-inf")),
        float(m.get("pnl_over_dd") or 0.0),
        float(m.get("win_rate") or 0.0),
        int(m.get("trades") or 0),
    )


def _dedupe_best_milestones(items: list[dict]) -> list[dict]:
    best_by_key: dict[str, dict] = {}
    for item in items:
        key = str(item.get("key") or "")
        if not key:
            continue
        prev = best_by_key.get(key)
        if prev is None or _milestone_sort_key(item) > _milestone_sort_key(prev):
            best_by_key[key] = item
    return sorted(best_by_key.values(), key=_milestone_sort_key, reverse=True)


def _merge_and_write_milestones(
    *,
    out_path: Path,
    eligible_new: list[dict],
    merge_existing: bool,
    add_top_pnl_dd: int,
    add_top_pnl: int,
    symbol: str,
    start: date,
    end: date,
    signal_bar_size: str,
    use_rth: bool,
    milestone_min_win: float,
    milestone_min_trades: int,
    milestone_min_pnl_dd: float,
) -> int:
    items = list(eligible_new)
    add_top_dd = max(0, int(add_top_pnl_dd or 0))
    add_top_pnl = max(0, int(add_top_pnl or 0))
    if merge_existing and (add_top_dd > 0 or add_top_pnl > 0):
        by_dd = sorted(items, key=_milestone_sort_key, reverse=True)[:add_top_dd] if add_top_dd > 0 else []
        by_pnl = sorted(items, key=_milestone_sort_key_pnl, reverse=True)[:add_top_pnl] if add_top_pnl > 0 else []
        seen: set[str] = set()
        selected: list[dict] = []
        for item in by_dd + by_pnl:
            key = str(item.get("key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            selected.append(item)
        items = selected

    if merge_existing and out_path.exists():
        try:
            existing_payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing_payload = {}
        items.extend(_collect_milestone_items_from_payload(existing_payload, symbol=symbol))

    unique = _dedupe_best_milestones(items)
    groups: list[dict] = []
    for idx, item in enumerate(unique, start=1):
        metrics = item["metrics"]
        groups.append(
            {
                "name": _milestone_group_name_from_strategy(
                    rank=idx,
                    strategy=item["strategy"],
                    metrics=metrics,
                    note=str(item.get("note") or "").strip(),
                ),
                "filters": item["filters"],
                "entries": [{"symbol": symbol, "metrics": metrics, "strategy": item["strategy"]}],
            }
        )
    payload = {
        "name": "spot_milestones",
        "generated_at": utc_now_iso_z(),
        "notes": (
            f"Auto-generated via evolve_spot.py (post-fix). "
            f"window={start.isoformat()}→{end.isoformat()}, bar_size={signal_bar_size}, use_rth={use_rth}. "
            f"thresholds: win>={float(milestone_min_win):.2f}, trades>={int(milestone_min_trades)}, "
            f"pnl/dd>={float(milestone_min_pnl_dd):.2f}."
        ),
        "groups": groups,
    }
    write_json(out_path, payload, sort_keys=False)
    return len(groups)


def _pump_subprocess_output(prefix: str, stream) -> None:
    for line in iter(stream.readline, ""):
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _run_parallel_worker_specs(
    *,
    specs: list[tuple[str, list[str]]],
    jobs: int,
    capture_error: str,
    failure_label: str,
) -> None:
    if not specs:
        return
    jobs_eff = max(1, min(int(jobs), len(specs)))
    pending = list(specs)
    running: list[tuple[str, subprocess.Popen, threading.Thread, float]] = []
    failures: list[tuple[str, int]] = []

    while pending or running:
        while pending and len(running) < jobs_eff and not failures:
            worker_name, cmd = pending.pop(0)
            print(f"START {worker_name}", flush=True)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if proc.stdout is None:
                raise RuntimeError(str(capture_error))
            t = threading.Thread(target=_pump_subprocess_output, args=(worker_name, proc.stdout), daemon=True)
            t.start()
            running.append((worker_name, proc, t, pytime.perf_counter()))

        finished = False
        for idx, (worker_name, proc, t, started_at) in enumerate(running):
            rc = proc.poll()
            if rc is None:
                continue
            finished = True
            elapsed = pytime.perf_counter() - float(started_at)
            print(f"DONE  {worker_name} exit={rc} elapsed={elapsed:0.1f}s", flush=True)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
            if rc != 0:
                failures.append((worker_name, int(rc)))
            running.pop(idx)
            break

        if failures:
            for _worker_name, proc, _t, _started_at in running:
                try:
                    proc.terminate()
                except Exception:
                    pass
            for _worker_name, proc, _t, _started_at in running:
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            break

        if not finished:
            pytime.sleep(0.05)

    if failures:
        worker_name, rc = failures[0]
        raise SystemExit(f"{failure_label} failed: {worker_name} (exit={rc})")


def _run_parallel_json_worker_plan(
    *,
    jobs_eff: int,
    tmp_prefix: str,
    worker_tag: str,
    out_prefix: str,
    build_cmd,
    capture_error: str,
    failure_label: str,
    missing_label: str,
    invalid_label: str,
) -> dict[int, dict]:
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        tmp_root = Path(tmpdir)
        specs: list[tuple[str, list[str]]] = []
        out_paths: dict[int, Path] = {}
        for worker_id in range(max(1, int(jobs_eff))):
            out_path = tmp_root / f"{out_prefix}_{worker_id}.json"
            out_paths[worker_id] = out_path
            specs.append((f"{worker_tag}:{worker_id}", list(build_cmd(int(worker_id), int(jobs_eff), out_path))))

        _run_parallel_worker_specs(
            specs=specs,
            jobs=int(jobs_eff),
            capture_error=str(capture_error),
            failure_label=str(failure_label),
        )

        payloads: dict[int, dict] = {}
        for worker_id, out_path in out_paths.items():
            if not out_path.exists():
                raise SystemExit(f"Missing {missing_label} output: {worker_tag}:{worker_id} ({out_path})")
            try:
                payload = json.loads(out_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid {invalid_label} output JSON: {worker_tag}:{worker_id} ({out_path})") from exc
            if isinstance(payload, dict):
                payloads[int(worker_id)] = payload
    return payloads


def _run_parallel_stage_kernel(
    *,
    stage_label: str,
    jobs: int,
    total: int,
    default_jobs: int,
    offline: bool,
    offline_error: str,
    tmp_prefix: str,
    worker_tag: str,
    out_prefix: str,
    build_cmd,
    capture_error: str,
    failure_label: str,
    missing_label: str,
    invalid_label: str,
) -> tuple[int, dict[int, dict]]:
    if not bool(offline):
        raise SystemExit(str(offline_error))
    jobs_eff = min(int(jobs), int(default_jobs), int(total)) if int(total) > 0 else 1
    jobs_eff = max(1, int(jobs_eff))
    print(f"{stage_label} parallel: workers={jobs_eff} total={int(total)}", flush=True)
    payloads = _run_parallel_json_worker_plan(
        jobs_eff=int(jobs_eff),
        tmp_prefix=str(tmp_prefix),
        worker_tag=str(worker_tag),
        out_prefix=str(out_prefix),
        build_cmd=build_cmd,
        capture_error=str(capture_error),
        failure_label=str(failure_label),
        missing_label=str(missing_label),
        invalid_label=str(invalid_label),
    )
    return int(jobs_eff), payloads


def _collect_parallel_payload_records(
    *,
    payloads: dict[int, dict],
    records_key: str = "records",
    tested_key: str = "tested",
    decode_record=None,
    on_record=None,
    dedupe_key=None,
) -> int:
    tested_total = 0
    seen: set[str] | None = set() if callable(dedupe_key) else None
    for payload in payloads.values():
        if not isinstance(payload, dict):
            continue
        tested_total += int(payload.get(tested_key) or 0)
        records = payload.get(records_key) or []
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_obj = decode_record(rec) if callable(decode_record) else rec
            if rec_obj is None:
                continue
            if seen is not None:
                try:
                    rec_key = dedupe_key(rec_obj)
                except Exception:
                    rec_key = None
                if rec_key is not None:
                    rec_key_s = str(rec_key)
                    if rec_key_s in seen:
                        continue
                    seen.add(rec_key_s)
            if callable(on_record):
                on_record(rec_obj)
    return int(tested_total)


def _progress_snapshot(
    *,
    tested: int,
    total: int | None,
    started_at: float,
) -> tuple[float, float, float, float, float]:
    elapsed = max(0.0, float(pytime.perf_counter()) - float(started_at))
    rate = (float(tested) / elapsed) if elapsed > 0 else 0.0
    total_i = int(total) if total is not None else 0
    remaining = max(0, int(total_i) - int(tested)) if total is not None else 0
    eta_sec = (float(remaining) / rate) if rate > 0 else 0.0
    pct = ((float(tested) / float(total_i)) * 100.0) if total is not None and total_i > 0 else 0.0
    return elapsed, rate, float(remaining), float(eta_sec), float(pct)


def _progress_line(
    *,
    label: str,
    tested: int,
    total: int | None,
    kept: int,
    started_at: float,
    rate_unit: str = "s",
) -> str:
    elapsed, rate, _remaining, eta_sec, pct = _progress_snapshot(
        tested=int(tested),
        total=(int(total) if total is not None else None),
        started_at=float(started_at),
    )
    total_i = int(total) if total is not None else 0
    if total is not None and total_i > 0:
        return (
            f"{label} {int(tested)}/{total_i} ({pct:0.1f}%) kept={int(kept)} "
            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m rate={rate:0.2f}/{rate_unit}"
        )
    return (
        f"{label} tested={int(tested)} kept={int(kept)} "
        f"elapsed={elapsed:0.1f}s rate={rate:0.2f}/{rate_unit}"
    )


def _run_axis_subprocess_plan(
    *,
    label: str,
    axes: tuple[str, ...],
    jobs: int,
    base_cli: list[str],
    axis_jobs_resolver,
    write_milestones: bool,
    tmp_prefix: str,
) -> dict[str, dict]:
    jobs_eff = min(int(jobs), len(axes))
    print(f"{label}: jobs={jobs_eff} axes={len(axes)}", flush=True)

    milestone_paths: dict[str, Path] = {}
    milestone_payloads: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        tmp_root = Path(tmpdir)
        specs: list[tuple[str, list[str]]] = []
        for axis_name in axes:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "tradebot.backtest",
                "spot",
                *base_cli,
                "--axis",
                str(axis_name),
                "--jobs",
                str(int(axis_jobs_resolver(axis_name))),
            ]
            if bool(write_milestones):
                out_path = tmp_root / f"milestones_{axis_name}.json"
                milestone_paths[axis_name] = out_path
                cmd += ["--milestones-out", str(out_path)]
            specs.append((str(axis_name), cmd))

        _run_parallel_worker_specs(
            specs=specs,
            jobs=jobs_eff,
            capture_error=f"Failed to capture {label} worker stdout.",
            failure_label=f"{label} axis",
        )

        if bool(write_milestones):
            for axis_name, out_path in milestone_paths.items():
                if not out_path.exists():
                    continue
                try:
                    payload = json.loads(out_path.read_text())
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    milestone_payloads[axis_name] = payload

    return milestone_payloads


def _entry_days_labels(days: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for d in days:
        try:
            idx = int(d)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(_WDAYS):
            out.append(_WDAYS[idx])
    return out


def _filters_payload(filters: FiltersConfig | None) -> dict | None:
    if filters is None:
        return None
    raw = asdict(filters)
    out: dict[str, object] = {}
    for key in (
        "rv_min",
        "rv_max",
        "ema_spread_min_pct",
        "ema_spread_min_pct_down",
        "ema_slope_min_pct",
        "ema_slope_signed_min_pct_up",
        "ema_slope_signed_min_pct_down",
        "volume_ratio_min",
    ):
        if raw.get(key) is not None:
            out[key] = raw[key]
    if raw.get("volume_ratio_min") is not None and raw.get("volume_ema_period") is not None:
        out["volume_ema_period"] = raw["volume_ema_period"]
    if raw.get("entry_start_hour_et") is not None and raw.get("entry_end_hour_et") is not None:
        out["entry_start_hour_et"] = raw["entry_start_hour_et"]
        out["entry_end_hour_et"] = raw["entry_end_hour_et"]
    if raw.get("entry_start_hour") is not None and raw.get("entry_end_hour") is not None:
        out["entry_start_hour"] = raw["entry_start_hour"]
        out["entry_end_hour"] = raw["entry_end_hour"]
    if int(raw.get("skip_first_bars") or 0) > 0:
        out["skip_first_bars"] = int(raw["skip_first_bars"])
    if int(raw.get("cooldown_bars") or 0) > 0:
        out["cooldown_bars"] = int(raw["cooldown_bars"])
    if raw.get("risk_entry_cutoff_hour_et") is not None:
        out["risk_entry_cutoff_hour_et"] = int(raw["risk_entry_cutoff_hour_et"])

    # Shock overlay (engine feature). Only include when enabled.
    shock_gate_mode = str(raw.get("shock_gate_mode") or "off").strip().lower()
    if shock_gate_mode in ("", "0", "false", "none", "null"):
        shock_gate_mode = "off"
    if shock_gate_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        shock_gate_mode = "off"
    if shock_gate_mode != "off":
        out["shock_gate_mode"] = shock_gate_mode
        detector = str(raw.get("shock_detector") or "atr_ratio").strip().lower()
        if detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
            detector = "daily_atr_pct"
        elif detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
            detector = "daily_drawdown"
        elif detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
            detector = "tr_ratio"
        elif detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
            detector = "atr_ratio"
        else:
            detector = "atr_ratio"
        out["shock_detector"] = detector

        scale_detector = raw.get("shock_scale_detector")
        if scale_detector is not None:
            scale_detector = str(scale_detector).strip().lower()
        if scale_detector in ("", "0", "false", "none", "null", "off"):
            scale_detector = None
        if scale_detector is not None:
            if scale_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
                scale_detector = "daily_atr_pct"
            elif scale_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
                scale_detector = "daily_drawdown"
            elif scale_detector in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
                scale_detector = "tr_ratio"
            elif scale_detector in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
                scale_detector = "atr_ratio"
            else:
                scale_detector = None
        if scale_detector is not None:
            out["shock_scale_detector"] = scale_detector

        out["shock_direction_source"] = str(raw.get("shock_direction_source") or "regime").strip().lower()
        out["shock_direction_lookback"] = int(raw.get("shock_direction_lookback") or 2)
        if bool(raw.get("shock_regime_override_dir")):
            out["shock_regime_override_dir"] = True
        for key in (
            "shock_regime_supertrend_multiplier",
            "shock_cooling_regime_supertrend_multiplier",
            "shock_daily_cooling_atr_pct",
            "shock_risk_scale_target_atr_pct",
        ):
            if raw.get(key) is not None:
                out[key] = raw[key]
        if raw.get("shock_risk_scale_target_atr_pct") is not None:
            out["shock_risk_scale_min_mult"] = float(raw.get("shock_risk_scale_min_mult") or 0.2)
            apply_to = raw.get("shock_risk_scale_apply_to")
            if apply_to is not None:
                apply_to = str(apply_to).strip().lower()
            if apply_to in ("", "0", "false", "none", "null"):
                apply_to = None
            if apply_to in ("cap", "notional_cap", "max_notional", "cap_only", "cap-only"):
                apply_to = "cap"
            elif apply_to in ("both", "all", "cap_and_risk", "risk_and_cap", "cap+risk"):
                apply_to = "both"
            else:
                apply_to = "risk"
            if apply_to != "risk":
                out["shock_risk_scale_apply_to"] = str(apply_to)
        for key in (
            "shock_short_risk_mult_factor",
            "shock_long_risk_mult_factor",
            "shock_long_risk_mult_factor_down",
            "shock_stop_loss_pct_mult",
            "shock_profit_target_pct_mult",
        ):
            if raw.get(key) is not None:
                out[key] = raw[key]

        if detector == "daily_atr_pct" or scale_detector == "daily_atr_pct":
            out["shock_daily_atr_period"] = int(raw.get("shock_daily_atr_period") or 14)
            out["shock_daily_on_atr_pct"] = float(raw.get("shock_daily_on_atr_pct") or 0.0)
            out["shock_daily_off_atr_pct"] = float(raw.get("shock_daily_off_atr_pct") or 0.0)
            if raw.get("shock_daily_on_tr_pct") is not None:
                out["shock_daily_on_tr_pct"] = float(raw.get("shock_daily_on_tr_pct") or 0.0)
        if detector == "daily_drawdown" or scale_detector == "daily_drawdown":
            out["shock_drawdown_lookback_days"] = int(raw.get("shock_drawdown_lookback_days") or 20)
            out["shock_on_drawdown_pct"] = float(raw.get("shock_on_drawdown_pct") or 0.0)
            out["shock_off_drawdown_pct"] = float(raw.get("shock_off_drawdown_pct") or 0.0)
        if detector in ("atr_ratio", "tr_ratio") or scale_detector in ("atr_ratio", "tr_ratio"):
            # "atr_ratio" and "tr_ratio" share this main ratio knob family (TR uses these as fallback too).
            out["shock_atr_fast_period"] = int(raw.get("shock_atr_fast_period") or 7)
            out["shock_atr_slow_period"] = int(raw.get("shock_atr_slow_period") or 50)
            out["shock_on_ratio"] = float(raw.get("shock_on_ratio") or 0.0)
            out["shock_off_ratio"] = float(raw.get("shock_off_ratio") or 0.0)
            out["shock_min_atr_pct"] = float(raw.get("shock_min_atr_pct") or 0.0)

    # TR% risk overlays (engine feature). Include only when enabled.
    overlay_any = False
    if raw.get("riskoff_tr5_med_pct") is not None:
        out["riskoff_tr5_med_pct"] = float(raw.get("riskoff_tr5_med_pct") or 0.0)
        out["riskoff_tr5_lookback_days"] = int(raw.get("riskoff_tr5_lookback_days") or 5)
        out["riskoff_short_risk_mult_factor"] = float(
            1.0 if raw.get("riskoff_short_risk_mult_factor") is None else raw.get("riskoff_short_risk_mult_factor")
        )
        out["riskoff_long_risk_mult_factor"] = float(
            1.0 if raw.get("riskoff_long_risk_mult_factor") is None else raw.get("riskoff_long_risk_mult_factor")
        )
        overlay_any = True

    if raw.get("riskpanic_tr5_med_pct") is not None and raw.get("riskpanic_neg_gap_ratio_min") is not None:
        out["riskpanic_tr5_med_pct"] = float(raw.get("riskpanic_tr5_med_pct") or 0.0)
        out["riskpanic_neg_gap_ratio_min"] = float(raw.get("riskpanic_neg_gap_ratio_min") or 0.0)
        if raw.get("riskpanic_neg_gap_abs_pct_min") is not None:
            out["riskpanic_neg_gap_abs_pct_min"] = float(raw.get("riskpanic_neg_gap_abs_pct_min") or 0.0)
        out["riskpanic_lookback_days"] = int(raw.get("riskpanic_lookback_days") or 5)
        if raw.get("riskpanic_tr5_med_delta_min_pct") is not None:
            out["riskpanic_tr5_med_delta_min_pct"] = float(raw.get("riskpanic_tr5_med_delta_min_pct") or 0.0)
            out["riskpanic_tr5_med_delta_lookback_days"] = int(raw.get("riskpanic_tr5_med_delta_lookback_days") or 1)
        out["riskpanic_long_risk_mult_factor"] = float(
            1.0
            if raw.get("riskpanic_long_risk_mult_factor") is None
            else raw.get("riskpanic_long_risk_mult_factor")
        )
        scale_mode = str(raw.get("riskpanic_long_scale_mode") or "off").strip().lower()
        if scale_mode in ("linear", "lin", "delta", "linear_delta", "linear_tr_delta"):
            scale_mode = "linear"
        elif scale_mode in ("", "0", "false", "none", "null", "off"):
            scale_mode = "off"
        else:
            scale_mode = "off"
        if scale_mode != "off":
            out["riskpanic_long_scale_mode"] = scale_mode
            if raw.get("riskpanic_long_scale_tr_delta_max_pct") is not None:
                try:
                    delta_max = float(raw.get("riskpanic_long_scale_tr_delta_max_pct"))
                except (TypeError, ValueError):
                    delta_max = 0.0
                if delta_max > 0:
                    out["riskpanic_long_scale_tr_delta_max_pct"] = float(delta_max)
        out["riskpanic_short_risk_mult_factor"] = float(
            1.0
            if raw.get("riskpanic_short_risk_mult_factor") is None
            else raw.get("riskpanic_short_risk_mult_factor")
        )
        overlay_any = True

    if raw.get("riskpop_tr5_med_pct") is not None and raw.get("riskpop_pos_gap_ratio_min") is not None:
        out["riskpop_tr5_med_pct"] = float(raw.get("riskpop_tr5_med_pct") or 0.0)
        out["riskpop_pos_gap_ratio_min"] = float(raw.get("riskpop_pos_gap_ratio_min") or 0.0)
        if raw.get("riskpop_pos_gap_abs_pct_min") is not None:
            out["riskpop_pos_gap_abs_pct_min"] = float(raw.get("riskpop_pos_gap_abs_pct_min") or 0.0)
        out["riskpop_lookback_days"] = int(raw.get("riskpop_lookback_days") or 5)
        if raw.get("riskpop_tr5_med_delta_min_pct") is not None:
            out["riskpop_tr5_med_delta_min_pct"] = float(raw.get("riskpop_tr5_med_delta_min_pct") or 0.0)
            out["riskpop_tr5_med_delta_lookback_days"] = int(raw.get("riskpop_tr5_med_delta_lookback_days") or 1)
        out["riskpop_long_risk_mult_factor"] = float(
            1.0 if raw.get("riskpop_long_risk_mult_factor") is None else raw.get("riskpop_long_risk_mult_factor")
        )
        out["riskpop_short_risk_mult_factor"] = float(
            1.0 if raw.get("riskpop_short_risk_mult_factor") is None else raw.get("riskpop_short_risk_mult_factor")
        )
        overlay_any = True

    if overlay_any:
        out["riskoff_mode"] = str(raw.get("riskoff_mode") or "hygiene").strip().lower()

    return out or None


def _spot_strategy_payload(cfg: ConfigBundle, *, meta: ContractMeta) -> dict:
    strategy = asdict(cfg.strategy)
    strategy["entry_days"] = _entry_days_labels(cfg.strategy.entry_days)
    strategy["signal_bar_size"] = str(cfg.backtest.bar_size)
    strategy["signal_use_rth"] = bool(cfg.backtest.use_rth)
    strategy.pop("filters", None)

    # Ensure MNQ presets load as futures in the UI (otherwise `spot_sec_type` may default to STK).
    sym = str(cfg.strategy.symbol or "").strip().upper()
    if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
        strategy.setdefault("spot_sec_type", "FUT")
        strategy.setdefault("spot_exchange", str(meta.exchange or "CME"))
    else:
        strategy.setdefault("spot_sec_type", "STK")
        strategy.setdefault("spot_exchange", "SMART")
    return strategy


def _milestone_key(cfg: ConfigBundle) -> str:
    strategy = asdict(cfg.strategy)
    strategy.pop("filters", None)
    return _strategy_fingerprint(
        strategy,
        filters=_filters_payload(cfg.strategy.filters),
        signal_bar_size=str(cfg.backtest.bar_size),
        signal_use_rth=bool(cfg.backtest.use_rth),
    )


def _milestone_group_name(*, rank: int, cfg: ConfigBundle, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    strat = cfg.strategy
    rbar = str(getattr(strat, "regime_bar_size", "") or "").strip() or "?"
    tag = ""
    if str(getattr(strat, "regime_mode", "") or "").strip().lower() == "supertrend":
        tag = f"ST({getattr(strat,'supertrend_atr_period', '?')},{getattr(strat,'supertrend_multiplier','?')},{getattr(strat,'supertrend_source','?')})@{rbar}"
    elif getattr(strat, "regime_ema_preset", None):
        tag = f"EMA({getattr(strat,'regime_ema_preset','?')})@{rbar}"
    if str(getattr(strat, "regime2_mode", "off") or "off").strip().lower() != "off":
        r2bar = str(getattr(strat, "regime2_bar_size", "") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _milestone_group_name_from_strategy(*, rank: int, strategy: dict, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    rbar = str(strategy.get("regime_bar_size") or "").strip() or "?"
    tag = ""
    if str(strategy.get("regime_mode") or "").strip().lower() == "supertrend":
        tag = (
            f"ST({strategy.get('supertrend_atr_period', '?')},"
            f"{strategy.get('supertrend_multiplier', '?')},"
            f"{strategy.get('supertrend_source', '?')})@{rbar}"
        )
    elif strategy.get("regime_ema_preset"):
        tag = f"EMA({strategy.get('regime_ema_preset', '?')})@{rbar}"
    if str(strategy.get("regime2_mode") or "off").strip().lower() != "off":
        r2bar = str(strategy.get("regime2_bar_size") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _score_row_pnl_dd(row: dict) -> tuple:
    return (
        float(row.get("pnl_over_dd") or float("-inf")),
        float(row.get("pnl") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _score_row_pnl(row: dict) -> tuple:
    return (
        float(row.get("pnl") or float("-inf")),
        float(row.get("pnl_over_dd") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _score_row_roi(row: dict) -> float:
    return float(row.get("roi") or 0.0)


def _score_row_win_rate(row: dict) -> float:
    return float(row.get("win_rate") or 0.0)


def _score_row_roi_dd(row: dict) -> float:
    roi = float(row.get("roi") or 0.0)
    dd_pct = float(row.get("dd_pct") or 0.0)
    if dd_pct <= 0.0:
        return float("-inf") if roi <= 0.0 else float("inf")
    return float(roi / dd_pct)


def _rank_cfg_rows(
    items: list[tuple[object, dict, str]],
    *,
    scorers: list[tuple[object, int]],
    limit: int | None = None,
    key_fn=None,
) -> list[tuple[object, dict, str]]:
    ranked: list[tuple[object, dict, str]] = []
    for score_fn, top_n in scorers:
        n = int(top_n)
        if n <= 0:
            continue
        ranked.extend(sorted(items, key=lambda t, fn=score_fn: fn(t[1]), reverse=True)[:n])
    seen: set[str] = set()
    out: list[tuple[object, dict, str]] = []
    max_items = None if limit is None else max(0, int(limit))
    for cfg, row, note in ranked:
        if callable(key_fn):
            key = key_fn(cfg, row, note)
        else:
            if not isinstance(cfg, ConfigBundle):
                continue
            key = _milestone_key(cfg)
        try:
            already_seen = key in seen
        except TypeError:
            key = json.dumps(key, sort_keys=True, default=str)
            already_seen = key in seen
        if already_seen:
            continue
        seen.add(key)
        out.append((cfg, row, note))
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _rank_cfg_rows_with_meta(
    items: list[tuple[ConfigBundle, dict, str, object]],
    *,
    scorers: list[tuple[object, int]],
    limit: int | None = None,
) -> list[tuple[ConfigBundle, dict, str, object]]:
    ranked_core = _rank_cfg_rows(
        [(cfg, row, note) for cfg, row, note, _meta in items],
        scorers=scorers,
        limit=limit,
    )
    meta_by_key: dict[str, object] = {}
    for cfg, _row, _note, meta in items:
        key = _milestone_key(cfg)
        if key not in meta_by_key:
            meta_by_key[key] = meta
    out: list[tuple[ConfigBundle, dict, str, object]] = []
    for cfg, row, note in ranked_core:
        out.append((cfg, row, note, meta_by_key.get(_milestone_key(cfg))))
    return out


def _risk_overlay_off_template(*, extended: bool = False) -> dict[str, object]:
    out: dict[str, object] = {
        "risk_entry_cutoff_hour_et": None,
        "riskoff_tr5_med_pct": None,
        "riskpanic_tr5_med_pct": None,
        "riskpanic_neg_gap_ratio_min": None,
        "riskpop_tr5_med_pct": None,
        "riskpop_pos_gap_ratio_min": None,
    }
    if bool(extended):
        out.update(
            {
                "riskoff_mode": None,
                "riskoff_tr5_lookback_days": None,
                "riskoff_short_risk_mult_factor": None,
                "riskoff_long_risk_mult_factor": None,
                "riskpanic_lookback_days": None,
                "riskpanic_short_risk_mult_factor": None,
                "riskpop_lookback_days": None,
                "riskpop_long_risk_mult_factor": None,
                "riskpop_short_risk_mult_factor": None,
            }
        )
    return out


def _build_champ_refine_risk_variants(*, is_slv: bool) -> list[tuple[dict[str, object], str]]:
    risk_off = _risk_overlay_off_template()
    if bool(is_slv):
        return [(dict(risk_off), "risk=off")]
    registry: list[tuple[dict[str, object], str]] = [
        (
            {
                "riskoff_tr5_med_pct": 9.0,
                "riskoff_lookback_days": 5,
                "riskoff_mode": "hygiene",
                "riskoff_long_risk_mult_factor": 0.7,
                "riskoff_short_risk_mult_factor": 0.7,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskoff TRmed5>=9 both=0.7 cutoff<15",
        ),
        (
            {
                "riskoff_tr5_med_pct": 10.0,
                "riskoff_lookback_days": 5,
                "riskoff_mode": "hygiene",
                "riskoff_long_risk_mult_factor": 0.5,
                "riskoff_short_risk_mult_factor": 0.5,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskoff TRmed5>=10 both=0.5 cutoff<15",
        ),
        (
            {
                "riskpanic_tr5_med_pct": 9.0,
                "riskpanic_neg_gap_ratio_min": 0.6,
                "riskpanic_lookback_days": 5,
                "riskpanic_short_risk_mult_factor": 0.5,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpanic TRmed5>=9 gap>=0.6 short=0.5 cutoff<15",
        ),
        (
            {
                "riskpanic_tr5_med_pct": 9.0,
                "riskpanic_neg_gap_ratio_min": 0.6,
                "riskpanic_lookback_days": 5,
                "riskpanic_short_risk_mult_factor": 0.0,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpanic TRmed5>=9 gap>=0.6 short=0 cutoff<15",
        ),
        (
            {
                "riskpanic_tr5_med_pct": 10.0,
                "riskpanic_neg_gap_ratio_min": 0.7,
                "riskpanic_lookback_days": 5,
                "riskpanic_short_risk_mult_factor": 0.5,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpanic TRmed5>=10 gap>=0.7 short=0.5 cutoff<15",
        ),
        (
            {
                "riskpop_tr5_med_pct": 9.0,
                "riskpop_pos_gap_ratio_min": 0.6,
                "riskpop_lookback_days": 5,
                "riskpop_long_risk_mult_factor": 1.2,
                "riskpop_short_risk_mult_factor": 0.0,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpop TRmed5>=9 gap+=0.6 long=1.2 short=0 cutoff<15",
        ),
        (
            {
                "riskpop_tr5_med_pct": 9.0,
                "riskpop_pos_gap_ratio_min": 0.6,
                "riskpop_lookback_days": 5,
                "riskpop_long_risk_mult_factor": 1.5,
                "riskpop_short_risk_mult_factor": 0.0,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpop TRmed5>=9 gap+=0.6 long=1.5 short=0 cutoff<15",
        ),
        (
            {
                "riskpop_tr5_med_pct": 10.0,
                "riskpop_pos_gap_ratio_min": 0.7,
                "riskpop_lookback_days": 5,
                "riskpop_long_risk_mult_factor": 1.5,
                "riskpop_short_risk_mult_factor": 0.0,
                "risk_entry_cutoff_hour_et": 15,
            },
            "riskpop TRmed5>=10 gap+=0.7 long=1.5 short=0 cutoff<15",
        ),
    ]
    out: list[tuple[dict[str, object], str]] = [(dict(risk_off), "risk=off")]
    for over, note in registry:
        payload = dict(risk_off)
        payload.update(over)
        out.append((payload, str(note)))
    return out


def _build_champ_refine_shock_variants(*, is_slv: bool) -> list[tuple[dict[str, object], str]]:
    out: list[tuple[dict[str, object], str]] = [({"shock_gate_mode": "off"}, "shock=off")]
    if not bool(is_slv):
        for on_atr, off_atr in ((12.5, 12.0), (13.0, 12.5), (13.5, 13.0), (14.0, 13.5), (14.5, 14.0)):
            for sl_mult in (0.75, 1.0):
                out.append(
                    (
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "daily_atr_pct",
                            "shock_daily_atr_period": 14,
                            "shock_daily_on_atr_pct": float(on_atr),
                            "shock_daily_off_atr_pct": float(off_atr),
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": float(sl_mult),
                        },
                        f"shock=surf daily_atr on={on_atr:g} off={off_atr:g} sl_mult={sl_mult:g}",
                    )
                )
        for on_tr in (9.0, 11.0, 14.0):
            out.append(
                (
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "daily_atr_pct",
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 13.5,
                        "shock_daily_off_atr_pct": 13.0,
                        "shock_daily_on_tr_pct": float(on_tr),
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                    },
                    f"shock=surf daily_atr on=13.5 off=13.0 on_tr>={on_tr:g} sl_mult=0.75",
                )
            )
        out.extend(
            [
                (
                    {
                        "shock_gate_mode": "block_longs",
                        "shock_detector": "daily_atr_pct",
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 13.5,
                        "shock_daily_off_atr_pct": 13.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                    },
                    "shock=block_longs daily_atr on=13.5 off=13.0 sl_mult=0.75",
                ),
                (
                    {
                        "shock_gate_mode": "block",
                        "shock_detector": "daily_atr_pct",
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 13.5,
                        "shock_daily_off_atr_pct": 13.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                    },
                    "shock=block daily_atr on=13.5 off=13.0",
                ),
            ]
        )
        for detector in ("tr_ratio", "atr_ratio"):
            for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
                out.append(
                    (
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": str(detector),
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": float(on_ratio),
                            "shock_off_ratio": float(off_ratio),
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                        },
                        f"shock=surf {detector} on={on_ratio:g} off={off_ratio:g} min_atr=7 sl_mult=0.75",
                    )
                )
        out.append(
            (
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 13.5,
                    "shock_daily_off_atr_pct": 13.0,
                    "shock_direction_source": "regime",
                    "shock_direction_lookback": 2,
                    "shock_stop_loss_pct_mult": 0.75,
                },
                "shock=surf daily_atr dir=regime lb=2 on=13.5 off=13.0 sl_mult=0.75",
            )
        )
        return out

    for on_atr, off_atr in ((3.0, 2.5), (3.5, 3.0), (4.0, 3.5), (4.5, 4.0), (5.0, 4.5), (6.0, 5.0)):
        for sl_mult in (0.75, 1.0):
            out.append(
                (
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "daily_atr_pct",
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": float(on_atr),
                        "shock_daily_off_atr_pct": float(off_atr),
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": float(sl_mult),
                    },
                    f"shock=surf daily_atr on={on_atr:g} off={off_atr:g} sl_mult={sl_mult:g}",
                )
            )
    for on_tr in (4.0, 5.0, 6.0):
        out.append(
            (
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 4.5,
                    "shock_daily_off_atr_pct": 4.0,
                    "shock_daily_on_tr_pct": float(on_tr),
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_stop_loss_pct_mult": 0.75,
                },
                f"shock=surf daily_atr on=4.5 off=4.0 on_tr>={on_tr:g} sl_mult=0.75",
            )
        )
    for det, fast_p, slow_p, on_r, off_r, min_atr, dir_src, dir_lb, target_atr, min_mult, down_mult in (
        ("tr_ratio", 3, 21, 1.25, 1.15, 0.8, "signal", 1, 3.0, 0.10, 0.60),
        ("tr_ratio", 3, 21, 1.30, 1.20, 1.0, "signal", 1, 3.5, 0.20, 0.70),
        ("tr_ratio", 5, 30, 1.25, 1.15, 1.0, "signal", 2, 3.5, 0.20, 0.60),
        ("tr_ratio", 7, 50, 1.35, 1.25, 1.2, "regime", 2, 4.0, 0.20, 0.60),
        ("tr_ratio", 7, 50, 1.45, 1.30, 1.5, "regime", 2, 4.0, 0.20, 0.50),
        ("atr_ratio", 3, 21, 1.25, 1.15, 0.8, "signal", 1, 3.0, 0.10, 0.60),
        ("atr_ratio", 7, 50, 1.35, 1.25, 1.2, "regime", 2, 4.0, 0.20, 0.60),
    ):
        out.append(
            (
                {
                    "shock_gate_mode": "detect",
                    "shock_detector": str(det),
                    "shock_atr_fast_period": int(fast_p),
                    "shock_atr_slow_period": int(slow_p),
                    "shock_on_ratio": float(on_r),
                    "shock_off_ratio": float(off_r),
                    "shock_min_atr_pct": float(min_atr),
                    "shock_direction_source": str(dir_src),
                    "shock_direction_lookback": int(dir_lb),
                    "shock_risk_scale_target_atr_pct": float(target_atr),
                    "shock_risk_scale_min_mult": float(min_mult),
                    "shock_long_risk_mult_factor_down": float(down_mult),
                },
                f"shock=detect {det}({fast_p}/{slow_p}) {on_r:g}/{off_r:g} min_atr%={min_atr:g} "
                f"dir={dir_src} lb={dir_lb} scale@{target_atr:g}->{min_mult:g} down={down_mult:g}",
            )
        )
    for on_atr, off_atr, target_atr, min_mult, down_mult in (
        (4.0, 3.5, 3.0, 0.20, 0.70),
        (4.0, 3.5, 3.0, 0.10, 0.60),
        (4.5, 4.0, 3.5, 0.20, 0.60),
        (5.0, 4.5, 4.0, 0.20, 0.60),
    ):
        out.append(
            (
                {
                    "shock_gate_mode": "detect",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": float(on_atr),
                    "shock_daily_off_atr_pct": float(off_atr),
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_risk_scale_target_atr_pct": float(target_atr),
                    "shock_risk_scale_min_mult": float(min_mult),
                    "shock_long_risk_mult_factor_down": float(down_mult),
                },
                f"shock=detect daily_atr on={on_atr:g} off={off_atr:g} scale@{target_atr:g}->{min_mult:g} down={down_mult:g}",
            )
        )
    for dd_lb, dd_on, dd_off, down_mult in (
        (40, -10.0, -6.0, 0.60),
        (20, -7.0, -4.0, 0.70),
    ):
        out.append(
            (
                {
                    "shock_gate_mode": "detect",
                    "shock_detector": "daily_drawdown",
                    "shock_drawdown_lookback_days": int(dd_lb),
                    "shock_on_drawdown_pct": float(dd_on),
                    "shock_off_drawdown_pct": float(dd_off),
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 2,
                    "shock_long_risk_mult_factor_down": float(down_mult),
                },
                f"shock=detect dd lb={dd_lb} on={dd_on:g}% off={dd_off:g}% down={down_mult:g}",
            )
        )
    return out


def _build_st37_refine_risk_variants() -> list[tuple[dict[str, object], str]]:
    risk_off = _risk_overlay_off_template(extended=True)
    out: list[tuple[dict[str, object], str]] = [(dict(risk_off), "risk=off")]

    panic_base = {
        **risk_off,
        "risk_entry_cutoff_hour_et": 15,
        "riskpanic_tr5_med_pct": 9.0,
        "riskpanic_neg_gap_ratio_min": 0.6,
        "riskpanic_lookback_days": 5,
        "riskpanic_short_risk_mult_factor": 0.5,
        "riskoff_mode": "hygiene",
    }
    out.append((panic_base, "riskpanic base (TRmed>=9 gap>=0.6 short=0.5 cutoff<15)"))
    for v in (1.0, 0.5, 0.2, 0.0):
        out.append(({**panic_base, "riskpanic_short_risk_mult_factor": float(v)}, f"riskpanic short={v:g}"))
    for tr in (8.0, 9.0, 10.0, 11.0):
        out.append(({**panic_base, "riskpanic_tr5_med_pct": float(tr)}, f"riskpanic TRmed>={tr:g}"))
    for ratio in (0.5, 0.6, 0.7):
        out.append(({**panic_base, "riskpanic_neg_gap_ratio_min": float(ratio)}, f"riskpanic gap>={ratio:g}"))
    for lb in (3, 5, 7):
        out.append(({**panic_base, "riskpanic_lookback_days": int(lb)}, f"riskpanic lookback={lb}d"))
    for cutoff in (None, 15, 16):
        out.append(({**panic_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskpanic cutoff<{cutoff or '-'}"))
    for mode in ("hygiene", "directional"):
        out.append(({**panic_base, "riskoff_mode": str(mode)}, f"riskpanic mode={mode}"))

    riskoff_base = {
        **risk_off,
        "risk_entry_cutoff_hour_et": 15,
        "riskoff_tr5_med_pct": 9.0,
        "riskoff_tr5_lookback_days": 5,
        "riskoff_mode": "directional",
        "riskoff_long_risk_mult_factor": 0.8,
        "riskoff_short_risk_mult_factor": 0.5,
    }
    out.append((riskoff_base, "riskoff base (TRmed>=9 long=0.8 short=0.5 cutoff<15)"))
    for tr in (8.0, 9.0, 10.0, 11.0):
        out.append(({**riskoff_base, "riskoff_tr5_med_pct": float(tr)}, f"riskoff TRmed>={tr:g}"))
    for lb in (3, 5, 7):
        out.append(({**riskoff_base, "riskoff_tr5_lookback_days": int(lb)}, f"riskoff lookback={lb}d"))
    for long_f in (0.6, 0.8, 1.0):
        out.append(({**riskoff_base, "riskoff_long_risk_mult_factor": float(long_f)}, f"riskoff long={long_f:g}"))
    for short_f in (1.0, 0.5, 0.2, 0.0):
        out.append(({**riskoff_base, "riskoff_short_risk_mult_factor": float(short_f)}, f"riskoff short={short_f:g}"))
    for cutoff in (None, 15, 16):
        out.append(({**riskoff_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskoff cutoff<{cutoff or '-'}"))
    for mode in ("hygiene", "directional"):
        out.append(({**riskoff_base, "riskoff_mode": str(mode)}, f"riskoff mode={mode}"))

    pop_base = {
        **risk_off,
        "risk_entry_cutoff_hour_et": 15,
        "riskpop_tr5_med_pct": 9.0,
        "riskpop_pos_gap_ratio_min": 0.6,
        "riskpop_lookback_days": 5,
        "riskpop_long_risk_mult_factor": 1.2,
        "riskpop_short_risk_mult_factor": 0.5,
        "riskoff_mode": "hygiene",
    }
    out.append((pop_base, "riskpop base (TRmed>=9 gap>=0.6 long=1.2 short=0.5 cutoff<15)"))
    for tr in (8.0, 9.0, 10.0, 11.0):
        out.append(({**pop_base, "riskpop_tr5_med_pct": float(tr)}, f"riskpop TRmed>={tr:g}"))
    for ratio in (0.5, 0.6, 0.7):
        out.append(({**pop_base, "riskpop_pos_gap_ratio_min": float(ratio)}, f"riskpop gap>={ratio:g}"))
    for lb in (3, 5, 7):
        out.append(({**pop_base, "riskpop_lookback_days": int(lb)}, f"riskpop lookback={lb}d"))
    for long_f in (0.6, 0.8, 1.0, 1.2, 1.5):
        out.append(({**pop_base, "riskpop_long_risk_mult_factor": float(long_f), "riskpop_short_risk_mult_factor": 1.0}, f"riskpop long={long_f:g} short=1.0"))
    for short_f in (1.0, 0.5, 0.2, 0.0):
        out.append(({**pop_base, "riskpop_long_risk_mult_factor": 1.2, "riskpop_short_risk_mult_factor": float(short_f)}, f"riskpop long=1.2 short={short_f:g}"))
    out.append(({**pop_base, "riskpop_long_risk_mult_factor": 1.5, "riskpop_short_risk_mult_factor": 0.0}, "riskpop long=1.5 short=0.0"))
    for cutoff in (None, 15, 16):
        out.append(({**pop_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskpop cutoff<{cutoff or '-'}"))
    for mode in ("hygiene", "directional"):
        out.append(({**pop_base, "riskoff_mode": str(mode)}, f"riskpop mode={mode}"))

    out.append(
        (
            {
                **risk_off,
                **panic_base,
                "riskoff_tr5_med_pct": riskoff_base.get("riskoff_tr5_med_pct"),
                "riskoff_tr5_lookback_days": riskoff_base.get("riskoff_tr5_lookback_days"),
                "riskoff_mode": riskoff_base.get("riskoff_mode"),
                "riskoff_long_risk_mult_factor": riskoff_base.get("riskoff_long_risk_mult_factor"),
                "riskoff_short_risk_mult_factor": riskoff_base.get("riskoff_short_risk_mult_factor"),
            },
            "riskoff+panic base",
        )
    )
    out.append(
        (
            {
                **risk_off,
                **pop_base,
                "riskoff_tr5_med_pct": riskoff_base.get("riskoff_tr5_med_pct"),
                "riskoff_tr5_lookback_days": riskoff_base.get("riskoff_tr5_lookback_days"),
                "riskoff_mode": riskoff_base.get("riskoff_mode"),
                "riskoff_long_risk_mult_factor": riskoff_base.get("riskoff_long_risk_mult_factor"),
                "riskoff_short_risk_mult_factor": riskoff_base.get("riskoff_short_risk_mult_factor"),
            },
            "riskoff+pop base",
        )
    )
    return out


def _build_st37_refine_shock_variants() -> list[tuple[dict[str, object], str]]:
    out: list[tuple[dict[str, object], str]] = [({}, "shock=off")]
    for on_atr, off_atr in ((13.5, 13.0), (14.0, 13.5), (14.5, 14.0)):
        base = {
            "shock_gate_mode": "surf",
            "shock_detector": "daily_atr_pct",
            "shock_daily_atr_period": 14,
            "shock_daily_on_atr_pct": float(on_atr),
            "shock_daily_off_atr_pct": float(off_atr),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
            "shock_stop_loss_pct_mult": 0.75,
        }
        out.append((base, f"shock=surf daily_atr on={on_atr:g} off={off_atr:g}"))
        for tr_on in (9.0, 10.0, 11.0):
            out.append(({**base, "shock_daily_on_tr_pct": float(tr_on)}, f"shock=surf daily_atr on={on_atr:g} off={off_atr:g} tr_on={tr_on:g}"))
    for detector in ("atr_ratio", "tr_ratio"):
        for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
            out.append(
                (
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": str(detector),
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": float(on_ratio),
                        "shock_off_ratio": float(off_ratio),
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                    },
                    f"shock=surf {detector} on={on_ratio:g} off={off_ratio:g}",
                )
            )
    for lb, on_dd, off_dd in (
        (20, -15.0, -10.0),
        (20, -20.0, -10.0),
        (30, -20.0, -15.0),
    ):
        out.append(
            (
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "daily_drawdown",
                    "shock_drawdown_lookback_days": int(lb),
                    "shock_on_drawdown_pct": float(on_dd),
                    "shock_off_drawdown_pct": float(off_dd),
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_stop_loss_pct_mult": 0.75,
                },
                f"shock=surf daily_dd lb={lb} on={on_dd:g} off={off_dd:g}",
            )
        )
    return out


def _print_top(rows: list[dict], *, title: str, top_n: int, sort_key) -> None:
    print("")
    print(title)
    print("-" * len(title))
    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    for idx, row in enumerate(rows_sorted[: max(1, int(top_n))], start=1):
        pnl = float(row.get("pnl") or 0.0)
        dd = float(row.get("dd") or 0.0)
        roi = float(row.get("roi") or 0.0) * 100.0
        dd_pct = float(row.get("dd_pct") or 0.0) * 100.0
        trades = int(row.get("trades") or 0)
        win = float(row.get("win_rate") or 0.0) * 100.0
        pnl_over_dd = float(row.get("pnl_over_dd") or 0.0)
        note = row.get("note") or ""
        print(
            f"{idx:>2}. tr={trades:>4} win={win:>5.1f}% "
            f"pnl={pnl:>10.1f} dd={dd:>8.1f} pnl/dd={pnl_over_dd:>6.2f} "
            f"roi={roi:>6.2f}% dd%={dd_pct:>6.2f}% {note}"
        )


def _print_leaderboards(rows: list[dict], *, title: str, top_n: int) -> None:
    _print_top(rows, title=f"{title} — Top by pnl/dd", top_n=top_n, sort_key=_score_row_pnl_dd)
    _print_top(rows, title=f"{title} — Top by pnl", top_n=top_n, sort_key=_score_row_pnl)


def _seed_groups_from_path(seed_path: Path) -> list[dict]:
    try:
        payload = json.loads(seed_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid seed milestones payload: {seed_path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid seed milestones payload: {seed_path}")
    raw_groups = payload.get("groups") or []
    if not isinstance(raw_groups, list):
        raise SystemExit(f"Invalid seed milestones groups: {seed_path}")
    return raw_groups


def _seed_sort_key_default(item: dict) -> tuple:
    m = item.get("metrics") or {}
    return (
        float(m.get("pnl_over_dd") or float("-inf")),
        float(m.get("pnl") or float("-inf")),
        float(m.get("win_rate") or 0.0),
        int(m.get("trades") or 0),
    )


_SEED_SCORE_REGISTRY: dict[str, object] = {
    "pnl_dd": lambda item: _score_row_pnl_dd(item.get("metrics") or {}),
    "pnl": lambda item: _score_row_pnl(item.get("metrics") or {}),
    "roi": lambda item: float((item.get("metrics") or {}).get("roi") or 0.0),
    "win": lambda item: float((item.get("metrics") or {}).get("win_rate") or 0.0),
    "trades": lambda item: (
        int((item.get("metrics") or {}).get("trades") or 0),
        float((item.get("metrics") or {}).get("pnl") or float("-inf")),
        float((item.get("metrics") or {}).get("pnl_over_dd") or float("-inf")),
    ),
    "roi_dd": lambda item: float((item.get("metrics") or {}).get("roi_over_dd_pct") or 0.0),
    "stability_roi_dd": lambda item: (
        float((item.get("eval") or {}).get("stability_min_roi_dd") or 0.0),
        float((item.get("metrics") or {}).get("roi_over_dd_pct") or 0.0),
        float((item.get("metrics") or {}).get("pnl_over_dd") or 0.0),
        float((item.get("metrics") or {}).get("roi") or 0.0),
        float((item.get("metrics") or {}).get("pnl") or 0.0),
    ),
}


def _seed_candidate_key(item: dict) -> str:
    raw = {
        "strategy": item.get("strategy") if isinstance(item.get("strategy"), dict) else {},
        "filters": item.get("filters") if isinstance(item.get("filters"), dict) else None,
    }
    return json.dumps(raw, sort_keys=True, default=str)


def _seed_rank_slice(
    candidates: list[dict],
    *,
    scorer: str,
    top_n: int,
) -> list[dict]:
    score_fn = _SEED_SCORE_REGISTRY.get(str(scorer))
    n = max(0, int(top_n))
    if n <= 0 or not callable(score_fn):
        return []
    return sorted(candidates, key=lambda item, fn=score_fn: fn(item), reverse=True)[:n]


def _seed_dedupe_candidates(
    candidates: list[dict],
    *,
    limit: int | None = None,
    key_fn=_seed_candidate_key,
) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    max_items = None if limit is None else max(0, int(limit))
    for item in candidates:
        key = str(key_fn(item))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _seed_best_by_family(
    candidates: list[dict],
    *,
    family_key_fn,
    scorer: str = "pnl_dd",
) -> list[dict]:
    score_fn = _SEED_SCORE_REGISTRY.get(str(scorer))
    if not callable(score_fn):
        return []
    best_by_family: dict[object, dict] = {}
    for item in candidates:
        fam = family_key_fn(item)
        prev = best_by_family.get(fam)
        if prev is None or score_fn(item) > score_fn(prev):
            best_by_family[fam] = item
    return sorted(best_by_family.values(), key=lambda item, fn=score_fn: fn(item), reverse=True)


def _seed_candidates_for_context(
    *,
    raw_groups: list[dict],
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    min_trades: int = 0,
    predicate: Callable[[dict, dict, dict, dict], bool] | None = None,
) -> list[dict]:
    out: list[dict] = []
    symbol_norm = str(symbol).strip().upper()
    bar_norm = str(signal_bar_size).strip().lower()
    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strat = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strat, dict) or not isinstance(metrics, dict):
            continue
        if str(strat.get("instrument", "spot") or "spot").strip().lower() != "spot":
            continue
        if str(entry.get("symbol") or strat.get("symbol") or "").strip().upper() != symbol_norm:
            continue
        if str(strat.get("signal_bar_size") or "").strip().lower() != bar_norm:
            continue
        if bool(strat.get("signal_use_rth")) != bool(use_rth):
            continue
        if int(min_trades) > 0:
            try:
                trades = int(metrics.get("trades") or 0)
            except (TypeError, ValueError):
                trades = 0
            if int(trades) < int(min_trades):
                continue
        if predicate is not None and not bool(predicate(group, entry, strat, metrics)):
            continue
        candidate = {
            "group_name": str(group.get("name") or ""),
            "strategy": strat,
            "filters": group.get("filters") if isinstance(group.get("filters"), dict) else None,
            "metrics": metrics,
        }
        eval_payload = group.get("_eval")
        if isinstance(eval_payload, dict):
            candidate["eval"] = dict(eval_payload)
        out.append(candidate)
    return out


def _resolve_seed_milestones_path(
    *,
    seed_milestones: str | None,
    axis_tag: str,
    default_path: str | None = None,
) -> Path:
    if seed_milestones:
        seed_path = Path(str(seed_milestones))
    elif default_path:
        seed_path = Path(str(default_path))
    else:
        raise SystemExit(f"--axis {axis_tag} requires --seed-milestones <milestones.json>")
    if not seed_path.exists():
        raise SystemExit(f"--axis {axis_tag} requires --seed-milestones (missing {seed_path})")
    return seed_path


def _load_seed_candidates(
    *,
    seed_milestones: str | None,
    axis_tag: str,
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    default_path: str | None = None,
    min_trades: int = 0,
    predicate: Callable[[dict, dict, dict, dict], bool] | None = None,
) -> tuple[Path, list[dict]]:
    seed_path = _resolve_seed_milestones_path(
        seed_milestones=seed_milestones,
        axis_tag=axis_tag,
        default_path=default_path,
    )
    candidates = _seed_candidates_for_context(
        raw_groups=_seed_groups_from_path(seed_path),
        symbol=symbol,
        signal_bar_size=signal_bar_size,
        use_rth=use_rth,
        min_trades=int(min_trades),
        predicate=predicate,
    )
    return seed_path, candidates


def _seed_top_candidates(
    candidates: list[dict],
    *,
    seed_top: int,
    sort_key: Callable[[dict], tuple] = _seed_sort_key_default,
) -> list[dict]:
    return sorted(candidates, key=sort_key, reverse=True)[: max(1, int(seed_top))]


def _select_seed_candidates_default(candidates: list[dict], *, seed_top: int) -> list[dict]:
    return _seed_top_candidates(candidates, seed_top=int(seed_top))


def _select_seed_candidates_champ_refine(candidates: list[dict], *, seed_top: int) -> list[dict]:
    def _family_key(item: dict) -> tuple:
        st = item.get("strategy") or {}
        return (
            str(st.get("ema_preset") or ""),
            str(st.get("ema_entry_mode") or ""),
            str(st.get("regime_mode") or ""),
            str(st.get("regime_bar_size") or ""),
            str(st.get("spot_exit_mode") or ""),
        )

    family_winners = _seed_best_by_family(
        candidates,
        family_key_fn=_family_key,
        scorer="pnl_dd",
    )
    seed_pool = list(family_winners[: max(1, int(seed_top))])
    seed_pool.extend(_seed_rank_slice(candidates, scorer="pnl", top_n=max(5, int(seed_top) // 4)))
    seed_pool.extend(_seed_rank_slice(candidates, scorer="roi", top_n=max(5, int(seed_top) // 4)))
    seed_pool.extend(_seed_rank_slice(candidates, scorer="win", top_n=max(5, int(seed_top) // 4)))
    return _seed_dedupe_candidates(seed_pool, limit=int(seed_top), key_fn=_seed_candidate_key)


def _select_seed_candidates_exit_pivot(candidates: list[dict], *, seed_top: int) -> list[dict]:
    seed_pool = []
    seed_pool.extend(_seed_rank_slice(candidates, scorer="pnl", top_n=max(1, int(seed_top))))
    seed_pool.extend(_seed_rank_slice(candidates, scorer="trades", top_n=max(1, min(int(seed_top), 5))))
    return _seed_dedupe_candidates(seed_pool, limit=None, key_fn=_seed_candidate_key)


def _select_seed_candidates_st37_refine(candidates: list[dict], *, seed_top: int) -> list[dict]:
    cand_sorted = _seed_rank_slice(
        candidates,
        scorer="stability_roi_dd",
        top_n=len(candidates),
    )
    return _seed_dedupe_candidates(
        cand_sorted,
        limit=int(seed_top),
        key_fn=_seed_candidate_key,
    )


_SEED_SELECTION_POLICY_REGISTRY: dict[str, Callable[..., list[dict]]] = {
    "default": _select_seed_candidates_default,
    "champ_refine": _select_seed_candidates_champ_refine,
    "exit_pivot": _select_seed_candidates_exit_pivot,
    "st37_refine": _select_seed_candidates_st37_refine,
}


def _seed_select_candidates(
    candidates: list[dict],
    *,
    seed_top: int,
    policy: str = "default",
) -> list[dict]:
    seed_top_i = max(1, int(seed_top))
    selector = _SEED_SELECTION_POLICY_REGISTRY.get(str(policy).strip().lower() or "default")
    if not callable(selector):
        selector = _select_seed_candidates_default
    try:
        selected = selector(candidates, seed_top=seed_top_i)  # type: ignore[misc]
    except TypeError:
        selected = selector(candidates, seed_top_i)  # type: ignore[misc]
    if not isinstance(selected, list):
        return []
    return [item for item in selected if isinstance(item, dict)]


def _load_spot_milestones() -> dict | None:
    path = Path(__file__).resolve().parent / "spot_milestones.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _milestone_entry_for(
    milestones: dict | None,
    *,
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    sort_by: str,
    prefer_realism: bool = False,
) -> tuple[dict, dict | None, dict] | None:
    if not milestones:
        return None

    groups = milestones.get("groups") or []
    candidates: list[tuple[dict, dict | None, dict]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strategy = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strategy, dict) or not isinstance(metrics, dict):
            continue
        if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
            continue
        if str(strategy.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
            continue
        if bool(strategy.get("signal_use_rth")) != bool(use_rth):
            continue
        if prefer_realism:
            if str(strategy.get("spot_entry_fill_mode") or "").strip().lower() != "next_open":
                continue
            if not bool(strategy.get("spot_intrabar_exits")):
                continue
            try:
                comm = float(strategy.get("spot_commission_per_share") or 0.0)
            except (TypeError, ValueError):
                comm = 0.0
            try:
                comm_min = float(strategy.get("spot_commission_min") or 0.0)
            except (TypeError, ValueError):
                comm_min = 0.0
            if comm <= 0.0 and comm_min <= 0.0:
                continue
        filters = group.get("filters")
        candidates.append((strategy, filters if isinstance(filters, dict) else None, metrics))

    if not candidates:
        return None

    def _score(c: tuple[dict, dict | None, dict]) -> tuple:
        _, _, m = c
        if str(sort_by).strip().lower() == "pnl":
            return _score_row_pnl(m)
        return _score_row_pnl_dd(m)

    return sorted(candidates, key=_score, reverse=True)[0]


def _apply_milestone_base(
    cfg: ConfigBundle, *, strategy: dict, filters: dict | None
) -> ConfigBundle:
    # Milestone strategies come from `asdict(StrategyConfig)` which flattens nested dataclasses.
    # Only copy scalar knobs we know are safe/needed for backtest reproduction/sweeps.
    keep_keys = (
        "ema_preset",
        "ema_entry_mode",
        "entry_confirm_bars",
        "entry_signal",
        "orb_window_mins",
        "orb_risk_reward",
        "orb_target_mode",
        "orb_open_time_et",
        "regime_mode",
        "regime_bar_size",
        "regime_ema_preset",
        "supertrend_atr_period",
        "supertrend_multiplier",
        "supertrend_source",
        "regime2_mode",
        "regime2_bar_size",
        "regime2_ema_preset",
        "regime2_supertrend_atr_period",
        "regime2_supertrend_multiplier",
        "regime2_supertrend_source",
        "spot_exit_mode",
        "spot_atr_period",
        "spot_pt_atr_mult",
        "spot_sl_atr_mult",
        "spot_profit_target_pct",
        "spot_stop_loss_pct",
        "spot_exit_time_et",
        "spot_close_eod",
        "spot_entry_fill_mode",
        "spot_flip_exit_fill_mode",
        "spot_intrabar_exits",
        "spot_spread",
        "spot_commission_per_share",
        "spot_commission_min",
        "spot_slippage_per_share",
        "spot_mark_to_market",
        "spot_drawdown_mode",
        "spot_sizing_mode",
        "spot_notional_pct",
        "spot_risk_pct",
        "spot_short_risk_mult",
        "spot_max_notional_pct",
        "spot_min_qty",
        "spot_max_qty",
        "exit_on_signal_flip",
        "flip_exit_mode",
        "flip_exit_gate_mode",
        "flip_exit_min_hold_bars",
        "flip_exit_only_if_profit",
        "tick_gate_mode",
        "tick_gate_symbol",
        "tick_gate_exchange",
        "tick_band_ma_period",
        "tick_width_z_lookback",
        "tick_width_z_enter",
        "tick_width_z_exit",
        "tick_width_slope_lookback",
        "tick_neutral_policy",
        "tick_direction_policy",
    )

    strat_over: dict[str, object] = {}
    for key in keep_keys:
        if key in strategy:
            strat_over[key] = strategy[key]

    out = replace(cfg, strategy=replace(cfg.strategy, **strat_over))

    if not filters:
        return replace(out, strategy=replace(out.strategy, filters=None))

    f = _parse_filters(filters)
    if _filters_payload(f) is None:
        f = None
    return replace(out, strategy=replace(out.strategy, filters=f))
# endregion


# region CLI
class _SpotSweepsHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if action.help is argparse.SUPPRESS:
            return help_text
        if "%(default)" in help_text:
            return help_text
        if action.default is argparse.SUPPRESS or action.default is None:
            return help_text
        return f"{help_text} (default: %(default)s)"


_BASE_HELP_NOTES: dict[str, str] = {
    "default": "Minimal baseline profile (EMA 2/4, fixed pct exits, no extra gates).",
    "champion": "Loads highest pnl/dd winner from spot milestones for current symbol/bar/rth (preferred).",
    "champion_pnl": "Loads highest pnl winner from spot milestones for current symbol/bar/rth.",
    "dual_regime": "Regime + regime2 baseline profile used for dual-regime refinement flows.",
}

_AXIS_HELP_NOTES: dict[str, str] = {
    "ema": "EMA preset timing sweep (direction speed).",
    "entry_mode": "Entry semantics sweep (cross vs trend) with directional rules.",
    "combo_fast": "Bounded 3-stage multi-axis funnel for fast corner hunting.",
    "combo_full": "Full suite run: one-axis sweeps + joint sweeps + bounded funnels (very slow).",
    "squeeze": "Regime2 squeeze flow then small vol/TOD/confirm pocket.",
    "volume": "Volume gate sweep (ratio threshold x EMA period).",
    "rv": "Realized volatility gate sweep.",
    "tod": "Time-of-day gate sweep (ET entry windows).",
    "tod_interaction": "TOD interaction micro-grid (overnight-heavy windows).",
    "perm_joint": "Joint permissions sweep: TOD x spread x volume.",
    "weekday": "Weekday entry gating sweep.",
    "exit_time": "Fixed ET flatten-time sweep.",
    "atr": "Core ATR exits sweep.",
    "atr_fine": "Fine ATR PT/SL pocket sweep.",
    "atr_ultra": "Ultra-fine ATR PT/SL micro-grid.",
    "r2_atr": "Regime2 x ATR joint sweep.",
    "r2_tod": "Regime2 x TOD joint sweep.",
    "ema_perm_joint": "EMA x permission gates joint sweep.",
    "tick_perm_joint": "TICK gate x permission gates joint sweep.",
    "regime_atr": "Regime (ST) x ATR joint sweep.",
    "ema_regime": "EMA x regime bias joint sweep.",
    "chop_joint": "Chop-killer joint sweep (slope x cooldown x skip-open).",
    "ema_atr": "EMA x ATR exits joint sweep.",
    "tick_ema": "TICK width x EMA joint sweep.",
    "ptsl": "Fixed-percent PT/SL exits sweep (non-ATR).",
    "hf_scalp": "HF scalp cadence sweep.",
    "hold": "Flip-exit minimum-hold-bars sweep.",
    "spot_short_risk_mult": "Short-side risk multiplier sweep.",
    "orb": "ORB sweep (open-time, window, target semantics).",
    "orb_joint": "ORB x regime x TICK joint sweep.",
    "frontier": "Frontier sweep over shortlist dimensions.",
    "regime": "Primary Supertrend regime params/timeframe sweep.",
    "regime2": "Secondary (regime2) Supertrend params/timeframe sweep.",
    "regime2_ema": "Regime2 EMA confirm sweep.",
    "joint": "Targeted regime x regime2 interaction hunt.",
    "micro_st": "Micro sweep around current ST/ST2 neighborhood.",
    "flip_exit": "Flip-exit semantics/gating sweep.",
    "confirm": "Entry confirmation bars sweep.",
    "spread": "EMA spread quality gate sweep.",
    "spread_fine": "Fine EMA spread threshold sweep.",
    "spread_down": "Directional down-side EMA spread gate sweep.",
    "slope": "EMA slope quality gate sweep.",
    "slope_signed": "Signed directional EMA slope gate sweep.",
    "cooldown": "Entry cooldown bars sweep.",
    "skip_open": "Skip-first-bars-after-open sweep.",
    "shock": "Shock detector/mode/threshold sweep.",
    "risk_overlays": "Risk overlay family sweep (riskoff/riskpanic/riskpop).",
    "loosen": "Loosenings sweep (single-position parity + EOD behavior).",
    "loosen_atr": "Loosenings x ATR joint sweep (single-position parity).",
    "tick": "Raschke-style $TICK width gate sweep.",
    "gate_matrix": "Bounded cross-product matrix of permission gates.",
    "seeded_refine": "Seeded refinement bundle (champ_refine + overlay_family + st37_refine).",
    "champ_refine": "Champion-centered refinement around interaction edges.",
    "st37_refine": "v31/st37-style refinement stages (permissions, overlays, exits).",
    "shock_alpha_refine": "Shock alpha refinement pocket around winning detectors.",
    "shock_velocity_refine": "Shock velocity refinement (narrow grid).",
    "shock_velocity_refine_wide": "Shock velocity refinement (wide grid).",
    "shock_throttle_refine": "Overlay family member: shock throttle core refine.",
    "shock_throttle_tr_ratio": "Overlay family member: TR-ratio throttle sweep.",
    "shock_throttle_drawdown": "Overlay family member: drawdown throttle sweep.",
    "riskpanic_micro": "Overlay family member: riskpanic micro-grid.",
    "overlay_family": "Run one/all seeded overlay families (see --overlay-family-kind).",
    "exit_pivot": "Exit-style pivot sweep around active champion semantics.",
}

_INTERNAL_FLAG_HELP_LINES: tuple[str, ...] = (
    "--gate-matrix-stage2 PATH: gate_matrix stage2 worker payload JSON.",
    "--gate-matrix-worker INT / --gate-matrix-workers INT / --gate-matrix-out PATH: gate_matrix shard controls.",
    "--gate-matrix-run-min-trades INT: stage-specific min trades override for gate_matrix.",
    "--combo-fast-stage1 PATH / --combo-fast-stage2 PATH / --combo-fast-stage3 PATH: combo_fast stage payloads.",
    "--combo-fast-worker INT / --combo-fast-workers INT / --combo-fast-out PATH: combo_fast shard controls.",
    "--combo-fast-run-min-trades INT: stage-specific min trades override for combo_fast.",
    "--risk-overlays-worker INT / --risk-overlays-workers INT / --risk-overlays-out PATH: risk_overlays shard controls.",
    "--risk-overlays-run-min-trades INT: stage-specific min trades override for risk_overlays.",
    "--seeded-micro-stage PATH / --seeded-micro-worker INT / --seeded-micro-workers INT / --seeded-micro-out PATH.",
    "--seeded-micro-run-min-trades INT: stage-specific min trades override for seeded micro stages.",
    "--champ-refine-stage3a PATH / --champ-refine-stage3b PATH: champ_refine stage payloads.",
    "--champ-refine-worker INT / --champ-refine-workers INT / --champ-refine-out PATH: champ_refine shard controls.",
    "--champ-refine-run-min-trades INT: stage-specific min trades override for champ_refine.",
    "--st37-refine-stage1 PATH / --st37-refine-stage2 PATH: st37_refine stage payloads.",
    "--st37-refine-worker INT / --st37-refine-workers INT / --st37-refine-out PATH: st37_refine shard controls.",
    "--st37-refine-run-min-trades INT: stage-specific min trades override for st37_refine.",
    "--shock-velocity-worker INT / --shock-velocity-workers INT / --shock-velocity-out PATH: shock_velocity shard controls.",
)


def _spot_sweeps_help_epilog() -> str:
    axis_lines: list[str] = [
        "Axis Catalog:",
        "  all                        Serial axis_all plan (or parallel when --jobs > 1 with --offline).",
    ]
    for axis_name in _AXIS_CHOICES:
        axis_desc = str(_AXIS_HELP_NOTES.get(axis_name) or "See source.")
        axis_lines.append(f"  {axis_name:<26} {axis_desc}")

    base_lines = ["Base Profiles:"]
    for key in ("default", "champion", "champion_pnl", "dual_regime"):
        base_lines.append(f"  {key:<14} {_BASE_HELP_NOTES.get(key, '')}")

    internal_lines = ["Internal Orchestration Flags (auto-managed by parent stages):"]
    for raw in _INTERNAL_FLAG_HELP_LINES:
        internal_lines.append(f"  {raw}")

    return "\n".join(
        (
            "Examples:",
            "  python -m tradebot.backtest spot --offline --axis all --start 2025-01-08 --end 2026-01-08",
            "  python -m tradebot.backtest spot --axis regime --symbol MNQ --bar-size '1 hour'",
            "  python -m tradebot.backtest spot --axis combo_full --offline --jobs 8",
            "  python -m tradebot.backtest spot --axis champ_refine --seed-milestones my_pool.json --seed-top 30",
            "",
            "Execution Notes:",
            "  - --start/--end are inclusive dates (YYYY-MM-DD).",
            "  - --bar-size is signal timeframe; ORB sweeps always use 15m signal bars internally.",
            "  - --spot-exec-bar-size overrides only execution bars; signals still use --bar-size.",
            "  - --jobs <= 0 (or omitted) means auto CPU count; effective jobs are clamped to available CPUs.",
            "  - Parallel worker modes require --offline to avoid concurrent IBKR sessions.",
            "  - --cache-dir stores bar cache plus run_cfg cache DB (spot_sweeps_run_cfg_cache.sqlite3).",
            "  - --write-milestones emits UI preset candidates filtered by milestone thresholds.",
            "  - --seed-milestones can override champion source for champion/champion_pnl base modes.",
            "",
            *base_lines,
            "",
            *axis_lines,
            "",
            *internal_lines,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled spot evolution sweeps for the spot backtest engine.\n"
            "Canonical entrypoint: python -m tradebot.backtest spot ..."
        ),
        formatter_class=_SpotSweepsHelpFormatter,
        epilog=_spot_sweeps_help_epilog(),
    )
    parser.add_argument(
        "--symbol",
        default="MNQ",
        help="Instrument symbol to backtest (uppercased).",
    )
    parser.add_argument(
        "--start",
        default="2025-01-08",
        help="Inclusive backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2026-01-08",
        help="Inclusive backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--bar-size",
        default="1 hour",
        help="Signal bar size (e.g. '30 mins', '1 hour'). ORB axis uses 15m regardless.",
    )
    parser.add_argument(
        "--spot-exec-bar-size",
        default=None,
        help=(
            "Optional execution timeframe for fill simulation (e.g. '5 mins'). "
            "Signals still run on --bar-size."
        ),
    )
    parser.add_argument(
        "--use-rth",
        action="store_true",
        default=False,
        help="Use regular trading hours bars (RTH) instead of full-session bars.",
    )
    parser.add_argument(
        "--cache-dir",
        default="db",
        help="Cache root for historical bars and sweep caches.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Use cached bars only (no IBKR calls). Requires cache to be present.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Parallelism for --axis all/combo_full (spawns per-axis worker processes), plus internal sharding for "
            "risk_overlays and gate_matrix stage2. 0/omitted = auto (CPU count). Use --offline."
        ),
    )
    parser.add_argument(
        "--base",
        default="champion",
        choices=("default", "champion", "champion_pnl", "dual_regime"),
        help="Base profile to mutate before running the selected axis (see Base Profiles below).",
    )
    parser.add_argument(
        "--max-open-trades",
        type=int,
        default=None,
        help="Deprecated no-op for spot; kept only for CLI compatibility.",
    )
    parser.add_argument(
        "--close-eod",
        action="store_true",
        default=False,
        help="Force end-of-day flattening on spot positions.",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=False,
        help="Force spot to long-only (directional_spot = {'up': BUY 1}, no shorts).",
    )
    parser.add_argument(
        "--realism2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Spot realism (default): next-open fills, intrabar exits, liquidation marking, intrabar drawdown, "
            "position sizing, commission minimums, and stop gap handling. Defaults: spread=$0.01, "
            "commission=$0.005/share (min $1.00), risk sizing=1%% equity risk, max notional=50%%."
        ),
    )
    parser.add_argument(
        "--spot-spread",
        type=float,
        default=None,
        help="Spot spread in price units (e.g. 0.01). If omitted: 0.01 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-commission",
        type=float,
        default=None,
        help=(
            "Spot commission per share/contract (price units). "
            "If omitted: 0.005 with --realism2, else 0.0."
        ),
    )
    parser.add_argument(
        "--spot-commission-min",
        type=float,
        default=None,
        help="Spot commission minimum per order (price units). If omitted: 1.0 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-slippage",
        type=float,
        default=None,
        help="Spot slippage per share (price units). If omitted: 0.0.",
    )
    parser.add_argument(
        "--spot-sizing-mode",
        default=None,
        choices=("fixed", "notional_pct", "risk_pct"),
        help="Spot sizing mode (v2): fixed qty, %% notional, or %% equity risk-to-stop. If omitted: risk_pct with --realism2, else fixed.",
    )
    parser.add_argument(
        "--spot-risk-pct",
        type=float,
        default=None,
        help="Risk per trade as fraction of equity (v2). If omitted: 0.01 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-notional-pct",
        type=float,
        default=None,
        help="Notional allocation per trade as fraction of equity (v2). If omitted: 0.0.",
    )
    parser.add_argument(
        "--spot-max-notional-pct",
        type=float,
        default=None,
        help="Max notional per trade as fraction of equity (v2). If omitted: 0.50 with --realism2, else 1.0.",
    )
    parser.add_argument(
        "--spot-min-qty",
        type=int,
        default=None,
        help="Min shares/contracts per trade (v2). If omitted: 1.",
    )
    parser.add_argument(
        "--spot-max-qty",
        type=int,
        default=None,
        help="Max shares/contracts per trade (v2); 0 means unlimited. If omitted: 0.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=100,
        help="Minimum trades required for a config to be kept in sweep outputs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many leaderboard rows to print per sweep.",
    )
    parser.add_argument(
        "--write-milestones",
        action="store_true",
        default=False,
        help="Write tradebot/backtest/spot_milestones.json from eligible sweep results (UI presets).",
    )
    parser.add_argument(
        "--merge-milestones",
        action="store_true",
        default=False,
        help="Merge eligible presets into an existing milestones JSON instead of overwriting from scratch.",
    )
    parser.add_argument(
        "--milestones-out",
        default="tradebot/backtest/spot_milestones.json",
        help="Output JSON path for --write-milestones.",
    )
    parser.add_argument(
        "--milestone-min-win",
        type=float,
        default=0.55,
        help="Milestone eligibility: minimum win rate (0..1).",
    )
    parser.add_argument(
        "--milestone-min-trades",
        type=int,
        default=200,
        help="Milestone eligibility: minimum trades.",
    )
    parser.add_argument(
        "--milestone-min-pnl-dd",
        type=float,
        default=8.0,
        help="Milestone eligibility: minimum pnl/dd ratio.",
    )
    parser.add_argument(
        "--milestone-add-top-pnl-dd",
        type=int,
        default=0,
        help=(
            "When used with --merge-milestones, limits how many NEW presets from this run are added "
            "(top by pnl/dd). 0 = no limit."
        ),
    )
    parser.add_argument(
        "--milestone-add-top-pnl",
        type=int,
        default=0,
        help=(
            "When used with --merge-milestones, limits how many NEW presets from this run are added "
            "(top by pnl). 0 = no limit."
        ),
    )
    parser.add_argument(
        "--seed-milestones",
        default=None,
        help=(
            "Optional milestones JSON used as a seed pool for seeded refine sweeps "
            "(e.g. --axis champ_refine). Also used as champion source override for --base champion/champion_pnl."
        ),
    )
    parser.add_argument(
        "--seed-top",
        type=int,
        default=20,
        help="How many seeds to take from --seed-milestones (after filtering).",
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=("all", *_AXIS_CHOICES),
        metavar="AXIS",
        help="Axis to execute. Use 'all' for the axis_all plan. See Axis Catalog below.",
    )
    parser.add_argument(
        "--overlay-family-kind",
        default="all",
        choices=("all", "shock_throttle_refine", "shock_throttle_tr_ratio", "shock_throttle_drawdown", "riskpanic_micro"),
        metavar="FAMILY",
        help=(
            "Used by --axis overlay_family. "
            "'all' runs all seeded overlay families; otherwise runs only the selected family."
        ),
    )
    parser.add_argument(
        "--risk-overlays-riskoff-trs",
        default=None,
        help="CSV float override for risk_overlays riskoff TR%% median thresholds.",
    )
    parser.add_argument(
        "--risk-overlays-riskpanic-trs",
        default=None,
        help="CSV float override for risk_overlays riskpanic TR%% median thresholds.",
    )
    parser.add_argument(
        "--risk-overlays-riskpanic-long-factors",
        default=None,
        help="CSV float override for riskpanic_long_risk_mult_factor (e.g. 1,0.8,0.6,0.4).",
    )
    parser.add_argument(
        "--risk-overlays-riskpop-trs",
        default=None,
        help="CSV float override for risk_overlays riskpop TR%% median thresholds.",
    )
    parser.add_argument(
        "--risk-overlays-skip-pop",
        action="store_true",
        default=False,
        help="risk_overlays: skip riskpop stage (riskoff+riskpanic only).",
    )
    # Internal flags (used by combo_full/gate_matrix parallel sharding).
    parser.add_argument("--gate-matrix-stage2", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gate-matrix-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gate-matrix-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gate-matrix-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gate-matrix-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-stage1", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-stage2", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-stage3", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-fast-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--risk-overlays-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--risk-overlays-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--risk-overlays-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--risk-overlays-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeded-micro-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeded-micro-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeded-micro-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeded-micro-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeded-micro-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-stage3a", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-stage3b", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--champ-refine-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-stage1", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-stage2", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--st37-refine-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shock-velocity-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shock-velocity-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shock-velocity-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_open_trades is not None:
        print("[compat] --max-open-trades is deprecated and ignored for spot backtests.", flush=True)

    def _default_jobs() -> int:
        detected = os.cpu_count()
        if detected is None:
            return 1
        try:
            detected_i = int(detected)
        except (TypeError, ValueError):
            return 1
        return max(1, detected_i)

    try:
        jobs_raw = int(args.jobs) if args.jobs is not None else 0
    except (TypeError, ValueError):
        jobs_raw = 0
    detected_jobs = _default_jobs()
    jobs = detected_jobs if int(jobs_raw) <= 0 else min(int(jobs_raw), int(detected_jobs))
    jobs = max(1, int(jobs))

    symbol = str(args.symbol).strip().upper()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    use_rth = bool(args.use_rth)
    offline = bool(args.offline)
    cache_dir = Path(args.cache_dir)
    start_dt = datetime.combine(start, time(0, 0))
    end_dt = datetime.combine(end, time(23, 59))
    signal_bar_size = str(args.bar_size).strip() or "1 hour"
    spot_exec_bar_size = str(args.spot_exec_bar_size).strip() if args.spot_exec_bar_size else None
    if spot_exec_bar_size and parse_bar_size(spot_exec_bar_size) is None:
        raise SystemExit(f"Invalid --spot-exec-bar-size: {spot_exec_bar_size!r}")
    close_eod = bool(args.close_eod)
    long_only = bool(args.long_only)
    realism2 = bool(args.realism2)
    spot_spread = float(args.spot_spread) if args.spot_spread is not None else (0.01 if realism2 else 0.0)
    spot_commission = (
        float(args.spot_commission)
        if args.spot_commission is not None
        else (0.005 if realism2 else 0.0)
    )
    spot_commission_min = (
        float(args.spot_commission_min)
        if args.spot_commission_min is not None
        else (1.0 if realism2 else 0.0)
    )
    spot_slippage = float(args.spot_slippage) if args.spot_slippage is not None else 0.0

    sizing_mode = (
        str(args.spot_sizing_mode).strip().lower()
        if args.spot_sizing_mode is not None
        else ("risk_pct" if realism2 else "fixed")
    )
    if sizing_mode not in ("fixed", "notional_pct", "risk_pct"):
        sizing_mode = "fixed"
    spot_risk_pct = float(args.spot_risk_pct) if args.spot_risk_pct is not None else (0.01 if realism2 else 0.0)
    spot_notional_pct = (
        float(args.spot_notional_pct) if args.spot_notional_pct is not None else 0.0
    )
    spot_max_notional_pct = (
        float(args.spot_max_notional_pct) if args.spot_max_notional_pct is not None else (0.50 if realism2 else 1.0)
    )
    spot_min_qty = int(args.spot_min_qty) if args.spot_min_qty is not None else 1
    spot_max_qty = int(args.spot_max_qty) if args.spot_max_qty is not None else 0
    run_min_trades = int(args.min_trades)
    if args.gate_matrix_run_min_trades is not None:
        try:
            run_min_trades = int(args.gate_matrix_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if args.combo_fast_run_min_trades is not None:
        try:
            run_min_trades = int(args.combo_fast_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if args.risk_overlays_run_min_trades is not None:
        try:
            run_min_trades = int(args.risk_overlays_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if args.seeded_micro_run_min_trades is not None:
        try:
            run_min_trades = int(args.seeded_micro_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if args.champ_refine_run_min_trades is not None:
        try:
            run_min_trades = int(args.champ_refine_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if args.st37_refine_run_min_trades is not None:
        try:
            run_min_trades = int(args.st37_refine_run_min_trades)
        except (TypeError, ValueError):
            run_min_trades = int(args.min_trades)
    if bool(args.write_milestones):
        run_min_trades = min(run_min_trades, int(args.milestone_min_trades))

    if offline:
        _require_offline_cache_or_die(
            cache_dir=cache_dir,
            symbol=symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=signal_bar_size,
            use_rth=use_rth,
        )
        if spot_exec_bar_size and str(spot_exec_bar_size) != str(signal_bar_size):
            _require_offline_cache_or_die(
                cache_dir=cache_dir,
                symbol=symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=spot_exec_bar_size,
                use_rth=use_rth,
            )

    data = IBKRHistoricalData()
    if offline:
        is_future = symbol in ("MNQ", "MBT")
        exchange = "CME" if is_future else "SMART"
        multiplier = 1.0
        if is_future:
            multiplier = {"MNQ": 2.0, "MBT": 0.1}.get(symbol, 1.0)
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=0.01)
    else:
        try:
            _, meta = data.resolve_contract(symbol, exchange=None)
        except Exception as exc:
            raise SystemExit(
                "IBKR API connection failed. Start IB Gateway / TWS (or run with --offline after prefetching cached bars)."
            ) from exc

    milestones = _load_spot_milestones()
    # Seeded runs: if a seed milestones file includes a matching strategy for this symbol/bar/rth,
    # prefer it as the "champion" source so we don't have to mutate spot_milestones.json.
    if args.seed_milestones:
        try:
            seed_path = Path(args.seed_milestones)
        except Exception:
            seed_path = None
        if seed_path and seed_path.exists():
            try:
                seed_payload = json.loads(seed_path.read_text())
            except Exception:
                seed_payload = None
            if isinstance(seed_payload, dict):
                base_name = str(args.base).strip().lower()
                if base_name in ("champion", "champion_pnl"):
                    has_match = _milestone_entry_for(
                        seed_payload,
                        symbol=symbol,
                        signal_bar_size=str(signal_bar_size),
                        use_rth=use_rth,
                        sort_by="pnl_dd",
                        prefer_realism=realism2,
                    )
                    if has_match is not None:
                        milestones = seed_payload

    run_calls_total = 0
    run_cfg_cache: dict[tuple, dict | None] = {}
    run_cfg_cache_hits = 0
    run_cfg_fingerprint_hits = 0
    run_cfg_persistent_hits = 0
    run_cfg_persistent_writes = 0
    run_cfg_fingerprint_cache: dict[str, tuple[tuple[tuple[int, object | None, object | None], tuple[int, object | None, object | None], tuple[int, object | None, object | None]], dict | None]] = {}
    _RUN_CFG_CACHE_MISS = object()
    run_cfg_persistent_path = cache_dir / "spot_sweeps_run_cfg_cache.sqlite3"
    run_cfg_persistent_conn: sqlite3.Connection | None = None
    run_cfg_persistent_enabled = True
    run_cfg_persistent_lock = threading.Lock()

    def _run_cfg_persistent_conn() -> sqlite3.Connection | None:
        nonlocal run_cfg_persistent_conn, run_cfg_persistent_enabled
        if not bool(run_cfg_persistent_enabled):
            return None
        if run_cfg_persistent_conn is not None:
            return run_cfg_persistent_conn
        try:
            conn = sqlite3.connect(
                str(run_cfg_persistent_path),
                timeout=15.0,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_cfg_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            run_cfg_persistent_conn = conn
            return conn
        except Exception:
            run_cfg_persistent_enabled = False
            run_cfg_persistent_conn = None
            return None

    def _run_cfg_persistent_key(*, cfg_key: str, ctx_sig: tuple) -> str:
        raw = {
            "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
            "cfg_key": str(cfg_key),
            "ctx_sig": ctx_sig,
            "run_min_trades": int(run_min_trades),
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _run_cfg_persistent_get(*, cache_key: str) -> dict | None | object:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return _RUN_CFG_CACHE_MISS
        try:
            with run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT payload_json FROM run_cfg_cache WHERE cache_key=?",
                    (str(cache_key),),
                ).fetchone()
        except Exception:
            return _RUN_CFG_CACHE_MISS
        if row is None:
            return _RUN_CFG_CACHE_MISS
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            return _RUN_CFG_CACHE_MISS
        if payload is None:
            return None
        return dict(payload) if isinstance(payload, dict) else _RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_set(*, cache_key: str, payload: dict | None) -> None:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO run_cfg_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (str(cache_key), payload_json, float(pytime.time())),
                )
        except Exception:
            return

    def _merge_filters(base_filters: FiltersConfig | None, overrides: dict[str, object]) -> FiltersConfig | None:
        """Merge base filters with overrides, where `None` deletes a key.

        Used to build joint permission sweeps without being constrained by the combo_fast funnel.
        """
        merged: dict[str, object] = dict(_filters_payload(base_filters) or {})
        for key, val in overrides.items():
            if val is None:
                merged.pop(key, None)
            else:
                merged[key] = val

        # Keep TOD gating consistent (both-or-neither).
        if ("entry_start_hour_et" in merged) ^ ("entry_end_hour_et" in merged):
            merged.pop("entry_start_hour_et", None)
            merged.pop("entry_end_hour_et", None)
        if ("entry_start_hour" in merged) ^ ("entry_end_hour" in merged):
            merged.pop("entry_start_hour", None)
            merged.pop("entry_end_hour", None)

        # Volume gate requires both knobs.
        if merged.get("volume_ratio_min") is None:
            merged.pop("volume_ema_period", None)

        # Riskpanic overlay requires both knobs.
        if ("riskpanic_tr5_med_pct" in merged) ^ ("riskpanic_neg_gap_ratio_min" in merged):
            merged.pop("riskpanic_tr5_med_pct", None)
            merged.pop("riskpanic_neg_gap_ratio_min", None)

        # Riskpop overlay requires both knobs.
        if ("riskpop_tr5_med_pct" in merged) ^ ("riskpop_pos_gap_ratio_min" in merged):
            merged.pop("riskpop_tr5_med_pct", None)
            merged.pop("riskpop_pos_gap_ratio_min", None)

        f = _parse_filters(merged)
        return f if _filters_payload(f) is not None else None

    def _ranked_keys_by_row_scores(best_by_key: dict, *, top_pnl: int = 8, top_pnl_dd: int = 8) -> list:
        ranked = _rank_cfg_rows(
            [
                (key, rec["row"], "")
                for key, rec in best_by_key.items()
                if isinstance(rec, dict) and isinstance(rec.get("row"), dict)
            ],
            scorers=[(_score_row_pnl, int(top_pnl)), (_score_row_pnl_dd, int(top_pnl_dd))],
            key_fn=lambda key, _row, _note: key,
        )
        return [key for key, _row, _note in ranked]

    def _bars(bar_size: str) -> list:
        if offline:
            return data.load_cached_bars(
                symbol=symbol,
                exchange=None,
                start=start_dt,
                end=end_dt,
                bar_size=str(bar_size),
                use_rth=use_rth,
                cache_dir=cache_dir,
            )
        return data.load_or_fetch_bars(
            symbol=symbol,
            exchange=None,
            start=start_dt,
            end=end_dt,
            bar_size=str(bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )

    bar_cache: dict[str, list] = {}

    def _bars_cached(bar_size: str) -> list:
        key = str(bar_size)
        cached = bar_cache.get(key)
        if cached is not None:
            return cached
        loaded = _bars(key)
        bar_cache[key] = loaded
        return loaded

    regime_bars_1d = _bars_cached("1 day")
    if not regime_bars_1d:
        raise SystemExit("No 1 day regime bars returned (IBKR).")

    def _regime_bars_for(cfg: ConfigBundle) -> list | None:
        regime_bar = str(getattr(cfg.strategy, "regime_bar_size", "") or "").strip() or str(cfg.backtest.bar_size)
        if str(regime_bar) == str(cfg.backtest.bar_size):
            return None
        bars = _bars_cached(regime_bar)
        if not bars:
            raise SystemExit(f"No {regime_bar} regime bars returned (IBKR).")
        return bars

    def _regime2_bars_for(cfg: ConfigBundle) -> list | None:
        mode = str(getattr(cfg.strategy, "regime2_mode", "off") or "off").strip().lower()
        if mode == "off":
            return None
        regime_bar = str(getattr(cfg.strategy, "regime2_bar_size", "") or "").strip() or str(cfg.backtest.bar_size)
        if str(regime_bar) == str(cfg.backtest.bar_size):
            return None
        bars = _bars_cached(regime_bar)
        if not bars:
            raise SystemExit(f"No {regime_bar} regime2 bars returned (IBKR).")
        return bars

    tick_cache: dict[tuple[str, str], tuple[datetime, list]] = {}

    def _tick_bars_for(cfg: ConfigBundle) -> list | None:
        tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
        if tick_mode == "off":
            return None
        if tick_mode != "raschke":
            return None

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

        warm_days = max(60, int(z_lookback) + int(ma_period) + int(slope_lb) + 5)
        tick_start_dt = start_dt - timedelta(days=int(warm_days))
        # $TICK is defined for RTH only (NYSE hours).
        tick_use_rth = True

        def _load_tick_daily(symbol: str, exchange: str) -> list:
            try:
                if offline:
                    return data.load_cached_bars(
                        symbol=symbol,
                        exchange=exchange,
                        start=tick_start_dt,
                        end=end_dt,
                        bar_size="1 day",
                        use_rth=tick_use_rth,
                        cache_dir=cache_dir,
                    )
                return data.load_or_fetch_bars(
                    symbol=symbol,
                    exchange=exchange,
                    start=tick_start_dt,
                    end=end_dt,
                    bar_size="1 day",
                    use_rth=tick_use_rth,
                    cache_dir=cache_dir,
                )
            except FileNotFoundError:
                return []

        def _from_cache(symbol: str, exchange: str) -> list | None:
            cached = tick_cache.get((symbol, exchange))
            if cached is None:
                return None
            cached_start, cached_bars = cached
            if cached_start <= tick_start_dt:
                return cached_bars
            return None

        cached = _from_cache(tick_symbol, tick_exchange)
        if cached is not None:
            return cached

        tick_bars = _load_tick_daily(tick_symbol, tick_exchange)
        used_symbol = tick_symbol
        used_exchange = tick_exchange
        # Offline friendly fallback: IBKR permissions may block NYSE TICK, but AMEX TICK is often available.
        if not tick_bars and tick_symbol.upper() == "TICK-NYSE":
            fallback_symbol = "TICK-AMEX"
            fallback_exchange = "AMEX"
            cached_fb = _from_cache(fallback_symbol, fallback_exchange)
            if cached_fb is not None:
                tick_bars = cached_fb
                used_symbol = fallback_symbol
                used_exchange = fallback_exchange
            else:
                fb = _load_tick_daily(fallback_symbol, fallback_exchange)
                if fb:
                    tick_bars = fb
                    used_symbol = fallback_symbol
                    used_exchange = fallback_exchange
        if not tick_bars:
            hint = (
                " (cache empty; run once without --offline to populate, requires market data permissions)"
                if offline
                else " (check IBKR market data permissions for NYSE IND)"
            )
            extra = " (try TICK-AMEX/AMEX if available)" if tick_symbol.upper() == "TICK-NYSE" else ""
            raise SystemExit(f"No $TICK bars available for {tick_symbol} ({tick_exchange}){hint}{extra}.")
        tick_cache[(used_symbol, used_exchange)] = (tick_start_dt, tick_bars)
        return tick_bars

    def _context_bars_for_cfg(
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
        regime_bars: list | None = None,
        regime2_bars: list | None = None,
    ) -> tuple[list, list | None, list | None]:
        bars_eff = bars if bars is not None else _bars_cached(str(cfg.backtest.bar_size))
        regime_eff = _regime_bars_for(cfg) if regime_bars is None else regime_bars
        regime2_eff = _regime2_bars_for(cfg) if regime2_bars is None else regime2_bars
        return bars_eff, regime_eff, regime2_eff

    def _bars_signature(series: list | None) -> tuple[int, object | None, object | None]:
        if not series:
            return (0, None, None)
        first = series[0]
        last = series[-1]
        return (
            len(series),
            getattr(first, "ts", None),
            getattr(last, "ts", None),
        )

    axis_progress_state: dict[str, object] = {
        "active": False,
        "axis_key": "",
        "label": "",
        "tested": 0,
        "kept": 0,
        "total": None,
        "started_at": 0.0,
        "last_report": 0.0,
        "last_reported_tested": 0,
        "report_every": 0,
        "heartbeat_sec": 20.0,
        "suppress": False,
    }
    axis_progress_history_path = cache_dir / "spot_axis_total_hints.json"
    axis_progress_history: dict[str, int] = {}
    try:
        raw_hist = json.loads(axis_progress_history_path.read_text())
        if isinstance(raw_hist, dict):
            for key, val in raw_hist.items():
                try:
                    iv = int(val)
                except (TypeError, ValueError):
                    continue
                if iv > 0:
                    axis_progress_history[str(key).strip().lower()] = int(iv)
    except Exception:
        axis_progress_history = {}

    def _axis_total_hint(axis_name: str) -> int | None:
        axis = str(axis_name).strip().lower()
        static_simple: dict[str, int] = {
            "ema": 6,
            "entry_mode": 6,
            "volume": 13,
            "rv": 29,
            "tod": 29,
            "tod_interaction": 81,
            "exit_time": 7,
            "regime2_ema": 12,
            "flip_exit": 48,
            "confirm": 4,
            "hold": 7,
            "spot_short_risk_mult": 13,
            "slope": 6,
            "cooldown": 7,
            "skip_open": 6,
        }
        static_val = static_simple.get(axis)
        if isinstance(static_val, int) and static_val > 0:
            return int(static_val)
        if axis in _ATR_EXIT_PROFILE_REGISTRY:
            profile = _ATR_EXIT_PROFILE_REGISTRY.get(axis) or {}
            return int(
                len(tuple(profile.get("atr_periods") or ()))
                * len(tuple(profile.get("pt_mults") or ()))
                * len(tuple(profile.get("sl_mults") or ()))
            )
        if axis in _SPREAD_PROFILE_REGISTRY:
            profile = _SPREAD_PROFILE_REGISTRY.get(axis) or {}
            return int(len(tuple(profile.get("values") or ())))
        if axis == "perm_joint":
            profile = _PERM_JOINT_PROFILE
            return int(
                len(tuple(profile.get("tod_windows") or ()))
                * len(tuple(profile.get("spread_variants") or ()))
                * len(tuple(profile.get("vol_variants") or ()))
            )
        if axis == "regime":
            profile = _REGIME_ST_PROFILE
            return int(
                len(tuple(profile.get("bars") or ()))
                * len(tuple(profile.get("atr_periods") or ()))
                * len(tuple(profile.get("multipliers") or ()))
                * len(tuple(profile.get("sources") or ()))
            )
        if axis == "regime2":
            profile = _REGIME2_ST_PROFILE
            return int(
                len(tuple(profile.get("atr_periods") or ()))
                * len(tuple(profile.get("multipliers") or ()))
                * len(tuple(profile.get("sources") or ()))
            )
        if axis == "shock":
            profile = _SHOCK_SWEEP_PROFILE
            preset_count = int(
                len(tuple(profile.get("ratio_rows") or ()))
                + len(tuple(profile.get("daily_atr_rows") or ()))
                + len(tuple(profile.get("drawdown_rows") or ()))
            )
            return int(
                preset_count
                * len(tuple(profile.get("modes") or ()))
                * len(tuple(profile.get("dir_variants") or ()))
                * len(tuple(profile.get("sl_mults") or ()))
                * len(tuple(profile.get("pt_mults") or ()))
                * len(tuple(profile.get("short_risk_factors") or ()))
            )
        hist = axis_progress_history.get(str(axis))
        if isinstance(hist, int) and hist > 0:
            return int(hist)
        return None

    def _axis_progress_begin(*, axis_name: str) -> None:
        axis_key = str(axis_name).strip().lower()
        axis_progress_state["active"] = True
        axis_progress_state["axis_key"] = str(axis_key)
        axis_progress_state["label"] = f"{axis_key} axis"
        axis_progress_state["tested"] = 0
        axis_progress_state["kept"] = 0
        axis_progress_state["total"] = _axis_total_hint(str(axis_key))
        started_at = float(pytime.perf_counter())
        axis_progress_state["started_at"] = started_at
        axis_progress_state["last_report"] = started_at
        axis_progress_state["last_reported_tested"] = 0
        axis_progress_state["report_every"] = 200
        axis_progress_state["heartbeat_sec"] = 20.0
        axis_progress_state["suppress"] = False

    def _axis_progress_record(*, kept: bool) -> None:
        if not bool(axis_progress_state.get("active")):
            return
        axis_progress_state["tested"] = int(axis_progress_state.get("tested") or 0) + 1
        if bool(kept):
            axis_progress_state["kept"] = int(axis_progress_state.get("kept") or 0) + 1
        if bool(axis_progress_state.get("suppress")):
            return
        tested = int(axis_progress_state.get("tested") or 0)
        report_every = int(axis_progress_state.get("report_every") or 0)
        started_at = float(axis_progress_state.get("started_at") or 0.0)
        total = axis_progress_state.get("total")
        hit_report_every = report_every > 0 and (tested % report_every == 0)
        hit_total = isinstance(total, int) and int(total) > 0 and tested >= int(total)
        now = float(pytime.perf_counter())
        hit_heartbeat = (now - float(axis_progress_state.get("last_report") or started_at)) >= float(
            axis_progress_state.get("heartbeat_sec") or 20.0
        )
        if not (hit_report_every or hit_total or hit_heartbeat):
            return
        print(
            _progress_line(
                label=str(axis_progress_state.get("label") or "axis"),
                tested=int(tested),
                total=(int(total) if isinstance(total, int) else None),
                kept=int(axis_progress_state.get("kept") or 0),
                started_at=float(started_at),
                rate_unit="cfg/s",
            ),
            flush=True,
        )
        axis_progress_state["last_report"] = now
        axis_progress_state["last_reported_tested"] = int(tested)

    def _axis_progress_end() -> None:
        if not bool(axis_progress_state.get("active")):
            return
        tested = int(axis_progress_state.get("tested") or 0)
        axis_key = str(axis_progress_state.get("axis_key") or "").strip().lower()
        last_reported_tested = int(axis_progress_state.get("last_reported_tested") or 0)
        if tested > 0 and tested != last_reported_tested:
            print(
                _progress_line(
                    label=str(axis_progress_state.get("label") or "axis"),
                    tested=tested,
                    total=(
                        int(axis_progress_state.get("total"))
                        if isinstance(axis_progress_state.get("total"), int)
                        else None
                    ),
                    kept=int(axis_progress_state.get("kept") or 0),
                    started_at=float(axis_progress_state.get("started_at") or 0.0),
                    rate_unit="cfg/s",
                ),
                flush=True,
            )
            if axis_key:
                axis_progress_history[str(axis_key)] = int(tested)
                try:
                    write_json(axis_progress_history_path, axis_progress_history, sort_keys=True)
                except Exception:
                    pass
        axis_progress_state["active"] = False
        axis_progress_state["axis_key"] = ""
        axis_progress_state["label"] = ""
        axis_progress_state["tested"] = 0
        axis_progress_state["kept"] = 0
        axis_progress_state["total"] = None
        axis_progress_state["started_at"] = 0.0
        axis_progress_state["last_report"] = 0.0
        axis_progress_state["last_reported_tested"] = 0
        axis_progress_state["suppress"] = False

    def _run_cfg(
        *, cfg: ConfigBundle, bars: list | None = None, regime_bars: list | None = None, regime2_bars: list | None = None
    ) -> dict | None:
        nonlocal run_calls_total, run_cfg_cache_hits, run_cfg_fingerprint_hits, run_cfg_persistent_hits, run_cfg_persistent_writes
        run_calls_total += 1
        bars_eff, regime_eff, regime2_eff = _context_bars_for_cfg(
            cfg=cfg,
            bars=bars,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
        )
        cfg_key = _milestone_key(cfg)
        bars_sig = _bars_signature(bars_eff)
        regime_sig = _bars_signature(regime_eff)
        regime2_sig = _bars_signature(regime2_eff)
        ctx_sig = (bars_sig, regime_sig, regime2_sig)
        fp_cached = run_cfg_fingerprint_cache.get(cfg_key)
        if fp_cached is not None and fp_cached[0] == ctx_sig:
            run_cfg_cache_hits += 1
            run_cfg_fingerprint_hits += 1
            fp_row = fp_cached[1]
            row = dict(fp_row) if isinstance(fp_row, dict) else None
            _axis_progress_record(kept=bool(row))
            return row
        cache_key = (
            cfg_key,
            bars_sig,
            regime_sig,
            regime2_sig,
        )
        persistent_key = _run_cfg_persistent_key(cfg_key=cfg_key, ctx_sig=ctx_sig)
        cached = run_cfg_cache.get(cache_key, _RUN_CFG_CACHE_MISS)
        if cached is not _RUN_CFG_CACHE_MISS:
            run_cfg_cache_hits += 1
            run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, cached if isinstance(cached, dict) else None)
            row = dict(cached) if isinstance(cached, dict) else None
            _axis_progress_record(kept=bool(row))
            return row
        persisted = _run_cfg_persistent_get(cache_key=str(persistent_key))
        if persisted is not _RUN_CFG_CACHE_MISS:
            run_cfg_cache_hits += 1
            run_cfg_persistent_hits += 1
            run_cfg_cache[cache_key] = persisted if isinstance(persisted, dict) else None
            run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, persisted if isinstance(persisted, dict) else None)
            row = dict(persisted) if isinstance(persisted, dict) else None
            _axis_progress_record(kept=bool(row))
            return row
        tick_bars = _tick_bars_for(cfg)
        exec_bars = None
        exec_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
        if exec_size and str(exec_size) != str(cfg.backtest.bar_size):
            exec_bars = _bars_cached(exec_size)
        s = _run_spot_backtest_summary(
            cfg,
            bars_eff,
            meta,
            regime_bars=regime_eff,
            regime2_bars=regime2_eff,
            tick_bars=tick_bars,
            exec_bars=exec_bars,
        )
        if int(s.trades) < int(run_min_trades):
            run_cfg_cache[cache_key] = None
            run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, None)
            _run_cfg_persistent_set(cache_key=str(persistent_key), payload=None)
            run_cfg_persistent_writes += 1
            _axis_progress_record(kept=False)
            return None
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        roi = float(getattr(s, "roi", 0.0) or 0.0)
        dd_pct = float(getattr(s, "max_drawdown_pct", 0.0) or 0.0)
        out = {
            "trades": int(s.trades),
            "win_rate": float(s.win_rate),
            "pnl": pnl,
            "dd": dd,
            "roi": roi,
            "dd_pct": dd_pct,
            "pnl_over_dd": (pnl / dd) if dd > 0 else None,
        }
        run_cfg_cache[cache_key] = out
        run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, out)
        _run_cfg_persistent_set(cache_key=str(persistent_key), payload=out)
        run_cfg_persistent_writes += 1
        _axis_progress_record(kept=True)
        return dict(out)

    def _run_sweep(
        *,
        plan,
        bars: list,
        total: int | None = None,
        progress_label: str | None = None,
        report_every: int = 0,
        heartbeat_sec: float = 0.0,
        record_milestones: bool = True,
    ) -> tuple[int, list[tuple[ConfigBundle, dict, str, dict | None]]]:
        tested = 0
        kept: list[tuple[ConfigBundle, dict, str, dict | None]] = []
        t0 = pytime.perf_counter()
        last = float(t0)
        total_i = int(total) if total is not None else None
        suppress_prev = bool(axis_progress_state.get("suppress"))
        if progress_label:
            axis_progress_state["suppress"] = True

        try:
            for cfg, note, meta_item in plan:
                tested += 1
                if progress_label:
                    now = pytime.perf_counter()
                    hit_report_every = int(report_every) > 0 and (tested % int(report_every) == 0)
                    hit_total = total_i is not None and tested == int(total_i)
                    hit_heartbeat = float(heartbeat_sec) > 0 and (now - last) >= float(heartbeat_sec)
                    if hit_report_every or hit_total or hit_heartbeat:
                        print(
                            _progress_line(
                                label=str(progress_label),
                                tested=int(tested),
                                total=total_i,
                                kept=len(kept),
                                started_at=t0,
                                rate_unit="s",
                            ),
                            flush=True,
                        )
                        last = float(now)

                row = _run_cfg(cfg=cfg, bars=bars)
                if not row:
                    continue

                note_s = str(note or "")
                row = dict(row)
                if note_s:
                    row["note"] = note_s
                    if bool(record_milestones):
                        _record_milestone(cfg, row, note_s)
                kept.append((cfg, row, note_s, meta_item))
        finally:
            axis_progress_state["suppress"] = suppress_prev

        return tested, kept

    def _cfg_from_strategy_filters_payload(strategy_payload, filters_payload) -> ConfigBundle | None:
        if not isinstance(strategy_payload, dict):
            return None
        try:
            filters_obj = _filters_from_payload(filters_payload if isinstance(filters_payload, dict) else None)
            strategy_obj = _strategy_from_payload(strategy_payload, filters=filters_obj)
        except Exception:
            return None
        return _mk_bundle(
            strategy=strategy_obj,
            start=start,
            end=end,
            bar_size=signal_bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
            offline=offline,
        )

    def _rows_from_kept(kept: list[tuple[ConfigBundle, dict, str, dict | None]]) -> list[tuple[ConfigBundle, dict, str]]:
        out: list[tuple[ConfigBundle, dict, str]] = []
        for cfg, row, note, _meta in kept:
            out.append((cfg, row, note))
        return out

    def _encode_cfg_payload(
        cfg: ConfigBundle,
        *,
        note: str | None = None,
        note_key: str = "note",
        extra: dict | None = None,
    ) -> dict:
        payload = {
            "strategy": _spot_strategy_payload(cfg, meta=meta),
            "filters": _filters_payload(cfg.strategy.filters),
        }
        if note is not None:
            payload[str(note_key)] = str(note)
        if isinstance(extra, dict):
            payload.update(extra)
        return payload

    def _decode_cfg_payload(
        payload: object,
        *,
        note_key: str = "note",
        default_note: str = "",
    ) -> tuple[ConfigBundle, str] | None:
        if not isinstance(payload, dict):
            return None
        cfg = _cfg_from_strategy_filters_payload(payload.get("strategy"), payload.get("filters"))
        if cfg is None:
            return None
        note = str(payload.get(note_key) or "").strip() or str(default_note)
        return cfg, note

    def _load_worker_stage_payload(
        *,
        schema_name: str,
        payload_path: Path,
    ) -> dict[str, object]:
        try:
            payload = json.loads(payload_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid {schema_name} payload JSON: {payload_path}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid {schema_name} payload: expected object ({payload_path})")

        def _require_list(key: str) -> list:
            raw = payload.get(key)
            if not isinstance(raw, list):
                raise SystemExit(f"{schema_name} payload missing '{key}' list: {payload_path}")
            return raw

        def _decode_cfg_list(*, key: str, note_key: str, seed_prefix: bool = False) -> list[tuple[ConfigBundle, str]]:
            out: list[tuple[ConfigBundle, str]] = []
            for item in _require_list(key):
                decoded = _decode_cfg_payload(item, note_key=note_key)
                if decoded is None:
                    continue
                cfg_obj, note = decoded
                if bool(seed_prefix):
                    seed_tag = str(item.get("seed_tag") or "seed") if isinstance(item, dict) else "seed"
                    note = f"{seed_tag} | {note}"
                out.append((cfg_obj, str(note)))
            return out

        def _decode_overrides_list(*, key: str) -> list[tuple[dict[str, object], str]]:
            out: list[tuple[dict[str, object], str]] = []
            for item in _require_list(key):
                if not isinstance(item, dict):
                    continue
                over = item.get("overrides")
                if not isinstance(over, dict):
                    continue
                out.append((over, str(item.get("note") or "")))
            if not out:
                raise SystemExit(f"{schema_name} payload '{key}' empty/invalid: {payload_path}")
            return out

        name = str(schema_name).strip().lower()
        if name == "champ_refine_stage3a":
            return {
                "seed_tag": str(payload.get("seed_tag") or "seed"),
                "cfg_pairs": _decode_cfg_list(key="base_exit_variants", note_key="exit_note"),
            }
        if name == "champ_refine_stage3b":
            return {
                "seed_tag": str(payload.get("seed_tag") or "seed"),
                "cfg_pairs": _decode_cfg_list(key="shortlist", note_key="base_note"),
            }
        if name == "st37_refine_stage1":
            seeds_local: list[dict] = []
            for item in _require_list("seeds"):
                if not isinstance(item, dict):
                    continue
                strategy = item.get("strategy")
                if not isinstance(strategy, dict):
                    continue
                filters_payload = item.get("filters")
                if filters_payload is not None and not isinstance(filters_payload, dict):
                    filters_payload = None
                seeds_local.append(
                    {
                        "strategy": strategy,
                        "filters": filters_payload,
                        "group_name": str(item.get("group_name") or ""),
                    }
                )
            return {"seeds": seeds_local}
        if name == "st37_refine_stage2":
            return {
                "cfg_pairs": _decode_cfg_list(key="shortlist", note_key="base_note", seed_prefix=True),
                "risk_variants": _decode_overrides_list(key="risk_variants"),
                "shock_variants": _decode_overrides_list(key="shock_variants"),
            }
        if name == "combo_fast_stage2":
            return {
                "cfg_pairs": _decode_cfg_list(key="shortlist", note_key="base_note"),
            }
        if name == "combo_fast_stage3":
            return {
                "cfg_pairs": _decode_cfg_list(key="bases", note_key="base_note"),
            }
        if name == "seeded_micro_stage":
            return {
                "axis_tag": str(payload.get("axis_tag") or ""),
                "cfg_pairs": _decode_cfg_list(key="cfgs", note_key="note"),
            }
        if name == "gate_matrix_stage2":
            seeds_local: list[tuple[ConfigBundle, str, str]] = []
            for item in _require_list("seeds"):
                if not isinstance(item, dict):
                    continue
                cfg_seed = _cfg_from_strategy_filters_payload(item.get("strategy"), item.get("filters"))
                if cfg_seed is None:
                    continue
                seeds_local.append(
                    (
                        cfg_seed,
                        str(item.get("seed_note") or ""),
                        str(item.get("family") or ""),
                    )
                )
            return {"seed_triples": seeds_local}
        raise SystemExit(f"Unknown worker payload schema: {schema_name!r}")

    def _worker_records_from_kept(kept: list[tuple[ConfigBundle, dict, str, dict | None]]) -> list[dict]:
        records: list[dict] = []
        for cfg, row, note, _meta in kept:
            records.append(_encode_cfg_payload(cfg, note=note, extra={"row": row}))
        return records

    def _run_sharded_stage_worker(
        *,
        stage_label: str,
        worker_raw,
        workers_raw,
        out_path_raw: str,
        out_flag_name: str,
        plan_all,
        bars: list,
        report_every: int,
        heartbeat_sec: float = 0.0,
    ) -> None:
        if not offline:
            raise SystemExit(f"{stage_label} worker mode requires --offline (avoid parallel IBKR sessions).")
        out_path_str = str(out_path_raw or "").strip()
        if not out_path_str:
            raise SystemExit(f"--{out_flag_name} is required for {stage_label} worker mode.")
        out_path = Path(out_path_str)

        worker_id, workers = _parse_worker_shard(worker_raw, workers_raw, label=str(stage_label))
        total = len(plan_all)
        local_total = (total // workers) + (1 if worker_id < (total % workers) else 0)
        shard_plan = (item for combo_idx, item in enumerate(plan_all) if (combo_idx % int(workers)) == int(worker_id))
        tested, kept = _run_sweep(
            plan=shard_plan,
            bars=bars,
            total=local_total,
            progress_label=f"{stage_label} worker {worker_id+1}/{workers}",
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=False,
        )
        records = _worker_records_from_kept(kept)
        out_payload = {"tested": int(tested), "kept": len(records), "records": records}
        write_json(out_path, out_payload, sort_keys=False)
        print(f"{stage_label} worker done tested={tested} kept={len(records)} out={out_path}", flush=True)

    def _decode_payload_schema_cfg_row(
        rec: dict,
        *,
        default_note: str,
        context: dict | None = None,
    ) -> tuple[ConfigBundle, dict, str] | None:
        decoded = _decode_cfg_payload(rec, note_key="note", default_note=str(default_note))
        if decoded is None:
            return None
        cfg, note = decoded
        row = rec.get("row")
        if not isinstance(row, dict):
            return None
        row_out = dict(row)
        row_out["note"] = note
        return cfg, row_out, str(note)

    def _collect_stage_cfg_payload_rows(
        *,
        payloads: dict[int, dict],
        default_note: str,
        on_item=None,
    ) -> int:
        def _decode_record(rec: dict):
            return _decode_payload_schema_cfg_row(rec, default_note=str(default_note), context=None)

        return _collect_parallel_payload_records(
            payloads=payloads,
            records_key="records",
            tested_key="tested",
            decode_record=_decode_record,
            on_record=on_item,
        )

    def _collect_stage_rows_from_payloads(
        *,
        payloads: dict[int, dict],
        default_note: str,
        on_row,
        dedupe_by_milestone_key: bool = False,
    ) -> int:
        seen_keys: set[str] | None = set() if bool(dedupe_by_milestone_key) else None

        def _on_item(item: tuple[ConfigBundle, dict, str] | None) -> None:
            if item is None:
                return
            cfg, row, note = item
            if seen_keys is not None:
                cfg_key = _milestone_key(cfg)
                if cfg_key in seen_keys:
                    return
                seen_keys.add(cfg_key)
            on_row(cfg, row, note)

        return _collect_stage_cfg_payload_rows(
            payloads=payloads,
            default_note=str(default_note),
            on_item=_on_item,
        )

    def _run_stage_serial(
        *,
        stage_label: str,
        plan,
        bars: list,
        total: int,
        report_every: int,
        heartbeat_sec: float = 0.0,
        record_milestones: bool = True,
    ) -> tuple[int, list[tuple[ConfigBundle, dict, str]]]:
        tested, kept = _run_sweep(
            plan=plan,
            bars=bars,
            total=total,
            progress_label=str(stage_label),
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=bool(record_milestones),
        )
        return int(tested), _rows_from_kept(kept)

    def _run_stage_cfg_rows(
        *,
        stage_label: str,
        total: int,
        jobs_req: int,
        bars: list,
        report_every: int,
        on_row,
        serial_plan=None,
        serial_plan_builder=None,
        heartbeat_sec: float = 0.0,
        parallel_payloads_builder=None,
        parallel_default_note: str = "",
        parallel_dedupe_by_milestone_key: bool = True,
        record_milestones: bool = True,
    ) -> int:
        def _on_row_local(cfg: ConfigBundle, row: dict, note: str) -> None:
            if bool(record_milestones):
                _record_milestone(cfg, row, note)
            on_row(cfg, row, note)

        if int(jobs_req) > 1 and int(total) > 0 and callable(parallel_payloads_builder):
            payloads = parallel_payloads_builder()
            return _collect_stage_rows_from_payloads(
                payloads=payloads,
                default_note=str(parallel_default_note or stage_label),
                on_row=_on_row_local,
                dedupe_by_milestone_key=bool(parallel_dedupe_by_milestone_key),
            )
        serial_plan_eff = serial_plan_builder() if callable(serial_plan_builder) else serial_plan
        tested, serial_rows = _run_stage_serial(
            stage_label=str(stage_label),
            plan=serial_plan_eff,
            bars=bars,
            total=int(total),
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=bool(record_milestones),
        )
        for cfg, row, note in serial_rows:
            on_row(cfg, row, note)
        return int(tested)

    def _stage_parallel_base_cli(*, flags_with_values: tuple[str, ...]) -> list[str]:
        return _strip_flags(
            list(sys.argv[1:]),
            flags=("--write-milestones", "--merge-milestones"),
            flags_with_values=("--axis", "--jobs", "--milestones-out", *flags_with_values),
        )

    def _run_parallel_stage_with_payload(
        *,
        axis_name: str,
        stage_label: str,
        total: int,
        jobs: int,
        payload: dict,
        payload_filename: str,
        temp_prefix: str,
        worker_tmp_prefix: str,
        worker_tag: str,
        out_prefix: str,
        stage_flag: str,
        worker_flag: str,
        workers_flag: str,
        out_flag: str,
        strip_flags_with_values: tuple[str, ...],
        run_min_trades_flag: str | None,
        run_min_trades: int | None,
        capture_error: str,
        failure_label: str,
        missing_label: str,
        invalid_label: str,
    ) -> dict[int, dict]:
        base_cli = _stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmpdir:
            payload_path = Path(tmpdir) / str(payload_filename)
            write_json(payload_path, payload, sort_keys=False)
            _jobs_eff, payloads = _run_parallel_stage_kernel(
                stage_label=str(stage_label),
                jobs=int(jobs),
                total=int(total),
                default_jobs=int(_default_jobs()),
                offline=bool(offline),
                offline_error=f"--jobs>1 for {axis_name} requires --offline (avoid parallel IBKR sessions).",
                tmp_prefix=str(worker_tmp_prefix),
                worker_tag=str(worker_tag),
                out_prefix=str(out_prefix),
                build_cmd=lambda worker_id, workers_n, out_path: [
                    sys.executable,
                    "-u",
                    "-m",
                    "tradebot.backtest",
                    "spot",
                    *base_cli,
                    "--axis",
                    str(axis_name),
                    "--jobs",
                    "1",
                    str(stage_flag),
                    str(payload_path),
                    str(worker_flag),
                    str(worker_id),
                    str(workers_flag),
                    str(workers_n),
                    str(out_flag),
                    str(out_path),
                    *(
                        [str(run_min_trades_flag), str(int(run_min_trades))]
                        if (run_min_trades_flag and run_min_trades is not None)
                        else []
                    ),
                ],
                capture_error=str(capture_error),
                failure_label=str(failure_label),
                missing_label=str(missing_label),
                invalid_label=str(invalid_label),
            )
            return payloads

    def _run_parallel_stage(
        *,
        axis_name: str | None,
        stage_label: str,
        total: int,
        jobs: int,
        worker_tmp_prefix: str,
        worker_tag: str,
        out_prefix: str,
        worker_flag: str,
        workers_flag: str,
        out_flag: str,
        strip_flags_with_values: tuple[str, ...],
        capture_error: str,
        failure_label: str,
        missing_label: str,
        invalid_label: str,
        run_min_trades_flag: str | None = None,
        run_min_trades: int | None = None,
        stage_flag: str | None = None,
        stage_value: str | None = None,
        stage_args: tuple[str, ...] = (),
        entrypoint: tuple[str, ...] = ("-m", "tradebot.backtest", "spot"),
    ) -> dict[int, dict]:
        base_cli = _stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        _jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label=str(stage_label),
            jobs=int(jobs),
            total=int(total),
            default_jobs=int(_default_jobs()),
            offline=bool(offline),
            offline_error=f"--jobs>1 for {axis_name or stage_label} requires --offline (avoid parallel IBKR sessions).",
            tmp_prefix=str(worker_tmp_prefix),
            worker_tag=str(worker_tag),
            out_prefix=str(out_prefix),
            build_cmd=lambda worker_id, workers_n, out_path: [
                sys.executable,
                "-u",
                *[str(p) for p in entrypoint],
                *base_cli,
                *(["--axis", str(axis_name)] if axis_name else []),
                "--jobs",
                "1",
                *(
                    [str(stage_flag), str(stage_value)]
                    if (stage_flag and stage_value is not None)
                    else ([str(stage_flag)] if stage_flag else [])
                ),
                *[str(arg) for arg in stage_args],
                str(worker_flag),
                str(worker_id),
                str(workers_flag),
                str(workers_n),
                str(out_flag),
                str(out_path),
                *(
                    [str(run_min_trades_flag), str(int(run_min_trades))]
                    if (run_min_trades_flag and run_min_trades is not None)
                    else []
                ),
            ],
            capture_error=str(capture_error),
            failure_label=str(failure_label),
            missing_label=str(missing_label),
            invalid_label=str(invalid_label),
        )
        return payloads

    def _collect_axis_milestone_items(
        *,
        milestone_payloads: dict[str, dict],
        milestone_axes: tuple[str, ...],
    ) -> list[dict]:
        eligible_new: list[dict] = []
        for axis_name in milestone_axes:
            payload = milestone_payloads.get(axis_name)
            if isinstance(payload, dict):
                eligible_new.extend(_collect_milestone_items_from_payload(payload, symbol=symbol))
        return eligible_new

    def _merge_axis_parallel_milestones(
        *,
        milestone_payloads: dict[str, dict],
        milestone_axes: tuple[str, ...],
    ) -> int | None:
        if not bool(args.write_milestones):
            return None
        eligible_new = _collect_axis_milestone_items(
            milestone_payloads=milestone_payloads,
            milestone_axes=milestone_axes,
        )
        out_path = Path(args.milestones_out)
        total = _merge_and_write_milestones(
            out_path=out_path,
            eligible_new=eligible_new,
            merge_existing=bool(args.merge_milestones),
            add_top_pnl_dd=int(args.milestone_add_top_pnl_dd or 0),
            add_top_pnl=int(args.milestone_add_top_pnl or 0),
            symbol=symbol,
            start=start,
            end=end,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            milestone_min_win=float(args.milestone_min_win),
            milestone_min_trades=int(args.milestone_min_trades),
            milestone_min_pnl_dd=float(args.milestone_min_pnl_dd),
        )
        print(f"Wrote {out_path} ({total} eligible presets).", flush=True)
        return int(total)

    def _run_parallel_axis_stage(
        *,
        label: str,
        axes: tuple[str, ...],
        jobs_req: int,
        axis_jobs_resolver,
        tmp_prefix: str,
        offline_error: str,
    ) -> dict[str, dict]:
        if not offline:
            raise SystemExit(str(offline_error))
        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags=("--merge-milestones",),
            flags_with_values=("--axis", "--jobs", "--milestones-out"),
        )
        return _run_axis_subprocess_plan(
            label=str(label),
            axes=axes,
            jobs=int(jobs_req),
            base_cli=base_cli,
            axis_jobs_resolver=axis_jobs_resolver,
            write_milestones=bool(args.write_milestones),
            tmp_prefix=str(tmp_prefix),
        )

    def _axis_plan_parts(axis_plan: list[tuple[str, str, bool]]) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
        axes = tuple(axis_name for axis_name, _profile, _emit in axis_plan)
        axis_profile_by_name = {axis_name: str(profile) for axis_name, profile, _emit in axis_plan}
        milestone_axes = tuple(axis_name for axis_name, _profile, emit in axis_plan if bool(emit))
        return axes, axis_profile_by_name, milestone_axes

    def _run_axis_plan_parallel_if_requested(
        *,
        axis_plan: list[tuple[str, str, bool]],
        jobs_req: int,
        label: str,
        tmp_prefix: str,
        offline_error: str,
    ) -> bool:
        nonlocal milestones_written
        if int(jobs_req) <= 1:
            return False
        axes, axis_profile_by_name, milestone_axes = _axis_plan_parts(axis_plan)
        milestone_payloads = _run_parallel_axis_stage(
            label=str(label),
            axes=axes,
            jobs_req=int(jobs_req),
            axis_jobs_resolver=lambda axis_name: min(int(jobs_req), int(_default_jobs()))
            if axis_profile_by_name.get(str(axis_name), "single") == "scaled"
            else 1,
            tmp_prefix=str(tmp_prefix),
            offline_error=str(offline_error),
        )
        if bool(args.write_milestones):
            _merge_axis_parallel_milestones(
                milestone_payloads=milestone_payloads,
                milestone_axes=milestone_axes,
            )
            milestones_written = True
        return True

    def _run_axis_plan_serial(
        axis_plan: list[tuple[str, str, bool]],
        *,
        timed: bool = False,
    ) -> None:
        def _run_axis_callable(axis_name: str, fn, *, timed_local: bool) -> None:
            before_calls = int(run_calls_total)
            t0 = pytime.perf_counter()
            total_hint = _axis_total_hint(str(axis_name))
            total_hint_s = str(total_hint) if total_hint is not None else "?"
            if bool(timed_local):
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            else:
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            _axis_progress_begin(axis_name=str(axis_name))
            try:
                fn()
            finally:
                _axis_progress_end()
            elapsed = pytime.perf_counter() - t0
            tested = int(run_calls_total) - int(before_calls)
            print(f"DONE  {axis_name} tested={tested} elapsed={elapsed:0.1f}s", flush=True)
            print("", flush=True)

        for axis_name, _profile, _emit in axis_plan:
            fn_obj = axis_registry.get(str(axis_name))
            fn = fn_obj if callable(fn_obj) else None
            if fn is None:
                continue
            _run_axis_callable(str(axis_name), fn, timed_local=bool(timed))

    def _iter_seed_bundles(seeds: list[dict]):
        for seed_i, item in enumerate(seeds, start=1):
            try:
                filters_obj = _filters_from_payload(item.get("filters"))
                strategy_obj = _strategy_from_payload(item.get("strategy") or {}, filters=filters_obj)
            except Exception:
                continue
            cfg_seed = _mk_bundle(
                strategy=strategy_obj,
                start=start,
                end=end,
                bar_size=signal_bar_size,
                use_rth=use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )
            yield seed_i, item, cfg_seed, str(item.get("group_name") or f"seed#{seed_i:02d}")

    def _iter_seed_micro_plan(
        *,
        seeds: list[dict],
        include_base_note: str | None = None,
        build_variants=None,
    ):
        for _seed_i, item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            if include_base_note:
                yield cfg_seed, f"{seed_note} | {include_base_note}", None
            if build_variants is None:
                continue
            yield from build_variants(cfg_seed, seed_note, item)

    def _run_cfg_pairs_grid(
        *,
        axis_tag: str,
        cfg_pairs: list[tuple[ConfigBundle, str]],
        rows: list[dict],
        report_every: int = 0,
        heartbeat_sec: float = 0.0,
    ) -> int:
        nonlocal run_calls_total
        if args.seeded_micro_stage:
            payload_path = Path(str(args.seeded_micro_stage))
            payload_decoded = _load_worker_stage_payload(
                schema_name="seeded_micro_stage",
                payload_path=payload_path,
            )
            payload_axis = str(payload_decoded.get("axis_tag") or "").strip().lower()
            if payload_axis and payload_axis != str(axis_tag).strip().lower():
                raise SystemExit(
                    f"cfg_pairs worker payload axis mismatch: expected {axis_tag} got {payload_axis} ({payload_path})"
                )
            plan_all = [
                (cfg, note, None)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]
            _run_sharded_stage_worker(
                stage_label=str(axis_tag),
                worker_raw=args.seeded_micro_worker,
                workers_raw=args.seeded_micro_workers,
                out_path_raw=str(args.seeded_micro_out or ""),
                out_flag_name="seeded-micro-out",
                plan_all=plan_all,
                bars=_bars_cached(signal_bar_size),
                report_every=max(1, int(report_every)),
                heartbeat_sec=float(heartbeat_sec),
            )
            return -1

        plan_all = [(cfg, str(note), None) for cfg, note in cfg_pairs]
        total_eff = len(plan_all)
        tested_total = _run_stage_cfg_rows(
            stage_label=str(axis_tag),
            total=int(total_eff),
            jobs_req=int(jobs),
            bars=_bars_cached(signal_bar_size),
            report_every=max(1, int(report_every)),
            heartbeat_sec=float(heartbeat_sec),
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan=plan_all,
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name=str(axis_tag),
                stage_label=str(axis_tag),
                total=int(total_eff),
                jobs=int(jobs),
                payload={
                    "axis_tag": str(axis_tag),
                    "cfgs": [_encode_cfg_payload(cfg, note=note) for cfg, note, _meta in plan_all],
                },
                payload_filename="cfg_pairs_payload.json",
                temp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_",
                worker_tmp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_worker_",
                worker_tag=f"cfgpairs:{str(axis_tag)}",
                out_prefix="cfg_pairs_out",
                stage_flag="--seeded-micro-stage",
                worker_flag="--seeded-micro-worker",
                workers_flag="--seeded-micro-workers",
                out_flag="--seeded-micro-out",
                strip_flags_with_values=(
                    "--seeded-micro-stage",
                    "--seeded-micro-worker",
                    "--seeded-micro-workers",
                    "--seeded-micro-out",
                    "--seeded-micro-run-min-trades",
                ),
                run_min_trades_flag="--seeded-micro-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error=f"Failed to capture {axis_tag} worker stdout.",
                failure_label=f"{axis_tag} worker",
                missing_label=str(axis_tag),
                invalid_label=str(axis_tag),
            ),
            parallel_default_note=str(axis_tag),
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        return int(tested_total)

    def _run_seeded_micro_grid(
        *,
        axis_tag: str,
        seeds: list[dict],
        rows: list[dict],
        build_variants,
        total: int | None = None,
        report_every: int = 0,
        heartbeat_sec: float = 0.0,
        include_base_note: str | None = None,
    ) -> int:
        plan_all = list(
            _iter_seed_micro_plan(
                seeds=seeds,
                include_base_note=include_base_note,
                build_variants=build_variants,
            )
        )
        total_eff = len(plan_all)
        if total is not None and int(total) != int(total_eff):
            print(
                f"{axis_tag}: normalized total={total_eff} (declared={int(total)})",
                flush=True,
            )
        cfg_pairs = [(cfg, str(note)) for cfg, note, _meta in plan_all]
        tested_total = _run_cfg_pairs_grid(
            axis_tag=str(axis_tag),
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=max(1, int(report_every)),
            heartbeat_sec=float(heartbeat_sec),
        )
        return int(tested_total)

    def _base_bundle(*, bar_size: str, filters: FiltersConfig | None) -> ConfigBundle:
        cfg = _bundle_base(
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
            offline=offline,
            filters=filters,
            spot_close_eod=close_eod,
        )
        if spot_exec_bar_size:
            cfg = replace(cfg, strategy=replace(cfg.strategy, spot_exec_bar_size=spot_exec_bar_size))
        base_name = str(args.base).strip().lower()
        if base_name in ("champion", "champion_pnl"):
            sort_by = "pnl" if base_name == "champion_pnl" else "pnl_dd"
            selected = _milestone_entry_for(
                milestones,
                symbol=symbol,
                signal_bar_size=str(bar_size),
                use_rth=use_rth,
                sort_by=sort_by,
                prefer_realism=realism2,
            )
            if selected is not None:
                base_strategy, base_filters, _ = selected
                cfg = _apply_milestone_base(cfg, strategy=base_strategy, filters=base_filters)
            # Allow sweeps to layer additional filters on top of the milestone baseline
            # (e.g., keep the champion's TOD window and add volume/spread/cooldown filters).
            if filters is not None:
                base_payload = _filters_payload(cfg.strategy.filters) or {}
                over_payload = _filters_payload(filters) or {}
                merged = dict(base_payload)
                merged.update(over_payload)
                merged_filters = _parse_filters(merged)
                if _filters_payload(merged_filters) is None:
                    merged_filters = None
                cfg = replace(cfg, strategy=replace(cfg.strategy, filters=merged_filters))
        elif base_name == "dual_regime":
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    regime2_mode="supertrend",
                    regime2_bar_size="4 hours",
                    regime2_supertrend_atr_period=2,
                    regime2_supertrend_multiplier=0.3,
                    regime2_supertrend_source="close",
                ),
            )

        if long_only:
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    directional_spot={"up": SpotLegConfig(action="BUY", qty=1)},
                ),
            )

        # Realism overrides (backtest only).
        if realism2:
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    spot_entry_fill_mode="next_open",
                    spot_flip_exit_fill_mode="next_open",
                    spot_intrabar_exits=True,
                    spot_spread=float(spot_spread),
                    spot_commission_per_share=float(spot_commission),
                    spot_commission_min=float(spot_commission_min),
                    spot_slippage_per_share=float(spot_slippage),
                    spot_mark_to_market="liquidation",
                    spot_drawdown_mode="intrabar",
                    spot_sizing_mode=str(sizing_mode),
                    spot_notional_pct=float(spot_notional_pct),
                    spot_risk_pct=float(spot_risk_pct),
                    spot_max_notional_pct=float(spot_max_notional_pct),
                    spot_min_qty=int(spot_min_qty),
                    spot_max_qty=int(spot_max_qty),
                ),
            )
        return cfg

    milestone_rows: list[tuple[ConfigBundle, dict, str]] = []
    milestones_written = False

    def _record_milestone(cfg: ConfigBundle, row: dict, note: str) -> None:
        if not bool(args.write_milestones):
            return
        milestone_rows.append((cfg, row, str(note)))

    # Populated once after sweep functions are declared; used by axis dispatchers.
    axis_registry: dict[str, object] = {}

    def _sweep_volume() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        ratios = [None, 1.0, 1.1, 1.2, 1.5]
        periods = [10, 20, 30]
        rows: list[dict] = []
        for ratio in ratios:
            if ratio is None:
                variants = [(None, None)]
            else:
                variants = [(ratio, p) for p in periods]
            for ratio_min, ema_p in variants:
                f = _mk_filters(volume_ratio_min=ratio_min, volume_ema_period=ema_p)
                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                row = _run_cfg(
                    cfg=cfg, bars=bars_sig
                )
                if not row:
                    continue
                note = "-" if ratio_min is None else f"vol>={ratio_min}@{ema_p}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="A) Volume gate sweep", top_n=int(args.top))

    def _sweep_rv() -> None:
        """Orthogonal gate: annualized realized-vol (EWMA) band."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rv_mins = [None, 0.25, 0.3, 0.35, 0.4, 0.45]
        rv_maxs = [None, 0.7, 0.8, 0.9, 1.0]
        rows: list[dict] = []
        for rv_min in rv_mins:
            for rv_max in rv_maxs:
                if rv_min is None and rv_max is None:
                    continue
                f = _mk_filters(rv_min=rv_min, rv_max=rv_max)
                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"rv_min={rv_min} rv_max={rv_max}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="RV gate sweep (annualized EWMA vol)", top_n=int(args.top))

    def _sweep_ema() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21"]
        rows: list[dict] = []
        for preset in presets:
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, ema_preset=str(preset)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"ema={preset}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="0) Timing sweep (EMA preset)", top_n=int(args.top))

    def _sweep_entry_mode() -> None:
        """Timing semantics: cross vs trend entries (+ small confirm grid)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        for mode in ("cross", "trend"):
            for confirm in (0, 1, 2):
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        ema_entry_mode=str(mode),
                        entry_confirm_bars=int(confirm),
                    ),
                )
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"entry_mode={mode} confirm={confirm}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Entry mode sweep (cross vs trend)", top_n=int(args.top))

    def _sweep_tod() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        windows = [
            (None, None, "base"),
            (9, 16, "RTH 9–16 ET"),
            (10, 15, "10–15 ET"),
            (11, 16, "11–16 ET"),
        ]
        # Overnight micro-grid (wraps midnight in ET): this has been a high-leverage permission layer
        # post-lookahead-fix, and is cheap to explore.
        for start_h in (16, 17, 18, 19, 20):
            for end_h in (2, 3, 4, 5, 6):
                windows.append((start_h, end_h, f"{start_h:02d}–{end_h:02d} ET"))
        rows: list[dict] = []
        for start_h, end_h, label in windows:
            f = _mk_filters(entry_start_hour_et=start_h, entry_end_hour_et=end_h)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            row["note"] = label
            _record_milestone(cfg, row, label)
            rows.append(row)
        _print_leaderboards(rows, title="B) Time-of-day gate sweep (ET)", top_n=int(args.top))

    def _sweep_tod_interaction() -> None:
        """Small interaction grid around the proven overnight TOD gate."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        tod_starts = [17, 18, 19]
        tod_ends = [3, 4, 5]
        skip_vals = [0, 1, 2]
        cooldown_vals = [0, 1, 2]
        for start_h in tod_starts:
            for end_h in tod_ends:
                for skip in skip_vals:
                    for cooldown in cooldown_vals:
                        f = _mk_filters(
                            entry_start_hour_et=int(start_h),
                            entry_end_hour_et=int(end_h),
                            skip_first_bars=int(skip),
                            cooldown_bars=int(cooldown),
                        )
                        cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"tod={start_h:02d}-{end_h:02d} ET skip={skip} cd={cooldown}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="TOD interaction sweep (overnight micro-grid)", top_n=int(args.top))

    def _sweep_perm_joint() -> None:
        """Joint permission sweep: TOD × spread × volume (no funnel pruning)."""
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(cfg=base)
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        profile = _PERM_JOINT_PROFILE
        tod_windows: tuple = tuple(profile.get("tod_windows") or ())
        spread_variants: tuple = tuple(profile.get("spread_variants") or ())
        vol_variants: tuple = tuple(profile.get("vol_variants") or ())
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for _, _, tod_note, tod_over in tod_windows:
            for spread_note, spread_over in spread_variants:
                for vol_note, vol_over in vol_variants:
                    overrides: dict[str, object] = {}
                    overrides.update(tod_over)
                    overrides.update(spread_over)
                    overrides.update(vol_over)
                    f = _merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    note = f"{tod_note} | {spread_note} | {vol_note}"
                    cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag="perm_joint",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=200,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Permission joint sweep (TOD × spread × volume)", top_n=int(args.top))

    def _sweep_ema_perm_joint() -> None:
        """Joint sweep: EMA preset × (TOD/spread/volume) permission gates."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Stage 1: evaluate presets with base filters only.
        best_by_ema: dict[str, dict] = {}
        for preset in presets:
            cfg = replace(base, strategy=replace(base.strategy, ema_preset=str(preset), entry_signal="ema"))
            row = _run_cfg(cfg=cfg)
            if not row:
                continue
            best_by_ema[str(preset)] = {"row": row}

        shortlisted = _ranked_keys_by_row_scores(best_by_ema, top_pnl=5, top_pnl_dd=5)
        if not shortlisted:
            print("No eligible EMA presets (try lowering --min-trades).")
            return
        print("")
        print(f"EMA×Perm: stage1 shortlisted ema={len(shortlisted)} (from {len(best_by_ema)})")

        tod_variants = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
            ("tod=18-05 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 5}),
            ("tod=18-06 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 6}),
            ("tod=17-04 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 4}),
            ("tod=19-04 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 4}),
            ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
        ]
        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
            ("spread>=0.0030", {"ema_spread_min_pct": 0.003}),
            ("spread>=0.0040", {"ema_spread_min_pct": 0.004}),
            ("spread>=0.0050", {"ema_spread_min_pct": 0.005}),
            ("spread>=0.0070", {"ema_spread_min_pct": 0.007}),
            ("spread>=0.0100", {"ema_spread_min_pct": 0.01}),
        ]
        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
        ]

        rows: list[dict] = []
        for preset in shortlisted:
            for tod_note, tod_over in tod_variants:
                for spread_note, spread_over in spread_variants:
                    for vol_note, vol_over in vol_variants:
                        overrides: dict[str, object] = {}
                        overrides.update(tod_over)
                        overrides.update(spread_over)
                        overrides.update(vol_over)
                        f = _merge_filters(base_filters, overrides=overrides)
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                ema_preset=str(preset),
                                entry_signal="ema",
                                filters=f,
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"ema={preset} | {tod_note} | {spread_note} | {vol_note}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × permission joint sweep", top_n=int(args.top))

    def _sweep_tick_perm_joint() -> None:
        """Joint sweep: Raschke $TICK gate × (TOD/spread/volume) permission gates."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        # Stage 1: scan tick params using base permission filters (cheap shortlist).
        best_by_tick: dict[tuple, dict] = {}
        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]
        policies = ["allow", "block"]
        dir_policies = ["both", "wide_only"]
        for dir_policy in dir_policies:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                cfg = replace(
                                    base,
                                    strategy=replace(
                                        base.strategy,
                                        tick_gate_mode="raschke",
                                        tick_gate_symbol="TICK-AMEX",
                                        tick_gate_exchange="AMEX",
                                        tick_neutral_policy=str(policy),
                                        tick_direction_policy=str(dir_policy),
                                        tick_band_ma_period=10,
                                        tick_width_z_lookback=int(lookback),
                                        tick_width_z_enter=float(z_enter),
                                        tick_width_z_exit=float(z_exit),
                                        tick_width_slope_lookback=int(slope_lb),
                                    ),
                                )
                                row = _run_cfg(cfg=cfg)
                                if not row:
                                    continue
                                tick_key = (
                                    str(dir_policy),
                                    str(policy),
                                    float(z_enter),
                                    float(z_exit),
                                    int(slope_lb),
                                    int(lookback),
                                )
                                current = best_by_tick.get(tick_key)
                                if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                                    best_by_tick[tick_key] = {"row": row}

        shortlisted = _ranked_keys_by_row_scores(best_by_tick, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible tick candidates (check $TICK cache/permissions, or lower --min-trades).")
            return
        print("")
        print(f"TICK×Perm: stage1 shortlisted tick={len(shortlisted)} (from {len(best_by_tick)})")

        tod_variants = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
            ("tod=18-05 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 5}),
            ("tod=18-06 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 6}),
            ("tod=17-04 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 4}),
            ("tod=19-04 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 4}),
        ]
        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
            ("spread>=0.0030", {"ema_spread_min_pct": 0.003}),
            ("spread>=0.0040", {"ema_spread_min_pct": 0.004}),
            ("spread>=0.0050", {"ema_spread_min_pct": 0.005}),
            ("spread>=0.0070", {"ema_spread_min_pct": 0.007}),
        ]
        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
        ]

        rows: list[dict] = []
        tested = 0
        total = len(shortlisted) * len(tod_variants) * len(spread_variants) * len(vol_variants)
        t0 = pytime.perf_counter()
        report_every = 200
        for tick_key in shortlisted:
            dir_policy, policy, z_enter, z_exit, slope_lb, lookback = tick_key
            for tod_note, tod_over in tod_variants:
                for spread_note, spread_over in spread_variants:
                    for vol_note, vol_over in vol_variants:
                        tested += 1
                        if tested % report_every == 0 or tested == total:
                            print(
                                _progress_line(
                                    label="tick_perm_joint stage2",
                                    tested=int(tested),
                                    total=int(total),
                                    kept=len(rows),
                                    started_at=t0,
                                    rate_unit="s",
                                ),
                                flush=True,
                            )
                        overrides: dict[str, object] = {}
                        overrides.update(tod_over)
                        overrides.update(spread_over)
                        overrides.update(vol_over)
                        f = _merge_filters(base_filters, overrides=overrides)
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                filters=f,
                                tick_gate_mode="raschke",
                                tick_gate_symbol="TICK-AMEX",
                                tick_gate_exchange="AMEX",
                                tick_neutral_policy=str(policy),
                                tick_direction_policy=str(dir_policy),
                                tick_band_ma_period=10,
                                tick_width_z_lookback=int(lookback),
                                tick_width_z_enter=float(z_enter),
                                tick_width_z_exit=float(z_exit),
                                tick_width_slope_lookback=int(slope_lb),
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = (
                            f"tick=raschke dir={dir_policy} policy={policy} z_in={z_enter:g} z_out={z_exit:g} "
                            f"slope={slope_lb} lb={lookback} | {tod_note} | {spread_note} | {vol_note}"
                        )
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick × permission joint sweep", top_n=int(args.top))

    def _sweep_ema_regime() -> None:
        """Joint interaction hunt: direction (EMA preset) × regime1 (Supertrend bias)."""
        bars_sig = _bars_cached(signal_bar_size)

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Keep this bounded but broad enough to catch the interaction pockets:
        # - 4h: micro + macro ST params
        # - 1d: smaller curated set (heavier and less likely, but still worth checking)
        regimes: list[tuple[str, int, float, str]] = []

        rbar = "4 hours"
        atr_ps_4h = [2, 3, 4, 5, 6, 7, 10, 14, 21]
        mults_4h = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        for atr_p in atr_ps_4h:
            for mult in mults_4h:
                for src in ("hl2", "close"):
                    regimes.append((rbar, int(atr_p), float(mult), str(src)))

        rbar = "1 day"
        atr_ps_1d = [7, 10, 14, 21]
        mults_1d = [0.4, 0.6, 0.8, 1.0, 1.2]
        for atr_p in atr_ps_1d:
            for mult in mults_1d:
                for src in ("hl2", "close"):
                    regimes.append((rbar, int(atr_p), float(mult), str(src)))

        rows: list[dict] = []
        for preset in presets:
            for rbar, atr_p, mult, src in regimes:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        ema_preset=str(preset),
                        entry_signal="ema",
                        regime_mode="supertrend",
                        regime_bar_size=str(rbar),
                        supertrend_atr_period=int(atr_p),
                        supertrend_multiplier=float(mult),
                        supertrend_source=str(src),
                    ),
                )
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"ema={preset} | ST({atr_p},{mult:g},{src})@{rbar}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × regime joint sweep (direction × bias)", top_n=int(args.top))

    def _sweep_chop_joint() -> None:
        """Joint chop filter stack: slope × cooldown × skip-open (keeps everything else fixed)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        slope_vals = [None, 0.005, 0.01, 0.02, 0.03]
        cooldown_vals = [0, 1, 2, 3, 4, 6]
        skip_vals = [0, 1, 2, 3]

        rows: list[dict] = []
        for slope in slope_vals:
            for cooldown in cooldown_vals:
                for skip in skip_vals:
                    overrides: dict[str, object] = {
                        "ema_slope_min_pct": float(slope) if slope is not None else None,
                        "cooldown_bars": int(cooldown),
                        "skip_first_bars": int(skip),
                    }
                    f = _merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    row = _run_cfg(cfg=cfg)
                    if not row:
                        continue
                    slope_note = "-" if slope is None else f"slope>={float(slope):g}"
                    note = f"{slope_note} | cooldown={cooldown} | skip={skip}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Chop joint sweep (slope × cooldown × skip-open)", top_n=int(args.top))

    def _sweep_tick_ema() -> None:
        """Joint interaction hunt: Raschke $TICK (wide-only bias) × EMA preset."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]
        policies = ["allow", "block"]
        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]

        rows: list[dict] = []
        for preset in presets:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                cfg = replace(
                                    base,
                                    strategy=replace(
                                        base.strategy,
                                        entry_signal="ema",
                                        ema_preset=str(preset),
                                        tick_gate_mode="raschke",
                                        tick_gate_symbol="TICK-AMEX",
                                        tick_gate_exchange="AMEX",
                                        tick_neutral_policy=str(policy),
                                        tick_direction_policy="wide_only",
                                        tick_band_ma_period=10,
                                        tick_width_z_lookback=int(lookback),
                                        tick_width_z_enter=float(z_enter),
                                        tick_width_z_exit=float(z_exit),
                                        tick_width_slope_lookback=int(slope_lb),
                                    ),
                                )
                                row = _run_cfg(cfg=cfg)
                                if not row:
                                    continue
                                note = (
                                    f"ema={preset} | tick=wide_only policy={policy} z_in={z_enter:g} "
                                    f"z_out={z_exit:g} slope={slope_lb} lb={lookback}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick × EMA joint sweep (Raschke wide-only bias)", top_n=int(args.top))

    def _sweep_ema_atr() -> None:
        """Joint interaction hunt: direction (EMA preset) × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Stage 1: shortlist EMA presets against the base bias/permissions.
        best_by_ema: dict[str, dict] = {}
        for preset in presets:
            cfg = replace(base, strategy=replace(base.strategy, ema_preset=str(preset), entry_signal="ema"))
            row = _run_cfg(cfg=cfg)
            if not row:
                continue
            best_by_ema[str(preset)] = {"row": row}

        shortlisted = _ranked_keys_by_row_scores(best_by_ema, top_pnl=5, top_pnl_dd=5)
        if not shortlisted:
            print("No eligible EMA presets (try lowering --min-trades).")
            return
        print("")
        print(f"EMA×ATR: stage1 shortlisted ema={len(shortlisted)} (from {len(best_by_ema)})")

        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

        rows: list[dict] = []
        for preset in shortlisted:
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                ema_preset=str(preset),
                                entry_signal="ema",
                                spot_exit_mode="atr",
                                spot_atr_period=int(atr_p),
                                spot_pt_atr_mult=float(pt_m),
                                spot_sl_atr_mult=float(sl_m),
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"ema={preset} | ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × ATR joint sweep (direction × exits)", top_n=int(args.top))

    def _sweep_weekdays() -> None:
        """Gate exploration: which UTC weekdays contribute to the edge."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        day_sets: list[tuple[tuple[int, ...], str]] = [
            ((0, 1, 2, 3, 4), "Mon-Fri"),
            ((0, 1, 2, 3), "Mon-Thu"),
            ((1, 2, 3, 4), "Tue-Fri"),
            ((1, 2, 3), "Tue-Thu"),
            ((2, 3, 4), "Wed-Fri"),
            ((0, 1, 2), "Mon-Wed"),
            ((0, 1, 2, 3, 4, 5, 6), "All days"),
        ]

        rows: list[dict] = []
        for days, label in day_sets:
            cfg = replace(base, strategy=replace(base.strategy, entry_days=tuple(days)))
            row = _run_cfg(cfg=cfg)
            if not row:
                continue
            note = f"days={label}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Weekday sweep (UTC weekday gating)", top_n=int(args.top))

    def _sweep_exit_time() -> None:
        """Session-aware exit experiment: force a daily time-based flatten (ET)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        times = [
            None,
            "04:00",
            "09:30",
            "10:00",
            "11:00",
            "16:00",
            "17:00",
        ]
        rows: list[dict] = []
        for t in times:
            cfg = replace(base, strategy=replace(base.strategy, spot_exit_time_et=t))
            row = _run_cfg(cfg=cfg)
            if not row:
                continue
            note = "-" if t is None else f"exit_time={t} ET"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Exit-time sweep (ET flatten)", top_n=int(args.top))

    def _fmt_sweep_float(value: float, decimals: int | None) -> str:
        if decimals is None:
            return f"{float(value):g}"
        return f"{float(value):.{int(decimals)}f}"

    def _run_atr_exit_profile(profile_name: str) -> None:
        profile = _ATR_EXIT_PROFILE_REGISTRY.get(str(profile_name))
        if not isinstance(profile, dict):
            raise SystemExit(f"Unknown ATR sweep profile: {profile_name!r}")
        axis_tag = str(profile_name).strip().lower()
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        atr_periods = tuple(profile.get("atr_periods") or ())
        pt_mults = tuple(profile.get("pt_mults") or ())
        sl_mults = tuple(profile.get("sl_mults") or ())
        decimals_raw = profile.get("decimals")
        decimals = int(decimals_raw) if isinstance(decimals_raw, int) else None
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            spot_exit_mode="atr",
                            spot_atr_period=int(atr_p),
                            spot_pt_atr_mult=float(pt_m),
                            spot_sl_atr_mult=float(sl_m),
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=None,
                        ),
                    )
                    note = (
                        f"ATR({int(atr_p)}) "
                        f"PTx{_fmt_sweep_float(float(pt_m), decimals)} "
                        f"SLx{_fmt_sweep_float(float(sl_m), decimals)}"
                    )
                    cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag=str(axis_tag),
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=50,
            heartbeat_sec=10.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(rows, title=str(profile.get("title") or "ATR exits sweep"), top_n=int(args.top))

    def _sweep_atr_exits() -> None:
        _run_atr_exit_profile("atr")

    def _sweep_atr_exits_fine() -> None:
        """Fine-grained ATR exit sweep around the current champion neighborhood."""
        _run_atr_exit_profile("atr_fine")

    def _sweep_atr_exits_ultra() -> None:
        """Ultra-fine ATR exit sweep around the current best PT neighborhood."""
        _run_atr_exit_profile("atr_ultra")

    def _sweep_r2_atr() -> None:
        """Joint interaction hunt: regime2 confirm × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Stage 1: coarse scan to shortlist promising regime2 settings.
        r2_variants: list[tuple[dict, str]] = [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off"),
        ]
        r2_bar_sizes = ["4 hours", "1 day"]
        r2_atr_periods = [7, 10, 11, 14, 21]
        r2_multipliers = [0.6, 0.8, 1.0, 1.2, 1.5]
        r2_sources = ["hl2", "close"]
        for r2_bar in r2_bar_sizes:
            for atr_p in r2_atr_periods:
                for mult in r2_multipliers:
                    for src in r2_sources:
                        r2_variants.append(
                            (
                                {
                                    "regime2_mode": "supertrend",
                                    "regime2_bar_size": str(r2_bar),
                                    "regime2_supertrend_atr_period": int(atr_p),
                                    "regime2_supertrend_multiplier": float(mult),
                                    "regime2_supertrend_source": str(src),
                                },
                                f"r2=ST2({r2_bar}:{atr_p},{mult},{src})",
                            )
                        )

        exit_stage1: list[tuple[dict, str]] = [
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.8,
                    "spot_sl_atr_mult": 1.6,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx0.80 SLx1.60",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.9,
                    "spot_sl_atr_mult": 1.6,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx0.90 SLx1.60",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 21,
                    "spot_pt_atr_mult": 0.9,
                    "spot_sl_atr_mult": 1.4,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(21) PTx0.90 SLx1.40",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 1.0,
                    "spot_sl_atr_mult": 1.5,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx1.00 SLx1.50",
            ),
        ]

        stage1: list[tuple[tuple, dict, str]] = []
        for r2_over, r2_note in r2_variants:
            for exit_over, exit_note in exit_stage1:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                        regime2_bar_size=r2_over.get("regime2_bar_size"),
                        regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                        regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                        regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                        spot_exit_mode=str(exit_over["spot_exit_mode"]),
                        spot_atr_period=int(exit_over["spot_atr_period"]),
                        spot_pt_atr_mult=float(exit_over["spot_pt_atr_mult"]),
                        spot_sl_atr_mult=float(exit_over["spot_sl_atr_mult"]),
                        spot_profit_target_pct=exit_over["spot_profit_target_pct"],
                        spot_stop_loss_pct=exit_over["spot_stop_loss_pct"],
                    ),
                )
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                r2_key = (
                    str(getattr(cfg.strategy, "regime2_mode", "off") or "off"),
                    str(getattr(cfg.strategy, "regime2_bar_size", "") or ""),
                    int(getattr(cfg.strategy, "regime2_supertrend_atr_period", 0) or 0),
                    float(getattr(cfg.strategy, "regime2_supertrend_multiplier", 0.0) or 0.0),
                    str(getattr(cfg.strategy, "regime2_supertrend_source", "") or ""),
                )
                note = f"{r2_note} | {exit_note}"
                row["note"] = note
                stage1.append((r2_key, row, note))

        if not stage1:
            print("No eligible results in stage1 (try lowering --min-trades).")
            return

        # Shortlist by best observed metrics per regime2 key.
        best_by_r2: dict[tuple, dict] = {}
        for r2_key, row, note in stage1:
            current = best_by_r2.get(r2_key)
            if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                best_by_r2[r2_key] = {"row": row, "note": note}

        shortlisted_keys = _ranked_keys_by_row_scores(best_by_r2, top_pnl=8, top_pnl_dd=8)

        print("")
        print(f"R2×ATR: stage1 shortlisted r2={len(shortlisted_keys)} (from {len(best_by_r2)})")

        # Stage 2: exit microgrid for shortlisted regime2 settings.
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
        atr_periods = [14, 21]

        rows: list[dict] = []
        for r2_key in shortlisted_keys:
            r2_mode, r2_bar, r2_atr, r2_mult, r2_src = r2_key
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime2_mode=str(r2_mode),
                                regime2_bar_size=str(r2_bar) or None,
                                regime2_supertrend_atr_period=int(r2_atr or 10),
                                regime2_supertrend_multiplier=float(r2_mult or 3.0),
                                regime2_supertrend_source=str(r2_src or "hl2"),
                                spot_exit_mode="atr",
                                spot_atr_period=int(atr_p),
                                spot_pt_atr_mult=float(pt_m),
                                spot_sl_atr_mult=float(sl_m),
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        if str(r2_mode).strip().lower() == "off":
                            r2_note = "r2=off"
                        else:
                            r2_note = f"r2=ST2({r2_bar}:{r2_atr},{r2_mult:g},{r2_src})"
                        note = f"{r2_note} | ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 × ATR joint sweep (PT<1.0 pocket)", top_n=int(args.top))

    def _sweep_r2_tod() -> None:
        """Joint interaction hunt: regime2 confirm × TOD window (keeps exits fixed)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        # Stage 1: scan regime2 settings with the current base TOD.
        r2_variants: list[tuple[dict, str]] = [({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off")]
        for r2_bar in ("4 hours", "1 day"):
            for atr_p in (3, 5, 7, 10, 11, 14, 21):
                for mult in (0.6, 0.8, 1.0, 1.2, 1.5):
                    for src in ("hl2", "close"):
                        r2_variants.append(
                            (
                                {
                                    "regime2_mode": "supertrend",
                                    "regime2_bar_size": str(r2_bar),
                                    "regime2_supertrend_atr_period": int(atr_p),
                                    "regime2_supertrend_multiplier": float(mult),
                                    "regime2_supertrend_source": str(src),
                                },
                                f"r2=ST2({r2_bar}:{atr_p},{mult:g},{src})",
                            )
                        )

        best_by_r2: dict[tuple, dict] = {}
        for r2_over, r2_note in r2_variants:
            cfg = replace(
                base,
                strategy=replace(
                    base.strategy,
                    regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                    regime2_bar_size=r2_over.get("regime2_bar_size"),
                    regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                    regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                    regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                ),
            )
            row = _run_cfg(cfg=cfg)
            if not row:
                continue
            r2_key = (
                str(getattr(cfg.strategy, "regime2_mode", "off") or "off"),
                str(getattr(cfg.strategy, "regime2_bar_size", "") or ""),
                int(getattr(cfg.strategy, "regime2_supertrend_atr_period", 0) or 0),
                float(getattr(cfg.strategy, "regime2_supertrend_multiplier", 0.0) or 0.0),
                str(getattr(cfg.strategy, "regime2_supertrend_source", "") or ""),
            )
            current = best_by_r2.get(r2_key)
            if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                best_by_r2[r2_key] = {"row": row, "note": r2_note}

        shortlisted = _ranked_keys_by_row_scores(best_by_r2, top_pnl=10, top_pnl_dd=10)
        if not shortlisted:
            print("No eligible regime2 candidates (try lowering --min-trades).")
            return
        print("")
        print(f"R2×TOD: stage1 shortlisted r2={len(shortlisted)} (from {len(best_by_r2)})")

        tod_variants: list[tuple[str, dict[str, object]]] = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
            ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
            ("tod=11-16 ET", {"entry_start_hour_et": 11, "entry_end_hour_et": 16}),
        ]
        for start_h in (16, 17, 18, 19, 20):
            for end_h in (2, 3, 4, 5, 6):
                tod_variants.append((f"tod={start_h:02d}-{end_h:02d} ET", {"entry_start_hour_et": start_h, "entry_end_hour_et": end_h}))

        rows: list[dict] = []
        for r2_key in shortlisted:
            r2_mode, r2_bar, r2_atr, r2_mult, r2_src = r2_key
            for tod_note, tod_over in tod_variants:
                f = _merge_filters(base_filters, overrides=tod_over)
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        filters=f,
                        regime2_mode=str(r2_mode),
                        regime2_bar_size=str(r2_bar) or None,
                        regime2_supertrend_atr_period=int(r2_atr or 10),
                        regime2_supertrend_multiplier=float(r2_mult or 3.0),
                        regime2_supertrend_source=str(r2_src or "hl2"),
                    ),
                )
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                if str(r2_mode).strip().lower() == "off":
                    r2_note = "r2=off"
                else:
                    r2_note = f"r2=ST2({r2_bar}:{r2_atr},{r2_mult:g},{r2_src})"
                note = f"{r2_note} | {tod_note}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 × TOD joint sweep", top_n=int(args.top))

    def _sweep_regime_atr() -> None:
        """Joint interaction hunt: regime (bias) × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Stage 1: scan regime settings using a representative low-PT exit.
        best_by_regime: dict[tuple, dict] = {}
        for rbar in ("4 hours", "1 day"):
            for atr_p in (3, 5, 6, 7, 10, 14, 21):
                for mult in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
                    for src in ("hl2", "close"):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=str(rbar),
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                                regime2_mode="off",
                                regime2_bar_size=None,
                                spot_exit_mode="atr",
                                spot_atr_period=14,
                                spot_pt_atr_mult=0.7,
                                spot_sl_atr_mult=1.6,
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        row = _run_cfg(cfg=cfg, bars=bars_sig)
                        if not row:
                            continue
                        key = (str(rbar), int(atr_p), float(mult), str(src))
                        current = best_by_regime.get(key)
                        if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                            best_by_regime[key] = {"row": row}

        shortlisted = _ranked_keys_by_row_scores(best_by_regime, top_pnl=10, top_pnl_dd=10)
        if not shortlisted:
            print("No eligible regime candidates (try lowering --min-trades).")
            return
        print("")
        print(f"Regime×ATR: stage1 shortlisted regimes={len(shortlisted)} (from {len(best_by_regime)})")

        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

        plan: list[tuple[ConfigBundle, str, dict | None]] = []
        for rbar, atr_p, mult, src in shortlisted:
            for exit_atr in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=str(rbar),
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                                regime2_mode="off",
                                regime2_bar_size=None,
                                spot_exit_mode="atr",
                                spot_atr_period=int(exit_atr),
                                spot_pt_atr_mult=float(pt_m),
                                spot_sl_atr_mult=float(sl_m),
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        note = (
                            f"ST({atr_p},{mult:g},{src})@{rbar} | "
                            f"ATR({exit_atr}) PTx{pt_m:.2f} SLx{sl_m:.2f} | r2=off"
                        )
                        plan.append((cfg, note, None))
        _tested, kept = _run_sweep(plan=plan, bars=bars_sig, total=len(plan), progress_label="Regime×ATR stage2")
        rows = [row for _cfg, row, _note, _meta in kept]

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime × ATR joint sweep (PT<1.0 pocket)", top_n=int(args.top))

    def _sweep_ptsl() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        pt_vals = [0.005, 0.01, 0.015, 0.02]
        sl_vals = [0.015, 0.02, 0.03]
        plan = []
        for pt in pt_vals:
            for sl in sl_vals:
                cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                cfg = replace(
                    cfg,
                    strategy=replace(
                        cfg.strategy,
                        spot_profit_target_pct=float(pt),
                        spot_stop_loss_pct=float(sl),
                        spot_exit_mode="pct",
                    ),
                )
                plan.append((cfg, f"PT={pt:.3f} SL={sl:.3f}", None))
        _tested, kept = _run_sweep(plan=plan, bars=bars_sig)
        rows = [row for _, row, _note, _meta in kept]
        _print_leaderboards(rows, title="PT/SL sweep (fixed pct exits)", top_n=int(args.top))

    def _sweep_hf_scalp() -> None:
        """High-frequency spot axis (stacked stop+flip + cadence knobs + stability overlays).

        Designed to discover "many trades/day" shapes under realism2 without requiring a seeded champion.

        Stage 1: stacked stop-loss + flip-profit (fast-runner-friendly baseline).
        Stage 2: sweep cadence knobs around the best stage-1 candidates (TOD, cooldown, skip-open, confirm).
        Stage 3: apply a small set of TQQQ v34-inspired stability overlays (shock/permission/regime interactions).
        Stage 4: expand slower knobs (spot_close_eod) on a tiny shortlist.
        """
        bars_sig = _bars_cached(signal_bar_size)

        def _shortlist(
            items: list[tuple[ConfigBundle, dict, str]],
            *,
            top_pnl_dd: int,
            top_pnl: int,
            top_trades: int = 0,
        ) -> list[tuple[ConfigBundle, dict, str]]:
            by_dd = sorted(items, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[: int(top_pnl_dd)]
            by_pnl = sorted(items, key=lambda t: _score_row_pnl(t[1]), reverse=True)[: int(top_pnl)]
            by_trades = (
                sorted(
                    items,
                    key=lambda t: (
                        int(t[1].get("trades") or 0),
                        float(t[1].get("pnl_over_dd") or float("-inf")),
                        float(t[1].get("pnl") or float("-inf")),
                    ),
                    reverse=True,
                )[: int(top_trades)]
                if int(top_trades) > 0
                else []
            )
            seen: set[str] = set()
            out: list[tuple[ConfigBundle, dict, str]] = []
            for cfg, row, note in by_dd + by_pnl + by_trades:
                key = _milestone_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                out.append((cfg, row, note))
            return out

        def _print_top_trades(rows: list[dict], *, title: str, top_n: int) -> None:
            ranked = sorted(rows, key=lambda r: int(r.get("trades") or 0), reverse=True)[: max(0, int(top_n))]
            if not ranked:
                return
            print("")
            print(f"{title} — Top by trades")
            print("-" * max(18, len(title) + 15))
            for idx, r in enumerate(ranked, 1):
                trades = int(r.get("trades") or 0)
                win = float(r.get("win_rate") or 0.0) * 100.0
                pnl = float(r.get("pnl") or 0.0)
                dd = float(r.get("dd") or 0.0)
                pnl_dd = r.get("pnl_over_dd")
                pnl_dd_s = f"{float(pnl_dd):6.2f}" if pnl_dd is not None else "  None"
                roi = float(r.get("roi") or 0.0) * 100.0
                dd_pct = float(r.get("dd_pct") or 0.0) * 100.0
                note = str(r.get("note") or "")
                print(
                    f"{idx:2d}. tr={trades:4d} win={win:5.1f}% pnl={pnl:9.1f} dd={dd:8.1f} pnl/dd={pnl_dd_s} "
                    f"roi={roi:6.2f}% dd%={dd_pct:6.2f}% {note}"
                )

        # Stage 1: stop-loss + flip-profit baseline (keep it fast-runner-friendly).
        #
        # Note: v1 used very tight stops (sub-0.6%), which produced many 1y winners but was negative over 10y for
        # every candidate. v2 widens the stop grid + EMA presets to reduce whipsaw and improve decade stability.
        ema_presets = ["3/7", "4/9", "5/13", "8/21", "9/21", "21/50"]
        stop_only_vals = [0.0060, 0.0080, 0.0100, 0.0120, 0.0150, 0.0200]
        flip_hold_vals = [0, 2, 4]
        # Keep stages 1-3 on the fast summary runner path (single-position, close_eod=False),
        # then expand close_eod on a tiny shortlist at the end.
        stage_fast_close_eod = False
        expand_close_eod_vals = [False, True]

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                ema_entry_mode="trend",
                exit_on_signal_flip=False,
                flip_exit_only_if_profit=True,
                flip_exit_min_hold_bars=0,
            ),
        )

        stage1: list[tuple[ConfigBundle, dict, str]] = []
        rows: list[dict] = []

        # Stage 1 session baseline: wide RTH window, no cooldown/skip, no confirm.
        # Keep a small permission grid; permission gating is a proven stabilizer in this codebase.
        perm_variants_stage1: list[tuple[dict[str, object], str]] = [
            ({}, "perm=off"),
            ({"ema_spread_min_pct": 0.003, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.04}, "perm=v34"),
        ]

        regime_variants_stage1: list[tuple[dict[str, object], str]] = [
            (
                {"regime_mode": "ema", "regime_ema_preset": None, "regime_bar_size": str(signal_bar_size)},
                "regime=off",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
                "regime=ST(7,0.5,hl2)@4h",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                },
                "regime=ST(14,0.6,hl2)@1d",
            ),
        ]
        for ema_preset in ema_presets:
            for regime_patch, regime_note in regime_variants_stage1:
                for perm_patch, perm_note in perm_variants_stage1:
                    f = _mk_filters(
                        entry_start_hour_et=9,
                        entry_end_hour_et=16,
                        cooldown_bars=0,
                        skip_first_bars=0,
                        overrides=perm_patch,
                    )
                    for sl in stop_only_vals:
                        for hold in flip_hold_vals:
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    ema_preset=str(ema_preset),
                                    entry_confirm_bars=0,
                                    spot_exit_mode="pct",
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=float(sl),
                                    exit_on_signal_flip=True,
                                    flip_exit_only_if_profit=True,
                                    flip_exit_min_hold_bars=int(hold),
                                    flip_exit_gate_mode="off",
                                    spot_close_eod=bool(stage_fast_close_eod),
                                    spot_short_risk_mult=0.01,
                                    filters=f,
                                    **regime_patch,
                                ),
                            )
                            row = _run_cfg(cfg=cfg, bars=bars_sig)
                            if not row:
                                continue
                            note = (
                                f"stop+flip | EMA={ema_preset} confirm=0 | {regime_note} | {perm_note} | "
                                f"tod=9-16 ET skip=0 cd=0 close_eod={int(stage_fast_close_eod)} | "
                                f"SL={sl:.4f} hold={hold}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            stage1.append((cfg, row, note))
                            rows.append(row)

        _print_leaderboards(rows, title="HF scalper: stage1 (stop+flip)", top_n=int(args.top))
        _print_top_trades(rows, title="HF scalper: stage1 (stop+flip)", top_n=int(args.top))

        if not stage1:
            print("HF scalper: stage1 produced 0 results; nothing to refine.", flush=True)
            return

        # Stage 2: sweep cadence knobs around the best stage1 candidates.
        target_trades = max(0, int(args.milestone_min_trades or 0))
        stage1_hi = [t for t in stage1 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        shortlist_pool = stage1_hi if stage1_hi else stage1
        shortlisted = _shortlist(shortlist_pool, top_pnl_dd=10, top_pnl=10, top_trades=10)
        print("")
        print(f"HF scalper: stage2 seeds={len(shortlisted)} (pool={len(shortlist_pool)} target_trades={target_trades})", flush=True)

        confirm_vals = [0, 1]
        tod_variants = [(9, 16, "tod=9-16 ET"), (10, 15, "tod=10-15 ET"), (11, 16, "tod=11-16 ET")]
        cooldown_vals = [0, 2]
        skip_open_vals = [0, 1, 2]
        close_eod_vals = [False]

        stage2: list[tuple[ConfigBundle, dict, str]] = []
        rows2: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted:
            for confirm in confirm_vals:
                for entry_s, entry_e, tod_note in tod_variants:
                    for cooldown in cooldown_vals:
                        for skip_open in skip_open_vals:
                            for close_eod in close_eod_vals:
                                base_payload = _filters_payload(seed_cfg.strategy.filters) or {}
                                raw = dict(base_payload)
                                raw["entry_start_hour_et"] = int(entry_s)
                                raw["entry_end_hour_et"] = int(entry_e)
                                raw["cooldown_bars"] = int(cooldown)
                                raw["skip_first_bars"] = int(skip_open)
                                f = _parse_filters(raw)
                                if _filters_payload(f) is None:
                                    f = None
                                cfg = replace(
                                    seed_cfg,
                                    strategy=replace(
                                        seed_cfg.strategy,
                                        entry_confirm_bars=int(confirm),
                                        spot_close_eod=bool(close_eod),
                                        filters=f,
                                    ),
                                )
                                row = _run_cfg(cfg=cfg, bars=bars_sig)
                                if not row:
                                    continue
                                note = (
                                    f"{seed_note} | {tod_note} skip={skip_open} cd={cooldown} "
                                    f"close_eod={int(close_eod)} confirm={confirm}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                stage2.append((cfg, row, note))
                                rows2.append(row)

        _print_leaderboards(rows2, title="HF scalper: stage2 (cadence knobs)", top_n=int(args.top))
        _print_top_trades(rows2, title="HF scalper: stage2 (cadence knobs)", top_n=int(args.top))

        if not stage2:
            print("HF scalper: stage2 produced 0 results; skipping overlays.", flush=True)
            return

        # Stage 3: apply a small overlay grid (v34-inspired) to the best stage2 candidates.
        stage2_hi = [t for t in stage2 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        overlay_pool = stage2_hi if stage2_hi else stage2
        shortlisted2 = _shortlist(overlay_pool, top_pnl_dd=8, top_pnl=8, top_trades=8)
        print("")
        print(f"HF scalper: stage3 seeds={len(shortlisted2)} (pool={len(overlay_pool)})", flush=True)

        # Overlays:
        # - Regime: off vs 4h supertrend (v34-like)
        regime_variants: list[tuple[dict[str, object], str]] = [
            (
                {
                    "regime_mode": "ema",
                    "regime_ema_preset": None,
                    "regime_bar_size": str(signal_bar_size),
                },
                "regime=off",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
                "regime=ST(7,0.5,hl2)@4h",
            ),
            (
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 0.6,
                    "supertrend_source": "hl2",
                },
                "regime=ST(14,0.6,hl2)@1d",
            ),
        ]

        # - Permission: off vs v34-like thresholds (kept small; SLV needs its own calibration later).
        perm_variants: list[tuple[dict[str, object] | None, str]] = [
            (None, "perm=seed"),
            (
                {"ema_spread_min_pct": None, "ema_slope_min_pct": None, "ema_spread_min_pct_down": None},
                "perm=off",
            ),
            ({"ema_spread_min_pct": 0.003, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.04}, "perm=v34"),
        ]

        # - Shock: off vs detect(tr_ratio) with SLV-scaled min_atr_pct and a couple ratio thresholds.
        shock_variants: list[tuple[dict[str, object] | None, str]] = [
            (None, "shock=seed"),
            ({"shock_gate_mode": "off"}, "shock=off"),
            (
                {
                    "shock_gate_mode": "block",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 4.5,
                    "shock_daily_off_atr_pct": 4.0,
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                },
                "shock=block daily_atr% 4.5/4.0",
            ),
            (
                {
                    "shock_gate_mode": "detect",
                    "shock_detector": "tr_ratio",
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_on_ratio": 1.30,
                    "shock_off_ratio": 1.20,
                    "shock_min_atr_pct": 1.5,
                    "shock_risk_scale_target_atr_pct": 3.5,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_stop_loss_pct_mult": 1.0,
                    "shock_profit_target_pct_mult": 1.0,
                },
                "shock=detect tr_ratio(3/21) 1.30/1.20 min_atr%=1.5",
            ),
        ]

        # - Short sizing asymmetry: mimic v34's "shorts can be toxic" behavior.
        short_mult_vals = [1.0, 0.2, 0.01, 0.0]

        flip_variants: list[tuple[dict[str, object], str]] = [
            ({"exit_on_signal_flip": False}, "flip=off"),
            (
                {
                    "exit_on_signal_flip": True,
                    "flip_exit_only_if_profit": True,
                    "flip_exit_min_hold_bars": 2,
                    "flip_exit_gate_mode": "off",
                },
                "flip=profit hold=2",
            ),
        ]

        stage3: list[tuple[ConfigBundle, dict, str]] = []
        rows3: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted2:
            seed_filters = seed_cfg.strategy.filters
            entry_s = getattr(seed_filters, "entry_start_hour_et", None) if seed_filters is not None else None
            entry_e = getattr(seed_filters, "entry_end_hour_et", None) if seed_filters is not None else None
            cooldown = int(getattr(seed_filters, "cooldown_bars", 0) or 0) if seed_filters is not None else 0
            skip_open = int(getattr(seed_filters, "skip_first_bars", 0) or 0) if seed_filters is not None else 0

            for regime_patch, regime_note in regime_variants:
                for perm_patch, perm_note in perm_variants:
                    for shock_patch, shock_note in shock_variants:
                        base_payload = _filters_payload(seed_cfg.strategy.filters) or {}
                        raw = dict(base_payload)
                        if entry_s is not None and entry_e is not None:
                            raw["entry_start_hour_et"] = int(entry_s)
                            raw["entry_end_hour_et"] = int(entry_e)
                        raw["cooldown_bars"] = int(cooldown)
                        raw["skip_first_bars"] = int(skip_open)
                        if perm_patch is not None:
                            raw.update(perm_patch)
                        if shock_patch is not None:
                            raw.update(shock_patch)
                        f2 = _parse_filters(raw)
                        if _filters_payload(f2) is None:
                            f2 = None

                        for short_mult in short_mult_vals:
                            for flip_patch, flip_note in flip_variants:
                                cfg = seed_cfg
                                cfg = replace(
                                    cfg,
                                    strategy=replace(
                                        cfg.strategy,
                                        filters=f2,
                                        spot_short_risk_mult=float(short_mult),
                                        **regime_patch,
                                        **flip_patch,
                                    ),
                                )
                                row = _run_cfg(cfg=cfg, bars=bars_sig)
                                if not row:
                                    continue
                                note = (
                                    f"{seed_note} | {regime_note} | {perm_note} | {shock_note} | "
                                    f"short_mult={short_mult:g} | {flip_note}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                stage3.append((cfg, row, note))
                                rows3.append(row)

        _print_leaderboards(rows3, title="HF scalper: stage3 (v34-inspired overlays)", top_n=int(args.top))
        _print_top_trades(rows3, title="HF scalper: stage3 (v34-inspired overlays)", top_n=int(args.top))

        # Stage 4: expand close_eod on a tiny shortlist.
        if not stage3:
            return
        stage3_hi = [t for t in stage3 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        expand_pool = stage3_hi if stage3_hi else stage3
        shortlisted3 = _shortlist(expand_pool, top_pnl_dd=6, top_pnl=6, top_trades=6)
        print("")
        print(f"HF scalper: expand seeds={len(shortlisted3)} (pool={len(expand_pool)})", flush=True)

        rows4: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted3:
            for close_eod in expand_close_eod_vals:
                cfg = replace(
                    seed_cfg,
                    strategy=replace(
                        seed_cfg.strategy,
                        spot_close_eod=bool(close_eod),
                    ),
                )
                row = _run_cfg(cfg=cfg, bars=bars_sig)
                if not row:
                    continue
                note = f"{seed_note} | expand close_eod={int(close_eod)}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows4.append(row)

        _print_leaderboards(rows4, title="HF scalper: expansion (close_eod)", top_n=int(args.top))
        _print_top_trades(rows4, title="HF scalper: expansion (close_eod)", top_n=int(args.top))

    def _sweep_hold() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for hold in (0, 1, 2, 3, 4, 6, 8):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, flip_exit_min_hold_bars=int(hold)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"hold={hold}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Flip-exit min hold sweep", top_n=int(args.top))

    def _sweep_spot_short_risk_mult() -> None:
        """Sweep the short sizing multiplier (only affects spot_sizing_mode=risk_pct)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        vals = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0]
        rows: list[dict] = []
        for mult in vals:
            cfg = replace(base, strategy=replace(base.strategy, spot_short_risk_mult=float(mult)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"spot_short_risk_mult={mult:g}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Spot short risk multiplier sweep", top_n=int(args.top))

    def _sweep_orb() -> None:
        bars_15m = _bars_cached("15 mins")
        base = _base_bundle(bar_size="15 mins", filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_15m
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        rr_vals = [0.618, 0.707, 0.786, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override (not merge) filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    entry_signal="orb",
                                    ema_preset=None,
                                    entry_confirm_bars=0,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
                                    orb_risk_reward=float(rr),
                                    orb_target_mode=str(target_mode),
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=None,
                                ),
                            )
                            row = _run_cfg(
                                cfg=cfg,
                                bars=bars_15m,
                            )
                            if not row:
                                continue
                            vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                            note = (
                                f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} "
                                f"tod={start_h:02d}-{end_h:02d} ET {vol_note}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="D) ORB sweep (open-time + window)", top_n=int(args.top))

    def _sweep_orb_joint() -> None:
        """Joint ORB exploration: ORB params × (regime bias) × (optional tick bias).

        Note: ORB uses its own stop/target derived from the opening range, so EMA-based
        quality gates (spread/slope) aren't applicable here unless we compute EMA in
        parallel. We stick to regime/tick/volume/TOD gates that remain well-defined.
        """
        bars_15m = _bars_cached("15 mins")

        # Start from the selected base shape, but neutralize regime/tick so stage1 can
        # shortlist ORB mechanics without hidden gating.
        base = _base_bundle(bar_size="15 mins", filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                entry_signal="orb",
                ema_preset=None,
                entry_confirm_bars=0,
                regime_mode="ema",
                regime_bar_size="15 mins",
                regime_ema_preset=None,
                regime2_mode="off",
                regime2_bar_size=None,
                tick_gate_mode="off",
            ),
        )
        base_row = _run_cfg(
            cfg=base,
            bars=bars_15m,
        )
        if base_row:
            base_row["note"] = "base (orb, no regime/tick)"
            _record_milestone(base, base_row, str(base_row["note"]))

        rr_vals = [0.618, 0.707, 0.786, 0.8, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]

        # Stage 1: find the best ORB mechanics without regime/tick overlays.
        best_by_orb: dict[tuple, dict] = {}
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
                                    orb_risk_reward=float(rr),
                                    orb_target_mode=str(target_mode),
                                ),
                            )
                            row = _run_cfg(
                                cfg=cfg,
                                bars=bars_15m,
                            )
                            if not row:
                                continue
                            orb_key = (str(open_time), int(window_mins), str(target_mode), float(rr), vol_min)
                            best_by_orb[orb_key] = {"row": row}

        shortlisted = _ranked_keys_by_row_scores(best_by_orb, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible ORB candidates (try lowering --min-trades).")
            return
        print("")
        print(f"ORB×(regime/tick): stage1 shortlisted orb={len(shortlisted)} (from {len(best_by_orb)})")

        # Stage 2: apply a small curated set of regime overlays + tick "wide-only" bias.
        regime_variants: list[tuple[str, dict[str, object]]] = [
            ("regime=off", {"regime_mode": "ema", "regime_bar_size": "15 mins", "regime_ema_preset": None}),
        ]
        for atr_p, mult, src in (
            (3, 0.4, "hl2"),
            (6, 0.6, "hl2"),
            (7, 0.6, "hl2"),
            (14, 0.6, "hl2"),
            (21, 0.5, "close"),
            (21, 0.6, "hl2"),
        ):
            regime_variants.append(
                (
                    f"ST({atr_p},{mult:g},{src})@4h",
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "4 hours",
                        "supertrend_atr_period": int(atr_p),
                        "supertrend_multiplier": float(mult),
                        "supertrend_source": str(src),
                    },
                )
            )

        tick_variants: list[tuple[str, dict[str, object]]] = [
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=wide_only allow (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "allow",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
            (
                "tick=wide_only block (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
        ]

        rows: list[dict] = []
        for open_time, window_mins, target_mode, rr, vol_min in shortlisted:
            start_h, end_h = 9, 16
            if str(open_time) == "18:00":
                start_h, end_h = 18, 4
            f = _mk_filters(
                entry_start_hour_et=int(start_h),
                entry_end_hour_et=int(end_h),
                volume_ratio_min=vol_min,
                volume_ema_period=20 if vol_min is not None else None,
            )

            for regime_note, reg_over in regime_variants:
                for tick_note, tick_over in tick_variants:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            filters=f,
                            orb_open_time_et=str(open_time),
                            orb_window_mins=int(window_mins),
                            orb_risk_reward=float(rr),
                            orb_target_mode=str(target_mode),
                            regime_mode=str(reg_over.get("regime_mode") or "ema"),
                            regime_bar_size=str(reg_over.get("regime_bar_size") or "15 mins"),
                            regime_ema_preset=reg_over.get("regime_ema_preset"),
                            supertrend_atr_period=int(reg_over.get("supertrend_atr_period") or 10),
                            supertrend_multiplier=float(reg_over.get("supertrend_multiplier") or 3.0),
                            supertrend_source=str(reg_over.get("supertrend_source") or "hl2"),
                            tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                            tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                            tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                            tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                            tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                            tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                            tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                            tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                            tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                            tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
                        ),
                    )
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_15m,
                    )
                    if not row:
                        continue
                    vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                    note = (
                        f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} "
                        f"tod={start_h:02d}-{end_h:02d} ET {vol_note} | {regime_note} | {tick_note}"
                    )
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="ORB joint sweep (ORB × regime × tick)", top_n=int(args.top))

    def _sweep_regime() -> None:
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        profile = _REGIME_ST_PROFILE
        regime_bar_sizes = tuple(profile.get("bars") or ())
        atr_periods = tuple(profile.get("atr_periods") or ())
        multipliers = tuple(profile.get("multipliers") or ())
        sources = tuple(profile.get("sources") or ())
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for rbar in regime_bar_sizes:
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=str(rbar),
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                            ),
                        )
                        note = f"ST({atr_p},{mult},{src}) @{rbar}"
                        cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag="regime",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=100,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(rows, title="Regime sweep (Supertrend params + timeframe)", top_n=int(args.top))

    def _sweep_regime2() -> None:
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(cfg=base)
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        profile = _REGIME2_ST_PROFILE
        atr_periods = tuple(profile.get("atr_periods") or ())
        multipliers = tuple(profile.get("multipliers") or ())
        sources = tuple(profile.get("sources") or ())
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for atr_p in atr_periods:
            for mult in multipliers:
                for src in sources:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            regime2_mode="supertrend",
                            regime2_bar_size="4 hours",
                            regime2_supertrend_atr_period=int(atr_p),
                            regime2_supertrend_multiplier=float(mult),
                            regime2_supertrend_source=str(src),
                        ),
                    )
                    note = f"ST2(4h:{atr_p},{mult},{src})"
                    cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag="regime2",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=100,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Dual regime sweep (regime2 Supertrend @ 4h)", top_n=int(args.top))

    def _sweep_regime2_ema() -> None:
        """Confirm layer: EMA trend gate on a higher timeframe (4h/1d)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]
        rows: list[dict] = []
        for r2_bar in ("4 hours", "1 day"):
            for preset in presets:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        regime2_mode="ema",
                        regime2_bar_size=str(r2_bar),
                        regime2_ema_preset=str(preset),
                    ),
                )
                row = _run_cfg(cfg=cfg)
                if not row:
                    continue
                note = f"r2=EMA({preset})@{r2_bar}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 EMA sweep (trend confirm)", top_n=int(args.top))

    def _sweep_joint() -> None:
        """Targeted interaction hunt: sweep regime + regime2 together (keeps base filters)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Keep this tight and focused; the point is to cover interaction edges that the combo_fast funnel can miss.
        regime_bar_sizes = ["4 hours"]
        regime_atr_periods = [10, 14, 21]
        regime_multipliers = [0.4, 0.5, 0.6]
        regime_sources = ["close", "hl2"]

        r2_bar_sizes = ["4 hours", "1 day"]
        r2_atr_periods = [3, 4, 5, 6, 7, 10, 14]
        r2_multipliers = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        r2_sources = ["close", "hl2"]

        rows: list[dict] = []
        for rbar in regime_bar_sizes:
            for atr_p in regime_atr_periods:
                for mult in regime_multipliers:
                    for src in regime_sources:
                        for r2_bar in r2_bar_sizes:
                            for r2_atr in r2_atr_periods:
                                for r2_mult in r2_multipliers:
                                    for r2_src in r2_sources:
                                        cfg = replace(
                                            base,
                                            strategy=replace(
                                                base.strategy,
                                                regime_mode="supertrend",
                                                regime_bar_size=rbar,
                                                supertrend_atr_period=int(atr_p),
                                                supertrend_multiplier=float(mult),
                                                supertrend_source=str(src),
                                                regime2_mode="supertrend",
                                                regime2_bar_size=str(r2_bar),
                                                regime2_supertrend_atr_period=int(r2_atr),
                                                regime2_supertrend_multiplier=float(r2_mult),
                                                regime2_supertrend_source=str(r2_src),
                                            ),
                                        )
                                        row = _run_cfg(cfg=cfg)
                                        if not row:
                                            continue
                                        note = (
                                            f"ST({atr_p},{mult},{src})@{rbar} + "
                                            f"ST2({r2_bar}:{r2_atr},{r2_mult},{r2_src})"
                                        )
                                        row["note"] = note
                                        _record_milestone(cfg, row, note)
                                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Joint sweep (regime × regime2)", top_n=int(args.top))

    def _sweep_micro_st() -> None:
        """Micro sweep around the current ST + ST2 neighborhood (tighter, more granular)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        regime_atr_periods = [14, 21]
        regime_multipliers = [0.4, 0.45, 0.5, 0.55, 0.6]

        r2_atr_periods = [4, 5, 6]
        r2_multipliers = [0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4]

        rows: list[dict] = []
        for atr_p in regime_atr_periods:
            for mult in regime_multipliers:
                for r2_atr in r2_atr_periods:
                    for r2_mult in r2_multipliers:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size="4 hours",
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source="close",
                                regime2_mode="supertrend",
                                regime2_bar_size="4 hours",
                                regime2_supertrend_atr_period=int(r2_atr),
                                regime2_supertrend_multiplier=float(r2_mult),
                                regime2_supertrend_source="close",
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"ST({atr_p},{mult},close) + ST2(4h:{r2_atr},{r2_mult},close)"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Micro ST sweep (granular mults)", top_n=int(args.top))

    def _sweep_flip_exit() -> None:
        """Targeted exit semantics: flip-exit mode + profit-only gating."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        for exit_on_flip in (True, False):
            for mode in ("entry", "state", "cross"):
                for only_profit in (False, True):
                    for hold in (0, 2, 4, 6):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                exit_on_signal_flip=bool(exit_on_flip),
                                flip_exit_mode=str(mode),
                                flip_exit_only_if_profit=bool(only_profit),
                                flip_exit_min_hold_bars=int(hold),
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = (
                            f"flip={'on' if exit_on_flip else 'off'} mode={mode} "
                            f"hold={hold} only_profit={int(only_profit)}"
                        )
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Flip-exit semantics sweep", top_n=int(args.top))

    def _sweep_confirm() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for confirm in (0, 1, 2, 3):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, entry_confirm_bars=int(confirm)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"confirm={confirm}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Confirm-bars sweep (quality gate)", top_n=int(args.top))

    def _run_spread_profile(profile_name: str) -> None:
        profile = _SPREAD_PROFILE_REGISTRY.get(str(profile_name))
        if not isinstance(profile, dict):
            raise SystemExit(f"Unknown spread sweep profile: {profile_name!r}")
        axis_tag = str(profile_name).strip().lower()
        field_name = str(profile.get("field") or "")
        note_prefix = str(profile.get("note_prefix") or "spread")
        decimals_raw = profile.get("decimals")
        decimals = int(decimals_raw) if isinstance(decimals_raw, int) else None
        values = tuple(profile.get("values") or ())
        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for raw in values:
            spread_val = float(raw) if raw is not None else None
            overrides: dict[str, object] = {field_name: spread_val}
            f = _mk_filters(overrides=overrides)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            note = "-" if spread_val is None else f"{note_prefix}>={_fmt_sweep_float(spread_val, decimals)}"
            cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag=str(axis_tag),
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=20,
            heartbeat_sec=8.0,
        )
        if int(tested_total) < 0:
            return
        _print_leaderboards(rows, title=str(profile.get("title") or "EMA spread sweep"), top_n=int(args.top))

    def _sweep_spread() -> None:
        _run_spread_profile("spread")

    def _sweep_spread_fine() -> None:
        """Fine-grained sweep around the current champion spread gate."""
        _run_spread_profile("spread_fine")

    def _sweep_spread_down() -> None:
        """Directional permission: sweep stricter EMA spread gate for down entries only."""
        _run_spread_profile("spread_down")

    def _sweep_slope() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for slope in (None, 0.005, 0.01, 0.02, 0.03, 0.05):
            f = _mk_filters(ema_slope_min_pct=float(slope) if slope is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = "-" if slope is None else f"slope>={slope}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA slope sweep (quality gate)", top_n=int(args.top))

    def _sweep_slope_signed() -> None:
        """Directional slope gate: require EMA fast slope to be positive/negative by direction."""
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []

        thr_vals = [None, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
        variants: list[tuple[float | None, float | None, str]] = [(None, None, "signed_slope=off")]
        for up_thr in thr_vals:
            if up_thr is None:
                continue
            variants.append((float(up_thr), None, f"slope_up>={up_thr:g}"))
        for down_thr in thr_vals:
            if down_thr is None:
                continue
            variants.append((None, float(down_thr), f"slope_down>={down_thr:g}"))
        for both_thr in (0.005, 0.01, 0.02, 0.03):
            variants.append((float(both_thr), float(both_thr), f"slope_signed>={both_thr:g} (both)"))

        for up_thr, down_thr, note in variants:
            f = _mk_filters(
                overrides={
                    "ema_slope_signed_min_pct_up": up_thr,
                    "ema_slope_signed_min_pct_down": down_thr,
                }
            )
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA signed-slope sweep (directional permission)", top_n=int(args.top))

    def _sweep_cooldown() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for cooldown in (0, 1, 2, 3, 4, 6, 8):
            f = _mk_filters(cooldown_bars=int(cooldown))
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"cooldown={cooldown}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Cooldown sweep (quality gate)", top_n=int(args.top))

    def _sweep_skip_open() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for skip in (0, 1, 2, 3, 4, 6):
            f = _mk_filters(skip_first_bars=int(skip))
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"skip_first={skip}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Skip-open sweep (quality gate)", top_n=int(args.top))

    def _sweep_shock() -> None:
        """Shock overlay sweep (detectors, modes, and a few core threshold grids)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        profile = _SHOCK_SWEEP_PROFILE
        modes = tuple(profile.get("modes") or ())
        dir_variants = tuple(profile.get("dir_variants") or ())
        sl_mults = tuple(profile.get("sl_mults") or ())
        pt_mults = tuple(profile.get("pt_mults") or ())
        short_risk_factors = tuple(profile.get("short_risk_factors") or ())

        presets: list[tuple[str, dict[str, object], str]] = []
        for detector, fast, slow, on, off, min_pct in tuple(profile.get("ratio_rows") or ()):
            presets.append(
                (
                    str(detector),
                    {
                        "shock_atr_fast_period": int(fast),
                        "shock_atr_slow_period": int(slow),
                        "shock_on_ratio": float(on),
                        "shock_off_ratio": float(off),
                        "shock_min_atr_pct": float(min_pct),
                    },
                    f"{detector} fast={fast} slow={slow} on={on:g} off={off:g} min={min_pct:g}",
                )
            )
        for period, on_atr, off_atr, tr_on in tuple(profile.get("daily_atr_rows") or ()):
            presets.append(
                (
                    "daily_atr_pct",
                    {
                        "shock_daily_atr_period": int(period),
                        "shock_daily_on_atr_pct": float(on_atr),
                        "shock_daily_off_atr_pct": float(off_atr),
                        "shock_daily_on_tr_pct": float(tr_on) if tr_on is not None else None,
                    },
                    f"daily_atr_pct p={period} on={on_atr:g} off={off_atr:g} tr_on={tr_on if tr_on is not None else '-'}",
                )
            )
        for lb, dd_on, dd_off in tuple(profile.get("drawdown_rows") or ()):
            presets.append(
                (
                    "daily_drawdown",
                    {
                        "shock_drawdown_lookback_days": int(lb),
                        "shock_on_drawdown_pct": float(dd_on),
                        "shock_off_drawdown_pct": float(dd_off),
                    },
                    f"daily_drawdown lb={lb} on={dd_on:g} off={dd_off:g}",
                )
            )

        rows: list[dict] = []
        cfg_pairs: list[tuple[ConfigBundle, str]] = []
        for detector, params, det_note in presets:
            for mode in modes:
                for dir_src, dir_lb, dir_note in dir_variants:
                    for sl_mult in sl_mults:
                        for pt_mult in pt_mults:
                            for short_factor in short_risk_factors:
                                overrides = {
                                    "shock_gate_mode": str(mode),
                                    "shock_detector": str(detector),
                                    "shock_direction_source": str(dir_src),
                                    "shock_direction_lookback": int(dir_lb),
                                    "shock_stop_loss_pct_mult": float(sl_mult),
                                    "shock_profit_target_pct_mult": float(pt_mult),
                                    "shock_short_risk_mult_factor": float(short_factor),
                                }
                                overrides.update(params)
                                f = _mk_filters(overrides=overrides)
                                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                note = (
                                    f"shock={mode} {det_note} | {dir_note} | "
                                    f"sl_mult={sl_mult:g} pt_mult={pt_mult:g} short_factor={short_factor:g}"
                                )
                                cfg_pairs.append((cfg, note))
        tested_total = _run_cfg_pairs_grid(
            axis_tag="shock",
            cfg_pairs=cfg_pairs,
            rows=rows,
            report_every=50,
            heartbeat_sec=20.0,
        )
        if int(tested_total) < 0:
            return

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Shock sweep (modes × detectors × thresholds)", top_n=int(args.top))

    def _sweep_risk_overlays() -> None:
        """Risk-off / risk-panic / risk-pop TR% overlays (TR median + gap pressure + optional TR-velocity)."""
        nonlocal run_calls_total
        bars_sig = _bars_cached(signal_bar_size)
        skip_pop = bool(getattr(args, "risk_overlays_skip_pop", False))

        def _parse_tr_thresholds(flag: str, raw: object | None) -> list[float] | None:
            s = str(raw or "").strip()
            if not s:
                return None
            out: list[float] = []
            for part in s.split(","):
                part = str(part or "").strip()
                if not part:
                    continue
                try:
                    v = float(part)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"Invalid {flag}: {part!r}") from exc
                if v <= 0:
                    continue
                out.append(float(v))
            if not out:
                return None
            out = sorted(set(out))
            return out

        def _parse_nonneg_factors(flag: str, raw: object | None) -> list[float] | None:
            s = str(raw or "").strip()
            if not s:
                return None
            out: list[float] = []
            for part in s.split(","):
                part = str(part or "").strip()
                if not part:
                    continue
                try:
                    v = float(part)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"Invalid {flag}: {part!r}") from exc
                if v < 0:
                    continue
                out.append(float(v))
            if not out:
                return None
            return sorted(set(out))

        # Risk-off: TR% median above threshold (no gap condition).
        riskoff_trs = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]
        riskoff_trs_over = _parse_tr_thresholds("--risk-overlays-riskoff-trs", args.risk_overlays_riskoff_trs)
        if riskoff_trs_over is not None:
            riskoff_trs = riskoff_trs_over
        riskoff_lbs = [3, 5, 7, 10]
        riskoff_modes = ["hygiene", "directional"]
        # Optional late-day cutoff (ET hour). When set, this only matters on risk-off days.
        riskoff_cutoffs_et = [None, 15, 16]
        riskoff_total = len(riskoff_trs) * len(riskoff_lbs) * len(riskoff_modes) * len(riskoff_cutoffs_et)

        # Risk-panic: TR% median + negative gap ratio.
        panic_trs = [8.0, 9.0, 10.0, 12.0]
        panic_trs_over = _parse_tr_thresholds("--risk-overlays-riskpanic-trs", args.risk_overlays_riskpanic_trs)
        if panic_trs_over is not None:
            panic_trs = panic_trs_over
        panic_long_factors_raw = _parse_nonneg_factors(
            "--risk-overlays-riskpanic-long-factors", args.risk_overlays_riskpanic_long_factors
        )
        panic_long_factors: list[float | None] = [None]
        if panic_long_factors_raw is not None:
            panic_long_factors = [float(v) for v in panic_long_factors_raw]
        neg_ratios = [0.5, 0.6, 0.8]
        panic_lbs = [5, 10]
        panic_short_factors = [1.0, 0.5, 0.2, 0.0]
        panic_cutoffs_et = [None, 15, 16]
        # Optional stricter definition of "gap day": require |gap| >= threshold.
        panic_neg_gap_abs_pcts = [None, 0.01, 0.02]
        # Optional TR median "velocity": require TRmed(today)-TRmed(prev) >= delta, or over a wider lookback.
        panic_tr_delta_variants: list[tuple[float | None, int, str]] = [
            (None, 1, "trΔ=off"),
            (0.25, 1, "trΔ>=0.25@1d"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (0.75, 1, "trΔ>=0.75@1d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ]
        panic_total = (
            len(panic_trs)
            * len(neg_ratios)
            * len(panic_lbs)
            * len(panic_long_factors)
            * len(panic_short_factors)
            * len(panic_cutoffs_et)
            * len(panic_neg_gap_abs_pcts)
            * len(panic_tr_delta_variants)
        )

        # Risk-pop: TR% median + positive gap ratio.
        pop_trs = [7.0, 8.0, 9.0, 10.0, 12.0]
        pop_trs_over = _parse_tr_thresholds("--risk-overlays-riskpop-trs", args.risk_overlays_riskpop_trs)
        if pop_trs_over is not None:
            pop_trs = pop_trs_over
        pos_ratios = [0.5, 0.6, 0.8]
        pop_lbs = [5, 10]
        pop_long_factors = [0.6, 0.8, 1.0, 1.2, 1.5]
        pop_short_factors = [1.0, 0.5, 0.2, 0.0]
        pop_cutoffs_et = [None, 15]
        pop_modes = ["hygiene", "directional"]
        pop_pos_gap_abs_pcts = [None, 0.01, 0.02]
        pop_tr_delta_variants: list[tuple[float | None, int, str]] = [
            (None, 1, "trΔ=off"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ]

        pop_total = (
            len(pop_trs)
            * len(pos_ratios)
            * len(pop_lbs)
            * len(pop_long_factors)
            * len(pop_short_factors)
            * len(pop_cutoffs_et)
            * len(pop_modes)
            * len(pop_pos_gap_abs_pcts)
            * len(pop_tr_delta_variants)
        )
        if skip_pop:
            pop_total = 0
        total = riskoff_total + panic_total + pop_total

        def _iter_risk_overlay_specs():
            for tr_med in riskoff_trs:
                for lb in riskoff_lbs:
                    for mode in riskoff_modes:
                        for cutoff in riskoff_cutoffs_et:
                            overrides = {
                                "riskoff_tr5_med_pct": float(tr_med),
                                "riskoff_tr5_lookback_days": int(lb),
                                "riskoff_mode": str(mode),
                                "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                "riskpanic_tr5_med_pct": None,
                                "riskpanic_neg_gap_ratio_min": None,
                            }
                            cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                            note = f"riskoff TRmed{lb}>={tr_med:g} mode={mode} {cut_note}"
                            yield overrides, note, None

            for tr_med in panic_trs:
                for neg_ratio in neg_ratios:
                    for lb in panic_lbs:
                        for long_factor in panic_long_factors:
                            for short_factor in panic_short_factors:
                                for cutoff in panic_cutoffs_et:
                                    for abs_gap in panic_neg_gap_abs_pcts:
                                        for tr_delta_min, tr_delta_lb, tr_delta_note in panic_tr_delta_variants:
                                            overrides = {
                                                "riskoff_tr5_med_pct": None,
                                                "riskpanic_tr5_med_pct": float(tr_med),
                                                "riskpanic_neg_gap_ratio_min": float(neg_ratio),
                                                "riskpanic_neg_gap_abs_pct_min": (
                                                    float(abs_gap) if abs_gap is not None else None
                                                ),
                                                "riskpanic_lookback_days": int(lb),
                                                "riskpanic_tr5_med_delta_min_pct": (
                                                    float(tr_delta_min) if tr_delta_min is not None else None
                                                ),
                                                "riskpanic_tr5_med_delta_lookback_days": int(tr_delta_lb),
                                                "riskpanic_short_risk_mult_factor": float(short_factor),
                                                "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                            }
                                            if long_factor is not None:
                                                overrides["riskpanic_long_risk_mult_factor"] = float(long_factor)
                                            if (
                                                long_factor is not None
                                                and float(long_factor) < 1.0
                                                and tr_delta_min is not None
                                            ):
                                                overrides["riskpanic_long_scale_mode"] = "linear"
                                            cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                                            gap_note = "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                                            long_note = "" if long_factor is None else f" long_factor={long_factor:g}"
                                            scale_note = ""
                                            if overrides.get("riskpanic_long_scale_mode") == "linear":
                                                scale_note = " scale=lin"
                                            note = (
                                                f"riskpanic TRmed{lb}>={tr_med:g} neg_gap>={neg_ratio:g} {gap_note} "
                                                f"{tr_delta_note}{scale_note} short_factor={short_factor:g}{long_note} {cut_note}"
                                            )
                                            yield overrides, note, None

            if not skip_pop:
                for tr_med in pop_trs:
                    for pos_ratio in pos_ratios:
                        for lb in pop_lbs:
                            for long_factor in pop_long_factors:
                                for short_factor in pop_short_factors:
                                    for cutoff in pop_cutoffs_et:
                                        for mode in pop_modes:
                                            for abs_gap in pop_pos_gap_abs_pcts:
                                                for tr_delta_min, tr_delta_lb, tr_delta_note in pop_tr_delta_variants:
                                                    overrides = {
                                                        "riskoff_tr5_med_pct": None,
                                                        "riskpanic_tr5_med_pct": None,
                                                        "riskpanic_neg_gap_ratio_min": None,
                                                        "riskpop_tr5_med_pct": float(tr_med),
                                                        "riskpop_pos_gap_ratio_min": float(pos_ratio),
                                                        "riskpop_pos_gap_abs_pct_min": (
                                                            float(abs_gap) if abs_gap is not None else None
                                                        ),
                                                        "riskpop_lookback_days": int(lb),
                                                        "riskpop_tr5_med_delta_min_pct": (
                                                            float(tr_delta_min) if tr_delta_min is not None else None
                                                        ),
                                                        "riskpop_tr5_med_delta_lookback_days": int(tr_delta_lb),
                                                        "riskpop_long_risk_mult_factor": float(long_factor),
                                                        "riskpop_short_risk_mult_factor": float(short_factor),
                                                        "risk_entry_cutoff_hour_et": (
                                                            int(cutoff) if cutoff is not None else None
                                                        ),
                                                        "riskoff_mode": str(mode),
                                                    }
                                                    cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                                                    gap_note = "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                                                    note = (
                                                        f"riskpop TRmed{lb}>={tr_med:g} pos_gap>={pos_ratio:g} {gap_note} "
                                                        f"{tr_delta_note} mode={mode} long_factor={long_factor:g} "
                                                        f"short_factor={short_factor:g} {cut_note}"
                                                    )
                                                    yield overrides, note, None

        def _iter_risk_overlay_plan():
            for overrides, note, meta_item in _iter_risk_overlay_specs():
                f = _mk_filters(overrides=overrides)
                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                yield cfg, note, meta_item

        if args.risk_overlays_worker is not None:
            plan_all = list(_iter_risk_overlay_plan())
            if len(plan_all) != int(total):
                raise SystemExit(f"risk_overlays worker internal error: combos={len(plan_all)} expected={total}")
            _run_sharded_stage_worker(
                stage_label="risk_overlays",
                worker_raw=args.risk_overlays_worker,
                workers_raw=args.risk_overlays_workers,
                out_path_raw=str(args.risk_overlays_out or ""),
                out_flag_name="risk-overlays-out",
                plan_all=plan_all,
                bars=bars_sig,
                report_every=50,
                heartbeat_sec=50.0,
            )
            return

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []

        def _on_risk_overlay_row(_cfg: ConfigBundle, row: dict, _note: str) -> None:
            rows.append(row)

        tested_total = _run_stage_cfg_rows(
            stage_label="risk_overlays",
            total=total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=50,
            heartbeat_sec=50.0,
            on_row=_on_risk_overlay_row,
            serial_plan_builder=_iter_risk_overlay_plan,
            parallel_payloads_builder=lambda: _run_parallel_stage(
                axis_name="risk_overlays",
                stage_label="risk_overlays",
                total=total,
                jobs=int(jobs),
                worker_tmp_prefix="tradebot_risk_overlays_",
                worker_tag="ro",
                out_prefix="risk_overlays_out",
                worker_flag="--risk-overlays-worker",
                workers_flag="--risk-overlays-workers",
                out_flag="--risk-overlays-out",
                strip_flags_with_values=(
                    "--risk-overlays-worker",
                    "--risk-overlays-workers",
                    "--risk-overlays-out",
                    "--risk-overlays-run-min-trades",
                ),
                run_min_trades_flag="--risk-overlays-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture risk_overlays worker stdout.",
                failure_label="risk_overlays worker",
                missing_label="risk_overlays",
                invalid_label="risk_overlays",
            ),
            parallel_default_note="risk_overlays",
            parallel_dedupe_by_milestone_key=False,
            record_milestones=True,
        )
        if jobs > 1 and total > 0:
            run_calls_total += int(tested_total)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="TR% risk overlay sweep (riskoff + riskpanic + riskpop)", top_n=int(args.top))

    def _sweep_loosen() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for close_eod in (False, True):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    spot_close_eod=bool(close_eod),
                ),
            )
            row = _run_cfg(
                cfg=cfg, bars=bars_sig
            )
            if not row:
                continue
            note = f"close_eod={int(close_eod)}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Loosenings sweep (single-position + EOD exit)", top_n=int(args.top))

    def _sweep_loosen_atr() -> None:
        """Interaction hunt: close_eod × ATR exits under single-position parity."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Keep the grid tight around the post-fix high-PnL neighborhood.
        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.65, 0.7, 0.75, 0.8]
        sl_mults = [1.2, 1.4, 1.6, 1.8, 2.0]
        close_eod_vals = [False, True]

        rows: list[dict] = []
        for close_eod in close_eod_vals:
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                spot_close_eod=bool(close_eod),
                                spot_exit_mode="atr",
                                spot_atr_period=int(atr_p),
                                spot_pt_atr_mult=float(pt_m),
                                spot_sl_atr_mult=float(sl_m),
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"close_eod={int(close_eod)} | ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Loosen × ATR joint sweep (single-position × exits)", top_n=int(args.top))

    def _sweep_tick() -> None:
        """Permission layer: Raschke-style $TICK width gate (daily, RTH only)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "tick=off (base)"
            _record_milestone(base, base_row, "tick=off (base)")

        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]
        policies = ["allow", "block"]
        dir_policies = ["both", "wide_only"]
        regime2_variants: list[tuple[dict, str]] = []
        base_r2_mode = str(getattr(base.strategy, "regime2_mode", "off") or "off").strip().lower()
        if base_r2_mode != "off":
            regime2_variants.append(
                (
                    {
                        "regime2_mode": str(getattr(base.strategy, "regime2_mode") or "off"),
                        "regime2_bar_size": getattr(base.strategy, "regime2_bar_size", None),
                        "regime2_supertrend_atr_period": getattr(base.strategy, "regime2_supertrend_atr_period", None),
                        "regime2_supertrend_multiplier": getattr(base.strategy, "regime2_supertrend_multiplier", None),
                        "regime2_supertrend_source": getattr(base.strategy, "regime2_supertrend_source", None),
                    },
                    "r2=base",
                )
            )
        regime2_variants += [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off"),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:3,0.25,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 5,
                    "regime2_supertrend_multiplier": 0.2,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:5,0.2,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "1 day",
                    "regime2_supertrend_atr_period": 7,
                    "regime2_supertrend_multiplier": 0.4,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(1d:7,0.4,close)",
            ),
        ]

        rows: list[dict] = []
        for dir_policy in dir_policies:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                for r2_over, r2_note in regime2_variants:
                                    strat = base.strategy
                                    cfg = replace(
                                        base,
                                        strategy=replace(
                                            strat,
                                            tick_gate_mode="raschke",
                                            tick_gate_symbol="TICK-AMEX",
                                            tick_gate_exchange="AMEX",
                                            tick_neutral_policy=str(policy),
                                            tick_direction_policy=str(dir_policy),
                                            tick_band_ma_period=10,
                                            tick_width_z_lookback=int(lookback),
                                            tick_width_z_enter=float(z_enter),
                                            tick_width_z_exit=float(z_exit),
                                            tick_width_slope_lookback=int(slope_lb),
                                            regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                                            regime2_bar_size=r2_over.get("regime2_bar_size"),
                                            regime2_supertrend_atr_period=int(
                                                r2_over.get("regime2_supertrend_atr_period") or 10
                                            ),
                                            regime2_supertrend_multiplier=float(
                                                r2_over.get("regime2_supertrend_multiplier") or 3.0
                                            ),
                                            regime2_supertrend_source=str(
                                                r2_over.get("regime2_supertrend_source") or "hl2"
                                            ),
                                        ),
                                    )
                                    row = _run_cfg(cfg=cfg)
                                    if not row:
                                        continue
                                    note = (
                                        f"tick=raschke dir={dir_policy} policy={policy} z_in={z_enter} "
                                        f"z_out={z_exit} slope={slope_lb} lb={lookback} {r2_note}"
                                    )
                                    row["note"] = note
                                    _record_milestone(cfg, row, note)
                                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick gate sweep ($TICK width)", top_n=int(args.top))

    def _sweep_frontier() -> None:
        """Summarize the current milestones set as a multi-objective frontier."""
        groups = milestones.get("groups", []) if isinstance(milestones, dict) else []
        rows: list[dict] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            entries = group.get("entries") or []
            if not entries or not isinstance(entries, list):
                continue
            entry = entries[0]
            if not isinstance(entry, dict):
                continue
            strat = entry.get("strategy") or {}
            metrics = entry.get("metrics") or {}
            if not isinstance(strat, dict) or not isinstance(metrics, dict):
                continue
            if str(strat.get("instrument", "spot") or "spot").strip().lower() != "spot":
                continue
            if str(strat.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
                continue
            if bool(strat.get("signal_use_rth")) != bool(use_rth):
                continue
            if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
                continue
            try:
                trades = int(metrics.get("trades") or 0)
                win = float(metrics.get("win_rate") or 0.0)
                pnl = float(metrics.get("pnl") or 0.0)
                dd = float(metrics.get("max_drawdown") or 0.0)
                pnl_dd = metrics.get("pnl_over_dd")
                pnl_over_dd = float(pnl_dd) if pnl_dd is not None else (pnl / dd if dd > 0 else None)
            except (TypeError, ValueError):
                continue
            note = str(group.get("name") or "").strip() or "milestone"
            rows.append(
                {
                    "trades": trades,
                    "win_rate": win,
                    "pnl": pnl,
                    "dd": dd,
                    "pnl_over_dd": pnl_over_dd,
                    "note": note,
                }
            )

        if not rows:
            print("No matching spot milestones found for this bar_size/symbol.")
            return

        _print_leaderboards(rows, title="Milestones frontier (current presets)", top_n=int(args.top))

        print("")
        print("Frontier by win-rate constraint (best pnl):")
        for thr in (0.55, 0.58, 0.60, 0.62, 0.65):
            eligible = [r for r in rows if int(r.get("trades") or 0) >= int(run_min_trades) and float(r.get("win_rate") or 0.0) >= thr]
            if not eligible:
                continue
            best = max(eligible, key=lambda r: float(r.get("pnl") or float("-inf")))
            print(
                f"- win>={thr:.2f}: pnl={best['pnl']:.1f} pnl/dd={(best['pnl_over_dd'] or 0):.2f} "
                f"win={best['win_rate']*100:.1f}% tr={best['trades']} note={best.get('note')}"
            )

    def _sweep_combo_fast() -> None:
        """A constrained multi-axis sweep to find "corner" winners (fast bounded funnel).

        Keep this computationally bounded and reproducible. The intent is to combine
        the highest-leverage levers we’ve found so far:
        - direction layer interactions (EMA preset + entry mode)
        - regime sensitivity (Supertrend timeframe + params)
        - exits (pct vs ATR), including the PT<1.0 ATR pocket
        - loosenings (single-position parity + EOD close)
        - optional regime2 confirm (small curated set)
        - a small set of quality gates (spread/slope/TOD/rv/exit-time/tick)
        """
        nonlocal run_calls_total
        bars_sig = _bars_cached(signal_bar_size)

        # Stage 2 variants are constant and are also used by the stage2 worker/sharding mode.
        exit_variants: list[tuple[dict, str]] = []
        for pt, sl in (
            (0.005, 0.02),
            (0.005, 0.03),
            (0.01, 0.03),
            (0.015, 0.03),
            # Higher RR pocket (PT > SL): helps when stop-first intrabar tie-break punishes low-RR setups.
            (0.02, 0.015),
            (0.03, 0.015),
            # Bigger PT/SL pocket: trend systems often need a wider profit capture window.
            (0.05, 0.03),
            (0.08, 0.04),
        ):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": float(pt),
                        "spot_stop_loss_pct": float(sl),
                        "spot_atr_period": 14,
                        "spot_pt_atr_mult": 1.5,
                        "spot_sl_atr_mult": 1.0,
                    },
                    f"PT={pt:.3f} SL={sl:.3f}",
                )
            )
        # Stop-only (no PT): "exit on next cross / regime flip" families.
        for sl in (0.03, 0.05):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": float(sl),
                        "spot_atr_period": 14,
                        "spot_pt_atr_mult": 1.5,
                        "spot_sl_atr_mult": 1.0,
                    },
                    f"PT=off SL={sl:.3f}",
                )
            )
        for atr_p, pt_m, sl_m in (
            # Risk-adjusted champ neighborhood.
            (7, 1.0, 1.0),
            (7, 1.0, 1.5),
            (7, 1.12, 1.5),
            # Net-PnL pocket (PTx<1.0).
            (10, 0.80, 1.80),
            (10, 0.90, 1.80),
            (14, 0.70, 1.60),
            (14, 0.75, 1.60),
            (14, 0.80, 1.60),
            (21, 0.65, 1.60),
            (21, 0.70, 1.80),
            # Higher RR pocket (PTx > SLx): try to counter stop-first intrabar ambiguity.
            (14, 2.00, 1.00),
            (21, 2.00, 1.00),
        ):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "atr",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": None,
                        "spot_atr_period": int(atr_p),
                        "spot_pt_atr_mult": float(pt_m),
                        "spot_sl_atr_mult": float(sl_m),
                    },
                    f"ATR({atr_p}) PTx{pt_m} SLx{sl_m}",
                )
            )

        # Keep this small; in spot mode this axis only toggles close_eod.
        loosen_variants: list[tuple[bool, str]] = [
            (False, "close_eod=0"),
            (True, "close_eod=1"),
        ]

        hold_vals = (0, 4)

        regime2_variants: list[tuple[dict, str]] = [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "no_r2"),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
                "ST2(4h:3,0.25,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 5,
                    "regime2_supertrend_multiplier": 0.2,
                    "regime2_supertrend_source": "close",
                },
                "ST2(4h:5,0.2,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "1 day",
                    "regime2_supertrend_atr_period": 7,
                    "regime2_supertrend_multiplier": 0.4,
                    "regime2_supertrend_source": "close",
                },
                "ST2(1d:7,0.4,close)",
            ),
        ]

        def _mk_stage2_cfg(
            base_cfg: ConfigBundle,
            base_note: str,
            *,
            exit_over: dict,
            exit_note: str,
            hold: int,
            close_eod: bool,
            loose_note: str,
            r2_over: dict,
            r2_note: str,
        ) -> tuple[ConfigBundle, str]:
            strat = base_cfg.strategy
            cfg = replace(
                base_cfg,
                strategy=replace(
                    strat,
                    spot_exit_mode=str(exit_over["spot_exit_mode"]),
                    spot_profit_target_pct=exit_over["spot_profit_target_pct"],
                    spot_stop_loss_pct=exit_over["spot_stop_loss_pct"],
                    spot_atr_period=int(exit_over["spot_atr_period"]),
                    spot_pt_atr_mult=float(exit_over["spot_pt_atr_mult"]),
                    spot_sl_atr_mult=float(exit_over["spot_sl_atr_mult"]),
                    flip_exit_min_hold_bars=int(hold),
                    spot_close_eod=bool(close_eod),
                    regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                    regime2_bar_size=r2_over.get("regime2_bar_size"),
                    regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                    regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                    regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                ),
            )
            note = f"{base_note} | {exit_note} | hold={hold} | {loose_note} | {r2_note}"
            return cfg, note

        def _build_stage2_plan(shortlist_local: list[tuple[ConfigBundle, str]]) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, base_note) in enumerate(shortlist_local):
                for exit_idx, (exit_over, exit_note) in enumerate(exit_variants):
                    for hold in hold_vals:
                        for loosen_idx, (close_eod, loose_note) in enumerate(loosen_variants):
                            for r2_idx, (r2_over, r2_note) in enumerate(regime2_variants):
                                cfg, note = _mk_stage2_cfg(
                                    base_cfg,
                                    base_note,
                                    exit_over=exit_over,
                                    exit_note=exit_note,
                                    hold=int(hold),
                                    close_eod=bool(close_eod),
                                    loose_note=loose_note,
                                    r2_over=r2_over,
                                    r2_note=r2_note,
                                )
                                plan.append(
                                    (
                                        cfg,
                                        note,
                                        {
                                            "base_idx": int(base_idx),
                                            "exit_idx": int(exit_idx),
                                            "hold": int(hold),
                                            "loosen_idx": int(loosen_idx),
                                            "r2_idx": int(r2_idx),
                                        },
                                    )
                                )
            return plan

        if args.combo_fast_stage2:
            payload_path = Path(str(args.combo_fast_stage2))
            payload_decoded = _load_worker_stage_payload(
                schema_name="combo_fast_stage2",
                payload_path=payload_path,
            )
            shortlist_local = [
                (cfg, note)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]

            stage2_plan_all = _build_stage2_plan(shortlist_local)
            _run_sharded_stage_worker(
                stage_label="combo_fast stage2",
                worker_raw=args.combo_fast_worker,
                workers_raw=args.combo_fast_workers,
                out_path_raw=str(args.combo_fast_out or ""),
                out_flag_name="combo-fast-out",
                plan_all=stage2_plan_all,
                bars=bars_sig,
                report_every=100,
                heartbeat_sec=0.0,
            )
            return

        # Stage 3 variants are constant and are also used by the stage3 worker/sharding mode.
        tick_variants: list[tuple[dict, str]] = [
            ({"tick_gate_mode": "off"}, "tick=off"),
            (
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
                "tick=raschke(wide_only block z=1.0/0.5 slope=3 lb=252)",
            ),
        ]

        quality_variants: list[tuple[float | None, float | None, float | None, str]] = [
            # (spread_min, spread_min_down, slope_min, note)
            (None, None, None, "qual=off"),
            (0.003, None, None, "spread>=0.003"),
            (0.003, 0.006, None, "spread>=0.003 down>=0.006"),
            (0.003, 0.008, None, "spread>=0.003 down>=0.008"),
            (0.003, 0.010, None, "spread>=0.003 down>=0.010"),
            (0.003, 0.015, None, "spread>=0.003 down>=0.015"),
            (0.003, 0.030, None, "spread>=0.003 down>=0.030"),
            (0.003, 0.050, None, "spread>=0.003 down>=0.050"),
            (0.005, None, None, "spread>=0.005"),
            (0.005, 0.010, None, "spread>=0.005 down>=0.010"),
            (0.005, 0.012, None, "spread>=0.005 down>=0.012"),
            (0.005, 0.015, None, "spread>=0.005 down>=0.015"),
            (0.005, 0.030, None, "spread>=0.005 down>=0.030"),
            (0.005, 0.050, None, "spread>=0.005 down>=0.050"),
            (0.005, 0.010, 0.01, "spread>=0.005 down>=0.010 slope>=0.01"),
        ]

        rv_variants: list[tuple[float | None, float | None, str]] = [
            (None, None, "rv=off"),
            (0.25, 0.8, "rv=0.25..0.80"),
        ]

        exit_time_variants: list[tuple[str | None, str]] = [
            (None, "exit_time=off"),
            ("17:00", "exit_time=17:00 ET"),
        ]

        tod_variants: list[tuple[int | None, int | None, int, int, str]] = [
            (None, None, 0, 0, "tod=any"),
            (18, 4, 0, 0, "tod=18-04 ET"),
            (18, 4, 1, 2, "tod=18-04 ET (skip=1 cd=2)"),
            (10, 15, 0, 0, "tod=10-15 ET"),
        ]

        def _mk_stage3_cfg(
            base_cfg: ConfigBundle,
            *,
            tick_over: dict,
            spread_min: float | None,
            spread_min_down: float | None,
            slope_min: float | None,
            rv_min: float | None,
            rv_max: float | None,
            exit_time: str | None,
            tod_s: int | None,
            tod_e: int | None,
            skip: int,
            cooldown: int,
        ) -> ConfigBundle:
            f = _mk_filters(
                rv_min=rv_min,
                rv_max=rv_max,
                ema_spread_min_pct=spread_min,
                ema_spread_min_pct_down=spread_min_down,
                ema_slope_min_pct=slope_min,
                cooldown_bars=int(cooldown),
                skip_first_bars=int(skip),
                entry_start_hour_et=tod_s,
                entry_end_hour_et=tod_e,
            )
            return replace(
                base_cfg,
                strategy=replace(
                    base_cfg.strategy,
                    filters=f,
                    spot_exit_time_et=exit_time,
                    tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                    tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                    tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                    tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                    tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                    tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                    tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                    tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                    tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                    tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
                ),
            )

        def _build_stage3_plan(bases_local: list[tuple[ConfigBundle, str]]) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, base_note) in enumerate(bases_local):
                for tick_idx, (tick_over, tick_note) in enumerate(tick_variants):
                    for qual_idx, (spread_min, spread_min_down, slope_min, qual_note) in enumerate(quality_variants):
                        for rv_idx, (rv_min, rv_max, rv_note) in enumerate(rv_variants):
                            for exit_time_idx, (exit_time, exit_time_note) in enumerate(exit_time_variants):
                                for tod_idx, (tod_s, tod_e, skip, cooldown, tod_note) in enumerate(tod_variants):
                                    cfg = _mk_stage3_cfg(
                                        base_cfg,
                                        tick_over=tick_over,
                                        spread_min=spread_min,
                                        spread_min_down=spread_min_down,
                                        slope_min=slope_min,
                                        rv_min=rv_min,
                                        rv_max=rv_max,
                                        exit_time=exit_time,
                                        tod_s=tod_s,
                                        tod_e=tod_e,
                                        skip=int(skip),
                                        cooldown=int(cooldown),
                                    )
                                    note = (
                                        f"{base_note} | {tick_note} | {qual_note} | "
                                        f"{rv_note} | {exit_time_note} | {tod_note}"
                                    )
                                    plan.append(
                                        (
                                            cfg,
                                            note,
                                            {
                                                "base_idx": int(base_idx),
                                                "tick_idx": int(tick_idx),
                                                "qual_idx": int(qual_idx),
                                                "rv_idx": int(rv_idx),
                                                "exit_time_idx": int(exit_time_idx),
                                                "tod_idx": int(tod_idx),
                                            },
                                        )
                                    )
            return plan

        if args.combo_fast_stage3:
            payload_path = Path(str(args.combo_fast_stage3))
            payload_decoded = _load_worker_stage_payload(
                schema_name="combo_fast_stage3",
                payload_path=payload_path,
            )
            bases_local = [
                (cfg, note)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]

            stage3_plan_all = _build_stage3_plan(bases_local)
            _run_sharded_stage_worker(
                stage_label="combo_fast stage3",
                worker_raw=args.combo_fast_worker,
                workers_raw=args.combo_fast_workers,
                out_path_raw=str(args.combo_fast_out or ""),
                out_flag_name="combo-fast-out",
                plan_all=stage3_plan_all,
                bars=bars_sig,
                report_every=200,
                heartbeat_sec=0.0,
            )
            return

        # Stage 1: direction × regime sensitivity (bounded) and keep a small diverse shortlist.
        stage1: list[tuple[ConfigBundle, dict, str]] = []
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        # Ensure stage1 isn't silently gated by whatever the current milestone base uses.
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                filters=None,
                tick_gate_mode="off",
                spot_exit_time_et=None,
            ),
        )

        direction_variants: list[tuple[str, str, int, str]] = []
        base_preset = str(base.strategy.ema_preset or "").strip()
        base_mode = str(base.strategy.ema_entry_mode or "trend").strip().lower()
        base_confirm = int(base.strategy.entry_confirm_bars or 0)
        if base_preset and base_mode in ("cross", "trend"):
            direction_variants.append((base_preset, base_mode, base_confirm, f"ema={base_preset} {base_mode}"))

        for preset, mode in (
            ("2/4", "cross"),
            ("3/7", "cross"),
            ("3/7", "trend"),
            ("4/9", "cross"),
            ("4/9", "trend"),
            ("5/10", "cross"),
            ("9/21", "cross"),
            ("9/21", "trend"),
        ):
            direction_variants.append((preset, mode, 0, f"ema={preset} {mode}"))

        seen_dir: set[tuple[str, str, int]] = set()
        direction_variants = [
            v
            for v in direction_variants
            if (v[0], v[1], v[2]) not in seen_dir and not seen_dir.add((v[0], v[1], v[2]))
        ]

        regime_bar_sizes = ["4 hours", "1 day"]
        atr_periods = [3, 7, 10, 14, 21]
        multipliers = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        sources = ["close", "hl2"]
        stage1_exit_variants: list[tuple[dict, str]] = [
            (
                {"spot_exit_mode": "pct", "spot_profit_target_pct": 0.015, "spot_stop_loss_pct": 0.03},
                "PT=0.015 SL=0.030",
            ),
            (
                {"spot_exit_mode": "pct", "spot_profit_target_pct": None, "spot_stop_loss_pct": 0.03},
                "PT=off SL=0.030",
            ),
        ]

        def _mk_stage1_cfg(
            *,
            ema_preset: str,
            entry_mode: str,
            confirm: int,
            rbar: str,
            atr_p: int,
            mult: float,
            src: str,
            exit_over: dict,
            dir_note: str,
            exit_note: str,
        ) -> tuple[ConfigBundle, str]:
            cfg = replace(
                base,
                strategy=replace(
                    base.strategy,
                    entry_signal="ema",
                    ema_preset=str(ema_preset),
                    ema_entry_mode=str(entry_mode),
                    entry_confirm_bars=int(confirm),
                    regime_mode="supertrend",
                    regime_bar_size=rbar,
                    supertrend_atr_period=int(atr_p),
                    supertrend_multiplier=float(mult),
                    supertrend_source=str(src),
                    regime2_mode="off",
                    spot_exit_mode=str(exit_over["spot_exit_mode"]),
                    spot_profit_target_pct=exit_over.get("spot_profit_target_pct"),
                    spot_stop_loss_pct=exit_over.get("spot_stop_loss_pct"),
                ),
            )
            note = f"{dir_note} c={confirm} | ST({atr_p},{mult},{src}) @{rbar} | {exit_note}"
            return cfg, note

        def _build_stage1_plan() -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for dir_idx, (ema_preset, entry_mode, confirm, dir_note) in enumerate(direction_variants):
                for rbar_idx, rbar in enumerate(regime_bar_sizes):
                    for atr_idx, atr_p in enumerate(atr_periods):
                        for mult_idx, mult in enumerate(multipliers):
                            for src_idx, src in enumerate(sources):
                                for exit_idx, (exit_over, exit_note) in enumerate(stage1_exit_variants):
                                    cfg, note = _mk_stage1_cfg(
                                        ema_preset=str(ema_preset),
                                        entry_mode=str(entry_mode),
                                        confirm=int(confirm),
                                        rbar=str(rbar),
                                        atr_p=int(atr_p),
                                        mult=float(mult),
                                        src=str(src),
                                        exit_over=exit_over,
                                        dir_note=str(dir_note),
                                        exit_note=str(exit_note),
                                    )
                                    plan.append(
                                        (
                                            cfg,
                                            note,
                                            {
                                                "dir_idx": int(dir_idx),
                                                "rbar_idx": int(rbar_idx),
                                                "atr_idx": int(atr_idx),
                                                "mult_idx": int(mult_idx),
                                                "src_idx": int(src_idx),
                                                "exit_idx": int(exit_idx),
                                            },
                                        )
                                    )
            return plan

        stage1_plan_all = _build_stage1_plan()
        stage1_total = len(stage1_plan_all)
        if args.combo_fast_stage1:
            _run_sharded_stage_worker(
                stage_label="combo_fast stage1",
                worker_raw=args.combo_fast_worker,
                workers_raw=args.combo_fast_workers,
                out_path_raw=str(args.combo_fast_out or ""),
                out_flag_name="combo-fast-out",
                plan_all=stage1_plan_all,
                bars=bars_sig,
                report_every=200,
                heartbeat_sec=0.0,
            )
            return

        stage1_tested = 0
        report_every_stage1 = 200
        print(f"combo_fast sweep: stage1 total={stage1_total} (progress every {report_every_stage1})", flush=True)
        stage1_tested = _run_stage_cfg_rows(
            stage_label="combo_fast sweep: stage1",
            total=stage1_total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=report_every_stage1,
            on_row=lambda cfg, row, note: stage1.append((cfg, row, note)),
            serial_plan=stage1_plan_all,
            parallel_payloads_builder=lambda: _run_parallel_stage(
                axis_name="combo_fast",
                stage_label="combo_fast stage1",
                total=stage1_total,
                jobs=int(jobs),
                worker_tmp_prefix="tradebot_combo_fast1_",
                worker_tag="cf1",
                out_prefix="stage1_out",
                stage_flag="--combo-fast-stage1",
                stage_value="1",
                worker_flag="--combo-fast-worker",
                workers_flag="--combo-fast-workers",
                out_flag="--combo-fast-out",
                strip_flags_with_values=(
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
                run_min_trades_flag="--combo-fast-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture combo_fast stage1 worker stdout.",
                failure_label="combo_fast stage1 worker",
                missing_label="combo_fast stage1",
                invalid_label="combo_fast stage1",
            ),
            parallel_default_note="combo_fast stage1",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=False,
        )
        if jobs > 1 and stage1_total > 0:
            run_calls_total += int(stage1_tested)

        shortlist = _rank_cfg_rows(
            stage1,
            scorers=[(_score_row_pnl_dd, 15), (_score_row_pnl, 7)],
        )
        print("")
        print(f"combo_fast sweep: shortlist regimes={len(shortlist)} (from stage1={len(stage1)})")

        # Stage 2: for each shortlisted regime, sweep exits + loosenings, and (optionally) a small regime2 set.
        stage2: list[tuple[ConfigBundle, dict, str]] = []
        report_every = 200
        stage2_plan_all = _build_stage2_plan([(cfg, note) for cfg, _row, note in shortlist])
        stage2_total = len(stage2_plan_all)
        print(f"combo_fast sweep: stage2 total={stage2_total} (progress every {report_every})", flush=True)
        tested = _run_stage_cfg_rows(
            stage_label="combo_fast sweep: stage2",
            total=stage2_total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=report_every,
            on_row=lambda cfg, row, note: stage2.append((cfg, row, note)),
            serial_plan=stage2_plan_all,
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name="combo_fast",
                stage_label="combo_fast stage2",
                total=stage2_total,
                jobs=int(jobs),
                payload={
                    "shortlist": [
                        _encode_cfg_payload(base_cfg, note=base_note, note_key="base_note")
                        for base_cfg, _row, base_note in shortlist
                    ],
                },
                payload_filename="stage2_payload.json",
                temp_prefix="tradebot_combo_fast_",
                worker_tmp_prefix="tradebot_combo_fast2_",
                worker_tag="cf2",
                out_prefix="stage2_out",
                stage_flag="--combo-fast-stage2",
                worker_flag="--combo-fast-worker",
                workers_flag="--combo-fast-workers",
                out_flag="--combo-fast-out",
                strip_flags_with_values=(
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
                run_min_trades_flag="--combo-fast-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture combo_fast stage2 worker stdout.",
                failure_label="combo_fast stage2 worker",
                missing_label="combo_fast stage2",
                invalid_label="combo_fast stage2",
            ),
            parallel_default_note="combo_fast stage2",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        if jobs > 1 and stage2_total > 0:
            run_calls_total += int(tested)

        print(f"combo_fast sweep: stage2 tested={tested} kept={len(stage2)} (min_trades={run_min_trades})")

        # Stage 3: apply a small set of quality gates on the top stage2 candidates.
        top_stage2 = _rank_cfg_rows(
            stage2,
            scorers=[(_score_row_pnl_dd, 15), (_score_row_pnl, 7)],
        )

        stage3_plan_all = _build_stage3_plan([(cfg, base_note) for cfg, _row, base_note in top_stage2])
        stage3_total = len(stage3_plan_all)
        stage3_tested = 0
        report_every_stage3 = 200
        print(f"combo_fast sweep: stage3 total={stage3_total} (progress every {report_every_stage3})", flush=True)

        stage3: list[dict] = []
        stage3_tested = _run_stage_cfg_rows(
            stage_label="combo_fast sweep: stage3",
            total=stage3_total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=report_every_stage3,
            on_row=lambda _cfg, row, _note: stage3.append(row),
            serial_plan=stage3_plan_all,
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name="combo_fast",
                stage_label="combo_fast stage3",
                total=stage3_total,
                jobs=int(jobs),
                payload={
                    "bases": [
                        _encode_cfg_payload(base_cfg, note=base_note, note_key="base_note")
                        for base_cfg, _row, base_note in top_stage2
                    ],
                },
                payload_filename="stage3_payload.json",
                temp_prefix="tradebot_combo_fast3_",
                worker_tmp_prefix="tradebot_combo_fast3_",
                worker_tag="cf3",
                out_prefix="stage3_out",
                stage_flag="--combo-fast-stage3",
                worker_flag="--combo-fast-worker",
                workers_flag="--combo-fast-workers",
                out_flag="--combo-fast-out",
                strip_flags_with_values=(
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
                run_min_trades_flag="--combo-fast-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture combo_fast stage3 worker stdout.",
                failure_label="combo_fast stage3 worker",
                missing_label="combo_fast stage3",
                invalid_label="combo_fast stage3",
            ),
            parallel_default_note="combo_fast stage3",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        if jobs > 1 and stage3_total > 0:
            run_calls_total += int(stage3_tested)

        _print_leaderboards(stage3, title="combo_fast sweep (multi-axis, constrained)", top_n=int(args.top))

    def _sweep_combo_full() -> None:
        """An extremely comprehensive run that executes the full spot sweep suite.

        This intentionally leans toward "do everything we can" rather than a single funnel:
        it runs the one-axis sweeps, the named joint sweeps, and then the bounded
        `combo_fast` funnel. Use this when you want coverage, not turnaround time.
        """
        nonlocal milestones_written
        if offline:
            # ORB sweeps always use 15m bars; preflight early so we fail fast rather than hours in.
            _require_offline_cache_or_die(
                cache_dir=cache_dir,
                symbol=symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size="15 mins",
                use_rth=use_rth,
            )
            # Tick sweeps use daily $TICK bars (RTH only). Allow either AMEX or NYSE cache.
            tick_warm_start = start_dt - timedelta(days=400)
            tick_ok = False
            for tick_sym in ("TICK-AMEX", "TICK-NYSE"):
                try:
                    _require_offline_cache_or_die(
                        cache_dir=cache_dir,
                        symbol=tick_sym,
                        start_dt=tick_warm_start,
                        end_dt=end_dt,
                        bar_size="1 day",
                        use_rth=True,
                    )
                    tick_ok = True
                    break
                except SystemExit:
                    continue
            if not tick_ok:
                raise SystemExit(
                    "combo_full requires cached daily $TICK bars when running with --offline "
                    "(expected under db/TICK-AMEX or db/TICK-NYSE). Run once without --offline to fetch, "
                    "or skip tick-based sweeps by running --axis combo_fast instead."
                )

        print("")
        print("=== combo_full: running full sweep suite (very slow) ===")
        print("")

        axis_plan = list(_axis_mode_plan(mode="combo_full", include_seeded=False))
        seed_path_raw = str(getattr(args, "seed_milestones", "") or "").strip()
        if seed_path_raw:
            seed_path = Path(seed_path_raw)
            if seed_path.exists():
                axis_plan = list(_axis_mode_plan(mode="combo_full", include_seeded=True))
            else:
                print(
                    f"SKIP seeded_refine bundle ({','.join(_SEEDED_REFINEMENT_MEMBER_AXES)}): "
                    f"--seed-milestones not found ({seed_path})",
                    flush=True,
                )
        if _run_axis_plan_parallel_if_requested(
            axis_plan=axis_plan,
            jobs_req=int(jobs),
            label="combo_full parallel",
            tmp_prefix="tradebot_combo_full_",
            offline_error="--jobs>1 for combo_full requires --offline (avoid parallel IBKR sessions).",
        ):
            return

        _run_axis_plan_serial(axis_plan, timed=True)

    def _sweep_champ_refine() -> None:
        """Seeded, champ-focused refinement around a top-K candidate pool.

        Intent:
        - Avoid the full `combo_full` suite when you already have a promising pool.
        - Run only the high-leverage "champ discovery" levers we've learned:
          short asymmetry (`spot_short_risk_mult`), TOD/permission micro, signed slope,
          and a small shock + TR overlay pocket.

        This is intentionally bounded and should finish in a reasonable overnight window.
        """
        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag="champ_refine",
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            min_trades=int(run_min_trades),
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        family_winners = _seed_best_by_family(
            candidates,
            family_key_fn=lambda item: (
                str((item.get("strategy") or {}).get("ema_preset") or ""),
                str((item.get("strategy") or {}).get("ema_entry_mode") or ""),
                str((item.get("strategy") or {}).get("regime_mode") or ""),
                str((item.get("strategy") or {}).get("regime_bar_size") or ""),
                str((item.get("strategy") or {}).get("spot_exit_mode") or ""),
            ),
            scorer="pnl_dd",
        )
        seeds = _seed_select_candidates(
            candidates,
            seed_top=seed_top,
            policy="champ_refine",
        )

        print("")
        print("=== champ_refine: seeded refinement (bounded) ===")
        print(f"- seeds_in_file={len(candidates)} families={len(family_winners)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        tested_total = 0
        t0_all = pytime.perf_counter()
        heartbeat_sec = 50.0
        last_progress = float(t0_all)

        short_grid_base = [1.0, 0.2, 0.05, 0.02, 0.01, 0.0]
        is_slv = str(symbol).strip().upper() == "SLV"

        # Joint permission micro grid (cross-product) around the CURRENT champ.
        #
        # This covers interaction edges that "one-axis-at-a-time" sweeps can miss, and it
        # explicitly includes the tiny-delta winners we already observed:
        # - ema_spread_min_pct=0.0025 (better 10y+2y)
        # - ema_slope_min_pct=0.02 (better 10y+1y)
        # - ema_spread_min_pct_down=0.06 (better 10y+2y but slightly hurts 1y)
        perm_variants: list[tuple[dict[str, object], str]] = [({}, "perm=seed")]
        if is_slv:
            # Keep stage3a bounded for 10y runs (speed), but still probe a few distinct permission regimes.
            for spread, slope, down, note in (
                (0.0015, 0.01, 0.02, "perm=loose (0.0015/0.01/0.02)"),
                (0.0015, 0.03, 0.02, "perm=loose_slope (0.0015/0.03/0.02)"),
                (0.0030, 0.03, 0.04, "perm=mid (0.003/0.03/0.04)"),
                (0.0060, 0.03, 0.08, "perm=spready (0.006/0.03/0.08)"),
                (0.0030, 0.06, 0.08, "perm=tight_slope (0.003/0.06/0.08)"),
                (0.0060, 0.06, 0.08, "perm=tight (0.006/0.06/0.08)"),
            ):
                perm_variants.append(
                    (
                        {
                            "ema_spread_min_pct": float(spread),
                            "ema_slope_min_pct": float(slope),
                            "ema_spread_min_pct_down": float(down),
                        },
                        str(note),
                    )
                )
        else:
            spread_vals = [0.0025, 0.003, 0.004]
            slope_vals = [0.02, 0.03, 0.04]
            down_vals = [0.04, 0.05, 0.06]
            for spread in spread_vals:
                for slope in slope_vals:
                    for down in down_vals:
                        perm_variants.append(
                            (
                                {
                                    "ema_spread_min_pct": float(spread),
                                    "ema_slope_min_pct": float(slope),
                                    "ema_spread_min_pct_down": float(down),
                                },
                                f"perm spread={spread:g} slope={slope:g} down={down:g}",
                            )
                        )
        signed_slope_variants: list[tuple[dict[str, object], str]] = (
            [
                ({}, "sslope=off"),
            ]
            if is_slv
            else [
                ({}, "sslope=off"),
                (
                    {"ema_slope_signed_min_pct_up": 0.003, "ema_slope_signed_min_pct_down": 0.003},
                    "sslope=0.003/0.003",
                ),
                (
                    {"ema_slope_signed_min_pct_up": 0.005, "ema_slope_signed_min_pct_down": 0.005},
                    "sslope=0.005/0.005",
                ),
                (
                    {"ema_slope_signed_min_pct_up": 0.003, "ema_slope_signed_min_pct_down": 0.006},
                    "sslope=0.003/0.006",
                ),
            ]
        )
        if is_slv:
            # SLV legacy champs were discovered with full RTH-only cadence.
            #
            # For FULL24 runs, we explicitly probe a small "all-hours vs RTH" pocket so
            # a 24/5 seed can compete fairly (and so we can find an actually-all-hours champ).
            tod_variants: list[tuple[int | None, int | None, int, int, str]]
            if use_rth:
                tod_variants = [(9, 16, 0, 0, "tod=09-16")]
            else:
                tod_variants = [
                    (None, None, 0, 0, "tod=off"),
                    (9, 16, 0, 0, "tod=09-16"),
                ]
        else:
            tod_variants = [
                (None, None, 0, 0, "tod=seed"),
                (10, 15, 0, 0, "tod=10-15"),
                (10, 15, 1, 2, "tod=10-15 (skip=1 cd=2)"),
                (9, 16, 0, 0, "tod=09-16"),
                (10, 16, 0, 0, "tod=10-16"),
            ]

        # Shock + risk overlay pockets are centralized in helper registries.
        shock_variants = _build_champ_refine_shock_variants(is_slv=is_slv)
        risk_variants = _build_champ_refine_risk_variants(is_slv=is_slv)

        def _entry_variants_for_cfg(base_cfg: ConfigBundle) -> list[tuple[str, int, str]]:
            seed_mode = str(getattr(base_cfg.strategy, "ema_entry_mode", "cross") or "cross").strip().lower()
            if seed_mode not in ("cross", "trend"):
                seed_mode = "cross"
            try:
                seed_confirm = int(getattr(base_cfg.strategy, "entry_confirm_bars", 0) or 0)
            except (TypeError, ValueError):
                seed_confirm = 0
            other_mode = "trend" if seed_mode == "cross" else "cross"
            return [
                (seed_mode, seed_confirm, f"entry=seed({seed_mode} c={seed_confirm})"),
                (other_mode, 0, f"entry={other_mode} c=0"),
            ]

        def _build_stage3a_plan(
            base_exit_local: list[tuple[ConfigBundle, str]],
            *,
            seed_tag_local: str,
        ) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, exit_note) in enumerate(base_exit_local):
                entry_variants = _entry_variants_for_cfg(base_cfg)
                for tod_idx, (tod_s, tod_e, skip, cooldown, tod_note) in enumerate(tod_variants):
                    for perm_idx, (perm_over, perm_note) in enumerate(perm_variants):
                        for ss_idx, (ss_over, ss_note) in enumerate(signed_slope_variants):
                            for entry_idx, (entry_mode, entry_confirm, entry_note) in enumerate(entry_variants):
                                over: dict[str, object] = {}
                                over.update(perm_over)
                                over.update(ss_over)
                                over["skip_first_bars"] = int(skip)
                                over["cooldown_bars"] = int(cooldown)
                                over["entry_start_hour_et"] = tod_s
                                over["entry_end_hour_et"] = tod_e
                                f = _merge_filters(base_cfg.strategy.filters, over)
                                cfg = replace(
                                    base_cfg,
                                    strategy=replace(
                                        base_cfg.strategy,
                                        filters=f,
                                        ema_entry_mode=str(entry_mode),
                                        entry_confirm_bars=int(entry_confirm),
                                    ),
                                )
                                note = (
                                    f"{seed_tag_local} | short_mult={getattr(cfg.strategy,'spot_short_risk_mult', 1.0):g} | "
                                    f"{exit_note} | {entry_note} | {tod_note} | {perm_note} | {ss_note}"
                                )
                                plan.append(
                                    (
                                        cfg,
                                        note,
                                        {
                                            "base_idx": int(base_idx),
                                            "tod_idx": int(tod_idx),
                                            "perm_idx": int(perm_idx),
                                            "ss_idx": int(ss_idx),
                                            "entry_idx": int(entry_idx),
                                        },
                                    )
                                )
            return plan

        def _build_stage3b_plan(
            shortlist_local: list[tuple[ConfigBundle, str]],
        ) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, base_note) in enumerate(shortlist_local):
                for shock_idx, (shock_over, shock_note) in enumerate(shock_variants):
                    for risk_idx, (risk_over, risk_note) in enumerate(risk_variants):
                        over: dict[str, object] = {}
                        over.update(shock_over)
                        over.update(risk_over)
                        f = _merge_filters(base_cfg.strategy.filters, over)
                        cfg = replace(base_cfg, strategy=replace(base_cfg.strategy, filters=f))
                        note = f"{base_note} | {shock_note} | {risk_note}"
                        plan.append(
                            (
                                cfg,
                                note,
                                {
                                    "base_idx": int(base_idx),
                                    "shock_idx": int(shock_idx),
                                    "risk_idx": int(risk_idx),
                                },
                            )
                        )
            return plan

        report_every = 200

        if args.champ_refine_stage3a:
            payload_path = Path(str(args.champ_refine_stage3a))
            payload_decoded = _load_worker_stage_payload(
                schema_name="champ_refine_stage3a",
                payload_path=payload_path,
            )
            seed_tag = str(payload_decoded.get("seed_tag") or "seed")
            base_exit_local = [
                (cfg, note)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]

            stage3a_plan_all = _build_stage3a_plan(base_exit_local, seed_tag_local=seed_tag)
            stage3a_total = len(stage3a_plan_all)
            if len(stage3a_plan_all) != int(stage3a_total):
                raise SystemExit(
                    f"champ_refine stage3a worker internal error: combos={len(stage3a_plan_all)} expected={stage3a_total}"
                )

            _run_sharded_stage_worker(
                stage_label="champ_refine stage3a",
                worker_raw=args.champ_refine_worker,
                workers_raw=args.champ_refine_workers,
                out_path_raw=str(args.champ_refine_out or ""),
                out_flag_name="champ-refine-out",
                plan_all=stage3a_plan_all,
                bars=bars_sig,
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
            )
            return

        if args.champ_refine_stage3b:
            payload_path = Path(str(args.champ_refine_stage3b))
            payload_decoded = _load_worker_stage_payload(
                schema_name="champ_refine_stage3b",
                payload_path=payload_path,
            )
            shortlist_local = [
                (cfg, note)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]

            stage3b_plan_all = _build_stage3b_plan(shortlist_local)
            stage3b_total = len(stage3b_plan_all)
            if len(stage3b_plan_all) != int(stage3b_total):
                raise SystemExit(
                    f"champ_refine stage3b worker internal error: combos={len(stage3b_plan_all)} expected={stage3b_total}"
                )

            _run_sharded_stage_worker(
                stage_label="champ_refine stage3b",
                worker_raw=args.champ_refine_worker,
                workers_raw=args.champ_refine_workers,
                out_path_raw=str(args.champ_refine_out or ""),
                out_flag_name="champ-refine-out",
                plan_all=stage3b_plan_all,
                bars=bars_sig,
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
            )
            return

        for seed_idx, seed in enumerate(seeds, start=1):
            seed_metrics = seed.get("metrics") or {}
            try:
                seed_pnl_dd = float(seed_metrics.get("pnl_over_dd") or 0.0)
            except (TypeError, ValueError):
                seed_pnl_dd = 0.0
            try:
                seed_pnl = float(seed_metrics.get("pnl") or 0.0)
            except (TypeError, ValueError):
                seed_pnl = 0.0
            seed_name = str(seed.get("group_name") or "").strip() or f"seed_{seed_idx:02d}"
            st = seed.get("strategy") or {}
            seed_tag = (
                f"seed#{seed_idx:02d} "
                f"ema={st.get('ema_preset')} {st.get('ema_entry_mode')} "
                f"regime={st.get('regime_mode')}@{st.get('regime_bar_size')} "
                f"exit={st.get('spot_exit_mode')}"
            )
            print(f"champ_refine seed {seed_idx}/{len(seeds)}: pnl/dd={seed_pnl_dd:.2f} pnl={seed_pnl:.0f} {seed_tag}")

            base = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg_seed = _apply_milestone_base(base, strategy=seed["strategy"], filters=seed.get("filters"))

            base_row = _run_cfg(cfg=cfg_seed)
            if base_row:
                note = f"{seed_tag} | base"
                base_row["note"] = note
                _record_milestone(cfg_seed, base_row, note)
                rows.append(base_row)

            # Stage 1: short asymmetry scan (find a good multiplier pocket for this seed).
            seed_short_raw = getattr(cfg_seed.strategy, "spot_short_risk_mult", 1.0)
            try:
                seed_short = float(1.0 if seed_short_raw is None else seed_short_raw)
            except (TypeError, ValueError):
                seed_short = 1.0
            short_grid = [seed_short, *short_grid_base]
            short_vals: list[float] = []
            for v in short_grid:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f < 0.0:
                    continue
                if f not in short_vals:
                    short_vals.append(f)

            stage1: list[tuple[float, ConfigBundle, dict]] = []
            for mult in short_vals:
                cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, spot_short_risk_mult=float(mult)))
                row = _run_cfg(cfg=cfg)
                tested_total += 1
                now = pytime.perf_counter()
                if tested_total % report_every == 0 or (now - last_progress) >= heartbeat_sec:
                    print(
                        _progress_line(
                            label="champ_refine progress",
                            tested=int(tested_total),
                            total=None,
                            kept=len(rows),
                            started_at=t0_all,
                            rate_unit="cfg/s",
                        ),
                        flush=True,
                    )
                    last_progress = float(now)
                if not row:
                    continue
                note = f"{seed_tag} | short_mult={mult:g}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
                stage1.append((float(mult), cfg, row))

            if not stage1:
                continue

            stage1_sorted = sorted(stage1, key=lambda t: _score_row_pnl_dd(t[2]), reverse=True)
            top_short_mults: list[float] = []
            for mult, _, _ in stage1_sorted:
                if mult not in top_short_mults:
                    top_short_mults.append(mult)
                if len(top_short_mults) >= 2:
                    break
            if 0.01 not in top_short_mults:
                top_short_mults.append(0.01)
            top_short_mults = top_short_mults[:3]

            best_short_mult = top_short_mults[0]

            # Stage 2: micro bias neighborhood (Supertrend only), evaluated using the best short-mult from stage1.
            base_for_regime = replace(cfg_seed, strategy=replace(cfg_seed.strategy, spot_short_risk_mult=best_short_mult))
            regime_variants: list[ConfigBundle] = [base_for_regime]
            if str(getattr(base_for_regime.strategy, "regime_mode", "") or "").strip().lower() == "supertrend":
                try:
                    seed_atr = int(getattr(base_for_regime.strategy, "supertrend_atr_period", 10) or 10)
                except (TypeError, ValueError):
                    seed_atr = 10
                try:
                    seed_mult = float(getattr(base_for_regime.strategy, "supertrend_multiplier", 3.0) or 3.0)
                except (TypeError, ValueError):
                    seed_mult = 3.0
                seed_src = str(getattr(base_for_regime.strategy, "supertrend_source", "hl2") or "hl2").strip().lower()

                atr_vals = []
                atr_candidates = (seed_atr, 7, 10, 14) if not is_slv else (seed_atr, 5, 7, 10, 14)
                for v in atr_candidates:
                    if v not in atr_vals:
                        atr_vals.append(v)
                mult_vals: list[float] = []
                mult_candidates = (
                    (seed_mult - 0.05, seed_mult, seed_mult + 0.05, 0.45, 0.50, 0.55, 0.60)
                    if not is_slv
                    else (seed_mult - 0.05, seed_mult, seed_mult + 0.05, seed_mult + 0.10)
                )
                for v in mult_candidates:
                    if v <= 0:
                        continue
                    fv = float(v)
                    if fv not in mult_vals:
                        mult_vals.append(fv)
                src_vals: list[str] = []
                for v in (seed_src, "hl2", "close"):
                    sv = str(v).strip().lower()
                    if sv and sv not in src_vals:
                        src_vals.append(sv)

                stage2: list[tuple[ConfigBundle, dict]] = []
                atr_pick = atr_vals[:4] if is_slv else atr_vals[:3]
                for atr_p in atr_pick:
                    for mult in mult_vals[:4]:
                        for src in src_vals[:2]:
                            cfg = replace(
                                base_for_regime,
                                strategy=replace(
                                    base_for_regime.strategy,
                                    supertrend_atr_period=int(atr_p),
                                    supertrend_multiplier=float(mult),
                                    supertrend_source=str(src),
                                ),
                            )
                            row = _run_cfg(cfg=cfg)
                            tested_total += 1
                            now = pytime.perf_counter()
                            if tested_total % report_every == 0 or (now - last_progress) >= heartbeat_sec:
                                print(
                                    _progress_line(
                                        label="champ_refine progress",
                                        tested=int(tested_total),
                                        total=None,
                                        kept=len(rows),
                                        started_at=t0_all,
                                        rate_unit="cfg/s",
                                    ),
                                    flush=True,
                                )
                                last_progress = float(now)
                            if not row:
                                continue
                            note = f"{seed_tag} | ST({atr_p},{mult:g},{src}) short_mult={best_short_mult:g}"
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)
                            stage2.append((cfg, row))
                if stage2:
                    stage2_sorted = sorted(stage2, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[:2]
                    regime_variants = [t[0] for t in stage2_sorted]

            # Expand the shortlist: top regimes × top short mults.
            base_variants: list[ConfigBundle] = []
            for r_cfg in regime_variants[:2]:
                for mult in top_short_mults:
                    base_variants.append(
                        replace(r_cfg, strategy=replace(r_cfg.strategy, spot_short_risk_mult=float(mult)))
                    )

            # Stage 3A: lightweight micro over exit semantics + TOD/permission/signed-slope.
            #
            # The CURRENT champ family unlocked on:
            # - stop-only exits + reversal exits
            # - flip exits gated to profit-only
            # so we include a tiny exit pocket here (still bounded).
            base_exit_variants: list[tuple[ConfigBundle, str]] = []
            for base_cfg in base_variants:
                seen_exit: set[str] = set()

                def _add_exit(cfg: ConfigBundle, note: str) -> None:
                    key = _milestone_key(cfg)
                    if key in seen_exit:
                        return
                    seen_exit.add(key)
                    base_exit_variants.append((cfg, note))

                _add_exit(base_cfg, "exit=seed")
                if is_slv:
                    _add_exit(
                        replace(base_cfg, strategy=replace(base_cfg.strategy, spot_close_eod=True)),
                        "exit=seed close_eod=1",
                    )
                _add_exit(
                    replace(base_cfg, strategy=replace(base_cfg.strategy, flip_exit_min_hold_bars=2)),
                    "exit=seed hold=2",
                )

                # Champ-style stop-only + reversal exit (works even if the seed used ATR exits).
                sl_vals = (0.03, 0.04, 0.045) if not is_slv else (0.008, 0.010, 0.012, 0.015, 0.020)
                hold_vals = (2,) if not is_slv else (0, 2, 4)
                for sl in sl_vals:
                    for hold in hold_vals:
                        _add_exit(
                            replace(
                                base_cfg,
                                strategy=replace(
                                    base_cfg.strategy,
                                    spot_exit_mode="pct",
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=float(sl),
                                    exit_on_signal_flip=True,
                                    flip_exit_mode="entry",
                                    flip_exit_only_if_profit=True,
                                    flip_exit_min_hold_bars=int(hold),
                                    flip_exit_gate_mode="off",
                                ),
                            ),
                            f"exit=stop{sl:g} flipprofit hold={hold}",
                        )
                        if is_slv and float(sl) == 0.010 and hold in (0, 2):
                            _add_exit(
                                replace(
                                    base_cfg,
                                    strategy=replace(
                                        base_cfg.strategy,
                                        spot_exit_mode="pct",
                                        spot_profit_target_pct=None,
                                        spot_stop_loss_pct=float(sl),
                                        exit_on_signal_flip=True,
                                        flip_exit_mode="entry",
                                        flip_exit_only_if_profit=True,
                                        flip_exit_min_hold_bars=int(hold),
                                        flip_exit_gate_mode="off",
                                        spot_close_eod=True,
                                    ),
                                ),
                                f"exit=stop{sl:g} flipprofit hold={hold} close_eod=1",
                            )

                # Explicit "exit on the next flip" (no profit gate). Useful for reducing
                # long-hold drawdowns / improving stability.
                sl_flip_any = 0.04 if not is_slv else 0.012
                _add_exit(
                    replace(
                        base_cfg,
                        strategy=replace(
                            base_cfg.strategy,
                            spot_exit_mode="pct",
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=float(sl_flip_any),
                            exit_on_signal_flip=True,
                            flip_exit_mode="cross",
                            flip_exit_only_if_profit=False,
                            flip_exit_min_hold_bars=0,
                            flip_exit_gate_mode="off",
                        ),
                    ),
                    f"exit=stop{sl_flip_any:g} flipany cross hold=0",
                )

                # "Exit accuracy" gate (re-test in the modern shock/risk context).
                sl_accuracy = 0.04 if not is_slv else 0.012
                _add_exit(
                    replace(
                        base_cfg,
                        strategy=replace(
                            base_cfg.strategy,
                            spot_exit_mode="pct",
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=float(sl_accuracy),
                            exit_on_signal_flip=True,
                            flip_exit_mode="entry",
                            flip_exit_only_if_profit=True,
                            flip_exit_min_hold_bars=2,
                            flip_exit_gate_mode="regime_or_permission",
                        ),
                    ),
                    f"exit=stop{sl_accuracy:g} flipprofit hold=2 gate=reg_or_perm",
                )

                # Trend confirm micro (very small): sometimes improves stability by reducing noise.
                if str(getattr(base_cfg.strategy, "ema_entry_mode", "") or "").strip().lower() == "trend":
                    try:
                        seed_confirm = int(getattr(base_cfg.strategy, "entry_confirm_bars", 0) or 0)
                    except (TypeError, ValueError):
                        seed_confirm = 0
                    if seed_confirm == 0:
                        sl_confirm = 0.04 if not is_slv else 0.012
                        _add_exit(
                            replace(
                                base_cfg,
                                strategy=replace(
                                    base_cfg.strategy,
                                    entry_confirm_bars=1,
                                    spot_exit_mode="pct",
                                    spot_profit_target_pct=None,
                                    spot_stop_loss_pct=float(sl_confirm),
                                    exit_on_signal_flip=True,
                                    flip_exit_mode="entry",
                                    flip_exit_only_if_profit=True,
                                    flip_exit_min_hold_bars=2,
                                    flip_exit_gate_mode="off",
                                ),
                            ),
                            f"exit=stop{sl_confirm:g} flipprofit hold=2 confirm=1",
                        )

            stage3a: list[tuple[ConfigBundle, dict, str]] = []
            total_3a = len(base_exit_variants) * 2 * len(tod_variants) * len(perm_variants) * len(signed_slope_variants)
            print(f"  stage3a micro: total={total_3a}", flush=True)

            def _on_stage3a_row(cfg: ConfigBundle, row: dict, note: str) -> None:
                rows.append(row)
                stage3a.append((cfg, row, note))

            tested_total += _run_stage_cfg_rows(
                stage_label="  stage3a",
                total=total_3a,
                jobs_req=int(jobs),
                serial_plan_builder=lambda: _build_stage3a_plan(base_exit_variants, seed_tag_local=seed_tag),
                bars=bars_sig,
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
                on_row=_on_stage3a_row,
                parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                    axis_name="champ_refine",
                    stage_label="  stage3a",
                    total=total_3a,
                    jobs=int(jobs),
                    payload={
                        "seed_tag": seed_tag,
                        "base_exit_variants": [
                            _encode_cfg_payload(base_cfg, note=exit_note, note_key="exit_note")
                            for base_cfg, exit_note in base_exit_variants
                        ],
                    },
                    payload_filename="stage3a_payload.json",
                    temp_prefix="tradebot_champ_refine_3a_",
                    worker_tmp_prefix="tradebot_champ_refine_3a_run_",
                    worker_tag="cr3a",
                    out_prefix="stage3a_out",
                    stage_flag="--champ-refine-stage3a",
                    worker_flag="--champ-refine-worker",
                    workers_flag="--champ-refine-workers",
                    out_flag="--champ-refine-out",
                    strip_flags_with_values=(
                        "--champ-refine-stage3a",
                        "--champ-refine-stage3b",
                        "--champ-refine-worker",
                        "--champ-refine-workers",
                        "--champ-refine-out",
                        "--champ-refine-run-min-trades",
                    ),
                    run_min_trades_flag="--champ-refine-run-min-trades",
                    run_min_trades=int(run_min_trades),
                    capture_error="Failed to capture champ_refine stage3a worker stdout.",
                    failure_label="champ_refine stage3a worker",
                    missing_label="champ_refine stage3a",
                    invalid_label="champ_refine stage3a",
                ),
                parallel_default_note="stage3a",
                parallel_dedupe_by_milestone_key=True,
                record_milestones=True,
            )

            if not stage3a:
                continue

            shortlist = _rank_cfg_rows(
                stage3a,
                scorers=[(_score_row_pnl_dd, 6), (_score_row_pnl, 4), (_score_row_roi, 3), (_score_row_win_rate, 3)],
                limit=10,
            )

            # Stage 3B: apply shock + TR-overlay pockets to the shortlist.
            total_3b = len(shortlist) * len(shock_variants) * len(risk_variants)
            print(f"  stage3b shock+risk: shortlist={len(shortlist)} total={total_3b}", flush=True)

            def _on_stage3b_row(_cfg: ConfigBundle, row: dict, _note: str) -> None:
                rows.append(row)

            tested_total += _run_stage_cfg_rows(
                stage_label="  stage3b",
                total=total_3b,
                jobs_req=int(jobs),
                serial_plan_builder=lambda: _build_stage3b_plan([(cfg, note) for cfg, _row, note in shortlist]),
                bars=bars_sig,
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
                on_row=_on_stage3b_row,
                parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                    axis_name="champ_refine",
                    stage_label="  stage3b",
                    total=total_3b,
                    jobs=int(jobs),
                    payload={
                        "seed_tag": seed_tag,
                        "shortlist": [
                            _encode_cfg_payload(base_cfg, note=base_note, note_key="base_note")
                            for base_cfg, _row, base_note in shortlist
                        ],
                    },
                    payload_filename="stage3b_payload.json",
                    temp_prefix="tradebot_champ_refine_3b_",
                    worker_tmp_prefix="tradebot_champ_refine_3b_run_",
                    worker_tag="cr3b",
                    out_prefix="stage3b_out",
                    stage_flag="--champ-refine-stage3b",
                    worker_flag="--champ-refine-worker",
                    workers_flag="--champ-refine-workers",
                    out_flag="--champ-refine-out",
                    strip_flags_with_values=(
                        "--champ-refine-stage3a",
                        "--champ-refine-stage3b",
                        "--champ-refine-worker",
                        "--champ-refine-workers",
                        "--champ-refine-out",
                        "--champ-refine-run-min-trades",
                    ),
                    run_min_trades_flag="--champ-refine-run-min-trades",
                    run_min_trades=int(run_min_trades),
                    capture_error="Failed to capture champ_refine stage3b worker stdout.",
                    failure_label="champ_refine stage3b worker",
                    missing_label="champ_refine stage3b",
                    invalid_label="champ_refine stage3b",
                ),
                parallel_default_note="stage3b",
                parallel_dedupe_by_milestone_key=True,
                record_milestones=True,
            )

        print("")
        _print_leaderboards(rows, title="champ_refine (seeded, bounded)", top_n=int(args.top))

    def _sweep_shock_alpha_refine() -> None:
        """Seeded shock monetization micro grid (down-shock alpha, bounded).

        Goal: explore "stronger shock detection + monetization" without changing the base signal family,
        by sweeping:
        - earlier detectors (daily ATR% + optional TR%-trigger; TR-ratio "velocity"),
        - down-shock asymmetry (scale shorts up; scale longs down/zero),
        - risk scaling under extreme ATR% (so we don't nuke stability).
        """
        nonlocal run_calls_total

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag="shock_alpha_refine",
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            default_path="backtests/out/tqqq_exec5m_v34_champ_only_milestone.json",
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print("=== shock_alpha_refine: seeded shock monetization micro grid ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []
        bars_sig = _bars_cached(signal_bar_size)
        report_every = 100

        gate_modes = ["detect", "surf", "block_longs"]
        regime_override_dirs = [False, True]

        # "Monetize down-shocks" knobs (only active when shock=True and direction=down).
        short_risk_factors = [1.0, 2.0, 5.0, 12.0]
        long_down_factors = [1.0, 0.7, 0.4, 0.0]

        # When ATR% explodes, clamp the risk-dollars (prevents over-leverage in the worst regime).
        risk_scale_variants: list[tuple[float | None, float | None, str]] = [
            (None, None, "risk_scale=off"),
            (12.0, 0.2, "risk_scale=atr12 min=0.2"),
            (12.0, 0.3, "risk_scale=atr12 min=0.3"),
            (14.0, 0.2, "risk_scale=atr14 min=0.2"),
        ]

        detector_variants: list[tuple[dict[str, object], str]] = []
        # Daily ATR% family (v25/v31/v32 core), plus TR%-triggered early ON.
        for on_atr, off_atr in ((13.5, 13.0), (14.0, 13.5)):
            for on_tr in (None, 11.0, 14.0):
                over: dict[str, object] = {
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": float(on_atr),
                    "shock_daily_off_atr_pct": float(off_atr),
                }
                note = f"det=daily_atr on={on_atr:g} off={off_atr:g}"
                if on_tr is not None:
                    over["shock_daily_on_tr_pct"] = float(on_tr)
                    note += f" tr>={on_tr:g}"
                detector_variants.append((over, note))

        # TR-ratio shock (vol acceleration / velocity).
        # We include more-sensitive variants intended to trigger earlier in crash ramps.
        for fast, slow, on_ratio, off_ratio, min_tr in (
            # Baseline (v34 champ)
            (3, 21, 1.30, 1.20, 5.0),
            # Baseline (v33 champ)
            (5, 50, 1.45, 1.30, 7.0),
            # Moderate: lower on-ratio (still fairly strict minTR%)
            (5, 50, 1.35, 1.25, 7.0),
            # Moderate: allow slightly lower baseline TR%
            (5, 50, 1.35, 1.25, 5.0),
            (5, 21, 1.35, 1.25, 5.0),
            (3, 21, 1.35, 1.25, 5.0),
            # Aggressive: very early "vol velocity" triggers
            (5, 50, 1.30, 1.20, 5.0),
            (5, 21, 1.30, 1.20, 5.0),
        ):
            detector_variants.append(
                (
                    {
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": int(fast),
                        "shock_atr_slow_period": int(slow),
                        "shock_on_ratio": float(on_ratio),
                        "shock_off_ratio": float(off_ratio),
                        "shock_min_atr_pct": float(min_tr),
                    },
                    f"det=tr_ratio {fast}/{slow} on={on_ratio:g} off={off_ratio:g} minTR%={min_tr:g}",
                )
            )

        total = (
            len(seeds)
            * len(gate_modes)
            * len(detector_variants)
            * len(regime_override_dirs)
            * len(short_risk_factors)
            * len(long_down_factors)
            * len(risk_scale_variants)
        )

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for gate_mode in gate_modes:
                for det_over, det_note in detector_variants:
                    for override_dir in regime_override_dirs:
                        for short_factor in short_risk_factors:
                            for long_down in long_down_factors:
                                for target_atr, min_mult, scale_note in risk_scale_variants:
                                    f_over: dict[str, object] = {
                                        "shock_gate_mode": str(gate_mode),
                                        "shock_direction_source": "signal",
                                        "shock_direction_lookback": 1,
                                        "shock_regime_override_dir": bool(override_dir),
                                        "shock_short_risk_mult_factor": float(short_factor),
                                        "shock_long_risk_mult_factor_down": float(long_down),
                                    }
                                    if target_atr is None:
                                        f_over["shock_risk_scale_target_atr_pct"] = None
                                    else:
                                        f_over["shock_risk_scale_target_atr_pct"] = float(target_atr)
                                        if min_mult is not None:
                                            f_over["shock_risk_scale_min_mult"] = float(min_mult)
                                    f_over.update(det_over)
                                    f_obj = _merge_filters(cfg_seed.strategy.filters, f_over)
                                    cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                                    note = (
                                        f"{seed_note} | gate={gate_mode} {det_note} | "
                                        f"override_dir={int(override_dir)} | "
                                        f"short_factor={short_factor:g} long_down={long_down:g} | {scale_note}"
                                    )
                                    yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag="shock_alpha_refine",
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)

        _print_leaderboards(rows, title="shock_alpha_refine (seeded shock alpha micro)", top_n=int(args.top))

    def _sweep_shock_velocity_refine(*, wide: bool = False) -> None:
        """Seeded joint micro grid: TR-ratio shock sensitivity × TR% overlays (gap magnitude + TR velocity).

        Intent: dethrone the CURRENT champ by improving "pre-shock ramp" behavior while staying inside the
        same base strategy family (seeded from a champ milestone JSON).
        """
        nonlocal run_calls_total

        axis_tag = "shock_velocity_refine_wide" if wide else "shock_velocity_refine"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            default_path="backtests/out/tqqq_exec5m_v37_champ_only_milestone.json",
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print(f"=== {axis_tag}: seeded TR-ratio × TR-velocity overlays ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []
        bars_sig = _bars_cached(signal_bar_size)
        report_every = 100

        shock_variants: list[tuple[dict[str, object], str]] = []
        shock_fast_slow = ((3, 21), (5, 21), (5, 50))
        shock_on_off = ((1.25, 1.15), (1.30, 1.20), (1.35, 1.25), (1.40, 1.30))
        shock_min_tr = (4.0, 5.0, 6.0, 7.0)
        if wide:
            shock_fast_slow = (*shock_fast_slow, (3, 50))
            shock_on_off = ((1.20, 1.10), *shock_on_off)
            shock_min_tr = (3.0, *shock_min_tr)

        for fast, slow in shock_fast_slow:
            for on_ratio, off_ratio in shock_on_off:
                for min_tr in shock_min_tr:
                    shock_variants.append(
                        (
                            {
                                "shock_gate_mode": "detect",
                                "shock_detector": "tr_ratio",
                                "shock_direction_source": "signal",
                                "shock_direction_lookback": 1,
                                "shock_atr_fast_period": int(fast),
                                "shock_atr_slow_period": int(slow),
                                "shock_on_ratio": float(on_ratio),
                                "shock_off_ratio": float(off_ratio),
                                "shock_min_atr_pct": float(min_tr),
                            },
                            f"shock=detect tr_ratio {fast}/{slow} on={on_ratio:g} off={off_ratio:g} minTR%={min_tr:g}",
                        )
                    )

        def _risk_off_overrides() -> dict[str, object]:
            return {
                "risk_entry_cutoff_hour_et": None,
                "riskoff_tr5_med_pct": None,
                "riskpanic_tr5_med_pct": None,
                "riskpanic_neg_gap_ratio_min": None,
                "riskpanic_neg_gap_abs_pct_min": None,
                "riskpanic_tr5_med_delta_min_pct": None,
                "riskpanic_long_scale_mode": "off",
                "riskpanic_long_scale_tr_delta_max_pct": None,
                "riskpop_tr5_med_pct": None,
                "riskpop_pos_gap_ratio_min": None,
                "riskpop_pos_gap_abs_pct_min": None,
                "riskpop_tr5_med_delta_min_pct": None,
                "riskoff_mode": "hygiene",
            }

        risk_variants: list[tuple[dict[str, object], str]] = [(_risk_off_overrides(), "risk=off")]

        # Riskpanic: defensive overlay (neg gaps + TR-median + optional acceleration gate).
        panic_tr = 9.0
        panic_gap = 0.6
        panic_lb = 5
        panic_cutoffs = (None, 15)
        panic_short_factors = (1.0, 0.5)
        panic_abs_gap = (None, 0.01, 0.02)
        panic_long_extra = (0.8, 0.6, 0.0)
        panic_tr_delta_variants: tuple[tuple[float | None, int, str], ...] = (
            (None, 1, "trΔ=off"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (1.0, 1, "trΔ>=1.0@1d"),
        )
        if wide:
            panic_short_factors = (1.0, 0.5, 0.2)
            panic_abs_gap = (None, 0.01)
            panic_tr_delta_variants = (
                (None, 1, "trΔ=off"),
                (0.25, 1, "trΔ>=0.25@1d"),
                (0.5, 1, "trΔ>=0.5@1d"),
                (0.75, 1, "trΔ>=0.75@1d"),
                (1.0, 1, "trΔ>=1.0@1d"),
            )
            panic_long_extra = (0.8, 0.6, 0.4, 0.0)

        panic_scale_delta_max = (None, 0.5, 1.0, 2.0)
        if wide:
            panic_scale_delta_max = (None, 0.25, 0.5, 1.0, 2.0, 4.0)

        for cutoff in panic_cutoffs:
            for short_factor in panic_short_factors:
                for abs_gap in panic_abs_gap:
                    for tr_delta_min, tr_delta_lb, tr_delta_note in panic_tr_delta_variants:
                        long_factors = (1.0,)
                        if float(short_factor) == 1.0:
                            long_factors = (*long_factors, *panic_long_extra)

                        for long_factor in long_factors:
                            cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                            gap_note = "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                            base_over = {
                                **_risk_off_overrides(),
                                "riskpanic_tr5_med_pct": float(panic_tr),
                                "riskpanic_neg_gap_ratio_min": float(panic_gap),
                                "riskpanic_neg_gap_abs_pct_min": float(abs_gap) if abs_gap is not None else None,
                                "riskpanic_lookback_days": int(panic_lb),
                                "riskpanic_tr5_med_delta_min_pct": float(tr_delta_min) if tr_delta_min is not None else None,
                                "riskpanic_tr5_med_delta_lookback_days": int(tr_delta_lb),
                                "riskpanic_long_risk_mult_factor": float(long_factor),
                                "riskpanic_short_risk_mult_factor": float(short_factor),
                                "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                "riskoff_mode": "hygiene",
                            }
                            base_note = (
                                f"riskpanic TRmed{panic_lb}>=9 gap>={panic_gap:g} {gap_note} {tr_delta_note} "
                                f"long={long_factor:g} short={short_factor:g} {cut_note}"
                            )
                            risk_variants.append((base_over, base_note))

                            # Candidate-3 policy: pre-panic continuous scaling (only meaningful when long_factor<1 and trΔ is in play).
                            if tr_delta_min is not None and float(long_factor) < 1.0:
                                for delta_max in panic_scale_delta_max:
                                    dm_note = "Δmax=Δmin" if delta_max is None else f"Δmax={delta_max:g}"
                                    over = dict(base_over)
                                    over["riskpanic_long_scale_mode"] = "linear"
                                    over["riskpanic_long_scale_tr_delta_max_pct"] = (
                                        float(delta_max) if delta_max is not None else None
                                    )
                                    risk_variants.append((over, f"{base_note} | scale=lin {dm_note}"))

        # Riskpop: controlled "momentum-on" vs defensive variants (keep this tight: pop can destabilize).
        pop_tr = 8.0
        pop_gap = 0.6
        pop_lb = 5
        for abs_gap in (None, 0.01):
            for tr_delta_min, tr_delta_lb, tr_delta_note in (
                (None, 1, "trΔ=off"),
                (0.5, 5, "trΔ>=0.5@5d"),
            ):
                for long_factor, short_factor, mode_note in (
                    (0.8, 1.0, "defensive"),
                    (1.2, 0.0, "aggressive"),
                ):
                    gap_note = "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                    risk_variants.append(
                        (
                            {
                                **_risk_off_overrides(),
                                "riskpop_tr5_med_pct": float(pop_tr),
                                "riskpop_pos_gap_ratio_min": float(pop_gap),
                                "riskpop_pos_gap_abs_pct_min": float(abs_gap) if abs_gap is not None else None,
                                "riskpop_lookback_days": int(pop_lb),
                                "riskpop_tr5_med_delta_min_pct": (
                                    float(tr_delta_min) if tr_delta_min is not None else None
                                ),
                                "riskpop_tr5_med_delta_lookback_days": int(tr_delta_lb),
                                "riskpop_long_risk_mult_factor": float(long_factor),
                                "riskpop_short_risk_mult_factor": float(short_factor),
                                "risk_entry_cutoff_hour_et": 15,
                                "riskoff_mode": "hygiene",
                            },
                            f"riskpop({mode_note}) TRmed{pop_lb}>=8 gap+>={pop_gap:g} {gap_note} {tr_delta_note} "
                            f"long={long_factor:g} short={short_factor:g} cutoff<15",
                        )
                    )

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for shock_over, shock_note in shock_variants:
                for risk_over, risk_note in risk_variants:
                    over: dict[str, object] = {}
                    over.update(shock_over)
                    over.update(risk_over)
                    f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                    cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                    note = f"{seed_note} | {shock_note} | {risk_note}"
                    yield cfg, note, None

        plan_all = list(_iter_seed_micro_plan(seeds=seeds, build_variants=_build_variants))
        total = len(plan_all)
        print(f"- shock_variants={len(shock_variants)} risk_variants={len(risk_variants)} total={total}", flush=True)

        if args.shock_velocity_worker is not None:
            if not offline:
                raise SystemExit("shock_velocity_refine worker mode requires --offline (avoid parallel IBKR sessions).")
            out_path_raw = str(args.shock_velocity_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--shock-velocity-out is required for shock_velocity_refine worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.shock_velocity_worker,
                args.shock_velocity_workers,
                label="shock_velocity_refine",
            )

            local_total = (total // workers) + (1 if worker_id < (total % workers) else 0)
            shard_plan = (item for combo_idx, item in enumerate(plan_all) if (combo_idx % int(workers)) == int(worker_id))
            tested, kept = _run_sweep(
                plan=shard_plan,
                total=local_total,
                progress_label=f"{axis_tag} worker {worker_id+1}/{workers}",
                report_every=max(1, int(report_every)),
                heartbeat_sec=50.0,
                record_milestones=False,
            )
            records: list[dict] = []
            for cfg, row, note, _meta in kept:
                records.append(
                    {
                        "strategy": _spot_strategy_payload(cfg, meta=meta),
                        "filters": _filters_payload(cfg.strategy.filters),
                        "note": str(note),
                        "row": row,
                    }
                )
            out_payload = {"tested": int(tested), "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"{axis_tag} worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        tested_total = _run_stage_cfg_rows(
            stage_label=str(axis_tag),
            total=total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=max(1, int(report_every)),
            heartbeat_sec=50.0,
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan=plan_all,
            parallel_payloads_builder=lambda: _run_parallel_stage(
                axis_name=str(axis_tag),
                stage_label=str(axis_tag),
                total=total,
                jobs=int(jobs),
                worker_tmp_prefix="tradebot_shock_velocity_",
                worker_tag="sv",
                out_prefix="shock_velocity_out",
                worker_flag="--shock-velocity-worker",
                workers_flag="--shock-velocity-workers",
                out_flag="--shock-velocity-out",
                strip_flags_with_values=(
                    "--shock-velocity-worker",
                    "--shock-velocity-workers",
                    "--shock-velocity-out",
                ),
                capture_error=f"Failed to capture {axis_tag} worker stdout.",
                failure_label=f"{axis_tag} worker",
                missing_label=str(axis_tag),
                invalid_label=str(axis_tag),
            ),
            parallel_default_note=str(axis_tag),
            parallel_dedupe_by_milestone_key=False,
            record_milestones=True,
        )
        if jobs > 1 and total > 0:
            run_calls_total += int(tested_total)

        _print_leaderboards(
            rows,
            title=f"{axis_tag} (seeded tr_ratio × TR-velocity overlays)",
            top_n=int(args.top),
        )

    def _sweep_shock_throttle_refine() -> None:
        """Seeded micro-grid: shock risk scaling target/min-mult × a tiny stop-loss pocket.

        Intent: improve the CURRENT champ by shrinking risk in "moderate vol" conditions where
        the default risk scaling (target_atr_pct≈12) rarely engages.
        """
        nonlocal run_calls_total

        axis_tag = "shock_throttle_refine"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            default_path="backtests/out/tqqq_exec5m_v37_champ_only_milestone.json",
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print(f"=== {axis_tag}: seeded shock scaling target × min-mult pocket ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []
        report_every = 50

        def _shock_mode(filters: FiltersConfig | None) -> str:
            if filters is None:
                return "off"
            mode = str(getattr(filters, "shock_gate_mode", None) or "off").strip().lower()
            if mode in ("", "0", "false", "none", "null"):
                mode = "off"
            if mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
                mode = "off"
            return mode

        def _shock_detector(filters: FiltersConfig | None) -> str:
            if filters is None:
                return "atr_ratio"
            raw = str(getattr(filters, "shock_detector", None) or "atr_ratio").strip().lower()
            if raw in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
                return "daily_atr_pct"
            if raw in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
                return "daily_drawdown"
            if raw in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
                return "tr_ratio"
            if raw in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
                return "atr_ratio"
            return "atr_ratio"

        def _ensure_shock_detect_overrides(filters: FiltersConfig | None) -> dict[str, object]:
            if _shock_mode(filters) != "off":
                return {}
            # Needed so shock_atr_pct is available (no entry gating change).
            return {"shock_gate_mode": "detect", "shock_detector": "daily_atr_pct"}

        def _stop_pocket(seed_stop_pct: float | None) -> tuple[float, ...]:
            try:
                base = float(seed_stop_pct) if seed_stop_pct is not None else 0.0
            except (TypeError, ValueError):
                base = 0.0
            if base <= 0:
                base = 0.045
            pocket = (
                base * 0.70,
                base * 0.85,
                base * 0.925,
                base,
                base * 1.075,
                base * 1.15,
                base * 1.30,
            )
            # Guardrails: keep stops >0 and dedupe.
            out = sorted({float(round(p, 6)) for p in pocket if p > 0})
            if not out:
                return (float(base),)
            return tuple(out)

        def _targets_pocket(
            filters: FiltersConfig | None, *, detector: str, shock_missing: bool
        ) -> tuple[float, ...]:
            # Key insight: risk scaling applies whenever shock_atr_pct is available (not only when shock==True),
            # so lowering target_atr_pct can throttle sizing in moderate-vol chop without changing the detector.
            if detector == "daily_atr_pct":
                if bool(shock_missing):
                    pocket = (2.0, 3.0, 4.0, 4.5, 5.0, 6.0)
                else:
                    try:
                        on_raw = getattr(filters, "shock_daily_on_atr_pct", None) if filters is not None else None
                        off_raw = getattr(filters, "shock_daily_off_atr_pct", None) if filters is not None else None
                        on = float(on_raw) if on_raw is not None else None
                        off = float(off_raw) if off_raw is not None else None
                    except (TypeError, ValueError):
                        on = None
                        off = None
                    if on is not None and off is not None and on > 0 and off > 0:
                        base = min(on, off)
                        pocket = (
                            base * 0.50,
                            base * 0.75,
                            base,
                            max(on, off),
                            max(on, off) * 1.25,
                            max(on, off) * 1.50,
                        )
                    else:
                        pocket = (2.0, 3.0, 4.0, 4.5, 5.0, 6.0)
            else:
                try:
                    min_atr = float(getattr(filters, "shock_min_atr_pct", None) or 0.0) if filters is not None else 0.0
                except (TypeError, ValueError):
                    min_atr = 0.0
                anchor = min_atr if min_atr > 0 else 7.0
                pocket = (
                    anchor,
                    anchor * 1.5,
                    anchor * 2.0,
                    anchor * 2.5,
                    anchor * 3.0,
                    anchor * 4.0,
                )
            out = sorted({float(round(v, 6)) for v in pocket if v > 0})
            return tuple(out) if out else (12.0,)

        def _daily_threshold_pocket(filters: FiltersConfig | None, *, mode: str, detector: str) -> tuple[tuple[float | None, float | None], ...]:
            if mode != "surf" or detector != "daily_atr_pct" or filters is None:
                return ((None, None),)
            try:
                on = float(getattr(filters, "shock_daily_on_atr_pct"))
                off = float(getattr(filters, "shock_daily_off_atr_pct"))
            except (TypeError, ValueError):
                return ((None, None),)
            if on <= 0 or off <= 0:
                return ((None, None),)
            on_vals = sorted({max(0.1, on - 1.5), max(0.1, on - 0.5), on, on + 0.5})
            off_vals = sorted({max(0.1, off - 1.5), max(0.1, off - 0.5), off})
            pairs: list[tuple[float | None, float | None]] = []
            for off_v in off_vals:
                for on_v in on_vals:
                    if off_v > on_v:
                        continue
                    pairs.append((float(off_v), float(on_v)))
            return tuple(pairs) if pairs else ((None, None),)

        min_mults = (0.05, 0.1, 0.2)

        def _seed_variant_count(cfg_seed: ConfigBundle) -> int:
            base_filters = cfg_seed.strategy.filters
            mode = _shock_mode(base_filters)
            detect_over = _ensure_shock_detect_overrides(base_filters)
            detector_eff = _shock_detector(base_filters) if not detect_over else "daily_atr_pct"

            targets_f = _targets_pocket(base_filters, detector=detector_eff, shock_missing=bool(detect_over))
            targets: tuple[float | None, ...] = (None,) + targets_f
            stops = _stop_pocket(getattr(cfg_seed.strategy, "spot_stop_loss_pct", None))
            daily_pairs = _daily_threshold_pocket(base_filters, mode=mode, detector=detector_eff)

            targets_variants = sum((len(min_mults) if t is not None else 1) for t in targets)
            variants = int(targets_variants) * len(stops) * len(daily_pairs)
            return int(max(0, variants))

        total = 0
        for _seed_i, _item, cfg_seed, _seed_note in _iter_seed_bundles(seeds):
            total += _seed_variant_count(cfg_seed)

        if total <= 0:
            print("No usable seeds after parsing/filtering.", flush=True)
            return

        print(
            f"- variants: seed-aware pockets (min_mult={min_mults}); total={total} "
            f"(note: auto-enables shock_gate_mode=detect when missing)",
            flush=True,
        )

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            base_filters = cfg_seed.strategy.filters
            mode = _shock_mode(base_filters)
            detect_over = _ensure_shock_detect_overrides(base_filters)
            detector_eff = _shock_detector(base_filters) if not detect_over else "daily_atr_pct"
            targets_f = _targets_pocket(base_filters, detector=detector_eff, shock_missing=bool(detect_over))
            targets: tuple[float | None, ...] = (None,) + targets_f
            stops = _stop_pocket(getattr(cfg_seed.strategy, "spot_stop_loss_pct", None))
            daily_pairs = _daily_threshold_pocket(base_filters, mode=mode, detector=detector_eff)

            for off_v, on_v in daily_pairs:
                daily_over: dict[str, object] = {}
                if off_v is not None and on_v is not None:
                    daily_over = {"shock_daily_off_atr_pct": float(off_v), "shock_daily_on_atr_pct": float(on_v)}
                for target_atr in targets:
                    min_mult_iter = (None,) if target_atr is None else min_mults
                    for min_mult in min_mult_iter:
                        base_over: dict[str, object] = {}
                        if target_atr is not None:
                            base_over["shock_risk_scale_target_atr_pct"] = float(target_atr)
                            base_over["shock_risk_scale_min_mult"] = float(min_mult) if min_mult is not None else 0.2
                            base_over["shock_risk_scale_apply_to"] = "cap"
                        base_over.update(detect_over)
                        base_over.update(daily_over)
                        f_obj = _merge_filters(cfg_seed.strategy.filters, base_over)
                        for stop_pct in stops:
                            cfg = replace(
                                cfg_seed,
                                strategy=replace(
                                    cfg_seed.strategy,
                                    filters=f_obj,
                                    spot_stop_loss_pct=float(stop_pct),
                                ),
                            )
                            surf_note = ""
                            if off_v is not None and on_v is not None:
                                surf_note = f" surf(off={off_v:g},on={on_v:g})"
                            if target_atr is None:
                                scale_note = " shock_scale=off"
                            else:
                                scale_note = f" shock_scale target_atr%={target_atr:g} min_mult={min_mult:g} apply_to=cap"
                            note = f"{seed_note} |{scale_note}{surf_note} | stop%={stop_pct:g}"
                            yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag=axis_tag,
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded shock risk scaling micro)", top_n=int(args.top))

    def _sweep_shock_throttle_tr_ratio() -> None:
        """Seeded micro-grid: keep the base strategy identical, but compute shock risk scaling off TR-ratio ATR%.

        This is a pure "throttle" lever:
        - gate shock config (e.g. daily_atr_pct surf + stop tightening) stays unchanged
        - sizing is throttled using `shock_risk_scale_*` with `shock_scale_detector=tr_ratio` (detect-only)
        """
        nonlocal run_calls_total

        axis_tag = "shock_throttle_tr_ratio"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print(f"=== {axis_tag}: seeded TR-ratio throttle micro-grid ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []

        profile = _SHOCK_THROTTLE_TR_RATIO_PROFILE
        scale_periods: tuple[tuple[int, int], ...] = tuple(profile.get("periods") or ())
        # Telemetry (SLV 1h FULL24 tr_fast_pct): p75≈0.71%, p85≈0.90%, p92≈1.14% (1y window).
        # Include a few aggressive "surprise" targets to actually touch cap-bound trades.
        targets: tuple[float, ...] = tuple(profile.get("targets") or ())
        min_mults: tuple[float, ...] = tuple(profile.get("min_mults") or ())
        apply_tos: tuple[str, ...] = tuple(profile.get("apply_tos") or ())

        report_every = 50
        total = len(seeds) * len(scale_periods) * len(targets) * len(min_mults) * len(apply_tos)

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for fast_p, slow_p in scale_periods:
                for target_atr in targets:
                    for min_mult in min_mults:
                        for apply_to in apply_tos:
                            over = {
                                "shock_scale_detector": "tr_ratio",
                                "shock_atr_fast_period": int(fast_p),
                                "shock_atr_slow_period": int(slow_p),
                                "shock_risk_scale_target_atr_pct": float(target_atr),
                                "shock_risk_scale_min_mult": float(min_mult),
                                "shock_risk_scale_apply_to": str(apply_to),
                            }
                            f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                            if f_obj is None:
                                continue
                            cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                            note = (
                                f"{seed_note} | shock_scale=tr_ratio {fast_p}/{slow_p} "
                                f"target_atr%={target_atr:g} min_mult={min_mult:g} apply_to={apply_to}"
                            )
                            yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag=axis_tag,
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
            include_base_note="shock_scale=off (base)",
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded tr_ratio throttle micro)", top_n=int(args.top))

    def _sweep_shock_throttle_drawdown() -> None:
        """Seeded micro-grid: compute shock risk scaling off daily drawdown magnitude (detect-only).

        This is a pure throttle lever:
        - primary shock config (e.g. daily_atr_pct surf + stop tightening) stays unchanged
        - sizing is throttled using `shock_risk_scale_*` with `shock_scale_detector=daily_drawdown`
        """
        nonlocal run_calls_total

        axis_tag = "shock_throttle_drawdown"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print(f"=== {axis_tag}: seeded daily_drawdown throttle micro-grid ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []

        profile = _SHOCK_THROTTLE_DRAWDOWN_PROFILE
        lookbacks: tuple[int, ...] = tuple(profile.get("lookbacks") or ())
        # SLV 1y FULL24 stop-entries drawdown magnitude (20d peak):
        # p50≈1.17%, p75≈3.12%, p90≈9.70%, p95≈10.59%, max≈12.30%.
        targets: tuple[float, ...] = tuple(profile.get("targets") or ())
        min_mults: tuple[float, ...] = tuple(profile.get("min_mults") or ())
        apply_tos: tuple[str, ...] = tuple(profile.get("apply_tos") or ())

        report_every = 50
        total = len(seeds) * len(lookbacks) * len(targets) * len(min_mults) * len(apply_tos)

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for lb in lookbacks:
                for target_dd in targets:
                    for min_mult in min_mults:
                        for apply_to in apply_tos:
                            over = {
                                "shock_scale_detector": "daily_drawdown",
                                "shock_drawdown_lookback_days": int(lb),
                                "shock_risk_scale_target_atr_pct": float(target_dd),
                                "shock_risk_scale_min_mult": float(min_mult),
                                "shock_risk_scale_apply_to": str(apply_to),
                            }
                            f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                            if f_obj is None:
                                continue
                            cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                            note = (
                                f"{seed_note} | shock_scale=daily_drawdown lb={lb} "
                                f"target_dd%={target_dd:g} min_mult={min_mult:g} apply_to={apply_to}"
                            )
                            yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag=axis_tag,
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
            include_base_note="shock_scale=off (base)",
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded daily_drawdown throttle micro)", top_n=int(args.top))

    def _sweep_riskpanic_micro() -> None:
        """Seeded micro-grid: riskpanic overlay knobs inspired by the TQQQ v37→v39 needle-thread.

        Keeps strategy identical and focuses on the small set of knobs that moved the TQQQ stability champ:
        - risk_entry_cutoff_hour_et (late-day entry cutoff on risk days)
        - riskpanic_long_risk_mult_factor + riskpanic_long_scale_mode ("linear" pre-panic de-risking)
        - riskpanic_neg_gap_abs_pct_min (count all neg gaps vs only big gaps)
        - riskpanic_tr5_med_delta_min_pct (TR-velocity gate)

        Note: riskpanic detection requires (riskpanic_tr5_med_pct, riskpanic_neg_gap_ratio_min).
        For SLV FULL24, med(TR% last 5d) is on a ~2–3% scale, not TQQQ's ~9–10%.
        """
        nonlocal run_calls_total

        axis_tag = "riskpanic_micro"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_top_candidates(candidates, seed_top=seed_top)

        print("")
        print(f"=== {axis_tag}: seeded riskpanic micro-grid ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []

        profile = _RISKPANIC_MICRO_PROFILE
        cutoffs_et: tuple[int | None, ...] = tuple(profile.get("cutoffs_et") or ())
        panic_tr_meds: tuple[float, ...] = tuple(profile.get("panic_tr_meds") or ())
        neg_gap_ratios: tuple[float, ...] = tuple(profile.get("neg_gap_ratios") or ())
        neg_gap_abs_pcts: tuple[float | None, ...] = tuple(profile.get("neg_gap_abs_pcts") or ())
        tr_delta_mins: tuple[float | None, ...] = tuple(profile.get("tr_delta_mins") or ())
        long_factors: tuple[float, ...] = tuple(profile.get("long_factors") or ())
        scale_modes: tuple[str | None, ...] = tuple(profile.get("scale_modes") or ())

        report_every = 50
        total = (
            len(seeds)
            * len(cutoffs_et)
            * len(panic_tr_meds)
            * len(neg_gap_ratios)
            * len(neg_gap_abs_pcts)
            * len(tr_delta_mins)
            * len(long_factors)
            * len(scale_modes)
        )

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for cutoff in cutoffs_et:
                for tr_med in panic_tr_meds:
                    for neg_ratio in neg_gap_ratios:
                        for abs_gap in neg_gap_abs_pcts:
                            for tr_delta_min in tr_delta_mins:
                                for long_factor in long_factors:
                                    for scale_mode in scale_modes:
                                        over: dict[str, object] = {
                                            # Disable other overlay families unless they exist in the seed.
                                            "riskoff_tr5_med_pct": None,
                                            "riskpop_tr5_med_pct": None,
                                            # Panic detector (enables overlay engine).
                                            "riskpanic_tr5_med_pct": float(tr_med),
                                            "riskpanic_neg_gap_ratio_min": float(neg_ratio),
                                            "riskpanic_neg_gap_abs_pct_min": abs_gap,
                                            "riskpanic_lookback_days": 5,
                                            "riskpanic_tr5_med_delta_min_pct": tr_delta_min,
                                            "riskpanic_tr5_med_delta_lookback_days": 1,
                                            # Panic policy.
                                            "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                            "riskpanic_long_risk_mult_factor": float(long_factor),
                                            "riskpanic_short_risk_mult_factor": 1.0,
                                            "riskpanic_long_scale_mode": scale_mode,
                                        }
                                        f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                                        if f_obj is None:
                                            continue
                                        cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                                        cut_note = "-" if cutoff is None else str(cutoff)
                                        abs_note = "None" if abs_gap is None else f"{abs_gap:g}"
                                        delta_note = "=off" if tr_delta_min is None else f">={tr_delta_min:g}"
                                        mode_note = "off" if not scale_mode else str(scale_mode)
                                        note = (
                                            f"{seed_note} | cutoff<{cut_note} | panic med5>={tr_med:g} gap>={neg_ratio:g} abs>={abs_note} "
                                            f"trΔ{delta_note} | long={long_factor:g} scale={mode_note}"
                                        )
                                        yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag=axis_tag,
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
            include_base_note="base",
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded riskpanic micro)", top_n=int(args.top))

    def _sweep_overlay_family(*, family: str | None = None) -> None:
        family_registry = {
            "shock_throttle_refine": _sweep_shock_throttle_refine,
            "shock_throttle_tr_ratio": _sweep_shock_throttle_tr_ratio,
            "shock_throttle_drawdown": _sweep_shock_throttle_drawdown,
            "riskpanic_micro": _sweep_riskpanic_micro,
        }
        family_req = str(family or getattr(args, "overlay_family_kind", "all") or "all").strip().lower()
        if family_req == "all":
            selected = tuple(family_registry.keys())
            print("")
            print("=== overlay_family: running all seeded overlay families ===")
            print("")
        else:
            if family_req not in family_registry:
                valid = ", ".join(family_registry.keys())
                raise SystemExit(f"Unknown overlay family: {family_req!r} (expected one of: {valid}, all)")
            selected = (family_req,)

        for family_name in selected:
            fn = family_registry.get(str(family_name))
            if callable(fn):
                fn()

    def _sweep_exit_pivot() -> None:
        """Seeded micro-grid: exit-model pivot for higher-cadence lanes (PT/SL + flip semantics + close_eod)."""
        nonlocal run_calls_total

        axis_tag = "exit_pivot"

        seed_path, candidates = _load_seed_candidates(
            seed_milestones=args.seed_milestones,
            axis_tag=axis_tag,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
        )

        if not candidates:
            print(f"No matching seed candidates found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_select_candidates(
            candidates,
            seed_top=seed_top,
            policy="exit_pivot",
        )

        print("")
        print(f"=== {axis_tag}: seeded exit pivot micro-grid ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        rows: list[dict] = []

        pt_vals: tuple[float | None, ...] = (None, 0.0015, 0.002, 0.003, 0.004, 0.006)
        sl_vals: tuple[float, ...] = (0.003, 0.004, 0.006, 0.008, 0.01, 0.012)
        only_profit_vals: tuple[bool, ...] = (False, True)
        close_eod_vals: tuple[bool, ...] = (False, True)
        report_every = 100
        total = len(seeds) * len(close_eod_vals) * len(only_profit_vals) * len(pt_vals) * len(sl_vals)

        def _build_variants(cfg_seed: ConfigBundle, seed_note: str, _item: dict):
            for close_eod in close_eod_vals:
                for only_profit in only_profit_vals:
                    for pt in pt_vals:
                        for sl in sl_vals:
                            cfg = replace(
                                cfg_seed,
                                strategy=replace(
                                    cfg_seed.strategy,
                                    spot_exit_mode="pct",
                                    spot_profit_target_pct=pt,
                                    spot_stop_loss_pct=float(sl),
                                    exit_on_signal_flip=True,
                                    flip_exit_only_if_profit=bool(only_profit),
                                    spot_close_eod=bool(close_eod),
                                ),
                            )
                            pt_note = "None" if pt is None else f"{pt:g}"
                            note = (
                                f"{seed_note} | close_eod={int(close_eod)} "
                                f"only_profit={int(only_profit)} | PT={pt_note} SL={sl:g}"
                            )
                            yield cfg, note, None

        tested_total = _run_seeded_micro_grid(
            axis_tag=axis_tag,
            seeds=seeds,
            rows=rows,
            build_variants=_build_variants,
            total=total,
            report_every=report_every,
            include_base_note="base",
        )
        if int(tested_total) < 0:
            return
        if jobs > 1 and int(tested_total) > 0:
            run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded exit pivot)", top_n=int(args.top))

    def _sweep_st37_refine() -> None:
        """Refine the 3/7 trend + SuperTrend(4h) cluster (seeded) with v31-style gates + overlays.

        High-level goal:
        - Take the strong 3/7 trend + ST(4h) winners (which can have monster 2y roi/dd),
          then sweep the missing v31-style "permission" gates (spread/slope), TOD windows,
          and (separately) tighten the riskpanic + shock pockets.
        - Finally, sweep a small exit/flip semantics pocket anchored on the current v31 exit style.

        Intended usage:
        - seed with a kingmaker output (spot_multitimeframe --write-top) or another milestones file
          that contains the 3/7 ST4h family you want to explore.
        """
        seed_path = _resolve_seed_milestones_path(
            seed_milestones=args.seed_milestones,
            axis_tag="st37_refine",
        )
        raw_groups = _seed_groups_from_path(seed_path)

        def _st37_seed_predicate(_group: dict, _entry: dict, strat: dict, _metrics: dict) -> bool:
            # Lock to the 3/7 trend + ST(4h) neighborhood by default.
            if str(strat.get("ema_preset") or "").strip() != "3/7":
                return False
            if str(strat.get("ema_entry_mode") or "").strip().lower() != "trend":
                return False
            if str(strat.get("regime_mode") or "").strip().lower() != "supertrend":
                return False
            if str(strat.get("regime_bar_size") or "").strip().lower() != "4 hours":
                return False
            try:
                st_atr = int(strat.get("supertrend_atr_period") or 0)
            except (TypeError, ValueError):
                st_atr = 0
            try:
                st_mult = float(strat.get("supertrend_multiplier") or 0.0)
            except (TypeError, ValueError):
                st_mult = 0.0
            st_src = str(strat.get("supertrend_source") or "").strip().lower()
            return st_atr == 7 and abs(st_mult - 0.5) <= 1e-9 and st_src == "hl2"

        candidates = _seed_candidates_for_context(
            raw_groups=raw_groups,
            symbol=symbol,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            predicate=_st37_seed_predicate,
        )

        if not candidates:
            print(f"No matching 3/7 trend + ST(4h) seeds found in {seed_path} for {symbol} {signal_bar_size} rth={use_rth}.")
            return

        seed_top = max(1, int(args.seed_top or 0))
        seeds = _seed_select_candidates(
            candidates,
            seed_top=seed_top,
            policy="st37_refine",
        )

        print("")
        print("=== st37_refine: 3/7 trend + ST(4h) refinement (seeded) ===")
        print(f"- seeds_in_file={len(candidates)} selected={len(seeds)} seed_top={seed_top}")
        print(f"- seed_path={seed_path}")
        print("")

        # Inspect the current v31-like kingmaker champ exit semantics (for anchoring stage3).
        v31_exit = None
        try:
            champ_path = Path(__file__).resolve().parent / "spot_champions.json"
            if champ_path.exists():
                champs = json.loads(champ_path.read_text())
                for g in champs.get("groups") or []:
                    if not isinstance(g, dict):
                        continue
                    entries = g.get("entries") or []
                    if not entries:
                        continue
                    entry = entries[0]
                    if not isinstance(entry, dict):
                        continue
                    st = entry.get("strategy") or {}
                    if not isinstance(st, dict):
                        continue
                    if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
                        continue
                    if str(st.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
                        continue
                    if bool(st.get("signal_use_rth")) != bool(use_rth):
                        continue
                    v31_exit = {
                        "spot_exit_mode": st.get("spot_exit_mode"),
                        "spot_profit_target_pct": st.get("spot_profit_target_pct"),
                        "spot_stop_loss_pct": st.get("spot_stop_loss_pct"),
                        "spot_atr_period": st.get("spot_atr_period"),
                        "spot_pt_atr_mult": st.get("spot_pt_atr_mult"),
                        "spot_sl_atr_mult": st.get("spot_sl_atr_mult"),
                        "spot_exit_time_et": st.get("spot_exit_time_et"),
                        "exit_on_signal_flip": st.get("exit_on_signal_flip"),
                        "flip_exit_mode": st.get("flip_exit_mode"),
                        "flip_exit_only_if_profit": st.get("flip_exit_only_if_profit"),
                        "flip_exit_min_hold_bars": st.get("flip_exit_min_hold_bars"),
                        "flip_exit_gate_mode": st.get("flip_exit_gate_mode"),
                    }
                    break
        except Exception:
            v31_exit = None

        if v31_exit:
            print("v31 (kingmaker champ) exit semantics (as stored in spot_champions.json):")
            for k, v in v31_exit.items():
                print(f"- {k}={v!r}")
            print("")

        bars_sig = _bars_cached(signal_bar_size)
        heartbeat_sec = 50.0

        def _mk_stage2_cfg(base_cfg: ConfigBundle, *, risk_over: dict[str, object], shock_over: dict[str, object]) -> ConfigBundle:
            over: dict[str, object] = {}
            over.update(risk_over)
            over.update(shock_over)
            f = _merge_filters(base_cfg.strategy.filters, overrides=over)
            return replace(base_cfg, strategy=replace(base_cfg.strategy, filters=f))

        def _build_stage2_plan(
            shortlist_local: list[tuple[ConfigBundle, str]],
            *,
            risk_variants_local: list[tuple[dict[str, object], str]],
            shock_variants_local: list[tuple[dict[str, object], str]],
        ) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, base_note) in enumerate(shortlist_local):
                for risk_idx, (risk_over, risk_note) in enumerate(risk_variants_local):
                    for shock_idx, (shock_over, shock_note) in enumerate(shock_variants_local):
                        cfg = _mk_stage2_cfg(base_cfg, risk_over=risk_over, shock_over=shock_over)
                        note = f"{base_note} | {risk_note} | {shock_note}"
                        plan.append(
                            (
                                cfg,
                                note,
                                {
                                    "base_idx": int(base_idx),
                                    "risk_idx": int(risk_idx),
                                    "shock_idx": int(shock_idx),
                                },
                            )
                        )
            return plan

        # Stage2 worker mode must early-exit before stage1 runs; this is invoked by the stage2
        # sharded runner when --jobs>1.
        if args.st37_refine_stage2:
            payload_path = Path(str(args.st37_refine_stage2))
            payload_decoded = _load_worker_stage_payload(
                schema_name="st37_refine_stage2",
                payload_path=payload_path,
            )
            shortlist_local = [
                (cfg, note)
                for cfg, note in (payload_decoded.get("cfg_pairs") or [])
                if isinstance(note, str)
            ]
            risk_variants_local = [
                (over, note)
                for over, note in (payload_decoded.get("risk_variants") or [])
                if isinstance(over, dict)
            ]
            shock_variants_local = [
                (over, note)
                for over, note in (payload_decoded.get("shock_variants") or [])
                if isinstance(over, dict)
            ]

            stage2_plan_all = _build_stage2_plan(
                shortlist_local,
                risk_variants_local=risk_variants_local,
                shock_variants_local=shock_variants_local,
            )
            _run_sharded_stage_worker(
                stage_label="st37_refine stage2",
                worker_raw=args.st37_refine_worker,
                workers_raw=args.st37_refine_workers,
                out_path_raw=str(args.st37_refine_out or ""),
                out_flag_name="st37-refine-out",
                plan_all=stage2_plan_all,
                bars=bars_sig,
                report_every=200,
                heartbeat_sec=heartbeat_sec,
            )
            return

        # Stage 1: sweep v31-style permission gates + TOD window + (SL, short_mult).
        perm_spread_vals = [None, 0.0025, 0.0030, 0.0035, 0.0040]
        perm_spread_down_vals = [None, 0.04, 0.05, 0.06]
        perm_slope_vals = [None, 0.02, 0.03, 0.04]
        signed_slope_variants: list[tuple[dict[str, object], str]] = [
            ({}, "signed=off"),
            ({"ema_slope_signed_min_pct_down": 0.005}, "signed_down>=0.005"),
            ({"ema_slope_signed_min_pct_up": 0.005, "ema_slope_signed_min_pct_down": 0.005}, "signed_both>=0.005"),
        ]
        perm_variants: list[tuple[dict[str, object], str]] = []
        for spread in perm_spread_vals:
            for spread_down in perm_spread_down_vals:
                for slope in perm_slope_vals:
                    for signed_over, signed_note in signed_slope_variants:
                        if spread is None and spread_down is None and slope is None and not signed_over:
                            perm_variants.append(({}, "perm=off"))
                            continue
                        over: dict[str, object] = {
                            "ema_spread_min_pct": spread,
                            "ema_spread_min_pct_down": spread_down,
                            "ema_slope_min_pct": slope,
                        }
                        over.update(signed_over)
                        perm_variants.append((over, f"perm spread={spread} down={spread_down} slope={slope} {signed_note}"))

        tod_variants: list[tuple[int | None, int | None, str]] = [
            (9, 15, "tod=09-15"),
            (9, 16, "tod=09-16"),
            (10, 15, "tod=10-15"),
            (10, 16, "tod=10-16"),
        ]

        sl_vals = (0.03, 0.04)
        short_mult_vals = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)

        def _mk_seed_cfg(seed: dict) -> tuple[ConfigBundle, str]:
            base = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg_seed = _apply_milestone_base(base, strategy=seed["strategy"], filters=seed.get("filters"))
            seed_tag = str(seed.get("group_name") or "").strip() or "seed"
            return cfg_seed, seed_tag

        def _mk_stage1_cfg(
            cfg_seed: ConfigBundle,
            seed_tag: str,
            *,
            perm_over: dict[str, object],
            perm_note: str,
            tod_s: int | None,
            tod_e: int | None,
            tod_note: str,
            sl_pct: float,
            short_mult: float,
        ) -> tuple[ConfigBundle, str]:
            cfg = replace(
                cfg_seed,
                strategy=replace(cfg_seed.strategy, spot_stop_loss_pct=float(sl_pct), spot_short_risk_mult=float(short_mult)),
            )
            over: dict[str, object] = {}
            over.update(perm_over)
            over["entry_start_hour_et"] = tod_s
            over["entry_end_hour_et"] = tod_e
            f = _merge_filters(cfg_seed.strategy.filters, overrides=over)
            cfg = replace(cfg, strategy=replace(cfg.strategy, filters=f))
            note = f"st37 {seed_tag} | {perm_note} | {tod_note} | SL={sl_pct:g} short={short_mult:g}"
            return cfg, note

        def _build_stage1_plan(seed_items: list[dict]) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for seed_idx, seed in enumerate(seed_items):
                cfg_seed, seed_tag = _mk_seed_cfg(seed)
                for perm_idx, (perm_over, perm_note) in enumerate(perm_variants):
                    for tod_idx, (tod_s, tod_e, tod_note) in enumerate(tod_variants):
                        for sl_idx, sl_pct in enumerate(sl_vals):
                            for short_idx, short_mult in enumerate(short_mult_vals):
                                cfg, note = _mk_stage1_cfg(
                                    cfg_seed,
                                    seed_tag,
                                    perm_over=perm_over,
                                    perm_note=perm_note,
                                    tod_s=tod_s,
                                    tod_e=tod_e,
                                    tod_note=tod_note,
                                    sl_pct=float(sl_pct),
                                    short_mult=float(short_mult),
                                )
                                plan.append(
                                    (
                                        cfg,
                                        note,
                                        {
                                            "seed_idx": int(seed_idx),
                                            "perm_idx": int(perm_idx),
                                            "tod_idx": int(tod_idx),
                                            "sl_idx": int(sl_idx),
                                            "short_idx": int(short_idx),
                                        },
                                    )
                                )
            return plan

        if args.st37_refine_stage1:
            payload_path = Path(str(args.st37_refine_stage1))
            payload_decoded = _load_worker_stage_payload(
                schema_name="st37_refine_stage1",
                payload_path=payload_path,
            )
            seeds_local = list(payload_decoded.get("seeds") or [])

            stage1_plan_all = _build_stage1_plan(seeds_local)
            _run_sharded_stage_worker(
                stage_label="st37_refine stage1",
                worker_raw=args.st37_refine_worker,
                workers_raw=args.st37_refine_workers,
                out_path_raw=str(args.st37_refine_out or ""),
                out_flag_name="st37-refine-out",
                plan_all=stage1_plan_all,
                bars=bars_sig,
                report_every=200,
                heartbeat_sec=heartbeat_sec,
            )
            return

        stage1_rows: list[tuple[ConfigBundle, dict, str]] = []
        stage1_total = len(seeds) * len(perm_variants) * len(tod_variants) * len(sl_vals) * len(short_mult_vals)
        print(f"st37_refine: stage1 total={stage1_total} (perm×tod×sl×short)", flush=True)
        report_every = 200

        def _on_stage1_row(cfg: ConfigBundle, row: dict, note: str) -> None:
            stage1_rows.append((cfg, row, note))

        tested_1 = _run_stage_cfg_rows(
            stage_label="st37_refine stage1",
            total=stage1_total,
            jobs_req=int(jobs),
            serial_plan_builder=lambda: _build_stage1_plan(seeds),
            bars=bars_sig,
            report_every=report_every,
            heartbeat_sec=heartbeat_sec,
            on_row=_on_stage1_row,
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name="st37_refine",
                stage_label="st37_refine stage1",
                total=stage1_total,
                jobs=int(jobs),
                payload={
                    "seeds": [
                        {
                            "group_name": str(s.get("group_name") or ""),
                            "strategy": s["strategy"],
                            "filters": s.get("filters"),
                        }
                        for s in seeds
                    ],
                },
                payload_filename="stage1_payload.json",
                temp_prefix="tradebot_st37_refine_1_",
                worker_tmp_prefix="tradebot_st37_refine_stage1_",
                worker_tag="st37:1",
                out_prefix="stage1_out",
                stage_flag="--st37-refine-stage1",
                worker_flag="--st37-refine-worker",
                workers_flag="--st37-refine-workers",
                out_flag="--st37-refine-out",
                strip_flags_with_values=(
                    "--st37-refine-stage1",
                    "--st37-refine-stage2",
                    "--st37-refine-worker",
                    "--st37-refine-workers",
                    "--st37-refine-out",
                    "--st37-refine-run-min-trades",
                ),
                run_min_trades_flag="--st37-refine-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture st37_refine stage1 worker stdout.",
                failure_label="st37_refine stage1 worker",
                missing_label="st37_refine stage1",
                invalid_label="st37_refine stage1",
            ),
            parallel_default_note="st37 stage1",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )

        print(f"st37_refine: stage1 kept={len(stage1_rows)} tested={tested_1}", flush=True)
        if not stage1_rows:
            return

        stage1_shortlist = _rank_cfg_rows(
            stage1_rows,
            scorers=[(_score_row_roi_dd, 30), (_score_row_pnl_dd, 20)],
            limit=30,
        )

        print(f"st37_refine: stage1 shortlist={len(stage1_shortlist)}", flush=True)

        # Stage 2: risk overlays + shock pocket sweeps around the shortlisted configs.
        #
        # Goal: aggressively explore the TR-median + gap-ratio overlays (riskpanic + riskpop),
        # plus plain TR-median hygiene overlays (riskoff), with the full aggressive/defensive
        # sizing ends included (notably riskpop_short_factor=0.0 to hard-block shorts).
        risk_variants = _build_st37_refine_risk_variants()
        shock_variants = _build_st37_refine_shock_variants()

        # st37_refine stage2 worker mode is handled at the top of this sweep (before stage1).

        stage2_rows: list[tuple[ConfigBundle, dict, str]] = []
        stage2_plan = _build_stage2_plan(
            [(cfg, note) for cfg, _row, note in stage1_shortlist],
            risk_variants_local=risk_variants,
            shock_variants_local=shock_variants,
        )
        stage2_total = len(stage2_plan)
        print(f"st37_refine: stage2 total={stage2_total} (risk×shock)", flush=True)

        def _on_stage2_row(cfg: ConfigBundle, row: dict, note: str) -> None:
            stage2_rows.append((cfg, row, note))

        tested_2 = _run_stage_cfg_rows(
            stage_label="st37_refine stage2",
            total=stage2_total,
            jobs_req=int(jobs),
            serial_plan=stage2_plan,
            bars=bars_sig,
            report_every=report_every,
            heartbeat_sec=heartbeat_sec,
            on_row=_on_stage2_row,
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name="st37_refine",
                stage_label="st37_refine stage2",
                total=stage2_total,
                jobs=int(jobs),
                payload={
                    "shortlist": [
                        _encode_cfg_payload(
                            cfg,
                            note=note,
                            note_key="base_note",
                            extra={"seed_tag": str(note.split("|", 1)[0]).strip()},
                        )
                        for cfg, _row, note in stage1_shortlist
                    ],
                    "risk_variants": [{"overrides": risk_over, "note": risk_note} for risk_over, risk_note in risk_variants],
                    "shock_variants": [
                        {"overrides": shock_over, "note": shock_note} for shock_over, shock_note in shock_variants
                    ],
                },
                payload_filename="stage2_payload.json",
                temp_prefix="tradebot_st37_refine_2_",
                worker_tmp_prefix="tradebot_st37_refine_stage2_",
                worker_tag="st37:2",
                out_prefix="stage2_out",
                stage_flag="--st37-refine-stage2",
                worker_flag="--st37-refine-worker",
                workers_flag="--st37-refine-workers",
                out_flag="--st37-refine-out",
                strip_flags_with_values=(
                    "--st37-refine-stage1",
                    "--st37-refine-stage2",
                    "--st37-refine-worker",
                    "--st37-refine-workers",
                    "--st37-refine-out",
                    "--st37-refine-run-min-trades",
                ),
                run_min_trades_flag="--st37-refine-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture st37_refine stage2 worker stdout.",
                failure_label="st37_refine stage2 worker",
                missing_label="st37_refine stage2",
                invalid_label="st37_refine stage2",
            ),
            parallel_default_note="st37 stage2",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )

        print(f"st37_refine: stage2 kept={len(stage2_rows)} tested={tested_2}", flush=True)
        if not stage2_rows:
            return

        stage2_shortlist = _rank_cfg_rows(
            stage2_rows,
            scorers=[(_score_row_roi_dd, 25), (_score_row_pnl_dd, 15)],
            limit=25,
        )

        print(f"st37_refine: stage2 shortlist={len(stage2_shortlist)} (for exit sweep)", flush=True)

        # Stage 3: exit semantics pocket (anchored on v31, plus a small ATR-exit family).
        v31_exit_mode = "pct"
        v31_pt = None
        v31_sl = 0.04
        v31_atr_p = 14
        v31_ptx = 1.5
        v31_slx = 1.0
        v31_exit_time = None
        v31_exit_on_flip = True
        v31_flip_mode = "entry"
        v31_only_profit = True
        v31_hold = 2
        v31_gate_mode = "off"
        if v31_exit:
            try:
                v31_exit_mode = str(v31_exit.get("spot_exit_mode") or "pct").strip().lower()
            except Exception:
                v31_exit_mode = "pct"
            if v31_exit_mode not in ("pct", "atr"):
                v31_exit_mode = "pct"
            v31_pt = v31_exit.get("spot_profit_target_pct")
            try:
                v31_sl = float(v31_exit.get("spot_stop_loss_pct") or v31_sl)
            except (TypeError, ValueError):
                pass
            try:
                v31_atr_p = int(v31_exit.get("spot_atr_period") or v31_atr_p)
            except (TypeError, ValueError):
                pass
            if "spot_pt_atr_mult" in v31_exit and v31_exit.get("spot_pt_atr_mult") is not None:
                try:
                    v31_ptx = float(v31_exit.get("spot_pt_atr_mult"))
                except (TypeError, ValueError):
                    pass
            if "spot_sl_atr_mult" in v31_exit and v31_exit.get("spot_sl_atr_mult") is not None:
                try:
                    v31_slx = float(v31_exit.get("spot_sl_atr_mult"))
                except (TypeError, ValueError):
                    pass
            v31_exit_time = v31_exit.get("spot_exit_time_et")
            v31_exit_on_flip = bool(v31_exit.get("exit_on_signal_flip")) if "exit_on_signal_flip" in v31_exit else True
            v31_flip_mode = str(v31_exit.get("flip_exit_mode") or v31_flip_mode)
            v31_only_profit = (
                bool(v31_exit.get("flip_exit_only_if_profit"))
                if "flip_exit_only_if_profit" in v31_exit
                else v31_only_profit
            )
            try:
                v31_hold = int(v31_exit.get("flip_exit_min_hold_bars") or v31_hold)
            except (TypeError, ValueError):
                pass
            v31_gate_mode = str(v31_exit.get("flip_exit_gate_mode") or v31_gate_mode)

        sl_sweep = sorted({max(0.01, v31_sl - 0.01), v31_sl, v31_sl + 0.01})
        hold_sweep = sorted({max(0, int(v31_hold) - 1), int(v31_hold), int(v31_hold) + 1})
        exit_variants: list[tuple[dict[str, object], str]] = []
        exit_variants.append(
            (
                {
                    "spot_exit_mode": v31_exit_mode,
                    "spot_profit_target_pct": v31_pt,
                    "spot_stop_loss_pct": v31_sl,
                    "spot_atr_period": v31_atr_p,
                    "spot_pt_atr_mult": v31_ptx,
                    "spot_sl_atr_mult": v31_slx,
                    "spot_exit_time_et": v31_exit_time,
                    "exit_on_signal_flip": v31_exit_on_flip,
                    "flip_exit_mode": v31_flip_mode,
                    "flip_exit_only_if_profit": v31_only_profit,
                    "flip_exit_min_hold_bars": v31_hold,
                    "flip_exit_gate_mode": v31_gate_mode,
                },
                f"exit=v31 {v31_exit_mode} stop{v31_sl:g} flip={int(v31_exit_on_flip)} "
                f"mode={v31_flip_mode} only_profit={int(v31_only_profit)} hold={v31_hold}",
            )
        )
        for sl in sl_sweep:
            if abs(float(sl) - float(v31_sl)) < 1e-9:
                continue
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": float(sl),
                        "exit_on_signal_flip": v31_exit_on_flip,
                        "flip_exit_mode": v31_flip_mode,
                        "flip_exit_only_if_profit": v31_only_profit,
                        "flip_exit_min_hold_bars": int(v31_hold),
                        "flip_exit_gate_mode": v31_gate_mode,
                    },
                    f"exit=pct stop{sl:g} mode={v31_flip_mode} only_profit={int(v31_only_profit)} hold={v31_hold}",
                )
            )
        for hold in hold_sweep:
            if int(hold) == int(v31_hold):
                continue
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": float(v31_sl),
                        "exit_on_signal_flip": v31_exit_on_flip,
                        "flip_exit_mode": v31_flip_mode,
                        "flip_exit_only_if_profit": v31_only_profit,
                        "flip_exit_min_hold_bars": int(hold),
                        "flip_exit_gate_mode": v31_gate_mode,
                    },
                    f"exit=pct stop{v31_sl:g} mode={v31_flip_mode} only_profit={int(v31_only_profit)} hold={hold}",
                )
            )
        # ATR exit pocket (no pct PT/SL; uses ATR multipliers). Keeps flip-exit on to limit long holds.
        for atr_p, pt_m, sl_m in (
            (10, 0.90, 1.80),
            (14, 0.75, 1.60),
            (14, 0.80, 1.60),
            (21, 0.70, 1.80),
        ):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "atr",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": None,
                        "spot_atr_period": int(atr_p),
                        "spot_pt_atr_mult": float(pt_m),
                        "spot_sl_atr_mult": float(sl_m),
                        "exit_on_signal_flip": True,
                        "flip_exit_mode": "entry",
                        "flip_exit_only_if_profit": True,
                        "flip_exit_min_hold_bars": 2,
                    },
                    f"exit=ATR({atr_p}) PTx{pt_m:g} SLx{sl_m:g} flipprofit hold=2",
                )
            )

        def _mk_stage3_cfg(base_cfg: ConfigBundle, *, exit_over: dict[str, object]) -> ConfigBundle:
            ptx_raw = (
                exit_over.get("spot_pt_atr_mult")
                if "spot_pt_atr_mult" in exit_over
                else getattr(base_cfg.strategy, "spot_pt_atr_mult", 1.5)
            )
            slx_raw = (
                exit_over.get("spot_sl_atr_mult")
                if "spot_sl_atr_mult" in exit_over
                else getattr(base_cfg.strategy, "spot_sl_atr_mult", 1.0)
            )
            return replace(
                base_cfg,
                strategy=replace(
                    base_cfg.strategy,
                    spot_exit_mode=str(exit_over.get("spot_exit_mode") or base_cfg.strategy.spot_exit_mode),
                    spot_profit_target_pct=exit_over.get("spot_profit_target_pct"),
                    spot_stop_loss_pct=exit_over.get("spot_stop_loss_pct"),
                    spot_atr_period=int(exit_over.get("spot_atr_period") or getattr(base_cfg.strategy, "spot_atr_period", 14)),
                    spot_pt_atr_mult=float(1.5 if ptx_raw is None else ptx_raw),
                    spot_sl_atr_mult=float(1.0 if slx_raw is None else slx_raw),
                    spot_exit_time_et=(
                        exit_over.get("spot_exit_time_et")
                        if "spot_exit_time_et" in exit_over
                        else getattr(base_cfg.strategy, "spot_exit_time_et", None)
                    ),
                    exit_on_signal_flip=bool(
                        exit_over.get("exit_on_signal_flip")
                        if "exit_on_signal_flip" in exit_over
                        else getattr(base_cfg.strategy, "exit_on_signal_flip", True)
                    ),
                    flip_exit_mode=str(exit_over.get("flip_exit_mode") or getattr(base_cfg.strategy, "flip_exit_mode", "entry")),
                    flip_exit_only_if_profit=bool(
                        exit_over.get("flip_exit_only_if_profit")
                        if "flip_exit_only_if_profit" in exit_over
                        else getattr(base_cfg.strategy, "flip_exit_only_if_profit", False)
                    ),
                    flip_exit_min_hold_bars=int(
                        exit_over.get("flip_exit_min_hold_bars") or getattr(base_cfg.strategy, "flip_exit_min_hold_bars", 2)
                    ),
                    flip_exit_gate_mode=str(exit_over.get("flip_exit_gate_mode") or getattr(base_cfg.strategy, "flip_exit_gate_mode", "off")),
                ),
            )

        stage3_plan: list[tuple[ConfigBundle, str, dict]] = []
        for base_idx, (base_cfg, _row, base_note) in enumerate(stage2_shortlist):
            for exit_idx, (exit_over, exit_note) in enumerate(exit_variants):
                cfg = _mk_stage3_cfg(base_cfg, exit_over=exit_over)
                note = f"{base_note} | {exit_note}"
                stage3_plan.append((cfg, note, {"base_idx": int(base_idx), "exit_idx": int(exit_idx)}))

        stage3_rows: list[tuple[ConfigBundle, dict, str]] = []
        stage3_total = len(stage3_plan)
        print(f"st37_refine: stage3 total={stage3_total} (exit pocket)", flush=True)
        tested_3, stage3_rows_new = _run_stage_serial(
            stage_label="st37_refine stage3",
            plan=stage3_plan,
            bars=bars_sig,
            total=stage3_total,
            report_every=report_every,
            heartbeat_sec=heartbeat_sec,
        )
        stage3_rows.extend(stage3_rows_new)

        print(f"st37_refine: stage3 kept={len(stage3_rows)} tested={tested_3}", flush=True)
        final_rows = stage3_rows or stage2_rows or stage1_rows
        rows_only = [r for _, r, _ in final_rows]

        def _score_roi_dd(row: dict) -> tuple:
            return (
                _roi_dd(row),
                float(row.get("pnl") or 0.0),
                float(row.get("win_rate") or 0.0),
                int(row.get("trades") or 0),
            )

        _print_top(rows_only, title="st37_refine — Top by roi/dd%", top_n=int(args.top), sort_key=_score_roi_dd)
        _print_leaderboards(rows_only, title="st37_refine", top_n=int(args.top))

    def _sweep_seeded_refine_bundle() -> None:
        _resolve_seed_milestones_path(
            seed_milestones=args.seed_milestones,
            axis_tag="seeded_refine",
        )
        print("")
        print("=== seeded_refine: unified seeded refinement bundle ===")
        print(
            f"- members={','.join(_SEEDED_REFINEMENT_MEMBER_AXES)}",
            flush=True,
        )
        print("")
        _sweep_champ_refine()
        _sweep_overlay_family()
        _sweep_st37_refine()

    def _sweep_gate_matrix() -> None:
        """Bounded cross-product of major gates (overnight-capable exhaustive discovery)."""
        nonlocal run_calls_total
        perm_pack = {
            "ema_spread_min_pct": 0.003,
            "ema_slope_min_pct": 0.01,
            "ema_spread_min_pct_down": 0.03,
            "ema_slope_signed_min_pct_down": 0.005,
            "rv_min": 0.15,
            "rv_max": 1.0,
            "volume_ratio_min": 1.2,
            "volume_ema_period": 20,
        }

        tick_pack = {
            "tick_gate_mode": "raschke",
            "tick_gate_symbol": "TICK-AMEX",
            "tick_gate_exchange": "AMEX",
            "tick_neutral_policy": "allow",
            "tick_direction_policy": "both",
            "tick_band_ma_period": 10,
            "tick_width_z_lookback": 252,
            "tick_width_z_enter": 1.0,
            "tick_width_z_exit": 0.5,
            "tick_width_slope_lookback": 3,
        }

        shock_pack = {
            "shock_gate_mode": "surf",
            "shock_detector": "daily_atr_pct",
            "shock_daily_atr_period": 14,
            "shock_daily_on_atr_pct": 13.5,
            "shock_daily_off_atr_pct": 13.0,
            "shock_daily_on_tr_pct": 9.0,
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
            "shock_stop_loss_pct_mult": 0.75,
        }

        riskoff_pack = {
            "riskoff_tr5_med_pct": 9.0,
            "riskoff_tr5_lookback_days": 5,
            "riskoff_mode": "hygiene",
            "risk_entry_cutoff_hour_et": 15,
        }
        riskpanic_pack = {
            "riskpanic_tr5_med_pct": 9.0,
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_lookback_days": 5,
            "riskpanic_short_risk_mult_factor": 0.5,
            "risk_entry_cutoff_hour_et": 15,
        }
        riskpop_pack = {
            "riskpop_tr5_med_pct": 9.0,
            "riskpop_pos_gap_ratio_min": 0.6,
            "riskpop_lookback_days": 5,
            "riskpop_long_risk_mult_factor": 1.2,
            "riskpop_short_risk_mult_factor": 0.5,
            "risk_entry_cutoff_hour_et": 15,
        }

        regime2_pack = {
            "regime2_mode": "supertrend",
            "regime2_bar_size": "4 hours",
            "regime2_supertrend_atr_period": 2,
            "regime2_supertrend_multiplier": 0.3,
            "regime2_supertrend_source": "close",
        }

        short_mults = [1.0, 0.2, 0.05, 0.02, 0.01, 0.0]

        def _mk_stage2_cfg(
            seed_cfg: ConfigBundle,
            seed_note: str,
            family: str,
            *,
            perm_on: bool,
            tick_on: bool,
            shock_on: bool,
            riskoff_on: bool,
            riskpanic_on: bool,
            riskpop_on: bool,
            regime2_on: bool,
            short_mult: float,
        ) -> tuple[ConfigBundle, str]:
            filt_over: dict[str, object] = {}
            if perm_on:
                filt_over.update(perm_pack)
            if shock_on:
                filt_over.update(shock_pack)
            if riskoff_on:
                filt_over.update(riskoff_pack)
            if riskpanic_on:
                filt_over.update(riskpanic_pack)
            if riskpop_on:
                filt_over.update(riskpop_pack)
            f = _mk_filters(overrides=filt_over) if filt_over else None

            strat = seed_cfg.strategy
            strat = replace(
                strat,
                filters=f,
                spot_short_risk_mult=float(short_mult),
                tick_gate_mode="off" if not tick_on else str(tick_pack["tick_gate_mode"]),
                tick_gate_symbol=str(tick_pack["tick_gate_symbol"]),
                tick_gate_exchange=str(tick_pack["tick_gate_exchange"]),
                tick_neutral_policy=str(tick_pack["tick_neutral_policy"]),
                tick_direction_policy=str(tick_pack["tick_direction_policy"]),
                tick_band_ma_period=int(tick_pack["tick_band_ma_period"]),
                tick_width_z_lookback=int(tick_pack["tick_width_z_lookback"]),
                tick_width_z_enter=float(tick_pack["tick_width_z_enter"]),
                tick_width_z_exit=float(tick_pack["tick_width_z_exit"]),
                tick_width_slope_lookback=int(tick_pack["tick_width_slope_lookback"]),
            )
            if not regime2_on:
                strat = replace(strat, regime2_mode="off", regime2_bar_size=None)
            else:
                strat = replace(
                    strat,
                    regime2_mode=str(regime2_pack["regime2_mode"]),
                    regime2_bar_size=str(regime2_pack["regime2_bar_size"]),
                    regime2_supertrend_atr_period=int(regime2_pack["regime2_supertrend_atr_period"]),
                    regime2_supertrend_multiplier=float(regime2_pack["regime2_supertrend_multiplier"]),
                    regime2_supertrend_source=str(regime2_pack["regime2_supertrend_source"]),
                )

            cfg = replace(seed_cfg, strategy=strat)
            note = (
                f"{seed_note} | gates="
                f"perm={int(perm_on)} tick={int(tick_on)} shock={int(shock_on)} "
                f"riskoff={int(riskoff_on)} riskpanic={int(riskpanic_on)} riskpop={int(riskpop_on)} "
                f"r2={int(regime2_on)} short_mult={short_mult:g} family={family}"
            )
            return cfg, note

        def _build_stage2_plan(
            seed_triples: list[tuple[ConfigBundle, str, str]],
        ) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for seed_idx, (seed_cfg, seed_note, family) in enumerate(seed_triples):
                for perm_on in (False, True):
                    for tick_on in (False, True):
                        for shock_on in (False, True):
                            for riskoff_on in (False, True):
                                for riskpanic_on in (False, True):
                                    for riskpop_on in (False, True):
                                        for regime2_on in (False, True):
                                            for short_mult in short_mults:
                                                cfg, note = _mk_stage2_cfg(
                                                    seed_cfg,
                                                    seed_note,
                                                    family,
                                                    perm_on=perm_on,
                                                    tick_on=tick_on,
                                                    shock_on=shock_on,
                                                    riskoff_on=riskoff_on,
                                                    riskpanic_on=riskpanic_on,
                                                    riskpop_on=riskpop_on,
                                                    regime2_on=regime2_on,
                                                    short_mult=float(short_mult),
                                                )
                                                plan.append(
                                                    (
                                                        cfg,
                                                        note,
                                                        {
                                                            "seed_idx": int(seed_idx),
                                                            "perm_on": bool(perm_on),
                                                            "tick_on": bool(tick_on),
                                                            "shock_on": bool(shock_on),
                                                            "riskoff_on": bool(riskoff_on),
                                                            "riskpanic_on": bool(riskpanic_on),
                                                            "riskpop_on": bool(riskpop_on),
                                                            "regime2_on": bool(regime2_on),
                                                            "short_mult": float(short_mult),
                                                        },
                                                    )
                                                )
            return plan

        if args.gate_matrix_stage2:
            payload_path = Path(str(args.gate_matrix_stage2))
            payload_decoded = _load_worker_stage_payload(
                schema_name="gate_matrix_stage2",
                payload_path=payload_path,
            )
            seeds_local = list(payload_decoded.get("seed_triples") or [])
            stage2_plan_all = _build_stage2_plan(seeds_local)
            total = len(seeds_local) * 2 * 2 * 2 * 2 * 2 * 2 * 2 * len(short_mults)
            if len(stage2_plan_all) != int(total):
                raise SystemExit(
                    f"gate_matrix stage2 worker internal error: combos={len(stage2_plan_all)} expected={total}"
                )
            _run_sharded_stage_worker(
                stage_label="gate_matrix stage2",
                worker_raw=args.gate_matrix_worker,
                workers_raw=args.gate_matrix_workers,
                out_path_raw=str(args.gate_matrix_out or ""),
                out_flag_name="gate-matrix-out",
                plan_all=stage2_plan_all,
                bars=bars_sig,
                report_every=100,
                heartbeat_sec=0.0,
            )
            return

        bars_sig = _bars_cached(signal_bar_size)

        # Stage 1: seed scan (direction × bias × exit family), with gates OFF.
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                filters=None,
                tick_gate_mode="off",
                regime2_mode="off",
                regime2_bar_size=None,
                spot_exit_time_et=None,
            ),
        )

        direction_variants: list[tuple[str, str, int, str]] = []
        base_preset = str(base.strategy.ema_preset or "").strip()
        base_mode = str(base.strategy.ema_entry_mode or "trend").strip().lower()
        base_confirm = int(base.strategy.entry_confirm_bars or 0)
        if base_preset and base_mode in ("cross", "trend"):
            direction_variants.append((base_preset, base_mode, base_confirm, f"ema={base_preset} {base_mode}"))
        for preset, mode in (
            ("2/4", "cross"),
            ("3/7", "cross"),
            ("3/7", "trend"),
            ("4/9", "cross"),
            ("4/9", "trend"),
            ("5/10", "trend"),
            ("8/21", "trend"),
        ):
            if base_preset and str(base_preset) == str(preset) and str(base_mode) == str(mode):
                continue
            direction_variants.append((str(preset), str(mode), 0, f"ema={preset} {mode}"))

        regimes: list[tuple[str, int, float, str, str]] = []
        for rbar, atr_p, mult, src in (
            ("4 hours", 2, 0.3, "close"),
            ("4 hours", 5, 0.4, "hl2"),
            ("4 hours", 10, 0.8, "hl2"),
            ("4 hours", 14, 1.0, "hl2"),
            ("1 day", 10, 1.0, "hl2"),
            ("1 day", 14, 1.5, "hl2"),
        ):
            regimes.append((str(rbar), int(atr_p), float(mult), str(src), f"ST({atr_p},{mult:g},{src})@{rbar}"))

        exit_variants: list[tuple[str, dict[str, object]]] = [
            (
                "pct",
                {"spot_exit_mode": "pct", "spot_profit_target_pct": 0.01, "spot_stop_loss_pct": 0.03},
            ),
            (
                "pct",
                {"spot_exit_mode": "pct", "spot_profit_target_pct": 0.015, "spot_stop_loss_pct": 0.04},
            ),
            (
                "pct_stop",
                {"spot_exit_mode": "pct", "spot_profit_target_pct": None, "spot_stop_loss_pct": 0.03},
            ),
            (
                "atr",
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.9,
                    "spot_sl_atr_mult": 1.5,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
            ),
            (
                "atr",
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.75,
                    "spot_sl_atr_mult": 1.8,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
            ),
        ]

        stage1_plan: list[tuple[ConfigBundle, str, dict]] = []
        for preset, mode, confirm, dir_note in direction_variants:
            for rbar, atr_p, mult, src, reg_note in regimes:
                for exit_family, exit_over in exit_variants:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            ema_preset=str(preset),
                            ema_entry_mode=str(mode),
                            entry_confirm_bars=int(confirm),
                            filters=None,
                            tick_gate_mode="off",
                            regime2_mode="off",
                            regime2_bar_size=None,
                            regime_mode="supertrend",
                            regime_bar_size=str(rbar),
                            supertrend_atr_period=int(atr_p),
                            supertrend_multiplier=float(mult),
                            supertrend_source=str(src),
                            **exit_over,
                        ),
                    )
                    note = f"{dir_note} | {reg_note} | exit={exit_family}"
                    stage1_plan.append((cfg, note, {"exit_family": str(exit_family)}))

        total = len(stage1_plan)
        _tested, kept = _run_sweep(
            plan=stage1_plan,
            bars=bars_sig,
            total=total,
            progress_label="gate_matrix stage1",
            report_every=50,
        )
        stage1: list[tuple[ConfigBundle, dict, str, str]] = []
        for cfg, row, note, meta_item in kept:
            family = str(meta_item.get("exit_family") or "") if isinstance(meta_item, dict) else ""
            stage1.append((cfg, row, note, family))

        if not stage1:
            print("Gate-matrix: no stage1 seeds eligible (try lowering --min-trades).")
            return

        families = sorted({t[3] for t in stage1})
        seeds: list[tuple[ConfigBundle, dict, str, str]] = []
        for fam in families:
            ranked_family = _rank_cfg_rows_with_meta(
                [t for t in stage1 if t[3] == fam],
                scorers=[(_score_row_pnl_dd, 3), (_score_row_pnl, 2)],
                limit=None,
            )
            for cfg, row, note, meta_family in ranked_family:
                seeds.append((cfg, row, note, str(meta_family or fam)))
        # Keep this bounded.
        max_seeds = 8
        seeds = seeds[:max_seeds]
        print("")
        print(f"Gate-matrix: stage1 candidates={len(stage1)} seeds={len(seeds)} families={families}")

        # Stage 2: gate cross-product around the shortlist.
        rows: list[dict] = []
        total = len(seeds) * 2 * 2 * 2 * 2 * 2 * 2 * 2 * len(short_mults)
        seed_triples = [(seed_cfg, str(seed_note), str(family)) for seed_cfg, _row, seed_note, family in seeds]
        tested_total = _run_stage_cfg_rows(
            stage_label="gate_matrix stage2",
            total=total,
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=200,
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan_builder=lambda: _build_stage2_plan(seed_triples),
            parallel_payloads_builder=lambda: _run_parallel_stage_with_payload(
                axis_name="gate_matrix",
                stage_label="gate_matrix stage2",
                total=total,
                jobs=int(jobs),
                payload={
                    "seeds": [
                        {
                            "strategy": _spot_strategy_payload(seed_cfg, meta=meta),
                            "filters": _filters_payload(seed_cfg.strategy.filters),
                            "seed_note": str(seed_note),
                            "family": str(family),
                        }
                        for seed_cfg, _row, seed_note, family in seeds
                    ],
                },
                payload_filename="stage2_payload.json",
                temp_prefix="tradebot_gate_matrix_",
                worker_tmp_prefix="tradebot_gate_matrix2_",
                worker_tag="gm2",
                out_prefix="stage2_out",
                stage_flag="--gate-matrix-stage2",
                worker_flag="--gate-matrix-worker",
                workers_flag="--gate-matrix-workers",
                out_flag="--gate-matrix-out",
                strip_flags_with_values=(
                    "--gate-matrix-stage2",
                    "--gate-matrix-worker",
                    "--gate-matrix-workers",
                    "--gate-matrix-out",
                    "--gate-matrix-run-min-trades",
                ),
                run_min_trades_flag="--gate-matrix-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error="Failed to capture gate_matrix worker stdout.",
                failure_label="gate_matrix stage2 worker",
                missing_label="gate_matrix stage2",
                invalid_label="gate_matrix stage2",
            ),
            parallel_default_note="gate_matrix stage2",
            parallel_dedupe_by_milestone_key=False,
            record_milestones=True,
        )
        if jobs > 1:
            run_calls_total += int(tested_total)

        _print_leaderboards(rows, title="Gate-matrix sweep (bounded cross-product)", top_n=int(args.top))

    def _sweep_squeeze() -> None:
        """Squeeze a few high-leverage axes from the current champion baseline.

        Targeted (fast): regime2 timeframe, volume gate, and time-of-day windows,
        including small combinations of these axes.
        """
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Stage 1: sweep regime2 timeframe + params (bounded), with no extra filters.
        stage1: list[tuple[ConfigBundle, dict, str]] = []
        stage1.append((base, base_row, "base") if base_row else (base, {}, "base"))
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
        sources = ["close", "hl2"]
        for r2_bar in ("4 hours", "1 day"):
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime2_mode="supertrend",
                                regime2_bar_size=r2_bar,
                                regime2_supertrend_atr_period=int(atr_p),
                                regime2_supertrend_multiplier=float(mult),
                                regime2_supertrend_source=str(src),
                                filters=None,
                                entry_confirm_bars=0,
                            ),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"r2=ST({atr_p},{mult},{src})@{r2_bar}"
                        row["note"] = note
                        stage1.append((cfg, row, note))

        stage1 = [t for t in stage1 if t[1]]
        shortlisted = _rank_cfg_rows(
            stage1,
            scorers=[(_score_row_pnl_dd, 15), (_score_row_pnl, 10)],
        )
        print("")
        print(f"Squeeze sweep: stage1 candidates={len(stage1)} shortlist={len(shortlisted)} (min_trades={run_min_trades})")

        # Stage 2: apply volume + TOD + confirm gates on the shortlist (small combos).
        vol_variants = [
            (None, None, "vol=-"),
            (1.0, 20, "vol>=1.0@20"),
            (1.1, 20, "vol>=1.1@20"),
            (1.2, 20, "vol>=1.2@20"),
            (1.5, 10, "vol>=1.5@10"),
            (1.5, 20, "vol>=1.5@20"),
        ]
        tod_variants = [
            (None, None, "tod=base"),
            (18, 3, "tod=18-03 ET"),
            (9, 16, "tod=9-16 ET"),
            (10, 15, "tod=10-15 ET"),
            (11, 16, "tod=11-16 ET"),
        ]
        confirm_variants = [(0, "confirm=0"), (1, "confirm=1"), (2, "confirm=2")]

        rows: list[dict] = []
        for base_cfg, _, base_note in shortlisted:
            for vratio, vema, v_note in vol_variants:
                for tod_s, tod_e, tod_note in tod_variants:
                    for confirm, confirm_note in confirm_variants:
                        f = _mk_filters(
                            volume_ratio_min=vratio,
                            volume_ema_period=vema,
                            entry_start_hour_et=tod_s,
                            entry_end_hour_et=tod_e,
                        )
                        cfg = replace(
                            base_cfg,
                            strategy=replace(base_cfg.strategy, filters=f, entry_confirm_bars=int(confirm)),
                        )
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"{base_note} | {v_note} | {tod_note} | {confirm_note}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Squeeze sweep (regime2 tf+params → vol/TOD/confirm)", top_n=int(args.top))

    axis = str(args.axis).strip().lower()
    print(
        f"{symbol} spot evolve sweep ({start.isoformat()} -> {end.isoformat()}, use_rth={use_rth}, "
        f"bar_size={signal_bar_size}, offline={offline}, base={args.base}, axis={axis}, "
        f"jobs={jobs}, "
        f"long_only={long_only} realism={'v2' if realism2 else 'off'} "
        f"spread={spot_spread:g} comm={spot_commission:g} comm_min={spot_commission_min:g} "
        f"slip={spot_slippage:g} sizing={sizing_mode} risk={spot_risk_pct:g} max_notional={spot_max_notional_pct:g})"
    )

    if axis == "all" and jobs > 1:
        axis_plan = _axis_mode_plan(mode="axis_all", include_seeded=False)
        _run_axis_plan_parallel_if_requested(
            axis_plan=list(axis_plan),
            jobs_req=int(jobs),
            label="axis=all parallel",
            tmp_prefix="tradebot_axis_all_",
            offline_error="--jobs>1 for --axis all requires --offline (avoid parallel IBKR sessions).",
        )
        return

    axis_registry.update(
        {
        "ema": _sweep_ema,
        "entry_mode": _sweep_entry_mode,
        "combo_fast": _sweep_combo_fast,
        "combo_full": _sweep_combo_full,
        "squeeze": _sweep_squeeze,
        "volume": _sweep_volume,
        "rv": _sweep_rv,
        "tod": _sweep_tod,
        "tod_interaction": _sweep_tod_interaction,
        "perm_joint": _sweep_perm_joint,
        "weekday": _sweep_weekdays,
        "exit_time": _sweep_exit_time,
        "atr": _sweep_atr_exits,
        "atr_fine": _sweep_atr_exits_fine,
        "atr_ultra": _sweep_atr_exits_ultra,
        "r2_atr": _sweep_r2_atr,
        "r2_tod": _sweep_r2_tod,
        "ema_perm_joint": _sweep_ema_perm_joint,
        "tick_perm_joint": _sweep_tick_perm_joint,
        "regime_atr": _sweep_regime_atr,
        "ema_regime": _sweep_ema_regime,
        "chop_joint": _sweep_chop_joint,
        "ema_atr": _sweep_ema_atr,
        "tick_ema": _sweep_tick_ema,
        "ptsl": _sweep_ptsl,
        "hf_scalp": _sweep_hf_scalp,
        "hold": _sweep_hold,
        "spot_short_risk_mult": _sweep_spot_short_risk_mult,
        "orb": _sweep_orb,
        "orb_joint": _sweep_orb_joint,
        "frontier": _sweep_frontier,
        "regime": _sweep_regime,
        "regime2": _sweep_regime2,
        "regime2_ema": _sweep_regime2_ema,
        "joint": _sweep_joint,
        "micro_st": _sweep_micro_st,
        "flip_exit": _sweep_flip_exit,
        "confirm": _sweep_confirm,
        "spread": _sweep_spread,
        "spread_fine": _sweep_spread_fine,
        "spread_down": _sweep_spread_down,
        "slope": _sweep_slope,
        "slope_signed": _sweep_slope_signed,
        "cooldown": _sweep_cooldown,
        "skip_open": _sweep_skip_open,
        "shock": _sweep_shock,
        "risk_overlays": _sweep_risk_overlays,
        "loosen": _sweep_loosen,
        "loosen_atr": _sweep_loosen_atr,
        "tick": _sweep_tick,
        "gate_matrix": _sweep_gate_matrix,
        "seeded_refine": _sweep_seeded_refine_bundle,
        "champ_refine": _sweep_champ_refine,
        "st37_refine": _sweep_st37_refine,
        "shock_alpha_refine": _sweep_shock_alpha_refine,
        "shock_velocity_refine": lambda: _sweep_shock_velocity_refine(wide=False),
        "shock_velocity_refine_wide": lambda: _sweep_shock_velocity_refine(wide=True),
        "shock_throttle_refine": lambda: _sweep_overlay_family(family="shock_throttle_refine"),
        "shock_throttle_tr_ratio": lambda: _sweep_overlay_family(family="shock_throttle_tr_ratio"),
        "shock_throttle_drawdown": lambda: _sweep_overlay_family(family="shock_throttle_drawdown"),
        "riskpanic_micro": lambda: _sweep_overlay_family(family="riskpanic_micro"),
        "overlay_family": _sweep_overlay_family,
        "exit_pivot": _sweep_exit_pivot,
        }
    )

    if axis == "all":
        _run_axis_plan_serial(list(_axis_mode_plan(mode="axis_all", include_seeded=False)), timed=False)
    else:
        fn_obj = axis_registry.get(str(axis))
        fn = fn_obj if callable(fn_obj) else None
        if fn is not None:
            _run_axis_plan_serial([(str(axis), "single", False)], timed=False)

    if int(run_cfg_cache_hits) > 0 and int(run_calls_total) > 0:
        hit_rate = float(run_cfg_cache_hits) / float(run_calls_total)
        print(
            f"run_cfg cache: entries={len(run_cfg_cache)} hits={run_cfg_cache_hits}/{run_calls_total} "
            f"({hit_rate*100.0:0.1f}%) fp_hits={int(run_cfg_fingerprint_hits)}",
            flush=True,
        )
    if bool(run_cfg_persistent_enabled):
        print(
            f"run_cfg persistent cache: path={run_cfg_persistent_path} "
            f"hits={int(run_cfg_persistent_hits)} writes={int(run_cfg_persistent_writes)}",
            flush=True,
        )

    if bool(args.write_milestones) and not bool(milestones_written):
        eligible_new = _collect_milestone_items_from_rows(
            milestone_rows,
            meta=meta,
            min_win=float(args.milestone_min_win),
            min_trades=int(args.milestone_min_trades),
            min_pnl_dd=float(args.milestone_min_pnl_dd),
        )
        out_path = Path(args.milestones_out)
        total = _merge_and_write_milestones(
            out_path=out_path,
            eligible_new=eligible_new,
            merge_existing=bool(args.merge_milestones),
            add_top_pnl_dd=int(args.milestone_add_top_pnl_dd or 0),
            add_top_pnl=int(args.milestone_add_top_pnl or 0),
            symbol=symbol,
            start=start,
            end=end,
            signal_bar_size=signal_bar_size,
            use_rth=use_rth,
            milestone_min_win=float(args.milestone_min_win),
            milestone_min_trades=int(args.milestone_min_trades),
            milestone_min_pnl_dd=float(args.milestone_min_pnl_dd),
        )
        print(f"Wrote {out_path} ({total} eligible presets).")

    if not offline:
        data.disconnect()


# endregion


# region Multiwindow (stability eval / kingmaker)
# NOTE: this code used to live in tradebot/backtest/run_backtest_multitimeframe.py.
# It is consolidated here so spot backtesting has one canonical module.

# region Constants
_MW_WDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
# endregion


# region Parse Helpers
def _weekdays_from_payload(value) -> tuple[int, ...]:
    if not value:
        return (0, 1, 2, 3, 4)
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
            continue
        key = str(item).strip().upper()[:3]
        if key in _MW_WDAYS:
            out.append(_MW_WDAYS[key])
    return tuple(out) if out else (0, 1, 2, 3, 4)
# endregion


# region Payload Conversion
def _spot_leg_from_payload(raw) -> SpotLegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"directional_spot leg must be an object, got: {raw!r}")
    action = str(raw.get("action") or "").strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"directional_spot.action must be BUY/SELL, got: {action!r}")
    qty = int(raw.get("qty") or 1)
    if qty <= 0:
        raise ValueError(f"directional_spot.qty must be positive, got: {qty!r}")
    return SpotLegConfig(action=action, qty=qty)


def _leg_from_payload(raw) -> LegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"leg must be an object, got: {raw!r}")
    action = str(raw.get("action") or "").strip().upper()
    right = str(raw.get("right") or "").strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"leg.action must be BUY/SELL, got: {action!r}")
    if right not in ("PUT", "CALL"):
        raise ValueError(f"leg.right must be PUT/CALL, got: {right!r}")
    moneyness = float(raw.get("moneyness_pct") or 0.0)
    qty = int(raw.get("qty") or 1)
    if qty <= 0:
        raise ValueError(f"leg.qty must be positive, got: {qty!r}")
    return LegConfig(action=action, right=right, moneyness_pct=moneyness, qty=qty)


def _filters_from_payload(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"filters must be an object, got: {raw!r}")
    return _parse_filters(raw)


def _strategy_from_payload(strategy: dict, *, filters: FiltersConfig | None) -> SpotStrategyConfig:
    if not isinstance(strategy, dict):
        raise ValueError(f"strategy must be an object, got: {strategy!r}")

    raw = dict(strategy)
    raw.pop("signal_bar_size", None)
    raw.pop("signal_use_rth", None)
    raw.pop("spot_sec_type", None)
    raw.pop("spot_exchange", None)
    raw.pop("max_open_trades", None)

    entry_days = _weekdays_from_payload(raw.get("entry_days") or [])
    raw["entry_days"] = entry_days

    raw.setdefault("flip_exit_gate_mode", "off")
    raw["filters"] = filters

    # Normalize nested structures back into dataclasses.
    dspot = raw.get("directional_spot")
    if dspot is not None:
        if not isinstance(dspot, dict):
            raise ValueError(f"directional_spot must be an object, got: {dspot!r}")
        parsed: dict[str, SpotLegConfig] = {}
        for k, v in dspot.items():
            key = str(k).strip()
            if not key:
                continue
            parsed[key] = _spot_leg_from_payload(v)
        raw["directional_spot"] = parsed or None

    dlegs = raw.get("directional_legs")
    if dlegs is not None:
        if not isinstance(dlegs, dict):
            raise ValueError(f"directional_legs must be an object, got: {dlegs!r}")
        parsed_dl: dict[str, tuple[LegConfig, ...]] = {}
        for k, legs in dlegs.items():
            key = str(k).strip()
            if not key or not legs:
                continue
            if not isinstance(legs, list):
                continue
            parsed_dl[key] = tuple(_leg_from_payload(l) for l in legs)
        raw["directional_legs"] = parsed_dl or None

    legs = raw.get("legs")
    if legs is not None:
        if not isinstance(legs, list):
            raise ValueError(f"legs must be a list, got: {legs!r}")
        raw["legs"] = tuple(_leg_from_payload(l) for l in legs)

    return SpotStrategyConfig(**raw)


def _mk_bundle(
    *,
    strategy: SpotStrategyConfig,
    start: date,
    end: date,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
) -> ConfigBundle:
    backtest = BacktestConfig(
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        starting_cash=100_000.0,
        risk_free_rate=0.02,
        cache_dir=Path(cache_dir),
        calibration_dir=Path(cache_dir) / "calibration",
        output_dir=Path("backtests/out"),
        calibrate=False,
        offline=bool(offline),
    )
    synthetic = SyntheticConfig(
        rv_lookback=60,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.2,
        iv_floor=0.05,
        term_slope=0.02,
        skew=-0.25,
        min_spread_pct=0.1,
    )
    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


# endregion


# region Evaluation
def _metrics_from_summary(summary) -> dict:
    pnl = float(getattr(summary, "total_pnl", 0.0) or 0.0)
    dd = float(getattr(summary, "max_drawdown", 0.0) or 0.0)
    trades = int(getattr(summary, "trades", 0) or 0)
    win_rate = float(getattr(summary, "win_rate", 0.0) or 0.0)
    roi = float(getattr(summary, "roi", 0.0) or 0.0)
    dd_pct = float(getattr(summary, "max_drawdown_pct", 0.0) or 0.0)
    pnl_dd = pnl / dd if dd > 0 else (math.inf if pnl > 0 else -math.inf if pnl < 0 else 0.0)
    roi_dd = (
        roi / dd_pct
        if dd_pct > 0
        else (math.inf if roi > 0 else -math.inf if roi < 0 else 0.0)
    )
    return {
        "trades": trades,
        "win_rate": win_rate,
        "pnl": pnl,
        "dd": dd,
        "pnl_over_dd": pnl_dd,
        "roi": roi,
        "dd_pct": dd_pct,
        "roi_over_dd_pct": roi_dd,
    }


def _load_bars(
    data: IBKRHistoricalData,
    *,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
) -> list:
    if offline:
        return data.load_cached_bars(
            symbol=symbol,
            exchange=exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
    return data.load_or_fetch_bars(
        symbol=symbol,
        exchange=exchange,
        start=start_dt,
        end=end_dt,
        bar_size=bar_size,
        use_rth=use_rth,
        cache_dir=cache_dir,
    )


def _die_empty_bars(
    *,
    kind: str,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
    offline: bool,
) -> None:
    tag = "rth" if use_rth else "full24"
    expected = _expected_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start_dt=start_dt,
        end_dt=end_dt,
        bar_size=str(bar_size),
        use_rth=use_rth,
    )
    covering = _find_covering_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    print("")
    print(f"[ERROR] No bars returned ({kind}):")
    print(f"- symbol={symbol} exchange={exchange or 'SMART'} bar={bar_size} {tag} offline={offline}")
    print(f"- window={start_dt.date().isoformat()}→{end_dt.date().isoformat()}")
    if expected.exists():
        print(f"- expected_cache={expected} (exists)")
    else:
        print(f"- expected_cache={expected} (missing)")
    if covering is not None and covering != expected:
        print(f"- covering_cache={covering}")
    if offline:
        print("")
        print("Fix:")
        print("- Re-run once without --offline to fetch/populate the cache via IBKR.")
        print("- If the cache file exists but is empty/corrupt, delete it and re-fetch.")
    else:
        print("")
        print("Fix:")
        print("- Verify IB Gateway / TWS is connected and you have market data permissions for this symbol/timeframe.")
        print("- If IBKR returns empty due to pacing/subscription limits, retry or prefetch once then re-run with --offline.")
    raise SystemExit(2)


def _preflight_offline_cache_or_die(
    *,
    symbol: str,
    candidates: list[dict],
    windows: list[tuple[date, date]],
    signal_bar_size: str,
    use_rth: bool,
    cache_dir: Path,
) -> None:
    missing: list[dict] = []
    checked: set[tuple[str, str, str, str, bool]] = set()

    def _require_cached(
        *,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> None:
        key = (
            str(symbol),
            start_dt.date().isoformat(),
            end_dt.date().isoformat(),
            str(bar_size),
            bool(use_rth),
        )
        if key in checked:
            return
        checked.add(key)
        covering = _find_covering_cache_path(
            cache_dir=cache_dir,
            symbol=str(symbol),
            start=start_dt,
            end=end_dt,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
        )
        if covering is None:
            missing.append(
                {
                    "symbol": str(symbol),
                    "bar_size": str(bar_size),
                    "start": start_dt.date().isoformat(),
                    "end": end_dt.date().isoformat(),
                    "use_rth": bool(use_rth),
                    "expected": str(
                        _expected_cache_path(
                            cache_dir=cache_dir,
                            symbol=str(symbol),
                            start_dt=start_dt,
                            end_dt=end_dt,
                            bar_size=str(bar_size),
                            use_rth=bool(use_rth),
                        )
                    ),
                }
            )

    for wstart, wend in windows:
        start_dt = datetime.combine(wstart, time(0, 0))
        end_dt = datetime.combine(wend, time(23, 59))

        # Always required for every candidate (signal bars).
        _require_cached(
            symbol=str(symbol),
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=str(signal_bar_size),
            use_rth=use_rth,
        )

        for cand in candidates:
            strat = cand.get("strategy") or {}

            # Multi-timeframe regime bars when regime is computed on a different bar size.
            regime_mode = str(strat.get("regime_mode", "ema") or "ema").strip().lower()
            regime_bar = str(strat.get("regime_bar_size") or "").strip() or str(signal_bar_size)
            if regime_mode == "supertrend":
                if str(regime_bar) != str(signal_bar_size):
                    _require_cached(
                        symbol=str(symbol),
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=regime_bar,
                        use_rth=use_rth,
                    )
            else:
                if strat.get("regime_ema_preset") and str(regime_bar) != str(signal_bar_size):
                    _require_cached(
                        symbol=str(symbol),
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=regime_bar,
                        use_rth=use_rth,
                    )

            # Regime2 confirm bars (if enabled and on a different timeframe).
            regime2_mode = str(strat.get("regime2_mode", "off") or "off").strip().lower()
            if regime2_mode != "off":
                regime2_bar = str(strat.get("regime2_bar_size") or "").strip() or str(signal_bar_size)
                if str(regime2_bar) != str(signal_bar_size):
                    _require_cached(
                        symbol=str(symbol),
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=regime2_bar,
                        use_rth=use_rth,
                    )

            # Multi-resolution execution bars (e.g. 5 mins) for spot_exec_bar_size.
            exec_size = str(strat.get("spot_exec_bar_size") or "").strip()
            if exec_size and str(exec_size) != str(signal_bar_size):
                _require_cached(
                    symbol=str(symbol),
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=exec_size,
                    use_rth=use_rth,
                )

            # Tick gate warmup bars (1 day, RTH).
            tick_mode = str(strat.get("tick_gate_mode", "off") or "off").strip().lower()
            if tick_mode != "off":
                try:
                    z_lookback = int(strat.get("tick_width_z_lookback") or 252)
                except (TypeError, ValueError):
                    z_lookback = 252
                try:
                    ma_period = int(strat.get("tick_band_ma_period") or 10)
                except (TypeError, ValueError):
                    ma_period = 10
                try:
                    slope_lb = int(strat.get("tick_width_slope_lookback") or 3)
                except (TypeError, ValueError):
                    slope_lb = 3
                tick_warm_days = max(60, z_lookback + ma_period + slope_lb + 5)
                tick_start_dt = start_dt - timedelta(days=tick_warm_days)
                tick_symbol = str(strat.get("tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
                _require_cached(
                    symbol=tick_symbol,
                    start_dt=tick_start_dt,
                    end_dt=end_dt,
                    bar_size="1 day",
                    use_rth=True,
                )

    if not missing:
        return

    print("")
    print("[ERROR] --offline was requested, but required cached bars are missing:")
    for item in missing[:25]:
        tag = "rth" if item["use_rth"] else "full"
        print(
            f"- {item['symbol']} {item['bar_size']} {tag} {item['start']}→{item['end']} "
            f"(expected: {item['expected']})"
        )
    if len(missing) > 25:
        print(f"- … plus {len(missing) - 25} more missing caches")
    print("")
    print("Fix:")
    print("- Re-run without --offline to fetch via IBKR (and populate db/ cache).")
    print("- Or prefetch the missing bars explicitly before running with --offline.")
    raise SystemExit(2)


def _score_key(item: dict) -> tuple:
    return (
        float(item.get("stability_min_roi_dd") or item.get("stability_min_pnl_dd") or float("-inf")),
        float(item.get("stability_min_roi") or item.get("stability_min_pnl") or float("-inf")),
        float(item.get("full_roi_over_dd_pct") or item.get("full_pnl_over_dd") or float("-inf")),
        float(item.get("full_roi") or item.get("full_pnl") or float("-inf")),
        float(item.get("full_win") or 0.0),
        int(item.get("full_trades") or 0),
    )


def _strategy_key(strategy: dict, *, filters: dict | None) -> str:
    return _strategy_fingerprint(strategy, filters=filters)


# endregion


# region CLI
def spot_multitimeframe_main() -> None:
    ap = argparse.ArgumentParser(prog="tradebot.backtest.multitimeframe")
    ap.add_argument("--milestones", required=True, help="Input spot milestones JSON to evaluate.")
    ap.add_argument("--symbol", default="TQQQ", help="Symbol to filter (default: TQQQ).")
    ap.add_argument("--bar-size", default="1 hour", help="Signal bar size filter (default: 1 hour).")
    ap.add_argument("--use-rth", action="store_true", help="Filter to RTH-only strategies.")
    ap.add_argument("--offline", action="store_true", help="Use cached bars only (no IBKR fetch).")
    ap.add_argument("--cache-dir", default="db", help="Bars cache dir (default: db).")
    ap.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto). Requires --offline for >1.")
    ap.add_argument("--top", type=int, default=200, help="How many candidates to evaluate (after sorting).")
    ap.add_argument("--min-trades", type=int, default=200, help="Min trades per window.")
    ap.add_argument(
        "--min-trades-per-year",
        type=float,
        default=None,
        help=(
            "Min trades per year per window (e.g. 500 => 1y>=500, 2y>=1000, 10y>=5000). "
            "Enforced as ceil(window_years * min_trades_per_year)."
        ),
    )
    ap.add_argument("--min-win", type=float, default=0.0, help="Min win rate per window (0..1).")
    ap.add_argument(
        "--max-open",
        type=int,
        default=None,
        help="Deprecated no-op for spot; kept only for CLI compatibility.",
    )
    ap.add_argument(
        "--allow-unlimited-stacking",
        action="store_true",
        default=False,
        help="Deprecated no-op for spot; kept only for CLI compatibility.",
    )
    ap.add_argument(
        "--require-close-eod",
        action="store_true",
        default=False,
        help="Require spot_close_eod=true (forces strategies to close at end of day).",
    )
    ap.add_argument(
        "--require-positive-pnl",
        action="store_true",
        default=False,
        help="Require pnl>0 in every evaluation window.",
    )
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="Evaluation window formatted YYYY-MM-DD:YYYY-MM-DD. Repeatable.",
    )
    ap.add_argument(
        "--include-full",
        action="store_true",
        help="Also evaluate the full window from the milestones payload notes (best-effort).",
    )
    ap.add_argument(
        "--write-top",
        type=int,
        default=0,
        help="Write a small milestones JSON of the top K stability winners (0 disables).",
    )
    ap.add_argument(
        "--out",
        default="backtests/out/multitimeframe_top.json",
        help="Output file for --write-top (default: backtests/out/multitimeframe_top.json).",
    )
    # Internal flags (used by parallel worker sharding).
    ap.add_argument("--multitimeframe-worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-workers", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-out", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()
    if args.max_open is not None:
        print("[compat] --max-open is deprecated and ignored for spot multitimeframe eval.", flush=True)
    if bool(args.allow_unlimited_stacking):
        print("[compat] --allow-unlimited-stacking is deprecated and ignored for spot multitimeframe eval.", flush=True)
    try:
        min_trades_per_year = float(args.min_trades_per_year) if args.min_trades_per_year is not None else None
    except (TypeError, ValueError):
        min_trades_per_year = None
    if min_trades_per_year is not None and min_trades_per_year < 0:
        raise SystemExit("--min-trades-per-year must be >= 0")

    def _default_jobs() -> int:
        detected = os.cpu_count()
        if detected is None:
            return 1
        try:
            detected_i = int(detected)
        except (TypeError, ValueError):
            return 1
        return max(1, detected_i)

    try:
        jobs = int(args.jobs) if args.jobs is not None else 0
    except (TypeError, ValueError):
        jobs = 0
    jobs_eff = _default_jobs() if jobs <= 0 else min(int(jobs), _default_jobs())
    jobs_eff = max(1, int(jobs_eff))

    milestones_path = Path(args.milestones)
    payload = json.loads(milestones_path.read_text())
    groups = payload.get("groups") or []
    symbol = str(args.symbol).strip().upper()
    bar_size = str(args.bar_size).strip().lower()
    use_rth = bool(args.use_rth)

    candidates: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        filters_payload = group.get("filters")
        entries = group.get("entries") or []
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strat = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strat, dict) or not isinstance(metrics, dict):
            continue
        if str(strat.get("instrument", "spot") or "spot").strip().lower() != "spot":
            continue
        if str(strat.get("symbol") or "").strip().upper() != symbol:
            continue
        if str(strat.get("signal_bar_size") or "").strip().lower() != bar_size:
            continue
        if bool(strat.get("signal_use_rth")) != use_rth:
            continue
        candidates.append(
            {
                "group_name": str(group.get("name") or ""),
                "filters": filters_payload,
                "strategy": strat,
                "metrics": metrics,
            }
        )

    if not candidates:
        raise SystemExit(f"No candidates found for {symbol} bar={bar_size} rth={use_rth} in {milestones_path}")

    def _sort_key_seed(item: dict) -> tuple:
        m = item.get("metrics") or {}
        return (
            float(m.get("pnl_over_dd") or float("-inf")),
            float(m.get("pnl") or float("-inf")),
            float(m.get("win_rate") or 0.0),
            int(m.get("trades") or 0),
        )

    candidates = sorted(candidates, key=_sort_key_seed, reverse=True)[: max(1, int(args.top))]
    jobs_eff = max(1, min(int(jobs_eff), len(candidates)))

    windows: list[tuple[date, date]] = []
    for raw in args.window or []:
        windows.append(_parse_window(raw))
    if not windows:
        windows = [
            (_parse_date("2023-01-01"), _parse_date("2024-01-01")),
            (_parse_date("2024-01-01"), _parse_date("2025-01-01")),
            (_parse_date("2025-01-01"), date.today()),
        ]

    cache_dir = Path(args.cache_dir)
    offline = bool(args.offline)
    multiwindow_cache_path = cache_dir / "spot_multiwindow_eval_cache.sqlite3"
    multiwindow_cache_conn: sqlite3.Connection | None = None
    multiwindow_cache_enabled = True
    multiwindow_cache_lock = threading.Lock()
    multiwindow_cache_hits = 0
    multiwindow_cache_writes = 0
    _MULTIWINDOW_CACHE_MISS = object()

    def _multiwindow_cache_conn() -> sqlite3.Connection | None:
        nonlocal multiwindow_cache_conn, multiwindow_cache_enabled
        if not bool(multiwindow_cache_enabled):
            return None
        if multiwindow_cache_conn is not None:
            return multiwindow_cache_conn
        try:
            conn = sqlite3.connect(
                str(multiwindow_cache_path),
                timeout=15.0,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS multiwindow_eval_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            multiwindow_cache_conn = conn
            return conn
        except Exception:
            multiwindow_cache_enabled = False
            multiwindow_cache_conn = None
            return None

    def _multiwindow_windows_signature() -> tuple[tuple[str, str], ...]:
        return tuple((a.isoformat(), b.isoformat()) for a, b in windows)

    def _multiwindow_cache_key(*, strategy_payload: dict, filters_payload: dict | None) -> str:
        raw = {
            "version": str(_MULTIWINDOW_CACHE_ENGINE_VERSION),
            "strategy_key": _strategy_key(strategy_payload, filters=filters_payload),
            "windows": _multiwindow_windows_signature(),
            "min_trades": int(args.min_trades),
            "min_trades_per_year": float(min_trades_per_year) if min_trades_per_year is not None else None,
            "min_win": float(args.min_win),
            "require_close_eod": bool(args.require_close_eod),
            "require_positive_pnl": bool(args.require_positive_pnl),
            "offline": bool(offline),
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _multiwindow_cache_get(*, cache_key: str) -> dict | None | object:
        conn = _multiwindow_cache_conn()
        if conn is None:
            return _MULTIWINDOW_CACHE_MISS
        try:
            with multiwindow_cache_lock:
                row = conn.execute(
                    "SELECT payload_json FROM multiwindow_eval_cache WHERE cache_key=?",
                    (str(cache_key),),
                ).fetchone()
        except Exception:
            return _MULTIWINDOW_CACHE_MISS
        if row is None:
            return _MULTIWINDOW_CACHE_MISS
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            return _MULTIWINDOW_CACHE_MISS
        if payload is None:
            return None
        return dict(payload) if isinstance(payload, dict) else _MULTIWINDOW_CACHE_MISS

    def _multiwindow_cache_set(*, cache_key: str, payload: dict | None) -> None:
        conn = _multiwindow_cache_conn()
        if conn is None:
            return
        try:
            payload_json = json.dumps(payload, sort_keys=True, default=str)
            with multiwindow_cache_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO multiwindow_eval_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (str(cache_key), payload_json, float(pytime.time())),
                )
        except Exception:
            return

    def _required_trades_for_window(wstart: date, wend: date) -> int:
        required = int(args.min_trades)
        if min_trades_per_year is None:
            return required
        days = int((wend - wstart).days) + 1
        years = max(0.0, float(days) / 365.25)
        req_by_year = int(math.ceil(years * float(min_trades_per_year)))
        return max(required, req_by_year)

    def _make_bars_loader(data: IBKRHistoricalData):
        bars_cache: dict[tuple[str, str | None, str, str, str, bool, bool], list] = {}

        def _load_bars_cached(
            *,
            symbol: str,
            exchange: str | None,
            start_dt: datetime,
            end_dt: datetime,
            bar_size: str,
            use_rth: bool,
            offline: bool,
        ) -> list:
            key = (
                str(symbol),
                str(exchange) if exchange is not None else None,
                start_dt.isoformat(),
                end_dt.isoformat(),
                str(bar_size),
                bool(use_rth),
                bool(offline),
            )
            cached = bars_cache.get(key)
            if cached is not None:
                return cached
            bars = _load_bars(
                data,
                symbol=symbol,
                exchange=exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bar_size,
                use_rth=use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )
            bars_cache[key] = bars
            return bars

        return _load_bars_cached

    meta_cache: dict[tuple[str, str | None, bool], ContractMeta] = {}

    def _resolve_meta(bundle: ConfigBundle, *, data: IBKRHistoricalData | None) -> ContractMeta:
        key = (str(bundle.strategy.symbol), bundle.strategy.exchange, bool(offline))
        cached = meta_cache.get(key)
        if cached is not None:
            return cached
        is_future = bundle.strategy.symbol in ("MNQ", "MBT")
        if offline or data is None:
            exchange = "CME" if is_future else "SMART"
            meta = ContractMeta(
                symbol=bundle.strategy.symbol,
                exchange=exchange,
                multiplier=_spot_multiplier(bundle.strategy.symbol, is_future),
                min_tick=0.01,
            )
        else:
            _, resolved = data.resolve_contract(bundle.strategy.symbol, bundle.strategy.exchange)
            meta = ContractMeta(
                symbol=resolved.symbol,
                exchange=resolved.exchange,
                multiplier=_spot_multiplier(bundle.strategy.symbol, is_future, default=resolved.multiplier),
                min_tick=resolved.min_tick,
            )
        meta_cache[key] = meta
        return meta

    def _load_window_context_bars(
        *,
        bundle: ConfigBundle,
        start_dt: datetime,
        end_dt: datetime,
        load_bars_cached,
    ) -> tuple[list | None, list | None, list | None, list | None]:
        base_bar = str(bundle.backtest.bar_size)
        regime_bar = str(getattr(bundle.strategy, "regime_bar_size", "") or "").strip()

        regime_bars = None
        regime_mode = str(getattr(bundle.strategy, "regime_mode", "") or "").strip().lower()
        needs_regime = False
        if regime_bar and regime_bar != base_bar:
            if regime_mode == "supertrend":
                needs_regime = True
            elif bool(getattr(bundle.strategy, "regime_ema_preset", None)):
                needs_regime = True
        if needs_regime:
            regime_bars = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=regime_bar,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not regime_bars:
                _die_empty_bars(
                    kind="regime",
                    cache_dir=cache_dir,
                    symbol=bundle.strategy.symbol,
                    exchange=bundle.strategy.exchange,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=regime_bar,
                    use_rth=bundle.backtest.use_rth,
                    offline=bundle.backtest.offline,
                )

        regime2_bars = None
        regime2_mode = str(getattr(bundle.strategy, "regime2_mode", "off") or "off").strip().lower()
        regime2_bar = str(getattr(bundle.strategy, "regime2_bar_size", "") or "").strip() or base_bar
        if regime2_mode != "off" and regime2_bar != base_bar:
            regime2_bars = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=regime2_bar,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not regime2_bars:
                _die_empty_bars(
                    kind="regime2",
                    cache_dir=cache_dir,
                    symbol=bundle.strategy.symbol,
                    exchange=bundle.strategy.exchange,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=regime2_bar,
                    use_rth=bundle.backtest.use_rth,
                    offline=bundle.backtest.offline,
                )

        tick_bars = None
        tick_mode = str(getattr(bundle.strategy, "tick_gate_mode", "off") or "off").strip().lower()
        if tick_mode != "off":
            try:
                z_lookback = int(getattr(bundle.strategy, "tick_width_z_lookback", 252) or 252)
            except (TypeError, ValueError):
                z_lookback = 252
            try:
                ma_period = int(getattr(bundle.strategy, "tick_band_ma_period", 10) or 10)
            except (TypeError, ValueError):
                ma_period = 10
            try:
                slope_lb = int(getattr(bundle.strategy, "tick_width_slope_lookback", 3) or 3)
            except (TypeError, ValueError):
                slope_lb = 3
            tick_warm_days = max(60, z_lookback + ma_period + slope_lb + 5)
            tick_start_dt = start_dt - timedelta(days=tick_warm_days)
            tick_symbol = str(getattr(bundle.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
            tick_exchange = str(getattr(bundle.strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip()
            tick_bars = load_bars_cached(
                symbol=tick_symbol,
                exchange=tick_exchange,
                start_dt=tick_start_dt,
                end_dt=end_dt,
                bar_size="1 day",
                use_rth=True,
                offline=bundle.backtest.offline,
            )
            if not tick_bars:
                _die_empty_bars(
                    kind="tick_gate",
                    cache_dir=cache_dir,
                    symbol=tick_symbol,
                    exchange=tick_exchange,
                    start_dt=tick_start_dt,
                    end_dt=end_dt,
                    bar_size="1 day",
                    use_rth=True,
                    offline=bundle.backtest.offline,
                )

        exec_bars = None
        exec_size = str(getattr(bundle.strategy, "spot_exec_bar_size", "") or "").strip()
        if exec_size and exec_size != base_bar:
            exec_bars = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=exec_size,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not exec_bars:
                _die_empty_bars(
                    kind="exec",
                    cache_dir=cache_dir,
                    symbol=bundle.strategy.symbol,
                    exchange=bundle.strategy.exchange,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=exec_size,
                    use_rth=bundle.backtest.use_rth,
                    offline=bundle.backtest.offline,
                )
        return regime_bars, regime2_bars, tick_bars, exec_bars

    def _evaluate_candidate_multiwindow(
        cand: dict,
        *,
        load_bars_cached,
        data: IBKRHistoricalData | None,
    ) -> dict | None:
        nonlocal multiwindow_cache_hits, multiwindow_cache_writes
        filters_payload = cand.get("filters")
        strategy_payload = cand["strategy"]
        cache_key = _multiwindow_cache_key(
            strategy_payload=strategy_payload if isinstance(strategy_payload, dict) else {},
            filters_payload=filters_payload if isinstance(filters_payload, dict) else None,
        )
        cached = _multiwindow_cache_get(cache_key=cache_key)
        if cached is not _MULTIWINDOW_CACHE_MISS:
            multiwindow_cache_hits += 1
            return dict(cached) if isinstance(cached, dict) else None

        def _cache_and_return(payload: dict | None) -> dict | None:
            nonlocal multiwindow_cache_writes
            _multiwindow_cache_set(cache_key=cache_key, payload=payload if isinstance(payload, dict) else None)
            multiwindow_cache_writes += 1
            return dict(payload) if isinstance(payload, dict) else None

        filters = _filters_from_payload(filters_payload)
        strat_cfg = _strategy_from_payload(strategy_payload, filters=filters)
        if bool(args.require_close_eod) and not bool(getattr(strat_cfg, "spot_close_eod", False)):
            return _cache_and_return(None)

        sig_bar_size = str(strategy_payload.get("signal_bar_size") or args.bar_size)
        sig_use_rth = (
            use_rth if strategy_payload.get("signal_use_rth") is None else bool(strategy_payload.get("signal_use_rth"))
        )

        per_window: list[dict] = []
        for wstart, wend in windows:
            bundle = _mk_bundle(
                strategy=strat_cfg,
                start=wstart,
                end=wend,
                bar_size=sig_bar_size,
                use_rth=sig_use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )

            start_dt = datetime.combine(bundle.backtest.start, time(0, 0))
            end_dt = datetime.combine(bundle.backtest.end, time(23, 59))
            bars_sig = load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bundle.backtest.bar_size,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not bars_sig:
                _die_empty_bars(
                    kind="signal",
                    cache_dir=cache_dir,
                    symbol=bundle.strategy.symbol,
                    exchange=bundle.strategy.exchange,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    bar_size=str(bundle.backtest.bar_size),
                    use_rth=bundle.backtest.use_rth,
                    offline=bundle.backtest.offline,
                )

            regime_bars, regime2_bars, tick_bars, exec_bars = _load_window_context_bars(
                bundle=bundle,
                start_dt=start_dt,
                end_dt=end_dt,
                load_bars_cached=load_bars_cached,
            )
            summary = _run_spot_backtest_summary(
                bundle,
                bars_sig,
                _resolve_meta(bundle, data=data),
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
                tick_bars=tick_bars,
                exec_bars=exec_bars,
            )
            m = _metrics_from_summary(summary)
            if bool(args.require_positive_pnl) and float(m["pnl"]) <= 0:
                return _cache_and_return(None)
            req_trades = _required_trades_for_window(wstart, wend)
            if m["trades"] < int(req_trades) or m["win_rate"] < float(args.min_win):
                return _cache_and_return(None)
            per_window.append(
                {
                    "start": wstart.isoformat(),
                    "end": wend.isoformat(),
                    **m,
                }
            )

        if not per_window:
            return _cache_and_return(None)
        min_pnl_dd = min(float(x["pnl_over_dd"]) for x in per_window)
        min_pnl = min(float(x["pnl"]) for x in per_window)
        min_roi_dd = min(float(x.get("roi_over_dd_pct") or 0.0) for x in per_window)
        min_roi = min(float(x.get("roi") or 0.0) for x in per_window)
        primary = per_window[0] if per_window else {}
        return _cache_and_return(
            {
            "key": _strategy_key(strategy_payload, filters=filters_payload),
            "strategy": strategy_payload,
            "filters": filters_payload,
            "seed_group_name": cand.get("group_name"),
            "full_trades": int(primary.get("trades") or 0),
            "full_win": float(primary.get("win_rate") or 0.0),
            "full_pnl": float(primary.get("pnl") or 0.0),
            "full_dd": float(primary.get("dd") or 0.0),
            "full_pnl_over_dd": float(primary.get("pnl_over_dd") or 0.0),
            "full_roi": float(primary.get("roi") or 0.0),
            "full_dd_pct": float(primary.get("dd_pct") or 0.0),
            "full_roi_over_dd_pct": float(primary.get("roi_over_dd_pct") or 0.0),
            "stability_min_pnl_dd": min_pnl_dd,
            "stability_min_pnl": min_pnl,
            "stability_min_roi_dd": min_roi_dd,
            "stability_min_roi": min_roi,
            "windows": per_window,
            }
        )

    def _evaluate_candidate_multiwindow_shard(
        *,
        load_bars_cached,
        data: IBKRHistoricalData | None,
        worker_id: int,
        workers: int,
        progress_mode: str,
    ) -> tuple[int, list[dict]]:
        out_rows: list[dict] = []
        tested = 0
        started = pytime.perf_counter()
        report_every = 10
        worker_total = (len(candidates) // int(workers)) + (1 if int(worker_id) < (len(candidates) % int(workers)) else 0)
        for idx, cand in enumerate(candidates, start=1):
            if ((idx - 1) % int(workers)) != int(worker_id):
                continue
            tested += 1
            row = _evaluate_candidate_multiwindow(cand, load_bars_cached=load_bars_cached, data=data)
            if row is not None:
                out_rows.append(row)

            if tested % report_every != 0:
                continue
            line = _progress_line(
                label=(
                    f"multitimeframe worker {worker_id+1}/{workers}"
                    if progress_mode == "worker"
                    else "multitimeframe serial"
                ),
                tested=int(tested),
                total=int(worker_total),
                kept=len(out_rows),
                started_at=started,
                rate_unit="cands/s",
            )
            print(line, flush=True)
        return tested, out_rows

    def _emit_multitimeframe_results(*, out_rows: list[dict], tested_total: int | None = None, workers: int | None = None) -> None:
        out_rows = sorted(out_rows, key=_score_key, reverse=True)
        print("")
        print(f"Multiwindow results: {len(out_rows)} candidates passed filters.")
        print(f"- symbol={symbol} bar={args.bar_size} rth={use_rth} offline={offline}")
        print(f"- windows={', '.join([f'{a.isoformat()}→{b.isoformat()}' for a,b in windows])}")
        extra = f" min_trades_per_year={float(min_trades_per_year):g}" if min_trades_per_year is not None else ""
        print(f"- min_trades={int(args.min_trades)} min_win={float(args.min_win):0.2f}{extra}")
        if tested_total is not None and workers is not None:
            print(f"- workers={int(workers)} tested_total={int(tested_total)}")
        if bool(multiwindow_cache_enabled):
            print(
                f"- eval_cache={multiwindow_cache_path} hits={int(multiwindow_cache_hits)} writes={int(multiwindow_cache_writes)}",
                flush=True,
            )
        print("")

        show = min(20, len(out_rows))
        for rank, item in enumerate(out_rows[:show], start=1):
            st = item["strategy"]
            print(
                f"{rank:2d}. stability(min roi/dd)={item.get('stability_min_roi_dd', 0.0):.2f} "
                f"full roi/dd={item.get('full_roi_over_dd_pct', 0.0):.2f} "
                f"roi={item.get('full_roi', 0.0)*100:.1f}% dd%={item.get('full_dd_pct', 0.0)*100:.1f}% "
                f"pnl={item['full_pnl']:.1f} "
                f"win={item['full_win']*100:.1f}% tr={item['full_trades']} "
                f"ema={st.get('ema_preset')} {st.get('ema_entry_mode')} "
                f"regime={st.get('regime_mode')} rbar={st.get('regime_bar_size')}"
            )

        if int(args.write_top or 0) <= 0:
            return
        top_k = max(1, int(args.write_top))
        now = utc_now_iso_z()
        groups_out: list[dict] = []
        for idx, item in enumerate(out_rows[:top_k], start=1):
            strategy = item["strategy"]
            filters_payload = item.get("filters")
            key = _strategy_key(strategy, filters=filters_payload)
            metrics = {
                "pnl": float(item.get("full_pnl") or 0.0),
                "roi": float(item.get("full_roi") or 0.0),
                "win_rate": float(item.get("full_win") or 0.0),
                "trades": int(item.get("full_trades") or 0),
                "max_drawdown": float(item.get("full_dd") or 0.0),
                "max_drawdown_pct": float(item.get("full_dd_pct") or 0.0),
                "pnl_over_dd": float(item.get("full_pnl_over_dd") or 0.0),
                "roi_over_dd_pct": float(item.get("full_roi_over_dd_pct") or 0.0),
            }
            groups_out.append(
                {
                    "name": f"Spot ({symbol}) KINGMAKER #{idx:02d} roi/dd={metrics['roi_over_dd_pct']:.2f} "
                    f"roi={metrics['roi']*100:.1f}% dd%={metrics['max_drawdown_pct']*100:.1f}% "
                    f"win={metrics['win_rate']*100:.1f}% tr={metrics['trades']} pnl={metrics['pnl']:.1f}",
                    "filters": filters_payload,
                    "entries": [{"symbol": symbol, "metrics": metrics, "strategy": strategy}],
                    "_eval": {
                        "stability_min_pnl_dd": float(item.get("stability_min_pnl_dd") or 0.0),
                        "stability_min_pnl": float(item.get("stability_min_pnl") or 0.0),
                        "stability_min_roi_dd": float(item.get("stability_min_roi_dd") or 0.0),
                        "stability_min_roi": float(item.get("stability_min_roi") or 0.0),
                        "windows": item.get("windows") or [],
                    },
                    "_key": key,
                }
            )
        out_payload = {
            "name": "multitimeframe_top",
            "generated_at": now,
            "source": str(milestones_path),
            "windows": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in windows],
            "groups": groups_out,
        }
        out_path = Path(args.out)
        write_json(out_path, out_payload, sort_keys=False)
        print(f"\nWrote {out_path} (top={top_k}).")

    def _collect_multitimeframe_rows_from_payloads(*, payloads: dict[int, dict]) -> tuple[int, list[dict]]:
        out_rows: list[dict] = []

        def _decode_row(rec: dict) -> dict | None:
            return dict(rec) if isinstance(rec, dict) else None

        def _row_key(row: dict) -> str:
            strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
            filters_payload = row.get("filters") if isinstance(row.get("filters"), dict) else None
            return _strategy_key(strategy, filters=filters_payload)

        tested_total = _collect_parallel_payload_records(
            payloads=payloads,
            records_key="rows",
            tested_key="tested",
            decode_record=_decode_row,
            on_record=lambda row: out_rows.append(dict(row)),
            dedupe_key=_row_key,
        )
        return int(tested_total), out_rows

    if args.multitimeframe_worker is not None:
        if not offline:
            raise SystemExit("multitimeframe worker mode requires --offline (avoid parallel IBKR sessions).")
        out_path_raw = str(args.multitimeframe_out or "").strip()
        if not out_path_raw:
            raise SystemExit("--multitimeframe-out is required for multitimeframe worker mode.")
        out_path = Path(out_path_raw)

        worker_id, workers = _parse_worker_shard(
            args.multitimeframe_worker,
            args.multitimeframe_workers,
            label="multitimeframe",
        )

        _preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )

        data = IBKRHistoricalData()
        _load_bars_cached = _make_bars_loader(data)

        tested, out_rows = _evaluate_candidate_multiwindow_shard(
            load_bars_cached=_load_bars_cached,
            data=data,
            worker_id=int(worker_id),
            workers=int(workers),
            progress_mode="worker",
        )

        out_payload = {"tested": tested, "kept": len(out_rows), "rows": out_rows}
        write_json(out_path, out_payload, sort_keys=False)
        print(f"multitimeframe worker done tested={tested} kept={len(out_rows)} out={out_path}", flush=True)
        return

    if jobs_eff > 1:
        if not offline:
            raise SystemExit("--jobs>1 for multitimeframe requires --offline (avoid parallel IBKR sessions).")

        _preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )

        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags_with_values=("--jobs", "--multitimeframe-worker", "--multitimeframe-workers", "--multitimeframe-out"),
        )

        jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label="multitimeframe",
            jobs=int(jobs_eff),
            total=len(candidates),
            default_jobs=int(jobs_eff),
            offline=bool(offline),
            offline_error="--jobs>1 for multitimeframe requires --offline (avoid parallel IBKR sessions).",
            tmp_prefix="tradebot_multitimeframe_",
            worker_tag="mt",
            out_prefix="multitimeframe_out",
            build_cmd=lambda worker_id, workers_n, out_path: [
                sys.executable,
                "-u",
                "-m",
                "tradebot.backtest",
                "spot_multitimeframe",
                *base_cli,
                "--jobs",
                "1",
                "--multitimeframe-worker",
                str(worker_id),
                "--multitimeframe-workers",
                str(workers_n),
                "--multitimeframe-out",
                str(out_path),
            ],
            capture_error="Failed to capture multitimeframe worker stdout.",
            failure_label="multitimeframe worker",
            missing_label="multitimeframe",
            invalid_label="multitimeframe",
        )

        tested_total, out_rows = _collect_multitimeframe_rows_from_payloads(payloads=payloads)

        _emit_multitimeframe_results(out_rows=out_rows, tested_total=tested_total, workers=jobs_eff)

        return

    data = IBKRHistoricalData()
    _load_bars_cached = _make_bars_loader(data)

    if not offline:
        try:
            data.connect()
        except Exception as exc:
            raise SystemExit(
                "IBKR API connection failed. Start IB Gateway / TWS (or run with --offline after prefetching cached bars)."
            ) from exc

    if offline:
        _preflight_offline_cache_or_die(
            symbol=symbol,
            candidates=candidates,
            windows=windows,
            signal_bar_size=str(args.bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )

    _tested_serial, out_rows = _evaluate_candidate_multiwindow_shard(
        load_bars_cached=_load_bars_cached,
        data=data,
        worker_id=0,
        workers=1,
        progress_mode="serial",
    )

    if not offline:
        data.disconnect()

    _emit_multitimeframe_results(out_rows=out_rows)

multitimeframe_main = spot_multitimeframe_main


# endregion

if __name__ == "__main__":
    # Default execution path: evolution sweeps CLI.
    main()
