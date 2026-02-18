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
import itertools
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import time as pytime
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .cli_utils import (
    parse_date as _parse_date,
    parse_window as _parse_window,
)
from .config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    SpotStrategyConfig,
    SpotLegConfig,
    SyntheticConfig,
    _parse_filters,
)
from .spot_codec import (
    filters_from_payload as _codec_filters_from_payload,
    filters_payload as _codec_filters_payload,
    make_bundle as _codec_make_bundle,
    metrics_from_summary as _codec_metrics_from_summary,
    spot_strategy_payload as _codec_spot_strategy_payload,
    strategy_from_payload as _codec_strategy_from_payload,
)
from .data import ContractMeta, IBKRHistoricalData, ensure_offline_cached_window
from .engine import (
    _run_spot_backtest_summary,
    _spot_multiplier,
    _spot_prepare_summary_series_pack,
    _spot_series_pack_cache_state,
)
from .sweep_fingerprint import (
    _canonicalize_fingerprint_value,
    _strategy_fingerprint,
)
from .sweep_parallel import (
    _collect_parallel_payload_records,
    _parse_worker_shard,
    _progress_line,
    _run_parallel_stage_kernel,
    _run_parallel_worker_specs,
    _strip_flags,
)
from .sweeps import (
    utc_now_iso_z,
    write_json,
)
from ..series import bars_list
from ..series_cache import series_cache_service
from ..time_utils import now_et as _now_et
from ..signals import parse_bar_size
from ..spot.fill_modes import SPOT_FILL_MODE_NEXT_TRADABLE_BAR, normalize_spot_fill_mode

_SERIES_CACHE = series_cache_service()
_SWEEP_BARS_NAMESPACE = "spot.sweeps.bars"
_SWEEP_TICK_NAMESPACE = "spot.sweeps.tick"


# region Cache Helpers
def _require_offline_cache_or_die(
    *,
    data: IBKRHistoricalData,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
) -> None:
    cache_ok, expected, _resolved, missing_ranges, err = ensure_offline_cached_window(
        data=data,
        cache_dir=cache_dir,
        symbol=str(symbol),
        exchange=exchange,
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    if cache_ok:
        return
    tag = "rth" if use_rth else "full24"
    missing_fmt: list[str] = []
    for s, e in missing_ranges:
        if s == e:
            missing_fmt.append(s.isoformat())
        else:
            missing_fmt.append(f"{s.isoformat()}..{e.isoformat()}")
    missing_note = f" missing={';'.join(missing_fmt)}" if missing_fmt else ""
    detail = f" detail={err}" if str(err or "").strip() else ""
    raise SystemExit(
        f"--offline was requested, but cached bars are missing for {symbol} {bar_size} {tag} "
        f"{start_dt.date().isoformat()}→{end_dt.date().isoformat()} "
        f"(expected: {expected}{missing_note}{detail}). "
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


@dataclass(frozen=True)
class AxisSurfaceSpec:
    name: str
    include_in_axis_all: bool = False
    help_text: str = "See source."
    fast_path: str = "yes"
    sharding_class: str = "none"
    coverage_note: str = ""
    parallel_profile_axis_all: str = "single"
    dimensional_cost_source: str = ""
    total_hint_mode: str = ""
    total_hint_static: int | None = None
    total_hint_dims: tuple[str, ...] = ()


_AXIS_SURFACE_SPECS: tuple[AxisSurfaceSpec, ...] = (
    AxisSurfaceSpec("ema", True, "EMA preset timing sweep (direction speed).", total_hint_static=6),
    AxisSurfaceSpec("entry_mode", False, "Entry semantics sweep (cross vs trend) with directional rules.", total_hint_static=6),
    AxisSurfaceSpec(
        "combo_full",
        False,
        "Unified tight Cartesian sweep over centralized combo dimensions.",
        fast_path="partial",
        sharding_class="stage",
        coverage_note="unified Cartesian core (mixed-radix shard ranges)",
        dimensional_cost_source="combo_full_cartesian_tight",
        total_hint_mode="combo_full",
    ),
    AxisSurfaceSpec("volume", True, "Volume gate sweep (ratio threshold x EMA period).", total_hint_static=13),
    AxisSurfaceSpec("rv", True, "Realized volatility gate sweep.", total_hint_static=29),
    AxisSurfaceSpec("tod", True, "Time-of-day gate sweep (ET entry windows).", total_hint_static=29),
    AxisSurfaceSpec("weekday", False, "Weekday entry gating sweep."),
    AxisSurfaceSpec("exit_time", False, "Fixed ET flatten-time sweep.", total_hint_static=7),
    AxisSurfaceSpec("atr", True, "Core ATR exits sweep.", fast_path="no", total_hint_mode="atr_profile"),
    AxisSurfaceSpec("atr_fine", False, "Fine ATR PT/SL pocket sweep.", fast_path="no", total_hint_mode="atr_profile"),
    AxisSurfaceSpec("atr_ultra", False, "Ultra-fine ATR PT/SL micro-grid.", fast_path="no", total_hint_mode="atr_profile"),
    AxisSurfaceSpec("chop_joint", False, "Chop-killer joint sweep (slope x cooldown x skip-open)."),
    AxisSurfaceSpec("ptsl", True, "Fixed-percent PT/SL exits sweep with flip/close_eod semantics (non-ATR)."),
    AxisSurfaceSpec("hf_scalp", False, "HF scalp cadence sweep.", fast_path="partial"),
    AxisSurfaceSpec("hold", True, "Flip-exit minimum-hold-bars sweep.", total_hint_static=7),
    AxisSurfaceSpec("spot_short_risk_mult", True, "Short-side risk multiplier sweep.", total_hint_static=13),
    AxisSurfaceSpec("orb", True, "ORB sweep (open-time, window, target semantics).", fast_path="no"),
    AxisSurfaceSpec("orb_joint", False, "ORB x regime x TICK joint sweep.", fast_path="no"),
    AxisSurfaceSpec("frontier", False, "Frontier sweep over shortlist dimensions.", fast_path="partial"),
    AxisSurfaceSpec("regime", True, "Primary Supertrend regime params/timeframe sweep.", total_hint_mode="regime_profile"),
    AxisSurfaceSpec("regime2", True, "Secondary (regime2) Supertrend params/timeframe sweep.", total_hint_mode="regime2_profile"),
    AxisSurfaceSpec("regime2_ema", False, "Regime2 EMA confirm sweep.", total_hint_static=12),
    AxisSurfaceSpec("joint", True, "Targeted regime x regime2 interaction hunt."),
    AxisSurfaceSpec("micro_st", False, "Micro sweep around current ST/ST2 neighborhood."),
    AxisSurfaceSpec("flip_exit", True, "Flip-exit semantics/gating sweep.", fast_path="partial", total_hint_static=48),
    AxisSurfaceSpec("confirm", True, "Entry confirmation bars sweep.", total_hint_static=4),
    AxisSurfaceSpec("spread", True, "EMA spread quality gate sweep.", total_hint_mode="spread_profile"),
    AxisSurfaceSpec("spread_fine", False, "Fine EMA spread threshold sweep.", total_hint_mode="spread_profile"),
    AxisSurfaceSpec("spread_down", False, "Directional down-side EMA spread gate sweep.", total_hint_mode="spread_profile"),
    AxisSurfaceSpec("slope", True, "EMA slope quality gate sweep.", total_hint_static=6),
    AxisSurfaceSpec("slope_signed", True, "Signed directional EMA slope gate sweep."),
    AxisSurfaceSpec("cooldown", True, "Entry cooldown bars sweep.", total_hint_static=7),
    AxisSurfaceSpec("skip_open", True, "Skip-first-bars-after-open sweep.", total_hint_static=6),
    AxisSurfaceSpec(
        "shock",
        True,
        "Shock detector/mode/threshold sweep (+ monetization and throttle pocket).",
        dimensional_cost_source="shock",
        total_hint_mode="shock_profile",
    ),
    AxisSurfaceSpec("loosen", True, "Loosenings sweep (single-position parity + EOD behavior)."),
    AxisSurfaceSpec("tick", True, "Raschke-style $TICK width gate sweep.", fast_path="no"),
)
_AXIS_CHOICES = tuple(spec.name for spec in _AXIS_SURFACE_SPECS)
_AXIS_ALL_PLAN = tuple(spec.name for spec in _AXIS_SURFACE_SPECS if bool(spec.include_in_axis_all))
_AXIS_INCLUDE_IN_AXIS_ALL_DEFAULTS = frozenset(_AXIS_ALL_PLAN)
_COMBO_FULL_CARTESIAN_DIM_ORDER: tuple[str, ...] = (
    "timing_profile",
    "direction",
    "confirm",
    "perm",
    "tod",
    "vol",
    "cadence",
    "regime",
    "regime2",
    "exit",
    "tick",
    "shock",
    "slope",
    "risk",
    "short_mult",
)
_COMBO_FULL_PAIR_DIM_VARIANT_SPECS: tuple[tuple[str, str], ...] = (
    ("direction", "direction_variants"),
    ("perm", "perm_variants"),
    ("tod", "tod_variants"),
    ("vol", "vol_variants"),
    ("cadence", "cadence_variants"),
    ("regime", "regime_variants"),
    ("regime2", "regime2_variants"),
    ("exit", "exit_variants"),
    ("tick", "tick_variants"),
    ("shock", "shock_variants"),
    ("slope", "slope_variants"),
    ("risk", "risk_variants"),
)
_COMBO_FULL_NOTE_PAIR_DIM_ORDER: tuple[str, ...] = (
    "perm",
    "tod",
    "vol",
    "cadence",
    "regime",
    "regime2",
    "exit",
    "tick",
    "shock",
    "slope",
    "risk",
)
_COMBO_FULL_COVERAGE_TIER_REGISTRY: dict[str, dict[str, object]] = {
    "full": {"freeze_dims": (), "hint_axis": "combo_full", "hint_mode": "combo_full"},
    "profile": {
        "freeze_dims": (
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "customizer": "hf_timing_sniper",
        "hint_axis": "hf_timing_sniper",
        "hint_mode": "combo_subset",
        "hint_dims": ("timing_profile",),
        "hint_static": 6,
    },
    "gate": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "hint_axis": "perm_joint",
        "hint_mode": "combo_subset",
        "hint_dims": ("perm", "tod", "vol", "cadence"),
    },
    "ema": {
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "hint_axis": "ema_perm_joint",
        "hint_mode": "combo_subset",
        "hint_dims": ("direction", "perm", "tod", "vol"),
    },
    "tick": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "hint_axis": "tick_perm_joint",
        "hint_mode": "combo_subset",
        "hint_dims": ("tick", "perm", "tod", "vol"),
    },
    "regime": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime2",
            "tick",
            "shock",
            "slope",
            "risk",
            "short_mult",
        ),
        "hint_axis": "regime_atr",
        "hint_mode": "combo_subset",
        "hint_dims": ("regime", "exit"),
    },
    "risk": {
        "freeze_dims": (
            "timing_profile",
            "direction",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "shock",
            "slope",
            "short_mult",
        ),
        "customizer": "risk_overlays",
        "hint_axis": "risk_overlays",
        "hint_mode": "combo_subset",
        "hint_dims": ("risk",),
    },
}
_COMBO_FULL_PRESET_ALIAS_REGISTRY: dict[str, dict[str, object]] = {
    "squeeze": {
        "tier": "regime",
        "customizer": "squeeze",
        "freeze_dims": ("timing_profile", "direction", "perm", "cadence", "regime", "exit", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_static": 23130,
    },
    "tod_interaction": {
        "tier": "gate",
        "customizer": "tod_interaction",
        "freeze_dims": ("timing_profile", "direction", "confirm", "perm", "vol", "regime", "regime2", "exit", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_static": 81,
    },
    "perm_joint": {"tier": "gate"},
    "ema_perm_joint": {"tier": "ema"},
    "tick_perm_joint": {"tier": "tick"},
    "regime_atr": {"tier": "regime"},
    "ema_regime": {
        "tier": "ema",
        "customizer": "ema_regime",
        "freeze_dims": ("timing_profile", "confirm", "perm", "tod", "vol", "cadence", "regime2", "exit", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("direction", "regime"),
        "hint_static": 22,
    },
    "tick_ema": {
        "tier": "tick",
        "customizer": "tick_ema",
        "freeze_dims": ("timing_profile", "confirm", "perm", "tod", "vol", "cadence", "regime", "regime2", "exit", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("direction", "tick"),
        "hint_static": 505,
    },
    "ema_atr": {
        "tier": "ema",
        "customizer": "ema_atr",
        "freeze_dims": ("timing_profile", "confirm", "perm", "tod", "vol", "cadence", "regime", "regime2", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("direction", "exit"),
        "hint_static": 1009,
    },
    "r2_atr": {
        "tier": "regime",
        "customizer": "r2_atr",
        "freeze_dims": ("timing_profile", "direction", "confirm", "perm", "tod", "vol", "cadence", "regime", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("regime2", "exit"),
        "hint_static": 289,
    },
    "r2_tod": {
        "tier": "regime",
        "customizer": "r2_tod",
        "freeze_dims": ("timing_profile", "direction", "confirm", "perm", "vol", "cadence", "regime", "exit", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("regime2", "tod"),
        "hint_static": 29,
    },
    "loosen_atr": {
        "tier": "risk",
        "customizer": "loosen_atr",
        "freeze_dims": ("timing_profile", "direction", "confirm", "perm", "tod", "vol", "cadence", "regime", "regime2", "tick", "shock", "slope", "risk", "short_mult"),
        "hint_dims": ("exit",),
        "hint_static": 151,
    },
    "risk_overlays": {"tier": "risk", "customizer": "risk_overlays"},
    "gate_matrix": {
        "tier": "gate",
        "customizer": "gate_matrix",
        "freeze_dims": ("timing_profile", "direction", "confirm", "regime", "exit", "slope", "vol", "cadence"),
        "hint_mode": "gate_matrix",
    },
    "lf_shock_sniper": {
        "tier": "risk",
        "customizer": "lf_shock_sniper",
        "freeze_dims": (
            "timing_profile",
            "confirm",
            "perm",
            "tod",
            "vol",
            "cadence",
            "regime",
            "regime2",
            "exit",
            "tick",
            "slope",
            "risk",
            "short_mult",
        ),
        "hint_dims": ("direction", "shock"),
        "hint_static": 10,
    },
    "hf_timing_sniper": {
        "tier": "profile",
        "customizer": "hf_timing_sniper",
        "hint_dims": ("timing_profile",),
        "hint_static": 6,
    },
}
_COMBO_FULL_PRESET_TIER_NAMES: tuple[str, ...] = tuple(_COMBO_FULL_COVERAGE_TIER_REGISTRY.keys())
_COMBO_FULL_PRESET_ALIAS_NAMES: tuple[str, ...] = tuple(_COMBO_FULL_PRESET_ALIAS_REGISTRY.keys())
_COMBO_FULL_PRESET_AXES: tuple[str, ...] = tuple(_COMBO_FULL_PRESET_TIER_NAMES + _COMBO_FULL_PRESET_ALIAS_NAMES)
_AXIS_HANDLER_NAME_OVERRIDES: dict[str, str] = {
    "weekday": "_sweep_weekdays",
    "atr": "_sweep_atr_exits",
    "atr_fine": "_sweep_atr_exits_fine",
    "atr_ultra": "_sweep_atr_exits_ultra",
}
_UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS = sorted(set(_AXIS_HANDLER_NAME_OVERRIDES) - set(_AXIS_CHOICES))
if _UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS:
    raise ValueError(f"Unknown axis handler override keys: {_UNKNOWN_AXIS_HANDLER_OVERRIDE_KEYS}")


def _dedupe_axis_names(names) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        key = str(raw).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _axis_handler_name(axis_name: str) -> str:
    key = str(axis_name).strip().lower()
    if not key:
        raise ValueError("axis_name must be non-empty")
    override = _AXIS_HANDLER_NAME_OVERRIDES.get(key)
    if override:
        return str(override)
    return f"_sweep_{key}"


def _build_axis_registry_from_scope(local_scope: dict[str, object]) -> dict[str, object]:
    registry: dict[str, object] = {}
    missing: list[str] = []
    for axis_name in _AXIS_CHOICES:
        handler_name = _axis_handler_name(str(axis_name))
        fn_obj = local_scope.get(handler_name)
        if callable(fn_obj):
            registry[str(axis_name)] = fn_obj
        else:
            missing.append(f"{axis_name}->{handler_name}")
    if missing:
        raise RuntimeError(f"Axis handlers missing: {', '.join(missing)}")
    return registry


def _axis_mode_plan(*, mode: str) -> tuple[tuple[str, str, bool], ...]:
    mode_key = str(mode).strip().lower()
    if mode_key != "axis_all":
        raise ValueError(f"Unknown axis mode: {mode!r}")

    def _profile_for_axis(axis_name: str) -> str:
        spec_map = globals().get("_AXIS_EXECUTION_SPEC_BY_NAME")
        if not isinstance(spec_map, dict):
            return "single"
        spec = spec_map.get(str(axis_name))
        if spec is None:
            return "single"
        return str(getattr(spec, "parallel_profile_axis_all", "single") or "single")

    spec_map = globals().get("_AXIS_EXECUTION_SPEC_BY_NAME")
    if not isinstance(spec_map, dict):
        return tuple((axis_name, "single", True) for axis_name in _AXIS_ALL_PLAN)
    out: list[tuple[str, str, bool]] = []
    for axis_name in _AXIS_CHOICES:
        spec = spec_map.get(str(axis_name))
        if not isinstance(spec, AxisExecutionSpec) or not bool(spec.include_in_axis_all):
            continue
        out.append((str(axis_name), _profile_for_axis(str(axis_name)), True))
    return tuple(out)
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
_AXIS_DIMENSION_REGISTRY: dict[str, dict[str, object]] = {
    "perm_joint": {
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
        "perm_variants": (
            ("perm=base", {}),
            (
                "perm=off",
                {
                    "ema_spread_min_pct": None,
                    "ema_slope_min_pct": None,
                    "ema_spread_min_pct_down": None,
                    "ema_slope_signed_min_pct_up": None,
                    "ema_slope_signed_min_pct_down": None,
                },
            ),
            (
                "perm=loose 0.0015/0.01/0.02",
                {"ema_spread_min_pct": 0.0015, "ema_slope_min_pct": 0.01, "ema_spread_min_pct_down": 0.02},
            ),
            (
                "perm=loose_slope 0.0015/0.03/0.02",
                {"ema_spread_min_pct": 0.0015, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.02},
            ),
            (
                "perm=mid 0.003/0.03/0.04",
                {"ema_spread_min_pct": 0.003, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.04},
            ),
            (
                "perm=spready 0.006/0.03/0.08",
                {"ema_spread_min_pct": 0.006, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.08},
            ),
            (
                "perm=tight_slope 0.003/0.06/0.08",
                {"ema_spread_min_pct": 0.003, "ema_slope_min_pct": 0.06, "ema_spread_min_pct_down": 0.08},
            ),
            (
                "perm=tight 0.006/0.06/0.08",
                {"ema_spread_min_pct": 0.006, "ema_slope_min_pct": 0.06, "ema_spread_min_pct_down": 0.08},
            ),
            (
                "perm=signed_down",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                    "ema_slope_signed_min_pct_down": 0.005,
                },
            ),
            (
                "perm=signed_both",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.04,
                    "ema_slope_signed_min_pct_up": 0.005,
                    "ema_slope_signed_min_pct_down": 0.005,
                },
            ),
            ("perm=fine spread>=0.002", {"ema_spread_min_pct": 0.002}),
            ("perm=fine spread>=0.004", {"ema_spread_min_pct": 0.004}),
            ("perm=fine spread>=0.007", {"ema_spread_min_pct": 0.007}),
        ),
        "vol_variants": (
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.0@20", {"volume_ratio_min": 1.0, "volume_ema_period": 20}),
            ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
            ("vol>=1.5@20", {"volume_ratio_min": 1.5, "volume_ema_period": 20}),
        ),
        "cadence_variants": (
            ("cad=base", {}),
            ("cad=skip1 cd2", {"skip_first_bars": 1, "cooldown_bars": 2}),
            ("cad=skip2 cd2", {"skip_first_bars": 2, "cooldown_bars": 2}),
        ),
        "cost_hints": {"tod_windows": 1.2, "perm_variants": 1.0, "vol_variants": 0.6, "cadence_variants": 0.4},
    },
    "gate_matrix": {
        "perm_variants": (
            ("perm=off", {}),
            (
                "perm=core",
                {
                    "ema_spread_min_pct": 0.003,
                    "ema_slope_min_pct": 0.01,
                    "ema_spread_min_pct_down": 0.03,
                    "ema_slope_signed_min_pct_down": 0.005,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.2,
                    "volume_ema_period": 20,
                },
            ),
            (
                "perm=loose",
                {
                    "ema_spread_min_pct": 0.0015,
                    "ema_slope_min_pct": 0.03,
                    "ema_spread_min_pct_down": 0.02,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.0,
                    "volume_ema_period": 20,
                },
            ),
            (
                "perm=tight",
                {
                    "ema_spread_min_pct": 0.006,
                    "ema_slope_min_pct": 0.06,
                    "ema_spread_min_pct_down": 0.08,
                    "ema_slope_signed_min_pct_up": 0.005,
                    "ema_slope_signed_min_pct_down": 0.005,
                    "rv_min": 0.15,
                    "rv_max": 1.0,
                    "volume_ratio_min": 1.2,
                    "volume_ema_period": 20,
                },
            ),
        ),
        "tod_variants": (
            ("tod=off", None, None),
            ("tod=09-16 ET", 9, 16),
            ("tod=10-15 ET", 10, 15),
            ("tod=18-04 ET", 18, 4),
        ),
        "short_mults": (1.0, 0.2, 0.05, 0.02, 0.01, 0.0),
        "cost_hints": {"perm_variants": 1.0, "tod_variants": 0.8, "short_mults": 0.4},
    },
    "shock": {
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
        "advanced_modes": ("detect", "surf", "block_longs"),
        "advanced_detectors": (
            (
                {
                    "shock_detector": "tr_ratio",
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_on_ratio": 1.30,
                    "shock_off_ratio": 1.20,
                    "shock_min_atr_pct": 5.0,
                },
                "det=tr_ratio 3/21 on=1.30 off=1.20 min=5",
            ),
            (
                {
                    "shock_detector": "tr_ratio",
                    "shock_atr_fast_period": 5,
                    "shock_atr_slow_period": 50,
                    "shock_on_ratio": 1.35,
                    "shock_off_ratio": 1.25,
                    "shock_min_atr_pct": 7.0,
                },
                "det=tr_ratio 5/50 on=1.35 off=1.25 min=7",
            ),
            (
                {
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 13.5,
                    "shock_daily_off_atr_pct": 13.0,
                    "shock_daily_on_tr_pct": 9.0,
                },
                "det=daily_atr p14 on=13.5 off=13 tr_on=9",
            ),
        ),
        "advanced_short_risk_factors": (1.0, 2.0, 5.0, 12.0),
        "advanced_long_down_factors": (1.0, 0.7, 0.4, 0.0),
        "advanced_scales": (
            ({}, "scale=off"),
            (
                {"shock_risk_scale_target_atr_pct": 12.0, "shock_risk_scale_min_mult": 0.2},
                "scale=atr12 min=0.2",
            ),
            (
                {
                    "shock_scale_detector": "tr_ratio",
                    "shock_atr_fast_period": 3,
                    "shock_atr_slow_period": 21,
                    "shock_risk_scale_target_atr_pct": 0.45,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_risk_scale_apply_to": "both",
                },
                "scale=tr_ratio 3/21 target=0.45 min=0.2 both",
            ),
            (
                {
                    "shock_scale_detector": "daily_drawdown",
                    "shock_drawdown_lookback_days": 20,
                    "shock_risk_scale_target_atr_pct": 8.0,
                    "shock_risk_scale_min_mult": 0.2,
                    "shock_risk_scale_apply_to": "both",
                },
                "scale=daily_drawdown lb=20 target=8 min=0.2 both",
            ),
        ),
        "cost_hints": {"modes": 1.0, "detectors": 1.2, "advanced": 1.4},
    },
    "risk_overlays": {
        "riskoff_trs": (6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        "riskoff_lbs": (3, 5, 7, 10),
        "riskoff_modes": ("hygiene", "directional"),
        "riskoff_cutoffs_et": (None, 15, 16),
        "panic_trs": (2.75, 3.0, 3.25, 8.0, 9.0, 10.0, 12.0),
        "panic_neg_ratios": (0.5, 0.6, 0.8),
        "panic_lbs": (5, 10),
        "panic_short_factors": (1.0, 0.5, 0.2, 0.0),
        "panic_long_factors": (None, 1.0, 0.4, 0.0),
        "panic_cutoffs_et": (None, 15, 16),
        "panic_neg_gap_abs_pcts": (None, 0.005, 0.01, 0.02),
        "panic_tr_delta_variants": (
            (None, 1, "trΔ=off"),
            (0.25, 1, "trΔ>=0.25@1d"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (0.75, 1, "trΔ>=0.75@1d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ),
        "pop_trs": (7.0, 8.0, 9.0, 10.0, 12.0),
        "pop_pos_ratios": (0.5, 0.6, 0.8),
        "pop_lbs": (5, 10),
        "pop_long_factors": (0.6, 0.8, 1.0, 1.2, 1.5),
        "pop_short_factors": (1.0, 0.5, 0.2, 0.0),
        "pop_cutoffs_et": (None, 15),
        "pop_modes": ("hygiene", "directional"),
        "pop_pos_gap_abs_pcts": (None, 0.01, 0.02),
        "pop_tr_delta_variants": (
            (None, 1, "trΔ=off"),
            (0.5, 1, "trΔ>=0.5@1d"),
            (0.5, 5, "trΔ>=0.5@5d"),
            (1.0, 1, "trΔ>=1.0@1d"),
            (1.0, 5, "trΔ>=1.0@5d"),
        ),
        "cost_hints": {"riskoff": 1.0, "riskpanic": 1.3, "riskpop": 1.2},
    },
    "combo_full_cartesian_tight": {
        # Unified tight Cartesian for combo_full: all dimensions are crossed in one core.
        "direction_variants": (
            ("ema=2/4 cross", {"entry_signal": "ema", "ema_preset": "2/4", "ema_entry_mode": "cross"}),
            ("ema=4/9 cross", {"entry_signal": "ema", "ema_preset": "4/9", "ema_entry_mode": "cross"}),
        ),
        "confirm_bars": (0, 1),
        "perm_variants": (
            (
                "perm=off",
                {
                    "ema_spread_min_pct": None,
                    "ema_slope_min_pct": None,
                    "ema_spread_min_pct_down": None,
                    "ema_slope_signed_min_pct_up": None,
                    "ema_slope_signed_min_pct_down": None,
                },
            ),
            (
                "perm=mid 0.003/0.03/0.04",
                {"ema_spread_min_pct": 0.003, "ema_slope_min_pct": 0.03, "ema_spread_min_pct_down": 0.04},
            ),
            (
                "perm=tight 0.006/0.06/0.08",
                {"ema_spread_min_pct": 0.006, "ema_slope_min_pct": 0.06, "ema_spread_min_pct_down": 0.08},
            ),
        ),
        "tod_variants": (
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
        ),
        "vol_variants": (
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
        ),
        "cadence_variants": (
            ("cad=base", {}),
            ("cad=skip1 cd2", {"skip_first_bars": 1, "cooldown_bars": 2}),
        ),
        "regime_variants": (
            (
                "regime=ST(4h:7,0.5,hl2)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 7,
                    "supertrend_multiplier": 0.5,
                    "supertrend_source": "hl2",
                },
            ),
            (
                "regime=ST(1d:14,1.0,hl2)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "1 day",
                    "supertrend_atr_period": 14,
                    "supertrend_multiplier": 1.0,
                    "supertrend_source": "hl2",
                },
            ),
            (
                "regime=ST(4h:10,0.8,close)",
                {
                    "regime_mode": "supertrend",
                    "regime_bar_size": "4 hours",
                    "supertrend_atr_period": 10,
                    "supertrend_multiplier": 0.8,
                    "supertrend_source": "close",
                },
            ),
        ),
        "regime2_variants": (
            ("r2=off", {"regime2_mode": "off", "regime2_bar_size": None}),
            (
                "r2=ST(4h:3,0.25,close)",
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
            ),
        ),
        "exit_variants": (
            (
                "exit=pct(0.015,0.03)",
                {
                    "spot_exit_mode": "pct",
                    "spot_profit_target_pct": 0.015,
                    "spot_stop_loss_pct": 0.03,
                    "spot_pt_atr_mult": None,
                    "spot_sl_atr_mult": None,
                },
            ),
            (
                "exit=stop_only(0.03)",
                {
                    "spot_exit_mode": "pct",
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": 0.03,
                    "spot_pt_atr_mult": None,
                    "spot_sl_atr_mult": None,
                },
            ),
            (
                "exit=atr(14,0.8,1.6)",
                {
                    "spot_exit_mode": "atr",
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.8,
                    "spot_sl_atr_mult": 1.6,
                },
            ),
        ),
        "tick_variants": (
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=raschke",
                {
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
                },
            ),
        ),
        "shock_variants": (
            ("shock=off", {"shock_gate_mode": "off"}),
            (
                "shock=surf_daily",
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "daily_atr_pct",
                    "shock_daily_atr_period": 14,
                    "shock_daily_on_atr_pct": 13.5,
                    "shock_daily_off_atr_pct": 13.0,
                    "shock_daily_on_tr_pct": 9.0,
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_stop_loss_pct_mult": 0.75,
                    "shock_profit_target_pct_mult": 1.0,
                },
            ),
            (
                "shock=surf_atr_ratio",
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "atr_ratio",
                    "shock_atr_fast_period": 7,
                    "shock_atr_slow_period": 50,
                    "shock_on_ratio": 1.55,
                    "shock_off_ratio": 1.30,
                    "shock_min_atr_pct": 7.0,
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_stop_loss_pct_mult": 0.75,
                    "shock_profit_target_pct_mult": 1.0,
                },
            ),
            (
                "shock=surf_tr_ratio",
                {
                    "shock_gate_mode": "surf",
                    "shock_detector": "tr_ratio",
                    "shock_atr_fast_period": 7,
                    "shock_atr_slow_period": 50,
                    "shock_on_ratio": 1.55,
                    "shock_off_ratio": 1.30,
                    "shock_min_atr_pct": 7.0,
                    "shock_direction_source": "signal",
                    "shock_direction_lookback": 1,
                    "shock_stop_loss_pct_mult": 0.75,
                    "shock_profit_target_pct_mult": 1.0,
                },
            ),
        ),
        "slope_variants": (
            ("slope=off", {}),
            ("slope>=0.01", {"ema_slope_min_pct": 0.01}),
        ),
        "risk_variants": (
            ("risk=off", {}),
            (
                "risk=riskoff9",
                {
                    "riskoff_tr5_med_pct": 9.0,
                    "riskoff_lookback_days": 5,
                    "riskoff_mode": "hygiene",
                    "risk_entry_cutoff_hour_et": 15,
                    "riskpanic_tr5_med_pct": None,
                    "riskpanic_neg_gap_ratio_min": None,
                },
            ),
            (
                "risk=riskpanic9",
                {
                    "riskoff_tr5_med_pct": None,
                    "riskpanic_tr5_med_pct": 9.0,
                    "riskpanic_neg_gap_ratio_min": 0.6,
                    "riskpanic_lookback_days": 5,
                    "riskpanic_short_risk_mult_factor": 0.5,
                    "risk_entry_cutoff_hour_et": 15,
                },
            ),
            (
                "risk=riskpop9",
                {
                    "riskoff_tr5_med_pct": None,
                    "riskpanic_tr5_med_pct": None,
                    "riskpanic_neg_gap_ratio_min": None,
                    "riskpop_tr5_med_pct": 9.0,
                    "riskpop_pos_gap_ratio_min": 0.6,
                    "riskpop_lookback_days": 5,
                    "riskpop_long_risk_mult_factor": 1.2,
                    "riskpop_short_risk_mult_factor": 0.5,
                    "risk_entry_cutoff_hour_et": 15,
                    "riskoff_mode": "hygiene",
                },
            ),
        ),
        # Timing/rats profiles: base by default; HF presets can expand this dimension.
        "timing_profile_variants": (
            (
                "timing=base",
                {
                    "strategy_overrides": {},
                    "filter_overrides": {},
                },
            ),
        ),
        "hf_profile_variants": (
            (
                "timing=hf_symm_v10",
                {
                    "strategy_overrides": {
                        "ema_preset": "3/7",
                        "ema_entry_mode": "trend",
                        "entry_confirm_bars": 0,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": 0.026,
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 7,
                        "supertrend_multiplier": 0.4,
                        "supertrend_source": "close",
                        "tick_gate_mode": "off",
                        "spot_exec_bar_size": "1 min",
                    },
                    "filter_overrides": {
                        "ema_spread_min_pct": 0.00075,
                        "ema_spread_min_pct_down": 0.014,
                        "ema_slope_min_pct": 0.004,
                        "shock_gate_mode": "detect",
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_short_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor_down": 1.0,
                        "shock_stop_loss_pct_mult": 1.0,
                        "shock_profit_target_pct_mult": 1.0,
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 4.0,
                        "shock_daily_off_atr_pct": 4.0,
                        "shock_daily_on_tr_pct": 4.0,
                        "shock_on_drawdown_pct": -20.0,
                        "shock_off_drawdown_pct": -10.0,
                        "shock_scale_detector": "tr_ratio",
                        "shock_drawdown_lookback_days": 10,
                        "shock_risk_scale_target_atr_pct": 12.0,
                        "shock_risk_scale_min_mult": 0.2,
                        "shock_risk_scale_apply_to": "both",
                        "shock_atr_fast_period": 5,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.30,
                        "shock_off_ratio": 1.25,
                        "shock_min_atr_pct": 4.0,
                        "riskpanic_tr5_med_pct": 2.25,
                        "riskpanic_neg_gap_ratio_min": 0.65,
                        "riskpanic_neg_gap_abs_pct_min": 0.005,
                        "riskpanic_lookback_days": 6,
                        "riskpanic_tr5_med_delta_min_pct": 0.5,
                        "riskpanic_tr5_med_delta_lookback_days": 1,
                        "riskpanic_long_risk_mult_factor": 0.0,
                        "riskpanic_short_risk_mult_factor": 1.0,
                        "riskpanic_long_scale_mode": "linear",
                        "riskpanic_long_scale_tr_delta_max_pct": None,
                        "entry_permission_mode": "off",
                        "entry_time_gate_mode": "off",
                        "ratsv_enabled": True,
                        "ratsv_slope_window_bars": 5,
                        "ratsv_tr_fast_bars": 5,
                        "ratsv_tr_slow_bars": 20,
                        "ratsv_branch_a_rank_min": 0.0240,
                        "ratsv_branch_a_tr_ratio_min": 0.96,
                        "ratsv_branch_a_slope_med_min_pct": 0.000024,
                        "ratsv_branch_a_slope_vel_min_pct": 0.000012,
                        "ratsv_branch_a_cross_age_max_bars": 4,
                        "ratsv_branch_b_rank_min": 0.22,
                        "ratsv_branch_b_tr_ratio_min": 1.12,
                        "ratsv_branch_b_slope_med_min_pct": 0.00085,
                        "ratsv_branch_b_slope_vel_min_pct": 0.00006,
                        "ratsv_branch_b_cross_age_max_bars": 4,
                        "ratsv_probe_cancel_max_bars": 5,
                        "ratsv_probe_cancel_slope_adverse_min_pct": 0.00029,
                        "ratsv_probe_cancel_tr_ratio_min": 0.95,
                        "ratsv_adverse_release_min_hold_bars": 1,
                        "ratsv_adverse_release_slope_adverse_min_pct": 0.0007,
                        "ratsv_adverse_release_tr_ratio_min": 1.0325,
                    },
                },
            ),
            (
                "timing=hf_symm_v9",
                {
                    "strategy_overrides": {
                        "ema_preset": "3/7",
                        "ema_entry_mode": "trend",
                        "entry_confirm_bars": 0,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": 0.026,
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 7,
                        "supertrend_multiplier": 0.4,
                        "supertrend_source": "close",
                        "tick_gate_mode": "off",
                        "spot_exec_bar_size": "1 min",
                    },
                    "filter_overrides": {
                        "ema_spread_min_pct": 0.00075,
                        "ema_spread_min_pct_down": 0.014,
                        "ema_slope_min_pct": 0.004,
                        "shock_gate_mode": "detect",
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_short_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor": 1.0,
                        "shock_long_risk_mult_factor_down": 1.0,
                        "shock_stop_loss_pct_mult": 1.0,
                        "shock_profit_target_pct_mult": 1.0,
                        "shock_daily_atr_period": 14,
                        "shock_daily_on_atr_pct": 4.0,
                        "shock_daily_off_atr_pct": 4.0,
                        "shock_daily_on_tr_pct": 4.0,
                        "shock_on_drawdown_pct": -20.0,
                        "shock_off_drawdown_pct": -10.0,
                        "shock_scale_detector": "tr_ratio",
                        "shock_drawdown_lookback_days": 10,
                        "shock_risk_scale_target_atr_pct": 12.0,
                        "shock_risk_scale_min_mult": 0.2,
                        "shock_risk_scale_apply_to": "both",
                        "shock_atr_fast_period": 5,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": 1.30,
                        "shock_off_ratio": 1.25,
                        "shock_min_atr_pct": 4.0,
                        "riskpanic_tr5_med_pct": 2.25,
                        "riskpanic_neg_gap_ratio_min": 0.65,
                        "riskpanic_neg_gap_abs_pct_min": 0.005,
                        "riskpanic_lookback_days": 6,
                        "riskpanic_tr5_med_delta_min_pct": 0.5,
                        "riskpanic_tr5_med_delta_lookback_days": 1,
                        "riskpanic_long_risk_mult_factor": 0.0,
                        "riskpanic_short_risk_mult_factor": 1.0,
                        "riskpanic_long_scale_mode": "linear",
                        "riskpanic_long_scale_tr_delta_max_pct": None,
                        "entry_permission_mode": "off",
                        "entry_time_gate_mode": "off",
                        "ratsv_enabled": True,
                        "ratsv_slope_window_bars": 5,
                        "ratsv_tr_fast_bars": 5,
                        "ratsv_tr_slow_bars": 20,
                        "ratsv_branch_a_rank_min": 0.0245,
                        "ratsv_branch_a_tr_ratio_min": 0.96,
                        "ratsv_branch_a_slope_med_min_pct": 0.000024,
                        "ratsv_branch_a_slope_vel_min_pct": 0.000012,
                        "ratsv_branch_a_cross_age_max_bars": 5,
                        "ratsv_branch_b_rank_min": 0.22,
                        "ratsv_branch_b_tr_ratio_min": 1.12,
                        "ratsv_branch_b_slope_med_min_pct": 0.00085,
                        "ratsv_branch_b_slope_vel_min_pct": 0.00006,
                        "ratsv_branch_b_cross_age_max_bars": 4,
                        "ratsv_probe_cancel_max_bars": 5,
                        "ratsv_probe_cancel_slope_adverse_min_pct": 0.00029,
                        "ratsv_probe_cancel_tr_ratio_min": 0.95,
                        "ratsv_adverse_release_min_hold_bars": 1,
                        "ratsv_adverse_release_slope_adverse_min_pct": 0.0007,
                        "ratsv_adverse_release_tr_ratio_min": 1.0325,
                    },
                },
            ),
        ),
        "short_mults": (1.0, 0.2),
        # Dominant ordering used by mixed-radix shard ranges.
        "dominant_dims": ("timing_profile", "tick", "regime", "regime2", "risk", "shock", "exit", "slope"),
        "cost_hints": {
            "timing_profile": 1.8,
            "tick": 1.4,
            "regime": 1.3,
            "regime2": 1.2,
            "risk": 1.5,
            "shock": 1.3,
            "exit": 1.2,
            "direction": 1.0,
            "perm": 1.1,
            "tod": 1.0,
            "vol": 0.8,
            "cadence": 0.7,
            "confirm": 0.8,
            "slope": 0.9,
            "short_mult": 0.7,
        },
    },
    "cost_model": {
        "base": 1.0,
        "regime_cross_tf": 0.5,
        "regime2_cross_tf": 0.5,
        "tick_gate_on": 0.75,
        "exec_cross_tf": 0.75,
        "perm_gate_on": 0.15,
        "tod_gate_on": 0.12,
        "volume_gate_on": 0.08,
        "cadence_gate_on": 0.05,
        "shock_gate_on": 0.4,
        "riskoff_overlay_on": 0.2,
        "riskpanic_overlay_on": 0.35,
        "riskpop_overlay_on": 0.3,
    },
    "cache": {
        "dimension_keys": (
            "ema_entry_mode",
            "entry_confirm_bars",
            "spot_short_risk_mult",
            "spot_exit_mode",
            "spot_profit_target_pct",
            "spot_stop_loss_pct",
            "spot_atr_period",
            "spot_pt_atr_mult",
            "spot_sl_atr_mult",
            "spot_close_eod",
            "exit_on_signal_flip",
            "flip_exit_mode",
            "flip_exit_only_if_profit",
            "flip_exit_min_hold_bars",
            "flip_exit_gate_mode",
            "regime_mode",
            "regime_bar_size",
            "supertrend_atr_period",
            "supertrend_multiplier",
            "supertrend_source",
            "regime2_mode",
            "regime2_bar_size",
            "regime2_supertrend_atr_period",
            "regime2_supertrend_multiplier",
            "regime2_supertrend_source",
            "tick_gate_mode",
            "tick_gate_symbol",
            "tick_gate_exchange",
            "tick_neutral_policy",
            "tick_direction_policy",
            "tick_width_z_lookback",
            "tick_width_z_enter",
            "tick_width_z_exit",
            "tick_width_slope_lookback",
            "ema_spread_min_pct",
            "ema_spread_min_pct_down",
            "ema_slope_min_pct",
            "ema_slope_signed_min_pct_up",
            "ema_slope_signed_min_pct_down",
            "volume_ratio_min",
            "volume_ema_period",
            "entry_start_hour_et",
            "entry_end_hour_et",
            "cooldown_bars",
            "skip_first_bars",
            "shock_gate_mode",
            "shock_detector",
            "shock_direction_source",
            "shock_direction_lookback",
            "shock_scale_detector",
            "shock_stop_loss_pct_mult",
            "shock_profit_target_pct_mult",
            "shock_short_risk_mult_factor",
            "shock_long_risk_mult_factor",
            "shock_long_risk_mult_factor_down",
            "shock_risk_scale_target_atr_pct",
            "shock_risk_scale_min_mult",
            "shock_risk_scale_apply_to",
            "riskoff_tr5_med_pct",
            "riskoff_tr5_lookback_days",
            "riskoff_mode",
            "riskoff_long_risk_mult_factor",
            "riskoff_short_risk_mult_factor",
            "riskpanic_tr5_med_pct",
            "riskpanic_neg_gap_ratio_min",
            "riskpanic_neg_gap_abs_pct_min",
            "riskpanic_lookback_days",
            "riskpanic_tr5_med_delta_min_pct",
            "riskpanic_tr5_med_delta_lookback_days",
            "riskpanic_long_risk_mult_factor",
            "riskpanic_short_risk_mult_factor",
            "riskpanic_long_scale_mode",
            "risk_entry_cutoff_hour_et",
            "riskpop_tr5_med_pct",
            "riskpop_pos_gap_ratio_min",
            "riskpop_pos_gap_abs_pct_min",
            "riskpop_lookback_days",
            "riskpop_tr5_med_delta_min_pct",
            "riskpop_tr5_med_delta_lookback_days",
            "riskpop_long_risk_mult_factor",
            "riskpop_short_risk_mult_factor",
            "entry_permission_mode",
            "entry_time_gate_mode",
            "ratsv_enabled",
            "ratsv_slope_window_bars",
            "ratsv_tr_fast_bars",
            "ratsv_tr_slow_bars",
            "ratsv_branch_a_rank_min",
            "ratsv_branch_a_tr_ratio_min",
            "ratsv_branch_a_slope_med_min_pct",
            "ratsv_branch_a_slope_vel_min_pct",
            "ratsv_branch_a_cross_age_max_bars",
            "ratsv_branch_b_rank_min",
            "ratsv_branch_b_tr_ratio_min",
            "ratsv_branch_b_slope_med_min_pct",
            "ratsv_branch_b_slope_vel_min_pct",
            "ratsv_branch_b_cross_age_max_bars",
            "ratsv_probe_cancel_max_bars",
            "ratsv_probe_cancel_slope_adverse_min_pct",
            "ratsv_probe_cancel_tr_ratio_min",
            "ratsv_adverse_release_min_hold_bars",
            "ratsv_adverse_release_slope_adverse_min_pct",
            "ratsv_adverse_release_tr_ratio_min",
            "spot_exec_bar_size",
        ),
        "stage_frontier": {
            # Conservative dominance rule: require repeated misses before pruning.
            "min_eval_count": 3,
            "max_keep_count": 0,
            "max_best_pnl": 0.0,
            "max_best_pnl_over_dd": 0.0,
        },
        "winner_projection": {
            # Run-min-trades agnostic dominance envelope.
            "min_eval_count": 8,
            "max_keep_count": 0,
            "max_best_pnl": 0.0,
            "max_best_pnl_over_dd": 0.0,
        },
        "dimension_value_utility": {
            # Minimum sample count before treating a dimension-value row as predictive.
            "min_eval_count": 6,
            # Weights for utility scoring used by cartesian planning/sharding.
            "weight_keep_rate": 0.70,
            "weight_hit_rate": 0.20,
            "weight_confidence": 0.10,
            # Confidence normalizer in eval-count units.
            "confidence_eval_scale": 24.0,
            # Floor for eval seconds in utility denominator.
            "eval_sec_floor": 0.01,
            # Skip low-cardinality stage writes; utility hints are noisy there.
            "write_min_total": 128,
            # Sampling factor for very large stages (1 = write every eval).
            "write_sample_mod": 1,
        },
        "dimension_upper_bound": {
            # Minimum observations before an upper-bound row can influence ordering.
            "min_eval_count": 6,
            # Low-ceiling frontier (<= both) gets deferred toward tail.
            "low_ceiling_max_keep_count": 0,
            "low_ceiling_max_best_pnl": 0.0,
            "low_ceiling_max_best_pnl_over_dd": 0.0,
            # Confidence normalization for scoring.
            "confidence_eval_scale": 24.0,
            # Skip low-cardinality stage writes; upper-bound stats are low-signal there.
            "write_min_total": 96,
            # Sampling factor for very large stages (1 = write every eval).
            "write_sample_mod": 1,
        },
        "run_cfg_persistent": {
            # Enable RAM-first batching in worker-stage processes.
            "ram_first_worker": 1,
            # Flush when pending batch reaches this size.
            "batch_write_size": 256,
            # Flush at least this often (seconds), even under low throughput.
            "batch_write_interval_sec": 2.0,
        },
        "series_pack_prewarm": {
            # Build/reuse summary series packs once per unique key in worker loops.
            "enabled": 1,
            # Only activate prewarm logic for meaningful stage sizes.
            "min_total": 128,
            # Guardrail for unique per-stage pack contexts.
            "max_unique": 4096,
        },
        "planner_heartbeat": {
            # Parent monitor cadence for worker-heartbeat aggregation output.
            "monitor_interval_sec": 30.0,
            # Mark worker stale if heartbeat age exceeds this threshold.
            "stale_after_sec": 180.0,
            # Grace period for workers that have not written their first heartbeat row.
            "bootstrap_grace_sec": 180.0,
            # How many stale recycle attempts to allow before failing stage.
            "max_stale_retries": 1,
        },
        "claim_first_planner": {
            # Force serial runs of large stages into worker-claim mode.
            "enabled": 1,
            "serial_force_worker": 1,
            "min_total": 512,
            "stage_labels": ("combo_full_cartesian", "shock"),
        },
        "jobs_tuner": {
            # Auto-downshift worker fanout for tiny stages to reduce spawn overhead.
            "enabled": 1,
            "min_items_per_worker": 64,
            # 0 means "no explicit max", only clamp by detected CPUs and stage size.
            "max_workers": 0,
            # Soft caps discovered from local benchmarks:
            # <=256 items scales best around 3 workers; <=4096 around 8 workers.
            # Larger stages are unconstrained (except by CPU count / explicit max_workers).
            "soft_max_workers_by_total": ((256, 3), (4096, 8)),
        },
        "claim_span_tuner": {
            # Adaptive claim span for dynamic-claim workers (balance overhead vs stragglers).
            "enabled": 1,
            "target_claims_per_worker": 24,
            "min_claim_span": 32,
            "max_claim_span": 2048,
            "max_batch_multiple": 8,
        },
        "cartesian_rank_manifest": {
            # Trigger compaction only when row volume is meaningful.
            "compact_min_rows": 1024,
            # Avoid repeated compaction churn on the same stage/window key.
            "compact_min_interval_sec": 120.0,
        },
        "stage_unresolved_summary": {
            # Recompute unresolved spans when summary row is older than this TTL.
            "ttl_sec": 21600.0,
        },
        "rank_dominance_stamp": {
            # Trigger stamp compaction only when signature/range rows are meaningful.
            "compact_min_rows": 512,
            # Avoid repeated stamp compaction churn on the same stage/window key.
            "compact_min_interval_sec": 120.0,
            # Discard stale stamp rows older than this TTL window.
            "ttl_sec": 1209600.0,
        },
    },
}

_PERM_JOINT_PROFILE: dict[str, tuple] = {
    "tod_windows": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("tod_windows") or ()),
    "perm_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("perm_variants") or ()),
    "vol_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("vol_variants") or ()),
    "cadence_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("perm_joint", {}).get("cadence_variants") or ()),
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
    "modes": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("modes") or ()),
    "dir_variants": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("dir_variants") or ()),
    "sl_mults": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("sl_mults") or ()),
    "pt_mults": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("pt_mults") or ()),
    "short_risk_factors": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("short_risk_factors") or ()),
    "ratio_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("ratio_rows") or ()),
    "daily_atr_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("daily_atr_rows") or ()),
    "drawdown_rows": tuple(_AXIS_DIMENSION_REGISTRY.get("shock", {}).get("drawdown_rows") or ()),
}
_RUN_CFG_CACHE_ENGINE_VERSION = "spot_stage_v8"
_RANK_BIN_SIZE = 2048
_AXIS_DIMENSION_FINGERPRINT_KEYS: tuple[str, ...] = tuple(
    str(k)
    for k in (tuple(_AXIS_DIMENSION_REGISTRY.get("cache", {}).get("dimension_keys") or ()))
    if str(k).strip()
)


def _axis_dimension_fingerprint(cfg: ConfigBundle) -> str:
    # Use a full strategy+filters fingerprint to avoid false cache collisions
    # when new knobs are introduced but not yet mirrored in narrow dimension key lists.
    strategy = asdict(cfg.strategy)
    filters_payload = _filters_payload(cfg.strategy.filters) or {}
    strategy.pop("filters", None)
    dims = {
        "strategy": _canonicalize_fingerprint_value(strategy),
        "filters": _canonicalize_fingerprint_value(filters_payload),
    }
    return json.dumps(dims, sort_keys=True, default=str)


def _window_signature(
    *,
    bars_sig: tuple[int, object | None, object | None],
    regime_sig: tuple[int, object | None, object | None],
    regime2_sig: tuple[int, object | None, object | None],
) -> str:
    raw = {
        "bars": tuple(bars_sig),
        "regime": tuple(regime_sig),
        "regime2": tuple(regime2_sig),
    }
    return json.dumps(_canonicalize_fingerprint_value(raw), sort_keys=True, default=str)


def _combo_full_dimension_space_signature(
    *,
    ordered_dims: tuple[str, ...] | list[str],
    size_by_dim: dict[str, int],
    timing_profile_variants: list[tuple[str, dict[str, object], dict[str, object]]],
    confirm_bars: list[int],
    pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]],
    short_mults: list[float],
) -> str:
    """Stable fingerprint for combo_full Cartesian variant space.

    Rank-manifest cache keys must change when variant payload values change, even
    if dimension cardinalities stay constant.
    """
    dim_rows: list[tuple[str, tuple[object, ...]]] = []
    for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER:
        key = str(dim_name)
        if key == "timing_profile":
            rows = tuple(
                (
                    str(label),
                    _canonicalize_fingerprint_value(dict(strat_over) if isinstance(strat_over, dict) else {}),
                    _canonicalize_fingerprint_value(dict(filt_over) if isinstance(filt_over, dict) else {}),
                )
                for label, strat_over, filt_over in tuple(timing_profile_variants or ())
            )
        elif key == "confirm":
            rows = tuple(int(v) for v in tuple(confirm_bars or ()))
        elif key == "short_mult":
            rows = tuple(float(v) for v in tuple(short_mults or ()))
        else:
            rows = tuple(
                (
                    str(label),
                    _canonicalize_fingerprint_value(dict(payload) if isinstance(payload, dict) else {}),
                )
                for label, payload in tuple(pair_variants_by_dim.get(str(key)) or ())
            )
        dim_rows.append((str(key), tuple(rows)))
    raw = {
        "ordered_dims": tuple(str(v) for v in tuple(ordered_dims or ())),
        "size_by_dim": tuple(
            (str(dim_name), int(size_by_dim.get(str(dim_name), 0) or 0))
            for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
        ),
        "dim_rows": tuple(dim_rows),
    }
    return hashlib.sha1(
        json.dumps(_canonicalize_fingerprint_value(raw), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _registry_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _cache_config(section: str) -> dict[str, object]:
    cache_cfg = _AXIS_DIMENSION_REGISTRY.get("cache", {})
    if not isinstance(cache_cfg, dict):
        return {}
    raw = cache_cfg.get(str(section))
    return dict(raw) if isinstance(raw, dict) else {}


def _cfg_label_set(raw: object) -> set[str]:
    if not isinstance(raw, (tuple, list, set)):
        return set()
    out: set[str] = set()
    for item in raw:
        key = str(item or "").strip().lower()
        if key:
            out.add(str(key))
    return out


def _claim_first_stage_enabled(*, stage_label: str, total: int) -> bool:
    cfg = _cache_config("claim_first_planner")
    if not bool(_registry_float(cfg.get("enabled"), 1.0) > 0.0):
        return False
    min_total = max(1, int(_registry_float(cfg.get("min_total"), 512.0)))
    if int(total) < int(min_total):
        return False
    labels = _cfg_label_set(cfg.get("stage_labels"))
    if not labels:
        return True
    return str(stage_label or "").strip().lower() in labels


def _claim_first_serial_force_worker_enabled() -> bool:
    cfg = _cache_config("claim_first_planner")
    return bool(_registry_float(cfg.get("serial_force_worker"), 1.0) > 0.0)


def _tuned_parallel_jobs(
    *,
    stage_label: str,
    jobs_requested: int,
    total: int,
    default_jobs: int,
) -> int:
    jobs_req_i = max(1, int(jobs_requested))
    total_i = max(0, int(total))
    default_i = max(1, int(default_jobs))
    jobs_eff = min(int(jobs_req_i), int(default_i), int(total_i)) if int(total_i) > 0 else 1
    jobs_eff = max(1, int(jobs_eff))
    cfg = _cache_config("jobs_tuner")
    if not bool(_registry_float(cfg.get("enabled"), 1.0) > 0.0):
        return int(jobs_eff)
    min_items = max(1, int(_registry_float(cfg.get("min_items_per_worker"), 64.0)))
    max_workers_cfg = max(0, int(_registry_float(cfg.get("max_workers"), 0.0)))
    if int(max_workers_cfg) > 0:
        jobs_eff = min(int(jobs_eff), int(max_workers_cfg))
    soft_caps_raw = cfg.get("soft_max_workers_by_total")
    if isinstance(soft_caps_raw, (tuple, list)):
        soft_caps: list[tuple[int, int]] = []
        for row in soft_caps_raw:
            if not (isinstance(row, (tuple, list)) and len(row) >= 2):
                continue
            try:
                max_total_i = int(row[0])
                cap_i = int(row[1])
            except (TypeError, ValueError):
                continue
            if int(max_total_i) <= 0 or int(cap_i) <= 0:
                continue
            soft_caps.append((int(max_total_i), int(cap_i)))
        if soft_caps and int(total_i) > 0:
            matching = [entry for entry in soft_caps if int(total_i) <= int(entry[0])]
            if matching:
                _, cap_soft = min(matching, key=lambda row: int(row[0]))
                jobs_eff = min(int(jobs_eff), max(1, int(cap_soft)))
    if int(total_i) > 0 and int(min_items) > 0:
        cap = int(math.ceil(float(total_i) / float(min_items)))
        jobs_eff = min(int(jobs_eff), max(1, int(cap)))
    _ = stage_label  # keep signature explicit; stage-specific tuning can be added without API changes.
    return max(1, int(jobs_eff))


def _axis_cost_hint(axis_name: str, key: str, default: float) -> float:
    axis_dims = _AXIS_DIMENSION_REGISTRY.get(str(axis_name), {})
    hints = axis_dims.get("cost_hints")
    if isinstance(hints, dict):
        return _registry_float(hints.get(str(key)), default)
    return float(default)


def _cost_model_weight(name: str, default: float) -> float:
    dims = _AXIS_DIMENSION_REGISTRY.get("cost_model", {})
    if isinstance(dims, dict):
        return _registry_float(dims.get(str(name)), default)
    return float(default)


def _combo_full_dim_size_from_registry(*, dims: dict[str, object], dim_key: str) -> int:
    key = str(dim_key).strip()
    if not key:
        return 0
    if key == "confirm":
        return int(len(tuple(dims.get("confirm_bars") or ())))
    if key == "short_mult":
        return int(len(tuple(dims.get("short_mults") or ())))
    variants_raw = dims.get(f"{key}_variants")
    if isinstance(variants_raw, (list, tuple)):
        return int(len(tuple(variants_raw)))
    return 0


def _cardinality(*sizes: object) -> int:
    total = 1
    for raw in sizes:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return 0
        if n <= 0:
            return 0
        total *= n
    return int(total)


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


def _filters_payload(filters: FiltersConfig | None) -> dict | None:
    return _codec_filters_payload(filters)


def _spot_strategy_payload(cfg: ConfigBundle, *, meta: ContractMeta) -> dict:
    return _codec_spot_strategy_payload(cfg, meta=meta)


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


def _risk_pack_riskoff(
    *,
    tr_med: float,
    lookback_days: int = 5,
    mode: str = "hygiene",
    long_factor: float | None = None,
    short_factor: float | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskoff_tr5_med_pct": float(tr_med),
        "riskoff_tr5_lookback_days": int(lookback_days),
        "riskoff_mode": str(mode),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if long_factor is not None:
        out["riskoff_long_risk_mult_factor"] = float(long_factor)
    if short_factor is not None:
        out["riskoff_short_risk_mult_factor"] = float(short_factor)
    return out


def _risk_pack_riskpanic(
    *,
    tr_med: float,
    neg_gap_ratio: float,
    lookback_days: int = 5,
    short_factor: float | None = 0.5,
    long_factor: float | None = None,
    mode: str | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskpanic_tr5_med_pct": float(tr_med),
        "riskpanic_neg_gap_ratio_min": float(neg_gap_ratio),
        "riskpanic_lookback_days": int(lookback_days),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if short_factor is not None:
        out["riskpanic_short_risk_mult_factor"] = float(short_factor)
    if long_factor is not None:
        out["riskpanic_long_risk_mult_factor"] = float(long_factor)
    if mode is not None:
        out["riskoff_mode"] = str(mode)
    return out


def _risk_pack_riskpop(
    *,
    tr_med: float,
    pos_gap_ratio: float,
    lookback_days: int = 5,
    long_factor: float | None = 1.2,
    short_factor: float | None = 0.5,
    mode: str | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskpop_tr5_med_pct": float(tr_med),
        "riskpop_pos_gap_ratio_min": float(pos_gap_ratio),
        "riskpop_lookback_days": int(lookback_days),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if long_factor is not None:
        out["riskpop_long_risk_mult_factor"] = float(long_factor)
    if short_factor is not None:
        out["riskpop_short_risk_mult_factor"] = float(short_factor)
    if mode is not None:
        out["riskoff_mode"] = str(mode)
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
            fill_mode = normalize_spot_fill_mode(strategy.get("spot_entry_fill_mode"), default="close")
            if fill_mode != SPOT_FILL_MODE_NEXT_TRADABLE_BAR:
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
        # Prefer strategy-embedded filters (newer milestones); fall back to group-level
        # filters for legacy payloads.
        filters = strategy.get("filters")
        if not isinstance(filters, dict):
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
    # Decode milestone payload through the shared codec so sweep baselines inherit
    # the same shape as runtime/backtest payloads (including dual-branch/rats filters).
    merged_filters: dict[str, object] | None = None
    if isinstance(filters, dict):
        merged_filters = dict(filters)
    nested_filters = strategy.get("filters") if isinstance(strategy, dict) else None
    if isinstance(nested_filters, dict):
        if merged_filters is None:
            merged_filters = {}
        merged_filters.update(nested_filters)

    parsed_filters: FiltersConfig | None = None
    if isinstance(merged_filters, dict):
        parsed_filters = _codec_filters_from_payload(merged_filters)
        if _filters_payload(parsed_filters) is None:
            parsed_filters = None

    parsed_strategy = _codec_strategy_from_payload(strategy, filters=parsed_filters)
    return replace(cfg, strategy=parsed_strategy)
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

@dataclass(frozen=True)
class AxisExecutionSpec:
    include_in_axis_all: bool
    help_text: str
    fast_path: str = "yes"
    sharding_class: str = "none"
    coverage_note: str = ""
    parallel_profile_axis_all: str = "single"
    dimensional_cost_priors: tuple[tuple[str, float], ...] = ()
    total_hint_mode: str = ""
    total_hint_static: int | None = None
    total_hint_dims: tuple[str, ...] = ()


_AXIS_SURFACE_SPEC_BY_NAME: dict[str, AxisSurfaceSpec] = {}
_AXIS_SURFACE_SPEC_DUPLICATES: list[str] = []
for _spec in _AXIS_SURFACE_SPECS:
    _key = str(_spec.name).strip()
    if not _key:
        raise ValueError("AxisSurfaceSpec name must be non-empty.")
    if _key in _AXIS_SURFACE_SPEC_BY_NAME:
        _AXIS_SURFACE_SPEC_DUPLICATES.append(_key)
        continue
    _AXIS_SURFACE_SPEC_BY_NAME[_key] = _spec
if _AXIS_SURFACE_SPEC_DUPLICATES:
    raise ValueError(f"Duplicate AxisSurfaceSpec keys: {sorted(set(_AXIS_SURFACE_SPEC_DUPLICATES))}")

_AXIS_SURFACE_SPEC_MISSING_KEYS = sorted(set(_AXIS_CHOICES) - set(_AXIS_SURFACE_SPEC_BY_NAME))
if _AXIS_SURFACE_SPEC_MISSING_KEYS:
    raise ValueError(f"Missing AxisSurfaceSpec keys: {_AXIS_SURFACE_SPEC_MISSING_KEYS}")


def _build_axis_surface_maps() -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, tuple[str, ...]]]:
    help_map: dict[str, str] = {}
    hint_static: dict[str, int] = {}
    hint_mode: dict[str, str] = {}
    hint_dims: dict[str, tuple[str, ...]] = {}
    for spec in _AXIS_SURFACE_SPECS:
        key = str(spec.name).strip()
        if not key:
            continue
        help_map[key] = str(spec.help_text or "See source.")
        if spec.total_hint_static is not None:
            try:
                hint_static[key] = int(spec.total_hint_static)
            except (TypeError, ValueError):
                pass
        mode = str(spec.total_hint_mode or "").strip()
        if mode:
            hint_mode[key] = str(mode)
        dims = tuple(str(dim).strip() for dim in tuple(spec.total_hint_dims or ()) if str(dim).strip())
        if dims:
            hint_dims[key] = tuple(dims)
    return help_map, hint_static, hint_mode, hint_dims


(
    _AXIS_HELP_TEXT_BY_NAME,
    _AXIS_TOTAL_HINT_STATIC_BY_NAME,
    _AXIS_TOTAL_HINT_MODE_BY_NAME,
    _AXIS_TOTAL_HINT_DIMS_BY_NAME,
) = _build_axis_surface_maps()


def _axis_dimensional_cost_priors(axis_name: str, *, source: str | None = None) -> tuple[tuple[str, float], ...]:
    source_key = str(source or axis_name).strip()
    if not source_key:
        return ()
    dims = _AXIS_DIMENSION_REGISTRY.get(source_key, {})
    if not isinstance(dims, dict):
        return ()
    priors_raw = dims.get("cost_hints")
    if not isinstance(priors_raw, dict):
        return ()
    priors: list[tuple[str, float]] = []
    for key, val in priors_raw.items():
        key_s = str(key).strip()
        if not key_s:
            continue
        try:
            priors.append((key_s, float(val)))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(priors, key=lambda row: row[0]))


def _build_axis_execution_specs() -> dict[str, AxisExecutionSpec]:
    out: dict[str, AxisExecutionSpec] = {}
    for axis_name in _AXIS_CHOICES:
        axis_key = str(axis_name)
        surface_spec = _AXIS_SURFACE_SPEC_BY_NAME.get(str(axis_key))
        in_all = bool(surface_spec.include_in_axis_all) if isinstance(surface_spec, AxisSurfaceSpec) else (str(axis_key) in _AXIS_INCLUDE_IN_AXIS_ALL_DEFAULTS)
        dim_source = (
            str(surface_spec.dimensional_cost_source).strip()
            if isinstance(surface_spec, AxisSurfaceSpec) and str(surface_spec.dimensional_cost_source).strip()
            else str(axis_name)
        )
        hint_dims = tuple(
            str(dim).strip()
            for dim in (
                tuple(surface_spec.total_hint_dims)
                if isinstance(surface_spec, AxisSurfaceSpec) and isinstance(surface_spec.total_hint_dims, tuple)
                else tuple(_AXIS_TOTAL_HINT_DIMS_BY_NAME.get(str(axis_name), ()))
            )
            if str(dim).strip()
        )
        hint_mode = (
            str(surface_spec.total_hint_mode or "").strip()
            if isinstance(surface_spec, AxisSurfaceSpec) and str(surface_spec.total_hint_mode or "").strip()
            else str(_AXIS_TOTAL_HINT_MODE_BY_NAME.get(str(axis_name), "")).strip()
        )
        hint_static = (
            int(surface_spec.total_hint_static)
            if isinstance(surface_spec, AxisSurfaceSpec) and isinstance(surface_spec.total_hint_static, int)
            else (
                int(_AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis_name)))
                if isinstance(_AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis_name)), int)
                else None
            )
        )
        out[str(axis_name)] = AxisExecutionSpec(
            include_in_axis_all=bool(in_all),
            help_text=(
                str(surface_spec.help_text or "See source.")
                if isinstance(surface_spec, AxisSurfaceSpec)
                else str(_AXIS_HELP_TEXT_BY_NAME.get(str(axis_name)) or "See source.")
            ),
            fast_path=(
                str(surface_spec.fast_path or "yes")
                if isinstance(surface_spec, AxisSurfaceSpec)
                else "yes"
            ),
            sharding_class=(
                str(surface_spec.sharding_class or "none")
                if isinstance(surface_spec, AxisSurfaceSpec)
                else "none"
            ),
            coverage_note=(
                str(surface_spec.coverage_note or "")
                if isinstance(surface_spec, AxisSurfaceSpec)
                else ""
            ),
            parallel_profile_axis_all=(
                str(surface_spec.parallel_profile_axis_all or "single")
                if isinstance(surface_spec, AxisSurfaceSpec)
                else "single"
            ),
            dimensional_cost_priors=_axis_dimensional_cost_priors(str(axis_name), source=dim_source),
            total_hint_mode=str(hint_mode),
            total_hint_static=(int(hint_static) if isinstance(hint_static, int) else None),
            total_hint_dims=tuple(hint_dims),
        )
    return out


_AXIS_EXECUTION_SPEC_BY_NAME: dict[str, AxisExecutionSpec] = _build_axis_execution_specs()


def _combo_full_preset_key(name: str) -> str:
    key = str(name or "").strip().lower()
    if key == "none":
        return ""
    return str(key)


def _combo_full_preset_axes(
    *,
    include_tiers: bool = True,
    include_aliases: bool = True,
) -> tuple[str, ...]:
    out: list[str] = []
    if bool(include_tiers):
        out.extend(str(name) for name in _COMBO_FULL_PRESET_TIER_NAMES)
    if bool(include_aliases):
        out.extend(str(name) for name in _COMBO_FULL_PRESET_ALIAS_NAMES)
    return tuple(out)


def _combo_full_preset_spec(name: str) -> dict[str, object]:
    key = _combo_full_preset_key(str(name))
    if not key:
        return {}
    if key in _COMBO_FULL_COVERAGE_TIER_REGISTRY:
        tier_key = str(key)
        tier_spec = _COMBO_FULL_COVERAGE_TIER_REGISTRY.get(tier_key) or {}
        raw_dims = tier_spec.get("freeze_dims")
        freeze_dims = tuple(str(dim).strip() for dim in tuple(raw_dims or ()) if str(dim).strip())
        customizer = str(tier_spec.get("customizer") or "").strip().lower()
        hint_axis = str(tier_spec.get("hint_axis") or "").strip().lower()
        return {
            "name": str(tier_key),
            "tier": str(tier_key),
            "freeze_dims": tuple(freeze_dims),
            "customizer": str(customizer),
            "hint_axis": str(hint_axis),
            "is_alias": False,
        }
    alias_spec = _COMBO_FULL_PRESET_ALIAS_REGISTRY.get(str(key))
    if not isinstance(alias_spec, dict):
        return {}
    tier_key = str(alias_spec.get("tier") or "").strip().lower()
    tier_spec = _COMBO_FULL_COVERAGE_TIER_REGISTRY.get(str(tier_key)) or {}
    raw_dims = alias_spec.get("freeze_dims")
    if not isinstance(raw_dims, (tuple, list)):
        raw_dims = tier_spec.get("freeze_dims")
    freeze_dims = tuple(str(dim).strip() for dim in tuple(raw_dims or ()) if str(dim).strip())
    customizer = str(alias_spec.get("customizer") or tier_spec.get("customizer") or "").strip().lower()
    hint_axis = str(alias_spec.get("hint_axis") or "").strip().lower()
    if not hint_axis:
        hint_axis = str(key) if str(key) in _AXIS_TOTAL_HINT_MODE_BY_NAME or str(key) in _AXIS_TOTAL_HINT_STATIC_BY_NAME else str(tier_spec.get("hint_axis") or "")
    return {
        "name": str(key),
        "tier": str(tier_key or "custom"),
        "freeze_dims": tuple(freeze_dims),
        "customizer": str(customizer),
        "hint_axis": str(hint_axis).strip().lower(),
        "is_alias": True,
    }


def _combo_full_preset_tier(name: str) -> str:
    spec = _combo_full_preset_spec(str(name))
    tier = str(spec.get("tier") or "").strip().lower()
    return tier or "custom"


def _combo_full_preset_hint_axis(name: str) -> str:
    spec = _combo_full_preset_spec(str(name))
    return str(spec.get("hint_axis") or "").strip().lower()


def _combo_full_preset_customizer(name: str) -> str:
    spec = _combo_full_preset_spec(str(name))
    return str(spec.get("customizer") or "").strip().lower()


def _combo_full_preset_hint_spec(name: str) -> dict[str, object]:
    spec = _combo_full_preset_spec(str(name))
    tier = str(spec.get("tier") or "").strip().lower()
    tier_base = _COMBO_FULL_COVERAGE_TIER_REGISTRY.get(str(tier), {})
    out: dict[str, object] = {}
    if isinstance(tier_base, dict):
        if tier_base.get("hint_mode") is not None:
            out["mode"] = tier_base.get("hint_mode")
        if tier_base.get("hint_dims") is not None:
            out["dims"] = tier_base.get("hint_dims")
        if tier_base.get("hint_static") is not None:
            out["static"] = tier_base.get("hint_static")
    preset_key = _combo_full_preset_key(str(name))
    alias_override = _COMBO_FULL_PRESET_ALIAS_REGISTRY.get(str(preset_key), {})
    if isinstance(alias_override, dict):
        if alias_override.get("hint_mode") is not None:
            out["mode"] = alias_override.get("hint_mode")
        if alias_override.get("hint_dims") is not None:
            out["dims"] = alias_override.get("hint_dims")
        if alias_override.get("hint_static") is not None:
            out["static"] = alias_override.get("hint_static")
    return out


def _combo_full_preset_freeze_dims(name: str) -> tuple[str, ...]:
    spec = _combo_full_preset_spec(str(name))
    raw = spec.get("freeze_dims")
    if not isinstance(raw, (tuple, list)):
        return ()
    return tuple(str(dim).strip() for dim in tuple(raw) if str(dim).strip())


def _combo_full_preset_choices(
    *,
    include_empty: bool = True,
    include_none: bool = True,
    include_aliases: bool = True,
) -> tuple[str, ...]:
    out: list[str] = []
    if bool(include_empty):
        out.append("")
    if bool(include_none):
        out.append("none")
    out.extend(str(axis_name) for axis_name in _combo_full_preset_axes(include_tiers=True, include_aliases=bool(include_aliases)))
    return tuple(out)


def _axis_catalog_lines() -> list[str]:
    lines: list[str] = [
        "Axis Catalog:",
        "  all                        Serial axis_all plan (or parallel when --jobs > 1 with --offline).",
    ]
    for axis_name in _AXIS_CHOICES:
        spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(str(axis_name))
        axis_desc = str(spec.help_text if isinstance(spec, AxisExecutionSpec) else "See source.")
        lines.append(f"  {axis_name:<26} {axis_desc}")
    return lines


def _combo_full_preset_catalog_lines() -> list[str]:
    lines: list[str] = ["combo_full Coverage Tiers:"]
    for tier in _combo_full_preset_axes(include_tiers=True, include_aliases=False):
        freeze_dims = _combo_full_preset_freeze_dims(str(tier))
        alias_count = sum(
            1
            for alias in _combo_full_preset_axes(include_tiers=False, include_aliases=True)
            if _combo_full_preset_tier(str(alias)) == str(tier)
        )
        lines.append(f"  {tier:<10} frozen_dims={len(freeze_dims):<3} aliases={int(alias_count)}")
    aliases = _combo_full_preset_axes(include_tiers=False, include_aliases=True)
    if aliases:
        lines.append(f"  preset_aliases(hidden): {', '.join(aliases)}")
    return lines


def _axis_coverage_row(axis_name: str) -> tuple[str, str, str, str]:
    axis_key = str(axis_name).strip().lower()
    spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(axis_key)
    sharded = "yes" if isinstance(spec, AxisExecutionSpec) and str(spec.sharding_class) != "none" else "no"
    cached = "yes"
    fast_path = str(spec.fast_path if isinstance(spec, AxisExecutionSpec) else "yes")
    notes = str(spec.coverage_note if isinstance(spec, AxisExecutionSpec) else "")
    return sharded, cached, fast_path, notes


def _spot_sweep_coverage_map_markdown(*, generated_on: date | None = None) -> str:
    generated = generated_on if generated_on is not None else _now_et().date()
    lines: list[str] = [
        "# Spot Sweep Coverage Map",
        "",
        f"Generated: {generated.isoformat()}",
        "",
        "Legend: `sharded` = stage-level worker sharding inside the axis; `cached` = run_cfg+context cache applies; `fast_path` = expected fast-summary eligibility (`yes`/`partial`/`no`).",
        "",
        "| axis | sharded | cached | fast_path | notes |",
        "|---|---|---|---|---|",
    ]
    for axis_name in _AXIS_CHOICES:
        sharded, cached, fast_path, notes = _axis_coverage_row(axis_name)
        lines.append(f"| `{axis_name}` | {sharded} | {cached} | {fast_path} | {notes} |")
    lines.extend(
        (
            "",
            "## Notes",
            "- `combo_full` and `--axis all` additionally support axis-level subprocess orchestration.",
            "- Persistent cross-process run_cfg cache is enabled via sqlite (`spot_sweeps_run_cfg_cache.sqlite3`).",
            "- Fast-path labels are conservative and reflect the current gate in `_can_use_fast_summary_path`.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_spot_sweep_coverage_map(path: Path, *, generated_on: date | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_spot_sweep_coverage_map_markdown(generated_on=generated_on))


_INTERNAL_FLAG_HELP_LINES: tuple[str, ...] = (
    "--combo-full-cartesian-stage VALUE: combo_full Cartesian worker stage marker.",
    "--combo-full-cartesian-worker INT / --combo-full-cartesian-workers INT / --combo-full-cartesian-out PATH: combo_full Cartesian shard controls.",
    "--combo-full-cartesian-run-min-trades INT: stage-specific min trades override for combo_full Cartesian.",
    f"--combo-full-preset {{{','.join(_combo_full_preset_choices(include_empty=False, include_none=True, include_aliases=False))}}}: "
    "optional combo_full range preset selector.",
)


def _spot_sweeps_help_epilog() -> str:
    axis_lines = _axis_catalog_lines()

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
            "  python -m tradebot.backtest spot --axis combo_full --combo-full-preset profile --offline",
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
            *_combo_full_preset_catalog_lines(),
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
            "Parallelism for --axis all/combo_full (spawns per-axis worker processes). "
            "0/omitted = auto (CPU count). Use --offline."
        ),
    )
    parser.add_argument(
        "--base",
        default="champion",
        choices=("default", "champion", "champion_pnl", "dual_regime"),
        help="Base profile to mutate before running the selected axis (see Base Profiles below).",
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
            "Optional milestones JSON used as champion source override for --base champion/champion_pnl."
        ),
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=("all", *_AXIS_CHOICES),
        metavar="AXIS",
        help="Axis to execute. Use 'all' for the axis_all plan. See Axis Catalog below.",
    )
    parser.add_argument("--sync-axis-docs", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--axis-docs-out",
        default="tradebot/backtest/spot_sweep_coverage_map.md",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--risk-overlays-skip-pop",
        action="store_true",
        default=False,
        help="risk_overlays: skip riskpop stage (riskoff+riskpanic only).",
    )
    # Internal flags (used by combo_full parallel sharding).
    parser.add_argument("--combo-full-cartesian-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-run-min-trades", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--combo-full-include-tick",
        action="store_true",
        default=False,
        help=(
            "combo_full: include $TICK gate variants. "
            "Default keeps tick fixed off for faster/offline-friendly runs."
        ),
    )
    parser.add_argument(
        "--combo-full-preset",
        default="",
        choices=_combo_full_preset_choices(include_empty=True, include_none=True),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cfg-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.sync_axis_docs):
        out_path = Path(str(args.axis_docs_out or "tradebot/backtest/spot_sweep_coverage_map.md"))
        _write_spot_sweep_coverage_map(out_path, generated_on=_now_et().date())
        print(f"Wrote {out_path}")
        return

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

    sizing_mode_arg_explicit = args.spot_sizing_mode is not None
    sizing_mode = (
        str(args.spot_sizing_mode).strip().lower()
        if sizing_mode_arg_explicit
        else ("risk_pct" if realism2 else "fixed")
    )
    if sizing_mode not in ("fixed", "notional_pct", "risk_pct"):
        sizing_mode = "fixed"
    spot_risk_pct_arg_explicit = args.spot_risk_pct is not None
    spot_risk_pct = float(args.spot_risk_pct) if spot_risk_pct_arg_explicit else (0.01 if realism2 else 0.0)
    spot_notional_pct_arg_explicit = args.spot_notional_pct is not None
    spot_notional_pct = (
        float(args.spot_notional_pct) if spot_notional_pct_arg_explicit else 0.0
    )
    spot_max_notional_pct_arg_explicit = args.spot_max_notional_pct is not None
    spot_max_notional_pct = (
        float(args.spot_max_notional_pct) if spot_max_notional_pct_arg_explicit else (0.50 if realism2 else 1.0)
    )
    spot_min_qty_arg_explicit = args.spot_min_qty is not None
    spot_max_qty_arg_explicit = args.spot_max_qty is not None
    spot_min_qty = int(args.spot_min_qty) if spot_min_qty_arg_explicit else 1
    spot_max_qty = int(args.spot_max_qty) if spot_max_qty_arg_explicit else 0

    def _resolve_run_min_trades() -> int:
        base = int(args.min_trades)
        out = int(base)
        for attr in (
            "combo_full_cartesian_run_min_trades",
        ):
            raw = getattr(args, attr, None)
            if raw is None:
                continue
            try:
                out = int(raw)
            except (TypeError, ValueError):
                out = int(base)
        return int(out)

    run_min_trades = _resolve_run_min_trades()
    if bool(args.write_milestones):
        run_min_trades = min(run_min_trades, int(args.milestone_min_trades))
    axis = str(args.axis).strip().lower()

    def _stage_cache_scope(stage_label: str) -> str:
        stage_key = str(stage_label).strip().lower()
        if not stage_key:
            return ""
        return f"{stage_key}|m{int(run_min_trades)}"

    def _default_seed_milestones_path() -> str | None:
        raw = str(getattr(args, "milestones_path", "") or "").strip()
        if not raw:
            return None
        p = Path(raw)
        if not p.exists():
            return None
        return str(p)

    data = IBKRHistoricalData()
    if offline:
        _require_offline_cache_or_die(
            data=data,
            cache_dir=cache_dir,
            symbol=symbol,
            exchange=None,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=signal_bar_size,
            use_rth=use_rth,
        )
        if spot_exec_bar_size and str(spot_exec_bar_size) != str(signal_bar_size):
            _require_offline_cache_or_die(
                data=data,
                cache_dir=cache_dir,
                symbol=symbol,
                exchange=None,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=spot_exec_bar_size,
                use_rth=use_rth,
            )

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
    run_cfg_dim_index_writes = 0
    worker_plan_cache_hits = 0
    worker_plan_cache_writes = 0
    stage_cell_status_reads = 0
    stage_cell_status_writes = 0
    cartesian_manifest_reads = 0
    cartesian_manifest_writes = 0
    cartesian_manifest_hits = 0
    cartesian_rank_manifest_reads = 0
    cartesian_rank_manifest_writes = 0
    cartesian_rank_manifest_hits = 0
    cartesian_rank_manifest_compactions = 0
    cartesian_rank_manifest_pending_ttl_prunes = 0
    stage_rank_manifest_reads = 0
    stage_rank_manifest_writes = 0
    stage_rank_manifest_hits = 0
    stage_rank_manifest_compactions = 0
    stage_rank_manifest_pending_ttl_prunes = 0
    stage_unresolved_summary_reads = 0
    stage_unresolved_summary_writes = 0
    stage_unresolved_summary_hits = 0
    rank_dominance_stamp_reads = 0
    rank_dominance_stamp_writes = 0
    rank_dominance_stamp_hits = 0
    rank_dominance_manifest_applies = 0
    rank_dominance_stamp_compactions = 0
    rank_dominance_stamp_ttl_prunes = 0
    rank_bin_runtime_reads = 0
    rank_bin_runtime_writes = 0
    stage_frontier_reads = 0
    stage_frontier_writes = 0
    stage_frontier_hits = 0
    winner_projection_reads = 0
    winner_projection_writes = 0
    winner_projection_hits = 0
    dimension_utility_reads = 0
    dimension_utility_writes = 0
    dimension_utility_hint_hits = 0
    dimension_upper_bound_reads = 0
    dimension_upper_bound_writes = 0
    dimension_upper_bound_deferred = 0
    planner_heartbeat_reads = 0
    planner_heartbeat_writes = 0
    planner_heartbeat_stale_candidates = 0
    series_pack_mmap_hint_hits = 0
    series_pack_pickle_hint_hits = 0
    series_pack_state_manifest_reads = 0
    series_pack_state_manifest_writes = 0
    series_pack_state_manifest_hits = 0
    run_cfg_fingerprint_cache: dict[str, tuple[tuple[tuple[int, object | None, object | None], tuple[int, object | None, object | None], tuple[int, object | None, object | None]], dict | None]] = {}
    run_cfg_axis_fp_cache: dict[str, str] = {}
    run_cfg_dim_index_seen: set[str] = set()
    run_cfg_dim_index_loaded: dict[str, float] = {}
    run_cfg_dim_index_loaded_once = False
    run_cfg_window_sig_cache: dict[
        tuple[
            tuple[int, object | None, object | None],
            tuple[int, object | None, object | None],
            tuple[int, object | None, object | None],
        ],
        str,
    ] = {}
    run_cfg_series_pack_state_cache: dict[tuple[str, str], str] = {}
    status_span_manifest_compact_seen: dict[tuple[str, str, str, str], float] = {}
    rank_dominance_stamp_compact_seen: dict[tuple[str, str], float] = {}
    rank_dominance_manifest_applied_seen: set[tuple[str, str]] = set()
    _RUN_CFG_CACHE_MISS = object()
    _RUN_CFG_CACHE_UNSET = object()
    run_cfg_persistent_payload_cache: dict[str, dict | None | object] = {}
    run_cfg_persistent_path = cache_dir / "spot_sweeps_run_cfg_cache.sqlite3"
    run_cfg_persistent_conn: sqlite3.Connection | None = None
    run_cfg_persistent_enabled = True
    run_cfg_persistent_lock = threading.Lock()
    run_cfg_persistent_pending: dict[str, tuple[str, dict | None]] = {}
    run_cfg_persistent_last_flush_ts = float(pytime.perf_counter())
    run_cfg_persistent_cfg = _cache_config("run_cfg_persistent")
    run_cfg_persistent_batch_write_size = max(
        1, int(_registry_float(run_cfg_persistent_cfg.get("batch_write_size"), 256.0))
    )
    run_cfg_persistent_batch_write_interval_sec = max(
        0.0, float(_registry_float(run_cfg_persistent_cfg.get("batch_write_interval_sec"), 2.0))
    )
    run_cfg_stage_worker_mode = bool(getattr(args, "combo_full_cartesian_stage", None) or getattr(args, "cfg_stage", None))
    run_cfg_persistent_ram_first_enabled = bool(
        _registry_float(run_cfg_persistent_cfg.get("ram_first_worker"), 1.0) > 0.0
        and bool(offline)
        and bool(run_cfg_stage_worker_mode)
    )
    _STAGE_CELL_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated"))
    _CARTESIAN_CELL_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated", "dominated"))
    _CARTESIAN_RANK_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated", "dominated"))
    _STAGE_RANK_STATUS_VALUES = frozenset(("pending", "cached_hit", "evaluated", "dominated"))

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
            conn.execute(
                "CREATE TABLE IF NOT EXISTS axis_dimension_fingerprint_index ("
                "fingerprint TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "est_cost REAL NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS worker_plan_cache ("
                "cache_key TEXT PRIMARY KEY, "
                "payload_json TEXT NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_cell_status ("
                "stage_label TEXT NOT NULL, "
                "strategy_fingerprint TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, strategy_fingerprint, axis_dimension_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_cell_manifest ("
                "stage_label TEXT NOT NULL, "
                "dimension_vector_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "strategy_fingerprint TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, dimension_vector_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_rank_manifest ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "rank_lo INTEGER NOT NULL, "
                "rank_hi INTEGER NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, rank_lo, rank_hi))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cartesian_rank_cursor ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "next_rank INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_rank_manifest ("
                "stage_label TEXT NOT NULL, "
                "plan_signature TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "rank_lo INTEGER NOT NULL, "
                "rank_hi INTEGER NOT NULL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, plan_signature, window_signature, rank_lo, rank_hi))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_unresolved_summary ("
                "manifest_name TEXT NOT NULL, "
                "stage_label TEXT NOT NULL, "
                "plan_signature TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "total INTEGER NOT NULL, "
                "unresolved_count INTEGER NOT NULL, "
                "resolved_count INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(manifest_name, stage_label, plan_signature, window_signature, total))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rank_dominance_stamp ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "dominance_signature TEXT NOT NULL, "
                "rank_lo INTEGER NOT NULL, "
                "rank_hi INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, dominance_signature, rank_lo, rank_hi))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_frontier ("
                "stage_label TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "run_min_trades INTEGER NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "best_pnl_over_dd REAL, "
                "best_pnl REAL, "
                "best_win_rate REAL, "
                "best_trades INTEGER, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, axis_dimension_fingerprint, window_signature, run_min_trades))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_winner_projection ("
                "stage_label TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "best_pnl_over_dd REAL, "
                "best_pnl REAL, "
                "best_win_rate REAL, "
                "best_trades INTEGER, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, axis_dimension_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rank_bin_runtime ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "rank_bin INTEGER NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "total_eval_sec REAL NOT NULL, "
                "cache_hits INTEGER NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, rank_bin))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dimension_value_utility ("
                "stage_label TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "dimension_key TEXT NOT NULL, "
                "dimension_value TEXT NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "cache_hits INTEGER NOT NULL, "
                "total_eval_sec REAL NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, window_signature, dimension_key, dimension_value))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dimension_upper_bound ("
                "stage_label TEXT NOT NULL, "
                "axis_dimension_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "eval_count INTEGER NOT NULL, "
                "keep_count INTEGER NOT NULL, "
                "best_pnl_over_dd REAL, "
                "best_pnl REAL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, axis_dimension_fingerprint, window_signature))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS planner_heartbeat ("
                "stage_label TEXT NOT NULL, "
                "worker_id INTEGER NOT NULL, "
                "last_seen REAL NOT NULL, "
                "tested INTEGER NOT NULL, "
                "cached_hits INTEGER NOT NULL, "
                "total INTEGER NOT NULL, "
                "eta_sec REAL, "
                "status TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(stage_label, worker_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS series_pack_state_manifest ("
                "strategy_fingerprint TEXT NOT NULL, "
                "window_signature TEXT NOT NULL, "
                "state TEXT NOT NULL, "
                "updated_at REAL NOT NULL, "
                "PRIMARY KEY(strategy_fingerprint, window_signature))"
            )
            run_cfg_persistent_conn = conn
            return conn
        except Exception:
            run_cfg_persistent_enabled = False
            run_cfg_persistent_conn = None
            return None

    def _run_cfg_persistent_key(
        *,
        strategy_fingerprint: str,
        axis_dimension_fingerprint: str,
        window_signature: str,
    ) -> str:
        raw = {
            "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
            "strategy_fingerprint": str(strategy_fingerprint),
            "axis_dimension_fingerprint": str(axis_dimension_fingerprint),
            "window_signature": str(window_signature),
            "run_min_trades": int(run_min_trades),
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _run_cfg_persistent_cached_lookup(*, cache_key: str) -> dict | None | object:
        key = str(cache_key)
        if key not in run_cfg_persistent_payload_cache:
            return _RUN_CFG_CACHE_UNSET
        cached_payload = run_cfg_persistent_payload_cache.get(key)
        if isinstance(cached_payload, dict):
            return dict(cached_payload)
        if cached_payload is None:
            return None
        return _RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_cached_store(*, cache_key: str, payload: dict | None | object) -> dict | None | object:
        key = str(cache_key)
        if isinstance(payload, dict):
            payload_out = dict(payload)
            run_cfg_persistent_payload_cache[key] = payload_out
            return dict(payload_out)
        if payload is None:
            run_cfg_persistent_payload_cache[key] = None
            return None
        run_cfg_persistent_payload_cache[key] = _RUN_CFG_CACHE_MISS
        return _RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_decode_payload(*, payload_json_raw: object) -> dict | None | object:
        try:
            payload = json.loads(str(payload_json_raw))
        except Exception:
            return _RUN_CFG_CACHE_MISS
        if payload is None:
            return None
        if isinstance(payload, dict):
            return dict(payload)
        return _RUN_CFG_CACHE_MISS

    def _run_cfg_persistent_get(*, cache_key: str) -> dict | None | object:
        key = str(cache_key)
        cached = _run_cfg_persistent_cached_lookup(cache_key=str(key))
        if cached is not _RUN_CFG_CACHE_UNSET:
            return cached
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
        try:
            with run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT payload_json FROM run_cfg_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
        except Exception:
            return _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
        if row is None:
            return _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
        decoded = _run_cfg_persistent_decode_payload(payload_json_raw=row[0])
        return _run_cfg_persistent_cached_store(cache_key=str(key), payload=decoded)

    def _run_cfg_persistent_get_many(*, cache_keys: list[str]) -> dict[str, dict | None]:
        out: dict[str, dict | None] = {}
        missing_db_keys: list[str] = []
        seen: set[str] = set()
        for raw_key in cache_keys:
            key = str(raw_key or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            cached = _run_cfg_persistent_cached_lookup(cache_key=str(key))
            if cached is not _RUN_CFG_CACHE_UNSET:
                if cached is not _RUN_CFG_CACHE_MISS:
                    out[key] = cached if isinstance(cached, dict) else None
                continue
            missing_db_keys.append(key)

        if not missing_db_keys:
            return out

        conn = _run_cfg_persistent_conn()
        if conn is None:
            for key in missing_db_keys:
                _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
            return out

        found: set[str] = set()
        chunk_size = 400
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(missing_db_keys), chunk_size):
                    chunk = missing_db_keys[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT cache_key, payload_json FROM run_cfg_cache WHERE cache_key IN ({placeholders})",
                        tuple(chunk),
                    ).fetchall()
                    for cache_key_raw, payload_json_raw in rows:
                        key = str(cache_key_raw or "").strip()
                        if not key:
                            continue
                        found.add(key)
                        decoded = _run_cfg_persistent_decode_payload(payload_json_raw=payload_json_raw)
                        stored = _run_cfg_persistent_cached_store(cache_key=str(key), payload=decoded)
                        if stored is _RUN_CFG_CACHE_MISS:
                            continue
                        out[key] = stored if isinstance(stored, dict) else None
        except Exception:
            for key in missing_db_keys:
                if key not in run_cfg_persistent_payload_cache:
                    _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
            return out

        for key in missing_db_keys:
            if key not in found:
                _run_cfg_persistent_cached_store(cache_key=str(key), payload=_RUN_CFG_CACHE_MISS)
        return out

    def _run_cfg_persistent_flush_pending(*, force: bool = False) -> None:
        nonlocal run_cfg_persistent_last_flush_ts
        if not run_cfg_persistent_pending:
            return
        conn = _run_cfg_persistent_conn()
        if conn is None:
            run_cfg_persistent_pending.clear()
            run_cfg_persistent_last_flush_ts = float(pytime.perf_counter())
            return
        now = float(pytime.perf_counter())
        if not bool(force):
            if len(run_cfg_persistent_pending) < int(run_cfg_persistent_batch_write_size):
                if (now - float(run_cfg_persistent_last_flush_ts)) < float(run_cfg_persistent_batch_write_interval_sec):
                    return
        payload = list(run_cfg_persistent_pending.items())
        if not payload:
            run_cfg_persistent_last_flush_ts = float(now)
            return
        try:
            wall_ts = float(pytime.time())
            rows = [
                (str(cache_key), str(payload_json), float(wall_ts))
                for cache_key, (payload_json, _payload_obj) in payload
            ]
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO run_cfg_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    rows,
                )
            for cache_key, (_payload_json, payload_obj) in payload:
                _run_cfg_persistent_cached_store(
                    cache_key=str(cache_key),
                    payload=(dict(payload_obj) if isinstance(payload_obj, dict) else None),
                )
            run_cfg_persistent_pending.clear()
            run_cfg_persistent_last_flush_ts = float(now)
        except Exception:
            return

    def _run_cfg_persistent_set(*, cache_key: str, payload: dict | None) -> None:
        key = str(cache_key)
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            payload_obj = dict(payload) if isinstance(payload, dict) else None
            payload_json = json.dumps(payload_obj, sort_keys=True, default=str)
            if bool(run_cfg_persistent_ram_first_enabled):
                run_cfg_persistent_pending[str(key)] = (str(payload_json), payload_obj if isinstance(payload_obj, dict) else None)
                _run_cfg_persistent_cached_store(
                    cache_key=str(key),
                    payload=(dict(payload_obj) if isinstance(payload_obj, dict) else None),
                )
                _run_cfg_persistent_flush_pending(force=False)
                return
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO run_cfg_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (key, payload_json, float(pytime.time())),
                )
            _run_cfg_persistent_cached_store(cache_key=str(key), payload=(dict(payload_obj) if isinstance(payload_obj, dict) else None))
        except Exception:
            return

    def _stage_cell_status_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], str]:
        nonlocal stage_cell_status_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for strategy_fp, axis_dim_fp, window_sig in cells:
            cell_key = (str(strategy_fp), str(axis_dim_fp), str(window_sig))
            if cell_key in seen:
                continue
            seen.add(cell_key)
            deduped.append(cell_key)
        if not deduped:
            return {}
        out: dict[tuple[str, str, str], str] = {}
        chunk_size = 120
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join(
                        "(strategy_fingerprint=? AND axis_dimension_fingerprint=? AND window_signature=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key)]
                    for strategy_fp, axis_dim_fp, window_sig in chunk:
                        params.extend((str(strategy_fp), str(axis_dim_fp), str(window_sig)))
                    rows = conn.execute(
                        "SELECT strategy_fingerprint, axis_dimension_fingerprint, window_signature, status "
                        "FROM stage_cell_status WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows:
                        cell = (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""))
                        status = str(row[3] or "").strip().lower()
                        if not cell[0] or not cell[1] or not cell[2] or status not in _STAGE_CELL_STATUS_VALUES:
                            continue
                        out[cell] = status
            stage_cell_status_reads += len(out)
        except Exception:
            return {}
        return out

    def _stage_cell_status_set_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        nonlocal stage_cell_status_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        status_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2}
        merged: dict[tuple[str, str, str], str] = {}
        for strategy_fp, axis_dim_fp, window_sig, status_raw in rows:
            strategy_key = str(strategy_fp).strip()
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            status = str(status_raw or "").strip().lower()
            if not strategy_key or not axis_key or not window_key:
                continue
            if status not in _STAGE_CELL_STATUS_VALUES:
                continue
            cell_key = (strategy_key, axis_key, window_key)
            prev = merged.get(cell_key)
            if prev is None or status_priority.get(status, -1) >= status_priority.get(prev, -1):
                merged[cell_key] = status
        if not merged:
            return
        payload = [
            (
                str(stage_key),
                str(strategy_fp),
                str(axis_dim_fp),
                str(window_sig),
                str(status),
                float(pytime.time()),
            )
            for (strategy_fp, axis_dim_fp, window_sig), status in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO stage_cell_status("
                    "stage_label, strategy_fingerprint, axis_dimension_fingerprint, window_signature, status, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
                conn.execute(
                    "DELETE FROM stage_unresolved_summary WHERE manifest_name='stage_cell' AND stage_label=?",
                    (str(stage_key),),
                )
            stage_cell_status_writes += len(payload)
        except Exception:
            return

    def _cartesian_cell_manifest_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[str, str]]:
        nonlocal cartesian_manifest_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for dim_vec_fp, window_sig in cells:
            key = (str(dim_vec_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], tuple[str, str]] = {}
        chunk_size = 160
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join(
                        "(dimension_vector_fingerprint=? AND window_signature=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key)]
                    for dim_vec_fp, window_sig in chunk:
                        params.extend((str(dim_vec_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT dimension_vector_fingerprint, window_signature, strategy_fingerprint, status "
                        "FROM cartesian_cell_manifest WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        dim_vec_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        strategy_fp = str(row[2] or "").strip()
                        status = str(row[3] or "").strip().lower()
                        if not dim_vec_fp or not window_sig or status not in _CARTESIAN_CELL_STATUS_VALUES:
                            continue
                        out[(dim_vec_fp, window_sig)] = (status, strategy_fp)
            cartesian_manifest_reads += len(out)
        except Exception:
            return {}
        return out

    def _cartesian_cell_manifest_set_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, str]],
    ) -> None:
        nonlocal cartesian_manifest_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        status_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}
        merged: dict[tuple[str, str], tuple[str, str]] = {}
        for dim_vec_fp, window_sig, strategy_fp, status_raw in rows:
            dim_key = str(dim_vec_fp).strip()
            window_key = str(window_sig).strip()
            strategy_key = str(strategy_fp).strip()
            status = str(status_raw or "").strip().lower()
            if not dim_key or not window_key:
                continue
            if status not in _CARTESIAN_CELL_STATUS_VALUES:
                continue
            key = (dim_key, window_key)
            prev = merged.get(key)
            prev_status = prev[1] if isinstance(prev, tuple) and len(prev) == 2 else ""
            if prev is None or status_priority.get(status, -1) >= status_priority.get(str(prev_status), -1):
                merged[key] = (strategy_key, status)
        if not merged:
            return
        payload = [
            (
                str(stage_key),
                str(dim_vec_fp),
                str(window_sig),
                str(strategy_fp),
                str(status),
                float(pytime.time()),
            )
            for (dim_vec_fp, window_sig), (strategy_fp, status) in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO cartesian_cell_manifest("
                    "stage_label, dimension_vector_fingerprint, window_signature, strategy_fingerprint, status, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
            cartesian_manifest_writes += len(payload)
        except Exception:
            return

    def _status_span_manifest_spec(manifest_name: str) -> dict[str, object]:
        key = str(manifest_name or "").strip().lower()
        if key == "cartesian":
            return {
                "key": "cartesian",
                "table": "cartesian_rank_manifest",
                "cfg_key": "cartesian_rank_manifest",
                "status_values": _CARTESIAN_RANK_STATUS_VALUES,
                "has_plan_signature": False,
                "compact_min_rows": 1024.0,
                "compact_min_interval_sec": 120.0,
                "pending_ttl_sec": 86400.0,
            }
        if key == "stage":
            return {
                "key": "stage",
                "table": "stage_rank_manifest",
                "cfg_key": "stage_rank_manifest",
                "status_values": _STAGE_RANK_STATUS_VALUES,
                "has_plan_signature": True,
                "compact_min_rows": 512.0,
                "compact_min_interval_sec": 60.0,
                "pending_ttl_sec": 21600.0,
            }
        return {}

    def _status_span_manifest_counter_add(*, manifest_name: str, field: str, value: int) -> None:
        nonlocal cartesian_rank_manifest_reads, cartesian_rank_manifest_writes, cartesian_rank_manifest_compactions
        nonlocal cartesian_rank_manifest_pending_ttl_prunes, stage_rank_manifest_reads, stage_rank_manifest_writes
        nonlocal stage_rank_manifest_compactions, stage_rank_manifest_pending_ttl_prunes
        delta = int(max(0, int(value)))
        if delta <= 0:
            return
        key = str(manifest_name).strip().lower()
        field_key = str(field).strip().lower()
        if key == "cartesian":
            if field_key == "reads":
                cartesian_rank_manifest_reads += delta
            elif field_key == "writes":
                cartesian_rank_manifest_writes += delta
            elif field_key == "compactions":
                cartesian_rank_manifest_compactions += delta
            elif field_key == "pending_ttl_prunes":
                cartesian_rank_manifest_pending_ttl_prunes += delta
        elif key == "stage":
            if field_key == "reads":
                stage_rank_manifest_reads += delta
            elif field_key == "writes":
                stage_rank_manifest_writes += delta
            elif field_key == "compactions":
                stage_rank_manifest_compactions += delta
            elif field_key == "pending_ttl_prunes":
                stage_rank_manifest_pending_ttl_prunes += delta

    def _status_span_manifest_counter_add_hits(*, manifest_name: str, covered: int) -> None:
        nonlocal cartesian_rank_manifest_hits, stage_rank_manifest_hits
        delta = int(max(0, int(covered)))
        if delta <= 0:
            return
        key = str(manifest_name).strip().lower()
        if key == "cartesian":
            cartesian_rank_manifest_hits += delta
        elif key == "stage":
            stage_rank_manifest_hits += delta

    def _stage_unresolved_summary_get(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        plan_signature: str = "",
    ) -> tuple[int, int] | None:
        nonlocal stage_unresolved_summary_reads, stage_unresolved_summary_hits
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return None
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return None
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        try:
            total_i = int(total)
        except (TypeError, ValueError):
            return None
        if total_i <= 0 or not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return None
        cfg = _cache_config("stage_unresolved_summary")
        ttl_sec = max(0.0, float(_registry_float(cfg.get("ttl_sec"), 21600.0)))
        now_ts = float(pytime.time())
        row = None
        try:
            with run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT unresolved_count, resolved_count, updated_at "
                    "FROM stage_unresolved_summary "
                    "WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=? AND total=?",
                    (str(manifest_key), str(stage_key), str(plan_key), str(window_key), int(total_i)),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        stage_unresolved_summary_reads += 1
        try:
            unresolved_i = int(row[0] or 0)
            resolved_i = int(row[1] or 0)
            updated_at = float(row[2] or 0.0)
        except Exception:
            return None
        unresolved_i = int(max(0, min(int(total_i), int(unresolved_i))))
        resolved_i = int(max(0, min(int(total_i), int(resolved_i))))
        if float(ttl_sec) > 0.0 and float(updated_at) > 0.0 and (float(now_ts) - float(updated_at)) > float(ttl_sec):
            try:
                with run_cfg_persistent_lock:
                    conn.execute(
                        "DELETE FROM stage_unresolved_summary "
                        "WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=? AND total=?",
                        (str(manifest_key), str(stage_key), str(plan_key), str(window_key), int(total_i)),
                    )
            except Exception:
                pass
            return None
        stage_unresolved_summary_hits += 1
        return int(unresolved_i), int(resolved_i)

    def _stage_unresolved_summary_invalidate(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        plan_signature: str = "",
    ) -> None:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        if not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return
        try:
            with run_cfg_persistent_lock:
                conn.execute(
                    "DELETE FROM stage_unresolved_summary "
                    "WHERE manifest_name=? AND stage_label=? AND plan_signature=? AND window_signature=?",
                    (str(manifest_key), str(stage_key), str(plan_key), str(window_key)),
                )
        except Exception:
            return

    def _stage_unresolved_summary_set(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        unresolved_count: int,
        resolved_count: int,
        plan_signature: str = "",
    ) -> None:
        nonlocal stage_unresolved_summary_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        manifest_key = str(manifest_name or "").strip().lower()
        if manifest_key not in ("cartesian", "stage", "stage_cell"):
            return
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        plan_key = str(plan_signature).strip() if manifest_key in ("stage", "stage_cell") else ""
        if not stage_key or not window_key or (manifest_key in ("stage", "stage_cell") and not plan_key):
            return
        try:
            total_i = int(max(0, int(total)))
            unresolved_i = int(max(0, int(unresolved_count)))
            resolved_i = int(max(0, int(resolved_count)))
        except (TypeError, ValueError):
            return
        if total_i <= 0:
            return
        unresolved_i = int(max(0, min(int(total_i), int(unresolved_i))))
        resolved_i = int(max(0, min(int(total_i), int(resolved_i))))
        now_ts = float(pytime.time())
        try:
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT INTO stage_unresolved_summary("
                    "manifest_name, stage_label, plan_signature, window_signature, total, unresolved_count, resolved_count, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(manifest_name, stage_label, plan_signature, window_signature, total) DO UPDATE SET "
                    "unresolved_count=excluded.unresolved_count, "
                    "resolved_count=excluded.resolved_count, "
                    "updated_at=excluded.updated_at",
                    (
                        str(manifest_key),
                        str(stage_key),
                        str(plan_key),
                        str(window_key),
                        int(total_i),
                        int(unresolved_i),
                        int(resolved_i),
                        float(now_ts),
                    ),
                )
            stage_unresolved_summary_writes += 1
        except Exception:
            return

    def _status_span_rows_compact(
        *,
        rows: list[tuple[int, int, str]],
        status_values: frozenset[str],
    ) -> list[tuple[int, int, str]]:
        status_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}
        merged: dict[tuple[int, int], str] = {}
        for rank_lo_raw, rank_hi_raw, status_raw in rows:
            try:
                rank_lo = int(rank_lo_raw)
                rank_hi = int(rank_hi_raw)
            except (TypeError, ValueError):
                continue
            status = str(status_raw or "").strip().lower()
            if rank_lo < 0 or rank_hi < rank_lo or status not in status_values:
                continue
            key = (int(rank_lo), int(rank_hi))
            prev = merged.get(key)
            if prev is None or status_priority.get(status, -1) >= status_priority.get(str(prev), -1):
                merged[key] = str(status)
        if not merged:
            return []
        ordered = sorted(
            ((int(lo), int(hi), str(status)) for (lo, hi), status in merged.items()),
            key=lambda row: (int(row[0]), int(row[1])),
        )
        out: list[tuple[int, int, str]] = []
        for rank_lo, rank_hi, status in ordered:
            if out:
                prev_lo, prev_hi, prev_status = out[-1]
                if str(prev_status) == str(status) and int(rank_lo) <= int(prev_hi) + 1:
                    out[-1] = (int(prev_lo), max(int(prev_hi), int(rank_hi)), str(status))
                    continue
            out.append((int(rank_lo), int(rank_hi), str(status)))
        return out

    def _status_span_manifest_set_many(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
        plan_signature: str = "",
        replace_scope: bool = False,
    ) -> None:
        spec = _status_span_manifest_spec(str(manifest_name))
        if not isinstance(spec, dict) or not spec:
            return
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip()
        if not stage_key or not window_key:
            return
        if has_plan_signature and not plan_key:
            return
        status_values = spec.get("status_values")
        if not isinstance(status_values, frozenset):
            return
        compact_rows = _status_span_rows_compact(rows=list(rows or ()), status_values=status_values)
        did_mutate = bool(replace_scope) or bool(compact_rows)
        now_ts = float(pytime.time())
        try:
            with run_cfg_persistent_lock:
                if bool(replace_scope):
                    if has_plan_signature:
                        conn.execute(
                            "DELETE FROM stage_rank_manifest WHERE stage_label=? AND plan_signature=? AND window_signature=?",
                            (str(stage_key), str(plan_key), str(window_key)),
                        )
                    else:
                        conn.execute(
                            "DELETE FROM cartesian_rank_manifest WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        )
                if compact_rows:
                    if has_plan_signature:
                        payload = [
                            (
                                str(stage_key),
                                str(plan_key),
                                str(window_key),
                                int(rank_lo),
                                int(rank_hi),
                                str(status),
                                float(now_ts),
                            )
                            for rank_lo, rank_hi, status in compact_rows
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO stage_rank_manifest("
                            "stage_label, plan_signature, window_signature, rank_lo, rank_hi, status, updated_at"
                            ") VALUES(?,?,?,?,?,?,?)",
                            payload,
                        )
                    else:
                        payload = [
                            (
                                str(stage_key),
                                str(window_key),
                                int(rank_lo),
                                int(rank_hi),
                                str(status),
                                float(now_ts),
                            )
                            for rank_lo, rank_hi, status in compact_rows
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO cartesian_rank_manifest("
                            "stage_label, window_signature, rank_lo, rank_hi, status, updated_at"
                            ") VALUES(?,?,?,?,?,?)",
                            payload,
                        )
            if bool(did_mutate):
                _stage_unresolved_summary_invalidate(
                    manifest_name=str(spec.get("key") or ""),
                    stage_label=str(stage_key),
                    plan_signature=(str(plan_key) if has_plan_signature else ""),
                    window_signature=str(window_key),
                )
            _status_span_manifest_counter_add(
                manifest_name=str(spec.get("key") or ""),
                field="writes",
                value=len(compact_rows),
            )
        except Exception:
            return

    def _status_span_manifest_get_many(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        plan_signature: str = "",
    ) -> list[tuple[int, int, str]]:
        spec = _status_span_manifest_spec(str(manifest_name))
        if not isinstance(spec, dict) or not spec:
            return []
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return []
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip()
        if not stage_key or not window_key:
            return []
        if has_plan_signature and not plan_key:
            return []
        status_values = spec.get("status_values")
        if not isinstance(status_values, frozenset):
            return []
        cfg = _cache_config(str(spec.get("cfg_key") or ""))
        compact_min_rows = max(64, int(_registry_float(cfg.get("compact_min_rows"), float(spec.get("compact_min_rows") or 512.0))))
        compact_min_interval = max(
            5.0,
            float(_registry_float(cfg.get("compact_min_interval_sec"), float(spec.get("compact_min_interval_sec") or 120.0))),
        )
        pending_ttl_sec = max(0.0, float(_registry_float(cfg.get("pending_ttl_sec"), float(spec.get("pending_ttl_sec") or 0.0))))
        now_ts = float(pytime.time())
        out: list[tuple[int, int, str]] = []
        stale_pending = 0
        try:
            with run_cfg_persistent_lock:
                if has_plan_signature:
                    rows_db = conn.execute(
                        "SELECT rank_lo, rank_hi, status, updated_at FROM stage_rank_manifest "
                        "WHERE stage_label=? AND plan_signature=? AND window_signature=?",
                        (str(stage_key), str(plan_key), str(window_key)),
                    ).fetchall()
                else:
                    rows_db = conn.execute(
                        "SELECT rank_lo, rank_hi, status, updated_at FROM cartesian_rank_manifest "
                        "WHERE stage_label=? AND window_signature=?",
                        (str(stage_key), str(window_key)),
                    ).fetchall()
            for row in rows_db:
                try:
                    rank_lo = int(row[0])
                    rank_hi = int(row[1])
                except (TypeError, ValueError):
                    continue
                status = str(row[2] or "").strip().lower()
                try:
                    updated_at = float(row[3] or 0.0)
                except (TypeError, ValueError):
                    updated_at = 0.0
                if rank_lo < 0 or rank_hi < rank_lo or status not in status_values:
                    continue
                if (
                    status == "pending"
                    and float(pending_ttl_sec) > 0.0
                    and float(updated_at) > 0.0
                    and (float(now_ts) - float(updated_at)) > float(pending_ttl_sec)
                ):
                    stale_pending += 1
                    continue
                out.append((int(rank_lo), int(rank_hi), str(status)))
        except Exception:
            return []
        _status_span_manifest_counter_add(
            manifest_name=str(spec.get("key") or ""),
            field="reads",
            value=len(out),
        )
        if not out:
            if int(stale_pending) > 0:
                _status_span_manifest_set_many(
                    manifest_name=str(spec.get("key") or ""),
                    stage_label=str(stage_key),
                    plan_signature=str(plan_key),
                    window_signature=str(window_key),
                    rows=[],
                    replace_scope=True,
                )
                _status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="pending_ttl_prunes",
                    value=int(stale_pending),
                )
            return []

        compacted = _status_span_rows_compact(rows=list(out), status_values=status_values)
        compact_key = (
            str(spec.get("key") or ""),
            str(stage_key),
            str(plan_key if has_plan_signature else ""),
            str(window_key),
        )
        last_ts = float(status_span_manifest_compact_seen.get(compact_key, 0.0) or 0.0)
        interval_ok = (float(now_ts) - float(last_ts)) >= float(compact_min_interval)
        should_rewrite = bool(
            int(stale_pending) > 0
            or (
                interval_ok
                and len(out) >= int(compact_min_rows)
                and len(compacted) < len(out)
            )
        )
        if should_rewrite:
            _status_span_manifest_set_many(
                manifest_name=str(spec.get("key") or ""),
                stage_label=str(stage_key),
                plan_signature=str(plan_key),
                window_signature=str(window_key),
                rows=list(compacted),
                replace_scope=True,
            )
            if len(compacted) < len(out):
                _status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="compactions",
                    value=1,
                )
            if int(stale_pending) > 0:
                _status_span_manifest_counter_add(
                    manifest_name=str(spec.get("key") or ""),
                    field="pending_ttl_prunes",
                    value=int(stale_pending),
                )
            out = list(compacted)
        status_span_manifest_compact_seen[compact_key] = float(now_ts)
        return out

    def _status_span_manifest_unresolved_ranges(
        *,
        manifest_name: str,
        stage_label: str,
        window_signature: str,
        total: int,
        plan_signature: str = "",
    ) -> tuple[tuple[int, int], ...]:
        total_i = int(total)
        if total_i <= 0:
            return ()
        spec = _status_span_manifest_spec(str(manifest_name))
        manifest_key = str(spec.get("key") or str(manifest_name)).strip().lower()
        has_plan_signature = bool(spec.get("has_plan_signature"))
        plan_key = str(plan_signature).strip() if has_plan_signature else ""
        if has_plan_signature and not plan_key:
            return ((0, int(total_i - 1)),)
        summary = _stage_unresolved_summary_get(
            manifest_name=str(manifest_key),
            stage_label=str(stage_label),
            plan_signature=str(plan_key),
            window_signature=str(window_signature),
            total=int(total_i),
        )
        if isinstance(summary, tuple) and len(summary) == 2:
            try:
                unresolved_count = int(summary[0])
                resolved_count = int(summary[1])
            except (TypeError, ValueError):
                unresolved_count = -1
                resolved_count = -1
            if unresolved_count == 0:
                if resolved_count > 0:
                    _status_span_manifest_counter_add_hits(
                        manifest_name=str(manifest_key),
                        covered=int(resolved_count),
                    )
                return ()
            if unresolved_count >= int(total_i):
                return ((0, int(total_i - 1)),)
        rows = _status_span_manifest_get_many(
            manifest_name=str(manifest_name),
            stage_label=str(stage_label),
            window_signature=str(window_signature),
            plan_signature=str(plan_signature),
        )
        if not rows:
            _stage_unresolved_summary_set(
                manifest_name=str(manifest_key),
                stage_label=str(stage_label),
                plan_signature=str(plan_key),
                window_signature=str(window_signature),
                total=int(total_i),
                unresolved_count=int(total_i),
                resolved_count=0,
            )
            return ((0, int(total_i - 1)),)
        resolved_ranges: list[tuple[int, int]] = []
        for rank_lo, rank_hi, status in rows:
            if str(status) not in ("cached_hit", "evaluated", "dominated"):
                continue
            lo = max(0, int(rank_lo))
            hi = min(int(total_i - 1), int(rank_hi))
            if hi < lo:
                continue
            resolved_ranges.append((int(lo), int(hi)))
        if not resolved_ranges:
            _stage_unresolved_summary_set(
                manifest_name=str(manifest_key),
                stage_label=str(stage_label),
                plan_signature=str(plan_key),
                window_signature=str(window_signature),
                total=int(total_i),
                unresolved_count=int(total_i),
                resolved_count=0,
            )
            return ((0, int(total_i - 1)),)
        resolved_ranges.sort(key=lambda row: (int(row[0]), int(row[1])))
        merged: list[tuple[int, int]] = []
        for lo, hi in resolved_ranges:
            if not merged:
                merged.append((int(lo), int(hi)))
                continue
            prev_lo, prev_hi = merged[-1]
            if int(lo) <= int(prev_hi) + 1:
                merged[-1] = (int(prev_lo), max(int(prev_hi), int(hi)))
            else:
                merged.append((int(lo), int(hi)))
        unresolved: list[tuple[int, int]] = []
        cursor = 0
        covered = 0
        for lo, hi in merged:
            lo_i = max(0, int(lo))
            hi_i = min(int(total_i - 1), int(hi))
            if hi_i < lo_i:
                continue
            if cursor < lo_i:
                unresolved.append((int(cursor), int(lo_i - 1)))
            covered += max(0, int(hi_i - lo_i + 1))
            cursor = int(max(cursor, hi_i + 1))
        if cursor < total_i:
            unresolved.append((int(cursor), int(total_i - 1)))
        if covered > 0:
            _status_span_manifest_counter_add_hits(
                manifest_name=str(manifest_key),
                covered=int(covered),
            )
        unresolved_count = sum(max(0, int(hi) - int(lo) + 1) for lo, hi in unresolved)
        _stage_unresolved_summary_set(
            manifest_name=str(manifest_key),
            stage_label=str(stage_label),
            plan_signature=str(plan_key),
            window_signature=str(window_signature),
            total=int(total_i),
            unresolved_count=int(unresolved_count),
            resolved_count=int(max(0, int(total_i) - int(unresolved_count))),
        )
        return tuple(unresolved)

    def _cartesian_rank_manifest_get_many(
        *,
        stage_label: str,
        window_signature: str,
    ) -> list[tuple[int, int, str]]:
        return _status_span_manifest_get_many(
            manifest_name="cartesian",
            stage_label=str(stage_label),
            window_signature=str(window_signature),
        )

    def _cartesian_rank_manifest_set_many(
        *,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
    ) -> None:
        _status_span_manifest_set_many(
            manifest_name="cartesian",
            stage_label=str(stage_label),
            window_signature=str(window_signature),
            rows=list(rows or ()),
        )

    def _stage_rank_manifest_set_many(
        *,
        stage_label: str,
        plan_signature: str,
        window_signature: str,
        rows: list[tuple[int, int, str]],
    ) -> None:
        _status_span_manifest_set_many(
            manifest_name="stage",
            stage_label=str(stage_label),
            plan_signature=str(plan_signature),
            window_signature=str(window_signature),
            rows=list(rows or ()),
        )

    def _stage_rank_manifest_unresolved_ranges(
        *,
        stage_label: str,
        plan_signature: str,
        window_signature: str,
        total: int,
    ) -> tuple[tuple[int, int], ...]:
        return _status_span_manifest_unresolved_ranges(
            manifest_name="stage",
            stage_label=str(stage_label),
            plan_signature=str(plan_signature),
            window_signature=str(window_signature),
            total=int(total),
        )

    def _rank_dominance_stamp_get_many(
        *,
        stage_label: str,
        window_signature: str,
    ) -> list[tuple[str, int, int]]:
        nonlocal rank_dominance_stamp_reads, rank_dominance_stamp_compactions, rank_dominance_stamp_ttl_prunes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return []
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return []
        cfg = _cache_config("rank_dominance_stamp")
        compact_min_rows = max(64, int(_registry_float(cfg.get("compact_min_rows"), 512.0)))
        compact_min_interval = max(5.0, float(_registry_float(cfg.get("compact_min_interval_sec"), 120.0)))
        ttl_sec = max(0.0, float(_registry_float(cfg.get("ttl_sec"), 1209600.0)))
        now_ts = float(pytime.time())
        compact_key = (str(stage_key), str(window_key))
        out: list[tuple[str, int, int]] = []
        stale_rows = 0
        should_consider_compact = False
        try:
            with run_cfg_persistent_lock:
                rows_db = conn.execute(
                    "SELECT dominance_signature, rank_lo, rank_hi, updated_at FROM rank_dominance_stamp "
                    "WHERE stage_label=? AND window_signature=?",
                    (str(stage_key), str(window_key)),
                ).fetchall()
            for row in rows_db:
                sig = str(row[0] or "").strip().lower()
                try:
                    rank_lo = int(row[1])
                    rank_hi = int(row[2])
                except (TypeError, ValueError):
                    continue
                try:
                    updated_at = float(row[3] or 0.0)
                except (TypeError, ValueError):
                    updated_at = 0.0
                if not sig or rank_lo < 0 or rank_hi < rank_lo:
                    continue
                if float(ttl_sec) > 0.0 and float(updated_at) > 0.0 and (float(now_ts) - float(updated_at)) > float(ttl_sec):
                    stale_rows += 1
                    continue
                out.append((str(sig), int(rank_lo), int(rank_hi)))
            rank_dominance_stamp_reads += len(out)
            should_consider_compact = (
                len(out) >= int(compact_min_rows)
                or int(stale_rows) > 0
            )
        except Exception:
            return []
        if not out:
            if int(stale_rows) > 0:
                try:
                    with run_cfg_persistent_lock:
                        conn.execute(
                            "DELETE FROM rank_dominance_stamp WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        )
                except Exception:
                    pass
                rank_dominance_stamp_ttl_prunes += int(stale_rows)
            return []
        last_compact_ts = float(rank_dominance_stamp_compact_seen.get(compact_key, 0.0) or 0.0)
        should_compact_now = bool(should_consider_compact and (float(now_ts) - float(last_compact_ts)) >= float(compact_min_interval))
        if not should_compact_now:
            return out
        grouped: dict[str, list[tuple[int, int]]] = {}
        for sig, rank_lo, rank_hi in out:
            grouped.setdefault(str(sig), []).append((int(rank_lo), int(rank_hi)))
        compacted: list[tuple[str, int, int]] = []
        for sig, ranges in grouped.items():
            ordered = sorted(ranges, key=lambda row: (int(row[0]), int(row[1])))
            if not ordered:
                continue
            cur_lo = int(ordered[0][0])
            cur_hi = int(ordered[0][1])
            for rank_lo, rank_hi in ordered[1:]:
                if int(rank_lo) <= int(cur_hi) + 1:
                    cur_hi = max(int(cur_hi), int(rank_hi))
                else:
                    compacted.append((str(sig), int(cur_lo), int(cur_hi)))
                    cur_lo = int(rank_lo)
                    cur_hi = int(rank_hi)
            compacted.append((str(sig), int(cur_lo), int(cur_hi)))
        rewrite_needed = bool(int(stale_rows) > 0 or len(compacted) < len(out))
        if rewrite_needed:
            try:
                with run_cfg_persistent_lock:
                    conn.execute(
                        "DELETE FROM rank_dominance_stamp WHERE stage_label=? AND window_signature=?",
                        (str(stage_key), str(window_key)),
                    )
                _rank_dominance_stamp_set_many(
                    stage_label=str(stage_key),
                    window_signature=str(window_key),
                    rows=list(compacted),
                )
                out = list(compacted)
                rank_dominance_stamp_compactions += 1
            except Exception:
                pass
        if int(stale_rows) > 0:
            rank_dominance_stamp_ttl_prunes += int(stale_rows)
        rank_dominance_stamp_compact_seen[compact_key] = float(now_ts)
        return out

    def _rank_dominance_stamp_set_many(
        *,
        stage_label: str,
        window_signature: str,
        rows: list[tuple[str, int, int]],
    ) -> None:
        nonlocal rank_dominance_stamp_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key or not rows:
            return
        grouped: dict[str, list[tuple[int, int]]] = {}
        for sig_raw, rank_lo_raw, rank_hi_raw in rows:
            sig = str(sig_raw or "").strip().lower()
            try:
                rank_lo = int(rank_lo_raw)
                rank_hi = int(rank_hi_raw)
            except (TypeError, ValueError):
                continue
            if not sig or rank_lo < 0 or rank_hi < rank_lo:
                continue
            grouped.setdefault(str(sig), []).append((int(rank_lo), int(rank_hi)))
        if not grouped:
            return
        payload: list[tuple[str, str, str, int, int, float]] = []
        now_ts = float(pytime.time())
        for sig, ranges in grouped.items():
            ordered = sorted(ranges, key=lambda row: (int(row[0]), int(row[1])))
            if not ordered:
                continue
            cur_lo = int(ordered[0][0])
            cur_hi = int(ordered[0][1])
            for rank_lo, rank_hi in ordered[1:]:
                if int(rank_lo) <= int(cur_hi) + 1:
                    cur_hi = max(int(cur_hi), int(rank_hi))
                else:
                    payload.append((str(stage_key), str(window_key), str(sig), int(cur_lo), int(cur_hi), float(now_ts)))
                    cur_lo = int(rank_lo)
                    cur_hi = int(rank_hi)
            payload.append((str(stage_key), str(window_key), str(sig), int(cur_lo), int(cur_hi), float(now_ts)))
        if not payload:
            return
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO rank_dominance_stamp("
                    "stage_label, window_signature, dominance_signature, rank_lo, rank_hi, updated_at"
                    ") VALUES(?,?,?,?,?,?)",
                    payload,
                )
            rank_dominance_stamp_writes += len(payload)
        except Exception:
            return

    def _apply_rank_dominance_stamps_to_manifest(
        *,
        stage_label: str,
        window_signature: str,
        total: int,
    ) -> None:
        nonlocal rank_dominance_stamp_hits, rank_dominance_manifest_applies
        total_i = int(total)
        if total_i <= 0:
            return
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return
        seen_key = (str(stage_key), str(window_key))
        if seen_key in rank_dominance_manifest_applied_seen:
            return
        stamp_rows = _rank_dominance_stamp_get_many(
            stage_label=str(stage_key),
            window_signature=str(window_key),
        )
        rank_dominance_manifest_applied_seen.add(seen_key)
        if not stamp_rows:
            return
        manifest_rows: list[tuple[int, int, str]] = []
        covered = 0
        for _sig, rank_lo, rank_hi in stamp_rows:
            lo_i = max(0, int(rank_lo))
            hi_i = min(int(total_i - 1), int(rank_hi))
            if hi_i < lo_i:
                continue
            manifest_rows.append((int(lo_i), int(hi_i), "dominated"))
            covered += max(0, int(hi_i - lo_i + 1))
        if not manifest_rows:
            return
        _cartesian_rank_manifest_set_many(
            stage_label=str(stage_key),
            window_signature=str(window_key),
            rows=manifest_rows,
        )
        rank_dominance_manifest_applies += len(manifest_rows)
        rank_dominance_stamp_hits += int(covered)

    def _cartesian_rank_manifest_unresolved_ranges(
        *,
        stage_label: str,
        window_signature: str,
        total: int,
    ) -> tuple[tuple[int, int], ...]:
        total_i = int(total)
        if total_i <= 0:
            return ()
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        _apply_rank_dominance_stamps_to_manifest(
            stage_label=str(stage_key),
            window_signature=str(window_key),
            total=int(total_i),
        )
        return _status_span_manifest_unresolved_ranges(
            manifest_name="cartesian",
            stage_label=str(stage_key),
            window_signature=str(window_key),
            total=int(total_i),
        )

    def _cartesian_rank_manifest_claim_next_range(
        *,
        stage_label: str,
        window_signature: str,
        total: int,
        max_span: int,
    ) -> tuple[int, int] | None:
        total_i = int(total)
        if total_i <= 0:
            return None
        max_span_i = max(1, int(max_span))
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return None
        stage_key = _stage_cache_scope(stage_label)
        window_key = str(window_signature).strip()
        if not stage_key or not window_key:
            return None
        cfg = _cache_config("cartesian_rank_manifest")
        pending_ttl_sec = max(0.0, float(_registry_float(cfg.get("pending_ttl_sec"), 86400.0)))
        now_ts = float(pytime.time())
        stale_pending_pruned = 0
        reads_seen = 0
        claim_row: tuple[int, int] | None = None
        attempts = 6
        status_values = ("pending", "cached_hit", "evaluated", "dominated")
        status_placeholders = ",".join("?" for _ in status_values)
        for attempt_idx in range(attempts):
            try:
                with run_cfg_persistent_lock:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        if float(pending_ttl_sec) > 0.0:
                            stale_pending_pruned = int(
                                conn.execute(
                                    "SELECT COUNT(*) FROM cartesian_rank_manifest "
                                    "WHERE stage_label=? AND window_signature=? AND status='pending' "
                                    "AND updated_at>0 AND (? - updated_at) > ?",
                                    (str(stage_key), str(window_key), float(now_ts), float(pending_ttl_sec)),
                                ).fetchone()[0]
                                or 0
                            )
                            if int(stale_pending_pruned) > 0:
                                conn.execute(
                                    "DELETE FROM cartesian_rank_manifest "
                                    "WHERE stage_label=? AND window_signature=? AND status='pending' "
                                    "AND updated_at>0 AND (? - updated_at) > ?",
                                    (str(stage_key), str(window_key), float(now_ts), float(pending_ttl_sec)),
                                )

                        cursor_row = conn.execute(
                            "SELECT next_rank FROM cartesian_rank_cursor "
                            "WHERE stage_label=? AND window_signature=?",
                            (str(stage_key), str(window_key)),
                        ).fetchone()
                        reads_seen += 1
                        cursor = 0
                        if isinstance(cursor_row, tuple) and len(cursor_row) >= 1:
                            try:
                                cursor = int(cursor_row[0])
                            except (TypeError, ValueError):
                                cursor = 0
                        cursor = max(0, min(int(total_i), int(cursor)))
                        if int(stale_pending_pruned) > 0 and int(cursor) > 0:
                            # Pending TTL pruning can reopen gaps behind the persisted cursor.
                            cursor = 0

                        while int(cursor) < int(total_i):
                            cover_row = conn.execute(
                                "SELECT rank_lo, rank_hi FROM cartesian_rank_manifest "
                                "WHERE stage_label=? AND window_signature=? "
                                "AND rank_lo<=? AND rank_hi>=? "
                                f"AND status IN ({status_placeholders}) "
                                "ORDER BY rank_hi DESC LIMIT 1",
                                (
                                    str(stage_key),
                                    str(window_key),
                                    int(cursor),
                                    int(cursor),
                                    *status_values,
                                ),
                            ).fetchone()
                            reads_seen += 1
                            if not (isinstance(cover_row, tuple) and len(cover_row) >= 2):
                                break
                            try:
                                cover_hi = int(cover_row[1])
                            except (TypeError, ValueError):
                                break
                            if int(cover_hi) < int(cursor):
                                break
                            cursor = int(min(int(total_i), int(cover_hi) + 1))

                        if int(cursor) < int(total_i):
                            next_row = conn.execute(
                                "SELECT rank_lo, rank_hi FROM cartesian_rank_manifest "
                                "WHERE stage_label=? AND window_signature=? AND rank_lo>=? "
                                f"AND status IN ({status_placeholders}) "
                                "ORDER BY rank_lo ASC LIMIT 1",
                                (
                                    str(stage_key),
                                    str(window_key),
                                    int(cursor),
                                    *status_values,
                                ),
                            ).fetchone()
                            reads_seen += 1
                            next_lo = int(total_i)
                            next_hi = None
                            if isinstance(next_row, tuple) and len(next_row) >= 2:
                                try:
                                    next_lo = int(next_row[0])
                                    next_hi = int(next_row[1])
                                except (TypeError, ValueError):
                                    next_lo = int(total_i)
                                    next_hi = None
                            if int(next_lo) <= int(cursor):
                                if isinstance(next_hi, int) and int(next_hi) >= int(cursor):
                                    cursor = int(min(int(total_i), int(next_hi) + 1))
                                else:
                                    cursor = int(min(int(total_i), int(cursor) + 1))
                            else:
                                claim_lo = int(cursor)
                                gap_hi = int(min(int(total_i - 1), int(next_lo) - 1))
                                claim_hi = int(min(int(gap_hi), int(claim_lo + int(max_span_i) - 1)))
                                if int(claim_hi) >= int(claim_lo):
                                    conn.execute(
                                        "INSERT OR REPLACE INTO cartesian_rank_manifest("
                                        "stage_label, window_signature, rank_lo, rank_hi, status, updated_at"
                                        ") VALUES(?,?,?,?,?,?)",
                                        (
                                            str(stage_key),
                                            str(window_key),
                                            int(claim_lo),
                                            int(claim_hi),
                                            "pending",
                                            float(now_ts),
                                        ),
                                    )
                                    claim_row = (int(claim_lo), int(claim_hi))
                                    cursor = int(min(int(total_i), int(claim_hi) + 1))

                        conn.execute(
                            "INSERT OR REPLACE INTO cartesian_rank_cursor("
                            "stage_label, window_signature, next_rank, updated_at"
                            ") VALUES(?,?,?,?)",
                            (str(stage_key), str(window_key), int(cursor), float(now_ts)),
                        )
                        conn.execute("COMMIT")
                    except Exception:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                break
            except sqlite3.OperationalError as exc:
                msg = str(exc or "").lower()
                if "locked" in msg and int(attempt_idx) < int(attempts - 1):
                    pytime.sleep(min(0.1, 0.01 * float(attempt_idx + 1)))
                    continue
                return None
            except Exception:
                return None

        if int(reads_seen) > 0:
            _status_span_manifest_counter_add(manifest_name="cartesian", field="reads", value=int(reads_seen))
        if int(stale_pending_pruned) > 0:
            _status_span_manifest_counter_add(
                manifest_name="cartesian",
                field="pending_ttl_prunes",
                value=int(stale_pending_pruned),
            )
        _status_span_manifest_counter_add(manifest_name="cartesian", field="writes", value=1)
        if isinstance(claim_row, tuple):
            _status_span_manifest_counter_add(manifest_name="cartesian", field="writes", value=1)
        return claim_row

    def _series_pack_state_manifest_get_many(
        *,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], str]:
        nonlocal series_pack_state_manifest_reads, series_pack_state_manifest_hits
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for strategy_fp, window_sig in cells:
            key = (str(strategy_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], str] = {}
        chunk_size = 140
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(strategy_fingerprint=? AND window_signature=?)" for _ in chunk)
                    params: list[object] = []
                    for strategy_fp, window_sig in chunk:
                        params.extend((str(strategy_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT strategy_fingerprint, window_signature, state "
                        "FROM series_pack_state_manifest WHERE " + where,
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        strategy_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        state = str(row[2] or "").strip().lower()
                        if not strategy_fp or not window_sig:
                            continue
                        if state not in ("mmap", "pickle", "none"):
                            continue
                        out[(str(strategy_fp), str(window_sig))] = str(state)
            series_pack_state_manifest_reads += len(out)
            series_pack_state_manifest_hits += len(out)
        except Exception:
            return {}
        return out

    def _series_pack_state_manifest_set_many(
        *,
        rows: list[tuple[str, str, str]],
    ) -> None:
        nonlocal series_pack_state_manifest_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        if not rows:
            return
        merged: dict[tuple[str, str], str] = {}
        state_priority = {"none": 0, "pickle": 1, "mmap": 2}
        for strategy_fp_raw, window_sig_raw, state_raw in rows:
            strategy_fp = str(strategy_fp_raw).strip()
            window_sig = str(window_sig_raw).strip()
            state = str(state_raw or "").strip().lower()
            if not strategy_fp or not window_sig or state not in state_priority:
                continue
            key = (str(strategy_fp), str(window_sig))
            prev = merged.get(key)
            if prev is None or state_priority.get(state, -1) >= state_priority.get(str(prev), -1):
                merged[key] = str(state)
        if not merged:
            return
        payload = [
            (str(strategy_fp), str(window_sig), str(state), float(pytime.time()))
            for (strategy_fp, window_sig), state in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO series_pack_state_manifest("
                    "strategy_fingerprint, window_signature, state, updated_at"
                    ") VALUES(?,?,?,?) "
                    "ON CONFLICT(strategy_fingerprint, window_signature) DO UPDATE SET "
                    "state=excluded.state, updated_at=excluded.updated_at",
                    payload,
                )
            series_pack_state_manifest_writes += len(payload)
        except Exception:
            return

    def _rank_bin_from_rank(rank: int) -> int:
        return int(max(0, int(rank)) // int(max(1, int(_RANK_BIN_SIZE))))

    def _rank_bin_runtime_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, int]],
    ) -> dict[tuple[str, int], dict[str, float]]:
        nonlocal rank_bin_runtime_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for window_sig, rank_bin in cells:
            window_key = str(window_sig).strip()
            try:
                bin_i = int(rank_bin)
            except (TypeError, ValueError):
                continue
            key = (window_key, int(bin_i))
            if not window_key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, int], dict[str, float]] = {}
        chunk_size = 160
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(window_signature=? AND rank_bin=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for window_sig, rank_bin in chunk:
                        params.extend((str(window_sig), int(rank_bin)))
                    rows_db = conn.execute(
                        "SELECT window_signature, rank_bin, eval_count, total_eval_sec, cache_hits "
                        "FROM rank_bin_runtime WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        window_sig = str(row[0] or "").strip()
                        try:
                            rank_bin = int(row[1])
                            eval_count = int(row[2] or 0)
                            total_eval_sec = float(row[3] or 0.0)
                            cache_hits = int(row[4] or 0)
                        except Exception:
                            continue
                        if not window_sig or eval_count <= 0:
                            continue
                        avg_eval_sec = float(total_eval_sec) / float(max(1, int(eval_count)))
                        hit_rate = float(cache_hits) / float(max(1, int(eval_count)))
                        out[(window_sig, int(rank_bin))] = {
                            "avg_eval_sec": float(avg_eval_sec),
                            "hit_rate": float(max(0.0, min(1.0, hit_rate))),
                            "eval_count": float(eval_count),
                        }
            rank_bin_runtime_reads += len(out)
        except Exception:
            return {}
        return out

    def _rank_bin_runtime_set_many(
        *,
        stage_label: str,
        rows: list[tuple[str, int, int, float, int]],
    ) -> None:
        nonlocal rank_bin_runtime_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, int], tuple[int, float, int]] = {}
        for window_sig, rank_bin, eval_count, total_eval_sec, cache_hits in rows:
            window_key = str(window_sig).strip()
            try:
                bin_i = int(rank_bin)
                eval_i = int(eval_count)
                sec_f = float(total_eval_sec)
                hit_i = int(cache_hits)
            except Exception:
                continue
            if not window_key or eval_i <= 0 or sec_f < 0.0:
                continue
            key = (window_key, int(bin_i))
            prev = merged.get(key)
            if prev is None:
                merged[key] = (int(eval_i), float(sec_f), int(max(0, hit_i)))
            else:
                merged[key] = (
                    int(prev[0]) + int(eval_i),
                    float(prev[1]) + float(sec_f),
                    int(prev[2]) + int(max(0, hit_i)),
                )
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(window_sig),
                int(rank_bin),
                int(rec[0]),
                float(rec[1]),
                int(rec[2]),
                now_ts,
            )
            for (window_sig, rank_bin), rec in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO rank_bin_runtime("
                    "stage_label, window_signature, rank_bin, eval_count, total_eval_sec, cache_hits, updated_at"
                    ") VALUES(?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, window_signature, rank_bin) DO UPDATE SET "
                    "eval_count=rank_bin_runtime.eval_count + excluded.eval_count, "
                    "total_eval_sec=rank_bin_runtime.total_eval_sec + excluded.total_eval_sec, "
                    "cache_hits=rank_bin_runtime.cache_hits + excluded.cache_hits, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            rank_bin_runtime_writes += len(payload)
        except Exception:
            return

    def _dimension_value_utility_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], dict[str, float]]:
        nonlocal dimension_utility_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for window_sig, dim_key, dim_value in cells:
            key = (str(window_sig).strip(), str(dim_key).strip(), str(dim_value).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str, str], dict[str, float]] = {}
        chunk_size = 120
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join(
                        "(window_signature=? AND dimension_key=? AND dimension_value=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key)]
                    for window_sig, dim_key, dim_value in chunk:
                        params.extend((str(window_sig), str(dim_key), str(dim_value)))
                    rows_db = conn.execute(
                        "SELECT window_signature, dimension_key, dimension_value, eval_count, keep_count, cache_hits, total_eval_sec "
                        "FROM dimension_value_utility WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        window_sig = str(row[0] or "").strip()
                        dim_key = str(row[1] or "").strip()
                        dim_value = str(row[2] or "").strip()
                        try:
                            eval_count = int(row[3] or 0)
                            keep_count = int(row[4] or 0)
                            cache_hits = int(row[5] or 0)
                            total_eval_sec = float(row[6] or 0.0)
                        except Exception:
                            continue
                        if not window_sig or not dim_key or eval_count <= 0:
                            continue
                        out[(window_sig, dim_key, dim_value)] = {
                            "eval_count": float(eval_count),
                            "keep_rate": float(max(0.0, min(1.0, float(keep_count) / float(max(1, eval_count))))),
                            "hit_rate": float(max(0.0, min(1.0, float(cache_hits) / float(max(1, eval_count))))),
                            "avg_eval_sec": float(total_eval_sec) / float(max(1, eval_count)),
                        }
            dimension_utility_reads += len(out)
        except Exception:
            return {}
        return out

    def _dimension_value_utility_set_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, str, int, int, int, float]],
    ) -> None:
        nonlocal dimension_utility_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, str, str], tuple[int, int, int, float]] = {}
        for window_sig, dim_key, dim_value, eval_count, keep_count, cache_hits, total_eval_sec in rows:
            window_key = str(window_sig).strip()
            dim_key_s = str(dim_key).strip()
            dim_value_s = str(dim_value).strip()
            try:
                eval_i = int(eval_count)
                keep_i = int(keep_count)
                hit_i = int(cache_hits)
                sec_f = float(total_eval_sec)
            except Exception:
                continue
            if not window_key or not dim_key_s or eval_i <= 0 or sec_f < 0.0:
                continue
            key = (window_key, dim_key_s, dim_value_s)
            prev = merged.get(key)
            keep_i = int(max(0, min(eval_i, keep_i)))
            hit_i = int(max(0, min(eval_i, hit_i)))
            if prev is None:
                merged[key] = (int(eval_i), int(keep_i), int(hit_i), float(sec_f))
            else:
                merged[key] = (
                    int(prev[0]) + int(eval_i),
                    int(prev[1]) + int(keep_i),
                    int(prev[2]) + int(hit_i),
                    float(prev[3]) + float(sec_f),
                )
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(window_sig),
                str(dim_key),
                str(dim_value),
                int(rec[0]),
                int(rec[1]),
                int(rec[2]),
                float(rec[3]),
                now_ts,
            )
            for (window_sig, dim_key, dim_value), rec in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO dimension_value_utility("
                    "stage_label, window_signature, dimension_key, dimension_value, eval_count, keep_count, cache_hits, total_eval_sec, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, window_signature, dimension_key, dimension_value) DO UPDATE SET "
                    "eval_count=dimension_value_utility.eval_count + excluded.eval_count, "
                    "keep_count=dimension_value_utility.keep_count + excluded.keep_count, "
                    "cache_hits=dimension_value_utility.cache_hits + excluded.cache_hits, "
                    "total_eval_sec=dimension_value_utility.total_eval_sec + excluded.total_eval_sec, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            dimension_utility_writes += len(payload)
        except Exception:
            return

    def _dimension_upper_bound_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        nonlocal dimension_upper_bound_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for axis_dim_fp, window_sig in cells:
            key = (str(axis_dim_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], dict[str, object]] = {}
        chunk_size = 140
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join("(axis_dimension_fingerprint=? AND window_signature=?)" for _ in chunk)
                    params: list[object] = [str(stage_key)]
                    for axis_dim_fp, window_sig in chunk:
                        params.extend((str(axis_dim_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT axis_dimension_fingerprint, window_signature, eval_count, keep_count, best_pnl_over_dd, best_pnl "
                        "FROM dimension_upper_bound WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        axis_dim_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        if not axis_dim_fp or not window_sig:
                            continue
                        out[(axis_dim_fp, window_sig)] = {
                            "eval_count": int(row[2] or 0),
                            "keep_count": int(row[3] or 0),
                            "best_pnl_over_dd": (None if row[4] is None else float(row[4])),
                            "best_pnl": (None if row[5] is None else float(row[5])),
                        }
            dimension_upper_bound_reads += len(out)
        except Exception:
            return {}
        return out

    def _dimension_upper_bound_upsert_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, dict | None]],
    ) -> None:
        nonlocal dimension_upper_bound_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for axis_dim_fp, window_sig, row in rows:
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            if not axis_key or not window_key:
                continue
            key = (axis_key, window_key)
            rec = merged.get(key)
            if rec is None:
                rec = {
                    "eval_count": 0,
                    "keep_count": 0,
                    "best_pnl_over_dd": None,
                    "best_pnl": None,
                }
                merged[key] = rec
            rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
            if not isinstance(row, dict):
                continue
            rec["keep_count"] = int(rec.get("keep_count") or 0) + 1
            for metric_key, rec_key, caster in (
                ("pnl_over_dd", "best_pnl_over_dd", float),
                ("pnl", "best_pnl", float),
            ):
                raw_val = row.get(metric_key)
                if raw_val is None:
                    continue
                try:
                    val = caster(raw_val)
                except (TypeError, ValueError):
                    continue
                prev = rec.get(rec_key)
                if prev is None or float(val) > float(prev):
                    rec[rec_key] = val
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(axis_dim_fp),
                str(window_sig),
                int(rec.get("eval_count") or 0),
                int(rec.get("keep_count") or 0),
                rec.get("best_pnl_over_dd"),
                rec.get("best_pnl"),
                now_ts,
            )
            for (axis_dim_fp, window_sig), rec in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO dimension_upper_bound("
                    "stage_label, axis_dimension_fingerprint, window_signature, eval_count, keep_count, best_pnl_over_dd, best_pnl, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, axis_dimension_fingerprint, window_signature) DO UPDATE SET "
                    "eval_count=dimension_upper_bound.eval_count + excluded.eval_count, "
                    "keep_count=dimension_upper_bound.keep_count + excluded.keep_count, "
                    "best_pnl_over_dd=CASE "
                    "WHEN dimension_upper_bound.best_pnl_over_dd IS NULL THEN excluded.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd IS NULL THEN dimension_upper_bound.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd > dimension_upper_bound.best_pnl_over_dd THEN excluded.best_pnl_over_dd "
                    "ELSE dimension_upper_bound.best_pnl_over_dd END, "
                    "best_pnl=CASE "
                    "WHEN dimension_upper_bound.best_pnl IS NULL THEN excluded.best_pnl "
                    "WHEN excluded.best_pnl IS NULL THEN dimension_upper_bound.best_pnl "
                    "WHEN excluded.best_pnl > dimension_upper_bound.best_pnl THEN excluded.best_pnl "
                    "ELSE dimension_upper_bound.best_pnl END, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            dimension_upper_bound_writes += len(payload)
        except Exception:
            return

    def _dimension_upper_bound_score(frontier_row: dict[str, object] | None) -> float:
        if not isinstance(frontier_row, dict):
            return 0.0
        cfg = _cache_config("dimension_upper_bound")
        min_eval_count = max(1, int(_registry_float(cfg.get("min_eval_count"), 6.0)))
        low_ceiling_max_keep_count = max(0, int(_registry_float(cfg.get("low_ceiling_max_keep_count"), 0.0)))
        low_ceiling_max_best_pnl = float(_registry_float(cfg.get("low_ceiling_max_best_pnl"), 0.0))
        low_ceiling_max_best_pnl_dd = float(_registry_float(cfg.get("low_ceiling_max_best_pnl_over_dd"), 0.0))
        confidence_eval_scale = max(1.0, float(_registry_float(cfg.get("confidence_eval_scale"), 24.0)))
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        if eval_count < int(min_eval_count):
            return 0.0
        best_pnl_raw = frontier_row.get("best_pnl")
        best_pnl_dd_raw = frontier_row.get("best_pnl_over_dd")
        best_pnl = float(best_pnl_raw) if best_pnl_raw is not None else float("-inf")
        best_pnl_dd = float(best_pnl_dd_raw) if best_pnl_dd_raw is not None else float("-inf")
        is_low_ceiling = bool(
            keep_count <= int(low_ceiling_max_keep_count)
            and best_pnl <= float(low_ceiling_max_best_pnl)
            and best_pnl_dd <= float(low_ceiling_max_best_pnl_dd)
        )
        if is_low_ceiling:
            return -1.0
        confidence = float(min(1.0, float(eval_count) / float(confidence_eval_scale)))
        keep_rate = float(max(0.0, min(1.0, float(keep_count) / float(max(1, eval_count)))))
        upside = (0.65 * max(0.0, float(best_pnl_dd if math.isfinite(best_pnl_dd) else 0.0))) + (
            0.35 * max(0.0, float(best_pnl if math.isfinite(best_pnl) else 0.0))
        )
        return float((confidence * upside) + (0.25 * keep_rate))

    def _upper_bound_dominance_signature(frontier_row: dict[str, object] | None) -> str:
        if not isinstance(frontier_row, dict):
            return ""
        score = float(_dimension_upper_bound_score(frontier_row))
        if score >= 0.0:
            return ""
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        best_pnl_raw = frontier_row.get("best_pnl")
        best_pnl_dd_raw = frontier_row.get("best_pnl_over_dd")
        try:
            best_pnl = None if best_pnl_raw is None else round(float(best_pnl_raw), 6)
        except (TypeError, ValueError):
            best_pnl = None
        try:
            best_pnl_dd = None if best_pnl_dd_raw is None else round(float(best_pnl_dd_raw), 6)
        except (TypeError, ValueError):
            best_pnl_dd = None
        raw = {
            "rule": "upper_bound_low_ceiling",
            "eval_count": int(eval_count),
            "keep_count": int(keep_count),
            "best_pnl": best_pnl,
            "best_pnl_over_dd": best_pnl_dd,
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _planner_heartbeat_set(
        *,
        stage_label: str,
        worker_id: int,
        tested: int,
        cached_hits: int,
        total: int,
        eta_sec: float | None,
        status: str,
    ) -> None:
        nonlocal planner_heartbeat_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return
        try:
            worker_i = int(worker_id)
            tested_i = int(max(0, int(tested)))
            cached_i = int(max(0, int(cached_hits)))
            total_i = int(max(0, int(total)))
            eta_f = None if eta_sec is None else float(max(0.0, float(eta_sec)))
            status_s = str(status or "").strip().lower() or "running"
        except Exception:
            return
        now_ts = float(pytime.time())
        try:
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT INTO planner_heartbeat("
                    "stage_label, worker_id, last_seen, tested, cached_hits, total, eta_sec, status, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, worker_id) DO UPDATE SET "
                    "last_seen=excluded.last_seen, "
                    "tested=excluded.tested, "
                    "cached_hits=excluded.cached_hits, "
                    "total=excluded.total, "
                    "eta_sec=excluded.eta_sec, "
                    "status=excluded.status, "
                    "updated_at=excluded.updated_at",
                    (
                        str(stage_key),
                        int(worker_i),
                        float(now_ts),
                        int(tested_i),
                        int(cached_i),
                        int(total_i),
                        (float(eta_f) if eta_f is not None else None),
                        str(status_s),
                        float(now_ts),
                    ),
                )
            planner_heartbeat_writes += 1
        except Exception:
            return

    def _planner_heartbeat_clear_stage(*, stage_label: str) -> None:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return
        try:
            with run_cfg_persistent_lock:
                conn.execute(
                    "DELETE FROM planner_heartbeat WHERE stage_label=?",
                    (str(stage_key),),
                )
        except Exception:
            return

    def _planner_heartbeat_get_many(
        *,
        stage_label: str,
        worker_ids: list[int],
    ) -> dict[int, dict[str, float | int | str | None]]:
        nonlocal planner_heartbeat_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        worker_ids_clean: list[int] = []
        seen: set[int] = set()
        for raw in worker_ids:
            try:
                worker_i = int(raw)
            except (TypeError, ValueError):
                continue
            if worker_i < 0 or worker_i in seen:
                continue
            seen.add(worker_i)
            worker_ids_clean.append(int(worker_i))
        if not worker_ids_clean:
            return {}
        out: dict[int, dict[str, float | int | str | None]] = {}
        chunk_size = 160
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(worker_ids_clean), chunk_size):
                    chunk = worker_ids_clean[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        "SELECT worker_id, last_seen, tested, cached_hits, total, eta_sec, status "
                        "FROM planner_heartbeat WHERE stage_label=? AND worker_id IN (" + placeholders + ")",
                        tuple([str(stage_key), *[int(worker_i) for worker_i in chunk]]),
                    ).fetchall()
                    for row in rows:
                        try:
                            worker_i = int(row[0])
                            last_seen = float(row[1] or 0.0)
                            tested_i = int(row[2] or 0)
                            cached_i = int(row[3] or 0)
                            total_i = int(row[4] or 0)
                            eta_raw = row[5]
                            eta_f = None if eta_raw is None else float(max(0.0, float(eta_raw)))
                            status_s = str(row[6] or "").strip().lower()
                        except Exception:
                            continue
                        out[int(worker_i)] = {
                            "last_seen": float(last_seen),
                            "tested": int(tested_i),
                            "cached_hits": int(cached_i),
                            "total": int(total_i),
                            "eta_sec": (float(eta_f) if eta_f is not None else None),
                            "status": str(status_s),
                        }
            planner_heartbeat_reads += len(out)
        except Exception:
            return {}
        return out

    def _stage_frontier_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        nonlocal stage_frontier_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        run_min = int(run_min_trades)
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for axis_dim_fp, window_sig in cells:
            key = (str(axis_dim_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], dict[str, object]] = {}
        chunk_size = 140
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join(
                        "(axis_dimension_fingerprint=? AND window_signature=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key), int(run_min)]
                    for axis_dim_fp, window_sig in chunk:
                        params.extend((str(axis_dim_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT axis_dimension_fingerprint, window_signature, eval_count, keep_count, "
                        "best_pnl_over_dd, best_pnl, best_win_rate, best_trades "
                        "FROM stage_frontier WHERE stage_label=? AND run_min_trades=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        axis_dim_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        if not axis_dim_fp or not window_sig:
                            continue
                        out[(axis_dim_fp, window_sig)] = {
                            "eval_count": int(row[2] or 0),
                            "keep_count": int(row[3] or 0),
                            "best_pnl_over_dd": (None if row[4] is None else float(row[4])),
                            "best_pnl": (None if row[5] is None else float(row[5])),
                            "best_win_rate": (None if row[6] is None else float(row[6])),
                            "best_trades": (None if row[7] is None else int(row[7])),
                        }
            stage_frontier_reads += len(out)
        except Exception:
            return {}
        return out

    def _stage_frontier_upsert_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, dict | None]],
    ) -> None:
        nonlocal stage_frontier_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        run_min = int(run_min_trades)
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for axis_dim_fp, window_sig, row in rows:
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            if not axis_key or not window_key:
                continue
            key = (axis_key, window_key)
            rec = merged.get(key)
            if rec is None:
                rec = {
                    "eval_count": 0,
                    "keep_count": 0,
                    "best_pnl_over_dd": None,
                    "best_pnl": None,
                    "best_win_rate": None,
                    "best_trades": None,
                }
                merged[key] = rec
            rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
            if not isinstance(row, dict):
                continue
            rec["keep_count"] = int(rec.get("keep_count") or 0) + 1
            for metric_key, rec_key, caster in (
                ("pnl_over_dd", "best_pnl_over_dd", float),
                ("pnl", "best_pnl", float),
                ("win_rate", "best_win_rate", float),
                ("trades", "best_trades", int),
            ):
                raw_val = row.get(metric_key)
                if raw_val is None:
                    continue
                try:
                    val = caster(raw_val)
                except (TypeError, ValueError):
                    continue
                prev = rec.get(rec_key)
                if prev is None or float(val) > float(prev):
                    rec[rec_key] = val
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(axis_dim_fp),
                str(window_sig),
                int(run_min),
                int(rec.get("eval_count") or 0),
                int(rec.get("keep_count") or 0),
                rec.get("best_pnl_over_dd"),
                rec.get("best_pnl"),
                rec.get("best_win_rate"),
                rec.get("best_trades"),
                now_ts,
            )
            for (axis_dim_fp, window_sig), rec in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO stage_frontier("
                    "stage_label, axis_dimension_fingerprint, window_signature, run_min_trades, "
                    "eval_count, keep_count, best_pnl_over_dd, best_pnl, best_win_rate, best_trades, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, axis_dimension_fingerprint, window_signature, run_min_trades) DO UPDATE SET "
                    "eval_count=stage_frontier.eval_count + excluded.eval_count, "
                    "keep_count=stage_frontier.keep_count + excluded.keep_count, "
                    "best_pnl_over_dd=CASE "
                    "WHEN stage_frontier.best_pnl_over_dd IS NULL THEN excluded.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd IS NULL THEN stage_frontier.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd > stage_frontier.best_pnl_over_dd THEN excluded.best_pnl_over_dd "
                    "ELSE stage_frontier.best_pnl_over_dd END, "
                    "best_pnl=CASE "
                    "WHEN stage_frontier.best_pnl IS NULL THEN excluded.best_pnl "
                    "WHEN excluded.best_pnl IS NULL THEN stage_frontier.best_pnl "
                    "WHEN excluded.best_pnl > stage_frontier.best_pnl THEN excluded.best_pnl "
                    "ELSE stage_frontier.best_pnl END, "
                    "best_win_rate=CASE "
                    "WHEN stage_frontier.best_win_rate IS NULL THEN excluded.best_win_rate "
                    "WHEN excluded.best_win_rate IS NULL THEN stage_frontier.best_win_rate "
                    "WHEN excluded.best_win_rate > stage_frontier.best_win_rate THEN excluded.best_win_rate "
                    "ELSE stage_frontier.best_win_rate END, "
                    "best_trades=CASE "
                    "WHEN stage_frontier.best_trades IS NULL THEN excluded.best_trades "
                    "WHEN excluded.best_trades IS NULL THEN stage_frontier.best_trades "
                    "WHEN excluded.best_trades > stage_frontier.best_trades THEN excluded.best_trades "
                    "ELSE stage_frontier.best_trades END, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            stage_frontier_writes += len(payload)
        except Exception:
            return

    def _stage_frontier_is_dominated(frontier_row: dict[str, object] | None) -> bool:
        if not isinstance(frontier_row, dict):
            return False
        cfg = _cache_config("stage_frontier")
        min_eval_count = max(1, int(_registry_float(cfg.get("min_eval_count"), 3.0)))
        max_keep_count = max(0, int(_registry_float(cfg.get("max_keep_count"), 0.0)))
        max_best_pnl = float(_registry_float(cfg.get("max_best_pnl"), 0.0))
        max_best_pnl_dd = float(_registry_float(cfg.get("max_best_pnl_over_dd"), 0.0))
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        if eval_count < int(min_eval_count):
            return False
        if keep_count > int(max_keep_count):
            return False
        best_pnl = frontier_row.get("best_pnl")
        best_pnl_dd = frontier_row.get("best_pnl_over_dd")
        best_pnl_f = float(best_pnl) if best_pnl is not None else float("-inf")
        best_pnl_dd_f = float(best_pnl_dd) if best_pnl_dd is not None else float("-inf")
        return bool(best_pnl_f <= float(max_best_pnl) and best_pnl_dd_f <= float(max_best_pnl_dd))

    def _winner_projection_get_many(
        *,
        stage_label: str,
        cells: list[tuple[str, str]],
    ) -> dict[tuple[str, str], dict[str, object]]:
        nonlocal winner_projection_reads
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key:
            return {}
        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for axis_dim_fp, window_sig in cells:
            key = (str(axis_dim_fp).strip(), str(window_sig).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if not deduped:
            return {}
        out: dict[tuple[str, str], dict[str, object]] = {}
        chunk_size = 140
        try:
            with run_cfg_persistent_lock:
                for start_i in range(0, len(deduped), chunk_size):
                    chunk = deduped[start_i : start_i + chunk_size]
                    if not chunk:
                        continue
                    where = " OR ".join(
                        "(axis_dimension_fingerprint=? AND window_signature=?)"
                        for _ in chunk
                    )
                    params: list[object] = [str(stage_key)]
                    for axis_dim_fp, window_sig in chunk:
                        params.extend((str(axis_dim_fp), str(window_sig)))
                    rows_db = conn.execute(
                        "SELECT axis_dimension_fingerprint, window_signature, eval_count, keep_count, "
                        "best_pnl_over_dd, best_pnl, best_win_rate, best_trades "
                        "FROM stage_winner_projection WHERE stage_label=? AND (" + where + ")",
                        tuple(params),
                    ).fetchall()
                    for row in rows_db:
                        axis_dim_fp = str(row[0] or "").strip()
                        window_sig = str(row[1] or "").strip()
                        if not axis_dim_fp or not window_sig:
                            continue
                        out[(axis_dim_fp, window_sig)] = {
                            "eval_count": int(row[2] or 0),
                            "keep_count": int(row[3] or 0),
                            "best_pnl_over_dd": (None if row[4] is None else float(row[4])),
                            "best_pnl": (None if row[5] is None else float(row[5])),
                            "best_win_rate": (None if row[6] is None else float(row[6])),
                            "best_trades": (None if row[7] is None else int(row[7])),
                        }
            winner_projection_reads += len(out)
        except Exception:
            return {}
        return out

    def _winner_projection_upsert_many(
        *,
        stage_label: str,
        rows: list[tuple[str, str, dict | None]],
    ) -> None:
        nonlocal winner_projection_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        stage_key = _stage_cache_scope(stage_label)
        if not stage_key or not rows:
            return
        merged: dict[tuple[str, str], dict[str, object]] = {}
        for axis_dim_fp, window_sig, row in rows:
            axis_key = str(axis_dim_fp).strip()
            window_key = str(window_sig).strip()
            if not axis_key or not window_key:
                continue
            key = (axis_key, window_key)
            rec = merged.get(key)
            if rec is None:
                rec = {
                    "eval_count": 0,
                    "keep_count": 0,
                    "best_pnl_over_dd": None,
                    "best_pnl": None,
                    "best_win_rate": None,
                    "best_trades": None,
                }
                merged[key] = rec
            rec["eval_count"] = int(rec.get("eval_count") or 0) + 1
            if not isinstance(row, dict):
                continue
            rec["keep_count"] = int(rec.get("keep_count") or 0) + 1
            for metric_key, rec_key, caster in (
                ("pnl_over_dd", "best_pnl_over_dd", float),
                ("pnl", "best_pnl", float),
                ("win_rate", "best_win_rate", float),
                ("trades", "best_trades", int),
            ):
                raw_val = row.get(metric_key)
                if raw_val is None:
                    continue
                try:
                    val = caster(raw_val)
                except (TypeError, ValueError):
                    continue
                prev = rec.get(rec_key)
                if prev is None or float(val) > float(prev):
                    rec[rec_key] = val
        if not merged:
            return
        now_ts = float(pytime.time())
        payload = [
            (
                str(stage_key),
                str(axis_dim_fp),
                str(window_sig),
                int(rec.get("eval_count") or 0),
                int(rec.get("keep_count") or 0),
                rec.get("best_pnl_over_dd"),
                rec.get("best_pnl"),
                rec.get("best_win_rate"),
                rec.get("best_trades"),
                now_ts,
            )
            for (axis_dim_fp, window_sig), rec in merged.items()
        ]
        try:
            with run_cfg_persistent_lock:
                conn.executemany(
                    "INSERT INTO stage_winner_projection("
                    "stage_label, axis_dimension_fingerprint, window_signature, "
                    "eval_count, keep_count, best_pnl_over_dd, best_pnl, best_win_rate, best_trades, updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(stage_label, axis_dimension_fingerprint, window_signature) DO UPDATE SET "
                    "eval_count=stage_winner_projection.eval_count + excluded.eval_count, "
                    "keep_count=stage_winner_projection.keep_count + excluded.keep_count, "
                    "best_pnl_over_dd=CASE "
                    "WHEN stage_winner_projection.best_pnl_over_dd IS NULL THEN excluded.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd IS NULL THEN stage_winner_projection.best_pnl_over_dd "
                    "WHEN excluded.best_pnl_over_dd > stage_winner_projection.best_pnl_over_dd THEN excluded.best_pnl_over_dd "
                    "ELSE stage_winner_projection.best_pnl_over_dd END, "
                    "best_pnl=CASE "
                    "WHEN stage_winner_projection.best_pnl IS NULL THEN excluded.best_pnl "
                    "WHEN excluded.best_pnl IS NULL THEN stage_winner_projection.best_pnl "
                    "WHEN excluded.best_pnl > stage_winner_projection.best_pnl THEN excluded.best_pnl "
                    "ELSE stage_winner_projection.best_pnl END, "
                    "best_win_rate=CASE "
                    "WHEN stage_winner_projection.best_win_rate IS NULL THEN excluded.best_win_rate "
                    "WHEN excluded.best_win_rate IS NULL THEN stage_winner_projection.best_win_rate "
                    "WHEN excluded.best_win_rate > stage_winner_projection.best_win_rate THEN excluded.best_win_rate "
                    "ELSE stage_winner_projection.best_win_rate END, "
                    "best_trades=CASE "
                    "WHEN stage_winner_projection.best_trades IS NULL THEN excluded.best_trades "
                    "WHEN excluded.best_trades IS NULL THEN stage_winner_projection.best_trades "
                    "WHEN excluded.best_trades > stage_winner_projection.best_trades THEN excluded.best_trades "
                    "ELSE stage_winner_projection.best_trades END, "
                    "updated_at=excluded.updated_at",
                    payload,
                )
            winner_projection_writes += len(payload)
        except Exception:
            return

    def _winner_projection_is_dominated(frontier_row: dict[str, object] | None) -> bool:
        if not isinstance(frontier_row, dict):
            return False
        cfg = _cache_config("winner_projection")
        min_eval_count = max(1, int(_registry_float(cfg.get("min_eval_count"), 8.0)))
        max_keep_count = max(0, int(_registry_float(cfg.get("max_keep_count"), 0.0)))
        max_best_pnl = float(_registry_float(cfg.get("max_best_pnl"), 0.0))
        max_best_pnl_dd = float(_registry_float(cfg.get("max_best_pnl_over_dd"), 0.0))
        eval_count = int(frontier_row.get("eval_count") or 0)
        keep_count = int(frontier_row.get("keep_count") or 0)
        if eval_count < int(min_eval_count):
            return False
        if keep_count > int(max_keep_count):
            return False
        best_pnl = frontier_row.get("best_pnl")
        best_pnl_dd = frontier_row.get("best_pnl_over_dd")
        best_pnl_f = float(best_pnl) if best_pnl is not None else float("-inf")
        best_pnl_dd_f = float(best_pnl_dd) if best_pnl_dd is not None else float("-inf")
        return bool(best_pnl_f <= float(max_best_pnl) and best_pnl_dd_f <= float(max_best_pnl_dd))

    def _run_cfg_dimension_index_set(*, fingerprint: str, payload_json: str, est_cost: float) -> None:
        nonlocal run_cfg_dim_index_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO axis_dimension_fingerprint_index(fingerprint, payload_json, est_cost, updated_at) "
                    "VALUES(?,?,?,?)",
                    (str(fingerprint), str(payload_json), float(est_cost), float(pytime.time())),
                )
            if float(est_cost) > 0.0:
                run_cfg_dim_index_loaded[str(fingerprint)] = float(est_cost)
            run_cfg_dim_index_writes += 1
        except Exception:
            return

    def _run_cfg_dimension_index_load(*, limit: int = 50000) -> dict[str, float]:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return {}
        try:
            with run_cfg_persistent_lock:
                rows = conn.execute(
                    "SELECT fingerprint, est_cost FROM axis_dimension_fingerprint_index "
                    "WHERE est_cost > 0 ORDER BY updated_at DESC LIMIT ?",
                    (int(max(1, int(limit))),),
                ).fetchall()
        except Exception:
            return {}
        out: dict[str, float] = {}
        for row in rows:
            try:
                fp = str(row[0] or "")
                est = float(row[1] or 0.0)
            except Exception:
                continue
            if not fp or est <= 0.0 or fp in out:
                continue
            out[fp] = est
        return out

    def _worker_plan_cache_key(*, stage_label: str, workers: int, plan_all) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(_RUN_CFG_CACHE_ENGINE_VERSION).encode("utf-8"))
        hasher.update(str(stage_label).strip().lower().encode("utf-8"))
        hasher.update(str(int(workers)).encode("utf-8"))
        hasher.update(str(len(plan_all)).encode("utf-8"))
        for item in plan_all:
            cfg = item[0] if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], ConfigBundle) else None
            if cfg is None:
                hasher.update(str(item).encode("utf-8"))
                continue
            hasher.update(hashlib.sha1(_milestone_key(cfg).encode("utf-8")).digest())
            hasher.update(hashlib.sha1(_axis_dimension_fingerprint(cfg).encode("utf-8")).digest())
        return hasher.hexdigest()

    def _worker_plan_cache_get(*, cache_key: str) -> list[list[int]] | None:
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return None
        try:
            with run_cfg_persistent_lock:
                row = conn.execute(
                    "SELECT payload_json FROM worker_plan_cache WHERE cache_key=?",
                    (str(cache_key),),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
        except Exception:
            return None
        if not isinstance(payload, list):
            return None
        out: list[list[int]] = []
        try:
            for bucket in payload:
                if not isinstance(bucket, list):
                    return None
                out.append([int(idx) for idx in bucket if int(idx) >= 0])
        except Exception:
            return None
        return out

    def _worker_plan_cache_set(*, cache_key: str, buckets: list[list[int]]) -> None:
        nonlocal worker_plan_cache_writes
        conn = _run_cfg_persistent_conn()
        if conn is None:
            return
        try:
            payload_json = json.dumps(buckets, sort_keys=False, default=str)
            with run_cfg_persistent_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO worker_plan_cache(cache_key, payload_json, updated_at) VALUES(?,?,?)",
                    (str(cache_key), str(payload_json), float(pytime.time())),
                )
            worker_plan_cache_writes += 1
        except Exception:
            return

    def _merge_filters(base_filters: FiltersConfig | None, overrides: dict[str, object]) -> FiltersConfig | None:
        """Merge base filters with overrides, where `None` deletes a key.

        Used to build joint permission sweeps without being constrained by the compact preset funnel.
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
            series = data.load_cached_bar_series(
                symbol=symbol,
                exchange=None,
                start=start_dt,
                end=end_dt,
                bar_size=str(bar_size),
                use_rth=use_rth,
                cache_dir=cache_dir,
            )
            return bars_list(series)
        series = data.load_or_fetch_bar_series(
            symbol=symbol,
            exchange=None,
            start=start_dt,
            end=end_dt,
            bar_size=str(bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
        return bars_list(series)

    def _bars_cached(bar_size: str) -> list:
        key = (
            str(symbol).upper(),
            start_dt.isoformat(),
            end_dt.isoformat(),
            str(bar_size),
            bool(use_rth),
            bool(offline),
        )
        cached = _SERIES_CACHE.get(namespace=_SWEEP_BARS_NAMESPACE, key=key)
        if isinstance(cached, list):
            return cached
        loaded = _bars(str(bar_size))
        _SERIES_CACHE.set(namespace=_SWEEP_BARS_NAMESPACE, key=key, value=loaded)
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
                    series = data.load_cached_bar_series(
                        symbol=symbol,
                        exchange=exchange,
                        start=tick_start_dt,
                        end=end_dt,
                        bar_size="1 day",
                        use_rth=tick_use_rth,
                        cache_dir=cache_dir,
                    )
                    return bars_list(series)
                series = data.load_or_fetch_bar_series(
                    symbol=symbol,
                    exchange=exchange,
                    start=tick_start_dt,
                    end=end_dt,
                    bar_size="1 day",
                    use_rth=tick_use_rth,
                    cache_dir=cache_dir,
                )
                return bars_list(series)
            except FileNotFoundError:
                return []

        def _from_cache(symbol: str, exchange: str) -> list | None:
            cache_key = (str(symbol), str(exchange), bool(offline))
            cached = _SERIES_CACHE.get(namespace=_SWEEP_TICK_NAMESPACE, key=cache_key)
            if not isinstance(cached, tuple) or len(cached) != 2:
                return None
            cached_start, cached_bars = cached
            if not isinstance(cached_start, datetime) or not isinstance(cached_bars, list):
                return None
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
        cache_key = (str(used_symbol), str(used_exchange), bool(offline))
        _SERIES_CACHE.set(namespace=_SWEEP_TICK_NAMESPACE, key=cache_key, value=(tick_start_dt, tick_bars))
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
        include_combo_baseline = not bool(getattr(args, "combo_full_cartesian_stage", None))
        spec = _AXIS_EXECUTION_SPEC_BY_NAME.get(str(axis))
        hint_static = int(spec.total_hint_static) if isinstance(spec, AxisExecutionSpec) and isinstance(spec.total_hint_static, int) else None
        hint_mode = str(spec.total_hint_mode or "").strip().lower() if isinstance(spec, AxisExecutionSpec) else ""
        hint_dims = tuple(spec.total_hint_dims or ()) if isinstance(spec, AxisExecutionSpec) else ()
        if hint_static is None:
            raw_static = _AXIS_TOTAL_HINT_STATIC_BY_NAME.get(str(axis))
            if raw_static is not None:
                try:
                    hint_static = int(raw_static)
                except (TypeError, ValueError):
                    hint_static = None
        if not hint_mode:
            hint_mode = str(_AXIS_TOTAL_HINT_MODE_BY_NAME.get(str(axis), "")).strip().lower()
        if not hint_dims:
            hint_dims = tuple(_AXIS_TOTAL_HINT_DIMS_BY_NAME.get(str(axis), ()))
        if str(axis) in set(_combo_full_preset_axes(include_tiers=True, include_aliases=True)):
            preset_hint = _combo_full_preset_hint_spec(str(axis))
            raw_static = preset_hint.get("static")
            if hint_static is None and raw_static is not None:
                try:
                    hint_static = int(raw_static)
                except (TypeError, ValueError):
                    hint_static = None
            if not hint_mode:
                hint_mode = str(preset_hint.get("mode") or "").strip().lower()
            if not hint_dims:
                raw_dims = preset_hint.get("dims")
                if isinstance(raw_dims, (tuple, list)):
                    hint_dims = tuple(str(dim).strip() for dim in tuple(raw_dims) if str(dim).strip())
        if isinstance(hint_static, int) and hint_static > 0:
            return int(hint_static)

        def _combo_dim_labels(dim_key: str) -> tuple[str, ...]:
            combo_dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
            if not isinstance(combo_dims, dict):
                return ()
            raw_variants = combo_dims.get(f"{dim_key}_variants")
            labels_from_variants: list[str] = []
            if isinstance(raw_variants, (list, tuple)):
                for row in raw_variants:
                    if not (isinstance(row, (list, tuple)) and len(row) >= 1):
                        continue
                    label = str(row[0] or "").strip()
                    if label:
                        labels_from_variants.append(label)
            return tuple(labels_from_variants)

        if hint_mode == "atr_profile":
            profile = _ATR_EXIT_PROFILE_REGISTRY.get(str(axis)) or {}
            total = _cardinality(
                len(tuple(profile.get("atr_periods") or ())),
                len(tuple(profile.get("pt_mults") or ())),
                len(tuple(profile.get("sl_mults") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "spread_profile":
            profile = _SPREAD_PROFILE_REGISTRY.get(str(axis)) or {}
            total = len(tuple(profile.get("values") or ()))
            if total > 0:
                return int(total)
        if hint_mode == "regime_profile":
            profile = _REGIME_ST_PROFILE
            total = _cardinality(
                len(tuple(profile.get("bars") or ())),
                len(tuple(profile.get("atr_periods") or ())),
                len(tuple(profile.get("multipliers") or ())),
                len(tuple(profile.get("sources") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "regime2_profile":
            profile = _REGIME2_ST_PROFILE
            total = _cardinality(
                len(tuple(profile.get("atr_periods") or ())),
                len(tuple(profile.get("multipliers") or ())),
                len(tuple(profile.get("sources") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "shock_profile":
            profile = _SHOCK_SWEEP_PROFILE
            preset_count = int(
                len(tuple(profile.get("ratio_rows") or ()))
                + len(tuple(profile.get("daily_atr_rows") or ()))
                + len(tuple(profile.get("drawdown_rows") or ()))
            )
            total = _cardinality(
                int(preset_count),
                len(tuple(profile.get("modes") or ())),
                len(tuple(profile.get("dir_variants") or ())),
                len(tuple(profile.get("sl_mults") or ())),
                len(tuple(profile.get("pt_mults") or ())),
                len(tuple(profile.get("short_risk_factors") or ())),
            )
            if total > 0:
                return int(total)
        if hint_mode == "combo_subset" and hint_dims:
            sizes: list[int] = []
            risk_tier_hint = str(axis) == "risk_overlays" or _combo_full_preset_tier(str(axis)) == "risk"
            for dim_key in hint_dims:
                labels = list(_combo_dim_labels(str(dim_key)))
                if risk_tier_hint and str(dim_key) == "risk" and bool(getattr(args, "risk_overlays_skip_pop", False)):
                    labels = [lbl for lbl in labels if "riskpop" not in str(lbl).lower()]
                sizes.append(max(1, len(labels)))
            total = _cardinality(*sizes)
            if total > 0:
                return int(total) + (1 if include_combo_baseline else 0)
        if hint_mode == "gate_matrix":
            combo_dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
            gate_dims = _AXIS_DIMENSION_REGISTRY.get("gate_matrix", {})
            if isinstance(combo_dims, dict):
                perm_total = len(tuple(gate_dims.get("perm_variants") or ())) if isinstance(gate_dims, dict) else 0
                if perm_total <= 0:
                    perm_total = len(_combo_dim_labels("perm"))
                tod_total = len(tuple(gate_dims.get("tod_variants") or ())) if isinstance(gate_dims, dict) else 0
                if tod_total <= 0:
                    tod_total = len(_combo_dim_labels("tod"))
                short_total = len(tuple(gate_dims.get("short_mults") or ())) if isinstance(gate_dims, dict) else 0
                if short_total <= 0:
                    short_total = len(tuple(combo_dims.get("short_mults") or ()))
                total = _cardinality(
                    max(1, int(perm_total)),
                    max(1, int(tod_total)),
                    max(1, len(_combo_dim_labels("regime2"))),
                    max(1, len(_combo_dim_labels("tick"))),
                    max(1, len(_combo_dim_labels("shock"))),
                    max(1, len(_combo_dim_labels("risk"))),
                    max(1, int(short_total)),
                )
                if total > 0:
                    return int(total) + (1 if include_combo_baseline else 0)
        if hint_mode == "combo_full":
            combo_preset = _combo_full_preset_key(str(getattr(args, "combo_full_preset", "") or ""))
            if combo_preset in set(_combo_full_preset_axes()):
                preset_hint_axis = _combo_full_preset_hint_axis(combo_preset)
                preset_total = None
                if preset_hint_axis and preset_hint_axis != "combo_full":
                    preset_total = _axis_total_hint(str(preset_hint_axis))
                if isinstance(preset_total, int) and preset_total > 0:
                    return int(preset_total)
            dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
            if isinstance(dims, dict):
                size_by_dim = {
                    str(dim_name): _combo_full_dim_size_from_registry(dims=dims, dim_key=str(dim_name))
                    for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
                }
                total = _cardinality(
                    *[
                        max(1, int(size_by_dim.get(str(dim_name), 0) or 0))
                        if str(dim_name) == "slope"
                        else int(size_by_dim.get(str(dim_name), 0) or 0)
                        for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
                    ]
                )
                if total > 0:
                    return int(total) + (1 if include_combo_baseline else 0)
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

    def _run_cfg_cache_coords(
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
        regime_bars: list | None = None,
        regime2_bars: list | None = None,
        update_dim_index: bool = True,
    ) -> tuple[
        tuple[tuple[int, object | None, object | None], tuple[int, object | None, object | None], tuple[int, object | None, object | None]],
        tuple[str, str, str],
        str,
        str,
    ]:
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

        axis_dim_fp = run_cfg_axis_fp_cache.get(cfg_key)
        if axis_dim_fp is None:
            axis_dim_fp = _axis_dimension_fingerprint(cfg)
            run_cfg_axis_fp_cache[cfg_key] = str(axis_dim_fp)
        if bool(update_dim_index) and axis_dim_fp not in run_cfg_dim_index_seen:
            run_cfg_dim_index_seen.add(str(axis_dim_fp))
            est_cost = 1.0
            try:
                est_cost = float(_cfg_eval_cost_hint(cfg))
            except Exception:
                est_cost = 1.0
            _run_cfg_dimension_index_set(
                fingerprint=str(axis_dim_fp),
                payload_json=str(axis_dim_fp),
                est_cost=float(est_cost),
            )

        window_sig = run_cfg_window_sig_cache.get(ctx_sig)
        if window_sig is None:
            window_sig = _window_signature(
                bars_sig=bars_sig,
                regime_sig=regime_sig,
                regime2_sig=regime2_sig,
            )
            run_cfg_window_sig_cache[ctx_sig] = str(window_sig)

        cache_key = (
            str(cfg_key),
            str(axis_dim_fp),
            str(window_sig),
        )
        persistent_key = _run_cfg_persistent_key(
            strategy_fingerprint=str(cfg_key),
            axis_dimension_fingerprint=str(axis_dim_fp),
            window_signature=str(window_sig),
        )
        return ctx_sig, cache_key, str(axis_dim_fp), str(persistent_key)

    def _stage_partition_plan_by_cache(
        *,
        stage_label: str,
        plan_all,
        bars: list | None,
    ) -> tuple[
        list[tuple[ConfigBundle, str, dict | None]],
        list[tuple[tuple[str, str, str], ConfigBundle, dict | None, str, dict | None]],
        dict[int, tuple[str, str, str]],
        int,
    ]:
        nonlocal run_calls_total, run_cfg_cache_hits, run_cfg_fingerprint_hits, run_cfg_persistent_hits
        nonlocal stage_frontier_hits, cartesian_manifest_hits, winner_projection_hits

        pending_plan: list[tuple[ConfigBundle, str, dict | None]] = []
        pending_cell_map: dict[int, tuple[str, str, str]] = {}
        cached_hits: list[tuple[tuple[str, str, str], ConfigBundle, dict | None, str, dict | None]] = []
        prefetched_tested = 0
        status_updates: list[tuple[str, str, str, str]] = []
        manifest_updates: list[tuple[str, str, str, str]] = []
        planner_started = float(pytime.perf_counter())
        planner_last = float(planner_started)
        planner_heartbeat_sec = 15.0
        try:
            planner_total = int(len(plan_all))
        except Exception:
            planner_total = None

        def _planner_progress(*, phase: str, processed: int, force: bool = False) -> None:
            nonlocal planner_last
            now = float(pytime.perf_counter())
            if not bool(force) and (now - planner_last) < float(planner_heartbeat_sec):
                return
            total_s = str(planner_total) if isinstance(planner_total, int) and planner_total > 0 else "?"
            print(
                f"{stage_label} planner[{phase}] processed={int(processed)}/{total_s} "
                f"pending={len(pending_plan)} unresolved={len(unresolved)} "
                f"cached={len(cached_hits)} elapsed={now - planner_started:0.1f}s",
                flush=True,
            )
            planner_last = float(now)

        unresolved: list[
            tuple[
                ConfigBundle,
                str,
                dict | None,
                tuple[
                    tuple[int, object | None, object | None],
                    tuple[int, object | None, object | None],
                    tuple[int, object | None, object | None],
                ],
                tuple[str, str, str],
                str,
                tuple[str, str, str],
            ]
        ] = []
        persistent_keys: list[str] = []
        cell_keys_for_status: list[tuple[str, str, str]] = []
        stage_cell_hasher = hashlib.sha1()
        stage_cell_window_set: set[str] = set()
        stage_cell_window_hasher = hashlib.sha1()

        for idx, item in enumerate(plan_all, start=1):
            if not (isinstance(item, tuple) and len(item) >= 2):
                _planner_progress(phase="scan", processed=int(idx))
                continue
            cfg = item[0] if isinstance(item[0], ConfigBundle) else None
            if cfg is None:
                _planner_progress(phase="scan", processed=int(idx))
                continue
            note_s = str(item[1] or "")
            meta_item = item[2] if len(item) >= 3 and isinstance(item[2], dict) else None
            ctx_sig, cache_key, axis_dim_fp, persistent_key = _run_cfg_cache_coords(
                cfg=cfg,
                bars=bars,
                update_dim_index=True,
            )
            cfg_key = str(cache_key[0])
            cell_key = (str(cache_key[0]), str(axis_dim_fp), str(cache_key[2]))
            cell_keys_for_status.append(cell_key)
            stage_cell_hasher.update(str(cell_key[0]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1f")
            stage_cell_hasher.update(str(cell_key[1]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1f")
            stage_cell_hasher.update(str(cell_key[2]).encode("utf-8"))
            stage_cell_hasher.update(b"\x1e")
            window_sig = str(cell_key[2]).strip()
            if window_sig:
                stage_cell_window_set.add(str(window_sig))
                stage_cell_window_hasher.update(str(window_sig).encode("utf-8"))
                stage_cell_window_hasher.update(b"\x1f")

            fp_cached = run_cfg_fingerprint_cache.get(cfg_key)
            if fp_cached is not None and fp_cached[0] == ctx_sig:
                prefetched_tested += 1
                run_calls_total += 1
                run_cfg_cache_hits += 1
                run_cfg_fingerprint_hits += 1
                fp_row = fp_cached[1]
                row = dict(fp_row) if isinstance(fp_row, dict) else None
                _axis_progress_record(kept=bool(row))
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "cached_hit"))
                if isinstance(row, dict):
                    row["note"] = note_s
                cached_hits.append((cell_key, cfg, row if isinstance(row, dict) else None, note_s, meta_item))
                _planner_progress(phase="scan", processed=int(idx))
                continue

            cached = run_cfg_cache.get(cache_key, _RUN_CFG_CACHE_MISS)
            if cached is not _RUN_CFG_CACHE_MISS:
                prefetched_tested += 1
                run_calls_total += 1
                run_cfg_cache_hits += 1
                run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, cached if isinstance(cached, dict) else None)
                row = dict(cached) if isinstance(cached, dict) else None
                _axis_progress_record(kept=bool(row))
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "cached_hit"))
                if isinstance(row, dict):
                    row["note"] = note_s
                cached_hits.append((cell_key, cfg, row if isinstance(row, dict) else None, note_s, meta_item))
                _planner_progress(phase="scan", processed=int(idx))
                continue

            unresolved.append((cfg, note_s, meta_item, ctx_sig, cache_key, str(persistent_key), cell_key))
            persistent_keys.append(str(persistent_key))
            _planner_progress(phase="scan", processed=int(idx))

        _planner_progress(
            phase="scan",
            processed=(int(planner_total) if isinstance(planner_total, int) and planner_total > 0 else len(unresolved)),
            force=True,
        )

        stage_cell_total = int(len(cell_keys_for_status))
        stage_cell_plan_signature = str(stage_cell_hasher.hexdigest()) if int(stage_cell_total) > 0 else ""
        if int(stage_cell_total) <= 0:
            stage_cell_window_signature = ""
        elif len(stage_cell_window_set) == 1:
            stage_cell_window_signature = next(iter(stage_cell_window_set))
        else:
            stage_cell_window_signature = "multi:" + str(stage_cell_window_hasher.hexdigest())
        stage_cell_summary_all_resolved = False
        if int(stage_cell_total) > 0 and stage_cell_plan_signature and stage_cell_window_signature:
            stage_summary = _stage_unresolved_summary_get(
                manifest_name="stage_cell",
                stage_label=str(stage_label),
                plan_signature=str(stage_cell_plan_signature),
                window_signature=str(stage_cell_window_signature),
                total=int(stage_cell_total),
            )
            if isinstance(stage_summary, tuple) and len(stage_summary) == 2:
                try:
                    stage_cell_summary_all_resolved = int(stage_summary[0]) <= 0
                except (TypeError, ValueError):
                    stage_cell_summary_all_resolved = False

        persisted_by_key = _run_cfg_persistent_get_many(cache_keys=persistent_keys) if persistent_keys else {}
        frontier_by_dim_window: dict[tuple[str, str], dict[str, object]] = {}
        upper_bound_by_dim_window: dict[tuple[str, str], dict[str, object]] = {}
        winner_projection_by_dim_window: dict[tuple[str, str], dict[str, object]] = {}
        manifest_by_dim_window: dict[tuple[str, str], tuple[str, str]] = {}
        if not bool(stage_cell_summary_all_resolved):
            frontier_by_dim_window = _stage_frontier_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
            upper_bound_by_dim_window = _dimension_upper_bound_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
            winner_projection_by_dim_window = _winner_projection_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
            manifest_by_dim_window = _cartesian_cell_manifest_get_many(
                stage_label=str(stage_label),
                cells=[(axis_dim_fp, window_sig) for _, axis_dim_fp, window_sig in cell_keys_for_status],
            )
        prior_status = (
            {(strategy_fp, axis_dim_fp, window_sig): "cached_hit" for strategy_fp, axis_dim_fp, window_sig in cell_keys_for_status}
            if bool(stage_cell_summary_all_resolved)
            else _stage_cell_status_get_many(stage_label=str(stage_label), cells=cell_keys_for_status)
        )
        rank_dominance_stamp_rows_by_window: dict[str, list[tuple[str, int, int]]] = {}

        unresolved_total = len(unresolved)
        for idx_u, (cfg, note_s, meta_item, ctx_sig, cache_key, persistent_key, cell_key) in enumerate(unresolved, start=1):
            persisted = persisted_by_key.get(str(persistent_key), _RUN_CFG_CACHE_MISS)
            cfg_key = str(cache_key[0])
            if persisted is not _RUN_CFG_CACHE_MISS:
                prefetched_tested += 1
                run_calls_total += 1
                run_cfg_cache_hits += 1
                run_cfg_persistent_hits += 1
                run_cfg_cache[cache_key] = persisted if isinstance(persisted, dict) else None
                run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, persisted if isinstance(persisted, dict) else None)
                row = dict(persisted) if isinstance(persisted, dict) else None
                _axis_progress_record(kept=bool(row))
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "cached_hit"))
                if isinstance(row, dict):
                    row["note"] = note_s
                cached_hits.append((cell_key, cfg, row if isinstance(row, dict) else None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            prev_cell_status = str(prior_status.get(cell_key) or "").strip().lower()
            if prev_cell_status in ("cached_hit", "evaluated"):
                prefetched_tested += 1
                run_calls_total += 1
                _axis_progress_record(kept=False)
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "cached_hit"))
                cached_hits.append((cell_key, cfg, None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            frontier_key = (str(cell_key[1]), str(cell_key[2]))
            upper_bound_row = upper_bound_by_dim_window.get(frontier_key)
            upper_bound_sig = _upper_bound_dominance_signature(upper_bound_row)
            if upper_bound_sig:
                prefetched_tested += 1
                run_calls_total += 1
                _axis_progress_record(kept=False)
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "dominated"))
                if isinstance(meta_item, dict):
                    try:
                        rank_i = int(meta_item.get("_mr_rank"))
                    except (TypeError, ValueError):
                        rank_i = None
                    if rank_i is not None and int(rank_i) >= 0:
                        window_key = str(cell_key[2]).strip()
                        if window_key:
                            rank_dominance_stamp_rows_by_window.setdefault(str(window_key), []).append(
                                (str(upper_bound_sig), int(rank_i), int(rank_i))
                            )
                cached_hits.append((cell_key, cfg, None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            frontier_row = frontier_by_dim_window.get(frontier_key)
            if _stage_frontier_is_dominated(frontier_row):
                prefetched_tested += 1
                run_calls_total += 1
                stage_frontier_hits += 1
                _axis_progress_record(kept=False)
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "dominated"))
                cached_hits.append((cell_key, cfg, None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            winner_projection_row = winner_projection_by_dim_window.get(frontier_key)
            if _winner_projection_is_dominated(winner_projection_row):
                prefetched_tested += 1
                run_calls_total += 1
                winner_projection_hits += 1
                _axis_progress_record(kept=False)
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "dominated"))
                cached_hits.append((cell_key, cfg, None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            manifest_state = manifest_by_dim_window.get(frontier_key)
            manifest_status = str(manifest_state[0]).strip().lower() if isinstance(manifest_state, tuple) else ""
            manifest_strategy_fp = str(manifest_state[1]).strip() if isinstance(manifest_state, tuple) and len(manifest_state) >= 2 else ""
            if manifest_status in ("cached_hit", "evaluated", "dominated"):
                prefetched_tested += 1
                run_calls_total += 1
                cartesian_manifest_hits += 1
                resolved_row: dict | None = None
                if manifest_status in ("cached_hit", "evaluated"):
                    probe_strategy_fp = manifest_strategy_fp or str(cell_key[0])
                    probe_key = _run_cfg_persistent_key(
                        strategy_fingerprint=str(probe_strategy_fp),
                        axis_dimension_fingerprint=str(cell_key[1]),
                        window_signature=str(cell_key[2]),
                    )
                    probe_cached = _run_cfg_persistent_get(cache_key=str(probe_key))
                    if probe_cached is not _RUN_CFG_CACHE_MISS:
                        run_cfg_cache_hits += 1
                        run_cfg_persistent_hits += 1
                        run_cfg_cache[cache_key] = probe_cached if isinstance(probe_cached, dict) else None
                        run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, probe_cached if isinstance(probe_cached, dict) else None)
                        resolved_row = dict(probe_cached) if isinstance(probe_cached, dict) else None

                _axis_progress_record(kept=bool(resolved_row))
                status_updates.append((cell_key[0], cell_key[1], cell_key[2], "cached_hit"))
                manifest_updates.append(
                    (
                        cell_key[1],
                        cell_key[2],
                        manifest_strategy_fp or cell_key[0],
                        "dominated" if manifest_status == "dominated" else "cached_hit",
                    )
                )
                if isinstance(resolved_row, dict):
                    resolved_row["note"] = note_s
                cached_hits.append((cell_key, cfg, resolved_row if isinstance(resolved_row, dict) else None, note_s, meta_item))
                _planner_progress(phase="resolve", processed=int(idx_u))
                continue

            pending_idx = len(pending_plan)
            pending_plan.append((cfg, note_s, meta_item))
            pending_cell_map[pending_idx] = cell_key
            status_updates.append((cell_key[0], cell_key[1], cell_key[2], "pending"))
            manifest_updates.append((cell_key[1], cell_key[2], cell_key[0], "pending"))
            _planner_progress(phase="resolve", processed=int(idx_u))

        if unresolved_total > 0:
            _planner_progress(phase="resolve", processed=int(unresolved_total), force=True)

        status_updates_filtered: list[tuple[str, str, str, str]] = []
        for strategy_fp, axis_dim_fp, window_sig, status in status_updates:
            prev = prior_status.get((strategy_fp, axis_dim_fp, window_sig))
            if str(prev or "") == str(status):
                continue
            status_updates_filtered.append((strategy_fp, axis_dim_fp, window_sig, status))
        if status_updates_filtered:
            _stage_cell_status_set_many(stage_label=str(stage_label), rows=status_updates_filtered)

        manifest_updates_filtered: list[tuple[str, str, str, str]] = []
        manifest_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}
        for axis_dim_fp, window_sig, strategy_fp, status in manifest_updates:
            prev_state = manifest_by_dim_window.get((str(axis_dim_fp), str(window_sig)))
            prev_status = str(prev_state[0]).strip().lower() if isinstance(prev_state, tuple) else ""
            if prev_status == str(status):
                continue
            if manifest_priority.get(str(status), -1) < manifest_priority.get(str(prev_status), -1):
                continue
            manifest_updates_filtered.append((axis_dim_fp, window_sig, strategy_fp, status))
        if manifest_updates_filtered:
            _cartesian_cell_manifest_set_many(stage_label=str(stage_label), rows=manifest_updates_filtered)

        if rank_dominance_stamp_rows_by_window:
            for window_sig, stamp_rows in rank_dominance_stamp_rows_by_window.items():
                _rank_dominance_stamp_set_many(
                    stage_label=str(stage_label),
                    window_signature=str(window_sig),
                    rows=list(stamp_rows),
                )

        if int(stage_cell_total) > 0 and stage_cell_plan_signature and stage_cell_window_signature:
            unresolved_count = int(len(pending_plan))
            _stage_unresolved_summary_set(
                manifest_name="stage_cell",
                stage_label=str(stage_label),
                plan_signature=str(stage_cell_plan_signature),
                window_signature=str(stage_cell_window_signature),
                total=int(stage_cell_total),
                unresolved_count=int(unresolved_count),
                resolved_count=int(max(0, int(stage_cell_total) - int(unresolved_count))),
            )

        return pending_plan, cached_hits, pending_cell_map, int(prefetched_tested)

    def _run_cfg(
        *,
        cfg: ConfigBundle,
        bars: list | None = None,
        regime_bars: list | None = None,
        regime2_bars: list | None = None,
        prepared_context: tuple[list, list | None, list | None, list | None, list | None, object | None] | None = None,
        progress_callback=None,
    ) -> dict | None:
        nonlocal run_calls_total, run_cfg_cache_hits, run_cfg_fingerprint_hits, run_cfg_persistent_hits, run_cfg_persistent_writes
        run_calls_total += 1

        def _emit_cfg_progress(**payload: object) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(dict(payload))
            except Exception:
                return

        prepared_pack = None
        tick_bars = None
        exec_bars = None
        if (
            isinstance(prepared_context, tuple)
            and len(prepared_context) >= 6
            and isinstance(prepared_context[0], list)
        ):
            bars_eff = prepared_context[0]
            regime_eff = prepared_context[1] if isinstance(prepared_context[1], list) else None
            regime2_eff = prepared_context[2] if isinstance(prepared_context[2], list) else None
            tick_bars = prepared_context[3] if isinstance(prepared_context[3], list) else None
            exec_bars = prepared_context[4] if isinstance(prepared_context[4], list) else None
            prepared_pack = prepared_context[5]
        else:
            bars_eff, regime_eff, regime2_eff = _context_bars_for_cfg(
                cfg=cfg,
                bars=bars,
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
            )
        ctx_sig, cache_key, axis_dim_fp, persistent_key = _run_cfg_cache_coords(
            cfg=cfg,
            bars=bars_eff,
            regime_bars=regime_eff,
            regime2_bars=regime2_eff,
            update_dim_index=True,
        )
        cfg_key = str(cache_key[0])
        fp_cached = run_cfg_fingerprint_cache.get(cfg_key)
        if fp_cached is not None and fp_cached[0] == ctx_sig:
            run_cfg_cache_hits += 1
            run_cfg_fingerprint_hits += 1
            fp_row = fp_cached[1]
            row = dict(fp_row) if isinstance(fp_row, dict) else None
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            _axis_progress_record(kept=bool(row))
            return row
        cached = run_cfg_cache.get(cache_key, _RUN_CFG_CACHE_MISS)
        if cached is not _RUN_CFG_CACHE_MISS:
            run_cfg_cache_hits += 1
            run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, cached if isinstance(cached, dict) else None)
            row = dict(cached) if isinstance(cached, dict) else None
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            _axis_progress_record(kept=bool(row))
            return row
        persisted = _run_cfg_persistent_get(cache_key=str(persistent_key))
        if persisted is not _RUN_CFG_CACHE_MISS:
            run_cfg_cache_hits += 1
            run_cfg_persistent_hits += 1
            run_cfg_cache[cache_key] = persisted if isinstance(persisted, dict) else None
            run_cfg_fingerprint_cache[cfg_key] = (ctx_sig, persisted if isinstance(persisted, dict) else None)
            row = dict(persisted) if isinstance(persisted, dict) else None
            _emit_cfg_progress(phase="cfg.cache_hit", cached=True, kept=bool(row))
            _axis_progress_record(kept=bool(row))
            return row
        if tick_bars is None:
            tick_bars = _tick_bars_for(cfg)
        if exec_bars is None:
            exec_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
            if exec_size and str(exec_size) != str(cfg.backtest.bar_size):
                exec_bars = _bars_cached(exec_size)
        _emit_cfg_progress(
            phase="cfg.context_ready",
            cached=False,
            signal_total=int(len(bars_eff) if isinstance(bars_eff, list) else 0),
            regime_total=int(len(regime_eff) if isinstance(regime_eff, list) else 0),
            regime2_total=int(len(regime2_eff) if isinstance(regime2_eff, list) else 0),
            tick_total=int(len(tick_bars) if isinstance(tick_bars, list) else 0),
            exec_total=int(len(exec_bars) if isinstance(exec_bars, list) else int(len(bars_eff) if isinstance(bars_eff, list) else 0)),
        )
        eval_started = pytime.perf_counter()
        s = _run_spot_backtest_summary(
            cfg,
            bars_eff,
            meta,
            regime_bars=regime_eff,
            regime2_bars=regime2_eff,
            tick_bars=tick_bars,
            exec_bars=exec_bars,
            prepared_series_pack=prepared_pack,
            progress_callback=progress_callback,
        )
        eval_elapsed = max(1e-6, float(pytime.perf_counter()) - float(eval_started))
        _emit_cfg_progress(
            phase="cfg.engine_done",
            elapsed_sec=float(eval_elapsed),
            trades=int(getattr(s, "trades", 0) or 0),
        )
        _run_cfg_dimension_index_set(
            fingerprint=str(axis_dim_fp),
            payload_json=str(axis_dim_fp),
            est_cost=float(eval_elapsed),
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
        frontier_stage_label: str | None = None,
        progress_callback=None,
    ) -> tuple[int, list[tuple[ConfigBundle, dict, str, dict | None]]]:
        tested = 0
        kept: list[tuple[ConfigBundle, dict, str, dict | None]] = []
        frontier_updates: list[tuple[str, str, dict | None]] = []
        winner_projection_updates: list[tuple[str, str, dict | None]] = []
        dimension_upper_bound_updates: list[tuple[str, str, dict | None]] = []
        rank_runtime_updates: dict[tuple[str, int], tuple[int, float, int]] = {}
        dimension_utility_updates: dict[tuple[str, str, str], tuple[int, int, int, float]] = {}
        t0 = pytime.perf_counter()
        last = float(t0)
        total_i = int(total) if total is not None else None
        suppress_prev = bool(axis_progress_state.get("suppress"))
        heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 0.0
        heartbeat_eff = max(5.0, float(heartbeat_eff)) if float(heartbeat_eff) > 0.0 else 0.0
        eval_inflight_started_at = 0.0
        eval_inflight_active = False
        eval_phase_state: dict[str, object] = {}
        eval_phase_lock = threading.Lock()
        sweep_heartbeat_stop = threading.Event()
        sweep_heartbeat_thread: threading.Thread | None = None
        dim_util_cfg = _cache_config("dimension_value_utility")
        dim_upper_cfg = _cache_config("dimension_upper_bound")
        dim_util_write_min_total = max(0, int(_registry_float(dim_util_cfg.get("write_min_total"), 128.0)))
        dim_upper_write_min_total = max(0, int(_registry_float(dim_upper_cfg.get("write_min_total"), 96.0)))
        dim_util_write_sample_mod = max(1, int(_registry_float(dim_util_cfg.get("write_sample_mod"), 1.0)))
        dim_upper_write_sample_mod = max(1, int(_registry_float(dim_upper_cfg.get("write_sample_mod"), 1.0)))
        allow_dim_util_writes = bool(
            frontier_stage_label
            and (total_i is None or int(total_i) >= int(dim_util_write_min_total))
        )
        allow_dim_upper_writes = bool(
            frontier_stage_label
            and (total_i is None or int(total_i) >= int(dim_upper_write_min_total))
        )
        series_pack_prewarm_cfg = _cache_config("series_pack_prewarm")
        series_pack_prewarm_enabled = bool(_registry_float(series_pack_prewarm_cfg.get("enabled"), 1.0) > 0.0)
        series_pack_prewarm_min_total = max(0, int(_registry_float(series_pack_prewarm_cfg.get("min_total"), 32.0)))
        series_pack_prewarm_max_unique = max(1, int(_registry_float(series_pack_prewarm_cfg.get("max_unique"), 4096.0)))
        use_prepared_context = bool(
            series_pack_prewarm_enabled
            and (total_i is None or int(total_i) >= int(series_pack_prewarm_min_total))
        )
        prepared_context_by_cache_key: dict[
            tuple[str, str, str],
            tuple[list, list | None, list | None, list | None, list | None, object | None],
        ] = {}
        prepared_series_pack_by_hash: dict[str, object | None] = {}
        if progress_label:
            axis_progress_state["suppress"] = True

        def _emit_progress(done: bool = False) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(
                    tested=int(tested),
                    total=(int(total_i) if total_i is not None else None),
                    kept=int(len(kept)),
                    elapsed=max(0.0, float(pytime.perf_counter()) - float(t0)),
                    done=bool(done),
                )
            except Exception:
                return

        def _emit_sweep_heartbeat() -> None:
            nonlocal last
            if not progress_label and not callable(progress_callback):
                return
            now = float(pytime.perf_counter())
            if progress_label:
                line = _progress_line(
                    label=str(progress_label),
                    tested=int(tested),
                    total=total_i,
                    kept=len(kept),
                    started_at=t0,
                    rate_unit="s",
                )
                if bool(eval_inflight_active) and float(eval_inflight_started_at) > 0.0:
                    line += f" inflight={max(0.0, now - float(eval_inflight_started_at)):0.1f}s"
                with eval_phase_lock:
                    phase_snap = dict(eval_phase_state)
                phase_name = str(phase_snap.get("phase") or "").strip()
                if phase_name:
                    line += f" stage={phase_name}"
                exec_idx = phase_snap.get("exec_idx")
                exec_total = phase_snap.get("exec_total")
                if isinstance(exec_idx, int) and isinstance(exec_total, int) and int(exec_total) > 0:
                    line += f" exec={int(exec_idx)}/{int(exec_total)}"
                sig_idx = phase_snap.get("sig_idx")
                sig_total = phase_snap.get("sig_total")
                if isinstance(sig_idx, int) and isinstance(sig_total, int) and int(sig_total) > 0:
                    line += f" sig={int(sig_idx)}/{int(sig_total)}"
                trades_live = phase_snap.get("trades")
                if isinstance(trades_live, int):
                    line += f" trades={int(trades_live)}"
                window_idx = phase_snap.get("window_idx")
                window_total = phase_snap.get("window_total")
                if isinstance(window_idx, int) and isinstance(window_total, int) and int(window_total) > 0:
                    line += f" window={int(window_idx)}/{int(window_total)}"
                print(line, flush=True)
            last = float(now)
            _emit_progress(done=False)

        def _update_eval_phase(event: dict | None) -> None:
            if not isinstance(event, dict):
                return
            with eval_phase_lock:
                for key in (
                    "phase",
                    "path",
                    "window_idx",
                    "window_total",
                    "signal_total",
                    "regime_total",
                    "regime2_total",
                    "tick_total",
                    "exec_total",
                    "sig_idx",
                    "exec_idx",
                    "open_count",
                    "trades",
                    "cached",
                    "kept",
                ):
                    if key in event:
                        eval_phase_state[str(key)] = event.get(key)

        def _sweep_heartbeat_worker() -> None:
            if float(heartbeat_eff) <= 0.0:
                return
            while not sweep_heartbeat_stop.wait(float(heartbeat_eff)):
                if bool(eval_inflight_active):
                    _emit_sweep_heartbeat()

        if float(heartbeat_eff) > 0.0 and (progress_label or callable(progress_callback)):
            sweep_heartbeat_thread = threading.Thread(target=_sweep_heartbeat_worker, daemon=True)
            sweep_heartbeat_thread.start()

        def _flush_rank_runtime_updates() -> None:
            if not frontier_stage_label or not rank_runtime_updates:
                return
            rows_to_set = [
                (str(window_sig), int(rank_bin), int(rec[0]), float(rec[1]), int(rec[2]))
                for (window_sig, rank_bin), rec in rank_runtime_updates.items()
                if str(window_sig) and int(rec[0]) > 0
            ]
            rank_runtime_updates.clear()
            if rows_to_set:
                _rank_bin_runtime_set_many(stage_label=str(frontier_stage_label), rows=rows_to_set)

        def _flush_dimension_utility_updates() -> None:
            if not frontier_stage_label or not dimension_utility_updates:
                return
            rows_to_set = [
                (
                    str(window_sig),
                    str(dim_key),
                    str(dim_value),
                    int(rec[0]),
                    int(rec[1]),
                    int(rec[2]),
                    float(rec[3]),
                )
                for (window_sig, dim_key, dim_value), rec in dimension_utility_updates.items()
                if str(window_sig) and str(dim_key) and int(rec[0]) > 0
            ]
            dimension_utility_updates.clear()
            if rows_to_set:
                _dimension_value_utility_set_many(stage_label=str(frontier_stage_label), rows=rows_to_set)

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
                        _emit_progress(done=False)

                prepared_context = None
                if bool(use_prepared_context):
                    _ctx_sig_pc, cache_key_pc, _axis_dim_fp_pc, _persistent_key_pc = _run_cfg_cache_coords(
                        cfg=cfg,
                        bars=bars,
                        update_dim_index=False,
                    )
                    prepared_context = prepared_context_by_cache_key.get(cache_key_pc)
                    if prepared_context is None and len(prepared_context_by_cache_key) < int(series_pack_prewarm_max_unique):
                        bars_eff_pc, regime_eff_pc, regime2_eff_pc = _context_bars_for_cfg(
                            cfg=cfg,
                            bars=bars,
                            regime_bars=None,
                            regime2_bars=None,
                        )
                        tick_bars_pc = _tick_bars_for(cfg)
                        exec_bars_pc = None
                        exec_size_pc = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
                        if exec_size_pc and str(exec_size_pc) != str(cfg.backtest.bar_size):
                            exec_bars_pc = _bars_cached(exec_size_pc)
                        pack_hash, prepared_pack = _spot_prepare_summary_series_pack(
                            cfg=cfg,
                            signal_bars=bars_eff_pc,
                            regime_bars=regime_eff_pc,
                            regime2_bars=regime2_eff_pc,
                            tick_bars=tick_bars_pc,
                            exec_bars=exec_bars_pc,
                        )
                        if pack_hash:
                            if pack_hash in prepared_series_pack_by_hash:
                                prepared_pack = prepared_series_pack_by_hash.get(pack_hash)
                            elif len(prepared_series_pack_by_hash) < int(series_pack_prewarm_max_unique):
                                prepared_series_pack_by_hash[str(pack_hash)] = prepared_pack
                        prepared_context = (
                            bars_eff_pc,
                            regime_eff_pc,
                            regime2_eff_pc,
                            tick_bars_pc,
                            exec_bars_pc,
                            prepared_pack,
                        )
                        prepared_context_by_cache_key[cache_key_pc] = prepared_context

                cache_hits_before = int(run_cfg_cache_hits)
                eval_started = pytime.perf_counter()
                eval_inflight_started_at = float(eval_started)
                eval_inflight_active = True
                try:
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars,
                        prepared_context=prepared_context,
                        progress_callback=_update_eval_phase,
                    )
                finally:
                    eval_inflight_active = False
                    eval_inflight_started_at = 0.0
                eval_elapsed = max(1e-6, float(pytime.perf_counter()) - float(eval_started))
                cache_hit_eval = 1 if int(run_cfg_cache_hits) > int(cache_hits_before) else 0
                if frontier_stage_label:
                    _ctx_sig, cache_key, axis_dim_fp, _persistent_key = _run_cfg_cache_coords(
                        cfg=cfg,
                        bars=bars,
                        update_dim_index=False,
                    )
                    window_sig = str(cache_key[2])
                    frontier_updates.append((str(axis_dim_fp), str(window_sig), row if isinstance(row, dict) else None))
                    if len(frontier_updates) >= 200:
                        _stage_frontier_upsert_many(
                            stage_label=str(frontier_stage_label),
                            rows=frontier_updates,
                        )
                        frontier_updates.clear()
                    winner_projection_updates.append((str(axis_dim_fp), str(window_sig), row if isinstance(row, dict) else None))
                    if len(winner_projection_updates) >= 200:
                        _winner_projection_upsert_many(
                            stage_label=str(frontier_stage_label),
                            rows=winner_projection_updates,
                        )
                        winner_projection_updates.clear()
                    if bool(allow_dim_upper_writes) and (
                        int(dim_upper_write_sample_mod) <= 1
                        or (int(tested) % int(dim_upper_write_sample_mod) == 0)
                    ):
                        dimension_upper_bound_updates.append((str(axis_dim_fp), str(window_sig), row if isinstance(row, dict) else None))
                        if len(dimension_upper_bound_updates) >= 200:
                            _dimension_upper_bound_upsert_many(
                                stage_label=str(frontier_stage_label),
                                rows=dimension_upper_bound_updates,
                            )
                            dimension_upper_bound_updates.clear()
                    rank_raw = meta_item.get("_mr_rank") if isinstance(meta_item, dict) else None
                    try:
                        rank_i = int(rank_raw)
                    except (TypeError, ValueError):
                        rank_i = -1
                    if rank_i >= 0 and window_sig:
                        rank_bin = int(_rank_bin_from_rank(rank_i))
                        cell = (str(window_sig), int(rank_bin))
                        prev = rank_runtime_updates.get(cell)
                        if prev is None:
                            rank_runtime_updates[cell] = (1, float(eval_elapsed), int(cache_hit_eval))
                        else:
                            rank_runtime_updates[cell] = (
                                int(prev[0]) + 1,
                                float(prev[1]) + float(eval_elapsed),
                                int(prev[2]) + int(cache_hit_eval),
                            )
                        if len(rank_runtime_updates) >= 256:
                            _flush_rank_runtime_updates()
                    if (
                        window_sig
                        and isinstance(meta_item, dict)
                        and bool(allow_dim_util_writes)
                        and (
                            int(dim_util_write_sample_mod) <= 1
                            or (int(tested) % int(dim_util_write_sample_mod) == 0)
                        )
                    ):
                        keep_i = 1 if isinstance(row, dict) else 0
                        for raw_dim_key, raw_dim_value in meta_item.items():
                            dim_key = str(raw_dim_key or "").strip()
                            if not dim_key or dim_key.startswith("_"):
                                continue
                            dim_value = str(raw_dim_value)
                            cell = (str(window_sig), str(dim_key), str(dim_value))
                            prev = dimension_utility_updates.get(cell)
                            if prev is None:
                                dimension_utility_updates[cell] = (1, int(keep_i), int(cache_hit_eval), float(eval_elapsed))
                            else:
                                dimension_utility_updates[cell] = (
                                    int(prev[0]) + 1,
                                    int(prev[1]) + int(keep_i),
                                    int(prev[2]) + int(cache_hit_eval),
                                    float(prev[3]) + float(eval_elapsed),
                                )
                        if len(dimension_utility_updates) >= 768:
                            _flush_dimension_utility_updates()
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
            sweep_heartbeat_stop.set()
            if sweep_heartbeat_thread is not None:
                try:
                    sweep_heartbeat_thread.join(timeout=1.0)
                except Exception:
                    pass
            if frontier_updates and frontier_stage_label:
                _stage_frontier_upsert_many(
                    stage_label=str(frontier_stage_label),
                    rows=frontier_updates,
                )
            if winner_projection_updates and frontier_stage_label:
                _winner_projection_upsert_many(
                    stage_label=str(frontier_stage_label),
                    rows=winner_projection_updates,
                )
            if dimension_upper_bound_updates and frontier_stage_label:
                _dimension_upper_bound_upsert_many(
                    stage_label=str(frontier_stage_label),
                    rows=dimension_upper_bound_updates,
                )
            _flush_rank_runtime_updates()
            _flush_dimension_utility_updates()
            _run_cfg_persistent_flush_pending(force=True)
            _emit_progress(done=True)
            axis_progress_state["suppress"] = suppress_prev

        return tested, kept

    def _cfg_from_strategy_filters_payload(strategy_payload, filters_payload) -> ConfigBundle | None:
        if not isinstance(strategy_payload, dict):
            return None
        try:
            filters_obj = _codec_filters_from_payload(filters_payload if isinstance(filters_payload, dict) else None)
            strategy_obj = _codec_strategy_from_payload(strategy_payload, filters=filters_obj)
        except Exception:
            return None
        return _codec_make_bundle(
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
        cfg_catalog: dict[str, tuple[dict, dict | None]] | None = None,
    ) -> tuple[ConfigBundle, str] | None:
        if not isinstance(payload, dict):
            return None
        strategy_payload = payload.get("strategy")
        filters_payload = payload.get("filters")
        if filters_payload is not None and not isinstance(filters_payload, dict):
            filters_payload = None
        cfg_ref = str(payload.get("cfg_ref") or "").strip()
        if cfg_ref and isinstance(cfg_catalog, dict):
            resolved = cfg_catalog.get(cfg_ref)
            if isinstance(resolved, tuple) and len(resolved) == 2:
                strategy_payload, filters_payload = resolved
        cfg = _cfg_from_strategy_filters_payload(strategy_payload, filters_payload)
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

        cfg_catalog: dict[str, tuple[dict, dict | None]] = {}
        cfg_catalog_raw = payload.get("_cfg_catalog")
        if isinstance(cfg_catalog_raw, list):
            for item in cfg_catalog_raw:
                if not isinstance(item, dict):
                    continue
                cfg_ref = str(item.get("cfg_ref") or "").strip()
                strategy_payload = item.get("strategy")
                filters_payload = item.get("filters")
                if not cfg_ref or not isinstance(strategy_payload, dict):
                    continue
                if filters_payload is not None and not isinstance(filters_payload, dict):
                    filters_payload = None
                cfg_catalog[cfg_ref] = (dict(strategy_payload), dict(filters_payload) if isinstance(filters_payload, dict) else None)

        def _require_list(key: str) -> list:
            raw = payload.get(key)
            if not isinstance(raw, list):
                raise SystemExit(f"{schema_name} payload missing '{key}' list: {payload_path}")
            return raw

        def _decode_cfg_list(*, key: str, note_key: str) -> list[tuple[ConfigBundle, str]]:
            out: list[tuple[ConfigBundle, str]] = []
            for item in _require_list(key):
                decoded = _decode_cfg_payload(item, note_key=note_key, cfg_catalog=cfg_catalog)
                if decoded is None:
                    continue
                cfg_obj, note = decoded
                out.append((cfg_obj, str(note)))
            return out

        name = str(schema_name).strip().lower()
        if name == "cfg_stage":
            return {
                "axis_tag": str(payload.get("axis_tag") or ""),
                "cfg_pairs": _decode_cfg_list(key="cfgs", note_key="note"),
            }
        raise SystemExit(f"Unknown worker payload schema: {schema_name!r}")

    def _cfg_catalog_from_payload(payload: dict) -> dict[str, tuple[dict, dict | None]]:
        out: dict[str, tuple[dict, dict | None]] = {}
        cfg_catalog_raw = payload.get("_cfg_catalog")
        if not isinstance(cfg_catalog_raw, list):
            return out
        for item in cfg_catalog_raw:
            if not isinstance(item, dict):
                continue
            cfg_ref = str(item.get("cfg_ref") or "").strip()
            strategy_payload = item.get("strategy")
            filters_payload = item.get("filters")
            if not cfg_ref or not isinstance(strategy_payload, dict):
                continue
            if filters_payload is not None and not isinstance(filters_payload, dict):
                filters_payload = None
            out[cfg_ref] = (dict(strategy_payload), dict(filters_payload) if isinstance(filters_payload, dict) else None)
        return out

    def _cfg_stage_payload_signature(*, axis_tag: str, cfg_records: list[dict], cfg_catalog: dict[str, tuple[dict, dict | None]]) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(_RUN_CFG_CACHE_ENGINE_VERSION).encode("utf-8"))
        hasher.update(str(axis_tag).strip().lower().encode("utf-8"))
        hasher.update(str(int(run_min_trades)).encode("utf-8"))
        hasher.update(str(len(cfg_records)).encode("utf-8"))
        hasher.update(str(len(cfg_catalog)).encode("utf-8"))
        for rec in cfg_records:
            if not isinstance(rec, dict):
                hasher.update(str(rec).encode("utf-8"))
                continue
            cfg_ref = str(rec.get("cfg_ref") or "").strip()
            if cfg_ref:
                hasher.update(str(cfg_ref).encode("utf-8"))
                continue
            payload_sig = hashlib.sha1(json.dumps(rec, sort_keys=True, default=str).encode("utf-8")).digest()
            hasher.update(payload_sig)
        return hasher.hexdigest()

    def _worker_records_from_kept(kept: list[tuple[ConfigBundle, dict, str, dict | None]]) -> list[dict]:
        records: list[dict] = []
        for cfg, row, note, _meta in kept:
            records.append(_encode_cfg_payload(cfg, note=note, extra={"row": row}))
        return records

    def _plan_item_cfg(item) -> ConfigBundle | None:
        if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], ConfigBundle):
            return item[0]
        return None

    def _plan_item_meta(item) -> dict | None:
        if isinstance(item, tuple) and len(item) >= 3 and isinstance(item[2], dict):
            return item[2]
        return None

    def _plan_item_mixed_radix_rank(item) -> int | None:
        meta_item = _plan_item_meta(item)
        if not isinstance(meta_item, dict):
            return None
        raw = meta_item.get("_mr_rank")
        try:
            out = int(raw)
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return int(out)

    def _plan_item_stage_rank(item) -> int | None:
        meta_item = _plan_item_meta(item)
        if not isinstance(meta_item, dict):
            return None
        raw = meta_item.get("_stage_rank")
        try:
            out = int(raw)
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return int(out)

    def _plan_items_with_stage_ranks(plan_all) -> list:
        out: list = []
        for rank_i, item in enumerate(list(plan_all or ())):
            if not isinstance(item, tuple):
                out.append(item)
                continue
            if len(item) < 2:
                out.append(item)
                continue
            meta_item = item[2] if len(item) >= 3 and isinstance(item[2], dict) else {}
            meta_out = dict(meta_item) if isinstance(meta_item, dict) else {}
            meta_out["_stage_rank"] = int(rank_i)
            prefix = [item[0], item[1], meta_out]
            if len(item) > 3:
                prefix.extend(item[3:])
            out.append(tuple(prefix))
        return out

    def _stage_rank_manifest_plan_signature(*, stage_label: str, plan_all) -> str:
        hasher = hashlib.sha1()
        hasher.update(str(_RUN_CFG_CACHE_ENGINE_VERSION).encode("utf-8"))
        hasher.update(str(stage_label).strip().lower().encode("utf-8"))
        items = list(plan_all or ())
        hasher.update(str(len(items)).encode("utf-8"))
        for item in items:
            cfg = _plan_item_cfg(item)
            if cfg is None:
                hasher.update(str(item).encode("utf-8"))
                continue
            hasher.update(hashlib.sha1(_milestone_key(cfg).encode("utf-8")).digest())
            hasher.update(hashlib.sha1(_axis_dimension_fingerprint(cfg).encode("utf-8")).digest())
        return hasher.hexdigest()

    def _stage_rank_manifest_window_signature(*, bars: list | None) -> str:
        bars_sig = _bars_signature(bars)
        return _window_signature(
            bars_sig=bars_sig,
            regime_sig=(0, None, None),
            regime2_sig=(0, None, None),
        )

    def _plan_item_dimension_values(item) -> tuple[tuple[str, str], ...]:
        meta_item = _plan_item_meta(item)
        if not isinstance(meta_item, dict):
            return ()
        out: list[tuple[str, str]] = []
        for raw_key, raw_val in meta_item.items():
            key = str(raw_key or "").strip()
            if not key or key.startswith("_"):
                continue
            out.append((str(key), str(raw_val)))
        out.sort(key=lambda row: row[0])
        return tuple(out)

    def _mixed_radix_warm_ranges(
        *,
        plan_all,
        max_bins: int = 24,
    ) -> tuple[tuple[int, int, int], ...]:
        ranks: list[int] = []
        for item in plan_all:
            rank = _plan_item_mixed_radix_rank(item)
            if rank is None:
                continue
            ranks.append(int(rank))
        if not ranks:
            return ()
        r_min = int(min(ranks))
        r_max = int(max(ranks))
        span_total = max(1, int(r_max - r_min + 1))
        bins_target = max(1, min(int(max_bins), max(1, int(math.sqrt(len(ranks))))))
        bin_span = max(1, int(math.ceil(float(span_total) / float(bins_target))))
        bin_counts: dict[int, int] = {}
        for rank in ranks:
            b = int((int(rank) - int(r_min)) // int(bin_span))
            bin_counts[b] = int(bin_counts.get(b, 0)) + 1
        ordered_bins = sorted(bin_counts.items(), key=lambda row: (-int(row[1]), int(row[0])))
        out: list[tuple[int, int, int]] = []
        for b, count in ordered_bins:
            lo = int(r_min + (int(b) * int(bin_span)))
            hi = int(min(int(r_max), int(lo + int(bin_span) - 1)))
            out.append((int(lo), int(hi), int(count)))
        return tuple(out)

    def _compress_rank_status_rows(rank_status: dict[int, str]) -> list[tuple[int, int, str]]:
        rows = [
            (int(rank), str(status))
            for rank, status in rank_status.items()
            if str(status) in ("cached_hit", "evaluated", "dominated")
        ]
        if not rows:
            return []
        rows.sort(key=lambda row: int(row[0]))
        out: list[tuple[int, int, str]] = []
        cur_lo = int(rows[0][0])
        cur_hi = int(rows[0][0])
        cur_status = str(rows[0][1])
        for rank, status in rows[1:]:
            rank_i = int(rank)
            status_s = str(status)
            if status_s == cur_status and rank_i == (int(cur_hi) + 1):
                cur_hi = int(rank_i)
                continue
            out.append((int(cur_lo), int(cur_hi), str(cur_status)))
            cur_lo = int(rank_i)
            cur_hi = int(rank_i)
            cur_status = str(status_s)
        out.append((int(cur_lo), int(cur_hi), str(cur_status)))
        return out

    def _partition_rank_ranges_for_workers(
        *,
        ranges: tuple[tuple[int, int], ...],
        workers: int,
    ) -> list[list[tuple[int, int]]]:
        workers_n = max(1, int(workers))
        out: list[list[tuple[int, int]]] = [[] for _ in range(workers_n)]
        if workers_n <= 1:
            out[0] = list(ranges)
            return out
        loads: list[int] = [0 for _ in range(workers_n)]
        ordered = sorted(
            [(int(lo), int(hi)) for lo, hi in ranges if int(hi) >= int(lo)],
            key=lambda row: (-(int(row[1]) - int(row[0]) + 1), int(row[0])),
        )
        total_span = sum(int(rank_hi) - int(rank_lo) + 1 for rank_lo, rank_hi in ordered)
        # Split into finer chunks than worker count so each worker receives multiple disjoint
        # ranges; this reduces straggler risk when rank-local cost is non-uniform.
        target_chunk_span = max(1, int(math.ceil(float(total_span) / float(max(1, workers_n * 4)))))
        chunks: list[tuple[int, int]] = []
        for rank_lo, rank_hi in ordered:
            cursor = int(rank_lo)
            hi_i = int(rank_hi)
            while int(cursor) <= int(hi_i):
                chunk_hi = min(int(hi_i), int(cursor) + int(target_chunk_span) - 1)
                chunks.append((int(cursor), int(chunk_hi)))
                cursor = int(chunk_hi) + 1
        for rank_lo, rank_hi in chunks:
            span = int(rank_hi) - int(rank_lo) + 1
            target = min(range(workers_n), key=lambda wid: (int(loads[wid]), int(wid)))
            out[int(target)].append((int(rank_lo), int(rank_hi)))
            loads[int(target)] += int(span)
        for bucket in out:
            bucket.sort(key=lambda row: int(row[0]))
        return out

    def _cfg_eval_cost_hint(cfg: ConfigBundle) -> float:
        strat = cfg.strategy
        signal_bar = str(cfg.backtest.bar_size)
        filters_payload = _filters_payload(strat.filters) or {}
        axis_fp = _axis_dimension_fingerprint(cfg)
        persisted_cost = float(run_cfg_dim_index_loaded.get(str(axis_fp), 0.0) or 0.0)
        cost = _cost_model_weight("base", 1.0)

        regime_mode = str(getattr(strat, "regime_mode", "ema") or "ema").strip().lower()
        regime_bar = str(getattr(strat, "regime_bar_size", "") or "").strip() or signal_bar
        if regime_bar != signal_bar and (
            regime_mode == "supertrend" or bool(getattr(strat, "regime_ema_preset", None))
        ):
            cost += _cost_model_weight("regime_cross_tf", 0.5)

        regime2_mode = str(getattr(strat, "regime2_mode", "off") or "off").strip().lower()
        regime2_bar = str(getattr(strat, "regime2_bar_size", "") or "").strip() or signal_bar
        if regime2_mode != "off" and regime2_bar != signal_bar:
            cost += _cost_model_weight("regime2_cross_tf", 0.5)

        if str(getattr(strat, "tick_gate_mode", "off") or "off").strip().lower() != "off":
            cost += _cost_model_weight("tick_gate_on", 0.75)

        exec_size = str(getattr(strat, "spot_exec_bar_size", "") or "").strip()
        if exec_size and exec_size != signal_bar:
            cost += _cost_model_weight("exec_cross_tf", 0.75)

        perm_on = any(
            filters_payload.get(k) is not None
            for k in (
                "ema_spread_min_pct",
                "ema_spread_min_pct_down",
                "ema_slope_min_pct",
                "ema_slope_signed_min_pct_up",
                "ema_slope_signed_min_pct_down",
            )
        )
        if perm_on:
            cost += _cost_model_weight("perm_gate_on", 0.15) * _axis_cost_hint("perm_joint", "perm_variants", 1.0)

        if filters_payload.get("entry_start_hour_et") is not None and filters_payload.get("entry_end_hour_et") is not None:
            cost += _cost_model_weight("tod_gate_on", 0.12) * _axis_cost_hint("perm_joint", "tod_windows", 1.0)

        if filters_payload.get("volume_ratio_min") is not None:
            cost += _cost_model_weight("volume_gate_on", 0.08) * _axis_cost_hint("perm_joint", "vol_variants", 1.0)

        cooldown_raw = filters_payload.get("cooldown_bars")
        skip_raw = filters_payload.get("skip_first_bars")
        try:
            cooldown_on = float(cooldown_raw or 0.0) > 0.0
        except (TypeError, ValueError):
            cooldown_on = bool(cooldown_raw)
        try:
            skip_on = float(skip_raw or 0.0) > 0.0
        except (TypeError, ValueError):
            skip_on = bool(skip_raw)
        if cooldown_on or skip_on:
            cost += _cost_model_weight("cadence_gate_on", 0.05) * _axis_cost_hint("perm_joint", "cadence_variants", 1.0)

        if str(filters_payload.get("shock_gate_mode") or "off").strip().lower() != "off":
            cost += _cost_model_weight("shock_gate_on", 0.4) * _axis_cost_hint("shock", "modes", 1.0)
        if filters_payload.get("riskoff_tr5_med_pct") is not None:
            cost += _cost_model_weight("riskoff_overlay_on", 0.2) * _axis_cost_hint("risk_overlays", "riskoff", 1.0)
        if filters_payload.get("riskpanic_tr5_med_pct") is not None and filters_payload.get("riskpanic_neg_gap_ratio_min") is not None:
            cost += _cost_model_weight("riskpanic_overlay_on", 0.35) * _axis_cost_hint("risk_overlays", "riskpanic", 1.0)
        if filters_payload.get("riskpop_tr5_med_pct") is not None and filters_payload.get("riskpop_pos_gap_ratio_min") is not None:
            cost += _cost_model_weight("riskpop_overlay_on", 0.3) * _axis_cost_hint("risk_overlays", "riskpop", 1.0)
        if persisted_cost > 0.0:
            return float((0.65 * float(persisted_cost)) + (0.35 * float(cost)))
        return float(cost)

    def _cfg_locality_bucket_key(cfg: ConfigBundle) -> str:
        strat = cfg.strategy
        signal_bar = str(cfg.backtest.bar_size)
        regime_mode = str(getattr(strat, "regime_mode", "ema") or "ema").strip().lower()
        regime_bar = str(getattr(strat, "regime_bar_size", "") or "").strip() or signal_bar
        regime2_mode = str(getattr(strat, "regime2_mode", "off") or "off").strip().lower()
        regime2_bar = str(getattr(strat, "regime2_bar_size", "") or "").strip() or signal_bar
        tick_mode = str(getattr(strat, "tick_gate_mode", "off") or "off").strip().lower()
        exec_size = str(getattr(strat, "spot_exec_bar_size", "") or "").strip() or signal_bar
        filters_payload = _filters_payload(strat.filters) or {}
        raw = {
            "signal_bar": signal_bar,
            "regime": (regime_mode, regime_bar),
            "regime2": (regime2_mode, regime2_bar),
            "tick_mode": tick_mode,
            "exec_size": exec_size,
            "shock_mode": str(filters_payload.get("shock_gate_mode") or "off").strip().lower(),
            "risk_profile": (
                filters_payload.get("riskoff_tr5_med_pct") is not None,
                filters_payload.get("riskpanic_tr5_med_pct") is not None,
                filters_payload.get("riskpop_tr5_med_pct") is not None,
            ),
            "axis_dim_fp": _axis_dimension_fingerprint(cfg),
        }
        return json.dumps(raw, sort_keys=True, default=str)

    def _dimension_value_utility_score(
        row: dict[str, float] | None,
        *,
        cfg: dict[str, object] | None = None,
    ) -> float:
        if not isinstance(row, dict):
            return 0.0
        cfg_eff = cfg if isinstance(cfg, dict) else _cache_config("dimension_value_utility")
        min_eval_count = int(_registry_float(cfg_eff.get("min_eval_count"), 6.0))
        weight_keep = float(_registry_float(cfg_eff.get("weight_keep_rate"), 0.70))
        weight_hit = float(_registry_float(cfg_eff.get("weight_hit_rate"), 0.20))
        weight_conf = float(_registry_float(cfg_eff.get("weight_confidence"), 0.10))
        confidence_eval_scale = float(_registry_float(cfg_eff.get("confidence_eval_scale"), 24.0))
        eval_floor = float(_registry_float(cfg_eff.get("eval_sec_floor"), 0.01))

        eval_count = float(max(0.0, float(row.get("eval_count", 0.0) or 0.0)))
        if eval_count < float(max(1, min_eval_count)):
            return 0.0
        keep_rate = float(max(0.0, min(1.0, float(row.get("keep_rate", 0.0) or 0.0))))
        hit_rate = float(max(0.0, min(1.0, float(row.get("hit_rate", 0.0) or 0.0))))
        avg_eval_sec = float(max(eval_floor, float(row.get("avg_eval_sec", 0.0) or 0.0)))
        confidence = float(min(1.0, eval_count / float(max(1.0, confidence_eval_scale))))
        numerator = (weight_keep * keep_rate) + (weight_hit * hit_rate) + (weight_conf * confidence)
        return float(max(0.0, numerator) / max(eval_floor, avg_eval_sec))

    def _worker_bucketed_indices(
        *,
        plan_all,
        workers: int,
        bars: list | None = None,
        stage_label: str = "",
        warm_ranges: tuple[tuple[int, int, int], ...] | None = None,
    ) -> list[list[int]]:
        nonlocal run_cfg_dim_index_loaded_once, series_pack_mmap_hint_hits, series_pack_pickle_hint_hits
        nonlocal dimension_utility_hint_hits
        if not bool(run_cfg_dim_index_loaded_once):
            loaded = _run_cfg_dimension_index_load()
            if loaded:
                run_cfg_dim_index_loaded.update(loaded)
            run_cfg_dim_index_loaded_once = True
        workers_n = max(1, int(workers))
        total = len(plan_all)
        if workers_n <= 1:
            return [list(range(total))]
        if total <= 1:
            out_small: list[list[int]] = [[] for _ in range(workers_n)]
            if total == 1:
                out_small[0].append(0)
            return out_small

        grouped: dict[str, list[tuple[int, float, float]]] = {}
        stage_key = _stage_cache_scope(stage_label)
        plan_items: list[
            tuple[
                int,
                str,
                float,
                tuple[str, str, str] | None,
                str | None,
                tuple[str, str] | None,
                str | None,
                tuple[tuple[str, str], ...],
                int | None,
                ConfigBundle | None,
            ]
        ] = []
        persistent_keys: list[str] = []
        state_keys_pending_fetch: list[tuple[str, str]] = []
        for idx, item in enumerate(plan_all):
            cfg = _plan_item_cfg(item)
            mixed_radix_rank = _plan_item_mixed_radix_rank(item)
            dimension_pairs = _plan_item_dimension_values(item)
            if cfg is None:
                cost = 1.0
                bucket_key = "default"
                plan_items.append((int(idx), str(bucket_key), float(cost), None, None, None, None, dimension_pairs, mixed_radix_rank, None))
            else:
                cost = _cfg_eval_cost_hint(cfg)
                bucket_key = _cfg_locality_bucket_key(cfg)
                _ctx_sig, cache_key, _axis_dim_fp, persistent_key = _run_cfg_cache_coords(
                    cfg=cfg,
                    bars=bars,
                    update_dim_index=False,
                )
                state_key = (str(cache_key[0]), str(cache_key[2]))
                plan_items.append(
                    (
                        int(idx),
                        str(bucket_key),
                        float(cost),
                        cache_key,
                        str(persistent_key),
                        state_key,
                        str(cache_key[2]),
                        dimension_pairs,
                        mixed_radix_rank,
                        cfg,
                    )
                )
                persistent_keys.append(str(persistent_key))
                if state_key not in run_cfg_series_pack_state_cache:
                    state_keys_pending_fetch.append((str(state_key[0]), str(state_key[1])))

        if state_keys_pending_fetch:
            persisted_states = _series_pack_state_manifest_get_many(cells=list(state_keys_pending_fetch))
            for state_key, state_val in persisted_states.items():
                run_cfg_series_pack_state_cache[(str(state_key[0]), str(state_key[1]))] = str(state_val)

        state_manifest_updates: list[tuple[str, str, str]] = []
        plan_items_resolved: list[
            tuple[
                int,
                str,
                float,
                tuple[str, str, str] | None,
                str | None,
                str,
                str | None,
                tuple[tuple[str, str], ...],
                int | None,
            ]
        ] = []
        for (
            idx,
            bucket_key,
            cost,
            cache_key,
            persistent_key,
            state_key,
            window_sig,
            dimension_pairs,
            mixed_radix_rank,
            cfg,
        ) in plan_items:
            series_pack_state = "none"
            if state_key is not None and isinstance(cfg, ConfigBundle):
                series_pack_state = str(run_cfg_series_pack_state_cache.get(state_key) or "").strip().lower()
                if series_pack_state not in ("mmap", "pickle", "none"):
                    bars_eff, regime_eff, regime2_eff = _context_bars_for_cfg(
                        cfg=cfg,
                        bars=bars,
                    )
                    exec_bars_eff = bars_eff
                    exec_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
                    if exec_size and str(exec_size) != str(cfg.backtest.bar_size):
                        exec_bars_eff = _bars_cached(exec_size)
                    tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
                    tick_bars_eff = _tick_bars_for(cfg) if tick_mode != "off" else None
                    filters_obj = cfg.strategy.filters
                    needs_rv = filters_obj is not None and (
                        getattr(filters_obj, "rv_min", None) is not None or getattr(filters_obj, "rv_max", None) is not None
                    )
                    needs_volume = filters_obj is not None and getattr(filters_obj, "volume_ratio_min", None) is not None
                    series_pack_state = str(
                        _spot_series_pack_cache_state(
                            cfg=cfg,
                            signal_bars=bars_eff,
                            exec_bars=exec_bars_eff,
                            regime_bars=regime_eff,
                            regime2_bars=regime2_eff,
                            tick_bars=tick_bars_eff,
                            include_rv=bool(needs_rv),
                            include_volume=bool(needs_volume),
                            include_tick=(tick_mode != "off"),
                        )
                    ).strip().lower()
                    if series_pack_state not in ("mmap", "pickle", "none"):
                        series_pack_state = "none"
                    run_cfg_series_pack_state_cache[(str(state_key[0]), str(state_key[1]))] = str(series_pack_state)
                    state_manifest_updates.append((str(state_key[0]), str(state_key[1]), str(series_pack_state)))
            plan_items_resolved.append(
                (
                    int(idx),
                    str(bucket_key),
                    float(cost),
                    cache_key,
                    str(persistent_key) if persistent_key is not None else None,
                    str(series_pack_state or "none"),
                    str(window_sig) if window_sig is not None else None,
                    dimension_pairs,
                    mixed_radix_rank,
                )
            )
        if state_manifest_updates:
            _series_pack_state_manifest_set_many(rows=state_manifest_updates)
        plan_items = plan_items_resolved

        persisted_by_key = _run_cfg_persistent_get_many(cache_keys=persistent_keys) if persistent_keys else {}
        runtime_cells: list[tuple[str, int]] = []
        utility_cells: list[tuple[str, str, str]] = []
        if stage_key:
            for (
                _idx,
                _bucket_key,
                _cost,
                cache_key,
                _persistent_key,
                _series_pack_state,
                window_sig,
                dimension_pairs,
                mixed_radix_rank,
            ) in plan_items:
                if cache_key is None or window_sig is None or mixed_radix_rank is None:
                    if window_sig is not None and dimension_pairs:
                        for dim_key, dim_value in dimension_pairs:
                            utility_cells.append((str(window_sig), str(dim_key), str(dim_value)))
                    continue
                runtime_cells.append((str(window_sig), int(_rank_bin_from_rank(int(mixed_radix_rank)))))
                if dimension_pairs:
                    for dim_key, dim_value in dimension_pairs:
                        utility_cells.append((str(window_sig), str(dim_key), str(dim_value)))
        runtime_by_cell = (
            _rank_bin_runtime_get_many(stage_label=str(stage_key), cells=runtime_cells)
            if runtime_cells
            else {}
        )
        dim_utility_cfg = _cache_config("dimension_value_utility")
        dim_utility_by_cell = (
            _dimension_value_utility_get_many(stage_label=str(stage_key), cells=utility_cells)
            if utility_cells
            else {}
        )
        prepared_items: list[tuple[int, str, float, int | None, float]] = []
        for (
            idx,
            bucket_key,
            cost,
            cache_key,
            persistent_key,
            series_pack_state,
            window_sig,
            dimension_pairs,
            mixed_radix_rank,
        ) in plan_items:
            adjusted_cost = float(cost)
            utility_score = 0.0
            utility_scores: list[float] = []
            if window_sig is not None and dimension_pairs:
                for dim_key, dim_value in dimension_pairs:
                    utility_row = dim_utility_by_cell.get((str(window_sig), str(dim_key), str(dim_value)))
                    utility = _dimension_value_utility_score(utility_row, cfg=dim_utility_cfg)
                    if utility > 0.0:
                        utility_scores.append(float(utility))
                if utility_scores:
                    utility_score = float(sum(float(v) for v in utility_scores) / float(max(1, len(utility_scores))))
                    dimension_utility_hint_hits += len(utility_scores)
            if cache_key is not None and persistent_key is not None:
                cached = run_cfg_cache.get(cache_key, _RUN_CFG_CACHE_MISS)
                if cached is _RUN_CFG_CACHE_MISS:
                    persisted = persisted_by_key.get(str(persistent_key), _RUN_CFG_CACHE_MISS)
                    if persisted is not _RUN_CFG_CACHE_MISS:
                        run_cfg_cache[cache_key] = persisted if isinstance(persisted, dict) else None
                        cached = persisted
                if cached is not _RUN_CFG_CACHE_MISS:
                    adjusted_cost = max(0.001, float(adjusted_cost) * 0.03)
                else:
                    if window_sig is not None and mixed_radix_rank is not None:
                        hint = runtime_by_cell.get((str(window_sig), int(_rank_bin_from_rank(int(mixed_radix_rank)))))
                        if isinstance(hint, dict):
                            avg_eval_sec = float(hint.get("avg_eval_sec", 0.0) or 0.0)
                            hit_rate = float(hint.get("hit_rate", 0.0) or 0.0)
                            if avg_eval_sec > 0.0:
                                adjusted_cost = max(
                                    0.001,
                                    (0.45 * float(adjusted_cost)) + (0.55 * float(avg_eval_sec)),
                                )
                            if hit_rate > 0.0:
                                adjusted_cost = max(
                                    0.001,
                                    float(adjusted_cost) * (1.0 - (0.55 * max(0.0, min(1.0, hit_rate)))),
                                )
                    if str(series_pack_state) == "mmap":
                        adjusted_cost = max(0.001, float(adjusted_cost) * 0.20)
                        series_pack_mmap_hint_hits += 1
                    elif str(series_pack_state) == "pickle":
                        adjusted_cost = max(0.001, float(adjusted_cost) * 0.45)
                        series_pack_pickle_hint_hits += 1
            prepared_items.append((int(idx), str(bucket_key), float(adjusted_cost), mixed_radix_rank, float(utility_score)))
            grouped.setdefault(str(bucket_key), []).append((int(idx), float(adjusted_cost), float(utility_score)))

        mixed_radix_ready = (
            len(prepared_items) > 0
            and all((row[3] is not None) for row in prepared_items)
        )
        if bool(mixed_radix_ready):
            warm_ranges_eff = tuple(warm_ranges or ())

            def _range_rank_key(rank: int) -> tuple[int, int]:
                if not warm_ranges_eff:
                    return (0, int(rank))
                for idx_r, (lo, hi, _count) in enumerate(warm_ranges_eff):
                    if int(lo) <= int(rank) <= int(hi):
                        return (int(idx_r), int(rank))
                return (len(warm_ranges_eff), int(rank))

            sorted_items = sorted(
                prepared_items,
                key=lambda row: (*_range_rank_key(int(row[3] or 0)), -float(row[4]), int(row[0])),
            )
            total_cost = sum(float(row[2]) for row in sorted_items)
            target_cost = max(0.001, float(total_cost) / float(max(1, workers_n)))
            out_ranges: list[list[int]] = [[] for _ in range(workers_n)]
            worker_id = 0
            worker_cost = 0.0
            for pos, (idx, _bucket_key, adjusted_cost, _rank, _utility_score) in enumerate(sorted_items):
                remaining_items = int(len(sorted_items) - int(pos))
                remaining_workers = int(workers_n - int(worker_id))
                if (
                    int(worker_id) < int(workers_n - 1)
                    and float(worker_cost) >= float(target_cost)
                    and int(remaining_items) > int(remaining_workers - 1)
                ):
                    worker_id += 1
                    worker_cost = 0.0
                out_ranges[int(worker_id)].append(int(idx))
                worker_cost += float(adjusted_cost)
            return out_ranges

        locality_buckets: list[tuple[str, float, list[tuple[int, float, float]]]] = []
        for bucket_key, items in grouped.items():
            items_sorted = sorted(items, key=lambda row: (-float(row[2]), -float(row[1]), int(row[0])))
            bucket_cost = sum(float(cost) for _idx, cost, _utility_score in items_sorted)
            locality_buckets.append((str(bucket_key), float(bucket_cost), items_sorted))
        locality_buckets.sort(key=lambda row: (-float(row[1]), str(row[0])))

        loads = [0.0] * workers_n
        counts = [0] * workers_n
        out: list[list[int]] = [[] for _ in range(workers_n)]
        for _bucket_key, bucket_cost, items in locality_buckets:
            target = min(range(workers_n), key=lambda wid: (loads[wid], counts[wid], wid))
            out[target].extend(int(idx) for idx, _cost, _utility_score in items)
            loads[target] += float(bucket_cost)
            counts[target] += len(items)
        return out

    def _ordered_plan_indices_by_dimension_utility(
        *,
        stage_label: str,
        plan_all,
        bars: list | None = None,
    ) -> list[int]:
        stage_key = _stage_cache_scope(stage_label)
        total = len(plan_all)
        if total <= 1 or not stage_key:
            return list(range(total))
        utility_cfg = _cache_config("dimension_value_utility")
        utility_cells: list[tuple[str, str, str]] = []
        item_meta: list[tuple[int, str, tuple[tuple[str, str], ...], int | None]] = []
        for idx, item in enumerate(plan_all):
            cfg = _plan_item_cfg(item)
            if cfg is None:
                continue
            dim_pairs = _plan_item_dimension_values(item)
            if not dim_pairs:
                continue
            _ctx_sig, cache_key, _axis_dim_fp, _persistent_key = _run_cfg_cache_coords(
                cfg=cfg,
                bars=bars,
                update_dim_index=False,
            )
            window_sig = str(cache_key[2])
            if not window_sig:
                continue
            for dim_key, dim_value in dim_pairs:
                utility_cells.append((str(window_sig), str(dim_key), str(dim_value)))
            item_meta.append((int(idx), str(window_sig), dim_pairs, _plan_item_mixed_radix_rank(item)))
        if not item_meta:
            return list(range(total))
        utility_by_cell = _dimension_value_utility_get_many(stage_label=str(stage_key), cells=utility_cells)
        score_by_idx: dict[int, float] = {}
        rank_by_idx: dict[int, int] = {}
        for idx, window_sig, dim_pairs, mr_rank in item_meta:
            scores: list[float] = []
            for dim_key, dim_value in dim_pairs:
                row = utility_by_cell.get((str(window_sig), str(dim_key), str(dim_value)))
                score = _dimension_value_utility_score(row, cfg=utility_cfg)
                if score > 0.0:
                    scores.append(float(score))
            if scores:
                score_by_idx[int(idx)] = float(sum(float(v) for v in scores) / float(max(1, len(scores))))
            if isinstance(mr_rank, int):
                rank_by_idx[int(idx)] = int(mr_rank)
        if not score_by_idx:
            return list(range(total))
        sentinel_rank = int(10**18)
        return sorted(
            list(range(total)),
            key=lambda idx: (
                -float(score_by_idx.get(int(idx), 0.0)),
                int(rank_by_idx.get(int(idx), sentinel_rank)),
                int(idx),
            ),
        )

    def _ordered_plan_indices_by_upper_bound(
        *,
        stage_label: str,
        plan_all,
        bars: list | None = None,
    ) -> tuple[list[int], int]:
        nonlocal dimension_upper_bound_deferred
        stage_key = _stage_cache_scope(stage_label)
        total = len(plan_all)
        if total <= 1 or not stage_key:
            return list(range(total)), 0
        bound_cells: list[tuple[str, str]] = []
        item_meta: list[tuple[int, str, str, int | None]] = []
        for idx, item in enumerate(plan_all):
            cfg = _plan_item_cfg(item)
            if cfg is None:
                continue
            _ctx_sig, cache_key, axis_dim_fp, _persistent_key = _run_cfg_cache_coords(
                cfg=cfg,
                bars=bars,
                update_dim_index=False,
            )
            window_sig = str(cache_key[2])
            axis_fp = str(axis_dim_fp)
            if not window_sig or not axis_fp:
                continue
            bound_cells.append((axis_fp, window_sig))
            item_meta.append((int(idx), axis_fp, window_sig, _plan_item_mixed_radix_rank(item)))
        if not item_meta:
            return list(range(total)), 0
        bound_by_cell = _dimension_upper_bound_get_many(stage_label=str(stage_key), cells=bound_cells)
        score_by_idx: dict[int, float] = {}
        rank_by_idx: dict[int, int] = {}
        deferred = 0
        for idx, axis_fp, window_sig, mr_rank in item_meta:
            row = bound_by_cell.get((str(axis_fp), str(window_sig)))
            score = float(_dimension_upper_bound_score(row))
            score_by_idx[int(idx)] = float(score)
            if score < 0.0:
                deferred += 1
            if isinstance(mr_rank, int):
                rank_by_idx[int(idx)] = int(mr_rank)
        if not score_by_idx:
            return list(range(total)), 0
        if deferred > 0:
            dimension_upper_bound_deferred += int(deferred)
        sentinel_rank = int(10**18)
        ordered = sorted(
            list(range(total)),
            key=lambda idx: (
                -(max(0.0, float(score_by_idx.get(int(idx), 0.0)))),
                1 if float(score_by_idx.get(int(idx), 0.0)) < 0.0 else 0,
                int(rank_by_idx.get(int(idx), sentinel_rank)),
                int(idx),
            ),
        )
        return ordered, int(deferred)

    def _run_sharded_stage_worker_kernel(
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
        plan_total: int | None = None,
        plan_item_from_rank=None,
        rank_manifest_window_signature: str = "",
        rank_batch_size: int = 384,
    ) -> None:
        def _run_sharded_stage_worker_lazy_rank(
            *,
            total_ranks: int,
            item_from_rank,
            manifest_window_signature: str,
            batch_size: int,
        ) -> None:
            if not offline:
                raise SystemExit(f"{stage_label} worker mode requires --offline (avoid parallel IBKR sessions).")
            out_path_str = str(out_path_raw or "").strip()
            if not out_path_str:
                raise SystemExit(f"--{out_flag_name} is required for {stage_label} worker mode.")
            out_path = Path(out_path_str)
            worker_id, workers = _parse_worker_shard(worker_raw, workers_raw, label=str(stage_label))
            unresolved_ranges = _cartesian_rank_manifest_unresolved_ranges(
                stage_label=str(stage_label),
                window_signature=str(manifest_window_signature),
                total=int(total_ranks),
            )
            unresolved_total = sum(max(0, int(rank_hi) - int(rank_lo) + 1) for rank_lo, rank_hi in unresolved_ranges)
            dynamic_claim_mode = _run_cfg_persistent_conn() is not None
            worker_ranges: list[tuple[int, int]] = []
            local_total = 0
            if not bool(dynamic_claim_mode):
                range_buckets = _partition_rank_ranges_for_workers(
                    ranges=tuple(unresolved_ranges),
                    workers=int(workers),
                )
                if int(worker_id) >= len(range_buckets):
                    raise SystemExit(f"Invalid {stage_label} worker shard: worker={worker_id} workers={workers}.")
                worker_ranges = list(range_buckets[int(worker_id)])
                local_total = sum(max(0, int(rank_hi) - int(rank_lo) + 1) for rank_lo, rank_hi in worker_ranges)
            if unresolved_ranges:
                preview = ", ".join(
                    f"{int(rank_lo)}-{int(rank_hi)}"
                    for rank_lo, rank_hi in tuple(unresolved_ranges[:8])
                )
                if preview:
                    print(
                        f"{stage_label} unresolved rank ranges total={int(unresolved_total)}/{int(total_ranks)} {preview}",
                        flush=True,
                    )
            if worker_ranges:
                assigned_preview = ", ".join(
                    f"{int(rank_lo)}-{int(rank_hi)}"
                    for rank_lo, rank_hi in tuple(worker_ranges[:8])
                )
                if assigned_preview:
                    print(
                        f"{stage_label} worker {int(worker_id)+1}/{int(workers)} ranges {assigned_preview}",
                        flush=True,
                    )
            _planner_heartbeat_set(
                stage_label=str(stage_label),
                worker_id=int(worker_id),
                tested=0,
                cached_hits=0,
                total=(0 if bool(dynamic_claim_mode) else int(local_total)),
                eta_sec=(0.0 if int(unresolved_total) <= 0 else None),
                status=("done" if int(unresolved_total) <= 0 else "starting"),
            )
            if int(unresolved_total) <= 0:
                write_json(out_path, {"tested": 0, "kept": 0, "records": []}, sort_keys=False)
                print(
                    f"{stage_label} worker done tested=0 kept=0 out={out_path} (no unresolved ranks)",
                    flush=True,
                )
                return
            if not bool(dynamic_claim_mode) and int(local_total) <= 0:
                write_json(out_path, {"tested": 0, "kept": 0, "records": []}, sort_keys=False)
                print(
                    f"{stage_label} worker done tested=0 kept=0 out={out_path} (no shard assignment)",
                    flush=True,
                )
                return

            worker_started_at = float(pytime.perf_counter())
            tested_eval_total = 0
            cached_hits_total = 0
            processed_ranks = 0
            claimed_ranks = 0
            records: list[dict] = []
            batch_size_i = max(1, int(batch_size))
            small_total_threshold = int(max(1, int(workers) * int(batch_size_i) * 2))
            if int(total_ranks) <= int(small_total_threshold):
                claim_span_default = max(
                    4,
                    int(
                        math.ceil(
                            float(max(1, int(total_ranks)))
                            / float(max(1, int(workers) * 3))
                        )
                    ),
                )
                claim_span_default = min(int(batch_size_i), int(claim_span_default))
            else:
                claim_span_default = max(
                    int(batch_size_i),
                    min(
                        int(batch_size_i * 8),
                        int(
                            math.ceil(
                                float(max(1, int(total_ranks)))
                                / float(max(1, int(workers) * 64))
                            )
                        ),
                    ),
                )
            claim_span_i = int(claim_span_default)
            claim_cfg = _cache_config("claim_span_tuner")
            if bool(_registry_float(claim_cfg.get("enabled"), 1.0) > 0.0):
                target_claims = max(2, int(_registry_float(claim_cfg.get("target_claims_per_worker"), 24.0)))
                min_claim_span = max(1, int(_registry_float(claim_cfg.get("min_claim_span"), 32.0)))
                max_claim_span = max(int(min_claim_span), int(_registry_float(claim_cfg.get("max_claim_span"), 2048.0)))
                max_batch_multiple = max(1, int(_registry_float(claim_cfg.get("max_batch_multiple"), 8.0)))
                tuned_span = int(
                    math.ceil(
                        float(max(1, int(total_ranks)))
                        / float(max(1, int(workers) * int(target_claims)))
                    )
                )
                tuned_span = max(int(min_claim_span), min(int(max_claim_span), int(tuned_span)))
                tuned_span = min(int(tuned_span), int(max(1, int(batch_size_i)) * int(max_batch_multiple)))
                claim_span_i = max(1, int(tuned_span))
            claim_span_i = max(1, int(claim_span_i))
            if bool(dynamic_claim_mode):
                print(
                    f"{stage_label} worker {int(worker_id)+1}/{int(workers)} dynamic-claim enabled "
                    f"(claim_span={int(claim_span_i)}, batch={int(batch_size_i)})",
                    flush=True,
                )
            heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 30.0
            last_progress_ts = float(worker_started_at)

            def _emit_worker_heartbeat(*, done: bool = False, tested_override: int | None = None) -> None:
                nonlocal claimed_ranks
                elapsed = max(0.0, float(pytime.perf_counter()) - float(worker_started_at))
                processed = (
                    int(max(0, int(tested_override)))
                    if tested_override is not None
                    else int(max(0, int(processed_ranks)))
                )
                total_i = int(max(0, int(claimed_ranks))) if bool(dynamic_claim_mode) else int(max(0, int(local_total)))
                if total_i <= 0:
                    eta_f: float | None = 0.0
                else:
                    remaining = max(0, int(total_i) - int(processed))
                    rate = (float(processed) / elapsed) if elapsed > 0 else 0.0
                    eta_f = float(remaining) / float(rate) if rate > 0.0 else None
                    if done:
                        eta_f = 0.0
                _planner_heartbeat_set(
                    stage_label=str(stage_label),
                    worker_id=int(worker_id),
                    tested=int(processed),
                    cached_hits=int(cached_hits_total),
                    total=int(total_i),
                    eta_sec=eta_f,
                    status=("done" if bool(done) else "running"),
                )

            _emit_worker_heartbeat(done=False)

            def _process_rank_batch(batch_ranks: list[int]) -> None:
                nonlocal tested_eval_total, cached_hits_total, processed_ranks, records, last_progress_ts
                if not batch_ranks:
                    return
                plan_batch: list[tuple[ConfigBundle, str, dict | None]] = []
                for rank in batch_ranks:
                    item = item_from_rank(int(rank))
                    if not (isinstance(item, tuple) and len(item) >= 2):
                        raise SystemExit(f"{stage_label} rank decoder returned invalid item for rank={int(rank)}")
                    cfg = item[0] if isinstance(item[0], ConfigBundle) else None
                    note = str(item[1] or "")
                    meta_item = item[2] if len(item) >= 3 and isinstance(item[2], dict) else None
                    if cfg is None:
                        raise SystemExit(f"{stage_label} rank decoder returned invalid cfg for rank={int(rank)}")
                    if not isinstance(meta_item, dict):
                        meta_item = {}
                    if "_mr_rank" not in meta_item:
                        meta_item = dict(meta_item)
                        meta_item["_mr_rank"] = int(rank)
                    plan_batch.append((cfg, note, meta_item))
                pending_plan, cached_hits, pending_cell_map, _prefetched = _stage_partition_plan_by_cache(
                    stage_label=str(stage_label),
                    plan_all=plan_batch,
                    bars=bars,
                )
                if pending_plan:
                    ordered_indices = _ordered_plan_indices_by_dimension_utility(
                        stage_label=str(stage_label),
                        plan_all=pending_plan,
                        bars=bars,
                    )
                    if ordered_indices and ordered_indices != list(range(len(pending_plan))):
                        pending_plan = [pending_plan[int(i)] for i in ordered_indices]
                        if pending_cell_map:
                            pending_cell_map = {
                                int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                                for new_idx, old_idx in enumerate(ordered_indices)
                                if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                                and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                            }
                    bound_indices, deferred_count = _ordered_plan_indices_by_upper_bound(
                        stage_label=str(stage_label),
                        plan_all=pending_plan,
                        bars=bars,
                    )
                    if bound_indices and bound_indices != list(range(len(pending_plan))):
                        pending_plan = [pending_plan[int(i)] for i in bound_indices]
                        if pending_cell_map:
                            pending_cell_map = {
                                int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                                for new_idx, old_idx in enumerate(bound_indices)
                                if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                                and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                            }
                    if int(deferred_count) > 0:
                        print(
                            f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                            flush=True,
                        )
                tested_batch = 0
                kept_batch: list[tuple[ConfigBundle, dict, str, dict | None]] = []
                if pending_plan:
                    tested_batch, kept_batch = _run_sweep(
                        plan=(pending_plan[idx] for idx in range(len(pending_plan))),
                        bars=bars,
                        total=len(pending_plan),
                        progress_label=f"{stage_label} worker {int(worker_id)+1}/{int(workers)}",
                        report_every=0,
                        heartbeat_sec=float(heartbeat_eff),
                        record_milestones=False,
                        frontier_stage_label=str(stage_label),
                        progress_callback=lambda tested, _total, _kept, _elapsed, done: _emit_worker_heartbeat(
                            done=bool(done),
                            tested_override=int(processed_ranks) + int(tested),
                        ),
                    )
                if pending_cell_map:
                    evaluated_rows = [
                        (strategy_fp, axis_dim_fp, window_sig, "evaluated")
                        for idx, (strategy_fp, axis_dim_fp, window_sig) in pending_cell_map.items()
                        if int(idx) < len(pending_plan)
                    ]
                    if evaluated_rows:
                        _stage_cell_status_set_many(stage_label=str(stage_label), rows=evaluated_rows)
                        _cartesian_cell_manifest_set_many(
                            stage_label=str(stage_label),
                            rows=[
                                (axis_dim_fp, window_sig, strategy_fp, "evaluated")
                                for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows
                            ],
                        )
                tested_eval_total += int(tested_batch)
                cached_hits_total += int(len(cached_hits))
                processed_ranks += int(len(batch_ranks))
                for rec in _worker_records_from_kept(kept_batch):
                    records.append(rec)
                for _cell_key, cfg_cached, row_cached, note_cached, _meta_cached in cached_hits:
                    if isinstance(row_cached, dict):
                        records.append(_encode_cfg_payload(cfg_cached, note=note_cached, extra={"row": row_cached}))

                rank_status: dict[int, str] = {int(rank): "pending" for rank in batch_ranks}
                cached_manifest_cells: list[tuple[str, str, int]] = []
                for cell_key, _cfg, _row, _note, meta_item in cached_hits:
                    rank_i = None
                    if isinstance(meta_item, dict):
                        try:
                            rank_i = int(meta_item.get("_mr_rank"))
                        except (TypeError, ValueError):
                            rank_i = None
                    if rank_i is None or rank_i not in rank_status:
                        continue
                    rank_status[int(rank_i)] = "cached_hit"
                    if isinstance(cell_key, tuple) and len(cell_key) == 3:
                        axis_dim_fp = str(cell_key[1] or "").strip()
                        window_sig = str(cell_key[2] or "").strip()
                        if axis_dim_fp and window_sig:
                            cached_manifest_cells.append((axis_dim_fp, window_sig, int(rank_i)))
                if cached_manifest_cells:
                    manifest_lookup = _cartesian_cell_manifest_get_many(
                        stage_label=str(stage_label),
                        cells=[(axis_dim_fp, window_sig) for axis_dim_fp, window_sig, _rank in cached_manifest_cells],
                    )
                    for axis_dim_fp, window_sig, rank_i in cached_manifest_cells:
                        manifest_state = manifest_lookup.get((str(axis_dim_fp), str(window_sig)))
                        manifest_status = (
                            str(manifest_state[0]).strip().lower()
                            if isinstance(manifest_state, tuple) and len(manifest_state) >= 1
                            else ""
                        )
                        if manifest_status == "dominated":
                            rank_status[int(rank_i)] = "dominated"
                for item in pending_plan:
                    meta_item = item[2] if isinstance(item, tuple) and len(item) >= 3 and isinstance(item[2], dict) else None
                    if not isinstance(meta_item, dict):
                        continue
                    try:
                        rank_i = int(meta_item.get("_mr_rank"))
                    except (TypeError, ValueError):
                        continue
                    if rank_i in rank_status:
                        rank_status[int(rank_i)] = "evaluated"
                rank_rows = _compress_rank_status_rows(rank_status)
                if rank_rows:
                    _cartesian_rank_manifest_set_many(
                        stage_label=str(stage_label),
                        window_signature=str(manifest_window_signature),
                        rows=rank_rows,
                    )
                now = float(pytime.perf_counter())
                if (now - float(last_progress_ts)) >= float(max(5.0, heartbeat_eff)):
                    progress_total = int(max(0, int(claimed_ranks))) if bool(dynamic_claim_mode) else int(local_total)
                    print(
                        f"{stage_label} worker {int(worker_id)+1}/{int(workers)} "
                        f"processed={int(processed_ranks)}/{int(progress_total)} "
                        f"eval={int(tested_eval_total)} cached={int(cached_hits_total)} kept={len(records)}",
                        flush=True,
                    )
                    last_progress_ts = float(now)
                _emit_worker_heartbeat(done=False)

            if bool(dynamic_claim_mode):
                claim_count = 0
                while True:
                    claimed = _cartesian_rank_manifest_claim_next_range(
                        stage_label=str(stage_label),
                        window_signature=str(manifest_window_signature),
                        total=int(total_ranks),
                        max_span=int(claim_span_i),
                    )
                    if not (isinstance(claimed, tuple) and len(claimed) == 2):
                        break
                    claim_lo = int(claimed[0])
                    claim_hi = int(claimed[1])
                    if int(claim_hi) < int(claim_lo):
                        continue
                    claim_count += 1
                    claimed_ranks += max(0, int(claim_hi) - int(claim_lo) + 1)
                    if int(claim_count) <= 6:
                        print(
                            f"{stage_label} worker {int(worker_id)+1}/{int(workers)} claim {int(claim_lo)}-{int(claim_hi)}",
                            flush=True,
                        )
                    rank_cursor = int(claim_lo)
                    batch_span_eff = max(int(batch_size_i), int(claim_span_i))
                    while int(rank_cursor) <= int(claim_hi):
                        batch_hi = min(int(claim_hi), int(rank_cursor) + int(batch_span_eff) - 1)
                        _process_rank_batch(list(range(int(rank_cursor), int(batch_hi) + 1)))
                        rank_cursor = int(batch_hi) + 1
            else:
                for rank_lo, rank_hi in worker_ranges:
                    lo_i = int(rank_lo)
                    hi_i = int(rank_hi)
                    rank_cursor = int(lo_i)
                    while int(rank_cursor) <= int(hi_i):
                        batch_hi = min(int(hi_i), int(rank_cursor) + int(batch_size_i) - 1)
                        _process_rank_batch(list(range(int(rank_cursor), int(batch_hi) + 1)))
                        rank_cursor = int(batch_hi) + 1

            _emit_worker_heartbeat(done=True)
            tested_total = int(tested_eval_total) + int(cached_hits_total)
            write_json(out_path, {"tested": int(tested_total), "kept": len(records), "records": records}, sort_keys=False)
            print(
                f"{stage_label} worker done tested={int(tested_total)} kept={len(records)} out={out_path}",
                flush=True,
            )

        if callable(plan_item_from_rank):
            total_i = int(plan_total or 0)
            window_sig = str(rank_manifest_window_signature or "").strip()
            if total_i <= 0:
                raise SystemExit(f"{stage_label} lazy-rank worker requires positive total rank count.")
            if not window_sig:
                raise SystemExit(f"{stage_label} lazy-rank worker requires rank manifest window signature.")
            _run_sharded_stage_worker_lazy_rank(
                total_ranks=int(total_i),
                item_from_rank=plan_item_from_rank,
                manifest_window_signature=str(window_sig),
                batch_size=int(rank_batch_size),
            )
            return

        if plan_all is None:
            raise SystemExit(f"{stage_label} worker mode requires plan_all when lazy-rank mode is not active.")
        nonlocal worker_plan_cache_hits
        if not offline:
            raise SystemExit(f"{stage_label} worker mode requires --offline (avoid parallel IBKR sessions).")
        out_path_str = str(out_path_raw or "").strip()
        if not out_path_str:
            raise SystemExit(f"--{out_flag_name} is required for {stage_label} worker mode.")
        out_path = Path(out_path_str)

        worker_id, workers = _parse_worker_shard(worker_raw, workers_raw, label=str(stage_label))
        _planner_heartbeat_set(
            stage_label=str(stage_label),
            worker_id=int(worker_id),
            tested=0,
            cached_hits=0,
            total=0,
            eta_sec=None,
            status="starting",
        )
        pending_plan, cached_hits, pending_cell_map, _prefetched_total = _stage_partition_plan_by_cache(
            stage_label=str(stage_label),
            plan_all=plan_all,
            bars=bars,
        )
        if pending_plan:
            bound_ordered_indices, deferred_count = _ordered_plan_indices_by_upper_bound(
                stage_label=str(stage_label),
                plan_all=pending_plan,
                bars=bars,
            )
            if bound_ordered_indices and bound_ordered_indices != list(range(len(pending_plan))):
                pending_plan = [pending_plan[int(i)] for i in bound_ordered_indices]
                if pending_cell_map:
                    pending_cell_map = {
                        int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                        for new_idx, old_idx in enumerate(bound_ordered_indices)
                        if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                        and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                    }
            if int(deferred_count) > 0:
                print(
                    f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                    flush=True,
                )
        total = len(pending_plan)
        warm_ranges = _mixed_radix_warm_ranges(plan_all=pending_plan)
        if warm_ranges:
            preview = ", ".join(
                f"{int(lo)}-{int(hi)}:{int(count)}"
                for lo, hi, count in tuple(warm_ranges[:6])
            )
            if preview:
                print(f"{stage_label} warm unresolved rank bins {preview}", flush=True)
        cache_key = _worker_plan_cache_key(stage_label=str(stage_label), workers=int(workers), plan_all=pending_plan)
        worker_buckets = _worker_plan_cache_get(cache_key=str(cache_key))

        def _is_valid_worker_buckets(raw_buckets) -> bool:
            if not isinstance(raw_buckets, list):
                return False
            if len(raw_buckets) != int(workers):
                return False
            flat: list[int] = []
            for bucket in raw_buckets:
                if not isinstance(bucket, list):
                    return False
                for idx in bucket:
                    try:
                        i = int(idx)
                    except (TypeError, ValueError):
                        return False
                    if i < 0 or i >= int(total):
                        return False
                    flat.append(i)
            if len(flat) != int(total):
                return False
            return len(set(flat)) == int(total)

        if not _is_valid_worker_buckets(worker_buckets):
            worker_buckets = _worker_bucketed_indices(
                plan_all=pending_plan,
                workers=int(workers),
                bars=bars,
                stage_label=str(stage_label),
                warm_ranges=warm_ranges,
            )
            if _is_valid_worker_buckets(worker_buckets):
                _worker_plan_cache_set(cache_key=str(cache_key), buckets=worker_buckets)
        else:
            worker_plan_cache_hits += 1
        if int(worker_id) >= len(worker_buckets):
            raise SystemExit(f"Invalid {stage_label} worker shard: worker={worker_id} workers={workers}.")

        def _cached_owner(cell_key: tuple[str, str, str], workers_n: int) -> int:
            raw = "\x1f".join((str(cell_key[0]), str(cell_key[1]), str(cell_key[2])))
            h = hashlib.sha1(raw.encode("utf-8")).digest()
            return int.from_bytes(h[:4], byteorder="big", signed=False) % max(1, int(workers_n))

        cached_hits_local = [
            (cell_key, cfg, row, note, meta_item)
            for cell_key, cfg, row, note, meta_item in cached_hits
            if int(_cached_owner(cell_key, int(workers))) == int(worker_id)
        ]
        heartbeat_eff = float(heartbeat_sec) if float(heartbeat_sec) > 0.0 else 30.0
        worker_indices = worker_buckets[int(worker_id)]
        local_total = len(worker_indices)
        cached_hits_total = int(len(cached_hits_local))
        worker_started_at = float(pytime.perf_counter())

        def _emit_worker_heartbeat(*, tested_live: int, done: bool = False) -> None:
            elapsed = max(0.0, float(pytime.perf_counter()) - float(worker_started_at))
            tested_i = int(max(0, int(tested_live)))
            total_i = int(max(0, int(local_total)))
            if total_i <= 0:
                eta_f: float | None = 0.0
            else:
                remaining = max(0, int(total_i) - int(tested_i))
                rate = (float(tested_i) / elapsed) if elapsed > 0 else 0.0
                eta_f = float(remaining) / float(rate) if rate > 0.0 else None
                if done:
                    eta_f = 0.0
            _planner_heartbeat_set(
                stage_label=str(stage_label),
                worker_id=int(worker_id),
                tested=int(tested_i),
                cached_hits=int(cached_hits_total),
                total=int(total_i),
                eta_sec=eta_f,
                status=("done" if bool(done) else "running"),
            )

        _emit_worker_heartbeat(tested_live=0, done=False)
        shard_plan = (pending_plan[idx] for idx in worker_indices)
        tested, kept = _run_sweep(
            plan=shard_plan,
            bars=bars,
            total=local_total,
            progress_label=f"{stage_label} worker {worker_id+1}/{workers}",
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_eff),
            record_milestones=False,
            frontier_stage_label=str(stage_label),
            progress_callback=lambda tested, total, kept, elapsed, done: _emit_worker_heartbeat(
                tested_live=int(tested),
                done=bool(done),
            ),
        )
        _emit_worker_heartbeat(tested_live=int(tested), done=True)
        if pending_cell_map:
            evaluated_rows = [
                (strategy_fp, axis_dim_fp, window_sig, "evaluated")
                for idx in worker_indices
                for strategy_fp, axis_dim_fp, window_sig in (
                    (pending_cell_map.get(int(idx)) or ("", "", "")),
                )
                if strategy_fp and axis_dim_fp and window_sig
            ]
            if evaluated_rows:
                _stage_cell_status_set_many(stage_label=str(stage_label), rows=evaluated_rows)
                _cartesian_cell_manifest_set_many(
                    stage_label=str(stage_label),
                    rows=[
                        (axis_dim_fp, window_sig, strategy_fp, "evaluated")
                        for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows
                    ],
                )
        records = _worker_records_from_kept(kept)
        for _cell_key, cfg, row, note, _meta in cached_hits_local:
            if isinstance(row, dict):
                records.append(_encode_cfg_payload(cfg, note=note, extra={"row": row}))
        tested_total = int(tested) + len(cached_hits_local)
        out_payload = {"tested": int(tested_total), "kept": len(records), "records": records}
        write_json(out_path, out_payload, sort_keys=False)
        print(f"{stage_label} worker done tested={tested_total} kept={len(records)} out={out_path}", flush=True)

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
        plan_total: int | None = None,
        plan_item_from_rank=None,
        rank_manifest_window_signature: str = "",
        rank_batch_size: int = 384,
    ) -> None:
        _run_sharded_stage_worker_kernel(
            stage_label=str(stage_label),
            worker_raw=worker_raw,
            workers_raw=workers_raw,
            out_path_raw=str(out_path_raw or ""),
            out_flag_name=str(out_flag_name),
            plan_all=plan_all,
            bars=bars,
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            plan_total=(int(plan_total) if plan_total is not None else None),
            plan_item_from_rank=plan_item_from_rank,
            rank_manifest_window_signature=str(rank_manifest_window_signature or ""),
            rank_batch_size=int(rank_batch_size),
        )

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
            frontier_stage_label=str(stage_label),
        )
        return int(tested), _rows_from_kept(kept)

    def _prune_pending_plan_by_manifest(
        *,
        stage_label: str,
        pending_plan: list[tuple[ConfigBundle, str, dict | None]],
        pending_cell_map: dict[int, tuple[str, str, str]],
    ) -> tuple[list[tuple[ConfigBundle, str, dict | None]], dict[int, tuple[str, str, str]], int, dict[int, str]]:
        nonlocal run_calls_total, winner_projection_hits
        if not pending_plan or not pending_cell_map:
            return pending_plan, pending_cell_map, 0, {}

        cells: list[tuple[str, str, str]] = []
        for idx in range(len(pending_plan)):
            cell_key = pending_cell_map.get(int(idx))
            if isinstance(cell_key, tuple) and len(cell_key) == 3:
                cells.append((str(cell_key[0]), str(cell_key[1]), str(cell_key[2])))
        if not cells:
            return pending_plan, pending_cell_map, 0, {}

        prior_status = _stage_cell_status_get_many(stage_label=str(stage_label), cells=cells)
        frontier_by_dim_window = _stage_frontier_get_many(
            stage_label=str(stage_label),
            cells=[(axis_dim_fp, window_sig) for _strategy_fp, axis_dim_fp, window_sig in cells],
        )
        winner_projection_by_dim_window = _winner_projection_get_many(
            stage_label=str(stage_label),
            cells=[(axis_dim_fp, window_sig) for _strategy_fp, axis_dim_fp, window_sig in cells],
        )
        manifest_by_dim_window = _cartesian_cell_manifest_get_many(
            stage_label=str(stage_label),
            cells=[(axis_dim_fp, window_sig) for _strategy_fp, axis_dim_fp, window_sig in cells],
        )

        kept_plan: list[tuple[ConfigBundle, str, dict | None]] = []
        kept_cell_map: dict[int, tuple[str, str, str]] = {}
        status_updates: list[tuple[str, str, str, str]] = []
        manifest_updates: list[tuple[str, str, str, str]] = []
        pruned_rank_status: dict[int, str] = {}
        skipped = 0

        for idx, item in enumerate(pending_plan):
            cell_key = pending_cell_map.get(int(idx))
            if not (isinstance(cell_key, tuple) and len(cell_key) == 3):
                kept_plan.append(item)
                continue

            strategy_fp = str(cell_key[0])
            axis_dim_fp = str(cell_key[1])
            window_sig = str(cell_key[2])
            dim_window_key = (axis_dim_fp, window_sig)
            prune_manifest_status = ""
            prune_strategy_fp = strategy_fp

            prev_cell_status = str(prior_status.get((strategy_fp, axis_dim_fp, window_sig)) or "").strip().lower()
            if prev_cell_status in ("cached_hit", "evaluated"):
                prune_manifest_status = "cached_hit"
            else:
                frontier_row = frontier_by_dim_window.get(dim_window_key)
                if _stage_frontier_is_dominated(frontier_row):
                    prune_manifest_status = "dominated"
                else:
                    winner_projection_row = winner_projection_by_dim_window.get(dim_window_key)
                    if _winner_projection_is_dominated(winner_projection_row):
                        winner_projection_hits += 1
                        prune_manifest_status = "dominated"
                    else:
                        manifest_state = manifest_by_dim_window.get(dim_window_key)
                        manifest_status = (
                            str(manifest_state[0]).strip().lower()
                            if isinstance(manifest_state, tuple) and len(manifest_state) >= 1
                            else ""
                        )
                        manifest_strategy_fp = (
                            str(manifest_state[1]).strip()
                            if isinstance(manifest_state, tuple) and len(manifest_state) >= 2
                            else ""
                        )
                        if manifest_status in ("dominated", "cached_hit", "evaluated"):
                            prune_manifest_status = "dominated" if manifest_status == "dominated" else "cached_hit"
                            if manifest_strategy_fp:
                                prune_strategy_fp = str(manifest_strategy_fp)

            if prune_manifest_status:
                skipped += 1
                run_calls_total += 1
                _axis_progress_record(kept=False)
                status_updates.append((strategy_fp, axis_dim_fp, window_sig, "cached_hit"))
                manifest_updates.append((axis_dim_fp, window_sig, prune_strategy_fp, prune_manifest_status))
                rank_i = _plan_item_stage_rank(item)
                if isinstance(rank_i, int):
                    rank_status = "dominated" if str(prune_manifest_status) == "dominated" else "cached_hit"
                    prev_status = str(pruned_rank_status.get(int(rank_i), "")).strip().lower()
                    priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}
                    if priority.get(str(rank_status), -1) >= priority.get(str(prev_status), -1):
                        pruned_rank_status[int(rank_i)] = str(rank_status)
                continue

            new_idx = len(kept_plan)
            kept_plan.append(item)
            kept_cell_map[new_idx] = (strategy_fp, axis_dim_fp, window_sig)

        if status_updates:
            _stage_cell_status_set_many(stage_label=str(stage_label), rows=status_updates)

        if manifest_updates:
            manifest_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}
            filtered_updates: list[tuple[str, str, str, str]] = []
            for axis_dim_fp, window_sig, strategy_fp, status in manifest_updates:
                prev_state = manifest_by_dim_window.get((str(axis_dim_fp), str(window_sig)))
                prev_status = str(prev_state[0]).strip().lower() if isinstance(prev_state, tuple) else ""
                if prev_status == str(status):
                    continue
                if manifest_priority.get(str(status), -1) < manifest_priority.get(str(prev_status), -1):
                    continue
                filtered_updates.append((axis_dim_fp, window_sig, strategy_fp, status))
            if filtered_updates:
                _cartesian_cell_manifest_set_many(stage_label=str(stage_label), rows=filtered_updates)

        if skipped > 0:
            print(
                f"{stage_label} pre-shard prune skipped={int(skipped)} remaining={len(kept_plan)}",
                flush=True,
            )
        return kept_plan, kept_cell_map, int(skipped), dict(pruned_rank_status)

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
        parallel_payload_supports_plan: bool = False,
        parallel_default_note: str = "",
        parallel_dedupe_by_milestone_key: bool = True,
        record_milestones: bool = True,
    ) -> int:
        nonlocal run_calls_total

        def _on_row_local(cfg: ConfigBundle, row: dict, note: str) -> None:
            if bool(record_milestones):
                _record_milestone(cfg, row, note)
            on_row(cfg, row, note)

        claim_first_serial_worker = bool(
            int(jobs_req) <= 1
            and bool(offline)
            and callable(parallel_payloads_builder)
            and _claim_first_serial_force_worker_enabled()
            and _claim_first_stage_enabled(stage_label=str(stage_label), total=int(total))
        )
        if bool(claim_first_serial_worker):
            print(
                f"{stage_label} claim-first planner: using worker-claim path for serial run "
                f"(total={int(total)})",
                flush=True,
            )

        needs_serial_plan = (
            (int(jobs_req) <= 1 and not bool(claim_first_serial_worker))
            or bool(parallel_payload_supports_plan)
            or not callable(parallel_payloads_builder)
        )
        serial_plan_eff_raw = (
            (serial_plan_builder() if callable(serial_plan_builder) else serial_plan)
            if bool(needs_serial_plan)
            else None
        )
        serial_plan_eff = _plan_items_with_stage_ranks(serial_plan_eff_raw) if bool(needs_serial_plan) else []

        prefilter_here = bool(
            (int(jobs_req) <= 1 and not bool(claim_first_serial_worker))
            or bool(parallel_payload_supports_plan)
        )
        prefetched_tested = 0
        pending_plan = serial_plan_eff
        pending_cell_map: dict[int, tuple[str, str, str]] = {}
        stage_manifest_plan_signature = ""
        stage_manifest_window_signature = ""
        stage_rank_status_updates: dict[int, str] = {}
        stage_rank_status_priority = {"pending": 0, "cached_hit": 1, "evaluated": 2, "dominated": 3}

        def _mark_stage_rank_status(rank_raw, status_raw: str) -> None:
            try:
                rank_i = int(rank_raw)
            except (TypeError, ValueError):
                return
            if rank_i < 0:
                return
            status = str(status_raw or "").strip().lower()
            if status not in _STAGE_RANK_STATUS_VALUES:
                return
            prev = str(stage_rank_status_updates.get(int(rank_i), "")).strip().lower()
            if stage_rank_status_priority.get(status, -1) >= stage_rank_status_priority.get(prev, -1):
                stage_rank_status_updates[int(rank_i)] = str(status)

        def _flush_stage_rank_status_manifest() -> None:
            if not stage_manifest_plan_signature or not stage_manifest_window_signature:
                return
            rank_rows = _compress_rank_status_rows(stage_rank_status_updates)
            if not rank_rows:
                return
            _stage_rank_manifest_set_many(
                stage_label=str(stage_label),
                plan_signature=str(stage_manifest_plan_signature),
                window_signature=str(stage_manifest_window_signature),
                rows=list(rank_rows),
            )

        if prefilter_here and serial_plan_eff:
            stage_manifest_plan_signature = _stage_rank_manifest_plan_signature(
                stage_label=str(stage_label),
                plan_all=serial_plan_eff,
            )
            stage_manifest_window_signature = _stage_rank_manifest_window_signature(bars=bars)
            unresolved_ranges = _stage_rank_manifest_unresolved_ranges(
                stage_label=str(stage_label),
                plan_signature=str(stage_manifest_plan_signature),
                window_signature=str(stage_manifest_window_signature),
                total=len(serial_plan_eff),
            )
            if unresolved_ranges and not (
                len(unresolved_ranges) == 1
                and int(unresolved_ranges[0][0]) == 0
                and int(unresolved_ranges[0][1]) >= int(len(serial_plan_eff) - 1)
            ):
                keep_indices: list[int] = []
                for rank_lo, rank_hi in unresolved_ranges:
                    lo_i = max(0, int(rank_lo))
                    hi_i = min(int(len(serial_plan_eff) - 1), int(rank_hi))
                    if hi_i < lo_i:
                        continue
                    keep_indices.extend(range(int(lo_i), int(hi_i) + 1))
                pending_plan = [serial_plan_eff[int(i)] for i in keep_indices]
                skipped_by_rank_manifest = max(0, int(len(serial_plan_eff) - len(pending_plan)))
                if skipped_by_rank_manifest > 0:
                    prefetched_tested += int(skipped_by_rank_manifest)
                    run_calls_total += int(skipped_by_rank_manifest)
                    print(
                        f"{stage_label} stage-rank manifest skipped={int(skipped_by_rank_manifest)} "
                        f"remaining={len(pending_plan)}",
                        flush=True,
                    )
            else:
                pending_plan = list(serial_plan_eff)
            pending_plan, cached_hits, pending_cell_map, prefetched_cache_tested = _stage_partition_plan_by_cache(
                stage_label=str(stage_label),
                plan_all=pending_plan,
                bars=bars,
            )
            prefetched_tested += int(prefetched_cache_tested)
            for _cell_key, cfg, row, note, meta_item in cached_hits:
                if isinstance(meta_item, dict):
                    _mark_stage_rank_status(meta_item.get("_stage_rank"), "cached_hit")
                if isinstance(row, dict):
                    _on_row_local(cfg, row, note)
            pending_plan, pending_cell_map, pruned_here, pruned_rank_status = _prune_pending_plan_by_manifest(
                stage_label=str(stage_label),
                pending_plan=list(pending_plan),
                pending_cell_map=dict(pending_cell_map),
            )
            for rank_i, status in pruned_rank_status.items():
                _mark_stage_rank_status(rank_i, str(status))
            prefetched_tested += int(pruned_here)
            if pending_plan:
                ordered_indices = _ordered_plan_indices_by_dimension_utility(
                    stage_label=str(stage_label),
                    plan_all=pending_plan,
                    bars=bars,
                )
                if ordered_indices and ordered_indices != list(range(len(pending_plan))):
                    pending_plan = [pending_plan[int(i)] for i in ordered_indices]
                    if pending_cell_map:
                        pending_cell_map = {
                            int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                            for new_idx, old_idx in enumerate(ordered_indices)
                            if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                            and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                        }
            if pending_plan:
                bound_ordered_indices, deferred_count = _ordered_plan_indices_by_upper_bound(
                    stage_label=str(stage_label),
                    plan_all=pending_plan,
                    bars=bars,
                )
                if bound_ordered_indices and bound_ordered_indices != list(range(len(pending_plan))):
                    pending_plan = [pending_plan[int(i)] for i in bound_ordered_indices]
                    if pending_cell_map:
                        pending_cell_map = {
                            int(new_idx): pending_cell_map.get(int(old_idx), ("", "", ""))
                            for new_idx, old_idx in enumerate(bound_ordered_indices)
                            if isinstance(pending_cell_map.get(int(old_idx)), tuple)
                            and len(pending_cell_map.get(int(old_idx)) or ()) == 3
                        }
                if int(deferred_count) > 0:
                    print(
                        f"{stage_label} upper-bound prepass deferred={int(deferred_count)}",
                        flush=True,
                    )

        parallel_total = len(pending_plan) if bool(parallel_payload_supports_plan) else (
            len(serial_plan_eff) if bool(needs_serial_plan) else int(total)
        )
        if (int(jobs_req) > 1 or bool(claim_first_serial_worker)) and int(parallel_total) > 0 and callable(parallel_payloads_builder):
            if bool(parallel_payload_supports_plan):
                payloads = parallel_payloads_builder(pending_plan)
            else:
                payloads = parallel_payloads_builder()
            tested_parallel = _collect_stage_rows_from_payloads(
                payloads=payloads,
                default_note=str(parallel_default_note or stage_label),
                on_row=_on_row_local,
                dedupe_by_milestone_key=bool(parallel_dedupe_by_milestone_key),
            )
            for item in pending_plan:
                _mark_stage_rank_status(_plan_item_stage_rank(item), "evaluated")
            _flush_stage_rank_status_manifest()
            run_calls_total += int(tested_parallel)
            return int(prefetched_tested) + int(tested_parallel)

        tested, serial_rows = _run_stage_serial(
            stage_label=str(stage_label),
            plan=pending_plan,
            bars=bars,
            total=len(pending_plan),
            report_every=int(report_every),
            heartbeat_sec=float(heartbeat_sec),
            record_milestones=bool(record_milestones),
        )
        if pending_cell_map:
            evaluated_rows = [
                (strategy_fp, axis_dim_fp, window_sig, "evaluated")
                for idx, (strategy_fp, axis_dim_fp, window_sig) in pending_cell_map.items()
                if int(idx) < len(pending_plan)
            ]
            if evaluated_rows:
                _stage_cell_status_set_many(stage_label=str(stage_label), rows=evaluated_rows)
                _cartesian_cell_manifest_set_many(
                    stage_label=str(stage_label),
                    rows=[
                        (axis_dim_fp, window_sig, strategy_fp, "evaluated")
                        for strategy_fp, axis_dim_fp, window_sig, _status in evaluated_rows
                    ],
                )
        for item in pending_plan:
            _mark_stage_rank_status(_plan_item_stage_rank(item), "evaluated")
        _flush_stage_rank_status_manifest()
        for cfg, row, note in serial_rows:
            on_row(cfg, row, note)
        return int(prefetched_tested) + int(tested)

    def _stage_parallel_base_cli(*, flags_with_values: tuple[str, ...]) -> list[str]:
        return _strip_flags(
            list(sys.argv[1:]),
            flags=("--write-milestones", "--merge-milestones"),
            flags_with_values=("--axis", "--jobs", "--milestones-out", *flags_with_values),
        )

    def _compact_parallel_payload_cfg_refs(payload: dict) -> dict:
        catalog: dict[str, dict[str, object]] = {}

        def _compact(node):
            if isinstance(node, list):
                return [_compact(item) for item in node]
            if not isinstance(node, dict):
                return node
            strategy_payload = node.get("strategy")
            filters_payload = node.get("filters")
            if isinstance(strategy_payload, dict):
                filters_payload_norm = dict(filters_payload) if isinstance(filters_payload, dict) else None
                cfg_ref = _strategy_fingerprint(
                    strategy_payload,
                    filters=filters_payload_norm,
                    signal_bar_size=str(signal_bar_size),
                    signal_use_rth=bool(use_rth),
                )
                if cfg_ref not in catalog:
                    cfg_obj = _cfg_from_strategy_filters_payload(strategy_payload, filters_payload_norm)
                    est_cost = 1.0
                    axis_dim_fp: str | None = None
                    if cfg_obj is not None:
                        est_cost = float(_cfg_eval_cost_hint(cfg_obj))
                        axis_dim_fp = _axis_dimension_fingerprint(cfg_obj)
                    catalog[cfg_ref] = {
                        "cfg_ref": str(cfg_ref),
                        "strategy": dict(strategy_payload),
                        "filters": dict(filters_payload_norm) if isinstance(filters_payload_norm, dict) else None,
                        "est_cost": float(est_cost),
                        "axis_dim_fp": axis_dim_fp,
                    }
                out_item = {k: _compact(v) for k, v in node.items() if k not in ("strategy", "filters", "cfg_ref")}
                out_item["cfg_ref"] = str(cfg_ref)
                return out_item
            return {k: _compact(v) for k, v in node.items()}

        compact_payload = _compact(dict(payload))
        if catalog:
            compact_payload["_cfg_catalog"] = list(catalog.values())
        return compact_payload

    def _worker_name_to_worker_id(worker_name: str) -> int | None:
        raw = str(worker_name or "").strip()
        if not raw:
            return None
        token = raw.rsplit(":", 1)[-1]
        try:
            out = int(token)
        except (TypeError, ValueError):
            return None
        if out < 0:
            return None
        return int(out)

    def _planner_parallel_status_probe(
        *,
        stage_label: str,
        stage_total: int,
    ):
        nonlocal planner_heartbeat_stale_candidates
        cfg = _cache_config("planner_heartbeat")
        stale_after_sec = max(30.0, float(_registry_float(cfg.get("stale_after_sec"), 180.0)))
        bootstrap_grace_sec = max(30.0, float(_registry_float(cfg.get("bootstrap_grace_sec"), stale_after_sec)))
        stage_started_at = float(pytime.time())

        def _probe(running_workers, pending_count: int) -> dict[str, object]:
            worker_ids: list[int] = []
            worker_id_by_name: dict[str, int] = {}
            elapsed_by_name: dict[str, float] = {}
            for item in list(running_workers or ()):
                if not (isinstance(item, tuple) and len(item) >= 2):
                    continue
                worker_name = str(item[0] or "").strip()
                if not worker_name:
                    continue
                try:
                    elapsed_sec = float(item[1] or 0.0)
                except (TypeError, ValueError):
                    elapsed_sec = 0.0
                worker_id = _worker_name_to_worker_id(worker_name)
                if worker_id is None:
                    continue
                worker_ids.append(int(worker_id))
                worker_id_by_name[str(worker_name)] = int(worker_id)
                elapsed_by_name[str(worker_name)] = float(max(0.0, elapsed_sec))

            hb_rows = _planner_heartbeat_get_many(stage_label=str(stage_label), worker_ids=worker_ids)
            tested_sum = 0
            cached_sum = 0
            eta_vals: list[float] = []
            stale_names: list[str] = []
            now_ts = float(pytime.time())
            for worker_name, worker_id in worker_id_by_name.items():
                row = hb_rows.get(int(worker_id))
                if isinstance(row, dict):
                    tested_sum += int(row.get("tested") or 0)
                    cached_sum += int(row.get("cached_hits") or 0)
                    eta_raw = row.get("eta_sec")
                    if eta_raw is not None:
                        try:
                            eta_f = float(eta_raw)
                        except (TypeError, ValueError):
                            eta_f = -1.0
                        if eta_f >= 0.0:
                            eta_vals.append(float(eta_f))
                    status_s = str(row.get("status") or "").strip().lower()
                    try:
                        last_seen = float(row.get("last_seen") or 0.0)
                    except (TypeError, ValueError):
                        last_seen = 0.0
                    age = max(0.0, float(now_ts - float(last_seen)))
                    if status_s != "done" and age >= float(stale_after_sec):
                        stale_names.append(str(worker_name))
                    continue

                elapsed_sec = float(elapsed_by_name.get(str(worker_name), 0.0))
                if elapsed_sec >= float(bootstrap_grace_sec) or (now_ts - stage_started_at) >= float(bootstrap_grace_sec):
                    stale_names.append(str(worker_name))

            if stale_names:
                planner_heartbeat_stale_candidates += int(len(stale_names))
            total_eff = max(0, int(stage_total))
            eta_max = max(eta_vals) if eta_vals else 0.0
            line = (
                f"{stage_label} planner heartbeat running={len(worker_id_by_name)} "
                f"pending={int(pending_count)} hb_rows={len(hb_rows)} "
                f"tested={int(tested_sum)}/{int(total_eff)} cached_hits={int(cached_sum)} "
                f"eta~{float(eta_max)/60.0:0.1f}m stale={len(stale_names)}"
            )
            return {"line": line, "stale": tuple(stale_names)}

        return _probe

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
        planner_stage_label: str | None = None,
        prefetched_tested_if_empty: int = 0,
    ) -> dict[int, dict]:
        base_cli = _stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        planner_stage_key = str(planner_stage_label or stage_label).strip().lower()
        planner_cfg = _cache_config("planner_heartbeat")
        monitor_interval_sec = max(5.0, float(_registry_float(planner_cfg.get("monitor_interval_sec"), 30.0)))
        max_stale_retries = max(0, int(_registry_float(planner_cfg.get("max_stale_retries"), 1.0)))
        probe = _planner_parallel_status_probe(
            stage_label=str(planner_stage_key),
            stage_total=int(total),
        )
        _planner_heartbeat_clear_stage(stage_label=str(planner_stage_key))
        if int(total) <= 0:
            prefetched_i = max(0, int(prefetched_tested_if_empty))
            print(
                f"{stage_label} parallel: unresolved=0 skip worker launch"
                + (f" prefetched={int(prefetched_i)}" if int(prefetched_i) > 0 else ""),
                flush=True,
            )
            if int(prefetched_i) <= 0:
                return {}
            return {0: {"tested": int(prefetched_i), "kept": 0, "records": []}}
        default_jobs_i = int(_default_jobs())
        jobs_tuned = _tuned_parallel_jobs(
            stage_label=str(stage_label),
            jobs_requested=int(jobs),
            total=int(total),
            default_jobs=int(default_jobs_i),
        )
        if int(jobs_tuned) != int(max(1, int(jobs))):
            print(
                f"{stage_label} jobs tuner: requested={int(max(1, int(jobs)))} "
                f"tuned={int(jobs_tuned)} total={int(total)}",
                flush=True,
            )
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmpdir:
            payload_path = Path(tmpdir) / str(payload_filename)
            payload_compact = _compact_parallel_payload_cfg_refs(payload)
            write_json(payload_path, payload_compact, sort_keys=False)
            _jobs_eff, payloads = _run_parallel_stage_kernel(
                stage_label=str(stage_label),
                jobs=int(jobs_tuned),
                total=int(total),
                default_jobs=int(default_jobs_i),
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
                status_heartbeat_sec=float(monitor_interval_sec),
                worker_status_probe=probe,
                max_stale_retries=int(max_stale_retries),
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
        planner_stage_label: str | None = None,
        prefetched_tested_if_empty: int = 0,
    ) -> dict[int, dict]:
        base_cli = _stage_parallel_base_cli(flags_with_values=strip_flags_with_values)
        planner_stage_key = str(planner_stage_label or stage_label).strip().lower()
        planner_cfg = _cache_config("planner_heartbeat")
        monitor_interval_sec = max(5.0, float(_registry_float(planner_cfg.get("monitor_interval_sec"), 30.0)))
        max_stale_retries = max(0, int(_registry_float(planner_cfg.get("max_stale_retries"), 1.0)))
        probe = _planner_parallel_status_probe(
            stage_label=str(planner_stage_key),
            stage_total=int(total),
        )
        _planner_heartbeat_clear_stage(stage_label=str(planner_stage_key))
        if int(total) <= 0:
            prefetched_i = max(0, int(prefetched_tested_if_empty))
            print(
                f"{stage_label} parallel: unresolved=0 skip worker launch"
                + (f" prefetched={int(prefetched_i)}" if int(prefetched_i) > 0 else ""),
                flush=True,
            )
            if int(prefetched_i) <= 0:
                return {}
            return {0: {"tested": int(prefetched_i), "kept": 0, "records": []}}
        default_jobs_i = int(_default_jobs())
        jobs_tuned = _tuned_parallel_jobs(
            stage_label=str(stage_label),
            jobs_requested=int(jobs),
            total=int(total),
            default_jobs=int(default_jobs_i),
        )
        if int(jobs_tuned) != int(max(1, int(jobs))):
            print(
                f"{stage_label} jobs tuner: requested={int(max(1, int(jobs)))} "
                f"tuned={int(jobs_tuned)} total={int(total)}",
                flush=True,
            )
        _jobs_eff, payloads = _run_parallel_stage_kernel(
            stage_label=str(stage_label),
            jobs=int(jobs_tuned),
            total=int(total),
            default_jobs=int(default_jobs_i),
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
            status_heartbeat_sec=float(monitor_interval_sec),
            worker_status_probe=probe,
            max_stale_retries=int(max_stale_retries),
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
        worker_stage_mode = bool(
            args.cfg_worker is not None
            or args.combo_full_cartesian_worker is not None
        )
        if bool(worker_stage_mode):
            print(
                "worker-stage mode: axis-level progress disabled; using worker/stage heartbeats.",
                flush=True,
            )

        def _run_axis_callable(axis_name: str, fn, *, timed_local: bool) -> None:
            before_calls = int(run_calls_total)
            t0 = pytime.perf_counter()
            total_hint = _axis_total_hint(str(axis_name))
            total_hint_s = str(total_hint) if total_hint is not None else "?"
            axis_watchdog_stop = threading.Event()
            axis_watchdog_thread: threading.Thread | None = None
            axis_watchdog_sec = 30.0
            if bool(timed_local):
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            else:
                print(f"START {axis_name} total={total_hint_s}", flush=True)
            axis_progress_enabled = not bool(worker_stage_mode)

            def _axis_watchdog() -> None:
                cadence = max(5.0, float(axis_watchdog_sec))
                while not axis_watchdog_stop.wait(cadence):
                    now = float(pytime.perf_counter())
                    if bool(axis_progress_enabled):
                        last_report = float(axis_progress_state.get("last_report") or 0.0)
                        # If inner axis progress already reported recently, suppress watchdog noise.
                        if float(last_report) > 0.0 and (now - float(last_report)) < float(cadence * 0.8):
                            continue
                    tested_live = max(0, int(run_calls_total) - int(before_calls))
                    if bool(axis_progress_enabled):
                        kept_live = int(axis_progress_state.get("kept") or 0)
                        total_live = (
                            int(axis_progress_state.get("total"))
                            if isinstance(axis_progress_state.get("total"), int)
                            else (int(total_hint) if isinstance(total_hint, int) else None)
                        )
                    else:
                        kept_live = 0
                        total_live = int(total_hint) if isinstance(total_hint, int) else None
                    line = _progress_line(
                        label=f"{axis_name} watchdog",
                        tested=int(tested_live),
                        total=total_live,
                        kept=int(kept_live),
                        started_at=float(t0),
                        rate_unit="cfg/s",
                    )
                    line += " heartbeat=axis"
                    print(line, flush=True)

            axis_watchdog_thread = threading.Thread(target=_axis_watchdog, daemon=True)
            axis_watchdog_thread.start()
            if bool(axis_progress_enabled):
                _axis_progress_begin(axis_name=str(axis_name))
            try:
                fn()
            finally:
                axis_watchdog_stop.set()
                if axis_watchdog_thread is not None:
                    try:
                        axis_watchdog_thread.join(timeout=1.0)
                    except Exception:
                        pass
                if bool(axis_progress_enabled):
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
                filters_obj = _codec_filters_from_payload(item.get("filters"))
                strategy_obj = _codec_strategy_from_payload(item.get("strategy") or {}, filters=filters_obj)
            except Exception:
                continue
            cfg_seed = _codec_make_bundle(
                strategy=strategy_obj,
                start=start,
                end=end,
                bar_size=signal_bar_size,
                use_rth=use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )
            yield seed_i, item, cfg_seed, str(item.get("group_name") or f"seed#{seed_i:02d}")

    def _iter_cfg_stage_plan(
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
        if args.cfg_stage:
            payload_path = Path(str(args.cfg_stage))
            try:
                payload_raw = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid cfg_stage payload JSON: {payload_path}") from exc
            if not isinstance(payload_raw, dict):
                raise SystemExit(f"Invalid cfg_stage payload: expected object ({payload_path})")
            payload_axis = str(payload_raw.get("axis_tag") or "").strip().lower()
            if payload_axis and payload_axis != str(axis_tag).strip().lower():
                raise SystemExit(
                    f"cfg_pairs worker payload axis mismatch: expected {axis_tag} got {payload_axis} ({payload_path})"
                )
            cfg_records_raw = payload_raw.get("cfgs")
            if not isinstance(cfg_records_raw, list):
                raise SystemExit(f"cfg_stage payload missing 'cfgs' list: {payload_path}")
            cfg_records = [dict(rec) for rec in cfg_records_raw if isinstance(rec, dict)]
            cfg_catalog = _cfg_catalog_from_payload(payload_raw)
            rank_window_signature = _cfg_stage_payload_signature(
                axis_tag=str(payload_axis or axis_tag),
                cfg_records=list(cfg_records),
                cfg_catalog=dict(cfg_catalog),
            )

            def _cfg_item_from_rank(rank: int) -> tuple[ConfigBundle, str, dict]:
                rank_i = int(rank)
                if rank_i < 0 or rank_i >= len(cfg_records):
                    raise SystemExit(f"cfg_stage rank out of bounds: rank={rank_i} total={len(cfg_records)}")
                decoded = _decode_cfg_payload(
                    cfg_records[rank_i],
                    note_key="note",
                    cfg_catalog=cfg_catalog,
                )
                if decoded is None:
                    raise SystemExit(f"cfg_stage rank decode failed: rank={rank_i}")
                cfg_obj, note = decoded
                return cfg_obj, str(note), {"_mr_rank": int(rank_i)}

            _run_sharded_stage_worker(
                stage_label=str(axis_tag),
                worker_raw=args.cfg_worker,
                workers_raw=args.cfg_workers,
                out_path_raw=str(args.cfg_out or ""),
                out_flag_name="cfg-out",
                plan_all=None,
                bars=_bars_cached(signal_bar_size),
                report_every=max(1, int(report_every)),
                heartbeat_sec=float(heartbeat_sec),
                plan_total=len(cfg_records),
                plan_item_from_rank=_cfg_item_from_rank,
                rank_manifest_window_signature=str(rank_window_signature),
                rank_batch_size=256,
            )
            return -1

        bars_stage = _bars_cached(signal_bar_size)
        plan_all: list[tuple[ConfigBundle, str, None]] = [(cfg, str(note), None) for cfg, note in cfg_pairs]

        def _cfg_pairs_parallel_payloads(pending_plan_items) -> dict[int, dict]:
            pending_cfgs = list(pending_plan_items or ())
            return _run_parallel_stage_with_payload(
                axis_name=str(axis_tag),
                stage_label=str(axis_tag),
                total=len(pending_cfgs),
                jobs=int(jobs),
                payload={
                    "axis_tag": str(axis_tag),
                    "cfgs": [_encode_cfg_payload(cfg, note=note) for cfg, note, _meta in pending_cfgs],
                },
                payload_filename="cfg_pairs_payload.json",
                temp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_",
                worker_tmp_prefix=f"tradebot_{str(axis_tag)}_cfgpairs_worker_",
                worker_tag=f"cfgpairs:{str(axis_tag)}",
                out_prefix="cfg_pairs_out",
                stage_flag="--cfg-stage",
                worker_flag="--cfg-worker",
                workers_flag="--cfg-workers",
                out_flag="--cfg-out",
                strip_flags_with_values=(
                    "--cfg-stage",
                    "--cfg-worker",
                    "--cfg-workers",
                    "--cfg-out",
                    "--combo-full-cartesian-run-min-trades",
                ),
                run_min_trades_flag="--combo-full-cartesian-run-min-trades",
                run_min_trades=int(run_min_trades),
                capture_error=f"Failed to capture {axis_tag} worker stdout.",
                failure_label=f"{axis_tag} worker",
                missing_label=str(axis_tag),
                invalid_label=str(axis_tag),
            )

        tested_total = _run_stage_cfg_rows(
            stage_label=str(axis_tag),
            total=len(plan_all),
            jobs_req=int(jobs),
            bars=bars_stage,
            report_every=max(1, int(report_every)),
            heartbeat_sec=float(heartbeat_sec),
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan=plan_all,
            parallel_payloads_builder=_cfg_pairs_parallel_payloads,
            parallel_payload_supports_plan=True,
            parallel_default_note=str(axis_tag),
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        return int(tested_total)

    def _run_cfg_stage_grid(
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
            _iter_cfg_stage_plan(
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
            champion_like = base_name in ("champion", "champion_pnl")
            sizing_mode_eff = (
                str(sizing_mode)
                if (not champion_like) or sizing_mode_arg_explicit
                else str(getattr(cfg.strategy, "spot_sizing_mode", sizing_mode) or sizing_mode)
            )
            spot_notional_pct_eff = (
                float(spot_notional_pct)
                if (not champion_like) or spot_notional_pct_arg_explicit
                else float(getattr(cfg.strategy, "spot_notional_pct", spot_notional_pct) or 0.0)
            )
            spot_risk_pct_eff = (
                float(spot_risk_pct)
                if (not champion_like) or spot_risk_pct_arg_explicit
                else float(getattr(cfg.strategy, "spot_risk_pct", spot_risk_pct) or 0.0)
            )
            spot_max_notional_pct_eff = (
                float(spot_max_notional_pct)
                if (not champion_like) or spot_max_notional_pct_arg_explicit
                else float(getattr(cfg.strategy, "spot_max_notional_pct", spot_max_notional_pct) or 1.0)
            )
            spot_min_qty_eff = (
                int(spot_min_qty)
                if (not champion_like) or spot_min_qty_arg_explicit
                else int(getattr(cfg.strategy, "spot_min_qty", spot_min_qty) or 1)
            )
            spot_max_qty_eff = (
                int(spot_max_qty)
                if (not champion_like) or spot_max_qty_arg_explicit
                else int(getattr(cfg.strategy, "spot_max_qty", spot_max_qty) or 0)
            )
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    spot_entry_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
                    spot_flip_exit_fill_mode=SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
                    spot_intrabar_exits=True,
                    spot_spread=float(spot_spread),
                    spot_commission_per_share=float(spot_commission),
                    spot_commission_min=float(spot_commission_min),
                    spot_slippage_per_share=float(spot_slippage),
                    spot_mark_to_market="liquidation",
                    spot_drawdown_mode="intrabar",
                    spot_sizing_mode=str(sizing_mode_eff),
                    spot_notional_pct=float(spot_notional_pct_eff),
                    spot_risk_pct=float(spot_risk_pct_eff),
                    spot_max_notional_pct=float(spot_max_notional_pct_eff),
                    spot_min_qty=int(spot_min_qty_eff),
                    spot_max_qty=int(spot_max_qty_eff),
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

    def _sweep_ptsl() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        base_cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_strat = base_cfg.strategy
        base_mode = str(getattr(args, "base", "") or "").strip().lower()
        seeded_local = bool(str(getattr(args, "seed_milestones", "") or "").strip()) or base_mode in (
            "champion",
            "champion_pnl",
        )
        # Seeded runs should mutate around the current champion neighborhood (tight, high-value).
        # Unseeded runs keep the broad legacy PT/SL surface.
        if seeded_local:
            base_pt_raw = getattr(base_strat, "spot_profit_target_pct", None)
            base_pt = float(base_pt_raw) if base_pt_raw is not None and float(base_pt_raw) > 0 else None
            base_sl_raw = getattr(base_strat, "spot_stop_loss_pct", None)
            base_sl = float(base_sl_raw) if base_sl_raw is not None and float(base_sl_raw) > 0 else 0.01
            base_only_profit = bool(getattr(base_strat, "flip_exit_only_if_profit", False))
            base_close_eod = bool(getattr(base_strat, "spot_close_eod", False))
            base_hold = max(0, int(getattr(base_strat, "flip_exit_min_hold_bars", 0) or 0))
            base_gate = str(getattr(base_strat, "flip_exit_gate_mode", "regime_or_permission") or "regime_or_permission")
            base_flip_mode = str(getattr(base_strat, "flip_exit_mode", "entry") or "entry")

            pt_vals: list[float | None]
            if base_pt is None:
                pt_vals = [None, 0.0015, 0.0025]
            else:
                pt_vals = sorted(
                    {
                        round(max(0.0005, min(0.05, float(base_pt) * 0.8)), 6),
                        round(max(0.0005, min(0.05, float(base_pt))), 6),
                        round(max(0.0005, min(0.05, float(base_pt) * 1.2)), 6),
                    }
                )
            sl_vals = sorted(
                {
                    round(max(0.001, min(0.08, float(base_sl) * 0.8)), 6),
                    round(max(0.001, min(0.08, float(base_sl))), 6),
                    round(max(0.001, min(0.08, float(base_sl) * 1.2)), 6),
                }
            )
            only_profit_vals = list(dict.fromkeys([base_only_profit, not base_only_profit]))
            close_eod_vals = list(dict.fromkeys([base_close_eod, not base_close_eod]))
            hold_vals = sorted({int(base_hold), int(max(0, base_hold + 2))})
            alt_gate = "off" if str(base_gate).strip().lower() != "off" else "regime_or_permission"
            gate_modes = list(dict.fromkeys([str(base_gate), str(alt_gate)]))
            flip_modes = [str(base_flip_mode)]
            run_title = "PT/SL sweep (seeded-local mutation)"
        else:
            # Unified fixed-percent exit pocket:
            # combines PT/SL and exit-pivot neighborhoods in one core sweep.
            pt_vals = [None, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.01, 0.015, 0.02]
            sl_vals = [0.003, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03]
            only_profit_vals = [False, True]
            close_eod_vals = [False, True]
            hold_vals = [0, 2]
            gate_modes = ["off", "regime_or_permission"]
            flip_modes = ["entry"]
            run_title = "PT/SL sweep (fixed pct exits + flip/close_eod semantics)"
        plan = []
        for pt in pt_vals:
            for sl in sl_vals:
                for only_profit in only_profit_vals:
                    for close_eod in close_eod_vals:
                        for hold in hold_vals:
                            for gate_mode in gate_modes:
                                for flip_mode in flip_modes:
                                    cfg = replace(
                                        base_cfg,
                                        strategy=replace(
                                            base_cfg.strategy,
                                            spot_profit_target_pct=(float(pt) if pt is not None else None),
                                            spot_stop_loss_pct=float(sl),
                                            spot_exit_mode="pct",
                                            exit_on_signal_flip=True,
                                            flip_exit_mode=str(flip_mode),
                                            flip_exit_only_if_profit=bool(only_profit),
                                            flip_exit_min_hold_bars=int(hold),
                                            flip_exit_gate_mode=str(gate_mode),
                                            spot_close_eod=bool(close_eod),
                                        ),
                                    )
                                    pt_note = "None" if pt is None else f"{float(pt):.4f}"
                                    note = (
                                        f"PT={pt_note} SL={float(sl):.4f} "
                                        f"flip={str(flip_mode)} hold={int(hold)} only_profit={int(only_profit)} "
                                        f"gate={gate_mode} close_eod={int(close_eod)}"
                                    )
                                    plan.append((cfg, note, None))
        _tested, kept = _run_sweep(
            plan=plan,
            bars=bars_sig,
            total=len(plan),
            progress_label="ptsl axis",
            report_every=50,
            heartbeat_sec=15.0,
        )
        rows = [row for _, row, _note, _meta in kept]
        _print_leaderboards(rows, title=run_title, top_n=int(args.top))

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

        # Keep this tight and focused; the point is to cover interaction edges that compact preset funnels can miss.
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

        # Unified advanced pocket from centralized axis dimensions.
        shock_dims = _AXIS_DIMENSION_REGISTRY.get("shock", {})
        advanced_modes = tuple(shock_dims.get("advanced_modes") or ())
        advanced_detectors: tuple[tuple[dict[str, object], str], ...] = tuple(
            shock_dims.get("advanced_detectors") or ()
        )
        advanced_short_factors = tuple(shock_dims.get("advanced_short_risk_factors") or ())
        advanced_long_down_factors = tuple(shock_dims.get("advanced_long_down_factors") or ())
        advanced_scales: tuple[tuple[dict[str, object], str], ...] = tuple(shock_dims.get("advanced_scales") or ())
        for mode in advanced_modes:
            for det_over, det_note in advanced_detectors:
                for dir_src, dir_lb, dir_note in dir_variants:
                    for sl_mult in sl_mults:
                        for short_factor in advanced_short_factors:
                            for long_down in advanced_long_down_factors:
                                for scale_over, scale_note in advanced_scales:
                                    overrides = {
                                        "shock_gate_mode": str(mode),
                                        "shock_direction_source": str(dir_src),
                                        "shock_direction_lookback": int(dir_lb),
                                        "shock_stop_loss_pct_mult": float(sl_mult),
                                        "shock_profit_target_pct_mult": 1.0,
                                        "shock_short_risk_mult_factor": float(short_factor),
                                        "shock_long_risk_mult_factor_down": float(long_down),
                                    }
                                    overrides.update(det_over)
                                    overrides.update(scale_over)
                                    f = _mk_filters(overrides=overrides)
                                    cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                    note = (
                                        f"shock_adv={mode} {det_note} | {dir_note} | "
                                        f"sl_mult={sl_mult:g} short_factor={short_factor:g} "
                                        f"long_down={long_down:g} | {scale_note}"
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

    def _sweep_combo_full() -> None:
        """Unified tight Cartesian sweep over centralized combo dimensions."""

        bars_sig = _bars_cached(signal_bar_size)
        dims = _AXIS_DIMENSION_REGISTRY.get("combo_full_cartesian_tight", {})
        if not isinstance(dims, dict):
            raise SystemExit("combo_full dimension registry missing: combo_full_cartesian_tight")
        combo_full_preset = _combo_full_preset_key(str(getattr(args, "combo_full_preset", "") or ""))
        valid_combo_presets = set(_combo_full_preset_axes())
        if combo_full_preset and combo_full_preset not in valid_combo_presets:
            raise SystemExit(f"Unknown combo_full preset: {combo_full_preset!r}")

        direction_catalog_default: dict[str, dict[str, object]] = {
            "ema=2/4 cross": {
                "entry_signal": "ema",
                "ema_preset": "2/4",
                "ema_entry_mode": "cross",
            },
            "ema=4/9 cross": {
                "entry_signal": "ema",
                "ema_preset": "4/9",
                "ema_entry_mode": "cross",
            },
        }
        regime_catalog_default: dict[str, dict[str, object]] = {
            "regime=ST(4h:7,0.5,hl2)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "4 hours",
                "supertrend_atr_period": 7,
                "supertrend_multiplier": 0.5,
                "supertrend_source": "hl2",
            },
            "regime=ST(1d:14,1.0,hl2)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "1 day",
                "supertrend_atr_period": 14,
                "supertrend_multiplier": 1.0,
                "supertrend_source": "hl2",
            },
            "regime=ST(4h:10,0.8,close)": {
                "regime_mode": "supertrend",
                "regime_bar_size": "4 hours",
                "supertrend_atr_period": 10,
                "supertrend_multiplier": 0.8,
                "supertrend_source": "close",
            },
        }
        regime2_catalog_default: dict[str, dict[str, object]] = {
            "r2=off": {"regime2_mode": "off", "regime2_bar_size": None},
            "r2=ST(4h:3,0.25,close)": {
                "regime2_mode": "supertrend",
                "regime2_bar_size": "4 hours",
                "regime2_supertrend_atr_period": 3,
                "regime2_supertrend_multiplier": 0.25,
                "regime2_supertrend_source": "close",
            },
        }
        exit_catalog_default: dict[str, dict[str, object]] = {
            "exit=pct(0.015,0.03)": {
                "spot_exit_mode": "pct",
                "spot_profit_target_pct": 0.015,
                "spot_stop_loss_pct": 0.03,
                "spot_pt_atr_mult": None,
                "spot_sl_atr_mult": None,
            },
            "exit=stop_only(0.03)": {
                "spot_exit_mode": "pct",
                "spot_profit_target_pct": None,
                "spot_stop_loss_pct": 0.03,
                "spot_pt_atr_mult": None,
                "spot_sl_atr_mult": None,
            },
            "exit=atr(14,0.8,1.6)": {
                "spot_exit_mode": "atr",
                "spot_profit_target_pct": None,
                "spot_stop_loss_pct": None,
                "spot_atr_period": 14,
                "spot_pt_atr_mult": 0.8,
                "spot_sl_atr_mult": 1.6,
            },
        }
        tick_catalog_default: dict[str, dict[str, object]] = {
            "tick=off": {"tick_gate_mode": "off"},
            "tick=raschke": {
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
            },
        }
        shock_catalog_default: dict[str, dict[str, object]] = {
            "shock=off": {"shock_gate_mode": "off"},
            "shock=surf_daily": {
                "shock_gate_mode": "surf",
                "shock_detector": "daily_atr_pct",
                "shock_daily_atr_period": 14,
                "shock_daily_on_atr_pct": 13.5,
                "shock_daily_off_atr_pct": 13.0,
                "shock_daily_on_tr_pct": 9.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
            "shock=surf_atr_ratio": {
                "shock_gate_mode": "surf",
                "shock_detector": "atr_ratio",
                "shock_atr_fast_period": 7,
                "shock_atr_slow_period": 50,
                "shock_on_ratio": 1.55,
                "shock_off_ratio": 1.30,
                "shock_min_atr_pct": 7.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
            "shock=surf_tr_ratio": {
                "shock_gate_mode": "surf",
                "shock_detector": "tr_ratio",
                "shock_atr_fast_period": 7,
                "shock_atr_slow_period": 50,
                "shock_on_ratio": 1.55,
                "shock_off_ratio": 1.30,
                "shock_min_atr_pct": 7.0,
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "shock_stop_loss_pct_mult": 0.75,
                "shock_profit_target_pct_mult": 1.0,
            },
        }
        slope_catalog_default: dict[str, dict[str, object]] = {
            "slope=off": {},
            "slope>=0.01": {"ema_slope_min_pct": 0.01},
        }
        risk_catalog_default: dict[str, dict[str, object]] = {
            "risk=off": {},
            "risk=riskoff9": _risk_pack_riskoff(tr_med=9.0, lookback_days=5, mode="hygiene", cutoff_hour_et=15),
            "risk=riskpanic9": _risk_pack_riskpanic(
                tr_med=9.0,
                neg_gap_ratio=0.6,
                lookback_days=5,
                short_factor=0.5,
                cutoff_hour_et=15,
            ),
            "risk=riskpop9": _risk_pack_riskpop(
                tr_med=9.0,
                pos_gap_ratio=0.6,
                lookback_days=5,
                long_factor=1.2,
                short_factor=0.5,
                cutoff_hour_et=15,
            ),
        }

        def _pairs_from_registry(
            *,
            dim_name: str,
            variants_key: str,
            fallback_catalog: dict[str, dict[str, object]],
        ) -> list[tuple[str, dict[str, object]]]:
            out: list[tuple[str, dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    out.append((label, dict(payload)))
            if out:
                return out
            fallback_rows = [
                (str(label), dict(payload))
                for label, payload in tuple(fallback_catalog.items())
                if isinstance(label, str) and isinstance(payload, dict)
            ]
            if not fallback_rows:
                raise SystemExit(f"combo_full requires at least one {dim_name} variant.")
            return fallback_rows

        def _timing_profiles_from_registry(
            *,
            variants_key: str,
        ) -> list[tuple[str, dict[str, object], dict[str, object]]]:
            out: list[tuple[str, dict[str, object], dict[str, object]]] = []
            raw_variants = dims.get(str(variants_key))
            if isinstance(raw_variants, (list, tuple)):
                for row in tuple(raw_variants):
                    if not (isinstance(row, (list, tuple)) and len(row) >= 2):
                        continue
                    label = str(row[0] or "").strip()
                    payload = row[1]
                    if not label or not isinstance(payload, dict):
                        continue
                    strat_over = payload.get("strategy_overrides")
                    filt_over = payload.get("filter_overrides")
                    strat_dict = dict(strat_over) if isinstance(strat_over, dict) else {}
                    filt_dict = dict(filt_over) if isinstance(filt_over, dict) else {}
                    out.append((str(label), strat_dict, filt_dict))
            return out

        perm_catalog_default = {
            str(note): dict(over)
            for note, over in tuple(_PERM_JOINT_PROFILE.get("perm_variants") or ())
            if isinstance(note, str) and isinstance(over, dict)
        }
        tod_catalog_default = {
            str(note): dict(over)
            for _start_h, _end_h, note, over in tuple(_PERM_JOINT_PROFILE.get("tod_windows") or ())
            if isinstance(note, str) and isinstance(over, dict)
        }
        vol_catalog_default = {
            str(note): dict(over)
            for note, over in tuple(_PERM_JOINT_PROFILE.get("vol_variants") or ())
            if isinstance(note, str) and isinstance(over, dict)
        }
        cadence_catalog_default = {
            str(note): dict(over)
            for note, over in tuple(_PERM_JOINT_PROFILE.get("cadence_variants") or ())
            if isinstance(note, str) and isinstance(over, dict)
        }
        pair_catalog_by_dim: dict[str, dict[str, dict[str, object]]] = {
            "direction": direction_catalog_default,
            "perm": perm_catalog_default,
            "tod": tod_catalog_default,
            "vol": vol_catalog_default,
            "cadence": cadence_catalog_default,
            "regime": regime_catalog_default,
            "regime2": regime2_catalog_default,
            "exit": exit_catalog_default,
            "tick": tick_catalog_default,
            "shock": shock_catalog_default,
            "slope": slope_catalog_default,
            "risk": risk_catalog_default,
        }
        pair_variants_by_dim: dict[str, list[tuple[str, dict[str, object]]]] = {
            str(dim_name): _pairs_from_registry(
                dim_name=str(dim_name),
                variants_key=str(variants_key),
                fallback_catalog=pair_catalog_by_dim[str(dim_name)],
            )
            for dim_name, variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }
        if not bool(getattr(args, "combo_full_include_tick", False)):
            tick_rows = [
                (str(label), dict(payload))
                for label, payload in tuple(pair_variants_by_dim.get("tick") or ())
                if str((payload or {}).get("tick_gate_mode") or "off").strip().lower() in ("off", "", "none", "false", "0")
            ]
            if not tick_rows:
                tick_rows = [("tick=off", {"tick_gate_mode": "off"})]
            pair_variants_by_dim["tick"] = tick_rows
        confirm_bars = [int(v) for v in tuple(dims.get("confirm_bars") or ())]
        timing_profile_variants = _timing_profiles_from_registry(variants_key="timing_profile_variants")
        if not timing_profile_variants:
            timing_profile_variants = [("timing=base", {}, {})]
        short_mults = [float(v) for v in tuple(dims.get("short_mults") or ())]
        if not confirm_bars or not short_mults:
            raise SystemExit("combo_full requires non-empty confirm_bars and short_mults.")

        dim_state: dict[str, list[object]] = {
            "timing_profile": list(timing_profile_variants),
            "confirm": list(confirm_bars),
            **{str(dim_name): list(rows) for dim_name, rows in pair_variants_by_dim.items()},
            "short_mult": list(short_mults),
        }

        def _freeze_dims(*dim_names: str) -> None:
            for dim_name in dim_names:
                rows = list(dim_state.get(str(dim_name)) or ())
                if not rows:
                    raise SystemExit(f"combo_full preset requires non-empty {dim_name} variants.")
                dim_state[str(dim_name)] = [rows[0]]

        def _set_dim_rows(dim_name: str, rows: list[object]) -> None:
            values = list(rows or ())
            if not values:
                raise SystemExit(f"combo_full preset generated empty {dim_name} variants.")
            dim_state[str(dim_name)] = list(values)

        def _combo_full_base_bundle() -> ConfigBundle:
            root = _base_bundle(bar_size=signal_bar_size, filters=None)
            return replace(
                root,
                strategy=replace(
                    root.strategy,
                    filters=None,
                    tick_gate_mode="off",
                    regime2_mode="off",
                    regime2_bar_size=None,
                    spot_exit_time_et=None,
                ),
            )

        def _ema_direction_rows() -> list[tuple[str, dict[str, object]]]:
            rows: list[tuple[str, dict[str, object]]] = []
            for preset in ("2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"):
                rows.append(
                    (
                        f"ema={preset} cross",
                        {"entry_signal": "ema", "ema_preset": str(preset), "ema_entry_mode": "cross"},
                    )
                )
            return rows

        def _atr_exit_rows(*, with_close_eod: bool = False) -> list[tuple[str, dict[str, object]]]:
            rows: list[tuple[str, dict[str, object]]] = []
            atr_periods = (10, 14, 21)
            if with_close_eod:
                pt_mults = (0.6, 0.65, 0.7, 0.75, 0.8)
                sl_mults = (1.2, 1.4, 1.6, 1.8, 2.0)
                close_eod_vals = (False, True)
            else:
                pt_mults = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
                sl_mults = (1.2, 1.4, 1.5, 1.6, 1.8, 2.0)
                close_eod_vals = (False,)
            for close_eod in close_eod_vals:
                for atr_p in atr_periods:
                    for pt_m in pt_mults:
                        for sl_m in sl_mults:
                            label = (
                                f"exit=atr({int(atr_p)},{float(pt_m):.2f},{float(sl_m):.2f})"
                                + (f" close_eod={int(bool(close_eod))}" if with_close_eod else "")
                            )
                            payload = {
                                "spot_exit_mode": "atr",
                                "spot_atr_period": int(atr_p),
                                "spot_pt_atr_mult": float(pt_m),
                                "spot_sl_atr_mult": float(sl_m),
                                "spot_profit_target_pct": None,
                                "spot_stop_loss_pct": None,
                            }
                            if with_close_eod:
                                payload["spot_close_eod"] = bool(close_eod)
                            rows.append((label, payload))
            return rows

        def _preset_squeeze() -> None:
            regime2_rows: list[tuple[str, dict[str, object]]] = [("r2=off", {"regime2_mode": "off", "regime2_bar_size": None})]
            atr_periods = (2, 3, 4, 5, 6, 7, 10, 11)
            multipliers = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
            for r2_bar in ("4 hours", "1 day"):
                for atr_p in atr_periods:
                    for mult in multipliers:
                        for src in ("close", "hl2"):
                            regime2_rows.append(
                                (
                                    f"r2=ST({int(atr_p)},{float(mult):g},{str(src)})@{str(r2_bar)}",
                                    {
                                        "regime2_mode": "supertrend",
                                        "regime2_bar_size": str(r2_bar),
                                        "regime2_supertrend_atr_period": int(atr_p),
                                        "regime2_supertrend_multiplier": float(mult),
                                        "regime2_supertrend_source": str(src),
                                    },
                                )
                            )
            vol_rows: list[tuple[str, dict[str, object]]] = [
                ("vol=-", {"volume_ratio_min": None, "volume_ema_period": None}),
                ("vol>=1.0@20", {"volume_ratio_min": 1.0, "volume_ema_period": 20}),
                ("vol>=1.1@20", {"volume_ratio_min": 1.1, "volume_ema_period": 20}),
                ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
                ("vol>=1.5@10", {"volume_ratio_min": 1.5, "volume_ema_period": 10}),
                ("vol>=1.5@20", {"volume_ratio_min": 1.5, "volume_ema_period": 20}),
            ]
            tod_rows: list[tuple[str, dict[str, object]]] = [
                ("tod=base", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
                ("tod=18-03 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 3}),
                ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
                ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
                ("tod=11-16 ET", {"entry_start_hour_et": 11, "entry_end_hour_et": 16}),
            ]
            _set_dim_rows("regime2", regime2_rows)
            _set_dim_rows("vol", vol_rows)
            _set_dim_rows("tod", tod_rows)
            _set_dim_rows("confirm", [0, 1, 2])
            _set_dim_rows("short_mult", [1.0])

        def _preset_tod_interaction() -> None:
            tod_rows: list[tuple[str, dict[str, object]]] = []
            for start_h in (17, 18, 19):
                for end_h in (3, 4, 5):
                    tod_rows.append(
                        (
                            f"tod={int(start_h):02d}-{int(end_h):02d} ET",
                            {"entry_start_hour_et": int(start_h), "entry_end_hour_et": int(end_h)},
                        )
                    )
            cadence_rows: list[tuple[str, dict[str, object]]] = []
            for skip in (0, 1, 2):
                for cooldown in (0, 1, 2):
                    cadence_rows.append(
                        (
                            f"cad=skip{int(skip)} cd{int(cooldown)}",
                            {"skip_first_bars": int(skip), "cooldown_bars": int(cooldown)},
                        )
                    )
            _set_dim_rows("tod", tod_rows)
            _set_dim_rows("cadence", cadence_rows)
            _set_dim_rows("short_mult", [1.0])

        def _preset_risk_overlays() -> None:
            if bool(getattr(args, "risk_overlays_skip_pop", False)):
                filtered = [row for row in list(dim_state.get("risk") or ()) if "riskpop" not in str(row[0]).lower()]
                _set_dim_rows("risk", filtered)
            _set_dim_rows("short_mult", [1.0])

        def _preset_gate_matrix() -> None:
            gate_dims = _AXIS_DIMENSION_REGISTRY.get("gate_matrix", {})
            if isinstance(gate_dims, dict):
                perm_raw = tuple(gate_dims.get("perm_variants") or ())
                if perm_raw:
                    _set_dim_rows(
                        "perm",
                        [
                            (str(label), dict(payload))
                            for label, payload in perm_raw
                            if isinstance(label, str) and isinstance(payload, dict)
                        ],
                    )
                tod_raw = tuple(gate_dims.get("tod_variants") or ())
                if tod_raw:
                    _set_dim_rows(
                        "tod",
                        [
                            (
                                str(label),
                                {
                                    "entry_start_hour_et": (int(start_h) if start_h is not None else None),
                                    "entry_end_hour_et": (int(end_h) if end_h is not None else None),
                                },
                            )
                            for label, start_h, end_h in tod_raw
                            if isinstance(label, str)
                        ],
                    )
                short_raw = tuple(gate_dims.get("short_mults") or ())
                if short_raw:
                    _set_dim_rows("short_mult", [float(v) for v in short_raw])

        def _preset_ema_regime() -> None:
            _set_dim_rows("direction", _ema_direction_rows())
            _set_dim_rows("short_mult", [1.0])

        def _preset_tick_ema() -> None:
            _set_dim_rows("direction", _ema_direction_rows())
            tick_rows: list[tuple[str, dict[str, object]]] = []
            for policy in ("allow", "block"):
                for z_enter in (0.8, 1.0, 1.2):
                    for z_exit in (0.4, 0.5, 0.6):
                        for slope_lb in (3, 5):
                            for lookback in (126, 252):
                                tick_rows.append(
                                    (
                                        f"tick=wide policy={policy} z_in={float(z_enter):g} "
                                        f"z_out={float(z_exit):g} slope={int(slope_lb)} lb={int(lookback)}",
                                        {
                                            "tick_gate_mode": "raschke",
                                            "tick_gate_symbol": "TICK-AMEX",
                                            "tick_gate_exchange": "AMEX",
                                            "tick_neutral_policy": str(policy),
                                            "tick_direction_policy": "wide_only",
                                            "tick_band_ma_period": 10,
                                            "tick_width_z_lookback": int(lookback),
                                            "tick_width_z_enter": float(z_enter),
                                            "tick_width_z_exit": float(z_exit),
                                            "tick_width_slope_lookback": int(slope_lb),
                                        },
                                    )
                                )
            if bool(getattr(args, "combo_full_include_tick", False)):
                _set_dim_rows("tick", tick_rows)
            else:
                _set_dim_rows("tick", [("tick=off", {"tick_gate_mode": "off"})])
            _set_dim_rows("short_mult", [1.0])

        def _preset_ema_atr() -> None:
            _set_dim_rows("direction", _ema_direction_rows())
            _set_dim_rows("exit", _atr_exit_rows(with_close_eod=False))
            _set_dim_rows("short_mult", [1.0])

        def _preset_r2_atr() -> None:
            _set_dim_rows("exit", _atr_exit_rows(with_close_eod=False))
            _set_dim_rows("short_mult", [1.0])

        def _preset_r2_tod() -> None:
            tod_rows = [
                (str(note), dict(over))
                for _start_h, _end_h, note, over in tuple(_PERM_JOINT_PROFILE.get("tod_windows") or ())
                if isinstance(note, str) and isinstance(over, dict)
            ]
            if tod_rows:
                _set_dim_rows("tod", tod_rows)
            _set_dim_rows("short_mult", [1.0])

        def _preset_loosen_atr() -> None:
            _set_dim_rows("exit", _atr_exit_rows(with_close_eod=True))
            _set_dim_rows("short_mult", [1.0])

        def _preset_lf_shock_sniper() -> None:
            _set_dim_rows(
                "regime",
                [
                    (
                        "regime=ST(1d:14,1.0,hl2)",
                        {
                            "regime_mode": "supertrend",
                            "regime_bar_size": "1 day",
                            "supertrend_atr_period": 14,
                            "supertrend_multiplier": 1.0,
                            "supertrend_source": "hl2",
                        },
                    )
                ],
            )
            _set_dim_rows("regime2", [("r2=off", {"regime2_mode": "off", "regime2_bar_size": None})])
            _set_dim_rows("perm", [("perm=off", {"ema_spread_min_pct": None, "ema_slope_min_pct": None})])
            _set_dim_rows("tod", [("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None})])
            _set_dim_rows("vol", [("vol=off", {"volume_ratio_min": None, "volume_ema_period": None})])
            _set_dim_rows("cadence", [("cad=base", {})])
            _set_dim_rows("tick", [("tick=off", {"tick_gate_mode": "off"})])
            _set_dim_rows("risk", [("risk=off", {})])
            _set_dim_rows(
                "shock",
                [
                    ("shock=off", {"shock_gate_mode": "off"}),
                    (
                        "shock=tr_ratio on=1.300 off=1.200",
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "tr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": 1.300,
                            "shock_off_ratio": 1.200,
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                            "shock_profit_target_pct_mult": 1.0,
                        },
                    ),
                    (
                        "shock=tr_ratio on=1.325 off=1.225",
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "tr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": 1.325,
                            "shock_off_ratio": 1.225,
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                            "shock_profit_target_pct_mult": 1.0,
                        },
                    ),
                    (
                        "shock=tr_ratio on=1.350 off=1.250",
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "tr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": 1.350,
                            "shock_off_ratio": 1.250,
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                            "shock_profit_target_pct_mult": 1.0,
                        },
                    ),
                    (
                        "shock=tr_ratio on=1.355 off=1.255",
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "tr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": 1.355,
                            "shock_off_ratio": 1.255,
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                            "shock_profit_target_pct_mult": 1.0,
                        },
                    ),
                ],
            )
            _set_dim_rows("short_mult", [1.0])

        def _preset_hf_timing_sniper() -> None:
            base = _combo_full_base_bundle()
            hf_profiles = _timing_profiles_from_registry(variants_key="hf_profile_variants")
            if not hf_profiles:
                _set_dim_rows("short_mult", [1.0])
                return
            base_filters_payload = _filters_payload(getattr(base.strategy, "filters", None)) or {}
            base_filter_row = ("base_filters", dict(base_filters_payload))
            _set_dim_rows(
                "direction",
                [
                    (
                        "direction=base",
                        {
                            "entry_signal": str(getattr(base.strategy, "entry_signal", "ema") or "ema"),
                            "ema_preset": str(getattr(base.strategy, "ema_preset", "3/7") or "3/7"),
                            "ema_entry_mode": str(getattr(base.strategy, "ema_entry_mode", "trend") or "trend"),
                        },
                    )
                ],
            )
            _set_dim_rows("confirm", [int(getattr(base.strategy, "entry_confirm_bars", 0) or 0)])
            _set_dim_rows("perm", [base_filter_row])
            _set_dim_rows("tod", [base_filter_row])
            _set_dim_rows("vol", [base_filter_row])
            _set_dim_rows("cadence", [base_filter_row])
            _set_dim_rows(
                "regime",
                [
                    (
                        "regime=base",
                        {
                            "regime_mode": str(getattr(base.strategy, "regime_mode", "supertrend") or "supertrend"),
                            "regime_bar_size": str(getattr(base.strategy, "regime_bar_size", "1 day") or "1 day"),
                            "supertrend_atr_period": int(getattr(base.strategy, "supertrend_atr_period", 7) or 7),
                            "supertrend_multiplier": float(getattr(base.strategy, "supertrend_multiplier", 0.4) or 0.4),
                            "supertrend_source": str(getattr(base.strategy, "supertrend_source", "close") or "close"),
                        },
                    )
                ],
            )
            _set_dim_rows(
                "regime2",
                [
                    (
                        "regime2=base",
                        {
                            "regime2_mode": str(getattr(base.strategy, "regime2_mode", "off") or "off"),
                            "regime2_bar_size": getattr(base.strategy, "regime2_bar_size", None),
                            "regime2_ema_preset": getattr(base.strategy, "regime2_ema_preset", None),
                            "regime2_supertrend_atr_period": int(
                                getattr(base.strategy, "regime2_supertrend_atr_period", 10) or 10
                            ),
                            "regime2_supertrend_multiplier": float(
                                getattr(base.strategy, "regime2_supertrend_multiplier", 3.0) or 3.0
                            ),
                            "regime2_supertrend_source": str(
                                getattr(base.strategy, "regime2_supertrend_source", "hl2") or "hl2"
                            ),
                        },
                    )
                ],
            )
            _set_dim_rows(
                "exit",
                [
                    (
                        "exit=base",
                        {
                            "spot_exit_mode": str(getattr(base.strategy, "spot_exit_mode", "pct") or "pct"),
                            "spot_profit_target_pct": getattr(base.strategy, "spot_profit_target_pct", None),
                            "spot_stop_loss_pct": getattr(base.strategy, "spot_stop_loss_pct", None),
                            "spot_atr_period": int(getattr(base.strategy, "spot_atr_period", 14) or 14),
                            "spot_pt_atr_mult": float(getattr(base.strategy, "spot_pt_atr_mult", 1.5) or 1.5),
                            "spot_sl_atr_mult": float(getattr(base.strategy, "spot_sl_atr_mult", 1.0) or 1.0),
                            "spot_close_eod": bool(getattr(base.strategy, "spot_close_eod", False)),
                        },
                    )
                ],
            )
            _set_dim_rows(
                "tick",
                [
                    (
                        "tick=base",
                        {
                            "tick_gate_mode": str(getattr(base.strategy, "tick_gate_mode", "off") or "off"),
                            "tick_gate_symbol": str(getattr(base.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE"),
                            "tick_gate_exchange": str(getattr(base.strategy, "tick_gate_exchange", "NYSE") or "NYSE"),
                            "tick_neutral_policy": str(getattr(base.strategy, "tick_neutral_policy", "allow") or "allow"),
                            "tick_direction_policy": str(getattr(base.strategy, "tick_direction_policy", "both") or "both"),
                            "tick_band_ma_period": int(getattr(base.strategy, "tick_band_ma_period", 10) or 10),
                            "tick_width_z_lookback": int(getattr(base.strategy, "tick_width_z_lookback", 252) or 252),
                            "tick_width_z_enter": float(getattr(base.strategy, "tick_width_z_enter", 1.0) or 1.0),
                            "tick_width_z_exit": float(getattr(base.strategy, "tick_width_z_exit", 0.5) or 0.5),
                            "tick_width_slope_lookback": int(getattr(base.strategy, "tick_width_slope_lookback", 3) or 3),
                        },
                    )
                ],
            )
            _set_dim_rows("shock", [base_filter_row])
            _set_dim_rows("slope", [base_filter_row])
            _set_dim_rows("risk", [base_filter_row])
            _set_dim_rows("short_mult", [float(getattr(base.strategy, "spot_short_risk_mult", 1.0) or 1.0)])
            base_rows = [row for row in hf_profiles if "hf_symm" in str(row[0]).lower()]
            # Keep this preset intentionally tight: one HF symm anchor, then mutate
            # only the highest-value timing/slope/velocity/branch-size pockets.
            if base_rows:
                base_rows = [base_rows[0]]
            if not base_rows:
                base_rows = list(hf_profiles[:1])
            if not base_rows:
                base_rows = list(hf_profiles[:1])
            timing_rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
            seen_labels: set[str] = set()
            base_mode_profiles: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
                (
                    "overlay_only_hybrid_baseline",
                    {
                        "ratsv_enabled": True,
                        "ratsv_slope_slow_window_bars": 11,
                    },
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 14,
                        "supertrend_multiplier": 0.6,
                        "supertrend_source": "hl2",
                        "regime2_mode": "off",
                        "regime2_bar_size": None,
                        "spot_policy_graph": "aggressive",
                        "spot_risk_overlay_policy": "trend_bias",
                        "spot_resize_mode": "target",
                        "spot_resize_min_delta_qty": 3,
                        "spot_resize_max_step_qty": 2,
                        "spot_resize_cooldown_bars": 6,
                        "spot_resize_adaptive_mode": "hybrid",
                        "spot_resize_adaptive_min_mult": 0.90,
                        "spot_resize_adaptive_max_mult": 1.40,
                        "spot_resize_adaptive_slope_ref_pct": 0.06,
                        "spot_resize_adaptive_vel_ref_pct": 0.04,
                        "spot_resize_adaptive_tr_ratio_ref": 1.00,
                        "spot_graph_overlay_atr_hi_pct": 8.0,
                        "spot_graph_overlay_atr_hi_min_mult": 0.85,
                        "spot_graph_overlay_trend_boost_max": 1.35,
                        "spot_graph_overlay_slope_ref_pct": 0.06,
                        "spot_graph_overlay_tr_ratio_ref": 1.05,
                        "spot_graph_overlay_trend_floor_mult": 0.90,
                        "exit_on_signal_flip": True,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                    },
                ),
                (
                    "overlay_only_hybrid_predref",
                    {
                        "ratsv_enabled": True,
                        "ratsv_slope_slow_window_bars": 11,
                    },
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "1 day",
                        "supertrend_atr_period": 14,
                        "supertrend_multiplier": 0.6,
                        "supertrend_source": "hl2",
                        "regime2_mode": "off",
                        "regime2_bar_size": None,
                        "spot_policy_graph": "aggressive",
                        "spot_risk_overlay_policy": "trend_bias",
                        "spot_resize_mode": "target",
                        "spot_resize_min_delta_qty": 3,
                        "spot_resize_max_step_qty": 2,
                        "spot_resize_cooldown_bars": 6,
                        "spot_resize_adaptive_mode": "hybrid",
                        "spot_resize_adaptive_min_mult": 0.90,
                        "spot_resize_adaptive_max_mult": 1.60,
                        "spot_resize_adaptive_atr_target_pct": 8.0,
                        "spot_resize_adaptive_atr_vel_ref_pct": 0.25,
                        "spot_resize_adaptive_slope_ref_pct": 0.055,
                        "spot_resize_adaptive_vel_ref_pct": 0.032,
                        "spot_resize_adaptive_tr_ratio_ref": 0.98,
                        "spot_exit_flip_hold_tr_ratio_min": 1.00,
                        "spot_exit_flip_hold_slow_slope_min_pct": 0.000002,
                        "spot_exit_flip_hold_slow_slope_vel_min_pct": 0.000001,
                        "spot_graph_overlay_atr_hi_pct": 8.0,
                        "spot_graph_overlay_atr_hi_min_mult": 0.84,
                        "spot_graph_overlay_atr_vel_ref_pct": 0.25,
                        "spot_graph_overlay_trend_boost_max": 1.75,
                        "spot_graph_overlay_slope_ref_pct": 0.055,
                        "spot_graph_overlay_tr_ratio_ref": 1.00,
                        "spot_graph_overlay_trend_floor_mult": 0.88,
                        "exit_on_signal_flip": True,
                        "flip_exit_mode": "cross",
                        "flip_exit_gate_mode": "regime_or_permission",
                        "flip_exit_min_hold_bars": 0,
                        "flip_exit_only_if_profit": False,
                    },
                ),
            )
            bridge_only = str(os.environ.get("TB_HF_TIMING_SNIPER_BRIDGE", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            mode_profiles: tuple[tuple[str, dict[str, object], dict[str, object]], ...]
            if bridge_only:
                predref_filter_over = dict(base_mode_profiles[1][1])
                predref_strategy_over = dict(base_mode_profiles[1][2])
                bridge_rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
                for flip_hold_tr_ratio in (0.97, 0.99):
                    for trend_floor_mult in (0.84, 0.86):
                        strat_over = dict(predref_strategy_over)
                        strat_over["spot_exit_flip_hold_tr_ratio_min"] = float(flip_hold_tr_ratio)
                        strat_over["spot_graph_overlay_trend_floor_mult"] = float(trend_floor_mult)
                        tag = (
                            f"overlay_only_hybrid_predref_bridge_tr{float(flip_hold_tr_ratio):0.2f}"
                            f"_floor{float(trend_floor_mult):0.2f}"
                        )
                        bridge_rows.append((str(tag), dict(predref_filter_over), strat_over))
                mode_profiles = tuple(bridge_rows)
            else:
                mode_profiles = tuple(base_mode_profiles)
            rank_values = (0.0035,) if bridge_only else (0.0035, 0.0185)
            cross_age_values = (4, 6) if bridge_only else (6, 10)
            slope_pairs = (
                (0.000002, 0.000001),
                (0.000006, 0.000002),
            )
            branch_b_mult_values = (1.20, 1.60)
            variants: list[tuple[float, int, float, float, float, str, dict[str, object], dict[str, object]]] = []
            for tag, filt_extra, strat_extra in mode_profiles:
                for rank_min in rank_values:
                    for cross_age in cross_age_values:
                        for slope_med, slope_vel in slope_pairs:
                            for branch_b_mult in branch_b_mult_values:
                                variants.append(
                                    (
                                        float(rank_min),
                                        int(cross_age),
                                        float(slope_med),
                                        float(slope_vel),
                                        float(branch_b_mult),
                                        str(tag),
                                        dict(filt_extra),
                                        dict(strat_extra),
                                    )
                                )
            for label, strat_over, filt_over in tuple(base_rows):
                for rank_min, cross_age, slope_med, slope_vel, branch_b_mult, tag, filt_extra, strat_extra in variants:
                    filt = dict(filt_over)
                    filt["ratsv_branch_a_rank_min"] = float(rank_min)
                    filt["ratsv_branch_a_cross_age_max_bars"] = int(cross_age)
                    filt["ratsv_branch_a_slope_med_min_pct"] = float(slope_med)
                    filt["ratsv_branch_a_slope_vel_min_pct"] = float(slope_vel)
                    filt.update(dict(filt_extra))
                    strat = dict(strat_over)
                    strat["spot_branch_b_size_mult"] = float(branch_b_mult)
                    strat.update(dict(strat_extra))
                    custom_label = (
                        f"{str(label)} | sniper rank={float(rank_min):0.4f} "
                        f"cross={int(cross_age)} "
                        f"slope={float(slope_med):0.6f}/{float(slope_vel):0.6f} "
                        f"b_mult={float(branch_b_mult):0.2f} tag={tag}"
                    )
                    if custom_label in seen_labels:
                        continue
                    seen_labels.add(custom_label)
                    timing_rows.append((str(custom_label), strat, filt))
            if timing_rows:
                _set_dim_rows("timing_profile", list(timing_rows))
            _set_dim_rows("short_mult", [1.0])

        preset_customizers: dict[str, object] = {
            "squeeze": _preset_squeeze,
            "tod_interaction": _preset_tod_interaction,
            "risk_overlays": _preset_risk_overlays,
            "gate_matrix": _preset_gate_matrix,
            "lf_shock_sniper": _preset_lf_shock_sniper,
            "hf_timing_sniper": _preset_hf_timing_sniper,
            "ema_regime": _preset_ema_regime,
            "tick_ema": _preset_tick_ema,
            "ema_atr": _preset_ema_atr,
            "r2_atr": _preset_r2_atr,
            "r2_tod": _preset_r2_tod,
            "loosen_atr": _preset_loosen_atr,
        }
        required_customizer_keys = {
            str(_combo_full_preset_customizer(name))
            for name in _combo_full_preset_axes(include_tiers=True, include_aliases=True)
            if str(_combo_full_preset_customizer(name)).strip()
        }
        unknown_preset_customizers = sorted(required_customizer_keys - set(preset_customizers))
        if unknown_preset_customizers:
            raise SystemExit(
                f"Unknown combo_full preset customizers: {', '.join(unknown_preset_customizers)}"
            )

        if combo_full_preset:
            preset_spec = _combo_full_preset_spec(str(combo_full_preset))
            freeze_dims = tuple(preset_spec.get("freeze_dims") or ())
            if not freeze_dims:
                raise SystemExit(f"Unknown combo_full preset: {combo_full_preset!r}")
            _freeze_dims(*freeze_dims)
            customizer_key = str(preset_spec.get("customizer") or "").strip().lower()
            customizer = preset_customizers.get(str(customizer_key))
            if callable(customizer):
                customizer()

        timing_profile_variants = list(dim_state["timing_profile"])
        confirm_bars = [int(v) for v in list(dim_state["confirm"])]
        pair_variants_by_dim = {
            str(dim_name): list(dim_state[str(dim_name)])
            for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
        }
        short_mults = [float(v) for v in list(dim_state["short_mult"])]
        filter_override_dims = ("perm", "tod", "vol", "cadence", "shock", "slope", "risk")
        strategy_override_dims = ("direction", "regime", "regime2", "exit", "tick")

        requires_tick_daily = any(
            str((payload or {}).get("tick_gate_mode") or "off").strip().lower() != "off"
            for _label, payload in pair_variants_by_dim["tick"]
        )
        if offline and requires_tick_daily:
            tick_warm_start = start_dt - timedelta(days=400)
            tick_ok = False
            for tick_sym in ("TICK-AMEX", "TICK-NYSE"):
                try:
                    _require_offline_cache_or_die(
                        data=data,
                        cache_dir=cache_dir,
                        symbol=tick_sym,
                        exchange=None,
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
                    "(expected under db/TICK-AMEX or db/TICK-NYSE). Run once without --offline to fetch."
                )

        size_by_dim = {
            str(dim_name): len(tuple(dim_state.get(str(dim_name)) or ()))
            for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
        }
        total = _cardinality(*tuple(size_by_dim.values()))
        if total <= 0:
            raise SystemExit("combo_full has empty Cartesian dimensions.")

        dominant_dims_raw = tuple(dims.get("dominant_dims") or ())
        dominant_dims = [str(name).strip() for name in dominant_dims_raw if str(name).strip() in size_by_dim]
        ordered_dims = list(dominant_dims)
        for dim_name in size_by_dim:
            if dim_name not in ordered_dims:
                ordered_dims.append(dim_name)

        def _mixed_radix_rank(dim_indices: dict[str, int]) -> int:
            rank = 0
            for dim_name in ordered_dims:
                rank = (int(rank) * int(size_by_dim[dim_name])) + int(dim_indices.get(dim_name, 0))
            return int(rank)

        def _mixed_radix_indices_from_rank(rank: int) -> dict[str, int]:
            rank_i = int(rank)
            if rank_i < 0 or rank_i >= int(total):
                raise ValueError(f"rank out of range: {rank_i} not in [0,{int(total)-1}]")
            out: dict[str, int] = {}
            rem = int(rank_i)
            for dim_name in reversed(ordered_dims):
                size_i = int(size_by_dim.get(str(dim_name), 0) or 0)
                if size_i <= 0:
                    raise ValueError(f"invalid dimension cardinality: {dim_name}={size_i}")
                out[str(dim_name)] = int(rem % size_i)
                rem //= size_i
            return out

        base = _combo_full_base_bundle()

        def _combo_full_cfg_note_meta_from_dim_indices(
            dim_indices: dict[str, int],
            *,
            rank_override: int | None = None,
        ) -> tuple[ConfigBundle, str, dict]:
            dim_index_by_name = {
                str(dim_name): int(dim_indices.get(str(dim_name), 0))
                for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
            }
            timing_i = int(dim_index_by_name["timing_profile"])
            conf_i = int(dim_index_by_name["confirm"])
            short_i = int(dim_index_by_name["short_mult"])
            timing_label, timing_strat_over, timing_filter_over = timing_profile_variants[timing_i]
            pair_variant_row_by_dim = {
                str(dim_name): pair_variants_by_dim[str(dim_name)][int(dim_index_by_name[str(dim_name)])]
                for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
            }
            pair_label_by_dim = {
                str(dim_name): str(row[0])
                for dim_name, row in pair_variant_row_by_dim.items()
            }
            pair_overrides_by_dim = {
                str(dim_name): dict(row[1])
                for dim_name, row in pair_variant_row_by_dim.items()
            }
            confirm = int(confirm_bars[conf_i])
            short_mult = float(short_mults[short_i])

            filters_overrides: dict[str, object] = {}
            for dim_name in filter_override_dims:
                over = pair_overrides_by_dim.get(str(dim_name))
                if over:
                    filters_overrides.update(over)
            if timing_filter_over:
                filters_overrides.update(timing_filter_over)
            filters_obj = _mk_filters(overrides=filters_overrides) if filters_overrides else None
            strat = base.strategy
            strategy_overrides: dict[str, object] = {}
            for dim_name in strategy_override_dims:
                over = pair_overrides_by_dim.get(str(dim_name))
                if over:
                    strategy_overrides.update(over)
            if timing_strat_over:
                strategy_overrides.update(timing_strat_over)
            strategy_overrides.pop("entry_confirm_bars", None)
            strategy_overrides.pop("spot_short_risk_mult", None)
            strategy_overrides.pop("filters", None)
            strat = replace(
                strat,
                filters=filters_obj,
                entry_confirm_bars=int(confirm),
                spot_short_risk_mult=float(short_mult),
                **strategy_overrides,
            )
            cfg = replace(base, strategy=strat)
            meta_item = {str(dim_name): int(dim_index_by_name[str(dim_name)]) for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER}
            meta_item["_mr_rank"] = int(_mixed_radix_rank(dim_indices)) if rank_override is None else int(rank_override)
            note_parts = [
                str(timing_label),
                str(pair_label_by_dim["direction"]),
                f"c={int(confirm)}",
                *[str(pair_label_by_dim[str(dim_name)]) for dim_name in _COMBO_FULL_NOTE_PAIR_DIM_ORDER],
                f"short_mult={float(short_mult):g}",
            ]
            note = " | ".join(note_parts)
            return cfg, str(note), meta_item

        def _combo_full_cfg_note_meta_from_rank(rank: int) -> tuple[ConfigBundle, str, dict]:
            dim_indices = _mixed_radix_indices_from_rank(int(rank))
            return _combo_full_cfg_note_meta_from_dim_indices(dim_indices, rank_override=int(rank))

        def _iter_combo_full_cartesian_plan():
            index_ranges = tuple(range(int(size_by_dim[str(dim_name)])) for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER)
            for raw_indices in itertools.product(*index_ranges):
                dim_indices = {
                    str(dim_name): int(idx)
                    for dim_name, idx in zip(_COMBO_FULL_CARTESIAN_DIM_ORDER, raw_indices)
                }
                yield _combo_full_cfg_note_meta_from_dim_indices(dim_indices)

        combo_dim_space_sig = _combo_full_dimension_space_signature(
            ordered_dims=tuple(ordered_dims),
            size_by_dim=size_by_dim,
            timing_profile_variants=timing_profile_variants,
            confirm_bars=confirm_bars,
            pair_variants_by_dim=pair_variants_by_dim,
            short_mults=short_mults,
        )

        def _combo_full_worker_stage_window_signature() -> str:
            raw = {
                "version": str(_RUN_CFG_CACHE_ENGINE_VERSION),
                "stage": "combo_full_cartesian",
                "symbol": str(symbol),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "signal_bar_size": str(signal_bar_size),
                "use_rth": bool(use_rth),
                "run_min_trades": int(run_min_trades),
                "preset": str(combo_full_preset or ""),
                "ordered_dims": tuple(str(v) for v in ordered_dims),
                "size_by_dim": tuple((str(k), int(v)) for k, v in size_by_dim.items()),
                "dim_space_sig": str(combo_dim_space_sig),
                "bars_sig": _bars_signature(bars_sig),
            }
            return hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        if args.combo_full_cartesian_stage:
            _run_sharded_stage_worker(
                stage_label="combo_full_cartesian",
                worker_raw=args.combo_full_cartesian_worker,
                workers_raw=args.combo_full_cartesian_workers,
                out_path_raw=str(args.combo_full_cartesian_out or ""),
                out_flag_name="combo-full-cartesian-out",
                plan_all=None,
                bars=bars_sig,
                report_every=0,
                heartbeat_sec=30.0,
                plan_total=int(total),
                plan_item_from_rank=_combo_full_cfg_note_meta_from_rank,
                rank_manifest_window_signature=_combo_full_worker_stage_window_signature(),
                rank_batch_size=2048,
            )
            return

        print("")
        print("=== combo_full: unified tight Cartesian core ===")
        if combo_full_preset:
            print(
                f"combo_full preset active: {combo_full_preset} (tier={_combo_full_preset_tier(str(combo_full_preset))})",
                flush=True,
            )
        print(
            "combo_full dimensions: total="
            f"{int(total)} "
            + " ".join(
                f"{str(dim_name)}={int(size_by_dim.get(str(dim_name), 0) or 0)}"
                for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER
            ),
            flush=True,
        )
        print(
            f"combo_full sharding order: {','.join(ordered_dims)}",
            flush=True,
        )
        print("")

        base_row = _run_cfg(cfg=base, bars=bars_sig)
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        combo_stage_args = tuple(
            ("--combo-full-preset", str(combo_full_preset))
            if combo_full_preset
            else ()
        )
        combo_manifest_window_sig = _combo_full_worker_stage_window_signature()

        def _combo_full_parallel_totals() -> tuple[int, int]:
            try:
                unresolved_ranges = _cartesian_rank_manifest_unresolved_ranges(
                    stage_label="combo_full_cartesian",
                    window_signature=str(combo_manifest_window_sig),
                    total=int(total),
                )
                unresolved_total = sum(
                    max(0, int(rank_hi) - int(rank_lo) + 1)
                    for rank_lo, rank_hi in tuple(unresolved_ranges)
                )
            except Exception:
                unresolved_total = int(total)
            unresolved_i = max(0, int(unresolved_total))
            prefetched_i = max(0, int(total) - int(unresolved_i))
            return int(unresolved_i), int(prefetched_i)

        tested_total = _run_stage_cfg_rows(
            stage_label="combo_full_cartesian",
            total=int(total),
            jobs_req=int(jobs),
            bars=bars_sig,
            report_every=200,
            heartbeat_sec=30.0,
            on_row=lambda _cfg, row, _note: rows.append(row),
            serial_plan_builder=_iter_combo_full_cartesian_plan,
            parallel_payloads_builder=lambda: (
                lambda unresolved_i, prefetched_i: _run_parallel_stage(
                    axis_name="combo_full",
                    stage_label="combo_full Cartesian",
                    total=int(unresolved_i),
                    jobs=int(jobs),
                    worker_tmp_prefix="tradebot_combo_full_cartesian_",
                    worker_tag="cfc",
                    out_prefix="combo_full_cartesian_out",
                    stage_flag="--combo-full-cartesian-stage",
                    stage_value="1",
                    worker_flag="--combo-full-cartesian-worker",
                    workers_flag="--combo-full-cartesian-workers",
                    out_flag="--combo-full-cartesian-out",
                    strip_flags_with_values=(
                        "--combo-full-cartesian-stage",
                        "--combo-full-cartesian-worker",
                        "--combo-full-cartesian-workers",
                        "--combo-full-cartesian-out",
                        "--combo-full-cartesian-run-min-trades",
                        "--combo-full-preset",
                    ),
                    run_min_trades_flag="--combo-full-cartesian-run-min-trades",
                    run_min_trades=int(run_min_trades),
                    stage_args=combo_stage_args,
                    capture_error="Failed to capture combo_full Cartesian worker stdout.",
                    failure_label="combo_full Cartesian worker",
                    missing_label="combo_full Cartesian",
                    invalid_label="combo_full Cartesian",
                    planner_stage_label="combo_full_cartesian",
                    prefetched_tested_if_empty=int(prefetched_i),
                )
            )(*_combo_full_parallel_totals()),
            parallel_default_note="combo_full Cartesian",
            parallel_dedupe_by_milestone_key=True,
            record_milestones=True,
        )
        if base_row:
            rows.append(base_row)
        print(
            f"combo_full Cartesian tested={int(tested_total)} kept={len(rows)} min_trades={int(run_min_trades)}",
            flush=True,
        )
        _print_leaderboards(rows, title="combo_full sweep (unified tight Cartesian)", top_n=int(args.top))

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
        axis_plan = _axis_mode_plan(mode="axis_all")
        _run_axis_plan_parallel_if_requested(
            axis_plan=list(axis_plan),
            jobs_req=int(jobs),
            label="axis=all parallel",
            tmp_prefix="tradebot_axis_all_",
            offline_error="--jobs>1 for --axis all requires --offline (avoid parallel IBKR sessions).",
        )
        return

    axis_registry = _build_axis_registry_from_scope(locals())

    if axis == "all":
        _run_axis_plan_serial(list(_axis_mode_plan(mode="axis_all")), timed=False)
    else:
        fn_obj = axis_registry.get(str(axis))
        fn = fn_obj if callable(fn_obj) else None
        if fn is not None:
            _run_axis_plan_serial([(str(axis), "single", False)], timed=False)

    if int(run_cfg_cache_hits) > 0 and int(run_calls_total) > 0:
        hit_rate = float(run_cfg_cache_hits) / float(run_calls_total)
        print(
            f"run_cfg cache (strategy+axis_dims+window): entries={len(run_cfg_cache)} hits={run_cfg_cache_hits}/{run_calls_total} "
            f"({hit_rate*100.0:0.1f}%) fp_hits={int(run_cfg_fingerprint_hits)}",
            flush=True,
        )
    if bool(run_cfg_persistent_enabled):
        print(
            f"run_cfg persistent cache (strategy+axis_dims+window): path={run_cfg_persistent_path} "
            f"hits={int(run_cfg_persistent_hits)} writes={int(run_cfg_persistent_writes)} "
            f"dim_index_writes={int(run_cfg_dim_index_writes)} "
            f"worker_plan_hits={int(worker_plan_cache_hits)} worker_plan_writes={int(worker_plan_cache_writes)} "
            f"stage_status_reads={int(stage_cell_status_reads)} stage_status_writes={int(stage_cell_status_writes)} "
            f"cartesian_manifest_reads={int(cartesian_manifest_reads)} "
            f"cartesian_manifest_writes={int(cartesian_manifest_writes)} "
            f"cartesian_manifest_hits={int(cartesian_manifest_hits)} "
            f"cartesian_rank_manifest_reads={int(cartesian_rank_manifest_reads)} "
            f"cartesian_rank_manifest_writes={int(cartesian_rank_manifest_writes)} "
            f"cartesian_rank_manifest_hits={int(cartesian_rank_manifest_hits)} "
            f"cartesian_rank_manifest_compactions={int(cartesian_rank_manifest_compactions)} "
            f"cartesian_rank_manifest_pending_ttl_prunes={int(cartesian_rank_manifest_pending_ttl_prunes)} "
            f"stage_rank_manifest_reads={int(stage_rank_manifest_reads)} "
            f"stage_rank_manifest_writes={int(stage_rank_manifest_writes)} "
            f"stage_rank_manifest_hits={int(stage_rank_manifest_hits)} "
            f"stage_rank_manifest_compactions={int(stage_rank_manifest_compactions)} "
            f"stage_rank_manifest_pending_ttl_prunes={int(stage_rank_manifest_pending_ttl_prunes)} "
            f"stage_unresolved_summary_reads={int(stage_unresolved_summary_reads)} "
            f"stage_unresolved_summary_writes={int(stage_unresolved_summary_writes)} "
            f"stage_unresolved_summary_hits={int(stage_unresolved_summary_hits)} "
            f"rank_dominance_stamp_reads={int(rank_dominance_stamp_reads)} "
            f"rank_dominance_stamp_writes={int(rank_dominance_stamp_writes)} "
            f"rank_dominance_stamp_hits={int(rank_dominance_stamp_hits)} "
            f"rank_dominance_manifest_applies={int(rank_dominance_manifest_applies)} "
            f"rank_dominance_stamp_compactions={int(rank_dominance_stamp_compactions)} "
            f"rank_dominance_stamp_ttl_prunes={int(rank_dominance_stamp_ttl_prunes)} "
            f"rank_bin_runtime_reads={int(rank_bin_runtime_reads)} "
            f"rank_bin_runtime_writes={int(rank_bin_runtime_writes)} "
            f"stage_frontier_reads={int(stage_frontier_reads)} stage_frontier_writes={int(stage_frontier_writes)} "
            f"stage_frontier_hits={int(stage_frontier_hits)} "
            f"winner_projection_reads={int(winner_projection_reads)} "
            f"winner_projection_writes={int(winner_projection_writes)} "
            f"winner_projection_hits={int(winner_projection_hits)} "
            f"dimension_utility_reads={int(dimension_utility_reads)} "
            f"dimension_utility_writes={int(dimension_utility_writes)} "
            f"dimension_utility_hint_hits={int(dimension_utility_hint_hits)} "
            f"dimension_upper_bound_reads={int(dimension_upper_bound_reads)} "
            f"dimension_upper_bound_writes={int(dimension_upper_bound_writes)} "
            f"dimension_upper_bound_deferred={int(dimension_upper_bound_deferred)} "
            f"planner_heartbeat_reads={int(planner_heartbeat_reads)} "
            f"planner_heartbeat_writes={int(planner_heartbeat_writes)} "
            f"planner_heartbeat_stale_candidates={int(planner_heartbeat_stale_candidates)} "
            f"series_pack_mmap_hints={int(series_pack_mmap_hint_hits)} "
            f"series_pack_pickle_hints={int(series_pack_pickle_hint_hits)} "
            f"series_pack_state_manifest_reads={int(series_pack_state_manifest_reads)} "
            f"series_pack_state_manifest_writes={int(series_pack_state_manifest_writes)} "
            f"series_pack_state_manifest_hits={int(series_pack_state_manifest_hits)}",
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


if __name__ == "__main__":
    # Default execution path: evolution sweeps CLI.
    main()
