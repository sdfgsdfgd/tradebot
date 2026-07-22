"""Shared configuration, cache-policy, and search-cost mechanics."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

from ...backtest.cache_ops import ensure_cached_window_with_policy
from ...backtest.config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    SpotLegConfig,
    SpotStrategyConfig,
    SyntheticConfig,
)
from ...backtest.config_filters import _parse_filters
from ...backtest.data import IBKRHistoricalData
from .dimensions import _AXIS_DIMENSION_REGISTRY, _SWEEP_COST_MODEL, _SWEEP_RUNTIME_POLICY
from .milestones import _filters_payload


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
    cache_policy: str = "strict",
) -> None:
    cache_ok, expected, _resolved, missing_ranges, err = ensure_cached_window_with_policy(
        data=data,
        cache_dir=cache_dir,
        symbol=str(symbol),
        exchange=exchange,
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        cache_policy=str(cache_policy),
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
        f"(expected: {expected}{missing_note}{detail}; cache_policy={str(cache_policy).strip().lower() or 'strict'}). "
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


def _registry_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _runtime_policy(section: str) -> dict[str, object]:
    raw = _SWEEP_RUNTIME_POLICY.get(str(section))
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
    cfg = _runtime_policy("claim_first_planner")
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
    cfg = _runtime_policy("claim_first_planner")
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
    cfg = _runtime_policy("jobs_tuner")
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
    return _registry_float(_SWEEP_COST_MODEL.get(str(name)), default)


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
