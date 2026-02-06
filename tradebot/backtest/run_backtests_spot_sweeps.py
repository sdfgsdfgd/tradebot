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
import json
import math
import os
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
    SpotLegConfig,
    StrategyConfig,
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
    max_open_trades: int = 2,
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

    strategy = StrategyConfig(
        name="spot_evolve",
        instrument="spot",
        symbol=str(symbol).strip().upper(),
        exchange=None,
        right="PUT",
        entry_days=(0, 1, 2, 3, 4),
        max_entries_per_day=0,
        max_open_trades=int(max_open_trades),
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
_AXIS_ALL_PLAN = (
    "ema",
    "volume",
    "rv",
    "tod",
    "atr",
    "ptsl",
    "hold",
    "orb",
    "regime",
    "regime2",
    "joint",
    "flip_exit",
    "confirm",
    "spread",
    "slope",
    "slope_signed",
    "cooldown",
    "skip_open",
    "shock",
    "risk_overlays",
    "loosen",
    "tick",
    "spot_short_risk_mult",
)
_COMBO_FULL_PLAN = (
    "ema",
    "entry_mode",
    "confirm",
    "weekday",
    "tod",
    "volume",
    "rv",
    "spread",
    "spread_fine",
    "spread_down",
    "slope",
    "slope_signed",
    "cooldown",
    "skip_open",
    "shock",
    "risk_overlays",
    "ptsl",
    "exit_time",
    "hold",
    "spot_short_risk_mult",
    "flip_exit",
    "loosen",
    "loosen_atr",
    "atr",
    "atr_fine",
    "atr_ultra",
    "regime",
    "regime2",
    "regime2_ema",
    "joint",
    "micro_st",
    "orb",
    "hf_scalp",
    "orb_joint",
    "tod_interaction",
    "perm_joint",
    "ema_perm_joint",
    "tick_perm_joint",
    "chop_joint",
    "tick_ema",
    "ema_regime",
    "ema_atr",
    "regime_atr",
    "r2_atr",
    "r2_tod",
    "tick",
    "gate_matrix",
    "squeeze",
    "combo_fast",
    "frontier",
)


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
            if int(strategy.get("max_open_trades") or 0) == 0:
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
        "max_open_trades",
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
def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled spot evolution sweeps (MNQ spot)")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--start", default="2025-01-08")
    parser.add_argument("--end", default="2026-01-08")
    parser.add_argument(
        "--bar-size",
        default="1 hour",
        help="Signal bar size (e.g. '30 mins', '1 hour'). ORB axis uses 15m regardless.",
    )
    parser.add_argument(
        "--spot-exec-bar-size",
        default=None,
        help="Optional execution bar size for spot simulation (e.g. '5 mins'). Signals still run on --bar-size.",
    )
    parser.add_argument("--use-rth", action="store_true", default=False)
    parser.add_argument("--cache-dir", default="db")
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
        help="Select the base strategy shape to start from (champion comes from spot_milestones.json).",
    )
    parser.add_argument("--max-open-trades", type=int, default=2)
    parser.add_argument("--close-eod", action="store_true", default=False)
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
    parser.add_argument("--spot-spread", type=float, default=None, help="Spot spread in price units (e.g. 0.01)")
    parser.add_argument(
        "--spot-commission",
        type=float,
        default=None,
        help="Spot commission per share/contract (price units). (Backtest-only; embedded into fills.)",
    )
    parser.add_argument(
        "--spot-commission-min",
        type=float,
        default=None,
        help="Spot commission minimum per order (price units), e.g. 1.0 for $1.00 min.",
    )
    parser.add_argument(
        "--spot-slippage",
        type=float,
        default=None,
        help="Spot slippage per share (price units). Applied on entry/stop/flip (market-like fills).",
    )
    parser.add_argument(
        "--spot-sizing-mode",
        default=None,
        choices=("fixed", "notional_pct", "risk_pct"),
        help="Spot sizing mode (v2): fixed qty, %% notional, or %% equity risk-to-stop.",
    )
    parser.add_argument("--spot-risk-pct", type=float, default=None, help="Risk per trade as fraction of equity (v2).")
    parser.add_argument(
        "--spot-notional-pct",
        type=float,
        default=None,
        help="Notional allocation per trade as fraction of equity (v2).",
    )
    parser.add_argument(
        "--spot-max-notional-pct",
        type=float,
        default=None,
        help="Max notional per trade as fraction of equity (v2).",
    )
    parser.add_argument("--spot-min-qty", type=int, default=None, help="Min shares per trade (v2).")
    parser.add_argument("--spot-max-qty", type=int, default=None, help="Max shares per trade, 0=none (v2).")
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--top", type=int, default=15)
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
        help="Output path for --write-milestones",
    )
    parser.add_argument("--milestone-min-win", type=float, default=0.55)
    parser.add_argument("--milestone-min-trades", type=int, default=200)
    parser.add_argument("--milestone-min-pnl-dd", type=float, default=8.0)
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
            "(e.g. --axis champ_refine)."
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
        choices=(
            "all",
            "ema",
            "entry_mode",
            "combo_fast",
            "combo_full",
            "squeeze",
            "volume",
            "rv",
            "tod",
            "tod_interaction",
            "perm_joint",
            "weekday",
            "exit_time",
            "atr",
            "atr_fine",
            "atr_ultra",
            "r2_atr",
            "r2_tod",
            "ema_perm_joint",
            "tick_perm_joint",
            "regime_atr",
            "ema_regime",
            "chop_joint",
            "ema_atr",
            "tick_ema",
            "ptsl",
            "hf_scalp",
            "hold",
            "spot_short_risk_mult",
            "orb",
            "orb_joint",
            "frontier",
            "regime",
            "regime2",
            "regime2_ema",
            "joint",
            "micro_st",
            "flip_exit",
            "confirm",
            "spread",
            "spread_fine",
            "spread_down",
            "slope",
            "slope_signed",
            "cooldown",
            "skip_open",
            "shock",
            "risk_overlays",
            "loosen",
            "loosen_atr",
            "tick",
            "gate_matrix",
            "champ_refine",
            "st37_refine",
            "shock_alpha_refine",
            "shock_velocity_refine",
            "shock_velocity_refine_wide",
            "shock_throttle_refine",
            "shock_throttle_tr_ratio",
            "shock_throttle_drawdown",
            "riskpanic_micro",
            "exit_pivot",
        ),
        help="Run one axis sweep (or all in sequence)",
    )
    parser.add_argument(
        "--risk-overlays-riskoff-trs",
        default=None,
        help="Override risk_overlays riskoff TR pct median thresholds (comma-separated floats).",
    )
    parser.add_argument(
        "--risk-overlays-riskpanic-trs",
        default=None,
        help="Override risk_overlays riskpanic TR pct median thresholds (comma-separated floats).",
    )
    parser.add_argument(
        "--risk-overlays-riskpanic-long-factors",
        default=None,
        help="risk_overlays: sweep riskpanic_long_risk_mult_factor (comma-separated floats, e.g. 1,0.8,0.6,0.4).",
    )
    parser.add_argument(
        "--risk-overlays-riskpop-trs",
        default=None,
        help="Override risk_overlays riskpop TR pct median thresholds (comma-separated floats).",
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
    max_open_trades = int(args.max_open_trades)
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

    def _shortlisted_keys(best_by_key: dict, *, top_pnl: int = 8, top_pnl_dd: int = 8) -> list:
        by_pnl = sorted(best_by_key.items(), key=lambda t: _score_row_pnl(t[1]["row"]), reverse=True)[: int(top_pnl)]
        by_dd = sorted(best_by_key.items(), key=lambda t: _score_row_pnl_dd(t[1]["row"]), reverse=True)[
            : int(top_pnl_dd)
        ]
        out = []
        seen = set()
        for key, _ in by_pnl + by_dd:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

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

    def _run_cfg(
        *, cfg: ConfigBundle, bars: list | None = None, regime_bars: list | None = None, regime2_bars: list | None = None
    ) -> dict | None:
        nonlocal run_calls_total
        run_calls_total += 1
        bars_eff, regime_eff, regime2_eff = _context_bars_for_cfg(
            cfg=cfg,
            bars=bars,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
        )
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
            return None
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        roi = float(getattr(s, "roi", 0.0) or 0.0)
        dd_pct = float(getattr(s, "max_drawdown_pct", 0.0) or 0.0)
        return {
            "trades": int(s.trades),
            "win_rate": float(s.win_rate),
            "pnl": pnl,
            "dd": dd,
            "roi": roi,
            "dd_pct": dd_pct,
            "pnl_over_dd": (pnl / dd) if dd > 0 else None,
        }

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

        for cfg, note, meta_item in plan:
            tested += 1
            if progress_label:
                now = pytime.perf_counter()
                hit_report_every = int(report_every) > 0 and (tested % int(report_every) == 0)
                hit_total = total_i is not None and tested == int(total_i)
                hit_heartbeat = float(heartbeat_sec) > 0 and (now - last) >= float(heartbeat_sec)
                if hit_report_every or hit_total or hit_heartbeat:
                    elapsed = now - t0
                    rate = (tested / elapsed) if elapsed > 0 else 0.0
                    if total_i is not None and total_i > 0:
                        remaining = total_i - tested
                        eta_sec = (remaining / rate) if rate > 0 else 0.0
                        pct = (tested / total_i) * 100.0
                        print(
                            f"{progress_label} {tested}/{total_i} ({pct:0.1f}%) kept={len(kept)} "
                            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                            flush=True,
                        )
                    else:
                        print(
                            f"{progress_label} tested={tested} kept={len(kept)} rate={rate:0.2f}/s "
                            f"elapsed={elapsed:0.1f}s",
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

        return tested, kept

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

    def _emit_seed_base_row(
        *,
        cfg_seed: ConfigBundle,
        seed_note: str,
        rows: list[dict],
        base_note: str = "base",
    ) -> None:
        base_row = _run_cfg(cfg=cfg_seed)
        if not base_row:
            return
        note = f"{seed_note} | {base_note}"
        base_row["note"] = note
        _record_milestone(cfg_seed, base_row, note)
        rows.append(base_row)

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
            max_open_trades=max_open_trades,
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

    def _sweep_volume() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                    cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        tod_windows: list[tuple[int | None, int | None, str, dict[str, object]]] = []
        tod_windows.append((None, None, "tod=base", {}))
        tod_windows.append((None, None, "tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}))
        for start_h, end_h, label in (
            (9, 16, "tod=09-16 ET"),
            (10, 15, "tod=10-15 ET"),
            (11, 16, "tod=11-16 ET"),
        ):
            tod_windows.append((start_h, end_h, label, {"entry_start_hour_et": int(start_h), "entry_end_hour_et": int(end_h)}))
        for start_h in (17, 18, 19):
            for end_h in (3, 4, 5):
                label = f"tod={start_h:02d}-{end_h:02d} ET"
                tod_windows.append(
                    (start_h, end_h, label, {"entry_start_hour_et": int(start_h), "entry_end_hour_et": int(end_h)})
                )

        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
        ]
        for s in (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01):
            spread_variants.append((f"spread>={s:.4f}", {"ema_spread_min_pct": float(s)}))

        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
        ]
        for ratio in (1.0, 1.1, 1.2, 1.5):
            for ema_p in (10, 20):
                vol_variants.append((f"vol>={ratio}@{ema_p}", {"volume_ratio_min": float(ratio), "volume_ema_period": int(ema_p)}))

        rows: list[dict] = []
        tested = 0
        total = len(tod_windows) * len(spread_variants) * len(vol_variants)
        t0 = pytime.perf_counter()
        report_every = 200
        for _, _, tod_note, tod_over in tod_windows:
            for spread_note, spread_over in spread_variants:
                for vol_note, vol_over in vol_variants:
                    tested += 1
                    if tested % report_every == 0 or tested == total:
                        elapsed = pytime.perf_counter() - t0
                        rate = (tested / elapsed) if elapsed > 0 else 0.0
                        remaining = total - tested
                        eta_sec = (remaining / rate) if rate > 0 else 0.0
                        pct = (tested / total * 100.0) if total > 0 else 0.0
                        print(
                            f"perm_joint progress {tested}/{total} ({pct:0.1f}%) kept={len(rows)} "
                            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                            flush=True,
                        )
                    overrides: dict[str, object] = {}
                    overrides.update(tod_over)
                    overrides.update(spread_over)
                    overrides.update(vol_over)
                    f = _merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    row = _run_cfg(cfg=cfg)
                    if not row:
                        continue
                    note = f"{tod_note} | {spread_note} | {vol_note}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Permission joint sweep (TOD × spread × volume)", top_n=int(args.top))

    def _sweep_ema_perm_joint() -> None:
        """Joint sweep: EMA preset × (TOD/spread/volume) permission gates."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

        shortlisted = _shortlisted_keys(best_by_ema, top_pnl=5, top_pnl_dd=5)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

        shortlisted = _shortlisted_keys(best_by_tick, top_pnl=8, top_pnl_dd=8)
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
                            elapsed = pytime.perf_counter() - t0
                            rate = (tested / elapsed) if elapsed > 0 else 0.0
                            remaining = total - tested
                            eta_sec = (remaining / rate) if rate > 0 else 0.0
                            pct = (tested / total * 100.0) if total > 0 else 0.0
                            print(
                                f"tick_perm_joint stage2 {tested}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

        shortlisted = _shortlisted_keys(best_by_ema, top_pnl=5, top_pnl_dd=5)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

    def _sweep_atr_exits() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        atr_periods = [7, 10, 14, 21]
        # Include a low-PT pocket (PTx<1.0): this has produced materially higher net PnL post-fix.
        pt_mults = [0.6, 0.8, 0.9, 1.0, 1.5, 2.0]
        sl_mults = [1.0, 1.5, 2.0]
        rows: list[dict] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                    cfg = replace(
                        cfg,
                        strategy=replace(
                            cfg.strategy,
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
                    note = f"ATR({atr_p}) PTx{pt_m} SLx{sl_m}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="C) ATR exits sweep (1h timing + 1d Supertrend)", top_n=int(args.top))

    def _sweep_atr_exits_fine() -> None:
        """Fine-grained ATR exit sweep around the current champion neighborhood."""
        bars_sig = _bars_cached(signal_bar_size)
        # Cover both the risk-adjusted champ neighborhood (ATR 7/10) and the net-PnL pocket (ATR 14/21).
        atr_periods = [7, 10, 14, 21]
        pt_mults = [0.8, 0.9, 1.0, 1.1, 1.2]
        sl_mults = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        rows: list[dict] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                    cfg = replace(
                        cfg,
                        strategy=replace(
                            cfg.strategy,
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
                    note = f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="ATR exits fine sweep (PT/SL multipliers)", top_n=int(args.top))

    def _sweep_atr_exits_ultra() -> None:
        """Ultra-fine ATR exit sweep around the current best PT neighborhood."""
        bars_sig = _bars_cached(signal_bar_size)
        atr_periods = [7]
        pt_mults = [1.05, 1.08, 1.10, 1.12, 1.15]
        sl_mults = [1.35, 1.40, 1.45, 1.50, 1.55]
        rows: list[dict] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                    cfg = replace(
                        cfg,
                        strategy=replace(
                            cfg.strategy,
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
                    note = f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="ATR exits ultra-fine sweep (PT/SL micro-grid)", top_n=int(args.top))

    def _sweep_r2_atr() -> None:
        """Joint interaction hunt: regime2 confirm × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

        ranked_by_pnl = sorted(best_by_r2.items(), key=lambda t: _score_row_pnl(t[1]["row"]), reverse=True)[:8]
        ranked_by_dd = sorted(best_by_r2.items(), key=lambda t: _score_row_pnl_dd(t[1]["row"]), reverse=True)[:8]
        shortlisted_keys = []
        seen: set[tuple] = set()
        for r2_key, _ in ranked_by_pnl + ranked_by_dd:
            if r2_key in seen:
                continue
            seen.add(r2_key)
            shortlisted_keys.append(r2_key)

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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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

        shortlisted = _shortlisted_keys(best_by_r2, top_pnl=10, top_pnl_dd=10)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=_regime_bars_for(cfg),
                            regime2_bars=None,
                        )
                        if not row:
                            continue
                        key = (str(rbar), int(atr_p), float(mult), str(src))
                        current = best_by_regime.get(key)
                        if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                            best_by_regime[key] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_regime, top_pnl=10, top_pnl_dd=10)
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
        Stage 4: expand slower knobs (max_open_trades, spot_close_eod) on a tiny shortlist.
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

        # Stage 1: stacked stop-loss + flip-profit baseline (keep it fast-runner-friendly).
        #
        # Note: v1 used very tight stops (sub-0.6%), which produced many 1y winners but was negative over 10y for
        # every candidate. v2 widens the stop grid + EMA presets to reduce whipsaw and improve decade stability.
        ema_presets = ["3/7", "4/9", "5/13", "8/21", "9/21", "21/50"]
        stop_only_vals = [0.0060, 0.0080, 0.0100, 0.0120, 0.0150, 0.0200]
        flip_hold_vals = [0, 2, 4]
        # Keep stages 1-3 on the fast summary runner path (max_open_trades=1, close_eod=False),
        # then expand the slow knobs on a tiny shortlist at the end.
        stage_fast_max_open = 1
        stage_fast_close_eod = False
        expand_max_open_vals = [1, 2, 3, 5]
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
                                    max_open_trades=int(stage_fast_max_open),
                                    spot_close_eod=bool(stage_fast_close_eod),
                                    spot_short_risk_mult=0.01,
                                    filters=f,
                                    **regime_patch,
                                ),
                            )
                            row = _run_cfg(
                                cfg=cfg,
                                bars=bars_sig,
                                regime_bars=_regime_bars_for(cfg),
                                regime2_bars=None,
                            )
                            if not row:
                                continue
                            note = (
                                f"stacked stop+flip | EMA={ema_preset} confirm=0 | {regime_note} | {perm_note} | "
                                f"tod=9-16 ET skip=0 cd=0 close_eod={int(stage_fast_close_eod)} | "
                                f"SL={sl:.4f} hold={hold} max_open={stage_fast_max_open}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            stage1.append((cfg, row, note))
                            rows.append(row)

        _print_leaderboards(rows, title="HF scalper: stage1 (stacked stop+flip)", top_n=int(args.top))
        _print_top_trades(rows, title="HF scalper: stage1 (stacked stop+flip)", top_n=int(args.top))

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
                                row = _run_cfg(
                                    cfg=cfg,
                                    bars=bars_sig,
                                    regime_bars=_regime_bars_for(cfg),
                                    regime2_bars=None,
                                )
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
                                row = _run_cfg(
                                    cfg=cfg,
                                    bars=bars_sig,
                                    regime_bars=_regime_bars_for(cfg),
                                    regime2_bars=None,
                                )
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

        # Stage 4: expand the slow knobs (max_open_trades / close_eod) on a tiny shortlist.
        if not stage3:
            return
        stage3_hi = [t for t in stage3 if int(t[1].get("trades") or 0) >= int(target_trades)] if target_trades else []
        expand_pool = stage3_hi if stage3_hi else stage3
        shortlisted3 = _shortlist(expand_pool, top_pnl_dd=6, top_pnl=6, top_trades=6)
        print("")
        print(f"HF scalper: expand seeds={len(shortlisted3)} (pool={len(expand_pool)})", flush=True)

        rows4: list[dict] = []
        for seed_cfg, _, seed_note in shortlisted3:
            for max_open in expand_max_open_vals:
                for close_eod in expand_close_eod_vals:
                    cfg = replace(
                        seed_cfg,
                        strategy=replace(
                            seed_cfg.strategy,
                            max_open_trades=int(max_open),
                            spot_close_eod=bool(close_eod),
                        ),
                    )
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=None,
                    )
                    if not row:
                        continue
                    note = f"{seed_note} | expand close_eod={int(close_eod)} max_open={max_open}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows4.append(row)

        _print_leaderboards(rows4, title="HF scalper: expansion (close_eod/max_open)", top_n=int(args.top))
        _print_top_trades(rows4, title="HF scalper: expansion (close_eod/max_open)", top_n=int(args.top))

    def _sweep_hold() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for hold in (0, 1, 2, 3, 4, 6, 8):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, flip_exit_min_hold_bars=int(hold)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        vals = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0]
        rows: list[dict] = []
        for mult in vals:
            cfg = replace(base, strategy=replace(base.strategy, spot_short_risk_mult=float(mult)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_15m, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                                regime_bars=_regime_bars_for(cfg),
                                regime2_bars=_regime2_bars_for(cfg),
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
            regime_bars=_regime_bars_for(base),
            regime2_bars=_regime2_bars_for(base),
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
                                regime_bars=_regime_bars_for(cfg),
                                regime2_bars=_regime2_bars_for(cfg),
                            )
                            if not row:
                                continue
                            orb_key = (str(open_time), int(window_mins), str(target_mode), float(rr), vol_min)
                            best_by_orb[orb_key] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_orb, top_pnl=8, top_pnl_dd=8)
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
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=_regime2_bars_for(cfg),
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
        bars_sig = _bars_cached(signal_bar_size)
        regime_bars_4h = _bars_cached("4 hours")
        if not regime_bars_4h:
            raise SystemExit("No 4 hours regime bars returned (IBKR).")

        rows: list[dict] = []
        regime_bars_by_size = {"4 hours": regime_bars_4h, "1 day": regime_bars_1d}
        regime_bar_sizes = ["4 hours", "1 day"]
        # Include both the newer “micro” ST params and some legacy/wider values we’ve
        # historically tested (e.g. mult=0.8 or 2.0) so the unified sweeps cover them.
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11, 14, 21]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        sources = ["close", "hl2"]
        tested = 0
        total = len(regime_bar_sizes) * len(atr_periods) * len(multipliers) * len(sources)
        t0 = pytime.perf_counter()
        report_every = 100
        for rbar in regime_bar_sizes:
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        tested += 1
                        if tested % report_every == 0 or tested == total:
                            elapsed = pytime.perf_counter() - t0
                            rate = (tested / elapsed) if elapsed > 0 else 0.0
                            remaining = total - tested
                            eta_sec = (remaining / rate) if rate > 0 else 0.0
                            pct = (tested / total * 100.0) if total > 0 else 0.0
                            print(
                                f"regime progress {tested}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                flush=True,
                            )
                        cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                        cfg = replace(
                            cfg,
                            strategy=replace(
                                cfg.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=rbar,
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                            ),
                        )
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=regime_bars_by_size[rbar],
                            regime2_bars=_regime2_bars_for(cfg),
                        )
                        if not row:
                            continue
                        note = f"ST({atr_p},{mult},{src}) @{rbar}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        _print_leaderboards(rows, title="Regime sweep (Supertrend params + timeframe)", top_n=int(args.top))

    def _sweep_regime2() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        regime2_bars_4h = _bars_cached("4 hours")
        if not regime2_bars_4h:
            raise SystemExit("No 4 hours regime2 bars returned (IBKR).")

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11, 14, 21]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        sources = ["close", "hl2"]
        tested = 0
        total = len(atr_periods) * len(multipliers) * len(sources)
        t0 = pytime.perf_counter()
        report_every = 100
        for atr_p in atr_periods:
            for mult in multipliers:
                for src in sources:
                    tested += 1
                    if tested % report_every == 0 or tested == total:
                        elapsed = pytime.perf_counter() - t0
                        rate = (tested / elapsed) if elapsed > 0 else 0.0
                        remaining = total - tested
                        eta_sec = (remaining / rate) if rate > 0 else 0.0
                        pct = (tested / total * 100.0) if total > 0 else 0.0
                        print(
                            f"regime2 progress {tested}/{total} ({pct:0.1f}%) kept={len(rows)} "
                            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                            flush=True,
                        )
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
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=regime2_bars_4h,
                    )
                    if not row:
                        continue
                    note = f"ST2(4h:{atr_p},{mult},{src})"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Dual regime sweep (regime2 Supertrend @ 4h)", top_n=int(args.top))

    def _sweep_regime2_ema() -> None:
        """Confirm layer: EMA trend gate on a higher timeframe (4h/1d)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"confirm={confirm}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Confirm-bars sweep (quality gate)", top_n=int(args.top))

    def _sweep_spread() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for spread in (None, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
            f = _mk_filters(ema_spread_min_pct=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            spread_note = "-" if spread is None else f"spread>={spread}"
            row["note"] = spread_note
            _record_milestone(cfg, row, spread_note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread sweep (quality gate)", top_n=int(args.top))

    def _sweep_spread_fine() -> None:
        """Fine-grained sweep around the current champion spread gate."""
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        spreads = [None, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]
        for spread in spreads:
            f = _mk_filters(ema_spread_min_pct=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            spread_note = "-" if spread is None else f"spread>={float(spread):.4f}"
            row["note"] = spread_note
            _record_milestone(cfg, row, spread_note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread fine sweep (quality gate)", top_n=int(args.top))

    def _sweep_spread_down() -> None:
        """Directional permission: sweep stricter EMA spread gate for down entries only."""
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        spreads = [None, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.02, 0.03, 0.05]
        for spread in spreads:
            f = _mk_filters(ema_spread_min_pct_down=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = "-" if spread is None else f"spread_down>={float(spread):.4f}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread DOWN sweep (directional permission)", top_n=int(args.top))

    def _sweep_slope() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for slope in (None, 0.005, 0.01, 0.02, 0.03, 0.05):
            f = _mk_filters(ema_slope_min_pct=float(slope) if slope is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
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
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        modes = ["detect", "block", "block_longs", "block_shorts", "surf"]
        dir_variants = [("regime", 2, "dir=regime@2"), ("signal", 1, "dir=signal@1")]
        sl_mults = [1.0, 0.75]
        pt_mults = [1.0, 0.75]
        short_risk_factors = [1.0, 0.5]

        ratio_presets: list[tuple[str, dict[str, object], str]] = []
        for detector in ("atr_ratio", "tr_ratio"):
            for fast, slow, on, off, min_pct in (
                (5, 30, 1.35, 1.20, 6.0),
                (7, 50, 1.55, 1.30, 7.0),
                (10, 80, 1.45, 1.25, 7.0),
                (14, 120, 1.35, 1.20, 9.0),
                (7, 30, 1.70, 1.40, 7.0),
            ):
                ratio_presets.append(
                    (
                        detector,
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

        daily_atr_presets: list[tuple[str, dict[str, object], str]] = []
        for period, on_atr, off_atr, tr_on in (
            (14, 13.0, 11.0, None),
            (14, 13.5, 13.0, None),
            (14, 14.0, 13.0, None),
            (14, 14.0, 13.0, 9.0),
            (10, 13.0, 11.0, 9.0),
            (21, 14.0, 13.0, 10.0),
        ):
            daily_atr_presets.append(
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

        drawdown_presets: list[tuple[str, dict[str, object], str]] = []
        for lb, dd_on, dd_off in (
            (10, -15.0, -8.0),
            (20, -20.0, -10.0),
            (20, -25.0, -15.0),
            (30, -25.0, -15.0),
            (60, -30.0, -20.0),
        ):
            drawdown_presets.append(
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

        presets = ratio_presets + daily_atr_presets + drawdown_presets
        rows: list[dict] = []
        tested = 0
        total = len(modes) * len(dir_variants) * len(sl_mults) * len(pt_mults) * len(short_risk_factors) * len(presets)
        t0 = pytime.perf_counter()
        report_every = 50
        for detector, params, det_note in presets:
            for mode in modes:
                for dir_src, dir_lb, dir_note in dir_variants:
                    for sl_mult in sl_mults:
                        for pt_mult in pt_mults:
                            for short_factor in short_risk_factors:
                                tested += 1
                                if tested % report_every == 0 or tested == total:
                                    elapsed = pytime.perf_counter() - t0
                                    rate = (tested / elapsed) if elapsed > 0 else 0.0
                                    remaining = total - tested
                                    eta_sec = (remaining / rate) if rate > 0 else 0.0
                                    pct = (tested / total * 100.0) if total > 0 else 0.0
                                    print(
                                        f"shock progress {tested}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                        f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                        flush=True,
                                    )

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
                                row = _run_cfg(cfg=cfg)
                                if not row:
                                    continue
                                note = (
                                    f"shock={mode} {det_note} | {dir_note} | "
                                    f"sl_mult={sl_mult:g} pt_mult={pt_mult:g} short_factor={short_factor:g}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                rows.append(row)

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

        if args.risk_overlays_worker is not None:
            if not offline:
                raise SystemExit("risk_overlays worker mode requires --offline (avoid parallel IBKR sessions).")
            out_path_raw = str(args.risk_overlays_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--risk-overlays-out is required for risk_overlays worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.risk_overlays_worker,
                args.risk_overlays_workers,
                label="risk_overlays",
            )

            local_total = (total // workers) + (1 if worker_id < (total % workers) else 0)
            tested = 0
            combo_idx = 0
            report_every = 50
            t0 = pytime.perf_counter()
            records: list[dict] = []

            def _progress(label: str) -> None:
                elapsed = pytime.perf_counter() - t0
                rate = (tested / elapsed) if elapsed > 0 else 0.0
                remaining = local_total - tested
                eta_sec = (remaining / rate) if rate > 0 else 0.0
                pct = (tested / local_total * 100.0) if local_total > 0 else 0.0
                print(
                    f"risk_overlays worker {worker_id+1}/{workers} {label} "
                    f"{tested}/{local_total} ({pct:0.1f}%) kept={len(records)} "
                    f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                    flush=True,
                )

            for tr_med in riskoff_trs:
                for lb in riskoff_lbs:
                    for mode in riskoff_modes:
                        for cutoff in riskoff_cutoffs_et:
                            assigned = (combo_idx % workers) == worker_id
                            combo_idx += 1
                            if not assigned:
                                continue
                            tested += 1
                            if tested % report_every == 0 or tested == local_total:
                                _progress("riskoff")

                            overrides = {
                                "riskoff_tr5_med_pct": float(tr_med),
                                "riskoff_tr5_lookback_days": int(lb),
                                "riskoff_mode": str(mode),
                                "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                "riskpanic_tr5_med_pct": None,
                                "riskpanic_neg_gap_ratio_min": None,
                            }
                            f = _mk_filters(overrides=overrides)
                            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                            row = _run_cfg(cfg=cfg)
                            if not row:
                                continue
                            cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                            note = f"riskoff TRmed{lb}>={tr_med:g} mode={mode} {cut_note}"
                            records.append({"overrides": overrides, "note": note, "row": row})

            for tr_med in panic_trs:
                for neg_ratio in neg_ratios:
                    for lb in panic_lbs:
                        for long_factor in panic_long_factors:
                            for short_factor in panic_short_factors:
                                for cutoff in panic_cutoffs_et:
                                    for abs_gap in panic_neg_gap_abs_pcts:
                                        for tr_delta_min, tr_delta_lb, tr_delta_note in panic_tr_delta_variants:
                                            assigned = (combo_idx % workers) == worker_id
                                            combo_idx += 1
                                            if not assigned:
                                                continue
                                            tested += 1
                                            if tested % report_every == 0 or tested == local_total:
                                                _progress("riskpanic")

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
                                            # v39-style: pre-panic continuous scaling (requires TR-velocity gate + long shrink).
                                            if (
                                                long_factor is not None
                                                and float(long_factor) < 1.0
                                                and tr_delta_min is not None
                                            ):
                                                overrides["riskpanic_long_scale_mode"] = "linear"
                                            f = _mk_filters(overrides=overrides)
                                            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                            row = _run_cfg(cfg=cfg)
                                            if not row:
                                                continue
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
                                            records.append({"overrides": overrides, "note": note, "row": row})

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
                                                    assigned = (combo_idx % workers) == worker_id
                                                    combo_idx += 1
                                                    if not assigned:
                                                        continue
                                                    tested += 1
                                                    if tested % report_every == 0 or tested == local_total:
                                                        _progress("riskpop")

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
                                                    f = _mk_filters(overrides=overrides)
                                                    cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                                    row = _run_cfg(cfg=cfg)
                                                    if not row:
                                                        continue
                                                    cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                                                    gap_note = (
                                                        "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                                                    )
                                                    note = (
                                                        f"riskpop TRmed{lb}>={tr_med:g} pos_gap>={pos_ratio:g} {gap_note} "
                                                        f"{tr_delta_note} mode={mode} long_factor={long_factor:g} "
                                                        f"short_factor={short_factor:g} {cut_note}"
                                                    )
                                                    records.append({"overrides": overrides, "note": note, "row": row})

            if combo_idx != total:
                raise SystemExit(f"risk_overlays worker internal error: combos={combo_idx} expected={total}")

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"risk_overlays worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []

        tested_total = 0
        if jobs > 1 and total > 0:
            if not offline:
                raise SystemExit("--jobs>1 for risk_overlays requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--risk-overlays-worker",
                    "--risk-overlays-workers",
                    "--risk-overlays-out",
                    "--risk-overlays-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(total)) if total > 0 else 1
            print(f"risk_overlays parallel: workers={jobs_eff} total={total}", flush=True)

            payloads = _run_parallel_json_worker_plan(
                jobs_eff=jobs_eff,
                tmp_prefix="tradebot_risk_overlays_",
                worker_tag="ro",
                out_prefix="risk_overlays_out",
                build_cmd=lambda worker_id, workers_n, out_path: [
                    sys.executable,
                    "-u",
                    "-m",
                    "tradebot.backtest",
                    "spot",
                    *base_cli,
                    "--axis",
                    "risk_overlays",
                    "--jobs",
                    "1",
                    "--risk-overlays-worker",
                    str(worker_id),
                    "--risk-overlays-workers",
                    str(workers_n),
                    "--risk-overlays-out",
                    str(out_path),
                    "--risk-overlays-run-min-trades",
                    str(int(run_min_trades)),
                ],
                capture_error="Failed to capture risk_overlays worker stdout.",
                failure_label="risk_overlays worker",
                missing_label="risk_overlays",
                invalid_label="risk_overlays",
            )

            for worker_id in range(jobs_eff):
                payload = payloads.get(int(worker_id))
                if not isinstance(payload, dict):
                    continue
                tested_total += int(payload.get("tested") or 0)
                for rec in payload.get("records") or []:
                    if not isinstance(rec, dict):
                        continue
                    overrides = rec.get("overrides")
                    note = rec.get("note")
                    row = rec.get("row")
                    if not isinstance(overrides, dict) or not isinstance(note, str) or not isinstance(row, dict):
                        continue
                    row = dict(row)
                    f = _mk_filters(overrides=overrides)
                    cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

            run_calls_total += int(tested_total)
        else:
            tested = 0
            t0 = pytime.perf_counter()
            report_every = 50
            for tr_med in riskoff_trs:
                for lb in riskoff_lbs:
                    for mode in riskoff_modes:
                        for cutoff in riskoff_cutoffs_et:
                            tested += 1
                            if tested % report_every == 0 or tested == riskoff_total:
                                elapsed = pytime.perf_counter() - t0
                                rate = (tested / elapsed) if elapsed > 0 else 0.0
                                remaining = riskoff_total - tested
                                eta_sec = (remaining / rate) if rate > 0 else 0.0
                                pct = (tested / riskoff_total * 100.0) if riskoff_total > 0 else 0.0
                                print(
                                    f"riskoff progress {tested}/{riskoff_total} ({pct:0.1f}%) kept={len(rows)} "
                                    f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                    flush=True,
                                )

                            f = _mk_filters(
                                overrides={
                                    "riskoff_tr5_med_pct": float(tr_med),
                                    "riskoff_tr5_lookback_days": int(lb),
                                    "riskoff_mode": str(mode),
                                    "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None,
                                    "riskpanic_tr5_med_pct": None,
                                    "riskpanic_neg_gap_ratio_min": None,
                                }
                            )
                            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                            row = _run_cfg(cfg=cfg)
                            if not row:
                                continue
                            cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                            note = f"riskoff TRmed{lb}>={tr_med:g} mode={mode} {cut_note}"
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

            tested = 0
            t0 = pytime.perf_counter()
            for tr_med in panic_trs:
                for neg_ratio in neg_ratios:
                    for lb in panic_lbs:
                        for long_factor in panic_long_factors:
                            for short_factor in panic_short_factors:
                                for cutoff in panic_cutoffs_et:
                                    for abs_gap in panic_neg_gap_abs_pcts:
                                        for tr_delta_min, tr_delta_lb, tr_delta_note in panic_tr_delta_variants:
                                            tested += 1
                                            if tested % report_every == 0 or tested == panic_total:
                                                elapsed = pytime.perf_counter() - t0
                                                rate = (tested / elapsed) if elapsed > 0 else 0.0
                                                remaining = panic_total - tested
                                                eta_sec = (remaining / rate) if rate > 0 else 0.0
                                                pct = (tested / panic_total * 100.0) if panic_total > 0 else 0.0
                                                print(
                                                    f"riskpanic progress {tested}/{panic_total} ({pct:0.1f}%) kept={len(rows)} "
                                                    f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                                    flush=True,
                                                )

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
                                            # v39-style: pre-panic continuous scaling (requires TR-velocity gate + long shrink).
                                            if (
                                                long_factor is not None
                                                and float(long_factor) < 1.0
                                                and tr_delta_min is not None
                                            ):
                                                overrides["riskpanic_long_scale_mode"] = "linear"
                                            f = _mk_filters(overrides=overrides)
                                            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                            row = _run_cfg(cfg=cfg)
                                            if not row:
                                                continue
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
                                            row["note"] = note
                                            _record_milestone(cfg, row, note)
                                            rows.append(row)

            if not skip_pop:
                tested = 0
                t0 = pytime.perf_counter()
                for tr_med in pop_trs:
                    for pos_ratio in pos_ratios:
                        for lb in pop_lbs:
                            for long_factor in pop_long_factors:
                                for short_factor in pop_short_factors:
                                    for cutoff in pop_cutoffs_et:
                                        for mode in pop_modes:
                                            for abs_gap in pop_pos_gap_abs_pcts:
                                                for tr_delta_min, tr_delta_lb, tr_delta_note in pop_tr_delta_variants:
                                                    tested += 1
                                                    if tested % report_every == 0 or tested == pop_total:
                                                        elapsed = pytime.perf_counter() - t0
                                                        rate = (tested / elapsed) if elapsed > 0 else 0.0
                                                        remaining = pop_total - tested
                                                        eta_sec = (remaining / rate) if rate > 0 else 0.0
                                                        pct = (tested / pop_total * 100.0) if pop_total > 0 else 0.0
                                                        print(
                                                            f"riskpop progress {tested}/{pop_total} ({pct:0.1f}%) kept={len(rows)} "
                                                            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                                            flush=True,
                                                        )

                                                    f = _mk_filters(
                                                        overrides={
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
                                                    )
                                                    cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                                                    row = _run_cfg(cfg=cfg)
                                                    if not row:
                                                        continue
                                                    cut_note = "-" if cutoff is None else f"cutoff<{cutoff:02d} ET"
                                                    gap_note = (
                                                        "-" if abs_gap is None else f"|gap|>={abs_gap*100:0.0f}%"
                                                    )
                                                    note = (
                                                        f"riskpop TRmed{lb}>={tr_med:g} pos_gap>={pos_ratio:g} {gap_note} "
                                                        f"{tr_delta_note} mode={mode} long_factor={long_factor:g} "
                                                        f"short_factor={short_factor:g} {cut_note}"
                                                    )
                                                    row["note"] = note
                                                    _record_milestone(cfg, row, note)
                                                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="TR% risk overlay sweep (riskoff + riskpanic + riskpop)", top_n=int(args.top))

    def _sweep_loosen() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for max_open in (1, 2, 3, 0):
            for close_eod in (False, True):
                cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                cfg = replace(
                    cfg,
                    strategy=replace(
                        cfg.strategy,
                        max_open_trades=int(max_open),
                        spot_close_eod=bool(close_eod),
                    ),
                )
                row = _run_cfg(
                    cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
                )
                if not row:
                    continue
                note = f"max_open={max_open} close_eod={int(close_eod)}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="Loosenings sweep (stacking + EOD exit)", top_n=int(args.top))

    def _sweep_loosen_atr() -> None:
        """Interaction hunt: stacking (max_open) × ATR exits (includes PTx < 1.0 pocket)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Keep the grid tight around the post-fix high-PnL neighborhood.
        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.65, 0.7, 0.75, 0.8]
        sl_mults = [1.2, 1.4, 1.6, 1.8, 2.0]
        max_open_vals = [2, 3, 0]
        close_eod_vals = [False, True]

        rows: list[dict] = []
        for max_open in max_open_vals:
            for close_eod in close_eod_vals:
                for atr_p in atr_periods:
                    for pt_m in pt_mults:
                        for sl_m in sl_mults:
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    max_open_trades=int(max_open),
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
                            note = (
                                f"max_open={max_open} close_eod={int(close_eod)} | "
                                f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Loosen × ATR joint sweep (stacking × exits)", top_n=int(args.top))

    def _sweep_tick() -> None:
        """Permission layer: Raschke-style $TICK width gate (daily, RTH only)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
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
        - loosenings (stacking + EOD close)
        - optional regime2 confirm (small curated set)
        - a small set of quality gates (spread/slope/TOD/rv/exit-time/tick)
        """
        nonlocal run_calls_total
        bars_sig = _bars_cached(signal_bar_size)

        regime_bars_4h = _bars_cached("4 hours")
        if not regime_bars_4h:
            raise SystemExit("No 4 hours regime bars returned (IBKR).")

        regime_bars_by_size = {"4 hours": regime_bars_4h, "1 day": regime_bars_1d}

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

        # Keep this small; we already have a dedicated loosenings axis. Here we only try
        # a few representative "stacking vs risk trimming" variants.
        loosen_variants: list[tuple[int, bool, str]] = [
            (1, False, "max_open=1 close_eod=0"),
            (2, False, "max_open=2 close_eod=0"),
            (1, True, "max_open=1 close_eod=1"),
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
            max_open: int,
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
                    max_open_trades=int(max_open),
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

        def _cfg_from_payload(strategy_payload, filters_payload) -> ConfigBundle | None:
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

        def _build_stage2_plan(shortlist_local: list[tuple[ConfigBundle, str]]) -> list[tuple[ConfigBundle, str, dict]]:
            plan: list[tuple[ConfigBundle, str, dict]] = []
            for base_idx, (base_cfg, base_note) in enumerate(shortlist_local):
                for exit_idx, (exit_over, exit_note) in enumerate(exit_variants):
                    for hold in hold_vals:
                        for loosen_idx, (max_open, close_eod, loose_note) in enumerate(loosen_variants):
                            for r2_idx, (r2_over, r2_note) in enumerate(regime2_variants):
                                cfg, note = _mk_stage2_cfg(
                                    base_cfg,
                                    base_note,
                                    exit_over=exit_over,
                                    exit_note=exit_note,
                                    hold=int(hold),
                                    max_open=int(max_open),
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
            if not offline:
                raise SystemExit("combo_fast stage2 worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.combo_fast_stage2))
            out_path_raw = str(args.combo_fast_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--combo-fast-out is required for combo_fast stage2 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.combo_fast_worker,
                args.combo_fast_workers,
                label="combo_fast",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid combo_fast stage2 payload JSON: {payload_path}") from exc
            raw_shortlist = payload.get("shortlist") if isinstance(payload, dict) else None
            if not isinstance(raw_shortlist, list):
                raise SystemExit(f"combo_fast stage2 payload missing 'shortlist' list: {payload_path}")

            shortlist_local: list[tuple[ConfigBundle, str]] = []
            for item in raw_shortlist:
                if not isinstance(item, dict):
                    continue
                strat_payload = item.get("strategy")
                filters_payload = item.get("filters")
                base_note = str(item.get("base_note") or "")
                base_cfg = _cfg_from_payload(strat_payload, filters_payload)
                if base_cfg is None:
                    continue
                shortlist_local.append((base_cfg, base_note))

            stage2_plan_all = _build_stage2_plan(shortlist_local)
            stage2_total = len(stage2_plan_all)
            local_total = (stage2_total // workers) + (1 if worker_id < (stage2_total % workers) else 0)
            shard_plan = (
                item for combo_idx, item in enumerate(stage2_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"combo_fast stage2 worker {worker_id+1}/{workers}",
                report_every=100,
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

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"combo_fast stage2 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
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
            if not offline:
                raise SystemExit("combo_fast stage3 worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.combo_fast_stage3))
            out_path_raw = str(args.combo_fast_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--combo-fast-out is required for combo_fast stage3 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.combo_fast_worker,
                args.combo_fast_workers,
                label="combo_fast",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid combo_fast stage3 payload JSON: {payload_path}") from exc
            raw_bases = payload.get("bases") if isinstance(payload, dict) else None
            if not isinstance(raw_bases, list):
                raise SystemExit(f"combo_fast stage3 payload missing 'bases' list: {payload_path}")

            bases_local: list[tuple[ConfigBundle, str]] = []
            for item in raw_bases:
                if not isinstance(item, dict):
                    continue
                strat_payload = item.get("strategy")
                filters_payload = item.get("filters")
                base_note = str(item.get("base_note") or "")
                base_cfg = _cfg_from_payload(strat_payload, filters_payload)
                if base_cfg is None:
                    continue
                bases_local.append((base_cfg, base_note))

            stage3_plan_all = _build_stage3_plan(bases_local)
            stage3_total = len(stage3_plan_all)
            local_total = (stage3_total // workers) + (1 if worker_id < (stage3_total % workers) else 0)
            shard_plan = (
                item for combo_idx, item in enumerate(stage3_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"combo_fast stage3 worker {worker_id+1}/{workers}",
                report_every=200,
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

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"combo_fast stage3 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        def _ranked(items: list[tuple[ConfigBundle, dict, str]], *, top_pnl_dd: int, top_pnl: int) -> list:
            by_dd = sorted(items, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[: int(top_pnl_dd)]
            by_pnl = sorted(items, key=lambda t: _score_row_pnl(t[1]), reverse=True)[: int(top_pnl)]
            seen: set[str] = set()
            out: list[tuple[ConfigBundle, dict, str]] = []
            for cfg, row, note in by_dd + by_pnl:
                key = _milestone_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                out.append((cfg, row, note))
            return out

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
            if not offline:
                raise SystemExit("combo_fast stage1 worker mode requires --offline (avoid parallel IBKR sessions).")
            out_path_raw = str(args.combo_fast_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--combo-fast-out is required for combo_fast stage1 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.combo_fast_worker,
                args.combo_fast_workers,
                label="combo_fast",
            )

            local_total = (stage1_total // workers) + (1 if worker_id < (stage1_total % workers) else 0)
            shard_plan = (
                item for combo_idx, item in enumerate(stage1_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"combo_fast stage1 worker {worker_id+1}/{workers}",
                report_every=200,
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

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"combo_fast stage1 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        stage1_tested = 0
        report_every_stage1 = 200
        print(f"combo_fast sweep: stage1 total={stage1_total} (progress every {report_every_stage1})", flush=True)
        if jobs > 1 and stage1_total > 0:
            if not offline:
                raise SystemExit("--jobs>1 for combo_fast requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(stage1_total)) if stage1_total > 0 else 1
            print(f"combo_fast stage1 parallel: workers={jobs_eff} total={stage1_total}", flush=True)

            payloads = _run_parallel_json_worker_plan(
                jobs_eff=jobs_eff,
                tmp_prefix="tradebot_combo_fast1_",
                worker_tag="cf1",
                out_prefix="stage1_out",
                build_cmd=lambda worker_id, workers_n, out_path: [
                    sys.executable,
                    "-u",
                    "-m",
                    "tradebot.backtest",
                    "spot",
                    *base_cli,
                    "--axis",
                    "combo_fast",
                    "--jobs",
                    "1",
                    "--combo-fast-stage1",
                    "1",
                    "--combo-fast-worker",
                    str(worker_id),
                    "--combo-fast-workers",
                    str(workers_n),
                    "--combo-fast-out",
                    str(out_path),
                    "--combo-fast-run-min-trades",
                    str(int(run_min_trades)),
                ],
                capture_error="Failed to capture combo_fast stage1 worker stdout.",
                failure_label="combo_fast stage1 worker",
                missing_label="combo_fast stage1",
                invalid_label="combo_fast stage1",
            )

            tested_total = 0
            for worker_id in range(jobs_eff):
                payload = payloads.get(int(worker_id))
                if not isinstance(payload, dict):
                    continue
                tested_total += int(payload.get("tested") or 0)
                for rec in payload.get("records") or []:
                    if not isinstance(rec, dict):
                        continue
                    cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                    if cfg is None:
                        continue
                    row = rec.get("row")
                    if not isinstance(row, dict):
                        continue
                    row = dict(row)
                    note = str(rec.get("note") or "").strip() or "combo_fast stage1"
                    row["note"] = note
                    stage1.append((cfg, row, note))

            stage1_tested = int(tested_total)
            run_calls_total += int(tested_total)
        else:
            stage1_tested, stage1_kept = _run_sweep(
                plan=stage1_plan_all,
                bars=bars_sig,
                total=stage1_total,
                progress_label="combo_fast sweep: stage1",
                report_every=report_every_stage1,
                record_milestones=False,
            )
            for cfg, row, note, _meta in stage1_kept:
                stage1.append((cfg, row, note))

        shortlist = _ranked(stage1, top_pnl_dd=15, top_pnl=7)
        print("")
        print(f"combo_fast sweep: shortlist regimes={len(shortlist)} (from stage1={len(stage1)})")

        # Stage 2: for each shortlisted regime, sweep exits + loosenings, and (optionally) a small regime2 set.
        stage2: list[tuple[ConfigBundle, dict, str]] = []
        report_every = 200
        stage2_plan_all = _build_stage2_plan([(cfg, note) for cfg, _row, note in shortlist])
        stage2_total = len(stage2_plan_all)
        print(f"combo_fast sweep: stage2 total={stage2_total} (progress every {report_every})", flush=True)
        tested = 0
        if jobs > 1:
            if not offline:
                raise SystemExit("--jobs>1 for combo_fast requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(stage2_total)) if stage2_total > 0 else 1
            print(f"combo_fast stage2 parallel: workers={jobs_eff} total={stage2_total}", flush=True)

            with tempfile.TemporaryDirectory(prefix="tradebot_combo_fast_") as tmpdir:
                tmp_root = Path(tmpdir)
                payload_path = tmp_root / "stage2_payload.json"
                shortlist_payload: list[dict] = []
                for base_cfg, _, base_note in shortlist:
                    shortlist_payload.append(
                        {
                            "strategy": _spot_strategy_payload(base_cfg, meta=meta),
                            "filters": _filters_payload(base_cfg.strategy.filters),
                            "base_note": str(base_note),
                        }
                    )
                write_json(payload_path, {"shortlist": shortlist_payload}, sort_keys=False)

                payloads = _run_parallel_json_worker_plan(
                    jobs_eff=jobs_eff,
                    tmp_prefix="tradebot_combo_fast2_",
                    worker_tag="cf2",
                    out_prefix="stage2_out",
                    build_cmd=lambda worker_id, workers_n, out_path: [
                        sys.executable,
                        "-u",
                        "-m",
                        "tradebot.backtest",
                        "spot",
                        *base_cli,
                        "--axis",
                        "combo_fast",
                        "--jobs",
                        "1",
                        "--combo-fast-stage2",
                        str(payload_path),
                        "--combo-fast-worker",
                        str(worker_id),
                        "--combo-fast-workers",
                        str(workers_n),
                        "--combo-fast-out",
                        str(out_path),
                        "--combo-fast-run-min-trades",
                        str(int(run_min_trades)),
                    ],
                    capture_error="Failed to capture combo_fast stage2 worker stdout.",
                    failure_label="combo_fast stage2 worker",
                    missing_label="combo_fast stage2",
                    invalid_label="combo_fast stage2",
                )

                tested_total = 0
                for worker_id in range(jobs_eff):
                    payload = payloads.get(int(worker_id))
                    if not isinstance(payload, dict):
                        continue
                    tested_total += int(payload.get("tested") or 0)
                    for rec in payload.get("records") or []:
                        if not isinstance(rec, dict):
                            continue
                        cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                        if cfg is None:
                            continue
                        row = rec.get("row")
                        if not isinstance(row, dict):
                            continue
                        row = dict(row)
                        note = str(rec.get("note") or "").strip() or "combo_fast stage2"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        stage2.append((cfg, row, note))

                tested = int(tested_total)
                run_calls_total += int(tested_total)
        else:
            tested, stage2_kept = _run_sweep(
                plan=stage2_plan_all,
                bars=bars_sig,
                total=stage2_total,
                progress_label="combo_fast sweep: stage2",
                report_every=report_every,
            )
            for cfg, row, note, _meta in stage2_kept:
                stage2.append((cfg, row, note))

        print(f"combo_fast sweep: stage2 tested={tested} kept={len(stage2)} (min_trades={run_min_trades})")

        # Stage 3: apply a small set of quality gates on the top stage2 candidates.
        top_stage2 = _ranked(stage2, top_pnl_dd=15, top_pnl=7)

        stage3_plan_all = _build_stage3_plan([(cfg, base_note) for cfg, _row, base_note in top_stage2])
        stage3_total = len(stage3_plan_all)
        stage3_tested = 0
        report_every_stage3 = 200
        print(f"combo_fast sweep: stage3 total={stage3_total} (progress every {report_every_stage3})", flush=True)

        stage3: list[dict] = []
        if jobs > 1 and stage3_total > 0:
            if not offline:
                raise SystemExit("--jobs>1 for combo_fast requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--combo-fast-stage1",
                    "--combo-fast-stage2",
                    "--combo-fast-stage3",
                    "--combo-fast-worker",
                    "--combo-fast-workers",
                    "--combo-fast-out",
                    "--combo-fast-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(stage3_total)) if stage3_total > 0 else 1
            print(f"combo_fast stage3 parallel: workers={jobs_eff} total={stage3_total}", flush=True)

            with tempfile.TemporaryDirectory(prefix="tradebot_combo_fast3_") as tmpdir:
                tmp_root = Path(tmpdir)
                payload_path = tmp_root / "stage3_payload.json"
                bases_payload: list[dict] = []
                for base_cfg, _, base_note in top_stage2:
                    bases_payload.append(
                        {
                            "strategy": _spot_strategy_payload(base_cfg, meta=meta),
                            "filters": _filters_payload(base_cfg.strategy.filters),
                            "base_note": str(base_note),
                        }
                    )
                write_json(payload_path, {"bases": bases_payload}, sort_keys=False)

                payloads = _run_parallel_json_worker_plan(
                    jobs_eff=jobs_eff,
                    tmp_prefix="tradebot_combo_fast3_",
                    worker_tag="cf3",
                    out_prefix="stage3_out",
                    build_cmd=lambda worker_id, workers_n, out_path: [
                        sys.executable,
                        "-u",
                        "-m",
                        "tradebot.backtest",
                        "spot",
                        *base_cli,
                        "--axis",
                        "combo_fast",
                        "--jobs",
                        "1",
                        "--combo-fast-stage3",
                        str(payload_path),
                        "--combo-fast-worker",
                        str(worker_id),
                        "--combo-fast-workers",
                        str(workers_n),
                        "--combo-fast-out",
                        str(out_path),
                        "--combo-fast-run-min-trades",
                        str(int(run_min_trades)),
                    ],
                    capture_error="Failed to capture combo_fast stage3 worker stdout.",
                    failure_label="combo_fast stage3 worker",
                    missing_label="combo_fast stage3",
                    invalid_label="combo_fast stage3",
                )

                tested_total = 0
                for worker_id in range(jobs_eff):
                    payload = payloads.get(int(worker_id))
                    if not isinstance(payload, dict):
                        continue
                    tested_total += int(payload.get("tested") or 0)
                    for rec in payload.get("records") or []:
                        if not isinstance(rec, dict):
                            continue
                        cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                        if cfg is None:
                            continue
                        row = rec.get("row")
                        if not isinstance(row, dict):
                            continue
                        row = dict(row)
                        note = str(rec.get("note") or "").strip() or "combo_fast stage3"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        stage3.append(row)

                stage3_tested = int(tested_total)
                run_calls_total += int(tested_total)
        else:
            stage3_tested, stage3_kept = _run_sweep(
                plan=stage3_plan_all,
                bars=bars_sig,
                total=stage3_total,
                progress_label="combo_fast sweep: stage3",
                report_every=report_every_stage3,
            )
            for _cfg, row, _note, _meta in stage3_kept:
                stage3.append(row)

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

        if jobs > 1:
            if not offline:
                raise SystemExit("--jobs>1 for combo_full requires --offline (avoid parallel IBKR sessions).")

            axes = tuple(_COMBO_FULL_PLAN)

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--merge-milestones",),
                flags_with_values=("--axis", "--jobs", "--milestones-out"),
            )

            milestone_payloads = _run_axis_subprocess_plan(
                label="combo_full parallel",
                axes=axes,
                jobs=int(jobs),
                base_cli=base_cli,
                axis_jobs_resolver=lambda axis_name: min(int(jobs), int(_default_jobs()))
                if str(axis_name) in ("gate_matrix", "combo_fast", "risk_overlays")
                else 1,
                write_milestones=bool(args.write_milestones),
                tmp_prefix="tradebot_combo_full_",
            )
            if bool(args.write_milestones):
                eligible_new: list[dict] = []
                for axis_name in axes:
                    payload = milestone_payloads.get(axis_name)
                    if isinstance(payload, dict):
                        eligible_new.extend(_collect_milestone_items_from_payload(payload, symbol=symbol))
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
                milestones_written = True
                print(f"Wrote {out_path} ({total} eligible presets).", flush=True)

            return

        def _run_axis(label: str, fn, *, total: int | None = None) -> None:
            before_kept = len(milestone_rows)
            before_calls = int(run_calls_total)
            t0 = pytime.perf_counter()
            total_label = str(int(total)) if total is not None else "?"
            print(f"START {label} total={total_label}", flush=True)
            fn()
            elapsed = pytime.perf_counter() - t0
            tested = int(run_calls_total) - int(before_calls)
            kept = len(milestone_rows) - before_kept
            print(f"DONE  {label} tested={tested} kept={kept} elapsed={elapsed:0.1f}s", flush=True)
            print("", flush=True)

        combo_full_registry = {
            "ema": _sweep_ema,
            "entry_mode": _sweep_entry_mode,
            "confirm": _sweep_confirm,
            "weekday": _sweep_weekdays,
            "tod": _sweep_tod,
            "volume": _sweep_volume,
            "rv": _sweep_rv,
            "spread": _sweep_spread,
            "spread_fine": _sweep_spread_fine,
            "spread_down": _sweep_spread_down,
            "slope": _sweep_slope,
            "slope_signed": _sweep_slope_signed,
            "cooldown": _sweep_cooldown,
            "skip_open": _sweep_skip_open,
            "shock": _sweep_shock,
            "risk_overlays": _sweep_risk_overlays,
            "ptsl": _sweep_ptsl,
            "exit_time": _sweep_exit_time,
            "hold": _sweep_hold,
            "spot_short_risk_mult": _sweep_spot_short_risk_mult,
            "flip_exit": _sweep_flip_exit,
            "loosen": _sweep_loosen,
            "loosen_atr": _sweep_loosen_atr,
            "atr": _sweep_atr_exits,
            "atr_fine": _sweep_atr_exits_fine,
            "atr_ultra": _sweep_atr_exits_ultra,
            "regime": _sweep_regime,
            "regime2": _sweep_regime2,
            "regime2_ema": _sweep_regime2_ema,
            "joint": _sweep_joint,
            "micro_st": _sweep_micro_st,
            "orb": _sweep_orb,
            "hf_scalp": _sweep_hf_scalp,
            "orb_joint": _sweep_orb_joint,
            "tod_interaction": _sweep_tod_interaction,
            "perm_joint": _sweep_perm_joint,
            "ema_perm_joint": _sweep_ema_perm_joint,
            "tick_perm_joint": _sweep_tick_perm_joint,
            "chop_joint": _sweep_chop_joint,
            "tick_ema": _sweep_tick_ema,
            "ema_regime": _sweep_ema_regime,
            "ema_atr": _sweep_ema_atr,
            "regime_atr": _sweep_regime_atr,
            "r2_atr": _sweep_r2_atr,
            "r2_tod": _sweep_r2_tod,
            "tick": _sweep_tick,
            "gate_matrix": _sweep_gate_matrix,
            "squeeze": _sweep_squeeze,
            "combo_fast": _sweep_combo_fast,
            "frontier": _sweep_frontier,
        }
        for axis_name in _COMBO_FULL_PLAN:
            fn = combo_full_registry.get(str(axis_name))
            if fn is not None:
                _run_axis(str(axis_name), fn)

        seed_path_raw = str(getattr(args, "seed_milestones", "") or "").strip()
        if seed_path_raw:
            seed_path = Path(seed_path_raw)
            if seed_path.exists():
                _run_axis("st37_refine", _sweep_st37_refine)
            else:
                print(f"SKIP st37_refine: --seed-milestones not found ({seed_path})", flush=True)

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

        def _seed_key(item: dict) -> str:
            st = item.get("strategy") or {}
            flt = item.get("filters")
            raw = {"strategy": st, "filters": flt}
            return json.dumps(raw, sort_keys=True, default=str)

        def _family_key(item: dict) -> tuple:
            st = item.get("strategy") or {}
            return (
                str(st.get("ema_preset") or ""),
                str(st.get("ema_entry_mode") or ""),
                str(st.get("regime_mode") or ""),
                str(st.get("regime_bar_size") or ""),
                str(st.get("spot_exit_mode") or ""),
            )

        # Prefer diversity: keep the best seed per "family", then take the top-K families.
        best_per_family: dict[tuple, dict] = {}
        for item in candidates:
            fam = _family_key(item)
            prev = best_per_family.get(fam)
            if prev is None or _score_row_pnl_dd(item["metrics"]) > _score_row_pnl_dd(prev["metrics"]):
                best_per_family[fam] = item

        family_winners = sorted(best_per_family.values(), key=lambda x: _score_row_pnl_dd(x["metrics"]), reverse=True)

        # Add a few "outlier" seeds (high pnl / high ROI / high win) even if they share families.
        by_pnl = sorted(candidates, key=lambda x: _score_row_pnl(x["metrics"]), reverse=True)
        by_roi = sorted(candidates, key=lambda x: float((x.get("metrics") or {}).get("roi") or 0.0), reverse=True)
        by_win = sorted(candidates, key=lambda x: float((x.get("metrics") or {}).get("win_rate") or 0.0), reverse=True)

        seed_top = max(1, int(args.seed_top or 0))
        seed_pool: list[dict] = []
        for src in (
            family_winners[:seed_top],
            by_pnl[: max(5, seed_top // 4)],
            by_roi[: max(5, seed_top // 4)],
            by_win[: max(5, seed_top // 4)],
        ):
            for item in src:
                seed_pool.append(item)

        seen_seed: set[str] = set()
        seeds: list[dict] = []
        for item in seed_pool:
            key = _seed_key(item)
            if key in seen_seed:
                continue
            seen_seed.add(key)
            seeds.append(item)
            if len(seeds) >= seed_top:
                break

        print("")
        print("=== champ_refine: seeded refinement (bounded) ===")
        print(f"- seeds_in_file={len(candidates)} families={len(best_per_family)} selected={len(seeds)} seed_top={seed_top}")
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

        # Shock pocket (includes the v25/v31 daily ATR% family + a few TR-ratio variants).
        shock_variants: list[tuple[dict[str, object], str]] = [
            ({"shock_gate_mode": "off"}, "shock=off"),
        ]
        if not is_slv:
            for on_atr, off_atr in ((12.5, 12.0), (13.0, 12.5), (13.5, 13.0), (14.0, 13.5), (14.5, 14.0)):
                for sl_mult in (0.75, 1.0):
                    shock_variants.append(
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
                shock_variants.append(
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
            shock_variants.append(
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
                )
            )
            shock_variants.append(
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
                )
            )
            for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
                shock_variants.append(
                    (
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "tr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": float(on_ratio),
                            "shock_off_ratio": float(off_ratio),
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                        },
                        f"shock=surf tr_ratio on={on_ratio:g} off={off_ratio:g} min_atr=7 sl_mult=0.75",
                    )
                )
            for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
                shock_variants.append(
                    (
                        {
                            "shock_gate_mode": "surf",
                            "shock_detector": "atr_ratio",
                            "shock_atr_fast_period": 7,
                            "shock_atr_slow_period": 50,
                            "shock_on_ratio": float(on_ratio),
                            "shock_off_ratio": float(off_ratio),
                            "shock_min_atr_pct": 7.0,
                            "shock_direction_source": "signal",
                            "shock_direction_lookback": 1,
                            "shock_stop_loss_pct_mult": 0.75,
                        },
                        f"shock=surf atr_ratio on={on_ratio:g} off={off_ratio:g} min_atr=7 sl_mult=0.75",
                    )
                )
            shock_variants.append(
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
        else:
            # SLV: daily ATR% and TR-ratio operate on a much smaller scale; explore a wider-but-still-bounded pocket.
            for on_atr, off_atr in ((3.0, 2.5), (3.5, 3.0), (4.0, 3.5), (4.5, 4.0), (5.0, 4.5), (6.0, 5.0)):
                for sl_mult in (0.75, 1.0):
                    shock_variants.append(
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
                shock_variants.append(
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
            # v4: shock/risk scaling pocket (not trade blocking). Keep cadence high, focus on decade DD%.
            # - Use `detect` mode so the shock engine runs (enables ATR%-based sizing scale),
            #   without turning into a hard gate like `block`.
            for det, fast_p, slow_p, on_r, off_r, min_atr, dir_src, dir_lb, target_atr, min_mult, down_mult in (
                ("tr_ratio", 3, 21, 1.25, 1.15, 0.8, "signal", 1, 3.0, 0.10, 0.60),
                ("tr_ratio", 3, 21, 1.30, 1.20, 1.0, "signal", 1, 3.5, 0.20, 0.70),
                ("tr_ratio", 5, 30, 1.25, 1.15, 1.0, "signal", 2, 3.5, 0.20, 0.60),
                ("tr_ratio", 7, 50, 1.35, 1.25, 1.2, "regime", 2, 4.0, 0.20, 0.60),
                ("tr_ratio", 7, 50, 1.45, 1.30, 1.5, "regime", 2, 4.0, 0.20, 0.50),
                ("atr_ratio", 3, 21, 1.25, 1.15, 0.8, "signal", 1, 3.0, 0.10, 0.60),
                ("atr_ratio", 7, 50, 1.35, 1.25, 1.2, "regime", 2, 4.0, 0.20, 0.60),
            ):
                shock_variants.append(
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
                shock_variants.append(
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
                shock_variants.append(
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

        # TR/gap overlays pocket (panic = defensive, pop = aggressive).
        def _risk_off_overrides() -> dict[str, object]:
            return {
                "risk_entry_cutoff_hour_et": None,
                "riskoff_tr5_med_pct": None,
                "riskpanic_tr5_med_pct": None,
                "riskpanic_neg_gap_ratio_min": None,
                "riskpop_tr5_med_pct": None,
                "riskpop_pos_gap_ratio_min": None,
            }

        if not is_slv:
            risk_variants: list[tuple[dict[str, object], str]] = [
                (_risk_off_overrides(), "risk=off"),
                (
                    {
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
                        **_risk_off_overrides(),
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
        else:
            # SLV v4+ focus: avoid risk overlays that cancel entries (keep cadence high).
            risk_variants = [
                (_risk_off_overrides(), "risk=off"),
            ]

        def _cfg_from_payload(strategy_payload, filters_payload) -> ConfigBundle | None:
            if not isinstance(strategy_payload, dict):
                return None
            try:
                filters_obj = _filters_from_payload(filters_payload)
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
            if not offline:
                raise SystemExit("champ_refine stage3a worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.champ_refine_stage3a))
            out_path_raw = str(args.champ_refine_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--champ-refine-out is required for champ_refine stage3a worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.champ_refine_worker,
                args.champ_refine_workers,
                label="champ_refine stage3a",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid champ_refine stage3a payload JSON: {payload_path}") from exc
            raw_base = payload.get("base_exit_variants") if isinstance(payload, dict) else None
            if not isinstance(raw_base, list):
                raise SystemExit(f"champ_refine stage3a payload missing 'base_exit_variants' list: {payload_path}")
            seed_tag = str(payload.get("seed_tag") or "seed") if isinstance(payload, dict) else "seed"

            base_exit_local: list[tuple[ConfigBundle, str]] = []
            for item in raw_base:
                if not isinstance(item, dict):
                    continue
                strat_payload = item.get("strategy") or {}
                filters_payload = item.get("filters")
                exit_note = str(item.get("exit_note") or "")
                cfg = _cfg_from_payload(strat_payload, filters_payload)
                if cfg is None:
                    continue
                base_exit_local.append((cfg, exit_note))

            stage3a_plan_all = _build_stage3a_plan(base_exit_local, seed_tag_local=seed_tag)
            stage3a_total = len(stage3a_plan_all)
            local_total = (stage3a_total // workers) + (1 if worker_id < (stage3a_total % workers) else 0)
            if len(stage3a_plan_all) != int(stage3a_total):
                raise SystemExit(
                    f"champ_refine stage3a worker internal error: combos={len(stage3a_plan_all)} expected={stage3a_total}"
                )

            shard_plan = (
                item for combo_idx, item in enumerate(stage3a_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"champ_refine stage3a worker {worker_id+1}/{workers} {seed_tag}",
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
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

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"champ_refine stage3a worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        if args.champ_refine_stage3b:
            if not offline:
                raise SystemExit("champ_refine stage3b worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.champ_refine_stage3b))
            out_path_raw = str(args.champ_refine_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--champ-refine-out is required for champ_refine stage3b worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.champ_refine_worker,
                args.champ_refine_workers,
                label="champ_refine stage3b",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid champ_refine stage3b payload JSON: {payload_path}") from exc
            raw_shortlist = payload.get("shortlist") if isinstance(payload, dict) else None
            if not isinstance(raw_shortlist, list):
                raise SystemExit(f"champ_refine stage3b payload missing 'shortlist' list: {payload_path}")
            seed_tag = str(payload.get("seed_tag") or "seed") if isinstance(payload, dict) else "seed"

            shortlist_local: list[tuple[ConfigBundle, str]] = []
            for item in raw_shortlist:
                if not isinstance(item, dict):
                    continue
                strat_payload = item.get("strategy") or {}
                filters_payload = item.get("filters")
                base_note = str(item.get("base_note") or "")
                cfg = _cfg_from_payload(strat_payload, filters_payload)
                if cfg is None:
                    continue
                shortlist_local.append((cfg, base_note))

            stage3b_plan_all = _build_stage3b_plan(shortlist_local)
            stage3b_total = len(stage3b_plan_all)
            local_total = (stage3b_total // workers) + (1 if worker_id < (stage3b_total % workers) else 0)
            if len(stage3b_plan_all) != int(stage3b_total):
                raise SystemExit(
                    f"champ_refine stage3b worker internal error: combos={len(stage3b_plan_all)} expected={stage3b_total}"
                )

            shard_plan = (
                item for combo_idx, item in enumerate(stage3b_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"champ_refine stage3b worker {worker_id+1}/{workers} {seed_tag}",
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
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

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"champ_refine stage3b worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
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
            seed_short = float(getattr(cfg_seed.strategy, "spot_short_risk_mult", 1.0) or 1.0)
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
                    elapsed = now - t0_all
                    rate = tested_total / elapsed if elapsed > 0 else 0.0
                    print(f"champ_refine progress tested={tested_total} ({rate:0.2f} cfg/s)", flush=True)
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
                                elapsed = now - t0_all
                                rate = tested_total / elapsed if elapsed > 0 else 0.0
                                print(f"champ_refine progress tested={tested_total} ({rate:0.2f} cfg/s)", flush=True)
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
                    # SLV v4+ focus: keep high cadence by locking stacking (max_open=5).
                    if is_slv:
                        base_variants.append(
                            replace(
                                r_cfg,
                                strategy=replace(
                                    r_cfg.strategy,
                                    spot_short_risk_mult=float(mult),
                                    max_open_trades=5,
                                ),
                            )
                        )
                    else:
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
            total_3a = 0
            for base_cfg, _exit_note in base_exit_variants:
                seed_mode = str(getattr(base_cfg.strategy, "ema_entry_mode", "cross") or "cross").strip().lower()
                if seed_mode not in ("cross", "trend"):
                    seed_mode = "cross"
                other_mode = "trend" if seed_mode == "cross" else "cross"
                total_3a += 2 * len(tod_variants) * len(perm_variants) * len(signed_slope_variants)
            print(f"  stage3a micro: total={total_3a}", flush=True)
            if jobs > 1:
                if not offline:
                    raise SystemExit("--jobs>1 for champ_refine requires --offline (avoid parallel IBKR sessions).")

                base_cli = _strip_flags(
                    list(sys.argv[1:]),
                    flags=("--write-milestones", "--merge-milestones"),
                    flags_with_values=(
                        "--axis",
                        "--jobs",
                        "--milestones-out",
                        "--champ-refine-stage3a",
                        "--champ-refine-stage3b",
                        "--champ-refine-worker",
                        "--champ-refine-workers",
                        "--champ-refine-out",
                        "--champ-refine-run-min-trades",
                    ),
                )

                jobs_eff = min(int(jobs), int(_default_jobs()), int(total_3a)) if total_3a > 0 else 1
                print(f"  stage3a parallel: workers={jobs_eff} total={total_3a}", flush=True)

                with tempfile.TemporaryDirectory(prefix="tradebot_champ_refine_3a_") as tmpdir:
                    tmp_root = Path(tmpdir)
                    payload_path = tmp_root / "stage3a_payload.json"
                    base_payload: list[dict] = []
                    for base_cfg, exit_note in base_exit_variants:
                        base_payload.append(
                            {
                                "strategy": _spot_strategy_payload(base_cfg, meta=meta),
                                "filters": _filters_payload(base_cfg.strategy.filters),
                                "exit_note": str(exit_note),
                            }
                        )
                    write_json(payload_path, {"seed_tag": seed_tag, "base_exit_variants": base_payload}, sort_keys=False)

                    payloads = _run_parallel_json_worker_plan(
                        jobs_eff=jobs_eff,
                        tmp_prefix="tradebot_champ_refine_3a_run_",
                        worker_tag="cr3a",
                        out_prefix="stage3a_out",
                        build_cmd=lambda worker_id, workers_n, out_path: [
                            sys.executable,
                            "-u",
                            "-m",
                            "tradebot.backtest",
                            "spot",
                            *base_cli,
                            "--axis",
                            "champ_refine",
                            "--jobs",
                            "1",
                            "--champ-refine-stage3a",
                            str(payload_path),
                            "--champ-refine-worker",
                            str(worker_id),
                            "--champ-refine-workers",
                            str(workers_n),
                            "--champ-refine-out",
                            str(out_path),
                            "--champ-refine-run-min-trades",
                            str(int(run_min_trades)),
                        ],
                        capture_error="Failed to capture champ_refine stage3a worker stdout.",
                        failure_label="champ_refine stage3a worker",
                        missing_label="champ_refine stage3a",
                        invalid_label="champ_refine stage3a",
                    )

                    tested_total_3a = 0
                    for worker_id in range(jobs_eff):
                        payload = payloads.get(int(worker_id))
                        if not isinstance(payload, dict):
                            continue
                        tested_total_3a += int(payload.get("tested") or 0)
                        for rec in payload.get("records") or []:
                            if not isinstance(rec, dict):
                                continue
                            cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                            if cfg is None:
                                continue
                            row = rec.get("row")
                            if not isinstance(row, dict):
                                continue
                            note = str(rec.get("note") or "").strip()
                            note = (
                                note
                                or f"{seed_tag} | short_mult={getattr(cfg.strategy,'spot_short_risk_mult', 1.0):g} | stage3a"
                            )
                            row = dict(row)
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)
                            stage3a.append((cfg, row, note))

                    tested_total += int(tested_total_3a)
            else:
                stage3a_plan = _build_stage3a_plan(base_exit_variants, seed_tag_local=seed_tag)
                tested_3a, kept_3a = _run_sweep(
                    plan=stage3a_plan,
                    bars=bars_sig,
                    total=total_3a,
                    progress_label="  stage3a",
                    report_every=report_every,
                    heartbeat_sec=heartbeat_sec,
                )
                tested_total += int(tested_3a)
                for cfg, row, note, _meta in kept_3a:
                    rows.append(row)
                    stage3a.append((cfg, row, note))

            if not stage3a:
                continue

            # Shortlist: keep diversity (pnl/dd + pnl + roi + win), deduped.
            by_dd = sorted(stage3a, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[:6]
            by_pnl = sorted(stage3a, key=lambda t: _score_row_pnl(t[1]), reverse=True)[:4]
            by_roi = sorted(stage3a, key=lambda t: float((t[1] or {}).get("roi") or 0.0), reverse=True)[:3]
            by_win = sorted(stage3a, key=lambda t: float((t[1] or {}).get("win_rate") or 0.0), reverse=True)[:3]
            shortlist: list[tuple[ConfigBundle, dict, str]] = []
            seen_cfg: set[str] = set()
            for cfg, row, note in by_dd + by_pnl + by_roi + by_win:
                key = _milestone_key(cfg)
                if key in seen_cfg:
                    continue
                seen_cfg.add(key)
                shortlist.append((cfg, row, note))
                if len(shortlist) >= 10:
                    break

            # Stage 3B: apply shock + TR-overlay pockets to the shortlist.
            total_3b = len(shortlist) * len(shock_variants) * len(risk_variants)
            print(f"  stage3b shock+risk: shortlist={len(shortlist)} total={total_3b}", flush=True)
            if jobs > 1:
                if not offline:
                    raise SystemExit("--jobs>1 for champ_refine requires --offline (avoid parallel IBKR sessions).")

                base_cli = _strip_flags(
                    list(sys.argv[1:]),
                    flags=("--write-milestones", "--merge-milestones"),
                    flags_with_values=(
                        "--axis",
                        "--jobs",
                        "--milestones-out",
                        "--champ-refine-stage3a",
                        "--champ-refine-stage3b",
                        "--champ-refine-worker",
                        "--champ-refine-workers",
                        "--champ-refine-out",
                        "--champ-refine-run-min-trades",
                    ),
                )

                jobs_eff = min(int(jobs), int(_default_jobs()), int(total_3b)) if total_3b > 0 else 1
                print(f"  stage3b parallel: workers={jobs_eff} total={total_3b}", flush=True)

                with tempfile.TemporaryDirectory(prefix="tradebot_champ_refine_3b_") as tmpdir:
                    tmp_root = Path(tmpdir)
                    payload_path = tmp_root / "stage3b_payload.json"
                    shortlist_payload: list[dict] = []
                    for base_cfg, _, base_note in shortlist:
                        shortlist_payload.append(
                            {
                                "strategy": _spot_strategy_payload(base_cfg, meta=meta),
                                "filters": _filters_payload(base_cfg.strategy.filters),
                                "base_note": str(base_note),
                            }
                        )
                    write_json(payload_path, {"seed_tag": seed_tag, "shortlist": shortlist_payload}, sort_keys=False)

                    payloads = _run_parallel_json_worker_plan(
                        jobs_eff=jobs_eff,
                        tmp_prefix="tradebot_champ_refine_3b_run_",
                        worker_tag="cr3b",
                        out_prefix="stage3b_out",
                        build_cmd=lambda worker_id, workers_n, out_path: [
                            sys.executable,
                            "-u",
                            "-m",
                            "tradebot.backtest",
                            "spot",
                            *base_cli,
                            "--axis",
                            "champ_refine",
                            "--jobs",
                            "1",
                            "--champ-refine-stage3b",
                            str(payload_path),
                            "--champ-refine-worker",
                            str(worker_id),
                            "--champ-refine-workers",
                            str(workers_n),
                            "--champ-refine-out",
                            str(out_path),
                            "--champ-refine-run-min-trades",
                            str(int(run_min_trades)),
                        ],
                        capture_error="Failed to capture champ_refine stage3b worker stdout.",
                        failure_label="champ_refine stage3b worker",
                        missing_label="champ_refine stage3b",
                        invalid_label="champ_refine stage3b",
                    )

                    tested_total_3b = 0
                    for worker_id in range(jobs_eff):
                        payload = payloads.get(int(worker_id))
                        if not isinstance(payload, dict):
                            continue
                        tested_total_3b += int(payload.get("tested") or 0)
                        for rec in payload.get("records") or []:
                            if not isinstance(rec, dict):
                                continue
                            cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                            if cfg is None:
                                continue
                            row = rec.get("row")
                            if not isinstance(row, dict):
                                continue
                            note = str(rec.get("note") or "").strip() or "stage3b"
                            row = dict(row)
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

                    tested_total += int(tested_total_3b)
            else:
                shortlist_pairs = [(cfg, note) for cfg, _row, note in shortlist]
                stage3b_plan = _build_stage3b_plan(shortlist_pairs)
                tested_3b, kept_3b = _run_sweep(
                    plan=stage3b_plan,
                    bars=bars_sig,
                    total=total_3b,
                    progress_label="  stage3b",
                    report_every=report_every,
                    heartbeat_sec=heartbeat_sec,
                )
                tested_total += int(tested_3b)
                for _cfg, row, _note, _meta in kept_3b:
                    rows.append(row)

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
        tested_total = 0
        t0 = pytime.perf_counter()
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

        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            for gate_mode in gate_modes:
                for det_over, det_note in detector_variants:
                    for override_dir in regime_override_dirs:
                        for short_factor in short_risk_factors:
                            for long_down in long_down_factors:
                                for target_atr, min_mult, scale_note in risk_scale_variants:
                                    tested_total += 1
                                    if tested_total % report_every == 0 or tested_total == total:
                                        elapsed = pytime.perf_counter() - t0
                                        rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                        remaining = total - tested_total
                                        eta_sec = (remaining / rate) if rate > 0 else 0.0
                                        pct = (tested_total / total * 100.0) if total > 0 else 0.0
                                        print(
                                            f"shock_alpha_refine {tested_total}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                            flush=True,
                                        )

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
                                    row = _run_cfg(cfg=cfg)
                                    if not row:
                                        continue
                                    note = (
                                        f"{seed_note} | gate={gate_mode} {det_note} | "
                                        f"override_dir={int(override_dir)} | "
                                        f"short_factor={short_factor:g} long_down={long_down:g} | {scale_note}"
                                    )
                                    row["note"] = note
                                    _record_milestone(cfg, row, note)
                                    rows.append(row)

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
        tested_total = 0
        t0 = pytime.perf_counter()
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

        total = len(seeds) * len(shock_variants) * len(risk_variants)
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
            tested = 0
            kept = 0
            combo_idx = 0
            report_every_local = 100
            t0_local = pytime.perf_counter()
            records: list[dict] = []

            def _progress() -> None:
                elapsed = pytime.perf_counter() - t0_local
                rate = (tested / elapsed) if elapsed > 0 else 0.0
                remaining = local_total - tested
                eta_sec = (remaining / rate) if rate > 0 else 0.0
                pct = (tested / local_total * 100.0) if local_total > 0 else 0.0
                print(
                    f"{axis_tag} worker {worker_id+1}/{workers} "
                    f"{tested}/{local_total} ({pct:0.1f}%) kept={kept} "
                    f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                    flush=True,
                )

            for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):

                for shock_over, shock_note in shock_variants:
                    for risk_over, risk_note in risk_variants:
                        assigned = (combo_idx % workers) == worker_id
                        combo_idx += 1
                        if not assigned:
                            continue
                        tested += 1
                        if tested % report_every_local == 0 or tested == local_total:
                            _progress()

                        over: dict[str, object] = {}
                        over.update(shock_over)
                        over.update(risk_over)
                        f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                        cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"{seed_note} | {shock_note} | {risk_note}"
                        records.append(
                            {
                                "strategy": _spot_strategy_payload(cfg, meta=meta),
                                "filters": _filters_payload(cfg.strategy.filters),
                                "note": note,
                                "row": row,
                            }
                        )
                        kept += 1

            if combo_idx != total:
                raise SystemExit(f"{axis_tag} worker internal error: combos={combo_idx} expected={total}")

            out_payload = {"tested": tested, "kept": kept, "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"{axis_tag} worker done tested={tested} kept={kept} out={out_path}", flush=True)
            return

        tested_total = 0
        if jobs > 1 and total > 0:
            if not offline:
                raise SystemExit(f"--jobs>1 for {axis_tag} requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--shock-velocity-worker",
                    "--shock-velocity-workers",
                    "--shock-velocity-out",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(total)) if total > 0 else 1
            print(f"{axis_tag} parallel: workers={jobs_eff} total={total}", flush=True)

            payloads = _run_parallel_json_worker_plan(
                jobs_eff=jobs_eff,
                tmp_prefix="tradebot_shock_velocity_",
                worker_tag="sv",
                out_prefix="shock_velocity_out",
                build_cmd=lambda worker_id, workers_n, out_path: [
                    sys.executable,
                    "-u",
                    "-m",
                    "tradebot.backtest",
                    "spot",
                    *base_cli,
                    "--axis",
                    axis_tag,
                    "--jobs",
                    "1",
                    "--shock-velocity-worker",
                    str(worker_id),
                    "--shock-velocity-workers",
                    str(workers_n),
                    "--shock-velocity-out",
                    str(out_path),
                ],
                capture_error=f"Failed to capture {axis_tag} worker stdout.",
                failure_label=f"{axis_tag} worker",
                missing_label=str(axis_tag),
                invalid_label=str(axis_tag),
            )

            for worker_id in range(jobs_eff):
                payload = payloads.get(int(worker_id))
                if not isinstance(payload, dict):
                    continue
                tested_total += int(payload.get("tested") or 0)
                for rec in payload.get("records") or []:
                    if not isinstance(rec, dict):
                        continue
                    strat_payload = rec.get("strategy")
                    filters_payload = rec.get("filters")
                    note = rec.get("note")
                    row = rec.get("row")
                    if not isinstance(strat_payload, dict) or not isinstance(note, str) or not isinstance(row, dict):
                        continue
                    row = dict(row)
                    filters_obj = _filters_from_payload(filters_payload) if isinstance(filters_payload, dict) else None
                    cfg = _mk_bundle(
                        strategy=_strategy_from_payload(strat_payload, filters=filters_obj),
                        start=start,
                        end=end,
                        bar_size=signal_bar_size,
                        use_rth=use_rth,
                        cache_dir=cache_dir,
                        offline=offline,
                    )
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

            run_calls_total += int(tested_total)
        else:
            for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):

                for shock_over, shock_note in shock_variants:
                    for risk_over, risk_note in risk_variants:
                        tested_total += 1
                        if tested_total % report_every == 0 or tested_total == total:
                            elapsed = pytime.perf_counter() - t0
                            rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                            remaining = total - tested_total
                            eta_sec = (remaining / rate) if rate > 0 else 0.0
                            pct = (tested_total / total * 100.0) if total > 0 else 0.0
                            print(
                                f"{axis_tag} {tested_total}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                flush=True,
                            )

                        over: dict[str, object] = {}
                        over.update(shock_over)
                        over.update(risk_over)
                        f_obj = _merge_filters(cfg_seed.strategy.filters, over)
                        cfg = replace(cfg_seed, strategy=replace(cfg_seed.strategy, filters=f_obj))
                        row = _run_cfg(cfg=cfg)
                        if not row:
                            continue
                        note = f"{seed_note} | {shock_note} | {risk_note}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

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
        t0 = pytime.perf_counter()
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

        work: list[dict] = []
        total = 0
        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):

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
            if variants <= 0:
                continue
            total += int(variants)
            work.append(
                {
                    "cfg_seed": cfg_seed,
                    "seed_note": seed_note,
                    "detect_over": detect_over,
                    "mode": mode,
                    "detector": detector_eff,
                    "targets": targets,
                    "stops": stops,
                    "daily_pairs": daily_pairs,
                }
            )

        if not work:
            print("No usable seeds after parsing/filtering.", flush=True)
            return

        print(
            f"- variants: seed-aware pockets (min_mult={min_mults}); total={total} "
            f"(note: auto-enables shock_gate_mode=detect when missing)",
            flush=True,
        )

        tested_total = 0
        for item in work:
            cfg_seed = item["cfg_seed"]
            seed_note = item["seed_note"]
            detect_over = item["detect_over"]
            targets = item["targets"]
            stops = item["stops"]
            daily_pairs = item["daily_pairs"]

            for off_on in daily_pairs:
                off_v, on_v = off_on
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
                            tested_total += 1
                            if tested_total % report_every == 0 or tested_total == total:
                                elapsed = pytime.perf_counter() - t0
                                rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                remaining = total - tested_total
                                eta_sec = (remaining / rate) if rate > 0 else 0.0
                                pct = (tested_total / total * 100.0) if total > 0 else 0.0
                                print(
                                    f"{axis_tag} {tested_total}/{total} ({pct:0.1f}%) kept={len(rows)} "
                                    f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                    flush=True,
                                )

                            cfg = replace(
                                cfg_seed,
                                strategy=replace(
                                    cfg_seed.strategy,
                                    filters=f_obj,
                                    spot_stop_loss_pct=float(stop_pct),
                                ),
                            )
                            row = _run_cfg(cfg=cfg)
                            if not row:
                                continue

                            surf_note = ""
                            if off_v is not None and on_v is not None:
                                surf_note = f" surf(off={off_v:g},on={on_v:g})"
                            if target_atr is None:
                                scale_note = " shock_scale=off"
                            else:
                                scale_note = f" shock_scale target_atr%={target_atr:g} min_mult={min_mult:g} apply_to=cap"
                            note = f"{seed_note} |{scale_note}{surf_note} | stop%={stop_pct:g}"
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

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

        scale_periods: tuple[tuple[int, int], ...] = ((2, 50), (3, 50), (5, 50), (3, 21))
        # Telemetry (SLV 1h FULL24 tr_fast_pct): p75≈0.71%, p85≈0.90%, p92≈1.14% (1y window).
        # Include a few aggressive "surprise" targets to actually touch cap-bound trades.
        targets: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.7, 0.9, 1.1)
        min_mults: tuple[float, ...] = (0.05, 0.1, 0.2)
        apply_tos: tuple[str, ...] = ("cap", "both")

        t0 = pytime.perf_counter()
        tested_total = 0
        report_every = 50

        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            _emit_seed_base_row(cfg_seed=cfg_seed, seed_note=seed_note, rows=rows, base_note="shock_scale=off (base)")

            for fast_p, slow_p in scale_periods:
                for target_atr in targets:
                    for min_mult in min_mults:
                        for apply_to in apply_tos:
                            tested_total += 1
                            if tested_total % report_every == 0:
                                elapsed = pytime.perf_counter() - t0
                                rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                print(f"{axis_tag} tested={tested_total} kept={len(rows)} rate={rate:0.1f}/s", flush=True)

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
                            row = _run_cfg(cfg=cfg)
                            if not row:
                                continue
                            note = (
                                f"{seed_note} | shock_scale=tr_ratio {fast_p}/{slow_p} "
                                f"target_atr%={target_atr:g} min_mult={min_mult:g} apply_to={apply_to}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

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

        lookbacks: tuple[int, ...] = (10, 20, 40)
        # SLV 1y FULL24 stop-entries drawdown magnitude (20d peak):
        # p50≈1.17%, p75≈3.12%, p90≈9.70%, p95≈10.59%, max≈12.30%.
        targets: tuple[float, ...] = (3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
        min_mults: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
        apply_tos: tuple[str, ...] = ("cap", "both")

        t0 = pytime.perf_counter()
        tested_total = 0
        report_every = 50

        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            _emit_seed_base_row(cfg_seed=cfg_seed, seed_note=seed_note, rows=rows, base_note="shock_scale=off (base)")

            for lb in lookbacks:
                for target_dd in targets:
                    for min_mult in min_mults:
                        for apply_to in apply_tos:
                            tested_total += 1
                            if tested_total % report_every == 0:
                                elapsed = pytime.perf_counter() - t0
                                rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                print(f"{axis_tag} tested={tested_total} kept={len(rows)} rate={rate:0.1f}/s", flush=True)

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
                            row = _run_cfg(cfg=cfg)
                            if not row:
                                continue
                            note = (
                                f"{seed_note} | shock_scale=daily_drawdown lb={lb} "
                                f"target_dd%={target_dd:g} min_mult={min_mult:g} apply_to={apply_to}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

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

        cutoffs_et: tuple[int | None, ...] = (None, 15)
        panic_tr_meds: tuple[float, ...] = (2.75, 3.0, 3.25)
        neg_gap_ratios: tuple[float, ...] = (0.5, 0.6)
        neg_gap_abs_pcts: tuple[float | None, ...] = (None, 0.005)
        tr_delta_mins: tuple[float | None, ...] = (None, 0.25, 0.5, 0.75)
        long_factors: tuple[float, ...] = (1.0, 0.4, 0.0)
        scale_modes: tuple[str | None, ...] = (None, "linear")

        t0 = pytime.perf_counter()
        tested_total = 0
        report_every = 50

        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            _emit_seed_base_row(cfg_seed=cfg_seed, seed_note=seed_note, rows=rows, base_note="base")

            for cutoff in cutoffs_et:
                for tr_med in panic_tr_meds:
                    for neg_ratio in neg_gap_ratios:
                        for abs_gap in neg_gap_abs_pcts:
                            for tr_delta_min in tr_delta_mins:
                                for long_factor in long_factors:
                                    for scale_mode in scale_modes:
                                        tested_total += 1
                                        if tested_total % report_every == 0:
                                            elapsed = pytime.perf_counter() - t0
                                            rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                            print(f"{axis_tag} tested={tested_total} kept={len(rows)} rate={rate:0.1f}/s", flush=True)

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
                                        row = _run_cfg(cfg=cfg)
                                        if not row:
                                            continue

                                        cut_note = "-" if cutoff is None else str(cutoff)
                                        abs_note = "None" if abs_gap is None else f"{abs_gap:g}"
                                        delta_note = "=off" if tr_delta_min is None else f">={tr_delta_min:g}"
                                        mode_note = "off" if not scale_mode else str(scale_mode)
                                        note = (
                                            f"{seed_note} | cutoff<{cut_note} | panic med5>={tr_med:g} gap>={neg_ratio:g} abs>={abs_note} "
                                            f"trΔ{delta_note} | long={long_factor:g} scale={mode_note}"
                                        )
                                        row["note"] = note
                                        _record_milestone(cfg, row, note)
                                        rows.append(row)

        run_calls_total += int(tested_total)
        _print_leaderboards(rows, title=f"{axis_tag} (seeded riskpanic micro)", top_n=int(args.top))

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

        def _sort_key_pnl(item: dict) -> tuple:
            m = item.get("metrics") or {}
            return (
                float(m.get("pnl") or float("-inf")),
                int(m.get("trades") or 0),
                float(m.get("pnl_over_dd") or float("-inf")),
                float(m.get("win_rate") or 0.0),
            )

        def _sort_key_trades(item: dict) -> tuple:
            m = item.get("metrics") or {}
            return (
                int(m.get("trades") or 0),
                float(m.get("pnl") or float("-inf")),
                float(m.get("pnl_over_dd") or float("-inf")),
            )

        seed_top = max(1, int(args.seed_top or 0))
        by_pnl = sorted(candidates, key=_sort_key_pnl, reverse=True)[:seed_top]
        by_trades = sorted(candidates, key=_sort_key_trades, reverse=True)[: max(1, min(seed_top, 5))]

        seen: set[str] = set()
        seeds: list[dict] = []
        for item in by_pnl + by_trades:
            key = json.dumps(item.get("strategy") or {}, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(item)

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
        max_open_vals: tuple[int, ...] = (1, 2)

        t0 = pytime.perf_counter()
        tested_total = 0
        report_every = 100

        for _seed_i, _item, cfg_seed, seed_note in _iter_seed_bundles(seeds):
            _emit_seed_base_row(cfg_seed=cfg_seed, seed_note=seed_note, rows=rows, base_note="base")

            for max_open in max_open_vals:
                for close_eod in close_eod_vals:
                    for only_profit in only_profit_vals:
                        for pt in pt_vals:
                            for sl in sl_vals:
                                tested_total += 1
                                if tested_total % report_every == 0:
                                    elapsed = pytime.perf_counter() - t0
                                    rate = (tested_total / elapsed) if elapsed > 0 else 0.0
                                    print(
                                        f"{axis_tag} tested={tested_total} kept={len(rows)} rate={rate:0.1f}/s",
                                        flush=True,
                                    )

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
                                        max_open_trades=int(max_open),
                                    ),
                                )
                                row = _run_cfg(cfg=cfg)
                                if not row:
                                    continue

                                pt_note = "None" if pt is None else f"{pt:g}"
                                note = (
                                    f"{seed_note} | max_open={max_open} close_eod={int(close_eod)} "
                                    f"only_profit={int(only_profit)} | PT={pt_note} SL={sl:g}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                rows.append(row)

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

        def _seed_key(item: dict) -> str:
            st = item.get("strategy") or {}
            flt = item.get("filters")
            raw = {"strategy": st, "filters": flt}
            return json.dumps(raw, sort_keys=True, default=str)

        def _seed_score(item: dict) -> tuple:
            # Prefer stability-ranked inputs (kingmaker output), but fall back to pnl/dd.
            ev = item.get("eval") or {}
            try:
                stab = float(ev.get("stability_min_roi_dd") or 0.0)
            except (TypeError, ValueError):
                stab = 0.0
            m = item.get("metrics") or {}
            try:
                roi_dd = float(m.get("roi_over_dd_pct") or 0.0)
            except (TypeError, ValueError):
                roi_dd = 0.0
            return (
                stab,
                roi_dd,
                float(m.get("pnl_over_dd") or 0.0),
                float(m.get("roi") or 0.0),
                float(m.get("pnl") or 0.0),
            )

        seed_top = max(1, int(args.seed_top or 0))
        cand_sorted = sorted(candidates, key=_seed_score, reverse=True)
        seen: set[str] = set()
        seeds: list[dict] = []
        for item in cand_sorted:
            key = _seed_key(item)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(item)
            if len(seeds) >= seed_top:
                break

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

        # Stage2 worker mode must early-exit before stage1 runs; this is invoked by the stage2
        # sharded runner when --jobs>1.
        if args.st37_refine_stage2:
            if not offline:
                raise SystemExit("st37_refine stage2 worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.st37_refine_stage2))
            out_path_raw = str(args.st37_refine_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--st37-refine-out is required for st37_refine stage2 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.st37_refine_worker,
                args.st37_refine_workers,
                label="st37_refine stage2",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid st37_refine stage2 payload JSON: {payload_path}") from exc

            raw_shortlist = payload.get("shortlist") if isinstance(payload, dict) else None
            if not isinstance(raw_shortlist, list):
                raise SystemExit(f"st37_refine stage2 payload missing 'shortlist' list: {payload_path}")

            def _parse_variants(raw: object, *, name: str) -> list[tuple[dict[str, object], str]]:
                if not isinstance(raw, list):
                    raise SystemExit(f"st37_refine stage2 payload missing '{name}' list: {payload_path}")
                out: list[tuple[dict[str, object], str]] = []
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    over = item.get("overrides")
                    note = item.get("note")
                    if not isinstance(over, dict):
                        continue
                    out.append((over, str(note or "")))
                if not out:
                    raise SystemExit(f"st37_refine stage2 payload '{name}' empty/invalid: {payload_path}")
                return out

            risk_variants_local = _parse_variants(payload.get("risk_variants") if isinstance(payload, dict) else None, name="risk_variants")
            shock_variants_local = _parse_variants(payload.get("shock_variants") if isinstance(payload, dict) else None, name="shock_variants")

            shortlist_local: list[tuple[ConfigBundle, str]] = []
            for item in raw_shortlist:
                if not isinstance(item, dict):
                    continue
                if not isinstance(item.get("strategy"), dict):
                    continue
                flt = item.get("filters")
                if flt is not None and not isinstance(flt, dict):
                    flt = None
                seed_tag = str(item.get("seed_tag") or "seed")
                base_note = str(item.get("base_note") or "")
                base = _base_bundle(bar_size=signal_bar_size, filters=None)
                cfg_seed = _apply_milestone_base(base, strategy=item["strategy"], filters=flt)
                shortlist_local.append((cfg_seed, f"{seed_tag} | {base_note}"))

            stage2_plan_all = _build_stage2_plan(
                shortlist_local,
                risk_variants_local=risk_variants_local,
                shock_variants_local=shock_variants_local,
            )
            stage2_total = len(stage2_plan_all)
            local_total = (stage2_total // workers) + (1 if worker_id < (stage2_total % workers) else 0)
            shard_plan = (
                item for combo_idx, item in enumerate(stage2_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"st37_refine stage2 worker {worker_id+1}/{workers}",
                report_every=200,
                heartbeat_sec=heartbeat_sec,
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
            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"st37_refine stage2 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
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

        def _cfg_from_payload(strategy_payload, filters_payload) -> ConfigBundle | None:
            if not isinstance(strategy_payload, dict):
                return None
            filters_obj = filters_payload if isinstance(filters_payload, dict) else None
            try:
                base = _base_bundle(bar_size=signal_bar_size, filters=None)
                return _apply_milestone_base(base, strategy=strategy_payload, filters=filters_obj)
            except Exception:
                return None

        def _mk_stage2_cfg(base_cfg: ConfigBundle, *, risk_over: dict[str, object], shock_over: dict[str, object]) -> ConfigBundle:
            over: dict[str, object] = {}
            over.update(risk_over)
            over.update(shock_over)
            f = _merge_filters(base_cfg.strategy.filters, overrides=over)
            return replace(base_cfg, strategy=replace(base_cfg.strategy, filters=f))

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

        if args.st37_refine_stage1:
            if not offline:
                raise SystemExit("st37_refine stage1 worker mode requires --offline (avoid parallel IBKR sessions).")
            payload_path = Path(str(args.st37_refine_stage1))
            out_path_raw = str(args.st37_refine_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--st37-refine-out is required for st37_refine stage1 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.st37_refine_worker,
                args.st37_refine_workers,
                label="st37_refine stage1",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid st37_refine stage1 payload JSON: {payload_path}") from exc
            raw_seeds = payload.get("seeds") if isinstance(payload, dict) else None
            if not isinstance(raw_seeds, list):
                raise SystemExit(f"st37_refine stage1 payload missing 'seeds' list: {payload_path}")

            seeds_local: list[dict] = []
            for item in raw_seeds:
                if not isinstance(item, dict):
                    continue
                if not isinstance(item.get("strategy"), dict):
                    continue
                flt = item.get("filters")
                if flt is not None and not isinstance(flt, dict):
                    flt = None
                seeds_local.append({"strategy": item["strategy"], "filters": flt, "group_name": str(item.get("group_name") or "")})

            stage1_plan_all = _build_stage1_plan(seeds_local)
            stage1_total = len(stage1_plan_all)
            local_total = (stage1_total // workers) + (1 if worker_id < (stage1_total % workers) else 0)
            shard_plan = (
                item for combo_idx, item in enumerate(stage1_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"st37_refine stage1 worker {worker_id+1}/{workers}",
                report_every=200,
                heartbeat_sec=heartbeat_sec,
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
            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"st37_refine stage1 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
            return

        stage1_rows: list[tuple[ConfigBundle, dict, str]] = []
        stage1_total = len(seeds) * len(perm_variants) * len(tod_variants) * len(sl_vals) * len(short_mult_vals)
        print(f"st37_refine: stage1 total={stage1_total} (perm×tod×sl×short)", flush=True)
        tested_1 = 0
        report_every = 200
        if jobs > 1:
            if not offline:
                raise SystemExit("--jobs>1 for st37_refine requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--st37-refine-stage1",
                    "--st37-refine-stage2",
                    "--st37-refine-worker",
                    "--st37-refine-workers",
                    "--st37-refine-out",
                    "--st37-refine-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(stage1_total)) if stage1_total > 0 else 1
            print(f"st37_refine stage1 parallel: workers={jobs_eff} total={stage1_total}", flush=True)

            with tempfile.TemporaryDirectory(prefix="tradebot_st37_refine_1_") as tmpdir:
                tmp_root = Path(tmpdir)
                payload_path = tmp_root / "stage1_payload.json"
                seeds_payload: list[dict] = []
                for s in seeds:
                    seeds_payload.append(
                        {"group_name": str(s.get("group_name") or ""), "strategy": s["strategy"], "filters": s.get("filters")}
                    )
                write_json(payload_path, {"seeds": seeds_payload}, sort_keys=False)

                payloads = _run_parallel_json_worker_plan(
                    jobs_eff=jobs_eff,
                    tmp_prefix="tradebot_st37_refine_stage1_",
                    worker_tag="st37:1",
                    out_prefix="stage1_out",
                    build_cmd=lambda worker_id, workers_n, out_path: [
                        sys.executable,
                        "-u",
                        "-m",
                        "tradebot.backtest",
                        "spot",
                        *base_cli,
                        "--axis",
                        "st37_refine",
                        "--jobs",
                        "1",
                        "--st37-refine-stage1",
                        str(payload_path),
                        "--st37-refine-worker",
                        str(worker_id),
                        "--st37-refine-workers",
                        str(workers_n),
                        "--st37-refine-out",
                        str(out_path),
                        "--st37-refine-run-min-trades",
                        str(int(run_min_trades)),
                    ],
                    capture_error="Failed to capture st37_refine stage1 worker stdout.",
                    failure_label="st37_refine stage1 worker",
                    missing_label="st37_refine stage1",
                    invalid_label="st37_refine stage1",
                )

                tested_total = 0
                for worker_id in range(jobs_eff):
                    payload = payloads.get(int(worker_id))
                    if not isinstance(payload, dict):
                        continue
                    tested_total += int(payload.get("tested") or 0)
                    for rec in payload.get("records") or []:
                        if not isinstance(rec, dict):
                            continue
                        cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                        if cfg is None:
                            continue
                        row = rec.get("row")
                        if not isinstance(row, dict):
                            continue
                        note = str(rec.get("note") or "").strip() or "st37 stage1"
                        row = dict(row)
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        stage1_rows.append((cfg, row, note))

                tested_1 = int(tested_total)
        else:
            stage1_plan = _build_stage1_plan(seeds)
            tested_1, kept_1 = _run_sweep(
                plan=stage1_plan,
                bars=bars_sig,
                total=stage1_total,
                progress_label="st37_refine stage1",
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
            )
            for cfg, row, note, _meta in kept_1:
                stage1_rows.append((cfg, row, note))

        print(f"st37_refine: stage1 kept={len(stage1_rows)} tested={tested_1}", flush=True)
        if not stage1_rows:
            return

        def _roi_dd(row: dict) -> float:
            try:
                roi = float(row.get("roi") or 0.0)
            except (TypeError, ValueError):
                roi = 0.0
            try:
                dd_pct = float(row.get("dd_pct") or 0.0)
            except (TypeError, ValueError):
                dd_pct = 0.0
            if dd_pct <= 0:
                return float("-inf") if roi <= 0 else float("inf")
            return roi / dd_pct

        stage1_sorted = sorted(stage1_rows, key=lambda t: (_roi_dd(t[1]), float(t[1].get("pnl_over_dd") or 0.0)), reverse=True)
        seen_cfg: set[str] = set()
        stage1_shortlist: list[tuple[ConfigBundle, dict, str]] = []
        for cfg, row, note in stage1_sorted:
            key = _milestone_key(cfg)
            if key in seen_cfg:
                continue
            seen_cfg.add(key)
            stage1_shortlist.append((cfg, row, note))
            if len(stage1_shortlist) >= 30:
                break

        print(f"st37_refine: stage1 shortlist={len(stage1_shortlist)}", flush=True)

        # Stage 2: risk overlays + shock pocket sweeps around the shortlisted configs.
        #
        # Goal: aggressively explore the TR-median + gap-ratio overlays (riskpanic + riskpop),
        # plus plain TR-median hygiene overlays (riskoff), with the full aggressive/defensive
        # sizing ends included (notably riskpop_short_factor=0.0 to hard-block shorts).
        risk_off = {
            "risk_entry_cutoff_hour_et": None,
            "riskoff_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": None,
            "riskpanic_neg_gap_ratio_min": None,
            "riskpop_tr5_med_pct": None,
            "riskpop_pos_gap_ratio_min": None,
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
        risk_variants: list[tuple[dict[str, object], str]] = [(risk_off, "risk=off")]

        panic_base = {
            **risk_off,
            "risk_entry_cutoff_hour_et": 15,
            "riskpanic_tr5_med_pct": 9.0,
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_lookback_days": 5,
            "riskpanic_short_risk_mult_factor": 0.5,
            "riskoff_mode": "hygiene",
        }
        risk_variants.append((panic_base, "riskpanic base (TRmed>=9 gap>=0.6 short=0.5 cutoff<15)"))
        for v in (1.0, 0.5, 0.2, 0.0):
            risk_variants.append(({**panic_base, "riskpanic_short_risk_mult_factor": float(v)}, f"riskpanic short={v:g}"))
        for tr in (8.0, 9.0, 10.0, 11.0):
            risk_variants.append(({**panic_base, "riskpanic_tr5_med_pct": float(tr)}, f"riskpanic TRmed>={tr:g}"))
        for ratio in (0.5, 0.6, 0.7):
            risk_variants.append(({**panic_base, "riskpanic_neg_gap_ratio_min": float(ratio)}, f"riskpanic gap>={ratio:g}"))
        for lb in (3, 5, 7):
            risk_variants.append(({**panic_base, "riskpanic_lookback_days": int(lb)}, f"riskpanic lookback={lb}d"))
        for cutoff in (None, 15, 16):
            risk_variants.append(
                ({**panic_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskpanic cutoff<{cutoff or '-'}")
            )
        for mode in ("hygiene", "directional"):
            risk_variants.append(({**panic_base, "riskoff_mode": str(mode)}, f"riskpanic mode={mode}"))

        riskoff_base = {
            **risk_off,
            "risk_entry_cutoff_hour_et": 15,
            "riskoff_tr5_med_pct": 9.0,
            "riskoff_tr5_lookback_days": 5,
            "riskoff_mode": "directional",
            "riskoff_long_risk_mult_factor": 0.8,
            "riskoff_short_risk_mult_factor": 0.5,
        }
        risk_variants.append((riskoff_base, "riskoff base (TRmed>=9 long=0.8 short=0.5 cutoff<15)"))
        for tr in (8.0, 9.0, 10.0, 11.0):
            risk_variants.append(({**riskoff_base, "riskoff_tr5_med_pct": float(tr)}, f"riskoff TRmed>={tr:g}"))
        for lb in (3, 5, 7):
            risk_variants.append(({**riskoff_base, "riskoff_tr5_lookback_days": int(lb)}, f"riskoff lookback={lb}d"))
        for long_f in (0.6, 0.8, 1.0):
            risk_variants.append(({**riskoff_base, "riskoff_long_risk_mult_factor": float(long_f)}, f"riskoff long={long_f:g}"))
        for short_f in (1.0, 0.5, 0.2, 0.0):
            risk_variants.append(({**riskoff_base, "riskoff_short_risk_mult_factor": float(short_f)}, f"riskoff short={short_f:g}"))
        for cutoff in (None, 15, 16):
            risk_variants.append(
                ({**riskoff_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskoff cutoff<{cutoff or '-'}")
            )
        for mode in ("hygiene", "directional"):
            risk_variants.append(({**riskoff_base, "riskoff_mode": str(mode)}, f"riskoff mode={mode}"))

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
        risk_variants.append((pop_base, "riskpop base (TRmed>=9 gap>=0.6 long=1.2 short=0.5 cutoff<15)"))
        for tr in (8.0, 9.0, 10.0, 11.0):
            risk_variants.append(({**pop_base, "riskpop_tr5_med_pct": float(tr)}, f"riskpop TRmed>={tr:g}"))
        for ratio in (0.5, 0.6, 0.7):
            risk_variants.append(({**pop_base, "riskpop_pos_gap_ratio_min": float(ratio)}, f"riskpop gap>={ratio:g}"))
        for lb in (3, 5, 7):
            risk_variants.append(({**pop_base, "riskpop_lookback_days": int(lb)}, f"riskpop lookback={lb}d"))
        # Defensive (2A): reduce long risk when pop is on.
        for long_f in (0.6, 0.8, 1.0, 1.2, 1.5):
            risk_variants.append(
                ({**pop_base, "riskpop_long_risk_mult_factor": float(long_f), "riskpop_short_risk_mult_factor": 1.0}, f"riskpop long={long_f:g} short=1.0")
            )
        # Aggressive (2B): block shorts in pop regimes.
        for short_f in (1.0, 0.5, 0.2, 0.0):
            risk_variants.append(
                ({**pop_base, "riskpop_long_risk_mult_factor": 1.2, "riskpop_short_risk_mult_factor": float(short_f)}, f"riskpop long=1.2 short={short_f:g}")
            )
        risk_variants.append(
            ({**pop_base, "riskpop_long_risk_mult_factor": 1.5, "riskpop_short_risk_mult_factor": 0.0}, "riskpop long=1.5 short=0.0")
        )
        for cutoff in (None, 15, 16):
            risk_variants.append(
                ({**pop_base, "risk_entry_cutoff_hour_et": int(cutoff) if cutoff is not None else None}, f"riskpop cutoff<{cutoff or '-'}")
            )
        for mode in ("hygiene", "directional"):
            risk_variants.append(({**pop_base, "riskoff_mode": str(mode)}, f"riskpop mode={mode}"))

        # Mixed overlays (small set; keep bounded).
        risk_variants.append(
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
        risk_variants.append(
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

        shock_variants: list[tuple[dict[str, object], str]] = [
            ({}, "shock=off"),
        ]
        # v31-style daily ATR% surf pocket (+ optional TR-trigger).
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
            shock_variants.append((base, f"shock=surf daily_atr on={on_atr:g} off={off_atr:g}"))
            for tr_on in (9.0, 10.0, 11.0):
                shock_variants.append(
                    (
                        {**base, "shock_daily_on_tr_pct": float(tr_on)},
                        f"shock=surf daily_atr on={on_atr:g} off={off_atr:g} tr_on={tr_on:g}",
                    )
                )

        # ATR-ratio pocket.
        for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
            shock_variants.append(
                (
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "atr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": float(on_ratio),
                        "shock_off_ratio": float(off_ratio),
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                    },
                    f"shock=surf atr_ratio on={on_ratio:g} off={off_ratio:g}",
                )
            )

        # TR-ratio pocket (often substitutes for daily ATR gating).
        for on_ratio, off_ratio in ((1.35, 1.25), (1.45, 1.30), (1.55, 1.30)):
            shock_variants.append(
                (
                    {
                        "shock_gate_mode": "surf",
                        "shock_detector": "tr_ratio",
                        "shock_atr_fast_period": 7,
                        "shock_atr_slow_period": 50,
                        "shock_on_ratio": float(on_ratio),
                        "shock_off_ratio": float(off_ratio),
                        "shock_min_atr_pct": 7.0,
                        "shock_direction_source": "signal",
                        "shock_direction_lookback": 1,
                        "shock_stop_loss_pct_mult": 0.75,
                    },
                    f"shock=surf tr_ratio on={on_ratio:g} off={off_ratio:g}",
                )
            )

        # Daily drawdown pocket (more "crash aware" than ATR%).
        for lb, on_dd, off_dd in (
            (20, -15.0, -10.0),
            (20, -20.0, -10.0),
            (30, -20.0, -15.0),
        ):
            shock_variants.append(
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

        # st37_refine stage2 worker mode is handled at the top of this sweep (before stage1).

        stage2_rows: list[tuple[ConfigBundle, dict, str]] = []
        stage2_plan = _build_stage2_plan(
            [(cfg, note) for cfg, _row, note in stage1_shortlist],
            risk_variants_local=risk_variants,
            shock_variants_local=shock_variants,
        )
        stage2_total = len(stage2_plan)
        print(f"st37_refine: stage2 total={stage2_total} (risk×shock)", flush=True)
        tested_2 = 0
        if jobs > 1:
            if not offline:
                raise SystemExit("--jobs>1 for st37_refine requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--st37-refine-stage1",
                    "--st37-refine-stage2",
                    "--st37-refine-worker",
                    "--st37-refine-workers",
                    "--st37-refine-out",
                    "--st37-refine-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(stage2_total)) if stage2_total > 0 else 1
            print(f"st37_refine stage2 parallel: workers={jobs_eff} total={stage2_total}", flush=True)

            with tempfile.TemporaryDirectory(prefix="tradebot_st37_refine_2_") as tmpdir:
                tmp_root = Path(tmpdir)
                payload_path = tmp_root / "stage2_payload.json"
                shortlist_payload: list[dict] = []
                for cfg, _row, note in stage1_shortlist:
                    shortlist_payload.append(
                        {
                            "seed_tag": str(note.split("|", 1)[0]).strip(),
                            "base_note": str(note),
                            "strategy": _spot_strategy_payload(cfg, meta=meta),
                            "filters": _filters_payload(cfg.strategy.filters),
                        }
                    )
                risk_payload: list[dict] = []
                for risk_over, risk_note in risk_variants:
                    risk_payload.append({"overrides": risk_over, "note": risk_note})
                shock_payload: list[dict] = []
                for shock_over, shock_note in shock_variants:
                    shock_payload.append({"overrides": shock_over, "note": shock_note})
                write_json(
                    payload_path,
                    {"shortlist": shortlist_payload, "risk_variants": risk_payload, "shock_variants": shock_payload},
                    sort_keys=False,
                )

                payloads = _run_parallel_json_worker_plan(
                    jobs_eff=jobs_eff,
                    tmp_prefix="tradebot_st37_refine_stage2_",
                    worker_tag="st37:2",
                    out_prefix="stage2_out",
                    build_cmd=lambda worker_id, workers_n, out_path: [
                        sys.executable,
                        "-u",
                        "-m",
                        "tradebot.backtest",
                        "spot",
                        *base_cli,
                        "--axis",
                        "st37_refine",
                        "--jobs",
                        "1",
                        "--st37-refine-stage2",
                        str(payload_path),
                        "--st37-refine-worker",
                        str(worker_id),
                        "--st37-refine-workers",
                        str(workers_n),
                        "--st37-refine-out",
                        str(out_path),
                        "--st37-refine-run-min-trades",
                        str(int(run_min_trades)),
                    ],
                    capture_error="Failed to capture st37_refine stage2 worker stdout.",
                    failure_label="st37_refine stage2 worker",
                    missing_label="st37_refine stage2",
                    invalid_label="st37_refine stage2",
                )

                tested_total = 0
                for worker_id in range(jobs_eff):
                    payload = payloads.get(int(worker_id))
                    if not isinstance(payload, dict):
                        continue
                    tested_total += int(payload.get("tested") or 0)
                    for rec in payload.get("records") or []:
                        if not isinstance(rec, dict):
                            continue
                        cfg = _cfg_from_payload(rec.get("strategy"), rec.get("filters"))
                        if cfg is None:
                            continue
                        row = rec.get("row")
                        if not isinstance(row, dict):
                            continue
                        note = str(rec.get("note") or "").strip() or "st37 stage2"
                        row = dict(row)
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        stage2_rows.append((cfg, row, note))

                tested_2 = int(tested_total)
        else:
            tested_2, kept_2 = _run_sweep(
                plan=stage2_plan,
                bars=bars_sig,
                total=stage2_total,
                progress_label="st37_refine stage2",
                report_every=report_every,
                heartbeat_sec=heartbeat_sec,
            )
            for cfg, row, note, _meta in kept_2:
                stage2_rows.append((cfg, row, note))

        print(f"st37_refine: stage2 kept={len(stage2_rows)} tested={tested_2}", flush=True)
        if not stage2_rows:
            return

        stage2_sorted = sorted(stage2_rows, key=lambda t: (_roi_dd(t[1]), float(t[1].get("pnl_over_dd") or 0.0)), reverse=True)
        stage2_shortlist: list[tuple[ConfigBundle, dict, str]] = []
        seen_cfg = set()
        for cfg, row, note in stage2_sorted:
            key = _milestone_key(cfg)
            if key in seen_cfg:
                continue
            seen_cfg.add(key)
            stage2_shortlist.append((cfg, row, note))
            if len(stage2_shortlist) >= 25:
                break

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
            try:
                v31_ptx = float(v31_exit.get("spot_pt_atr_mult") or v31_ptx)
            except (TypeError, ValueError):
                pass
            try:
                v31_slx = float(v31_exit.get("spot_sl_atr_mult") or v31_slx)
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
            return replace(
                base_cfg,
                strategy=replace(
                    base_cfg.strategy,
                    spot_exit_mode=str(exit_over.get("spot_exit_mode") or base_cfg.strategy.spot_exit_mode),
                    spot_profit_target_pct=exit_over.get("spot_profit_target_pct"),
                    spot_stop_loss_pct=exit_over.get("spot_stop_loss_pct"),
                    spot_atr_period=int(exit_over.get("spot_atr_period") or getattr(base_cfg.strategy, "spot_atr_period", 14)),
                    spot_pt_atr_mult=float(exit_over.get("spot_pt_atr_mult") or getattr(base_cfg.strategy, "spot_pt_atr_mult", 1.5)),
                    spot_sl_atr_mult=float(exit_over.get("spot_sl_atr_mult") or getattr(base_cfg.strategy, "spot_sl_atr_mult", 1.0)),
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
        tested_3, kept_3 = _run_sweep(
            plan=stage3_plan,
            bars=bars_sig,
            total=stage3_total,
            progress_label="st37_refine stage3",
            report_every=report_every,
            heartbeat_sec=heartbeat_sec,
        )
        for cfg, row, note, _meta in kept_3:
            stage3_rows.append((cfg, row, note))

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

        if args.gate_matrix_stage2:
            payload_path = Path(str(args.gate_matrix_stage2))
            out_path_raw = str(args.gate_matrix_out or "").strip()
            if not out_path_raw:
                raise SystemExit("--gate-matrix-out is required for gate_matrix stage2 worker mode.")
            out_path = Path(out_path_raw)

            worker_id, workers = _parse_worker_shard(
                args.gate_matrix_worker,
                args.gate_matrix_workers,
                label="gate_matrix",
            )

            try:
                payload = json.loads(payload_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid gate_matrix stage2 payload JSON: {payload_path}") from exc
            raw_seeds = payload.get("seeds") if isinstance(payload, dict) else None
            if not isinstance(raw_seeds, list):
                raise SystemExit(f"gate_matrix stage2 payload missing 'seeds' list: {payload_path}")

            seeds_local: list[tuple[ConfigBundle, str, str]] = []
            for item in raw_seeds:
                if not isinstance(item, dict):
                    continue
                strat_payload = item.get("strategy") or {}
                filters_payload = item.get("filters")
                seed_note = str(item.get("seed_note") or "")
                family = str(item.get("family") or "")
                if not isinstance(strat_payload, dict):
                    continue
                try:
                    filters_obj = _filters_from_payload(filters_payload)
                    strategy_obj = _strategy_from_payload(strat_payload, filters=filters_obj)
                except Exception:
                    continue
                seed_cfg = _mk_bundle(
                    strategy=strategy_obj,
                    start=start,
                    end=end,
                    bar_size=signal_bar_size,
                    use_rth=use_rth,
                    cache_dir=cache_dir,
                    offline=offline,
                )
                seeds_local.append((seed_cfg, seed_note, family))

            bars_sig = _bars_cached(signal_bar_size)
            total = len(seeds_local) * 2 * 2 * 2 * 2 * 2 * 2 * 2 * len(short_mults)
            local_total = (total // workers) + (1 if worker_id < (total % workers) else 0)

            stage2_plan_all: list[tuple[ConfigBundle, str, dict]] = []
            for seed_idx, (seed_cfg, seed_note, family) in enumerate(seeds_local):
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
                                                stage2_plan_all.append(
                                                    (
                                                        cfg,
                                                        note,
                                                        {
                                                            "seed_idx": seed_idx,
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

            if len(stage2_plan_all) != int(total):
                raise SystemExit(
                    f"gate_matrix stage2 worker internal error: combos={len(stage2_plan_all)} expected={total}"
                )

            shard_plan = (
                item for combo_idx, item in enumerate(stage2_plan_all) if (combo_idx % int(workers)) == int(worker_id)
            )
            tested, kept = _run_sweep(
                plan=shard_plan,
                bars=bars_sig,
                total=local_total,
                progress_label=f"gate_matrix stage2 worker {worker_id+1}/{workers}",
                report_every=100,
                record_milestones=False,
            )
            records: list[dict] = []
            for _cfg, row, _note, meta_item in kept:
                if not isinstance(meta_item, dict):
                    continue
                rec = dict(meta_item)
                rec["row"] = row
                records.append(rec)

            out_payload = {"tested": tested, "kept": len(records), "records": records}
            write_json(out_path, out_payload, sort_keys=False)
            print(f"gate_matrix stage2 worker done tested={tested} kept={len(records)} out={out_path}", flush=True)
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

        def _ranked(items: list[tuple[ConfigBundle, dict, str, str]], *, top_pnl_dd: int, top_pnl: int) -> list:
            by_dd = sorted(items, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[: int(top_pnl_dd)]
            by_pnl = sorted(items, key=lambda t: _score_row_pnl(t[1]), reverse=True)[: int(top_pnl)]
            seen: set[str] = set()
            out: list[tuple[ConfigBundle, dict, str, str]] = []
            for cfg, row, note, family in by_dd + by_pnl:
                key = _milestone_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                out.append((cfg, row, note, family))
            return out

        families = sorted({t[3] for t in stage1})
        seeds: list[tuple[ConfigBundle, dict, str, str]] = []
        for fam in families:
            seeds.extend(_ranked([t for t in stage1 if t[3] == fam], top_pnl_dd=3, top_pnl=2))
        # Keep this bounded.
        max_seeds = 8
        seeds = seeds[:max_seeds]
        print("")
        print(f"Gate-matrix: stage1 candidates={len(stage1)} seeds={len(seeds)} families={families}")

        # Stage 2: gate cross-product around the shortlist.
        rows: list[dict] = []
        total = len(seeds) * 2 * 2 * 2 * 2 * 2 * 2 * 2 * len(short_mults)
        if jobs > 1:
            if not offline:
                raise SystemExit("--jobs>1 for gate_matrix requires --offline (avoid parallel IBKR sessions).")

            base_cli = _strip_flags(
                list(sys.argv[1:]),
                flags=("--write-milestones", "--merge-milestones"),
                flags_with_values=(
                    "--axis",
                    "--jobs",
                    "--milestones-out",
                    "--gate-matrix-stage2",
                    "--gate-matrix-worker",
                    "--gate-matrix-workers",
                    "--gate-matrix-out",
                    "--gate-matrix-run-min-trades",
                ),
            )

            jobs_eff = min(int(jobs), int(_default_jobs()), int(total)) if total > 0 else 1
            print(f"gate_matrix stage2 parallel: workers={jobs_eff} total={total}", flush=True)

            with tempfile.TemporaryDirectory(prefix="tradebot_gate_matrix_") as tmpdir:
                tmp_root = Path(tmpdir)
                payload_path = tmp_root / "stage2_payload.json"
                seeds_payload: list[dict] = []
                for seed_cfg, _, seed_note, family in seeds:
                    seeds_payload.append(
                        {
                            "strategy": _spot_strategy_payload(seed_cfg, meta=meta),
                            "filters": _filters_payload(seed_cfg.strategy.filters),
                            "seed_note": str(seed_note),
                            "family": str(family),
                        }
                    )
                write_json(payload_path, {"seeds": seeds_payload}, sort_keys=False)

                payloads = _run_parallel_json_worker_plan(
                    jobs_eff=jobs_eff,
                    tmp_prefix="tradebot_gate_matrix2_",
                    worker_tag="gm2",
                    out_prefix="stage2_out",
                    build_cmd=lambda worker_id, workers_n, out_path: [
                        sys.executable,
                        "-u",
                        "-m",
                        "tradebot.backtest",
                        "spot",
                        *base_cli,
                        "--axis",
                        "gate_matrix",
                        "--jobs",
                        "1",
                        "--gate-matrix-stage2",
                        str(payload_path),
                        "--gate-matrix-worker",
                        str(worker_id),
                        "--gate-matrix-workers",
                        str(workers_n),
                        "--gate-matrix-out",
                        str(out_path),
                        "--gate-matrix-run-min-trades",
                        str(int(run_min_trades)),
                    ],
                    capture_error="Failed to capture gate_matrix worker stdout.",
                    failure_label="gate_matrix stage2 worker",
                    missing_label="gate_matrix stage2",
                    invalid_label="gate_matrix stage2",
                )

                tested_total = 0
                for worker_id in range(jobs_eff):
                    payload = payloads.get(int(worker_id))
                    if not isinstance(payload, dict):
                        continue
                    tested_total += int(payload.get("tested") or 0)
                    for rec in payload.get("records") or []:
                        if not isinstance(rec, dict):
                            continue
                        seed_idx_raw = rec.get("seed_idx")
                        if seed_idx_raw is None:
                            continue
                        try:
                            seed_idx = int(seed_idx_raw)
                        except (TypeError, ValueError):
                            continue
                        if seed_idx < 0 or seed_idx >= len(seeds):
                            continue
                        base_seed_cfg, _, seed_note, family = seeds[seed_idx]
                        cfg, note = _mk_stage2_cfg(
                            base_seed_cfg,
                            str(seed_note),
                            str(family),
                            perm_on=bool(rec.get("perm_on")),
                            tick_on=bool(rec.get("tick_on")),
                            shock_on=bool(rec.get("shock_on")),
                            riskoff_on=bool(rec.get("riskoff_on")),
                            riskpanic_on=bool(rec.get("riskpanic_on")),
                            riskpop_on=bool(rec.get("riskpop_on")),
                            regime2_on=bool(rec.get("regime2_on")),
                            short_mult=float(rec.get("short_mult") or 0.0),
                        )
                        row = rec.get("row")
                        if not isinstance(row, dict):
                            continue
                        row = dict(row)
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

                run_calls_total += int(tested_total)
        else:
            stage2_plan: list[tuple[ConfigBundle, str, dict | None]] = []
            for seed_cfg, _, seed_note, family in seeds:
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
                                                stage2_plan.append((cfg, note, None))
            _tested, kept = _run_sweep(
                plan=stage2_plan,
                bars=bars_sig,
                total=total,
                progress_label="gate_matrix stage2",
                report_every=200,
            )
            rows.extend(row for _cfg, row, _note, _meta in kept)

        _print_leaderboards(rows, title="Gate-matrix sweep (bounded cross-product)", top_n=int(args.top))

    def _sweep_squeeze() -> None:
        """Squeeze a few high-leverage axes from the current champion baseline.

        Targeted (fast): regime2 timeframe, volume gate, and time-of-day windows,
        including small combinations of these axes.
        """
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        def _shortlist(items: list[tuple[ConfigBundle, dict, str]], *, top_pnl_dd: int, top_pnl: int) -> list:
            by_dd = sorted(items, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[: int(top_pnl_dd)]
            by_pnl = sorted(items, key=lambda t: _score_row_pnl(t[1]), reverse=True)[: int(top_pnl)]
            seen: set[str] = set()
            out: list[tuple[ConfigBundle, dict, str]] = []
            for cfg, row, note in by_dd + by_pnl:
                key = _milestone_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                out.append((cfg, row, note))
            return out

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
        shortlisted = _shortlist(stage1, top_pnl_dd=15, top_pnl=10)
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
        if not offline:
            raise SystemExit("--jobs>1 for --axis all requires --offline (avoid parallel IBKR sessions).")

        axes = tuple(_AXIS_ALL_PLAN)

        base_cli = _strip_flags(
            list(sys.argv[1:]),
            flags=("--merge-milestones",),
            flags_with_values=("--axis", "--jobs", "--milestones-out"),
        )

        milestone_payloads = _run_axis_subprocess_plan(
            label="axis=all parallel",
            axes=axes,
            jobs=int(jobs),
            base_cli=base_cli,
            axis_jobs_resolver=lambda axis_name: min(int(jobs), int(_default_jobs()))
            if str(axis_name) == "risk_overlays"
            else 1,
            write_milestones=bool(args.write_milestones),
            tmp_prefix="tradebot_axis_all_",
        )
        if bool(args.write_milestones):
            eligible_new: list[dict] = []
            for axis_name in axes:
                payload = milestone_payloads.get(axis_name)
                if isinstance(payload, dict):
                    eligible_new.extend(_collect_milestone_items_from_payload(payload, symbol=symbol))
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

        return

    sweep_registry = {
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
        "champ_refine": _sweep_champ_refine,
        "st37_refine": _sweep_st37_refine,
        "shock_alpha_refine": _sweep_shock_alpha_refine,
        "shock_velocity_refine": lambda: _sweep_shock_velocity_refine(wide=False),
        "shock_velocity_refine_wide": lambda: _sweep_shock_velocity_refine(wide=True),
        "shock_throttle_refine": _sweep_shock_throttle_refine,
        "shock_throttle_tr_ratio": _sweep_shock_throttle_tr_ratio,
        "shock_throttle_drawdown": _sweep_shock_throttle_drawdown,
        "riskpanic_micro": _sweep_riskpanic_micro,
        "exit_pivot": _sweep_exit_pivot,
    }

    if axis == "all":
        for axis_name in _AXIS_ALL_PLAN:
            fn = sweep_registry.get(str(axis_name))
            if fn is not None:
                fn()
    else:
        fn = sweep_registry.get(str(axis))
        if fn is not None:
            fn()

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


def _strategy_from_payload(strategy: dict, *, filters: FiltersConfig | None) -> StrategyConfig:
    if not isinstance(strategy, dict):
        raise ValueError(f"strategy must be an object, got: {strategy!r}")

    raw = dict(strategy)
    raw.pop("signal_bar_size", None)
    raw.pop("signal_use_rth", None)
    raw.pop("spot_sec_type", None)
    raw.pop("spot_exchange", None)

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

    return StrategyConfig(**raw)


def _mk_bundle(
    *,
    strategy: StrategyConfig,
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
        help=(
            "Require strategy.max_open_trades to be <= this value. "
            "Note: 0 means unlimited stacking and is rejected unless --allow-unlimited-stacking is set."
        ),
    )
    ap.add_argument(
        "--allow-unlimited-stacking",
        action="store_true",
        default=False,
        help="Allow max_open_trades=0 strategies (unlimited stacking).",
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
        filters_payload = cand.get("filters")
        filters = _filters_from_payload(filters_payload)
        strategy_payload = cand["strategy"]
        strat_cfg = _strategy_from_payload(strategy_payload, filters=filters)
        if bool(args.require_close_eod) and not bool(getattr(strat_cfg, "spot_close_eod", False)):
            return None
        if args.max_open is not None:
            max_open = int(getattr(strat_cfg, "max_open_trades", 0) or 0)
            if max_open == 0 and not bool(args.allow_unlimited_stacking):
                return None
            if max_open != 0 and max_open > int(args.max_open):
                return None

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
                return None
            req_trades = _required_trades_for_window(wstart, wend)
            if m["trades"] < int(req_trades) or m["win_rate"] < float(args.min_win):
                return None
            per_window.append(
                {
                    "start": wstart.isoformat(),
                    "end": wend.isoformat(),
                    **m,
                }
            )

        if not per_window:
            return None
        min_pnl_dd = min(float(x["pnl_over_dd"]) for x in per_window)
        min_pnl = min(float(x["pnl"]) for x in per_window)
        min_roi_dd = min(float(x.get("roi_over_dd_pct") or 0.0) for x in per_window)
        min_roi = min(float(x.get("roi") or 0.0) for x in per_window)
        primary = per_window[0] if per_window else {}
        return {
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
        for idx, cand in enumerate(candidates, start=1):
            if ((idx - 1) % int(workers)) != int(worker_id):
                continue
            tested += 1
            row = _evaluate_candidate_multiwindow(cand, load_bars_cached=load_bars_cached, data=data)
            if row is not None:
                out_rows.append(row)

            if tested % report_every != 0:
                continue
            elapsed = pytime.perf_counter() - started
            rate = tested / elapsed if elapsed > 0 else 0.0
            if progress_mode == "worker":
                print(
                    f"multitimeframe worker {worker_id+1}/{workers} tested={tested} kept={len(out_rows)} "
                    f"({rate:0.2f} cands/s)",
                    flush=True,
                )
            else:
                print(f"[{tested}/{len(candidates)}] evaluated… ({rate:0.2f} cands/s, kept={len(out_rows)})", flush=True)
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

        print(f"multitimeframe parallel: workers={jobs_eff} candidates={len(candidates)}", flush=True)

        payloads = _run_parallel_json_worker_plan(
            jobs_eff=jobs_eff,
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

        tested_total = 0
        out_rows: list[dict] = []
        for worker_id in range(jobs_eff):
            payload = payloads.get(int(worker_id))
            if not isinstance(payload, dict):
                continue
            tested_total += int(payload.get("tested") or 0)
            for row in payload.get("rows") or []:
                if isinstance(row, dict):
                    out_rows.append(row)

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
