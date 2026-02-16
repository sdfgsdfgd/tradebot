#!/usr/bin/env python3
"""Temporary SLV 10m high-frequency attack runner.

Profiles:
- hyper10: dual lane (RTH + FULL24), pnl-first ranking, TQQQ+SLV transfer overlays.
- scalp10: FULL24-heavy aggressive lane, pnl-first ranking.
- precision_guard: km50/km52-centered quality pocket with narrow crisis overlays.
- hour_expansion: FULL24 hour-window expansion with selective protection overlays.

This script intentionally calls canonical `spot_multitimeframe` so we keep
cache/sharding/engine behavior identical to the main backtest stack.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

RE_KINGMAKER = re.compile(r"KINGMAKER\s*#(\d+)", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_window(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if ":" not in text:
        raise SystemExit(f"Invalid window (expected YYYY-MM-DD:YYYY-MM-DD): {raw!r}")
    start_s, end_s = [x.strip() for x in text.split(":", 1)]
    if len(start_s) != 10 or len(end_s) != 10:
        raise SystemExit(f"Invalid window (expected YYYY-MM-DD:YYYY-MM-DD): {raw!r}")
    return start_s, end_s


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def _blank_metrics() -> dict[str, float | int]:
    return {
        "pnl": 0.0,
        "roi": 0.0,
        "win_rate": 0.0,
        "trades": 0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "pnl_over_dd": 0.0,
        "roi_over_dd_pct": 0.0,
    }


def _merge_dict(base: dict | None, overrides: dict[str, object]) -> dict:
    out = dict(base or {})
    for key, value in overrides.items():
        if value is None:
            out.pop(str(key), None)
        else:
            out[str(key)] = value
    return out


def _payload_from_groups(
    *,
    name: str,
    source: str,
    windows: list[dict[str, str]],
    groups: list[dict],
    notes: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "generated_at": _utc_now_iso(),
        "source": source,
        "windows": windows,
        "groups": groups,
        "notes": list(notes or []),
    }


def _run_and_tee(*, cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(cmd)}")
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"Command failed ({rc}): {shlex.join(cmd)}")


def _iter_groups(payload: dict, *, symbol: str, bar_size: str) -> Iterable[dict]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return
    symbol_u = str(symbol).strip().upper()
    bar_l = str(bar_size).strip().lower()
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries")
        if not isinstance(entries, list) or not entries:
            continue
        e0 = entries[0]
        if not isinstance(e0, dict):
            continue
        strategy = e0.get("strategy")
        if not isinstance(strategy, dict):
            continue
        if str(strategy.get("symbol") or "").strip().upper() != symbol_u:
            continue
        if str(strategy.get("signal_bar_size") or "").strip().lower() != bar_l:
            continue
        yield group


def _group_lane_is_rth(group: dict) -> bool:
    entries = group.get("entries")
    if not isinstance(entries, list) or not entries:
        return False
    e0 = entries[0]
    if not isinstance(e0, dict):
        return False
    strategy = e0.get("strategy")
    if not isinstance(strategy, dict):
        return False
    return bool(strategy.get("signal_use_rth"))


def _group_metrics(group: dict) -> dict:
    entries = group.get("entries")
    if not isinstance(entries, list) or not entries:
        return _blank_metrics()
    e0 = entries[0]
    if not isinstance(e0, dict):
        return _blank_metrics()
    metrics = e0.get("metrics")
    if not isinstance(metrics, dict):
        return _blank_metrics()
    return {
        "pnl": float(metrics.get("pnl") or 0.0),
        "roi": float(metrics.get("roi") or 0.0),
        "win_rate": float(metrics.get("win_rate") or 0.0),
        "trades": int(metrics.get("trades") or 0),
        "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
        "pnl_over_dd": float(metrics.get("pnl_over_dd") or 0.0),
        "roi_over_dd_pct": float(metrics.get("roi_over_dd_pct") or 0.0),
    }


def _sort_groups_pnl_first(groups: list[dict]) -> list[dict]:
    def _key(g: dict) -> tuple:
        m = _group_metrics(g)
        return (
            float(m.get("pnl") or 0.0),
            int(m.get("trades") or 0),
            float(m.get("roi") or 0.0),
            -float(m.get("max_drawdown_pct") or 0.0),
            float(m.get("roi_over_dd_pct") or 0.0),
        )

    return sorted(list(groups), key=_key, reverse=True)


def _seed_from_group(group: dict) -> tuple[str, dict, dict]:
    name = str(group.get("name") or "")
    entries = group.get("entries") or [{}]
    e0 = entries[0] if entries else {}
    strategy = dict((e0.get("strategy") or {}))
    filters = dict((group.get("filters") or {}))
    return name, strategy, filters


def _extract_kingmaker_id(name: str) -> int | None:
    m = RE_KINGMAKER.search(str(name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _pick_group(
    groups: list[dict],
    *,
    kingmaker_id: int | None,
    label: str,
) -> dict:
    if not groups:
        raise SystemExit(f"No groups available for {label}.")
    if kingmaker_id is None:
        return groups[0]
    for group in groups:
        gid = _extract_kingmaker_id(str(group.get("name") or ""))
        if gid == int(kingmaker_id):
            return group
    raise SystemExit(f"Could not find KINGMAKER #{int(kingmaker_id):02d} in {label}.")


def _run_multitimeframe(
    *,
    repo_root: Path,
    milestones: Path,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    offline: bool,
    cache_dir: Path,
    jobs: int,
    top: int,
    min_trades: int,
    min_trades_per_year: float | None,
    min_win: float,
    windows: list[str],
    require_positive_pnl: bool,
    out_path: Path,
    log_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tradebot.backtest",
        "spot_multitimeframe",
        "--milestones",
        str(milestones),
        "--symbol",
        str(symbol),
        "--bar-size",
        str(bar_size),
        "--cache-dir",
        str(cache_dir),
        "--jobs",
        str(int(jobs)),
        "--top",
        str(int(top)),
        "--min-trades",
        str(int(min_trades)),
        "--min-win",
        str(float(min_win)),
        "--write-top",
        str(int(top)),
        "--out",
        str(out_path),
    ]
    if use_rth:
        cmd.append("--use-rth")
    if offline:
        cmd.append("--offline")
    if require_positive_pnl:
        cmd.append("--require-positive-pnl")
    if min_trades_per_year is not None:
        cmd.extend(["--min-trades-per-year", str(float(min_trades_per_year))])
    for window in windows:
        cmd.extend(["--window", str(window)])
    _run_and_tee(cmd=cmd, cwd=repo_root, log_path=log_path)


def _evaluate_payload_by_lane(
    *,
    repo_root: Path,
    in_payload_path: Path,
    out_payload_path: Path,
    run_tag: str,
    symbol: str,
    bar_size: str,
    cache_dir: Path,
    jobs: int,
    offline: bool,
    windows: list[str],
    min_trades: int,
    min_trades_per_year: float | None,
    require_positive_pnl: bool,
    lane_mode: str,
    top_keep: int | None = None,
) -> dict:
    payload = _read_json(in_payload_path)
    groups_all = list(_iter_groups(payload, symbol=symbol, bar_size=bar_size))
    if not groups_all:
        out = _payload_from_groups(
            name=f"{run_tag}_empty",
            source=str(in_payload_path),
            windows=payload.get("windows") or [],
            groups=[],
            notes=["No candidates after symbol/bar filter"],
        )
        _write_json(out_payload_path, out)
        return {"tested": 0, "kept": 0}

    lane_groups: dict[bool, list[dict]] = {True: [], False: []}
    for g in groups_all:
        lane_groups[_group_lane_is_rth(g)].append(g)

    kept_merged: list[dict] = []
    tested_total = 0
    lane_mode_norm = str(lane_mode or "all").strip().lower()
    if lane_mode_norm == "rth":
        lane_order = (True,)
    elif lane_mode_norm in ("full24", "full", "24h", "24/5"):
        lane_order = (False,)
    else:
        lane_order = (True, False)

    for lane in lane_order:
        groups_lane = lane_groups.get(lane) or []
        if not groups_lane:
            continue
        lane_name = "rth" if lane else "full24"
        lane_in = out_payload_path.with_name(f"{out_payload_path.stem}_{lane_name}_in.json")
        lane_out = out_payload_path.with_name(f"{out_payload_path.stem}_{lane_name}_raw.json")
        lane_log = out_payload_path.with_name(f"{out_payload_path.stem}_{lane_name}.log")
        lane_payload = _payload_from_groups(
            name=f"{run_tag}_{lane_name}_lane",
            source=str(in_payload_path),
            windows=payload.get("windows") or [],
            groups=groups_lane,
            notes=[f"lane={lane_name}", f"count={len(groups_lane)}"],
        )
        _write_json(lane_in, lane_payload)
        _run_multitimeframe(
            repo_root=repo_root,
            milestones=lane_in,
            symbol=symbol,
            bar_size=bar_size,
            use_rth=bool(lane),
            offline=bool(offline),
            cache_dir=cache_dir,
            jobs=int(jobs),
            top=len(groups_lane),
            min_trades=int(min_trades),
            min_trades_per_year=min_trades_per_year,
            min_win=0.0,
            windows=windows,
            require_positive_pnl=bool(require_positive_pnl),
            out_path=lane_out,
            log_path=lane_log,
        )
        lane_eval = _read_json(lane_out)
        lane_kept = lane_eval.get("groups") if isinstance(lane_eval, dict) else None
        if isinstance(lane_kept, list):
            kept_merged.extend(lane_kept)
        tested_total += len(groups_lane)

    ranked = _sort_groups_pnl_first(kept_merged)
    if top_keep is not None and int(top_keep) > 0:
        ranked = ranked[: int(top_keep)]
    out = _payload_from_groups(
        name=run_tag,
        source=str(in_payload_path),
        windows=payload.get("windows") or [],
        groups=ranked,
        notes=[
            "rank=pnl_first",
            f"tested={tested_total}",
            f"kept={len(kept_merged)}",
            f"output_kept={len(ranked)}",
        ],
    )
    _write_json(out_payload_path, out)
    return {"tested": tested_total, "kept": len(kept_merged), "output_kept": len(ranked)}


def _neutral_filters(base_filters: dict) -> dict:
    return _merge_dict(
        base_filters,
        {
            "shock_gate_mode": "off",
            "shock_detector": None,
            "shock_scale_detector": None,
            "shock_risk_scale_target_atr_pct": None,
            "shock_risk_scale_min_mult": None,
            "shock_risk_scale_apply_to": None,
            "shock_drawdown_lookback_days": None,
            "riskoff_mode": None,
            "riskoff_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": None,
            "riskpanic_neg_gap_ratio_min": None,
            "riskpanic_neg_gap_abs_pct_min": None,
            "riskpanic_lookback_days": None,
            "riskpanic_tr5_med_delta_min_pct": None,
            "riskpanic_tr5_med_delta_lookback_days": None,
            "riskpanic_long_risk_mult_factor": None,
            "riskpanic_short_risk_mult_factor": None,
            "riskpanic_long_scale_mode": None,
            "riskpanic_long_scale_tr_delta_max_pct": None,
            "risk_entry_cutoff_hour_et": None,
            "riskpop_tr5_med_pct": None,
            "riskpop_pos_gap_ratio_min": None,
            "riskpop_pos_gap_abs_pct_min": None,
        },
    )


def _hyper10_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    filters_base = _neutral_filters(base_filters)
    groups: list[dict] = []
    rank = 0

    ema_presets = ("3/7", "4/9", "5/13", "8/21")
    regime_profiles: tuple[tuple[str, dict[str, object]], ...] = (
        ("regime=off", {"regime_mode": "off"}),
        (
            "regime=10m_st7_0.4_close",
            {
                "regime_mode": "supertrend",
                "regime_bar_size": "10 mins",
                "supertrend_atr_period": 7,
                "supertrend_multiplier": 0.4,
                "supertrend_source": "close",
            },
        ),
        (
            "regime=4h_st7_0.5_hl2",
            {
                "regime_mode": "supertrend",
                "regime_bar_size": "4 hours",
                "supertrend_atr_period": 7,
                "supertrend_multiplier": 0.5,
                "supertrend_source": "hl2",
            },
        ),
    )
    for signal_use_rth in (True, False):
        for ema in ema_presets:
            for regime_note, regime_over in regime_profiles:
                for stop_loss in (0.006, 0.008):
                    for close_eod in (False, True):
                        for flip_only_profit in (False, True):
                            for flip_hold in (0, 1):
                                rank += 1
                                strategy = dict(base_strategy)
                                strategy.update(
                                    {
                                        "instrument": "spot",
                                        "symbol": str(symbol).strip().upper(),
                                        "signal_bar_size": "10 mins",
                                        "signal_use_rth": bool(signal_use_rth),
                                        "spot_exec_bar_size": "5 mins",
                                        "ema_entry_mode": "trend",
                                        "entry_confirm_bars": 0,
                                        "ema_preset": str(ema),
                                        "spot_stop_loss_pct": float(stop_loss),
                                        "spot_close_eod": bool(close_eod),
                                        "exit_on_signal_flip": True,
                                        "flip_exit_mode": "entry",
                                        "flip_exit_only_if_profit": bool(flip_only_profit),
                                        "flip_exit_min_hold_bars": int(flip_hold),
                                        "regime2_mode": "off",
                                    }
                                )
                                strategy.update(regime_over)
                                if strategy.get("regime_mode") == "off":
                                    strategy["regime_bar_size"] = "10 mins"
                                strategy.pop("regime2_bar_size", None)
                                strategy.pop("regime2_supertrend_atr_period", None)
                                strategy.pop("regime2_supertrend_multiplier", None)
                                strategy.pop("regime2_supertrend_source", None)
                                strategy.pop("regime2_ema_preset", None)

                                filters = dict(filters_base)
                                if signal_use_rth:
                                    filters = _merge_dict(filters, {"entry_start_hour_et": 9, "entry_end_hour_et": 16})
                                else:
                                    filters = _merge_dict(filters, {"entry_start_hour_et": None, "entry_end_hour_et": None})

                                note = (
                                    f"HYPERA #{rank:04d} lane={'RTH' if signal_use_rth else 'FULL24'} "
                                    f"ema={ema} {regime_note} sl={stop_loss:g} "
                                    f"eod={int(close_eod)} flip_prof={int(flip_only_profit)} hold={flip_hold}"
                                )
                                groups.append(
                                    {
                                        "name": note,
                                        "filters": filters,
                                        "entries": [
                                            {
                                                "symbol": str(symbol).strip().upper(),
                                                "metrics": _blank_metrics(),
                                                "strategy": strategy,
                                            }
                                        ],
                                        "_eval": {"stage": "A", "profile": "hyper10", "note": note},
                                    }
                                )
    return groups


def _hyper10_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (10, 8.0, 0.05, "both"),
        (20, 8.0, 0.05, "both"),
        (10, 10.0, 0.05, "both"),
        (10, 8.0, 0.10, "both"),
    )
    trratio_specs: tuple[tuple[int, int, float, float, float, float, float, str], ...] = (
        (3, 50, 1.30, 1.20, 3.0, 8.0, 0.05, "both"),
        (3, 50, 1.30, 1.20, 3.0, 12.0, 0.20, "both"),
        (5, 50, 1.40, 1.30, 3.0, 8.0, 0.10, "cap"),
        (5, 50, 1.40, 1.30, 4.0, 12.0, 0.20, "both"),
        (3, 21, 1.30, 1.20, 3.0, 8.0, 0.20, "cap"),
        (5, 50, 1.50, 1.40, 4.0, 12.0, 0.20, "both"),
        (3, 50, 1.40, 1.30, 3.0, 8.0, 0.05, "both"),
        (5, 50, 1.30, 1.20, 3.0, 6.0, 0.05, "cap"),
    )
    risk_specs: list[tuple[float, float, float]] = []
    for tr_med in (2.75, 3.0, 3.25):
        for tr_delta in (0.25, 0.5):
            for long_factor in (0.2, 0.0):
                risk_specs.append((float(tr_med), float(tr_delta), float(long_factor)))

    cross_specs: tuple[tuple[int, tuple[int, float, float, str], tuple[int, int, float, float, float, float, float, str], tuple[float, float, float]], ...] = (
        (1, drawdown_specs[0], trratio_specs[0], (2.75, 0.25, 0.2)),
        (2, drawdown_specs[1], trratio_specs[2], (3.0, 0.25, 0.2)),
        (3, drawdown_specs[2], trratio_specs[3], (3.0, 0.5, 0.0)),
        (4, drawdown_specs[3], trratio_specs[5], (3.25, 0.5, 0.0)),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
        }

    def _over_riskpanic(tr_med: float, tr_delta: float, long_factor: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "risk_entry_cutoff_hour_et": 15,
            "riskpanic_long_risk_mult_factor": float(long_factor),
            "riskpanic_short_risk_mult_factor": 1.0,
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    symbol_u = str(symbol).strip().upper()
    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            f = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"HYPERB #{rank:04d} seed={seed_idx} dd={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": f,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "hyper10", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in trratio_specs:
            rank += 1
            f = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"HYPERB #{rank:04d} seed={seed_idx} trratio={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": f,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "hyper10", "seed": seed_name, "kind": "trratio_only"},
                }
            )

        for tr_med, tr_delta, long_factor in risk_specs:
            for short_mult in (0.02, 0.01):
                rank += 1
                f = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta, long_factor))
                s = dict(seed_strategy)
                s["spot_short_risk_mult"] = float(short_mult)
                note = (
                    f"HYPERB #{rank:04d} seed={seed_idx} panic(tr={tr_med:g},d={tr_delta:g},L={long_factor:g})"
                    f" short={short_mult:g}"
                )
                groups.append(
                    {
                        "name": note,
                        "filters": f,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": s}],
                        "_eval": {"stage": "B", "profile": "hyper10", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd, trr, rp in cross_specs:
            for short_mult in (0.02, 0.01):
                rank += 1
                f = _merge_dict(seed_filters, _over_dd(*dd))
                f = _merge_dict(f, _over_trratio(*trr))
                f = _merge_dict(f, _over_riskpanic(*rp))
                s = dict(seed_strategy)
                s["spot_short_risk_mult"] = float(short_mult)
                note = f"HYPERB #{rank:04d} seed={seed_idx} cross={cross_idx} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": f,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": s}],
                        "_eval": {"stage": "B", "profile": "hyper10", "seed": seed_name, "kind": "cross"},
                    }
                )
    return groups


def _scalp10_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    filters_base = _neutral_filters(base_filters)
    groups: list[dict] = []
    rank = 0

    ema_presets = ("2/4", "3/7", "4/9")
    regime_profiles: tuple[tuple[str, dict[str, object]], ...] = (
        ("regime=off", {"regime_mode": "off"}),
        (
            "regime=10m_st7_0.35_close",
            {
                "regime_mode": "supertrend",
                "regime_bar_size": "10 mins",
                "supertrend_atr_period": 7,
                "supertrend_multiplier": 0.35,
                "supertrend_source": "close",
            },
        ),
        (
            "regime=10m_st10_0.4_close",
            {
                "regime_mode": "supertrend",
                "regime_bar_size": "10 mins",
                "supertrend_atr_period": 10,
                "supertrend_multiplier": 0.4,
                "supertrend_source": "close",
            },
        ),
    )

    for ema in ema_presets:
        for regime_note, regime_over in regime_profiles:
            for stop_loss in (0.004, 0.006, 0.008):
                for short_mult in (0.05, 0.02):
                    for flip_hold in (0, 1):
                        for confirm_bars in (0, 1):
                            rank += 1
                            strategy = dict(base_strategy)
                            strategy.update(
                                {
                                    "instrument": "spot",
                                    "symbol": str(symbol).strip().upper(),
                                    "signal_bar_size": "10 mins",
                                    "signal_use_rth": False,
                                    "spot_exec_bar_size": "5 mins",
                                    "ema_entry_mode": "trend",
                                    "entry_confirm_bars": int(confirm_bars),
                                    "ema_preset": str(ema),
                                    "spot_stop_loss_pct": float(stop_loss),
                                    "spot_close_eod": False,
                                    "exit_on_signal_flip": True,
                                    "flip_exit_mode": "entry",
                                    "flip_exit_only_if_profit": False,
                                    "flip_exit_min_hold_bars": int(flip_hold),
                                    "regime2_mode": "off",
                                    "spot_short_risk_mult": float(short_mult),
                                }
                            )
                            strategy.update(regime_over)
                            if strategy.get("regime_mode") == "off":
                                strategy["regime_bar_size"] = "10 mins"
                            strategy.pop("regime2_bar_size", None)
                            strategy.pop("regime2_supertrend_atr_period", None)
                            strategy.pop("regime2_supertrend_multiplier", None)
                            strategy.pop("regime2_supertrend_source", None)
                            strategy.pop("regime2_ema_preset", None)

                            filters = _merge_dict(filters_base, {"entry_start_hour_et": None, "entry_end_hour_et": None})
                            note = (
                                f"SCALPA #{rank:04d} FULL24 ema={ema} {regime_note} "
                                f"sl={stop_loss:g} short={short_mult:g} hold={flip_hold} confirm={confirm_bars}"
                            )
                            groups.append(
                                {
                                    "name": note,
                                    "filters": filters,
                                    "entries": [
                                        {
                                            "symbol": str(symbol).strip().upper(),
                                            "metrics": _blank_metrics(),
                                            "strategy": strategy,
                                        }
                                    ],
                                    "_eval": {"stage": "A", "profile": "scalp10", "note": note},
                                }
                            )
    return groups


def _scalp10_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (5, 6.0, 0.10, "both"),
        (10, 8.0, 0.10, "both"),
        (10, 8.0, 0.20, "both"),
        (20, 10.0, 0.10, "both"),
    )
    trratio_specs: tuple[tuple[int, int, float, float, float, float, float, str], ...] = (
        (3, 21, 1.25, 1.15, 2.5, 6.0, 0.10, "both"),
        (3, 21, 1.30, 1.20, 3.0, 8.0, 0.10, "both"),
        (3, 50, 1.30, 1.20, 3.0, 8.0, 0.10, "both"),
        (3, 50, 1.35, 1.25, 3.0, 10.0, 0.20, "both"),
        (5, 50, 1.40, 1.30, 4.0, 10.0, 0.20, "cap"),
        (5, 50, 1.45, 1.35, 4.0, 12.0, 0.20, "both"),
    )
    risk_specs: tuple[tuple[float, float], ...] = (
        (2.5, 0.25),
        (2.5, 0.5),
        (2.75, 0.25),
        (2.75, 0.5),
        (3.0, 0.25),
        (3.0, 0.5),
    )
    cross_specs: tuple[tuple[int, tuple[int, float, float, str], tuple[int, int, float, float, float, float, float, str], tuple[float, float]], ...] = (
        (1, drawdown_specs[0], trratio_specs[1], risk_specs[2]),
        (2, drawdown_specs[1], trratio_specs[3], risk_specs[3]),
        (3, drawdown_specs[2], trratio_specs[4], risk_specs[5]),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
        }

    def _over_riskpanic(tr_med: float, tr_delta: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "risk_entry_cutoff_hour_et": 15,
            "riskpanic_long_risk_mult_factor": 0.0,
            "riskpanic_short_risk_mult_factor": 1.0,
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    symbol_u = str(symbol).strip().upper()
    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            f = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"SCALPB #{rank:04d} seed={seed_idx} dd={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": f,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "scalp10", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in trratio_specs:
            rank += 1
            f = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"SCALPB #{rank:04d} seed={seed_idx} trratio={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": f,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "scalp10", "seed": seed_name, "kind": "trratio_only"},
                }
            )

        for tr_med, tr_delta in risk_specs:
            for short_mult in (0.05, 0.02):
                rank += 1
                f = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta))
                s = dict(seed_strategy)
                s["spot_short_risk_mult"] = float(short_mult)
                note = f"SCALPB #{rank:04d} seed={seed_idx} panic={tr_med:g}/{tr_delta:g} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": f,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": s}],
                        "_eval": {"stage": "B", "profile": "scalp10", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd, trr, rp in cross_specs:
            for short_mult in (0.05, 0.02):
                rank += 1
                f = _merge_dict(seed_filters, _over_dd(*dd))
                f = _merge_dict(f, _over_trratio(*trr))
                f = _merge_dict(f, _over_riskpanic(*rp))
                s = dict(seed_strategy)
                s["spot_short_risk_mult"] = float(short_mult)
                note = f"SCALPB #{rank:04d} seed={seed_idx} cross={cross_idx} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": f,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": s}],
                        "_eval": {"stage": "B", "profile": "scalp10", "seed": seed_name, "kind": "cross"},
                    }
                )
    return groups


def _island_bridge_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()
    filters_base = dict(base_filters or {})
    tod_specs: tuple[tuple[str, dict[str, object]], ...] = (
        ("tod=8-15", {"entry_start_hour_et": 8, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
        ("tod=9-16", {"entry_start_hour_et": 9, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
        ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None, "risk_entry_cutoff_hour_et": None}),
    )

    for ema_preset in ("8/21", "5/13"):
        for confirm_bars in (1, 0):
            for stop_loss in (0.018, 0.02):
                for tod_note, tod_over in tod_specs:
                    for short_mult in (0.0, 0.0025):
                        rank += 1
                        strategy = dict(base_strategy)
                        strategy.update(
                            {
                                "instrument": "spot",
                                "symbol": symbol_u,
                                "signal_bar_size": "10 mins",
                                "signal_use_rth": False,
                                "spot_exec_bar_size": "5 mins",
                                "ema_entry_mode": "trend",
                                "ema_preset": str(ema_preset),
                                "entry_confirm_bars": int(confirm_bars),
                                "spot_stop_loss_pct": float(stop_loss),
                                "spot_close_eod": False,
                                "exit_on_signal_flip": True,
                                "flip_exit_mode": "entry",
                                "flip_exit_only_if_profit": True,
                                "flip_exit_min_hold_bars": 0,
                                "regime_mode": "supertrend",
                                "regime2_mode": "off",
                                "spot_short_risk_mult": float(short_mult),
                            }
                        )
                        strategy.pop("regime2_bar_size", None)
                        strategy.pop("regime2_supertrend_atr_period", None)
                        strategy.pop("regime2_supertrend_multiplier", None)
                        strategy.pop("regime2_supertrend_source", None)
                        strategy.pop("regime2_ema_preset", None)

                        filters = _merge_dict(filters_base, tod_over)
                        note = (
                            f"IBA #{rank:04d} FULL24 ema={ema_preset} confirm={confirm_bars} "
                            f"sl={stop_loss:g} {tod_note} short={short_mult:g}"
                        )
                        groups.append(
                            {
                                "name": note,
                                "filters": filters,
                                "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                                "_eval": {"stage": "A", "profile": "island_bridge", "note": note},
                            }
                        )
    return groups


def _island_bridge_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (10, 8.0, 0.05, "both"),
        (20, 8.0, 0.05, "both"),
        (10, 10.0, 0.05, "both"),
        (10, 8.0, 0.10, "both"),
    )
    trratio_specs: tuple[tuple[int, int, float, float, float, float, float, str, str], ...] = (
        (3, 50, 1.30, 1.20, 3.0, 12.0, 0.20, "both", "detect"),
        (3, 50, 1.30, 1.20, 3.0, 10.0, 0.10, "both", "detect"),
        (5, 50, 1.40, 1.30, 4.0, 12.0, 0.20, "both", "detect"),
        (5, 50, 1.40, 1.30, 4.0, 10.0, 0.10, "cap", "detect"),
        (3, 21, 1.30, 1.20, 3.0, 10.0, 0.20, "both", "surf"),
        (3, 21, 1.25, 1.15, 2.5, 8.0, 0.10, "both", "surf"),
    )
    risk_specs: tuple[tuple[float, float, float], ...] = (
        (3.0, 0.5, 0.2),
        (3.0, 0.75, 0.0),
        (3.25, 0.5, 0.2),
        (3.25, 0.75, 0.0),
        (3.5, 0.5, 0.2),
        (3.5, 0.75, 0.0),
    )
    cross_specs: tuple[
        tuple[
            int,
            tuple[int, float, float, str],
            tuple[int, int, float, float, float, float, float, str, str],
            tuple[float, float, float],
        ],
        ...,
    ] = (
        (1, drawdown_specs[0], trratio_specs[0], risk_specs[2]),
        (2, drawdown_specs[1], trratio_specs[2], risk_specs[3]),
        (3, drawdown_specs[2], trratio_specs[4], risk_specs[4]),
        (4, drawdown_specs[3], trratio_specs[5], risk_specs[5]),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
        gate_mode: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": str(gate_mode),
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_riskpanic(tr_med: float, tr_delta: float, long_factor: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "risk_entry_cutoff_hour_et": 15,
            "riskpanic_long_risk_mult_factor": float(long_factor),
            "riskpanic_short_risk_mult_factor": 1.0,
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"IBB #{rank:04d} seed={seed_idx} dd={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "island_bridge", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in trratio_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"IBB #{rank:04d} seed={seed_idx} trratio={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "island_bridge", "seed": seed_name, "kind": "trratio_only"},
                }
            )

        for tr_med, tr_delta, long_factor in risk_specs:
            for short_mult in (0.0, 0.0025):
                rank += 1
                filters = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta, long_factor))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = (
                    f"IBB #{rank:04d} seed={seed_idx} panic(tr={tr_med:g},d={tr_delta:g},L={long_factor:g})"
                    f" short={short_mult:g}"
                )
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "island_bridge", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd, trr, rp in cross_specs:
            for short_mult in (0.0, 0.0025):
                rank += 1
                filters = _merge_dict(seed_filters, _over_dd(*dd))
                filters = _merge_dict(filters, _over_trratio(*trr))
                filters = _merge_dict(filters, _over_riskpanic(*rp))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = f"IBB #{rank:04d} seed={seed_idx} cross={cross_idx} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "island_bridge", "seed": seed_name, "kind": "cross"},
                    }
                )

    return groups


def _precision_guard_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()
    filters_base = dict(base_filters or {})
    tod_specs: tuple[tuple[str, dict[str, object]], ...] = (
        ("tod=8-14", {"entry_start_hour_et": 8, "entry_end_hour_et": 14, "risk_entry_cutoff_hour_et": 14}),
        ("tod=8-15", {"entry_start_hour_et": 8, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
        ("tod=9-15", {"entry_start_hour_et": 9, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
    )
    geom_specs: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "geom=base",
            {
                "ema_spread_min_pct": 0.0015,
                "ema_spread_min_pct_down": 0.02,
                "ema_slope_min_pct": 0.01,
            },
        ),
        (
            "geom=tight",
            {
                "ema_spread_min_pct": 0.002,
                "ema_spread_min_pct_down": 0.025,
                "ema_slope_min_pct": 0.015,
            },
        ),
    )

    for confirm_bars in (1, 2):
        for stop_loss in (0.018, 0.02):
            for flip_hold in (0, 1):
                for tod_note, tod_over in tod_specs:
                    for geom_note, geom_over in geom_specs:
                        rank += 1
                        strategy = dict(base_strategy)
                        strategy.update(
                            {
                                "instrument": "spot",
                                "symbol": symbol_u,
                                "signal_bar_size": "10 mins",
                                "signal_use_rth": False,
                                "spot_exec_bar_size": "5 mins",
                                "ema_entry_mode": "trend",
                                "ema_preset": "8/21",
                                "entry_confirm_bars": int(confirm_bars),
                                "spot_stop_loss_pct": float(stop_loss),
                                "spot_close_eod": False,
                                "exit_on_signal_flip": True,
                                "flip_exit_mode": "entry",
                                "flip_exit_only_if_profit": True,
                                "flip_exit_min_hold_bars": int(flip_hold),
                                "regime_mode": "supertrend",
                                "regime_bar_size": "1 day",
                                "supertrend_atr_period": 7,
                                "supertrend_multiplier": 0.4,
                                "supertrend_source": "close",
                                "regime2_mode": "off",
                                "spot_short_risk_mult": 0.0,
                            }
                        )
                        strategy.pop("regime2_bar_size", None)
                        strategy.pop("regime2_supertrend_atr_period", None)
                        strategy.pop("regime2_supertrend_multiplier", None)
                        strategy.pop("regime2_supertrend_source", None)
                        strategy.pop("regime2_ema_preset", None)

                        filters = _merge_dict(filters_base, tod_over)
                        filters = _merge_dict(filters, geom_over)
                        note = (
                            f"PGA #{rank:04d} FULL24 ema=8/21 confirm={confirm_bars} sl={stop_loss:g} "
                            f"hold={flip_hold} {tod_note} {geom_note}"
                        )
                        groups.append(
                            {
                                "name": note,
                                "filters": filters,
                                "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                                "_eval": {"stage": "A", "profile": "precision_guard", "note": note},
                            }
                        )
    return groups


def _precision_guard_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (10, 8.0, 0.05, "both"),
        (15, 8.0, 0.08, "both"),
        (10, 10.0, 0.05, "both"),
        (20, 10.0, 0.08, "both"),
    )
    trratio_specs: tuple[tuple[int, int, float, float, float, float, float, str, str], ...] = (
        (5, 50, 1.45, 1.35, 3.5, 10.0, 0.15, "both", "detect"),
        (5, 50, 1.50, 1.40, 4.0, 12.0, 0.20, "both", "detect"),
        (5, 50, 1.55, 1.45, 4.5, 12.0, 0.20, "both", "detect"),
        (3, 50, 1.45, 1.35, 3.5, 10.0, 0.20, "both", "detect"),
        (5, 34, 1.45, 1.35, 3.5, 10.0, 0.15, "both", "detect"),
    )
    risk_specs: tuple[tuple[float, float, float], ...] = (
        (3.0, 0.5, 0.0),
        (3.0, 0.75, 0.0),
        (3.25, 0.5, 0.0),
        (3.25, 0.75, 0.0),
        (3.5, 0.5, 0.0),
        (3.25, 0.5, 0.1),
    )
    cross_specs: tuple[
        tuple[
            int,
            tuple[int, float, float, str],
            tuple[int, int, float, float, float, float, float, str, str],
            tuple[float, float, float],
        ],
        ...,
    ] = (
        (1, drawdown_specs[0], trratio_specs[0], risk_specs[0]),
        (2, drawdown_specs[1], trratio_specs[1], risk_specs[2]),
        (3, drawdown_specs[2], trratio_specs[2], risk_specs[3]),
        (4, drawdown_specs[3], trratio_specs[3], risk_specs[4]),
        (5, drawdown_specs[0], trratio_specs[4], risk_specs[5]),
        (6, drawdown_specs[2], trratio_specs[1], risk_specs[1]),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
        gate_mode: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": str(gate_mode),
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_riskpanic(tr_med: float, tr_delta: float, long_factor: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "risk_entry_cutoff_hour_et": 15,
            "riskpanic_long_risk_mult_factor": float(long_factor),
            "riskpanic_short_risk_mult_factor": 1.0,
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"PGB #{rank:04d} seed={seed_idx} dd={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "precision_guard", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in trratio_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"PGB #{rank:04d} seed={seed_idx} trratio={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "precision_guard", "seed": seed_name, "kind": "trratio_only"},
                }
            )

        for tr_med, tr_delta, long_factor in risk_specs:
            for short_mult in (0.0, 0.001):
                rank += 1
                filters = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta, long_factor))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = (
                    f"PGB #{rank:04d} seed={seed_idx} panic(tr={tr_med:g},d={tr_delta:g},L={long_factor:g})"
                    f" short={short_mult:g}"
                )
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "precision_guard", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd, trr, rp in cross_specs:
            for short_mult in (0.0, 0.001):
                rank += 1
                filters = _merge_dict(seed_filters, _over_dd(*dd))
                filters = _merge_dict(filters, _over_trratio(*trr))
                filters = _merge_dict(filters, _over_riskpanic(*rp))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = f"PGB #{rank:04d} seed={seed_idx} cross={cross_idx} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "precision_guard", "seed": seed_name, "kind": "cross"},
                    }
                )
    return groups


def _graph_compact_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()
    filters_base = _neutral_filters(base_filters)

    tod_specs: tuple[tuple[str, dict[str, object]], ...] = (
        ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None, "risk_entry_cutoff_hour_et": None}),
        ("tod=8-16", {"entry_start_hour_et": 8, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
    )
    exit_specs: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "flip=state_h6_prof1",
            {
                "flip_exit_mode": "state",
                "flip_exit_only_if_profit": True,
                "flip_exit_min_hold_bars": 6,
            },
        ),
        (
            "flip=cross_h0_prof0",
            {
                "flip_exit_mode": "cross",
                "flip_exit_only_if_profit": False,
                "flip_exit_min_hold_bars": 0,
            },
        ),
        (
            "flip=entry_h0_prof0",
            {
                "flip_exit_mode": "entry",
                "flip_exit_only_if_profit": False,
                "flip_exit_min_hold_bars": 0,
            },
        ),
    )
    stop_specs: tuple[tuple[str, float], ...] = (
        ("sl=0.0192", 0.0192),
        ("sl=0.0160", 0.0160),
    )
    risk_specs: tuple[tuple[str, float], ...] = (
        ("risk=0.019", 0.019),
        ("risk=0.016", 0.016),
    )
    graph_specs: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
        (
            "graph=def_lo",
            {
                "spot_policy_graph": "defensive",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_atr_defensive",
                "spot_risk_overlay_policy": "atr_compress",
                "spot_entry_tr_ratio_min": 0.98,
                "spot_entry_slope_med_abs_min_pct": 0.000002,
                "spot_entry_slope_vel_abs_min_pct": 0.000001,
                "spot_exit_flip_hold_tr_ratio_min": 0.99,
                "spot_resize_adaptive_mode": "atr",
                "spot_resize_adaptive_min_mult": 0.40,
                "spot_resize_adaptive_max_mult": 1.00,
                "spot_graph_overlay_atr_hi_pct": 7.0,
                "spot_graph_overlay_atr_hi_min_mult": 0.35,
            },
            {},
        ),
        (
            "graph=def_mid",
            {
                "spot_policy_graph": "defensive",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_atr_defensive",
                "spot_risk_overlay_policy": "atr_compress",
                "spot_entry_tr_ratio_min": 1.00,
                "spot_entry_slope_med_abs_min_pct": 0.000003,
                "spot_entry_slope_vel_abs_min_pct": 0.0000015,
                "spot_entry_atr_vel_min_pct": 0.03,
                "spot_entry_atr_accel_min_pct": 0.007,
                "spot_exit_flip_hold_tr_ratio_min": 1.00,
                "spot_resize_adaptive_mode": "atr",
                "spot_resize_adaptive_min_mult": 0.35,
                "spot_resize_adaptive_max_mult": 1.00,
                "spot_graph_overlay_atr_hi_pct": 6.5,
                "spot_graph_overlay_atr_hi_min_mult": 0.30,
            },
            {},
        ),
        (
            "graph=hf_probe_lo",
            {
                "spot_policy_graph": "hf_probe",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_slope_probe",
                "spot_risk_overlay_policy": "trend_bias",
                "spot_entry_tr_ratio_min": 0.96,
                "spot_entry_slope_med_abs_min_pct": 0.000002,
                "spot_entry_slope_vel_abs_min_pct": 0.000001,
                "spot_exit_flip_hold_tr_ratio_min": 0.98,
                "spot_resize_adaptive_mode": "slope",
                "spot_resize_adaptive_min_mult": 0.50,
                "spot_resize_adaptive_max_mult": 1.35,
            },
            {},
        ),
        (
            "graph=hf_probe_mid",
            {
                "spot_policy_graph": "hf_probe",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_slope_probe",
                "spot_risk_overlay_policy": "trend_bias",
                "spot_entry_tr_ratio_min": 1.00,
                "spot_entry_slope_med_abs_min_pct": 0.000003,
                "spot_entry_slope_vel_abs_min_pct": 0.0000015,
                "spot_entry_atr_vel_min_pct": 0.03,
                "spot_entry_atr_accel_min_pct": 0.007,
                "spot_exit_flip_hold_tr_ratio_min": 1.00,
                "spot_resize_adaptive_mode": "slope",
                "spot_resize_adaptive_min_mult": 0.45,
                "spot_resize_adaptive_max_mult": 1.30,
            },
            {},
        ),
        (
            "graph=hf_probe_crash",
            {
                "spot_policy_graph": "hf_probe",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_slope_probe",
                "spot_risk_overlay_policy": "trend_bias",
                "spot_entry_tr_ratio_min": 1.00,
                "spot_entry_slope_med_abs_min_pct": 0.000003,
                "spot_entry_slope_vel_abs_min_pct": 0.0000015,
                "spot_entry_atr_vel_min_pct": 0.03,
                "spot_entry_atr_accel_min_pct": 0.007,
                "spot_exit_flip_hold_tr_ratio_min": 1.00,
                "spot_resize_adaptive_mode": "slope",
                "spot_resize_adaptive_min_mult": 0.45,
                "spot_resize_adaptive_max_mult": 1.30,
            },
            {
                "shock_gate_mode": "detect",
                "shock_scale_detector": "daily_drawdown",
                "shock_drawdown_lookback_days": 20,
                "shock_risk_scale_target_atr_pct": 8.0,
                "shock_risk_scale_min_mult": 0.2,
                "shock_risk_scale_apply_to": "both",
                "shock_direction_source": "signal",
                "shock_direction_lookback": 1,
                "riskoff_tr5_med_pct": None,
                "riskpop_tr5_med_pct": None,
                "riskpanic_tr5_med_pct": 2.75,
                "riskpanic_neg_gap_ratio_min": 0.6,
                "riskpanic_neg_gap_abs_pct_min": 0.005,
                "riskpanic_lookback_days": 5,
                "riskpanic_tr5_med_delta_min_pct": 0.5,
                "riskpanic_tr5_med_delta_lookback_days": 1,
                "riskpanic_long_risk_mult_factor": 0.0,
                "riskpanic_short_risk_mult_factor": 1.0,
                "riskpanic_long_scale_mode": "linear",
                "riskpanic_long_scale_tr_delta_max_pct": None,
                "risk_entry_cutoff_hour_et": 15,
            },
        ),
        (
            "graph=aggr_guard",
            {
                "spot_policy_graph": "aggressive",
                "spot_entry_policy": "slope_tr_guard",
                "spot_exit_policy": "slope_flip_guard",
                "spot_resize_policy": "adaptive_hybrid_aggressive",
                "spot_risk_overlay_policy": "trend_bias",
                "spot_entry_tr_ratio_min": 0.98,
                "spot_entry_slope_med_abs_min_pct": 0.000002,
                "spot_entry_slope_vel_abs_min_pct": 0.000001,
                "spot_exit_flip_hold_tr_ratio_min": 0.99,
                "spot_resize_adaptive_mode": "hybrid",
                "spot_resize_adaptive_min_mult": 0.90,
                "spot_resize_adaptive_max_mult": 1.45,
            },
            {},
        ),
    )

    for tod_note, tod_over in tod_specs:
        for exit_note, exit_over in exit_specs:
            for stop_note, stop_loss in stop_specs:
                for risk_note, risk_pct in risk_specs:
                    for graph_note, graph_over, filt_over in graph_specs:
                        rank += 1
                        strategy = dict(base_strategy)
                        strategy.update(
                            {
                                "instrument": "spot",
                                "symbol": symbol_u,
                                "signal_bar_size": "10 mins",
                                "signal_use_rth": False,
                                "spot_exec_bar_size": "1 min",
                                "ema_entry_mode": "trend",
                                "ema_preset": "5/13",
                                "entry_confirm_bars": 0,
                                "spot_stop_loss_pct": float(stop_loss),
                                "spot_profit_target_pct": None,
                                "spot_risk_pct": float(risk_pct),
                                "spot_close_eod": False,
                                "exit_on_signal_flip": True,
                                "regime_mode": "supertrend",
                                "regime_bar_size": "1 day",
                                "supertrend_atr_period": 7,
                                "supertrend_multiplier": 0.4,
                                "supertrend_source": "close",
                                "regime2_mode": "off",
                                "spot_short_risk_mult": 0.04,
                            }
                        )
                        strategy.update(dict(exit_over))
                        strategy.update(dict(graph_over))
                        strategy.pop("regime2_bar_size", None)
                        strategy.pop("regime2_supertrend_atr_period", None)
                        strategy.pop("regime2_supertrend_multiplier", None)
                        strategy.pop("regime2_supertrend_source", None)
                        strategy.pop("regime2_ema_preset", None)

                        filters = _merge_dict(filters_base, tod_over)
                        filters = _merge_dict(filters, dict(filt_over))
                        note = (
                            f"GCA #{rank:04d} {tod_note} {exit_note} {stop_note} {risk_note} "
                            f"{graph_note}"
                        )
                        groups.append(
                            {
                                "name": note,
                                "filters": filters,
                                "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                                "_eval": {"stage": "A", "profile": "graph_compact", "note": note},
                            }
                        )
    return groups


def _graph_compact_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (10, 8.0, 0.2, "both"),
        (20, 8.0, 0.2, "both"),
        (20, 10.0, 0.2, "cap"),
    )
    trratio_specs: tuple[tuple[int, int, float, float, float, float, float, str, str], ...] = (
        (3, 21, 1.30, 1.20, 3.0, 8.0, 0.2, "both", "detect"),
        (5, 34, 1.40, 1.30, 3.5, 10.0, 0.2, "both", "detect"),
        (3, 21, 1.30, 1.20, 3.0, 8.0, 0.2, "both", "surf"),
    )
    panic_specs: tuple[tuple[float, float, float, float], ...] = (
        (2.75, 0.5, 0.2, 1.0),
        (2.75, 0.5, 0.0, 1.5),
        (3.0, 0.5, 0.0, 1.5),
        (2.5, 0.25, 0.0, 2.0),
    )
    short_mults = (0.0, 0.02, 0.04)
    cross_specs: tuple[tuple[int, int, int, int], ...] = (
        (0, 0, 0, 15),
        (1, 1, 1, 15),
        (2, 2, 2, 16),
        (0, 1, 3, 15),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
        gate_mode: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": str(gate_mode),
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_riskpanic(tr_med: float, tr_delta: float, long_factor: float, short_factor: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "riskpanic_long_risk_mult_factor": float(long_factor),
            "riskpanic_short_risk_mult_factor": float(short_factor),
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"GCB #{rank:04d} seed={seed_idx} dd={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "graph_compact", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in trratio_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"GCB #{rank:04d} seed={seed_idx} trratio={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "graph_compact", "seed": seed_name, "kind": "trratio_only"},
                }
            )

        for tr_med, tr_delta, long_factor, short_factor in panic_specs:
            for short_mult in short_mults:
                rank += 1
                filters = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta, long_factor, short_factor))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = (
                    f"GCB #{rank:04d} seed={seed_idx} panic(tr={tr_med:g},d={tr_delta:g},"
                    f"L={long_factor:g},S={short_factor:g}) short={short_mult:g}"
                )
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "graph_compact", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd_idx, tr_idx, panic_idx, cutoff in (
            (idx + 1, *spec) for idx, spec in enumerate(cross_specs)
        ):
            dd = drawdown_specs[int(dd_idx)]
            trr = trratio_specs[int(tr_idx)]
            panic = panic_specs[int(panic_idx)]
            for short_mult in short_mults:
                rank += 1
                filters = _merge_dict(seed_filters, _over_dd(*dd))
                filters = _merge_dict(filters, _over_trratio(*trr))
                filters = _merge_dict(filters, _over_riskpanic(*panic))
                filters = _merge_dict(filters, {"risk_entry_cutoff_hour_et": int(cutoff)})
                strategy = dict(seed_strategy)
                strategy.update(
                    {
                        "spot_short_risk_mult": float(short_mult),
                        "flip_exit_mode": "cross",
                        "flip_exit_only_if_profit": False,
                        "flip_exit_min_hold_bars": 0,
                        "spot_stop_loss_pct": 0.016,
                        "spot_risk_pct": 0.016,
                    }
                )
                note = f"GCB #{rank:04d} seed={seed_idx} cross={cross_idx} cutoff={cutoff} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "graph_compact", "seed": seed_name, "kind": "cross"},
                    }
                )
    return groups


def _hour_expansion_stage_a(*, symbol: str, base_strategy: dict, base_filters: dict) -> list[dict]:
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()
    filters_base = dict(base_filters or {})
    tod_specs: tuple[tuple[str, dict[str, object]], ...] = (
        ("tod=6-14", {"entry_start_hour_et": 6, "entry_end_hour_et": 14, "risk_entry_cutoff_hour_et": 14}),
        ("tod=6-15", {"entry_start_hour_et": 6, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
        ("tod=6-16", {"entry_start_hour_et": 6, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
        ("tod=7-14", {"entry_start_hour_et": 7, "entry_end_hour_et": 14, "risk_entry_cutoff_hour_et": 14}),
        ("tod=7-15", {"entry_start_hour_et": 7, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
        ("tod=7-16", {"entry_start_hour_et": 7, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
        ("tod=8-15", {"entry_start_hour_et": 8, "entry_end_hour_et": 15, "risk_entry_cutoff_hour_et": 15}),
        ("tod=8-16", {"entry_start_hour_et": 8, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
        ("tod=9-16", {"entry_start_hour_et": 9, "entry_end_hour_et": 16, "risk_entry_cutoff_hour_et": 16}),
    )
    geom_specs: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "geom=base",
            {
                "ema_spread_min_pct": 0.0015,
                "ema_spread_min_pct_down": 0.02,
                "ema_slope_min_pct": 0.01,
            },
        ),
        (
            "geom=tight",
            {
                "ema_spread_min_pct": 0.0025,
                "ema_spread_min_pct_down": 0.03,
                "ema_slope_min_pct": 0.02,
            },
        ),
    )

    for ema_preset in ("8/21", "5/13"):
        for stop_loss in (0.02, 0.022):
            for short_mult in (0.0, 0.0025):
                for tod_note, tod_over in tod_specs:
                    for geom_note, geom_over in geom_specs:
                        rank += 1
                        strategy = dict(base_strategy)
                        strategy.update(
                            {
                                "instrument": "spot",
                                "symbol": symbol_u,
                                "signal_bar_size": "10 mins",
                                "signal_use_rth": False,
                                "spot_exec_bar_size": "5 mins",
                                "ema_entry_mode": "trend",
                                "ema_preset": str(ema_preset),
                                "entry_confirm_bars": 1,
                                "spot_stop_loss_pct": float(stop_loss),
                                "spot_close_eod": False,
                                "exit_on_signal_flip": True,
                                "flip_exit_mode": "entry",
                                "flip_exit_only_if_profit": True,
                                "flip_exit_min_hold_bars": 0,
                                "regime_mode": "supertrend",
                                "regime_bar_size": "1 day",
                                "supertrend_atr_period": 7,
                                "supertrend_multiplier": 0.4,
                                "supertrend_source": "close",
                                "regime2_mode": "off",
                                "spot_short_risk_mult": float(short_mult),
                            }
                        )
                        strategy.pop("regime2_bar_size", None)
                        strategy.pop("regime2_supertrend_atr_period", None)
                        strategy.pop("regime2_supertrend_multiplier", None)
                        strategy.pop("regime2_supertrend_source", None)
                        strategy.pop("regime2_ema_preset", None)

                        filters = _merge_dict(filters_base, tod_over)
                        filters = _merge_dict(filters, geom_over)
                        note = (
                            f"HEA #{rank:04d} FULL24 ema={ema_preset} sl={stop_loss:g} "
                            f"short={short_mult:g} {tod_note} {geom_note}"
                        )
                        groups.append(
                            {
                                "name": note,
                                "filters": filters,
                                "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                                "_eval": {"stage": "A", "profile": "hour_expansion", "note": note},
                            }
                        )
    return groups


def _hour_expansion_stage_b(*, symbol: str, seed_groups: list[dict]) -> list[dict]:
    seeds = [_seed_from_group(g) for g in seed_groups]
    groups: list[dict] = []
    rank = 0
    symbol_u = str(symbol).strip().upper()

    drawdown_specs: tuple[tuple[int, float, float, str], ...] = (
        (10, 8.0, 0.05, "both"),
        (20, 8.0, 0.08, "both"),
        (30, 10.0, 0.08, "both"),
        (20, 12.0, 0.10, "both"),
        (10, 10.0, 0.10, "cap"),
        (20, 10.0, 0.15, "cap"),
    )
    tr_detect_specs: tuple[tuple[int, int, float, float, float, float, float, str, str], ...] = (
        (5, 50, 1.45, 1.35, 4.0, 10.0, 0.15, "both", "detect"),
        (5, 50, 1.50, 1.40, 4.0, 12.0, 0.20, "both", "detect"),
        (5, 50, 1.55, 1.45, 5.0, 12.0, 0.25, "both", "detect"),
        (3, 50, 1.45, 1.35, 4.0, 10.0, 0.20, "both", "detect"),
        (3, 34, 1.45, 1.35, 3.5, 10.0, 0.20, "cap", "detect"),
        (5, 34, 1.50, 1.40, 4.0, 12.0, 0.20, "cap", "detect"),
        (5, 21, 1.45, 1.35, 3.5, 10.0, 0.20, "cap", "detect"),
        (3, 21, 1.40, 1.30, 3.0, 8.0, 0.15, "cap", "detect"),
    )
    tr_surf_specs: tuple[tuple[int, int, float, float, float, float, float, str, str], ...] = (
        (3, 21, 1.30, 1.20, 3.0, 8.0, 0.20, "both", "surf"),
        (3, 21, 1.35, 1.25, 3.5, 10.0, 0.20, "both", "surf"),
        (3, 34, 1.35, 1.25, 3.5, 10.0, 0.20, "cap", "surf"),
        (5, 34, 1.40, 1.30, 4.0, 12.0, 0.25, "cap", "surf"),
        (3, 50, 1.35, 1.25, 3.5, 10.0, 0.20, "both", "surf"),
        (5, 50, 1.40, 1.30, 4.0, 12.0, 0.25, "cap", "surf"),
    )
    risk_specs: tuple[tuple[float, float, float], ...] = (
        (2.75, 0.5, 0.2),
        (2.75, 0.75, 0.0),
        (3.0, 0.5, 0.2),
        (3.0, 0.75, 0.0),
        (3.25, 0.5, 0.2),
        (3.25, 0.75, 0.0),
        (3.5, 0.5, 0.2),
        (3.5, 0.75, 0.0),
    )
    cross_specs: tuple[
        tuple[
            int,
            tuple[int, float, float, str],
            tuple[int, int, float, float, float, float, float, str, str],
            tuple[float, float, float],
            int,
        ],
        ...,
    ] = (
        (1, drawdown_specs[0], tr_detect_specs[0], risk_specs[0], 14),
        (2, drawdown_specs[1], tr_detect_specs[1], risk_specs[3], 15),
        (3, drawdown_specs[2], tr_detect_specs[2], risk_specs[5], 16),
        (4, drawdown_specs[4], tr_surf_specs[1], risk_specs[2], 14),
        (5, drawdown_specs[5], tr_surf_specs[3], risk_specs[4], 15),
        (6, drawdown_specs[3], tr_detect_specs[5], risk_specs[7], 16),
        (7, drawdown_specs[1], tr_surf_specs[4], risk_specs[6], 15),
        (8, drawdown_specs[0], tr_detect_specs[7], risk_specs[1], 14),
    )

    def _over_dd(lb: int, target: float, min_mult: float, apply_to: str) -> dict[str, object]:
        return {
            "shock_gate_mode": "detect",
            "shock_scale_detector": "daily_drawdown",
            "shock_drawdown_lookback_days": int(lb),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_trratio(
        fast: int,
        slow: int,
        on_ratio: float,
        off_ratio: float,
        min_atr_pct: float,
        target: float,
        min_mult: float,
        apply_to: str,
        gate_mode: str,
    ) -> dict[str, object]:
        return {
            "shock_gate_mode": str(gate_mode),
            "shock_scale_detector": "tr_ratio",
            "shock_atr_fast_period": int(fast),
            "shock_atr_slow_period": int(slow),
            "shock_on_ratio": float(on_ratio),
            "shock_off_ratio": float(off_ratio),
            "shock_min_atr_pct": float(min_atr_pct),
            "shock_risk_scale_target_atr_pct": float(target),
            "shock_risk_scale_min_mult": float(min_mult),
            "shock_risk_scale_apply_to": str(apply_to),
            "shock_direction_source": "signal",
            "shock_direction_lookback": 1,
        }

    def _over_riskpanic(tr_med: float, tr_delta: float, long_factor: float) -> dict[str, object]:
        return {
            "riskoff_tr5_med_pct": None,
            "riskpop_tr5_med_pct": None,
            "riskpanic_tr5_med_pct": float(tr_med),
            "riskpanic_neg_gap_ratio_min": 0.6,
            "riskpanic_neg_gap_abs_pct_min": 0.005,
            "riskpanic_lookback_days": 5,
            "riskpanic_tr5_med_delta_min_pct": float(tr_delta),
            "riskpanic_tr5_med_delta_lookback_days": 1,
            "riskpanic_long_risk_mult_factor": float(long_factor),
            "riskpanic_short_risk_mult_factor": 1.0,
            "riskpanic_long_scale_mode": "linear",
            "riskpanic_long_scale_tr_delta_max_pct": None,
        }

    for seed_idx, (seed_name, seed_strategy, seed_filters) in enumerate(seeds, start=1):
        for dd in drawdown_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_dd(*dd))
            note = f"HEB #{rank:04d} seed={seed_idx} angle=dd cfg={dd}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "hour_expansion", "seed": seed_name, "kind": "drawdown_only"},
                }
            )

        for trr in tr_detect_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"HEB #{rank:04d} seed={seed_idx} angle=detect cfg={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "hour_expansion", "seed": seed_name, "kind": "tr_detect"},
                }
            )

        for trr in tr_surf_specs:
            rank += 1
            filters = _merge_dict(seed_filters, _over_trratio(*trr))
            note = f"HEB #{rank:04d} seed={seed_idx} angle=surf cfg={trr}"
            groups.append(
                {
                    "name": note,
                    "filters": filters,
                    "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": dict(seed_strategy)}],
                    "_eval": {"stage": "B", "profile": "hour_expansion", "seed": seed_name, "kind": "tr_surf"},
                }
            )

        for tr_med, tr_delta, long_factor in risk_specs:
            for short_mult in (0.0, 0.0025):
                rank += 1
                filters = _merge_dict(seed_filters, _over_riskpanic(tr_med, tr_delta, long_factor))
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = (
                    f"HEB #{rank:04d} seed={seed_idx} angle=risk panic(tr={tr_med:g},d={tr_delta:g},L={long_factor:g})"
                    f" short={short_mult:g}"
                )
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "hour_expansion", "seed": seed_name, "kind": "riskpanic_only"},
                    }
                )

        for cross_idx, dd, trr, rp, cutoff in cross_specs:
            for short_mult in (0.0, 0.0025):
                rank += 1
                filters = _merge_dict(seed_filters, _over_dd(*dd))
                filters = _merge_dict(filters, _over_trratio(*trr))
                filters = _merge_dict(filters, _over_riskpanic(*rp))
                filters = _merge_dict(filters, {"risk_entry_cutoff_hour_et": int(cutoff)})
                strategy = dict(seed_strategy)
                strategy["spot_short_risk_mult"] = float(short_mult)
                note = f"HEB #{rank:04d} seed={seed_idx} angle=cross idx={cross_idx} cutoff={cutoff} short={short_mult:g}"
                groups.append(
                    {
                        "name": note,
                        "filters": filters,
                        "entries": [{"symbol": symbol_u, "metrics": _blank_metrics(), "strategy": strategy}],
                        "_eval": {"stage": "B", "profile": "hour_expansion", "seed": seed_name, "kind": "cross"},
                    }
                )

    return groups


def _windows_from_strings(raw_windows: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for raw in raw_windows:
        s, e = _parse_window(raw)
        out.append({"start": s, "end": e})
    return out


def _load_seed_base(
    seed_path: Path,
    *,
    symbol: str,
    bar_size: str,
    kingmaker_id: int | None,
) -> tuple[dict, dict]:
    payload = _read_json(seed_path)
    groups = list(_iter_groups(payload, symbol=symbol, bar_size=bar_size))
    if not groups:
        raise SystemExit(f"No seed groups for {symbol} {bar_size} in {seed_path}")
    selected = _pick_group(groups, kingmaker_id=kingmaker_id, label=f"seed payload {seed_path}")
    seed_name, seed_strategy, seed_filters = _seed_from_group(selected)
    print(f"seed_base={seed_name}")
    return seed_strategy, seed_filters


def _window_key(item: dict) -> tuple[str, str]:
    return (str(item.get("start") or ""), str(item.get("end") or ""))


def _load_champion_window_pnls(
    path: Path,
    *,
    symbol: str,
    bar_size: str,
    kingmaker_id: int | None,
) -> dict[tuple[str, str], float]:
    payload = _read_json(path)
    groups = list(_iter_groups(payload, symbol=symbol, bar_size=bar_size))
    if not groups:
        raise SystemExit(f"No champion groups for {symbol} {bar_size} in {path}")
    g0 = _pick_group(groups, kingmaker_id=kingmaker_id, label=f"champion payload {path}")
    eval_obj = g0.get("_eval") if isinstance(g0, dict) else None
    windows = (eval_obj or {}).get("windows") if isinstance(eval_obj, dict) else None
    out: dict[tuple[str, str], float] = {}
    if isinstance(windows, list) and windows:
        for w in windows:
            if not isinstance(w, dict):
                continue
            out[_window_key(w)] = float(w.get("pnl") or 0.0)
    if out:
        return out
    m = _group_metrics(g0)
    src_windows = payload.get("windows") or []
    if isinstance(src_windows, list) and src_windows:
        wk = _window_key(src_windows[0])
        out[wk] = float(m.get("pnl") or 0.0)
    return out


def _find_champion_pnl_beaters(
    *,
    promotion_payload_path: Path,
    champion_by_window: dict[tuple[str, str], float],
) -> list[tuple[float, dict]]:
    payload = _read_json(promotion_payload_path)
    out: list[tuple[float, dict]] = []
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        return out
    for g in groups:
        if not isinstance(g, dict):
            continue
        eval_obj = g.get("_eval")
        windows = (eval_obj or {}).get("windows") if isinstance(eval_obj, dict) else None
        if not isinstance(windows, list) or not windows:
            continue
        beat_all = True
        margin_sum = 0.0
        for w in windows:
            if not isinstance(w, dict):
                beat_all = False
                break
            wk = _window_key(w)
            champ_pnl = champion_by_window.get(wk)
            if champ_pnl is None:
                beat_all = False
                break
            pnl = float(w.get("pnl") or 0.0)
            if pnl <= float(champ_pnl):
                beat_all = False
                break
            margin_sum += float(pnl) - float(champ_pnl)
        if beat_all:
            out.append((margin_sum, g))
    out.sort(key=lambda x: float(x[0]), reverse=True)
    return out


def _profile_run(
    *,
    profile: str,
    repo_root: Path,
    out_dir: Path,
    run_id: str,
    symbol: str,
    seed_strategy: dict,
    seed_filters: dict,
    cache_dir: Path,
    jobs: int,
    offline: bool,
    stage_window: str,
    promotion_windows: list[str],
    stage_min_trades: int,
    stage_min_tpy: float,
    promotion_min_trades: int,
    promotion_min_tpy: float,
    champion_by_window: dict[tuple[str, str], float],
    lane_mode: str,
) -> None:
    stage_windows_payload = _windows_from_strings([stage_window])
    promo_windows_payload = _windows_from_strings(promotion_windows)

    if profile == "hyper10":
        stage_a_groups = _hyper10_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 8
        stage_b_top_keep = 80
        stage_b_builder = _hyper10_stage_b
    elif profile == "scalp10":
        stage_a_groups = _scalp10_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 10
        stage_b_top_keep = 80
        stage_b_builder = _scalp10_stage_b
    elif profile == "island_bridge":
        stage_a_groups = _island_bridge_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 12
        stage_b_top_keep = 120
        stage_b_builder = _island_bridge_stage_b
    elif profile == "precision_guard":
        stage_a_groups = _precision_guard_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 10
        stage_b_top_keep = 160
        stage_b_builder = _precision_guard_stage_b
    elif profile == "hour_expansion":
        stage_a_groups = _hour_expansion_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 16
        stage_b_top_keep = 160
        stage_b_builder = _hour_expansion_stage_b
    elif profile == "graph_compact":
        stage_a_groups = _graph_compact_stage_a(symbol=symbol, base_strategy=seed_strategy, base_filters=seed_filters)
        stage_a_seed_top = 8
        stage_b_top_keep = 120
        stage_b_builder = _graph_compact_stage_b
    else:
        raise SystemExit(f"Unknown profile: {profile}")

    profile_tag = f"{run_id}_{profile}"
    stage_a_candidates = out_dir / f"{profile_tag}_stageA_candidates.json"
    stage_a_ranked = out_dir / f"{profile_tag}_stageA_ranked.json"
    stage_b_candidates = out_dir / f"{profile_tag}_stageB_candidates.json"
    stage_b_ranked = out_dir / f"{profile_tag}_stageB_ranked_top{stage_b_top_keep}.json"
    promotion_ranked = out_dir / f"{profile_tag}_promotion_top80.json"

    _write_json(
        stage_a_candidates,
        _payload_from_groups(
            name=f"{profile_tag}_stageA_candidates",
            source="seed_base",
            windows=stage_windows_payload,
            groups=stage_a_groups,
            notes=[f"profile={profile}", "stage=A"],
        ),
    )
    print(f"[{profile}] stageA candidates={len(stage_a_groups)}")
    res_a = _evaluate_payload_by_lane(
        repo_root=repo_root,
        in_payload_path=stage_a_candidates,
        out_payload_path=stage_a_ranked,
        run_tag=f"{profile_tag}_stageA_ranked",
        symbol=symbol,
        bar_size="10 mins",
        cache_dir=cache_dir,
        jobs=jobs,
        offline=offline,
        windows=[stage_window],
        min_trades=stage_min_trades,
        min_trades_per_year=stage_min_tpy,
        require_positive_pnl=False,
        lane_mode=lane_mode,
        top_keep=None,
    )
    print(f"[{profile}] stageA tested={res_a.get('tested')} kept={res_a.get('kept')} ranked={res_a.get('output_kept')}")
    stage_a_payload = _read_json(stage_a_ranked)
    stage_a_groups_ranked = stage_a_payload.get("groups") if isinstance(stage_a_payload, dict) else None
    if not isinstance(stage_a_groups_ranked, list) or not stage_a_groups_ranked:
        print(f"[{profile}] stageA empty; skip profile")
        return

    seeds = stage_a_groups_ranked[:stage_a_seed_top]
    stage_b_groups = stage_b_builder(symbol=symbol, seed_groups=seeds)
    _write_json(
        stage_b_candidates,
        _payload_from_groups(
            name=f"{profile_tag}_stageB_candidates",
            source=str(stage_a_ranked),
            windows=stage_windows_payload,
            groups=stage_b_groups,
            notes=[f"profile={profile}", "stage=B", f"seeds={len(seeds)}"],
        ),
    )
    print(f"[{profile}] stageB candidates={len(stage_b_groups)}")
    res_b = _evaluate_payload_by_lane(
        repo_root=repo_root,
        in_payload_path=stage_b_candidates,
        out_payload_path=stage_b_ranked,
        run_tag=f"{profile_tag}_stageB_ranked",
        symbol=symbol,
        bar_size="10 mins",
        cache_dir=cache_dir,
        jobs=jobs,
        offline=offline,
        windows=[stage_window],
        min_trades=stage_min_trades,
        min_trades_per_year=stage_min_tpy,
        require_positive_pnl=False,
        lane_mode=lane_mode,
        top_keep=stage_b_top_keep,
    )
    print(f"[{profile}] stageB tested={res_b.get('tested')} kept={res_b.get('kept')} ranked={res_b.get('output_kept')}")
    stage_b_payload = _read_json(stage_b_ranked)
    stage_b_groups_ranked = stage_b_payload.get("groups") if isinstance(stage_b_payload, dict) else None
    if not isinstance(stage_b_groups_ranked, list) or not stage_b_groups_ranked:
        print(f"[{profile}] stageB empty; skip promotion")
        return

    _write_json(
        stage_b_ranked,
        _payload_from_groups(
            name=f"{profile_tag}_stageB_ranked_top{stage_b_top_keep}",
            source=str(stage_b_candidates),
            windows=stage_windows_payload,
            groups=stage_b_groups_ranked,
            notes=[f"profile={profile}", "rank=pnl_first", f"top_keep={stage_b_top_keep}"],
        ),
    )

    res_p = _evaluate_payload_by_lane(
        repo_root=repo_root,
        in_payload_path=stage_b_ranked,
        out_payload_path=promotion_ranked,
        run_tag=f"{profile_tag}_promotion",
        symbol=symbol,
        bar_size="10 mins",
        cache_dir=cache_dir,
        jobs=jobs,
        offline=offline,
        windows=promotion_windows,
        min_trades=promotion_min_trades,
        min_trades_per_year=promotion_min_tpy,
        require_positive_pnl=True,
        lane_mode=lane_mode,
        top_keep=80,
    )
    print(f"[{profile}] promotion tested={res_p.get('tested')} kept={res_p.get('kept')} ranked={res_p.get('output_kept')}")

    beaters = _find_champion_pnl_beaters(
        promotion_payload_path=promotion_ranked,
        champion_by_window=champion_by_window,
    )
    print(f"[{profile}] beat_champion_all_windows={len(beaters)}")
    if beaters:
        top_margin, top_group = beaters[0]
        tm = _group_metrics(top_group)
        print(
            f"[{profile}] best_beater margin_sum={top_margin:.1f} "
            f"pnl={tm.get('pnl'):.1f} tr={tm.get('trades')} name={top_group.get('name')}"
        )

    print(f"[{profile}] artifacts:")
    print(f"  - {stage_a_candidates}")
    print(f"  - {stage_a_ranked}")
    print(f"  - {stage_b_candidates}")
    print(f"  - {stage_b_ranked}")
    print(f"  - {promotion_ranked}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Temporary SLV 10m high-frequency attack runner "
            "(hyper10 + scalp10 + island_bridge + precision_guard + hour_expansion + graph_compact)."
        )
    )
    ap.add_argument("--symbol", default="SLV")
    ap.add_argument("--seed-milestones", default="backtests/slv/slv_exec5m_v30_seed_v25_as_10m_rth_top80.json")
    ap.add_argument("--seed-bar-size", default="10 mins")
    ap.add_argument("--seed-kingmaker-id", type=int, default=None)
    ap.add_argument(
        "--champion-milestones",
        default="backtests/slv/slv_exec5m_v25_shock_throttle_drawdown_1h_10y2y1y_mintr500_top80_20260206_173719.json",
    )
    ap.add_argument("--champion-bar-size", default="1 hour")
    ap.add_argument("--champion-kingmaker-id", type=int, default=None)
    ap.add_argument("--out-dir", default="backtests/slv")
    ap.add_argument("--cache-dir", default="db")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--profiles",
        default="hyper10,scalp10",
        help="Comma list: hyper10, scalp10, island_bridge, precision_guard, hour_expansion, graph_compact",
    )
    ap.add_argument(
        "--lane-mode",
        default="all",
        choices=("all", "rth", "full24"),
        help="Execution lane filter. Use full24 to disable all RTH evaluations.",
    )

    ap.add_argument("--stage-window", default="2024-01-08:2026-01-08")
    ap.add_argument("--stage-min-trades", type=int, default=0)
    ap.add_argument("--stage-min-trades-per-year", type=float, default=600.0)

    ap.add_argument(
        "--promotion-window",
        action="append",
        default=None,
        help="Repeatable promotion windows YYYY-MM-DD:YYYY-MM-DD",
    )
    ap.add_argument("--promotion-min-trades", type=int, default=0)
    ap.add_argument("--promotion-min-trades-per-year", type=float, default=1000.0)
    args = ap.parse_args()

    symbol = str(args.symbol).strip().upper()
    _parse_window(str(args.stage_window))
    promotion_windows = list(args.promotion_window or ["2016-01-08:2026-01-08", "2024-01-08:2026-01-08", "2025-01-08:2026-01-08"])
    for w in promotion_windows:
        _parse_window(str(w))

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    run_id = str(args.run_id or f"slv_exec5m_hfattack_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    seed_path = (repo_root / str(args.seed_milestones)).resolve()
    champ_path = (repo_root / str(args.champion_milestones)).resolve()
    if not seed_path.exists():
        raise SystemExit(f"Missing seed milestones: {seed_path}")
    if not champ_path.exists():
        raise SystemExit(f"Missing champion milestones: {champ_path}")

    seed_strategy, seed_filters = _load_seed_base(
        seed_path,
        symbol=symbol,
        bar_size=str(args.seed_bar_size),
        kingmaker_id=args.seed_kingmaker_id,
    )
    champion_by_window = _load_champion_window_pnls(
        champ_path,
        symbol=symbol,
        bar_size=str(args.champion_bar_size),
        kingmaker_id=args.champion_kingmaker_id,
    )
    print("champion_pnl_by_window:")
    for (s, e), pnl in champion_by_window.items():
        print(f"  {s}:{e} pnl={pnl:.1f}")

    profiles_req = [p.strip().lower() for p in str(args.profiles).split(",") if p.strip()]
    profiles = [
        p
        for p in profiles_req
        if p in ("hyper10", "scalp10", "island_bridge", "precision_guard", "hour_expansion", "graph_compact")
    ]
    if not profiles:
        raise SystemExit(
            "No valid profiles selected. Use --profiles "
            "hyper10,scalp10,island_bridge,precision_guard,hour_expansion,graph_compact"
        )

    for profile in profiles:
        print("")
        print(f"=== profile={profile} run_id={run_id} lane_mode={args.lane_mode} ===")
        _profile_run(
            profile=profile,
            repo_root=repo_root,
            out_dir=out_dir,
            run_id=run_id,
            symbol=symbol,
            seed_strategy=seed_strategy,
            seed_filters=seed_filters,
            cache_dir=cache_dir,
            jobs=int(args.jobs),
            offline=bool(args.offline),
            stage_window=str(args.stage_window),
            promotion_windows=[str(w) for w in promotion_windows],
            stage_min_trades=int(args.stage_min_trades),
            stage_min_tpy=float(args.stage_min_trades_per_year),
            promotion_min_trades=int(args.promotion_min_trades),
            promotion_min_tpy=float(args.promotion_min_trades_per_year),
            champion_by_window=champion_by_window,
            lane_mode=str(args.lane_mode),
        )


if __name__ == "__main__":
    main()
