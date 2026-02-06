#!/usr/bin/env python3
"""Standalone SLV precision audit for champion vs dethroners.

Runs canonical spot backtests (full trade capture) for selected candidates and
computes entry-timing precision on 10m candles via local-extrema distance.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

# Allow running as `python backtests/slv/slv_precision_audit.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradebot.backtest.data import ContractMeta, IBKRHistoricalData
from tradebot.backtest.engine import _run_spot_backtest, _spot_multiplier
from tradebot.backtest.models import Bar, SpotTrade
from tradebot.backtest.run_backtests_spot_sweeps import (
    _filters_from_payload,
    _load_bars,
    _mk_bundle,
    _strategy_from_payload,
)
from tradebot.backtest.sweeps import write_json


RE_KINGMAKER = re.compile(r"KINGMAKER\s*#(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    source_path: Path
    kingmaker_id: int
    group_name: str
    strategy: dict[str, Any]
    filters: dict[str, Any] | None


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _extract_kingmaker_id(name: str) -> int | None:
    m = RE_KINGMAKER.search(str(name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected object JSON at {path}, got {type(obj).__name__}.")
    return obj


def _pick_group(payload: dict[str, Any], *, kingmaker_id: int) -> dict[str, Any]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise SystemExit("Invalid payload: missing `groups` list.")
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = _extract_kingmaker_id(str(group.get("name") or ""))
        if gid == int(kingmaker_id):
            return group
    raise SystemExit(f"Could not find KINGMAKER #{kingmaker_id:02d} in payload.")


def _candidate_from_group(*, source_path: Path, label: str, group: dict[str, Any], kingmaker_id: int) -> CandidateSpec:
    entries = group.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"Invalid group for {label}: missing entries.")
    e0 = entries[0]
    if not isinstance(e0, dict):
        raise SystemExit(f"Invalid group for {label}: first entry is not an object.")
    strategy = e0.get("strategy")
    if not isinstance(strategy, dict):
        raise SystemExit(f"Invalid group for {label}: missing strategy object.")
    filters_raw = group.get("filters")
    filters = dict(filters_raw) if isinstance(filters_raw, dict) else None
    return CandidateSpec(
        label=str(label),
        source_path=Path(source_path),
        kingmaker_id=int(kingmaker_id),
        group_name=str(group.get("name") or ""),
        strategy=dict(strategy),
        filters=filters,
    )


def _resolve_meta(
    *,
    symbol: str,
    exchange: str | None,
    offline: bool,
    data: IBKRHistoricalData | None,
) -> ContractMeta:
    is_future = str(symbol).upper() in ("MNQ", "MBT")
    if offline or data is None:
        exch = "CME" if is_future else "SMART"
        return ContractMeta(
            symbol=str(symbol),
            exchange=str(exch),
            multiplier=_spot_multiplier(str(symbol), is_future),
            min_tick=0.01,
        )
    _, resolved = data.resolve_contract(str(symbol), exchange)
    return ContractMeta(
        symbol=str(resolved.symbol),
        exchange=str(resolved.exchange),
        multiplier=_spot_multiplier(str(symbol), is_future, default=resolved.multiplier),
        min_tick=float(resolved.min_tick),
    )


def _load_context_bars(
    *,
    bundle,
    data: IBKRHistoricalData,
    cache_dir: Path,
    start_dt: datetime,
    end_dt: datetime,
    offline: bool,
) -> tuple[list[Bar] | None, list[Bar] | None, list[Bar] | None, list[Bar] | None]:
    base_bar = str(bundle.backtest.bar_size)
    regime_bars: list[Bar] | None = None
    regime2_bars: list[Bar] | None = None
    tick_bars: list[Bar] | None = None
    exec_bars: list[Bar] | None = None

    regime_mode = str(getattr(bundle.strategy, "regime_mode", "") or "").strip().lower()
    regime_bar = str(getattr(bundle.strategy, "regime_bar_size", "") or "").strip()
    needs_regime = False
    if regime_bar and regime_bar != base_bar:
        if regime_mode == "supertrend":
            needs_regime = True
        elif bool(getattr(bundle.strategy, "regime_ema_preset", None)):
            needs_regime = True
    if needs_regime:
        regime_bars = _load_bars(
            data,
            symbol=bundle.strategy.symbol,
            exchange=bundle.strategy.exchange,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=regime_bar,
            use_rth=bundle.backtest.use_rth,
            cache_dir=cache_dir,
            offline=offline,
        )
        if not regime_bars:
            raise SystemExit(f"Missing regime bars: {bundle.strategy.symbol} {regime_bar}")

    regime2_mode = str(getattr(bundle.strategy, "regime2_mode", "off") or "off").strip().lower()
    regime2_bar = str(getattr(bundle.strategy, "regime2_bar_size", "") or "").strip() or base_bar
    if regime2_mode != "off" and regime2_bar != base_bar:
        regime2_bars = _load_bars(
            data,
            symbol=bundle.strategy.symbol,
            exchange=bundle.strategy.exchange,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=regime2_bar,
            use_rth=bundle.backtest.use_rth,
            cache_dir=cache_dir,
            offline=offline,
        )
        if not regime2_bars:
            raise SystemExit(f"Missing regime2 bars: {bundle.strategy.symbol} {regime2_bar}")

    tick_mode = str(getattr(bundle.strategy, "tick_gate_mode", "off") or "off").strip().lower()
    if tick_mode != "off":
        z_lookback = int(getattr(bundle.strategy, "tick_width_z_lookback", 252) or 252)
        ma_period = int(getattr(bundle.strategy, "tick_band_ma_period", 10) or 10)
        slope_lb = int(getattr(bundle.strategy, "tick_width_slope_lookback", 3) or 3)
        tick_warm_days = max(60, z_lookback + ma_period + slope_lb + 5)
        tick_start_dt = start_dt - timedelta(days=tick_warm_days)
        tick_symbol = str(getattr(bundle.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
        tick_exchange = str(getattr(bundle.strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip()
        tick_bars = _load_bars(
            data,
            symbol=tick_symbol,
            exchange=tick_exchange,
            start_dt=tick_start_dt,
            end_dt=end_dt,
            bar_size="1 day",
            use_rth=True,
            cache_dir=cache_dir,
            offline=offline,
        )
        if not tick_bars:
            raise SystemExit(f"Missing tick-gate bars: {tick_symbol} 1 day")

    exec_size = str(getattr(bundle.strategy, "spot_exec_bar_size", "") or "").strip()
    if exec_size and exec_size != base_bar:
        exec_bars = _load_bars(
            data,
            symbol=bundle.strategy.symbol,
            exchange=bundle.strategy.exchange,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=exec_size,
            use_rth=bundle.backtest.use_rth,
            cache_dir=cache_dir,
            offline=offline,
        )
        if not exec_bars:
            raise SystemExit(f"Missing exec bars: {bundle.strategy.symbol} {exec_size}")

    return regime_bars, regime2_bars, tick_bars, exec_bars


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    qf = min(1.0, max(0.0, float(q)))
    idx = int(round((len(sorted_vals) - 1) * qf))
    return float(sorted_vals[idx])


def _precision_stats(
    *,
    trades: list[SpotTrade],
    bars_10m: list[Bar],
    radius: int,
    forward_bars: int,
    multiplier: float,
) -> dict[str, Any]:
    if not bars_10m:
        return {
            "samples": 0,
            "avg_dist_local_extrema_pct": 0.0,
            "med_dist_local_extrema_pct": 0.0,
            "p90_dist_local_extrema_pct": 0.0,
            "within_0p25_pct": 0.0,
            "within_0p50_pct": 0.0,
            "within_1p00_pct": 0.0,
            "avg_bar_edge_pos": 0.0,
            "med_bar_edge_pos": 0.0,
            "avg_mfe_pct_fwd": 0.0,
            "avg_mae_pct_fwd": 0.0,
            "avg_mfe_over_mae_fwd": 0.0,
            "avg_pnl_per_trade": 0.0,
            "long_samples": 0,
            "short_samples": 0,
        }

    ts_vec = [b.ts for b in bars_10m]
    dists: list[float] = []
    edge_pos: list[float] = []
    mfe_vals: list[float] = []
    mae_vals: list[float] = []
    pnl_vals: list[float] = []
    long_n = 0
    short_n = 0
    eps = 1e-12

    for tr in trades:
        if tr.qty == 0:
            continue
        idx = bisect.bisect_right(ts_vec, tr.entry_time) - 1
        if idx < 0 or idx >= len(bars_10m):
            continue

        lo_i = max(0, idx - radius)
        hi_i = min(len(bars_10m) - 1, idx + radius)
        local = bars_10m[lo_i : hi_i + 1]
        if not local:
            continue

        local_low = min(float(b.low) for b in local)
        local_high = max(float(b.high) for b in local)
        center = bars_10m[idx]
        center_range = max(eps, float(center.high) - float(center.low))
        entry_px = float(tr.entry_price)

        f_hi_i = min(len(bars_10m) - 1, idx + max(0, int(forward_bars)))
        fwd = bars_10m[idx : f_hi_i + 1]
        fwd_low = min(float(b.low) for b in fwd) if fwd else entry_px
        fwd_high = max(float(b.high) for b in fwd) if fwd else entry_px

        if tr.qty > 0:
            long_n += 1
            dist_pct = max(0.0, (entry_px - local_low) / max(local_low, eps) * 100.0)
            edge = (entry_px - float(center.low)) / center_range
            mfe = max(0.0, (fwd_high - entry_px) / max(entry_px, eps) * 100.0)
            mae = max(0.0, (entry_px - fwd_low) / max(entry_px, eps) * 100.0)
        else:
            short_n += 1
            dist_pct = max(0.0, (local_high - entry_px) / max(local_high, eps) * 100.0)
            edge = (float(center.high) - entry_px) / center_range
            mfe = max(0.0, (entry_px - fwd_low) / max(entry_px, eps) * 100.0)
            mae = max(0.0, (fwd_high - entry_px) / max(entry_px, eps) * 100.0)

        dists.append(float(dist_pct))
        edge_pos.append(float(min(1.0, max(0.0, edge))))
        mfe_vals.append(float(mfe))
        mae_vals.append(float(mae))
        pnl_vals.append(float(tr.pnl(multiplier)))

    if not dists:
        return {
            "samples": 0,
            "avg_dist_local_extrema_pct": 0.0,
            "med_dist_local_extrema_pct": 0.0,
            "p90_dist_local_extrema_pct": 0.0,
            "within_0p25_pct": 0.0,
            "within_0p50_pct": 0.0,
            "within_1p00_pct": 0.0,
            "avg_bar_edge_pos": 0.0,
            "med_bar_edge_pos": 0.0,
            "avg_mfe_pct_fwd": 0.0,
            "avg_mae_pct_fwd": 0.0,
            "avg_mfe_over_mae_fwd": 0.0,
            "avg_pnl_per_trade": 0.0,
            "long_samples": 0,
            "short_samples": 0,
        }

    d_sorted = sorted(dists)
    e_sorted = sorted(edge_pos)
    mfe_sum = sum(mfe_vals)
    mae_sum = sum(mae_vals)
    n = float(len(dists))
    return {
        "samples": int(len(dists)),
        "avg_dist_local_extrema_pct": float(sum(dists) / n),
        "med_dist_local_extrema_pct": float(median(dists)),
        "p90_dist_local_extrema_pct": float(_quantile(d_sorted, 0.90)),
        "within_0p25_pct": float(100.0 * sum(1 for x in dists if x <= 0.25) / n),
        "within_0p50_pct": float(100.0 * sum(1 for x in dists if x <= 0.50) / n),
        "within_1p00_pct": float(100.0 * sum(1 for x in dists if x <= 1.00) / n),
        "avg_bar_edge_pos": float(sum(edge_pos) / n),
        "med_bar_edge_pos": float(median(edge_pos)),
        "avg_mfe_pct_fwd": float(mfe_sum / n),
        "avg_mae_pct_fwd": float(mae_sum / n),
        "avg_mfe_over_mae_fwd": float((mfe_sum / max(mae_sum, eps))),
        "avg_pnl_per_trade": float(sum(pnl_vals) / n),
        "long_samples": int(long_n),
        "short_samples": int(short_n),
    }


def _summary_to_metrics(summary) -> dict[str, Any]:
    dd = float(getattr(summary, "max_drawdown", 0.0) or 0.0)
    dd_pct = float(getattr(summary, "max_drawdown_pct", 0.0) or 0.0)
    pnl = float(getattr(summary, "total_pnl", 0.0) or 0.0)
    roi = float(getattr(summary, "roi", 0.0) or 0.0)
    pnl_dd = pnl / dd if dd > 0 else (math.inf if pnl > 0 else (-math.inf if pnl < 0 else 0.0))
    roi_dd = roi / dd_pct if dd_pct > 0 else (math.inf if roi > 0 else (-math.inf if roi < 0 else 0.0))
    return {
        "trades": int(getattr(summary, "trades", 0) or 0),
        "wins": int(getattr(summary, "wins", 0) or 0),
        "losses": int(getattr(summary, "losses", 0) or 0),
        "win_rate": float(getattr(summary, "win_rate", 0.0) or 0.0),
        "pnl": pnl,
        "roi": roi,
        "max_drawdown": dd,
        "max_drawdown_pct": dd_pct,
        "pnl_over_dd": float(pnl_dd),
        "roi_over_dd_pct": float(roi_dd),
        "avg_hold_hours": float(getattr(summary, "avg_hold_hours", 0.0) or 0.0),
    }


def _parse_windows_from_payload(payload: dict[str, Any]) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    windows = payload.get("windows")
    if not isinstance(windows, list):
        raise SystemExit("Payload missing windows[]")
    for w in windows:
        if not isinstance(w, dict):
            continue
        s = w.get("start")
        e = w.get("end")
        if not s or not e:
            continue
        out.append((_parse_date(str(s)), _parse_date(str(e))))
    if not out:
        raise SystemExit("No valid windows found in payload.")
    return out


def _build_candidate_set(args) -> tuple[list[CandidateSpec], list[tuple[date, date]]]:
    plus_payload = _read_json(args.plus2_file)
    windows = _parse_windows_from_payload(plus_payload)
    out: list[CandidateSpec] = []

    grp = _pick_group(plus_payload, kingmaker_id=int(args.plus2_champion_id))
    out.append(
        _candidate_from_group(
            source_path=args.plus2_file,
            label=f"v31_champion_km{int(args.plus2_champion_id):02d}",
            group=grp,
            kingmaker_id=int(args.plus2_champion_id),
        )
    )

    for raw in args.plus2_beater_ids:
        gid = int(raw)
        grp_b = _pick_group(plus_payload, kingmaker_id=gid)
        out.append(
            _candidate_from_group(
                source_path=args.plus2_file,
                label=f"v31_beater_km{gid:02d}",
                group=grp_b,
                kingmaker_id=gid,
            )
        )

    if args.include_v25:
        v25_payload = _read_json(args.v25_file)
        grp_v25 = _pick_group(v25_payload, kingmaker_id=int(args.v25_id))
        out.append(
            _candidate_from_group(
                source_path=args.v25_file,
                label=f"v25_legacy_km{int(args.v25_id):02d}",
                group=grp_v25,
                kingmaker_id=int(args.v25_id),
            )
        )
    return out, windows


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SLV champion precision audit (entry vs 10m local extrema)")
    p.add_argument(
        "--plus2-file",
        type=Path,
        default=Path("backtests/slv/slv_full24_timegate_plus2_top80_promo_raw.json"),
        help="Source multitimeframe payload for v31 candidates.",
    )
    p.add_argument(
        "--plus2-champion-id",
        type=int,
        default=1,
        help="KINGMAKER id in plus2 payload to treat as champion.",
    )
    p.add_argument(
        "--plus2-beater-id",
        dest="plus2_beater_ids",
        action="append",
        type=int,
        default=[51, 61],
        help="Additional KINGMAKER ids from plus2 payload to compare against champion (repeatable).",
    )
    p.add_argument(
        "--include-v25",
        action="store_true",
        default=True,
        help="Include legacy v25 benchmark in the audit.",
    )
    p.add_argument(
        "--no-include-v25",
        dest="include_v25",
        action="store_false",
        help="Skip legacy v25 benchmark.",
    )
    p.add_argument(
        "--v25-file",
        type=Path,
        default=Path("backtests/slv/slv_exec5m_v25_shock_throttle_drawdown_1h_10y2y1y_mintr500_top80_20260206_173719.json"),
        help="Source payload for legacy v25 benchmark.",
    )
    p.add_argument("--v25-id", type=int, default=1, help="KINGMAKER id in v25 payload.")
    p.add_argument("--cache-dir", type=Path, default=Path("db"), help="Bar cache directory.")
    p.add_argument("--offline", action="store_true", default=True, help="Use cached bars only.")
    p.add_argument("--no-offline", dest="offline", action="store_false", help="Allow IBKR fetch.")
    p.add_argument(
        "--precision-bar-size",
        default="10 mins",
        help="Bar size used for entry precision audit (default: 10 mins).",
    )
    p.add_argument(
        "--precision-use-rth",
        action="store_true",
        default=False,
        help="Use RTH bars for precision mapping (default: full24).",
    )
    p.add_argument(
        "--extrema-radius-bars",
        type=int,
        default=3,
        help="Local extrema radius in bars around entry index (default: 3).",
    )
    p.add_argument(
        "--forward-bars",
        type=int,
        default=12,
        help="Forward window in bars for MFE/MAE probe (default: 12).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: backtests/slv/slv_precision_audit_<utc>.json).",
    )
    return p


def main() -> None:
    args = _arg_parser().parse_args()
    candidates, windows = _build_candidate_set(args)
    if not candidates:
        raise SystemExit("No candidates selected.")

    if args.out is None:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("backtests/slv") / f"slv_precision_audit_{stamp}.json"
    else:
        out_path = Path(args.out)

    data = IBKRHistoricalData()
    payload: dict[str, Any] = {
        "name": "slv_precision_audit",
        "generated_at": _utc_now_iso(),
        "precision_bar_size": str(args.precision_bar_size),
        "precision_use_rth": bool(args.precision_use_rth),
        "extrema_radius_bars": int(args.extrema_radius_bars),
        "forward_bars": int(args.forward_bars),
        "windows": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in windows],
        "candidates": [],
    }

    print("")
    print("SLV precision audit")
    print(f"- candidates={len(candidates)} windows={len(windows)} precision_bar={args.precision_bar_size} rth={bool(args.precision_use_rth)}")
    print(f"- offline={bool(args.offline)} cache_dir={args.cache_dir}")
    print("")

    for cand in candidates:
        print(f"[candidate] {cand.label} :: KINGMAKER #{cand.kingmaker_id:02d}")
        filters_cfg = _filters_from_payload(cand.filters)
        strat_cfg = _strategy_from_payload(cand.strategy, filters=filters_cfg)

        signal_bar_size = str(cand.strategy.get("signal_bar_size") or "10 mins")
        signal_use_rth = bool(cand.strategy.get("signal_use_rth")) if ("signal_use_rth" in cand.strategy) else False

        cand_out: dict[str, Any] = {
            "label": cand.label,
            "source_path": str(cand.source_path),
            "kingmaker_id": int(cand.kingmaker_id),
            "group_name": cand.group_name,
            "signal_bar_size": signal_bar_size,
            "signal_use_rth": bool(signal_use_rth),
            "windows": [],
        }

        for wstart, wend in windows:
            bundle = _mk_bundle(
                strategy=strat_cfg,
                start=wstart,
                end=wend,
                bar_size=signal_bar_size,
                use_rth=signal_use_rth,
                cache_dir=args.cache_dir,
                offline=bool(args.offline),
            )
            start_dt = datetime.combine(wstart, time(0, 0))
            end_dt = datetime.combine(wend, time(23, 59))
            bars_sig = _load_bars(
                data,
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bundle.backtest.bar_size,
                use_rth=bundle.backtest.use_rth,
                cache_dir=args.cache_dir,
                offline=bool(args.offline),
            )
            if not bars_sig:
                raise SystemExit(
                    f"No signal bars for {bundle.strategy.symbol} {bundle.backtest.bar_size} "
                    f"{wstart.isoformat()}->{wend.isoformat()} rth={bundle.backtest.use_rth}"
                )

            regime_bars, regime2_bars, tick_bars, exec_bars = _load_context_bars(
                bundle=bundle,
                data=data,
                cache_dir=args.cache_dir,
                start_dt=start_dt,
                end_dt=end_dt,
                offline=bool(args.offline),
            )
            meta = _resolve_meta(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                offline=bool(args.offline),
                data=(None if bool(args.offline) else data),
            )
            result = _run_spot_backtest(
                bundle,
                bars_sig,
                meta,
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
                tick_bars=tick_bars,
                exec_bars=exec_bars,
            )

            bars_10m = _load_bars(
                data,
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=str(args.precision_bar_size),
                use_rth=bool(args.precision_use_rth),
                cache_dir=args.cache_dir,
                offline=bool(args.offline),
            )
            if not bars_10m:
                raise SystemExit(
                    f"No precision bars for {bundle.strategy.symbol} {args.precision_bar_size} "
                    f"{wstart.isoformat()}->{wend.isoformat()} rth={bool(args.precision_use_rth)}"
                )

            trades = [t for t in result.trades if isinstance(t, SpotTrade)]
            metrics = _summary_to_metrics(result.summary)
            precision = _precision_stats(
                trades=trades,
                bars_10m=bars_10m,
                radius=max(0, int(args.extrema_radius_bars)),
                forward_bars=max(0, int(args.forward_bars)),
                multiplier=float(meta.multiplier),
            )
            row = {
                "start": wstart.isoformat(),
                "end": wend.isoformat(),
                "metrics": metrics,
                "precision": precision,
            }
            cand_out["windows"].append(row)
            print(
                "  "
                + f"{wstart.isoformat()}->{wend.isoformat()} "
                + f"tr={metrics['trades']} pnl={metrics['pnl']:.1f} dd={metrics['max_drawdown']:.1f} "
                + f"dist_med={precision['med_dist_local_extrema_pct']:.3f}% "
                + f"<=0.5%={precision['within_0p50_pct']:.1f}% "
                + f"mfe/mae={precision['avg_mfe_over_mae_fwd']:.2f}"
            )

        payload["candidates"].append(cand_out)
        print("")

    write_json(out_path, payload, sort_keys=False)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
