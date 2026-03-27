"""Export a spot predictive research table from a milestones preset.

Goal:
- turn the current entry/exit observability into a reusable dataset
- label each closed trade with short/medium forward path quality after entry
- keep the tool generic across spot presets, symbols, and windows
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path

from ..data import ContractMeta, IBKRHistoricalData
from ..engine import _run_spot_backtest, _spot_multiplier
from ..models import SpotTrade
from ..multiwindow_helpers import load_bars
from ..spot_codec import effective_filters_payload, filters_from_payload, make_bundle, strategy_from_payload
from ..spot_context import load_spot_context_bars, spot_signal_warmup_days_from_strategy


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_window(raw: str) -> tuple[date, date]:
    text = str(raw).strip()
    if ":" not in text:
        raise SystemExit(f"Invalid --window {raw!r} (expected YYYY-MM-DD:YYYY-MM-DD)")
    a, b = text.split(":", 1)
    start = _parse_date(a)
    end = _parse_date(b)
    if end <= start:
        raise SystemExit(f"Invalid window range: {start}..{end}")
    return start, end


def _age_bucket(age: int | None) -> str:
    if age is None:
        return "none"
    if age < 100:
        return "0_99"
    if age < 500:
        return "100_499"
    if age < 1600:
        return "500_1599"
    return "1600_plus"


def _horizon_specs(exec_bar_size: str) -> list[tuple[str, int]]:
    label = str(exec_bar_size or "").strip().lower()
    if label.startswith("1 min"):
        return [("15m", 15), ("1h", 60), ("6h30m", 390)]
    if label.startswith("2 min"):
        return [("15m", 8), ("1h", 30), ("6h30m", 195)]
    if label.startswith("5 min"):
        return [("15m", 3), ("1h", 12), ("6h30m", 78)]
    if label.startswith("10 min"):
        return [("30m", 3), ("1h", 6), ("6h30m", 39)]
    if label.startswith("15 min"):
        return [("45m", 3), ("1h", 4), ("6h30m", 26)]
    if label.startswith("30 min"):
        return [("1h30m", 3), ("3h", 6), ("6h30m", 13)]
    if label.startswith("1 hour"):
        return [("3h", 3), ("6h", 6), ("13h", 13)]
    return [("3b", 3), ("12b", 12), ("78b", 78)]


def _load_candidate(
    *,
    milestones_path: Path,
    symbol: str,
    bar_size: str,
    use_rth: bool,
) -> tuple[dict, dict | None]:
    payload = json.loads(milestones_path.read_text())
    groups = payload.get("groups") or []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not entries or not isinstance(entries[0], dict):
            continue
        strategy = entries[0].get("strategy") or {}
        if not isinstance(strategy, dict):
            continue
        if str(strategy.get("symbol") or "").strip().upper() != symbol:
            continue
        if str(strategy.get("signal_bar_size") or "").strip().lower() != bar_size:
            continue
        if bool(strategy.get("signal_use_rth")) != use_rth:
            continue
        return strategy, (group.get("filters") if isinstance(group.get("filters"), dict) else None)
    raise SystemExit(
        f"No matching spot preset found in {milestones_path} for symbol={symbol} bar={bar_size} rth={use_rth}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a predictive spot trade table from a milestones preset.")
    ap.add_argument("--milestones", required=True, help="Input milestones / champion JSON.")
    ap.add_argument("--symbol", default="TQQQ", help="Symbol filter (default: TQQQ).")
    ap.add_argument("--bar-size", default="5 mins", help="Signal bar size filter (default: 5 mins).")
    ap.add_argument("--use-rth", action="store_true", default=False, help="Require signal_use_rth=true.")
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="YYYY-MM-DD:YYYY-MM-DD (repeatable). Required.",
    )
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args(argv)

    if not args.window:
        raise SystemExit("At least one --window is required.")

    symbol = str(args.symbol).strip().upper()
    bar_size = str(args.bar_size).strip().lower()
    use_rth = bool(args.use_rth)
    milestones_path = Path(args.milestones)
    out_path = Path(args.out)

    strategy_payload, group_filters = _load_candidate(
        milestones_path=milestones_path,
        symbol=symbol,
        bar_size=bar_size,
        use_rth=use_rth,
    )
    strategy_payload = deepcopy(strategy_payload)
    effective_filters = effective_filters_payload(group_filters=group_filters, strategy=strategy_payload)
    filters = filters_from_payload(effective_filters)
    strat_cfg = strategy_from_payload(strategy_payload, filters=filters)

    data = IBKRHistoricalData()
    meta = ContractMeta(
        symbol=strat_cfg.symbol,
        exchange="SMART",
        multiplier=_spot_multiplier(strat_cfg.symbol, False),
        min_tick=0.01,
    )
    exec_bar_size = str(getattr(strat_cfg, "spot_exec_bar_size", "") or bar_size)
    horizon_specs = _horizon_specs(exec_bar_size)

    fieldnames = [
        "window",
        "symbol",
        "entry_time",
        "exit_time",
        "qty",
        "entry_branch",
        "exit_reason",
        "pnl",
        "regime4_state",
        "shock_dir",
        "hard_dir",
        "release_age_bars",
        "release_age_bucket",
        "entry_15m_pos",
        "entry_1h_pos",
        "entry_6h30m_pos",
        "tr_ratio",
        "tr_median_pct",
        "slope_med_pct",
        "slope_vel_pct",
        "slope_med_slow_pct",
        "slope_vel_slow_pct",
        "shock_atr_pct",
        "shock_atr_vel_pct",
        "shock_atr_accel_pct",
        "exit_signal_age_bars",
    ]
    for label, _bars in horizon_specs:
        fieldnames.extend([f"{label}_fwd_close_pct", f"{label}_mfe_pct", f"{label}_mae_pct"])

    rows: list[dict[str, object]] = []
    for raw_window in args.window:
        start, end = _parse_window(raw_window)
        bundle = make_bundle(
            strategy=strat_cfg,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=Path("db"),
            offline=True,
        )
        start_dt = datetime.combine(start, time(0, 0))
        end_dt = datetime.combine(end, time(23, 59))
        signal_start_dt = start_dt - timedelta(
            days=max(
                0,
                int(
                    spot_signal_warmup_days_from_strategy(
                        strategy=bundle.strategy,
                        default_signal_bar_size=str(bundle.backtest.bar_size),
                        default_signal_use_rth=bool(bundle.backtest.use_rth),
                    )
                ),
            )
        )
        bars_sig = load_bars(
            data,
            symbol=bundle.strategy.symbol,
            exchange=bundle.strategy.exchange,
            start_dt=signal_start_dt,
            end_dt=end_dt,
            bar_size=bundle.backtest.bar_size,
            use_rth=bundle.backtest.use_rth,
            cache_dir=Path("db"),
            offline=True,
        )
        ctx = load_spot_context_bars(
            strategy=bundle.strategy,
            default_symbol=str(bundle.strategy.symbol),
            default_exchange=bundle.strategy.exchange,
            default_signal_bar_size=str(bundle.backtest.bar_size),
            default_signal_use_rth=bool(bundle.backtest.use_rth),
            start_dt=start_dt,
            end_dt=end_dt,
            load_requirement=lambda req, req_start, req_end: load_bars(
                data,
                symbol=req.symbol,
                exchange=req.exchange,
                start_dt=req_start,
                end_dt=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
                cache_dir=Path("db"),
                offline=True,
            ),
            on_missing=lambda req, req_start, req_end: (_ for _ in ()).throw(
                RuntimeError(f"missing {req.kind} bars for {req.symbol}")
            ),
        )
        result = _run_spot_backtest(
            bundle,
            bars_sig,
            meta,
            regime_bars=ctx.regime_bars,
            regime2_bars=ctx.regime2_bars,
            regime2_bear_hard_bars=ctx.regime2_bear_hard_bars,
            tick_bars=ctx.tick_bars,
            exec_bars=ctx.exec_bars,
        )
        exec_bars = ctx.exec_bars if isinstance(ctx.exec_bars, list) else bars_sig
        ts_to_idx = {bar.ts: i for i, bar in enumerate(exec_bars)}
        window_label = f"{start.isoformat()}->{end.isoformat()}"
        for trade in result.trades:
            if not isinstance(trade, SpotTrade) or trade.exit_time is None:
                continue
            dt = trade.decision_trace or {}
            gi = dt.get("entry_guard_inputs") or {}
            ep = dt.get("entry_local_extrema_probe") or {}
            exits = dt.get("exits") if isinstance(dt.get("exits"), list) else []
            last_exit = exits[-1] if exits else {}
            signal_snapshot = last_exit.get("signal_snapshot") if isinstance(last_exit, dict) else {}
            entry_idx = ts_to_idx.get(trade.entry_time)
            if entry_idx is None:
                continue
            row = {
                "window": window_label,
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "qty": int(trade.qty),
                "entry_branch": trade.entry_branch or dt.get("entry_branch"),
                "exit_reason": trade.exit_reason,
                "pnl": float(trade.pnl(1.0)),
                "regime4_state": dt.get("regime4_state"),
                "shock_dir": dt.get("shock_dir"),
                "hard_dir": dt.get("regime2_bear_hard_dir"),
                "release_age_bars": dt.get("regime2_bear_hard_release_age_bars"),
                "release_age_bucket": _age_bucket(dt.get("regime2_bear_hard_release_age_bars")),
                "entry_15m_pos": ((ep.get("15m") or {}).get("range_pos") if isinstance(ep, dict) else None),
                "entry_1h_pos": ((ep.get("1h") or {}).get("range_pos") if isinstance(ep, dict) else None),
                "entry_6h30m_pos": ((ep.get("6h30m") or {}).get("range_pos") if isinstance(ep, dict) else None),
                "tr_ratio": gi.get("tr_ratio"),
                "tr_median_pct": gi.get("tr_median_pct"),
                "slope_med_pct": gi.get("slope_med_pct"),
                "slope_vel_pct": gi.get("slope_vel_pct"),
                "slope_med_slow_pct": gi.get("slope_med_slow_pct"),
                "slope_vel_slow_pct": gi.get("slope_vel_slow_pct"),
                "shock_atr_pct": gi.get("shock_atr_pct"),
                "shock_atr_vel_pct": gi.get("shock_atr_vel_pct"),
                "shock_atr_accel_pct": gi.get("shock_atr_accel_pct"),
                "exit_signal_age_bars": (
                    signal_snapshot.get("signal_snapshot_age_bars") if isinstance(signal_snapshot, dict) else None
                ),
            }
            entry_price = float(trade.entry_price)
            for label, bars_ahead in horizon_specs:
                start_idx = int(entry_idx) + 1
                end_idx = min(len(exec_bars) - 1, int(entry_idx) + int(bars_ahead))
                if start_idx > end_idx:
                    row[f"{label}_fwd_close_pct"] = None
                    row[f"{label}_mfe_pct"] = None
                    row[f"{label}_mae_pct"] = None
                    continue
                window = exec_bars[start_idx : end_idx + 1]
                row[f"{label}_fwd_close_pct"] = ((float(window[-1].close) - entry_price) / entry_price) * 100.0
                row[f"{label}_mfe_pct"] = ((max(float(b.high) for b in window) - entry_price) / entry_price) * 100.0
                row[f"{label}_mae_pct"] = ((min(float(b.low) for b in window) - entry_price) / entry_price) * 100.0
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} rows={len(rows)} horizons={[label for label, _ in horizon_specs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
