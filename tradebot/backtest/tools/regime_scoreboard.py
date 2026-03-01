"""Regime scoreboard for spot backtests.

Goal: make "bad months / chop / downturn" measurable.

This tool runs a backtest from a config JSON, then outputs a per-month table:
- outcomes (pnl, trade-DD, pnl/DD)
- overlay firing rates (riskpanic/shock/ddBoost etc) observed at entry
- timing distribution (how much entry activity happens in the open)

Design notes:
- We intentionally use the in-memory BacktestResult (trades + equity) so we don't
  need to scrape CSVs or depend on output_dir.
- We group trade metrics by *entry month* so overlay flags at entry line up with
  outcome attribution (good enough for steering evolutions).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from ..config import load_config
from ..engine import run_backtest
from ..models import EquityPoint, SpotTrade
from ...engine import _trade_date, _trade_hour_et, _ts_to_et


def _parse_yyyy_mm_dd(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.1f}"


def _fmt_ratio(x: float | None) -> str:
    if x is None:
        return ""
    if math.isinf(x):
        return "inf"
    return f"{x:.3f}"


def _max_drawdown(points: list[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for v in points:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    return max_dd


def _trade_level_dd_by_exit(trades: list[SpotTrade]) -> float:
    """Compute drawdown of cumulative *realized* trade PnL within the group.

    Ordering by exit_time gives a stable realized curve.
    """

    closed = [t for t in trades if t.exit_time is not None]
    closed.sort(key=lambda t: t.exit_time or t.entry_time)
    curve: list[float] = []
    cum = 0.0
    for t in closed:
        cum += t.pnl(multiplier=1.0)
        curve.append(cum)
    return _max_drawdown(curve) if curve else 0.0


def _equity_month_view(points: list[EquityPoint]) -> dict[str, list[float]]:
    by_month: dict[str, list[float]] = defaultdict(list)
    for p in points:
        d = _trade_date(p.ts)
        by_month[_month_key(d)].append(float(p.equity))
    return dict(by_month)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-month regime scoreboard for spot backtests.")
    ap.add_argument("--config", required=True, help="Backtest config JSON.")
    ap.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD).")
    ap.add_argument("--out", default="backtests/out/regime_scoreboard.md", help="Output markdown path.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    start = _parse_yyyy_mm_dd(args.start) if args.start else cfg.backtest.start
    end = _parse_yyyy_mm_dd(args.end) if args.end else cfg.backtest.end

    # Avoid writing artifacts; this tool is analysis-first.
    cfg = replace(cfg, backtest=replace(cfg.backtest, start=start, end=end, output_dir="", offline=True))

    res = run_backtest(cfg)
    trades = [t for t in res.trades if isinstance(t, SpotTrade) and t.exit_time is not None]

    # Trade-level metrics grouped by entry month (ET).
    by_month: dict[str, list[SpotTrade]] = defaultdict(list)
    for t in trades:
        by_month[_month_key(_trade_date(t.entry_time))].append(t)

    # Equity mark-to-market view grouped by month (ET).
    equity_by_month = _equity_month_view(res.equity)

    # Timing window: first 2 hours of RTH open (09:30–11:30 ET).
    open_start_min = 9 * 60 + 30
    open_end_min = 11 * 60 + 30

    lines: list[str] = []
    lines.append("# Regime Scoreboard")
    lines.append("")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Window: `{start} -> {end}`")
    lines.append(f"- Trades (closed): **{len(trades)}**")
    lines.append("")
    lines.append("## Monthly (grouped by trade entry month, ET)")
    lines.append("")
    lines.append("| Month | Trades | PnL | Trade DD | PnL/DD | Long PnL | Short PnL | Riskpanic% | Shock% | ddBoost% | BranchB% | Open2h% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    months = sorted(by_month.keys())
    worst_by_pdd: list[tuple[float, str]] = []
    worst_by_dd: list[tuple[float, str]] = []

    for m in months:
        mt = by_month[m]
        pnl = sum(t.pnl(multiplier=1.0) for t in mt)
        trade_dd = _trade_level_dd_by_exit(mt)
        pdd = (pnl / trade_dd) if trade_dd > 0 else (float("inf") if pnl > 0 else None)

        long_pnl = sum(t.pnl(multiplier=1.0) for t in mt if t.qty > 0)
        short_pnl = sum(t.pnl(multiplier=1.0) for t in mt if t.qty < 0)

        riskpanic_n = 0
        shock_n = 0
        ddboost_n = 0
        branch_b_n = 0
        open2h_n = 0

        for t in mt:
            dt = t.decision_trace or {}
            if bool(dt.get("riskpanic")):
                riskpanic_n += 1
            if bool(dt.get("shock")):
                shock_n += 1
            if bool(dt.get("shock_short_boost_applied")):
                ddboost_n += 1

            br = (t.entry_branch or dt.get("entry_branch") or "").strip().lower()
            if br == "b":
                branch_b_n += 1

            et = _ts_to_et(t.entry_time)
            mins = et.hour * 60 + et.minute
            if open_start_min <= mins < open_end_min:
                open2h_n += 1

        denom = max(len(mt), 1)
        riskpanic_pct = 100.0 * riskpanic_n / denom
        shock_pct = 100.0 * shock_n / denom
        ddboost_pct = 100.0 * ddboost_n / denom
        branchb_pct = 100.0 * branch_b_n / denom
        open2h_pct = 100.0 * open2h_n / denom

        if pdd is not None and not math.isinf(pdd):
            worst_by_pdd.append((pdd, m))
        worst_by_dd.append((trade_dd, m))

        lines.append(
            "| "
            + " | ".join(
                [
                    m,
                    f"{len(mt)}",
                    _fmt_money(pnl),
                    _fmt_money(trade_dd),
                    _fmt_ratio(pdd),
                    _fmt_money(long_pnl),
                    _fmt_money(short_pnl),
                    f"{riskpanic_pct:5.1f}",
                    f"{shock_pct:5.1f}",
                    f"{ddboost_pct:5.1f}",
                    f"{branchb_pct:5.1f}",
                    f"{open2h_pct:5.1f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Monthly (mark-to-market equity view, ET)")
    lines.append("")
    lines.append("| Month | MTM PnL | MTM DD | MTM PnL/DD |")
    lines.append("|---|---:|---:|---:|")
    for m in sorted(equity_by_month.keys()):
        eq = equity_by_month[m]
        if not eq:
            continue
        mtm_pnl = eq[-1] - eq[0]
        mtm_dd = _max_drawdown(eq)
        mtm_pdd = (mtm_pnl / mtm_dd) if mtm_dd > 0 else (float("inf") if mtm_pnl > 0 else None)
        lines.append(f"| {m} | {_fmt_money(mtm_pnl)} | {_fmt_money(mtm_dd)} | {_fmt_ratio(mtm_pdd)} |")

    # Worst-month callouts (trade-level).
    worst_by_pdd.sort(key=lambda t: t[0])
    worst_by_dd.sort(key=lambda t: t[0], reverse=True)

    lines.append("")
    lines.append("## Callouts")
    lines.append("")
    if worst_by_pdd:
        lines.append("- Worst trade-level months by pnl/dd (entry-month attribution):")
        for pdd, m in worst_by_pdd[:5]:
            lines.append(f"  - `{m}` pnl/dd={pdd:.3f}")
    if worst_by_dd:
        lines.append("- Biggest trade-level drawdown months by trade-DD (entry-month attribution):")
        for dd, m in worst_by_dd[:5]:
            lines.append(f"  - `{m}` trade-DD={dd:,.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

