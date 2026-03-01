"""Spot contract evaluator (multi-window robustness gate).

Why this exists:
- spot_multitimeframe is great for sweeps, but it doesn't standardize "promotion contract"
  windows or produce an audit-friendly report with per-window metrics.
- We want to prevent "recent-only gods" by evaluating the same config across multiple
  shifted 1Y windows (and 2Y / later 10Y) with the real exec bar size.

This tool runs full engine backtests per window (offline), then prints:
- pnl/dd per window
- a 'promotion floor' and 'robustness floor'
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

from ..config import load_config
from ..engine import run_backtest


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.1f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _fmt_ratio(x: float | None) -> str:
    if x is None:
        return ""
    if math.isinf(x):
        return "inf"
    return f"{x:.3f}"


def _window_years(start: date, end: date) -> float:
    days = max((end - start).days, 1)
    return days / 365.25


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a spot config across multiple windows.")
    ap.add_argument("--config", required=True, help="Config JSON (replay) to evaluate.")
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="YYYY-MM-DD:YYYY-MM-DD (repeatable).",
    )
    ap.add_argument(
        "--min-trades-per-year",
        type=int,
        default=500,
        help="Reject windows with fewer than ceil(years * N) trades.",
    )
    ap.add_argument("--out", default=None, help="Optional markdown output path.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg = replace(cfg, backtest=replace(cfg.backtest, offline=True, output_dir=""))

    windows: list[tuple[str, date, date]] = []
    for raw in args.window:
        text = str(raw).strip()
        if ":" not in text:
            raise SystemExit(f"Invalid --window {raw!r} (expected YYYY-MM-DD:YYYY-MM-DD)")
        a, b = text.split(":", 1)
        start = _parse_date(a)
        end = _parse_date(b)
        if end <= start:
            raise SystemExit(f"Invalid window range: {start}..{end}")
        label = f"{start.isoformat()}→{end.isoformat()}"
        windows.append((label, start, end))

    if not windows:
        raise SystemExit("No windows provided. Use --window YYYY-MM-DD:YYYY-MM-DD (repeatable).")

    rows: list[dict[str, object]] = []
    for label, start, end in windows:
        years = _window_years(start, end)
        min_trades = int(math.ceil(years * int(args.min_trades_per_year)))
        c = replace(cfg, backtest=replace(cfg.backtest, start=start, end=end, output_dir=""))
        s = run_backtest(c).summary
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        pdd = (pnl / dd) if dd > 0 else (float("inf") if pnl > 0 else None)
        tr = int(s.trades or 0)
        ok = (tr >= min_trades) and (pnl > 0)
        rows.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "years": years,
                "min_trades": min_trades,
                "ok": ok,
                "tr": tr,
                "win": float(s.win_rate or 0.0),
                "pnl": pnl,
                "dd": dd,
                "pdd": pdd,
                "roi": float(s.roi or 0.0),
                "dd_pct": float(s.max_drawdown_pct or 0.0),
            }
        )

    floors = [r["pdd"] for r in rows if isinstance(r.get("pdd"), float) and not math.isinf(float(r["pdd"]))]  # type: ignore[arg-type]
    floor = min(floors) if floors else None

    lines: list[str] = []
    lines.append("# Spot Contract Eval")
    lines.append("")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- min_trades_per_year={int(args.min_trades_per_year)}")
    lines.append("")
    lines.append("| Window | OK | Trades | MinTrades | Win | PnL | DD | PnL/DD | ROI | DD% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["label"]),
                    "yes" if bool(r["ok"]) else "no",
                    str(int(r["tr"])),
                    str(int(r["min_trades"])),
                    _fmt_pct(float(r["win"])),
                    _fmt_money(float(r["pnl"])),
                    _fmt_money(float(r["dd"])),
                    _fmt_ratio(r["pdd"] if isinstance(r.get("pdd"), float) else None),
                    _fmt_pct(float(r["roi"])),
                    _fmt_pct(float(r["dd_pct"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"- Floor (min pnl/dd across windows): **{_fmt_ratio(floor)}**")
    lines.append("")

    report = "\n".join(lines) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Wrote {out_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

