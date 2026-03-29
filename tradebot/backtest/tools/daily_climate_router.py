"""CLI wrapper for the central daily climate router prototype."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from ...climate_router import (
    buyhold_year_pdd,
    classify_climate_v2,
    compute_year_features,
    hf_host_year_stats,
    load_daily_bars_from_intraday_csv,
    load_hf_host_strategy,
    moving_average_year_pdd,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prototype daily climate router for TQQQ host selection.")
    ap.add_argument("--intraday-csv", default="db/TQQQ/TQQQ_2016-01-01_2026-01-19_1min_rth.csv")
    ap.add_argument(
        "--hf-milestones",
        default="backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json",
    )
    ap.add_argument("--hf-label", default="v43")
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--out-csv", default="/tmp/tqqq_daily_climate_years_v2.csv")
    ap.add_argument("--out-md", default="/tmp/tqqq_daily_climate_router_v2.md")
    args = ap.parse_args(argv)

    intraday_csv = Path(args.intraday_csv)
    hf_milestones = Path(args.hf_milestones)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    daily_bars = load_daily_bars_from_intraday_csv(intraday_csv)
    hf_strategy, hf_bar_size, hf_use_rth = load_hf_host_strategy(hf_milestones)

    rows: list[dict[str, object]] = []
    for year in range(int(args.start_year), int(args.end_year) + 1):
        features = compute_year_features(daily_bars, year)
        decision = classify_climate_v2(features)
        buy_pnl, buy_dd, buy_pdd = buyhold_year_pdd(daily_bars, year)
        sma_pnl, sma_dd, sma_pdd = moving_average_year_pdd(daily_bars, year, window=200)
        hf_pnl, hf_dd, hf_pdd, hf_trades, hf_win_rate = hf_host_year_stats(
            strategy=hf_strategy,
            year=year,
            bar_size=hf_bar_size,
            use_rth=hf_use_rth,
        )

        chosen = {
            "buyhold": ("buyhold", buy_pnl, buy_dd, buy_pdd, 0, 0.0),
            "sma200": ("sma200", sma_pnl, sma_dd, sma_pdd, 0, 0.0),
            "hf_host": (str(args.hf_label), hf_pnl, hf_dd, hf_pdd, hf_trades, hf_win_rate),
        }[decision.chosen_host]

        rows.append(
            {
                "year": int(year),
                "ret": float(features.ret),
                "maxdd": float(features.maxdd),
                "rv": float(features.rv),
                "atr_med": float(features.atr_med),
                "atr_mean": float(features.atr_mean),
                "up_frac": float(features.up_frac),
                "efficiency": float(features.efficiency),
                "dd_frac_ge_10pct": float(features.dd_frac_ge_10pct),
                "climate_v2": str(decision.climate),
                "buyhold_pnl": float(buy_pnl),
                "buyhold_dd_pct": float(buy_dd),
                "buyhold_pdd": float(buy_pdd),
                "sma200_pnl": float(sma_pnl),
                "sma200_dd_pct": float(sma_dd),
                "sma200_pdd": float(sma_pdd),
                "hf_host_label": str(args.hf_label),
                "hf_host_pnl": float(hf_pnl),
                "hf_host_dd_pct": float(hf_dd),
                "hf_host_pdd": float(hf_pdd),
                "hf_host_trades": int(hf_trades),
                "hf_host_win_rate": float(hf_win_rate),
                "chosen_host_v2": str(chosen[0]),
                "chosen_pnl_v2": float(chosen[1]),
                "chosen_dd_pct_v2": float(chosen[2]),
                "chosen_pdd_v2": float(chosen[3]),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    chosen_vals = [float(row["chosen_pdd_v2"]) for row in rows]
    lines: list[str] = []
    lines.append("# Daily Climate Router Prototype v2")
    lines.append("")
    lines.append("Rule set:")
    lines.append("- `bull_grind_low_vol`: `ret>0`, `maxdd<=0.40`, `rv<=0.55`, `dd_frac_ge_10pct<0.45` -> host=`buyhold`")
    lines.append("- `positive_high_stress_transition`: `ret>0` and not bull-grind -> host=`hf_host`")
    lines.append("- `negative_extreme_bear`: `ret<=0` and (`maxdd>=0.70` or `rv>=0.85` or `dd_frac_ge_10pct>=0.80`) -> host=`hf_host`")
    lines.append("- `negative_transition_bear`: otherwise -> host=`sma200`")
    lines.append("")
    lines.append("| Year | ret | maxdd | rv | eff | dd>=10% frac | climate | buyhold | sma200 | hf_host | chosen | chosen pdd |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |")
    for row in rows:
        lines.append(
            f"| {int(row['year'])} | {float(row['ret']):.3f} | {float(row['maxdd']):.3f} | "
            f"{float(row['rv']):.3f} | {float(row['efficiency']):.3f} | {float(row['dd_frac_ge_10pct']):.3f} | "
            f"{row['climate_v2']} | {float(row['buyhold_pdd']):.3f} | {float(row['sma200_pdd']):.3f} | "
            f"{float(row['hf_host_pdd']):.3f} | {row['chosen_host_v2']} | {float(row['chosen_pdd_v2']):.3f} |"
        )
    lines.append("")
    lines.append(f"Worst-year chosen pdd: **{min(chosen_vals):.3f}**")
    lines.append(f"Average chosen pdd: **{mean(chosen_vals):.3f}**")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(out_csv)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
