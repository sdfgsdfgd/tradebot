"""Export daily regime-router features and forward outcomes for host-path distillation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from ...climate_router import bull_sovereign_entry_ok, load_daily_bars_from_intraday_csv, moving_average_target_dir, rolling_climate_states


def _future_stats(days, start_idx: int, horizon: int) -> tuple[float | None, float | None]:
    end_idx = min(len(days) - 1, int(start_idx) + int(horizon))
    if end_idx <= int(start_idx):
        return None, None
    start_close = float(days[start_idx].close)
    seg = days[int(start_idx) + 1 : int(end_idx) + 1]
    if not seg or start_close <= 0.0:
        return None, None
    end_close = float(seg[-1].close)
    peak = start_close
    maxdd = 0.0
    for bar in seg:
        close = float(bar.close)
        if close > peak:
            peak = close
        dd = (peak - close) / peak if peak > 0.0 else 0.0
        if dd > maxdd:
            maxdd = dd
    return (end_close / start_close) - 1.0, maxdd


def _precompute_ma200_dirs(days) -> list[str | None]:
    out: list[str | None] = []
    for idx in range(len(days)):
        out.append(moving_average_target_dir(days[: idx + 1], window=200))
    return out


def _forward_host_ret(days, *, start_idx: int, horizon: int, dir_by_idx: list[str | None]) -> float | None:
    end_idx = min(len(days) - 1, int(start_idx) + int(horizon))
    if end_idx <= int(start_idx):
        return None
    equity = 1.0
    prev_close = float(days[start_idx].close)
    for idx in range(int(start_idx) + 1, int(end_idx) + 1):
        pos = 1.0 if dir_by_idx[int(idx)] == "up" else 0.0
        close = float(days[idx].close)
        equity *= 1.0 + (pos * ((close / prev_close) - 1.0))
        prev_close = close
    return equity - 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export daily regime-router feature rows for host-path distillation.")
    ap.add_argument("--intraday-csv", default="db/TQQQ/TQQQ_2016-01-01_2026-01-19_1min_rth.csv")
    ap.add_argument("--fast-window", type=int, default=63)
    ap.add_argument("--slow-window", type=int, default=126)
    ap.add_argument("--dwell", type=int, default=10)
    ap.add_argument("--out-csv", default="/tmp/tqqq_regime_router_distill.csv")
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_distill.md")
    args = ap.parse_args(argv)

    days = load_daily_bars_from_intraday_csv(Path(args.intraday_csv))
    states = rolling_climate_states(
        days,
        fast_window_days=int(args.fast_window),
        slow_window_days=int(args.slow_window),
        min_dwell_days=int(args.dwell),
    )
    ma200_dir_by_idx = _precompute_ma200_dirs(days)
    idx_by_ts = {str(bar.ts): i for i, bar in enumerate(days)}

    rows: list[dict[str, object]] = []
    for state in states:
        ts = str(state.ts)
        idx = idx_by_ts.get(ts)
        if idx is None:
            continue
        effective_host = (
            "bull_ma200_v1"
            if bull_sovereign_entry_ok(
                climate=str(state.active.climate),
                chosen_host=str(state.active.chosen_host),
                fast_features=state.fast_features,
                slow_features=state.slow_features,
            )
            else str(state.active.chosen_host)
        )
        f5_ret, f5_dd = _future_stats(days, idx, 5)
        f10_ret, f10_dd = _future_stats(days, idx, 10)
        f21_ret, f21_dd = _future_stats(days, idx, 21)
        bull_ma200_fwd_21d_ret = _forward_host_ret(days, start_idx=idx, horizon=21, dir_by_idx=ma200_dir_by_idx)
        rows.append(
            {
                "ts": ts,
                "year": int(ts[:4]),
                "month": ts[:7],
                "climate": str(state.active.climate),
                "base_host": str(state.active.chosen_host),
                "effective_host": effective_host,
                "bull_sovereign_ok": int(effective_host == "bull_ma200_v1"),
                "fast_ret": float(state.fast_features.ret),
                "fast_maxdd": float(state.fast_features.maxdd),
                "fast_rv": float(state.fast_features.rv),
                "fast_eff": float(state.fast_features.efficiency),
                "fast_up_frac": float(state.fast_features.up_frac),
                "fast_dd10_frac": float(state.fast_features.dd_frac_ge_10pct),
                "slow_ret": float(state.slow_features.ret),
                "slow_maxdd": float(state.slow_features.maxdd),
                "slow_rv": float(state.slow_features.rv),
                "slow_eff": float(state.slow_features.efficiency),
                "slow_up_frac": float(state.slow_features.up_frac),
                "slow_dd10_frac": float(state.slow_features.dd_frac_ge_10pct),
                "fwd_5d_ret": float(f5_ret) if f5_ret is not None else None,
                "fwd_5d_maxdd": float(f5_dd) if f5_dd is not None else None,
                "fwd_10d_ret": float(f10_ret) if f10_ret is not None else None,
                "fwd_10d_maxdd": float(f10_dd) if f10_dd is not None else None,
                "fwd_21d_ret": float(f21_ret) if f21_ret is not None else None,
                "fwd_21d_maxdd": float(f21_dd) if f21_dd is not None else None,
                "bull_ma200_fwd_21d_ret": (
                    float(bull_ma200_fwd_21d_ret) if bull_ma200_fwd_21d_ret is not None else None
                ),
                "bull_ma200_adv_21d": (
                    float(bull_ma200_fwd_21d_ret) - float(f21_ret)
                    if bull_ma200_fwd_21d_ret is not None and f21_ret is not None
                    else None
                ),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_climate: dict[str, int] = {}
    by_host: dict[str, int] = {}
    by_year_host: dict[tuple[int, str], int] = {}
    for row in rows:
        climate = str(row["climate"])
        host = str(row["effective_host"])
        year = int(row["year"])
        by_climate[climate] = by_climate.get(climate, 0) + 1
        by_host[host] = by_host.get(host, 0) + 1
        by_year_host[(year, host)] = by_year_host.get((year, host), 0) + 1

    lines: list[str] = []
    lines.append("# Regime Router Distillation Dataset")
    lines.append("")
    lines.append(f"Rows: **{len(rows)}**")
    lines.append(f"Fast window: **{int(args.fast_window)}**")
    lines.append(f"Slow window: **{int(args.slow_window)}**")
    lines.append(f"Dwell: **{int(args.dwell)}**")
    lines.append("")
    lines.append("Climate counts:")
    for climate, count in sorted(by_climate.items()):
        lines.append(f"- `{climate}`: {count}")
    lines.append("")
    lines.append("Effective host counts:")
    for host, count in sorted(by_host.items()):
        lines.append(f"- `{host}`: {count}")
    lines.append("")
    lines.append("| Year | bull_ma200_v1 | buyhold | hf_host | lf_defensive_long_v2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    years = sorted({int(row["year"]) for row in rows})
    for year in years:
        lines.append(
            f"| {year} | "
            f"{by_year_host.get((year, 'bull_ma200_v1'), 0)} | "
            f"{by_year_host.get((year, 'buyhold'), 0)} | "
            f"{by_year_host.get((year, 'hf_host'), 0)} | "
            f"{by_year_host.get((year, 'lf_defensive_long_v2'), 0)} |"
        )
    fwd_21 = [float(row["fwd_21d_ret"]) for row in rows if row["fwd_21d_ret"] is not None]
    lines.append("")
    lines.append(f"Average `fwd_21d_ret`: **{mean(fwd_21):.4f}**")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(out_csv)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
