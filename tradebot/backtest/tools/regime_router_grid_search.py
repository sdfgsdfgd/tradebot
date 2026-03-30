"""Exact combinatorial sweep for regime-router timing and host-detection modes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from statistics import mean

from ...climate_router import ClimateDecision, load_hf_host_strategy
from ...backtest.spot_codec import make_bundle, metrics_from_summary
from ...backtest.engine import run_backtest


@dataclass(frozen=True)
class TimingProfile:
    name: str
    fast: int
    slow: int
    dwell: int


TIMING_PRESETS: dict[str, TimingProfile] = {
    "base": TimingProfile(name="base", fast=63, slow=126, dwell=10),
    "midfast": TimingProfile(name="midfast", fast=42, slow=126, dwell=10),
    "fast": TimingProfile(name="fast", fast=28, slow=126, dwell=3),
}


def _apply_bear_mode(*, mode: str, crash_features, mid_features, fast_features, slow_features):
    if mode == "current":
        return None
    if mode == "early_hf_r1":
        if (
            float(mid_features.ret) <= -0.10
            and float(mid_features.maxdd) >= 0.24
            and float(mid_features.rv) >= 0.65
            and float(fast_features.ret) < 0.0
            and float(slow_features.rv) >= 0.50
        ):
            return ClimateDecision(climate="early_crash_sentinel", chosen_host="hf_host")
        return None
    raise SystemExit(f"Unknown bear mode: {mode}")


def _apply_bull_mode(*, mode: str, out: ClimateDecision, fast_features, slow_features):
    if str(out.chosen_host) != "buyhold":
        return out
    if mode == "current":
        return out
    if mode == "off":
        return out
    if mode == "all_ma200":
        return ClimateDecision(climate=str(out.climate), chosen_host="bull_ma200_v1")
    raise SystemExit(f"Unknown bull mode: {mode}")


def _patch_router(*, bull_mode: str, bear_mode: str):
    import tradebot.climate_router as cr

    orig_classify = cr.classify_rolling_climate_v5
    orig_bull_ok = cr.bull_sovereign_entry_ok

    def patched_classify(*, crash_features, fast_features, slow_features, active=None):
        forced = _apply_bear_mode(
            mode=str(bear_mode),
            crash_features=crash_features,
            mid_features=fast_features,
            fast_features=fast_features,
            slow_features=slow_features,
        )
        if forced is not None:
            return forced
        out = orig_classify(
            crash_features=crash_features,
            fast_features=fast_features,
            slow_features=slow_features,
            active=active,
        )
        return _apply_bull_mode(
            mode=str(bull_mode),
            out=out,
            fast_features=fast_features,
            slow_features=slow_features,
        )

    def patched_bull_ok(*, climate, chosen_host, fast_features, slow_features):
        if str(bull_mode) == "off":
            return False
        if str(bull_mode) == "all_ma200":
            return str(climate or "") == "bull_grind_low_vol" and str(chosen_host or "") == "buyhold"
        return orig_bull_ok(
            climate=climate,
            chosen_host=chosen_host,
            fast_features=fast_features,
            slow_features=slow_features,
        )

    cr.classify_rolling_climate_v5 = patched_classify
    cr.bull_sovereign_entry_ok = patched_bull_ok
    return cr, orig_classify, orig_bull_ok


def _restore_router(cr, orig_classify, orig_bull_ok) -> None:
    cr.classify_rolling_climate_v5 = orig_classify
    cr.bull_sovereign_entry_ok = orig_bull_ok


def _replay_candidate(*, timing: TimingProfile, bull_mode: str, bear_mode: str, years: tuple[int, ...]) -> dict[int, float]:
    cr, orig_classify, orig_bull_ok = _patch_router(bull_mode=bull_mode, bear_mode=bear_mode)
    try:
        preset = Path("backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json")
        strategy, bar_size, use_rth = load_hf_host_strategy(preset)
        strategy = replace(
            strategy,
            regime_router=True,
            regime_router_fast_window_days=int(timing.fast),
            regime_router_slow_window_days=int(timing.slow),
            regime_router_min_dwell_days=int(timing.dwell),
        )
        out: dict[int, float] = {}
        for year in years:
            cfg = make_bundle(
                strategy=strategy,
                start=date(year, 1, 1),
                end=date(year + 1, 1, 1),
                bar_size=bar_size,
                use_rth=use_rth,
                cache_dir="db",
                offline=True,
            )
            res = run_backtest(cfg)
            out[int(year)] = float(metrics_from_summary(res.summary)["pnl_over_dd"])
        return out
    finally:
        _restore_router(cr, orig_classify, orig_bull_ok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Exact grid search for regime-router timing and bull/bear modes.")
    ap.add_argument("--timings", nargs="*", default=["base", "midfast"])
    ap.add_argument("--bull-modes", nargs="*", default=["current", "all_ma200"])
    ap.add_argument("--bear-modes", nargs="*", default=["current", "early_hf_r1"])
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_grid_search.md")
    args = ap.parse_args(argv)

    years = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
    rows: list[tuple[float, float, str, str, str, dict[int, float]]] = []
    for timing_name in args.timings:
        timing = TIMING_PRESETS[str(timing_name)]
        for bull_mode in args.bull_modes:
            for bear_mode in args.bear_modes:
                vals = _replay_candidate(
                    timing=timing,
                    bull_mode=str(bull_mode),
                    bear_mode=str(bear_mode),
                    years=years,
                )
                avg = mean(vals.values())
                worst = min(vals.values())
                rows.append((float(avg), float(worst), timing.name, str(bull_mode), str(bear_mode), vals))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    lines: list[str] = []
    lines.append("# Regime Router Grid Search")
    lines.append("")
    for rank, (avg, worst, timing, bull_mode, bear_mode, vals) in enumerate(rows, start=1):
        lines.append(f"## Candidate {rank}")
        lines.append("")
        lines.append(f"- timing: `{timing}`")
        lines.append(f"- bull mode: `{bull_mode}`")
        lines.append(f"- bear mode: `{bear_mode}`")
        lines.append(f"- avg: `{avg:.6f}`")
        lines.append(f"- worst: `{worst:.6f}`")
        lines.append("- years: " + ", ".join(f"`{year}:{vals[year]:.6f}`" for year in years))
        lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
