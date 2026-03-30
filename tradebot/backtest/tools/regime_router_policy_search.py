"""Search interpretable router policies against the distillation dataset, then exact-replay finalists."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from statistics import mean

from ...climate_router import ClimateDecision, load_hf_host_strategy
from ...backtest.engine import run_backtest
from ...backtest.spot_codec import make_bundle, metrics_from_summary


@dataclass(frozen=True)
class BullRule:
    slow_rv_min: float
    slow_eff_max: float
    slow_dd_min: float
    fast_dd_min: float
    slow_up_frac_max: float

    def matches(self, row: dict[str, str]) -> bool:
        return (
            row.get("climate") == "bull_grind_low_vol"
            and float(row["slow_rv"]) >= float(self.slow_rv_min)
            and float(row["slow_eff"]) <= float(self.slow_eff_max)
            and float(row["slow_maxdd"]) >= float(self.slow_dd_min)
            and float(row["fast_maxdd"]) >= float(self.fast_dd_min)
            and float(row["slow_up_frac"]) <= float(self.slow_up_frac_max)
        )


@dataclass(frozen=True)
class BearRule:
    slow_ret_max: float
    slow_dd_min: float
    slow_rv_min: float
    slow_eff_max: float
    fast_ret_max: float
    fast_dd_min: float

    def matches(self, row: dict[str, str]) -> bool:
        return (
            float(row["slow_ret"]) <= float(self.slow_ret_max)
            and float(row["slow_maxdd"]) >= float(self.slow_dd_min)
            and float(row["slow_rv"]) >= float(self.slow_rv_min)
            and float(row["slow_eff"]) <= float(self.slow_eff_max)
            and float(row["fast_ret"]) <= float(self.fast_ret_max)
            and float(row["fast_maxdd"]) >= float(self.fast_dd_min)
        )


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _score_bull_rule(rows: list[dict[str, str]], rule: BullRule) -> tuple[float, int]:
    pos: list[float] = []
    neg: list[float] = []
    for row in rows:
        if row.get("climate") != "bull_grind_low_vol":
            continue
        raw = row.get("bull_ma200_adv_21d")
        if raw in (None, ""):
            continue
        value = float(raw)
        (pos if rule.matches(row) else neg).append(value)
    if len(pos) < 20 or len(neg) < 20:
        return -1e9, 0
    return mean(pos) - mean(neg), len(pos)


def _score_bear_rule(rows: list[dict[str, str]], rule: BearRule) -> tuple[float, int]:
    toxic: list[float] = []
    safe: list[float] = []
    for row in rows:
        if row.get("climate") != "bull_grind_low_vol":
            continue
        raw = row.get("fwd_21d_ret")
        if raw in (None, ""):
            continue
        value = float(raw)
        (toxic if rule.matches(row) else safe).append(value)
    if len(toxic) < 20 or len(safe) < 20:
        return -1e9, 0
    return mean(safe) - mean(toxic), len(toxic)


def _top_rules(rows: list[dict[str, str]], *, top_k: int) -> tuple[list[BullRule], list[BearRule]]:
    bull_rules: list[tuple[float, int, BullRule]] = []
    for slow_rv_min in (0.50, 0.55, 0.60):
        for slow_eff_max in (0.05, 0.08, 0.10, 0.12):
            for slow_dd_min in (0.18, 0.22, 0.26, 0.30):
                for fast_dd_min in (0.10, 0.14, 0.18, 0.22):
                    for slow_up_frac_max in (0.54, 0.56, 0.58, 0.60):
                        rule = BullRule(
                            slow_rv_min=float(slow_rv_min),
                            slow_eff_max=float(slow_eff_max),
                            slow_dd_min=float(slow_dd_min),
                            fast_dd_min=float(fast_dd_min),
                            slow_up_frac_max=float(slow_up_frac_max),
                        )
                        score, support = _score_bull_rule(rows, rule)
                        bull_rules.append((score, support, rule))
    bear_rules: list[tuple[float, int, BearRule]] = []
    for slow_ret_max in (0.00, 0.05, 0.10, 0.15):
        for slow_dd_min in (0.20, 0.24, 0.28, 0.32):
            for slow_rv_min in (0.50, 0.55, 0.60):
                for slow_eff_max in (0.05, 0.08, 0.10, 0.12):
                    for fast_ret_max in (-0.10, -0.05, 0.00):
                        for fast_dd_min in (0.10, 0.14, 0.18, 0.22):
                            rule = BearRule(
                                slow_ret_max=float(slow_ret_max),
                                slow_dd_min=float(slow_dd_min),
                                slow_rv_min=float(slow_rv_min),
                                slow_eff_max=float(slow_eff_max),
                                fast_ret_max=float(fast_ret_max),
                                fast_dd_min=float(fast_dd_min),
                            )
                            score, support = _score_bear_rule(rows, rule)
                            bear_rules.append((score, support, rule))
    bull_rules.sort(key=lambda item: (item[0], item[1]), reverse=True)
    bear_rules.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [rule for _score, _support, rule in bull_rules[:top_k]], [rule for _score, _support, rule in bear_rules[:top_k]]


def _exact_replay(
    *,
    bull_rule: BullRule,
    bear_rule: BearRule,
    fast_window: int,
    slow_window: int,
    dwell: int,
) -> tuple[dict[int, float], float, float]:
    import tradebot.climate_router as cr

    orig = cr.classify_rolling_climate_v5

    def patched(*, crash_features, mid_features, fast_features, slow_features, active=None):
        if bear_rule.matches(
            {
                "slow_ret": str(slow_features.ret),
                "slow_maxdd": str(slow_features.maxdd),
                "slow_rv": str(slow_features.rv),
                "slow_eff": str(slow_features.efficiency),
                "fast_ret": str(fast_features.ret),
                "fast_maxdd": str(fast_features.maxdd),
                "climate": "",
            }
        ):
            return ClimateDecision(climate="pre_bear_handoff", chosen_host="hf_host")
        out = orig(
            crash_features=crash_features,
            mid_features=mid_features,
            fast_features=fast_features,
            slow_features=slow_features,
            active=active,
        )
        if out.chosen_host == "buyhold" and bull_rule.matches(
            {
                "slow_rv": str(slow_features.rv),
                "slow_eff": str(slow_features.efficiency),
                "slow_maxdd": str(slow_features.maxdd),
                "fast_maxdd": str(fast_features.maxdd),
                "slow_up_frac": str(slow_features.up_frac),
                "climate": "bull_grind_low_vol",
            }
        ):
            return ClimateDecision(climate="messy_bull_recovery", chosen_host="bull_ma200_v1")
        return out

    cr.classify_rolling_climate_v5 = patched
    try:
        preset = Path("backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json")
        strategy, bar_size, use_rth = load_hf_host_strategy(preset)
        strategy = replace(
            strategy,
            regime_router=True,
            regime_router_fast_window_days=int(fast_window),
            regime_router_slow_window_days=int(slow_window),
            regime_router_min_dwell_days=int(dwell),
        )
        years = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
        vals: dict[int, float] = {}
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
            vals[int(year)] = float(metrics_from_summary(res.summary)["pnl_over_dd"])
        avg = mean(vals.values())
        worst = min(vals.values())
        return vals, float(avg), float(worst)
    finally:
        cr.classify_rolling_climate_v5 = orig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search interpretable router policies using the distillation dataset.")
    ap.add_argument("--csv", default="/tmp/tqqq_regime_router_distill.csv")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_policy_search.md")
    args = ap.parse_args(argv)

    rows = _load_rows(Path(args.csv))
    bull_rules, bear_rules = _top_rules(rows, top_k=int(args.top_k))
    timing = [
        (63, 126, 10),
        (42, 126, 10),
        (28, 126, 10),
        (63, 84, 10),
    ]

    candidates: list[tuple[float, float, tuple[int, int, int], BullRule, BearRule, dict[int, float]]] = []
    for windows in timing:
        for bull in bull_rules:
            for bear in bear_rules:
                vals, avg, worst = _exact_replay(
                    bull_rule=bull,
                    bear_rule=bear,
                    fast_window=int(windows[0]),
                    slow_window=int(windows[1]),
                    dwell=int(windows[2]),
                )
                candidates.append((avg, worst, windows, bull, bear, vals))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[str] = []
    out.append("# Regime Router Policy Search")
    out.append("")
    for rank, (avg, worst, windows, bull, bear, vals) in enumerate(candidates[:10], start=1):
        out.append(f"## Candidate {rank}")
        out.append("")
        out.append(f"- windows: `{windows[0]}/{windows[1]}/{windows[2]}`")
        out.append(
            f"- bull rule: `slow_rv>={bull.slow_rv_min} slow_eff<={bull.slow_eff_max} slow_dd>={bull.slow_dd_min} fast_dd>={bull.fast_dd_min} slow_up<={bull.slow_up_frac_max}`"
        )
        out.append(
            f"- bear rule: `slow_ret<={bear.slow_ret_max} slow_dd>={bear.slow_dd_min} slow_rv>={bear.slow_rv_min} slow_eff<={bear.slow_eff_max} fast_ret<={bear.fast_ret_max} fast_dd>={bear.fast_dd_min}`"
        )
        out.append(f"- avg: `{avg:.6f}`")
        out.append(f"- worst: `{worst:.6f}`")
        out.append(
            "- years: "
            + ", ".join(f"`{year}:{vals[year]:.6f}`" for year in sorted(vals))
        )
        out.append("")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(out) + "\n")
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
