"""Search interpretable router policies against the distillation dataset, then exact-replay finalists."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from statistics import mean
from time import monotonic

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


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(raw).split(",") if str(part).strip())


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(float(part.strip())) for part in str(raw).split(",") if str(part).strip())


def _parse_year_weights(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for part in str(raw or "").split(","):
        item = str(part).strip()
        if not item:
            continue
        year, weight = item.split("=", 1)
        out[int(year.strip())] = float(weight.strip())
    return out


def _parse_year_values(raw: str) -> dict[int, float]:
    return _parse_year_weights(raw)


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


def _top_rules(
    rows: list[dict[str, str]],
    *,
    top_k: int,
    bull_slow_rv_mins: tuple[float, ...],
    bull_slow_eff_maxes: tuple[float, ...],
    bull_slow_dd_mins: tuple[float, ...],
    bull_fast_dd_mins: tuple[float, ...],
    bull_slow_up_frac_maxes: tuple[float, ...],
    bear_slow_ret_maxes: tuple[float, ...],
    bear_slow_dd_mins: tuple[float, ...],
    bear_slow_rv_mins: tuple[float, ...],
    bear_slow_eff_maxes: tuple[float, ...],
    bear_fast_ret_maxes: tuple[float, ...],
    bear_fast_dd_mins: tuple[float, ...],
) -> tuple[list[BullRule], list[BearRule]]:
    bull_rules: list[tuple[float, int, BullRule]] = []
    for slow_rv_min in bull_slow_rv_mins:
        for slow_eff_max in bull_slow_eff_maxes:
            for slow_dd_min in bull_slow_dd_mins:
                for fast_dd_min in bull_fast_dd_mins:
                    for slow_up_frac_max in bull_slow_up_frac_maxes:
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
    for slow_ret_max in bear_slow_ret_maxes:
        for slow_dd_min in bear_slow_dd_mins:
            for slow_rv_min in bear_slow_rv_mins:
                for slow_eff_max in bear_slow_eff_maxes:
                    for fast_ret_max in bear_fast_ret_maxes:
                        for fast_dd_min in bear_fast_dd_mins:
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
    preset: Path,
    bull_rule: BullRule,
    bear_rule: BearRule,
    fast_window: int,
    slow_window: int,
    dwell: int,
    years: tuple[int, ...],
) -> tuple[dict[int, float], float, float]:
    import tradebot.climate_router as cr

    orig_classify = cr.classify_rolling_climate_v5
    orig_bull_ok = cr.bull_sovereign_entry_ok

    def patched(*, crash_features, fast_features, slow_features, active=None, config=None):
        if bear_rule.matches(
            {
                "slow_ret": str(slow_features.ret),
                "slow_maxdd": str(slow_features.maxdd),
                "slow_rv": str(slow_features.rv),
                "slow_eff": str(slow_features.efficiency),
                "fast_ret": str(fast_features.ret),
                "fast_maxdd": str(fast_features.maxdd),
            }
        ):
            return ClimateDecision(climate="pre_bear_handoff", chosen_host="hf_host")
        return orig_classify(
            crash_features=crash_features,
            fast_features=fast_features,
            slow_features=slow_features,
            active=active,
            config=config,
        )

    def patched_bull_ok(*, climate, chosen_host, fast_features, slow_features):
        return bull_rule.matches(
            {
                "slow_rv": str(getattr(slow_features, "rv", 0.0)),
                "slow_eff": str(getattr(slow_features, "efficiency", 0.0)),
                "slow_maxdd": str(getattr(slow_features, "maxdd", 0.0)),
                "fast_maxdd": str(getattr(fast_features, "maxdd", 0.0)),
                "slow_up_frac": str(getattr(slow_features, "up_frac", 0.0)),
                "climate": str(climate or ""),
            }
        ) and str(chosen_host or "") == "buyhold"

    cr.classify_rolling_climate_v5 = patched
    cr.bull_sovereign_entry_ok = patched_bull_ok
    try:
        strategy, bar_size, use_rth = load_hf_host_strategy(preset)
        strategy = replace(
            strategy,
            regime_router=True,
            regime_router_fast_window_days=int(fast_window),
            regime_router_slow_window_days=int(slow_window),
            regime_router_min_dwell_days=int(dwell),
        )
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
        cr.classify_rolling_climate_v5 = orig_classify
        cr.bull_sovereign_entry_ok = orig_bull_ok


def _baseline_replay(*, preset: Path, fast_window: int, slow_window: int, dwell: int, years: tuple[int, ...]) -> dict[int, float]:
    strategy, bar_size, use_rth = load_hf_host_strategy(preset)
    strategy = replace(
        strategy,
        regime_router=True,
        regime_router_fast_window_days=int(fast_window),
        regime_router_slow_window_days=int(slow_window),
        regime_router_min_dwell_days=int(dwell),
    )
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
    return vals


def _objective_score(
    vals: dict[int, float],
    *,
    baseline: dict[int, float] | None,
    floor_target: float,
    floor_penalty: float,
    floor_weights: dict[int, float],
    preserve_penalty: float,
    preserve_weights: dict[int, float],
) -> tuple[float, int]:
    score = float(mean(vals.values()))
    below_floor = 0
    for year, value in vals.items():
        shortfall = max(0.0, float(floor_target) - float(value))
        if shortfall > 0.0:
            below_floor += 1
            score -= float(floor_penalty) * shortfall * float(floor_weights.get(int(year), 1.0))
    if baseline:
        for year, weight in preserve_weights.items():
            if int(year) not in vals or int(year) not in baseline:
                continue
            gap = max(0.0, float(baseline[int(year)]) - float(vals[int(year)]))
            score -= float(preserve_penalty) * gap * float(weight)
    return float(score), int(below_floor)


def _render_report(
    *,
    out_md: Path,
    baseline: dict[int, float],
    args,
    candidates: list[tuple[float, float, float, int, tuple[int, int, int], BullRule, BearRule, dict[int, float]]],
) -> None:
    ordered = sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out: list[str] = []
    out.append("# Regime Router Policy Search")
    out.append("")
    out.append("## Objective")
    out.append("")
    out.append(f"- floor target: `{float(args.floor_target):.3f}`")
    out.append(f"- floor penalty: `{float(args.floor_penalty):.3f}`")
    out.append(f"- floor weights: `{args.floor_weights}`")
    out.append(f"- preserve penalty: `{float(args.preserve_penalty):.3f}`")
    out.append(f"- preserve weights: `{args.preserve_weights}`")
    out.append(
        "- baseline: "
        + ", ".join(f"`{year}:{baseline[year]:.6f}`" for year in sorted(baseline))
    )
    out.append("")
    for rank, (objective, avg, worst, below_floor, windows, bull, bear, vals) in enumerate(ordered[:10], start=1):
        out.append(f"## Candidate {rank}")
        out.append("")
        out.append(f"- windows: `{windows[0]}/{windows[1]}/{windows[2]}`")
        out.append(
            f"- bull rule: `slow_rv>={bull.slow_rv_min} slow_eff<={bull.slow_eff_max} slow_dd>={bull.slow_dd_min} fast_dd>={bull.fast_dd_min} slow_up<={bull.slow_up_frac_max}`"
        )
        out.append(
            f"- bear rule: `slow_ret<={bear.slow_ret_max} slow_dd>={bear.slow_dd_min} slow_rv>={bear.slow_rv_min} slow_eff<={bear.slow_eff_max} fast_ret<={bear.fast_ret_max} fast_dd>={bear.fast_dd_min}`"
        )
        out.append(f"- objective: `{objective:.6f}`")
        out.append(f"- avg: `{avg:.6f}`")
        out.append(f"- worst: `{worst:.6f}`")
        out.append(f"- years below floor: `{below_floor}`")
        out.append("- years: " + ", ".join(f"`{year}:{vals[year]:.6f}`" for year in sorted(vals)))
        deltas = ", ".join(
            f"`{year}:{(vals[year] - baseline[year]):+.6f}`"
            for year in sorted(vals)
            if year in baseline
        )
        out.append(f"- vs baseline: {deltas}")
        out.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search interpretable router policies using the distillation dataset.")
    ap.add_argument("--csv", default="/tmp/tqqq_regime_router_distill.csv")
    ap.add_argument(
        "--preset",
        default="backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v46_routerOnCompositeContextConfidence_20260403.json",
    )
    ap.add_argument("--years", default="2017,2018,2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--fast-values", default="42,63,84")
    ap.add_argument("--slow-values", default="84,105,126")
    ap.add_argument("--dwell-values", default="5,10,15")
    ap.add_argument("--baseline-fast", type=int, default=63)
    ap.add_argument("--baseline-slow", type=int, default=84)
    ap.add_argument("--baseline-dwell", type=int, default=10)
    ap.add_argument("--baseline-values", default="")
    ap.add_argument("--floor-target", type=float, default=0.50)
    ap.add_argument("--floor-penalty", type=float, default=4.0)
    ap.add_argument("--floor-weights", default="2018=4,2022=4,2017=2,2020=2,2019=2,2025=3")
    ap.add_argument("--preserve-penalty", type=float, default=2.0)
    ap.add_argument("--preserve-weights", default="2019=2,2025=3")
    ap.add_argument("--bull-slow-rv-mins", default="0.50,0.55,0.60")
    ap.add_argument("--bull-slow-eff-maxes", default="0.05,0.08,0.10")
    ap.add_argument("--bull-slow-dd-mins", default="0.18,0.22,0.26,0.30")
    ap.add_argument("--bull-fast-dd-mins", default="0.10,0.14,0.18,0.22")
    ap.add_argument("--bull-slow-up-frac-maxes", default="0.54,0.56,0.58,0.60")
    ap.add_argument("--bear-slow-ret-maxes", default="0.00,0.05,0.10,0.15")
    ap.add_argument("--bear-slow-dd-mins", default="0.20,0.24,0.28,0.32")
    ap.add_argument("--bear-slow-rv-mins", default="0.50,0.55,0.60")
    ap.add_argument("--bear-slow-eff-maxes", default="0.05,0.08,0.10")
    ap.add_argument("--bear-fast-ret-maxes", default="-0.10,-0.05,0.00")
    ap.add_argument("--bear-fast-dd-mins", default="0.10,0.14,0.18,0.22")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_policy_search.md")
    args = ap.parse_args(argv)

    preset = Path(args.preset)
    years = tuple(int(part.strip()) for part in str(args.years).split(",") if str(part).strip())
    if not years:
        raise SystemExit("No --years specified")

    rows = _load_rows(Path(args.csv))
    bull_rules, bear_rules = _top_rules(
        rows,
        top_k=int(args.top_k),
        bull_slow_rv_mins=_parse_float_list(args.bull_slow_rv_mins),
        bull_slow_eff_maxes=_parse_float_list(args.bull_slow_eff_maxes),
        bull_slow_dd_mins=_parse_float_list(args.bull_slow_dd_mins),
        bull_fast_dd_mins=_parse_float_list(args.bull_fast_dd_mins),
        bull_slow_up_frac_maxes=_parse_float_list(args.bull_slow_up_frac_maxes),
        bear_slow_ret_maxes=_parse_float_list(args.bear_slow_ret_maxes),
        bear_slow_dd_mins=_parse_float_list(args.bear_slow_dd_mins),
        bear_slow_rv_mins=_parse_float_list(args.bear_slow_rv_mins),
        bear_slow_eff_maxes=_parse_float_list(args.bear_slow_eff_maxes),
        bear_fast_ret_maxes=_parse_float_list(args.bear_fast_ret_maxes),
        bear_fast_dd_mins=_parse_float_list(args.bear_fast_dd_mins),
    )
    timings = [
        (fast, slow, dwell)
        for fast in _parse_int_list(args.fast_values)
        for slow in _parse_int_list(args.slow_values)
        for dwell in _parse_int_list(args.dwell_values)
        if int(slow) >= int(fast)
    ]
    if str(args.baseline_values or "").strip():
        baseline = _parse_year_values(args.baseline_values)
        print(
            "[baseline] provided "
            + ", ".join(f"{year}:{baseline[year]:.6f}" for year in sorted(baseline)),
            flush=True,
        )
    else:
        baseline_started = monotonic()
        print(
            f"[baseline] start windows={int(args.baseline_fast)}/{int(args.baseline_slow)}/{int(args.baseline_dwell)}",
            flush=True,
        )
        baseline = _baseline_replay(
            preset=preset,
            fast_window=int(args.baseline_fast),
            slow_window=int(args.baseline_slow),
            dwell=int(args.baseline_dwell),
            years=years,
        )
        baseline_elapsed = monotonic() - baseline_started
        print(
            "[baseline] done "
            + ", ".join(f"{year}:{baseline[year]:.6f}" for year in sorted(baseline))
            + f" elapsed={baseline_elapsed:.1f}s",
            flush=True,
        )
    floor_weights = _parse_year_weights(args.floor_weights)
    preserve_weights = _parse_year_weights(args.preserve_weights)

    candidates: list[tuple[float, float, float, int, tuple[int, int, int], BullRule, BearRule, dict[int, float]]] = []
    total = max(1, len(timings) * len(bull_rules) * len(bear_rules))
    progress_every = max(1, int(args.progress_every))
    started = monotonic()
    seen = 0
    for windows in timings:
        for bull in bull_rules:
            for bear in bear_rules:
                seen += 1
                vals, avg, worst = _exact_replay(
                    preset=preset,
                    bull_rule=bull,
                    bear_rule=bear,
                    fast_window=int(windows[0]),
                    slow_window=int(windows[1]),
                    dwell=int(windows[2]),
                    years=years,
                )
                objective, below_floor = _objective_score(
                    vals,
                    baseline=baseline,
                    floor_target=float(args.floor_target),
                    floor_penalty=float(args.floor_penalty),
                    floor_weights=floor_weights,
                    preserve_penalty=float(args.preserve_penalty),
                    preserve_weights=preserve_weights,
                )
                candidates.append((objective, avg, worst, below_floor, windows, bull, bear, vals))
                if seen == 1 or seen % progress_every == 0 or seen == total:
                    elapsed = max(1e-9, monotonic() - started)
                    eta = (elapsed / float(seen)) * float(max(0, total - seen))
                    best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
                    print(
                        f"[{seen}/{total}] windows={windows[0]}/{windows[1]}/{windows[2]} "
                        f"obj={objective:.6f} avg={avg:.6f} worst={worst:.6f} below={below_floor} "
                        f"best_obj={best[0]:.6f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                    _render_report(
                        out_md=Path(args.out_md),
                        baseline=baseline,
                        args=args,
                        candidates=candidates,
                    )
    out_md = Path(args.out_md)
    _render_report(out_md=out_md, baseline=baseline, args=args, candidates=candidates)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
