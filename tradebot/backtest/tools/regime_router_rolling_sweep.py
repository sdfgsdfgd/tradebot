"""Rolling 1Y regime-router evaluation and tight sweeps (bad-window focused).

This tool is analysis-first:
- find the worst rolling windows (by pnl/dd) across the last N years
- print regime-router host/climate composition for those windows
- run a tight fast/slow/dwell sweep against an evaluation set

Notes:
- Uses the real backtest engine (offline).
- Windowing uses trading days derived from the daily bars file.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from ...climate_router import (
    DailyRegimeRouterEngine,
    RegimeRouterConfig,
    load_daily_bars_from_intraday_csv,
    load_hf_host_strategy,
)
from ..engine import run_backtest
from ..models import SpotTrade
from ..spot_codec import make_bundle, metrics_from_summary


@dataclass(frozen=True)
class WindowSpec:
    start: date
    end: date

    @property
    def label(self) -> str:
        return f"{self.start.isoformat()}→{self.end.isoformat()}"


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(float(part.strip())) for part in str(raw or "").split(",") if str(part).strip())


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _window_years(start: date, end: date) -> float:
    return max((end - start).days, 1) / 365.25


def _rolling_windows(
    daily_ts: list[str],
    *,
    range_start: date,
    range_end: date,
    window_trading_days: int,
    step_trading_days: int,
) -> list[WindowSpec]:
    # daily_ts is a list of trading-day strings "YYYY-MM-DD" sorted asc.
    start_idx = 0
    end_idx = len(daily_ts)
    while start_idx < end_idx and _parse_date(daily_ts[start_idx]) < range_start:
        start_idx += 1
    while end_idx > start_idx and _parse_date(daily_ts[end_idx - 1]) > range_end:
        end_idx -= 1

    needed = max(2, int(window_trading_days))
    step = max(1, int(step_trading_days))
    out: list[WindowSpec] = []
    for idx in range(start_idx, max(start_idx, end_idx - needed + 1), step):
        end_i = idx + needed - 1
        if end_i >= end_idx:
            break
        start_d = _parse_date(daily_ts[idx])
        end_d = _parse_date(daily_ts[end_i])
        if start_d < range_start or end_d > range_end:
            continue
        out.append(WindowSpec(start=start_d, end=end_d))
    return out


def _router_day_states(
    daily_bars,
    *,
    cfg: RegimeRouterConfig,
) -> dict[str, tuple[str | None, str | None, bool, bool]]:
    """Per-day router state.

    Mapping: day_ts -> (climate, chosen_host, bull_sovereign_ok, ready)
    """

    engine = DailyRegimeRouterEngine(config=cfg)
    states: dict[str, tuple[str | None, str | None, bool, bool]] = {}

    prev_day: str | None = None
    for bar in daily_bars:
        snap = engine.update_bar(
            ts=str(bar.ts),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            hf_entry_dir=None,
        )
        # `snap` reflects the finalized state of prev_day (because day change finalizes prev_day).
        if prev_day is not None:
            states[prev_day] = (snap.climate, snap.chosen_host, bool(snap.bull_sovereign_ok), bool(snap.ready))
        prev_day = str(bar.ts)

    if prev_day is not None:
        # Force-finalize the last trading day with a dummy next-day bar.
        dummy = _parse_date(prev_day) + timedelta(days=1)
        snap = engine.update_bar(
            ts=dummy.isoformat(),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            hf_entry_dir=None,
        )
        states[prev_day] = (snap.climate, snap.chosen_host, bool(snap.bull_sovereign_ok), bool(snap.ready))
    return states


def _window_router_summary(
    window: WindowSpec,
    *,
    day_states: dict[str, tuple[str | None, str | None, bool, bool]],
) -> dict[str, object]:
    days = [ts for ts in sorted(day_states.keys()) if window.start <= _parse_date(ts) <= window.end]
    if not days:
        return {"days": 0, "ready_days": 0, "host_counts": {}, "climate_counts": {}, "switches": 0, "segments": []}

    host_counts: Counter[str] = Counter()
    climate_counts: Counter[str] = Counter()
    ready_days = 0
    switches = 0
    segments: list[tuple[str, str, str, str, int]] = []

    last_key: tuple[str, str, bool, bool] | None = None
    for ts in days:
        climate, host, bull_ok, ready = day_states.get(ts, (None, None, False, False))
        if bool(ready):
            ready_days += 1
        host_s = str(host or "none")
        climate_s = str(climate or "none")
        host_counts[host_s] += 1
        climate_counts[climate_s] += 1

        key = (climate_s, host_s, bool(bull_ok), bool(ready))
        if last_key is None:
            last_key = key
            segments.append((ts, ts, f"{host_s}{'*' if bull_ok else ''}", climate_s, 1))
            continue
        if key == last_key:
            seg_start, _seg_end, seg_host, seg_climate, seg_n = segments[-1]
            segments[-1] = (seg_start, ts, seg_host, seg_climate, int(seg_n) + 1)
            continue
        switches += 1
        last_key = key
        segments.append((ts, ts, f"{host_s}{'*' if bull_ok else ''}", climate_s, 1))

    return {
        "days": int(len(days)),
        "ready_days": int(ready_days),
        "host_counts": dict(host_counts),
        "climate_counts": dict(climate_counts),
        "switches": int(switches),
        "segments": segments,
    }


def _window_overlay_summary(trades: list[SpotTrade]) -> dict[str, object]:
    closed = [t for t in trades if t.exit_time is not None]
    denom = max(1, len(closed))

    stop_n = 0
    flip_n = 0
    riskoff_n = 0
    riskpanic_n = 0
    shock_n = 0
    ddboost_n = 0

    for t in closed:
        reason = str(t.exit_reason or "").strip().lower()
        if reason == "stop_loss_pct":
            stop_n += 1
        elif reason == "flip":
            flip_n += 1

        dt = t.decision_trace if isinstance(t.decision_trace, dict) else {}
        if bool(dt.get("riskoff")):
            riskoff_n += 1
        if bool(dt.get("riskpanic")):
            riskpanic_n += 1
        if bool(dt.get("shock")):
            shock_n += 1
        if bool(dt.get("shock_short_boost_applied")):
            ddboost_n += 1

    return {
        "trades_closed": int(len(closed)),
        "stop_pct": float(stop_n) / float(denom),
        "flip_pct": float(flip_n) / float(denom),
        "riskoff_pct": float(riskoff_n) / float(denom),
        "riskpanic_pct": float(riskpanic_n) / float(denom),
        "shock_pct": float(shock_n) / float(denom),
        "ddboost_pct": float(ddboost_n) / float(denom),
    }


def _evaluate_window(*, strategy, bar_size: str, use_rth: bool, window: WindowSpec) -> dict[str, object]:
    cfg = make_bundle(
        strategy=strategy,
        start=window.start,
        end=window.end,
        bar_size=bar_size,
        use_rth=use_rth,
        cache_dir=Path("db"),
        offline=True,
    )
    res = run_backtest(cfg)
    m = metrics_from_summary(res.summary)
    trades = [t for t in res.trades if isinstance(t, SpotTrade)]
    overlay = _window_overlay_summary(trades)
    return {
        "window": window,
        "pdd": float(m.get("pnl_over_dd") or 0.0),
        "pnl": float(m.get("pnl") or 0.0),
        "dd": float(m.get("dd") or 0.0),
        "dd_pct": float(m.get("dd_pct") or 0.0),
        "trades": int(m.get("trades") or 0),
        "win": float(m.get("win_rate") or 0.0),
        "overlay": overlay,
    }


def _objective(
    rows: list[dict[str, object]],
    *,
    neg_penalty: float,
    min_trades_per_year: float,
) -> tuple[float, dict[str, float]]:
    vals = [float(r["pdd"]) for r in rows]
    avg = float(mean(vals)) if vals else float("-inf")
    worst = float(min(vals)) if vals else float("inf")
    neg_mass = sum(max(0.0, -float(v)) for v in vals)

    trade_shortfall = 0.0
    for r in rows:
        w: WindowSpec = r["window"]
        years = _window_years(w.start, w.end)
        min_trades = float(min_trades_per_year) * float(years)
        trade_shortfall += max(0.0, float(min_trades) - float(r.get("trades") or 0))

    score = avg - (float(neg_penalty) * float(neg_mass)) - (0.0001 * float(trade_shortfall))
    return score, {"avg": avg, "worst": worst, "neg_mass": float(neg_mass), "trade_shortfall": float(trade_shortfall)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rolling 1Y regime-router bad-window eval + tight sweeps.")
    ap.add_argument(
        "--preset",
        default="backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v43_compositeContextConfidence_20260319.json",
        help="HF-host preset JSON (seed strategy).",
    )
    ap.add_argument("--intraday-csv", default="db/TQQQ/TQQQ_2016-01-01_2026-01-19_1min_rth.csv")
    ap.add_argument("--range-start", default="2016-01-01")
    ap.add_argument("--range-end", default="2025-12-31")
    ap.add_argument("--window-trading-days", type=int, default=252)
    ap.add_argument("--step-trading-days", type=int, default=21)
    ap.add_argument("--worst-k", type=int, default=12)
    ap.add_argument("--best-k", type=int, default=6)
    ap.add_argument("--sample-k", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=5)
    ap.add_argument("--out-md", default="/tmp/tqqq_regime_router_rolling_sweep.md")

    ap.add_argument("--fast-values", default="42,63,84")
    ap.add_argument("--slow-values", default="84,105,126")
    ap.add_argument("--dwell-values", default="5,10,15")
    ap.add_argument("--bull-on-values", default="3")
    ap.add_argument("--bull-off-values", default="7")
    ap.add_argument("--neg-penalty", type=float, default=0.5)
    ap.add_argument("--min-trades-per-year", type=float, default=50.0)
    ap.add_argument("--no-sweep", action="store_true", default=False)
    args = ap.parse_args(argv)

    preset = Path(args.preset)
    base_strategy, bar_size, use_rth = load_hf_host_strategy(preset)

    range_start = _parse_date(args.range_start)
    range_end = _parse_date(args.range_end)
    if range_end <= range_start:
        raise SystemExit(f"Invalid range: {range_start}..{range_end}")

    daily_bars = load_daily_bars_from_intraday_csv(Path(args.intraday_csv))
    daily_ts = [str(b.ts) for b in daily_bars]
    windows = _rolling_windows(
        daily_ts,
        range_start=range_start,
        range_end=range_end,
        window_trading_days=int(args.window_trading_days),
        step_trading_days=int(args.step_trading_days),
    )
    if not windows:
        raise SystemExit("No windows generated (check range/step/window args)")

    baseline_strategy = replace(
        base_strategy,
        regime_router=True,
        regime_router_fast_window_days=63,
        regime_router_slow_window_days=84,
        regime_router_min_dwell_days=10,
        regime_router_bull_sovereign_on_confirm_days=1,
        regime_router_bull_sovereign_off_confirm_days=7,
    )
    baseline_router_cfg = RegimeRouterConfig(
        enabled=True,
        fast_window_days=63,
        slow_window_days=84,
        min_dwell_days=10,
        bull_sovereign_on_confirm_days=1,
        bull_sovereign_off_confirm_days=7,
        crash_hf_slow_ret_max=float(getattr(baseline_strategy, "regime_router_crash_hf_slow_ret_max", -0.25) or -0.25),
        damage_positive_lock_maxdd_min=float(getattr(baseline_strategy, "regime_router_damage_positive_lock_maxdd_min", 0.24) or 0.24),
        damage_positive_lock_ret_max=float(getattr(baseline_strategy, "regime_router_damage_positive_lock_ret_max", 0.20) or 0.20),
        damage_positive_lock_eff_max=float(getattr(baseline_strategy, "regime_router_damage_positive_lock_eff_max", 0.10) or 0.10),
    )
    day_states = _router_day_states(daily_bars, cfg=baseline_router_cfg)

    baseline_rows: list[dict[str, object]] = []
    progress_every = max(1, int(args.progress_every))
    for idx, window in enumerate(windows, start=1):
        row = _evaluate_window(strategy=baseline_strategy, bar_size=bar_size, use_rth=use_rth, window=window)
        row["router"] = _window_router_summary(window, day_states=day_states)
        baseline_rows.append(row)
        if idx == 1 or idx % progress_every == 0 or idx == len(windows):
            print(f"[baseline {idx}/{len(windows)}] {window.label} pdd={float(row['pdd']):+.3f} tr={int(row['trades'])}", flush=True)

    baseline_rows_sorted = sorted(baseline_rows, key=lambda r: float(r["pdd"]))
    worst = baseline_rows_sorted[: max(1, int(args.worst_k))]
    best = sorted(baseline_rows, key=lambda r: float(r["pdd"]), reverse=True)[: max(0, int(args.best_k))]
    sampled: list[dict[str, object]] = []
    sample_k = max(0, int(args.sample_k))
    if sample_k > 0:
        stride = max(1, int(len(windows) // max(1, sample_k)))
        by_start = sorted(baseline_rows, key=lambda r: (r["window"].start, r["window"].end))
        sampled = [by_start[i] for i in range(0, len(by_start), stride)][:sample_k]

    eval_map: dict[str, dict[str, object]] = {}
    for row in worst + best + sampled:
        w: WindowSpec = row["window"]
        eval_map[w.label] = row
    eval_rows = list(eval_map.values())
    eval_windows = [r["window"] for r in eval_rows]

    baseline_score, baseline_obj = _objective(
        eval_rows,
        neg_penalty=float(args.neg_penalty),
        min_trades_per_year=float(args.min_trades_per_year),
    )
    print(
        "[eval-set] "
        f"windows={len(eval_windows)} worst_k={int(args.worst_k)} best_k={int(args.best_k)} sample_k={int(args.sample_k)} "
        f"baseline_score={baseline_score:.6f} avg={baseline_obj['avg']:+.3f} worst={baseline_obj['worst']:+.3f} neg_mass={baseline_obj['neg_mass']:.3f}",
        flush=True,
    )

    lines: list[str] = []
    lines.append("# Regime Router Rolling Sweep")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- preset: `{preset}`")
    lines.append(f"- range: `{range_start.isoformat()} -> {range_end.isoformat()}`")
    lines.append(
        f"- windows: trading_days={int(args.window_trading_days)} step={int(args.step_trading_days)} count={len(windows)}"
    )
    lines.append("- baseline router: fast/slow/dwell=`63/84/10` bull_on/off=`1/7`")
    lines.append("")
    lines.append("## Worst Windows (baseline)")
    lines.append("")
    lines.append(
        "| Rank | Window | PnL/DD | Trades | Win | PnL | DD | DD% | Host Mix | Climate Mix | Shock | Riskpanic | Stop% | Flip% | Switches |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(worst[:20], start=1):
        w: WindowSpec = row["window"]
        router = row.get("router") or {}
        overlay = row.get("overlay") or {}
        hosts = router.get("host_counts") or {}
        climates = router.get("climate_counts") or {}
        host_mix = ", ".join(
            f"{k}:{int(v)}" for k, v in sorted(hosts.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]
        )
        climate_mix = ", ".join(
            f"{k}:{int(v)}" for k, v in sorted(climates.items(), key=lambda kv: int(kv[1]), reverse=True)[:3]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    w.label,
                    f"{float(row['pdd']):+.3f}",
                    str(int(row["trades"])),
                    _fmt_pct(float(row["win"])),
                    f"{float(row['pnl']):.1f}",
                    f"{float(row['dd']):.1f}",
                    _fmt_pct(float(row["dd_pct"])),
                    host_mix,
                    climate_mix,
                    _fmt_pct(float(overlay.get("shock_pct") or 0.0)),
                    _fmt_pct(float(overlay.get("riskpanic_pct") or 0.0)),
                    _fmt_pct(float(overlay.get("stop_pct") or 0.0)),
                    _fmt_pct(float(overlay.get("flip_pct") or 0.0)),
                    str(int(router.get("switches") or 0)),
                ]
            )
            + " |"
        )
        segments = router.get("segments") or []
        if segments:
            lines.append(
                "  - segments: "
                + ", ".join(f"`{s[0]}..{s[1]} {s[2]} {s[3]} ({s[4]}d)`" for s in segments[:6])
            )
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md, flush=True)

    if bool(args.no_sweep):
        return 0

    fast_values = _parse_int_list(args.fast_values)
    slow_values = _parse_int_list(args.slow_values)
    dwell_values = _parse_int_list(args.dwell_values)
    bull_on_values = _parse_int_list(args.bull_on_values)
    bull_off_values = _parse_int_list(args.bull_off_values)
    candidates = [
        (fast, slow, dwell, bull_on, bull_off)
        for fast in fast_values
        for slow in slow_values
        for dwell in dwell_values
        for bull_on in bull_on_values
        for bull_off in bull_off_values
        if int(slow) >= int(fast)
    ]
    if not candidates:
        raise SystemExit("No sweep candidates generated (check --fast/--slow/--dwell values)")

    print(f"[sweep] candidates={len(candidates)} eval_windows={len(eval_windows)}", flush=True)
    best_score = float("-inf")
    best_cfg: dict[str, object] | None = None

    for idx, (fast, slow, dwell, bull_on, bull_off) in enumerate(candidates, start=1):
        strat = replace(
            base_strategy,
            regime_router=True,
            regime_router_fast_window_days=int(fast),
            regime_router_slow_window_days=int(slow),
            regime_router_min_dwell_days=int(dwell),
            regime_router_bull_sovereign_on_confirm_days=int(bull_on),
            regime_router_bull_sovereign_off_confirm_days=int(bull_off),
        )
        rows = [_evaluate_window(strategy=strat, bar_size=bar_size, use_rth=use_rth, window=w) for w in eval_windows]
        score, obj = _objective(
            rows,
            neg_penalty=float(args.neg_penalty),
            min_trades_per_year=float(args.min_trades_per_year),
        )
        if float(score) > float(best_score):
            best_score = float(score)
            best_cfg = {
                "fast": int(fast),
                "slow": int(slow),
                "dwell": int(dwell),
                "bull_on": int(bull_on),
                "bull_off": int(bull_off),
                "score": float(score),
                "avg": float(obj["avg"]),
                "worst": float(obj["worst"]),
                "neg_mass": float(obj["neg_mass"]),
            }

        if idx == 1 or idx % progress_every == 0 or idx == len(candidates):
            best_s = best_cfg or {}
            print(
                f"[{idx}/{len(candidates)}] {fast}/{slow}/{dwell} bull={bull_on}/{bull_off} "
                f"score={score:.6f} avg={obj['avg']:+.3f} worst={obj['worst']:+.3f} neg={obj['neg_mass']:.3f} "
                f"best={best_s.get('score', float('-inf')):.6f} best_t={best_s.get('fast')}/{best_s.get('slow')}/{best_s.get('dwell')}",
                flush=True,
            )

    if best_cfg:
        print(
            "[best] "
            f"fast/slow/dwell={best_cfg['fast']}/{best_cfg['slow']}/{best_cfg['dwell']} "
            f"bull_on/off={best_cfg['bull_on']}/{best_cfg['bull_off']} "
            f"score={best_cfg['score']:.6f} avg={best_cfg['avg']:+.3f} worst={best_cfg['worst']:+.3f} neg_mass={best_cfg['neg_mass']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
