"""Flip autopsy for the daily regime router.

Prints host/climate switches with feature deltas and threshold context.

Typical use:
  python -m tradebot.backtest.tools.regime_router_flip_autopsy \\
    --preset backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v48_routerOnCompositeContextConfidence_20260404.json \\
    --start 2024-03-01 --end 2024-06-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ...climate_router import (
    ClimateDecision,
    RegimeRouterConfig,
    YearFeatures,
    bull_sovereign_entry_ok,
    classify_rolling_climate_v5,
    compute_window_features,
    load_daily_bars_from_intraday_csv,
    load_hf_host_strategy,
    named_host_target_dir,
    regime_router_config,
    regime_router_dwell_days,
)


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


@dataclass(frozen=True)
class RouterDay:
    day: str
    ready: bool
    proposed: ClimateDecision | None
    active: ClimateDecision | None
    pending_days: int
    bull_sovereign_raw_ok: bool
    bull_sovereign_active: bool
    bull_on_streak: int
    bull_off_streak: int
    effective_host: str | None
    effective_dir: str | None
    crash: YearFeatures | None
    fast: YearFeatures | None
    slow: YearFeatures | None


def _simulate_router_days(*, days, cfg: RegimeRouterConfig) -> list[RouterDay]:
    completed = []
    active: ClimateDecision | None = None
    pending: ClimateDecision | None = None
    pending_days = 0

    bull_active = False
    bull_on_streak = 0
    bull_off_streak = 0
    bull_raw_ok = False

    out: list[RouterDay] = []
    fast_n = max(2, int(cfg.fast_window_days))
    slow_n = max(int(fast_n), int(cfg.slow_window_days))
    crash_n = max(2, min(21, int(fast_n)))

    for bar in days:
        completed.append(bar)
        day = str(bar.ts)[:10]

        if len(completed) < int(slow_n):
            active = None
            pending = None
            pending_days = 0
            bull_active = False
            bull_on_streak = 0
            bull_off_streak = 0
            bull_raw_ok = False
            out.append(
                RouterDay(
                    day=day,
                    ready=False,
                    proposed=None,
                    active=None,
                    pending_days=0,
                    bull_sovereign_raw_ok=False,
                    bull_sovereign_active=False,
                    bull_on_streak=0,
                    bull_off_streak=0,
                    effective_host=None,
                    effective_dir=None,
                    crash=None,
                    fast=None,
                    slow=None,
                )
            )
            continue

        crash = compute_window_features(
            completed,
            label=len(completed),
            start_idx=len(completed) - int(crash_n),
            end_idx=len(completed),
        )
        fast = compute_window_features(
            completed,
            label=len(completed),
            start_idx=len(completed) - int(fast_n),
            end_idx=len(completed),
        )
        slow = compute_window_features(
            completed,
            label=len(completed),
            start_idx=len(completed) - int(slow_n),
            end_idx=len(completed),
        )
        proposed = classify_rolling_climate_v5(
            crash_features=crash,
            fast_features=fast,
            slow_features=slow,
            active=active,
            config=cfg,
        )

        if active is None:
            active = proposed
            pending = None
            pending_days = 0
        elif proposed == active:
            pending = None
            pending_days = 0
        else:
            if pending is None or pending != proposed:
                pending = proposed
                pending_days = 1
            else:
                pending_days += 1
            dwell_req = regime_router_dwell_days(
                active=active,
                proposed=proposed,
                base_dwell_days=int(cfg.min_dwell_days),
            )
            if int(pending_days) >= int(dwell_req):
                active = proposed
                pending = None
                pending_days = 0

        base_climate = str(active.climate) if active is not None else None
        base_host = str(active.chosen_host) if active is not None else None

        if base_climate != "bull_grind_low_vol" or base_host != "buyhold":
            bull_active = False
            bull_on_streak = 0
            bull_off_streak = 0
            bull_raw_ok = False
        else:
            bull_raw_ok = bool(
                bull_sovereign_entry_ok(
                    climate=base_climate,
                    chosen_host=base_host,
                    fast_features=fast,
                    slow_features=slow,
                )
            )
            on_confirm = max(1, int(cfg.bull_sovereign_on_confirm_days))
            off_confirm = max(1, int(cfg.bull_sovereign_off_confirm_days))
            if bull_raw_ok:
                bull_off_streak = 0
                if bull_active:
                    bull_on_streak = 0
                else:
                    bull_on_streak += 1
                    if int(bull_on_streak) >= int(on_confirm):
                        bull_active = True
                        bull_on_streak = 0
            else:
                bull_on_streak = 0
                if bull_active:
                    bull_off_streak += 1
                    if int(bull_off_streak) >= int(off_confirm):
                        bull_active = False
                        bull_off_streak = 0
                else:
                    bull_off_streak = 0

        effective_host = (
            "bull_ma200_v1"
            if bool(bull_active) and str(base_host or "") == "buyhold"
            else str(base_host) if base_host is not None else None
        )
        dir_raw = named_host_target_dir(completed, effective_host) if effective_host is not None else None
        effective_dir = str(dir_raw) if dir_raw in ("up", "down") else None

        out.append(
            RouterDay(
                day=day,
                ready=True,
                proposed=proposed,
                active=active,
                pending_days=int(pending_days),
                bull_sovereign_raw_ok=bool(bull_raw_ok),
                bull_sovereign_active=bool(bull_active),
                bull_on_streak=int(bull_on_streak),
                bull_off_streak=int(bull_off_streak),
                effective_host=effective_host,
                effective_dir=effective_dir,
                crash=crash,
                fast=fast,
                slow=slow,
            )
        )

    return out


def _fmt(x: float | None, *, digits: int = 3) -> str:
    if x is None:
        return "none"
    return f"{float(x):+.{int(digits)}f}"


def _print_segments(rows: list[RouterDay]) -> None:
    segs: list[tuple[str, str, int, str, str, str | None, RouterDay]] = []
    last_key: tuple[str, str, str | None, bool] | None = None

    for r in rows:
        climate = str(r.active.climate) if r.active is not None else "none"
        host = str(r.effective_host) if r.effective_host is not None else "none"
        key = (host, climate, r.effective_dir, bool(r.bull_sovereign_active))
        if last_key is None or key != last_key:
            segs.append((r.day, r.day, 1, host, climate, r.effective_dir, r))
            last_key = key
        else:
            a, _b, n, h, c, d, r0 = segs[-1]
            segs[-1] = (a, r.day, int(n) + 1, h, c, d, r0)

    print(f"segments={len(segs)}")
    for a, b, n, host, climate, eff_dir, r0 in segs:
        bull = "*" if bool(r0.bull_sovereign_active) else ""
        crash = r0.crash
        fast = r0.fast
        slow = r0.slow
        tail = ""
        if crash is not None and fast is not None and slow is not None:
            tail = (
                f" crash_ret={_fmt(crash.ret)} crash_dd={_fmt(crash.maxdd)} crash_rv={_fmt(crash.rv)}"
                f" fast_ret={_fmt(fast.ret)} slow_ret={_fmt(slow.ret)}"
                f" fast_dd={_fmt(fast.maxdd)} slow_dd={_fmt(slow.maxdd)}"
            )
        print(
            f"{a}→{b} ({n:3d}d) host={host}{bull} dir={(eff_dir or 'none'):4s} climate={climate:26s}{tail}"
        )


def _print_switches(rows: list[RouterDay], *, cfg: RegimeRouterConfig) -> None:
    def key(r: RouterDay) -> tuple[str, str, str | None, bool]:
        climate = str(r.active.climate) if r.active is not None else "none"
        host = str(r.effective_host) if r.effective_host is not None else "none"
        return (host, climate, r.effective_dir, bool(r.bull_sovereign_active))

    def episode_checks(r: RouterDay) -> str:
        crash = r.crash
        if crash is None:
            return ""
        return (
            f" episode(ret<={cfg.hf_takeover_crash_ret_max:+.2f}?"
            f"{float(crash.ret) <= float(cfg.hf_takeover_crash_ret_max)}"
            f" dd>={cfg.hf_takeover_crash_maxdd_min:.2f}?"
            f"{float(crash.maxdd) >= float(cfg.hf_takeover_crash_maxdd_min)}"
            f" rv<={cfg.hf_takeover_crash_rv_max:.2f}?"
            f"{float(crash.rv) <= float(cfg.hf_takeover_crash_rv_max)})"
        )

    switches = 0
    prev = rows[0]
    for r in rows[1:]:
        if key(r) == key(prev):
            prev = r
            continue
        switches += 1
        a_host, a_clim, a_dir, a_bull = key(prev)
        b_host, b_clim, b_dir, b_bull = key(r)
        print(
            f"SWITCH {r.day} {a_host}{'*' if a_bull else ''}/{a_clim}/{a_dir or 'none'}"
            f" -> {b_host}{'*' if b_bull else ''}/{b_clim}/{b_dir or 'none'}"
        )
        crash = r.crash
        fast = r.fast
        slow = r.slow
        if crash is not None:
            print(f"  crash ret={_fmt(crash.ret)} dd={_fmt(crash.maxdd)} rv={_fmt(crash.rv)}{episode_checks(r)}")
        if fast is not None:
            print(f"  fast  ret={_fmt(fast.ret)} dd={_fmt(fast.maxdd)} rv={_fmt(fast.rv)} eff={_fmt(fast.efficiency)}")
        if slow is not None:
            print(f"  slow  ret={_fmt(slow.ret)} dd={_fmt(slow.maxdd)} rv={_fmt(slow.rv)} eff={_fmt(slow.efficiency)}")
        prev = r

    print(f"switches={switches}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flip autopsy for the daily regime router.")
    ap.add_argument(
        "--preset",
        default="backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v48_routerOnCompositeContextConfidence_20260404.json",
    )
    ap.add_argument(
        "--intraday-csv",
        default="db/TQQQ/TQQQ_2016-01-01_2026-01-19_1hour_rth.csv",
        help="Any cached intraday bars file; this tool aggregates to daily bars.",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-segments", action="store_true")
    ap.add_argument("--no-switches", action="store_true")
    args = ap.parse_args(argv)

    preset = Path(args.preset)
    strategy, _bar_size, _use_rth = load_hf_host_strategy(preset)
    cfg = regime_router_config(strategy)

    days = load_daily_bars_from_intraday_csv(Path(args.intraday_csv))
    rows = _simulate_router_days(days=days, cfg=cfg)

    start_default = _parse_date(rows[0].day)
    end_default = _parse_date(rows[-1].day)
    start = _parse_date(args.start) if args.start is not None else start_default
    end = _parse_date(args.end) if args.end is not None else end_default
    scoped = [r for r in rows if start <= _parse_date(r.day) <= end]
    if not scoped:
        raise SystemExit("No rows in requested range")

    print(f"preset={preset.name}")
    print(f"range={start.isoformat()}→{end.isoformat()} days={len(scoped)}")
    print(
        f"cfg fast/slow/dwell={cfg.fast_window_days}/{cfg.slow_window_days}/{cfg.min_dwell_days} "
        f"episode(ret<={cfg.hf_takeover_crash_ret_max:+.2f},dd>={cfg.hf_takeover_crash_maxdd_min:.2f},rv<={cfg.hf_takeover_crash_rv_max:.2f})"
    )

    if not bool(args.no_segments):
        _print_segments(scoped)
    if not bool(args.no_switches):
        _print_switches(scoped, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

