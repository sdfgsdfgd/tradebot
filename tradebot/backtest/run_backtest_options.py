"""Options backtest tooling (leaderboard sweeps).

This keeps bot presets and documentation reproducible without scraping markdown.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import time

from .cli_utils import parse_date as _parse_date
from .config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    OptionsStrategyConfig,
    SyntheticConfig,
)
from .engine import run_backtest
from .sweeps import Progress, count_total_combos, fmt_duration, normalize_jobs, write_json


# region CLI
def options_leaderboard_main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard JSON from offline sweeps")
    parser.add_argument("--symbol", default="SLV")
    parser.add_argument("--start", default="2025-07-02")
    parser.add_argument("--end", default="2026-01-02")
    parser.add_argument("--bar-size", default="1 hour")
    parser.add_argument("--use-rth", action="store_true")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto)")
    parser.add_argument(
        "--progress-sec",
        type=float,
        default=120.0,
        help="Progress update interval in seconds (0 = only group start/end)",
    )
    parser.add_argument("--out", default="tradebot/backtest/leaderboard.json")
    parser.add_argument(
        "--include-spot-milestones",
        dest="include_spot_milestones",
        action="store_true",
        default=True,
        help="Append tradebot/backtest/spot_milestones.json groups into the generated output (default: on)",
    )
    parser.add_argument(
        "--no-include-spot-milestones",
        dest="include_spot_milestones",
        action="store_false",
        help="Disable appending spot milestones into the generated output",
    )
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    jobs = normalize_jobs(int(args.jobs))

    interval_sec = float(args.progress_sec)
    interval_sec = interval_sec if interval_sec >= 0 else 120.0
    sweep_start = time.monotonic()

    grid = {
        "dte": [0, 5, 10, 20],
        "moneyness_pct": [1.0, 2.0],
        "profit_target": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
        "stop_loss": [0.35, 0.8, 1.0],
        "ema_preset": ["3/7", "9/21", "20/50"],
        "ema_entry_mode": ["trend", "cross"],
        "exit_on_signal_flip": [False, True],
        "flip_exit_min_hold_bars": [6],
        "flip_exit_only_if_profit": [True],
        "min_trades": 8,
    }

    filters = {
        "rv_min": 0.15,
        "rv_max": 0.60,
        "ema_spread_min_pct": 0.05,
        "ema_slope_min_pct": 0.01,
        "entry_start_hour": 10,
        "entry_end_hour": 15,
        "skip_first_bars": 2,
        "cooldown_bars": 4,
    }

    base_backtest = BacktestConfig(
        start=start,
        end=end,
        bar_size=args.bar_size,
        use_rth=bool(args.use_rth),
        starting_cash=10000.0,
        risk_free_rate=0.02,
        cache_dir=Path("db"),
        calibration_dir=Path("db/calibration"),
        output_dir=Path("backtests/out"),
        calibrate=False,
        offline=True,
    )

    synthetic = SyntheticConfig(
        rv_lookback=60,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.2,
        iv_floor=0.05,
        term_slope=0.02,
        skew=-0.25,
        min_spread_pct=0.1,
    )

    groups = [
        _group_spec(
            "Long CALL (unfiltered)",
            "CALL",
            [{"action": "BUY", "right": "CALL", "qty": 1}],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Long CALL (filtered)",
            "CALL",
            [{"action": "BUY", "right": "CALL", "qty": 1}],
            filters,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Short PUT (unfiltered)",
            "PUT",
            [{"action": "SELL", "right": "PUT", "qty": 1}],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Short PUT (filtered)",
            "PUT",
            [{"action": "SELL", "right": "PUT", "qty": 1}],
            filters,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Risk Reversal (unfiltered) [BUY CALL + SELL PUT]",
            "CALL",
            [
                {"action": "BUY", "right": "CALL", "qty": 1},
                {"action": "SELL", "right": "PUT", "qty": 1},
            ],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Risk Reversal (filtered) [BUY CALL + SELL PUT]",
            "CALL",
            [
                {"action": "BUY", "right": "CALL", "qty": 1},
                {"action": "SELL", "right": "PUT", "qty": 1},
            ],
            filters,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Put Credit Spread (unfiltered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
            ],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Put Credit Spread (filtered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
            ],
            filters,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Iron Condor (unfiltered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
                {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "CALL", "qty": 1, "moneyness_offset_pct": 1.0},
            ],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Iron Condor (filtered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
                {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "CALL", "qty": 1, "moneyness_offset_pct": 1.0},
            ],
            filters,
            ema_entry_mode="trend",
        ),
    ]

    # Directional flip presets: choose legs based on the EMA-derived up/down direction.
    # Useful for reversal-style strategies (e.g. 3/7 cross entry + flip exit).
    groups.extend(
        [
            {
                "name": "Directional Flip Credit Spreads (unfiltered) [PUT up / CALL down]",
                "right": "PUT",
                "filters": None,
                "directional_leg_templates": {
                    "up": [
                        {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
                    ],
                    "down": [
                        {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "CALL", "qty": 1, "moneyness_offset_pct": 1.0},
                    ],
                },
            },
            {
                "name": "Directional Flip Credit Spreads (filtered) [PUT up / CALL down]",
                "right": "PUT",
                "filters": filters,
                "directional_leg_templates": {
                    "up": [
                        {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "PUT", "qty": 1, "moneyness_offset_pct": 1.0},
                    ],
                    "down": [
                        {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "CALL", "qty": 1, "moneyness_offset_pct": 1.0},
                    ],
                },
            },
        ]
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bar_size": args.bar_size,
        "use_rth": bool(args.use_rth),
        "grid": grid,
        "groups": [],
    }

    progress = Progress(
        total=count_total_combos(grid) * len(groups),
        interval_sec=interval_sec,
        groups=len(groups),
    )
    for group_idx, group in enumerate(groups, start=1):
        progress.start_group(group_idx, group["name"], total=count_total_combos(grid))
        entries = _run_group(
            symbol=args.symbol,
            backtest=base_backtest,
            synthetic=synthetic,
            grid=grid,
            group=group,
            progress=progress,
            jobs=jobs,
        )
        progress.finish_group()
        payload["groups"].append(
            {
                "name": group["name"],
                "filters": group["filters"],
                "entries": entries,
            }
        )

    if bool(args.include_spot_milestones):
        milestones_path = Path(__file__).resolve().parent / "spot_milestones.json"
        try:
            spot_payload = json.loads(milestones_path.read_text())
            if isinstance(spot_payload, dict):
                spot_groups = spot_payload.get("groups", [])
                if isinstance(spot_groups, list) and spot_groups:
                    payload["groups"].extend(list(spot_groups))
        except Exception:
            # Keep leaderboard generation robust; milestones are optional.
            pass

    out_path = Path(args.out)
    write_json(out_path, payload, sort_keys=True)

    elapsed = time.monotonic() - sweep_start
    total = count_total_combos(grid) * len(groups)
    kept = sum(len(g.get("entries", [])) for g in payload.get("groups", []))
    print(f"Done: {kept}/{total} kept | jobs={jobs} | elapsed {fmt_duration(elapsed)}", flush=True)
# endregion


# region Group Runner
def _group_spec(
    name: str,
    right: str,
    leg_templates: list[dict],
    filters: dict | None,
    *,
    ema_entry_mode: str,
) -> dict:
    return {
        "name": name,
        "right": right,
        "leg_templates": leg_templates,
        "filters": filters,
        "ema_entry_mode": ema_entry_mode,
    }


def _run_group(
    *,
    symbol: str,
    backtest: BacktestConfig,
    synthetic: SyntheticConfig,
    grid: dict,
    group: dict,
    progress: "Progress",
    top_n: int = 2000,
    jobs: int = 1,
) -> list[dict]:
    jobs = normalize_jobs(int(jobs))
    filters_cfg = _filters_cfg(group.get("filters"))
    min_trades = int(grid["min_trades"])

    def _combos():
        g = grid
        base_combos = product(
            g["dte"],
            g["moneyness_pct"],
            g["profit_target"],
            g["stop_loss"],
            g["ema_preset"],
            g["ema_entry_mode"],
        )
        for dte, moneyness, pt, sl, ema, entry_mode in base_combos:
            for flip in g["exit_on_signal_flip"]:
                if flip:
                    flip_variants = product(g["flip_exit_min_hold_bars"], g["flip_exit_only_if_profit"])
                else:
                    flip_variants = [(0, False)]
                for hold, only_profit in flip_variants:
                    yield (dte, moneyness, pt, sl, ema, str(entry_mode), bool(flip), int(hold), bool(only_profit))

    results: list[dict] = []
    if jobs == 1:
        for dte, moneyness, pt, sl, ema, entry_mode, flip, hold, only_profit in _combos():
            progress.advance()
            entry = _run_combo(
                symbol=symbol,
                backtest=backtest,
                synthetic=synthetic,
                group=group,
                filters_cfg=filters_cfg,
                min_trades=min_trades,
                dte=dte,
                moneyness=moneyness,
                profit_target=pt,
                stop_loss=sl,
                ema_preset=ema,
                ema_entry_mode=entry_mode,
                exit_on_signal_flip=flip,
                flip_exit_min_hold_bars=hold,
                flip_exit_only_if_profit=only_profit,
            )
            if entry is not None:
                results.append(entry)
        results.sort(key=lambda row: (row["metrics"]["pnl"], row["metrics"]["win_rate"]), reverse=True)
        return results[:top_n]

    ctx = {
        "cwd": os.getcwd(),
        "symbol": symbol,
        "backtest": backtest,
        "synthetic": synthetic,
        "group": group,
        "filters_cfg": filters_cfg,
        "min_trades": min_trades,
    }
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(ctx,)) as pool:
        for entry in pool.map(_run_combo_worker, _combos(), chunksize=8):
            progress.advance()
            if entry is not None:
                results.append(entry)

    results.sort(key=lambda row: (row["metrics"]["pnl"], row["metrics"]["win_rate"]), reverse=True)
    return results[:top_n]


# endregion


# region Combo Runner
def _filters_cfg(raw: dict | None) -> FiltersConfig | None:
    if not raw:
        return None
    return FiltersConfig(
        rv_min=raw.get("rv_min"),
        rv_max=raw.get("rv_max"),
        ema_spread_min_pct=raw.get("ema_spread_min_pct"),
        ema_slope_min_pct=raw.get("ema_slope_min_pct"),
        entry_start_hour=raw.get("entry_start_hour"),
        entry_end_hour=raw.get("entry_end_hour"),
        skip_first_bars=int(raw.get("skip_first_bars", 0)),
        cooldown_bars=int(raw.get("cooldown_bars", 0)),
    )


def _run_combo(
    *,
    symbol: str,
    backtest: BacktestConfig,
    synthetic: SyntheticConfig,
    group: dict,
    filters_cfg: FiltersConfig | None,
    min_trades: int,
    dte: int,
    moneyness: float,
    profit_target: float,
    stop_loss: float,
    ema_preset: str,
    ema_entry_mode: str,
    exit_on_signal_flip: bool,
    flip_exit_min_hold_bars: int,
    flip_exit_only_if_profit: bool,
) -> dict | None:
    directional_legs = None
    legs = None
    legs_for_display: tuple[LegConfig, ...] = ()

    if group.get("directional_leg_templates"):
        directional_legs = {}
        for key in ("up", "down"):
            templates = group.get("directional_leg_templates", {}).get(key) or []
            if not templates:
                continue
            directional_legs[key] = tuple(
                LegConfig(
                    action=leg["action"],
                    right=leg["right"],
                    moneyness_pct=float(moneyness) + float(leg.get("moneyness_offset_pct", 0.0)),
                    qty=int(leg["qty"]),
                )
                for leg in templates
            )
        if directional_legs:
            legs_for_display = directional_legs.get("up") or next(iter(directional_legs.values()))
    else:
        legs = tuple(
            LegConfig(
                action=leg["action"],
                right=leg["right"],
                moneyness_pct=float(moneyness) + float(leg.get("moneyness_offset_pct", 0.0)),
                qty=int(leg["qty"]),
            )
            for leg in group["leg_templates"]
        )
        legs_for_display = legs

    strategy = OptionsStrategyConfig(
        name="credit_spread",
        instrument="options",
        symbol=symbol,
        exchange=None,
        right=str(group.get("right", "PUT")),
        entry_days=(0, 1, 2, 3, 4),
        max_entries_per_day=0,
        max_open_trades=0,
        dte=int(dte),
        otm_pct=1.0,
        width_pct=1.0,
        profit_target=float(profit_target),
        stop_loss=float(stop_loss),
        exit_dte=0,
        quantity=1,
        stop_loss_basis="max_loss",
        min_credit=0.01,
        ema_preset=ema_preset,
        ema_entry_mode=str(ema_entry_mode),
        entry_confirm_bars=0,
        regime_ema_preset=None,
        regime_bar_size=None,
        ema_directional=False,
        exit_on_signal_flip=bool(exit_on_signal_flip),
        flip_exit_mode="entry",
        flip_exit_gate_mode="off",
        flip_exit_min_hold_bars=int(flip_exit_min_hold_bars),
        flip_exit_only_if_profit=bool(flip_exit_only_if_profit),
        direction_source="ema",
        directional_legs=directional_legs,
        directional_spot=None,
        legs=legs,
        filters=filters_cfg,
        spot_profit_target_pct=None,
        spot_stop_loss_pct=None,
        spot_close_eod=False,
    )

    result = run_backtest(ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic))
    summary = result.summary
    if summary.trades < min_trades:
        return None
    if summary.total_pnl <= 0:
        return None

    strategy_payload = {
        "instrument": "options",
        "dte": int(dte),
        "profit_target": float(profit_target),
        "stop_loss": float(stop_loss),
        "ema_preset": str(ema_preset) if ema_preset is not None else None,
        "ema_entry_mode": str(ema_entry_mode),
        "exit_on_signal_flip": bool(exit_on_signal_flip),
        "flip_exit_min_hold_bars": int(flip_exit_min_hold_bars),
        "flip_exit_only_if_profit": bool(flip_exit_only_if_profit),
        "legs": [
            {
                "action": leg.action,
                "right": leg.right,
                "moneyness_pct": leg.moneyness_pct,
                "qty": leg.qty,
            }
            for leg in legs_for_display
        ],
        "entry_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    }
    if directional_legs:
        strategy_payload["directional_legs"] = {
            key: [
                {
                    "action": leg.action,
                    "right": leg.right,
                    "moneyness_pct": leg.moneyness_pct,
                    "qty": leg.qty,
                }
                for leg in dlegs
            ]
            for key, dlegs in directional_legs.items()
        }

    return {
        "metrics": {
            "pnl": summary.total_pnl,
            "win_rate": summary.win_rate,
            "trades": summary.trades,
            "avg_hold_hours": summary.avg_hold_hours,
            "max_drawdown": summary.max_drawdown,
        },
        "strategy": strategy_payload,
    }


# endregion


# region Multiprocessing
_WORKER_CTX: dict | None = None


def _init_worker(ctx: dict) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx
    cwd = ctx.get("cwd")
    if cwd:
        os.chdir(cwd)


def _run_combo_worker(combo: tuple[int, float, float, float, str, str, bool, int, bool]) -> dict | None:
    if _WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized")
    dte, moneyness, pt, sl, ema, entry_mode, flip, hold, only_profit = combo
    return _run_combo(
        symbol=_WORKER_CTX["symbol"],
        backtest=_WORKER_CTX["backtest"],
        synthetic=_WORKER_CTX["synthetic"],
        group=_WORKER_CTX["group"],
        filters_cfg=_WORKER_CTX["filters_cfg"],
        min_trades=_WORKER_CTX["min_trades"],
        dte=dte,
        moneyness=moneyness,
        profit_target=pt,
        stop_loss=sl,
        ema_preset=ema,
        ema_entry_mode=entry_mode,
        exit_on_signal_flip=flip,
        flip_exit_min_hold_bars=hold,
        flip_exit_only_if_profit=only_profit,
    )


# endregion

main = options_leaderboard_main


if __name__ == "__main__":
    options_leaderboard_main()
