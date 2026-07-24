"""Options backtest tooling (leaderboard sweeps).

This keeps bot presets and documentation reproducible without scraping markdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from datetime import date, datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import time

from .cli_utils import parse_date as _parse_date
from ..chart_data.history import cache_data_revision
from .config import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    OptionsStrategyConfig,
    SyntheticConfig,
)
from .engine import OptionsBacktestSourcePool, run_backtest
from .sweeps import Progress, count_total_combos, fmt_duration, normalize_jobs, write_json
from ..chart_data.cache import series_cache_service


_OPTIONS_RESULT_NAMESPACE = "options.sweep_result.v1"
_OPTIONS_RESULT_CACHE = series_cache_service()


def _sweep_revision(*, backtest: BacktestConfig, symbol: str) -> str:
    """Bind warm results to the exact code tree and cached market-data state."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.blake2b(digest_size=20)
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        cache_data_revision(Path(backtest.cache_dir) / str(symbol)).encode()
    )
    calibration = Path(backtest.calibration_dir) / f"{symbol}.json"
    digest.update(b"\0calibration\0")
    digest.update(calibration.read_bytes() if calibration.is_file() else b"<missing>")
    return digest.hexdigest()


def _combo_cache_key(
    *,
    revision: str,
    symbol: str,
    backtest: BacktestConfig,
    synthetic: SyntheticConfig,
    group: dict,
    min_trades: int,
    combo: tuple,
) -> str:
    payload = json.dumps(
        {
            "revision": revision,
            "symbol": symbol,
            "backtest": asdict(backtest),
            "synthetic": asdict(synthetic),
            "group": group,
            "min_trades": int(min_trades),
            "combo": combo,
        },
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=20).hexdigest()


def _valid_result_payload(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"entry"}
        and (value["entry"] is None or isinstance(value["entry"], dict))
    )


# region CLI
def options_leaderboard_main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard JSON from offline sweeps")
    parser.add_argument("--symbol", default="SLV")
    parser.add_argument("--start", default="2025-07-02")
    parser.add_argument("--end", default="2026-01-02")
    parser.add_argument("--bar-size", default="1 hour")
    parser.add_argument("--use-rth", action="store_true")
    parser.add_argument(
        "--wing-points",
        type=float,
        default=None,
        help="Use a fixed OTM point offset for defined-risk wings instead of the legacy 1% offset",
    )
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto)")
    parser.add_argument(
        "--progress-sec",
        type=float,
        default=30.0,
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
    if args.wing_points is not None and float(args.wing_points) <= 0:
        parser.error("--wing-points must be positive")

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    jobs = normalize_jobs(int(args.jobs))

    interval_sec = float(args.progress_sec)
    interval_sec = interval_sec if interval_sec >= 0 else 30.0
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
    wing = (
        {"otm_offset_points": float(args.wing_points)}
        if args.wing_points is not None
        else {"moneyness_offset_pct": 1.0}
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
                {"action": "BUY", "right": "PUT", "qty": 1, **wing},
            ],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Put Credit Spread (filtered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, **wing},
            ],
            filters,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Iron Condor (unfiltered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, **wing},
                {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "CALL", "qty": 1, **wing},
            ],
            None,
            ema_entry_mode="trend",
        ),
        _group_spec(
            "Iron Condor (filtered)",
            "PUT",
            [
                {"action": "SELL", "right": "PUT", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "PUT", "qty": 1, **wing},
                {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                {"action": "BUY", "right": "CALL", "qty": 1, **wing},
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
                        {"action": "BUY", "right": "PUT", "qty": 1, **wing},
                    ],
                    "down": [
                        {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "CALL", "qty": 1, **wing},
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
                        {"action": "BUY", "right": "PUT", "qty": 1, **wing},
                    ],
                    "down": [
                        {"action": "SELL", "right": "CALL", "qty": 1, "moneyness_offset_pct": 0.0},
                        {"action": "BUY", "right": "CALL", "qty": 1, **wing},
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
    revision = _sweep_revision(backtest=base_backtest, symbol=args.symbol)
    result_db = base_backtest.cache_dir / "options_sweeps_results.sqlite3"
    worker_ctx = {
        "cwd": os.getcwd(),
        "symbol": args.symbol,
        "backtest": base_backtest,
        "synthetic": synthetic,
        "groups": tuple(
            (group, _filters_cfg(group.get("filters")), int(grid["min_trades"]))
            for group in groups
        ),
    }
    pool_context = (
        ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_init_worker,
            initargs=(worker_ctx,),
        )
        if jobs > 1
        else nullcontext(None)
    )
    source_pool = OptionsBacktestSourcePool() if jobs == 1 else None
    try:
        with pool_context as pool:
            for group_idx, group in enumerate(groups):
                progress.start_group(
                    group_idx + 1,
                    group["name"],
                    total=count_total_combos(grid),
                )
                entries = _run_group(
                    symbol=args.symbol,
                    backtest=base_backtest,
                    synthetic=synthetic,
                    grid=grid,
                    group=group,
                    progress=progress,
                    jobs=jobs,
                    pool=pool,
                    worker_group_idx=group_idx,
                    source_pool=source_pool,
                    result_db=result_db,
                    revision=revision,
                )
                progress.finish_group()
                payload["groups"].append(
                    {
                        "name": group["name"],
                        "filters": group["filters"],
                        "entries": entries,
                    }
                )
    finally:
        if source_pool is not None:
            source_pool.close()

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
    pool: ProcessPoolExecutor | None = None,
    worker_group_idx: int = 0,
    source_pool: OptionsBacktestSourcePool | None = None,
    result_db: Path | None = None,
    revision: str | None = None,
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

    combos = list(_combos())
    cache_keys = [
        (
            _combo_cache_key(
                revision=revision,
                symbol=symbol,
                backtest=backtest,
                synthetic=synthetic,
                group=group,
                min_trades=min_trades,
                combo=combo,
            )
            if revision is not None and result_db is not None
            else ""
        )
        for combo in combos
    ]
    cached = _OPTIONS_RESULT_CACHE.get_persistent_many(
        db_path=result_db,
        namespace=_OPTIONS_RESULT_NAMESPACE,
        key_hashes=(key for key in cache_keys if key),
        validator=_valid_result_payload,
    )
    results: list[dict] = []
    pending: list[tuple[str, tuple]] = []
    for key, combo in zip(cache_keys, combos):
        payload = cached.get(key)
        if payload is None:
            pending.append((key, combo))
            continue
        progress.advance()
        entry = payload["entry"]
        if entry is not None:
            results.append(entry)

    if not pending:
        results.sort(
            key=lambda row: (
                row["metrics"]["pnl"],
                row["metrics"]["win_rate"],
            ),
            reverse=True,
        )
        return results[:top_n]

    writes: dict[str, object] = {}
    if jobs == 1:
        owns_sources = source_pool is None
        sources = source_pool or OptionsBacktestSourcePool()
        try:
            for key, combo in pending:
                dte, moneyness, pt, sl, ema, entry_mode, flip, hold, only_profit = combo
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
                    source_pool=sources,
                )
                progress.advance()
                if key:
                    writes[key] = {"entry": entry}
                if entry is not None:
                    results.append(entry)
        finally:
            if owns_sources:
                sources.close()
    else:
        pool_context = (
            nullcontext(pool)
            if pool is not None
            else ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_worker,
                initargs=(
                    {
                        "cwd": os.getcwd(),
                        "symbol": symbol,
                        "backtest": backtest,
                        "synthetic": synthetic,
                        "groups": ((group, filters_cfg, min_trades),),
                    },
                ),
            )
        )
        task_group_idx = worker_group_idx if pool is not None else 0
        with pool_context as active_pool:
            assert active_pool is not None
            tasks = ((task_group_idx, combo) for _, combo in pending)
            for (key, _), entry in zip(
                pending,
                active_pool.map(_run_combo_worker, tasks, chunksize=8),
            ):
                progress.advance()
                if key:
                    writes[key] = {"entry": entry}
                if entry is not None:
                    results.append(entry)

    _OPTIONS_RESULT_CACHE.set_persistent_many(
        db_path=result_db,
        namespace=_OPTIONS_RESULT_NAMESPACE,
        values=writes,
    )

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
    source_pool: OptionsBacktestSourcePool | None = None,
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
                    otm_offset_points=float(leg.get("otm_offset_points", 0.0)),
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
                otm_offset_points=float(leg.get("otm_offset_points", 0.0)),
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

    cfg = ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)
    result = run_backtest(
        cfg,
        options_source=source_pool.source_for(cfg) if source_pool is not None else None,
    )
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
                **(
                    {"otm_offset_points": leg.otm_offset_points}
                    if leg.otm_offset_points
                    else {}
                ),
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
                    **(
                        {"otm_offset_points": leg.otm_offset_points}
                        if leg.otm_offset_points
                        else {}
                    ),
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
_WORKER_SOURCES: OptionsBacktestSourcePool | None = None


def _init_worker(ctx: dict) -> None:
    global _WORKER_CTX, _WORKER_SOURCES
    _WORKER_CTX = ctx
    _WORKER_SOURCES = OptionsBacktestSourcePool()
    cwd = ctx.get("cwd")
    if cwd:
        os.chdir(cwd)


def _run_combo_worker(task: tuple[int, tuple]) -> dict | None:
    if _WORKER_CTX is None or _WORKER_SOURCES is None:
        raise RuntimeError("Worker context not initialized")
    group_idx, combo = task
    group, filters_cfg, min_trades = _WORKER_CTX["groups"][group_idx]
    dte, moneyness, pt, sl, ema, entry_mode, flip, hold, only_profit = combo
    return _run_combo(
        symbol=_WORKER_CTX["symbol"],
        backtest=_WORKER_CTX["backtest"],
        synthetic=_WORKER_CTX["synthetic"],
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
        source_pool=_WORKER_SOURCES,
    )


# endregion

main = options_leaderboard_main


if __name__ == "__main__":
    options_leaderboard_main()
