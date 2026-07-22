"""Command-line contract for spot sweep research."""

from __future__ import annotations

import argparse
import os

from .catalog import (
    _AXIS_CHOICES,
    _SpotSweepsHelpFormatter,
    _combo_full_preset_choices,
    _spot_sweeps_help_epilog,
)


def parse_spot_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Controlled spot evolution sweeps for the spot backtest engine.\nCanonical entrypoint: python -m tradebot.backtest spot ..."),
        formatter_class=_SpotSweepsHelpFormatter,
        epilog=_spot_sweeps_help_epilog(),
    )
    parser.add_argument(
        "--symbol",
        default="MNQ",
        help="Instrument symbol to backtest (uppercased).",
    )
    parser.add_argument(
        "--start",
        default="2025-01-08",
        help="Inclusive backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2026-01-08",
        help="Inclusive backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--bar-size",
        default="1 hour",
        help="Signal bar size (e.g. '30 mins', '1 hour'). ORB axis uses 15m regardless.",
    )
    parser.add_argument(
        "--spot-exec-bar-size",
        default=None,
        help=("Optional execution timeframe for fill simulation (e.g. '5 mins'). Signals still run on --bar-size."),
    )
    parser.add_argument(
        "--use-rth",
        action="store_true",
        default=False,
        help="Use regular trading hours bars (RTH) instead of full-session bars.",
    )
    parser.add_argument(
        "--cache-dir",
        default="db",
        help="Cache root for historical bars and sweep caches.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help=("Use cached bars at evaluation time. With --cache-policy=auto, preflight may hydrate missing caches before run."),
    )
    parser.add_argument(
        "--cache-policy",
        default="auto",
        choices=("auto", "strict"),
        help=(
            "Offline cache preflight policy. "
            "auto = hydrate via cache manager (resample-from-cache or fetch) before evaluating; "
            "strict = fail on any missing cache."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=("Parallelism for --axis all/combo_full (spawns per-axis worker processes). 0/omitted = auto (CPU count). Use --offline."),
    )
    parser.add_argument(
        "--base",
        default="champion",
        choices=("default", "champion", "champion_pnl", "dual_regime"),
        help="Base profile to mutate before running the selected axis (see Base Profiles below).",
    )
    parser.add_argument(
        "--close-eod",
        action="store_true",
        default=False,
        help="Force end-of-day flattening on spot positions.",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=False,
        help="Force spot to long-only (directional_spot = {'up': BUY 1}, no shorts).",
    )
    parser.add_argument(
        "--realism2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Spot realism (default): next-open fills, intrabar exits, liquidation marking, intrabar drawdown, "
            "position sizing, commission minimums, and stop gap handling. Defaults: spread=$0.01, "
            "commission=$0.005/share (min $1.00), risk sizing=1%% equity risk, max notional=50%%."
        ),
    )
    parser.add_argument(
        "--spot-spread",
        type=float,
        default=None,
        help="Spot spread in price units (e.g. 0.01). If omitted: 0.01 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-commission",
        type=float,
        default=None,
        help=("Spot commission per share/contract (price units). If omitted: 0.005 with --realism2, else 0.0."),
    )
    parser.add_argument(
        "--spot-commission-min",
        type=float,
        default=None,
        help="Spot commission minimum per order (price units). If omitted: 1.0 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-slippage",
        type=float,
        default=None,
        help="Spot slippage per share (price units). If omitted: 0.0.",
    )
    parser.add_argument(
        "--spot-sizing-mode",
        default=None,
        choices=("fixed", "notional_pct", "risk_pct"),
        help="Spot sizing mode (v2): fixed qty, %% notional, or %% equity risk-to-stop. If omitted: risk_pct with --realism2, else fixed.",
    )
    parser.add_argument(
        "--spot-risk-pct",
        type=float,
        default=None,
        help="Risk per trade as fraction of equity (v2). If omitted: 0.01 with --realism2, else 0.0.",
    )
    parser.add_argument(
        "--spot-notional-pct",
        type=float,
        default=None,
        help="Notional allocation per trade as fraction of equity (v2). If omitted: 0.0.",
    )
    parser.add_argument(
        "--spot-max-notional-pct",
        type=float,
        default=None,
        help="Max notional per trade as fraction of equity (v2). If omitted: 0.50 with --realism2, else 1.0.",
    )
    parser.add_argument(
        "--spot-min-qty",
        type=int,
        default=None,
        help="Min shares/contracts per trade (v2). If omitted: 1.",
    )
    parser.add_argument(
        "--spot-max-qty",
        type=int,
        default=None,
        help="Max shares/contracts per trade (v2); 0 means unlimited. If omitted: 0.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=100,
        help="Minimum trades required for a config to be kept in sweep outputs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many leaderboard rows to print per sweep.",
    )
    parser.add_argument(
        "--write-milestones",
        action="store_true",
        default=False,
        help="Write tradebot/backtest/spot_milestones.json from eligible sweep results (UI presets).",
    )
    parser.add_argument(
        "--merge-milestones",
        action="store_true",
        default=False,
        help="Merge eligible presets into an existing milestones JSON instead of overwriting from scratch.",
    )
    parser.add_argument(
        "--milestones-out",
        default="tradebot/backtest/spot_milestones.json",
        help="Output JSON path for --write-milestones.",
    )
    parser.add_argument(
        "--milestone-min-win",
        type=float,
        default=0.55,
        help="Milestone eligibility: minimum win rate (0..1).",
    )
    parser.add_argument(
        "--milestone-min-trades",
        type=int,
        default=200,
        help="Milestone eligibility: minimum trades.",
    )
    parser.add_argument(
        "--milestone-min-pnl-dd",
        type=float,
        default=8.0,
        help="Milestone eligibility: minimum pnl/dd ratio.",
    )
    parser.add_argument(
        "--milestone-add-top-pnl-dd",
        type=int,
        default=0,
        help=("When used with --merge-milestones, limits how many NEW presets from this run are added (top by pnl/dd). 0 = no limit."),
    )
    parser.add_argument(
        "--milestone-add-top-pnl",
        type=int,
        default=0,
        help=("When used with --merge-milestones, limits how many NEW presets from this run are added (top by pnl). 0 = no limit."),
    )
    parser.add_argument(
        "--seed-milestones",
        default=None,
        help=("Optional milestones JSON used as champion source override for --base champion/champion_pnl."),
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=("all", *_AXIS_CHOICES),
        metavar="AXIS",
        help="Axis to execute. Use 'all' for the axis_all plan. See Axis Catalog below.",
    )
    parser.add_argument("--sync-axis-docs", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--axis-docs-out",
        default="tradebot/backtest/spot_sweep_coverage_map.md",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--risk-overlays-skip-pop",
        action="store_true",
        default=False,
        help="risk_overlays: skip riskpop stage (riskoff+riskpanic only).",
    )
    # Internal flags (used by combo_full parallel sharding).
    parser.add_argument("--combo-full-cartesian-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--combo-full-cartesian-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--combo-full-cartesian-run-min-trades",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--combo-full-include-tick",
        action="store_true",
        default=False,
        help=("combo_full: include $TICK gate variants. Default keeps tick fixed off for faster/offline-friendly runs."),
    )
    parser.add_argument(
        "--combo-full-preset",
        default="",
        choices=_combo_full_preset_choices(include_empty=True, include_none=True),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cfg-stage", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cfg-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def default_jobs() -> int:
    detected = os.cpu_count()
    if detected is None:
        return 1
    try:
        detected_i = int(detected)
    except (TypeError, ValueError):
        return 1
    return max(1, detected_i)


def resolve_run_min_trades(args: argparse.Namespace) -> int:
    base = int(args.min_trades)
    raw = getattr(args, "combo_full_cartesian_run_min_trades", None)
    if raw is None:
        return base
    try:
        return int(raw)
    except (TypeError, ValueError):
        return base
