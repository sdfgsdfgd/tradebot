"""Controlled spot evolution sweeps (one new axis at a time).

Designed to start from the current MNQ spot 12m champion (or a selected base)
and explore incremental improvements without confounding:
  0) timing (EMA preset)
  A) volume gating
  B) time-of-day (ET) gating
  C) ATR-scaled exits
  D) ORB + Fibonacci target variants (15m)
  E) Supertrend regime sensitivity squeeze
  F) Dual regime gating (regime2)
  G) Chop-killer quality filters (spread/slope/cooldown/skip-open)

All knobs are opt-in; default bot behavior is unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timezone
from pathlib import Path

from .config import BacktestConfig, ConfigBundle, FiltersConfig, SpotLegConfig, StrategyConfig, SyntheticConfig
from .data import ContractMeta, IBKRHistoricalData
from .engine import _run_spot_backtest


def _parse_date(value: str) -> date:
    year_s, month_s, day_s = str(value).strip().split("-")
    return date(int(year_s), int(month_s), int(day_s))


def _bundle_base(
    *,
    symbol: str,
    start: date,
    end: date,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
    filters: FiltersConfig | None,
    entry_signal: str = "ema",
    ema_preset: str | None = "2/4",
    entry_confirm_bars: int = 0,
    spot_exit_mode: str = "pct",
    spot_atr_period: int = 14,
    spot_pt_atr_mult: float = 1.5,
    spot_sl_atr_mult: float = 1.0,
    orb_window_mins: int = 15,
    orb_risk_reward: float = 2.0,
    orb_target_mode: str = "rr",
    spot_profit_target_pct: float | None = 0.015,
    spot_stop_loss_pct: float | None = 0.03,
    flip_exit_min_hold_bars: int = 4,
    max_open_trades: int = 2,
    spot_close_eod: bool = False,
) -> ConfigBundle:
    backtest = BacktestConfig(
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        starting_cash=100_000.0,
        risk_free_rate=0.02,
        cache_dir=Path(cache_dir),
        calibration_dir=Path(cache_dir) / "calibration",
        output_dir=Path("backtests/out"),
        calibrate=False,
        offline=bool(offline),
    )

    strategy = StrategyConfig(
        name="spot_evolve",
        instrument="spot",
        symbol=str(symbol).strip().upper(),
        exchange=None,
        right="PUT",
        entry_days=(0, 1, 2, 3, 4),
        max_entries_per_day=0,
        max_open_trades=int(max_open_trades),
        dte=0,
        otm_pct=0.0,
        width_pct=0.0,
        profit_target=0.0,
        stop_loss=0.0,
        exit_dte=0,
        quantity=1,
        stop_loss_basis="max_loss",
        min_credit=None,
        ema_preset=ema_preset,
        ema_entry_mode="cross",
        entry_confirm_bars=int(entry_confirm_bars),
        regime_ema_preset=None,
        regime_bar_size="4 hours",
        ema_directional=False,
        exit_on_signal_flip=True,
        flip_exit_mode="entry",
        flip_exit_min_hold_bars=int(flip_exit_min_hold_bars),
        flip_exit_only_if_profit=False,
        direction_source="ema",
        directional_legs=None,
        directional_spot={
            "up": SpotLegConfig(action="BUY", qty=1),
            "down": SpotLegConfig(action="SELL", qty=1),
        },
        legs=None,
        filters=filters,
        spot_profit_target_pct=spot_profit_target_pct,
        spot_stop_loss_pct=spot_stop_loss_pct,
        spot_close_eod=bool(spot_close_eod),
        entry_signal=str(entry_signal),
        orb_window_mins=int(orb_window_mins),
        orb_risk_reward=float(orb_risk_reward),
        orb_target_mode=str(orb_target_mode),
        spot_exit_mode=str(spot_exit_mode),
        spot_atr_period=int(spot_atr_period),
        spot_pt_atr_mult=float(spot_pt_atr_mult),
        spot_sl_atr_mult=float(spot_sl_atr_mult),
        regime_mode="supertrend",
        supertrend_atr_period=5,
        supertrend_multiplier=0.4,
        supertrend_source="hl2",
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
    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def _mk_filters(
    *,
    ema_spread_min_pct: float | None = None,
    ema_slope_min_pct: float | None = None,
    cooldown_bars: int = 0,
    skip_first_bars: int = 0,
    volume_ratio_min: float | None = None,
    volume_ema_period: int | None = None,
    entry_start_hour_et: int | None = None,
    entry_end_hour_et: int | None = None,
) -> FiltersConfig | None:
    f = FiltersConfig(
        rv_min=None,
        rv_max=None,
        ema_spread_min_pct=ema_spread_min_pct,
        ema_slope_min_pct=ema_slope_min_pct,
        entry_start_hour=None,
        entry_end_hour=None,
        skip_first_bars=int(skip_first_bars),
        cooldown_bars=int(cooldown_bars),
        entry_start_hour_et=entry_start_hour_et,
        entry_end_hour_et=entry_end_hour_et,
        volume_ema_period=volume_ema_period,
        volume_ratio_min=volume_ratio_min,
    )
    if (
        f.rv_min is None
        and f.rv_max is None
        and f.ema_spread_min_pct is None
        and f.ema_slope_min_pct is None
        and f.entry_start_hour is None
        and f.entry_end_hour is None
        and f.entry_start_hour_et is None
        and f.entry_end_hour_et is None
        and f.skip_first_bars == 0
        and f.cooldown_bars == 0
        and f.volume_ratio_min is None
        and f.volume_ema_period is None
    ):
        return None
    return f


_WDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _entry_days_labels(days: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for d in days:
        try:
            idx = int(d)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(_WDAYS):
            out.append(_WDAYS[idx])
    return out


def _filters_payload(filters: FiltersConfig | None) -> dict | None:
    if filters is None:
        return None
    raw = asdict(filters)
    out: dict[str, object] = {}
    for key in ("rv_min", "rv_max", "ema_spread_min_pct", "ema_slope_min_pct", "volume_ratio_min"):
        if raw.get(key) is not None:
            out[key] = raw[key]
    if raw.get("volume_ratio_min") is not None and raw.get("volume_ema_period") is not None:
        out["volume_ema_period"] = raw["volume_ema_period"]
    if raw.get("entry_start_hour_et") is not None and raw.get("entry_end_hour_et") is not None:
        out["entry_start_hour_et"] = raw["entry_start_hour_et"]
        out["entry_end_hour_et"] = raw["entry_end_hour_et"]
    if raw.get("entry_start_hour") is not None and raw.get("entry_end_hour") is not None:
        out["entry_start_hour"] = raw["entry_start_hour"]
        out["entry_end_hour"] = raw["entry_end_hour"]
    if int(raw.get("skip_first_bars") or 0) > 0:
        out["skip_first_bars"] = int(raw["skip_first_bars"])
    if int(raw.get("cooldown_bars") or 0) > 0:
        out["cooldown_bars"] = int(raw["cooldown_bars"])
    return out or None


def _spot_strategy_payload(cfg: ConfigBundle, *, meta: ContractMeta) -> dict:
    strategy = asdict(cfg.strategy)
    strategy["entry_days"] = _entry_days_labels(cfg.strategy.entry_days)
    strategy["signal_bar_size"] = str(cfg.backtest.bar_size)
    strategy["signal_use_rth"] = bool(cfg.backtest.use_rth)
    strategy.pop("filters", None)

    # Ensure MNQ presets load as futures in the UI (otherwise `spot_sec_type` may default to STK).
    sym = str(cfg.strategy.symbol or "").strip().upper()
    if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
        strategy.setdefault("spot_sec_type", "FUT")
        strategy.setdefault("spot_exchange", str(meta.exchange or "CME"))
    else:
        strategy.setdefault("spot_sec_type", "STK")
        strategy.setdefault("spot_exchange", "SMART")
    return strategy


def _milestone_key(cfg: ConfigBundle) -> str:
    # Keep this stable and compact; include filters.
    strategy = asdict(cfg.strategy)
    filters = _filters_payload(cfg.strategy.filters)
    strategy["filters"] = filters
    strategy["signal_bar_size"] = str(cfg.backtest.bar_size)
    strategy["signal_use_rth"] = bool(cfg.backtest.use_rth)
    return json.dumps(strategy, sort_keys=True, default=str)


def _milestone_group_name(*, rank: int, cfg: ConfigBundle, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    strat = cfg.strategy
    rbar = str(getattr(strat, "regime_bar_size", "") or "").strip() or "?"
    tag = ""
    if str(getattr(strat, "regime_mode", "") or "").strip().lower() == "supertrend":
        tag = f"ST({getattr(strat,'supertrend_atr_period', '?')},{getattr(strat,'supertrend_multiplier','?')},{getattr(strat,'supertrend_source','?')})@{rbar}"
    elif getattr(strat, "regime_ema_preset", None):
        tag = f"EMA({getattr(strat,'regime_ema_preset','?')})@{rbar}"
    if str(getattr(strat, "regime2_mode", "off") or "off").strip().lower() != "off":
        r2bar = str(getattr(strat, "regime2_bar_size", "") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _milestone_group_name_from_strategy(*, rank: int, strategy: dict, metrics: dict, note: str | None) -> str:
    pnl = float(metrics.get("pnl") or 0.0)
    win = float(metrics.get("win_rate") or 0.0) * 100.0
    trades = int(metrics.get("trades") or 0)
    pnl_dd = float(metrics.get("pnl_over_dd") or 0.0)
    rbar = str(strategy.get("regime_bar_size") or "").strip() or "?"
    tag = ""
    if str(strategy.get("regime_mode") or "").strip().lower() == "supertrend":
        tag = (
            f"ST({strategy.get('supertrend_atr_period', '?')},"
            f"{strategy.get('supertrend_multiplier', '?')},"
            f"{strategy.get('supertrend_source', '?')})@{rbar}"
        )
    elif strategy.get("regime_ema_preset"):
        tag = f"EMA({strategy.get('regime_ema_preset', '?')})@{rbar}"
    if str(strategy.get("regime2_mode") or "off").strip().lower() != "off":
        r2bar = str(strategy.get("regime2_bar_size") or "").strip() or "?"
        tag += f" + R2@{r2bar}"
    base = f"Spot (MNQ) 12m (post-fix) #{rank:02d} pnl/dd={pnl_dd:.2f} pnl={pnl:.0f} win={win:.1f}% tr={trades}"
    if tag:
        base += f" — {tag}"
    if note:
        base += f" [{note}]"
    return base


def _score_row_pnl_dd(row: dict) -> tuple:
    return (
        float(row.get("pnl_over_dd") or float("-inf")),
        float(row.get("pnl") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _score_row_pnl(row: dict) -> tuple:
    return (
        float(row.get("pnl") or float("-inf")),
        float(row.get("pnl_over_dd") or 0.0),
        float(row.get("win_rate") or 0.0),
        int(row.get("trades") or 0),
    )


def _print_top(rows: list[dict], *, title: str, top_n: int, sort_key) -> None:
    print("")
    print(title)
    print("-" * len(title))
    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    for idx, row in enumerate(rows_sorted[: max(1, int(top_n))], start=1):
        pnl = float(row.get("pnl") or 0.0)
        dd = float(row.get("dd") or 0.0)
        trades = int(row.get("trades") or 0)
        win = float(row.get("win_rate") or 0.0) * 100.0
        pnl_over_dd = float(row.get("pnl_over_dd") or 0.0)
        note = row.get("note") or ""
        print(f"{idx:>2}. tr={trades:>4} win={win:>5.1f}% pnl={pnl:>10.1f} dd={dd:>8.1f} pnl/dd={pnl_over_dd:>6.2f} {note}")


def _print_leaderboards(rows: list[dict], *, title: str, top_n: int) -> None:
    _print_top(rows, title=f"{title} — Top by pnl/dd", top_n=top_n, sort_key=_score_row_pnl_dd)
    _print_top(rows, title=f"{title} — Top by pnl", top_n=top_n, sort_key=_score_row_pnl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled spot evolution sweeps (MNQ spot)")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--start", default="2025-01-08")
    parser.add_argument("--end", default="2026-01-08")
    parser.add_argument(
        "--bar-size",
        default="1 hour",
        help="Signal bar size (e.g. '30 mins', '1 hour'). ORB axis uses 15m regardless.",
    )
    parser.add_argument("--use-rth", action="store_true", default=False)
    parser.add_argument("--cache-dir", default="db")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Use cached bars only (no IBKR calls). Requires cache to be present.",
    )
    parser.add_argument(
        "--base",
        default="champion",
        choices=("champion", "dual_regime"),
        help="Select the base strategy shape to start from.",
    )
    parser.add_argument("--max-open-trades", type=int, default=2)
    parser.add_argument("--close-eod", action="store_true", default=False)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--top", type=int, default=15)
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
        help="Output path for --write-milestones",
    )
    parser.add_argument("--milestone-min-win", type=float, default=0.55)
    parser.add_argument("--milestone-min-trades", type=int, default=200)
    parser.add_argument("--milestone-min-pnl-dd", type=float, default=8.0)
    parser.add_argument(
        "--axis",
        default="all",
        choices=(
            "all",
            "ema",
            "combo",
            "volume",
            "tod",
            "atr",
            "ptsl",
            "hold",
            "orb",
            "regime",
            "regime2",
            "confirm",
            "spread",
            "slope",
            "cooldown",
            "skip_open",
            "loosen",
        ),
        help="Run one axis sweep (or all in sequence)",
    )
    args = parser.parse_args()

    symbol = str(args.symbol).strip().upper()
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    use_rth = bool(args.use_rth)
    offline = bool(args.offline)
    cache_dir = Path(args.cache_dir)
    start_dt = datetime.combine(start, time(0, 0))
    end_dt = datetime.combine(end, time(23, 59))
    signal_bar_size = str(args.bar_size).strip() or "1 hour"
    max_open_trades = int(args.max_open_trades)
    close_eod = bool(args.close_eod)
    run_min_trades = int(args.min_trades)
    if bool(args.write_milestones):
        run_min_trades = min(run_min_trades, int(args.milestone_min_trades))

    data = IBKRHistoricalData()
    if offline:
        is_future = symbol in ("MNQ", "MBT")
        exchange = "CME" if is_future else "SMART"
        multiplier = 1.0
        if is_future:
            multiplier = {"MNQ": 2.0, "MBT": 0.1}.get(symbol, 1.0)
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=0.01)
    else:
        _, meta = data.resolve_contract(symbol, exchange=None)

    def _bars(bar_size: str) -> list:
        if offline:
            return data.load_cached_bars(
                symbol=symbol,
                exchange=None,
                start=start_dt,
                end=end_dt,
                bar_size=str(bar_size),
                use_rth=use_rth,
                cache_dir=cache_dir,
            )
        return data.load_or_fetch_bars(
            symbol=symbol,
            exchange=None,
            start=start_dt,
            end=end_dt,
            bar_size=str(bar_size),
            use_rth=use_rth,
            cache_dir=cache_dir,
        )

    bar_cache: dict[str, list] = {}

    def _bars_cached(bar_size: str) -> list:
        key = str(bar_size)
        cached = bar_cache.get(key)
        if cached is not None:
            return cached
        loaded = _bars(key)
        bar_cache[key] = loaded
        return loaded

    regime_bars_1d = _bars_cached("1 day")
    if not regime_bars_1d:
        raise SystemExit("No 1 day regime bars returned (IBKR).")

    def _regime_bars_for(cfg: ConfigBundle) -> list | None:
        regime_bar = str(getattr(cfg.strategy, "regime_bar_size", "") or "").strip() or str(cfg.backtest.bar_size)
        if str(regime_bar) == str(cfg.backtest.bar_size):
            return None
        bars = _bars_cached(regime_bar)
        if not bars:
            raise SystemExit(f"No {regime_bar} regime bars returned (IBKR).")
        return bars

    def _regime2_bars_for(cfg: ConfigBundle) -> list | None:
        mode = str(getattr(cfg.strategy, "regime2_mode", "off") or "off").strip().lower()
        if mode == "off":
            return None
        regime_bar = str(getattr(cfg.strategy, "regime2_bar_size", "") or "").strip() or str(cfg.backtest.bar_size)
        if str(regime_bar) == str(cfg.backtest.bar_size):
            return None
        bars = _bars_cached(regime_bar)
        if not bars:
            raise SystemExit(f"No {regime_bar} regime2 bars returned (IBKR).")
        return bars

    def _run_cfg(
        *, cfg: ConfigBundle, bars: list, regime_bars: list | None, regime2_bars: list | None
    ) -> dict | None:
        out = _run_spot_backtest(cfg, bars, meta, regime_bars=regime_bars, regime2_bars=regime2_bars)
        s = out.summary
        if int(s.trades) < int(run_min_trades):
            return None
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        return {
            "trades": int(s.trades),
            "win_rate": float(s.win_rate),
            "pnl": pnl,
            "dd": dd,
            "pnl_over_dd": (pnl / dd) if dd > 0 else None,
        }

    def _base_bundle(*, bar_size: str, filters: FiltersConfig | None) -> ConfigBundle:
        cfg = _bundle_base(
            symbol=symbol,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
            offline=offline,
            filters=filters,
            max_open_trades=max_open_trades,
            spot_close_eod=close_eod,
        )
        if str(args.base).strip().lower() == "dual_regime":
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    regime2_mode="supertrend",
                    regime2_bar_size="4 hours",
                    regime2_supertrend_atr_period=2,
                    regime2_supertrend_multiplier=0.3,
                    regime2_supertrend_source="close",
                ),
            )
        return cfg

    milestone_rows: list[tuple[ConfigBundle, dict, str]] = []

    def _record_milestone(cfg: ConfigBundle, row: dict, note: str) -> None:
        if not bool(args.write_milestones):
            return
        milestone_rows.append((cfg, row, str(note)))

    def _sweep_volume() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        ratios = [None, 1.0, 1.1, 1.2, 1.5]
        periods = [10, 20, 30]
        rows: list[dict] = []
        for ratio in ratios:
            if ratio is None:
                variants = [(None, None)]
            else:
                variants = [(ratio, p) for p in periods]
            for ratio_min, ema_p in variants:
                f = _mk_filters(volume_ratio_min=ratio_min, volume_ema_period=ema_p)
                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                row = _run_cfg(
                    cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
                )
                if not row:
                    continue
                note = "-" if ratio_min is None else f"vol>={ratio_min}@{ema_p}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="A) Volume gate sweep", top_n=int(args.top))

    def _sweep_ema() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21"]
        rows: list[dict] = []
        for preset in presets:
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, ema_preset=str(preset)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"ema={preset}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="0) Timing sweep (EMA preset)", top_n=int(args.top))

    def _sweep_tod() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        windows = [
            (None, None, "base"),
            (9, 16, "RTH 9–16 ET"),
            (10, 15, "10–15 ET"),
            (11, 16, "11–16 ET"),
            (18, 3, "18–03 ET"),
        ]
        rows: list[dict] = []
        for start_h, end_h, label in windows:
            f = _mk_filters(entry_start_hour_et=start_h, entry_end_hour_et=end_h)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            row["note"] = label
            _record_milestone(cfg, row, label)
            rows.append(row)
        _print_leaderboards(rows, title="B) Time-of-day gate sweep (ET)", top_n=int(args.top))

    def _sweep_atr_exits() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        atr_periods = [7, 10, 14, 21]
        pt_mults = [1.0, 1.5, 2.0]
        sl_mults = [1.0, 1.5, 2.0]
        rows: list[dict] = []
        for atr_p in atr_periods:
            for pt_m in pt_mults:
                for sl_m in sl_mults:
                    cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                    cfg = replace(
                        cfg,
                        strategy=replace(
                            cfg.strategy,
                            spot_exit_mode="atr",
                            spot_atr_period=int(atr_p),
                            spot_pt_atr_mult=float(pt_m),
                            spot_sl_atr_mult=float(sl_m),
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=None,
                        ),
                    )
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=_regime2_bars_for(cfg),
                    )
                    if not row:
                        continue
                    note = f"ATR({atr_p}) PTx{pt_m} SLx{sl_m}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="C) ATR exits sweep (1h timing + 1d Supertrend)", top_n=int(args.top))

    def _sweep_ptsl() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        pt_vals = [0.005, 0.01, 0.015, 0.02]
        sl_vals = [0.015, 0.02, 0.03]
        rows: list[dict] = []
        for pt in pt_vals:
            for sl in sl_vals:
                cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                cfg = replace(
                    cfg,
                    strategy=replace(
                        cfg.strategy,
                        spot_profit_target_pct=float(pt),
                        spot_stop_loss_pct=float(sl),
                        spot_exit_mode="pct",
                    ),
                )
                row = _run_cfg(
                    cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
                )
                if not row:
                    continue
                note = f"PT={pt:.3f} SL={sl:.3f}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="PT/SL sweep (fixed pct exits)", top_n=int(args.top))

    def _sweep_hold() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for hold in (0, 1, 2, 3, 4, 6, 8):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, flip_exit_min_hold_bars=int(hold)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"hold={hold}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Flip-exit min hold sweep", top_n=int(args.top))

    def _sweep_orb() -> None:
        bars_15m = _bars_cached("15 mins")
        rows: list[dict] = []
        # Include classic fib ratios as well (0.618, 1.618) since ORB stop is the opposite OR extreme.
        rr_vals = [0.618, 1.0, 1.5, 1.618, 2.0, 2.5, 3.0]
        vol_vals = [None, 1.0, 1.2, 1.5]
        for target_mode in ("rr", "or_range"):
            for rr in rr_vals:
                for vol_min in vol_vals:
                    f = _mk_filters(volume_ratio_min=vol_min, volume_ema_period=20 if vol_min is not None else None)
                    cfg = _base_bundle(bar_size="15 mins", filters=f)
                    cfg = replace(
                        cfg,
                        strategy=replace(
                            cfg.strategy,
                            entry_signal="orb",
                            ema_preset=None,
                            entry_confirm_bars=0,
                            orb_window_mins=15,
                            orb_risk_reward=float(rr),
                            orb_target_mode=str(target_mode),
                            spot_profit_target_pct=None,
                            spot_stop_loss_pct=None,
                        ),
                    )
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_15m,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=_regime2_bars_for(cfg),
                    )
                    if not row:
                        continue
                    vol_note = "-" if vol_min is None else f"vol>={vol_min}@20"
                    note = f"ORB 15m {target_mode} rr={rr} {vol_note}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="D) ORB sweep (15m timing + 1d Supertrend)", top_n=int(args.top))

    def _sweep_regime() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        regime_bars_4h = _bars_cached("4 hours")
        if not regime_bars_4h:
            raise SystemExit("No 4 hours regime bars returned (IBKR).")

        rows: list[dict] = []
        regime_bars_by_size = {"4 hours": regime_bars_4h, "1 day": regime_bars_1d}
        regime_bar_sizes = ["4 hours", "1 day"]
        # Include both the newer “micro” ST params and some legacy/wider values we’ve
        # historically tested (e.g. mult=0.8 or 2.0) so the unified sweeps cover them.
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11, 14, 21]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        sources = ["close", "hl2"]
        for rbar in regime_bar_sizes:
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                        cfg = replace(
                            cfg,
                            strategy=replace(
                                cfg.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=rbar,
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                            ),
                        )
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=regime_bars_by_size[rbar],
                            regime2_bars=_regime2_bars_for(cfg),
                        )
                        if not row:
                            continue
                        note = f"ST({atr_p},{mult},{src}) @{rbar}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        _print_leaderboards(rows, title="Regime sweep (Supertrend params + timeframe)", top_n=int(args.top))

    def _sweep_regime2() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        regime2_bars_4h = _bars_cached("4 hours")
        if not regime2_bars_4h:
            raise SystemExit("No 4 hours regime2 bars returned (IBKR).")

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11, 14, 21]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        sources = ["close", "hl2"]
        for atr_p in atr_periods:
            for mult in multipliers:
                for src in sources:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            regime2_mode="supertrend",
                            regime2_bar_size="4 hours",
                            regime2_supertrend_atr_period=int(atr_p),
                            regime2_supertrend_multiplier=float(mult),
                            regime2_supertrend_source=str(src),
                        ),
                    )
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=regime2_bars_4h,
                    )
                    if not row:
                        continue
                    note = f"ST2(4h:{atr_p},{mult},{src})"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Dual regime sweep (regime2 Supertrend @ 4h)", top_n=int(args.top))

    def _sweep_confirm() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for confirm in (0, 1, 2, 3):
            cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
            cfg = replace(cfg, strategy=replace(cfg.strategy, entry_confirm_bars=int(confirm)))
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"confirm={confirm}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Confirm-bars sweep (quality gate)", top_n=int(args.top))

    def _sweep_spread() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for spread in (None, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
            f = _mk_filters(ema_spread_min_pct=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            spread_note = "-" if spread is None else f"spread>={spread}"
            row["note"] = spread_note
            _record_milestone(cfg, row, spread_note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread sweep (quality gate)", top_n=int(args.top))

    def _sweep_slope() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for slope in (None, 0.005, 0.01, 0.02, 0.03, 0.05):
            f = _mk_filters(ema_slope_min_pct=float(slope) if slope is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = "-" if slope is None else f"slope>={slope}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA slope sweep (quality gate)", top_n=int(args.top))

    def _sweep_cooldown() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for cooldown in (0, 1, 2, 3, 4, 6, 8):
            f = _mk_filters(cooldown_bars=int(cooldown))
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"cooldown={cooldown}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Cooldown sweep (quality gate)", top_n=int(args.top))

    def _sweep_skip_open() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for skip in (0, 1, 2, 3, 4, 6):
            f = _mk_filters(skip_first_bars=int(skip))
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = f"skip_first={skip}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="Skip-open sweep (quality gate)", top_n=int(args.top))

    def _sweep_loosen() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        for max_open in (1, 2, 3, 0):
            for close_eod in (False, True):
                cfg = _base_bundle(bar_size=signal_bar_size, filters=None)
                cfg = replace(
                    cfg,
                    strategy=replace(
                        cfg.strategy,
                        max_open_trades=int(max_open),
                        spot_close_eod=bool(close_eod),
                    ),
                )
                row = _run_cfg(
                    cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
                )
                if not row:
                    continue
                note = f"max_open={max_open} close_eod={int(close_eod)}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        _print_leaderboards(rows, title="Loosenings sweep (stacking + EOD exit)", top_n=int(args.top))

    def _sweep_combo() -> None:
        """A constrained multi-axis sweep to find "corner" winners.

        Keep this computationally bounded and reproducible. The intent is to combine
        the highest-leverage levers we’ve found so far:
        - regime sensitivity (Supertrend timeframe + params)
        - exits (pct vs ATR)
        - loosenings (stacking + EOD close)
        - optional regime2 confirm (small curated set)
        - a small set of quality gates (volume/spread/cooldown/skip-open/confirm/TOD)
        """
        bars_sig = _bars_cached(signal_bar_size)

        regime_bars_4h = _bars_cached("4 hours")
        if not regime_bars_4h:
            raise SystemExit("No 4 hours regime bars returned (IBKR).")

        regime_bars_by_size = {"4 hours": regime_bars_4h, "1 day": regime_bars_1d}

        def _ranked(items: list[tuple[ConfigBundle, dict, str]], *, top_pnl_dd: int, top_pnl: int) -> list:
            by_dd = sorted(items, key=lambda t: _score_row_pnl_dd(t[1]), reverse=True)[: int(top_pnl_dd)]
            by_pnl = sorted(items, key=lambda t: _score_row_pnl(t[1]), reverse=True)[: int(top_pnl)]
            seen: set[str] = set()
            out: list[tuple[ConfigBundle, dict, str]] = []
            for cfg, row, note in by_dd + by_pnl:
                key = _milestone_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                out.append((cfg, row, note))
            return out

        # Stage 1: sweep regime sensitivity only (broad) and keep a small diverse shortlist.
        stage1: list[tuple[ConfigBundle, dict, str]] = []
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        regime_bar_sizes = ["4 hours", "1 day"]
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11, 14, 21]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
        sources = ["close", "hl2"]
        for rbar in regime_bar_sizes:
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=rbar,
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                                regime2_mode="off",
                            ),
                        )
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=regime_bars_by_size[rbar],
                            regime2_bars=None,
                        )
                        if not row:
                            continue
                        note = f"ST({atr_p},{mult},{src}) @{rbar}"
                        row["note"] = note
                        stage1.append((cfg, row, note))

        shortlist = _ranked(stage1, top_pnl_dd=20, top_pnl=10)
        print("")
        print(f"Combo sweep: shortlist regimes={len(shortlist)} (from stage1={len(stage1)})")

        # Stage 2: for each shortlisted regime, sweep exits + loosenings, and (optionally) a small regime2 set.
        exit_variants: list[tuple[dict, str]] = []
        for pt, sl in (
            (0.005, 0.02),
            (0.005, 0.03),
            (0.01, 0.03),
            (0.015, 0.03),
        ):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": float(pt),
                        "spot_stop_loss_pct": float(sl),
                        "spot_atr_period": 14,
                        "spot_pt_atr_mult": 1.5,
                        "spot_sl_atr_mult": 1.0,
                    },
                    f"PT={pt:.3f} SL={sl:.3f}",
                )
            )
        for atr_p, pt_m, sl_m in (
            (7, 1.0, 1.0),
            (7, 1.0, 1.5),
            (14, 1.0, 1.0),
            (14, 1.0, 1.5),
        ):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "atr",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": None,
                        "spot_atr_period": int(atr_p),
                        "spot_pt_atr_mult": float(pt_m),
                        "spot_sl_atr_mult": float(sl_m),
                    },
                    f"ATR({atr_p}) PTx{pt_m} SLx{sl_m}",
                )
            )

        # Keep this small; we already have a dedicated loosenings axis. Here we only try
        # a few representative "stacking vs risk trimming" variants.
        loosen_variants: list[tuple[int, bool, str]] = [
            (2, False, "max_open=2 close_eod=0"),
            (0, False, "max_open=0 close_eod=0"),
            (1, True, "max_open=1 close_eod=1"),
        ]

        hold_vals = (0, 4)

        regime2_variants: list[tuple[dict, str]] = [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "no_r2"),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
                "ST2(4h:3,0.25,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 5,
                    "regime2_supertrend_multiplier": 0.2,
                    "regime2_supertrend_source": "close",
                },
                "ST2(4h:5,0.2,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "1 day",
                    "regime2_supertrend_atr_period": 7,
                    "regime2_supertrend_multiplier": 0.4,
                    "regime2_supertrend_source": "close",
                },
                "ST2(1d:7,0.4,close)",
            ),
        ]

        stage2: list[tuple[ConfigBundle, dict, str]] = []
        tested = 0
        for base_cfg, _, base_note in shortlist:
            for exit_over, exit_note in exit_variants:
                for hold in hold_vals:
                    for max_open, close_eod, loose_note in loosen_variants:
                        for r2_over, r2_note in regime2_variants:
                            strat = base_cfg.strategy
                            cfg = replace(
                                base_cfg,
                                strategy=replace(
                                    strat,
                                    spot_exit_mode=str(exit_over["spot_exit_mode"]),
                                    spot_profit_target_pct=exit_over["spot_profit_target_pct"],
                                    spot_stop_loss_pct=exit_over["spot_stop_loss_pct"],
                                    spot_atr_period=int(exit_over["spot_atr_period"]),
                                    spot_pt_atr_mult=float(exit_over["spot_pt_atr_mult"]),
                                    spot_sl_atr_mult=float(exit_over["spot_sl_atr_mult"]),
                                    flip_exit_min_hold_bars=int(hold),
                                    max_open_trades=int(max_open),
                                    spot_close_eod=bool(close_eod),
                                    regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                                    regime2_bar_size=r2_over.get("regime2_bar_size"),
                                    regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                                    regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                                    regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                                ),
                            )
                            row = _run_cfg(
                                cfg=cfg,
                                bars=bars_sig,
                                regime_bars=_regime_bars_for(cfg) or regime_bars_by_size[str(cfg.strategy.regime_bar_size)],
                                regime2_bars=_regime2_bars_for(cfg),
                            )
                            tested += 1
                            if not row:
                                continue
                            note = f"{base_note} | {exit_note} | hold={hold} | {loose_note} | {r2_note}"
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            stage2.append((cfg, row, note))

        print(f"Combo sweep: stage2 tested={tested} kept={len(stage2)} (min_trades={run_min_trades})")

        # Stage 3: apply a small set of quality gates on the top stage2 candidates.
        top_stage2 = _ranked(stage2, top_pnl_dd=20, top_pnl=10)
        vol_variants = [(None, None, "-"), (1.2, 20, "vol>=1.2@20")]
        spread_variants = [(None, "-"), (0.01, "spread>=0.01")]
        cooldown_variants = [(0, "cooldown=0"), (4, "cooldown=4")]
        skip_variants = [(0, "skip=0"), (1, "skip=1")]
        confirm_variants = [(0, "confirm=0"), (1, "confirm=1")]
        tod_variants = [
            (None, None, "tod=base"),
            (18, 3, "tod=18-03 ET"),
        ]

        stage3: list[dict] = []
        for base_cfg, base_row, base_note in top_stage2:
            for vratio, vema, vnote in vol_variants:
                for spread, spread_note in spread_variants:
                    for cooldown, cd_note in cooldown_variants:
                        for skip, skip_note in skip_variants:
                            for confirm, confirm_note in confirm_variants:
                                for tod_s, tod_e, tod_note in tod_variants:
                                    f = _mk_filters(
                                        volume_ratio_min=vratio,
                                        volume_ema_period=vema,
                                        ema_spread_min_pct=spread,
                                        cooldown_bars=int(cooldown),
                                        skip_first_bars=int(skip),
                                        entry_start_hour_et=tod_s,
                                        entry_end_hour_et=tod_e,
                                    )
                                    cfg = replace(
                                        base_cfg,
                                        strategy=replace(
                                            base_cfg.strategy,
                                            filters=f,
                                            entry_confirm_bars=int(confirm),
                                        ),
                                    )
                                    row = _run_cfg(
                                        cfg=cfg,
                                        bars=bars_sig,
                                        regime_bars=_regime_bars_for(cfg) or regime_bars_by_size[
                                            str(cfg.strategy.regime_bar_size)
                                        ],
                                        regime2_bars=_regime2_bars_for(cfg),
                                    )
                                    if not row:
                                        continue
                                    note = f"{base_note} | {vnote} {spread_note} {cd_note} {skip_note} {confirm_note} {tod_note}"
                                    row["note"] = note
                                    _record_milestone(cfg, row, note)
                                    stage3.append(row)

        _print_leaderboards(stage3, title="Combo sweep (multi-axis, constrained)", top_n=int(args.top))

    axis = str(args.axis).strip().lower()
    print(
        f"{symbol} spot evolve sweep ({start.isoformat()} -> {end.isoformat()}, use_rth={use_rth}, "
        f"bar_size={signal_bar_size}, offline={offline}, base={args.base}, axis={axis})"
    )

    if axis in ("all", "ema"):
        _sweep_ema()
    if axis == "combo":
        _sweep_combo()
    if axis in ("all", "volume"):
        _sweep_volume()
    if axis in ("all", "tod"):
        _sweep_tod()
    if axis in ("all", "atr"):
        _sweep_atr_exits()
    if axis in ("all", "ptsl"):
        _sweep_ptsl()
    if axis in ("all", "hold"):
        _sweep_hold()
    if axis in ("all", "orb"):
        _sweep_orb()
    if axis in ("all", "regime"):
        _sweep_regime()
    if axis in ("all", "regime2"):
        _sweep_regime2()
    if axis in ("all", "confirm"):
        _sweep_confirm()
    if axis in ("all", "spread"):
        _sweep_spread()
    if axis in ("all", "slope"):
        _sweep_slope()
    if axis in ("all", "cooldown"):
        _sweep_cooldown()
    if axis in ("all", "skip_open"):
        _sweep_skip_open()
    if axis in ("all", "loosen"):
        _sweep_loosen()

    if bool(args.write_milestones):
        eligible: list[dict] = []
        for cfg, row, note in milestone_rows:
            try:
                win = float(row.get("win_rate") or 0.0)
            except (TypeError, ValueError):
                win = 0.0
            try:
                trades = int(row.get("trades") or 0)
            except (TypeError, ValueError):
                trades = 0
            pnl_dd_raw = row.get("pnl_over_dd")
            try:
                pnl_dd = float(pnl_dd_raw) if pnl_dd_raw is not None else None
            except (TypeError, ValueError):
                pnl_dd = None
            if win < float(args.milestone_min_win):
                continue
            if trades < int(args.milestone_min_trades):
                continue
            if pnl_dd is None or pnl_dd < float(args.milestone_min_pnl_dd):
                continue
            strategy = _spot_strategy_payload(cfg, meta=meta)
            filters = _filters_payload(cfg.strategy.filters)
            key_obj = dict(strategy)
            key_obj["filters"] = filters
            eligible.append(
                {
                    "key": json.dumps(key_obj, sort_keys=True, default=str),
                    "strategy": strategy,
                    "filters": filters,
                    "note": note,
                    "metrics": {
                        "pnl": float(row.get("pnl") or 0.0),
                        "win_rate": float(row.get("win_rate") or 0.0),
                        "trades": int(row.get("trades") or 0),
                        "max_drawdown": float(row.get("dd") or 0.0),
                        "pnl_over_dd": row.get("pnl_over_dd"),
                    },
                }
            )

        out_path = Path(args.milestones_out)
        if bool(args.merge_milestones) and out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                existing = {}
            for group in existing.get("groups") or []:
                filters = group.get("filters")
                raw_name = str(group.get("name") or "")
                parsed_note = None
                if raw_name.endswith("]") and "[" in raw_name:
                    try:
                        parsed_note = raw_name[raw_name.rfind("[") + 1 : -1].strip() or None
                    except Exception:
                        parsed_note = None
                for entry in group.get("entries") or []:
                    strat = entry.get("strategy") or {}
                    metrics = entry.get("metrics") or {}
                    key_obj = dict(strat)
                    key_obj["filters"] = filters
                    eligible.append(
                        {
                            "key": json.dumps(key_obj, sort_keys=True, default=str),
                            "strategy": strat,
                            "filters": filters,
                            "note": parsed_note,
                            "metrics": {
                                "pnl": float(metrics.get("pnl") or 0.0),
                                "win_rate": float(metrics.get("win_rate") or 0.0),
                                "trades": int(metrics.get("trades") or 0),
                                "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
                                "pnl_over_dd": metrics.get("pnl_over_dd"),
                            },
                        }
                    )

        def _sort_key(item: dict) -> tuple:
            m = item.get("metrics") or {}
            return (
                float(m.get("pnl_over_dd") or float("-inf")),
                float(m.get("pnl") or 0.0),
                float(m.get("win_rate") or 0.0),
                int(m.get("trades") or 0),
            )

        best_by_key: dict[str, dict] = {}
        for item in eligible:
            key = str(item.get("key") or "")
            if not key:
                continue
            current = best_by_key.get(key)
            if current is None or _sort_key(item) > _sort_key(current):
                best_by_key[key] = item

        unique = sorted(best_by_key.values(), key=_sort_key, reverse=True)
        groups: list[dict] = []
        for idx, item in enumerate(unique, start=1):
            metrics = item["metrics"]
            entry = {"symbol": symbol, "metrics": metrics, "strategy": item["strategy"]}
            groups.append(
                {
                    "name": _milestone_group_name_from_strategy(
                        rank=idx, strategy=item["strategy"], metrics=metrics, note=str(item.get("note") or "").strip()
                    ),
                    "filters": item["filters"],
                    "entries": [entry],
                }
            )

        payload = {
            "name": "spot_milestones",
            "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "notes": (
                f"Auto-generated via evolve_spot.py (post-fix). "
                f"window={start.isoformat()}→{end.isoformat()}, bar_size={signal_bar_size}, use_rth={use_rth}. "
                f"thresholds: win>={float(args.milestone_min_win):.2f}, trades>={int(args.milestone_min_trades)}, "
                f"pnl/dd>={float(args.milestone_min_pnl_dd):.2f}."
            ),
            "groups": groups,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
        print(f"Wrote {out_path} ({len(groups)} eligible presets).")

    if not offline:
        data.disconnect()


if __name__ == "__main__":
    main()
