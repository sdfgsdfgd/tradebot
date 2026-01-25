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
  H) $TICK width gate (Raschke-style)
  I) Joint sweeps (interaction hunts)

All knobs are opt-in; default bot behavior is unchanged.
"""

from __future__ import annotations

import argparse
import json
import time as pytime
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .config import BacktestConfig, ConfigBundle, FiltersConfig, SpotLegConfig, StrategyConfig, SyntheticConfig
from .data import ContractMeta, IBKRHistoricalData, _find_covering_cache_path
from .engine import _run_spot_backtest
from ..signals import parse_bar_size


def _parse_date(value: str) -> date:
    year_s, month_s, day_s = str(value).strip().split("-")
    return date(int(year_s), int(month_s), int(day_s))


def _expected_cache_path(
    *,
    cache_dir: Path,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
) -> Path:
    tag = "rth" if use_rth else "full"
    safe_bar = str(bar_size).replace(" ", "")
    return cache_dir / symbol / f"{symbol}_{start_dt.date()}_{end_dt.date()}_{safe_bar}_{tag}.csv"


def _require_offline_cache_or_die(
    *,
    cache_dir: Path,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
) -> None:
    covering = _find_covering_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    if covering is not None:
        return
    expected = _expected_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start_dt=start_dt,
        end_dt=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    tag = "rth" if use_rth else "full"
    raise SystemExit(
        f"--offline was requested, but cached bars are missing for {symbol} {bar_size} {tag} "
        f"{start_dt.date().isoformat()}→{end_dt.date().isoformat()} (expected: {expected}). "
        "Re-run without --offline to fetch via IBKR (or prefetch the cache first)."
    )


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
        flip_exit_gate_mode="off",
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
    rv_min: float | None = None,
    rv_max: float | None = None,
    ema_spread_min_pct: float | None = None,
    ema_spread_min_pct_down: float | None = None,
    ema_slope_min_pct: float | None = None,
    cooldown_bars: int = 0,
    skip_first_bars: int = 0,
    volume_ratio_min: float | None = None,
    volume_ema_period: int | None = None,
    entry_start_hour_et: int | None = None,
    entry_end_hour_et: int | None = None,
) -> FiltersConfig | None:
    f = FiltersConfig(
        rv_min=rv_min,
        rv_max=rv_max,
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
        ema_spread_min_pct_down=ema_spread_min_pct_down,
    )
    if (
        f.rv_min is None
        and f.rv_max is None
        and f.ema_spread_min_pct is None
        and f.ema_spread_min_pct_down is None
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
    for key in (
        "rv_min",
        "rv_max",
        "ema_spread_min_pct",
        "ema_spread_min_pct_down",
        "ema_slope_min_pct",
        "volume_ratio_min",
    ):
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
        roi = float(row.get("roi") or 0.0) * 100.0
        dd_pct = float(row.get("dd_pct") or 0.0) * 100.0
        trades = int(row.get("trades") or 0)
        win = float(row.get("win_rate") or 0.0) * 100.0
        pnl_over_dd = float(row.get("pnl_over_dd") or 0.0)
        note = row.get("note") or ""
        print(
            f"{idx:>2}. tr={trades:>4} win={win:>5.1f}% "
            f"pnl={pnl:>10.1f} dd={dd:>8.1f} pnl/dd={pnl_over_dd:>6.2f} "
            f"roi={roi:>6.2f}% dd%={dd_pct:>6.2f}% {note}"
        )


def _print_leaderboards(rows: list[dict], *, title: str, top_n: int) -> None:
    _print_top(rows, title=f"{title} — Top by pnl/dd", top_n=top_n, sort_key=_score_row_pnl_dd)
    _print_top(rows, title=f"{title} — Top by pnl", top_n=top_n, sort_key=_score_row_pnl)


def _load_spot_milestones() -> dict | None:
    path = Path(__file__).resolve().parent / "spot_milestones.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _milestone_entry_for(
    milestones: dict | None,
    *,
    symbol: str,
    signal_bar_size: str,
    use_rth: bool,
    sort_by: str,
    prefer_realism: bool = False,
) -> tuple[dict, dict | None, dict] | None:
    if not milestones:
        return None

    groups = milestones.get("groups") or []
    candidates: list[tuple[dict, dict | None, dict]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("entries") or []
        if not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        strategy = entry.get("strategy") or {}
        metrics = entry.get("metrics") or {}
        if not isinstance(strategy, dict) or not isinstance(metrics, dict):
            continue
        if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
            continue
        if str(strategy.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
            continue
        if bool(strategy.get("signal_use_rth")) != bool(use_rth):
            continue
        if prefer_realism:
            if str(strategy.get("spot_entry_fill_mode") or "").strip().lower() != "next_open":
                continue
            if not bool(strategy.get("spot_intrabar_exits")):
                continue
            if int(strategy.get("max_open_trades") or 0) == 0:
                continue
        filters = group.get("filters")
        candidates.append((strategy, filters if isinstance(filters, dict) else None, metrics))

    if not candidates:
        return None

    def _score(c: tuple[dict, dict | None, dict]) -> tuple:
        _, _, m = c
        if str(sort_by).strip().lower() == "pnl":
            return _score_row_pnl(m)
        return _score_row_pnl_dd(m)

    return sorted(candidates, key=_score, reverse=True)[0]


def _apply_milestone_base(
    cfg: ConfigBundle, *, strategy: dict, filters: dict | None
) -> ConfigBundle:
    # Milestone strategies come from `asdict(StrategyConfig)` which flattens nested dataclasses.
    # Only copy scalar knobs we know are safe/needed for backtest reproduction/sweeps.
    keep_keys = (
        "ema_preset",
        "ema_entry_mode",
        "entry_confirm_bars",
        "entry_signal",
        "orb_window_mins",
        "orb_risk_reward",
        "orb_target_mode",
        "orb_open_time_et",
        "regime_mode",
        "regime_bar_size",
        "regime_ema_preset",
        "supertrend_atr_period",
        "supertrend_multiplier",
        "supertrend_source",
        "regime2_mode",
        "regime2_bar_size",
        "regime2_ema_preset",
        "regime2_supertrend_atr_period",
        "regime2_supertrend_multiplier",
        "regime2_supertrend_source",
        "spot_exit_mode",
        "spot_atr_period",
        "spot_pt_atr_mult",
        "spot_sl_atr_mult",
        "spot_profit_target_pct",
        "spot_stop_loss_pct",
        "spot_exit_time_et",
        "spot_close_eod",
        "spot_entry_fill_mode",
        "spot_flip_exit_fill_mode",
        "spot_intrabar_exits",
        "spot_spread",
        "spot_commission_per_share",
        "spot_commission_min",
        "spot_slippage_per_share",
        "spot_mark_to_market",
        "spot_drawdown_mode",
        "spot_sizing_mode",
        "spot_notional_pct",
        "spot_risk_pct",
        "spot_max_notional_pct",
        "spot_min_qty",
        "spot_max_qty",
        "exit_on_signal_flip",
        "flip_exit_mode",
        "flip_exit_min_hold_bars",
        "flip_exit_only_if_profit",
        "max_open_trades",
        "tick_gate_mode",
        "tick_gate_symbol",
        "tick_gate_exchange",
        "tick_band_ma_period",
        "tick_width_z_lookback",
        "tick_width_z_enter",
        "tick_width_z_exit",
        "tick_width_slope_lookback",
        "tick_neutral_policy",
        "tick_direction_policy",
    )

    strat_over: dict[str, object] = {}
    for key in keep_keys:
        if key in strategy:
            strat_over[key] = strategy[key]

    out = replace(cfg, strategy=replace(cfg.strategy, **strat_over))

    if not filters:
        return replace(out, strategy=replace(out.strategy, filters=None))

    f = _mk_filters(
        ema_spread_min_pct=filters.get("ema_spread_min_pct"),
        ema_spread_min_pct_down=filters.get("ema_spread_min_pct_down"),
        ema_slope_min_pct=filters.get("ema_slope_min_pct"),
        cooldown_bars=int(filters.get("cooldown_bars") or 0),
        skip_first_bars=int(filters.get("skip_first_bars") or 0),
        volume_ratio_min=filters.get("volume_ratio_min"),
        volume_ema_period=filters.get("volume_ema_period"),
        entry_start_hour_et=filters.get("entry_start_hour_et"),
        entry_end_hour_et=filters.get("entry_end_hour_et"),
    )
    return replace(out, strategy=replace(out.strategy, filters=f))


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
    parser.add_argument(
        "--spot-exec-bar-size",
        default=None,
        help="Optional execution bar size for spot simulation (e.g. '5 mins'). Signals still run on --bar-size.",
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
        choices=("default", "champion", "champion_pnl", "dual_regime"),
        help="Select the base strategy shape to start from (champion comes from spot_milestones.json).",
    )
    parser.add_argument("--max-open-trades", type=int, default=2)
    parser.add_argument("--close-eod", action="store_true", default=False)
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=False,
        help="Force spot to long-only (directional_spot = {'up': BUY 1}, no shorts).",
    )
    parser.add_argument(
        "--realism",
        action="store_true",
        default=False,
        help=(
            "Enable spot realism v1: entry fills at next bar open, flip exits fill at next bar open, "
            "intrabar PT/SL using OHLC, liquidation (bid/ask) marking, intrabar drawdown. "
            "Defaults to spread=$0.01/share and commission=$0.00/share unless overridden."
        ),
    )
    parser.add_argument(
        "--realism2",
        action="store_true",
        default=False,
        help=(
            "Enable spot realism v2 (superset of v1): adds position sizing (ROI-based), "
            "commission minimums, and stop gap handling. Defaults: spread=$0.01, "
            "commission=$0.005/share (min $1.00), risk sizing=1%% equity risk, max notional=50%%."
        ),
    )
    parser.add_argument("--spot-spread", type=float, default=None, help="Spot spread in price units (e.g. 0.01)")
    parser.add_argument(
        "--spot-commission",
        type=float,
        default=None,
        help="Spot commission per share/contract (price units). (Backtest-only; embedded into fills.)",
    )
    parser.add_argument(
        "--spot-commission-min",
        type=float,
        default=None,
        help="Spot commission minimum per order (price units), e.g. 1.0 for $1.00 min.",
    )
    parser.add_argument(
        "--spot-slippage",
        type=float,
        default=None,
        help="Spot slippage per share (price units). Applied on entry/stop/flip (market-like fills).",
    )
    parser.add_argument(
        "--spot-sizing-mode",
        default=None,
        choices=("fixed", "notional_pct", "risk_pct"),
        help="Spot sizing mode (v2): fixed qty, %% notional, or %% equity risk-to-stop.",
    )
    parser.add_argument("--spot-risk-pct", type=float, default=None, help="Risk per trade as fraction of equity (v2).")
    parser.add_argument(
        "--spot-notional-pct",
        type=float,
        default=None,
        help="Notional allocation per trade as fraction of equity (v2).",
    )
    parser.add_argument(
        "--spot-max-notional-pct",
        type=float,
        default=None,
        help="Max notional per trade as fraction of equity (v2).",
    )
    parser.add_argument("--spot-min-qty", type=int, default=None, help="Min shares per trade (v2).")
    parser.add_argument("--spot-max-qty", type=int, default=None, help="Max shares per trade, 0=none (v2).")
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
        "--milestone-add-top-pnl-dd",
        type=int,
        default=0,
        help=(
            "When used with --merge-milestones, limits how many NEW presets from this run are added "
            "(top by pnl/dd). 0 = no limit."
        ),
    )
    parser.add_argument(
        "--milestone-add-top-pnl",
        type=int,
        default=0,
        help=(
            "When used with --merge-milestones, limits how many NEW presets from this run are added "
            "(top by pnl). 0 = no limit."
        ),
    )
    parser.add_argument(
        "--axis",
        default="all",
        choices=(
            "all",
            "ema",
            "entry_mode",
            "combo",
            "squeeze",
            "volume",
            "rv",
            "tod",
            "tod_interaction",
            "perm_joint",
            "weekday",
            "exit_time",
            "atr",
            "atr_fine",
            "atr_ultra",
            "r2_atr",
            "r2_tod",
            "ema_perm_joint",
            "tick_perm_joint",
            "regime_atr",
            "ema_regime",
            "chop_joint",
            "ema_atr",
            "tick_ema",
            "ptsl",
            "hold",
            "orb",
            "orb_joint",
            "frontier",
            "regime",
            "regime2",
            "regime2_ema",
            "joint",
            "micro_st",
            "flip_exit",
            "confirm",
            "spread",
            "spread_fine",
            "spread_down",
            "slope",
            "cooldown",
            "skip_open",
            "loosen",
            "loosen_atr",
            "tick",
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
    spot_exec_bar_size = str(args.spot_exec_bar_size).strip() if args.spot_exec_bar_size else None
    if spot_exec_bar_size and parse_bar_size(spot_exec_bar_size) is None:
        raise SystemExit(f"Invalid --spot-exec-bar-size: {spot_exec_bar_size!r}")
    max_open_trades = int(args.max_open_trades)
    close_eod = bool(args.close_eod)
    long_only = bool(args.long_only)
    realism2 = bool(args.realism2)
    realism = bool(args.realism) or realism2
    spot_spread = float(args.spot_spread) if args.spot_spread is not None else (0.01 if realism else 0.0)
    spot_commission = (
        float(args.spot_commission)
        if args.spot_commission is not None
        else (0.005 if realism2 else 0.0)
    )
    spot_commission_min = (
        float(args.spot_commission_min)
        if args.spot_commission_min is not None
        else (1.0 if realism2 else 0.0)
    )
    spot_slippage = float(args.spot_slippage) if args.spot_slippage is not None else 0.0

    sizing_mode = (
        str(args.spot_sizing_mode).strip().lower()
        if args.spot_sizing_mode is not None
        else ("risk_pct" if realism2 else "fixed")
    )
    if sizing_mode not in ("fixed", "notional_pct", "risk_pct"):
        sizing_mode = "fixed"
    spot_risk_pct = float(args.spot_risk_pct) if args.spot_risk_pct is not None else (0.01 if realism2 else 0.0)
    spot_notional_pct = (
        float(args.spot_notional_pct) if args.spot_notional_pct is not None else 0.0
    )
    spot_max_notional_pct = (
        float(args.spot_max_notional_pct) if args.spot_max_notional_pct is not None else (0.50 if realism2 else 1.0)
    )
    spot_min_qty = int(args.spot_min_qty) if args.spot_min_qty is not None else 1
    spot_max_qty = int(args.spot_max_qty) if args.spot_max_qty is not None else 0
    run_min_trades = int(args.min_trades)
    if bool(args.write_milestones):
        run_min_trades = min(run_min_trades, int(args.milestone_min_trades))

    if offline:
        _require_offline_cache_or_die(
            cache_dir=cache_dir,
            symbol=symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=signal_bar_size,
            use_rth=use_rth,
        )
        if spot_exec_bar_size and str(spot_exec_bar_size) != str(signal_bar_size):
            _require_offline_cache_or_die(
                cache_dir=cache_dir,
                symbol=symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=spot_exec_bar_size,
                use_rth=use_rth,
            )

    data = IBKRHistoricalData()
    if offline:
        is_future = symbol in ("MNQ", "MBT")
        exchange = "CME" if is_future else "SMART"
        multiplier = 1.0
        if is_future:
            multiplier = {"MNQ": 2.0, "MBT": 0.1}.get(symbol, 1.0)
        meta = ContractMeta(symbol=symbol, exchange=exchange, multiplier=multiplier, min_tick=0.01)
    else:
        try:
            _, meta = data.resolve_contract(symbol, exchange=None)
        except Exception as exc:
            raise SystemExit(
                "IBKR API connection failed. Start IB Gateway / TWS (or run with --offline after prefetching cached bars)."
            ) from exc

    milestones = _load_spot_milestones()

    def _merge_filters(base_filters: FiltersConfig | None, *, overrides: dict[str, object]) -> FiltersConfig | None:
        """Merge base filters with overrides, where `None` deletes a key.

        Used to build joint permission sweeps without being constrained by the combo funnel.
        """
        merged: dict[str, object] = dict(_filters_payload(base_filters) or {})
        for key, val in overrides.items():
            if val is None:
                merged.pop(key, None)
            else:
                merged[key] = val

        # Keep TOD gating consistent (both-or-neither).
        if ("entry_start_hour_et" in merged) ^ ("entry_end_hour_et" in merged):
            merged.pop("entry_start_hour_et", None)
            merged.pop("entry_end_hour_et", None)

        # Volume gate requires both knobs.
        if merged.get("volume_ratio_min") is None:
            merged.pop("volume_ema_period", None)

        return _mk_filters(
            rv_min=merged.get("rv_min"),
            rv_max=merged.get("rv_max"),
            ema_spread_min_pct=merged.get("ema_spread_min_pct"),
            ema_spread_min_pct_down=merged.get("ema_spread_min_pct_down"),
            ema_slope_min_pct=merged.get("ema_slope_min_pct"),
            cooldown_bars=int(merged.get("cooldown_bars") or 0),
            skip_first_bars=int(merged.get("skip_first_bars") or 0),
            volume_ratio_min=merged.get("volume_ratio_min"),
            volume_ema_period=merged.get("volume_ema_period"),
            entry_start_hour_et=merged.get("entry_start_hour_et"),
            entry_end_hour_et=merged.get("entry_end_hour_et"),
        )

    def _shortlisted_keys(best_by_key: dict, *, top_pnl: int = 8, top_pnl_dd: int = 8) -> list:
        by_pnl = sorted(best_by_key.items(), key=lambda t: _score_row_pnl(t[1]["row"]), reverse=True)[: int(top_pnl)]
        by_dd = sorted(best_by_key.items(), key=lambda t: _score_row_pnl_dd(t[1]["row"]), reverse=True)[
            : int(top_pnl_dd)
        ]
        out = []
        seen = set()
        for key, _ in by_pnl + by_dd:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

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

    tick_cache: dict[tuple[str, str], tuple[datetime, list]] = {}

    def _tick_bars_for(cfg: ConfigBundle) -> list | None:
        tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
        if tick_mode == "off":
            return None
        if tick_mode != "raschke":
            return None

        tick_symbol = str(getattr(cfg.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
        tick_exchange = str(getattr(cfg.strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip()
        try:
            z_lookback = int(getattr(cfg.strategy, "tick_width_z_lookback", 252) or 252)
        except (TypeError, ValueError):
            z_lookback = 252
        try:
            ma_period = int(getattr(cfg.strategy, "tick_band_ma_period", 10) or 10)
        except (TypeError, ValueError):
            ma_period = 10
        try:
            slope_lb = int(getattr(cfg.strategy, "tick_width_slope_lookback", 3) or 3)
        except (TypeError, ValueError):
            slope_lb = 3

        warm_days = max(60, int(z_lookback) + int(ma_period) + int(slope_lb) + 5)
        tick_start_dt = start_dt - timedelta(days=int(warm_days))
        # $TICK is defined for RTH only (NYSE hours).
        tick_use_rth = True

        def _load_tick_daily(symbol: str, exchange: str) -> list:
            try:
                if offline:
                    return data.load_cached_bars(
                        symbol=symbol,
                        exchange=exchange,
                        start=tick_start_dt,
                        end=end_dt,
                        bar_size="1 day",
                        use_rth=tick_use_rth,
                        cache_dir=cache_dir,
                    )
                return data.load_or_fetch_bars(
                    symbol=symbol,
                    exchange=exchange,
                    start=tick_start_dt,
                    end=end_dt,
                    bar_size="1 day",
                    use_rth=tick_use_rth,
                    cache_dir=cache_dir,
                )
            except FileNotFoundError:
                return []

        def _from_cache(symbol: str, exchange: str) -> list | None:
            cached = tick_cache.get((symbol, exchange))
            if cached is None:
                return None
            cached_start, cached_bars = cached
            if cached_start <= tick_start_dt:
                return cached_bars
            return None

        cached = _from_cache(tick_symbol, tick_exchange)
        if cached is not None:
            return cached

        tick_bars = _load_tick_daily(tick_symbol, tick_exchange)
        used_symbol = tick_symbol
        used_exchange = tick_exchange
        # Offline friendly fallback: IBKR permissions may block NYSE TICK, but AMEX TICK is often available.
        if not tick_bars and tick_symbol.upper() == "TICK-NYSE":
            fallback_symbol = "TICK-AMEX"
            fallback_exchange = "AMEX"
            cached_fb = _from_cache(fallback_symbol, fallback_exchange)
            if cached_fb is not None:
                tick_bars = cached_fb
                used_symbol = fallback_symbol
                used_exchange = fallback_exchange
            else:
                fb = _load_tick_daily(fallback_symbol, fallback_exchange)
                if fb:
                    tick_bars = fb
                    used_symbol = fallback_symbol
                    used_exchange = fallback_exchange
        if not tick_bars:
            hint = (
                " (cache empty; run once without --offline to populate, requires market data permissions)"
                if offline
                else " (check IBKR market data permissions for NYSE IND)"
            )
            extra = " (try TICK-AMEX/AMEX if available)" if tick_symbol.upper() == "TICK-NYSE" else ""
            raise SystemExit(f"No $TICK bars available for {tick_symbol} ({tick_exchange}){hint}{extra}.")
        tick_cache[(used_symbol, used_exchange)] = (tick_start_dt, tick_bars)
        return tick_bars

    def _run_cfg(
        *, cfg: ConfigBundle, bars: list, regime_bars: list | None, regime2_bars: list | None
    ) -> dict | None:
        tick_bars = _tick_bars_for(cfg)
        exec_bars = None
        exec_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
        if exec_size and str(exec_size) != str(cfg.backtest.bar_size):
            exec_bars = _bars_cached(exec_size)
        out = _run_spot_backtest(
            cfg,
            bars,
            meta,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
            exec_bars=exec_bars,
        )
        s = out.summary
        if int(s.trades) < int(run_min_trades):
            return None
        pnl = float(s.total_pnl or 0.0)
        dd = float(s.max_drawdown or 0.0)
        roi = float(getattr(s, "roi", 0.0) or 0.0)
        dd_pct = float(getattr(s, "max_drawdown_pct", 0.0) or 0.0)
        return {
            "trades": int(s.trades),
            "win_rate": float(s.win_rate),
            "pnl": pnl,
            "dd": dd,
            "roi": roi,
            "dd_pct": dd_pct,
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
        if spot_exec_bar_size:
            cfg = replace(cfg, strategy=replace(cfg.strategy, spot_exec_bar_size=spot_exec_bar_size))
        base_name = str(args.base).strip().lower()
        if base_name in ("champion", "champion_pnl"):
            sort_by = "pnl" if base_name == "champion_pnl" else "pnl_dd"
            selected = _milestone_entry_for(
                milestones,
                symbol=symbol,
                signal_bar_size=str(bar_size),
                use_rth=use_rth,
                sort_by=sort_by,
                prefer_realism=realism,
            )
            if selected is not None:
                base_strategy, base_filters, _ = selected
                cfg = _apply_milestone_base(cfg, strategy=base_strategy, filters=base_filters)
            # Allow sweeps to layer additional filters on top of the milestone baseline
            # (e.g., keep the champion's TOD window and add volume/spread/cooldown filters).
            if filters is not None:
                base_payload = _filters_payload(cfg.strategy.filters) or {}
                over_payload = _filters_payload(filters) or {}
                merged = dict(base_payload)
                merged.update(over_payload)
                merged_filters = _mk_filters(
                    ema_spread_min_pct=merged.get("ema_spread_min_pct"),
                    ema_spread_min_pct_down=merged.get("ema_spread_min_pct_down"),
                    ema_slope_min_pct=merged.get("ema_slope_min_pct"),
                    cooldown_bars=int(merged.get("cooldown_bars") or 0),
                    skip_first_bars=int(merged.get("skip_first_bars") or 0),
                    volume_ratio_min=merged.get("volume_ratio_min"),
                    volume_ema_period=merged.get("volume_ema_period"),
                    entry_start_hour_et=merged.get("entry_start_hour_et"),
                    entry_end_hour_et=merged.get("entry_end_hour_et"),
                )
                cfg = replace(cfg, strategy=replace(cfg.strategy, filters=merged_filters))
        elif base_name == "dual_regime":
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

        if long_only:
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    directional_spot={"up": SpotLegConfig(action="BUY", qty=1)},
                ),
            )

        # Realism v1 overrides (backtest only).
        if realism:
            cfg = replace(
                cfg,
                strategy=replace(
                    cfg.strategy,
                    spot_entry_fill_mode="next_open",
                    spot_flip_exit_fill_mode="next_open",
                    spot_intrabar_exits=True,
                    spot_spread=float(spot_spread),
                    spot_commission_per_share=float(spot_commission),
                    spot_commission_min=float(spot_commission_min),
                    spot_slippage_per_share=float(spot_slippage),
                    spot_mark_to_market="liquidation",
                    spot_drawdown_mode="intrabar",
                    spot_sizing_mode=str(sizing_mode),
                    spot_notional_pct=float(spot_notional_pct),
                    spot_risk_pct=float(spot_risk_pct),
                    spot_max_notional_pct=float(spot_max_notional_pct),
                    spot_min_qty=int(spot_min_qty),
                    spot_max_qty=int(spot_max_qty),
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

    def _sweep_rv() -> None:
        """Orthogonal gate: annualized realized-vol (EWMA) band."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rv_mins = [None, 0.25, 0.3, 0.35, 0.4, 0.45]
        rv_maxs = [None, 0.7, 0.8, 0.9, 1.0]
        rows: list[dict] = []
        for rv_min in rv_mins:
            for rv_max in rv_maxs:
                if rv_min is None and rv_max is None:
                    continue
                f = _mk_filters(rv_min=rv_min, rv_max=rv_max)
                cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                row = _run_cfg(
                    cfg=cfg,
                    bars=bars_sig,
                    regime_bars=_regime_bars_for(cfg),
                    regime2_bars=_regime2_bars_for(cfg),
                )
                if not row:
                    continue
                note = f"rv_min={rv_min} rv_max={rv_max}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="RV gate sweep (annualized EWMA vol)", top_n=int(args.top))

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

    def _sweep_entry_mode() -> None:
        """Timing semantics: cross vs trend entries (+ small confirm grid)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        for mode in ("cross", "trend"):
            for confirm in (0, 1, 2):
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        ema_entry_mode=str(mode),
                        entry_confirm_bars=int(confirm),
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
                note = f"entry_mode={mode} confirm={confirm}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Entry mode sweep (cross vs trend)", top_n=int(args.top))

    def _sweep_tod() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        windows = [
            (None, None, "base"),
            (9, 16, "RTH 9–16 ET"),
            (10, 15, "10–15 ET"),
            (11, 16, "11–16 ET"),
        ]
        # Overnight micro-grid (wraps midnight in ET): this has been a high-leverage permission layer
        # post-lookahead-fix, and is cheap to explore.
        for start_h in (16, 17, 18, 19, 20):
            for end_h in (2, 3, 4, 5, 6):
                windows.append((start_h, end_h, f"{start_h:02d}–{end_h:02d} ET"))
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

    def _sweep_tod_interaction() -> None:
        """Small interaction grid around the proven overnight TOD gate."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        tod_starts = [17, 18, 19]
        tod_ends = [3, 4, 5]
        skip_vals = [0, 1, 2]
        cooldown_vals = [0, 1, 2]
        for start_h in tod_starts:
            for end_h in tod_ends:
                for skip in skip_vals:
                    for cooldown in cooldown_vals:
                        f = _mk_filters(
                            entry_start_hour_et=int(start_h),
                            entry_end_hour_et=int(end_h),
                            skip_first_bars=int(skip),
                            cooldown_bars=int(cooldown),
                        )
                        cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=_regime_bars_for(cfg),
                            regime2_bars=_regime2_bars_for(cfg),
                        )
                        if not row:
                            continue
                        note = f"tod={start_h:02d}-{end_h:02d} ET skip={skip} cd={cooldown}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="TOD interaction sweep (overnight micro-grid)", top_n=int(args.top))

    def _sweep_perm_joint() -> None:
        """Joint permission sweep: TOD × spread × volume (no funnel pruning)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        tod_windows: list[tuple[int | None, int | None, str, dict[str, object]]] = []
        tod_windows.append((None, None, "tod=base", {}))
        tod_windows.append((None, None, "tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}))
        for start_h, end_h, label in (
            (9, 16, "tod=09-16 ET"),
            (10, 15, "tod=10-15 ET"),
            (11, 16, "tod=11-16 ET"),
        ):
            tod_windows.append((start_h, end_h, label, {"entry_start_hour_et": int(start_h), "entry_end_hour_et": int(end_h)}))
        for start_h in (17, 18, 19):
            for end_h in (3, 4, 5):
                label = f"tod={start_h:02d}-{end_h:02d} ET"
                tod_windows.append(
                    (start_h, end_h, label, {"entry_start_hour_et": int(start_h), "entry_end_hour_et": int(end_h)})
                )

        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
        ]
        for s in (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01):
            spread_variants.append((f"spread>={s:.4f}", {"ema_spread_min_pct": float(s)}))

        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
        ]
        for ratio in (1.0, 1.1, 1.2, 1.5):
            for ema_p in (10, 20):
                vol_variants.append((f"vol>={ratio}@{ema_p}", {"volume_ratio_min": float(ratio), "volume_ema_period": int(ema_p)}))

        rows: list[dict] = []
        for _, _, tod_note, tod_over in tod_windows:
            for spread_note, spread_over in spread_variants:
                for vol_note, vol_over in vol_variants:
                    overrides: dict[str, object] = {}
                    overrides.update(tod_over)
                    overrides.update(spread_over)
                    overrides.update(vol_over)
                    f = _merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=_regime2_bars_for(cfg),
                    )
                    if not row:
                        continue
                    note = f"{tod_note} | {spread_note} | {vol_note}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Permission joint sweep (TOD × spread × volume)", top_n=int(args.top))

    def _sweep_ema_perm_joint() -> None:
        """Joint sweep: EMA preset × (TOD/spread/volume) permission gates."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Stage 1: evaluate presets with base filters only.
        best_by_ema: dict[str, dict] = {}
        for preset in presets:
            cfg = replace(base, strategy=replace(base.strategy, ema_preset=str(preset), entry_signal="ema"))
            row = _run_cfg(
                cfg=cfg,
                bars=bars_sig,
                regime_bars=_regime_bars_for(cfg),
                regime2_bars=_regime2_bars_for(cfg),
            )
            if not row:
                continue
            best_by_ema[str(preset)] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_ema, top_pnl=5, top_pnl_dd=5)
        if not shortlisted:
            print("No eligible EMA presets (try lowering --min-trades).")
            return
        print("")
        print(f"EMA×Perm: stage1 shortlisted ema={len(shortlisted)} (from {len(best_by_ema)})")

        tod_variants = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
            ("tod=18-05 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 5}),
            ("tod=18-06 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 6}),
            ("tod=17-04 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 4}),
            ("tod=19-04 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 4}),
            ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
        ]
        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
            ("spread>=0.0030", {"ema_spread_min_pct": 0.003}),
            ("spread>=0.0040", {"ema_spread_min_pct": 0.004}),
            ("spread>=0.0050", {"ema_spread_min_pct": 0.005}),
            ("spread>=0.0070", {"ema_spread_min_pct": 0.007}),
            ("spread>=0.0100", {"ema_spread_min_pct": 0.01}),
        ]
        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
        ]

        rows: list[dict] = []
        for preset in shortlisted:
            for tod_note, tod_over in tod_variants:
                for spread_note, spread_over in spread_variants:
                    for vol_note, vol_over in vol_variants:
                        overrides: dict[str, object] = {}
                        overrides.update(tod_over)
                        overrides.update(spread_over)
                        overrides.update(vol_over)
                        f = _merge_filters(base_filters, overrides=overrides)
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                ema_preset=str(preset),
                                entry_signal="ema",
                                filters=f,
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
                        note = f"ema={preset} | {tod_note} | {spread_note} | {vol_note}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × permission joint sweep", top_n=int(args.top))

    def _sweep_tick_perm_joint() -> None:
        """Joint sweep: Raschke $TICK gate × (TOD/spread/volume) permission gates."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        # Stage 1: scan tick params using base permission filters (cheap shortlist).
        best_by_tick: dict[tuple, dict] = {}
        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]
        policies = ["allow", "block"]
        dir_policies = ["both", "wide_only"]
        for dir_policy in dir_policies:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                cfg = replace(
                                    base,
                                    strategy=replace(
                                        base.strategy,
                                        tick_gate_mode="raschke",
                                        tick_gate_symbol="TICK-AMEX",
                                        tick_gate_exchange="AMEX",
                                        tick_neutral_policy=str(policy),
                                        tick_direction_policy=str(dir_policy),
                                        tick_band_ma_period=10,
                                        tick_width_z_lookback=int(lookback),
                                        tick_width_z_enter=float(z_enter),
                                        tick_width_z_exit=float(z_exit),
                                        tick_width_slope_lookback=int(slope_lb),
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
                                tick_key = (
                                    str(dir_policy),
                                    str(policy),
                                    float(z_enter),
                                    float(z_exit),
                                    int(slope_lb),
                                    int(lookback),
                                )
                                current = best_by_tick.get(tick_key)
                                if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                                    best_by_tick[tick_key] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_tick, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible tick candidates (check $TICK cache/permissions, or lower --min-trades).")
            return
        print("")
        print(f"TICK×Perm: stage1 shortlisted tick={len(shortlisted)} (from {len(best_by_tick)})")

        tod_variants = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=18-04 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 4}),
            ("tod=18-05 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 5}),
            ("tod=18-06 ET", {"entry_start_hour_et": 18, "entry_end_hour_et": 6}),
            ("tod=17-04 ET", {"entry_start_hour_et": 17, "entry_end_hour_et": 4}),
            ("tod=19-04 ET", {"entry_start_hour_et": 19, "entry_end_hour_et": 4}),
        ]
        spread_variants: list[tuple[str, dict[str, object]]] = [
            ("spread=base", {}),
            ("spread=off", {"ema_spread_min_pct": None}),
            ("spread>=0.0030", {"ema_spread_min_pct": 0.003}),
            ("spread>=0.0040", {"ema_spread_min_pct": 0.004}),
            ("spread>=0.0050", {"ema_spread_min_pct": 0.005}),
            ("spread>=0.0070", {"ema_spread_min_pct": 0.007}),
        ]
        vol_variants: list[tuple[str, dict[str, object]]] = [
            ("vol=base", {}),
            ("vol=off", {"volume_ratio_min": None, "volume_ema_period": None}),
            ("vol>=1.2@20", {"volume_ratio_min": 1.2, "volume_ema_period": 20}),
        ]

        rows: list[dict] = []
        for tick_key in shortlisted:
            dir_policy, policy, z_enter, z_exit, slope_lb, lookback = tick_key
            for tod_note, tod_over in tod_variants:
                for spread_note, spread_over in spread_variants:
                    for vol_note, vol_over in vol_variants:
                        overrides: dict[str, object] = {}
                        overrides.update(tod_over)
                        overrides.update(spread_over)
                        overrides.update(vol_over)
                        f = _merge_filters(base_filters, overrides=overrides)
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                filters=f,
                                tick_gate_mode="raschke",
                                tick_gate_symbol="TICK-AMEX",
                                tick_gate_exchange="AMEX",
                                tick_neutral_policy=str(policy),
                                tick_direction_policy=str(dir_policy),
                                tick_band_ma_period=10,
                                tick_width_z_lookback=int(lookback),
                                tick_width_z_enter=float(z_enter),
                                tick_width_z_exit=float(z_exit),
                                tick_width_slope_lookback=int(slope_lb),
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
                        note = (
                            f"tick=raschke dir={dir_policy} policy={policy} z_in={z_enter:g} z_out={z_exit:g} "
                            f"slope={slope_lb} lb={lookback} | {tod_note} | {spread_note} | {vol_note}"
                        )
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick × permission joint sweep", top_n=int(args.top))

    def _sweep_ema_regime() -> None:
        """Joint interaction hunt: direction (EMA preset) × regime1 (Supertrend bias)."""
        bars_sig = _bars_cached(signal_bar_size)

        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Keep this bounded but broad enough to catch the interaction pockets:
        # - 4h: micro + macro ST params
        # - 1d: smaller curated set (heavier and less likely, but still worth checking)
        regimes: list[tuple[str, int, float, str]] = []

        rbar = "4 hours"
        atr_ps_4h = [2, 3, 4, 5, 6, 7, 10, 14, 21]
        mults_4h = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        for atr_p in atr_ps_4h:
            for mult in mults_4h:
                for src in ("hl2", "close"):
                    regimes.append((rbar, int(atr_p), float(mult), str(src)))

        rbar = "1 day"
        atr_ps_1d = [7, 10, 14, 21]
        mults_1d = [0.4, 0.6, 0.8, 1.0, 1.2]
        for atr_p in atr_ps_1d:
            for mult in mults_1d:
                for src in ("hl2", "close"):
                    regimes.append((rbar, int(atr_p), float(mult), str(src)))

        rows: list[dict] = []
        for preset in presets:
            for rbar, atr_p, mult, src in regimes:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        ema_preset=str(preset),
                        entry_signal="ema",
                        regime_mode="supertrend",
                        regime_bar_size=str(rbar),
                        supertrend_atr_period=int(atr_p),
                        supertrend_multiplier=float(mult),
                        supertrend_source=str(src),
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
                note = f"ema={preset} | ST({atr_p},{mult:g},{src})@{rbar}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × regime joint sweep (direction × bias)", top_n=int(args.top))

    def _sweep_chop_joint() -> None:
        """Joint chop filter stack: slope × cooldown × skip-open (keeps everything else fixed)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters
        slope_vals = [None, 0.005, 0.01, 0.02, 0.03]
        cooldown_vals = [0, 1, 2, 3, 4, 6]
        skip_vals = [0, 1, 2, 3]

        rows: list[dict] = []
        for slope in slope_vals:
            for cooldown in cooldown_vals:
                for skip in skip_vals:
                    overrides: dict[str, object] = {
                        "ema_slope_min_pct": float(slope) if slope is not None else None,
                        "cooldown_bars": int(cooldown),
                        "skip_first_bars": int(skip),
                    }
                    f = _merge_filters(base_filters, overrides=overrides)
                    cfg = replace(base, strategy=replace(base.strategy, filters=f))
                    row = _run_cfg(
                        cfg=cfg,
                        bars=bars_sig,
                        regime_bars=_regime_bars_for(cfg),
                        regime2_bars=_regime2_bars_for(cfg),
                    )
                    if not row:
                        continue
                    slope_note = "-" if slope is None else f"slope>={float(slope):g}"
                    note = f"{slope_note} | cooldown={cooldown} | skip={skip}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Chop joint sweep (slope × cooldown × skip-open)", top_n=int(args.top))

    def _sweep_tick_ema() -> None:
        """Joint interaction hunt: Raschke $TICK (wide-only bias) × EMA preset."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]
        policies = ["allow", "block"]
        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]

        rows: list[dict] = []
        for preset in presets:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                cfg = replace(
                                    base,
                                    strategy=replace(
                                        base.strategy,
                                        entry_signal="ema",
                                        ema_preset=str(preset),
                                        tick_gate_mode="raschke",
                                        tick_gate_symbol="TICK-AMEX",
                                        tick_gate_exchange="AMEX",
                                        tick_neutral_policy=str(policy),
                                        tick_direction_policy="wide_only",
                                        tick_band_ma_period=10,
                                        tick_width_z_lookback=int(lookback),
                                        tick_width_z_enter=float(z_enter),
                                        tick_width_z_exit=float(z_exit),
                                        tick_width_slope_lookback=int(slope_lb),
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
                                note = (
                                    f"ema={preset} | tick=wide_only policy={policy} z_in={z_enter:g} "
                                    f"z_out={z_exit:g} slope={slope_lb} lb={lookback}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick × EMA joint sweep (Raschke wide-only bias)", top_n=int(args.top))

    def _sweep_ema_atr() -> None:
        """Joint interaction hunt: direction (EMA preset) × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["2/4", "3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]

        # Stage 1: shortlist EMA presets against the base bias/permissions.
        best_by_ema: dict[str, dict] = {}
        for preset in presets:
            cfg = replace(base, strategy=replace(base.strategy, ema_preset=str(preset), entry_signal="ema"))
            row = _run_cfg(
                cfg=cfg,
                bars=bars_sig,
                regime_bars=_regime_bars_for(cfg),
                regime2_bars=_regime2_bars_for(cfg),
            )
            if not row:
                continue
            best_by_ema[str(preset)] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_ema, top_pnl=5, top_pnl_dd=5)
        if not shortlisted:
            print("No eligible EMA presets (try lowering --min-trades).")
            return
        print("")
        print(f"EMA×ATR: stage1 shortlisted ema={len(shortlisted)} (from {len(best_by_ema)})")

        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

        rows: list[dict] = []
        for preset in shortlisted:
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                ema_preset=str(preset),
                                entry_signal="ema",
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
                        note = f"ema={preset} | ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="EMA × ATR joint sweep (direction × exits)", top_n=int(args.top))

    def _sweep_weekdays() -> None:
        """Gate exploration: which UTC weekdays contribute to the edge."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        day_sets: list[tuple[tuple[int, ...], str]] = [
            ((0, 1, 2, 3, 4), "Mon-Fri"),
            ((0, 1, 2, 3), "Mon-Thu"),
            ((1, 2, 3, 4), "Tue-Fri"),
            ((1, 2, 3), "Tue-Thu"),
            ((2, 3, 4), "Wed-Fri"),
            ((0, 1, 2), "Mon-Wed"),
            ((0, 1, 2, 3, 4, 5, 6), "All days"),
        ]

        rows: list[dict] = []
        for days, label in day_sets:
            cfg = replace(base, strategy=replace(base.strategy, entry_days=tuple(days)))
            row = _run_cfg(
                cfg=cfg,
                bars=bars_sig,
                regime_bars=_regime_bars_for(cfg),
                regime2_bars=_regime2_bars_for(cfg),
            )
            if not row:
                continue
            note = f"days={label}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Weekday sweep (UTC weekday gating)", top_n=int(args.top))

    def _sweep_exit_time() -> None:
        """Session-aware exit experiment: force a daily time-based flatten (ET)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        times = [
            None,
            "04:00",
            "09:30",
            "10:00",
            "11:00",
            "16:00",
            "17:00",
        ]
        rows: list[dict] = []
        for t in times:
            cfg = replace(base, strategy=replace(base.strategy, spot_exit_time_et=t))
            row = _run_cfg(
                cfg=cfg,
                bars=bars_sig,
                regime_bars=_regime_bars_for(cfg),
                regime2_bars=_regime2_bars_for(cfg),
            )
            if not row:
                continue
            note = "-" if t is None else f"exit_time={t} ET"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Exit-time sweep (ET flatten)", top_n=int(args.top))

    def _sweep_atr_exits() -> None:
        bars_sig = _bars_cached(signal_bar_size)
        atr_periods = [7, 10, 14, 21]
        # Include a low-PT pocket (PTx<1.0): this has produced materially higher net PnL post-fix.
        pt_mults = [0.6, 0.8, 0.9, 1.0, 1.5, 2.0]
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

    def _sweep_atr_exits_fine() -> None:
        """Fine-grained ATR exit sweep around the current champion neighborhood."""
        bars_sig = _bars_cached(signal_bar_size)
        # Cover both the risk-adjusted champ neighborhood (ATR 7/10) and the net-PnL pocket (ATR 14/21).
        atr_periods = [7, 10, 14, 21]
        pt_mults = [0.8, 0.9, 1.0, 1.1, 1.2]
        sl_mults = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
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
                    note = f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="ATR exits fine sweep (PT/SL multipliers)", top_n=int(args.top))

    def _sweep_atr_exits_ultra() -> None:
        """Ultra-fine ATR exit sweep around the current best PT neighborhood."""
        bars_sig = _bars_cached(signal_bar_size)
        atr_periods = [7]
        pt_mults = [1.05, 1.08, 1.10, 1.12, 1.15]
        sl_mults = [1.35, 1.40, 1.45, 1.50, 1.55]
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
                    note = f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)
        _print_leaderboards(rows, title="ATR exits ultra-fine sweep (PT/SL micro-grid)", top_n=int(args.top))

    def _sweep_r2_atr() -> None:
        """Joint interaction hunt: regime2 confirm × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Stage 1: coarse scan to shortlist promising regime2 settings.
        r2_variants: list[tuple[dict, str]] = [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off"),
        ]
        r2_bar_sizes = ["4 hours", "1 day"]
        r2_atr_periods = [7, 10, 11, 14, 21]
        r2_multipliers = [0.6, 0.8, 1.0, 1.2, 1.5]
        r2_sources = ["hl2", "close"]
        for r2_bar in r2_bar_sizes:
            for atr_p in r2_atr_periods:
                for mult in r2_multipliers:
                    for src in r2_sources:
                        r2_variants.append(
                            (
                                {
                                    "regime2_mode": "supertrend",
                                    "regime2_bar_size": str(r2_bar),
                                    "regime2_supertrend_atr_period": int(atr_p),
                                    "regime2_supertrend_multiplier": float(mult),
                                    "regime2_supertrend_source": str(src),
                                },
                                f"r2=ST2({r2_bar}:{atr_p},{mult},{src})",
                            )
                        )

        exit_stage1: list[tuple[dict, str]] = [
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.8,
                    "spot_sl_atr_mult": 1.6,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx0.80 SLx1.60",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 0.9,
                    "spot_sl_atr_mult": 1.6,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx0.90 SLx1.60",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 21,
                    "spot_pt_atr_mult": 0.9,
                    "spot_sl_atr_mult": 1.4,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(21) PTx0.90 SLx1.40",
            ),
            (
                {
                    "spot_exit_mode": "atr",
                    "spot_atr_period": 14,
                    "spot_pt_atr_mult": 1.0,
                    "spot_sl_atr_mult": 1.5,
                    "spot_profit_target_pct": None,
                    "spot_stop_loss_pct": None,
                },
                "ATR(14) PTx1.00 SLx1.50",
            ),
        ]

        stage1: list[tuple[tuple, dict, str]] = []
        for r2_over, r2_note in r2_variants:
            for exit_over, exit_note in exit_stage1:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                        regime2_bar_size=r2_over.get("regime2_bar_size"),
                        regime2_supertrend_atr_period=int(r2_over.get("regime2_supertrend_atr_period") or 10),
                        regime2_supertrend_multiplier=float(r2_over.get("regime2_supertrend_multiplier") or 3.0),
                        regime2_supertrend_source=str(r2_over.get("regime2_supertrend_source") or "hl2"),
                        spot_exit_mode=str(exit_over["spot_exit_mode"]),
                        spot_atr_period=int(exit_over["spot_atr_period"]),
                        spot_pt_atr_mult=float(exit_over["spot_pt_atr_mult"]),
                        spot_sl_atr_mult=float(exit_over["spot_sl_atr_mult"]),
                        spot_profit_target_pct=exit_over["spot_profit_target_pct"],
                        spot_stop_loss_pct=exit_over["spot_stop_loss_pct"],
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
                r2_key = (
                    str(getattr(cfg.strategy, "regime2_mode", "off") or "off"),
                    str(getattr(cfg.strategy, "regime2_bar_size", "") or ""),
                    int(getattr(cfg.strategy, "regime2_supertrend_atr_period", 0) or 0),
                    float(getattr(cfg.strategy, "regime2_supertrend_multiplier", 0.0) or 0.0),
                    str(getattr(cfg.strategy, "regime2_supertrend_source", "") or ""),
                )
                note = f"{r2_note} | {exit_note}"
                row["note"] = note
                stage1.append((r2_key, row, note))

        if not stage1:
            print("No eligible results in stage1 (try lowering --min-trades).")
            return

        # Shortlist by best observed metrics per regime2 key.
        best_by_r2: dict[tuple, dict] = {}
        for r2_key, row, note in stage1:
            current = best_by_r2.get(r2_key)
            if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                best_by_r2[r2_key] = {"row": row, "note": note}

        ranked_by_pnl = sorted(best_by_r2.items(), key=lambda t: _score_row_pnl(t[1]["row"]), reverse=True)[:8]
        ranked_by_dd = sorted(best_by_r2.items(), key=lambda t: _score_row_pnl_dd(t[1]["row"]), reverse=True)[:8]
        shortlisted_keys = []
        seen: set[tuple] = set()
        for r2_key, _ in ranked_by_pnl + ranked_by_dd:
            if r2_key in seen:
                continue
            seen.add(r2_key)
            shortlisted_keys.append(r2_key)

        print("")
        print(f"R2×ATR: stage1 shortlisted r2={len(shortlisted_keys)} (from {len(best_by_r2)})")

        # Stage 2: exit microgrid for shortlisted regime2 settings.
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
        atr_periods = [14, 21]

        rows: list[dict] = []
        for r2_key in shortlisted_keys:
            r2_mode, r2_bar, r2_atr, r2_mult, r2_src = r2_key
            for atr_p in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime2_mode=str(r2_mode),
                                regime2_bar_size=str(r2_bar) or None,
                                regime2_supertrend_atr_period=int(r2_atr or 10),
                                regime2_supertrend_multiplier=float(r2_mult or 3.0),
                                regime2_supertrend_source=str(r2_src or "hl2"),
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
                        if str(r2_mode).strip().lower() == "off":
                            r2_note = "r2=off"
                        else:
                            r2_note = f"r2=ST2({r2_bar}:{r2_atr},{r2_mult:g},{r2_src})"
                        note = f"{r2_note} | ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 × ATR joint sweep (PT<1.0 pocket)", top_n=int(args.top))

    def _sweep_r2_tod() -> None:
        """Joint interaction hunt: regime2 confirm × TOD window (keeps exits fixed)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        base_filters = base.strategy.filters

        # Stage 1: scan regime2 settings with the current base TOD.
        r2_variants: list[tuple[dict, str]] = [({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off")]
        for r2_bar in ("4 hours", "1 day"):
            for atr_p in (3, 5, 7, 10, 11, 14, 21):
                for mult in (0.6, 0.8, 1.0, 1.2, 1.5):
                    for src in ("hl2", "close"):
                        r2_variants.append(
                            (
                                {
                                    "regime2_mode": "supertrend",
                                    "regime2_bar_size": str(r2_bar),
                                    "regime2_supertrend_atr_period": int(atr_p),
                                    "regime2_supertrend_multiplier": float(mult),
                                    "regime2_supertrend_source": str(src),
                                },
                                f"r2=ST2({r2_bar}:{atr_p},{mult:g},{src})",
                            )
                        )

        best_by_r2: dict[tuple, dict] = {}
        for r2_over, r2_note in r2_variants:
            cfg = replace(
                base,
                strategy=replace(
                    base.strategy,
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
                regime_bars=_regime_bars_for(cfg),
                regime2_bars=_regime2_bars_for(cfg),
            )
            if not row:
                continue
            r2_key = (
                str(getattr(cfg.strategy, "regime2_mode", "off") or "off"),
                str(getattr(cfg.strategy, "regime2_bar_size", "") or ""),
                int(getattr(cfg.strategy, "regime2_supertrend_atr_period", 0) or 0),
                float(getattr(cfg.strategy, "regime2_supertrend_multiplier", 0.0) or 0.0),
                str(getattr(cfg.strategy, "regime2_supertrend_source", "") or ""),
            )
            current = best_by_r2.get(r2_key)
            if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                best_by_r2[r2_key] = {"row": row, "note": r2_note}

        shortlisted = _shortlisted_keys(best_by_r2, top_pnl=10, top_pnl_dd=10)
        if not shortlisted:
            print("No eligible regime2 candidates (try lowering --min-trades).")
            return
        print("")
        print(f"R2×TOD: stage1 shortlisted r2={len(shortlisted)} (from {len(best_by_r2)})")

        tod_variants: list[tuple[str, dict[str, object]]] = [
            ("tod=base", {}),
            ("tod=off", {"entry_start_hour_et": None, "entry_end_hour_et": None}),
            ("tod=09-16 ET", {"entry_start_hour_et": 9, "entry_end_hour_et": 16}),
            ("tod=10-15 ET", {"entry_start_hour_et": 10, "entry_end_hour_et": 15}),
            ("tod=11-16 ET", {"entry_start_hour_et": 11, "entry_end_hour_et": 16}),
        ]
        for start_h in (16, 17, 18, 19, 20):
            for end_h in (2, 3, 4, 5, 6):
                tod_variants.append((f"tod={start_h:02d}-{end_h:02d} ET", {"entry_start_hour_et": start_h, "entry_end_hour_et": end_h}))

        rows: list[dict] = []
        for r2_key in shortlisted:
            r2_mode, r2_bar, r2_atr, r2_mult, r2_src = r2_key
            for tod_note, tod_over in tod_variants:
                f = _merge_filters(base_filters, overrides=tod_over)
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        filters=f,
                        regime2_mode=str(r2_mode),
                        regime2_bar_size=str(r2_bar) or None,
                        regime2_supertrend_atr_period=int(r2_atr or 10),
                        regime2_supertrend_multiplier=float(r2_mult or 3.0),
                        regime2_supertrend_source=str(r2_src or "hl2"),
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
                if str(r2_mode).strip().lower() == "off":
                    r2_note = "r2=off"
                else:
                    r2_note = f"r2=ST2({r2_bar}:{r2_atr},{r2_mult:g},{r2_src})"
                note = f"{r2_note} | {tod_note}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 × TOD joint sweep", top_n=int(args.top))

    def _sweep_regime_atr() -> None:
        """Joint interaction hunt: regime (bias) × ATR exits (includes PTx < 1.0)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Stage 1: scan regime settings using a representative low-PT exit.
        best_by_regime: dict[tuple, dict] = {}
        for rbar in ("4 hours", "1 day"):
            for atr_p in (3, 5, 6, 7, 10, 14, 21):
                for mult in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
                    for src in ("hl2", "close"):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=str(rbar),
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                                regime2_mode="off",
                                regime2_bar_size=None,
                                spot_exit_mode="atr",
                                spot_atr_period=14,
                                spot_pt_atr_mult=0.7,
                                spot_sl_atr_mult=1.6,
                                spot_profit_target_pct=None,
                                spot_stop_loss_pct=None,
                            ),
                        )
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=_regime_bars_for(cfg),
                            regime2_bars=None,
                        )
                        if not row:
                            continue
                        key = (str(rbar), int(atr_p), float(mult), str(src))
                        current = best_by_regime.get(key)
                        if current is None or _score_row_pnl(row) > _score_row_pnl(current["row"]):
                            best_by_regime[key] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_regime, top_pnl=10, top_pnl_dd=10)
        if not shortlisted:
            print("No eligible regime candidates (try lowering --min-trades).")
            return
        print("")
        print(f"Regime×ATR: stage1 shortlisted regimes={len(shortlisted)} (from {len(best_by_regime)})")

        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        sl_mults = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]

        rows: list[dict] = []
        for rbar, atr_p, mult, src in shortlisted:
            for exit_atr in atr_periods:
                for pt_m in pt_mults:
                    for sl_m in sl_mults:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size=str(rbar),
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source=str(src),
                                regime2_mode="off",
                                regime2_bar_size=None,
                                spot_exit_mode="atr",
                                spot_atr_period=int(exit_atr),
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
                            regime2_bars=None,
                        )
                        if not row:
                            continue
                        note = (
                            f"ST({atr_p},{mult:g},{src})@{rbar} | "
                            f"ATR({exit_atr}) PTx{pt_m:.2f} SLx{sl_m:.2f} | r2=off"
                        )
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime × ATR joint sweep (PT<1.0 pocket)", top_n=int(args.top))

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
        base = _base_bundle(bar_size="15 mins", filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_15m, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        rr_vals = [0.618, 0.707, 0.786, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override (not merge) filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    entry_signal="orb",
                                    ema_preset=None,
                                    entry_confirm_bars=0,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
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
                            note = (
                                f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} "
                                f"tod={start_h:02d}-{end_h:02d} ET {vol_note}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="D) ORB sweep (open-time + window)", top_n=int(args.top))

    def _sweep_orb_joint() -> None:
        """Joint ORB exploration: ORB params × (regime bias) × (optional tick bias).

        Note: ORB uses its own stop/target derived from the opening range, so EMA-based
        quality gates (spread/slope) aren't applicable here unless we compute EMA in
        parallel. We stick to regime/tick/volume/TOD gates that remain well-defined.
        """
        bars_15m = _bars_cached("15 mins")

        # Start from the selected base shape, but neutralize regime/tick so stage1 can
        # shortlist ORB mechanics without hidden gating.
        base = _base_bundle(bar_size="15 mins", filters=None)
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                entry_signal="orb",
                ema_preset=None,
                entry_confirm_bars=0,
                regime_mode="ema",
                regime_bar_size="15 mins",
                regime_ema_preset=None,
                regime2_mode="off",
                regime2_bar_size=None,
                tick_gate_mode="off",
            ),
        )
        base_row = _run_cfg(
            cfg=base,
            bars=bars_15m,
            regime_bars=_regime_bars_for(base),
            regime2_bars=_regime2_bars_for(base),
        )
        if base_row:
            base_row["note"] = "base (orb, no regime/tick)"
            _record_milestone(base, base_row, str(base_row["note"]))

        rr_vals = [0.618, 0.707, 0.786, 0.8, 1.0, 1.272, 1.618, 2.0]
        vol_vals = [None, 1.2]
        window_vals = [15, 30, 60]
        sessions: list[tuple[str, int, int]] = [
            ("09:30", 9, 16),  # RTH open
            ("18:00", 18, 4),  # Globex open (overnight window wraps)
        ]

        # Stage 1: find the best ORB mechanics without regime/tick overlays.
        best_by_orb: dict[tuple, dict] = {}
        for open_time, start_h, end_h in sessions:
            for window_mins in window_vals:
                for target_mode in ("rr", "or_range"):
                    for rr in rr_vals:
                        for vol_min in vol_vals:
                            f = _mk_filters(
                                entry_start_hour_et=int(start_h),
                                entry_end_hour_et=int(end_h),
                                volume_ratio_min=vol_min,
                                volume_ema_period=20 if vol_min is not None else None,
                            )
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    # Override filters so ORB isn't blocked by EMA-only gates.
                                    filters=f,
                                    orb_open_time_et=str(open_time),
                                    orb_window_mins=int(window_mins),
                                    orb_risk_reward=float(rr),
                                    orb_target_mode=str(target_mode),
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
                            orb_key = (str(open_time), int(window_mins), str(target_mode), float(rr), vol_min)
                            best_by_orb[orb_key] = {"row": row}

        shortlisted = _shortlisted_keys(best_by_orb, top_pnl=8, top_pnl_dd=8)
        if not shortlisted:
            print("No eligible ORB candidates (try lowering --min-trades).")
            return
        print("")
        print(f"ORB×(regime/tick): stage1 shortlisted orb={len(shortlisted)} (from {len(best_by_orb)})")

        # Stage 2: apply a small curated set of regime overlays + tick "wide-only" bias.
        regime_variants: list[tuple[str, dict[str, object]]] = [
            ("regime=off", {"regime_mode": "ema", "regime_bar_size": "15 mins", "regime_ema_preset": None}),
        ]
        for atr_p, mult, src in (
            (3, 0.4, "hl2"),
            (6, 0.6, "hl2"),
            (7, 0.6, "hl2"),
            (14, 0.6, "hl2"),
            (21, 0.5, "close"),
            (21, 0.6, "hl2"),
        ):
            regime_variants.append(
                (
                    f"ST({atr_p},{mult:g},{src})@4h",
                    {
                        "regime_mode": "supertrend",
                        "regime_bar_size": "4 hours",
                        "supertrend_atr_period": int(atr_p),
                        "supertrend_multiplier": float(mult),
                        "supertrend_source": str(src),
                    },
                )
            )

        tick_variants: list[tuple[str, dict[str, object]]] = [
            ("tick=off", {"tick_gate_mode": "off"}),
            (
                "tick=wide_only allow (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "allow",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
            (
                "tick=wide_only block (z=1.0/0.5 slope=3 lb=252)",
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
            ),
        ]

        rows: list[dict] = []
        for open_time, window_mins, target_mode, rr, vol_min in shortlisted:
            start_h, end_h = 9, 16
            if str(open_time) == "18:00":
                start_h, end_h = 18, 4
            f = _mk_filters(
                entry_start_hour_et=int(start_h),
                entry_end_hour_et=int(end_h),
                volume_ratio_min=vol_min,
                volume_ema_period=20 if vol_min is not None else None,
            )

            for regime_note, reg_over in regime_variants:
                for tick_note, tick_over in tick_variants:
                    cfg = replace(
                        base,
                        strategy=replace(
                            base.strategy,
                            filters=f,
                            orb_open_time_et=str(open_time),
                            orb_window_mins=int(window_mins),
                            orb_risk_reward=float(rr),
                            orb_target_mode=str(target_mode),
                            regime_mode=str(reg_over.get("regime_mode") or "ema"),
                            regime_bar_size=str(reg_over.get("regime_bar_size") or "15 mins"),
                            regime_ema_preset=reg_over.get("regime_ema_preset"),
                            supertrend_atr_period=int(reg_over.get("supertrend_atr_period") or 10),
                            supertrend_multiplier=float(reg_over.get("supertrend_multiplier") or 3.0),
                            supertrend_source=str(reg_over.get("supertrend_source") or "hl2"),
                            tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                            tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                            tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                            tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                            tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                            tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                            tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                            tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                            tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                            tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
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
                    note = (
                        f"ORB open={open_time} w={window_mins} {target_mode} rr={rr} "
                        f"tod={start_h:02d}-{end_h:02d} ET {vol_note} | {regime_note} | {tick_note}"
                    )
                    row["note"] = note
                    _record_milestone(cfg, row, note)
                    rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="ORB joint sweep (ORB × regime × tick)", top_n=int(args.top))

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

    def _sweep_regime2_ema() -> None:
        """Confirm layer: EMA trend gate on a higher timeframe (4h/1d)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        presets = ["3/7", "4/9", "5/10", "8/21", "9/21", "21/50"]
        rows: list[dict] = []
        for r2_bar in ("4 hours", "1 day"):
            for preset in presets:
                cfg = replace(
                    base,
                    strategy=replace(
                        base.strategy,
                        regime2_mode="ema",
                        regime2_bar_size=str(r2_bar),
                        regime2_ema_preset=str(preset),
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
                note = f"r2=EMA({preset})@{r2_bar}"
                row["note"] = note
                _record_milestone(cfg, row, note)
                rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Regime2 EMA sweep (trend confirm)", top_n=int(args.top))

    def _sweep_joint() -> None:
        """Targeted interaction hunt: sweep regime + regime2 together (keeps base filters)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Keep this tight and focused; the point is to cover interaction edges that the combo funnel can miss.
        regime_bar_sizes = ["4 hours"]
        regime_atr_periods = [10, 14, 21]
        regime_multipliers = [0.4, 0.5, 0.6]
        regime_sources = ["close", "hl2"]

        r2_bar_sizes = ["4 hours", "1 day"]
        r2_atr_periods = [3, 4, 5, 6, 7, 10, 14]
        r2_multipliers = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        r2_sources = ["close", "hl2"]

        rows: list[dict] = []
        for rbar in regime_bar_sizes:
            for atr_p in regime_atr_periods:
                for mult in regime_multipliers:
                    for src in regime_sources:
                        for r2_bar in r2_bar_sizes:
                            for r2_atr in r2_atr_periods:
                                for r2_mult in r2_multipliers:
                                    for r2_src in r2_sources:
                                        cfg = replace(
                                            base,
                                            strategy=replace(
                                                base.strategy,
                                                regime_mode="supertrend",
                                                regime_bar_size=rbar,
                                                supertrend_atr_period=int(atr_p),
                                                supertrend_multiplier=float(mult),
                                                supertrend_source=str(src),
                                                regime2_mode="supertrend",
                                                regime2_bar_size=str(r2_bar),
                                                regime2_supertrend_atr_period=int(r2_atr),
                                                regime2_supertrend_multiplier=float(r2_mult),
                                                regime2_supertrend_source=str(r2_src),
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
                                        note = (
                                            f"ST({atr_p},{mult},{src})@{rbar} + "
                                            f"ST2({r2_bar}:{r2_atr},{r2_mult},{r2_src})"
                                        )
                                        row["note"] = note
                                        _record_milestone(cfg, row, note)
                                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Joint sweep (regime × regime2)", top_n=int(args.top))

    def _sweep_micro_st() -> None:
        """Micro sweep around the current ST + ST2 neighborhood (tighter, more granular)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        regime_atr_periods = [14, 21]
        regime_multipliers = [0.4, 0.45, 0.5, 0.55, 0.6]

        r2_atr_periods = [4, 5, 6]
        r2_multipliers = [0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.4]

        rows: list[dict] = []
        for atr_p in regime_atr_periods:
            for mult in regime_multipliers:
                for r2_atr in r2_atr_periods:
                    for r2_mult in r2_multipliers:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime_mode="supertrend",
                                regime_bar_size="4 hours",
                                supertrend_atr_period=int(atr_p),
                                supertrend_multiplier=float(mult),
                                supertrend_source="close",
                                regime2_mode="supertrend",
                                regime2_bar_size="4 hours",
                                regime2_supertrend_atr_period=int(r2_atr),
                                regime2_supertrend_multiplier=float(r2_mult),
                                regime2_supertrend_source="close",
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
                        note = f"ST({atr_p},{mult},close) + ST2(4h:{r2_atr},{r2_mult},close)"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Micro ST sweep (granular mults)", top_n=int(args.top))

    def _sweep_flip_exit() -> None:
        """Targeted exit semantics: flip-exit mode + profit-only gating."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        rows: list[dict] = []
        for exit_on_flip in (True, False):
            for mode in ("entry", "state", "cross"):
                for only_profit in (False, True):
                    for hold in (0, 2, 4, 6):
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                exit_on_signal_flip=bool(exit_on_flip),
                                flip_exit_mode=str(mode),
                                flip_exit_only_if_profit=bool(only_profit),
                                flip_exit_min_hold_bars=int(hold),
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
                        note = (
                            f"flip={'on' if exit_on_flip else 'off'} mode={mode} "
                            f"hold={hold} only_profit={int(only_profit)}"
                        )
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Flip-exit semantics sweep", top_n=int(args.top))

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

    def _sweep_spread_fine() -> None:
        """Fine-grained sweep around the current champion spread gate."""
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        spreads = [None, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008]
        for spread in spreads:
            f = _mk_filters(ema_spread_min_pct=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            spread_note = "-" if spread is None else f"spread>={float(spread):.4f}"
            row["note"] = spread_note
            _record_milestone(cfg, row, spread_note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread fine sweep (quality gate)", top_n=int(args.top))

    def _sweep_spread_down() -> None:
        """Directional permission: sweep stricter EMA spread gate for down entries only."""
        bars_sig = _bars_cached(signal_bar_size)
        rows: list[dict] = []
        spreads = [None, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.02, 0.03, 0.05]
        for spread in spreads:
            f = _mk_filters(ema_spread_min_pct_down=float(spread) if spread is not None else None)
            cfg = _base_bundle(bar_size=signal_bar_size, filters=f)
            row = _run_cfg(
                cfg=cfg, bars=bars_sig, regime_bars=_regime_bars_for(cfg), regime2_bars=_regime2_bars_for(cfg)
            )
            if not row:
                continue
            note = "-" if spread is None else f"spread_down>={float(spread):.4f}"
            row["note"] = note
            _record_milestone(cfg, row, note)
            rows.append(row)
        _print_leaderboards(rows, title="EMA spread DOWN sweep (directional permission)", top_n=int(args.top))

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

    def _sweep_loosen_atr() -> None:
        """Interaction hunt: stacking (max_open) × ATR exits (includes PTx < 1.0 pocket)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        # Keep the grid tight around the post-fix high-PnL neighborhood.
        atr_periods = [10, 14, 21]
        pt_mults = [0.6, 0.65, 0.7, 0.75, 0.8]
        sl_mults = [1.2, 1.4, 1.6, 1.8, 2.0]
        max_open_vals = [2, 3, 0]
        close_eod_vals = [False, True]

        rows: list[dict] = []
        for max_open in max_open_vals:
            for close_eod in close_eod_vals:
                for atr_p in atr_periods:
                    for pt_m in pt_mults:
                        for sl_m in sl_mults:
                            cfg = replace(
                                base,
                                strategy=replace(
                                    base.strategy,
                                    max_open_trades=int(max_open),
                                    spot_close_eod=bool(close_eod),
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
                            note = (
                                f"max_open={max_open} close_eod={int(close_eod)} | "
                                f"ATR({atr_p}) PTx{pt_m:.2f} SLx{sl_m:.2f}"
                            )
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Loosen × ATR joint sweep (stacking × exits)", top_n=int(args.top))

    def _sweep_tick() -> None:
        """Permission layer: Raschke-style $TICK width gate (daily, RTH only)."""
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "tick=off (base)"
            _record_milestone(base, base_row, "tick=off (base)")

        z_enters = [0.8, 1.0, 1.2]
        z_exits = [0.4, 0.5, 0.6]
        slope_lbs = [3, 5]
        lookbacks = [126, 252]
        policies = ["allow", "block"]
        dir_policies = ["both", "wide_only"]
        regime2_variants: list[tuple[dict, str]] = []
        base_r2_mode = str(getattr(base.strategy, "regime2_mode", "off") or "off").strip().lower()
        if base_r2_mode != "off":
            regime2_variants.append(
                (
                    {
                        "regime2_mode": str(getattr(base.strategy, "regime2_mode") or "off"),
                        "regime2_bar_size": getattr(base.strategy, "regime2_bar_size", None),
                        "regime2_supertrend_atr_period": getattr(base.strategy, "regime2_supertrend_atr_period", None),
                        "regime2_supertrend_multiplier": getattr(base.strategy, "regime2_supertrend_multiplier", None),
                        "regime2_supertrend_source": getattr(base.strategy, "regime2_supertrend_source", None),
                    },
                    "r2=base",
                )
            )
        regime2_variants += [
            ({"regime2_mode": "off", "regime2_bar_size": None}, "r2=off"),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 3,
                    "regime2_supertrend_multiplier": 0.25,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:3,0.25,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "4 hours",
                    "regime2_supertrend_atr_period": 5,
                    "regime2_supertrend_multiplier": 0.2,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(4h:5,0.2,close)",
            ),
            (
                {
                    "regime2_mode": "supertrend",
                    "regime2_bar_size": "1 day",
                    "regime2_supertrend_atr_period": 7,
                    "regime2_supertrend_multiplier": 0.4,
                    "regime2_supertrend_source": "close",
                },
                "r2=ST(1d:7,0.4,close)",
            ),
        ]

        rows: list[dict] = []
        for dir_policy in dir_policies:
            for policy in policies:
                for z_enter in z_enters:
                    for z_exit in z_exits:
                        for slope_lb in slope_lbs:
                            for lookback in lookbacks:
                                for r2_over, r2_note in regime2_variants:
                                    strat = base.strategy
                                    cfg = replace(
                                        base,
                                        strategy=replace(
                                            strat,
                                            tick_gate_mode="raschke",
                                            tick_gate_symbol="TICK-AMEX",
                                            tick_gate_exchange="AMEX",
                                            tick_neutral_policy=str(policy),
                                            tick_direction_policy=str(dir_policy),
                                            tick_band_ma_period=10,
                                            tick_width_z_lookback=int(lookback),
                                            tick_width_z_enter=float(z_enter),
                                            tick_width_z_exit=float(z_exit),
                                            tick_width_slope_lookback=int(slope_lb),
                                            regime2_mode=str(r2_over.get("regime2_mode") or "off"),
                                            regime2_bar_size=r2_over.get("regime2_bar_size"),
                                            regime2_supertrend_atr_period=int(
                                                r2_over.get("regime2_supertrend_atr_period") or 10
                                            ),
                                            regime2_supertrend_multiplier=float(
                                                r2_over.get("regime2_supertrend_multiplier") or 3.0
                                            ),
                                            regime2_supertrend_source=str(
                                                r2_over.get("regime2_supertrend_source") or "hl2"
                                            ),
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
                                    note = (
                                        f"tick=raschke dir={dir_policy} policy={policy} z_in={z_enter} "
                                        f"z_out={z_exit} slope={slope_lb} lb={lookback} {r2_note}"
                                    )
                                    row["note"] = note
                                    _record_milestone(cfg, row, note)
                                    rows.append(row)
        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Tick gate sweep ($TICK width)", top_n=int(args.top))

    def _sweep_frontier() -> None:
        """Summarize the current milestones set as a multi-objective frontier."""
        groups = milestones.get("groups", []) if isinstance(milestones, dict) else []
        rows: list[dict] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            entries = group.get("entries") or []
            if not entries or not isinstance(entries, list):
                continue
            entry = entries[0]
            if not isinstance(entry, dict):
                continue
            strat = entry.get("strategy") or {}
            metrics = entry.get("metrics") or {}
            if not isinstance(strat, dict) or not isinstance(metrics, dict):
                continue
            if str(strat.get("instrument", "spot") or "spot").strip().lower() != "spot":
                continue
            if str(strat.get("signal_bar_size") or "").strip().lower() != str(signal_bar_size).strip().lower():
                continue
            if bool(strat.get("signal_use_rth")) != bool(use_rth):
                continue
            if str(entry.get("symbol") or "").strip().upper() != str(symbol).strip().upper():
                continue
            try:
                trades = int(metrics.get("trades") or 0)
                win = float(metrics.get("win_rate") or 0.0)
                pnl = float(metrics.get("pnl") or 0.0)
                dd = float(metrics.get("max_drawdown") or 0.0)
                pnl_dd = metrics.get("pnl_over_dd")
                pnl_over_dd = float(pnl_dd) if pnl_dd is not None else (pnl / dd if dd > 0 else None)
            except (TypeError, ValueError):
                continue
            note = str(group.get("name") or "").strip() or "milestone"
            rows.append(
                {
                    "trades": trades,
                    "win_rate": win,
                    "pnl": pnl,
                    "dd": dd,
                    "pnl_over_dd": pnl_over_dd,
                    "note": note,
                }
            )

        if not rows:
            print("No matching spot milestones found for this bar_size/symbol.")
            return

        _print_leaderboards(rows, title="Milestones frontier (current presets)", top_n=int(args.top))

        print("")
        print("Frontier by win-rate constraint (best pnl):")
        for thr in (0.55, 0.58, 0.60, 0.62, 0.65):
            eligible = [r for r in rows if int(r.get("trades") or 0) >= int(run_min_trades) and float(r.get("win_rate") or 0.0) >= thr]
            if not eligible:
                continue
            best = max(eligible, key=lambda r: float(r.get("pnl") or float("-inf")))
            print(
                f"- win>={thr:.2f}: pnl={best['pnl']:.1f} pnl/dd={(best['pnl_over_dd'] or 0):.2f} "
                f"win={best['win_rate']*100:.1f}% tr={best['trades']} note={best.get('note')}"
            )

    def _sweep_combo() -> None:
        """A constrained multi-axis sweep to find "corner" winners.

        Keep this computationally bounded and reproducible. The intent is to combine
        the highest-leverage levers we’ve found so far:
        - direction layer interactions (EMA preset + entry mode)
        - regime sensitivity (Supertrend timeframe + params)
        - exits (pct vs ATR), including the PT<1.0 ATR pocket
        - loosenings (stacking + EOD close)
        - optional regime2 confirm (small curated set)
        - a small set of quality gates (spread/slope/TOD/rv/exit-time/tick)
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

        # Stage 1: direction × regime sensitivity (bounded) and keep a small diverse shortlist.
        stage1: list[tuple[ConfigBundle, dict, str]] = []
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        # Ensure stage1 isn't silently gated by whatever the current milestone base uses.
        base = replace(
            base,
            strategy=replace(
                base.strategy,
                filters=None,
                tick_gate_mode="off",
                spot_exit_time_et=None,
            ),
        )

        direction_variants: list[tuple[str, str, int, str]] = []
        base_preset = str(base.strategy.ema_preset or "").strip()
        base_mode = str(base.strategy.ema_entry_mode or "trend").strip().lower()
        base_confirm = int(base.strategy.entry_confirm_bars or 0)
        if base_preset and base_mode in ("cross", "trend"):
            direction_variants.append((base_preset, base_mode, base_confirm, f"ema={base_preset} {base_mode}"))

        for preset, mode in (
            ("2/4", "cross"),
            ("3/7", "cross"),
            ("3/7", "trend"),
            ("4/9", "cross"),
            ("4/9", "trend"),
            ("5/10", "cross"),
            ("9/21", "cross"),
            ("9/21", "trend"),
        ):
            direction_variants.append((preset, mode, 0, f"ema={preset} {mode}"))

        seen_dir: set[tuple[str, str, int]] = set()
        direction_variants = [
            v
            for v in direction_variants
            if (v[0], v[1], v[2]) not in seen_dir and not seen_dir.add((v[0], v[1], v[2]))
        ]

        regime_bar_sizes = ["4 hours", "1 day"]
        atr_periods = [3, 7, 10, 14, 21]
        multipliers = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        sources = ["close", "hl2"]
        stage1_exit_variants: list[tuple[dict, str]] = [
            (
                {"spot_exit_mode": "pct", "spot_profit_target_pct": 0.015, "spot_stop_loss_pct": 0.03},
                "PT=0.015 SL=0.030",
            ),
            (
                {"spot_exit_mode": "pct", "spot_profit_target_pct": None, "spot_stop_loss_pct": 0.03},
                "PT=off SL=0.030",
            ),
        ]
        stage1_total = (
            len(direction_variants)
            * len(regime_bar_sizes)
            * len(atr_periods)
            * len(multipliers)
            * len(sources)
            * len(stage1_exit_variants)
        )
        stage1_tested = 0
        report_every_stage1 = 200
        stage1_t0 = pytime.perf_counter()
        print(f"Combo sweep: stage1 total={stage1_total} (progress every {report_every_stage1})", flush=True)
        for ema_preset, entry_mode, confirm, dir_note in direction_variants:
            for rbar in regime_bar_sizes:
                for atr_p in atr_periods:
                    for mult in multipliers:
                        for src in sources:
                            for exit_over, exit_note in stage1_exit_variants:
                                cfg = replace(
                                    base,
                                    strategy=replace(
                                        base.strategy,
                                        entry_signal="ema",
                                        ema_preset=str(ema_preset),
                                        ema_entry_mode=str(entry_mode),
                                        entry_confirm_bars=int(confirm),
                                        regime_mode="supertrend",
                                        regime_bar_size=rbar,
                                        supertrend_atr_period=int(atr_p),
                                        supertrend_multiplier=float(mult),
                                        supertrend_source=str(src),
                                        regime2_mode="off",
                                        spot_exit_mode=str(exit_over["spot_exit_mode"]),
                                        spot_profit_target_pct=exit_over.get("spot_profit_target_pct"),
                                        spot_stop_loss_pct=exit_over.get("spot_stop_loss_pct"),
                                    ),
                                )
                                row = _run_cfg(
                                    cfg=cfg,
                                    bars=bars_sig,
                                    regime_bars=regime_bars_by_size[rbar],
                                    regime2_bars=None,
                                )
                                stage1_tested += 1
                                if stage1_tested % report_every_stage1 == 0:
                                    elapsed = pytime.perf_counter() - stage1_t0
                                    rate = (stage1_tested / elapsed) if elapsed > 0 else 0.0
                                    remaining = stage1_total - stage1_tested
                                    eta_sec = (remaining / rate) if rate > 0 else 0.0
                                    pct = (stage1_tested / stage1_total * 100.0) if stage1_total > 0 else 0.0
                                    print(
                                        f"Combo sweep: stage1 {stage1_tested}/{stage1_total} ({pct:0.1f}%) "
                                        f"kept={len(stage1)} elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                        flush=True,
                                    )
                                if not row:
                                    continue
                                note = f"{dir_note} c={confirm} | ST({atr_p},{mult},{src}) @{rbar} | {exit_note}"
                                row["note"] = note
                                stage1.append((cfg, row, note))

        shortlist = _ranked(stage1, top_pnl_dd=15, top_pnl=7)
        print("")
        print(f"Combo sweep: shortlist regimes={len(shortlist)} (from stage1={len(stage1)})")

        # Stage 2: for each shortlisted regime, sweep exits + loosenings, and (optionally) a small regime2 set.
        exit_variants: list[tuple[dict, str]] = []
        for pt, sl in (
            (0.005, 0.02),
            (0.005, 0.03),
            (0.01, 0.03),
            (0.015, 0.03),
            # Higher RR pocket (PT > SL): helps when stop-first intrabar tie-break punishes low-RR setups.
            (0.02, 0.015),
            (0.03, 0.015),
            # Bigger PT/SL pocket: trend systems often need a wider profit capture window.
            (0.05, 0.03),
            (0.08, 0.04),
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
        # Stop-only (no PT): "exit on next cross / regime flip" families.
        for sl in (0.03, 0.05):
            exit_variants.append(
                (
                    {
                        "spot_exit_mode": "pct",
                        "spot_profit_target_pct": None,
                        "spot_stop_loss_pct": float(sl),
                        "spot_atr_period": 14,
                        "spot_pt_atr_mult": 1.5,
                        "spot_sl_atr_mult": 1.0,
                    },
                    f"PT=off SL={sl:.3f}",
                )
            )
        for atr_p, pt_m, sl_m in (
            # Risk-adjusted champ neighborhood.
            (7, 1.0, 1.0),
            (7, 1.0, 1.5),
            (7, 1.12, 1.5),
            # Net-PnL pocket (PTx<1.0).
            (10, 0.80, 1.80),
            (10, 0.90, 1.80),
            (14, 0.70, 1.60),
            (14, 0.75, 1.60),
            (14, 0.80, 1.60),
            (21, 0.65, 1.60),
            (21, 0.70, 1.80),
            # Higher RR pocket (PTx > SLx): try to counter stop-first intrabar ambiguity.
            (14, 2.00, 1.00),
            (21, 2.00, 1.00),
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
            (1, False, "max_open=1 close_eod=0"),
            (2, False, "max_open=2 close_eod=0"),
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
        report_every = 200
        t0 = pytime.perf_counter()
        stage2_total = len(shortlist) * len(exit_variants) * len(hold_vals) * len(loosen_variants) * len(regime2_variants)
        print(f"Combo sweep: stage2 total={stage2_total} (progress every {report_every})", flush=True)
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
                            if tested % report_every == 0:
                                elapsed = pytime.perf_counter() - t0
                                rate = (tested / elapsed) if elapsed > 0 else 0.0
                                remaining = stage2_total - tested
                                eta_sec = (remaining / rate) if rate > 0 else 0.0
                                pct = (tested / stage2_total * 100.0) if stage2_total > 0 else 0.0
                                print(
                                    f"Combo sweep: stage2 {tested}/{stage2_total} ({pct:0.1f}%) "
                                    f"kept={len(stage2)} elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                    flush=True,
                                )
                            if not row:
                                continue
                            note = f"{base_note} | {exit_note} | hold={hold} | {loose_note} | {r2_note}"
                            row["note"] = note
                            _record_milestone(cfg, row, note)
                            stage2.append((cfg, row, note))

        print(f"Combo sweep: stage2 tested={tested} kept={len(stage2)} (min_trades={run_min_trades})")

        # Stage 3: apply a small set of quality gates on the top stage2 candidates.
        top_stage2 = _ranked(stage2, top_pnl_dd=15, top_pnl=7)

        tick_variants: list[tuple[dict, str]] = [
            ({"tick_gate_mode": "off"}, "tick=off"),
            (
                {
                    "tick_gate_mode": "raschke",
                    "tick_gate_symbol": "TICK-AMEX",
                    "tick_gate_exchange": "AMEX",
                    "tick_neutral_policy": "block",
                    "tick_direction_policy": "wide_only",
                    "tick_band_ma_period": 10,
                    "tick_width_z_lookback": 252,
                    "tick_width_z_enter": 1.0,
                    "tick_width_z_exit": 0.5,
                    "tick_width_slope_lookback": 3,
                },
                "tick=raschke(wide_only block z=1.0/0.5 slope=3 lb=252)",
            ),
        ]

        quality_variants: list[tuple[float | None, float | None, float | None, str]] = [
            # (spread_min, spread_min_down, slope_min, note)
            (None, None, None, "qual=off"),
            (0.003, None, None, "spread>=0.003"),
            (0.003, 0.006, None, "spread>=0.003 down>=0.006"),
            (0.003, 0.008, None, "spread>=0.003 down>=0.008"),
            (0.003, 0.010, None, "spread>=0.003 down>=0.010"),
            (0.003, 0.015, None, "spread>=0.003 down>=0.015"),
            (0.003, 0.030, None, "spread>=0.003 down>=0.030"),
            (0.003, 0.050, None, "spread>=0.003 down>=0.050"),
            (0.005, None, None, "spread>=0.005"),
            (0.005, 0.010, None, "spread>=0.005 down>=0.010"),
            (0.005, 0.012, None, "spread>=0.005 down>=0.012"),
            (0.005, 0.015, None, "spread>=0.005 down>=0.015"),
            (0.005, 0.030, None, "spread>=0.005 down>=0.030"),
            (0.005, 0.050, None, "spread>=0.005 down>=0.050"),
            (0.005, 0.010, 0.01, "spread>=0.005 down>=0.010 slope>=0.01"),
        ]

        rv_variants: list[tuple[float | None, float | None, str]] = [
            (None, None, "rv=off"),
            (0.25, 0.8, "rv=0.25..0.80"),
        ]

        exit_time_variants: list[tuple[str | None, str]] = [
            (None, "exit_time=off"),
            ("17:00", "exit_time=17:00 ET"),
        ]

        tod_variants: list[tuple[int | None, int | None, int, int, str]] = [
            (None, None, 0, 0, "tod=any"),
            (18, 4, 0, 0, "tod=18-04 ET"),
            (18, 4, 1, 2, "tod=18-04 ET (skip=1 cd=2)"),
            (10, 15, 0, 0, "tod=10-15 ET"),
        ]

        stage3_total = (
            len(top_stage2)
            * len(tick_variants)
            * len(quality_variants)
            * len(rv_variants)
            * len(exit_time_variants)
            * len(tod_variants)
        )
        stage3_tested = 0
        stage3_t0 = pytime.perf_counter()
        report_every_stage3 = 200
        print(f"Combo sweep: stage3 total={stage3_total} (progress every {report_every_stage3})", flush=True)

        stage3: list[dict] = []
        for base_cfg, base_row, base_note in top_stage2:
            for tick_over, tick_note in tick_variants:
                for spread_min, spread_min_down, slope_min, qual_note in quality_variants:
                    for rv_min, rv_max, rv_note in rv_variants:
                        for exit_time, exit_time_note in exit_time_variants:
                            for tod_s, tod_e, skip, cooldown, tod_note in tod_variants:
                                f = _mk_filters(
                                    rv_min=rv_min,
                                    rv_max=rv_max,
                                    ema_spread_min_pct=spread_min,
                                    ema_spread_min_pct_down=spread_min_down,
                                    ema_slope_min_pct=slope_min,
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
                                        spot_exit_time_et=exit_time,
                                        tick_gate_mode=str(tick_over.get("tick_gate_mode") or "off"),
                                        tick_gate_symbol=str(tick_over.get("tick_gate_symbol") or "TICK-NYSE"),
                                        tick_gate_exchange=str(tick_over.get("tick_gate_exchange") or "NYSE"),
                                        tick_neutral_policy=str(tick_over.get("tick_neutral_policy") or "allow"),
                                        tick_direction_policy=str(tick_over.get("tick_direction_policy") or "both"),
                                        tick_band_ma_period=int(tick_over.get("tick_band_ma_period") or 10),
                                        tick_width_z_lookback=int(tick_over.get("tick_width_z_lookback") or 252),
                                        tick_width_z_enter=float(tick_over.get("tick_width_z_enter") or 1.0),
                                        tick_width_z_exit=float(tick_over.get("tick_width_z_exit") or 0.5),
                                        tick_width_slope_lookback=int(tick_over.get("tick_width_slope_lookback") or 3),
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
                                stage3_tested += 1
                                if stage3_tested % report_every_stage3 == 0:
                                    elapsed = pytime.perf_counter() - stage3_t0
                                    rate = (stage3_tested / elapsed) if elapsed > 0 else 0.0
                                    remaining = stage3_total - stage3_tested
                                    eta_sec = (remaining / rate) if rate > 0 else 0.0
                                    pct = (stage3_tested / stage3_total * 100.0) if stage3_total > 0 else 0.0
                                    print(
                                        f"Combo sweep: stage3 {stage3_tested}/{stage3_total} ({pct:0.1f}%) "
                                        f"kept={len(stage3)} elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m",
                                        flush=True,
                                    )
                                if not row:
                                    continue
                                note = (
                                    f"{base_note} | {tick_note} | {qual_note} | {rv_note} | "
                                    f"{exit_time_note} | {tod_note}"
                                )
                                row["note"] = note
                                _record_milestone(cfg, row, note)
                                stage3.append(row)

        _print_leaderboards(stage3, title="Combo sweep (multi-axis, constrained)", top_n=int(args.top))

    def _sweep_squeeze() -> None:
        """Squeeze a few high-leverage axes from the current champion baseline.

        Targeted (fast): regime2 timeframe, volume gate, and time-of-day windows,
        including small combinations of these axes.
        """
        bars_sig = _bars_cached(signal_bar_size)
        base = _base_bundle(bar_size=signal_bar_size, filters=None)
        base_row = _run_cfg(
            cfg=base, bars=bars_sig, regime_bars=_regime_bars_for(base), regime2_bars=_regime2_bars_for(base)
        )
        if base_row:
            base_row["note"] = "base"
            _record_milestone(base, base_row, "base")

        def _shortlist(items: list[tuple[ConfigBundle, dict, str]], *, top_pnl_dd: int, top_pnl: int) -> list:
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

        # Stage 1: sweep regime2 timeframe + params (bounded), with no extra filters.
        stage1: list[tuple[ConfigBundle, dict, str]] = []
        stage1.append((base, base_row, "base") if base_row else (base, {}, "base"))
        atr_periods = [2, 3, 4, 5, 6, 7, 10, 11]
        multipliers = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
        sources = ["close", "hl2"]
        for r2_bar in ("4 hours", "1 day"):
            for atr_p in atr_periods:
                for mult in multipliers:
                    for src in sources:
                        cfg = replace(
                            base,
                            strategy=replace(
                                base.strategy,
                                regime2_mode="supertrend",
                                regime2_bar_size=r2_bar,
                                regime2_supertrend_atr_period=int(atr_p),
                                regime2_supertrend_multiplier=float(mult),
                                regime2_supertrend_source=str(src),
                                filters=None,
                                entry_confirm_bars=0,
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
                        note = f"r2=ST({atr_p},{mult},{src})@{r2_bar}"
                        row["note"] = note
                        stage1.append((cfg, row, note))

        stage1 = [t for t in stage1 if t[1]]
        shortlisted = _shortlist(stage1, top_pnl_dd=15, top_pnl=10)
        print("")
        print(f"Squeeze sweep: stage1 candidates={len(stage1)} shortlist={len(shortlisted)} (min_trades={run_min_trades})")

        # Stage 2: apply volume + TOD + confirm gates on the shortlist (small combos).
        vol_variants = [
            (None, None, "vol=-"),
            (1.0, 20, "vol>=1.0@20"),
            (1.1, 20, "vol>=1.1@20"),
            (1.2, 20, "vol>=1.2@20"),
            (1.5, 10, "vol>=1.5@10"),
            (1.5, 20, "vol>=1.5@20"),
        ]
        tod_variants = [
            (None, None, "tod=base"),
            (18, 3, "tod=18-03 ET"),
            (9, 16, "tod=9-16 ET"),
            (10, 15, "tod=10-15 ET"),
            (11, 16, "tod=11-16 ET"),
        ]
        confirm_variants = [(0, "confirm=0"), (1, "confirm=1"), (2, "confirm=2")]

        rows: list[dict] = []
        for base_cfg, _, base_note in shortlisted:
            for vratio, vema, v_note in vol_variants:
                for tod_s, tod_e, tod_note in tod_variants:
                    for confirm, confirm_note in confirm_variants:
                        f = _mk_filters(
                            volume_ratio_min=vratio,
                            volume_ema_period=vema,
                            entry_start_hour_et=tod_s,
                            entry_end_hour_et=tod_e,
                        )
                        cfg = replace(
                            base_cfg,
                            strategy=replace(base_cfg.strategy, filters=f, entry_confirm_bars=int(confirm)),
                        )
                        row = _run_cfg(
                            cfg=cfg,
                            bars=bars_sig,
                            regime_bars=_regime_bars_for(cfg),
                            regime2_bars=_regime2_bars_for(cfg),
                        )
                        if not row:
                            continue
                        note = f"{base_note} | {v_note} | {tod_note} | {confirm_note}"
                        row["note"] = note
                        _record_milestone(cfg, row, note)
                        rows.append(row)

        if base_row:
            rows.append(base_row)
        _print_leaderboards(rows, title="Squeeze sweep (regime2 tf+params → vol/TOD/confirm)", top_n=int(args.top))

    axis = str(args.axis).strip().lower()
    print(
        f"{symbol} spot evolve sweep ({start.isoformat()} -> {end.isoformat()}, use_rth={use_rth}, "
        f"bar_size={signal_bar_size}, offline={offline}, base={args.base}, axis={axis}, "
        f"long_only={long_only} realism={'v2' if realism2 else ('v1' if realism else 'off')} "
        f"spread={spot_spread:g} comm={spot_commission:g} comm_min={spot_commission_min:g} "
        f"slip={spot_slippage:g} sizing={sizing_mode} risk={spot_risk_pct:g} max_notional={spot_max_notional_pct:g})"
    )

    if axis in ("all", "ema"):
        _sweep_ema()
    if axis == "entry_mode":
        _sweep_entry_mode()
    if axis == "combo":
        _sweep_combo()
    if axis == "squeeze":
        _sweep_squeeze()
    if axis in ("all", "volume"):
        _sweep_volume()
    if axis in ("all", "rv"):
        _sweep_rv()
    if axis in ("all", "tod"):
        _sweep_tod()
    if axis == "tod_interaction":
        _sweep_tod_interaction()
    if axis == "perm_joint":
        _sweep_perm_joint()
    if axis == "ema_perm_joint":
        _sweep_ema_perm_joint()
    if axis == "tick_perm_joint":
        _sweep_tick_perm_joint()
    if axis == "ema_regime":
        _sweep_ema_regime()
    if axis == "chop_joint":
        _sweep_chop_joint()
    if axis == "tick_ema":
        _sweep_tick_ema()
    if axis == "ema_atr":
        _sweep_ema_atr()
    if axis == "weekday":
        _sweep_weekdays()
    if axis == "exit_time":
        _sweep_exit_time()
    if axis in ("all", "atr"):
        _sweep_atr_exits()
    if axis == "atr_fine":
        _sweep_atr_exits_fine()
    if axis == "atr_ultra":
        _sweep_atr_exits_ultra()
    if axis == "r2_atr":
        _sweep_r2_atr()
    if axis == "r2_tod":
        _sweep_r2_tod()
    if axis == "regime_atr":
        _sweep_regime_atr()
    if axis in ("all", "ptsl"):
        _sweep_ptsl()
    if axis in ("all", "hold"):
        _sweep_hold()
    if axis in ("all", "orb"):
        _sweep_orb()
    if axis == "orb_joint":
        _sweep_orb_joint()
    if axis in ("all", "regime"):
        _sweep_regime()
    if axis in ("all", "regime2"):
        _sweep_regime2()
    if axis == "regime2_ema":
        _sweep_regime2_ema()
    if axis in ("all", "joint"):
        _sweep_joint()
    if axis == "micro_st":
        _sweep_micro_st()
    if axis in ("all", "flip_exit"):
        _sweep_flip_exit()
    if axis in ("all", "confirm"):
        _sweep_confirm()
    if axis in ("all", "spread"):
        _sweep_spread()
    if axis == "spread_fine":
        _sweep_spread_fine()
    if axis == "spread_down":
        _sweep_spread_down()
    if axis in ("all", "slope"):
        _sweep_slope()
    if axis in ("all", "cooldown"):
        _sweep_cooldown()
    if axis in ("all", "skip_open"):
        _sweep_skip_open()
    if axis in ("all", "loosen"):
        _sweep_loosen()
    if axis == "loosen_atr":
        _sweep_loosen_atr()
    if axis in ("all", "tick"):
        _sweep_tick()
    if axis == "frontier":
        _sweep_frontier()

    if bool(args.write_milestones):
        eligible_new: list[dict] = []
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
            eligible_new.append(
                {
                    "key": json.dumps(key_obj, sort_keys=True, default=str),
                    "strategy": strategy,
                    "filters": filters,
                    "note": note,
                    "metrics": {
                        "pnl": float(row.get("pnl") or 0.0),
                        "roi": float(row.get("roi") or 0.0),
                        "win_rate": float(row.get("win_rate") or 0.0),
                        "trades": int(row.get("trades") or 0),
                        "max_drawdown": float(row.get("dd") or 0.0),
                        "max_drawdown_pct": float(row.get("dd_pct") or 0.0),
                        "pnl_over_dd": row.get("pnl_over_dd"),
                    },
                }
            )

        out_path = Path(args.milestones_out)
        eligible: list[dict] = []

        def _sort_key(item: dict) -> tuple:
            m = item.get("metrics") or {}
            return (
                float(m.get("pnl_over_dd") or float("-inf")),
                float(m.get("pnl") or 0.0),
                float(m.get("win_rate") or 0.0),
                int(m.get("trades") or 0),
            )

        def _sort_key_pnl(item: dict) -> tuple:
            m = item.get("metrics") or {}
            return (
                float(m.get("pnl") or float("-inf")),
                float(m.get("pnl_over_dd") or 0.0),
                float(m.get("win_rate") or 0.0),
                int(m.get("trades") or 0),
            )

        add_top_dd = max(0, int(args.milestone_add_top_pnl_dd or 0))
        add_top_pnl = max(0, int(args.milestone_add_top_pnl or 0))
        if bool(args.merge_milestones) and (add_top_dd > 0 or add_top_pnl > 0):
            by_dd = sorted(eligible_new, key=_sort_key, reverse=True)[:add_top_dd] if add_top_dd > 0 else []
            by_pnl = (
                sorted(eligible_new, key=_sort_key_pnl, reverse=True)[:add_top_pnl] if add_top_pnl > 0 else []
            )
            seen_new: set[str] = set()
            limited_new: list[dict] = []
            for item in by_dd + by_pnl:
                key = str(item.get("key") or "")
                if not key or key in seen_new:
                    continue
                seen_new.add(key)
                limited_new.append(item)
            eligible.extend(limited_new)
        else:
            eligible.extend(eligible_new)

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
                                "roi": float(metrics.get("roi") or 0.0),
                                "win_rate": float(metrics.get("win_rate") or 0.0),
                                "trades": int(metrics.get("trades") or 0),
                                "max_drawdown": float(metrics.get("max_drawdown") or 0.0),
                                "max_drawdown_pct": float(metrics.get("max_drawdown_pct") or 0.0),
                                "pnl_over_dd": metrics.get("pnl_over_dd"),
                            },
                        }
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
