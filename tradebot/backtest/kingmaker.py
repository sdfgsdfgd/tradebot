"""Kingmaker evaluator: stability scoring across multiple windows (spot).

This is intentionally small and pragmatic:
- Load candidates from a spot milestones JSON (e.g. combo sweep output)
- Re-run each candidate across multiple date windows
- Rank by worst-window pnl/dd (stability) and print a compact report

Primary use case: TQQQ long-only "realism v1" stability hunt.
"""

from __future__ import annotations

import argparse
import json
import math
import time as pytime
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .config import BacktestConfig, ConfigBundle, FiltersConfig, LegConfig, SpotLegConfig, StrategyConfig, SyntheticConfig
from .data import ContractMeta, IBKRHistoricalData
from .engine import _run_spot_backtest, _spot_multiplier

_WDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _parse_date(value: str) -> date:
    year_s, month_s, day_s = str(value).strip().split("-")
    return date(int(year_s), int(month_s), int(day_s))


def _parse_window(value: str) -> tuple[date, date]:
    raw = str(value).strip()
    if ":" not in raw:
        raise ValueError("Window must be formatted like YYYY-MM-DD:YYYY-MM-DD")
    start_s, end_s = raw.split(":", 1)
    return _parse_date(start_s), _parse_date(end_s)


def _weekdays_from_payload(value) -> tuple[int, ...]:
    if not value:
        return (0, 1, 2, 3, 4)
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(item)
            continue
        key = str(item).strip().upper()[:3]
        if key in _WDAYS:
            out.append(_WDAYS[key])
    return tuple(out) if out else (0, 1, 2, 3, 4)


def _spot_leg_from_payload(raw) -> SpotLegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"directional_spot leg must be an object, got: {raw!r}")
    action = str(raw.get("action") or "").strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"directional_spot.action must be BUY/SELL, got: {action!r}")
    qty = int(raw.get("qty") or 1)
    if qty <= 0:
        raise ValueError(f"directional_spot.qty must be positive, got: {qty!r}")
    return SpotLegConfig(action=action, qty=qty)


def _leg_from_payload(raw) -> LegConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"leg must be an object, got: {raw!r}")
    action = str(raw.get("action") or "").strip().upper()
    right = str(raw.get("right") or "").strip().upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"leg.action must be BUY/SELL, got: {action!r}")
    if right not in ("PUT", "CALL"):
        raise ValueError(f"leg.right must be PUT/CALL, got: {right!r}")
    moneyness = float(raw.get("moneyness_pct") or 0.0)
    qty = int(raw.get("qty") or 1)
    if qty <= 0:
        raise ValueError(f"leg.qty must be positive, got: {qty!r}")
    return LegConfig(action=action, right=right, moneyness_pct=moneyness, qty=qty)


def _filters_from_payload(raw) -> FiltersConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"filters must be an object, got: {raw!r}")
    return FiltersConfig(
        rv_min=(float(raw["rv_min"]) if raw.get("rv_min") is not None else None),
        rv_max=(float(raw["rv_max"]) if raw.get("rv_max") is not None else None),
        ema_spread_min_pct=(
            float(raw["ema_spread_min_pct"]) if raw.get("ema_spread_min_pct") is not None else None
        ),
        ema_slope_min_pct=(
            float(raw["ema_slope_min_pct"]) if raw.get("ema_slope_min_pct") is not None else None
        ),
        entry_start_hour=(int(raw["entry_start_hour"]) if raw.get("entry_start_hour") is not None else None),
        entry_end_hour=(int(raw["entry_end_hour"]) if raw.get("entry_end_hour") is not None else None),
        skip_first_bars=int(raw.get("skip_first_bars") or 0),
        cooldown_bars=int(raw.get("cooldown_bars") or 0),
        entry_start_hour_et=(int(raw["entry_start_hour_et"]) if raw.get("entry_start_hour_et") is not None else None),
        entry_end_hour_et=(int(raw["entry_end_hour_et"]) if raw.get("entry_end_hour_et") is not None else None),
        volume_ema_period=(int(raw["volume_ema_period"]) if raw.get("volume_ema_period") is not None else None),
        volume_ratio_min=(float(raw["volume_ratio_min"]) if raw.get("volume_ratio_min") is not None else None),
    )


def _strategy_from_payload(strategy: dict, *, filters: FiltersConfig | None) -> StrategyConfig:
    if not isinstance(strategy, dict):
        raise ValueError(f"strategy must be an object, got: {strategy!r}")

    raw = dict(strategy)
    raw.pop("signal_bar_size", None)
    raw.pop("signal_use_rth", None)
    raw.pop("spot_sec_type", None)
    raw.pop("spot_exchange", None)

    entry_days = _weekdays_from_payload(raw.get("entry_days") or [])
    raw["entry_days"] = entry_days

    raw["filters"] = filters

    # Normalize nested structures back into dataclasses.
    dspot = raw.get("directional_spot")
    if dspot is not None:
        if not isinstance(dspot, dict):
            raise ValueError(f"directional_spot must be an object, got: {dspot!r}")
        parsed: dict[str, SpotLegConfig] = {}
        for k, v in dspot.items():
            key = str(k).strip()
            if not key:
                continue
            parsed[key] = _spot_leg_from_payload(v)
        raw["directional_spot"] = parsed or None

    dlegs = raw.get("directional_legs")
    if dlegs is not None:
        if not isinstance(dlegs, dict):
            raise ValueError(f"directional_legs must be an object, got: {dlegs!r}")
        parsed_dl: dict[str, tuple[LegConfig, ...]] = {}
        for k, legs in dlegs.items():
            key = str(k).strip()
            if not key or not legs:
                continue
            if not isinstance(legs, list):
                continue
            parsed_dl[key] = tuple(_leg_from_payload(l) for l in legs)
        raw["directional_legs"] = parsed_dl or None

    legs = raw.get("legs")
    if legs is not None:
        if not isinstance(legs, list):
            raise ValueError(f"legs must be a list, got: {legs!r}")
        raw["legs"] = tuple(_leg_from_payload(l) for l in legs)

    return StrategyConfig(**raw)


def _mk_bundle(
    *,
    strategy: StrategyConfig,
    start: date,
    end: date,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
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


def _metrics_from_summary(summary) -> dict:
    pnl = float(getattr(summary, "total_pnl", 0.0) or 0.0)
    dd = float(getattr(summary, "max_drawdown", 0.0) or 0.0)
    trades = int(getattr(summary, "trades", 0) or 0)
    win_rate = float(getattr(summary, "win_rate", 0.0) or 0.0)
    pnl_dd = pnl / dd if dd > 0 else (math.inf if pnl > 0 else -math.inf if pnl < 0 else 0.0)
    return {"trades": trades, "win_rate": win_rate, "pnl": pnl, "dd": dd, "pnl_over_dd": pnl_dd}


def _load_bars(
    data: IBKRHistoricalData,
    *,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    offline: bool,
) -> list:
    if offline:
        return data.load_cached_bars(
            symbol=symbol,
            exchange=exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
    return data.load_or_fetch_bars(
        symbol=symbol,
        exchange=exchange,
        start=start_dt,
        end=end_dt,
        bar_size=bar_size,
        use_rth=use_rth,
        cache_dir=cache_dir,
    )


def _score_key(item: dict) -> tuple:
    return (
        float(item.get("stability_min_pnl_dd") or float("-inf")),
        float(item.get("stability_min_pnl") or float("-inf")),
        float(item.get("full_pnl_over_dd") or float("-inf")),
        float(item.get("full_pnl") or float("-inf")),
        float(item.get("full_win") or 0.0),
        int(item.get("full_trades") or 0),
    )


def _strategy_key(strategy: dict, *, filters: dict | None) -> str:
    raw = dict(strategy)
    raw["filters"] = filters
    return json.dumps(raw, sort_keys=True, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(prog="tradebot.backtest.kingmaker")
    ap.add_argument("--milestones", required=True, help="Input spot milestones JSON to evaluate.")
    ap.add_argument("--symbol", default="TQQQ", help="Symbol to filter (default: TQQQ).")
    ap.add_argument("--bar-size", default="1 hour", help="Signal bar size filter (default: 1 hour).")
    ap.add_argument("--use-rth", action="store_true", help="Filter to RTH-only strategies.")
    ap.add_argument("--offline", action="store_true", help="Use cached bars only (no IBKR fetch).")
    ap.add_argument("--cache-dir", default="db", help="Bars cache dir (default: db).")
    ap.add_argument("--top", type=int, default=200, help="How many candidates to evaluate (after sorting).")
    ap.add_argument("--min-trades", type=int, default=200, help="Min trades per window.")
    ap.add_argument("--min-win", type=float, default=0.0, help="Min win rate per window (0..1).")
    ap.add_argument(
        "--max-open",
        type=int,
        default=None,
        help=(
            "Require strategy.max_open_trades to be <= this value. "
            "Note: 0 means unlimited stacking and is rejected unless --allow-unlimited-stacking is set."
        ),
    )
    ap.add_argument(
        "--allow-unlimited-stacking",
        action="store_true",
        default=False,
        help="Allow max_open_trades=0 strategies (unlimited stacking).",
    )
    ap.add_argument(
        "--require-close-eod",
        action="store_true",
        default=False,
        help="Require spot_close_eod=true (forces strategies to close at end of day).",
    )
    ap.add_argument(
        "--require-positive-pnl",
        action="store_true",
        default=False,
        help="Require pnl>0 in every evaluation window.",
    )
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="Evaluation window formatted YYYY-MM-DD:YYYY-MM-DD. Repeatable.",
    )
    ap.add_argument(
        "--include-full",
        action="store_true",
        help="Also evaluate the full window from the milestones payload notes (best-effort).",
    )
    ap.add_argument(
        "--write-top",
        type=int,
        default=0,
        help="Write a small milestones JSON of the top K stability winners (0 disables).",
    )
    ap.add_argument(
        "--out",
        default="backtests/out/kingmaker_top.json",
        help="Output file for --write-top (default: backtests/out/kingmaker_top.json).",
    )

    args = ap.parse_args()

    milestones_path = Path(args.milestones)
    payload = json.loads(milestones_path.read_text())
    groups = payload.get("groups") or []
    symbol = str(args.symbol).strip().upper()
    bar_size = str(args.bar_size).strip().lower()
    use_rth = bool(args.use_rth)

    candidates: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        filters_payload = group.get("filters")
        entries = group.get("entries") or []
        if not isinstance(entries, list) or not entries:
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
        if str(strat.get("symbol") or "").strip().upper() != symbol:
            continue
        if str(strat.get("signal_bar_size") or "").strip().lower() != bar_size:
            continue
        if bool(strat.get("signal_use_rth")) != use_rth:
            continue
        candidates.append(
            {
                "group_name": str(group.get("name") or ""),
                "filters": filters_payload,
                "strategy": strat,
                "metrics": metrics,
            }
        )

    if not candidates:
        raise SystemExit(f"No candidates found for {symbol} bar={bar_size} rth={use_rth} in {milestones_path}")

    def _sort_key_seed(item: dict) -> tuple:
        m = item.get("metrics") or {}
        return (
            float(m.get("pnl_over_dd") or float("-inf")),
            float(m.get("pnl") or float("-inf")),
            float(m.get("win_rate") or 0.0),
            int(m.get("trades") or 0),
        )

    candidates = sorted(candidates, key=_sort_key_seed, reverse=True)[: max(1, int(args.top))]

    windows: list[tuple[date, date]] = []
    for raw in args.window or []:
        windows.append(_parse_window(raw))
    if not windows:
        windows = [
            (_parse_date("2023-01-01"), _parse_date("2024-01-01")),
            (_parse_date("2024-01-01"), _parse_date("2025-01-01")),
            (_parse_date("2025-01-01"), date.today()),
        ]

    cache_dir = Path(args.cache_dir)
    offline = bool(args.offline)

    data = IBKRHistoricalData()
    bars_cache: dict[tuple[str, str | None, str, str, str, bool, bool], list] = {}

    def _load_bars_cached(
        *,
        symbol: str,
        exchange: str | None,
        start_dt: datetime,
        end_dt: datetime,
        bar_size: str,
        use_rth: bool,
        offline: bool,
    ) -> list:
        key = (
            str(symbol),
            str(exchange) if exchange is not None else None,
            start_dt.isoformat(),
            end_dt.isoformat(),
            str(bar_size),
            bool(use_rth),
            bool(offline),
        )
        cached = bars_cache.get(key)
        if cached is not None:
            return cached
        bars = _load_bars(
            data,
            symbol=symbol,
            exchange=exchange,
            start_dt=start_dt,
            end_dt=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
            offline=offline,
        )
        bars_cache[key] = bars
        return bars

    out_rows: list[dict] = []
    started = pytime.perf_counter()
    report_every = 10

    for idx, cand in enumerate(candidates, start=1):
        filters_payload = cand.get("filters")
        filters = _filters_from_payload(filters_payload)
        strategy_payload = cand["strategy"]
        strat_cfg = _strategy_from_payload(strategy_payload, filters=filters)
        if bool(args.require_close_eod) and not bool(getattr(strat_cfg, "spot_close_eod", False)):
            continue
        if args.max_open is not None:
            max_open = int(getattr(strat_cfg, "max_open_trades", 0) or 0)
            if max_open == 0 and not bool(args.allow_unlimited_stacking):
                continue
            if max_open != 0 and max_open > int(args.max_open):
                continue

        sig_bar_size = str(strategy_payload.get("signal_bar_size") or args.bar_size)
        sig_use_rth = bool(strategy_payload.get("signal_use_rth"))

        per_window: list[dict] = []
        ok = True
        for wstart, wend in windows:
            bundle = _mk_bundle(
                strategy=strat_cfg,
                start=wstart,
                end=wend,
                bar_size=sig_bar_size,
                use_rth=sig_use_rth,
                cache_dir=cache_dir,
                offline=offline,
            )

            start_dt = datetime.combine(bundle.backtest.start, time(0, 0))
            end_dt = datetime.combine(bundle.backtest.end, time(23, 59))
            bars_sig = _load_bars_cached(
                symbol=bundle.strategy.symbol,
                exchange=bundle.strategy.exchange,
                start_dt=start_dt,
                end_dt=end_dt,
                bar_size=bundle.backtest.bar_size,
                use_rth=bundle.backtest.use_rth,
                offline=bundle.backtest.offline,
            )
            if not bars_sig:
                ok = False
                break

            is_future = bundle.strategy.symbol in ("MNQ", "MBT")
            if offline:
                exchange = "CME" if is_future else "SMART"
                meta = ContractMeta(
                    symbol=bundle.strategy.symbol,
                    exchange=exchange,
                    multiplier=_spot_multiplier(bundle.strategy.symbol, is_future),
                    min_tick=0.01,
                )
            else:
                _, meta = data.resolve_contract(bundle.strategy.symbol, bundle.strategy.exchange)
                meta = ContractMeta(
                    symbol=meta.symbol,
                    exchange=meta.exchange,
                    multiplier=_spot_multiplier(bundle.strategy.symbol, is_future, default=meta.multiplier),
                    min_tick=meta.min_tick,
                )

            # Load multi-timeframe regime/regime2 bars only when needed.
            regime_bars = None
            if str(bundle.strategy.regime_mode).strip().lower() == "supertrend":
                if bundle.strategy.regime_bar_size and str(bundle.strategy.regime_bar_size) != str(bundle.backtest.bar_size):
                    regime_bars = _load_bars_cached(
                        symbol=bundle.strategy.symbol,
                        exchange=bundle.strategy.exchange,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=str(bundle.strategy.regime_bar_size),
                        use_rth=bundle.backtest.use_rth,
                        offline=bundle.backtest.offline,
                    )
            else:
                if (
                    bundle.strategy.regime_ema_preset
                    and bundle.strategy.regime_bar_size
                    and str(bundle.strategy.regime_bar_size) != str(bundle.backtest.bar_size)
                ):
                    regime_bars = _load_bars_cached(
                        symbol=bundle.strategy.symbol,
                        exchange=bundle.strategy.exchange,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=str(bundle.strategy.regime_bar_size),
                        use_rth=bundle.backtest.use_rth,
                        offline=bundle.backtest.offline,
                    )

            regime2_bars = None
            if str(bundle.strategy.regime2_mode or "off").strip().lower() != "off":
                r2bar = str(bundle.strategy.regime2_bar_size or bundle.backtest.bar_size)
                if r2bar != str(bundle.backtest.bar_size):
                    regime2_bars = _load_bars_cached(
                        symbol=bundle.strategy.symbol,
                        exchange=bundle.strategy.exchange,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bar_size=r2bar,
                        use_rth=bundle.backtest.use_rth,
                        offline=bundle.backtest.offline,
                    )

            tick_bars = None
            tick_mode = str(getattr(bundle.strategy, "tick_gate_mode", "off") or "off").strip().lower()
            if tick_mode != "off":
                try:
                    z_lookback = int(getattr(bundle.strategy, "tick_width_z_lookback", 252) or 252)
                except (TypeError, ValueError):
                    z_lookback = 252
                try:
                    ma_period = int(getattr(bundle.strategy, "tick_band_ma_period", 10) or 10)
                except (TypeError, ValueError):
                    ma_period = 10
                try:
                    slope_lb = int(getattr(bundle.strategy, "tick_width_slope_lookback", 3) or 3)
                except (TypeError, ValueError):
                    slope_lb = 3
                tick_warm_days = max(60, z_lookback + ma_period + slope_lb + 5)
                tick_start_dt = start_dt - timedelta(days=tick_warm_days)
                tick_symbol = str(getattr(bundle.strategy, "tick_gate_symbol", "TICK-NYSE") or "TICK-NYSE").strip()
                tick_exchange = str(getattr(bundle.strategy, "tick_gate_exchange", "NYSE") or "NYSE").strip()
                tick_bars = _load_bars_cached(
                    symbol=tick_symbol,
                    exchange=tick_exchange,
                    start_dt=tick_start_dt,
                    end_dt=end_dt,
                    bar_size="1 day",
                    use_rth=True,
                    offline=bundle.backtest.offline,
                )

            result = _run_spot_backtest(
                bundle,
                bars_sig,
                meta,
                regime_bars=regime_bars,
                regime2_bars=regime2_bars,
                tick_bars=tick_bars,
            )
            m = _metrics_from_summary(result.summary)
            if bool(args.require_positive_pnl) and float(m["pnl"]) <= 0:
                ok = False
                break
            if m["trades"] < int(args.min_trades) or m["win_rate"] < float(args.min_win):
                ok = False
                break
            per_window.append(
                {
                    "start": wstart.isoformat(),
                    "end": wend.isoformat(),
                    **m,
                }
            )

        if not ok or not per_window:
            continue

        min_pnl_dd = min(float(x["pnl_over_dd"]) for x in per_window)
        min_pnl = min(float(x["pnl"]) for x in per_window)
        full_m = cand.get("metrics") or {}
        out_rows.append(
            {
                "key": _strategy_key(strategy_payload, filters=filters_payload),
                "strategy": strategy_payload,
                "filters": filters_payload,
                "seed_group_name": cand.get("group_name"),
                "full_trades": int(full_m.get("trades") or 0),
                "full_win": float(full_m.get("win_rate") or 0.0),
                "full_pnl": float(full_m.get("pnl") or 0.0),
                "full_dd": float(full_m.get("max_drawdown") or 0.0),
                "full_pnl_over_dd": float(full_m.get("pnl_over_dd") or 0.0),
                "stability_min_pnl_dd": min_pnl_dd,
                "stability_min_pnl": min_pnl,
                "windows": per_window,
            }
        )

        if idx % report_every == 0:
            elapsed = pytime.perf_counter() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(f"[{idx}/{len(candidates)}] evaluated… ({rate:0.2f} cands/s)", flush=True)

    if not offline:
        data.disconnect()

    out_rows = sorted(out_rows, key=_score_key, reverse=True)
    print("")
    print(f"Kingmaker results: {len(out_rows)} candidates passed filters.")
    print(f"- symbol={symbol} bar={args.bar_size} rth={use_rth} offline={offline}")
    print(f"- windows={', '.join([f'{a.isoformat()}→{b.isoformat()}' for a,b in windows])}")
    print(f"- min_trades={int(args.min_trades)} min_win={float(args.min_win):0.2f}")
    print("")

    show = min(20, len(out_rows))
    for rank, item in enumerate(out_rows[:show], start=1):
        st = item["strategy"]
        print(
            f"{rank:2d}. stability(min pnl/dd)={item['stability_min_pnl_dd']:.2f} "
            f"full pnl/dd={item['full_pnl_over_dd']:.2f} pnl={item['full_pnl']:.1f} "
            f"win={item['full_win']*100:.1f}% tr={item['full_trades']} "
            f"ema={st.get('ema_preset')} {st.get('ema_entry_mode')} "
            f"regime={st.get('regime_mode')} rbar={st.get('regime_bar_size')}"
        )

    if int(args.write_top or 0) > 0:
        top_k = max(1, int(args.write_top))
        now = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        groups_out: list[dict] = []
        for idx, item in enumerate(out_rows[:top_k], start=1):
            m = item.get("metrics") or {}
            strategy = item["strategy"]
            filters_payload = item.get("filters")
            key = _strategy_key(strategy, filters=filters_payload)
            # Use the full-window metrics for the saved preset. (Stability details belong in README.)
            metrics = {
                "pnl": float(item.get("full_pnl") or 0.0),
                "win_rate": float(item.get("full_win") or 0.0),
                "trades": int(item.get("full_trades") or 0),
                "max_drawdown": float(item.get("full_dd") or 0.0),
                "pnl_over_dd": float(item.get("full_pnl_over_dd") or 0.0),
            }
            groups_out.append(
                {
                    "name": f"Spot ({symbol}) KINGMAKER #{idx:02d} pnl/dd={metrics['pnl_over_dd']:.2f} "
                    f"pnl={metrics['pnl']:.1f} win={metrics['win_rate']*100:.1f}% tr={metrics['trades']}",
                    "filters": filters_payload,
                    "entries": [{"symbol": symbol, "metrics": metrics, "strategy": strategy}],
                    "_eval": {
                        "stability_min_pnl_dd": float(item.get("stability_min_pnl_dd") or 0.0),
                        "stability_min_pnl": float(item.get("stability_min_pnl") or 0.0),
                        "windows": item.get("windows") or [],
                    },
                    "_key": key,
                }
            )
        out_payload = {
            "name": "kingmaker_top",
            "generated_at": now,
            "source": str(milestones_path),
            "windows": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in windows],
            "groups": groups_out,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=False))
        print(f"\nWrote {out_path} (top={top_k}).")


if __name__ == "__main__":
    main()
