"""Live-vs-replay drift detector (spot signal parity).

This tool is meant to run shortly after a live session (nightly / post-close):
- It reads the bot journal CSV (tradebot/ui/out/bot_journal_*.csv).
- For each ORDER_STAGED row, it extracts the live signal snapshot fields embedded in `order_journal`.
- It replays SpotSignalEvaluator on IBKR historical bars (same bar_size/use_rth + the recorded duration/source),
  and diffs the key discrete decision surfaces:
    - regime-router host/climate/dir (+ dwell)
    - regime4_state / hard_dir
    - entry_dir (including entry_dir=None "flat" blocks)

Typical use:
  python -m tradebot.backtest.tools.spot_live_drift_detector --symbol TQQQ --date 2026-04-09
  python -m tradebot.backtest.tools.spot_live_drift_detector --symbol TQQQ --start 2026-04-08 --end 2026-04-09
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ib_insync import Stock

from ...client import IBKRClient
from ...config import load_config
from ...engine import resolve_spot_regime2_spec, resolve_spot_regime_spec
from ...spot_engine import SpotSignalEvaluator, SpotSignalSnapshot


def _parse_date(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def _parse_dt(raw: object) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _latest_journal_path(out_dir: Path) -> Path | None:
    try:
        paths = sorted(out_dir.glob("bot_journal_*.csv"))
    except Exception:
        return None
    if not paths:
        return None
    # Filename already encodes date/time; sort is sufficient.
    return paths[-1]


def _load_journal_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _maybe_json(raw: object) -> dict[str, object]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _eval_last_snapshot(*, evaluator: SpotSignalEvaluator, bars: list) -> SpotSignalSnapshot | None:
    last_snap: SpotSignalSnapshot | None = None
    trade_day_fn = getattr(evaluator, "_trade_date", None)
    for idx, bar in enumerate(bars):
        next_bar = bars[idx + 1] if idx + 1 < len(bars) else None
        if next_bar is None:
            is_last_bar = True
        elif callable(trade_day_fn):
            is_last_bar = bool(trade_day_fn(next_bar.ts) != trade_day_fn(bar.ts))
        else:
            is_last_bar = next_bar.ts.date() != bar.ts.date()
        evaluator.update_exec_bar(bar, is_last_bar=bool(is_last_bar))
        snap = evaluator.update_signal_bar(bar)
        if snap is not None:
            last_snap = snap
    return last_snap


def _probe_from_snap(snap: SpotSignalSnapshot) -> dict[str, object]:
    return {
        "bar_ts": snap.bar_ts.isoformat(),
        "entry_dir": str(snap.entry_dir) if snap.entry_dir in ("up", "down") else None,
        "entry_branch": str(snap.entry_branch) if snap.entry_branch in ("a", "b") else None,
        "shock_dir": str(snap.shock_dir) if snap.shock_dir in ("up", "down") else None,
        "shock_atr_pct": float(snap.shock_atr_pct) if snap.shock_atr_pct is not None else None,
        "regime4_state": str(snap.regime4_state) if snap.regime4_state else None,
        "hard_dir": (
            str(snap.regime2_bear_hard_dir) if snap.regime2_bear_hard_dir in ("up", "down") else None
        ),
        "router_ready": bool(getattr(snap, "regime_router_ready", False)),
        "router_climate": str(getattr(snap, "regime_router_climate", "") or "") or None,
        "router_host": str(getattr(snap, "regime_router_host", "") or "") or None,
        "router_entry_dir": (
            str(getattr(snap, "regime_router_entry_dir", None))
            if getattr(snap, "regime_router_entry_dir", None) in ("up", "down")
            else None
        ),
        "router_host_managed": bool(getattr(snap, "regime_router_host_managed", False)),
        "router_bull_ok": bool(getattr(snap, "regime_router_bull_sovereign_ok", False)),
        "router_dwell_days": int(getattr(snap, "regime_router_dwell_days", 0) or 0),
        "router_crash_ret": float(getattr(snap, "regime_router_crash_ret"))
        if getattr(snap, "regime_router_crash_ret", None) is not None
        else None,
        "router_crash_maxdd": float(getattr(snap, "regime_router_crash_maxdd"))
        if getattr(snap, "regime_router_crash_maxdd", None) is not None
        else None,
    }


def _probe_from_live_order_journal(order_journal: dict[str, object]) -> dict[str, object]:
    return {
        "bar_ts": str(order_journal.get("bar_ts") or ""),
        "entry_dir": order_journal.get("entry_dir") if order_journal.get("entry_dir") in ("up", "down") else None,
        "entry_branch": order_journal.get("entry_branch") if order_journal.get("entry_branch") in ("a", "b") else None,
        "shock_dir": order_journal.get("shock_dir") if order_journal.get("shock_dir") in ("up", "down") else None,
        "shock_atr_pct": order_journal.get("shock_atr_pct"),
        "regime4_state": order_journal.get("regime4_state") if order_journal.get("regime4_state") else None,
        "hard_dir": order_journal.get("hard_dir") if order_journal.get("hard_dir") in ("up", "down") else None,
        "router_ready": bool(order_journal.get("regime_router_ready", False)),
        "router_climate": str(order_journal.get("regime_router_climate", "") or "") or None,
        "router_host": str(order_journal.get("regime_router_host", "") or "") or None,
        "router_entry_dir": (
            str(order_journal.get("regime_router_entry_dir", None))
            if order_journal.get("regime_router_entry_dir", None) in ("up", "down")
            else None
        ),
        "router_host_managed": bool(order_journal.get("regime_router_host_managed", False)),
        "router_bull_ok": bool(order_journal.get("regime_router_bull_sovereign_ok", False)),
        "router_dwell_days": int(order_journal.get("regime_router_dwell_days", 0) or 0),
        "router_crash_ret": order_journal.get("regime_router_crash_ret"),
        "router_crash_maxdd": order_journal.get("regime_router_crash_maxdd"),
    }


def _diff_keys(*, live: dict[str, object], replay: dict[str, object]) -> list[str]:
    keys = [
        "entry_dir",
        "entry_branch",
        "regime4_state",
        "hard_dir",
        "router_ready",
        "router_climate",
        "router_host",
        "router_entry_dir",
        "router_host_managed",
        "router_bull_ok",
        "router_dwell_days",
    ]
    diffs: list[str] = []
    for k in keys:
        if live.get(k) != replay.get(k):
            diffs.append(k)
    return diffs


def _pick_duration(health: dict[str, object] | None, *, fallback: str) -> str:
    if isinstance(health, dict):
        raw = health.get("duration_str")
        if isinstance(raw, str) and raw.strip():
            return str(raw).strip()
    return str(fallback)


def _pick_source(health: dict[str, object] | None, *, fallback: str) -> str:
    if isinstance(health, dict):
        raw = health.get("source")
        if isinstance(raw, str) and raw.strip():
            return str(raw).strip().upper()
    return str(fallback).strip().upper() or "TRADES"


@dataclass(frozen=True)
class ReplaySpec:
    bar_size: str
    use_rth: bool
    duration_str: str
    what_to_show: str


async def _fetch_bars_trimmed(
    *,
    client: IBKRClient,
    contract,
    spec: ReplaySpec,
    end_ts: datetime,
    cache: dict[tuple, list],
) -> list:
    key = (int(getattr(contract, "conId", 0) or 0), spec.bar_size, bool(spec.use_rth), spec.what_to_show, spec.duration_str)
    if key in cache:
        bars = cache[key]
    else:
        bars = await client.historical_bars_ohlcv(
            contract,
            duration_str=str(spec.duration_str),
            bar_size=str(spec.bar_size),
            use_rth=bool(spec.use_rth),
            what_to_show=str(spec.what_to_show),
            cache_ttl_sec=0.0,
        )
        bars.sort(key=lambda b: b.ts)
        cache[key] = list(bars)
    # Trim to the live snapshot bar_ts to avoid including post-snapshot bars.
    return [b for b in bars if isinstance(getattr(b, "ts", None), datetime) and b.ts <= end_ts]


async def main_async(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spot live-vs-replay drift detector (bot journal parity audit).")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--journal", default=None, help="Path to bot_journal_*.csv (defaults to newest in tradebot/ui/out).")
    ap.add_argument("--out-dir", default="tradebot/ui/out", help="Bot journal directory (default: tradebot/ui/out).")
    ap.add_argument("--event", default="ORDER_STAGED", help="Journal event to audit (default: ORDER_STAGED).")
    ap.add_argument("--date", default=None, help="Single ISO date (ET) to audit, e.g. 2026-04-09.")
    ap.add_argument("--start", default=None, help="ISO date start (ET).")
    ap.add_argument("--end", default=None, help="ISO date end (ET).")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to audit (0=all).")
    ap.add_argument("--bars-fallback-duration", default="2 W", help="Fallback duration if not recorded in journal.")
    args = ap.parse_args(argv)

    symbol = str(args.symbol).strip().upper()
    event = str(args.event).strip().upper()

    journal_path = Path(args.journal) if args.journal else _latest_journal_path(Path(args.out_dir))
    if journal_path is None or not journal_path.exists():
        raise SystemExit(f"No journal found (out_dir={args.out_dir!r})")

    if args.date is not None:
        start_d = _parse_date(args.date)
        end_d = start_d
    else:
        start_d = _parse_date(args.start) if args.start else date.min
        end_d = _parse_date(args.end) if args.end else date.max

    rows = _load_journal_rows(journal_path)
    candidates: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("event", "") or "").strip().upper() != event:
            continue
        if str(row.get("symbol", "") or "").strip().upper() != symbol:
            continue
        data = _maybe_json(row.get("data_json"))
        order_journal = data.get("order_journal") if isinstance(data.get("order_journal"), dict) else None
        if not isinstance(order_journal, dict):
            continue
        bar_ts = _parse_dt(order_journal.get("bar_ts"))
        if bar_ts is None:
            continue
        if not (start_d <= bar_ts.date() <= end_d):
            continue
        candidates.append(
            {
                "row": row,
                "data": data,
                "order_journal": order_journal,
                "bar_ts": bar_ts,
            }
        )
    if args.limit and int(args.limit) > 0:
        candidates = candidates[: int(args.limit)]

    print(f"journal={journal_path}")
    print(f"symbol={symbol} event={event} range={start_d.isoformat()}→{end_d.isoformat()} rows={len(candidates)}")
    if not candidates:
        return 0

    cfg = load_config()
    client = IBKRClient(cfg)
    await client.connect_proxy()
    await client.connect()

    contract = Stock(symbol, "SMART", "USD")
    qualified = await client.qualify_proxy_contracts(contract)
    if qualified:
        contract = qualified[0]

    cache: dict[tuple, list] = {}
    drift_count = 0
    ok_count = 0
    t_all = time.time()

    for i, item in enumerate(candidates, start=1):
        row = item["row"]
        data = item["data"]
        order_journal = item["order_journal"]
        bar_ts: datetime = item["bar_ts"]

        strategy = data.get("strategy") if isinstance(data.get("strategy"), dict) else None
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else None
        if not isinstance(strategy, dict):
            print(f"[{i:02d}/{len(candidates):02d}] SKIP missing strategy (bar_ts={bar_ts.isoformat()})")
            continue

        bar_size = str(strategy.get("signal_bar_size") or "5 mins").strip() or "5 mins"
        use_rth = bool(strategy.get("signal_use_rth", True))

        # Resolve regime / regime2 bar sizes the same way the UI does.
        regime_mode, _regime_preset, regime_bar_size, use_mtf_regime = resolve_spot_regime_spec(
            bar_size=bar_size,
            regime_mode_raw=strategy.get("regime_mode"),
            regime_ema_preset_raw=strategy.get("regime_ema_preset"),
            regime_bar_size_raw=strategy.get("regime_bar_size"),
        )
        regime2_mode, _regime2_preset, regime2_bar_size, use_mtf_regime2 = resolve_spot_regime2_spec(
            bar_size=bar_size,
            regime2_mode_raw=strategy.get("regime2_mode"),
            regime2_ema_preset_raw=strategy.get("regime2_ema_preset"),
            regime2_bar_size_raw=strategy.get("regime2_bar_size"),
        )
        _ = (regime_mode, regime2_mode)  # keep local parity probes explicit

        signal_health = order_journal.get("signal_bar_health") if isinstance(order_journal.get("signal_bar_health"), dict) else None
        regime_health = order_journal.get("regime_bar_health") if isinstance(order_journal.get("regime_bar_health"), dict) else None
        regime2_health = order_journal.get("regime2_bar_health") if isinstance(order_journal.get("regime2_bar_health"), dict) else None

        sig_spec = ReplaySpec(
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            duration_str=_pick_duration(signal_health, fallback=str(args.bars_fallback_duration)),
            what_to_show=_pick_source(signal_health, fallback="TRADES"),
        )
        reg_spec = ReplaySpec(
            bar_size=str(regime_bar_size),
            use_rth=bool(use_rth),
            duration_str=_pick_duration(regime_health, fallback=str(sig_spec.duration_str)),
            what_to_show=_pick_source(regime_health, fallback=str(sig_spec.what_to_show)),
        )
        reg2_spec = ReplaySpec(
            bar_size=str(regime2_bar_size),
            use_rth=bool(use_rth),
            duration_str=_pick_duration(regime2_health, fallback=str(sig_spec.duration_str)),
            what_to_show=_pick_source(regime2_health, fallback=str(sig_spec.what_to_show)),
        )

        t0 = time.time()
        sig_bars = await _fetch_bars_trimmed(client=client, contract=contract, spec=sig_spec, end_ts=bar_ts, cache=cache)
        regime_bars = (
            await _fetch_bars_trimmed(client=client, contract=contract, spec=reg_spec, end_ts=bar_ts, cache=cache)
            if bool(use_mtf_regime)
            else None
        )
        regime2_bars = (
            await _fetch_bars_trimmed(client=client, contract=contract, spec=reg2_spec, end_ts=bar_ts, cache=cache)
            if bool(use_mtf_regime2)
            else None
        )

        evaluator = SpotSignalEvaluator(
            strategy=strategy,
            filters=filters,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            naive_ts_mode="et",
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
        )
        replay_snap = _eval_last_snapshot(evaluator=evaluator, bars=sig_bars)
        dt = time.time() - t0

        live_probe = _probe_from_live_order_journal(order_journal)
        if replay_snap is None:
            drift_count += 1
            print(f"[{i:02d}/{len(candidates):02d}] DRIFT bar_ts={bar_ts.isoformat()} replay=None ({dt:.1f}s)")
            continue

        replay_probe = _probe_from_snap(replay_snap)
        diffs = _diff_keys(live=live_probe, replay=replay_probe)
        if diffs:
            drift_count += 1
            print(f"[{i:02d}/{len(candidates):02d}] DRIFT bar_ts={bar_ts.isoformat()} diffs={diffs} ({dt:.1f}s)")
            for key in diffs:
                print(f"  {key}: live={live_probe.get(key)!r} replay={replay_probe.get(key)!r}")
        else:
            ok_count += 1
            print(f"[{i:02d}/{len(candidates):02d}] OK    bar_ts={bar_ts.isoformat()} ({dt:.1f}s)")

    total = time.time() - t_all
    print(f"ok={ok_count} drift={drift_count} total={len(candidates)} ({total:.1f}s)")
    try:
        await client.disconnect()
    except Exception:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    return int(asyncio.run(main_async(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

