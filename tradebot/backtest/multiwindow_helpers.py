from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
import time as pytime

from .cache_ops_lib import ensure_cached_window_with_policy
from .cli_utils import expected_cache_path
from .data import IBKRHistoricalData, cache_covers_window
from .spot_context import spot_bar_requirements_from_strategy
from ..series import bars_list


def load_bars(
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
        series = data.load_cached_bar_series(
            symbol=symbol,
            exchange=exchange,
            start=start_dt,
            end=end_dt,
            bar_size=bar_size,
            use_rth=use_rth,
            cache_dir=cache_dir,
        )
        return bars_list(series)
    series = data.load_or_fetch_bar_series(
        symbol=symbol,
        exchange=exchange,
        start=start_dt,
        end=end_dt,
        bar_size=bar_size,
        use_rth=use_rth,
        cache_dir=cache_dir,
    )
    return bars_list(series)


def die_empty_bars(
    *,
    kind: str,
    cache_dir: Path,
    symbol: str,
    exchange: str | None,
    start_dt: datetime,
    end_dt: datetime,
    bar_size: str,
    use_rth: bool,
    offline: bool,
) -> None:
    tag = "rth" if use_rth else "full24"
    expected = expected_cache_path(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start_dt=start_dt,
        end_dt=end_dt,
        bar_size=str(bar_size),
        use_rth=use_rth,
    )
    cache_ok, covering, missing_ranges = cache_covers_window(
        cache_dir=cache_dir,
        symbol=str(symbol),
        start=start_dt,
        end=end_dt,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
    )
    print("")
    print(f"[ERROR] No bars returned ({kind}):")
    print(f"- symbol={symbol} exchange={exchange or 'SMART'} bar={bar_size} {tag} offline={offline}")
    print(f"- window={start_dt.date().isoformat()}→{end_dt.date().isoformat()}")
    if expected.exists():
        print(f"- expected_cache={expected} (exists)")
    else:
        print(f"- expected_cache={expected} (missing)")
    if covering is not None and covering != expected:
        print(f"- covering_cache={covering}")
    if cache_ok and covering is None:
        print("- cache_coverage=overlap-full (stitchable)")
    if missing_ranges:
        missing_fmt: list[str] = []
        for s, e in missing_ranges[:5]:
            if s == e:
                missing_fmt.append(s.isoformat())
            else:
                missing_fmt.append(f"{s.isoformat()}..{e.isoformat()}")
        extra = "" if len(missing_ranges) <= 5 else f" (+{len(missing_ranges)-5} more)"
        print(f"- missing_ranges={', '.join(missing_fmt)}{extra}")
    if offline:
        print("")
        print("Fix:")
        print("- Re-run once without --offline to fetch/populate the cache via IBKR.")
        print("- If the cache file exists but is empty/corrupt, delete it and re-fetch.")
    else:
        print("")
        print("Fix:")
        print("- Verify IB Gateway / TWS is connected and you have market data permissions for this symbol/timeframe.")
        print("- If IBKR returns empty due to pacing/subscription limits, retry or prefetch once then re-run with --offline.")
    raise SystemExit(2)


def preflight_offline_cache_or_die(
    *,
    symbol: str,
    candidates: list[dict],
    windows: list[tuple[date, date]],
    signal_bar_size: str,
    use_rth: bool,
    cache_dir: Path,
    cache_policy: str = "strict",
) -> None:
    missing: list[dict] = []
    checked: set[tuple[str, str, str, str, bool]] = set()
    checks_total = 0
    started_at = float(pytime.perf_counter())
    last_report = float(started_at)
    heartbeat_sec = 15.0

    def _maybe_report(*, force: bool = False) -> None:
        nonlocal last_report
        now = float(pytime.perf_counter())
        if not bool(force) and (now - float(last_report)) < float(max(5.0, heartbeat_sec)):
            return
        elapsed = max(0.0, now - float(started_at))
        print(
            "multitimeframe preflight "
            f"checks={int(checks_total)} unique={len(checked)} missing={len(missing)} "
            f"elapsed={elapsed:0.1f}s",
            flush=True,
        )
        last_report = float(now)

    print(
        "multitimeframe preflight start "
        f"candidates={len(candidates)} windows={len(windows)} cache_policy={str(cache_policy).strip().lower() or 'strict'}",
        flush=True,
    )

    data = IBKRHistoricalData()

    def _require_cached(
        *,
        req_symbol: str,
        req_exchange: str | None,
        start_dt: datetime,
        end_dt: datetime,
        bar_size: str,
        use_rth: bool,
    ) -> None:
        nonlocal checks_total
        checks_total += 1
        key = (
            str(req_symbol),
            start_dt.date().isoformat(),
            end_dt.date().isoformat(),
            str(bar_size),
            bool(use_rth),
        )
        if key in checked:
            return
        checked.add(key)

        cache_ok, expected, _resolved, missing_ranges, err = ensure_cached_window_with_policy(
            data=data,
            cache_dir=cache_dir,
            symbol=str(req_symbol),
            exchange=req_exchange,
            start=start_dt,
            end=end_dt,
            bar_size=str(bar_size),
            use_rth=bool(use_rth),
            cache_policy=str(cache_policy),
        )
        if cache_ok:
            return
        missing_fmt: list[str] = []
        for s, e in missing_ranges:
            if s == e:
                missing_fmt.append(s.isoformat())
            else:
                missing_fmt.append(f"{s.isoformat()}..{e.isoformat()}")
        missing.append(
            {
                "symbol": str(req_symbol),
                "bar_size": str(bar_size),
                "start": start_dt.date().isoformat(),
                "end": end_dt.date().isoformat(),
                "use_rth": bool(use_rth),
                "missing_ranges": missing_fmt,
                "expected": str(expected),
                "error": str(err or "").strip(),
            }
        )
        _maybe_report(force=False)

    try:
        for wstart, wend in windows:
            start_dt = datetime.combine(wstart, time(0, 0))
            end_dt = datetime.combine(wend, time(23, 59))
            for cand in candidates:
                strat = cand.get("strategy")
                if not isinstance(strat, dict):
                    strat = {}
                reqs = spot_bar_requirements_from_strategy(
                    strategy=strat,
                    default_symbol=str(symbol),
                    default_exchange=None,
                    default_signal_bar_size=str(signal_bar_size),
                    default_signal_use_rth=bool(use_rth),
                    include_signal=True,
                )
                for req in reqs:
                    req_start_dt = start_dt - timedelta(days=max(0, int(req.warmup_days)))
                    _require_cached(
                        req_symbol=str(req.symbol),
                        req_exchange=req.exchange,
                        start_dt=req_start_dt,
                        end_dt=end_dt,
                        bar_size=str(req.bar_size),
                        use_rth=bool(req.use_rth),
                    )
    finally:
        try:
            data.disconnect()
        except Exception:
            pass
    _maybe_report(force=True)

    if not missing:
        print(
            "multitimeframe preflight done "
            f"checks={int(checks_total)} unique={len(checked)} missing=0",
            flush=True,
        )
        return

    print("")
    print("[ERROR] --offline was requested, but required cached bars are missing:")
    for item in missing[:25]:
        tag = "rth" if item["use_rth"] else "full"
        missing_ranges = item.get("missing_ranges") or []
        missing_note = f" missing={';'.join(missing_ranges)}" if missing_ranges else ""
        print(
            f"- {item['symbol']} {item['bar_size']} {tag} {item['start']}→{item['end']} "
            f"(expected: {item['expected']}{missing_note})"
        )
        if item.get("error"):
            print(f"  detail: {item['error']}")
    if len(missing) > 25:
        print(f"- … plus {len(missing) - 25} more missing caches")
    print("")
    print("Fix:")
    print("- Re-run without --offline to fetch via IBKR (and populate db/ cache).")
    print("- Or prefetch the missing bars explicitly before running with --offline.")
    raise SystemExit(2)
