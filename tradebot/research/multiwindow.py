"""CLI and result contract for spot multi-window research."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..backtest.sweep_fingerprint import _strategy_fingerprint
from ..backtest.sweep_parallel import _collect_parallel_payload_records
from ..backtest.sweeps import utc_now_iso_z, write_json
from ..spot.codec import effective_filters_payload


@dataclass(frozen=True)
class MultiwindowReport:
    symbol: str
    bar_size: str
    use_rth: bool
    offline: bool
    windows: tuple[tuple[date, date], ...]
    min_trades: int
    min_win: float
    min_trades_per_year: float | None
    milestones_path: Path
    write_top: int
    out_path: Path
    cache_path: Path
    track: str | None = None
    incumbent_source: str | None = None
    incumbent_windows: tuple[dict, ...] = ()


def score_key(item: dict) -> tuple:
    stability = item.get("stability") if isinstance(item.get("stability"), dict) else {}
    primary = item.get("primary") if isinstance(item.get("primary"), dict) else {}
    return (
        float(stability.get("min_roi_over_dd") or stability.get("min_pnl_over_dd") or float("-inf")),
        float(stability.get("min_roi") or stability.get("min_pnl") or float("-inf")),
        float(primary.get("roi_over_dd_pct") or primary.get("pnl_over_dd") or float("-inf")),
        float(primary.get("roi") or primary.get("pnl") or float("-inf")),
        float(primary.get("win_rate") or 0.0),
        int(primary.get("trades") or 0),
    )


def promotion_receipt(
    candidate_windows: list[dict],
    incumbent_windows: tuple[dict, ...],
) -> dict | None:
    """Compare exact windows without pretending every HF/LF lane has one policy."""
    incumbent = {
        (str(row.get("start") or ""), str(row.get("end") or "")): row
        for row in incumbent_windows
        if isinstance(row, dict) and row.get("start") and row.get("end")
    }
    if not incumbent:
        return None
    candidate = {
        (str(row.get("start") or ""), str(row.get("end") or "")): row
        for row in candidate_windows
        if isinstance(row, dict) and row.get("start") and row.get("end")
    }

    def number(row: dict, *keys: str) -> float:
        for key in keys:
            raw = row.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return 0.0

    windows: list[dict] = []
    for key, baseline in incumbent.items():
        current = candidate.get(key)
        if current is None:
            continue
        candidate_ratio = number(current, "roi_over_dd_pct", "pnl_over_dd")
        incumbent_ratio = number(baseline, "roi_over_dd_pct", "pnl_over_dd")
        candidate_pnl = number(current, "pnl")
        incumbent_pnl = number(baseline, "pnl")
        candidate_trades = int(number(current, "trades"))
        incumbent_trades = int(number(baseline, "trades"))
        windows.append(
            {
                "start": key[0],
                "end": key[1],
                "ratio_delta": candidate_ratio - incumbent_ratio,
                "pnl_delta": candidate_pnl - incumbent_pnl,
                "trade_delta": candidate_trades - incumbent_trades,
                "candidate_ratio": candidate_ratio,
                "incumbent_ratio": incumbent_ratio,
            }
        )

    complete = len(windows) == len(incumbent)
    ratio_deltas = [float(row["ratio_delta"]) for row in windows]
    pnl_deltas = [float(row["pnl_delta"]) for row in windows]
    trade_deltas = [int(row["trade_delta"]) for row in windows]
    candidate_floor = min((float(row["candidate_ratio"]) for row in windows), default=0.0)
    incumbent_floor = min((float(row["incumbent_ratio"]) for row in windows), default=0.0)
    return {
        "complete": complete,
        "matched_windows": len(windows),
        "incumbent_windows": len(incumbent),
        "positive_all": complete
        and all(number(candidate[key], "pnl") > 0.0 for key in incumbent),
        "ratio_dominates_all": complete
        and all(delta >= 0.0 for delta in ratio_deltas)
        and any(delta > 0.0 for delta in ratio_deltas),
        "pnl_dominates_all": complete
        and all(delta >= 0.0 for delta in pnl_deltas)
        and any(delta > 0.0 for delta in pnl_deltas),
        "activity_preserved_all": complete and all(delta >= 0 for delta in trade_deltas),
        "floor_delta": candidate_floor - incumbent_floor if complete else None,
        "windows": windows,
    }


def strategy_key(strategy: dict, *, filters: dict | None) -> str:
    return _strategy_fingerprint(strategy, filters=filters)


def multiwindow_cache_key(
    *,
    engine_version: str,
    data_revision: str,
    strategy: dict,
    filters: dict | None,
    windows: tuple[tuple[date, date], ...],
    min_trades: int,
    min_trades_per_year: float | None,
    min_win: float,
    require_close_eod: bool,
    require_positive_pnl: bool,
    offline: bool,
) -> str:
    """Bind a derived multiwindow result to policy, strategy, and market-data revision."""
    payload = {
        "version": str(engine_version),
        "data_revision": str(data_revision),
        "strategy_key": strategy_key(strategy, filters=filters),
        "windows": tuple((start.isoformat(), end.isoformat()) for start, end in windows),
        "min_trades": int(min_trades),
        "min_trades_per_year": (
            float(min_trades_per_year) if min_trades_per_year is not None else None
        ),
        "min_win": float(min_win),
        "require_close_eod": bool(require_close_eod),
        "require_positive_pnl": bool(require_positive_pnl),
        "offline": bool(offline),
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def candidate_shortlist(
    payload: dict,
    *,
    symbol: str,
    bar_size: str,
    use_rth: bool,
    limit: int,
    track: str = "auto",
) -> list[dict]:
    """Build a deterministic, diverse seed set for the stability gate."""
    symbol_key = str(symbol).strip().upper()
    bar_key = str(bar_size).strip().lower()
    track_key = str(track or "auto").strip().upper()
    candidates: list[dict] = []
    groups = payload.get("groups") if isinstance(payload, dict) else None
    for group in groups if isinstance(groups, list) else ():
        if not isinstance(group, dict):
            continue
        group_track = str(group.get("_track") or payload.get("track") or "").strip().upper()
        if track_key != "AUTO" and group_track and group_track != track_key:
            continue
        group_filters = group.get("filters") if isinstance(group.get("filters"), dict) else None
        entries = group.get("entries")
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict):
                continue
            strategy = entry.get("strategy")
            metrics = entry.get("metrics")
            if not isinstance(strategy, dict) or not isinstance(metrics, dict):
                continue
            entry_symbol = str(entry.get("symbol") or strategy.get("symbol") or "").strip().upper()
            if entry_symbol != symbol_key:
                continue
            if str(strategy.get("instrument") or "spot").strip().lower() != "spot":
                continue
            if str(strategy.get("signal_bar_size") or "").strip().lower() != bar_key:
                continue
            if bool(strategy.get("signal_use_rth")) != bool(use_rth):
                continue
            candidates.append(
                {
                    "group_name": str(group.get("name") or ""),
                    "filters": effective_filters_payload(
                        group_filters=group_filters,
                        strategy=strategy,
                    ),
                    "strategy": strategy,
                    "metrics": metrics,
                    "track": group_track or None,
                }
            )

    def metric(candidate: dict, *names: str) -> float:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        for name in names:
            raw = metrics.get(name)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
        return float("-inf")

    objectives = (
        lambda candidate: metric(candidate, "roi_over_dd_pct", "pnl_over_dd"),
        lambda candidate: metric(candidate, "roi", "pnl"),
        lambda candidate: metric(candidate, "win_rate"),
        lambda candidate: metric(candidate, "trades"),
    )

    # Duplicate rows are historical noise; retain the strongest evidence once.
    best: dict[str, dict] = {}
    for candidate in candidates:
        key = strategy_key(candidate["strategy"], filters=candidate.get("filters"))
        previous = best.get(key)
        if previous is None or tuple(score(candidate) for score in objectives) > tuple(
            score(previous) for score in objectives
        ):
            best[key] = candidate

    keyed = sorted(best.items(), key=lambda item: item[0])
    if not keyed:
        return []
    rankings = [
        sorted(keyed, key=lambda item, score=score: (score(item[1]), item[0]), reverse=True)
        for score in objectives
    ]
    fused = {key: 0.0 for key, _candidate in keyed}
    for ranking in rankings:
        for rank, (key, _candidate) in enumerate(ranking, start=1):
            fused[key] += 1.0 / (60.0 + float(rank))

    # Preserve each objective leader, then fill with consistently strong candidates.
    ordered_keys: list[str] = []
    for ranking in rankings:
        leader = ranking[0][0]
        if leader not in ordered_keys:
            ordered_keys.append(leader)
    for key, _candidate in sorted(
        keyed,
        key=lambda item: (fused[item[0]], item[0]),
        reverse=True,
    ):
        if key not in ordered_keys:
            ordered_keys.append(key)
    return [best[key] for key in ordered_keys[: max(1, int(limit))]]


def parse_multiwindow_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="tradebot.backtest.multitimeframe")
    ap.add_argument("--milestones", required=True, help="Input spot milestones JSON to evaluate.")
    ap.add_argument("--symbol", default="TQQQ", help="Symbol to filter (default: TQQQ).")
    ap.add_argument("--bar-size", default="1 hour", help="Signal bar size filter (default: 1 hour).")
    ap.add_argument("--use-rth", action="store_true", help="Filter to RTH-only strategies.")
    ap.add_argument(
        "--track",
        default="auto",
        choices=("auto", "hf", "lf"),
        help=(
            "HF/LF research lineage and current-crown comparison windows; "
            "auto preserves existing artifact metadata when available."
        ),
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use cached bars at evaluation time. "
            "With --cache-policy=auto, preflight may hydrate missing caches before run."
        ),
    )
    ap.add_argument(
        "--cache-policy",
        default="auto",
        choices=("auto", "strict"),
        help=(
            "Offline cache preflight policy. "
            "auto = hydrate via cache manager (resample-from-cache or fetch) before evaluating; "
            "strict = fail on any missing cache."
        ),
    )
    ap.add_argument("--cache-dir", default="db", help="Bars cache dir (default: db).")
    ap.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto). Requires --offline for >1.")
    ap.add_argument("--top", type=int, default=200, help="How many candidates to evaluate (after sorting).")
    ap.add_argument("--min-trades", type=int, default=200, help="Min trades per window.")
    ap.add_argument(
        "--min-trades-per-year",
        type=float,
        default=None,
        help=(
            "Min trades per year per window (e.g. 500 => 1y>=500, 2y>=1000, 10y>=5000). "
            "Enforced as ceil(window_years * min_trades_per_year)."
        ),
    )
    ap.add_argument("--min-win", type=float, default=0.0, help="Min win rate per window (0..1).")
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
        "--write-top",
        type=int,
        default=0,
        help="Write a small milestones JSON of the top K stability winners (0 disables).",
    )
    ap.add_argument(
        "--out",
        default="backtests/out/multitimeframe_top.json",
        help="Output file for --write-top (default: backtests/out/multitimeframe_top.json).",
    )
    ap.add_argument("--multitimeframe-worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-workers", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--multitimeframe-out", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def collect_multiwindow_rows(*, payloads: dict[int, dict]) -> tuple[int, list[dict]]:
    out_rows: list[dict] = []

    def _decode_row(rec: dict) -> dict | None:
        return dict(rec) if isinstance(rec, dict) else None

    def _row_key(row: dict) -> str:
        strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
        filters_payload = row.get("filters") if isinstance(row.get("filters"), dict) else None
        return strategy_key(strategy, filters=filters_payload)

    tested_total = _collect_parallel_payload_records(
        payloads=payloads,
        records_key="rows",
        tested_key="tested",
        decode_record=_decode_row,
        on_record=lambda row: out_rows.append(dict(row)),
        dedupe_key=_row_key,
    )
    return int(tested_total), out_rows


def emit_multiwindow_results(
    *,
    report: MultiwindowReport,
    out_rows: list[dict],
    tested_total: int | None = None,
    workers: int | None = None,
    cache_enabled: bool,
    cache_hits: int,
    cache_writes: int,
) -> None:
    out_rows = sorted(out_rows, key=score_key, reverse=True)
    print("")
    print(f"Multiwindow results: {len(out_rows)} candidates passed filters.")
    print(
        f"- symbol={report.symbol} track={report.track or 'unclassified'} "
        f"bar={report.bar_size} rth={report.use_rth} offline={report.offline}"
    )
    if report.incumbent_source:
        print(f"- incumbent={report.incumbent_source}")
    print(f"- windows={', '.join([f'{a.isoformat()}→{b.isoformat()}' for a,b in report.windows])}")
    extra = (
        f" min_trades_per_year={float(report.min_trades_per_year):g}"
        if report.min_trades_per_year is not None
        else ""
    )
    print(f"- min_trades={report.min_trades} min_win={report.min_win:0.2f}{extra}")
    if tested_total is not None and workers is not None:
        print(f"- workers={int(workers)} tested_total={int(tested_total)}")
    if bool(cache_enabled):
        print(f"- eval_cache={report.cache_path} hits={int(cache_hits)} writes={int(cache_writes)}", flush=True)
    print("")

    show = min(20, len(out_rows))
    for rank, item in enumerate(out_rows[:show], start=1):
        st = item["strategy"]
        primary = item["primary"]
        stability = item["stability"]
        promotion = promotion_receipt(item.get("windows") or [], report.incumbent_windows)
        promotion_label = ""
        if promotion is not None:
            floor_delta = promotion.get("floor_delta")
            floor_text = f"{float(floor_delta):+.3f}" if floor_delta is not None else "partial"
            promotion_label = f" dethrone_floor={floor_text}"
        print(
            f"{rank:2d}. stability(min roi/dd)={stability.get('min_roi_over_dd', 0.0):.2f} "
            f"primary roi/dd={primary.get('roi_over_dd_pct', 0.0):.2f} "
            f"roi={primary.get('roi', 0.0)*100:.1f}% dd%={primary.get('dd_pct', 0.0)*100:.1f}% "
            f"pnl={primary.get('pnl', 0.0):.1f} "
            f"win={primary.get('win_rate', 0.0)*100:.1f}% tr={primary.get('trades', 0)} "
            f"ema={st.get('ema_preset')} {st.get('ema_entry_mode')} "
            f"regime={st.get('regime_mode')} rbar={st.get('regime_bar_size')}"
            f"{promotion_label}"
        )

    if report.write_top <= 0:
        return
    top_k = max(1, report.write_top)
    now = utc_now_iso_z()
    groups_out: list[dict] = []
    for idx, item in enumerate(out_rows[:top_k], start=1):
        strategy = dict(item["strategy"])
        strategy.setdefault("signal_bar_size", report.bar_size)
        strategy.setdefault("signal_use_rth", report.use_rth)
        filters_payload = item.get("filters")
        key = strategy_key(strategy, filters=filters_payload)
        primary = item["primary"]
        stability = item["stability"]
        promotion = promotion_receipt(item.get("windows") or [], report.incumbent_windows)
        metrics = {
            "pnl": float(primary.get("pnl") or 0.0),
            "roi": float(primary.get("roi") or 0.0),
            "win_rate": float(primary.get("win_rate") or 0.0),
            "trades": int(primary.get("trades") or 0),
            "max_drawdown": float(primary.get("dd") or 0.0),
            "max_drawdown_pct": float(primary.get("dd_pct") or 0.0),
            "pnl_over_dd": float(primary.get("pnl_over_dd") or 0.0),
            "roi_over_dd_pct": float(primary.get("roi_over_dd_pct") or 0.0),
        }
        group = {
            "name": f"Spot ({report.symbol}) KINGMAKER #{idx:02d} roi/dd={metrics['roi_over_dd_pct']:.2f} "
            f"roi={metrics['roi']*100:.1f}% dd%={metrics['max_drawdown_pct']*100:.1f}% "
            f"win={metrics['win_rate']*100:.1f}% tr={metrics['trades']} pnl={metrics['pnl']:.1f}",
            "filters": filters_payload,
            "entries": [{"symbol": report.symbol, "metrics": metrics, "strategy": strategy}],
            "_eval": {
                "stability": dict(stability),
                "windows": item.get("windows") or [],
            },
            "_key": key,
        }
        if report.track:
            group["_track"] = report.track
        if promotion is not None:
            group["_eval"]["promotion"] = promotion
        groups_out.append(group)
    out_payload = {
        "schema": "tradebot.research.multiwindow.v1",
        "name": "multitimeframe_top",
        "generated_at": now,
        "source": str(report.milestones_path),
        "track": report.track,
        "incumbent": report.incumbent_source,
        "bar_size": report.bar_size,
        "use_rth": report.use_rth,
        "windows": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in report.windows],
        "groups": groups_out,
    }
    write_json(report.out_path, out_payload, sort_keys=False)
    print(f"\nWrote {report.out_path} (top={top_k}).")
