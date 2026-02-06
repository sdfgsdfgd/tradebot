#!/usr/bin/env python3
"""Compare two live bot journal CSVs and report execution/gate deltas.

Usage:
  python3 tradebot/ui/tools/journal_ab_compare.py --a <pathA> --b <pathB>

If --a/--b are omitted, the script tries to pick the latest two files with
`spot_exec_feed_mode` values in INSTANCE_CREATED rows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class JournalSummary:
    path: Path
    mode: str
    strategy: str
    start_ts: datetime | None
    end_ts: datetime | None
    row_count: int
    signal_count: int
    trigger_entry_count: int
    trigger_exit_count: int
    trigger_exit_stop_count: int
    trigger_exit_profit_count: int
    gate_reason_counts: Counter
    failed_filter_counts: Counter


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _build_summary(path: Path, rows: list[dict[str, str]]) -> JournalSummary:
    mode = ""
    strategy = ""
    gate_reason_counts: Counter = Counter()
    failed_filter_counts: Counter = Counter()
    signal_count = 0
    trigger_entry_count = 0
    trigger_exit_count = 0
    trigger_exit_stop_count = 0
    trigger_exit_profit_count = 0

    for row in rows:
        event = str(row.get("event") or "")
        reason = str(row.get("reason") or "")
        payload_raw = row.get("data_json") or ""
        payload: dict[str, object]
        if payload_raw:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

        if event == "INSTANCE_CREATED" and isinstance(payload, dict):
            strategy_payload = payload.get("strategy")
            if isinstance(strategy_payload, dict):
                if not mode:
                    mode = str(strategy_payload.get("spot_exec_feed_mode") or "")
                if not strategy:
                    strategy = str(strategy_payload.get("name") or "")

        if event == "SIGNAL":
            signal_count += 1

        if event != "GATE":
            continue

        gate_reason_counts[reason] += 1
        if reason == "TRIGGER_ENTRY":
            trigger_entry_count += 1
        if reason == "TRIGGER_EXIT":
            trigger_exit_count += 1
            why = str(payload.get("reason") or "").lower()
            if "stop" in why:
                trigger_exit_stop_count += 1
            if "profit" in why:
                trigger_exit_profit_count += 1

        failed = payload.get("failed_filters")
        if isinstance(failed, list):
            for name in failed:
                failed_filter_counts[str(name)] += 1

    start_ts = _parse_ts(rows[0].get("ts_et") if rows else None)
    end_ts = _parse_ts(rows[-1].get("ts_et") if rows else None)

    return JournalSummary(
        path=path,
        mode=mode,
        strategy=strategy,
        start_ts=start_ts,
        end_ts=end_ts,
        row_count=len(rows),
        signal_count=signal_count,
        trigger_entry_count=trigger_entry_count,
        trigger_exit_count=trigger_exit_count,
        trigger_exit_stop_count=trigger_exit_stop_count,
        trigger_exit_profit_count=trigger_exit_profit_count,
        gate_reason_counts=gate_reason_counts,
        failed_filter_counts=failed_filter_counts,
    )


def _latest_by_mode(paths: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(paths):
        rows = _load_rows(path)
        mode = ""
        for row in rows:
            if row.get("event") != "INSTANCE_CREATED":
                continue
            try:
                payload = json.loads(row.get("data_json") or "{}")
            except json.JSONDecodeError:
                continue
            strategy = payload.get("strategy") if isinstance(payload, dict) else None
            if isinstance(strategy, dict):
                mode = str(strategy.get("spot_exec_feed_mode") or "")
                if mode:
                    break
        if mode:
            out[mode] = path
    return out


def _fmt_ts(ts: datetime | None) -> str:
    return ts.isoformat() if ts is not None else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two live journal CSV runs")
    parser.add_argument("--a", type=Path, help="Journal A path")
    parser.add_argument("--b", type=Path, help="Journal B path")
    parser.add_argument(
        "--glob",
        default="tradebot/ui/out/bot_journal_*_ET.csv",
        help="Glob used when --a/--b omitted",
    )
    args = parser.parse_args()

    if args.a and args.b:
        path_a = args.a
        path_b = args.b
    else:
        candidates = [Path(p) for p in glob.glob(args.glob)]
        if not candidates:
            print("No journal files found")
            return 1
        by_mode = _latest_by_mode(candidates)
        if "ticks_side" in by_mode and "poll" in by_mode:
            path_a = by_mode["ticks_side"]
            path_b = by_mode["poll"]
        else:
            print("Could not auto-pick both modes (need ticks_side and poll).")
            print("Found modes:", ", ".join(sorted(by_mode.keys())) or "<none>")
            return 2

    rows_a = _load_rows(path_a)
    rows_b = _load_rows(path_b)
    sum_a = _build_summary(path_a, rows_a)
    sum_b = _build_summary(path_b, rows_b)
    if (
        sum_a.start_ts is not None
        and sum_a.end_ts is not None
        and sum_b.start_ts is not None
        and sum_b.end_ts is not None
    ):
        overlap_start = max(sum_a.start_ts, sum_b.start_ts)
        overlap_end = min(sum_a.end_ts, sum_b.end_ts)
        if overlap_start >= overlap_end:
            print("WARNING: no overlapping time window between A and B.")
            print("WARNING: deltas below are not a true same-window A/B comparison.")
            print()

    print("A", sum_a.path)
    print(" mode", sum_a.mode)
    print(" strategy", sum_a.strategy)
    print(" window", _fmt_ts(sum_a.start_ts), "->", _fmt_ts(sum_a.end_ts))
    print(" entries", sum_a.trigger_entry_count, "exits", sum_a.trigger_exit_count)
    print(" stop_exits", sum_a.trigger_exit_stop_count, "profit_exits", sum_a.trigger_exit_profit_count)
    print(" failed_filters", dict(sum_a.failed_filter_counts))
    print(" gate_reasons", dict(sum_a.gate_reason_counts))
    print()

    print("B", sum_b.path)
    print(" mode", sum_b.mode)
    print(" strategy", sum_b.strategy)
    print(" window", _fmt_ts(sum_b.start_ts), "->", _fmt_ts(sum_b.end_ts))
    print(" entries", sum_b.trigger_entry_count, "exits", sum_b.trigger_exit_count)
    print(" stop_exits", sum_b.trigger_exit_stop_count, "profit_exits", sum_b.trigger_exit_profit_count)
    print(" failed_filters", dict(sum_b.failed_filter_counts))
    print(" gate_reasons", dict(sum_b.gate_reason_counts))
    print()

    print("DELTAS (B - A)")
    print(" entries", sum_b.trigger_entry_count - sum_a.trigger_entry_count)
    print(" exits", sum_b.trigger_exit_count - sum_a.trigger_exit_count)
    print(" stop_exits", sum_b.trigger_exit_stop_count - sum_a.trigger_exit_stop_count)
    print(" profit_exits", sum_b.trigger_exit_profit_count - sum_a.trigger_exit_profit_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
