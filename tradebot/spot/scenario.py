"""Scenario tools for shared spot lifecycle diagnostics.

These helpers operate on the lifecycle decision kernel output and can be used
for both backtest and live diagnostics.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from .lifecycle import SpotLifecycleDecision


def lifecycle_trace_row(
    *,
    bar_ts: datetime | None,
    stage: str,
    decision: SpotLifecycleDecision,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = decision.as_payload()
    spot_intent = payload.get("spot_intent") if isinstance(payload.get("spot_intent"), Mapping) else None
    trace = payload.get("trace") if isinstance(payload.get("trace"), Mapping) else None
    row: dict[str, object] = {
        "bar_ts": bar_ts.isoformat() if isinstance(bar_ts, datetime) else None,
        "stage": str(stage),
        "intent": str(payload.get("intent") or ""),
        "reason": str(payload.get("reason") or ""),
        "gate": str(payload.get("gate") or ""),
        "direction": payload.get("direction"),
        "fill_mode": str(payload.get("fill_mode") or "close"),
        "blocked": bool(payload.get("blocked", False)),
        "spot_intent": str(spot_intent.get("intent") or "") if isinstance(spot_intent, Mapping) else "",
        "spot_intent_reason": str(spot_intent.get("reason") or "") if isinstance(spot_intent, Mapping) else "",
        "spot_intent_resize_kind": (
            str(spot_intent.get("resize_kind") or "") if isinstance(spot_intent, Mapping) else ""
        ),
        "trace_stage": str(trace.get("stage") or "") if isinstance(trace, Mapping) else "",
        "trace_path": str(trace.get("path") or "") if isinstance(trace, Mapping) else "",
    }
    if isinstance(context, Mapping):
        for key, value in context.items():
            row[str(key)] = value
    return row


def why_not_exit_resize_report(
    lifecycle_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for row in lifecycle_rows:
        stage = str(row.get("stage") or "")
        if stage not in ("open_exit", "open_resize"):
            continue
        intent = str(row.get("intent") or "")
        if intent != "hold":
            continue
        target = "exit" if stage == "open_exit" else "resize"
        report.append(
            {
                "bar_ts": row.get("bar_ts"),
                "target": target,
                "gate": row.get("gate"),
                "reason": row.get("reason"),
                "blocked": bool(row.get("blocked", False)),
                "direction": row.get("direction"),
                "spot_intent": row.get("spot_intent"),
                "spot_intent_reason": row.get("spot_intent_reason"),
                "spot_intent_resize_kind": row.get("spot_intent_resize_kind"),
                "trace_path": row.get("trace_path"),
                "symbol": row.get("symbol"),
                "sig_idx": row.get("sig_idx"),
                "exec_idx": row.get("exec_idx"),
            }
        )
    report.sort(key=lambda r: (str(r.get("bar_ts") or ""), str(r.get("target") or "")))
    return report


def write_rows_csv(
    *,
    rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    keys: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for key in row.keys():
            k = str(key)
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
    if not keys:
        keys = ["bar_ts", "stage", "intent", "reason", "gate"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({k: row.get(k) for k in keys})
    return out
