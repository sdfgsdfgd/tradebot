from __future__ import annotations

from datetime import datetime

from tradebot.spot.lifecycle import SpotLifecycleDecision
from tradebot.spot.scenario import lifecycle_trace_row, why_not_exit_resize_report


def test_why_not_exit_resize_report_filters_hold_rows() -> None:
    ts = datetime(2026, 2, 14, 15, 30, 0)
    hold_exit = SpotLifecycleDecision(
        intent="hold",
        reason="holding",
        gate="HOLDING",
        trace={"stage": "open", "path": "hold"},
    )
    resize_fire = SpotLifecycleDecision(
        intent="resize",
        reason="target_delta",
        gate="TRIGGER_RESIZE",
        trace={"stage": "open", "path": "resize"},
    )
    hold_resize = SpotLifecycleDecision(
        intent="hold",
        reason="resize_cooldown",
        gate="BLOCKED_RESIZE_COOLDOWN",
        blocked=True,
        trace={"stage": "open", "path": "resize"},
    )

    rows = [
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_exit",
            decision=hold_exit,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_resize",
            decision=resize_fire,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_resize",
            decision=hold_resize,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
    ]

    report = why_not_exit_resize_report(rows)
    assert len(report) == 2
    assert [str(row["target"]) for row in report] == ["exit", "resize"]
    assert str(report[1]["reason"]) == "resize_cooldown"
    assert bool(report[1]["blocked"]) is True
