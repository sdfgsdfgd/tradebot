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


def test_backtest_trace_receipt_projection_matches_shared_schema() -> None:
    import json
    from dataclasses import MISSING, fields

    from tradebot.spot.lifecycle import SpotLifecycleDecision
    from tradebot.spot.policy_contract import SpotDecisionTrace, SpotIntentDecision
    from tradebot.spot.scenario import project_spot_trace_receipt

    def _fallback(field) -> object:
        annotation = str(field.type)
        name = str(field.name)
        if "bool" in annotation:
            return False
        if "int" in annotation:
            return 1
        if "float" in annotation:
            return 1.0
        if "str" in annotation:
            return name
        if "dict" in annotation or "Mapping" in annotation:
            return {}
        if "list" in annotation:
            return []
        if "tuple" in annotation:
            return ()
        return None

    def _typed(cls, **overrides):
        kwargs: dict[str, object] = {}
        for field in fields(cls):
            if field.name in overrides:
                kwargs[field.name] = overrides[field.name]
                continue
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            kwargs[field.name] = _fallback(field)
        return cls(**kwargs)

    sizing = _typed(
        SpotDecisionTrace,
        action="BUY",
        sizing_mode="fixed",
        lot=1,
        quantity_mult=1,
        base_signed_qty=2,
        entry_price=101.25,
        equity_ref=2200.0,
        cash_ref=2200.0,
        riskoff=False,
        riskpanic=False,
        riskpop=False,
        shock=False,
        riskoff_mode="directional",
        regime2_ready=True,
        riskoff_long_factor=1.0,
        riskoff_short_factor=1.0,
        riskpanic_long_factor=1.0,
        riskpanic_short_factor=1.0,
        riskpop_long_factor=1.0,
        riskpop_short_factor=1.0,
        graph_overlay_trace={"policy": "neutral", "nested": {"risk_mult": 1.0}},
        signed_qty_final=2,
        signed_qty_after_branch=2,
    )
    intent = _typed(
        SpotIntentDecision,
        intent="resize",
        reason="target_delta",
        gate="TRIGGER_RESIZE",
        action="BUY",
        order_qty=1,
        delta_qty=1,
        target_qty=2,
        resize_kind="scale_in",
    )
    lifecycle = SpotLifecycleDecision(
        intent="resize",
        reason="target_delta",
        gate="TRIGGER_RESIZE",
        direction="up",
        fill_mode="close",
        spot_intent=intent,
        spot_decision=sizing.as_payload(),
        trace={
            "stage": "open",
            "path": "resize",
            "graph_entry": {
                "gate": "TRIGGER_ENTRY",
                "trace": {"policy": "slope_tr_guard", "threshold": 1.3},
            },
        },
    )
    fill = {
        "status": "filled",
        "filled_qty": 2.0,
        "remaining_qty": 0.0,
        "executed_qty": 2.0,
        "details": {"avg_fill_price": 101.25},
    }
    accounting = {
        "qty": 2,
        "entry_price": 101.25,
        "margin_required": 50.0,
        "exit_price": None,
        "exit_reason": None,
        "entry_basis": {
            "qty": 2.0,
            "price": 101.25,
            "source": "broker_average_cost",
        },
    }

    typed = project_spot_trace_receipt(
        sizing=sizing,
        intent=intent,
        lifecycle=lifecycle,
        fill=fill,
        accounting=accounting,
    )
    mapped = project_spot_trace_receipt(
        sizing=sizing.as_payload(),
        intent=intent.as_payload(),
        lifecycle=lifecycle.as_payload(),
        fill=fill,
        accounting=accounting,
    )

    assert list(typed) == [
        "schema",
        "sizing",
        "intent",
        "lifecycle",
        "fill",
        "accounting",
    ]
    assert typed["schema"] == "spot-trace-receipt-v1"
    assert typed == mapped
    assert typed["lifecycle"]["trace"]["graph_entry"]["trace"]["policy"] == "slope_tr_guard"
    assert json.loads(json.dumps(typed, sort_keys=True)) == typed

    typed["fill"]["details"]["avg_fill_price"] = 999.0
    typed["accounting"]["entry_basis"]["price"] = 999.0
    assert fill["details"]["avg_fill_price"] == 101.25
    assert accounting["entry_basis"]["price"] == 101.25
