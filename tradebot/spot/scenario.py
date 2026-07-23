"""Scenario tools for shared spot lifecycle diagnostics.

These helpers operate on the lifecycle decision kernel output and can be used
for both backtest and live diagnostics.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from copy import deepcopy
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


def _project_spot_trace_dimension(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = value
    else:
        as_payload = getattr(value, "as_payload", None)
        if not callable(as_payload):
            raise TypeError(
                "trace receipt dimensions must be mappings, typed payload objects, or None"
            )
        payload = as_payload()
    if not isinstance(payload, Mapping):
        raise TypeError("trace payload object's as_payload() must return a mapping")
    return deepcopy(dict(payload))


def project_spot_trace_receipt(
    *,
    sizing: object = None,
    intent: object = None,
    lifecycle: object = None,
    fill: object = None,
    accounting: object = None,
) -> dict[str, object]:
    return {
        "schema": "spot-trace-receipt-v1",
        "sizing": _project_spot_trace_dimension(sizing),
        "intent": _project_spot_trace_dimension(intent),
        "lifecycle": _project_spot_trace_dimension(lifecycle),
        "fill": _project_spot_trace_dimension(fill),
        "accounting": _project_spot_trace_dimension(accounting),
    }

def project_backtest_spot_trade_receipt(
    *,
    decision_trace: object,
    qty: object,
    entry_price: object,
    margin_required: object,
    exit_price: object = None,
    exit_reason: object = None,
    exit_time: object = None,
) -> dict[str, object]:
    """Project one final backtest trade receipt from its canonical trace state."""

    trace = dict(decision_trace) if isinstance(decision_trace, Mapping) else {}
    trace.pop("spot_trace_receipt", None)
    exit_time_payload = (
        exit_time.isoformat() if isinstance(exit_time, datetime) else exit_time
    )
    return project_spot_trace_receipt(
        sizing=trace,
        intent=trace.get("spot_intent"),
        lifecycle=trace.get("spot_lifecycle"),
        fill={
            "resizes": list(trace.get("resizes") or []),
            "exits": list(trace.get("exits") or []),
        },
        accounting={
            "qty": qty,
            "entry_price": entry_price,
            "margin_required": margin_required,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_time": exit_time_payload,
        },
    )

def project_live_spot_execution_receipt(
    *,
    journal: object,
    status: object,
    filled_qty: object,
    remaining_qty: object,
    executed_qty: object,
    basis_applied_filled_qty: object,
    entry_basis_qty: object,
    entry_basis_price: object,
    entry_basis_source: object,
) -> dict[str, object]:
    """Project current live fill and entry-basis state onto a spot journal."""

    payload = dict(journal) if isinstance(journal, Mapping) else {}
    return project_spot_trace_receipt(
        sizing=payload.get("spot_decision"),
        intent=payload.get("spot_intent"),
        lifecycle=payload.get("spot_lifecycle"),
        fill={
            "status": str(status or ""),
            "filled_qty": float(filled_qty or 0.0),
            "remaining_qty": float(remaining_qty or 0.0),
            "executed_qty": float(executed_qty or 0.0),
            "basis_applied_filled_qty": float(
                basis_applied_filled_qty or 0.0
            ),
        },
        accounting={
            "entry_basis_qty": (
                float(entry_basis_qty) if entry_basis_qty is not None else None
            ),
            "entry_basis_price": (
                float(entry_basis_price)
                if entry_basis_price is not None
                else None
            ),
            "entry_basis_source": (
                str(entry_basis_source)
                if entry_basis_source is not None
                else None
            ),
        },
    )


def project_live_spot_order_journal(
    *,
    snapshot: object,
    intent: str,
    direction: str | None,
    entry_branch: str,
    branch_size_mult: float | None,
    sizing: object,
    lifecycle: object,
    spot_intent: object,
    exit_mode: str,
    stop_loss_pct: float | None,
    stop_price: float | None,
    target_price: float | None,
    sizing_currency: str,
    net_liq: float | None,
    net_liq_currency: str | None,
    net_liq_fx_rate: float | None,
    buying_power: float | None,
    buying_power_currency: str | None,
    buying_power_fx_rate: float | None,
    chase_orders: bool,
) -> dict[str, object]:
    """Project the stable live-order diagnostic payload from shared spot decisions."""

    snap = snapshot
    signal = getattr(snap, "signal", None)
    receipt = project_spot_trace_receipt(
        sizing=sizing,
        intent=spot_intent,
        lifecycle=lifecycle,
    )
    sizing_payload = receipt["sizing"] or {}
    intent_payload = receipt["intent"] or {}
    lifecycle_payload = receipt["lifecycle"] or {}
    journal: dict[str, object] = {
        "intent": intent,
        "direction": direction,
        "bar_ts": snap.bar_ts.isoformat(),
        "close": float(snap.close),
        "signal": {
            "state": getattr(signal, "state", None),
            "entry_dir": getattr(signal, "entry_dir", None),
            "regime_dir": getattr(signal, "regime_dir", None),
            "ema_ready": bool(getattr(signal, "ema_ready", False)),
        },
        "entry_dir": getattr(snap, "entry_dir", None),
        "regime4_state": str(getattr(snap, "regime4_state", "") or "") or None,
        "hard_dir": (
            str(getattr(snap, "regime2_bear_hard_dir"))
            if getattr(snap, "regime2_bear_hard_dir", None) in ("up", "down")
            else None
        ),
        "entry_branch": entry_branch if entry_branch in ("a", "b") else None,
        "branch_size_mult": float(branch_size_mult)
        if branch_size_mult is not None
        else None,
        "spot_decision": sizing_payload,
        "spot_lifecycle": lifecycle_payload,
        "spot_intent": intent_payload,
        "size_funnel": {
            "signed_qty_final": int(getattr(sizing, "signed_qty_final", 0)),
            "signed_qty_after_branch": int(
                getattr(
                    sizing,
                    "signed_qty_after_branch",
                    getattr(sizing, "signed_qty_final", 0),
                )
            ),
            "resize_target_qty": int(getattr(spot_intent, "target_qty", 0)),
            "intent_qty": int(getattr(spot_intent, "order_qty", 0)),
        },
        "ratsv": {
            "side_rank": float(getattr(snap, "ratsv_side_rank"))
            if getattr(snap, "ratsv_side_rank", None) is not None
            else None,
            "tr_ratio": float(getattr(snap, "ratsv_tr_ratio"))
            if getattr(snap, "ratsv_tr_ratio", None) is not None
            else None,
            "fast_slope_pct": float(getattr(snap, "ratsv_fast_slope_pct"))
            if getattr(snap, "ratsv_fast_slope_pct", None) is not None
            else None,
            "fast_slope_med_pct": float(getattr(snap, "ratsv_fast_slope_med_pct"))
            if getattr(snap, "ratsv_fast_slope_med_pct", None) is not None
            else None,
            "fast_slope_vel_pct": float(getattr(snap, "ratsv_fast_slope_vel_pct"))
            if getattr(snap, "ratsv_fast_slope_vel_pct", None) is not None
            else None,
            "slow_slope_med_pct": float(getattr(snap, "ratsv_slow_slope_med_pct"))
            if getattr(snap, "ratsv_slow_slope_med_pct", None) is not None
            else None,
            "slow_slope_vel_pct": float(getattr(snap, "ratsv_slow_slope_vel_pct"))
            if getattr(snap, "ratsv_slow_slope_vel_pct", None) is not None
            else None,
            "slope_vel_consistency": float(
                getattr(snap, "ratsv_slope_vel_consistency")
            )
            if getattr(snap, "ratsv_slope_vel_consistency", None) is not None
            else None,
            "cross_age_bars": int(getattr(snap, "ratsv_cross_age_bars"))
            if getattr(snap, "ratsv_cross_age_bars", None) is not None
            else None,
        },
        "bars_in_day": int(snap.bars_in_day),
        "rv": float(snap.rv) if snap.rv is not None else None,
        "volume": float(snap.volume) if snap.volume is not None else None,
        "shock": bool(snap.shock) if snap.shock is not None else None,
        "shock_dir": snap.shock_dir,
        "shock_detector": str(getattr(snap, "shock_detector", "") or ""),
        "shock_direction_source_effective": str(
            getattr(snap, "shock_direction_source_effective", "") or ""
        ),
        "shock_scale_detector": str(
            getattr(snap, "shock_scale_detector", "") or ""
        ),
        "shock_dir_ret_sum_pct": float(getattr(snap, "shock_dir_ret_sum_pct"))
        if getattr(snap, "shock_dir_ret_sum_pct", None) is not None
        else None,
        "shock_atr_pct": float(snap.shock_atr_pct)
        if snap.shock_atr_pct is not None
        else None,
        "shock_drawdown_pct": float(getattr(snap, "shock_drawdown_pct"))
        if getattr(snap, "shock_drawdown_pct", None) is not None
        else None,
        "shock_drawdown_on_pct": float(getattr(snap, "shock_drawdown_on_pct"))
        if getattr(snap, "shock_drawdown_on_pct", None) is not None
        else None,
        "shock_drawdown_off_pct": float(getattr(snap, "shock_drawdown_off_pct"))
        if getattr(snap, "shock_drawdown_off_pct", None) is not None
        else None,
        "shock_drawdown_dist_on_pct": float(
            getattr(snap, "shock_drawdown_dist_on_pct")
        )
        if getattr(snap, "shock_drawdown_dist_on_pct", None) is not None
        else None,
        "shock_drawdown_dist_on_vel_pp": float(
            getattr(snap, "shock_drawdown_dist_on_vel_pp")
        )
        if getattr(snap, "shock_drawdown_dist_on_vel_pp", None) is not None
        else None,
        "shock_drawdown_dist_on_accel_pp": float(
            getattr(snap, "shock_drawdown_dist_on_accel_pp")
        )
        if getattr(snap, "shock_drawdown_dist_on_accel_pp", None) is not None
        else None,
        "shock_prearm_down_streak_bars": int(
            getattr(snap, "shock_prearm_down_streak_bars")
        )
        if getattr(snap, "shock_prearm_down_streak_bars", None) is not None
        else None,
        "shock_drawdown_dist_off_pct": float(
            getattr(snap, "shock_drawdown_dist_off_pct")
        )
        if getattr(snap, "shock_drawdown_dist_off_pct", None) is not None
        else None,
        "shock_atr_vel_pct": float(getattr(snap, "shock_atr_vel_pct"))
        if getattr(snap, "shock_atr_vel_pct", None) is not None
        else None,
        "shock_atr_accel_pct": float(getattr(snap, "shock_atr_accel_pct"))
        if getattr(snap, "shock_atr_accel_pct", None) is not None
        else None,
        "shock_peak_close": float(getattr(snap, "shock_peak_close"))
        if getattr(snap, "shock_peak_close", None) is not None
        else None,
        "shock_dir_down_streak_bars": int(
            getattr(snap, "shock_dir_down_streak_bars")
        )
        if getattr(snap, "shock_dir_down_streak_bars", None) is not None
        else None,
        "shock_dir_up_streak_bars": int(getattr(snap, "shock_dir_up_streak_bars"))
        if getattr(snap, "shock_dir_up_streak_bars", None) is not None
        else None,
        "riskoff": bool(snap.risk.riskoff) if snap.risk is not None else None,
        "riskpanic": bool(snap.risk.riskpanic) if snap.risk is not None else None,
        "atr": float(snap.atr) if snap.atr is not None else None,
        "or_high": float(snap.or_high) if snap.or_high is not None else None,
        "or_low": float(snap.or_low) if snap.or_low is not None else None,
        "or_ready": bool(snap.or_ready),
        "exit_mode": exit_mode,
        "stop_loss_pct": float(stop_loss_pct)
        if stop_loss_pct is not None
        else None,
        "stop_price": float(stop_price) if stop_price is not None else None,
        "target_price": float(target_price) if target_price is not None else None,
        "sizing_currency": sizing_currency,
        "net_liq": float(net_liq) if net_liq is not None else None,
        "net_liq_currency": str(net_liq_currency)
        if net_liq_currency is not None
        else None,
        "net_liq_fx_rate": float(net_liq_fx_rate)
        if net_liq_fx_rate is not None
        else None,
        "buying_power": float(buying_power) if buying_power is not None else None,
        "buying_power_currency": str(buying_power_currency)
        if buying_power_currency is not None
        else None,
        "buying_power_fx_rate": float(buying_power_fx_rate)
        if buying_power_fx_rate is not None
        else None,
        "exec_policy": "LADDER",
        "exec_mode": "OPTIMISTIC",
        "chase_orders": bool(chase_orders),
    }
    for key, attr in (
        ("signal_bar_health", "bar_health"),
        ("regime_bar_health", "regime_bar_health"),
        ("regime2_bar_health", "regime2_bar_health"),
    ):
        raw = getattr(snap, attr, None)
        payload = dict(raw) if isinstance(raw, dict) else None
        if payload is not None:
            for field, value in list(payload.items()):
                if isinstance(value, datetime):
                    payload[str(field)] = value.isoformat()
        journal[key] = payload
    journal["spot_trace_receipt"] = project_live_spot_execution_receipt(
        journal=journal,
        status="STAGED",
        filled_qty=0.0,
        remaining_qty=0.0,
        executed_qty=0.0,
        basis_applied_filled_qty=0.0,
        entry_basis_qty=None,
        entry_basis_price=None,
        entry_basis_source=None,
    )
    return journal
