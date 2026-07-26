"""Frozen XSP opening-edge candidate and deterministic prefix economics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ..backtest.data import ContractMeta
from ..backtest.engine import _run_spot_backtest
from ..engines.directional_impulse import DirectionalImpulseAdmissionPolicy
from ..engines.market import xsp_rth_evaluation_slots, xsp_trading_date
from ..time_utils import ET_ZONE, NaiveTsModeInput, to_et, to_utc_naive
from .live_calibration import SELECTED_EQUITY_SCHEMA, calibration_fingerprint
from .spot_sweeps.support import _bundle_base, _mk_filters


XSP_OPENING_EDGE_VERSION = "xsp.opening-edge-directional.v1"
XSP_OPENING_EDGE_RUNTIME_REVISION = "close-time-parity-r1"
XSP_DIRECTIONAL_SHADOW_POLICY = {
    "authority": "preregistered_shadow_evidence_only",
    "unit": "$1_per_XSP_point",
    "capital_reference_usd": 1_000,
    "capital_reference_authority": "user_reported_design_reference_only",
    "max_drawdown_points": 25.0,
    "max_session_loss_points": 5.0,
    "minimum_week_closed_trades": 2,
    "maximum_top_five_win_share": 0.5,
    "slot_tolerance_seconds": 90.0,
    "order_authority": "none",
}
XSP_OPENING_EDGE_POLICY = DirectionalImpulseAdmissionPolicy()
XSP_OPENING_EDGE_ADMISSION = XSP_OPENING_EDGE_POLICY.as_payload()
XSP_OPENING_EDGE_CONTRACT = {
    "schema": XSP_OPENING_EDGE_VERSION,
    "runtime_revision": XSP_OPENING_EDGE_RUNTIME_REVISION,
    "bar_size": "5 mins",
    "timestamp_semantics": "bar_close",
    "use_rth": True,
    "entry_signal": "directional_impulse",
    "admission": XSP_OPENING_EDGE_ADMISSION,
    "entry_fill": "next_open",
    "exit": "inverse_source_flip_or_eod",
    "flip_fill": "next_open",
    "flip_hold_bars": 12,
    "max_entries_per_session": 5,
    "quantity": 1,
    "unit": "$1_per_XSP_point",
    "round_trip_friction_points": 0.10,
    "initial_stop": None,
    "trail": None,
    "profit_target": None,
    "fizzle": None,
    "order_authority": "none",
}
_META = ContractMeta(symbol="XSP", exchange="CBOE", multiplier=1.0, min_tick=0.01)


def xsp_opening_edge_bundle(
    *,
    start: date,
    end: date,
    flip_hold_bars: int = 12,
    primary_ema: str | None = None,
):
    """Build the one canonical fixed-unit candidate configuration."""

    base = _bundle_base(
        symbol="XSP",
        start=start,
        end=end,
        bar_size="5 mins",
        use_rth=True,
        cache_dir=Path("db"),
        offline=True,
        filters=_mk_filters(cooldown_bars=0),
        starting_cash=1_000.0,
        entry_signal="directional_impulse",
        spot_profit_target_pct=None,
        spot_stop_loss_pct=None,
        flip_exit_min_hold_bars=int(flip_hold_bars),
        spot_close_eod=True,
    )
    return replace(
        base,
        strategy=replace(
            base.strategy,
            regime_mode="ema" if primary_ema else "off",
            regime_ema_preset=primary_ema,
            regime2_mode="off",
            directional_impulse_admission=XSP_OPENING_EDGE_ADMISSION,
            max_entries_per_day=5,
            spot_entry_fill_mode="next_open",
            spot_flip_exit_fill_mode="next_open",
            spot_controlled_flip=True,
            exit_on_signal_flip=True,
            flip_exit_min_hold_bars=int(flip_hold_bars),
            flip_exit_only_if_profit=False,
            spot_intrabar_exits=True,
            spot_spread=0.0,
            spot_commission_per_share=0.05,
            spot_commission_min=0.0,
            spot_slippage_per_share=0.0,
            spot_mark_to_market="close",
            spot_drawdown_mode="intrabar",
            spot_sizing_mode="fixed",
            spot_min_qty=1,
            spot_max_qty=1,
            spot_excursion_exit=None,
        ),
    )


_FINGERPRINT_TEMPLATE = xsp_opening_edge_bundle(
    start=date(2000, 1, 1),
    end=date(2000, 1, 1),
)
XSP_OPENING_EDGE_CONFIG_FINGERPRINT = calibration_fingerprint(
    {
        "contract": XSP_OPENING_EDGE_CONTRACT,
        "strategy": asdict(_FINGERPRINT_TEMPLATE.strategy),
        "synthetic": asdict(_FINGERPRINT_TEMPLATE.synthetic),
        "backtest": {
            "bar_size": _FINGERPRINT_TEMPLATE.backtest.bar_size,
            "use_rth": _FINGERPRINT_TEMPLATE.backtest.use_rth,
            "starting_cash": _FINGERPRINT_TEMPLATE.backtest.starting_cash,
        },
    }
)


def xsp_opening_edge_run_start(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at: datetime,
) -> datetime:
    """Recover one restart-stable prospective start, or open today's RTH run."""

    starts = set()
    for row in records:
        evidence = row.get("evidence")
        equity = (
            evidence.get("candidate_equity")
            if isinstance(evidence, Mapping)
            else None
        )
        if (
            row.get("kind") == "checkpoint"
            and row.get("strategy_version") == XSP_OPENING_EDGE_VERSION
            and isinstance(equity, Mapping)
            and equity.get("run_started_at_utc")
        ):
            starts.add(str(equity["run_started_at_utc"]))
    if len(starts) > 1:
        raise ValueError("XSP opening-edge candidate run start drift")
    if starts:
        return datetime.fromisoformat(starts.pop().replace("Z", "+00:00"))
    trading_day = xsp_trading_date(observed_at)
    if trading_day is None:
        raise ValueError("XSP opening-edge candidate requires a trading date")
    return datetime.combine(trading_day, time(9, 30), tzinfo=ET_ZONE).astimezone(
        timezone.utc
    )


def xsp_opening_edge_candidate_equity(
    bars: Sequence[object],
    *,
    run_started_at: datetime,
    observed_at: datetime,
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> dict[str, object]:
    """Replay one causal prefix through the normal engine; never submit an order."""

    observed_utc = observed_at.astimezone(timezone.utc)
    started_utc = run_started_at.astimezone(timezone.utc)
    normalized = tuple(
        replace(
            bar,
            ts=to_utc_naive(
                getattr(bar, "ts"),
                naive_ts_mode=naive_ts_mode,
            ),
        )
        for bar in bars
        if to_utc_naive(
            getattr(bar, "ts"),
            naive_ts_mode=naive_ts_mode,
        ).replace(tzinfo=timezone.utc)
        <= observed_utc
    )
    if not normalized:
        raise ValueError("XSP opening-edge candidate requires bars")
    session_day = to_et(observed_utc).date()
    slots = xsp_rth_evaluation_slots(session_day)
    session_close = slots[-1] - timedelta(minutes=2) if slots else None
    final_session_complete = bool(
        session_close
        and to_et(normalized[-1].ts, naive_ts_mode="utc") >= session_close
    )
    cfg = xsp_opening_edge_bundle(
        start=to_et(started_utc).date(),
        end=session_day,
    )
    result = _run_spot_backtest(
        cfg,
        normalized,
        _META,
        final_session_complete=final_session_complete,
    )
    started_naive = started_utc.replace(tzinfo=None)
    trades = [
        trade
        for trade in result.trades
        if trade.entry_time >= started_naive
    ]
    rows = []
    for trade in trades:
        trace = trade.decision_trace if isinstance(trade.decision_trace, Mapping) else {}
        guard_inputs = trace.get("entry_guard_inputs")
        signal_at = (
            guard_inputs.get("signal_bar_ts")
            if isinstance(guard_inputs, Mapping)
            else None
        )
        try:
            signal_at_utc = (
                to_utc_naive(
                    datetime.fromisoformat(str(signal_at)),
                    naive_ts_mode="utc",
                )
                .replace(tzinfo=timezone.utc)
                .isoformat()
                if signal_at
                else None
            )
        except ValueError:
            signal_at_utc = None
        rows.append(
            {
                "decision_at_utc": signal_at_utc,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": (
                    trade.exit_time.isoformat()
                    if trade.exit_time is not None
                    else None
                ),
                "direction": "up" if int(trade.qty) > 0 else "down",
                "entry_price": float(trade.entry_price),
                "exit_price": (
                    float(trade.exit_price) if trade.exit_price is not None else None
                ),
                "exit_reason": str(trade.exit_reason or ""),
                "net_points": float(trade.pnl(1.0)),
            }
        )
    marked = [row for row in rows if row["exit_reason"] == "end"]
    closed = [row for row in rows if row["exit_reason"] != "end"]
    friction = float(XSP_OPENING_EDGE_CONTRACT["round_trip_friction_points"])

    def economics(selected: Sequence[Mapping[str, object]]) -> tuple[float, float, float]:
        net = sum(float(row["net_points"]) for row in selected)
        cost = friction * len(selected)
        return net + cost, cost, net

    cumulative_gross, cumulative_cost, cumulative_net = economics(rows)
    session_rows = [
        row
        for row in rows
        if to_et(
            datetime.fromisoformat(str(row["entry_time"])),
            naive_ts_mode="utc",
        ).date()
        == session_day
    ]
    session_gross, session_cost, session_net = economics(session_rows)
    realized_net = sum(float(row["net_points"]) for row in closed)
    open_mark = sum(float(row["net_points"]) for row in marked)
    drawdown = float(result.summary.max_drawdown)
    gross_wins = sorted(
        (
            float(row["net_points"]) + friction
            for row in closed
            if float(row["net_points"]) + friction > 0.0
        ),
        reverse=True,
    )
    breaches = []
    if drawdown > float(XSP_DIRECTIONAL_SHADOW_POLICY["max_drawdown_points"]):
        breaches.append("drawdown_limit_breached")
    if session_net < -float(
        XSP_DIRECTIONAL_SHADOW_POLICY["max_session_loss_points"]
    ):
        breaches.append("session_loss_limit_breached")
    run_id = calibration_fingerprint(
        {
            "strategy_version": XSP_OPENING_EDGE_VERSION,
            "config_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
            "run_started_at_utc": started_utc.isoformat(),
        }
    )
    return {
        "schema": "xsp.candidate-equity.v1",
        "authority": "prospective_counterfactual_only",
        "run_id": run_id,
        "run_started_at_utc": started_utc.isoformat(),
        "config_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        "capital_sleeve": "xsp-directional-unit",
        "unit": "$1_per_XSP_point",
        "cumulative_gross_points": cumulative_gross,
        "cumulative_cost_points": cumulative_cost,
        "cumulative_net_points": cumulative_net,
        "cumulative_realized_net_points": realized_net,
        "open_mark_points": open_mark,
        "session_gross_points": session_gross,
        "session_cost_points": session_cost,
        "session_net_points": session_net,
        "closed_trades": len(closed),
        "gross_wins_points": sum(gross_wins),
        "top_five_gross_wins_points": sum(gross_wins[:5]),
        "maximum_drawdown_points": drawdown,
        "reconciled": True,
        "attribution_complete": True,
        "safety_breaches": breaches,
        "latest_position": marked[-1] if marked else None,
        "latest_trade": rows[-1] if rows else None,
        "trade_ledger_fingerprint": calibration_fingerprint(rows),
        "order_authority": "none",
    }


def xsp_opening_edge_selected_equity(
    candidate_equity: Mapping[str, object],
    *,
    run_id: str,
    capital_sleeve: str,
) -> dict[str, object]:
    """Bind the unchanged counterfactual ledger to one preselected shadow run."""

    if (
        candidate_equity.get("schema") != "xsp.candidate-equity.v1"
        or candidate_equity.get("config_fingerprint")
        != XSP_OPENING_EDGE_CONFIG_FINGERPRINT
        or candidate_equity.get("order_authority") != "none"
        or not str(run_id).strip()
        or not str(capital_sleeve).strip()
    ):
        raise ValueError("invalid Opening Edge selected-equity source")
    return {
        **candidate_equity,
        "schema": SELECTED_EQUITY_SCHEMA,
        "authority": "selected_shadow_evidence_only",
        "run_id": str(run_id),
        "capital_sleeve": str(capital_sleeve),
        "order_authority": "none",
    }
