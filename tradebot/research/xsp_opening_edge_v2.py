"""Prospective, non-submitting ownership of the Opening Edge v2 crown."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ..backtest.data import ContractMeta
from ..backtest.engine import _run_spot_backtest
from ..backtest.models import BacktestResult, Bar, SpotTrade
from ..backtest.spot_codec import filters_from_payload, strategy_from_payload
from ..engines.market import (
    xsp_bar_trading_date,
    xsp_session_label_et,
    xsp_trading_date,
)
from ..spot.champions import discover_current_champions, load_champion_group, repo_root
from ..time_utils import ET_ZONE, NaiveTsModeInput, to_et
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .xsp_candidate import xsp_opening_edge_bundle
from .xsp_context import xsp_fundamental_context_at, xsp_trade_attribution
from .xsp_opening_edge_tapes import (
    normalize_xsp_v2_bars,
    split_xsp_v2_sessions,
    xsp_opening_edge_v2_gth_signal_bars,
)
from .xsp_opening_edge_minute_context import (
    XspRthOneMinuteContext,
    load_xsp_rth_one_minute_context,
)
from .xsp_opening_edge_run_start import (
    XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
    next_xsp_v2_run_start,
    xsp_opening_edge_v2_run_start,
)

__all__ = (
    "XSP_OPENING_EDGE_V2_TRANSPORT_VERSION",
    "next_xsp_v2_run_start",
    "xsp_opening_edge_v2_run_start",
)


XSP_OPENING_EDGE_V2_VERSION = "xsp.opening-edge-v2-balanced-24x5.v1"
XSP_OPENING_EDGE_V2_UNIT = "$1_per_XSP_point"
XSP_OPENING_EDGE_V2_CAPITAL = 1_000.0
XSP_OPENING_EDGE_V2_HISTORY_DURATION = "2 W"
XSP_OPENING_EDGE_V2_FRESHNESS_SECONDS = 600.0
XSP_OPENING_EDGE_V2_RISK = {"max_drawdown_points": 25.0, "max_session_loss_points": 5.0}
XSP_OPENING_EDGE_V2_COSTS = {
    "research": {
        "spread": 0.0,
        "commission_per_side": 0.05,
        "round_trip_points": 0.10,
        "authority": "frozen_historical_contract",
    },
    "broker": {
        "spread": 0.03,
        "commission_per_side": 1.01540245,
        "round_trip_points": 2.0608049,
        "authority": "ibkr_spy_one_share_what_if_2026-07-27",
    },
}
XSP_OPENING_EDGE_V2_EXECUTION_GATE = {
    "verdict": "HOLD",
    "eligible": False,
    "reason": "measured_spy_cost_exceeds_historical_edge",
    "audit_fingerprint": "e4a88d284824317ceee66d11f903efcbfa9eb186d50044e872a427f95a759010",
    "audit_sha256": "7a19334c1d2288a5d85bb7e454ebef9f440a54409de0e96d40a9affab263858b",
    "historical_transport": {
        "net_points": 178.48,
        "trades": 725,
        "profit_factor": 1.4572,
        "max_drawdown_points": 26.64,
    },
}
_SPY_META = ContractMeta(symbol="SPY", exchange="SMART", multiplier=1.0, min_tick=0.01)


@dataclass(frozen=True)
class XspOpeningEdgeV2Spec:
    artifact_path: Path
    artifact_sha256: str
    declaration_path: Path
    declaration_version: str
    strategy_key: str
    group: Mapping[str, object]
    config_fingerprint: str


def load_xsp_opening_edge_v2_spec(
    *,
    root: Path | None = None,
) -> XspOpeningEdgeV2Spec:
    """Load only the content-addressed current XSP LF crown."""

    resolved_root = (root or repo_root()).resolve()
    refs = discover_current_champions(
        root=resolved_root,
        symbols=("XSP",),
        tracks=("LF",),
    )
    ref = refs[0] if len(refs) == 1 and str(refs[0].version) == "2" else None
    artifact_path = (
        ref.artifact_path
        if ref is not None
        else resolved_root / "backtests/xsp/opening_edge_v2_balanced_24x5.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    strategy_key = (
        str(ref.strategy_key or "")
        if ref is not None
        else "xsp-opening-edge-v2-balanced-24x5"
    )
    group = (
        load_champion_group(ref)
        if ref is not None
        else next(
            (
                row
                for row in payload.get("groups", ())
                if isinstance(row, Mapping)
                and str(row.get("_key") or "") == strategy_key
            ),
            None,
        )
    )
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != XSP_OPENING_EDGE_V2_VERSION
        or not strategy_key
        or not isinstance(group, Mapping)
        or str(group.get("_key") or "") != strategy_key
        or payload.get("order_authority") != "none"
    ):
        raise ValueError("frozen XSP LF predecessor is not Opening Edge v2")
    identity = {
        "schema": XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
        "artifact_sha256": artifact_sha256,
        "declaration_version": "2",
        "strategy_key": strategy_key,
        "signal_clock": "XSP",
        "execution_symbol": "SPY",
        "cost_profiles": XSP_OPENING_EDGE_V2_COSTS,
        "execution_gate": XSP_OPENING_EDGE_V2_EXECUTION_GATE,
        "order_authority": "none",
    }
    return XspOpeningEdgeV2Spec(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        declaration_path=(ref.declaration_path if ref is not None else artifact_path),
        declaration_version="2",
        strategy_key=strategy_key,
        group=dict(group),
        config_fingerprint=calibration_fingerprint(identity),
    )


def _entry(spec: XspOpeningEdgeV2Spec, name: str) -> Mapping[str, object]:
    entries = spec.group.get("entries")
    if not isinstance(entries, Sequence):
        raise ValueError("Opening Edge v2 crown has no entries")
    matches = [
        row for row in entries if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("strategy"), Mapping):
        raise ValueError(f"Opening Edge v2 requires one {name!r} entry")
    return matches[0]


def xsp_opening_edge_v2_bundle(
    spec: XspOpeningEdgeV2Spec,
    *,
    lane: str,
    start: date,
    end: date,
    cost_profile: str,
    rth_signal_symbol: str = "SPY",
    rth_entry_name: str = "RTH Core",
):
    """Hydrate the frozen leaf while keeping execution and clock ownership explicit."""

    lane_clean = str(lane).strip().lower()
    profile = XSP_OPENING_EDGE_V2_COSTS.get(str(cost_profile).strip().lower())
    if lane_clean not in {"rth", "gth"} or not isinstance(profile, Mapping):
        raise ValueError("invalid Opening Edge v2 lane or cost profile")
    entry = _entry(
        spec,
        rth_entry_name if lane_clean == "rth" else "GTH Down Sleeve",
    )
    filters = filters_from_payload(spec.group.get("filters"))
    base = xsp_opening_edge_bundle(start=start, end=end)
    strategy_payload = asdict(base.strategy)
    strategy_payload.update(dict(entry["strategy"]))
    strategy = strategy_from_payload(strategy_payload, filters=filters)
    signal_symbol = (
        str(rth_signal_symbol).strip().upper() if lane_clean == "rth" else "XSP"
    )
    strategy = replace(
        strategy,
        symbol=signal_symbol,
        exchange="CBOE" if signal_symbol == "XSP" else "SMART",
        spot_spread=float(profile["spread"]),
        spot_commission_per_share=float(profile["commission_per_side"]),
        spot_commission_min=0.0,
        spot_slippage_per_share=0.0,
        spot_sizing_mode="fixed",
        spot_min_qty=1,
        spot_max_qty=1,
    )
    return replace(
        base,
        backtest=replace(
            base.backtest,
            use_rth=lane_clean == "rth",
            starting_cash=XSP_OPENING_EDGE_V2_CAPITAL,
        ),
        strategy=strategy,
    )


def _lane_result(
    spec: XspOpeningEdgeV2Spec,
    *,
    lane: str,
    cost_profile: str,
    signal_bars: Sequence[Bar],
    execution_bars: Sequence[Bar],
    run_trading_date: date,
    run_started_at: datetime,
    observed_at: datetime,
    rth_signal_symbol: str,
    rth_entry_name: str = "RTH Core",
    spot_state_owner: object | None = None,
) -> BacktestResult:
    if not signal_bars or not execution_bars:
        raise ValueError(f"Opening Edge v2 {lane} requires signal and execution bars")
    session = xsp_session_label_et(observed_at)
    lane_complete = (
        session != "GTH" if lane == "gth" else session not in {"RTH", "CURB"}
    )
    cfg = xsp_opening_edge_v2_bundle(
        spec,
        lane=lane,
        start=run_trading_date,
        end=xsp_trading_date(observed_at) or to_et(observed_at).date(),
        cost_profile=cost_profile,
        rth_signal_symbol=rth_signal_symbol,
        rth_entry_name=rth_entry_name,
    )
    return _run_spot_backtest(
        cfg,
        signal_bars,
        _SPY_META,
        exec_bars=execution_bars,
        final_session_complete=lane_complete,
        spot_state_owner=spot_state_owner,
        entry_not_before=run_started_at.astimezone(timezone.utc).replace(tzinfo=None),
    )


def _trade_row(
    trade: SpotTrade,
    *,
    lane: str,
    cost_profile: str,
) -> dict[str, object]:
    net = float(trade.pnl(1.0))
    cost = float(XSP_OPENING_EDGE_V2_COSTS[cost_profile]["round_trip_points"])
    trading_day = xsp_bar_trading_date(
        trade.entry_time,
        naive_ts_mode="utc",
    )
    return {
        "lane": lane,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": (
            trade.exit_time.isoformat() if trade.exit_time is not None else None
        ),
        "trading_date": (trading_day.isoformat() if trading_day is not None else None),
        "direction": "up" if int(trade.qty) > 0 else "down",
        "entry_price": float(trade.entry_price),
        "exit_price": (
            float(trade.exit_price) if trade.exit_price is not None else None
        ),
        "exit_reason": str(trade.exit_reason or ""),
        "gross_points": net + cost,
        "cost_points": cost,
        "net_points": net,
        "attribution": xsp_trade_attribution(trade),
    }


def _combined_drawdown(results: Sequence[BacktestResult]) -> float:
    events: dict[datetime, dict[int, float]] = {}
    for index, result in enumerate(results):
        for point in result.equity:
            events.setdefault(point.ts, {})[index] = (
                float(point.equity) - XSP_OPENING_EDGE_V2_CAPITAL
            )
    latest = [0.0] * len(results)
    peak = XSP_OPENING_EDGE_V2_CAPITAL
    drawdown = 0.0
    for ts in sorted(events):
        for index, value in events[ts].items():
            latest[index] = value
        equity = XSP_OPENING_EDGE_V2_CAPITAL + sum(latest)
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return float(drawdown)


def _equity(
    spec: XspOpeningEdgeV2Spec,
    *,
    cost_profile: str,
    rth: BacktestResult,
    gth: BacktestResult,
    run_started_at: datetime,
    observed_at: datetime,
    rth_signal_symbol: str,
    strategy_version: str = XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
    schema: str = "xsp.opening-edge-v2-candidate-equity.v1",
) -> dict[str, object]:
    rows = sorted(
        (
            *(
                _trade_row(row, lane="gth", cost_profile=cost_profile)
                for row in gth.trades
            ),
            *(
                _trade_row(row, lane="rth", cost_profile=cost_profile)
                for row in rth.trades
            ),
        ),
        key=lambda row: str(row["entry_time"]),
    )
    current_day = xsp_trading_date(observed_at)
    session_rows = [
        row
        for row in rows
        if current_day is not None
        and row.get("trading_date") == current_day.isoformat()
    ]
    closed = [row for row in rows if row["exit_reason"] != "end"]
    marked = [row for row in rows if row["exit_reason"] == "end"]
    gross_wins = sorted(
        (
            float(row["gross_points"])
            for row in closed
            if float(row["gross_points"]) > 0.0
        ),
        reverse=True,
    )
    maximum_drawdown = _combined_drawdown((gth, rth))
    cumulative_gross = sum(float(row["gross_points"]) for row in rows)
    cumulative_cost = sum(float(row["cost_points"]) for row in rows)
    cumulative_net = sum(float(row["net_points"]) for row in rows)
    session_gross = sum(float(row["gross_points"]) for row in session_rows)
    session_cost = sum(float(row["cost_points"]) for row in session_rows)
    session_net = sum(float(row["net_points"]) for row in session_rows)
    breaches = []
    if maximum_drawdown > float(XSP_OPENING_EDGE_V2_RISK["max_drawdown_points"]):
        breaches.append("drawdown_limit_breached")
    if session_net < -float(XSP_OPENING_EDGE_V2_RISK["max_session_loss_points"]):
        breaches.append("session_loss_limit_breached")
    identity = {
        "strategy_version": strategy_version,
        "crown_config_fingerprint": spec.config_fingerprint,
        "cost_profile": cost_profile,
        "rth_signal_symbol": str(rth_signal_symbol).upper(),
        "execution_symbol": "SPY",
        "run_started_at_utc": run_started_at.astimezone(timezone.utc).isoformat(),
    }
    return {
        "schema": schema,
        "authority": "prospective_counterfactual_only",
        "run_id": calibration_fingerprint(identity),
        "run_started_at_utc": identity["run_started_at_utc"],
        "config_fingerprint": calibration_fingerprint(identity),
        "capital_sleeve": "xsp-directional-unit",
        "unit": XSP_OPENING_EDGE_V2_UNIT,
        "cost_profile": dict(XSP_OPENING_EDGE_V2_COSTS[cost_profile]),
        "rth_signal_symbol": str(rth_signal_symbol).upper(),
        "gth_signal_symbol": "XSP",
        "signal_clock": "XSP",
        "execution_symbol": "SPY",
        "cumulative_gross_points": cumulative_gross,
        "cumulative_cost_points": cumulative_cost,
        "cumulative_net_points": cumulative_net,
        "cumulative_realized_net_points": sum(
            float(row["net_points"]) for row in closed
        ),
        "open_mark_points": sum(float(row["net_points"]) for row in marked),
        "session_gross_points": session_gross,
        "session_cost_points": session_cost,
        "session_net_points": session_net,
        "closed_trades": len(closed),
        "gross_wins_points": sum(gross_wins),
        "top_five_gross_wins_points": sum(gross_wins[:5]),
        "maximum_drawdown_points": maximum_drawdown,
        "latest_position": marked[-1] if marked else None,
        "latest_trade": closed[-1] if closed else None,
        "trade_ledger_fingerprint": calibration_fingerprint(rows),
        "reconciled": abs(cumulative_net - cumulative_gross + cumulative_cost) <= 1e-7,
        "attribution_complete": all(
            row["attribution"]["decision_trace_fingerprint"] for row in rows
        ),
        "safety_breaches": breaches,
        "observed_at_utc": observed_at.astimezone(timezone.utc).isoformat(),
        "order_authority": "none",
    }


def xsp_opening_edge_v2_equities(
    *,
    spec: XspOpeningEdgeV2Spec,
    spy_bars: Sequence[Bar],
    observed_at: datetime,
    run_started_at: datetime,
    xsp_rth_bars: Sequence[Bar] | None = None,
    xsp_daily_bars: Sequence[Bar] | None = None,
    naive_ts_mode: NaiveTsModeInput = "utc",
    rth_state_owner_factory: Callable[[], object] | None = None,
    rth_entry_name: str = "RTH Core",
    strategy_version: str = XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
    paired_schema: str = "xsp.opening-edge-v2-paired-equity.v1",
    candidate_schema: str = "xsp.opening-edge-v2-candidate-equity.v1",
    execution_gate: Mapping[str, object] = XSP_OPENING_EDGE_V2_EXECUTION_GATE,
) -> dict[str, object]:
    """Replay the same causal tape under research and measured broker costs."""

    normalized_spy = normalize_xsp_v2_bars(
        spy_bars,
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    spy_gth, spy_rth = split_xsp_v2_sessions(normalized_spy)
    normalized_xsp = (
        normalize_xsp_v2_bars(
            tuple(xsp_rth_bars),
            observed_at=observed_at,
            naive_ts_mode=naive_ts_mode,
        )
        if xsp_rth_bars
        else ()
    )
    gth_signal = xsp_opening_edge_v2_gth_signal_bars(
        normalized_spy,
        normalized_xsp,
    )
    if not normalized_xsp or not gth_signal:
        raise ValueError("Opening Edge v2 requires exact XSP anchors")
    rth_signal = normalized_xsp
    rth_signal_symbol = "XSP"
    run_trading_date = xsp_trading_date(run_started_at)
    if run_trading_date is None:
        raise ValueError("Opening Edge v2 run start must be inside an XSP session")
    profiles: dict[str, object] = {}
    signal_observations: dict[str, object] | None = None
    rth_state: dict[str, object] | None = None
    for cost_profile in ("research", "broker"):
        gth = _lane_result(
            spec,
            lane="gth",
            cost_profile=cost_profile,
            signal_bars=gth_signal,
            execution_bars=spy_gth,
            run_trading_date=run_trading_date,
            run_started_at=run_started_at,
            observed_at=observed_at,
            rth_signal_symbol="XSP",
            rth_entry_name=rth_entry_name,
        )
        owner = (
            rth_state_owner_factory() if rth_state_owner_factory is not None else None
        )
        rth = _lane_result(
            spec,
            lane="rth",
            cost_profile=cost_profile,
            signal_bars=rth_signal,
            execution_bars=spy_rth,
            run_trading_date=run_trading_date,
            run_started_at=run_started_at,
            observed_at=observed_at,
            rth_signal_symbol=rth_signal_symbol,
            rth_entry_name=rth_entry_name,
            spot_state_owner=owner,
        )
        observations = {
            "gth": gth.latest_signal_snapshot,
            "rth": rth.latest_signal_snapshot,
        }
        if signal_observations is not None and observations != signal_observations:
            raise ValueError(
                "Opening Edge v2 signal observations drifted by cost profile"
            )
        signal_observations = observations
        state_payload = (
            owner.state_payload()
            if owner is not None and callable(getattr(owner, "state_payload", None))
            else None
        )
        if rth_state is not None and state_payload != rth_state:
            raise ValueError("Opening Edge state owner drifted by cost profile")
        rth_state = state_payload
        profiles[cost_profile] = _equity(
            spec,
            cost_profile=cost_profile,
            rth=rth,
            gth=gth,
            run_started_at=run_started_at,
            observed_at=observed_at,
            rth_signal_symbol=rth_signal_symbol,
            strategy_version=strategy_version,
            schema=candidate_schema,
        )
    return {
        "schema": paired_schema,
        "crown_artifact_sha256": spec.artifact_sha256,
        "crown_config_fingerprint": spec.config_fingerprint,
        "rth_signal_source": rth_signal_symbol,
        "gth_signal_source": "prior_xsp_close_anchored_spy_returns",
        "gth_signal_bars": len(gth_signal),
        "gth_signal_tape_fingerprint": calibration_fingerprint(
            [
                (row.ts.isoformat(), row.open, row.high, row.low, row.close)
                for row in gth_signal
            ]
        ),
        "execution_symbol": "SPY",
        "signal_clock": "XSP",
        "signal_observations": signal_observations,
        "rth_state_owner": rth_state,
        "daily_context_bars": len(xsp_daily_bars or ()),
        "spy_tape_fingerprint": calibration_fingerprint(
            [
                (
                    row.ts.isoformat(),
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                )
                for row in normalized_spy
            ]
        ),
        "profiles": profiles,
        "execution_eligibility": dict(execution_gate),
        "order_authority": "none",
    }


async def advance_xsp_opening_edge_v2_from_ibkr(
    ledger: LiveCalibrationLedger,
    *,
    client,
    observed_at: datetime,
    run_started_at: datetime,
    duration_str: str = XSP_OPENING_EDGE_V2_HISTORY_DURATION,
    news_snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    recorded_at: datetime | None = None,
    spec: XspOpeningEdgeV2Spec | None = None,
    strategy_id: str = XSP_OPENING_EDGE_V2_VERSION,
    strategy_version: str = XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
    spec_loader: Callable[[], XspOpeningEdgeV2Spec] = load_xsp_opening_edge_v2_spec,
    paired_equity_builder: Callable[..., dict[str, object]] = (
        xsp_opening_edge_v2_equities
    ),
    daily_context_duration_str: str | None = None,
    execution_gate: Mapping[str, object] = XSP_OPENING_EDGE_V2_EXECUTION_GATE,
    run_start_validator: Callable[[datetime], bool] | None = None,
    include_rth_one_minute_context: bool = False,
    rth_one_minute_duration_str: str | None = None,
) -> dict[str, object]:
    """Advance one pre-frozen, non-submitting v2 observer from IBKR history."""

    from ib_insync import Index, Stock

    from ..chart_data.history import normalize_bars_to_close
    from ..utils.bar_utils import trim_incomplete_last_bar

    if observed_at.tzinfo is None or run_started_at.tzinfo is None:
        raise ValueError("Opening Edge v2 observer timestamps must be timezone-aware")
    observed_utc = observed_at.astimezone(timezone.utc)
    run_started_utc = run_started_at.astimezone(timezone.utc)
    checkpoint_recorded_at = (
        recorded_at.astimezone(timezone.utc)
        if recorded_at is not None and recorded_at.tzinfo is not None
        else datetime.now(timezone.utc)
        if recorded_at is None
        else None
    )
    if checkpoint_recorded_at is None:
        raise ValueError("Opening Edge v2 recorded_at must be timezone-aware")
    resolved_spec = spec or spec_loader()

    run_trading_day = xsp_trading_date(run_started_utc)
    if run_trading_day is None:
        raise ValueError("Opening Edge v2 run start must be inside an XSP session")
    canonical_start = (
        datetime.combine(
            run_trading_day - timedelta(days=1),
            time(20, 15),
            tzinfo=ET_ZONE,
        ).astimezone(timezone.utc)
        if run_start_validator is None
        else run_started_utc
    )
    if (
        run_start_validator is not None and not run_start_validator(run_started_utc)
    ) or (run_start_validator is None and run_started_utc != canonical_start):
        raise ValueError("Opening Edge v2 run start must be the exact GTH boundary")

    prior_starts: set[str] = set()
    prior_configs: set[str] = set()
    for row in ledger.records():
        evidence = row.get("evidence")
        if (
            row.get("kind") != "checkpoint"
            or row.get("strategy_version") != strategy_version
            or not isinstance(evidence, Mapping)
        ):
            continue
        if evidence.get("run_started_at_utc"):
            prior_starts.add(str(evidence["run_started_at_utc"]))
        if evidence.get("crown_config_fingerprint"):
            prior_configs.add(str(evidence["crown_config_fingerprint"]))
    expected_start = run_started_utc.isoformat()
    if len(prior_starts) > 1 or (prior_starts and prior_starts != {expected_start}):
        raise ValueError("Opening Edge v2 observer run start drift")
    if len(prior_configs) > 1 or (
        prior_configs and prior_configs != {resolved_spec.config_fingerprint}
    ):
        raise ValueError("Opening Edge v2 observer crown config drift")
    if not prior_starts and checkpoint_recorded_at > run_started_utc:
        raise ValueError(
            "Opening Edge v2 run start must be frozen before observation begins"
        )

    session = xsp_session_label_et(observed_utc)
    trading_day = xsp_trading_date(observed_utc)
    skip_reason = (
        "run_not_started"
        if observed_utc < run_started_utc
        else "unsupported_session"
        if session not in {None, "GTH", "RTH", "CURB"}
        else "closed_calendar"
        if session is None
        else None
    )
    if skip_reason is not None:
        status = (
            "UNSUPPORTED_SESSION" if skip_reason == "unsupported_session" else "CLOSED"
        )
        checkpoint_session = session if status == "UNSUPPORTED_SESSION" else "CLOSED"
        checkpoint = ledger.checkpoint(
            evaluation_as_of=observed_utc,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            trading_date=trading_day.isoformat() if trading_day else None,
            session=checkpoint_session,
            status=status,
            evidence={
                "run_started_at_utc": expected_start,
                "crown_config_fingerprint": resolved_spec.config_fingerprint,
                "broker_request_skipped": skip_reason,
                "paired_equity": None,
                "order_authority": "none",
            },
            recorded_at=checkpoint_recorded_at,
        )
        return {
            **ledger.receipt(),
            "status": "ok",
            "evaluation_status": status,
            "session": None if checkpoint_session == "CLOSED" else checkpoint_session,
            "run_started_at_utc": expected_start,
            "broker_request_skipped": skip_reason,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "paired_equity": None,
            "order_authority": "none",
        }

    qualified_spy = await client.qualify_proxy_contracts(Stock("SPY", "SMART", "USD"))
    spy_contract = next(
        (
            row
            for row in qualified_spy
            if int(getattr(row, "conId", 0) or 0) > 0
            and str(getattr(row, "secType", "") or "").strip().upper() == "STK"
            and str(getattr(row, "symbol", "") or "").strip().upper() == "SPY"
        ),
        None,
    )
    if spy_contract is None:
        raise RuntimeError("IBKR did not qualify SPY as STK/SMART")

    observed_et_naive = to_et(observed_utc).replace(tzinfo=None)
    raw_spy = await client.historical_bars_ohlcv(
        spy_contract,
        duration_str=str(duration_str),
        bar_size="5 mins",
        use_rth=False,
        what_to_show="TRADES",
        cache_ttl_sec=0.0,
    )
    complete_spy = trim_incomplete_last_bar(
        list(raw_spy),
        bar_size="5 mins",
        now_ref=observed_et_naive,
    )
    spy_bars = normalize_bars_to_close(
        complete_spy,
        symbol="SPY",
        bar_size="5 mins",
        use_rth=False,
        naive_ts_mode="et",
    )
    normalized_spy = normalize_xsp_v2_bars(
        spy_bars,
        observed_at=observed_utc,
        naive_ts_mode="et",
    )
    spy_gth, spy_rth = split_xsp_v2_sessions(normalized_spy)
    latest_spy_close = (
        normalized_spy[-1].ts.replace(tzinfo=timezone.utc) if normalized_spy else None
    )
    latest_spy_age = (
        max(0.0, (observed_utc - latest_spy_close).total_seconds())
        if latest_spy_close is not None
        else None
    )
    spy_fresh = bool(
        latest_spy_age is not None
        and latest_spy_age <= XSP_OPENING_EDGE_V2_FRESHNESS_SECONDS
    )

    xsp_contract = None
    raw_xsp: Sequence[Bar] = ()
    xsp_bars: Sequence[Bar] = ()
    raw_xsp_daily: Sequence[Bar] = ()
    xsp_daily_bars: Sequence[Bar] = ()
    minute_context = XspRthOneMinuteContext()
    xsp_error = None
    try:
        qualified_xsp = await client.qualify_proxy_contracts(
            Index("XSP", "CBOE", "USD")
        )
        xsp_contract = next(
            (
                row
                for row in qualified_xsp
                if int(getattr(row, "conId", 0) or 0) > 0
                and str(getattr(row, "secType", "") or "").strip().upper() == "IND"
                and str(getattr(row, "symbol", "") or "").strip().upper() == "XSP"
            ),
            None,
        )
        if xsp_contract is None:
            xsp_error = "qualification_unavailable"
        else:
            raw_xsp = await client.historical_bars_ohlcv(
                xsp_contract,
                duration_str=str(duration_str),
                bar_size="5 mins",
                use_rth=True,
                what_to_show="TRADES",
                cache_ttl_sec=0.0,
            )
            complete_xsp = trim_incomplete_last_bar(
                list(raw_xsp),
                bar_size="5 mins",
                now_ref=observed_et_naive,
            )
            xsp_bars = normalize_bars_to_close(
                complete_xsp,
                symbol="XSP",
                bar_size="5 mins",
                use_rth=True,
                naive_ts_mode="et",
            )
            if include_rth_one_minute_context and session in {"RTH", "CURB"}:
                minute_context = await load_xsp_rth_one_minute_context(
                    client,
                    spy_contract=spy_contract,
                    xsp_contract=xsp_contract,
                    duration_str=str(rth_one_minute_duration_str or duration_str),
                    observed_et_naive=observed_et_naive,
                )
            if daily_context_duration_str:
                raw_xsp_daily = await client.historical_bars_ohlcv(
                    xsp_contract,
                    duration_str=str(daily_context_duration_str),
                    bar_size="1 day",
                    use_rth=True,
                    what_to_show="TRADES",
                    cache_ttl_sec=0.0,
                )
                xsp_daily_bars = normalize_bars_to_close(
                    raw_xsp_daily,
                    symbol="XSP",
                    bar_size="1 day",
                    use_rth=True,
                    naive_ts_mode="et",
                )
    except Exception as exc:
        xsp_error = f"{type(exc).__name__}: {exc}"

    normalized_xsp = normalize_xsp_v2_bars(
        xsp_bars,
        observed_at=observed_utc,
        naive_ts_mode="et",
    )
    gth_anchor_ready = bool(
        xsp_opening_edge_v2_gth_signal_bars(normalized_spy, normalized_xsp)
    )
    if normalized_xsp and not gth_anchor_ready and xsp_error is None:
        xsp_error = "missing_completed_anchor"
    latest_xsp_close = (
        normalized_xsp[-1].ts.replace(tzinfo=timezone.utc) if normalized_xsp else None
    )
    latest_xsp_age = (
        max(0.0, (observed_utc - latest_xsp_close).total_seconds())
        if latest_xsp_close is not None
        else None
    )
    rth_provenance_fresh = bool(
        normalized_xsp
        and (
            session not in {"RTH", "CURB"}
            or (
                latest_xsp_age is not None
                and latest_xsp_age <= XSP_OPENING_EDGE_V2_FRESHNESS_SECONDS
            )
        )
    )
    one_minute_context_ready = minute_context.ready(
        required=include_rth_one_minute_context and session in {"RTH", "CURB"},
        observed_at=observed_utc,
        freshness_seconds=XSP_OPENING_EDGE_V2_FRESHNESS_SECONDS,
    )
    builder_kwargs = minute_context.builder_kwargs(enabled=include_rth_one_minute_context)
    paired_equity = (
        paired_equity_builder(
            spec=resolved_spec,
            spy_bars=spy_bars,
            xsp_rth_bars=xsp_bars or None,
            xsp_daily_bars=xsp_daily_bars or None,
            observed_at=observed_utc,
            run_started_at=run_started_utc,
            naive_ts_mode="et",
            **builder_kwargs,
        )
        if spy_gth and spy_rth and gth_anchor_ready and one_minute_context_ready
        else None
    )
    evaluation_status = (
        "NO_DATA"
        if paired_equity is None
        else "EVALUATED"
        if spy_fresh and rth_provenance_fresh
        else "STALE_DATA"
    )
    daily_context_count = (
        int(paired_equity.get("daily_context_bars") or 0)
        if isinstance(paired_equity, Mapping)
        else len(xsp_daily_bars)
    )

    fundamental = xsp_fundamental_context_at(
        news_snapshot,
        decision_at=observed_utc,
    )
    fundamental_log = {
        field: fundamental.get(field)
        for field in (
            "source",
            "authority",
            "usable",
            "signal_as_of_utc",
            "snapshot_fingerprint",
            "direction",
            "impact",
            "confidence",
            "age_seconds",
            "horizon_hours",
            "reason",
            "signed_pressure",
            "pressure_delta",
            "pressure_interval_seconds",
            "pressure_velocity_per_hour",
        )
    }
    last_request = getattr(client, "last_historical_request", None)
    spy_request = last_request(spy_contract) if callable(last_request) else None
    xsp_request = (
        last_request(xsp_contract)
        if callable(last_request) and xsp_contract is not None
        else None
    )
    checkpoint = ledger.checkpoint(
        evaluation_as_of=observed_utc,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=session,
        status=evaluation_status,
        evidence={
            "run_started_at_utc": expected_start,
            "crown_config_fingerprint": resolved_spec.config_fingerprint,
            "paired_equity": paired_equity,
            "spy_contract": {
                "con_id": int(getattr(spy_contract, "conId", 0) or 0),
                "symbol": str(getattr(spy_contract, "symbol", "") or ""),
                "sec_type": str(getattr(spy_contract, "secType", "") or ""),
                "exchange": str(getattr(spy_contract, "exchange", "") or ""),
                "currency": str(getattr(spy_contract, "currency", "") or ""),
            },
            "spy_historical_request": spy_request,
            "spy_raw_bars": len(raw_spy),
            "spy_complete_close_aligned_bars": len(normalized_spy),
            "spy_gth_bars": len(spy_gth),
            "spy_rth_bars": len(spy_rth),
            "latest_spy_close_utc": (
                latest_spy_close.isoformat() if latest_spy_close is not None else None
            ),
            "latest_spy_age_sec": latest_spy_age,
            "spy_history_fresh": spy_fresh,
            **minute_context.evidence(enabled=include_rth_one_minute_context),
            **({"one_minute_context_ready": one_minute_context_ready}
               if include_rth_one_minute_context else {}),
            "xsp_contract_con_id": (
                int(getattr(xsp_contract, "conId", 0) or 0)
                if xsp_contract is not None
                else None
            ),
            "xsp_historical_request": xsp_request,
            "xsp_raw_bars": len(raw_xsp),
            "xsp_complete_close_aligned_bars": len(normalized_xsp),
            "xsp_daily_raw_bars": len(raw_xsp_daily),
            "xsp_daily_context_bars": daily_context_count,
            "latest_xsp_close_utc": (
                latest_xsp_close.isoformat() if latest_xsp_close is not None else None
            ),
            "latest_xsp_age_sec": latest_xsp_age,
            "rth_signal_source": "XSP" if normalized_xsp else None,
            "rth_provenance_fresh": rth_provenance_fresh,
            "xsp_error": xsp_error,
            "fundamental_pressure": fundamental_log,
            "execution_eligibility": dict(execution_gate),
            "order_authority": "none",
        },
        recorded_at=checkpoint_recorded_at,
    )
    return {
        **ledger.receipt(),
        "status": "ok" if paired_equity is not None else "no_bars",
        "evaluation_status": evaluation_status,
        "session": session,
        "run_started_at_utc": expected_start,
        "freshness_ok": evaluation_status == "EVALUATED",
        "spy_contract": dict(checkpoint["evidence"]["spy_contract"]),
        "spy_raw_bars": len(raw_spy),
        "spy_complete_close_aligned_bars": len(normalized_spy),
        "xsp_raw_bars": len(raw_xsp),
        "xsp_complete_close_aligned_bars": len(normalized_xsp),
        "xsp_daily_context_bars": daily_context_count,
        "rth_signal_source": "XSP" if normalized_xsp else None,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "paired_equity": paired_equity,
        "recorded_at_utc": checkpoint_recorded_at.isoformat(),
        "fundamental_pressure": fundamental_log,
        "order_authority": "none",
    }
