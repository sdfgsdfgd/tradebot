"""Prospective, non-submitting ownership of Opening Edge v3 Regime Harmony."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from ..backtest.data import ContractMeta
from ..backtest.models import Bar
from ..engines.market import xsp_session_label_et, xsp_trading_date
from ..spot.champions import (
    discover_current_champions,
    load_champion_group,
    repo_root,
)
from .xsp_opening_edge_state import (
    XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS,
    XspDailyBar,
    XspOpeningEdgeV3StateOwner,
    merge_xsp_daily_context,
    validate_xsp_daily_bars,
    xsp_daily_bars_from_intraday,
)
from ..time_utils import NaiveTsModeInput, to_et
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .xsp_dual_clock import (
    XSP_DUAL_CLOCK_PAIRED_SCHEMA,
    XSP_DUAL_CLOCK_SOURCE_VERSION,
    XSP_DUAL_CLOCK_VERSION,
    xsp_dual_clock_bridge_result,
    xsp_dual_clock_emissions,
    xsp_dual_clock_target,
)
from .xsp_opening_edge_v2 import (
    XSP_OPENING_EDGE_V2_COSTS,
    XSP_OPENING_EDGE_V2_HISTORY_DURATION,
    XspOpeningEdgeV2Spec,
    advance_xsp_opening_edge_v2_from_ibkr,
    normalize_xsp_v2_bars,
    xsp_opening_edge_v2_bundle,
    xsp_opening_edge_v2_equities,
)


XSP_OPENING_EDGE_V3_VERSION = "xsp.opening-edge-v3-regime-harmony-24x5.v1"
XSP_OPENING_EDGE_V3_TRANSPORT_VERSION = (
    "xsp.opening-edge-v3-regime-harmony-spy-transport.v1"
)
XSP_OPENING_EDGE_V3_HISTORY_DURATION = XSP_OPENING_EDGE_V2_HISTORY_DURATION
XSP_OPENING_EDGE_V3_CONTEXT_SCHEMA = "xsp.opening-edge-v3-daily-context-seed.v1"
XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA = (
    "xsp.opening-edge-v3-daily-context-state.v1"
)
XSP_OPENING_EDGE_V3_NEWS_PAIR_SCHEMA = (
    "xsp.opening-edge-v3-fundamental-pair.v1"
)
XSP_OPENING_EDGE_V3_NEWS_HORIZON_MINUTES = 60
XSP_OPENING_EDGE_V3_EXECUTION_GATE = {
    "verdict": "HOLD",
    "eligible": False,
    "reason": "v3_rth_cash_transport_requires_selection_proof",
    "qualified_transport": {
        "scope": "RTH whole-share UPRO/SPXU",
        "current_fixed_full_cross_net_usd": 337.0615,
        "current_fixed_intrabar_drawdown_usd": 135.3126,
        "tiered_full_cross_net_usd": 911.8653,
        "tiered_intrabar_drawdown_usd": 102.6829,
    },
    "open_gates": (
        "fresh_v3_instrument_books",
        "exact_quote_derived_quantity_preview",
        "effective_tiered_commission_preview",
        "restart_and_identity_proof",
        "gth_cash_transport_or_explicit_rth_scope",
    ),
    "order_authority": "none",
}


@dataclass(frozen=True)
class XspOpeningEdgeV3Spec:
    artifact_path: Path
    artifact_sha256: str
    declaration_path: Path
    declaration_version: str
    strategy_key: str
    group: Mapping[str, object]
    config_fingerprint: str
    state_owner_sha256: str
    daily_context_seed_path: Path
    daily_context_seed_sha256: str
    daily_context_seed: tuple[XspDailyBar, ...]


def load_xsp_opening_edge_v3_spec(
    *,
    root: Path | None = None,
) -> XspOpeningEdgeV3Spec:
    """Load only the content-addressed current v3 XSP LF crown."""

    resolved_root = (root or repo_root()).resolve()
    refs = discover_current_champions(
        root=resolved_root,
        symbols=("XSP",),
        tracks=("LF",),
    )
    if len(refs) != 1:
        raise ValueError("exactly one current XSP LF crown is required")
    ref = refs[0]
    payload = json.loads(ref.artifact_path.read_text(encoding="utf-8"))
    group = load_champion_group(ref)
    artifact_sha256 = hashlib.sha256(ref.artifact_path.read_bytes()).hexdigest()
    state_owner_path = resolved_root / "tradebot/research/xsp_opening_edge_state.py"
    state_owner_sha256 = hashlib.sha256(state_owner_path.read_bytes()).hexdigest()
    receipts = payload.get("receipts")
    if not isinstance(receipts, Mapping):
        raise ValueError("Opening Edge v3 receipts are missing")
    daily_context_seed_path = resolved_root / str(
        receipts.get("daily_context_seed") or ""
    )
    daily_context_seed_sha256 = hashlib.sha256(
        daily_context_seed_path.read_bytes()
    ).hexdigest()
    seed_payload = json.loads(daily_context_seed_path.read_text(encoding="utf-8"))
    if not isinstance(seed_payload, Mapping):
        raise ValueError("Opening Edge v3 daily context seed is invalid")
    seed_rows = seed_payload.get("sessions")
    if (
        seed_payload.get("schema") != XSP_OPENING_EDGE_V3_CONTEXT_SCHEMA
        or seed_payload.get("order_authority") != "none"
        or str(receipts.get("daily_context_seed_sha256") or "")
        != daily_context_seed_sha256
        or not isinstance(seed_rows, Sequence)
    ):
        raise ValueError("Opening Edge v3 daily context seed is invalid")
    daily_context_seed = validate_xsp_daily_bars(
        tuple(
            XspDailyBar(
                day=datetime.fromisoformat(str(row["day"])).date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in seed_rows
            if isinstance(row, Mapping)
        )
    )
    if len(daily_context_seed) != len(seed_rows):
        raise ValueError("Opening Edge v3 daily context seed row is invalid")
    seed_fingerprint = XspOpeningEdgeV3StateOwner(
        daily_context_seed
    ).context_fingerprint
    if seed_fingerprint != seed_payload.get("context_fingerprint"):
        raise ValueError("Opening Edge v3 daily context fingerprint drifted")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != XSP_OPENING_EDGE_V3_VERSION
        or str(ref.version or "") != "3"
        or not str(ref.strategy_key or "").strip()
        or not isinstance(group, Mapping)
        or str(group.get("_key") or "") != str(ref.strategy_key)
        or payload.get("order_authority") != "none"
    ):
        raise ValueError("current XSP LF declaration is not Opening Edge v3")
    identity = {
        "schema": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "artifact_sha256": artifact_sha256,
        "state_owner_sha256": state_owner_sha256,
        "daily_context_seed_sha256": daily_context_seed_sha256,
        "declaration_version": str(ref.version),
        "strategy_key": str(ref.strategy_key),
        "signal_clock": "XSP",
        "execution_symbol": "SPY",
        "daily_context_sessions": XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS,
        "daily_context_seed_sessions": len(daily_context_seed),
        "cost_profiles": XSP_OPENING_EDGE_V2_COSTS,
        "execution_gate": XSP_OPENING_EDGE_V3_EXECUTION_GATE,
        "order_authority": "none",
    }
    return XspOpeningEdgeV3Spec(
        artifact_path=ref.artifact_path,
        artifact_sha256=artifact_sha256,
        declaration_path=ref.declaration_path,
        declaration_version=str(ref.version),
        strategy_key=str(ref.strategy_key),
        group=dict(group),
        config_fingerprint=calibration_fingerprint(identity),
        state_owner_sha256=state_owner_sha256,
        daily_context_seed_path=daily_context_seed_path,
        daily_context_seed_sha256=daily_context_seed_sha256,
        daily_context_seed=daily_context_seed,
    )


def xsp_opening_edge_v3_bundle(
    spec: XspOpeningEdgeV3Spec,
    **kwargs,
):
    """Hydrate a v3 leaf through the unchanged v2 transport contract."""

    return xsp_opening_edge_v2_bundle(
        cast(XspOpeningEdgeV2Spec, spec),
        rth_entry_name="RTH Regime Harmony Core",
        **kwargs,
    )


def _daily_context(
    bars: Sequence[Bar],
    *,
    observed_at: datetime,
    naive_ts_mode: NaiveTsModeInput,
) -> tuple[XspDailyBar, ...]:
    normalized = normalize_xsp_v2_bars(
        bars,
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    rows = tuple(
        XspDailyBar(
            day=to_et(row.ts, naive_ts_mode="utc").date(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
        )
        for row in normalized
    )
    if len(rows) < XSP_OPENING_EDGE_V3_CONTEXT_SESSIONS:
        raise ValueError("Opening Edge v3 daily context is underwarmed")
    if len({row.day for row in rows}) != len(rows):
        raise ValueError("Opening Edge v3 daily context contains duplicate days")
    return rows


def xsp_opening_edge_v3_equities(
    *,
    spec: XspOpeningEdgeV3Spec,
    spy_bars: Sequence[Bar],
    observed_at: datetime,
    run_started_at: datetime,
    xsp_rth_bars: Sequence[Bar] | None = None,
    xsp_daily_bars: Sequence[Bar] | None = None,
    persisted_daily_bars: Sequence[XspDailyBar] = (),
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> dict[str, object]:
    """Replay v3 from the frozen five-minute context seed and exact appends."""

    _ = xsp_daily_bars
    normalized_rth = normalize_xsp_v2_bars(
        tuple(xsp_rth_bars or ()),
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    fresh_daily = xsp_daily_bars_from_intraday(normalized_rth)
    daily = merge_xsp_daily_context(
        spec.daily_context_seed,
        persisted=persisted_daily_bars,
        fresh=fresh_daily,
    )
    paired = xsp_opening_edge_v2_equities(
        spec=cast(XspOpeningEdgeV2Spec, spec),
        spy_bars=spy_bars,
        xsp_rth_bars=xsp_rth_bars,
        xsp_daily_bars=cast(Sequence[Bar], daily),
        observed_at=observed_at,
        run_started_at=run_started_at,
        naive_ts_mode=naive_ts_mode,
        rth_state_owner_factory=lambda: XspOpeningEdgeV3StateOwner(daily),
        rth_entry_name="RTH Regime Harmony Core",
        strategy_version=XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        paired_schema="xsp.opening-edge-v3-paired-equity.v1",
        candidate_schema="xsp.opening-edge-v3-candidate-equity.v1",
        execution_gate=XSP_OPENING_EDGE_V3_EXECUTION_GATE,
    )
    context_owner = XspOpeningEdgeV3StateOwner(daily)
    context_day = xsp_trading_date(observed_at)
    context_state = (
        context_owner.context_for_day(context_day)
        if context_day is not None
        else None
    )
    paired["state_owner_sha256"] = spec.state_owner_sha256
    paired["daily_context_seed_sha256"] = spec.daily_context_seed_sha256
    paired["daily_context_bars"] = len(daily)
    paired["daily_context_fingerprint"] = context_owner.context_fingerprint
    paired["daily_context_state"] = (
        {
            "schema": XSP_OPENING_EDGE_V3_CONTEXT_STATE_SCHEMA,
            "trading_day": context_day.isoformat(),
            "context_as_of_day": context_state["as_of_day"],
            "state_fingerprint": calibration_fingerprint(context_state),
            "state": dict(context_state),
        }
        if context_day is not None and context_state is not None
        else None
    )
    paired["daily_context_appends"] = [
        {**asdict(row), "day": row.day.isoformat()}
        for row in daily
        if row.day > spec.daily_context_seed[-1].day
    ]
    return paired


def xsp_opening_edge_p009_equities(
    *,
    spec: XspOpeningEdgeV3Spec,
    spy_bars: Sequence[Bar],
    observed_at: datetime,
    run_started_at: datetime,
    xsp_rth_bars: Sequence[Bar] | None = None,
    xsp_daily_bars: Sequence[Bar] | None = None,
    spy_rth_one_minute_bars: Sequence[Bar] = (),
    xsp_rth_one_minute_bars: Sequence[Bar] = (),
    persisted_daily_bars: Sequence[XspDailyBar] = (),
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> dict[str, object]:
    """Overlay the frozen opening bridge without changing the v3 owner."""

    paired = xsp_opening_edge_v3_equities(
        spec=spec,
        spy_bars=spy_bars,
        observed_at=observed_at,
        run_started_at=run_started_at,
        xsp_rth_bars=xsp_rth_bars,
        xsp_daily_bars=xsp_daily_bars,
        persisted_daily_bars=persisted_daily_bars,
        naive_ts_mode=naive_ts_mode,
    )
    normalized_rth = normalize_xsp_v2_bars(
        tuple(xsp_rth_bars or ()),
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    normalized_spy = normalize_xsp_v2_bars(
        spy_bars,
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    normalized_xsp_one = normalize_xsp_v2_bars(
        xsp_rth_one_minute_bars,
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    normalized_spy_one = normalize_xsp_v2_bars(
        spy_rth_one_minute_bars,
        observed_at=observed_at,
        naive_ts_mode=naive_ts_mode,
    )
    profiles = paired.get("profiles")
    broker_profile = (
        profiles.get("broker") if isinstance(profiles, Mapping) else None
    )
    v3_position = (
        broker_profile.get("latest_position")
        if isinstance(broker_profile, Mapping)
        else None
    )
    emissions = ()
    target = dict(v3_position) if isinstance(v3_position, Mapping) else None
    bridge_state = None
    if normalized_xsp_one and normalized_spy_one:
        fresh_daily = xsp_daily_bars_from_intraday(normalized_rth)
        daily = merge_xsp_daily_context(
            spec.daily_context_seed,
            persisted=persisted_daily_bars,
            fresh=fresh_daily,
        )
        emissions = xsp_dual_clock_emissions(
            xsp_rth_one_minute=normalized_xsp_one,
            spy_rth_one_minute=normalized_spy_one,
            spy_full_five_minute=normalized_spy,
        )
        start = xsp_trading_date(run_started_at)
        end = xsp_trading_date(observed_at)
        if start is None or end is None:
            raise ValueError("P-009 source must be inside one XSP run")
        cfg = xsp_opening_edge_v3_bundle(
            spec,
            lane="rth",
            start=start,
            end=end,
            cost_profile="research",
            rth_signal_symbol="XSP",
        )
        cfg = replace(
            cfg,
            strategy=replace(cfg.strategy, spot_exec_bar_size="1 min"),
        )
        result, owner = xsp_dual_clock_bridge_result(
            cfg=cfg,
            v3_bars=normalized_rth,
            exec_bars=normalized_xsp_one,
            emissions=emissions,
            daily_context=daily,
            meta=ContractMeta(
                symbol="XSP", exchange="CBOE", multiplier=1.0, min_tick=0.01
            ),
            entry_not_before=run_started_at.astimezone(timezone.utc).replace(
                tzinfo=None
            ),
            final_session_complete=xsp_session_label_et(observed_at)
            not in {"RTH", "CURB"},
        )
        target = xsp_dual_clock_target(
            bridge_result=result,
            bridge_owner=owner,
            v3_position=v3_position if isinstance(v3_position, Mapping) else None,
            observed_at=observed_at,
        )
        bridge_state = owner.state_payload()

    identity = {
        "strategy_version": XSP_DUAL_CLOCK_VERSION,
        "source_strategy_version": XSP_DUAL_CLOCK_SOURCE_VERSION,
        "v3_crown_config_fingerprint": paired["crown_config_fingerprint"],
        "minute_window": [5, 7],
        "minimum_true_range_multiple": 0.5,
        "minimum_volume_authority_level": 20.0,
        "slow_front": "15_completed_five_minute_bars_75_minutes",
    }
    paired.update(
        schema=XSP_DUAL_CLOCK_PAIRED_SCHEMA,
        strategy_version=XSP_DUAL_CLOCK_SOURCE_VERSION,
        v3_crown_config_fingerprint=paired["crown_config_fingerprint"],
        crown_config_fingerprint=calibration_fingerprint(identity),
        dual_clock_target=target,
        dual_clock_emissions=[row.as_payload() for row in emissions],
        dual_clock_state=bridge_state,
        dual_clock_identity=identity,
    )
    return paired


def _persisted_daily_context(
    records: Sequence[Mapping[str, object]],
    *,
    strategy_versions: Sequence[str] = (XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,),
) -> tuple[XspDailyBar, ...]:
    by_day: dict[date, XspDailyBar] = {}
    for record in records:
        if record.get("strategy_version") not in set(strategy_versions):
            continue
        evidence = record.get("evidence")
        paired = (
            evidence.get("paired_equity") if isinstance(evidence, Mapping) else None
        )
        appends = (
            paired.get("daily_context_appends", ())
            if isinstance(paired, Mapping)
            else ()
        )
        if not isinstance(appends, Sequence):
            raise ValueError("Opening Edge v3 persisted context is malformed")
        for raw in appends:
            if not isinstance(raw, Mapping):
                raise ValueError("Opening Edge v3 persisted context row is malformed")
            row = XspDailyBar(
                day=datetime.fromisoformat(str(raw["day"])).date(),
                open=float(raw["open"]),
                high=float(raw["high"]),
                low=float(raw["low"]),
                close=float(raw["close"]),
            )
            existing = by_day.get(row.day)
            if existing is not None and existing != row:
                raise ValueError("Opening Edge v3 persisted context drifted")
            by_day[row.day] = row
    return validate_xsp_daily_bars(
        tuple(by_day[day] for day in sorted(by_day)),
        minimum_sessions=0,
    )


def xsp_opening_edge_p009_run_start(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at: datetime,
) -> datetime:
    starts = {
        str(evidence["run_started_at_utc"])
        for row in records
        if row.get("kind") == "checkpoint"
        and row.get("strategy_version") == XSP_DUAL_CLOCK_SOURCE_VERSION
        and isinstance((evidence := row.get("evidence")), Mapping)
        and evidence.get("run_started_at_utc")
    }
    if len(starts) > 1:
        raise ValueError("Opening Edge P-009 observer run start drift")
    if starts:
        return datetime.fromisoformat(starts.pop().replace("Z", "+00:00"))
    return next_xsp_v3_run_start(observed_at)


def xsp_opening_edge_v3_run_start(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at: datetime,
) -> datetime:
    starts = {
        str(evidence["run_started_at_utc"])
        for row in records
        if row.get("kind") == "checkpoint"
        and row.get("strategy_version") == XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        and isinstance((evidence := row.get("evidence")), Mapping)
        and evidence.get("run_started_at_utc")
    }
    if len(starts) > 1:
        raise ValueError("Opening Edge v3 observer run start drift")
    if starts:
        return datetime.fromisoformat(starts.pop().replace("Z", "+00:00"))
    return next_xsp_v3_run_start(observed_at)


def next_xsp_v3_run_start(observed_at: datetime) -> datetime:
    observed_et = to_et(observed_at)
    candidate = observed_et.replace(second=0, microsecond=0) + timedelta(
        minutes=5 - observed_et.minute % 5
    )
    for _ in range(12 * 24 * 8):
        if is_xsp_v3_run_start(candidate):
            return candidate.astimezone(timezone.utc)
        candidate += timedelta(minutes=5)
    raise ValueError("unable to resolve next Opening Edge v3 run start")


def is_xsp_v3_run_start(value: datetime) -> bool:
    candidate = to_et(value)
    return bool(
        candidate.second == 0
        and candidate.microsecond == 0
        and candidate.minute % 5 == 0
        and xsp_session_label_et(candidate) == "GTH"
    )


def xsp_opening_edge_v3_fundamental_pairs(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join actual v3 entries to exact forward checkpoints without backfill."""

    def timestamp(value: object) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )

    checkpoints = [
        row
        for row in records
        if row.get("kind") == "checkpoint"
        and row.get("strategy_version") == XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
        and row.get("status") == "EVALUATED"
        and isinstance(row.get("evidence"), Mapping)
        and isinstance(row["evidence"].get("paired_equity"), Mapping)
    ]
    trades: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for checkpoint in checkpoints:
        paired = checkpoint["evidence"]["paired_equity"]
        profiles = paired.get("profiles")
        research = (
            profiles.get("research") if isinstance(profiles, Mapping) else None
        )
        if not isinstance(research, Mapping):
            continue
        for field in ("latest_position", "latest_trade"):
            trade = research.get(field)
            attribution = (
                trade.get("attribution") if isinstance(trade, Mapping) else None
            )
            entry = (
                attribution.get("entry")
                if isinstance(attribution, Mapping)
                else None
            )
            if not isinstance(trade, Mapping) or not isinstance(entry, Mapping):
                continue
            lane = str(trade.get("lane") or "").lower()
            direction = str(trade.get("direction") or "").lower()
            signal_at = str(entry.get("signal_bar_ts") or "")
            entry_at = str(trade.get("entry_time") or "")
            if (
                lane not in {"rth", "gth"}
                or direction not in {"up", "down"}
                or (lane == "gth" and direction != "down")
                or not signal_at
                or not entry_at
            ):
                continue
            trades.setdefault(
                (lane, entry_at, direction, signal_at),
                trade,
            )

    pairs = []
    for lane, entry_raw, direction, signal_raw in sorted(trades):
        try:
            decision_at = timestamp(signal_raw)
            entry_at = timestamp(entry_raw)
        except (TypeError, ValueError):
            continue
        source_candidates = []
        for checkpoint in checkpoints:
            evidence = checkpoint["evidence"]
            paired = evidence["paired_equity"]
            observations = paired.get("signal_observations")
            snapshot = (
                observations.get(lane)
                if isinstance(observations, Mapping)
                else None
            )
            control = (
                snapshot.get("entry_control")
                if isinstance(snapshot, Mapping)
                else None
            )
            try:
                recorded_at = timestamp(checkpoint["recorded_at_utc"])
                signal_at = timestamp(snapshot["signal_bar_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not isinstance(control, Mapping)
                or signal_at != decision_at
                or snapshot.get("signal_snapshot_age_bars") != 0
                or str(control.get("direction") or "").lower() != direction
                or recorded_at >= entry_at
            ):
                continue
            source_candidates.append((recorded_at, checkpoint, snapshot))
        if not source_candidates:
            continue
        _, source, source_snapshot = min(
            source_candidates,
            key=lambda row: row[0],
        )
        outcome_at = decision_at + timedelta(
            minutes=XSP_OPENING_EDGE_V3_NEWS_HORIZON_MINUTES
        )
        outcome_candidates = []
        for checkpoint in checkpoints:
            if checkpoint.get("trading_date") != source.get("trading_date"):
                continue
            paired = checkpoint["evidence"]["paired_equity"]
            observations = paired.get("signal_observations")
            snapshot = (
                observations.get(lane)
                if isinstance(observations, Mapping)
                else None
            )
            try:
                recorded_at = timestamp(checkpoint["recorded_at_utc"])
                signal_at = timestamp(snapshot["signal_bar_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                signal_at == outcome_at
                and snapshot.get("signal_snapshot_age_bars") == 0
                and recorded_at >= outcome_at
            ):
                outcome_candidates.append((recorded_at, checkpoint, snapshot))
        if not outcome_candidates:
            continue
        _, outcome, outcome_snapshot = min(
            outcome_candidates,
            key=lambda row: row[0],
        )
        try:
            decision_close = float(source_snapshot["close"])
            outcome_close = float(outcome_snapshot["close"])
        except (KeyError, TypeError, ValueError):
            continue
        news_raw = source["evidence"].get("fundamental_pressure")
        news = dict(news_raw) if isinstance(news_raw, Mapping) else {}
        news.setdefault("source", "causal_news")
        news.setdefault("authority", "observation_only")
        sign = 1.0 if direction == "up" else -1.0
        forecast_id = calibration_fingerprint(
            {
                "schema": XSP_OPENING_EDGE_V3_NEWS_PAIR_SCHEMA,
                "source_checkpoint_id": source["checkpoint_id"],
                "lane": lane,
                "direction": direction,
                "decision_as_of_utc": decision_at.isoformat(),
                "outcome_not_before_utc": outcome_at.isoformat(),
                "entry": "confirmed_v3_lifecycle_entry",
                "exit": "exact_60m_signal_close",
                "friction_points": XSP_OPENING_EDGE_V2_COSTS["research"][
                    "round_trip_points"
                ],
            }
        )
        pairs.append(
            {
                "forecast_id": forecast_id,
                "decision_at": decision_at,
                "direction": direction,
                "ta_points": (
                    (outcome_close - decision_close) * sign
                    - float(
                        XSP_OPENING_EDGE_V2_COSTS["research"][
                            "round_trip_points"
                        ]
                    )
                ),
                "context": {
                    "schema": XSP_OPENING_EDGE_V3_NEWS_PAIR_SCHEMA,
                    "evidence_mode": "forward_v3_checkpoint",
                    "lane": lane,
                    "source_checkpoint_id": source["checkpoint_id"],
                    "outcome_checkpoint_id": outcome["checkpoint_id"],
                    "decision_close": decision_close,
                    "outcome_close": outcome_close,
                    "directional_impulse": source_snapshot.get(
                        "directional_impulse"
                    ),
                    "entry_control": source_snapshot.get("entry_control"),
                    "daily_context_state": source["evidence"][
                        "paired_equity"
                    ].get("daily_context_state"),
                    "fundamental_pressure": news,
                },
                "evidence_mode": "forward_v3_checkpoint",
                "prospective": True,
            }
        )
    return pairs


async def advance_xsp_opening_edge_v3_from_ibkr(
    ledger: LiveCalibrationLedger,
    *,
    client,
    observed_at: datetime,
    run_started_at: datetime,
    duration_str: str = XSP_OPENING_EDGE_V3_HISTORY_DURATION,
    news_snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    recorded_at: datetime | None = None,
    spec: XspOpeningEdgeV3Spec | None = None,
) -> dict[str, object]:
    """Advance one pre-frozen, non-submitting v3 observer."""

    resolved_spec = spec or load_xsp_opening_edge_v3_spec()
    persisted_daily = _persisted_daily_context(tuple(ledger.records()))

    def build_equities(**kwargs) -> dict[str, object]:
        kwargs["spec"] = resolved_spec
        kwargs["persisted_daily_bars"] = persisted_daily
        return xsp_opening_edge_v3_equities(**kwargs)

    return await advance_xsp_opening_edge_v2_from_ibkr(
        ledger,
        client=client,
        observed_at=observed_at,
        run_started_at=run_started_at,
        duration_str=duration_str,
        news_snapshot=news_snapshot,
        recorded_at=recorded_at,
        spec=cast(XspOpeningEdgeV2Spec, resolved_spec),
        strategy_id=XSP_OPENING_EDGE_V3_VERSION,
        strategy_version=XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        spec_loader=cast(object, load_xsp_opening_edge_v3_spec),
        paired_equity_builder=cast(object, build_equities),
        execution_gate=XSP_OPENING_EDGE_V3_EXECUTION_GATE,
        run_start_validator=is_xsp_v3_run_start,
    )


async def advance_xsp_opening_edge_p009_from_ibkr(
    ledger: LiveCalibrationLedger,
    *,
    client,
    observed_at: datetime,
    run_started_at: datetime,
    duration_str: str = XSP_OPENING_EDGE_V3_HISTORY_DURATION,
    news_snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    recorded_at: datetime | None = None,
    spec: XspOpeningEdgeV3Spec | None = None,
) -> dict[str, object]:
    """Advance the centralized P-009 source with no order authority."""

    resolved_spec = spec or load_xsp_opening_edge_v3_spec()
    persisted_daily = _persisted_daily_context(
        tuple(ledger.records()),
        strategy_versions=(
            XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
            XSP_DUAL_CLOCK_SOURCE_VERSION,
        ),
    )

    def build_equities(**kwargs) -> dict[str, object]:
        kwargs["spec"] = resolved_spec
        kwargs["persisted_daily_bars"] = persisted_daily
        return xsp_opening_edge_p009_equities(**kwargs)

    return await advance_xsp_opening_edge_v2_from_ibkr(
        ledger,
        client=client,
        observed_at=observed_at,
        run_started_at=run_started_at,
        duration_str=duration_str,
        news_snapshot=news_snapshot,
        recorded_at=recorded_at,
        spec=cast(XspOpeningEdgeV2Spec, resolved_spec),
        strategy_id=XSP_DUAL_CLOCK_VERSION,
        strategy_version=XSP_DUAL_CLOCK_SOURCE_VERSION,
        spec_loader=cast(object, load_xsp_opening_edge_v3_spec),
        paired_equity_builder=cast(object, build_equities),
        execution_gate=XSP_OPENING_EDGE_V3_EXECUTION_GATE,
        run_start_validator=is_xsp_v3_run_start,
        include_rth_one_minute_context=True,
        rth_one_minute_duration_str="1 D",
    )
