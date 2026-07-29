"""Prospective, non-submitting ownership of Opening Edge v3 Regime Harmony."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from ..backtest.models import Bar
from ..engines.market import xsp_session_label_et
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
    paired["state_owner_sha256"] = spec.state_owner_sha256
    paired["daily_context_seed_sha256"] = spec.daily_context_seed_sha256
    paired["daily_context_bars"] = len(daily)
    paired["daily_context_fingerprint"] = XspOpeningEdgeV3StateOwner(
        daily
    ).context_fingerprint
    paired["daily_context_appends"] = [
        row.as_payload() for row in daily if row.day > spec.daily_context_seed[-1].day
    ]
    return paired


def _persisted_daily_context(
    records: Sequence[Mapping[str, object]],
) -> tuple[XspDailyBar, ...]:
    by_day: dict[date, XspDailyBar] = {}
    for record in records:
        evidence = record.get("evidence")
        paired = (
            evidence.get("paired_equity") if isinstance(evidence, Mapping) else None
        )
        appends = (
            paired.get("daily_context_appends") if isinstance(paired, Mapping) else ()
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
