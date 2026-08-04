from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.chart_data.series import OhlcvBar
from tradebot.engines.directional_impulse import (
    DirectionalImpulseHorizon,
    DirectionalImpulseSnapshot,
)
from tradebot.research.mcl_two_speed_auction import (
    MCL_TWO_SPEED_AUCTION_AUTHORITY,
    MCL_TWO_SPEED_AUCTION_HORIZONS,
    MCL_TWO_SPEED_AUCTION_POLICY,
    MCL_TWO_SPEED_AUCTION_VERSION,
    MclAuctionBar,
    MclTwoSpeedAuctionEngine,
    route_mcl_v18_direction,
)
from tradebot.spot.champions import discover_current_champions, load_champion_group


START = datetime(2026, 8, 4, 4, 0, tzinfo=timezone.utc)


def test_v18_research_crown_is_machine_bound_but_cannot_trade() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = discover_current_champions(root=root, symbols=("MCL",), tracks=("HF",))

    assert len(refs) == 1
    ref = refs[0]
    declaration = json.loads(ref.declaration_path.read_text())
    artifact = json.loads(ref.artifact_path.read_text())
    group = load_champion_group(ref)

    assert ref.version == "18"
    assert group is not None
    assert group["_key"] == "mcl-two-speed-auction-relay-v18"
    assert hashlib.sha256(ref.artifact_path.read_bytes()).hexdigest() == declaration[
        "artifact_sha256"
    ]
    assert declaration["promotion"]["eligible"] is True
    assert declaration["promotion"]["order_authority"] == "none"
    assert artifact["authority"] == "historical_research_crown_only"
    assert artifact["order_authority"] == "none"
    assert artifact["selection_authority"] == "none"
    assert artifact["capital_authority"] == "none"
    assert artifact["coronation"]["history_exception"][
        "generic_july_2023_requirement"
    ] == "waived_for_mcl_research_coronation"
    assert artifact["graduation_enrollment"]["lifecycle_state"] == "CROWNED"
    assert artifact["graduation_enrollment"]["live_24h"] == "NOT_STARTED"
    assert artifact["prospective"]["complete_unseen_raw_turns"] == 0
    leaderboard = (root / "backtests/mcl/leaderboard.md").read_text()
    assert "CR-001 · 2026-08-04 · MCL Two-Speed Auction Relay — V18" in leaderboard
    assert artifact["schema"] in leaderboard
    assert declaration["artifact_sha256"] in leaderboard


def _horizon(bars: int, velocity: float | None) -> DirectionalImpulseHorizon:
    return DirectionalImpulseHorizon(
        bars=bars,
        elapsed_minutes=float(bars * 5),
        observations=bars + 1,
        anchor_lag_minutes=0.0,
        return_pct=0.1,
        slope_pct_per_bar=0.01,
        slope_velocity_pct_per_bar=velocity,
        slope_angle_deg=1.0,
        efficiency=0.5,
        tr_mean_pct=0.1,
        turn=None,
        turn_age_bars=1,
    )


def _snapshot(
    *,
    turn: str | None = None,
    trend: str | None = None,
    velocities: tuple[float | None, ...] = (-0.1, -0.1, -0.1, 0.1, 0.1),
) -> DirectionalImpulseSnapshot:
    horizons = tuple(
        _horizon(bars, velocity)
        for bars, velocity in zip(MCL_TWO_SPEED_AUCTION_HORIZONS, velocities)
    )
    return DirectionalImpulseSnapshot(
        ready=True,
        direction=trend,
        abstain_reason=None,
        direction_score=-0.2 if trend == "down" else 0.2,
        coherence=0.8,
        conviction=0.5,
        atr_fast_pct=0.1,
        atr_slow_pct=0.2,
        atr_ratio=0.5,
        atr_velocity_pct=0.01,
        atr_acceleration_pct=0.001,
        turn_sequence_direction=None,
        turn_sequence_order=None,
        turn_sequence_span_bars=None,
        observed_horizons=len(horizons),
        required_turn_horizons=3,
        turn_ready=True,
        turn_abstain_reason=None,
        smoothed_direction_score=-0.1 if trend == "down" else 0.1,
        trend_state=trend,
        state_age_bars=0 if turn else 1,
        retrace_atr=2.5,
        turn_event=turn,
        horizons=horizons,
    )


class _Sensor:
    warmup_bars = 97

    def __init__(self, *snapshots: DirectionalImpulseSnapshot) -> None:
        self.snapshots = list(snapshots)

    def update(self, **_kwargs: object) -> DirectionalImpulseSnapshot:
        return self.snapshots.pop(0)


def _pair(
    minute: int,
    *,
    cl: float,
    mcl: float,
    contract: str = "202608",
) -> MclAuctionBar:
    ts = START + timedelta(minutes=minute)

    def bar(close: float) -> OhlcvBar:
        return OhlcvBar(ts, close, close, close, close, 1.0)

    return MclAuctionBar(contract, bar(cl), bar(mcl))


def test_v18_identity_and_turn_policy_are_frozen() -> None:
    assert MCL_TWO_SPEED_AUCTION_VERSION == "mcl.two-speed-auction-relay.v18"
    assert MCL_TWO_SPEED_AUCTION_AUTHORITY == "signal_state_only_no_orders_no_capital"
    assert MCL_TWO_SPEED_AUCTION_HORIZONS == (6, 12, 24, 48, 96)
    assert MCL_TWO_SPEED_AUCTION_POLICY.as_payload() == {
        "session_mode": "window",
        "smooth_alpha": 0.15,
        "initial_score": 0.075,
        "turn_score": 0.06,
        "retrace_atr": 2.0,
        "min_state_bars": 24,
        "cooldown_bars": 24,
        "min_observed_horizons": 3,
        "bar_duration_seconds": 300.0,
        "start_et": "00:00",
        "end_et": "23:59",
    }


@pytest.mark.parametrize(
    ("raw_ticks", "basis_ticks", "breadth", "expected"),
    [
        (4, 1, 5, -1),
        (2, -1, 5, -1),
        (2, 0, 0, 1),
        (2, 1, 3, 1),
        (2, 1, 2, -1),
    ],
)
def test_v18_router_preserves_large_lag_equal_and_lead_law(
    raw_ticks: int,
    basis_ticks: int,
    breadth: int,
    expected: int,
) -> None:
    assert route_mcl_v18_direction(
        1,
        raw_parity_ticks=raw_ticks,
        velocity_breadth=breadth,
        basis_velocity_ticks=basis_ticks,
    ) == expected


def test_raw_turn_reduces_risk_then_matures_exactly_one_bar_later() -> None:
    engine = MclTwoSpeedAuctionEngine()
    engine._sensor = _Sensor(  # type: ignore[assignment]
        _snapshot(trend="down"),
        _snapshot(turn="down", trend="down"),
        _snapshot(
            trend="down",
            velocities=(0.1, 0.1, 0.1, 0.1, 0.1),
        ),
    )

    state = engine.update(_pair(0, cl=80.0, mcl=80.0))
    raw = engine.update(_pair(5, cl=79.95, mcl=79.94))
    mature = engine.update(_pair(10, cl=79.94, mcl=79.93))

    assert state.phase == "STATE"
    assert raw.phase == "RAW_TURN"
    assert raw.risk_reduction is True
    assert raw.raw_direction == raw.proposed_direction == -1
    assert raw.velocity_breadth == 3
    assert raw.raw_parity_ticks == 6
    assert raw.basis_velocity_ticks == 1
    assert mature.phase == "MATURATION"
    assert mature.signal_at_utc == raw.observed_at_utc
    assert mature.retained is True
    assert mature.parity_aligned is True
    assert mature.velocity_aligned is False
    assert mature.route == "failed_auction"
    assert mature.admitted_direction == 1
    assert mature.as_payload()["submitted_orders"] == 0


def test_raw_turn_without_parity_still_reduces_risk_but_cannot_mature() -> None:
    engine = MclTwoSpeedAuctionEngine()
    engine._sensor = _Sensor(  # type: ignore[assignment]
        _snapshot(trend="down"),
        _snapshot(turn="down", trend="down"),
        _snapshot(trend="down"),
    )

    engine.update(_pair(0, cl=80.0, mcl=80.0))
    raw = engine.update(_pair(5, cl=79.95, mcl=80.01))
    following = engine.update(_pair(10, cl=79.94, mcl=80.00))

    assert raw.phase == "RAW_TURN"
    assert raw.risk_reduction is True
    assert raw.parity_aligned is False
    assert raw.proposed_direction is None
    assert following.phase == "STATE"
    assert following.admitted_direction is None


def test_contract_change_discards_unmatured_turn() -> None:
    engine = MclTwoSpeedAuctionEngine()
    engine._sensor = _Sensor(  # type: ignore[assignment]
        _snapshot(trend="up"),
        _snapshot(turn="up", trend="up", velocities=(0.1,) * 5),
        _snapshot(trend="up"),
    )

    engine.update(_pair(0, cl=80.0, mcl=80.0))
    raw = engine.update(_pair(5, cl=80.02, mcl=80.02))
    reset = engine.update(
        _pair(10, cl=81.0, mcl=81.0, contract="202609")
    )

    assert raw.proposed_direction == 1
    assert reset.phase == "STATE"
    assert reset.contract_reset is True
    assert reset.cl_move == reset.mcl_move == 0.0
    assert reset.admitted_direction is None


def test_bar_contract_rejects_ambiguous_or_nonincreasing_time() -> None:
    ts = START.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        MclAuctionBar(
            "202608",
            OhlcvBar(ts, 1, 1, 1, 1, 1),
            OhlcvBar(ts, 1, 1, 1, 1, 1),
        )

    engine = MclTwoSpeedAuctionEngine()
    engine._sensor = _Sensor(  # type: ignore[assignment]
        _snapshot(trend="up"),
    )
    first = _pair(0, cl=80.0, mcl=80.0)
    engine.update(first)
    with pytest.raises(ValueError, match="must increase"):
        engine.update(first)
