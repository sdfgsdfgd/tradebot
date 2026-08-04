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
    MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD,
    MCL_TWO_SPEED_AUCTION_VERSION,
    MclAuctionBar,
    MclAuctionDecision,
    MclAuctionMinute,
    MclTwoSpeedAuctionEngine,
    MclTwoSpeedAuctionLifecycle,
    route_mcl_v18_direction,
)
from tradebot.spot.champions import discover_current_champions


START = datetime(2026, 8, 4, 4, 0, tzinfo=timezone.utc)


def test_stage112_executable_research_crown_is_machine_bound_but_cannot_trade() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = discover_current_champions(root=root, symbols=("MCL",), tracks=("HF",))

    assert len(refs) == 1
    ref = refs[0]
    declaration = json.loads(ref.declaration_path.read_text())
    artifact = json.loads(ref.artifact_path.read_text())

    assert ref.version == "112"
    assert ref.strategy_key == "mcl-two-speed-shock-arbiter-v112"
    assert hashlib.sha256(ref.artifact_path.read_bytes()).hexdigest() == declaration[
        "artifact_sha256"
    ]
    assert declaration["promotion"]["eligible"] is True
    assert declaration["promotion"]["order_authority"] == "none"
    assert artifact["selection"] == "EXECUTABLE_CROWN"
    assert artifact["strategy_version"] == "mcl.two-speed-shock-arbiter.v112"
    assert artifact["economics"]["trades"] == 388
    assert artifact["economics"]["primary_net_usd"] == pytest.approx(3942.99)
    assert artifact["economics"]["primary_intrabar_drawdown_usd"] == pytest.approx(
        603.63
    )
    assert artifact["economics"]["primary_pnl_to_intrabar_drawdown"] == pytest.approx(
        6.532130609810673
    )
    assert artifact["cadence_exception"]["authorized_by_user"] is True
    assert artifact["invariants"]["limit_only"] is True
    assert artifact["invariants"]["market_orders_allowed"] is False
    leaderboard = (root / "backtests/mcl/leaderboard.md").read_text()
    assert "CR-004 · 2026-08-04 · Shock-Aware Two-Speed Arbiter — Stage 112" in leaderboard
    assert artifact["schema"] in leaderboard
    assert declaration["artifact_sha256"] in leaderboard


def test_v18_lifecycle_parity_receipt_binds_every_frozen_trade() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "backtests/mcl/mcl_two_speed_auction_v18_lifecycle_owner_parity.json").read_text()
    )

    assert receipt["authority"].endswith("no_orders_no_capital")
    assert receipt["manifest_sha256"] == (
        "1085ce89502cc1c511d1887d137d30c9c37e54ded23e761afbdf8c46a2736d55"
    )
    assert receipt["tape"]["rows"] == 707_136
    assert receipt["expected_trades"] == receipt["actual_trades"] == 338
    assert receipt["expected_sha256"] == receipt["actual_sha256"] == (
        "788a5c3ff577b05b8d0f25b1794ab4b7f5e5d47795b48fbee5c3184577fac2a5"
    )
    assert receipt["first_mismatch"] is None
    assert receipt["exact_trade_parity"] is True
    assert receipt["submitted_orders"] == 0


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


def _minute(
    minute: int,
    *,
    cl: tuple[float, float, float, float] = (80.0, 80.0, 80.0, 80.0),
    mcl: tuple[float, float, float, float] = (80.0, 80.0, 80.0, 80.0),
    contract: str = "202608",
) -> MclAuctionMinute:
    ts = START + timedelta(minutes=minute)

    def bar(values: tuple[float, float, float, float]) -> OhlcvBar:
        return OhlcvBar(ts, *values, 1.0)

    return MclAuctionMinute(contract, bar(cl), bar(mcl))


def _decision(
    minute: int,
    phase: str,
    *,
    raw: int | None = None,
    admitted: int | None = None,
    route: str | None = None,
    signal_minute: int | None = None,
) -> MclAuctionDecision:
    return MclAuctionDecision(
        observed_at_utc=START + timedelta(minutes=minute),
        contract_key="202608",
        phase=phase,  # type: ignore[arg-type]
        signal_at_utc=(
            START + timedelta(minutes=signal_minute)
            if signal_minute is not None
            else None
        ),
        raw_direction=raw,
        proposed_direction=raw,
        admitted_direction=admitted,
        route=route,  # type: ignore[arg-type]
        risk_reduction=phase == "RAW_TURN",
        contract_reset=False,
        cl_move=0.01,
        mcl_move=0.01,
        velocity_aligned=True,
        velocity_breadth=5,
        parity_aligned=True,
        retained=True if phase == "MATURATION" else None,
        raw_parity_ticks=1,
        basis_velocity_ticks=0,
        snapshot=_snapshot(trend="up"),
    )


class _DecisionEngine:
    def __init__(self, *decisions: MclAuctionDecision) -> None:
        self.decisions = list(decisions)

    def update(self, _bar: MclAuctionBar) -> MclAuctionDecision:
        return self.decisions.pop(0)


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


def test_v18_lifecycle_enters_next_minute_and_raw_turn_flattens_unconditionally(
) -> None:
    lifecycle = MclTwoSpeedAuctionLifecycle(
        _DecisionEngine(  # type: ignore[arg-type]
            _decision(5, "STATE"),
            _decision(10, "MATURATION", raw=1, admitted=1, route="continuation", signal_minute=5),
            _decision(15, "RAW_TURN", raw=-1, signal_minute=15),
        )
    )
    for minute in range(1, 17):
        mcl = (
            (80.0, 80.4, 79.9, 80.2)
            if minute == 11
            else (80.1, 80.3, 79.8, 80.0)
            if minute == 16
            else (80.0, 80.2, 79.9, 80.0)
        )
        step = lifecycle.update(_minute(minute, mcl=mcl))

    assert step.closed_trades == lifecycle.trades
    assert lifecycle.position is None
    assert len(lifecycle.trades) == 1
    trade = lifecycle.trades[0]
    assert trade.route == "continuation"
    assert trade.signal_at_utc == START + timedelta(minutes=5)
    assert trade.entry_at_utc == START + timedelta(minutes=11)
    assert trade.exit_at_utc == START + timedelta(minutes=16)
    assert trade.entry_price == 80.0
    assert trade.exit_price == 80.1
    assert trade.exit_reason == "raw_turn_invalidation"
    assert trade.raw_pnl_usd == pytest.approx(10.0)
    assert trade.primary_pnl_usd == pytest.approx(
        10.0 - MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD
    )
    assert trade.mfe_usd == pytest.approx(40.0)
    assert trade.mae_usd == pytest.approx(-10.0)


def test_failed_auction_profit_memory_uses_completed_excursion_before_stop() -> None:
    lifecycle = MclTwoSpeedAuctionLifecycle(
        _DecisionEngine(  # type: ignore[arg-type]
            _decision(5, "STATE"),
            _decision(10, "RAW_TURN", raw=1, signal_minute=10),
            _decision(
                15,
                "MATURATION",
                raw=1,
                admitted=1,
                route="failed_auction",
                signal_minute=10,
            ),
        )
    )
    for minute in range(1, 18):
        mcl = (
            (100.0, 100.6, 99.9, 100.5)
            if minute == 16
            else (100.2, 100.3, 100.1, 100.2)
            if minute == 17
            else (100.0, 100.1, 99.9, 100.0)
        )
        lifecycle.update(_minute(minute, cl=(100.0,) * 4, mcl=mcl))

    assert lifecycle.position is None
    trade = lifecycle.trades[0]
    assert trade.exit_reason == "profit_memory"
    assert trade.entry_at_utc == START + timedelta(minutes=16)
    assert trade.exit_at_utc == START + timedelta(minutes=17)
    assert trade.entry_price == 100.0
    assert trade.exit_price == pytest.approx(100.15)
    assert trade.mfe_usd == pytest.approx(60.0)
    assert trade.mae_usd == pytest.approx(-10.0)


def test_contract_roll_flattens_at_last_known_close_and_clears_pending_state() -> None:
    lifecycle = MclTwoSpeedAuctionLifecycle(
        _DecisionEngine(  # type: ignore[arg-type]
            _decision(5, "STATE"),
            _decision(10, "MATURATION", raw=1, admitted=1, route="continuation", signal_minute=5),
        )
    )
    for minute in range(1, 12):
        lifecycle.update(_minute(minute, mcl=(80.0, 80.2, 79.9, 80.1)))
    rolled = lifecycle.update(
        _minute(12, contract="202609", mcl=(81.0, 81.2, 80.9, 81.1))
    )

    assert rolled.contract_reset is True
    assert rolled.opened_position is False
    assert lifecycle.position is None
    assert len(rolled.closed_trades) == 1
    assert rolled.closed_trades[0].exit_reason == "contract_roll"
    assert rolled.closed_trades[0].exit_at_utc == START + timedelta(minutes=11)
    assert rolled.closed_trades[0].exit_price == 80.1


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
