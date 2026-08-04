"""Exact, product-specific V18 CL-discovery/MCL-transport signal owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Literal

from ..chart_data.series import OhlcvBar
from ..engines.directional_impulse import (
    DirectionalImpulseEngine,
    DirectionalImpulseSnapshot,
    DirectionalTurnPolicy,
)


MCL_TWO_SPEED_AUCTION_VERSION = "mcl.two-speed-auction-relay.v18"
MCL_TWO_SPEED_AUCTION_AUTHORITY = "signal_state_only_no_orders_no_capital"
MCL_TWO_SPEED_AUCTION_HORIZONS = (6, 12, 24, 48, 96)
MCL_TWO_SPEED_AUCTION_TICK_SIZE = 0.01
MCL_TWO_SPEED_AUCTION_POLICY = DirectionalTurnPolicy(
    session_mode="window",
    smooth_alpha=0.15,
    initial_score=0.075,
    turn_score=0.06,
    retrace_atr=2.0,
    min_state_bars=24,
    cooldown_bars=24,
    min_observed_horizons=3,
    bar_duration=timedelta(minutes=5),
    start_et=time(0),
    end_et=time(23, 59),
)


@dataclass(frozen=True)
class MclAuctionBar:
    """One completed, matched-contract CL/MCL five-minute observation."""

    contract_key: str
    cl: OhlcvBar
    mcl: OhlcvBar

    def __post_init__(self) -> None:
        if not self.contract_key.strip():
            raise ValueError("MCL auction bar requires a contract key")
        if self.cl.ts != self.mcl.ts:
            raise ValueError("CL/MCL auction timestamps do not match")
        if self.cl.ts.tzinfo is None:
            raise ValueError("MCL auction timestamps must be timezone-aware")

    @property
    def ts(self) -> datetime:
        return self.cl.ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class MclAuctionDecision:
    """One causal V18 state transition; execution remains a separate owner."""

    observed_at_utc: datetime
    contract_key: str
    phase: Literal["STATE", "RAW_TURN", "MATURATION"]
    signal_at_utc: datetime | None
    raw_direction: int | None
    proposed_direction: int | None
    admitted_direction: int | None
    route: Literal["continuation", "failed_auction"] | None
    risk_reduction: bool
    contract_reset: bool
    cl_move: float
    mcl_move: float
    velocity_aligned: bool | None
    velocity_breadth: int | None
    parity_aligned: bool | None
    retained: bool | None
    raw_parity_ticks: int | None
    basis_velocity_ticks: int | None
    snapshot: DirectionalImpulseSnapshot

    def as_payload(self) -> dict[str, object]:
        return {
            "strategy_version": MCL_TWO_SPEED_AUCTION_VERSION,
            "authority": MCL_TWO_SPEED_AUCTION_AUTHORITY,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "contract_key": self.contract_key,
            "phase": self.phase,
            "signal_at_utc": (
                self.signal_at_utc.isoformat()
                if self.signal_at_utc is not None
                else None
            ),
            "raw_direction": self.raw_direction,
            "proposed_direction": self.proposed_direction,
            "admitted_direction": self.admitted_direction,
            "route": self.route,
            "risk_reduction": bool(self.risk_reduction),
            "contract_reset": bool(self.contract_reset),
            "cl_move": float(self.cl_move),
            "mcl_move": float(self.mcl_move),
            "velocity_aligned": self.velocity_aligned,
            "velocity_breadth": self.velocity_breadth,
            "parity_aligned": self.parity_aligned,
            "retained": self.retained,
            "raw_parity_ticks": self.raw_parity_ticks,
            "basis_velocity_ticks": self.basis_velocity_ticks,
            "snapshot": self.snapshot.as_payload(),
            "submitted_orders": 0,
        }


@dataclass(frozen=True)
class _MaturationTarget:
    direction: int
    signal_at_utc: datetime
    raw_parity_ticks: int
    velocity_breadth: int
    basis_velocity_ticks: int


def route_mcl_v18_direction(
    direction: int,
    *,
    raw_parity_ticks: int,
    velocity_breadth: int,
    basis_velocity_ticks: int,
) -> int:
    """Apply V18's frozen lag/equality/lead-breadth routing law."""

    if direction not in (-1, 1):
        raise ValueError("MCL auction direction must be -1 or 1")
    if raw_parity_ticks < 0 or velocity_breadth < 0:
        raise ValueError("MCL auction routing evidence cannot be negative")
    follow = (
        False
        if raw_parity_ticks > 3 or basis_velocity_ticks < 0
        else True
        if basis_velocity_ticks == 0
        else velocity_breadth >= 3
    )
    return direction if follow else -direction


def _velocity_aligned(snapshot: DirectionalImpulseSnapshot, direction: int) -> bool:
    fast = next((row for row in snapshot.horizons if row.bars == 6), None)
    velocity = fast.slope_velocity_pct_per_bar if fast is not None else None
    return velocity is not None and direction * float(velocity) > 0.0


class MclTwoSpeedAuctionEngine:
    """Own exact V18 raw turns, one-bar maturation, and transport routing."""

    def __init__(self) -> None:
        self._sensor = DirectionalImpulseEngine(
            horizons=MCL_TWO_SPEED_AUCTION_HORIZONS,
            min_direction_score=0.20,
            min_coherence=0.60,
            bar_duration=timedelta(minutes=5),
            max_anchor_lag=timedelta(minutes=65),
            turn_policy=MCL_TWO_SPEED_AUCTION_POLICY,
        )
        self._previous: MclAuctionBar | None = None
        self._maturation: _MaturationTarget | None = None

    @property
    def warmup_bars(self) -> int:
        return self._sensor.warmup_bars

    def update(self, bar: MclAuctionBar) -> MclAuctionDecision:
        previous = self._previous
        if previous is not None and bar.ts <= previous.ts:
            raise ValueError("MCL auction timestamps must increase")
        contract_reset = previous is not None and (
            bar.contract_key != previous.contract_key
        )
        if contract_reset:
            self._maturation = None
            previous = None

        snapshot = self._sensor.update(
            high=float(bar.cl.high),
            low=float(bar.cl.low),
            close=float(bar.cl.close),
            session_key=bar.contract_key,
            ts=bar.ts,
        )
        cl_move = (
            float(bar.cl.close) - float(previous.cl.close)
            if previous is not None
            else 0.0
        )
        mcl_move = (
            float(bar.mcl.close) - float(previous.mcl.close)
            if previous is not None
            else 0.0
        )
        self._previous = bar

        if snapshot.turn_event is not None:
            decision = self._raw_turn(
                bar,
                snapshot,
                cl_move=cl_move,
                mcl_move=mcl_move,
                contract_reset=contract_reset,
            )
        elif self._maturation is not None:
            decision = self._mature(
                bar,
                snapshot,
                cl_move=cl_move,
                mcl_move=mcl_move,
                contract_reset=contract_reset,
            )
        else:
            decision = MclAuctionDecision(
                observed_at_utc=bar.ts,
                contract_key=bar.contract_key,
                phase="STATE",
                signal_at_utc=None,
                raw_direction=None,
                proposed_direction=None,
                admitted_direction=None,
                route=None,
                risk_reduction=False,
                contract_reset=contract_reset,
                cl_move=cl_move,
                mcl_move=mcl_move,
                velocity_aligned=None,
                velocity_breadth=None,
                parity_aligned=None,
                retained=None,
                raw_parity_ticks=None,
                basis_velocity_ticks=None,
                snapshot=snapshot,
            )
        return decision

    def _raw_turn(
        self,
        bar: MclAuctionBar,
        snapshot: DirectionalImpulseSnapshot,
        *,
        cl_move: float,
        mcl_move: float,
        contract_reset: bool,
    ) -> MclAuctionDecision:
        direction = 1 if snapshot.turn_event == "up" else -1
        velocity_aligned = _velocity_aligned(snapshot, direction)
        parity_aligned = direction * mcl_move > 0.0
        velocity_breadth = sum(
            direction * float(row.slope_velocity_pct_per_bar or 0.0) > 0.0
            for row in snapshot.horizons
        )
        candidate = velocity_aligned and parity_aligned
        raw_parity_ticks = round(abs(mcl_move) / MCL_TWO_SPEED_AUCTION_TICK_SIZE)
        basis_velocity_ticks = round(
            direction
            * (mcl_move - cl_move)
            / MCL_TWO_SPEED_AUCTION_TICK_SIZE
        )
        self._maturation = (
            _MaturationTarget(
                direction=direction,
                signal_at_utc=bar.ts,
                raw_parity_ticks=raw_parity_ticks,
                velocity_breadth=velocity_breadth,
                basis_velocity_ticks=basis_velocity_ticks,
            )
            if candidate
            else None
        )
        return MclAuctionDecision(
            observed_at_utc=bar.ts,
            contract_key=bar.contract_key,
            phase="RAW_TURN",
            signal_at_utc=bar.ts,
            raw_direction=direction,
            proposed_direction=direction if candidate else None,
            admitted_direction=None,
            route=None,
            risk_reduction=True,
            contract_reset=contract_reset,
            cl_move=cl_move,
            mcl_move=mcl_move,
            velocity_aligned=velocity_aligned,
            velocity_breadth=velocity_breadth,
            parity_aligned=parity_aligned,
            retained=None,
            raw_parity_ticks=raw_parity_ticks,
            basis_velocity_ticks=basis_velocity_ticks,
            snapshot=snapshot,
        )

    def _mature(
        self,
        bar: MclAuctionBar,
        snapshot: DirectionalImpulseSnapshot,
        *,
        cl_move: float,
        mcl_move: float,
        contract_reset: bool,
    ) -> MclAuctionDecision:
        target = self._maturation
        if target is None:
            raise RuntimeError("MCL auction maturation target disappeared")
        self._maturation = None
        direction = target.direction
        retained = snapshot.trend_state == ("up" if direction > 0 else "down")
        velocity_aligned = _velocity_aligned(snapshot, direction)
        parity_aligned = direction * mcl_move > 0.0
        routed = route_mcl_v18_direction(
            direction,
            raw_parity_ticks=target.raw_parity_ticks,
            velocity_breadth=target.velocity_breadth,
            basis_velocity_ticks=target.basis_velocity_ticks,
        )
        route = "continuation" if routed == direction else "failed_auction"
        admitted = retained and parity_aligned
        return MclAuctionDecision(
            observed_at_utc=bar.ts,
            contract_key=bar.contract_key,
            phase="MATURATION",
            signal_at_utc=target.signal_at_utc,
            raw_direction=direction,
            proposed_direction=direction,
            admitted_direction=routed if admitted else None,
            route=route,
            risk_reduction=False,
            contract_reset=contract_reset,
            cl_move=cl_move,
            mcl_move=mcl_move,
            velocity_aligned=velocity_aligned,
            velocity_breadth=target.velocity_breadth,
            parity_aligned=parity_aligned,
            retained=retained,
            raw_parity_ticks=target.raw_parity_ticks,
            basis_velocity_ticks=target.basis_velocity_ticks,
            snapshot=snapshot,
        )
