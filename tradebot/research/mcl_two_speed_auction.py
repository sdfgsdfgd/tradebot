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
MCL_TWO_SPEED_AUCTION_MULTIPLIER = 100.0
MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD = 3.52
MCL_TWO_SPEED_AUCTION_STRESS_COST_USD = 5.52
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


@dataclass(frozen=True)
class MclAuctionMinute:
    """One causal matched-contract minute used by the exact V18 lifecycle."""

    contract_key: str
    cl: OhlcvBar
    mcl: OhlcvBar

    def __post_init__(self) -> None:
        if not self.contract_key.strip():
            raise ValueError("MCL lifecycle minute requires a contract key")
        if self.cl.ts != self.mcl.ts:
            raise ValueError("CL/MCL lifecycle minute timestamps do not match")
        if self.cl.ts.tzinfo is None:
            raise ValueError("MCL lifecycle minute timestamps must be timezone-aware")

    @property
    def ts(self) -> datetime:
        return self.cl.ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class MclAuctionTrade:
    """One completed costed V18 trade; broker execution remains downstream."""

    route: Literal["continuation", "failed_auction"]
    direction: int
    signal_at_utc: datetime
    entry_at_utc: datetime
    exit_at_utc: datetime
    entry_price: float
    exit_price: float
    exit_reason: str
    raw_pnl_usd: float
    primary_pnl_usd: float
    stress_pnl_usd: float
    mfe_usd: float
    mae_usd: float

    def as_payload(self) -> dict[str, object]:
        return {
            "branch": "acceptance" if self.route == "continuation" else "failure",
            "direction": "up" if self.direction > 0 else "down",
            "signal_time": self.signal_at_utc.astimezone(timezone.utc).isoformat(),
            "entry_time": self.entry_at_utc.astimezone(timezone.utc).isoformat(),
            "exit_time": self.exit_at_utc.astimezone(timezone.utc).isoformat(),
            "entry": float(self.entry_price),
            "exit": float(self.exit_price),
            "reason": self.exit_reason,
            "raw_pnl": float(self.raw_pnl_usd),
            "primary_pnl": float(self.primary_pnl_usd),
            "stress_pnl": float(self.stress_pnl_usd),
            "mfe": float(self.mfe_usd),
            "mae": float(self.mae_usd),
        }


@dataclass(frozen=True)
class MclAuctionLifecycleStep:
    """One minute transition from the pure, non-submitting V18 lifecycle."""

    observed_at_utc: datetime
    decision: MclAuctionDecision | None
    opened_position: bool
    closed_trades: tuple[MclAuctionTrade, ...]
    contract_reset: bool


@dataclass
class _MclAuctionPosition:
    direction: int
    signal_at_utc: datetime
    entry_at_utc: datetime
    entry_price: float
    route: Literal["continuation", "failed_auction"]
    mfe_usd: float = 0.0
    mae_usd: float = 0.0


def _aggregate_mcl_minutes(rows: list[MclAuctionMinute]) -> MclAuctionBar:
    if len(rows) != 5 or any(
        right.ts - left.ts != timedelta(minutes=1)
        for left, right in zip(rows, rows[1:])
    ):
        raise ValueError("MCL lifecycle requires five consecutive minutes")

    def aggregate(side: str) -> OhlcvBar:
        values = [getattr(row, side) for row in rows]
        return OhlcvBar(
            rows[-1].ts,
            float(values[0].open),
            max(float(value.high) for value in values),
            min(float(value.low) for value in values),
            float(values[-1].close),
            sum(float(value.volume) for value in values),
        )

    return MclAuctionBar(rows[-1].contract_key, aggregate("cl"), aggregate("mcl"))


class MclTwoSpeedAuctionLifecycle:
    """Own V18 next-minute entry, raw-turn flatten, profit memory, and roll."""

    def __init__(self, engine: MclTwoSpeedAuctionEngine | None = None) -> None:
        self._engine = engine or MclTwoSpeedAuctionEngine()
        self._minutes: list[MclAuctionMinute] = []
        self._previous: MclAuctionMinute | None = None
        self._position: _MclAuctionPosition | None = None
        self._pending_flatten = False
        self._pending_direction: int | None = None
        self._pending_signal_at_utc: datetime | None = None
        self._pending_route: Literal["continuation", "failed_auction"] | None = None
        self._trades: list[MclAuctionTrade] = []

    @property
    def trades(self) -> tuple[MclAuctionTrade, ...]:
        return tuple(self._trades)

    @property
    def position(self) -> dict[str, object] | None:
        value = self._position
        if value is None:
            return None
        return {
            "direction": value.direction,
            "signal_at_utc": value.signal_at_utc.isoformat(),
            "entry_at_utc": value.entry_at_utc.isoformat(),
            "entry_price": value.entry_price,
            "route": value.route,
            "mfe_usd": value.mfe_usd,
            "mae_usd": value.mae_usd,
        }

    def _close(
        self,
        *,
        observed_at: datetime,
        price: float,
        reason: str,
    ) -> MclAuctionTrade | None:
        position = self._position
        if position is None:
            return None
        raw = (
            position.direction
            * (float(price) - position.entry_price)
            * MCL_TWO_SPEED_AUCTION_MULTIPLIER
        )
        trade = MclAuctionTrade(
            route=position.route,
            direction=position.direction,
            signal_at_utc=position.signal_at_utc,
            entry_at_utc=position.entry_at_utc,
            exit_at_utc=observed_at,
            entry_price=position.entry_price,
            exit_price=float(price),
            exit_reason=reason,
            raw_pnl_usd=raw,
            primary_pnl_usd=raw - MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD,
            stress_pnl_usd=raw - MCL_TWO_SPEED_AUCTION_STRESS_COST_USD,
            mfe_usd=position.mfe_usd,
            mae_usd=position.mae_usd,
        )
        self._position = None
        self._trades.append(trade)
        return trade

    def _apply_pending(
        self,
        minute: MclAuctionMinute,
    ) -> tuple[bool, list[MclAuctionTrade]]:
        closed = []
        if self._pending_flatten:
            trade = self._close(
                observed_at=minute.ts,
                price=float(minute.mcl.open),
                reason="raw_turn_invalidation",
            )
            if trade is not None:
                closed.append(trade)
            self._pending_flatten = False
        direction = self._pending_direction
        signal_at = self._pending_signal_at_utc
        route = self._pending_route
        self._pending_direction = None
        self._pending_signal_at_utc = None
        self._pending_route = None
        if direction is None or signal_at is None or route is None:
            return False, closed
        if self._position is not None and self._position.direction == direction:
            return False, closed
        if self._position is not None:
            trade = self._close(
                observed_at=minute.ts,
                price=float(minute.mcl.open),
                reason="opposite_turn",
            )
            if trade is not None:
                closed.append(trade)
        self._position = _MclAuctionPosition(
            direction=direction,
            signal_at_utc=signal_at,
            entry_at_utc=minute.ts,
            entry_price=float(minute.mcl.open),
            route=route,
        )
        return True, closed

    def _mark(self, minute: MclAuctionMinute) -> MclAuctionTrade | None:
        position = self._position
        if position is None:
            return None
        if position.route == "failed_auction":
            activation = (
                position.entry_price * 0.005 * MCL_TWO_SPEED_AUCTION_MULTIPLIER
            )
            if position.mfe_usd >= activation:
                protected = 0.25 * position.mfe_usd
                stop = (
                    position.entry_price
                    + position.direction
                    * protected
                    / MCL_TWO_SPEED_AUCTION_MULTIPLIER
                )
                price = (
                    min(float(minute.mcl.open), stop)
                    if position.direction > 0 and float(minute.mcl.low) <= stop
                    else max(float(minute.mcl.open), stop)
                    if position.direction < 0 and float(minute.mcl.high) >= stop
                    else None
                )
                if price is not None:
                    return self._close(
                        observed_at=minute.ts,
                        price=price,
                        reason="profit_memory",
                    )
        if position.direction > 0:
            position.mfe_usd = max(
                position.mfe_usd,
                (float(minute.mcl.high) - position.entry_price)
                * MCL_TWO_SPEED_AUCTION_MULTIPLIER,
            )
            position.mae_usd = min(
                position.mae_usd,
                (float(minute.mcl.low) - position.entry_price)
                * MCL_TWO_SPEED_AUCTION_MULTIPLIER,
            )
        else:
            position.mfe_usd = max(
                position.mfe_usd,
                (position.entry_price - float(minute.mcl.low))
                * MCL_TWO_SPEED_AUCTION_MULTIPLIER,
            )
            position.mae_usd = min(
                position.mae_usd,
                (position.entry_price - float(minute.mcl.high))
                * MCL_TWO_SPEED_AUCTION_MULTIPLIER,
            )
        return None

    def _bind_decision(self, decision: MclAuctionDecision) -> None:
        if decision.phase == "RAW_TURN":
            self._pending_flatten = self._position is not None
            self._pending_direction = None
            self._pending_signal_at_utc = None
            self._pending_route = None
        elif (
            decision.phase == "MATURATION"
            and decision.admitted_direction in (-1, 1)
            and decision.signal_at_utc is not None
            and decision.route is not None
        ):
            self._pending_direction = decision.admitted_direction
            self._pending_signal_at_utc = decision.signal_at_utc
            self._pending_route = decision.route

    def update(self, minute: MclAuctionMinute) -> MclAuctionLifecycleStep:
        previous = self._previous
        if previous is not None and minute.ts <= previous.ts:
            raise ValueError("MCL lifecycle minute timestamps must increase")
        contract_reset = previous is not None and (
            minute.contract_key != previous.contract_key
        )
        closed = []
        if contract_reset:
            trade = self._close(
                observed_at=previous.ts,
                price=float(previous.mcl.close),
                reason="contract_roll",
            )
            if trade is not None:
                closed.append(trade)
            self._pending_flatten = False
            self._pending_direction = None
            self._pending_signal_at_utc = None
            self._pending_route = None
            self._minutes.clear()
        elif previous is None or minute.ts - previous.ts != timedelta(minutes=1):
            self._minutes.clear()

        opened, pending_closed = self._apply_pending(minute)
        closed.extend(pending_closed)
        marked = self._mark(minute)
        if marked is not None:
            closed.append(marked)

        self._minutes.append(minute)
        decision = None
        if minute.ts.minute % 5 == 0:
            if len(self._minutes) == 5:
                decision = self._engine.update(_aggregate_mcl_minutes(self._minutes))
                self._bind_decision(decision)
            self._minutes.clear()
        self._previous = minute
        return MclAuctionLifecycleStep(
            observed_at_utc=minute.ts,
            decision=decision,
            opened_position=opened,
            closed_trades=tuple(closed),
            contract_reset=contract_reset,
        )

    def finish(self) -> MclAuctionTrade | None:
        """Close only a finite historical replay; a live owner never calls this."""

        previous = self._previous
        if previous is None:
            return None
        return self._close(
            observed_at=previous.ts,
            price=float(previous.mcl.close),
            reason="dataset_end",
        )
