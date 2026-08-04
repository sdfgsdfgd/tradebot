"""Stage-112 V18 and minute-shock position, risk, and roll arbiter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from .mcl_minute_shock import (
    MclMinuteShockEngine,
    MclMinuteShockTransition,
    MclShockMinute,
)
from .mcl_two_speed_auction import (
    MCL_TWO_SPEED_AUCTION_MULTIPLIER,
    MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD,
    MCL_TWO_SPEED_AUCTION_STRESS_COST_USD,
    MclAuctionDecision,
    MclAuctionMinute,
    MclTwoSpeedAuctionEngine,
    _aggregate_mcl_minutes,
)


MCL_TWO_SPEED_SHOCK_VERSION = "mcl.two-speed-shock-arbiter.v112"

@dataclass(slots=True)
class _MclCombinedPosition:
    direction: int
    signal_at_utc: datetime
    entry_at_utc: datetime
    entry_price: float
    owner: Literal["v18", "shock"]
    origin_owner: Literal["v18", "shock"]
    route: str
    mfe_usd: float = 0.0
    mae_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class MclCombinedTrade:
    branch: str
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
    owner: Literal["v18", "shock"]
    origin_owner: Literal["v18", "shock"]
    route: str

    def as_payload(self) -> dict[str, object]:
        return {
            "branch": self.branch,
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
            "owner": self.owner,
            "origin_owner": self.origin_owner,
            "route": self.route,
        }


@dataclass(frozen=True, slots=True)
class MclCombinedLifecycleStep:
    observed_at_utc: datetime
    v18_decision: MclAuctionDecision | None
    shock_transition: MclMinuteShockTransition
    opened_position: bool
    owner_transfer: bool
    closed_trades: tuple[MclCombinedTrade, ...]
    contract_reset: bool
    gap_reset: bool


_MCL_ET = ZoneInfo("America/New_York")


def mcl_weekly_flat_blocked_at_open(observed_at_utc: datetime) -> bool:
    """Return whether this minute opened inside Friday's frozen flat interval."""

    opened = (observed_at_utc - timedelta(minutes=1)).astimezone(_MCL_ET)
    return opened.weekday() == 4 and (opened.hour, opened.minute) >= (16, 53)


class MclTwoSpeedShockLifecycle:
    """One Stage-112 V18/shock position owner with the live risk envelope."""

    raw_loss_cap_usd = 300.0
    weekly_flat_et = time(16, 53)

    def __init__(
        self,
        *,
        engine: MclTwoSpeedAuctionEngine | None = None,
        shock: MclMinuteShockEngine | None = None,
    ) -> None:
        self._engine = engine or MclTwoSpeedAuctionEngine()
        self._shock = shock or MclMinuteShockEngine()
        self._minutes: list[MclAuctionMinute] = []
        self._previous: MclAuctionMinute | None = None
        self._position: _MclCombinedPosition | None = None
        self._pending_raw_flatten = False
        self._pending_v18_entry: tuple[int, datetime, str] | None = None
        self._pending_raw_loss_cap = False
        self._cap_armed_at: datetime | None = None
        self._cap_trigger_raw_usd: float | None = None
        self._pending_weekly_flat = False
        self._weekly_flat_armed_at: datetime | None = None
        self._trades: list[MclCombinedTrade] = []
        self._counters: Counter[str] = Counter()
        self._cap_events: list[dict[str, object]] = []
        self._weekly_flat_events: list[dict[str, object]] = []

    @property
    def trades(self) -> tuple[MclCombinedTrade, ...]:
        return tuple(self._trades)

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    @property
    def cap_events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._cap_events)

    @property
    def weekly_flat_events(self) -> tuple[dict[str, object], ...]:
        return tuple(self._weekly_flat_events)

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
            "owner": value.owner,
            "origin_owner": value.origin_owner,
            "route": value.route,
            "mfe_usd": value.mfe_usd,
            "mae_usd": value.mae_usd,
        }

    def _close(
        self,
        minute: MclAuctionMinute,
        *,
        price: float,
        reason: str,
    ) -> MclCombinedTrade | None:
        position = self._position
        if position is None:
            return None
        raw = (
            position.direction
            * (float(price) - position.entry_price)
            * MCL_TWO_SPEED_AUCTION_MULTIPLIER
        )
        trade = MclCombinedTrade(
            branch=(
                "shock"
                if position.owner == "shock"
                else "acceptance"
                if position.route == "continuation"
                else "failure"
            ),
            direction=position.direction,
            signal_at_utc=position.signal_at_utc,
            entry_at_utc=position.entry_at_utc,
            exit_at_utc=minute.ts,
            entry_price=position.entry_price,
            exit_price=float(price),
            exit_reason=reason,
            raw_pnl_usd=raw,
            primary_pnl_usd=raw - MCL_TWO_SPEED_AUCTION_PRIMARY_COST_USD,
            stress_pnl_usd=raw - MCL_TWO_SPEED_AUCTION_STRESS_COST_USD,
            mfe_usd=position.mfe_usd,
            mae_usd=position.mae_usd,
            owner=position.owner,
            origin_owner=position.origin_owner,
            route=position.route,
        )
        self._trades.append(trade)
        self._counters[f"close:{reason}"] += 1
        if reason == "raw_loss_cap":
            self._cap_events.append(
                {
                    "owner": position.owner,
                    "direction": "up" if position.direction > 0 else "down",
                    "signal_time": position.signal_at_utc.isoformat(),
                    "entry_time": position.entry_at_utc.isoformat(),
                    "armed_at": (
                        self._cap_armed_at.isoformat()
                        if self._cap_armed_at is not None
                        else None
                    ),
                    "trigger_raw_usd": self._cap_trigger_raw_usd,
                    "exit_time": minute.ts.isoformat(),
                    "exit_price": float(price),
                    "primary_pnl": trade.primary_pnl_usd,
                    "stress_pnl": trade.stress_pnl_usd,
                }
            )
        elif reason == "weekly_closure":
            self._weekly_flat_events.append(
                {
                    "owner": position.owner,
                    "direction": "up" if position.direction > 0 else "down",
                    "signal_time": position.signal_at_utc.isoformat(),
                    "entry_time": position.entry_at_utc.isoformat(),
                    "armed_at": (
                        self._weekly_flat_armed_at.isoformat()
                        if self._weekly_flat_armed_at is not None
                        else None
                    ),
                    "exit_time": minute.ts.isoformat(),
                    "exit_price": float(price),
                    "primary_pnl": trade.primary_pnl_usd,
                    "stress_pnl": trade.stress_pnl_usd,
                }
            )
        self._position = None
        self._pending_raw_loss_cap = False
        self._cap_armed_at = None
        self._cap_trigger_raw_usd = None
        self._pending_weekly_flat = False
        self._weekly_flat_armed_at = None
        return trade

    def _open(
        self,
        minute: MclAuctionMinute,
        *,
        direction: int,
        signal_at: datetime,
        owner: Literal["v18", "shock"],
        route: str,
    ) -> None:
        self._position = _MclCombinedPosition(
            direction=direction,
            signal_at_utc=signal_at,
            entry_at_utc=minute.ts,
            entry_price=float(minute.mcl.open),
            owner=owner,
            origin_owner=owner,
            route=route,
        )
        self._counters[f"open:{owner}:{route}"] += 1

    def _shock_takeover(
        self,
        minute: MclAuctionMinute,
        *,
        direction: int,
        signal_at: datetime,
        route: str,
        closed: list[MclCombinedTrade],
    ) -> tuple[bool, bool]:
        position = self._position
        if position is not None and position.direction == direction:
            if position.owner != "shock":
                position.owner = "shock"
                position.route = route
                self._counters["same_direction_owner_transfers"] += 1
                return False, True
            return False, False
        if position is not None:
            trade = self._close(
                minute,
                price=float(minute.mcl.open),
                reason="shock_opposite_takeover",
            )
            if trade is not None:
                closed.append(trade)
            self._counters["opposite_direction_shock_flips"] += 1
        self._open(
            minute,
            direction=direction,
            signal_at=signal_at,
            owner="shock",
            route=route,
        )
        return True, False

    def _apply_pending(
        self,
        minute: MclAuctionMinute,
        shock: MclMinuteShockTransition,
    ) -> tuple[bool, bool, list[MclCombinedTrade]]:
        opened = transferred = False
        closed: list[MclCombinedTrade] = []
        shock_entry = (
            (shock.entry_direction, shock.entry_signal_at_utc)
            if shock.entry_direction in (-1, 1)
            and shock.entry_signal_at_utc is not None
            else None
        )
        weekly_block = mcl_weekly_flat_blocked_at_open(minute.ts)
        if self._pending_weekly_flat or weekly_block:
            if shock_entry is not None:
                self._counters["shock_entries_suppressed_by_weekly_flat"] += 1
            if self._pending_v18_entry is not None:
                self._counters["v18_entries_suppressed_by_weekly_flat"] += 1
            if self._position is not None:
                reason = (
                    "raw_loss_cap"
                    if self._pending_raw_loss_cap
                    else "weekly_closure"
                )
                trade = self._close(
                    minute, price=float(minute.mcl.open), reason=reason
                )
                if trade is not None:
                    closed.append(trade)
                self._counters[f"{reason}_priority_boundaries"] += 1
            else:
                self._pending_raw_loss_cap = False
                self._cap_armed_at = None
                self._cap_trigger_raw_usd = None
                self._pending_weekly_flat = False
                self._weekly_flat_armed_at = None
            self._pending_raw_flatten = False
            self._pending_v18_entry = None
            return opened, transferred, closed

        if self._pending_raw_loss_cap:
            if shock_entry is not None:
                self._counters["shock_entries_suppressed_by_raw_loss_cap"] += 1
            if self._pending_v18_entry is not None:
                self._counters["v18_entries_suppressed_by_raw_loss_cap"] += 1
            trade = self._close(
                minute, price=float(minute.mcl.open), reason="raw_loss_cap"
            )
            if trade is not None:
                closed.append(trade)
            self._pending_raw_flatten = False
            self._pending_v18_entry = None
            self._counters["raw_loss_cap_priority_boundaries"] += 1
            return opened, transferred, closed

        flattened = self._pending_raw_flatten
        if flattened:
            trade = self._close(
                minute,
                price=float(minute.mcl.open),
                reason="raw_turn_invalidation",
            )
            if trade is not None:
                closed.append(trade)
            self._counters["raw_turn_priority_boundaries"] += 1
        self._pending_raw_flatten = False

        if shock.exit_reason is not None and not flattened:
            if self._position is not None and self._position.owner == "shock":
                trade = self._close(
                    minute,
                    price=float(minute.mcl.open),
                    reason=shock.exit_reason,
                )
                if trade is not None:
                    closed.append(trade)
            else:
                self._counters["shock_release_without_owned_position"] += 1

        shock_admitted = shock_entry is not None and not flattened
        if shock_entry is not None:
            if flattened:
                self._counters["shock_entries_suppressed_by_raw_turn"] += 1
            else:
                opened, transferred = self._shock_takeover(
                    minute,
                    direction=shock_entry[0],
                    signal_at=shock_entry[1],
                    route="shock_continuation",
                    closed=closed,
                )

        v18 = self._pending_v18_entry
        self._pending_v18_entry = None
        if v18 is None or flattened or shock_admitted:
            if v18 is not None:
                self._counters["v18_entries_suppressed_by_priority"] += 1
            return opened, transferred, closed
        direction, signal_at, route = v18
        if shock.active_direction_at_open in (-1, 1):
            if direction != shock.active_direction_at_open:
                self._counters["opposite_v18_entries_vetoed_during_shock"] += 1
                return opened, transferred, closed
            acquired, owner_transfer = self._shock_takeover(
                minute,
                direction=direction,
                signal_at=signal_at,
                route="shock_reacquisition",
                closed=closed,
            )
            opened = opened or acquired
            transferred = transferred or owner_transfer
            self._counters["same_direction_v18_shock_reacquisitions"] += 1
            return opened, transferred, closed
        if self._position is not None and self._position.direction == direction:
            return opened, transferred, closed
        if self._position is not None:
            trade = self._close(
                minute,
                price=float(minute.mcl.open),
                reason="opposite_turn",
            )
            if trade is not None:
                closed.append(trade)
        self._open(
            minute,
            direction=direction,
            signal_at=signal_at,
            owner="v18",
            route=route,
        )
        return True, transferred, closed

    def _mark(self, minute: MclAuctionMinute) -> MclCombinedTrade | None:
        position = self._position
        if position is None:
            return None
        if position.owner == "v18" and position.route == "failed_auction":
            activation = (
                position.entry_price
                * 0.005
                * MCL_TWO_SPEED_AUCTION_MULTIPLIER
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
                    return self._close(minute, price=price, reason="profit_memory")
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
        raw = (
            position.direction
            * (float(minute.mcl.close) - position.entry_price)
            * MCL_TWO_SPEED_AUCTION_MULTIPLIER
        )
        if raw <= -self.raw_loss_cap_usd:
            self._pending_raw_loss_cap = True
            self._cap_armed_at = minute.ts
            self._cap_trigger_raw_usd = raw
            self._counters["raw_loss_cap_armed"] += 1
            return None
        closed = minute.ts.astimezone(_MCL_ET)
        if (
            closed.weekday() == 4
            and (closed.hour, closed.minute)
            >= (self.weekly_flat_et.hour, self.weekly_flat_et.minute)
        ):
            self._pending_weekly_flat = True
            self._weekly_flat_armed_at = minute.ts
            self._counters["weekly_closure_armed"] += 1
        return None

    def _bind_v18(self, decision: MclAuctionDecision) -> None:
        if decision.phase == "RAW_TURN":
            self._pending_raw_flatten = self._position is not None
            self._pending_v18_entry = None
        elif (
            decision.phase == "MATURATION"
            and decision.admitted_direction in (-1, 1)
            and decision.signal_at_utc is not None
            and decision.route is not None
        ):
            self._pending_v18_entry = (
                decision.admitted_direction,
                decision.signal_at_utc,
                decision.route,
            )

    def update(self, minute: MclAuctionMinute) -> MclCombinedLifecycleStep:
        previous = self._previous
        if previous is not None and minute.ts <= previous.ts:
            raise ValueError("MCL combined lifecycle timestamps must increase")
        contract_reset = previous is not None and (
            minute.contract_key != previous.contract_key
        )
        gap_reset = previous is not None and (
            minute.ts - previous.ts != timedelta(minutes=1)
        )
        closed: list[MclCombinedTrade] = []
        if contract_reset and previous is not None:
            trade = self._close(
                previous,
                price=float(previous.mcl.close),
                reason="contract_roll",
            )
            if trade is not None:
                closed.append(trade)
            self._pending_raw_flatten = False
            self._pending_v18_entry = None
            self._minutes.clear()
        elif gap_reset and previous is not None:
            if self._position is not None and self._position.owner == "shock":
                trade = self._close(
                    previous,
                    price=float(previous.mcl.close),
                    reason="data_gap",
                )
                if trade is not None:
                    closed.append(trade)
            self._minutes.clear()
        elif previous is None:
            self._minutes.clear()

        shock = self._shock.update(
            MclShockMinute(minute.contract_key, minute.cl, minute.mcl)
        )
        opened, transferred, pending_closed = self._apply_pending(minute, shock)
        closed.extend(pending_closed)
        marked = self._mark(minute)
        if marked is not None:
            closed.append(marked)

        self._minutes.append(minute)
        decision = None
        if minute.ts.minute % 5 == 0:
            if len(self._minutes) == 5:
                decision = self._engine.update(_aggregate_mcl_minutes(self._minutes))
                self._bind_v18(decision)
            self._minutes.clear()
        self._previous = minute
        return MclCombinedLifecycleStep(
            observed_at_utc=minute.ts,
            v18_decision=decision,
            shock_transition=shock,
            opened_position=opened,
            owner_transfer=transferred,
            closed_trades=tuple(closed),
            contract_reset=contract_reset,
            gap_reset=gap_reset,
        )

    def finish(self) -> MclCombinedTrade | None:
        """Close only a finite historical replay; a live owner never calls this."""

        previous = self._previous
        if previous is None:
            return None
        return self._close(
            previous,
            price=float(previous.mcl.close),
            reason="dataset_end",
        )
