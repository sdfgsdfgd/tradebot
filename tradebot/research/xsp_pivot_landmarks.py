"""Future-aware XSP pivot landmarks for offline evaluation only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import math
import statistics

from ..backtest.models import Bar
from .xsp_opening_edge_state import XspDailyBar, validate_xsp_daily_bars


XSP_PIVOT_LANDMARK_VERSION = "xsp.phase-front-pivot-landmarks.v1"
XSP_PIVOT_LANDMARK_AUTHORITY = (
    "offline_future_aware_labels_only_no_signal_no_orders_no_capital"
)
XSP_PIVOT_SCALE_SESSIONS = 21
XSP_PIVOT_CONFIRMATION_MULTIPLE = 0.25
XSP_PIVOT_SERIOUS_MULTIPLE = 0.50


@dataclass(frozen=True)
class XspPivotScale:
    trading_day: date
    scale: float
    prior_sessions: int
    first_prior_day: date
    last_prior_day: date

    def as_payload(self) -> dict[str, object]:
        return {
            "trading_day": self.trading_day.isoformat(),
            "scale": float(self.scale),
            "prior_sessions": int(self.prior_sessions),
            "first_prior_day": self.first_prior_day.isoformat(),
            "last_prior_day": self.last_prior_day.isoformat(),
        }


@dataclass(frozen=True)
class XspPivotLandmark:
    lane: str
    trading_day: date
    incoming_direction: int | None
    outgoing_direction: int | None
    extreme_at_utc: datetime | None
    extreme_price: float
    causal_confirmation_utc: datetime | None
    terminal_at_utc: datetime
    terminal_price: float
    outgoing_excursion: float
    scale: float
    serious: bool
    classes: tuple[str, ...]
    right_censored: bool = False
    flags: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, object]:
        incoming = (
            "up"
            if self.incoming_direction == 1
            else "down"
            if self.incoming_direction == -1
            else None
        )
        outgoing = (
            "up"
            if self.outgoing_direction == 1
            else "down"
            if self.outgoing_direction == -1
            else None
        )
        identity = "|".join(
            (
                XSP_PIVOT_LANDMARK_VERSION,
                self.lane,
                self.trading_day.isoformat(),
                str(incoming),
                str(outgoing),
                self.extreme_at_utc.isoformat()
                if self.extreme_at_utc is not None
                else "",
                self.causal_confirmation_utc.isoformat()
                if self.causal_confirmation_utc is not None
                else "",
                str(self.right_censored),
            )
        )
        return {
            "schema": XSP_PIVOT_LANDMARK_VERSION,
            "authority": XSP_PIVOT_LANDMARK_AUTHORITY,
            "landmark_id": hashlib.sha256(identity.encode()).hexdigest(),
            "lane": self.lane,
            "trading_date": self.trading_day.isoformat(),
            "class": self.classes[0],
            "classes": list(self.classes),
            "incoming_direction": incoming,
            "outgoing_direction": outgoing,
            "extreme_utc": (
                self.extreme_at_utc.isoformat()
                if self.extreme_at_utc is not None
                else None
            ),
            "extreme_price": float(self.extreme_price),
            "causal_confirmation_utc": (
                self.causal_confirmation_utc.isoformat()
                if self.causal_confirmation_utc is not None
                else None
            ),
            "terminal_utc": self.terminal_at_utc.isoformat(),
            "terminal_price": float(self.terminal_price),
            "outgoing_excursion": float(self.outgoing_excursion),
            "outgoing_excursion_scale": float(
                self.outgoing_excursion / self.scale
            ),
            "frozen_scale_points": float(self.scale),
            "severity": (
                "RIGHT_CENSORED"
                if self.right_censored
                else "SERIOUS"
                if self.serious
                else "RECOIL_ONLY"
            ),
            "serious": bool(self.serious),
            "right_censored": bool(self.right_censored),
            "intrabar_order_unresolved": (
                "INTRABAR_ORDER_UNRESOLVED" in self.flags
            ),
            "flags": list(self.flags),
            "outcomes": None,
            "submitted_orders": 0,
        }


@dataclass
class _Wave:
    incoming_direction: int | None
    direction: int
    origin_at: datetime | None
    origin_price: float
    confirmed_at: datetime
    confirmation_index: int
    excursion: float = 0.0
    terminal_at: datetime | None = None
    terminal_price: float | None = None
    opening: bool = False
    flags: tuple[str, ...] = ()

    def observe_price(self, observed_at: datetime, price: float) -> None:
        excursion = self.direction * (float(price) - self.origin_price)
        if self.terminal_at is None or excursion > self.excursion:
            self.excursion = max(0.0, float(excursion))
            self.terminal_at = observed_at
            self.terminal_price = float(price)

    def observe_close(self, bar: Bar) -> None:
        self.observe_price(bar.ts, float(bar.close))

    def observe_bar(self, bar: Bar) -> None:
        price = float(bar.high) if self.direction > 0 else float(bar.low)
        self.observe_price(bar.ts, price)


def xsp_prior_tr_scales(
    daily_bars: Sequence[XspDailyBar],
    *,
    sessions: int = XSP_PIVOT_SCALE_SESSIONS,
) -> dict[date, XspPivotScale]:
    """Freeze each day's scale from prior complete Wilder true ranges."""

    rows = validate_xsp_daily_bars(daily_bars, minimum_sessions=0)
    if sessions < 1:
        raise ValueError("XSP pivot scale sessions must be positive")
    true_ranges: list[float | None] = [None]
    for previous, current in zip(rows, rows[1:]):
        true_ranges.append(
            max(
                float(current.high) - float(current.low),
                abs(float(current.high) - float(previous.close)),
                abs(float(current.low) - float(previous.close)),
            )
        )
    output: dict[date, XspPivotScale] = {}
    for index, row in enumerate(rows):
        start = index - sessions
        if start < 1:
            continue
        values = true_ranges[start:index]
        if len(values) != sessions or any(value is None for value in values):
            continue
        scale = statistics.median(float(value) for value in values if value is not None)
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        output[row.day] = XspPivotScale(
            trading_day=row.day,
            scale=float(scale),
            prior_sessions=sessions,
            first_prior_day=rows[start].day,
            last_prior_day=rows[index - 1].day,
        )
    return output


def _validated_bars(bars: Sequence[Bar]) -> tuple[Bar, ...]:
    rows = tuple(bars)
    for index, row in enumerate(rows):
        values = (row.open, row.high, row.low, row.close)
        if (
            not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
            or float(row.low) > min(float(row.open), float(row.close))
            or float(row.high) < max(float(row.open), float(row.close))
        ):
            raise ValueError("XSP pivot landmark tape has malformed OHLC")
        if index and row.ts <= rows[index - 1].ts:
            raise ValueError("XSP pivot landmark tape must be ordered and unique")
    return rows


def _pivot_classes(
    *,
    lane: str,
    wave: _Wave,
    serious: bool,
    regime: str,
    opening_gap_direction: int,
    prior_recoil_direction: int | None,
) -> tuple[str, ...]:
    if not serious:
        return ("RECOIL_ONLY",)
    if wave.opening:
        classes = ["OPENING_DRIVE"]
        if opening_gap_direction:
            classes.append(
                "GAP_CONTINUATION"
                if wave.direction == opening_gap_direction
                else "GAP_REVERSAL"
            )
        return tuple(classes)
    if prior_recoil_direction is not None and wave.direction == -prior_recoil_direction:
        return ("TREND_REACCELERATION",)
    if lane == "GTH":
        return ("GTH_OVERNIGHT_PIVOT",)
    if lane == "RTH" and wave.confirmation_index < 18:
        return ("OPENING_PIVOT",)
    if regime == "SHOCK":
        return ("SHOCK_HANDOFF_PIVOT",)
    if regime in {"ONE_SIDED", "ELEVATED"}:
        return ("PRESSURE_PIVOT",)
    return ("QUIET_STRUCTURAL_PIVOT",)


def label_xsp_pivot_landmarks(
    bars: Sequence[Bar],
    *,
    lane: str,
    trading_day: date,
    reference_close: float,
    reference_at_utc: datetime | None = None,
    scale: float,
    regime_by_close: Mapping[datetime, str] | None = None,
    confirmation_multiple: float = XSP_PIVOT_CONFIRMATION_MULTIPLE,
    serious_multiple: float = XSP_PIVOT_SERIOUS_MULTIPLE,
) -> tuple[XspPivotLandmark, ...]:
    """Label one lane after the entire path is known; never import this live."""

    rows = _validated_bars(bars)
    if lane not in {"RTH", "GTH", "CURB"}:
        raise ValueError("XSP pivot landmark lane is invalid")
    if not rows:
        return ()
    if (
        not math.isfinite(reference_close)
        or reference_close <= 0.0
        or not math.isfinite(scale)
        or scale <= 0.0
        or confirmation_multiple <= 0.0
        or serious_multiple <= confirmation_multiple
    ):
        raise ValueError("XSP pivot landmark scale law is invalid")

    confirmation = float(scale) * float(confirmation_multiple)
    serious_distance = float(scale) * float(serious_multiple)
    gap = float(rows[0].open) - float(reference_close)
    gap_direction = 0 if abs(gap) < confirmation else (1 if gap > 0.0 else -1)
    regime_by_close = regime_by_close or {}
    direction = 0
    extreme = float(reference_close)
    extreme_at: datetime | None = None
    wave: _Wave | None = None
    landmarks: list[XspPivotLandmark] = []
    unresolved = False
    prior_recoil_direction: int | None = None

    def settle(current: _Wave) -> None:
        nonlocal prior_recoil_direction
        if current.terminal_at is None or current.terminal_price is None:
            raise RuntimeError("XSP pivot wave has no causal outgoing path")
        serious = current.excursion >= serious_distance
        regime = str(regime_by_close.get(current.confirmed_at, "ORDINARY"))
        classes = _pivot_classes(
            lane=lane,
            wave=current,
            serious=serious,
            regime=regime,
            opening_gap_direction=gap_direction,
            prior_recoil_direction=prior_recoil_direction,
        )
        landmarks.append(
            XspPivotLandmark(
                lane=lane,
                trading_day=trading_day,
                incoming_direction=current.incoming_direction,
                outgoing_direction=current.direction,
                extreme_at_utc=current.origin_at,
                extreme_price=current.origin_price,
                causal_confirmation_utc=current.confirmed_at,
                terminal_at_utc=current.terminal_at,
                terminal_price=current.terminal_price,
                outgoing_excursion=float(current.excursion),
                scale=float(scale),
                serious=serious,
                classes=classes,
                flags=current.flags,
            )
        )
        prior_recoil_direction = None if serious else current.direction

    for index, bar in enumerate(rows):
        if direction == 0:
            up_intrabar = float(bar.high) >= reference_close + confirmation
            down_intrabar = float(bar.low) <= reference_close - confirmation
            unresolved = unresolved or (up_intrabar and down_intrabar)
            close_move = float(bar.close) - reference_close
            if abs(close_move) < confirmation:
                continue
            direction = 1 if close_move > 0.0 else -1
            extreme = float(bar.close)
            extreme_at = bar.ts
            wave = _Wave(
                incoming_direction=None,
                direction=direction,
                origin_at=reference_at_utc,
                origin_price=float(reference_close),
                confirmed_at=bar.ts,
                confirmation_index=index,
                opening=True,
                flags=("INTRABAR_ORDER_UNRESOLVED",) if unresolved else (),
            )
            wave.observe_close(bar)
            continue

        assert wave is not None
        prior_extreme = extreme
        if direction > 0:
            if float(bar.high) > extreme:
                extreme = float(bar.high)
                extreme_at = bar.ts
            if float(bar.close) > extreme - confirmation:
                wave.observe_bar(bar)
                continue
        else:
            if float(bar.low) < extreme:
                extreme = float(bar.low)
                extreme_at = bar.ts
            if float(bar.close) < extreme + confirmation:
                wave.observe_bar(bar)
                continue

        wave.observe_bar(bar)
        settle(wave)
        next_flags = (
            ("INTRABAR_ORDER_UNRESOLVED",)
            if extreme != prior_extreme
            else ()
        )
        direction = -direction
        wave = _Wave(
            incoming_direction=-direction,
            direction=direction,
            origin_at=extreme_at,
            origin_price=float(extreme),
            confirmed_at=bar.ts,
            confirmation_index=index,
            flags=next_flags,
        )
        extreme = float(bar.close)
        extreme_at = bar.ts
        wave.observe_close(bar)

    if wave is not None:
        wave.observe_bar(rows[-1])
        settle(wave)
        if wave.terminal_at is None or wave.terminal_price is None:
            raise RuntimeError("XSP terminal pivot wave is missing")
        landmarks.append(
            XspPivotLandmark(
                lane=lane,
                trading_day=trading_day,
                incoming_direction=wave.direction,
                outgoing_direction=-wave.direction,
                extreme_at_utc=wave.terminal_at,
                extreme_price=wave.terminal_price,
                causal_confirmation_utc=None,
                terminal_at_utc=rows[-1].ts,
                terminal_price=float(rows[-1].close),
                outgoing_excursion=0.0,
                scale=float(scale),
                serious=False,
                classes=("RIGHT_CENSORED",),
                right_censored=True,
            )
        )
    return tuple(landmarks)
