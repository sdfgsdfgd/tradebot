"""Pure causal seconds-to-multiscale evidence for prospective MCL shocks."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import median

from ..chart_data.series import OhlcvBar
from ..news.contract import (
    observe_news_signal,
    publication_id,
    select_news_snapshot_at,
)
from ..time_utils import ET_ZONE
from .mcl_shock_crest import (
    MclShockBookEvidence,
    MclShockDecision,
    MclShockObservation,
)


_TICK_SIZE = 0.01


@dataclass(frozen=True, slots=True)
class MclShockPoint:
    observed_at_utc: datetime
    open: float
    close: float
    trade_volume: float
    signed_flow: float


def _utc(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("MCL shock timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _identity(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sign(value: float) -> int | None:
    return 1 if value > 0.0 else -1 if value < 0.0 else None


def _session(value: datetime) -> date | None:
    local = _utc(value).astimezone(ET_ZONE)
    clock = local.time().replace(tzinfo=None)
    if 17 <= clock.hour < 18:
        return None
    return (local + timedelta(days=1)).date() if clock.hour >= 18 else local.date()


def _summary(row: Mapping[str, object], symbol: str) -> Mapping[str, object]:
    books = row.get("books")
    if not isinstance(books, Mapping) or not isinstance(books.get(symbol), Mapping):
        raise ValueError("MCL shock tape has no book")
    summary = books[symbol].get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("MCL shock tape has no book summary")
    return summary


def _point(row: Mapping[str, object], symbol: str) -> MclShockPoint | None:
    summary = _summary(row, symbol)
    micro = summary.get("microprice_ohlc")
    if not isinstance(micro, list) or len(micro) != 4:
        return None
    values = [float(item) for item in micro]
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        return None
    return MclShockPoint(
        observed_at_utc=row["_time"],
        open=values[0],
        close=values[3],
        trade_volume=float(summary.get("trade_volume") or 0.0),
        signed_flow=float(summary.get("signed_trade_volume_proxy") or 0.0),
    )


def _interval(
    points: deque[MclShockPoint], *, end: datetime, seconds: int
) -> dict[str, float] | None:
    start = end - timedelta(seconds=seconds)
    selected = [point for point in points if point.observed_at_utc > start]
    if len(selected) < max(2, seconds // 4):
        return None
    elapsed = max(
        1.0,
        (selected[-1].observed_at_utc - selected[0].observed_at_utc).total_seconds(),
    )
    return {
        "velocity": (selected[-1].close - selected[0].open) / _TICK_SIZE / elapsed,
        "trade_volume": sum(point.trade_volume for point in selected),
        "signed_flow": sum(point.signed_flow for point in selected),
    }


def _slope(values: Sequence[float], bars: int, end: int = 0) -> float:
    stop = len(values) + end
    start = stop - bars - 1
    if start < 0 or stop <= 0 or values[start] <= 0.0:
        raise ValueError("MCL shock slope is underwarmed")
    return 100.0 * ((values[stop - 1] / values[start]) - 1.0) / bars


def _slow_context(
    maps: Mapping[str, Mapping[datetime, OhlcvBar]],
    stamps: Mapping[str, Sequence[datetime]],
    *,
    symbol: str,
    when: datetime,
    price: float,
) -> tuple[float, float, float] | None:
    minute = when.replace(second=0, microsecond=0)
    values = stamps[symbol]
    index = bisect_right(values, minute)
    closes = [
        float(maps[symbol][stamp].close)
        for stamp in values[max(0, index - 18):index]
    ]
    closes.append(float(price))
    if len(closes) < 18:
        return None
    current = _slope(closes, 15)
    previous = _slope(closes, 15, -1)
    velocity = current - previous
    return current, velocity, velocity - (previous - _slope(closes, 15, -2))


def build_mcl_shock_observations(
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    *,
    contract_key: str,
    eligible_start: datetime,
) -> list[tuple[dict[str, object], MclShockObservation]]:
    stamps = {symbol: sorted(bars[symbol]) for symbol in ("CL", "MCL")}
    points: dict[str, deque[MclShockPoint]] = {
        symbol: deque() for symbol in ("CL", "MCL")
    }
    prior_five: dict[str, float | None] = {"CL": None, "MCL": None}
    active_minute: datetime | None = None
    minute_volume = 0.0
    output = []
    for raw in rows:
        row = dict(raw)
        when = row["_time"]
        minute = when.replace(second=0, microsecond=0)
        if minute != active_minute:
            active_minute = minute
            minute_volume = 0.0
        minute_volume += float(_summary(row, "MCL").get("trade_volume") or 0.0)
        current = {symbol: _point(row, symbol) for symbol in ("CL", "MCL")}
        for symbol, point in current.items():
            if point is not None:
                points[symbol].append(point)
            cutoff = when - timedelta(seconds=60)
            while points[symbol] and points[symbol][0].observed_at_utc <= cutoff:
                points[symbol].popleft()
        clocks = {
            symbol: {
                seconds: _interval(points[symbol], end=when, seconds=seconds)
                for seconds in (5, 15, 60)
            }
            for symbol in ("CL", "MCL")
        }
        if any(
            clocks[symbol][seconds] is None
            for symbol in clocks
            for seconds in clocks[symbol]
        ) or any(point is None for point in current.values()):
            continue
        slow = {
            symbol: _slow_context(
                bars,
                stamps,
                symbol=symbol,
                when=when,
                price=current[symbol].close,
            )
            for symbol in ("CL", "MCL")
        }
        if any(value is None for value in slow.values()):
            continue
        close_time = minute + timedelta(minutes=1)
        mcl_index = bisect_right(
            stamps["MCL"], close_time - timedelta(microseconds=1)
        )
        prior = [
            float(bars["MCL"][stamp].volume)
            for stamp in stamps["MCL"][max(0, mcl_index - 10):mcl_index]
        ]
        baseline = median(prior) if prior else 0.0
        if baseline <= 0.0:
            continue
        evidence = {}
        for symbol in ("CL", "MCL"):
            five_volume = float(clocks[symbol][5]["trade_volume"])
            volume_velocity = (
                five_volume - float(prior_five[symbol])
                if prior_five[symbol] is not None
                else 0.0
            )
            prior_five[symbol] = five_volume
            evidence[symbol] = MclShockBookEvidence(
                velocity_5s=float(clocks[symbol][5]["velocity"]),
                velocity_15s=float(clocks[symbol][15]["velocity"]),
                velocity_60s=float(clocks[symbol][60]["velocity"]),
                slope_15m=float(slow[symbol][0]),
                velocity_15m=float(slow[symbol][1]),
                acceleration_15m=float(slow[symbol][2]),
                signed_flow_15s=float(clocks[symbol][15]["signed_flow"]),
                volume_velocity_5s=volume_velocity,
            )
        if when < _utc(eligible_start):
            continue
        output.append(
            (
                row,
                MclShockObservation(
                    observed_at_utc=when,
                    contract_key=contract_key,
                    mcl_microprice=float(current["MCL"].close),
                    volume_multiple=minute_volume / baseline,
                    cl=evidence["CL"],
                    mcl=evidence["MCL"],
                    spread_eligible=True,
                    fresh_top=row.get("market_data_types") == {"CL": 1, "MCL": 1},
                ),
            )
        )
    return output


def _true_range(bar: OhlcvBar, previous_close: float) -> float:
    return max(
        float(bar.high) - float(bar.low),
        abs(float(bar.high) - previous_close),
        abs(float(bar.low) - previous_close),
    )


def project_mcl_shock_cross_scale(
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    *,
    when: datetime,
    direction: int | None,
) -> dict[str, object]:
    minute = when.replace(second=0, microsecond=0)
    common = sorted(set(bars["CL"]) & set(bars["MCL"]))
    selected = common[:bisect_right(common, minute)]
    books = {}
    for symbol in ("CL", "MCL"):
        values = [float(bars[symbol][stamp].close) for stamp in selected]
        slope = velocity = acceleration = None
        if len(values) >= 243:
            slope = _slope(values, 240)
            previous = _slope(values, 240, -1)
            velocity = slope - previous
            acceleration = velocity - (previous - _slope(values, 240, -2))
        ranges = [
            _true_range(bars[symbol][right], float(bars[symbol][left].close))
            for left, right in zip(selected, selected[1:])
        ]
        tr_velocity = tr_acceleration = None
        if len(ranges) >= 720:
            current = median(ranges[-240:])
            previous = median(ranges[-480:-240])
            older = median(ranges[-720:-480])
            if previous > 0.0 and older > 0.0:
                tr_velocity = current / previous - 1.0
                tr_acceleration = tr_velocity - (previous / older - 1.0)
        books[symbol] = {
            "rolling_240m_slope_pct_per_bar": slope,
            "rolling_240m_slope_velocity_pct_per_bar": velocity,
            "rolling_240m_slope_acceleration_pct_per_bar": acceleration,
            "rolling_240m_tr_velocity": tr_velocity,
            "rolling_240m_tr_acceleration": tr_acceleration,
            "slope_aligned": (
                _sign(float(slope)) == direction
                if slope is not None and direction in (-1, 1)
                else None
            ),
        }
    basis = None
    if len(selected) >= 32:
        cl = [float(bars["CL"][stamp].close) for stamp in selected]
        mcl = [float(bars["MCL"][stamp].close) for stamp in selected]
        current = 100.0 * ((mcl[-1] / mcl[-31]) - (cl[-1] / cl[-31]))
        previous = 100.0 * ((mcl[-2] / mcl[-32]) - (cl[-2] / cl[-32]))
        basis = {
            "mcl_minus_cl_30m_pct": current,
            "velocity_pct": current - previous,
        }
    current_session = _session(when)
    grouped: dict[date, list[datetime]] = defaultdict(list)
    for stamp in selected:
        session = _session(stamp)
        if session is not None and session != current_session:
            grouped[session].append(stamp)
    prior_sessions = sorted(grouped)[-5:]
    weekly = None
    if len(prior_sessions) == 5:
        rows = [stamp for session in prior_sessions for stamp in grouped[session]]
        first, last = rows[0], rows[-1]
        start = float(bars["MCL"][first].open)
        ranges = [
            _true_range(bars["MCL"][right], float(bars["MCL"][left].close))
            for left, right in zip(rows, rows[1:])
            if right - left == timedelta(minutes=1)
        ]
        weekly = {
            "sessions": [value.isoformat() for value in prior_sessions],
            "as_of_utc": last.isoformat(),
            "return_pct": (
                100.0 * (float(bars["MCL"][last].close) / start - 1.0)
                if start > 0.0
                else None
            ),
            "median_minute_tr": median(ranges) if ranges else None,
        }
    return {
        "clock": "completed_minute_rolling_240m_and_prior_five_sessions",
        "books": books,
        "basis": basis,
        "weekly_prior": weekly,
    }


def project_mcl_shock_news(
    snapshots: Sequence[Mapping[str, object]], *, when: datetime
) -> dict[str, object] | None:
    selected = select_news_snapshot_at(snapshots, as_of=when)
    if selected is None:
        return None
    current = observe_news_signal(selected, symbol="MCL", as_of=when)
    index = max(position for position, row in enumerate(snapshots) if row == selected)
    previous_snapshot = select_news_snapshot_at(
        tuple(row for position, row in enumerate(snapshots) if position != index),
        as_of=when,
    )
    previous = (
        observe_news_signal(previous_snapshot, symbol="MCL", as_of=when)
        if previous_snapshot is not None
        else None
    )
    current_at = _utc(current.snapshot_as_of_utc)
    pressure = current.direction * current.impact / 100.0
    prior_pressure = previous.direction * previous.impact / 100.0 if previous else 0.0
    delta = pressure - prior_pressure if previous is not None else 0.0
    elapsed = (
        (current_at - _utc(previous.snapshot_as_of_utc)).total_seconds() / 3600.0
        if previous is not None
        else 0.0
    )
    return {
        **current.as_payload(),
        "publication_id": selected.get("publication_id") or publication_id(selected),
        "prior_publication_id": (
            previous_snapshot.get("publication_id")
            or publication_id(previous_snapshot)
            if previous_snapshot is not None
            else None
        ),
        "signed_pressure": pressure,
        "pressure_delta": delta,
        "pressure_velocity_per_hour": delta / elapsed if elapsed > 0.0 else 0.0,
        "treatment_age_hours": (when - current_at).total_seconds() / 3600.0,
        "authority": "non_scoring_context_only",
    }


def _observation_payload(value: MclShockObservation) -> dict[str, object]:
    def book(item: MclShockBookEvidence) -> dict[str, float]:
        return {
            name: float(getattr(item, name)) for name in item.__dataclass_fields__
        }

    return {
        "observed_at_utc": _utc(value.observed_at_utc).isoformat(),
        "contract_key": value.contract_key,
        "mcl_microprice": float(value.mcl_microprice),
        "volume_multiple": float(value.volume_multiple),
        "cl": book(value.cl),
        "mcl": book(value.mcl),
        "spread_eligible": bool(value.spread_eligible),
        "fresh_top": bool(value.fresh_top),
    }


def project_mcl_shock_transition(
    decision: MclShockDecision,
    observation: MclShockObservation,
    *,
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "decision": decision.as_payload(),
        "observation": _observation_payload(observation),
        "cross_scale": project_mcl_shock_cross_scale(
            bars,
            when=observation.observed_at_utc,
            direction=decision.shock_direction,
        ),
        "news": project_mcl_shock_news(news, when=observation.observed_at_utc),
    }


def project_mcl_shock_bar_prefix(
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    rows = []
    for stamp in sorted(set(bars["CL"]) & set(bars["MCL"])):
        if start <= stamp <= end:
            def payload(bar: OhlcvBar) -> dict[str, float]:
                return {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }

            rows.append(
                {
                    "ts": stamp.isoformat(),
                    "CL": payload(bars["CL"][stamp]),
                    "MCL": payload(bars["MCL"][stamp]),
                }
            )
    return {
        "common_rows": len(rows),
        "first_common_close_utc": rows[0]["ts"] if rows else None,
        "last_common_close_utc": rows[-1]["ts"] if rows else None,
        "sha256": _identity(rows),
    }


def audit_mcl_shock_volume_clock(
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, object]:
    tape: dict[datetime, float] = defaultdict(float)
    for row in rows:
        when = row["_time"]
        if start <= when < end:
            close = when.replace(second=0, microsecond=0) + timedelta(minutes=1)
            tape[close] += float(_summary(row, "MCL").get("trade_volume") or 0.0)
    common = sorted(
        stamp for stamp in tape if stamp in bars["MCL"] and stamp <= end
    )
    differences = [
        tape[stamp] - float(bars["MCL"][stamp].volume) for stamp in common
    ]
    ratios = [
        tape[stamp] / float(bars["MCL"][stamp].volume)
        for stamp in common
        if float(bars["MCL"][stamp].volume) > 0.0
    ]
    return {
        "semantics": "IB_TCP_packet_receipt_seconds_vs_finalized_broker_minutes",
        "common_completed_minutes": len(common),
        "exact_minutes": sum(abs(value) <= 1e-9 for value in differences),
        "mismatched_minutes": sum(abs(value) > 1e-9 for value in differences),
        "tape_volume_total": sum(tape[stamp] for stamp in common),
        "broker_volume_total": sum(
            float(bars["MCL"][stamp].volume) for stamp in common
        ),
        "median_positive_volume_ratio": median(ratios) if ratios else None,
        "maximum_absolute_minute_difference": max(
            (abs(value) for value in differences), default=0.0
        ),
    }
