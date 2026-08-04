"""Outcome-blind cross-scale morphology around exact MCL V18 turns."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import statistics

from .mcl_turn_tape import MCL_TURN_TAPE_SCHEMA
from .mcl_two_speed_auction import MclAuctionBar, MclAuctionDecision


MCL_PREDICTIVE_ONSET_VERSION = "mcl.predictive-intelligence-onset-atlas.v1"
MCL_PREDICTIVE_ONSET_AUTHORITY = (
    "count_and_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_PREDICTIVE_ONSET_FAMILIES = (
    "one_second_to_four_hour_temporal_torsion",
    "five_minute_to_four_hour_volatility_phase",
    "one_second_to_fifteen_minute_to_one_week_phase_locking",
    "one_second_dual_book_to_thirty_minute_basis_to_news_shock",
    "liquidity_elasticity_cascade",
)
_TICK_SIZE = 0.01


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("MCL onset timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("MCL onset evidence must be finite")
    return result


def _shape(value: float, *, direction: int | None = None) -> str:
    signed = float(value) * (direction if direction is not None else 1)
    return (
        "WITH_RAW_DIRECTION"
        if signed > 0.0
        else "AGAINST_RAW_DIRECTION"
        if signed < 0.0
        else "FLAT"
    )


def _plain_shape(value: float) -> str:
    return "UP" if value > 0.0 else "DOWN" if value < 0.0 else "FLAT"


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


@dataclass(frozen=True)
class MclWeeklyPrior:
    """Strictly prior finalized five-session context; never a direction owner."""

    as_of_utc: datetime
    return_pct: float
    return_velocity_pct: float
    tr_velocity_pct: float
    state_age_sessions: int

    def __post_init__(self) -> None:
        _utc(self.as_of_utc)
        for value in (
            self.return_pct,
            self.return_velocity_pct,
            self.tr_velocity_pct,
        ):
            _finite(value)
        if self.state_age_sessions < 1:
            raise ValueError("MCL weekly state age must be positive")

    def as_payload(self, *, raw_direction: int) -> dict[str, object]:
        return {
            "as_of_utc": _utc(self.as_of_utc).isoformat(),
            "return_pct": float(self.return_pct),
            "return_velocity_pct": float(self.return_velocity_pct),
            "tr_velocity_pct": float(self.tr_velocity_pct),
            "state_age_sessions": int(self.state_age_sessions),
            "return_shape": _shape(self.return_pct, direction=raw_direction),
            "return_velocity_shape": _shape(
                self.return_velocity_pct,
                direction=raw_direction,
            ),
            "tr_velocity_shape": _plain_shape(self.tr_velocity_pct),
        }


@dataclass(frozen=True)
class MclOnsetNewsContext:
    """Latest causal news state; direction remains owned by V18."""

    published_at_utc: datetime
    horizon_hours: float
    signed_pressure: float
    pressure_delta: float
    pressure_velocity_per_hour: float
    impact: float
    confidence: float

    def __post_init__(self) -> None:
        _utc(self.published_at_utc)
        for value in (
            self.horizon_hours,
            self.signed_pressure,
            self.pressure_delta,
            self.pressure_velocity_per_hour,
            self.impact,
            self.confidence,
        ):
            _finite(value)
        if self.horizon_hours <= 0.0:
            raise ValueError("MCL news horizon must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("MCL news confidence is invalid")

    def as_payload(
        self,
        *,
        treatment_at_utc: datetime,
        raw_direction: int,
    ) -> dict[str, object]:
        treatment = _utc(treatment_at_utc)
        published = _utc(self.published_at_utc)
        age_hours = (treatment - published).total_seconds() / 3600.0
        if age_hours < 0.0:
            raise ValueError("MCL onset news is from the future")
        fresh = age_hours <= self.horizon_hours
        return {
            "published_at_utc": published.isoformat(),
            "age_hours": age_hours,
            "horizon_hours": float(self.horizon_hours),
            "fresh": fresh,
            "signed_pressure": float(self.signed_pressure),
            "pressure_delta": float(self.pressure_delta),
            "pressure_velocity_per_hour": float(self.pressure_velocity_per_hour),
            "impact": float(self.impact),
            "confidence": float(self.confidence),
            "pressure_shape": _shape(
                self.signed_pressure,
                direction=raw_direction,
            ),
            "delta_shape": _shape(self.pressure_delta, direction=raw_direction),
            "velocity_shape": _shape(
                self.pressure_velocity_per_hour,
                direction=raw_direction,
            ),
        }


def _horizon(decision: MclAuctionDecision, bars: int):
    return next(
        (row for row in decision.snapshot.horizons if row.bars == bars),
        None,
    )


def _slope(values: Sequence[float], bars: int, end: int = 0) -> float:
    stop = len(values) + int(end)
    start = stop - int(bars) - 1
    if start < 0 or stop <= 0:
        raise ValueError("MCL onset bar context is underwarmed")
    anchor = float(values[start])
    if anchor <= 0.0:
        raise ValueError("MCL onset price anchor is invalid")
    return 100.0 * ((float(values[stop - 1]) / anchor) - 1.0) / float(bars)


def _slope_velocity(values: Sequence[float], bars: int, end: int = 0) -> float:
    return _slope(values, bars, end) - _slope(values, bars, end - 1)


def _slope_acceleration(values: Sequence[float], bars: int) -> float:
    return _slope_velocity(values, bars) - _slope_velocity(values, bars, -1)


def _linear_slope(values: Sequence[float], bars: int, end: int = 0) -> float:
    stop = len(values) + int(end)
    start = stop - int(bars) - 1
    if start < 0 or stop <= 0:
        raise ValueError("MCL onset linear context is underwarmed")
    return (float(values[stop - 1]) - float(values[start])) / float(bars)


def _linear_slope_velocity(
    values: Sequence[float],
    bars: int,
    end: int = 0,
) -> float:
    return _linear_slope(values, bars, end) - _linear_slope(
        values,
        bars,
        end - 1,
    )


def _linear_slope_acceleration(values: Sequence[float], bars: int) -> float:
    return _linear_slope_velocity(values, bars) - _linear_slope_velocity(
        values,
        bars,
        -1,
    )


def _true_range_pct(current: MclAuctionBar, previous: MclAuctionBar) -> float:
    close = float(previous.cl.close)
    if close <= 0.0:
        raise ValueError("MCL onset previous close is invalid")
    value = max(
        float(current.cl.high) - float(current.cl.low),
        abs(float(current.cl.high) - close),
        abs(float(current.cl.low) - close),
    )
    return 100.0 * value / close


def _age_class(value: int | None) -> str:
    age = int(value or 0)
    return (
        "FRESH_0_TO_6"
        if age <= 6
        else "MATURING_7_TO_24"
        if age <= 24
        else "ESTABLISHED_25_TO_48"
        if age <= 48
        else "OLD_OVER_48"
    )


def project_mcl_completed_bar_onset(
    decisions: Sequence[MclAuctionDecision],
    bars: Sequence[MclAuctionBar],
    *,
    weekly_prior: MclWeeklyPrior | None = None,
    news: MclOnsetNewsContext | None = None,
    four_hour_clock: str = "adjacent",
) -> dict[str, object]:
    """Project outcome-blind 5m/15m/30m/4h/weekly context at a V18 raw turn."""

    if len(decisions) < 3 or len(bars) < 9:
        raise ValueError("MCL onset context is underwarmed")
    raw = decisions[-1]
    if raw.phase != "RAW_TURN" or raw.raw_direction not in (-1, 1):
        raise ValueError("MCL onset context requires an exact V18 raw turn")
    direction = int(raw.raw_direction)
    if bars[-1].ts != raw.observed_at_utc:
        raise ValueError("MCL onset bar and decision clocks do not align")
    contract = raw.contract_key
    if any(bar.contract_key != contract for bar in bars[-9:]):
        raise ValueError("MCL onset context cannot cross a contract roll")
    if any(
        bars[index].ts <= bars[index - 1].ts
        for index in range(1, len(bars))
    ):
        raise ValueError("MCL onset bars must be ordered and unique")
    if any(
        decisions[index].observed_at_utc <= decisions[index - 1].observed_at_utc
        for index in range(1, len(decisions))
    ):
        raise ValueError("MCL onset decisions must be ordered and unique")

    if four_hour_clock not in {"adjacent", "finalized_sparse"}:
        raise ValueError("unsupported MCL onset four-hour clock")
    observations = [
        (decision, horizon)
        for decision in decisions
        if decision.contract_key == contract
        and (horizon := _horizon(decision, 48)) is not None
    ]
    if len(observations) < 3 or observations[-1][0] is not raw:
        raise ValueError("MCL onset four-hour context is underwarmed")
    if four_hour_clock == "adjacent" and tuple(
        decision for decision, _horizon_row in observations[-3:]
    ) != tuple(decisions[-3:]):
        raise ValueError("MCL onset adjacent four-hour context is underwarmed")
    (decision2, previous2), (decision1, previous1), (_, current) = observations[-3:]
    gaps_minutes = [
        (decision1.observed_at_utc - decision2.observed_at_utc).total_seconds()
        / 60.0,
        (raw.observed_at_utc - decision1.observed_at_utc).total_seconds() / 60.0,
    ]
    if any(value <= 0.0 for value in gaps_minutes):
        raise ValueError("MCL onset four-hour observation clock is invalid")
    if four_hour_clock == "adjacent":
        if current.slope_velocity_pct_per_bar is None:
            raise ValueError("MCL onset four-hour slope velocity is underwarmed")
        if previous1.slope_velocity_pct_per_bar is None:
            raise ValueError("MCL onset prior four-hour slope velocity is underwarmed")
        h4_slope_velocity = float(current.slope_velocity_pct_per_bar)
        h4_slope_acceleration = h4_slope_velocity - float(
            previous1.slope_velocity_pct_per_bar
        )
        h4_tr_velocity = float(current.tr_mean_pct) - float(previous1.tr_mean_pct)
        h4_tr_acceleration = h4_tr_velocity - (
            float(previous1.tr_mean_pct) - float(previous2.tr_mean_pct)
        )
        h4_units = "adjacent_completed_decision_delta"
    else:
        prior_hours, current_hours = (value / 60.0 for value in gaps_minutes)
        prior_slope_rate = (
            float(previous1.slope_pct_per_bar)
            - float(previous2.slope_pct_per_bar)
        ) / prior_hours
        h4_slope_velocity = (
            float(current.slope_pct_per_bar)
            - float(previous1.slope_pct_per_bar)
        ) / current_hours
        h4_slope_acceleration = h4_slope_velocity - prior_slope_rate
        prior_tr_rate = (
            float(previous1.tr_mean_pct) - float(previous2.tr_mean_pct)
        ) / prior_hours
        h4_tr_velocity = (
            float(current.tr_mean_pct) - float(previous1.tr_mean_pct)
        ) / current_hours
        h4_tr_acceleration = h4_tr_velocity - prior_tr_rate
        h4_units = "finalized_observation_delta_per_actual_elapsed_hour"

    closes = [float(bar.cl.close) for bar in bars]
    basis = [
        (float(bar.mcl.close) - float(bar.cl.close)) / _TICK_SIZE
        for bar in bars
    ]
    tr_current = _true_range_pct(bars[-1], bars[-2])
    tr_previous = _true_range_pct(bars[-2], bars[-3])
    five_tr_velocity = tr_current - tr_previous
    five_range = max(
        float(bars[-1].cl.high) - float(bars[-1].cl.low),
        abs(float(bars[-1].cl.high) - float(bars[-2].cl.close)),
        abs(float(bars[-1].cl.low) - float(bars[-2].cl.close)),
    )
    five_path_efficiency = (
        direction
        * (float(bars[-1].cl.close) - float(bars[-1].cl.open))
        / five_range
        if five_range > 0.0
        else 0.0
    )
    fifteen_slope_velocity = _slope_velocity(closes, 3)
    fifteen_slope_acceleration = _slope_acceleration(closes, 3)
    basis_slope_velocity = _linear_slope_velocity(basis, 6)
    basis_slope_acceleration = _linear_slope_acceleration(basis, 6)
    incumbent_state_age = decisions[-2].snapshot.state_age_bars

    weekly_payload = None
    if weekly_prior is not None:
        if _utc(weekly_prior.as_of_utc) >= raw.observed_at_utc:
            raise ValueError("MCL weekly prior is not strictly causal")
        weekly_payload = weekly_prior.as_payload(raw_direction=direction)
    news_payload = (
        news.as_payload(
            treatment_at_utc=raw.observed_at_utc,
            raw_direction=direction,
        )
        if news is not None
        else None
    )

    fast_tr = _plain_shape(five_tr_velocity)
    slow_tr = _plain_shape(h4_tr_velocity)
    volatility_phase = (
        "JOINT_EXPANSION"
        if fast_tr == slow_tr == "UP"
        else "LOCAL_IGNITION_FROM_SLOW_COMPRESSION"
        if fast_tr == "UP" and slow_tr != "UP"
        else "LOCAL_EXHAUSTION_INSIDE_SLOW_EXPANSION"
        if fast_tr != "UP" and slow_tr == "UP"
        else "JOINT_COMPRESSION"
    )
    payload: dict[str, object] = {
        "strategy_version": MCL_PREDICTIVE_ONSET_VERSION,
        "authority": MCL_PREDICTIVE_ONSET_AUTHORITY,
        "observed_at_utc": raw.observed_at_utc.isoformat(),
        "contract_key": contract,
        "raw_direction": direction,
        "v18": {
            "phase": raw.phase,
            "proposed_direction": raw.proposed_direction,
            "risk_reduction": raw.risk_reduction,
            "raw_state_age_bars": raw.snapshot.state_age_bars,
            "incumbent_state_age_bars": incumbent_state_age,
            "trend_state": raw.snapshot.trend_state,
        },
        "completed_bar_features": {
            "five_minute_tr_velocity_pct": five_tr_velocity,
            "five_minute_directional_path_efficiency": five_path_efficiency,
            "fifteen_minute_slope_velocity_pct_per_bar": fifteen_slope_velocity,
            "fifteen_minute_slope_acceleration_pct_per_bar": (
                fifteen_slope_acceleration
            ),
            "four_hour_slope_velocity_pct_per_bar": h4_slope_velocity,
            "four_hour_slope_acceleration_pct_per_bar": h4_slope_acceleration,
            "four_hour_tr_velocity_pct": h4_tr_velocity,
            "four_hour_tr_acceleration_pct": h4_tr_acceleration,
            "four_hour_clock": four_hour_clock,
            "four_hour_measure_units": h4_units,
            "four_hour_observation_gaps_minutes": gaps_minutes,
            "thirty_minute_basis_slope_velocity_ticks_per_bar": (
                basis_slope_velocity
            ),
            "thirty_minute_basis_slope_acceleration_ticks_per_bar": (
                basis_slope_acceleration
            ),
        },
        "weekly_prior": weekly_payload,
        "news": news_payload,
        "bar_family_shapes": {
            "one_second_to_four_hour_temporal_torsion_anchor": [
                _shape(h4_slope_velocity, direction=direction),
                _shape(h4_slope_acceleration, direction=direction),
                _plain_shape(h4_tr_velocity),
                _plain_shape(h4_tr_acceleration),
            ],
            "five_minute_to_four_hour_volatility_phase": {
                "phase": volatility_phase,
                "five_minute_tr": fast_tr,
                "four_hour_tr": slow_tr,
                "four_hour_tr_acceleration": _plain_shape(h4_tr_acceleration),
                "five_minute_path": _shape(five_path_efficiency),
            },
            "fifteen_minute_to_one_week_phase_anchor": (
                {
                    "fifteen_minute_velocity": _shape(
                        fifteen_slope_velocity,
                        direction=direction,
                    ),
                    "fifteen_minute_acceleration": _shape(
                        fifteen_slope_acceleration,
                        direction=direction,
                    ),
                    "weekly_return": weekly_payload["return_shape"],
                    "weekly_velocity": weekly_payload["return_velocity_shape"],
                    "weekly_tr": weekly_payload["tr_velocity_shape"],
                }
                if weekly_payload is not None
                else None
            ),
            "thirty_minute_basis_to_news_anchor": {
                "basis_velocity": _shape(
                    basis_slope_velocity,
                    direction=direction,
                ),
                "basis_acceleration": _shape(
                    basis_slope_acceleration,
                    direction=direction,
                ),
                "news": (
                    {
                        "fresh": news_payload["fresh"],
                        "pressure": news_payload["pressure_shape"],
                        "delta": news_payload["delta_shape"],
                        "velocity": news_payload["velocity_shape"],
                    }
                    if news_payload is not None
                    else None
                ),
            },
            "liquidity_elasticity_slow_anchor": {
                "five_minute_path": _shape(five_path_efficiency),
                "four_hour_slope_acceleration": _shape(
                    h4_slope_acceleration,
                    direction=direction,
                ),
                "trend_age": _age_class(incumbent_state_age),
            },
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
    return payload


def _record_time(record: Mapping[str, object]) -> datetime:
    value = datetime.fromisoformat(str(record["bucket_start_utc"]))
    return _utc(value)


def _verify_record(record: Mapping[str, object]) -> None:
    if record.get("schema") != MCL_TURN_TAPE_SCHEMA:
        raise ValueError("MCL onset event schema drifted")
    if not bool(record.get("valid_evidence")):
        raise ValueError("MCL onset event prefix contains invalid evidence")
    content = dict(record)
    record_id = str(content.pop("record_id", ""))
    if hashlib.sha256(_canonical(content)).hexdigest() != record_id:
        raise ValueError("MCL onset event record hash drifted")


def _rate(points: Sequence[tuple[datetime, float]]) -> float | None:
    if len(points) < 2:
        return None
    elapsed = (points[-1][0] - points[0][0]).total_seconds()
    return (points[-1][1] - points[0][1]) / elapsed if elapsed > 0.0 else None


def _window_event_features(
    records: Sequence[Mapping[str, object]],
    *,
    start: datetime,
    end: datetime,
    raw_direction: int,
) -> dict[str, object]:
    selected = [record for record in records if start <= _record_time(record) < end]
    midpoint = start + ((end - start) / 2)
    books: dict[str, dict[str, object]] = {}
    for symbol in ("CL", "MCL"):
        points: list[tuple[datetime, float]] = []
        ranges: list[tuple[datetime, float]] = []
        quote_counts: list[tuple[datetime, float]] = []
        spreads: list[float] = []
        pressure = signed_prints = 0.0
        for record in selected:
            stamp = _record_time(record)
            book = record["books"][symbol]
            summary = book["summary"]
            micro = summary.get("microprice_ohlc")
            if isinstance(micro, list) and len(micro) == 4:
                points.append((stamp, float(micro[-1]) / _TICK_SIZE))
                ranges.append((stamp, (float(micro[1]) - float(micro[2])) / _TICK_SIZE))
            quote_counts.append((stamp, float(summary["bid_ask_events"])))
            spread = summary.get("spread_ticks_min_max_last")
            if isinstance(spread, list) and len(spread) == 3:
                spreads.append(float(spread[-1]))
            size = summary["same_price_size_proxy"]
            pressure += (
                float(size["bid_add"])
                + float(size["ask_remove"])
                - float(size["ask_add"])
                - float(size["bid_remove"])
            )
            signed_prints += float(summary["signed_trade_volume_proxy"])
        early_points = [value for value in points if value[0] < midpoint]
        late_points = [value for value in points if value[0] >= midpoint]
        early_ranges = [value for stamp, value in ranges if stamp < midpoint]
        late_ranges = [value for stamp, value in ranges if stamp >= midpoint]
        early_quotes = [value for stamp, value in quote_counts if stamp < midpoint]
        late_quotes = [value for stamp, value in quote_counts if stamp >= midpoint]
        early_rate = _rate(early_points)
        late_rate = _rate(late_points)
        slope_velocity = (
            late_rate - early_rate
            if early_rate is not None and late_rate is not None
            else None
        )
        tr_velocity = (
            statistics.fmean(late_ranges) - statistics.fmean(early_ranges)
            if early_ranges and late_ranges
            else None
        )
        quote_acceleration = (
            statistics.fmean(late_quotes) - statistics.fmean(early_quotes)
            if early_quotes and late_quotes
            else None
        )
        displacement = points[-1][1] - points[0][1] if len(points) >= 2 else None
        books[symbol] = {
            "active_seconds": len(points),
            "directional_microprice_displacement_ticks": (
                raw_direction * displacement if displacement is not None else None
            ),
            "directional_microprice_slope_velocity_ticks_per_second2": (
                raw_direction * slope_velocity
                if slope_velocity is not None
                else None
            ),
            "microprice_tr_velocity_ticks": tr_velocity,
            "quote_intensity_acceleration": quote_acceleration,
            "spread_last_minus_first_ticks": (
                spreads[-1] - spreads[0] if len(spreads) >= 2 else None
            ),
            "maximum_spread_ticks": max(spreads) if spreads else None,
            "directional_same_price_size_pressure_proxy": raw_direction * pressure,
            "directional_signed_trade_volume_proxy": raw_direction * signed_prints,
        }
    leaders = Counter(
        str(record["cross_book"]["first_mid_move_leader"])
        for record in selected
        if record["cross_book"].get("first_mid_move_leader") is not None
    )
    leads = [
        int(record["cross_book"]["mcl_minus_cl_first_mid_move_us"])
        for record in selected
        if record["cross_book"].get("mcl_minus_cl_first_mid_move_us") is not None
    ]
    bases = [
        record["cross_book"].get("basis_ticks_ohlc")
        for record in selected
    ]
    bases = [value for value in bases if isinstance(value, list) and len(value) == 4]
    return {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "active_seconds": len(selected),
        "books": books,
        "first_mid_move_leaders": dict(leaders),
        "median_mcl_minus_cl_first_mid_move_us": (
            statistics.median(leads) if leads else None
        ),
        "basis_open_close_ticks": (
            [float(bases[0][0]), float(bases[-1][-1])] if bases else None
        ),
    }


def project_mcl_event_onset(
    records: Sequence[Mapping[str, object]],
    *,
    raw_turn_at_utc: datetime,
    raw_direction: int,
    prefix_start_utc: datetime,
    prefix_end_utc: datetime,
) -> dict[str, object]:
    """Project exact prospective event morphology around one complete raw turn."""

    if raw_direction not in (-1, 1):
        raise ValueError("MCL onset raw direction must be -1 or 1")
    turn = _utc(raw_turn_at_utc)
    required_start = turn - timedelta(seconds=60)
    required_end = turn + timedelta(minutes=5)
    if _utc(prefix_start_utc) > required_start or _utc(prefix_end_utc) < required_end:
        raise ValueError("MCL onset event window is incomplete")
    if not records:
        raise ValueError("MCL onset event window is empty")
    for record in records:
        _verify_record(record)
    ordered = sorted(records, key=_record_time)
    if list(records) != ordered or any(
        _record_time(ordered[index]) <= _record_time(ordered[index - 1])
        for index in range(1, len(ordered))
    ):
        raise ValueError("MCL onset event records must be ordered and unique")
    generations = {str(record["generation_sha256"]) for record in ordered}
    if len(generations) != 1:
        raise ValueError("MCL onset event generation drifted")

    windows = {
        "pre_turn_60s": _window_event_features(
            ordered,
            start=required_start,
            end=turn,
            raw_direction=raw_direction,
        ),
        "turn_response_60s": _window_event_features(
            ordered,
            start=turn,
            end=turn + timedelta(seconds=60),
            raw_direction=raw_direction,
        ),
        "maturation_4m": _window_event_features(
            ordered,
            start=turn + timedelta(seconds=60),
            end=required_end,
            raw_direction=raw_direction,
        ),
    }
    ignition_intervals = (
        (
            "closing_baseline_60_30s",
            turn - timedelta(seconds=60),
            turn - timedelta(seconds=30),
        ),
        (
            "closing_acceleration_30_15s",
            turn - timedelta(seconds=30),
            turn - timedelta(seconds=15),
        ),
        (
            "closing_commitment_15_5s",
            turn - timedelta(seconds=15),
            turn - timedelta(seconds=5),
        ),
        (
            "closing_trigger_5_0s",
            turn - timedelta(seconds=5),
            turn,
        ),
        ("spark_0_5s", turn, turn + timedelta(seconds=5)),
        (
            "acceptance_5_15s",
            turn + timedelta(seconds=5),
            turn + timedelta(seconds=15),
        ),
        (
            "propagation_15_30s",
            turn + timedelta(seconds=15),
            turn + timedelta(seconds=30),
        ),
        (
            "persistence_30_60s",
            turn + timedelta(seconds=30),
            turn + timedelta(seconds=60),
        ),
    )
    for name, start, end in ignition_intervals:
        windows[name] = _window_event_features(
            ordered,
            start=start,
            end=end,
            raw_direction=raw_direction,
        )
    response = windows["turn_response_60s"]
    maturation = windows["maturation_4m"]
    pre = windows["pre_turn_60s"]
    event_shapes: dict[str, object] = {}
    for symbol in ("CL", "MCL"):
        pre_book = pre["books"][symbol]
        response_book = response["books"][symbol]
        mature_book = maturation["books"][symbol]
        event_shapes[symbol] = {
            "response_slope_velocity": (
                _shape(float(response_book["directional_microprice_slope_velocity_ticks_per_second2"]))
                if response_book["directional_microprice_slope_velocity_ticks_per_second2"]
                is not None
                else None
            ),
            "maturation_slope_velocity": (
                _shape(float(mature_book["directional_microprice_slope_velocity_ticks_per_second2"]))
                if mature_book["directional_microprice_slope_velocity_ticks_per_second2"]
                is not None
                else None
            ),
            "response_tr_velocity": (
                _plain_shape(float(response_book["microprice_tr_velocity_ticks"]))
                if response_book["microprice_tr_velocity_ticks"] is not None
                else None
            ),
            "maturation_tr_velocity": (
                _plain_shape(float(mature_book["microprice_tr_velocity_ticks"]))
                if mature_book["microprice_tr_velocity_ticks"] is not None
                else None
            ),
            "response_displacement": (
                _shape(float(response_book["directional_microprice_displacement_ticks"]))
                if response_book["directional_microprice_displacement_ticks"]
                is not None
                else None
            ),
            "spread_response": (
                _plain_shape(float(response_book["spread_last_minus_first_ticks"]))
                if response_book["spread_last_minus_first_ticks"] is not None
                else None
            ),
            "pressure_persistence": _plain_shape(
                float(response_book["directional_same_price_size_pressure_proxy"])
                + float(mature_book["directional_same_price_size_pressure_proxy"])
                - float(pre_book["directional_same_price_size_pressure_proxy"])
            ),
        }

    pre_basis = pre["basis_open_close_ticks"]
    response_basis = response["basis_open_close_ticks"]
    mature_basis = maturation["basis_open_close_ticks"]
    basis_shape = None
    if pre_basis and response_basis and mature_basis:
        anchor = float(pre_basis[-1])
        response_distance = abs(float(response_basis[-1]) - anchor)
        mature_distance = abs(float(mature_basis[-1]) - anchor)
        basis_shape = (
            "RECONVERGED"
            if mature_distance < response_distance
            else "DIVERGED"
            if mature_distance > response_distance
            else "UNCHANGED"
        )

    ignition_shapes = {}
    for name, _start, _end in ignition_intervals:
        window = windows[name]
        ignition_shapes[name] = {
            "books": {
                symbol: {
                    "displacement": (
                        _shape(float(book["directional_microprice_displacement_ticks"]))
                        if book["directional_microprice_displacement_ticks"]
                        is not None
                        else None
                    ),
                    "slope_velocity": (
                        _shape(
                            float(
                                book[
                                    "directional_microprice_slope_velocity_ticks_per_second2"
                                ]
                            )
                        )
                        if book[
                            "directional_microprice_slope_velocity_ticks_per_second2"
                        ]
                        is not None
                        else None
                    ),
                    "tr_velocity": (
                        _plain_shape(float(book["microprice_tr_velocity_ticks"]))
                        if book["microprice_tr_velocity_ticks"] is not None
                        else None
                    ),
                    "quote_intensity_acceleration": (
                        _plain_shape(float(book["quote_intensity_acceleration"]))
                        if book["quote_intensity_acceleration"] is not None
                        else None
                    ),
                    "spread_elasticity": (
                        _plain_shape(float(book["spread_last_minus_first_ticks"]))
                        if book["spread_last_minus_first_ticks"] is not None
                        else None
                    ),
                    "top_size_pressure": _shape(
                        float(book["directional_same_price_size_pressure_proxy"])
                    ),
                    "signed_prints": _shape(
                        float(book["directional_signed_trade_volume_proxy"])
                    ),
                }
                for symbol, book in window["books"].items()
            },
            "first_mid_move_leaders": window["first_mid_move_leaders"],
            "median_mcl_minus_cl_first_mid_move_us": window[
                "median_mcl_minus_cl_first_mid_move_us"
            ],
        }
    return {
        "strategy_version": MCL_PREDICTIVE_ONSET_VERSION,
        "authority": MCL_PREDICTIVE_ONSET_AUTHORITY,
        "raw_turn_at_utc": turn.isoformat(),
        "raw_direction": raw_direction,
        "generation_sha256": next(iter(generations)),
        "windows": windows,
        "event_shapes": {
            "books": event_shapes,
            "response_leaders": response["first_mid_move_leaders"],
            "maturation_leaders": maturation["first_mid_move_leaders"],
            "basis_response_to_maturation": basis_shape,
            "velocity_ignition_ladder": ignition_shapes,
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }


def combine_mcl_predictive_onset_atlas(
    completed_bar: Mapping[str, object],
    event: Mapping[str, object],
) -> dict[str, object]:
    """Join the two causal clocks without scoring, selecting, or acting."""

    if completed_bar.get("authority") != MCL_PREDICTIVE_ONSET_AUTHORITY:
        raise ValueError("MCL completed-bar onset authority drifted")
    if event.get("authority") != MCL_PREDICTIVE_ONSET_AUTHORITY:
        raise ValueError("MCL event onset authority drifted")
    if completed_bar.get("observed_at_utc") != event.get("raw_turn_at_utc"):
        raise ValueError("MCL onset atlas clocks do not align")
    if completed_bar.get("raw_direction") != event.get("raw_direction"):
        raise ValueError("MCL onset atlas directions do not align")
    bar_shapes = completed_bar["bar_family_shapes"]
    event_shapes = event["event_shapes"]
    books = event_shapes["books"]
    news_anchor = bar_shapes["thirty_minute_basis_to_news_anchor"]
    return {
        "schema": MCL_PREDICTIVE_ONSET_VERSION,
        "authority": MCL_PREDICTIVE_ONSET_AUTHORITY,
        "observed_at_utc": completed_bar["observed_at_utc"],
        "raw_direction": completed_bar["raw_direction"],
        "families": {
            "one_second_to_four_hour_temporal_torsion": {
                "CL_fast": books["CL"],
                "MCL_fast": books["MCL"],
                "four_hour_anchor": bar_shapes[
                    "one_second_to_four_hour_temporal_torsion_anchor"
                ],
            },
            "five_minute_to_four_hour_volatility_phase": bar_shapes[
                "five_minute_to_four_hour_volatility_phase"
            ],
            "one_second_to_fifteen_minute_to_one_week_phase_locking": {
                "fast": {
                    "response_leaders": event_shapes["response_leaders"],
                    "maturation_leaders": event_shapes["maturation_leaders"],
                    "CL": books["CL"],
                    "MCL": books["MCL"],
                },
                "meso_slow": bar_shapes["fifteen_minute_to_one_week_phase_anchor"],
            },
            "one_second_dual_book_to_thirty_minute_basis_to_news_shock": {
                "fast_basis": event_shapes["basis_response_to_maturation"],
                "meso_news": news_anchor,
            },
            "liquidity_elasticity_cascade": {
                "fast": books,
                "meso_slow": bar_shapes["liquidity_elasticity_slow_anchor"],
            },
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
