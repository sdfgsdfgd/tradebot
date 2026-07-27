"""Timestamp-causal external evidence projected beside the XSP shadow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from ..backtest.quotes import QuoteSnapshot, option_parity_observation
from ..engines.market import xsp_session_label_et, xsp_trading_date
from ..news.contract import (
    NewsError,
    observe_news_signal,
    select_news_snapshot_at,
)
from ..time_utils import UTC, to_et
from .live_calibration import calibration_fingerprint


XSP_OPTION_CONTEXT_MAX_LAG = timedelta(minutes=7)
XSP_OPTION_CHANGE_MAX_SPAN = timedelta(minutes=15)
XSP_PREOPEN_PARITY_HORIZONS_MINUTES = (120, 240, 360)
XSP_PREOPEN_BOUNDARY_MAX_LAG = timedelta(minutes=10)
XSP_PREOPEN_RESEARCH_VERSION = "xsp.preopen-option-path.v2"
XSP_PREOPEN_RESEARCH_MAX_AGE_SECONDS = 360.0


def _xsp_preopen_option_path(
    snapshots: Sequence[tuple[datetime, QuoteSnapshot]],
    *,
    decision_at: datetime,
) -> dict[str, object]:
    """Freeze the final causal GTH parity path for an RTH decision."""

    base: dict[str, object] = {
        "schema": XSP_PREOPEN_RESEARCH_VERSION,
        "source": "option_nbbo_parity",
        "authority": "observation_only",
        "anchor_policy": "option_model_consensus_only",
        "max_age_seconds": XSP_PREOPEN_RESEARCH_MAX_AGE_SECONDS,
        "horizons_minutes": XSP_PREOPEN_PARITY_HORIZONS_MINUTES,
    }
    trading_date = xsp_trading_date(decision_at)
    if xsp_session_label_et(decision_at) != "RTH" or trading_date is None:
        return {**base, "usable": False, "reasons": ("rth_decision_required",)}
    rth_open = to_et(decision_at).replace(
        hour=9,
        minute=30,
        second=0,
        microsecond=0,
    ).astimezone(UTC)
    observations = []
    for captured_at, snapshot in snapshots:
        if (
            snapshot.session != "GTH"
            or xsp_trading_date(captured_at) != trading_date
        ):
            continue
        observation = option_parity_observation(
            snapshot,
            max_age_sec=XSP_PREOPEN_RESEARCH_MAX_AGE_SECONDS,
            allow_underlying_anchor=False,
        )
        if observation["usable"]:
            observations.append((captured_at, snapshot, observation))
    endpoints = [
        row
        for row in observations
        if timedelta(0)
        <= rth_open - row[0]
        <= XSP_PREOPEN_BOUNDARY_MAX_LAG
    ]
    if not endpoints:
        return {
            **base,
            "usable": False,
            "reasons": ("no_usable_gth_boundary_observation",),
        }
    end_at, end_snapshot, end = max(endpoints, key=lambda row: row[0])
    horizons = {}
    missing = []
    for minutes in XSP_PREOPEN_PARITY_HORIZONS_MINUTES:
        target = end_at - timedelta(minutes=minutes)
        anchors = []
        for captured_at, snapshot, observation in observations:
            lag = target - captured_at
            if (
                snapshot.target_expiry != end_snapshot.target_expiry
                or not timedelta(0) <= lag <= XSP_OPTION_CONTEXT_MAX_LAG
            ):
                continue
            anchors.append((captured_at, observation))
        if not anchors:
            missing.append(f"missing_{minutes}m_anchor")
            horizons[str(minutes)] = {
                "usable": False,
                "reason": "no_causal_same_expiry_anchor",
            }
            continue
        anchor_at, anchor = max(anchors, key=lambda row: row[0])
        interval_seconds = (end_at - anchor_at).total_seconds()
        value_change = float(end["value"]) - float(anchor["value"])
        horizons[str(minutes)] = {
            "usable": True,
            "anchor_ts": anchor["ts"],
            "anchor_chain_fingerprint": anchor["chain_fingerprint"],
            "anchor_pairs": anchor["pairs"],
            "anchor_dispersion_points": anchor["dispersion_points"],
            "anchor_median_relative_spread": anchor["median_relative_spread"],
            "anchor_strikes": anchor["strikes"],
            "anchor_max_age_seconds": anchor["max_age_seconds"],
            "anchor_reference_value": anchor["anchor"],
            "anchor_market_data_types": anchor["market_data_types"],
            "anchor_source": anchor["anchor_source"],
            "anchor_value": anchor["value"],
            "interval_seconds": interval_seconds,
            "value_change_points": value_change,
            "value_velocity_points_per_minute": (
                value_change * 60.0 / interval_seconds
            ),
            "direction": (
                "up"
                if value_change > 0.0
                else "down"
                if value_change < 0.0
                else "flat"
            ),
        }
    return {
        **base,
        "usable": not missing,
        "reasons": tuple(missing),
        "trading_date": trading_date.isoformat(),
        "end_ts": end["ts"],
        "end_chain_fingerprint": end["chain_fingerprint"],
        "end_target_expiry": end_snapshot.target_expiry,
        "end_pairs": end["pairs"],
        "end_dispersion_points": end["dispersion_points"],
        "end_median_relative_spread": end["median_relative_spread"],
        "end_strikes": end["strikes"],
        "end_max_age_seconds": end["max_age_seconds"],
        "end_reference_value": end["anchor"],
        "end_market_data_types": end["market_data_types"],
        "end_anchor_source": end["anchor_source"],
        "end_value": end["value"],
        "horizons": horizons,
    }


def xsp_option_context_at(
    snapshots: Sequence[QuoteSnapshot],
    *,
    decision_at: datetime,
) -> dict[str, object]:
    """Freeze the latest causal parity observation and its prior movement."""

    decision = (
        decision_at.replace(tzinfo=UTC)
        if decision_at.tzinfo is None
        else decision_at.astimezone(UTC)
    )
    session = xsp_session_label_et(decision)
    parsed = []
    for snapshot in snapshots:
        try:
            captured_at = datetime.fromisoformat(snapshot.ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=UTC)
        captured_at = captured_at.astimezone(UTC)
        parsed.append((captured_at, snapshot))
    preopen_path = _xsp_preopen_option_path(parsed, decision_at=decision)
    same_session = []
    for captured_at, snapshot in parsed:
        age = decision - captured_at
        if snapshot.session == session and age >= timedelta(0):
            same_session.append((captured_at, snapshot))
    candidates = [
        row
        for row in same_session
        if decision - row[0] <= XSP_OPTION_CONTEXT_MAX_LAG
    ]
    if not candidates:
        return {
            "source": "option_nbbo_parity",
            "authority": "observation_only",
            "usable": False,
            "reasons": ("no_causal_same_session_snapshot",),
            "preopen_path": preopen_path,
        }
    captured_at, snapshot = max(candidates, key=lambda row: row[0])
    observation = option_parity_observation(snapshot)
    prior = None
    for prior_at, prior_snapshot in sorted(
        same_session,
        key=lambda row: row[0],
        reverse=True,
    ):
        span = captured_at - prior_at
        if (
            timedelta(0) < span <= XSP_OPTION_CHANGE_MAX_SPAN
            and prior_snapshot.target_expiry == snapshot.target_expiry
        ):
            prior_observation = option_parity_observation(prior_snapshot)
            if prior_observation["usable"]:
                prior = (prior_at, prior_observation)
                break
    change: dict[str, object] = {
        "usable": False,
        "reasons": (
            ("current_observation_unusable",)
            if not observation["usable"]
            else ("no_prior_usable_same_session_snapshot",)
        ),
    }
    if observation["usable"] and prior is not None:
        prior_at, prior_observation = prior
        interval_seconds = (captured_at - prior_at).total_seconds()
        value_change = float(observation["value"]) - float(
            prior_observation["value"]
        )
        change = {
            "usable": True,
            "reasons": (),
            "prior_ts": prior_observation["ts"],
            "prior_chain_fingerprint": prior_observation["chain_fingerprint"],
            "prior_pairs": prior_observation["pairs"],
            "prior_dispersion_points": prior_observation["dispersion_points"],
            "prior_median_relative_spread": prior_observation[
                "median_relative_spread"
            ],
            "prior_strikes": prior_observation["strikes"],
            "prior_max_age_seconds": prior_observation["max_age_seconds"],
            "prior_reference_value": prior_observation["anchor"],
            "prior_market_data_types": prior_observation["market_data_types"],
            "prior_anchor_source": prior_observation["anchor_source"],
            "prior_value": prior_observation["value"],
            "interval_seconds": interval_seconds,
            "value_change_points": value_change,
            "value_velocity_points_per_minute": (
                value_change * 60.0 / interval_seconds
            ),
            "direction": (
                "up"
                if value_change > 0.0
                else "down"
                if value_change < 0.0
                else "flat"
            ),
        }
    return {
        **observation,
        "authority": "observation_only",
        "decision_lag_seconds": (decision - captured_at).total_seconds(),
        "parity_change": change,
        "preopen_path": preopen_path,
    }


def xsp_fundamental_context_at(
    snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    *,
    decision_at: datetime,
) -> dict[str, object]:
    """Freeze the timestamp-valid news state without granting trade authority."""

    context = {
        "source": "causal_news",
        "authority": "observation_only",
    }
    if snapshot is None:
        return {**context, "usable": False, "reason": "missing"}
    history: Sequence[Mapping[str, object]] = ()
    selected: Mapping[str, object] | None
    if isinstance(snapshot, Mapping):
        selected = snapshot
    else:
        history = snapshot
        try:
            selected = select_news_snapshot_at(history, as_of=decision_at)
        except NewsError:
            return {
                **context,
                "snapshot_fingerprint": calibration_fingerprint(history),
                "usable": False,
                "reason": "invalid_snapshot_history",
            }
        if selected is None:
            return {
                **context,
                "snapshot_fingerprint": calibration_fingerprint(history),
                "usable": False,
                "reason": "not_recorded_at_decision",
            }
    try:
        observation = observe_news_signal(
            selected,
            symbol="XSP",
            as_of=decision_at,
        )
    except NewsError:
        return {
            **context,
            "snapshot_fingerprint": calibration_fingerprint(selected),
            "usable": False,
            "reason": "invalid_snapshot",
        }
    pressure = (
        round(
            observation.direction
            * observation.impact
            / 100.0
            * observation.confidence,
            6,
        )
        if observation.usable
        else None
    )
    previous_pressure = None
    pressure_interval_seconds = None
    if history:
        try:
            selected_at = datetime.fromisoformat(
                str(selected["snapshot_as_of_utc"]).replace("Z", "+00:00")
            )
            previous = select_news_snapshot_at(
                history,
                as_of=selected_at - timedelta(microseconds=1),
            )
            if previous is not None:
                previous_observation = observe_news_signal(
                    previous,
                    symbol="XSP",
                    as_of=decision_at,
                )
                if previous_observation.run_status in (
                    "published",
                    "no_new_evidence",
                ):
                    previous_at = datetime.fromisoformat(
                        previous_observation.snapshot_as_of_utc.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                    pressure_interval_seconds = (
                        selected_at - previous_at
                    ).total_seconds()
                    previous_pressure = round(
                        previous_observation.direction
                        * previous_observation.impact
                        / 100.0
                        * previous_observation.confidence,
                        6,
                    )
        except (KeyError, TypeError, ValueError, NewsError):
            previous_pressure = None
            pressure_interval_seconds = None
    pressure_delta = (
        round(pressure - previous_pressure, 6)
        if pressure is not None and previous_pressure is not None
        else None
    )
    return {
        **context,
        "snapshot_fingerprint": calibration_fingerprint(selected),
        **observation.as_payload(),
        "signed_pressure": pressure,
        "pressure_delta": pressure_delta,
        "pressure_interval_seconds": pressure_interval_seconds,
        "pressure_velocity_per_hour": (
            round(pressure_delta * 3600.0 / pressure_interval_seconds, 6)
            if pressure_delta is not None
            and pressure_interval_seconds is not None
            and pressure_interval_seconds > 0.0
            else None
        ),
    }
