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


def xsp_trade_attribution(trade: object) -> dict[str, object]:
    """Project the engine's causal signal/excursion trace without policy copies."""

    trace_raw = getattr(trade, "decision_trace", None)
    trace = dict(trace_raw) if isinstance(trace_raw, Mapping) else {}
    entry_raw = trace.get("entry_guard_inputs")
    entry = dict(entry_raw) if isinstance(entry_raw, Mapping) else {}
    control_raw = entry.get("entry_control")
    control = (
        {
            key: control_raw.get(key)
            for key in (
                "source",
                "proposed_direction",
                "controls",
                "blocked_by",
                "direction",
                "branch",
            )
        }
        if isinstance(control_raw, Mapping)
        else None
    )
    impulse_raw = entry.get("directional_impulse")
    exits_raw = trace.get("exits")
    exit_raw = (
        exits_raw[-1]
        if isinstance(exits_raw, list)
        and exits_raw
        and isinstance(exits_raw[-1], Mapping)
        else {}
    )
    exit_signal_raw = exit_raw.get("signal_snapshot")
    return {
        "schema": "xsp.trade-attribution.v1",
        "decision_trace_fingerprint": (
            calibration_fingerprint(trace) if trace else None
        ),
        "bars_held": int(getattr(trade, "bars_held", 0) or 0),
        "execution_mfe_points": float(
            getattr(trade, "max_favorable_excursion", 0.0) or 0.0
        ),
        "execution_mae_points": float(
            getattr(trade, "max_adverse_excursion", 0.0) or 0.0
        ),
        "entry": {
            "signal_bar_ts": entry.get("signal_bar_ts"),
            "source_direction": entry.get("signal_source_dir"),
            "control": control,
            "directional_impulse": (
                dict(impulse_raw)
                if isinstance(impulse_raw, Mapping)
                else None
            ),
            "market_state": {
                key: entry.get(key)
                for key in (
                    "shock_atr_pct",
                    "shock_atr_vel_pct",
                    "shock_atr_accel_pct",
                    "tr_ratio",
                    "tr_median_pct",
                    "slope_med_pct",
                    "slope_vel_pct",
                    "slope_med_slow_pct",
                    "slope_vel_slow_pct",
                )
            },
            "local_extrema": (
                dict(value)
                if isinstance(
                    value := trace.get("entry_local_extrema_probe"),
                    Mapping,
                )
                else None
            ),
        },
        "exit": {
            "stage": exit_raw.get("stage"),
            "bar_ts": exit_raw.get("bar_ts"),
            "signal_snapshot": (
                dict(exit_signal_raw)
                if isinstance(exit_signal_raw, Mapping)
                else None
            ),
            "local_extrema": (
                dict(value)
                if isinstance(
                    value := exit_raw.get("local_extrema_probe"),
                    Mapping,
                )
                else None
            ),
        },
    }


def xsp_execution_state_context(
    source_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the current causal engine state beside one execution plan."""

    session = str(source_receipt.get("session") or "").upper()
    paired = source_receipt.get("paired_equity")
    observations = (
        paired.get("signal_observations")
        if isinstance(paired, Mapping)
        else None
    )
    signal_session = "RTH" if session == "CURB" else session
    observation = (
        observations.get(signal_session.lower())
        if isinstance(observations, Mapping)
        else None
    )
    daily_context = (
        paired.get("daily_context_state")
        if isinstance(paired, Mapping)
        else None
    )
    impulse = (
        observation.get("directional_impulse")
        if isinstance(observation, Mapping)
        else None
    )
    entry_control = (
        observation.get("entry_control")
        if isinstance(observation, Mapping)
        else None
    )
    signal_bar_ts = (
        str(observation.get("signal_bar_ts") or "")
        if isinstance(observation, Mapping)
        else ""
    )
    if (
        session not in {"RTH", "GTH", "CURB"}
        or not isinstance(observation, Mapping)
        or observation.get("schema") != "spot.signal-snapshot.v1"
        or not signal_bar_ts
        or not isinstance(impulse, Mapping)
        or not isinstance(entry_control, Mapping)
        or not isinstance(daily_context, Mapping)
        or not daily_context.get("state_fingerprint")
    ):
        raise ValueError("XSP execution state context is incomplete")
    pressure = source_receipt.get("fundamental_pressure")
    if pressure is not None and not isinstance(pressure, Mapping):
        raise ValueError("XSP execution news context is invalid")
    return {
        "schema": "xsp.execution-state-context.v1",
        "source_checkpoint_id": source_receipt.get("checkpoint_id"),
        "source_recorded_at_utc": source_receipt.get("recorded_at_utc"),
        "session": session,
        **({"signal_session": signal_session} if session == "CURB" else {}),
        "signal_bar_ts": signal_bar_ts,
        "signal_snapshot_age_bars": observation.get(
            "signal_snapshot_age_bars"
        ),
        "signal_source_dir": observation.get("signal_source_dir"),
        "entry_control": dict(entry_control),
        "directional_impulse": dict(impulse),
        "market_state": {
            key: observation.get(key)
            for key in (
                "hard_dir",
                "regime4_state",
                "release_age_bars",
                "shock_dir",
                "shock_atr_pct",
                "shock_atr_vel_pct",
                "shock_atr_accel_pct",
                "tr_ratio",
                "tr_median_pct",
                "slope_med_pct",
                "slope_vel_pct",
                "slope_med_slow_pct",
                "slope_vel_slow_pct",
            )
        },
        "daily_context_state": dict(daily_context),
        "fundamental_pressure": (
            dict(pressure) if isinstance(pressure, Mapping) else None
        ),
    }


def xsp_execution_signal_context(
    paired_equity: Mapping[str, object],
) -> dict[str, object] | None:
    """Freeze one profile-parity-checked engine signal for later execution."""

    identity_fields = (
        "lane",
        "direction",
        "entry_time_utc",
        "signal_bar_ts",
        "control",
        "directional_impulse",
        "market_state",
    )
    profiles = paired_equity.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("XSP execution attribution requires paired profiles")
    contexts: list[dict[str, object] | None] = []
    identities: list[dict[str, object] | None] = []
    for name in ("research", "broker"):
        profile = profiles.get(name)
        if not isinstance(profile, Mapping):
            raise ValueError("XSP execution attribution requires both profiles")
        position = profile.get("latest_position")
        if position is None:
            contexts.append(None)
            identities.append(None)
            continue
        attribution = (
            position.get("attribution")
            if isinstance(position, Mapping)
            else None
        )
        entry = (
            attribution.get("entry")
            if isinstance(attribution, Mapping)
            else None
        )
        impulse = (
            entry.get("directional_impulse")
            if isinstance(entry, Mapping)
            else None
        )
        market_state = (
            entry.get("market_state")
            if isinstance(entry, Mapping)
            else None
        )
        signal_bar_ts = (
            str(entry.get("signal_bar_ts") or "")
            if isinstance(entry, Mapping)
            else ""
        )
        if (
            not isinstance(position, Mapping)
            or position.get("lane") not in {"rth", "gth"}
            or position.get("direction") not in {"up", "down"}
            or not position.get("entry_time")
            or not signal_bar_ts
            or not isinstance(attribution, Mapping)
            or not attribution.get("decision_trace_fingerprint")
            or not isinstance(impulse, Mapping)
            or not isinstance(market_state, Mapping)
        ):
            raise ValueError("XSP open position has no causal engine attribution")
        control = entry.get("control")
        extrema = entry.get("local_extrema")
        context = {
            "schema": "xsp.execution-signal-context.v1",
            "lane": position["lane"],
            "direction": position["direction"],
            "entry_time_utc": position["entry_time"],
            "signal_bar_ts": signal_bar_ts,
            "decision_trace_fingerprint": attribution[
                "decision_trace_fingerprint"
            ],
            "control": dict(control) if isinstance(control, Mapping) else None,
            "directional_impulse": dict(impulse),
            "market_state": dict(market_state),
            "local_extrema": (
                dict(extrema) if isinstance(extrema, Mapping) else None
            ),
        }
        contexts.append(context)
        identities.append(
            {field: context.get(field) for field in identity_fields}
        )
    if identities[0] != identities[1]:
        raise ValueError("XSP research/broker execution attribution drift")
    return contexts[0]
