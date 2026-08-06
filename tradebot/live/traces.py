"""Compact, strategy-neutral traces projected from immutable live evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..news.contract import (
    NewsError,
    NewsSignalObservation,
    load_news_history,
    observe_news_signal,
    publication_id,
)
from .runs import _identity


LIVE_STRATEGY_TRACE_SCHEMA = "live.strategy-trace.v1"
_NEWS_CHANGE_WINDOWS_HOURS = (4, 24, 168)
_NEWS_SYMBOL_BY_TRACE_KEY = {"1OZ": "GC"}
_NewsObservationCache = MutableMapping[tuple[str, str], NewsSignalObservation]


def _map(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def causal_news_paths(latest_path: Path, *, as_of: datetime) -> tuple[Path, ...]:
    """Return the bounded latest/current/prior-month causal-news sources."""

    current = as_of.astimezone(timezone.utc).date().replace(day=1)
    previous = (current - timedelta(days=1)).replace(day=1)
    history = latest_path.parent / "history"
    return (
        history / f"{previous.year:04d}-{previous.month:02d}.jsonl",
        history / f"{current.year:04d}-{current.month:02d}.jsonl",
        latest_path,
    )


def load_causal_news(paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
    """Load and deduplicate a bounded causal-news publication sequence."""

    rows: list[dict[str, object]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            try:
                rows.extend(load_news_history(path))
            except NewsError:
                continue
        else:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(
            row.get("publication_id")
            or f"{row.get('snapshot_as_of_utc')}|{row.get('signal_as_of_utc')}"
        )
        unique[key] = row
    return tuple(
        sorted(
            unique.values(), key=lambda row: str(row.get("snapshot_as_of_utc") or "")
        )
    )


def _select_causal_snapshot(
    snapshots: Sequence[Mapping[str, object]],
    *,
    as_of: datetime,
    symbol: str | None = None,
    excluded_index: int | None = None,
) -> tuple[int, Mapping[str, object]] | None:
    """Select from already-published history without reparsing every timestamp."""

    cutoff = as_of.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    selected: tuple[str, int, Mapping[str, object]] | None = None
    for index, snapshot in enumerate(snapshots):
        if index == excluded_index:
            continue
        published = str(snapshot.get("snapshot_as_of_utc") or "")
        if published > cutoff:
            continue
        if symbol is not None and symbol not in _map(
            _map(snapshot.get("analysis")).get("assets")
        ):
            continue
        candidate = (published, index, snapshot)
        if selected is None or candidate[:2] > selected[:2]:
            selected = candidate
    return (selected[1], selected[2]) if selected is not None else None


def _observe_cached_news(
    snapshot: Mapping[str, object],
    *,
    symbol: str,
    as_of: datetime,
    cache: _NewsObservationCache,
) -> NewsSignalObservation:
    """Validate each immutable publication once; derive only time-varying freshness."""

    identifier = str(snapshot.get("publication_id") or publication_id(snapshot))
    key = (identifier, symbol)
    base = cache.get(key)
    if base is None:
        published_at = _utc(snapshot.get("snapshot_as_of_utc"))
        if published_at is None:
            raise NewsError("news snapshot has an invalid publication time")
        base = observe_news_signal(snapshot, symbol=symbol, as_of=published_at)
        cache[key] = base
    decision_at = as_of.astimezone(timezone.utc)
    signal_at = _utc(base.signal_as_of_utc)
    snapshot_at = _utc(base.snapshot_as_of_utc)
    if signal_at is None or snapshot_at is None:
        raise NewsError("news snapshot has invalid signal or publication time")
    age = (decision_at - signal_at).total_seconds()
    available = decision_at >= snapshot_at
    usable = (
        base.run_status in ("published", "no_new_evidence")
        and available
        and 0 <= age <= base.horizon_hours * 3600
    )
    reason = (
        "fresh"
        if usable
        else "future"
        if age < 0 or not available
        else "stale"
        if age > base.horizon_hours * 3600
        else f"run_status:{base.run_status or 'missing'}"
    )
    return replace(base, age_seconds=float(age), usable=usable, reason=reason)


def _news_change_windows(
    snapshots: Sequence[Mapping[str, object]],
    *,
    symbol: str,
    current_at: datetime,
    current_pressure: float,
    observation_cache: _NewsObservationCache,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for hours in _NEWS_CHANGE_WINDOWS_HOURS:
        try:
            selected = _select_causal_snapshot(
                snapshots,
                as_of=current_at - timedelta(hours=hours),
                symbol=symbol,
            )
            anchor = selected[1] if selected is not None else None
            anchor_at = _utc(anchor.get("snapshot_as_of_utc")) if anchor else None
            observed = (
                _observe_cached_news(
                    anchor,
                    symbol=symbol,
                    as_of=anchor_at,
                    cache=observation_cache,
                )
                if anchor is not None and anchor_at is not None
                else None
            )
        except (NewsError, TypeError, ValueError):
            anchor_at = None
            observed = None
        if observed is None or anchor_at is None:
            windows.append({"hours": hours, "available": False})
            continue
        elapsed = (current_at - anchor_at).total_seconds() / 3600.0
        if elapsed <= 0.0:
            windows.append({"hours": hours, "available": False})
            continue
        anchor_pressure = observed.direction * observed.impact / 100.0
        delta = current_pressure - anchor_pressure
        windows.append(
            {
                "hours": hours,
                "available": True,
                "elapsed_hours": elapsed,
                "pressure_delta": delta,
                "pressure_velocity_per_hour": delta / elapsed,
                "anchor_snapshot_as_of_utc": observed.snapshot_as_of_utc,
            }
        )
    return windows


def _driver_scores(
    snapshot: Mapping[str, object],
    *,
    symbol: str,
    drivers: Sequence[str],
) -> list[dict[str, object]]:
    """Project ranked per-asset scores for the aggregate's declared drivers."""

    wanted = {str(driver) for driver in drivers}
    asset = str(symbol).strip().lower()
    ranked: list[dict[str, object]] = []
    for bucket, values in _map(snapshot.get("event_snapshot")).items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for raw in values:
            event = _map(raw)
            event_id = str(event.get("id") or "")
            if event_id not in wanted:
                continue
            score = _map(event.get(asset))
            direction = score.get("direction")
            impact = _number(score.get("impact"))
            if direction not in (-1, 0, 1) or impact is None:
                continue
            ranked.append(
                {
                    "id": event_id,
                    "label": str(event.get("event") or event_id),
                    "direction": int(direction),
                    "impact": int(impact),
                    "confidence": _number(event.get("confidence")),
                    "status": event.get("status"),
                    "basis": event.get("basis"),
                    "bucket": str(bucket),
                }
            )
    return sorted(
        ranked,
        key=lambda row: (-int(row["impact"]), str(row["id"])),
    )


def _causal_news_at(
    snapshots: Sequence[Mapping[str, object]],
    *,
    symbol: str,
    as_of: datetime,
    authority: str,
    observation_cache: _NewsObservationCache | None = None,
) -> dict[str, object]:
    cache: _NewsObservationCache = (
        observation_cache if observation_cache is not None else {}
    )
    try:
        selected_item = _select_causal_snapshot(snapshots, as_of=as_of)
        if selected_item is None:
            return {}
        selected_index, selected = selected_item
        current = _observe_cached_news(
            selected,
            symbol=symbol,
            as_of=as_of,
            cache=cache,
        )
        selected_id = str(selected.get("publication_id") or publication_id(selected))
        prior_item = _select_causal_snapshot(
            snapshots,
            as_of=as_of,
            excluded_index=selected_index,
        )
        prior_snapshot = prior_item[1] if prior_item is not None else None
        prior = (
            _observe_cached_news(
                prior_snapshot,
                symbol=symbol,
                as_of=as_of,
                cache=cache,
            )
            if prior_snapshot is not None
            else None
        )
    except (NewsError, TypeError, ValueError):
        return {}
    pressure = current.direction * current.impact / 100.0
    prior_pressure = prior.direction * prior.impact / 100.0 if prior else pressure
    delta = pressure - prior_pressure
    current_at = _utc(current.snapshot_as_of_utc)
    prior_at = _utc(prior.snapshot_as_of_utc) if prior else None
    seconds = (
        (current_at - prior_at).total_seconds()
        if current_at is not None and prior_at is not None
        else 0.0
    )
    payload = {
        **current.as_payload(),
        "authority": authority,
        "snapshot_fingerprint": selected_id,
        "signed_pressure": pressure,
        "pressure_delta": delta,
        "pressure_velocity_per_hour": delta / (seconds / 3600.0)
        if seconds > 0
        else 0.0,
    }
    if current_at is not None:
        payload["change_windows"] = _news_change_windows(
            snapshots,
            symbol=symbol,
            current_at=current_at,
            current_pressure=pressure,
            observation_cache=cache,
        )
    driver_scores = _driver_scores(
        selected,
        symbol=symbol,
        drivers=current.drivers,
    )
    if driver_scores:
        payload["driver_scores"] = driver_scores
    return payload


def _news(value: object) -> dict[str, object]:
    source = _map(value)
    keys = (
        "authority",
        "source",
        "symbol",
        "direction",
        "impact",
        "confidence",
        "horizon_hours",
        "change",
        "signed_pressure",
        "pressure_delta",
        "pressure_velocity_per_hour",
        "age_seconds",
        "usable",
        "reason",
        "signal_as_of_utc",
        "snapshot_as_of_utc",
        "snapshot_fingerprint",
        "publication_id",
    )
    result = {key: source.get(key) for key in keys if source.get(key) is not None}
    drivers = source.get("drivers")
    if isinstance(drivers, Sequence) and not isinstance(drivers, (str, bytes)):
        result["drivers"] = [str(driver) for driver in drivers[:5]]
    driver_scores = source.get("driver_scores")
    if isinstance(driver_scores, Sequence) and not isinstance(
        driver_scores, (str, bytes)
    ):
        result["driver_scores"] = [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "label",
                    "direction",
                    "impact",
                    "confidence",
                    "status",
                    "basis",
                    "bucket",
                )
                if item.get(key) is not None
            }
            for raw in driver_scores[:5]
            if (item := _map(raw))
        ]
    change_windows = source.get("change_windows")
    if isinstance(change_windows, Sequence) and not isinstance(
        change_windows, (str, bytes)
    ):
        result["change_windows"] = [
            {
                key: item.get(key)
                for key in (
                    "hours",
                    "available",
                    "elapsed_hours",
                    "pressure_delta",
                    "pressure_velocity_per_hour",
                    "anchor_snapshot_as_of_utc",
                )
                if item.get(key) is not None
            }
            for raw in change_windows
            if (item := _map(raw))
        ]
    return result


def _horizons(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    projected = []
    for raw in value:
        item = _map(raw)
        minutes = _number(item.get("elapsed_minutes"))
        bars = _number(item.get("bars"))
        if minutes is None and bars is None:
            continue
        projected.append(
            {
                "minutes": minutes,
                "bars": int(bars) if bars is not None else None,
                "angle": _number(item.get("slope_angle_deg")),
                "slope": _number(item.get("slope_pct_per_bar")),
                "slope_velocity": _number(item.get("slope_velocity_pct_per_bar")),
                "return": _number(item.get("return_pct")),
                "efficiency": _number(item.get("efficiency")),
                "turn": item.get("turn"),
                "turn_age_bars": item.get("turn_age_bars"),
            }
        )
    return projected


def _impulse(
    context: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    extra_decision: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    impulse = _map(context.get("directional_impulse"))
    entry = _map(context.get("entry_control"))
    decision = {
        "ready": impulse.get("ready"),
        "direction": impulse.get("direction"),
        "abstain_reason": impulse.get("abstain_reason"),
        "trend": impulse.get("trend_state"),
        "turn": impulse.get("turn_event"),
        "turn_ready": impulse.get("turn_ready"),
        "turn_sequence_direction": impulse.get("turn_sequence_direction"),
        "turn_sequence_order": impulse.get("turn_sequence_order"),
        "state_age_bars": impulse.get("state_age_bars"),
        "coherence": _number(impulse.get("coherence")),
        "conviction": _number(impulse.get("conviction")),
        "direction_score": _number(impulse.get("direction_score")),
        "smoothed_direction_score": _number(impulse.get("smoothed_direction_score")),
        "proposed_direction": entry.get("proposed_direction"),
        "blocked_by": entry.get("blocked_by"),
        "source": entry.get("source"),
        "signal_source_direction": context.get("signal_source_dir"),
        "signal_bar_utc": context.get("signal_bar_ts"),
        "session": context.get("signal_session") or context.get("session"),
        "controls": list(entry.get("controls") or ()),
        **dict(extra_decision or {}),
    }
    decision = {key: value for key, value in decision.items() if value is not None}
    volatility = {
        "atr_ratio": _number(impulse.get("atr_ratio")),
        "atr_velocity": _number(impulse.get("atr_velocity_pct")),
        "atr_acceleration": _number(impulse.get("atr_acceleration_pct")),
        "atr_fast": _number(impulse.get("atr_fast_pct")),
        "atr_slow": _number(impulse.get("atr_slow_pct")),
        "unit": "pct",
    }
    return decision, volatility, _horizons(impulse.get("horizons"))


def _xsp_context(
    context: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    decision, volatility, horizons = _impulse(context, plan=plan)
    daily = _map(_map(context.get("daily_context_state")).get("state"))
    long_context = {
        key: daily.get(key)
        for key in (
            "directions",
            "transition",
            "hard_direction",
            "soft_direction",
            "fast_direction",
            "mid_direction",
            "damage_phase",
            "tr_phase",
            "tr_velocity",
            "tr_acceleration",
            "return_velocity",
            "return_acceleration",
        )
        if daily.get(key) is not None
    }
    market = {
        key: value
        for key, value in _map(context.get("market_state")).items()
        if value is not None
    }
    return {
        "family": "IMPULSE",
        "decision": decision,
        "volatility": volatility,
        "horizons": horizons,
        "long_context": long_context,
        "market": market,
        "news": _news(context.get("fundamental_pressure")),
    }


def _gold_context(context: Mapping[str, object]) -> dict[str, object]:
    signal = _map(context.get("signal"))
    daily = _map(signal.get("daily"))
    h4 = _map(signal.get("h4"))
    owner = _map(context.get("owner_state"))
    decision = {
        "ready": signal.get("usable"),
        "direction": signal.get("raw_direction"),
        "proposed_direction": signal.get("proposed_direction"),
        "turn": signal.get("raw_turn"),
        "decision_bar_utc": signal.get("decision_bar_end_utc"),
        "macro_trigger": signal.get("macro_trigger"),
        "daily_hard": daily.get("hard_direction"),
        "daily_soft": daily.get("soft_direction"),
        "h4_hard": h4.get("hard_direction"),
        "owner_target": owner.get("target_direction"),
    }
    macro = _map(context.get("macro"))
    macro_horizons = []
    for window, raw in sorted(
        _map(macro.get("horizons")).items(),
        key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else 10_000,
    ):
        item = _map(raw)
        macro_horizons.append(
            {
                "window": str(window),
                "direction": item.get("direction"),
                "velocity": item.get("velocity"),
                "acceleration": item.get("acceleration"),
                "state_age": item.get("state_age"),
            }
        )
    pair = _map(context.get("contract_pair"))
    return {
        "family": "GOLD_REGIME",
        "decision": {
            key: value for key, value in decision.items() if value is not None
        },
        "volatility": {
            "atr_ratio": _number(daily.get("atr_ratio_14_63")),
            "atr_velocity": _number(daily.get("atr_velocity")),
            "atr_acceleration": None,
            "atr_fast": _number(daily.get("atr14")),
            "atr_slow": None,
            "unit": "daily_gold",
        },
        "horizons": [],
        "long_context": {
            "daily": {
                key: daily.get(key)
                for key in ("end", "hard_age", "high_contracting")
                if daily.get(key) is not None
            }
        },
        "market": {
            "h4_atr14_dollars": _number(h4.get("atr14_dollars")),
            "h4_atr_velocity_dollars": _number(h4.get("atr_velocity_dollars")),
            "h4_fast_slope_dollars": _number(h4.get("signed_fast_slope_dollars")),
            "h4_fast_acceleration_dollars": _number(
                h4.get("signed_fast_acceleration_dollars")
            ),
            "h4_spread_velocity_dollars": _number(
                h4.get("signed_spread_velocity_dollars")
            ),
            "basis_usd": _number(pair.get("basis_usd")),
            "basis_bps": _number(pair.get("basis_bps_of_gc")),
            "contract_month": pair.get("contract_month"),
        },
        "macro": {
            "authority": macro.get("authority"),
            "source": macro.get("source"),
            "horizons": macro_horizons,
        },
        "news": _news(context.get("news")),
    }


def _mcl_context(
    plan: Mapping[str, object],
    *,
    records_by_id: Mapping[str, Mapping[str, object]],
    news_snapshots: Sequence[Mapping[str, object]],
    news_observation_cache: _NewsObservationCache | None = None,
) -> tuple[dict[str, object], dict[str, object]] | None:
    source_id = str(plan.get("source_checkpoint_id") or "")
    source_record = records_by_id.get(source_id)
    if not isinstance(source_record, Mapping):
        return None
    source_evidence = _map(source_record.get("evidence"))
    source = _map(source_evidence.get("source"))
    latest = _map(source.get("latest_decision"))
    snapshot = _map(latest.get("snapshot"))
    if not snapshot:
        return None
    context = {"directional_impulse": snapshot}
    decision, volatility, horizons = _impulse(
        context,
        plan=plan,
        extra_decision={
            "raw_direction": latest.get("raw_direction"),
            "proposed_direction": latest.get("proposed_direction"),
            "admitted_direction": latest.get("admitted_direction"),
            "route": latest.get("route"),
            "parity_aligned": latest.get("parity_aligned"),
            "risk_reduction": latest.get("risk_reduction"),
            "velocity_aligned": latest.get("velocity_aligned"),
            "velocity_breadth": latest.get("velocity_breadth"),
            "basis_velocity_ticks": latest.get("basis_velocity_ticks"),
            "cl_move": _number(latest.get("cl_move")),
            "mcl_move": _number(latest.get("mcl_move")),
            "signal_bar_utc": latest.get("observed_at_utc"),
        },
    )
    last_turn = _map(source.get("last_raw_turn"))
    last_turn_decision = _map(last_turn.get("decision"))
    observed = (
        _utc(latest.get("observed_at_utc"))
        or _utc(source.get("latest_common_close_utc"))
        or _utc(source_record.get("recorded_at_utc"))
    )
    news = (
        _causal_news_at(
            news_snapshots,
            symbol="MCL",
            as_of=observed,
            authority="non_scoring_context_only",
            observation_cache=news_observation_cache,
        )
        if observed is not None and news_snapshots
        else {}
    )
    return (
        {
            "family": "MCL_IMPULSE",
            "decision": decision,
            "volatility": volatility,
            "horizons": horizons,
            "long_context": {
                "last_raw_turn_id": last_turn.get("event_id"),
                "last_raw_turn_utc": last_turn.get("observed_at_utc"),
                "last_raw_turn_direction": (
                    last_turn.get("direction")
                    or last_turn_decision.get("raw_direction")
                    or last_turn_decision.get("proposed_direction")
                ),
                "source_rows": _map(source.get("rows")),
            },
            "market": {
                "contract_month": source.get("contract_month"),
            },
            "news": news,
        },
        {
            "source_checkpoint_id": source_id,
            "source_recorded_at_utc": source_record.get("recorded_at_utc"),
            "source_schema": source_evidence.get("schema"),
            "source_authority": source.get("authority"),
        },
    )


def project_execution_trace(
    record: Mapping[str, object],
    *,
    run: Mapping[str, object],
    trace_key: str,
    records_by_id: Mapping[str, Mapping[str, object]],
    news_snapshots: Sequence[Mapping[str, object]] = (),
    news_observation_cache: _NewsObservationCache | None = None,
) -> dict[str, object]:
    """Normalize one immutable execution checkpoint without trading authority."""

    evidence = _map(record.get("evidence"))
    plan = _map(evidence.get("plan"))
    risk = _map(evidence.get("risk_state"))
    context = _map(plan.get("execution_state_context"))
    provenance: dict[str, object] = {
        "context_schema": context.get("schema"),
        "source_checkpoint_id": context.get("source_checkpoint_id"),
        "source_recorded_at_utc": context.get("source_recorded_at_utc"),
    }
    if context.get("schema") == "gold.1oz-execution-state-context.v1" or (
        "signal" in context and "contract_pair" in context
    ):
        telemetry = _gold_context(context)
    elif "directional_impulse" in context:
        telemetry = _xsp_context(context, plan)
    else:
        mcl = _mcl_context(
            plan,
            records_by_id=records_by_id,
            news_snapshots=news_snapshots,
            news_observation_cache=news_observation_cache,
        )
        if mcl is not None:
            telemetry, mcl_provenance = mcl
            provenance.update(mcl_provenance)
        else:
            telemetry = {
                "family": "STATE",
                "decision": {},
                "volatility": {},
                "horizons": [],
                "long_context": {},
                "market": {},
                "news": {},
            }
    news = _news(telemetry.get("news"))
    recorded_at = _utc(record.get("recorded_at_utc"))
    if news_snapshots and recorded_at is not None and not news.get("change_windows"):
        historical = _news(
            _causal_news_at(
                news_snapshots,
                symbol=_NEWS_SYMBOL_BY_TRACE_KEY.get(trace_key, trace_key),
                as_of=recorded_at,
                authority=str(
                    news.get("authority") or "non_scoring_context_only"
                ),
                observation_cache=news_observation_cache,
            )
        )
        news = {**historical, **news}
    telemetry = {**telemetry, "news": news}
    leg = _map(plan.get("leg"))
    broker = _map(evidence.get("broker_order"))
    economics = {
        "net": _number(risk.get("run_net_usd")),
        "drawdown": _number(risk.get("drawdown_usd")),
        "cost": _number(risk.get("run_cost_usd")),
        "fills": risk.get("fill_count"),
        "trades": risk.get("closed_trades"),
    }
    body: dict[str, object] = {
        "schema": LIVE_STRATEGY_TRACE_SCHEMA,
        "event_id": record.get("checkpoint_id"),
        "recorded_at_utc": record.get("recorded_at_utc"),
        "sleeve_id": run.get("sleeve_id"),
        "run_id": run.get("run_id"),
        "strategy_id": run.get("strategy_id"),
        "trace_key": trace_key,
        "label": run.get("label"),
        "phase": evidence.get("phase") or "STATE",
        "status": plan.get("status"),
        "reason": plan.get("reason"),
        "target_direction": plan.get("target_direction"),
        "target_symbol": plan.get("target_symbol"),
        "action": leg.get("action") or broker.get("action"),
        "symbol": leg.get("symbol") or broker.get("symbol"),
        "quantity": leg.get("quantity", broker.get("quantity")),
        "holdings": dict(_map(plan.get("holdings"))),
        "economics": economics,
        **telemetry,
        "provenance": {
            key: value for key, value in provenance.items() if value is not None
        },
    }
    news_identity = dict(_map(body.get("news")))
    news_identity.pop("age_seconds", None)
    observation = {
        key: body.get(key)
        for key in (
            "phase",
            "status",
            "reason",
            "target_direction",
            "target_symbol",
            "action",
            "symbol",
            "quantity",
            "holdings",
            "economics",
            "family",
            "decision",
            "volatility",
            "horizons",
            "long_context",
            "market",
            "macro",
        )
    }
    observation_decision = dict(_map(observation.get("decision")))
    for key in (
        "signal_bar_utc",
        "basis_velocity_ticks",
        "cl_move",
        "mcl_move",
    ):
        observation_decision.pop(key, None)
    observation["decision"] = observation_decision
    observation_long = dict(_map(observation.get("long_context")))
    observation_long.pop("source_rows", None)
    observation["long_context"] = observation_long
    observation["news"] = news_identity
    decision = _map(body.get("decision"))
    long_context = _map(body.get("long_context"))
    news = _map(body.get("news"))
    episode = {
        "phase": body.get("phase"),
        "status": body.get("status"),
        "reason": body.get("reason"),
        "target_direction": body.get("target_direction"),
        "action": body.get("action"),
        "holdings": body.get("holdings"),
        "decision": {
            key: decision.get(key)
            for key in (
                "ready",
                "direction",
                "raw_direction",
                "proposed_direction",
                "admitted_direction",
                "abstain_reason",
                "trend",
                "turn",
                "turn_sequence_direction",
                "turn_sequence_order",
                "blocked_by",
                "daily_hard",
                "daily_soft",
                "h4_hard",
                "decision_bar_utc",
            )
        },
        "long": {
            key: long_context.get(key)
            for key in (
                "transition",
                "hard_direction",
                "soft_direction",
                "tr_phase",
                "last_raw_turn_id",
            )
        },
        "news": {
            key: news.get(key)
            for key in (
                "snapshot_fingerprint",
                "direction",
                "change",
                "usable",
                "reason",
            )
        },
        "fills": economics.get("fills"),
        "trades": economics.get("trades"),
    }
    return {
        **body,
        "observation_id": _identity(observation),
        "episode_id": _identity(episode),
    }


def _delta(
    current: Mapping[str, object], previous: Mapping[str, object]
) -> dict[str, object]:
    def numbers(left: object, right: object, keys: Sequence[str]) -> dict[str, float]:
        result = {}
        left_map, right_map = _map(left), _map(right)
        for key in keys:
            now, before = _number(left_map.get(key)), _number(right_map.get(key))
            if now is not None and before is not None:
                result[key] = now - before
        return result

    def horizon_key(item: Mapping[str, object]) -> tuple[str, float] | None:
        bars = _number(item.get("bars"))
        if bars is not None:
            return ("bars", bars)
        minutes = _number(item.get("minutes"))
        return ("minutes", minutes) if minutes is not None else None

    prior_horizons = {
        key: item
        for item in previous.get("horizons", ())
        if isinstance(item, Mapping) and (key := horizon_key(item)) is not None
    }
    horizon_deltas = []
    for item in current.get("horizons", ()):
        if not isinstance(item, Mapping):
            continue
        prior = prior_horizons.get(horizon_key(item))
        if prior is None:
            continue
        values = numbers(
            item,
            prior,
            ("angle", "slope", "slope_velocity", "return", "efficiency"),
        )
        if values:
            horizon_deltas.append(
                {
                    "minutes": item.get("minutes"),
                    "bars": item.get("bars"),
                    **values,
                }
            )
    return {
        "volatility": numbers(
            current.get("volatility"),
            previous.get("volatility"),
            ("atr_ratio", "atr_velocity", "atr_acceleration", "atr_fast", "atr_slow"),
        ),
        "decision": numbers(
            current.get("decision"),
            previous.get("decision"),
            ("coherence", "conviction", "direction_score", "smoothed_direction_score"),
        ),
        "economics": numbers(
            current.get("economics"),
            previous.get("economics"),
            ("net", "drawdown", "cost", "fills", "trades"),
        ),
        "horizons": horizon_deltas,
    }


def compact_strategy_traces(
    traces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Collapse exact consecutive observations and retain semantic episode deltas."""

    compacted: list[dict[str, object]] = []
    previous_observation: dict[str, object] | None = None
    for raw in traces:
        current = dict(raw)
        if compacted and compacted[-1].get("observation_id") == current.get(
            "observation_id"
        ):
            prior = compacted[-1]
            prior["sample_count"] = int(prior.get("sample_count") or 1) + 1
            prior["last_recorded_at_utc"] = current.get("recorded_at_utc")
            prior["last_event_id"] = current.get("event_id")
            for key in ("decision", "long_context", "market", "provenance", "news"):
                prior[key] = current.get(key)
            continue
        event_id = str(current.get("event_id") or _identity(current))
        current.update(
            {
                "trace_id": event_id,
                "first_recorded_at_utc": current.get("recorded_at_utc"),
                "last_recorded_at_utc": current.get("recorded_at_utc"),
                "last_event_id": current.get("event_id"),
                "sample_count": 1,
                "episode_start": (
                    previous_observation is None
                    or previous_observation.get("episode_id")
                    != current.get("episode_id")
                ),
                "delta": (
                    _delta(current, previous_observation)
                    if previous_observation is not None
                    else {
                        "volatility": {},
                        "decision": {},
                        "economics": {},
                        "horizons": [],
                    }
                ),
            }
        )
        compacted.append(current)
        previous_observation = current
    return compacted
