"""Canonical, non-submitting prospective evidence for the 1OZ gold quest."""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..news.contract import (
    NewsError,
    observe_news_signal,
    select_news_snapshot_at,
)
from .gold_context import (
    gold_bar_time as _bar_time,
    gold_bar_value as _bar_value,
    gold_complete_bars as _complete_bars,
    gold_daily_timeline as _daily_timeline,
    gold_finite as _finite,
    gold_h4_timeline,
    gold_latest_index as _latest_index,
    gold_macro_timeline as _macro_timeline,
    gold_utc as _utc,
)
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint


GOLD_ONSET_VERSION = "gold.1oz-prospective-onset.v2"
GOLD_ONSET_PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "backtests/gold/one_oz_onset_tape_preregistration_v2.json"
)
GOLD_ONSET_PROSPECTIVE_START = datetime(
    2026, 8, 3, 3, 30, 17, tzinfo=timezone.utc
)
GOLD_ONSET_HORIZONS_HOURS = (4, 12, 24)
GOLD_ONSET_FINANCING_USD = 2.32
GOLD_ONSET_CONFIG_FINGERPRINT = calibration_fingerprint(
    json.loads(GOLD_ONSET_PREREGISTRATION.read_text(encoding="utf-8"))
)
_MONTHS = {
    code: month
    for month, code in enumerate("FGHJKMNQUVXZ", 1)
}


def gold_news_context(
    history: Sequence[Mapping[str, object]], *, as_of: datetime
) -> dict[str, object]:
    """Return one causal GC aggregate with no strategy authority."""

    base: dict[str, object] = {
        "source": "causal_news",
        "symbol": "GC",
        "authority": "attribution_only",
    }
    try:
        selected = select_news_snapshot_at(history, as_of=_utc(as_of))
    except NewsError:
        return {**base, "usable": False, "reason": "invalid_history"}
    if selected is None:
        return {**base, "usable": False, "reason": "not_recorded"}
    try:
        observation = observe_news_signal(selected, symbol="GC", as_of=_utc(as_of))
    except NewsError:
        return {**base, "usable": False, "reason": "invalid_snapshot"}
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
    interval_seconds = None
    try:
        selected_at = _utc(selected["snapshot_as_of_utc"])
        previous = select_news_snapshot_at(
            history, as_of=selected_at - timedelta(microseconds=1)
        )
        if previous is not None:
            prior = observe_news_signal(previous, symbol="GC", as_of=_utc(as_of))
            previous_at = _utc(prior.snapshot_as_of_utc)
            if prior.run_status in ("published", "no_new_evidence"):
                previous_pressure = round(
                    prior.direction * prior.impact / 100.0 * prior.confidence,
                    6,
                )
                interval_seconds = (selected_at - previous_at).total_seconds()
    except (KeyError, TypeError, ValueError, NewsError):
        previous_pressure = None
        interval_seconds = None
    delta = (
        round(float(pressure) - previous_pressure, 6)
        if pressure is not None and previous_pressure is not None
        else None
    )
    return {
        **base,
        **observation.as_payload(),
        "snapshot_fingerprint": calibration_fingerprint(selected),
        "signed_pressure": pressure,
        "pressure_delta": delta,
        "pressure_interval_seconds": interval_seconds,
        "pressure_velocity_per_hour": (
            round(delta * 3600.0 / interval_seconds, 6)
            if delta is not None and interval_seconds and interval_seconds > 0
            else None
        ),
    }


def gold_macro_context(
    uup_rows: Sequence[object],
    tip_rows: Sequence[object],
    *,
    as_of: datetime,
) -> dict[str, object]:
    timeline = _macro_timeline(uup_rows, tip_rows, as_of=as_of)
    if not timeline:
        return {
            "source": "UUP_TIP_completed_RTH",
            "authority": "attribution_only",
            "usable": False,
            "reason": "underwarmed",
        }
    latest = timeline[-1]
    freshness = (_utc(as_of) - _utc(latest["end"])).total_seconds()
    horizons = dict(latest["horizons"])
    return {
        "source": "UUP_TIP_completed_RTH",
        "authority": "attribution_only",
        "usable": True,
        "end_utc": _utc(latest["end"]).isoformat(),
        "freshness_seconds": freshness,
        "horizons": horizons,
        "total_direction_neutral": all(
            dict(horizons[str(horizon)])["direction"] == "mixed"
            for horizon in (5, 21, 63)
        ),
    }


def gold_signal_context(
    h4_rows: Sequence[object],
    daily_rows: Sequence[object],
    uup_rows: Sequence[object],
    tip_rows: Sequence[object],
    *,
    as_of: datetime,
) -> dict[str, object]:
    """Reconstruct Stage-12 and Stage-22 signal gates from completed bars."""

    daily = _daily_timeline(daily_rows, as_of=as_of)
    macro = _macro_timeline(uup_rows, tip_rows, as_of=as_of)
    if not daily:
        return {"usable": False, "reason": "daily_underwarmed"}
    daily_times = [row["end"] for row in daily]
    macro_times = [row["end"] for row in macro]
    pending_macro_index = None
    latest = None
    for h4 in gold_h4_timeline(h4_rows, as_of=as_of):
        stamp = _utc(h4["end"])
        close = float(h4["close"])
        current_state = h4["raw_direction"]
        sign = 1.0 if current_state == "up" else -1.0 if current_state == "down" else None
        fast_slope = _finite(h4["fast_slope_dollars"])
        spread_velocity = _finite(h4["spread_velocity_dollars"])
        fast_acceleration = _finite(h4["fast_acceleration_dollars"])
        signed_fast = sign * fast_slope if sign is not None and fast_slope is not None else None
        signed_spread = (
            sign * spread_velocity
            if sign is not None and spread_velocity is not None
            else None
        )
        signed_acceleration = (
            sign * fast_acceleration
            if sign is not None and fast_acceleration is not None
            else None
        )
        financing_bars = (
            GOLD_ONSET_FINANCING_USD / signed_fast
            if signed_fast is not None and signed_fast > 0.0
            else None
        )
        financing_clock = (
            "unready"
            if signed_fast is None
            else "within_12h"
            if signed_fast >= GOLD_ONSET_FINANCING_USD / 3.0
            else "within_24h"
            if signed_fast >= GOLD_ONSET_FINANCING_USD / 6.0
            else "beyond_24h"
        )
        h4_context = {
            "authority": "attribution_only",
            "hard_direction": current_state,
            "signed_fast_slope_dollars": signed_fast,
            "signed_spread_velocity_dollars": signed_spread,
            "signed_fast_acceleration_dollars": signed_acceleration,
            "atr14_dollars": h4["atr14_dollars"],
            "atr_velocity_dollars": h4["atr_velocity_dollars"],
            "cost_per_atr14": (
                GOLD_ONSET_FINANCING_USD / float(h4["atr14_dollars"])
                if h4["atr14_dollars"] is not None
                and float(h4["atr14_dollars"]) > 0.0
                else None
            ),
            "fast_bars_to_finance": financing_bars,
            "financing_clock": financing_clock,
            "path": h4["path"],
        }
        daily_index = _latest_index(daily_times, stamp)
        macro_index = _latest_index(macro_times, stamp)
        if daily_index < 0:
            continue
        day = daily[daily_index]
        proposed = h4["proposed_direction"]
        hard_supports = bool(
            proposed in ("up", "down")
            and day["hard_direction"] == proposed
            and day["hard_age"] is not None
            and int(day["hard_age"]) >= 6
        )
        stage12 = proposed if hard_supports else None
        stage12_block = (
            "gold_hard_regime_maturation"
            if proposed in ("up", "down") and not hard_supports
            else None
        )
        stage22 = stage12
        stage22_block = stage12_block
        stage22_source = "stage12"
        if pending_macro_index is not None:
            still_up = bool(
                current_state == "up"
                and day["hard_direction"] == "up"
                and day["hard_age"] is not None
                and int(day["hard_age"]) >= 6
            )
            if not still_up:
                pending_macro_index = None
            elif macro_index - pending_macro_index >= 2:
                stage22 = "up"
                stage22_block = None
                stage22_source = "macro_matured"
                pending_macro_index = None
            else:
                stage22 = None
                stage22_block = "gold_macro_uncertainty_maturation"
                stage22_source = "macro_pending"
        trigger = False
        if macro_index >= 0:
            h5 = dict(macro[macro_index]["horizons"])["5"]
            trigger = bool(
                h5["direction"] == "mixed"
                and h5["velocity"] == "mixed"
                and int(h5["state_age"]) == 1
            )
        if pending_macro_index is None and stage22 == "up" and trigger:
            pending_macro_index = macro_index
            stage22 = None
            stage22_block = "gold_macro_uncertainty_maturation"
            stage22_source = "macro_started"
        latest = {
            "usable": True,
            "decision_bar_end_utc": stamp.isoformat(),
            "decision_close": close,
            "raw_direction": current_state,
            "raw_turn": h4["raw_turn"],
            "proposed_direction": proposed,
            "stage_12": {
                "admitted_direction": stage12,
                "blocked_by": stage12_block,
                "retention": "high_contracting_trail" if day["high_contracting"] else "stopless_flip_owner",
            },
            "stage_22": {
                "admitted_direction": stage22,
                "blocked_by": stage22_block,
                "source": stage22_source,
                "pending_macro_index": pending_macro_index,
            },
            "h4": h4_context,
            "daily": {**day, "end": _utc(day["end"]).isoformat()},
            "macro_trigger": trigger,
        }
    return latest or {"usable": False, "reason": "h4_underwarmed"}


def _contract_month(row: Mapping[str, object]) -> str | None:
    local = str(row.get("local_symbol") or "").strip().upper()
    match = re.search(r"([FGHJKMNQUVXZ])(\d)$", local)
    expiry = str(row.get("expiry") or "")
    digits = "".join(char for char in expiry if char.isdigit())
    if match is None or len(digits) < 4:
        return None
    month = _MONTHS[match.group(1)]
    return f"{int(digits[:4]):04d}-{month:02d}"


def select_gold_contract_pair(
    quotes: Sequence[Mapping[str, object]], *, observed_at: datetime
) -> dict[str, object]:
    """Select a live, liquid GC/1OZ pair in one contract month."""

    now = _utc(observed_at)
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    rejected = []
    for source in quotes:
        row = dict(source)
        symbol = str(row.get("symbol") or "").strip().upper()
        month = _contract_month(row)
        bid, ask, volume = (_finite(row.get(name)) for name in ("bid", "ask", "volume"))
        stamp_raw = row.get("observed_at_utc") or row.get("time")
        try:
            stamp = _utc(stamp_raw)
        except (TypeError, ValueError):
            stamp = None
        reasons = []
        if symbol not in ("GC", "1OZ"):
            reasons.append("unsupported_symbol")
        if month is None:
            reasons.append("unknown_contract_month")
        if int(row.get("market_data_type") or 0) != 1:
            reasons.append("not_live")
        if bid is None or ask is None or ask < bid:
            reasons.append("invalid_book")
        spread = ask - bid if bid is not None and ask is not None else None
        if spread is not None and spread > 2.0:
            reasons.append("spread_above_2_usd")
        if volume is None or volume < 100:
            reasons.append("volume_below_100")
        age = (now - stamp).total_seconds() if stamp is not None else None
        if age is None or age < 0 or age > 30:
            reasons.append("stale_quote")
        if reasons:
            rejected.append({"symbol": symbol, "local_symbol": row.get("local_symbol"), "reasons": reasons})
            continue
        assert month is not None and bid is not None and ask is not None and volume is not None and stamp is not None
        candidates[(month, symbol)] = {
            "symbol": symbol,
            "local_symbol": str(row.get("local_symbol") or ""),
            "con_id": int(row.get("con_id") or 0),
            "expiry": str(row.get("expiry") or ""),
            "market_data_type": 1,
            "contract_month": month,
            "bid": bid,
            "bid_size": _finite(row.get("bid_size")),
            "ask": ask,
            "ask_size": _finite(row.get("ask_size")),
            "last": _finite(row.get("last")),
            "mid": (bid + ask) / 2.0,
            "spread": spread,
            "volume": volume,
            "observed_at_utc": stamp.isoformat(),
            "age_seconds": age,
        }
    pairs = []
    for month in sorted({key[0] for key in candidates}):
        gc = candidates.get((month, "GC"))
        one = candidates.get((month, "1OZ"))
        if gc is None or one is None:
            continue
        pairs.append((min(float(gc["volume"]), float(one["volume"])), -float(gc["spread"]) - float(one["spread"]), month, gc, one))
    if not pairs:
        return {
            "usable": False,
            "reason": "no_live_liquid_shared_contract_month",
            "rejected": rejected,
        }
    _volume, _spread, month, gc, one = max(pairs)
    basis = float(one["mid"]) - float(gc["mid"])
    return {
        "usable": True,
        "authority": "market_data_only",
        "contract_month": month,
        "selector": "highest_minimum_pair_volume_then_lowest_combined_spread",
        "gc": gc,
        "one_oz": one,
        "basis_usd": basis,
        "basis_bps_of_gc": basis / float(gc["mid"]) * 10_000.0,
        "rejected": rejected,
    }


def build_gold_onset_context(
    *,
    xau_h4: Sequence[object],
    xau_daily: Sequence[object],
    uup_daily: Sequence[object],
    tip_daily: Sequence[object],
    quotes: Sequence[Mapping[str, object]],
    news_history: Sequence[Mapping[str, object]],
    source_points: Mapping[str, Mapping[str, object]],
    observed_at: datetime,
) -> dict[str, object]:
    signal = gold_signal_context(
        xau_h4,
        xau_daily,
        uup_daily,
        tip_daily,
        as_of=observed_at,
    )
    decision_at = (
        _utc(signal["decision_bar_end_utc"])
        if signal.get("usable") and signal.get("decision_bar_end_utc")
        else _utc(observed_at)
    )
    macro = gold_macro_context(uup_daily, tip_daily, as_of=decision_at)
    pair = select_gold_contract_pair(quotes, observed_at=observed_at)
    news = gold_news_context(news_history, as_of=decision_at)
    stage12 = signal.get("stage_12") if isinstance(signal, Mapping) else None
    stage22 = signal.get("stage_22") if isinstance(signal, Mapping) else None
    directions = {
        "stage_12": dict(stage12).get("admitted_direction") if isinstance(stage12, Mapping) else None,
        "stage_22": dict(stage22).get("admitted_direction") if isinstance(stage22, Mapping) else None,
    }
    point_limits = {"XAUUSD": 3600.0, "GC": 3600.0, "1OZ": 1800.0}
    points = {
        symbol: {
            "close": _finite(dict(source_points.get(symbol) or {}).get("close")),
            "bar_end_utc": dict(source_points.get(symbol) or {}).get("bar_end_utc"),
            "age_seconds": _finite(dict(source_points.get(symbol) or {}).get("age_seconds")),
        }
        for symbol in point_limits
    }
    parity_usable = all(
        point["close"] is not None
        and point["bar_end_utc"] is not None
        and point["age_seconds"] is not None
        and 0.0 <= float(point["age_seconds"]) <= point_limits[symbol]
        for symbol, point in points.items()
    )
    return {
        "schema": "gold.1oz-prospective-onset-context.v2",
        "authority": "prospective_research_only",
        "observed_at_utc": _utc(observed_at).isoformat(),
        "signal": signal,
        "macro": macro,
        "news": news,
        "exchange_parity": pair,
        "source_points": points,
        "source_closes": {
            symbol: points[symbol]["close"] for symbol in points
        },
        "timing_parity": {
            "usable": parity_usable,
            "maximum_age_seconds": point_limits,
            "reason": None if parity_usable else "source_bar_timing_mismatch",
        },
        "counterfactual_directions": directions,
        "total_cross_asset_neutral_short": bool(
            macro.get("total_direction_neutral")
            and any(direction == "down" for direction in directions.values())
        ),
        "slow_financing_neutral_short": bool(
            macro.get("total_direction_neutral")
            and any(direction == "down" for direction in directions.values())
            and isinstance(signal.get("h4"), Mapping)
            and dict(signal["h4"]).get("financing_clock") == "beyond_24h"
        ),
        "order_authority": "none",
        "submitted_orders": 0,
    }


def _instrument_outcomes(
    rows: Sequence[object], *, decision_at: datetime, decision_close: float
) -> dict[str, object] | None:
    bars = _complete_bars(rows, as_of=decision_at + timedelta(hours=24))
    times = [_bar_time(row) for row in bars]
    output = {}
    for hours in GOLD_ONSET_HORIZONS_HOURS:
        target = decision_at + timedelta(hours=hours)
        index = _latest_index(times, target)
        if index < 0 or target - times[index] > timedelta(hours=1):
            return None
        left = bisect.bisect_right(times, decision_at)
        window = bars[left : index + 1]
        if not window:
            return None
        close = _bar_value(bars[index], "close")
        high_delta = max(_bar_value(row, "high") - decision_close for row in window)
        low_delta = min(_bar_value(row, "low") - decision_close for row in window)
        output[str(hours)] = {
            "outcome_bar_end_utc": times[index].isoformat(),
            "close": close,
            "return_usd": close - decision_close,
            "return_pct": (close / decision_close - 1.0) * 100.0,
            "high_delta_usd": high_delta,
            "low_delta_usd": low_delta,
            "range_usd": max(_bar_value(row, "high") for row in window) - min(_bar_value(row, "low") for row in window),
        }
    return output


def _oriented_outcomes(
    raw: Mapping[str, object], direction: str | None
) -> dict[str, object] | None:
    if direction not in ("up", "down"):
        return None
    sign = 1.0 if direction == "up" else -1.0
    output = {}
    for hours in GOLD_ONSET_HORIZONS_HOURS:
        row = dict(raw[str(hours)])
        terminal = sign * float(row["return_usd"])
        mfe = max(sign * float(row["high_delta_usd"]), sign * float(row["low_delta_usd"]))
        mae = min(sign * float(row["high_delta_usd"]), sign * float(row["low_delta_usd"]))
        output[str(hours)] = {
            "return_usd": terminal,
            "mfe_usd": mfe,
            "mae_usd": mae,
            "financed": mfe >= GOLD_ONSET_FINANCING_USD,
            "giveback_usd": mfe - terminal,
            "reversal": terminal < 0.0,
        }
    return output


def advance_gold_onset_tape(
    ledger: LiveCalibrationLedger,
    *,
    context: Mapping[str, object],
    outcome_bars: Mapping[str, Sequence[object]],
    observed_at: datetime,
) -> dict[str, object]:
    """Freeze one unseen H4 state and settle prior 24-hour outcomes."""

    now = _utc(observed_at)
    signal = context.get("signal")
    if not isinstance(signal, Mapping) or not signal.get("usable"):
        checkpoint = ledger.checkpoint(
            evaluation_as_of=now,
            strategy_id="NO_TRADE",
            strategy_version=GOLD_ONSET_VERSION,
            trading_date=now.date().isoformat(),
            session="GOLD_24X5",
            status="NO_DATA",
            evidence={**dict(context), "order_authority": "none", "submitted_orders": 0},
            recorded_at=now,
        )
        return {"frozen": 0, "settled": 0, "checkpoint": checkpoint, "ledger": ledger.receipt()}
    decision_at = _utc(signal["decision_bar_end_utc"])
    context_payload = dict(context)
    identity = {
        "strategy_id": "NO_TRADE",
        "strategy_version": GOLD_ONSET_VERSION,
        "decision_as_of_utc": decision_at.isoformat(),
        "tape_fingerprint": calibration_fingerprint(
            {
                "decision_as_of_utc": decision_at.isoformat(),
                "strategy_version": GOLD_ONSET_VERSION,
                "config_fingerprint": GOLD_ONSET_CONFIG_FINGERPRINT,
                "source_closes": context_payload.get("source_closes"),
                "source_points": context_payload.get("source_points"),
            }
        ),
        "config_fingerprint": GOLD_ONSET_CONFIG_FINGERPRINT,
        "capital_sleeve": "gold-research-only",
    }
    records = list(ledger.records())
    identity_ids = {
        str(row["identity_id"])
        for row in records
        if row.get("kind") == "forecast"
    }
    frozen = 0
    if decision_at > GOLD_ONSET_PROSPECTIVE_START and calibration_fingerprint(identity) not in identity_ids:
        if now >= decision_at + timedelta(hours=24):
            raise ValueError("gold onset tape forbids late forecast backfill")
        directions = dict(context_payload.get("counterfactual_directions") or {})
        ledger.freeze(
            identity=identity,
            forecast={
                "decision": "NO_TRADE",
                "outcome_not_before_utc": (decision_at + timedelta(hours=24)).isoformat(),
                "pnl_distribution": {"research_only": True},
                "risk": {"max_loss": 0.0},
                "costs": {"modeled_round_trip_usd": GOLD_ONSET_FINANCING_USD},
                "fill_assumptions": {"orders": 0, "market_data_only": True},
            },
            context=context_payload,
            counterfactuals=[
                {"strategy_id": name, "decision": direction or "flat"}
                for name, direction in sorted(directions.items())
            ],
            gates={
                "selected_admissible": False,
                "macro_direction_authority": False,
                "news_direction_authority": False,
                "order_authority": "none",
            },
            recorded_at=now,
        )
        frozen = 1

    records = list(ledger.records())
    settled_ids = {
        str(row["forecast_id"])
        for row in records
        if row.get("kind") == "result"
    }
    settled = 0
    for forecast in records:
        if forecast.get("kind") != "forecast" or forecast.get("forecast_id") in settled_ids:
            continue
        prior_identity = forecast.get("identity")
        prior_context = forecast.get("context")
        if not isinstance(prior_identity, Mapping) or not isinstance(prior_context, Mapping) or prior_identity.get("strategy_version") != GOLD_ONSET_VERSION:
            continue
        prior_decision = _utc(prior_identity["decision_as_of_utc"])
        if now < prior_decision + timedelta(hours=24):
            continue
        instrument_results = {}
        complete = True
        for symbol in ("XAUUSD", "GC", "1OZ"):
            source = outcome_bars.get(symbol, ())
            source_context = prior_context.get("source_closes")
            closes = dict(source_context) if isinstance(source_context, Mapping) else {}
            decision_close = _finite(closes.get(symbol))
            if decision_close is None:
                complete = False
                break
            result = _instrument_outcomes(source, decision_at=prior_decision, decision_close=decision_close)
            if result is None:
                complete = False
                break
            instrument_results[symbol] = result
        if not complete:
            continue
        directions = dict(prior_context.get("counterfactual_directions") or {})
        oriented = {
            name: {
                symbol: _oriented_outcomes(instrument_results[symbol], direction)
                for symbol in instrument_results
            }
            for name, direction in directions.items()
        }
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": (prior_decision + timedelta(hours=24)).isoformat(),
                "instruments": instrument_results,
                "counterfactuals": oriented,
            },
            drift={"signal": "prospective_counterfactual_only", "economic": "no_orders"},
            verdict="HOLD",
            settled_at=now,
        )
        settled_ids.add(str(forecast["forecast_id"]))
        settled += 1

    checkpoint = ledger.checkpoint(
        evaluation_as_of=now,
        strategy_id="NO_TRADE",
        strategy_version=GOLD_ONSET_VERSION,
        trading_date=now.date().isoformat(),
        session="GOLD_24X5",
        status="EVALUATED",
        evidence={**context_payload, "order_authority": "none", "submitted_orders": 0},
        recorded_at=now,
    )
    return {
        "authority": "prospective_research_only",
        "submitted_orders": 0,
        "frozen": frozen,
        "settled": settled,
        "checkpoint": checkpoint,
        "ledger": ledger.receipt(),
    }
