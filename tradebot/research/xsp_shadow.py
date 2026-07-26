"""Causal XSP directional-observer calibration on one close-aligned bar tape."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from ..backtest.quotes import QuoteSnapshot
from ..engines.directional_impulse import DirectionalTurnPolicy
from ..engines.market import (
    xsp_rth_evaluation_slots,
    xsp_session_label_et,
    xsp_trading_date,
)
from ..spot.evaluator_common import BarLike, SpotSignalSnapshot
from ..spot_engine import SpotSignalEvaluator
from ..time_utils import NaiveTsModeInput, UTC, to_et, to_utc_naive
from .live_calibration import (
    XSP_DIRECTIONAL_OBSERVER_VERSION,
    LiveCalibrationLedger,
    calibration_fingerprint,
)
from .xsp_candidate import (
    XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
    XSP_OPENING_EDGE_VERSION,
    xsp_opening_edge_candidate_equity,
    xsp_opening_edge_run_start,
    xsp_opening_edge_selected_equity,
)
from .xsp_benchmarks import xsp_profitability_policy_from_selected_run
from .xsp_context import (
    xsp_fundamental_context_at,
    xsp_option_context_at,
)


XSP_DIRECTIONAL_HORIZONS_MINUTES = (30, 60, 120)
XSP_DIRECTIONAL_FRICTION_POINTS = 0.10
XSP_DIRECTIONAL_HISTORY_DURATION = "1 W"


def _utc(ts: datetime, *, naive_ts_mode: NaiveTsModeInput) -> datetime:
    return to_utc_naive(ts, naive_ts_mode=naive_ts_mode).replace(tzinfo=UTC)


def _update_tape_hash(
    hasher,
    bar: BarLike,
    *,
    naive_ts_mode: NaiveTsModeInput,
) -> None:
    hasher.update(
        json.dumps(
            [
                _utc(bar.ts, naive_ts_mode=naive_ts_mode).isoformat(),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            ],
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def xsp_bar_tape_fingerprint(
    bars: Iterable[BarLike],
    *,
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> str:
    hasher = hashlib.sha256()
    for bar in bars:
        _update_tape_hash(hasher, bar, naive_ts_mode=naive_ts_mode)
    return hasher.hexdigest()


def freeze_xsp_directional_observation(
    ledger: LiveCalibrationLedger,
    *,
    snapshot: SpotSignalSnapshot,
    tape_fingerprint: str,
    recorded_at: datetime,
    evidence_mode: str,
    option_context: Mapping[str, object] | None = None,
    fundamental_context: Mapping[str, object] | None = None,
    naive_ts_mode: NaiveTsModeInput = "utc",
    horizons_minutes: Sequence[int] = XSP_DIRECTIONAL_HORIZONS_MINUTES,
) -> list[dict[str, object]]:
    """Freeze one close-aligned directional turn without order authority."""

    impulse = snapshot.directional_impulse
    direction = impulse.turn_event if impulse is not None else None
    if direction not in ("up", "down"):
        return []

    decision_at = _utc(snapshot.bar_ts, naive_ts_mode=naive_ts_mode)
    policy = DirectionalTurnPolicy()
    existing = {}
    for row in ledger.records():
        identity = row.get("identity")
        forecast = row.get("forecast")
        if (
            row.get("kind") != "forecast"
            or not isinstance(identity, Mapping)
            or not isinstance(forecast, Mapping)
            or identity.get("strategy_version") != XSP_DIRECTIONAL_OBSERVER_VERSION
        ):
            continue
        existing.setdefault(
            (
                str(identity.get("decision_as_of_utc") or ""),
                str(forecast.get("outcome_not_before_utc") or ""),
            ),
            row,
        )
    rows = []
    for horizon in sorted({int(value) for value in horizons_minutes if int(value) > 0}):
        config = {
            "observer": XSP_DIRECTIONAL_OBSERVER_VERSION,
            "policy": policy.as_payload(),
            "horizon_minutes": horizon,
            "entry": "next_bar_open",
            "exit": "first_bar_close_at_or_after_horizon",
            "unit": "$1_per_XSP_point",
            "friction_points": XSP_DIRECTIONAL_FRICTION_POINTS,
            "option_context": (
                str(option_context.get("source"))
                if option_context is not None
                else None
            ),
        }
        context = {
            "evidence_mode": str(evidence_mode),
            "symbol": "XSP",
            "session": xsp_session_label_et(decision_at),
            "decision_close": float(snapshot.close),
            "directional_impulse": impulse.as_payload(),
            "entry_control": snapshot.entry_control_trace(),
        }
        if option_context is not None:
            context["option_parity"] = dict(option_context)
        context["fundamental_pressure"] = dict(
            fundamental_context
            or {
                "source": "causal_news",
                "authority": "observation_only",
                "usable": False,
                "reason": "not_recorded_at_decision",
            }
        )
        identity = {
            "strategy_id": "NO_TRADE",
            "strategy_version": XSP_DIRECTIONAL_OBSERVER_VERSION,
            "decision_as_of_utc": decision_at.isoformat(),
            "tape_fingerprint": str(tape_fingerprint),
            "config_fingerprint": calibration_fingerprint(config),
            "capital_sleeve": "xsp-directional-unit",
        }
        slot = (
            decision_at.isoformat(),
            (decision_at + timedelta(minutes=horizon)).isoformat(),
        )
        if slot in existing:
            rows.append(existing[slot])
            continue
        row = ledger.freeze(
            identity=identity,
            forecast={
                "decision": "NO_TRADE",
                "outcome_not_before_utc": (
                    decision_at + timedelta(minutes=horizon)
                ).isoformat(),
                "pnl_distribution": {"status": "unavailable_unpromoted_observer"},
                "risk": {"selected_max_loss_points": 0.0},
                "costs": {
                    "selected_points": 0.0,
                    "counterfactual_round_trip_points": (
                        XSP_DIRECTIONAL_FRICTION_POINTS
                    ),
                },
                "fill_assumptions": {
                    "selected": "none",
                    "counterfactual": "synthetic_next_bar_open",
                    "broker_fill": False,
                },
            },
            context=context,
            counterfactuals=[
                {
                    "strategy_id": "directional_impulse.observer",
                    "decision": str(direction).upper(),
                    "authority": "observation_only",
                    "eligible": False,
                }
            ],
            gates={
                "selected_admissible": False,
                "selected_reason": "no_xsp_directional_champion",
                "counterfactual_authority": "observation_only",
                "promotion_from_single_forecast": False,
            },
            recorded_at=recorded_at,
        )
        existing[slot] = row
        rows.append(row)
    return rows


def settle_xsp_directional_observations(
    ledger: LiveCalibrationLedger,
    bars: Sequence[BarLike],
    *,
    settled_at: datetime,
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> list[dict[str, object]]:
    """Settle forecasts from canonical bars timestamped at their close."""

    ordered = sorted(
        bars,
        key=lambda bar: _utc(bar.ts, naive_ts_mode=naive_ts_mode),
    )
    timed = [(_utc(bar.ts, naive_ts_mode=naive_ts_mode), bar) for bar in ordered]
    records = list(ledger.records())
    settled_ids = {
        str(row["forecast_id"]) for row in records if row.get("kind") == "result"
    }
    results = []
    for forecast in records:
        forecast_id = str(forecast.get("forecast_id") or "")
        identity = forecast.get("identity")
        if (
            forecast.get("kind") != "forecast"
            or forecast_id in settled_ids
            or not isinstance(identity, dict)
            or identity.get("strategy_version") != XSP_DIRECTIONAL_OBSERVER_VERSION
        ):
            continue
        decision_at = datetime.fromisoformat(
            str(identity["decision_as_of_utc"]).replace("Z", "+00:00")
        )
        target = datetime.fromisoformat(
            str(dict(forecast["forecast"])["outcome_not_before_utc"]).replace(
                "Z", "+00:00"
            )
        )
        entry = next(
            (
                (ts, bar)
                for ts, bar in timed
                if ts == decision_at + timedelta(minutes=5)
            ),
            None,
        )
        outcome = next(
            ((ts, bar) for ts, bar in timed if ts == target),
            None,
        )
        decision_day = to_et(decision_at).date()
        if (
            entry is None
            or outcome is None
            or to_et(entry[0]).date() != decision_day
            or to_et(outcome[0]).date() != decision_day
        ):
            continue
        counterfactuals = forecast.get("counterfactuals")
        direction = (
            str(counterfactuals[0].get("decision") or "").strip().lower()
            if isinstance(counterfactuals, list)
            and counterfactuals
            and isinstance(counterfactuals[0], dict)
            else ""
        )
        if direction not in ("up", "down"):
            continue
        entry_price = float(entry[1].open)
        exit_price = float(outcome[1].close)
        gross = (exit_price - entry_price) * (1.0 if direction == "up" else -1.0)
        net = gross - XSP_DIRECTIONAL_FRICTION_POINTS
        results.append(
            ledger.settle(
                forecast_id=forecast_id,
                observed={
                    "outcome_as_of_utc": outcome[0].isoformat(),
                    "shadow_pnl": 0.0,
                    "package_pnl": None,
                    "leg_pnl": None,
                    "account_pnl": None,
                    "counterfactuals": [
                        {
                            "strategy_id": "directional_impulse.observer",
                            "direction": direction,
                            "entry_as_of_utc": entry[0].isoformat(),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_points": gross,
                            "net_points": net,
                        }
                    ],
                },
                drift={
                    "data": "same_tape",
                    "decision": "selected_no_trade_preserved",
                    "pricing": "not_applicable",
                    "execution": "synthetic_observer_no_broker_fill",
                    "economic": {
                        "selected_points": 0.0,
                        "counterfactual_points": net,
                    },
                    "safety": "no_order_authority",
                },
                verdict="HOLD",
                settled_at=settled_at,
            )
        )
    return results


def advance_xsp_directional_shadow(
    ledger: LiveCalibrationLedger,
    bars: Sequence[BarLike],
    *,
    observed_at: datetime,
    option_snapshots: Sequence[QuoteSnapshot] = (),
    news_snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    naive_ts_mode: NaiveTsModeInput = "utc",
    freeze_new: bool = True,
) -> dict[str, object]:
    """Freeze only still-forward turns from close-aligned bars, then settle."""

    observed_utc = _utc(observed_at, naive_ts_mode=naive_ts_mode)
    ordered = sorted(
        (
            bar
            for bar in bars
            if _utc(bar.ts, naive_ts_mode=naive_ts_mode) <= observed_utc
        ),
        key=lambda bar: _utc(bar.ts, naive_ts_mode=naive_ts_mode),
    )
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "directional_impulse",
            "regime_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
        naive_ts_mode=naive_ts_mode,
    )
    tape_hasher = hashlib.sha256()
    known_forecasts = {
        str(row["forecast_id"])
        for row in ledger.records()
        if row.get("kind") == "forecast"
    }
    frozen = 0
    for bar in ordered:
        _update_tape_hash(tape_hasher, bar, naive_ts_mode=naive_ts_mode)
        snapshot = evaluator.update_signal_bar(bar)
        if snapshot is None:
            continue
        decision_at = _utc(snapshot.bar_ts, naive_ts_mode=naive_ts_mode)
        if not freeze_new or not (
            decision_at
            <= observed_utc
            < decision_at + timedelta(minutes=min(XSP_DIRECTIONAL_HORIZONS_MINUTES))
        ):
            continue
        for row in freeze_xsp_directional_observation(
            ledger,
            snapshot=snapshot,
            tape_fingerprint=tape_hasher.hexdigest(),
            recorded_at=observed_utc,
            evidence_mode="forward_broker_history",
            option_context=xsp_option_context_at(
                option_snapshots,
                decision_at=decision_at,
            ),
            fundamental_context=xsp_fundamental_context_at(
                news_snapshot,
                decision_at=decision_at,
            ),
            naive_ts_mode=naive_ts_mode,
        ):
            forecast_id = str(row["forecast_id"])
            if forecast_id not in known_forecasts:
                known_forecasts.add(forecast_id)
                frozen += 1

    settled = settle_xsp_directional_observations(
        ledger,
        ordered,
        settled_at=observed_utc,
        naive_ts_mode=naive_ts_mode,
    )
    receipt = ledger.receipt()
    latest_close = (
        _utc(ordered[-1].ts, naive_ts_mode=naive_ts_mode) if ordered else None
    )
    return {
        **receipt,
        "processed_bars": len(ordered),
        "new_forecasts": frozen,
        "new_results": len(settled),
        "freeze_new": bool(freeze_new),
        "option_snapshots": len(option_snapshots),
        "latest_bar_close_utc": (
            latest_close.isoformat() if latest_close is not None else None
        ),
        "latest_bar_age_sec": (
            max(0.0, (observed_utc - latest_close).total_seconds())
            if latest_close is not None
            else None
        ),
    }


async def advance_xsp_shadow_from_ibkr(
    ledger: LiveCalibrationLedger,
    *,
    client,
    observed_at: datetime,
    duration_str: str = XSP_DIRECTIONAL_HISTORY_DURATION,
    option_snapshots: Sequence[QuoteSnapshot] = (),
    news_snapshot: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    selected_run: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Advance the non-submitting shadow from canonical IBKR XSP history."""

    from ib_insync import Index

    from ..chart_data.history import normalize_bars_to_close
    from ..utils.bar_utils import trim_incomplete_last_bar

    observed_utc = _utc(observed_at, naive_ts_mode="et")
    observed_et = to_et(observed_at)
    session = xsp_session_label_et(observed_at)
    trading_day = xsp_trading_date(observed_at)
    slots = xsp_rth_evaluation_slots(trading_day) if trading_day else ()
    after_rth_close = bool(
        slots and observed_et > slots[-1] + timedelta(seconds=90)
    )
    skip_reason = (
        "unsupported_session"
        if session is not None and session != "RTH"
        else "closed_calendar"
        if not slots
        else "after_rth_close"
        if after_rth_close
        else None
    )
    if skip_reason is not None:
        evaluation_status = (
            "UNSUPPORTED_SESSION"
            if skip_reason == "unsupported_session"
            else "CLOSED"
        )
        checkpoint_session = (
            session if evaluation_status == "UNSUPPORTED_SESSION" else "CLOSED"
        )
        ledger.checkpoint(
            evaluation_as_of=observed_utc,
            strategy_id="NO_TRADE",
            strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
            trading_date=trading_day.isoformat() if trading_day else None,
            session=checkpoint_session,
            status=evaluation_status,
            evidence={
                "cash_tape_fingerprint": xsp_bar_tape_fingerprint(
                    (),
                    naive_ts_mode="et",
                ),
                "complete_close_aligned_bars": 0,
                "latest_bar_close_utc": None,
                "latest_bar_age_sec": None,
                "cash_history_fresh": False,
                "option_snapshots": len(option_snapshots),
                "broker_request_skipped": skip_reason,
                "order_authority": "none",
            },
            recorded_at=observed_utc,
        )
        return {
            **ledger.receipt(),
            "status": "ok",
            "evaluation_status": evaluation_status,
            "session": None if evaluation_status == "CLOSED" else session,
            "freshness_ok": False,
            "contract": None,
            "raw_bars": 0,
            "complete_close_aligned_bars": 0,
            "historical_request": None,
            "broker_request_skipped": skip_reason,
        }

    qualified = await client.qualify_proxy_contracts(Index("XSP", "CBOE", "USD"))
    contract = next(
        (
            row
            for row in qualified
            if int(getattr(row, "conId", 0) or 0) > 0
            and str(getattr(row, "secType", "") or "").strip().upper() == "IND"
        ),
        None,
    )
    if contract is None:
        raise RuntimeError("IBKR did not qualify XSP as IND/CBOE")

    observed_et_naive = observed_et.replace(tzinfo=None)
    raw_bars = await client.historical_bars_ohlcv(
        contract,
        duration_str=str(duration_str),
        bar_size="5 mins",
        use_rth=True,
        what_to_show="TRADES",
        cache_ttl_sec=0.0,
    )
    complete = trim_incomplete_last_bar(
        list(raw_bars),
        bar_size="5 mins",
        now_ref=observed_et_naive,
    )
    bars = normalize_bars_to_close(
        complete,
        symbol="XSP",
        bar_size="5 mins",
        use_rth=True,
        naive_ts_mode="et",
    )
    latest_close = _utc(bars[-1].ts, naive_ts_mode="et") if bars else None
    latest_age_sec = (
        max(0.0, (observed_utc - latest_close).total_seconds())
        if latest_close is not None
        else None
    )
    freshness_ok = bool(
        session == "RTH"
        and latest_age_sec is not None
        and latest_age_sec <= 600.0
    )
    evaluation_status = (
        "CLOSED"
        if session is None
        else "UNSUPPORTED_SESSION"
        if session != "RTH"
        else "NO_DATA"
        if latest_age_sec is None
        else "EVALUATED"
        if freshness_ok
        else "STALE_DATA"
    )
    historical_request = client.last_historical_request(contract)
    receipt = advance_xsp_directional_shadow(
        ledger,
        bars,
        observed_at=observed_at,
        option_snapshots=option_snapshots,
        news_snapshot=news_snapshot,
        naive_ts_mode="et",
        freeze_new=freshness_ok,
    )
    run_started_at = (
        xsp_opening_edge_run_start(
            tuple(ledger.records()),
            observed_at=observed_utc,
        )
        if freshness_ok
        else None
    )
    candidate_equity = (
        xsp_opening_edge_candidate_equity(
            bars,
            run_started_at=run_started_at,
            observed_at=observed_utc,
            naive_ts_mode="et",
        )
        if freshness_ok
        else None
    )
    selected_equity = None
    if candidate_equity is not None and selected_run is not None:
        policy = xsp_profitability_policy_from_selected_run(selected_run)
        if (
            policy.strategy_id != XSP_OPENING_EDGE_VERSION
            or policy.strategy_version != XSP_OPENING_EDGE_VERSION
            or policy.config_fingerprint != XSP_OPENING_EDGE_CONFIG_FINGERPRINT
            or datetime.fromisoformat(
                str(selected_run["selected_at_utc"]).replace("Z", "+00:00")
            )
            > run_started_at
        ):
            raise ValueError("selected run does not own Opening Edge v1")
        selected_equity = xsp_opening_edge_selected_equity(
            candidate_equity,
            run_id=policy.run_id,
            capital_sleeve=policy.capital_sleeve,
        )
    ledger.checkpoint(
        evaluation_as_of=observed_utc,
        strategy_id=XSP_OPENING_EDGE_VERSION,
        strategy_version=XSP_OPENING_EDGE_VERSION,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=session or "CLOSED",
        status=evaluation_status,
        evidence={
            "candidate_equity": candidate_equity,
            **(
                {"selected_equity": selected_equity}
                if selected_equity is not None
                else {}
            ),
            "cash_tape_fingerprint": xsp_bar_tape_fingerprint(
                bars,
                naive_ts_mode="et",
            ),
            "cash_history_fresh": freshness_ok,
            "order_authority": "none",
        },
        recorded_at=observed_utc,
    )
    ledger.checkpoint(
        evaluation_as_of=observed_utc,
        strategy_id="NO_TRADE",
        strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
        trading_date=trading_day.isoformat() if trading_day else None,
        session=session or "CLOSED",
        status=evaluation_status,
        evidence={
            "cash_tape_fingerprint": xsp_bar_tape_fingerprint(
                bars,
                naive_ts_mode="et",
            ),
            "complete_close_aligned_bars": len(bars),
            "latest_bar_close_utc": (
                latest_close.isoformat() if latest_close is not None else None
            ),
            "latest_bar_age_sec": latest_age_sec,
            "cash_history_fresh": freshness_ok,
            "option_snapshots": len(option_snapshots),
            "order_authority": "none",
        },
        recorded_at=observed_utc,
    )
    receipt.update(ledger.receipt())
    return {
        **receipt,
        "status": "ok" if bars else "no_bars",
        "evaluation_status": evaluation_status,
        "session": session,
        "freshness_ok": freshness_ok,
        "contract": {
            "con_id": int(getattr(contract, "conId", 0) or 0),
            "symbol": str(getattr(contract, "symbol", "") or ""),
            "sec_type": str(getattr(contract, "secType", "") or ""),
            "exchange": str(getattr(contract, "exchange", "") or ""),
            "currency": str(getattr(contract, "currency", "") or ""),
        },
        "raw_bars": len(raw_bars),
        "complete_close_aligned_bars": len(bars),
        "historical_request": historical_request,
        "opening_edge_candidate": candidate_equity,
        "selected_equity": selected_equity,
    }


def replay_xsp_directional_shadow(
    ledger: LiveCalibrationLedger,
    bars: Sequence[BarLike],
    *,
    option_snapshots: Sequence[QuoteSnapshot] = (),
    news_snapshots: Sequence[Mapping[str, object]] = (),
    naive_ts_mode: NaiveTsModeInput = "utc",
) -> dict[str, object]:
    ordered = sorted(
        bars,
        key=lambda bar: _utc(bar.ts, naive_ts_mode=naive_ts_mode),
    )
    evaluator = SpotSignalEvaluator(
        strategy={
            "entry_signal": "directional_impulse",
            "regime_mode": "off",
        },
        filters=None,
        bar_size="5 mins",
        use_rth=True,
        naive_ts_mode=naive_ts_mode,
    )
    tape_hasher = hashlib.sha256()
    for bar in ordered:
        _update_tape_hash(tape_hasher, bar, naive_ts_mode=naive_ts_mode)
        snapshot = evaluator.update_signal_bar(bar)
        if snapshot is None:
            continue
        freeze_xsp_directional_observation(
            ledger,
            snapshot=snapshot,
            tape_fingerprint=tape_hasher.hexdigest(),
            recorded_at=_utc(bar.ts, naive_ts_mode=naive_ts_mode),
            evidence_mode="historical_replay",
            option_context=(
                xsp_option_context_at(
                    option_snapshots,
                    decision_at=_utc(snapshot.bar_ts, naive_ts_mode=naive_ts_mode),
                )
                if option_snapshots
                else None
            ),
            fundamental_context=(
                xsp_fundamental_context_at(
                    news_snapshots,
                    decision_at=_utc(
                        snapshot.bar_ts,
                        naive_ts_mode=naive_ts_mode,
                    ),
                )
                if news_snapshots
                else None
            ),
            naive_ts_mode=naive_ts_mode,
        )
    if ordered:
        settle_xsp_directional_observations(
            ledger,
            ordered,
            settled_at=_utc(ordered[-1].ts, naive_ts_mode=naive_ts_mode),
            naive_ts_mode=naive_ts_mode,
        )
    return ledger.receipt()


def main(argv: Sequence[str] | None = None) -> int:
    from .xsp_shadow_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
