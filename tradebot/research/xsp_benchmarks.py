"""Observation-only XSP benchmarks over the immutable calibration ledger."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
import math
import statistics

from ..time_utils import to_et
from .live_calibration import (
    XSP_DIRECTIONAL_OBSERVER_VERSION,
    LiveCalibrationLedger,
    XspProfitabilityPolicy,
    calibration_fingerprint,
)
from .xsp_candidate import (
    XSP_DIRECTIONAL_SHADOW_POLICY,
    XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
    XSP_OPENING_EDGE_VERSION,
)
from .xsp_context import XSP_PREOPEN_RESEARCH_VERSION


XSP_OPTION_PARITY_OBSERVER_VERSION = "xsp.option-parity-participation.v2"
XSP_OPTION_PARITY_CANDIDATE_VERSION = "xsp.option-parity-aligned-candidate.v1"
XSP_OPTION_LIQUIDITY_CANDIDATE_VERSION = (
    "xsp.option-parity-aligned-liquidity-candidate.v1"
)
XSP_SHADOW_RECOMMENDATION_VERSION = "xsp.shadow-candidate-recommendation.v1"
XSP_SELECTED_SHADOW_RUN_VERSION = "xsp.selected-shadow-run.v1"
XSP_OPTION_PARITY_PRIMARY_HORIZON_MINUTES = 60
XSP_OPTION_PARITY_MIN_USABLE_PAIRS = 30
XSP_OPTION_PARITY_MIN_COMPLETE_SESSIONS = 5
XSP_OPTION_LIQUIDITY_COHORTS = (
    "strengthening",
    "stable",
    "weakening",
    "mixed",
    "unavailable",
)
XSP_PREOPEN_PARITY_COHORTS = (
    "aligned_all",
    "opposed_all",
    "reversal_into",
    "mixed",
    "unavailable",
)
XSP_FUNDAMENTAL_OBSERVER_VERSION = "xsp.fundamental-defensive-observer.v1"
XSP_FUNDAMENTAL_PRIMARY_HORIZON_MINUTES = 60
XSP_FUNDAMENTAL_MIN_IMPACT = 70
XSP_FUNDAMENTAL_MIN_CONFIDENCE = 0.80


def xsp_opening_edge_shadow_recommendation() -> dict[str, object]:
    """Preregister the frozen research crown for shadow-only selection."""

    candidate = {
        "schema": XSP_OPENING_EDGE_VERSION,
        "eligible": True,
        "evidence_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        "failed_checks": (),
        "strategy_version": XSP_OPENING_EDGE_VERSION,
        "config_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        "capital_sleeve": "xsp-directional-unit",
    }
    recommendation = {
        "schema": XSP_SHADOW_RECOMMENDATION_VERSION,
        "authority": "recommendation_only",
        "scope": "selected_shadow_only",
        "verdict": "PROMOTE",
        "recommended_candidate_schema": XSP_OPENING_EDGE_VERSION,
        "selection_authority": "none_until_explicit_run_freeze",
        "order_authority": "none",
        "open_position_strategy_switch_allowed": False,
        "profitability_clock_started": False,
        "preregistered_selected_run_policy": dict(
            XSP_DIRECTIONAL_SHADOW_POLICY
        ),
        "source_ledger_sha256": calibration_fingerprint(
            {
                "strategy_id": XSP_OPENING_EDGE_VERSION,
                "config_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
                "authority": "frozen_research_crown",
            }
        ),
        "candidates": (candidate,),
    }
    recommendation["fingerprint"] = calibration_fingerprint(recommendation)
    return recommendation


def xsp_selected_shadow_run(
    ledger: LiveCalibrationLedger,
    recommendation: Mapping[str, object],
    *,
    run_id: str,
    strategy_version: str,
    config_fingerprint: str,
    capital_sleeve: str,
    selected_at: datetime | str,
) -> dict[str, object]:
    """Freeze one eligible recommendation without granting order authority."""
    unsigned = dict(recommendation)
    recommendation_fingerprint = str(unsigned.pop("fingerprint", ""))
    candidate_schema = recommendation.get("recommended_candidate_schema")
    expected = (
        xsp_opening_edge_shadow_recommendation()
        if candidate_schema == XSP_OPENING_EDGE_VERSION
        else xsp_option_parity_participation_benchmark(ledger)[
            "shadow_recommendation"
        ]
    )
    errors = []
    if (
        recommendation.get("schema") != XSP_SHADOW_RECOMMENDATION_VERSION
        or recommendation.get("authority") != "recommendation_only"
        or calibration_fingerprint(unsigned) != recommendation_fingerprint
    ):
        errors.append("invalid_recommendation")
    if recommendation != expected:
        errors.append("recommendation_not_current_for_ledger")
    candidates = recommendation.get("candidates")
    candidate = next(
        (
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("schema") == candidate_schema
        ),
        None,
    ) if isinstance(candidates, Sequence) else None
    if (
        recommendation.get("verdict") != "PROMOTE"
        or candidate_schema not in {
            XSP_OPENING_EDGE_VERSION,
            XSP_OPTION_PARITY_CANDIDATE_VERSION,
            XSP_OPTION_LIQUIDITY_CANDIDATE_VERSION,
        }
        or not isinstance(candidate, Mapping)
        or candidate.get("eligible") is not True
        or candidate.get("failed_checks") not in ((), [])
    ):
        errors.append("recommendation_not_eligible")
    if recommendation.get("preregistered_selected_run_policy") != (
        XSP_DIRECTIONAL_SHADOW_POLICY
    ):
        errors.append("risk_policy_drift")
    identity = {
        "requested_run_id": str(run_id).strip(),
        "strategy_id": str(candidate_schema or ""),
        "strategy_version": str(strategy_version).strip(),
        "config_fingerprint": str(config_fingerprint).strip(),
        "capital_sleeve": str(capital_sleeve).strip(),
    }
    if not all(identity.values()):
        errors.append("incomplete_run_identity")
    if candidate_schema == XSP_OPENING_EDGE_VERSION and (
        not isinstance(candidate, Mapping)
        or identity["strategy_version"] != candidate.get("strategy_version")
        or identity["config_fingerprint"] != candidate.get("config_fingerprint")
        or identity["capital_sleeve"] != candidate.get("capital_sleeve")
    ):
        errors.append("candidate_identity_drift")
    try:
        parsed = (
            selected_at
            if isinstance(selected_at, datetime)
            else datetime.fromisoformat(str(selected_at).replace("Z", "+00:00"))
        )
        if parsed.tzinfo is None:
            raise ValueError
        selected_at_utc = parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        errors.append("invalid_selection_timestamp")
        selected_at_utc = ""
    if errors:
        raise ValueError(
            "XSP shadow selection refused: " + ", ".join(sorted(set(errors)))
        )
    selection = {
        "schema": XSP_SELECTED_SHADOW_RUN_VERSION,
        "authority": "selected_shadow_evidence_only",
        "selected_at_utc": selected_at_utc,
        **identity,
        "recommendation_fingerprint": recommendation_fingerprint,
        "source_ledger_sha256": recommendation["source_ledger_sha256"],
        "candidate_evidence_fingerprint": candidate["evidence_fingerprint"],
        "risk_policy": dict(XSP_DIRECTIONAL_SHADOW_POLICY),
        "order_authority": "none",
        "open_position_strategy_switch_allowed": False,
        "profitability_clock_started": False,
    }
    selection["selection_id"] = calibration_fingerprint(selection)
    selection["run_id"] = selection["selection_id"]
    return selection


def xsp_profitability_policy_from_selected_run(
    selection: Mapping[str, object],
) -> XspProfitabilityPolicy:
    """Project the only admissible profitability policy from a frozen selection."""

    unsigned = dict(selection)
    selection_id = str(unsigned.pop("selection_id", ""))
    run_id = str(unsigned.pop("run_id", ""))
    errors = []
    if (
        selection.get("schema") != XSP_SELECTED_SHADOW_RUN_VERSION
        or selection.get("authority") != "selected_shadow_evidence_only"
        or not selection_id
        or run_id != selection_id
        or calibration_fingerprint(unsigned) != selection_id
    ):
        errors.append("invalid_selection")
    if (
        selection.get("order_authority") != "none"
        or selection.get("open_position_strategy_switch_allowed") is not False
        or selection.get("profitability_clock_started") is not False
    ):
        errors.append("selection_authority_drift")
    policy = selection.get("risk_policy")
    if policy != XSP_DIRECTIONAL_SHADOW_POLICY:
        errors.append("risk_policy_drift")
    required_identity = (
        "requested_run_id",
        "strategy_id",
        "strategy_version",
        "config_fingerprint",
        "capital_sleeve",
    )
    if any(not str(selection.get(field, "")).strip() for field in required_identity):
        errors.append("incomplete_run_identity")
    if errors:
        raise ValueError(
            "XSP profitability policy refused: " + ", ".join(sorted(set(errors)))
        )
    return XspProfitabilityPolicy(
        run_id=selection_id,
        strategy_id=str(selection["strategy_id"]),
        strategy_version=str(selection["strategy_version"]),
        config_fingerprint=str(selection["config_fingerprint"]),
        capital_sleeve=str(selection["capital_sleeve"]),
        max_drawdown_points=float(policy["max_drawdown_points"]),
        max_session_loss_points=float(policy["max_session_loss_points"]),
        minimum_week_closed_trades=int(policy["minimum_week_closed_trades"]),
        maximum_top_five_win_share=float(policy["maximum_top_five_win_share"]),
        slot_tolerance_seconds=float(policy["slot_tolerance_seconds"]),
    )


def xsp_fundamental_defensive_benchmark(
    ledger: LiveCalibrationLedger,
) -> dict[str, object]:
    """Score one preregistered defensive veto without granting trade authority."""

    pairs = []
    for settled in ledger.settled_directional_pairs(
        horizon_minutes=XSP_FUNDAMENTAL_PRIMARY_HORIZON_MINUTES,
    ):
        context = settled["context"]
        news = context.get("fundamental_pressure")
        if not isinstance(news, Mapping):
            continue
        direction = str(settled["direction"])
        try:
            news_direction = int(news.get("direction", 0))
            impact = int(news.get("impact", -1))
            confidence = float(news.get("confidence", -1.0))
        except (TypeError, ValueError):
            continue
        opposing = (direction == "up" and news_direction < 0) or (
            direction == "down" and news_direction > 0
        )
        vetoed = bool(
            news.get("source") == "causal_news"
            and news.get("authority") == "observation_only"
            and news.get("usable") is True
            and opposing
            and impact >= XSP_FUNDAMENTAL_MIN_IMPACT
            and confidence >= XSP_FUNDAMENTAL_MIN_CONFIDENCE
        )
        baseline_points = float(settled["ta_points"])
        defended_points = 0.0 if vetoed else baseline_points
        pairs.append(
            {
                "forecast_id": str(settled["forecast_id"]),
                "decision_as_of_utc": settled["decision_at"].isoformat(),
                "direction": direction,
                "evidence_mode": settled["evidence_mode"],
                "prospective": settled["prospective"],
                "news_snapshot_fingerprint": news.get("snapshot_fingerprint"),
                "news_signal_as_of_utc": news.get("signal_as_of_utc"),
                "news_usable": news.get("usable") is True,
                "opposing": opposing,
                "vetoed": vetoed,
                "ta_points": baseline_points,
                "defended_points": defended_points,
                "delta_points": defended_points - baseline_points,
            }
        )

    vetoes = [row for row in pairs if row["vetoed"]]
    prospective = [row for row in pairs if row["prospective"]]
    prospective_vetoes = [row for row in prospective if row["vetoed"]]
    ta_points = sum(float(row["ta_points"]) for row in pairs)
    defended_points = sum(float(row["defended_points"]) for row in pairs)
    prospective_ta_points = sum(float(row["ta_points"]) for row in prospective)
    prospective_defended_points = sum(
        float(row["defended_points"]) for row in prospective
    )
    return {
        "schema": XSP_FUNDAMENTAL_OBSERVER_VERSION,
        "authority": "observation_only",
        "promotion_eligible": False,
        "primary_horizon_minutes": XSP_FUNDAMENTAL_PRIMARY_HORIZON_MINUTES,
        "policy": {
            "action": "opposing_signal_veto_only",
            "min_impact": XSP_FUNDAMENTAL_MIN_IMPACT,
            "min_confidence": XSP_FUNDAMENTAL_MIN_CONFIDENCE,
            "missing_or_ineligible": "unchanged",
            "prospective_evidence_mode": "forward_broker_history",
        },
        "source_ledger_sha256": ledger.receipt()["sha256"],
        "pair_fingerprint": calibration_fingerprint(pairs),
        "pairs": len(pairs),
        "prospective_pairs": len(prospective),
        "mechanics_pairs": len(pairs) - len(prospective),
        "news_usable_pairs": sum(bool(row["news_usable"]) for row in pairs),
        "vetoes": len(vetoes),
        "avoided_loss_points": sum(
            -float(row["ta_points"])
            for row in vetoes
            if float(row["ta_points"]) < 0
        ),
        "foregone_gain_points": sum(
            float(row["ta_points"])
            for row in vetoes
            if float(row["ta_points"]) > 0
        ),
        "ta_observer_points": ta_points,
        "defended_observer_points": defended_points,
        "paired_delta_points": defended_points - ta_points,
        "prospective_pair_fingerprint": calibration_fingerprint(prospective),
        "prospective_news_usable_pairs": sum(
            bool(row["news_usable"]) for row in prospective
        ),
        "prospective_vetoes": len(prospective_vetoes),
        "prospective_avoided_loss_points": sum(
            -float(row["ta_points"])
            for row in prospective_vetoes
            if float(row["ta_points"]) < 0
        ),
        "prospective_foregone_gain_points": sum(
            float(row["ta_points"])
            for row in prospective_vetoes
            if float(row["ta_points"]) > 0
        ),
        "prospective_ta_observer_points": prospective_ta_points,
        "prospective_defended_observer_points": prospective_defended_points,
        "prospective_paired_delta_points": (
            prospective_defended_points - prospective_ta_points
        ),
        "economic_interpretation": "overlapping_observer_events_not_tradable_equity",
    }


def _cohort_summary(
    rows: Sequence[dict[str, object]],
    *,
    key: str = "cohort",
    cohorts: Sequence[str] = ("aligned", "opposed", "flat", "unavailable"),
) -> dict[str, dict[str, object]]:
    result = {}
    for cohort in cohorts:
        cohort_rows = [row for row in rows if row[key] == cohort]
        values = [float(row["ta_points"]) for row in cohort_rows]
        result[cohort] = {
            "pairs": len(cohort_rows),
            "wins": sum(value > 0.0 for value in values),
            "losses": sum(value < 0.0 for value in values),
            "net_points": sum(values),
            "mean_points": (sum(values) / len(values) if values else None),
        }
    return result


def _option_liquidity_cohort(
    parity: Mapping[str, object] | None,
    change: Mapping[str, object] | None,
) -> str:
    """Classify threshold-free Pareto movement in the observed option surface."""

    if (
        not isinstance(parity, Mapping)
        or not isinstance(change, Mapping)
        or parity.get("usable") is not True
        or change.get("usable") is not True
    ):
        return "unavailable"
    comparisons = []
    for current_key, prior_key, higher_is_better in (
        ("pairs", "prior_pairs", True),
        ("dispersion_points", "prior_dispersion_points", False),
        ("median_relative_spread", "prior_median_relative_spread", False),
        ("max_age_seconds", "prior_max_age_seconds", False),
    ):
        try:
            current = float(parity[current_key])
            prior = float(change[prior_key])
        except (KeyError, TypeError, ValueError):
            return "unavailable"
        if not math.isfinite(current) or not math.isfinite(prior) or min(current, prior) < 0:
            return "unavailable"
        delta = current - prior if higher_is_better else prior - current
        comparisons.append(1 if delta > 0 else -1 if delta < 0 else 0)
    improving = any(value > 0 for value in comparisons)
    deteriorating = any(value < 0 for value in comparisons)
    return (
        "strengthening"
        if improving and not deteriorating
        else "weakening"
        if deteriorating and not improving
        else "mixed"
        if improving and deteriorating
        else "stable"
    )


def _non_overlapping_sequence(
    rows: Sequence[dict[str, object]],
    *,
    cohort: str | None = None,
    liquidity_cohort: str | None = None,
) -> list[dict[str, object]]:
    accepted = []
    available_at: datetime | None = None
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["decision_as_of_utc"]),
            str(item["forecast_id"]),
        ),
    ):
        if (
            (cohort is not None and row["cohort"] != cohort)
            or (
                liquidity_cohort is not None
                and row["liquidity_cohort"] != liquidity_cohort
            )
        ):
            continue
        decision_at = datetime.fromisoformat(str(row["decision_as_of_utc"]))
        if available_at is not None and decision_at < available_at:
            continue
        accepted.append(row)
        available_at = decision_at + timedelta(
            minutes=XSP_OPTION_PARITY_PRIMARY_HORIZON_MINUTES
        )
    return accepted


def _sequence_evidence(
    rows: Sequence[dict[str, object]],
    *,
    complete_dates: Sequence[date],
) -> dict[str, object]:
    values = [float(row["ta_points"]) for row in rows]
    daily = {day: 0.0 for day in complete_dates}
    direction_values: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        day = to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        daily[day] = daily.get(day, 0.0) + value
        direction_values[str(row["direction"])].append(value)
    daily_values = [daily[day] for day in sorted(daily)]
    daily_mean = statistics.fmean(daily_values) if daily_values else 0.0
    daily_std = statistics.stdev(daily_values) if len(daily_values) > 1 else 0.0
    daily_lcb95 = (
        daily_mean - 1.96 * daily_std / math.sqrt(len(daily_values))
        if len(daily_values) > 1
        else daily_mean
    )
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    equity = peak = max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    total = sum(values)
    return {
        "trades": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "loss_rate": len(losses) / len(rows) if rows else None,
        "net_points": total,
        "mean_points": statistics.fmean(values) if values else None,
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0.0 else None
        ),
        "maximum_drawdown_points": max_drawdown,
        "daily_mean_points": daily_mean,
        "daily_lcb95_points": daily_lcb95,
        "minimum_leave_one_session_out_points": (
            min((total - value for value in daily_values), default=0.0)
        ),
        "largest_win_share": (
            max(wins) / gross_profit if gross_profit > 0.0 else None
        ),
        "trades_per_complete_session": (
            len(rows) / len(complete_dates) if complete_dates else 0.0
        ),
        "directions": {
            direction: {
                "trades": len(direction_values.get(direction, ())),
                "net_points": sum(direction_values.get(direction, ())),
            }
            for direction in ("up", "down")
        },
        "forecast_ids": [str(row["forecast_id"]) for row in rows],
        "fingerprint": calibration_fingerprint(rows),
    }


def xsp_option_parity_participation_benchmark(
    ledger: LiveCalibrationLedger,
) -> dict[str, object]:
    """Stratify parity and score one frozen aligned shadow candidate."""

    pairs = []
    for settled in ledger.settled_directional_pairs(
        horizon_minutes=XSP_OPTION_PARITY_PRIMARY_HORIZON_MINUTES,
    ):
        context = settled["context"]
        parity = context.get("option_parity")
        change = parity.get("parity_change") if isinstance(parity, Mapping) else None
        direction = str(settled["direction"])
        change_direction = (
            str(change.get("direction") or "").strip().lower()
            if isinstance(change, Mapping)
            else ""
        )
        usable = bool(
            isinstance(parity, Mapping)
            and parity.get("source") == "option_nbbo_parity"
            and parity.get("authority") == "observation_only"
            and parity.get("usable") is True
            and isinstance(change, Mapping)
            and change.get("usable") is True
            and change_direction in ("up", "down", "flat")
        )
        cohort = (
            "flat"
            if usable and change_direction == "flat"
            else "aligned"
            if usable and change_direction == direction
            else "opposed"
            if usable
            else "unavailable"
        )
        preopen = parity.get("preopen_path") if isinstance(parity, Mapping) else None
        horizons = preopen.get("horizons") if isinstance(preopen, Mapping) else None
        preopen_directions = {}
        if isinstance(horizons, Mapping):
            for key in ("120", "240", "360"):
                value = horizons.get(key)
                if isinstance(value, Mapping):
                    preopen_directions[key] = (
                        str(value.get("direction") or "").strip().lower()
                    )
        preopen_usable = bool(
            isinstance(preopen, Mapping)
            and preopen.get("schema") == XSP_PREOPEN_RESEARCH_VERSION
            and preopen.get("source") == "option_nbbo_parity"
            and preopen.get("authority") == "observation_only"
            and preopen.get("usable") is True
            and set(preopen_directions) == {"120", "240", "360"}
            and set(preopen_directions.values()) <= {"up", "down", "flat"}
        )
        path_directions = tuple(
            preopen_directions.get(key) for key in ("120", "240", "360")
        )
        preopen_cohort = (
            "unavailable"
            if not preopen_usable
            else "aligned_all"
            if all(value == direction for value in path_directions)
            else "opposed_all"
            if all(
                value in ("up", "down") and value != direction
                for value in path_directions
            )
            else "reversal_into"
            if path_directions[0] == direction
            and path_directions[2] in ("up", "down")
            and path_directions[2] != direction
            else "mixed"
        )
        liquidity_cohort = _option_liquidity_cohort(
            parity if isinstance(parity, Mapping) else None,
            change if isinstance(change, Mapping) else None,
        )
        pairs.append(
            {
                "forecast_id": str(settled["forecast_id"]),
                "decision_as_of_utc": settled["decision_at"].isoformat(),
                "direction": direction,
                "evidence_mode": settled["evidence_mode"],
                "prospective": settled["prospective"],
                "cohort": cohort,
                "ta_points": float(settled["ta_points"]),
                "option_ts": (
                    parity.get("ts") if isinstance(parity, Mapping) else None
                ),
                "chain_fingerprint": (
                    parity.get("chain_fingerprint")
                    if isinstance(parity, Mapping)
                    else None
                ),
                "prior_ts": (
                    change.get("prior_ts") if isinstance(change, Mapping) else None
                ),
                "prior_chain_fingerprint": (
                    change.get("prior_chain_fingerprint")
                    if isinstance(change, Mapping)
                    else None
                ),
                "parity_direction": change_direction if usable else None,
                "liquidity_cohort": liquidity_cohort,
                "preopen_cohort": preopen_cohort,
                "preopen_path_fingerprint": (
                    calibration_fingerprint(preopen) if preopen_usable else None
                ),
                "preopen_directions": (
                    preopen_directions if preopen_usable else None
                ),
            }
        )

    usable = [row for row in pairs if row["cohort"] != "unavailable"]
    prospective = [row for row in pairs if row["prospective"]]
    prospective_usable = [
        row for row in prospective if row["cohort"] != "unavailable"
    ]
    diagnostic_sessions = {
        to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        for row in usable
    }
    prospective_event_dates = {
        to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        for row in prospective_usable
    }
    complete_dates = {
        datetime.fromisoformat(value).date()
        for value in ledger.complete_xsp_checkpoint_sessions(
            strategy_id="NO_TRADE",
            strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
        )
    }
    prospective_sessions = prospective_event_dates & complete_dates
    sample_eligible = [
        row
        for row in prospective_usable
        if to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        in complete_dates
    ]
    liquidity_sample_eligible = [
        row
        for row in sample_eligible
        if row["liquidity_cohort"] != "unavailable"
    ]
    liquidity_complete_sessions = {
        to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        for row in liquidity_sample_eligible
    }
    prospective_dates = {
        to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        for row in prospective
    }
    coverage_dates = (
        sorted(day for day in complete_dates if day >= min(prospective_dates))
        if prospective_dates
        else []
    )
    complete_prospective = [
        row
        for row in prospective
        if to_et(datetime.fromisoformat(str(row["decision_as_of_utc"]))).date()
        in coverage_dates
    ]
    baseline = _sequence_evidence(
        _non_overlapping_sequence(complete_prospective),
        complete_dates=coverage_dates,
    )
    aligned = _sequence_evidence(
        _non_overlapping_sequence(complete_prospective, cohort="aligned"),
        complete_dates=coverage_dates,
    )
    aligned_liquidity = _sequence_evidence(
        _non_overlapping_sequence(
            complete_prospective,
            cohort="aligned",
            liquidity_cohort="strengthening",
        ),
        complete_dates=coverage_dates,
    )
    required_trades = max(4, math.ceil(len(coverage_dates) / 2))
    sample_gate = (
        len(sample_eligible) >= XSP_OPTION_PARITY_MIN_USABLE_PAIRS
        and len(prospective_sessions) >= XSP_OPTION_PARITY_MIN_COMPLETE_SESSIONS
    )
    liquidity_sample_gate = (
        len(liquidity_sample_eligible) >= XSP_OPTION_PARITY_MIN_USABLE_PAIRS
        and len(liquidity_complete_sessions)
        >= XSP_OPTION_PARITY_MIN_COMPLETE_SESSIONS
    )
    checks = {
        "sample": sample_gate,
        "cadence": int(aligned["trades"]) >= required_trades,
        "both_directions": all(
            int(aligned["directions"][direction]["trades"]) >= 2
            and float(aligned["directions"][direction]["net_points"]) > 0.0
            for direction in ("up", "down")
        ),
        "positive_net": float(aligned["net_points"]) > 0.0,
        "positive_daily_lcb95": float(aligned["daily_lcb95_points"]) > 0.0,
        "positive_leave_one_session_out": (
            float(aligned["minimum_leave_one_session_out_points"]) > 0.0
        ),
        "bounded_concentration": (
            aligned["largest_win_share"] is not None
            and float(aligned["largest_win_share"]) <= 0.5
        ),
        "higher_mean_than_ta": (
            aligned["mean_points"] is not None
            and baseline["mean_points"] is not None
            and float(aligned["mean_points"]) > float(baseline["mean_points"])
        ),
        "lower_loss_rate_than_ta": (
            aligned["loss_rate"] is not None
            and baseline["loss_rate"] is not None
            and float(aligned["loss_rate"]) < float(baseline["loss_rate"])
        ),
    }
    shadow_candidate_gate = all(checks.values())
    liquidity_checks = {
        "sample": liquidity_sample_gate,
        "cadence": int(aligned_liquidity["trades"]) >= required_trades,
        "both_directions": all(
            int(aligned_liquidity["directions"][direction]["trades"]) >= 2
            and float(aligned_liquidity["directions"][direction]["net_points"]) > 0.0
            for direction in ("up", "down")
        ),
        "positive_net": float(aligned_liquidity["net_points"]) > 0.0,
        "positive_daily_lcb95": (
            float(aligned_liquidity["daily_lcb95_points"]) > 0.0
        ),
        "positive_leave_one_session_out": (
            float(aligned_liquidity["minimum_leave_one_session_out_points"]) > 0.0
        ),
        "bounded_concentration": (
            aligned_liquidity["largest_win_share"] is not None
            and float(aligned_liquidity["largest_win_share"]) <= 0.5
        ),
        "higher_mean_than_ta": (
            aligned_liquidity["mean_points"] is not None
            and baseline["mean_points"] is not None
            and float(aligned_liquidity["mean_points"])
            > float(baseline["mean_points"])
        ),
        "lower_loss_rate_than_ta": (
            aligned_liquidity["loss_rate"] is not None
            and baseline["loss_rate"] is not None
            and float(aligned_liquidity["loss_rate"])
            < float(baseline["loss_rate"])
        ),
        "higher_mean_than_aligned": (
            aligned_liquidity["mean_points"] is not None
            and aligned["mean_points"] is not None
            and float(aligned_liquidity["mean_points"])
            > float(aligned["mean_points"])
        ),
        "lower_loss_rate_than_aligned": (
            aligned_liquidity["loss_rate"] is not None
            and aligned["loss_rate"] is not None
            and float(aligned_liquidity["loss_rate"])
            < float(aligned["loss_rate"])
        ),
    }
    liquidity_candidate_gate = all(liquidity_checks.values())
    recommended_schema = (
        XSP_OPTION_LIQUIDITY_CANDIDATE_VERSION
        if liquidity_candidate_gate
        else XSP_OPTION_PARITY_CANDIDATE_VERSION
        if shadow_candidate_gate
        else None
    )
    source_ledger_sha256 = str(ledger.receipt()["sha256"])
    candidate_receipts = (
        {
            "schema": XSP_OPTION_PARITY_CANDIDATE_VERSION,
            "eligible": shadow_candidate_gate,
            "evidence_fingerprint": aligned["fingerprint"],
            "failed_checks": tuple(
                key for key, passed in checks.items() if not passed
            ),
        },
        {
            "schema": XSP_OPTION_LIQUIDITY_CANDIDATE_VERSION,
            "eligible": liquidity_candidate_gate,
            "evidence_fingerprint": aligned_liquidity["fingerprint"],
            "failed_checks": tuple(
                key for key, passed in liquidity_checks.items() if not passed
            ),
        },
    )
    shadow_recommendation = {
        "schema": XSP_SHADOW_RECOMMENDATION_VERSION,
        "authority": "recommendation_only",
        "scope": "selected_shadow_only",
        "verdict": "PROMOTE" if recommended_schema is not None else "HOLD",
        "recommended_candidate_schema": recommended_schema,
        "selection_authority": "none_until_explicit_run_freeze",
        "order_authority": "none",
        "open_position_strategy_switch_allowed": False,
        "profitability_clock_started": False,
        "preregistered_selected_run_policy": dict(
            XSP_DIRECTIONAL_SHADOW_POLICY
        ),
        "source_ledger_sha256": source_ledger_sha256,
        "candidates": candidate_receipts,
    }
    shadow_recommendation["fingerprint"] = calibration_fingerprint(
        shadow_recommendation
    )
    return {
        "schema": XSP_OPTION_PARITY_OBSERVER_VERSION,
        "authority": "observation_only",
        "promotion_eligible": False,
        "primary_horizon_minutes": XSP_OPTION_PARITY_PRIMARY_HORIZON_MINUTES,
        "policy": {
            "action": "classify_only",
            "cohorts": ("aligned", "opposed", "flat", "unavailable"),
            "liquidity_cohorts": XSP_OPTION_LIQUIDITY_COHORTS,
            "liquidity_classification": (
                "pareto_pairs_up_dispersion_spread_age_down"
            ),
            "preopen_cohorts": XSP_PREOPEN_PARITY_COHORTS,
            "minimum_usable_pairs": XSP_OPTION_PARITY_MIN_USABLE_PAIRS,
            "minimum_complete_sessions": XSP_OPTION_PARITY_MIN_COMPLETE_SESSIONS,
            "prospective_evidence_mode": "forward_broker_history",
            "complete_session": "every canonical RTH evaluation slot",
        },
        "source_ledger_sha256": source_ledger_sha256,
        "pair_fingerprint": calibration_fingerprint(pairs),
        "prospective_pair_fingerprint": calibration_fingerprint(prospective),
        "pairs": len(pairs),
        "usable_pairs": len(usable),
        "diagnostic_complete_sessions": len(diagnostic_sessions),
        "prospective_pairs": len(prospective),
        "prospective_usable_pairs": len(prospective_usable),
        "sample_eligible_pairs": len(sample_eligible),
        "liquidity_sample_eligible_pairs": len(liquidity_sample_eligible),
        "liquidity_complete_sessions": len(liquidity_complete_sessions),
        "liquidity_sample_gate": liquidity_sample_gate,
        "complete_sessions": len(prospective_sessions),
        "complete_session_preopen_usable_pairs": sum(
            row["preopen_cohort"] != "unavailable"
            for row in complete_prospective
        ),
        "complete_session_dates": sorted(
            day.isoformat() for day in prospective_sessions
        ),
        "sample_gate": sample_gate,
        "cohorts": _cohort_summary(pairs),
        "prospective_cohorts": _cohort_summary(prospective),
        "sample_eligible_cohorts": _cohort_summary(sample_eligible),
        "liquidity_cohorts": _cohort_summary(
            pairs,
            key="liquidity_cohort",
            cohorts=XSP_OPTION_LIQUIDITY_COHORTS,
        ),
        "prospective_liquidity_cohorts": _cohort_summary(
            prospective,
            key="liquidity_cohort",
            cohorts=XSP_OPTION_LIQUIDITY_COHORTS,
        ),
        "complete_session_liquidity_cohorts": _cohort_summary(
            complete_prospective,
            key="liquidity_cohort",
            cohorts=XSP_OPTION_LIQUIDITY_COHORTS,
        ),
        "preopen_cohorts": _cohort_summary(
            pairs,
            key="preopen_cohort",
            cohorts=XSP_PREOPEN_PARITY_COHORTS,
        ),
        "prospective_preopen_cohorts": _cohort_summary(
            prospective,
            key="preopen_cohort",
            cohorts=XSP_PREOPEN_PARITY_COHORTS,
        ),
        "complete_session_preopen_cohorts": _cohort_summary(
            complete_prospective,
            key="preopen_cohort",
            cohorts=XSP_PREOPEN_PARITY_COHORTS,
        ),
        "ta_observer_points": sum(float(row["ta_points"]) for row in pairs),
        "prospective_ta_observer_points": sum(
            float(row["ta_points"]) for row in prospective
        ),
        "sample_eligible_ta_observer_points": sum(
            float(row["ta_points"]) for row in sample_eligible
        ),
        "aligned_candidate": {
            "schema": XSP_OPTION_PARITY_CANDIDATE_VERSION,
            "authority": "observation_only",
            "shadow_candidate_eligible": shadow_candidate_gate,
            "policy": {
                "action": "aligned_only",
                "event_selection": "earliest_then_60m_non_overlapping",
                "baseline_universe": "all_prospective_complete_session_decisions",
                "minimum_trades": required_trades,
                "minimum_trades_per_complete_session": 0.5,
                "minimum_trades_per_direction": 2,
                "minimum_direction_net_points": "strictly_positive",
                "minimum_daily_lcb95_points": "strictly_positive",
                "minimum_leave_one_session_out_points": "strictly_positive",
                "maximum_largest_win_share": 0.5,
                "incremental_value": "higher_mean_and_lower_loss_rate_than_ta",
            },
            "coverage_complete_sessions": len(coverage_dates),
            "coverage_complete_session_dates": [
                day.isoformat() for day in coverage_dates
            ],
            "baseline": baseline,
            "candidate": aligned,
            "gate_checks": checks,
            "economic_interpretation": (
                "non_overlapping_observation_only_shadow_counterfactual"
            ),
        },
        "aligned_liquidity_candidate": {
            "schema": XSP_OPTION_LIQUIDITY_CANDIDATE_VERSION,
            "authority": "observation_only",
            "shadow_candidate_eligible": liquidity_candidate_gate,
            "policy": {
                "action": "aligned_and_pareto_strengthening_only",
                "event_selection": "earliest_then_60m_non_overlapping",
                "baseline_universe": "all_prospective_complete_session_decisions",
                "reference_candidate": XSP_OPTION_PARITY_CANDIDATE_VERSION,
                "minimum_usable_pairs": XSP_OPTION_PARITY_MIN_USABLE_PAIRS,
                "minimum_complete_sessions": (
                    XSP_OPTION_PARITY_MIN_COMPLETE_SESSIONS
                ),
                "minimum_trades": required_trades,
                "minimum_trades_per_direction": 2,
                "minimum_direction_net_points": "strictly_positive",
                "minimum_daily_lcb95_points": "strictly_positive",
                "minimum_leave_one_session_out_points": "strictly_positive",
                "maximum_largest_win_share": 0.5,
                "incremental_value": (
                    "higher_mean_and_lower_loss_rate_than_ta_and_aligned"
                ),
            },
            "coverage_complete_sessions": len(coverage_dates),
            "coverage_complete_session_dates": [
                day.isoformat() for day in coverage_dates
            ],
            "baseline": baseline,
            "aligned_reference": aligned,
            "candidate": aligned_liquidity,
            "gate_checks": liquidity_checks,
            "economic_interpretation": (
                "non_overlapping_observation_only_shadow_counterfactual"
            ),
        },
        "shadow_recommendation": shadow_recommendation,
        "economic_interpretation": "overlapping_observer_events_not_tradable_equity",
    }
