"""Deterministically rank frozen SPYU/SPXU execution profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


RANKING_SCHEMA = "xsp.opening-edge-v2-spyu-selection-ranking.v2"
RESULT_SCHEMA = "xsp.opening-edge-v2-spyu-selection-ranking-result.v1"
LATENCY_SCHEMA = "xsp.opening-edge-v2-spyu-entry-latency-stress.v1"
STARTING_CASH_USD = 1_350.0
FAMILIES = {
    "notional": "xsp.opening-edge-v2-spyu-spxu-replay.v1",
    "five_slot": "xsp.opening-edge-v2-spyu-spxu-cash-partition.v1",
    "two_slot": "xsp.opening-edge-v2-spyu-spxu-two-slot.v1",
}
FAMILY_ORDER = {"two_slot": 0, "five_slot": 1, "notional": 2}
COHORT_FLOOR_MEMBERS = [
    "recent.net_usd",
    "prior_complete_year.net_usd",
    "latest_complete_year.net_usd",
    "current_partial.net_usd",
    "full_available_history.up_net_usd",
    "full_available_history.down_net_usd",
    "scheduled_exit_next_open.full_available_history.net_usd",
]
DESCENDING_FIELDS = [
    "minimum_cohort_net_usd / max(intrabar_max_drawdown_usd, 1e-12)",
    "full_available_history.net_usd / max(intrabar_max_drawdown_usd, 1e-12)",
    "minimum_cohort_net_usd / fixed_entry_notional_usd",
    "full_available_history.net_usd / fixed_entry_notional_usd",
    "trades_per_year",
]
ASCENDING_FIELDS = [
    "intrabar_max_drawdown_usd / starting_settled_cash_usd",
    "fixed_entry_notional_usd",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _number(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _profile_notional(
    family: str,
    profile_id: str,
    result: Mapping[str, object],
) -> float:
    if family == "notional":
        prefix, separator, pricing = profile_id.partition(":")
        key, equals, value = prefix.partition("=")
        if (
            not separator
            or pricing != "fixed_measured"
            or key != "notional"
            or not equals
        ):
            raise ValueError("invalid fixed-measured notional profile")
        return _number(value, name="fixed entry notional")
    if profile_id != "fixed_measured":
        raise ValueError("invalid cash-partition profile")
    capital = result.get("capital_identity")
    if not isinstance(capital, Mapping):
        raise ValueError(f"{family}: capital identity is missing")
    if _number(
        capital.get("starting_settled_cash_usd"),
        name="starting settled cash",
    ) != STARTING_CASH_USD:
        raise ValueError(f"{family}: starting cash identity drift")
    return _number(
        capital.get("fixed_entry_notional_usd"),
        name="fixed entry notional",
    )


def _candidate(
    *,
    family: str,
    profile_id: str,
    result: Mapping[str, object],
    result_sha256: str,
) -> dict[str, object]:
    profiles = result.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError(f"{family}: profiles are missing")
    profile = profiles.get(profile_id)
    if (
        not isinstance(profile, Mapping)
        or profile.get("operationally_eligible") is not True
    ):
        raise ValueError(f"{family}:{profile_id}: eligibility drift")
    current = profile.get("current")
    scheduled = profile.get("scheduled")
    if not isinstance(current, Mapping) or not isinstance(scheduled, Mapping):
        raise ValueError(f"{family}:{profile_id}: replay states are missing")
    quantity_ranges_raw = current.get("qty_ranges")
    if (
        not isinstance(quantity_ranges_raw, Mapping)
        or set(quantity_ranges_raw) != {"SPYU", "SPXU"}
    ):
        raise ValueError(f"{family}:{profile_id}: quantity ranges are missing")
    quantity_ranges: dict[str, list[int]] = {}
    for symbol in ("SPYU", "SPXU"):
        bounds = quantity_ranges_raw[symbol]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(
                f"{family}:{profile_id}:{symbol} quantity range is invalid"
            )
        try:
            lower, upper = (int(value) for value in bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{family}:{profile_id}:{symbol} quantity range is invalid"
            ) from exc
        if (
            any(isinstance(value, bool) or float(value) != int(value) for value in bounds)
            or lower <= 0
            or upper < lower
        ):
            raise ValueError(
                f"{family}:{profile_id}:{symbol} quantity range is invalid"
            )
        quantity_ranges[symbol] = [lower, upper]
    periods = current.get("periods")
    scheduled_periods = scheduled.get("periods")
    if not isinstance(periods, Mapping) or not isinstance(
        scheduled_periods,
        Mapping,
    ):
        raise ValueError(f"{family}:{profile_id}: periods are missing")

    def period(name: str) -> Mapping[str, object]:
        row = periods.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"{family}:{profile_id}:{name} is missing")
        return row

    full = period("full_available_history")
    scheduled_full = scheduled_periods.get("full_available_history")
    if not isinstance(scheduled_full, Mapping):
        raise ValueError(
            f"{family}:{profile_id}: scheduled full period is missing"
        )
    cohort_nets = {
        "recent": _number(period("recent").get("net_usd"), name="recent net"),
        "prior_complete_year": _number(
            period("prior_complete_year").get("net_usd"),
            name="prior complete year net",
        ),
        "latest_complete_year": _number(
            period("latest_complete_year").get("net_usd"),
            name="latest complete year net",
        ),
        "current_partial": _number(
            period("current_partial").get("net_usd"),
            name="current partial net",
        ),
        "full_up": _number(full.get("up_net_usd"), name="full up net"),
        "full_down": _number(
            full.get("down_net_usd"),
            name="full down net",
        ),
        "scheduled_full": _number(
            scheduled_full.get("net_usd"),
            name="scheduled full net",
        ),
    }
    minimum_cohort_net = min(cohort_nets.values())
    full_net = _number(full.get("net_usd"), name="full net")
    drawdown = _number(
        current.get("intrabar_max_drawdown_usd"),
        name="intrabar drawdown",
    )
    if drawdown < 0:
        raise ValueError("intrabar drawdown cannot be negative")
    notional = _profile_notional(family, profile_id, result)
    if notional <= 0 or notional > STARTING_CASH_USD:
        raise ValueError("fixed entry notional is outside the cash identity")
    trades_per_year = _number(
        current.get("trades_per_year"),
        name="trades per year",
    )
    metrics = {
        "minimum_cohort_net_usd": minimum_cohort_net,
        "full_net_usd": full_net,
        "intrabar_max_drawdown_usd": drawdown,
        "fixed_entry_notional_usd": notional,
        "trades_per_year": trades_per_year,
        "minimum_cohort_net_to_drawdown": (
            minimum_cohort_net / max(drawdown, 1e-12)
        ),
        "full_net_to_drawdown": full_net / max(drawdown, 1e-12),
        "minimum_cohort_net_to_notional": minimum_cohort_net / notional,
        "full_net_to_notional": full_net / notional,
        "drawdown_to_starting_cash": drawdown / STARTING_CASH_USD,
    }
    identity = {
        "family": family,
        "profile_id": profile_id,
        "result_sha256": result_sha256,
        "fixed_entry_notional_usd": notional,
        "quantity_rule": "floor(fixed_entry_notional_usd / fresh_limit_price)",
        "historical_quantity_ranges": quantity_ranges,
        "frozen_max_quantities": {
            symbol: bounds[1] for symbol, bounds in quantity_ranges.items()
        },
        "metrics": metrics,
    }
    return {
        **identity,
        "cohort_net_usd": cohort_nets,
        "nominee_id": _fingerprint(identity),
    }


def rank_spyu_transport(
    *,
    ranking_path: Path,
    result_paths: Mapping[str, Path],
    latency_path: Path,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Return one frozen-profile nominee or an explicit HOLD receipt."""

    ranking = _load(ranking_path)
    ranking_rule = ranking.get("ranking")
    boundary = ranking.get("selection_boundary")
    if (
        ranking.get("schema") != RANKING_SCHEMA
        or ranking.get("authority") != "preregistered_research_ranking_only"
        or ranking.get("order_authority") != "none"
        or ranking.get("profitability_clock_started") is not False
        or ranking.get("cohort_floor_members") != COHORT_FLOOR_MEMBERS
        or not isinstance(ranking_rule, Mapping)
        or ranking_rule.get("order") != "lexicographic"
        or ranking_rule.get("descending") != DESCENDING_FIELDS
        or ranking_rule.get("ascending") != ASCENDING_FIELDS
        or ranking_rule.get("stable_family_tiebreak")
        != ["two_slot", "five_slot", "notional"]
        or ranking_rule.get("final_tiebreak")
        != "profile_id lexical ascending"
        or not isinstance(boundary, Mapping)
        or boundary.get("ranking_does_not_select") is not True
        or boundary.get("usd_1350_cash_identity_required") is not True
    ):
        raise ValueError("invalid SPYU ranking contract")
    if set(result_paths) != set(FAMILIES):
        raise ValueError("exactly three SPYU result families are required")

    results: dict[str, Mapping[str, object]] = {}
    result_receipts: dict[str, dict[str, object]] = {}
    nominees: set[str] = set()
    gate_hashes = ranking.get("gate_sha256")
    if not isinstance(gate_hashes, Mapping):
        raise ValueError("SPYU ranking gate identities are missing")
    for family, expected_schema in FAMILIES.items():
        path = result_paths[family]
        result = _load(path)
        eligible = result.get("operationally_eligible")
        if (
            result.get("schema") != expected_schema
            or result.get("authority") != "research_and_broker_preview_only"
            or result.get("order_authority") != "none"
            or result.get("profitability_clock_started") is not False
            or not isinstance(eligible, list)
            or result.get("gate_sha256") != gate_hashes.get(family)
        ):
            raise ValueError(f"{family}: invalid frozen result")
        if len(eligible) != len(set(map(str, eligible))):
            raise ValueError(f"{family}: duplicate eligible profile")
        results[family] = result
        digest = _sha256(path)
        result_receipts[family] = {
            "path": str(path),
            "sha256": digest,
            "schema": expected_schema,
        }
        nominees.update(f"{family}:{profile_id}" for profile_id in eligible)

    latency = _load(latency_path)
    latency_nominees = latency.get("base_nominees")
    latency_profiles = latency.get("profiles")
    latency_results = latency.get("base_results")
    if (
        latency.get("schema") != LATENCY_SCHEMA
        or latency.get("authority") != "execution_stress_only"
        or latency.get("order_authority") != "none"
        or latency.get("profitability_clock_started") is not False
        or not isinstance(latency_nominees, list)
        or not isinstance(latency_profiles, Mapping)
        or not isinstance(latency_results, Mapping)
        or set(map(str, latency_nominees)) != nominees
        or set(map(str, latency_profiles)) != nominees
    ):
        raise ValueError("latency receipt does not bind every base nominee")
    for family, receipt in result_receipts.items():
        latency_result = latency_results.get(family)
        if (
            not isinstance(latency_result, Mapping)
            or latency_result.get("sha256") != receipt["sha256"]
            or latency_result.get("operationally_eligible")
            != results[family]["operationally_eligible"]
        ):
            raise ValueError(f"{family}: latency base-result identity drift")

    candidates = []
    for key in sorted(nominees):
        family, profile_id = key.split(":", 1)
        stress = latency_profiles[key]
        if not isinstance(stress, Mapping):
            raise ValueError(f"{key}: latency profile is missing")
        if stress.get("stress_pass") is not True:
            continue
        candidates.append(
            _candidate(
                family=family,
                profile_id=profile_id,
                result=results[family],
                result_sha256=result_receipts[family]["sha256"],
            )
        )

    all_survive = (
        bool(nominees)
        and latency.get("all_base_nominees_survive") is True
        and len(candidates) == len(nominees)
        and latency.get("verdict")
        == "LATENCY_STRESS_PASS_SELECTION_STILL_HOLD"
    )
    candidates.sort(
        key=lambda row: (
            -float(row["metrics"]["minimum_cohort_net_to_drawdown"]),
            -float(row["metrics"]["full_net_to_drawdown"]),
            -float(row["metrics"]["minimum_cohort_net_to_notional"]),
            -float(row["metrics"]["full_net_to_notional"]),
            -float(row["metrics"]["trades_per_year"]),
            float(row["metrics"]["drawdown_to_starting_cash"]),
            float(row["metrics"]["fixed_entry_notional_usd"]),
            FAMILY_ORDER[str(row["family"])],
            str(row["profile_id"]),
        )
    )
    nominee = candidates[0] if all_survive else None
    reason = (
        "no_operational_profile"
        if not nominees
        else "latency_stress_failed"
        if not all_survive
        else "ranked_nominee_requires_fresh_rth_execution_proof"
    )
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    return {
        "schema": RESULT_SCHEMA,
        "observed_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "authority": "research_ranking_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "selected_shadow_created": False,
        "ranking_contract": {
            "path": str(ranking_path),
            "sha256": _sha256(ranking_path),
        },
        "base_results": result_receipts,
        "latency_result": {
            "path": str(latency_path),
            "sha256": _sha256(latency_path),
        },
        "base_nominees": sorted(nominees),
        "all_base_nominees_survive_latency": all_survive,
        "ranked_candidates": candidates if all_survive else [],
        "nominee": nominee,
        "verdict": "NOMINEE_STILL_HOLD" if nominee else "HOLD",
        "reason": reason,
        "selection_blockers": (
            [
                "fresh_spyu_intraday_indicative_value",
                "fresh_broker_qualified_rth_nbbo",
                "historical_full_quantity_symbol_dwell_validation",
                "fresh_non_transmitting_exact_quantity_what_if",
                "transport_bound_selected_run",
            ]
            if nominee
            else []
        ),
    }


def _write_atomic(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--notional", type=Path, required=True)
    parser.add_argument("--five-slot", type=Path, required=True)
    parser.add_argument("--two-slot", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    latency = _load(args.latency)
    observed_at_raw = latency.get("observed_at_utc")
    if not isinstance(observed_at_raw, str):
        raise ValueError("latency receipt timestamp is missing")
    observed_at = datetime.fromisoformat(
        observed_at_raw.replace("Z", "+00:00")
    )
    receipt = rank_spyu_transport(
        ranking_path=args.ranking,
        result_paths={
            "notional": args.notional,
            "five_slot": args.five_slot,
            "two_slot": args.two_slot,
        },
        latency_path=args.latency,
        observed_at=observed_at,
    )
    _write_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
