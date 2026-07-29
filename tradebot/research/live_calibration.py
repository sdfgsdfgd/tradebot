"""Append-only live-versus-replay calibration evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from ..engines.market import xsp_rth_evaluation_slots
from ..time_utils import ET_ZONE
from .xsp_profitability import (
    LIVE_PROFITABILITY_SCHEMA,
    SELECTED_CASH_EQUITY_SCHEMA as SELECTED_CASH_EQUITY_SCHEMA,
    SELECTED_EQUITY_SCHEMA as SELECTED_EQUITY_SCHEMA,
    XspProfitabilityPolicy,
    empty_xsp_profitability_receipt,
    xsp_profitability_amount_fields,
    xsp_profitability_contract,
)

LIVE_CALIBRATION_SCHEMA = "live_calibration.v1"
XSP_DIRECTIONAL_OBSERVER_VERSION = "xsp.directional-observer.v1"
LIVE_CALIBRATION_VERDICTS = {"PROMOTE", "HOLD", "REVISE", "QUARANTINE", "STOP"}
LIVE_CALIBRATION_CHECKPOINT_STATUSES = {
    "CLOSED", "EVALUATED", "NO_DATA", "STALE_DATA", "UNSUPPORTED_SESSION",
}
_IDENTITY_FIELDS = {
    "strategy_id", "strategy_version", "decision_as_of_utc",
    "tape_fingerprint", "config_fingerprint", "capital_sleeve",
}
_FORECAST_FIELDS = {
    "decision", "outcome_not_before_utc", "pnl_distribution",
    "risk", "costs", "fill_assumptions",
}
_RECORD_FIELDS = {
    "forecast": ("forecast_id", (
        "identity", "forecast", "context", "counterfactuals", "gates",
    )),
    "result": ("result_id", ("forecast_id", "observed", "drift", "verdict")),
    "checkpoint": ("checkpoint_id", (
        "evaluation_as_of_utc", "strategy_id", "strategy_version",
        "trading_date", "session", "status", "evidence",
    )),
}


def _utc_iso(value: datetime | str) -> str:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("calibration timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _record_address_valid(record: Mapping[str, object]) -> bool:
    try:
        kind = str(record.get("kind"))
        key, fields = _RECORD_FIELDS[kind]
        address = _digest({field: record[field] for field in fields})
        addressed = record.get("schema") == LIVE_CALIBRATION_SCHEMA and record.get(key) == address
        if kind == "checkpoint" and not addressed:
            payload = {field: record[field] for field in (*fields, "recorded_at_utc")}
            addressed = record.get("checkpoint_id") == _digest(payload)
        return bool(addressed) and (
            key != "forecast_id"
            or record.get("identity_id") == _digest(record["identity"])
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def calibration_fingerprint(payload: object) -> str:
    """Stable identity for a tape, configuration, or evidence payload."""
    return _digest(payload)


def _repair_tail(handle) -> bool:
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end <= 0:
        return False
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        return False
    cursor = end
    while cursor > 0:
        size = min(8192, cursor)
        cursor -= size
        handle.seek(cursor)
        newline = handle.read(size).rfind(b"\n")
        if newline >= 0:
            cursor += newline + 1
            break
    handle.truncate(cursor)
    return True


class LiveCalibrationLedger:
    """Durably freeze forecasts, then append their observed outcomes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> Iterator[dict[str, object]]:
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{self.path}:{line_no}: invalid calibration JSON"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"{self.path}:{line_no}: calibration record must be an object"
                        )
                    if not _record_address_valid(record):
                        raise ValueError(
                            f"{self.path}:{line_no}: invalid calibration content address"
                        )
                    yield record
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def freeze(
        self,
        *,
        identity: Mapping[str, object],
        forecast: Mapping[str, object],
        context: Mapping[str, object],
        counterfactuals: Sequence[Mapping[str, object]],
        gates: Mapping[str, object],
        recorded_at: datetime | str,
    ) -> dict[str, object]:
        missing_identity = _IDENTITY_FIELDS - set(identity)
        missing_forecast = _FORECAST_FIELDS - set(forecast)
        if missing_identity or missing_forecast:
            raise ValueError(
                "missing calibration fields: "
                f"identity={sorted(missing_identity)} "
                f"forecast={sorted(missing_forecast)}"
            )
        decision_at = _utc_iso(str(identity["decision_as_of_utc"]))
        recorded_at_utc = _utc_iso(recorded_at)
        outcome_not_before = _utc_iso(str(forecast["outcome_not_before_utc"]))
        if recorded_at_utc < decision_at:
            raise ValueError("forecast was recorded before its decision evidence existed")
        if recorded_at_utc >= outcome_not_before:
            raise ValueError("forecast was recorded at or after its outcome window opened")

        frozen_identity = dict(identity)
        frozen_identity["decision_as_of_utc"] = decision_at
        frozen_forecast = dict(forecast)
        frozen_forecast["outcome_not_before_utc"] = outcome_not_before
        content = {
            "identity": frozen_identity,
            "forecast": frozen_forecast,
            "context": dict(context),
            "counterfactuals": [dict(row) for row in counterfactuals],
            "gates": dict(gates),
        }
        record = {
            "schema": LIVE_CALIBRATION_SCHEMA,
            "kind": "forecast",
            "identity_id": _digest(frozen_identity),
            "forecast_id": _digest(content),
            "recorded_at_utc": recorded_at_utc,
            **content,
        }
        return self._append_unique(
            record,
            key="forecast_id",
            unique_field="identity_id",
            conflict="calibration identity already has a different forecast",
        )

    def settle(
        self,
        *,
        forecast_id: str,
        observed: Mapping[str, object],
        drift: Mapping[str, object],
        verdict: str,
        settled_at: datetime | str,
    ) -> dict[str, object]:
        verdict = str(verdict).strip().upper()
        if verdict not in LIVE_CALIBRATION_VERDICTS:
            raise ValueError(f"unsupported calibration verdict: {verdict!r}")
        records = list(self.records())
        forecast = next(
            (
                row
                for row in records
                if row.get("kind") == "forecast"
                and row.get("forecast_id") == forecast_id
            ),
            None,
        )
        if forecast is None:
            raise ValueError(f"unknown calibration forecast: {forecast_id}")
        if "outcome_as_of_utc" not in observed:
            raise ValueError("observed calibration requires outcome_as_of_utc")
        settled_at_utc = _utc_iso(settled_at)
        outcome_as_of = _utc_iso(str(observed["outcome_as_of_utc"]))
        outcome_not_before = str(
            dict(forecast["forecast"])["outcome_not_before_utc"]
        )
        if outcome_as_of < outcome_not_before:
            raise ValueError("calibration outcome predates its forecast horizon")
        if settled_at_utc < outcome_as_of:
            raise ValueError("calibration settlement predates its observed outcome")
        frozen_observed = dict(observed)
        frozen_observed["outcome_as_of_utc"] = outcome_as_of
        content = {
            "forecast_id": forecast_id,
            "observed": frozen_observed,
            "drift": dict(drift),
            "verdict": verdict,
        }
        record = {
            "schema": LIVE_CALIBRATION_SCHEMA,
            "kind": "result",
            "result_id": _digest(content),
            "settled_at_utc": settled_at_utc,
            **content,
        }
        prior = next(
            (
                row
                for row in records
                if row.get("kind") == "result"
                and row.get("forecast_id") == forecast_id
            ),
            None,
        )
        if prior is not None:
            if prior.get("result_id") == record["result_id"]:
                return prior
            raise ValueError(f"calibration forecast already settled: {forecast_id}")
        return self._append_unique(
            record,
            key="result_id",
            unique_field="forecast_id",
            conflict=f"calibration forecast already settled: {forecast_id}",
        )

    def checkpoint(
        self,
        *,
        evaluation_as_of: datetime | str,
        strategy_id: str,
        strategy_version: str,
        trading_date: str | None,
        session: str,
        status: str,
        evidence: Mapping[str, object],
        recorded_at: datetime | str,
    ) -> dict[str, object]:
        """Append one invocation receipt independently of signal activity."""

        evaluation_at = _utc_iso(evaluation_as_of)
        recorded_at_utc = _utc_iso(recorded_at)
        if recorded_at_utc < evaluation_at:
            raise ValueError("checkpoint was recorded before its evaluation time")
        status = str(status).strip().upper()
        if status not in LIVE_CALIBRATION_CHECKPOINT_STATUSES:
            raise ValueError(f"unsupported calibration checkpoint: {status!r}")
        content = {
            "evaluation_as_of_utc": evaluation_at,
            "strategy_id": str(strategy_id),
            "strategy_version": str(strategy_version),
            "trading_date": str(trading_date) if trading_date else None,
            "session": str(session),
            "status": status,
            "evidence": dict(evidence),
            "recorded_at_utc": recorded_at_utc,
        }
        record = {
            "schema": LIVE_CALIBRATION_SCHEMA,
            "kind": "checkpoint",
            "checkpoint_id": _digest(content),
            **content,
        }
        return self._append_unique(record, key="checkpoint_id")

    def receipt(self) -> dict[str, object]:
        raw = b""
        if self.path.exists():
            with self.path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    raw = handle.read()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        records = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{self.path}:{line_no}: invalid calibration JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{self.path}:{line_no}: calibration record must be an object"
                )
            if not _record_address_valid(record):
                raise ValueError(
                    f"{self.path}:{line_no}: invalid calibration content address"
                )
            records.append(record)
        forecasts = {
            str(row["forecast_id"])
            for row in records
            if row.get("kind") == "forecast"
        }
        settled = {
            str(row["forecast_id"])
            for row in records
            if row.get("kind") == "result"
        }
        checkpoints = [
            row for row in records if row.get("kind") == "checkpoint"
        ]
        checkpoint_statuses = {
            status: sum(row.get("status") == status for row in checkpoints)
            for status in sorted(LIVE_CALIBRATION_CHECKPOINT_STATUSES)
            if any(row.get("status") == status for row in checkpoints)
        }
        return {
            "schema": LIVE_CALIBRATION_SCHEMA,
            "path": str(self.path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(records),
            "forecasts": len(forecasts),
            "results": len(settled),
            "checkpoints": len(checkpoints),
            "checkpoint_statuses": checkpoint_statuses,
            "unsettled": sorted(forecasts - settled),
        }

    def settled_directional_pairs(self, *, horizon_minutes: int) -> list[dict[str, object]]:
        """Join immutable observer forecasts to their exact settled outcomes."""
        records = list(self.records())
        forecasts = {
            str(row["forecast_id"]): row for row in records
            if row.get("kind") == "forecast" and row.get("forecast_id")
        }
        pairs = []
        for result in records:
            forecast = forecasts.get(str(result.get("forecast_id") or ""))
            observed = result.get("observed")
            if (result.get("kind") != "result" or forecast is None
                    or not isinstance(observed, Mapping)):
                continue
            identity = forecast.get("identity")
            frozen = forecast.get("forecast")
            outcomes = observed.get("counterfactuals")
            expected = forecast.get("counterfactuals")
            context = forecast.get("context")
            if (
                not isinstance(identity, Mapping)
                or not isinstance(frozen, Mapping)
                or not isinstance(outcomes, list)
                or not isinstance(expected, list)
                or not isinstance(context, Mapping)
                or identity.get("strategy_id") != "NO_TRADE"
                or identity.get("strategy_version") != XSP_DIRECTIONAL_OBSERVER_VERSION
                or frozen.get("decision") != "NO_TRADE"
            ):
                continue
            try:
                decision_at = datetime.fromisoformat(_utc_iso(str(
                    identity["decision_as_of_utc"]
                )))
                outcome_at = datetime.fromisoformat(_utc_iso(str(
                    frozen["outcome_not_before_utc"]
                )))
                observed_at = datetime.fromisoformat(_utc_iso(str(
                    observed["outcome_as_of_utc"]
                )))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                outcome_at - decision_at != timedelta(minutes=horizon_minutes)
                or observed_at != outcome_at
            ):
                continue
            expected_rows = [row for row in expected if isinstance(row, Mapping) and row.get("strategy_id") == "directional_impulse.observer"]
            outcome_rows = [row for row in outcomes if isinstance(row, Mapping) and row.get("strategy_id") == "directional_impulse.observer"]
            if len(expected_rows) != 1 or len(outcome_rows) != 1:
                continue
            baseline = outcome_rows[0]
            evidence_mode = str(context.get("evidence_mode") or "").strip()
            expected_direction = str(expected_rows[0].get("decision") or "").strip().lower()
            direction = str(baseline.get("direction") or "").strip().lower()
            try:
                baseline_points = float(baseline["net_points"])
            except (KeyError, TypeError, ValueError):
                continue
            if direction not in ("up", "down") or direction != expected_direction:
                continue
            pairs.append(
                {
                    "forecast_id": str(result["forecast_id"]),
                    "decision_at": decision_at,
                    "direction": direction,
                    "ta_points": baseline_points,
                    "context": context,
                    "evidence_mode": evidence_mode,
                    "prospective": evidence_mode == "forward_broker_history",
                }
            )
        return pairs

    def complete_xsp_checkpoint_sessions(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        slot_tolerance_seconds: float = 90.0,
    ) -> tuple[str, ...]:
        """Return only dates with one coherent EVALUATED checkpoint per slot."""
        by_day: dict[date, list[tuple[dict[str, object], datetime, datetime]]] = {}
        for row in self.records():
            if (
                row.get("kind") != "checkpoint"
                or row.get("strategy_id") != strategy_id
                or row.get("strategy_version") != strategy_version
                or row.get("session") != "RTH"
            ):
                continue
            try:
                evaluation_at = datetime.fromisoformat(
                    _utc_iso(str(row["evaluation_as_of_utc"]))
                )
                recorded_at = datetime.fromisoformat(_utc_iso(str(row["recorded_at_utc"])))
                trading_day = date.fromisoformat(str(row["trading_date"]))
            except (KeyError, TypeError, ValueError):
                continue
            by_day.setdefault(trading_day, []).append((row, evaluation_at, recorded_at))

        complete = []
        for trading_day, rows in sorted(by_day.items()):
            slots = xsp_rth_evaluation_slots(trading_day)
            if not slots:
                continue
            for slot in slots:
                slot_utc = slot.astimezone(timezone.utc)
                candidates = [
                    row
                    for row, evaluation_at, recorded_at in rows
                    if abs((evaluation_at - slot_utc).total_seconds()) <= slot_tolerance_seconds
                    and abs((recorded_at - slot_utc).total_seconds()) <= slot_tolerance_seconds
                ]
                signatures = {
                    (str(row.get("status")), _digest(row.get("evidence")))
                    for row in candidates
                }
                if len(signatures) != 1 or next(iter(signatures))[0] != "EVALUATED":
                    break
            else:
                complete.append(trading_day.isoformat())
        return tuple(complete)

    def xsp_profitability_receipt(
        self,
        *,
        policy: XspProfitabilityPolicy,
        as_of: datetime | str,
        _prefix: bool = False,
    ) -> dict[str, object]:
        """Fail closed over one selected, reconciled XSP equity run."""
        observed_at = datetime.fromisoformat(_utc_iso(as_of))
        equity_contract, policy_errors = xsp_profitability_contract(policy)
        if policy_errors:
            return empty_xsp_profitability_receipt(
                policy=policy,
                observed_at=observed_at,
                status="NOT_STARTED",
                reasons=policy_errors,
            )
        assert isinstance(equity_contract, Mapping)
        evidence_key = str(equity_contract["evidence_key"])
        required_equity_fields = equity_contract["fields"]
        assert isinstance(required_equity_fields, set)
        amount_fields = xsp_profitability_amount_fields(equity_contract)

        matching = [
            row
            for row in self.records()
            if row.get("kind") == "checkpoint"
            and row.get("strategy_id") == policy.strategy_id
            and row.get("strategy_version") == policy.strategy_version
            and isinstance(row.get("evidence"), Mapping)
            and evidence_key in row["evidence"]
        ]
        if not matching:
            return empty_xsp_profitability_receipt(
                policy=policy,
                observed_at=observed_at,
                status="NOT_STARTED",
                reasons=["no_selected_checkpoints"],
            )

        errors: set[str] = set()
        parsed: list[tuple[dict[str, object], datetime, datetime, date, dict[str, object]]] = []
        run_started: datetime | None = None
        for row in matching:
            try:
                evaluation_at = datetime.fromisoformat(_utc_iso(str(row["evaluation_as_of_utc"])))
                recorded_at = datetime.fromisoformat(_utc_iso(str(row["recorded_at_utc"])))
                trading_day = date.fromisoformat(str(row["trading_date"]))
            except (KeyError, TypeError, ValueError):
                errors.add("invalid_checkpoint_identity")
                continue
            if recorded_at > observed_at:
                continue
            evidence = row.get("evidence")
            equity = (
                evidence.get(evidence_key)
                if isinstance(evidence, Mapping)
                else None
            )
            if not isinstance(equity, Mapping):
                errors.add(f"missing_{evidence_key}")
                continue
            frozen = dict(equity)
            if required_equity_fields - set(frozen):
                errors.add(f"incomplete_{evidence_key}")
                continue
            if frozen.get("run_id") != policy.run_id:
                continue
            checkpoint_fields = (*_RECORD_FIELDS["checkpoint"][1], "recorded_at_utc")
            if row["checkpoint_id"] != _digest({field: row[field] for field in checkpoint_fields}):
                errors.add("unaddressed_checkpoint_time")
                continue
            try:
                started_at = datetime.fromisoformat(_utc_iso(str(frozen["run_started_at_utc"])))
                numeric = {
                    key: float(frozen[source])
                    for key, source in amount_fields.items()
                }
                closed_trades = int(frozen["closed_trades"])
            except (TypeError, ValueError):
                errors.add(f"invalid_{evidence_key}")
                continue
            if not all(map(math.isfinite, numeric.values())):
                errors.add("nonfinite_selected_equity")
                continue
            if (
                frozen.get("schema") != policy.equity_schema
                or frozen.get("config_fingerprint") != policy.config_fingerprint
                or frozen.get("capital_sleeve") != policy.capital_sleeve
                or frozen.get("unit") != policy.unit
            ):
                errors.add("selected_identity_drift")
                continue
            if run_started is None:
                run_started = started_at
            elif run_started != started_at:
                errors.add("run_start_drift")
                continue
            if started_at > evaluation_at:
                errors.add("checkpoint_predates_run")
                continue
            if (
                numeric["cumulative_cost"] < 0
                or numeric["session_cost"] < 0
                or numeric["gross_wins"] < 0
                or numeric["top_five_gross_wins"] < 0
                or numeric["top_five_gross_wins"]
                > numeric["gross_wins"] + 1e-9
                or closed_trades < 0
                or abs(
                    numeric["cumulative_net"]
                    - numeric["cumulative_gross"]
                    + numeric["cumulative_cost"]
                )
                > 1e-7
                or abs(
                    numeric["session_net"]
                    - numeric["session_gross"]
                    + numeric["session_cost"]
                )
                > 1e-7
                or abs(
                    numeric["cumulative_net"]
                    - numeric["cumulative_realized_net"]
                    - numeric["open_mark"]
                )
                > 1e-7
            ):
                errors.add("selected_economics_mismatch")
                continue
            breaches = frozen["safety_breaches"]
            if not isinstance(breaches, list) or any(
                not isinstance(item, str) for item in breaches
            ):
                errors.add("invalid_safety_breaches")
                continue
            frozen.update(numeric)
            frozen["closed_trades"] = closed_trades
            parsed.append((row, evaluation_at, recorded_at, trading_day, frozen))

        if run_started is None:
            return empty_xsp_profitability_receipt(
                policy=policy,
                observed_at=observed_at,
                status="INVALID_EVIDENCE" if errors else "NOT_STARTED",
                reasons=sorted(errors or {"no_selected_checkpoints"}),
            )

        first_day = run_started.astimezone(ET_ZONE).date()
        final_day = observed_at.astimezone(ET_ZONE).date()
        days = []
        cursor = first_day
        while cursor <= final_day:
            slots = xsp_rth_evaluation_slots(cursor)
            due = tuple(
                slot for slot in slots
                if run_started <= slot.astimezone(timezone.utc) <= observed_at
            )
            if due:
                days.append((cursor, slots, due))
            cursor += timedelta(days=1)

        sessions = []
        equity_path = [0.0]
        ordered_equity: list[dict[str, object]] = []
        coverage_broken = False
        prior_close: tuple[float, float, float] | None = None
        for trading_day, slots, due_slots in days:
            slot_rows = []
            missing_slots = []
            conflict_slots = []
            for slot in due_slots:
                slot_utc = slot.astimezone(timezone.utc)
                candidates = [
                    item
                    for item in parsed
                    if item[3] == trading_day
                    and item[0].get("session") == "RTH"
                    and abs((item[1] - slot_utc).total_seconds()) <= policy.slot_tolerance_seconds
                    and abs((item[2] - slot_utc).total_seconds()) <= policy.slot_tolerance_seconds
                ]
                if not candidates:
                    missing_slots.append(slot.isoformat())
                    continue
                signatures = {(str(item[0].get("status")), _digest(item[4])) for item in candidates}
                if len(signatures) != 1:
                    conflict_slots.append(slot.isoformat())
                    continue
                chosen = min(candidates, key=lambda item: (item[2], item[1]))
                if chosen[0].get("status") != "EVALUATED":
                    missing_slots.append(slot.isoformat())
                    continue
                slot_rows.append(chosen)
            covered = not missing_slots and not conflict_slots and len(slot_rows) == len(due_slots)
            complete = covered and len(due_slots) == len(slots)
            coverage_broken = coverage_broken or not covered
            if covered:
                bases = [
                    tuple(
                        item[4][f"cumulative_{key}"]
                        - item[4][f"session_{key}"]
                        for key in ("gross", "cost", "net")
                    )
                    for item in slot_rows
                ]
                if any(abs(left - right) > 1e-7 for base in bases for left, right in zip(base, bases[0])) or (
                    prior_close and any(abs(left - right) > 1e-7 for left, right in zip(bases[0], prior_close))
                ):
                    errors.add("inconsistent_session_rollup")
                prior_close = tuple(
                    slot_rows[-1][4][f"cumulative_{key}"]
                    for key in ("gross", "cost", "net")
                )
                ordered_equity.extend(item[4] for item in slot_rows)
                equity_path.extend(
                    float(item[4]["cumulative_net"]) for item in slot_rows
                )
            sessions.append(
                {
                    "trading_date": trading_day.isoformat(),
                    "expected_slots": len(slots),
                    "due_slots": len(due_slots),
                    "evaluated_slots": len(slot_rows),
                    "missing_slots": missing_slots,
                    "conflict_slots": conflict_slots,
                    "covered_to_as_of": covered,
                    "complete": complete,
                    "completed_at_utc": slots[-1].astimezone(timezone.utc).isoformat() if complete else None,
                    (
                        "net_usd"
                        if policy.unit == "USD"
                        else "net_points"
                    ): (
                        float(slot_rows[-1][4]["session_net"])
                        if covered
                        else None
                    ),
                }
            )

        if not sessions:
            errors.add("no_complete_session_due")
        if coverage_broken:
            errors.add("incomplete_session_coverage")
        if ordered_equity:
            first = ordered_equity[0]
            if any(
                abs(float(first[key])) > 1e-9
                for key in (
                    "cumulative_gross",
                    "cumulative_cost",
                    "cumulative_net",
                    "cumulative_realized_net",
                    "open_mark",
                )
            ) or int(first["closed_trades"]) != 0:
                errors.add("nonzero_run_baseline")
            previous = first
            for current in ordered_equity[1:]:
                if (
                    float(current["cumulative_cost"])
                    < float(previous["cumulative_cost"]) - 1e-9
                    or int(current["closed_trades"]) < int(previous["closed_trades"])
                    or float(current["gross_wins"])
                    < float(previous["gross_wins"]) - 1e-9
                    or float(current["top_five_gross_wins"])
                    < float(previous["top_five_gross_wins"]) - 1e-9
                ):
                    errors.add("nonmonotonic_selected_economics")
                    break
                previous = current
            if any(row.get("reconciled") is not True for row in ordered_equity):
                errors.add("unreconciled_selected_economics")
            if any(row.get("attribution_complete") is not True for row in ordered_equity):
                errors.add("incomplete_selected_attribution")
            if any(row["safety_breaches"] for row in ordered_equity):
                errors.add("selected_safety_breach")

        high = 0.0
        drawdown = 0.0
        for value in equity_path:
            high = max(high, value)
            drawdown = max(drawdown, high - value)
        final = ordered_equity[-1] if ordered_equity else None
        net = float(final["cumulative_net"]) if final else 0.0
        costs = float(final["cumulative_cost"]) if final else 0.0
        closed_trades = int(final["closed_trades"]) if final else 0
        gross_wins = float(final["gross_wins"]) if final else 0.0
        top_five = float(final["top_five_gross_wins"]) if final else 0.0
        top_five_share = top_five / gross_wins if gross_wins > 0 else None
        session_net_key = "net_usd" if policy.unit == "USD" else "net_points"
        worst_session = min(
            (
                float(row[session_net_key])
                for row in sessions
                if row[session_net_key] is not None
            ),
            default=0.0,
        )
        if drawdown > policy.max_drawdown_points + 1e-9:
            errors.add("drawdown_limit_breached")
        if worst_session < -policy.max_session_loss_points - 1e-9:
            errors.add("session_loss_limit_breached")

        complete_sessions = sum(bool(row["complete"]) for row in sessions)
        first_slot = days[0][2][0].astimezone(timezone.utc) if days else run_started
        elapsed = max(0.0, (observed_at - first_slot).total_seconds())
        milestones = {}
        if not _prefix:
            complete_rows = [row for row in sessions if row["complete"]]
            for name, seconds, required_sessions in (
                ("24h", 24 * 3600, 1),
                ("48h", 48 * 3600, 2),
                ("five_session_week", 7 * 24 * 3600, 5),
            ):
                window_end = first_slot + timedelta(seconds=seconds)
                deadline = (
                    max(
                        window_end,
                        datetime.fromisoformat(str(complete_rows[required_sessions - 1]["completed_at_utc"])),
                    )
                    if len(complete_rows) >= required_sessions
                    else window_end
                )
                evidence_deadline = deadline + timedelta(seconds=policy.slot_tolerance_seconds)
                prefix = (
                    self.xsp_profitability_receipt(
                        policy=policy, as_of=evidence_deadline, _prefix=True,
                    )
                    if observed_at >= evidence_deadline
                    and len(complete_rows) >= required_sessions
                    else None
                )
                prefix_economics = prefix.get("economics") if prefix else None
                reasons = list(prefix.get("reasons", ())) if prefix else []
                if observed_at < evidence_deadline:
                    reasons.append("elapsed_time_incomplete")
                if len(complete_rows) < required_sessions:
                    reasons.append("eligible_sessions_incomplete")
                net_key = "net_usd" if policy.unit == "USD" else "net_points"
                if not prefix_economics or prefix_economics[net_key] <= 0:
                    reasons.append("net_not_positive")
                if name == "five_session_week" and prefix_economics:
                    if prefix_economics["closed_trades"] < policy.minimum_week_closed_trades:
                        reasons.append("insufficient_closed_trades")
                    share = prefix_economics["top_five_win_share"]
                    if share is None or share > policy.maximum_top_five_win_share:
                        reasons.append("win_concentration_exceeded")
                milestones[name] = {
                    "passed": not reasons,
                    "economic_window_end_utc": window_end.isoformat(),
                    "evidence_as_of_utc": evidence_deadline.isoformat(),
                    "required_elapsed_seconds": seconds,
                    "elapsed_seconds": max(0.0, min(elapsed, float(seconds))),
                    "required_complete_sessions": required_sessions,
                    "complete_sessions": prefix["clock"]["complete_sessions"] if prefix else complete_sessions,
                    "economics": prefix_economics,
                    "reasons": sorted(set(reasons)),
                }

        return {
            "schema": LIVE_PROFITABILITY_SCHEMA,
            "authority": "selected_reconciled_economics_only",
            "as_of_utc": observed_at.isoformat(),
            "status": (
                "INVALID_EVIDENCE"
                if errors
                else "PASSED"
                if not _prefix and all(row["passed"] for row in milestones.values())
                else "ACTIVE"
            ),
            "policy": {
                "run_id": policy.run_id,
                "strategy_id": policy.strategy_id,
                "strategy_version": policy.strategy_version,
                "config_fingerprint": policy.config_fingerprint,
                "capital_sleeve": policy.capital_sleeve,
                "unit": policy.unit,
                "equity_schema": policy.equity_schema,
                (
                    "max_drawdown_usd"
                    if policy.unit == "USD"
                    else "max_drawdown_points"
                ): policy.max_drawdown_points,
                (
                    "max_session_loss_usd"
                    if policy.unit == "USD"
                    else "max_session_loss_points"
                ): policy.max_session_loss_points,
                "minimum_week_closed_trades": policy.minimum_week_closed_trades,
                "maximum_top_five_win_share": policy.maximum_top_five_win_share,
                "slot_tolerance_seconds": policy.slot_tolerance_seconds,
            },
            "clock": {
                "run_started_at_utc": run_started.isoformat(),
                "coverage_started_at_utc": first_slot.isoformat(),
                "elapsed_seconds": elapsed,
                "complete_sessions": complete_sessions,
                "coverage_broken": coverage_broken,
            },
            "economics": {
                "unit": policy.unit,
                (
                    "gross_usd"
                    if policy.unit == "USD"
                    else "gross_points"
                ): float(final["cumulative_gross"]) if final else 0.0,
                (
                    "cost_usd"
                    if policy.unit == "USD"
                    else "cost_points"
                ): costs,
                (
                    "net_usd"
                    if policy.unit == "USD"
                    else "net_points"
                ): net,
                (
                    "realized_net_usd"
                    if policy.unit == "USD"
                    else "realized_net_points"
                ): (
                    float(final["cumulative_realized_net"])
                    if final
                    else 0.0
                ),
                (
                    "open_mark_usd"
                    if policy.unit == "USD"
                    else "open_mark_points"
                ): float(final["open_mark"]) if final else 0.0,
                (
                    "maximum_drawdown_usd"
                    if policy.unit == "USD"
                    else "maximum_drawdown_points"
                ): drawdown,
                (
                    "worst_session_usd"
                    if policy.unit == "USD"
                    else "worst_session_points"
                ): worst_session,
                "closed_trades": closed_trades,
                (
                    "gross_wins_usd"
                    if policy.unit == "USD"
                    else "gross_wins_points"
                ): gross_wins,
                (
                    "top_five_gross_wins_usd"
                    if policy.unit == "USD"
                    else "top_five_gross_wins_points"
                ): top_five,
                "top_five_win_share": top_five_share,
            },
            "sessions": sessions,
            "milestones": milestones,
            "reasons": sorted(errors),
        }

    def _append_unique(
        self,
        record: dict[str, object],
        *,
        key: str,
        unique_field: str | None = None,
        conflict: str = "calibration record conflicts with immutable evidence",
    ) -> dict[str, object]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical(record) + b"\n"
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                repaired = _repair_tail(handle)
                handle.seek(0)
                for existing_line in handle:
                    if not existing_line.strip():
                        continue
                    existing = json.loads(existing_line)
                    if not _record_address_valid(existing):
                        raise ValueError("existing calibration record has invalid content address")
                    if existing.get(key) == record[key]:
                        if repaired:
                            handle.flush()
                            os.fsync(handle.fileno())
                        return existing
                    if (
                        unique_field
                        and existing.get("kind") == record.get("kind")
                        and existing.get(unique_field) == record.get(unique_field)
                    ):
                        raise ValueError(conflict)
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record
