"""Immutable prospective coverage succession for XSP profitability proof."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ..engines.market import xsp_rth_cash_evaluation_slots
from ..time_utils import ET_ZONE
from .live_calibration import LiveCalibrationLedger
from .live_graduation import evidence_sha256, validate_live_graduation_receipt
from .xsp_profitability import (
    SELECTED_CASH_EQUITY_SCHEMA,
    XspProfitabilityPolicy,
)


XSP_PROFITABILITY_COVERAGE_EPOCH_SCHEMA = (
    "xsp.live-profitability-coverage-epoch.v1"
)


def _utc(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("XSP coverage timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _load(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: XSP coverage proof must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def build_xsp_profitability_coverage_epoch(
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    predecessor_receipt_paths: Sequence[Path],
    registered_at_utc: datetime | str,
    eligible_start_utc: datetime | str,
    repo_root: Path,
) -> dict[str, object]:
    """Freeze a prospective evidence clock without resetting live economics."""

    registered = _utc(registered_at_utc)
    eligible = _utc(eligible_start_utc)
    root = repo_root.resolve()
    selected_path = selection_path.resolve()
    try:
        selected_relative = selected_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("XSP coverage selection must live in the repository") from exc
    selection_id = str(selection.get("selection_id") or "")
    slots = xsp_rth_cash_evaluation_slots(eligible.astimezone(ET_ZONE).date())
    if (
        len(selection_id) != 64
        or any(value not in "0123456789abcdef" for value in selection_id)
        or registered >= eligible
        or eligible not in {slot.astimezone(timezone.utc) for slot in slots}
    ):
        raise ValueError("XSP coverage epoch identity is invalid")

    candidates = []
    checkpoint_fields = (
        "evaluation_as_of_utc",
        "strategy_id",
        "strategy_version",
        "trading_date",
        "session",
        "status",
        "evidence",
    )
    for row in records:
        evidence = row.get("evidence")
        equity = (
            evidence.get("selected_cash_equity")
            if isinstance(evidence, Mapping)
            else None
        )
        try:
            recorded_at = _utc(row["recorded_at_utc"])
            checkpoint = {field: row[field] for field in checkpoint_fields}
            addresses = {
                evidence_sha256(checkpoint),
                evidence_sha256(
                    {**checkpoint, "recorded_at_utc": row["recorded_at_utc"]}
                ),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if (
            row.get("kind") == "checkpoint"
            and row.get("checkpoint_id") in addresses
            and recorded_at <= registered
            and isinstance(equity, Mapping)
            and equity.get("run_id") == selection_id
        ):
            candidates.append((recorded_at, row, evidence, equity))
    if not candidates:
        raise ValueError("XSP coverage epoch has no terminal selected checkpoint")
    _recorded, terminal_row, terminal_evidence, terminal_equity = max(
        candidates,
        key=lambda item: (item[0], str(item[1].get("checkpoint_id") or "")),
    )
    broker = terminal_evidence.get("broker_state")
    try:
        gross = float(terminal_equity["cumulative_gross_usd"])
        cost = float(terminal_equity["cumulative_cost_usd"])
        net = float(terminal_equity["cumulative_net_usd"])
        realized = float(terminal_equity["cumulative_realized_net_usd"])
        open_mark = float(terminal_equity["open_mark_usd"])
        closed_trades = int(terminal_equity["closed_trades"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("XSP coverage terminal economics are invalid") from exc
    if (
        terminal_row.get("status") != "EVALUATED"
        or terminal_evidence.get("phase") != "STATE"
        or terminal_evidence.get("submitted_orders") != 0
        or not isinstance(broker, Mapping)
        or broker.get("positions") != {"SPXU": 0.0, "UPRO": 0.0}
        or broker.get("open_orders") != []
        or terminal_equity.get("schema") != SELECTED_CASH_EQUITY_SCHEMA
        or terminal_equity.get("reconciled") is not True
        or terminal_equity.get("attribution_complete") is not True
        or terminal_equity.get("safety_breaches") != []
        or not all(
            math.isfinite(value)
            for value in (gross, cost, net, realized, open_mark)
        )
        or cost < 0
        or closed_trades < 0
        or abs(net - gross + cost) > 1e-7
        or abs(net - realized - open_mark) > 1e-7
        or abs(open_mark) > 1e-9
    ):
        raise ValueError("XSP coverage epoch requires terminal flat broker truth")

    predecessor_receipts = []
    for receipt_path in predecessor_receipt_paths:
        resolved = receipt_path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("XSP predecessor receipt must live in the repository") from exc
        receipt, fingerprint = _load(resolved)
        validated = validate_live_graduation_receipt(receipt)
        subject = receipt.get("subject")
        target = receipt.get("target")
        cutoff = _utc(target.get("cutoff_utc")) if isinstance(target, Mapping) else None
        if (
            not isinstance(subject, Mapping)
            or subject.get("selection_id") != selection_id
            or cutoff is None
            or cutoff > registered
        ):
            raise ValueError("XSP predecessor graduation receipt is invalid")
        predecessor_receipts.append(
            {
                "path": relative,
                "sha256": fingerprint,
                "receipt_id": validated["receipt_id"],
                "milestone": target["milestone"],
                "cutoff_utc": target["cutoff_utc"],
                "verdict": receipt["verdict"],
            }
        )
    predecessor_receipts.sort(
        key=lambda row: (str(row["cutoff_utc"]), str(row["path"]))
    )
    if (
        len(predecessor_receipts) != 3
        or len({row["path"] for row in predecessor_receipts}) != 3
        or len({row["receipt_id"] for row in predecessor_receipts}) != 3
    ):
        raise ValueError("XSP coverage epoch requires all three predecessor receipts")

    body = {
        "schema": XSP_PROFITABILITY_COVERAGE_EPOCH_SCHEMA,
        "authority": "prospective_coverage_clock_only_cumulative_economics_unchanged",
        "registered_at_utc": registered.isoformat(),
        "eligible_start_utc": eligible.isoformat(),
        "selection": {
            "selection_id": selection_id,
            "path": selected_relative,
            "sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        },
        "terminal_checkpoint": {
            "checkpoint_id": terminal_row["checkpoint_id"],
            "recorded_at_utc": terminal_row["recorded_at_utc"],
            "evaluation_as_of_utc": terminal_row["evaluation_as_of_utc"],
            "evidence_sha256": evidence_sha256(terminal_evidence),
            "selected_cash_equity": dict(terminal_equity),
            "broker_positions": dict(broker["positions"]),
            "broker_open_orders": [],
            "submitted_orders": 0,
        },
        "predecessor_receipts": predecessor_receipts,
        "invariants": {
            "strategy_selection_changed": False,
            "runtime_risk_reset": False,
            "capital_plan_changed": False,
            "pre_epoch_economics_inherited": True,
            "missing_slots_backfilled": False,
            "profitability_threshold_changed": False,
            "broker_or_timer_mutated": False,
        },
        "submitted_orders": 0,
    }
    return {**body, "epoch_id": evidence_sha256(body)}


def load_xsp_profitability_coverage_epoch(
    path: Path,
    *,
    selection: Mapping[str, object],
    selection_path: Path,
    records: Sequence[Mapping[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    """Rebuild and validate one content-addressed XSP coverage epoch."""

    epoch, _fingerprint = _load(path)
    receipts = epoch.get("predecessor_receipts")
    if (
        epoch.get("schema") != XSP_PROFITABILITY_COVERAGE_EPOCH_SCHEMA
        or not isinstance(receipts, Sequence)
        or isinstance(receipts, (str, bytes))
    ):
        raise ValueError("invalid XSP profitability coverage epoch")
    try:
        receipt_paths = [
            repo_root / str(row["path"])
            for row in receipts
            if isinstance(row, Mapping)
        ]
        rebuilt = build_xsp_profitability_coverage_epoch(
            selection=selection,
            selection_path=selection_path,
            records=records,
            predecessor_receipt_paths=receipt_paths,
            registered_at_utc=epoch["registered_at_utc"],
            eligible_start_utc=epoch["eligible_start_utc"],
            repo_root=repo_root,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("invalid XSP profitability coverage epoch") from exc
    if rebuilt != epoch:
        raise ValueError("XSP profitability coverage epoch identity drift")
    return epoch


def xsp_profitability_policy_with_coverage_epoch(
    policy: XspProfitabilityPolicy,
    epoch: Mapping[str, object],
) -> XspProfitabilityPolicy:
    """Bind one validated prospective coverage clock to existing economics."""

    selection = epoch.get("selection")
    if (
        epoch.get("schema") != XSP_PROFITABILITY_COVERAGE_EPOCH_SCHEMA
        or not isinstance(selection, Mapping)
        or selection.get("selection_id") != policy.run_id
    ):
        raise ValueError("coverage epoch does not own this XSP run")
    return replace(
        policy,
        coverage_epoch_id=str(epoch["epoch_id"]),
        coverage_started_at_utc=str(epoch["eligible_start_utc"]),
    )


_CHECKPOINT_FIELDS = (
    "evaluation_as_of_utc",
    "strategy_id",
    "strategy_version",
    "trading_date",
    "session",
    "status",
    "evidence",
    "recorded_at_utc",
)
_OFFSET_FIELDS = {
    "cumulative_gross_usd": "cumulative_gross_usd",
    "cumulative_cost_usd": "cumulative_cost_usd",
    "cumulative_net_usd": "cumulative_net_usd",
    "cumulative_realized_net_usd": "cumulative_realized_net_usd",
    "closed_trades": "closed_trades",
    "gross_wins_usd": "gross_wins_usd",
    "top_five_gross_wins_usd": "top_five_gross_wins_usd",
}


def _selected_points(
    records: Sequence[Mapping[str, object]],
    policy: XspProfitabilityPolicy,
) -> tuple[list[tuple[datetime, datetime, Mapping[str, object], Mapping[str, object]]], set[str]]:
    points = []
    errors = set()
    for row in records:
        evidence = row.get("evidence")
        equity = (
            evidence.get("selected_cash_equity")
            if isinstance(evidence, Mapping)
            else None
        )
        if (
            row.get("kind") != "checkpoint"
            or row.get("strategy_id") != policy.strategy_id
            or row.get("strategy_version") != policy.strategy_version
            or not isinstance(equity, Mapping)
            or equity.get("run_id") != policy.run_id
        ):
            continue
        try:
            evaluation = _utc(row["evaluation_as_of_utc"])
            recorded = _utc(row["recorded_at_utc"])
            addressed = row.get("checkpoint_id") in {
                evidence_sha256({field: row[field] for field in _CHECKPOINT_FIELDS}),
                evidence_sha256(
                    {
                        field: row[field]
                        for field in _CHECKPOINT_FIELDS
                        if field != "recorded_at_utc"
                    }
                ),
            }
        except (KeyError, TypeError, ValueError):
            errors.add("invalid_checkpoint_identity")
            continue
        if not addressed:
            errors.add("unaddressed_checkpoint_time")
            continue
        points.append((evaluation, recorded, row, equity))
    points.sort(key=lambda item: (item[0], item[1], str(item[2].get("checkpoint_id"))))
    return points, errors


def _relative_coverage_records(
    points: Sequence[tuple[datetime, datetime, Mapping[str, object], Mapping[str, object]]],
    *,
    coverage_started: datetime,
    terminal: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    projected = []
    for evaluation, recorded, row, _equity in points:
        if evaluation < coverage_started or recorded < coverage_started:
            continue
        clone = deepcopy(dict(row))
        evidence = clone["evidence"]
        equity = evidence["selected_cash_equity"]
        equity["run_started_at_utc"] = coverage_started.isoformat()
        for current, prior in _OFFSET_FIELDS.items():
            equity[current] = float(equity[current]) - float(terminal[prior])
        equity["closed_trades"] = int(round(float(equity["closed_trades"])))
        clone["checkpoint_id"] = evidence_sha256(
            {field: clone[field] for field in _CHECKPOINT_FIELDS}
        )
        projected.append(clone)
    return tuple(projected)


def _path_risk(
    points: Sequence[tuple[datetime, datetime, Mapping[str, object], Mapping[str, object]]],
    cutoff: datetime,
) -> tuple[float, float]:
    path = [0.0]
    sessions = []
    for evaluation, recorded, _row, equity in points:
        if evaluation <= cutoff and recorded <= cutoff:
            path.append(float(equity["cumulative_net_usd"]))
            sessions.append(float(equity["session_net_usd"]))
    peak = drawdown = 0.0
    for value in path:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return drawdown, min(sessions, default=0.0)


def _cumulative_economics(
    relative: Mapping[str, object],
    terminal: Mapping[str, object],
    *,
    drawdown: float,
    worst_session: float,
) -> dict[str, object]:
    economics = dict(relative)
    for output, baseline in (
        ("gross_usd", "cumulative_gross_usd"),
        ("cost_usd", "cumulative_cost_usd"),
        ("net_usd", "cumulative_net_usd"),
        ("realized_net_usd", "cumulative_realized_net_usd"),
        ("open_mark_usd", "open_mark_usd"),
        ("closed_trades", "closed_trades"),
        ("gross_wins_usd", "gross_wins_usd"),
        ("top_five_gross_wins_usd", "top_five_gross_wins_usd"),
    ):
        economics[output] = float(economics[output]) + float(terminal[baseline])
    economics["closed_trades"] = int(round(float(economics["closed_trades"])))
    wins = float(economics["gross_wins_usd"])
    economics["top_five_win_share"] = (
        float(economics["top_five_gross_wins_usd"]) / wins if wins > 0 else None
    )
    economics["maximum_drawdown_usd"] = drawdown
    economics["worst_session_usd"] = worst_session
    return economics


def xsp_profitability_receipt_with_coverage_epoch(
    *,
    ledger: LiveCalibrationLedger,
    policy: XspProfitabilityPolicy,
    epoch: Mapping[str, object],
    as_of: datetime | str,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reduce a prospective coverage clock while retaining cumulative economics."""

    observed = _utc(as_of)
    terminal_row = epoch.get("terminal_checkpoint")
    terminal = (
        terminal_row.get("selected_cash_equity")
        if isinstance(terminal_row, Mapping)
        else None
    )
    if (
        not isinstance(terminal, Mapping)
        or epoch.get("epoch_id") != policy.coverage_epoch_id
        or epoch.get("eligible_start_utc") != policy.coverage_started_at_utc
        or terminal.get("run_id") != policy.run_id
        or terminal.get("schema") != SELECTED_CASH_EQUITY_SCHEMA
    ):
        raise ValueError("invalid XSP profitability coverage epoch evidence")
    coverage_started = _utc(policy.coverage_started_at_utc)
    run_started = _utc(terminal["run_started_at_utc"])
    if coverage_started < run_started:
        raise ValueError("XSP profitability coverage epoch predates selected run")
    points, evidence_errors = _selected_points(records, policy)
    if points:
        earliest = points[0][3]
        if any(
            abs(float(earliest[field])) > 1e-9
            for field in (
                "cumulative_gross_usd",
                "cumulative_cost_usd",
                "cumulative_net_usd",
                "cumulative_realized_net_usd",
                "open_mark_usd",
            )
        ) or int(earliest["closed_trades"]) != 0:
            evidence_errors.add("nonzero_original_run_baseline")
    else:
        evidence_errors.add("missing_original_run_baseline")
    post_epoch = [item[3] for item in points if item[0] >= coverage_started]
    if post_epoch and any(
        float(post_epoch[0][current]) < float(terminal[prior]) - 1e-9
        for current, prior in (
            ("cumulative_cost_usd", "cumulative_cost_usd"),
            ("closed_trades", "closed_trades"),
            ("gross_wins_usd", "gross_wins_usd"),
            ("top_five_gross_wins_usd", "top_five_gross_wins_usd"),
        )
    ):
        evidence_errors.add("coverage_epoch_economics_regressed")
    relative_records = _relative_coverage_records(
        points,
        coverage_started=coverage_started,
        terminal=terminal,
    )
    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=observed,
        _records=relative_records,
    )
    receipt["policy"].update(
        {
            "coverage_epoch_id": policy.coverage_epoch_id,
            "coverage_started_at_utc": policy.coverage_started_at_utc,
        }
    )
    receipt["clock"].update(
        {
            "run_started_at_utc": run_started.isoformat(),
            "coverage_started_at_utc": coverage_started.isoformat(),
            "coverage_epoch_id": policy.coverage_epoch_id,
        }
    )
    if isinstance(receipt.get("economics"), Mapping):
        drawdown, worst = _path_risk(points, observed)
        receipt["economics"] = _cumulative_economics(
            receipt["economics"], terminal, drawdown=drawdown, worst_session=worst
        )
    milestones = receipt.get("milestones")
    if isinstance(milestones, Mapping):
        for name, milestone in milestones.items():
            economics = milestone.get("economics")
            if not isinstance(economics, Mapping):
                continue
            cutoff = _utc(milestone["evidence_as_of_utc"])
            drawdown, worst = _path_risk(points, cutoff)
            economics = _cumulative_economics(
                economics, terminal, drawdown=drawdown, worst_session=worst
            )
            milestone["economics"] = economics
            reasons = set(milestone.get("reasons", ()))
            for reason in (
                "net_not_positive",
                "insufficient_closed_trades",
                "win_concentration_exceeded",
                "drawdown_limit_breached",
                "session_loss_limit_breached",
            ):
                reasons.discard(reason)
            if float(economics["net_usd"]) <= 0:
                reasons.add("net_not_positive")
            if drawdown > policy.max_drawdown_points + 1e-9:
                reasons.add("drawdown_limit_breached")
            if worst < -policy.max_session_loss_points - 1e-9:
                reasons.add("session_loss_limit_breached")
            if name == "five_session_week":
                if int(economics["closed_trades"]) < policy.minimum_week_closed_trades:
                    reasons.add("insufficient_closed_trades")
                share = economics["top_five_win_share"]
                if share is None or share > policy.maximum_top_five_win_share:
                    reasons.add("win_concentration_exceeded")
            milestone["reasons"] = sorted(reasons)
            milestone["passed"] = not reasons
    base_status = receipt["status"]
    reasons = set(receipt.get("reasons", ())) | evidence_errors
    for reason in ("drawdown_limit_breached", "session_loss_limit_breached"):
        reasons.discard(reason)
    economics = receipt.get("economics")
    if isinstance(economics, Mapping):
        if float(economics["maximum_drawdown_usd"]) > policy.max_drawdown_points + 1e-9:
            reasons.add("drawdown_limit_breached")
        if float(economics["worst_session_usd"]) < -policy.max_session_loss_points - 1e-9:
            reasons.add("session_loss_limit_breached")
    receipt["reasons"] = sorted(reasons)
    invalid = bool(evidence_errors) or base_status == "INVALID_EVIDENCE" or bool(
        reasons & {"drawdown_limit_breached", "session_loss_limit_breached"}
    )
    receipt["status"] = (
        "INVALID_EVIDENCE"
        if invalid
        else "NOT_STARTED"
        if base_status == "NOT_STARTED"
        else "PASSED"
        if isinstance(milestones, Mapping)
        and all(bool(row.get("passed")) for row in milestones.values())
        else "ACTIVE"
    )
    return receipt
