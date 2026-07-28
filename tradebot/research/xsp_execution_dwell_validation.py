"""Validate one ranked XSP v2 transport against frozen execution books."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from tradebot.engines.execution import execution_price


PREREGISTRATION_SCHEMA = (
    "xsp.network-b-symbol-dwell-validation-preregistration.v2"
)
RANKING_SCHEMA = "xsp.opening-edge-v2-spyu-selection-ranking-result.v1"
RESULT_SCHEMA = "xsp.network-b-symbol-dwell-validation-result.v1"
PHASES = (
    ("OPTIMISTIC", 0.0, 6.0),
    ("MID", 6.0, 12.0),
    ("AGGRESSIVE", 12.0, 18.0),
    ("CROSS", 18.0, 24.0),
)
CONTRACT = SimpleNamespace(secType="STK", minTick=0.01)
EXPECTED_ACCEPTANCE = {
    "fixed_ranked_profile_maximum_quantity_required": True,
    "no_reduction_in_full_quantity_fill_coverage": True,
    "SPYU_BUY_total_savings_vs_control_must_be_positive": True,
    "SPYU_BUY_strict_majority_utc_calendar_quarters_nonnegative": True,
    "all_SPXU_and_SELL_paths_unchanged": True,
    "no_runtime_change_on_failure": True,
    "does_not_select_transport": True,
    "does_not_start_clock": True,
}


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


def _utc(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantity(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if float(value) != result or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _quote_at(
    rows: list[Mapping[str, object]],
    target: datetime,
) -> Mapping[str, object] | None:
    before = [row for row in rows if _utc(row["time_utc"]) <= target]
    if before:
        return before[-1]
    after = [row for row in rows if _utc(row["time_utc"]) > target]
    return after[0] if after else None


def _limit_price(
    mode: str,
    quote: Mapping[str, object],
    *,
    action: str,
) -> float:
    bid = _number(quote.get("bid"), name="bid")
    ask = _number(quote.get("ask"), name="ask")
    value = execution_price(
        CONTRACT,
        None,
        mode,
        action,
        bid=bid,
        ask=ask,
        last=None,
        fallback_price=(bid + ask) / 2.0,
        custom_price=None,
    )
    if value is None:
        raise ValueError(f"{mode}: shared execution price is unavailable")
    return float(value)


def _fill(
    document: Mapping[str, object],
    *,
    action: str,
    quantity: int,
    immediate_cross: bool,
) -> dict[str, object]:
    decision = _utc(document["decision_and_next_open_time_utc"])
    rows_raw = document.get("ticks_in_window")
    if not isinstance(rows_raw, list):
        raise ValueError("decision book ticks are missing")
    rows = sorted(
        (row for row in rows_raw if isinstance(row, Mapping)),
        key=lambda row: _utc(row["time_utc"]),
    )
    arrival = _quote_at(rows, decision)
    if arrival is None:
        return {"filled": False, "reason": "arrival_quote_missing"}
    arrival_cross = _number(
        arrival.get("ask" if action == "BUY" else "bid"),
        name="arrival cross",
    )
    phases = (("CROSS", 0.0, 0.0),) if immediate_cross else PHASES
    for mode, start_sec, end_sec in phases:
        start = decision + timedelta(seconds=start_sec)
        quote = _quote_at(rows, start)
        if quote is None:
            continue
        limit = _limit_price(mode, quote, action=action)
        candidates = [quote]
        if end_sec > start_sec:
            end = decision + timedelta(seconds=end_sec)
            candidates.extend(
                row for row in rows if start < _utc(row["time_utc"]) < end
            )
        for candidate in candidates:
            executable_price = _number(
                candidate.get("ask" if action == "BUY" else "bid"),
                name="executable price",
            )
            displayed_size = _number(
                candidate.get(
                    "ask_size" if action == "BUY" else "bid_size"
                ),
                name="displayed size",
            )
            price_reached = (
                executable_price <= limit + 1e-9
                if action == "BUY"
                else executable_price >= limit - 1e-9
            )
            if price_reached and displayed_size >= quantity:
                return {
                    "filled": True,
                    "mode": mode,
                    "latency_sec": max(
                        start_sec,
                        (_utc(candidate["time_utc"]) - decision).total_seconds(),
                    ),
                    "arrival_cross": arrival_cross,
                    "fill_limit": limit,
                    "executable_price": executable_price,
                    "displayed_size": displayed_size,
                }
    return {
        "filled": False,
        "reason": "full_quantity_not_filled",
        "arrival_cross": arrival_cross,
    }


def _manifest(
    books: Path,
    *,
    ledger_sha256: str,
) -> dict[str, object]:
    path = books / "manifest.json"
    manifest = _load(path)
    complete = int(manifest.get("complete", -1))
    eligible = int(manifest.get("eligible_rth_decisions", -2))
    remaining = int(manifest.get("remaining", -1))
    incomplete = int(manifest.get("incomplete_or_error", -1))
    if (
        manifest.get("schema")
        != "xsp.network-b-decision-book-manifest.v1"
        or manifest.get("read_only") is not True
        or manifest.get("submitted_orders") != 0
        or manifest.get("ledger_sha256") != ledger_sha256
        or complete <= 0
        or eligible <= 0
        or complete > eligible
        or remaining != eligible - complete
        or incomplete != remaining
        or not isinstance(manifest.get("extraneous_files_ignored"), int)
    ):
        raise ValueError(f"{books}: invalid terminal decision-book manifest")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "eligible_rth_decisions": eligible,
        "complete": complete,
        "unavailable_or_incomplete": remaining,
        "extraneous_files_ignored": manifest["extraneous_files_ignored"],
    }


def _expected_tasks(
    ledger_path: Path,
    *,
    action: str,
    cutoff: datetime,
) -> tuple[dict[tuple[str, str, str], Mapping[str, object]], str]:
    ledger = _load(ledger_path)
    expected_schema = (
        "xsp.network-b-entry-task-ledger.v2"
        if action == "BUY"
        else "xsp.network-b-exit-task-ledger.v2"
    )
    rows = ledger.get("ledger")
    if (
        ledger.get("schema") != expected_schema
        or ledger.get("authority")
        != "causal_historical_execution_validation_only"
        or ledger.get("order_authority") != "none"
        or ledger.get("submitted_orders") != 0
        or not isinstance(rows, list)
    ):
        raise ValueError(f"{ledger_path}: invalid task ledger")
    expected: dict[
        tuple[str, str, str], Mapping[str, object]
    ] = {}
    for trade in rows:
        if not isinstance(trade, Mapping) or trade.get("lane") != "rth":
            continue
        if trade.get("execution_action") != action:
            raise ValueError(f"{ledger_path}: execution action drift")
        direction = str(trade.get("direction"))
        if direction not in ("up", "down"):
            raise ValueError(f"{ledger_path}: invalid direction")
        decision = _utc(trade["entry_time"]) - timedelta(minutes=5)
        if decision >= cutoff:
            continue
        symbol = "SPYU" if direction == "up" else "SPXU"
        key = (decision.isoformat(), symbol, direction)
        if key in expected:
            raise ValueError(f"{ledger_path}: duplicate decision identity")
        expected[key] = trade
    return expected, _sha256(ledger_path)


def _validation_books(
    books: Path,
    *,
    expected: Mapping[
        tuple[str, str, str], Mapping[str, object]
    ],
) -> dict[tuple[str, str, str], tuple[Path, Mapping[str, object]]]:
    matched: dict[
        tuple[str, str, str], tuple[Path, Mapping[str, object]]
    ] = {}
    for path in sorted(books.glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = _load(path)
        key = (
            _utc(document["decision_and_next_open_time_utc"]).isoformat(),
            str(document.get("symbol")),
            str(document.get("direction")),
        )
        if key not in expected:
            continue
        if document.get("status") != "complete":
            continue
        if (
            document.get("schema") != "xsp.network-b-decision-book.v1"
            or document.get("read_only") is not True
            or document.get("submitted_orders") != 0
            or document.get("lane") != "rth"
            or document.get("source_trade") != expected[key]
        ):
            raise ValueError(f"{path}: validation book identity drift")
        if key in matched:
            raise ValueError(f"{path}: duplicate validation book")
        matched[key] = (path, document)
    if not matched:
        raise ValueError(f"{books}: no complete validation decision books")
    return matched


def _quarter(value: datetime) -> str:
    return f"{value.year:04d}-Q{(value.month - 1) // 3 + 1}"


def validate_symbol_dwell(
    *,
    preregistration_path: Path,
    ranking_path: Path,
    entry_ledger_path: Path,
    exit_ledger_path: Path,
    entry_books: Path,
    exit_books: Path,
) -> dict[str, object]:
    """Return a deterministic non-submitting validation receipt."""

    preregistration = _load(preregistration_path)
    validation = preregistration.get("validation")
    source_identity = preregistration.get("source_identity")
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("authority")
        != "causal_historical_execution_validation_only"
        or preregistration.get("order_authority") != "none"
        or preregistration.get("submitted_orders") != 0
        or preregistration.get("acceptance") != EXPECTED_ACCEPTANCE
        or not isinstance(validation, Mapping)
        or validation.get("outcomes_inspected_at_freeze") is not False
        or validation.get("calendar_partition")
        != "UTC calendar quarter of decision_and_next_open_time_utc"
        or not isinstance(source_identity, Mapping)
    ):
        raise ValueError("invalid symbol-dwell preregistration")
    minimum_quarters = int(
        validation.get("minimum_nonempty_calendar_quarters", 0)
    )
    if minimum_quarters < 2:
        raise ValueError("calendar-quarter minimum is too small")
    cutoff = _utc(validation["books_must_be_strictly_earlier_than_utc"])

    ranking = _load(ranking_path)
    if (
        ranking.get("schema") != RANKING_SCHEMA
        or ranking.get("authority") != "research_ranking_only"
        or ranking.get("order_authority") != "none"
        or ranking.get("profitability_clock_started") is not False
        or ranking.get("selected_shadow_created") is not False
    ):
        raise ValueError("invalid ranking receipt")
    base = {
        "schema": RESULT_SCHEMA,
        "observed_at_utc": ranking.get("observed_at_utc"),
        "authority": "historical_execution_validation_only",
        "order_authority": "none",
        "submitted_orders": 0,
        "profitability_clock_started": False,
        "selected_shadow_created": False,
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
        },
        "ranking": {
            "path": str(ranking_path),
            "sha256": _sha256(ranking_path),
        },
    }
    nominee = ranking.get("nominee")
    if ranking.get("verdict") == "HOLD" and nominee is None:
        receipt = {
            **base,
            "nominee_id": None,
            "verdict": "NOT_APPLICABLE_NO_NOMINEE",
            "reason": "ranking_produced_no_nominee",
            "validation": None,
        }
        receipt["identity_sha256"] = _fingerprint(receipt)
        return receipt
    if (
        ranking.get("verdict") != "NOMINEE_STILL_HOLD"
        or not isinstance(nominee, Mapping)
    ):
        raise ValueError("ranking nominee state is invalid")
    quantities_raw = nominee.get("frozen_max_quantities")
    ranges_raw = nominee.get("historical_quantity_ranges")
    if (
        not isinstance(quantities_raw, Mapping)
        or set(quantities_raw) != {"SPYU", "SPXU"}
        or not isinstance(ranges_raw, Mapping)
        or set(ranges_raw) != {"SPYU", "SPXU"}
    ):
        raise ValueError("ranked quantity identity is missing")
    quantities = {
        symbol: _quantity(
            quantities_raw[symbol],
            name=f"{symbol} frozen maximum quantity",
        )
        for symbol in ("SPYU", "SPXU")
    }
    for symbol, quantity in quantities.items():
        bounds = ranges_raw[symbol]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or _quantity(bounds[1], name=f"{symbol} upper quantity")
            != quantity
        ):
            raise ValueError("ranked quantity identity drift")

    expected_entry_sha = str(source_identity.get("entry_ledger_sha256"))
    expected_exit_sha = str(source_identity.get("exit_ledger_sha256"))
    entry_expected, entry_sha = _expected_tasks(
        entry_ledger_path,
        action="BUY",
        cutoff=cutoff,
    )
    exit_expected, exit_sha = _expected_tasks(
        exit_ledger_path,
        action="SELL",
        cutoff=cutoff,
    )
    if entry_sha != expected_entry_sha or exit_sha != expected_exit_sha:
        raise ValueError("task-ledger identity drift")
    manifests = {
        "BUY": _manifest(entry_books, ledger_sha256=entry_sha),
        "SELL": _manifest(exit_books, ledger_sha256=exit_sha),
    }
    books_by_action = {
        "BUY": _validation_books(entry_books, expected=entry_expected),
        "SELL": _validation_books(exit_books, expected=exit_expected),
    }

    control: list[dict[str, object]] = []
    challenger: list[dict[str, object]] = []
    for action in ("BUY", "SELL"):
        for key in sorted(books_by_action[action]):
            path, document = books_by_action[action][key]
            symbol = key[1]
            quantity = quantities[symbol]
            control_fill = _fill(
                document,
                action=action,
                quantity=quantity,
                immediate_cross=False,
            )
            challenger_fill = (
                _fill(
                    document,
                    action=action,
                    quantity=quantity,
                    immediate_cross=True,
                )
                if action == "BUY" and symbol == "SPYU"
                else dict(control_fill)
            )
            identity = {
                "book": path.name,
                "book_sha256": _sha256(path),
                "decision_time_utc": key[0],
                "symbol": symbol,
                "direction": key[2],
                "action": action,
                "quantity": quantity,
            }
            control.append({**identity, **control_fill})
            challenger.append({**identity, **challenger_fill})

    control_filled = sum(row["filled"] is True for row in control)
    challenger_filled = sum(row["filled"] is True for row in challenger)
    full_coverage = (
        control_filled == len(control)
        and challenger_filled == len(challenger)
    )
    unchanged_control = [
        row
        for row in control
        if not (row["action"] == "BUY" and row["symbol"] == "SPYU")
    ]
    unchanged_challenger = [
        row
        for row in challenger
        if not (row["action"] == "BUY" and row["symbol"] == "SPYU")
    ]
    unchanged = _fingerprint(unchanged_control) == _fingerprint(
        unchanged_challenger
    )
    challenger_by_book = {
        (row["action"], row["book"]): row for row in challenger
    }
    savings_rows = []
    quarters: dict[str, float] = defaultdict(float)
    for row in control:
        if row["action"] != "BUY" or row["symbol"] != "SPYU":
            continue
        other = challenger_by_book[(row["action"], row["book"])]
        if row["filled"] is not True or other["filled"] is not True:
            continue
        per_share = float(row["fill_limit"]) - float(other["fill_limit"])
        savings_usd = per_share * int(row["quantity"])
        period = _quarter(_utc(row["decision_time_utc"]))
        quarters[period] += savings_usd
        savings_rows.append(
            {
                "book": row["book"],
                "decision_time_utc": row["decision_time_utc"],
                "quantity": row["quantity"],
                "control_fill_limit": row["fill_limit"],
                "challenger_fill_limit": other["fill_limit"],
                "savings_per_share": per_share,
                "savings_usd": savings_usd,
                "calendar_quarter": period,
            }
        )
    quarter_rows = {
        period: round(value, 8)
        for period, value in sorted(quarters.items())
    }
    nonnegative_quarters = sum(value >= -1e-9 for value in quarters.values())
    strict_majority = (
        len(quarters) >= minimum_quarters
        and nonnegative_quarters > len(quarters) / 2
    )
    total_savings = sum(row["savings_usd"] for row in savings_rows)
    passed = (
        full_coverage
        and unchanged
        and len(savings_rows)
        == sum(
            row["action"] == "BUY" and row["symbol"] == "SPYU"
            for row in control
        )
        and total_savings > 1e-9
        and strict_majority
    )
    result = {
        "nominee_id": nominee.get("nominee_id"),
        "fixed_entry_notional_usd": nominee.get(
            "fixed_entry_notional_usd"
        ),
        "frozen_max_quantities": quantities,
        "cutoff_utc": cutoff.isoformat(),
        "calendar_partition": validation["calendar_partition"],
        "minimum_nonempty_calendar_quarters": minimum_quarters,
        "manifests": manifests,
        "validation_book_counts": {
            "BUY": len(books_by_action["BUY"]),
            "SELL": len(books_by_action["SELL"]),
            "total": len(control),
            "eligible_before_cutoff": {
                "BUY": len(entry_expected),
                "SELL": len(exit_expected),
            },
            "unavailable_or_incomplete_before_cutoff": {
                "BUY": len(entry_expected) - len(books_by_action["BUY"]),
                "SELL": len(exit_expected) - len(books_by_action["SELL"]),
            },
        },
        "fill_coverage": {
            "control": f"{control_filled}/{len(control)}",
            "challenger": f"{challenger_filled}/{len(challenger)}",
            "no_reduction": full_coverage,
        },
        "unchanged_SPXU_and_SELL_paths": unchanged,
        "SPYU_BUY_savings_vs_control": {
            "trades": len(savings_rows),
            "total_usd": round(total_savings, 8),
            "calendar_quarters_usd": quarter_rows,
            "nonnegative_quarters": nonnegative_quarters,
            "strict_majority_nonnegative": strict_majority,
        },
        "control_rows": control,
        "challenger_rows": challenger,
        "savings_rows": savings_rows,
    }
    receipt = {
        **base,
        "nominee_id": nominee.get("nominee_id"),
        "verdict": (
            "DWELL_VALIDATION_PASS_SELECTION_STILL_HOLD"
            if passed
            else "HOLD"
        ),
        "reason": (
            "full_quantity_symbol_dwell_validation_passed"
            if passed
            else "full_quantity_symbol_dwell_validation_failed"
        ),
        "validation": result,
    }
    receipt["identity_sha256"] = _fingerprint(receipt)
    return receipt


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
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--entry-ledger", type=Path, required=True)
    parser.add_argument("--exit-ledger", type=Path, required=True)
    parser.add_argument("--entry-books", type=Path, required=True)
    parser.add_argument("--exit-books", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate_symbol_dwell(
        preregistration_path=args.preregistration,
        ranking_path=args.ranking,
        entry_ledger_path=args.entry_ledger,
        exit_ledger_path=args.exit_ledger,
        entry_books=args.entry_books,
        exit_books=args.exit_books,
    )
    _write_atomic(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
