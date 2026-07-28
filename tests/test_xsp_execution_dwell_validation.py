from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tradebot.engines.execution import _exec_ladder_mode
from tradebot.research.xsp_execution_dwell_validation import (
    EXPECTED_ACCEPTANCE,
    PHASES,
    validate_symbol_dwell,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_dwell_windows_match_the_shared_execution_ladder() -> None:
    assert [
        (mode, _exec_ladder_mode(start), _exec_ladder_mode(end - 1e-9))
        for mode, start, end in PHASES
    ] == [(mode, mode, mode) for mode, _, _ in PHASES]
    assert _exec_ladder_mode(PHASES[-1][2]) is None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trade(decision: datetime, *, action: str) -> dict[str, object]:
    return {
        "direction": "up",
        "entry_price": 500.0,
        "entry_time": (decision + timedelta(minutes=5)).isoformat(),
        "exit_price": 501.0,
        "exit_reason": "flip",
        "exit_time": (decision + timedelta(hours=1)).isoformat(),
        "lane": "rth",
        "pnl": 1.0,
        "position_entry_time": decision.isoformat(),
        "position_exit_time": (decision + timedelta(hours=1)).isoformat(),
        "execution_action": action,
    }


def _ledger(
    path: Path,
    decisions: list[datetime],
    *,
    action: str,
) -> Path:
    return _write(
        path,
        {
            "schema": (
                "xsp.network-b-entry-task-ledger.v2"
                if action == "BUY"
                else "xsp.network-b-exit-task-ledger.v2"
            ),
            "authority": "causal_historical_execution_validation_only",
            "order_authority": "none",
            "submitted_orders": 0,
            "ledger": [
                _trade(decision, action=action) for decision in decisions
            ],
        },
    )


def _ticks(
    decision: datetime,
    *,
    action: str,
    immediate_ask: float = 10.0,
) -> list[dict[str, object]]:
    if action == "BUY":
        quotes = (
            (0, 9.90, immediate_ask),
            (6, 9.95, 10.05),
            (12, 10.00, 10.10),
            (18, 10.05, 10.10),
            (46, 10.05, 10.10),
        )
    else:
        quotes = (
            (0, 10.00, 10.10),
            (6, 9.95, 10.05),
            (12, 9.90, 10.00),
            (18, 9.90, 10.00),
            (46, 9.90, 10.00),
        )
    return [
        {
            "time_utc": (decision + timedelta(seconds=seconds)).isoformat(),
            "bid": bid,
            "bid_size": 100,
            "ask": ask,
            "ask_size": 100,
        }
        for seconds, bid, ask in quotes
    ]


def _books(
    directory: Path,
    ledger_path: Path,
    decisions: list[datetime],
    *,
    action: str,
    second_immediate_ask: float = 10.0,
) -> Path:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for index, (decision, trade) in enumerate(
        zip(decisions, ledger["ledger"], strict=True)
    ):
        _write(
            directory / f"{index:02d}-SPYU.json",
            {
                "schema": "xsp.network-b-decision-book.v1",
                "status": "complete",
                "read_only": True,
                "submitted_orders": 0,
                "symbol": "SPYU",
                "direction": "up",
                "lane": "rth",
                "decision_and_next_open_time_utc": decision.isoformat(),
                "ticks_in_window": _ticks(
                    decision,
                    action=action,
                    immediate_ask=(
                        second_immediate_ask if index == 1 else 10.0
                    ),
                ),
                "source_trade": trade,
            },
        )
    _write(
        directory / "manifest.json",
        {
            "schema": "xsp.network-b-decision-book-manifest.v1",
            "ledger_sha256": _sha256(ledger_path),
            "eligible_rth_decisions": len(decisions),
            "complete": len(decisions),
            "incomplete_or_error": 0,
            "missing": 0,
            "remaining": 0,
            "extraneous_files_ignored": 0,
            "read_only": True,
            "submitted_orders": 0,
        },
    )
    return directory


def _ranking(path: Path, *, nominee: bool = True) -> Path:
    selected = (
        {
            "nominee_id": "n" * 64,
            "fixed_entry_notional_usd": 650.0,
            "historical_quantity_ranges": {
                "SPYU": [20, 25],
                "SPXU": [50, 60],
            },
            "frozen_max_quantities": {"SPYU": 25, "SPXU": 60},
        }
        if nominee
        else None
    )
    return _write(
        path,
        {
            "schema": "xsp.opening-edge-v2-spyu-selection-ranking-result.v1",
            "observed_at_utc": "2026-07-28T23:15:00+00:00",
            "authority": "research_ranking_only",
            "order_authority": "none",
            "profitability_clock_started": False,
            "selected_shadow_created": False,
            "verdict": "NOMINEE_STILL_HOLD" if nominee else "HOLD",
            "nominee": selected,
        },
    )


def _preregistration(
    path: Path,
    *,
    entry_ledger: Path,
    exit_ledger: Path,
) -> Path:
    return _write(
        path,
        {
            "schema": (
                "xsp.network-b-symbol-dwell-validation-preregistration.v2"
            ),
            "authority": "causal_historical_execution_validation_only",
            "order_authority": "none",
            "submitted_orders": 0,
            "validation": {
                "outcomes_inspected_at_freeze": False,
                "books_must_be_strictly_earlier_than_utc": (
                    "2025-07-30T14:20:00Z"
                ),
                "calendar_partition": (
                    "UTC calendar quarter of "
                    "decision_and_next_open_time_utc"
                ),
                "minimum_nonempty_calendar_quarters": 2,
            },
            "source_identity": {
                "entry_ledger_sha256": _sha256(entry_ledger),
                "exit_ledger_sha256": _sha256(exit_ledger),
            },
            "acceptance": EXPECTED_ACCEPTANCE,
        },
    )


def _inputs(tmp_path: Path, *, second_ask: float = 10.0):
    decisions = [
        datetime(2025, 1, 2, 15, 30, tzinfo=timezone.utc),
        datetime(2025, 4, 2, 14, 30, tzinfo=timezone.utc),
    ]
    entry_ledger = _ledger(
        tmp_path / "entry-ledger.json",
        decisions,
        action="BUY",
    )
    exit_ledger = _ledger(
        tmp_path / "exit-ledger.json",
        decisions,
        action="SELL",
    )
    entry_books = _books(
        tmp_path / "entry-books",
        entry_ledger,
        decisions,
        action="BUY",
        second_immediate_ask=second_ask,
    )
    exit_books = _books(
        tmp_path / "exit-books",
        exit_ledger,
        decisions,
        action="SELL",
    )
    return {
        "preregistration_path": _preregistration(
            tmp_path / "prereg.json",
            entry_ledger=entry_ledger,
            exit_ledger=exit_ledger,
        ),
        "ranking_path": _ranking(tmp_path / "ranking.json"),
        "entry_ledger_path": entry_ledger,
        "exit_ledger_path": exit_ledger,
        "entry_books": entry_books,
        "exit_books": exit_books,
    }


def test_ranked_full_quantity_symbol_dwell_passes(tmp_path: Path) -> None:
    receipt = validate_symbol_dwell(**_inputs(tmp_path))

    assert receipt["verdict"] == (
        "DWELL_VALIDATION_PASS_SELECTION_STILL_HOLD"
    )
    assert receipt["order_authority"] == "none"
    assert receipt["profitability_clock_started"] is False
    assert receipt["validation"]["frozen_max_quantities"] == {
        "SPYU": 25,
        "SPXU": 60,
    }
    assert receipt["validation"]["fill_coverage"]["control"] == "4/4"
    savings = receipt["validation"]["SPYU_BUY_savings_vs_control"]
    assert savings["total_usd"] == 5.0
    assert savings["nonnegative_quarters"] == 2
    assert savings["strict_majority_nonnegative"] is True
    assert receipt["validation"]["unchanged_SPXU_and_SELL_paths"] is True


def test_losing_calendar_quarter_preserves_hold(tmp_path: Path) -> None:
    receipt = validate_symbol_dwell(
        **_inputs(tmp_path, second_ask=10.20)
    )

    assert receipt["verdict"] == "HOLD"
    savings = receipt["validation"]["SPYU_BUY_savings_vs_control"]
    assert savings["strict_majority_nonnegative"] is False


def test_unavailable_book_is_disclosed_and_cannot_fake_quarter_coverage(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    unavailable = inputs["entry_books"] / "01-SPYU.json"
    document = json.loads(unavailable.read_text(encoding="utf-8"))
    document["status"] = "incomplete"
    _write(unavailable, document)
    manifest = inputs["entry_books"] / "manifest.json"
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    summary.update(
        {
            "complete": 1,
            "incomplete_or_error": 1,
            "remaining": 1,
        }
    )
    _write(manifest, summary)

    receipt = validate_symbol_dwell(**inputs)

    assert receipt["verdict"] == "HOLD"
    counts = receipt["validation"]["validation_book_counts"]
    assert counts["BUY"] == 1
    assert counts["unavailable_or_incomplete_before_cutoff"]["BUY"] == 1
    assert (
        receipt["validation"]["SPYU_BUY_savings_vs_control"][
            "strict_majority_nonnegative"
        ]
        is False
    )


def test_no_nominee_does_not_require_book_evidence(tmp_path: Path) -> None:
    entry_ledger = _write(tmp_path / "entry.json", {})
    exit_ledger = _write(tmp_path / "exit.json", {})
    receipt = validate_symbol_dwell(
        preregistration_path=_preregistration(
            tmp_path / "prereg.json",
            entry_ledger=entry_ledger,
            exit_ledger=exit_ledger,
        ),
        ranking_path=_ranking(tmp_path / "ranking.json", nominee=False),
        entry_ledger_path=entry_ledger,
        exit_ledger_path=exit_ledger,
        entry_books=tmp_path / "absent-entry-books",
        exit_books=tmp_path / "absent-exit-books",
    )

    assert receipt["verdict"] == "NOT_APPLICABLE_NO_NOMINEE"
    assert receipt["validation"] is None
