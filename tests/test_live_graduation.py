from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tradebot.live.capital import admit_live_capital, build_live_capital_plan
from tradebot.research.live_graduation import (
    LIVE_GRADUATION_PREFIX_SCHEMA,
    canonical_json_bytes,
    evidence_sha256,
    live_calibration_logical_prefix,
    publish_live_graduation_receipt,
    reduce_live_graduation,
    validate_live_graduation_receipt,
)
from tradebot.research.xsp_profitability import (
    SELECTED_CASH_EQUITY_SCHEMA,
    XspProfitabilityPolicy,
    xsp_capital_owner_stability_graduation_gate,
    xsp_execution_graduation_gate,
    xsp_live_graduation_inputs,
    xsp_runtime_parity_graduation_gate,
)
from tradebot.research.xsp_capital_stability import (
    XSP_CAPITAL_OWNER_GENERATION_SCHEMA,
    xsp_zero_capital_migration_evidence,
)


CUTOFF = datetime(2026, 7, 31, 20, 17, tzinfo=timezone.utc)
SELECTION_ID = "c" * 64
SELECTION_SHA = "f" * 64


def _gate(status: str = "PASS", *reasons: str) -> dict[str, object]:
    return {
        "status": status,
        "reasons": list(reasons),
        "evidence": {"proof": status.lower()},
    }


def _subject() -> dict[str, object]:
    return {
        "strategy_id": "strategy",
        "strategy_version": "strategy.execution.v1",
        "signal_instrument": "TEST",
        "execution_sleeve": "TEST/INVERSE",
        "capital_sleeve": "test-cash",
        "selection_id": SELECTION_ID,
        "run_id": SELECTION_ID,
        "account_fingerprint": "a" * 64,
    }


def _profitability(
    *,
    passed: bool = True,
    reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    subject = _subject()
    return {
        "schema": "xsp.live-profitability.v1",
        "as_of_utc": CUTOFF.isoformat(),
        "status": "PASSED" if passed else "ACTIVE",
        "policy": {
            key: subject[key]
            for key in (
                "run_id",
                "strategy_id",
                "strategy_version",
                "capital_sleeve",
            )
        },
        "milestones": {
            name: {
                "passed": passed,
                "reasons": list(reasons),
                "evidence_as_of_utc": CUTOFF.isoformat(),
            }
            for name in ("24h", "48h", "five_session_week")
        },
        "reasons": list(reasons),
    }


def _reducer_inputs() -> dict[str, object]:
    subject = _subject()
    return {
        "subject": subject,
        "selection": {
            "selection_id": SELECTION_ID,
            "run_id": SELECTION_ID,
            "capital_sleeve": subject["capital_sleeve"],
            "selection_file_sha256": SELECTION_SHA,
        },
        "selection_file_sha256": SELECTION_SHA,
        "ledger_prefix": {
            "schema": LIVE_GRADUATION_PREFIX_SCHEMA,
            "cutoff_utc": CUTOFF.isoformat(),
            "candidate_records": 8,
            "included_records": 8,
            "excluded_for_dependency": 0,
            "clock_regressions": 1,
            "first_record_id": "1" * 64,
            "last_record_id": "8" * 64,
            "sha256": "d" * 64,
            "gates": {
                "restart": _gate(),
                "cash_risk_safety": _gate(),
                "attribution": _gate(),
                "execution": _gate(),
            },
        },
        "profitability_receipt": _profitability(),
        "runtime_parity_proof": _gate(),
        "capital_owner_stability_proof": _gate(),
    }


def _receipt(**changes: object) -> dict[str, object]:
    inputs = _reducer_inputs()
    inputs.update(changes)
    return reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        **inputs,
    )


def test_logical_prefix_is_dependency_closed_and_suffix_stable() -> None:
    source_id = "1" * 64
    execution_id = "2" * 64
    source = {
        "kind": "checkpoint",
        "checkpoint_id": source_id,
        "recorded_at_utc": "2026-07-31T10:02:00+00:00",
        "evidence": {},
    }
    execution = {
        "kind": "checkpoint",
        "checkpoint_id": execution_id,
        "recorded_at_utc": "2026-07-31T10:00:00+00:00",
        "evidence": {
            "selection_id": SELECTION_ID,
            "source_checkpoint_id": source_id,
        },
    }
    early, early_rows = live_calibration_logical_prefix(
        (source, execution),
        cutoff_utc="2026-07-31T10:01:00+00:00",
    )
    with_future, future_rows = live_calibration_logical_prefix(
        (
            source,
            execution,
            {
                "kind": "checkpoint",
                "checkpoint_id": "3" * 64,
                "recorded_at_utc": "2026-07-31T11:00:00+00:00",
                "evidence": {},
            },
        ),
        cutoff_utc="2026-07-31T10:01:00+00:00",
    )
    complete, complete_rows = live_calibration_logical_prefix(
        (source, execution),
        cutoff_utc="2026-07-31T10:03:00+00:00",
    )

    assert early == with_future
    assert early_rows == future_rows == ()
    assert early["candidate_records"] == 1
    assert early["excluded_for_dependency"] == 1
    assert tuple(row["checkpoint_id"] for row in complete_rows) == (
        source_id,
        execution_id,
    )
    assert complete["clock_regressions"] == 1
    expected = b"".join(
        canonical_json_bytes(row) + b"\n" for row in (source, execution)
    )
    assert complete["sha256"] == hashlib.sha256(expected).hexdigest()


def test_logical_prefix_excludes_a_result_without_its_forecast() -> None:
    forecast_id = "4" * 64
    result = {
        "kind": "result",
        "result_id": "5" * 64,
        "forecast_id": forecast_id,
        "settled_at_utc": "2026-07-31T10:00:00+00:00",
    }
    forecast = {
        "kind": "forecast",
        "forecast_id": forecast_id,
        "recorded_at_utc": "2026-07-31T10:02:00+00:00",
    }

    prefix, rows = live_calibration_logical_prefix(
        (result, forecast),
        cutoff_utc="2026-07-31T10:01:00+00:00",
    )

    assert rows == ()
    assert prefix["excluded_for_dependency"] == 1


@pytest.mark.parametrize(
    ("status", "verdict"),
    [
        ("PASS", "PROMOTE"),
        ("HOLD", "HOLD"),
        ("FAIL", "REVISE"),
        ("INVALID", "QUARANTINE"),
        ("STOP", "STOP"),
    ],
)
def test_reducer_emits_each_verdict(status: str, verdict: str) -> None:
    inputs = _reducer_inputs()
    inputs["ledger_prefix"]["gates"]["execution"] = _gate(
        status,
        f"execution_{status.lower()}",
    )

    receipt = reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        **inputs,
    )

    assert receipt["verdict"] == verdict
    assert validate_live_graduation_receipt(receipt) == receipt
    assert receipt["boundaries"]["broker_queried"] is False
    assert receipt["boundaries"]["submitted_orders"] == 0


def test_stop_precedes_quarantine_revise_and_hold() -> None:
    inputs = _reducer_inputs()
    inputs["runtime_parity_proof"] = _gate("INVALID", "runtime_drift")
    inputs["ledger_prefix"]["gates"].update(
        {
            "restart": _gate("HOLD", "restart_pending"),
            "execution": _gate("FAIL", "execution_bad"),
            "cash_risk_safety": _gate("STOP", "drawdown_limit_breached"),
        }
    )

    receipt = reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        **inputs,
    )

    assert receipt["verdict"] == "STOP"
    assert receipt["remaining_requirements"] == []


def test_later_target_requires_predecessor_milestones() -> None:
    inputs = _reducer_inputs()
    profitability = _profitability()
    profitability["status"] = "ACTIVE"
    profitability["milestones"]["24h"] = {
        "passed": False,
        "reasons": ["net_not_positive"],
        "evidence_as_of_utc": CUTOFF.isoformat(),
    }
    inputs["profitability_receipt"] = profitability

    receipt = reduce_live_graduation(
        target_milestone="48h",
        cutoff_utc=CUTOFF,
        **inputs,
    )

    assert receipt["target"]["required_predecessors"] == ["24h"]
    assert receipt["gates"]["profitability"]["status"] == "FAIL"
    assert receipt["verdict"] == "REVISE"


def test_receipt_is_immutable_idempotent_and_structurally_validated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graduation.json"
    promote = _receipt()
    hold_inputs = _reducer_inputs()
    hold_inputs["ledger_prefix"]["gates"]["execution"] = _gate(
        "HOLD", "execution_not_observed"
    )
    hold = reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        **hold_inputs,
    )

    assert publish_live_graduation_receipt(path, promote) is True
    assert publish_live_graduation_receipt(path, promote) is False
    with pytest.raises(ValueError, match="identity conflicts"):
        publish_live_graduation_receipt(path, hold)
    tampered = deepcopy(hold)
    tampered["reasons"] = []
    body = {key: value for key, value in tampered.items() if key != "receipt_id"}
    tampered["receipt_id"] = evidence_sha256(body)
    with pytest.raises(ValueError, match="invalid live graduation"):
        validate_live_graduation_receipt(tampered)


def _execution_selection() -> dict[str, object]:
    return {
        "selection_id": SELECTION_ID,
        "nominee": {
            "commission_limits_usd": {"UPRO": 0.45, "SPXU": 0.45},
            "contract_ids": {"UPRO": 61_228_752, "SPXU": 828_937_771},
        },
        "execution": {"policy_contract": {"auto_timeout_seconds": 324.0}},
    }


def _execution_rows(
    *,
    transition: str = "a" * 64,
    exec_id: str = "exec-1",
) -> list[dict[str, object]]:
    order_ref = f"XSPV3-{transition[:24]}"
    capital_plan = build_live_capital_plan(
        account_id="DU123",
        account_type="CASH",
        currency="USD",
        observed_settled_cash_usd=1_318.05,
        managed_capital_usd=900.45,
        sleeves=[
            {
                "sleeve_id": "xsp-cash",
                "strategy_id": "xsp.signal.v3",
                "run_id": SELECTION_ID,
                "selection_path": "selection.json",
                "selection_file_sha256": SELECTION_SHA,
                "capital_kind": "CASH_DEBIT",
                "weight_bps": 10_000,
            }
        ],
        reserve_reasons=["outside_selected_authority"],
        created_at_utc=CUTOFF,
    )
    plan = {
        "transition_id": transition,
        "capital_admission": admit_live_capital(
            capital_plan,
            intent="ENTER",
            account_id="DU123",
            account_type="CASH",
            currency="USD",
            sleeve_id="xsp-cash",
            run_id=SELECTION_ID,
            selection_file_sha256=SELECTION_SHA,
            capital_kind="CASH_DEBIT",
            projected_capital_usd=600.06,
            cash_debit_usd=600.06,
            available_cash_usd=1_318.05,
        ),
        "leg": {
            "action": "BUY",
            "symbol": "UPRO",
            "quantity": 6,
            "bid": 100.0,
            "ask": 100.01,
            "outside_rth": False,
        },
    }
    preview = {
        "commission": None,
        "min_commission": 0.28,
        "max_commission": 0.42,
        "commission_currency": "USD",
    }
    common = {
        "order_ref": order_ref,
        "plan": plan,
        "what_if_preview": preview,
    }
    return [
        {"evidence": {**common, "phase": "PREPARED", "submitted_orders": 0}},
        {
            "evidence": {
                **common,
                "phase": "SUBMITTED",
                "submitted_orders": 1,
                "ladder_transition": {
                    "schema": "xsp.execution-ladder-transition.v1",
                    "event": "ladder_mode_transition",
                    "action": "BUY",
                    "active_mode": "MID",
                    "elapsed_seconds": 6.0,
                    "quote_eligible": False,
                },
            }
        },
        {
            "evidence": {
                **common,
                "phase": "TERMINAL",
                "submitted_orders": 1,
                "broker_order": {
                    "done": True,
                    "order_ref": order_ref,
                    "symbol": "UPRO",
                    "con_id": 61_228_752,
                    "action": "BUY",
                    "quantity": 6,
                    "filled": 6,
                    "remaining": 0,
                    "limit_price": 100.01,
                    "fills": [
                        {
                            "exec_id": exec_id,
                            "time_utc": "2026-07-31T14:01:00+00:00",
                            "symbol": "UPRO",
                            "side": "BOT",
                            "shares": 6,
                            "price": 100.0,
                            "commission": 0.35,
                            "commission_currency": "USD",
                        }
                    ],
                },
            }
        },
    ]


def test_xsp_execution_gate_accepts_exact_order_and_paused_repricing() -> None:
    gate = xsp_execution_graduation_gate(
        _execution_selection(),
        _execution_rows(),
    )

    assert gate["status"] == "PASS"
    assert gate["evidence"]["orders"] == 1
    assert gate["evidence"]["fills"] == 1
    assert gate["evidence"]["quote_ineligible_transitions"] == 1
    assert gate["evidence"]["limit_price_improvement_usd"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("worse_fill", "FAIL"),
        ("commission", "FAIL"),
        ("fractional_quantity", "INVALID"),
    ],
)
def test_xsp_execution_gate_fails_closed(
    mutation: str,
    expected: str,
) -> None:
    rows = _execution_rows()
    if mutation == "worse_fill":
        rows[-1]["evidence"]["broker_order"]["fills"][0]["price"] = 100.02
    elif mutation == "commission":
        rows[-1]["evidence"]["broker_order"]["fills"][0]["commission"] = 0.47
    else:
        for row in rows:
            row["evidence"]["plan"]["leg"]["quantity"] = 6.5

    gate = xsp_execution_graduation_gate(_execution_selection(), rows)

    assert gate["status"] == expected


def test_xsp_execution_gate_stops_duplicate_broker_execution() -> None:
    rows = _execution_rows(transition="a" * 64, exec_id="duplicate")
    rows.extend(_execution_rows(transition="b" * 64, exec_id="duplicate"))

    gate = xsp_execution_graduation_gate(_execution_selection(), rows)

    assert gate["status"] == "STOP"
    assert "duplicate_broker_execution" in gate["reasons"]


def test_xsp_proof_adapters_rehash_current_owners(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime = xsp_runtime_parity_graduation_gate(
        repo_root / "backtests/xsp/opening_edge_v3_current_runtime_parity_audit.json",
        repo_root=repo_root,
    )
    owner = repo_root / "tradebot/research/xsp_live_transport.py"
    owner_sha = hashlib.sha256(owner.read_bytes()).hexdigest()
    manifest = tmp_path / "capital.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "xsp.opening-edge-v3-capital-owner-stability-manifest.v1",
                "authority": "frozen_post_selection_capital_owner_manifest",
                "observed_at_utc": CUTOFF.isoformat(),
                "source_revision": "revision",
                "selection": {
                    "selection_id": SELECTION_ID,
                    "selection_file_sha256": SELECTION_SHA,
                },
                "capital_semantic_surface": {
                    "tradebot/research/xsp_live_transport.py": owner_sha,
                },
                "checks": {"capital_semantics_unchanged": True},
                "verdict": "PASS_CAPITAL_OWNER_STABLE",
                "boundaries": {
                    "broker_queried": False,
                    "service_or_timer_mutated": False,
                    "selection_mutated": False,
                    "submitted_orders": 0,
                    "profitability_clock_mutated": False,
                },
            }
        ),
        encoding="utf-8",
    )
    capital = xsp_capital_owner_stability_graduation_gate(
        manifest,
        repo_root=repo_root,
        selection_id=SELECTION_ID,
        selection_file_sha256=SELECTION_SHA,
    )

    assert runtime["status"] == "PASS"
    assert capital["status"] == "PASS"


def _state_evidence() -> dict[str, object]:
    risk = {
        "valid": True,
        "fill_count": 0,
        "closed_trades": 0,
        "run_gross_usd": 0.0,
        "run_cost_usd": 0.0,
        "run_net_usd": 0.0,
        "run_realized_net_usd": 0.0,
        "open_mark_net_usd": 0.0,
        "session_gross_usd": 0.0,
        "session_cost_usd": 0.0,
        "session_net_usd": 0.0,
        "gross_wins_usd": 0.0,
        "top_five_gross_wins_usd": 0.0,
        "settled_cash_usd": 1_318.05,
        "drawdown_usd": 0.0,
        "holdings_from_fills": {"UPRO": 0.0, "SPXU": 0.0},
        "safety_breaches": [],
        "attribution_complete": True,
    }
    equity = {
        "schema": SELECTED_CASH_EQUITY_SCHEMA,
        "run_id": SELECTION_ID,
        "config_fingerprint": SELECTION_ID,
        "capital_sleeve": "xsp-cash",
        "cumulative_gross_usd": 0.0,
        "cumulative_cost_usd": 0.0,
        "cumulative_net_usd": 0.0,
        "cumulative_realized_net_usd": 0.0,
        "open_mark_usd": 0.0,
        "session_gross_usd": 0.0,
        "session_cost_usd": 0.0,
        "session_net_usd": 0.0,
        "closed_trades": 0,
        "gross_wins_usd": 0.0,
        "top_five_gross_wins_usd": 0.0,
        "reconciled": True,
        "attribution_complete": True,
        "safety_breaches": [],
    }
    return {
        "selection_id": SELECTION_ID,
        "phase": "STATE",
        "submitted_orders": 0,
        "plan": {"holdings": {"UPRO": 0, "SPXU": 0}},
        "broker_state": {
            "open_orders": [],
            "positions": {"UPRO": 0.0, "SPXU": 0.0},
        },
        "risk_state": risk,
        "selected_cash_equity": equity,
    }


def test_xsp_projection_reuses_prefix_cash_and_proof_contracts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    selected_at = CUTOFF - timedelta(minutes=10)
    selection = {
        "selection_id": SELECTION_ID,
        "selected_at_utc": selected_at.isoformat(),
        "run_started_at_utc": selected_at.isoformat(),
        "strategy_version": "xsp.signal.v3",
        "broker_at_selection": {
            "account_id": "DU123",
            "positions": {"UPRO": 0, "SPXU": 0},
        },
        **_execution_selection(),
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    policy = XspProfitabilityPolicy(
        run_id=SELECTION_ID,
        strategy_id="xsp.signal.v3",
        strategy_version="xsp.execution.v3",
        config_fingerprint=SELECTION_ID,
        capital_sleeve="xsp-cash",
        max_drawdown_points=135.0,
        max_session_loss_points=67.5,
        minimum_week_closed_trades=2,
        maximum_top_five_win_share=0.5,
        unit="USD",
        equity_schema=SELECTED_CASH_EQUITY_SCHEMA,
    )
    records = []
    for index, minute in enumerate((6, 1)):
        state_at = CUTOFF - timedelta(minutes=minute)
        source_id = str(index + 1) * 64
        state_id = str(index + 3) * 64
        records.extend(
            [
                {
                    "kind": "checkpoint",
                    "checkpoint_id": source_id,
                    "recorded_at_utc": (state_at + timedelta(seconds=2)).isoformat(),
                    "evidence": {},
                },
                {
                    "kind": "checkpoint",
                    "checkpoint_id": state_id,
                    "recorded_at_utc": state_at.isoformat(),
                    "strategy_id": policy.strategy_id,
                    "strategy_version": policy.strategy_version,
                    "evidence": {
                        **_state_evidence(),
                        "source_checkpoint_id": source_id,
                    },
                },
            ]
        )
    profitability = {
        **_profitability(passed=False, reasons=("elapsed_time_incomplete",)),
        "policy": {
            "run_id": policy.run_id,
            "strategy_id": policy.strategy_id,
            "strategy_version": policy.strategy_version,
            "capital_sleeve": policy.capital_sleeve,
        },
    }
    manifest = tmp_path / "capital.json"
    owner = repo_root / "tradebot/research/xsp_live_transport.py"
    manifest.write_text(
        json.dumps(
            {
                "schema": "xsp.opening-edge-v3-capital-owner-stability-manifest.v1",
                "authority": "frozen_post_selection_capital_owner_manifest",
                "observed_at_utc": CUTOFF.isoformat(),
                "source_revision": "revision",
                "selection": {
                    "selection_id": SELECTION_ID,
                    "selection_file_sha256": selection_sha,
                },
                "capital_semantic_surface": {
                    "tradebot/research/xsp_live_transport.py": hashlib.sha256(
                        owner.read_bytes()
                    ).hexdigest(),
                },
                "checks": {"capital_semantics_unchanged": True},
                "verdict": "PASS_CAPITAL_OWNER_STABLE",
                "boundaries": {
                    "broker_queried": False,
                    "service_or_timer_mutated": False,
                    "selection_mutated": False,
                    "submitted_orders": 0,
                    "profitability_clock_mutated": False,
                },
            }
        ),
        encoding="utf-8",
    )

    inputs = xsp_live_graduation_inputs(
        selection=selection,
        selection_path=selection_path,
        records=records,
        cutoff_utc=CUTOFF,
        policy=policy,
        profitability_receipt=profitability,
        runtime_parity_path=(
            repo_root
            / "backtests/xsp/opening_edge_v3_current_runtime_parity_audit.json"
        ),
        capital_owner_stability_path=manifest,
        repo_root=repo_root,
    )
    receipt = reduce_live_graduation(
        target_milestone="24h",
        cutoff_utc=CUTOFF,
        **inputs,
    )

    assert inputs["ledger_prefix"]["clock_regressions"] == 2
    assert receipt["gates"]["restart"]["status"] == "PASS"
    assert receipt["gates"]["cash_risk_safety"]["status"] == "PASS"
    assert receipt["gates"]["attribution"]["status"] == "PASS"
    assert receipt["gates"]["execution"]["status"] == "HOLD"
    assert receipt["gates"]["profitability"]["status"] == "HOLD"
    assert receipt["verdict"] == "HOLD"


def test_xsp_capital_owner_generation_preserves_only_zero_capital_prefix(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "owner.py"
    owner.write_text("OWNER = 1\n", encoding="utf-8")
    predecessor = tmp_path / "predecessor.json"
    selection = {
        "selection_id": SELECTION_ID,
        "selection_file_sha256": SELECTION_SHA,
    }
    predecessor.write_text(
        json.dumps(
            {
                "schema": "xsp.opening-edge-v3-capital-owner-stability-manifest.v1",
                "selection": selection,
            }
        ),
        encoding="utf-8",
    )
    record = {
        "kind": "checkpoint",
        "checkpoint_id": "9" * 64,
        "recorded_at_utc": (CUTOFF - timedelta(minutes=1)).isoformat(),
        "strategy_id": "xsp.signal.v3",
        "strategy_version": "xsp.execution.v3",
        "evidence": _state_evidence(),
    }
    migration = xsp_zero_capital_migration_evidence(
        [record],
        cutoff_utc=CUTOFF.isoformat(),
        strategy_id="xsp.signal.v3",
        strategy_version="xsp.execution.v3",
        run_id=SELECTION_ID,
    )
    manifest = tmp_path / "capital.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": XSP_CAPITAL_OWNER_GENERATION_SCHEMA,
                "authority": "frozen_post_selection_capital_owner_manifest",
                "observed_at_utc": CUTOFF.isoformat(),
                "source_revision": "revision",
                "selection": selection,
                "capital_semantic_surface": {
                    "owner.py": hashlib.sha256(owner.read_bytes()).hexdigest(),
                },
                "migration": {
                    "predecessor_path": predecessor.name,
                    "predecessor_sha256": hashlib.sha256(
                        predecessor.read_bytes()
                    ).hexdigest(),
                    "effective_at_utc": CUTOFF.isoformat(),
                    "zero_capital_prefix": migration,
                },
                "checks": {"zero_capital_migration_proven": True},
                "verdict": "PASS_CAPITAL_OWNER_STABLE",
                "boundaries": {
                    "broker_queried": False,
                    "service_or_timer_mutated": False,
                    "selection_mutated": False,
                    "submitted_orders": 0,
                    "profitability_clock_mutated": False,
                },
            }
        ),
        encoding="utf-8",
    )

    passed = xsp_capital_owner_stability_graduation_gate(
        manifest,
        repo_root=tmp_path,
        selection_id=SELECTION_ID,
        selection_file_sha256=SELECTION_SHA,
        records=[record],
        strategy_id="xsp.signal.v3",
        strategy_version="xsp.execution.v3",
    )
    changed = deepcopy(record)
    changed["evidence"]["risk_state"]["fill_count"] = 1
    rejected = xsp_capital_owner_stability_graduation_gate(
        manifest,
        repo_root=tmp_path,
        selection_id=SELECTION_ID,
        selection_file_sha256=SELECTION_SHA,
        records=[changed],
        strategy_id="xsp.signal.v3",
        strategy_version="xsp.execution.v3",
    )

    assert passed["status"] == "PASS"
    assert rejected["status"] == "INVALID"
    assert "capital_owner_migration_prefix_invalid" in rejected["reasons"]


def test_cli_graduation_branch_never_loads_broker_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tradebot.research import xsp_shadow_cli

    selected = tmp_path / "selected.json"
    selected.write_text("{}", encoding="utf-8")
    output = tmp_path / "graduation.json"
    captured: dict[str, object] = {}
    source_id = "9" * 64
    raw_records = (
        {
            "kind": "checkpoint",
            "checkpoint_id": source_id,
            "recorded_at_utc": (CUTOFF + timedelta(seconds=1)).isoformat(),
            "evidence": {},
        },
        {
            "kind": "checkpoint",
            "checkpoint_id": "8" * 64,
            "recorded_at_utc": CUTOFF.isoformat(),
            "evidence": {
                "selection_id": SELECTION_ID,
                "source_checkpoint_id": source_id,
            },
        },
    )

    class _Ledger:
        def __init__(self, path: str) -> None:
            captured["ledger"] = path

        def records(self) -> tuple[dict[str, object], ...]:
            return raw_records

        def xsp_profitability_receipt(self, **kwargs: object) -> dict[str, object]:
            captured["profitability"] = kwargs
            return {"profitability": True}

    monkeypatch.setattr(
        xsp_shadow_cli,
        "load_xsp_v3_transport_selection",
        lambda path: {"selection_id": SELECTION_ID, "path": str(path)},
    )
    monkeypatch.setattr(
        xsp_shadow_cli,
        "xsp_v3_transport_profitability_policy",
        lambda _selection: "policy",
    )
    monkeypatch.setattr(xsp_shadow_cli, "LiveCalibrationLedger", _Ledger)
    def _inputs(**kwargs: object) -> dict[str, object]:
        captured["inputs"] = kwargs
        return {}

    monkeypatch.setattr(xsp_shadow_cli, "xsp_live_graduation_inputs", _inputs)
    receipt = {"verdict": "HOLD", "receipt_id": "r"}
    monkeypatch.setattr(
        xsp_shadow_cli,
        "reduce_live_graduation",
        lambda **kwargs: captured.setdefault("reducer", kwargs) and receipt,
    )
    monkeypatch.setattr(
        xsp_shadow_cli,
        "publish_live_graduation_receipt",
        lambda path, value: captured.update(published=(path, value)),
    )
    monkeypatch.setattr(
        xsp_shadow_cli,
        "load_config",
        lambda: (_ for _ in ()).throw(AssertionError("broker config loaded")),
    )

    status = asyncio.run(
        xsp_shadow_cli._main_async(
            (
                "--mode",
                "opening-edge-v3",
                "--ledger",
                str(tmp_path / "ledger.jsonl"),
                "--selected-transport",
                str(selected),
                "--graduation-target",
                "24h",
                "--graduation-cutoff",
                CUTOFF.isoformat(),
                "--graduation-output",
                str(output),
            )
        )
    )

    assert status == 0
    assert captured["published"] == (output, receipt)
    assert captured["profitability"]["_records"] == ()
    assert captured["inputs"]["records"] == raw_records
    assert captured["inputs"]["capital_owner_stability_path"] == Path(
        "db/calibration/portfolio_capital_owner_stability.json"
    )
    assert json.loads(capsys.readouterr().out) == receipt
