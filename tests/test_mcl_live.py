from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.chart_data.series import OhlcvBar
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.mcl_live import (
    advance_mcl_live_transport,
    mcl_transport_risk_state,
    project_mcl_transport_plan,
)
from tradebot.research.mcl_live_reopen import (
    MCL_LIVE_SOURCE_AUTHORITY_FRESH,
    MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
    bind_mcl_maintenance_reopen_selection,
    refresh_mcl_live_source,
)
from tradebot.research.mcl_stage131 import (
    MCL_STAGE131_BINDING_KEY,
    bind_mcl_stage131_selection,
    build_mcl_stage131_coverage,
    project_mcl_stage131_entry_guard,
    publish_mcl_stage131_coverage,
)
from tradebot.research.mcl_live_transport import (
    MCL_LIVE_EXECUTION_VERSION,
    MCL_LIVE_SOURCE_SCHEMA,
    MCL_LIVE_SOURCE_VERSION,
    build_mcl_live_selection,
    load_mcl_live_selection_from_mapping,
    mcl_source_snapshot,
    persist_mcl_source_checkpoint,
)
from tradebot.research.mcl_profitability import (
    mcl_live_evaluation_slots,
    mcl_live_profitability_receipt,
    mcl_market_open,
    mcl_runtime_parity_graduation_gate,
    normalize_mcl_risk,
)
from tradebot.research.live_futures_profitability import (
    FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA,
)
from tradebot.research.live_graduation import evidence_sha256
from tradebot.research.mcl_profitability_epoch import (
    build_mcl_profitability_coverage_epoch,
    load_mcl_profitability_coverage_epoch,
    mcl_profitability_receipt_with_coverage_epoch,
)
from tradebot.research.mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION


ROOT = Path(__file__).resolve().parents[1]
AT = datetime(2026, 8, 4, 8, 31, 21, tzinfo=timezone.utc)


def _preview() -> dict[str, object]:
    old = json.loads(
        (ROOT / "backtests/mcl/mcl_v18_live_commissioning_preview_post_funding.json").read_text()
    )
    generation = json.loads(
        (ROOT / "backtests/mcl/mcl_turn_authenticity_microstructure_generation.json").read_text()
    )
    broker = old["broker"]
    return {
        "schema": "mcl.v18-live-commissioning-preview.v2",
        "observed_at_utc": AT.isoformat(),
        "authority": "fresh_nontransmitting_what_if_only",
        "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
        "broker": {
            "observed_at_utc": broker["observed_at_utc"],
            "account_id": broker["account_id"],
            "account_type": "CASH",
            "base_currency": "AUD",
            "settled_cash_usd": broker["settled_cash_usd"],
            "equity_with_loan_base": broker["equity_with_loan_aud"],
            "available_funds_base": broker["available_funds_aud"],
            "excess_liquidity_base": broker["excess_liquidity_aud"],
            "initial_margin_base": broker["initial_margin_aud"],
            "maintenance_margin_base": broker["maintenance_margin_aud"],
            "gross_position_value_base": broker["gross_position_value_aud"],
            "usd_to_base_rate": broker["usd_to_aud"],
            "positions": broker["positions"],
            "open_orders": [],
        },
        "contracts": generation["contracts"],
        "quote": old["quote"],
        "source": {
            "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
            "submitted_orders": 0,
        },
        "what_if": old["what_if"],
        "submitted_orders": 0,
    }


def _selection():
    return bind_mcl_stage131_selection(
        bind_mcl_maintenance_reopen_selection(
            build_mcl_live_selection(
                repository_root=ROOT,
                preview=_preview(),
                selected_at=AT + timedelta(seconds=1),
            ),
            repository_root=ROOT,
        ),
        repository_root=ROOT,
    )


def test_mcl_selection_binds_flat_limit_only_stage91_canary() -> None:
    selected = _selection()

    assert load_mcl_live_selection_from_mapping(selected) == selected
    assert selected["baseline"]["position"] == 0
    assert selected["baseline"]["inherited_target_authority"] == "none"
    assert selected["strategy_version"] == MCL_TWO_SPEED_SHOCK_VERSION
    assert selected["execution"]["order_type"] == "LMT"
    assert selected["execution"]["market_orders_allowed"] is False
    assert selected["risk"]["raw_loss_cap_usd"] == 300.0
    assert selected["risk"]["package_stressed_loss_usd"] == 305.52
    assert selected["allocation_successor"]["package_id"] == (
        "mcl-one-contract-stage91"
    )
    assert selected["allocation_successor"]["initial_margin_base_cents"] == 268_670
    assert selected["allocation_successor"]["maintenance_margin_base_cents"] == 214_936

    mutated = deepcopy(selected)
    mutated["execution"]["market_orders_allowed"] = True
    with pytest.raises(ValueError, match="selected-run contract is invalid"):
        load_mcl_live_selection_from_mapping(mutated)


def test_mcl_source_uses_completed_et_minutes_and_never_adopts_history() -> None:
    now = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)
    first = (now - timedelta(minutes=620)).astimezone(
        timezone(timedelta(hours=-4))
    ).replace(tzinfo=None)
    rows = []
    for index in range(620):
        ts = first + timedelta(minutes=index)
        close = 80 + index * 0.001
        rows.append(OhlcvBar(ts, close, close + 0.01, close - 0.01, close, 10))

    class Client:
        async def historical_bars_ohlcv(self, contract, **_kwargs):
            bump = 0.0 if contract.symbol == "CL" else 0.01
            return [
                OhlcvBar(
                    row.ts,
                    row.open + bump,
                    row.high + bump,
                    row.low + bump,
                    row.close + bump,
                    row.volume,
                )
                for row in rows
            ]

    selected = _selection()
    from tradebot.research.mcl_live_transport import mcl_live_contracts

    cl, mcl = mcl_live_contracts(selected)
    source = asyncio.run(
        mcl_source_snapshot(
            Client(),
            cl_contract=cl,
            mcl_contract=mcl,
            observed_at=now,
            selected_at=now,
            strategy_version=selected["strategy_version"],
        )
    )

    assert source["rows"]["common"] == 620
    assert source["latest_common_close_utc"] == now.isoformat()
    assert source["target"] is None
    assert source["synthetic_midcycle_entry_authority"] == "none"


@pytest.mark.parametrize(
    ("observed_at", "prior_at"),
    (
        (
            datetime(2026, 8, 4, 22, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 8, 4, 20, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 8, 9, 22, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 8, 7, 20, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 3, 8, 22, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 3, 6, 21, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 11, 1, 23, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 10, 30, 20, 59, tzinfo=timezone.utc),
        ),
    ),
)
def test_mcl_reopen_reuses_only_the_exact_prior_close_for_reconciliation(
    tmp_path: Path, observed_at: datetime, prior_at: datetime
) -> None:
    selected = _selection()
    ledger = LiveCalibrationLedger(tmp_path / "mcl.jsonl")
    saved = persist_mcl_source_checkpoint(
        ledger,
        selection=selected,
        source={
            "schema": "mcl.two-speed-auction-finalized-source-snapshot.v1",
            "strategy_version": selected["strategy_version"],
            "contract_month": str(selected["contracts"]["MCL"]["expiry"])[:6],
            "latest_common_close_utc": prior_at.isoformat(),
            "target": None,
            "submitted_orders": 0,
        },
        observed_at=prior_at + timedelta(seconds=10),
    )

    class Client:
        async def historical_bars_ohlcv(self, *_args, **_kwargs):
            raise AssertionError("exact reopen reconciliation requested history")

    source, authority = asyncio.run(
        refresh_mcl_live_source(
            ledger,
            client=Client(),
            selection=selected,
            observed_at=observed_at,
        )
    )

    assert source["checkpoint_id"] == saved["checkpoint_id"]
    assert authority == MCL_LIVE_SOURCE_AUTHORITY_REOPEN
    assert len(tuple(ledger.records())) == 1


@pytest.mark.parametrize(
    ("observed_at", "prior_at"),
    (
        (
            datetime(2026, 8, 4, 22, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 8, 4, 20, 58, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 8, 4, 22, 1, 10, tzinfo=timezone.utc),
            datetime(2026, 8, 4, 20, 59, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 8, 7, 22, 0, 10, tzinfo=timezone.utc),
            datetime(2026, 8, 7, 20, 59, tzinfo=timezone.utc),
        ),
    ),
)
def test_mcl_reopen_never_generalizes_to_an_adjacent_or_closed_boundary(
    tmp_path: Path, observed_at: datetime, prior_at: datetime
) -> None:
    selected = _selection()
    ledger = LiveCalibrationLedger(tmp_path / "mcl.jsonl")
    persist_mcl_source_checkpoint(
        ledger,
        selection=selected,
        source={
            "schema": "mcl.two-speed-auction-finalized-source-snapshot.v1",
            "strategy_version": selected["strategy_version"],
            "contract_month": str(selected["contracts"]["MCL"]["expiry"])[:6],
            "latest_common_close_utc": prior_at.isoformat(),
            "target": None,
            "submitted_orders": 0,
        },
        observed_at=prior_at + timedelta(seconds=10),
    )

    class Client:
        async def historical_bars_ohlcv(self, *_args, **_kwargs):
            raise RuntimeError("strict history requested")

    with pytest.raises(RuntimeError, match="strict history requested"):
        asyncio.run(
            refresh_mcl_live_source(
                ledger,
                client=Client(),
                selection=selected,
                observed_at=observed_at,
            )
        )


def test_mcl_reopen_advance_emits_the_required_reconciliation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradebot.research import mcl_live as runtime

    selected = _selection()
    ledger = LiveCalibrationLedger(tmp_path / "mcl.jsonl")
    prior = datetime(2026, 8, 4, 20, 59, tzinfo=timezone.utc)
    observed = datetime(2026, 8, 4, 22, 0, 10, tzinfo=timezone.utc)
    persist_mcl_source_checkpoint(
        ledger,
        selection=selected,
        source={
            "schema": "mcl.two-speed-auction-finalized-source-snapshot.v1",
            "strategy_version": selected["strategy_version"],
            "contract_month": str(selected["contracts"]["MCL"]["expiry"])[:6],
            "latest_common_close_utc": prior.isoformat(),
            "target": None,
            "submitted_orders": 0,
        },
        observed_at=prior + timedelta(seconds=10),
    )

    async def broker(*_args, **_kwargs):
        return {
            **selected["broker_at_selection"],
            "positions": [],
            "open_orders": [],
        }

    async def quote(*_args, **_kwargs):
        return object(), {
            "bid": 75.0,
            "ask": 75.01,
            "last": 75.0,
            "close": 75.0,
            "age_seconds": 0.1,
            "market_data_type": 1,
            "health": {"eligible": True},
        }

    class Client:
        async def historical_bars_ohlcv(self, *_args, **_kwargs):
            raise AssertionError("reopen reconciliation requested history")

    monkeypatch.setattr(runtime, "broker_account_snapshot", broker)
    monkeypatch.setattr(runtime, "_live_quote", quote)
    result = asyncio.run(
        advance_mcl_live_transport(
            ledger,
            client=Client(),
            selection=selected,
            capital_plan={},
            selection_file_sha256="a" * 64,
            observed_at=observed,
        )
    )
    records = tuple(ledger.records())
    state = records[-1]

    assert result["status"] == "HOLD"
    assert result["plan"]["reason"] == "maintenance_reopen_reconciliation_only"
    assert result["plan"]["source_authority"] == MCL_LIVE_SOURCE_AUTHORITY_REOPEN
    assert result["plan"]["leg"] is None
    assert result["submitted_orders"] == 0
    assert state["evaluation_as_of_utc"] == observed.isoformat()
    assert state["evidence"]["phase"] == "STATE"
    assert state["evidence"]["risk_state"]["position_from_fills"] == 0


def _source_checkpoint(
    selected, *, event_at: datetime, event_id: str, owner: str = "v18"
):
    target = {
        "event_id": event_id,
        "observed_at_utc": event_at.isoformat(),
        "signal_at_utc": event_at.isoformat(),
        "direction": 1,
        "route": "failed_auction",
        "owner": owner,
        "decision": {},
    }
    return {
        "checkpoint_id": "2" * 64,
        "strategy_version": MCL_LIVE_SOURCE_VERSION,
        "recorded_at_utc": (event_at + timedelta(seconds=5)).isoformat(),
        "status": "EVALUATED",
        "evidence": {
            "schema": MCL_LIVE_SOURCE_SCHEMA,
            "selection_id": selected["selection_id"],
            "target": target,
            "source": {
                "latest_common_close_utc": event_at.isoformat(),
            },
        },
    }


def _stage131_fixture(selected):
    raw_at = AT + timedelta(minutes=5)
    target_at = raw_at + timedelta(minutes=5)
    raw = {
        "event_id": "3" * 64,
        "observed_at_utc": raw_at.isoformat(),
        "signal_at_utc": raw_at.isoformat(),
        "direction": -1,
        "owner": "v18",
        "decision": {"phase": "RAW_TURN", "raw_direction": -1},
    }
    target = {
        "event_id": "4" * 64,
        "observed_at_utc": target_at.isoformat(),
        "signal_at_utc": raw_at.isoformat(),
        "direction": 1,
        "route": "failed_auction",
        "owner": "v18",
        "decision": {
            "phase": "MATURATION",
            "raw_direction": -1,
            "admitted_direction": 1,
        },
    }
    contract_key = str(selected["contracts"]["MCL"]["expiry"])[:6]
    generation_id = "5" * 64
    identity = {
        "generation_id": generation_id,
        "selection_id": selected["selection_id"],
        "contract_key": contract_key,
        "started_at_utc": (raw_at - timedelta(minutes=20)).isoformat(),
        "first_record_id": "6" * 64,
    }
    episode = {
        "episode_id": hashlib.sha256(
            json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "identity": identity,
        "terminal_at_utc": raw_at.isoformat(),
        "terminal_authority_direction": -1,
        "authority_waves": [{"direction": -1}],
        "terminal": {"reasons": ["stage112_v18_raw_turn"]},
    }
    source = {
        "contract_month": contract_key,
        "latest_common_close_utc": target_at.isoformat(),
        "last_raw_turn": raw,
        "target": target,
    }
    checkpoint = {
        "checkpoint_id": "7" * 64,
        "strategy_version": MCL_LIVE_SOURCE_VERSION,
        "recorded_at_utc": (target_at + timedelta(seconds=5)).isoformat(),
        "status": "EVALUATED",
        "evidence": {
            "schema": MCL_LIVE_SOURCE_SCHEMA,
            "selection_id": selected["selection_id"],
            "source": source,
            "last_raw_turn": raw,
            "target": target,
        },
    }
    stamp = raw_at
    bars = {
        symbol: {stamp: OhlcvBar(stamp, 77.0, 77.1, 76.9, 77.0, 100)}
        for symbol in ("CL", "MCL")
    }
    coverage = build_mcl_stage131_coverage(
        generation={
            "generation_id": generation_id,
            "selection_id": selected["selection_id"],
            "strategy_version": MCL_TWO_SPEED_SHOCK_VERSION,
        },
        selection=selected,
        rows=[{"_time": stamp}],
        bars=bars,
        complete_episodes=[episode],
        open_episode=None,
        recorded_at=stamp + timedelta(seconds=45),
    )
    context = {
        "generation_id": generation_id,
        "selection_id": selected["selection_id"],
        "coverage": coverage,
        "episodes": [episode],
    }
    return checkpoint, context, target_at


def test_mcl_stage131_is_selection_bound_and_coverage_is_atomic(tmp_path: Path) -> None:
    selected = _selection()
    checkpoint, context, _target_at = _stage131_fixture(selected)
    path = tmp_path / "coverage.json"

    published = publish_mcl_stage131_coverage(path, context["coverage"])

    assert MCL_STAGE131_BINDING_KEY in selected["evidence"]
    assert json.loads(path.read_text()) == published
    assert published["complete_episode_ids"] == [
        context["episodes"][0]["episode_id"]
    ]
    assert checkpoint["evidence"]["target"]["direction"] == 1


def test_mcl_stage131_vetoes_only_flat_entry_and_never_weakens_exit() -> None:
    selected = _selection()
    source, context, target_at = _stage131_fixture(selected)
    guard = project_mcl_stage131_entry_guard(source, context=context)

    held = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=-1,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=target_at + timedelta(minutes=1, seconds=5),
        entry_guard=guard,
    )
    flat = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=target_at + timedelta(minutes=1, seconds=5),
        entry_guard=guard,
    )

    assert guard["action"] == "VETO_OPPOSITE_EXACT_EPOCH"
    assert flat["status"] == "HOLD"
    assert flat["reason"] == "stage131_veto_opposite_exact_epoch"
    assert flat["target_direction"] == 1
    assert flat["leg"] is None
    assert held["status"] == "ACTIONABLE"
    assert held["reason"] == "raw_turn_or_source_flatten"
    assert held["leg"]["action"] == "BUY"
    assert held["leg"]["initial_mode"] == "CROSS"


def test_mcl_stage131_missing_defers_but_proven_absence_preserves_clock() -> None:
    selected = _selection()
    source, context, target_at = _stage131_fixture(selected)
    missing = project_mcl_stage131_entry_guard(
        source, context={**context, "coverage": None}
    )
    absent_coverage = {
        **context["coverage"],
        "complete_episode_ids": [],
        "complete_episode_set_sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    body = dict(absent_coverage)
    body.pop("state_id")
    absent_coverage["state_id"] = hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    absent = project_mcl_stage131_entry_guard(
        source, context={**context, "coverage": absent_coverage, "episodes": []}
    )

    deferred = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=target_at + timedelta(minutes=1, seconds=5),
        entry_guard=missing,
    )
    allowed = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=target_at + timedelta(minutes=1, seconds=5),
        entry_guard=absent,
    )

    assert missing["action"] == "DEFER_COVERAGE_MISSING"
    assert deferred["reason"] == "stage131_coverage_missing"
    assert absent["action"] == "ALLOW_COVERAGE_PROVEN"
    assert allowed["reason"] == "fresh_source_admission"
    assert allowed["leg"]["action"] == "BUY"


def test_mcl_plan_waits_for_next_minute_and_consumes_each_admission_once() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(minutes=5)
    event_id = "3" * 64
    source = _source_checkpoint(selected, event_at=event_at, event_id=event_id)
    risk = {"safety_breaches": []}

    early = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(seconds=30),
    )
    fresh = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    consumed = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state=risk,
        consumed_admissions={event_id},
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )

    assert early["reason"] == "next_minute_entry_not_due"
    assert fresh["status"] == "ACTIONABLE"
    assert fresh["leg"] == {
        "symbol": "MCL",
        "action": "BUY",
        "quantity": 1,
        "initial_mode": "OPTIMISTIC",
        "chase_mode": "AUTO",
        "phase_speed_multiplier": 1.0,
        "outside_rth": True,
    }
    assert consumed["reason"] == "admission_already_consumed"


def test_mcl_reopen_authority_blocks_entry_but_preserves_incumbent_reduction() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(
        minutes=5
    )
    event_id = "8" * 64
    source = _source_checkpoint(
        selected, event_at=event_at, event_id=event_id, owner="shock"
    )
    source["evidence"]["target"]["route"] = "shock_continuation"
    now = event_at + timedelta(minutes=1, seconds=5)

    fresh = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=now,
    )
    blocked = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
        broker_position=0,
        risk_state={"safety_breaches": []},
        consumed_admissions=set(),
        observed_at=now,
    )
    reduced = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
        broker_position=1,
        risk_state={
            "safety_breaches": ["raw_loss_cap"],
            "admission_event_id": event_id,
        },
        consumed_admissions=set(),
        observed_at=now,
    )
    retained = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_REOPEN,
        broker_position=1,
        risk_state={"safety_breaches": [], "admission_event_id": event_id},
        consumed_admissions=set(),
        observed_at=now,
    )

    assert fresh["reason"] == "fresh_source_admission"
    assert fresh["status"] == "ACTIONABLE"
    assert blocked["reason"] == "maintenance_reopen_entry_locked"
    assert blocked["source_authority"] == MCL_LIVE_SOURCE_AUTHORITY_REOPEN
    assert blocked["leg"] is None
    assert blocked["transition_id"] != fresh["transition_id"]
    assert reduced["reason"] == "raw_loss_cap"
    assert reduced["leg"]["action"] == "SELL"
    assert reduced["leg"]["chase_mode"] == "RELENTLESS"
    assert retained["reason"] == "target_already_owned"
    assert retained["leg"] is None


def test_mcl_shock_entry_uses_accelerated_limit_ladder_and_friday_lock() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(minutes=5)
    source = _source_checkpoint(
        selected, event_at=event_at, event_id="5" * 64, owner="shock"
    )
    source["evidence"]["target"]["route"] = "shock_continuation"
    risk = {"safety_breaches": []}
    active = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    locked = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=0,
        risk_state=risk,
        consumed_admissions=set(),
        observed_at=datetime(2026, 8, 7, 20, 54, tzinfo=timezone.utc),
    )

    assert active["reason"] == "fresh_source_admission"
    assert active["leg"]["phase_speed_multiplier"] == 2.0
    assert active["leg"]["chase_mode"] == "AUTO"
    assert locked["reason"] == "weekly_closure_entry_lock"
    assert locked["leg"] is None


def test_mcl_same_direction_cannot_inherit_a_new_admission_across_restart() -> None:
    selected = _selection()
    event_at = datetime.fromisoformat(selected["selected_at_utc"]) + timedelta(
        minutes=5
    )
    source = _source_checkpoint(
        selected, event_at=event_at, event_id="6" * 64, owner="shock"
    )
    source["evidence"]["target"]["route"] = "shock_continuation"
    retained = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=1,
        risk_state={"safety_breaches": [], "admission_event_id": "6" * 64},
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )
    replaced = project_mcl_transport_plan(
        selection=selected,
        source_checkpoint=source,
        source_authority=MCL_LIVE_SOURCE_AUTHORITY_FRESH,
        broker_position=1,
        risk_state={"safety_breaches": [], "admission_event_id": "7" * 64},
        consumed_admissions=set(),
        observed_at=event_at + timedelta(minutes=1, seconds=5),
    )

    assert retained["reason"] == "target_already_owned"
    assert retained["leg"] is None
    assert replaced["reason"] == "source_admission_identity_changed"
    assert replaced["leg"]["action"] == "SELL"
    assert replaced["leg"]["chase_mode"] == "RELENTLESS"


def test_mcl_actual_fill_risk_uses_failed_auction_memory_and_raw_cap() -> None:
    selected = _selection()
    event_id = "4" * 64
    entry_at = AT + timedelta(minutes=10)
    plan = {
        "reason": "fresh_v18_admission",
        "target_route": "failed_auction",
        "admission_event_id": event_id,
    }
    record = {
        "strategy_version": MCL_LIVE_EXECUTION_VERSION,
        "evidence": {
            "selection_id": selected["selection_id"],
            "phase": "TERMINAL",
            "order_ref": "MCLV18-test",
            "plan": plan,
            "broker_order": {
                "con_id": selected["contracts"]["MCL"]["con_id"],
                "fills": [
                    {
                        "exec_id": "entry",
                        "time_utc": entry_at.isoformat(),
                        "side": "BOT",
                        "shares": 1,
                        "price": 80.0,
                        "commission": 0.76,
                        "commission_currency": "USD",
                    }
                ],
            },
        },
    }
    bars = [
        OhlcvBar(
            entry_at + timedelta(minutes=1),
            80.0,
            80.5,
            79.9,
            80.4,
            10,
        ),
        OhlcvBar(
            entry_at + timedelta(minutes=2),
            80.4,
            80.45,
            80.1,
            80.2,
            10,
        ),
    ]
    risk = mcl_transport_risk_state(
        selection=selected,
        records=[record],
        observed_at=entry_at + timedelta(minutes=3),
        liquidation_price=76.9,
        completed_mcl_bars=bars,
    )

    assert risk["position_from_fills"] == 1
    assert risk["owner"] == "v18"
    assert risk["admission_event_id"] == event_id
    assert risk["mfe_usd"] == 50.0
    assert risk["profit_memory_stop"] == 80.125
    assert risk["exit_triggers"] == ["failed_auction_profit_memory"]
    assert set(risk["safety_breaches"]) == {
        "raw_loss_cap",
        "run_drawdown_limit_breached",
    }
    assert risk["run_gross_usd"] == -310.0
    assert risk["run_cost_usd"] == 1.52
    assert risk["run_net_usd"] == -311.52
    assert risk["drawdown_usd"] == 311.52
    assert risk["fill_count"] == 1
    assert risk["attribution_complete"] is True


def test_mcl_live_worker_is_shared_locked_limit_only_and_maintenance_aware() -> None:
    service = (ROOT / "deploy/systemd/tradebot-mcl-live.service").read_text()
    timer = (ROOT / "deploy/systemd/tradebot-mcl-live.timer").read_text()
    runtime = (ROOT / "tradebot/research/mcl_live.py").read_text()

    assert "%t/tradebot-live-account.lock" in service
    assert "/usr/bin/flock --exclusive --wait 180" in service
    assert "Environment=IBKR_READONLY=0" in service
    assert "MCL_STAGE131_COVERAGE=" in service
    assert "MCL_SHOCK_WAVE_LEDGER=" in service
    assert "tradebot.research.mcl_live_cli" in service
    assert "Restart=on-failure" in service
    assert "Sun *-*-* 18..23:*:10 America/New_York" in timer
    assert "Mon..Thu *-*-* 00..16:*:10 America/New_York" in timer
    assert "Mon..Thu *-*-* 18..23:*:10 America/New_York" in timer
    assert "Fri *-*-* 00..16:*:10 America/New_York" in timer
    assert "MarketOrder" not in runtime
    assert "place_market" not in runtime


def test_mcl_live_binding_uses_the_one_durable_worker() -> None:
    from tradebot.live.strategies import LIVE_STRATEGY_BINDINGS

    binding = next(
        item for item in LIVE_STRATEGY_BINDINGS if item.champion_symbol == "MCL"
    )
    assert binding.strategy_id == MCL_TWO_SPEED_SHOCK_VERSION
    assert binding.execution_strategy_version == MCL_LIVE_EXECUTION_VERSION
    assert binding.timer_unit == "tradebot-mcl-live.timer"
    assert binding.service_unit == "tradebot-mcl-live.service"
    assert binding.champion_track == "HF"


def _profitability_risk(
    *, net: float = 0.0, fills: int = 0, trades: int = 0
) -> dict[str, object]:
    return {
        "valid": True,
        "attribution_complete": True,
        "position_from_fills": 0.0,
        "run_realized_gross_usd": net,
        "run_realized_cost_usd": 0.0,
        "run_realized_net_usd": net,
        "open_mark_gross_usd": 0.0,
        "open_mark_cost_usd": 0.0,
        "open_mark_net_usd": 0.0,
        "run_gross_usd": net,
        "run_cost_usd": 0.0,
        "run_net_usd": net,
        "peak_run_net_usd": max(0.0, net),
        "drawdown_usd": max(0.0, -net),
        "closed_trades": trades,
        "gross_wins_usd": max(0.0, net),
        "top_five_gross_wins_usd": max(0.0, net),
        "fill_count": fills,
        "exit_triggers": [],
        "safety_breaches": [],
    }


def _profitability_state(
    selected: dict[str, object],
    evaluated: datetime,
    *,
    net: float = 0.0,
    fills: int = 0,
    trades: int = 0,
) -> dict[str, object]:
    recorded = evaluated + timedelta(seconds=10)
    checkpoint = hashlib.sha256(evaluated.isoformat().encode()).hexdigest()
    return {
        "kind": "checkpoint",
        "checkpoint_id": checkpoint,
        "recorded_at_utc": recorded.isoformat(),
        "evaluation_as_of_utc": recorded.isoformat(),
        "strategy_id": selected["strategy_version"],
        "strategy_version": MCL_LIVE_EXECUTION_VERSION,
        "status": "EVALUATED",
        "evidence": {
            "selection_id": selected["selection_id"],
            "phase": "STATE",
            "submitted_orders": 0,
            "plan": {"held_direction": None, "leg": None},
            "broker_state": {"positions": [], "open_orders": []},
            "risk_state": _profitability_risk(
                net=net, fills=fills, trades=trades
            ),
        },
    }


def test_mcl_clock_owns_gth_and_excludes_daily_and_weekend_closures() -> None:
    assert mcl_market_open("2026-08-03T20:59:00+00:00")
    assert not mcl_market_open("2026-08-03T21:00:00+00:00")
    assert not mcl_market_open("2026-08-02T21:59:00+00:00")
    assert mcl_market_open("2026-08-02T22:00:00+00:00")
    slots = mcl_live_evaluation_slots(
        "2026-08-03T20:58:00+00:00", "2026-08-03T22:01:00+00:00"
    )
    assert [row.isoformat() for row in slots] == [
        "2026-08-03T20:59:00+00:00",
        "2026-08-03T22:00:00+00:00",
        "2026-08-03T22:01:00+00:00",
    ]


def test_mcl_legacy_zero_prefix_normalizes_but_nonzero_history_cannot() -> None:
    legacy = {
        "schema": "mcl.two-speed-auction-risk-state.v1",
        "position_from_fills": 0,
        "open_exec_id": None,
        "entry_time_utc": None,
        "entry_price": None,
        "run_realized_net_usd": 0.0,
        "closed_trades": 0,
        "unrealized_raw_usd": 0.0,
        "fill_ledger_fingerprint": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "safety_breaches": [],
    }
    normalized = normalize_mcl_risk(legacy)
    assert normalized["valid"] is True
    assert normalized["run_net_usd"] == 0.0
    assert normalized["fill_count"] == 0
    with pytest.raises(ValueError, match="cannot be normalized"):
        normalize_mcl_risk({**legacy, "position_from_fills": 1})


def test_mcl_positive_24h_requires_complete_minutes_and_authentic_fill() -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    end = baseline + timedelta(hours=24)
    rows = [_profitability_state(selected, baseline)]
    rows.extend(
        _profitability_state(selected, slot, net=4.0, fills=2, trades=1)
        for slot in mcl_live_evaluation_slots(baseline, end)
    )
    receipt = mcl_live_profitability_receipt(
        rows,
        selection=selected,
        as_of=end + timedelta(seconds=55),
    )
    assert receipt["status"] == "ACTIVE"
    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["milestones"]["48h"]["passed"] is False
    assert receipt["economics"]["net_usd"] == 4.0
    assert receipt["economics"]["fills"] == 2


def test_mcl_coverage_epoch_inherits_economics_and_quarantines_new_gaps() -> None:
    selected = _selection()
    start = datetime(2026, 8, 4, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)
    terminal_risk = _profitability_risk(net=-11.0, fills=2, trades=1)
    epoch = {
        "schema": FUTURES_PROFITABILITY_COVERAGE_EPOCH_SCHEMA,
        "epoch_id": "c" * 64,
        "eligible_start_utc": start.isoformat(),
        "selection": {"selection_id": selected["selection_id"]},
        "terminal_checkpoint": {"risk_state": terminal_risk},
    }
    rows = [_profitability_state(selected, start, net=-11.0, fills=2, trades=1)]
    rows.extend(
        _profitability_state(selected, slot, net=5.0, fills=4, trades=2)
        for slot in mcl_live_evaluation_slots(start, end)
    )

    receipt = mcl_profitability_receipt_with_coverage_epoch(
        rows,
        selection=selected,
        as_of=end + timedelta(seconds=55),
        coverage_epoch=epoch,
    )
    missing = mcl_profitability_receipt_with_coverage_epoch(
        rows[:-2] + rows[-1:],
        selection=selected,
        as_of=end + timedelta(seconds=55),
        coverage_epoch=epoch,
    )

    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["clock"]["coverage_started_at_utc"] == start.isoformat()
    assert receipt["clock"]["coverage_epoch_id"] == "c" * 64
    assert receipt["economics"]["net_usd"] == 5.0
    assert receipt["economics"]["maximum_drawdown_usd"] == 11.0
    assert receipt["sessions"][0]["net_usd"] == 16.0
    assert missing["status"] == "INVALID_EVIDENCE"
    assert "incomplete_session_coverage" in missing["reasons"]


def test_mcl_coverage_epoch_rehashes_selection_receipts_and_terminal_state(
    tmp_path: Path,
) -> None:
    selected = _selection()
    selection_path = tmp_path / f"db/calibration/selections/{selected['selection_id']}.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selected), encoding="utf-8")
    registered = datetime(2026, 8, 4, 9, 0, 30, tzinfo=timezone.utc)
    eligible = datetime(2026, 8, 4, 9, 1, tzinfo=timezone.utc)

    predecessor = json.loads(
        (ROOT / "backtests/mcl/mcl_v18_shock_stage112_preliminary_24h_graduation.json").read_text()
    )
    predecessor["subject"]["selection_id"] = selected["selection_id"]
    predecessor["subject"]["run_id"] = selected["selection_id"]
    predecessor["target"]["cutoff_utc"] = (registered - timedelta(minutes=1)).isoformat()
    predecessor.pop("receipt_id")
    predecessor["receipt_id"] = evidence_sha256(predecessor)
    predecessor_path = tmp_path / "backtests/mcl/predecessor.json"
    predecessor_path.parent.mkdir(parents=True)
    predecessor_path.write_text(json.dumps(predecessor), encoding="utf-8")

    preregistration_path = tmp_path / "backtests/mcl/preregistration.json"
    preregistration_path.write_text(
        json.dumps(
            {
                "registered_at_utc": registered.isoformat(),
                "eligible_start_utc": eligible.isoformat(),
                "selection": {
                    "selection_id": selected["selection_id"],
                    "path": selection_path.relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    terminal = _profitability_state(selected, registered - timedelta(seconds=20))
    fields = (
        "evaluation_as_of_utc",
        "strategy_id",
        "strategy_version",
        "trading_date",
        "session",
        "status",
        "evidence",
        "recorded_at_utc",
    )
    terminal.update(
        {
            "trading_date": registered.date().isoformat(),
            "session": "MCL_GTH",
        }
    )
    terminal["checkpoint_id"] = evidence_sha256(
        {field: terminal[field] for field in fields}
    )

    epoch = build_mcl_profitability_coverage_epoch(
        selection=selected,
        selection_path=selection_path,
        records=(terminal,),
        predecessor_receipt_paths=(predecessor_path,),
        preregistration_path=preregistration_path,
        registered_at_utc=registered,
        eligible_start_utc=eligible,
        repo_root=tmp_path,
    )
    with pytest.raises(
        ValueError, match="MCL coverage epoch milestone boundary is closed"
    ):
        build_mcl_profitability_coverage_epoch(
            selection=selected,
            selection_path=selection_path,
            records=(terminal,),
            predecessor_receipt_paths=(predecessor_path,),
            preregistration_path=preregistration_path,
            registered_at_utc="2026-08-07T07:18:00+00:00",
            eligible_start_utc="2026-08-07T08:00:00+00:00",
            repo_root=tmp_path,
        )
    epoch_path = tmp_path / "backtests/mcl/epoch.json"
    epoch_path.write_text(json.dumps(epoch), encoding="utf-8")

    assert load_mcl_profitability_coverage_epoch(
        epoch_path,
        selection=selected,
        selection_path=selection_path,
        records=(terminal,),
        repo_root=tmp_path,
    ) == epoch
    tampered = deepcopy(epoch)
    tampered["terminal_checkpoint"]["submitted_orders"] = 1
    epoch_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="futures profitability coverage epoch"):
        load_mcl_profitability_coverage_epoch(
            epoch_path,
            selection=selected,
            selection_path=selection_path,
            records=(terminal,),
            repo_root=tmp_path,
        )


def test_mcl_same_minute_receipt_noise_collapses_by_economic_state() -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    first = _profitability_state(selected, baseline + timedelta(minutes=1))
    duplicate = deepcopy(first)
    duplicate["checkpoint_id"] = "d" * 64
    duplicate["recorded_at_utc"] = (baseline + timedelta(minutes=1, seconds=30)).isoformat()
    duplicate["evaluation_as_of_utc"] = duplicate["recorded_at_utc"]
    duplicate["evidence"]["source_checkpoint_id"] = "e" * 64
    duplicate["evidence"]["quote"] = {"bid": 75.81, "ask": 75.82}
    duplicate["evidence"]["broker_state"]["observed_at_utc"] = duplicate[
        "recorded_at_utc"
    ]
    duplicate["evidence"]["risk_state"].update(
        {
            "as_of_utc": duplicate["recorded_at_utc"],
            "liquidation_price": 75.81,
            "observed_at_utc": duplicate["recorded_at_utc"],
        }
    )

    receipt = mcl_live_profitability_receipt(
        [
            _profitability_state(selected, baseline),
            first,
            duplicate,
        ],
        selection=selected,
        as_of=baseline + timedelta(minutes=2),
    )

    assert receipt["status"] == "ACTIVE"
    assert receipt["reasons"] == []
    assert receipt["clock"]["evaluated_slots"] == 1


@pytest.mark.parametrize(
    "change",
    ("fill", "position", "safety", "order"),
)
def test_mcl_same_minute_material_state_change_remains_a_conflict(
    change: str,
) -> None:
    selected = _selection()
    baseline = datetime(2026, 8, 4, 8, 32, tzinfo=timezone.utc)
    first = _profitability_state(selected, baseline + timedelta(minutes=1))
    changed = deepcopy(first)
    changed["checkpoint_id"] = "f" * 64
    changed["recorded_at_utc"] = (baseline + timedelta(minutes=1, seconds=30)).isoformat()
    changed["evaluation_as_of_utc"] = changed["recorded_at_utc"]
    if change == "fill":
        changed["evidence"]["risk_state"]["fill_count"] = 1
    elif change == "position":
        changed["evidence"]["broker_state"]["positions"] = [
            {"symbol": "MCL", "con_id": selected["contracts"]["MCL"]["con_id"], "quantity": 1.0}
        ]
    elif change == "safety":
        changed["evidence"]["risk_state"]["safety_breaches"] = [
            "raw_loss_cap"
        ]
    else:
        changed["evidence"]["broker_state"]["open_orders"] = [
            {
                "symbol": "MCL",
                "con_id": selected["contracts"]["MCL"]["con_id"],
                "order_ref": "MCLV18-test",
                "status": "Submitted",
            }
        ]

    receipt = mcl_live_profitability_receipt(
        [
            _profitability_state(selected, baseline),
            first,
            changed,
        ],
        selection=selected,
        as_of=baseline + timedelta(minutes=2),
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert "conflicting_session_coverage" in receipt["reasons"]


def test_mcl_runtime_gate_rehashes_the_selected_stage112_owners() -> None:
    passed = mcl_runtime_parity_graduation_gate(
        selection=_selection(), repo_root=ROOT
    )
    assert passed["status"] == "PASS"
    changed = _selection()
    changed["evidence"]["lifecycle_parity"]["sha256"] = "0" * 64
    rejected = mcl_runtime_parity_graduation_gate(
        selection=changed, repo_root=ROOT
    )
    assert rejected["status"] == "INVALID"


def test_mcl_cli_graduation_never_loads_broker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradebot.research import mcl_live_cli

    selected = _selection()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selected), encoding="utf-8")
    output = tmp_path / "graduation.json"
    monkeypatch.setattr(mcl_live_cli, "load_live_capital_plan", lambda _path: {})
    monkeypatch.setattr(
        mcl_live_cli,
        "load_allocated_live_selection",
        lambda *_args, **_kwargs: (selected, selection_path, "0" * 64),
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "LiveCalibrationLedger",
        lambda _path: type("Ledger", (), {"records": lambda self: ()})(),
    )
    monkeypatch.setattr(
        mcl_live_cli, "live_calibration_logical_prefix", lambda *_args, **_kwargs: ({}, ())
    )
    monkeypatch.setattr(
        mcl_live_cli, "mcl_live_profitability_receipt", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        mcl_live_cli, "mcl_live_graduation_inputs", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "reduce_live_graduation",
        lambda **_kwargs: {"verdict": "HOLD"},
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "publish_live_graduation_receipt",
        lambda path, receipt: path.write_text(json.dumps(receipt)),
    )
    monkeypatch.setattr(
        mcl_live_cli,
        "load_config",
        lambda: pytest.fail("graduation queried broker configuration"),
    )
    code = asyncio.run(
        mcl_live_cli._main_async(
            [
                "--capital-plan",
                str(tmp_path / "capital.json"),
                "--graduation-target",
                "24h",
                "--graduation-cutoff",
                "2026-08-04T10:00:00+00:00",
                "--graduation-output",
                str(output),
            ]
        )
    )
    assert code == 0
    assert json.loads(output.read_text()) == {"verdict": "HOLD"}


def test_mcl_epoch_cli_is_broker_free_and_binds_epoch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradebot.research import mcl_profitability_epoch

    selected = _selection()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selected), encoding="utf-8")
    output = tmp_path / "graduation.json"
    epoch = {
        "epoch_id": "e" * 64,
        "eligible_start_utc": "2026-08-04T09:01:00+00:00",
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mcl_profitability_epoch, "load_live_capital_plan", lambda _path: {}
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "load_allocated_live_selection",
        lambda *_args, **_kwargs: (selected, selection_path, "0" * 64),
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "LiveCalibrationLedger",
        lambda _path: type("Ledger", (), {"records": lambda self: ()})(),
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "load_mcl_profitability_coverage_epoch",
        lambda *_args, **_kwargs: epoch,
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "live_calibration_logical_prefix",
        lambda *_args, **_kwargs: ({}, ()),
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "mcl_profitability_receipt_with_coverage_epoch",
        lambda *_args, **kwargs: captured.setdefault("profitability", kwargs)
        and {},
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "mcl_graduation_inputs_with_coverage_epoch",
        lambda **kwargs: captured.setdefault("inputs", kwargs) and {},
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "reduce_live_graduation",
        lambda **_kwargs: {"verdict": "HOLD"},
    )
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "publish_live_graduation_receipt",
        lambda path, receipt: path.write_text(json.dumps(receipt)),
    )

    code = mcl_profitability_epoch.main(
        [
            "--capital-plan",
            str(tmp_path / "capital.json"),
            "--graduation-target",
            "24h",
            "--graduation-cutoff",
            "2026-08-04T10:00:00+00:00",
            "--graduation-coverage-epoch",
            str(tmp_path / "epoch.json"),
            "--graduation-output",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text()) == {"verdict": "HOLD"}
    assert captured["profitability"]["coverage_epoch"] == epoch
    assert captured["inputs"]["coverage_epoch"] == epoch


def test_mcl_epoch_adapter_adds_only_graduation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tradebot.research import mcl_profitability_epoch

    base = {
        "selection": {"selection_id": "s"},
        "ledger_prefix": {"schema": "prefix"},
        "subject": {"strategy_id": "mcl"},
    }
    monkeypatch.setattr(
        mcl_profitability_epoch,
        "mcl_live_graduation_inputs",
        lambda **_kwargs: deepcopy(base),
    )
    epoch = {
        "epoch_id": "e" * 64,
        "eligible_start_utc": "2026-08-07T08:00:00+00:00",
    }

    inputs = mcl_profitability_epoch.mcl_graduation_inputs_with_coverage_epoch(
        coverage_epoch=epoch
    )

    assert inputs["subject"] == base["subject"]
    assert inputs["selection"] == {
        **base["selection"],
        "coverage_epoch_id": epoch["epoch_id"],
        "coverage_started_at_utc": epoch["eligible_start_utc"],
    }
    assert inputs["ledger_prefix"] == {
        **base["ledger_prefix"],
        "coverage_epoch_id": epoch["epoch_id"],
        "coverage_started_at_utc": epoch["eligible_start_utc"],
    }
