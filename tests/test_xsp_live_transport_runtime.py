from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from ib_insync import Stock

from tradebot.engines.execution import execution_policy_contract
from tradebot.research.live_calibration import (
    LiveCalibrationLedger,
    calibration_fingerprint,
)
from tradebot.research.xsp_live_transport import (
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V2_TRANSPORT_PLAN_SCHEMA,
    XSP_V2_TRANSPORT_SELECTION_SCHEMA,
    project_xsp_v2_transport_plan,
)
from tradebot.research.xsp_live_transport_runtime import (
    XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
    XSP_V2_TRANSPORT_EXECUTION_VERSION,
    advance_xsp_v2_live_transport,
    execute_xsp_v2_transport_plan,
    xsp_v2_transport_risk_state,
    xsp_v2_transport_order_ref,
)
from tradebot.research.xsp_opening_edge_v2 import (
    XSP_OPENING_EDGE_V2_VERSION,
)


SELECTED_AT = datetime(2026, 7, 29, 13, 38, tzinfo=timezone.utc)
OBSERVED_AT = SELECTED_AT + timedelta(minutes=4)


@dataclass
class _Preview:
    status: str | None = "PreSubmitted"
    init_margin_before: float | None = None
    init_margin_change: float | None = None
    init_margin_after: float | None = None
    maintenance_margin_before: float | None = None
    maintenance_margin_change: float | None = None
    maintenance_margin_after: float | None = None
    equity_with_loan_before: float | None = None
    equity_with_loan_change: float | None = None
    equity_with_loan_after: float | None = None
    commission: float | None = 1.0
    min_commission: float | None = 1.0
    max_commission: float | None = 1.0
    commission_currency: str | None = "USD"
    warning_text: str | None = None


def _trade(
    contract,
    order_ref: str,
    *,
    action: str = "BUY",
    quantity: int = 8,
):
    side = "BOT" if action == "BUY" else "SLD"
    fill = SimpleNamespace(
        execution=SimpleNamespace(
            execId="exec-117",
            side=side,
            shares=quantity,
            price=30.50,
        ),
        commissionReport=SimpleNamespace(commission=1.0, currency="USD"),
        time=OBSERVED_AT,
    )
    return SimpleNamespace(
        contract=contract,
        order=SimpleNamespace(
            orderId=17,
            permId=117,
            orderRef=order_ref,
            action=action,
            totalQuantity=quantity,
            lmtPrice=30.50,
        ),
        orderStatus=SimpleNamespace(
            status="Filled",
            filled=quantity,
            remaining=0,
            avgFillPrice=30.50,
        ),
        fills=[fill],
        isDone=lambda: True,
    )


class _Client:
    def __init__(self, contract) -> None:
        self.contract = contract
        self.matches = []
        self.previewed = []
        self.placed = []
        self.trade = None

    async def reconcile_trades_for_order_ref(self, order_ref: str):
        return list(self.matches or ([self.trade] if self.trade else []))

    async def preview_limit_order(self, *args):
        self.previewed.append(args)
        return _Preview()

    async def place_limit_order(self, *args):
        self.placed.append(args)
        self.trade = _trade(
            self.contract,
            str(args[-1]),
            action=str(args[1]),
            quantity=int(args[2]),
        )
        return self.trade

    async def reconcile_order_state(self, **_kwargs):
        return {"trade": self.trade or self.matches[0]}


class _LiveClient(_Client):
    def __init__(self, spyu, spxu) -> None:
        super().__init__(spyu)
        self.contracts = {"SPYU": spyu, "SPXU": spxu}
        self.portfolio = [
            SimpleNamespace(
                contract=SimpleNamespace(
                    symbol="TQQQ",
                    conId=72_539_702,
                    secType="STK",
                ),
                position=1,
            )
        ]
        self.external_open_trades = []
        self.cash_observed_at = datetime.now(timezone.utc)
        self.tickers = {
            "SPYU": SimpleNamespace(
                contract=spyu,
                bid=30.48,
                ask=30.50,
                last=30.49,
                close=30.47,
                bidSize=100,
                askSize=100,
                marketDataType=1,
                tbTopQuoteUpdatedMono=time.monotonic(),
            ),
            "SPXU": SimpleNamespace(
                contract=spxu,
                bid=10.18,
                ask=10.20,
                last=10.19,
                close=10.17,
                bidSize=100,
                askSize=100,
                marketDataType=1,
                tbTopQuoteUpdatedMono=time.monotonic(),
            ),
        }

    async def fetch_portfolio(self):
        return list(self.portfolio)

    def account_id(self):
        return "DU123456"

    def account_text_value(self, tag):
        return "STKCASH" if tag == "TradingType-S" else None

    def account_value(self, tag, *, currency=None):
        assert (tag, currency) == ("CashBalance", "USD")
        return 1_500.0, "USD", self.cash_observed_at

    def open_trades(self):
        return list(self.external_open_trades)

    async def qualify_proxy_contracts(self, *_contracts):
        return [self.contracts["SPYU"], self.contracts["SPXU"]]

    async def ensure_ticker(self, contract, **_kwargs):
        return self.tickers[contract.symbol]

    @staticmethod
    def generic_tick_value(_ticker, tick_type):
        assert tick_type == 96
        return 30.49, datetime.now(timezone.utc)


def _actionable(tmp_path: Path):
    del tmp_path
    body = {
        "schema": XSP_V2_TRANSPORT_SELECTION_SCHEMA,
        "selected_at_utc": SELECTED_AT.isoformat(),
        "run_started_at_utc": SELECTED_AT.isoformat(),
        "strategy_version": XSP_OPENING_EDGE_V2_VERSION,
        "authority": "selected_live_cash_transport",
        "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
        "profitability_clock_started": True,
        "execution_session": "RTH",
        "direction_symbols": {"up": "SPYU", "down": "SPXU"},
        "nominee": {
            "family": "five_slot",
            "profile_id": "fixed_measured",
            "fixed_entry_notional_usd": 260.0,
            "capital_identity": {
                "starting_cash_identity_usd": 1_350.0,
                "fixed_entry_notional_usd": 260.0,
                "cash_slots": 5,
                "maximum_gross_purchase_notional_usd": 1_300.0,
                "settlement": "strict_T_plus_1_settled_cash_only",
                "unsettled_sale_proceeds_reused": False,
            },
            "historical_quantity_ranges": {
                "SPYU": [7, 10],
                "SPXU": [20, 30],
            },
            "commission_limits_usd": {
                "SPYU": 1.010129,
                "SPXU": 1.010093,
            },
            "contract_ids": {
                "SPYU": 669_475_151,
                "SPXU": 533_620_647,
            },
        },
        "baseline_state": None,
        "broker_at_selection": {
            "observed_at_utc": (
                SELECTED_AT - timedelta(seconds=5)
            ).isoformat(),
            "cash_observed_at_utc": (
                SELECTED_AT - timedelta(seconds=10)
            ).isoformat(),
            "account_id": "DU123456",
            "account_type": "CASH",
            "settled_cash_usd": 1_350.0,
            "minimum_settled_cash_usd": 1_305.050645,
            "positions": {"SPYU": 0, "SPXU": 0},
            "unrelated_positions": [],
            "open_orders": [],
        },
        "risk": {
            "starting_cash_identity_usd": 1_350.0,
            "settlement": "strict_T_plus_1_settled_cash_only",
            "max_drawdown_usd": 135.0,
            "max_session_loss_usd": 67.5,
            "gth_execution_allowed": False,
        },
        "execution": {
            "SPYU_BUY": {
                "initial_mode": "CROSS",
                "chase_mode": "RELENTLESS",
            },
            "SPXU_BUY": {
                "initial_mode": "OPTIMISTIC",
                "chase_mode": "AUTO",
            },
            "SELL": {
                "initial_mode": "OPTIMISTIC",
                "chase_mode": "AUTO",
            },
            "sell_before_buy": True,
            "partial_buy": "hold_filled_quantity_without_top_up",
            "partial_sell": "no_new_buy_until_flat",
            "stale_or_ambiguous_state": "HOLD",
            "fresh_streaming_nbbo_required": True,
            "stale_top_action": "pause_repricing_until_fresh_or_timeout",
            "policy_contract": execution_policy_contract(),
        },
        "evidence": {
            "ranking": {},
            "dwell": {},
            "preview": {},
            "source_checkpoint_id": "source",
            "source_recorded_at_utc": SELECTED_AT.isoformat(),
        },
    }
    selection = {
        **body,
        "selection_id": calibration_fingerprint(body),
    }
    plan = {
        "schema": XSP_V2_TRANSPORT_PLAN_SCHEMA,
        "selection_id": selection["selection_id"],
        "transition_id": "a" * 64,
        "source_checkpoint_id": "source-next",
        "source_session": "RTH",
        "order_authority": XSP_V2_TRANSPORT_ORDER_AUTHORITY,
        "status": "ACTIONABLE",
        "leg": {
            "action": "BUY",
            "symbol": "SPYU",
            "quantity": 8,
            "initial_mode": "CROSS",
            "chase_mode": "RELENTLESS",
            "outside_rth": False,
            "bid": 30.48,
            "ask": 30.50,
        },
    }
    contract = Stock("SPYU", "SMART", "USD")
    contract.conId = 669475151
    ticker = SimpleNamespace(
        bid=30.48,
        ask=30.50,
        last=30.49,
        close=30.47,
    )
    return selection, plan, contract, ticker


def _fill_record(
    selection,
    *,
    exec_id: str,
    when: datetime,
    symbol: str,
    side: str,
    shares: int,
    price: float,
):
    return {
        "kind": "checkpoint",
        "strategy_version": XSP_V2_TRANSPORT_EXECUTION_VERSION,
        "trading_date": when.date().isoformat(),
        "evidence": {
            "selection_id": selection["selection_id"],
            "phase": "TERMINAL",
            "broker_order": {
                "filled": shares,
                "fills": [
                    {
                        "exec_id": exec_id,
                        "time_utc": when.isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "shares": shares,
                        "price": price,
                        "commission": 1.0,
                        "commission_currency": "USD",
                    }
                ],
            },
        },
    }


def _live_source(
    position: dict[str, object] | None,
    *,
    recorded_at: datetime,
) -> dict[str, object]:
    profile = {
        "run_started_at_utc": SELECTED_AT.isoformat(),
        "latest_position": position,
    }
    return {
        "evaluation_status": "EVALUATED",
        "freshness_ok": True,
        "session": "RTH",
        "order_authority": "none",
        "checkpoint_id": "live-source",
        "recorded_at_utc": recorded_at.isoformat(),
        "paired_equity": {
            "crown_config_fingerprint": "crown",
            "profiles": {
                "research": dict(profile),
                "broker": dict(profile),
            },
        },
    }


def test_selected_leg_is_previewed_submitted_and_terminal_once(
    tmp_path: Path,
) -> None:
    selection, plan, contract, ticker = _actionable(tmp_path)
    ledger = LiveCalibrationLedger(tmp_path / "execution.jsonl")
    client = _Client(contract)

    result = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT,
        )
    )

    assert result["status"] == "TERMINAL"
    assert result["submitted_orders"] == 1
    assert len(client.previewed) == len(client.placed) == 1
    assert client.placed[0][-1] == xsp_v2_transport_order_ref(plan)
    assert client.placed[0][-2] is False
    assert [row["evidence"]["phase"] for row in ledger.records()] == [
        "PREPARED",
        "SUBMITTED",
        "TERMINAL",
    ]

    again = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT + timedelta(seconds=1),
        )
    )
    assert again["submitted_orders"] == 0
    assert len(client.placed) == 1


def test_extended_hours_authority_is_sell_only(tmp_path: Path) -> None:
    selection, plan, contract, ticker = _actionable(tmp_path)
    plan["leg"] = {
        **plan["leg"],
        "action": "SELL",
        "initial_mode": "OPTIMISTIC",
        "chase_mode": "AUTO",
        "outside_rth": True,
    }
    ledger = LiveCalibrationLedger(tmp_path / "extended-exit.jsonl")
    client = _Client(contract)

    result = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT,
        )
    )
    assert result["status"] == "TERMINAL"
    assert client.previewed[0][-2] is True
    assert client.placed[0][-2] is True

    buy = deepcopy(plan)
    buy["leg"]["action"] = "BUY"
    with pytest.raises(ValueError, match="transport leg"):
        asyncio.run(
            execute_xsp_v2_transport_plan(
                LiveCalibrationLedger(tmp_path / "forbidden-buy.jsonl"),
                client=_Client(contract),
                selection=selection,
                plan=buy,
                contract=contract,
                ticker=ticker,
                observed_at=OBSERVED_AT,
            )
        )


def test_prepared_restart_adopts_exact_broker_order_without_resubmission(
    tmp_path: Path,
) -> None:
    selection, plan, contract, ticker = _actionable(tmp_path)
    ledger = LiveCalibrationLedger(tmp_path / "execution.jsonl")
    client = _Client(contract)
    order_ref = xsp_v2_transport_order_ref(plan)
    client.matches = [_trade(contract, order_ref)]
    ledger.checkpoint(
        evaluation_as_of=OBSERVED_AT,
        strategy_id=XSP_OPENING_EDGE_V2_VERSION,
        strategy_version=XSP_V2_TRANSPORT_EXECUTION_VERSION,
        trading_date="2026-07-29",
        session="RTH",
        status="EVALUATED",
        evidence={
            "schema": XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
            "selection_id": selection["selection_id"],
            "transition_id": plan["transition_id"],
            "phase": "PREPARED",
            "order_ref": order_ref,
            "what_if_preview": {"commission": 1.0},
        },
        recorded_at=OBSERVED_AT,
    )

    result = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=datetime(2026, 7, 29, 13, 45, tzinfo=timezone.utc),
        )
    )

    assert result["submitted_orders"] == 0
    assert client.previewed == []
    assert client.placed == []
    assert [row["evidence"]["phase"] for row in ledger.records()] == [
        "PREPARED",
        "TERMINAL",
    ]


def test_resumed_pending_order_ages_from_first_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection, plan, contract, ticker = _actionable(tmp_path)
    ledger = LiveCalibrationLedger(tmp_path / "execution.jsonl")
    client = _Client(contract)
    order_ref = xsp_v2_transport_order_ref(plan)
    pending = _trade(contract, order_ref)
    pending.orderStatus = SimpleNamespace(
        status="Submitted",
        filled=0,
        remaining=8,
        avgFillPrice=0,
    )
    pending.fills = []
    pending.isDone = lambda: False
    client.matches = [pending]
    client.trade = pending
    first_submitted = OBSERVED_AT - timedelta(minutes=4)
    for recorded_at in (first_submitted, OBSERVED_AT - timedelta(minutes=2)):
        ledger.checkpoint(
            evaluation_as_of=recorded_at,
            strategy_id=XSP_OPENING_EDGE_V2_VERSION,
            strategy_version=XSP_V2_TRANSPORT_EXECUTION_VERSION,
            trading_date="2026-07-29",
            session="RTH",
            status="EVALUATED",
            evidence={
                "schema": XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
                "selection_id": selection["selection_id"],
                "transition_id": plan["transition_id"],
                "phase": "SUBMITTED",
                "order_ref": order_ref,
                "plan": plan,
                "what_if_preview": {"commission": 1.0},
            },
            recorded_at=recorded_at,
        )

    captured: dict[str, object] = {}

    class _Execution:
        def __init__(self, **_kwargs) -> None:
            pass

        async def chase(
            self,
            trade,
            _action,
            *,
            elapsed_offset_sec: float,
            require_fresh_top: bool,
            **_kwargs,
        ) -> None:
            captured["elapsed_offset_sec"] = elapsed_offset_sec
            captured["require_fresh_top"] = require_fresh_top
            filled = _trade(contract, order_ref)
            trade.order = filled.order
            trade.orderStatus = filled.orderStatus
            trade.fills = filled.fills
            trade.isDone = filled.isDone

    monkeypatch.setattr(
        "tradebot.research.xsp_live_transport_runtime.LiveOrderExecution",
        _Execution,
    )

    result = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT,
        )
    )

    assert result["status"] == "TERMINAL"
    assert captured == {
        "elapsed_offset_sec": 240.0,
        "require_fresh_top": True,
    }


def test_done_order_waits_for_commission_before_terminal_receipt(
    tmp_path: Path,
) -> None:
    selection, plan, contract, ticker = _actionable(tmp_path)
    ledger = LiveCalibrationLedger(tmp_path / "execution.jsonl")
    client = _Client(contract)
    client.trade = _trade(contract, xsp_v2_transport_order_ref(plan))
    client.trade.fills[0].commissionReport = None
    ledger.checkpoint(
        evaluation_as_of=OBSERVED_AT,
        strategy_id=XSP_OPENING_EDGE_V2_VERSION,
        strategy_version=XSP_V2_TRANSPORT_EXECUTION_VERSION,
        trading_date="2026-07-29",
        session="RTH",
        status="EVALUATED",
        evidence={
            "schema": XSP_V2_TRANSPORT_EXECUTION_SCHEMA,
            "selection_id": selection["selection_id"],
            "transition_id": plan["transition_id"],
            "phase": "SUBMITTED",
            "order_ref": xsp_v2_transport_order_ref(plan),
            "what_if_preview": {"commission": 1.0},
        },
        recorded_at=OBSERVED_AT,
    )

    pending = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT + timedelta(seconds=1),
        )
    )
    assert pending["status"] == "PENDING"
    assert client.placed == []
    assert tuple(ledger.records())[-1]["evidence"]["phase"] == "SUBMITTED"

    client.trade.fills[0].commissionReport = SimpleNamespace(
        commission=1.0,
        currency="USD",
    )
    terminal = asyncio.run(
        execute_xsp_v2_transport_plan(
            ledger,
            client=client,
            selection=selection,
            plan=plan,
            contract=contract,
            ticker=ticker,
            observed_at=OBSERVED_AT + timedelta(seconds=2),
        )
    )
    assert terminal["status"] == "TERMINAL"
    assert client.placed == []
    assert tuple(ledger.records())[-1]["evidence"]["phase"] == "TERMINAL"


def test_risk_state_uses_actual_fills_commissions_and_liquidation_bid(
    tmp_path: Path,
) -> None:
    selection, plan, _contract, _ticker = _actionable(tmp_path)
    buy_time = OBSERVED_AT - timedelta(minutes=2)
    sell_time = OBSERVED_AT - timedelta(minutes=1)
    records = (
        {
            "kind": "checkpoint",
            "strategy_version": XSP_V2_TRANSPORT_EXECUTION_VERSION,
            "trading_date": "2026-07-29",
            "evidence": {
                "selection_id": selection["selection_id"],
                "phase": "TERMINAL",
                "broker_order": {
                    "filled": 8,
                    "fills": [
                        {
                            "exec_id": "buy-1",
                            "time_utc": buy_time.isoformat(),
                            "symbol": "SPYU",
                            "side": "BOT",
                            "shares": 8,
                            "price": 30.0,
                            "commission": 1.0,
                            "commission_currency": "USD",
                        }
                    ],
                },
            },
        },
    )
    open_risk = xsp_v2_transport_risk_state(
        selection=selection,
        records=records,
        observed_at=OBSERVED_AT,
        liquidation_bids={"SPYU": 31.0},
    )
    assert open_risk["open_mark_net_usd"] == 7.0
    assert open_risk["run_net_usd"] == 7.0

    closed_records = records + (
        {
            "kind": "checkpoint",
            "strategy_version": XSP_V2_TRANSPORT_EXECUTION_VERSION,
            "trading_date": "2026-07-29",
            "evidence": {
                "selection_id": selection["selection_id"],
                "phase": "TERMINAL",
                "broker_order": {
                    "filled": 8,
                    "fills": [
                        {
                            "exec_id": "sell-1",
                            "time_utc": sell_time.isoformat(),
                            "symbol": "SPYU",
                            "side": "SLD",
                            "shares": 8,
                            "price": 31.5,
                            "commission": 1.0,
                            "commission_currency": "USD",
                        }
                    ],
                },
            },
        },
    )
    closed_risk = xsp_v2_transport_risk_state(
        selection=selection,
        records=closed_records,
        observed_at=OBSERVED_AT,
        liquidation_bids={},
    )
    assert closed_risk["run_realized_net_usd"] == 10.0
    assert closed_risk["open_mark_net_usd"] == 0.0
    assert closed_risk["fill_count"] == 2


def test_partial_buy_is_held_without_top_up(tmp_path: Path) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    partial = _fill_record(
        selection,
        exec_id="partial-buy",
        when=OBSERVED_AT - timedelta(minutes=1),
        symbol="SPYU",
        side="BOT",
        shares=3,
        price=30.0,
    )
    risk = xsp_v2_transport_risk_state(
        selection=selection,
        records=(partial,),
        observed_at=OBSERVED_AT,
        liquidation_bids={"SPYU": 30.48},
    )

    plan = project_xsp_v2_transport_plan(
        selection=selection,
        source_receipt=_live_source(
            {
                "lane": "rth",
                "direction": "up",
                "entry_time": (SELECTED_AT + timedelta(minutes=2)).isoformat(),
                "trading_date": "2026-07-29",
                "entry_price": 750.0,
                "exit_reason": "end",
            },
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 3, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=float(risk["settled_cash_usd"]),
        quotes={},
    )

    assert risk["holdings_from_fills"] == {"SPYU": 3.0, "SPXU": 0.0}
    assert plan["status"] == "UNCHANGED"
    assert plan["reason"] == "target_already_owned"
    assert plan["leg"] is None


def test_partial_sell_retries_only_the_remainder_before_any_buy(
    tmp_path: Path,
) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    buy = _fill_record(
        selection,
        exec_id="full-buy",
        when=OBSERVED_AT - timedelta(minutes=2),
        symbol="SPYU",
        side="BOT",
        shares=8,
        price=30.0,
    )
    partial_sell = _fill_record(
        selection,
        exec_id="partial-sell",
        when=OBSERVED_AT - timedelta(minutes=1),
        symbol="SPYU",
        side="SLD",
        shares=3,
        price=30.48,
    )
    risk = xsp_v2_transport_risk_state(
        selection=selection,
        records=(buy, partial_sell),
        observed_at=OBSERVED_AT,
        liquidation_bids={"SPYU": 30.48},
    )

    plan = project_xsp_v2_transport_plan(
        selection=selection,
        source_receipt=_live_source(
            {
                "lane": "rth",
                "direction": "down",
                "entry_time": (SELECTED_AT + timedelta(minutes=2)).isoformat(),
                "trading_date": "2026-07-29",
                "entry_price": 749.0,
                "exit_reason": "end",
            },
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 5, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=float(risk["settled_cash_usd"]),
        quotes={
            "SPYU": {
                "bid": 30.48,
                "ask": 30.50,
                "age_seconds": 0.5,
                "market_data_type": 1,
            }
        },
    )

    assert risk["holdings_from_fills"] == {"SPYU": 5.0, "SPXU": 0.0}
    assert plan["reason"] == "sell_incumbent_before_target"
    assert plan["leg"]["action"] == "SELL"
    assert plan["leg"]["symbol"] == "SPYU"
    assert plan["leg"]["quantity"] == 5


def test_five_slot_cash_never_reuses_same_day_sale_proceeds(
    tmp_path: Path,
) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    records = []
    for index in range(5):
        buy_time = OBSERVED_AT + timedelta(minutes=index * 2)
        records.extend(
            [
                _fill_record(
                    selection,
                    exec_id=f"buy-{index}",
                    when=buy_time,
                    symbol="SPYU",
                    side="BOT",
                    shares=8,
                    price=32.0,
                ),
                _fill_record(
                    selection,
                    exec_id=f"sell-{index}",
                    when=buy_time + timedelta(minutes=1),
                    symbol="SPYU",
                    side="SLD",
                    shares=8,
                    price=32.0,
                ),
            ]
        )
    same_day = xsp_v2_transport_risk_state(
        selection=selection,
        records=tuple(records),
        observed_at=OBSERVED_AT + timedelta(minutes=15),
        liquidation_bids={},
    )
    assert same_day["settled_cash_usd"] == 65.0
    assert same_day["pending_settlement_usd"] == 1_275.0
    assert same_day["pending_settlements"] == [
        {
            "settlement_date": "2026-07-30",
            "proceeds_usd": 1_275.0,
        }
    ]

    sixth_buy = _fill_record(
        selection,
        exec_id="buy-5",
        when=OBSERVED_AT + timedelta(minutes=16),
        symbol="SPYU",
        side="BOT",
        shares=8,
        price=32.0,
    )
    with pytest.raises(ValueError, match="exceeds settled USD reserve"):
        xsp_v2_transport_risk_state(
            selection=selection,
            records=tuple([*records, sixth_buy]),
            observed_at=OBSERVED_AT + timedelta(minutes=17),
            liquidation_bids={"SPYU": 32.0},
        )

    next_session = xsp_v2_transport_risk_state(
        selection=selection,
        records=tuple(records),
        observed_at=OBSERVED_AT + timedelta(days=1),
        liquidation_bids={},
    )
    assert next_session["settled_cash_usd"] == 1_340.0
    assert next_session["pending_settlement_usd"] == 0.0


def test_cash_state_starts_from_selected_broker_cash_not_risk_identity(
    tmp_path: Path,
) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    selection = deepcopy(selection)
    selection["broker_at_selection"]["settled_cash_usd"] = 1_320.0
    body = {
        key: value
        for key, value in selection.items()
        if key != "selection_id"
    }
    selection["selection_id"] = calibration_fingerprint(body)

    state = xsp_v2_transport_risk_state(
        selection=selection,
        records=(),
        observed_at=OBSERVED_AT,
        liquidation_bids={},
    )

    assert selection["risk"]["starting_cash_identity_usd"] == 1_350.0
    assert state["starting_settled_cash_usd"] == 1_320.0
    assert state["settled_cash_usd"] == 1_320.0


def test_two_slot_third_same_day_buy_fails_closed(tmp_path: Path) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    selection = deepcopy(selection)
    selection["nominee"].update(
        {
            "family": "two_slot",
            "profile_id": "fixed_measured",
            "fixed_entry_notional_usd": 650.0,
            "capital_identity": {
                "starting_cash_identity_usd": 1_350.0,
                "fixed_entry_notional_usd": 650.0,
                "cash_slots": 2,
                "maximum_gross_purchase_notional_usd": 1_300.0,
                "settlement": "strict_T_plus_1_settled_cash_only",
                "unsettled_sale_proceeds_reused": False,
            },
        }
    )
    selection["broker_at_selection"]["minimum_settled_cash_usd"] = (
        1_302.020258
    )
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    selection["selection_id"] = calibration_fingerprint(body)
    records = []
    for index in range(2):
        buy_time = OBSERVED_AT + timedelta(minutes=index * 2)
        records.extend(
            [
                _fill_record(
                    selection,
                    exec_id=f"two-buy-{index}",
                    when=buy_time,
                    symbol="SPYU",
                    side="BOT",
                    shares=1,
                    price=649.0,
                ),
                _fill_record(
                    selection,
                    exec_id=f"two-sell-{index}",
                    when=buy_time + timedelta(minutes=1),
                    symbol="SPYU",
                    side="SLD",
                    shares=1,
                    price=649.0,
                ),
            ]
        )
    third = _fill_record(
        selection,
        exec_id="two-buy-2",
        when=OBSERVED_AT + timedelta(minutes=5),
        symbol="SPYU",
        side="BOT",
        shares=1,
        price=649.0,
    )
    with pytest.raises(ValueError, match="exceeds settled USD reserve"):
        xsp_v2_transport_risk_state(
            selection=selection,
            records=tuple([*records, third]),
            observed_at=OBSERVED_AT + timedelta(minutes=6),
            liquidation_bids={"SPYU": 649.0},
        )


def test_friday_sale_settles_on_monday(tmp_path: Path) -> None:
    selection, _plan, _contract, _ticker = _actionable(tmp_path)
    friday = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
    records = (
        _fill_record(
            selection,
            exec_id="friday-buy",
            when=friday,
            symbol="SPYU",
            side="BOT",
            shares=8,
            price=32.0,
        ),
        _fill_record(
            selection,
            exec_id="friday-sell",
            when=friday + timedelta(minutes=1),
            symbol="SPYU",
            side="SLD",
            shares=8,
            price=32.0,
        ),
    )
    friday_state = xsp_v2_transport_risk_state(
        selection=selection,
        records=records,
        observed_at=friday + timedelta(minutes=2),
        liquidation_bids={},
    )
    assert friday_state["pending_settlements"] == [
        {
            "settlement_date": "2026-08-03",
            "proceeds_usd": 255.0,
        }
    ]
    monday_state = xsp_v2_transport_risk_state(
        selection=selection,
        records=records,
        observed_at=friday + timedelta(days=3),
        liquidation_bids={},
    )
    assert monday_state["settled_cash_usd"] == 1_348.0
    assert monday_state["pending_settlements"] == []


def test_live_recurrence_preserves_unrelated_holding_and_stays_flat(
    tmp_path: Path,
) -> None:
    selection, _plan, spyu, _ticker = _actionable(tmp_path)
    spxu = Stock("SPXU", "SMART", "USD")
    spxu.conId = 533_620_647
    client = _LiveClient(spyu, spxu)
    ledger = LiveCalibrationLedger(tmp_path / "live.jsonl")

    result = asyncio.run(
        advance_xsp_v2_live_transport(
            ledger,
            client=client,
            selection=selection,
            source_receipt=_live_source(
                None,
                recorded_at=OBSERVED_AT - timedelta(seconds=30),
            ),
            observed_at=OBSERVED_AT,
            quote_wait_seconds=0,
        )
    )

    assert result["status"] == "UNCHANGED"
    assert result["plan"]["reason"] == "flat_target"
    assert result["submitted_orders"] == 0
    assert client.placed == []
    assert tuple(ledger.records())[-1]["evidence"]["broker_state"][
        "unrelated_positions"
    ] == [
        {
            "symbol": "TQQQ",
            "con_id": 72_539_702,
            "sec_type": "STK",
            "quantity": 1.0,
        }
    ]


def test_live_recurrence_rejects_stale_account_cash_snapshot(
    tmp_path: Path,
) -> None:
    selection, _plan, spyu, _ticker = _actionable(tmp_path)
    spxu = Stock("SPXU", "SMART", "USD")
    spxu.conId = 533_620_647
    client = _LiveClient(spyu, spxu)
    client.cash_observed_at -= timedelta(seconds=91)

    with pytest.raises(ValueError, match="fresh broker cash-account state"):
        asyncio.run(
            advance_xsp_v2_live_transport(
                LiveCalibrationLedger(tmp_path / "stale-cash.jsonl"),
                client=client,
                selection=selection,
                source_receipt=_live_source(
                    None,
                    recorded_at=OBSERVED_AT - timedelta(seconds=30),
                ),
                observed_at=OBSERVED_AT,
                quote_wait_seconds=0,
            )
        )


def test_live_recurrence_executes_one_fresh_post_selection_up_leg(
    tmp_path: Path,
) -> None:
    selection, _plan, spyu, _ticker = _actionable(tmp_path)
    spxu = Stock("SPXU", "SMART", "USD")
    spxu.conId = 533_620_647
    client = _LiveClient(spyu, spxu)
    ledger = LiveCalibrationLedger(tmp_path / "live.jsonl")
    position = {
        "lane": "rth",
        "direction": "up",
        "entry_time": (SELECTED_AT + timedelta(minutes=2)).isoformat(),
        "trading_date": "2026-07-29",
        "entry_price": 750.0,
        "exit_reason": "end",
    }

    result = asyncio.run(
        advance_xsp_v2_live_transport(
            ledger,
            client=client,
            selection=selection,
            source_receipt=_live_source(
                position,
                recorded_at=OBSERVED_AT - timedelta(seconds=30),
            ),
            observed_at=OBSERVED_AT,
            quote_wait_seconds=0,
        )
    )

    assert result["status"] == "EXECUTED"
    assert result["plan"]["leg"]["symbol"] == "SPYU"
    assert result["plan"]["leg"]["quantity"] == 8
    assert result["submitted_orders"] == 1
    assert len(client.placed) == 1
    assert [row["evidence"]["phase"] for row in ledger.records()] == [
        "STATE",
        "PREPARED",
        "SUBMITTED",
        "TERMINAL",
    ]


def test_live_recurrence_liquidates_selected_sleeve_before_rth_close(
    tmp_path: Path,
) -> None:
    selection, _plan, spyu, _ticker = _actionable(tmp_path)
    spxu = Stock("SPXU", "SMART", "USD")
    spxu.conId = 533_620_647
    client = _LiveClient(spyu, spxu)
    client.portfolio.append(
        SimpleNamespace(
            contract=SimpleNamespace(
                symbol="SPYU",
                conId=669_475_151,
                secType="STK",
            ),
            position=8,
        )
    )
    ledger = LiveCalibrationLedger(tmp_path / "live-eod.jsonl")
    buy_time = SELECTED_AT + timedelta(minutes=2)
    buy = _fill_record(
        selection,
        exec_id="live-eod-buy",
        when=buy_time,
        symbol="SPYU",
        side="BOT",
        shares=8,
        price=30.0,
    )
    ledger.checkpoint(
        evaluation_as_of=buy_time,
        strategy_id=XSP_OPENING_EDGE_V2_VERSION,
        strategy_version=XSP_V2_TRANSPORT_EXECUTION_VERSION,
        trading_date="2026-07-29",
        session="RTH",
        status="EVALUATED",
        evidence=buy["evidence"],
        recorded_at=buy_time,
    )
    observed_at = datetime(2026, 7, 29, 19, 57, tzinfo=timezone.utc)
    position = {
        "lane": "rth",
        "direction": "up",
        "entry_time": buy_time.isoformat(),
        "trading_date": "2026-07-29",
        "entry_price": 750.0,
        "exit_reason": "end",
    }

    result = asyncio.run(
        advance_xsp_v2_live_transport(
            ledger,
            client=client,
            selection=selection,
            source_receipt=_live_source(
                position,
                recorded_at=observed_at - timedelta(seconds=30),
            ),
            observed_at=observed_at,
            quote_wait_seconds=0,
        )
    )

    assert result["status"] == "EXECUTED"
    assert result["plan"]["reason"] == "rth_end_liquidation"
    assert result["plan"]["leg"]["action"] == "SELL"
    assert result["plan"]["leg"]["outside_rth"] is False
    assert client.placed[0][1:3] == ("SELL", 8)
    assert client.placed[0][-2] is False
