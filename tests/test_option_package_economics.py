from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
import sys
from types import SimpleNamespace
import types

from tradebot.option_package import (
    OptionPackage,
    ResolvedOptionLeg,
    option_package_debit_value,
    option_package_risk,
    option_product_facts,
)

from ib_insync import Bag, Option, Stock
import pytest

from tradebot.backtest.models import OptionTrade
import tradebot.backtest.engine_options as backtest_engine
from tradebot.backtest.synth import IVSurfaceParams


_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

import tradebot.ui.bot_engine_runtime as bot_engine_runtime_module
import tradebot.ui.bot_screen.orders as bot_orders_module
import tradebot.ui.bot_order_builder as bot_order_builder_module
from tradebot.ui.bot import BotScreen
from tradebot.ui.bot_engine_runtime import BotEngineRuntimeMixin
from tradebot.ui.bot_models import _BotInstance, _BotLegOrder, _BotOrder
from tradebot.ui.bot_order_builder import BotOrderBuilderMixin


def _recording_kernel(calls: list[list[tuple[str, int, float | None]]]):
    def _record(rows) -> float | None:
        materialized = [
            (str(action), int(ratio), None if value is None else float(value))
            for action, ratio, value in rows
        ]
        calls.append(materialized)
        return option_package_debit_value(materialized)

    return _record


def _resolved_leg(
    *,
    action: str,
    right: str,
    strike: float,
    ratio: int = 1,
    expiry: str = "20260724",
) -> ResolvedOptionLeg:
    return ResolvedOptionLeg(action, right, strike, ratio, expiry)


def test_option_package_debit_value_credit_debit_and_ratios() -> None:
    assert option_package_debit_value(
        [("SELL", 1, 2.00), ("BUY", 1, 1.25)]
    ) == pytest.approx(-0.75)
    assert option_package_debit_value(
        [("BUY", 2, 0.40), ("SELL", 1, 1.10)]
    ) == pytest.approx(-0.30)
    assert option_package_debit_value(
        [("SELL", 3, 0.75), ("BUY", 1, 0.20)]
    ) == pytest.approx(-2.05)


def test_option_package_debit_value_missing_empty_and_invalid_action() -> None:
    assert option_package_debit_value([]) == 0.0
    assert option_package_debit_value(
        [("BUY", 1, None), ("SELL", 1, 1.0)]
    ) is None
    with pytest.raises(ValueError, match="unsupported option leg action"):
        option_package_debit_value([("OPEN", 1, 1.0)])


def _option_order_fixture() -> tuple[_BotOrder, dict[int, SimpleNamespace]]:
    short = Option(
        symbol="XSP",
        lastTradeDateOrContractMonth="20260209",
        strike=100.0,
        right="P",
        exchange="SMART",
        currency="USD",
    )
    short.conId = 1001
    short.minTick = 0.01
    long = Option(
        symbol="XSP",
        lastTradeDateOrContractMonth="20260209",
        strike=95.0,
        right="P",
        exchange="SMART",
        currency="USD",
    )
    long.conId = 1002
    long.minTick = 0.01
    underlying = Stock(symbol="XSP", exchange="SMART", currency="USD")
    underlying.conId = 900
    bag = Bag(symbol="XSP", exchange="SMART", currency="USD", comboLegs=[])
    order = _BotOrder(
        instance_id=1,
        preset=None,
        underlying=underlying,
        order_contract=bag,
        legs=[
            _BotLegOrder(contract=short, action="SELL", ratio=1),
            _BotLegOrder(contract=long, action="BUY", ratio=1),
        ],
        action="BUY",
        quantity=1,
        limit_price=-0.50,
        created_at=datetime(2026, 2, 9, 10, 0),
        exec_mode="OPTIMISTIC",
    )
    tickers = {
        1001: SimpleNamespace(
            contract=short,
            bid=2.00,
            ask=2.20,
            last=2.10,
            minTick=0.01,
        ),
        1002: SimpleNamespace(
            contract=long,
            bid=1.00,
            ask=1.20,
            last=1.10,
            minTick=0.01,
        ),
    }
    return order, tickers


def test_live_combo_reprice_consumes_canonical_package_kernel(monkeypatch) -> None:
    order, tickers = _option_order_fixture()
    calls: list[list[tuple[str, int, float | None]]] = []
    monkeypatch.setattr(
        bot_orders_module,
        "option_package_debit_value",
        _recording_kernel(calls),
    )

    class _Client:
        @staticmethod
        async def ensure_ticker(contract, *, owner: str):
            assert owner == "bot"
            return tickers[int(contract.conId)]

    harness = SimpleNamespace(_client=_Client())
    changed = asyncio.run(BotScreen._reprice_order(harness, order, mode="MID"))

    assert changed is True
    assert order.action == "BUY"
    assert order.limit_price == pytest.approx(-1.00)
    assert (order.bid, order.ask, order.last) == pytest.approx((-1.00, -1.00, -1.00))
    assert len(calls) == 4
    assert all([(row[0], row[1]) for row in call] == [("SELL", 1), ("BUY", 1)] for call in calls)


def test_live_combo_quote_signature_consumes_canonical_package_kernel(monkeypatch) -> None:
    order, tickers = _option_order_fixture()
    calls: list[list[tuple[str, int, float | None]]] = []
    monkeypatch.setattr(
        bot_engine_runtime_module,
        "option_package_debit_value",
        _recording_kernel(calls),
    )

    class _Client:
        @staticmethod
        def ticker_for_con_id(con_id: int):
            return tickers[int(con_id)]

    harness = SimpleNamespace(_client=_Client())
    signature = BotEngineRuntimeMixin._order_quote_signature(harness, order)

    assert signature == pytest.approx((-1.00, -1.00, -1.00))
    assert len(calls) == 3
    assert all([(row[0], row[1]) for row in call] == [("SELL", 1), ("BUY", 1)] for call in calls)


def test_option_builder_stages_native_credit_bag_through_canonical_kernel(monkeypatch) -> None:
    calls: list[list[tuple[str, int, float | None]]] = []
    monkeypatch.setattr(
        bot_order_builder_module,
        "option_package_debit_value",
        _recording_kernel(calls),
    )
    monkeypatch.setattr(
        bot_order_builder_module,
        "_now_et",
        lambda: datetime(2026, 2, 9, 10, 0),
    )

    underlying = Stock(symbol="XSP", exchange="SMART", currency="USD")
    underlying.conId = 900

    class _Client:
        @staticmethod
        async def stock_option_chain(symbol: str):
            assert symbol == "XSP"
            return underlying, SimpleNamespace(
                expirations=["20260209"],
                strikes=[95.0, 100.0],
                tradingClass="XSP",
            )

        @staticmethod
        async def qualify_proxy_contracts(*contracts):
            for contract in contracts:
                contract.conId = 1001 if float(contract.strike) == 100.0 else 1002
                contract.minTick = 0.01
                contract.multiplier = "100"
            return list(contracts)

        @staticmethod
        async def ensure_ticker(contract, *, owner: str):
            assert owner == "bot"
            if str(contract.secType).upper() == "STK":
                return SimpleNamespace(
                    contract=contract,
                    bid=99.90,
                    ask=100.10,
                    last=100.00,
                    minTick=0.01,
                    marketDataType=1,
                    marketPrice=lambda: 100.00,
                )
            if float(contract.strike) == 100.0:
                bid, ask, last = 2.00, 2.20, 2.10
            else:
                bid, ask, last = 1.00, 1.20, 1.10
            return SimpleNamespace(
                contract=contract,
                bid=bid,
                ask=ask,
                last=last,
                minTick=0.01,
                marketDataType=1,
            )

        @staticmethod
        def proxy_error():
            return None

    class _Harness(BotOrderBuilderMixin):
        def __init__(self) -> None:
            self._client = _Client()
            self._payload = None
            self._tracked_conids: set[int] = set()
            self._orders: list[_BotOrder] = []
            self._status = ""

        @staticmethod
        def _strategy_instrument(_strategy) -> str:
            return "options"

        @staticmethod
        def _initial_exec_mode(**_kwargs) -> str:
            return "MID"

        @staticmethod
        def _reset_daily_counters_if_needed(_instance) -> None:
            return None

        @staticmethod
        def _order_reservation_summary():
            from tradebot.order_reservation import OrderReservationSummary

            return OrderReservationSummary(
                account="DU2200",
                active_count=0,
                unknown_active_count=0,
                reserved_max_loss=0.0,
                complete=True,
                reason="reservation_complete",
            )

        def _add_order(self, order: _BotOrder) -> None:
            self._orders.append(order)

        def _render_status(self) -> None:
            return None

        @staticmethod
        def _journal_write(**_kwargs) -> None:
            return None

    instance = _BotInstance(
        instance_id=1,
        group="xsp-credit",
        symbol="XSP",
        strategy={
            "instrument": "options",
            "entry_signal": "ema",
            "ema_preset": "9/21",
            "dte": 0,
            "quantity": 2,
            "xsp_reservation_capacity_usd": 2200.0,
            "legs": [
                {"action": "SELL", "right": "PUT", "moneyness_pct": 0.0, "qty": 1},
                {"action": "BUY", "right": "PUT", "moneyness_pct": 5.0, "qty": 1},
            ],
        },
        filters=None,
    )
    harness = _Harness()

    asyncio.run(
        harness._create_order_for_instance(
            instance,
            intent="enter",
            direction="up",
            signal_bar_ts=datetime(2026, 2, 9, 10, 0),
        )
    )

    assert len(harness._orders) == 1
    staged = harness._orders[0]
    assert isinstance(staged.order_contract, Bag)
    assert staged.action == "BUY"
    assert staged.quantity == 2
    assert staged.limit_price == pytest.approx(-1.00)
    assert [
        (int(leg.conId), int(leg.ratio), str(leg.action), str(leg.exchange))
        for leg in staged.order_contract.comboLegs
    ] == [
        (1001, 1, "SELL", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    package_risk = getattr(staged, "package_risk", None)
    assert package_risk is not None, "_BotOrder.package_risk is missing"
    assert package_risk.as_payload() == {
        "structure": "vertical_credit",
        "right": "PUT",
        "expiry": "20260209",
        "width": 5.0,
        "debit_value": -1.0,
        "multiplier": 100.0,
        "quantity": 2,
        "max_loss": 800.0,
    }
    assert len(calls) == 4


@pytest.mark.parametrize("unsupported_intent", ["roll", "rescue"])
def test_option_builder_rejects_unsupported_transition_intent_before_mutation(
    monkeypatch,
    unsupported_intent: str,
) -> None:
    monkeypatch.setattr(
        bot_order_builder_module,
        "_now_et",
        lambda: datetime(2026, 2, 9, 10, 0),
    )

    underlying = Stock(symbol="XSP", exchange="SMART", currency="USD")
    underlying.conId = 900

    class _Client:
        def __init__(self) -> None:
            self.stock_option_chain_calls = 0
            self.qualify_calls = 0
            self.ensure_ticker_calls = 0

        async def stock_option_chain(self, symbol: str):
            self.stock_option_chain_calls += 1
            assert symbol == "XSP"
            return underlying, SimpleNamespace(
                expirations=["20260209"],
                strikes=[95.0, 100.0],
                tradingClass="XSP",
            )

        async def qualify_proxy_contracts(self, *contracts):
            self.qualify_calls += 1
            for contract in contracts:
                contract.conId = 1001 if float(contract.strike) == 100.0 else 1002
                contract.minTick = 0.01
                contract.multiplier = "100"
            return list(contracts)

        async def ensure_ticker(self, contract, *, owner: str):
            self.ensure_ticker_calls += 1
            assert owner == "bot"
            if str(contract.secType).upper() == "STK":
                return SimpleNamespace(
                    contract=contract,
                    bid=99.90,
                    ask=100.10,
                    last=100.00,
                    minTick=0.01,
                    marketDataType=1,
                    marketPrice=lambda: 100.00,
                )
            if float(contract.strike) == 100.0:
                bid, ask, last = 2.00, 2.20, 2.10
            else:
                bid, ask, last = 1.00, 1.20, 1.10
            return SimpleNamespace(
                contract=contract,
                bid=bid,
                ask=ask,
                last=last,
                minTick=0.01,
                marketDataType=1,
            )

        @staticmethod
        def proxy_error():
            return None

    class _Harness(BotOrderBuilderMixin):
        def __init__(self) -> None:
            self._client = _Client()
            self._payload = None
            self._tracked_conids: set[int] = set()
            self._orders: list[_BotOrder] = []
            self._events: list[tuple[str, str | None, dict[str, object]]] = []
            self._status = ""
            self.reservation_summary_calls = 0
            self.add_order_calls = 0

        @staticmethod
        def _strategy_instrument(_strategy) -> str:
            return "options"

        @staticmethod
        def _initial_exec_mode(**_kwargs) -> str:
            return "MID"

        @staticmethod
        def _reset_daily_counters_if_needed(_instance) -> None:
            return None

        def _order_reservation_summary(self):
            from tradebot.order_reservation import OrderReservationSummary

            self.reservation_summary_calls += 1
            return OrderReservationSummary(
                account="DU2200",
                active_count=0,
                unknown_active_count=0,
                reserved_max_loss=0.0,
                complete=True,
                reason="reservation_complete",
            )

        def _add_order(self, order: _BotOrder) -> None:
            self.add_order_calls += 1
            self._orders.append(order)

        def _render_status(self) -> None:
            return None

        def _journal_write(
            self,
            *,
            event: str,
            reason: str | None = None,
            data: dict | None = None,
            **_kwargs,
        ) -> None:
            self._events.append((str(event), reason, dict(data or {})))

    signal_bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _BotInstance(
        instance_id=1,
        group="xsp-credit",
        symbol="XSP",
        strategy={
            "instrument": "options",
            "entry_signal": "ema",
            "ema_preset": "9/21",
            "dte": 0,
            "xsp_reservation_capacity_usd": 2200.0,
            "legs": [
                {"action": "SELL", "right": "PUT", "moneyness_pct": 0.0, "qty": 1},
                {"action": "BUY", "right": "PUT", "moneyness_pct": 5.0, "qty": 1},
            ],
        },
        filters=None,
    )
    harness = _Harness()

    asyncio.run(
        harness._create_order_for_instance(
            instance,
            intent=unsupported_intent,
            direction="up",
            signal_bar_ts=signal_bar_ts,
        )
    )

    assert harness._orders == []
    assert harness.add_order_calls == 0
    assert harness._client.stock_option_chain_calls == 0
    assert harness._client.qualify_calls == 0
    assert harness._client.ensure_ticker_calls == 0
    assert harness.reservation_summary_calls == 0
    assert harness._tracked_conids == set()
    assert instance.touched_conids == set()
    assert instance.entries_today == 0
    assert instance.open_direction is None
    assert instance.last_entry_bar_ts is None
    assert harness._events == [
        (
            "ORDER_SKIPPED",
            unsupported_intent,
            {
                "skip_reason": "intent_unsupported",
                "intent": unsupported_intent,
                "symbol": "XSP",
            },
        )
    ]


_MISSING_XSP_RESERVATION_CAPACITY = object()


def _run_xsp_capacity_builder_case(
    monkeypatch,
    *,
    available_capacity=_MISSING_XSP_RESERVATION_CAPACITY,
    reserved_max_loss: float = 650.0,
    min_credit: float | None = None,
    events: list[str] | None = None,
):
    from tradebot.order_reservation import (
        OrderReservationSummary,
        evaluate_order_reservation_capacity,
    )

    package_calls: list[list[tuple[str, int, float | None]]] = []
    capacity_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        bot_order_builder_module,
        "option_package_debit_value",
        _recording_kernel(package_calls),
    )
    monkeypatch.setattr(
        bot_order_builder_module,
        "_now_et",
        lambda: datetime(2026, 2, 9, 10, 0),
    )

    def _record_capacity(request, summary):
        capacity_calls.append((request, summary))
        return evaluate_order_reservation_capacity(request, summary)

    monkeypatch.setattr(
        bot_order_builder_module,
        "evaluate_order_reservation_capacity",
        _record_capacity,
        raising=False,
    )

    underlying = Stock(symbol="XSP", exchange="SMART", currency="USD")
    underlying.conId = 900

    class _Client:
        @staticmethod
        async def stock_option_chain(symbol: str):
            if events is not None:
                events.append("stock_option_chain")
            assert symbol == "XSP"
            return underlying, SimpleNamespace(
                expirations=["20260209"],
                strikes=[95.0, 100.0],
                tradingClass="XSP",
            )

        @staticmethod
        async def qualify_proxy_contracts(*contracts):
            for contract in contracts:
                contract.conId = 1001 if float(contract.strike) == 100.0 else 1002
                contract.minTick = 0.01
                contract.multiplier = "100"
            return list(contracts)

        @staticmethod
        async def ensure_ticker(contract, *, owner: str):
            assert owner == "bot"
            if str(contract.secType).upper() == "STK":
                return SimpleNamespace(
                    contract=contract,
                    bid=99.90,
                    ask=100.10,
                    last=100.00,
                    minTick=0.01,
                    marketDataType=1,
                    marketPrice=lambda: 100.00,
                )
            if float(contract.strike) == 100.0:
                bid, ask, last = 2.00, 2.20, 2.10
            else:
                bid, ask, last = 1.00, 1.20, 1.10
            return SimpleNamespace(
                contract=contract,
                bid=bid,
                ask=ask,
                last=last,
                minTick=0.01,
                marketDataType=1,
            )

        @staticmethod
        def proxy_error():
            return None

    summary = OrderReservationSummary(
        account="DU2200",
        active_count=2,
        unknown_active_count=0,
        reserved_max_loss=reserved_max_loss,
        complete=True,
        reason="reservation_complete",
    )

    class _Harness(BotOrderBuilderMixin):
        def __init__(self) -> None:
            self._client = _Client()
            self._payload = None
            self._tracked_conids: set[int] = set()
            self._orders: list[_BotOrder] = []
            self._status = ""
            self._events: list[dict[str, object]] = []

        @staticmethod
        def _strategy_instrument(_strategy) -> str:
            return "options"

        @staticmethod
        def _initial_exec_mode(**_kwargs) -> str:
            return "MID"

        @staticmethod
        def _reset_daily_counters_if_needed(_instance) -> None:
            return None

        @staticmethod
        def _order_reservation_summary():
            return summary

        def _add_order(self, order: _BotOrder) -> None:
            self._orders.append(order)

        def _render_status(self) -> None:
            return None

        def _journal_write(self, **kwargs) -> None:
            self._events.append(dict(kwargs))

    strategy = {
        "instrument": "options",
        "entry_signal": "ema",
        "ema_preset": "9/21",
        "dte": 0,
        "legs": [
            {"action": "SELL", "right": "PUT", "moneyness_pct": 0.0, "qty": 1},
            {"action": "BUY", "right": "PUT", "moneyness_pct": 5.0, "qty": 1},
        ],
    }
    if available_capacity is not _MISSING_XSP_RESERVATION_CAPACITY:
        strategy["xsp_reservation_capacity_usd"] = available_capacity
    if min_credit is not None:
        strategy["min_credit"] = min_credit

    instance = _BotInstance(
        instance_id=1,
        group="xsp-credit",
        symbol="XSP",
        strategy=strategy,
        filters=None,
    )
    harness = _Harness()
    signal_bar_ts = datetime(2026, 2, 9, 10, 0)

    asyncio.run(
        harness._create_order_for_instance(
            instance,
            intent="enter",
            direction="up",
            signal_bar_ts=signal_bar_ts,
        )
    )

    return harness, instance, signal_bar_ts, package_calls, capacity_calls


def test_option_package_entry_intent_is_frozen_strict_and_source_agnostic() -> None:
    from dataclasses import FrozenInstanceError

    from tradebot import option_package

    assert hasattr(option_package, "OptionPackageEntryIntent")
    assert hasattr(option_package, "option_package_entry_intent")

    intent_type = option_package.OptionPackageEntryIntent
    project = option_package.option_package_entry_intent
    typed_legs = (
        option_package.LegConfig(
            action="SELL",
            right="PUT",
            moneyness_pct=0.0,
            qty=1,
        ),
        option_package.LegConfig(
            action="BUY",
            right="PUT",
            moneyness_pct=5.0,
            qty=1,
            delta=-0.15,
        ),
    )
    raw_legs = [
        {
            "action": " sell ",
            "right": " put ",
            "moneyness_pct": 0.0,
            "qty": 1,
        },
        {
            "action": "buy",
            "right": "put",
            "moneyness_pct": 5.0,
            "qty": 1,
            "delta": -0.15,
        },
    ]

    typed_source = SimpleNamespace(
        legs=typed_legs,
        dte=0,
        quantity=2,
        min_credit=1.50,
    )
    mapping_source = {
        "legs": raw_legs,
        "dte": "0",
        "quantity": "2",
        "min_credit": "1.50",
    }
    expected = intent_type(
        legs=typed_legs,
        dte=0,
        quantity=2,
        min_credit=1.50,
    )

    assert project(typed_source) == expected
    assert project(mapping_source) == expected
    assert project(
        {
            "legs": [],
            "dte": "0",
            "quantity": "2",
            "min_credit": "1.50",
        },
        legs=raw_legs,
        path="directional_legs.up",
    ) == expected

    defaults = project({"legs": raw_legs})
    assert defaults.dte == 0
    assert defaults.quantity == 1
    assert defaults.min_credit is None

    with pytest.raises((FrozenInstanceError, AttributeError)):
        defaults.quantity = 3

    invalid_values = (
        ("dte", None),
        ("dte", True),
        ("dte", -1),
        ("dte", 2.5),
        ("dte", "bad"),
        ("quantity", None),
        ("quantity", True),
        ("quantity", 0),
        ("quantity", -1),
        ("quantity", 2.5),
        ("quantity", "bad"),
        ("min_credit", True),
        ("min_credit", -0.01),
        ("min_credit", float("nan")),
        ("min_credit", float("inf")),
        ("min_credit", "bad"),
    )
    for field, value in invalid_values:
        invalid = dict(mapping_source)
        invalid[field] = value
        with pytest.raises(ValueError, match=field):
            project(invalid)

    with pytest.raises(ValueError, match="legs"):
        project({"dte": 0, "quantity": 1, "min_credit": None})


def test_backtest_and_live_option_entries_delegate_to_canonical_intent_before_resolution(
    monkeypatch,
) -> None:
    import tradebot.backtest.strategy as backtest_strategy

    from tradebot.option_package import LegConfig

    canonical_legs = (
        LegConfig(action="SELL", right="PUT", moneyness_pct=0.0, qty=1),
        LegConfig(action="BUY", right="PUT", moneyness_pct=5.0, qty=1),
    )
    projected = SimpleNamespace(
        legs=canonical_legs,
        dte=0,
        quantity=2,
        min_credit=None,
    )

    backtest_events: list[tuple] = []

    def _backtest_projector(_strategy, *, legs=None, path="legs"):
        backtest_events.append(("intent", legs, path))
        return projected

    def _record_expiry(_trade_date, dte):
        backtest_events.append(("expiry", dte))
        return date(2026, 2, 9)

    def _record_legs(legs, _spot, quantity):
        backtest_events.append(("legs", legs, quantity))
        return (
            _resolved_leg(action="SELL", right="PUT", strike=100.0, expiry="20260209"),
            _resolved_leg(action="BUY", right="PUT", strike=95.0, expiry="20260209"),
        )

    monkeypatch.setattr(
        backtest_strategy,
        "option_package_entry_intent",
        _backtest_projector,
        raising=False,
    )
    monkeypatch.setattr(backtest_strategy, "_expiry_from_dte", _record_expiry)
    monkeypatch.setattr(backtest_strategy, "_build_legs", _record_legs)

    cfg = SimpleNamespace(
        dte=-99,
        quantity=0,
        min_credit=float("nan"),
        legs=canonical_legs,
        right="PUT",
        otm_pct=0.0,
        width_pct=5.0,
        entry_days=(0, 1, 2, 3, 4),
    )
    spec = backtest_strategy.OptionPackageStrategy(cfg).build_spec(
        datetime(2026, 2, 9, 10, 0),
        100.0,
        legs_override=canonical_legs,
    )

    live_events: list[str] = []

    def _live_projector(_strategy, *, legs=None, path="legs"):
        live_events.append("intent")
        assert path == "legs"
        assert legs is not None
        return projected

    monkeypatch.setattr(
        bot_order_builder_module,
        "option_package_entry_intent",
        _live_projector,
        raising=False,
    )
    harness, _instance, _signal_bar_ts, _package_calls, _capacity_calls = (
        _run_xsp_capacity_builder_case(
            monkeypatch,
            available_capacity=2200.0,
            events=live_events,
        )
    )

    assert {
        "backtest_events": backtest_events,
        "backtest_leg_ratios": [leg.ratio for leg in spec.legs],
        "backtest_quantity": spec.quantity,
        "live_events": live_events,
        "live_order_quantity": harness._orders[0].quantity if harness._orders else None,
    } == {
        "backtest_events": [
            ("intent", canonical_legs, "legs_override"),
            ("expiry", 0),
            ("legs", canonical_legs, date(2026, 2, 9)),
        ],
        "backtest_leg_ratios": [1, 1],
        "backtest_quantity": 2,
        "live_events": ["intent", "stock_option_chain"],
        "live_order_quantity": 2,
    }


def test_option_builder_consumes_explicit_xsp_reservation_capacity_before_staging(
    monkeypatch,
) -> None:
    harness, instance, signal_bar_ts, package_calls, capacity_calls = (
        _run_xsp_capacity_builder_case(
            monkeypatch,
            available_capacity=2200.0,
            reserved_max_loss=650.0,
        )
    )

    assert len(capacity_calls) == 1, "native XSP builder did not evaluate reservation capacity"
    request, summary = capacity_calls[0]
    assert request.account == "DU2200"
    assert request.product_domain == "XSP"
    assert request.sec_type == "BAG"
    assert request.structure == "vertical_credit"
    assert request.candidate_max_loss == 400.0
    assert request.available_capacity == 2200.0
    assert summary.reserved_max_loss == 650.0
    assert len(harness._orders) == 1
    assert instance.touched_conids == {1001, 1002}
    assert instance.entries_today == 1
    assert instance.open_direction == "up"
    assert instance.last_entry_bar_ts == signal_bar_ts
    assert "Created order BUY BAG XSP" in harness._status
    assert len(package_calls) == 4


def test_option_builder_blocks_credit_below_strategy_minimum_before_capacity_or_staging(
    monkeypatch,
) -> None:
    harness, instance, signal_bar_ts, package_calls, capacity_calls = (
        _run_xsp_capacity_builder_case(
            monkeypatch,
            available_capacity=2200.0,
            reserved_max_loss=650.0,
            min_credit=1.50,
        )
    )

    assert capacity_calls == []
    assert harness._orders == []
    assert instance.touched_conids == set()
    assert instance.entries_today == 0
    assert instance.open_direction is None
    assert instance.last_entry_bar_ts is None
    assert harness._status == "Order: credit 1.00 below minimum 1.50"
    assert instance.order_trigger_last_error == "Order: credit 1.00 below minimum 1.50"
    assert instance.order_trigger_retry_reason == "minimum_credit_not_met"
    assert len(package_calls) == 4

    assert len(harness._events) == 1
    event = harness._events[0]
    assert event["event"] == "ORDER_BUILD_FAILED"
    assert event["instance"] is instance
    assert event["order"] is None
    assert event["reason"] == "enter"
    data = event["data"]
    assert data["error"] == "Order: credit 1.00 below minimum 1.50"
    assert data["direction"] == "up"
    assert data["signal_bar_ts"] == signal_bar_ts.isoformat()
    assert data["retry_reason"] == "minimum_credit_not_met"
    assert data["credit"] == pytest.approx(1.00)
    assert data["min_credit"] == pytest.approx(1.50)
    assert data["debit_value"] == pytest.approx(-1.00)
    assert data["tick"] == pytest.approx(0.01)


def test_option_builder_blocks_capacity_exceeded_before_any_staging_mutation(
    monkeypatch,
) -> None:
    harness, instance, _signal_bar_ts, package_calls, capacity_calls = (
        _run_xsp_capacity_builder_case(
            monkeypatch,
            available_capacity=1000.0,
            reserved_max_loss=650.0,
        )
    )

    assert len(capacity_calls) == 1, "native XSP builder did not evaluate reservation capacity"
    request, summary = capacity_calls[0]
    assert request.candidate_max_loss == 400.0
    assert request.available_capacity == 1000.0
    assert summary.reserved_max_loss == 650.0
    assert harness._orders == []
    assert instance.touched_conids == set()
    assert instance.entries_today == 0
    assert instance.open_direction is None
    assert instance.last_entry_bar_ts is None
    assert harness._status == "Capacity: capacity_exceeded"
    assert instance.order_trigger_last_error == "Capacity: capacity_exceeded"
    assert instance.order_trigger_retry_reason == "capacity_exceeded"
    assert len(package_calls) == 4


def test_option_builder_fails_closed_when_xsp_reservation_capacity_is_missing(
    monkeypatch,
) -> None:
    harness, instance, _signal_bar_ts, package_calls, capacity_calls = (
        _run_xsp_capacity_builder_case(
            monkeypatch,
            reserved_max_loss=650.0,
        )
    )

    assert len(capacity_calls) == 1, "native XSP builder did not evaluate reservation capacity"
    request, summary = capacity_calls[0]
    assert request.candidate_max_loss == 400.0
    assert request.available_capacity is None
    assert summary.reserved_max_loss == 650.0
    assert harness._orders == []
    assert instance.touched_conids == set()
    assert instance.entries_today == 0
    assert instance.open_direction is None
    assert instance.last_entry_bar_ts is None
    assert harness._status == "Capacity: capacity_unavailable"
    assert instance.order_trigger_last_error == "Capacity: capacity_unavailable"
    assert instance.order_trigger_retry_reason == "capacity_unavailable"
    assert len(package_calls) == 4


def _surface() -> IVSurfaceParams:
    return IVSurfaceParams(
        rv_lookback=20,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.0,
        iv_floor=0.01,
        term_slope=0.0,
        skew=0.0,
    )


def test_backtest_package_value_explicitly_inverts_canonical_debit_value(monkeypatch) -> None:
    calls: list[list[tuple[str, int, float | None]]] = []
    monkeypatch.setattr(
        backtest_engine,
        "option_package_debit_value",
        _recording_kernel(calls),
    )
    monkeypatch.setattr(backtest_engine, "iv_atm", lambda *_args, **_kwargs: 0.20)
    monkeypatch.setattr(backtest_engine, "iv_for_strike", lambda *_args, **_kwargs: 0.20)
    monkeypatch.setattr(
        backtest_engine,
        "black_scholes",
        lambda _forward, strike, *_args, **_kwargs: 2.0 if float(strike) == 100.0 else 1.0,
    )

    spec = SimpleNamespace(
        expiry=date(2026, 2, 9),
        legs=[
            _resolved_leg(action="SELL", right="PUT", strike=100.0, expiry="20260209"),
            _resolved_leg(action="BUY", right="PUT", strike=95.0, expiry="20260209"),
        ],
    )
    bar = SimpleNamespace(ts=datetime(2026, 2, 9, 10, 0), close=100.0)
    cfg = SimpleNamespace(
        backtest=SimpleNamespace(
            use_rth=True,
            bar_size="5 mins",
            risk_free_rate=0.0,
        ),
        synthetic=SimpleNamespace(min_spread_pct=0.0),
    )

    value = backtest_engine._trade_value_from_spec(
        spec,
        bar,
        0.20,
        cfg,
        _surface(),
        0.10,
        option_product_facts("XSP"),
        mode="entry",
        calibration=None,
    )

    assert value == pytest.approx(0.95)
    assert calls == [[("SELL", 1, 2.0), ("BUY", 1, 1.0)]]


def test_backtest_multi_quantity_credit_vertical_risk_matches_canonical_package_economics() -> None:
    quantity = 2
    multiplier = 100.0
    package = OptionPackage(
        product=option_product_facts("XSP", multiplier=multiplier),
        legs=(
            _resolved_leg(action="SELL", right="PUT", strike=100.0),
            _resolved_leg(action="BUY", right="PUT", strike=95.0),
        ),
        debit_value=-1.0,
        quantity=quantity,
    )
    canonical = option_package_risk(package)
    assert canonical is not None

    trade = OptionTrade(
        package=package,
        risk=canonical,
        entry_time=datetime(2026, 7, 23, 10, 0),
        stop_loss=0.5,
        profit_target=0.5,
        margin_required=canonical.max_loss,
    )

    assert canonical.max_loss == pytest.approx(800.0)
    assert trade.max_loss == pytest.approx(4.0)
    assert trade.margin_required == pytest.approx(canonical.max_loss)
    assert backtest_engine._hit_stop(trade, 2.9, "max_loss") is False
    assert backtest_engine._hit_stop(trade, 3.0, "max_loss") is True


def test_backtest_option_exit_thresholds_disable_when_non_positive() -> None:
    def _trade(*, profit_target: float = 0.5, stop_loss: float = 0.5) -> SimpleNamespace:
        return SimpleNamespace(
            entry_price=1.0,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_loss=4.0,
        )

    assert backtest_engine._hit_profit(_trade(), 0.5) is True
    assert backtest_engine._hit_stop(_trade(), 1.5, "credit") is True
    assert backtest_engine._hit_stop(_trade(), 3.0, "max_loss") is True

    assert backtest_engine._hit_profit(_trade(profit_target=float("nan")), 0.0) is False
    assert backtest_engine._hit_profit(_trade(profit_target=float("inf")), 0.0) is False
    assert backtest_engine._hit_stop(_trade(stop_loss=float("nan")), 10.0, "max_loss") is False
    assert backtest_engine._hit_stop(_trade(stop_loss=float("inf")), 10.0, "max_loss") is False

    for threshold in (0.0, -0.1, float("-inf")):
        assert backtest_engine._hit_profit(
            _trade(profit_target=threshold),
            0.0,
        ) is False
        assert backtest_engine._hit_stop(
            _trade(stop_loss=threshold),
            10.0,
            "credit",
        ) is False
        assert backtest_engine._hit_stop(
            _trade(stop_loss=threshold),
            10.0,
            "max_loss",
        ) is False


def test_backtest_has_no_parallel_option_risk_or_expiry_payoff_engine() -> None:
    assert not hasattr(backtest_engine, "_max_loss")
    assert not hasattr(backtest_engine, "_max_loss_estimate")
    assert not hasattr(backtest_engine, "_payoff_at_expiry")
