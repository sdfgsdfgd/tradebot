from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
import sys
from types import SimpleNamespace
import types

from tradebot.option_package import option_package_debit_value

from ib_insync import Bag, Option, Stock
import pytest

from tradebot.backtest.models import OptionLeg
import tradebot.backtest.engine as backtest_engine
from tradebot.backtest.synth import IVSurfaceParams


_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

import tradebot.ui.bot as bot_module
import tradebot.ui.bot_engine_runtime as bot_engine_runtime_module
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
        bot_module,
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
    assert staged.quantity == 1
    assert staged.limit_price == pytest.approx(-1.00)
    assert [
        (int(leg.conId), int(leg.ratio), str(leg.action), str(leg.exchange))
        for leg in staged.order_contract.comboLegs
    ] == [
        (1001, 1, "SELL", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    assert len(calls) == 4


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
            OptionLeg(action="SELL", right="PUT", strike=100.0, qty=1),
            OptionLeg(action="BUY", right="PUT", strike=95.0, qty=1),
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
        False,
        mode="entry",
        calibration=None,
    )

    assert value == pytest.approx(0.95)
    assert calls == [[("SELL", 1, 2.0), ("BUY", 1, 1.0)]]


def test_expiry_payoff_consumes_canonical_debit_value_without_inversion(monkeypatch) -> None:
    calls: list[list[tuple[str, int, float | None]]] = []
    monkeypatch.setattr(
        backtest_engine,
        "option_package_debit_value",
        _recording_kernel(calls),
    )
    legs = [
        OptionLeg(action="SELL", right="CALL", strike=100.0, qty=1),
        OptionLeg(action="BUY", right="CALL", strike=105.0, qty=1),
    ]

    payoff = backtest_engine._payoff_at_expiry(legs, 110.0)

    assert payoff == pytest.approx(-5.0)
    assert calls == [[("SELL", 1, 10.0), ("BUY", 1, 5.0)]]
