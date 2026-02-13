from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_common_module():
    path = Path(__file__).resolve().parents[1] / "tradebot" / "ui" / "common.py"
    spec = importlib.util.spec_from_file_location("tradebot_ui_common_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMMON = _load_common_module()
_unrealized_pnl_values = _COMMON._unrealized_pnl_values


def _item(
    *,
    sec_type: str = "OPT",
    position: float = 1.0,
    average_cost: float = 0.0,
    market_price: float = 0.0,
    market_value: float = 0.0,
    unrealized: float | None = None,
):
    contract = SimpleNamespace(secType=sec_type, multiplier="100")
    return SimpleNamespace(
        contract=contract,
        position=position,
        averageCost=average_cost,
        marketPrice=market_price,
        marketValue=market_value,
        unrealizedPNL=unrealized,
    )


def test_unrealized_values_prefer_broker_value_when_available() -> None:
    item = _item(
        sec_type="OPT",
        position=1.0,
        average_cost=144.0488,
        market_price=1.37510965,
        market_value=137.51,
        unrealized=-6.54,
    )

    unreal, pct = _unrealized_pnl_values(item, mark_price=1.01)

    assert unreal == -6.54
    assert pct is not None
    assert round(pct, 2) == -4.54


def test_unrealized_values_model_when_broker_value_missing() -> None:
    item = _item(
        sec_type="STK",
        position=2.0,
        average_cost=10.0,
        market_price=10.0,
        market_value=20.0,
        unrealized=None,
    )

    unreal, pct = _unrealized_pnl_values(item, mark_price=12.0)

    assert unreal == 4.0
    assert pct == 20.0
