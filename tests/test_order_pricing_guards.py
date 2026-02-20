from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

from tradebot.ui.common import _limit_price_for_mode, _quote_health, _sanitize_nbbo, _tick_size, _ticker_line


@pytest.mark.parametrize("action", ["BUY", "SELL"])
@pytest.mark.parametrize("mode", ["OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"])
def test_limit_price_ignores_negative_quote_sentinels(action: str, mode: str) -> None:
    assert _limit_price_for_mode(-1.0, -1.0, -1.0, action=action, mode=mode) is None


def test_limit_price_uses_positive_last_when_nbbo_missing() -> None:
    assert _limit_price_for_mode(-1.0, -1.0, 70.25, action="BUY", mode="MID") == 70.25


def test_limit_price_unchanged_for_valid_cross_quote() -> None:
    assert _limit_price_for_mode(70.0, 70.2, 70.1, action="BUY", mode="CROSS") == 70.2


def test_sanitize_nbbo_rejects_non_positive_values() -> None:
    assert _sanitize_nbbo(-1.0, 0.0, float("nan")) == (None, None, None)


def test_quote_health_flags_actionable_nbbo() -> None:
    out = _quote_health(bid=70.0, ask=70.2, last=None, close=69.5)
    assert out["has_nbbo"] is True
    assert out["has_actionable"] is True
    assert out["has_close_only"] is False


def test_quote_health_detects_close_only_mode() -> None:
    out = _quote_health(bid=-1.0, ask=-1.0, last=None, close=69.5)
    assert out["has_nbbo"] is False
    assert out["has_last"] is False
    assert out["has_close_only"] is True
    assert out["has_actionable"] is False


def test_quote_health_detects_one_sided_quote() -> None:
    out = _quote_health(bid=3.32, ask=None, last=None, close=2.74)
    assert out["has_bid"] is True
    assert out["has_ask"] is False
    assert out["has_one_sided"] is True
    assert out["has_nbbo"] is False
    assert out["has_actionable"] is False


def test_ticker_line_can_use_display_fallback_price() -> None:
    class _Ticker:
        bid = None
        ask = None
        last = None
        close = None
        prevLast = None
        marketDataType = 1

        @staticmethod
        def marketPrice() -> float:
            return 123.45

    text = _ticker_line(
        ("NQ",),
        {"NQ": "NASDAQ"},
        {"NQ": _Ticker()},
        None,
        "",
        allow_display_fallback=True,
    )
    assert "123.45" in text.plain


def test_tick_size_uses_market_rule_price_ladder() -> None:
    contract = types.SimpleNamespace(
        secType="FOP",
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05), (5.0, 0.25), (100.0, 0.5)),
    )
    assert _tick_size(contract, None, 4.9) == 0.05
    assert _tick_size(contract, None, 90.0) == 0.25
    assert _tick_size(contract, None, 101.3) == 0.5


def test_tick_size_prefers_ticker_ladder_over_contract_defaults() -> None:
    contract = types.SimpleNamespace(
        secType="FOP",
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05),),
    )
    ticker = types.SimpleNamespace(
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05), (5.0, 0.25), (100.0, 0.5)),
    )
    assert _tick_size(contract, ticker, 101.3) == 0.5
