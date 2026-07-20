from __future__ import annotations

from types import SimpleNamespace

import pytest
from ib_insync import Contract

from tradebot.ui.common import (
    _SyntheticPortfolioItem,
    _option_display_price,
    _pct24_72_from_price,
    _price_direction_glyph,
    _quote_age_ribbon,
)


def test_pct24_uses_latest_close_when_ticker_close_exists() -> None:
    ticker = SimpleNamespace(bid=101.0, ask=103.0, last=None, close=100.0, prevLast=100.0)
    pct24, pct72 = _pct24_72_from_price(
        price=102.0,
        ticker=ticker,
        session_prev_close=99.5,
        session_close_3ago=96.0,
    )
    assert pct24 == pytest.approx(2.0)
    assert pct72 == pytest.approx(((102.0 - 96.0) / 96.0) * 100.0)


def test_pct24_is_zero_when_price_equals_close() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None, close=67_000.0, prevLast=67_000.0)
    pct24, pct72 = _pct24_72_from_price(
        price=67_000.0,
        ticker=ticker,
        session_prev_close=67_000.0,
        session_close_3ago=64_500.0,
    )
    assert pct24 == pytest.approx(0.0)
    assert pct72 == pytest.approx(((67_000.0 - 64_500.0) / 64_500.0) * 100.0)


def test_pct24_is_none_when_close_baseline_missing() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None, close=None, prevLast=None)
    pct24, pct72 = _pct24_72_from_price(
        price=67_000.0,
        ticker=ticker,
        session_prev_close=None,
        session_close_3ago=64_500.0,
    )
    assert pct24 is None
    assert pct72 == pytest.approx(((67_000.0 - 64_500.0) / 64_500.0) * 100.0)


def test_option_display_price_ignores_snapshot_market_price_for_fop_without_quote() -> None:
    item = SimpleNamespace(contract=SimpleNamespace(secType="FOP"), marketPrice=234.0)

    assert _option_display_price(item, None) is None


def test_shared_quote_visuals_preserve_direction_and_freshness_contract() -> None:
    assert _price_direction_glyph(1.0, -2.0).plain == "▲"
    assert _price_direction_glyph(None, -2.0).plain == "▼"
    assert _price_direction_glyph(None, None).plain == "•"
    assert _quote_age_ribbon(None).plain == ""
    assert _quote_age_ribbon(0.0).plain == "▰▰▰▰"
    assert _quote_age_ribbon(10.0).plain == "▱▱▱▱"


def test_shared_synthetic_portfolio_item_is_constructible_and_mutable() -> None:
    item = _SyntheticPortfolioItem(contract=Contract(symbol="AAPL"))
    item.position = 2.0

    assert item.contract.symbol == "AAPL"
    assert item.position == 2.0
