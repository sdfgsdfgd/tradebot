from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradebot.ui.common import _pct24_72_from_price


def test_pct24_uses_latest_close_when_actionable_quote_exists() -> None:
    ticker = SimpleNamespace(bid=101.0, ask=103.0, last=None, close=100.0, prevLast=100.0)
    pct24, pct72 = _pct24_72_from_price(
        price=102.0,
        ticker=ticker,
        session_prev_close=99.5,
        session_prev_close_1ago=98.0,
        session_close_3ago=96.0,
    )
    assert pct24 == pytest.approx(2.0)
    assert pct72 == pytest.approx(((102.0 - 96.0) / 96.0) * 100.0)


def test_pct24_uses_prev1_close_when_quote_is_close_only() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None, close=67_000.0, prevLast=67_000.0)
    pct24, pct72 = _pct24_72_from_price(
        price=67_000.0,
        ticker=ticker,
        session_prev_close=67_000.0,
        session_prev_close_1ago=65_500.0,
        session_close_3ago=64_500.0,
    )
    assert pct24 == pytest.approx(((67_000.0 - 65_500.0) / 65_500.0) * 100.0)
    assert pct72 == pytest.approx(((67_000.0 - 64_500.0) / 64_500.0) * 100.0)


def test_pct24_is_none_when_no_actionable_quote_and_no_prev1_baseline() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None, close=67_000.0, prevLast=67_000.0)
    pct24, pct72 = _pct24_72_from_price(
        price=67_000.0,
        ticker=ticker,
        session_prev_close=67_000.0,
        session_prev_close_1ago=None,
        session_close_3ago=64_500.0,
    )
    assert pct24 is None
    assert pct72 == pytest.approx(((67_000.0 - 64_500.0) / 64_500.0) * 100.0)
