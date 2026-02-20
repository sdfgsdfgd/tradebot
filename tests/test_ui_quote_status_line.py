from __future__ import annotations

from time import monotonic
from types import SimpleNamespace

from tradebot.ui.common import _market_data_label, _quote_status_line


def test_quote_status_line_includes_fallback_source_and_non_entitlement_code() -> None:
    ticker = SimpleNamespace(
        bid=5016.5,
        ask=5017.25,
        last=5017.0,
        tbQuoteSource="delayed-snapshot",
        tbQuoteAsOf="2026-02-19T13:45:00",
        tbQuoteUpdatedMono=monotonic() - 5.0,
        tbTopQuoteUpdatedMono=monotonic() - 2.0,
        tbTopQuoteMoveCount=7,
        tbQuoteErrorCode=201,
    )

    text = _quote_status_line(ticker).plain

    assert "src delayed-snapshot" in text
    assert "asof 13:45:00" in text
    assert "code 201" in text
    assert "age " in text
    assert "topchg " in text
    assert "moves 7" in text


def test_quote_status_line_collapses_close_only_source_and_hides_10090_with_close() -> None:
    ticker = SimpleNamespace(
        bid=None,
        ask=None,
        last=None,
        close=4.93,
        prevLast=4.93,
        tbQuoteSource="delayed-snapshot",
        tbQuoteUpdatedMono=monotonic(),
        tbQuoteErrorCode=10090,
    )

    text = _quote_status_line(ticker).plain

    assert "src close-only" in text
    assert "code 10090" not in text


def test_quote_status_line_keeps_10090_when_no_quote_or_close() -> None:
    ticker = SimpleNamespace(
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        tbQuoteSource="unavailable",
        tbQuoteErrorCode=10090,
    )

    text = _quote_status_line(ticker).plain

    assert "src unavailable" in text
    assert "code 10090" in text


def test_quote_status_line_without_fallback_metadata_remains_simple() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None)

    text = _quote_status_line(ticker).plain

    assert text.startswith("MD Quotes: bid/ask n/a")
    assert "src " not in text
    assert "topchg " not in text


def test_market_data_label_distinguishes_delayed_frozen() -> None:
    assert _market_data_label(SimpleNamespace(marketDataType=1)) == "Live"
    assert _market_data_label(SimpleNamespace(marketDataType=2)) == "Live-Frozen"
    assert _market_data_label(SimpleNamespace(marketDataType=3)) == "Delayed"
    assert _market_data_label(SimpleNamespace(marketDataType=4)) == "Delayed-Frozen"
