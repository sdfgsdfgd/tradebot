from __future__ import annotations

from datetime import datetime
from time import monotonic
from types import SimpleNamespace

from tradebot.ui.common import (
    _market_data_label,
    _market_data_tag,
    _market_session_bucket,
    _quote_status_line,
)


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

    assert "SRC delayed-snapshot" in text
    assert "ASOF 13:45:00" in text
    assert "CODE 201" in text
    assert "UNCHANGED 2s" in text


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

    assert "SRC close-only" in text
    assert "CODE 10090" not in text


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

    assert "SRC unavailable" in text
    assert "CODE 10090" in text


def test_quote_status_line_without_fallback_metadata_remains_simple() -> None:
    ticker = SimpleNamespace(bid=None, ask=None, last=None)

    text = _quote_status_line(ticker).plain

    assert text.startswith("QUOTE  PRIMARY ? · AWAITING N/A")
    assert "SRC " not in text
    assert "UNCHANGED " not in text


def test_quote_integrity_badge_distinguishes_direct_smart_and_delayed_quotes() -> None:
    direct = SimpleNamespace(
        contract=SimpleNamespace(
            secType="STK",
            exchange="ARCA",
            primaryExchange="NYSE",
        ),
        bid=144.55,
        ask=145.30,
        last=None,
        close=142.80,
        marketDataType=1,
        tbRequestedMdType=1,
        tbQuoteSource="stream",
        tbTopQuoteUpdatedMono=monotonic() - 1126.0,
    )
    smart = SimpleNamespace(
        contract=SimpleNamespace(
            secType="STK",
            exchange="SMART",
            primaryExchange="NYSE",
        ),
        bid=144.54,
        ask=145.25,
        last=144.90,
        close=142.80,
        marketDataType=1,
        tbRequestedMdType=1,
        tbQuoteSource="stream",
        tbTopQuoteUpdatedMono=monotonic() - 1.0,
    )
    delayed = SimpleNamespace(
        contract=SimpleNamespace(
            secType="STK",
            exchange="NYSE",
            primaryExchange="NYSE",
        ),
        bid=144.10,
        ask=144.40,
        last=144.25,
        close=142.80,
        marketDataType=3,
        tbRequestedMdType=3,
        tbQuoteSource="delayed-snapshot",
        tbQuoteAsOf="2026-08-07T15:55:00",
        tbTopQuoteUpdatedMono=monotonic() - 1080.0,
    )

    direct_text = _quote_status_line(direct).plain
    smart_text = _quote_status_line(smart).plain
    delayed_text = _quote_status_line(delayed).plain

    assert "BBO DIRECT · ARCA · LIVE · VENUE ONLY" in direct_text
    assert "UNCHANGED 19m" in direct_text
    assert "BID/ASK ✓ · LAST —" in direct_text
    assert "SMART AGG · LIVE" in smart_text
    assert "BBO DIRECT" not in smart_text
    assert "PRIMARY NYSE · DELAYED · CONTEXT ONLY" in delayed_text
    assert "ASOF 15:55:00" in delayed_text


def test_market_data_presentation_distinguishes_all_ibkr_types() -> None:
    expected = {
        1: ("Live", " [L]"),
        2: ("Frozen", " [F]"),
        3: ("Delayed", " [D]"),
        4: ("Delayed-Frozen", " [DF]"),
    }
    for md_type, presentation in expected.items():
        ticker = SimpleNamespace(marketDataType=md_type)
        assert (_market_data_label(ticker), _market_data_tag(ticker)) == presentation


def test_ui_market_session_uses_calendar_aware_phase_authority() -> None:
    assert _market_session_bucket(datetime(2025, 11, 28, 12, 59)) == "RTH"
    assert _market_session_bucket(datetime(2025, 11, 28, 13, 0)) == "POST"
    assert _market_session_bucket(datetime(2025, 11, 28, 17, 0)) == "CLOSED"
    assert _market_session_bucket(datetime(2026, 8, 9, 20, 0)) == "OVERNIGHT"
