from __future__ import annotations

from types import SimpleNamespace

from tradebot.ui.favorites import FavoritesScreen


def test_market_data_text_prefers_stream_source_over_delayed_md_type() -> None:
    ticker = SimpleNamespace(
        marketDataType=3,
        tbQuoteSource="stream",
    )

    text = FavoritesScreen._market_data_text(ticker)

    assert text.plain == "[L]"


def test_market_data_text_keeps_delayed_label_for_delayed_snapshot_source() -> None:
    ticker = SimpleNamespace(
        marketDataType=3,
        tbQuoteSource="delayed-snapshot",
    )

    text = FavoritesScreen._market_data_text(ticker)

    assert text.plain == "[D]"


def test_market_data_text_keeps_live_label_for_live_md_type() -> None:
    ticker = SimpleNamespace(
        marketDataType=1,
        tbQuoteSource="",
    )

    text = FavoritesScreen._market_data_text(ticker)

    assert text.plain == "[L]"
