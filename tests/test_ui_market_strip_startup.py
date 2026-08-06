from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.ui.app import PositionsApp
from tradebot.ui.portfolio.table import PortfolioTable


class _StripHarness:
    _schedule_market_strip_render = PortfolioTable._schedule_market_strip_render
    _flush_market_strip = PortfolioTable._flush_market_strip

    def __init__(self) -> None:
        self._market_strip_task: asyncio.Task | None = None
        self.events: list[str] = []

    def _sync_market_strip(self) -> None:
        self.events.append("sync")

    def _render_ticker_bar(self) -> None:
        self.events.append("render")


def test_market_strip_render_lane_is_independent_and_coalesced() -> None:
    async def _run() -> None:
        harness = _StripHarness()

        harness._schedule_market_strip_render()
        first_task = harness._market_strip_task
        harness._schedule_market_strip_render()

        assert harness._market_strip_task is first_task
        assert first_task is not None
        await first_task
        assert harness.events == ["sync", "render"]

    asyncio.run(_run())


def test_mount_starts_market_lanes_before_account_snapshot() -> None:
    events: list[str] = []
    table = SimpleNamespace(
        cursor_type=None,
        focus=lambda: events.append("focus"),
    )
    ticker = SimpleNamespace()
    status = SimpleNamespace()
    search = SimpleNamespace(display=True)
    widgets = iter((table, ticker, status, search))

    class _Client:
        def set_update_callback(self, _callback) -> None:
            events.append("callback")

        def start_market_data(self) -> None:
            events.append("market")

    fake = SimpleNamespace(
        query_one=lambda *_args, **_kwargs: next(widgets),
        _setup_columns=lambda: events.append("columns"),
        _bot_runtime=SimpleNamespace(install=lambda _app: events.append("bot")),
        _client=_Client(),
        _mark_stream_dirty=lambda: None,
        _schedule_market_strip_render=lambda: events.append("strip"),
        _mark_dirty=lambda **_kwargs: events.append("snapshot"),
    )

    asyncio.run(PositionsApp.on_mount(fake))

    assert events.index("market") < events.index("snapshot")
    assert search.display is False
