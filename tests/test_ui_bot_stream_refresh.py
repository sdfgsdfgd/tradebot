from __future__ import annotations

import asyncio

from tradebot.ui.bot import BotScreen


class _LockStub:
    @staticmethod
    def locked() -> bool:
        return False


class _StreamRefreshProbe:
    def __init__(self) -> None:
        self._stream_dirty = True
        self._refresh_sec = 0.25
        self._refresh_lock = _LockStub()
        self.calls: list[tuple[str, object | None]] = []

    def _refresh_instances_table(self, *, refresh_dependents: bool = True) -> None:
        self.calls.append(("instances", refresh_dependents))
        if refresh_dependents:
            self.calls.append(("orders_via_dependents", None))
            self.calls.append(("logs_via_dependents", None))

    def _refresh_orders_table(self) -> None:
        self.calls.append(("orders", None))

    def _render_status(self) -> None:
        self.calls.append(("status", None))


def _run_flush(probe: _StreamRefreshProbe) -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(BotScreen._flush_stream_refresh(probe))


def test_stream_refresh_disables_dependent_table_refreshes() -> None:
    probe = _StreamRefreshProbe()
    _run_flush(probe)

    assert ("instances", False) in probe.calls
    assert ("logs_via_dependents", None) not in probe.calls


def test_stream_refresh_renders_orders_once() -> None:
    probe = _StreamRefreshProbe()
    _run_flush(probe)

    orders_calls = [item for item in probe.calls if item[0] == "orders"]
    assert len(orders_calls) == 1
