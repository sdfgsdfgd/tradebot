from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from ib_insync import Stock

from tradebot.client import OhlcvBar
from tradebot.ui.bot import BotScreen


def _screen() -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return BotScreen(client=SimpleNamespace(), refresh_sec=1.0)


def _screen_with_client(client) -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return BotScreen(client=client, refresh_sec=1.0)


def test_signal_timeout_fallback_ladder_for_3m() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("3 M") == ("2 M", "1 M", "1 W", "2 D", "1 D")


def test_signal_timeout_fallback_ladder_for_2m() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("2 M") == ("1 M", "1 W", "2 D", "1 D")


def test_signal_timeout_fallback_ladder_for_6m_includes_3m_first() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("6 M") == ("3 M", "2 M", "1 M", "1 W", "2 D", "1 D")


def test_signal_timeout_fallback_ladder_stops_at_week_or_lower() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("1 W") == ()
    assert screen._signal_timeout_fallback_durations("2 D") == ()
    assert screen._signal_timeout_fallback_durations("1 D") == ()


class _FallbackClient:
    def __init__(self, *, fail_durations: set[str]) -> None:
        self.fail_durations = {str(item).strip() for item in fail_durations}
        self.calls: list[str] = []
        self._diag_by_con_id: dict[int, dict[str, object]] = {}

    async def historical_bars_ohlcv(
        self,
        contract,
        *,
        duration_str: str,
        bar_size: str,
        use_rth: bool,
        what_to_show: str,
        cache_ttl_sec: float,
    ):
        duration = str(duration_str).strip()
        self.calls.append(duration)
        con_id = int(getattr(contract, "conId", 0) or 0)
        if duration in self.fail_durations:
            self._diag_by_con_id[con_id] = {
                "status": "timeout",
                "request": {
                    "duration_str": duration,
                    "bar_size": str(bar_size),
                    "what_to_show": str(what_to_show),
                    "use_rth": bool(use_rth),
                },
            }
            return []
        self._diag_by_con_id[con_id] = {
            "status": "ok",
            "request": {
                "duration_str": duration,
                "bar_size": str(bar_size),
                "what_to_show": str(what_to_show),
                "use_rth": bool(use_rth),
            },
        }
        return [
            OhlcvBar(ts=datetime(2026, 2, 20, 11, 40), open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0),
            OhlcvBar(ts=datetime(2026, 2, 20, 11, 50), open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0),
        ]

    def last_historical_request(self, contract):
        con_id = int(getattr(contract, "conId", 0) or 0)
        return self._diag_by_con_id.get(con_id)


def test_signal_fetch_bars_fallback_prefers_1m_before_1w() -> None:
    client = _FallbackClient(fail_durations={"3 M", "2 M"})
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 10001

    _, health = asyncio.run(
        screen._signal_fetch_bars(
            contract=contract,
            duration_str="3 M",
            bar_size="10 mins",
            use_rth=False,
            now_ref=datetime(2026, 2, 20, 12, 5),
            strict_zero_gap=False,
            heal_if_stale=False,
        )
    )

    assert client.calls == ["3 M", "2 M", "1 M"]
    assert isinstance(health, dict)
    assert str(health.get("duration_str")) == "1 M"
    assert bool(health.get("timeout_fallback_used")) is True
    assert str(health.get("timeout_fallback_from_duration")) == "3 M"
    assert list(health.get("timeout_fallback_attempts") or []) == ["2 M", "1 M"]


def test_signal_fetch_bars_fallback_reaches_1w_when_months_fail() -> None:
    client = _FallbackClient(fail_durations={"3 M", "2 M", "1 M"})
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 10002

    _, health = asyncio.run(
        screen._signal_fetch_bars(
            contract=contract,
            duration_str="3 M",
            bar_size="10 mins",
            use_rth=False,
            now_ref=datetime(2026, 2, 20, 12, 5),
            strict_zero_gap=False,
            heal_if_stale=False,
        )
    )

    assert client.calls == ["3 M", "2 M", "1 M", "1 W"]
    assert isinstance(health, dict)
    assert str(health.get("duration_str")) == "1 W"
    assert bool(health.get("timeout_fallback_used")) is True
    assert str(health.get("timeout_fallback_from_duration")) == "3 M"
    assert list(health.get("timeout_fallback_attempts") or []) == ["2 M", "1 M", "1 W"]
