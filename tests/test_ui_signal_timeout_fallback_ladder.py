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


def test_signal_duration_str_starts_at_floor_for_daily_shock() -> None:
    screen = _screen()
    filters = {
        "shock_gate_mode": "detect",
        "shock_detector": "daily_drawdown",
        "shock_drawdown_lookback_days": 10,
    }
    assert screen._signal_duration_str("10 mins", filters=filters) == "1 M"


def test_signal_duration_str_starts_at_floor_for_regime_router() -> None:
    screen = _screen()
    strategy = {
        "regime_router": True,
        "regime_router_slow_window_days": 84,
    }
    assert screen._signal_duration_str("5 mins", strategy=strategy, use_rth=True) == "6 M"


def test_signal_strategy_payload_preserves_router_and_unknown_fields() -> None:
    screen = _screen()
    base_strategy = {
        "regime_router": True,
        "regime_router_fast_window_days": 63,
        "regime_router_slow_window_days": 84,
        "regime_router_min_dwell_days": 10,
        "regime_router_bull_sovereign_on_confirm_days": 1,
        "regime_router_bull_sovereign_off_confirm_days": 7,
        "custom_transport_key": "keep_me",
    }
    payload = screen._signal_strategy_payload(
        base_strategy_raw=base_strategy,
        entry_signal="ema",
        ema_preset_raw="5/13",
        entry_mode_raw="cross",
        entry_confirm_bars=1,
        spot_dual_branch_enabled_raw=None,
        spot_dual_branch_priority_raw=None,
        spot_branch_a_ema_preset_raw=None,
        spot_branch_a_entry_confirm_bars_raw=None,
        spot_branch_a_min_signed_slope_pct_raw=None,
        spot_branch_a_max_signed_slope_pct_raw=None,
        spot_branch_a_size_mult_raw=None,
        spot_branch_b_ema_preset_raw=None,
        spot_branch_b_entry_confirm_bars_raw=None,
        spot_branch_b_min_signed_slope_pct_raw=None,
        spot_branch_b_max_signed_slope_pct_raw=None,
        spot_branch_b_size_mult_raw=None,
        orb_window_mins_raw=None,
        orb_open_time_et_raw=None,
        spot_exit_mode_raw=None,
        spot_atr_period_raw=None,
        regime_mode="ema",
        regime_preset="20/50",
        supertrend_atr_period_raw=None,
        supertrend_multiplier_raw=None,
        supertrend_source_raw=None,
        regime2_mode="off",
        regime2_preset=None,
        regime2_supertrend_atr_period_raw=None,
        regime2_supertrend_multiplier_raw=None,
        regime2_supertrend_source_raw=None,
    )
    assert payload.get("custom_transport_key") == "keep_me"
    assert payload.get("regime_router") is True
    assert payload.get("regime_router_bull_sovereign_off_confirm_days") == 7


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


def test_signal_timeout_fallback_ladder_respects_min_duration_floor() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("2 M", min_duration="1 M") == ("1 M",)
    assert screen._signal_timeout_fallback_durations("2 M", min_duration="2 M") == ()


def test_signal_fetch_bars_fallback_stops_at_min_duration_floor() -> None:
    client = _FallbackClient(fail_durations={"3 M", "2 M", "1 M"})
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 10003

    bars, health = asyncio.run(
        screen._signal_fetch_bars(
            contract=contract,
            duration_str="3 M",
            min_duration_str="1 M",
            bar_size="10 mins",
            use_rth=False,
            now_ref=datetime(2026, 2, 20, 12, 5),
            strict_zero_gap=False,
            heal_if_stale=False,
        )
    )

    assert bars is None
    assert client.calls == ["3 M", "2 M", "1 M"]
    assert isinstance(health, dict)
    assert str(health.get("duration_str")) == "3 M"
    assert list(health.get("timeout_fallback_attempts") or []) == ["2 M", "1 M"]
