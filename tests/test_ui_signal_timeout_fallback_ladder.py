from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from ib_insync import Stock

from tradebot.client import OhlcvBar
from tradebot.time_utils import now_et_naive
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


def test_signal_timeout_fallback_ladder_stops_at_day_or_lower() -> None:
    screen = _screen()
    assert screen._signal_timeout_fallback_durations("1 W") == ("2 D", "1 D")
    assert screen._signal_timeout_fallback_durations("2 D") == ("1 D",)
    assert screen._signal_timeout_fallback_durations("1 D") == ()


def test_signal_duration_str_starts_at_floor_for_daily_shock() -> None:
    screen = _screen()
    filters = {
        "shock_gate_mode": "detect",
        "shock_detector": "daily_drawdown",
        "shock_drawdown_lookback_days": 10,
    }
    assert screen._signal_duration_str("10 mins", filters=filters) == "1 M"


def test_signal_duration_str_ignores_legacy_router_metadata() -> None:
    screen = _screen()
    strategy = {
        "regime_router": True,
        "regime_router_slow_window_days": 84,
    }
    assert screen._signal_duration_str("5 mins", strategy=strategy, use_rth=True) == "1 W"


def test_signal_strategy_payload_preserves_unknown_transport_fields() -> None:
    screen = _screen()
    base_strategy = {
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

    def proxy_error(self):
        return None

    def reconnect_phase(self):
        return None


class _SnapshotClient:
    def __init__(self, *, daily_bars: int = 140, intraday_bars: int = 80) -> None:
        self.daily_bars = int(daily_bars)
        self.intraday_bars = int(intraday_bars)
        self.calls: list[tuple[str, str]] = []

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
        _ = contract, what_to_show, cache_ttl_sec, use_rth
        duration = str(duration_str).strip()
        size = str(bar_size).strip()
        self.calls.append((duration, size))
        now_ref = now_et_naive()
        if "day" in size.lower():
            end_day = now_ref.date() - timedelta(days=1)
            bars: list[OhlcvBar] = []
            for idx in range(int(self.daily_bars)):
                day = end_day - timedelta(days=int(self.daily_bars - idx - 1))
                bars.append(
                    OhlcvBar(
                        ts=datetime.combine(day, datetime.min.time()),
                        open=1.0,
                        high=1.0,
                        low=1.0,
                        close=1.0,
                        volume=1.0,
                    )
                )
            return bars

        step_mins = 5 if "5" in size else 10
        bars = []
        last_ts = now_ref - timedelta(minutes=max(2, step_mins + 1))
        start_ts = last_ts - timedelta(minutes=step_mins * int(self.intraday_bars - 1))
        for idx in range(int(self.intraday_bars)):
            ts = start_ts + timedelta(minutes=step_mins * idx)
            bars.append(OhlcvBar(ts=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
        return bars

    def last_historical_request(self, contract):
        _ = contract
        return None

    def proxy_error(self):
        return None

    def reconnect_phase(self):
        return None


def test_signal_snapshot_regime2_off_does_not_require_regime2_bars() -> None:
    client = _SnapshotClient()
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 12301

    snap = asyncio.run(
        screen._signal_snapshot_for_contract(
            contract=contract,
            ema_preset_raw="5/13",
            bar_size="5 mins",
            use_rth=False,
            regime_mode_raw="ema",
            regime2_mode_raw="off",
        )
    )

    assert snap is not None


def test_signal_snapshot_ignores_legacy_router_daily_seed_request() -> None:
    client = _SnapshotClient(daily_bars=160, intraday_bars=90)
    screen = _screen_with_client(client)
    contract = Stock("TQQQ", "SMART", "USD")
    contract.conId = 12302

    base_strategy = {
        "regime_router": True,
        "regime_router_fast_window_days": 63,
        "regime_router_slow_window_days": 84,
        "regime_router_min_dwell_days": 10,
    }

    snap = asyncio.run(
        screen._signal_snapshot_for_contract(
            contract=contract,
            ema_preset_raw="5/13",
            bar_size="5 mins",
            use_rth=True,
            regime_mode_raw="ema",
            regime2_mode_raw="off",
            base_strategy_raw=base_strategy,
        )
    )

    assert snap is not None
    assert ("1 W", "5 mins") in client.calls
    assert not any(bar_size == "1 day" for _, bar_size in client.calls)


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


def test_signal_fetch_bars_exposes_only_causally_closed_timestamps() -> None:
    client = _FallbackClient(fail_durations=set())
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 10005

    series, health = asyncio.run(
        screen._signal_fetch_bars(
            contract=contract,
            duration_str="1 W",
            bar_size="10 mins",
            use_rth=False,
            now_ref=datetime(2026, 2, 20, 12, 5),
            strict_zero_gap=False,
            heal_if_stale=False,
        )
    )

    assert series is not None
    assert [bar.ts for bar in series] == [
        datetime(2026, 2, 20, 11, 50),
        datetime(2026, 2, 20, 12, 0),
    ]
    assert series.meta.extra["timestamp_semantics"] == "close"
    assert health["last_bar_ts"] == datetime(2026, 2, 20, 11, 50)


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


def test_signal_fetch_bars_fallback_downgrades_week_to_days() -> None:
    client = _FallbackClient(fail_durations={"1 W"})
    screen = _screen_with_client(client)
    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 10004

    _, health = asyncio.run(
        screen._signal_fetch_bars(
            contract=contract,
            duration_str="1 W",
            bar_size="10 mins",
            use_rth=False,
            now_ref=datetime(2026, 2, 20, 12, 5),
            strict_zero_gap=False,
            heal_if_stale=False,
        )
    )

    assert client.calls == ["1 W", "2 D"]
    assert isinstance(health, dict)
    assert str(health.get("duration_str")) == "2 D"
    assert bool(health.get("timeout_fallback_used")) is True
    assert str(health.get("timeout_fallback_from_duration")) == "1 W"
    assert list(health.get("timeout_fallback_attempts") or []) == ["2 D"]


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
