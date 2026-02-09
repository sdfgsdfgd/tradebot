from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import types
from unittest.mock import patch

from ib_insync import Future, Stock

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig
from tradebot.engine import flip_exit_gate_blocked, signal_filter_checks

_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

from tradebot.ui.bot_engine_runtime import BotEngineRuntimeMixin
from tradebot.ui.bot_journal import BotJournal
from tradebot.ui.bot_models import _BotInstance, _BotOrder
from tradebot.ui.bot_order_builder import BotOrderBuilderMixin
from tradebot.ui.bot_signal_runtime import BotSignalRuntimeMixin


@dataclass(frozen=True)
class _RawBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _new_client() -> IBKRClient:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    cfg = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=123,
        proxy_client_id=124,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.5,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=60.0,
        reconnect_slow_interval_sec=60.0,
    )
    return IBKRClient(cfg)


def test_historical_full24_stitches_overnight_and_keeps_what_to_show_cache_separate() -> None:
    client = _new_client()
    calls: list[tuple[str, str, bool]] = []

    async def _fake_request(
        contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        calls.append((str(getattr(contract, "exchange", "") or ""), str(what_to_show), bool(use_rth)))
        stamp_overnight = datetime(2026, 2, 8, 21, 15)
        stamp_rth = datetime(2026, 2, 9, 10, 0)
        exchange = str(getattr(contract, "exchange", "") or "").upper()
        if exchange == "OVERNIGHT":
            if str(what_to_show).upper() == "MIDPOINT":
                return [_RawBar(stamp_overnight, 220.0, 220.0, 220.0, 220.0, 20.0)]
            return [_RawBar(stamp_overnight, 210.0, 210.0, 210.0, 210.0, 10.0)]
        if str(what_to_show).upper() == "MIDPOINT":
            return [_RawBar(stamp_rth, 120.0, 120.0, 120.0, 120.0, 2.0)]
        return [
            _RawBar(stamp_overnight, 101.0, 101.0, 101.0, 101.0, 1.0),
            _RawBar(stamp_rth, 111.0, 111.0, 111.0, 111.0, 1.0),
        ]

    client._request_historical_data = _fake_request  # type: ignore[method-assign]

    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    contract.conId = 1001

    trades_bars = asyncio.run(
        client.historical_bars_ohlcv(
            contract,
            duration_str="2 D",
            bar_size="5 mins",
            use_rth=False,
            what_to_show="TRADES",
            cache_ttl_sec=60.0,
        )
    )
    midpoint_bars = asyncio.run(
        client.historical_bars_ohlcv(
            contract,
            duration_str="2 D",
            bar_size="5 mins",
            use_rth=False,
            what_to_show="MIDPOINT",
            cache_ttl_sec=60.0,
        )
    )

    by_ts_trades = {bar.ts: bar for bar in trades_bars}
    by_ts_midpoint = {bar.ts: bar for bar in midpoint_bars}
    overnight_ts = datetime(2026, 2, 8, 21, 15)
    assert by_ts_trades[overnight_ts].close == 210.0
    assert by_ts_midpoint[overnight_ts].close == 220.0

    assert any(what == "TRADES" for _, what, _ in calls)
    assert any(what == "MIDPOINT" for _, what, _ in calls)
    assert any(exchange == "OVERNIGHT" and what == "TRADES" for exchange, what, _ in calls)
    assert any(exchange == "OVERNIGHT" and what == "MIDPOINT" for exchange, what, _ in calls)


def test_historical_bars_ohlcv_empty_cache_expires_quickly() -> None:
    client = _new_client()
    calls = 0

    async def _fake_request(
        contract,
        *,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        nonlocal calls
        calls += 1
        return []

    client._request_historical_data = _fake_request  # type: ignore[method-assign]

    contract = Future(symbol="MES", lastTradeDateOrContractMonth="202603", exchange="CME", currency="USD")
    contract.conId = 2001

    monotonic_calls = {"n": 0}

    def _fake_monotonic() -> float:
        monotonic_calls["n"] += 1
        return 1000.0 if monotonic_calls["n"] < 4 else 1001.2

    with patch("tradebot.client.time.monotonic", side_effect=_fake_monotonic):
        asyncio.run(
            client.historical_bars_ohlcv(
                contract,
                duration_str="1 W",
                bar_size="10 mins",
                use_rth=False,
                what_to_show="TRADES",
                cache_ttl_sec=30.0,
            )
        )
        asyncio.run(
            client.historical_bars_ohlcv(
                contract,
                duration_str="1 W",
                bar_size="10 mins",
                use_rth=False,
                what_to_show="TRADES",
                cache_ttl_sec=30.0,
            )
        )

    # Empty snapshots should use a short cache TTL so a second request can recover quickly.
    assert calls == 2


class _ExitGateHarness(BotSignalRuntimeMixin):
    pass


class _EntryDayHarness(BotSignalRuntimeMixin):
    @staticmethod
    def _strategy_instrument(strategy: dict) -> str:
        value = strategy.get("instrument", "spot")
        cleaned = str(value or "spot").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    @staticmethod
    def _signal_use_rth(instance: _BotInstance) -> bool:
        raw = instance.strategy.get("signal_use_rth")
        return bool(raw)


class _GapGateHarness(BotSignalRuntimeMixin):
    def __init__(self, *, instance: _BotInstance, snap, diag: dict | None = None) -> None:
        self._instances = [instance]
        self._orders = []
        self._positions = []
        self._payload = None
        self._order_task = None
        self._events: list[tuple[str, str | None, dict | None]] = []
        self._snap = snap
        self._last_signal_snapshot_diag = dict(diag) if isinstance(diag, dict) else {
            "stage": "ok",
            "bars_count": 100,
            "regime_bars_count": 100,
            "regime2_bars_count": 100,
        }

    def _check_order_trigger_watchdogs(self, *, now_et: datetime) -> None:
        return None

    def _journal_write(self, *, event: str, reason: str | None = None, data: dict | None = None, **kwargs) -> None:
        self._events.append((str(event), reason, dict(data) if isinstance(data, dict) else data))

    async def _signal_contract(self, instance: _BotInstance, symbol: str):
        return Stock(symbol=symbol, exchange="SMART", currency="USD")

    async def _signal_snapshot_for_contract(self, **kwargs):
        return self._snap

    def _signal_snapshot_kwargs(self, *args, **kwargs):
        return {}

    @staticmethod
    def _signal_bar_size(instance: _BotInstance) -> str:
        return "10 mins"

    @staticmethod
    def _signal_use_rth(instance: _BotInstance) -> bool:
        return False

    @staticmethod
    def _strategy_instrument(strategy: dict) -> str:
        value = strategy.get("instrument", "spot")
        cleaned = str(value or "spot").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    def _resolve_open_positions(self, instance: _BotInstance, *, symbol: str, signal_contract):
        return "spot", [], None

    def _auto_process_pending_next_open(self, **kwargs) -> bool:
        return False


class _QuoteStarveHarness(BotSignalRuntimeMixin):
    def __init__(self) -> None:
        self._events: list[tuple[str, dict | None]] = []

        class _Client:
            def __init__(self):
                self.released: list[int] = []
                self.started = 0

            def release_ticker(self, con_id: int, *, owner: str = "bot") -> None:
                self.released.append(int(con_id))

            def start_proxy_tickers(self) -> None:
                self.started += 1

            @staticmethod
            def proxy_error() -> str | None:
                return None

        self._client = _Client()

    def _gate(self, status: str, data: dict | None = None) -> None:
        self._events.append((str(status), dict(data or {})))


class _PendingNextOpenHarness(BotSignalRuntimeMixin):
    def __init__(self) -> None:
        self.queued: list[dict[str, object]] = []

    @staticmethod
    def _signal_bar_size(instance: _BotInstance) -> str:
        return str(instance.strategy.get("signal_bar_size") or "10 mins")

    @staticmethod
    def _signal_use_rth(instance: _BotInstance) -> bool:
        return bool(instance.strategy.get("signal_use_rth"))

    def _queue_order(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
        trigger_reason: str | None = None,
        trigger_mode: str | None = None,
    ) -> None:
        self.queued.append(
            {
                "intent": str(intent),
                "direction": direction,
                "signal_bar_ts": signal_bar_ts,
                "trigger_reason": trigger_reason,
                "trigger_mode": trigger_mode,
            }
        )


def _new_instance(*, strategy: dict | None = None, filters: dict | None = None) -> _BotInstance:
    return _BotInstance(
        instance_id=1,
        group="g",
        symbol="SLV",
        strategy=dict(strategy or {}),
        filters=dict(filters) if isinstance(filters, dict) else filters,
    )


def test_exit_gate_blocks_same_bar_after_fill_marker_only() -> None:
    harness = _ExitGateHarness()
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.last_exit_bar_ts = bar_ts
    gates: list[str] = []

    async def _run() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates.append(str(status)),
        )

    assert asyncio.run(_run()) is False
    assert "BLOCKED_EXIT_SAME_BAR" in gates


def test_exit_gate_blocks_retry_limit_and_cooldown() -> None:
    harness = _ExitGateHarness()
    bar_ts = datetime(2026, 2, 9, 10, 0)

    instance_limit = _new_instance(strategy={"exit_retry_max_per_bar": 2})
    instance_limit.exit_retry_bar_ts = bar_ts
    instance_limit.exit_retry_count = 2
    gates_limit: list[str] = []

    async def _run_limit() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance_limit,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates_limit.append(str(status)),
        )

    assert asyncio.run(_run_limit()) is False
    assert "BLOCKED_EXIT_RETRY_LIMIT" in gates_limit

    instance_cd = _new_instance(strategy={"exit_retry_max_per_bar": 3})
    instance_cd.exit_retry_bar_ts = bar_ts
    instance_cd.exit_retry_count = 1
    instance_cd.exit_retry_cooldown_until = datetime(2026, 2, 9, 10, 5)
    gates_cd: list[str] = []

    async def _run_cd() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance_cd,
            snap=SimpleNamespace(bar_ts=bar_ts),
            instrument="spot",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates_cd.append(str(status)),
        )

    assert asyncio.run(_run_cd()) is False
    assert "BLOCKED_EXIT_RETRY_COOLDOWN" in gates_cd


def test_entry_weekday_maps_sunday_overnight_to_monday_for_spot_non_rth() -> None:
    harness = _EntryDayHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "entry_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        }
    )
    sunday_overnight = datetime(2026, 2, 8, 23, 32)
    assert harness._entry_weekday_for_ts(instance, sunday_overnight) == 0
    assert harness._can_order_now(instance, now_et=sunday_overnight) is True


def test_entry_weekday_keeps_sunday_for_rth_mode() -> None:
    harness = _EntryDayHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_use_rth": True,
            "spot_sec_type": "STK",
            "entry_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        }
    )
    sunday_overnight = datetime(2026, 2, 8, 23, 32)
    assert harness._entry_weekday_for_ts(instance, sunday_overnight) == 6
    assert harness._can_order_now(instance, now_et=sunday_overnight) is False


def test_signal_filter_time_respects_runtime_et_wall_clock_bars() -> None:
    harness = _EntryDayHarness()
    filters = {"entry_start_hour_et": 10, "entry_end_hour_et": 15}
    before_open = datetime(2026, 2, 9, 9, 50)
    in_window = datetime(2026, 2, 9, 10, 0)

    before_checks = signal_filter_checks(
        filters,
        bar_ts=harness._as_et_aware(before_open),
        bars_in_day=1,
        close=73.0,
    )
    in_window_checks = signal_filter_checks(
        filters,
        bar_ts=harness._as_et_aware(in_window),
        bars_in_day=1,
        close=73.0,
    )

    assert before_checks["time"] is False
    assert in_window_checks["time"] is True


def test_pending_next_open_cancels_on_risk_overlay_date_roll() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={"instrument": "spot"},
        filters={"riskoff_mode": "hygiene", "risk_entry_cutoff_hour_et": 4},
    )
    instance.pending_entry_direction = "up"
    instance.pending_entry_signal_bar_ts = datetime(2026, 2, 8, 15, 50)
    instance.pending_entry_due_ts = datetime(2026, 2, 9, 4, 0)
    gates: list[str] = []

    fired = harness._auto_process_pending_next_open(
        instance=instance,
        instrument="spot",
        open_items=[],
        open_dir=None,
        now_wall=datetime(2026, 2, 9, 4, 0),
        snap=SimpleNamespace(risk=SimpleNamespace(riskoff=True, riskpanic=False, riskpop=False), shock_dir="up"),
        gate=lambda status, data=None: gates.append(str(status)),
    )

    assert fired is False
    assert not harness.queued
    assert instance.pending_entry_due_ts is None
    assert "CANCEL_PENDING_ENTRY_RISK_OVERLAY" in gates


def test_pending_next_open_cancels_on_directional_shock_mismatch() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={"instrument": "spot"},
        filters={"riskoff_mode": "directional"},
    )
    instance.pending_entry_direction = "up"
    instance.pending_entry_signal_bar_ts = datetime(2026, 2, 9, 3, 50)
    instance.pending_entry_due_ts = datetime(2026, 2, 9, 4, 0)
    gates: list[str] = []

    fired = harness._auto_process_pending_next_open(
        instance=instance,
        instrument="spot",
        open_items=[],
        open_dir=None,
        now_wall=datetime(2026, 2, 9, 4, 0),
        snap=SimpleNamespace(risk=SimpleNamespace(riskoff=True, riskpanic=False, riskpop=False), shock_dir="down"),
        gate=lambda status, data=None: gates.append(str(status)),
    )

    assert fired is False
    assert not harness.queued
    assert instance.pending_entry_due_ts is None
    assert "CANCEL_PENDING_ENTRY_RISK_OVERLAY" in gates


def test_pending_next_open_triggers_when_directional_shock_matches() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={"instrument": "spot"},
        filters={"riskoff_mode": "directional"},
    )
    instance.pending_entry_direction = "up"
    instance.pending_entry_signal_bar_ts = datetime(2026, 2, 9, 3, 50)
    instance.pending_entry_due_ts = datetime(2026, 2, 9, 4, 0)
    gates: list[str] = []

    fired = harness._auto_process_pending_next_open(
        instance=instance,
        instrument="spot",
        open_items=[],
        open_dir=None,
        now_wall=datetime(2026, 2, 9, 4, 0),
        snap=SimpleNamespace(risk=SimpleNamespace(riskoff=True, riskpanic=False, riskpop=False), shock_dir="up"),
        gate=lambda status, data=None: gates.append(str(status)),
    )

    assert fired is True
    assert len(harness.queued) == 1
    assert harness.queued[0]["intent"] == "enter"
    assert harness.queued[0]["direction"] == "up"
    assert instance.pending_entry_due_ts is None
    assert "TRIGGER_ENTRY" in gates


def test_schedule_next_open_emits_due_from_and_now_wall() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "10 mins",
            "spot_exec_bar_size": "5 mins",
        }
    )
    signal_bar_ts = datetime(2026, 2, 9, 11, 10)
    gate_events: list[tuple[str, dict | None]] = []

    fired = harness._schedule_pending_entry_next_open(
        instance=instance,
        direction="up",
        signal_bar_ts=signal_bar_ts,
        now_wall=datetime(2026, 2, 9, 11, 29),
        gate=lambda status, data=None: gate_events.append((str(status), dict(data or {}))),
    )

    assert fired is True
    assert gate_events
    status, payload = gate_events[-1]
    assert status == "TRIGGER_ENTRY"
    assert payload.get("next_open_due_from") == "2026-02-09T11:20:00"
    assert payload.get("now_wall_ts") == "2026-02-09T11:29:00"


def test_journal_signal_and_order_build_failed_compact_diagnostics() -> None:
    signal_entry = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="SIGNAL",
        reason=None,
        instance_id="1",
        symbol="SLV",
        extra={},
        detail={
            "bar_ts": "2026-02-09T11:10:00",
            "bar_health": {"lag_bars": 1.94, "stale": False, "gap_detected": False},
            "signal": {"state": "up", "entry_dir": "up", "regime_dir": "up", "regime_ready": True},
            "meta": {
                "entry_signal": "ema",
                "signal_bar_size": "10 mins",
                "shock_gate_mode": "detect",
                "shock_detector": "tr_ratio",
                "shock_scale_detector": "daily_atr_pct",
            },
        },
    )
    assert "bar=11:10" in str(signal_entry["msg"])
    assert "lag=1.94b" in str(signal_entry["msg"])
    assert "stale=0" in str(signal_entry["msg"])
    assert "gap=0" in str(signal_entry["msg"])
    assert "shm=detect" in str(signal_entry["msg"])
    assert "shdet=tr_ratio" in str(signal_entry["msg"])
    assert "shscale=daily_atr_pct" in str(signal_entry["msg"])

    failed_entry = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="ORDER_BUILD_FAILED",
        reason="enter",
        instance_id="1",
        symbol="SLV",
        extra={"error": "Quote: no bid/ask/last (cannot price)", "order_journal": {"order_attempt": 2}},
        detail={
            "order_attempt": 2,
            "retry_reason": "quote_unpriced",
            "quote": {
                "bid": None,
                "ask": None,
                "last": None,
                "md_ok": False,
                "live": False,
                "delayed": True,
                "frozen": False,
                "ticker_age_ms": 1250,
            },
        },
    )
    msg = str(failed_entry["msg"])
    assert "attempt=2" in msg
    assert "retry=quote_unpriced" in msg
    assert "md_ok=0" in msg
    assert "delayed=1" in msg
    assert "age_ms=1250" in msg


def test_flip_exit_gate_mode_regime_blocks_supported_position() -> None:
    signal = SimpleNamespace(
        ema_ready=True,
        ema_fast=101.0,
        ema_slow=100.0,
        state="down",
        cross_up=False,
        cross_down=False,
        regime_ready=True,
        regime_dir="up",
    )
    assert (
        flip_exit_gate_blocked(
            gate_mode_raw="regime",
            filters=None,
            close=100.0,
            signal=signal,
            trade_dir="up",
        )
        is True
    )


def test_flip_exit_gate_mode_permission_blocks_supported_position() -> None:
    signal = SimpleNamespace(
        ema_ready=True,
        ema_fast=101.0,
        ema_slow=100.0,
        state="down",
        cross_up=False,
        cross_down=False,
        regime_ready=False,
        regime_dir="down",
    )
    assert (
        flip_exit_gate_blocked(
            gate_mode_raw="permission",
            filters={"ema_spread_min_pct": 0.5},
            close=100.0,
            signal=signal,
            trade_dir="up",
        )
        is True
    )


def test_auto_tick_blocks_waiting_data_gap_when_flat() -> None:
    bar_ts = datetime(2026, 2, 9, 12, 0)
    signal = SimpleNamespace(
        state="up",
        entry_dir="up",
        cross_up=False,
        cross_down=False,
        ema_ready=True,
        regime_dir="up",
        regime_ready=True,
        ema_fast=74.0,
        ema_slow=73.5,
        prev_ema_fast=73.9,
    )
    snap = SimpleNamespace(
        bar_ts=bar_ts,
        close=74.2,
        signal=signal,
        bars_in_day=5,
        rv=None,
        volume=1000.0,
        volume_ema=None,
        volume_ema_ready=True,
        shock=True,
        shock_dir="up",
        shock_atr_pct=0.4,
        risk=None,
        atr=None,
        or_high=None,
        or_low=None,
        or_ready=False,
        bar_health={
            "stale": False,
            "gap_detected": True,
            "last_bar_ts": bar_ts,
            "recent_gap_count": 1,
            "max_gap_bars": 3.0,
        },
    )
    instance = _new_instance(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "5/13",
            "instrument": "spot",
        }
    )
    harness = _GapGateHarness(instance=instance, snap=snap)

    asyncio.run(harness._auto_order_tick())

    assert any(
        event == "GATE" and reason == "WAITING_DATA_GAP"
        for event, reason, _data in harness._events
    )


def test_auto_tick_blocks_waiting_preflight_bars_when_history_too_short() -> None:
    bar_ts = datetime(2026, 2, 9, 12, 0)
    signal = SimpleNamespace(
        state="up",
        entry_dir="up",
        cross_up=False,
        cross_down=False,
        ema_ready=True,
        regime_dir="up",
        regime_ready=True,
        ema_fast=74.0,
        ema_slow=73.5,
        prev_ema_fast=73.9,
    )
    snap = SimpleNamespace(
        bar_ts=bar_ts,
        close=74.2,
        signal=signal,
        bars_in_day=5,
        rv=None,
        volume=1000.0,
        volume_ema=None,
        volume_ema_ready=True,
        shock=False,
        shock_dir="up",
        shock_atr_pct=0.4,
        risk=None,
        atr=1.2,
        or_high=None,
        or_low=None,
        or_ready=False,
        bar_health={
            "stale": False,
            "gap_detected": False,
            "last_bar_ts": bar_ts,
            "recent_gap_count": 0,
            "max_gap_bars": 0.0,
        },
    )
    instance = _new_instance(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "5/13",
            "regime_mode": "supertrend",
            "regime_bar_size": "1 day",
            "supertrend_atr_period": 7,
            "instrument": "spot",
        }
    )
    harness = _GapGateHarness(
        instance=instance,
        snap=snap,
        diag={
            "stage": "ok",
            "bars_count": 5,
            "regime_bars_count": 3,
            "regime2_bars_count": 0,
        },
    )

    asyncio.run(harness._auto_order_tick())

    assert any(
        event == "GATE" and reason == "WAITING_PREFLIGHT_BARS"
        for event, reason, _data in harness._events
    )


def test_quote_starvation_watchdog_repairs_and_recovers() -> None:
    harness = _QuoteStarveHarness()
    instance = _new_instance(
        strategy={
            "spot_quote_starve_warn_sec": 5,
            "spot_quote_starve_kill_sec": 10,
            "spot_quote_starve_repair_cooldown_sec": 3,
            "spot_quote_starve_action": "force_exit",
        }
    )
    now = datetime(2026, 2, 9, 12, 0, 0)

    first = harness._spot_quote_starvation_watchdog(
        instance=instance,
        now_wall=now,
        symbol="SLV",
        con_id=1234,
        bid=None,
        ask=None,
        last=None,
        ticker_market_price=None,
        market_price=None,
        gate=harness._gate,
    )
    assert first["starving"] is True
    assert first["force_exit"] is False

    warn = harness._spot_quote_starvation_watchdog(
        instance=instance,
        now_wall=now.replace(second=6),
        symbol="SLV",
        con_id=1234,
        bid=None,
        ask=None,
        last=None,
        ticker_market_price=None,
        market_price=None,
        gate=harness._gate,
    )
    assert warn["starving"] is True
    assert harness._client.released == [1234]
    assert harness._client.started >= 1

    kill = harness._spot_quote_starvation_watchdog(
        instance=instance,
        now_wall=now.replace(second=12),
        symbol="SLV",
        con_id=1234,
        bid=None,
        ask=None,
        last=None,
        ticker_market_price=None,
        market_price=None,
        gate=harness._gate,
    )
    assert kill["force_exit"] is True
    assert any(status == "QUOTE_STARVE_KILL_SWITCH" for status, _ in harness._events)

    recovered = harness._spot_quote_starvation_watchdog(
        instance=instance,
        now_wall=now.replace(second=13),
        symbol="SLV",
        con_id=1234,
        bid=73.1,
        ask=73.2,
        last=73.15,
        ticker_market_price=73.15,
        market_price=73.15,
        gate=harness._gate,
    )
    assert recovered["starving"] is False
    assert instance.quote_starvation_since is None
    assert any(status == "QUOTE_STARVE_RECOVERED" for status, _ in harness._events)


class _FakeTrade:
    def __init__(
        self,
        status: str,
        *,
        why_held: str = "",
        log_messages: list[str] | None = None,
    ) -> None:
        self.orderStatus = SimpleNamespace(status=status, whyHeld=why_held)
        self.log = [SimpleNamespace(message=msg) for msg in list(log_messages or [])]
        self.fills = []

    def isDone(self) -> bool:
        return True


class _EngineHarness(BotEngineRuntimeMixin):
    def __init__(self, *, instance: _BotInstance, order: _BotOrder, client=None) -> None:
        self._orders = [order]
        self._instances = [instance]
        self._last_chase_ts = -1_000_000_000.0
        self._active_panel = "instances"
        self._order_rows = []
        self._events: list[tuple[str, dict]] = []
        self._client = (
            client
            if client is not None
            else SimpleNamespace(pop_order_error=lambda order_id, max_age_sec=120.0: None)
        )

    async def _reprice_order(self, order: _BotOrder, *, mode: str) -> bool:
        return False

    def _journal_write(self, *, event: str, data: dict | None = None, **kwargs) -> None:
        self._events.append((str(event), dict(data or {})))

    def _refresh_orders_table(self) -> None:
        return None


def _new_order(
    *,
    status: str,
    signal_bar_ts: datetime,
    order_id: int | None = 777,
    why_held: str = "",
    log_messages: list[str] | None = None,
) -> _BotOrder:
    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    return _BotOrder(
        instance_id=1,
        preset=None,
        underlying=contract,
        order_contract=contract,
        legs=[],
        action="SELL",
        quantity=1,
        limit_price=1.0,
        created_at=datetime(2026, 2, 9, 10, 0),
        status="WORKING",
        order_id=order_id,
        intent="exit",
        signal_bar_ts=signal_bar_ts,
        trade=_FakeTrade(status, why_held=why_held, log_messages=log_messages),
    )


def test_engine_inactive_exit_marks_retryable_and_sets_cooldown() -> None:
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance(strategy={"exit_retry_cooldown_sec": 2})
    order = _new_order(status="Inactive", signal_bar_ts=bar_ts)
    harness = _EngineHarness(instance=instance, order=order)

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "INACTIVE"
    assert instance.exit_retry_bar_ts == bar_ts
    assert instance.exit_retry_count == 1
    assert instance.exit_retry_cooldown_until is not None
    assert any(event == "ORDER_DONE" and data.get("retryable") is True for event, data in harness._events)


def test_engine_filled_exit_sets_same_bar_lock_and_clears_retry_state() -> None:
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.exit_retry_bar_ts = bar_ts
    instance.exit_retry_count = 2
    instance.exit_retry_cooldown_until = datetime(2026, 2, 9, 10, 5)
    order = _new_order(status="Filled", signal_bar_ts=bar_ts)
    harness = _EngineHarness(instance=instance, order=order)

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "FILLED"
    assert instance.last_exit_bar_ts == bar_ts
    assert instance.exit_retry_bar_ts is None
    assert instance.exit_retry_count == 0
    assert instance.exit_retry_cooldown_until is None
    assert any(event == "ORDER_FILLED" and data.get("exit_lock_bar_ts") for event, data in harness._events)


def test_engine_inactive_exit_attaches_ib_reject_reason() -> None:
    class _RejectClient:
        def pop_order_error(self, order_id, *, max_age_sec: float = 120.0):
            assert int(order_id) == 777
            return {
                "code": 201,
                "message": "The time-in-force GTC is invalid for this combination of exchange and security type.",
            }

    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance(strategy={"exit_retry_cooldown_sec": 2})
    order = _new_order(status="Inactive", signal_bar_ts=bar_ts, order_id=777)
    harness = _EngineHarness(instance=instance, order=order, client=_RejectClient())

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "INACTIVE"
    assert order.error is not None
    assert order.error.startswith("IB 201:")
    done_payload = next(data for event, data in harness._events if event == "ORDER_DONE")
    assert done_payload.get("ib_error_code") == 201
    assert "time-in-force GTC is invalid" in str(done_payload.get("ib_error_message") or "")


def test_order_error_cache_pop_and_expiry() -> None:
    client = _new_client()
    client._remember_order_error(1001, 201, "bad tif")
    payload = client.pop_order_error(1001)
    assert payload == {"code": 201, "message": "bad tif"}
    assert client.pop_order_error(1001) is None

    client._order_error_cache[1002] = (time.monotonic() - 600.0, 201, "stale")
    assert client.pop_order_error(1002, max_age_sec=120.0) is None

    client._remember_order_error(1003, 2104, "Market data farm connection is OK")
    assert client.pop_order_error(1003) is None


def test_place_limit_order_overnight_uses_day_tif(monkeypatch) -> None:
    class _FakeIB:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def placeOrder(self, contract, order):
            self.calls.append((contract, order))
            return SimpleNamespace(contract=contract, order=SimpleNamespace(orderId=123, permId=0))

    client = _new_client()
    fake_ib = _FakeIB()
    client._ib = fake_ib

    async def _fake_connect() -> None:
        return None

    client.connect = _fake_connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (False, True))

    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    asyncio.run(client.place_limit_order(contract, "BUY", 1, 73.42, outside_rth=True))

    placed_contract, placed_order = fake_ib.calls[-1]
    assert str(getattr(placed_contract, "exchange", "")).upper() == "OVERNIGHT"
    assert str(getattr(placed_order, "tif", "")).upper() == "DAY"
    assert bool(getattr(placed_order, "outsideRth", False)) is False


def test_place_limit_order_premarket_uses_gtc_outside_rth(monkeypatch) -> None:
    class _FakeIB:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def placeOrder(self, contract, order):
            self.calls.append((contract, order))
            return SimpleNamespace(contract=contract, order=SimpleNamespace(orderId=123, permId=0))

    client = _new_client()
    fake_ib = _FakeIB()
    client._ib = fake_ib

    async def _fake_connect() -> None:
        return None

    client.connect = _fake_connect  # type: ignore[method-assign]
    monkeypatch.setattr("tradebot.client._session_flags", lambda _now: (True, False))

    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    asyncio.run(client.place_limit_order(contract, "SELL", 2, 73.11, outside_rth=True))

    placed_contract, placed_order = fake_ib.calls[-1]
    assert str(getattr(placed_contract, "exchange", "")).upper() == "SMART"
    assert str(getattr(placed_order, "tif", "")).upper() == "GTC"
    assert bool(getattr(placed_order, "outsideRth", False)) is True


def test_initial_exec_mode_escalates_stop_exit_retries() -> None:
    instance = _new_instance()
    instance.order_trigger_reason = "stop_loss_pct"

    instance.exit_retry_count = 0
    assert (
        BotOrderBuilderMixin._initial_exec_mode(
            instance=instance,
            instrument="spot",
            intent_clean="exit",
        )
        == "OPTIMISTIC"
    )

    instance.exit_retry_count = 1
    assert (
        BotOrderBuilderMixin._initial_exec_mode(
            instance=instance,
            instrument="spot",
            intent_clean="exit",
        )
        == "MID"
    )

    instance.exit_retry_count = 2
    assert (
        BotOrderBuilderMixin._initial_exec_mode(
            instance=instance,
            instrument="spot",
            intent_clean="exit",
        )
        == "AGGRESSIVE"
    )

    assert (
        BotOrderBuilderMixin._initial_exec_mode(
            instance=instance,
            instrument="options",
            intent_clean="exit",
        )
        == "OPTIMISTIC"
    )
