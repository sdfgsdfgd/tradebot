from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import types
from unittest.mock import patch

from ib_insync import Bag, ComboLeg, Contract, Future, Stock

from tradebot.backtest.engine import _spot_strategy_sec_type
from tradebot.backtest.spot_codec import spot_strategy_payload, strategy_from_payload
from tradebot.client import BrokerOrderPreview, IBKRClient, _session_flags
from tradebot.config import IBKRConfig
import tradebot.live.options as live_options_module
from tradebot.live.options import QualifiedOptionLeg
from tradebot.option_package import OptionPackageRisk
from tradebot.order_admission import evaluate_order_admission
from tradebot.research.spot_sweeps.support import _bundle_base
from tradebot.spot.gates import flip_exit_gate_blocked, signal_filter_checks

_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

from tradebot.ui.bot_engine_runtime import BotEngineRuntimeMixin
from tradebot.ui.bot_journal import BotJournal
from tradebot.ui.bot_journal_diagnostics import JournalDiagnostics
from tradebot.ui.bot_models import _BotInstance, _BotOrder
from tradebot.ui.bot_order_builder import BotOrderBuilderMixin
from tradebot.ui.bot_signal_runtime import BotSignalRuntimeMixin
from tradebot.ui.bot import BotScreen
import tradebot.ui.bot_screen.orders as bot_orders_module
import tradebot.ui.bot_signal_runtime as bot_signal_runtime_module


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


def test_ib_bar_datetime_converts_aware_timestamp_to_et_naive() -> None:
    parsed = IBKRClient._ib_bar_datetime(datetime(2026, 2, 10, 14, 30, tzinfo=timezone.utc))
    assert parsed == datetime(2026, 2, 10, 9, 30)


def test_historical_full24_stitches_overnight_and_keeps_what_to_show_cache_separate() -> None:
    client = _new_client()
    calls: list[tuple[str, str, bool]] = []

    async def _fake_request(
        contract,
        *,
        end_ts: datetime | None = None,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        _ = end_ts
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
        end_ts: datetime | None = None,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        _ = end_ts
        nonlocal calls
        calls += 1
        return []

    client._request_historical_data = _fake_request  # type: ignore[method-assign]

    contract = Future(symbol="MES", lastTradeDateOrContractMonth="202603", exchange="CME", currency="USD")
    contract.conId = 2001

    with patch("tradebot.client.time.monotonic", return_value=1000.0):
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
    with patch("tradebot.client.time.monotonic", return_value=1000.5):
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
    with patch("tradebot.client.time.monotonic", return_value=1002.0):
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


def test_historical_bars_ohlcv_timeout_backoff_throttles_retries() -> None:
    client = _new_client()
    calls = 0
    reconnects = 0

    def _fake_request_reconnect() -> None:
        nonlocal reconnects
        reconnects += 1

    async def _fake_request_for_stream(
        contract,
        *,
        end_ts: datetime | None = None,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        _ = end_ts
        nonlocal calls
        calls += 1
        con_id = int(getattr(contract, "conId", 0) or 0)
        payload = {
            "status": "timeout",
            "ts": "2026-04-16T00:00:00-04:00",
            "timeout_sec": 25.0,
            "request": {
                "duration_str": str(duration_str),
                "bar_size": str(bar_size),
                "what_to_show": str(what_to_show),
                "use_rth": bool(use_rth),
                "use_proxy": False,
            },
            "contract": {
                "con_id": int(con_id),
                "sec_type": str(getattr(contract, "secType", "") or "").strip().upper(),
                "symbol": str(getattr(contract, "symbol", "") or "").strip().upper(),
                "exchange": str(getattr(contract, "exchange", "") or "").strip().upper(),
                "primary_exchange": str(getattr(contract, "primaryExchange", "") or "").strip().upper(),
                "currency": str(getattr(contract, "currency", "") or "").strip().upper(),
            },
            "bars_count": 0,
            "detail": "simulated timeout",
        }
        client._store_historical_request_payload(payload)
        return []

    client._request_reconnect = _fake_request_reconnect  # type: ignore[method-assign]
    client._request_historical_data_for_stream = _fake_request_for_stream  # type: ignore[method-assign]

    contract = Future(symbol="MES", lastTradeDateOrContractMonth="202603", exchange="CME", currency="USD")
    contract.conId = 3001

    with patch("tradebot.client.time.monotonic", return_value=1000.0):
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
    with patch("tradebot.client.time.monotonic", return_value=1001.0):
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
    with patch("tradebot.client.time.monotonic", return_value=1006.0):
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
    with patch("tradebot.client.time.monotonic", return_value=1017.0):
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

    assert calls == 3
    assert reconnects == 1


def test_historical_bars_ohlcv_backoff_is_duration_scoped() -> None:
    client = _new_client()
    calls: list[str] = []

    async def _fake_request_for_stream(
        contract,
        *,
        end_ts: datetime | None = None,
        duration_str: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ):
        _ = contract, end_ts, bar_size, what_to_show, use_rth
        calls.append(str(duration_str))
        payload = {
            "status": "timeout",
            "ts": "2026-04-16T00:00:00-04:00",
            "timeout_sec": 25.0,
            "request": {
                "duration_str": str(duration_str),
                "bar_size": "10 mins",
                "what_to_show": "TRADES",
                "use_rth": False,
                "use_proxy": False,
            },
            "contract": {
                "con_id": 3002,
                "sec_type": "FUT",
                "symbol": "MES",
                "exchange": "CME",
                "primary_exchange": "CME",
                "currency": "USD",
            },
            "bars_count": 0,
            "detail": "simulated timeout",
        }
        client._store_historical_request_payload(payload)
        return []

    client._request_historical_data_for_stream = _fake_request_for_stream  # type: ignore[method-assign]

    contract = Future(symbol="MES", lastTradeDateOrContractMonth="202603", exchange="CME", currency="USD")
    contract.conId = 3002

    with patch("tradebot.client.time.monotonic", return_value=1000.0):
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
    # If the backoff key ignores duration_str, this call would be suppressed.
    with patch("tradebot.client.time.monotonic", return_value=1001.0):
        asyncio.run(
            client.historical_bars_ohlcv(
                contract,
                duration_str="2 D",
                bar_size="10 mins",
                use_rth=False,
                what_to_show="TRADES",
                cache_ttl_sec=30.0,
            )
        )

    assert calls == ["1 W", "2 D"]


class _ExitGateHarness(BotSignalRuntimeMixin):
    pass


class _EntryDayHarness(BotSignalRuntimeMixin):
    @staticmethod
    def _strategy_instrument(strategy: dict) -> str:
        value = strategy.get("instrument", "spot")
        cleaned = str(value or "spot").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    @staticmethod
    def _signal_bar_size(instance: _BotInstance) -> str:
        return str(instance.strategy.get("signal_bar_size") or "1 hour")

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


class _NoSignalRecoveryHarness(_GapGateHarness):
    def __init__(
        self,
        *,
        instance: _BotInstance,
        snapshots: list[object | None],
        diags: list[dict | None] | None = None,
    ) -> None:
        first_snap = snapshots[0] if snapshots else None
        first_diag = diags[0] if isinstance(diags, list) and diags else None
        super().__init__(instance=instance, snap=first_snap, diag=first_diag)
        self._snapshots = list(snapshots)
        self._diags = list(diags) if isinstance(diags, list) else []
        self._snap_idx = 0

    async def _signal_snapshot_for_contract(self, **kwargs):
        if self._snapshots:
            idx = min(self._snap_idx, len(self._snapshots) - 1)
            snap = self._snapshots[idx]
        else:
            snap = None
        if self._diags:
            idx = min(self._snap_idx, len(self._diags) - 1)
            diag = self._diags[idx]
            if isinstance(diag, dict):
                self._last_signal_snapshot_diag = dict(diag)
        self._snap_idx += 1
        return snap


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


class _PendingEntryGuardHarness(BotSignalRuntimeMixin):
    @staticmethod
    def _entry_limit_ok(instance: _BotInstance) -> bool:
        return True


class _EntryContextGateHarness(BotSignalRuntimeMixin):
    def __init__(self) -> None:
        self.queued: list[dict[str, object]] = []

    @staticmethod
    def _strategy_instrument(strategy: dict) -> str:
        value = strategy.get("instrument", "spot")
        cleaned = str(value or "spot").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    @staticmethod
    def _entry_limit_ok(instance: _BotInstance) -> bool:
        return True

    @staticmethod
    def _signal_bar_size(instance: _BotInstance) -> str:
        return str(instance.strategy.get("signal_bar_size") or "5 mins")

    @staticmethod
    def _signal_use_rth(instance: _BotInstance) -> bool:
        return bool(instance.strategy.get("signal_use_rth"))

    @staticmethod
    def _entry_direction_for_instance(instance: _BotInstance, snap) -> str | None:
        _ = (instance, snap)
        return "up"

    @staticmethod
    def _allowed_entry_directions(instance: _BotInstance) -> set[str]:
        _ = instance
        return {"up"}

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
                "instance_id": int(instance.instance_id),
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


def test_exit_gate_skips_holding_while_pending_exit_next_open_active() -> None:
    harness = _ExitGateHarness()
    harness._should_exit_on_dte = lambda *args, **kwargs: False  # type: ignore[attr-defined]
    harness._options_position_values = lambda *args, **kwargs: (None, None)  # type: ignore[attr-defined]
    harness._should_exit_on_flip = lambda *args, **kwargs: False  # type: ignore[attr-defined]
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance(strategy={})
    instance.pending_exit_due_ts = datetime(2026, 2, 9, 11, 0)
    gates: list[str] = []

    async def _run() -> bool:
        return await harness._auto_maybe_exit_open_positions(
            instance=instance,
            snap=SimpleNamespace(bar_ts=bar_ts, close=74.2),
            instrument="options",
            open_items=[object()],
            open_dir="up",
            now_et=datetime(2026, 2, 9, 10, 1),
            gate=lambda status, data=None: gates.append(str(status)),
        )

    assert asyncio.run(_run()) is False
    assert "HOLDING" not in gates


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


def test_entry_weekday_allows_numeric_entry_days_from_champion_payload() -> None:
    harness = _EntryDayHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "entry_days": [0, 1, 2, 3, 4],
        }
    )
    sunday_overnight = datetime(2026, 2, 8, 23, 32)
    assert harness._entry_weekday_for_ts(instance, sunday_overnight) == 0
    assert harness._can_order_now(instance, now_et=sunday_overnight) is True


def test_mcl_research_codec_preserves_futures_identity_for_backtest_and_live_weekday() -> None:
    base = _bundle_base(
        symbol="MCL",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size="5 mins",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
        entry_signal="ema",
    )
    source = spot_strategy_payload(
        base,
        meta=SimpleNamespace(exchange="NYMEX"),
    )
    source.update(
        {
            "spot_sec_type": "FUT",
            "spot_exchange": "NYMEX",
            "spot_next_open_session": "tradable_24x5",
        }
    )

    typed = strategy_from_payload(dict(source), filters=base.strategy.filters)
    encoded = spot_strategy_payload(
        replace(base, strategy=typed),
        meta=SimpleNamespace(exchange="NYMEX"),
    )
    instance = _new_instance(strategy=encoded)
    sunday_overnight = datetime(2026, 2, 8, 23, 32)

    observed = {
        "typed_sec_type": getattr(typed, "spot_sec_type", None),
        "typed_exchange": getattr(typed, "spot_exchange", None),
        "encoded_sec_type": encoded.get("spot_sec_type"),
        "encoded_exchange": encoded.get("spot_exchange"),
        "backtest_sec_type": _spot_strategy_sec_type(strategy=typed),
        "live_weekday": _EntryDayHarness()._entry_weekday_for_ts(
            instance,
            sunday_overnight,
        ),
    }
    assert observed == {
        "typed_sec_type": "FUT",
        "typed_exchange": "NYMEX",
        "encoded_sec_type": "FUT",
        "encoded_exchange": "NYMEX",
        "backtest_sec_type": "FUT",
        "live_weekday": 6,
    }


def test_bot_config_defaults_mcl_spot_identity_from_canonical_registry() -> None:
    from tradebot.ui.bot_screen.config import BotConfigScreen

    def _build(symbol: str, *, supplied: str | None = None) -> str:
        strategy = {"instrument": "spot", "symbol": symbol}
        if supplied is not None:
            strategy["spot_sec_type"] = supplied
        holder = SimpleNamespace(
            _strategy=strategy,
            _symbol=symbol,
            _filters={},
            _fields=[],
        )
        BotConfigScreen._build_fields(holder)
        return str(holder._strategy.get("spot_sec_type") or "")

    assert _build("MCL") == "FUT"
    assert _build("MNQ") == "FUT"
    assert _build("SLV") == "STK"
    assert _build("MCL", supplied="STK") == "STK"


def test_canonical_contract_identity_registry_covers_mcl_mnq_and_equity_controls() -> None:
    from tradebot.contract_identity import (
        future_exchange_for_symbol,
        future_multiplier_for_symbol,
        is_future_symbol,
        normalize_contract_symbol,
    )

    assert normalize_contract_symbol(" mcl ") == "MCL"
    assert future_exchange_for_symbol("mcl") == "NYMEX"
    assert is_future_symbol("MCL") is True
    assert future_multiplier_for_symbol("MCL") == 100.0

    assert future_exchange_for_symbol("mnq") == "CME"
    assert is_future_symbol("MNQ") is True
    assert future_multiplier_for_symbol("MNQ") == 2.0

    assert future_exchange_for_symbol("slv") is None
    assert is_future_symbol("SLV") is False
    assert future_multiplier_for_symbol("SLV") == 1.0


def test_signal_preflight_requires_regime2_supertrend_warmup_for_tqqq_hf_style_payload() -> None:
    harness = _EntryDayHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "5 mins",
            "signal_use_rth": True,
            "ema_preset": "5/13",
            "entry_signal": "ema",
            "regime_mode": "ema",
            "regime_bar_size": "5 mins",
            "regime2_mode": "supertrend",
            "regime2_bar_size": "30 mins",
            "regime2_supertrend_atr_period": 10,
            "regime2_bear_hard_mode": "supertrend",
            "regime2_bear_hard_bar_size": "4 hours",
            "regime2_bear_hard_supertrend_atr_period": 10,
        },
        filters={},
    )
    req = harness._signal_preflight_requirements(instance)
    assert req["signal_bars_min"] >= 13
    assert req["regime2_bars_min"] >= 60


def test_signal_preflight_does_not_require_router_daily_warmup_for_router_enabled_payload() -> None:
    harness = _EntryDayHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "5 mins",
            "signal_use_rth": True,
            "entry_signal": "ema",
            "ema_preset": "5/13",
            "regime_router": True,
            "regime_router_slow_window_days": 84,
        },
        filters={},
    )
    req = harness._signal_preflight_requirements(instance)
    assert req["signal_bars_min"] >= 13
    assert req["signal_bars_min"] < 1000
    active = list(req.get("active_requirements") or [])
    assert not any(str(item.get("name")) == "regime_router_daily" for item in active)


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


def test_auto_try_queue_entry_skips_recheck_when_pending_next_open_exists() -> None:
    harness = _PendingEntryGuardHarness()
    instance = _new_instance(strategy={"instrument": "spot"})
    instance.pending_entry_due_ts = datetime(2026, 2, 9, 4, 0)
    instance.pending_entry_direction = "down"
    instance.pending_entry_signal_bar_ts = datetime(2026, 2, 9, 3, 42)
    gates: list[str] = []

    fired = harness._auto_try_queue_entry(
        instance=instance,
        snap=SimpleNamespace(bar_ts=datetime(2026, 2, 9, 3, 50)),
        gate=lambda status, data=None: gates.append(str(status)),
        now_et=datetime(2026, 2, 9, 3, 50),
    )

    assert fired is False
    assert gates == []


def test_auto_try_queue_entry_passes_entry_context_into_graph_gate() -> None:
    harness = _EntryContextGateHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "5 mins",
            "signal_use_rth": True,
            "spot_entry_policy": "slope_tr_guard",
            "spot_entry_context_confidence_mode": "continuation_v1",
        },
        filters={},
    )
    gates: list[str] = []

    fired = harness._auto_try_queue_entry(
        instance=instance,
        snap=SimpleNamespace(
            bar_ts=datetime(2026, 2, 9, 10, 0),
            close=100.0,
            atr=None,
            entry_branch="a",
            regime4_state="trend_up_clean",
            shock_dir="down",
            regime2_bear_hard_dir="up",
            regime2_bear_hard_release_age_bars=1000,
            regime_router_host_managed=False,
            regime_router_bull_sovereign_ok=False,
        ),
        gate=lambda status, data=None: gates.append(str(status)),
        now_et=datetime(2026, 2, 9, 10, 0),
    )

    assert fired is False
    assert not harness.queued
    assert "BLOCKED_GRAPH_ENTRY_CONTEXT_CONFIDENCE" in gates


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


def test_stock_session_flags_include_overnight_gap_between_0350_and_0400() -> None:
    outside_rth, include_overnight = _session_flags(datetime(2026, 2, 10, 3, 45))
    assert outside_rth is False
    assert include_overnight is True

    outside_rth, include_overnight = _session_flags(datetime(2026, 2, 10, 3, 55))
    assert outside_rth is False
    assert include_overnight is False

    outside_rth, include_overnight = _session_flags(datetime(2026, 2, 10, 4, 1))
    assert outside_rth is True
    assert include_overnight is False


def test_next_open_due_aligns_to_0400_after_stock_overnight_gap() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "1 min",
            "spot_exec_bar_size": "1 min",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
        }
    )
    due = harness._spot_next_open_due_ts(instance, datetime(2026, 2, 10, 3, 49))
    assert due == datetime(2026, 2, 10, 4, 0)


def test_next_open_due_auto_full24_keeps_stock_overnight_window() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "10 mins",
            "spot_exec_bar_size": "5 mins",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
        }
    )
    due = harness._spot_next_open_due_ts(instance, datetime(2026, 2, 10, 21, 20))
    assert due == datetime(2026, 2, 10, 21, 30)


def test_next_open_due_tradable_24x5_keeps_stock_overnight_window() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "10 mins",
            "spot_exec_bar_size": "5 mins",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "spot_next_open_session": "tradable_24x5",
        }
    )
    due = harness._spot_next_open_due_ts(instance, datetime(2026, 2, 10, 21, 20))
    assert due == datetime(2026, 2, 10, 21, 30)


def test_next_open_due_tradable_24x5_respects_overnight_gap() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "1 min",
            "spot_exec_bar_size": "1 min",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "spot_next_open_session": "tradable_24x5",
        }
    )
    due = harness._spot_next_open_due_ts(instance, datetime(2026, 2, 10, 3, 49))
    assert due == datetime(2026, 2, 10, 4, 0)


def test_next_open_due_tradable_24x5_rolls_friday_post_close_to_sunday_2000() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "1 min",
            "spot_exec_bar_size": "1 min",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "spot_next_open_session": "tradable_24x5",
        }
    )
    due = harness._spot_next_open_due_ts(instance, datetime(2026, 2, 13, 20, 4))
    assert due == datetime(2026, 2, 15, 20, 0)


def test_schedule_next_bar_uses_immediate_exec_boundary_not_session_open() -> None:
    harness = _PendingNextOpenHarness()
    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "signal_bar_size": "10 mins",
            "spot_exec_bar_size": "5 mins",
            "signal_use_rth": False,
            "spot_sec_type": "STK",
            "spot_next_open_session": "extended",
        }
    )
    gate_events: list[tuple[str, dict | None]] = []
    fired = harness._schedule_pending_entry_next_open(
        instance=instance,
        direction="up",
        fill_mode="next_bar",
        signal_bar_ts=datetime(2026, 2, 9, 21, 20),
        now_wall=datetime(2026, 2, 9, 21, 29),
        gate=lambda status, data=None: gate_events.append((str(status), dict(data or {}))),
    )
    assert fired is True
    assert gate_events
    status, payload = gate_events[-1]
    assert status == "PENDING_ENTRY_NEXT_OPEN"
    assert payload.get("next_open_due") == "2026-02-09T21:30:00"
    assert payload.get("next_open_due_from") == "2026-02-09T21:30:00"


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
        fill_mode="next_tradable_bar",
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
    signal_msg = str(signal_entry["msg"])
    assert "dir=[bold #ffd166]⟪[/]" in signal_msg
    assert "bias=[bold #ffd166]⟪[/]" in signal_msg
    assert "time=11:10" in signal_msg
    assert "lag:2b" in signal_msg
    assert "stale=[bold #73d89e]✓[/]" in signal_msg
    assert "gap=[bold #73d89e]✓[/]" in signal_msg
    assert "active_knobs" in signal_msg
    assert "shock_gate=detect" in signal_msg
    assert "shock_scale=daily_atr_pct" in signal_msg

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


def test_journal_signal_entry_shows_block_emoji_on_regime_veto() -> None:
    signal_entry = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="SIGNAL",
        reason=None,
        instance_id="1",
        symbol="SLV",
        extra={},
        detail={
            "bar_ts": "2026-02-09T11:10:00",
            "signal": {"state": "up", "entry_dir": None, "regime_dir": "down", "regime_ready": True},
        },
    )
    signal_msg = str(signal_entry["msg"])
    assert "entry=[bold #ff5f87]🚫[/]" in signal_msg


def test_journal_gate_formats_filter_map_and_lifecycle_context() -> None:
    gate_entry = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="GATE",
        reason="BLOCKED_FILTERS",
        instance_id="1",
        symbol="SLV",
        extra={
            "filters": {
                "ema_spread_min_pct": 0.3,
                "ema_slope_min_pct": 0.1,
                "volume_ratio_min": 1.2,
            },
        },
        detail={
            "bar_ts": "2026-02-09T11:10:00",
            "bar_hour_et": 11,
            "bars_in_day": 4,
            "state": "up",
            "entry_dir": "up",
            "regime_dir": "down",
            "rv": 12.3,
            "cooldown_ok": True,
            "volume": 820.0,
            "volume_ema": 1000.0,
            "volume_ema_ready": True,
            "ema_spread_pct": 0.238,
            "ema_slope_pct": 0.068,
            "filter_checks": {
                "rv": True,
                "time": True,
                "skip_first": True,
                "cooldown": True,
                "shock_gate": True,
                "permission": False,
                "volume": False,
            },
            "failed_filters": ["permission", "volume"],
            "entry_ctx": {
                "entry_dir": "up",
                "allowed_directions": ["down", "up"],
                "direction_ok": True,
                "entry_capacity": False,
                "entries_today": 1,
                "max_entries_per_day": 1,
            },
            "next_open_ctx": {
                "allowed": False,
                "reason": "next_open_not_allowed",
                "due_ts": "2026-02-09T20:00:00",
                "fill_mode": "next_tradable_bar",
            },
            "spot_lifecycle": {
                "trace": {
                    "stage": "flat",
                    "path": "hold",
                    "graph_entry": {
                        "gate": "BLOCKED_GRAPH_ENTRY_TR_RATIO",
                        "reason": "graph_entry_tr_ratio",
                        "trace": {
                            "policy": "slope_tr_guard",
                            "tr_ratio": 1.12,
                            "min": 1.30,
                            "graph": {"profile": "defensive"},
                        },
                    },
                }
            },
        },
    )
    msg = str(gate_entry["msg"])
    assert "filter_map" in msg
    assert "filter_fail" in msg
    assert "permission:" in msg
    assert "volume:" in msg
    assert "entry_ctx" in msg
    assert "next_open_ctx" in msg
    assert "graph_entry" in msg


def test_journal_gate_health_token_distinguishes_signal_vs_regime_blockers() -> None:
    signal_gate = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="GATE",
        reason="WAITING_DATA_GAP",
        instance_id="1",
        symbol="SLV",
        extra={},
        detail={
            "bar_ts": "2026-02-09T11:10:00",
            "bar_health": {"stale": False, "gap_detected": True},
        },
    )
    regime_gate = BotJournal._in_app_entry(
        now_et=datetime(2026, 2, 9, 11, 29),
        event="GATE",
        reason="WAITING_REGIME_HEALTH",
        instance_id="1",
        symbol="SLV",
        extra={},
        detail={
            "bar_ts": "2026-02-09T11:10:00",
            "bar_health": {"stale": False, "gap_detected": False},
            "regime_bar_health": {"stale": True, "gap_detected": False},
        },
    )
    assert "health_gate=signal(gap)" in str(signal_gate["msg"])
    assert "health_gate=regime(stale)" in str(regime_gate["msg"])


def test_active_knob_tokens_ignore_max_tokens_limit() -> None:
    kwargs = dict(
        meta={
            "entry_signal": "ema",
            "signal_bar_size": "5 mins",
            "signal_use_rth": False,
            "regime_mode": "ema",
            "regime_bar_size": "10 mins",
            "regime2_mode": "ema",
            "regime2_bar_size": "15 mins",
            "regime2_apply_to": "both",
            "spot_exec_feed_mode": "ticks_side",
            "spot_exec_bar_size": "5 mins",
            "shock_gate_mode": "detect",
            "shock_detector": "tr_ratio",
            "shock_scale_detector": "daily_atr_pct",
        },
        strategy={
            "ema_preset": "5/13",
            "regime_ema_preset": "20/50",
            "regime2_ema_preset": "8/21",
            "spot_resize_policy": "off",
            "spot_risk_overlay_policy": "none",
            "tick_gate_mode": "off",
        },
        filters={
            "riskoff_mode": "directional",
            "shock_gate_mode": "detect",
            "shock_detector": "tr_ratio",
            "shock_scale_detector": "daily_atr_pct",
        },
    )
    tokens_low_limit = JournalDiagnostics._active_knob_tokens(**kwargs, max_tokens=2)
    tokens_high_limit = JournalDiagnostics._active_knob_tokens(**kwargs, max_tokens=99)

    assert tokens_low_limit == tokens_high_limit
    assert len(tokens_low_limit) > 2
    assert "execution=ticks_side@5m" in tokens_low_limit
    assert "shock_scale=daily_atr_pct" in tokens_low_limit


def test_bot_journal_persists_repeat_count_with_repeat_from_ts(tmp_path) -> None:
    journal = BotJournal(tmp_path)
    payload = {"mode": "spot", "reason": "flip", "next_open_due_from": "2026-02-09T11:20:00"}
    for _ in range(3):
        journal.write(
            event="GATE",
            instance=None,
            order=None,
            reason="PENDING_EXIT_NEXT_OPEN",
            data=dict(payload),
            strategy_instrument=lambda _: "spot",
        )
    journal.close()

    path = journal.path
    assert path is not None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    data_json = json.loads(row.get("data_json") or "{}")
    assert data_json.get("next_open_due_from") == "2026-02-09T11:20:00"
    assert int(data_json.get("log_repeat_count") or 0) == 3
    assert str(data_json.get("log_repeat_from_ts_et") or "")
    assert str(data_json.get("log_repeat_from_ts_utc") or "")
    assert str(data_json.get("log_repeat_from_ts_et") or "") <= str(row.get("ts_et") or "")


def test_bot_journal_repeat_compaction_requires_full_row_identity(tmp_path) -> None:
    journal = BotJournal(tmp_path)
    journal.write(
        event="GATE",
        instance=None,
        order=None,
        reason="HOLDING",
        data={"direction": "up", "items": 1},
        strategy_instrument=lambda _: "spot",
    )
    journal.write(
        event="GATE",
        instance=None,
        order=None,
        reason="HOLDING",
        data={"direction": "down", "items": 1},
        strategy_instrument=lambda _: "spot",
    )
    journal.close()

    path = journal.path
    assert path is not None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    first_payload = json.loads(rows[0].get("data_json") or "{}")
    second_payload = json.loads(rows[1].get("data_json") or "{}")
    assert first_payload.get("direction") == "up"
    assert second_payload.get("direction") == "down"
    assert "log_repeat_count" not in first_payload
    assert "log_repeat_count" not in second_payload


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


def test_auto_tick_blocks_waiting_regime_health_when_flat() -> None:
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
        atr=None,
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
        regime_bar_health={
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
        event == "GATE" and reason == "WAITING_REGIME_HEALTH"
        for event, reason, _data in harness._events
    )


def test_auto_tick_does_not_block_regime_health_on_sunday_daily_carry() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    screen = BotScreen(client=SimpleNamespace(), refresh_sec=1.0)

    class _TsBar:
        def __init__(self, ts: datetime) -> None:
            self.ts = ts

    regime_health = screen._signal_bar_health(
        bars=[
            _TsBar(datetime(2026, 2, 18, 0, 0)),
            _TsBar(datetime(2026, 2, 19, 0, 0)),
            _TsBar(datetime(2026, 2, 20, 0, 0)),
        ],
        bar_size="1 day",
        now_ref=datetime(2026, 2, 22, 22, 49),
        use_rth=False,
        sec_type="STK",
        source="TRADES",
        strict_zero_gap=True,
    )
    assert regime_health["stale"] is False

    bar_ts = datetime(2026, 2, 22, 22, 30)
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
        atr=None,
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
        regime_bar_health=regime_health,
    )
    instance = _new_instance(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "5/13",
            "regime_mode": "supertrend",
            "regime_bar_size": "1 day",
            "instrument": "spot",
        },
        filters={
            "entry_start_hour_et": 9,
            "entry_end_hour_et": 10,
        },
    )
    harness = _GapGateHarness(instance=instance, snap=snap)

    asyncio.run(harness._auto_order_tick())

    assert not any(
        event == "GATE" and reason == "WAITING_REGIME_HEALTH"
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


def test_no_signal_recovered_is_logged_on_transition_to_live_snapshot() -> None:
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
    live_snap = SimpleNamespace(
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
    harness = _NoSignalRecoveryHarness(
        instance=instance,
        snapshots=[None, live_snap],
        diags=[
            {"stage": "signal_bars"},
            {"stage": "ok", "bars_count": 5, "regime_bars_count": 3, "regime2_bars_count": 0},
        ],
    )

    asyncio.run(harness._auto_order_tick())
    asyncio.run(harness._auto_order_tick())

    assert any(
        event == "GATE" and reason == "NO_SIGNAL_SNAPSHOT"
        for event, reason, _data in harness._events
    )
    assert any(
        event == "GATE" and reason == "SIGNAL_RECOVERED"
        for event, reason, _data in harness._events
    )


def test_signal_unrecovered_is_logged_after_threshold() -> None:
    instance = _new_instance(
        strategy={
            "entry_signal": "ema",
            "ema_preset": "5/13",
            "instrument": "spot",
            "signal_unrecovered_alert_sec": 60,
            "signal_unrecovered_repeat_sec": 60,
        }
    )
    harness = _NoSignalRecoveryHarness(
        instance=instance,
        snapshots=[None, None],
        diags=[{"stage": "signal_bars"}, {"stage": "signal_bars"}],
    )
    t0 = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 2, 9, 12, 1, 5, tzinfo=timezone.utc)
    with patch.object(bot_signal_runtime_module, "_now_et", side_effect=[t0, t1]):
        asyncio.run(harness._auto_order_tick())
        asyncio.run(harness._auto_order_tick())

    assert any(
        event == "GATE" and reason == "NO_SIGNAL_PERSISTING"
        for event, reason, _data in harness._events
    )
    assert any(
        event == "GATE" and reason == "SIGNAL_UNRECOVERED"
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
    intent: str = "exit",
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
        intent=str(intent),
        signal_bar_ts=signal_bar_ts,
        trade=_FakeTrade(status, why_held=why_held, log_messages=log_messages),
    )


def _broker_trade(
    *,
    order_id: int,
    perm_id: int,
    status: str,
    filled: float,
    remaining: float,
    done: bool,
):
    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    return SimpleNamespace(
        order=SimpleNamespace(
            orderId=int(order_id),
            permId=int(perm_id),
            lmtPrice=1.0,
        ),
        orderStatus=SimpleNamespace(
            status=str(status),
            whyHeld="",
            filled=float(filled),
            remaining=float(remaining),
        ),
        contract=contract,
        log=[],
        fills=[],
        isDone=lambda: bool(done),
    )


def test_resize_builder_defers_cooldown_timestamp_to_confirmed_fill() -> None:
    class _Trace:
        signed_qty_final = 2
        signed_qty_after_branch = 2

        @staticmethod
        def as_payload() -> dict[str, object]:
            return {"signed_qty_final": 2, "signed_qty_after_branch": 2}

        def with_branch_scaling(self, **kwargs):
            self.signed_qty_after_branch = int(kwargs.get("signed_qty_after_branch", 2))
            return self

    class _Intent:
        order_qty = 1
        order_action = "BUY"
        target_qty = 2

        @staticmethod
        def as_payload() -> dict[str, object]:
            return {
                "order_qty": 1,
                "order_action": "BUY",
                "target_qty": 2,
            }

    class _Lifecycle:
        intent = "resize"
        reason = "resize_target"
        spot_intent = _Intent()

        @staticmethod
        def as_payload() -> dict[str, object]:
            return {"intent": "resize", "reason": "resize_target"}

    contract = Stock(symbol="SLV", exchange="SMART", currency="USD")
    contract.conId = 1001
    contract.minTick = 0.01
    ticker = SimpleNamespace(
        contract=contract,
        bid=10.00,
        ask=10.02,
        last=10.01,
        minTick=0.01,
        marketDataType=1,
    )
    bar_ts = datetime(2026, 2, 9, 10, 0)
    prior_ts = datetime(2026, 2, 9, 9, 30)
    snapshot = SimpleNamespace(
        bar_ts=bar_ts,
        close=10.01,
        signal=SimpleNamespace(
            state="up",
            entry_dir="up",
            regime_dir="up",
            ema_ready=True,
        ),
        bars_in_day=12,
        rv=0.10,
        volume=1_000.0,
        shock=False,
        shock_dir=None,
        risk=None,
        atr=None,
        or_high=None,
        or_low=None,
        or_ready=False,
        entry_dir="up",
        entry_branch=None,
        shock_atr_pct=None,
        shock_atr_vel_pct=None,
        shock_atr_accel_pct=None,
        bar_health=None,
        regime_bar_health=None,
        regime2_bar_health=None,
    )

    class _BuilderClient:
        @staticmethod
        async def ensure_ticker(_contract, *, owner: str):
            assert owner == "bot"
            return ticker

        @staticmethod
        def account_value(tag: str, currency: str | None = None):
            value = 2_200.0 if tag == "NetLiquidation" else 1_800.0
            return value, currency or "USD", None

        @staticmethod
        async def convert_currency_value(value, *, from_currency: str, to_currency: str):
            assert from_currency == to_currency
            return float(value), 1.0

        @staticmethod
        def proxy_error():
            return None

    class _BuilderHarness(BotOrderBuilderMixin):
        def __init__(self) -> None:
            self._client = _BuilderClient()
            self._payload = None
            self._tracked_conids: set[int] = set()
            self._orders: list[_BotOrder] = []
            self._events: list[tuple[str, dict]] = []
            self._status = ""

        @staticmethod
        def _strategy_instrument(_strategy) -> str:
            return "spot"

        @staticmethod
        async def _signal_contract(_instance, _symbol):
            return contract

        @staticmethod
        def _signal_snapshot_kwargs(*args, **kwargs) -> dict[str, object]:
            return {}

        @staticmethod
        async def _signal_snapshot_for_contract(*args, **kwargs):
            return snapshot

        @staticmethod
        def _entry_direction_for_instance(_instance, _snapshot) -> str:
            return "up"

        @staticmethod
        async def _spot_contract(_instance, _symbol):
            return contract

        @staticmethod
        def _resolve_open_positions(*args, **kwargs):
            return "spot", [SimpleNamespace(position=1.0)], "up"

        @staticmethod
        def _signal_bar_size(_instance) -> str:
            return "1 hour"

        def _add_order(self, order: _BotOrder) -> None:
            self._orders.append(order)

        def _journal_write(self, *, event: str, data: dict | None = None, **kwargs) -> None:
            self._events.append((str(event), dict(data or {})))

        def _render_status(self) -> None:
            return None

    instance = _new_instance(
        strategy={
            "instrument": "spot",
            "entry_signal": "ema",
            "ema_preset": "9/21",
            "spot_resize_mode": "target",
            "spot_sizing_mode": "fixed",
            "spot_fixed_qty": 2,
        }
    )
    instance.open_direction = "up"
    instance.last_resize_bar_ts = prior_ts
    harness = _BuilderHarness()

    with (
        patch(
            "tradebot.ui.bot_order_builder.normalize_spot_entry_signal",
            return_value="ema",
        ),
        patch(
            "tradebot.ui.bot_order_builder.spot_runtime_spec_view",
            return_value=SimpleNamespace(exit_mode="pct"),
        ),
        patch(
            "tradebot.ui.bot_order_builder.spot_resolve_entry_action_qty",
            return_value=("BUY", 1),
        ),
        patch(
            "tradebot.ui.bot_order_builder.decide_open_position_intent",
            return_value=_Lifecycle(),
        ),
        patch(
            "tradebot.engine.spot_shock_exit_pct_multipliers",
            return_value=(1.0, 1.0),
        ),
        patch(
            "tradebot.engine.spot_scale_exit_pcts",
            return_value=(None, None),
        ),
        patch(
            "tradebot.engine.spot_calc_signed_qty_with_trace",
            return_value=(2, _Trace()),
        ),
        patch("tradebot.engine.spot_branch_size_mult", return_value=1.0),
        patch("tradebot.engine.spot_apply_branch_size_mult", return_value=2),
    ):
        asyncio.run(
            harness._create_order_for_instance(
                instance,
                intent="resize",
                direction="up",
                signal_bar_ts=bar_ts,
            )
        )

    assert len(harness._orders) == 1
    staged = harness._orders[0]
    assert staged.status == "STAGED"
    assert staged.intent == "resize"
    assert staged.signal_bar_ts == bar_ts
    assert staged.quantity == 1
    assert instance.last_resize_bar_ts == prior_ts


def _admission_preview() -> BrokerOrderPreview:
    return BrokerOrderPreview(
        status="PreSubmitted",
        init_margin_before=2200.0,
        init_margin_change=500.0,
        init_margin_after=1700.0,
        maintenance_margin_before=1800.0,
        maintenance_margin_change=400.0,
        maintenance_margin_after=1400.0,
        equity_with_loan_before=2200.0,
        equity_with_loan_change=-2.5,
        equity_with_loan_after=2197.5,
        commission=2.5,
        min_commission=1.0,
        max_commission=3.0,
        commission_currency="USD",
        warning_text="",
    )


def _admission_xsp_order(
    *,
    package_risk: OptionPackageRisk | None,
) -> _BotOrder:
    short_contract = Contract(
        secType="OPT",
        conId=1001,
        symbol="XSP",
        exchange="SMART",
        currency="USD",
    )
    long_contract = Contract(
        secType="OPT",
        conId=1002,
        symbol="XSP",
        exchange="SMART",
        currency="USD",
    )
    bag = Bag(
        symbol="XSP",
        exchange="SMART",
        currency="USD",
        comboLegs=[
            ComboLeg(conId=1001, ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=1002, ratio=1, action="BUY", exchange="SMART"),
        ],
    )
    return _BotOrder(
        instance_id=1,
        preset=None,
        underlying=Stock(symbol="XSP", exchange="SMART", currency="USD"),
        order_contract=bag,
        legs=[
            QualifiedOptionLeg(contract=short_contract, action="SELL", ratio=1),
            QualifiedOptionLeg(contract=long_contract, action="BUY", ratio=1),
        ],
        action="BUY",
        quantity=1,
        limit_price=-1.0,
        created_at=datetime(2026, 2, 9, 10, 0),
        package_risk=package_risk,
        intent="enter",
        direction="up",
        reason="enter",
        signal_bar_ts=datetime(2026, 2, 9, 10, 0),
    )


class _AdmissionSendClient:
    def __init__(self, preview: BrokerOrderPreview) -> None:
        self._config = SimpleNamespace(account="DU2200")
        self.preview = preview
        self.calls: list[tuple[str, object, str, float, float, bool]] = []
        self.trade = SimpleNamespace(
            order=SimpleNamespace(orderId=9101, permId=0),
        )

    async def preview_limit_order(
        self,
        contract,
        action: str,
        quantity: float,
        limit_price: float,
        outside_rth: bool,
    ):
        self.calls.append(
            (
                "preview",
                contract,
                str(action),
                float(quantity),
                float(limit_price),
                bool(outside_rth),
            )
        )
        return self.preview

    async def place_limit_order(
        self,
        contract,
        action: str,
        quantity: float,
        limit_price: float,
        outside_rth: bool,
    ):
        self.calls.append(
            (
                "place",
                contract,
                str(action),
                float(quantity),
                float(limit_price),
                bool(outside_rth),
            )
        )
        return self.trade


class _AdmissionSendHarness:
    def __init__(self, client: _AdmissionSendClient) -> None:
        self._client = client
        self._events: list[tuple[str, dict[str, object]]] = []
        self._status = ""

    def _journal_write(
        self,
        *,
        event: str,
        data: dict | None = None,
        **_kwargs,
    ) -> None:
        self._events.append((str(event), dict(data or {})))

    def _set_status(self, value: str, **_kwargs) -> None:
        self._status = str(value)

    def _refresh_orders_table(self) -> None:
        return None

    def _render_bot(self) -> None:
        return None


class _ReservationSummaryHarness:
    def __init__(self, orders: list[_BotOrder]) -> None:
        self._client = SimpleNamespace(
            _config=SimpleNamespace(account="DU2200"),
        )
        self._orders = list(orders)


def _bot_reservation_summary(harness: _ReservationSummaryHarness):
    summarize = getattr(BotScreen, "_order_reservation_summary", None)
    assert callable(
        summarize,
    ), "BotScreen._order_reservation_summary is missing"
    return summarize(harness)


def test_automated_xsp_admission_denies_missing_staged_risk_before_preview_or_placement(
    monkeypatch,
) -> None:
    captured: list[tuple[object, object]] = []

    def _recording_evaluate(request, facts):
        captured.append((request, facts))
        return evaluate_order_admission(request, facts)

    monkeypatch.setattr(
        live_options_module,
        "evaluate_order_admission",
        _recording_evaluate,
    )

    client = _AdmissionSendClient(_admission_preview())
    harness = _AdmissionSendHarness(client)
    order = _admission_xsp_order(package_risk=None)

    asyncio.run(BotScreen._send_order(harness, order))

    assert len(captured) == 1
    request, facts = captured[0]
    assert request.account == "DU2200"
    assert request.product_domain == "XSP"
    assert request.structure == ""
    assert request.sec_type == "BAG"
    assert request.symbol == "XSP"
    assert request.currency == "USD"
    assert request.exchange == "SMART"
    assert request.action == "BUY"
    assert request.quantity == 1
    assert request.limit_price == -1.0
    assert request.max_loss is None
    assert [
        (leg.con_id, leg.ratio, leg.action, leg.exchange)
        for leg in request.legs
    ] == [
        (1001, 1, "SELL", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    assert facts.status is None
    assert client.calls == []
    assert order.status == "BLOCKED"
    assert order.order_id is None
    assert order.trade is None
    assert order.sent_at is None
    assert [event for event, _data in harness._events] == ["ORDER_ADMISSION"]
    admission = harness._events[0][1]["admission"]
    assert admission["allow"] is False
    assert admission["reason"] == "structure_invalid"
    assert "structure_invalid" in harness._status


def test_every_non_xsp_bag_fails_closed_before_preview_or_placement() -> None:
    client = _AdmissionSendClient(_admission_preview())
    harness = _AdmissionSendHarness(client)
    order = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_credit",
            right="PUT",
            expiry="20260209",
            width=5.0,
            debit_value=-1.0,
            multiplier=100.0,
            quantity=1,
            max_loss=400.0,
        )
    )
    order.order_contract.symbol = "MCL"
    order.order_contract.exchange = "NYMEX"
    for leg in order.legs:
        leg.contract.symbol = "MCL"
        leg.contract.exchange = "NYMEX"

    asyncio.run(BotScreen._send_order(harness, order))

    assert client.calls == []
    assert order.status == "BLOCKED"
    assert order.error == "product_policy_unavailable"
    assert [event for event, _data in harness._events] == ["ORDER_ADMISSION"]


def test_automated_xsp_admission_previews_then_places_only_when_admitted(
    monkeypatch,
) -> None:
    captured: list[tuple[object, object]] = []

    def _recording_evaluate(request, facts):
        captured.append((request, facts))
        return evaluate_order_admission(request, facts)

    monkeypatch.setattr(
        live_options_module,
        "evaluate_order_admission",
        _recording_evaluate,
    )

    preview = _admission_preview()
    client = _AdmissionSendClient(preview)
    harness = _AdmissionSendHarness(client)
    order = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_credit",
            right="PUT",
            expiry="20260209",
            width=5.0,
            debit_value=-1.0,
            multiplier=100.0,
            quantity=1,
            max_loss=400.0,
        )
    )

    asyncio.run(BotScreen._send_order(harness, order))

    assert len(captured) == 2
    preflight_request, preflight_facts = captured[0]
    request, facts = captured[1]
    assert preflight_request == request
    assert preflight_facts.status is None
    assert request.account == "DU2200"
    assert request.product_domain == "XSP"
    assert request.structure == "vertical_credit"
    assert request.sec_type == "BAG"
    assert request.symbol == "XSP"
    assert request.currency == "USD"
    assert request.exchange == "SMART"
    assert request.action == "BUY"
    assert request.quantity == 1
    assert request.limit_price == -1.0
    assert request.max_loss == 400.0
    assert [
        (leg.con_id, leg.ratio, leg.action, leg.exchange)
        for leg in request.legs
    ] == [
        (1001, 1, "SELL", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    for field_name in preview.__dataclass_fields__:
        assert getattr(facts, field_name) == getattr(preview, field_name)

    assert [call[0] for call in client.calls] == ["preview", "place"]
    for _kind, contract, action, quantity, limit_price, outside_rth in client.calls:
        assert contract is order.order_contract
        assert action == "BUY"
        assert quantity == 1.0
        assert limit_price == -1.0
        assert outside_rth is False

    assert order.status == "WORKING"
    assert order.order_id == 9101
    assert order.trade is client.trade
    assert order.sent_at is not None
    assert [event for event, _data in harness._events] == [
        "ORDER_ADMISSION",
        "SENDING",
        "SENT",
    ]
    admission = harness._events[0][1]["admission"]
    assert admission["allow"] is True
    assert admission["reason"] == "broker_preview_admitted"


def test_automated_xsp_atomic_close_previews_then_places_with_exit_intent(
    monkeypatch,
) -> None:
    captured: list[tuple[object, object]] = []

    def _recording_evaluate(request, facts):
        captured.append((request, facts))
        return evaluate_order_admission(request, facts)

    monkeypatch.setattr(
        live_options_module,
        "evaluate_order_admission",
        _recording_evaluate,
    )

    preview = _admission_preview()
    client = _AdmissionSendClient(preview)
    harness = _AdmissionSendHarness(client)
    order = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_debit",
            right="PUT",
            expiry="20260209",
            width=5.0,
            debit_value=0.90,
            multiplier=100.0,
            quantity=1,
            max_loss=90.0,
        )
    )
    order.intent = "exit"
    order.reason = "exit"
    order.limit_price = 0.90
    order.legs = [
        QualifiedOptionLeg(
            contract=order.legs[0].contract,
            action="BUY",
            ratio=order.legs[0].ratio,
        ),
        QualifiedOptionLeg(
            contract=order.legs[1].contract,
            action="SELL",
            ratio=order.legs[1].ratio,
        ),
    ]
    order.order_contract.comboLegs[0].action = "BUY"
    order.order_contract.comboLegs[1].action = "SELL"

    asyncio.run(BotScreen._send_order(harness, order))

    assert len(captured) == 2
    preflight_request, preflight_facts = captured[0]
    request, facts = captured[1]
    assert preflight_request == request
    assert preflight_facts.status is None
    assert getattr(request, "intent", None) == "exit"
    assert request.account == "DU2200"
    assert request.product_domain == "XSP"
    assert request.structure == "vertical_debit"
    assert request.sec_type == "BAG"
    assert request.symbol == "XSP"
    assert request.currency == "USD"
    assert request.exchange == "SMART"
    assert request.action == "BUY"
    assert request.quantity == 1
    assert request.limit_price == 0.90
    assert request.max_loss == 90.0
    assert [
        (leg.con_id, leg.ratio, leg.action, leg.exchange)
        for leg in request.legs
    ] == [
        (1001, 1, "BUY", "SMART"),
        (1002, 1, "SELL", "SMART"),
    ]
    for field_name in preview.__dataclass_fields__:
        assert getattr(facts, field_name) == getattr(preview, field_name)

    assert [call[0] for call in client.calls] == ["preview", "place"]
    for _kind, contract, action, quantity, limit_price, outside_rth in client.calls:
        assert contract is order.order_contract
        assert action == "BUY"
        assert quantity == 1.0
        assert limit_price == 0.90
        assert outside_rth is False

    assert order.status == "WORKING"
    assert order.order_id == 9101
    assert order.trade is client.trade
    assert order.sent_at is not None
    assert [event for event, _data in harness._events] == [
        "ORDER_ADMISSION",
        "SENDING",
        "SENT",
    ]
    admission = harness._events[0][1]["admission"]
    assert admission["allow"] is True
    assert admission["reason"] == "broker_preview_admitted"
    assert admission["trace"]["intent"] == "exit"


def test_bot_reservation_summary_maps_active_local_xsp_orders() -> None:
    staged = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_credit",
            right="PUT",
            expiry="20260209",
            width=5.0,
            debit_value=-1.0,
            multiplier=100.0,
            quantity=1,
            max_loss=400.0,
        )
    )
    staged.status = "STAGED"

    canceling = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_credit",
            right="PUT",
            expiry="20260209",
            width=3.0,
            debit_value=-0.5,
            multiplier=100.0,
            quantity=1,
            max_loss=250.0,
        )
    )
    canceling.status = "CANCELING"

    terminal = _admission_xsp_order(
        package_risk=OptionPackageRisk(
            structure="vertical_credit",
            right="PUT",
            expiry="20260209",
            width=10.0,
            debit_value=-1.0,
            multiplier=100.0,
            quantity=1,
            max_loss=900.0,
        )
    )
    terminal.status = "FILLED"

    summary = _bot_reservation_summary(
        _ReservationSummaryHarness([staged, canceling, terminal])
    )

    assert summary.as_payload() == {
        "account": "DU2200",
        "active_count": 2,
        "unknown_active_count": 0,
        "reserved_max_loss": 650.0,
        "complete": True,
        "reason": "reservation_complete",
    }


def test_bot_reservation_summary_fails_closed_for_active_xsp_without_staged_risk() -> None:
    order = _admission_xsp_order(package_risk=None)
    order.status = "STAGED"

    summary = _bot_reservation_summary(
        _ReservationSummaryHarness([order])
    )

    assert summary.as_payload() == {
        "account": "DU2200",
        "active_count": 1,
        "unknown_active_count": 1,
        "reserved_max_loss": None,
        "complete": False,
        "reason": "active_risk_unknown",
    }


def test_resize_send_error_preserves_last_successful_cooldown_timestamp() -> None:
    class _FailingClient:
        @staticmethod
        async def place_limit_order(*args, **kwargs):
            raise RuntimeError("send boom")

    class _SendHarness:
        def __init__(self, *, instance: _BotInstance) -> None:
            self._client = _FailingClient()
            self._instances = [instance]
            self._events: list[tuple[str, dict]] = []
            self._status = ""

        def _journal_write(self, *, event: str, data: dict | None = None, **kwargs) -> None:
            self._events.append((str(event), dict(data or {})))

        def _set_status(self, value: str, **kwargs) -> None:
            self._status = str(value)

        def _refresh_orders_table(self) -> None:
            return None

        def _render_bot(self) -> None:
            return None

    prior_ts = datetime(2026, 2, 9, 9, 30)
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.last_resize_bar_ts = prior_ts
    order = _new_order(status="Submitted", signal_bar_ts=bar_ts, intent="resize")
    order.status = "STAGED"
    harness = _SendHarness(instance=instance)

    asyncio.run(BotScreen._send_order(harness, order))

    assert order.status == "ERROR"
    assert instance.last_resize_bar_ts == prior_ts
    assert any(event == "SEND_ERROR" for event, _data in harness._events)


def test_engine_terminal_resize_failures_preserve_last_successful_cooldown_timestamp() -> None:
    prior_ts = datetime(2026, 2, 9, 9, 30)
    bar_ts = datetime(2026, 2, 9, 10, 0)
    expected_status = {
        "Cancelled": "CANCELLED",
        "ApiCancelled": "CANCELLED",
        "Inactive": "INACTIVE",
    }

    for broker_status, terminal_status in expected_status.items():
        instance = _new_instance()
        instance.last_resize_bar_ts = prior_ts
        order = _new_order(
            status=broker_status,
            signal_bar_ts=bar_ts,
            intent="resize",
        )
        harness = _EngineHarness(instance=instance, order=order)

        asyncio.run(harness._chase_orders_tick())

        assert order.status == terminal_status
        assert instance.last_resize_bar_ts == prior_ts


def test_engine_filled_resize_advances_cooldown_timestamp() -> None:
    prior_ts = datetime(2026, 2, 9, 9, 30)
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.last_resize_bar_ts = prior_ts
    order = _new_order(status="Filled", signal_bar_ts=bar_ts, intent="resize")
    harness = _EngineHarness(instance=instance, order=order)

    asyncio.run(harness._chase_orders_tick())

    assert order.status == "FILLED"
    assert instance.last_resize_bar_ts == bar_ts
    filled_payload = next(data for event, data in harness._events if event == "ORDER_FILLED")
    assert filled_payload.get("resize_bar_ts") == bar_ts.isoformat()
    assert filled_payload.get("resize_applied") is True


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


def test_engine_canceling_order_timeout_requires_broker_refresh_before_resume() -> None:
    async def _run_case(
        *,
        current_status: str,
        reconciled_status: str,
    ):
        trade = _broker_trade(
            order_id=901,
            perm_id=456901,
            status="Submitted",
            filled=0.0,
            remaining=1.0,
            done=False,
        )
        bar_ts = datetime(2026, 2, 9, 10, 0)
        instance = _new_instance()
        order = _new_order(
            status="Submitted",
            signal_bar_ts=bar_ts,
            order_id=901,
        )
        order.status = "CANCELING"
        order.cancel_requested_at = -999.0
        order.sent_at = None
        order.exec_mode = None
        order.chase_last_reprice_ts = None
        order.chase_quote_signature = None
        order.trade = trade

        class _Client:
            def __init__(self) -> None:
                self.current_calls = 0
                self.reconcile_force_flags: list[bool] = []
                self.modify_calls = 0

            @staticmethod
            def _payload(status: str):
                return {
                    "order_id": 901,
                    "perm_id": 456901,
                    "raw_status": str(status),
                    "effective_status": str(status),
                    "filled_qty": 0.0,
                    "remaining_qty": 1.0,
                    "executed_qty": 0.0,
                    "open_present": True,
                    "is_done": False,
                    "is_terminal": False,
                    "trade": trade,
                }

            def current_order_state(
                self,
                *,
                order_id: int = 0,
                perm_id: int = 0,
            ):
                assert int(order_id) == 901
                assert int(perm_id) in (0, 456901)
                self.current_calls += 1
                return self._payload(current_status)

            async def reconcile_order_state(
                self,
                *,
                order_id: int = 0,
                perm_id: int = 0,
                force: bool = False,
            ):
                assert int(order_id) == 901
                assert int(perm_id) in (0, 456901)
                self.reconcile_force_flags.append(bool(force))
                return self._payload(reconciled_status)

            async def modify_limit_order(self, current_trade, limit_price: float):
                self.modify_calls += 1
                current_trade.order.lmtPrice = float(limit_price)
                return current_trade

            @staticmethod
            def pop_order_error(
                _order_id: int,
                *,
                max_age_sec: float = 120.0,
            ):
                return None

        class _RepriceHarness(_EngineHarness):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self.reprice_calls = 0

            async def _reprice_order(self, target: _BotOrder, *, mode: str) -> bool:
                self.reprice_calls += 1
                target.exec_mode = str(mode)
                target.limit_price = float(target.limit_price) + 0.05
                return True

        client = _Client()
        harness = _RepriceHarness(
            instance=instance,
            order=order,
            client=client,
        )
        await harness._chase_orders_tick()
        return order, client, harness

    explicit_order, explicit_client, explicit_harness = asyncio.run(
        _run_case(
            current_status="PendingCancel",
            reconciled_status="PendingCancel",
        )
    )
    stale_order, stale_client, stale_harness = asyncio.run(
        _run_case(
            current_status="Submitted",
            reconciled_status="PendingCancel",
        )
    )
    active_order, active_client, active_harness = asyncio.run(
        _run_case(
            current_status="Submitted",
            reconciled_status="Submitted",
        )
    )

    assert explicit_order.status == "CANCELING"
    assert explicit_order.cancel_requested_at is not None
    assert explicit_client.current_calls == 1
    assert explicit_client.reconcile_force_flags == []
    assert explicit_harness.reprice_calls == 0
    assert explicit_client.modify_calls == 0

    assert stale_order.status == "CANCELING"
    assert stale_order.cancel_requested_at is not None
    assert stale_client.current_calls == 1
    assert stale_client.reconcile_force_flags == [True]
    assert stale_harness.reprice_calls == 0
    assert stale_client.modify_calls == 0

    assert active_order.status == "WORKING"
    assert active_order.cancel_requested_at is None
    assert active_client.current_calls == 1
    assert active_client.reconcile_force_flags == [True]
    assert active_harness.reprice_calls == 1
    assert active_client.modify_calls == 1
    active_events = [event for event, _data in active_harness._events]
    assert active_events.count("CANCEL_ACK_TIMEOUT") == 1


def test_engine_rebinds_partial_working_order_before_reprice_after_reconnect() -> None:
    stale_trade = _broker_trade(
        order_id=777,
        perm_id=456001,
        status="Submitted",
        filled=0.0,
        remaining=1.0,
        done=False,
    )
    rebound_trade = _broker_trade(
        order_id=888,
        perm_id=456001,
        status="Submitted",
        filled=0.5,
        remaining=0.5,
        done=False,
    )

    class _Client:
        def __init__(self) -> None:
            self.state_calls = 0
            self.modified_trade = None

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            self.state_calls += 1
            assert int(order_id) == 777
            assert int(perm_id) in (0, 456001)
            return {
                "order_id": 888,
                "perm_id": 456001,
                "effective_status": "Submitted",
                "filled_qty": 0.5,
                "remaining_qty": 0.5,
                "executed_qty": 0.5,
                "is_terminal": False,
                "trade": rebound_trade,
            }

        @staticmethod
        def pop_order_error(_order_id: int, *, max_age_sec: float = 120.0):
            return None

        async def modify_limit_order(self, trade, limit_price: float):
            self.modified_trade = trade
            trade.order.lmtPrice = float(limit_price)
            return trade

    class _RepriceHarness(_EngineHarness):
        @staticmethod
        def _order_quote_signature(_order: _BotOrder):
            return (1.0, 1.0, 1.0)

        async def _reprice_order(self, order: _BotOrder, *, mode: str) -> bool:
            assert mode == "OPTIMISTIC"
            order.limit_price = 1.05
            return True

    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    order = _new_order(status="Submitted", signal_bar_ts=bar_ts, order_id=777)
    order.trade = stale_trade
    client = _Client()
    harness = _RepriceHarness(instance=instance, order=order, client=client)

    asyncio.run(harness._chase_orders_tick())

    assert client.state_calls == 1
    assert order.order_id == 888
    assert order.trade is rebound_trade
    assert order.status == "WORKING"
    assert client.modified_trade is rebound_trade
    assert rebound_trade.orderStatus.filled == 0.5
    assert rebound_trade.orderStatus.remaining == 0.5


def test_engine_projects_rebound_terminal_fill_after_reconnect() -> None:
    stale_trade = _broker_trade(
        order_id=777,
        perm_id=456001,
        status="Submitted",
        filled=0.0,
        remaining=1.0,
        done=False,
    )
    rebound_trade = _broker_trade(
        order_id=888,
        perm_id=456001,
        status="Filled",
        filled=1.0,
        remaining=0.0,
        done=True,
    )

    class _Client:
        def __init__(self) -> None:
            self.state_calls = 0
            self.modify_calls = 0

        def current_order_state(self, *, order_id: int = 0, perm_id: int = 0):
            self.state_calls += 1
            assert int(order_id) == 777
            assert int(perm_id) in (0, 456001)
            return {
                "order_id": 888,
                "perm_id": 456001,
                "effective_status": "Filled",
                "filled_qty": 1.0,
                "remaining_qty": 0.0,
                "executed_qty": 1.0,
                "is_terminal": True,
                "trade": rebound_trade,
            }

        @staticmethod
        def pop_order_error(_order_id: int, *, max_age_sec: float = 120.0):
            return None

        async def modify_limit_order(self, trade, limit_price: float):
            self.modify_calls += 1
            return trade

    prior_ts = datetime(2026, 2, 9, 9, 30)
    bar_ts = datetime(2026, 2, 9, 10, 0)
    instance = _new_instance()
    instance.last_resize_bar_ts = prior_ts
    order = _new_order(
        status="Submitted",
        signal_bar_ts=bar_ts,
        intent="resize",
        order_id=777,
    )
    order.trade = stale_trade
    client = _Client()
    harness = _EngineHarness(instance=instance, order=order, client=client)

    asyncio.run(harness._chase_orders_tick())

    assert client.state_calls == 1
    assert client.modify_calls == 0
    assert order.order_id == 888
    assert order.trade is rebound_trade
    assert order.status == "FILLED"
    assert instance.last_resize_bar_ts == bar_ts
    filled_payload = next(data for event, data in harness._events if event == "ORDER_FILLED")
    assert filled_payload.get("resize_bar_ts") == bar_ts.isoformat()
    assert filled_payload.get("resize_applied") is True


def test_engine_terminal_partial_package_cancel_preserves_fill_facts() -> None:
    async def _run_case(*, filled_qty: float, remaining_qty: float):
        trade = _broker_trade(
            order_id=880,
            perm_id=456880,
            status="Cancelled",
            filled=float(filled_qty),
            remaining=float(remaining_qty),
            done=True,
        )
        bag = Bag(
            symbol="XSP",
            exchange="SMART",
            currency="USD",
            comboLegs=[
                ComboLeg(
                    conId=1001,
                    ratio=1,
                    action="SELL",
                    exchange="SMART",
                ),
                ComboLeg(
                    conId=1002,
                    ratio=1,
                    action="BUY",
                    exchange="SMART",
                ),
            ],
        )
        trade.contract = bag
        trade.order.totalQuantity = 2
        trade.order.action = "BUY"

        bar_ts = datetime(2026, 2, 9, 10, 0)
        instance = _new_instance()
        instance.last_entry_bar_ts = bar_ts
        order = _new_order(
            status="Submitted",
            signal_bar_ts=bar_ts,
            intent="enter",
            order_id=880,
        )
        order.status = "WORKING"
        order.quantity = 2
        order.action = "BUY"
        order.intent = "enter"
        order.order_contract = bag
        order.underlying = Stock(
            symbol="XSP",
            exchange="SMART",
            currency="USD",
        )
        order.trade = trade
        order.sent_at = None
        order.exec_mode = None
        order.cancel_requested_at = None

        class _Client:
            def current_order_state(
                self,
                *,
                order_id: int = 0,
                perm_id: int = 0,
            ):
                assert int(order_id) == 880
                assert int(perm_id) in (0, 456880)
                return {
                    "order_id": 880,
                    "perm_id": 456880,
                    "effective_status": "Cancelled",
                    "filled_qty": float(filled_qty),
                    "remaining_qty": float(remaining_qty),
                    "executed_qty": float(filled_qty),
                    "is_terminal": True,
                    "trade": trade,
                }

            @staticmethod
            def pop_order_error(
                _order_id: int,
                *,
                max_age_sec: float = 120.0,
            ):
                return None

            @staticmethod
            async def modify_limit_order(current_trade, _limit_price: float):
                return current_trade

            @staticmethod
            async def cancel_trade(_trade) -> None:
                return None

        harness = _EngineHarness(
            instance=instance,
            order=order,
            client=_Client(),
        )
        await harness._chase_orders_tick()

        payloads = [
            data
            for event, data in harness._events
            if event == "ORDER_CANCELLED"
        ]
        assert len(payloads) == 1
        return instance, order, dict(payloads[0])

    zero_instance, zero_order, zero_payload = asyncio.run(
        _run_case(filled_qty=0.0, remaining_qty=2.0)
    )
    partial_instance, partial_order, partial_payload = asyncio.run(
        _run_case(filled_qty=1.0, remaining_qty=1.0)
    )

    assert zero_order.status == "CANCELLED"
    assert partial_order.status == "CANCELLED"
    assert zero_order.quantity == partial_order.quantity == 2

    assert getattr(zero_order, "filled_qty", None) == 0.0
    assert getattr(zero_order, "remaining_qty", None) == 2.0
    assert getattr(zero_order, "executed_qty", None) == 0.0
    assert getattr(partial_order, "filled_qty", None) == 1.0
    assert getattr(partial_order, "remaining_qty", None) == 1.0
    assert getattr(partial_order, "executed_qty", None) == 1.0

    assert zero_payload["filled_qty"] == 0.0
    assert zero_payload["remaining_qty"] == 2.0
    assert zero_payload["executed_qty"] == 0.0
    assert partial_payload["filled_qty"] == 1.0
    assert partial_payload["remaining_qty"] == 1.0
    assert partial_payload["executed_qty"] == 1.0
    assert zero_payload != partial_payload

    assert zero_instance.last_entry_bar_ts is None
    assert partial_instance.last_entry_bar_ts == partial_order.signal_bar_ts


def test_order_error_cache_pop_and_expiry() -> None:
    client = _new_client()
    client._remember_order_error(1001, 201, "bad tif")
    payload = client.pop_order_error(1001)
    assert payload == {"code": 201, "message": "bad tif"}
    assert client.pop_order_error(1001) is None

    client._remember_order_error(1004, 110, "The price does not conform to the minimum price variation for this contract.")
    payload_110 = client.pop_order_error(1004)
    assert payload_110 is not None
    assert payload_110.get("code") == 110

    client._order_error_cache[1002] = (time.monotonic() - 600.0, 201, "stale")
    assert client.pop_order_error(1002, max_age_sec=120.0) is None

    client._remember_order_error(1003, 2104, "Market data farm connection is OK")
    assert client.pop_order_error(1003) is None


def test_account_value_currency_override_selects_requested_currency() -> None:
    client = _new_client()

    class _FakeIB:
        @staticmethod
        def accountValues(_account: str):
            return [
                SimpleNamespace(tag="BuyingPower", currency="BASE", value="7685.82"),
                SimpleNamespace(tag="BuyingPower", currency="USD", value="5470.74"),
            ]

    client._ib = _FakeIB()  # type: ignore[assignment]

    value_default, currency_default, _ = client.account_value("BuyingPower")
    assert value_default == 7685.82
    assert currency_default == "BASE"

    value_usd, currency_usd, _ = client.account_value("BuyingPower", currency="usd")
    assert value_usd == 5470.74
    assert currency_usd == "USD"


def test_convert_currency_value_uses_inverse_rate_when_direct_missing() -> None:
    client = _new_client()

    async def _fake_fx_rate(from_currency: str, to_currency: str, *, max_age_sec: float = 15.0):
        assert max_age_sec > 0
        if (str(from_currency).upper(), str(to_currency).upper()) == ("USD", "AUD"):
            return 1.4
        return None

    client.fx_rate = _fake_fx_rate  # type: ignore[method-assign]

    converted, rate = asyncio.run(
        client.convert_currency_value(7686.0, from_currency="AUD", to_currency="USD")
    )
    assert converted is not None
    assert rate is not None
    assert abs(float(rate) - (1.0 / 1.4)) < 1e-9
    assert abs(float(converted) - (7686.0 / 1.4)) < 1e-6


def test_convert_currency_value_prefers_account_exchange_rate() -> None:
    client = _new_client()

    def _fake_account_value(tag: str, *, currency: str | None = None):
        if str(tag) != "ExchangeRate":
            return None, None, None
        code = str(currency or "").strip().upper()
        if code == "AUD":
            return 1.0, "AUD", None
        if code == "USD":
            return 1.4, "USD", None
        return None, code or None, None

    async def _fx_should_not_be_used(*args, **kwargs):
        raise AssertionError("fx_rate fallback should not be used when account ExchangeRate is available")

    client.account_value = _fake_account_value  # type: ignore[method-assign]
    client.fx_rate = _fx_should_not_be_used  # type: ignore[method-assign]

    converted, rate = asyncio.run(
        client.convert_currency_value(7686.0, from_currency="AUD", to_currency="USD")
    )
    assert converted is not None
    assert rate is not None
    assert abs(float(rate) - (1.0 / 1.4)) < 1e-9
    assert abs(float(converted) - (7686.0 / 1.4)) < 1e-6


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


def test_place_limit_order_fop_uses_detail_ticker_ladder_when_cache_empty() -> None:
    class _FakeIB:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def placeOrder(self, contract, order):
            self.calls.append((contract, order))
            return SimpleNamespace(contract=contract, order=SimpleNamespace(orderId=456, permId=0))

    client = _new_client()
    fake_ib = _FakeIB()
    client._ib = fake_ib

    async def _fake_connect() -> None:
        return None

    async def _fake_prime(_contract, *, ticker=None):
        return ()

    client.connect = _fake_connect  # type: ignore[method-assign]
    client._prime_contract_price_increments = _fake_prime  # type: ignore[method-assign]

    contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    contract.conId = 901001
    contract.minTick = 0.05

    ladder = ((0.0, 0.05), (5.0, 0.25), (100.0, 0.5))
    ticker_contract = Contract(secType="FOP", symbol="MNQ", exchange="CME", currency="USD")
    ticker_contract.conId = 901001
    setattr(ticker_contract, "tbPriceIncrements", ladder)
    detail_ticker = SimpleNamespace(contract=ticker_contract, tbPriceIncrements=ladder)
    client._detail_tickers[int(contract.conId)] = (fake_ib, detail_ticker)

    asyncio.run(client.place_limit_order(contract, "BUY", 1, 115.70, outside_rth=False))

    _placed_contract, placed_order = fake_ib.calls[-1]
    assert abs(float(getattr(placed_order, "lmtPrice", 0.0)) - 115.5) < 1e-9


def test_preview_limit_order_uses_canonical_xsp_credit_bag_without_placement() -> None:
    class _FakeIB:
        def __init__(self) -> None:
            self.preview_calls: list[tuple[object, object]] = []
            self.place_calls: list[tuple[object, object]] = []

        async def whatIfOrderAsync(self, contract, order):
            self.preview_calls.append((contract, order))
            return SimpleNamespace(
                status="PreSubmitted",
                initMarginBefore="2200.00",
                initMarginChange="500.00",
                initMarginAfter="1700.00",
                maintMarginBefore="1800.00",
                maintMarginChange="400.00",
                maintMarginAfter="1400.00",
                equityWithLoanBefore="2200.00",
                equityWithLoanChange="-2.50",
                equityWithLoanAfter="2197.50",
                commission="2.50",
                minCommission="1.00",
                maxCommission="3.00",
                commissionCurrency="USD",
                warningText="",
            )

        def placeOrder(self, contract, order):
            self.place_calls.append((contract, order))
            raise AssertionError("preview must not place an order")

    client = _new_client()
    fake_ib = _FakeIB()
    client._ib = fake_ib  # type: ignore[assignment]

    async def _fake_connect() -> None:
        return None

    client.connect = _fake_connect  # type: ignore[method-assign]

    preview_limit_order = getattr(client, "preview_limit_order", None)
    prepare_limit_order = getattr(client, "_prepare_limit_order", None)
    assert callable(preview_limit_order), "IBKRClient.preview_limit_order is missing"
    assert callable(prepare_limit_order), "IBKRClient._prepare_limit_order is missing"

    prepare_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def _tracked_prepare(*args, **kwargs):
        prepare_calls.append((tuple(args), dict(kwargs)))
        return await prepare_limit_order(*args, **kwargs)

    client._prepare_limit_order = _tracked_prepare  # type: ignore[attr-defined, method-assign]

    bag = Bag(
        symbol="XSP",
        exchange="SMART",
        currency="USD",
        comboLegs=[
            ComboLeg(conId=1001, ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=1002, ratio=1, action="BUY", exchange="SMART"),
        ],
    )
    bag.minTick = 0.01

    preview = asyncio.run(
        preview_limit_order(
            bag,
            "BUY",
            1,
            -1.004,
            outside_rth=False,
        )
    )

    assert len(prepare_calls) == 1
    assert len(fake_ib.preview_calls) == 1
    assert fake_ib.place_calls == []

    preview_contract, preview_order = fake_ib.preview_calls[0]
    assert str(getattr(preview_contract, "secType", "")).upper() == "BAG"
    assert str(getattr(preview_contract, "symbol", "")).upper() == "XSP"
    assert [
        (int(leg.conId), int(leg.ratio), str(leg.action), str(leg.exchange))
        for leg in getattr(preview_contract, "comboLegs", [])
    ] == [
        (1001, 1, "SELL", "SMART"),
        (1002, 1, "BUY", "SMART"),
    ]
    assert str(getattr(preview_order, "action", "")).upper() == "BUY"
    assert float(getattr(preview_order, "totalQuantity", 0.0)) == 1.0
    assert abs(float(getattr(preview_order, "lmtPrice", 0.0)) - (-1.00)) < 1e-9
    assert str(getattr(preview_order, "tif", "")).upper() == "GTC"
    assert bool(getattr(preview_order, "outsideRth", False)) is False

    assert type(preview).__name__ == "BrokerOrderPreview"
    assert preview.status == "PreSubmitted"
    assert preview.init_margin_change == 500.0
    assert preview.commission == 2.5
    assert preview.commission_currency == "USD"
    assert not hasattr(preview, "admitted")


def test_order_preview_normalizes_provider_values_without_inventing_capacity() -> None:
    normalizer = getattr(IBKRClient, "_normalize_order_preview", None)
    assert callable(normalizer), "IBKRClient._normalize_order_preview is missing"

    raw = SimpleNamespace(
        status="  PreSubmitted  ",
        initMarginBefore="2200.50",
        initMarginChange="-500.25",
        initMarginAfter="1.7976931348623157e+308",
        maintMarginBefore="",
        maintMarginChange="nan",
        maintMarginAfter="1700.25",
        equityWithLoanBefore="inf",
        equityWithLoanChange="-10.50",
        equityWithLoanAfter=None,
        commission="2.75",
        minCommission="1.00",
        maxCommission="1.7976931348623157e+308",
        commissionCurrency="USD",
        warningText="  Margin impact estimate only  ",
    )

    preview = normalizer(raw)

    assert type(preview).__name__ == "BrokerOrderPreview"
    assert preview.status == "PreSubmitted"
    assert preview.init_margin_before == 2200.5
    assert preview.init_margin_change == -500.25
    assert preview.init_margin_after is None
    assert preview.maintenance_margin_before is None
    assert preview.maintenance_margin_change is None
    assert preview.maintenance_margin_after == 1700.25
    assert preview.equity_with_loan_before is None
    assert preview.equity_with_loan_change == -10.5
    assert preview.equity_with_loan_after is None
    assert preview.commission == 2.75
    assert preview.min_commission == 1.0
    assert preview.max_commission is None
    assert preview.commission_currency == "USD"
    assert preview.warning_text == "Margin impact estimate only"
    assert not hasattr(preview, "admitted")


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


def test_live_sizing_assembly_delegates_via_typed_payload() -> None:
    import ast
    from pathlib import Path

    source = Path("tradebot/ui/bot_order_builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _leaf(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _target_names(node) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }

    factory_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _leaf(node.func) == "spot_sizing_input"
    ]
    wrapper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _leaf(node.func) == "spot_calc_signed_qty_with_trace"
    ]
    assert len(wrapper_calls) == 1, "expected one live spot sizing delegation"
    assert len(factory_calls) == len(wrapper_calls), (
        "live sizing adapter must call spot_sizing_input once per kernel "
        f"delegation: factory={len(factory_calls)} kernel={len(wrapper_calls)}"
    )

    factory_targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Call) or _leaf(value.func) != "spot_sizing_input":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                factory_targets.update(_target_names(target))

    raw_fields = {
        "strategy", "filters", "action", "lot", "entry_price", "stop_price",
        "stop_loss_pct", "shock", "shock_dir", "shock_atr_pct",
        "shock_dir_down_streak_bars", "shock_drawdown_dist_on_pct",
        "shock_drawdown_dist_on_vel_pp", "shock_drawdown_dist_on_accel_pp",
        "shock_prearm_down_streak_bars", "shock_ramp", "riskoff", "risk_dir",
        "riskpanic", "riskpop", "risk", "signal_entry_dir",
        "signal_regime_dir", "regime2_dir", "regime2_ready", "equity_ref",
        "cash_ref", "policy_graph", "policy_config",
    }
    call = wrapper_calls[0]
    expressions = [*call.args, *(keyword.value for keyword in call.keywords)]
    uses_inline_factory = any(
        isinstance(child, ast.Call) and _leaf(child.func) == "spot_sizing_input"
        for expression in expressions
        for child in ast.walk(expression)
    )
    uses_factory_result = any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id in factory_targets
        for expression in expressions
        for child in ast.walk(expression)
    )
    assert uses_inline_factory or uses_factory_result, (
        "live sizing kernel must receive the typed factory result"
    )
    raw_keywords = {
        keyword.arg for keyword in call.keywords if keyword.arg is not None
    } & raw_fields
    assert not raw_keywords, (
        "live sizing kernel still receives raw adapter fields: "
        f"{sorted(raw_keywords)}"
    )


def test_live_pending_state_mutation_matches_shared_transition_table() -> None:
    import inspect
    from datetime import datetime, timedelta

    from tradebot.spot import lifecycle as lifecycle_module
    from tradebot.ui import bot_signal_runtime as live_runtime

    planner = getattr(lifecycle_module, "plan_pending_mutation", None)
    assert callable(planner), "missing canonical plan_pending_mutation seam"

    plan_type = getattr(lifecycle_module, "SpotPendingMutationPlan", None)
    assert plan_type is not None, "missing typed SpotPendingMutationPlan contract"

    apply_mutation = getattr(live_runtime.BotSignalRuntimeMixin, "_apply_pending_mutation", None)
    assert callable(apply_mutation), "missing live pending-mutation adapter"

    runtime_source = inspect.getsource(live_runtime.BotSignalRuntimeMixin._auto_process_pending_next_open)
    assert "plan_pending_mutation(" in runtime_source
    assert "_apply_pending_mutation(" in runtime_source

    now = datetime(2026, 7, 20, 14, 0)
    due_past = now - timedelta(minutes=1)
    due_future = now + timedelta(minutes=1)
    common = {
        "now_ts": now,
        "has_open": False,
        "open_dir": None,
        "pending_entry_dir": None,
        "pending_entry_set_date": None,
        "pending_entry_due_ts": None,
        "pending_exit_reason": None,
        "pending_exit_due_ts": None,
        "risk_overlay_enabled": False,
        "riskoff_today": False,
        "riskpanic_today": False,
        "riskpop_today": False,
        "riskoff_mode": "hygiene",
        "shock_dir_now": None,
        "riskoff_end_hour": None,
        "naive_ts_mode": "et",
    }
    cases = [
        ("no_pending", {}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("exit_wait", {"has_open": True, "open_dir": "up", "pending_exit_reason": "flip", "pending_exit_due_ts": due_future}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("exit_due_open", {"has_open": True, "open_dir": "up", "pending_exit_reason": "flip", "pending_exit_due_ts": due_past}, {"clear_entry": False, "clear_exit": True, "queue_intent": "exit", "queue_direction": "up", "queue_reason": "flip"}),
        ("exit_due_flat", {"pending_exit_reason": "flip", "pending_exit_due_ts": due_past}, {"clear_entry": False, "clear_exit": True, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_wait", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_due_flat", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_past}, {"clear_entry": True, "clear_exit": False, "queue_intent": "enter", "queue_direction": "up", "queue_reason": "next_open"}),
        ("entry_due_open", {"has_open": True, "open_dir": "up", "pending_entry_dir": "down", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_past}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_cancel_date_roll", {"pending_entry_dir": "up", "pending_entry_set_date": now.date() - timedelta(days=1), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_cancel_directional_mismatch", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True, "riskoff_mode": "directional", "shock_dir_now": "down"}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_directional_match_wait", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True, "riskoff_mode": "directional", "shock_dir_now": "up"}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
    ]

    for scenario, overrides, expected in cases:
        kwargs = dict(common)
        kwargs.update(overrides)
        decision = lifecycle_module.decide_pending_next_open(**kwargs)
        mutation = planner(
            decision,
            pending_entry_direction=kwargs["pending_entry_dir"],
            pending_exit_reason=kwargs["pending_exit_reason"],
            open_dir=kwargs["open_dir"],
        )
        assert isinstance(mutation, plan_type), scenario
        assert mutation.as_payload() == expected, scenario

        instance = _new_instance(strategy={"instrument": "spot"})
        if kwargs["pending_entry_dir"] in ("up", "down") and kwargs["pending_entry_due_ts"] is not None:
            instance.pending_entry_direction = str(kwargs["pending_entry_dir"])
            instance.pending_entry_signal_bar_ts = now
            instance.pending_entry_due_ts = kwargs["pending_entry_due_ts"]
        if kwargs["pending_exit_due_ts"] is not None:
            instance.pending_exit_reason = str(kwargs["pending_exit_reason"] or "flip")
            instance.pending_exit_signal_bar_ts = now
            instance.pending_exit_due_ts = kwargs["pending_exit_due_ts"]

        before_entry = (
            instance.pending_entry_direction,
            instance.pending_entry_signal_bar_ts,
            instance.pending_entry_due_ts,
        )
        before_exit = (
            instance.pending_exit_reason,
            instance.pending_exit_signal_bar_ts,
            instance.pending_exit_due_ts,
        )

        apply_mutation(instance, mutation=mutation)

        after_entry = (
            instance.pending_entry_direction,
            instance.pending_entry_signal_bar_ts,
            instance.pending_entry_due_ts,
        )
        after_exit = (
            instance.pending_exit_reason,
            instance.pending_exit_signal_bar_ts,
            instance.pending_exit_due_ts,
        )
        assert after_entry == ((None, None, None) if expected["clear_entry"] else before_entry), scenario
        assert after_exit == ((None, None, None) if expected["clear_exit"] else before_exit), scenario


def test_live_entry_basis_reconciliation_matches_shared_fill_table() -> None:
    import inspect
    import math
    from types import SimpleNamespace

    from tradebot.spot import lifecycle as lifecycle_module
    from tradebot.ui import bot_engine_runtime as live_engine
    from tradebot.ui import bot_models
    from tradebot.ui.bot_screen import positions as positions_runtime

    reconciler = getattr(lifecycle_module, "reconcile_spot_entry_basis", None)
    assert callable(reconciler), "missing canonical reconcile_spot_entry_basis seam"

    state_type = getattr(lifecycle_module, "SpotEntryBasisState", None)
    assert state_type is not None, "missing typed SpotEntryBasisState contract"

    cases = [
        ("initial_long_fill", 0.0, None, 10.0, 100.0, None, None, 10.0, 100.0, "fill"),
        ("initial_short_fill", 0.0, None, -4.0, 50.0, None, None, -4.0, 50.0, "fill"),
        ("partial_entry_cancel_with_fill", 0.0, None, 3.0, 101.0, None, None, 3.0, 101.0, "fill"),
        ("same_direction_scale_in", 10.0, 100.0, 3.0, 110.0, None, None, 13.0, (10.0 * 100.0 + 3.0 * 110.0) / 13.0, "fill"),
        ("scale_out_preserves_basis", 10.0, 100.0, -4.0, 120.0, None, None, 6.0, 100.0, "fill"),
        ("flat_clears_basis", 6.0, 100.0, -6.0, 121.0, None, None, 0.0, None, "flat"),
        ("overclose_flip_resets_basis", 6.0, 100.0, -9.0, 121.0, None, None, -3.0, 121.0, "fill"),
        ("reconnect_duplicate_fill_noop", 13.0, (10.0 * 100.0 + 3.0 * 110.0) / 13.0, 0.0, 110.0, None, None, 13.0, (10.0 * 100.0 + 3.0 * 110.0) / 13.0, "fill"),
        ("broker_average_cost_authoritative", 13.0, (10.0 * 100.0 + 3.0 * 110.0) / 13.0, 0.0, None, 13.0, 102.5, 13.0, 102.5, "broker_average_cost"),
    ]

    for (
        scenario,
        previous_qty,
        previous_basis,
        fill_delta_qty,
        fill_price,
        broker_qty,
        broker_average_cost,
        expected_qty,
        expected_basis,
        expected_source,
    ) in cases:
        state = reconciler(
            previous_qty=previous_qty,
            previous_basis_price=previous_basis,
            fill_delta_qty=fill_delta_qty,
            fill_price=fill_price,
            broker_qty=broker_qty,
            broker_average_cost=broker_average_cost,
        )
        assert isinstance(state, state_type), scenario
        assert math.isclose(float(state.quantity), expected_qty, rel_tol=0.0, abs_tol=1e-12), scenario
        if expected_basis is None:
            assert state.basis_price is None, scenario
        else:
            assert state.basis_price is not None, scenario
            assert math.isclose(float(state.basis_price), expected_basis, rel_tol=0.0, abs_tol=1e-12), scenario
        assert str(state.source) == expected_source, scenario

    order_annotations = getattr(bot_models._BotOrder, "__annotations__", {})
    instance_annotations = getattr(bot_models._BotInstance, "__annotations__", {})
    assert "basis_applied_filled_qty" in order_annotations
    assert "spot_entry_basis_qty" in instance_annotations

    apply_fill = getattr(live_engine.BotEngineRuntimeMixin, "_apply_spot_entry_basis_fill", None)
    assert callable(apply_fill), "missing live entry-basis fill adapter"

    apply_broker = getattr(positions_runtime.BotPositionsMixin, "_apply_spot_broker_entry_basis", None)
    assert callable(apply_broker), "missing live broker entry-basis adapter"

    for (
        scenario,
        previous_qty,
        previous_basis,
        fill_delta_qty,
        fill_price,
        _broker_qty,
        _broker_average_cost,
        expected_qty,
        expected_basis,
        expected_source,
    ) in cases[:-1]:
        instance = _new_instance(strategy={"instrument": "spot"})
        instance.spot_entry_basis_qty = float(previous_qty)
        instance.spot_entry_basis_price = previous_basis
        instance.spot_entry_basis_source = "fill" if previous_basis is not None else None

        status = "Cancelled" if scenario == "partial_entry_cancel_with_fill" else "Filled"
        intent = "enter" if previous_qty == 0 else "resize"
        order = _new_order(
            status=status,
            signal_bar_ts=datetime(2026, 2, 9, 10, 0),
            intent=intent,
        )
        order.action = "BUY" if fill_delta_qty >= 0 else "SELL"
        cumulative_filled = abs(float(fill_delta_qty))
        if scenario == "reconnect_duplicate_fill_noop":
            cumulative_filled = 3.0
            order.basis_applied_filled_qty = 3.0
        else:
            order.basis_applied_filled_qty = 0.0

        state = apply_fill(
            instance,
            order,
            cumulative_filled=cumulative_filled,
            fill_price=fill_price,
        )
        assert isinstance(state, state_type), scenario
        assert math.isclose(float(state.quantity), expected_qty, rel_tol=0.0, abs_tol=1e-12), scenario
        assert math.isclose(float(instance.spot_entry_basis_qty), expected_qty, rel_tol=0.0, abs_tol=1e-12), scenario
        if expected_basis is None:
            assert state.basis_price is None and instance.spot_entry_basis_price is None, scenario
        else:
            assert state.basis_price is not None and instance.spot_entry_basis_price is not None, scenario
            assert math.isclose(float(state.basis_price), expected_basis, rel_tol=0.0, abs_tol=1e-12), scenario
            assert math.isclose(float(instance.spot_entry_basis_price), expected_basis, rel_tol=0.0, abs_tol=1e-12), scenario
        assert str(state.source) == expected_source, scenario
        assert math.isclose(float(order.basis_applied_filled_qty), cumulative_filled, rel_tol=0.0, abs_tol=1e-12), scenario

    broker_instance = _new_instance(strategy={"instrument": "spot"})
    broker_instance.spot_entry_basis_qty = 13.0
    broker_instance.spot_entry_basis_price = (10.0 * 100.0 + 3.0 * 110.0) / 13.0
    broker_instance.spot_entry_basis_source = "fill"
    broker_item = SimpleNamespace(position=13.0, averageCost=102.5)
    broker_state = apply_broker(broker_instance, broker_item)
    assert isinstance(broker_state, state_type)
    assert math.isclose(float(broker_instance.spot_entry_basis_qty), 13.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(broker_instance.spot_entry_basis_price), 102.5, rel_tol=0.0, abs_tol=1e-12)
    assert broker_instance.spot_entry_basis_source == "broker_average_cost"

    chase_source = inspect.getsource(live_engine.BotEngineRuntimeMixin._chase_orders_tick)
    fill_route = chase_source.find("_apply_spot_entry_basis_fill(")
    terminal_gate = chase_source.find("if is_done:")
    legacy_filled_gate = chase_source.find('if status == "Filled" and sec_type == "STK":')
    assert terminal_gate >= 0
    assert fill_route > terminal_gate
    assert legacy_filled_gate < 0 or fill_route < legacy_filled_gate

    refresh_source = inspect.getsource(positions_runtime.BotPositionsMixin._refresh_positions)
    assert "_apply_spot_broker_entry_basis(" in refresh_source


import pytest


@pytest.mark.parametrize("portfolio_mode", ("explicit_zero", "omitted"))
def test_live_broker_refresh_clears_entry_basis_without_open_position(
    portfolio_mode: str,
) -> None:
    import asyncio

    from ib_insync import PortfolioItem, Stock

    from tradebot.ui.bot_models import _BotInstance
    from tradebot.ui.bot_screen.positions import BotPositionsMixin

    contract = Stock("SLV", "SMART", "USD")
    contract.conId = 1001
    flat_item = PortfolioItem(
        contract=contract,
        position=0.0,
        marketPrice=102.0,
        marketValue=0.0,
        averageCost=0.0,
        unrealizedPNL=0.0,
        realizedPNL=0.0,
        account="DU-TEST",
    )
    portfolio = [flat_item] if portfolio_mode == "explicit_zero" else []

    instance = _BotInstance(
        instance_id=1,
        group="entry-basis-flat",
        symbol="SLV",
        strategy={"instrument": "spot", "spot_sec_type": "STK"},
        filters=None,
    )
    instance.spot_entry_basis_qty = 10.0
    instance.spot_entry_basis_price = 100.0
    instance.spot_entry_basis_source = "fill"

    class _Client:
        async def fetch_portfolio(self):
            return list(portfolio)

    class _Harness(BotPositionsMixin):
        def __init__(self) -> None:
            self._client = _Client()
            self._instances = [instance]
            self._positions = []
            self._tracked_conids = set()

        async def _prime_position_tickers(self) -> None:
            return None

        def _refresh_instances_table(self, *, refresh_dependents=True) -> None:
            return None

        def _refresh_orders_table(self) -> None:
            return None

        @staticmethod
        def _spot_sec_type(_instance, _symbol) -> str:
            return "STK"

        @staticmethod
        def _item_con_id(item) -> int:
            return int(getattr(getattr(item, "contract", None), "conId", 0) or 0)

    asyncio.run(_Harness()._refresh_positions())

    assert instance.spot_entry_basis_qty == 0.0, (
        f"{portfolio_mode}: broker refresh must clear stale spot entry basis when flat"
    )
    assert instance.spot_entry_basis_price is None, (
        f"{portfolio_mode}: broker refresh must clear stale spot entry basis when flat"
    )
    assert instance.spot_entry_basis_source == "flat", (
        f"{portfolio_mode}: broker refresh must clear stale spot entry basis when flat"
    )
