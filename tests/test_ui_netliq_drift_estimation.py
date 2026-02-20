from __future__ import annotations

import importlib.util
import sys
import asyncio
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_positions_app():
    path = Path(__file__).resolve().parents[1] / "tradebot" / "ui" / "app.py"
    module_name = "tradebot.ui.app_netliq_drift_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader

    bot_runtime_mod = type(sys)("tradebot.ui.bot_runtime")

    class _BotRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def install(self, *_args, **_kwargs) -> None:
            pass

        def toggle(self, *_args, **_kwargs) -> None:
            pass

    bot_runtime_mod.BotRuntime = _BotRuntime
    favorites_mod = type(sys)("tradebot.ui.favorites")
    favorites_mod.FavoritesScreen = type("FavoritesScreen", (), {})
    positions_mod = type(sys)("tradebot.ui.positions")
    positions_mod.PositionDetailScreen = type("PositionDetailScreen", (), {})

    prior_bot_runtime = sys.modules.get("tradebot.ui.bot_runtime")
    prior_favorites = sys.modules.get("tradebot.ui.favorites")
    prior_positions = sys.modules.get("tradebot.ui.positions")

    sys.modules["tradebot.ui.bot_runtime"] = bot_runtime_mod
    sys.modules["tradebot.ui.favorites"] = favorites_mod
    sys.modules["tradebot.ui.positions"] = positions_mod

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if prior_bot_runtime is None:
            sys.modules.pop("tradebot.ui.bot_runtime", None)
        else:
            sys.modules["tradebot.ui.bot_runtime"] = prior_bot_runtime
        if prior_favorites is None:
            sys.modules.pop("tradebot.ui.favorites", None)
        else:
            sys.modules["tradebot.ui.favorites"] = prior_favorites
        if prior_positions is None:
            sys.modules.pop("tradebot.ui.positions", None)
        else:
            sys.modules["tradebot.ui.positions"] = prior_positions
    return module.PositionsApp


PositionsApp = _load_positions_app()


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _item(con_id: int, *, sec_type: str = "STK"):
    return SimpleNamespace(
        contract=SimpleNamespace(conId=int(con_id), secType=str(sec_type)),
        unrealizedPNL=0.0,
    )


def test_netliq_unreal_drift_estimate_uses_overlap_delta() -> None:
    _ensure_event_loop()
    app = PositionsApp()
    items = [_item(11), _item(22)]
    official = {11: 100.0, 22: -20.0}
    estimate = {11: 130.0, 22: -10.0}

    app._official_unrealized_value = lambda item: official[int(item.contract.conId)]  # type: ignore[method-assign]
    app._mark_price = lambda _item: (1.0, False)  # type: ignore[method-assign]
    app._live_unrealized_metrics = (  # type: ignore[method-assign]
        lambda item, _mark: (estimate[int(item.contract.conId)], None)
    )

    value = app._estimate_net_liq_from_unreal_drift(5000.0, items)

    assert value == pytest.approx(5040.0)


def test_netliq_unreal_drift_estimate_skips_non_overlap_rows() -> None:
    _ensure_event_loop()
    app = PositionsApp()
    items = [_item(11), _item(22), _item(33)]
    official = {11: 10.0, 22: None, 33: 20.0}
    estimate = {11: 15.0, 22: 3.0, 33: None}

    app._official_unrealized_value = lambda item: official[int(item.contract.conId)]  # type: ignore[method-assign]
    app._mark_price = lambda _item: (1.0, False)  # type: ignore[method-assign]
    app._live_unrealized_metrics = (  # type: ignore[method-assign]
        lambda item, _mark: (estimate[int(item.contract.conId)], None)
    )

    value = app._estimate_net_liq_from_unreal_drift(1000.0, items)

    assert value == pytest.approx(1005.0)


def test_netliq_unreal_drift_estimate_returns_none_without_overlap() -> None:
    _ensure_event_loop()
    app = PositionsApp()
    items = [_item(11), _item(22)]
    official = {11: None, 22: None}
    estimate = {11: 4.0, 22: -1.0}

    app._official_unrealized_value = lambda item: official[int(item.contract.conId)]  # type: ignore[method-assign]
    app._mark_price = lambda _item: (1.0, False)  # type: ignore[method-assign]
    app._live_unrealized_metrics = (  # type: ignore[method-assign]
        lambda item, _mark: (estimate[int(item.contract.conId)], None)
    )

    value = app._estimate_net_liq_from_unreal_drift(1000.0, items)

    assert value is None


def test_netliq_amount_cell_shows_raw_only_when_estimate_missing() -> None:
    text = PositionsApp._net_liq_amount_cell(5796.18, None)
    assert text.plain == "5,796.18"


def test_netliq_amount_cell_shows_raw_and_estimate_when_present() -> None:
    text = PositionsApp._net_liq_amount_cell(5796.18, 5840.25)
    assert text.plain == "5,796.18 (5,840.25)"


class _CaptureTable:
    def __init__(self) -> None:
        self.rows: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def add_row(self, *args, **kwargs) -> None:
        self.rows.append((args, kwargs))


class _RefreshSnapshot:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.error: str | None = None
        self.updated_at = None

    def update(self, items: list[object], error: str | None) -> None:
        self.items = list(items)
        self.error = str(error) if error is not None else None


class _RefreshClient:
    def __init__(self, events: list[str], *, fail_fetch: bool = False) -> None:
        self._events = events
        self._fail_fetch = bool(fail_fetch)

    async def fetch_portfolio(self):
        self._events.append("fetch")
        if self._fail_fetch:
            raise RuntimeError("fetch failed")
        return []

    async def hard_refresh(self) -> None:
        self._events.append("hard")

    def start_index_tickers(self) -> None:
        self._events.append("index")

    def start_proxy_tickers(self) -> None:
        self._events.append("proxy")

    @staticmethod
    def index_tickers() -> dict:
        return {}

    @staticmethod
    def index_error() -> str | None:
        return None

    @staticmethod
    def proxy_tickers() -> dict:
        return {}

    @staticmethod
    def proxy_error() -> str | None:
        return None

    @staticmethod
    def pnl():
        return None

    @staticmethod
    def account_value(_tag: str):
        return (None, None, None)


def test_total_row_prefers_official_unrealized_when_available() -> None:
    table = _CaptureTable()
    fake_self = SimpleNamespace(
        _official_unrealized_value=lambda item: {11: 10.0, 22: None}.get(
            int(item.contract.conId)
        ),
        _mark_price=lambda _item: (None, False),
        _center_cell=lambda text, _width: text,
        _UNREAL_COL_WIDTH=32,
        _REALIZED_COL_WIDTH=24,
        _table=table,
        _row_keys=[],
    )
    fake_self._open_position_totals = lambda items: PositionsApp._open_position_totals(fake_self, items)
    items = [
        SimpleNamespace(
            contract=SimpleNamespace(conId=11, secType="STK"),
            unrealizedPNL=5.0,
            realizedPNL=0.0,
            averageCost=100.0,
            position=1.0,
            marketValue=105.0,
        ),
        SimpleNamespace(
            contract=SimpleNamespace(conId=22, secType="STK"),
            unrealizedPNL=3.0,
            realizedPNL=0.0,
            averageCost=100.0,
            position=1.0,
            marketValue=103.0,
        ),
    ]

    PositionsApp._add_total_row(fake_self, items)

    assert len(table.rows) == 1
    row_args, row_kwargs = table.rows[0]
    assert row_kwargs.get("key") == "total"
    assert row_args[4].plain.startswith("13.00")


def test_open_position_totals_supports_estimate_only_mode() -> None:
    fake_self = SimpleNamespace(
        _official_unrealized_value=lambda _item: 100.0,
        _mark_price=lambda _item: (None, False),
    )
    items = [
        SimpleNamespace(
            contract=SimpleNamespace(conId=11, secType="STK"),
            unrealizedPNL=5.0,
            realizedPNL=1.0,
            averageCost=100.0,
            position=1.0,
            marketValue=105.0,
        ),
    ]

    _u1, _r1, total_prefer_official, _pct1 = PositionsApp._open_position_totals(
        fake_self,
        items,
        prefer_official_unreal=True,
    )
    _u2, _r2, total_estimate_only, _pct2 = PositionsApp._open_position_totals(
        fake_self,
        items,
        prefer_official_unreal=False,
    )

    assert total_prefer_official == pytest.approx(101.0)
    assert total_estimate_only == pytest.approx(6.0)


def test_today_ibkr_value_uses_account_then_open_then_cache() -> None:
    _ensure_event_loop()
    app = PositionsApp()
    app._client = SimpleNamespace(pnl_single_daily=lambda _con_id: 12.5)
    app._pnl = SimpleNamespace(dailyPnL=99.0)
    items = [SimpleNamespace(contract=SimpleNamespace(conId=11, secType="STK"))]

    value, source = app._today_ibkr_value(items, et_day=date(2026, 2, 19))
    assert value == pytest.approx(99.0)
    assert source == "account"

    app._pnl = SimpleNamespace(dailyPnL=float("nan"))
    value, source = app._today_ibkr_value(items, et_day=date(2026, 2, 19))
    assert value == pytest.approx(12.5)
    assert source == "open"

    app._client = SimpleNamespace(pnl_single_daily=lambda _con_id: None)
    value, source = app._today_ibkr_value(items, et_day=date(2026, 2, 19))
    assert value == pytest.approx(99.0)
    assert source == "cache"

    value, source = app._today_ibkr_value(items, et_day=date(2026, 2, 20))
    assert value is None
    assert source == "warmup"


def test_refresh_positions_warms_streams_after_successful_snapshot() -> None:
    _ensure_event_loop()
    events: list[str] = []
    fake_self = SimpleNamespace(
        _refresh_lock=asyncio.Lock(),
        _dirty=False,
        _client=_RefreshClient(events),
        _snapshot=_RefreshSnapshot(),
        _sync_session_tickers=lambda: events.append("sync"),
        _maybe_update_buying_power_anchor=lambda: events.append("anchor"),
        _prime_change_data=lambda _items: events.append("prime"),
        _render_table=lambda: events.append("render"),
        _search_active=False,
    )

    asyncio.run(PositionsApp.refresh_positions(fake_self))

    assert events.index("fetch") < events.index("index")
    assert events.index("index") < events.index("proxy")


def test_refresh_positions_skips_stream_warmup_when_snapshot_fails() -> None:
    _ensure_event_loop()
    events: list[str] = []
    snapshot = _RefreshSnapshot()
    fake_self = SimpleNamespace(
        _refresh_lock=asyncio.Lock(),
        _dirty=False,
        _client=_RefreshClient(events, fail_fetch=True),
        _snapshot=snapshot,
        _sync_session_tickers=lambda: events.append("sync"),
        _maybe_update_buying_power_anchor=lambda: events.append("anchor"),
        _prime_change_data=lambda _items: events.append("prime"),
        _render_table=lambda: events.append("render"),
        _search_active=False,
    )

    asyncio.run(PositionsApp.refresh_positions(fake_self))

    assert "index" not in events
    assert "proxy" not in events
    assert snapshot.error == "fetch failed"
