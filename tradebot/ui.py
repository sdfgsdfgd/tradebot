"""Minimal TUI for displaying current portfolio positions."""
from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import Bag, ComboLeg, Contract, Option, PnL, PortfolioItem, Stock, Ticker, Trade
from rich.text import Text
from textual.app import App, ComposeResult
from textual import events
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, DataTable, Static

from .client import IBKRClient
from .config import load_config
from .decision_core import (
    AtrRatioShockEngine,
    DailyAtrPctShockEngine,
    DailyDrawdownShockEngine,
    EmaDecisionEngine,
    EmaDecisionSnapshot,
    OrbDecisionEngine,
    SupertrendEngine,
    apply_regime_gate,
    cooldown_ok_by_time,
    flip_exit_hit,
    parse_time_hhmm,
    realized_vol_from_closes,
    signal_filters_ok,
)
from .signals import (
    direction_from_action_right,
    ema_next,
    ema_periods,
    ema_state_direction,
    flip_exit_mode,
    parse_bar_size,
)
from .store import PortfolioSnapshot


def _fmt_expiry(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    if len(raw) == 6 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}"
    return raw


def _fmt_qty(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


class PositionsApp(App):
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "open_details", "Details"),
        ("ctrl+t", "toggle_bot", "Bot"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #ticker {
        height: 2;
        padding: 0 1;
    }

    #positions {
        height: 1fr;
    }

    #status {
        height: 1;
        padding: 0 1;
    }

    #detail-body {
        height: 1fr;
        layout: horizontal;
    }

    #detail-left {
        width: 2fr;
        height: 1fr;
        padding: 0 1;
    }

    #detail-right {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    #bot-body {
        height: 1fr;
        layout: vertical;
    }

    #bot-status {
        height: 5;
        padding: 0 1;
    }

    #bot-presets {
        height: 2fr;
        padding: 0 1;
        border: solid #003054;
    }

    #bot-instances {
        height: 10;
        padding: 0 1;
        border: solid #003054;
    }

    #bot-proposals {
        height: 1fr;
        padding: 0 1;
        border: solid #003054;
    }

    #bot-positions {
        height: 1fr;
        padding: 0 1;
        border: solid #003054;
    }

    DataTable > .datatable--cursor {
        background: #2a2a2a;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._client = IBKRClient(self._config)
        self._bot_screen: BotScreen | None = None
        self._snapshot = PortfolioSnapshot()
        self._refresh_lock = asyncio.Lock()
        self._dirty = False
        self._dirty_task: asyncio.Task | None = None
        self._row_count = 0
        self._row_keys: list[str] = []
        self._row_item_by_key: dict[str, PortfolioItem] = {}
        self._column_count = 0
        self._index_tickers: dict[str, Ticker] = {}
        self._index_error: str | None = None
        self._proxy_tickers: dict[str, Ticker] = {}
        self._proxy_error: str | None = None
        self._pnl: PnL | None = None
        self._ticker_con_ids: set[int] = set()
        self._session_closes_by_con_id: dict[int, tuple[float | None, float | None]] = {}
        self._option_underlying_con_id: dict[int, int] = {}
        self._ticker_loading: set[int] = set()
        self._closes_loading: set[int] = set()
        self._underlying_loading: set[int] = set()
        self._md_session: str | None = None
        self._net_liq: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._net_liq_daily_anchor: float | None = None
        self._buying_power: tuple[float | None, str | None, datetime | None] = (
            None,
            None,
            None,
        )
        self._buying_power_daily_anchor: float | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="ticker")
        yield DataTable(id="positions", zebra_stripes=True)
        yield Static("Starting...", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self._table = self.query_one(DataTable)
        self._ticker = self.query_one("#ticker", Static)
        self._status = self.query_one("#status", Static)
        self._setup_columns()
        self._table.cursor_type = "row"
        self._table.focus()
        self._client.set_update_callback(self._mark_dirty)
        await self.refresh_positions()

    async def on_unmount(self) -> None:
        await self._client.disconnect()

    def _setup_columns(self) -> None:
        self._columns = [
            "Symbol",
            "Expiry",
            "Right",
            "Strike",
            "Qty",
            "AvgCost",
            "Px 24-72",
            "Unreal",
            "Realized",
            "U 24-72",
        ]
        self._column_count = len(self._columns)
        self._table.add_columns(*self._columns)

    async def action_refresh(self) -> None:
        await self.refresh_positions(hard=True)

    def action_cursor_down(self) -> None:
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        self._table.action_cursor_up()

    def action_open_details(self) -> None:
        row_index = self._table.cursor_coordinate.row
        if row_index < 0 or row_index >= len(self._row_keys):
            return
        row_key = self._row_keys[row_index]
        item = self._row_item_by_key.get(row_key)
        if item:
            self.push_screen(
                PositionDetailScreen(
                    self._client,
                    item,
                    self._config.detail_refresh_sec,
                )
            )

    def action_toggle_bot(self) -> None:
        if isinstance(self.screen, BotScreen):
            self.pop_screen()
            return
        if self._bot_screen is None:
            self._bot_screen = BotScreen(self._client, self._config.detail_refresh_sec)
        self.push_screen(self._bot_screen)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        item = self._row_item_by_key.get(event.row_key.value)
        if item:
            self.push_screen(
                PositionDetailScreen(
                    self._client,
                    item,
                    self._config.detail_refresh_sec,
                )
            )

    async def refresh_positions(self, hard: bool = False) -> None:
        if self._refresh_lock.locked():
            self._dirty = True
            return
        async with self._refresh_lock:
            try:
                if hard:
                    await self._client.hard_refresh()
                items = await self._client.fetch_portfolio()
                self._snapshot.update(items, None)
            except Exception as exc:  # pragma: no cover - UI surface
                self._snapshot.update([], str(exc))
            self._sync_session_tickers()
            self._client.start_index_tickers()
            self._client.start_proxy_tickers()
            self._index_tickers = self._client.index_tickers()
            self._index_error = self._client.index_error()
            self._proxy_tickers = self._client.proxy_tickers()
            self._proxy_error = self._client.proxy_error()
            self._pnl = self._client.pnl()
            self._net_liq = self._client.account_value("NetLiquidation")
            self._maybe_update_netliq_anchor()
            self._buying_power = self._client.account_value("BuyingPower")
            self._maybe_update_buying_power_anchor()
            self._prime_change_data(self._snapshot.items)
            self._render_table()

    def _mark_dirty(self) -> None:
        self._dirty = True
        if self._dirty_task and not self._dirty_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._dirty_task = loop.create_task(self._flush_dirty())

    async def _flush_dirty(self) -> None:
        while True:
            await asyncio.sleep(self._config.refresh_sec)
            if not self._dirty:
                break
            self._dirty = False
            await self.refresh_positions()

    def _render_table(self) -> None:
        prev_coord = self._table.cursor_coordinate
        prev_row_key = None
        if 0 <= prev_coord.row < len(self._row_keys):
            prev_row_key = self._row_keys[prev_coord.row]
        prev_row_index = prev_coord.row
        prev_column = prev_coord.column

        self._table.clear()
        self._row_keys = []
        self._row_item_by_key = {}
        items = list(self._snapshot.items)
        self._row_count = sum(
            1 for item in items if item.contract.secType in _SECTION_TYPES
        )
        for title, sec_type in _SECTION_ORDER:
            self._add_section(title, sec_type, items)
        if self._buying_power[0] is not None:
            spacer_key = "spacer:buying_power"
            self._table.add_row(
                *(Text("") for _ in range(self._column_count)),
                key=spacer_key,
            )
            self._row_keys.append(spacer_key)
            self._add_buying_power_row(self._buying_power, self._pnl)
        self._add_total_row(items)
        self._add_daily_row(self._pnl)
        self._add_net_liq_row(self._net_liq, self._pnl)
        self._render_ticker_bar()
        self._status.update(self._status_text())
        self._restore_cursor(prev_row_key, prev_row_index, prev_column)

    def _maybe_update_netliq_anchor(self) -> None:
        value, _currency, updated_at = self._net_liq
        if value is None or updated_at is None:
            return
        if not hasattr(self, "_net_liq_updated_at") or self._net_liq_updated_at != updated_at:
            daily = _pnl_value(self._pnl)
            if daily is not None:
                self._net_liq_daily_anchor = daily
            self._net_liq_updated_at = updated_at

    def _maybe_update_buying_power_anchor(self) -> None:
        value, _currency, updated_at = self._buying_power
        if value is None or updated_at is None:
            return
        if (
            not hasattr(self, "_buying_power_updated_at")
            or self._buying_power_updated_at != updated_at
        ):
            daily = _pnl_value(self._pnl)
            if daily is not None:
                self._buying_power_daily_anchor = daily
            self._buying_power_updated_at = updated_at

    def _sync_session_tickers(self) -> None:
        session = _market_session_label()
        if session == self._md_session:
            return
        self._md_session = session
        self._ticker_con_ids.clear()
        self._ticker_loading.clear()

    def _status_text(self) -> str:
        conn = "connected" if self._client.is_connected else "disconnected"
        updated = self._snapshot.updated_at
        if updated:
            ts = updated.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts = "n/a"
        session = _market_session_label()
        base = f"IBKR {conn} | last update: {ts} | rows: {self._row_count}"
        base = f"{base} | MKT: {session} | MD: [L]=Live [D]=Delayed"
        if self._snapshot.error:
            return f"{base} | error: {self._snapshot.error}"
        return base

    def _add_section(self, title: str, sec_type: str, items: list[PortfolioItem]) -> None:
        header_key = f"header:{sec_type}"
        self._table.add_row(*self._section_header_row(title), key=header_key)
        self._row_keys.append(header_key)
        rows = [item for item in items if item.contract.secType == sec_type]
        rows.sort(key=_portfolio_sort_key, reverse=True)
        for item in rows:
            row_key = f"{sec_type}:{item.contract.conId}"
            change_text = self._contract_change_text(item)
            underlying_change = self._underlying_change_text(item)
            unreal_text, unreal_pct_text = self._unreal_texts(item)
            self._table.add_row(
                *_portfolio_row(
                    item,
                    change_text,
                    underlying_change,
                    unreal_text=unreal_text,
                    unreal_pct_text=unreal_pct_text,
                ),
                key=row_key,
            )
            self._row_keys.append(row_key)
            self._row_item_by_key[row_key] = item

    def _section_header_row(self, title: str) -> list[Text]:
        style = "bold white on #2b2b2b"
        row = [Text(f"— {title} —", style=style)]
        row.extend(Text("", style=style) for _ in range(self._column_count - 1))
        return row

    def _add_total_row(self, items: list[PortfolioItem]) -> None:
        total_unreal = 0.0
        total_real = 0.0
        total_cost = 0.0
        total_mkt = 0.0
        for item in items:
            if item.contract.secType not in _SECTION_TYPES:
                continue
            total_unreal += float(item.unrealizedPNL or 0.0)
            total_real += float(item.realizedPNL or 0.0)
            if item.averageCost:
                total_cost += abs(float(item.averageCost) * float(item.position))
            if item.marketValue:
                total_mkt += abs(float(item.marketValue))
        total_pnl = total_unreal + total_real
        denom = total_cost if total_cost > 0 else total_mkt
        pct = (total_pnl / denom * 100.0) if denom > 0 else None
        style = "bold white on #1f1f1f"
        label = Text("TOTAL (U+R)", style=style)
        blank = Text("")
        pnl_text = _pnl_text(total_pnl)
        pct_text = _pnl_pct_value(pct)
        unreal_text = _combined_value_pct(pnl_text, pct_text)
        realized_text = _pnl_text(total_real)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            unreal_text,
            realized_text,
            blank,
            key="total",
        )
        self._row_keys.append("total")

    def _add_daily_row(self, pnl: PnL | None) -> None:
        if not pnl:
            return
        daily = pnl.dailyPnL
        if daily is None or (isinstance(daily, float) and math.isnan(daily)):
            return
        style = "bold white on #1f1f1f"
        label = Text("DAILY P&L", style=style)
        blank = Text("")
        daily_text = _pnl_text(float(daily))
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            daily_text,
            blank,
            blank,
            key="daily",
        )
        self._row_keys.append("daily")

    def _add_net_liq_row(
        self, net_liq: tuple[float | None, str | None, datetime | None], pnl: PnL | None
    ) -> None:
        value, currency, updated_at = net_liq
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text("NET LIQ", style=style)
        blank = Text("")
        amount = _fmt_money(value)
        if currency:
            amount = f"{amount} {currency}"
        amount_text = Text(amount)
        ts_text = Text("")
        if updated_at:
            local_ts = updated_at.astimezone().strftime("%H:%M:%S")
            ts_text = Text(f"@ {local_ts}", style="dim")
        est_text = Text("")
        est_value = _estimate_net_liq(value, pnl, self._net_liq_daily_anchor)
        if est_value is not None:
            est_amount = _fmt_money(est_value)
            if currency:
                est_amount = f"{est_amount} {currency}"
            est_text = Text(f"~{est_amount}", style="yellow")
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            amount_text,
            ts_text,
            est_text,
            key="netliq",
        )
        self._row_keys.append("netliq")

    def _add_buying_power_row(
        self,
        buying_power: tuple[float | None, str | None, datetime | None],
        pnl: PnL | None,
    ) -> None:
        value, currency, updated_at = buying_power
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text("BUYING POWER", style=style)
        blank = Text("")
        amount = _fmt_money(value)
        if currency:
            amount = f"{amount} {currency}"
        amount_text = Text(amount)
        ts_text = Text("")
        if updated_at:
            local_ts = updated_at.astimezone().strftime("%H:%M:%S")
            ts_text = Text(f"@ {local_ts}", style="dim")
        est_text = Text("")
        est_value = _estimate_buying_power(value, pnl, self._buying_power_daily_anchor)
        if est_value is not None:
            est_amount = _fmt_money(est_value)
            if currency:
                est_amount = f"{est_amount} {currency}"
            est_text = Text(f"~{est_amount}", style="yellow")
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            blank,
            blank,
            blank,
            amount_text,
            ts_text,
            est_text,
            key="buying_power",
        )
        self._row_keys.append("buying_power")

    def _prime_change_data(self, items: list[PortfolioItem]) -> None:
        for item in items:
            if item.contract.secType not in _SECTION_TYPES:
                continue
            contract = item.contract
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._start_ticker_load(con_id, contract)
                self._start_closes_load(con_id, contract)
            if contract.secType in ("OPT", "FOP") and con_id:
                self._start_underlying_load(con_id, contract)

    def _start_ticker_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._ticker_con_ids or con_id in self._ticker_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ticker_loading.add(con_id)
        loop.create_task(self._load_ticker(con_id, contract))

    async def _load_ticker(self, con_id: int, contract: Contract) -> None:
        try:
            await self._client.ensure_ticker(contract)
            self._ticker_con_ids.add(con_id)
        except Exception:
            return
        finally:
            self._ticker_loading.discard(con_id)
        self._mark_dirty()

    def _start_closes_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._session_closes_by_con_id or con_id in self._closes_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._closes_loading.add(con_id)
        loop.create_task(self._load_closes(con_id, contract))

    async def _load_closes(self, con_id: int, contract: Contract) -> None:
        try:
            prev_close, close_3ago = await self._client.session_closes(contract)
            self._session_closes_by_con_id[con_id] = (prev_close, close_3ago)
        except Exception:
            return
        finally:
            self._closes_loading.discard(con_id)
        self._mark_dirty()

    def _start_underlying_load(self, option_con_id: int, contract: Contract) -> None:
        if option_con_id in self._option_underlying_con_id or option_con_id in self._underlying_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._underlying_loading.add(option_con_id)
        loop.create_task(self._load_underlying(option_con_id, contract))

    async def _load_underlying(self, option_con_id: int, contract: Contract) -> None:
        try:
            underlying = await self._client.resolve_underlying_contract(contract)
            if not underlying:
                return
            under_con_id = int(getattr(underlying, "conId", 0) or 0)
            if not under_con_id:
                return
            self._option_underlying_con_id[option_con_id] = under_con_id
            self._start_ticker_load(under_con_id, underlying)
            self._start_closes_load(under_con_id, underlying)
        except Exception:
            return
        finally:
            self._underlying_loading.discard(option_con_id)
        self._mark_dirty()

    def _contract_change_text(self, item: PortfolioItem) -> Text:
        contract = item.contract
        con_id = int(getattr(contract, "conId", 0) or 0)
        return self._change_text_for_con_id(con_id, include_price=True)

    def _underlying_change_text(self, item: PortfolioItem) -> Text:
        contract = item.contract
        if contract.secType not in ("OPT", "FOP"):
            return Text("")
        option_con_id = int(getattr(contract, "conId", 0) or 0)
        if not option_con_id:
            return Text("")
        under_con_id = self._option_underlying_con_id.get(option_con_id)
        if not under_con_id:
            return Text("")
        return self._change_text_for_con_id(under_con_id, include_price=False)

    def _change_text_for_con_id(self, con_id: int, *, include_price: bool) -> Text:
        if not con_id:
            return Text("")
        ticker = self._client.ticker_for_con_id(con_id)
        price = _ticker_price(ticker) if ticker else None
        prev_close = _ticker_close(ticker) if ticker else None
        cached = self._session_closes_by_con_id.get(con_id)
        if prev_close is None and cached:
            prev_close = cached[0]
        close_3ago = cached[1] if cached else None
        pct24 = _pct_change(price, prev_close)
        pct72 = _pct_change(price, close_3ago)
        if include_price:
            return _price_pct_dual_text(price, pct24, pct72)
        return _pct_dual_text(pct24, pct72)

    def _unreal_texts(self, item: PortfolioItem) -> tuple[Text, Text]:
        session = _market_session_label()
        if session == "MRKT":
            return _pnl_text(item.unrealizedPNL), _pnl_pct_text(item)
        contract = item.contract
        if contract.secType not in _SECTION_TYPES:
            return Text(""), Text("")
        mark_price, is_estimate = self._mark_price(item)
        if mark_price is None:
            return _pnl_text(item.unrealizedPNL), _pnl_pct_text(item)
        multiplier = _infer_multiplier(item)
        position = float(item.position or 0.0)
        mark_value = mark_price * position * multiplier
        cost_basis = _cost_basis(item)
        pnl = mark_value - cost_basis
        denom = abs(cost_basis) if cost_basis else abs(mark_value)
        pct = (pnl / denom * 100.0) if denom > 0 else None
        prefix = "~" if is_estimate else ""
        return _pnl_text(pnl, prefix=prefix), _pnl_pct_value(pct)

    def _mark_price(self, item: PortfolioItem) -> tuple[float | None, bool]:
        contract = item.contract
        con_id = int(getattr(contract, "conId", 0) or 0)
        ticker = self._client.ticker_for_con_id(con_id)
        if contract.secType == "OPT" or contract.secType == "FOP":
            return self._option_estimated_mark(item, ticker)
        price = _ticker_price(ticker) if ticker else None
        if price is None:
            try:
                price = float(item.marketPrice)
            except (TypeError, ValueError):
                price = None
        return price, False

    def _option_estimated_mark(
        self, item: PortfolioItem, option_ticker: Ticker | None
    ) -> tuple[float | None, bool]:
        option_con_id = int(getattr(item.contract, "conId", 0) or 0)
        under_con_id = self._option_underlying_con_id.get(option_con_id)
        under_ticker = self._client.ticker_for_con_id(under_con_id) if under_con_id else None
        under_price = _ticker_price(under_ticker) if under_ticker else None
        if under_price is None or under_price <= 0:
            return None, False
        model = getattr(option_ticker, "modelGreeks", None) if option_ticker else None
        delta = _safe_num(getattr(model, "delta", None)) if model else None
        gamma = _safe_num(getattr(model, "gamma", None)) if model else None
        under_close = _ticker_close(under_ticker) if under_ticker else None
        if under_close is None and under_con_id:
            cached = self._session_closes_by_con_id.get(under_con_id)
            under_close = cached[0] if cached else None
        ref_price = _safe_num(getattr(option_ticker, "close", None)) if option_ticker else None
        if ref_price is None:
            try:
                ref_price = float(item.marketPrice)
            except (TypeError, ValueError):
                ref_price = None
        if delta is not None and under_close is not None and under_close > 0 and ref_price is not None:
            d_under = float(under_price) - float(under_close)
            estimated = float(ref_price) + (float(delta) * d_under)
            if gamma is not None:
                estimated += 0.5 * float(gamma) * (d_under**2)
            return max(estimated, 0.0), True
        model_price = _safe_num(getattr(model, "optPrice", None)) if model else None
        if model_price is not None and model_price > 0:
            return float(model_price), True
        return None, False
    def _render_ticker_bar(self) -> None:
        prefix = f"MKT:{_market_session_label()} | "
        line1 = _ticker_line(
            _INDEX_ORDER,
            _INDEX_LABELS,
            self._index_tickers,
            self._index_error,
            prefix,
        )
        line2 = _ticker_line(
            _PROXY_ORDER,
            _PROXY_LABELS,
            self._proxy_tickers,
            self._proxy_error,
            " " * len(prefix),
        )
        text = Text()
        text.append_text(line1)
        text.append("\n")
        text.append_text(line2)
        self._ticker.update(text)

    def _restore_cursor(self, row_key: str | None, row_index: int, column: int) -> None:
        if not self._row_keys:
            return
        if row_key and row_key in self._row_keys:
            target_row = self._row_keys.index(row_key)
        else:
            target_row = min(max(row_index, 0), len(self._row_keys) - 1)
        target_col = min(max(column, 0), max(self._column_count - 1, 0))
        self._table.cursor_coordinate = (target_row, target_col)


class PositionDetailScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("b", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("r", "refresh_ticker", "Refresh MD"),
        ("up", "exec_up", "Exec Up"),
        ("down", "exec_down", "Exec Down"),
        ("j", "exec_down", "Exec Down"),
        ("k", "exec_up", "Exec Up"),
        ("left", "exec_jump_left", "Exec Jump Left"),
        ("right", "exec_jump_right", "Exec Jump Right"),
        ("h", "exec_left", "Exec Left"),
        ("l", "exec_right", "Exec Right"),
        ("c", "cancel_order", "Cancel Order"),
    ]

    def __init__(
        self, client: IBKRClient, item: PortfolioItem, refresh_sec: float
    ) -> None:
        super().__init__()
        self._client = client
        self._item = item
        self._refresh_sec = max(refresh_sec, 0.1)
        self._ticker: Ticker | None = None
        self._underlying_ticker: Ticker | None = None
        self._underlying_con_id: int | None = None
        self._underlying_label: str | None = None
        self._exec_rows = ["mid", "optimistic", "aggressive", "custom", "qty"]
        self._exec_selected = 0
        self._exec_custom_input = ""
        self._exec_qty_input = ""
        self._exec_qty = _default_order_qty(item)
        self._exec_status: str | None = None
        self._active_panel = "exec"
        self._orders_selected = 0
        self._orders_scroll = 0
        self._orders_rows: list[Trade] = []
        self._refresh_task = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Static("", id="detail-left"),
            Static("", id="detail-right"),
            id="detail-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._detail_left = self.query_one("#detail-left", Static)
        self._detail_right = self.query_one("#detail-right", Static)
        self._ticker = await self._client.ensure_ticker(self._item.contract)
        await self._load_underlying()
        self._refresh_task = self.set_interval(self._refresh_sec, self._render_details)
        self._render_details()

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id)
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id)

    def on_key(self, event: events.Key) -> None:
        if event.key == "backspace":
            self._handle_backspace()
            event.stop()
            return
        if event.character:
            if event.character == "B":
                self._submit_order("BUY")
                event.stop()
                return
            if event.character in ("S", "s"):
                self._submit_order("SELL")
                event.stop()
                return
            if event.character in "0123456789.":
                self._handle_digit(event.character)
                event.stop()
                return

    def action_exec_up(self) -> None:
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = max(self._orders_selected - 1, 0)
        else:
            self._exec_selected = max(self._exec_selected - 1, 0)
        self._render_details()

    def action_exec_down(self) -> None:
        if self._active_panel == "orders":
            if self._orders_rows:
                self._orders_selected = min(
                    self._orders_selected + 1, len(self._orders_rows) - 1
                )
        else:
            self._exec_selected = min(self._exec_selected + 1, len(self._exec_rows) - 1)
        self._render_details()

    def action_exec_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
            self._render_details()
            return
        selected = self._exec_rows[self._exec_selected]
        if self._active_panel != "exec" or selected not in ("custom", "qty"):
            self.app.pop_screen()
            return
        if selected == "custom":
            self._adjust_custom_price(-1)
        else:
            self._adjust_qty(-1)
        self._render_details()

    def action_exec_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details()
            return
        selected = self._exec_rows[self._exec_selected]
        if selected == "custom":
            self._adjust_custom_price(1)
        elif selected == "qty":
            self._adjust_qty(1)
        else:
            self._active_panel = "orders"
        self._render_details()

    def action_exec_jump_left(self) -> None:
        if self._active_panel == "orders":
            self._active_panel = "exec"
        else:
            self._exec_selected = self._exec_rows.index("custom")
        self._render_details()

    def action_exec_jump_right(self) -> None:
        if self._active_panel == "orders":
            self._render_details()
            return
        self._exec_selected = self._exec_rows.index("mid")
        self._render_details()

    def action_cancel_order(self) -> None:
        if self._active_panel != "orders":
            return
        trade = self._selected_order()
        if not trade:
            self._exec_status = "Cancel: no order"
            self._render_details()
            return
        order_id = trade.order.orderId or trade.order.permId or 0
        self._exec_status = f"Canceling #{order_id}"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Cancel: no loop"
            self._render_details()
            return
        loop.create_task(self._cancel_order(trade))

    async def action_refresh_ticker(self) -> None:
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            self._client.release_ticker(con_id)
        self._ticker = await self._client.ensure_ticker(self._item.contract)
        if self._underlying_con_id:
            self._client.release_ticker(self._underlying_con_id)
            self._underlying_con_id = None
            self._underlying_ticker = None
            self._underlying_label = None
            await self._load_underlying()
        self._exec_status = "MD refreshed"
        self._render_details()

    async def _load_underlying(self) -> None:
        contract = self._item.contract
        if contract.secType not in ("OPT", "FOP"):
            return
        underlying = await self._client.resolve_underlying_contract(contract)
        if not underlying:
            return
        self._underlying_ticker = await self._client.ensure_ticker(underlying)
        con_id = int(getattr(underlying, "conId", 0) or 0)
        if con_id:
            self._underlying_con_id = con_id
        symbol = getattr(underlying, "localSymbol", "") or getattr(underlying, "symbol", "")
        sec_type = getattr(underlying, "secType", "")
        if symbol or sec_type:
            self._underlying_label = " ".join(part for part in (symbol, sec_type) if part)

    def _render_details(self) -> None:
        try:
            con_id = int(self._item.contract.conId or 0)
            latest = self._client.portfolio_item(con_id) if con_id else None
            if latest:
                self._item = latest
            contract = self._item.contract
            lines: list[Text] = []
            lines.append(Text(f"{contract.symbol} {contract.secType}", style="bold"))
            lines.append(Text(f"ConId: {contract.conId}"))
            if contract.localSymbol:
                lines.append(Text(f"Local: {contract.localSymbol}"))
            if contract.exchange:
                lines.append(Text(f"Exchange: {contract.exchange}"))
            if contract.currency:
                lines.append(Text(f"Currency: {contract.currency}"))
            if contract.lastTradeDateOrContractMonth:
                lines.append(
                    Text(f"Expiry: {_fmt_expiry(contract.lastTradeDateOrContractMonth)}")
                )
            if contract.right:
                lines.append(Text(f"Right: {contract.right}"))
            if contract.strike:
                lines.append(Text(f"Strike: {_fmt_money(contract.strike)}"))
            if contract.multiplier:
                lines.append(Text(f"Contract Size: {contract.multiplier}"))
            lines.append(Text(""))
            lines.append(Text(f"Position: {_fmt_qty(float(self._item.position))}"))
            if self._item.averageCost:
                lines.append(Text(f"Avg Cost: {_fmt_money(float(self._item.averageCost))}"))
            if self._item.marketValue is not None:
                lines.append(Text(f"Market Value: {_fmt_money(float(self._item.marketValue))}"))
            if self._item.unrealizedPNL is not None:
                lines.append(
                    Text(f"Unrealized P&L: {_fmt_money(float(self._item.unrealizedPNL))}")
                )
            if self._item.realizedPNL is not None:
                lines.append(
                    Text(f"Realized P&L: {_fmt_money(float(self._item.realizedPNL))}")
                )
            lines.append(Text(""))
            if self._ticker:
                md_exchange = getattr(self._ticker.contract, "exchange", "") or ""
                md_label = _market_data_label(self._ticker)
                lines.append(
                    Text(
                        f"MD: {md_exchange or 'n/a'} ({md_label})",
                        style="dim",
                    )
                )
                md_local = getattr(self._ticker.contract, "localSymbol", "") or ""
                md_debug = f"MD Contract: {self._ticker.contract.secType}"
                if md_local:
                    md_debug += f" {md_local}"
                lines.append(Text(md_debug, style="dim"))
                lines.append(_quote_status_line(self._ticker))
                bid = _safe_num(self._ticker.bid)
                ask = _safe_num(self._ticker.ask)
                last = _safe_num(self._ticker.last)
                price = _ticker_price(self._ticker)
                mid = _midpoint(bid, ask)
                close = _ticker_close(self._ticker)
                mark = _mark_price(self._item)
                lines.append(
                    Text(
                        f"Bid: {_fmt_quote(bid)}  Ask: {_fmt_quote(ask)}  Last: {_fmt_quote(last)}"
                    )
                )
                if price is None and close is not None:
                    lines.append(Text(f"Price: Closed ({_fmt_quote(close)})", style="red"))
                elif price is None and mark is not None:
                    lines.append(Text(f"Price: Mark ({_fmt_quote(mark)})", style="yellow"))
                else:
                    lines.append(Text(f"Price: {_fmt_quote(price)}"))
            if self._underlying_ticker:
                lines.append(Text(""))
                label = self._underlying_label or "Underlying"
                lines.append(Text(f"{label}", style="bold"))
                md_exchange = getattr(self._underlying_ticker.contract, "exchange", "") or ""
                md_label = _market_data_label(self._underlying_ticker)
                lines.append(
                    Text(
                        f"MD: {md_exchange or 'n/a'} ({md_label})",
                        style="dim",
                    )
                )
                md_local = getattr(self._underlying_ticker.contract, "localSymbol", "") or ""
                md_debug = f"MD Contract: {self._underlying_ticker.contract.secType}"
                if md_local:
                    md_debug += f" {md_local}"
                lines.append(Text(md_debug, style="dim"))
                lines.append(_quote_status_line(self._underlying_ticker))
                bid = _safe_num(self._underlying_ticker.bid)
                ask = _safe_num(self._underlying_ticker.ask)
                last = _safe_num(self._underlying_ticker.last)
                lines.append(
                    Text(
                        f"Bid: {_fmt_quote(bid)}  Ask: {_fmt_quote(ask)}  Last: {_fmt_quote(last)}"
                    )
                )
            lines.extend(self._render_execution_block())
            self._detail_left.update(Text("\n").join(lines))
        except Exception as exc:
            self._detail_left.update(Text(f"Detail render error: {exc}", style="red"))
        try:
            self._render_orders_panel()
        except Exception as exc:
            self._detail_right.update(Text(f"Orders render error: {exc}", style="red"))

    def _render_execution_block(self) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT"):
            return []
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        tick = _tick_size(contract, self._ticker, last or mark)
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        optimistic_buy = _round_to_tick(_optimistic_price(bid, ask, mid, "BUY"), tick)
        optimistic_sell = _round_to_tick(_optimistic_price(bid, ask, mid, "SELL"), tick)
        aggressive_buy = _round_to_tick(_aggressive_price(bid, ask, mid, "BUY"), tick)
        aggressive_sell = _round_to_tick(_aggressive_price(bid, ask, mid, "SELL"), tick)
        custom = _parse_float(self._exec_custom_input)
        if custom is not None:
            custom = _round_to_tick(custom, tick)
        qty = self._exec_qty
        lines: list[Text] = []
        lines.append(Text(""))
        lines.append(Text("Execution", style="bold"))
        lines.append(Text(f"Tick: {tick:.{_tick_decimals(tick)}f}", style="dim"))
        lines.append(Text("B=Buy  S=Sell", style="dim"))
        if self._exec_status:
            lines.append(Text(self._exec_status, style="yellow"))
        rows = [
            ("mid", "Mid"),
            (
                "optimistic",
                f"Optimistic (B/S): {_fmt_quote(optimistic_buy)} / {_fmt_quote(optimistic_sell)}",
            ),
            (
                "aggressive",
                f"Aggressive (B/S): {_fmt_quote(aggressive_buy)} / {_fmt_quote(aggressive_sell)}",
            ),
            ("custom", f"Custom: {_fmt_quote(custom)}"),
            ("qty", f"Qty: {qty}"),
        ]
        for idx, (key, label) in enumerate(rows):
            style = (
                "bold on #2b2b2b"
                if self._active_panel == "exec" and idx == self._exec_selected
                else ""
            )
            if key == "mid":
                lines.append(Text(""))
                line = Text("Mid: ")
                line.append(_fmt_quote(mid), style="orange1")
            else:
                line = Text(label)
            if style:
                line.stylize(style)
            lines.append(line)
        return lines

    def _render_orders_panel(self) -> None:
        con_ids: list[int] = []
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            con_ids.append(con_id)
        if self._underlying_con_id and self._underlying_con_id != con_id:
            con_ids.append(self._underlying_con_id)
        trades = self._client.open_trades_for_conids(con_ids)
        trades.sort(key=_trade_sort_key, reverse=True)
        self._orders_rows = trades
        if not trades:
            self._orders_selected = 0
            self._orders_scroll = 0
            lines = [Text("Orders", style="bold"), Text("No open orders", style="dim")]
            self._detail_right.update(Text("\n").join(lines))
            return
        if self._orders_selected >= len(trades):
            self._orders_selected = len(trades) - 1
        header = Text("Label Status Side Qty Type@Price Fill/Rem Id", style="dim")
        lines: list[Text] = [Text("Orders", style="bold"), header]
        available = self._detail_right.size.height
        visible = len(trades)
        if available:
            visible = max(available - len(lines), 1)
        max_scroll = max(len(trades) - visible, 0)
        self._orders_scroll = min(max(self._orders_scroll, 0), max_scroll)
        if self._orders_selected < self._orders_scroll:
            self._orders_scroll = self._orders_selected
        elif self._orders_selected >= self._orders_scroll + visible:
            self._orders_scroll = self._orders_selected - visible + 1
        start = self._orders_scroll
        end = min(start + visible, len(trades))
        for idx in range(start, end):
            trade = trades[idx]
            line = self._format_order_line(trade)
            if self._active_panel == "orders" and idx == self._orders_selected:
                line.stylize("bold on #2b2b2b")
            lines.append(line)
        self._detail_right.update(Text("\n").join(lines))

    def _format_order_line(self, trade: Trade) -> Text:
        contract = trade.contract
        label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or "?"
        label = label[:10]
        status = trade.orderStatus.status or "n/a"
        status = status.replace("PreSubmitted", "PreSub")
        side = (trade.order.action or "?")[:1]
        qty = _fmt_qty(float(trade.order.totalQuantity or 0))
        order_type = trade.order.orderType or ""
        price = self._order_price(trade)
        if price is not None:
            type_label = f"{order_type}@{_fmt_quote(price)}"
        else:
            type_label = order_type
        filled = _fmt_qty(float(trade.orderStatus.filled or 0))
        remaining = _fmt_qty(float(trade.orderStatus.remaining or 0))
        order_id = trade.order.orderId or trade.order.permId or 0
        line = (
            f"{label:<10} {status:<10} {side:<1} {qty:>4} "
            f"{type_label:<12} {filled}/{remaining} #{order_id}"
        )
        return Text(line)

    def _order_price(self, trade: Trade) -> float | None:
        order = trade.order
        price = _safe_num(getattr(order, "lmtPrice", None))
        if price is not None:
            return price
        return _safe_num(getattr(order, "auxPrice", None))

    def _selected_order(self) -> Trade | None:
        if not self._orders_rows:
            return None
        idx = min(max(self._orders_selected, 0), len(self._orders_rows) - 1)
        return self._orders_rows[idx]

    async def _cancel_order(self, trade: Trade) -> None:
        try:
            await self._client.cancel_trade(trade)
            order_id = trade.order.orderId or trade.order.permId or 0
            self._exec_status = f"Cancel sent #{order_id}"
        except Exception as exc:
            self._exec_status = f"Cancel error: {exc}"
        self._render_details()

    def _handle_digit(self, char: str) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._exec_qty_input = _append_digit(self._exec_qty_input, char, allow_decimal=False)
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        else:
            if selected != "custom":
                self._exec_selected = self._exec_rows.index("custom")
            self._exec_custom_input = _append_digit(self._exec_custom_input, char, allow_decimal=True)
        self._render_details()

    def _handle_backspace(self) -> None:
        selected = self._exec_rows[self._exec_selected]
        if selected == "qty":
            self._exec_qty_input = self._exec_qty_input[:-1]
            parsed = _parse_int(self._exec_qty_input)
            if parsed:
                self._exec_qty = parsed
        else:
            self._exec_custom_input = self._exec_custom_input[:-1]
        self._render_details()

    def _adjust_custom_price(self, direction: int) -> None:
        tick = _tick_size(self._item.contract, self._ticker, _mark_price(self._item))
        current = _parse_float(self._exec_custom_input)
        if current is None:
            bid = _safe_num(self._ticker.bid) if self._ticker else None
            ask = _safe_num(self._ticker.ask) if self._ticker else None
            mid = _midpoint(bid, ask)
            current = _round_to_tick(mid, tick) or 0.0
        adjusted = _round_to_tick(current + (tick * direction), tick)
        if adjusted is None:
            return
        self._exec_custom_input = f"{adjusted:.{_tick_decimals(tick)}f}"

    def _adjust_qty(self, direction: int) -> None:
        next_qty = max(1, int(self._exec_qty) + direction)
        self._exec_qty = next_qty
        self._exec_qty_input = str(next_qty)

    def _submit_order(self, action: str) -> None:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT"):
            self._exec_status = "Exec: unsupported contract"
            self._render_details()
            return
        qty = int(self._exec_qty) if self._exec_qty else 0
        if qty <= 0:
            self._exec_status = "Exec: invalid qty"
            self._render_details()
            return
        price = self._selected_exec_price(action)
        if price is None:
            self._exec_status = "Exec: price n/a"
            self._render_details()
            return
        outside_rth = contract.secType == "STK"
        self._exec_status = f"Sending {action} {qty} @ {price:.2f}"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._exec_status = "Exec: no loop"
            self._render_details()
            return
        loop.create_task(self._place_order(action, qty, price, outside_rth))

    async def _place_order(
        self, action: str, qty: int, price: float, outside_rth: bool
    ) -> None:
        try:
            await self._client.place_limit_order(
                self._item.contract, action, qty, price, outside_rth
            )
            self._exec_status = f"Sent {action} {qty} @ {price:.2f}"
        except Exception as exc:
            self._exec_status = f"Exec error: {exc}"
        self._render_details()

    def _selected_exec_price(self, action: str) -> float | None:
        bid = _safe_num(self._ticker.bid) if self._ticker else None
        ask = _safe_num(self._ticker.ask) if self._ticker else None
        last = _safe_num(self._ticker.last) if self._ticker else None
        mark = _mark_price(self._item)
        tick = _tick_size(self._item.contract, self._ticker, last or mark)
        mid = _round_to_tick(_midpoint(bid, ask), tick)
        fallback = _round_to_tick(last or mark, tick)
        selected = self._exec_rows[self._exec_selected]
        if selected == "mid":
            return mid or fallback
        if selected == "optimistic":
            value = _optimistic_price(bid, ask, mid, action)
            return _round_to_tick(value, tick) or fallback
        if selected == "aggressive":
            value = _aggressive_price(bid, ask, mid, action)
            return _round_to_tick(value, tick) or fallback
        if selected == "custom":
            value = _parse_float(self._exec_custom_input)
            return _round_to_tick(value, tick) if value is not None else None
        return fallback


def _portfolio_sort_key(item: PortfolioItem) -> float:
    unreal = float(item.unrealizedPNL or 0.0)
    realized = float(item.realizedPNL or 0.0)
    return unreal + realized


def _portfolio_row(
    item: PortfolioItem,
    contract_change: Text,
    underlying_change: Text,
    *,
    unreal_text: Text | None = None,
    unreal_pct_text: Text | None = None,
) -> list[Text | str]:
    contract = item.contract
    expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth or "")
    right = contract.right or ""
    strike = _fmt_money(contract.strike) if contract.strike else ""
    qty = _fmt_qty(float(item.position))
    avg_cost = _fmt_money(float(item.averageCost)) if item.averageCost else ""
    unreal = unreal_text or _pnl_text(item.unrealizedPNL)
    unreal_pct = unreal_pct_text or _pnl_pct_text(item)
    unreal_combined = _combined_value_pct(unreal, unreal_pct)
    realized = _pnl_text(item.realizedPNL)
    return [
        contract.symbol,
        expiry,
        right,
        strike,
        qty,
        avg_cost,
        contract_change,
        unreal_combined,
        realized,
        underlying_change,
    ]


def _trade_sort_key(trade: Trade) -> int:
    order = trade.order
    order_id = getattr(order, "orderId", 0) or 0
    perm_id = getattr(order, "permId", 0) or 0
    return int(order_id or perm_id or 0)


@dataclass(frozen=True)
class _BotPreset:
    group: str
    entry: dict


@dataclass
class _BotInstance:
    instance_id: int
    group: str
    symbol: str
    strategy: dict
    filters: dict | None
    metrics: dict | None = None
    auto_trade: bool = False
    state: str = "RUNNING"
    last_propose_date: date | None = None
    open_direction: str | None = None
    last_entry_bar_ts: datetime | None = None
    last_exit_bar_ts: datetime | None = None
    entries_today: int = 0
    entries_today_date: date | None = None
    error: str | None = None
    spot_profit_target_price: float | None = None
    spot_stop_loss_price: float | None = None
    touched_conids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _BotConfigResult:
    mode: str  # "create" or "update"
    instance_id: int | None
    group: str
    symbol: str
    strategy: dict
    filters: dict | None
    auto_trade: bool


@dataclass(frozen=True)
class _BotConfigField:
    label: str
    kind: str  # "int" | "float" | "bool" | "enum" | "text"
    path: str
    options: tuple[str, ...] = ()


@dataclass
class _BotLegOrder:
    contract: Contract
    action: str  # BUY/SELL
    ratio: int


@dataclass
class _BotProposal:
    instance_id: int
    preset: _BotPreset | None
    underlying: Contract
    order_contract: Contract
    legs: list[_BotLegOrder]
    action: str  # BUY/SELL (order action for order_contract)
    quantity: int  # combo quantity (BAG) or contracts (single-leg)
    limit_price: float
    created_at: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    status: str = "PROPOSED"
    order_id: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class _SignalSnapshot:
    bar_ts: datetime
    close: float
    signal: EmaDecisionSnapshot
    bars_in_day: int
    rv: float | None
    volume: float | None = None
    volume_ema: float | None = None
    volume_ema_ready: bool = True
    shock: bool | None = None
    shock_dir: str | None = None
    atr: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    or_ready: bool = False


class BotConfigScreen(Screen[_BotConfigResult | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("q", "cancel", "Cancel"),
        ("enter", "save", "Save"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("left", "cycle_left", "Prev"),
        ("right", "cycle_right", "Next"),
        ("space", "toggle_bool", "Toggle"),
        ("e", "toggle_edit", "Edit"),
    ]

    def __init__(
        self,
        *,
        mode: str,
        instance_id: int | None,
        group: str,
        symbol: str,
        strategy: dict,
        filters: dict | None,
        auto_trade: bool,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._instance_id = instance_id
        self._group = group
        self._symbol = symbol
        self._strategy = copy.deepcopy(strategy)
        self._filters = copy.deepcopy(filters) if filters else None
        self._auto_trade = bool(auto_trade)

        self._fields: list[_BotConfigField] = []
        self._editing: _BotConfigField | None = None
        self._edit_buffer = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="bot-config-header")
        yield DataTable(id="bot-config", zebra_stripes=True)
        yield Static(
            "Enter=Save & start  Esc=Cancel  Type=Edit  e=Edit w/ current  Space=Toggle  ←/→=Cycle enum",
            id="bot-config-help",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._header = self.query_one("#bot-config-header", Static)
        self._table = self.query_one("#bot-config", DataTable)
        self._table.cursor_type = "row"
        self._table.add_columns("Field", "Value")
        self._build_fields()
        self._refresh_table()
        self._table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; treat Enter as Save.
        self.action_save()

    def _build_fields(self) -> None:
        instrument_raw = self._strategy.get("instrument", "options")
        instrument = "spot" if str(instrument_raw or "").strip().lower() == "spot" else "options"

        self._strategy.setdefault("entry_signal", "ema")
        self._strategy.setdefault("entry_confirm_bars", 0)
        self._strategy.setdefault("orb_window_mins", 15)
        self._strategy.setdefault("orb_risk_reward", 2.0)
        self._strategy.setdefault("orb_target_mode", "rr")
        self._strategy.setdefault("orb_open_time_et", "")
        self._strategy.setdefault("spot_exit_mode", "pct")
        self._strategy.setdefault("spot_atr_period", 14)
        self._strategy.setdefault("spot_pt_atr_mult", 1.5)
        self._strategy.setdefault("spot_sl_atr_mult", 1.0)
        self._strategy.setdefault("spot_exit_time_et", "")
        self._strategy.setdefault("regime_ema_preset", "")
        self._strategy.setdefault("regime_bar_size", "")
        self._strategy.setdefault("regime_mode", "ema")
        self._strategy.setdefault("regime2_mode", "off")
        self._strategy.setdefault("regime2_ema_preset", "")
        self._strategy.setdefault("regime2_bar_size", "")
        self._strategy.setdefault("regime2_supertrend_atr_period", 10)
        self._strategy.setdefault("regime2_supertrend_multiplier", 3.0)
        self._strategy.setdefault("regime2_supertrend_source", "hl2")
        self._strategy.setdefault("supertrend_atr_period", 10)
        self._strategy.setdefault("supertrend_multiplier", 3.0)
        self._strategy.setdefault("supertrend_source", "hl2")

        if instrument == "spot":
            sym = str(self._symbol or self._strategy.get("symbol") or "").strip().upper()
            default_sec_type = (
                "FUT" if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"} else "STK"
            )
            self._strategy.setdefault("spot_sec_type", default_sec_type)
            self._strategy.setdefault("spot_exchange", "")
            self._strategy.setdefault("spot_close_eod", False)
            if not isinstance(self._strategy.get("directional_spot"), dict):
                self._strategy["directional_spot"] = {
                    "up": {"action": "BUY", "qty": 1},
                    "down": {"action": "SELL", "qty": 1},
                }

        self._fields = [
            _BotConfigField("Symbol", "text", "symbol"),
            _BotConfigField("Auto trade", "bool", "auto_trade"),
            _BotConfigField("Instrument", "enum", "instrument", options=("options", "spot")),
            _BotConfigField(
                "Signal bar size",
                "enum",
                "signal_bar_size",
                options=("1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
            ),
            _BotConfigField("Signal use RTH", "bool", "signal_use_rth"),
            _BotConfigField("Entry days", "text", "entry_days"),
            _BotConfigField("Max entries/day", "int", "max_entries_per_day"),
            _BotConfigField("Entry signal", "enum", "entry_signal", options=("ema", "orb")),
            _BotConfigField("EMA preset", "text", "ema_preset"),
            _BotConfigField("EMA entry mode", "enum", "ema_entry_mode", options=("trend", "cross")),
            _BotConfigField("Entry confirm bars", "int", "entry_confirm_bars"),
            _BotConfigField("ORB window mins", "int", "orb_window_mins"),
            _BotConfigField("ORB risk reward", "float", "orb_risk_reward"),
            _BotConfigField("ORB target mode", "enum", "orb_target_mode", options=("rr", "or_range")),
            _BotConfigField("ORB open time (ET)", "text", "orb_open_time_et"),
            _BotConfigField("Regime mode", "enum", "regime_mode", options=("ema", "supertrend")),
            _BotConfigField("Regime EMA preset", "text", "regime_ema_preset"),
            _BotConfigField(
                "Regime bar size",
                "enum",
                "regime_bar_size",
                options=("", "1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
            ),
            _BotConfigField("Supertrend ATR period", "int", "supertrend_atr_period"),
            _BotConfigField("Supertrend multiplier", "float", "supertrend_multiplier"),
            _BotConfigField(
                "Supertrend source",
                "enum",
                "supertrend_source",
                options=("hl2", "close"),
            ),
            _BotConfigField(
                "Regime2 mode",
                "enum",
                "regime2_mode",
                options=("off", "ema", "supertrend"),
            ),
            _BotConfigField(
                "Regime2 apply to",
                "enum",
                "regime2_apply_to",
                options=("both", "longs", "shorts"),
            ),
            _BotConfigField("Regime2 EMA preset", "text", "regime2_ema_preset"),
            _BotConfigField(
                "Regime2 bar size",
                "enum",
                "regime2_bar_size",
                options=("", "1 hour", "4 hours", "30 mins", "15 mins", "5 mins", "1 day"),
            ),
            _BotConfigField("Regime2 Supertrend ATR period", "int", "regime2_supertrend_atr_period"),
            _BotConfigField("Regime2 Supertrend multiplier", "float", "regime2_supertrend_multiplier"),
            _BotConfigField(
                "Regime2 Supertrend source",
                "enum",
                "regime2_supertrend_source",
                options=("hl2", "close"),
            ),
            _BotConfigField("Entry start hour (ET)", "text", "filters.entry_start_hour_et"),
            _BotConfigField("Entry end hour (ET)", "text", "filters.entry_end_hour_et"),
            _BotConfigField("Entry start hour", "text", "filters.entry_start_hour"),
            _BotConfigField("Entry end hour", "text", "filters.entry_end_hour"),
            _BotConfigField("Volume ratio min", "float", "filters.volume_ratio_min"),
            _BotConfigField("Volume EMA period", "int", "filters.volume_ema_period"),
            _BotConfigField("Flip-exit", "bool", "exit_on_signal_flip"),
            _BotConfigField("Flip min hold bars", "int", "flip_exit_min_hold_bars"),
            _BotConfigField("Flip only if profit", "bool", "flip_exit_only_if_profit"),
        ]

        if instrument == "spot":
            self._fields.extend(
                [
                    _BotConfigField("Spot secType", "enum", "spot_sec_type", options=("STK", "FUT")),
                    _BotConfigField("Spot exchange", "text", "spot_exchange"),
                    _BotConfigField("Spot close EOD", "bool", "spot_close_eod"),
                    _BotConfigField("Spot exit time (ET)", "text", "spot_exit_time_et"),
                    _BotConfigField("Spot exit mode", "enum", "spot_exit_mode", options=("pct", "atr")),
                    _BotConfigField("Spot ATR period", "int", "spot_atr_period"),
                    _BotConfigField("Spot PT ATR mult", "float", "spot_pt_atr_mult"),
                    _BotConfigField("Spot SL ATR mult", "float", "spot_sl_atr_mult"),
                    _BotConfigField("Spot PT %", "float", "spot_profit_target_pct"),
                    _BotConfigField("Spot SL %", "float", "spot_stop_loss_pct"),
                    _BotConfigField("Spot up action", "enum", "directional_spot.up.action", options=("", "BUY")),
                    _BotConfigField("Spot up qty", "int", "directional_spot.up.qty"),
                    _BotConfigField("Spot down action", "enum", "directional_spot.down.action", options=("", "SELL")),
                    _BotConfigField("Spot down qty", "int", "directional_spot.down.qty"),
                ]
            )
            return

        self._fields.extend(
            [
                _BotConfigField("DTE", "int", "dte"),
                _BotConfigField("Profit target (PT)", "float", "profit_target"),
                _BotConfigField("Stop loss (SL)", "float", "stop_loss"),
                _BotConfigField("Exit DTE", "int", "exit_dte"),
                _BotConfigField("Stop loss basis", "enum", "stop_loss_basis", options=("max_loss", "credit")),
                _BotConfigField(
                    "Price mode",
                    "enum",
                    "price_mode",
                    options=("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"),
                ),
                _BotConfigField("Chase proposals", "bool", "chase_proposals"),
            ]
        )

        legs = self._strategy.get("legs", [])
        if not isinstance(legs, list):
            legs = []
            self._strategy["legs"] = legs
        for idx in range(len(legs)):
            prefix = f"legs.{idx}."
            self._fields.extend(
                [
                    _BotConfigField(f"Leg {idx+1} action", "enum", prefix + "action", options=("BUY", "SELL")),
                    _BotConfigField(f"Leg {idx+1} right", "enum", prefix + "right", options=("CALL", "PUT")),
                    _BotConfigField(f"Leg {idx+1} moneyness %", "float", prefix + "moneyness_pct"),
                    _BotConfigField(f"Leg {idx+1} delta", "float", prefix + "delta"),
                    _BotConfigField(f"Leg {idx+1} qty", "int", prefix + "qty"),
                ]
            )

    def _refresh_table(self) -> None:
        self._table.clear()
        mode = "Create" if self._mode == "create" else "Update"
        legs_desc = _legs_label(self._strategy.get("legs", []))
        dir_hint = _legs_direction_hint(
            self._strategy.get("legs", []),
            bool(self._strategy.get("ema_directional")),
        )
        lines = [
            Text(f"{mode}: {self._group}  ({self._symbol})", style="bold"),
            Text(f"Legs: {legs_desc}   Dir: {dir_hint}", style="dim"),
        ]
        self._header.update(Text("\n").join(lines))
        for field in self._fields:
            if self._editing and self._editing.path == field.path:
                self._table.add_row(field.label, self._edit_buffer)
                continue
            value = self._get_value(field)
            self._table.add_row(field.label, self._format_value(field, value))

    def _format_value(self, field: _BotConfigField, value: object) -> str:
        if field.kind == "bool":
            return "ON" if bool(value) else "OFF"
        if value is None:
            return ""
        if field.kind == "text" and isinstance(value, list):
            return ",".join(str(v) for v in value if v is not None)
        if field.kind == "float":
            try:
                return f"{float(value):.4g}"
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def _selected_field(self) -> _BotConfigField | None:
        row = self._table.cursor_coordinate.row
        if row < 0 or row >= len(self._fields):
            return None
        return self._fields[row]

    def _get_value(self, field: _BotConfigField) -> object:
        if field.path == "auto_trade":
            return self._auto_trade
        if field.path == "symbol":
            return self._symbol
        if field.path.startswith("filters."):
            if not self._filters:
                return None
            return _get_path(self._filters, field.path[len("filters.") :])
        return _get_path(self._strategy, field.path)

    def _set_value(self, field: _BotConfigField, value: object) -> None:
        if field.path == "auto_trade":
            self._auto_trade = bool(value)
            return
        if field.path == "symbol":
            self._symbol = str(value or "").strip().upper()
            return
        if field.path.startswith("filters."):
            if self._filters is None:
                self._filters = {}
            _set_path(self._filters, field.path[len("filters.") :], value)
            return
        _set_path(self._strategy, field.path, value)

    def action_cursor_down(self) -> None:
        if self._editing is not None:
            self._editing = None
            self._edit_buffer = ""
        self._table.action_cursor_down()

    def action_cursor_up(self) -> None:
        if self._editing is not None:
            self._editing = None
            self._edit_buffer = ""
        self._table.action_cursor_up()

    def action_toggle_edit(self) -> None:
        field = self._selected_field()
        if not field or field.kind not in ("int", "float", "text"):
            return
        if self._editing and self._editing.path == field.path:
            self._editing = None
            self._edit_buffer = ""
        else:
            self._editing = field
            self._edit_buffer = self._format_value(field, self._get_value(field))
        self._refresh_table()

    def action_toggle_bool(self) -> None:
        field = self._selected_field()
        if not field or field.kind != "bool":
            return
        current = bool(self._get_value(field))
        self._set_value(field, not current)
        self._refresh_table()

    def action_cycle_left(self) -> None:
        self._cycle_enum(-1)

    def action_cycle_right(self) -> None:
        self._cycle_enum(1)

    def _cycle_enum(self, direction: int) -> None:
        field = self._selected_field()
        if not field or field.kind != "enum" or not field.options:
            return
        current = self._get_value(field)
        current_str = "" if current is None else str(current)
        options = list(field.options)
        if current_str not in options:
            idx = 0
        else:
            idx = options.index(current_str)
        new_val = options[(idx + direction) % len(options)]
        if field.path == "ema_preset" and new_val == "":
            self._set_value(field, None)
        else:
            self._set_value(field, new_val)
        if field.path == "instrument":
            self._build_fields()
        self._refresh_table()

    def on_key(self, event: events.Key) -> None:
        if not self._editing:
            field = self._selected_field()
            if not field or not event.character:
                return
            char = event.character
            if field.kind == "int" and char.isdigit():
                self._editing = field
                self._edit_buffer = ""
            elif field.kind == "float" and (char in "0123456789." or char == "-"):
                self._editing = field
                self._edit_buffer = ""
            elif field.kind == "text":
                if event.key in (
                    "j",
                    "k",
                    "up",
                    "down",
                    "left",
                    "right",
                    "enter",
                    "escape",
                    "q",
                    "e",
                    "space",
                ):
                    return
                self._editing = field
                self._edit_buffer = ""
            else:
                return
        if event.key == "backspace":
            self._edit_buffer = self._edit_buffer[:-1]
            self._apply_edit_buffer()
            event.stop()
            return
        if not event.character:
            return
        char = event.character
        if self._editing.kind == "int":
            if char not in "0123456789":
                return
            self._edit_buffer = _append_digit(self._edit_buffer, char, allow_decimal=False)
            self._apply_edit_buffer()
            event.stop()
            return
        if self._editing.kind == "float":
            if char == "-" and not self._edit_buffer.startswith("-"):
                self._edit_buffer = "-" + self._edit_buffer
            else:
                if char not in "0123456789.":
                    return
                self._edit_buffer = _append_digit(self._edit_buffer, char, allow_decimal=True)
            self._apply_edit_buffer()
            event.stop()
            return
        if self._editing.kind == "text":
            if char:
                self._edit_buffer += char
                self._apply_edit_buffer()
                event.stop()

    def _apply_edit_buffer(self) -> None:
        field = self._editing
        if not field:
            return
        if field.kind == "text":
            if field.path == "entry_days":
                self._set_value(field, _parse_entry_days(self._edit_buffer))
                self._refresh_table()
                return
            if field.path.startswith("filters."):
                cleaned = self._edit_buffer.strip()
                value = None
                if cleaned.isdigit():
                    parsed = int(cleaned)
                    value = max(0, min(parsed, 23))
                self._set_value(field, value)
                self._refresh_table()
                return
            self._set_value(field, self._edit_buffer.strip())
            self._refresh_table()
            return
        if field.kind == "int":
            if not self._edit_buffer.isdigit():
                return
            parsed = int(self._edit_buffer)
            if field.path.endswith(".qty"):
                parsed = max(1, parsed)
            else:
                parsed = max(0, parsed)
            self._set_value(field, parsed)
            self._refresh_table()
            return
        if not self._edit_buffer.strip():
            self._set_value(field, None)
            self._refresh_table()
            return
        parsed = _parse_float(self._edit_buffer)
        if parsed is None:
            return
        self._set_value(field, float(parsed))
        self._refresh_table()

    def action_save(self) -> None:
        result = _BotConfigResult(
            mode=self._mode,
            instance_id=self._instance_id,
            group=self._group,
            symbol=self._symbol,
            strategy=self._strategy,
            filters=self._filters,
            auto_trade=self._auto_trade,
        )
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)


class BotScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("ctrl+t", "app.pop_screen", "Back"),
        ("ctrl+a", "toggle_presets", "Presets"),
        ("v", "toggle_scope", "Scope"),
        ("tab", "cycle_focus", "Focus"),
        ("h", "focus_prev", "Prev"),
        ("l", "focus_next", "Next"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("up", "cursor_up", "Up"),
        ("down", "cursor_down", "Down"),
        ("enter", "activate", "Select"),
        ("a", "toggle_auto_trade", "Auto"),
        ("space", "toggle_instance", "Run"),
        ("s", "stop_bot", "Stop"),
        ("d", "delete_instance", "Del"),
        ("p", "propose", "Propose"),
        ("f", "cycle_dte_filter", "Filter"),
        ("w", "cycle_win_filter", "Win"),
        ("r", "reload", "Reload"),
    ]

    def __init__(self, client: IBKRClient, refresh_sec: float) -> None:
        super().__init__()
        self._client = client
        self._refresh_sec = max(refresh_sec, 0.25)
        self._leaderboard_path = (
            Path(__file__).resolve().parent / "backtest" / "leaderboard.json"
        )
        self._spot_milestones_path = (
            Path(__file__).resolve().parent / "backtest" / "spot_milestones.json"
        )
        self._payload: dict | None = None
        self._presets: list[_BotPreset] = []
        self._preset_rows: list[_BotPreset | None] = []
        self._presets_visible = True
        self._filter_dte: int | None = None
        self._filter_min_win_rate: float | None = 0.5
        self._instances: list[_BotInstance] = []
        self._instance_rows: list[_BotInstance] = []
        self._next_instance_id = 1
        self._proposals: list[_BotProposal] = []
        self._proposal_rows: list[_BotProposal] = []
        self._positions: list[PortfolioItem] = []
        self._position_rows: list[PortfolioItem] = []
        self._status: str | None = None
        self._refresh_task = None
        self._proposal_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._tracked_conids: set[int] = set()
        self._active_panel = "presets"
        self._refresh_lock = asyncio.Lock()
        self._scope_all = False
        self._last_chase_ts = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static("", id="bot-status"),
            DataTable(id="bot-presets", zebra_stripes=True),
            DataTable(id="bot-instances", zebra_stripes=True),
            DataTable(id="bot-proposals", zebra_stripes=True),
            DataTable(id="bot-positions", zebra_stripes=True),
            id="bot-body",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._presets_table = self.query_one("#bot-presets", DataTable)
        self._status_panel = self.query_one("#bot-status", Static)
        self._proposals_table = self.query_one("#bot-proposals", DataTable)
        self._instances_table = self.query_one("#bot-instances", DataTable)
        self._positions_table = self.query_one("#bot-positions", DataTable)
        self._presets_table.border_title = "Presets"
        self._instances_table.border_title = "Instances"
        self._proposals_table.border_title = "Orders / Proposals"
        self._positions_table.border_title = "Bot Positions"
        self._presets_table.cursor_type = "row"
        self._proposals_table.cursor_type = "row"
        self._instances_table.cursor_type = "row"
        self._positions_table.cursor_type = "row"
        self._setup_tables()
        self._load_leaderboard()
        self._presets_table.display = self._presets_visible
        await self._refresh_positions()
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._render_status()
        self._focus_panel("presets")
        self._refresh_task = self.set_interval(self._refresh_sec, self._on_refresh_tick)

    async def on_unmount(self) -> None:
        if self._refresh_task:
            self._refresh_task.stop()
        for con_id in list(self._tracked_conids):
            self._client.release_ticker(con_id)
        self._tracked_conids.clear()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable captures Enter and emits RowSelected; hook it so Enter arms/sends.
        table_id = getattr(event.control, "id", None)
        if table_id == "bot-presets":
            self._active_panel = "presets"
        elif table_id == "bot-instances":
            self._active_panel = "instances"
        elif table_id == "bot-proposals":
            self._active_panel = "proposals"
        elif table_id == "bot-positions":
            self._active_panel = "positions"
        self.action_activate()

    def on_key(self, event: events.Key) -> None:
        if event.character == "X":
            self._submit_selected_proposal()
            event.stop()
            return
        if event.character == "S":
            self._kill_all()
            event.stop()
            return

    def action_cursor_down(self) -> None:
        self._cursor_move(1)

    def action_cursor_up(self) -> None:
        self._cursor_move(-1)

    def action_reload(self) -> None:
        self._load_leaderboard()
        self._status = "Reloaded leaderboard"
        self._render_status()

    def action_toggle_presets(self) -> None:
        self._presets_visible = not self._presets_visible
        self._presets_table.display = self._presets_visible
        if not self._presets_visible and self._active_panel == "presets":
            self._focus_panel("instances")
        elif self._presets_visible and self._active_panel != "presets":
            self._focus_panel("presets")
        self._status = f"Presets: {'ON' if self._presets_visible else 'OFF'}"
        self._render_status()
        self.refresh(layout=True)

    def action_toggle_scope(self) -> None:
        self._scope_all = not self._scope_all
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = "Scope: ALL" if self._scope_all else "Scope: Instance"
        self._render_status()
        self.refresh(layout=True)

    def action_cycle_focus(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev(self) -> None:
        self._cycle_focus(-1)

    def action_focus_next(self) -> None:
        self._cycle_focus(1)

    def _cycle_focus(self, direction: int) -> None:
        panels = ["presets", "instances", "proposals", "positions"]
        if not self._presets_visible:
            panels.remove("presets")
        try:
            idx = panels.index(self._active_panel)
        except ValueError:
            idx = 0
        self._focus_panel(panels[(idx + direction) % len(panels)])

    def _focus_panel(self, panel: str) -> None:
        self._active_panel = panel
        if panel == "presets":
            self._presets_table.focus()
            self._skip_preset_headers(1)
        elif panel == "instances":
            self._instances_table.focus()
        elif panel == "proposals":
            self._proposals_table.focus()
        else:
            self._positions_table.focus()
        self._render_status()

    def action_toggle_auto_trade(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Auto: select an instance"
            self._render_status()
            return
        instance.auto_trade = not instance.auto_trade
        self._refresh_instances_table()
        self._status = f"Instance {instance.instance_id}: auto trade {'ON' if instance.auto_trade else 'OFF'}"
        self._render_status()

    def action_toggle_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Run: select an instance"
            self._render_status()
            return
        instance.state = "PAUSED" if instance.state == "RUNNING" else "RUNNING"
        self._refresh_instances_table()
        self._status = f"Instance {instance.instance_id}: {instance.state}"
        self._render_status()

    def action_delete_instance(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Del: select an instance"
            self._render_status()
            return
        self._instances = [i for i in self._instances if i.instance_id != instance.instance_id]
        self._proposals = [p for p in self._proposals if p.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._status = f"Deleted instance {instance.instance_id}"
        self._render_status()

    def action_cycle_dte_filter(self) -> None:
        dtes: list[int] = []
        if self._payload:
            raw = self._payload.get("grid", {}).get("dte", [])
            for value in raw if isinstance(raw, list) else []:
                try:
                    dtes.append(int(value))
                except (TypeError, ValueError):
                    continue
        dtes = sorted(set(dtes))
        cycle: list[int | None] = [None]
        if 0 not in dtes:
            cycle.append(0)
        cycle.extend(dtes)
        try:
            idx = cycle.index(self._filter_dte)
        except ValueError:
            idx = 0
        self._filter_dte = cycle[(idx + 1) % len(cycle)]
        self._rebuild_presets_table()
        self._status = (
            "Filter: DTE=ALL" if self._filter_dte is None else f"Filter: DTE={self._filter_dte}"
        )
        self._render_status()
        self.refresh(layout=True)

    def action_cycle_win_filter(self) -> None:
        cycle: list[float | None] = [0.5, 0.6, 0.7, 0.8, None]
        try:
            idx = cycle.index(self._filter_min_win_rate)
        except ValueError:
            idx = 0
        self._filter_min_win_rate = cycle[(idx + 1) % len(cycle)]
        self._rebuild_presets_table()
        if self._filter_min_win_rate is None:
            self._status = "Filter: Win=ALL"
        else:
            self._status = f"Filter: Win≥{int(self._filter_min_win_rate * 100)}%"
        self._render_status()
        self.refresh(layout=True)

    def action_stop_bot(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Stop: select an instance"
            self._render_status()
            return
        instance.state = "PAUSED"
        instance.auto_trade = False
        self._proposals = [p for p in self._proposals if p.instance_id != instance.instance_id]
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = f"Stopped instance {instance.instance_id}: paused + auto OFF + cleared proposals"
        self._render_status()

    def _kill_all(self) -> None:
        for instance in self._instances:
            instance.state = "PAUSED"
            instance.auto_trade = False
        self._proposals.clear()
        self._refresh_instances_table()
        self._refresh_proposals_table()
        self._refresh_positions_table()
        self._status = "KILL: paused all + auto OFF + cleared proposals"
        self._render_status()

    def action_activate(self) -> None:
        if self._active_panel == "presets":
            preset = self._selected_preset()
            if not preset:
                self._status = "Preset: none selected"
                self._render_status()
                return
            self._open_config_for_preset(preset)
            return
        if self._active_panel == "instances":
            instance = self._selected_instance()
            if not instance:
                self._status = "Instance: none selected"
                self._render_status()
                return
            self._open_config_for_instance(instance)
            return
        if self._active_panel == "proposals":
            self._submit_selected_proposal()
            return
        self._status = "Positions: no action"
        self._render_status()

    def action_propose(self) -> None:
        instance = self._selected_instance()
        if not instance:
            self._status = "Propose: select an instance"
            self._render_status()
            return
        self._queue_proposal(instance, intent="enter", direction=None, signal_bar_ts=None)

    def _selected_preset(self) -> _BotPreset | None:
        row = self._presets_table.cursor_coordinate.row
        if row < 0 or row >= len(self._preset_rows):
            return None
        return self._preset_rows[row]

    def _selected_instance(self) -> _BotInstance | None:
        row = self._instances_table.cursor_coordinate.row
        if row < 0 or row >= len(self._instance_rows):
            return None
        return self._instance_rows[row]

    def _selected_proposal(self) -> _BotProposal | None:
        row = self._proposals_table.cursor_coordinate.row
        if row < 0 or row >= len(self._proposal_rows):
            return None
        return self._proposal_rows[row]

    def _scope_instance_id(self) -> int | None:
        if self._scope_all:
            return None
        instance = self._selected_instance()
        return instance.instance_id if instance else None

    def _open_config_for_preset(self, preset: _BotPreset) -> None:
        entry = preset.entry
        strategy = copy.deepcopy(entry.get("strategy", {}) or {})
        strategy.setdefault("instrument", "options")
        strategy.setdefault("price_mode", "OPTIMISTIC")
        strategy.setdefault("chase_proposals", True)
        strategy.setdefault("max_entries_per_day", 1)
        strategy.setdefault("exit_dte", 0)
        strategy.setdefault("stop_loss_basis", "max_loss")
        strategy.setdefault("spot_sec_type", "STK")
        strategy.setdefault("spot_exchange", "")
        if "signal_bar_size" not in strategy:
            strategy["signal_bar_size"] = str(self._payload.get("bar_size", "1 hour") if self._payload else "1 hour")
        if "signal_use_rth" not in strategy:
            strategy["signal_use_rth"] = bool(self._payload.get("use_rth", False) if self._payload else False)
        strategy.setdefault("spot_close_eod", False)
        if "directional_spot" not in strategy:
            strategy["directional_spot"] = {"up": {"action": "BUY", "qty": 1}, "down": {"action": "SELL", "qty": 1}}
        filters = _filters_for_group(self._payload, preset.group) if self._payload else None
        symbol = str(
            entry.get("symbol")
            or strategy.get("symbol")
            or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()

        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._status = "Config: cancelled"
                self._render_status()
                return
            instance = _BotInstance(
                instance_id=self._next_instance_id,
                group=result.group,
                symbol=result.symbol,
                strategy=result.strategy,
                filters=result.filters,
                metrics=entry.get("metrics"),
                auto_trade=result.auto_trade,
            )
            self._next_instance_id += 1
            self._instances.append(instance)
            self._refresh_instances_table()
            self._instances_table.cursor_coordinate = (max(len(self._instances) - 1, 0), 0)
            self._focus_panel("instances")
            self._status = f"Created instance {instance.instance_id}"
            self._render_status()

        self.app.push_screen(
            BotConfigScreen(
                mode="create",
                instance_id=None,
                group=preset.group,
                symbol=symbol,
                strategy=strategy,
                filters=filters,
                auto_trade=False,
            ),
            _on_done,
        )

    def _open_config_for_instance(self, instance: _BotInstance) -> None:
        def _on_done(result: _BotConfigResult | None) -> None:
            if not result:
                self._status = "Config: cancelled"
                self._render_status()
                return
            instance.group = result.group
            instance.symbol = result.symbol
            instance.strategy = result.strategy
            instance.filters = result.filters
            instance.auto_trade = result.auto_trade
            self._refresh_instances_table()
            self._status = f"Updated instance {instance.instance_id}"
            self._render_status()

        self.app.push_screen(
            BotConfigScreen(
                mode="update",
                instance_id=instance.instance_id,
                group=instance.group,
                symbol=instance.symbol,
                strategy=instance.strategy,
                filters=instance.filters,
                auto_trade=instance.auto_trade,
            ),
            _on_done,
        )

    def _load_leaderboard(self) -> None:
        try:
            payload = json.loads(self._leaderboard_path.read_text())
        except Exception as exc:
            self._payload = None
            self._presets = []
            self._preset_rows = []
            self._presets_table.clear(columns=True)
            self._presets_table.add_columns("Error")
            self._presets_table.add_row(str(exc))
            return
        if not isinstance(payload, dict):
            payload = {}

        try:
            if self._spot_milestones_path.exists():
                spot_payload = json.loads(self._spot_milestones_path.read_text())
                if isinstance(spot_payload, dict):
                    spot_groups = spot_payload.get("groups", [])
                    if isinstance(spot_groups, list) and spot_groups:
                        base_groups = payload.get("groups", [])
                        if not isinstance(base_groups, list):
                            base_groups = []
                        merged = list(base_groups)
                        seen = {
                            str(group.get("name"))
                            for group in merged
                            if isinstance(group, dict) and group.get("name") is not None
                        }
                        for group in spot_groups:
                            if not isinstance(group, dict):
                                continue
                            name = str(group.get("name"))
                            if name and name in seen:
                                continue
                            merged.append(group)
                            if name:
                                seen.add(name)
                        payload["groups"] = merged
        except Exception:
            # Keep the main leaderboard usable even if the milestones file is malformed.
            pass

        self._payload = payload
        self._rebuild_presets_table()

    def _rebuild_presets_table(self) -> None:
        self._presets = []
        self._preset_rows = []
        self._presets_table.clear(columns=True)
        self._presets_table.add_columns(
            "Group",
            "Legs",
            "DTE",
            "PT",
            "SL",
            "EMA",
            "PnL",
            "Win",
            "Tr",
        )
        payload = self._payload or {}
        for group in payload.get("groups", []):
            group_name = str(group.get("name", "?"))
            visible_entries: list[dict] = []
            for entry in group.get("entries", []):
                strat = entry.get("strategy", {})
                metrics = entry.get("metrics", {})
                instrument = str(strat.get("instrument", "options") or "options").strip().lower()
                if instrument not in ("options", "spot"):
                    continue
                legs = strat.get("legs", [])
                if instrument == "options":
                    if not isinstance(legs, list) or not legs:
                        continue
                try:
                    dte = int(strat.get("dte", 0))
                except (TypeError, ValueError):
                    dte = 0
                if self._filter_dte is not None and dte != self._filter_dte:
                    continue
                try:
                    win = float(metrics.get("win_rate", 0.0))
                except (TypeError, ValueError):
                    win = 0.0
                if self._filter_min_win_rate is not None and win < self._filter_min_win_rate:
                    continue
                visible_entries.append(entry)

            if not visible_entries:
                continue

            header = Text(group_name, style="bold")
            self._preset_rows.append(None)
            self._presets_table.add_row(header, "", "", "", "", "", "", "", "")
            for entry in visible_entries:
                preset = _BotPreset(group=group_name, entry=entry)
                self._presets.append(preset)
                metrics = entry.get("metrics", {})
                strat = entry.get("strategy", {})
                instrument = str(strat.get("instrument", "options") or "options").strip().lower()
                try:
                    dte = int(strat.get("dte", 0))
                except (TypeError, ValueError):
                    dte = 0
                if instrument == "spot":
                    sec_type = str(strat.get("spot_sec_type") or "").strip().upper()
                    legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT"
                else:
                    legs_desc = _legs_label(strat.get("legs", []))
                self._preset_rows.append(preset)
                self._presets_table.add_row(
                    group_name,
                    legs_desc,
                    "-" if instrument == "spot" else str(dte),
                    "-" if instrument == "spot" else _fmt_pct(float(strat.get("profit_target", 0.0)) * 100.0),
                    "-" if instrument == "spot" else _fmt_pct(float(strat.get("stop_loss", 0.0)) * 100.0),
                    str(strat.get("ema_preset", "")),
                    f"{float(metrics.get('pnl', 0.0)):.2f}",
                    _fmt_pct(float(metrics.get("win_rate", 0.0)) * 100.0),
                    str(int(metrics.get("trades", 0))),
                )

        self._move_cursor_to_first_preset()
        self._render_status()
        self._presets_table.refresh(repaint=True)

    def _move_cursor_to_first_preset(self) -> None:
        if not self._preset_rows:
            return
        for idx, preset in enumerate(self._preset_rows):
            if preset is not None:
                self._presets_table.cursor_coordinate = (idx, 0)
                return

    def _skip_preset_headers(self, direction: int) -> None:
        if self._active_panel != "presets":
            return
        if not self._preset_rows:
            return
        row = self._presets_table.cursor_coordinate.row
        row = max(0, min(row, len(self._preset_rows) - 1))
        if self._preset_rows[row] is not None:
            return
        scan = row
        while 0 <= scan < len(self._preset_rows) and self._preset_rows[scan] is None:
            scan += direction
        if 0 <= scan < len(self._preset_rows) and self._preset_rows[scan] is not None:
            self._presets_table.cursor_coordinate = (scan, 0)
            return

    def _setup_tables(self) -> None:
        self._instances_table.clear(columns=True)
        self._instances_table.add_columns("ID", "Strategy", "Legs", "DTE", "Auto", "State", "BT PnL")

        self._proposals_table.clear(columns=True)
        self._proposals_table.add_columns("When", "Inst", "Side", "Qty", "Contract", "Lmt", "B/A", "Status")

        self._positions_table.clear(columns=True)
        self._positions_table.add_columns(
            "Symbol",
            "Expiry",
            "Right",
            "Strike",
            "Qty",
            "AvgCost",
            "Px 24-72",
            "Unreal",
            "Realized",
            "U 24-72",
        )

    def _cursor_move(self, direction: int) -> None:
        if self._active_panel == "presets":
            if direction > 0:
                self._presets_table.action_cursor_down()
                self._skip_preset_headers(1)
            else:
                self._presets_table.action_cursor_up()
                self._skip_preset_headers(-1)
        elif self._active_panel == "instances":
            if direction > 0:
                self._instances_table.action_cursor_down()
            else:
                self._instances_table.action_cursor_up()
        elif self._active_panel == "proposals":
            if direction > 0:
                self._proposals_table.action_cursor_down()
            else:
                self._proposals_table.action_cursor_up()
        else:
            if direction > 0:
                self._positions_table.action_cursor_down()
            else:
                self._positions_table.action_cursor_up()
        if self._active_panel == "instances" and not self._scope_all:
            self._refresh_proposals_table()
            self._refresh_positions_table()
        self._render_status()

    def _refresh_instances_table(self) -> None:
        prev_row = self._instances_table.cursor_coordinate.row
        prev_id = None
        if 0 <= prev_row < len(self._instance_rows):
            prev_id = self._instance_rows[prev_row].instance_id

        self._instances_table.clear()
        self._instance_rows = []
        for instance in self._instances:
            instrument = self._strategy_instrument(instance.strategy or {})
            if instrument == "spot":
                sec_type = str((instance.strategy or {}).get("spot_sec_type") or "").strip().upper()
                legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT-STK"
                dte = "-"
            else:
                legs_desc = _legs_label(instance.strategy.get("legs", []))
                dte = instance.strategy.get("dte", "")
            auto = "ON" if instance.auto_trade else "OFF"
            bt_pnl = ""
            if instance.metrics:
                try:
                    bt_pnl = f"{float(instance.metrics.get('pnl', 0.0)):.0f}"
                except (TypeError, ValueError):
                    bt_pnl = ""
            self._instances_table.add_row(
                str(instance.instance_id),
                instance.group[:24],
                legs_desc[:24],
                str(dte),
                auto,
                instance.state,
                bt_pnl,
            )
            self._instance_rows.append(instance)

        if prev_id is not None:
            for idx, inst in enumerate(self._instance_rows):
                if inst.instance_id == prev_id:
                    self._instances_table.cursor_coordinate = (idx, 0)
                    break
        elif self._instance_rows:
            self._instances_table.cursor_coordinate = (0, 0)

        if not self._scope_all:
            self._refresh_proposals_table()
            self._refresh_positions_table()

    async def _refresh_positions(self) -> None:
        try:
            items = await self._client.fetch_portfolio()
        except Exception as exc:  # pragma: no cover - UI surface
            self._status = f"Positions error: {exc}"
            return
        self._positions = [item for item in items if item.contract.secType in _SECTION_TYPES]
        self._positions.sort(key=_portfolio_sort_key, reverse=True)
        self._refresh_positions_table()

    def _refresh_positions_table(self) -> None:
        self._positions_table.clear()
        self._position_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            return
        if self._scope_all:
            con_ids = set().union(*(i.touched_conids for i in self._instances))
        else:
            instance = next((i for i in self._instances if i.instance_id == scope), None)
            con_ids = set(instance.touched_conids) if instance else set()
        if not con_ids:
            return
        for item in self._positions:
            con_id = int(getattr(item.contract, "conId", 0) or 0)
            if con_id not in con_ids:
                continue
            self._positions_table.add_row(*_portfolio_row(item, Text(""), Text("")))
            self._position_rows.append(item)

    def _render_status(self) -> None:
        self._render_bot()

    async def _on_refresh_tick(self) -> None:
        if self._refresh_lock.locked():
            return
        async with self._refresh_lock:
            await self._refresh_positions()
            await self._auto_propose_tick()
            await self._chase_proposals_tick()
            self._auto_send_tick()
            self._render_status()

    async def _auto_propose_tick(self) -> None:
        if self._proposal_task and not self._proposal_task.done():
            return
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))

        for instance in self._instances:
            if instance.state != "RUNNING":
                continue
            if not self._can_propose_now(instance):
                continue

            pending = any(
                p.status == "PROPOSED" and p.instance_id == instance.instance_id for p in self._proposals
            )
            if pending:
                continue

            symbol = str(
                instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
            ).strip().upper()

            entry_signal = str(instance.strategy.get("entry_signal") or "ema").strip().lower()
            if entry_signal not in ("ema", "orb"):
                entry_signal = "ema"
            ema_preset = instance.strategy.get("ema_preset")
            if entry_signal == "ema" and not ema_preset:
                continue

            signal_contract = await self._signal_contract(instance, symbol)
            if signal_contract is None:
                continue

            snap = await self._signal_snapshot_for_contract(
                contract=signal_contract,
                ema_preset_raw=str(ema_preset) if ema_preset else None,
                bar_size=self._signal_bar_size(instance),
                use_rth=self._signal_use_rth(instance),
                entry_signal_raw=entry_signal,
                entry_mode_raw=instance.strategy.get("ema_entry_mode"),
                entry_confirm_bars=instance.strategy.get("entry_confirm_bars", 0),
                orb_window_mins_raw=instance.strategy.get("orb_window_mins"),
                regime_ema_preset_raw=instance.strategy.get("regime_ema_preset"),
                regime_bar_size_raw=instance.strategy.get("regime_bar_size"),
                regime_mode_raw=instance.strategy.get("regime_mode"),
                supertrend_atr_period_raw=instance.strategy.get("supertrend_atr_period"),
                supertrend_multiplier_raw=instance.strategy.get("supertrend_multiplier"),
                supertrend_source_raw=instance.strategy.get("supertrend_source"),
                regime2_ema_preset_raw=instance.strategy.get("regime2_ema_preset"),
                regime2_bar_size_raw=instance.strategy.get("regime2_bar_size"),
                regime2_mode_raw=instance.strategy.get("regime2_mode"),
                regime2_supertrend_atr_period_raw=instance.strategy.get("regime2_supertrend_atr_period"),
                regime2_supertrend_multiplier_raw=instance.strategy.get("regime2_supertrend_multiplier"),
                regime2_supertrend_source_raw=instance.strategy.get("regime2_supertrend_source"),
                filters=instance.filters if isinstance(instance.filters, dict) else None,
                spot_exit_mode_raw=instance.strategy.get("spot_exit_mode"),
                spot_atr_period_raw=instance.strategy.get("spot_atr_period"),
            )
            if snap is None:
                continue

            entry_days = instance.strategy.get("entry_days", [])
            if entry_days:
                allowed_days = {_weekday_num(day) for day in entry_days}
            else:
                allowed_days = {0, 1, 2, 3, 4}
            if snap.bar_ts.weekday() not in allowed_days:
                continue

            cooldown_bars = 0
            if isinstance(instance.filters, dict):
                raw = instance.filters.get("cooldown_bars", 0)
                try:
                    cooldown_bars = int(raw or 0)
                except (TypeError, ValueError):
                    cooldown_bars = 0
            cooldown_ok = cooldown_ok_by_time(
                current_bar_ts=snap.bar_ts,
                last_entry_bar_ts=instance.last_entry_bar_ts,
                bar_size=self._signal_bar_size(instance),
                cooldown_bars=cooldown_bars,
            )
	            if not signal_filters_ok(
	                instance.filters,
	                bar_ts=snap.bar_ts,
	                bars_in_day=snap.bars_in_day,
	                close=float(snap.close),
	                volume=snap.volume,
	                volume_ema=snap.volume_ema,
	                volume_ema_ready=snap.volume_ema_ready,
	                rv=snap.rv,
	                signal=snap.signal,
	                cooldown_ok=cooldown_ok,
	                shock=snap.shock,
	                shock_dir=snap.shock_dir,
	            ):
	                continue

            instrument = self._strategy_instrument(instance.strategy)
            open_items: list[PortfolioItem] = []
            open_dir: str | None = None
            if instrument == "spot":
                open_item = self._spot_open_position(
                    symbol=symbol,
                    sec_type=str(getattr(signal_contract, "secType", "") or "STK"),
                    con_id=int(getattr(signal_contract, "conId", 0) or 0),
                )
                if open_item is not None:
                    open_items = [open_item]
                    try:
                        pos = float(getattr(open_item, "position", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pos = 0.0
                    open_dir = "up" if pos > 0 else "down" if pos < 0 else None
            else:
                open_items = self._options_open_positions(instance)
                open_dir = instance.open_direction or self._open_direction_from_positions(open_items)

            if not open_items and instance.open_direction is not None:
                instance.open_direction = None
                instance.spot_profit_target_price = None
                instance.spot_stop_loss_price = None

            if open_items:
                if instance.last_exit_bar_ts is not None and instance.last_exit_bar_ts == snap.bar_ts:
                    continue

                if instrument == "spot":
                    open_item = open_items[0]
                    try:
                        pos = float(getattr(open_item, "position", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pos = 0.0
                    avg_cost = _safe_num(getattr(open_item, "averageCost", None))
                    market_price = _safe_num(getattr(open_item, "marketPrice", None))
                    if market_price is None:
                        ticker = await self._client.ensure_ticker(open_item.contract)
                        market_price = _ticker_price(ticker)

                    target_price = instance.spot_profit_target_price
                    stop_price = instance.spot_stop_loss_price
                    if (
                        pos
                        and market_price is not None
                        and market_price > 0
                        and (target_price is not None or stop_price is not None)
                    ):
                        try:
                            mp = float(market_price)
                        except (TypeError, ValueError):
                            mp = None
                        if mp is not None:
                            if target_price is not None:
                                try:
                                    target = float(target_price)
                                except (TypeError, ValueError):
                                    target = None
                                if target is not None and target > 0:
                                    if (pos > 0 and mp >= target) or (pos < 0 and mp <= target):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                            if stop_price is not None:
                                try:
                                    stop = float(stop_price)
                                except (TypeError, ValueError):
                                    stop = None
                                if stop is not None and stop > 0:
                                    if (pos > 0 and mp <= stop) or (pos < 0 and mp >= stop):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                    move = None
                    if (
                        target_price is None
                        and stop_price is None
                        and avg_cost is not None
                        and avg_cost > 0
                        and market_price is not None
                        and market_price > 0
                        and pos
                    ):
                        move = (market_price - avg_cost) / avg_cost
                        if pos < 0:
                            move = -move

                    try:
                        pt = (
                            float(instance.strategy.get("spot_profit_target_pct"))
                            if instance.strategy.get("spot_profit_target_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        pt = None
                    try:
                        sl = (
                            float(instance.strategy.get("spot_stop_loss_pct"))
                            if instance.strategy.get("spot_stop_loss_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        sl = None

                    if move is not None and pt is not None and move >= pt:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break
                    if move is not None and sl is not None and move <= -sl:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    exit_time = parse_time_hhmm(instance.strategy.get("spot_exit_time_et"))
                    if exit_time is not None and now_et.time() >= exit_time:
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    if bool(instance.strategy.get("spot_close_eod")) and (
                        now_et.hour > 15 or now_et.hour == 15 and now_et.minute >= 55
                    ):
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                if instrument != "spot":
                    if self._should_exit_on_dte(instance, open_items, now_et.date()):
                        self._queue_proposal(
                            instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts
                        )
                        break

                    entry_value, current_value = self._options_position_values(open_items)
                    if entry_value is not None and current_value is not None:
                        profit = float(entry_value) - float(current_value)
                        try:
                            profit_target = float(instance.strategy.get("profit_target", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            profit_target = 0.0
                        if profit_target > 0 and abs(entry_value) > 0:
                            if profit >= abs(entry_value) * profit_target:
                                self._queue_proposal(
                                    instance,
                                    intent="exit",
                                    direction=open_dir,
                                    signal_bar_ts=snap.bar_ts,
                                )
                                break

                        try:
                            stop_loss = float(instance.strategy.get("stop_loss", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            stop_loss = 0.0
                        if stop_loss > 0:
                            loss = max(0.0, float(current_value) - float(entry_value))
                            basis = str(instance.strategy.get("stop_loss_basis") or "max_loss").strip().lower()
                            if basis == "credit":
                                if entry_value >= 0:
                                    if current_value >= entry_value * (1.0 + stop_loss):
                                        self._queue_proposal(
                                            instance,
                                            intent="exit",
                                            direction=open_dir,
                                            signal_bar_ts=snap.bar_ts,
                                        )
                                        break
                                elif loss >= abs(entry_value) * stop_loss:
                                    self._queue_proposal(
                                        instance,
                                        intent="exit",
                                        direction=open_dir,
                                        signal_bar_ts=snap.bar_ts,
                                    )
                                    break
                            else:
                                max_loss = self._options_max_loss_estimate(open_items, spot=float(snap.close))
                                if max_loss is None or max_loss <= 0:
                                    max_loss = abs(entry_value)
                                if max_loss and loss >= float(max_loss) * stop_loss:
                                    self._queue_proposal(
                                        instance,
                                        intent="exit",
                                        direction=open_dir,
                                        signal_bar_ts=snap.bar_ts,
                                    )
                                    break

                if self._should_exit_on_flip(instance, snap, open_dir, open_items):
                    self._queue_proposal(instance, intent="exit", direction=open_dir, signal_bar_ts=snap.bar_ts)
                    break
                continue

            if not self._entry_limit_ok(instance):
                continue
            if instance.last_entry_bar_ts is not None and instance.last_entry_bar_ts == snap.bar_ts:
                continue

            instrument = self._strategy_instrument(instance.strategy)
            if instrument == "spot":
                exit_mode = str(instance.strategy.get("spot_exit_mode") or "pct").strip().lower()
                if exit_mode == "atr":
                    atr = float(snap.atr or 0.0) if snap.atr is not None else 0.0
                    if atr <= 0:
                        continue

            direction = self._entry_direction_for_instance(instance, snap)
            if direction is None:
                continue
            if direction not in self._allowed_entry_directions(instance):
                continue

            self._queue_proposal(
                instance,
                intent="enter",
                direction=direction,
                signal_bar_ts=snap.bar_ts,
            )
            break

    def _auto_send_tick(self) -> None:
        if self._send_task and not self._send_task.done():
            return
        proposal = next((p for p in self._proposals if p.status == "PROPOSED"), None)
        if proposal is None:
            return
        instance = next((i for i in self._instances if i.instance_id == proposal.instance_id), None)
        if not instance or not instance.auto_trade:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Auto: no loop"
            return
        self._send_task = loop.create_task(self._send_order(proposal))

    async def _chase_proposals_tick(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        now = loop.time()
        if now - self._last_chase_ts < 0.5:
            return
        self._last_chase_ts = now

        updated = False
        for proposal in self._proposals:
            if proposal.status != "PROPOSED":
                continue
            instance = next(
                (i for i in self._instances if i.instance_id == proposal.instance_id), None
            )
            if not instance:
                continue
            if instance.strategy.get("chase_proposals") is False:
                continue
            changed = await self._reprice_proposal(proposal, instance)
            updated = updated or changed
        if updated:
            self._refresh_proposals_table()
            if self._active_panel == "proposals" and self._proposal_rows:
                row = min(
                    self._proposals_table.cursor_coordinate.row, len(self._proposal_rows) - 1
                )
                self._proposals_table.cursor_coordinate = (max(row, 0), 0)

    async def _reprice_proposal(self, proposal: _BotProposal, instance: _BotInstance) -> bool:
        raw_mode = str(instance.strategy.get("price_mode") or "OPTIMISTIC").strip().upper()
        if raw_mode not in ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"):
            raw_mode = "OPTIMISTIC"
        price_mode = raw_mode

        def _leg_price(
            bid: float | None, ask: float | None, last: float | None, action: str
        ) -> float | None:
            mid = _midpoint(bid, ask)
            if price_mode == "CROSS":
                return ask if action == "BUY" else bid
            if price_mode == "MID":
                return mid or last
            if price_mode == "OPTIMISTIC":
                return _optimistic_price(bid, ask, mid, action) or mid or last
            if price_mode == "AGGRESSIVE":
                return _aggressive_price(bid, ask, mid, action) or mid or last
            return mid or last

        legs = proposal.legs or []
        if not legs:
            return False

        if len(legs) == 1 and proposal.order_contract.secType != "BAG":
            leg = legs[0]
            ticker = await self._client.ensure_ticker(leg.contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _leg_price(bid, ask, last, proposal.action)
            if limit is None:
                return False
            tick = _tick_size(leg.contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            changed = not math.isclose(limit, proposal.limit_price, rel_tol=0, abs_tol=tick / 2.0)
            proposal.limit_price = float(limit)
            proposal.bid = bid
            proposal.ask = ask
            proposal.last = last
            return changed

        debit_mid = 0.0
        debit_bid = 0.0
        debit_ask = 0.0
        desired_debit = 0.0
        tick = None
        for leg in legs:
            ticker = await self._client.ensure_ticker(leg.contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            mid = _midpoint(bid, ask)
            leg_mid = mid or last
            if leg_mid is None:
                return False
            leg_bid = bid or mid or last
            leg_ask = ask or mid or last
            leg_desired = _leg_price(bid, ask, last, leg.action)
            if leg_bid is None or leg_ask is None or leg_desired is None:
                return False
            leg_tick = _tick_size(leg.contract, ticker, leg_desired)
            tick = leg_tick if tick is None else min(tick, leg_tick)
            sign = 1.0 if leg.action == "BUY" else -1.0
            debit_mid += sign * float(leg_mid) * leg.ratio
            debit_bid += sign * float(leg_bid) * leg.ratio
            debit_ask += sign * float(leg_ask) * leg.ratio
            desired_debit += sign * float(leg_desired) * leg.ratio

        tick = tick or 0.01
        proposal.action = "BUY"
        new_limit = _round_to_tick(float(desired_debit), tick)
        if not new_limit:
            return False
        new_bid = float(debit_bid)
        new_ask = float(debit_ask)
        new_last = float(debit_mid)
        changed = not math.isclose(
            new_limit, proposal.limit_price, rel_tol=0, abs_tol=tick / 2.0
        )
        proposal.limit_price = float(new_limit)
        proposal.bid = new_bid
        proposal.ask = new_ask
        proposal.last = new_last
        return changed

    def _queue_proposal(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        if self._proposal_task and not self._proposal_task.done():
            self._status = "Propose: busy"
            self._render_status()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Propose: no loop"
            self._render_status()
            return
        action = "Exiting" if intent == "exit" else "Proposing"
        dir_note = f" ({direction})" if direction else ""
        self._status = f"{action} for instance {instance.instance_id}{dir_note}..."
        self._render_status()
        self._proposal_task = loop.create_task(
            self._propose_for_instance(
                instance,
                intent=str(intent),
                direction=direction,
                signal_bar_ts=signal_bar_ts,
            )
        )

    def _can_propose_now(self, instance: _BotInstance) -> bool:
        entry_days = instance.strategy.get("entry_days", [])
        if entry_days:
            allowed = {_weekday_num(day) for day in entry_days}
        else:
            allowed = {0, 1, 2, 3, 4}
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        if now.weekday() not in allowed:
            return False
        return True

    def _strategy_instrument(self, strategy: dict) -> str:
        value = strategy.get("instrument", "options")
        cleaned = str(value or "options").strip().lower()
        return "spot" if cleaned == "spot" else "options"

    def _spot_sec_type(self, instance: _BotInstance, symbol: str) -> str:
        raw = (instance.strategy or {}).get("spot_sec_type")
        cleaned = str(raw or "").strip().upper()
        if cleaned in ("STK", "FUT"):
            return cleaned
        sym = str(symbol or "").strip().upper()
        if sym in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
            return "FUT"
        return "STK"

    def _spot_exchange(self, instance: _BotInstance, symbol: str, *, sec_type: str) -> str:
        raw = (instance.strategy or {}).get("spot_exchange")
        cleaned = str(raw or "").strip().upper()
        if cleaned:
            return cleaned
        if sec_type == "FUT":
            sym = str(symbol or "").strip().upper()
            if sym in ("YM", "MYM"):
                return "CBOT"
            return "CME"
        return "SMART"

    async def _spot_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        sec_type = self._spot_sec_type(instance, symbol)
        exchange = self._spot_exchange(instance, symbol, sec_type=sec_type)
        if sec_type == "FUT":
            contract = await self._client.front_future(symbol, exchange=exchange, cache_ttl_sec=3600.0)
            if contract is None:
                return None
            return contract

        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    async def _signal_contract(self, instance: _BotInstance, symbol: str) -> Contract | None:
        if self._strategy_instrument(instance.strategy) == "spot":
            return await self._spot_contract(instance, symbol)
        contract = Stock(symbol=str(symbol).strip().upper(), exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(contract)
        return qualified[0] if qualified else contract

    def _signal_bar_size(self, instance: _BotInstance) -> str:
        raw = instance.strategy.get("signal_bar_size")
        if raw:
            return str(raw)
        if self._payload:
            return str(self._payload.get("bar_size", "1 hour"))
        return "1 hour"

    def _signal_use_rth(self, instance: _BotInstance) -> bool:
        raw = instance.strategy.get("signal_use_rth")
        if raw is not None:
            return bool(raw)
        if self._payload:
            return bool(self._payload.get("use_rth", False))
        return False

    def _signal_duration_str(self, bar_size: str) -> str:
        label = str(bar_size or "").strip().lower()
        if label.startswith(("5 mins", "15 mins", "30 mins")):
            return "1 W"
        if "hour" in label:
            return "2 W"
        if "day" in label:
            return "1 Y"
        return "2 W"

    async def _signal_snapshot_for_contract(
        self,
        *,
        contract: Contract,
        ema_preset_raw: str | None,
        bar_size: str,
        use_rth: bool,
        entry_signal_raw: str | None = None,
        orb_window_mins_raw: int | None = None,
        entry_mode_raw: str | None = None,
        entry_confirm_bars: int = 0,
        spot_exit_mode_raw: str | None = None,
        spot_atr_period_raw: int | None = None,
        regime_ema_preset_raw: str | None = None,
        regime_bar_size_raw: str | None = None,
        regime_mode_raw: str | None = None,
        supertrend_atr_period_raw: int | None = None,
        supertrend_multiplier_raw: float | None = None,
        supertrend_source_raw: str | None = None,
        regime2_ema_preset_raw: str | None = None,
        regime2_bar_size_raw: str | None = None,
        regime2_mode_raw: str | None = None,
        regime2_supertrend_atr_period_raw: int | None = None,
        regime2_supertrend_multiplier_raw: float | None = None,
        regime2_supertrend_source_raw: str | None = None,
        filters: dict | None = None,
    ) -> _SignalSnapshot | None:
        entry_signal = str(entry_signal_raw or "ema").strip().lower()
        if entry_signal not in ("ema", "orb"):
            entry_signal = "ema"
        slow_p = None
        if entry_signal == "ema":
            periods = ema_periods(ema_preset_raw)
            if periods is None:
                return None
            _, slow_p = periods

        bars = await self._client.historical_bars_ohlcv(
            contract,
            duration_str=self._signal_duration_str(bar_size),
            bar_size=bar_size,
            use_rth=use_rth,
            cache_ttl_sec=30.0,
        )
        if not bars:
            return None

        bar_def = parse_bar_size(bar_size)
        if bar_def is not None and len(bars) >= 2:
            now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            last_ts = bars[-1].ts
            if last_ts + bar_def.duration > now_ref:
                bars = bars[:-1]
        if slow_p is not None and len(bars) < (slow_p + 1):
            return None

        regime_mode = str(regime_mode_raw or "ema").strip().lower()
        if regime_mode not in ("ema", "supertrend"):
            regime_mode = "ema"

        regime_preset = str(regime_ema_preset_raw or "").strip()
        regime_bar_size = str(regime_bar_size_raw or "").strip()
        if not regime_bar_size or regime_bar_size.lower() in ("same", "default"):
            regime_bar_size = str(bar_size)
        if regime_mode == "supertrend":
            use_mtf_regime = str(regime_bar_size) != str(bar_size)
        else:
            use_mtf_regime = bool(regime_preset) and (str(regime_bar_size) != str(bar_size))

        volume_period: int | None = None
        if isinstance(filters, dict) and filters.get("volume_ratio_min") is not None:
            raw_period = filters.get("volume_ema_period")
            try:
                volume_period = int(raw_period) if raw_period is not None else 20
            except (TypeError, ValueError):
                volume_period = 20
            volume_period = max(1, volume_period)

        exit_mode = str(spot_exit_mode_raw or "pct").strip().lower()
        if exit_mode not in ("pct", "atr"):
            exit_mode = "pct"
        exit_atr_engine = None
        last_exit_atr = None
        if exit_mode == "atr":
            try:
                atr_p = int(spot_atr_period_raw) if spot_atr_period_raw is not None else 14
            except (TypeError, ValueError):
                atr_p = 14
            atr_p = max(1, atr_p)
            exit_atr_engine = SupertrendEngine(atr_period=atr_p, multiplier=1.0, source="hl2")

        ema_engine = None
        orb_engine = None
        if entry_signal == "ema":
            try:
                ema_engine = EmaDecisionEngine(
                    ema_preset=str(ema_preset_raw),
                    ema_entry_mode=entry_mode_raw,
                    entry_confirm_bars=entry_confirm_bars,
                    regime_ema_preset=(
                        None if (use_mtf_regime or regime_mode == "supertrend") else regime_ema_preset_raw
                    ),
                )
            except ValueError:
                return None
        else:
            try:
                window = int(orb_window_mins_raw) if orb_window_mins_raw is not None else 15
            except (TypeError, ValueError):
                window = 15
            window = max(1, window)
            orb_open_time = parse_time_hhmm(instance.strategy.get("orb_open_time_et"), default=time(9, 30))
            if orb_open_time is None:
                orb_open_time = time(9, 30)
            orb_engine = OrbDecisionEngine(window_mins=window, open_time_et=orb_open_time)

        last_ts = None
        last_close = None
        last_volume = None
        closes: list[float] = []
        last_signal = None
        volume_ema = None
        volume_count = 0
        shock_engine = None
        last_shock = None
        shock_mode = None
        if isinstance(filters, dict):
            shock_mode = filters.get("shock_gate_mode")
            if shock_mode is None:
                shock_mode = filters.get("shock_mode")
        if isinstance(shock_mode, bool):
            shock_mode = "block" if shock_mode else "off"
        shock_mode = str(shock_mode or "off").strip().lower()
        if shock_mode in ("", "0", "false", "none", "null"):
            shock_mode = "off"
        if shock_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
            shock_mode = "off"
        if shock_mode != "off":
            shock_detector = "atr_ratio"
            if isinstance(filters, dict):
                shock_detector = str(filters.get("shock_detector") or "atr_ratio").strip().lower()
            if shock_detector in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
                shock_detector = "daily_atr_pct"
            if shock_detector in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
                shock_detector = "daily_drawdown"
            if shock_detector not in ("atr_ratio", "daily_atr_pct", "daily_drawdown"):
                shock_detector = "atr_ratio"

            try:
                atr_fast = int(filters.get("shock_atr_fast_period")) if isinstance(filters, dict) else 7
            except (TypeError, ValueError):
                atr_fast = 7
            try:
                atr_slow = int(filters.get("shock_atr_slow_period")) if isinstance(filters, dict) else 50
            except (TypeError, ValueError):
                atr_slow = 50
            try:
                on_ratio = float(filters.get("shock_on_ratio")) if isinstance(filters, dict) else 1.55
            except (TypeError, ValueError):
                on_ratio = 1.55
            try:
                off_ratio = float(filters.get("shock_off_ratio")) if isinstance(filters, dict) else 1.30
            except (TypeError, ValueError):
                off_ratio = 1.30
            try:
                min_atr_pct = float(filters.get("shock_min_atr_pct")) if isinstance(filters, dict) else 7.0
            except (TypeError, ValueError):
                min_atr_pct = 7.0
            try:
                dir_lb = int(filters.get("shock_direction_lookback")) if isinstance(filters, dict) else 2
            except (TypeError, ValueError):
                dir_lb = 2
            if shock_detector == "daily_atr_pct":
                try:
                    daily_period = int(filters.get("shock_daily_atr_period")) if isinstance(filters, dict) else 14
                except (TypeError, ValueError):
                    daily_period = 14
                try:
                    daily_on = float(filters.get("shock_daily_on_atr_pct")) if isinstance(filters, dict) else 13.0
                except (TypeError, ValueError):
                    daily_on = 13.0
                try:
                    daily_off = float(filters.get("shock_daily_off_atr_pct")) if isinstance(filters, dict) else 11.0
                except (TypeError, ValueError):
                    daily_off = 11.0
                try:
                    daily_tr_on = float(filters.get("shock_daily_on_tr_pct")) if isinstance(filters, dict) else None
                except (TypeError, ValueError):
                    daily_tr_on = None
                if daily_tr_on is not None and daily_tr_on <= 0:
                    daily_tr_on = None
                if daily_off > daily_on:
                    daily_off = daily_on
                shock_engine = DailyAtrPctShockEngine(
                    atr_period=max(1, int(daily_period)),
                    on_atr_pct=float(daily_on),
                    off_atr_pct=float(daily_off),
                    on_tr_pct=float(daily_tr_on) if daily_tr_on is not None else None,
                    direction_lookback=max(1, int(dir_lb)),
                )
            elif shock_detector == "daily_drawdown":
                try:
                    dd_lb = int(filters.get("shock_drawdown_lookback_days")) if isinstance(filters, dict) else 20
                except (TypeError, ValueError):
                    dd_lb = 20
                try:
                    dd_on = float(filters.get("shock_on_drawdown_pct")) if isinstance(filters, dict) else -20.0
                except (TypeError, ValueError):
                    dd_on = -20.0
                try:
                    dd_off = float(filters.get("shock_off_drawdown_pct")) if isinstance(filters, dict) else -10.0
                except (TypeError, ValueError):
                    dd_off = -10.0
                if dd_off < dd_on:
                    dd_off = dd_on
                shock_engine = DailyDrawdownShockEngine(
                    lookback_days=max(2, int(dd_lb)),
                    on_drawdown_pct=float(dd_on),
                    off_drawdown_pct=float(dd_off),
                    direction_lookback=max(1, int(dir_lb)),
                )
            else:
                shock_engine = AtrRatioShockEngine(
                    atr_fast_period=max(1, int(atr_fast)),
                    atr_slow_period=max(1, int(atr_slow)),
                    on_ratio=float(on_ratio),
                    off_ratio=float(off_ratio),
                    min_atr_pct=float(min_atr_pct),
                    direction_lookback=max(1, int(dir_lb)),
                    source=str(supertrend_source_raw or "hl2").strip().lower() or "hl2",
                )
        supertrend_engine = None
        last_supertrend = None
        if regime_mode == "supertrend" and not use_mtf_regime:
            try:
                atr_p = int(supertrend_atr_period_raw) if supertrend_atr_period_raw is not None else 10
            except (TypeError, ValueError):
                atr_p = 10
            try:
                mult = (
                    float(supertrend_multiplier_raw)
                    if supertrend_multiplier_raw is not None
                    else 3.0
                )
            except (TypeError, ValueError):
                mult = 3.0
            src = str(supertrend_source_raw or "hl2").strip().lower() or "hl2"
            supertrend_engine = SupertrendEngine(atr_period=atr_p, multiplier=mult, source=src)

        for bar in bars:
            close = float(bar.close)
            if close <= 0:
                continue
            last_ts = bar.ts
            last_close = close
            last_volume = float(bar.volume)
            closes.append(close)
            if ema_engine is not None:
                last_signal = ema_engine.update(close)
            elif orb_engine is not None:
                last_signal = orb_engine.update(
                    ts=bar.ts,
                    high=float(bar.high),
                    low=float(bar.low),
                    close=close,
                )
            if volume_period is not None:
                volume_ema = ema_next(volume_ema, float(bar.volume), volume_period)
                volume_count += 1
            if supertrend_engine is not None:
                last_supertrend = supertrend_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )
            if shock_engine is not None and not use_mtf_regime:
                if isinstance(shock_engine, (DailyAtrPctShockEngine, DailyDrawdownShockEngine)):
                    last_shock = shock_engine.update(
                        day=bar.ts.date(),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
                else:
                    last_shock = shock_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
            if exit_atr_engine is not None:
                last_exit_atr = exit_atr_engine.update(
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                )

        if last_ts is None or last_close is None or last_signal is None or not last_signal.ema_ready:
            return None

        bars_in_day = sum(1 for bar in bars if bar.ts.date() == last_ts.date())
        rv = realized_vol_from_closes(
            closes,
            lookback=60,
            lam=0.94,
            bar_size=bar_size,
            use_rth=use_rth,
        )

        if regime_mode == "supertrend":
            if use_mtf_regime:
                try:
                    atr_p = int(supertrend_atr_period_raw) if supertrend_atr_period_raw is not None else 10
                except (TypeError, ValueError):
                    atr_p = 10
                try:
                    mult = (
                        float(supertrend_multiplier_raw)
                        if supertrend_multiplier_raw is not None
                        else 3.0
                    )
                except (TypeError, ValueError):
                    mult = 3.0
                src = str(supertrend_source_raw or "hl2").strip().lower() or "hl2"

                regime_duration = self._signal_duration_str(regime_bar_size)
                if shock_engine is not None and "hour" in str(regime_bar_size).strip().lower():
                    # Ensure we have enough history to compute the shock slow ATR (often 50+ bars on 4h).
                    regime_duration = "1 M" if atr_slow <= 60 else "2 M"
                regime_bars = await self._client.historical_bars_ohlcv(
                    contract,
                    duration_str=regime_duration,
                    bar_size=regime_bar_size,
                    use_rth=use_rth,
                    cache_ttl_sec=30.0,
                )
                if not regime_bars:
                    return None
                regime_def = parse_bar_size(regime_bar_size)
                if regime_def is not None and len(regime_bars) >= 2:
                    now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
                    reg_last_ts = regime_bars[-1].ts
                    if reg_last_ts + regime_def.duration > now_ref:
                        regime_bars = regime_bars[:-1]

                regime_engine = SupertrendEngine(atr_period=atr_p, multiplier=mult, source=src)
                last_regime = None
                for bar in regime_bars:
                    if bar.ts > last_ts:
                        break
                    last_regime = regime_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
                    if shock_engine is not None:
                        if isinstance(shock_engine, (DailyAtrPctShockEngine, DailyDrawdownShockEngine)):
                            last_shock = shock_engine.update(
                                day=bar.ts.date(),
                                high=float(bar.high),
                                low=float(bar.low),
                                close=float(bar.close),
                            )
                        else:
                            last_shock = shock_engine.update(
                                high=float(bar.high),
                                low=float(bar.low),
                                close=float(bar.close),
                            )
                regime_ready = bool(last_regime and last_regime.ready)
                regime_dir = last_regime.direction if last_regime is not None else None
            else:
                regime_ready = bool(last_supertrend and last_supertrend.ready)
                regime_dir = last_supertrend.direction if last_supertrend is not None else None

            last_signal = apply_regime_gate(
                last_signal,
                regime_dir=regime_dir,
                regime_ready=regime_ready,
            )
        elif use_mtf_regime:
            regime_periods = ema_periods(regime_preset)
            if regime_periods is None:
                return None
            _, regime_slow_p = regime_periods

            regime_bars = await self._client.historical_bars_ohlcv(
                contract,
                duration_str=(
                    ("1 M" if atr_slow <= 60 else "2 M")
                    if (shock_engine is not None and "hour" in str(regime_bar_size).strip().lower())
                    else self._signal_duration_str(regime_bar_size)
                ),
                bar_size=regime_bar_size,
                use_rth=use_rth,
                cache_ttl_sec=30.0,
            )
            if not regime_bars:
                return None

            regime_def = parse_bar_size(regime_bar_size)
            if regime_def is not None and len(regime_bars) >= 2:
                now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
                reg_last_ts = regime_bars[-1].ts
                if reg_last_ts + regime_def.duration > now_ref:
                    regime_bars = regime_bars[:-1]
            if len(regime_bars) < (regime_slow_p + 1):
                return None

            try:
                regime_engine = EmaDecisionEngine(
                    ema_preset=regime_preset,
                    ema_entry_mode="trend",
                    entry_confirm_bars=0,
                    regime_ema_preset=None,
                )
            except ValueError:
                return None

            last_regime = None
            for bar in regime_bars:
                if bar.ts > last_ts:
                    break
                if float(bar.close) <= 0:
                    continue
                last_regime = regime_engine.update(float(bar.close))
                if shock_engine is not None:
                    if isinstance(shock_engine, (DailyAtrPctShockEngine, DailyDrawdownShockEngine)):
                        last_shock = shock_engine.update(
                            day=bar.ts.date(),
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                        )
                    else:
                        last_shock = shock_engine.update(
                            high=float(bar.high),
                            low=float(bar.low),
                            close=float(bar.close),
                        )

            regime_ready = bool(last_regime and last_regime.ema_ready)
            regime_dir = last_regime.state if last_regime is not None else None
            last_signal = apply_regime_gate(
                last_signal,
                regime_dir=regime_dir,
                regime_ready=regime_ready,
            )
            if last_signal is None:
                return None

        regime2_mode = str(regime2_mode_raw or "off").strip().lower()
        if regime2_mode not in ("off", "ema", "supertrend"):
            regime2_mode = "off"
        if regime2_mode != "off":
            regime2_bar_size = str(regime2_bar_size_raw or "").strip()
            if not regime2_bar_size or regime2_bar_size.lower() in ("same", "default"):
                regime2_bar_size = str(bar_size)
            use_mtf_regime2 = str(regime2_bar_size) != str(bar_size)

            regime2_ready = False
            regime2_dir = None
            if regime2_mode == "supertrend":
                try:
                    atr_p = (
                        int(regime2_supertrend_atr_period_raw)
                        if regime2_supertrend_atr_period_raw is not None
                        else 10
                    )
                except (TypeError, ValueError):
                    atr_p = 10
                try:
                    mult = (
                        float(regime2_supertrend_multiplier_raw)
                        if regime2_supertrend_multiplier_raw is not None
                        else 3.0
                    )
                except (TypeError, ValueError):
                    mult = 3.0
                src = str(regime2_supertrend_source_raw or "hl2").strip().lower() or "hl2"

                if use_mtf_regime2:
                    regime2_bars = await self._client.historical_bars_ohlcv(
                        contract,
                        duration_str=self._signal_duration_str(regime2_bar_size),
                        bar_size=regime2_bar_size,
                        use_rth=use_rth,
                        cache_ttl_sec=30.0,
                    )
                    if not regime2_bars:
                        return None
                    regime2_def = parse_bar_size(regime2_bar_size)
                    if regime2_def is not None and len(regime2_bars) >= 2:
                        now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
                        reg_last_ts = regime2_bars[-1].ts
                        if reg_last_ts + regime2_def.duration > now_ref:
                            regime2_bars = regime2_bars[:-1]
                else:
                    regime2_bars = bars

                regime2_engine = SupertrendEngine(atr_period=atr_p, multiplier=mult, source=src)
                last_regime2 = None
                for bar in regime2_bars:
                    if bar.ts > last_ts:
                        break
                    last_regime2 = regime2_engine.update(
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                    )
                regime2_ready = bool(last_regime2 and last_regime2.ready)
                regime2_dir = last_regime2.direction if last_regime2 is not None else None
            else:
                regime2_preset = str(regime2_ema_preset_raw or "").strip()
                if not regime2_preset:
                    return None
                regime2_periods = ema_periods(regime2_preset)
                if regime2_periods is None:
                    return None
                _, regime2_slow_p = regime2_periods

                if use_mtf_regime2:
                    regime2_bars = await self._client.historical_bars_ohlcv(
                        contract,
                        duration_str=self._signal_duration_str(regime2_bar_size),
                        bar_size=regime2_bar_size,
                        use_rth=use_rth,
                        cache_ttl_sec=30.0,
                    )
                    if not regime2_bars:
                        return None

                    regime2_def = parse_bar_size(regime2_bar_size)
                    if regime2_def is not None and len(regime2_bars) >= 2:
                        now_ref = datetime.now(tz=timezone.utc).replace(tzinfo=None)
                        reg_last_ts = regime2_bars[-1].ts
                        if reg_last_ts + regime2_def.duration > now_ref:
                            regime2_bars = regime2_bars[:-1]
                else:
                    regime2_bars = bars

                if len(regime2_bars) < (regime2_slow_p + 1):
                    return None
                try:
                    regime2_engine = EmaDecisionEngine(
                        ema_preset=regime2_preset,
                        ema_entry_mode="trend",
                        entry_confirm_bars=0,
                        regime_ema_preset=None,
                    )
                except ValueError:
                    return None
                last_regime2 = None
                for bar in regime2_bars:
                    if bar.ts > last_ts:
                        break
                    if float(bar.close) <= 0:
                        continue
                    last_regime2 = regime2_engine.update(float(bar.close))

                regime2_ready = bool(last_regime2 and last_regime2.ema_ready)
                regime2_dir = last_regime2.state if last_regime2 is not None else None

            last_signal = apply_regime_gate(
                last_signal,
                regime_dir=regime2_dir,
                regime_ready=regime2_ready,
            )
            if last_signal is None:
                return None

        shock = bool(last_shock.shock) if (last_shock is not None and bool(last_shock.ready)) else None
        shock_dir = (
            str(last_shock.direction)
            if (
                last_shock is not None
                and bool(last_shock.ready)
                and bool(getattr(last_shock, "direction_ready", False))
                and getattr(last_shock, "direction", None) in ("up", "down")
            )
            else None
        )
        return _SignalSnapshot(
            bar_ts=last_ts,
            close=float(last_close),
            signal=last_signal,
            bars_in_day=int(bars_in_day),
            rv=float(rv) if rv is not None else None,
            volume=float(last_volume) if last_volume is not None else None,
            volume_ema=float(volume_ema) if volume_ema is not None else None,
            volume_ema_ready=bool(volume_count >= volume_period) if volume_period else True,
            shock=shock,
            shock_dir=shock_dir,
            atr=(
                float(last_exit_atr.atr)
                if last_exit_atr is not None and bool(last_exit_atr.ready) and last_exit_atr.atr is not None
                else None
            ),
            or_high=orb_engine.or_high if orb_engine is not None else None,
            or_low=orb_engine.or_low if orb_engine is not None else None,
            or_ready=bool(orb_engine and orb_engine.or_ready),
        )

    def _reset_daily_counters_if_needed(self, instance: _BotInstance) -> None:
        today = datetime.now(tz=ZoneInfo("America/New_York")).date()
        if instance.entries_today_date != today:
            instance.entries_today_date = today
            instance.entries_today = 0

    def _entry_limit_ok(self, instance: _BotInstance) -> bool:
        self._reset_daily_counters_if_needed(instance)
        raw = instance.strategy.get("max_entries_per_day", 1)
        try:
            max_entries = int(raw)
        except (TypeError, ValueError):
            max_entries = 1
        if max_entries <= 0:
            return True
        return instance.entries_today < max_entries

    def _spot_open_position(self, *, symbol: str, sec_type: str, con_id: int = 0) -> PortfolioItem | None:
        sym = str(symbol or "").strip().upper()
        stype = str(sec_type or "STK").strip().upper() or "STK"
        desired_con_id = int(con_id or 0)
        best = None
        best_abs = 0.0
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType != stype:
                continue
            if str(getattr(contract, "symbol", "") or "").strip().upper() != sym:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if not pos:
                continue
            if desired_con_id:
                try:
                    if int(getattr(contract, "conId", 0) or 0) == desired_con_id:
                        return item
                except (TypeError, ValueError):
                    pass
            abs_pos = abs(pos)
            if abs_pos > best_abs:
                best_abs = abs_pos
                best = item
        return best

    def _options_open_positions(self, instance: _BotInstance) -> list[PortfolioItem]:
        if not instance.touched_conids:
            return []
        open_items: list[PortfolioItem] = []
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType not in ("OPT", "FOP"):
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id not in instance.touched_conids:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if pos:
                open_items.append(item)
        return open_items

    def _options_position_values(self, items: list[PortfolioItem]) -> tuple[float | None, float | None]:
        if not items:
            return None, None
        cost_basis = 0.0
        market_value = 0.0
        for item in items:
            cost_basis += float(_cost_basis(item))
            mv = _safe_num(getattr(item, "marketValue", None))
            if mv is None:
                try:
                    mv = float(getattr(item, "position", 0.0) or 0.0) * float(getattr(item, "marketPrice", 0.0) or 0.0)
                    mv *= _infer_multiplier(item)
                except (TypeError, ValueError):
                    mv = 0.0
            market_value += float(mv)

        # Convert into the backtest sign convention: SELL credit = positive, BUY debit = negative.
        entry_value = -float(cost_basis)
        current_value = -float(market_value)
        return entry_value, current_value

    def _options_max_loss_estimate(self, items: list[PortfolioItem], *, spot: float) -> float | None:
        if not items:
            return None
        entry_value, _ = self._options_position_values(items)
        if entry_value is None:
            return None
        strikes: list[float] = []
        legs: list[tuple[str, str, float, int, float]] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            raw_right = str(getattr(contract, "right", "") or "").upper()
            right = "CALL" if raw_right in ("C", "CALL") else "PUT" if raw_right in ("P", "PUT") else ""
            if not right:
                continue
            try:
                strike = float(getattr(contract, "strike", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            qty = int(abs(pos))
            if qty <= 0:
                continue
            action = "BUY" if pos > 0 else "SELL"
            mult = _infer_multiplier(item)
            strikes.append(strike)
            legs.append((action, right, strike, qty, mult))
        if not strikes or not legs:
            return None
        strikes = sorted(set(strikes))
        high = max(float(spot), strikes[-1]) * 5.0
        candidates = [0.0] + strikes + [high]

        def _payoff(price: float) -> float:
            payoff = 0.0
            for action, right, strike, qty, mult in legs:
                if right == "CALL":
                    intrinsic = max(price - strike, 0.0)
                else:
                    intrinsic = max(strike - price, 0.0)
                sign = 1.0 if action == "BUY" else -1.0
                payoff += sign * intrinsic * float(qty) * float(mult)
            return payoff

        min_pnl = None
        for price in candidates:
            pnl = float(entry_value) + _payoff(float(price))
            if min_pnl is None or pnl < min_pnl:
                min_pnl = pnl
        if min_pnl is None:
            return None
        return max(0.0, -float(min_pnl))

    def _should_exit_on_dte(self, instance: _BotInstance, items: list[PortfolioItem], today: date) -> bool:
        raw_exit = instance.strategy.get("exit_dte", 0)
        try:
            exit_dte = int(raw_exit or 0)
        except (TypeError, ValueError):
            exit_dte = 0
        if exit_dte <= 0:
            return False
        raw_entry = instance.strategy.get("dte", 0)
        try:
            entry_dte = int(raw_entry or 0)
        except (TypeError, ValueError):
            entry_dte = 0
        if entry_dte > 0 and exit_dte >= entry_dte:
            return False

        expiries: list[date] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            exp = _contract_expiry_date(getattr(contract, "lastTradeDateOrContractMonth", None))
            if exp is not None:
                expiries.append(exp)
        if not expiries:
            return False
        remaining = min(_business_days_until(today, exp) for exp in expiries)
        return remaining <= exit_dte

    def _open_direction_from_positions(self, items: list[PortfolioItem]) -> str | None:
        if not items:
            return None
        biggest_any = None
        biggest_any_abs = 0.0
        biggest_short = None
        biggest_short_abs = 0.0
        for item in items:
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            abs_pos = abs(pos)
            if abs_pos > biggest_any_abs:
                biggest_any_abs = abs_pos
                biggest_any = item
            if pos < 0 and abs_pos > biggest_short_abs:
                biggest_short_abs = abs_pos
                biggest_short = item

        chosen = biggest_short or biggest_any
        if chosen is None:
            return None
        contract = getattr(chosen, "contract", None)
        if not contract:
            return None
        right_char = str(getattr(contract, "right", "") or "").upper()
        right = "CALL" if right_char in ("C", "CALL") else "PUT" if right_char in ("P", "PUT") else ""
        try:
            pos = float(getattr(chosen, "position", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        action = "BUY" if pos > 0 else "SELL" if pos < 0 else ""
        return direction_from_action_right(action, right)

    def _should_exit_on_flip(
        self,
        instance: _BotInstance,
        snap: _SignalSnapshot,
        open_dir: str | None,
        open_items: list[PortfolioItem],
    ) -> bool:
        if not flip_exit_hit(
            exit_on_signal_flip=bool(instance.strategy.get("exit_on_signal_flip")),
            open_dir=open_dir,
            signal=snap.signal,
            flip_exit_mode_raw=instance.strategy.get("flip_exit_mode"),
            ema_entry_mode_raw=instance.strategy.get("ema_entry_mode"),
        ):
            return False

        hold_bars_raw = instance.strategy.get("flip_exit_min_hold_bars", 0)
        try:
            hold_bars = int(hold_bars_raw)
        except (TypeError, ValueError):
            hold_bars = 0
        if hold_bars > 0 and instance.last_entry_bar_ts is not None:
            bar_def = parse_bar_size(self._signal_bar_size(instance))
            if bar_def is not None:
                if (snap.bar_ts - instance.last_entry_bar_ts) < (bar_def.duration * hold_bars):
                    return False

        if bool(instance.strategy.get("flip_exit_only_if_profit")):
            pnl = 0.0
            for item in open_items:
                try:
                    pnl += float(getattr(item, "unrealizedPNL", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
            if pnl <= 0:
                return False
        return True

    def _entry_direction_for_instance(self, instance: _BotInstance, snap: _SignalSnapshot) -> str | None:
        entry_dir = snap.signal.entry_dir
        return str(entry_dir) if entry_dir in ("up", "down") else None

    def _allowed_entry_directions(self, instance: _BotInstance) -> set[str]:
        strategy = instance.strategy or {}
        instrument = self._strategy_instrument(strategy)
        if instrument == "spot":
            mapping = strategy.get("directional_spot") if isinstance(strategy.get("directional_spot"), dict) else None
            if mapping:
                allowed = set()
                for key in ("up", "down"):
                    leg = mapping.get(key)
                    if not isinstance(leg, dict):
                        continue
                    action = str(leg.get("action", "")).strip().upper()
                    if action in ("BUY", "SELL"):
                        allowed.add(key)
                return allowed
            return {"up"}

        if isinstance(strategy.get("directional_legs"), dict):
            allowed = {k for k in ("up", "down") if strategy["directional_legs"].get(k)}
            return allowed or {"up", "down"}

        legs = strategy.get("legs", [])
        if isinstance(legs, list) and legs:
            first = legs[0] if isinstance(legs[0], dict) else None
            if isinstance(first, dict):
                bias = direction_from_action_right(first.get("action", ""), first.get("right", ""))
                if bias in ("up", "down"):
                    return {bias}
        return {"up", "down"}

    async def _strike_by_delta(
        self,
        *,
        symbol: str,
        expiry: str,
        right_char: str,
        strikes: list[float],
        trading_class: str | None,
        near_strike: float,
        target_delta: float,
    ) -> float | None:
        try:
            target = abs(float(target_delta))
        except (TypeError, ValueError):
            return None
        if target <= 0 or target > 1:
            return None
        try:
            strike_values = sorted(float(s) for s in strikes)
        except (TypeError, ValueError):
            return None
        if not strike_values:
            return None
        center_idx = min(
            range(len(strike_values)), key=lambda idx: abs(strike_values[idx] - near_strike)
        )
        window = strike_values[max(center_idx - 10, 0) : center_idx + 11]
        if not window:
            return None

        candidates = [
            Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=float(strike),
                right=right_char,
                exchange="SMART",
                currency="USD",
                tradingClass=trading_class,
            )
            for strike in window
        ]
        qualified = await self._client.qualify_proxy_contracts(*candidates)
        if qualified and len(qualified) == len(candidates):
            contracts: list[Contract] = list(qualified)
        else:
            contracts = list(candidates)

        best_strike: float | None = None
        best_diff: float | None = None
        for contract, strike in zip(contracts, window):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            delta = None
            for _ in range(6):
                for attr in ("modelGreeks", "bidGreeks", "askGreeks", "lastGreeks"):
                    greeks = getattr(ticker, attr, None)
                    if greeks is not None:
                        raw = getattr(greeks, "delta", None)
                        if raw is not None:
                            try:
                                delta = float(raw)
                            except (TypeError, ValueError):
                                delta = None
                            break
                if delta is not None:
                    break
                await asyncio.sleep(0.05)
            if delta is None:
                continue
            diff = abs(abs(delta) - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_strike = float(strike)

        # Avoid keeping a large number of quote subscriptions just to pick strike.
        for contract, strike in zip(contracts, window):
            if best_strike is not None and float(strike) == best_strike:
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._client.release_ticker(con_id)
                self._tracked_conids.discard(con_id)
        return best_strike

    async def _propose_for_instance(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        strat = instance.strategy or {}
        instrument = self._strategy_instrument(strat)
        intent_clean = str(intent or "enter").strip().lower()
        intent_clean = "exit" if intent_clean == "exit" else "enter"
        symbol = str(
            instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()

        raw_mode = str(strat.get("price_mode") or "OPTIMISTIC").strip().upper()
        if raw_mode not in ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"):
            raw_mode = "OPTIMISTIC"
        price_mode = raw_mode

        def _leg_price(bid: float | None, ask: float | None, last: float | None, action: str) -> float | None:
            mid = _midpoint(bid, ask)
            if price_mode == "CROSS":
                return ask if action == "BUY" else bid
            if price_mode == "MID":
                return mid or last
            if price_mode == "OPTIMISTIC":
                return _optimistic_price(bid, ask, mid, action) or mid or last
            if price_mode == "AGGRESSIVE":
                return _aggressive_price(bid, ask, mid, action) or mid or last
            return mid or last

        def _bump_entry_counters() -> None:
            self._reset_daily_counters_if_needed(instance)
            instance.entries_today += 1

        def _finalize_leg_orders(
            *,
            underlying: Contract,
            leg_orders: list[_BotLegOrder],
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]],
        ) -> None:
            if not leg_orders:
                self._status = "Propose: no legs configured"
                self._render_status()
                return

            # Compute net price in "debit units": BUY adds, SELL subtracts.
            debit_mid = 0.0
            debit_bid = 0.0
            debit_ask = 0.0
            desired_debit = 0.0
            tick = None
            for leg_order, (bid, ask, last, ticker) in zip(leg_orders, leg_quotes):
                mid = _midpoint(bid, ask)
                leg_mid = mid or last
                if leg_mid is None:
                    self._status = "Quote: missing mid/last (cannot price)"
                    self._render_status()
                    return
                leg_bid = bid or mid or last
                leg_ask = ask or mid or last
                if leg_bid is None or leg_ask is None:
                    self._status = "Quote: missing bid/ask (cannot price)"
                    self._render_status()
                    return
                leg_desired = _leg_price(bid, ask, last, leg_order.action)
                if leg_desired is None:
                    self._status = "Quote: missing bid/ask/last (cannot price)"
                    self._render_status()
                    return
                leg_tick = _tick_size(leg_order.contract, ticker, leg_desired)
                tick = leg_tick if tick is None else min(tick, leg_tick)
                sign = 1.0 if leg_order.action == "BUY" else -1.0
                debit_mid += sign * float(leg_mid) * leg_order.ratio
                debit_bid += sign * float(leg_bid) * leg_order.ratio
                debit_ask += sign * float(leg_ask) * leg_order.ratio
                desired_debit += sign * float(leg_desired) * leg_order.ratio

            tick = tick or 0.01

            if len(leg_orders) == 1:
                single = leg_orders[0]
                (bid, ask, last, ticker) = leg_quotes[0]
                limit = _leg_price(bid, ask, last, single.action)
                if limit is None:
                    self._status = "Quote: no bid/ask/last (cannot price)"
                    self._render_status()
                    return
                limit = _round_to_tick(float(limit), tick)
                proposal = _BotProposal(
                    instance_id=instance.instance_id,
                    preset=None,
                    underlying=underlying,
                    order_contract=single.contract,
                    legs=leg_orders,
                    action=single.action,
                    quantity=single.ratio,
                    limit_price=float(limit),
                    created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                    bid=bid,
                    ask=ask,
                    last=last,
                )
                con_id = int(getattr(single.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
                self._add_proposal(proposal)
                self._status = f"Proposed {single.action} {single.ratio} {symbol} @ {limit:.2f}"
                if intent_clean == "enter":
                    if direction in ("up", "down"):
                        instance.open_direction = str(direction)
                    if signal_bar_ts is not None:
                        instance.last_entry_bar_ts = signal_bar_ts
                    _bump_entry_counters()
                elif signal_bar_ts is not None:
                    instance.last_exit_bar_ts = signal_bar_ts
                self._render_status()
                return

            # Multi-leg combo: use IBKR's native encoding (can be negative for credits).
            order_action = "BUY"
            order_bid = debit_bid
            order_ask = debit_ask
            order_last = debit_mid
            order_limit = _round_to_tick(float(desired_debit), tick)
            if not order_limit:
                self._status = "Quote: combo price is 0 (cannot price)"
                self._render_status()
                return
            combo_legs: list[ComboLeg] = []
            for leg_order, (_, _, _, ticker) in zip(leg_orders, leg_quotes):
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if not con_id:
                    self._status = "Contract: missing conId for combo leg"
                    self._render_status()
                    return
                leg_exchange = (
                    getattr(getattr(ticker, "contract", None), "exchange", "") or ""
                ).strip()
                if not leg_exchange:
                    leg_exchange = (getattr(leg_order.contract, "exchange", "") or "").strip()
                if not leg_exchange:
                    leg_sec_type = str(getattr(leg_order.contract, "secType", "") or "").strip()
                    leg_exchange = "CME" if leg_sec_type == "FOP" else "SMART"
                combo_legs.append(
                    ComboLeg(
                        conId=con_id,
                        ratio=leg_order.ratio,
                        action=leg_order.action,
                        exchange=leg_exchange,
                    )
                )
            bag = Bag(symbol=symbol, exchange="SMART", currency="USD", comboLegs=combo_legs)

            proposal = _BotProposal(
                instance_id=instance.instance_id,
                preset=None,
                underlying=underlying,
                order_contract=bag,
                legs=leg_orders,
                action=order_action,
                quantity=1,
                limit_price=float(order_limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=order_bid,
                ask=order_ask,
                last=order_last,
            )
            for leg_order in leg_orders:
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
            self._add_proposal(proposal)
            self._status = (
                f"Proposed {order_action} BAG {symbol} @ {order_limit:.2f} ({len(leg_orders)} legs)"
            )
            if intent_clean == "enter":
                if direction in ("up", "down"):
                    instance.open_direction = str(direction)
                if signal_bar_ts is not None:
                    instance.last_entry_bar_ts = signal_bar_ts
                _bump_entry_counters()
            elif signal_bar_ts is not None:
                instance.last_exit_bar_ts = signal_bar_ts
            self._render_status()

        if intent_clean == "exit":
            if instrument == "spot":
                sec_type = self._spot_sec_type(instance, symbol)
                open_item = self._spot_open_position(symbol=symbol, sec_type=sec_type, con_id=0)
                if open_item is None:
                    self._status = f"Exit: no spot position for {symbol}"
                    self._render_status()
                    return
                try:
                    pos = float(getattr(open_item, "position", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pos = 0.0
                if not pos:
                    self._status = f"Exit: no spot position for {symbol}"
                    self._render_status()
                    return

                action = "SELL" if pos > 0 else "BUY"
                qty = int(abs(pos))
                if qty <= 0:
                    self._status = f"Exit: invalid position size for {symbol}"
                    self._render_status()
                    return

                contract = open_item.contract
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    self._tracked_conids.add(con_id)
                ticker = await self._client.ensure_ticker(contract)
                bid = _safe_num(getattr(ticker, "bid", None))
                ask = _safe_num(getattr(ticker, "ask", None))
                last = _safe_num(getattr(ticker, "last", None))
                limit = _leg_price(bid, ask, last, action)
                if limit is None:
                    self._status = "Quote: no bid/ask/last (cannot price)"
                    self._render_status()
                    return
                tick = _tick_size(contract, ticker, limit) or 0.01
                limit = _round_to_tick(float(limit), tick)
                proposal = _BotProposal(
                    instance_id=instance.instance_id,
                    preset=None,
                    underlying=contract,
                    order_contract=contract,
                    legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                    action=action,
                    quantity=qty,
                    limit_price=float(limit),
                    created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                    bid=bid,
                    ask=ask,
                    last=last,
                )
                if con_id:
                    instance.touched_conids.add(con_id)
                self._add_proposal(proposal)
                if signal_bar_ts is not None:
                    instance.last_exit_bar_ts = signal_bar_ts
                self._status = f"Proposed EXIT {action} {qty} {symbol} @ {limit:.2f}"
                self._render_status()
                return

            open_items = self._options_open_positions(instance)
            if not open_items:
                self._status = f"Exit: no option positions for instance {instance.instance_id}"
                self._render_status()
                return
            underlying = Stock(symbol=symbol, exchange="SMART", currency="USD")
            qualified = await self._client.qualify_proxy_contracts(underlying)
            if qualified:
                underlying = qualified[0]

            leg_orders: list[_BotLegOrder] = []
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
            for item in open_items:
                contract = item.contract
                try:
                    pos = float(getattr(item, "position", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if not pos:
                    continue
                ratio = int(abs(pos))
                if ratio <= 0:
                    continue
                action = "SELL" if pos > 0 else "BUY"
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    self._tracked_conids.add(con_id)
                ticker = await self._client.ensure_ticker(contract)
                bid = _safe_num(getattr(ticker, "bid", None))
                ask = _safe_num(getattr(ticker, "ask", None))
                last = _safe_num(getattr(ticker, "last", None))
                leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
                leg_quotes.append((bid, ask, last, ticker))

            if not leg_orders:
                self._status = f"Exit: no option positions for instance {instance.instance_id}"
                self._render_status()
                return
            _finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)
            return

        if instrument == "spot":
            entry_signal = str(strat.get("entry_signal") or "ema").strip().lower()
            if entry_signal not in ("ema", "orb"):
                entry_signal = "ema"
            exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
            if exit_mode not in ("pct", "atr"):
                exit_mode = "pct"

            snap = None
            if direction not in ("up", "down") or entry_signal == "orb" or exit_mode == "atr":
                signal_contract = await self._signal_contract(instance, symbol)
                snap = (
                    await self._signal_snapshot_for_contract(
                        contract=signal_contract,
                        ema_preset_raw=str(strat.get("ema_preset")) if strat.get("ema_preset") else None,
                        bar_size=self._signal_bar_size(instance),
                        use_rth=self._signal_use_rth(instance),
                        entry_signal_raw=entry_signal,
                        orb_window_mins_raw=strat.get("orb_window_mins"),
                        entry_mode_raw=strat.get("ema_entry_mode"),
                        entry_confirm_bars=strat.get("entry_confirm_bars", 0),
                        spot_exit_mode_raw=strat.get("spot_exit_mode"),
                        spot_atr_period_raw=strat.get("spot_atr_period"),
                        regime_ema_preset_raw=strat.get("regime_ema_preset"),
                        regime_bar_size_raw=strat.get("regime_bar_size"),
                        regime_mode_raw=strat.get("regime_mode"),
                        supertrend_atr_period_raw=strat.get("supertrend_atr_period"),
                        supertrend_multiplier_raw=strat.get("supertrend_multiplier"),
                        supertrend_source_raw=strat.get("supertrend_source"),
                        regime2_ema_preset_raw=strat.get("regime2_ema_preset"),
                        regime2_bar_size_raw=strat.get("regime2_bar_size"),
                        regime2_mode_raw=strat.get("regime2_mode"),
                        regime2_supertrend_atr_period_raw=strat.get("regime2_supertrend_atr_period"),
                        regime2_supertrend_multiplier_raw=strat.get("regime2_supertrend_multiplier"),
                        regime2_supertrend_source_raw=strat.get("regime2_supertrend_source"),
                        filters=instance.filters if isinstance(instance.filters, dict) else None,
                    )
                    if signal_contract is not None
                    else None
                )

            if direction not in ("up", "down") and snap is not None:
                direction = self._entry_direction_for_instance(instance, snap) or (
                    str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                )
            direction = direction if direction in ("up", "down") else "up"

            mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
            chosen = mapping.get(direction) if mapping else None
            if not isinstance(chosen, dict):
                if direction == "up":
                    chosen = {"action": "BUY", "qty": 1}
                else:
                    chosen = {"action": "SELL", "qty": 1}
            action = str(chosen.get("action", "")).strip().upper()
            if action not in ("BUY", "SELL"):
                self._status = f"Propose: invalid spot action for {direction}"
                self._render_status()
                return
            try:
                qty = int(chosen.get("qty", 1) or 1)
            except (TypeError, ValueError):
                qty = 1
            qty = max(1, abs(qty))

            contract = await self._spot_contract(instance, symbol)
            if contract is None:
                self._status = f"Contract: not found for {symbol}"
                self._render_status()
                return

            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _leg_price(bid, ask, last, action)
            if limit is None:
                self._status = "Quote: no bid/ask/last (cannot price)"
                self._render_status()
                return
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)

            instance.spot_profit_target_price = None
            instance.spot_stop_loss_price = None
            if snap is not None and entry_signal == "orb":
                try:
                    rr = float(strat.get("orb_risk_reward", 2.0) or 2.0)
                except (TypeError, ValueError):
                    rr = 2.0
                target_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
                if target_mode not in ("rr", "or_range"):
                    target_mode = "rr"
                or_high = snap.or_high
                or_low = snap.or_low
                if (
                    rr > 0
                    and bool(snap.or_ready)
                    and or_high is not None
                    and or_low is not None
                    and float(or_high) > 0
                    and float(or_low) > 0
                ):
                    stop = float(or_low) if direction == "up" else float(or_high)
                    if target_mode == "or_range":
                        rng = float(or_high) - float(or_low)
                        if rng > 0:
                            target = (
                                float(or_high) + (rr * rng)
                                if direction == "up"
                                else float(or_low) - (rr * rng)
                            )
                            if (
                                (direction == "up" and float(target) <= float(limit))
                                or (direction == "down" and float(target) >= float(limit))
                            ):
                                self._status = "Propose: ORB target already hit (skip)"
                                self._render_status()
                                return
                            instance.spot_profit_target_price = float(target)
                            instance.spot_stop_loss_price = float(stop)
                    else:
                        risk = abs(float(limit) - stop)
                        if risk > 0:
                            target = float(limit) + (rr * risk) if direction == "up" else float(limit) - (rr * risk)
                            instance.spot_profit_target_price = float(target)
                            instance.spot_stop_loss_price = float(stop)
            elif exit_mode == "atr":
                atr = float(snap.atr) if snap is not None and snap.atr is not None else 0.0
                if atr <= 0:
                    self._status = "Propose: ATR not ready (spot_exit_mode=atr)"
                    self._render_status()
                    return
                try:
                    pt_mult = float(strat.get("spot_pt_atr_mult", 1.5) or 1.5)
                except (TypeError, ValueError):
                    pt_mult = 1.5
                try:
                    sl_mult = float(strat.get("spot_sl_atr_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    sl_mult = 1.0
                if direction == "up":
                    instance.spot_profit_target_price = float(limit) + (pt_mult * atr)
                    instance.spot_stop_loss_price = float(limit) - (sl_mult * atr)
                else:
                    instance.spot_profit_target_price = float(limit) - (pt_mult * atr)
                    instance.spot_stop_loss_price = float(limit) + (sl_mult * atr)

            proposal = _BotProposal(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=bid,
                ask=ask,
                last=last,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            instance.open_direction = str(direction)
            if signal_bar_ts is not None:
                instance.last_entry_bar_ts = signal_bar_ts
            _bump_entry_counters()
            self._add_proposal(proposal)
            self._status = f"Proposed {action} {qty} {symbol} @ {limit:.2f} ({direction})"
            self._render_status()
            return

        legs_raw: list[dict] | None = None
        if isinstance(strat.get("directional_legs"), dict):
            dmap = strat.get("directional_legs") or {}
            if direction not in ("up", "down"):
                ema_preset = strat.get("ema_preset")
                if ema_preset:
                    signal_contract = await self._signal_contract(instance, symbol)
                    snap = (
                        await self._signal_snapshot_for_contract(
                            contract=signal_contract,
                            ema_preset_raw=str(ema_preset),
                            bar_size=self._signal_bar_size(instance),
                            use_rth=self._signal_use_rth(instance),
                            entry_mode_raw=strat.get("ema_entry_mode"),
                            entry_confirm_bars=strat.get("entry_confirm_bars", 0),
                            regime_ema_preset_raw=strat.get("regime_ema_preset"),
                            regime_bar_size_raw=strat.get("regime_bar_size"),
                            regime_mode_raw=strat.get("regime_mode"),
                            supertrend_atr_period_raw=strat.get("supertrend_atr_period"),
                            supertrend_multiplier_raw=strat.get("supertrend_multiplier"),
                            supertrend_source_raw=strat.get("supertrend_source"),
                            regime2_ema_preset_raw=strat.get("regime2_ema_preset"),
                            regime2_bar_size_raw=strat.get("regime2_bar_size"),
                            regime2_mode_raw=strat.get("regime2_mode"),
                            regime2_supertrend_atr_period_raw=strat.get("regime2_supertrend_atr_period"),
                            regime2_supertrend_multiplier_raw=strat.get("regime2_supertrend_multiplier"),
                            regime2_supertrend_source_raw=strat.get("regime2_supertrend_source"),
                            filters=instance.filters if isinstance(instance.filters, dict) else None,
                        )
                            if signal_contract is not None
                            else None
                    )
                    if snap is not None:
                        direction = self._entry_direction_for_instance(instance, snap) or (
                            str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                        )
            if direction in ("up", "down") and dmap.get(direction):
                legs_raw = dmap.get(direction)
            else:
                for key in ("up", "down"):
                    if dmap.get(key):
                        legs_raw = dmap.get(key)
                        direction = key
                        break

        if legs_raw is None:
            raw = strat.get("legs", []) or []
            legs_raw = raw if isinstance(raw, list) else []

        if not isinstance(legs_raw, list) or not legs_raw:
            self._status = "Propose: no legs configured"
            self._render_status()
            return

        dte_raw = strat.get("dte", 0)
        try:
            dte = int(dte_raw or 0)
        except (TypeError, ValueError):
            dte = 0

        chain_info = await self._client.stock_option_chain(symbol)
        if not chain_info:
            self._status = f"Chain: not found for {symbol}"
            self._render_status()
            return
        underlying, chain = chain_info
        underlying_ticker = await self._client.ensure_ticker(underlying)
        under_con_id = int(getattr(underlying, "conId", 0) or 0)
        if under_con_id:
            self._tracked_conids.add(under_con_id)
        spot = _ticker_price(underlying_ticker)
        if spot is None:
            self._status = f"Spot: n/a for {symbol}"
            self._render_status()
            return

        expiry = _pick_chain_expiry(date.today(), dte, getattr(chain, "expirations", []))
        if not expiry:
            self._status = f"Expiry: none for {symbol}"
            self._render_status()
            return

        # Build and qualify option legs.
        strikes = getattr(chain, "strikes", [])
        trading_class = getattr(chain, "tradingClass", None)
        option_candidates: list[Option] = []
        leg_specs: list[tuple[str, str, int, float, float | None]] = []
        for leg_raw in legs_raw:
            if not isinstance(leg_raw, dict):
                self._status = "Propose: invalid leg config"
                self._render_status()
                return
            action = str(leg_raw.get("action", "")).upper()
            right = str(leg_raw.get("right", "")).upper()
            if action not in ("BUY", "SELL") or right not in ("PUT", "CALL"):
                self._status = "Propose: invalid leg config"
                self._render_status()
                return
            try:
                ratio = int(leg_raw.get("qty", 1) or 1)
            except (TypeError, ValueError):
                ratio = 1
            ratio = max(1, abs(ratio))
            try:
                moneyness = float(leg_raw.get("moneyness_pct", 0.0) or 0.0)
            except (TypeError, ValueError):
                moneyness = 0.0
            delta_target = leg_raw.get("delta")
            try:
                delta_target = float(delta_target) if delta_target is not None else None
            except (TypeError, ValueError):
                delta_target = None

            target_strike = _strike_from_moneyness(spot, right, moneyness)
            right_char = "P" if right == "PUT" else "C"
            strike = None
            if delta_target is not None and strikes:
                strike = await self._strike_by_delta(
                    symbol=symbol,
                    expiry=expiry,
                    right_char=right_char,
                    strikes=list(strikes),
                    trading_class=trading_class,
                    near_strike=target_strike,
                    target_delta=delta_target,
                )
            if strike is None:
                strike = _nearest_strike(strikes, target_strike)
            if strike is None:
                self._status = f"Strike: none for {symbol}"
                self._render_status()
                return
            option_candidates.append(
                Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(strike),
                    right=right_char,
                    exchange="SMART",
                    currency="USD",
                    tradingClass=trading_class,
                )
            )
            leg_specs.append((action, right, ratio, moneyness, delta_target))

        qualified = await self._client.qualify_proxy_contracts(*option_candidates)
        if qualified and len(qualified) == len(option_candidates):
            option_contracts: list[Contract] = list(qualified)
        else:
            option_contracts = list(option_candidates)

        leg_orders: list[_BotLegOrder] = []
        leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
        for contract, (action, _, ratio, _, _) in zip(option_contracts, leg_specs):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract)
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
            leg_quotes.append((bid, ask, last, ticker))

        _finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)

    def _submit_proposal(self) -> None:
        self._submit_selected_proposal()

    def _submit_selected_proposal(self) -> None:
        proposal = self._selected_proposal()
        if not proposal:
            self._status = "Send: no proposal selected"
            self._render_bot()
            return
        if proposal.status != "PROPOSED":
            self._status = f"Send: already {proposal.status}"
            self._render_bot()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Send: no loop"
            self._render_bot()
            return
        self._status = "Sending order..."
        self._render_bot()
        self._send_task = loop.create_task(self._send_order(proposal))

    async def _send_order(self, proposal: _BotProposal) -> None:
        try:
            trade = await self._client.place_limit_order(
                proposal.order_contract,
                proposal.action,
                proposal.quantity,
                proposal.limit_price,
                outside_rth=False,
            )
            order_id = trade.order.orderId or trade.order.permId or 0
            proposal.status = "SENT"
            proposal.order_id = int(order_id or 0) or None
            self._status = f"Sent #{order_id} {proposal.action} {proposal.quantity} @ {proposal.limit_price:.2f}"
        except Exception as exc:
            proposal.status = "ERROR"
            proposal.error = str(exc)
            self._status = f"Send error: {exc}"
        self._refresh_proposals_table()
        self._render_bot()

    def _render_bot(self) -> None:
        lines: list[Text] = [Text("Bot Hub", style="bold")]
        if self._payload:
            symbol = self._payload.get("symbol", "?")
            start = self._payload.get("start", "?")
            end = self._payload.get("end", "?")
            bar_size = self._payload.get("bar_size", "?")
            lines.append(Text(f"Leaderboard: {symbol} {start}→{end} ({bar_size})", style="dim"))
        else:
            lines.append(Text("No leaderboard loaded", style="red"))

        lines.append(
            Text(
                "Enter=Config/Send  Ctrl+A=Presets  f=FilterDTE  w=FilterWin  v=Scope  Tab/h/l=Focus  p=Propose  a=Auto  Space=Run  s=Stop  S=Kill  d=Del  X=Send",
                style="dim",
            )
        )
        dte_label = "ALL" if self._filter_dte is None else str(self._filter_dte)
        win_label = (
            "ALL"
            if self._filter_min_win_rate is None
            else f"≥{int(self._filter_min_win_rate * 100)}%"
        )
        lines.append(Text(f"Filter: DTE={dte_label} (f)  Win={win_label} (w)", style="dim"))
        lines.append(
            Text(
                f"Focus: {self._active_panel}  Presets: {'ON' if self._presets_visible else 'OFF'}  "
                f"Scope: {'ALL' if self._scope_all else 'Instance'}  "
                f"Instances: {len(self._instances)}  Proposals: {len(self._proposal_rows)}",
                style="dim",
            )
        )

        if self._active_panel == "presets":
            selected = self._selected_preset()
            if selected:
                lines.append(Text(""))
                lines.append(Text("Selected preset", style="bold"))
                lines.extend(_preset_lines(selected))
        elif self._active_panel == "instances":
            instance = self._selected_instance()
            if instance:
                instrument = self._strategy_instrument(instance.strategy or {})
                if instrument == "spot":
                    sec_type = str((instance.strategy or {}).get("spot_sec_type") or "").strip().upper()
                    legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT-STK"
                    dte = "-"
                else:
                    legs_desc = _legs_label(instance.strategy.get("legs", []))
                    dte = instance.strategy.get("dte", "?")
                auto = "ON" if instance.auto_trade else "OFF"
                lines.append(Text(""))
                lines.append(Text(f"Selected instance #{instance.instance_id}", style="bold"))
                lines.append(Text(f"{instance.group}  DTE={dte}  Legs={legs_desc}", style="dim"))
                lines.append(Text(f"State={instance.state}  Auto trade={auto}", style="dim"))
        elif self._active_panel == "proposals":
            proposal = self._selected_proposal()
            if proposal:
                lines.append(Text(""))
                lines.append(Text(f"Selected proposal (inst {proposal.instance_id})", style="bold"))
                lines.extend(_proposal_lines(proposal))

        if self._status:
            lines.append(Text(""))
            lines.append(Text(self._status, style="yellow"))

        self._status_panel.update(Text("\n").join(lines))

    def _add_proposal(self, proposal: _BotProposal) -> None:
        self._proposals.append(proposal)
        self._refresh_proposals_table()
        if self._active_panel == "proposals":
            self._proposals_table.cursor_coordinate = (max(len(self._proposal_rows) - 1, 0), 0)

    def _refresh_proposals_table(self) -> None:
        self._proposals_table.clear()
        self._proposal_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            return
        for proposal in self._proposals:
            if scope is not None and proposal.instance_id != scope:
                continue
            self._proposals_table.add_row(*_proposal_row(proposal))
            self._proposal_rows.append(proposal)


def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


def _legs_label(legs: list[dict]) -> str:
    if not legs:
        return "-"
    parts = []
    for leg in legs:
        action = str(leg.get("action", "?"))[:1].upper()
        right = str(leg.get("right", "?"))[:1].upper()
        m = leg.get("moneyness_pct", 0.0)
        try:
            m = float(m)
        except (TypeError, ValueError):
            m = 0.0
        parts.append(f"{action}{right} {m:+.1f}%")
    return " + ".join(parts)


def _legs_direction_hint(legs: list[dict], ema_directional: bool) -> str:
    if ema_directional:
        return "EMA (up/down)"
    if not legs:
        return "ANY"
    first = legs[0] if isinstance(legs, list) else None
    if not isinstance(first, dict):
        return "ANY"
    action = str(first.get("action", "")).upper()
    right = str(first.get("right", "")).upper()
    if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "UP"
    if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "DOWN"
    return "ANY"


def _preset_lines(preset: _BotPreset) -> list[Text]:
    entry = preset.entry
    metrics = entry.get("metrics", {})
    strat = entry.get("strategy", {})
    instrument = str(strat.get("instrument", "options") or "options").strip().lower()

    legs_label = _legs_label(strat.get("legs", []))
    if instrument == "spot":
        mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
        if mapping and mapping.get("up") and mapping.get("down"):
            legs_label = "SPOT (up/down)"
        else:
            legs_label = "SPOT"

        signal_bar = str(strat.get("signal_bar_size") or "").strip()
        entry_signal = str(strat.get("entry_signal") or "ema").strip().lower()
        confirm = strat.get("entry_confirm_bars", 0)
        regime_mode = str(strat.get("regime_mode") or "ema").strip().lower()
        regime = str(strat.get("regime_ema_preset") or "").strip()
        regime_bar = str(strat.get("regime_bar_size") or "").strip()
        regime2_mode = str(strat.get("regime2_mode") or "off").strip().lower()
        regime2 = str(strat.get("regime2_ema_preset") or "").strip()
        regime2_bar = str(strat.get("regime2_bar_size") or "").strip()
        mode_parts: list[str] = []
        if entry_signal == "orb":
            window = strat.get("orb_window_mins", "?")
            rr = strat.get("orb_risk_reward", "?")
            tgt_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
            if tgt_mode not in ("rr", "or_range"):
                tgt_mode = "rr"
            mode_parts.append(f"ORB: {window}m {tgt_mode} rr={rr}")
        else:
            mode_parts.append(f"EMA: {strat.get('ema_preset', '')} cross c{confirm}")
        if regime_mode == "supertrend":
            atr_p = strat.get("supertrend_atr_period", "?")
            mult = strat.get("supertrend_multiplier", "?")
            src = str(strat.get("supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime: ST({atr_p},{mult},{src}) @ {regime_bar or '?'}")
        elif regime:
            mode_parts.append(f"Regime: {regime} @ {regime_bar or '?'}")
        if regime2_mode == "supertrend":
            atr_p = strat.get("regime2_supertrend_atr_period", "?")
            mult = strat.get("regime2_supertrend_multiplier", "?")
            src = str(strat.get("regime2_supertrend_source", "hl2") or "hl2").strip()
            mode_parts.append(f"Regime2: ST({atr_p},{mult},{src}) @ {regime2_bar or '?'}")
        elif regime2_mode == "ema" and regime2:
            mode_parts.append(f"Regime2: {regime2} @ {regime2_bar or '?'}")
        exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
        if exit_mode == "atr":
            atr_p = strat.get("spot_atr_period", "?")
            pt_mult = strat.get("spot_pt_atr_mult", "?")
            sl_mult = strat.get("spot_sl_atr_mult", "?")
            mode_parts.append(f"Exit: ATR({atr_p}) PTx{pt_mult} SLx{sl_mult}")
        if signal_bar:
            mode_parts.append(f"Bar: {signal_bar}")
        mode = "  ".join(mode_parts)
    else:
        pt = float(strat.get("profit_target", 0.0)) * 100.0
        sl = float(strat.get("stop_loss", 0.0)) * 100.0
        mode = (
            f"DTE: {strat.get('dte', '?')}  "
            f"PT: {_fmt_pct(pt)}  SL: {_fmt_pct(sl)}  "
            f"EMA: {strat.get('ema_preset', '')}"
        )

    pnl = float(metrics.get("pnl", 0.0))
    try:
        dd = float(metrics.get("max_drawdown")) if metrics.get("max_drawdown") is not None else None
    except (TypeError, ValueError):
        dd = None
    pnl_over_dd = None
    try:
        pnl_over_dd = (
            float(metrics.get("pnl_over_dd"))
            if metrics.get("pnl_over_dd") is not None
            else (pnl / dd if dd and dd > 0 else None)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        pnl_over_dd = None
    lines = [
        Text(preset.group),
        Text(f"Legs: {legs_label}", style="dim"),
        Text(mode, style="dim"),
        Text(
            f"PnL: {pnl:.2f}  "
            f"Win: {float(metrics.get('win_rate', 0.0)) * 100.0:.1f}%  "
            f"Trades: {int(metrics.get('trades', 0))}",
            style="dim",
        ),
    ]
    if dd is not None:
        extra = f"DD: {dd:.2f}"
        if pnl_over_dd is not None:
            extra += f"  PnL/DD: {pnl_over_dd:.2f}"
        lines.append(Text(extra, style="dim"))
    return lines


def _proposal_lines(proposal: _BotProposal) -> list[Text]:
    legs = proposal.legs or []
    legs_line: str | None = None
    if len(legs) == 1 and proposal.order_contract.secType != "BAG":
        contract = legs[0].contract
        local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
        sec_type = getattr(contract, "secType", "") or ""
        if sec_type == "STK":
            header = f"{local} STK".strip()
        elif sec_type == "FUT":
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            header = f"{local} {expiry} FUT".strip()
        else:
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "") or "?"
            right = getattr(contract, "right", "") or "?"
            strike = getattr(contract, "strike", None)
            header = f"{local} {expiry}{right} {strike}"
    else:
        symbol = getattr(proposal.order_contract, "symbol", "") or getattr(
            proposal.underlying, "symbol", ""
        )
        header = f"{symbol} BAG ({len(legs)} legs)"
        legs_desc: list[str] = []
        for leg in legs[:4]:
            contract = leg.contract
            local = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "")
            legs_desc.append(f"{leg.action[:1]}{local}".strip())
        if legs_desc:
            legs_line = "  ".join(legs_desc) + ("  …" if len(legs) > 4 else "")

    parts: list[Text] = [Text(header, style="dim")]
    if legs_line:
        parts.append(Text(f"Legs: {legs_line}", style="dim"))
    parts.append(Text(f"Side: {proposal.action}  Qty: {proposal.quantity}", style="dim"))
    parts.append(Text(f"Limit: {_fmt_quote(proposal.limit_price)}", style="dim"))
    parts.append(
        Text(
            f"Bid: {_fmt_quote(proposal.bid)}  Ask: {_fmt_quote(proposal.ask)}  "
            f"Last: {_fmt_quote(proposal.last)}",
            style="dim",
        )
    )
    return parts


def _weekday_num(label: str) -> int:
    key = label.strip().upper()[:3]
    mapping = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    return mapping.get(key, 0)


def _parse_entry_days(raw: str) -> list[str]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return []
    normalized = (
        cleaned.replace(";", ",")
        .replace("|", ",")
        .replace("/", ",")
        .replace("\\", ",")
        .replace(" ", ",")
    )
    out: list[str] = []
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        key = token.upper()[:3]
        if key in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"):
            out.append(key.title())
    return out


def _filters_for_group(payload: dict | None, group_name: str) -> dict | None:
    if not payload:
        return None
    for group in payload.get("groups", []):
        if str(group.get("name")) == group_name:
            return group.get("filters") or None
    return None


def _pick_chain_expiry(today: date, dte: int, expirations: list[str]) -> str | None:
    if not expirations:
        return None
    target = _add_business_days(today, dte)
    parsed: list[tuple[date, str]] = []
    for exp in expirations:
        dt = _parse_chain_date(exp)
        if dt:
            parsed.append((dt, exp))
    if not parsed:
        return None
    future = [(dt, exp) for dt, exp in parsed if dt >= target]
    candidates = future or parsed
    candidates.sort(key=lambda pair: abs((pair[0] - target).days))
    return candidates[0][1]


def _parse_chain_date(raw: str) -> date | None:
    raw = str(raw).strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def _add_business_days(anchor: date, days: int) -> date:
    current = anchor
    remaining = max(days, 0)
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _business_days_until(start: date, end: date) -> int:
    if end <= start:
        return 0
    days = 0
    cursor = start
    while cursor < end:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


def _contract_expiry_date(raw: object) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        except ValueError:
            return None
    if len(text) >= 6 and text[:6].isdigit():
        try:
            return date(int(text[:4]), int(text[4:6]), 1)
        except ValueError:
            return None
    return None


def _strike_from_moneyness(spot: float, right: str, moneyness_pct: float) -> float:
    # Negative moneyness means ITM (e.g., -1 = 1% ITM).
    if right == "PUT":
        return spot * (1 - moneyness_pct / 100.0)
    return spot * (1 + moneyness_pct / 100.0)


def _nearest_strike(strikes: list[float], target: float) -> float | None:
    if not strikes:
        return None
    try:
        return min((float(s) for s in strikes), key=lambda s: abs(s - target))
    except (TypeError, ValueError):
        return None


def _get_path(root: object, path: str) -> object:
    current: object = root
    for part in str(path).split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except (TypeError, ValueError):
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _set_path(root: object, path: str, value: object) -> None:
    if not isinstance(path, str) or not path:
        return
    parts = path.split(".")
    current = root
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        next_part = parts[idx + 1] if not last else ""

        if isinstance(current, dict):
            if last:
                current[part] = value
                return
            nxt = current.get(part)
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[part] = nxt
            current = nxt
            continue

        if isinstance(current, list):
            try:
                list_idx = int(part)
            except (TypeError, ValueError):
                return
            while len(current) <= list_idx:
                current.append({})
            if last:
                current[list_idx] = value
                return
            nxt = current[list_idx]
            if not isinstance(nxt, (dict, list)):
                nxt = [] if next_part.isdigit() else {}
                current[list_idx] = nxt
            current = nxt
            continue
        return


def _proposal_row(proposal: _BotProposal) -> tuple[str, str, str, str, str, str, str, str]:
    ts = proposal.created_at.astimezone().strftime("%H:%M:%S")
    inst = str(proposal.instance_id)
    contract = proposal.order_contract
    if contract.secType == "BAG" or len(proposal.legs) > 1:
        symbol = getattr(contract, "symbol", "") or getattr(proposal.underlying, "symbol", "") or "?"
        local = f"{symbol} BAG {len(proposal.legs)}L"
    else:
        leg_contract = proposal.legs[0].contract if proposal.legs else contract
        local = getattr(leg_contract, "localSymbol", "") or getattr(leg_contract, "symbol", "") or "?"
    local = str(local)[:12]
    side = proposal.action[:1]
    qty = str(int(proposal.quantity))
    limit = _fmt_quote(proposal.limit_price)
    bid = _fmt_quote(proposal.bid)
    ask = _fmt_quote(proposal.ask)
    bid_ask = f"{bid}/{ask}"
    status = proposal.status
    if proposal.order_id:
        status = f"{status} #{proposal.order_id}"
    if proposal.error and status == "ERROR":
        status = f"ERROR {proposal.error}"[:32]
    return (ts, inst, side, qty, local, limit, bid_ask, status)


def _pnl_text(value: float | None, *, prefix: str = "") -> Text:
    if value is None:
        return Text("")
    text = f"{prefix}{_fmt_money(value)}"
    if value > 0:
        return Text(text, style="green")
    if value < 0:
        return Text(text, style="red")
    return Text(text)


def _pnl_pct_text(item: PortfolioItem) -> Text:
    value = item.unrealizedPNL
    if value is None:
        return Text("")
    cost_basis = 0.0
    if item.averageCost:
        cost_basis = float(item.averageCost) * float(item.position)
    denom = abs(cost_basis) if cost_basis else abs(float(item.marketValue or 0.0))
    if denom <= 0:
        return Text("")
    pct = (float(value) / denom) * 100.0
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


def _pnl_pct_value(pct: float | None) -> Text:
    if pct is None:
        return Text("")
    text = f"{pct:.2f}%"
    if pct > 0:
        return Text(text, style="green")
    if pct < 0:
        return Text(text, style="red")
    return Text(text)


def _combined_value_pct(value: Text, pct: Text) -> Text:
    text = Text("")
    if value.plain:
        text.append_text(value)
    if pct.plain:
        if text.plain:
            text.append("  ")
        text.append("(")
        text.append_text(pct)
        text.append(")")
    return text


def _pct_change(price: float | None, baseline: float | None) -> float | None:
    if price is None or baseline is None:
        return None
    if baseline <= 0:
        return None
    return ((price - baseline) / baseline) * 100.0


def _pct_dual_text(pct24: float | None, pct72: float | None) -> Text:
    text = Text("")
    if pct24 is not None:
        text.append(f"{pct24:.2f}%", style=_pct_style(pct24))
    if pct72 is not None:
        if pct24 is not None:
            text.append("-", style="red")
        text.append(f"{pct72:.2f}%", style=_pct_style(pct72))
    return text


def _price_pct_dual_text(price: float | None, pct24: float | None, pct72: float | None) -> Text:
    text = Text("")
    if price is not None:
        style = _pct_style(pct24) if pct24 is not None else ""
        text.append(f"{price:,.2f}", style=style)
        if pct24 is not None or pct72 is not None:
            text.append(" ")
    text.append_text(_pct_dual_text(pct24, pct72))
    return text


def _pct_style(pct: float) -> str:
    if pct > 0:
        return "green"
    if pct < 0:
        return "red"
    return ""


def _pnl_value(pnl: PnL | None) -> float | None:
    if not pnl:
        return None
    value = pnl.dailyPnL
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _estimate_net_liq(
    net_liq: float, pnl: PnL | None, anchor: float | None
) -> float | None:
    daily = _pnl_value(pnl)
    if daily is None or anchor is None:
        return None
    return net_liq + (daily - anchor)


def _estimate_buying_power(
    buying_power: float, pnl: PnL | None, anchor: float | None
) -> float | None:
    daily = _pnl_value(pnl)
    if daily is None or anchor is None:
        return None
    return buying_power + (daily - anchor)


_SECTION_ORDER = (
    ("OPTIONS", "OPT"),
    ("STOCKS", "STK"),
    ("FUTURES", "FUT"),
    ("FUTURES OPT", "FOP"),
)
_SECTION_TYPES = {sec_type for _, sec_type in _SECTION_ORDER}

_INDEX_ORDER = ("NQ", "ES", "YM")
_INDEX_LABELS = {
    "NQ": "NASDAQ",
    "ES": "S&P",
    "YM": "DOW",
}
_PROXY_ORDER = ("QQQ", "TQQQ")
_PROXY_LABELS = {
    "QQQ": "QQQ",
    "TQQQ": "TQQQ",
}
_TICKER_WIDTHS = {
    "label": 10,
    "price": 9,
    "change": 9,
    "pct": 10,
}


def _ticker_price(ticker: Ticker) -> float | None:
    bid = _safe_num(getattr(ticker, "bid", None))
    ask = _safe_num(getattr(ticker, "ask", None))
    if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
        return (bid + ask) / 2.0
    last = _safe_num(getattr(ticker, "last", None))
    if last is not None and last > 0:
        return last
    try:
        value = float(ticker.marketPrice())
    except Exception:
        value = None
    if value is not None and value > 0 and not math.isnan(value):
        return value
    return _ticker_close(ticker)


def _ticker_close(ticker: Ticker) -> float | None:
    for attr in ("close", "prevLast"):
        value = getattr(ticker, attr, None)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or num == 0:
            continue
        return num
    return None


def _market_data_tag(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return " [L]"
    if md_type in (3, 4):
        return " [D]"
    return ""


def _market_data_label(ticker: Ticker) -> str:
    md_type = getattr(ticker, "marketDataType", None)
    if md_type in (1, 2):
        return "Live"
    if md_type in (3, 4):
        return "Delayed"
    return "n/a"


def _quote_status_line(ticker: Ticker) -> Text:
    bid = _safe_num(getattr(ticker, "bid", None))
    ask = _safe_num(getattr(ticker, "ask", None))
    last = _safe_num(getattr(ticker, "last", None))
    bid_ask = "ok" if bid is not None and ask is not None else "n/a"
    last_label = "ok" if last is not None else "n/a"
    return Text(f"MD Quotes: bid/ask {bid_ask} · last {last_label}", style="dim")


def _market_session_label() -> str:
    now_et = datetime.now(ZoneInfo("America/New_York")).time()
    if time(4, 0) <= now_et < time(9, 30):
        return "PRE"
    if time(9, 30) <= now_et < time(16, 0):
        return "MRKT"
    if time(16, 0) <= now_et < time(20, 0):
        return "POST"
    return "OVRNGHT"


def _ticker_line(
    order: tuple[str, ...],
    labels: dict[str, str],
    tickers: dict[str, Ticker],
    error: str | None,
    prefix: str,
) -> Text:
    if error:
        return Text(f"{prefix}Data error: {error}", style="red")
    text = Text()
    if prefix:
        text.append(prefix)
    for idx, symbol in enumerate(order):
        if idx:
            text.append(" | ", style="dim")
        label = labels[symbol]
        ticker = tickers.get(symbol)
        if not ticker:
            text.append_text(_ticker_missing(label))
            continue
        tag = _market_data_tag(ticker)
        price = _ticker_price(ticker)
        close = _ticker_close(ticker)
        if price is None or price <= 0:
            if close and close > 0:
                text.append_text(_ticker_closed(label + tag, close))
            else:
                text.append_text(_ticker_missing(label + tag))
            continue
        if close is None or close <= 0:
            text.append_text(_ticker_price_only(label + tag, price))
            continue
        change = price - close
        pct = (change / close) * 100.0
        style = "green" if change > 0 else "red" if change < 0 else ""
        text.append_text(_ticker_block(label + tag, price, change, pct, style))
    return text


def _ticker_block(label: str, price: float, change: float, pct: float, style: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    change_text = f"{change:+.2f}".rjust(_TICKER_WIDTHS["change"])
    pct_text = f"({pct:+.2f}%)".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(change_text, style=style)
    text.append(" ")
    text.append(pct_text, style=style)
    return text


def _ticker_missing(label: str) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "n/a".rjust(_TICKER_WIDTHS["price"])
    blank_change = "".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text, style="dim")
    text.append(" ", style="dim")
    text.append(price_text, style="dim")
    text.append(" ", style="dim")
    text.append(blank_change, style="dim")
    text.append(" ", style="dim")
    text.append(blank_pct, style="dim")
    return text


def _ticker_price_only(label: str, price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = f"{price:,.2f}".rjust(_TICKER_WIDTHS["price"])
    blank_change = "n/a".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text)
    text.append(" ")
    text.append(blank_change, style="dim")
    text.append(" ")
    text.append(blank_pct, style="dim")
    return text


def _ticker_closed(label: str, last_price: float) -> Text:
    label_text = label.ljust(_TICKER_WIDTHS["label"])
    price_text = "Closed".rjust(_TICKER_WIDTHS["price"])
    last_text = f"({last_price:,.2f})".rjust(_TICKER_WIDTHS["change"])
    blank_pct = "".rjust(_TICKER_WIDTHS["pct"])
    text = Text(label_text)
    text.append(" ")
    text.append(price_text, style="red")
    text.append(" ")
    text.append(last_text, style="red")
    text.append(" ")
    text.append(blank_pct, style="red")
    return text


def _safe_num(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or num == 0:
        return None
    return num


def _mark_price(item: PortfolioItem) -> float | None:
    value = _safe_num(getattr(item, "marketPrice", None))
    if value is not None:
        return value
    if item.marketValue is not None and item.position:
        try:
            return float(item.marketValue) / float(item.position)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _infer_multiplier(item: PortfolioItem) -> float:
    position = float(item.position or 0.0)
    market_value = _safe_num(getattr(item, "marketValue", None))
    market_price = _safe_num(getattr(item, "marketPrice", None))
    if position and market_value is not None and market_price is not None:
        denom = market_price * position
        if denom:
            mult = market_value / denom
            if mult and not math.isnan(mult):
                mult = abs(float(mult))
                if mult > 0:
                    return mult
    raw = getattr(item.contract, "multiplier", None)
    try:
        mult = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        mult = 1.0
    if math.isnan(mult) or mult <= 0:
        return 1.0
    return float(mult)


def _cost_basis(item: PortfolioItem) -> float:
    market_value = item.marketValue
    unreal = item.unrealizedPNL
    if market_value is not None and unreal is not None:
        try:
            return float(market_value) - float(unreal)
        except (TypeError, ValueError):
            pass
    avg_cost = item.averageCost
    position = item.position
    if avg_cost is not None and position is not None:
        try:
            return float(avg_cost) * float(position)
        except (TypeError, ValueError):
            pass
    return 0.0


def _midpoint(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _fmt_quote(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _default_order_qty(item: PortfolioItem) -> int:
    return 1


def _tick_size(contract, ticker: Ticker | None, ref_price: float | None) -> float:
    if ticker is not None:
        value = getattr(ticker, "minTick", None)
        if value:
            try:
                tick = float(value)
                if tick > 0:
                    return tick
            except (TypeError, ValueError):
                pass
    value = getattr(contract, "minTick", None)
    if value:
        try:
            tick = float(value)
            if tick > 0:
                return tick
        except (TypeError, ValueError):
            pass
    if contract.secType == "OPT":
        if ref_price is not None and ref_price >= 3:
            return 0.05
        return 0.01
    return 0.01


def _round_to_tick(value: float | None, tick: float) -> float | None:
    if value is None:
        return None
    if not tick:
        return value
    return round(value / tick) * tick


def _tick_decimals(tick: float) -> int:
    text = f"{tick:.10f}".rstrip("0").rstrip(".")
    if "." in text:
        return len(text.split(".")[1])
    return 0


def _optimistic_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    if mid is None:
        return bid if action == "BUY" else ask
    if action == "BUY":
        if bid is None:
            return mid
        return (mid + bid) / 2.0
    if ask is None:
        return mid
    return (mid + ask) / 2.0


def _aggressive_price(
    bid: float | None, ask: float | None, mid: float | None, action: str
) -> float | None:
    if action == "BUY":
        return ask or mid or bid
    return bid or mid or ask


def _append_digit(value: str, char: str, allow_decimal: bool) -> str:
    if char == "." and not allow_decimal:
        return value
    if char == "." and "." in value:
        return value
    if char == "." and not value:
        return "0."
    return value + char


def _parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: str) -> int | None:
    if not value:
        return None
    if not value.isdigit():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
