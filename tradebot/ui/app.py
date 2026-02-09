"""Portfolio TUI (positions) + bot hub entrypoint."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

from ib_insync import PnL, PortfolioItem, Ticker
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from ..client import IBKRClient
from ..config import load_config
from .bot_runtime import BotRuntime
from .common import (
    _INDEX_LABELS,
    _INDEX_ORDER,
    _PROXY_LABELS,
    _PROXY_ORDER,
    _SECTION_ORDER,
    _SECTION_TYPES,
    _combined_value_pct,
    _fmt_money,
    _cost_basis,
    _estimate_buying_power,
    _estimate_net_liq,
    _infer_multiplier,
    _market_session_label,
    _pct_change,
    _pct_dual_text,
    _pnl_pct_text,
    _pnl_pct_value,
    _pnl_text,
    _pnl_value,
    _portfolio_row,
    _portfolio_sort_key,
    _price_pct_dual_text,
    _quote_status_line,
    _safe_num,
    _ticker_close,
    _ticker_line,
    _ticker_price,
)
from .positions import PositionDetailScreen
from .store import PortfolioSnapshot


# region Positions UI
class PositionsApp(App):
    _SECTION_HEADER_STYLE_BY_TYPE = {
        "OPT": "bold #d7e8ff on #232834",
        "STK": "bold #d8f3e7 on #1f2d29",
        "FUT": "bold #ffe7c7 on #2e2720",
        "FOP": "bold #ece0ff on #2a2433",
    }

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

    #positions:focus {
        border: solid #26567a;
    }

    #positions > .datatable--cursor {
        background: #181b20;
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

    #bot-orders {
        height: 1fr;
        padding: 0 1;
        border: solid #003054;
    }

    #bot-logs {
        height: 1fr;
        padding: 0 1;
        border: solid #003054;
    }

    #bot-presets:focus,
    #bot-instances:focus,
    #bot-orders:focus,
    #bot-logs:focus,
    #bot-config:focus {
        border: solid #2c82c9;
    }

    #bot-presets > .datatable--cursor {
        background: #1c3348;
    }

    #bot-instances > .datatable--cursor {
        background: #1f3a33;
    }

    #bot-orders > .datatable--cursor {
        background: #3d321f;
    }

    #bot-logs > .datatable--cursor {
        background: #2f2f45;
    }

    #bot-config > .datatable--cursor {
        background: #2a3a4a;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = load_config()
        self._client = IBKRClient(self._config)
        self._bot_runtime = BotRuntime(self._client, self._config.detail_refresh_sec)
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
        self._bot_runtime.install(self)
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
        self._bot_runtime.toggle(self)

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
        section_rows_by_type = self._section_rows_by_type(items)
        self._row_count = sum(len(rows) for rows in section_rows_by_type.values())
        for title, sec_type in _SECTION_ORDER:
            rows = section_rows_by_type.get(sec_type, [])
            if not rows:
                continue
            self._add_section(title, sec_type, rows)
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
        conn = self._client.connection_state()
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

    def _section_rows_by_type(self, items: list[PortfolioItem]) -> dict[str, list[PortfolioItem]]:
        rows_by_type: dict[str, list[PortfolioItem]] = {sec_type: [] for _, sec_type in _SECTION_ORDER}
        for item in items:
            sec_type = str(getattr(item.contract, "secType", "") or "").strip().upper()
            if sec_type in rows_by_type:
                rows_by_type[sec_type].append(item)
        for rows in rows_by_type.values():
            rows.sort(key=_portfolio_sort_key, reverse=True)
        return rows_by_type

    def _add_section(self, title: str, sec_type: str, rows: list[PortfolioItem]) -> None:
        if not rows:
            return
        header_key = f"header:{sec_type}"
        self._table.add_row(*self._section_header_row(title, sec_type), key=header_key)
        self._row_keys.append(header_key)
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

    def _section_header_row(self, title: str, sec_type: str) -> list[Text]:
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, "bold white on #2b2b2b")
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
            await self._client.ensure_ticker(contract, owner="positions")
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
