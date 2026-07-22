"""Live bot instance and position-table presentation."""

from __future__ import annotations

import asyncio
import time as pytime

from ib_insync import Contract, PortfolioItem
from rich.text import Text

from ..bot_models import _BotInstance
from ..common import (
    _SECTION_TYPES,
    _cost_basis,
    _fmt_money,
    _fmt_quote,
    _infer_multiplier,
    _market_data_label,
    _pct24_72_from_price,
    _pnl_text,
    _portfolio_sort_key,
    _price_pct_dual_text,
    _safe_float,
    _safe_num,
    _ticker_close,
    _ticker_price,
)
from .formatting import _InstancePnlState, _center_table_row


class BotPositionsMixin:
    def _refresh_instances_table(self, *, refresh_dependents: bool = True) -> None:
        prev_row = self._instances_table.cursor_coordinate.row
        prev_id = None
        if 0 <= prev_row < len(self._instance_rows):
            prev_id = self._instance_rows[prev_row].instance_id

        self._instances_table.clear()
        self._instance_rows = []
        live_total_by_id: dict[int, float | None] = {}
        active_ids = {int(instance.instance_id) for instance in self._instances}
        self._instance_pnl_state_by_id = {
            int(instance_id): state
            for instance_id, state in self._instance_pnl_state_by_id.items()
            if int(instance_id) in active_ids
        }
        for instance in self._instances:
            instrument = self._strategy_instrument(instance.strategy or {})
            if instrument == "spot":
                dte = "-"
            else:
                dte = instance.strategy.get("dte", "")
            bt_pnl: Text | str = ""
            if instance.metrics:
                try:
                    bt_pnl = _pnl_text(float(instance.metrics.get("pnl", 0.0)))
                except (TypeError, ValueError):
                    bt_pnl = ""
            state_value = str(instance.state or "").strip().upper() or "UNKNOWN"
            if state_value == "RUNNING":
                state_cell: Text | str = Text(state_value, style="bold #73d89e")
            elif state_value == "PAUSED":
                state_cell = Text(state_value, style="bold #b8c0cb")
            else:
                state_cell = Text(state_value, style="bold #d6a56f")
            unreal_cell, realized_cell, total_cell, live_total = self._instance_live_cells(instance)
            live_total_by_id[int(instance.instance_id)] = live_total
            self._instances_table.add_row(
                *_center_table_row(
                    str(instance.instance_id),
                    instance.group[:24],
                    str(dte),
                    state_cell,
                    bt_pnl,
                    unreal_cell,
                    realized_cell,
                    total_cell,
                )
            )
            self._instance_rows.append(instance)
        self._instance_live_total_by_id = live_total_by_id

        if prev_id is not None:
            for idx, inst in enumerate(self._instance_rows):
                if inst.instance_id == prev_id:
                    self._instances_table.cursor_coordinate = (idx, 0)
                    break
        elif self._instance_rows:
            self._instances_table.cursor_coordinate = (0, 0)

        self._sync_row_marker(self._instances_table, force=True)
        if refresh_dependents and not self._scope_all:
            self._refresh_orders_table()
            self._refresh_logs_table()

    async def _refresh_positions(self) -> None:
        try:
            items = await self._client.fetch_portfolio()
        except Exception as exc:  # pragma: no cover - UI surface
            self._set_status(f"Positions error: {exc}")
            return
        self._positions = [item for item in items if item.contract.secType in _SECTION_TYPES]
        self._positions.sort(key=_portfolio_sort_key, reverse=True)
        await self._prime_position_tickers()
        self._refresh_instances_table(refresh_dependents=False)
        self._refresh_orders_table()

    async def _prime_position_tickers(self) -> None:
        watched_con_ids = set().union(*(instance.touched_conids for instance in self._instances))
        if not watched_con_ids:
            return
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if contract is None:
                continue
            try:
                con_id = int(getattr(contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                continue
            if con_id <= 0 or con_id not in watched_con_ids or con_id in self._tracked_conids:
                continue
            try:
                await self._client.ensure_ticker(contract, owner="bot")
            except Exception:
                continue
            self._tracked_conids.add(con_id)
            self._start_closes_load(con_id, contract)

    def _start_closes_load(self, con_id: int, contract: Contract) -> None:
        if not callable(getattr(self._client, "session_close_anchors", None)):
            return
        con_id = int(con_id or 0)
        if con_id <= 0:
            return
        cached = self._session_closes_by_con_id.get(con_id)
        if cached is not None and cached[1] is not None:
            return
        if con_id in self._closes_loading:
            return
        if cached is not None:
            now = pytime.monotonic()
            retry_at = float(self._closes_retry_at_by_con_id.get(con_id, 0.0) or 0.0)
            if now < retry_at:
                return
            self._closes_retry_at_by_con_id[con_id] = now + float(self._CLOSES_RETRY_SEC)
        else:
            self._closes_retry_at_by_con_id.pop(con_id, None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._closes_loading.add(con_id)
        loop.create_task(self._load_closes(con_id, contract))

    async def _load_closes(self, con_id: int, contract: Contract) -> None:
        try:
            prev_close, close_1ago, close_3ago = await self._client.session_close_anchors(contract)
            self._session_closes_by_con_id[int(con_id)] = (prev_close, close_3ago)
            self._session_close_1ago_by_con_id[int(con_id)] = close_1ago
            if close_3ago is not None:
                self._closes_retry_at_by_con_id.pop(int(con_id), None)
            else:
                self._closes_retry_at_by_con_id[int(con_id)] = (
                    pytime.monotonic() + float(self._CLOSES_RETRY_SEC)
                )
        except Exception:
            self._closes_retry_at_by_con_id[int(con_id)] = (
                pytime.monotonic() + float(self._CLOSES_RETRY_SEC)
            )
            return
        finally:
            self._closes_loading.discard(int(con_id))
        self._on_client_stream_update()

    def _position_mark_price(self, item: PortfolioItem) -> float | None:
        contract = getattr(item, "contract", None)
        if contract is None:
            return None
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        try:
            con_id = int(getattr(contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            con_id = 0
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        price = _ticker_price(ticker) if ticker else None
        if price is not None:
            return float(price)
        if sec_type == "FOP":
            return None
        market_price = _safe_num(getattr(item, "marketPrice", None))
        return float(market_price) if market_price is not None else None

    def _position_pnl_values(self, item: PortfolioItem) -> tuple[float | None, float | None]:
        unreal, _official, _estimate, _mark_price = self._position_unrealized_values(item)
        realized = _safe_num(getattr(item, "realizedPNL", None))
        return unreal, realized

    @staticmethod
    def _item_con_id(item: PortfolioItem) -> int:
        contract = getattr(item, "contract", None)
        try:
            return int(getattr(contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _official_unrealized_value(self, item: PortfolioItem) -> float | None:
        con_id = self._item_con_id(item)
        official = self._client.pnl_single_unrealized(con_id)
        if official is not None:
            return float(official)
        fallback = _safe_float(getattr(item, "unrealizedPNL", None))
        return float(fallback) if fallback is not None else None

    def _official_daily_value(self, item: PortfolioItem) -> float | None:
        con_id = self._item_con_id(item)
        if con_id <= 0:
            return None
        daily = self._client.pnl_single_daily(con_id)
        return float(daily) if daily is not None else None

    def _live_unrealized_value(self, item: PortfolioItem, mark_price: float | None) -> float | None:
        mark = _safe_num(mark_price)
        qty = _safe_num(getattr(item, "position", None))
        if mark is None or qty is None:
            return None
        if abs(float(qty)) <= 1e-12:
            return 0.0
        multiplier = _infer_multiplier(item)
        cost_basis = _cost_basis(item)
        return (float(mark) * float(qty) * float(multiplier)) - float(cost_basis)

    def _position_unrealized_values(
        self,
        item: PortfolioItem,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        con_id = self._item_con_id(item)
        mark_price = self._position_mark_price(item)
        official = self._official_unrealized_value(item)
        estimate = self._live_unrealized_value(item, mark_price)
        snapshot_unreal = _safe_float(getattr(item, "unrealizedPNL", None))
        unreal = official if official is not None else estimate
        if unreal is None and snapshot_unreal is not None:
            unreal = float(snapshot_unreal)

        now_mono = pytime.monotonic()
        if con_id > 0 and unreal is not None:
            self._last_unreal_by_conid[int(con_id)] = (float(unreal), float(now_mono))
        elif con_id > 0 and unreal is None:
            cached = self._last_unreal_by_conid.get(int(con_id))
            if cached is not None:
                cached_unreal, cached_mono = cached
                if (float(now_mono) - float(cached_mono)) <= float(self._UNREAL_STICKY_SEC):
                    unreal = float(cached_unreal)
        return unreal, official, estimate, mark_price

    @staticmethod
    def _edge_glyph_style(value: float | None) -> tuple[str, str]:
        if value is None:
            return "•", "dim"
        if value > 1e-6:
            return "▲", "green"
        if value < -1e-6:
            return "▼", "red"
        return "•", "dim"

    def _position_entry_now_cell(self, item: PortfolioItem, mark_price: float | None) -> Text:
        avg_cost = _safe_num(getattr(item, "averageCost", None))
        mark = _safe_num(mark_price)
        sec_type = str(getattr(getattr(item, "contract", None), "secType", "") or "").strip().upper()
        if mark is None and sec_type != "FOP":
            mark = _safe_num(getattr(item, "marketPrice", None))
        multiplier = abs(float(_infer_multiplier(item)))
        entry = float(avg_cost) if avg_cost is not None else None
        if (
            entry is not None
            and sec_type in ("OPT", "FOP", "FUT")
            and multiplier > 0
            and abs(multiplier - 1.0) > 1e-9
        ):
            entry = float(entry) / float(multiplier)

        edge_pct: float | None = None
        if entry is not None and mark is not None and abs(float(entry)) > 1e-12:
            raw_pct = ((float(mark) - float(entry)) / abs(float(entry))) * 100.0
            qty = _safe_num(getattr(item, "position", None))
            if qty is not None and qty < 0:
                raw_pct *= -1.0
            edge_pct = float(raw_pct)

        glyph, glyph_style = self._edge_glyph_style(edge_pct)
        entry_plain = _fmt_quote(entry)
        now_plain = _fmt_quote(mark)
        text = Text("")
        text.append(f"{glyph} ", style=glyph_style)
        text.append(entry_plain, style="grey50" if entry is not None else "dim")
        text.append("¦", style="grey35")
        text.append(f" {now_plain}", style=glyph_style if mark is not None else "dim")
        return text

    def _position_px_change_cell(self, item: PortfolioItem, mark_price: float | None) -> Text:
        contract = getattr(item, "contract", None)
        con_id = int(getattr(contract, "conId", 0) or 0) if contract is not None else 0
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper() if contract is not None else ""
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        price = _safe_num(mark_price)
        if price is None:
            price = _ticker_price(ticker) if ticker is not None else None
        if price is None and sec_type != "FOP":
            price = _safe_num(getattr(item, "marketPrice", None))

        if con_id and contract is not None:
            start_closes = getattr(self, "_start_closes_load", None)
            if callable(start_closes):
                start_closes(con_id, contract)
        closes_by_con_id = getattr(self, "_session_closes_by_con_id", None)
        cached = closes_by_con_id.get(con_id) if con_id and isinstance(closes_by_con_id, dict) else None
        session_prev_close = cached[0] if cached else None
        close_3ago = cached[1] if cached else None
        latest_close = _ticker_close(ticker) if ticker is not None else None
        if session_prev_close is None:
            session_prev_close = latest_close
        pct24, pct72 = _pct24_72_from_price(
            price=price,
            ticker=ticker,
            session_prev_close=session_prev_close,
            session_close_3ago=close_3ago,
        )
        ref = pct24 if pct24 is not None else pct72
        glyph, glyph_style = self._edge_glyph_style(ref)

        dual = _price_pct_dual_text(None, pct24, pct72, separator="·")
        # Remove internal left-pad from the shared formatter so glyph and value stay visually coupled.
        while dual.plain.startswith(" "):
            dual = dual[1:]
        text = Text("")
        text.append(f"{glyph} ", style=glyph_style)
        text.append_text(dual if dual.plain else Text("n/a", style="dim"))
        return text

    def _position_md_badge_cell(self, item: PortfolioItem) -> Text:
        contract = getattr(item, "contract", None)
        con_id = int(getattr(contract, "conId", 0) or 0) if contract is not None else 0
        ticker = self._client.ticker_for_con_id(con_id) if con_id else None
        if ticker is None:
            return Text("")
        label = _market_data_label(ticker)
        code, style = {
            "Live": ("L", "bold black on #73d89e"),
            "Live-Frozen": ("LF", "bold black on #7fc79f"),
            "Delayed": ("D", "bold black on #cfb473"),
            "Delayed-Frozen": ("DF", "bold black on #ad9b79"),
        }.get(label, ("?", "bold black on #7a8792"))
        return Text(f" {code} ", style=style)

    def _spot_instance_positions(self, instance: _BotInstance) -> list[PortfolioItem]:
        symbol = str(instance.symbol or "").strip().upper()
        if not symbol:
            return []
        sec_type = self._spot_sec_type(instance, symbol)
        touched = {int(con_id) for con_id in instance.touched_conids if int(con_id) > 0}
        matches: list[PortfolioItem] = []
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if contract is None:
                continue
            if str(getattr(contract, "secType", "") or "").strip().upper() != sec_type:
                continue
            if str(getattr(contract, "symbol", "") or "").strip().upper() != symbol:
                continue
            try:
                qty = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if abs(qty) <= 1e-12:
                continue
            matches.append(item)
        if not matches:
            return []

        if touched:
            scoped = [item for item in matches if self._item_con_id(item) in touched]
            if scoped:
                matches = scoped

        for item in matches:
            con_id = self._item_con_id(item)
            if con_id <= 0:
                continue
            instance.touched_conids.add(con_id)
        return matches

    def _instance_positions(self, instance: _BotInstance) -> list[PortfolioItem]:
        instrument = self._strategy_instrument(instance.strategy or {})
        if instrument == "spot":
            return self._spot_instance_positions(instance)
        return self._options_open_positions(instance)

    def _instance_realized_total(self, instance: _BotInstance, items: list[PortfolioItem]) -> float:
        state = self._instance_pnl_state_by_id.get(int(instance.instance_id))
        if state is None:
            state = _InstancePnlState()
            self._instance_pnl_state_by_id[int(instance.instance_id)] = state

        realized_by_conid: dict[int, float] = {}
        direct_realized = 0.0
        for item in items:
            realized = _safe_num(getattr(item, "realizedPNL", None))
            if realized is None:
                continue
            con_id = self._item_con_id(item)
            if con_id > 0:
                realized_by_conid[con_id] = float(realized)
            else:
                direct_realized += float(realized)

        for con_id, realized in realized_by_conid.items():
            prev = state.realized_seen_by_conid.get(con_id)
            if prev is None:
                state.realized_total += float(realized)
            else:
                delta = float(realized) - float(prev)
                if delta > 1e-9:
                    state.realized_total += float(delta)
            state.realized_seen_by_conid[con_id] = float(realized)

        return float(state.realized_total + direct_realized)

    def _instance_live_cells(
        self,
        instance: _BotInstance,
    ) -> tuple[Text | str, Text | str, Text | str, float | None]:
        items = self._instance_positions(instance)

        unreal_hybrid = 0.0
        unreal_hybrid_seen = False
        unreal_estimate = 0.0
        unreal_estimate_seen = False
        unreal_official_seen = False
        for item in items:
            unreal, official, estimate, _mark_price = self._position_unrealized_values(item)
            if unreal is not None:
                unreal_hybrid += float(unreal)
                unreal_hybrid_seen = True
            if official is not None:
                unreal_official_seen = True
            if estimate is not None:
                unreal_estimate += float(estimate)
                unreal_estimate_seen = True

        unreal_value = float(unreal_hybrid) if unreal_hybrid_seen else 0.0
        realized_value = self._instance_realized_total(instance, items)
        total_value = float(unreal_value + realized_value)
        is_running = str(getattr(instance, "state", "")).strip().upper() == "RUNNING"
        has_live = bool(items) or abs(float(realized_value)) > 1e-9 or is_running
        live_total = float(total_value) if has_live else None

        unreal_cell = _pnl_text(unreal_value)
        unreal_glyph, unreal_glyph_style = self._edge_glyph_style(unreal_value)
        unreal_with_glyph = Text("")
        unreal_with_glyph.append(f"{unreal_glyph} ", style=unreal_glyph_style)
        unreal_with_glyph.append_text(unreal_cell)
        unreal_cell = unreal_with_glyph
        if unreal_estimate_seen and not unreal_official_seen:
            unreal_cell.append(" ≈", style="dim")
        elif unreal_estimate_seen and unreal_hybrid_seen and abs(float(unreal_estimate) - float(unreal_hybrid)) >= 0.01:
            unreal_cell.append(" (", style="dim")
            est_style = "green" if unreal_estimate > 0 else "red" if unreal_estimate < 0 else ""
            unreal_cell.append(_fmt_money(float(unreal_estimate)), style=est_style)
            unreal_cell.append(")", style="dim")

        realized_cell = _pnl_text(realized_value)
        total_cell = _pnl_text(total_value)
        total_glyph, total_glyph_style = self._edge_glyph_style(total_value)
        total_with_glyph = Text("")
        total_with_glyph.append(f"{total_glyph} ", style=total_glyph_style)
        total_with_glyph.append_text(total_cell)
        total_cell = total_with_glyph
        return unreal_cell, realized_cell, total_cell, live_total
