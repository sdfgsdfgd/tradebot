"""Portfolio table refresh, account truth, and summary-row presentation."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone

from ib_insync import PnL, PortfolioItem
from rich.text import Text

from ...time_utils import to_et as _to_et_shared
from ..common import (
    _SECTION_ORDER,
    _SECTION_TYPES,
    _estimate_buying_power,
    _fmt_money,
    _market_session_label,
    _pnl_text,
    _pnl_value,
    _portfolio_row,
    _portfolio_sort_key,
    _unrealized_pnl_values,
)


class PortfolioTable:
    async def refresh_positions(self, hard: bool = False, *, fetch_snapshot: bool = True) -> None:
        if hard:
            fetch_snapshot = True
        if self._refresh_lock.locked():
            self._dirty = True
            if fetch_snapshot:
                self._dirty_needs_snapshot = True
            return
        async with self._refresh_lock:
            streams_warmed = False
            if fetch_snapshot:
                try:
                    if hard:
                        await self._client.hard_refresh()
                    items = await self._client.fetch_portfolio()
                    self._snapshot.update(items, None)
                    self._last_snapshot_fetch_mono = time.monotonic()
                    streams_warmed = True
                except Exception as exc:  # pragma: no cover - UI surface
                    self._snapshot.update([], str(exc))
            elif hard:
                try:
                    await self._client.hard_refresh()
                except Exception as exc:  # pragma: no cover - UI surface
                    self._snapshot.update([], str(exc))
            if streams_warmed:
                # Gate non-critical streams behind a successful account snapshot
                # so startup doesn't fan out while API session init is unstable.
                self._client.start_index_tickers()
                self._client.start_proxy_tickers()
            self._sync_session_tickers()
            self._index_tickers = self._client.index_tickers()
            self._index_error = self._client.index_error()
            self._proxy_tickers = self._client.proxy_tickers()
            self._proxy_error = self._client.proxy_error()
            self._pnl = self._client.pnl()
            self._net_liq = self._client.account_value("NetLiquidation")
            self._buying_power = self._client.account_value("BuyingPower")
            self._maybe_update_buying_power_anchor()
            self._prime_change_data(self._snapshot.items)
            self._render_table()
            if self._search_active:
                self._render_search()

    def _mark_stream_dirty(self) -> None:
        self._mark_dirty(fetch_snapshot=False)

    def _mark_dirty(self, *, fetch_snapshot: bool = False) -> None:
        self._dirty = True
        if fetch_snapshot:
            self._dirty_needs_snapshot = True
        if self._dirty_task and not self._dirty_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._dirty_task = loop.create_task(self._flush_dirty())

    async def _flush_dirty(self) -> None:
        while self._dirty:
            self._dirty = False
            fetch_snapshot = bool(self._dirty_needs_snapshot)
            self._dirty_needs_snapshot = False
            if not fetch_snapshot:
                if not self._snapshot.items:
                    fetch_snapshot = True
                else:
                    throttle_sec = max(
                        float(self._config.refresh_sec),
                        float(self._SNAPSHOT_THROTTLE_MIN_SEC),
                    )
                    elapsed = time.monotonic() - float(self._last_snapshot_fetch_mono)
                    fetch_snapshot = elapsed >= throttle_sec
            await self.refresh_positions(fetch_snapshot=fetch_snapshot)
            if not self._dirty:
                break
            await asyncio.sleep(self._config.refresh_sec)

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
        self._add_today_ibkr_row(items)
        self._add_net_liq_row(self._net_liq, items)
        self._render_ticker_bar()
        self._status.update(self._status_text())
        self._restore_cursor(prev_row_key, prev_row_index, prev_column)

    def _restore_cursor(self, row_key: str | None, row_index: int, column: int) -> None:
        if not self._row_keys:
            return
        if row_key and row_key in self._row_keys:
            target_row = self._row_keys.index(row_key)
        else:
            target_row = min(max(row_index, 0), len(self._row_keys) - 1)
        target_col = min(max(column, 0), max(self._column_count - 1, 0))
        self._table.cursor_coordinate = (target_row, target_col)

    def _maybe_update_buying_power_anchor(self) -> None:
        value, _currency, updated_at = self._buying_power
        if value is None:
            return
        daily = _pnl_value(self._pnl)
        if daily is None:
            return
        if updated_at is None:
            return
        if self._buying_power_daily_anchor is None and abs(float(daily)) <= 1e-9:
            # Mirror net-liq anchoring so buying-power doesn't jitter at boot.
            return
        if (
            not hasattr(self, "_buying_power_updated_at")
            or self._buying_power_updated_at != updated_at
        ):
            self._buying_power_daily_anchor = daily
            self._buying_power_updated_at = updated_at

    def _sync_session_tickers(self) -> None:
        session = _market_session_label()
        if session == self._md_session:
            return
        self._md_session = session
        self._ticker_con_ids.clear()
        self._ticker_loading.clear()
        self._derivative_actionable_px_by_con_id.clear()
        self._quote_signature_by_con_id.clear()
        self._quote_updated_mono_by_con_id.clear()

    def _status_text(self) -> str:
        conn = self._client.connection_state()
        updated = self._snapshot.updated_at
        if updated:
            ts = _to_et_shared(updated, naive_ts_mode="utc").strftime("%Y-%m-%d %H:%M:%S ET")
        else:
            ts = "n/a"
        session = _market_session_label()
        base = f"IBKR {conn} | last update: {ts} | rows: {self._row_count}"
        base = f"{base} | MKT: {session} | MD: [L]=Live [D]=Delayed"
        base = (
            f"{base} | PnL rows: pos(D|U)=broker daily/unrealized, "
            "total=account(U+R), today(IBKR)=broker daily"
        )
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
        self._table.add_row(
            *self._section_header_row(title, sec_type),
            key=header_key,
        )
        self._row_keys.append(header_key)
        for item in rows:
            base_row_key = self._portfolio_row_key(item)
            row_key = base_row_key
            suffix = 2
            while row_key in self._row_item_by_key or row_key in self._row_keys:
                row_key = f"{base_row_key}:{suffix}"
                suffix += 1
            change_text = self._contract_change_text(item)
            unreal_text, unreal_pct_text = self._unreal_texts(item)
            row_values = _portfolio_row(
                item,
                change_text,
                unreal_text=unreal_text,
                unreal_pct_text=unreal_pct_text,
            )
            row_values[2] = self._avg_cost_cell(item)
            row_values[1] = self._center_cell(row_values[1], self._QTY_COL_WIDTH)
            row_values[2] = self._center_cell(row_values[2], self._AVG_COL_WIDTH)
            row_values[4] = self._center_cell(row_values[4], self._UNREAL_COL_WIDTH)
            row_values[5] = self._center_cell(row_values[5], self._REALIZED_COL_WIDTH)
            self._table.add_row(
                *row_values,
                key=row_key,
            )
            self._row_keys.append(row_key)
            self._row_item_by_key[row_key] = item

    @staticmethod
    def _portfolio_row_key(item: PortfolioItem) -> str:
        contract = getattr(item, "contract", None)
        if contract is None:
            return "UNK"
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper() or "UNK"
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id > 0:
            return f"{sec_type}:{con_id}"
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper() or "?"
        exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        local_symbol = str(getattr(contract, "localSymbol", "") or "").strip().upper()
        if sec_type in ("OPT", "FOP"):
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip().upper()
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            strike_raw = getattr(contract, "strike", None)
            strike = ""
            if strike_raw not in (None, ""):
                try:
                    strike = f"{float(strike_raw):.6f}".rstrip("0").rstrip(".")
                except (TypeError, ValueError):
                    strike = str(strike_raw).strip().upper()
            return f"{sec_type}:{symbol}:{expiry}:{right}:{strike}:{exchange}:{local_symbol}"
        return f"{sec_type}:{symbol}:{exchange}:{local_symbol}"

    def _section_header_row(self, title: str, sec_type: str) -> list[Text]:
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, "bold white on #2b2b2b")
        row = [Text(f"— {title} —", style=style)]
        row.extend(Text("", style=style) for _ in range(self._column_count - 1))
        return row

    def _open_position_totals(
        self,
        items: list[PortfolioItem],
        *,
        prefer_official_unreal: bool = True,
    ) -> tuple[float, float, float, float | None]:
        total_unreal = 0.0
        total_real = 0.0
        total_cost = 0.0
        total_mkt = 0.0
        for item in items:
            if item.contract.secType not in _SECTION_TYPES:
                continue
            official_unreal = (
                self._official_unrealized_value(item)
                if prefer_official_unreal
                else None
            )
            if official_unreal is not None:
                total_unreal += float(official_unreal)
            else:
                mark_price, _ = self._mark_price(item)
                unreal, _ = _unrealized_pnl_values(item, mark_price=mark_price)
                if unreal is not None:
                    total_unreal += float(unreal)
            total_real += float(item.realizedPNL or 0.0)
            if item.averageCost:
                total_cost += abs(float(item.averageCost) * float(item.position))
            if item.marketValue:
                total_mkt += abs(float(item.marketValue))
        total_pnl = total_unreal + total_real
        denom = total_cost if total_cost > 0 else total_mkt
        pct = (total_pnl / denom * 100.0) if denom > 0 else None
        return total_unreal, total_real, total_pnl, pct

    @staticmethod
    def _current_et_day() -> date:
        now_utc = datetime.now(timezone.utc)
        return _to_et_shared(now_utc, naive_ts_mode="utc").date()

    def _open_positions_ibkr_daily(self, items: list[PortfolioItem]) -> float | None:
        total = 0.0
        seen = 0
        for item in items:
            contract = getattr(item, "contract", None)
            sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
            if sec_type not in _SECTION_TYPES:
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id <= 0:
                continue
            daily = self._client.pnl_single_daily(con_id)
            if daily is None:
                continue
            total += float(daily)
            seen += 1
        if seen <= 0:
            return None
        return float(total)

    def _today_ibkr_value(
        self,
        items: list[PortfolioItem],
        *,
        et_day: date | None = None,
    ) -> tuple[float | None, str]:
        day = et_day if et_day is not None else self._current_et_day()
        if self._today_ibkr_day != day:
            self._today_ibkr_day = day
            self._today_ibkr_last_value = None

        account_daily = _pnl_value(self._pnl)
        if account_daily is not None:
            value = float(account_daily)
            self._today_ibkr_last_value = value
            return value, "account"

        open_daily = self._open_positions_ibkr_daily(items)
        if open_daily is not None:
            return float(open_daily), "open"

        if self._today_ibkr_last_value is not None:
            return float(self._today_ibkr_last_value), "cache"
        return None, "warmup"

    def _add_total_row(self, items: list[PortfolioItem]) -> None:
        _account_unreal, account_real, account_total, account_currency, source = self._account_u_r_totals(
            items
        )
        estimate_total: float | None = None
        if account_total is not None and source == "account":
            drift = self._open_unreal_drift_delta(items)
            if drift is not None:
                estimate_total = float(account_total) + float(drift)
        elif source == "open":
            _u_est, _r_est, open_estimate_total, _pct_est = self._open_position_totals(
                items,
                prefer_official_unreal=False,
            )
            estimate_total = float(open_estimate_total)

        style = "bold white on #1f1f1f"
        label = Text(self._account_label("TOTAL (U+R, IBKR)", account_currency), style=style)
        blank = Text("")
        if account_total is None:
            total_text = Text("warming...", style="dim")
        else:
            total_text = _pnl_text(account_total)
            if source == "open":
                total_text.append(" ≈open", style="dim")
        if account_total is not None and estimate_total is not None:
            est_plain = _fmt_money(float(estimate_total))
            total_text.append(" (")
            total_text.append(est_plain, style=self._pnl_style(estimate_total))
            total_text.append(")")
        unreal_text = self._center_cell(total_text, self._UNREAL_COL_WIDTH)
        realized_text = _pnl_text(account_real)
        realized_text = self._center_cell(realized_text, self._REALIZED_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            unreal_text,
            realized_text,
            key="total",
        )
        self._row_keys.append("total")

    def _add_today_ibkr_row(self, items: list[PortfolioItem]) -> None:
        value, source = self._today_ibkr_value(items)
        style = "bold white on #1f1f1f"
        label = Text("TODAY PnL (IBKR)", style=style)
        blank = Text("")
        if value is None:
            amount = Text("warming...", style="dim")
        else:
            amount = _pnl_text(value)
            if source == "open":
                amount.append(" ≈open", style="dim")
            elif source == "cache":
                amount.append(" ≈hold", style="dim")
        amount = self._center_cell(amount, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            amount,
            blank,
            key="today_ibkr",
        )
        self._row_keys.append("today_ibkr")

    @staticmethod
    def _account_label(base: str, currency: str | None) -> str:
        curr = str(currency or "").strip().upper()
        return f"{base} ({curr})" if curr else base

    def _add_net_liq_row(
        self,
        net_liq: tuple[float | None, str | None, datetime | None],
        items: list[PortfolioItem],
    ) -> None:
        value, currency, _updated_at = net_liq
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text(self._account_label("NET LIQ", currency), style=style)
        blank = Text("")
        est_value = self._estimate_net_liq_from_unreal_drift(float(value), items)
        unreal_cell = self._net_liq_amount_cell(float(value), est_value)
        amount_text = self._center_cell(unreal_cell, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            amount_text,
            blank,
            key="netliq",
        )
        self._row_keys.append("netliq")

    def _official_unrealized_value(self, item: PortfolioItem) -> float | None:
        contract = getattr(item, "contract", None)
        con_id = int(getattr(contract, "conId", 0) or 0)
        official_unreal = self._client.pnl_single_unrealized(con_id)
        if official_unreal is None:
            # Keep broker truth available even when pnlSingle is warming/stale.
            official_unreal = self._float_or_none(getattr(item, "unrealizedPNL", None))
        return official_unreal

    def _official_daily_value(self, item: PortfolioItem) -> float | None:
        contract = getattr(item, "contract", None)
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id <= 0:
            return None
        return self._float_or_none(self._client.pnl_single_daily(con_id))

    def _account_metric_value(
        self,
        tag: str,
        *,
        currency: str | None = None,
    ) -> tuple[float | None, str | None, datetime | None]:
        account_value_fn = getattr(self._client, "account_value", None)
        if not callable(account_value_fn):
            return None, None, None
        try:
            value, curr, updated = account_value_fn(tag, currency=currency)
        except TypeError:
            value, curr, updated = account_value_fn(tag)
        except Exception:
            return None, None, None
        parsed = self._float_or_none(value)
        curr_clean = str(curr or "").strip().upper() or None
        return parsed, curr_clean, updated

    def _client_account_u_r_stream(self) -> tuple[float | None, float | None]:
        unreal_fn = getattr(self._client, "pnl_unrealized", None)
        real_fn = getattr(self._client, "pnl_realized", None)

        if callable(unreal_fn):
            try:
                stream_unreal = self._float_or_none(unreal_fn())
            except Exception:
                stream_unreal = None
        else:
            stream_unreal = self._float_or_none(getattr(self._pnl, "unrealizedPnL", None))

        if callable(real_fn):
            try:
                stream_real = self._float_or_none(real_fn())
            except Exception:
                stream_real = None
        else:
            stream_real = self._float_or_none(getattr(self._pnl, "realizedPnL", None))

        return stream_unreal, stream_real

    def _account_reporting_currency(self) -> str | None:
        net_liq = getattr(self, "_net_liq", None)
        if isinstance(net_liq, tuple) and len(net_liq) >= 2:
            curr = str(net_liq[1] or "").strip().upper()
            if curr:
                return curr
        for tag in ("NetLiquidation", "UnrealizedPnL", "RealizedPnL"):
            _value, curr, _updated = self._account_metric_value(tag)
            if curr:
                return curr
        return None

    def _account_u_r_totals(
        self,
        items: list[PortfolioItem],
    ) -> tuple[float | None, float | None, float | None, str | None, str]:
        stream_unreal, stream_real = self._client_account_u_r_stream()
        if stream_unreal is not None and stream_real is not None:
            stream_currency = self._account_reporting_currency()
            return (
                float(stream_unreal),
                float(stream_real),
                float(stream_unreal) + float(stream_real),
                stream_currency,
                "account",
            )

        unreal_value, unreal_curr, _unreal_updated = self._account_metric_value("UnrealizedPnL")
        real_value, real_curr, _real_updated = self._account_metric_value("RealizedPnL")

        if (
            unreal_value is not None
            and real_value is not None
            and (not unreal_curr or not real_curr or unreal_curr == real_curr)
        ):
            final_currency = unreal_curr or real_curr
            return (
                float(unreal_value),
                float(real_value),
                float(unreal_value) + float(real_value),
                final_currency,
                "account",
            )

        candidate_currencies: list[str] = []
        for raw_currency in (self._account_reporting_currency(), unreal_curr, real_curr):
            currency = str(raw_currency or "").strip().upper()
            if currency and currency not in candidate_currencies:
                candidate_currencies.append(currency)
        for currency in candidate_currencies:
            aligned_unreal, _u_curr, _u_updated = self._account_metric_value(
                "UnrealizedPnL",
                currency=currency,
            )
            aligned_real, _r_curr, _r_updated = self._account_metric_value(
                "RealizedPnL",
                currency=currency,
            )
            if aligned_unreal is None or aligned_real is None:
                continue
            return (
                float(aligned_unreal),
                float(aligned_real),
                float(aligned_unreal) + float(aligned_real),
                currency,
                "account",
            )

        _open_unreal, open_real, open_total, _open_pct = self._open_position_totals(items)
        return None, float(open_real), float(open_total), None, "open"

    def _open_unreal_overlap_totals(self, items: list[PortfolioItem]) -> tuple[float, float, int]:
        official_total = 0.0
        estimate_total = 0.0
        overlap_count = 0
        for item in items:
            contract = getattr(item, "contract", None)
            sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
            if sec_type not in _SECTION_TYPES:
                continue
            official_unreal = self._official_unrealized_value(item)
            mark_price, _is_estimate = self._mark_price(item)
            estimate_unreal, _estimate_pct = self._live_unrealized_metrics(item, mark_price)
            if official_unreal is None or estimate_unreal is None:
                continue
            official_total += float(official_unreal)
            estimate_total += float(estimate_unreal)
            overlap_count += 1
        return float(official_total), float(estimate_total), int(overlap_count)

    def _open_unreal_drift_delta(self, items: list[PortfolioItem]) -> float | None:
        official_total, estimate_total, overlap_count = self._open_unreal_overlap_totals(items)
        if overlap_count <= 0:
            return None
        return float(estimate_total - official_total)

    def _estimate_net_liq_from_unreal_drift(
        self,
        raw_net_liq: float,
        items: list[PortfolioItem],
    ) -> float | None:
        drift_delta = self._open_unreal_drift_delta(items)
        if drift_delta is None:
            return None
        return float(raw_net_liq) + float(drift_delta)

    @staticmethod
    def _net_liq_amount_cell(raw_value: float, est_value: float | None) -> Text:
        raw_plain = _fmt_money(raw_value)
        if est_value is None:
            return Text(raw_plain)
        est_plain = _fmt_money(est_value)
        text = Text(f"{raw_plain} ({est_plain})")
        est_start = len(raw_plain) + 2
        est_end = est_start + len(est_plain)
        delta = float(est_value) - float(raw_value)
        if delta > 0:
            text.stylize("green", est_start, est_end)
        elif delta < 0:
            text.stylize("red", est_start, est_end)
        return text

    def _add_buying_power_row(
        self,
        buying_power: tuple[float | None, str | None, datetime | None],
        pnl: PnL | None,
    ) -> None:
        value, currency, _updated_at = buying_power
        if value is None:
            return
        style = "bold white on #1f1f1f"
        label = Text(self._account_label("BUYING POWER", currency), style=style)
        blank = Text("")
        est_value = _estimate_buying_power(value, pnl, self._buying_power_daily_anchor)
        shown_value = est_value if est_value is not None else value
        unreal_cell = Text(_fmt_money(shown_value))
        amount_text = self._center_cell(unreal_cell, self._UNREAL_COL_WIDTH)
        self._table.add_row(
            label,
            blank,
            blank,
            blank,
            amount_text,
            blank,
            key="buying_power",
        )
        self._row_keys.append("buying_power")
