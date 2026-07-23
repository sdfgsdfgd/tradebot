"""Portfolio market-data loading and display-value derivation."""

from __future__ import annotations

import asyncio
import time

from ib_insync import Contract, PortfolioItem, Ticker
from rich.text import Text

from ...engines.execution import _ticker_close, _ticker_price
from ..common import (
    _INDEX_FUT_LABELS,
    _INDEX_FUT_ORDER,
    _INDEX_LABELS,
    _INDEX_ORDER,
    _PROXY_LABELS,
    _PROXY_ORDER,
    _SECTION_TYPES,
    _cost_basis,
    _infer_multiplier,
    _market_session_label,
    _option_display_price,
    _pct24_72_from_price,
    _quote_age_ribbon,
    _safe_float,
    _safe_num,
    _ticker_actionable_price,
    _ticker_line,
)


class PortfolioMarketValues:
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
        cached = self._session_closes_by_con_id.get(con_id)
        if cached is not None and cached[1] is not None:
            return
        if con_id in self._closes_loading:
            return
        if cached is not None:
            now = time.monotonic()
            retry_at = self._closes_retry_at_by_con_id.get(con_id, 0.0)
            if now < retry_at:
                return
            self._closes_retry_at_by_con_id[con_id] = now + self._CLOSES_RETRY_SEC
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
            self._session_closes_by_con_id[con_id] = (prev_close, close_3ago)
            self._session_close_1ago_by_con_id[con_id] = close_1ago
            if close_3ago is not None:
                self._closes_retry_at_by_con_id.pop(con_id, None)
            else:
                self._closes_retry_at_by_con_id[con_id] = (
                    time.monotonic() + self._CLOSES_RETRY_SEC
                )
        except Exception:
            self._closes_retry_at_by_con_id[con_id] = time.monotonic() + self._CLOSES_RETRY_SEC
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
        return self._change_text_for_con_id(con_id, item=item)

    def _change_text_for_con_id(self, con_id: int, *, item: PortfolioItem | None = None) -> Text:
        if not con_id:
            return Text("")
        ticker = self._client.ticker_for_con_id(con_id)
        quote_price = _ticker_price(ticker) if ticker else None
        price = quote_price
        if item is not None:
            sec_type = str(getattr(item.contract, "secType", "") or "").strip().upper()
            if sec_type in ("OPT", "FOP"):
                mark_price, _is_estimate = self._option_estimated_mark(item, ticker)
                if mark_price is not None:
                    price = float(mark_price)
                else:
                    display_price = _option_display_price(item, ticker)
                    if display_price is not None:
                        price = float(display_price)
        cached = self._session_closes_by_con_id.get(con_id)
        session_prev_close = cached[0] if cached else None
        close_3ago = cached[1] if cached else None
        pct24, pct72 = _pct24_72_from_price(
            price=price,
            ticker=ticker,
            session_prev_close=session_prev_close,
            session_close_3ago=close_3ago,
        )
        age_sec = self._quote_age_seconds(con_id, ticker, price if price is not None else quote_price)
        ribbon = _quote_age_ribbon(age_sec)
        text = self._aligned_px_change_text(
            pct24=pct24,
            pct72=pct72,
            ribbon=ribbon,
        )
        centered = self._center_cell(text, self._PX_24_72_COL_WIDTH)
        return centered if isinstance(centered, Text) else Text(str(centered))

    @staticmethod
    def _pct_text_style(value: float | None) -> str:
        if value is None:
            return "dim"
        if value > 0:
            return "green"
        if value < 0:
            return "red"
        return "dim"

    def _aligned_px_change_text(
        self,
        *,
        pct24: float | None,
        pct72: float | None,
        ribbon: Text | None = None,
    ) -> Text:
        # Compact layout: keep glyph/divider/percentages tightly coupled.
        ribbon_width = 4

        if pct24 is None:
            glyph = Text("•", style="dim")
        elif pct24 > 0:
            glyph = Text("▲", style="bold green")
        elif pct24 < 0:
            glyph = Text("▼", style="bold red")
        else:
            glyph = Text("•", style="dim")

        mid_text = "n/a" if pct24 is None else f"{float(pct24):.2f}%"
        right_text = "n/a" if pct72 is None else f"{float(pct72):.2f}%"

        ribbon_text = Text(" " * ribbon_width)
        if ribbon is not None and ribbon.plain:
            trimmed_ribbon = ribbon[:ribbon_width]
            left_pad = max((ribbon_width - len(trimmed_ribbon.plain)) // 2, 0)
            right_pad = max(ribbon_width - len(trimmed_ribbon.plain) - left_pad, 0)
            ribbon_text = Text(" " * left_pad)
            ribbon_text.append_text(trimmed_ribbon)
            ribbon_text.append(" " * right_pad)

        text = Text("")
        text.append(glyph.plain, style=self._pct_text_style(pct24))
        text.append(" ¦ ", style="grey35")
        text.append(mid_text, style=self._pct_text_style(pct24))
        text.append(" · ", style="dim")
        text.append(right_text, style=self._pct_text_style(pct72))
        text.append(" ", style="dim")
        text.append_text(ribbon_text)
        return text

    def _quote_age_seconds(
        self,
        con_id: int,
        ticker: Ticker | None,
        price: float | None,
    ) -> float | None:
        if not con_id or not ticker:
            return None
        bid = _safe_num(getattr(ticker, "bid", None))
        ask = _safe_num(getattr(ticker, "ask", None))
        last = _safe_num(getattr(ticker, "last", None))
        signature = (bid, ask, last, price)
        if not any(value is not None for value in signature):
            return None
        now = time.monotonic()
        previous = self._quote_signature_by_con_id.get(con_id)
        if previous != signature:
            self._quote_signature_by_con_id[con_id] = signature
            self._quote_updated_mono_by_con_id[con_id] = now
            return 0.0
        updated = self._quote_updated_mono_by_con_id.get(con_id)
        if updated is None:
            self._quote_updated_mono_by_con_id[con_id] = now
            return 0.0
        return max(0.0, now - updated)

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        return _safe_float(value)

    @staticmethod
    def _pnl_style(value: float | None) -> str:
        if value is None:
            return "dim"
        return "green" if value >= 0 else "red"

    def _live_unrealized_metrics(
        self,
        item: PortfolioItem,
        mark_price: float | None,
    ) -> tuple[float | None, float | None]:
        mark = self._float_or_none(mark_price)
        if mark is None:
            return None, None
        qty = self._float_or_none(getattr(item, "position", None))
        if qty is None:
            return None, None
        if abs(float(qty)) <= 1e-12:
            return 0.0, 0.0
        multiplier = _infer_multiplier(item)
        cost_basis = _cost_basis(item)
        mark_value = float(mark) * float(qty) * float(multiplier)
        unreal = mark_value - float(cost_basis)
        denom = abs(float(cost_basis)) if float(cost_basis) else abs(mark_value)
        pct = (unreal / denom * 100.0) if denom > 0 else None
        return unreal, pct

    @staticmethod
    def _entry_now_value(value: float | None, *, max_width: int | None = None) -> str:
        if value is None:
            return "n/a"
        parsed = float(value)
        if max_width is None:
            return f"{parsed:,.2f}"

        candidates = (
            f"{parsed:,.2f}",
            f"{parsed:,.1f}",
            f"{parsed:,.0f}",
            f"{parsed:.2f}",
            f"{parsed:.1f}",
            f"{parsed:.0f}",
        )
        for candidate in candidates:
            if len(candidate) <= max_width:
                return candidate

        abs_value = abs(parsed)
        for scale, suffix in (
            (1_000.0, "K"),
            (1_000_000.0, "M"),
            (1_000_000_000.0, "B"),
            (1_000_000_000_000.0, "T"),
        ):
            if abs_value < scale:
                continue
            compact_candidates = (
                f"{parsed / scale:.2f}{suffix}",
                f"{parsed / scale:.1f}{suffix}",
                f"{parsed / scale:.0f}{suffix}",
            )
            for compact in compact_candidates:
                if len(compact) <= max_width:
                    return compact

        fallback = f"{parsed:.0f}"
        if len(fallback) <= max_width:
            return fallback
        return fallback[: max(1, max_width)]

    def _entry_now_inputs(
        self,
        item: PortfolioItem,
    ) -> tuple[float | None, float | None, float | None]:
        avg_cost = self._float_or_none(getattr(item, "averageCost", None))
        mark_price, _is_estimate = self._mark_price(item)
        mark = self._float_or_none(mark_price)
        if avg_cost is None:
            return None, mark, None

        sec_type = str(getattr(item.contract, "secType", "") or "").strip().upper()
        multiplier = abs(float(_infer_multiplier(item)))
        entry = float(avg_cost)
        if sec_type in ("OPT", "FOP", "FUT") and multiplier > 0 and abs(multiplier - 1.0) > 1e-9:
            entry = entry / float(multiplier)

        edge_pct: float | None = None
        if mark is not None and abs(entry) > 1e-12:
            raw_pct = ((float(mark) - float(entry)) / abs(float(entry))) * 100.0
            qty = self._float_or_none(getattr(item, "position", None))
            if qty is not None and qty < 0:
                raw_pct *= -1.0
            edge_pct = raw_pct
        return entry, mark, edge_pct

    def _avg_cost_cell(self, item: PortfolioItem) -> Text:
        width = max(int(self._AVG_COL_WIDTH), 10)
        outer_pad = 1
        sep = "¦"
        core_width = max(width - (outer_pad * 2), 1)
        left_width = max((core_width - len(sep)) // 2, 1)
        right_width = max(core_width - left_width - len(sep), 1)

        entry, now, edge_pct = self._entry_now_inputs(item)
        entry_plain = self._entry_now_value(entry, max_width=max(1, left_width - 2))
        now_plain = self._entry_now_value(now, max_width=right_width)

        glyph = "•"
        edge_style = "dim"
        if edge_pct is not None:
            if edge_pct > 1e-6:
                glyph = "▲"
                edge_style = "green"
            elif edge_pct < -1e-6:
                glyph = "▼"
                edge_style = "red"

        left_plain = f"{glyph} {entry_plain}"
        if len(left_plain) > left_width:
            left_plain = left_plain[:left_width]
        if len(now_plain) > right_width:
            now_plain = now_plain[:right_width]
        left_block = self._center_with_sep_bias(left_plain, left_width, max_right_gap=1)
        right_block = self._center_with_sep_bias(now_plain, right_width, max_left_gap=1)
        core_plain = f"{left_block}{sep}{right_block}"

        if len(core_plain) < core_width:
            core_plain = f"{core_plain}{' ' * (core_width - len(core_plain))}"
        elif len(core_plain) > core_width:
            core_plain = core_plain[:core_width]

        text = Text(f"{' ' * outer_pad}{core_plain}{' ' * outer_pad}")
        left_start = outer_pad
        left_end = left_start + len(left_block)
        sep_start = left_end
        right_start = sep_start + len(sep)
        right_end = right_start + len(right_block)

        text.stylize("grey50", left_start, left_end)
        glyph_idx = left_block.find(glyph)
        if glyph_idx >= 0:
            text.stylize(
                edge_style,
                left_start + glyph_idx,
                left_start + glyph_idx + len(glyph),
            )
        text.stylize("grey35", sep_start, sep_start + len(sep))
        text.stylize(edge_style if now is not None else "dim", right_start, right_end)
        return text

    def _aligned_unreal_cell(self, daily: float | None, unreal: float | None) -> Text:
        if daily is None and unreal is None:
            return Text("warming...", style="dim")

        text = Text("")
        text.append("D ", style="dim")
        if daily is None:
            text.append("warm", style="dim")
        else:
            text.append(f"{float(daily):+,.2f}", style=self._pnl_style(float(daily)))

        text.append(" · ", style="dim")
        text.append("U ", style="dim")
        if unreal is None:
            text.append("warm", style="dim")
        else:
            text.append(f"{float(unreal):+,.2f}", style=self._pnl_style(float(unreal)))
        return text

    def _unreal_texts(self, item: PortfolioItem) -> tuple[Text, Text]:
        contract = item.contract
        if contract.secType not in _SECTION_TYPES:
            return Text(""), Text("")
        official_daily = self._official_daily_value(item)
        official_unreal = self._official_unrealized_value(item)
        return self._aligned_unreal_cell(official_daily, official_unreal), Text("")

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
        now_mono = time.monotonic()
        option_con_id = int(getattr(item.contract, "conId", 0) or 0)

        actionable = _ticker_actionable_price(option_ticker) if option_ticker else None
        if actionable is not None:
            if option_con_id:
                self._derivative_actionable_px_by_con_id[int(option_con_id)] = (
                    float(actionable),
                    float(now_mono),
                )
            return float(actionable), False

        if option_con_id:
            cached = self._derivative_actionable_px_by_con_id.get(int(option_con_id))
            if cached is not None:
                cached_px, cached_mono = cached
                if (float(now_mono) - float(cached_mono)) <= float(
                    self._DERIVATIVE_MARK_STICKY_SEC
                ):
                    return float(cached_px), False
                self._derivative_actionable_px_by_con_id.pop(int(option_con_id), None)

        if item.contract.secType == "FOP":
            display = _option_display_price(item, option_ticker)
            if display is None or display <= 0:
                return None, False
            return float(display), True
        portfolio_mark = _safe_num(getattr(item, "marketPrice", None))

        under_con_id = self._option_underlying_con_id.get(option_con_id)
        under_ticker = self._client.ticker_for_con_id(under_con_id) if under_con_id else None
        under_price = _ticker_price(under_ticker) if under_ticker else None
        model = getattr(option_ticker, "modelGreeks", None) if option_ticker else None
        delta = _safe_num(getattr(model, "delta", None)) if model else None
        gamma = _safe_num(getattr(model, "gamma", None)) if model else None
        under_close = _ticker_close(under_ticker) if under_ticker else None
        if under_close is None and under_con_id:
            cached = self._session_closes_by_con_id.get(under_con_id)
            under_close = cached[0] if cached else None
        ref_price = _safe_num(getattr(option_ticker, "close", None)) if option_ticker else None
        if ref_price is None:
            ref_price = portfolio_mark
        if (
            under_price is not None
            and under_price > 0
            and delta is not None
            and under_close is not None
            and under_close > 0
            and ref_price is not None
        ):
            d_under = float(under_price) - float(under_close)
            estimated = float(ref_price) + (float(delta) * d_under)
            if gamma is not None:
                estimated += 0.5 * float(gamma) * (d_under**2)
            return max(estimated, 0.0), True
        model_price = _safe_num(getattr(model, "optPrice", None)) if model else None
        if model_price is not None and model_price > 0:
            return float(model_price), True
        if ref_price is not None and ref_price > 0:
            return float(ref_price), True
        return None, False

    def _render_ticker_bar(self) -> None:
        prefix = f"MKT:{_market_session_label()} | "
        line0 = _ticker_line(
            _INDEX_FUT_ORDER,
            _INDEX_FUT_LABELS,
            self._index_tickers,
            self._index_error,
            prefix,
            allow_display_fallback=True,
        )
        line1 = _ticker_line(
            _INDEX_ORDER,
            _INDEX_LABELS,
            self._proxy_tickers,
            self._proxy_error,
            " " * len(prefix),
        )
        line2 = _ticker_line(
            _PROXY_ORDER,
            _PROXY_LABELS,
            self._proxy_tickers,
            self._proxy_error,
            " " * len(prefix),
        )
        text = Text()
        text.append_text(line0)
        text.append("\n")
        text.append_text(line1)
        text.append("\n")
        text.append_text(line2)
        self._ticker.update(text)
