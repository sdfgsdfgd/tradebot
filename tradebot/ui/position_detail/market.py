"""Live market state and presentation for a position detail screen."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from ib_insync import PortfolioItem, Ticker
from rich.text import Text

from ...client import IBKRClient
from ...engines.execution import (
    _midpoint,
    _quote_num_actionable,
    _tick_size,
    _ticker_close,
    _ticker_price,
)
from ..common import (
    _cost_basis,
    _fmt_expiry,
    _fmt_money,
    _fmt_qty,
    _fmt_quote,
    _infer_multiplier,
    _market_data_label,
    _mark_price,
    _option_display_price,
    _safe_float,
    _safe_num,
    _quote_status_line,
)
from .frame import box_bottom, box_row, box_top
from .chart import PositionChart


class PositionMarketView:
    """Owns bounded market samples and their position-detail representation."""

    _POSITION_PULSE_SEC = 1.6
    _MD_PROBE_BANNER_TTL_SEC = 10.0
    _DERIVATIVE_ACTIONABLE_STICKY_SEC = 40.0

    def __init__(
        self,
        client: IBKRClient,
        item: PortfolioItem,
        refresh_sec: float,
        *,
        session_prev_close: float | None = None,
        request_render: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client
        self._item = item
        self._ticker: Ticker | None = None
        self._underlying_ticker: Ticker | None = None
        self._underlying_label: str | None = None
        self._request_render = request_render
        self._session_prev_close = session_prev_close
        self.chart = PositionChart(refresh_sec)
        self._pos_prev_qty: float | None = None
        self._pos_delta = 0.0
        self._pos_pulse_until = 0.0
        self._last_tick_signature: tuple[float | None, ...] | None = None
        self._md_probe_requested_type: int | None = None
        self._md_probe_started_mono = 0.0
        self._derivative_actionable_px: float | None = None
        self._derivative_actionable_px_until_mono = 0.0

    def bind(
        self,
        item: PortfolioItem,
        ticker: Ticker | None,
        *,
        underlying_ticker: Ticker | None = None,
        underlying_label: str | None = None,
    ) -> None:
        self._item = item
        self._ticker = ticker
        self._underlying_ticker = underlying_ticker
        self._underlying_label = underlying_label
        self.chart.bind(item.contract, ticker)

    def set_session_close(self, previous: float | None) -> None:
        self._session_prev_close = previous

    def start_probe(self, ticker: Ticker | None) -> None:
        requested = getattr(ticker, "tbRequestedMdType", None) if ticker else None
        try:
            requested = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            requested = None
        self._md_probe_requested_type = (
            requested if requested in (1, 2, 3, 4) else self._md_type_value(ticker)
        )
        self._md_probe_started_mono = monotonic()

    def resolve_derivative_price(
        self,
        display_price: float | None,
        actionable: float | None,
        *,
        now_mono: float | None = None,
    ) -> float | None:
        now = float(now_mono) if now_mono is not None else monotonic()
        if actionable is not None:
            self._derivative_actionable_px = float(actionable)
            self._derivative_actionable_px_until_mono = (
                now + self._DERIVATIVE_ACTIONABLE_STICKY_SEC
            )
        sticky = self.sticky_price(now_mono=now)
        return float(sticky) if actionable is None and sticky is not None else display_price

    @staticmethod
    def _md_type_value(ticker: Ticker | None) -> int | None:
        if ticker is None:
            return None
        raw = getattr(ticker, "marketDataType", None)
        try:
            value = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            value = None
        if value in (1, 2, 3, 4):
            return value
        return None

    @staticmethod
    def _md_type_name(md_type: int | None) -> str:
        if md_type == 1:
            return "Live"
        if md_type == 2:
            return "Live-Frozen"
        if md_type == 3:
            return "Delayed"
        if md_type == 4:
            return "Delayed-Frozen"
        return "n/a"

    def _market_data_probe_row(self) -> Text | None:
        started_mono = float(self._md_probe_started_mono or 0.0)
        if started_mono <= 0:
            return None
        elapsed_sec = max(0.0, monotonic() - started_mono)
        if elapsed_sec > float(self._MD_PROBE_BANNER_TTL_SEC):
            return None
        req_type = self._md_probe_requested_type
        actual_type = self._md_type_value(self._ticker)
        remaining_sec = max(0.0, float(self._MD_PROBE_BANNER_TTL_SEC) - elapsed_sec)
        row = Text("MD Probe ", style="yellow")
        row.append("req ")
        row.append(self._md_type_name(req_type), style="bright_white")
        row.append(f" ({req_type if req_type is not None else 'n/a'})", style="dim")
        row.append(" -> now ")
        actual_style = "green" if req_type is not None and req_type == actual_type else "yellow"
        row.append(self._md_type_name(actual_type), style=actual_style)
        row.append(f" ({actual_type if actual_type is not None else 'n/a'})", style="dim")
        row.append(f"  {remaining_sec:.0f}s", style="dim")
        return row

    def sticky_price(self, *, now_mono: float | None = None) -> float | None:
        px = self._derivative_actionable_px
        if px is None:
            return None
        now = float(now_mono) if now_mono is not None else monotonic()
        if now <= float(self._derivative_actionable_px_until_mono or 0.0):
            return float(px)
        self._derivative_actionable_px = None
        self._derivative_actionable_px_until_mono = 0.0
        return None

    @staticmethod
    def _pnl_style(value: float | None) -> str:
        if value is None:
            return "dim"
        return "green" if value >= 0 else "red"

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        return _safe_float(value)

    def _official_unrealized(self) -> float | None:
        con_id = int(getattr(self._item.contract, "conId", 0) or 0)
        official = self._client.pnl_single_unrealized(con_id)
        if official is None:
            official = self._float_or_none(getattr(self._item, "unrealizedPNL", None))
        return official

    def _official_daily_contract_pnl(self) -> float | None:
        con_id = int(getattr(self._item.contract, "conId", 0) or 0)
        return self._client.pnl_single_daily(con_id)

    def live_unrealized(self, mark_price: float | None) -> float | None:
        mark = self._float_or_none(mark_price)
        if mark is None:
            return None
        qty = self._float_or_none(getattr(self._item, "position", None))
        if qty is None:
            return None
        if abs(float(qty)) <= 1e-12:
            return 0.0
        multiplier = _infer_multiplier(self._item)
        cost_basis = _cost_basis(self._item)
        return (float(mark) * float(qty) * float(multiplier)) - float(cost_basis)

    @staticmethod
    def _direction_glyph(value: float | None) -> Text:
        if value is None:
            return Text("•", style="dim")
        if value > 0:
            return Text("▲", style="bold green")
        if value < 0:
            return Text("▼", style="bold red")
        return Text("•", style="dim")

    def _position_beacon_row(self, qty: float, notional: float | None) -> Text:
        now = monotonic()
        if self._pos_prev_qty is None:
            self._pos_prev_qty = float(qty)
        elif float(qty) != float(self._pos_prev_qty):
            self._pos_delta = float(qty) - float(self._pos_prev_qty)
            self._pos_prev_qty = float(qty)
            self._pos_pulse_until = now + self._POSITION_PULSE_SEC

        direction = "FLAT"
        direction_style = "dim"
        if qty > 0:
            direction = "LONG"
            direction_style = "bold green"
        elif qty < 0:
            direction = "SHORT"
            direction_style = "bold red"

        qty_label = _fmt_qty(float(qty))
        if qty > 0:
            qty_label = f"+{qty_label}"

        row = Text("POS ", style="bold")
        row.append(f"{qty_label} sh")
        if notional is not None:
            signed = float(notional)
            sign = "+" if signed > 0 else "-" if signed < 0 else ""
            row.append(" (", style="dim")
            row.append(f"{sign}${abs(signed):,.0f}")
            row.append(")", style="dim")
        row.append("   ")
        row.append(direction, style=direction_style)

        pulse_active = now < float(self._pos_pulse_until)
        if pulse_active and self._pos_delta:
            delta = self._pos_delta
            delta_label = _fmt_qty(abs(delta))
            sign = "+" if delta > 0 else "-"
            row.append("   Δ ")
            row.append(f"{sign}{delta_label}", style="bold")
            row.append(" sh")
        if pulse_active:
            row.stylize("bold on #2f2f2f")
        return row

    def capture_tick(self) -> None:
        ticker = self._ticker
        if ticker is None:
            return
        contract = self._item.contract
        bid = _quote_num_actionable(getattr(ticker, "bid", None))
        ask = _quote_num_actionable(getattr(ticker, "ask", None))
        last = _quote_num_actionable(getattr(ticker, "last", None))
        last_size = _safe_num(getattr(ticker, "lastSize", None))
        bid_size = _safe_num(getattr(ticker, "bidSize", None))
        ask_size = _safe_num(getattr(ticker, "askSize", None))
        rt_trade_volume = _safe_num(getattr(ticker, "rtTradeVolume", None))
        rt_volume = _safe_num(getattr(ticker, "rtVolume", None))
        total_volume = _safe_num(getattr(ticker, "volume", None))
        signature = (
            bid,
            ask,
            last,
            last_size,
            bid_size,
            ask_size,
            rt_trade_volume,
            rt_volume,
            total_volume,
        )
        if signature == self._last_tick_signature:
            self._request_render()
            return
        self._last_tick_signature = signature

        mid = _midpoint(bid, ask)
        if contract.secType in ("OPT", "FOP"):
            now_mono = monotonic()
            actionable = (
                _option_display_price(self._item, ticker)
                if contract.secType == "FOP"
                else (mid or last)
            )
            if actionable is not None:
                self._derivative_actionable_px = float(actionable)
                self._derivative_actionable_px_until_mono = now_mono + float(
                    self._DERIVATIVE_ACTIONABLE_STICKY_SEC
                )
            if mid is None:
                mid = self.sticky_price(now_mono=now_mono)
        if mid is None:
            mid = _ticker_price(ticker)
        if mid is None:
            mid = _mark_price(self._item)
        if mid is None:
            return
        now = monotonic()
        prev_mid = float(self.chart.mid_tape[-1][1]) if self.chart.mid_tape else None
        mid_value = float(mid)
        tick = _tick_size(contract, ticker, mid_value)
        epsilon = max(float(tick) * 0.01, 1e-9)
        self.chart.record_mid(mid_value, epsilon=epsilon, ts=now)
        self.chart.record_size(last_size, ts=now)

        imbalance = None
        total_size = float((bid_size or 0.0) + (ask_size or 0.0))
        if bid_size is not None or ask_size is not None:
            imbalance = (
                float((bid_size or 0.0) - (ask_size or 0.0)) / total_size if total_size > 0 else 0.0
            )
        vol_burst = abs(mid_value - prev_mid) if prev_mid is not None else 0.0
        self.chart.record_aurora(imbalance=imbalance, vol_burst=vol_burst, ts=now)

        flow_qty = self.chart.cumulative_volume_delta(
            {
                "rtTradeVolume": rt_trade_volume,
                "rtVolume": rt_volume,
                "volume": total_volume,
            }
        )
        if flow_qty is None:
            flow_qty = self.chart.fallback_volume_delta(last_size=last_size, last_price=last)
        if flow_qty is not None and flow_qty > 0:
            direction = self.chart.flow_direction(
                price=last or mid_value,
                imbalance=imbalance,
                epsilon=epsilon,
            )
            self.chart.record_flow(flow_qty * direction, ts=now)
        self._request_render()

    def record(
        self,
        *,
        mid: float | None,
        spread: float | None,
        size: float | None,
        pnl: float | None,
        slip_proxy: float | None,
        imbalance: float | None,
        vol_burst: float | None,
    ) -> None:
        now = monotonic()
        tick = _tick_size(self._item.contract, self._ticker, mid)
        self.chart.record_market(
            mid=mid,
            mid_epsilon=max(float(tick) * 0.01, 1e-9),
            spread=spread,
            size=size,
            pnl=pnl,
            slip_proxy=slip_proxy,
            imbalance=imbalance,
            vol_burst=vol_burst,
            ts=now,
        )
















    @staticmethod
    def contract_header_title(contract: object) -> str:
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper() or "?"
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        if sec_type == "STK":
            kind = "STOCK"
        elif sec_type == "FUT":
            kind = "FUTURES"
        elif sec_type == "OPT":
            kind = "OPTIONS"
        elif sec_type == "FOP":
            kind = "FOP"
        else:
            kind = sec_type or "UNKNOWN"
        side = ""
        if sec_type in ("OPT", "FOP"):
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            if right == "C":
                side = " CALLS"
            elif right == "P":
                side = " PUTS"
        return f"{symbol} {kind}{side}"

    def render_hud(
        self,
        *,
        panel_width: int,
        bid: float | None,
        ask: float | None,
        last: float | None,
        price: float | None,
        mid: float | None,
        close: float | None,
        mark: float | None,
        spread: float | None,
    ) -> list[Text]:
        contract = self._item.contract
        inner = max(panel_width - 2, 24)
        spark_width = inner
        official_unreal = self._official_unrealized()
        fast_unreal = self.live_unrealized(price or mark)
        day_pnl = self._official_daily_contract_pnl()
        day_label = _fmt_money(day_pnl) if day_pnl is not None else "n/a"
        position_qty = float(self._item.position or 0.0)
        avg_cost = (
            _fmt_money(float(self._item.averageCost))
            if self._item.averageCost is not None
            else "n/a"
        )
        market_value_raw = _safe_num(self._item.marketValue)
        market_value = (
            _fmt_money(float(market_value_raw))
            if market_value_raw is not None
            else "n/a"
        )
        realized_num = self._float_or_none(getattr(self._item, "realizedPNL", None))
        realized = _fmt_money(realized_num) if realized_num is not None else "n/a"

        md_row = Text("MD: ")
        if self._ticker:
            md_exchange = getattr(self._ticker.contract, "exchange", "") or "n/a"
            md_label = _market_data_label(self._ticker)
            md_row.append(f"{md_exchange} ({md_label})", style="bright_cyan")
            req_type_raw = getattr(self._ticker, "tbRequestedMdType", None)
            try:
                req_type = int(req_type_raw) if req_type_raw is not None else None
            except (TypeError, ValueError):
                req_type = None
            if req_type in (1, 2, 3, 4):
                md_row.append(" req ", style="dim")
                md_row.append(self._md_type_name(req_type), style="dim")
        else:
            md_row.append("n/a", style="dim")
        quote_status = (
            _quote_status_line(self._ticker)
            if self._ticker
            else Text("MD Quotes: n/a", style="dim")
        )
        has_live_quote_now = bool(
            (bid is not None and ask is not None and bid <= ask)
            or (last is not None)
        )
        sticky_actionable = (
            self.sticky_price()
            if contract.secType in ("OPT", "FOP") and (not has_live_quote_now)
            else None
        )
        has_live_quote_effective = has_live_quote_now or (sticky_actionable is not None)
        held_quote_row: Text | None = None
        if sticky_actionable is not None and (not has_live_quote_now):
            remaining = max(
                0.0,
                float(self._derivative_actionable_px_until_mono or 0.0) - float(monotonic()),
            )
            held_quote_row = Text(
                f"NBBO GAP · holding {_fmt_quote(sticky_actionable)} ({remaining:.0f}s)",
                style="dim",
            )
        close_only_badge_row: Text | None = None
        if self._ticker:
            md_type = getattr(self._ticker, "marketDataType", None)
            is_delayed = md_type in (3, 4)
            if is_delayed and (not has_live_quote_now) and close is not None and close > 0:
                close_only_badge_row = Text("CLOSE-ONLY DELAYED FEED", style="bold black on yellow")
        no_quote_badge_row: Text | None = None
        if contract.secType in ("OPT", "FOP") and not has_live_quote_effective:
            no_quote_badge_row = Text("NO ACTIONABLE OPTION QUOTE YET", style="bold black on yellow")

        quote_row = Text("Bid ")
        quote_row.append(_fmt_quote(bid), style="green")
        quote_row.append("  Ask ")
        quote_row.append(_fmt_quote(ask), style="red")
        quote_row.append("  Last ")
        quote_row.append(_fmt_quote(last), style="bright_white")

        price_row = Text("Price ")
        if price is None and close is not None:
            price_row.append(f"Closed ({_fmt_quote(close)})", style="red")
        elif price is None and mark is not None:
            price_row.append(f"Mark ({_fmt_quote(mark)})", style="yellow")
        else:
            price_row.append(_fmt_quote(price), style="bright_white")

        headline = Text("MID ")
        headline.append(_fmt_quote(mid or price or mark), style="bright_cyan")
        headline.append("   SPRD ")
        headline.append(_fmt_quote(spread), style="cyan")
        position_row = self._position_beacon_row(position_qty, market_value_raw)

        trend_window_start, trend_window_end = self.chart.window_bounds()
        aurora_label_row = Text("Aurora", style="#8aa0b6")
        aurora_row = self.chart.mark_now(
            self.chart.aurora_strip(
                spark_width,
                window_start=trend_window_start,
                window_end=trend_window_end,
            )
        )
        trend_label_row = Text("1m Trend", style="cyan")
        trend_row_values, trend_now_row = self.chart.trend_rows(
            spark_width,
            window_start=trend_window_start,
            window_end=trend_window_end,
        )
        comet_color = self.chart.aurora_now_style()
        trend_rows = [Text(row, style="#63d9ff") for row in trend_row_values]
        trend_price = mid or price or mark
        trend_rows[trend_now_row] = self.chart.tag_price(
            trend_rows[trend_now_row], trend_price, color=comet_color
        )
        vol_label_row = Text("Vol Histogram", style="magenta")
        vol_row = self.chart.mark_now(
            self.chart.volume_histogram(
                spark_width,
                window_start=trend_window_start,
                window_end=trend_window_end,
            )
        )
        momentum_label_row = Text("Momentum", style="yellow")
        momentum_row = self.chart.mark_now(Text(self.chart.momentum(spark_width), style="yellow"))

        detail_row = Text(f"Avg {avg_cost}   MktVal {market_value}")
        ref_price = mid or price or mark
        pct_baseline = close
        if (
            contract.secType in ("OPT", "FOP")
            and not has_live_quote_effective
            and self._session_prev_close is not None
            and self._session_prev_close > 0
        ):
            pct_baseline = float(self._session_prev_close)
        pct24 = (
            ((float(ref_price) - float(pct_baseline)) / float(pct_baseline) * 100.0)
            if ref_price is not None and pct_baseline is not None and pct_baseline > 0
            else None
        )
        if pct24 is not None and position_qty < 0:
            pct24 *= -1.0
        pct24_prefix = self._direction_glyph(pct24)
        pct24_value = Text(" n/a", style="dim")
        if pct24 is not None:
            pct24_value = Text(f" {pct24:.2f}%")
            if pct24 > 0:
                pct24_value.stylize("green")
            elif pct24 < 0:
                pct24_value.stylize("red")
        tail_row = Text("")
        tail_row.append_text(pct24_prefix)
        tail_row.append_text(pct24_value)
        tail_row.append("   ")
        tail_row.append("✦ Unreal ", style="#8fbfff")
        official_start = -1
        official_end = -1
        estimate_start = -1
        estimate_end = -1
        if official_unreal is not None:
            official_start = len(tail_row.plain)
            tail_row.append(_fmt_money(official_unreal))
            official_end = len(tail_row.plain)
            if fast_unreal is not None:
                tail_row.append(" (", style="dim")
                estimate_start = len(tail_row.plain)
                tail_row.append(_fmt_money(fast_unreal))
                estimate_end = len(tail_row.plain)
                tail_row.append(")", style="dim")
        elif fast_unreal is not None:
            estimate_start = len(tail_row.plain)
            tail_row.append(_fmt_money(fast_unreal))
            estimate_end = len(tail_row.plain)
            tail_row.append(" ≈est", style="dim")
        else:
            tail_row.append("n/a", style="dim")
        tail_row.append(" ", style="dim")
        tail_row.append("(", style="dim")
        tail_row.append("◷ Day ", style="#8aa0b6")
        day_start = len(tail_row.plain)
        tail_row.append(day_label)
        day_end = len(tail_row.plain)
        tail_row.append(")", style="dim")
        tail_row.append("   ")
        tail_row.append("Realized ", style="#8aa0b6")
        realized_start = len(tail_row.plain)
        tail_row.append(realized)
        realized_end = len(tail_row.plain)
        if official_start >= 0 and official_end > official_start:
            tail_row.stylize(self._pnl_style(official_unreal), official_start, official_end)
        if estimate_start >= 0 and estimate_end > estimate_start:
            tail_row.stylize(self._pnl_style(fast_unreal), estimate_start, estimate_end)
        tail_row.stylize(self._pnl_style(day_pnl), day_start, day_end)
        tail_row.stylize(self._pnl_style(realized_num), realized_start, realized_end)

        lines: list[Text] = [
            box_top(self.contract_header_title(contract), inner, style="#2d8fd5"),
            box_row(md_row, inner, style="#2d8fd5"),
            box_row(quote_status, inner, style="#2d8fd5"),
            box_row(headline, inner, style="#2d8fd5"),
            box_row(position_row, inner, style="#2d8fd5"),
            box_row(detail_row, inner, style="#2d8fd5"),
            box_row(tail_row, inner, style="#2d8fd5"),
            box_row(quote_row, inner, style="#2d8fd5"),
            box_row(price_row, inner, style="#2d8fd5"),
            box_row(aurora_label_row, inner, style="#2d8fd5"),
            box_row(aurora_row, inner, style="#2d8fd5"),
            box_row(trend_label_row, inner, style="#2d8fd5"),
            *[box_row(trend_row, inner, style="#2d8fd5") for trend_row in trend_rows],
            box_row(vol_label_row, inner, style="#2d8fd5"),
            box_row(vol_row, inner, style="#2d8fd5"),
            box_row(momentum_label_row, inner, style="#2d8fd5"),
            box_row(momentum_row, inner, style="#2d8fd5"),
        ]
        md_probe_row = self._market_data_probe_row()
        badge_insert_idx = 3
        if md_probe_row is not None:
            lines.insert(badge_insert_idx, box_row(md_probe_row, inner, style="#2d8fd5"))
            badge_insert_idx += 1
        for badge_row in (held_quote_row, no_quote_badge_row, close_only_badge_row):
            if badge_row is None:
                continue
            lines.insert(badge_insert_idx, box_row(badge_row, inner, style="#2d8fd5"))
            badge_insert_idx += 1
        if contract.lastTradeDateOrContractMonth:
            expiry = _fmt_expiry(contract.lastTradeDateOrContractMonth)
            meta = Text(f"Expiry {expiry}")
            if contract.right:
                meta.append(f"  Right {contract.right}")
            if contract.strike:
                meta.append(f"  Strike {_fmt_money(contract.strike)}")
            lines.append(box_row(meta, inner, style="#2d8fd5"))
        if self._underlying_ticker:
            label = self._underlying_label or "Underlying"
            ubid = _quote_num_actionable(self._underlying_ticker.bid)
            uask = _quote_num_actionable(self._underlying_ticker.ask)
            ulast = _quote_num_actionable(self._underlying_ticker.last)
            if ulast is None:
                ulast = _ticker_price(self._underlying_ticker) or _ticker_close(self._underlying_ticker)
            under_row = Text(f"{label}: ")
            under_row.append(f"{_fmt_quote(ubid)}/{_fmt_quote(uask)}/{_fmt_quote(ulast)}", style="cyan")
            lines.append(box_row(under_row, inner, style="#2d8fd5"))
        lines.append(box_bottom(inner, style="#2d8fd5"))
        return lines
