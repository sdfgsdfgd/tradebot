"""Order and execution presentation for a position detail screen."""

from __future__ import annotations

from collections.abc import Callable
import textwrap
from time import monotonic

from ib_insync import PortfolioItem, Ticker, Trade
from rich.text import Text

from ...client import IBKRClient
from ...engines.execution import (
    ExecutionPolicy,
    _aggressive_price,
    execution_mode_label,
    _midpoint,
    _optimistic_price,
    _quote_num_actionable,
    _round_to_tick,
    _tick_decimals,
    _tick_size,
)
from ...live.execution import LiveOrderExecution, order_ids
from ..common import (
    _fmt_qty,
    _fmt_quote,
    _mark_price,
    _option_display_price,
    _safe_num,
    _trade_sort_key,
)
from ..time_compat import now_et as _now_et
from .chart import PositionChart
from .frame import box_bottom, box_row, box_rule, box_top, clip, meter


class PositionOrderView:
    """Owns order-panel state and its two terminal representations."""

    _ORDER_PANEL_NOTICE_TTL_SEC = 5 * 60.0

    def __init__(
        self,
        client: IBKRClient,
        execution: LiveOrderExecution,
        policy: Callable[[], ExecutionPolicy],
        chart: PositionChart,
        item: PortfolioItem,
        default_qty: int,
        *,
        initial_price: Callable[..., float | None],
        mode_price: Callable[..., float | None],
    ) -> None:
        self._client = client
        self._execution = execution
        self._policy_source = policy
        self._chart = chart
        self._item = item
        self._ticker: Ticker | None = None
        self._underlying_con_id: int | None = None
        self._initial_price = initial_price
        self._mode_price = mode_price
        self.exec_rows = [
            "ladder",
            "relentless",
            "relentless_delay",
            "optimistic",
            "mid",
            "aggressive",
            "cross",
            "custom",
            "qty",
        ]
        self.exec_selected = 0
        self.custom_input = ""
        self.custom_price: float | None = None
        self.qty_input = ""
        self.qty = int(default_qty)
        self.status: str | None = None
        self.active_panel = "exec"
        self.selected = 0
        self.scroll = 0
        self.rows: list[Trade] = []
        self._notice: tuple[float, str, str] | None = None

    @property
    def _policy(self) -> ExecutionPolicy:
        return self._policy_source()

    def bind(
        self,
        item: PortfolioItem,
        ticker: Ticker | None,
        *,
        underlying_con_id: int | None = None,
    ) -> None:
        self._item = item
        self._ticker = ticker
        self._underlying_con_id = underlying_con_id

    def set_notice(self, message: str, *, level: str = "info") -> None:
        text = str(message or "").strip()
        if not text:
            return
        cleaned_level = str(level or "info").strip().lower()
        if cleaned_level not in ("info", "warn", "error"):
            cleaned_level = "info"
        self._notice = (monotonic(), cleaned_level, text)

    def _notice_line(self) -> Text | None:
        payload = self._notice
        if payload is None:
            return None
        ts_mono, level, message = payload
        if (monotonic() - float(ts_mono)) > float(self._ORDER_PANEL_NOTICE_TTL_SEC):
            self._notice = None
            return None
        style = "bright_cyan"
        label = "INFO"
        if level == "warn":
            style = "yellow"
            label = "WARN"
        elif level == "error":
            style = "bold red"
            label = "ERR"
        line = Text(f"{label} ", style=style)
        line.append(str(message), style=style)
        return line

    @staticmethod
    def _exec_mode_label(mode: str) -> str:
        return execution_mode_label(mode)

    def selected_mode(self) -> str:
        selected = self.exec_rows[self.exec_selected]
        if selected == "relentless":
            return "RELENTLESS"
        if selected == "relentless_delay":
            return "RELENTLESS_DELAY"
        if selected == "optimistic":
            return "OPTIMISTIC"
        if selected == "mid":
            return "MID"
        if selected == "aggressive":
            return "AGGRESSIVE"
        if selected == "cross":
            return "CROSS"
        if selected == "custom":
            return "CUSTOM"
        return "AUTO"

    def render_execution(self, *, panel_width: int) -> list[Text]:
        contract = self._item.contract
        if contract.secType not in ("STK", "OPT", "FOP", "FUT"):
            return []
        bid = _quote_num_actionable(self._ticker.bid) if self._ticker else None
        ask = _quote_num_actionable(self._ticker.ask) if self._ticker else None
        last = _quote_num_actionable(self._ticker.last) if self._ticker else None
        mark = _option_display_price(self._item, self._ticker) if contract.secType in ("OPT", "FOP") else _mark_price(self._item)
        if contract.secType == "FOP":
            last_ref = (
                mark
                if mark is not None
                else (last if last is not None else (bid if bid is not None else ask))
            )
        else:
            last_ref = last if last is not None else (bid if bid is not None else (ask if ask is not None else mark))
        tick = _tick_size(contract, self._ticker, last_ref)
        mid_raw = _midpoint(bid, ask)
        fallback = _round_to_tick(last_ref, tick)
        mid = _round_to_tick(mid_raw, tick) or fallback
        optimistic_buy = _round_to_tick(_optimistic_price(bid, ask, mid_raw, "BUY"), tick) or mid
        optimistic_sell = _round_to_tick(_optimistic_price(bid, ask, mid_raw, "SELL"), tick) or mid
        aggressive_buy = _round_to_tick(_aggressive_price(bid, ask, mid_raw, "BUY"), tick) or mid
        aggressive_sell = _round_to_tick(_aggressive_price(bid, ask, mid_raw, "SELL"), tick) or mid
        cross_buy = _round_to_tick(ask, tick) if ask is not None else None
        cross_sell = _round_to_tick(bid, tick) if bid is not None else None
        custom = _round_to_tick(self.custom_price, tick)
        if custom is None:
            custom = _round_to_tick(mid, tick)
        ask_size = _safe_num(getattr(self._ticker, "askSize", None)) if self._ticker else None
        bid_size = _safe_num(getattr(self._ticker, "bidSize", None)) if self._ticker else None
        size_scale = max(1.0, ask_size or 0.0, bid_size or 0.0)
        qty = self.qty
        inner = max(panel_width - 2, 24)
        depth_width = max(min(inner - 26, 18), 8)
        lines: list[Text] = [box_top("Execution Ladder", inner, style="#d4922f")]
        if self.status:
            lines.append(box_row(Text(self.status, style="yellow"), inner, style="#d4922f"))
        has_actionable_quote = bool(
            (bid is not None and ask is not None and bid <= ask)
            or (last is not None)
        )
        quote_stale = self._policy.quote_is_stale(
            ticker=self._ticker,
            bid=bid,
            ask=ask,
            last=last,
        )
        open_shock = self._policy.in_open_shock(_now_et().time())
        relentless_buy = self._policy.relentless_price(
            action="BUY",
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
            recent_spreads=self._chart.spread_samples,
        )
        relentless_sell = self._policy.relentless_price(
            action="SELL",
            bid=bid,
            ask=ask,
            last_ref=last_ref,
            tick=tick,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
            recent_spreads=self._chart.spread_samples,
        )
        relentless_delay_buy = self._mode_price(
            "RELENTLESS_DELAY",
            "BUY",
            bid=bid,
            ask=ask,
            last=last,
            ticker=self._ticker,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
        )
        relentless_delay_sell = self._mode_price(
            "RELENTLESS_DELAY",
            "SELL",
            bid=bid,
            ask=ask,
            last=last,
            ticker=self._ticker,
            elapsed_sec=0.0,
            quote_stale=quote_stale,
            open_shock=open_shock,
            no_progress_reprices=0,
            arrival_ref=mid_raw or last_ref,
        )
        if contract.secType == "OPT" and not has_actionable_quote:
            lock_row = Text(
                "B/S locked: no actionable option quote yet (waiting for bid/ask/last)",
                style="bold black on yellow",
            )
            lines.append(box_row(lock_row, inner, style="#d4922f"))

        lines.append(box_rule("Depth", inner, style="#d4922f"))
        mid_size = ((ask_size or 0.0) + (bid_size or 0.0)) * 0.5
        ask_depth = meter((ask_size or 0.0) / size_scale, depth_width)
        mid_depth = meter(mid_size / size_scale, depth_width)
        bid_depth = meter((bid_size or 0.0) / size_scale, depth_width)
        lines.append(
            box_row(
                Text(
                    f"{_fmt_quote(cross_buy)} ask {_fmt_qty(ask_size or 0.0):>4} |{ask_depth}|",
                    style="red",
                ),
                inner,
                style="#d4922f",
            )
        )
        lines.append(
            box_row(
                Text(
                    f"{_fmt_quote(mid)} mid {_fmt_qty(mid_size):>4} |{mid_depth}|",
                    style="bright_white",
                ),
                inner,
                style="#d4922f",
            )
        )
        lines.append(
            box_row(
                Text(
                    f"{_fmt_quote(cross_sell)} bid {_fmt_qty(bid_size or 0.0):>4} |{bid_depth}|",
                    style="green",
                ),
                inner,
                style="#d4922f",
            )
        )

        lines.append(
            box_rule(
                f"Controls · tick {tick:.{_tick_decimals(tick)}f} · qty {qty}",
                inner,
                style="#d4922f",
            )
        )
        rows = [
            (
                "ladder",
                (
                    "Auto Ladder "
                    f"(OPT {_fmt_quote(optimistic_buy)}/{_fmt_quote(optimistic_sell)} "
                    f"-> MID {_fmt_quote(mid)} "
                    f"-> AGG {_fmt_quote(aggressive_buy)}/{_fmt_quote(aggressive_sell)} "
                    f"-> CROSS {_fmt_quote(cross_buy)}/{_fmt_quote(cross_sell)})"
                ),
            ),
            (
                "relentless",
                f"RLT       B/S {_fmt_quote(relentless_buy)} / {_fmt_quote(relentless_sell)}",
            ),
            (
                "relentless_delay",
                f"RLT ⚔ Delay B/S {_fmt_quote(relentless_delay_buy)} / {_fmt_quote(relentless_delay_sell)}",
            ),
            (
                "optimistic",
                f"OPT Only   B/S {_fmt_quote(optimistic_buy)} / {_fmt_quote(optimistic_sell)}",
            ),
            ("mid", f"MID Only   B/S {_fmt_quote(mid)} / {_fmt_quote(mid)}"),
            (
                "aggressive",
                f"AGG Only   B/S {_fmt_quote(aggressive_buy)} / {_fmt_quote(aggressive_sell)}",
            ),
            (
                "cross",
                f"CROSS Only B/S {_fmt_quote(cross_buy)} / {_fmt_quote(cross_sell)}",
            ),
            (
                "custom",
                f"CUSTOM     B/S {_fmt_quote(custom)} / {_fmt_quote(custom)}",
            ),
            (
                "qty",
                f"Qty {qty}",
            ),
        ]
        for idx, (key, label) in enumerate(rows):
            row = Text(label)
            if self.active_panel == "exec" and idx == self.exec_selected:
                row.stylize("bold on #2b2b2b")
            lines.append(box_row(row, inner, style="#d4922f"))
        lines.append(box_bottom(inner, style="#d4922f"))
        return lines

    def render_orders(self, *, panel_width: int, available: int = 0) -> Text:
        con_ids: list[int] = []
        con_id = int(self._item.contract.conId or 0)
        if con_id:
            con_ids.append(con_id)
        if self._underlying_con_id and self._underlying_con_id != con_id:
            con_ids.append(self._underlying_con_id)
        trades = self._client.open_trades_for_conids(con_ids)
        trades.sort(key=_trade_sort_key, reverse=True)
        self.rows = trades
        inner = max(panel_width - 2, 24)

        total_qty = sum(abs(float(trade.order.totalQuantity or 0.0)) for trade in trades)
        filled_qty = sum(abs(float(trade.orderStatus.filled or 0.0)) for trade in trades)
        fill_rate = (filled_qty / total_qty) if total_qty > 0 else 0.0
        cancel_like = 0
        replace_like = 0
        for trade in trades:
            status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").lower()
            if "cancel" in status:
                cancel_like += 1
            elif "pending" in status:
                replace_like += 1
        cancel_replace_rate = (
            float(cancel_like + replace_like) / float(len(trades))
            if trades
            else 0.0
        )
        fill_meter = meter(fill_rate, 8)
        cancel_meter = meter(cancel_replace_rate, 8)
        slip_spark = self._chart.sparkline(
            self._chart.slip_proxy_samples,
            max(min(inner - 21, 14), 8),
        )

        lines: list[Text] = [box_top("Orders", inner, style="#2f78c4")]
        metrics_row = Text("Fill Rate ")
        metrics_row.append(fill_meter, style="green")
        metrics_row.append(f" {int(round(fill_rate * 100)):>3}%")
        metrics_row.append("   Cancel/Replace ")
        metrics_row.append(cancel_meter, style="yellow")
        metrics_row.append(f" {int(round(cancel_replace_rate * 100)):>3}%")
        lines.append(box_row(metrics_row, inner, style="#2f78c4"))
        slip_row = Text(f"Slippage Dist {slip_spark}")
        slip_row.stylize("magenta")
        lines.append(box_row(slip_row, inner, style="#2f78c4"))
        lines.append(box_row(self._armed_mode_line(), inner, style="#2f78c4"))
        lines.append(box_row(self._active_chase_line(trades), inner, style="#2f78c4"))
        notice_line = self._notice_line()
        notice_reserved_rows = 0
        if notice_line is not None:
            lines.append(box_rule("Order Feed", inner, style="#2f78c4"))
            notice_plain = notice_line.plain
            prefix, sep, message = notice_plain.partition(" ")
            if sep:
                prefix_with_space = f"{prefix}{sep}"
                wrapped_notice = textwrap.wrap(
                    message,
                    width=max(inner, 1),
                    initial_indent=prefix_with_space,
                    subsequent_indent=" " * len(prefix_with_space),
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            else:
                wrapped_notice = textwrap.wrap(
                    notice_plain,
                    width=max(inner, 1),
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            if not wrapped_notice:
                wrapped_notice = [""]
            notice_style = notice_line.style
            for chunk in wrapped_notice:
                lines.append(box_row(Text(chunk, style=notice_style), inner, style="#2f78c4"))
            notice_reserved_rows = 1 + len(wrapped_notice)

        if not trades:
            self.selected = 0
            self.scroll = 0
            lines.append(box_row(Text("No open orders", style="dim"), inner, style="#2f78c4"))
        else:
            if self.selected >= len(trades):
                self.selected = len(trades) - 1
            reserved_without_trades = 7 + notice_reserved_rows
            visible = len(trades)
            if available:
                visible = max(available - reserved_without_trades, 1)
            max_scroll = max(len(trades) - visible, 0)
            self.scroll = min(max(self.scroll, 0), max_scroll)
            if self.selected < self.scroll:
                self.scroll = self.selected
            elif self.selected >= self.scroll + visible:
                self.scroll = self.selected - visible + 1
            header = Text("Label      Stat      Exec      S Qty Type@Price   Fill/Rem  Id", style="dim")
            lines.append(box_row(header, inner, style="#2f78c4"))
            start = self.scroll
            end = min(start + visible, len(trades))
            visible_trades = list(trades[start:end])
            state_snapshot = self._order_state_snapshot_for_trades(visible_trades)
            for idx in range(start, end):
                trade = trades[idx]
                line = self._format_order_line(
                    trade,
                    width=inner,
                    state_snapshot=state_snapshot,
                )
                if self.active_panel == "orders" and idx == self.selected:
                    line.stylize("bold on #2b2b2b")
                lines.append(box_row(line, inner, style="#2f78c4"))
            hidden = len(trades) - end
            if hidden > 0:
                lines.append(
                    box_row(
                        Text(f"... {hidden} more (j/k to scroll)", style="dim"),
                        inner,
                        style="#2f78c4",
                    )
                )

        lines.append(box_bottom(inner, style="#2f78c4"))
        return Text("\n").join(lines)

    @staticmethod
    def _status_compact(status: str) -> str:
        text = str(status or "").strip()
        if not text:
            return "n/a"
        return text.replace("PreSubmitted", "PreSub")[:9]

    def _armed_mode_line(self) -> Text:
        selected = self.selected_mode()
        selected_label = self._exec_mode_label(selected)
        buy = self._initial_price("BUY", mode=selected)
        sell = self._initial_price("SELL", mode=selected)
        line = Text("Armed ")
        line.append(selected_label, style="yellow")
        if selected == "AUTO":
            line.append(" (OPT->MID->AGG->CROSS)", style="dim")
        elif selected == "RELENTLESS":
            line.append(" (completion-first)", style="dim")
        elif selected == "RELENTLESS_DELAY":
            line.append(" (delay-aware offense)", style="dim")
        line.append("  B/S ")
        line.append(f"{_fmt_quote(buy)}/{_fmt_quote(sell)}", style="bright_white")
        return line

    def _active_chase_line(self, trades: list[Trade]) -> Text:
        for trade in trades:
            order_id, perm_id = order_ids(trade)
            display_id = order_id or perm_id
            state = self._execution.state(order_id=order_id, perm_id=perm_id)
            if not isinstance(state, dict):
                continue
            selected = str(state.get("selected") or "-")
            active = str(state.get("active") or "-")
            target = _safe_num(state.get("target_price"))
            try:
                mods = int(state.get("mods") or 0)
            except (TypeError, ValueError):
                mods = 0
            line = Text(f"Chase #{display_id} ")
            line.append(selected, style="yellow")
            if selected == "AUTO":
                line.append("->", style="dim")
                line.append(active, style="yellow")
            line.append(" @ ")
            line.append(_fmt_quote(target), style="bright_white")
            line.append(f"  mods {mods}", style="dim")
            return line
        return Text("Chase idle", style="dim")

    def _order_mode_for_trade(self, trade: Trade) -> str:
        order_id, perm_id = order_ids(trade)
        state = self._execution.state(order_id=order_id, perm_id=perm_id)
        if not isinstance(state, dict):
            return "-"
        selected = str(state.get("selected") or "-")
        active = str(state.get("active") or selected or "-")
        if selected == "AUTO":
            return f"A>{active}"[:9]
        if active and active != selected:
            return f"{selected}>{active}"[:9]
        return selected[:9]

    @staticmethod
    def _should_probe_effective_status(raw_status: str) -> bool:
        status = str(raw_status or "").strip()
        return status in ("", "PendingSubmit", "PendingSubmission", "ApiPending")

    def _order_state_snapshot_for_trades(
        self,
        trades: list[Trade],
    ) -> dict[int, dict[str, object]]:
        snapshot: dict[int, dict[str, object]] = {}
        for trade in trades:
            order_id, perm_id = order_ids(trade)
            keys = self._execution.keys(order_id=order_id, perm_id=perm_id)
            if not keys:
                continue
            if all(int(key) in snapshot for key in keys):
                continue
            raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
            if not self._should_probe_effective_status(raw_status):
                continue
            payload = self._execution.current_order_state(order_id=order_id, perm_id=perm_id)
            if not isinstance(payload, dict):
                continue
            for key in keys:
                snapshot[int(key)] = payload
        return snapshot

    def _effective_status_for_trade(
        self,
        trade: Trade,
        *,
        state_snapshot: dict[int, dict[str, object]] | None = None,
    ) -> str | None:
        order_id, perm_id = order_ids(trade)
        raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
        state = self._execution.state(order_id=order_id, perm_id=perm_id)
        if isinstance(state, dict):
            effective = str(state.get("effective_status") or "").strip()
            if effective and effective != raw_status:
                return effective
        payload = None
        if isinstance(state_snapshot, dict):
            for key in self._execution.keys(order_id=order_id, perm_id=perm_id):
                candidate = state_snapshot.get(int(key))
                if isinstance(candidate, dict):
                    payload = candidate
                    break
            if payload is None and not self._should_probe_effective_status(raw_status):
                return None
        if payload is None:
            payload = self._execution.current_order_state(order_id=order_id, perm_id=perm_id)
        if isinstance(payload, dict):
            effective = str(payload.get("effective_status") or "").strip()
            if effective and effective != raw_status:
                return effective
        return None

    def _format_order_line(
        self,
        trade: Trade,
        *,
        width: int,
        state_snapshot: dict[int, dict[str, object]] | None = None,
    ) -> Text:
        contract = trade.contract
        label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or "?"
        label = label[:10]
        raw_status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "").strip()
        status = self._status_compact(raw_status)
        mode = self._order_mode_for_trade(trade)
        side = (trade.order.action or "?")[:1].upper()
        qty = _fmt_qty(float(trade.order.totalQuantity or 0))
        order_type = trade.order.orderType or ""
        price = self._order_price(trade)
        if price is not None:
            type_label = f"{order_type}@{_fmt_quote(price)}"
        else:
            type_label = order_type
        type_label = type_label[:12]
        filled = _fmt_qty(float(trade.orderStatus.filled or 0))
        remaining = _fmt_qty(float(trade.orderStatus.remaining or 0))
        order_id = trade.order.orderId or trade.order.permId or 0
        effective_status = self._effective_status_for_trade(trade, state_snapshot=state_snapshot)
        effective_hint = ""
        if effective_status:
            effective_hint = f" ~{self._status_compact(effective_status)}"
        line = (
            f"{label:<10} {status:<9} {mode:<9} {side:<1} {qty:>3} "
            f"{type_label:<12} {filled:>4}/{remaining:<4} #{order_id}{effective_hint}"
        )
        return Text(clip(line, width))

    def _order_price(self, trade: Trade) -> float | None:
        order = trade.order
        price = _safe_num(getattr(order, "lmtPrice", None))
        if price is not None:
            return price
        return _safe_num(getattr(order, "auxPrice", None))

    def selected_order(self) -> Trade | None:
        if not self.rows:
            return None
        idx = min(max(self.selected, 0), len(self.rows) - 1)
        return self.rows[idx]
