"""Defined-risk option-package details, broker preview, and guarded submission."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from time import monotonic

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ...live.options import (
    LiveOptionPackageDraft,
    LiveOptionPackageQuote,
    preview_and_admit_option_order,
    quote_live_option_package,
)
from ...order_admission import OrderAdmissionDecision
from .frame import box_bottom, box_row, box_rule, box_top, pane_width


class OptionPackageDetailScreen(Screen):
    """One package surface: live leg quotes, exact payoff, what-if, then send."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("b", "app.pop_screen", "Back"),
        ("q", "app.pop_screen", "Back"),
        ("p", "preview", "What-if"),
        ("r", "refresh", "Refresh"),
        ("left", "mode_left", "Safer"),
        ("right", "mode_right", "Faster"),
        ("up", "quantity_up", "Qty +"),
        ("down", "quantity_down", "Qty -"),
    ]
    _MODES = ("OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS")
    _PREVIEW_TTL_SEC = 30.0

    def __init__(
        self,
        client: object,
        draft: LiveOptionPackageDraft,
        refresh_sec: float,
    ) -> None:
        super().__init__()
        self._client = client
        self._draft = draft
        self._refresh_sec = max(0.1, float(refresh_sec))
        self._tickers: dict[int, object] = {}
        self._mode_index = 1
        self._quantity = 1
        self._quote: LiveOptionPackageQuote | None = None
        self._preview_quote: LiveOptionPackageQuote | None = None
        self._preview: OrderAdmissionDecision | None = None
        self._preview_at: float | None = None
        self._status = "Loading live leg quotes…"
        self._broker_task: asyncio.Task | None = None
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Static("", id="detail-left"),
            Static("", id="detail-right"),
            id="detail-body",
        )
        yield Static(
            "Space-style package composed on Home • P what-if • Shift+B send admitted preview",
            id="detail-legend",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._left = self.query_one("#detail-left", Static)
        self._right = self.query_one("#detail-right", Static)
        self._timer = self.set_interval(self._refresh_sec, self._refresh_view)
        await self._load_tickers()

    async def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._broker_task is not None and not self._broker_task.done():
            self._broker_task.cancel()
        for leg in self._draft.legs:
            con_id = int(getattr(leg.contract, "conId", 0) or 0)
            if con_id:
                self._client.release_ticker(con_id, owner="details-package")

    async def _load_tickers(self) -> None:
        try:
            tickers = await asyncio.gather(
                *(
                    self._client.ensure_ticker(
                        leg.contract,
                        owner="details-package",
                    )
                    for leg in self._draft.legs
                )
            )
            self._tickers = {
                int(getattr(leg.contract, "conId", 0) or 0): ticker
                for leg, ticker in zip(self._draft.legs, tickers)
            }
            self._status = "Ready for broker what-if"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = f"Market data error: {exc}"
        self._refresh_view()

    def _current_quote(self) -> LiveOptionPackageQuote | None:
        tickers = [
            self._tickers.get(int(getattr(leg.contract, "conId", 0) or 0))
            for leg in self._draft.legs
        ]
        if any(ticker is None for ticker in tickers):
            return None
        symbol = str(getattr(self._draft.legs[0].contract, "symbol", "") or "")
        return quote_live_option_package(
            symbol=symbol,
            legs=self._draft.legs,
            tickers=tickers,
            quantity=self._quantity,
            intent="enter",
            mode=self._MODES[self._mode_index],
        )

    @staticmethod
    def _number(value: object, *, signed: bool = False) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{parsed:+,.2f}" if signed else f"{parsed:,.2f}"

    def _refresh_view(self) -> None:
        if not hasattr(self, "_left"):
            return
        self._quote = self._current_quote()
        width = pane_width(self._left, floor=62)
        inner = max(width - 2, 24)
        left: list[Text] = [
            box_top("OPTION PACKAGE", inner, style="#875bc7"),
        ]
        if self._quote is None:
            left.append(
                box_row(
                    Text("Awaiting complete live quotes", style="bold yellow"),
                    inner,
                    style="#875bc7",
                )
            )
        else:
            quote = self._quote
            risk = quote.live.risk
            title = (
                f"{quote.live.package.product.underlying_symbol} • "
                f"{risk.structure.replace('_', ' ').upper()} • {risk.expiry}"
            )
            left.append(box_row(Text(title, style="bold #d9bcff"), inner, style="#875bc7"))
            left.append(
                box_row(
                    f"Net bid {quote.bid_value:+.2f}   mid {quote.mid_value:+.2f}   "
                    f"ask {quote.ask_value:+.2f}   {self._MODES[self._mode_index]} "
                    f"{quote.limit_value:+.2f}",
                    inner,
                    style="#875bc7",
                )
            )
            max_profit = (
                "unbounded"
                if risk.max_profit is None
                else f"${risk.max_profit:,.2f}"
            )
            left.append(
                box_row(
                    Text.assemble(
                        ("Max profit ", "dim"),
                        (max_profit, "bold #77e6a8"),
                        ("   Max loss ", "dim"),
                        (f"${risk.max_loss:,.2f}", "bold #ff8f9d"),
                    ),
                    inner,
                    style="#875bc7",
                )
            )
            left.append(box_rule("LEGS", inner, style="#875bc7"))

        for index, leg in enumerate(self._draft.legs, start=1):
            contract = leg.contract
            ticker = self._tickers.get(int(getattr(contract, "conId", 0) or 0))
            bid = getattr(ticker, "bid", None) if ticker is not None else None
            ask = getattr(ticker, "ask", None) if ticker is not None else None
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            strike = self._number(getattr(contract, "strike", None))
            row = (
                f"{index}. {leg.action:<4} ×{leg.ratio}  {right} {strike}  "
                f"bid {self._number(bid)} / ask {self._number(ask)}"
            )
            left.append(box_row(row, inner, style="#875bc7"))
        left.append(box_bottom(inner, style="#875bc7"))
        self._left.update(Text("\n").join(left))

        right_width = pane_width(self._right, floor=46)
        right_inner = max(right_width - 2, 24)
        preview: list[Text] = [box_top("IBKR WHAT-IF", right_inner, style="#2f78c4")]
        preview.append(
            box_row(
                Text(self._status, style="bold #9fd7ff"),
                right_inner,
                style="#2f78c4",
            )
        )
        decision = self._preview
        if decision is None:
            preview.append(
                box_row(
                    "P previews only; nothing is submitted",
                    right_inner,
                    style="#2f78c4",
                )
            )
        else:
            style = "bold #77e6a8" if decision.allow else "bold #ff8f9d"
            preview.append(
                box_row(
                    Text(
                        f"{'ADMITTED' if decision.allow else 'BLOCKED'} • {decision.reason}",
                        style=style,
                    ),
                    right_inner,
                    style="#2f78c4",
                )
            )
            trace = decision.trace
            for label, key in (
                ("Initial margin Δ", "init_margin_change"),
                ("Initial margin after", "init_margin_after"),
                ("Equity+loan after", "equity_with_loan_after"),
                ("Commission", "commission"),
            ):
                preview.append(
                    box_row(
                        f"{label:<22} {self._number(trace.get(key), signed='change' in key)}",
                        right_inner,
                        style="#2f78c4",
                    )
                )
            age = (
                max(0.0, monotonic() - self._preview_at)
                if self._preview_at is not None
                else self._PREVIEW_TTL_SEC
            )
            preview.append(
                box_row(
                    f"Preview age {age:.1f}s / {self._PREVIEW_TTL_SEC:.0f}s",
                    right_inner,
                    style="#2f78c4",
                )
            )
        preview.append(box_rule("CONTROLS", right_inner, style="#2f78c4"))
        preview.append(
            box_row(
                f"←/→ price mode   ↑/↓ quantity ({self._quantity})",
                right_inner,
                style="#2f78c4",
            )
        )
        preview.append(
            box_row(
                "P what-if   Shift+B send exact admitted preview",
                right_inner,
                style="#2f78c4",
            )
        )
        preview.append(box_bottom(right_inner, style="#2f78c4"))
        self._right.update(Text("\n").join(preview))

    def _start(self, operation: Callable[[], Coroutine]) -> None:
        if self._broker_task is not None and not self._broker_task.done():
            self._status = "Broker request already running"
            self._refresh_view()
            return
        self._broker_task = asyncio.create_task(operation())

    def action_preview(self) -> None:
        self._start(self._preview_order)

    async def _preview_order(self) -> None:
        quote = self._current_quote()
        if quote is None:
            self._status = "Preview blocked: package quote unavailable"
            self._refresh_view()
            return
        self._status = "Requesting IBKR what-if…"
        self._refresh_view()
        try:
            config = getattr(self._client, "_config", None)
            decision = await preview_and_admit_option_order(
                self._client,
                account=str(getattr(config, "account", "") or ""),
                package=quote.live.package,
                risk=quote.live.risk,
                contract=quote.live.contract,
                legs=quote.live.legs,
                action="BUY",
                quantity=self._quantity,
                limit_price=quote.limit_value,
                intent="enter",
            )
            self._preview = decision
            self._preview_quote = quote
            self._preview_at = monotonic()
            self._status = (
                "Preview admitted; Shift+B sends this exact limit"
                if decision.allow
                else f"Preview blocked: {decision.reason}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._preview = None
            self._preview_quote = None
            self._preview_at = None
            self._status = f"Preview error: {exc}"
        self._refresh_view()

    def on_key(self, event: events.Key) -> None:
        if event.character == "B":
            self._start(self._send_previewed_order)
            event.stop()

    async def _send_previewed_order(self) -> None:
        quote = self._preview_quote
        age = (
            monotonic() - self._preview_at
            if self._preview_at is not None
            else self._PREVIEW_TTL_SEC + 1
        )
        if (
            quote is None
            or self._preview is None
            or not self._preview.allow
            or age > self._PREVIEW_TTL_SEC
        ):
            self._status = "Send blocked: run a fresh admitted what-if first"
            self._refresh_view()
            return
        self._status = "Submitting exact previewed BAG limit…"
        self._refresh_view()
        try:
            trade = await self._client.place_limit_order(
                quote.live.contract,
                "BUY",
                self._quantity,
                quote.limit_value,
                outside_rth=False,
            )
            order = getattr(trade, "order", None)
            order_id = int(
                getattr(order, "orderId", 0) or getattr(order, "permId", 0) or 0
            )
            self._status = (
                f"Submitted BAG #{order_id} @ {quote.limit_value:+.2f}"
                if order_id
                else f"Submitted BAG @ {quote.limit_value:+.2f}"
            )
            self._preview = None
            self._preview_quote = None
            self._preview_at = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = f"Submit error: {exc}"
        self._refresh_view()

    def _invalidate_preview(self) -> None:
        self._preview = None
        self._preview_quote = None
        self._preview_at = None
        self._status = "Package changed; run a fresh what-if"

    def action_mode_left(self) -> None:
        self._mode_index = max(0, self._mode_index - 1)
        self._invalidate_preview()
        self._refresh_view()

    def action_mode_right(self) -> None:
        self._mode_index = min(len(self._MODES) - 1, self._mode_index + 1)
        self._invalidate_preview()
        self._refresh_view()

    def action_quantity_up(self) -> None:
        self._quantity += 1
        self._invalidate_preview()
        self._refresh_view()

    def action_quantity_down(self) -> None:
        self._quantity = max(1, self._quantity - 1)
        self._invalidate_preview()
        self._refresh_view()

    def action_refresh(self) -> None:
        self._start(self._load_tickers)
