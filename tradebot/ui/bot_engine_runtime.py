"""Bot runtime loop orchestration mixin."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .common import (
    _EXEC_LADDER_TIMEOUT_SEC,
    _exec_chase_mode,
    _exec_chase_quote_signature,
    _exec_chase_should_reprice,
    _midpoint,
    _safe_num,
)

_DEFAULT_EXIT_RETRY_COOLDOWN_SEC = 3.0


def _latest_trade_log_message(trade) -> str | None:
    for record in reversed(list(getattr(trade, "log", []) or [])):
        message = str(getattr(record, "message", "") or "").strip()
        if message:
            return message
    return None


class BotEngineRuntimeMixin:
    async def _on_refresh_tick(self) -> None:
        if self._refresh_lock.locked():
            return
        async with self._refresh_lock:
            await self._refresh_positions()
            await self._chase_orders_tick()
            await self._auto_order_tick()
            self._auto_send_tick()
            self._render_status()

    def _auto_send_tick(self) -> None:
        if self._send_task and not self._send_task.done():
            return
        order = next((o for o in self._orders if o.status == "STAGED"), None)
        if order is None:
            return
        instance = next((i for i in self._instances if i.instance_id == order.instance_id), None)
        if not instance:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Send: no loop"
            return
        self._send_task = loop.create_task(self._send_order(order))

    def _order_quote_signature(self, order) -> tuple[float | None, float | None, float | None]:
        legs = list(getattr(order, "legs", []) or [])
        sec_type = str(getattr(getattr(order, "order_contract", None), "secType", "") or "").upper()

        if len(legs) == 1 and sec_type != "BAG":
            con_id = int(getattr(getattr(legs[0], "contract", None), "conId", 0) or 0)
            if not con_id:
                con_id = int(getattr(getattr(order, "order_contract", None), "conId", 0) or 0)
            ticker = self._client.ticker_for_con_id(con_id) if con_id else None
            bid = _safe_num(getattr(ticker, "bid", None)) if ticker else _safe_num(getattr(order, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None)) if ticker else _safe_num(getattr(order, "ask", None))
            last = _safe_num(getattr(ticker, "last", None)) if ticker else _safe_num(getattr(order, "last", None))
            return _exec_chase_quote_signature(bid, ask, last)

        if legs:
            debit_mid = 0.0
            debit_bid = 0.0
            debit_ask = 0.0
            has_mid = True
            has_bid = True
            has_ask = True
            for leg in legs:
                con_id = int(getattr(getattr(leg, "contract", None), "conId", 0) or 0)
                ticker = self._client.ticker_for_con_id(con_id) if con_id else None
                bid = _safe_num(getattr(ticker, "bid", None)) if ticker else None
                ask = _safe_num(getattr(ticker, "ask", None)) if ticker else None
                last = _safe_num(getattr(ticker, "last", None)) if ticker else None
                mid = _midpoint(bid, ask)
                sign = 1.0 if str(getattr(leg, "action", "")).strip().upper() == "BUY" else -1.0
                ratio = int(getattr(leg, "ratio", 1) or 1)
                if mid is None and last is None:
                    has_mid = False
                else:
                    debit_mid += sign * float(mid if mid is not None else last) * ratio
                if bid is None and mid is None and last is None:
                    has_bid = False
                else:
                    debit_bid += sign * float(bid if bid is not None else (mid if mid is not None else last)) * ratio
                if ask is None and mid is None and last is None:
                    has_ask = False
                else:
                    debit_ask += sign * float(ask if ask is not None else (mid if mid is not None else last)) * ratio
            out_bid = float(debit_bid) if has_bid else _safe_num(getattr(order, "bid", None))
            out_ask = float(debit_ask) if has_ask else _safe_num(getattr(order, "ask", None))
            out_last = float(debit_mid) if has_mid else _safe_num(getattr(order, "last", None))
            return _exec_chase_quote_signature(out_bid, out_ask, out_last)

        return _exec_chase_quote_signature(
            _safe_num(getattr(order, "bid", None)),
            _safe_num(getattr(order, "ask", None)),
            _safe_num(getattr(order, "last", None)),
        )

    async def _chase_orders_tick(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        now = loop.time()

        updated = False
        for order in self._orders:
            if order.status not in ("STAGED", "WORKING", "CANCELING"):
                continue
            instance = next(
                (i for i in self._instances if i.instance_id == order.instance_id), None
            )
            if not instance:
                continue
            if order.status == "STAGED":
                mode_now = "OPTIMISTIC"
                quote_sig = self._order_quote_signature(order)
                if not _exec_chase_should_reprice(
                    now_sec=now,
                    last_reprice_sec=order.chase_last_reprice_ts,
                    mode_now=mode_now,
                    prev_mode=order.exec_mode,
                    quote_signature=quote_sig,
                    prev_quote_signature=order.chase_quote_signature,
                    min_interval_sec=5.0,
                ):
                    continue
                changed = await self._reprice_order(order, mode=mode_now)
                order.chase_last_reprice_ts = float(now)
                order.chase_quote_signature = self._order_quote_signature(order)
                updated = updated or changed
                continue

            trade = order.trade
            if trade is None:
                order.status = "ERROR"
                order.error = "Missing IB trade handle for WORKING order"
                order.chase_last_reprice_ts = None
                order.chase_quote_signature = None
                updated = True
                continue

            is_done = False
            try:
                is_done = bool(trade.isDone())
            except Exception:
                is_done = False
            status_raw = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
            status = status_raw.strip()
            if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                is_done = True

            if is_done:
                prev_status = order.status
                if status == "Filled":
                    order.status = "FILLED"
                    done_event = "ORDER_FILLED"
                elif status in ("Cancelled", "ApiCancelled"):
                    order.status = "CANCELLED"
                    done_event = "ORDER_CANCELLED"
                else:
                    order.status = status.upper() if status else "DONE"
                    done_event = "ORDER_DONE"
                if prev_status in ("WORKING", "CANCELING") and order.status != prev_status:
                    done_data: dict[str, object] = {"ib_status": status_raw}
                    if order.exec_mode:
                        done_data["exec_mode_last"] = str(order.exec_mode)
                    if order.sent_at is not None:
                        elapsed = now - float(order.sent_at)
                        done_data["exec_elapsed_sec"] = float(elapsed)
                        mode_now = _exec_chase_mode(elapsed, selected_mode="AUTO")
                        done_data["exec_mode_now"] = str(mode_now) if mode_now is not None else "TIMEOUT"
                    sec_type = str(getattr(order.order_contract, "secType", "") or "").strip().upper()
                    intent = str(order.intent or "").strip().lower()
                    signal_bar_ts = order.signal_bar_ts if isinstance(order.signal_bar_ts, datetime) else None
                    if status == "Inactive":
                        error_payload = None
                        client = getattr(self, "_client", None)
                        if client is not None:
                            try:
                                error_payload = client.pop_order_error(order.order_id or 0)
                            except Exception:
                                error_payload = None
                        if isinstance(error_payload, dict):
                            try:
                                error_code = int(error_payload.get("code") or 0)
                            except (TypeError, ValueError):
                                error_code = 0
                            error_message = str(error_payload.get("message") or "").strip()
                            if error_code:
                                done_data["ib_error_code"] = int(error_code)
                            if error_message:
                                done_data["ib_error_message"] = error_message
                                order.error = (
                                    f"IB {error_code}: {error_message}"
                                    if error_code
                                    else error_message
                                )
                        if "ib_error_message" not in done_data:
                            why_held = str(
                                getattr(getattr(trade, "orderStatus", None), "whyHeld", "") or ""
                            ).strip()
                            if why_held:
                                done_data["ib_why_held"] = why_held
                                order.error = why_held
                            else:
                                log_message = _latest_trade_log_message(trade)
                                if log_message:
                                    done_data["ib_log_message"] = log_message
                                    order.error = log_message
                    if status == "Filled" and intent == "exit":
                        if signal_bar_ts is not None:
                            instance.last_exit_bar_ts = signal_bar_ts
                        instance.exit_retry_bar_ts = None
                        instance.exit_retry_count = 0
                        instance.exit_retry_cooldown_until = None
                        done_data["exit_lock_bar_ts"] = (
                            signal_bar_ts.isoformat() if signal_bar_ts is not None else None
                        )
                    if status == "Filled" and sec_type == "STK":
                        if intent == "enter":
                            basis_price = None
                            basis_source = None
                            for fill in reversed(list(getattr(trade, "fills", []) or [])):
                                px = getattr(getattr(fill, "execution", None), "price", None)
                                try:
                                    px_f = float(px) if px is not None else None
                                except (TypeError, ValueError):
                                    px_f = None
                                if px_f is not None and px_f > 0:
                                    basis_price = float(px_f)
                                    basis_source = "execution_fill"
                                    break
                            if basis_price is None:
                                try:
                                    px_f = float(order.limit_price)
                                except (TypeError, ValueError):
                                    px_f = None
                                if px_f is not None and px_f > 0:
                                    basis_price = float(px_f)
                                    basis_source = "order_limit"
                            if basis_price is not None:
                                instance.spot_entry_basis_price = float(basis_price)
                                instance.spot_entry_basis_source = str(basis_source or "entry_fill")
                                instance.spot_entry_basis_set_ts = datetime.now()
                                done_data["entry_basis_price"] = float(basis_price)
                                done_data["entry_basis_source"] = str(instance.spot_entry_basis_source)
                        elif intent == "exit":
                            instance.spot_entry_basis_price = None
                            instance.spot_entry_basis_source = None
                            instance.spot_entry_basis_set_ts = None
                            done_data["entry_basis_cleared"] = True
                    if status == "Inactive" and intent == "exit":
                        if signal_bar_ts is not None and instance.exit_retry_bar_ts != signal_bar_ts:
                            instance.exit_retry_bar_ts = signal_bar_ts
                            instance.exit_retry_count = 0
                        instance.exit_retry_count = max(0, int(instance.exit_retry_count or 0)) + 1
                        raw_cd = instance.strategy.get("exit_retry_cooldown_sec", _DEFAULT_EXIT_RETRY_COOLDOWN_SEC)
                        try:
                            cooldown_sec = float(raw_cd if raw_cd is not None else _DEFAULT_EXIT_RETRY_COOLDOWN_SEC)
                        except (TypeError, ValueError):
                            cooldown_sec = _DEFAULT_EXIT_RETRY_COOLDOWN_SEC
                        cooldown_sec = max(0.0, float(cooldown_sec))
                        now_wall = datetime.now(tz=ZoneInfo("America/New_York")).replace(tzinfo=None)
                        if cooldown_sec > 0:
                            instance.exit_retry_cooldown_until = now_wall + timedelta(seconds=cooldown_sec)
                        else:
                            instance.exit_retry_cooldown_until = now_wall
                        done_data["retryable"] = True
                        done_data["retry_count"] = int(instance.exit_retry_count)
                        done_data["retry_cooldown_sec"] = float(cooldown_sec)
                        done_data["retry_cooldown_until"] = (
                            instance.exit_retry_cooldown_until.isoformat()
                            if instance.exit_retry_cooldown_until is not None
                            else None
                        )
                        done_data["retry_bar_ts"] = signal_bar_ts.isoformat() if signal_bar_ts is not None else None
                    self._journal_write(
                        event=done_event,
                        order=order,
                        reason=None,
                        data=done_data,
                    )
                order.chase_last_reprice_ts = None
                order.chase_quote_signature = None
                updated = True
                continue

            if order.status != "WORKING":
                continue

            if order.sent_at is None:
                order.sent_at = float(now)
            elapsed = now - float(order.sent_at)
            mode = _exec_chase_mode(elapsed, selected_mode="AUTO")
            if mode is None:
                # Timed out: cancel and give up.
                try:
                    order.status = "CANCELING"
                    order.error = f"Timeout after {int(elapsed)}s"
                    self._journal_write(
                        event="ORDER_TIMEOUT_CANCEL",
                        order=order,
                        reason="timeout",
                        data={
                            "elapsed_sec": float(elapsed),
                            "timeout_sec": float(_EXEC_LADDER_TIMEOUT_SEC),
                        },
                    )
                    await self._client.cancel_trade(trade)
                    self._journal_write(
                        event="CANCEL_SENT",
                        order=order,
                        reason="timeout",
                        data={"elapsed_sec": float(elapsed)},
                    )
                    self._status = f"Timeout cancel sent #{order.order_id or 0}"
                except Exception as exc:
                    order.status = "WORKING"
                    order.error = f"Timeout cancel error: {exc}"
                    self._status = f"Timeout cancel error #{order.order_id or 0}: {exc}"
                    self._journal_write(
                        event="CANCEL_ERROR",
                        order=order,
                        reason="timeout",
                        data={"elapsed_sec": float(elapsed), "exc": str(exc)},
                    )
                order.chase_last_reprice_ts = None
                order.chase_quote_signature = None
                updated = True
                continue

            quote_sig = self._order_quote_signature(order)
            should_reprice = _exec_chase_should_reprice(
                now_sec=now,
                last_reprice_sec=order.chase_last_reprice_ts,
                mode_now=str(mode),
                prev_mode=order.exec_mode,
                quote_signature=quote_sig,
                prev_quote_signature=order.chase_quote_signature,
                min_interval_sec=5.0,
            )
            if not should_reprice:
                continue

            changed = await self._reprice_order(order, mode=mode)
            order.chase_last_reprice_ts = float(now)
            order.chase_quote_signature = self._order_quote_signature(order)
            updated = updated or changed
            if not changed:
                continue

            try:
                order.trade = await self._client.modify_limit_order(trade, float(order.limit_price))
                updated = True
            except Exception as exc:
                order.error = f"Chase error: {exc}"
                self._status = f"Chase error #{order.order_id or 0}: {exc}"
                updated = True
        if updated:
            self._refresh_orders_table()
            if self._active_panel == "orders" and self._order_rows:
                row = min(
                    self._orders_table.cursor_coordinate.row, len(self._order_rows) - 1
                )
                self._orders_table.cursor_coordinate = (max(row, 0), 0)
