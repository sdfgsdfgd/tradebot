"""Bot runtime loop orchestration mixin."""

from __future__ import annotations

import asyncio
from datetime import datetime

from .common import _EXEC_LADDER_TIMEOUT_SEC, _exec_ladder_mode


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

    async def _chase_orders_tick(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        now = loop.time()
        if now - self._last_chase_ts < 5.0:
            return
        self._last_chase_ts = now

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
                changed = await self._reprice_order(order, mode="OPTIMISTIC")
                updated = updated or changed
                continue

            trade = order.trade
            if trade is None:
                order.status = "ERROR"
                order.error = "Missing IB trade handle for WORKING order"
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
                        mode_now = _exec_ladder_mode(elapsed)
                        done_data["exec_mode_now"] = str(mode_now) if mode_now is not None else "TIMEOUT"
                    sec_type = str(getattr(order.order_contract, "secType", "") or "").strip().upper()
                    intent = str(order.intent or "").strip().lower()
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
                    self._journal_write(
                        event=done_event,
                        order=order,
                        reason=None,
                        data=done_data,
                    )
                updated = True
                continue

            if order.status != "WORKING":
                continue

            if order.sent_at is None:
                order.sent_at = float(now)
            elapsed = now - float(order.sent_at)
            mode = _exec_ladder_mode(elapsed)
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
                updated = True
                continue

            changed = await self._reprice_order(order, mode=mode)
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
