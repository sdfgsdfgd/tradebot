"""Bot runtime loop orchestration mixin."""

from __future__ import annotations

from ..spot.lifecycle import SpotEntryBasisState, reconcile_spot_entry_basis
from ..spot.scenario import project_live_spot_execution_receipt

import asyncio
from datetime import datetime, timedelta

from ..engines.execution import (
    _EXEC_AUTO_TIMEOUT_SEC,
    _exec_chase_mode,
    _exec_chase_quote_signature,
    _exec_chase_should_reprice,
    _midpoint,
    _sanitize_nbbo,
)
from ..option_package import option_package_debit_value
from ..time_utils import now_et_naive as _now_et_naive
from .common import _safe_num

_DEFAULT_EXIT_RETRY_COOLDOWN_SEC = 3.0
_CANCEL_ACK_TIMEOUT_SEC = 6.0


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
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None) if ticker else getattr(order, "bid", None),
                getattr(ticker, "ask", None) if ticker else getattr(order, "ask", None),
                getattr(ticker, "last", None) if ticker else getattr(order, "last", None),
            )
            return _exec_chase_quote_signature(bid, ask, last)

        if legs:
            mid_rows: list[tuple[str, int, float | None]] = []
            bid_rows: list[tuple[str, int, float | None]] = []
            ask_rows: list[tuple[str, int, float | None]] = []
            for leg in legs:
                con_id = int(getattr(getattr(leg, "contract", None), "conId", 0) or 0)
                ticker = self._client.ticker_for_con_id(con_id) if con_id else None
                bid, ask, last = _sanitize_nbbo(
                    getattr(ticker, "bid", None) if ticker else None,
                    getattr(ticker, "ask", None) if ticker else None,
                    getattr(ticker, "last", None) if ticker else None,
                )
                mid = _midpoint(bid, ask)
                action = (
                    "BUY"
                    if str(getattr(leg, "action", "")).strip().upper() == "BUY"
                    else "SELL"
                )
                ratio = int(getattr(leg, "ratio", 1) or 1)
                mid_value = mid if mid is not None else last
                bid_value = (
                    bid
                    if bid is not None
                    else (mid if mid is not None else last)
                )
                ask_value = (
                    ask
                    if ask is not None
                    else (mid if mid is not None else last)
                )
                mid_rows.append(
                    (
                        action,
                        ratio,
                        None if mid_value is None else float(mid_value),
                    )
                )
                bid_rows.append(
                    (
                        action,
                        ratio,
                        None if bid_value is None else float(bid_value),
                    )
                )
                ask_rows.append(
                    (
                        action,
                        ratio,
                        None if ask_value is None else float(ask_value),
                    )
                )
            debit_mid = option_package_debit_value(mid_rows)
            debit_bid = option_package_debit_value(bid_rows)
            debit_ask = option_package_debit_value(ask_rows)
            out_bid = (
                float(debit_bid)
                if debit_bid is not None
                else _safe_num(getattr(order, "bid", None))
            )
            out_ask = (
                float(debit_ask)
                if debit_ask is not None
                else _safe_num(getattr(order, "ask", None))
            )
            out_last = (
                float(debit_mid)
                if debit_mid is not None
                else _safe_num(getattr(order, "last", None))
            )
            # Native BAG credits are valid negative package prices. Preserve
            # their sign for quote-change detection while still rejecting
            # zero, NaN and otherwise non-numeric values.
            return (
                _safe_num(out_bid),
                _safe_num(out_ask),
                _safe_num(out_last),
            )

        return _exec_chase_quote_signature(
            _safe_num(getattr(order, "bid", None)),
            _safe_num(getattr(order, "ask", None)),
            _safe_num(getattr(order, "last", None)),
        )

    @staticmethod
    def _apply_spot_entry_basis_fill(
        instance: _BotInstance,
        order: _BotOrder,
        *,
        cumulative_filled: float,
        fill_price: float | None,
    ) -> SpotEntryBasisState:
        try:
            cumulative = max(0.0, float(cumulative_filled or 0.0))
        except (TypeError, ValueError):
            cumulative = 0.0
        try:
            applied = max(0.0, float(order.basis_applied_filled_qty or 0.0))
        except (TypeError, ValueError):
            applied = 0.0
        incremental_abs = max(0.0, float(cumulative - applied))
        action = str(order.action or "").strip().upper()
        signed_delta = (
            float(incremental_abs)
            if action == "BUY"
            else -float(incremental_abs)
            if action == "SELL"
            else 0.0
        )
        state = reconcile_spot_entry_basis(
            previous_qty=float(instance.spot_entry_basis_qty or 0.0),
            previous_basis_price=instance.spot_entry_basis_price,
            fill_delta_qty=float(signed_delta),
            fill_price=fill_price,
            broker_qty=None,
            broker_average_cost=None,
        )
        if incremental_abs > 1e-12:
            instance.spot_entry_basis_qty = float(state.quantity)
            instance.spot_entry_basis_price = (
                float(state.basis_price) if state.basis_price is not None else None
            )
            instance.spot_entry_basis_source = str(state.source)
            instance.spot_entry_basis_set_ts = _now_et_naive()
            order.basis_applied_filled_qty = float(cumulative)
        journal = order.journal if isinstance(order.journal, dict) else {}
        journal["spot_trace_receipt"] = project_live_spot_execution_receipt(
            journal=journal,
            status=order.status,
            filled_qty=order.filled_qty,
            remaining_qty=order.remaining_qty,
            executed_qty=order.executed_qty,
            basis_applied_filled_qty=order.basis_applied_filled_qty,
            entry_basis_qty=instance.spot_entry_basis_qty,
            entry_basis_price=instance.spot_entry_basis_price,
            entry_basis_source=instance.spot_entry_basis_source,
        )
        order.journal = journal
        return state

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
                order.cancel_requested_at = None
                order.chase_last_reprice_ts = None
                order.chase_quote_signature = None
                updated = True
                continue

            broker_state: dict[str, object] | None = None
            broker_effective_status = ""
            broker_is_terminal = False
            cancel_resume_verified = False
            client = getattr(self, "_client", None)
            current_order_state = getattr(client, "current_order_state", None)
            if callable(current_order_state):
                try:
                    perm_id = int(
                        getattr(getattr(trade, "order", None), "permId", 0) or 0
                    )
                except (TypeError, ValueError):
                    perm_id = 0
                try:
                    candidate_state = current_order_state(
                        order_id=int(order.order_id or 0),
                        perm_id=int(perm_id),
                    )
                except Exception:
                    candidate_state = None
                if isinstance(candidate_state, dict):
                    broker_state = candidate_state

            if order.status == "CANCELING":
                cancel_since = (
                    float(order.cancel_requested_at)
                    if order.cancel_requested_at is not None
                    else float(order.sent_at if order.sent_at is not None else now)
                )
                cancel_age = max(0.0, float(now) - float(cancel_since))
                cached_cancel_status = (
                    str(broker_state.get("effective_status") or "").strip()
                    if isinstance(broker_state, dict)
                    else ""
                )
                if (
                    cancel_age >= float(_CANCEL_ACK_TIMEOUT_SEC)
                    and cached_cancel_status != "PendingCancel"
                ):
                    reconcile_order_state = getattr(
                        client,
                        "reconcile_order_state",
                        None,
                    )
                    if callable(reconcile_order_state):
                        try:
                            cancel_perm_id = int(
                                getattr(
                                    getattr(trade, "order", None),
                                    "permId",
                                    0,
                                )
                                or 0
                            )
                        except (TypeError, ValueError):
                            cancel_perm_id = 0
                        try:
                            refreshed_state = await reconcile_order_state(
                                order_id=int(order.order_id or 0),
                                perm_id=int(cancel_perm_id),
                                force=True,
                            )
                        except Exception:
                            refreshed_state = None
                        if isinstance(refreshed_state, dict):
                            broker_state = refreshed_state
                            refreshed_cancel_status = str(
                                refreshed_state.get("effective_status") or ""
                            ).strip()
                            cancel_resume_verified = refreshed_cancel_status in (
                                "Submitted",
                                "PreSubmitted",
                            )

            if broker_state is not None:
                rebound_trade = broker_state.get("trade")
                if rebound_trade is not None:
                    trade = rebound_trade
                    order.trade = rebound_trade

                try:
                    rebound_order_id = int(broker_state.get("order_id") or 0)
                except (TypeError, ValueError):
                    rebound_order_id = 0
                if rebound_order_id <= 0:
                    try:
                        rebound_order_id = int(
                            getattr(getattr(trade, "order", None), "orderId", 0) or 0
                        )
                    except (TypeError, ValueError):
                        rebound_order_id = 0
                if rebound_order_id > 0:
                    order.order_id = int(rebound_order_id)

                broker_effective_status = str(
                    broker_state.get("effective_status") or ""
                ).strip()
                broker_is_terminal = bool(broker_state.get("is_terminal"))

                for field_name in ("filled_qty", "remaining_qty", "executed_qty"):
                    try:
                        field_value = max(
                            0.0,
                            float(broker_state.get(field_name) or 0.0),
                        )
                    except (TypeError, ValueError):
                        field_value = 0.0
                    setattr(order, field_name, float(field_value))

                rebound_status = getattr(trade, "orderStatus", None)
                if rebound_status is not None:
                    try:
                        existing_filled = float(
                            getattr(rebound_status, "filled", 0.0) or 0.0
                        )
                    except (TypeError, ValueError):
                        existing_filled = 0.0
                    projected_filled = existing_filled
                    for key in ("filled_qty", "executed_qty"):
                        try:
                            projected_filled = max(
                                projected_filled,
                                float(broker_state.get(key) or 0.0),
                            )
                        except (TypeError, ValueError):
                            continue
                    try:
                        rebound_status.filled = float(max(0.0, projected_filled))
                    except Exception:
                        pass

                    if "remaining_qty" in broker_state:
                        try:
                            projected_remaining = max(
                                0.0,
                                float(broker_state.get("remaining_qty") or 0.0),
                            )
                        except (TypeError, ValueError):
                            projected_remaining = None
                        if projected_remaining is not None:
                            try:
                                rebound_status.remaining = float(projected_remaining)
                            except Exception:
                                pass

            is_done = False
            try:
                is_done = bool(trade.isDone())
            except Exception:
                is_done = False
            status_raw = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
            status = broker_effective_status or status_raw.strip()
            if (
                status in ("Filled", "Cancelled", "ApiCancelled", "Inactive")
                or broker_is_terminal
            ):
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
                    done_data: dict[str, object] = {
                        "ib_status": status_raw,
                        "filled_qty": float(order.filled_qty),
                        "remaining_qty": float(order.remaining_qty),
                        "executed_qty": float(order.executed_qty),
                    }
                    if broker_effective_status and broker_effective_status != status_raw.strip():
                        done_data["ib_status_effective"] = broker_effective_status
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
                    if (
                        status in ("Cancelled", "ApiCancelled")
                        and intent == "enter"
                        and max(float(order.filled_qty), float(order.executed_qty)) <= 0.0
                        and signal_bar_ts is not None
                        and instance.last_entry_bar_ts == signal_bar_ts
                    ):
                        instance.last_entry_bar_ts = None
                        done_data["entry_lock_released"] = True
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
                    if status == "Filled" and intent == "resize":
                        if signal_bar_ts is not None:
                            instance.last_resize_bar_ts = signal_bar_ts
                        done_data["resize_bar_ts"] = (
                            signal_bar_ts.isoformat() if signal_bar_ts is not None else None
                        )
                    if sec_type == "STK" and intent in ("enter", "resize"):
                        try:
                            status_filled = max(
                                0.0,
                                float(
                                    getattr(
                                        getattr(trade, "orderStatus", None),
                                        "filled",
                                        0.0,
                                    )
                                    or 0.0
                                ),
                            )
                        except (TypeError, ValueError):
                            status_filled = 0.0
                        cumulative_filled = max(
                            float(order.filled_qty or 0.0),
                            float(order.executed_qty or 0.0),
                            float(status_filled),
                        )
                        if cumulative_filled > float(order.basis_applied_filled_qty or 0.0) + 1e-12:
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
                                avg_fill = getattr(
                                    getattr(trade, "orderStatus", None),
                                    "avgFillPrice",
                                    None,
                                )
                                try:
                                    avg_fill_f = float(avg_fill) if avg_fill is not None else None
                                except (TypeError, ValueError):
                                    avg_fill_f = None
                                if avg_fill_f is not None and avg_fill_f > 0:
                                    basis_price = float(avg_fill_f)
                                    basis_source = "average_fill"
                            if basis_price is None:
                                try:
                                    limit_f = float(order.limit_price)
                                except (TypeError, ValueError):
                                    limit_f = None
                                if limit_f is not None and limit_f > 0:
                                    basis_price = float(limit_f)
                                    basis_source = "order_limit"
                            if basis_price is not None:
                                basis_state = self._apply_spot_entry_basis_fill(
                                    instance,
                                    order,
                                    cumulative_filled=float(cumulative_filled),
                                    fill_price=float(basis_price),
                                )
                                done_data["entry_basis_qty"] = float(basis_state.quantity)
                                done_data["entry_basis_price"] = (
                                    float(basis_state.basis_price)
                                    if basis_state.basis_price is not None
                                    else None
                                )
                                done_data["entry_basis_source"] = str(basis_state.source)
                                done_data["entry_basis_fill_source"] = str(
                                    basis_source or "fill"
                                )
                    if status == "Filled" and sec_type == "STK":
                        if intent == "enter":
                            journal = getattr(order, "journal", None)
                            entry_branch = None
                            if isinstance(journal, dict):
                                raw_branch = journal.get("entry_branch")
                                if raw_branch in ("a", "b"):
                                    entry_branch = str(raw_branch)
                            instance.spot_entry_branch = entry_branch
                            done_data["entry_branch"] = entry_branch
                        elif intent == "resize":
                            done_data["resize_applied"] = True
                        elif intent == "exit":
                            instance.spot_entry_basis_qty = 0.0
                            instance.spot_entry_basis_price = None
                            instance.spot_entry_basis_source = None
                            instance.spot_entry_basis_set_ts = None
                            instance.spot_entry_branch = None
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
                        now_wall = _now_et_naive()
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
                order.cancel_requested_at = None
                order.chase_last_reprice_ts = None
                order.chase_quote_signature = None
                updated = True
                continue

            if order.status == "CANCELING":
                cancel_since = (
                    float(order.cancel_requested_at)
                    if order.cancel_requested_at is not None
                    else float(order.sent_at if order.sent_at is not None else now)
                )
                cancel_age = max(0.0, float(now) - float(cancel_since))
                if cancel_age < float(_CANCEL_ACK_TIMEOUT_SEC):
                    continue
                if not cancel_resume_verified:
                    continue
                order.status = "WORKING"
                order.cancel_requested_at = None
                order.error = f"Cancel ack timeout after {cancel_age:.1f}s; resuming chase"
                self._status = f"Cancel ack timeout #{order.order_id or 0}; resuming chase"
                self._journal_write(
                    event="CANCEL_ACK_TIMEOUT",
                    order=order,
                    reason="canceling-stale",
                    data={"age_sec": float(cancel_age)},
                )
                updated = True

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
                    order.cancel_requested_at = float(now)
                    order.error = f"Timeout after {int(elapsed)}s"
                    self._journal_write(
                        event="ORDER_TIMEOUT_CANCEL",
                        order=order,
                        reason="timeout",
                        data={
                            "elapsed_sec": float(elapsed),
                            "timeout_sec": float(_EXEC_AUTO_TIMEOUT_SEC),
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
                    order.cancel_requested_at = None
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
