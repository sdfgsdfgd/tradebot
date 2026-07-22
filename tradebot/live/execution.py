"""Live order identity, chase state, cancellation intent, and task ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from time import monotonic

from ib_insync import Trade

from ..engines.execution import (
    _EXEC_AUTO_TIMEOUT_SEC,
    _EXEC_LADDER_TIMEOUT_SEC,
    _EXEC_RELENTLESS_TIMEOUT_SEC,
    ExecutionPolicy,
    _exec_chase_mode,
    _exec_chase_quote_signature,
    _exec_chase_should_reprice,
    _finite_float,
    _midpoint,
    _quote_num_actionable,
    _round_to_tick,
    _tick_size,
    execution_mode_label,
)
from ..time_utils import now_et


CHASE_STATE_BY_ORDER: dict[int, dict[str, object]] = {}


def order_ids(trade: object) -> tuple[int, int]:
    order = getattr(trade, "order", None)
    try:
        order_id = int(getattr(order, "orderId", 0) or 0)
    except (TypeError, ValueError):
        order_id = 0
    try:
        perm_id = int(getattr(order, "permId", 0) or 0)
    except (TypeError, ValueError):
        perm_id = 0
    return max(0, order_id), max(0, perm_id)


class LiveOrderExecution:
    """Own live-order identity, transient state, and broker chase lifecycle."""

    def __init__(
        self,
        *,
        client: object | None = None,
        state_by_order: dict[int, dict[str, object]] | None = None,
        cancel_ttl_sec: float = 90.0,
        error_max_age_sec: float = 300.0,
        price_for_mode: Callable[..., float | None] | None = None,
        recent_spreads: Callable[[], Iterable[float]] | None = None,
        on_update: Callable[[str | None, str | None, str], None] | None = None,
    ) -> None:
        self.client = client
        self.state_by_order = CHASE_STATE_BY_ORDER if state_by_order is None else state_by_order
        self.cancel_ttl_sec = max(1.0, float(cancel_ttl_sec))
        self.error_max_age_sec = max(1.0, float(error_max_age_sec))
        self.price_for_mode = price_for_mode
        self.recent_spreads = recent_spreads or (lambda: ())
        self.on_update = on_update
        self.task_by_order: dict[int, asyncio.Task] = {}
        self.cancel_requested_at_by_order: dict[int, float] = {}

    @staticmethod
    def keys(*, order_id: int, perm_id: int) -> tuple[int, ...]:
        keys = [value for value in (int(order_id or 0), int(perm_id or 0)) if value > 0]
        return tuple(dict.fromkeys(keys))

    def state(self, *, order_id: int, perm_id: int) -> dict[str, object] | None:
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            state = self.state_by_order.get(key)
            if isinstance(state, dict):
                return state
        return None

    def update_state(
        self,
        *,
        order_id: int,
        perm_id: int,
        updates: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        keys = self.keys(order_id=order_id, perm_id=perm_id)
        if not keys:
            return None
        state = self.state(order_id=order_id, perm_id=perm_id)
        if state is None:
            state = {}
        if updates:
            state.update(updates)
        for key in keys:
            self.state_by_order[key] = state
        return state

    def clear_state(self, *, order_id: int, perm_id: int) -> None:
        state = self.state(order_id=order_id, perm_id=perm_id)
        if state is not None:
            for key, value in list(self.state_by_order.items()):
                if value is state:
                    self.state_by_order.pop(key, None)
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            self.state_by_order.pop(key, None)

    def register_task(self, task: asyncio.Task, *, order_id: int, perm_id: int) -> None:
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            self.task_by_order[key] = task

    def unregister_task(self, task: asyncio.Task) -> None:
        for key, value in list(self.task_by_order.items()):
            if value is task:
                self.task_by_order.pop(key, None)

    def cancel_task(self, *, order_id: int, perm_id: int) -> None:
        tasks = {
            self.task_by_order[key]
            for key in self.keys(order_id=order_id, perm_id=perm_id)
            if key in self.task_by_order
        }
        for task in tasks:
            if not task.done():
                task.cancel()

    def mark_cancel_requested(
        self,
        *,
        order_id: int,
        perm_id: int,
        now: float | None = None,
    ) -> None:
        timestamp = monotonic() if now is None else float(now)
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            self.cancel_requested_at_by_order[key] = timestamp

    def clear_cancel_requested(self, *, order_id: int, perm_id: int) -> None:
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            self.cancel_requested_at_by_order.pop(key, None)

    def cancel_requested(
        self,
        *,
        order_id: int,
        perm_id: int,
        now: float | None = None,
    ) -> bool:
        timestamp = monotonic() if now is None else float(now)
        for key in self.keys(order_id=order_id, perm_id=perm_id):
            requested_at = self.cancel_requested_at_by_order.get(key)
            if requested_at is None:
                continue
            if timestamp - requested_at <= self.cancel_ttl_sec:
                return True
            self.cancel_requested_at_by_order.pop(key, None)
        return False

    def _emit(
        self,
        *,
        status: str | None = None,
        notice: str | None = None,
        level: str = "info",
    ) -> None:
        if self.on_update is not None:
            self.on_update(status, notice, level)

    def latest_trade(self, *, order_id: int, perm_id: int, fallback: object) -> object:
        lookup = getattr(self.client, "trade_for_order_ids", None)
        if not callable(lookup):
            return fallback
        try:
            refreshed = lookup(
                order_id=int(order_id),
                perm_id=int(perm_id),
                include_closed=True,
            )
        except TypeError:
            try:
                refreshed = lookup(int(order_id), int(perm_id))
            except Exception:
                refreshed = None
        except Exception:
            refreshed = None
        return fallback if refreshed is None else refreshed

    def current_order_state(self, *, order_id: int, perm_id: int) -> dict[str, object] | None:
        current_state = getattr(self.client, "current_order_state", None)
        if not callable(current_state):
            return None
        try:
            payload = current_state(order_id=int(order_id), perm_id=int(perm_id))
        except TypeError:
            try:
                payload = current_state(int(order_id), int(perm_id))
            except Exception:
                payload = None
        except Exception:
            payload = None
        return payload if isinstance(payload, dict) else None

    async def reconcile_order_state(
        self,
        *,
        order_id: int,
        perm_id: int,
        force: bool = False,
    ) -> dict[str, object] | None:
        reconcile = getattr(self.client, "reconcile_order_state", None)
        if not callable(reconcile):
            return None
        try:
            payload = await reconcile(
                order_id=int(order_id),
                perm_id=int(perm_id),
                force=bool(force),
            )
        except TypeError:
            try:
                payload = await reconcile(int(order_id), int(perm_id))
            except Exception:
                payload = None
        except Exception:
            payload = None
        return payload if isinstance(payload, dict) else None

    def consume_order_error(self, order_id: int, perm_id: int = 0) -> tuple[int, str] | None:
        pop_error = getattr(self.client, "pop_order_error", None)
        if not callable(pop_error):
            return None
        for order_ref in self.keys(order_id=order_id, perm_id=perm_id):
            try:
                payload = pop_error(order_ref, max_age_sec=self.error_max_age_sec)
            except TypeError:
                try:
                    payload = pop_error(order_ref)
                except Exception:
                    payload = None
            except Exception:
                payload = None
            if not isinstance(payload, dict):
                continue
            try:
                code = int(payload.get("code") or 0)
            except (TypeError, ValueError):
                code = 0
            message = str(payload.get("message") or "").strip()
            if message:
                return code, message
        return None

    async def await_order_error(
        self,
        order_id: int,
        perm_id: int = 0,
        *,
        attempts: int = 4,
        interval_sec: float = 0.1,
    ) -> tuple[int, str] | None:
        for attempt in range(max(1, int(attempts))):
            payload = self.consume_order_error(order_id, perm_id)
            if payload is not None:
                return payload
            if attempt + 1 < max(1, int(attempts)):
                await asyncio.sleep(max(0.01, float(interval_sec)))
        return None

    @staticmethod
    def cap_price_hint(trade: object) -> float | None:
        cap = _finite_float(getattr(getattr(trade, "orderStatus", None), "mktCapPrice", None))
        return float(cap) if cap is not None and cap > 0 else None

    async def chase(
        self,
        trade: Trade,
        action: str,
        *,
        mode: str,
        policy: ExecutionPolicy,
        pending_ack_sec: float = 0.9,
        reconcile_interval_sec: float = 0.9,
        force_reconcile_interval_sec: float = 5.0,
        modify_error_backoff_sec: float = 1.0,
    ) -> None:
        """Reconcile and reprice one live limit order until it is terminal."""

        if self.client is None or self.price_for_mode is None:
            raise RuntimeError("live execution requires a client and pricing policy")
        order_id, perm_id = order_ids(trade)
        con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
        chase_owner = f"details-chase:{order_id or perm_id or con_id or id(trade)}"
        try:
            await self.client.ensure_ticker(trade.contract, owner=chase_owner)
        except Exception:
            pass
        started = asyncio.get_running_loop().time()
        selected_label = execution_mode_label(mode)
        selected_is_delay = str(mode or "").strip().upper() == "RELENTLESS_DELAY"
        last_reprice_ts: float | None = None
        previous_mode: str | None = None
        previous_quote: tuple[float | None, float | None, float | None] | None = None
        arrival_ref: float | None = None
        no_progress_reprices = 0
        last_filled_qty = 0.0
        last_live_probe_ts: float | None = None
        last_reconcile_ts: float | None = None
        last_force_reconcile_ts: float | None = None
        last_modify_error_ts: float | None = None
        pending_since_ts: float | None = None

        async def recover_delay_rejection(
            error_message: str,
            *,
            replace_terminal: bool,
            now: float,
        ) -> None:
            nonlocal trade, order_id, perm_id
            state = self.state(order_id=order_id, perm_id=perm_id) or {}
            try:
                prior = int(state.get("delay_recoveries") or 0)
            except (TypeError, ValueError):
                prior = 0
            first_rejection = _finite_float(state.get("delay_first_202_ts"))
            if (
                first_rejection is None
                or first_rejection <= 0
                or now - first_rejection > max(1.0, policy.delay_recover_window_sec)
            ):
                first_rejection = now
            step = policy.delay_next_step(prior)
            favorable, leg_sign = policy.delay_leg(action, step)
            order_price = _finite_float(getattr(getattr(trade, "order", None), "lmtPrice", None))
            anchor_hint = (
                self.cap_price_hint(trade)
                or policy.price_hint_from_error(error_message)
                or _finite_float(state.get("delay_anchor_price"))
            )
            sweep_anchor = _finite_float(state.get("delay_sweep_anchor_price"))
            if prior <= 0 and order_price is not None and order_price > 0:
                sweep_anchor = order_price
            if sweep_anchor is None or sweep_anchor <= 0:
                sweep_anchor = order_price

            ticker = self.client.ticker_for_con_id(con_id) if con_id else None
            bid = _quote_num_actionable(getattr(ticker, "bid", None)) if ticker else None
            ask = _quote_num_actionable(getattr(ticker, "ask", None)) if ticker else None
            last = _quote_num_actionable(getattr(ticker, "last", None)) if ticker else None
            if replace_terminal and (sweep_anchor is None or sweep_anchor <= 0):
                last_ref = last if last is not None else (bid if bid is not None else ask)
                sweep_anchor = _round_to_tick(
                    _midpoint(bid, ask) or last_ref,
                    _tick_size(trade.contract, ticker, last_ref),
                )
            updates: dict[str, object] = {
                "selected": selected_label,
                "active": execution_mode_label("RELENTLESS_DELAY"),
                "delay_recoveries": step,
                "delay_first_202_ts": float(first_rejection),
                "delay_last_202_ts": float(now),
                "delay_last_leg_sign": float(leg_sign),
                "delay_last_leg_name": "FAV" if favorable else "ADV",
                "delay_locked_price_dir": None,
            }
            if anchor_hint is not None and anchor_hint > 0:
                updates["delay_anchor_price"] = float(anchor_hint)
            if sweep_anchor is not None and sweep_anchor > 0:
                updates["delay_sweep_anchor_price"] = float(sweep_anchor)
            self.update_state(order_id=order_id, perm_id=perm_id, updates=updates)

            leg = "FAV" if favorable else "ADV"
            suffix = f" {leg} step {step}/{policy.delay_sweep_span()}"
            if anchor_hint is not None and anchor_hint > 0:
                suffix += f" cap {float(anchor_hint):.2f}"
            if not replace_terminal:
                message = f"RLT⚔Delay sweep #{int(order_id or perm_id or 0)}{suffix}"
                self._emit(status=message, notice=message, level="warn")
                await asyncio.sleep(policy.delay_recover_cooldown_sec)
                return

            retry_price = self.price_for_mode(
                "RELENTLESS_DELAY",
                action,
                bid=bid,
                ask=ask,
                last=last,
                ticker=ticker,
                elapsed_sec=float(now - started),
                quote_stale=policy.quote_is_stale(ticker=ticker, bid=bid, ask=ask, last=last),
                open_shock=policy.in_open_shock(now_et().time()),
                no_progress_reprices=int(no_progress_reprices),
                arrival_ref=arrival_ref,
                delay_recoveries=step,
                delay_anchor_price=anchor_hint,
                delay_sweep_anchor_price=sweep_anchor,
            )
            if retry_price is None:
                retry_price = order_price
            quantity = _finite_float(getattr(getattr(trade, "order", None), "totalQuantity", None))
            order_ref = int(order_id or perm_id or 0)
            if retry_price is None or quantity is None or quantity <= 0:
                self._emit(
                    notice=f"RLT⚔Delay no retryable qty/price for #{order_ref}",
                    level="warn",
                )
                await asyncio.sleep(policy.delay_recover_cooldown_sec)
                return
            try:
                trade = await self.client.place_limit_order(
                    trade.contract,
                    action,
                    float(quantity),
                    float(retry_price),
                    str(getattr(trade.contract, "secType", "") or "").strip().upper() == "STK",
                )
                live_order_id, live_perm_id = order_ids(trade)
                order_id = live_order_id or order_id
                perm_id = live_perm_id or perm_id
                order_ref = int(order_id or perm_id or 0)
                updates["target_price"] = float(retry_price)
                self.update_state(order_id=order_id, perm_id=perm_id, updates=updates)
                message = f"RLT⚔Delay sweep #{order_ref} @ {float(retry_price):.2f}{suffix}"
                self._emit(status=message, notice=message, level="warn")
            except Exception as exc:
                self._emit(
                    notice=f"RLT⚔Delay retry failed #{order_ref}: {exc}",
                    level="warn",
                )
            await asyncio.sleep(policy.delay_recover_cooldown_sec)

        try:
            while True:
                trade = self.latest_trade(order_id=order_id, perm_id=perm_id, fallback=trade)  # type: ignore[assignment]
                live_order_id, live_perm_id = order_ids(trade)
                order_id = live_order_id or order_id
                perm_id = live_perm_id or perm_id
                live_con_id = int(getattr(getattr(trade, "contract", None), "conId", 0) or 0)
                con_id = live_con_id or con_id
                current_task = asyncio.current_task()
                if current_task is not None:
                    self.register_task(current_task, order_id=order_id, perm_id=perm_id)

                loop_now = asyncio.get_running_loop().time()
                status_raw = str(
                    getattr(getattr(trade, "orderStatus", None), "status", "") or ""
                ).strip()
                status_effective = status_raw
                live_state = (
                    self.current_order_state(order_id=order_id, perm_id=perm_id)
                    if order_id or perm_id
                    else None
                )
                if live_state is not None:
                    live_trade = live_state.get("trade")
                    if isinstance(live_trade, Trade):
                        trade = live_trade
                        live_order_id, live_perm_id = order_ids(trade)
                        order_id = live_order_id or order_id
                        perm_id = live_perm_id or perm_id
                    status_effective = str(
                        live_state.get("effective_status") or status_effective
                    ).strip() or status_effective

                terminal_statuses = ("Filled", "Cancelled", "ApiCancelled", "Inactive")
                repricable_statuses = ("PreSubmitted", "Submitted")
                pending_statuses = ("PendingSubmit", "PendingSubmission", "ApiPending")
                cancel_requested = self.cancel_requested(order_id=order_id, perm_id=perm_id)
                if status_raw in pending_statuses:
                    pending_since_ts = pending_since_ts or loop_now
                else:
                    pending_since_ts = None
                pending_age = max(0.0, loop_now - pending_since_ts) if pending_since_ts else 0.0

                reconciled: dict[str, object] | None = None
                if (
                    (order_id or perm_id)
                    and status_raw in pending_statuses
                    and pending_age >= float(pending_ack_sec)
                    and (
                        last_reconcile_ts is None
                        or loop_now - last_reconcile_ts >= float(reconcile_interval_sec)
                    )
                ):
                    force = pending_age >= float(pending_ack_sec) * 2.0 and (
                        last_force_reconcile_ts is None
                        or loop_now - last_force_reconcile_ts >= float(force_reconcile_interval_sec)
                    )
                    reconciled = await self.reconcile_order_state(
                        order_id=order_id,
                        perm_id=perm_id,
                        force=bool(force),
                    )
                    last_reconcile_ts = loop_now
                    if force:
                        last_force_reconcile_ts = loop_now
                if reconciled is not None:
                    reconciled_trade = reconciled.get("trade")
                    if isinstance(reconciled_trade, Trade):
                        trade = reconciled_trade
                        live_order_id, live_perm_id = order_ids(trade)
                        order_id = live_order_id or order_id
                        perm_id = live_perm_id or perm_id
                    status_effective = str(
                        reconciled.get("effective_status") or status_effective
                    ).strip() or status_effective

                filled_now = _finite_float(
                    getattr(getattr(trade, "orderStatus", None), "filled", None)
                ) or 0.0
                for payload in (reconciled, live_state):
                    if payload is not None:
                        filled_now = max(filled_now, _finite_float(payload.get("filled_qty")) or 0.0)
                fill_progress = filled_now > last_filled_qty + 1e-9
                if fill_progress:
                    last_filled_qty = filled_now
                    no_progress_reprices = 0
                try:
                    is_done = bool(trade.isDone())
                except Exception:
                    is_done = False
                terminal = (
                    status_effective in terminal_statuses
                    or status_raw in terminal_statuses
                    or is_done
                    or bool(reconciled and reconciled.get("is_terminal"))
                    or bool(live_state and live_state.get("is_terminal"))
                )
                if terminal:
                    order_ref = int(order_id or perm_id or 0)
                    order_label = f"#{order_ref}" if order_ref else "order"
                    error = self.consume_order_error(order_id, perm_id) if order_ref else None
                    status_label = status_effective or status_raw
                    if not status_label:
                        status_label = "Done"
                    elif is_done and status_label not in terminal_statuses:
                        status_label = f"Done ({status_label})"
                    if status_raw and status_effective and status_raw != status_effective:
                        status_label = f"{status_label} [{status_raw}]"
                    if error is not None:
                        code, message = error
                        if selected_is_delay and code == 202 and not cancel_requested:
                            await recover_delay_rejection(
                                message,
                                replace_terminal=True,
                                now=loop_now,
                            )
                            continue
                        prefix = f"IB {code}: " if code else "IB: "
                        self._emit(notice=f"{order_label} {status_label}: {prefix}{message}", level="error")
                    elif status_raw in ("Cancelled", "ApiCancelled", "Inactive"):
                        held = str(
                            getattr(getattr(trade, "orderStatus", None), "whyHeld", "") or ""
                        ).strip()
                        message = f"{order_label} {status_raw}" + (f": {held}" if held else "")
                        self._emit(notice=message, level="warn")
                    elif status_effective == "Filled" or status_raw == "Filled":
                        self._emit(notice=f"Filled {order_label}")
                    elif is_done:
                        self._emit(notice=f"{order_label} {status_label}", level="warn")
                    self.clear_cancel_requested(order_id=order_id, perm_id=perm_id)
                    self.clear_state(order_id=order_id, perm_id=perm_id)
                    return

                order_ref = int(order_id or perm_id or 0)
                error = self.consume_order_error(order_id, perm_id) if order_ref else None
                if error is not None:
                    code, message = error
                    if code in (110, 201, 202, 10147, 10148, 10149):
                        if selected_is_delay and code == 202:
                            await recover_delay_rejection(
                                message,
                                replace_terminal=False,
                                now=loop_now,
                            )
                            last_modify_error_ts = loop_now
                            continue
                        status_label = status_raw or "Pending"
                        prefix = f"IB {code}: " if code else "IB: "
                        status = f"Chase halted #{order_ref} {status_label}: {prefix}{message}"
                        level = "warn" if code in (10147, 10148, 10149) else "error"
                        self._emit(
                            status=status,
                            notice=f"#{order_ref} {status_label}: {prefix}{message}",
                            level=level,
                        )
                        self.clear_state(order_id=order_id, perm_id=perm_id)
                        return

                elapsed = loop_now - started
                mode_now = _exec_chase_mode(elapsed, selected_mode=mode)
                if mode_now is None:
                    try:
                        self.mark_cancel_requested(order_id=order_id, perm_id=perm_id)
                        await self.client.cancel_trade(trade)
                        live_order_id, live_perm_id = order_ids(trade)
                        order_id = live_order_id or order_id
                        perm_id = live_perm_id or perm_id
                        order_ref = int(order_id or perm_id or 0)
                        cleaned_mode = str(mode or "").strip().upper()
                        timeout = (
                            _EXEC_RELENTLESS_TIMEOUT_SEC
                            if cleaned_mode in ("RELENTLESS", "RELENTLESS_DELAY")
                            else _EXEC_AUTO_TIMEOUT_SEC
                            if cleaned_mode in ("AUTO", "LADDER")
                            else _EXEC_LADDER_TIMEOUT_SEC
                        )
                        status = f"Timeout cancel sent #{order_ref} (> {float(timeout):.0f}s)"
                        self._emit(status=status, notice=status, level="warn")
                        error = (
                            await self.await_order_error(order_id, perm_id, attempts=3, interval_sec=0.1)
                            if order_ref
                            else None
                        )
                        if error is not None:
                            code, message = error
                            prefix = f"IB {code}: " if code else "IB: "
                            status = f"Timeout cancel #{order_ref}: {prefix}{message}"
                            level = "warn" if code in (10147, 10148, 10149) else "error"
                            self._emit(status=status, notice=status, level=level)
                    except Exception as exc:
                        status = f"Timeout cancel error: {exc}"
                        self._emit(status=status, notice=status, level="error")
                    self.clear_state(order_id=order_id, perm_id=perm_id)
                    return

                ticker = self.client.ticker_for_con_id(con_id) if con_id else None
                bid = _quote_num_actionable(getattr(ticker, "bid", None)) if ticker else None
                ask = _quote_num_actionable(getattr(ticker, "ask", None)) if ticker else None
                last = _quote_num_actionable(getattr(ticker, "last", None)) if ticker else None
                arrival_ref = arrival_ref or _midpoint(bid, ask) or last
                cleaned_mode = str(mode_now or "").strip().upper()
                relentless = cleaned_mode in ("RELENTLESS", "RELENTLESS_DELAY")
                delay_mode = cleaned_mode == "RELENTLESS_DELAY"
                quote_stale = (
                    policy.quote_is_stale(ticker=ticker, bid=bid, ask=ask, last=last)
                    if relentless
                    else False
                )
                open_shock = policy.in_open_shock(now_et().time()) if relentless else False
                spread = (
                    float(ask) - float(bid)
                    if relentless and bid is not None and ask is not None and ask >= bid
                    else None
                )
                last_ref = last if last is not None else (bid if bid is not None else ask)
                tick = _tick_size(trade.contract, ticker, last_ref) if relentless else 0.0
                pressure = (
                    policy.spread_pressure(
                        spread=spread,
                        tick=tick,
                        recent_spreads=self.recent_spreads(),
                    )
                    if relentless
                    else 1.0
                )
                if relentless and quote_stale and (
                    last_live_probe_ts is None or loop_now - last_live_probe_ts >= 4.0
                ):
                    try:
                        await self.client.refresh_live_snapshot_once(trade.contract)
                    except Exception:
                        pass
                    last_live_probe_ts = loop_now
                quote_signature = _exec_chase_quote_signature(bid, ask, last)
                interval = (
                    policy.reprice_interval(
                        quote_stale=bool(quote_stale),
                        open_shock=bool(open_shock),
                        no_progress_reprices=int(no_progress_reprices),
                        spread_pressure=float(pressure),
                    )
                    if relentless
                    else 5.0
                )
                should_reprice = _exec_chase_should_reprice(
                    now_sec=loop_now,
                    last_reprice_sec=last_reprice_ts,
                    mode_now=str(mode_now),
                    prev_mode=previous_mode,
                    quote_signature=quote_signature,
                    prev_quote_signature=previous_quote,
                    min_interval_sec=float(interval),
                )
                if cancel_requested or (status_effective or status_raw) not in repricable_statuses:
                    should_reprice = False
                if (
                    should_reprice
                    and last_modify_error_ts is not None
                    and loop_now - last_modify_error_ts < float(modify_error_backoff_sec)
                ):
                    should_reprice = False
                previous_mode = str(mode_now)
                previous_quote = quote_signature
                if order_id or perm_id:
                    updates: dict[str, object] = {
                        "selected": selected_label,
                        "active": execution_mode_label(str(mode_now)),
                    }
                    if status_effective and status_effective != status_raw:
                        updates["effective_status"] = status_effective
                    self.update_state(order_id=order_id, perm_id=perm_id, updates=updates)

                price: float | None = None
                if should_reprice:
                    state = self.state(order_id=order_id, perm_id=perm_id) or {}
                    try:
                        recoveries = int(state.get("delay_recoveries") or 0)
                    except (TypeError, ValueError):
                        recoveries = 0
                    anchor = _finite_float(state.get("delay_anchor_price"))
                    sweep_anchor = _finite_float(state.get("delay_sweep_anchor_price"))
                    last_rejection = _finite_float(state.get("delay_last_202_ts"))
                    locked_direction = _finite_float(state.get("delay_locked_price_dir"))
                    if (
                        delay_mode
                        and recoveries > 0
                        and last_rejection is not None
                        and loop_now - last_rejection
                        >= max(policy.delay_recover_settle_sec, policy.delay_recover_cooldown_sec)
                    ):
                        last_leg = _finite_float(state.get("delay_last_leg_sign"))
                        locked_direction = 1.0 if last_leg is not None and last_leg >= 0 else (
                            -1.0 if last_leg is not None else None
                        )
                        recoveries = 0
                        anchor = None
                        sweep_anchor = None
                        self.update_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            updates={
                                "delay_recoveries": 0,
                                "delay_anchor_price": None,
                                "delay_sweep_anchor_price": None,
                                "delay_first_202_ts": None,
                                "delay_last_202_ts": None,
                                "delay_last_leg_sign": None,
                                "delay_last_leg_name": None,
                                "delay_locked_price_dir": locked_direction,
                            },
                        )
                        if locked_direction is not None:
                            direction = "up" if locked_direction > 0 else "down"
                            self._emit(
                                notice=f"RLT⚔Delay lock engaged ({direction} side) after 202 settle"
                            )
                    price = self.price_for_mode(
                        str(mode_now),
                        action,
                        bid=bid,
                        ask=ask,
                        last=last,
                        ticker=ticker,
                        elapsed_sec=float(elapsed),
                        quote_stale=bool(quote_stale),
                        open_shock=bool(open_shock),
                        no_progress_reprices=int(no_progress_reprices),
                        arrival_ref=arrival_ref,
                        delay_recoveries=recoveries if delay_mode else 0,
                        delay_anchor_price=anchor if delay_mode else None,
                        delay_sweep_anchor_price=sweep_anchor if delay_mode else None,
                        delay_locked_price_dir=locked_direction if delay_mode else None,
                    )
                    if (order_id or perm_id) and price is not None:
                        self.update_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            updates={
                                "selected": selected_label,
                                "active": execution_mode_label(str(mode_now)),
                                "target_price": float(price),
                            },
                        )
                if price is not None:
                    current_price = _finite_float(
                        getattr(getattr(trade, "order", None), "lmtPrice", None)
                    )
                    compare_tick = _tick_size(trade.contract, ticker, price) or 0.01
                    if current_price is not None and abs(price - current_price) <= max(
                        compare_tick * 0.5,
                        1e-9,
                    ):
                        last_reprice_ts = loop_now
                        price = None
                if price is not None:
                    try:
                        trade = await self.client.modify_limit_order(trade, float(price))
                        applied = _finite_float(
                            getattr(getattr(trade, "order", None), "lmtPrice", None)
                        )
                        applied = applied if applied is not None and applied > 0 else float(price)
                        live_order_id, live_perm_id = order_ids(trade)
                        order_id = live_order_id or order_id
                        perm_id = live_perm_id or perm_id
                        order_ref = int(order_id or perm_id or 0)
                        active_label = execution_mode_label(str(mode_now))
                        state = self.state(order_id=order_id, perm_id=perm_id) or {}
                        try:
                            modifications = int(state.get("mods") or 0) + 1
                        except (TypeError, ValueError):
                            modifications = 1
                        self.update_state(
                            order_id=order_id,
                            perm_id=perm_id,
                            updates={
                                "selected": selected_label,
                                "active": active_label,
                                "target_price": float(applied),
                                "mods": modifications,
                            },
                        )
                        mode_view = (
                            f"{selected_label}->{active_label}"
                            if selected_label == "AUTO"
                            else active_label
                        )
                        order_label = f"#{order_ref}" if order_ref else "order"
                        self._emit(
                            status=(
                                f"Chasing {order_label} [{mode_view}] @ "
                                f"{float(applied):.2f} mod#{modifications}"
                            )
                        )
                        last_reprice_ts = loop_now
                        last_modify_error_ts = None
                        if not fill_progress:
                            no_progress_reprices += 1
                    except Exception as exc:
                        last_modify_error_ts = loop_now
                        status = f"Chase error: {exc}"
                        self._emit(status=status, notice=status, level="error")
                await asyncio.sleep(0.25)
        finally:
            current_task = asyncio.current_task()
            if current_task is not None:
                self.unregister_task(current_task)
            self.clear_state(order_id=order_id, perm_id=perm_id)
            if con_id:
                try:
                    self.client.release_ticker(con_id, owner=chase_owner)
                except Exception:
                    pass
