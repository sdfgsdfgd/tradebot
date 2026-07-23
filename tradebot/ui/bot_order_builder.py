"""Bot order construction mixin."""

from __future__ import annotations

import asyncio
from datetime import datetime

from ib_insync import Contract, Stock, Ticker

from ..engine import normalize_spot_entry_signal, spot_resolve_entry_action_qty, spot_runtime_spec_view
from ..engines.execution import (
    _limit_price_for_mode,
    _midpoint,
    _round_to_tick,
    _sanitize_nbbo,
    _tick_size,
    initial_execution_mode,
)
from ..live.options import QualifiedOptionLeg, normalize_option_position_close, quote_live_option_order, resolve_live_option_entry
from ..option_package import option_package_entry_intent
from ..order_reservation import OrderReservationCapacityRequest, evaluate_order_reservation_capacity
from ..spot.lifecycle import decide_open_position_intent
from ..spot.scenario import project_live_spot_order_journal
from ..time_utils import now_et as _now_et
from ..time_utils import now_et_naive as _now_et_naive
from .bot_journal import order_attempt_payload, order_build_failure_payload, order_quote_failure_payload
from .bot_models import _BotInstance, _BotOrder


class BotOrderBuilderMixin:
    async def _create_order_for_instance_exit(
        self,
        *,
        instance: _BotInstance,
        instrument: str,
        symbol: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
        intent_clean: str,
        mode: str,
        leg_price,
        fail,
        set_status,
        order_journal,
        finalize_leg_orders,
    ) -> None:
        if instrument == "spot":
            _instrument, open_items, _open_dir = self._resolve_open_positions(
                instance,
                symbol=symbol,
            )
            open_item = open_items[0] if open_items else None
            if open_item is None:
                return fail(f"Exit: no spot position for {symbol}")
            try:
                pos = float(getattr(open_item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if not pos:
                return fail(f"Exit: no spot position for {symbol}")

            action = "SELL" if pos > 0 else "BUY"
            qty = int(abs(pos))
            if qty <= 0:
                return fail(f"Exit: invalid position size for {symbol}")

            contract = open_item.contract
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None),
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
            )
            limit = leg_price(bid, ask, last, action)
            if limit is None:
                return fail(
                    "Quote: no bid/ask/last (cannot price)",
                    quote_payload=order_quote_failure_payload(
                        ticker=ticker,
                        bid=bid,
                        ask=ask,
                        last=last,
                        proxy_error=self._client.proxy_error(),
                    ),
                )
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[QualifiedOptionLeg(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=_now_et(),
                bid=bid,
                ask=ask,
                last=last,
                intent=intent_clean,
                direction=direction,
                reason="exit",
                signal_bar_ts=signal_bar_ts,
                journal=dict(order_journal),
                exec_mode=mode,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            self._add_order(order)
            set_status(f"Created order EXIT {action} {qty} {symbol} @ {limit:.2f}")
            return

        _instrument, open_items, _open_dir = self._resolve_open_positions(
            instance,
            symbol=symbol,
        )
        if not open_items:
            return fail(f"Exit: no option positions for instance {instance.instance_id}")
        underlying = Stock(symbol=symbol, exchange="SMART", currency="USD")
        qualified = await self._client.qualify_proxy_contracts(underlying)
        if qualified:
            underlying = qualified[0]

        try:
            leg_orders, package_quantity = normalize_option_position_close(
                tuple(
                    (item.contract, getattr(item, "position", 0.0))
                    for item in open_items
                )
            )
        except ValueError as exc:
            return fail(f"Exit: {exc}")

        leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
        for leg in leg_orders:
            con_id = int(getattr(leg.contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(leg.contract, owner="bot")
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None),
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
            )
            leg_quotes.append((bid, ask, last, ticker))

        finalize_leg_orders(
            underlying=underlying,
            leg_orders=list(leg_orders),
            leg_quotes=leg_quotes,
            package_quantity=package_quantity,
        )

    async def _create_order_for_instance(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        intent_clean = str(intent or "enter").strip().lower() or "enter"
        if intent_clean not in ("enter", "exit", "resize"):
            symbol = str(
                instance.symbol
                or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
            ).strip().upper()
            self._status = f"Unsupported order intent: {intent_clean}"
            self._render_status()
            instance.order_trigger_last_error = None
            instance.order_trigger_retry_reason = None
            clear_order_watch = getattr(self, "_clear_order_trigger_watch", None)
            if callable(clear_order_watch):
                clear_order_watch(instance)
            self._journal_write(
                event="ORDER_SKIPPED",
                instance=instance,
                order=None,
                reason=intent_clean,
                data={
                    "skip_reason": "intent_unsupported",
                    "intent": intent_clean,
                    "symbol": symbol,
                },
            )
            return

        strat = instance.strategy or {}
        instrument = self._strategy_instrument(strat)
        symbol = str(
            instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()
        if intent_clean == "resize" and instrument != "spot":
            self._status = "Resize not supported for options"
            self._render_status()
            instance.order_trigger_last_error = None
            instance.order_trigger_retry_reason = None
            clear_order_watch = getattr(self, "_clear_order_trigger_watch", None)
            if callable(clear_order_watch):
                clear_order_watch(instance)
            self._journal_write(
                event="ORDER_SKIPPED",
                instance=instance,
                order=None,
                reason="resize",
                data={
                    "skip_reason": "resize_options_unsupported",
                    "symbol": symbol,
                },
            )
            return

        entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
        ema_preset = str(strat.get("ema_preset") or "").strip()
        if entry_signal == "ema" and not ema_preset:
            raise RuntimeError(
                "FATAL: missing required strategy field `ema_preset` "
                f"(instance_id={instance.instance_id} group={instance.group!r} symbol={symbol!r})"
            )

        # All live execution uses the shared execution ladder (optimistic -> mid -> aggressive -> cross).
        # For repeated stop exits, start one rung higher so risk-off retries become marketable faster.
        mode = initial_execution_mode(
            instrument=instrument,
            intent=intent_clean,
            trigger_reason=instance.order_trigger_reason,
            exit_retry_count=instance.exit_retry_count,
        )

        def _leg_price(
            bid: float | None, ask: float | None, last: float | None, action: str
        ) -> float | None:
            return _limit_price_for_mode(bid, ask, last, action=action, mode=mode)

        def _bump_entry_counters() -> None:
            self._reset_daily_counters_if_needed(instance)
            instance.entries_today += 1

        def _set_status(message: str) -> None:
            self._status = message
            self._render_status()

        def _fail(
            message: str,
            *,
            quote_payload: dict[str, object] | None = None,
            retry_reason: str | None = None,
        ) -> None:
            _set_status(message)
            payload = order_build_failure_payload(
                message,
                instance,
                direction=direction,
                signal_bar_ts=signal_bar_ts,
                retry_reason=retry_reason,
                quote_payload=quote_payload,
            )
            reason_clean = str(payload["retry_reason"])
            instance.order_trigger_last_error = str(message or "")
            instance.order_trigger_retry_reason = reason_clean
            self._journal_write(
                event="ORDER_BUILD_FAILED",
                instance=instance,
                order=None,
                reason=intent_clean,
                data=payload,
            )

        def _finalize_leg_orders(
            *,
            underlying: Contract,
            leg_orders: list[QualifiedOptionLeg],
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]],
            package_quantity: int,
            entry_intent=None,
        ) -> None:
            if not leg_orders:
                return _fail("Order: no legs configured")

            order_quote = quote_live_option_order(
                symbol=symbol,
                legs=leg_orders,
                tickers=[quote[3] for quote in leg_quotes],
                quantity=(
                    package_quantity
                    if len(leg_orders) > 1
                    else leg_orders[0].ratio * package_quantity
                ),
                intent=intent_clean,
                mode=mode,
            )
            if order_quote is None:
                if len(leg_orders) == 1:
                    bid, ask, last, ticker = leg_quotes[0]
                    return _fail(
                        "Quote: no bid/ask/last (cannot price)",
                        quote_payload=order_quote_failure_payload(
                            ticker=ticker,
                            bid=bid,
                            ask=ask,
                            last=last,
                            mid=_midpoint(bid, ask),
                            proxy_error=self._client.proxy_error(),
                        ),
                    )
                return _fail(
                    "Quote: package legs are not jointly executable",
                    retry_reason="package_quote_unavailable",
                )

            order_limit = order_quote.limit_value
            tick = order_quote.tick
            if intent_clean == "enter" and order_limit < 0:
                if entry_intent is None:
                    return _fail("Order: option entry intent unavailable")
                credit = -float(order_limit)
                min_credit = entry_intent.required_credit(tick)
                if not entry_intent.admits_debit_value(order_limit, tick=tick):
                    return _fail(
                        f"Order: credit {credit:.2f} below minimum {min_credit:.2f}",
                        quote_payload={
                            "credit": float(credit),
                            "min_credit": float(min_credit),
                            "debit_value": float(order_limit),
                            "tick": float(tick),
                        },
                        retry_reason="minimum_credit_not_met",
                    )

            if (
                intent_clean in {"enter", "resize"}
                and symbol == "XSP"
                and order_quote.risk is not None
            ):
                reservation_summary = self._order_reservation_summary()
                capacity_decision = evaluate_order_reservation_capacity(
                    OrderReservationCapacityRequest(
                        account=reservation_summary.account,
                        product_domain=symbol,
                        sec_type=str(getattr(order_quote.contract, "secType", "") or ""),
                        structure=order_quote.risk.structure,
                        candidate_max_loss=order_quote.risk.max_loss,
                        available_capacity=strat.get(
                            "xsp_reservation_capacity_usd"
                        ),
                    ),
                    reservation_summary,
                )
                if not capacity_decision.allow:
                    return _fail(
                        f"Capacity: {capacity_decision.reason}",
                        retry_reason=capacity_decision.reason,
                    )

            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=underlying,
                order_contract=order_quote.contract,
                legs=list(order_quote.legs),
                action=order_quote.action,
                quantity=order_quote.quantity,
                limit_price=float(order_limit),
                created_at=_now_et(),
                package=order_quote.package,
                package_risk=order_quote.risk,
                bid=order_quote.bid_value,
                ask=order_quote.ask_value,
                last=order_quote.last_value,
                intent=intent_clean,
                direction=direction,
                reason=intent_clean,
                signal_bar_ts=signal_bar_ts,
                journal=order_attempt_payload(instance, required=True),
                exec_mode=mode,
            )
            for leg_order in leg_orders:
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
            self._add_order(order)
            if intent_clean == "enter":
                if direction in ("up", "down"):
                    instance.open_direction = str(direction)
                if signal_bar_ts is not None:
                    instance.last_entry_bar_ts = signal_bar_ts
                _bump_entry_counters()
            suffix = (
                f"BAG {symbol} @ {order_limit:.2f} ({len(leg_orders)} legs)"
                if len(leg_orders) > 1
                else f"{order_quote.quantity} {symbol} @ {order_limit:.2f}"
            )
            _set_status(f"Created order {order_quote.action} {suffix}")

        instrument_kind = "spot" if instrument == "spot" else "options"
        flow_key = (intent_clean, instrument_kind)
        flow = {
            ("exit", "spot"): "exit",
            ("exit", "options"): "exit",
            ("enter", "spot"): "spot_signal_sized",
            ("resize", "spot"): "spot_signal_sized",
            ("enter", "options"): "enter_options",
            ("resize", "options"): "enter_options",
        }.get(flow_key, "enter_options")

        if flow == "exit":
            await self._create_order_for_instance_exit(
                instance=instance,
                instrument=instrument,
                symbol=symbol,
                direction=direction,
                signal_bar_ts=signal_bar_ts,
                intent_clean=intent_clean,
                mode=mode,
                leg_price=_leg_price,
                fail=_fail,
                set_status=_set_status,
                order_journal=order_attempt_payload(instance, required=True),
                finalize_leg_orders=_finalize_leg_orders,
            )
            return

        if flow == "spot_signal_sized":
            is_resize = intent_clean == "resize"
            entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
            runtime_spec = spot_runtime_spec_view(strategy=strat, filters=instance.filters)
            exit_mode = str(runtime_spec.exit_mode)

            signal_contract = await self._signal_contract(instance, symbol)
            snap = (
                await self._signal_snapshot_for_contract(
                    contract=signal_contract,
                    **self._signal_snapshot_kwargs(
                        instance,
                        strategy=strat,
                        ema_preset_raw=str(strat.get("ema_preset")) if strat.get("ema_preset") else None,
                        entry_signal_raw=entry_signal,
                        include_orb=True,
                        include_spot_exit=True,
                    ),
                )
                if signal_contract is not None
                else None
            )
            if snap is None:
                return _fail(f"Signal: no snapshot for {symbol}")

            if direction not in ("up", "down") and snap is not None:
                direction = self._entry_direction_for_instance(instance, snap) or (
                    str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                )
            if direction not in ("up", "down") and is_resize:
                _instrument, open_items, open_dir = self._resolve_open_positions(
                    instance,
                    symbol=symbol,
                    signal_contract=signal_contract,
                )
                if open_items and open_dir in ("up", "down"):
                    direction = str(open_dir)
            direction = direction if direction in ("up", "down") else "up"

            resolved_entry = spot_resolve_entry_action_qty(
                strategy=strat,
                entry_dir=direction,
                needs_direction=False,
                fallback_short_sell=True,
            )
            if resolved_entry is None:
                return _fail(f"Order: invalid spot action for {direction}")
            action, qty = resolved_entry

            contract = await self._spot_contract(instance, symbol)
            if contract is None:
                return _fail(f"Contract: not found for {symbol}")

            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None),
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
            )
            limit = _leg_price(bid, ask, last, action)
            if limit is None:
                retry_window_ms_raw = strat.get("spot_quote_retry_window_ms", 600)
                retry_interval_ms_raw = strat.get("spot_quote_retry_interval_ms", 120)
                try:
                    retry_window_ms = max(0, int(retry_window_ms_raw or 0))
                except (TypeError, ValueError):
                    retry_window_ms = 600
                try:
                    retry_interval_ms = max(50, int(retry_interval_ms_raw or 0))
                except (TypeError, ValueError):
                    retry_interval_ms = 120
                retry_attempts = max(0, retry_window_ms // retry_interval_ms)
                for _ in range(retry_attempts):
                    await asyncio.sleep(retry_interval_ms / 1000.0)
                    bid, ask, last = _sanitize_nbbo(
                        getattr(ticker, "bid", None),
                        getattr(ticker, "ask", None),
                        getattr(ticker, "last", None),
                    )
                    limit = _leg_price(bid, ask, last, action)
                    if limit is not None:
                        break
            if limit is None:
                return _fail(
                    "Quote: no bid/ask/last (cannot price)",
                    quote_payload=order_quote_failure_payload(
                        ticker=ticker,
                        bid=bid,
                        ask=ask,
                        last=last,
                        proxy_error=self._client.proxy_error(),
                    ),
                )
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)

            if not is_resize:
                instance.spot_profit_target_price = None
                instance.spot_stop_loss_price = None
                if snap is not None and entry_signal == "orb":
                    try:
                        rr = float(strat.get("orb_risk_reward", 2.0) or 2.0)
                    except (TypeError, ValueError):
                        rr = 2.0
                    target_mode = str(strat.get("orb_target_mode", "rr") or "rr").strip().lower()
                    if target_mode not in ("rr", "or_range"):
                        target_mode = "rr"
                    or_high = snap.or_high
                    or_low = snap.or_low
                    if (
                        rr > 0
                        and bool(snap.or_ready)
                        and or_high is not None
                        and or_low is not None
                        and float(or_high) > 0
                        and float(or_low) > 0
                    ):
                        stop = float(or_low) if direction == "up" else float(or_high)
                        if target_mode == "or_range":
                            rng = float(or_high) - float(or_low)
                            if rng > 0:
                                target = (
                                    float(or_high) + (rr * rng)
                                    if direction == "up"
                                    else float(or_low) - (rr * rng)
                                )
                                if (
                                    (direction == "up" and float(target) <= float(limit))
                                    or (direction == "down" and float(target) >= float(limit))
                                ):
                                    return _fail("Order: ORB target already hit (skip)")
                                instance.spot_profit_target_price = float(target)
                                instance.spot_stop_loss_price = float(stop)
                        else:
                            risk = abs(float(limit) - stop)
                            if risk > 0:
                                target = (
                                    float(limit) + (rr * risk) if direction == "up" else float(limit) - (rr * risk)
                                )
                                instance.spot_profit_target_price = float(target)
                                instance.spot_stop_loss_price = float(stop)
                elif exit_mode == "atr":
                    atr = float(snap.atr) if snap is not None and snap.atr is not None else 0.0
                    if atr <= 0:
                        return _fail("Order: ATR not ready (spot_exit_mode=atr)")
                    pt_raw = strat.get("spot_pt_atr_mult", 1.5)
                    try:
                        pt_mult = float(1.5 if pt_raw is None else pt_raw)
                    except (TypeError, ValueError):
                        pt_mult = 1.5
                    sl_raw = strat.get("spot_sl_atr_mult", 1.0)
                    try:
                        sl_mult = float(1.0 if sl_raw is None else sl_raw)
                    except (TypeError, ValueError):
                        sl_mult = 1.0
                    if direction == "up":
                        instance.spot_profit_target_price = float(limit) + (pt_mult * atr)
                        instance.spot_stop_loss_price = float(limit) - (sl_mult * atr)
                    else:
                        instance.spot_profit_target_price = float(limit) - (pt_mult * atr)
                        instance.spot_stop_loss_price = float(limit) + (sl_mult * atr)

            # Spot sizing: mirror backtest semantics (fixed / notional_pct / risk_pct), with optional
            # shock/risk overlays applied via the filters snapshot.
            from ..engine import (
                spot_apply_branch_size_mult,
                spot_branch_size_mult,
                spot_calc_signed_qty_with_trace,
                spot_sizing_input,
                spot_scale_exit_pcts,
                spot_shock_exit_pct_multipliers,
            )

            filters = instance.filters if isinstance(instance.filters, dict) else None
            stop_loss_pct = None
            try:
                stop_loss_pct = (
                    float(strat.get("spot_stop_loss_pct"))
                    if strat.get("spot_stop_loss_pct") is not None
                    else None
                )
            except (TypeError, ValueError):
                stop_loss_pct = None
            if stop_loss_pct is not None and float(stop_loss_pct) <= 0:
                stop_loss_pct = None

            stop_price = instance.spot_stop_loss_price
            if stop_price is not None:
                try:
                    stop_price = float(stop_price)
                except (TypeError, ValueError):
                    stop_price = None
            if stop_price is not None and float(stop_price) <= 0:
                stop_price = None

            sl_mult, _ = spot_shock_exit_pct_multipliers(filters, shock=snap.shock)
            stop_loss_pct, _ = spot_scale_exit_pcts(
                stop_loss_pct=stop_loss_pct,
                profit_target_pct=None,
                stop_mult=sl_mult,
                profit_mult=1.0,
            )

            net_liq_val, net_liq_currency, _updated = self._client.account_value(
                "NetLiquidation",
                currency="USD",
            )
            if net_liq_val is None:
                net_liq_val, net_liq_currency, _updated = self._client.account_value("NetLiquidation")
            buying_power_val, buying_power_currency, _bp_updated = self._client.account_value(
                "BuyingPower",
                currency="USD",
            )
            if buying_power_val is None:
                buying_power_val, buying_power_currency, _bp_updated = self._client.account_value("BuyingPower")
            try:
                equity_ref = float(net_liq_val) if net_liq_val is not None else 0.0
            except (TypeError, ValueError):
                equity_ref = 0.0
            try:
                cash_ref = float(buying_power_val) if buying_power_val is not None else None
            except (TypeError, ValueError):
                cash_ref = None
            sizing_currency = str(getattr(contract, "currency", "") or "USD").strip().upper() or "USD"
            net_liq_fx_rate = None
            buying_power_fx_rate = None
            net_liq_currency_clean = str(net_liq_currency or "").strip().upper() if net_liq_currency is not None else None
            buying_power_currency_clean = (
                str(buying_power_currency or "").strip().upper() if buying_power_currency is not None else None
            )
            if (
                equity_ref > 0
                and net_liq_currency_clean
                and net_liq_currency_clean != sizing_currency
            ):
                converted_equity, fx_rate = await self._client.convert_currency_value(
                    float(equity_ref),
                    from_currency=net_liq_currency_clean,
                    to_currency=sizing_currency,
                )
                if converted_equity is None:
                    return _fail(
                        "Order: currency conversion unavailable for NetLiquidation "
                        f"{net_liq_currency_clean}->{sizing_currency}",
                        retry_reason="currency_conversion_unavailable",
                    )
                equity_ref = float(converted_equity)
                net_liq_fx_rate = float(fx_rate) if fx_rate is not None else None
                net_liq_currency = sizing_currency
            if (
                cash_ref is not None
                and cash_ref > 0
                and buying_power_currency_clean
                and buying_power_currency_clean != sizing_currency
            ):
                converted_cash, fx_rate = await self._client.convert_currency_value(
                    float(cash_ref),
                    from_currency=buying_power_currency_clean,
                    to_currency=sizing_currency,
                )
                if converted_cash is None:
                    return _fail(
                        "Order: currency conversion unavailable for BuyingPower "
                        f"{buying_power_currency_clean}->{sizing_currency}",
                        retry_reason="currency_conversion_unavailable",
                    )
                cash_ref = float(converted_cash)
                buying_power_fx_rate = float(fx_rate) if fx_rate is not None else None
                buying_power_currency = sizing_currency
            cash_ref = float(cash_ref) if cash_ref is not None else None

            riskoff = bool(snap.risk.riskoff) if snap.risk is not None else False
            riskpanic = bool(snap.risk.riskpanic) if snap.risk is not None else False
            riskpop = bool(snap.risk.riskpop) if snap.risk is not None else False

            sizing_input = spot_sizing_input(
                strategy=strat,
                filters=filters,
                action=str(action),
                lot=int(qty),
                entry_price=float(limit),
                stop_price=stop_price,
                stop_loss_pct=stop_loss_pct,
                shock=snap.shock,
                shock_dir=snap.shock_dir,
                shock_atr_pct=snap.shock_atr_pct,
                shock_dir_down_streak_bars=getattr(snap, "shock_dir_down_streak_bars", None),
                shock_drawdown_dist_on_pct=getattr(snap, "shock_drawdown_dist_on_pct", None),
                shock_drawdown_dist_on_vel_pp=getattr(snap, "shock_drawdown_dist_on_vel_pp", None),
                shock_drawdown_dist_on_accel_pp=getattr(snap, "shock_drawdown_dist_on_accel_pp", None),
                shock_prearm_down_streak_bars=getattr(snap, "shock_prearm_down_streak_bars", None),
                shock_ramp=getattr(snap, "shock_ramp", None),
                riskoff=riskoff,
                risk_dir=snap.shock_dir,
                riskpanic=riskpanic,
                riskpop=riskpop,
                risk=snap.risk,
                signal_entry_dir=getattr(getattr(snap, "signal", None), "entry_dir", None),
                signal_regime_dir=getattr(getattr(snap, "signal", None), "regime_dir", None),
                regime2_dir=(
                    str(getattr(snap, "regime2_dir"))
                    if getattr(snap, "regime2_dir", None) in ("up", "down")
                    else None
                ),
                regime2_ready=bool(getattr(snap, "regime2_ready", False)),
                equity_ref=float(equity_ref),
                cash_ref=cash_ref,
            )
            signed_qty, decision_trace = spot_calc_signed_qty_with_trace(sizing_input)
            if signed_qty == 0:
                return _fail(
                    "Order: spot sizing returned 0 qty",
                    quote_payload={"spot_decision": decision_trace.as_payload()},
                )

            entry_branch = str(instance.spot_entry_branch or "") if is_resize else str(getattr(snap, "entry_branch", "") or "")
            if entry_branch not in ("a", "b"):
                entry_branch = str(getattr(snap, "entry_branch", "") or "")
            branch_size_mult = spot_branch_size_mult(strategy=strat, entry_branch=entry_branch)
            signed_qty = spot_apply_branch_size_mult(
                signed_qty=int(signed_qty),
                size_mult=float(branch_size_mult),
                spot_min_qty=strat.get("spot_min_qty", 1),
                spot_max_qty=strat.get("spot_max_qty", 0),
            )
            decision_trace = decision_trace.with_branch_scaling(
                entry_branch=entry_branch,
                size_mult=float(branch_size_mult),
                signed_qty_after_branch=int(signed_qty),
            )

            current_qty = 0
            if is_resize:
                _instrument, open_items, _open_dir = self._resolve_open_positions(
                    instance,
                    symbol=symbol,
                    signal_contract=signal_contract,
                )
                open_item = open_items[0] if open_items else None
                if open_item is None:
                    return _fail("Resize: no spot position", retry_reason="resize_no_position")
                try:
                    current_qty = int(float(getattr(open_item, "position", 0.0) or 0.0))
                except (TypeError, ValueError):
                    current_qty = 0
                if current_qty == 0:
                    return _fail("Resize: no spot position", retry_reason="resize_no_position")

            lifecycle_decision = decide_open_position_intent(
                strategy=strat,
                bar_ts=snap.bar_ts if snap is not None else (_now_et_naive()),
                bar_size=self._signal_bar_size(instance),
                open_dir=direction,
                current_qty=int(current_qty),
                target_qty=int(signed_qty),
                spot_decision=decision_trace.as_payload(),
                last_resize_bar_ts=instance.last_resize_bar_ts,
                signal_entry_dir=getattr(getattr(snap, "signal", None), "entry_dir", None),
                shock_atr_pct=float(snap.shock_atr_pct) if snap is not None and snap.shock_atr_pct is not None else None,
                shock_atr_vel_pct=(
                    float(getattr(snap, "shock_atr_vel_pct", 0.0))
                    if snap is not None and getattr(snap, "shock_atr_vel_pct", None) is not None
                    else None
                ),
                shock_atr_accel_pct=(
                    float(getattr(snap, "shock_atr_accel_pct", 0.0))
                    if snap is not None and getattr(snap, "shock_atr_accel_pct", None) is not None
                    else None
                ),
                tr_ratio=float(getattr(snap, "ratsv_tr_ratio", 0.0)) if snap is not None and getattr(snap, "ratsv_tr_ratio", None) is not None else None,
                slope_med_pct=float(getattr(snap, "ratsv_fast_slope_med_pct", 0.0)) if snap is not None and getattr(snap, "ratsv_fast_slope_med_pct", None) is not None else None,
                slope_vel_pct=float(getattr(snap, "ratsv_fast_slope_vel_pct", 0.0)) if snap is not None and getattr(snap, "ratsv_fast_slope_vel_pct", None) is not None else None,
                slope_med_slow_pct=(
                    float(getattr(snap, "ratsv_slow_slope_med_pct", 0.0))
                    if snap is not None and getattr(snap, "ratsv_slow_slope_med_pct", None) is not None
                    else None
                ),
                slope_vel_slow_pct=(
                    float(getattr(snap, "ratsv_slow_slope_vel_pct", 0.0))
                    if snap is not None and getattr(snap, "ratsv_slow_slope_vel_pct", None) is not None
                    else None
                ),
            )
            intent_decision = lifecycle_decision.spot_intent
            if (
                str(lifecycle_decision.intent) == "hold"
                or intent_decision is None
                or int(intent_decision.order_qty) <= 0
                or str(intent_decision.order_action or "") not in ("BUY", "SELL")
            ):
                self._journal_write(
                    event="ORDER_SKIPPED",
                    instance=instance,
                    order=None,
                    reason=intent_clean,
                    data={
                        "skip_reason": str(lifecycle_decision.reason or "hold"),
                        "direction": direction,
                        "signal_bar_ts": signal_bar_ts.isoformat() if signal_bar_ts is not None else None,
                        "spot_lifecycle": lifecycle_decision.as_payload(),
                        "spot_intent": intent_decision.as_payload() if intent_decision is not None else None,
                        "spot_decision": decision_trace.as_payload(),
                        "size_funnel": {
                            "signed_qty_final": int(getattr(decision_trace, "signed_qty_final", 0)),
                            "signed_qty_after_branch": int(
                                getattr(
                                    decision_trace,
                                    "signed_qty_after_branch",
                                    getattr(decision_trace, "signed_qty_final", 0),
                                )
                            ),
                            "resize_target_qty": int(getattr(intent_decision, "target_qty", 0))
                            if intent_decision is not None
                            else 0,
                            "intent_qty": int(getattr(intent_decision, "order_qty", 0))
                            if intent_decision is not None
                            else 0,
                        },
                        **order_attempt_payload(instance, required=True),
                    },
                )
                clear_order_watch = getattr(self, "_clear_order_trigger_watch", None)
                if callable(clear_order_watch):
                    clear_order_watch(instance)
                instance.order_trigger_last_error = None
                instance.order_trigger_retry_reason = None
                _set_status(f"Order skipped ({intent_clean}): {lifecycle_decision.reason or 'hold'}")
                return

            action = str(intent_decision.order_action)
            qty = int(intent_decision.order_qty)
            if int(intent_decision.target_qty) > 0:
                direction = "up"
            elif int(intent_decision.target_qty) < 0:
                direction = "down"

            journal = project_live_spot_order_journal(
                snapshot=snap,
                intent=intent_clean,
                direction=direction,
                entry_branch=entry_branch,
                branch_size_mult=branch_size_mult,
                sizing=decision_trace,
                lifecycle=lifecycle_decision,
                spot_intent=intent_decision,
                exit_mode=exit_mode,
                stop_loss_pct=stop_loss_pct,
                stop_price=stop_price,
                target_price=instance.spot_profit_target_price,
                sizing_currency=sizing_currency,
                net_liq=equity_ref,
                net_liq_currency=net_liq_currency,
                net_liq_fx_rate=net_liq_fx_rate,
                buying_power=cash_ref,
                buying_power_currency=buying_power_currency,
                buying_power_fx_rate=buying_power_fx_rate,
                chase_orders=bool(strat.get("chase_orders", True)),
            )
            journal.update(order_attempt_payload(instance, required=True))

            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[QualifiedOptionLeg(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=_now_et(),
                bid=bid,
                ask=ask,
                last=last,
                intent=intent_clean,
                direction=direction,
                reason=intent_clean,
                signal_bar_ts=snap.bar_ts if snap is not None else signal_bar_ts,
                journal=journal,
                exec_mode=mode,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            instance.open_direction = str(direction)
            if signal_bar_ts is not None and intent_clean == "enter":
                instance.last_entry_bar_ts = signal_bar_ts
            if intent_clean == "enter":
                _bump_entry_counters()
            self._add_order(order)
            _set_status(f"Created order {action} {qty} {symbol} @ {limit:.2f} ({direction})")
            return

        legs_raw: object = None
        legs_path = "legs"
        if isinstance(strat.get("directional_legs"), dict):
            dmap = strat.get("directional_legs") or {}
            if direction not in ("up", "down"):
                ema_preset = strat.get("ema_preset")
                if ema_preset:
                    signal_contract = await self._signal_contract(instance, symbol)
                    snap = (
                        await self._signal_snapshot_for_contract(
                            contract=signal_contract,
                            **self._signal_snapshot_kwargs(
                                instance,
                                strategy=strat,
                                ema_preset_raw=str(ema_preset),
                            ),
                        )
                        if signal_contract is not None
                        else None
                    )
                    if snap is not None:
                        direction = self._entry_direction_for_instance(instance, snap) or (
                            str(snap.signal.state) if snap.signal.state in ("up", "down") else None
                        )
            if direction in ("up", "down") and dmap.get(direction):
                legs_raw = dmap.get(direction)
                legs_path = f"directional_legs.{direction}"
            else:
                for key in ("up", "down"):
                    if dmap.get(key):
                        legs_raw = dmap.get(key)
                        legs_path = f"directional_legs.{key}"
                        direction = key
                        break

        if legs_raw is None:
            raw = strat.get("legs", []) or []
            legs_raw = raw if isinstance(raw, list) else []

        try:
            entry_intent = option_package_entry_intent(
                strat,
                legs=legs_raw,
                path=legs_path,
            )
        except ValueError as exc:
            return _fail(f"Order: {exc}")
        try:
            resolved = await resolve_live_option_entry(
                self._client,
                symbol=symbol,
                intent=entry_intent,
                anchor=_now_et().date(),
                owner="bot",
            )
        except ValueError as exc:
            return _fail(str(exc))

        for contract in (resolved.underlying, *(leg.contract for leg in resolved.legs)):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)

        _finalize_leg_orders(
            underlying=resolved.underlying,
            leg_orders=list(resolved.legs),
            leg_quotes=[
                (*_sanitize_nbbo(ticker.bid, ticker.ask, ticker.last), ticker)
                for ticker in resolved.tickers
            ],
            package_quantity=entry_intent.quantity,
            entry_intent=entry_intent,
        )
