"""Bot order construction mixin."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

from ib_insync import Bag, ComboLeg, Contract, Option, Stock, Ticker

from ..engine import normalize_spot_entry_signal
from ..utils.date_utils import add_business_days
from .bot_models import _BotInstance, _BotLegOrder, _BotOrder
from .common import (
    _limit_price_for_mode,
    _midpoint,
    _round_to_tick,
    _safe_num,
    _tick_size,
    _ticker_price,
)


def _pick_chain_expiry(today: date, dte: int, expirations: list[str]) -> str | None:
    if not expirations:
        return None
    target = add_business_days(today, dte)
    parsed: list[tuple[date, str]] = []
    for exp in expirations:
        dt = _parse_chain_date(exp)
        if dt:
            parsed.append((dt, exp))
    if not parsed:
        return None
    future = [(dt, exp) for dt, exp in parsed if dt >= target]
    candidates = future or parsed
    candidates.sort(key=lambda pair: abs((pair[0] - target).days))
    return candidates[0][1]


def _parse_chain_date(raw: str) -> date | None:
    raw = str(raw).strip()
    if len(raw) != 8 or not raw.isdigit():
        return None
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def _strike_from_moneyness(spot: float, right: str, moneyness_pct: float) -> float:
    # Negative moneyness means ITM (e.g., -1 = 1% ITM).
    if right == "PUT":
        return spot * (1 - moneyness_pct / 100.0)
    return spot * (1 + moneyness_pct / 100.0)


def _nearest_strike(strikes: list[float], target: float) -> float | None:
    if not strikes:
        return None
    try:
        return min((float(s) for s in strikes), key=lambda s: abs(s - target))
    except (TypeError, ValueError):
        return None


class BotOrderBuilderMixin:
    async def _strike_by_delta(
        self,
        *,
        symbol: str,
        expiry: str,
        right_char: str,
        strikes: list[float],
        trading_class: str | None,
        near_strike: float,
        target_delta: float,
    ) -> float | None:
        try:
            target = abs(float(target_delta))
        except (TypeError, ValueError):
            return None
        if target <= 0 or target > 1:
            return None
        try:
            strike_values = sorted(float(s) for s in strikes)
        except (TypeError, ValueError):
            return None
        if not strike_values:
            return None
        center_idx = min(
            range(len(strike_values)), key=lambda idx: abs(strike_values[idx] - near_strike)
        )
        window = strike_values[max(center_idx - 10, 0) : center_idx + 11]
        if not window:
            return None

        candidates = [
            Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry,
                strike=float(strike),
                right=right_char,
                exchange="SMART",
                currency="USD",
                tradingClass=trading_class,
            )
            for strike in window
        ]
        qualified = await self._client.qualify_proxy_contracts(*candidates)
        if qualified and len(qualified) == len(candidates):
            contracts: list[Contract] = list(qualified)
        else:
            contracts = list(candidates)

        best_strike: float | None = None
        best_diff: float | None = None
        for contract, strike in zip(contracts, window):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            delta = None
            for _ in range(6):
                for attr in ("modelGreeks", "bidGreeks", "askGreeks", "lastGreeks"):
                    greeks = getattr(ticker, attr, None)
                    if greeks is not None:
                        raw = getattr(greeks, "delta", None)
                        if raw is not None:
                            try:
                                delta = float(raw)
                            except (TypeError, ValueError):
                                delta = None
                            break
                if delta is not None:
                    break
                await asyncio.sleep(0.05)
            if delta is None:
                continue
            diff = abs(abs(delta) - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_strike = float(strike)

        # Avoid keeping a large number of quote subscriptions just to pick strike.
        for contract, strike in zip(contracts, window):
            if best_strike is not None and float(strike) == best_strike:
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._client.release_ticker(con_id, owner="bot")
                self._tracked_conids.discard(con_id)
        return best_strike

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
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = leg_price(bid, ask, last, action)
            if limit is None:
                return fail("Quote: no bid/ask/last (cannot price)")
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=bid,
                ask=ask,
                last=last,
                intent=intent_clean,
                direction=direction,
                reason="exit",
                signal_bar_ts=signal_bar_ts,
                exec_mode=mode,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            self._add_order(order)
            if signal_bar_ts is not None:
                instance.last_exit_bar_ts = signal_bar_ts
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

        leg_orders: list[_BotLegOrder] = []
        leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
        for item in open_items:
            contract = item.contract
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not pos:
                continue
            ratio = int(abs(pos))
            if ratio <= 0:
                continue
            action = "SELL" if pos > 0 else "BUY"
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
            leg_quotes.append((bid, ask, last, ticker))

        if not leg_orders:
            return fail(f"Exit: no option positions for instance {instance.instance_id}")
        finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)

    async def _create_order_for_instance(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        strat = instance.strategy or {}
        instrument = self._strategy_instrument(strat)
        intent_clean = str(intent or "enter").strip().lower()
        intent_clean = "exit" if intent_clean == "exit" else "enter"
        symbol = str(
            instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
        ).strip().upper()

        entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
        ema_preset = str(strat.get("ema_preset") or "").strip()
        if entry_signal == "ema" and not ema_preset:
            raise RuntimeError(
                "FATAL: missing required strategy field `ema_preset` "
                f"(instance_id={instance.instance_id} group={instance.group!r} symbol={symbol!r})"
            )

        # All live execution uses the shared "execution ladder" (optimistic → mid → aggressive → cross).
        # Orders always start at the optimistic phase; the chase loop handles escalation.
        mode = "OPTIMISTIC"

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

        def _fail(message: str) -> None:
            _set_status(message)

        def _finalize_leg_orders(
            *,
            underlying: Contract,
            leg_orders: list[_BotLegOrder],
            leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]],
        ) -> None:
            if not leg_orders:
                return _fail("Order: no legs configured")

            # Compute net price in "debit units": BUY adds, SELL subtracts.
            debit_mid = 0.0
            debit_bid = 0.0
            debit_ask = 0.0
            desired_debit = 0.0
            tick = None
            for leg_order, (bid, ask, last, ticker) in zip(leg_orders, leg_quotes):
                mid = _midpoint(bid, ask)
                leg_mid = mid or last
                if leg_mid is None:
                    return _fail("Quote: missing mid/last (cannot price)")
                leg_bid = bid or mid or last
                leg_ask = ask or mid or last
                if leg_bid is None or leg_ask is None:
                    return _fail("Quote: missing bid/ask (cannot price)")
                leg_desired = _leg_price(bid, ask, last, leg_order.action)
                if leg_desired is None:
                    return _fail("Quote: missing bid/ask/last (cannot price)")
                leg_tick = _tick_size(leg_order.contract, ticker, leg_desired)
                tick = leg_tick if tick is None else min(tick, leg_tick)
                sign = 1.0 if leg_order.action == "BUY" else -1.0
                debit_mid += sign * float(leg_mid) * leg_order.ratio
                debit_bid += sign * float(leg_bid) * leg_order.ratio
                debit_ask += sign * float(leg_ask) * leg_order.ratio
                desired_debit += sign * float(leg_desired) * leg_order.ratio

            tick = tick or 0.01

            if len(leg_orders) == 1:
                single = leg_orders[0]
                (bid, ask, last, ticker) = leg_quotes[0]
                limit = _leg_price(bid, ask, last, single.action)
                if limit is None:
                    return _fail("Quote: no bid/ask/last (cannot price)")
                limit = _round_to_tick(float(limit), tick)
                order = _BotOrder(
                    instance_id=instance.instance_id,
                    preset=None,
                    underlying=underlying,
                    order_contract=single.contract,
                    legs=leg_orders,
                    action=single.action,
                    quantity=single.ratio,
                    limit_price=float(limit),
                    created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                    bid=bid,
                    ask=ask,
                    last=last,
                    intent=intent_clean,
                    direction=direction,
                    reason=intent_clean,
                    signal_bar_ts=signal_bar_ts,
                    exec_mode=mode,
                )
                con_id = int(getattr(single.contract, "conId", 0) or 0)
                if con_id:
                    instance.touched_conids.add(con_id)
                self._add_order(order)
                if intent_clean == "enter":
                    if direction in ("up", "down"):
                        instance.open_direction = str(direction)
                    if signal_bar_ts is not None:
                        instance.last_entry_bar_ts = signal_bar_ts
                    _bump_entry_counters()
                elif signal_bar_ts is not None:
                    instance.last_exit_bar_ts = signal_bar_ts
                _set_status(f"Created order {single.action} {single.ratio} {symbol} @ {limit:.2f}")
                return

            # Multi-leg combo: use IBKR's native encoding (can be negative for credits).
            order_action = "BUY"
            order_bid = debit_bid
            order_ask = debit_ask
            order_last = debit_mid
            order_limit = _round_to_tick(float(desired_debit), tick)
            if not order_limit:
                return _fail("Quote: combo price is 0 (cannot price)")
            combo_legs: list[ComboLeg] = []
            for leg_order, (_, _, _, ticker) in zip(leg_orders, leg_quotes):
                con_id = int(getattr(leg_order.contract, "conId", 0) or 0)
                if not con_id:
                    return _fail("Contract: missing conId for combo leg")
                leg_exchange = (
                    getattr(getattr(ticker, "contract", None), "exchange", "") or ""
                ).strip()
                if not leg_exchange:
                    leg_exchange = (getattr(leg_order.contract, "exchange", "") or "").strip()
                if not leg_exchange:
                    leg_sec_type = str(getattr(leg_order.contract, "secType", "") or "").strip()
                    leg_exchange = "CME" if leg_sec_type == "FOP" else "SMART"
                combo_legs.append(
                    ComboLeg(
                        conId=con_id,
                        ratio=leg_order.ratio,
                        action=leg_order.action,
                        exchange=leg_exchange,
                    )
                )
            bag = Bag(symbol=symbol, exchange="SMART", currency="USD", comboLegs=combo_legs)

            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=underlying,
                order_contract=bag,
                legs=leg_orders,
                action=order_action,
                quantity=1,
                limit_price=float(order_limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=order_bid,
                ask=order_ask,
                last=order_last,
                intent=intent_clean,
                direction=direction,
                reason=intent_clean,
                signal_bar_ts=signal_bar_ts,
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
            elif signal_bar_ts is not None:
                instance.last_exit_bar_ts = signal_bar_ts
            _set_status(f"Created order {order_action} BAG {symbol} @ {order_limit:.2f} ({len(leg_orders)} legs)")

        if intent_clean == "exit":
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
                finalize_leg_orders=_finalize_leg_orders,
            )
            return

        if instrument == "spot":
            entry_signal = normalize_spot_entry_signal(strat.get("entry_signal"))
            exit_mode = str(strat.get("spot_exit_mode") or "pct").strip().lower()
            if exit_mode not in ("pct", "atr"):
                exit_mode = "pct"

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
            direction = direction if direction in ("up", "down") else "up"

            mapping = strat.get("directional_spot") if isinstance(strat.get("directional_spot"), dict) else None
            chosen = mapping.get(direction) if mapping else None
            if not isinstance(chosen, dict):
                if direction == "up":
                    chosen = {"action": "BUY", "qty": 1}
                else:
                    chosen = {"action": "SELL", "qty": 1}
            action = str(chosen.get("action", "")).strip().upper()
            if action not in ("BUY", "SELL"):
                return _fail(f"Order: invalid spot action for {direction}")
            try:
                qty = int(chosen.get("qty", 1) or 1)
            except (TypeError, ValueError):
                qty = 1
            qty = max(1, abs(qty))

            contract = await self._spot_contract(instance, symbol)
            if contract is None:
                return _fail(f"Contract: not found for {symbol}")

            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            limit = _leg_price(bid, ask, last, action)
            if limit is None:
                return _fail("Quote: no bid/ask/last (cannot price)")
            tick = _tick_size(contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)

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
                            target = float(limit) + (rr * risk) if direction == "up" else float(limit) - (rr * risk)
                            instance.spot_profit_target_price = float(target)
                            instance.spot_stop_loss_price = float(stop)
            elif exit_mode == "atr":
                atr = float(snap.atr) if snap is not None and snap.atr is not None else 0.0
                if atr <= 0:
                    return _fail("Order: ATR not ready (spot_exit_mode=atr)")
                try:
                    pt_mult = float(strat.get("spot_pt_atr_mult", 1.5) or 1.5)
                except (TypeError, ValueError):
                    pt_mult = 1.5
                try:
                    sl_mult = float(strat.get("spot_sl_atr_mult", 1.0) or 1.0)
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
            from ..engine import spot_calc_signed_qty

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

            shock_now = bool(snap.shock) if snap.shock is not None else False
            if shock_now and filters is not None:
                try:
                    sl_mult = float(filters.get("shock_stop_loss_pct_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    sl_mult = 1.0
                if sl_mult > 0 and stop_loss_pct is not None and float(stop_loss_pct) > 0:
                    stop_loss_pct = min(float(stop_loss_pct) * float(sl_mult), 0.99)

            net_liq_val, _currency, _updated = self._client.account_value("NetLiquidation")
            buying_power_val, _bp_currency, _bp_updated = self._client.account_value("BuyingPower")
            try:
                equity_ref = float(net_liq_val) if net_liq_val is not None else 0.0
            except (TypeError, ValueError):
                equity_ref = 0.0
            try:
                cash_ref = float(buying_power_val) if buying_power_val is not None else None
            except (TypeError, ValueError):
                cash_ref = None

            riskoff = bool(snap.risk.riskoff) if snap.risk is not None else False
            riskpanic = bool(snap.risk.riskpanic) if snap.risk is not None else False
            riskpop = bool(snap.risk.riskpop) if snap.risk is not None else False

            signed_qty = spot_calc_signed_qty(
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
                riskoff=riskoff,
                risk_dir=snap.shock_dir,
                riskpanic=riskpanic,
                riskpop=riskpop,
                risk=snap.risk,
                equity_ref=float(equity_ref),
                cash_ref=cash_ref,
            )
            if signed_qty == 0:
                return _fail("Order: spot sizing returned 0 qty")
            action = "BUY" if int(signed_qty) > 0 else "SELL"
            qty = int(abs(int(signed_qty)))

            journal = {
                "intent": intent_clean,
                "direction": direction,
                "bar_ts": snap.bar_ts.isoformat() if snap is not None else None,
                "close": float(snap.close) if snap is not None else None,
                "signal": {
                    "state": getattr(getattr(snap, "signal", None), "state", None),
                    "entry_dir": getattr(getattr(snap, "signal", None), "entry_dir", None),
                    "regime_dir": getattr(getattr(snap, "signal", None), "regime_dir", None),
                    "ema_ready": bool(getattr(getattr(snap, "signal", None), "ema_ready", False)),
                },
                "bars_in_day": int(snap.bars_in_day) if snap is not None else None,
                "rv": float(snap.rv) if snap is not None and snap.rv is not None else None,
                "volume": float(snap.volume) if snap is not None and snap.volume is not None else None,
                "shock": bool(snap.shock) if snap is not None and snap.shock is not None else None,
                "shock_dir": snap.shock_dir if snap is not None else None,
                "shock_atr_pct": float(snap.shock_atr_pct)
                if snap is not None and snap.shock_atr_pct is not None
                else None,
                "riskoff": bool(snap.risk.riskoff) if snap is not None and snap.risk is not None else None,
                "riskpanic": bool(snap.risk.riskpanic) if snap is not None and snap.risk is not None else None,
                "atr": float(snap.atr) if snap is not None and snap.atr is not None else None,
                "or_high": float(snap.or_high) if snap is not None and snap.or_high is not None else None,
                "or_low": float(snap.or_low) if snap is not None and snap.or_low is not None else None,
                "or_ready": bool(snap.or_ready) if snap is not None else None,
                "exit_mode": exit_mode,
                "stop_loss_pct": float(stop_loss_pct) if stop_loss_pct is not None else None,
                "stop_price": float(stop_price) if stop_price is not None else None,
                "target_price": float(instance.spot_profit_target_price)
                if instance.spot_profit_target_price is not None
                else None,
                "net_liq": float(equity_ref) if equity_ref is not None else None,
                "buying_power": float(cash_ref) if cash_ref is not None else None,
                "exec_policy": "LADDER",
                "exec_mode": "OPTIMISTIC",
                "chase_orders": bool(strat.get("chase_orders", True)),
            }

            order = _BotOrder(
                instance_id=instance.instance_id,
                preset=None,
                underlying=contract,
                order_contract=contract,
                legs=[_BotLegOrder(contract=contract, action=action, ratio=qty)],
                action=action,
                quantity=qty,
                limit_price=float(limit),
                created_at=datetime.now(tz=ZoneInfo("America/New_York")),
                bid=bid,
                ask=ask,
                last=last,
                intent=intent_clean,
                direction=direction,
                reason="enter",
                signal_bar_ts=snap.bar_ts if snap is not None else signal_bar_ts,
                journal=journal,
                exec_mode=mode,
            )
            if con_id:
                instance.touched_conids.add(con_id)
            instance.open_direction = str(direction)
            if signal_bar_ts is not None:
                instance.last_entry_bar_ts = signal_bar_ts
            _bump_entry_counters()
            self._add_order(order)
            _set_status(f"Created order {action} {qty} {symbol} @ {limit:.2f} ({direction})")
            return

        legs_raw: list[dict] | None = None
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
            else:
                for key in ("up", "down"):
                    if dmap.get(key):
                        legs_raw = dmap.get(key)
                        direction = key
                        break

        if legs_raw is None:
            raw = strat.get("legs", []) or []
            legs_raw = raw if isinstance(raw, list) else []

        if not isinstance(legs_raw, list) or not legs_raw:
            return _fail("Order: no legs configured")

        dte_raw = strat.get("dte", 0)
        try:
            dte = int(dte_raw or 0)
        except (TypeError, ValueError):
            dte = 0

        chain_info = await self._client.stock_option_chain(symbol)
        if not chain_info:
            return _fail(f"Chain: not found for {symbol}")
        underlying, chain = chain_info
        underlying_ticker = await self._client.ensure_ticker(underlying, owner="bot")
        under_con_id = int(getattr(underlying, "conId", 0) or 0)
        if under_con_id:
            self._tracked_conids.add(under_con_id)
        spot = _ticker_price(underlying_ticker)
        if spot is None:
            return _fail(f"Spot: n/a for {symbol}")

        expiry = _pick_chain_expiry(date.today(), dte, getattr(chain, "expirations", []))
        if not expiry:
            return _fail(f"Expiry: none for {symbol}")

        # Build and qualify option legs.
        strikes = getattr(chain, "strikes", [])
        trading_class = getattr(chain, "tradingClass", None)
        option_candidates: list[Option] = []
        leg_specs: list[tuple[str, str, int, float, float | None]] = []
        for leg_raw in legs_raw:
            if not isinstance(leg_raw, dict):
                return _fail("Order: invalid leg config")
            action = str(leg_raw.get("action", "")).upper()
            right = str(leg_raw.get("right", "")).upper()
            if action not in ("BUY", "SELL") or right not in ("PUT", "CALL"):
                return _fail("Order: invalid leg config")
            try:
                ratio = int(leg_raw.get("qty", 1) or 1)
            except (TypeError, ValueError):
                ratio = 1
            ratio = max(1, abs(ratio))
            try:
                moneyness = float(leg_raw.get("moneyness_pct", 0.0) or 0.0)
            except (TypeError, ValueError):
                moneyness = 0.0
            delta_target = leg_raw.get("delta")
            try:
                delta_target = float(delta_target) if delta_target is not None else None
            except (TypeError, ValueError):
                delta_target = None

            target_strike = _strike_from_moneyness(spot, right, moneyness)
            right_char = "P" if right == "PUT" else "C"
            strike = None
            if delta_target is not None and strikes:
                strike = await self._strike_by_delta(
                    symbol=symbol,
                    expiry=expiry,
                    right_char=right_char,
                    strikes=list(strikes),
                    trading_class=trading_class,
                    near_strike=target_strike,
                    target_delta=delta_target,
                )
            if strike is None:
                strike = _nearest_strike(strikes, target_strike)
            if strike is None:
                return _fail(f"Strike: none for {symbol}")
            option_candidates.append(
                Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(strike),
                    right=right_char,
                    exchange="SMART",
                    currency="USD",
                    tradingClass=trading_class,
                )
            )
            leg_specs.append((action, right, ratio, moneyness, delta_target))

        qualified = await self._client.qualify_proxy_contracts(*option_candidates)
        if qualified and len(qualified) == len(option_candidates):
            option_contracts: list[Contract] = list(qualified)
        else:
            option_contracts = list(option_candidates)

        leg_orders: list[_BotLegOrder] = []
        leg_quotes: list[tuple[float | None, float | None, float | None, Ticker]] = []
        for contract, (action, _, ratio, _, _) in zip(option_contracts, leg_specs):
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id:
                self._tracked_conids.add(con_id)
            ticker = await self._client.ensure_ticker(contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            leg_orders.append(_BotLegOrder(contract=contract, action=action, ratio=ratio))
            leg_quotes.append((bid, ask, last, ticker))

        _finalize_leg_orders(underlying=underlying, leg_orders=leg_orders, leg_quotes=leg_quotes)
