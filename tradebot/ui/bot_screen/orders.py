"""Bot order repricing, admission, submission, and table presentation."""

from __future__ import annotations

import asyncio
import math
from datetime import date

from ib_insync import Contract, PortfolioItem
from rich.text import Text

from ...engines.execution import (
    _limit_price_for_mode,
    _midpoint,
    _round_to_tick,
    _sanitize_nbbo,
    _tick_size,
)
from ...option_package import option_package_debit_value
from ...order_admission import (
    OrderAdmissionFacts,
    OrderAdmissionLeg,
    OrderAdmissionRequest,
    evaluate_order_admission,
)
from ...order_reservation import (
    OrderReservation,
    OrderReservationSummary,
    summarize_order_reservations,
)
from ...signals import direction_from_action_right, parse_bar_size
from ...spot.graph_core import spot_dynamic_flip_hold_bars
from ...spot.gates import flip_exit_gate_blocked, flip_exit_hit
from ...time_utils import now_et as _now_et
from ...utils.date_utils import business_days_until
from ..bot_models import _BotInstance, _BotOrder, _SignalSnapshot
from ..common import _cost_basis, _infer_multiplier, _safe_num
from .formatting import (
    _center_table_row,
    _contract_expiry_date,
    _legs_label,
    _order_lines,
    _order_row,
    _position_as_order_row,
    _positions_subheader_row,
    _preset_lines,
)


class BotOrdersMixin:
    async def _reprice_order(self, order: _BotOrder, *, mode: str) -> bool:
        prev_mode = order.exec_mode
        order.exec_mode = str(mode or "").strip().upper() or None
        mode_changed = order.exec_mode != prev_mode
        legs = order.legs or []
        if not legs:
            return mode_changed

        if len(legs) == 1 and order.order_contract.secType != "BAG":
            leg = legs[0]
            ticker = await self._client.ensure_ticker(leg.contract, owner="bot")
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None),
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
            )
            limit = _limit_price_for_mode(bid, ask, last, action=leg.action, mode=mode)
            if limit is None:
                return False
            tick = _tick_size(leg.contract, ticker, limit) or 0.01
            limit = _round_to_tick(float(limit), tick)
            changed = not math.isclose(limit, order.limit_price, rel_tol=0, abs_tol=tick / 2.0)
            order.limit_price = float(limit)
            order.bid = bid
            order.ask = ask
            order.last = last
            return changed or mode_changed

        mid_rows: list[tuple[str, int, float]] = []
        bid_rows: list[tuple[str, int, float]] = []
        ask_rows: list[tuple[str, int, float]] = []
        desired_rows: list[tuple[str, int, float]] = []
        tick = None
        for leg in legs:
            ticker = await self._client.ensure_ticker(leg.contract, owner="bot")
            bid, ask, last = _sanitize_nbbo(
                getattr(ticker, "bid", None),
                getattr(ticker, "ask", None),
                getattr(ticker, "last", None),
            )
            mid = _midpoint(bid, ask)
            leg_mid = mid or last
            if leg_mid is None:
                return False
            leg_bid = bid or mid or last
            leg_ask = ask or mid or last
            leg_desired = _limit_price_for_mode(bid, ask, last, action=leg.action, mode=mode)
            if leg_bid is None or leg_ask is None or leg_desired is None:
                return False
            leg_tick = _tick_size(leg.contract, ticker, leg_desired)
            tick = leg_tick if tick is None else min(tick, leg_tick)
            action = "BUY" if leg.action == "BUY" else "SELL"
            mid_rows.append((action, leg.ratio, float(leg_mid)))
            bid_rows.append((action, leg.ratio, float(leg_bid)))
            ask_rows.append((action, leg.ratio, float(leg_ask)))
            desired_rows.append((action, leg.ratio, float(leg_desired)))

        debit_mid = option_package_debit_value(mid_rows)
        debit_bid = option_package_debit_value(bid_rows)
        debit_ask = option_package_debit_value(ask_rows)
        desired_debit = option_package_debit_value(desired_rows)
        assert debit_mid is not None
        assert debit_bid is not None
        assert debit_ask is not None
        assert desired_debit is not None

        tick = tick or 0.01
        order.action = "BUY"
        new_limit = _round_to_tick(float(desired_debit), tick)
        if not new_limit:
            return False
        new_bid = float(debit_bid)
        new_ask = float(debit_ask)
        new_last = float(debit_mid)
        changed = not math.isclose(
            new_limit, order.limit_price, rel_tol=0, abs_tol=tick / 2.0
        )
        order.limit_price = float(new_limit)
        order.bid = new_bid
        order.ask = new_ask
        order.last = new_last
        return changed or mode_changed

    def _reset_daily_counters_if_needed(self, instance: _BotInstance) -> None:
        today = _now_et().date()
        if instance.entries_today_date != today:
            instance.entries_today_date = today
            instance.entries_today = 0

    def _entry_limit_ok(self, instance: _BotInstance) -> bool:
        self._reset_daily_counters_if_needed(instance)
        raw = instance.strategy.get("max_entries_per_day", 1)
        try:
            max_entries = int(raw)
        except (TypeError, ValueError):
            max_entries = 1
        if max_entries <= 0:
            return True
        return instance.entries_today < max_entries

    def _spot_open_position(self, *, symbol: str, sec_type: str, con_id: int = 0) -> PortfolioItem | None:
        sym = str(symbol or "").strip().upper()
        stype = str(sec_type or "STK").strip().upper() or "STK"
        desired_con_id = int(con_id or 0)
        best = None
        best_abs = 0.0
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType != stype:
                continue
            if str(getattr(contract, "symbol", "") or "").strip().upper() != sym:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if not pos:
                continue
            if desired_con_id:
                try:
                    if int(getattr(contract, "conId", 0) or 0) == desired_con_id:
                        return item
                except (TypeError, ValueError):
                    pass
            abs_pos = abs(pos)
            if abs_pos > best_abs:
                best_abs = abs_pos
                best = item
        return best

    def _options_open_positions(self, instance: _BotInstance) -> list[PortfolioItem]:
        if not instance.touched_conids:
            return []
        open_items: list[PortfolioItem] = []
        for item in self._positions:
            contract = getattr(item, "contract", None)
            if not contract or contract.secType not in ("OPT", "FOP"):
                continue
            con_id = int(getattr(contract, "conId", 0) or 0)
            if con_id not in instance.touched_conids:
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            if pos:
                open_items.append(item)
        return open_items

    def _resolve_open_positions(
        self,
        instance: _BotInstance,
        *,
        symbol: str,
        signal_contract: Contract | None = None,
    ) -> tuple[str, list[PortfolioItem], str | None]:
        instrument = self._strategy_instrument(instance.strategy)
        if instrument == "spot":
            sec_type = str(getattr(signal_contract, "secType", "") or "").strip().upper()
            if not sec_type:
                sec_type = self._spot_sec_type(instance, symbol)
            con_id = int(getattr(signal_contract, "conId", 0) or 0) if signal_contract else 0
            item = self._spot_open_position(symbol=symbol, sec_type=sec_type, con_id=con_id)
            items = [item] if item is not None else []
            if not items:
                return instrument, items, None
            try:
                pos = float(getattr(items[0], "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            direction = "up" if pos > 0 else "down" if pos < 0 else None
            return instrument, items, direction

        items = self._options_open_positions(instance)
        direction = instance.open_direction or self._open_direction_from_positions(items)
        return instrument, items, direction

    def _options_position_values(self, items: list[PortfolioItem]) -> tuple[float | None, float | None]:
        if not items:
            return None, None
        cost_basis = 0.0
        market_value = 0.0
        for item in items:
            cost_basis += float(_cost_basis(item))
            mv = _safe_num(getattr(item, "marketValue", None))
            if mv is None:
                try:
                    mv = float(getattr(item, "position", 0.0) or 0.0) * float(getattr(item, "marketPrice", 0.0) or 0.0)
                    mv *= _infer_multiplier(item)
                except (TypeError, ValueError):
                    mv = 0.0
            market_value += float(mv)

        # Convert into the backtest sign convention: SELL credit = positive, BUY debit = negative.
        entry_value = -float(cost_basis)
        current_value = -float(market_value)
        return entry_value, current_value

    def _options_max_loss_estimate(self, items: list[PortfolioItem], *, spot: float) -> float | None:
        if not items:
            return None
        entry_value, _ = self._options_position_values(items)
        if entry_value is None:
            return None
        strikes: list[float] = []
        legs: list[tuple[str, str, float, int, float]] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            raw_right = str(getattr(contract, "right", "") or "").upper()
            right = "CALL" if raw_right in ("C", "CALL") else "PUT" if raw_right in ("P", "PUT") else ""
            if not right:
                continue
            try:
                strike = float(getattr(contract, "strike", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            qty = int(abs(pos))
            if qty <= 0:
                continue
            action = "BUY" if pos > 0 else "SELL"
            mult = _infer_multiplier(item)
            strikes.append(strike)
            legs.append((action, right, strike, qty, mult))
        if not strikes or not legs:
            return None
        strikes = sorted(set(strikes))
        high = max(float(spot), strikes[-1]) * 5.0
        candidates = [0.0] + strikes + [high]

        def _payoff(price: float) -> float:
            payoff = 0.0
            for action, right, strike, qty, mult in legs:
                if right == "CALL":
                    intrinsic = max(price - strike, 0.0)
                else:
                    intrinsic = max(strike - price, 0.0)
                sign = 1.0 if action == "BUY" else -1.0
                payoff += sign * intrinsic * float(qty) * float(mult)
            return payoff

        min_pnl = None
        for price in candidates:
            pnl = float(entry_value) + _payoff(float(price))
            if min_pnl is None or pnl < min_pnl:
                min_pnl = pnl
        if min_pnl is None:
            return None
        return max(0.0, -float(min_pnl))

    def _should_exit_on_dte(self, instance: _BotInstance, items: list[PortfolioItem], today: date) -> bool:
        raw_exit = instance.strategy.get("exit_dte", 0)
        try:
            exit_dte = int(raw_exit or 0)
        except (TypeError, ValueError):
            exit_dte = 0
        if exit_dte <= 0:
            return False
        raw_entry = instance.strategy.get("dte", 0)
        try:
            entry_dte = int(raw_entry or 0)
        except (TypeError, ValueError):
            entry_dte = 0
        if entry_dte > 0 and exit_dte >= entry_dte:
            return False

        expiries: list[date] = []
        for item in items:
            contract = getattr(item, "contract", None)
            if not contract:
                continue
            exp = _contract_expiry_date(getattr(contract, "lastTradeDateOrContractMonth", None))
            if exp is not None:
                expiries.append(exp)
        if not expiries:
            return False
        remaining = min(business_days_until(today, exp) for exp in expiries)
        return remaining <= exit_dte

    def _open_direction_from_positions(self, items: list[PortfolioItem]) -> str | None:
        if not items:
            return None
        biggest_any = None
        biggest_any_abs = 0.0
        biggest_short = None
        biggest_short_abs = 0.0
        for item in items:
            try:
                pos = float(getattr(item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            abs_pos = abs(pos)
            if abs_pos > biggest_any_abs:
                biggest_any_abs = abs_pos
                biggest_any = item
            if pos < 0 and abs_pos > biggest_short_abs:
                biggest_short_abs = abs_pos
                biggest_short = item

        chosen = biggest_short or biggest_any
        if chosen is None:
            return None
        contract = getattr(chosen, "contract", None)
        if not contract:
            return None
        right_char = str(getattr(contract, "right", "") or "").upper()
        right = "CALL" if right_char in ("C", "CALL") else "PUT" if right_char in ("P", "PUT") else ""
        try:
            pos = float(getattr(chosen, "position", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        action = "BUY" if pos > 0 else "SELL" if pos < 0 else ""
        return direction_from_action_right(action, right)

    def _should_exit_on_flip(
        self,
        instance: _BotInstance,
        snap: _SignalSnapshot,
        open_dir: str | None,
        open_items: list[PortfolioItem],
    ) -> bool:
        router_host_managed = bool(getattr(snap, "regime_router_host_managed", False))
        if router_host_managed and bool(getattr(snap, "regime_router_ready", False)):
            routed_dir = (
                str(getattr(snap, "entry_dir", None))
                if getattr(snap, "entry_dir", None) in ("up", "down")
                else None
            )
            if open_dir in ("up", "down") and routed_dir != str(open_dir):
                return True
        if router_host_managed:
            return False
        if not flip_exit_hit(
            exit_on_signal_flip=bool(instance.strategy.get("exit_on_signal_flip")),
            open_dir=open_dir,
            signal=snap.signal,
            flip_exit_mode_raw=instance.strategy.get("flip_exit_mode"),
            ema_entry_mode_raw=instance.strategy.get("ema_entry_mode"),
        ):
            return False

        tr_ratio = (
            float(getattr(snap, "ratsv_tr_ratio", 0.0)) if getattr(snap, "ratsv_tr_ratio", None) is not None else None
        )
        shock_atr_vel_pct = (
            float(getattr(snap, "shock_atr_vel_pct", 0.0))
            if getattr(snap, "shock_atr_vel_pct", None) is not None
            else None
        )
        hold_bars, _hold_trace = spot_dynamic_flip_hold_bars(
            strategy=instance.strategy,
            tr_ratio=tr_ratio,
            shock_atr_vel_pct=shock_atr_vel_pct,
        )
        if hold_bars > 0 and instance.last_entry_bar_ts is not None:
            bar_def = parse_bar_size(self._signal_bar_size(instance))
            if bar_def is not None:
                if (snap.bar_ts - instance.last_entry_bar_ts) < (bar_def.duration * hold_bars):
                    return False

        if bool(instance.strategy.get("flip_exit_only_if_profit")):
            pnl = 0.0
            for item in open_items:
                try:
                    pnl += float(getattr(item, "unrealizedPNL", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
            if pnl <= 0:
                return False

        if flip_exit_gate_blocked(
            gate_mode_raw=instance.strategy.get("flip_exit_gate_mode"),
            filters=instance.filters,
            close=float(snap.close),
            signal=snap.signal,
            trade_dir=open_dir,
        ):
            return False
        return True

    def _entry_direction_for_instance(self, instance: _BotInstance, snap: _SignalSnapshot) -> str | None:
        entry_dir = snap.entry_dir if getattr(snap, "entry_dir", None) in ("up", "down") else snap.signal.entry_dir
        return str(entry_dir) if entry_dir in ("up", "down") else None

    def _allowed_entry_directions(self, instance: _BotInstance) -> set[str]:
        strategy = instance.strategy or {}
        instrument = self._strategy_instrument(strategy)
        if instrument == "spot":
            mapping = strategy.get("directional_spot") if isinstance(strategy.get("directional_spot"), dict) else None
            if mapping:
                allowed = set()
                for key in ("up", "down"):
                    leg = mapping.get(key)
                    if not isinstance(leg, dict):
                        continue
                    action = str(leg.get("action", "")).strip().upper()
                    if action in ("BUY", "SELL"):
                        allowed.add(key)
                return allowed
            return {"up"}

        if isinstance(strategy.get("directional_legs"), dict):
            allowed = {k for k in ("up", "down") if strategy["directional_legs"].get(k)}
            return allowed or {"up", "down"}

        legs = strategy.get("legs", [])
        if isinstance(legs, list) and legs:
            first = legs[0] if isinstance(legs[0], dict) else None
            if isinstance(first, dict):
                bias = direction_from_action_right(first.get("action", ""), first.get("right", ""))
                if bias in ("up", "down"):
                    return {bias}
        return {"up", "down"}

    def _submit_order(self) -> None:
        self._submit_selected_order()

    def _submit_selected_order(self) -> None:
        order = self._selected_order()
        if not order:
            self._set_status("Send: no order selected", render_bot=True)
            return
        if order.status != "STAGED":
            self._set_status(f"Send: already {order.status}", render_bot=True)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("Send: no loop", render_bot=True)
            return
        self._set_status("Sending order...", render_bot=True)
        self._send_task = loop.create_task(self._send_order(order))

    async def _send_order(self, order: _BotOrder) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        try:
            sec_type = str(
                getattr(order.order_contract, "secType", "") or ""
            ).strip().upper()
            symbol = str(
                getattr(order.order_contract, "symbol", "") or ""
            ).strip().upper()
            if sec_type == "BAG" and symbol == "XSP":
                package_risk = order.package_risk
                request = OrderAdmissionRequest(
                    account=str(
                        getattr(
                            getattr(self._client, "_config", None),
                            "account",
                            "",
                        )
                        or ""
                    ).strip(),
                    intent=str(order.intent or "enter").strip().lower(),
                    product_domain=symbol,
                    structure=(
                        str(package_risk.structure)
                        if package_risk is not None
                        else ""
                    ),
                    sec_type=sec_type,
                    symbol=symbol,
                    currency=str(
                        getattr(order.order_contract, "currency", "") or ""
                    ).strip().upper(),
                    exchange=str(
                        getattr(order.order_contract, "exchange", "") or ""
                    ).strip().upper(),
                    action=str(order.action or "").strip().upper(),
                    quantity=int(order.quantity),
                    limit_price=float(order.limit_price),
                    max_loss=(
                        float(package_risk.max_loss)
                        if package_risk is not None
                        else None
                    ),
                    legs=tuple(
                        OrderAdmissionLeg(
                            con_id=int(
                                getattr(leg.contract, "conId", 0) or 0
                            ),
                            ratio=int(leg.ratio),
                            action=str(leg.action or "").strip().upper(),
                            exchange=str(
                                getattr(leg.contract, "exchange", "") or ""
                            ).strip().upper(),
                        )
                        for leg in order.legs
                    ),
                )

                facts = OrderAdmissionFacts()
                if package_risk is not None:
                    preview = await self._client.preview_limit_order(
                        order.order_contract,
                        order.action,
                        order.quantity,
                        order.limit_price,
                        outside_rth=False,
                    )
                    facts = OrderAdmissionFacts(
                        status=preview.status,
                        init_margin_before=preview.init_margin_before,
                        init_margin_change=preview.init_margin_change,
                        init_margin_after=preview.init_margin_after,
                        maintenance_margin_before=preview.maintenance_margin_before,
                        maintenance_margin_change=preview.maintenance_margin_change,
                        maintenance_margin_after=preview.maintenance_margin_after,
                        equity_with_loan_before=preview.equity_with_loan_before,
                        equity_with_loan_change=preview.equity_with_loan_change,
                        equity_with_loan_after=preview.equity_with_loan_after,
                        commission=preview.commission,
                        min_commission=preview.min_commission,
                        max_commission=preview.max_commission,
                        commission_currency=preview.commission_currency,
                        warning_text=preview.warning_text,
                    )

                decision = evaluate_order_admission(request, facts)
                self._journal_write(
                    event="ORDER_ADMISSION",
                    order=order,
                    reason=order.reason,
                    data={"admission": decision.as_payload()},
                )
                if not decision.allow:
                    order.status = "BLOCKED"
                    order.error = decision.reason
                    self._set_status(f"Order blocked: {decision.reason}")
                    self._refresh_orders_table()
                    self._render_bot()
                    return

            self._journal_write(event="SENDING", order=order, reason=order.reason, data=None)
            trade = await self._client.place_limit_order(
                order.order_contract,
                order.action,
                order.quantity,
                order.limit_price,
                outside_rth=order.order_contract.secType == "STK",
            )
            order_id = trade.order.orderId or trade.order.permId or 0
            order.status = "WORKING"
            order.order_id = int(order_id or 0) or None
            order.trade = trade
            order.sent_at = loop.time() if loop is not None else None
            order.cancel_requested_at = None
            self._set_status(f"Sent #{order_id} {order.action} {order.quantity} @ {order.limit_price:.2f}")
            self._journal_write(event="SENT", order=order, reason=order.reason, data=None)
        except Exception as exc:
            order.status = "ERROR"
            order.error = str(exc)
            self._set_status(f"Send error: {exc}")
            self._journal_write(event="SEND_ERROR", order=order, reason=order.reason, data={"exc": str(exc)})
        self._refresh_orders_table()
        self._render_bot()

    def _render_bot(self) -> None:
        lines: list[Text] = [Text("Bot Hub", style="bold")]
        if self._payload:
            symbol = self._payload.get("symbol", "?")
            start = self._payload.get("start", "?")
            end = self._payload.get("end", "?")
            bar_size = self._payload.get("bar_size", "?")
            lines.append(Text(f"Leaderboard: {symbol} {start}→{end} ({bar_size})", style="dim"))
        else:
            lines.append(Text("No leaderboard loaded", style="red"))

        lines.append(
            Text(
                "Enter=Config/Send  Ctrl+A/p=Presets  f=FilterDTE  w=FilterWin  v=Scope  Tab/h/l=Focus  c=Cancel  Space=Run/Toggle  s=Stop  S=Kill  d=Del  X=Send",
                style="dim",
            )
        )
        dte_label = "ALL" if self._filter_dte is None else str(self._filter_dte)
        win_label = (
            "ALL"
            if self._filter_min_win_rate is None
            else f"≥{int(self._filter_min_win_rate * 100)}%"
        )
        lines.append(Text(f"Filter: DTE={dte_label} (f)  Win={win_label} (w)", style="dim"))
        focus_style = {
            "presets": "bold #62b0ff",
            "instances": "bold #66d19e",
            "orders": "bold #f3b267",
            "logs": "bold #b7a6ff",
        }.get(self._active_panel, "bold")
        focus_line = Text("Focus: ", style="dim")
        focus_line.append(str(self._active_panel).upper(), style=focus_style)
        focus_line.append(
            f"  Presets: {'ON' if self._presets_visible else 'OFF'}  "
            f"Scope: {'ALL' if self._scope_all else 'Instance'}  "
            f"Instances: {len(self._instances)}  Orders: {len(self._order_rows)}",
            style="dim",
        )
        lines.append(focus_line)
        lines.append(Text("Hours legend: R=RTH, F=24/5, cXX=cutoff hour ET", style="dim"))

        if self._active_panel == "presets":
            selected = self._selected_preset()
            if selected:
                lines.append(Text(""))
                lines.append(Text("Selected preset", style="bold"))
                lines.extend(_preset_lines(selected))
                eval_payload = self._group_eval_by_name.get(selected.group)
                if isinstance(eval_payload, dict):
                    windows = eval_payload.get("windows")
                    if isinstance(windows, list) and windows:
                        lines.append(Text(""))
                        lines.append(Text("Multiwindow (stability)", style="bold"))
                        for w in windows:
                            if not isinstance(w, dict):
                                continue
                            start = str(w.get("start") or "").strip()
                            end = str(w.get("end") or "").strip()
                            label = f"{start}→{end}" if start and end else (start or end or "window")
                            try:
                                roi = float(w.get("roi", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                roi = 0.0
                            try:
                                dd_pct = float(w.get("dd_pct", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                dd_pct = 0.0
                            try:
                                pnl = float(w.get("pnl", 0.0) or 0.0)
                            except (TypeError, ValueError):
                                pnl = 0.0
                            pnl_mo: float | None = None
                            if start and end:
                                try:
                                    start_d = date.fromisoformat(start)
                                    end_d = date.fromisoformat(end)
                                except ValueError:
                                    start_d = None
                                    end_d = None
                                if start_d and end_d and end_d > start_d:
                                    months = max((end_d - start_d).days / 30.0, 1.0)
                                    pnl_mo = pnl / months
                            try:
                                trades = int(w.get("trades", 0) or 0)
                            except (TypeError, ValueError):
                                trades = 0
                            roi_text = Text(f"roi={roi*100:.1f}%", style="green" if roi > 0 else "")
                            dd_text = Text(f"dd={dd_pct*100:.1f}%", style="red" if dd_pct > 0 else "")
                            lines.append(
                                Text(f"{label}  ", style="dim")
                                + roi_text
                                + Text("  ", style="dim")
                                + dd_text
                                + Text(
                                    f"  tr={trades}  pnl={pnl:,.1f}"
                                    + (f"  pnl/mo={pnl_mo:,.0f}" if pnl_mo is not None else ""),
                                    style="dim",
                                )
                            )
        elif self._active_panel == "instances":
            instance = self._selected_instance()
            if instance:
                instrument = self._strategy_instrument(instance.strategy or {})
                if instrument == "spot":
                    sec_type = str((instance.strategy or {}).get("spot_sec_type") or "").strip().upper()
                    legs_desc = "SPOT-FUT" if sec_type == "FUT" else "SPOT-STK"
                    dte = "-"
                else:
                    legs_desc = _legs_label(instance.strategy.get("legs", []))
                    dte = instance.strategy.get("dte", "?")
                lines.append(Text(""))
                lines.append(Text(f"Selected instance #{instance.instance_id}", style="bold"))
                lines.append(Text(f"{instance.group}  DTE={dte}  Legs={legs_desc}", style="dim"))
                lines.append(Text(f"State={instance.state}", style="dim"))
        elif self._active_panel == "orders":
            order = self._selected_order()
            if order:
                lines.append(Text(""))
                lines.append(Text(f"Selected order (inst {order.instance_id})", style="bold"))
                lines.extend(_order_lines(order))

        if self._status:
            lines.append(Text(""))
            lines.append(Text(self._status, style="yellow"))

        self._status_panel.update(Text("\n").join(lines))

    def _order_reservation_summary(self) -> OrderReservationSummary:
        account = str(
            getattr(
                getattr(self._client, "_config", None),
                "account",
                "",
            )
            or ""
        ).strip()

        reservations: list[OrderReservation] = []
        for order in self._orders:
            contract = order.order_contract
            sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
            symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
            package_risk = order.package_risk

            structure = str(
                getattr(package_risk, "structure", "") or ""
            ).strip()
            max_loss = (
                getattr(package_risk, "max_loss", None)
                if package_risk is not None
                else None
            )

            if not structure and sec_type == "BAG" and symbol == "XSP":
                try:
                    signed_combo_price = float(order.limit_price)
                except (TypeError, ValueError):
                    signed_combo_price = 0.0
                if signed_combo_price < 0:
                    structure = "vertical_credit"

            reservations.append(
                OrderReservation(
                    account=account,
                    product_domain=symbol,
                    sec_type=sec_type,
                    structure=structure,
                    status=str(order.status or ""),
                    max_loss=max_loss,
                )
            )

        return summarize_order_reservations(
            reservations,
            account=account,
        )

    def _add_order(self, order: _BotOrder) -> None:
        self._orders.append(order)
        instance = next((i for i in self._instances if i.instance_id == order.instance_id), None)
        if instance is not None:
            self._clear_order_trigger_watch(instance)
        self._journal_write(event="ORDER_STAGED", instance=instance, order=order, reason=order.reason, data=None)
        self._refresh_orders_table()
        if self._active_panel == "orders":
            self._orders_table.cursor_coordinate = (max(len(self._order_rows) - 1, 0), 0)

    def _refresh_orders_table(self) -> None:
        self._orders_table.clear()
        self._order_rows = []
        scope = self._scope_instance_id()
        if scope is None and not self._scope_all:
            self._sync_row_marker(self._orders_table, force=True)
            return
        for order in self._orders:
            if scope is not None and order.instance_id != scope:
                continue
            self._orders_table.add_row(*_center_table_row(*_order_row(order)))
            self._order_rows.append(order)

        # Unify positions into the Orders table (so the bottom pane can be Logs).
        if self._scope_all:
            con_ids = set().union(*(i.touched_conids for i in self._instances))
        else:
            instance = next((i for i in self._instances if i.instance_id == scope), None)
            con_ids = set(instance.touched_conids) if instance else set()
        if not con_ids:
            self._sync_row_marker(self._orders_table, force=True)
            return
        self._orders_table.add_row(
            *_center_table_row("", "", "", "", Text("POSITIONS", style="bold"), "", "", "", "")
        )
        self._orders_table.add_row(*_center_table_row(*_positions_subheader_row()))
        for item in self._positions:
            try:
                con_id = int(getattr(item.contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                con_id = 0
            if con_id not in con_ids:
                continue
            unreal, _official_unreal, _estimate_unreal, mark_price = self._position_unrealized_values(item)
            daily = self._official_daily_value(item)
            realized = _safe_num(getattr(item, "realizedPNL", None))
            entry_now_text = self._position_entry_now_cell(item, mark_price)
            px_change_text = self._position_px_change_cell(item, mark_price)
            self._orders_table.add_row(
                *_center_table_row(
                    *_position_as_order_row(
                        item,
                        scope=scope,
                        daily=daily,
                        unreal=unreal,
                        realized=realized,
                        market_price=mark_price,
                        entry_now_text=entry_now_text,
                        px_change_text=px_change_text,
                    )
                )
            )
        self._sync_row_marker(self._orders_table, force=True)
