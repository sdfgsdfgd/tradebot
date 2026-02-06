"""Bot signal loop + auto-order trigger mixin."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from ..engine import cooldown_ok_by_time, normalize_spot_entry_signal, parse_time_hhmm, signal_filters_ok
from .bot_models import _BotInstance
from .common import _safe_num, _ticker_price


def _weekday_num(label: str) -> int:
    key = label.strip().upper()[:3]
    mapping = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    return mapping.get(key, 0)


class BotSignalRuntimeMixin:
    async def _auto_order_tick(self) -> None:
        if self._order_task and not self._order_task.done():
            return
        now_et = datetime.now(tz=ZoneInfo("America/New_York"))

        for instance in self._instances:
            if instance.state != "RUNNING":
                continue

            def _gate(status: str, data: dict | None = None) -> None:
                if instance.last_gate_status == status:
                    return
                instance.last_gate_status = status
                self._journal_write(event="GATE", instance=instance, reason=status, data=data)

            if not self._can_order_now(instance):
                _gate("BLOCKED_WEEKDAY_NOW", {"now_weekday": int(now_et.weekday())})
                continue

            pending = any(
                o.status in ("STAGED", "WORKING", "CANCELING") and o.instance_id == instance.instance_id
                for o in self._orders
            )
            if pending:
                _gate("PENDING_ORDER", None)
                continue

            symbol = str(
                instance.symbol or (self._payload.get("symbol", "SLV") if self._payload else "SLV")
            ).strip().upper()

            entry_signal = normalize_spot_entry_signal(instance.strategy.get("entry_signal"))
            ema_preset = instance.strategy.get("ema_preset")
            if entry_signal == "ema" and not ema_preset:
                raise RuntimeError(
                    "FATAL: missing required strategy field `ema_preset` "
                    f"(instance_id={instance.instance_id} group={instance.group!r} symbol={symbol!r})"
                )

            signal_contract = await self._signal_contract(instance, symbol)
            if signal_contract is None:
                _gate("NO_SIGNAL_CONTRACT", {"symbol": symbol})
                continue

            snap = await self._signal_snapshot_for_contract(
                contract=signal_contract,
                **self._signal_snapshot_kwargs(
                    instance,
                    ema_preset_raw=str(ema_preset) if ema_preset else None,
                    entry_signal_raw=entry_signal,
                    include_orb=True,
                    include_spot_exit=True,
                ),
            )
            if snap is None:
                _gate(
                    "NO_SIGNAL_SNAPSHOT",
                    {
                        "symbol": symbol,
                        "bar_size": self._signal_bar_size(instance),
                        "use_rth": bool(self._signal_use_rth(instance)),
                    },
                )
                continue

            risk = snap.risk
            signal_fingerprint = (
                str(snap.signal.state or ""),
                str(snap.signal.entry_dir or ""),
                str(snap.signal.regime_dir or ""),
                bool(snap.signal.regime_ready),
                bool(snap.or_ready),
                float(snap.or_high) if snap.or_high is not None else None,
                float(snap.or_low) if snap.or_low is not None else None,
                bool(snap.shock) if snap.shock is not None else None,
                str(snap.shock_dir or ""),
                bool(getattr(risk, "riskoff", False)) if risk is not None else None,
                bool(getattr(risk, "riskpanic", False)) if risk is not None else None,
                bool(getattr(risk, "riskpop", False)) if risk is not None else None,
            )
            if instance.last_signal_fingerprint != signal_fingerprint:
                instance.last_signal_fingerprint = signal_fingerprint
                self._journal_write(
                    event="SIGNAL",
                    instance=instance,
                    order=None,
                    reason=None,
                    data={
                        "symbol": symbol,
                        "bar_ts": snap.bar_ts.isoformat(),
                        "close": float(snap.close),
                        "signal": {
                            "state": snap.signal.state,
                            "entry_dir": snap.signal.entry_dir,
                            "cross_up": bool(snap.signal.cross_up),
                            "cross_down": bool(snap.signal.cross_down),
                            "ema_ready": bool(snap.signal.ema_ready),
                            "regime_dir": snap.signal.regime_dir,
                            "regime_ready": bool(snap.signal.regime_ready),
                        },
                        "orb": {
                            "ready": bool(snap.or_ready),
                            "high": float(snap.or_high) if snap.or_high is not None else None,
                            "low": float(snap.or_low) if snap.or_low is not None else None,
                        },
                        "shock": {
                            "shock": snap.shock,
                            "dir": snap.shock_dir,
                            "atr_pct": float(snap.shock_atr_pct)
                            if snap.shock_atr_pct is not None
                            else None,
                        },
                        "risk": {
                            "riskoff": bool(getattr(risk, "riskoff", False)) if risk is not None else None,
                            "riskpanic": bool(getattr(risk, "riskpanic", False)) if risk is not None else None,
                            "riskpop": bool(getattr(risk, "riskpop", False)) if risk is not None else None,
                        },
                        "rv": float(snap.rv) if snap.rv is not None else None,
                        "volume": float(snap.volume) if snap.volume is not None else None,
                        "volume_ema": float(snap.volume_ema) if snap.volume_ema is not None else None,
                        "volume_ema_ready": bool(snap.volume_ema_ready),
                    },
                )

            if (
                bool(snap.signal.cross_up) or bool(snap.signal.cross_down)
            ) and instance.last_cross_bar_ts != snap.bar_ts:
                instance.last_cross_bar_ts = snap.bar_ts
                self._journal_write(
                    event="CROSS",
                    instance=instance,
                    order=None,
                    reason="up" if bool(snap.signal.cross_up) else "down",
                    data={
                        "symbol": symbol,
                        "bar_ts": snap.bar_ts.isoformat(),
                        "close": float(snap.close),
                        "state": snap.signal.state,
                        "entry_dir": snap.signal.entry_dir,
                    },
                )

            entry_days = instance.strategy.get("entry_days", [])
            if entry_days:
                allowed_days = {_weekday_num(day) for day in entry_days}
            else:
                allowed_days = {0, 1, 2, 3, 4}
            if snap.bar_ts.weekday() not in allowed_days:
                _gate(
                    "BLOCKED_ENTRY_DAY",
                    {
                        "signal_weekday": int(snap.bar_ts.weekday()),
                        "allowed_days": sorted(int(d) for d in allowed_days),
                    },
                )
                continue

            cooldown_bars = 0
            if isinstance(instance.filters, dict):
                raw = instance.filters.get("cooldown_bars", 0)
                try:
                    cooldown_bars = int(raw or 0)
                except (TypeError, ValueError):
                    cooldown_bars = 0
            cooldown_ok = cooldown_ok_by_time(
                current_bar_ts=snap.bar_ts,
                last_entry_bar_ts=instance.last_entry_bar_ts,
                bar_size=self._signal_bar_size(instance),
                cooldown_bars=cooldown_bars,
            )
            if not signal_filters_ok(
                instance.filters,
                bar_ts=snap.bar_ts,
                bars_in_day=snap.bars_in_day,
                close=float(snap.close),
                volume=snap.volume,
                volume_ema=snap.volume_ema,
                volume_ema_ready=snap.volume_ema_ready,
                rv=snap.rv,
                signal=snap.signal,
                cooldown_ok=cooldown_ok,
                shock=snap.shock,
                shock_dir=snap.shock_dir,
            ):
                _gate(
                    "BLOCKED_FILTERS",
                    {
                        "symbol": symbol,
                        "bar_ts": snap.bar_ts.isoformat(),
                        "cooldown_ok": bool(cooldown_ok),
                        "rv": float(snap.rv) if snap.rv is not None else None,
                        "volume": float(snap.volume) if snap.volume is not None else None,
                        "volume_ema": float(snap.volume_ema) if snap.volume_ema is not None else None,
                        "volume_ema_ready": bool(snap.volume_ema_ready),
                        "shock": snap.shock,
                        "shock_dir": snap.shock_dir,
                        "entry_dir": snap.signal.entry_dir,
                    },
                )
                continue

            instrument, open_items, open_dir = self._resolve_open_positions(
                instance,
                symbol=symbol,
                signal_contract=signal_contract,
            )

            if not open_items and instance.open_direction is not None:
                instance.open_direction = None
                instance.spot_profit_target_price = None
                instance.spot_stop_loss_price = None

            if open_items:
                if instance.last_exit_bar_ts is not None and instance.last_exit_bar_ts == snap.bar_ts:
                    _gate(
                        "BLOCKED_EXIT_SAME_BAR",
                        {
                            "bar_ts": snap.bar_ts.isoformat(),
                            "direction": open_dir,
                            "items": len(open_items),
                        },
                    )
                    continue
                _gate("HOLDING", {"direction": open_dir, "items": len(open_items)})

                def _trigger_exit(reason: str, *, mode: str = instrument) -> None:
                    self._queue_order(
                        instance,
                        intent="exit",
                        direction=open_dir,
                        signal_bar_ts=snap.bar_ts,
                    )
                    _gate("TRIGGER_EXIT", {"mode": mode, "reason": reason})

                if instrument == "spot":
                    open_item = open_items[0]
                    try:
                        pos = float(getattr(open_item, "position", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pos = 0.0
                    avg_cost = _safe_num(getattr(open_item, "averageCost", None))
                    market_price = _safe_num(getattr(open_item, "marketPrice", None))
                    if market_price is None:
                        ticker = await self._client.ensure_ticker(open_item.contract, owner="bot")
                        market_price = _ticker_price(ticker)

                    target_price = instance.spot_profit_target_price
                    stop_price = instance.spot_stop_loss_price
                    if (
                        pos
                        and market_price is not None
                        and market_price > 0
                        and (target_price is not None or stop_price is not None)
                    ):
                        try:
                            mp = float(market_price)
                        except (TypeError, ValueError):
                            mp = None
                        if mp is not None:
                            if target_price is not None:
                                try:
                                    target = float(target_price)
                                except (TypeError, ValueError):
                                    target = None
                                if target is not None and target > 0:
                                    if (pos > 0 and mp >= target) or (pos < 0 and mp <= target):
                                        _trigger_exit("profit_target", mode="spot")
                                        break
                            if stop_price is not None:
                                try:
                                    stop = float(stop_price)
                                except (TypeError, ValueError):
                                    stop = None
                                if stop is not None and stop > 0:
                                    if (pos > 0 and mp <= stop) or (pos < 0 and mp >= stop):
                                        _trigger_exit("stop_loss", mode="spot")
                                        break
                    move = None
                    if (
                        target_price is None
                        and stop_price is None
                        and avg_cost is not None
                        and avg_cost > 0
                        and market_price is not None
                        and market_price > 0
                        and pos
                    ):
                        move = (market_price - avg_cost) / avg_cost
                        if pos < 0:
                            move = -move

                    try:
                        pt = (
                            float(instance.strategy.get("spot_profit_target_pct"))
                            if instance.strategy.get("spot_profit_target_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        pt = None
                    try:
                        sl = (
                            float(instance.strategy.get("spot_stop_loss_pct"))
                            if instance.strategy.get("spot_stop_loss_pct") is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        sl = None

                    # Dynamic shock SL/PT: mirror backtest semantics for pct-based exits.
                    if bool(snap.shock) and isinstance(instance.filters, dict):
                        try:
                            sl_mult = float(instance.filters.get("shock_stop_loss_pct_mult", 1.0) or 1.0)
                        except (TypeError, ValueError):
                            sl_mult = 1.0
                        try:
                            pt_mult = float(instance.filters.get("shock_profit_target_pct_mult", 1.0) or 1.0)
                        except (TypeError, ValueError):
                            pt_mult = 1.0
                        if sl_mult <= 0:
                            sl_mult = 1.0
                        if pt_mult <= 0:
                            pt_mult = 1.0
                        if sl is not None and float(sl) > 0:
                            sl = min(float(sl) * float(sl_mult), 0.99)
                        if pt is not None and float(pt) > 0:
                            pt = min(float(pt) * float(pt_mult), 0.99)

                    if move is not None and pt is not None and move >= pt:
                        _trigger_exit("profit_target_pct", mode="spot")
                        break
                    if move is not None and sl is not None and move <= -sl:
                        _trigger_exit("stop_loss_pct", mode="spot")
                        break

                    exit_time = parse_time_hhmm(instance.strategy.get("spot_exit_time_et"))
                    if exit_time is not None and now_et.time() >= exit_time:
                        _trigger_exit("exit_time", mode="spot")
                        break

                    if bool(instance.strategy.get("spot_close_eod")) and (
                        now_et.hour > 15 or now_et.hour == 15 and now_et.minute >= 55
                    ):
                        _trigger_exit("close_eod", mode="spot")
                        break

                if instrument != "spot":
                    if self._should_exit_on_dte(instance, open_items, now_et.date()):
                        _trigger_exit("dte", mode="options")
                        break

                    entry_value, current_value = self._options_position_values(open_items)
                    if entry_value is not None and current_value is not None:
                        profit = float(entry_value) - float(current_value)
                        try:
                            profit_target = float(instance.strategy.get("profit_target", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            profit_target = 0.0
                        if profit_target > 0 and abs(entry_value) > 0:
                            if profit >= abs(entry_value) * profit_target:
                                _trigger_exit("profit_target", mode="options")
                                break

                        try:
                            stop_loss = float(instance.strategy.get("stop_loss", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            stop_loss = 0.0
                        if stop_loss > 0:
                            loss = max(0.0, float(current_value) - float(entry_value))
                            basis = str(instance.strategy.get("stop_loss_basis") or "max_loss").strip().lower()
                            if basis == "credit":
                                if entry_value >= 0:
                                    if current_value >= entry_value * (1.0 + stop_loss):
                                        _trigger_exit("stop_loss_credit", mode="options")
                                        break
                                elif loss >= abs(entry_value) * stop_loss:
                                    _trigger_exit("stop_loss_credit", mode="options")
                                    break
                            else:
                                max_loss = self._options_max_loss_estimate(open_items, spot=float(snap.close))
                                if max_loss is None or max_loss <= 0:
                                    max_loss = abs(entry_value)
                                if max_loss and loss >= float(max_loss) * stop_loss:
                                    _trigger_exit("stop_loss_max_loss", mode="options")
                                    break

                if self._should_exit_on_flip(instance, snap, open_dir, open_items):
                    _trigger_exit("flip", mode=instrument)
                    break
                continue

            if not self._entry_limit_ok(instance):
                _gate("BLOCKED_ENTRY_LIMIT", {"entries_today": int(instance.entries_today)})
                continue
            if instance.last_entry_bar_ts is not None and instance.last_entry_bar_ts == snap.bar_ts:
                _gate("BLOCKED_ENTRY_SAME_BAR", {"bar_ts": snap.bar_ts.isoformat()})
                continue

            instrument = self._strategy_instrument(instance.strategy)
            if instrument == "spot":
                exit_mode = str(instance.strategy.get("spot_exit_mode") or "pct").strip().lower()
                if exit_mode == "atr":
                    atr = float(snap.atr or 0.0) if snap.atr is not None else 0.0
                    if atr <= 0:
                        _gate("BLOCKED_ATR_NOT_READY", {"atr": float(atr)})
                        continue

            direction = self._entry_direction_for_instance(instance, snap)
            if direction is None:
                _gate("WAITING_SIGNAL", {"bar_ts": snap.bar_ts.isoformat()})
                continue
            if direction not in self._allowed_entry_directions(instance):
                _gate("BLOCKED_DIRECTION", {"direction": direction})
                continue

            self._queue_order(
                instance,
                intent="enter",
                direction=direction,
                signal_bar_ts=snap.bar_ts,
            )
            _gate("TRIGGER_ENTRY", {"direction": direction})
            break

    def _queue_order(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
    ) -> None:
        if self._order_task and not self._order_task.done():
            self._status = "Order: busy"
            self._render_status()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = "Order: no loop"
            self._render_status()
            return
        action = "Exiting" if intent == "exit" else "Creating order"
        dir_note = f" ({direction})" if direction else ""
        self._status = f"{action} for instance {instance.instance_id}{dir_note}..."
        self._render_status()
        self._order_task = loop.create_task(
            self._create_order_for_instance(
                instance,
                intent=str(intent),
                direction=direction,
                signal_bar_ts=signal_bar_ts,
            )
        )

    def _can_order_now(self, instance: _BotInstance) -> bool:
        entry_days = instance.strategy.get("entry_days", [])
        if entry_days:
            allowed = {_weekday_num(day) for day in entry_days}
        else:
            allowed = {0, 1, 2, 3, 4, 5, 6}
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        if now.weekday() not in allowed:
            return False
        return True
