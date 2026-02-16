"""Bot signal loop + auto-order trigger mixin."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, time, timedelta

from ..engine import (
    cooldown_ok_by_time,
    normalize_shock_detector,
    normalize_shock_gate_mode,
    normalize_spot_entry_signal,
    parse_time_hhmm,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
    spot_intrabar_exit,
    spot_policy_config_view,
    spot_profit_level,
    spot_riskoff_end_hour,
    spot_riskoff_mode_from_filters,
    spot_runtime_spec_view,
    spot_scale_exit_pcts,
    spot_shock_exit_pct_multipliers,
    spot_stop_level,
)
from ..signals import ema_periods, parse_bar_size
from ..spot.lifecycle import (
    decide_flat_position_intent,
    decide_open_position_intent,
    decide_pending_next_open,
    signal_filter_checks,
)
from ..time_utils import now_et as _now_et
from ..time_utils import to_et as _to_et_shared
from .bot_models import _BotInstance
from .common import _market_session_bucket, _safe_num, _ticker_price

_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)
_EXT_OPEN = time(4, 0)
_EXT_CLOSE = time(20, 0)
_DEFAULT_ORDER_STAGE_TIMEOUT_SEC = 20.0
_DEFAULT_EXIT_RETRY_MAX_PER_BAR = 3
_BID_TICK_TYPES = {1, 66}
_ASK_TICK_TYPES = {2, 67}


def _weekday_num(label: str) -> int:
    key = label.strip().upper()[:3]
    mapping = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    return mapping.get(key, 0)


class BotSignalRuntimeMixin:
    async def _auto_order_tick(self) -> None:
        now_et = _now_et()
        now_wall = self._wall_time(now_et)
        no_signal_watchdog_sec = 60.0
        self._check_order_trigger_watchdogs(now_et=now_et)
        if self._order_task and not self._order_task.done():
            return

        for instance in self._instances:
            if instance.state != "RUNNING":
                continue

            def _gate(status: str, data: dict | None = None) -> None:
                status_text = str(status or "")
                fingerprint = (
                    status_text,
                    json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else "",
                )
                if instance.last_gate_fingerprint == fingerprint:
                    if status_text == "NO_SIGNAL_SNAPSHOT":
                        if instance.no_signal_snapshot_since is None:
                            instance.no_signal_snapshot_since = now_et
                        last_ping = instance.no_signal_snapshot_last_ping
                        should_ping = (
                            last_ping is None
                            or (now_et - last_ping).total_seconds() >= float(no_signal_watchdog_sec)
                        )
                        if should_ping:
                            instance.no_signal_snapshot_last_ping = now_et
                            since_sec = max(
                                0.0,
                                float((now_et - instance.no_signal_snapshot_since).total_seconds()),
                            )
                            payload: dict[str, object] = {"since_sec": since_sec}
                            if isinstance(data, dict):
                                for key in ("symbol", "bar_size", "use_rth", "diag"):
                                    if key in data:
                                        payload[key] = data.get(key)
                            self._journal_write(
                                event="GATE",
                                instance=instance,
                                reason="NO_SIGNAL_PERSISTING",
                                data=payload,
                            )
                    return
                instance.last_gate_fingerprint = fingerprint
                instance.last_gate_status = status_text
                if status_text == "NO_SIGNAL_SNAPSHOT":
                    if instance.no_signal_snapshot_since is None:
                        instance.no_signal_snapshot_since = now_et
                    instance.no_signal_snapshot_last_ping = now_et
                else:
                    instance.no_signal_snapshot_since = None
                    instance.no_signal_snapshot_last_ping = None
                self._journal_write(event="GATE", instance=instance, reason=status_text, data=data)

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
                diag = getattr(self, "_last_signal_snapshot_diag", None)
                _gate(
                    "NO_SIGNAL_SNAPSHOT",
                    {
                        "symbol": symbol,
                        "bar_size": self._signal_bar_size(instance),
                        "use_rth": bool(self._signal_use_rth(instance)),
                        "diag": dict(diag) if isinstance(diag, dict) else None,
                    },
                )
                continue

            risk = snap.risk
            bar_health_raw = snap.bar_health if isinstance(snap.bar_health, dict) else None
            bar_health = dict(bar_health_raw) if bar_health_raw is not None else None
            if bar_health is not None:
                last_bar_ts = bar_health.get("last_bar_ts")
                if isinstance(last_bar_ts, datetime):
                    bar_health["last_bar_ts"] = last_bar_ts.isoformat()
            stale_signal = bool(bar_health_raw and bar_health_raw.get("stale"))
            gap_signal = bool(bar_health_raw and bar_health_raw.get("gap_detected"))
            ema_fast = float(snap.signal.ema_fast) if snap.signal.ema_fast is not None else None
            ema_slow = float(snap.signal.ema_slow) if snap.signal.ema_slow is not None else None
            prev_ema_fast = float(snap.signal.prev_ema_fast) if snap.signal.prev_ema_fast is not None else None
            ema_spread_pct = None
            ema_slope_pct = None
            if ema_fast is not None and ema_slow is not None and snap.close:
                ema_spread_pct = ((ema_fast - ema_slow) / float(snap.close)) * 100.0
            if ema_fast is not None and prev_ema_fast is not None and snap.close:
                ema_slope_pct = ((ema_fast - prev_ema_fast) / float(snap.close)) * 100.0
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
                bool(bar_health_raw and bar_health_raw.get("stale")),
                (
                    bar_health_raw.get("last_bar_ts").isoformat()
                    if isinstance(bar_health_raw, dict)
                    and isinstance(bar_health_raw.get("last_bar_ts"), datetime)
                    else None
                ),
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
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "ema_spread_pct": ema_spread_pct,
                            "ema_slope_pct": ema_slope_pct,
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
                        "bars_in_day": int(snap.bars_in_day),
                        "volume": float(snap.volume) if snap.volume is not None else None,
                        "volume_ema": float(snap.volume_ema) if snap.volume_ema is not None else None,
                        "volume_ema_ready": bool(snap.volume_ema_ready),
                        "bar_health": bar_health,
                        "meta": {
                            "entry_signal": str(instance.strategy.get("entry_signal") or "ema"),
                            "signal_bar_size": self._signal_bar_size(instance),
                            "signal_use_rth": bool(self._signal_use_rth(instance)),
                            "regime_mode": str(instance.strategy.get("regime_mode") or "ema"),
                            "regime_bar_size": str(
                                instance.strategy.get("regime_bar_size") or self._signal_bar_size(instance)
                            ),
                            "regime2_mode": str(instance.strategy.get("regime2_mode") or "off"),
                            "regime2_bar_size": str(
                                instance.strategy.get("regime2_bar_size") or self._signal_bar_size(instance)
                            ),
                            "regime2_apply_to": str(instance.strategy.get("regime2_apply_to") or "both"),
                            "spot_exec_feed_mode": self._spot_exec_feed_mode(instance),
                            "spot_exec_bar_size": str(
                                instance.strategy.get("spot_exec_bar_size") or self._signal_bar_size(instance)
                            ),
                            "shock_gate_mode": normalize_shock_gate_mode(instance.filters),
                            "shock_detector": normalize_shock_detector(instance.filters),
                            "shock_scale_detector": str(
                                self._filters_get_value(instance.filters, "shock_scale_detector") or ""
                            ).strip(),
                        },
                    },
                )

            instrument, open_items, open_dir = self._resolve_open_positions(
                instance,
                symbol=symbol,
                signal_contract=signal_contract,
            )

            if not open_items:
                if instance.open_direction is not None:
                    instance.open_direction = None
                    instance.spot_profit_target_price = None
                    instance.spot_stop_loss_price = None
                instance.last_resize_bar_ts = None
                self._clear_pending_exit(instance)
                self._reset_exit_retry_state(instance)
                self._spot_reset_exec_bar(instance)
                self._reset_quote_starvation_state(instance)

            if stale_signal and not open_items:
                self._clear_pending_entry(instance)
                _gate(
                    "BLOCKED_STALE_SIGNAL",
                    {
                        "bar_ts": snap.bar_ts.isoformat(),
                        "symbol": symbol,
                        "bar_health": bar_health,
                    },
                )
                continue
            if stale_signal and open_items:
                _gate(
                    "STALE_SIGNAL_HOLDING",
                    {
                        "bar_ts": snap.bar_ts.isoformat(),
                        "symbol": symbol,
                        "bar_health": bar_health,
                    },
                )
            if gap_signal and not open_items:
                self._clear_pending_entry(instance)
                _gate(
                    "WAITING_DATA_GAP",
                    {
                        "bar_ts": snap.bar_ts.isoformat(),
                        "symbol": symbol,
                        "bar_health": bar_health,
                    },
                )
                continue
            if gap_signal and open_items:
                _gate(
                    "DATA_GAP_HOLDING",
                    {
                        "bar_ts": snap.bar_ts.isoformat(),
                        "symbol": symbol,
                        "bar_health": bar_health,
                    },
                )

            if not open_items:
                preflight_ready, preflight_payload = self._signal_preflight_status(instance=instance, snap=snap)
                if not preflight_ready:
                    self._clear_pending_entry(instance)
                    _gate("WAITING_PREFLIGHT_BARS", preflight_payload)
                    continue

            if self._auto_process_pending_next_open(
                instance=instance,
                instrument=instrument,
                open_items=open_items,
                open_dir=open_dir,
                now_wall=now_wall,
                snap=snap,
                gate=_gate,
            ):
                break

            if self._has_pending_order(instance):
                _gate("PENDING_ORDER", None)
                continue

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

            if open_items:
                if await self._auto_maybe_exit_open_positions(
                    instance=instance,
                    snap=snap,
                    instrument=instrument,
                    open_items=open_items,
                    open_dir=open_dir,
                    now_et=now_et,
                    gate=_gate,
                ):
                    break
                if await self._auto_maybe_resize_open_positions(
                    instance=instance,
                    snap=snap,
                    instrument=instrument,
                    open_items=open_items,
                    open_dir=open_dir,
                    gate=_gate,
                ):
                    break
                continue

            entry_weekday_now = self._entry_weekday_for_ts(instance, now_et)
            if not self._can_order_now(instance, now_et=now_et):
                _gate(
                    "BLOCKED_WEEKDAY_NOW",
                    {
                        "now_weekday": int(now_et.weekday()),
                        "entry_weekday": int(entry_weekday_now),
                    },
                )
                continue

            entry_days = instance.strategy.get("entry_days", [])
            if entry_days:
                allowed_days = {_weekday_num(day) for day in entry_days}
            else:
                allowed_days = {0, 1, 2, 3, 4}
            signal_weekday = self._entry_weekday_for_ts(instance, snap.bar_ts)
            if signal_weekday not in allowed_days:
                _gate(
                    "BLOCKED_ENTRY_DAY",
                    {
                        "signal_weekday": int(snap.bar_ts.weekday()),
                        "entry_weekday": int(signal_weekday),
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
            filter_bar_ts = self._as_et_aware(snap.bar_ts)
            filter_checks = signal_filter_checks(
                instance.filters,
                bar_ts=filter_bar_ts,
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
            )
            if not all(bool(v) for v in filter_checks.values()):
                failed_filters = [name for name, ok in filter_checks.items() if not bool(ok)]
                _gate(
                    "BLOCKED_FILTERS",
                    {
                        "symbol": symbol,
                        "bar_ts": snap.bar_ts.isoformat(),
                        "bar_ts_et": filter_bar_ts.isoformat(),
                        "bar_hour_et": int(filter_bar_ts.hour),
                        "bars_in_day": int(snap.bars_in_day),
                        "cooldown_ok": bool(cooldown_ok),
                        "rv": float(snap.rv) if snap.rv is not None else None,
                        "volume": float(snap.volume) if snap.volume is not None else None,
                        "volume_ema": float(snap.volume_ema) if snap.volume_ema is not None else None,
                        "volume_ema_ready": bool(snap.volume_ema_ready),
                        "shock": snap.shock,
                        "shock_dir": snap.shock_dir,
                        "state": snap.signal.state,
                        "entry_dir": snap.signal.entry_dir,
                        "regime_dir": snap.signal.regime_dir,
                        "filter_checks": filter_checks,
                        "failed_filters": failed_filters,
                    },
                )
                continue

            if self._auto_try_queue_entry(instance=instance, snap=snap, gate=_gate, now_et=now_et):
                break

    @staticmethod
    def _int_from(raw: object, *, default: int, min_value: int = 0) -> int:
        try:
            value = int(raw if raw is not None else default)
        except (TypeError, ValueError):
            value = int(default)
        return max(int(min_value), int(value))

    @staticmethod
    def _ema_slow_period(preset_raw: object) -> int:
        preset = str(preset_raw or "").strip()
        if not preset:
            return 0
        periods = ema_periods(preset)
        if periods is None:
            return 0
        return max(int(periods[0]), int(periods[1]))

    def _signal_preflight_requirements(self, instance: _BotInstance) -> dict[str, object]:
        strategy = instance.strategy if isinstance(instance.strategy, dict) else {}
        filters = instance.filters if isinstance(instance.filters, dict) else {}
        bar_size = self._signal_bar_size(instance)

        signal_need = 0
        regime_need = 0
        regime2_need = 0
        active: list[dict[str, object]] = []

        def _register(bucket: str, label: str, bars: int) -> None:
            nonlocal signal_need, regime_need, regime2_need
            need = max(0, int(bars))
            if need <= 0:
                return
            if bucket == "signal":
                signal_need = max(signal_need, need)
            elif bucket == "regime":
                regime_need = max(regime_need, need)
            elif bucket == "regime2":
                regime2_need = max(regime2_need, need)
            active.append({"bucket": bucket, "name": label, "bars": need})

        entry_signal = normalize_spot_entry_signal(strategy.get("entry_signal"))
        if entry_signal == "ema":
            slow = self._ema_slow_period(strategy.get("ema_preset"))
            _register("signal", "entry_ema", slow)

        strategy_instrument = "spot"
        strategy_instrument_fn = getattr(self, "_strategy_instrument", None)
        if callable(strategy_instrument_fn):
            try:
                strategy_instrument = str(strategy_instrument_fn(strategy) or "spot").strip().lower()
            except Exception:
                strategy_instrument = "spot"
        elif str(strategy.get("instrument") or "spot").strip().lower() != "spot":
            strategy_instrument = "options"
        if strategy_instrument == "spot":
            runtime_spec = spot_runtime_spec_view(strategy=strategy, filters=filters)
            exit_mode = str(runtime_spec.exit_mode)
            if exit_mode == "atr":
                atr_period = self._int_from(strategy.get("spot_atr_period"), default=14, min_value=1)
                _register("signal", "spot_exit_atr", atr_period)

        if isinstance(filters, dict) and filters.get("volume_ratio_min") is not None:
            vol_period = self._int_from(filters.get("volume_ema_period"), default=20, min_value=1)
            _register("signal", "volume_ema", vol_period)

        regime_mode, regime_preset, _regime_bar_size, use_mtf_regime = resolve_spot_regime_spec(
            bar_size=bar_size,
            regime_mode_raw=strategy.get("regime_mode"),
            regime_ema_preset_raw=strategy.get("regime_ema_preset"),
            regime_bar_size_raw=strategy.get("regime_bar_size"),
        )
        if regime_mode == "supertrend":
            st_period = self._int_from(strategy.get("supertrend_atr_period"), default=10, min_value=1)
            _register("regime" if use_mtf_regime else "signal", "regime_supertrend", st_period)
        elif regime_mode == "ema":
            regime_slow = self._ema_slow_period(regime_preset)
            if regime_slow > 0:
                _register("regime" if use_mtf_regime else "signal", "regime_ema", regime_slow)

        regime2_mode, regime2_preset, _regime2_bar_size, use_mtf_regime2 = resolve_spot_regime2_spec(
            bar_size=bar_size,
            regime2_mode_raw=strategy.get("regime2_mode"),
            regime2_ema_preset_raw=strategy.get("regime2_ema_preset"),
            regime2_bar_size_raw=strategy.get("regime2_bar_size"),
        )
        if regime2_mode == "supertrend":
            st2_period = self._int_from(strategy.get("regime2_supertrend_atr_period"), default=10, min_value=1)
            _register("regime2" if use_mtf_regime2 else "signal", "regime2_supertrend", st2_period)
        elif regime2_mode == "ema":
            regime2_slow = self._ema_slow_period(regime2_preset)
            if regime2_slow > 0:
                _register("regime2" if use_mtf_regime2 else "signal", "regime2_ema", regime2_slow)

        return {
            "signal_bars_min": int(signal_need),
            "regime_bars_min": int(regime_need),
            "regime2_bars_min": int(regime2_need),
            "active_requirements": active,
        }

    def _signal_preflight_status(self, *, instance: _BotInstance, snap) -> tuple[bool, dict]:
        req = self._signal_preflight_requirements(instance)
        diag = getattr(self, "_last_signal_snapshot_diag", None)
        diag_dict = dict(diag) if isinstance(diag, dict) else {}

        def _have(name: str) -> int:
            try:
                raw = diag_dict.get(name, 0)
                return max(0, int(raw or 0))
            except (TypeError, ValueError):
                return 0

        signal_have = _have("bars_count")
        regime_have = _have("regime_bars_count")
        regime2_have = _have("regime2_bars_count")
        signal_need = int(req.get("signal_bars_min", 0) or 0)
        regime_need = int(req.get("regime_bars_min", 0) or 0)
        regime2_need = int(req.get("regime2_bars_min", 0) or 0)

        missing: list[str] = []
        if signal_have < signal_need:
            missing.append("signal")
        if regime_have < regime_need:
            missing.append("regime")
        if regime2_have < regime2_need:
            missing.append("regime2")

        payload = {
            "bar_ts": snap.bar_ts.isoformat(),
            "diag_stage": str(diag_dict.get("stage") or ""),
            "signal_have": int(signal_have),
            "signal_need": int(signal_need),
            "regime_have": int(regime_have),
            "regime_need": int(regime_need),
            "regime2_have": int(regime2_have),
            "regime2_need": int(regime2_need),
            "missing": list(missing),
            "requirements": list(req.get("active_requirements") or []),
        }
        return len(missing) == 0, payload

    def _has_pending_order(self, instance: _BotInstance) -> bool:
        return any(
            o.status in ("STAGED", "WORKING", "CANCELING") and o.instance_id == instance.instance_id
            for o in self._orders
        )

    @staticmethod
    def _wall_time(now_et: datetime) -> datetime:
        return now_et.replace(tzinfo=None) if now_et.tzinfo is not None else now_et

    @staticmethod
    def _as_et_aware(ts: datetime) -> datetime:
        # Runtime signal bars are ET wall-clock timestamps with tz stripped at ingest.
        # Reattach ET here before shared-core filters that convert to ET.
        return _to_et_shared(ts, naive_ts_mode="et", default_naive_ts_mode="et")

    @staticmethod
    def _filters_get_value(filters: dict | object | None, key: str) -> object | None:
        if isinstance(filters, dict):
            return filters.get(key)
        return getattr(filters, key, None) if filters is not None else None

    def _spot_riskoff_mode(self, instance: _BotInstance) -> str:
        return spot_riskoff_mode_from_filters(instance.filters)

    def _spot_riskoff_end_hour(self, instance: _BotInstance) -> int | None:
        return spot_riskoff_end_hour(instance.filters)

    def _signal_bars_held_since_entry(self, *, instance: _BotInstance, bar_ts: datetime) -> int:
        entry_ts = instance.last_entry_bar_ts
        if entry_ts is None:
            return 0
        bar_def = parse_bar_size(self._signal_bar_size(instance))
        if bar_def is None:
            return 0
        span_sec = float(bar_def.duration.total_seconds())
        if span_sec <= 0:
            return 0
        held_sec = float((bar_ts - entry_ts).total_seconds())
        if held_sec <= 0:
            return 0
        return max(0, int(held_sec / span_sec))

    def _spot_ratsv_probe_cancel_hit(
        self,
        *,
        instance: _BotInstance,
        pos: float,
        held_bars: int,
        entry_branch: str | None,
        tr_ratio: float | None,
        slope_med: float | None,
    ) -> bool:
        if str(entry_branch or "") != "a":
            return False
        if pos == 0:
            return False
        filters = instance.filters
        try:
            max_bars = int(self._filters_get_value(filters, "ratsv_probe_cancel_max_bars") or 0)
        except (TypeError, ValueError):
            max_bars = 0
        if max_bars <= 0 or held_bars > int(max_bars):
            return False
        try:
            slope_min = float(self._filters_get_value(filters, "ratsv_probe_cancel_slope_adverse_min_pct") or 0.0)
        except (TypeError, ValueError):
            slope_min = 0.0
        if slope_min <= 0 or slope_med is None:
            return False
        adverse = float(slope_med) <= -float(slope_min) if pos > 0 else float(slope_med) >= float(slope_min)
        if not bool(adverse):
            return False
        tr_min_raw = self._filters_get_value(filters, "ratsv_probe_cancel_tr_ratio_min")
        if tr_min_raw is not None:
            try:
                tr_min = float(tr_min_raw)
            except (TypeError, ValueError):
                tr_min = None
            if tr_min is not None and tr_min > 0:
                if tr_ratio is None or float(tr_ratio) < float(tr_min):
                    return False
        return True

    def _spot_ratsv_adverse_release_hit(
        self,
        *,
        instance: _BotInstance,
        pos: float,
        held_bars: int,
        tr_ratio: float | None,
        slope_med: float | None,
        slope_vel: float | None,
    ) -> bool:
        if pos == 0:
            return False
        filters = instance.filters
        try:
            slope_min = float(self._filters_get_value(filters, "ratsv_adverse_release_slope_adverse_min_pct") or 0.0)
        except (TypeError, ValueError):
            slope_min = 0.0
        if slope_min <= 0:
            return False
        try:
            min_hold = int(self._filters_get_value(filters, "ratsv_adverse_release_min_hold_bars") or 0)
        except (TypeError, ValueError):
            min_hold = 0
        if min_hold > 0 and held_bars < int(min_hold):
            return False
        adverse = False
        if slope_med is not None:
            adverse = float(slope_med) <= -float(slope_min) if pos > 0 else float(slope_med) >= float(slope_min)
        if (not adverse) and slope_vel is not None:
            adverse = float(slope_vel) <= -float(slope_min) if pos > 0 else float(slope_vel) >= float(slope_min)
        if not bool(adverse):
            return False
        tr_min_raw = self._filters_get_value(filters, "ratsv_adverse_release_tr_ratio_min")
        if tr_min_raw is not None:
            try:
                tr_min = float(tr_min_raw)
            except (TypeError, ValueError):
                tr_min = None
            if tr_min is not None and tr_min > 0:
                if tr_ratio is None or float(tr_ratio) < float(tr_min):
                    return False
        return True

    def _order_stage_timeout_sec(self, instance: _BotInstance) -> float:
        raw = instance.strategy.get("order_stage_timeout_sec")
        try:
            parsed = float(raw) if raw is not None else _DEFAULT_ORDER_STAGE_TIMEOUT_SEC
        except (TypeError, ValueError):
            parsed = _DEFAULT_ORDER_STAGE_TIMEOUT_SEC
        return max(1.0, parsed)

    def _mark_order_trigger_watch(
        self,
        *,
        instance: _BotInstance,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
        now_et: datetime,
        reason: str | None = None,
        mode: str | None = None,
    ) -> None:
        now_wall = self._wall_time(now_et)
        intent_clean = str(intent or "")
        reason_clean = str(reason or "")
        mode_clean = str(mode or "")
        direction_clean = str(direction or "")
        prev_key = (
            str(instance.order_trigger_intent or ""),
            str(instance.order_trigger_reason or ""),
            str(instance.order_trigger_mode or ""),
            str(instance.order_trigger_direction or ""),
            instance.order_trigger_signal_bar_ts.isoformat() if instance.order_trigger_signal_bar_ts is not None else "",
        )
        new_key = (
            intent_clean,
            reason_clean,
            mode_clean,
            direction_clean,
            signal_bar_ts.isoformat() if signal_bar_ts is not None else "",
        )
        same_trigger = bool(instance.order_trigger_ts is not None and prev_key == new_key)
        if same_trigger:
            instance.order_trigger_attempt = max(1, int(instance.order_trigger_attempt or 0)) + 1
            if instance.order_trigger_retry_reason is None:
                retry_raw = str(instance.order_trigger_last_error or "").strip()
                instance.order_trigger_retry_reason = retry_raw or "retry"
        else:
            instance.order_trigger_attempt = 1
            instance.order_trigger_retry_reason = None
            instance.order_trigger_last_error = None
        instance.order_trigger_intent = intent_clean
        instance.order_trigger_reason = reason_clean
        instance.order_trigger_mode = mode_clean
        instance.order_trigger_direction = direction_clean
        instance.order_trigger_signal_bar_ts = signal_bar_ts
        instance.order_trigger_ts = now_wall
        instance.order_trigger_deadline_ts = now_wall + timedelta(seconds=self._order_stage_timeout_sec(instance))

    @staticmethod
    def _clear_order_trigger_watch(instance: _BotInstance) -> None:
        instance.order_trigger_intent = None
        instance.order_trigger_reason = None
        instance.order_trigger_mode = None
        instance.order_trigger_direction = None
        instance.order_trigger_signal_bar_ts = None
        instance.order_trigger_ts = None
        instance.order_trigger_deadline_ts = None
        instance.order_trigger_attempt = 0
        instance.order_trigger_retry_reason = None
        instance.order_trigger_last_error = None

    @staticmethod
    def _order_trigger_watch_payload(instance: _BotInstance) -> dict[str, object]:
        payload: dict[str, object] = {}
        attempt = max(0, int(instance.order_trigger_attempt or 0))
        if attempt > 0:
            payload["order_attempt"] = int(attempt)
        retry_reason = str(instance.order_trigger_retry_reason or "").strip()
        if retry_reason:
            payload["retry_reason"] = retry_reason
        return payload

    def _check_order_trigger_watchdogs(self, *, now_et: datetime) -> None:
        now_wall = self._wall_time(now_et)
        for instance in self._instances:
            if instance.state != "RUNNING":
                self._clear_order_trigger_watch(instance)
                continue
            deadline = instance.order_trigger_deadline_ts
            if deadline is None or now_wall < deadline:
                continue
            if self._order_task and not self._order_task.done():
                continue
            if self._has_pending_order(instance):
                continue
            payload = {
                "intent": instance.order_trigger_intent,
                "reason": instance.order_trigger_reason,
                "mode": instance.order_trigger_mode,
                "direction": instance.order_trigger_direction,
                "trigger_ts": instance.order_trigger_ts.isoformat() if instance.order_trigger_ts else None,
                "signal_bar_ts": (
                    instance.order_trigger_signal_bar_ts.isoformat()
                    if instance.order_trigger_signal_bar_ts is not None
                    else None
                ),
                "deadline_ts": deadline.isoformat(),
            }
            payload.update(self._order_trigger_watch_payload(instance))
            self._journal_write(
                event="ORDER_DROPPED",
                instance=instance,
                order=None,
                reason="not_staged",
                data=payload,
            )
            self._clear_order_trigger_watch(instance)

    def _spot_exec_duration(self, instance: _BotInstance) -> timedelta:
        raw = str(instance.strategy.get("spot_exec_bar_size") or self._signal_bar_size(instance) or "").strip()
        parsed = parse_bar_size(raw)
        if parsed is None or parsed.duration.total_seconds() <= 0:
            return timedelta(minutes=5)
        return parsed.duration

    @staticmethod
    def _bar_floor(ts: datetime, span: timedelta) -> datetime:
        sec = float(span.total_seconds())
        if sec <= 0:
            return ts
        day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = max(0.0, float((ts - day_start).total_seconds()))
        slot = int(elapsed // sec)
        return day_start + timedelta(seconds=(slot * sec))

    @classmethod
    def _bar_ceil(cls, ts: datetime, span: timedelta) -> datetime:
        floor_ts = cls._bar_floor(ts, span)
        if floor_ts >= ts:
            return floor_ts
        return floor_ts + span

    def _spot_signal_close_ts(self, instance: _BotInstance, signal_bar_ts: datetime) -> datetime:
        signal_def = parse_bar_size(self._signal_bar_size(instance))
        if signal_def is None or signal_def.duration.total_seconds() <= 0:
            return signal_bar_ts
        return signal_bar_ts + signal_def.duration

    def _spot_next_open_due_ts(self, instance: _BotInstance, signal_bar_ts: datetime) -> datetime:
        close_ts = self._spot_signal_close_ts(instance, signal_bar_ts)
        span = self._spot_exec_duration(instance)
        due_ts = self._bar_ceil(close_ts, span)
        return self._spot_align_exec_open(instance=instance, ts=due_ts, span=span)

    def _spot_session_window(self, instance: _BotInstance) -> tuple[time, time] | None:
        mode = str(instance.strategy.get("spot_next_open_session", "auto") or "auto").strip().lower()
        if mode not in ("auto", "rth", "extended", "always"):
            mode = "auto"
        if mode == "always":
            return None
        if mode == "rth":
            return (_RTH_OPEN, _RTH_CLOSE)
        if mode == "extended":
            return (_EXT_OPEN, _EXT_CLOSE)

        sec_type = str(instance.strategy.get("spot_sec_type") or "STK").strip().upper()
        if sec_type != "STK":
            return None
        return (_RTH_OPEN, _RTH_CLOSE) if bool(self._signal_use_rth(instance)) else (_EXT_OPEN, _EXT_CLOSE)

    @staticmethod
    def _spot_next_weekday_open(ts: datetime, open_time: time) -> datetime:
        candidate = datetime.combine(ts.date(), open_time)
        if candidate <= ts:
            candidate = candidate + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate = candidate + timedelta(days=1)
        return candidate

    @staticmethod
    def _spot_session_contains(ts: datetime, *, open_time: time, close_time: time) -> bool:
        if ts.weekday() >= 5:
            return False
        current = ts.time()
        return open_time <= current < close_time

    def _spot_align_exec_open(self, *, instance: _BotInstance, ts: datetime, span: timedelta) -> datetime:
        session = self._spot_session_window(instance)
        if session is None:
            return ts
        open_time, close_time = session
        candidate = ts
        while True:
            if candidate.weekday() >= 5:
                candidate = self._spot_next_weekday_open(candidate, open_time)
                continue
            day_open = datetime.combine(candidate.date(), open_time)
            day_close = datetime.combine(candidate.date(), close_time)
            if candidate < day_open:
                candidate = day_open
            elif candidate >= day_close:
                candidate = self._spot_next_weekday_open(candidate, open_time)
                continue

            aligned = self._bar_ceil(candidate, span)
            if aligned >= day_close:
                candidate = self._spot_next_weekday_open(candidate, open_time)
                continue
            if not self._spot_session_contains(aligned, open_time=open_time, close_time=close_time):
                candidate = self._spot_next_weekday_open(candidate, open_time)
                continue
            return aligned

    @staticmethod
    def _spot_exec_feed_mode(instance: _BotInstance) -> str:
        mode = str(instance.strategy.get("spot_exec_feed_mode", "ticks_side") or "ticks_side").strip().lower()
        if mode not in ("poll", "ticks_side", "ticks_quote"):
            mode = "ticks_side"
        return mode

    @staticmethod
    def _spot_reset_exec_bar(instance: _BotInstance) -> None:
        instance.exec_bar_ts = None
        instance.exec_bar_open = None
        instance.exec_bar_high = None
        instance.exec_bar_low = None
        instance.exec_tick_cursor = 0
        instance.exec_tick_by_tick_cursor = 0

    @staticmethod
    def _reset_quote_starvation_state(instance: _BotInstance) -> None:
        instance.quote_starvation_since = None
        instance.quote_starvation_stage = None
        instance.quote_starvation_last_repair_ts = None
        instance.quote_starvation_last_kill_ts = None

    def _spot_quote_starvation_watchdog(
        self,
        *,
        instance: _BotInstance,
        now_wall: datetime,
        symbol: str,
        con_id: int,
        bid: float | None,
        ask: float | None,
        last: float | None,
        ticker_market_price: float | None,
        market_price: float | None,
        gate,
    ) -> dict[str, object]:
        strategy = instance.strategy if isinstance(instance.strategy, dict) else {}
        warn_sec = float(self._int_from(strategy.get("spot_quote_starve_warn_sec"), default=45, min_value=5))
        kill_sec = float(self._int_from(strategy.get("spot_quote_starve_kill_sec"), default=180, min_value=10))
        if kill_sec < warn_sec:
            kill_sec = warn_sec
        repair_cooldown_sec = float(
            self._int_from(strategy.get("spot_quote_starve_repair_cooldown_sec"), default=15, min_value=3)
        )
        action = str(strategy.get("spot_quote_starve_action") or "repair_only").strip().lower()
        if action not in ("repair_only", "force_exit"):
            action = "repair_only"

        has_live_quote = any(
            value is not None and float(value) > 0.0
            for value in (bid, ask, last, ticker_market_price, market_price)
        )
        if has_live_quote:
            since = instance.quote_starvation_since
            if since is not None:
                down_sec = max(0.0, float((now_wall - since).total_seconds()))
                gate(
                    "QUOTE_STARVE_RECOVERED",
                    {
                        "symbol": symbol,
                        "down_sec": down_sec,
                    },
                )
            self._reset_quote_starvation_state(instance)
            return {"starving": False, "since_sec": 0.0, "force_exit": False, "action": action}

        if instance.quote_starvation_since is None:
            instance.quote_starvation_since = now_wall
            instance.quote_starvation_stage = None

        since = instance.quote_starvation_since
        since_sec = max(0.0, float((now_wall - since).total_seconds())) if since is not None else 0.0
        if since_sec >= kill_sec:
            stage = "kill"
        elif since_sec >= warn_sec:
            stage = "warn"
        else:
            stage = "fresh"

        if instance.quote_starvation_stage != stage:
            instance.quote_starvation_stage = stage
            gate(
                "QUOTE_STARVING_HOLDING",
                {
                    "symbol": symbol,
                    "stage": stage,
                    "since_sec": since_sec,
                    "warn_sec": warn_sec,
                    "kill_sec": kill_sec,
                    "proxy_error": self._client.proxy_error(),
                },
            )
            if stage == "kill":
                gate(
                    "QUOTE_STARVE_KILL_SWITCH",
                    {
                        "symbol": symbol,
                        "action": action,
                        "since_sec": since_sec,
                    },
                )

        last_repair = instance.quote_starvation_last_repair_ts
        repair_due = (
            last_repair is None
            or (now_wall - last_repair).total_seconds() >= float(repair_cooldown_sec)
        )
        if stage in ("warn", "kill") and repair_due:
            if int(con_id or 0) > 0:
                self._client.release_ticker(int(con_id), owner="bot")
            self._client.start_proxy_tickers()
            instance.quote_starvation_last_repair_ts = now_wall
            gate(
                "QUOTE_STARVE_REPAIR",
                {
                    "symbol": symbol,
                    "stage": stage,
                    "repair_cooldown_sec": repair_cooldown_sec,
                    "proxy_error": self._client.proxy_error(),
                },
            )

        force_exit = False
        if stage == "kill" and action == "force_exit":
            last_kill = instance.quote_starvation_last_kill_ts
            if last_kill is None or (now_wall - last_kill).total_seconds() >= 30.0:
                instance.quote_starvation_last_kill_ts = now_wall
                force_exit = True

        return {
            "starving": True,
            "since_sec": since_sec,
            "stage": stage,
            "force_exit": bool(force_exit),
            "action": action,
        }

    def _spot_exec_tick_prices(self, *, instance: _BotInstance, ticker, qty_sign: int) -> list[float]:
        mode = self._spot_exec_feed_mode(instance)
        if mode == "poll":
            return []
        prices: list[float] = []

        raw_ticks = list(getattr(ticker, "ticks", []) or [])
        tick_cursor = int(instance.exec_tick_cursor or 0)
        if tick_cursor < 0 or tick_cursor > len(raw_ticks):
            tick_cursor = 0
        new_ticks = raw_ticks[tick_cursor:]
        instance.exec_tick_cursor = len(raw_ticks)
        if len(new_ticks) > 256:
            new_ticks = new_ticks[-256:]
        for tick in new_ticks:
            price = _safe_num(getattr(tick, "price", None))
            if price is None or price <= 0:
                continue
            try:
                tick_type = int(getattr(tick, "tickType", -1) or -1)
            except (TypeError, ValueError):
                tick_type = -1
            if mode == "ticks_quote":
                if tick_type in _BID_TICK_TYPES or tick_type in _ASK_TICK_TYPES:
                    prices.append(float(price))
                continue
            if qty_sign > 0 and tick_type in _BID_TICK_TYPES:
                prices.append(float(price))
            elif qty_sign < 0 and tick_type in _ASK_TICK_TYPES:
                prices.append(float(price))

        raw_tbt = list(getattr(ticker, "tickByTicks", []) or [])
        tbt_cursor = int(instance.exec_tick_by_tick_cursor or 0)
        if tbt_cursor < 0 or tbt_cursor > len(raw_tbt):
            tbt_cursor = 0
        new_tbt = raw_tbt[tbt_cursor:]
        instance.exec_tick_by_tick_cursor = len(raw_tbt)
        if len(new_tbt) > 256:
            new_tbt = new_tbt[-256:]
        for item in new_tbt:
            bid_p = _safe_num(getattr(item, "bidPrice", None))
            ask_p = _safe_num(getattr(item, "askPrice", None))
            if mode == "ticks_quote":
                if bid_p is not None and bid_p > 0:
                    prices.append(float(bid_p))
                if ask_p is not None and ask_p > 0:
                    prices.append(float(ask_p))
                continue
            if qty_sign > 0 and bid_p is not None and bid_p > 0:
                prices.append(float(bid_p))
            elif qty_sign < 0 and ask_p is not None and ask_p > 0:
                prices.append(float(ask_p))
        return prices

    def _spot_exec_price_points(
        self,
        *,
        instance: _BotInstance,
        ticker,
        qty_sign: int,
        market_price: float | None,
        bid: float | None,
        ask: float | None,
    ) -> list[float]:
        prices: list[float] = []
        if qty_sign > 0 and bid is not None and bid > 0:
            prices.append(float(bid))
        elif qty_sign < 0 and ask is not None and ask > 0:
            prices.append(float(ask))
        elif market_price is not None and market_price > 0:
            prices.append(float(market_price))

        mode = self._spot_exec_feed_mode(instance)
        if mode == "ticks_quote":
            if bid is not None and bid > 0:
                prices.append(float(bid))
            if ask is not None and ask > 0:
                prices.append(float(ask))
        prices.extend(self._spot_exec_tick_prices(instance=instance, ticker=ticker, qty_sign=qty_sign))
        if not prices and market_price is not None and market_price > 0:
            prices.append(float(market_price))
        return prices

    def _spot_update_exec_bar(self, instance: _BotInstance, *, now_wall: datetime, prices: list[float]) -> None:
        cleaned: list[float] = []
        for raw in prices:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                cleaned.append(value)
        if not cleaned:
            return
        span = self._spot_exec_duration(instance)
        bar_ts = self._bar_floor(now_wall, span)
        open_price = float(cleaned[0])
        high_price = max(cleaned)
        low_price = min(cleaned)
        if instance.exec_bar_ts != bar_ts:
            instance.exec_bar_ts = bar_ts
            instance.exec_bar_open = open_price
            instance.exec_bar_high = high_price
            instance.exec_bar_low = low_price
            return
        hi = instance.exec_bar_high
        lo = instance.exec_bar_low
        instance.exec_bar_high = high_price if hi is None else max(float(hi), high_price)
        instance.exec_bar_low = low_price if lo is None else min(float(lo), low_price)

    @staticmethod
    def _clear_pending_entry(instance: _BotInstance) -> None:
        instance.pending_entry_direction = None
        instance.pending_entry_signal_bar_ts = None
        instance.pending_entry_due_ts = None

    @staticmethod
    def _clear_pending_exit(instance: _BotInstance) -> None:
        instance.pending_exit_reason = None
        instance.pending_exit_signal_bar_ts = None
        instance.pending_exit_due_ts = None

    @staticmethod
    def _reset_exit_retry_state(instance: _BotInstance) -> None:
        instance.exit_retry_bar_ts = None
        instance.exit_retry_count = 0
        instance.exit_retry_cooldown_until = None

    def _schedule_pending_entry_next_open(
        self,
        *,
        instance: _BotInstance,
        direction: str,
        signal_bar_ts: datetime,
        now_wall: datetime,
        gate,
    ) -> bool:
        due_from_ts = self._spot_signal_close_ts(instance, signal_bar_ts)
        due_ts = self._spot_next_open_due_ts(instance, signal_bar_ts)
        if due_ts <= now_wall:
            self._queue_order(
                instance,
                intent="enter",
                direction=direction,
                signal_bar_ts=signal_bar_ts,
                trigger_reason="next_open",
                trigger_mode="spot",
            )
            gate(
                "TRIGGER_ENTRY",
                {
                    "direction": direction,
                    "fill_mode": "next_open",
                    "next_open_due": due_ts.isoformat(),
                    "next_open_due_from": due_from_ts.isoformat(),
                    "now_wall_ts": now_wall.isoformat(),
                    "signal_bar_ts": signal_bar_ts.isoformat(),
                    **self._order_trigger_watch_payload(instance),
                },
            )
            return True

        if instance.pending_entry_due_ts is not None:
            return False

        instance.pending_entry_direction = str(direction)
        instance.pending_entry_signal_bar_ts = signal_bar_ts
        instance.pending_entry_due_ts = due_ts
        gate(
            "PENDING_ENTRY_NEXT_OPEN",
            {
                "direction": direction,
                "next_open_due": due_ts.isoformat(),
                "next_open_due_from": due_from_ts.isoformat(),
                "now_wall_ts": now_wall.isoformat(),
                "signal_bar_ts": signal_bar_ts.isoformat(),
            },
        )
        return True

    def _schedule_pending_exit_next_open(
        self,
        *,
        instance: _BotInstance,
        reason: str,
        direction: str | None,
        signal_bar_ts: datetime,
        now_wall: datetime,
        mode: str,
        gate,
    ) -> bool:
        due_from_ts = self._spot_signal_close_ts(instance, signal_bar_ts)
        due_ts = self._spot_next_open_due_ts(instance, signal_bar_ts)
        if due_ts <= now_wall:
            self._queue_order(
                instance,
                intent="exit",
                direction=direction,
                signal_bar_ts=signal_bar_ts,
                trigger_reason=reason,
                trigger_mode=mode,
            )
            gate(
                "TRIGGER_EXIT",
                {
                    "mode": mode,
                    "reason": reason,
                    "fill_mode": "next_open",
                    "next_open_due": due_ts.isoformat(),
                    "next_open_due_from": due_from_ts.isoformat(),
                    "now_wall_ts": now_wall.isoformat(),
                    "signal_bar_ts": signal_bar_ts.isoformat(),
                    **self._order_trigger_watch_payload(instance),
                },
            )
            return True

        if instance.pending_exit_due_ts is not None:
            return False

        instance.pending_exit_reason = str(reason)
        instance.pending_exit_signal_bar_ts = signal_bar_ts
        instance.pending_exit_due_ts = due_ts
        gate(
            "PENDING_EXIT_NEXT_OPEN",
            {
                "mode": mode,
                "reason": reason,
                "next_open_due": due_ts.isoformat(),
                "next_open_due_from": due_from_ts.isoformat(),
                "now_wall_ts": now_wall.isoformat(),
                "signal_bar_ts": signal_bar_ts.isoformat(),
            },
        )
        return True

    def _auto_process_pending_next_open(
        self,
        *,
        instance: _BotInstance,
        instrument: str,
        open_items: list,
        open_dir: str | None,
        now_wall: datetime,
        snap,
        gate,
    ) -> bool:
        if instrument != "spot":
            self._clear_pending_entry(instance)
            self._clear_pending_exit(instance)
            return False
        pending_exit_due = instance.pending_exit_due_ts
        pending_exit_reason = str(instance.pending_exit_reason or "flip")
        pending_exit_signal_bar_ts = instance.pending_exit_signal_bar_ts
        pending_entry_due = instance.pending_entry_due_ts
        pending_entry_direction = instance.pending_entry_direction
        pending_entry_signal_bar_ts = instance.pending_entry_signal_bar_ts
        pending_entry_set_date = pending_entry_signal_bar_ts.date() if pending_entry_signal_bar_ts is not None else None

        risk = getattr(snap, "risk", None) if snap is not None else None
        shock_dir_now = str(getattr(snap, "shock_dir", "") or "").strip().lower() if snap is not None else ""
        if shock_dir_now not in ("up", "down"):
            shock_dir_now = None
        riskoff_mode = self._spot_riskoff_mode(instance)
        riskoff_end_hour = self._spot_riskoff_end_hour(instance)

        decision = decide_pending_next_open(
            now_ts=now_wall,
            has_open=bool(open_items),
            open_dir=open_dir,
            pending_entry_dir=pending_entry_direction,
            pending_entry_set_date=pending_entry_set_date,
            pending_entry_due_ts=pending_entry_due,
            pending_exit_reason=pending_exit_reason,
            pending_exit_due_ts=pending_exit_due,
            risk_overlay_enabled=bool(risk is not None),
            riskoff_today=bool(getattr(risk, "riskoff", False)) if risk is not None else False,
            riskpanic_today=bool(getattr(risk, "riskpanic", False)) if risk is not None else False,
            riskpop_today=bool(getattr(risk, "riskpop", False)) if risk is not None else False,
            riskoff_mode=riskoff_mode,
            shock_dir_now=shock_dir_now,
            riskoff_end_hour=riskoff_end_hour,
            naive_ts_mode="et",
        )

        if decision.pending_clear_exit:
            self._clear_pending_exit(instance)
        if decision.pending_clear_entry:
            self._clear_pending_entry(instance)

        if decision.intent == "exit":
            due_from_ts = (
                self._spot_signal_close_ts(instance, pending_exit_signal_bar_ts)
                if pending_exit_signal_bar_ts is not None
                else (pending_exit_due or now_wall)
            )
            self._queue_order(
                instance,
                intent="exit",
                direction=open_dir,
                signal_bar_ts=pending_exit_signal_bar_ts,
                trigger_reason=pending_exit_reason,
                trigger_mode="spot",
            )
            gate(
                "TRIGGER_EXIT",
                {
                    "mode": "spot",
                    "reason": pending_exit_reason,
                    "fill_mode": "next_open",
                    "next_open_due": pending_exit_due.isoformat() if pending_exit_due is not None else None,
                    "next_open_due_from": due_from_ts.isoformat(),
                    "now_wall_ts": now_wall.isoformat(),
                    "signal_bar_ts": pending_exit_signal_bar_ts.isoformat() if pending_exit_signal_bar_ts is not None else None,
                    "spot_lifecycle": decision.as_payload(),
                    **self._order_trigger_watch_payload(instance),
                },
            )
            return True

        if decision.intent == "enter":
            due_from_ts = (
                self._spot_signal_close_ts(instance, pending_entry_signal_bar_ts)
                if pending_entry_signal_bar_ts is not None
                else (pending_entry_due or now_wall)
            )
            self._queue_order(
                instance,
                intent="enter",
                direction=pending_entry_direction,
                signal_bar_ts=pending_entry_signal_bar_ts,
                trigger_reason="next_open",
                trigger_mode="spot",
            )
            gate(
                "TRIGGER_ENTRY",
                {
                    "direction": pending_entry_direction,
                    "fill_mode": "next_open",
                    "next_open_due": pending_entry_due.isoformat() if pending_entry_due is not None else None,
                    "next_open_due_from": due_from_ts.isoformat(),
                    "now_wall_ts": now_wall.isoformat(),
                    "signal_bar_ts": (
                        pending_entry_signal_bar_ts.isoformat()
                        if pending_entry_signal_bar_ts is not None
                        else None
                    ),
                    "spot_lifecycle": decision.as_payload(),
                    **self._order_trigger_watch_payload(instance),
                },
            )
            return True

        if decision.gate == "CANCEL_PENDING_ENTRY_RISK_OVERLAY":
            gate(
                "CANCEL_PENDING_ENTRY_RISK_OVERLAY",
                {
                    "direction": str(pending_entry_direction) if pending_entry_direction is not None else None,
                    "next_open_due": pending_entry_due.isoformat() if pending_entry_due is not None else None,
                    "signal_bar_ts": (
                        pending_entry_signal_bar_ts.isoformat()
                        if pending_entry_signal_bar_ts is not None
                        else None
                    ),
                    "riskoff_mode": riskoff_mode,
                    "shock_dir": shock_dir_now,
                    "riskoff": bool(getattr(risk, "riskoff", False)) if risk is not None else False,
                    "riskpanic": bool(getattr(risk, "riskpanic", False)) if risk is not None else False,
                    "riskpop": bool(getattr(risk, "riskpop", False)) if risk is not None else False,
                    "riskoff_end_hour": riskoff_end_hour,
                    "spot_lifecycle": decision.as_payload(),
                },
            )
        elif decision.gate == "PENDING_EXIT_NEXT_OPEN" and instance.pending_exit_due_ts is not None:
            gate(
                "PENDING_EXIT_NEXT_OPEN",
                {
                    "mode": "spot",
                    "reason": str(instance.pending_exit_reason or "flip"),
                    "next_open_due": instance.pending_exit_due_ts.isoformat(),
                    "next_open_due_from": (
                        self._spot_signal_close_ts(instance, instance.pending_exit_signal_bar_ts).isoformat()
                        if instance.pending_exit_signal_bar_ts is not None
                        else instance.pending_exit_due_ts.isoformat()
                    ),
                    "signal_bar_ts": (
                        instance.pending_exit_signal_bar_ts.isoformat()
                        if instance.pending_exit_signal_bar_ts is not None
                        else None
                    ),
                    "spot_lifecycle": decision.as_payload(),
                },
            )
        elif decision.gate == "PENDING_ENTRY_NEXT_OPEN" and instance.pending_entry_due_ts is not None:
            gate(
                "PENDING_ENTRY_NEXT_OPEN",
                {
                    "direction": instance.pending_entry_direction,
                    "next_open_due": instance.pending_entry_due_ts.isoformat(),
                    "next_open_due_from": (
                        self._spot_signal_close_ts(instance, instance.pending_entry_signal_bar_ts).isoformat()
                        if instance.pending_entry_signal_bar_ts is not None
                        else instance.pending_entry_due_ts.isoformat()
                    ),
                    "signal_bar_ts": (
                        instance.pending_entry_signal_bar_ts.isoformat()
                        if instance.pending_entry_signal_bar_ts is not None
                        else None
                    ),
                    "spot_lifecycle": decision.as_payload(),
                },
            )
        return False

    async def _auto_maybe_exit_open_positions(
        self,
        *,
        instance: _BotInstance,
        snap,
        instrument: str,
        open_items: list,
        open_dir: str | None,
        now_et: datetime,
        gate,
    ) -> bool:
        if instance.exit_retry_bar_ts is not None and instance.exit_retry_bar_ts != snap.bar_ts:
            instance.exit_retry_bar_ts = snap.bar_ts
            instance.exit_retry_count = 0
            instance.exit_retry_cooldown_until = None
        elif instance.exit_retry_bar_ts is None:
            instance.exit_retry_bar_ts = snap.bar_ts

        if instance.last_exit_bar_ts is not None and instance.last_exit_bar_ts == snap.bar_ts:
            gate(
                "BLOCKED_EXIT_SAME_BAR",
                {
                    "bar_ts": snap.bar_ts.isoformat(),
                    "direction": open_dir,
                    "items": len(open_items),
                },
            )
            return False

        retry_count = max(0, int(instance.exit_retry_count or 0))
        raw_retry_limit = instance.strategy.get("exit_retry_max_per_bar", _DEFAULT_EXIT_RETRY_MAX_PER_BAR)
        try:
            retry_limit = int(raw_retry_limit)
        except (TypeError, ValueError):
            retry_limit = _DEFAULT_EXIT_RETRY_MAX_PER_BAR
        retry_limit = max(0, retry_limit)
        hard_limit = max(1, retry_limit)
        if retry_count >= hard_limit:
            gate(
                "BLOCKED_EXIT_RETRY_LIMIT",
                {
                    "bar_ts": snap.bar_ts.isoformat(),
                    "retry_count": retry_count,
                    "retry_limit": retry_limit,
                },
            )
            return False

        cooldown_until = instance.exit_retry_cooldown_until
        if cooldown_until is not None:
            now_wall = self._wall_time(now_et)
            if now_wall < cooldown_until:
                gate(
                    "BLOCKED_EXIT_RETRY_COOLDOWN",
                    {
                        "bar_ts": snap.bar_ts.isoformat(),
                        "retry_count": retry_count,
                        "cooldown_until": cooldown_until.isoformat(),
                    },
                )
                return False

        gate("HOLDING", {"direction": open_dir, "items": len(open_items)})

        def _trigger_exit(
            reason: str,
            *,
            mode: str = instrument,
            calc: dict | None = None,
            lifecycle=None,
        ) -> bool:
            self._queue_order(
                instance,
                intent="exit",
                direction=open_dir,
                signal_bar_ts=snap.bar_ts,
                trigger_reason=reason,
                trigger_mode=mode,
            )
            payload = {"mode": mode, "reason": reason}
            if calc:
                payload["calc"] = calc
            if lifecycle is not None:
                as_payload = getattr(lifecycle, "as_payload", None)
                payload["spot_lifecycle"] = as_payload() if callable(as_payload) else lifecycle
            payload.update(self._order_trigger_watch_payload(instance))
            gate("TRIGGER_EXIT", payload)
            return True

        if instrument == "spot":
            open_item = open_items[0]
            contract_symbol = str(
                getattr(getattr(open_item, "contract", None), "symbol", "")
                or getattr(instance, "symbol", "")
                or ""
            ).strip().upper()
            contract_con_id = int(getattr(getattr(open_item, "contract", None), "conId", 0) or 0)
            try:
                pos = float(getattr(open_item, "position", 0.0) or 0.0)
            except (TypeError, ValueError):
                pos = 0.0
            qty_sign = 1 if pos > 0 else -1 if pos < 0 else 0
            broker_avg_cost = _safe_num(getattr(open_item, "averageCost", None))
            tracked_entry_price = _safe_num(instance.spot_entry_basis_price)
            recent_enter_price = None
            for prior in reversed(self._orders):
                if int(getattr(prior, "instance_id", 0) or 0) != int(instance.instance_id):
                    continue
                if str(getattr(prior, "intent", "") or "").strip().lower() != "enter":
                    continue
                if str(getattr(prior, "status", "") or "").strip().upper() != "FILLED":
                    continue
                sec_type = str(getattr(getattr(prior, "order_contract", None), "secType", "") or "").strip().upper()
                if sec_type != "STK":
                    continue
                px = _safe_num(getattr(prior, "limit_price", None))
                if px is not None and px > 0:
                    recent_enter_price = float(px)
                    break
            basis_guard_delta_pct = None
            if tracked_entry_price is not None and tracked_entry_price > 0 and pos:
                avg_cost = float(tracked_entry_price)
                entry_basis_source = str(instance.spot_entry_basis_source or "tracked_entry")
            elif (
                broker_avg_cost is not None
                and broker_avg_cost > 0
                and recent_enter_price is not None
                and pos
            ):
                try:
                    basis_guard_delta_pct = abs(float(broker_avg_cost) - float(recent_enter_price)) / float(
                        recent_enter_price
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    basis_guard_delta_pct = None
                if basis_guard_delta_pct is not None and basis_guard_delta_pct > 0.003:
                    avg_cost = float(recent_enter_price)
                    entry_basis_source = "recent_enter_price_guard"
                else:
                    avg_cost = float(broker_avg_cost)
                    entry_basis_source = "portfolio_avg_cost"
            elif broker_avg_cost is not None and broker_avg_cost > 0 and pos:
                avg_cost = float(broker_avg_cost)
                entry_basis_source = "portfolio_avg_cost"
            elif recent_enter_price is not None and recent_enter_price > 0 and pos:
                avg_cost = float(recent_enter_price)
                entry_basis_source = "recent_enter_price_fallback"
            else:
                avg_cost = None
                entry_basis_source = None
            portfolio_market_price = _safe_num(getattr(open_item, "marketPrice", None))
            ticker = await self._client.ensure_ticker(open_item.contract, owner="bot")
            bid = _safe_num(getattr(ticker, "bid", None))
            ask = _safe_num(getattr(ticker, "ask", None))
            last = _safe_num(getattr(ticker, "last", None))
            mid = None
            if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
                mid = (bid + ask) / 2.0
            ticker_market_price = _ticker_price(ticker)

            quote_mid_last = mid if mid is not None else last
            quote_liq = quote_mid_last
            quote_liq_source = "mid" if mid is not None else "last" if last is not None else None
            if pos > 0 and bid is not None:
                quote_liq = bid
                quote_liq_source = "bid"
            elif pos < 0 and ask is not None:
                quote_liq = ask
                quote_liq_source = "ask"

            runtime_spec = spot_runtime_spec_view(strategy=instance.strategy, filters=instance.filters)
            mark_to_market = str(runtime_spec.mark_to_market)

            if mark_to_market == "liquidation":
                quote_market_price = quote_liq
                quote_market_source = quote_liq_source
            else:
                quote_market_price = quote_mid_last
                quote_market_source = "mid" if mid is not None else "last" if last is not None else None

            market_price = quote_market_price if quote_market_price is not None else portfolio_market_price
            market_price_source = quote_market_source if quote_market_price is not None else (
                "portfolio" if portfolio_market_price is not None else None
            )
            now_wall = self._wall_time(now_et)
            quote_watch = self._spot_quote_starvation_watchdog(
                instance=instance,
                now_wall=now_wall,
                symbol=contract_symbol,
                con_id=contract_con_id,
                bid=bid,
                ask=ask,
                last=last,
                ticker_market_price=ticker_market_price,
                market_price=market_price,
                gate=gate,
            )
            if bool(quote_watch.get("force_exit")):
                return _trigger_exit("quote_starvation_kill", mode="spot")
            quote_starving = bool(quote_watch.get("starving"))
            quote_starve_since_sec = (
                float(quote_watch.get("since_sec"))
                if quote_watch.get("since_sec") is not None
                else None
            )
            quote_watch_action = str(quote_watch.get("action") or "")
            if quote_starving:
                market_price = None
                market_price_source = None
            held_bars = self._signal_bars_held_since_entry(instance=instance, bar_ts=snap.bar_ts)
            entry_branch = str(instance.spot_entry_branch) if instance.spot_entry_branch in ("a", "b") else None
            ratsv_tr_ratio = (
                float(snap.ratsv_tr_ratio) if getattr(snap, "ratsv_tr_ratio", None) is not None else None
            )
            ratsv_slope_med = (
                float(snap.ratsv_fast_slope_med_pct)
                if getattr(snap, "ratsv_fast_slope_med_pct", None) is not None
                else None
            )
            ratsv_slope_vel = (
                float(snap.ratsv_fast_slope_vel_pct)
                if getattr(snap, "ratsv_fast_slope_vel_pct", None) is not None
                else None
            )
            exec_prices = self._spot_exec_price_points(
                instance=instance,
                ticker=ticker,
                qty_sign=qty_sign,
                market_price=market_price,
                bid=bid,
                ask=ask,
            )
            self._spot_update_exec_bar(instance, now_wall=now_wall, prices=exec_prices)
            exec_bar_ts = instance.exec_bar_ts
            exec_bar_open = instance.exec_bar_open
            exec_bar_high = instance.exec_bar_high
            exec_bar_low = instance.exec_bar_low
            session_bucket = _market_session_bucket(now_et)
            session_label = "AFTER" if session_bucket == "POST" else session_bucket

            def _spot_calc(
                *,
                move: float | None = None,
                threshold: float | None = None,
                target: float | None = None,
                stop: float | None = None,
                pt: float | None = None,
                sl: float | None = None,
                stop_level: float | None = None,
                profit_level: float | None = None,
                exec_hit_ref: float | None = None,
            ) -> dict:
                return {
                    "session": session_label,
                    "pos": float(pos),
                    "avg_cost": float(avg_cost) if avg_cost is not None else None,
                    "entry_basis_source": entry_basis_source,
                    "broker_avg_cost": float(broker_avg_cost) if broker_avg_cost is not None else None,
                    "tracked_entry_price": float(tracked_entry_price) if tracked_entry_price is not None else None,
                    "recent_enter_price": float(recent_enter_price) if recent_enter_price is not None else None,
                    "basis_guard_delta_pct": (
                        float(basis_guard_delta_pct) if basis_guard_delta_pct is not None else None
                    ),
                    "tracked_entry_set_ts": (
                        instance.spot_entry_basis_set_ts.isoformat()
                        if instance.spot_entry_basis_set_ts is not None
                        else None
                    ),
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "last": float(last) if last is not None else None,
                    "mid": float(mid) if mid is not None else None,
                    "ticker_market_price": float(ticker_market_price) if ticker_market_price is not None else None,
                    "portfolio_market_price": float(portfolio_market_price) if portfolio_market_price is not None else None,
                    "market_price_used": float(market_price) if market_price is not None else None,
                    "market_price_source": market_price_source,
                    "quote_starving": bool(quote_starving),
                    "quote_starve_since_sec": quote_starve_since_sec,
                    "quote_starve_action": quote_watch_action,
                    "exec_feed_mode": self._spot_exec_feed_mode(instance),
                    "exec_prices_count": int(len(exec_prices)),
                    "mark_to_market": mark_to_market,
                    "target_price": float(target) if target is not None else None,
                    "stop_price": float(stop) if stop is not None else None,
                    "profit_target_pct": float(pt) if pt is not None else None,
                    "stop_loss_pct": float(sl) if sl is not None else None,
                    "threshold": float(threshold) if threshold is not None else None,
                    "move": float(move) if move is not None else None,
                    "stop_level": float(stop_level) if stop_level is not None else None,
                    "profit_level": float(profit_level) if profit_level is not None else None,
                    "entry_branch": entry_branch,
                    "held_bars": int(held_bars),
                    "ratsv_tr_ratio": float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                    "ratsv_slope_med_pct": float(ratsv_slope_med) if ratsv_slope_med is not None else None,
                    "ratsv_slope_vel_pct": float(ratsv_slope_vel) if ratsv_slope_vel is not None else None,
                    "exec_bar_ts": exec_bar_ts.isoformat() if exec_bar_ts is not None else None,
                    "exec_bar_open": float(exec_bar_open) if exec_bar_open is not None else None,
                    "exec_bar_high": float(exec_bar_high) if exec_bar_high is not None else None,
                    "exec_bar_low": float(exec_bar_low) if exec_bar_low is not None else None,
                    "exec_hit_ref": float(exec_hit_ref) if exec_hit_ref is not None else None,
                    "contract_exchange": str(getattr(getattr(open_item, "contract", None), "exchange", "") or ""),
                }

            target_price = instance.spot_profit_target_price
            stop_price = instance.spot_stop_loss_price

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
            sl_mult, pt_mult = spot_shock_exit_pct_multipliers(instance.filters, shock=snap.shock)
            sl, pt = spot_scale_exit_pcts(
                stop_loss_pct=sl,
                profit_target_pct=pt,
                stop_mult=sl_mult,
                profit_mult=pt_mult,
            )

            profit_level = (
                spot_profit_level(
                    float(avg_cost),
                    qty_sign,
                    profit_target_price=target_price,
                    profit_target_pct=pt,
                )
                if avg_cost is not None and avg_cost > 0 and pos
                else spot_profit_level(
                    1.0,
                    qty_sign,
                    profit_target_price=target_price,
                    profit_target_pct=None,
                )
            )
            stop_level = (
                spot_stop_level(
                    float(avg_cost),
                    qty_sign,
                    stop_loss_price=stop_price,
                    stop_loss_pct=sl,
                )
                if avg_cost is not None and avg_cost > 0 and pos
                else spot_stop_level(
                    1.0,
                    qty_sign,
                    stop_loss_price=stop_price,
                    stop_loss_pct=None,
                )
            )

            intrabar_enabled = bool(runtime_spec.intrabar_exits)
            if (
                pos
                and intrabar_enabled
                and exec_bar_open is not None
                and exec_bar_high is not None
                and exec_bar_low is not None
                and (stop_level is not None or profit_level is not None)
            ):
                hit = spot_intrabar_exit(
                    qty=qty_sign,
                    bar_open=float(exec_bar_open),
                    bar_high=float(exec_bar_high),
                    bar_low=float(exec_bar_low),
                    stop_level=stop_level,
                    profit_level=profit_level,
                )
                if hit is not None:
                    kind, ref = hit
                    if kind == "stop":
                        reason_key = "stop_loss" if stop_price is not None else "stop_loss_pct"
                    else:
                        reason_key = "profit_target" if target_price is not None else "profit_target_pct"
                    return _trigger_exit(
                        reason_key,
                        mode="spot",
                        calc=_spot_calc(
                            target=float(target_price) if target_price is not None else None,
                            stop=float(stop_price) if stop_price is not None else None,
                            pt=pt,
                            sl=sl,
                            stop_level=stop_level,
                            profit_level=profit_level,
                            exec_hit_ref=float(ref),
                        ),
                    )

            move = None
            if avg_cost is not None and avg_cost > 0 and market_price is not None and market_price > 0 and pos:
                move = (market_price - avg_cost) / avg_cost
                if pos < 0:
                    move = -move

            if pos and market_price is not None and market_price > 0 and (stop_level is not None or profit_level is not None):
                try:
                    mp = float(market_price)
                except (TypeError, ValueError):
                    mp = None
                if mp is not None:
                    if stop_level is not None:
                        if (pos > 0 and mp <= float(stop_level)) or (pos < 0 and mp >= float(stop_level)):
                            return _trigger_exit(
                                "stop_loss" if stop_price is not None else "stop_loss_pct",
                                mode="spot",
                                calc=_spot_calc(
                                    move=move,
                                    threshold=-float(sl) if sl is not None else None,
                                    target=float(target_price) if target_price is not None else None,
                                    stop=float(stop_price) if stop_price is not None else None,
                                    pt=pt,
                                    sl=sl,
                                    stop_level=stop_level,
                                    profit_level=profit_level,
                                ),
                            )
                    if profit_level is not None:
                        if (pos > 0 and mp >= float(profit_level)) or (pos < 0 and mp <= float(profit_level)):
                            return _trigger_exit(
                                "profit_target" if target_price is not None else "profit_target_pct",
                                mode="spot",
                                calc=_spot_calc(
                                    move=move,
                                    threshold=float(pt) if pt is not None else None,
                                    target=float(target_price) if target_price is not None else None,
                                    stop=float(stop_price) if stop_price is not None else None,
                                    pt=pt,
                                    sl=sl,
                                    stop_level=stop_level,
                                    profit_level=profit_level,
                                ),
                            )

            exit_candidates: dict[str, bool] = {
                "ratsv_probe_cancel": bool(
                    self._spot_ratsv_probe_cancel_hit(
                        instance=instance,
                        pos=pos,
                        held_bars=int(held_bars),
                        entry_branch=entry_branch,
                        tr_ratio=ratsv_tr_ratio,
                        slope_med=ratsv_slope_med,
                    )
                ),
                "ratsv_adverse_release": bool(
                    self._spot_ratsv_adverse_release_hit(
                        instance=instance,
                        pos=pos,
                        held_bars=int(held_bars),
                        tr_ratio=ratsv_tr_ratio,
                        slope_med=ratsv_slope_med,
                        slope_vel=ratsv_slope_vel,
                    )
                ),
            }

            exit_time = parse_time_hhmm(instance.strategy.get("spot_exit_time_et"))
            if exit_time is not None and now_et.time() >= exit_time:
                exit_candidates["exit_time"] = True

            if bool(instance.strategy.get("spot_close_eod")) and (
                now_et.hour > 15 or now_et.hour == 15 and now_et.minute >= 55
            ):
                exit_candidates["close_eod"] = True

            exit_candidates["flip"] = bool(self._should_exit_on_flip(instance, snap, open_dir, open_items))

            lifecycle = decide_open_position_intent(
                strategy=instance.strategy,
                bar_ts=snap.bar_ts,
                bar_size=self._signal_bar_size(instance),
                open_dir=open_dir,
                current_qty=int(pos),
                exit_candidates=exit_candidates,
                signal_entry_dir=getattr(getattr(snap, "signal", None), "entry_dir", None),
                shock_atr_pct=float(snap.shock_atr_pct) if snap.shock_atr_pct is not None else None,
                shock_atr_vel_pct=(
                    float(getattr(snap, "shock_atr_vel_pct", 0.0))
                    if getattr(snap, "shock_atr_vel_pct", None) is not None
                    else None
                ),
                shock_atr_accel_pct=(
                    float(getattr(snap, "shock_atr_accel_pct", 0.0))
                    if getattr(snap, "shock_atr_accel_pct", None) is not None
                    else None
                ),
                tr_ratio=ratsv_tr_ratio,
                slope_med_pct=ratsv_slope_med,
                slope_vel_pct=ratsv_slope_vel,
                slope_med_slow_pct=(
                    float(getattr(snap, "ratsv_slow_slope_med_pct", 0.0))
                    if getattr(snap, "ratsv_slow_slope_med_pct", None) is not None
                    else None
                ),
                slope_vel_slow_pct=(
                    float(getattr(snap, "ratsv_slow_slope_vel_pct", 0.0))
                    if getattr(snap, "ratsv_slow_slope_vel_pct", None) is not None
                    else None
                ),
            )
            if lifecycle.intent == "exit":
                if lifecycle.fill_mode == "next_open":
                    scheduled = self._schedule_pending_exit_next_open(
                        instance=instance,
                        reason=str(lifecycle.reason or "flip"),
                        direction=open_dir,
                        signal_bar_ts=snap.bar_ts,
                        now_wall=self._wall_time(now_et),
                        mode="spot",
                        gate=gate,
                    )
                    if scheduled and lifecycle.queue_reentry_dir in ("up", "down"):
                        if instance.pending_entry_due_ts is None:
                            due_from_ts = self._spot_signal_close_ts(instance, snap.bar_ts)
                            due_ts = self._spot_next_open_due_ts(instance, snap.bar_ts)
                            instance.pending_entry_direction = str(lifecycle.queue_reentry_dir)
                            instance.pending_entry_signal_bar_ts = snap.bar_ts
                            instance.pending_entry_due_ts = due_ts
                            gate(
                                "PENDING_ENTRY_NEXT_OPEN",
                                {
                                    "direction": str(lifecycle.queue_reentry_dir),
                                    "reason": "controlled_flip",
                                    "next_open_due": due_ts.isoformat(),
                                    "next_open_due_from": due_from_ts.isoformat(),
                                    "signal_bar_ts": snap.bar_ts.isoformat(),
                                    "spot_lifecycle": lifecycle.as_payload(),
                                },
                            )
                    return bool(scheduled)
                return _trigger_exit(
                    str(lifecycle.reason or "flip"),
                    mode="spot",
                    calc=_spot_calc(),
                    lifecycle=lifecycle,
                )

        if instrument != "spot":
            if self._should_exit_on_dte(instance, open_items, now_et.date()):
                return _trigger_exit("dte", mode="options")

            entry_value, current_value = self._options_position_values(open_items)
            if entry_value is not None and current_value is not None:
                profit = float(entry_value) - float(current_value)
                try:
                    profit_target = float(instance.strategy.get("profit_target", 0.0) or 0.0)
                except (TypeError, ValueError):
                    profit_target = 0.0
                if profit_target > 0 and abs(entry_value) > 0:
                    if profit >= abs(entry_value) * profit_target:
                        return _trigger_exit("profit_target", mode="options")

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
                                return _trigger_exit("stop_loss_credit", mode="options")
                        elif loss >= abs(entry_value) * stop_loss:
                            return _trigger_exit("stop_loss_credit", mode="options")
                    else:
                        max_loss = self._options_max_loss_estimate(open_items, spot=float(snap.close))
                        if max_loss is None or max_loss <= 0:
                            max_loss = abs(entry_value)
                        if max_loss and loss >= float(max_loss) * stop_loss:
                            return _trigger_exit("stop_loss_max_loss", mode="options")

        if instrument != "spot" and self._should_exit_on_flip(instance, snap, open_dir, open_items):
            return _trigger_exit("flip", mode=instrument)
        return False

    async def _auto_maybe_resize_open_positions(
        self,
        *,
        instance: _BotInstance,
        snap,
        instrument: str,
        open_items: list,
        open_dir: str | None,
        gate,
    ) -> bool:
        if str(instrument or "").strip().lower() != "spot":
            return False
        if not open_items:
            return False

        policy = spot_policy_config_view(strategy=instance.strategy, filters=instance.filters)
        resize_mode = str(getattr(policy, "spot_resize_mode", "off") or "off").strip().lower()
        if resize_mode != "target":
            return False

        if instance.last_resize_bar_ts is not None and instance.last_resize_bar_ts == snap.bar_ts:
            return False

        self._queue_order(
            instance,
            intent="resize",
            direction=open_dir,
            signal_bar_ts=snap.bar_ts,
            trigger_reason="resize_target",
            trigger_mode="spot",
        )
        payload = {
            "direction": open_dir,
            "mode": "target",
            "cooldown_bars": int(getattr(policy, "spot_resize_cooldown_bars", 0) or 0),
        }
        payload.update(self._order_trigger_watch_payload(instance))
        gate("TRIGGER_RESIZE", payload)
        return True

    def _auto_try_queue_entry(self, *, instance: _BotInstance, snap, gate, now_et: datetime) -> bool:
        if instance.last_entry_bar_ts is not None and instance.last_entry_bar_ts == snap.bar_ts:
            gate("BLOCKED_ENTRY_SAME_BAR", {"bar_ts": snap.bar_ts.isoformat()})
            return False
        if instance.pending_exit_due_ts is not None or instance.pending_entry_due_ts is not None:
            return False

        instrument = self._strategy_instrument(instance.strategy)
        atr_ready = True
        if instrument == "spot":
            runtime_spec = spot_runtime_spec_view(strategy=instance.strategy, filters=instance.filters)
            exit_mode = str(runtime_spec.exit_mode)
            if exit_mode == "atr":
                atr = float(snap.atr or 0.0) if snap.atr is not None else 0.0
                if atr <= 0:
                    atr_ready = False

        direction = self._entry_direction_for_instance(instance, snap)
        allowed_directions = tuple(self._allowed_entry_directions(instance))
        decision = decide_flat_position_intent(
            strategy=instance.strategy,
            bar_ts=snap.bar_ts,
            entry_dir=direction,
            allowed_directions=allowed_directions,
            can_order_now=True,
            preflight_ok=True,
            filters_ok=True,
            entry_capacity=bool(self._entry_limit_ok(instance)),
            stale_signal=False,
            gap_signal=False,
            pending_exists=False,
            atr_ready=bool(atr_ready),
            next_open_allowed=True,
            shock_atr_pct=float(snap.shock_atr_pct) if getattr(snap, "shock_atr_pct", None) is not None else None,
            shock_atr_vel_pct=float(getattr(snap, "shock_atr_vel_pct", 0.0))
            if getattr(snap, "shock_atr_vel_pct", None) is not None
            else None,
            shock_atr_accel_pct=float(getattr(snap, "shock_atr_accel_pct", 0.0))
            if getattr(snap, "shock_atr_accel_pct", None) is not None
            else None,
            tr_ratio=float(getattr(snap, "ratsv_tr_ratio", 0.0))
            if getattr(snap, "ratsv_tr_ratio", None) is not None
            else None,
            slope_med_pct=float(getattr(snap, "ratsv_fast_slope_med_pct", 0.0))
            if getattr(snap, "ratsv_fast_slope_med_pct", None) is not None
            else None,
            slope_vel_pct=float(getattr(snap, "ratsv_fast_slope_vel_pct", 0.0))
            if getattr(snap, "ratsv_fast_slope_vel_pct", None) is not None
            else None,
            slope_med_slow_pct=float(getattr(snap, "ratsv_slow_slope_med_pct", 0.0))
            if getattr(snap, "ratsv_slow_slope_med_pct", None) is not None
            else None,
            slope_vel_slow_pct=float(getattr(snap, "ratsv_slow_slope_vel_pct", 0.0))
            if getattr(snap, "ratsv_slow_slope_vel_pct", None) is not None
            else None,
        )
        if decision.intent != "enter":
            payload = {"spot_lifecycle": decision.as_payload()}
            if decision.gate == "BLOCKED_ENTRY_LIMIT":
                payload["entries_today"] = int(instance.entries_today)
            if decision.gate == "BLOCKED_ATR_NOT_READY":
                payload["atr"] = float(snap.atr or 0.0) if snap.atr is not None else 0.0
            if decision.gate == "WAITING_SIGNAL":
                payload["bar_ts"] = snap.bar_ts.isoformat()
            if decision.gate == "BLOCKED_DIRECTION":
                payload["direction"] = direction
            gate(decision.gate, payload)
            return False

        if instrument == "spot":
            if str(decision.fill_mode) == "next_open":
                return self._schedule_pending_entry_next_open(
                    instance=instance,
                    direction=str(decision.direction),
                    signal_bar_ts=snap.bar_ts,
                    now_wall=self._wall_time(now_et),
                    gate=gate,
                )

        self._queue_order(
            instance,
            intent="enter",
            direction=str(decision.direction),
            signal_bar_ts=snap.bar_ts,
            trigger_reason="entry",
            trigger_mode=instrument,
        )
        payload = {"direction": str(decision.direction), "spot_lifecycle": decision.as_payload()}
        payload.update(self._order_trigger_watch_payload(instance))
        gate("TRIGGER_ENTRY", payload)
        return True

    def _queue_order(
        self,
        instance: _BotInstance,
        *,
        intent: str,
        direction: str | None,
        signal_bar_ts: datetime | None,
        trigger_reason: str | None = None,
        trigger_mode: str | None = None,
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
        if intent == "exit":
            action = "Exiting"
        elif intent == "resize":
            action = "Resizing"
        else:
            action = "Creating order"
        dir_note = f" ({direction})" if direction else ""
        self._status = f"{action} for instance {instance.instance_id}{dir_note}..."
        self._render_status()
        self._mark_order_trigger_watch(
            instance=instance,
            intent=str(intent),
            direction=direction,
            signal_bar_ts=signal_bar_ts,
            now_et=_now_et(),
            reason=trigger_reason,
            mode=trigger_mode,
        )

        async def _create_order_task() -> None:
            try:
                await self._create_order_for_instance(
                    instance,
                    intent=str(intent),
                    direction=direction,
                    signal_bar_ts=signal_bar_ts,
                )
            except Exception as exc:
                instance.order_trigger_last_error = str(exc)
                if not str(instance.order_trigger_retry_reason or "").strip():
                    instance.order_trigger_retry_reason = "order_build_exception"
                self._journal_write(
                    event="ORDER_BUILD_FAILED",
                    instance=instance,
                    order=None,
                    reason=str(intent),
                    data={
                        "error": str(exc),
                        "exc": str(exc),
                        "direction": direction,
                        "signal_bar_ts": signal_bar_ts.isoformat() if signal_bar_ts else None,
                        "retry_reason": "order_build_exception",
                        **self._order_trigger_watch_payload(instance),
                    },
                )
                self._status = f"Order build error: {exc}"
                self._render_status()

        self._order_task = loop.create_task(_create_order_task())

    def _entry_weekday_for_ts(self, instance: _BotInstance, ts: datetime) -> int:
        ts_et = self._as_et_aware(ts)
        weekday = int(ts_et.weekday())
        if self._strategy_instrument(instance.strategy) != "spot":
            return weekday
        if bool(self._signal_use_rth(instance)):
            return weekday
        sec_type = str(instance.strategy.get("spot_sec_type") or "STK").strip().upper()
        if sec_type != "STK":
            return weekday
        if weekday == 6 and ts_et.time() >= time(20, 0):
            # Sunday evening overnight session belongs to Monday's entry day.
            return 0
        return weekday

    def _can_order_now(self, instance: _BotInstance, *, now_et: datetime | None = None) -> bool:
        entry_days = instance.strategy.get("entry_days", [])
        if entry_days:
            allowed = {_weekday_num(day) for day in entry_days}
        else:
            allowed = {0, 1, 2, 3, 4, 5, 6}
        now = now_et if now_et is not None else _now_et()
        if self._entry_weekday_for_ts(instance, now) not in allowed:
            return False
        return True
