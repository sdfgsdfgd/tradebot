"""Shared spot lifecycle decision kernel.

This module centralizes spot lifecycle decisions used by both:
- live UI runtime (`tradebot/ui/bot_signal_runtime.py`)
- backtest runtime (`tradebot/backtest/engine.py`)

Kernel outputs one typed decision:
- `hold`
- `enter`
- `exit`
- `resize`

with a normalized gate/result payload for consistent diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time, timedelta

from ..signals import ema_slope_pct, ema_spread_pct, ema_state_direction, flip_exit_mode, parse_bar_size
from ..time_utils import NaiveTsModeInput
from ..time_utils import normalize_naive_ts_mode as _normalize_naive_ts_mode
from ..time_utils import to_et as _to_et
from ..time_utils import to_utc_naive as _to_utc_naive
from .fill_modes import (
    SPOT_FILL_MODE_CLOSE,
    SPOT_FILL_MODE_NEXT_BAR,
    SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    normalize_spot_fill_mode,
    spot_fill_mode_is_deferred,
    spot_fill_mode_is_next_tradable,
)
from .graph import SpotPolicyGraph, canonical_exit_reason as graph_canonical_exit_reason
from .graph import pick_exit_reason as graph_pick_exit_reason
from .policy import SpotIntentDecision, SpotPolicy


@dataclass(frozen=True)
class SpotLifecycleDecision:
    intent: str
    reason: str
    gate: str
    direction: str | None = None
    fill_mode: str = SPOT_FILL_MODE_CLOSE
    blocked: bool = False
    pending_clear_entry: bool = False
    pending_clear_exit: bool = False
    queue_reentry_dir: str | None = None
    spot_intent: SpotIntentDecision | None = None
    spot_decision: dict[str, object] | None = None
    trace: dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        if self.spot_intent is not None:
            payload["spot_intent"] = self.spot_intent.as_payload()
        return payload


@dataclass(frozen=True)
class SpotDeferredEntryPlan:
    fill_mode: str
    deferred: bool
    due_ts: datetime | None
    allowed: bool
    reason: str


def _get(source: Mapping[str, object] | object | None, key: str, default: object = None) -> object:
    return SpotPolicy._get(source, key, default)  # noqa: SLF001 - shared parser reuse by design


def _normalize_fill_mode(raw: object, *, default: str = SPOT_FILL_MODE_CLOSE) -> str:
    return normalize_spot_fill_mode(raw, default=str(default))


_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)
_EXT_OPEN = time(4, 0)
_EXT_CLOSE = time(20, 0)
_OVERNIGHT_OPEN = time(20, 0)
_OVERNIGHT_GAP_START = time(3, 50)
_OVERNIGHT_GAP_END = time(4, 0)


def normalize_next_open_session_mode(raw: object | None, *, default: str = "auto") -> str:
    mode = str(raw or default).strip().lower()
    aliases = {
        "full24": "tradable_24x5",
        "tradable": "tradable_24x5",
        "overnight_plus_extended": "tradable_24x5",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("auto", "rth", "extended", "always", "tradable_24x5"):
        fallback = str(default or "auto").strip().lower()
        return fallback if fallback in ("auto", "rth", "extended", "always", "tradable_24x5") else "auto"
    return mode


def signal_bar_close_ts(
    *,
    signal_bar_ts: datetime,
    signal_bar_size: str,
) -> datetime:
    signal_def = parse_bar_size(str(signal_bar_size or ""))
    if signal_def is None or signal_def.duration.total_seconds() <= 0:
        return signal_bar_ts
    return signal_bar_ts + signal_def.duration


def _spot_exec_span(exec_bar_size: str) -> timedelta:
    parsed = parse_bar_size(str(exec_bar_size or ""))
    if parsed is None or parsed.duration.total_seconds() <= 0:
        return timedelta(minutes=5)
    return parsed.duration


def _bar_floor(ts: datetime, span: timedelta) -> datetime:
    sec = float(span.total_seconds())
    if sec <= 0:
        return ts
    day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = max(0.0, float((ts - day_start).total_seconds()))
    slot = int(elapsed // sec)
    return day_start + timedelta(seconds=(slot * sec))


def _bar_ceil(ts: datetime, span: timedelta) -> datetime:
    floor_ts = _bar_floor(ts, span)
    if floor_ts >= ts:
        return floor_ts
    return floor_ts + span


def _spot_next_weekday_open(ts: datetime, open_time: time) -> datetime:
    candidate = datetime.combine(ts.date(), open_time)
    if candidate <= ts:
        candidate = candidate + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def _spot_session_contains(ts: datetime, *, open_time: time, close_time: time) -> bool:
    if ts.weekday() >= 5:
        return False
    current = ts.time()
    return open_time <= current < close_time


def _spot_stk_tradable_24x5_contains(ts: datetime) -> bool:
    weekday = ts.weekday()
    current = ts.time()
    if weekday == 5:
        return False
    if _OVERNIGHT_GAP_START <= current < _OVERNIGHT_GAP_END:
        return False
    if weekday == 6:
        return current >= _OVERNIGHT_OPEN
    if weekday == 4 and current >= _OVERNIGHT_OPEN:
        return False
    return True


def _spot_next_stk_tradable_24x5_start(ts: datetime) -> datetime:
    candidate = ts
    while True:
        weekday = candidate.weekday()
        current = candidate.time()
        if weekday == 5:
            sunday = candidate.date() + timedelta(days=1)
            candidate = datetime.combine(sunday, _OVERNIGHT_OPEN)
            continue
        if weekday == 6:
            if current < _OVERNIGHT_OPEN:
                return datetime.combine(candidate.date(), _OVERNIGHT_OPEN)
            return candidate
        if _OVERNIGHT_GAP_START <= current < _OVERNIGHT_GAP_END:
            return datetime.combine(candidate.date(), _OVERNIGHT_GAP_END)
        if weekday == 4 and current >= _OVERNIGHT_OPEN:
            sunday = candidate.date() + timedelta(days=2)
            return datetime.combine(sunday, _OVERNIGHT_OPEN)
        return candidate


def _spot_align_exec_open_stk_tradable_24x5(*, ts: datetime, span: timedelta) -> datetime:
    candidate = ts
    while True:
        if not _spot_stk_tradable_24x5_contains(candidate):
            candidate = _spot_next_stk_tradable_24x5_start(candidate)
            continue
        aligned = _bar_ceil(candidate, span)
        if _spot_stk_tradable_24x5_contains(aligned):
            return aligned
        candidate = _spot_next_stk_tradable_24x5_start(aligned)


def _spot_session_window(
    *,
    mode: str,
    sec_type: str,
    signal_use_rth: bool,
) -> tuple[time, time] | None:
    if mode == "always":
        return None
    if mode == "rth":
        return (_RTH_OPEN, _RTH_CLOSE)
    if mode == "extended":
        return (_EXT_OPEN, _EXT_CLOSE)
    if str(sec_type).strip().upper() != "STK":
        return None
    return (_RTH_OPEN, _RTH_CLOSE) if bool(signal_use_rth) else (_EXT_OPEN, _EXT_CLOSE)


def _spot_align_exec_open_et(
    *,
    ts: datetime,
    span: timedelta,
    mode: str,
    sec_type: str,
    signal_use_rth: bool,
) -> datetime:
    mode_clean = normalize_next_open_session_mode(mode, default="auto")
    if mode_clean == "auto" and str(sec_type).strip().upper() == "STK" and not bool(signal_use_rth):
        mode_clean = "tradable_24x5"
    if mode_clean == "tradable_24x5":
        if str(sec_type).strip().upper() == "STK":
            return _spot_align_exec_open_stk_tradable_24x5(ts=ts, span=span)
        return ts
    session = _spot_session_window(
        mode=mode_clean,
        sec_type=str(sec_type),
        signal_use_rth=bool(signal_use_rth),
    )
    if session is None:
        return ts
    open_time, close_time = session
    candidate = ts
    while True:
        if candidate.weekday() >= 5:
            candidate = _spot_next_weekday_open(candidate, open_time)
            continue
        day_open = datetime.combine(candidate.date(), open_time)
        day_close = datetime.combine(candidate.date(), close_time)
        if candidate < day_open:
            candidate = day_open
        elif candidate >= day_close:
            candidate = _spot_next_weekday_open(candidate, open_time)
            continue

        aligned = _bar_ceil(candidate, span)
        if aligned >= day_close:
            candidate = _spot_next_weekday_open(candidate, open_time)
            continue
        if not _spot_session_contains(aligned, open_time=open_time, close_time=close_time):
            candidate = _spot_next_weekday_open(candidate, open_time)
            continue
        return aligned


def _to_et_naive(ts: datetime, *, naive_ts_mode: NaiveTsModeInput) -> datetime:
    return _to_et(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc").replace(tzinfo=None)


def _from_et_naive(ts: datetime, *, naive_ts_mode: NaiveTsModeInput) -> datetime:
    mode = _normalize_naive_ts_mode(naive_ts_mode, default="et")
    if str(mode.value).strip().lower() == "utc":
        return _to_utc_naive(ts, naive_ts_mode="et", default_naive_ts_mode="utc")
    return ts


def next_open_due_ts(
    *,
    signal_close_ts: datetime,
    exec_bar_size: str,
    strategy: Mapping[str, object] | object | None = None,
    naive_ts_mode: NaiveTsModeInput = None,
) -> datetime:
    span = _spot_exec_span(exec_bar_size)
    due_from = _bar_ceil(signal_close_ts, span)
    due_from_et = _to_et_naive(due_from, naive_ts_mode=naive_ts_mode)
    mode = normalize_next_open_session_mode(_get(strategy, "spot_next_open_session", "auto"), default="auto")
    sec_type = str(_get(strategy, "spot_sec_type", "STK") or "STK").strip().upper()
    signal_use_rth_raw = _get(strategy, "signal_use_rth", True)
    if isinstance(signal_use_rth_raw, str):
        signal_use_rth = signal_use_rth_raw.strip().lower() not in ("0", "false", "off", "no")
    else:
        signal_use_rth = bool(signal_use_rth_raw)
    aligned_et = _spot_align_exec_open_et(
        ts=due_from_et,
        span=span,
        mode=mode,
        sec_type=sec_type,
        signal_use_rth=signal_use_rth,
    )
    return _from_et_naive(aligned_et, naive_ts_mode=naive_ts_mode)


def fill_due_ts(
    *,
    fill_mode: str,
    signal_close_ts: datetime,
    exec_bar_size: str,
    strategy: Mapping[str, object] | object | None = None,
    naive_ts_mode: NaiveTsModeInput = None,
) -> datetime | None:
    mode = normalize_spot_fill_mode(fill_mode, default=SPOT_FILL_MODE_CLOSE)
    if mode == SPOT_FILL_MODE_CLOSE:
        return None
    if mode == SPOT_FILL_MODE_NEXT_BAR:
        return _bar_ceil(signal_close_ts, _spot_exec_span(exec_bar_size))
    return next_open_due_ts(
        signal_close_ts=signal_close_ts,
        exec_bar_size=exec_bar_size,
        strategy=strategy,
        naive_ts_mode=naive_ts_mode,
    )


def deferred_entry_plan(
    *,
    fill_mode: str,
    signal_ts: datetime,
    signal_close_ts: datetime,
    exec_bar_size: str,
    strategy: Mapping[str, object] | object | None = None,
    riskoff_today: bool,
    riskoff_end_hour: int | None,
    exit_mode: str,
    atr_value: float | None,
    naive_ts_mode: NaiveTsModeInput = None,
    due_ts: datetime | None = None,
) -> SpotDeferredEntryPlan:
    mode = normalize_spot_fill_mode(fill_mode, default=SPOT_FILL_MODE_CLOSE)
    if not spot_fill_mode_is_deferred(mode):
        return SpotDeferredEntryPlan(
            fill_mode=mode,
            deferred=False,
            due_ts=None,
            allowed=True,
            reason="immediate",
        )

    resolved_due = due_ts
    if resolved_due is None:
        resolved_due = fill_due_ts(
            fill_mode=mode,
            signal_close_ts=signal_close_ts,
            exec_bar_size=exec_bar_size,
            strategy=strategy,
            naive_ts_mode=naive_ts_mode,
        )
    if resolved_due is None:
        return SpotDeferredEntryPlan(
            fill_mode=mode,
            deferred=True,
            due_ts=None,
            allowed=False,
            reason="deferred_due_missing",
        )

    if spot_fill_mode_is_next_tradable(mode):
        allowed = next_open_entry_allowed(
            signal_ts=signal_ts,
            next_ts=resolved_due,
            riskoff_today=bool(riskoff_today),
            riskoff_end_hour=riskoff_end_hour,
            exit_mode=str(exit_mode),
            atr_value=float(atr_value) if atr_value is not None else None,
        )
        return SpotDeferredEntryPlan(
            fill_mode=mode,
            deferred=True,
            due_ts=resolved_due,
            allowed=bool(allowed),
            reason="ok" if bool(allowed) else "next_open_not_allowed",
        )

    return SpotDeferredEntryPlan(
        fill_mode=mode,
        deferred=True,
        due_ts=resolved_due,
        allowed=True,
        reason="ok",
    )


def _bars_elapsed(start_ts: datetime | None, end_ts: datetime, *, bar_size: str) -> int:
    if start_ts is None:
        return 0
    parsed = parse_bar_size(str(bar_size or ""))
    if parsed is None or parsed.duration.total_seconds() <= 0:
        return 0
    elapsed = (end_ts - start_ts).total_seconds()
    if elapsed <= 0:
        return 0
    return int(elapsed // parsed.duration.total_seconds())


def canonical_exit_reason(reason: str | None) -> str:
    return graph_canonical_exit_reason(reason)


def pick_exit_reason(
    exit_candidates: Mapping[str, bool] | None,
    *,
    priority: Sequence[str] | None = None,
) -> str | None:
    return graph_pick_exit_reason(exit_candidates, priority=priority)


def entry_capacity_ok(
    *,
    open_count: int,
    max_entries_per_day: int,
    entries_today: int,
    weekday: int,
    entry_days: Sequence[int],
) -> bool:
    open_slots_ok = int(open_count) < 1
    entries_ok = int(max_entries_per_day) == 0 or int(entries_today) < int(max_entries_per_day)
    return bool(open_slots_ok and entries_ok and int(weekday) in {int(d) for d in entry_days})


def next_open_entry_allowed(
    *,
    signal_ts: datetime,
    next_ts: datetime,
    riskoff_today: bool,
    riskoff_end_hour: int | None,
    exit_mode: str,
    atr_value: float | None,
) -> bool:
    if bool(riskoff_today):
        if SpotPolicy._trade_date(next_ts) != SpotPolicy._trade_date(signal_ts):  # noqa: SLF001
            return False
        if riskoff_end_hour is not None and SpotPolicy._trade_hour_et(next_ts) >= int(riskoff_end_hour):  # noqa: SLF001
            return False
    if str(exit_mode).strip().lower() == "atr":
        if atr_value is None or float(atr_value) <= 0.0:
            return False
    return True


def permission_gate_status(
    filters: Mapping[str, object] | object | None,
    *,
    close: float,
    signal: object | None,
    entry_dir: str | None,
) -> tuple[bool, bool]:
    spread_min = _get(filters, "ema_spread_min_pct")
    spread_min_down = _get(filters, "ema_spread_min_pct_down")
    if entry_dir == "down" and spread_min_down is not None:
        spread_min = spread_min_down
    slope_min = _get(filters, "ema_slope_min_pct")

    slope_signed_up = _get(filters, "ema_slope_signed_min_pct_up") if entry_dir == "up" else None
    slope_signed_down = _get(filters, "ema_slope_signed_min_pct_down") if entry_dir == "down" else None

    active = (
        spread_min is not None
        or slope_min is not None
        or slope_signed_up is not None
        or slope_signed_down is not None
    )
    if not bool(active):
        return False, True

    if (
        signal is None
        or not bool(getattr(signal, "ema_ready", False))
        or getattr(signal, "ema_fast", None) is None
        or getattr(signal, "ema_slow", None) is None
    ):
        return True, False

    if spread_min is not None:
        try:
            spread_min_f = float(spread_min)
        except (TypeError, ValueError):
            spread_min_f = None
        if spread_min_f is not None:
            spread = ema_spread_pct(
                float(getattr(signal, "ema_fast")),
                float(getattr(signal, "ema_slow")),
                float(close),
            )
            if spread < spread_min_f:
                return True, False

    if slope_min is not None:
        try:
            slope_min_f = float(slope_min)
        except (TypeError, ValueError):
            slope_min_f = None
        if slope_min_f is not None:
            if getattr(signal, "prev_ema_fast", None) is None:
                return True, False
            slope = ema_slope_pct(
                float(getattr(signal, "ema_fast")),
                float(getattr(signal, "prev_ema_fast")),
                float(close),
            )
            if slope < slope_min_f:
                return True, False

    if slope_signed_up is not None:
        try:
            slope_signed_min = float(slope_signed_up)
        except (TypeError, ValueError):
            slope_signed_min = None
        if slope_signed_min is not None and slope_signed_min > 0:
            if getattr(signal, "prev_ema_fast", None) is None:
                return True, False
            denom = max(float(close), 1e-9)
            signed = (float(getattr(signal, "ema_fast")) - float(getattr(signal, "prev_ema_fast"))) / denom * 100.0
            if signed < float(slope_signed_min):
                return True, False

    if slope_signed_down is not None:
        try:
            slope_signed_min = float(slope_signed_down)
        except (TypeError, ValueError):
            slope_signed_min = None
        if slope_signed_min is not None and slope_signed_min > 0:
            if getattr(signal, "prev_ema_fast", None) is None:
                return True, False
            denom = max(float(close), 1e-9)
            signed = (float(getattr(signal, "ema_fast")) - float(getattr(signal, "prev_ema_fast"))) / denom * 100.0
            if signed > -float(slope_signed_min):
                return True, False

    return True, True


_SIGNAL_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "ema_fast",
    "ema_slow",
    "prev_ema_fast",
    "prev_ema_slow",
    "ema_ready",
    "cross_up",
    "cross_down",
    "state",
    "entry_dir",
    "regime_dir",
    "regime_ready",
)


def _clone_signal_snapshot(
    signal: object,
    *,
    entry_dir: str | None,
    regime_dir: str | None,
    regime_ready: bool,
) -> object:
    if is_dataclass(signal):
        try:
            return replace(
                signal,
                entry_dir=entry_dir,
                regime_dir=regime_dir,
                regime_ready=bool(regime_ready),
            )
        except TypeError:
            pass

    payload = {name: getattr(signal, name, None) for name in _SIGNAL_SNAPSHOT_FIELDS}
    payload["entry_dir"] = entry_dir
    payload["regime_dir"] = regime_dir
    payload["regime_ready"] = bool(regime_ready)
    try:
        return type(signal)(**payload)
    except Exception:
        return signal


def apply_regime_gate(
    signal: object | None,
    *,
    regime_dir: str | None,
    regime_ready: bool,
) -> object | None:
    if signal is None:
        return None
    cleaned_regime_dir = str(regime_dir) if regime_dir in ("up", "down") else None
    entry_dir = getattr(signal, "entry_dir", None)
    if entry_dir is not None:
        if not bool(regime_ready):
            entry_dir = None
        elif cleaned_regime_dir is None or cleaned_regime_dir != entry_dir:
            entry_dir = None
    return _clone_signal_snapshot(
        signal,
        entry_dir=entry_dir,
        regime_dir=cleaned_regime_dir,
        regime_ready=bool(regime_ready),
    )


def flip_exit_hit(
    *,
    exit_on_signal_flip: bool,
    open_dir: str | None,
    signal: object | None,
    flip_exit_mode_raw: str | None,
    ema_entry_mode_raw: str | None,
) -> bool:
    if not bool(exit_on_signal_flip):
        return False
    if open_dir not in ("up", "down"):
        return False
    if (
        signal is None
        or not bool(getattr(signal, "ema_ready", False))
        or getattr(signal, "ema_fast", None) is None
        or getattr(signal, "ema_slow", None) is None
    ):
        return False

    mode = flip_exit_mode(flip_exit_mode_raw, ema_entry_mode_raw)
    if mode == "cross":
        if open_dir == "up":
            return bool(getattr(signal, "cross_down", False))
        return bool(getattr(signal, "cross_up", False))

    state = getattr(signal, "state", None) or ema_state_direction(
        getattr(signal, "ema_fast", None),
        getattr(signal, "ema_slow", None),
    )
    if state is None:
        return False
    if open_dir == "up":
        return state == "down"
    return state == "up"


def _normalize_shock_gate_mode(filters: Mapping[str, object] | object | None) -> str:
    raw = _get(filters, "shock_gate_mode")
    if raw is None:
        raw = _get(filters, "shock_mode")
    if isinstance(raw, bool):
        raw = "block" if raw else "off"
    mode = str(raw or "off").strip().lower()
    if mode in ("", "0", "false", "none", "null"):
        mode = "off"
    if mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        mode = "off"
    return mode


def flip_exit_gate_blocked(
    *,
    gate_mode_raw: str | None,
    filters: Mapping[str, object] | object | None,
    close: float,
    signal: object | None,
    trade_dir: str | None,
) -> bool:
    gate_mode = str(gate_mode_raw or "off").strip().lower()
    if gate_mode not in (
        "off",
        "regime",
        "permission",
        "regime_or_permission",
        "regime_and_permission",
    ):
        gate_mode = "off"
    if gate_mode == "off" or signal is None or trade_dir not in ("up", "down"):
        return False

    bias_ok = bool(getattr(signal, "regime_ready", False)) and str(getattr(signal, "regime_dir", "")) == str(trade_dir)
    perm_active, perm_pass = permission_gate_status(
        filters,
        close=float(close),
        signal=signal,
        entry_dir=trade_dir,
    )
    perm_ok = bool(perm_active and perm_pass)

    if gate_mode == "regime":
        return bias_ok
    if gate_mode == "permission":
        return perm_ok
    if gate_mode == "regime_or_permission":
        return bias_ok or perm_ok
    if gate_mode == "regime_and_permission":
        return bias_ok and perm_ok
    return False


def _signal_filter_rv_ok(
    filters: Mapping[str, object] | object | None,
    *,
    rv: float | None,
) -> bool:
    rv_min = _get(filters, "rv_min")
    rv_max = _get(filters, "rv_max")
    if rv_min is None and rv_max is None:
        return True
    if rv is None:
        return False
    try:
        rv_min_f = float(rv_min) if rv_min is not None else None
    except (TypeError, ValueError):
        rv_min_f = None
    try:
        rv_max_f = float(rv_max) if rv_max is not None else None
    except (TypeError, ValueError):
        rv_max_f = None
    if rv_min_f is not None and float(rv) < rv_min_f:
        return False
    if rv_max_f is not None and float(rv) > rv_max_f:
        return False
    return True


def _hour_window_ok(hour: int, *, start_raw: object, end_raw: object) -> bool:
    try:
        start = int(start_raw)
        end = int(end_raw)
    except (TypeError, ValueError):
        return True
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _signal_filter_time_ok(
    filters: Mapping[str, object] | object | None,
    *,
    bar_ts: datetime,
) -> bool:
    entry_start_hour_et = _get(filters, "entry_start_hour_et")
    entry_end_hour_et = _get(filters, "entry_end_hour_et")
    if entry_start_hour_et is not None and entry_end_hour_et is not None:
        return _hour_window_ok(
            SpotPolicy._trade_hour_et(bar_ts),  # noqa: SLF001
            start_raw=entry_start_hour_et,
            end_raw=entry_end_hour_et,
        )

    entry_start_hour = _get(filters, "entry_start_hour")
    entry_end_hour = _get(filters, "entry_end_hour")
    if entry_start_hour is not None and entry_end_hour is not None:
        return _hour_window_ok(
            int(bar_ts.hour),
            start_raw=entry_start_hour,
            end_raw=entry_end_hour,
        )
    return True


def _signal_filter_skip_first_ok(
    filters: Mapping[str, object] | object | None,
    *,
    bars_in_day: int,
) -> bool:
    skip_first = _get(filters, "skip_first_bars")
    try:
        skip_first_n = int(skip_first or 0)
    except (TypeError, ValueError):
        skip_first_n = 0
    return not (skip_first_n > 0 and int(bars_in_day) <= skip_first_n)


def _signal_filter_shock_gate_ok(
    filters: Mapping[str, object] | object | None,
    *,
    shock: bool | None,
    shock_dir: str | None,
    signal: object | None,
) -> bool:
    shock_mode = _normalize_shock_gate_mode(filters)
    if shock_mode == "block":
        if shock is None:
            return False
        return not bool(shock)
    if shock_mode in ("block_longs", "block_shorts"):
        if shock is None:
            return False
        if not bool(shock):
            return True
        entry_dir = getattr(signal, "entry_dir", None) if signal is not None else None
        if shock_mode == "block_longs" and entry_dir == "up":
            return False
        if shock_mode == "block_shorts" and entry_dir == "down":
            return False
        return True
    if shock_mode == "surf":
        if shock is None:
            return False
        if not bool(shock):
            return True
        entry_dir = getattr(signal, "entry_dir", None) if signal is not None else None
        cleaned = str(shock_dir) if shock_dir in ("up", "down") else None
        if cleaned is None or entry_dir not in ("up", "down"):
            return False
        return entry_dir == cleaned
    return True


def _signal_filter_permission_ok(
    filters: Mapping[str, object] | object | None,
    *,
    close: float,
    signal: object | None,
) -> bool:
    entry_dir = getattr(signal, "entry_dir", None) if signal is not None else None
    _, perm_ok = permission_gate_status(filters, close=float(close), signal=signal, entry_dir=entry_dir)
    return bool(perm_ok)


def _signal_filter_volume_ok(
    filters: Mapping[str, object] | object | None,
    *,
    volume: float | None,
    volume_ema: float | None,
    volume_ema_ready: bool,
) -> bool:
    volume_ratio_min = _get(filters, "volume_ratio_min")
    if volume_ratio_min is None:
        return True
    try:
        ratio_min = float(volume_ratio_min)
    except (TypeError, ValueError):
        ratio_min = None
    if ratio_min is None:
        return True
    if not bool(volume_ema_ready):
        return False
    if volume is None or volume_ema is None:
        return False
    denom = float(volume_ema)
    if denom <= 0:
        return False
    ratio = float(volume) / denom
    return ratio >= ratio_min


@dataclass(frozen=True)
class _SignalFilterContext:
    bar_ts: datetime
    bars_in_day: int
    close: float
    volume: float | None
    volume_ema: float | None
    volume_ema_ready: bool
    rv: float | None
    signal: object | None
    cooldown_ok: bool
    shock: bool | None
    shock_dir: str | None


def _signal_filter_cooldown_ok(
    _filters: Mapping[str, object] | object | None,
    *,
    ctx: _SignalFilterContext,
) -> bool:
    return bool(ctx.cooldown_ok)


_SIGNAL_FILTER_REGISTRY: tuple[tuple[str, Callable[[Mapping[str, object] | object | None, _SignalFilterContext], bool]], ...] = (
    ("rv", lambda filters, ctx: _signal_filter_rv_ok(filters, rv=ctx.rv)),
    ("time", lambda filters, ctx: _signal_filter_time_ok(filters, bar_ts=ctx.bar_ts)),
    ("skip_first", lambda filters, ctx: _signal_filter_skip_first_ok(filters, bars_in_day=ctx.bars_in_day)),
    ("cooldown", lambda filters, ctx: _signal_filter_cooldown_ok(filters, ctx=ctx)),
    (
        "shock_gate",
        lambda filters, ctx: _signal_filter_shock_gate_ok(
            filters,
            shock=ctx.shock,
            shock_dir=ctx.shock_dir,
            signal=ctx.signal,
        ),
    ),
    ("permission", lambda filters, ctx: _signal_filter_permission_ok(filters, close=float(ctx.close), signal=ctx.signal)),
    (
        "volume",
        lambda filters, ctx: _signal_filter_volume_ok(
            filters,
            volume=ctx.volume,
            volume_ema=ctx.volume_ema,
            volume_ema_ready=ctx.volume_ema_ready,
        ),
    ),
)


def signal_filter_checks(
    filters: Mapping[str, object] | object | None,
    *,
    bar_ts: datetime,
    bars_in_day: int,
    close: float,
    volume: float | None = None,
    volume_ema: float | None = None,
    volume_ema_ready: bool = True,
    rv: float | None = None,
    signal: object | None = None,
    cooldown_ok: bool = True,
    shock: bool | None = None,
    shock_dir: str | None = None,
) -> dict[str, bool]:
    if filters is None:
        return {name: True for name, _predicate in _SIGNAL_FILTER_REGISTRY}
    ctx = _SignalFilterContext(
        bar_ts=bar_ts,
        bars_in_day=int(bars_in_day),
        close=float(close),
        volume=volume,
        volume_ema=volume_ema,
        volume_ema_ready=bool(volume_ema_ready),
        rv=rv,
        signal=signal,
        cooldown_ok=bool(cooldown_ok),
        shock=shock,
        shock_dir=shock_dir,
    )
    checks: dict[str, bool] = {}
    for name, predicate in _SIGNAL_FILTER_REGISTRY:
        checks[str(name)] = bool(predicate(filters, ctx))
    return checks


def signal_filters_ok(
    filters: Mapping[str, object] | object | None,
    *,
    bar_ts: datetime,
    bars_in_day: int,
    close: float,
    volume: float | None = None,
    volume_ema: float | None = None,
    volume_ema_ready: bool = True,
    rv: float | None = None,
    signal: object | None = None,
    cooldown_ok: bool = True,
    shock: bool | None = None,
    shock_dir: str | None = None,
) -> bool:
    checks = signal_filter_checks(
        filters,
        bar_ts=bar_ts,
        bars_in_day=int(bars_in_day),
        close=float(close),
        volume=float(volume) if volume is not None else None,
        volume_ema=float(volume_ema) if volume_ema is not None else None,
        volume_ema_ready=bool(volume_ema_ready),
        rv=float(rv) if rv is not None else None,
        signal=signal,
        cooldown_ok=bool(cooldown_ok),
        shock=shock,
        shock_dir=shock_dir,
    )
    return all(bool(v) for v in checks.values())


def adaptive_resize_target_qty(
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    base_target_qty: int,
    shock_atr_pct: float | None = None,
    shock_atr_vel_pct: float | None = None,
    shock_atr_accel_pct: float | None = None,
    tr_ratio: float | None = None,
    slope_med_pct: float | None = None,
    slope_vel_pct: float | None = None,
    slope_med_slow_pct: float | None = None,
    slope_vel_slow_pct: float | None = None,
) -> tuple[int, dict[str, object]]:
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    out = graph.resolve_resize_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(base_target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
    )
    return int(out.target_qty), dict(out.trace)


def decide_pending_next_open(
    *,
    now_ts: datetime,
    has_open: bool,
    open_dir: str | None,
    pending_entry_dir: str | None,
    pending_entry_set_date: date | None,
    pending_entry_due_ts: datetime | None,
    pending_exit_reason: str | None,
    pending_exit_due_ts: datetime | None,
    risk_overlay_enabled: bool,
    riskoff_today: bool,
    riskpanic_today: bool,
    riskpop_today: bool,
    riskoff_mode: str,
    shock_dir_now: str | None,
    riskoff_end_hour: int | None,
    pending_entry_fill_mode: str = SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    pending_exit_fill_mode: str = SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    naive_ts_mode: NaiveTsModeInput = None,
) -> SpotLifecycleDecision:
    entry_fill_mode = normalize_spot_fill_mode(pending_entry_fill_mode, default=SPOT_FILL_MODE_NEXT_TRADABLE_BAR)
    exit_fill_mode = normalize_spot_fill_mode(pending_exit_fill_mode, default=SPOT_FILL_MODE_NEXT_TRADABLE_BAR)

    if pending_exit_due_ts is not None and now_ts >= pending_exit_due_ts:
        if bool(has_open):
            reason = canonical_exit_reason(pending_exit_reason or "flip")
            return SpotLifecycleDecision(
                intent="exit",
                reason=reason or "flip",
                gate="TRIGGER_EXIT",
                direction=str(open_dir) if open_dir in ("up", "down") else None,
                fill_mode=exit_fill_mode,
                pending_clear_exit=True,
                trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
            )
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_exit_empty",
            gate="CLEAR_PENDING_EXIT",
            pending_clear_exit=True,
            trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
        )

    if pending_entry_dir in ("up", "down") and pending_entry_due_ts is not None:
        should_cancel = SpotPolicy.pending_entry_should_cancel(
            pending_dir=str(pending_entry_dir),
            pending_set_date=pending_entry_set_date,
            exec_ts=now_ts,
            risk_overlay_enabled=bool(risk_overlay_enabled),
            riskoff_today=bool(riskoff_today),
            riskpanic_today=bool(riskpanic_today),
            riskpop_today=bool(riskpop_today),
            riskoff_mode=str(riskoff_mode),
            shock_dir_now=shock_dir_now if shock_dir_now in ("up", "down") else None,
            riskoff_end_hour=riskoff_end_hour,
            naive_ts_mode=naive_ts_mode,
        )
        if bool(should_cancel):
            return SpotLifecycleDecision(
                intent="hold",
                reason="pending_entry_risk_overlay_cancel",
                gate="CANCEL_PENDING_ENTRY_RISK_OVERLAY",
                pending_clear_entry=True,
                trace={
                    "stage": "pending",
                    "pending_kind": "entry",
                    "direction": str(pending_entry_dir),
                    "due": pending_entry_due_ts.isoformat(),
                },
            )
        if now_ts >= pending_entry_due_ts:
            if not bool(has_open):
                return SpotLifecycleDecision(
                    intent="enter",
                    reason="next_open",
                    gate="TRIGGER_ENTRY",
                    direction=str(pending_entry_dir),
                    fill_mode=entry_fill_mode,
                    pending_clear_entry=True,
                    trace={
                        "stage": "pending",
                        "pending_kind": "entry",
                        "direction": str(pending_entry_dir),
                        "due": pending_entry_due_ts.isoformat(),
                    },
                )
            return SpotLifecycleDecision(
                intent="hold",
                reason="pending_entry_blocked_open",
                gate="CLEAR_PENDING_ENTRY",
                pending_clear_entry=True,
                trace={
                    "stage": "pending",
                    "pending_kind": "entry",
                    "direction": str(pending_entry_dir),
                    "due": pending_entry_due_ts.isoformat(),
                },
            )
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_entry_wait",
            gate="PENDING_ENTRY_NEXT_OPEN",
            direction=str(pending_entry_dir),
            fill_mode=entry_fill_mode,
            trace={
                "stage": "pending",
                "pending_kind": "entry",
                "direction": str(pending_entry_dir),
                "due": pending_entry_due_ts.isoformat(),
            },
        )

    if pending_exit_due_ts is not None:
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_exit_wait",
            gate="PENDING_EXIT_NEXT_OPEN",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            fill_mode=exit_fill_mode,
            trace={"stage": "pending", "pending_kind": "exit", "due": pending_exit_due_ts.isoformat()},
        )
    return SpotLifecycleDecision(intent="hold", reason="no_pending", gate="HOLDING", trace={"stage": "pending"})


def decide_open_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    bar_size: str,
    open_dir: str | None,
    current_qty: int,
    exit_candidates: Mapping[str, bool] | None = None,
    exit_priority: Sequence[str] | None = None,
    target_qty: int | None = None,
    spot_decision: dict[str, object] | None = None,
    last_resize_bar_ts: datetime | None = None,
    signal_entry_dir: str | None = None,
    shock_atr_pct: float | None = None,
    shock_atr_vel_pct: float | None = None,
    shock_atr_accel_pct: float | None = None,
    tr_ratio: float | None = None,
    tr_median_pct: float | None = None,
    slope_med_pct: float | None = None,
    slope_vel_pct: float | None = None,
    slope_med_slow_pct: float | None = None,
    slope_vel_slow_pct: float | None = None,
) -> SpotLifecycleDecision:
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    exit_pick = graph.resolve_exit_reason(
        strategy=strategy,
        open_dir=open_dir,
        signal_entry_dir=signal_entry_dir,
        exit_candidates=exit_candidates,
        exit_priority=exit_priority,
        tr_ratio=tr_ratio,
        tr_median_pct=tr_median_pct,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
    )
    exit_reason = exit_pick.reason
    if exit_reason:
        flip_fill = _normalize_fill_mode(_get(strategy, "spot_flip_exit_fill_mode", "close"), default="close")
        fill_mode = (
            str(flip_fill)
            if exit_reason == "flip" and spot_fill_mode_is_deferred(flip_fill)
            else SPOT_FILL_MODE_CLOSE
        )

        queue_reentry_dir = None
        if bool(_get(strategy, "spot_controlled_flip", False)) and exit_reason == "flip":
            desired = str(signal_entry_dir) if signal_entry_dir in ("up", "down") else None
            if desired is not None and desired != str(open_dir):
                queue_reentry_dir = desired

        return SpotLifecycleDecision(
            intent="exit",
            reason=str(exit_reason),
            gate="TRIGGER_EXIT",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            fill_mode=fill_mode,
            queue_reentry_dir=queue_reentry_dir,
            trace={
                "stage": "open",
                "path": "exit",
                "exit_reason": str(exit_reason),
                "fill_mode": fill_mode,
                "controlled_flip": bool(queue_reentry_dir is not None),
                "exit_policy": exit_pick.as_payload(),
            },
        )

    if target_qty is None:
        return SpotLifecycleDecision(
            intent="hold",
            reason="holding_no_resize_target",
            gate="HOLDING",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            trace={"stage": "open", "path": "hold"},
        )

    resize_target = graph.resolve_resize_target(
        strategy=strategy,
        current_qty=int(current_qty),
        base_target_qty=int(target_qty),
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
    )
    effective_target = int(resize_target.target_qty)
    adaptive = dict(resize_target.trace)
    spot_intent = SpotPolicy.resolve_position_intent(
        strategy=strategy,
        current_qty=int(current_qty),
        target_qty=int(effective_target),
    )
    if str(spot_intent.intent) == "resize":
        cfg = SpotPolicy.policy_config(strategy=strategy, filters=None)
        cooldown = max(0, int(cfg.spot_resize_cooldown_bars))
        if cooldown > 0:
            elapsed = _bars_elapsed(last_resize_bar_ts, bar_ts, bar_size=str(bar_size))
            if elapsed < int(cooldown):
                return SpotLifecycleDecision(
                    intent="hold",
                    reason="resize_cooldown",
                    gate="BLOCKED_RESIZE_COOLDOWN",
                    direction=str(open_dir) if open_dir in ("up", "down") else None,
                    blocked=True,
                    spot_intent=spot_intent,
                    spot_decision=spot_decision,
                    trace={
                        "stage": "open",
                        "path": "resize",
                        "cooldown_bars": int(cooldown),
                        "elapsed_bars": int(elapsed),
                        "resize_policy": adaptive,
                    },
                )
        return SpotLifecycleDecision(
            intent="resize",
            reason=str(spot_intent.reason or "target_delta"),
            gate="TRIGGER_RESIZE",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "resize", "resize_policy": adaptive},
        )

    if str(spot_intent.intent) == "enter":
        target_dir = "up" if int(spot_intent.target_qty) > 0 else "down" if int(spot_intent.target_qty) < 0 else None
        return SpotLifecycleDecision(
            intent="enter",
            reason=str(spot_intent.reason or "from_flat"),
            gate="TRIGGER_ENTRY",
            direction=target_dir,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "enter", "resize_policy": adaptive},
        )

    if str(spot_intent.intent) == "exit":
        return SpotLifecycleDecision(
            intent="exit",
            reason=str(spot_intent.reason or "target_zero"),
            gate="TRIGGER_EXIT",
            direction=str(open_dir) if open_dir in ("up", "down") else None,
            spot_intent=spot_intent,
            spot_decision=spot_decision,
            trace={"stage": "open", "path": "exit_from_intent", "resize_policy": adaptive},
        )

    blocked_gate = "HOLDING" if not bool(spot_intent.blocked) else "BLOCKED_RESIZE"
    return SpotLifecycleDecision(
        intent="hold",
        reason=str(spot_intent.reason or "holding"),
        gate=blocked_gate,
        direction=str(open_dir) if open_dir in ("up", "down") else None,
        blocked=bool(spot_intent.blocked),
        spot_intent=spot_intent,
        spot_decision=spot_decision,
        trace={"stage": "open", "path": "hold", "resize_policy": adaptive},
    )


def decide_flat_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    bar_ts: datetime,
    entry_dir: str | None,
    allowed_directions: Sequence[str],
    can_order_now: bool,
    preflight_ok: bool,
    filters_ok: bool,
    entry_capacity: bool,
    stale_signal: bool = False,
    gap_signal: bool = False,
    pending_exists: bool = False,
    atr_ready: bool = True,
    next_open_allowed: bool = True,
    shock_atr_pct: float | None = None,
    shock_atr_vel_pct: float | None = None,
    shock_atr_accel_pct: float | None = None,
    tr_ratio: float | None = None,
    tr_median_pct: float | None = None,
    slope_med_pct: float | None = None,
    slope_vel_pct: float | None = None,
    slope_med_slow_pct: float | None = None,
    slope_vel_slow_pct: float | None = None,
) -> SpotLifecycleDecision:
    if bool(stale_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="stale_signal",
            gate="BLOCKED_STALE_SIGNAL",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if bool(gap_signal):
        return SpotLifecycleDecision(
            intent="hold",
            reason="data_gap",
            gate="WAITING_DATA_GAP",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(preflight_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="preflight",
            gate="WAITING_PREFLIGHT_BARS",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if bool(pending_exists):
        return SpotLifecycleDecision(
            intent="hold",
            reason="pending_order",
            gate="PENDING_ORDER",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(can_order_now):
        return SpotLifecycleDecision(
            intent="hold",
            reason="weekday_gate",
            gate="BLOCKED_WEEKDAY_NOW",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(entry_capacity):
        return SpotLifecycleDecision(
            intent="hold",
            reason="entry_limit",
            gate="BLOCKED_ENTRY_LIMIT",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if entry_dir not in ("up", "down"):
        return SpotLifecycleDecision(
            intent="hold",
            reason="waiting_signal",
            gate="WAITING_SIGNAL",
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if str(entry_dir) not in {str(d) for d in allowed_directions}:
        return SpotLifecycleDecision(
            intent="hold",
            reason="direction_blocked",
            gate="BLOCKED_DIRECTION",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(filters_ok):
        return SpotLifecycleDecision(
            intent="hold",
            reason="filters",
            gate="BLOCKED_FILTERS",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    if not bool(atr_ready):
        return SpotLifecycleDecision(
            intent="hold",
            reason="atr_not_ready",
            gate="BLOCKED_ATR_NOT_READY",
            direction=str(entry_dir),
            blocked=True,
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )

    fill_mode = _normalize_fill_mode(_get(strategy, "spot_entry_fill_mode", "close"), default="close")
    if spot_fill_mode_is_next_tradable(fill_mode) and not bool(next_open_allowed):
        return SpotLifecycleDecision(
            intent="hold",
            reason="next_open_not_allowed",
            gate="BLOCKED_NEXT_OPEN",
            direction=str(entry_dir),
            blocked=True,
            fill_mode=str(fill_mode),
            trace={"stage": "flat", "bar_ts": bar_ts.isoformat()},
        )
    graph = SpotPolicyGraph.from_sources(strategy=strategy, filters=None)
    entry_gate = graph.evaluate_entry_gate(
        strategy=strategy,
        bar_ts=bar_ts,
        entry_dir=str(entry_dir) if entry_dir in ("up", "down") else None,
        shock_atr_pct=shock_atr_pct,
        shock_atr_vel_pct=shock_atr_vel_pct,
        shock_atr_accel_pct=shock_atr_accel_pct,
        tr_ratio=tr_ratio,
        tr_median_pct=tr_median_pct,
        slope_med_pct=slope_med_pct,
        slope_vel_pct=slope_vel_pct,
        slope_med_slow_pct=slope_med_slow_pct,
        slope_vel_slow_pct=slope_vel_slow_pct,
    )
    if not bool(entry_gate.allow):
        return SpotLifecycleDecision(
            intent="hold",
            reason=str(entry_gate.reason or "graph_entry_gate"),
            gate=str(entry_gate.gate or "BLOCKED_GRAPH_ENTRY"),
            direction=str(entry_dir) if entry_dir in ("up", "down") else None,
            fill_mode=str(fill_mode),
            blocked=True,
            trace={
                "stage": "flat",
                "bar_ts": bar_ts.isoformat(),
                "fill_mode": str(fill_mode),
                "graph_entry": entry_gate.as_payload(),
            },
        )
    return SpotLifecycleDecision(
        intent="enter",
        reason="entry",
        gate="TRIGGER_ENTRY",
        direction=str(entry_dir),
        fill_mode=str(fill_mode),
        blocked=False,
        trace={
            "stage": "flat",
            "bar_ts": bar_ts.isoformat(),
            "fill_mode": str(fill_mode),
            "graph_entry": entry_gate.as_payload(),
        },
    )
