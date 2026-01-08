"""Shared decision logic used by both live trading and backtests.

This module intentionally contains only:
- pure math / state machines (EMA, regime gating, debounce),
- small helper policies (filters, flip-exit hit detection),
- no IBKR calls, no async, no pricing models.

The goal is that both:
- `tradebot/ui.py` (live) and
- `tradebot/backtest/engine.py` (offline)
use the same entry/exit signal semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from .signals import (
    ema_cross,
    ema_next,
    ema_periods,
    ema_slope_pct,
    ema_spread_pct,
    ema_state_direction,
    flip_exit_mode,
    normalize_ema_entry_mode,
    parse_bar_size,
    trend_confirmed_state,
    update_cross_confirm,
)


def annualization_factor(bar_size: str, use_rth: bool) -> float:
    label = str(bar_size or "").strip().lower()
    if "day" in label:
        return 252.0
    bar_def = parse_bar_size(bar_size)
    if bar_def is None:
        return 252.0
    hours = bar_def.duration.total_seconds() / 3600.0
    if hours <= 0:
        return 252.0
    session_hours = 6.5 if use_rth else 24.0
    return 252.0 * (session_hours / hours)


def ewma_vol(returns: Iterable[float], lam: float) -> float:
    variance = 0.0
    alpha = 1.0 - float(lam)
    for r in returns:
        variance = float(lam) * variance + alpha * (float(r) * float(r))
    return math.sqrt(max(0.0, variance))


def annualized_ewma_vol(
    returns: Iterable[float],
    *,
    lam: float,
    bar_size: str,
    use_rth: bool,
) -> float:
    vol = ewma_vol(returns, lam)
    return vol * math.sqrt(annualization_factor(bar_size, use_rth))


def realized_vol_from_closes(
    closes: list[float],
    *,
    lookback: int,
    lam: float,
    bar_size: str,
    use_rth: bool,
) -> float | None:
    if len(closes) < 2:
        return None
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = float(closes[i - 1])
        cur = float(closes[i])
        if prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
        else:
            returns.append(0.0)
    if not returns:
        return None
    window = returns[-int(lookback) :] if int(lookback) > 0 else returns
    return annualized_ewma_vol(window, lam=float(lam), bar_size=bar_size, use_rth=use_rth)


def bars_elapsed(entry_ts: datetime, current_ts: datetime, *, bar_size: str) -> int:
    bar_def = parse_bar_size(bar_size)
    if bar_def is None:
        return 0
    dur = bar_def.duration
    if dur <= timedelta(0):
        return 0
    delta = current_ts - entry_ts
    if delta <= timedelta(0):
        return 0
    return int(delta.total_seconds() // dur.total_seconds())


def cooldown_ok_by_index(*, current_idx: int, last_entry_idx: int | None, cooldown_bars: int) -> bool:
    try:
        cooldown = int(cooldown_bars or 0)
    except (TypeError, ValueError):
        cooldown = 0
    if cooldown <= 0:
        return True
    if last_entry_idx is None:
        return True
    return (int(current_idx) - int(last_entry_idx)) >= cooldown


def cooldown_ok_by_time(
    *,
    current_bar_ts: datetime,
    last_entry_bar_ts: datetime | None,
    bar_size: str,
    cooldown_bars: int,
) -> bool:
    try:
        cooldown = int(cooldown_bars or 0)
    except (TypeError, ValueError):
        cooldown = 0
    if cooldown <= 0:
        return True
    if last_entry_bar_ts is None:
        return True
    return bars_elapsed(last_entry_bar_ts, current_bar_ts, bar_size=bar_size) >= cooldown


@dataclass(frozen=True)
class EmaDecisionSnapshot:
    ema_fast: float | None
    ema_slow: float | None
    prev_ema_fast: float | None
    prev_ema_slow: float | None
    ema_ready: bool
    cross_up: bool
    cross_down: bool
    state: str | None
    entry_dir: str | None
    regime_dir: str | None
    regime_ready: bool


class EmaDecisionEngine:
    def __init__(
        self,
        *,
        ema_preset: str,
        ema_entry_mode: str | None,
        entry_confirm_bars: int = 0,
        regime_ema_preset: str | None = None,
    ) -> None:
        periods = ema_periods(ema_preset)
        if periods is None:
            raise ValueError(f"Invalid EMA preset: {ema_preset!r}")
        self._fast_p, self._slow_p = periods
        self._entry_mode = normalize_ema_entry_mode(ema_entry_mode)
        try:
            self._confirm_bars = int(entry_confirm_bars or 0)
        except (TypeError, ValueError):
            self._confirm_bars = 0
        self._confirm_bars = max(0, self._confirm_bars)

        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._prev_ema_fast: float | None = None
        self._prev_ema_slow: float | None = None
        self._count = 0

        self._entry_state: str | None = None
        self._entry_streak = 0
        self._pending_cross_dir: str | None = None
        self._pending_cross_bars = 0

        self._regime_fast_p: int | None = None
        self._regime_slow_p: int | None = None
        regime_raw = str(regime_ema_preset or "").strip()
        regime_periods = ema_periods(regime_raw) if regime_raw else None
        if regime_periods is not None:
            self._regime_fast_p, self._regime_slow_p = regime_periods
        self._regime_fast: float | None = None
        self._regime_slow: float | None = None
        self._regime_count = 0

    def update(self, close: float) -> EmaDecisionSnapshot:
        if close <= 0:
            return EmaDecisionSnapshot(
                ema_fast=self._ema_fast,
                ema_slow=self._ema_slow,
                prev_ema_fast=self._prev_ema_fast,
                prev_ema_slow=self._prev_ema_slow,
                ema_ready=False,
                cross_up=False,
                cross_down=False,
                state=None,
                entry_dir=None,
                regime_dir=None,
                regime_ready=self._regime_fast_p is None,
            )

        if self._regime_fast_p is not None and self._regime_slow_p is not None:
            self._regime_fast = ema_next(self._regime_fast, close, self._regime_fast_p)
            self._regime_slow = ema_next(self._regime_slow, close, self._regime_slow_p)
            self._regime_count += 1

        self._prev_ema_fast = self._ema_fast
        self._prev_ema_slow = self._ema_slow
        self._ema_fast = ema_next(self._ema_fast, close, self._fast_p)
        self._ema_slow = ema_next(self._ema_slow, close, self._slow_p)
        self._count += 1

        ema_ready = (
            self._count >= self._slow_p and self._ema_fast is not None and self._ema_slow is not None
        )
        cross_up = False
        cross_down = False
        state = None
        entry_dir = None

        if ema_ready:
            state = ema_state_direction(self._ema_fast, self._ema_slow)
            if state is None:
                self._entry_state = None
                self._entry_streak = 0
            elif state == self._entry_state:
                self._entry_streak += 1
            else:
                self._entry_state = state
                self._entry_streak = 1

            if self._prev_ema_fast is not None and self._prev_ema_slow is not None:
                cross_up, cross_down = ema_cross(
                    self._prev_ema_fast,
                    self._prev_ema_slow,
                    self._ema_fast,
                    self._ema_slow,
                )

            if self._entry_mode == "cross":
                entry_dir, self._pending_cross_dir, self._pending_cross_bars = update_cross_confirm(
                    cross_up=bool(cross_up),
                    cross_down=bool(cross_down),
                    state=state,
                    confirm_bars=self._confirm_bars,
                    pending_dir=self._pending_cross_dir,
                    pending_bars=self._pending_cross_bars,
                )
            else:
                entry_dir = trend_confirmed_state(
                    state,
                    self._entry_streak,
                    confirm_bars=self._confirm_bars,
                )
        else:
            self._entry_state = None
            self._entry_streak = 0
            self._pending_cross_dir = None
            self._pending_cross_bars = 0

        regime_ready = True
        regime_dir = None
        if self._regime_fast_p is not None and self._regime_slow_p is not None:
            regime_ready = (
                self._regime_count >= self._regime_slow_p
                and self._regime_fast is not None
                and self._regime_slow is not None
            )
            if regime_ready:
                regime_dir = ema_state_direction(self._regime_fast, self._regime_slow)
            if entry_dir is not None and regime_dir != entry_dir:
                entry_dir = None
            if not regime_ready:
                entry_dir = None

        return EmaDecisionSnapshot(
            ema_fast=float(self._ema_fast) if self._ema_fast is not None else None,
            ema_slow=float(self._ema_slow) if self._ema_slow is not None else None,
            prev_ema_fast=float(self._prev_ema_fast) if self._prev_ema_fast is not None else None,
            prev_ema_slow=float(self._prev_ema_slow) if self._prev_ema_slow is not None else None,
            ema_ready=bool(ema_ready),
            cross_up=bool(cross_up),
            cross_down=bool(cross_down),
            state=state,
            entry_dir=str(entry_dir) if entry_dir is not None else None,
            regime_dir=str(regime_dir) if regime_dir is not None else None,
            regime_ready=bool(regime_ready),
        )


def apply_regime_gate(
    signal: EmaDecisionSnapshot | None,
    *,
    regime_dir: str | None,
    regime_ready: bool,
) -> EmaDecisionSnapshot | None:
    if signal is None:
        return None
    cleaned_regime_dir = str(regime_dir) if regime_dir in ("up", "down") else None
    entry_dir = signal.entry_dir
    if entry_dir is not None:
        if not bool(regime_ready):
            entry_dir = None
        elif cleaned_regime_dir is None or cleaned_regime_dir != entry_dir:
            entry_dir = None
    return EmaDecisionSnapshot(
        ema_fast=signal.ema_fast,
        ema_slow=signal.ema_slow,
        prev_ema_fast=signal.prev_ema_fast,
        prev_ema_slow=signal.prev_ema_slow,
        ema_ready=signal.ema_ready,
        cross_up=signal.cross_up,
        cross_down=signal.cross_down,
        state=signal.state,
        entry_dir=entry_dir,
        regime_dir=cleaned_regime_dir,
        regime_ready=bool(regime_ready),
    )


def flip_exit_hit(
    *,
    exit_on_signal_flip: bool,
    open_dir: str | None,
    signal: EmaDecisionSnapshot | None,
    flip_exit_mode_raw: str | None,
    ema_entry_mode_raw: str | None,
) -> bool:
    if not bool(exit_on_signal_flip):
        return False
    if open_dir not in ("up", "down"):
        return False
    if signal is None or not signal.ema_ready or signal.ema_fast is None or signal.ema_slow is None:
        return False

    mode = flip_exit_mode(flip_exit_mode_raw, ema_entry_mode_raw)
    if mode == "cross":
        if open_dir == "up":
            return bool(signal.cross_down)
        return bool(signal.cross_up)

    state = signal.state or ema_state_direction(signal.ema_fast, signal.ema_slow)
    if state is None:
        return False
    if open_dir == "up":
        return state == "down"
    return state == "up"


def signal_filters_ok(
    filters: Mapping[str, object] | object | None,
    *,
    bar_ts: datetime,
    bars_in_day: int,
    close: float,
    rv: float | None,
    signal: EmaDecisionSnapshot | None,
    cooldown_ok: bool = True,
) -> bool:
    if filters is None:
        return True

    def _get(key: str):
        if isinstance(filters, Mapping):
            return filters.get(key)
        return getattr(filters, key, None)

    rv_min = _get("rv_min")
    rv_max = _get("rv_max")
    if rv_min is not None or rv_max is not None:
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

    entry_start_hour = _get("entry_start_hour")
    entry_end_hour = _get("entry_end_hour")
    if entry_start_hour is not None and entry_end_hour is not None:
        try:
            start = int(entry_start_hour)
            end = int(entry_end_hour)
        except (TypeError, ValueError):
            start = None
            end = None
        if start is not None and end is not None:
            hour = int(bar_ts.hour)
            if start <= end:
                if not (start <= hour < end):
                    return False
            else:
                if not (hour >= start or hour < end):
                    return False

    skip_first = _get("skip_first_bars")
    try:
        skip_first_n = int(skip_first or 0)
    except (TypeError, ValueError):
        skip_first_n = 0
    if skip_first_n > 0 and int(bars_in_day) <= skip_first_n:
        return False

    if not bool(cooldown_ok):
        return False

    spread_min = _get("ema_spread_min_pct")
    if spread_min is not None:
        try:
            spread_min_f = float(spread_min)
        except (TypeError, ValueError):
            spread_min_f = None
        if spread_min_f is not None:
            if signal is None or not signal.ema_ready or signal.ema_fast is None or signal.ema_slow is None:
                return False
            spread = ema_spread_pct(signal.ema_fast, signal.ema_slow, close)
            if spread < spread_min_f:
                return False

    slope_min = _get("ema_slope_min_pct")
    if slope_min is not None:
        try:
            slope_min_f = float(slope_min)
        except (TypeError, ValueError):
            slope_min_f = None
        if slope_min_f is not None:
            if (
                signal is None
                or not signal.ema_ready
                or signal.ema_fast is None
                or signal.prev_ema_fast is None
            ):
                return False
            slope = ema_slope_pct(signal.ema_fast, signal.prev_ema_fast, close)
            if slope < slope_min_f:
                return False

    return True
