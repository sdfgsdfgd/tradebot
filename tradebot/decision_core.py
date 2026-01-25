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
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

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

_ET_ZONE = ZoneInfo("America/New_York")


def _ts_to_et(ts: datetime) -> datetime:
    """Interpret naive datetimes as UTC and return an ET-aware timestamp."""
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET_ZONE)


def parse_time_hhmm(value: object, *, default: time | None = None) -> time | None:
    """Parse times like '09:30' or '18:00' into a `datetime.time`.

    Returns `default` when parsing fails or value is empty/None.
    """
    if value is None:
        return default
    if isinstance(value, time):
        return value
    if isinstance(value, (int, float)):
        try:
            hour = int(value)
        except (TypeError, ValueError):
            return default
        if 0 <= hour <= 23:
            return time(hour=hour, minute=0)
        return default
    raw = str(value).strip()
    if not raw:
        return default
    if ":" not in raw:
        try:
            hour = int(raw)
        except (TypeError, ValueError):
            return default
        if 0 <= hour <= 23:
            return time(hour=hour, minute=0)
        return default
    parts = raw.split(":")
    if len(parts) != 2:
        return default
    try:
        hour = int(parts[0].strip())
        minute = int(parts[1].strip())
    except (TypeError, ValueError):
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default
    return time(hour=hour, minute=minute)


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


class OrbDecisionEngine:
    """Opening Range Breakout (ORB) entry signal.

    OR is defined as the high/low in the first N minutes after 9:30am ET.
    Emits a one-shot entry_dir ("up" or "down") when close breaks out of that range.
    """

    def __init__(
        self,
        *,
        window_mins: int = 15,
        open_time_et: time = time(9, 30),
    ) -> None:
        try:
            mins = int(window_mins)
        except (TypeError, ValueError):
            mins = 15
        self._window_mins = max(1, mins)
        self._open_time_et = open_time_et

        self._session_date = None
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_ready = False
        self._breakout_fired = False

    @property
    def or_high(self) -> float | None:
        return self._or_high

    @property
    def or_low(self) -> float | None:
        return self._or_low

    @property
    def or_ready(self) -> bool:
        return bool(self._or_ready)

    def update(self, *, ts: datetime, high: float, low: float, close: float) -> EmaDecisionSnapshot:
        ts_et = _ts_to_et(ts)
        session_date = ts_et.date()

        if self._session_date != session_date:
            self._session_date = session_date
            self._or_high = None
            self._or_low = None
            self._or_ready = False
            self._breakout_fired = False

        start = datetime.combine(session_date, self._open_time_et, tzinfo=_ET_ZONE)
        end = start + timedelta(minutes=int(self._window_mins))

        # Our bar timestamps are treated as bar-close timestamps (naive UTC → ET via `_ts_to_et`).
        # The OR window should therefore include bars whose close time lands within (start, end].
        in_or = start < ts_et <= end
        if in_or and high > 0 and low > 0:
            self._or_high = float(high) if self._or_high is None else max(self._or_high, float(high))
            self._or_low = float(low) if self._or_low is None else min(self._or_low, float(low))

        if not self._or_ready and self._or_high is not None and self._or_low is not None and ts_et >= end:
            self._or_ready = True

        entry_dir = None
        if self._or_ready and not self._breakout_fired and self._or_high is not None and self._or_low is not None:
            if float(close) > float(self._or_high):
                entry_dir = "up"
            elif float(close) < float(self._or_low):
                entry_dir = "down"
            if entry_dir is not None:
                self._breakout_fired = True

        return EmaDecisionSnapshot(
            ema_fast=None,
            ema_slow=None,
            prev_ema_fast=None,
            prev_ema_slow=None,
            ema_ready=True,
            cross_up=False,
            cross_down=False,
            state=None,
            entry_dir=entry_dir,
            regime_dir=None,
            regime_ready=True,
        )


@dataclass(frozen=True)
class SupertrendSnapshot:
    direction: str | None  # "up" | "down"
    ready: bool
    atr: float | None = None
    upper: float | None = None
    lower: float | None = None
    value: float | None = None


class SupertrendEngine:
    def __init__(
        self,
        *,
        atr_period: int = 10,
        multiplier: float = 3.0,
        source: str = "hl2",
    ) -> None:
        try:
            period = int(atr_period)
        except (TypeError, ValueError):
            period = 10
        self._atr_period = max(1, period)
        try:
            self._multiplier = float(multiplier)
        except (TypeError, ValueError):
            self._multiplier = 3.0
        self._source = str(source or "hl2").strip().lower()

        self._prev_close: float | None = None
        self._atr: float | None = None
        self._atr_seed_sum = 0.0
        self._atr_seed_count = 0
        self._final_upper: float | None = None
        self._final_lower: float | None = None
        self._direction: int | None = None  # 1=up, -1=down

    def update(self, *, high: float, low: float, close: float) -> SupertrendSnapshot:
        if close <= 0 or high <= 0 or low <= 0:
            return SupertrendSnapshot(direction=None, ready=False)

        tr = float(high) - float(low)
        if self._prev_close is not None:
            prev = float(self._prev_close)
            tr = max(tr, abs(float(high) - prev), abs(float(low) - prev))

        if self._atr is None:
            self._atr_seed_sum += tr
            self._atr_seed_count += 1
            if self._atr_seed_count >= self._atr_period:
                self._atr = self._atr_seed_sum / float(self._atr_period)
        else:
            # Wilder's smoothing (TradingView ta.rma): atr = (prev_atr*(p-1) + tr) / p
            p = float(self._atr_period)
            self._atr = (self._atr * (p - 1.0) + tr) / p

        prev_upper = self._final_upper
        prev_lower = self._final_lower
        prev_close = self._prev_close
        self._prev_close = float(close)

        if self._atr is None:
            return SupertrendSnapshot(direction=None, ready=False)

        hl2 = (float(high) + float(low)) / 2.0
        src = float(close) if self._source in ("close", "c") else hl2
        upper_basic = src + (self._multiplier * float(self._atr))
        lower_basic = src - (self._multiplier * float(self._atr))

        if prev_upper is None:
            upper = upper_basic
        else:
            upper = (
                upper_basic
                if (upper_basic < prev_upper) or (prev_close is not None and float(prev_close) > prev_upper)
                else prev_upper
            )

        if prev_lower is None:
            lower = lower_basic
        else:
            lower = (
                lower_basic
                if (lower_basic > prev_lower) or (prev_close is not None and float(prev_close) < prev_lower)
                else prev_lower
            )

        direction = self._direction
        if direction is None:
            direction = 1
        else:
            if direction == -1 and prev_upper is not None and float(close) > float(prev_upper):
                direction = 1
            elif direction == 1 and prev_lower is not None and float(close) < float(prev_lower):
                direction = -1

        self._final_upper = float(upper)
        self._final_lower = float(lower)
        self._direction = int(direction)

        value = float(lower) if direction == 1 else float(upper)
        return SupertrendSnapshot(
            direction="up" if direction == 1 else "down",
            ready=True,
            atr=float(self._atr),
            upper=float(upper),
            lower=float(lower),
            value=value,
        )


@dataclass(frozen=True)
class AtrRatioShockSnapshot:
    shock: bool
    ready: bool
    ratio: float | None = None
    atr_fast_pct: float | None = None
    atr_fast: float | None = None
    atr_slow: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False


class AtrRatioShockEngine:
    """Fast/slow ATR ratio shock detector with hysteresis.

    This is a *risk state* overlay, not a directional gate:
    - It flags when volatility is in an abnormal regime.
    - It can optionally provide a coarse direction (smoothed returns) to support "shock surfing"
      policies, but it is not meant to replace the primary entry signal.
    """

    def __init__(
        self,
        *,
        atr_fast_period: int = 7,
        atr_slow_period: int = 50,
        on_ratio: float = 1.55,
        off_ratio: float = 1.30,
        min_atr_pct: float = 7.0,
        source: str = "hl2",
        direction_lookback: int = 2,
    ) -> None:
        self._atr_fast = SupertrendEngine(
            atr_period=int(atr_fast_period),
            multiplier=1.0,
            source=source,
        )
        self._atr_slow = SupertrendEngine(
            atr_period=int(atr_slow_period),
            multiplier=1.0,
            source=source,
        )
        self._on_ratio = float(on_ratio)
        self._off_ratio = float(off_ratio)
        self._min_atr_pct = float(min_atr_pct)
        self._shock = False
        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=self._dir_lookback)
        self._direction: str | None = None
        self._atr_ready = False
        self._last_ratio: float | None = None
        self._last_atr_fast_pct: float | None = None
        self._last_atr_fast: float | None = None
        self._last_atr_slow: float | None = None

    def _update_direction(self, close: float) -> None:
        prev_close = self._dir_prev_close
        self._dir_prev_close = float(close)
        if prev_close is not None and prev_close > 0 and close > 0:
            self._ret_hist.append((float(close) / float(prev_close)) - 1.0)
            if len(self._ret_hist) >= self._dir_lookback:
                ret_sum = float(sum(self._ret_hist))
                if ret_sum > 0:
                    self._direction = "up"
                elif ret_sum < 0:
                    self._direction = "down"

    def _snapshot(self) -> AtrRatioShockSnapshot:
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
        if not bool(self._atr_ready):
            return AtrRatioShockSnapshot(
                shock=False,
                ready=False,
                ratio=None,
                atr_fast_pct=None,
                atr_fast=float(self._last_atr_fast) if self._last_atr_fast is not None else None,
                atr_slow=float(self._last_atr_slow) if self._last_atr_slow is not None else None,
                direction=direction,
                direction_ready=direction_ready,
            )
        return AtrRatioShockSnapshot(
            shock=bool(self._shock),
            ready=True,
            ratio=float(self._last_ratio) if self._last_ratio is not None else None,
            atr_fast_pct=float(self._last_atr_fast_pct) if self._last_atr_fast_pct is not None else None,
            atr_fast=float(self._last_atr_fast) if self._last_atr_fast is not None else None,
            atr_slow=float(self._last_atr_slow) if self._last_atr_slow is not None else None,
            direction=direction,
            direction_ready=direction_ready,
        )

    def update(
        self,
        *,
        high: float,
        low: float,
        close: float,
        update_direction: bool = True,
    ) -> AtrRatioShockSnapshot:
        if bool(update_direction):
            self._update_direction(float(close))

        fast = self._atr_fast.update(high=high, low=low, close=close)
        slow = self._atr_slow.update(high=high, low=low, close=close)
        self._last_atr_fast = float(fast.atr) if fast.atr is not None else None
        self._last_atr_slow = float(slow.atr) if slow.atr is not None else None

        if not bool(fast.ready) or not bool(slow.ready) or fast.atr is None or slow.atr is None:
            self._atr_ready = False
            return self._snapshot()
        if close <= 0:
            self._atr_ready = False
            return self._snapshot()

        atr_fast = float(fast.atr)
        atr_slow = float(slow.atr)
        ratio = atr_fast / max(atr_slow, 1e-9)
        atr_fast_pct = atr_fast / max(float(close), 1e-9) * 100.0
        self._last_ratio = float(ratio)
        self._last_atr_fast_pct = float(atr_fast_pct)
        self._atr_ready = True

        if not self._shock:
            if ratio >= self._on_ratio and atr_fast_pct >= self._min_atr_pct:
                self._shock = True
        else:
            if ratio <= self._off_ratio:
                self._shock = False

        return self._snapshot()

    def update_direction(self, *, close: float) -> AtrRatioShockSnapshot:
        self._update_direction(float(close))
        return self._snapshot()


@dataclass(frozen=True)
class DailyAtrPctShockSnapshot:
    shock: bool
    ready: bool
    atr_pct: float | None = None
    atr: float | None = None
    tr: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False


class DailyAtrPctShockEngine:
    """Daily ATR% shock detector with hysteresis.

    - Computes daily TR and ATR using Wilder smoothing (updates once per session/day).
    - Provides an intraday ATR estimate using TR-so-far for the current day and the last finalized ATR.
    - Direction is derived from smoothed close-to-close returns (bar-to-bar), to support "shock surfing".
    """

    def __init__(
        self,
        *,
        atr_period: int = 14,
        on_atr_pct: float = 13.0,
        off_atr_pct: float = 11.0,
        on_tr_pct: float | None = None,
        direction_lookback: int = 2,
    ) -> None:
        self._period = max(1, int(atr_period))
        self._on = float(on_atr_pct)
        self._off = float(off_atr_pct)
        if self._off > self._on:
            self._off = self._on

        self._shock = False
        self._prev_day_close: float | None = None
        self._atr: float | None = None
        self._tr_hist: deque[float] = deque(maxlen=self._period)
        self._on_tr_pct = None
        if on_tr_pct is not None:
            try:
                v = float(on_tr_pct)
            except (TypeError, ValueError):
                v = None
            if v is not None and v > 0:
                self._on_tr_pct = float(v)
        # When an intraday TrueRange% trigger is enabled, we treat a TR%-exceedance as a
        # "shock day" and keep shock ON for the remainder of that day (TR is monotonic within
        # a session). This prevents immediate off-flicker when ATR% smoothing remains low.
        self._tr_trigger_day: date | None = None

        self._cur_day: date | None = None
        self._cur_high: float | None = None
        self._cur_low: float | None = None
        self._cur_close: float | None = None

        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=self._dir_lookback)
        self._direction: str | None = None

    def _update_direction(self, close: float) -> None:
        prev_close = self._dir_prev_close
        self._dir_prev_close = float(close)
        if prev_close is not None and prev_close > 0 and close > 0:
            self._ret_hist.append((float(close) / float(prev_close)) - 1.0)
            if len(self._ret_hist) >= self._dir_lookback:
                ret_sum = float(sum(self._ret_hist))
                if ret_sum > 0:
                    self._direction = "up"
                elif ret_sum < 0:
                    self._direction = "down"

    @staticmethod
    def _true_range(high: float, low: float, prev_close: float | None) -> float:
        h = float(high)
        l = float(low)
        if prev_close is None:
            return max(0.0, h - l)
        pc = float(prev_close)
        return max(0.0, h - l, abs(h - pc), abs(l - pc))

    def _finalize_day(self) -> None:
        if self._cur_day is None:
            return
        if self._cur_high is None or self._cur_low is None or self._cur_close is None:
            return
        tr = self._true_range(self._cur_high, self._cur_low, self._prev_day_close)
        self._tr_hist.append(float(tr))
        if self._atr is None:
            if len(self._tr_hist) >= self._period:
                self._atr = float(sum(self._tr_hist) / float(self._period))
        else:
            self._atr = (float(self._atr) * float(self._period - 1) + float(tr)) / float(self._period)
        self._prev_day_close = float(self._cur_close)

    def _snapshot(
        self,
        *,
        shock: bool,
        ready: bool,
        atr_pct: float | None,
        atr: float | None,
        tr: float | None,
    ) -> DailyAtrPctShockSnapshot:
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
        return DailyAtrPctShockSnapshot(
            shock=bool(shock),
            ready=bool(ready),
            atr_pct=float(atr_pct) if atr_pct is not None else None,
            atr=float(atr) if atr is not None else None,
            tr=float(tr) if tr is not None else None,
            direction=direction,
            direction_ready=direction_ready,
        )

    def update(
        self,
        *,
        day: date,
        high: float,
        low: float,
        close: float,
        update_direction: bool = True,
    ) -> DailyAtrPctShockSnapshot:
        if bool(update_direction):
            self._update_direction(float(close))

        if self._cur_day is None:
            self._cur_day = day
            self._cur_high = float(high)
            self._cur_low = float(low)
            self._cur_close = float(close)
        elif day != self._cur_day:
            self._finalize_day()
            self._cur_day = day
            self._cur_high = float(high)
            self._cur_low = float(low)
            self._cur_close = float(close)
            self._tr_trigger_day = None
        else:
            self._cur_high = max(float(self._cur_high), float(high)) if self._cur_high is not None else float(high)
            self._cur_low = min(float(self._cur_low), float(low)) if self._cur_low is not None else float(low)
            self._cur_close = float(close)

        if close <= 0 or self._cur_high is None or self._cur_low is None:
            return self._snapshot(shock=False, ready=False, atr_pct=None, atr=self._atr, tr=None)

        tr_so_far = self._true_range(float(self._cur_high), float(self._cur_low), self._prev_day_close)
        if self._atr is None:
            denom = float(len(self._tr_hist) + 1)
            atr_est = (float(sum(self._tr_hist)) + float(tr_so_far)) / max(denom, 1.0)
            ready = False
        else:
            atr_est = (float(self._atr) * float(self._period - 1) + float(tr_so_far)) / float(self._period)
            ready = True

        atr_pct = float(atr_est) / max(float(close), 1e-9) * 100.0

        if bool(ready):
            tr_pct = None
            if self._on_tr_pct is not None and self._prev_day_close is not None and self._prev_day_close > 0:
                tr_pct = float(tr_so_far) / float(self._prev_day_close) * 100.0
                if tr_pct >= float(self._on_tr_pct):
                    self._tr_trigger_day = day
            if not self._shock:
                if atr_pct >= self._on or (tr_pct is not None and tr_pct >= float(self._on_tr_pct)):
                    self._shock = True
            else:
                # If TR%-triggered today, keep shock ON for the rest of the session.
                if self._tr_trigger_day == day:
                    pass
                elif atr_pct <= self._off:
                    self._shock = False

        shock = bool(self._shock) if bool(ready) else False
        return self._snapshot(shock=shock, ready=ready, atr_pct=atr_pct, atr=atr_est, tr=tr_so_far)


@dataclass(frozen=True)
class DailyDrawdownShockSnapshot:
    shock: bool
    ready: bool
    drawdown_pct: float | None = None
    peak_close: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False


class DailyDrawdownShockEngine:
    """Daily drawdown shock detector with hysteresis.

    Tracks close vs a rolling peak close over the last N finalized sessions and triggers
    shock when drawdown exceeds a threshold (e.g., <= -20%).
    """

    def __init__(
        self,
        *,
        lookback_days: int = 20,
        on_drawdown_pct: float = -20.0,
        off_drawdown_pct: float = -10.0,
        direction_lookback: int = 2,
    ) -> None:
        self._lookback = max(2, int(lookback_days))
        self._on = float(on_drawdown_pct)
        self._off = float(off_drawdown_pct)
        if self._off < self._on:
            self._off = self._on

        self._shock = False
        self._cur_day: date | None = None
        self._cur_close: float | None = None
        self._daily_closes: deque[float] = deque(maxlen=self._lookback)
        self._rolling_peak: float | None = None

        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=self._dir_lookback)
        self._direction: str | None = None

    def _update_direction(self, close: float) -> None:
        prev_close = self._dir_prev_close
        self._dir_prev_close = float(close)
        if prev_close is not None and prev_close > 0 and close > 0:
            self._ret_hist.append((float(close) / float(prev_close)) - 1.0)
            if len(self._ret_hist) >= self._dir_lookback:
                ret_sum = float(sum(self._ret_hist))
                if ret_sum > 0:
                    self._direction = "up"
                elif ret_sum < 0:
                    self._direction = "down"

    def _finalize_day(self) -> None:
        if self._cur_close is None:
            return
        close = float(self._cur_close)
        if close > 0:
            self._daily_closes.append(close)
            self._rolling_peak = max(self._daily_closes) if self._daily_closes else None

    def _snapshot(
        self,
        *,
        shock: bool,
        ready: bool,
        drawdown_pct: float | None,
        peak_close: float | None,
    ) -> DailyDrawdownShockSnapshot:
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
        return DailyDrawdownShockSnapshot(
            shock=bool(shock),
            ready=bool(ready),
            drawdown_pct=float(drawdown_pct) if drawdown_pct is not None else None,
            peak_close=float(peak_close) if peak_close is not None else None,
            direction=direction,
            direction_ready=direction_ready,
        )

    def update(
        self,
        *,
        day: date,
        high: float,
        low: float,
        close: float,
        update_direction: bool = True,
    ) -> DailyDrawdownShockSnapshot:
        if bool(update_direction):
            self._update_direction(float(close))

        if self._cur_day is None:
            self._cur_day = day
            self._cur_close = float(close)
        elif day != self._cur_day:
            self._finalize_day()
            self._cur_day = day
            self._cur_close = float(close)
        else:
            self._cur_close = float(close)

        peak = self._rolling_peak
        ready = bool(len(self._daily_closes) >= self._lookback and peak is not None and peak > 0 and close > 0)
        if not ready:
            return self._snapshot(shock=False, ready=False, drawdown_pct=None, peak_close=peak)

        peak_eff = max(float(peak), float(close))
        dd_pct = (float(close) / float(peak_eff) - 1.0) * 100.0

        if not self._shock:
            if dd_pct <= self._on:
                self._shock = True
        else:
            if dd_pct >= self._off:
                self._shock = False

        return self._snapshot(shock=bool(self._shock), ready=True, drawdown_pct=dd_pct, peak_close=peak_eff)


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
    volume: float | None = None,
    volume_ema: float | None = None,
    volume_ema_ready: bool = True,
    rv: float | None = None,
    signal: EmaDecisionSnapshot | None = None,
    cooldown_ok: bool = True,
    shock: bool | None = None,
    shock_dir: str | None = None,
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

    entry_start_hour_et = _get("entry_start_hour_et")
    entry_end_hour_et = _get("entry_end_hour_et")
    if entry_start_hour_et is not None and entry_end_hour_et is not None:
        try:
            start = int(entry_start_hour_et)
            end = int(entry_end_hour_et)
        except (TypeError, ValueError):
            start = None
            end = None
        if start is not None and end is not None:
            hour = int(_ts_to_et(bar_ts).hour)
            if start <= end:
                if not (start <= hour < end):
                    return False
            else:
                if not (hour >= start or hour < end):
                    return False
    else:
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

    shock_gate_mode = _get("shock_gate_mode")
    if shock_gate_mode is None:
        shock_gate_mode = _get("shock_mode")
    if isinstance(shock_gate_mode, bool):
        shock_gate_mode = "block" if shock_gate_mode else "off"
    shock_mode = str(shock_gate_mode or "off").strip().lower()
    if shock_mode in ("", "0", "false", "none", "null"):
        shock_mode = "off"
    if shock_mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        shock_mode = "off"
    if shock_mode == "block":
        # Like other filters, if the derived feature isn't ready, we block rather than guessing.
        if shock is None:
            return False
        if bool(shock):
            return False
    elif shock_mode in ("block_longs", "block_shorts"):
        if shock is None:
            return False
        if bool(shock):
            entry_dir = None
            if signal is not None:
                entry_dir = signal.entry_dir
            if shock_mode == "block_longs" and entry_dir == "up":
                return False
            if shock_mode == "block_shorts" and entry_dir == "down":
                return False
    elif shock_mode == "surf":
        # During a shock, only allow entries aligned with the shock direction.
        if shock is None:
            return False
        if bool(shock):
            entry_dir = None
            if signal is not None:
                entry_dir = signal.entry_dir
            cleaned = str(shock_dir) if shock_dir in ("up", "down") else None
            if cleaned is None or entry_dir not in ("up", "down"):
                return False
            if entry_dir != cleaned:
                return False

    spread_min = _get("ema_spread_min_pct")
    spread_min_down = _get("ema_spread_min_pct_down")
    # Optional directional override: allow stricter gating for down entries without affecting long entries.
    # This helps reduce short exposure on structurally up-trending instruments (e.g., TQQQ) while still
    # keeping "both directions" enabled.
    if signal is not None and signal.entry_dir == "down" and spread_min_down is not None:
        spread_min = spread_min_down
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

    volume_ratio_min = _get("volume_ratio_min")
    if volume_ratio_min is not None:
        try:
            ratio_min = float(volume_ratio_min)
        except (TypeError, ValueError):
            ratio_min = None
        if ratio_min is not None:
            if not bool(volume_ema_ready):
                return False
            if volume is None or volume_ema is None:
                return False
            denom = float(volume_ema)
            if denom <= 0:
                return False
            ratio = float(volume) / denom
            if ratio < ratio_min:
                return False

    return True
