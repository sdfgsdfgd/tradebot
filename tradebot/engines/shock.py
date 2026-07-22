"""Shock detection policy and state machines."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from ..signals import ema_next
from ..spot.policy_contract import parse_float as _parse_float
from ..spot.policy_contract import parse_int as _parse_int
from ..spot.policy_contract import source_value as _filters_get
from .signals import SupertrendEngine

# region Shock Gate / Engine Factory
def normalize_shock_gate_mode(filters: Mapping[str, object] | object | None) -> str:
    raw = _filters_get(filters, "shock_gate_mode")
    if raw is None:
        raw = _filters_get(filters, "shock_mode")
    if isinstance(raw, bool):
        raw = "block" if raw else "off"
    mode = str(raw or "off").strip().lower()
    if mode in ("", "0", "false", "none", "null"):
        mode = "off"
    if mode not in ("off", "detect", "block", "block_longs", "block_shorts", "surf"):
        mode = "off"
    return mode


def normalize_shock_detector(filters: Mapping[str, object] | object | None) -> str:
    raw = str(_filters_get(filters, "shock_detector") or "atr_ratio").strip().lower()
    if raw in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
        return "daily_atr_pct"
    if raw in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
        return "daily_drawdown"
    if raw in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
        return "tr_ratio"
    if raw in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
        return "atr_ratio"
    if raw not in ("atr_ratio", "tr_ratio", "daily_atr_pct", "daily_drawdown"):
        return "atr_ratio"
    return raw


def normalize_shock_direction_source(filters: Mapping[str, object] | object | None) -> str:
    raw = str(_filters_get(filters, "shock_direction_source") or "regime").strip().lower()
    return raw if raw in ("regime", "signal") else "regime"


def build_shock_engine(filters: Mapping[str, object] | object | None, *, source: str = "hl2") -> ShockEngine | None:
    mode = normalize_shock_gate_mode(filters)
    if mode == "off":
        return None

    detector = normalize_shock_detector(filters)
    dir_lb = _parse_int(_filters_get(filters, "shock_direction_lookback"), default=2, min_value=1)
    if detector == "daily_atr_pct":
        daily_period = _parse_int(_filters_get(filters, "shock_daily_atr_period"), default=14, min_value=1)
        daily_on = _parse_float(_filters_get(filters, "shock_daily_on_atr_pct"), default=13.0)
        daily_off = _parse_float(_filters_get(filters, "shock_daily_off_atr_pct"), default=11.0)
        daily_tr_on_raw = _filters_get(filters, "shock_daily_on_tr_pct")
        try:
            daily_tr_on = float(daily_tr_on_raw) if daily_tr_on_raw is not None else None
        except (TypeError, ValueError):
            daily_tr_on = None
        if daily_tr_on is not None and daily_tr_on <= 0:
            daily_tr_on = None
        if daily_off > daily_on:
            daily_off = daily_on
        return DailyAtrPctShockEngine(
            atr_period=int(daily_period),
            on_atr_pct=float(daily_on),
            off_atr_pct=float(daily_off),
            on_tr_pct=float(daily_tr_on) if daily_tr_on is not None else None,
            direction_lookback=int(dir_lb),
        )

    if detector == "daily_drawdown":
        dd_lb = _parse_int(_filters_get(filters, "shock_drawdown_lookback_days"), default=20, min_value=2)
        dd_on = _parse_float(_filters_get(filters, "shock_on_drawdown_pct"), default=-20.0)
        dd_off = _parse_float(_filters_get(filters, "shock_off_drawdown_pct"), default=-10.0)
        # For a negative drawdown threshold, OFF should be >= ON (less negative).
        if dd_off < dd_on:
            dd_off = dd_on
        return DailyDrawdownShockEngine(
            lookback_days=int(dd_lb),
            on_drawdown_pct=float(dd_on),
            off_drawdown_pct=float(dd_off),
            direction_lookback=int(dir_lb),
        )

    if detector == "tr_ratio":
        tr_fast = _parse_int(
            _filters_get(filters, "shock_tr_fast_period") or _filters_get(filters, "shock_atr_fast_period"),
            default=7,
            min_value=1,
        )
        tr_slow = _parse_int(
            _filters_get(filters, "shock_tr_slow_period") or _filters_get(filters, "shock_atr_slow_period"),
            default=50,
            min_value=1,
        )
        on_ratio = _parse_float(_filters_get(filters, "shock_on_ratio"), default=1.55)
        off_ratio = _parse_float(_filters_get(filters, "shock_off_ratio"), default=1.30)
        min_tr_pct = _parse_float(
            _filters_get(filters, "shock_min_tr_pct") or _filters_get(filters, "shock_min_atr_pct"),
            default=7.0,
        )
        return TrRatioShockEngine(
            tr_fast_period=int(tr_fast),
            tr_slow_period=int(tr_slow),
            on_ratio=float(on_ratio),
            off_ratio=float(off_ratio),
            min_tr_pct=float(min_tr_pct),
            direction_lookback=int(dir_lb),
        )

    atr_fast = _parse_int(_filters_get(filters, "shock_atr_fast_period"), default=7, min_value=1)
    atr_slow = _parse_int(_filters_get(filters, "shock_atr_slow_period"), default=50, min_value=1)
    on_ratio = _parse_float(_filters_get(filters, "shock_on_ratio"), default=1.55)
    off_ratio = _parse_float(_filters_get(filters, "shock_off_ratio"), default=1.30)
    min_atr_pct = _parse_float(_filters_get(filters, "shock_min_atr_pct"), default=7.0)
    return AtrRatioShockEngine(
        atr_fast_period=int(atr_fast),
        atr_slow_period=int(atr_slow),
        on_ratio=float(on_ratio),
        off_ratio=float(off_ratio),
        min_atr_pct=float(min_atr_pct),
        direction_lookback=int(dir_lb),
        source=str(source or "hl2").strip().lower() or "hl2",
    )

# endregion

# region Shock Engines
class _ShockDirectionMixin:
    _dir_lookback: int
    _dir_prev_close: float | None
    _ret_hist: deque[float]
    _direction: str | None
    _direction_ret_sum_pct: float | None

    def _init_direction_state(self, direction_lookback: int) -> None:
        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close = None
        self._ret_hist = deque(maxlen=self._dir_lookback)
        self._direction = None
        self._direction_ret_sum_pct = None

    def _update_direction(self, close: float) -> None:
        prev_close = self._dir_prev_close
        self._dir_prev_close = float(close)
        if prev_close is not None and prev_close > 0 and close > 0:
            self._ret_hist.append((float(close) / float(prev_close)) - 1.0)
            if len(self._ret_hist) >= self._dir_lookback:
                ret_sum = float(sum(self._ret_hist))
                self._direction_ret_sum_pct = float(ret_sum * 100.0)
                if ret_sum > 0:
                    self._direction = "up"
                elif ret_sum < 0:
                    self._direction = "down"
                else:
                    self._direction = None

    def _direction_state(self) -> tuple[str | None, bool, float | None]:
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
        ret_sum_pct = float(self._direction_ret_sum_pct) if self._direction_ret_sum_pct is not None else None
        return direction, direction_ready, ret_sum_pct


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
    direction_ret_sum_pct: float | None = None


class AtrRatioShockEngine(_ShockDirectionMixin):
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
        self._init_direction_state(direction_lookback)
        self._atr_ready = False
        self._last_ratio: float | None = None
        self._last_atr_fast_pct: float | None = None
        self._last_atr_fast: float | None = None
        self._last_atr_slow: float | None = None

    def _snapshot(self) -> AtrRatioShockSnapshot:
        direction, direction_ready, direction_ret_sum_pct = self._direction_state()
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
                direction_ret_sum_pct=direction_ret_sum_pct,
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
            direction_ret_sum_pct=direction_ret_sum_pct,
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
class TrRatioShockSnapshot:
    shock: bool
    ready: bool
    ratio: float | None = None
    tr_fast_pct: float | None = None
    tr_fast: float | None = None
    tr_slow: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False
    direction_ret_sum_pct: float | None = None


class TrRatioShockEngine(_ShockDirectionMixin):
    """Fast/slow True Range EMA ratio shock detector with hysteresis.

    Compared to ATR-ratio, this is a bit more "twitchy" (less smoothing), which can help
    earlier shock detection at the cost of more false positives.
    """

    def __init__(
        self,
        *,
        tr_fast_period: int = 7,
        tr_slow_period: int = 50,
        on_ratio: float = 1.55,
        off_ratio: float = 1.30,
        min_tr_pct: float = 7.0,
        direction_lookback: int = 2,
    ) -> None:
        self._fast_period = max(1, int(tr_fast_period))
        self._slow_period = max(1, int(tr_slow_period))
        self._on_ratio = float(on_ratio)
        self._off_ratio = float(off_ratio)
        self._min_tr_pct = float(min_tr_pct)
        self._shock = False

        self._init_direction_state(direction_lookback)

        self._tr_prev_close: float | None = None
        self._tr_fast_ema: float | None = None
        self._tr_slow_ema: float | None = None
        self._count = 0

        self._ready = False
        self._last_ratio: float | None = None
        self._last_tr_fast_pct: float | None = None
        self._last_tr_fast: float | None = None
        self._last_tr_slow: float | None = None

    def _true_range(self, *, high: float, low: float, close: float) -> float:
        prev_close = self._tr_prev_close
        self._tr_prev_close = float(close)
        if prev_close is None:
            return float(high) - float(low)
        return max(
            float(high) - float(low),
            abs(float(high) - float(prev_close)),
            abs(float(low) - float(prev_close)),
        )

    def _snapshot(self) -> TrRatioShockSnapshot:
        direction, direction_ready, direction_ret_sum_pct = self._direction_state()
        if not bool(self._ready):
            return TrRatioShockSnapshot(
                shock=False,
                ready=False,
                ratio=None,
                tr_fast_pct=None,
                tr_fast=float(self._last_tr_fast) if self._last_tr_fast is not None else None,
                tr_slow=float(self._last_tr_slow) if self._last_tr_slow is not None else None,
                direction=direction,
                direction_ready=direction_ready,
                direction_ret_sum_pct=direction_ret_sum_pct,
            )
        return TrRatioShockSnapshot(
            shock=bool(self._shock),
            ready=True,
            ratio=float(self._last_ratio) if self._last_ratio is not None else None,
            tr_fast_pct=float(self._last_tr_fast_pct) if self._last_tr_fast_pct is not None else None,
            tr_fast=float(self._last_tr_fast) if self._last_tr_fast is not None else None,
            tr_slow=float(self._last_tr_slow) if self._last_tr_slow is not None else None,
            direction=direction,
            direction_ready=direction_ready,
            direction_ret_sum_pct=direction_ret_sum_pct,
        )

    def update(
        self,
        *,
        high: float,
        low: float,
        close: float,
        update_direction: bool = True,
    ) -> TrRatioShockSnapshot:
        if bool(update_direction):
            self._update_direction(float(close))

        tr = float(self._true_range(high=high, low=low, close=close))
        self._count += 1
        self._tr_fast_ema = ema_next(self._tr_fast_ema, tr, self._fast_period)
        self._tr_slow_ema = ema_next(self._tr_slow_ema, tr, self._slow_period)
        self._last_tr_fast = float(self._tr_fast_ema) if self._tr_fast_ema is not None else None
        self._last_tr_slow = float(self._tr_slow_ema) if self._tr_slow_ema is not None else None

        if self._tr_fast_ema is None or self._tr_slow_ema is None or close <= 0:
            self._ready = False
            return self._snapshot()

        ready_bars = max(self._fast_period, self._slow_period)
        if self._count < ready_bars:
            self._ready = False
            return self._snapshot()

        ratio = float(self._tr_fast_ema) / max(float(self._tr_slow_ema), 1e-9)
        tr_fast_pct = float(self._tr_fast_ema) / max(float(close), 1e-9) * 100.0
        self._last_ratio = float(ratio)
        self._last_tr_fast_pct = float(tr_fast_pct)
        self._ready = True

        if not self._shock:
            if ratio >= self._on_ratio and tr_fast_pct >= self._min_tr_pct:
                self._shock = True
        else:
            if ratio <= self._off_ratio:
                self._shock = False

        return self._snapshot()

    def update_direction(self, *, close: float) -> TrRatioShockSnapshot:
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
    direction_ret_sum_pct: float | None = None


class DailyAtrPctShockEngine(_ShockDirectionMixin):
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

        self._init_direction_state(direction_lookback)
        # Cached last-computed values so `update_direction()` can avoid recomputing ATR/TR state.
        self._last_ready = False
        self._last_atr_pct: float | None = None
        self._last_atr: float | None = None
        self._last_tr: float | None = None

    @staticmethod
    def _true_range(high: float, low: float, prev_close: float | None) -> float:
        h = float(high)
        l = float(low)  # noqa: E741 - conventional high/low pairing
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
        direction, direction_ready, direction_ret_sum_pct = self._direction_state()
        return DailyAtrPctShockSnapshot(
            shock=bool(shock),
            ready=bool(ready),
            atr_pct=float(atr_pct) if atr_pct is not None else None,
            atr=float(atr) if atr is not None else None,
            tr=float(tr) if tr is not None else None,
            direction=direction,
            direction_ready=direction_ready,
            direction_ret_sum_pct=direction_ret_sum_pct,
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
            self._last_ready = False
            self._last_atr_pct = None
            self._last_atr = float(self._atr) if self._atr is not None else None
            self._last_tr = None
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
        self._last_ready = bool(ready)
        self._last_atr_pct = float(atr_pct)
        self._last_atr = float(atr_est)
        self._last_tr = float(tr_so_far)

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

    def update_direction(self, *, close: float) -> DailyAtrPctShockSnapshot:
        """Update direction-only state (used when shock_direction_source='signal').

        The daily shock engines are often updated by execution bars for intraday TR/ATR%,
        but direction can be driven by signal-bar closes. This helper avoids duplicating
        the heavy daily update path just to refresh direction.
        """
        self._update_direction(float(close))
        shock = bool(self._shock) if bool(self._last_ready) else False
        return self._snapshot(
            shock=shock,
            ready=bool(self._last_ready),
            atr_pct=self._last_atr_pct,
            atr=self._last_atr,
            tr=self._last_tr,
        )


@dataclass(frozen=True)
class DailyDrawdownShockSnapshot:
    shock: bool
    ready: bool
    drawdown_pct: float | None = None
    peak_close: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False
    direction_ret_sum_pct: float | None = None


class DailyDrawdownShockEngine(_ShockDirectionMixin):
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

        self._init_direction_state(direction_lookback)
        self._last_ready = False
        self._last_drawdown_pct: float | None = None
        self._last_peak_close: float | None = None

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
        direction, direction_ready, direction_ret_sum_pct = self._direction_state()
        return DailyDrawdownShockSnapshot(
            shock=bool(shock),
            ready=bool(ready),
            drawdown_pct=float(drawdown_pct) if drawdown_pct is not None else None,
            peak_close=float(peak_close) if peak_close is not None else None,
            direction=direction,
            direction_ready=direction_ready,
            direction_ret_sum_pct=direction_ret_sum_pct,
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
            self._last_ready = False
            self._last_drawdown_pct = None
            self._last_peak_close = float(peak) if peak is not None else None
            return self._snapshot(shock=False, ready=False, drawdown_pct=None, peak_close=peak)

        peak_eff = max(float(peak), float(close))
        dd_pct = (float(close) / float(peak_eff) - 1.0) * 100.0
        self._last_ready = True
        self._last_drawdown_pct = float(dd_pct)
        self._last_peak_close = float(peak_eff)

        if not self._shock:
            if dd_pct <= self._on:
                self._shock = True
        else:
            if dd_pct >= self._off:
                self._shock = False

        return self._snapshot(shock=bool(self._shock), ready=True, drawdown_pct=dd_pct, peak_close=peak_eff)

    def update_direction(self, *, close: float) -> DailyDrawdownShockSnapshot:
        """Update direction-only state (used when shock_direction_source='signal')."""
        self._update_direction(float(close))
        shock = bool(self._shock) if bool(self._last_ready) else False
        return self._snapshot(
            shock=shock,
            ready=bool(self._last_ready),
            drawdown_pct=self._last_drawdown_pct,
            peak_close=self._last_peak_close if self._last_peak_close is not None else self._rolling_peak,
        )


if TYPE_CHECKING:
    ShockEngine = AtrRatioShockEngine | TrRatioShockEngine | DailyAtrPctShockEngine | DailyDrawdownShockEngine
else:
    ShockEngine = object


# endregion
