"""Daily risk-overlay policy and state machine."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

from ..engine import _filters_get, _parse_int
from ..spot.policy import SpotPolicy

# region Risk Overlays (TR%)
@dataclass(frozen=True)
class RiskOverlaySnapshot:
    riskoff: bool
    riskpanic: bool
    riskpop: bool = False
    tr_median_pct: float | None = None
    tr_median_delta_pct: float | None = None
    neg_gap_ratio: float | None = None
    pos_gap_ratio: float | None = None


class TrPctRiskOverlayEngine:
    """Rolling daily TR% overlays used as a simple "risk state" signal.

    This engine is deliberately small and stateful so it can be shared by:
    - backtests (full-history replay), and
    - live mode (incremental updates).
    """

    def __init__(
        self,
        *,
        riskoff_tr_med_pct: float | None,
        riskoff_lookback_days: int,
        riskpanic_tr_med_pct: float | None,
        riskpanic_neg_gap_ratio_min: float | None,
        riskpanic_neg_gap_abs_pct_min: float | None,
        riskpanic_lookback_days: int,
        riskpanic_tr_med_delta_min_pct: float | None,
        riskpanic_tr_med_delta_lookback_days: int,
        riskpop_tr_med_pct: float | None,
        riskpop_pos_gap_ratio_min: float | None,
        riskpop_pos_gap_abs_pct_min: float | None,
        riskpop_lookback_days: int,
        riskpop_tr_med_delta_min_pct: float | None,
        riskpop_tr_med_delta_lookback_days: int,
    ) -> None:
        self._riskoff_tr_med_pct = float(riskoff_tr_med_pct) if riskoff_tr_med_pct is not None else None
        self._riskoff_lookback = max(1, int(riskoff_lookback_days))
        self._riskpanic_tr_med_pct = float(riskpanic_tr_med_pct) if riskpanic_tr_med_pct is not None else None
        self._riskpanic_neg_gap_ratio_min = (
            float(riskpanic_neg_gap_ratio_min) if riskpanic_neg_gap_ratio_min is not None else None
        )
        if self._riskpanic_neg_gap_ratio_min is not None:
            self._riskpanic_neg_gap_ratio_min = float(max(0.0, min(1.0, self._riskpanic_neg_gap_ratio_min)))
        self._riskpanic_neg_gap_abs_pct_min = (
            float(riskpanic_neg_gap_abs_pct_min) if riskpanic_neg_gap_abs_pct_min is not None else None
        )
        if self._riskpanic_neg_gap_abs_pct_min is not None:
            self._riskpanic_neg_gap_abs_pct_min = float(max(0.0, min(1.0, self._riskpanic_neg_gap_abs_pct_min)))
            if self._riskpanic_neg_gap_abs_pct_min <= 0:
                self._riskpanic_neg_gap_abs_pct_min = None
        self._riskpanic_lookback = max(1, int(riskpanic_lookback_days))
        self._riskpanic_tr_med_delta_min_pct = (
            float(riskpanic_tr_med_delta_min_pct) if riskpanic_tr_med_delta_min_pct is not None else None
        )
        self._riskpanic_tr_med_delta_lookback = max(1, int(riskpanic_tr_med_delta_lookback_days))
        self._riskpop_tr_med_pct = float(riskpop_tr_med_pct) if riskpop_tr_med_pct is not None else None
        self._riskpop_pos_gap_ratio_min = (
            float(riskpop_pos_gap_ratio_min) if riskpop_pos_gap_ratio_min is not None else None
        )
        if self._riskpop_pos_gap_ratio_min is not None:
            self._riskpop_pos_gap_ratio_min = float(max(0.0, min(1.0, self._riskpop_pos_gap_ratio_min)))
        self._riskpop_pos_gap_abs_pct_min = (
            float(riskpop_pos_gap_abs_pct_min) if riskpop_pos_gap_abs_pct_min is not None else None
        )
        if self._riskpop_pos_gap_abs_pct_min is not None:
            self._riskpop_pos_gap_abs_pct_min = float(max(0.0, min(1.0, self._riskpop_pos_gap_abs_pct_min)))
            if self._riskpop_pos_gap_abs_pct_min <= 0:
                self._riskpop_pos_gap_abs_pct_min = None
        self._riskpop_lookback = max(1, int(riskpop_lookback_days))
        self._riskpop_tr_med_delta_min_pct = (
            float(riskpop_tr_med_delta_min_pct) if riskpop_tr_med_delta_min_pct is not None else None
        )
        self._riskpop_tr_med_delta_lookback = max(1, int(riskpop_tr_med_delta_lookback_days))

        self._riskoff_tr_hist: deque[float] | None = (
            deque(maxlen=self._riskoff_lookback)
            if self._riskoff_tr_med_pct is not None and self._riskoff_tr_med_pct > 0
            else None
        )
        self._riskpanic_tr_hist: deque[float] | None = None
        self._riskpanic_neg_gap_hist: deque[int] | None = None
        if (
            self._riskpanic_tr_med_pct is not None
            and self._riskpanic_tr_med_pct > 0
            and self._riskpanic_neg_gap_ratio_min is not None
        ):
            self._riskpanic_tr_hist = deque(maxlen=self._riskpanic_lookback)
            self._riskpanic_neg_gap_hist = deque(maxlen=self._riskpanic_lookback)
        # Track TR-median deltas whenever the overlay is enabled (not only when delta gating is active).
        self._riskpanic_tr_med_hist: deque[float] | None = (
            deque(maxlen=max(2, int(self._riskpanic_tr_med_delta_lookback) + 1))
            if self._riskpanic_tr_hist is not None
            else None
        )
        self._riskpop_tr_hist: deque[float] | None = None
        self._riskpop_pos_gap_hist: deque[int] | None = None
        if (
            self._riskpop_tr_med_pct is not None
            and self._riskpop_tr_med_pct > 0
            and self._riskpop_pos_gap_ratio_min is not None
        ):
            self._riskpop_tr_hist = deque(maxlen=self._riskpop_lookback)
            self._riskpop_pos_gap_hist = deque(maxlen=self._riskpop_lookback)
        self._riskpop_tr_med_hist: deque[float] | None = (
            deque(maxlen=max(2, int(self._riskpop_tr_med_delta_lookback) + 1))
            if self._riskpop_tr_hist is not None
            else None
        )

        self._prev_close: float | None = None
        self._cur_day: date | None = None
        self._day_open: float | None = None
        self._day_high: float | None = None
        self._day_low: float | None = None

        self._riskoff_today = False
        self._riskpanic_today = False
        self._riskpop_today = False
        self._tr_median_pct: float | None = None
        self._tr_median_delta_pct: float | None = None
        self._neg_gap_ratio: float | None = None
        self._pos_gap_ratio: float | None = None

    @staticmethod
    def _day_true_range(high: float, low: float, prev_close: float) -> float:
        return max(
            max(0.0, float(high) - float(low)),
            abs(float(high) - float(prev_close)),
            abs(float(low) - float(prev_close)),
        )

    def _compute_today_flags(self) -> None:
        self._tr_median_pct = None
        self._tr_median_delta_pct = None
        self._neg_gap_ratio = None
        self._pos_gap_ratio = None

        self._riskoff_today = False
        tr_med_off: float | None = None
        if self._riskoff_tr_hist is not None and len(self._riskoff_tr_hist) >= self._riskoff_lookback:
            tr_vals = sorted(self._riskoff_tr_hist)
            tr_med_off = float(tr_vals[len(tr_vals) // 2])
            self._riskoff_today = bool(tr_med_off >= float(self._riskoff_tr_med_pct))

        self._riskpanic_today = False
        tr_med_panic: float | None = None
        tr_delta_panic: float | None = None
        if (
            self._riskpanic_tr_hist is not None
            and self._riskpanic_neg_gap_hist is not None
            and self._riskpanic_tr_med_pct is not None
            and self._riskpanic_neg_gap_ratio_min is not None
            and len(self._riskpanic_tr_hist) >= self._riskpanic_lookback
            and len(self._riskpanic_neg_gap_hist) >= self._riskpanic_lookback
        ):
            tr_vals = sorted(self._riskpanic_tr_hist)
            tr_med_panic = float(tr_vals[len(tr_vals) // 2])
            if self._riskpanic_tr_med_hist is not None:
                if len(self._riskpanic_tr_med_hist) >= int(self._riskpanic_tr_med_delta_lookback):
                    prev = list(self._riskpanic_tr_med_hist)[-int(self._riskpanic_tr_med_delta_lookback)]
                    tr_delta_panic = float(tr_med_panic) - float(prev)
                self._riskpanic_tr_med_hist.append(float(tr_med_panic))
            neg_ratio = float(sum(self._riskpanic_neg_gap_hist)) / float(len(self._riskpanic_neg_gap_hist))
            self._neg_gap_ratio = float(neg_ratio)
            ok = bool(tr_med_panic >= float(self._riskpanic_tr_med_pct) and neg_ratio >= float(self._riskpanic_neg_gap_ratio_min))
            if ok and self._riskpanic_tr_med_delta_min_pct is not None:
                ok = bool(tr_delta_panic is not None and tr_delta_panic >= float(self._riskpanic_tr_med_delta_min_pct))
            self._riskpanic_today = bool(ok)

        self._riskpop_today = False
        tr_med_pop: float | None = None
        tr_delta_pop: float | None = None
        if (
            self._riskpop_tr_hist is not None
            and self._riskpop_pos_gap_hist is not None
            and self._riskpop_tr_med_pct is not None
            and self._riskpop_pos_gap_ratio_min is not None
            and len(self._riskpop_tr_hist) >= self._riskpop_lookback
            and len(self._riskpop_pos_gap_hist) >= self._riskpop_lookback
        ):
            tr_vals = sorted(self._riskpop_tr_hist)
            tr_med_pop = float(tr_vals[len(tr_vals) // 2])
            if self._riskpop_tr_med_hist is not None:
                if len(self._riskpop_tr_med_hist) >= int(self._riskpop_tr_med_delta_lookback):
                    prev = list(self._riskpop_tr_med_hist)[-int(self._riskpop_tr_med_delta_lookback)]
                    tr_delta_pop = float(tr_med_pop) - float(prev)
                self._riskpop_tr_med_hist.append(float(tr_med_pop))
            pos_ratio = float(sum(self._riskpop_pos_gap_hist)) / float(len(self._riskpop_pos_gap_hist))
            self._pos_gap_ratio = float(pos_ratio)
            ok = bool(tr_med_pop >= float(self._riskpop_tr_med_pct) and pos_ratio >= float(self._riskpop_pos_gap_ratio_min))
            if ok and self._riskpop_tr_med_delta_min_pct is not None:
                ok = bool(tr_delta_pop is not None and tr_delta_pop >= float(self._riskpop_tr_med_delta_min_pct))
            self._riskpop_today = bool(ok)

        # Expose a single TR-median value for observability. Prefer the stricter overlays.
        if tr_med_panic is not None:
            self._tr_median_pct = float(tr_med_panic)
            self._tr_median_delta_pct = float(tr_delta_panic) if tr_delta_panic is not None else None
        elif tr_med_pop is not None:
            self._tr_median_pct = float(tr_med_pop)
            self._tr_median_delta_pct = float(tr_delta_pop) if tr_delta_pop is not None else None
        else:
            self._tr_median_pct = tr_med_off
            self._tr_median_delta_pct = None

    def update(
        self,
        *,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        is_last_bar: bool,
        trade_day: date | None = None,
    ) -> RiskOverlaySnapshot:
        if self._riskoff_tr_hist is None and self._riskpanic_tr_hist is None and self._riskpop_tr_hist is None:
            return RiskOverlaySnapshot(riskoff=False, riskpanic=False, riskpop=False)

        day = trade_day if isinstance(trade_day, date) else ts.date()
        if self._cur_day != day:
            self._cur_day = day
            self._compute_today_flags()

            self._day_open = float(open)
            self._day_high = float(high)
            self._day_low = float(low)

            # Panic: track whether the *next* session gapped down (computed at session open),
            # but do it after we compute today's risk flags so "today" does not include itself.
            if (
                self._riskpanic_neg_gap_hist is not None
                and self._prev_close is not None
                and float(self._prev_close) > 0
            ):
                gap_pct = (float(self._day_open) - float(self._prev_close)) / float(self._prev_close)
                neg_ok = bool(float(gap_pct) < 0)
                if self._riskpanic_neg_gap_abs_pct_min is not None:
                    neg_ok = bool(float(gap_pct) <= -float(self._riskpanic_neg_gap_abs_pct_min))
                self._riskpanic_neg_gap_hist.append(1 if neg_ok else 0)
                if self._riskpop_pos_gap_hist is not None:
                    pos_ok = bool(float(gap_pct) > 0)
                    if self._riskpop_pos_gap_abs_pct_min is not None:
                        pos_ok = bool(float(gap_pct) >= float(self._riskpop_pos_gap_abs_pct_min))
                    self._riskpop_pos_gap_hist.append(1 if pos_ok else 0)
            elif (
                self._riskpop_pos_gap_hist is not None
                and self._prev_close is not None
                and float(self._prev_close) > 0
            ):
                gap_pct = (float(self._day_open) - float(self._prev_close)) / float(self._prev_close)
                pos_ok = bool(float(gap_pct) > 0)
                if self._riskpop_pos_gap_abs_pct_min is not None:
                    pos_ok = bool(float(gap_pct) >= float(self._riskpop_pos_gap_abs_pct_min))
                self._riskpop_pos_gap_hist.append(1 if pos_ok else 0)

        if self._day_high is not None:
            self._day_high = max(float(self._day_high), float(high))
        if self._day_low is not None:
            self._day_low = min(float(self._day_low), float(low))

        if bool(is_last_bar):
            if (
                self._day_high is not None
                and self._day_low is not None
                and self._prev_close is not None
                and float(self._prev_close) > 0
            ):
                day_tr = self._day_true_range(float(self._day_high), float(self._day_low), float(self._prev_close))
                tr_pct = float(day_tr) / float(self._prev_close) * 100.0
                if self._riskoff_tr_hist is not None:
                    self._riskoff_tr_hist.append(float(tr_pct))
                if self._riskpanic_tr_hist is not None:
                    self._riskpanic_tr_hist.append(float(tr_pct))
                if self._riskpop_tr_hist is not None:
                    self._riskpop_tr_hist.append(float(tr_pct))
            self._prev_close = float(close)

        return RiskOverlaySnapshot(
            riskoff=bool(self._riskoff_today),
            riskpanic=bool(self._riskpanic_today),
            riskpop=bool(self._riskpop_today),
            tr_median_pct=float(self._tr_median_pct) if self._tr_median_pct is not None else None,
            tr_median_delta_pct=float(self._tr_median_delta_pct) if self._tr_median_delta_pct is not None else None,
            neg_gap_ratio=float(self._neg_gap_ratio) if self._neg_gap_ratio is not None else None,
            pos_gap_ratio=float(self._pos_gap_ratio) if self._pos_gap_ratio is not None else None,
        )


def build_tr_pct_risk_overlay_engine(
    filters: Mapping[str, object] | object | None,
) -> TrPctRiskOverlayEngine | None:
    if filters is None:
        return None

    raw_riskoff = _filters_get(filters, "riskoff_tr5_med_pct")
    try:
        riskoff_tr_med = float(raw_riskoff) if raw_riskoff is not None else None
    except (TypeError, ValueError):
        riskoff_tr_med = None
    if riskoff_tr_med is not None and riskoff_tr_med <= 0:
        riskoff_tr_med = None
    riskoff_lb = _parse_int(_filters_get(filters, "riskoff_tr5_lookback_days"), default=5, min_value=1)

    raw_panic = _filters_get(filters, "riskpanic_tr5_med_pct")
    try:
        riskpanic_tr_med = float(raw_panic) if raw_panic is not None else None
    except (TypeError, ValueError):
        riskpanic_tr_med = None
    if riskpanic_tr_med is not None and riskpanic_tr_med <= 0:
        riskpanic_tr_med = None

    raw_gap = _filters_get(filters, "riskpanic_neg_gap_ratio_min")
    try:
        riskpanic_gap_ratio = float(raw_gap) if raw_gap is not None else None
    except (TypeError, ValueError):
        riskpanic_gap_ratio = None
    if riskpanic_gap_ratio is not None:
        riskpanic_gap_ratio = float(max(0.0, min(1.0, riskpanic_gap_ratio)))

    raw_gap_abs = _filters_get(filters, "riskpanic_neg_gap_abs_pct_min")
    try:
        riskpanic_gap_abs = float(raw_gap_abs) if raw_gap_abs is not None else None
    except (TypeError, ValueError):
        riskpanic_gap_abs = None
    if riskpanic_gap_abs is not None:
        riskpanic_gap_abs = float(max(0.0, min(1.0, riskpanic_gap_abs)))
        if riskpanic_gap_abs <= 0:
            riskpanic_gap_abs = None

    riskpanic_lb = _parse_int(_filters_get(filters, "riskpanic_lookback_days"), default=5, min_value=1)

    raw_panic_delta = _filters_get(filters, "riskpanic_tr5_med_delta_min_pct")
    try:
        riskpanic_tr_delta_min = float(raw_panic_delta) if raw_panic_delta is not None else None
    except (TypeError, ValueError):
        riskpanic_tr_delta_min = None
    riskpanic_tr_delta_lb = _parse_int(
        _filters_get(filters, "riskpanic_tr5_med_delta_lookback_days"),
        default=1,
        min_value=1,
    )

    raw_pop = _filters_get(filters, "riskpop_tr5_med_pct")
    try:
        riskpop_tr_med = float(raw_pop) if raw_pop is not None else None
    except (TypeError, ValueError):
        riskpop_tr_med = None
    if riskpop_tr_med is not None and riskpop_tr_med <= 0:
        riskpop_tr_med = None

    raw_pos = _filters_get(filters, "riskpop_pos_gap_ratio_min")
    try:
        riskpop_pos_gap_ratio = float(raw_pos) if raw_pos is not None else None
    except (TypeError, ValueError):
        riskpop_pos_gap_ratio = None
    if riskpop_pos_gap_ratio is not None:
        riskpop_pos_gap_ratio = float(max(0.0, min(1.0, riskpop_pos_gap_ratio)))

    raw_pos_abs = _filters_get(filters, "riskpop_pos_gap_abs_pct_min")
    try:
        riskpop_pos_gap_abs = float(raw_pos_abs) if raw_pos_abs is not None else None
    except (TypeError, ValueError):
        riskpop_pos_gap_abs = None
    if riskpop_pos_gap_abs is not None:
        riskpop_pos_gap_abs = float(max(0.0, min(1.0, riskpop_pos_gap_abs)))
        if riskpop_pos_gap_abs <= 0:
            riskpop_pos_gap_abs = None

    riskpop_lb = _parse_int(_filters_get(filters, "riskpop_lookback_days"), default=5, min_value=1)

    raw_pop_delta = _filters_get(filters, "riskpop_tr5_med_delta_min_pct")
    try:
        riskpop_tr_delta_min = float(raw_pop_delta) if raw_pop_delta is not None else None
    except (TypeError, ValueError):
        riskpop_tr_delta_min = None
    riskpop_tr_delta_lb = _parse_int(
        _filters_get(filters, "riskpop_tr5_med_delta_lookback_days"),
        default=1,
        min_value=1,
    )

    enabled = (
        bool(riskoff_tr_med is not None)
        or bool(riskpanic_tr_med is not None and riskpanic_gap_ratio is not None)
        or bool(riskpop_tr_med is not None and riskpop_pos_gap_ratio is not None)
    )
    if not enabled:
        return None

    return TrPctRiskOverlayEngine(
        riskoff_tr_med_pct=riskoff_tr_med,
        riskoff_lookback_days=int(riskoff_lb),
        riskpanic_tr_med_pct=riskpanic_tr_med,
        riskpanic_neg_gap_ratio_min=riskpanic_gap_ratio,
        riskpanic_neg_gap_abs_pct_min=riskpanic_gap_abs,
        riskpanic_lookback_days=int(riskpanic_lb),
        riskpanic_tr_med_delta_min_pct=riskpanic_tr_delta_min,
        riskpanic_tr_med_delta_lookback_days=int(riskpanic_tr_delta_lb),
        riskpop_tr_med_pct=riskpop_tr_med,
        riskpop_pos_gap_ratio_min=riskpop_pos_gap_ratio,
        riskpop_pos_gap_abs_pct_min=riskpop_pos_gap_abs,
        riskpop_lookback_days=int(riskpop_lb),
        riskpop_tr_med_delta_min_pct=riskpop_tr_delta_min,
        riskpop_tr_med_delta_lookback_days=int(riskpop_tr_delta_lb),
    )


def risk_overlay_policy_from_filters(
    filters: Mapping[str, object] | object | None,
) -> tuple[str, float, float, float, float, float, float]:
    """Return (riskoff_mode, riskoff_long_factor, riskoff_short_factor, riskpanic_long_factor, riskpanic_short_factor, riskpop_long_factor, riskpop_short_factor)."""
    return SpotPolicy.risk_overlay_policy(filters)


# endregion
