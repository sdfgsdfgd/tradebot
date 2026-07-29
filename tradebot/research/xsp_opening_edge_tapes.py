"""Canonical close-time and cross-session tapes for Opening Edge observers."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, time, timezone

from ..backtest.models import Bar
from ..engines.market import (
    is_early_close_day,
    xsp_bar_session_label_et,
    xsp_bar_trading_date,
)
from ..time_utils import NaiveTsModeInput, to_et, to_utc_naive


def normalize_xsp_v2_bars(
    bars: Sequence[Bar],
    *,
    observed_at: datetime,
    naive_ts_mode: NaiveTsModeInput,
) -> tuple[Bar, ...]:
    """Return only close-stamped, causally available UTC-naive bars."""

    observed_utc = observed_at.astimezone(timezone.utc)
    return tuple(
        replace(
            bar,
            ts=to_utc_naive(bar.ts, naive_ts_mode=naive_ts_mode),
        )
        for bar in bars
        if to_utc_naive(
            bar.ts,
            naive_ts_mode=naive_ts_mode,
        ).replace(tzinfo=timezone.utc)
        <= observed_utc
    )


def split_xsp_v2_sessions(
    bars: Sequence[Bar],
) -> tuple[tuple[Bar, ...], tuple[Bar, ...]]:
    """Split a UTC close tape by the crown's XSP session clock."""

    gth: list[Bar] = []
    rth: list[Bar] = []
    for bar in bars:
        session = xsp_bar_session_label_et(
            bar.ts,
            naive_ts_mode="utc",
        )
        if session == "GTH":
            gth.append(bar)
        elif session == "RTH":
            rth.append(bar)
    return tuple(gth), tuple(rth)


def xsp_opening_edge_v2_gth_signal_bars(
    spy_bars: Sequence[Bar],
    xsp_rth_bars: Sequence[Bar],
) -> tuple[Bar, ...]:
    """Project GTH SPY returns from the last exact completed XSP RTH close."""

    spy_by_ts = {row.ts: row for row in spy_bars}
    anchors: dict[date, float] = {}
    for row in xsp_rth_bars:
        day = xsp_bar_trading_date(row.ts)
        expected = time(13, 0) if day and is_early_close_day(day) else time(16, 0)
        spy = spy_by_ts.get(row.ts)
        if (
            day is not None
            and to_et(row.ts, naive_ts_mode="utc").time() == expected
            and spy is not None
            and float(spy.close) > 0.0
        ):
            anchors[day] = float(row.close) / float(spy.close)
    anchor_days = sorted(anchors)
    projected = []
    for row in spy_bars:
        day = xsp_bar_trading_date(row.ts)
        index = bisect_left(anchor_days, day) - 1 if day is not None else -1
        if xsp_bar_session_label_et(row.ts) != "GTH" or index < 0:
            continue
        scale = anchors[anchor_days[index]]
        projected.append(
            Bar(
                row.ts,
                float(row.open) * scale,
                float(row.high) * scale,
                float(row.low) * scale,
                float(row.close) * scale,
                0.0,
            )
        )
    return tuple(projected)
