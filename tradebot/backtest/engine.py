"""Backtest runner for synthetic options strategies."""
from __future__ import annotations

import math
import hashlib
import os
import shutil
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import NamedTuple, Union

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional acceleration dependency
    _np = None

from .config import ConfigBundle, SpotLegConfig
from .calibration import ensure_calibration, load_calibration
from .data import IBKRHistoricalData, ContractMeta
from .models import Bar, EquityPoint, OptionLeg, OptionTrade, SpotTrade, SummaryStats
from .spot_context import SpotBarRequirement, load_spot_context_bars
from .strategy import CreditSpreadStrategy, TradeSpec
from ..series import BarSeries, bars_list
from ..series_cache import series_cache_service
from ..spot.lifecycle import (
    apply_regime_gate,
    decide_flat_position_intent,
    decide_open_position_intent,
    decide_pending_next_open,
    deferred_entry_plan as lifecycle_deferred_entry_plan,
    fill_due_ts as lifecycle_fill_due_ts,
    entry_capacity_ok as lifecycle_entry_capacity_ok,
    flip_exit_hit,
    flip_exit_gate_blocked as lifecycle_flip_exit_gate_blocked,
    next_open_due_ts as lifecycle_next_open_due_ts,
    next_open_entry_allowed as lifecycle_next_open_entry_allowed,
    signal_filters_ok as lifecycle_signal_filters_ok,
)
from ..spot.fill_modes import (
    SPOT_FILL_MODE_CLOSE,
    SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    normalize_spot_fill_mode,
    spot_fill_mode_is_deferred,
)
from ..spot.graph import SpotPolicyGraph, spot_dynamic_flip_hold_bars
from ..spot.scenario import lifecycle_trace_row, why_not_exit_resize_report, write_rows_csv
from .synth import IVSurfaceParams, black_76, black_scholes, iv_atm, iv_for_strike, mid_edge_quote
from ..utils.date_utils import business_days_until
from ..engine import (
    EmaDecisionEngine,
    EmaDecisionSnapshot,
    OrbDecisionEngine,
    RiskOverlaySnapshot,
    SupertrendEngine,
    _trade_date,
    _trade_hour_et,
    _trade_weekday,
    _ts_to_et,
    annualized_ewma_vol,
    build_shock_engine,
    build_tr_pct_risk_overlay_engine,
    cooldown_ok_by_index,
    normalize_shock_detector,
    normalize_shock_direction_source,
    normalize_shock_gate_mode,
    normalize_spot_entry_signal,
    parse_time_hhmm,
    realized_vol_from_closes,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
    risk_overlay_policy_from_filters,
    spot_apply_branch_size_mult,
    spot_branch_size_mult,
    spot_calc_signed_qty_with_trace,
    spot_hit_profit,
    spot_hit_stop,
    spot_pending_entry_should_cancel,
    spot_exec_price as _spot_exec_price,
    spot_intrabar_exit,
    spot_intrabar_worst_ref,
    spot_mark_price as _spot_mark_price,
    spot_policy_config_view,
    spot_runtime_spec_view,
    spot_profit_level,
    spot_resolve_entry_action_qty,
    spot_riskoff_end_hour,
    spot_scale_exit_pcts,
    spot_shock_exit_pct_multipliers,
    spot_stop_level,
)
from ..signals import ema_next, ema_periods as _ema_periods_shared


# region Internal Caches (Spot Backtest)
_SERIES_CACHE = series_cache_service()
_SPOT_SERIES_CORE_NAMESPACE = "spot.series.core"
_SPOT_SERIES_PACK_NAMESPACE = "spot.series.pack"
_SPOT_SERIES_PACK_MMAP_NAMESPACE = "spot.series.pack.mmap"
_SPOT_EXEC_ALIGNMENT_NAMESPACE = "spot.exec.alignment"
_SPOT_DAILY_OHLC_NAMESPACE = "spot.daily.ohlc"
_SPOT_RISK_OVERLAY_DAY_NAMESPACE = "spot.risk.overlay.day"
_SPOT_SHOCK_SERIES_NAMESPACE = "spot.shock.series"
_SPOT_DD_TREES_NAMESPACE = "spot.dd.trees"
_SPOT_STOP_TREE_NAMESPACE = "spot.stop.tree"
_SPOT_PROFIT_TREE_NAMESPACE = "spot.profit.tree"
_SPOT_SIGNAL_SERIES_NAMESPACE = "spot.signal.series"
_SPOT_FLIP_TREE_NAMESPACE = "spot.flip.tree"
_SPOT_FLIP_NEXT_SIG_NAMESPACE = "spot.flip.next_sig"
_SPOT_RV_SERIES_NAMESPACE = "spot.rv.series"
_SPOT_VOLUME_EMA_SERIES_NAMESPACE = "spot.volume_ema.series"
_SPOT_TICK_GATE_SERIES_NAMESPACE = "spot.tick_gate.series"


class _SpotExecAlignment(NamedTuple):
    sig_idx_by_exec_idx: list[int]  # -1 when exec bar isn't a signal close
    exec_idx_by_sig_idx: list[int]  # -1 when signal ts isn't present in exec bars
    signal_exec_indices: list[int]  # exec indices in ascending order that are signal closes
    signal_sig_indices: list[int]  # matching signal indices for `signal_exec_indices`


def _spot_exec_alignment(signal_bars: list[Bar], exec_bars: list[Bar]) -> _SpotExecAlignment:
    """Return stable alignment maps between signal-bar closes and execution bars.

    Cached by (id(signal_bars), id(exec_bars)) so sweeps don't rebuild dicts per config.
    """
    key = (id(signal_bars), id(exec_bars))
    cached = _SERIES_CACHE.get(namespace=_SPOT_EXEC_ALIGNMENT_NAMESPACE, key=key)
    if isinstance(cached, _SpotExecAlignment):
        return cached

    sig_idx_by_exec_idx = [-1] * len(exec_bars)
    exec_idx_by_sig_idx = [-1] * len(signal_bars)
    signal_exec_indices: list[int] = []
    signal_sig_indices: list[int] = []

    sig_i = 0
    for exec_i, ex in enumerate(exec_bars):
        ts = ex.ts
        while sig_i < len(signal_bars) and signal_bars[sig_i].ts < ts:
            sig_i += 1
        if sig_i < len(signal_bars) and signal_bars[sig_i].ts == ts:
            sig_idx_by_exec_idx[exec_i] = sig_i
            exec_idx_by_sig_idx[sig_i] = exec_i
            signal_exec_indices.append(exec_i)
            signal_sig_indices.append(sig_i)
            sig_i += 1

    out = _SpotExecAlignment(
        sig_idx_by_exec_idx=sig_idx_by_exec_idx,
        exec_idx_by_sig_idx=exec_idx_by_sig_idx,
        signal_exec_indices=signal_exec_indices,
        signal_sig_indices=signal_sig_indices,
    )
    _SERIES_CACHE.set(namespace=_SPOT_EXEC_ALIGNMENT_NAMESPACE, key=key, value=out)
    return out


class _SpotDailyOhlc(NamedTuple):
    day: date
    ts: datetime
    open: float
    high: float
    low: float
    close: float


_FAST_PATH_RATSV_BLOCK_KEYS: tuple[str, ...] = (
    "ratsv_rank_min",
    "ratsv_tr_ratio_min",
    "ratsv_slope_med_min_pct",
    "ratsv_slope_vel_min_pct",
    "ratsv_slope_slow_window_bars",
    "ratsv_slope_med_slow_min_pct",
    "ratsv_slope_vel_slow_min_pct",
    "ratsv_slope_vel_consistency_bars",
    "ratsv_slope_vel_consistency_min",
    "ratsv_cross_age_max_bars",
    "ratsv_branch_a_rank_min",
    "ratsv_branch_a_tr_ratio_min",
    "ratsv_branch_a_slope_med_min_pct",
    "ratsv_branch_a_slope_vel_min_pct",
    "ratsv_branch_a_slope_med_slow_min_pct",
    "ratsv_branch_a_slope_vel_slow_min_pct",
    "ratsv_branch_a_slope_vel_consistency_bars",
    "ratsv_branch_a_slope_vel_consistency_min",
    "ratsv_branch_a_cross_age_max_bars",
    "ratsv_branch_b_rank_min",
    "ratsv_branch_b_tr_ratio_min",
    "ratsv_branch_b_slope_med_min_pct",
    "ratsv_branch_b_slope_vel_min_pct",
    "ratsv_branch_b_slope_med_slow_min_pct",
    "ratsv_branch_b_slope_vel_slow_min_pct",
    "ratsv_branch_b_slope_vel_consistency_bars",
    "ratsv_branch_b_slope_vel_consistency_min",
    "ratsv_branch_b_cross_age_max_bars",
    "ratsv_probe_cancel_max_bars",
    "ratsv_probe_cancel_slope_adverse_min_pct",
    "ratsv_probe_cancel_tr_ratio_min",
    "ratsv_adverse_release_min_hold_bars",
    "ratsv_adverse_release_slope_adverse_min_pct",
    "ratsv_adverse_release_tr_ratio_min",
)
_SPOT_SERIES_CORE_CACHE_VERSION = "spot-series-core-v1"
_SPOT_SERIES_PACK_CACHE_VERSION = "spot-series-pack-v1"
_SPOT_SERIES_PACK_MMAP_CACHE_VERSION = "spot-series-pack-mmap-v1"


BarSeriesInput = Union[list[Bar], BarSeries[Bar]]


def _bars_input_list(value: BarSeriesInput) -> list[Bar]:
    return bars_list(value)


def _bars_input_optional_list(value: BarSeriesInput | None) -> list[Bar] | None:
    if value is None:
        return None
    return bars_list(value)


class _SpotShockSeries(NamedTuple):
    shock_by_exec_idx: list[bool | None]
    shock_atr_pct_by_exec_idx: list[float | None]
    shock_dir_by_exec_idx: list[str | None]
    shock_by_sig_idx: list[bool | None]
    shock_atr_pct_by_sig_idx: list[float | None]
    shock_dir_by_sig_idx: list[str | None]


class _SpotDdTreeLong:
    __slots__ = ("n", "size", "_max_close", "_min_low", "_max_dd")

    def __init__(self, *, close_vals: list[float], low_vals: list[float]) -> None:
        n = len(close_vals)
        self.n = int(n)
        size = 1 << ((n - 1).bit_length()) if n > 0 else 1
        self.size = int(size)
        max_close = [float("-inf")] * (2 * size)
        min_low = [float("inf")] * (2 * size)
        max_dd = [0.0] * (2 * size)
        base = size
        for i in range(n):
            max_close[base + i] = float(close_vals[i])
            min_low[base + i] = float(low_vals[i])
        for i in range(size - 1, 0, -1):
            left = i * 2
            right = left + 1
            lc = max_close[left]
            rc = max_close[right]
            ll = min_low[left]
            rl = min_low[right]
            max_close[i] = lc if lc >= rc else rc
            min_low[i] = ll if ll <= rl else rl
            cross = lc - rl
            dd = max_dd[left]
            if max_dd[right] > dd:
                dd = max_dd[right]
            if cross > dd:
                dd = cross
            max_dd[i] = dd
        self._max_close = max_close
        self._min_low = min_low
        self._max_dd = max_dd

    @staticmethod
    def _merge(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        a_mc, a_ml, a_dd = a
        b_mc, b_ml, b_dd = b
        mc = a_mc if a_mc >= b_mc else b_mc
        ml = a_ml if a_ml <= b_ml else b_ml
        cross = a_mc - b_ml
        dd = a_dd
        if b_dd > dd:
            dd = b_dd
        if cross > dd:
            dd = cross
        return mc, ml, dd

    def query(self, l: int, r: int) -> tuple[float, float, float]:
        """Return (max_close, min_low, max_close_to_low_dd) over [l, r)."""
        if l < 0:
            l = 0
        if r > self.n:
            r = self.n
        if l >= r:
            return float("-inf"), float("inf"), 0.0

        size = self.size
        max_close = self._max_close
        min_low = self._min_low
        max_dd = self._max_dd

        left_res = (float("-inf"), float("inf"), 0.0)
        right_res = (float("-inf"), float("inf"), 0.0)
        l += size
        r += size
        while l < r:
            if l & 1:
                left_res = self._merge(
                    left_res,
                    (max_close[l], min_low[l], max_dd[l]),
                )
                l += 1
            if r & 1:
                r -= 1
                right_res = self._merge(
                    (max_close[r], min_low[r], max_dd[r]),
                    right_res,
                )
            l //= 2
            r //= 2
        return self._merge(left_res, right_res)


class _SpotDdTreeShort:
    __slots__ = ("n", "size", "_min_close", "_max_high", "_max_dd")

    def __init__(self, *, close_vals: list[float], high_vals: list[float]) -> None:
        n = len(close_vals)
        self.n = int(n)
        size = 1 << ((n - 1).bit_length()) if n > 0 else 1
        self.size = int(size)
        min_close = [float("inf")] * (2 * size)
        max_high = [float("-inf")] * (2 * size)
        max_dd = [0.0] * (2 * size)
        base = size
        for i in range(n):
            min_close[base + i] = float(close_vals[i])
            max_high[base + i] = float(high_vals[i])
        for i in range(size - 1, 0, -1):
            left = i * 2
            right = left + 1
            lc = min_close[left]
            rc = min_close[right]
            lh = max_high[left]
            rh = max_high[right]
            min_close[i] = lc if lc <= rc else rc
            max_high[i] = lh if lh >= rh else rh
            cross = rh - lc
            dd = max_dd[left]
            if max_dd[right] > dd:
                dd = max_dd[right]
            if cross > dd:
                dd = cross
            max_dd[i] = dd
        self._min_close = min_close
        self._max_high = max_high
        self._max_dd = max_dd

    @staticmethod
    def _merge(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        a_mc, a_mh, a_dd = a
        b_mc, b_mh, b_dd = b
        mc = a_mc if a_mc <= b_mc else b_mc
        mh = a_mh if a_mh >= b_mh else b_mh
        cross = b_mh - a_mc
        dd = a_dd
        if b_dd > dd:
            dd = b_dd
        if cross > dd:
            dd = cross
        return mc, mh, dd

    def query(self, l: int, r: int) -> tuple[float, float, float]:
        """Return (min_close, max_high, max_high_minus_earlier_min_close) over [l, r)."""
        if l < 0:
            l = 0
        if r > self.n:
            r = self.n
        if l >= r:
            return float("inf"), float("-inf"), 0.0

        size = self.size
        min_close = self._min_close
        max_high = self._max_high
        max_dd = self._max_dd

        left_res = (float("inf"), float("-inf"), 0.0)
        right_res = (float("inf"), float("-inf"), 0.0)
        l += size
        r += size
        while l < r:
            if l & 1:
                left_res = self._merge(
                    left_res,
                    (min_close[l], max_high[l], max_dd[l]),
                )
                l += 1
            if r & 1:
                r -= 1
                right_res = self._merge(
                    (min_close[r], max_high[r], max_dd[r]),
                    right_res,
                )
            l //= 2
            r //= 2
        return self._merge(left_res, right_res)


class _MinFirstLeqTree:
    __slots__ = ("n", "size", "_mins")

    def __init__(self, values: list[float]) -> None:
        n = len(values)
        self.n = int(n)
        size = 1 << ((n - 1).bit_length()) if n > 0 else 1
        self.size = int(size)
        mins = [float("inf")] * (2 * size)
        base = size
        for i in range(n):
            mins[base + i] = float(values[i])
        for i in range(size - 1, 0, -1):
            left = i * 2
            right = left + 1
            mins[i] = mins[left] if mins[left] <= mins[right] else mins[right]
        self._mins = mins

    def find_first_leq(self, *, start: int, threshold: float) -> int | None:
        if start < 0:
            start = 0
        if start >= self.n:
            return None

        mins = self._mins

        def _search(node: int, l: int, r: int) -> int | None:
            if r <= start:
                return None
            if mins[node] > threshold:
                return None
            if node >= self.size:
                idx = l
                return idx if idx < self.n else None
            mid = (l + r) // 2
            left = node * 2
            res = _search(left, l, mid)
            if res is not None:
                return res
            return _search(left + 1, mid, r)

        return _search(1, 0, self.size)


class _MaxFirstGeTree:
    __slots__ = ("n", "size", "_maxs")

    def __init__(self, values: list[float]) -> None:
        n = len(values)
        self.n = int(n)
        size = 1 << ((n - 1).bit_length()) if n > 0 else 1
        self.size = int(size)
        maxs = [float("-inf")] * (2 * size)
        base = size
        for i in range(n):
            maxs[base + i] = float(values[i])
        for i in range(size - 1, 0, -1):
            left = i * 2
            right = left + 1
            maxs[i] = maxs[left] if maxs[left] >= maxs[right] else maxs[right]
        self._maxs = maxs

    def find_first_ge(self, *, start: int, threshold: float) -> int | None:
        if start < 0:
            start = 0
        if start >= self.n:
            return None

        maxs = self._maxs

        def _search(node: int, l: int, r: int) -> int | None:
            if r <= start:
                return None
            if maxs[node] < threshold:
                return None
            if node >= self.size:
                idx = l
                return idx if idx < self.n else None
            mid = (l + r) // 2
            left = node * 2
            res = _search(left, l, mid)
            if res is not None:
                return res
            return _search(left + 1, mid, r)

        return _search(1, 0, self.size)


class _MaxFirstGtTree:
    __slots__ = ("n", "size", "_maxs")

    def __init__(self, values: list[float]) -> None:
        n = len(values)
        self.n = int(n)
        size = 1 << ((n - 1).bit_length()) if n > 0 else 1
        self.size = int(size)
        maxs = [float("-inf")] * (2 * size)
        base = size
        for i in range(n):
            maxs[base + i] = float(values[i])
        for i in range(size - 1, 0, -1):
            left = i * 2
            right = left + 1
            maxs[i] = maxs[left] if maxs[left] >= maxs[right] else maxs[right]
        self._maxs = maxs

    def find_first_gt(self, *, start: int, threshold: float) -> int | None:
        if start < 0:
            start = 0
        if start >= self.n:
            return None

        maxs = self._maxs

        def _search(node: int, l: int, r: int) -> int | None:
            if r <= start:
                return None
            if maxs[node] <= threshold:
                return None
            if node >= self.size:
                idx = l
                return idx if idx < self.n else None
            mid = (l + r) // 2
            left = node * 2
            res = _search(left, l, mid)
            if res is not None:
                return res
            return _search(left + 1, mid, r)

        return _search(1, 0, self.size)


class _SpotSignalSeries(NamedTuple):
    signal_by_sig_idx: list[EmaDecisionSnapshot | None]
    bars_in_day_by_sig_idx: list[int]
    entry_dir_by_sig_idx: list[str | None]
    entry_branch_by_sig_idx: list[str | None]


class _SpotRvSeries(NamedTuple):
    rv_by_sig_idx: list[float | None]


class _SpotVolumeEmaSeries(NamedTuple):
    volume_ema_by_sig_idx: list[float | None]
    volume_ema_ready_by_sig_idx: list[bool]


class _SpotTickGateSeries(NamedTuple):
    tick_ready_by_sig_idx: list[bool]
    tick_dir_by_sig_idx: list[str | None]


class _SpotSeriesCorePack(NamedTuple):
    align: _SpotExecAlignment
    signal_series: _SpotSignalSeries
    tick_series: _SpotTickGateSeries | None


class _SpotSeriesPack(NamedTuple):
    core: _SpotSeriesCorePack
    shock_series: _SpotShockSeries
    risk_by_day: dict[date, object]
    rv_series: _SpotRvSeries | None
    volume_series: _SpotVolumeEmaSeries | None


class _SpotExecProfile(NamedTuple):
    entry_fill_mode: str
    flip_fill_mode: str
    exit_mode: str
    intrabar_exits: bool
    close_eod: bool
    spread: float
    commission_per_share: float
    commission_min: float
    slippage_per_share: float
    mark_to_market: str
    drawdown_mode: str


def _spot_bars_signature(bars: list[Bar] | None) -> tuple[object, ...]:
    if not bars:
        return (0, None, None, None, None)
    first = bars[0]
    last = bars[-1]
    return (
        int(len(bars)),
        first.ts.isoformat(),
        last.ts.isoformat(),
        round(float(first.open), 8),
        round(float(last.close), 8),
    )


def _spot_signal_series_signature(*, cfg: ConfigBundle) -> tuple[object, ...]:
    strat = cfg.strategy
    return (
        str(cfg.backtest.bar_size),
        bool(cfg.backtest.use_rth),
        str(getattr(strat, "entry_signal", "ema") or "ema"),
        str(getattr(strat, "ema_preset", "") or ""),
        str(getattr(strat, "ema_entry_mode", "") or ""),
        int(getattr(strat, "entry_confirm_bars", 0) or 0),
        bool(getattr(strat, "spot_dual_branch_enabled", False)),
        str(getattr(strat, "spot_dual_branch_priority", "b_then_a") or "b_then_a"),
        str(getattr(strat, "spot_branch_a_ema_preset", "") or ""),
        int(getattr(strat, "spot_branch_a_entry_confirm_bars", 0) or 0)
        if getattr(strat, "spot_branch_a_entry_confirm_bars", None) is not None
        else None,
        float(getattr(strat, "spot_branch_a_min_signed_slope_pct", 0.0) or 0.0)
        if getattr(strat, "spot_branch_a_min_signed_slope_pct", None) is not None
        else None,
        float(getattr(strat, "spot_branch_a_max_signed_slope_pct", 0.0) or 0.0)
        if getattr(strat, "spot_branch_a_max_signed_slope_pct", None) is not None
        else None,
        str(getattr(strat, "spot_branch_b_ema_preset", "") or ""),
        int(getattr(strat, "spot_branch_b_entry_confirm_bars", 0) or 0)
        if getattr(strat, "spot_branch_b_entry_confirm_bars", None) is not None
        else None,
        float(getattr(strat, "spot_branch_b_min_signed_slope_pct", 0.0) or 0.0)
        if getattr(strat, "spot_branch_b_min_signed_slope_pct", None) is not None
        else None,
        float(getattr(strat, "spot_branch_b_max_signed_slope_pct", 0.0) or 0.0)
        if getattr(strat, "spot_branch_b_max_signed_slope_pct", None) is not None
        else None,
        str(getattr(strat, "regime_mode", "ema") or "ema"),
        str(getattr(strat, "regime_ema_preset", "") or ""),
        str(getattr(strat, "regime_bar_size", "") or ""),
        int(getattr(strat, "supertrend_atr_period", 10) or 10),
        float(getattr(strat, "supertrend_multiplier", 3.0) or 3.0),
        str(getattr(strat, "supertrend_source", "hl2") or "hl2"),
        str(getattr(strat, "regime2_mode", "off") or "off"),
        str(getattr(strat, "regime2_ema_preset", "") or ""),
        str(getattr(strat, "regime2_bar_size", "") or ""),
        int(getattr(strat, "regime2_supertrend_atr_period", 10) or 10),
        float(getattr(strat, "regime2_supertrend_multiplier", 3.0) or 3.0),
        str(getattr(strat, "regime2_supertrend_source", "hl2") or "hl2"),
    )


def _spot_signature_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(k), _spot_signature_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, set):
        vals = [_spot_signature_value(v) for v in value]
        vals.sort(key=repr)
        return tuple(vals)
    if isinstance(value, (list, tuple)):
        return tuple(_spot_signature_value(v) for v in value)
    if isinstance(value, float):
        return round(float(value), 12)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _spot_filter_signature(
    filters: object | None,
    *,
    include_prefixes: tuple[str, ...] = (),
    include_keys: tuple[str, ...] = (),
) -> tuple[object, ...]:
    if filters is None:
        return ("off",)
    names = {str(k) for k in include_keys if str(k).strip()}
    try:
        for raw_name in vars(filters).keys():
            name = str(raw_name)
            if any(name.startswith(prefix) for prefix in include_prefixes):
                names.add(name)
    except Exception:
        pass
    if not names:
        return ("off",)
    out: list[tuple[str, object]] = []
    for name in sorted(names):
        try:
            raw_val = getattr(filters, name, None)
        except Exception:
            raw_val = None
        out.append((str(name), _spot_signature_value(raw_val)))
    return tuple(out)


def _spot_tick_gate_settings(
    strategy: object,
) -> tuple[str, str, str, int, int, float, float, int]:
    tick_mode = str(getattr(strategy, "tick_gate_mode", "off") or "off").strip().lower()
    if tick_mode not in ("off", "raschke"):
        tick_mode = "off"
    tick_neutral_policy = str(getattr(strategy, "tick_neutral_policy", "allow") or "allow").strip().lower()
    if tick_neutral_policy not in ("allow", "block"):
        tick_neutral_policy = "allow"
    tick_direction_policy = str(getattr(strategy, "tick_direction_policy", "both") or "both").strip().lower()
    if tick_direction_policy not in ("both", "wide_only"):
        tick_direction_policy = "both"
    tick_ma_period = max(1, int(getattr(strategy, "tick_band_ma_period", 10) or 10))
    tick_z_lookback = max(5, int(getattr(strategy, "tick_width_z_lookback", 252) or 252))
    tick_z_enter = float(getattr(strategy, "tick_width_z_enter", 1.0) or 1.0)
    tick_z_exit = max(0.0, float(getattr(strategy, "tick_width_z_exit", 0.5) or 0.5))
    tick_slope_lookback = max(1, int(getattr(strategy, "tick_width_slope_lookback", 3) or 3))
    return (
        str(tick_mode),
        str(tick_neutral_policy),
        str(tick_direction_policy),
        int(tick_ma_period),
        int(tick_z_lookback),
        float(tick_z_enter),
        float(tick_z_exit),
        int(tick_slope_lookback),
    )


def _spot_apply_tick_gate_to_entry_dir(
    *,
    entry_dir: str | None,
    tick_ready: bool,
    tick_dir: str | None,
    tick_neutral_policy: str,
) -> str | None:
    if not bool(tick_ready):
        return None if str(tick_neutral_policy) == "block" else entry_dir
    if tick_dir not in ("up", "down"):
        return None if str(tick_neutral_policy) == "block" else entry_dir
    if entry_dir is not None and str(entry_dir) != str(tick_dir):
        return None
    return entry_dir


def _spot_tick_gate_series(
    *,
    signal_bars: list[Bar],
    tick_bars: list[Bar] | None,
    strategy: object,
) -> _SpotTickGateSeries | None:
    (
        tick_mode,
        _tick_neutral_policy,
        tick_direction_policy,
        tick_ma_period,
        tick_z_lookback,
        tick_z_enter,
        tick_z_exit,
        tick_slope_lookback,
    ) = _spot_tick_gate_settings(strategy)
    if str(tick_mode) == "off" or not tick_bars:
        return None

    key = (
        id(signal_bars),
        id(tick_bars),
        str(tick_mode),
        str(tick_direction_policy),
        int(tick_ma_period),
        int(tick_z_lookback),
        float(tick_z_enter),
        float(tick_z_exit),
        int(tick_slope_lookback),
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_TICK_GATE_SERIES_NAMESPACE, key=key)
    if isinstance(cached, _SpotTickGateSeries):
        return cached

    tick_idx = 0
    tick_state = "neutral"
    tick_dir: str | None = None
    tick_ready = False
    tick_highs: deque[float] = deque(maxlen=tick_ma_period)
    tick_lows: deque[float] = deque(maxlen=tick_ma_period)
    tick_high_sum = 0.0
    tick_low_sum = 0.0
    tick_widths: deque[float] = deque(maxlen=tick_z_lookback)
    tick_width_hist: list[float] = []

    tick_ready_by_sig_idx: list[bool] = [False] * len(signal_bars)
    tick_dir_by_sig_idx: list[str | None] = [None] * len(signal_bars)
    for sig_i, sig_bar in enumerate(signal_bars):
        while tick_idx < len(tick_bars) and tick_bars[tick_idx].ts <= sig_bar.ts:
            tbar = tick_bars[tick_idx]
            high_v = float(tbar.high)
            low_v = float(tbar.low)

            if len(tick_highs) == tick_highs.maxlen:
                tick_high_sum -= tick_highs[0]
            if len(tick_lows) == tick_lows.maxlen:
                tick_low_sum -= tick_lows[0]
            tick_highs.append(high_v)
            tick_lows.append(low_v)
            tick_high_sum += high_v
            tick_low_sum += low_v

            tick_ready = False
            tick_dir = None
            if len(tick_highs) >= tick_ma_period and len(tick_lows) >= tick_ma_period:
                upper = tick_high_sum / float(tick_ma_period)
                lower = tick_low_sum / float(tick_ma_period)
                width = float(upper) - float(lower)
                tick_widths.append(width)
                tick_width_hist.append(width)

                min_z = min(tick_z_lookback, 30)
                if len(tick_widths) >= max(5, min_z) and len(tick_width_hist) >= (tick_slope_lookback + 1):
                    w_list = list(tick_widths)
                    mean = sum(w_list) / float(len(w_list))
                    var = sum((w - mean) ** 2 for w in w_list) / float(len(w_list))
                    std = math.sqrt(var)
                    z = (width - mean) / std if std > 1e-9 else 0.0
                    delta = width - tick_width_hist[-1 - tick_slope_lookback]

                    if tick_state == "neutral":
                        if z >= tick_z_enter and delta > 0:
                            tick_state = "wide"
                        elif z <= (-tick_z_enter) and delta < 0:
                            tick_state = "narrow"
                    elif tick_state == "wide":
                        if z < tick_z_exit:
                            tick_state = "neutral"
                    elif tick_state == "narrow":
                        if z > (-tick_z_exit):
                            tick_state = "neutral"

                    if tick_state == "wide":
                        tick_dir = "up"
                    elif tick_state == "narrow":
                        tick_dir = "down" if tick_direction_policy == "both" else None
                    else:
                        tick_dir = None
                    tick_ready = True

            tick_idx += 1

        tick_ready_by_sig_idx[sig_i] = bool(tick_ready)
        tick_dir_by_sig_idx[sig_i] = str(tick_dir) if tick_dir in ("up", "down") else None

    out = _SpotTickGateSeries(
        tick_ready_by_sig_idx=tick_ready_by_sig_idx,
        tick_dir_by_sig_idx=tick_dir_by_sig_idx,
    )
    _SERIES_CACHE.set(namespace=_SPOT_TICK_GATE_SERIES_NAMESPACE, key=key, value=out)
    return out


def _spot_core_cache_db_path(cache_dir: object | None) -> Path | None:
    if cache_dir is None:
        return None
    try:
        root = Path(str(cache_dir)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root / "spot_series_core_cache.sqlite3"
    except Exception:
        return None


def _spot_series_core_persistent_get(*, cache_dir: object | None, key_hash: str) -> _SpotSeriesCorePack | None:
    loaded = _SERIES_CACHE.get_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_CORE_NAMESPACE,
        key_hash=str(key_hash),
        validator=lambda obj: isinstance(obj, _SpotSeriesCorePack),
    )
    return loaded if isinstance(loaded, _SpotSeriesCorePack) else None


def _spot_series_core_persistent_set(*, cache_dir: object | None, key_hash: str, payload: _SpotSeriesCorePack) -> None:
    _SERIES_CACHE.set_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_CORE_NAMESPACE,
        key_hash=str(key_hash),
        value=payload,
    )


def _spot_series_pack_persistent_get(*, cache_dir: object | None, key_hash: str) -> _SpotSeriesPack | None:
    loaded = _SERIES_CACHE.get_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_NAMESPACE,
        key_hash=str(key_hash),
        validator=lambda obj: isinstance(obj, _SpotSeriesPack),
    )
    return loaded if isinstance(loaded, _SpotSeriesPack) else None


def _spot_series_pack_persistent_set(*, cache_dir: object | None, key_hash: str, payload: _SpotSeriesPack) -> None:
    _SERIES_CACHE.set_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_NAMESPACE,
        key_hash=str(key_hash),
        value=payload,
    )


def _spot_series_pack_mmap_root(cache_dir: object | None) -> Path | None:
    base = _spot_core_cache_db_path(cache_dir)
    if base is None:
        return None
    try:
        root = base.parent / "spot_series_pack_mmap"
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        return None


def _spot_series_pack_mmap_manifest_get(*, cache_dir: object | None, key_hash: str) -> dict | None:
    loaded = _SERIES_CACHE.get_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_MMAP_NAMESPACE,
        key_hash=str(key_hash),
        validator=lambda obj: isinstance(obj, dict),
    )
    return dict(loaded) if isinstance(loaded, dict) else None


def _spot_series_pack_mmap_manifest_set(*, cache_dir: object | None, key_hash: str, manifest: dict) -> None:
    _SERIES_CACHE.set_persistent(
        db_path=_spot_core_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_MMAP_NAMESPACE,
        key_hash=str(key_hash),
        value=dict(manifest),
    )


def _spot_mmap_supported() -> bool:
    return _np is not None


def _spot_pack_write_npy(*, out_dir: Path, name: str, arr) -> None:
    if _np is None:
        raise RuntimeError("numpy unavailable")
    path = out_dir / f"{name}.npy"
    mm = _np.lib.format.open_memmap(str(path), mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[...] = arr
    del mm


def _spot_pack_read_npy(*, pack_dir: Path, name: str):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    return _np.load(str(pack_dir / f"{name}.npy"), mmap_mode="r")


def _spot_pack_encode_optional_float(values: list[float | None]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    arr = _np.full((len(values),), _np.nan, dtype=_np.float64)
    for i, value in enumerate(values):
        if value is None:
            continue
        try:
            arr[i] = float(value)
        except (TypeError, ValueError):
            continue
    return arr


def _spot_pack_decode_optional_float(arr) -> list[float | None]:
    if _np is None:
        return []
    raw = _np.asarray(arr, dtype=_np.float64)
    out: list[float | None] = [None] * int(raw.shape[0])
    for i, value in enumerate(raw.tolist()):
        if isinstance(value, float) and _np.isnan(value):
            out[i] = None
        else:
            out[i] = float(value)
    return out


def _spot_pack_encode_optional_bool(values: list[bool | None]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    arr = _np.full((len(values),), -1, dtype=_np.int8)
    for i, value in enumerate(values):
        if value is None:
            continue
        arr[i] = 1 if bool(value) else 0
    return arr


def _spot_pack_decode_optional_bool(arr) -> list[bool | None]:
    if _np is None:
        return []
    raw = _np.asarray(arr, dtype=_np.int8).tolist()
    out: list[bool | None] = []
    for value in raw:
        iv = int(value)
        if iv < 0:
            out.append(None)
        else:
            out.append(bool(iv))
    return out


def _spot_pack_encode_bool(values: list[bool]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    return _np.asarray([1 if bool(v) else 0 for v in values], dtype=_np.uint8)


def _spot_pack_decode_bool(arr) -> list[bool]:
    if _np is None:
        return []
    return [bool(int(v)) for v in _np.asarray(arr, dtype=_np.uint8).tolist()]


def _spot_pack_encode_int(values: list[int]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    return _np.asarray([int(v) for v in values], dtype=_np.int32)


def _spot_pack_decode_int(arr) -> list[int]:
    if _np is None:
        return []
    return [int(v) for v in _np.asarray(arr, dtype=_np.int32).tolist()]


def _spot_pack_encode_dir(values: list[str | None]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    code = {"up": 1, "down": 2}
    out = _np.zeros((len(values),), dtype=_np.int8)
    for i, value in enumerate(values):
        out[i] = int(code.get(str(value), 0))
    return out


def _spot_pack_decode_dir(arr) -> list[str | None]:
    decode = {1: "up", 2: "down"}
    if _np is None:
        return []
    raw = _np.asarray(arr, dtype=_np.int8).tolist()
    return [decode.get(int(v)) for v in raw]


def _spot_pack_encode_branch(values: list[str | None]):
    if _np is None:
        raise RuntimeError("numpy unavailable")
    code = {"a": 1, "b": 2}
    out = _np.zeros((len(values),), dtype=_np.int8)
    for i, value in enumerate(values):
        out[i] = int(code.get(str(value), 0))
    return out


def _spot_pack_decode_branch(arr) -> list[str | None]:
    decode = {1: "a", 2: "b"}
    if _np is None:
        return []
    raw = _np.asarray(arr, dtype=_np.int8).tolist()
    return [decode.get(int(v)) for v in raw]


def _spot_series_pack_mmap_persistent_set(*, cache_dir: object | None, key_hash: str, payload: _SpotSeriesPack) -> bool:
    if not _spot_mmap_supported():
        return False
    root = _spot_series_pack_mmap_root(cache_dir)
    if root is None:
        return False
    tmp_dir = root / f".{key_hash}.tmp.{os.getpid()}.{abs(hash(str(key_hash))) % 1000000}"
    target_dir = root / str(key_hash)
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        align = payload.core.align
        signal = payload.core.signal_series
        tick = payload.core.tick_series
        shock = payload.shock_series
        risk_by_day = payload.risk_by_day
        rv = payload.rv_series
        volume = payload.volume_series

        _spot_pack_write_npy(out_dir=tmp_dir, name="align_sig_idx_by_exec_idx", arr=_spot_pack_encode_int(align.sig_idx_by_exec_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="align_exec_idx_by_sig_idx", arr=_spot_pack_encode_int(align.exec_idx_by_sig_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="align_signal_exec_indices", arr=_spot_pack_encode_int(align.signal_exec_indices))
        _spot_pack_write_npy(out_dir=tmp_dir, name="align_signal_sig_indices", arr=_spot_pack_encode_int(align.signal_sig_indices))

        n_sig = len(signal.signal_by_sig_idx)
        if _np is None:
            raise RuntimeError("numpy unavailable")
        signal_present = _np.zeros((n_sig,), dtype=_np.uint8)
        ema_fast = _np.full((n_sig,), _np.nan, dtype=_np.float64)
        ema_slow = _np.full((n_sig,), _np.nan, dtype=_np.float64)
        prev_ema_fast = _np.full((n_sig,), _np.nan, dtype=_np.float64)
        prev_ema_slow = _np.full((n_sig,), _np.nan, dtype=_np.float64)
        ema_ready = _np.zeros((n_sig,), dtype=_np.uint8)
        cross_up = _np.zeros((n_sig,), dtype=_np.uint8)
        cross_down = _np.zeros((n_sig,), dtype=_np.uint8)
        state_code = _np.zeros((n_sig,), dtype=_np.int8)
        entry_dir_code = _np.zeros((n_sig,), dtype=_np.int8)
        regime_dir_code = _np.zeros((n_sig,), dtype=_np.int8)
        regime_ready = _np.zeros((n_sig,), dtype=_np.uint8)
        dir_code = {"up": 1, "down": 2}
        for i, snap in enumerate(signal.signal_by_sig_idx):
            if snap is None:
                continue
            signal_present[i] = 1
            if snap.ema_fast is not None:
                ema_fast[i] = float(snap.ema_fast)
            if snap.ema_slow is not None:
                ema_slow[i] = float(snap.ema_slow)
            if snap.prev_ema_fast is not None:
                prev_ema_fast[i] = float(snap.prev_ema_fast)
            if snap.prev_ema_slow is not None:
                prev_ema_slow[i] = float(snap.prev_ema_slow)
            ema_ready[i] = 1 if bool(snap.ema_ready) else 0
            cross_up[i] = 1 if bool(snap.cross_up) else 0
            cross_down[i] = 1 if bool(snap.cross_down) else 0
            state_code[i] = int(dir_code.get(str(snap.state), 0))
            entry_dir_code[i] = int(dir_code.get(str(snap.entry_dir), 0))
            regime_dir_code[i] = int(dir_code.get(str(snap.regime_dir), 0))
            regime_ready[i] = 1 if bool(snap.regime_ready) else 0

        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_present", arr=signal_present)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_ema_fast", arr=ema_fast)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_ema_slow", arr=ema_slow)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_prev_ema_fast", arr=prev_ema_fast)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_prev_ema_slow", arr=prev_ema_slow)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_ema_ready", arr=ema_ready)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_cross_up", arr=cross_up)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_cross_down", arr=cross_down)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_state_code", arr=state_code)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_entry_dir_code", arr=entry_dir_code)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_regime_dir_code", arr=regime_dir_code)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_regime_ready", arr=regime_ready)
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_bars_in_day", arr=_spot_pack_encode_int(signal.bars_in_day_by_sig_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_entry_dir_by_sig_idx", arr=_spot_pack_encode_dir(signal.entry_dir_by_sig_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="signal_entry_branch_by_sig_idx", arr=_spot_pack_encode_branch(signal.entry_branch_by_sig_idx))

        if tick is not None:
            _spot_pack_write_npy(out_dir=tmp_dir, name="tick_ready_by_sig_idx", arr=_spot_pack_encode_bool(tick.tick_ready_by_sig_idx))
            _spot_pack_write_npy(out_dir=tmp_dir, name="tick_dir_by_sig_idx", arr=_spot_pack_encode_dir(tick.tick_dir_by_sig_idx))

        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_by_exec_idx", arr=_spot_pack_encode_optional_bool(shock.shock_by_exec_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_atr_pct_by_exec_idx", arr=_spot_pack_encode_optional_float(shock.shock_atr_pct_by_exec_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_dir_by_exec_idx", arr=_spot_pack_encode_dir(shock.shock_dir_by_exec_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_by_sig_idx", arr=_spot_pack_encode_optional_bool(shock.shock_by_sig_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_atr_pct_by_sig_idx", arr=_spot_pack_encode_optional_float(shock.shock_atr_pct_by_sig_idx))
        _spot_pack_write_npy(out_dir=tmp_dir, name="shock_dir_by_sig_idx", arr=_spot_pack_encode_dir(shock.shock_dir_by_sig_idx))

        if risk_by_day:
            days_sorted = sorted((d, v) for d, v in risk_by_day.items() if isinstance(d, date))
            day_ord: list[int] = []
            riskoff: list[bool] = []
            riskpanic: list[bool] = []
            riskpop: list[bool] = []
            tr_med: list[float | None] = []
            tr_delta: list[float | None] = []
            neg_ratio: list[float | None] = []
            pos_ratio: list[float | None] = []
            for day_item, snap in days_sorted:
                day_ord.append(int(day_item.toordinal()))
                riskoff.append(bool(getattr(snap, "riskoff", False)))
                riskpanic.append(bool(getattr(snap, "riskpanic", False)))
                riskpop.append(bool(getattr(snap, "riskpop", False)))
                tr_med.append(float(getattr(snap, "tr_median_pct")) if getattr(snap, "tr_median_pct", None) is not None else None)
                tr_delta.append(
                    float(getattr(snap, "tr_median_delta_pct"))
                    if getattr(snap, "tr_median_delta_pct", None) is not None
                    else None
                )
                neg_ratio.append(float(getattr(snap, "neg_gap_ratio")) if getattr(snap, "neg_gap_ratio", None) is not None else None)
                pos_ratio.append(float(getattr(snap, "pos_gap_ratio")) if getattr(snap, "pos_gap_ratio", None) is not None else None)
            _spot_pack_write_npy(out_dir=tmp_dir, name="risk_day_ordinal", arr=_spot_pack_encode_int(day_ord))
            _spot_pack_write_npy(out_dir=tmp_dir, name="riskoff_by_day", arr=_spot_pack_encode_bool(riskoff))
            _spot_pack_write_npy(out_dir=tmp_dir, name="riskpanic_by_day", arr=_spot_pack_encode_bool(riskpanic))
            _spot_pack_write_npy(out_dir=tmp_dir, name="riskpop_by_day", arr=_spot_pack_encode_bool(riskpop))
            _spot_pack_write_npy(out_dir=tmp_dir, name="risk_tr_median_pct", arr=_spot_pack_encode_optional_float(tr_med))
            _spot_pack_write_npy(out_dir=tmp_dir, name="risk_tr_median_delta_pct", arr=_spot_pack_encode_optional_float(tr_delta))
            _spot_pack_write_npy(out_dir=tmp_dir, name="risk_neg_gap_ratio", arr=_spot_pack_encode_optional_float(neg_ratio))
            _spot_pack_write_npy(out_dir=tmp_dir, name="risk_pos_gap_ratio", arr=_spot_pack_encode_optional_float(pos_ratio))

        if rv is not None:
            _spot_pack_write_npy(out_dir=tmp_dir, name="rv_by_sig_idx", arr=_spot_pack_encode_optional_float(rv.rv_by_sig_idx))
        if volume is not None:
            _spot_pack_write_npy(out_dir=tmp_dir, name="volume_ema_by_sig_idx", arr=_spot_pack_encode_optional_float(volume.volume_ema_by_sig_idx))
            _spot_pack_write_npy(out_dir=tmp_dir, name="volume_ema_ready_by_sig_idx", arr=_spot_pack_encode_bool(volume.volume_ema_ready_by_sig_idx))

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        tmp_dir.rename(target_dir)
        manifest = {
            "version": str(_SPOT_SERIES_PACK_MMAP_CACHE_VERSION),
            "key_hash": str(key_hash),
            "dir": str(target_dir.name),
            "has_tick": bool(tick is not None),
            "has_risk": bool(risk_by_day),
            "has_rv": bool(rv is not None),
            "has_volume": bool(volume is not None),
        }
        _spot_series_pack_mmap_manifest_set(
            cache_dir=cache_dir,
            key_hash=str(key_hash),
            manifest=manifest,
        )
        return True
    except Exception:
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return False


def _spot_series_pack_mmap_persistent_get(*, cache_dir: object | None, key_hash: str) -> _SpotSeriesPack | None:
    if not _spot_mmap_supported():
        return None
    root = _spot_series_pack_mmap_root(cache_dir)
    if root is None:
        return None
    manifest = _spot_series_pack_mmap_manifest_get(cache_dir=cache_dir, key_hash=str(key_hash))
    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("version") or "") != str(_SPOT_SERIES_PACK_MMAP_CACHE_VERSION):
        return None
    pack_dir = root / str(manifest.get("dir") or str(key_hash))
    if not pack_dir.exists():
        return None
    try:
        align = _SpotExecAlignment(
            sig_idx_by_exec_idx=_spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="align_sig_idx_by_exec_idx")),
            exec_idx_by_sig_idx=_spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="align_exec_idx_by_sig_idx")),
            signal_exec_indices=_spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="align_signal_exec_indices")),
            signal_sig_indices=_spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="align_signal_sig_indices")),
        )

        signal_present = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_present")
        ema_fast = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_ema_fast")
        ema_slow = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_ema_slow")
        prev_ema_fast = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_prev_ema_fast")
        prev_ema_slow = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_prev_ema_slow")
        ema_ready = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_ema_ready")
        cross_up = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_cross_up")
        cross_down = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_cross_down")
        state_code = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_state_code")
        entry_dir_code = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_entry_dir_code")
        regime_dir_code = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_regime_dir_code")
        regime_ready = _spot_pack_read_npy(pack_dir=pack_dir, name="signal_regime_ready")

        if _np is None:
            return None
        state_decode = {1: "up", 2: "down"}
        n_sig = int(_np.asarray(signal_present).shape[0])
        signal_by_sig_idx: list[EmaDecisionSnapshot | None] = [None] * n_sig
        ema_fast_list = _np.asarray(ema_fast, dtype=_np.float64).tolist()
        ema_slow_list = _np.asarray(ema_slow, dtype=_np.float64).tolist()
        prev_ema_fast_list = _np.asarray(prev_ema_fast, dtype=_np.float64).tolist()
        prev_ema_slow_list = _np.asarray(prev_ema_slow, dtype=_np.float64).tolist()
        ema_ready_list = _np.asarray(ema_ready, dtype=_np.uint8).tolist()
        cross_up_list = _np.asarray(cross_up, dtype=_np.uint8).tolist()
        cross_down_list = _np.asarray(cross_down, dtype=_np.uint8).tolist()
        state_code_list = _np.asarray(state_code, dtype=_np.int8).tolist()
        entry_dir_code_list = _np.asarray(entry_dir_code, dtype=_np.int8).tolist()
        regime_dir_code_list = _np.asarray(regime_dir_code, dtype=_np.int8).tolist()
        regime_ready_list = _np.asarray(regime_ready, dtype=_np.uint8).tolist()
        signal_present_list = _np.asarray(signal_present, dtype=_np.uint8).tolist()
        for i in range(n_sig):
            if int(signal_present_list[i]) <= 0:
                continue
            fast_val = float(ema_fast_list[i]) if not _np.isnan(float(ema_fast_list[i])) else None
            slow_val = float(ema_slow_list[i]) if not _np.isnan(float(ema_slow_list[i])) else None
            prev_fast_val = float(prev_ema_fast_list[i]) if not _np.isnan(float(prev_ema_fast_list[i])) else None
            prev_slow_val = float(prev_ema_slow_list[i]) if not _np.isnan(float(prev_ema_slow_list[i])) else None
            signal_by_sig_idx[i] = EmaDecisionSnapshot(
                ema_fast=fast_val,
                ema_slow=slow_val,
                prev_ema_fast=prev_fast_val,
                prev_ema_slow=prev_slow_val,
                ema_ready=bool(int(ema_ready_list[i])),
                cross_up=bool(int(cross_up_list[i])),
                cross_down=bool(int(cross_down_list[i])),
                state=state_decode.get(int(state_code_list[i])),
                entry_dir=state_decode.get(int(entry_dir_code_list[i])),
                regime_dir=state_decode.get(int(regime_dir_code_list[i])),
                regime_ready=bool(int(regime_ready_list[i])),
            )

        signal_series = _SpotSignalSeries(
            signal_by_sig_idx=signal_by_sig_idx,
            bars_in_day_by_sig_idx=_spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="signal_bars_in_day")),
            entry_dir_by_sig_idx=_spot_pack_decode_dir(_spot_pack_read_npy(pack_dir=pack_dir, name="signal_entry_dir_by_sig_idx")),
            entry_branch_by_sig_idx=_spot_pack_decode_branch(
                _spot_pack_read_npy(pack_dir=pack_dir, name="signal_entry_branch_by_sig_idx")
            ),
        )

        tick_series: _SpotTickGateSeries | None = None
        if bool(manifest.get("has_tick")):
            tick_series = _SpotTickGateSeries(
                tick_ready_by_sig_idx=_spot_pack_decode_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="tick_ready_by_sig_idx")),
                tick_dir_by_sig_idx=_spot_pack_decode_dir(_spot_pack_read_npy(pack_dir=pack_dir, name="tick_dir_by_sig_idx")),
            )

        core = _SpotSeriesCorePack(
            align=align,
            signal_series=signal_series,
            tick_series=tick_series,
        )
        shock_series = _SpotShockSeries(
            shock_by_exec_idx=_spot_pack_decode_optional_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="shock_by_exec_idx")),
            shock_atr_pct_by_exec_idx=_spot_pack_decode_optional_float(
                _spot_pack_read_npy(pack_dir=pack_dir, name="shock_atr_pct_by_exec_idx")
            ),
            shock_dir_by_exec_idx=_spot_pack_decode_dir(_spot_pack_read_npy(pack_dir=pack_dir, name="shock_dir_by_exec_idx")),
            shock_by_sig_idx=_spot_pack_decode_optional_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="shock_by_sig_idx")),
            shock_atr_pct_by_sig_idx=_spot_pack_decode_optional_float(
                _spot_pack_read_npy(pack_dir=pack_dir, name="shock_atr_pct_by_sig_idx")
            ),
            shock_dir_by_sig_idx=_spot_pack_decode_dir(_spot_pack_read_npy(pack_dir=pack_dir, name="shock_dir_by_sig_idx")),
        )

        risk_by_day: dict[date, object] = {}
        if bool(manifest.get("has_risk")):
            day_ordinal = _spot_pack_decode_int(_spot_pack_read_npy(pack_dir=pack_dir, name="risk_day_ordinal"))
            riskoff = _spot_pack_decode_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="riskoff_by_day"))
            riskpanic = _spot_pack_decode_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="riskpanic_by_day"))
            riskpop = _spot_pack_decode_bool(_spot_pack_read_npy(pack_dir=pack_dir, name="riskpop_by_day"))
            tr_med = _spot_pack_decode_optional_float(_spot_pack_read_npy(pack_dir=pack_dir, name="risk_tr_median_pct"))
            tr_delta = _spot_pack_decode_optional_float(
                _spot_pack_read_npy(pack_dir=pack_dir, name="risk_tr_median_delta_pct")
            )
            neg_ratio = _spot_pack_decode_optional_float(_spot_pack_read_npy(pack_dir=pack_dir, name="risk_neg_gap_ratio"))
            pos_ratio = _spot_pack_decode_optional_float(_spot_pack_read_npy(pack_dir=pack_dir, name="risk_pos_gap_ratio"))
            n_day = min(
                len(day_ordinal),
                len(riskoff),
                len(riskpanic),
                len(riskpop),
                len(tr_med),
                len(tr_delta),
                len(neg_ratio),
                len(pos_ratio),
            )
            for i in range(n_day):
                day_key = date.fromordinal(int(day_ordinal[i]))
                risk_by_day[day_key] = RiskOverlaySnapshot(
                    riskoff=bool(riskoff[i]),
                    riskpanic=bool(riskpanic[i]),
                    riskpop=bool(riskpop[i]),
                    tr_median_pct=float(tr_med[i]) if tr_med[i] is not None else None,
                    tr_median_delta_pct=float(tr_delta[i]) if tr_delta[i] is not None else None,
                    neg_gap_ratio=float(neg_ratio[i]) if neg_ratio[i] is not None else None,
                    pos_gap_ratio=float(pos_ratio[i]) if pos_ratio[i] is not None else None,
                )

        rv_series: _SpotRvSeries | None = None
        if bool(manifest.get("has_rv")):
            rv_series = _SpotRvSeries(
                rv_by_sig_idx=_spot_pack_decode_optional_float(_spot_pack_read_npy(pack_dir=pack_dir, name="rv_by_sig_idx"))
            )
        volume_series: _SpotVolumeEmaSeries | None = None
        if bool(manifest.get("has_volume")):
            volume_series = _SpotVolumeEmaSeries(
                volume_ema_by_sig_idx=_spot_pack_decode_optional_float(
                    _spot_pack_read_npy(pack_dir=pack_dir, name="volume_ema_by_sig_idx")
                ),
                volume_ema_ready_by_sig_idx=_spot_pack_decode_bool(
                    _spot_pack_read_npy(pack_dir=pack_dir, name="volume_ema_ready_by_sig_idx")
                ),
            )
        return _SpotSeriesPack(
            core=core,
            shock_series=shock_series,
            risk_by_day=risk_by_day,
            rv_series=rv_series,
            volume_series=volume_series,
        )
    except Exception:
        return None


def _spot_series_pack_key_info(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    regime_bars: list[Bar] | None,
    regime2_bars: list[Bar] | None,
    tick_bars: list[Bar] | None,
    include_rv: bool,
    include_volume: bool,
    include_tick: bool,
) -> tuple[tuple[object, ...], tuple[object, ...], str, int]:
    strat = cfg.strategy
    filters = strat.filters
    raw_period = getattr(filters, "volume_ema_period", None) if filters is not None else None
    try:
        vol_period = int(raw_period) if raw_period is not None else 20
    except (TypeError, ValueError):
        vol_period = 20

    core_key = (
        str(_SPOT_SERIES_CORE_CACHE_VERSION),
        _spot_bars_signature(signal_bars),
        _spot_bars_signature(exec_bars),
        _spot_bars_signature(regime_bars),
        _spot_bars_signature(regime2_bars),
        _spot_bars_signature(tick_bars if include_tick else None),
        _spot_signal_series_signature(cfg=cfg),
        _spot_tick_gate_settings(strat),
    )
    shock_signature = _spot_filter_signature(
        filters,
        include_prefixes=("shock_",),
        include_keys=("shock_gate_mode",),
    )
    risk_signature = _spot_filter_signature(
        filters,
        include_prefixes=("riskoff_", "riskpanic_", "riskpop_"),
        include_keys=("riskoff_mode", "riskpanic_long_scale_mode", "risk_entry_cutoff_hour_et"),
    )
    rv_signature = (
        bool(include_rv),
        int(cfg.synthetic.rv_lookback),
        float(cfg.synthetic.rv_ewma_lambda),
        str(cfg.backtest.bar_size),
        bool(cfg.backtest.use_rth),
    )
    volume_signature = (bool(include_volume), int(vol_period))
    pack_key = (
        str(_SPOT_SERIES_PACK_CACHE_VERSION),
        core_key,
        shock_signature,
        risk_signature,
        rv_signature,
        volume_signature,
    )
    pack_key_hash = hashlib.sha1(repr(pack_key).encode("utf-8")).hexdigest()
    return core_key, pack_key, str(pack_key_hash), int(vol_period)


def _spot_series_pack_cache_state(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    regime_bars: list[Bar] | None,
    regime2_bars: list[Bar] | None,
    tick_bars: list[Bar] | None,
    include_rv: bool,
    include_volume: bool,
    include_tick: bool,
) -> str:
    _core_key, pack_key, pack_key_hash, _vol_period = _spot_series_pack_key_info(
        cfg=cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
        tick_bars=tick_bars,
        include_rv=bool(include_rv),
        include_volume=bool(include_volume),
        include_tick=bool(include_tick),
    )
    cached_pack = _SERIES_CACHE.get(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key)
    if isinstance(cached_pack, _SpotSeriesPack):
        return "memory"
    use_persist = bool(getattr(cfg.backtest, "offline", False))
    if not bool(use_persist):
        return "none"
    manifest = _spot_series_pack_mmap_manifest_get(
        cache_dir=cfg.backtest.cache_dir,
        key_hash=str(pack_key_hash),
    )
    if isinstance(manifest, dict) and str(manifest.get("version") or "") == str(_SPOT_SERIES_PACK_MMAP_CACHE_VERSION):
        root = _spot_series_pack_mmap_root(cfg.backtest.cache_dir)
        if root is not None:
            pack_dir = root / str(manifest.get("dir") or str(pack_key_hash))
            if pack_dir.exists():
                return "mmap"
    if _SERIES_CACHE.has_persistent(
        db_path=_spot_core_cache_db_path(cfg.backtest.cache_dir),
        namespace=_SPOT_SERIES_PACK_NAMESPACE,
        key_hash=str(pack_key_hash),
    ):
        return "pickle"
    return "none"


def _spot_prepare_summary_series_pack(
    *,
    cfg: ConfigBundle,
    signal_bars: BarSeriesInput,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
) -> tuple[str, object | None]:
    """Prepare (or warm) the summary runner series-pack and return its stable key hash.

    The returned object is an internal pack payload that can be passed back into
    `_run_spot_backtest_summary(..., prepared_series_pack=...)` to bypass repeated
    pack construction in tight sweep loops.
    """
    signal_list = _bars_input_list(signal_bars)
    regime_list = _bars_input_optional_list(regime_bars)
    regime2_list = _bars_input_optional_list(regime2_bars)
    tick_list = _bars_input_optional_list(tick_bars)
    exec_list = _bars_input_optional_list(exec_bars)

    exec_bar_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
    if exec_bar_size and str(exec_bar_size) != str(cfg.backtest.bar_size):
        if exec_list is None or not exec_list:
            return "", None
    else:
        exec_list = signal_list

    filters = cfg.strategy.filters
    needs_rv = bool(filters is not None and (getattr(filters, "rv_min", None) is not None or getattr(filters, "rv_max", None) is not None))
    needs_volume = bool(filters is not None and getattr(filters, "volume_ratio_min", None) is not None)
    tick_mode = str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
    include_tick = bool(str(tick_mode) != "off")
    use_fast = False
    try:
        use_fast = bool(
            _can_use_fast_summary_path(
                cfg,
                signal_bars=signal_list,
                exec_bars=exec_list,
                tick_bars=tick_list,
            )
        )
    except Exception:
        use_fast = False
    include_rv = bool(needs_rv) if bool(use_fast) else False
    include_volume = bool(needs_volume) if bool(use_fast) else False

    _core_key, pack_key, pack_key_hash, _vol_period = _spot_series_pack_key_info(
        cfg=cfg,
        signal_bars=signal_list,
        exec_bars=exec_list,
        regime_bars=regime_list,
        regime2_bars=regime2_list,
        tick_bars=tick_list,
        include_rv=bool(include_rv),
        include_volume=bool(include_volume),
        include_tick=bool(include_tick),
    )
    cached_pack = _SERIES_CACHE.get(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key)
    if isinstance(cached_pack, _SpotSeriesPack):
        return str(pack_key_hash), cached_pack
    prepared = _spot_build_series_pack(
        cfg=cfg,
        signal_bars=signal_list,
        exec_bars=exec_list,
        regime_bars=regime_list,
        regime2_bars=regime2_list,
        tick_bars=tick_list,
        include_rv=bool(include_rv),
        include_volume=bool(include_volume),
        include_tick=bool(include_tick),
    )
    return str(pack_key_hash), prepared


def _spot_build_series_pack(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    regime_bars: list[Bar] | None,
    regime2_bars: list[Bar] | None,
    tick_bars: list[Bar] | None,
    include_rv: bool,
    include_volume: bool,
    include_tick: bool,
) -> _SpotSeriesPack:
    strat = cfg.strategy
    filters = strat.filters
    use_persist = bool(getattr(cfg.backtest, "offline", False))
    core_key, pack_key, pack_key_hash, vol_period = _spot_series_pack_key_info(
        cfg=cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
        tick_bars=tick_bars,
        include_rv=bool(include_rv),
        include_volume=bool(include_volume),
        include_tick=bool(include_tick),
    )
    cached_pack = _SERIES_CACHE.get(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key)
    if isinstance(cached_pack, _SpotSeriesPack):
        return cached_pack
    pack_loaded: _SpotSeriesPack | None = None
    if bool(use_persist):
        pack_loaded = _spot_series_pack_mmap_persistent_get(
            cache_dir=cfg.backtest.cache_dir,
            key_hash=str(pack_key_hash),
        )
        if pack_loaded is None:
            pack_loaded = _spot_series_pack_persistent_get(
                cache_dir=cfg.backtest.cache_dir,
                key_hash=str(pack_key_hash),
            )
    if pack_loaded is not None:
        _SERIES_CACHE.set(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key, value=pack_loaded)
        return pack_loaded

    core_cached = _SERIES_CACHE.get(namespace=_SPOT_SERIES_CORE_NAMESPACE, key=core_key)
    core = core_cached if isinstance(core_cached, _SpotSeriesCorePack) else None
    if core is None:
        key_hash = hashlib.sha1(repr(core_key).encode("utf-8")).hexdigest()
        core_loaded = (
            _spot_series_core_persistent_get(cache_dir=cfg.backtest.cache_dir, key_hash=str(key_hash))
            if bool(use_persist)
            else None
        )
        if core_loaded is None:
            core_loaded = _SpotSeriesCorePack(
                align=_spot_exec_alignment(signal_bars, exec_bars),
                signal_series=_spot_signal_series(
                    cfg=cfg,
                    signal_bars=signal_bars,
                    regime_bars=regime_bars,
                    regime2_bars=regime2_bars,
                ),
                tick_series=(
                    _spot_tick_gate_series(
                        signal_bars=signal_bars,
                        tick_bars=tick_bars,
                        strategy=strat,
                    )
                    if bool(include_tick)
                    else None
                ),
            )
            if bool(use_persist):
                _spot_series_core_persistent_set(
                    cache_dir=cfg.backtest.cache_dir,
                    key_hash=str(key_hash),
                    payload=core_loaded,
                )
        _SERIES_CACHE.set(namespace=_SPOT_SERIES_CORE_NAMESPACE, key=core_key, value=core_loaded)
        core = core_loaded

    st_src_for_shock = str(getattr(strat, "supertrend_source", "hl2") or "hl2").strip().lower() or "hl2"
    shock_series = _spot_shock_series(
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        regime_bars=regime_bars,
        filters=filters,
        st_src_for_shock=st_src_for_shock,
    )
    risk_by_day = _spot_risk_overlay_flags_by_day(exec_bars, filters)

    rv_series: _SpotRvSeries | None = None
    if bool(include_rv) and filters is not None:
        rv_series = _spot_rv_series(
            signal_bars=signal_bars,
            rv_lookback=int(cfg.synthetic.rv_lookback),
            rv_ewma_lambda=float(cfg.synthetic.rv_ewma_lambda),
            bar_size=str(cfg.backtest.bar_size),
            use_rth=bool(cfg.backtest.use_rth),
        )

    volume_series: _SpotVolumeEmaSeries | None = None
    if bool(include_volume) and filters is not None:
        volume_series = _spot_volume_ema_series(signal_bars=signal_bars, period=int(vol_period))

    out = _SpotSeriesPack(
        core=core,
        shock_series=shock_series,
        risk_by_day=risk_by_day,
        rv_series=rv_series,
        volume_series=volume_series,
    )
    _SERIES_CACHE.set(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key, value=out)
    if bool(use_persist):
        mmap_written = _spot_series_pack_mmap_persistent_set(
            cache_dir=cfg.backtest.cache_dir,
            key_hash=str(pack_key_hash),
            payload=out,
        )
        if not bool(mmap_written):
            _spot_series_pack_persistent_set(
                cache_dir=cfg.backtest.cache_dir,
                key_hash=str(pack_key_hash),
                payload=out,
            )
    return out


def _spot_direction_allowed(
    *,
    entry_dir: str | None,
    needs_direction: bool,
    directional_spot: object | None,
) -> str | None:
    if bool(needs_direction):
        if entry_dir is None:
            return None
        if directional_spot is None:
            return None
        try:
            return str(entry_dir) if str(entry_dir) in directional_spot else None
        except Exception:
            return None
    return "up" if str(entry_dir) == "up" else None


def _spot_resolve_entry_dir(
    *,
    signal: EmaDecisionSnapshot | None,
    entry_dir: str | None,
    ema_needed: bool,
    sig_idx: int | None,
    tick_mode: str,
    tick_series: _SpotTickGateSeries | None,
    tick_neutral_policy: str,
    needs_direction: bool,
    directional_spot: object | None,
) -> tuple[str | None, bool]:
    ema_ready = bool(signal is not None and bool(getattr(signal, "ema_ready", False)))
    resolved = entry_dir
    if resolved is None and signal is not None:
        resolved = signal.entry_dir
    if bool(ema_needed) and not bool(ema_ready):
        resolved = None
    if (
        str(tick_mode) != "off"
        and tick_series is not None
        and sig_idx is not None
        and 0 <= int(sig_idx) < len(tick_series.tick_ready_by_sig_idx)
    ):
        tick_ready = bool(tick_series.tick_ready_by_sig_idx[int(sig_idx)])
        tick_dir_raw = tick_series.tick_dir_by_sig_idx[int(sig_idx)]
        tick_dir = str(tick_dir_raw) if tick_dir_raw in ("up", "down") else None
        resolved = _spot_apply_tick_gate_to_entry_dir(
            entry_dir=resolved,
            tick_ready=bool(tick_ready),
            tick_dir=tick_dir,
            tick_neutral_policy=str(tick_neutral_policy),
        )
    resolved = _spot_direction_allowed(
        entry_dir=resolved,
        needs_direction=bool(needs_direction),
        directional_spot=directional_spot,
    )
    return resolved, bool(ema_ready)


def _spot_exec_profile(strategy: object) -> _SpotExecProfile:
    runtime = spot_runtime_spec_view(
        strategy=strategy,
        filters=getattr(strategy, "filters", None),
    )
    return _SpotExecProfile(
        entry_fill_mode=str(runtime.entry_fill_mode),
        flip_fill_mode=str(runtime.flip_exit_fill_mode),
        exit_mode=str(runtime.exit_mode),
        intrabar_exits=bool(runtime.intrabar_exits),
        close_eod=bool(runtime.close_eod),
        spread=float(runtime.spread),
        commission_per_share=float(runtime.commission_per_share),
        commission_min=float(runtime.commission_min),
        slippage_per_share=float(runtime.slippage_per_share),
        mark_to_market=str(runtime.mark_to_market),
        drawdown_mode=str(runtime.drawdown_mode),
    )


def _spot_rv_series(
    *,
    signal_bars: list[Bar],
    rv_lookback: int,
    rv_ewma_lambda: float,
    bar_size: str,
    use_rth: bool,
) -> _SpotRvSeries:
    key = (id(signal_bars), int(rv_lookback), float(rv_ewma_lambda), str(bar_size), bool(use_rth))
    cached = _SERIES_CACHE.get(namespace=_SPOT_RV_SERIES_NAMESPACE, key=key)
    if isinstance(cached, _SpotRvSeries):
        return cached

    lb = max(1, int(rv_lookback))
    lam = float(rv_ewma_lambda)
    returns: deque[float] = deque(maxlen=lb)
    prev_close: float | None = None
    rv_by_sig_idx: list[float | None] = [None] * len(signal_bars)
    for i, bar in enumerate(signal_bars):
        close = float(bar.close)
        if close <= 0:
            continue
        if prev_close is not None and float(prev_close) > 0 and close > 0:
            returns.append(math.log(close / float(prev_close)))
        prev_close = float(close)
        rv = annualized_ewma_vol(returns, lam=lam, bar_size=str(bar_size), use_rth=bool(use_rth))
        rv_by_sig_idx[i] = float(rv) if rv is not None else None

    out = _SpotRvSeries(rv_by_sig_idx=rv_by_sig_idx)
    _SERIES_CACHE.set(namespace=_SPOT_RV_SERIES_NAMESPACE, key=key, value=out)
    return out


def _spot_volume_ema_series(*, signal_bars: list[Bar], period: int) -> _SpotVolumeEmaSeries:
    p = max(1, int(period))
    key = (id(signal_bars), int(p))
    cached = _SERIES_CACHE.get(namespace=_SPOT_VOLUME_EMA_SERIES_NAMESPACE, key=key)
    if isinstance(cached, _SpotVolumeEmaSeries):
        return cached

    ema: float | None = None
    count = 0
    ema_by_sig_idx: list[float | None] = [None] * len(signal_bars)
    ready_by_sig_idx: list[bool] = [False] * len(signal_bars)
    for i, bar in enumerate(signal_bars):
        vol = float(bar.volume) if bar.volume is not None else 0.0
        ema = ema_next(ema, float(vol), int(p))
        count += 1
        ema_by_sig_idx[i] = float(ema) if ema is not None else None
        ready_by_sig_idx[i] = bool(count >= int(p))

    out = _SpotVolumeEmaSeries(volume_ema_by_sig_idx=ema_by_sig_idx, volume_ema_ready_by_sig_idx=ready_by_sig_idx)
    _SERIES_CACHE.set(namespace=_SPOT_VOLUME_EMA_SERIES_NAMESPACE, key=key, value=out)
    return out


def _spot_signal_series(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    regime_bars: list[Bar] | None,
    regime2_bars: list[Bar] | None,
) -> _SpotSignalSeries:
    """Precompute EMA/regime-gated signal snapshots once per bar set + strategy params.

    This intentionally runs with `filters=None` so the series is reusable across the sweep
    (permission gates / shock / overlays are evaluated per-config).
    """
    strat = cfg.strategy
    key = (
        id(signal_bars),
        id(regime_bars) if regime_bars else 0,
        id(regime2_bars) if regime2_bars else 0,
        *_spot_signal_series_signature(cfg=cfg),
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_SIGNAL_SERIES_NAMESPACE, key=key)
    if isinstance(cached, _SpotSignalSeries):
        return cached

    from ..spot_engine import SpotSignalEvaluator

    evaluator = SpotSignalEvaluator(
        strategy=strat,
        filters=None,
        bar_size=str(cfg.backtest.bar_size),
        use_rth=bool(cfg.backtest.use_rth),
        naive_ts_mode="utc",
        rv_lookback=int(cfg.synthetic.rv_lookback),
        rv_ewma_lambda=float(cfg.synthetic.rv_ewma_lambda),
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
    )
    signal_by_sig_idx: list[EmaDecisionSnapshot | None] = [None] * len(signal_bars)
    bars_in_day_by_sig_idx: list[int] = [0] * len(signal_bars)
    entry_dir_by_sig_idx: list[str | None] = [None] * len(signal_bars)
    entry_branch_by_sig_idx: list[str | None] = [None] * len(signal_bars)
    for i, bar in enumerate(signal_bars):
        snap = evaluator.update_signal_bar(bar)
        if snap is None:
            continue
        signal_by_sig_idx[i] = snap.signal
        bars_in_day_by_sig_idx[i] = int(snap.bars_in_day)
        entry_dir_by_sig_idx[i] = str(snap.entry_dir) if snap.entry_dir in ("up", "down") else None
        entry_branch_by_sig_idx[i] = str(snap.entry_branch) if snap.entry_branch in ("a", "b") else None

    out = _SpotSignalSeries(
        signal_by_sig_idx=signal_by_sig_idx,
        bars_in_day_by_sig_idx=bars_in_day_by_sig_idx,
        entry_dir_by_sig_idx=entry_dir_by_sig_idx,
        entry_branch_by_sig_idx=entry_branch_by_sig_idx,
    )
    _SERIES_CACHE.set(namespace=_SPOT_SIGNAL_SERIES_NAMESPACE, key=key, value=out)
    return out


def _spot_flip_trees(
    *,
    signal_bars: list[Bar],
    signal_series: _SpotSignalSeries,
    exit_on_signal_flip: bool,
    flip_exit_mode: str,
    ema_entry_mode: str,
    only_if_profit: bool,
) -> tuple[_MaxFirstGtTree | None, _MaxFirstGtTree | None]:
    """Return (long_tree, short_tree) to find next profitable flip exit on signal bars.

    Only used for the fast runner when `only_if_profit=True`.
    """
    if not bool(exit_on_signal_flip):
        return None, None
    if not bool(only_if_profit):
        return None, None

    key = (
        id(signal_bars),
        id(signal_series.signal_by_sig_idx),
        bool(exit_on_signal_flip),
        str(flip_exit_mode or ""),
        str(ema_entry_mode or ""),
        "only_profit",
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_FLIP_TREE_NAMESPACE, key=key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached  # type: ignore[return-value]

    closes = [float(b.close) for b in signal_bars]
    up_vals: list[float] = []
    down_vals: list[float] = []
    for i, sig in enumerate(signal_series.signal_by_sig_idx):
        hit_up = flip_exit_hit(
            exit_on_signal_flip=True,
            open_dir="up",
            signal=sig,
            flip_exit_mode_raw=flip_exit_mode,
            ema_entry_mode_raw=ema_entry_mode,
        )
        hit_down = flip_exit_hit(
            exit_on_signal_flip=True,
            open_dir="down",
            signal=sig,
            flip_exit_mode_raw=flip_exit_mode,
            ema_entry_mode_raw=ema_entry_mode,
        )
        up_vals.append(float(closes[i]) if hit_up else float("-inf"))
        down_vals.append((-float(closes[i])) if hit_down else float("-inf"))

    long_tree = _MaxFirstGtTree(up_vals)
    short_tree = _MaxFirstGtTree(down_vals)
    out = (long_tree, short_tree)
    _SERIES_CACHE.set(namespace=_SPOT_FLIP_TREE_NAMESPACE, key=key, value=out)
    return out


def _spot_flip_next_sig_idx(
    *,
    signal_series: _SpotSignalSeries,
    exit_on_signal_flip: bool,
    flip_exit_mode: str,
    ema_entry_mode: str,
) -> tuple[list[int], list[int]]:
    if not bool(exit_on_signal_flip):
        n = len(signal_series.signal_by_sig_idx)
        return ([-1] * n, [-1] * n)

    key = (
        id(signal_series.signal_by_sig_idx),
        bool(exit_on_signal_flip),
        str(flip_exit_mode or ""),
        str(ema_entry_mode or ""),
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_FLIP_NEXT_SIG_NAMESPACE, key=key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    n = len(signal_series.signal_by_sig_idx)
    hit_up = [False] * n
    hit_down = [False] * n
    for i, sig in enumerate(signal_series.signal_by_sig_idx):
        hit_up[i] = bool(
            flip_exit_hit(
                exit_on_signal_flip=True,
                open_dir="up",
                signal=sig,
                flip_exit_mode_raw=flip_exit_mode,
                ema_entry_mode_raw=ema_entry_mode,
            )
        )
        hit_down[i] = bool(
            flip_exit_hit(
                exit_on_signal_flip=True,
                open_dir="down",
                signal=sig,
                flip_exit_mode_raw=flip_exit_mode,
                ema_entry_mode_raw=ema_entry_mode,
            )
        )

    next_up = [-1] * n
    next_down = [-1] * n
    nxt = -1
    for i in range(n - 1, -1, -1):
        if hit_up[i]:
            nxt = int(i)
        next_up[i] = int(nxt)
    nxt = -1
    for i in range(n - 1, -1, -1):
        if hit_down[i]:
            nxt = int(i)
        next_down[i] = int(nxt)

    out = (next_up, next_down)
    _SERIES_CACHE.set(namespace=_SPOT_FLIP_NEXT_SIG_NAMESPACE, key=key, value=out)
    return out


def _spot_daily_ohlc(exec_bars: list[Bar]) -> list[_SpotDailyOhlc]:
    key = id(exec_bars)
    cached = _SERIES_CACHE.get(namespace=_SPOT_DAILY_OHLC_NAMESPACE, key=key)
    if isinstance(cached, list):
        return cached
    out: list[_SpotDailyOhlc] = []
    cur_day: date | None = None
    day_open = 0.0
    day_high = 0.0
    day_low = 0.0
    day_close = 0.0
    last_ts: datetime | None = None
    for bar in exec_bars:
        d = _trade_date(bar.ts)
        if cur_day != d:
            if cur_day is not None and last_ts is not None:
                out.append(
                    _SpotDailyOhlc(
                        day=cur_day,
                        ts=last_ts,
                        open=float(day_open),
                        high=float(day_high),
                        low=float(day_low),
                        close=float(day_close),
                    )
                )
            cur_day = d
            day_open = float(bar.open)
            day_high = float(bar.high)
            day_low = float(bar.low)
            day_close = float(bar.close)
            last_ts = bar.ts
            continue
        day_high = max(float(day_high), float(bar.high))
        day_low = min(float(day_low), float(bar.low))
        day_close = float(bar.close)
        last_ts = bar.ts
    if cur_day is not None and last_ts is not None:
        out.append(
            _SpotDailyOhlc(
                day=cur_day,
                ts=last_ts,
                open=float(day_open),
                high=float(day_high),
                low=float(day_low),
                close=float(day_close),
            )
        )
    _SERIES_CACHE.set(namespace=_SPOT_DAILY_OHLC_NAMESPACE, key=key, value=out)
    return out


def _spot_risk_overlay_flags_by_day(exec_bars: list[Bar], filters: object | None) -> dict[date, object]:
    """Precompute risk overlay snapshots by day for a given filter knob set.

    Returns a mapping day -> RiskOverlaySnapshot (or empty when overlays are disabled).
    """
    if filters is None:
        return {}
    engine = build_tr_pct_risk_overlay_engine(filters)
    if engine is None:
        return {}
    risk_key = (
        getattr(filters, "riskoff_tr5_med_pct", None),
        getattr(filters, "riskoff_tr5_lookback_days", None),
        getattr(filters, "riskpanic_tr5_med_pct", None),
        getattr(filters, "riskpanic_neg_gap_ratio_min", None),
        getattr(filters, "riskpanic_neg_gap_abs_pct_min", None),
        getattr(filters, "riskpanic_lookback_days", None),
        getattr(filters, "riskpanic_tr5_med_delta_min_pct", None),
        getattr(filters, "riskpanic_tr5_med_delta_lookback_days", None),
        getattr(filters, "riskpop_tr5_med_pct", None),
        getattr(filters, "riskpop_pos_gap_ratio_min", None),
        getattr(filters, "riskpop_pos_gap_abs_pct_min", None),
        getattr(filters, "riskpop_lookback_days", None),
        getattr(filters, "riskpop_tr5_med_delta_min_pct", None),
        getattr(filters, "riskpop_tr5_med_delta_lookback_days", None),
    )
    cache_key = (id(exec_bars), risk_key)
    cached = _SERIES_CACHE.get(namespace=_SPOT_RISK_OVERLAY_DAY_NAMESPACE, key=cache_key)
    if isinstance(cached, dict):
        return cached
    by_day: dict[date, object] = {}
    for day_bar in _spot_daily_ohlc(exec_bars):
        snap = engine.update(
            ts=day_bar.ts,
            open=float(day_bar.open),
            high=float(day_bar.high),
            low=float(day_bar.low),
            close=float(day_bar.close),
            is_last_bar=True,
            trade_day=day_bar.day,
        )
        by_day[day_bar.day] = snap
    _SERIES_CACHE.set(namespace=_SPOT_RISK_OVERLAY_DAY_NAMESPACE, key=cache_key, value=by_day)
    return by_day


def _spot_shock_series(
    *,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    regime_bars: list[Bar] | None,
    filters: object | None,
    st_src_for_shock: str,
) -> _SpotShockSeries:
    """Precompute shock view series on both exec bars and signal bars.

    Semantics match `SpotSignalEvaluator._shock_view` + the mtf update schedule used in
    `SpotSignalEvaluator.update_signal_bar/update_exec_bar`.
    """
    if filters is None:
        return _SpotShockSeries(
            shock_by_exec_idx=[None] * len(exec_bars),
            shock_atr_pct_by_exec_idx=[None] * len(exec_bars),
            shock_dir_by_exec_idx=[None] * len(exec_bars),
            shock_by_sig_idx=[None] * len(signal_bars),
            shock_atr_pct_by_sig_idx=[None] * len(signal_bars),
            shock_dir_by_sig_idx=[None] * len(signal_bars),
        )

    mode = normalize_shock_gate_mode(filters)
    if mode == "off":
        return _SpotShockSeries(
            shock_by_exec_idx=[None] * len(exec_bars),
            shock_atr_pct_by_exec_idx=[None] * len(exec_bars),
            shock_dir_by_exec_idx=[None] * len(exec_bars),
            shock_by_sig_idx=[None] * len(signal_bars),
            shock_atr_pct_by_sig_idx=[None] * len(signal_bars),
            shock_dir_by_sig_idx=[None] * len(signal_bars),
        )

    detector = normalize_shock_detector(filters)
    dir_source = normalize_shock_direction_source(filters)
    dir_lb = int(getattr(filters, "shock_direction_lookback", 2) or 2)
    scale_detector_raw = getattr(filters, "shock_scale_detector", None)
    if scale_detector_raw is not None:
        scale_detector_raw = str(scale_detector_raw).strip().lower()
    if scale_detector_raw in ("", "0", "false", "none", "null", "off"):
        scale_detector_raw = None
    scale_detector = None
    if scale_detector_raw is not None:
        if scale_detector_raw in ("daily", "daily_atr", "daily_atr_pct", "daily_atr14", "daily_atr%"):
            scale_detector = "daily_atr_pct"
        elif scale_detector_raw in ("drawdown", "daily_drawdown", "daily-dd", "dd", "peak_dd", "peak_drawdown"):
            scale_detector = "daily_drawdown"
        elif scale_detector_raw in ("tr_ratio", "tr-ratio", "tr_ratio_pct", "tr_ratio%"):
            scale_detector = "tr_ratio"
        elif scale_detector_raw in ("atr_ratio", "ratio", "atr-ratio", "atr_ratio_pct", "atr_ratio%"):
            scale_detector = "atr_ratio"

    # Cache by bar identities + normalized shock knobs.
    shock_key: tuple[object, ...]
    if detector == "daily_atr_pct":
        shock_key = (
            mode,
            detector,
            dir_source,
            int(dir_lb),
            int(getattr(filters, "shock_daily_atr_period", 14) or 14),
            float(getattr(filters, "shock_daily_on_atr_pct", 13.0) or 13.0),
            float(getattr(filters, "shock_daily_off_atr_pct", 11.0) or 11.0),
            getattr(filters, "shock_daily_on_tr_pct", None),
        )
    elif detector == "daily_drawdown":
        shock_key = (
            mode,
            detector,
            dir_source,
            int(dir_lb),
            int(getattr(filters, "shock_drawdown_lookback_days", 20) or 20),
            float(getattr(filters, "shock_on_drawdown_pct", -20.0) or -20.0),
            float(getattr(filters, "shock_off_drawdown_pct", -10.0) or -10.0),
        )
    elif detector in ("atr_ratio", "tr_ratio"):
        shock_key = (
            mode,
            detector,
            dir_source,
            int(dir_lb),
            int(getattr(filters, "shock_atr_fast_period", 7) or 7),
            int(getattr(filters, "shock_atr_slow_period", 50) or 50),
            float(getattr(filters, "shock_on_ratio", 1.55) or 1.55),
            float(getattr(filters, "shock_off_ratio", 1.30) or 1.30),
            float(getattr(filters, "shock_min_atr_pct", 7.0) or 7.0),
            str(st_src_for_shock or "hl2"),
        )
    else:
        shock_key = (mode, detector, dir_source, int(dir_lb))

    if scale_detector is not None:
        scale_key: tuple[object, ...]
        if scale_detector == "daily_atr_pct":
            scale_key = (
                "scale",
                scale_detector,
                int(getattr(filters, "shock_daily_atr_period", 14) or 14),
                float(getattr(filters, "shock_daily_on_atr_pct", 13.0) or 13.0),
                float(getattr(filters, "shock_daily_off_atr_pct", 11.0) or 11.0),
                getattr(filters, "shock_daily_on_tr_pct", None),
            )
        elif scale_detector == "daily_drawdown":
            scale_key = (
                "scale",
                scale_detector,
                int(getattr(filters, "shock_drawdown_lookback_days", 20) or 20),
                float(getattr(filters, "shock_on_drawdown_pct", -20.0) or -20.0),
                float(getattr(filters, "shock_off_drawdown_pct", -10.0) or -10.0),
            )
        else:
            scale_key = (
                "scale",
                scale_detector,
                int(getattr(filters, "shock_atr_fast_period", 7) or 7),
                int(getattr(filters, "shock_atr_slow_period", 50) or 50),
                float(getattr(filters, "shock_on_ratio", 1.55) or 1.55),
                float(getattr(filters, "shock_off_ratio", 1.30) or 1.30),
                float(getattr(filters, "shock_min_atr_pct", 7.0) or 7.0),
                str(st_src_for_shock or "hl2"),
            )
        shock_key = (*shock_key, scale_key)

    cache_key = (id(exec_bars), id(signal_bars), id(regime_bars) if regime_bars else 0, shock_key)
    cached = _SERIES_CACHE.get(namespace=_SPOT_SHOCK_SERIES_NAMESPACE, key=cache_key)
    if isinstance(cached, _SpotShockSeries):
        return cached

    shock_by_sig_idx: list[bool | None] = [None] * len(signal_bars)
    shock_atr_pct_by_sig_idx: list[float | None] = [None] * len(signal_bars)
    shock_dir_by_sig_idx: list[str | None] = [None] * len(signal_bars)

    shock_by_exec_idx: list[bool | None] = [None] * len(exec_bars)
    shock_atr_pct_by_exec_idx: list[float | None] = [None] * len(exec_bars)
    shock_dir_by_exec_idx: list[str | None] = [None] * len(exec_bars)

    def _atr_pct_from(snap: object | None) -> float | None:
        if snap is None:
            return None
        atr_pct = getattr(snap, "atr_pct", None)
        if atr_pct is None:
            atr_pct = getattr(snap, "atr_fast_pct", None)
        if atr_pct is None:
            atr_pct = getattr(snap, "tr_fast_pct", None)
        try:
            return float(atr_pct) if atr_pct is not None else None
        except (TypeError, ValueError):
            return None

    def _scale_atr_pct_series() -> tuple[list[float | None], list[float | None]] | None:
        if scale_detector is None:
            return None

        scale_filters: dict[str, object] = {
            "shock_gate_mode": "detect",
            "shock_detector": str(scale_detector),
            "shock_direction_source": "regime",
            "shock_direction_lookback": int(dir_lb),
            "shock_atr_fast_period": int(getattr(filters, "shock_atr_fast_period", 7) or 7),
            "shock_atr_slow_period": int(getattr(filters, "shock_atr_slow_period", 50) or 50),
            "shock_on_ratio": float(getattr(filters, "shock_on_ratio", 1.55) or 1.55),
            "shock_off_ratio": float(getattr(filters, "shock_off_ratio", 1.30) or 1.30),
            "shock_min_atr_pct": float(getattr(filters, "shock_min_atr_pct", 7.0) or 7.0),
            "shock_daily_atr_period": int(getattr(filters, "shock_daily_atr_period", 14) or 14),
            "shock_daily_on_atr_pct": float(getattr(filters, "shock_daily_on_atr_pct", 13.0) or 13.0),
            "shock_daily_off_atr_pct": float(getattr(filters, "shock_daily_off_atr_pct", 11.0) or 11.0),
            "shock_daily_on_tr_pct": getattr(filters, "shock_daily_on_tr_pct", None),
            "shock_drawdown_lookback_days": int(getattr(filters, "shock_drawdown_lookback_days", 20) or 20),
            "shock_on_drawdown_pct": float(getattr(filters, "shock_on_drawdown_pct", -20.0) or -20.0),
            "shock_off_drawdown_pct": float(getattr(filters, "shock_off_drawdown_pct", -10.0) or -10.0),
        }
        scale_engine = build_shock_engine(scale_filters, source=str(st_src_for_shock or "hl2").strip().lower() or "hl2")
        if scale_engine is None:
            return None

        scale_atr_pct_by_sig_idx: list[float | None] = [None] * len(signal_bars)
        scale_atr_pct_by_exec_idx: list[float | None] = [None] * len(exec_bars)

        if scale_detector in ("daily_atr_pct", "daily_drawdown"):
            last = None
            for i, bar in enumerate(exec_bars):
                last = scale_engine.update(
                    day=_trade_date(bar.ts),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    update_direction=False,
                )
                scale_atr_pct_by_exec_idx[i] = _atr_pct_from(last)

            align = _spot_exec_alignment(signal_bars, exec_bars)
            for sig_i, _sig_bar in enumerate(signal_bars):
                exec_i = align.exec_idx_by_sig_idx[sig_i]
                if exec_i >= 0:
                    scale_atr_pct_by_sig_idx[sig_i] = scale_atr_pct_by_exec_idx[exec_i]
            return scale_atr_pct_by_sig_idx, scale_atr_pct_by_exec_idx

        last = None
        for sig_i, sig_bar in enumerate(signal_bars):
            last = scale_engine.update(
                high=float(sig_bar.high),
                low=float(sig_bar.low),
                close=float(sig_bar.close),
                update_direction=False,
            )
            ready = bool(getattr(last, "ready", False)) if last is not None else False
            if ready:
                scale_atr_pct_by_sig_idx[sig_i] = _atr_pct_from(last)

        # Propagate scale ATR% to every exec bar (constant between signal closes).
        align = _spot_exec_alignment(signal_bars, exec_bars)
        cur: float | None = None
        for exec_i, _bar in enumerate(exec_bars):
            sig_i = align.sig_idx_by_exec_idx[exec_i]
            if sig_i >= 0:
                cur = scale_atr_pct_by_sig_idx[sig_i]
            scale_atr_pct_by_exec_idx[exec_i] = cur

        return scale_atr_pct_by_sig_idx, scale_atr_pct_by_exec_idx

    # Daily detectors: shock is advanced by execution bars; optionally compute direction on signal closes.
    if detector in ("daily_atr_pct", "daily_drawdown"):
        shock_engine = build_shock_engine(filters, source=str(st_src_for_shock or "hl2").strip().lower() or "hl2")
        if shock_engine is None:
            out = _SpotShockSeries(
                shock_by_exec_idx=shock_by_exec_idx,
                shock_atr_pct_by_exec_idx=shock_atr_pct_by_exec_idx,
                shock_dir_by_exec_idx=shock_dir_by_exec_idx,
                shock_by_sig_idx=shock_by_sig_idx,
                shock_atr_pct_by_sig_idx=shock_atr_pct_by_sig_idx,
                shock_dir_by_sig_idx=shock_dir_by_sig_idx,
            )
            _SERIES_CACHE.set(namespace=_SPOT_SHOCK_SERIES_NAMESPACE, key=cache_key, value=out)
            return out

        # 1) advance shock state by exec bars
        last_shock = None
        for i, bar in enumerate(exec_bars):
            last_shock = shock_engine.update(
                day=_trade_date(bar.ts),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                update_direction=(dir_source != "signal"),
            )
            shock_by_exec_idx[i] = bool(getattr(last_shock, "shock", False))
            atr_pct = getattr(last_shock, "atr_pct", None)
            shock_atr_pct_by_exec_idx[i] = float(atr_pct) if atr_pct is not None else None
            if dir_source != "signal":
                if bool(getattr(last_shock, "direction_ready", False)) and getattr(last_shock, "direction", None) in ("up", "down"):
                    shock_dir_by_exec_idx[i] = str(getattr(last_shock, "direction"))
                else:
                    shock_dir_by_exec_idx[i] = None

        # 2) map shock state to signal bars (shock on/off comes from the exec bar at the same timestamp)
        align = _spot_exec_alignment(signal_bars, exec_bars)
        # Direction on signal closes when configured.
        dir_prev_close: float | None = None
        ret_hist: deque[float] = deque(maxlen=max(1, int(dir_lb)))
        direction: str | None = None
        for sig_i, sig_bar in enumerate(signal_bars):
            exec_i = align.exec_idx_by_sig_idx[sig_i]
            if exec_i >= 0:
                shock_by_sig_idx[sig_i] = bool(shock_by_exec_idx[exec_i]) if shock_by_exec_idx[exec_i] is not None else False
                shock_atr_pct_by_sig_idx[sig_i] = shock_atr_pct_by_exec_idx[exec_i]
            if dir_source == "signal":
                close = float(sig_bar.close)
                if dir_prev_close is not None and dir_prev_close > 0 and close > 0:
                    ret_hist.append((close / float(dir_prev_close)) - 1.0)
                    if len(ret_hist) >= max(1, int(dir_lb)):
                        ret_sum = float(sum(ret_hist))
                        if ret_sum > 0:
                            direction = "up"
                        elif ret_sum < 0:
                            direction = "down"
                dir_prev_close = float(close)
                ready = bool(direction in ("up", "down") and len(ret_hist) >= max(1, int(dir_lb)))
                shock_dir_by_sig_idx[sig_i] = str(direction) if ready else None
            elif exec_i >= 0:
                shock_dir_by_sig_idx[sig_i] = shock_dir_by_exec_idx[exec_i]

        # 3) propagate direction to exec bars (direction only changes on signal closes in this mode)
        if dir_source == "signal":
            cur_dir: str | None = None
            for exec_i, _bar in enumerate(exec_bars):
                sig_i = align.sig_idx_by_exec_idx[exec_i]
                shock_dir_by_exec_idx[exec_i] = cur_dir
                if sig_i >= 0:
                    # Direction is updated on the signal close, but it should only affect
                    # *subsequent* execution bars (SpotSignalEvaluator updates direction in
                    # `update_signal_bar`, which is called after `update_exec_bar` for this bar).
                    cur_dir = shock_dir_by_sig_idx[sig_i]

        scale_series = _scale_atr_pct_series()
        if scale_series is not None:
            shock_atr_pct_by_sig_idx, shock_atr_pct_by_exec_idx = scale_series

        out = _SpotShockSeries(
            shock_by_exec_idx=shock_by_exec_idx,
            shock_atr_pct_by_exec_idx=shock_atr_pct_by_exec_idx,
            shock_dir_by_exec_idx=shock_dir_by_exec_idx,
            shock_by_sig_idx=shock_by_sig_idx,
            shock_atr_pct_by_sig_idx=shock_atr_pct_by_sig_idx,
            shock_dir_by_sig_idx=shock_dir_by_sig_idx,
        )
        _SERIES_CACHE.set(namespace=_SPOT_SHOCK_SERIES_NAMESPACE, key=cache_key, value=out)
        return out

    # Non-daily detectors: shock is advanced on regime bars (if provided) and read on signal closes.
    shock_engine = build_shock_engine(filters, source=str(st_src_for_shock or "hl2").strip().lower() or "hl2")
    if shock_engine is None:
        out = _SpotShockSeries(
            shock_by_exec_idx=shock_by_exec_idx,
            shock_atr_pct_by_exec_idx=shock_atr_pct_by_exec_idx,
            shock_dir_by_exec_idx=shock_dir_by_exec_idx,
            shock_by_sig_idx=shock_by_sig_idx,
            shock_atr_pct_by_sig_idx=shock_atr_pct_by_sig_idx,
            shock_dir_by_sig_idx=shock_dir_by_sig_idx,
        )
        _SERIES_CACHE.set(namespace=_SPOT_SHOCK_SERIES_NAMESPACE, key=cache_key, value=out)
        return out

    use_mtf = bool(regime_bars)
    reg_idx = 0
    last_shock = None
    for sig_i, sig_bar in enumerate(signal_bars):
        if use_mtf and regime_bars:
            while reg_idx < len(regime_bars) and regime_bars[reg_idx].ts <= sig_bar.ts:
                reg_bar = regime_bars[reg_idx]
                last_shock = shock_engine.update(
                    high=float(reg_bar.high),
                    low=float(reg_bar.low),
                    close=float(reg_bar.close),
                    update_direction=(dir_source != "signal"),
                )
                reg_idx += 1
        else:
            last_shock = shock_engine.update(
                high=float(sig_bar.high),
                low=float(sig_bar.low),
                close=float(sig_bar.close),
                update_direction=(dir_source != "signal"),
            )

        # Special-case: mtf ATR-ratio + direction_source=signal updates direction on the signal closes.
        if (
            detector == "atr_ratio"
            and use_mtf
            and dir_source == "signal"
            and hasattr(shock_engine, "update_direction")
        ):
            last_shock = shock_engine.update_direction(close=float(sig_bar.close))

        ready = bool(getattr(last_shock, "ready", False)) if last_shock is not None else False
        if not ready:
            shock_by_sig_idx[sig_i] = None
            shock_atr_pct_by_sig_idx[sig_i] = None
            shock_dir_by_sig_idx[sig_i] = None
            continue

        shock_by_sig_idx[sig_i] = bool(getattr(last_shock, "shock", False))
        atr_pct = getattr(last_shock, "atr_pct", None)
        if atr_pct is None:
            atr_pct = getattr(last_shock, "atr_fast_pct", None)
        if atr_pct is None:
            atr_pct = getattr(last_shock, "tr_fast_pct", None)
        shock_atr_pct_by_sig_idx[sig_i] = float(atr_pct) if atr_pct is not None else None
        if bool(getattr(last_shock, "direction_ready", False)) and getattr(last_shock, "direction", None) in ("up", "down"):
            shock_dir_by_sig_idx[sig_i] = str(getattr(last_shock, "direction"))
        else:
            shock_dir_by_sig_idx[sig_i] = None

    # Propagate shock view to every exec bar (constant between signal closes).
    align = _spot_exec_alignment(signal_bars, exec_bars)
    cur_shock = None
    cur_atr_pct = None
    cur_dir = None
    sig_ptr = 0
    for exec_i, _bar in enumerate(exec_bars):
        sig_i = align.sig_idx_by_exec_idx[exec_i]
        if sig_i >= 0:
            cur_shock = shock_by_sig_idx[sig_i]
            cur_atr_pct = shock_atr_pct_by_sig_idx[sig_i]
            cur_dir = shock_dir_by_sig_idx[sig_i]
        shock_by_exec_idx[exec_i] = cur_shock
        shock_atr_pct_by_exec_idx[exec_i] = cur_atr_pct
        shock_dir_by_exec_idx[exec_i] = cur_dir

    scale_series = _scale_atr_pct_series()
    if scale_series is not None:
        shock_atr_pct_by_sig_idx, shock_atr_pct_by_exec_idx = scale_series

    out = _SpotShockSeries(
        shock_by_exec_idx=shock_by_exec_idx,
        shock_atr_pct_by_exec_idx=shock_atr_pct_by_exec_idx,
        shock_dir_by_exec_idx=shock_dir_by_exec_idx,
        shock_by_sig_idx=shock_by_sig_idx,
        shock_atr_pct_by_sig_idx=shock_atr_pct_by_sig_idx,
        shock_dir_by_sig_idx=shock_dir_by_sig_idx,
    )
    _SERIES_CACHE.set(namespace=_SPOT_SHOCK_SERIES_NAMESPACE, key=cache_key, value=out)
    return out


def _spot_dd_trees(*, exec_bars: list[Bar], spread: float, mark_to_market: str) -> tuple[_SpotDdTreeLong, _SpotDdTreeShort]:
    key = (id(exec_bars), float(spread), str(mark_to_market or "close"))
    cached = _SERIES_CACHE.get(namespace=_SPOT_DD_TREES_NAMESPACE, key=key)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached[0], cached[1]  # type: ignore[return-value]

    half = max(0.0, float(spread)) / 2.0 if str(mark_to_market).strip().lower() == "liquidation" else 0.0
    close_long: list[float] = []
    low_long: list[float] = []
    close_short: list[float] = []
    high_short: list[float] = []
    for bar in exec_bars:
        c = float(bar.close)
        close_long.append(c - half)
        low_long.append(float(bar.low) - half)
        close_short.append(c + half)
        high_short.append(float(bar.high) + half)

    long_tree = _SpotDdTreeLong(close_vals=close_long, low_vals=low_long)
    short_tree = _SpotDdTreeShort(close_vals=close_short, high_vals=high_short)
    _SERIES_CACHE.set(namespace=_SPOT_DD_TREES_NAMESPACE, key=key, value=(long_tree, short_tree))
    return long_tree, short_tree


def _spot_effective_pct(base_pct: float | None, *, mult: float, shock_on: bool) -> float | None:
    if base_pct is None:
        return None
    pct = float(base_pct)
    if pct <= 0:
        return None
    if bool(shock_on):
        pct *= float(mult)
    if pct <= 0:
        return None
    return min(float(pct), 0.99)


def _spot_effective_pct_by_exec_idx(
    *,
    shock_by_exec_idx: list[bool | None],
    base_pct: float | None,
    mult: float,
) -> list[float | None]:
    out: list[float | None] = [None] * len(shock_by_exec_idx)
    for i in range(len(shock_by_exec_idx)):
        prev = shock_by_exec_idx[i - 1] if i > 0 else None
        shock_on = bool(prev) if prev is not None else False
        out[i] = _spot_effective_pct(base_pct, mult=float(mult), shock_on=bool(shock_on))
    return out


def _spot_stop_tree(
    *,
    exec_bars: list[Bar],
    shock: _SpotShockSeries,
    base_stop_pct: float,
    shock_stop_mult: float,
    direction: str,
    mode: str,
) -> object:
    """Build a stop-hit search tree for a given stop configuration.

    - direction: "up" (long) uses lows; "down" (short) uses highs.
    - mode: "intrabar" only (this is the only supported mode for the fast runner).
    """
    cache_key = (
        id(exec_bars),
        id(shock.shock_by_exec_idx),
        float(base_stop_pct),
        float(shock_stop_mult),
        str(direction),
        str(mode),
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_STOP_TREE_NAMESPACE, key=cache_key)
    if cached is not None:
        return cached

    stop_pct = float(base_stop_pct)
    if stop_pct <= 0 or stop_pct >= 0.99:
        tree: object = _MinFirstLeqTree([])  # unused, but keeps types simple
        _SERIES_CACHE.set(namespace=_SPOT_STOP_TREE_NAMESPACE, key=cache_key, value=tree)
        return tree

    mult = float(shock_stop_mult)
    if mult <= 0:
        mult = 1.0

    shock_by_exec = shock.shock_by_exec_idx
    stop_pct_by_exec = _spot_effective_pct_by_exec_idx(
        shock_by_exec_idx=shock_by_exec,
        base_pct=float(stop_pct),
        mult=float(mult),
    )
    if str(direction) == "up":
        triggers: list[float] = []
        for i, bar in enumerate(exec_bars):
            pct_eff = float(stop_pct_by_exec[i] if stop_pct_by_exec[i] is not None else 0.0)
            denom = max(1e-9, 1.0 - float(pct_eff))
            triggers.append(float(bar.low) / denom)
        tree = _MinFirstLeqTree(triggers)
    else:
        triggers = []
        for i, bar in enumerate(exec_bars):
            pct_eff = float(stop_pct_by_exec[i] if stop_pct_by_exec[i] is not None else 0.0)
            denom = 1.0 + float(pct_eff)
            denom = max(1e-9, float(denom))
            triggers.append(float(bar.high) / denom)
        tree = _MaxFirstGeTree(triggers)

    _SERIES_CACHE.set(namespace=_SPOT_STOP_TREE_NAMESPACE, key=cache_key, value=tree)
    return tree


def _spot_profit_tree(
    *,
    exec_bars: list[Bar],
    shock: _SpotShockSeries,
    base_profit_pct: float,
    shock_profit_mult: float,
    direction: str,
    mode: str,
) -> object:
    """Build a profit-hit search tree for a given profit-target configuration.

    - direction: "up" (long) uses highs; "down" (short) uses lows.
    - mode: "intrabar" only (this is the only supported mode for the fast runner).
    """
    cache_key = (
        id(exec_bars),
        id(shock.shock_by_exec_idx),
        float(base_profit_pct),
        float(shock_profit_mult),
        str(direction),
        str(mode),
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_PROFIT_TREE_NAMESPACE, key=cache_key)
    if cached is not None:
        return cached

    pt_pct = float(base_profit_pct)
    if pt_pct <= 0 or pt_pct >= 0.99:
        tree: object = _MaxFirstGeTree([])  # unused, but keeps types simple
        _SERIES_CACHE.set(namespace=_SPOT_PROFIT_TREE_NAMESPACE, key=cache_key, value=tree)
        return tree

    mult = float(shock_profit_mult)
    if mult <= 0:
        mult = 1.0

    shock_by_exec = shock.shock_by_exec_idx
    profit_pct_by_exec = _spot_effective_pct_by_exec_idx(
        shock_by_exec_idx=shock_by_exec,
        base_pct=float(pt_pct),
        mult=float(mult),
    )
    if str(direction) == "up":
        triggers: list[float] = []
        for i, bar in enumerate(exec_bars):
            pct_eff = float(profit_pct_by_exec[i] if profit_pct_by_exec[i] is not None else 0.0)
            denom = 1.0 + float(pct_eff)
            denom = max(1e-9, float(denom))
            triggers.append(float(bar.high) / denom)
        tree = _MaxFirstGeTree(triggers)
    else:
        triggers = []
        for i, bar in enumerate(exec_bars):
            pct_eff = float(profit_pct_by_exec[i] if profit_pct_by_exec[i] is not None else 0.0)
            denom = max(1e-9, 1.0 - float(pct_eff))
            triggers.append(float(bar.low) / denom)
        tree = _MinFirstLeqTree(triggers)

    _SERIES_CACHE.set(namespace=_SPOT_PROFIT_TREE_NAMESPACE, key=cache_key, value=tree)
    return tree


# endregion


# region Public API
@dataclass(frozen=True)
class BacktestResult:
    trades: list[OptionTrade | SpotTrade]
    equity: list[EquityPoint]
    summary: SummaryStats
    lifecycle_trace: list[dict[str, object]] | None = None


def _load_backtest_series(
    *,
    data: IBKRHistoricalData,
    cfg: ConfigBundle,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> BarSeries[Bar]:
    loader = data.load_cached_bar_series if bool(cfg.backtest.offline) else data.load_or_fetch_bar_series
    return loader(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=str(bar_size),
        use_rth=bool(use_rth),
        cache_dir=cfg.backtest.cache_dir,
    )


def _load_backtest_bars(
    *,
    data: IBKRHistoricalData,
    cfg: ConfigBundle,
    symbol: str,
    exchange: str | None,
    start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> list[Bar]:
    return _load_backtest_series(
        data=data,
        cfg=cfg,
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end,
        bar_size=bar_size,
        use_rth=use_rth,
    ).as_list()


def _load_backtest_bars_offline_fallback_start(
    *,
    data: IBKRHistoricalData,
    cfg: ConfigBundle,
    symbol: str,
    exchange: str | None,
    start: datetime,
    fallback_start: datetime,
    end: datetime,
    bar_size: str,
    use_rth: bool,
) -> BarSeries[Bar]:
    if not bool(cfg.backtest.offline) or start == fallback_start:
        return _load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
    try:
        return _load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )
    except FileNotFoundError:
        return _load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=symbol,
            exchange=exchange,
            start=fallback_start,
            end=end,
            bar_size=bar_size,
            use_rth=use_rth,
        )


def _resolve_backtest_contract_meta(*, data: IBKRHistoricalData, cfg: ConfigBundle) -> ContractMeta:
    is_future = cfg.strategy.symbol in ("MNQ", "MBT")
    if cfg.backtest.offline:
        exchange = "CME" if is_future else "SMART"
        if cfg.strategy.instrument == "spot":
            multiplier = _spot_multiplier(cfg.strategy.symbol, is_future)
        else:
            multiplier = 1.0 if is_future else 100.0
        return ContractMeta(symbol=cfg.strategy.symbol, exchange=exchange, multiplier=multiplier, min_tick=0.01)

    _, resolved = data.resolve_contract(cfg.strategy.symbol, cfg.strategy.exchange)
    if cfg.strategy.instrument == "spot":
        return ContractMeta(
            symbol=resolved.symbol,
            exchange=resolved.exchange,
            multiplier=_spot_multiplier(cfg.strategy.symbol, is_future, default=resolved.multiplier),
            min_tick=resolved.min_tick,
        )
    if (not is_future) and resolved.exchange == "SMART":
        return ContractMeta(
            symbol=resolved.symbol,
            exchange=resolved.exchange,
            multiplier=100.0,
            min_tick=resolved.min_tick,
        )
    return resolved


def _load_spot_backtest_context_bars(
    *,
    data: IBKRHistoricalData,
    cfg: ConfigBundle,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[BarSeries[Bar] | None, BarSeries[Bar] | None, BarSeries[Bar] | None, BarSeries[Bar] | None]:
    def _load_requirement(req: SpotBarRequirement, req_start: datetime, req_end: datetime):
        if str(req.kind) == "regime":
            return _load_backtest_bars_offline_fallback_start(
                data=data,
                cfg=cfg,
                symbol=req.symbol,
                exchange=req.exchange,
                start=req_start,
                fallback_start=start_dt,
                end=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
            )
        return _load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=req.symbol,
            exchange=req.exchange,
            start=req_start,
            end=req_end,
            bar_size=str(req.bar_size),
            use_rth=bool(req.use_rth),
        )

    context = load_spot_context_bars(
        strategy=cfg.strategy,
        default_symbol=str(cfg.strategy.symbol),
        default_exchange=cfg.strategy.exchange,
        default_signal_bar_size=str(cfg.backtest.bar_size),
        default_signal_use_rth=bool(cfg.backtest.use_rth),
        start_dt=start_dt,
        end_dt=end_dt,
        load_requirement=_load_requirement,
    )
    return context.regime_bars, context.regime2_bars, context.tick_bars, context.exec_bars


def run_backtest(cfg: ConfigBundle) -> BacktestResult:
    data = IBKRHistoricalData()
    start_dt = datetime.combine(cfg.backtest.start, time(0, 0))
    end_dt = datetime.combine(cfg.backtest.end, time(23, 59))
    bar_series = _load_backtest_series(
        data=data,
        cfg=cfg,
        symbol=cfg.strategy.symbol,
        exchange=cfg.strategy.exchange,
        start=start_dt,
        end=end_dt,
        bar_size=cfg.backtest.bar_size,
        use_rth=cfg.backtest.use_rth,
    )
    bars = bar_series.as_list()
    if not bars:
        raise RuntimeError("No bars loaded for backtest")
    meta = _resolve_backtest_contract_meta(data=data, cfg=cfg)

    if cfg.strategy.instrument == "spot":
        regime_bars, regime2_bars, tick_bars, exec_bars = _load_spot_backtest_context_bars(
            data=data,
            cfg=cfg,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        result = _run_spot_backtest(
            cfg,
            bar_series,
            meta,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
            exec_bars=exec_bars,
        )
        data.disconnect()
        return result

    from .engine_options import run_options_backtest

    result = run_options_backtest(
        cfg=cfg,
        bars=bars,
        meta=meta,
        data=data,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    data.disconnect()
    return result

# endregion


# region Spot Backtest
def _spot_multiplier(symbol: str, is_future: bool, default: float = 1.0) -> float:
    if not is_future:
        return 1.0
    overrides = {
        "MNQ": 2.0,  # Micro E-mini Nasdaq-100
        "MBT": 0.1,  # Micro Bitcoin (0.1 BTC)
    }
    return overrides.get(symbol, default if default > 0 else 1.0)


def _spot_liquidation_value(
    *,
    open_trades: list[SpotTrade],
    ref_price: float,
    spread: float,
    mark_to_market: str,
    multiplier: float,
) -> float:
    total = 0.0
    for trade in open_trades:
        total += (
            trade.qty
            * _spot_mark_price(float(ref_price), qty=trade.qty, spread=spread, mode=mark_to_market)
            * float(multiplier)
        )
    return float(total)


def _spot_riskoff_end_hour(filters) -> int | None:
    return spot_riskoff_end_hour(filters)


def _spot_apply_exit_accounting(
    *,
    cash: float,
    margin_used: float,
    qty: int,
    exit_price: float,
    margin_required: float,
    multiplier: float,
) -> tuple[float, float]:
    next_cash = float(cash) + (int(qty) * float(exit_price)) * float(multiplier)
    next_margin = max(0.0, float(margin_used) - float(margin_required))
    return float(next_cash), float(next_margin)


def _spot_exec_exit_and_account(
    *,
    qty: int,
    exit_ref_price: float,
    exit_time: datetime,
    reason: str,
    cash: float,
    margin_used: float,
    margin_required: float,
    spread: float,
    commission_per_share: float,
    commission_min: float,
    slippage_per_share: float,
    multiplier: float,
    apply_slippage: bool = True,
    trade: SpotTrade | None = None,
    trades: list[SpotTrade] | None = None,
) -> tuple[float, float, float]:
    side = "sell" if int(qty) > 0 else "buy"
    exit_price = _spot_exec_price(
        float(exit_ref_price),
        side=side,
        qty=int(qty),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
        apply_slippage=bool(apply_slippage),
    )
    if trade is not None and trades is not None:
        _close_spot_trade(trade, exit_time, float(exit_price), reason, trades)
    next_cash, next_margin = _spot_apply_exit_accounting(
        cash=float(cash),
        margin_used=float(margin_used),
        qty=int(qty),
        exit_price=float(exit_price),
        margin_required=float(margin_required),
        multiplier=float(multiplier),
    )
    return float(exit_price), float(next_cash), float(next_margin)


def _spot_exec_exit_common(
    *,
    qty: int,
    margin_required: float,
    exit_ref_price: float,
    exit_time: datetime,
    reason: str,
    cash: float,
    margin_used: float,
    spread: float,
    commission_per_share: float,
    commission_min: float,
    slippage_per_share: float,
    multiplier: float,
    apply_slippage: bool | None = None,
    trade: SpotTrade | None = None,
    trades: list[SpotTrade] | None = None,
) -> tuple[float, float, float]:
    apply_slippage_eff = (str(reason) != "profit") if apply_slippage is None else bool(apply_slippage)
    return _spot_exec_exit_and_account(
        qty=int(qty),
        exit_ref_price=float(exit_ref_price),
        exit_time=exit_time,
        reason=str(reason),
        cash=float(cash),
        margin_used=float(margin_used),
        margin_required=float(margin_required),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
        multiplier=float(multiplier),
        apply_slippage=bool(apply_slippage_eff),
        trade=trade,
        trades=trades,
    )


def _spot_entry_accounting(
    *,
    cash: float,
    margin_used: float,
    signed_qty: int,
    entry_price: float,
    mark_ref_price: float,
    liquidation_value: float,
    spread: float,
    mark_to_market: str,
    multiplier: float,
) -> tuple[bool, float, float, float]:
    margin_required = abs(int(signed_qty) * float(entry_price)) * float(multiplier)
    cash_after = float(cash) - (int(signed_qty) * float(entry_price)) * float(multiplier)
    margin_after = float(margin_used) + float(margin_required)
    candidate_mark = (
        int(signed_qty)
        * _spot_mark_price(float(mark_ref_price), qty=int(signed_qty), spread=float(spread), mode=str(mark_to_market))
        * float(multiplier)
    )
    equity_after = float(cash_after) + float(liquidation_value) + float(candidate_mark)
    ok = float(cash_after) >= 0.0 and float(equity_after) >= float(margin_after)
    return bool(ok), float(cash_after), float(margin_after), float(margin_required)


def _spot_entry_leg_for_direction(
    *,
    strategy,
    entry_dir: str | None,
    needs_direction: bool,
) -> SpotLegConfig | None:
    resolved = spot_resolve_entry_action_qty(
        strategy=strategy,
        entry_dir=entry_dir,
        needs_direction=needs_direction,
        fallback_short_sell=False,
    )
    if resolved is None:
        return None
    action, qty = resolved
    return SpotLegConfig(action=str(action), qty=max(1, int(qty)))


def _spot_branch_size_mult(*, strategy, entry_branch: str | None) -> float:
    return spot_branch_size_mult(strategy=strategy, entry_branch=entry_branch)


def _spot_next_open_entry_allowed(
    *,
    signal_ts: datetime,
    next_ts: datetime,
    riskoff_today: bool,
    riskoff_end_hour: int | None,
    exit_mode: str,
    atr_value: float | None,
) -> bool:
    return lifecycle_next_open_entry_allowed(
        signal_ts=signal_ts,
        next_ts=next_ts,
        riskoff_today=bool(riskoff_today),
        riskoff_end_hour=riskoff_end_hour,
        exit_mode=str(exit_mode),
        atr_value=float(atr_value) if atr_value is not None else None,
    )


def _spot_strategy_sec_type(*, strategy) -> str:
    raw = str(getattr(strategy, "spot_sec_type", "") or "").strip().upper()
    if raw:
        return raw
    symbol = str(getattr(strategy, "symbol", "") or "").strip().upper()
    if symbol in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
        return "FUT"
    return "STK"


def _spot_next_open_due_ts(
    *,
    strategy,
    signal_close_ts: datetime,
    exec_bar_size: str,
    signal_use_rth: bool | None = None,
    spot_sec_type: str | None = None,
) -> datetime:
    if isinstance(strategy, Mapping):
        mode_raw = strategy.get("spot_next_open_session")
        use_rth_raw = strategy.get("signal_use_rth", True)
        sec_type_raw = strategy.get("spot_sec_type")
    else:
        mode_raw = getattr(strategy, "spot_next_open_session", None)
        use_rth_raw = getattr(strategy, "signal_use_rth", True)
        sec_type_raw = getattr(strategy, "spot_sec_type", None)
    strategy_view = {
        "spot_next_open_session": mode_raw,
        "signal_use_rth": (
            bool(signal_use_rth)
            if signal_use_rth is not None
            else bool(use_rth_raw)
        ),
        "spot_sec_type": str(spot_sec_type or sec_type_raw or _spot_strategy_sec_type(strategy=strategy)),
    }
    return lifecycle_next_open_due_ts(
        signal_close_ts=signal_close_ts,
        exec_bar_size=str(exec_bar_size or ""),
        strategy=strategy_view,
        naive_ts_mode="utc",
    )


def _spot_fill_due_ts(
    *,
    strategy,
    fill_mode: str,
    signal_close_ts: datetime,
    exec_bar_size: str,
    signal_use_rth: bool | None = None,
    spot_sec_type: str | None = None,
) -> datetime | None:
    if isinstance(strategy, Mapping):
        mode_raw = strategy.get("spot_next_open_session")
        use_rth_raw = strategy.get("signal_use_rth", True)
        sec_type_raw = strategy.get("spot_sec_type")
    else:
        mode_raw = getattr(strategy, "spot_next_open_session", None)
        use_rth_raw = getattr(strategy, "signal_use_rth", True)
        sec_type_raw = getattr(strategy, "spot_sec_type", None)
    strategy_view = {
        "spot_next_open_session": mode_raw,
        "signal_use_rth": (
            bool(signal_use_rth)
            if signal_use_rth is not None
            else bool(use_rth_raw)
        ),
        "spot_sec_type": str(spot_sec_type or sec_type_raw or _spot_strategy_sec_type(strategy=strategy)),
    }
    return lifecycle_fill_due_ts(
        fill_mode=normalize_spot_fill_mode(fill_mode, default=SPOT_FILL_MODE_CLOSE),
        signal_close_ts=signal_close_ts,
        exec_bar_size=str(exec_bar_size or ""),
        strategy=strategy_view,
        naive_ts_mode="utc",
    )


def _spot_entry_capacity_ok(
    *,
    open_count: int,
    max_entries_per_day: int,
    entries_today: int,
    weekday: int,
    entry_days: tuple[int, ...] | list[int],
) -> bool:
    return lifecycle_entry_capacity_ok(
        open_count=int(open_count),
        max_entries_per_day=int(max_entries_per_day),
        entries_today=int(entries_today),
        weekday=int(weekday),
        entry_days=entry_days,
    )


def _spot_flat_entry_intent(
    *,
    strategy,
    bar_ts: datetime,
    direction: str | None,
    filters_ok: bool,
    entry_capacity: bool,
    pending_exists: bool,
    next_open_allowed: bool,
    can_order_now: bool,
    preflight_ok: bool,
    stale_signal: bool,
    gap_signal: bool,
    exit_mode: str,
    atr_value: float | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
):
    atr_ready = bool(str(exit_mode) != "atr" or (atr_value is not None and float(atr_value) > 0.0))
    return decide_flat_position_intent(
        strategy=strategy,
        bar_ts=bar_ts,
        entry_dir=direction,
        allowed_directions=("up", "down"),
        can_order_now=bool(can_order_now),
        preflight_ok=bool(preflight_ok),
        filters_ok=bool(filters_ok),
        entry_capacity=bool(entry_capacity),
        stale_signal=bool(stale_signal),
        gap_signal=bool(gap_signal),
        pending_exists=bool(pending_exists),
        atr_ready=bool(atr_ready),
        next_open_allowed=bool(next_open_allowed),
        shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
        shock_atr_vel_pct=float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
        shock_atr_accel_pct=float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
        tr_ratio=float(tr_ratio) if tr_ratio is not None else None,
        tr_median_pct=float(tr_median_pct) if tr_median_pct is not None else None,
        slope_med_pct=float(slope_med_pct) if slope_med_pct is not None else None,
        slope_vel_pct=float(slope_vel_pct) if slope_vel_pct is not None else None,
        slope_med_slow_pct=float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
        slope_vel_slow_pct=float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
    )


def _spot_flat_entry_decision_from_signal(
    *,
    strategy,
    filters,
    signal_bar: Bar,
    signal: EmaDecisionSnapshot | None,
    direction: str | None,
    bars_in_day: int,
    volume_ema: float | None,
    volume_ema_ready: bool,
    rv: float | None,
    cooldown_ok: bool,
    shock: bool | None,
    shock_dir: str | None,
    open_count: int,
    entries_today: int,
    pending_exists: bool,
    next_open_allowed: bool,
    exit_mode: str,
    atr_value: float | None,
    shock_atr_pct: float | None,
    shock_atr_vel_pct: float | None,
    shock_atr_accel_pct: float | None,
    tr_ratio: float | None,
    tr_median_pct: float | None,
    slope_med_pct: float | None,
    slope_vel_pct: float | None,
    slope_med_slow_pct: float | None,
    slope_vel_slow_pct: float | None,
):
    filters_ok = lifecycle_signal_filters_ok(
        filters,
        bar_ts=signal_bar.ts,
        bars_in_day=int(bars_in_day),
        close=float(signal_bar.close),
        volume=float(signal_bar.volume),
        volume_ema=float(volume_ema) if volume_ema is not None else None,
        volume_ema_ready=bool(volume_ema_ready),
        rv=float(rv) if rv is not None else None,
        signal=signal,
        cooldown_ok=bool(cooldown_ok),
        shock=shock,
        shock_dir=shock_dir,
    )
    entry_capacity = _spot_entry_capacity_ok(
        open_count=int(open_count),
        max_entries_per_day=int(getattr(strategy, "max_entries_per_day", 0) or 0),
        entries_today=int(entries_today),
        weekday=_trade_weekday(signal_bar.ts),
        entry_days=getattr(strategy, "entry_days", ()),
    )
    return _spot_flat_entry_intent(
        strategy=strategy,
        bar_ts=signal_bar.ts,
        direction=direction,
        filters_ok=bool(filters_ok),
        entry_capacity=bool(entry_capacity),
        pending_exists=bool(pending_exists),
        next_open_allowed=bool(next_open_allowed),
        can_order_now=True,
        preflight_ok=True,
        stale_signal=False,
        gap_signal=False,
        exit_mode=str(exit_mode),
        atr_value=float(atr_value) if atr_value is not None else None,
        shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
        shock_atr_vel_pct=float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
        shock_atr_accel_pct=float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
        tr_ratio=float(tr_ratio) if tr_ratio is not None else None,
        tr_median_pct=float(tr_median_pct) if tr_median_pct is not None else None,
        slope_med_pct=float(slope_med_pct) if slope_med_pct is not None else None,
        slope_vel_pct=float(slope_vel_pct) if slope_vel_pct is not None else None,
        slope_med_slow_pct=float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
        slope_vel_slow_pct=float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
    )


def _spot_open_position_intent(
    *,
    strategy,
    bar_ts: datetime,
    bar_size: str,
    open_dir: str | None,
    current_qty: int,
    exit_candidates: dict[str, bool] | None = None,
    exit_priority: tuple[str, ...] | list[str] | None = None,
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
):
    return decide_open_position_intent(
        strategy=strategy,
        bar_ts=bar_ts,
        bar_size=str(bar_size),
        open_dir=str(open_dir) if open_dir in ("up", "down") else None,
        current_qty=int(current_qty),
        exit_candidates=exit_candidates,
        exit_priority=exit_priority,
        target_qty=int(target_qty) if target_qty is not None else None,
        spot_decision=spot_decision,
        last_resize_bar_ts=last_resize_bar_ts,
        signal_entry_dir=str(signal_entry_dir) if signal_entry_dir in ("up", "down") else None,
        shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
        shock_atr_vel_pct=float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
        shock_atr_accel_pct=float(shock_atr_accel_pct) if shock_atr_accel_pct is not None else None,
        tr_ratio=float(tr_ratio) if tr_ratio is not None else None,
        tr_median_pct=float(tr_median_pct) if tr_median_pct is not None else None,
        slope_med_pct=float(slope_med_pct) if slope_med_pct is not None else None,
        slope_vel_pct=float(slope_vel_pct) if slope_vel_pct is not None else None,
        slope_med_slow_pct=float(slope_med_slow_pct) if slope_med_slow_pct is not None else None,
        slope_vel_slow_pct=float(slope_vel_slow_pct) if slope_vel_slow_pct is not None else None,
    )


def _spot_exit_reason_from_lifecycle(
    *,
    lifecycle,
    exit_candidates: dict[str, bool] | None,
    fallback_priority: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    if str(getattr(lifecycle, "intent", "") or "").strip().lower() == "exit":
        reason = str(getattr(lifecycle, "reason", "") or "").strip()
        if reason:
            return reason
    if not isinstance(exit_candidates, dict):
        return None
    for raw_reason in tuple(fallback_priority or ()):
        reason = str(raw_reason)
        if bool(exit_candidates.get(reason, False)):
            return reason
    for raw_reason, raw_hit in exit_candidates.items():
        if bool(raw_hit):
            return str(raw_reason)
    return None


def _spot_pending_entry_should_cancel(
    *,
    pending_dir: str,
    pending_set_date: date | None,
    exec_ts: datetime,
    risk_overlay_enabled: bool,
    riskoff_today: bool,
    riskpanic_today: bool,
    riskpop_today: bool,
    riskoff_mode: str,
    shock_dir_now: str | None,
    riskoff_end_hour: int | None,
) -> bool:
    return spot_pending_entry_should_cancel(
        pending_dir=pending_dir,
        pending_set_date=pending_set_date,
        exec_ts=exec_ts,
        risk_overlay_enabled=risk_overlay_enabled,
        riskoff_today=riskoff_today,
        riskpanic_today=riskpanic_today,
        riskpop_today=riskpop_today,
        riskoff_mode=riskoff_mode,
        shock_dir_now=shock_dir_now,
        riskoff_end_hour=riskoff_end_hour,
    )


def _spot_try_open_entry(
    *,
    cfg: ConfigBundle,
    meta: ContractMeta,
    entry_signal: str,
    entry_dir: str,
    entry_branch: str | None,
    entry_leg: SpotLegConfig,
    entry_time: datetime,
    entry_ref_price: float,
    mark_ref_price: float,
    atr_value: float | None,
    exit_mode: str,
    orb_engine,
    filters,
    shock_now: bool | None,
    shock_dir_now: str | None,
    shock_atr_pct_now: float | None,
    shock_dir_down_streak_bars_now: int | None,
    shock_drawdown_dist_on_pct_now: float | None,
    shock_drawdown_dist_on_vel_pp_now: float | None,
    shock_drawdown_dist_on_accel_pp_now: float | None,
    shock_prearm_down_streak_bars_now: int | None,
    shock_ramp_now: dict[str, object] | None = None,
    signal_entry_dir_now: str | None,
    signal_regime_dir_now: str | None,
    riskoff_today: bool,
    riskpanic_today: bool,
    riskpop_today: bool,
    risk_snapshot,
    cash: float,
    margin_used: float,
    liquidation_value: float,
    spread: float,
    commission_per_share: float,
    commission_min: float,
    slippage_per_share: float,
    mark_to_market: str,
) -> tuple[SpotTrade, float, float] | None:
    action = str(getattr(entry_leg, "action", "BUY") or "BUY").strip().upper()
    side = "buy" if action == "BUY" else "sell"
    lot = max(1, int(getattr(entry_leg, "qty", 1) or 1))
    base_signed_qty = int(lot) * int(cfg.strategy.quantity)
    if action != "BUY":
        base_signed_qty = -base_signed_qty

    entry_price_est = _spot_exec_price(
        float(entry_ref_price),
        side=side,
        qty=int(base_signed_qty),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
    )

    can_open = True
    target_price: float | None = None
    stop_price: float | None = None
    profit_target_pct = cfg.strategy.spot_profit_target_pct
    stop_loss_pct = cfg.strategy.spot_stop_loss_pct

    if entry_signal == "orb" and orb_engine is not None and entry_dir in ("up", "down"):
        orb_high = orb_engine.or_high
        orb_low = orb_engine.or_low
        if orb_high is not None and orb_low is not None and orb_high > 0 and orb_low > 0:
            stop_price = float(orb_low) if entry_dir == "up" else float(orb_high)
            rr = float(getattr(cfg.strategy, "orb_risk_reward", 2.0) or 2.0)
            target_mode = str(getattr(cfg.strategy, "orb_target_mode", "rr") or "rr").strip().lower()
            if target_mode not in ("rr", "or_range"):
                target_mode = "rr"
            if rr <= 0:
                can_open = False
            elif target_mode == "or_range":
                rng = float(orb_high) - float(orb_low)
                if rng <= 0:
                    can_open = False
                else:
                    target_price = (
                        float(orb_high) + (rr * rng) if entry_dir == "up" else float(orb_low) - (rr * rng)
                    )
            else:
                risk = abs(float(entry_price_est) - float(stop_price))
                if risk <= 0:
                    can_open = False
                else:
                    target_price = (
                        float(entry_price_est) + (rr * risk)
                        if entry_dir == "up"
                        else float(entry_price_est) - (rr * risk)
                    )
        profit_target_pct = None
        stop_loss_pct = None
    elif exit_mode == "atr":
        atr = float(atr_value or 0.0)
        if atr > 0 and entry_dir in ("up", "down"):
            pt_raw = getattr(cfg.strategy, "spot_pt_atr_mult", 1.5)
            sl_raw = getattr(cfg.strategy, "spot_sl_atr_mult", 1.0)
            try:
                pt_mult = float(1.5 if pt_raw is None else pt_raw)
            except (TypeError, ValueError):
                pt_mult = 1.5
            try:
                sl_mult = float(1.0 if sl_raw is None else sl_raw)
            except (TypeError, ValueError):
                sl_mult = 1.0
            if base_signed_qty > 0:
                target_price = float(entry_price_est) + (pt_mult * atr)
                stop_price = float(entry_price_est) - (sl_mult * atr)
            else:
                target_price = float(entry_price_est) - (pt_mult * atr)
                stop_price = float(entry_price_est) + (sl_mult * atr)
            profit_target_pct = None
            stop_loss_pct = None
        else:
            can_open = False

    base_profit_target_pct = profit_target_pct
    base_stop_loss_pct = stop_loss_pct

    shock_on = bool(shock_now) if shock_now is not None else False
    decision_trace_payload: dict[str, object] | None = None
    if can_open:
        sl_mult, pt_mult = spot_shock_exit_pct_multipliers(filters, shock=shock_on)
        stop_loss_pct, profit_target_pct = spot_scale_exit_pcts(
            stop_loss_pct=stop_loss_pct,
            profit_target_pct=profit_target_pct,
            stop_mult=sl_mult,
            profit_mult=pt_mult,
        )

    if can_open:
        signed_qty, decision_trace = spot_calc_signed_qty_with_trace(
            strategy=cfg.strategy,
            filters=filters,
            action=action,
            lot=int(lot),
            entry_price=float(entry_price_est),
            stop_price=stop_price,
            stop_loss_pct=stop_loss_pct,
            shock=bool(shock_on),
            shock_dir=shock_dir_now,
            shock_atr_pct=shock_atr_pct_now,
            shock_dir_down_streak_bars=shock_dir_down_streak_bars_now,
            shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct_now,
            shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp_now,
            shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp_now,
            shock_prearm_down_streak_bars=shock_prearm_down_streak_bars_now,
            shock_ramp=shock_ramp_now,
            riskoff=bool(riskoff_today),
            risk_dir=shock_dir_now,
            riskpanic=bool(riskpanic_today),
            riskpop=bool(riskpop_today),
            risk=risk_snapshot,
            signal_entry_dir=signal_entry_dir_now,
            signal_regime_dir=signal_regime_dir_now,
            equity_ref=float(cash) + float(liquidation_value),
            cash_ref=float(cash),
        )
        if signed_qty == 0:
            can_open = False
        else:
            size_mult = _spot_branch_size_mult(strategy=cfg.strategy, entry_branch=entry_branch)
            signed_qty = spot_apply_branch_size_mult(
                signed_qty=int(signed_qty),
                size_mult=float(size_mult),
                spot_min_qty=int(getattr(cfg.strategy, "spot_min_qty", 1) or 1),
                spot_max_qty=int(getattr(cfg.strategy, "spot_max_qty", 0) or 0),
            )
            decision_trace = decision_trace.with_branch_scaling(
                entry_branch=entry_branch,
                size_mult=float(size_mult),
                signed_qty_after_branch=int(signed_qty),
            )
            lifecycle = _spot_open_position_intent(
                strategy=cfg.strategy,
                bar_ts=entry_time,
                bar_size=str(cfg.backtest.bar_size),
                open_dir=None,
                current_qty=0,
                target_qty=int(signed_qty),
                spot_decision=decision_trace.as_payload(),
                shock_atr_pct=float(shock_atr_pct_now) if shock_atr_pct_now is not None else None,
                tr_ratio=float(getattr(risk_snapshot, "tr_ratio", 0.0))
                if risk_snapshot is not None and getattr(risk_snapshot, "tr_ratio", None) is not None
                else None,
                tr_median_pct=float(getattr(risk_snapshot, "tr_median_pct", 0.0))
                if risk_snapshot is not None and getattr(risk_snapshot, "tr_median_pct", None) is not None
                else None,
                slope_med_pct=float(getattr(risk_snapshot, "tr_median_delta_pct", 0.0))
                if risk_snapshot is not None and getattr(risk_snapshot, "tr_median_delta_pct", None) is not None
                else None,
                slope_vel_pct=float(getattr(risk_snapshot, "tr_slope_vel_pct", 0.0))
                if risk_snapshot is not None and getattr(risk_snapshot, "tr_slope_vel_pct", None) is not None
                else None,
            )
            intent_decision = lifecycle.spot_intent
            if lifecycle.intent != "enter" or intent_decision is None or int(intent_decision.order_qty) <= 0:
                can_open = False
            else:
                signed_qty = int(intent_decision.delta_qty)
                decision_trace_payload = decision_trace.as_payload()
                decision_trace_payload["spot_intent"] = intent_decision.as_payload()
                decision_trace_payload["spot_lifecycle"] = lifecycle.as_payload()
    else:
        signed_qty = 0

    if not can_open:
        return None

    entry_price = _spot_exec_price(
        float(entry_ref_price),
        side=side,
        qty=int(signed_qty),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
    )

    if entry_signal == "orb" and orb_engine is not None and entry_dir in ("up", "down"):
        if stop_price is not None:
            rr = float(getattr(cfg.strategy, "orb_risk_reward", 2.0) or 2.0)
            target_mode = str(getattr(cfg.strategy, "orb_target_mode", "rr") or "rr").strip().lower()
            if target_mode not in ("rr", "or_range"):
                target_mode = "rr"
            if rr <= 0:
                return None
            if target_mode == "rr":
                risk = abs(float(entry_price) - float(stop_price))
                if risk <= 0:
                    return None
                target_price = (
                    float(entry_price) + (rr * risk) if entry_dir == "up" else float(entry_price) - (rr * risk)
                )
    elif exit_mode == "atr":
        atr = float(atr_value or 0.0)
        if atr > 0 and entry_dir in ("up", "down"):
            pt_raw = getattr(cfg.strategy, "spot_pt_atr_mult", 1.5)
            sl_raw = getattr(cfg.strategy, "spot_sl_atr_mult", 1.0)
            try:
                pt_mult = float(1.5 if pt_raw is None else pt_raw)
            except (TypeError, ValueError):
                pt_mult = 1.5
            try:
                sl_mult = float(1.0 if sl_raw is None else sl_raw)
            except (TypeError, ValueError):
                sl_mult = 1.0
            if int(signed_qty) > 0:
                target_price = float(entry_price) + (pt_mult * atr)
                stop_price = float(entry_price) - (sl_mult * atr)
            else:
                target_price = float(entry_price) - (pt_mult * atr)
                stop_price = float(entry_price) + (sl_mult * atr)

    ok, cash_after, margin_after, margin_required = _spot_entry_accounting(
        cash=float(cash),
        margin_used=float(margin_used),
        signed_qty=int(signed_qty),
        entry_price=float(entry_price),
        mark_ref_price=float(mark_ref_price),
        liquidation_value=float(liquidation_value),
        spread=float(spread),
        mark_to_market=str(mark_to_market),
        multiplier=float(meta.multiplier),
    )
    if not ok:
        return None

    candidate = SpotTrade(
        symbol=cfg.strategy.symbol,
        qty=int(signed_qty),
        entry_time=entry_time,
        entry_price=float(entry_price),
        entry_branch=str(entry_branch) if entry_branch in ("a", "b") else None,
        decision_trace=decision_trace_payload,
        base_profit_target_pct=base_profit_target_pct,
        base_stop_loss_pct=base_stop_loss_pct,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        profit_target_price=target_price,
        stop_loss_price=stop_price,
    )
    candidate.margin_required = float(margin_required)
    return candidate, float(cash_after), float(margin_after)


def _spot_apply_opened_trade(
    *,
    opened: tuple[SpotTrade, float, float],
    open_trades: list[SpotTrade],
) -> tuple[float, float]:
    candidate, cash_after, margin_after = opened
    open_trades.append(candidate)
    return float(cash_after), float(margin_after)


def _spot_apply_resize_trade(
    *,
    trade: SpotTrade,
    delta_qty: int,
    entry_ref_price: float,
    mark_ref_price: float,
    cash: float,
    margin_used: float,
    spread: float,
    commission_per_share: float,
    commission_min: float,
    slippage_per_share: float,
    mark_to_market: str,
    multiplier: float,
    lifecycle_payload: dict[str, object] | None = None,
    decision_payload: dict[str, object] | None = None,
) -> tuple[bool, float, float]:
    delta = int(delta_qty)
    if delta == 0:
        return False, float(cash), float(margin_used)
    old_qty = int(trade.qty)
    new_qty = int(old_qty + delta)
    if new_qty == 0:
        return False, float(cash), float(margin_used)

    side = "buy" if delta > 0 else "sell"
    resize_price = _spot_exec_price(
        float(entry_ref_price),
        side=side,
        qty=int(delta),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
    )
    next_cash = float(cash) - (int(delta) * float(resize_price) * float(multiplier))

    old_abs = abs(int(old_qty))
    new_abs = abs(int(new_qty))
    if old_qty != 0 and ((old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0)) and new_abs > old_abs:
        # Scale-in keeps weighted entry basis for remaining inventory.
        weight_old = float(old_abs)
        weight_add = float(new_abs - old_abs)
        if (weight_old + weight_add) > 0:
            trade.entry_price = ((float(trade.entry_price) * weight_old) + (float(resize_price) * weight_add)) / (
                weight_old + weight_add
            )
    trade.qty = int(new_qty)

    old_margin = float(trade.margin_required or 0.0)
    mark_price = _spot_mark_price(float(mark_ref_price), qty=int(new_qty), spread=float(spread), mode=str(mark_to_market))
    new_margin = abs(int(new_qty)) * float(mark_price) * float(multiplier)
    trade.margin_required = float(new_margin)
    next_margin_used = max(0.0, float(margin_used) - float(old_margin) + float(new_margin))

    trace = trade.decision_trace if isinstance(trade.decision_trace, dict) else {}
    resizes = trace.get("resizes")
    rows = list(resizes) if isinstance(resizes, list) else []
    rows.append(
        {
            "delta_qty": int(delta),
            "new_qty": int(new_qty),
            "resize_price": float(resize_price),
            "lifecycle": dict(lifecycle_payload) if isinstance(lifecycle_payload, dict) else None,
            "spot_decision": dict(decision_payload) if isinstance(decision_payload, dict) else None,
        }
    )
    trace["resizes"] = rows
    trade.decision_trace = trace

    return True, float(next_cash), float(next_margin_used)


def _spot_opened_trade_summary_state(
    *,
    opened: tuple[SpotTrade, float, float],
    entry_exec_idx: int,
) -> tuple[int, int, datetime, float, float, float, dict[str, object] | None]:
    candidate, cash_after, margin_after = opened
    return (
        int(candidate.qty),
        int(entry_exec_idx),
        candidate.entry_time,
        float(candidate.entry_price),
        float(margin_after),
        float(cash_after),
        candidate.decision_trace if isinstance(candidate.decision_trace, dict) else None,
    )


def _summary_apply_closed_trade(
    *,
    pnl: float,
    entry_time: datetime,
    exit_time: datetime,
    trades: int,
    wins: int,
    losses: int,
    total_pnl: float,
    win_sum: float,
    loss_sum: float,
    hold_sum: float,
    hold_n: int,
) -> tuple[int, int, int, float, float, float, float, int]:
    pnl_f = float(pnl)
    total_pnl += pnl_f
    trades += 1
    if pnl_f >= 0:
        wins += 1
        win_sum += pnl_f
    else:
        losses += 1
        loss_sum += pnl_f
    hold_sum += (exit_time - entry_time).total_seconds() / 3600.0
    hold_n += 1
    return int(trades), int(wins), int(losses), float(total_pnl), float(win_sum), float(loss_sum), float(hold_sum), int(hold_n)


def _spot_emit_progress(progress_callback, **payload: object) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback(dict(payload))
    except Exception:
        return


def _run_spot_backtest(
    cfg: ConfigBundle,
    bars: BarSeriesInput,
    meta: ContractMeta,
    *,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
) -> BacktestResult:
    signal_bars = _bars_input_list(bars)
    regime_bars_list = _bars_input_optional_list(regime_bars)
    regime2_bars_list = _bars_input_optional_list(regime2_bars)
    tick_bars_list = _bars_input_optional_list(tick_bars)
    exec_bars_list = _bars_input_optional_list(exec_bars)

    exec_bar_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
    if exec_bar_size and str(exec_bar_size) != str(cfg.backtest.bar_size):
        if exec_bars_list is None:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars was not provided "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
        if not exec_bars_list:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars is empty "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
        return _run_spot_backtest_exec_loop(
            cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars_list,
            meta=meta,
            regime_bars=regime_bars_list,
            regime2_bars=regime2_bars_list,
            tick_bars=tick_bars_list,
        )

    # Canonical spot runner: single-res is just exec_bars=signal_bars.
    return _run_spot_backtest_exec_loop(
        cfg,
        signal_bars=signal_bars,
        exec_bars=signal_bars,
        meta=meta,
        regime_bars=regime_bars_list,
        regime2_bars=regime2_bars_list,
        tick_bars=tick_bars_list,
    )


def _run_spot_backtest_summary(
    cfg: ConfigBundle,
    bars: BarSeriesInput,
    meta: ContractMeta,
    *,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> SummaryStats:
    """Spot backtest optimized for sweeps that only need `SummaryStats`.

    Keeps semantics aligned with `_run_spot_backtest_exec_loop`, but skips building the
    full equity curve list (and the generic `_summarize` pass over it).
    """
    signal_bars = _bars_input_list(bars)
    regime_bars_list = _bars_input_optional_list(regime_bars)
    regime2_bars_list = _bars_input_optional_list(regime2_bars)
    tick_bars_list = _bars_input_optional_list(tick_bars)
    exec_bars_list = _bars_input_optional_list(exec_bars)

    exec_bar_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
    _spot_emit_progress(
        progress_callback,
        phase="summary.prepare",
        signal_total=int(len(signal_bars)),
        exec_total=int(len(exec_bars_list) if isinstance(exec_bars_list, list) else len(signal_bars)),
    )
    if exec_bar_size and str(exec_bar_size) != str(cfg.backtest.bar_size):
        if exec_bars_list is None:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars was not provided "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
        if not exec_bars_list:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars is empty "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
        return _run_spot_backtest_exec_loop_summary(
            cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars_list,
            meta=meta,
            regime_bars=regime_bars_list,
            regime2_bars=regime2_bars_list,
            tick_bars=tick_bars_list,
            prepared_series_pack=prepared_series_pack,
            progress_callback=progress_callback,
        )

    # Canonical spot runner: single-res is just exec_bars=signal_bars.
    return _run_spot_backtest_exec_loop_summary(
        cfg,
        signal_bars=signal_bars,
        exec_bars=signal_bars,
        meta=meta,
        regime_bars=regime_bars_list,
        regime2_bars=regime2_bars_list,
        tick_bars=tick_bars_list,
        prepared_series_pack=prepared_series_pack,
        progress_callback=progress_callback,
    )

# "Multi-res" here just means signal bars and execution bars can be different resolutions
# (e.g. signal=30m, exec=5m for intrabar stops/targets). If they are the same timeframe,
# pass exec_bars=signal_bars and this behaves like a single-resolution backtest.
def _run_spot_backtest_exec_loop(
    cfg: ConfigBundle,
    *,
    signal_bars: BarSeriesInput,
    exec_bars: BarSeriesInput,
    meta: ContractMeta,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    capture_equity: bool = True,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> BacktestResult:
    """Spot backtest using an execution-bar loop.

    - Signals/filters are evaluated on `signal_bars` (cfg.backtest.bar_size).
    - Execution + exits are simulated on `exec_bars` (cfg.strategy.spot_exec_bar_size).
    - Single-resolution runs pass exec_bars=signal_bars.
    """

    signal_bars = _bars_input_list(signal_bars)
    exec_bars = _bars_input_list(exec_bars)
    regime_bars = _bars_input_optional_list(regime_bars)
    regime2_bars = _bars_input_optional_list(regime2_bars)
    tick_bars = _bars_input_optional_list(tick_bars)

    cash = cfg.backtest.starting_cash
    margin_used = 0.0
    equity_curve: list[EquityPoint] = []
    trades: list[SpotTrade] = []
    open_trades: list[SpotTrade] = []
    trace_out_raw = getattr(cfg.strategy, "spot_lifecycle_trace_path", None)
    why_not_out_raw = getattr(cfg.strategy, "spot_why_not_report_path", None)
    capture_lifecycle = bool(getattr(cfg.strategy, "spot_capture_lifecycle_trace", False)) or bool(
        str(trace_out_raw or "").strip()
    ) or bool(str(why_not_out_raw or "").strip())
    lifecycle_rows: list[dict[str, object]] | None = [] if capture_lifecycle else None

    if not signal_bars:
        raise ValueError("signal_bars is empty")
    if not exec_bars:
        raise ValueError("exec_bars is empty")

    filters = cfg.strategy.filters
    entry_signal = normalize_spot_entry_signal(getattr(cfg.strategy, "entry_signal", "ema"))

    ema_periods = _ema_periods(cfg.strategy.ema_preset) if entry_signal == "ema" else None
    needs_direction = cfg.strategy.directional_spot is not None
    if entry_signal == "ema" and ema_periods is None:
        raise ValueError("spot backtests require ema_preset")
    ema_needed = entry_signal == "ema"

    _regime_mode, _regime_preset, _regime_bar, use_mtf_regime_cfg = resolve_spot_regime_spec(
        bar_size=cfg.backtest.bar_size,
        regime_mode_raw=getattr(cfg.strategy, "regime_mode", "ema"),
        regime_ema_preset_raw=getattr(cfg.strategy, "regime_ema_preset", None),
        regime_bar_size_raw=getattr(cfg.strategy, "regime_bar_size", None),
    )
    use_mtf_regime = bool(regime_bars) and bool(use_mtf_regime_cfg)
    regime2_mode, _regime2_preset, _regime2_bar, use_mtf_regime2_cfg = resolve_spot_regime2_spec(
        bar_size=cfg.backtest.bar_size,
        regime2_mode_raw=getattr(cfg.strategy, "regime2_mode", "off"),
        regime2_ema_preset_raw=getattr(cfg.strategy, "regime2_ema_preset", None),
        regime2_bar_size_raw=getattr(cfg.strategy, "regime2_bar_size", None),
    )
    if regime2_mode != "off" and bool(use_mtf_regime2_cfg) and not regime2_bars:
        raise ValueError("regime2_mode enabled but regime2_bars was not provided for multi-timeframe regime2")
    use_mtf_regime2 = bool(regime2_bars) and bool(use_mtf_regime2_cfg)

    from ..spot_engine import SpotSignalEvaluator, SpotSignalSnapshot

    evaluator = SpotSignalEvaluator(
        strategy=cfg.strategy,
        filters=filters,
        bar_size=str(cfg.backtest.bar_size),
        use_rth=bool(cfg.backtest.use_rth),
        naive_ts_mode="utc",
        rv_lookback=int(cfg.synthetic.rv_lookback),
        rv_ewma_lambda=float(cfg.synthetic.rv_ewma_lambda),
        regime_bars=regime_bars if use_mtf_regime else None,
        regime2_bars=regime2_bars if use_mtf_regime2 else None,
    )
    orb_engine = evaluator.orb_engine
    last_sig_snap: SpotSignalSnapshot | None = None
    shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view

    (
        tick_mode,
        tick_neutral_policy,
        _tick_direction_policy,
        _tick_ma_period,
        _tick_z_lookback,
        _tick_z_enter,
        _tick_z_exit,
        _tick_slope_lookback,
    ) = _spot_tick_gate_settings(cfg.strategy)
    series_pack = prepared_series_pack if isinstance(prepared_series_pack, _SpotSeriesPack) else None
    if series_pack is None:
        series_pack = _spot_build_series_pack(
            cfg=cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
            include_rv=False,
            include_volume=False,
            include_tick=(str(tick_mode) != "off"),
        )
    align = series_pack.core.align
    tick_series = series_pack.core.tick_series

    exec_profile = _spot_exec_profile(cfg.strategy)
    exit_mode = str(exec_profile.exit_mode)
    spot_exit_time = parse_time_hhmm(getattr(cfg.strategy, "spot_exit_time_et", None))
    spot_exec_bar_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or cfg.backtest.bar_size or "")
    spot_sec_type = _spot_strategy_sec_type(strategy=cfg.strategy)
    spot_entry_fill_mode = str(exec_profile.entry_fill_mode)
    spot_flip_exit_fill_mode = str(exec_profile.flip_fill_mode)
    spot_intrabar_exits = bool(exec_profile.intrabar_exits)
    spot_close_eod = bool(exec_profile.close_eod)
    spot_spread = float(exec_profile.spread)
    spot_commission = float(exec_profile.commission_per_share)
    spot_commission_min = float(exec_profile.commission_min)
    spot_slippage = float(exec_profile.slippage_per_share)
    spot_mark_to_market = str(exec_profile.mark_to_market)
    spot_drawdown_mode = str(exec_profile.drawdown_mode)
    equity_peak: float | None = None
    equity_max_dd = 0.0

    def _record_equity_point(ts: datetime, equity: float) -> None:
        nonlocal equity_peak, equity_max_dd
        value = float(equity)
        if capture_equity:
            equity_curve.append(EquityPoint(ts=ts, equity=value))
        if equity_peak is None or value > equity_peak:
            equity_peak = value
        dd = float(equity_peak) - value
        if dd > equity_max_dd:
            equity_max_dd = dd

    def _capture_lifecycle(
        *,
        stage: str,
        decision,
        bar_ts: datetime,
        exec_idx: int,
        sig_idx: int | None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if lifecycle_rows is None:
            return
        merged_context: dict[str, object] = {
            "symbol": str(cfg.strategy.symbol),
            "exec_idx": int(exec_idx),
            "sig_idx": int(sig_idx) if sig_idx is not None else None,
        }
        if isinstance(context, Mapping):
            for key, value in context.items():
                merged_context[str(key)] = value
        lifecycle_rows.append(
            lifecycle_trace_row(
                bar_ts=bar_ts,
                stage=str(stage),
                decision=decision,
                context=merged_context,
            )
        )

    def _spot_liquidation(ref_price: float) -> float:
        return _spot_liquidation_value(
            open_trades=open_trades,
            ref_price=float(ref_price),
            spread=spot_spread,
            mark_to_market=spot_mark_to_market,
            multiplier=meta.multiplier,
        )

    def _exec_trade_exit(
        trade: SpotTrade,
        *,
        exit_ref_price: float,
        exit_time: datetime,
        reason: str,
        apply_slippage: bool | None = None,
    ) -> tuple[float, float, float]:
        return _spot_exec_exit_common(
            qty=int(trade.qty),
            margin_required=float(trade.margin_required),
            exit_ref_price=float(exit_ref_price),
            exit_time=exit_time,
            reason=str(reason),
            cash=float(cash),
            margin_used=float(margin_used),
            spread=float(spot_spread),
            commission_per_share=float(spot_commission),
            commission_min=float(spot_commission_min),
            slippage_per_share=float(spot_slippage),
            multiplier=float(meta.multiplier),
            apply_slippage=apply_slippage,
            trade=trade,
            trades=trades,
        )

    pending_entry_dir: str | None = None
    pending_entry_branch: str | None = None
    pending_entry_set_date: date | None = None
    pending_entry_due_ts: datetime | None = None
    pending_exit_all = False
    pending_exit_reason = ""
    pending_exit_due_ts: datetime | None = None

    (
        riskoff_mode,
        riskoff_long_factor,
        riskoff_short_factor,
        riskpanic_long_factor,
        riskpanic_short_factor,
        riskpop_long_factor,
        riskpop_short_factor,
    ) = risk_overlay_policy_from_filters(filters)
    risk_overlay_enabled = bool(evaluator.risk_overlay_enabled)
    riskoff_today = False
    riskpanic_today = False
    riskpop_today = False
    riskoff_end_hour = _spot_riskoff_end_hour(filters) if risk_overlay_enabled else None
    last_entry_sig_idx: int | None = None
    last_resize_bar_ts: datetime | None = None

    exec_last_date = None
    entries_today = 0
    exec_total = int(len(exec_bars))
    progress_stride = max(1, int(max(64, exec_total // 200)))

    for idx, bar in enumerate(exec_bars):
        next_bar = exec_bars[idx + 1] if idx + 1 < len(exec_bars) else None
        is_last_bar = next_bar is None or _trade_date(next_bar.ts) != _trade_date(bar.ts)
        sig_map_idx = align.sig_idx_by_exec_idx[idx] if idx < len(align.sig_idx_by_exec_idx) else -1
        sig_idx = int(sig_map_idx) if int(sig_map_idx) >= 0 else None
        sig_bar = signal_bars[int(sig_idx)] if sig_idx is not None and int(sig_idx) < len(signal_bars) else None
        if int(idx) == 0 or int((idx + 1) % int(progress_stride)) == 0 or int(idx + 1) >= int(exec_total):
            _spot_emit_progress(
                progress_callback,
                phase="engine.exec",
                path="exec",
                exec_idx=int(idx + 1),
                exec_total=int(exec_total),
                sig_idx=(int(sig_idx + 1) if sig_idx is not None else None),
                sig_total=int(len(signal_bars)),
                open_count=int(len(open_trades)),
                trades=int(len(trades)),
            )

        if exec_last_date != _trade_date(bar.ts):
            exec_last_date = _trade_date(bar.ts)
            entries_today = 0
        shock_now_prev = shock_prev
        shock_dir_prev_now = shock_dir_prev
        shock_atr_pct_prev_now = shock_atr_pct_prev

        evaluator.update_exec_bar(bar, is_last_bar=bool(is_last_bar))
        if evaluator.last_risk is not None:
            riskoff_today = bool(evaluator.last_risk.riskoff)
            riskpanic_today = bool(evaluator.last_risk.riskpanic)
            riskpop_today = bool(getattr(evaluator.last_risk, "riskpop", False))
        else:
            riskoff_today = False
            riskpanic_today = False
            riskpop_today = False

        open_dir_now = None
        if open_trades:
            open_dir_now = "up" if int(open_trades[0].qty) > 0 else "down"
        pending_decision = decide_pending_next_open(
            now_ts=bar.ts,
            has_open=bool(open_trades),
            open_dir=open_dir_now,
            pending_entry_dir=pending_entry_dir,
            pending_entry_set_date=pending_entry_set_date,
            pending_entry_due_ts=pending_entry_due_ts if pending_entry_dir in ("up", "down") else None,
            pending_exit_reason=pending_exit_reason,
            pending_exit_due_ts=pending_exit_due_ts if bool(pending_exit_all) else None,
            risk_overlay_enabled=bool(risk_overlay_enabled),
            riskoff_today=bool(riskoff_today),
            riskpanic_today=bool(riskpanic_today),
            riskpop_today=bool(riskpop_today),
            riskoff_mode=str(riskoff_mode),
            shock_dir_now=shock_dir_prev_now,
            riskoff_end_hour=riskoff_end_hour,
            pending_entry_fill_mode=spot_entry_fill_mode,
            pending_exit_fill_mode=(
                spot_flip_exit_fill_mode if str(pending_exit_reason or "").strip().lower() == "flip" else SPOT_FILL_MODE_CLOSE
            ),
            naive_ts_mode="utc",
        )
        _capture_lifecycle(
            stage="pending",
            decision=pending_decision,
            bar_ts=bar.ts,
            exec_idx=int(idx),
            sig_idx=int(sig_idx) if sig_idx is not None else None,
        )
        if pending_decision.pending_clear_exit:
            pending_exit_all = False
            pending_exit_reason = ""
            pending_exit_due_ts = None

        if pending_decision.intent == "exit" and open_trades:
            exit_ref = float(bar.open)
            for trade in list(open_trades):
                _exit_price, cash, margin_used = _exec_trade_exit(
                    trade,
                    exit_ref_price=exit_ref,
                    exit_time=bar.ts,
                    reason=str(pending_decision.reason or pending_exit_reason or "flip"),
                    apply_slippage=True,
                )
            open_trades = []

        if pending_decision.intent == "enter" and pending_entry_dir in ("up", "down"):
            can_fill_pending = _spot_entry_capacity_ok(
                open_count=len(open_trades),
                max_entries_per_day=int(cfg.strategy.max_entries_per_day),
                entries_today=int(entries_today),
                weekday=_trade_weekday(bar.ts),
                entry_days=cfg.strategy.entry_days,
            )
            if can_fill_pending:
                entry_dir = pending_entry_dir
                entry_branch = pending_entry_branch if pending_entry_branch in ("a", "b") else None
                pending_entry_dir = None
                pending_entry_branch = None
                pending_entry_set_date = None
                pending_entry_due_ts = None

                entry_leg = _spot_entry_leg_for_direction(
                    strategy=cfg.strategy,
                    entry_dir=entry_dir,
                    needs_direction=needs_direction,
                )
                if entry_leg is not None and entry_dir in ("up", "down"):
                    liquidation_open = _spot_liquidation(float(bar.open))
                    atr_value = float(last_sig_snap.atr) if last_sig_snap is not None and last_sig_snap.atr is not None else None
                    opened = _spot_try_open_entry(
                        cfg=cfg,
                        meta=meta,
                        entry_signal=entry_signal,
                        entry_dir=entry_dir,
                        entry_branch=entry_branch,
                        entry_leg=entry_leg,
                        entry_time=bar.ts,
                        entry_ref_price=float(bar.open),
                        mark_ref_price=float(bar.open),
                        atr_value=atr_value,
                        exit_mode=exit_mode,
                        orb_engine=orb_engine,
                        filters=filters,
                        shock_now=shock_now_prev,
                        shock_dir_now=shock_dir_prev_now,
                        shock_atr_pct_now=shock_atr_pct_prev_now,
                        shock_dir_down_streak_bars_now=(
                            int(getattr(last_sig_snap, "shock_dir_down_streak_bars", 0))
                            if last_sig_snap is not None and getattr(last_sig_snap, "shock_dir_down_streak_bars", None) is not None
                            else None
                        ),
                        shock_drawdown_dist_on_pct_now=(
                            float(getattr(last_sig_snap, "shock_drawdown_dist_on_pct", 0.0))
                            if last_sig_snap is not None and getattr(last_sig_snap, "shock_drawdown_dist_on_pct", None) is not None
                            else None
                        ),
                        shock_drawdown_dist_on_vel_pp_now=(
                            float(getattr(last_sig_snap, "shock_drawdown_dist_on_vel_pp", 0.0))
                            if last_sig_snap is not None and getattr(last_sig_snap, "shock_drawdown_dist_on_vel_pp", None) is not None
                            else None
                        ),
                        shock_drawdown_dist_on_accel_pp_now=(
                            float(getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", 0.0))
                            if last_sig_snap is not None
                            and getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", None) is not None
                            else None
                        ),
                        shock_prearm_down_streak_bars_now=(
                            int(getattr(last_sig_snap, "shock_prearm_down_streak_bars", 0))
                            if last_sig_snap is not None
                            and getattr(last_sig_snap, "shock_prearm_down_streak_bars", None) is not None
                            else None
                        ),
                        shock_ramp_now=(
                            dict(getattr(last_sig_snap, "shock_ramp"))
                            if last_sig_snap is not None and isinstance(getattr(last_sig_snap, "shock_ramp", None), dict)
                            else None
                        ),
                        signal_entry_dir_now=(
                            str(getattr(last_sig_snap, "entry_dir", None))
                            if last_sig_snap is not None and getattr(last_sig_snap, "entry_dir", None) in ("up", "down")
                            else None
                        ),
                        signal_regime_dir_now=(
                            str(getattr(getattr(last_sig_snap, "signal", None), "regime_dir", None))
                            if last_sig_snap is not None
                            and getattr(getattr(last_sig_snap, "signal", None), "regime_dir", None) in ("up", "down")
                            else None
                        ),
                        riskoff_today=bool(riskoff_today),
                        riskpanic_today=bool(riskpanic_today),
                        riskpop_today=bool(riskpop_today),
                        risk_snapshot=evaluator.last_risk if risk_overlay_enabled else None,
                        cash=float(cash),
                        margin_used=float(margin_used),
                        liquidation_value=float(liquidation_open),
                        spread=float(spot_spread),
                        commission_per_share=float(spot_commission),
                        commission_min=float(spot_commission_min),
                        slippage_per_share=float(spot_slippage),
                        mark_to_market=str(spot_mark_to_market),
                    )
                    if opened is not None:
                        cash, margin_used = _spot_apply_opened_trade(opened=opened, open_trades=open_trades)
                        entries_today += 1
            else:
                pending_entry_dir = None
                pending_entry_branch = None
                pending_entry_set_date = None
                pending_entry_due_ts = None

        if pending_decision.pending_clear_entry:
            pending_entry_dir = None
            pending_entry_branch = None
            pending_entry_set_date = None
            pending_entry_due_ts = None

        # Dynamic shock SL/PT: apply the shock multipliers to *open* trades using the shock
        # state from the prior execution bar (no lookahead within this bar).
        if open_trades and filters is not None and evaluator.shock_enabled:
            shock_now = bool(shock_now_prev) if shock_now_prev is not None else False
            sl_mult, pt_mult = spot_shock_exit_pct_multipliers(filters, shock=shock_now)
            for trade in open_trades:
                scaled_sl, scaled_pt = spot_scale_exit_pcts(
                    stop_loss_pct=trade.base_stop_loss_pct,
                    profit_target_pct=trade.base_profit_target_pct,
                    stop_mult=sl_mult,
                    profit_mult=pt_mult,
                )
                if (
                    trade.stop_loss_price is None
                    and scaled_sl is not None
                ):
                    trade.stop_loss_pct = float(scaled_sl)
                if (
                    trade.profit_target_price is None
                    and scaled_pt is not None
                ):
                    trade.profit_target_pct = float(scaled_pt)

        # Signal processing happens on signal-bar closes (after this bar completes).
        rv = None
        sig_bars_in_day = 0
        volume_ema = None
        volume_ema_ready = True
        shock = None
        shock_dir = None
        shock_atr_pct = None
        shock_atr_vel = None
        shock_atr_accel = None
        shock_drawdown_dist_on_pct = None
        shock_drawdown_dist_on_vel_pp = None
        atr = None
        signal = None
        entry_signal_dir = None
        entry_regime_dir = None
        entry_signal_branch = None
        shock_dir_down_streak_bars = None
        ratsv_tr_ratio = None
        risk_tr_median_pct = None
        ratsv_fast_slope_med = None
        ratsv_fast_slope_vel = None
        ratsv_slow_slope_med = None
        ratsv_slow_slope_vel = None
        ratsv_slope_vel_consistency = None
        if sig_bar is not None:
            sig_snap = evaluator.update_signal_bar(sig_bar)
            if sig_snap is not None:
                last_sig_snap = sig_snap
                rv = sig_snap.rv
                sig_bars_in_day = int(sig_snap.bars_in_day)
                volume_ema = sig_snap.volume_ema
                volume_ema_ready = bool(sig_snap.volume_ema_ready)
                shock = sig_snap.shock
                shock_dir = sig_snap.shock_dir
                shock_atr_pct = sig_snap.shock_atr_pct
                shock_atr_vel = getattr(sig_snap, "shock_atr_vel_pct", None)
                shock_atr_accel = getattr(sig_snap, "shock_atr_accel_pct", None)
                shock_drawdown_dist_on_pct = getattr(sig_snap, "shock_drawdown_dist_on_pct", None)
                shock_drawdown_dist_on_vel_pp = getattr(sig_snap, "shock_drawdown_dist_on_vel_pp", None)
                atr = sig_snap.atr
                signal = sig_snap.signal
                entry_signal_dir = sig_snap.entry_dir
                entry_regime_dir = sig_snap.signal.regime_dir if sig_snap.signal is not None else None
                entry_signal_branch = sig_snap.entry_branch
                shock_dir_down_streak_bars = getattr(sig_snap, "shock_dir_down_streak_bars", None)
                ratsv_tr_ratio = sig_snap.ratsv_tr_ratio
                risk_snap = getattr(sig_snap, "risk", None)
                risk_tr_median_pct = (
                    float(getattr(risk_snap, "tr_median_pct"))
                    if risk_snap is not None and getattr(risk_snap, "tr_median_pct", None) is not None
                    else None
                )
                ratsv_fast_slope_med = sig_snap.ratsv_fast_slope_med_pct
                ratsv_fast_slope_vel = sig_snap.ratsv_fast_slope_vel_pct
                ratsv_slow_slope_med = getattr(sig_snap, "ratsv_slow_slope_med_pct", None)
                ratsv_slow_slope_vel = getattr(sig_snap, "ratsv_slow_slope_vel_pct", None)
                ratsv_slope_vel_consistency = getattr(sig_snap, "ratsv_slope_vel_consistency", None)

        # Track worst-in-bar equity using the execution bars (for drawdown realism).
        if spot_drawdown_mode == "intrabar" and open_trades:
            worst_liquidation = 0.0
            for trade in open_trades:
                stop_level = spot_stop_level(
                    float(trade.entry_price),
                    int(trade.qty),
                    stop_loss_price=trade.stop_loss_price,
                    stop_loss_pct=trade.stop_loss_pct,
                )
                worst_ref = spot_intrabar_worst_ref(
                    qty=int(trade.qty),
                    bar_open=float(bar.open),
                    bar_high=float(bar.high),
                    bar_low=float(bar.low),
                    stop_level=stop_level,
                )
                worst_liquidation += (
                    trade.qty
                    * _spot_mark_price(worst_ref, qty=trade.qty, spread=spot_spread, mode=spot_mark_to_market)
                    * meta.multiplier
                )
            _record_equity_point(bar.ts - timedelta(microseconds=1), cash + worst_liquidation)

        # Exit checks (profit/stop always; flip only on signal-bar closes).
        if open_trades:
            still_open: list[SpotTrade] = []
            for trade in open_trades:
                exit_candidates: dict[str, bool] = {}
                exit_ref_by_reason: dict[str, float] = {}
                apply_slippage_by_reason: dict[str, bool] = {}

                if spot_intrabar_exits:
                    stop_level = spot_stop_level(
                        float(trade.entry_price),
                        int(trade.qty),
                        stop_loss_price=trade.stop_loss_price,
                        stop_loss_pct=trade.stop_loss_pct,
                    )
                    profit_level = spot_profit_level(
                        float(trade.entry_price),
                        int(trade.qty),
                        profit_target_price=trade.profit_target_price,
                        profit_target_pct=trade.profit_target_pct,
                    )
                    hit = spot_intrabar_exit(
                        qty=int(trade.qty),
                        bar_open=float(bar.open),
                        bar_high=float(bar.high),
                        bar_low=float(bar.low),
                        stop_level=stop_level,
                        profit_level=profit_level,
                    )
                    if hit is not None:
                        kind, ref = hit
                        if kind == "stop":
                            reason = "stop_loss" if trade.stop_loss_price is not None else "stop_loss_pct"
                        else:
                            reason = "profit_target" if trade.profit_target_price is not None else "profit_target_pct"
                        exit_candidates[reason] = True
                        exit_ref_by_reason[reason] = float(ref)
                        apply_slippage_by_reason[reason] = (kind != "profit")
                else:
                    if spot_hit_profit(
                        entry_price=float(trade.entry_price),
                        qty=int(trade.qty),
                        price=float(bar.close),
                        profit_target_price=trade.profit_target_price,
                        profit_target_pct=trade.profit_target_pct,
                    ):
                        reason = "profit_target" if trade.profit_target_price is not None else "profit_target_pct"
                        exit_candidates[reason] = True
                        exit_ref_by_reason[reason] = float(bar.close)
                        apply_slippage_by_reason[reason] = False
                    elif spot_hit_stop(
                        entry_price=float(trade.entry_price),
                        qty=int(trade.qty),
                        price=float(bar.close),
                        stop_loss_price=trade.stop_loss_price,
                        stop_loss_pct=trade.stop_loss_pct,
                    ):
                        reason = "stop_loss" if trade.stop_loss_price is not None else "stop_loss_pct"
                        exit_candidates[reason] = True
                        exit_ref_by_reason[reason] = float(bar.close)
                        apply_slippage_by_reason[reason] = True

                is_signal_close = sig_idx is not None
                if (
                    is_signal_close
                    and _spot_ratsv_probe_cancel_hit(
                        cfg,
                        trade=trade,
                        bar=bar,
                        tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                        slope_med=float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
                    )
                ):
                    exit_candidates["ratsv_probe_cancel"] = True
                    exit_ref_by_reason["ratsv_probe_cancel"] = float(bar.close)
                    apply_slippage_by_reason["ratsv_probe_cancel"] = True
                if (
                    is_signal_close
                    and _spot_ratsv_adverse_release_hit(
                        cfg,
                        trade=trade,
                        bar=bar,
                        tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                        slope_med=float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
                        slope_vel=float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None,
                    )
                ):
                    exit_candidates["ratsv_adverse_release"] = True
                    exit_ref_by_reason["ratsv_adverse_release"] = float(bar.close)
                    apply_slippage_by_reason["ratsv_adverse_release"] = True
                if is_signal_close and _spot_hit_flip_exit(
                    cfg,
                    trade,
                    bar,
                    signal,
                    tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                    shock_atr_vel_pct=float(shock_atr_vel) if shock_atr_vel is not None else None,
                    tr_median_pct=(
                        float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                    ),
                ):
                    exit_candidates["flip"] = True
                    exit_ref_by_reason["flip"] = float(bar.close)
                    apply_slippage_by_reason["flip"] = True
                if spot_exit_time is not None:
                    ts_et = _ts_to_et(bar.ts)
                    if ts_et.time() >= spot_exit_time:
                        exit_candidates["exit_time"] = True
                        exit_ref_by_reason["exit_time"] = float(bar.close)
                        apply_slippage_by_reason["exit_time"] = True
                if bool(spot_close_eod) and is_last_bar:
                    exit_candidates["close_eod"] = True
                    exit_ref_by_reason["close_eod"] = float(bar.close)
                    apply_slippage_by_reason["close_eod"] = True

                lifecycle = _spot_open_position_intent(
                    strategy=cfg.strategy,
                    bar_ts=bar.ts,
                    bar_size=str(cfg.backtest.bar_size),
                    open_dir="up" if int(trade.qty) > 0 else "down",
                    current_qty=int(trade.qty),
                    exit_candidates=exit_candidates,
                    signal_entry_dir=entry_signal_dir,
                    shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
                    shock_atr_vel_pct=float(shock_atr_vel) if shock_atr_vel is not None else None,
                    shock_atr_accel_pct=float(shock_atr_accel) if shock_atr_accel is not None else None,
                    tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                    tr_median_pct=(
                        float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                    ),
                    slope_med_pct=float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
                    slope_vel_pct=float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None,
                    slope_med_slow_pct=float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None,
                    slope_vel_slow_pct=float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None,
                )
                _capture_lifecycle(
                    stage="open_exit",
                    decision=lifecycle,
                    bar_ts=bar.ts,
                    exec_idx=int(idx),
                    sig_idx=int(sig_idx) if sig_idx is not None else None,
                    context={
                        "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
                        "shock_atr_vel_pct": float(shock_atr_vel) if shock_atr_vel is not None else None,
                        "shock_atr_accel_pct": float(shock_atr_accel) if shock_atr_accel is not None else None,
                        "ratsv_tr_ratio": float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                        "risk_tr_median_pct": (
                            float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                        ),
                        "ratsv_fast_slope_med_pct": (
                            float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None
                        ),
                        "ratsv_fast_slope_vel_pct": (
                            float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None
                        ),
                        "ratsv_slow_slope_med_pct": (
                            float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None
                        ),
                        "ratsv_slow_slope_vel_pct": (
                            float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None
                        ),
                        "ratsv_slope_vel_consistency": (
                            float(ratsv_slope_vel_consistency)
                            if ratsv_slope_vel_consistency is not None
                            else None
                        ),
                    },
                )
                resolved_exit_reason = _spot_exit_reason_from_lifecycle(
                    lifecycle=lifecycle,
                    exit_candidates=exit_candidates,
                    fallback_priority=(
                        "stop_loss",
                        "stop_loss_pct",
                        "profit_target",
                        "profit_target_pct",
                        "flip",
                        "ratsv_probe_cancel",
                        "ratsv_adverse_release",
                        "exit_time",
                        "close_eod",
                    ),
                )
                if (
                    lifecycle.intent == "exit"
                    and str(resolved_exit_reason or "") == "flip"
                    and spot_fill_mode_is_deferred(lifecycle.fill_mode)
                    and next_bar is not None
                ):
                    due_ts = _spot_fill_due_ts(
                        strategy=cfg.strategy,
                        fill_mode=str(lifecycle.fill_mode),
                        signal_close_ts=bar.ts,
                        exec_bar_size=spot_exec_bar_size,
                        signal_use_rth=bool(cfg.backtest.use_rth),
                        spot_sec_type=spot_sec_type,
                    )
                    if due_ts is None:
                        still_open.append(trade)
                        continue
                    pending_exit_all = True
                    pending_exit_reason = "flip"
                    pending_exit_due_ts = due_ts
                    if lifecycle.queue_reentry_dir in ("up", "down") and pending_entry_dir is None:
                        pending_entry_dir = str(lifecycle.queue_reentry_dir)
                        pending_entry_branch = entry_signal_branch if entry_signal_branch in ("a", "b") else None
                        pending_entry_set_date = _trade_date(bar.ts)
                        pending_entry_due_ts = due_ts
                    still_open.append(trade)
                    continue

                if lifecycle.intent == "exit" and resolved_exit_reason is not None:
                    reason = str(resolved_exit_reason)
                    exit_ref = exit_ref_by_reason.get(reason, float(bar.close))
                    apply_slippage = apply_slippage_by_reason.get(reason, True)
                    _exit_price, cash, margin_used = _exec_trade_exit(
                        trade,
                        exit_ref_price=float(exit_ref),
                        exit_time=bar.ts,
                        reason=reason,
                        apply_slippage=bool(apply_slippage),
                    )
                else:
                    still_open.append(trade)
            open_trades = still_open

        # Update equity after processing this execution bar.
        liquidation = 0.0
        for trade in open_trades:
            liquidation += (
                trade.qty
                * _spot_mark_price(float(bar.close), qty=trade.qty, spread=spot_spread, mode=spot_mark_to_market)
                * meta.multiplier
            )
        _record_equity_point(bar.ts, cash + liquidation)

        if sig_bar is None or sig_idx is None:
            shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view
            continue

        direction, _ = _spot_resolve_entry_dir(
            signal=signal,
            entry_dir=entry_signal_dir,
            ema_needed=bool(ema_needed),
            sig_idx=int(sig_idx),
            tick_mode=str(tick_mode),
            tick_series=tick_series,
            tick_neutral_policy=str(tick_neutral_policy),
            needs_direction=bool(needs_direction),
            directional_spot=cfg.strategy.directional_spot,
        )
        entry_ok = bool(direction is not None)

        cooldown_ok = cooldown_ok_by_index(
            current_idx=int(sig_idx),
            last_entry_idx=last_entry_sig_idx,
            cooldown_bars=filters.cooldown_bars if filters else 0,
        )
        effective_open = len(open_trades)
        if pending_entry_dir is not None:
            effective_open += 1
        entry_plan = lifecycle_deferred_entry_plan(
            fill_mode=spot_entry_fill_mode,
            signal_ts=bar.ts,
            signal_close_ts=bar.ts,
            exec_bar_size=spot_exec_bar_size,
            strategy=cfg.strategy,
            riskoff_today=bool(riskoff_today),
            riskoff_end_hour=riskoff_end_hour,
            exit_mode=exit_mode,
            atr_value=float(atr) if atr is not None else None,
            naive_ts_mode="utc",
        )
        if next_bar is None and entry_plan.deferred:
            entry_plan = replace(entry_plan, due_ts=None, allowed=False, reason="no_next_bar")
        next_open_ok = bool(entry_plan.allowed and pending_entry_dir is None)
        entry_decision = _spot_flat_entry_decision_from_signal(
            strategy=cfg.strategy,
            filters=filters,
            signal_bar=sig_bar,
            signal=signal,
            direction=direction,
            bars_in_day=int(sig_bars_in_day),
            volume_ema=float(volume_ema) if volume_ema is not None else None,
            volume_ema_ready=bool(volume_ema_ready),
            rv=float(rv) if rv is not None else None,
            cooldown_ok=bool(cooldown_ok),
            shock=shock,
            shock_dir=shock_dir,
            open_count=int(effective_open),
            entries_today=int(entries_today),
            pending_exists=bool(pending_entry_dir is not None or pending_exit_all),
            next_open_allowed=bool(next_open_ok),
            exit_mode=exit_mode,
            atr_value=float(atr) if atr is not None else None,
            shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
            shock_atr_vel_pct=float(shock_atr_vel) if shock_atr_vel is not None else None,
            shock_atr_accel_pct=float(shock_atr_accel) if shock_atr_accel is not None else None,
            tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
            tr_median_pct=float(risk_tr_median_pct) if risk_tr_median_pct is not None else None,
            slope_med_pct=float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
            slope_vel_pct=float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None,
            slope_med_slow_pct=float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None,
            slope_vel_slow_pct=float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None,
        )
        _capture_lifecycle(
            stage="flat_entry",
            decision=entry_decision,
            bar_ts=sig_bar.ts,
            exec_idx=int(idx),
            sig_idx=int(sig_idx),
            context={
                "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
                "shock_atr_vel_pct": float(shock_atr_vel) if shock_atr_vel is not None else None,
                "shock_atr_accel_pct": float(shock_atr_accel) if shock_atr_accel is not None else None,
                "ratsv_tr_ratio": float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                "risk_tr_median_pct": (
                    float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                ),
                "ratsv_fast_slope_med_pct": float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
                "ratsv_fast_slope_vel_pct": float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None,
                "ratsv_slow_slope_med_pct": float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None,
                "ratsv_slow_slope_vel_pct": float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None,
                "ratsv_slope_vel_consistency": (
                    float(ratsv_slope_vel_consistency) if ratsv_slope_vel_consistency is not None else None
                ),
            },
        )
        if entry_decision.intent == "enter":
            direction = str(entry_decision.direction or direction)
            spot_leg = _spot_entry_leg_for_direction(
                strategy=cfg.strategy,
                entry_dir=direction,
                needs_direction=needs_direction,
            )
            if spot_leg is not None and direction in ("up", "down"):
                if spot_fill_mode_is_deferred(entry_decision.fill_mode):
                    same_mode = normalize_spot_fill_mode(
                        entry_decision.fill_mode,
                        default=spot_entry_fill_mode,
                    ) == str(entry_plan.fill_mode)
                    entry_schedule = lifecycle_deferred_entry_plan(
                        fill_mode=str(entry_decision.fill_mode),
                        signal_ts=bar.ts,
                        signal_close_ts=bar.ts,
                        exec_bar_size=spot_exec_bar_size,
                        strategy=cfg.strategy,
                        riskoff_today=bool(riskoff_today),
                        riskoff_end_hour=riskoff_end_hour,
                        exit_mode=exit_mode,
                        atr_value=float(atr) if atr is not None else None,
                        naive_ts_mode="utc",
                        due_ts=entry_plan.due_ts if same_mode else None,
                    )
                    if next_bar is None and entry_schedule.deferred:
                        entry_schedule = replace(entry_schedule, due_ts=None, allowed=False, reason="no_next_bar")
                    can_schedule = bool(
                        entry_schedule.allowed
                        and entry_schedule.due_ts is not None
                        and pending_entry_dir is None
                    )
                    if can_schedule:
                        pending_entry_dir = direction
                        pending_entry_branch = entry_signal_branch if entry_signal_branch in ("a", "b") else None
                        pending_entry_set_date = _trade_date(bar.ts)
                        pending_entry_due_ts = entry_schedule.due_ts
                        last_entry_sig_idx = int(sig_idx)
                else:
                    liquidation_close = _spot_liquidation(float(bar.close))
                    opened = _spot_try_open_entry(
                        cfg=cfg,
                        meta=meta,
                        entry_signal=entry_signal,
                        entry_dir=direction,
                        entry_branch=entry_signal_branch if entry_signal_branch in ("a", "b") else None,
                        entry_leg=spot_leg,
                        entry_time=sig_bar.ts,
                        entry_ref_price=float(bar.close),
                        mark_ref_price=float(bar.close),
                        atr_value=float(atr) if atr is not None else None,
                        exit_mode=exit_mode,
                        orb_engine=orb_engine,
                        filters=filters,
                        shock_now=shock,
                        shock_dir_now=shock_dir,
                        shock_atr_pct_now=shock_atr_pct,
                        shock_dir_down_streak_bars_now=(
                            int(shock_dir_down_streak_bars) if shock_dir_down_streak_bars is not None else None
                        ),
                        shock_drawdown_dist_on_pct_now=(
                            float(shock_drawdown_dist_on_pct) if shock_drawdown_dist_on_pct is not None else None
                        ),
                        shock_drawdown_dist_on_vel_pp_now=(
                            float(shock_drawdown_dist_on_vel_pp) if shock_drawdown_dist_on_vel_pp is not None else None
                        ),
                        shock_drawdown_dist_on_accel_pp_now=(
                            float(getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", 0.0))
                            if last_sig_snap is not None
                            and getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", None) is not None
                            else None
                        ),
                        shock_prearm_down_streak_bars_now=(
                            int(getattr(last_sig_snap, "shock_prearm_down_streak_bars", 0))
                            if last_sig_snap is not None
                            and getattr(last_sig_snap, "shock_prearm_down_streak_bars", None) is not None
                            else None
                        ),
                        shock_ramp_now=(
                            dict(getattr(last_sig_snap, "shock_ramp"))
                            if last_sig_snap is not None and isinstance(getattr(last_sig_snap, "shock_ramp", None), dict)
                            else None
                        ),
                        signal_entry_dir_now=(
                            str(entry_signal_dir) if entry_signal_dir in ("up", "down") else None
                        ),
                        signal_regime_dir_now=(
                            str(entry_regime_dir) if entry_regime_dir in ("up", "down") else None
                        ),
                        riskoff_today=bool(riskoff_today),
                        riskpanic_today=bool(riskpanic_today),
                        riskpop_today=bool(riskpop_today),
                        risk_snapshot=evaluator.last_risk if risk_overlay_enabled else None,
                        cash=float(cash),
                        margin_used=float(margin_used),
                        liquidation_value=float(liquidation_close),
                        spread=float(spot_spread),
                        commission_per_share=float(spot_commission),
                        commission_min=float(spot_commission_min),
                        slippage_per_share=float(spot_slippage),
                        mark_to_market=str(spot_mark_to_market),
                    )
                    if opened is not None:
                        cash, margin_used = _spot_apply_opened_trade(opened=opened, open_trades=open_trades)
                        entries_today += 1
                        last_entry_sig_idx = int(sig_idx)

        if sig_idx is not None and open_trades and (not pending_exit_all) and len(open_trades) == 1:
            trade = open_trades[0]
            trade_dir = "up" if int(trade.qty) > 0 else "down"
            resize_leg = _spot_entry_leg_for_direction(
                strategy=cfg.strategy,
                entry_dir=trade_dir,
                needs_direction=needs_direction,
            )
            if resize_leg is not None and trade_dir in ("up", "down"):
                action = str(getattr(resize_leg, "action", "BUY") or "BUY").strip().upper()
                lot = max(1, int(getattr(resize_leg, "qty", 1) or 1))
                base_signed_qty = int(lot) * int(cfg.strategy.quantity)
                if action != "BUY":
                    base_signed_qty = -base_signed_qty
                side = "buy" if action == "BUY" else "sell"
                entry_price_est = _spot_exec_price(
                    float(bar.close),
                    side=side,
                    qty=int(base_signed_qty),
                    spread=float(spot_spread),
                    commission_per_share=float(spot_commission),
                    commission_min=float(spot_commission_min),
                    slippage_per_share=float(spot_slippage),
                )

                stop_price = float(trade.stop_loss_price) if trade.stop_loss_price is not None else None
                if stop_price is not None and stop_price <= 0:
                    stop_price = None
                stop_loss_pct = float(trade.stop_loss_pct) if trade.stop_loss_pct is not None else None
                if stop_loss_pct is not None and stop_loss_pct <= 0:
                    stop_loss_pct = None

                liquidation_close = _spot_liquidation(float(bar.close))
                signed_target, resize_trace = spot_calc_signed_qty_with_trace(
                    strategy=cfg.strategy,
                    filters=filters,
                    action=action,
                    lot=int(lot),
                    entry_price=float(entry_price_est),
                    stop_price=stop_price,
                    stop_loss_pct=stop_loss_pct,
                    shock=shock,
                    shock_dir=shock_dir,
                    shock_atr_pct=shock_atr_pct,
                    shock_dir_down_streak_bars=(
                        int(shock_dir_down_streak_bars) if shock_dir_down_streak_bars is not None else None
                    ),
                    shock_drawdown_dist_on_pct=(
                        float(shock_drawdown_dist_on_pct) if shock_drawdown_dist_on_pct is not None else None
                    ),
                    shock_drawdown_dist_on_vel_pp=(
                        float(shock_drawdown_dist_on_vel_pp) if shock_drawdown_dist_on_vel_pp is not None else None
                    ),
                    shock_drawdown_dist_on_accel_pp=(
                        float(getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", 0.0))
                        if last_sig_snap is not None
                        and getattr(last_sig_snap, "shock_drawdown_dist_on_accel_pp", None) is not None
                        else None
                    ),
                    shock_prearm_down_streak_bars=(
                        int(getattr(last_sig_snap, "shock_prearm_down_streak_bars", 0))
                        if last_sig_snap is not None
                        and getattr(last_sig_snap, "shock_prearm_down_streak_bars", None) is not None
                        else None
                    ),
                    shock_ramp=getattr(last_sig_snap, "shock_ramp", None) if last_sig_snap is not None else None,
                    riskoff=bool(riskoff_today),
                    risk_dir=shock_dir,
                    riskpanic=bool(riskpanic_today),
                    riskpop=bool(riskpop_today),
                    risk=evaluator.last_risk if risk_overlay_enabled else None,
                    signal_entry_dir=str(entry_signal_dir) if entry_signal_dir in ("up", "down") else None,
                    signal_regime_dir=str(entry_regime_dir) if entry_regime_dir in ("up", "down") else None,
                    equity_ref=float(cash) + float(liquidation_close),
                    cash_ref=float(cash),
                )
                if int(signed_target) != 0:
                    size_mult = _spot_branch_size_mult(strategy=cfg.strategy, entry_branch=trade.entry_branch)
                    signed_target = spot_apply_branch_size_mult(
                        signed_qty=int(signed_target),
                        size_mult=float(size_mult),
                        spot_min_qty=int(getattr(cfg.strategy, "spot_min_qty", 1) or 1),
                        spot_max_qty=int(getattr(cfg.strategy, "spot_max_qty", 0) or 0),
                    )
                    resize_trace = resize_trace.with_branch_scaling(
                        entry_branch=trade.entry_branch,
                        size_mult=float(size_mult),
                        signed_qty_after_branch=int(signed_target),
                    )
                    lifecycle = _spot_open_position_intent(
                        strategy=cfg.strategy,
                        bar_ts=bar.ts,
                        bar_size=str(cfg.backtest.bar_size),
                        open_dir=trade_dir,
                        current_qty=int(trade.qty),
                        target_qty=int(signed_target),
                        spot_decision=resize_trace.as_payload(),
                        last_resize_bar_ts=last_resize_bar_ts,
                        signal_entry_dir=entry_signal_dir,
                        shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
                        shock_atr_vel_pct=float(shock_atr_vel) if shock_atr_vel is not None else None,
                        shock_atr_accel_pct=float(shock_atr_accel) if shock_atr_accel is not None else None,
                        tr_ratio=float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                        tr_median_pct=(
                            float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                        ),
                        slope_med_pct=float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None,
                        slope_vel_pct=float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None,
                        slope_med_slow_pct=float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None,
                        slope_vel_slow_pct=float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None,
                    )
                    _capture_lifecycle(
                        stage="open_resize",
                        decision=lifecycle,
                        bar_ts=bar.ts,
                        exec_idx=int(idx),
                        sig_idx=int(sig_idx),
                        context={
                            "shock_atr_pct": float(shock_atr_pct) if shock_atr_pct is not None else None,
                            "shock_atr_vel_pct": float(shock_atr_vel) if shock_atr_vel is not None else None,
                            "shock_atr_accel_pct": float(shock_atr_accel) if shock_atr_accel is not None else None,
                            "ratsv_tr_ratio": float(ratsv_tr_ratio) if ratsv_tr_ratio is not None else None,
                            "risk_tr_median_pct": (
                                float(risk_tr_median_pct) if risk_tr_median_pct is not None else None
                            ),
                            "ratsv_fast_slope_med_pct": (
                                float(ratsv_fast_slope_med) if ratsv_fast_slope_med is not None else None
                            ),
                            "ratsv_fast_slope_vel_pct": (
                                float(ratsv_fast_slope_vel) if ratsv_fast_slope_vel is not None else None
                            ),
                            "ratsv_slow_slope_med_pct": (
                                float(ratsv_slow_slope_med) if ratsv_slow_slope_med is not None else None
                            ),
                            "ratsv_slow_slope_vel_pct": (
                                float(ratsv_slow_slope_vel) if ratsv_slow_slope_vel is not None else None
                            ),
                            "ratsv_slope_vel_consistency": (
                                float(ratsv_slope_vel_consistency)
                                if ratsv_slope_vel_consistency is not None
                                else None
                            ),
                        },
                    )
                    if lifecycle.intent == "resize" and lifecycle.spot_intent is not None:
                        applied, cash, margin_used = _spot_apply_resize_trade(
                            trade=trade,
                            delta_qty=int(lifecycle.spot_intent.delta_qty),
                            entry_ref_price=float(bar.close),
                            mark_ref_price=float(bar.close),
                            cash=float(cash),
                            margin_used=float(margin_used),
                            spread=float(spot_spread),
                            commission_per_share=float(spot_commission),
                            commission_min=float(spot_commission_min),
                            slippage_per_share=float(spot_slippage),
                            mark_to_market=str(spot_mark_to_market),
                            multiplier=float(meta.multiplier),
                            lifecycle_payload=lifecycle.as_payload(),
                            decision_payload=resize_trace.as_payload(),
                        )
                        if applied:
                            last_resize_bar_ts = bar.ts
                    elif lifecycle.intent == "exit":
                        if spot_fill_mode_is_deferred(lifecycle.fill_mode) and next_bar is not None:
                            pending_exit_due_ts = _spot_fill_due_ts(
                                strategy=cfg.strategy,
                                fill_mode=str(lifecycle.fill_mode),
                                signal_close_ts=bar.ts,
                                exec_bar_size=spot_exec_bar_size,
                                signal_use_rth=bool(cfg.backtest.use_rth),
                                spot_sec_type=spot_sec_type,
                            )
                            if pending_exit_due_ts is not None:
                                pending_exit_all = True
                                pending_exit_reason = str(lifecycle.reason or "target_zero")
                            else:
                                _exit_price, cash, margin_used = _exec_trade_exit(
                                    trade,
                                    exit_ref_price=float(bar.close),
                                    exit_time=bar.ts,
                                    reason=str(lifecycle.reason or "target_zero"),
                                    apply_slippage=True,
                                )
                                open_trades = []
                        else:
                            _exit_price, cash, margin_used = _exec_trade_exit(
                                trade,
                                exit_ref_price=float(bar.close),
                                exit_time=bar.ts,
                                reason=str(lifecycle.reason or "target_zero"),
                                apply_slippage=True,
                            )
                            open_trades = []

        shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view

    if open_trades:
        last_bar = exec_bars[-1]
        for trade in open_trades:
            _exit_price, cash, margin_used = _exec_trade_exit(
                trade,
                exit_ref_price=float(last_bar.close),
                exit_time=last_bar.ts,
                reason="end",
                apply_slippage=True,
            )

    summary = _summarize_from_trades_and_max_dd(
        trades,
        starting_cash=cfg.backtest.starting_cash,
        max_dd=float(equity_max_dd),
        multiplier=meta.multiplier,
    )
    trace_path = str(trace_out_raw or "").strip()
    if trace_path and lifecycle_rows is not None:
        write_rows_csv(rows=lifecycle_rows, out_path=trace_path)
    why_not_path = str(why_not_out_raw or "").strip()
    if why_not_path and lifecycle_rows is not None:
        write_rows_csv(rows=why_not_exit_resize_report(lifecycle_rows), out_path=why_not_path)
    return BacktestResult(
        trades=trades,
        equity=equity_curve,
        summary=summary,
        lifecycle_trace=lifecycle_rows if lifecycle_rows is not None else None,
    )


def _run_spot_backtest_exec_loop_summary_fast(
    cfg: ConfigBundle,
    *,
    signal_bars: BarSeriesInput,
    exec_bars: BarSeriesInput,
    meta: ContractMeta,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    debug_trades: list[dict[str, object]] | None = None,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> SummaryStats:
    """Event-driven summary-only spot backtest for sweeps.

    Designed for high-volume sweep workloads:
    - signals evaluated on signal bars (e.g. 30m),
    - execution/exits simulated on exec bars (e.g. 5m),
    - only summary stats are needed (no full equity curve).
    """
    signal_bars = _bars_input_list(signal_bars)
    exec_bars = _bars_input_list(exec_bars)
    regime_bars = _bars_input_optional_list(regime_bars)
    regime2_bars = _bars_input_optional_list(regime2_bars)
    tick_bars = _bars_input_optional_list(tick_bars)

    if not signal_bars:
        raise ValueError("signal_bars is empty")
    if not exec_bars:
        raise ValueError("exec_bars is empty")
    _spot_emit_progress(
        progress_callback,
        phase="engine.prepare",
        path="fast",
        signal_total=int(len(signal_bars)),
        exec_total=int(len(exec_bars)),
    )

    strat = cfg.strategy
    filters = strat.filters

    exec_profile = _spot_exec_profile(strat)
    spot_entry_fill_mode = str(exec_profile.entry_fill_mode)
    spot_flip_exit_fill_mode = str(exec_profile.flip_fill_mode)
    exit_on_signal_flip = bool(getattr(strat, "exit_on_signal_flip", False))
    spot_exec_bar_size = str(getattr(strat, "spot_exec_bar_size", "") or cfg.backtest.bar_size or "")
    spot_sec_type = _spot_strategy_sec_type(strategy=strat)
    if not spot_fill_mode_is_deferred(spot_entry_fill_mode):
        raise ValueError(
            "fast summary runner requires spot_entry_fill_mode in {'next_bar','next_tradable_bar'}"
        )
    if exit_on_signal_flip and not spot_fill_mode_is_deferred(spot_flip_exit_fill_mode):
        raise ValueError(
            "fast summary runner requires deferred spot_flip_exit_fill_mode when flip exits are enabled"
        )

    spot_spread = float(exec_profile.spread)
    spot_commission = float(exec_profile.commission_per_share)
    spot_commission_min = float(exec_profile.commission_min)
    spot_slippage = float(exec_profile.slippage_per_share)
    spot_mark_to_market = str(exec_profile.mark_to_market)
    spot_drawdown_mode = str(exec_profile.drawdown_mode)
    if spot_drawdown_mode != "intrabar":
        raise ValueError("fast summary runner currently only supports spot_drawdown_mode='intrabar'")

    base_stop_pct = strat.spot_stop_loss_pct
    if base_stop_pct is None or float(base_stop_pct) <= 0:
        raise ValueError("fast summary runner requires spot_stop_loss_pct for pct exits")
    base_pt_pct = strat.spot_profit_target_pct

    max_entries_per_day = int(getattr(strat, "max_entries_per_day", 0) or 0)

    needs_direction = strat.directional_spot is not None
    signal_bar_hours = float(_bar_hours(str(cfg.backtest.bar_size)))
    (
        tick_mode,
        tick_neutral_policy,
        _tick_direction_policy,
        _tick_ma_period,
        _tick_z_lookback,
        _tick_z_enter,
        _tick_z_exit,
        _tick_slope_lookback,
    ) = _spot_tick_gate_settings(strat)
    needs_rv = filters is not None and (
        getattr(filters, "rv_min", None) is not None or getattr(filters, "rv_max", None) is not None
    )
    needs_volume = filters is not None and getattr(filters, "volume_ratio_min", None) is not None

    # Precompute reusable series/trees (cached across configs via bar ids + knob keys).
    series_pack = prepared_series_pack if isinstance(prepared_series_pack, _SpotSeriesPack) else None
    if series_pack is None:
        series_pack = _spot_build_series_pack(
            cfg=cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
            include_rv=bool(needs_rv),
            include_volume=bool(needs_volume),
            include_tick=(str(tick_mode) != "off"),
        )
    align = series_pack.core.align
    signal_series = series_pack.core.signal_series
    tick_series = series_pack.core.tick_series
    shock_series = series_pack.shock_series
    risk_by_day = series_pack.risk_by_day
    rv_series = series_pack.rv_series
    volume_series = series_pack.volume_series
    risk_overlay_enabled = bool(risk_by_day) and filters is not None
    riskoff_mode, *_ = risk_overlay_policy_from_filters(filters)

    riskoff_end_hour = _spot_riskoff_end_hour(filters) if risk_overlay_enabled else None

    shock_stop_mult, shock_profit_mult = spot_shock_exit_pct_multipliers(filters, shock=True)

    dd_long_tree, dd_short_tree = _spot_dd_trees(exec_bars=exec_bars, spread=spot_spread, mark_to_market=spot_mark_to_market)
    stop_long_tree = _spot_stop_tree(
        exec_bars=exec_bars,
        shock=shock_series,
        base_stop_pct=float(base_stop_pct),
        shock_stop_mult=float(shock_stop_mult),
        direction="up",
        mode="intrabar",
    )
    stop_short_tree = _spot_stop_tree(
        exec_bars=exec_bars,
        shock=shock_series,
        base_stop_pct=float(base_stop_pct),
        shock_stop_mult=float(shock_stop_mult),
        direction="down",
        mode="intrabar",
    )

    profit_long_tree = None
    profit_short_tree = None
    if base_pt_pct is not None and float(base_pt_pct) > 0:
        profit_long_tree = _spot_profit_tree(
            exec_bars=exec_bars,
            shock=shock_series,
            base_profit_pct=float(base_pt_pct),
            shock_profit_mult=float(shock_profit_mult),
            direction="up",
            mode="intrabar",
        )
        profit_short_tree = _spot_profit_tree(
            exec_bars=exec_bars,
            shock=shock_series,
            base_profit_pct=float(base_pt_pct),
            shock_profit_mult=float(shock_profit_mult),
            direction="down",
            mode="intrabar",
        )

    flip_long_tree, flip_short_tree = _spot_flip_trees(
        signal_bars=signal_bars,
        signal_series=signal_series,
        exit_on_signal_flip=bool(exit_on_signal_flip),
        flip_exit_mode=str(getattr(strat, "flip_exit_mode", "entry") or "entry"),
        ema_entry_mode=str(getattr(strat, "ema_entry_mode", "trend") or "trend"),
        only_if_profit=bool(getattr(strat, "flip_exit_only_if_profit", False)),
    )
    flip_next_up, flip_next_down = _spot_flip_next_sig_idx(
        signal_series=signal_series,
        exit_on_signal_flip=bool(exit_on_signal_flip),
        flip_exit_mode=str(getattr(strat, "flip_exit_mode", "entry") or "entry"),
        ema_entry_mode=str(getattr(strat, "ema_entry_mode", "trend") or "trend"),
    )
    if bool(exit_on_signal_flip) and bool(getattr(strat, "flip_exit_only_if_profit", False)):
        if flip_long_tree is None or flip_short_tree is None:
            raise ValueError("fast summary runner requires flip trees when flip_exit_only_if_profit is enabled")

    signal_ts = [b.ts for b in signal_bars]
    exec_ts = [b.ts for b in exec_bars]

    def _next_fill_from_signal_exec_idx(sig_exec_idx: int, *, fill_mode: str) -> tuple[int, datetime] | None:
        if sig_exec_idx < 0 or sig_exec_idx >= len(exec_bars):
            return None
        due_ts = _spot_fill_due_ts(
            strategy=strat,
            fill_mode=str(fill_mode),
            signal_close_ts=exec_bars[int(sig_exec_idx)].ts,
            exec_bar_size=spot_exec_bar_size,
            signal_use_rth=bool(cfg.backtest.use_rth),
            spot_sec_type=spot_sec_type,
        )
        if due_ts is None:
            return None
        entry_exec_idx = int(bisect_right(exec_ts, due_ts))
        entry_exec_idx = max(int(sig_exec_idx) + 1, int(entry_exec_idx))
        if entry_exec_idx < 0 or entry_exec_idx >= len(exec_bars):
            return None
        return int(entry_exec_idx), due_ts

    def _risk_flags(day: date) -> tuple[bool, bool, bool]:
        if not risk_overlay_enabled:
            return False, False, False
        snap = risk_by_day.get(day)
        if snap is None:
            return False, False, False
        return bool(getattr(snap, "riskoff", False)), bool(getattr(snap, "riskpanic", False)), bool(
            getattr(snap, "riskpop", False)
        )

    def _shock_prev(exec_idx: int) -> tuple[bool, str | None, float | None]:
        if exec_idx <= 0:
            return False, None, None
        prev = exec_idx - 1
        shock_now = shock_series.shock_by_exec_idx[prev]
        shock_dir_now = shock_series.shock_dir_by_exec_idx[prev]
        shock_atr_pct_now = shock_series.shock_atr_pct_by_exec_idx[prev]
        return (
            bool(shock_now) if shock_now is not None else False,
            str(shock_dir_now) if shock_dir_now in ("up", "down") else None,
            float(shock_atr_pct_now) if shock_atr_pct_now is not None else None,
        )

    stop_pct_by_exec = _spot_effective_pct_by_exec_idx(
        shock_by_exec_idx=shock_series.shock_by_exec_idx,
        base_pct=float(base_stop_pct),
        mult=float(shock_stop_mult),
    )
    pt_pct_by_exec = _spot_effective_pct_by_exec_idx(
        shock_by_exec_idx=shock_series.shock_by_exec_idx,
        base_pct=float(base_pt_pct) if base_pt_pct is not None else None,
        mult=float(shock_profit_mult),
    )

    def _maybe_cancel_pending_entry(
        *,
        pending_dir: str,
        pending_set_date: date | None,
        exec_idx: int,
    ) -> bool:
        """Return True if a next-open entry should be canceled at execution time."""
        ts = exec_bars[exec_idx].ts
        day = _trade_date(ts)
        riskoff_today, riskpanic_today, riskpop_today = _risk_flags(day)
        _shock_on, shock_dir_now, _shock_atr = _shock_prev(exec_idx)
        return _spot_pending_entry_should_cancel(
            pending_dir=str(pending_dir),
            pending_set_date=pending_set_date,
            exec_ts=ts,
            risk_overlay_enabled=bool(risk_overlay_enabled),
            riskoff_today=bool(riskoff_today),
            riskpanic_today=bool(riskpanic_today),
            riskpop_today=bool(riskpop_today),
            riskoff_mode=str(riskoff_mode),
            shock_dir_now=shock_dir_now,
            riskoff_end_hour=riskoff_end_hour,
        )

    cash = float(cfg.backtest.starting_cash)
    peak_equity = float(cash)
    max_dd = 0.0

    trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    win_sum = 0.0
    loss_sum = 0.0
    hold_sum = 0.0
    hold_n = 0

    entries_today = 0
    entries_today_date: date | None = None
    last_entry_sig_idx: int | None = None

    open_qty = 0
    open_entry_exec_idx = -1
    open_entry_time: datetime | None = None
    open_entry_price = 0.0
    open_margin_required = 0.0
    open_decision_trace: dict[str, object] | None = None

    sig_cursor = 0
    sig_total = int(len(signal_bars))
    exec_total = int(len(exec_bars))
    progress_stride = max(1, int(max(16, sig_total // 200)))
    last_progress_sig = -1

    def _emit_fast_progress(*, exec_idx_hint: int | None = None, force: bool = False) -> None:
        nonlocal last_progress_sig
        sig_done = max(0, min(int(sig_total), int(sig_cursor)))
        if not bool(force):
            if sig_done <= 0:
                return
            if int(sig_done - int(last_progress_sig)) < int(progress_stride):
                return
        last_progress_sig = int(sig_done)
        exec_done = None
        if exec_idx_hint is not None:
            exec_done = int(max(0, int(exec_idx_hint) + 1))
        elif sig_done > 0 and (sig_done - 1) < len(align.exec_idx_by_sig_idx):
            mapped = int(align.exec_idx_by_sig_idx[int(sig_done - 1)])
            if mapped >= 0:
                exec_done = int(mapped + 1)
        _spot_emit_progress(
            progress_callback,
            phase="engine.exec",
            path="fast",
            sig_idx=int(sig_done),
            sig_total=int(sig_total),
            exec_idx=(int(exec_done) if exec_done is not None else None),
            exec_total=int(exec_total),
            open_count=(1 if int(open_qty) != 0 else 0),
            trades=int(trades),
        )

    def _set_day_for_exec_idx(exec_idx: int) -> None:
        nonlocal entries_today_date, entries_today
        d = _trade_date(exec_bars[exec_idx].ts)
        if entries_today_date != d:
            entries_today_date = d
            entries_today = 0

    def _apply_opened_entry_state(*, opened: SpotTrade, entry_exec_idx: int) -> None:
        nonlocal cash, open_qty, open_entry_exec_idx, open_entry_time, open_entry_price, open_margin_required
        nonlocal open_decision_trace
        nonlocal entries_today
        (
            open_qty,
            open_entry_exec_idx,
            open_entry_time,
            open_entry_price,
            open_margin_required,
            cash,
            open_decision_trace,
        ) = _spot_opened_trade_summary_state(
            opened=opened,
            entry_exec_idx=int(entry_exec_idx),
        )
        entries_today += 1

    def _clear_open_entry_state() -> None:
        nonlocal open_qty, open_entry_exec_idx, open_entry_time, open_entry_price, open_margin_required
        nonlocal open_decision_trace
        open_qty = 0
        open_entry_exec_idx = -1
        open_entry_time = None
        open_entry_price = 0.0
        open_margin_required = 0.0
        open_decision_trace = None

    def _try_open_entry_from_signal(
        *,
        sig_idx: int,
        sig_exec_idx: int,
        entry_exec_idx: int,
        fill_mode: str,
        fill_due_ts: datetime,
    ) -> bool:
        nonlocal cash, open_qty, open_entry_exec_idx, open_entry_time, open_entry_price, open_margin_required
        nonlocal entries_today, last_entry_sig_idx
        if sig_exec_idx < 0 or entry_exec_idx < 0 or entry_exec_idx >= len(exec_bars):
            return False

        _set_day_for_exec_idx(sig_exec_idx)

        sig = signal_series.signal_by_sig_idx[sig_idx]
        entry_branch = signal_series.entry_branch_by_sig_idx[sig_idx]
        entry_dir, _ = _spot_resolve_entry_dir(
            signal=sig,
            entry_dir=signal_series.entry_dir_by_sig_idx[sig_idx],
            ema_needed=True,
            sig_idx=int(sig_idx),
            tick_mode=str(tick_mode),
            tick_series=tick_series,
            tick_neutral_policy=str(tick_neutral_policy),
            needs_direction=bool(needs_direction),
            directional_spot=strat.directional_spot,
        )
        if entry_dir is None:
            return False

        cooldown_ok = cooldown_ok_by_index(
            current_idx=int(sig_idx),
            last_entry_idx=last_entry_sig_idx,
            cooldown_bars=filters.cooldown_bars if filters else 0,
        )
        shock_now = shock_series.shock_by_sig_idx[sig_idx]
        shock_dir_now = shock_series.shock_dir_by_sig_idx[sig_idx]
        rv_now = rv_series.rv_by_sig_idx[sig_idx] if rv_series is not None else None
        volume_ema_now = volume_series.volume_ema_by_sig_idx[sig_idx] if volume_series is not None else None
        volume_ema_ready_now = volume_series.volume_ema_ready_by_sig_idx[sig_idx] if volume_series is not None else True
        riskoff_today, _riskpanic_today, _riskpop_today = _risk_flags(_trade_date(exec_bars[sig_exec_idx].ts))
        entry_plan = lifecycle_deferred_entry_plan(
            fill_mode=str(fill_mode),
            signal_ts=exec_bars[sig_exec_idx].ts,
            signal_close_ts=exec_bars[sig_exec_idx].ts,
            exec_bar_size=spot_exec_bar_size,
            strategy=strat,
            riskoff_today=bool(riskoff_today),
            riskoff_end_hour=riskoff_end_hour,
            exit_mode="pct",
            atr_value=None,
            naive_ts_mode="utc",
            due_ts=fill_due_ts,
        )

        entry_decision = _spot_flat_entry_decision_from_signal(
            strategy=strat,
            filters=filters,
            signal_bar=signal_bars[sig_idx],
            signal=sig,
            direction=str(entry_dir) if entry_dir in ("up", "down") else None,
            bars_in_day=int(signal_series.bars_in_day_by_sig_idx[sig_idx]),
            volume_ema=float(volume_ema_now) if volume_ema_now is not None else None,
            volume_ema_ready=bool(volume_ema_ready_now),
            rv=float(rv_now) if rv_now is not None else None,
            cooldown_ok=bool(cooldown_ok),
            shock=shock_now,
            shock_dir=shock_dir_now,
            open_count=0,
            entries_today=int(entries_today),
            pending_exists=False,
            next_open_allowed=bool(entry_plan.allowed),
            exit_mode="pct",
            atr_value=None,
            shock_atr_pct=(
                float(shock_series.shock_atr_pct_by_sig_idx[sig_idx])
                if shock_series.shock_atr_pct_by_sig_idx[sig_idx] is not None
                else None
            ),
            shock_atr_vel_pct=None,
            shock_atr_accel_pct=None,
            tr_ratio=None,
            tr_median_pct=None,
            slope_med_pct=None,
            slope_vel_pct=None,
            slope_med_slow_pct=None,
            slope_vel_slow_pct=None,
        )
        if entry_decision.intent != "enter":
            return False
        if not spot_fill_mode_is_deferred(entry_decision.fill_mode):
            return False

        last_entry_sig_idx = int(sig_idx)
        pending_set_date = _trade_date(exec_bars[sig_exec_idx].ts)
        pending_dir = str(entry_decision.direction or entry_dir)
        if pending_dir not in ("up", "down"):
            return False
        pending_branch = str(entry_branch) if entry_branch in ("a", "b") else None

        _set_day_for_exec_idx(entry_exec_idx)
        if not _spot_entry_capacity_ok(
            open_count=0,
            max_entries_per_day=int(max_entries_per_day),
            entries_today=int(entries_today),
            weekday=_trade_weekday(exec_bars[entry_exec_idx].ts),
            entry_days=strat.entry_days,
        ):
            return False
        if _maybe_cancel_pending_entry(
            pending_dir=pending_dir,
            pending_set_date=pending_set_date,
            exec_idx=entry_exec_idx,
        ):
            return False

        entry_leg = _spot_entry_leg_for_direction(
            strategy=strat,
            entry_dir=pending_dir,
            needs_direction=needs_direction,
        )
        if entry_leg is None:
            return False

        bar = exec_bars[entry_exec_idx]
        shock_prev_on, shock_dir_prev_now, shock_atr_pct_prev_now = _shock_prev(entry_exec_idx)
        riskoff_fill, riskpanic_fill, riskpop_fill = _risk_flags(_trade_date(bar.ts))
        opened = _spot_try_open_entry(
            cfg=cfg,
            meta=meta,
            entry_signal="ema",
            entry_dir=pending_dir,
            entry_branch=pending_branch,
            entry_leg=entry_leg,
            entry_time=bar.ts,
            entry_ref_price=float(bar.open),
            mark_ref_price=float(bar.open),
            atr_value=None,
            exit_mode="pct",
            orb_engine=None,
            filters=filters,
            shock_now=bool(shock_prev_on),
            shock_dir_now=shock_dir_prev_now,
            shock_atr_pct_now=shock_atr_pct_prev_now,
            shock_dir_down_streak_bars_now=None,
            shock_drawdown_dist_on_pct_now=None,
            shock_drawdown_dist_on_vel_pp_now=None,
            shock_drawdown_dist_on_accel_pp_now=None,
            shock_prearm_down_streak_bars_now=None,
            shock_ramp_now=None,
            signal_entry_dir_now=str(pending_dir) if pending_dir in ("up", "down") else None,
            signal_regime_dir_now=(
                str(getattr(sig, "regime_dir", None)) if getattr(sig, "regime_dir", None) in ("up", "down") else None
            ),
            riskoff_today=bool(riskoff_fill),
            riskpanic_today=bool(riskpanic_fill),
            riskpop_today=bool(riskpop_fill),
            risk_snapshot=risk_by_day.get(_trade_date(bar.ts)),
            cash=float(cash),
            margin_used=0.0,
            liquidation_value=0.0,
            spread=float(spot_spread),
            commission_per_share=float(spot_commission),
            commission_min=float(spot_commission_min),
            slippage_per_share=float(spot_slippage),
            mark_to_market=str(spot_mark_to_market),
        )
        if opened is None:
            return False

        _apply_opened_entry_state(opened=opened, entry_exec_idx=int(entry_exec_idx))
        return True

    def _try_flip_same_bar_reentry(
        *,
        exit_reason: str,
        flip_sig_idx: int | None,
        flip_exec_idx: int | None,
        flip_due_ts: datetime | None,
    ) -> bool:
        nonlocal sig_cursor
        # Full spot engine only queues an immediate flip re-entry when controlled flip is enabled.
        # Without that knob, a flip exit just closes the position and the next entry is evaluated
        # on subsequent signal bars. Keeping this aligned avoids over-trading in fast summary mode.
        if not bool(getattr(strat, "spot_controlled_flip", False)):
            return False
        if exit_reason != "flip" or flip_sig_idx is None or flip_exec_idx is None or flip_due_ts is None:
            return False
        sig_idx = int(flip_sig_idx)
        sig_exec = align.exec_idx_by_sig_idx[sig_idx]
        if _try_open_entry_from_signal(
            sig_idx=int(sig_idx),
            sig_exec_idx=int(sig_exec),
            entry_exec_idx=int(flip_exec_idx),
            fill_mode=spot_entry_fill_mode,
            fill_due_ts=flip_due_ts,
        ):
            sig_cursor = max(int(sig_cursor), int(sig_idx + 1))
            return True
        return False

    def _is_intrabar_event_reason(reason: str) -> bool:
        key = str(reason or "").strip().lower()
        return key in (
            "stop",
            "stop_loss",
            "stop_loss_pct",
            "profit",
            "profit_target",
            "profit_target_pct",
        )

    def _is_profit_event_reason(reason: str) -> bool:
        key = str(reason or "").strip().lower()
        return key in ("profit", "profit_target", "profit_target_pct")

    def _advance_sig_cursor_after_exit(*, exit_reason: str, exit_exec_idx: int) -> None:
        nonlocal sig_cursor
        if _is_intrabar_event_reason(exit_reason):
            sig_at_exit = align.sig_idx_by_exec_idx[int(exit_exec_idx)]
            if sig_at_exit >= 0:
                sig_cursor = max(int(sig_cursor), int(sig_at_exit))
                return
            sig_cursor = max(int(sig_cursor), int(bisect_right(signal_ts, exec_bars[int(exit_exec_idx)].ts)))
            return
        sig_cursor = max(int(sig_cursor), int(bisect_left(signal_ts, exec_bars[int(exit_exec_idx)].ts)))

    while True:
        if open_qty == 0:
            # Find the next signal close that schedules an entry.
            opened = False
            while sig_cursor < len(signal_bars):
                sig_idx = sig_cursor
                sig_cursor += 1
                _emit_fast_progress()

                sig_exec_idx = align.exec_idx_by_sig_idx[sig_idx]
                if sig_exec_idx < 0:
                    continue
                next_fill = _next_fill_from_signal_exec_idx(
                    int(sig_exec_idx),
                    fill_mode=spot_entry_fill_mode,
                )
                if next_fill is None:
                    continue
                entry_exec_idx, next_fill_due_ts = next_fill
                if _try_open_entry_from_signal(
                    sig_idx=int(sig_idx),
                    sig_exec_idx=int(sig_exec_idx),
                    entry_exec_idx=int(entry_exec_idx),
                    fill_mode=spot_entry_fill_mode,
                    fill_due_ts=next_fill_due_ts,
                ):
                    opened = True
                    break

            if not opened:
                break

        # Open trade: compute next exit event.
        if open_qty == 0:
            continue

        if open_entry_time is None or open_entry_exec_idx < 0:
            raise RuntimeError("internal spot fast runner error: open trade missing entry state")

        trade_dir = "up" if open_qty > 0 else "down"
        stop_idx = None
        if trade_dir == "up":
            stop_idx = stop_long_tree.find_first_leq(start=open_entry_exec_idx, threshold=float(open_entry_price))  # type: ignore[attr-defined]
        else:
            stop_idx = stop_short_tree.find_first_ge(start=open_entry_exec_idx, threshold=float(open_entry_price))  # type: ignore[attr-defined]

        profit_idx = None
        if base_pt_pct is not None and float(base_pt_pct) > 0 and profit_long_tree is not None and profit_short_tree is not None:
            if trade_dir == "up":
                profit_idx = profit_long_tree.find_first_ge(start=open_entry_exec_idx, threshold=float(open_entry_price))  # type: ignore[attr-defined]
            else:
                profit_idx = profit_short_tree.find_first_leq(start=open_entry_exec_idx, threshold=float(open_entry_price))  # type: ignore[attr-defined]

        flip_sig_idx = None
        flip_exec_idx = None
        flip_due_ts = None
        if bool(exit_on_signal_flip) and strat.direction_source == "ema":
            hold_bars = int(getattr(strat, "flip_exit_min_hold_bars", 0) or 0)
            hold_hours = float(signal_bar_hours) * float(max(0, hold_bars))
            hold_start_ts = open_entry_time + timedelta(hours=hold_hours)
            start_sig = bisect_left(signal_ts, hold_start_ts)
            start_sig = max(int(start_sig), int(sig_cursor))
            if bool(getattr(strat, "flip_exit_only_if_profit", False)):
                if trade_dir == "up" and flip_long_tree is not None:
                    flip_sig_idx = flip_long_tree.find_first_gt(start=start_sig, threshold=float(open_entry_price))
                elif trade_dir == "down" and flip_short_tree is not None:
                    flip_sig_idx = flip_short_tree.find_first_gt(start=start_sig, threshold=-float(open_entry_price))
            elif start_sig < len(signal_ts):
                if trade_dir == "up":
                    nxt = flip_next_up[int(start_sig)] if int(start_sig) < len(flip_next_up) else -1
                    flip_sig_idx = int(nxt) if int(nxt) >= 0 else None
                else:
                    nxt = flip_next_down[int(start_sig)] if int(start_sig) < len(flip_next_down) else -1
                    flip_sig_idx = int(nxt) if int(nxt) >= 0 else None
            if flip_sig_idx is not None:
                sc_exec = align.exec_idx_by_sig_idx[int(flip_sig_idx)]
                if sc_exec >= 0:
                    next_fill = _next_fill_from_signal_exec_idx(
                        int(sc_exec),
                        fill_mode=spot_flip_exit_fill_mode,
                    )
                    if next_fill is not None:
                        flip_exec_idx, flip_due_ts = next_fill

        last_exec_idx = len(exec_bars) - 1
        stop_reason = "stop_loss_pct"
        profit_reason = "profit_target_pct"
        exit_exec_idx = int(last_exec_idx)
        exit_reason = "end"
        reason_idx: dict[str, int] = {}
        if stop_idx is not None:
            reason_idx[str(stop_reason)] = int(stop_idx)
            exit_exec_idx = min(int(exit_exec_idx), int(stop_idx))
        if profit_idx is not None:
            reason_idx[str(profit_reason)] = int(profit_idx)
            exit_exec_idx = min(int(exit_exec_idx), int(profit_idx))
        if flip_exec_idx is not None:
            reason_idx["flip"] = int(flip_exec_idx)
            exit_exec_idx = min(int(exit_exec_idx), int(flip_exec_idx))

        exit_candidates_now: dict[str, bool] = {}
        for reason_name, reason_exec_idx in reason_idx.items():
            if int(reason_exec_idx) == int(exit_exec_idx):
                exit_candidates_now[str(reason_name)] = True
        if exit_candidates_now:
            signal_entry_dir_exit = None
            if flip_sig_idx is not None and int(reason_idx.get("flip", -1)) == int(exit_exec_idx):
                signal_entry_dir_exit = signal_series.entry_dir_by_sig_idx[int(flip_sig_idx)]
            lifecycle = _spot_open_position_intent(
                strategy=strat,
                bar_ts=exec_bars[int(exit_exec_idx)].ts,
                bar_size=str(cfg.backtest.bar_size),
                open_dir=trade_dir,
                current_qty=int(open_qty),
                exit_candidates=exit_candidates_now,
                signal_entry_dir=signal_entry_dir_exit,
                shock_atr_pct=shock_series.shock_atr_pct_by_exec_idx[int(exit_exec_idx)],
            )
            resolved_exit_reason = _spot_exit_reason_from_lifecycle(
                lifecycle=lifecycle,
                exit_candidates=exit_candidates_now,
                fallback_priority=(str(stop_reason), str(profit_reason), "flip"),
            )
            if resolved_exit_reason is not None:
                exit_reason = str(resolved_exit_reason)

        # Drawdown during the holding window (mark-to-market), computed from cached segment trees.
        close_end_idx = int(exit_exec_idx + 1) if exit_reason == "end" else int(exit_exec_idx)
        dd_mark = 0.0
        worst_mark: float | None = None
        peak_mark: float | None = None
        if open_qty > 0:
            mc, ml, mdd = dd_long_tree.query(open_entry_exec_idx, close_end_idx)
            dd_mark = float(mdd)
            worst_mark = float(ml) if math.isfinite(float(ml)) else None
            peak_mark = float(mc) if math.isfinite(float(mc)) else None
        else:
            mc, mh, mdd = dd_short_tree.query(open_entry_exec_idx, close_end_idx)
            dd_mark = float(mdd)
            worst_mark = float(mh) if math.isfinite(float(mh)) else None
            peak_mark = float(mc) if math.isfinite(float(mc)) else None

        # For intrabar stop/profit exits, include the exit bar's stop-corrected worst reference.
        if _is_intrabar_event_reason(exit_reason):
            bar = exec_bars[int(exit_exec_idx)]
            stop_pct_exit = stop_pct_by_exec[int(exit_exec_idx)]
            pt_pct_exit = pt_pct_by_exec[int(exit_exec_idx)]
            stop_level = spot_stop_level(float(open_entry_price), int(open_qty), stop_loss_pct=stop_pct_exit)
            worst_ref = spot_intrabar_worst_ref(
                qty=int(open_qty),
                bar_open=float(bar.open),
                bar_high=float(bar.high),
                bar_low=float(bar.low),
                stop_level=stop_level,
            )
            worst_mark_exit = _spot_mark_price(worst_ref, qty=int(open_qty), spread=spot_spread, mode=spot_mark_to_market)

            if open_qty > 0:
                if peak_mark is not None:
                    dd_cross = float(peak_mark) - float(worst_mark_exit)
                    if dd_cross > dd_mark:
                        dd_mark = float(dd_cross)
                if worst_mark is None or float(worst_mark_exit) < float(worst_mark):
                    worst_mark = float(worst_mark_exit)
            else:
                if peak_mark is not None:
                    dd_cross = float(worst_mark_exit) - float(peak_mark)
                    if dd_cross > dd_mark:
                        dd_mark = float(dd_cross)
                if worst_mark is None or float(worst_mark_exit) > float(worst_mark):
                    worst_mark = float(worst_mark_exit)

        if worst_mark is not None:
            min_equity = float(cash) + (float(open_qty) * float(worst_mark)) * meta.multiplier
            dd_from_incoming = float(peak_equity) - float(min_equity)
            if dd_from_incoming > max_dd:
                max_dd = float(dd_from_incoming)

        dd_dollars = abs(int(open_qty)) * float(dd_mark) * meta.multiplier
        if dd_dollars > max_dd:
            max_dd = float(dd_dollars)

        if peak_mark is not None:
            peak_equity_trade = float(cash) + (float(open_qty) * float(peak_mark)) * meta.multiplier
            if peak_equity_trade > peak_equity:
                peak_equity = float(peak_equity_trade)

        # Execute exit.
        exit_bar = exec_bars[int(exit_exec_idx)]
        exit_time = exit_bar.ts
        if exit_reason == "flip":
            exit_ref = float(exit_bar.open)
        elif exit_reason == "end":
            exit_ref = float(exit_bar.close)
        else:
            stop_pct_exit = stop_pct_by_exec[int(exit_exec_idx)]
            pt_pct_exit = pt_pct_by_exec[int(exit_exec_idx)]
            stop_level = spot_stop_level(float(open_entry_price), int(open_qty), stop_loss_pct=stop_pct_exit)
            profit_level = spot_profit_level(float(open_entry_price), int(open_qty), profit_target_pct=pt_pct_exit)
            hit = spot_intrabar_exit(
                qty=int(open_qty),
                bar_open=float(exit_bar.open),
                bar_high=float(exit_bar.high),
                bar_low=float(exit_bar.low),
                stop_level=stop_level,
                profit_level=profit_level,
            )
            if hit is None:
                # Tree said it should hit, but be defensive and fallback to close.
                exit_ref = float(exit_bar.close)
            else:
                _r, exit_ref = hit
                # `spot_intrabar_exit` can only return stop/profit; keep our chosen reason (stop precedence already handled).

        exit_price, cash, _ = _spot_exec_exit_common(
            qty=int(open_qty),
            margin_required=float(open_margin_required),
            exit_ref_price=float(exit_ref),
            exit_time=exit_time,
            reason=exit_reason,
            cash=float(cash),
            margin_used=float(open_margin_required),
            spread=float(spot_spread),
            commission_per_share=float(spot_commission),
            commission_min=float(spot_commission_min),
            slippage_per_share=float(spot_slippage),
            multiplier=float(meta.multiplier),
            apply_slippage=(not _is_profit_event_reason(exit_reason)),
        )

        pnl = (float(exit_price) - float(open_entry_price)) * float(open_qty) * meta.multiplier
        (
            trades,
            wins,
            losses,
            total_pnl,
            win_sum,
            loss_sum,
            hold_sum,
            hold_n,
        ) = _summary_apply_closed_trade(
            pnl=float(pnl),
            entry_time=open_entry_time,
            exit_time=exit_time,
            trades=int(trades),
            wins=int(wins),
            losses=int(losses),
            total_pnl=float(total_pnl),
            win_sum=float(win_sum),
            loss_sum=float(loss_sum),
            hold_sum=float(hold_sum),
            hold_n=int(hold_n),
        )

        if debug_trades is not None:
            debug_trades.append(
                {
                    "qty": int(open_qty),
                    "entry_exec_idx": int(open_entry_exec_idx),
                    "entry_ts": open_entry_time,
                    "entry_price": float(open_entry_price),
                    "exit_exec_idx": int(exit_exec_idx),
                    "exit_ts": exit_time,
                    "exit_price": float(exit_price),
                    "exit_reason": str(exit_reason),
                    "spot_decision": dict(open_decision_trace) if isinstance(open_decision_trace, dict) else None,
                }
            )

        _clear_open_entry_state()

        # Flip exits can optionally schedule a same-open entry; attempt it here.
        if _try_flip_same_bar_reentry(
            exit_reason=str(exit_reason),
            flip_sig_idx=flip_sig_idx,
            flip_exec_idx=flip_exec_idx,
            flip_due_ts=flip_due_ts,
        ):
            _emit_fast_progress(exec_idx_hint=int(exit_exec_idx))
            continue

        # If we are flat at the close of this bar, update close-equity drawdown/peak.
        if exit_reason != "end":
            equity_close = float(cash)
            if equity_close > peak_equity:
                peak_equity = float(equity_close)
            dd_close = float(peak_equity) - float(equity_close)
            if dd_close > max_dd:
                max_dd = float(dd_close)

        # Resume entry scanning from the next signal close after this time.
        if exit_reason == "end":
            _emit_fast_progress(exec_idx_hint=int(exit_exec_idx), force=True)
            break
        _advance_sig_cursor_after_exit(
            exit_reason=str(exit_reason),
            exit_exec_idx=int(exit_exec_idx),
        )
        _emit_fast_progress(exec_idx_hint=int(exit_exec_idx))

    win_rate = wins / trades if trades else 0.0
    avg_win = win_sum / wins if wins else 0.0
    avg_loss = loss_sum / losses if losses else 0.0
    avg_hold = hold_sum / hold_n if hold_n else 0.0
    roi = (total_pnl / cfg.backtest.starting_cash) if cfg.backtest.starting_cash > 0 else 0.0
    max_dd_pct = (max_dd / cfg.backtest.starting_cash) if cfg.backtest.starting_cash > 0 else 0.0
    return SummaryStats(
        trades=int(trades),
        wins=int(wins),
        losses=int(losses),
        win_rate=float(win_rate),
        total_pnl=float(total_pnl),
        roi=float(roi),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        max_drawdown=float(max_dd),
        max_drawdown_pct=float(max_dd_pct),
        avg_hold_hours=float(avg_hold),
    )


def _can_use_fast_summary_path(
    cfg: ConfigBundle,
    *,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    tick_bars: list[Bar] | None,
) -> bool:
    def _is_disabled(raw: object | None) -> bool:
        if raw is None:
            return True
        if isinstance(raw, bool):
            return not bool(raw)
        if isinstance(raw, (int, float)):
            return abs(float(raw)) <= 1e-12
        text = str(raw).strip().lower()
        return text in ("", "0", "0.0", "off", "false", "none", "null", "no")

    strat = cfg.strategy
    filters = getattr(strat, "filters", None)
    policy_cfg = spot_policy_config_view(strategy=strat, filters=filters)
    policy_pack = str(getattr(policy_cfg, "spot_policy_pack", "") or "").strip().lower()
    if policy_pack not in ("", "neutral"):
        return False

    graph = SpotPolicyGraph.from_sources(strategy=strat, filters=filters)
    if str(getattr(graph, "profile_name", "neutral") or "neutral").strip().lower() != "neutral":
        return False
    if str(getattr(graph, "entry_policy", "default") or "default").strip().lower() != "default":
        return False
    if str(getattr(graph, "exit_policy", "priority") or "priority").strip().lower() != "priority":
        return False
    if str(getattr(graph, "resize_policy", "adaptive") or "adaptive").strip().lower() != "adaptive":
        return False
    if str(getattr(graph, "risk_overlay_policy", "legacy") or "legacy").strip().lower() != "legacy":
        return False

    guard_scale_mode = str(getattr(strat, "spot_guard_threshold_scale_mode", "off") or "off").strip().lower()
    if guard_scale_mode not in ("", "0", "false", "none", "null", "off"):
        return False
    hold_dynamic_mode = str(getattr(strat, "spot_flip_hold_dynamic_mode", "off") or "off").strip().lower()
    if hold_dynamic_mode not in ("", "0", "false", "none", "null", "off"):
        return False

    if not _is_disabled(getattr(strat, "spot_controlled_flip", None)):
        return False
    if str(getattr(policy_cfg, "spot_resize_mode", "off") or "off").strip().lower() == "target":
        return False
    if filters is not None:
        if bool(getattr(filters, "ratsv_enabled", False)):
            return False
        for key in _FAST_PATH_RATSV_BLOCK_KEYS:
            raw = getattr(filters, key, None)
            if raw in (None, False, 0, 0.0, "", "0"):
                continue
            return False

    stop_loss_pct = getattr(strat, "spot_stop_loss_pct", None)
    stop_ok = stop_loss_pct is not None and float(stop_loss_pct or 0.0) > 0.0
    exec_profile = _spot_exec_profile(strat)
    entry_fill_mode = str(exec_profile.entry_fill_mode)
    flip_fill_mode = str(exec_profile.flip_fill_mode)
    exit_on_signal_flip = bool(getattr(strat, "exit_on_signal_flip", False))
    flip_fill_ok = (not bool(exit_on_signal_flip)) or spot_fill_mode_is_deferred(flip_fill_mode)
    tick_gate_mode = str(getattr(strat, "tick_gate_mode", "off") or "off").strip().lower()
    tick_gate_off = tick_gate_mode in ("off", "", "none", "false", "0")
    if not bool(tick_gate_off):
        return False
    flip_gate_mode = str(getattr(strat, "flip_exit_gate_mode", "off") or "off").strip().lower()
    flip_gate_off = flip_gate_mode in ("off", "", "none", "false", "0")
    flip_gate_ok = (not bool(exit_on_signal_flip)) or bool(flip_gate_off)
    direction_source = str(getattr(strat, "direction_source", "ema") or "ema").strip().lower()
    direction_ok = (not bool(exit_on_signal_flip)) or (direction_source == "ema")
    tick_ok = (bool(tick_gate_off) and (tick_bars is None or isinstance(tick_bars, list))) or (
        (not bool(tick_gate_off)) and isinstance(tick_bars, list) and bool(len(tick_bars))
    )
    return (
        str(getattr(strat, "entry_signal", "ema") or "ema").strip().lower() == "ema"
        and spot_fill_mode_is_deferred(entry_fill_mode)
        and bool(flip_fill_ok)
        and bool(exec_profile.intrabar_exits)
        and str(exec_profile.drawdown_mode) == "intrabar"
        and str(exec_profile.exit_mode) == "pct"
        and bool(stop_ok)
        and getattr(strat, "spot_exit_time_et", None) is None
        and not bool(exec_profile.close_eod)
        and bool(tick_ok)
        and bool(flip_gate_ok)
        and bool(direction_ok)
        and bool(signal_bars)
        and bool(exec_bars)
    )


def _run_spot_backtest_exec_loop_summary(
    cfg: ConfigBundle,
    *,
    signal_bars: BarSeriesInput,
    exec_bars: BarSeriesInput,
    meta: ContractMeta,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> SummaryStats:
    """Summary-only spot backtest (avoids materializing full equity curve)."""
    signal_bars = _bars_input_list(signal_bars)
    exec_bars = _bars_input_list(exec_bars)
    regime_bars = _bars_input_optional_list(regime_bars)
    regime2_bars = _bars_input_optional_list(regime2_bars)
    tick_bars = _bars_input_optional_list(tick_bars)

    if _can_use_fast_summary_path(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        tick_bars=tick_bars,
    ):
        _spot_emit_progress(
            progress_callback,
            phase="summary.path",
            path="fast",
            signal_total=int(len(signal_bars)),
            exec_total=int(len(exec_bars)),
        )
        return _run_spot_backtest_exec_loop_summary_fast(
            cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars,
            meta=meta,
            regime_bars=regime_bars,
            regime2_bars=regime2_bars,
            tick_bars=tick_bars,
            prepared_series_pack=prepared_series_pack,
            progress_callback=progress_callback,
        )

    _spot_emit_progress(
        progress_callback,
        phase="summary.path",
        path="exec",
        signal_total=int(len(signal_bars)),
        exec_total=int(len(exec_bars)),
    )
    return _run_spot_backtest_exec_loop(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        meta=meta,
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
        tick_bars=tick_bars,
        capture_equity=False,
        prepared_series_pack=prepared_series_pack,
        progress_callback=progress_callback,
    ).summary


def _flip_exit_base_checks(
    cfg: ConfigBundle,
    *,
    trade_dir: str | None,
    entry_time: datetime,
    bar_ts: datetime,
    signal: EmaDecisionSnapshot | None,
    tr_ratio: float | None = None,
    shock_atr_vel_pct: float | None = None,
    tr_median_pct: float | None = None,
) -> bool:
    if cfg.strategy.direction_source != "ema":
        return False
    if trade_dir is None:
        return False
    if not flip_exit_hit(
        exit_on_signal_flip=bool(cfg.strategy.exit_on_signal_flip),
        open_dir=trade_dir,
        signal=signal,
        flip_exit_mode_raw=cfg.strategy.flip_exit_mode,
        ema_entry_mode_raw=cfg.strategy.ema_entry_mode,
    ):
        return False
    hold_bars, _hold_trace = spot_dynamic_flip_hold_bars(
        strategy=cfg.strategy,
        tr_ratio=float(tr_ratio) if tr_ratio is not None else None,
        shock_atr_vel_pct=float(shock_atr_vel_pct) if shock_atr_vel_pct is not None else None,
        tr_median_pct=float(tr_median_pct) if tr_median_pct is not None else None,
    )
    if hold_bars > 0:
        held = _bars_held(cfg.backtest.bar_size, entry_time, bar_ts)
        if held < int(hold_bars):
            return False
    return True


def _spot_ratsv_probe_cancel_hit(
    cfg: ConfigBundle,
    *,
    trade: SpotTrade,
    bar: Bar,
    tr_ratio: float | None,
    slope_med: float | None,
) -> bool:
    filters = getattr(cfg.strategy, "filters", None)
    if filters is None:
        return False
    if str(getattr(trade, "entry_branch", None) or "") != "a":
        return False
    try:
        max_bars = int(getattr(filters, "ratsv_probe_cancel_max_bars", 0) or 0)
    except (TypeError, ValueError):
        max_bars = 0
    if max_bars <= 0:
        return False
    held = _bars_held(str(cfg.backtest.bar_size), trade.entry_time, bar.ts)
    if held > int(max_bars):
        return False

    try:
        slope_min = float(getattr(filters, "ratsv_probe_cancel_slope_adverse_min_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        slope_min = 0.0
    if slope_min <= 0:
        return False

    if slope_med is None:
        return False
    adverse = float(slope_med) <= -float(slope_min) if int(trade.qty) > 0 else float(slope_med) >= float(slope_min)
    if not bool(adverse):
        return False

    tr_min_raw = getattr(filters, "ratsv_probe_cancel_tr_ratio_min", None)
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
    cfg: ConfigBundle,
    *,
    trade: SpotTrade,
    bar: Bar,
    tr_ratio: float | None,
    slope_med: float | None,
    slope_vel: float | None,
) -> bool:
    filters = getattr(cfg.strategy, "filters", None)
    if filters is None:
        return False

    try:
        slope_min = float(getattr(filters, "ratsv_adverse_release_slope_adverse_min_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        slope_min = 0.0
    if slope_min <= 0:
        return False

    try:
        min_hold = int(getattr(filters, "ratsv_adverse_release_min_hold_bars", 0) or 0)
    except (TypeError, ValueError):
        min_hold = 0
    if min_hold > 0:
        held = _bars_held(str(cfg.backtest.bar_size), trade.entry_time, bar.ts)
        if held < int(min_hold):
            return False

    adverse = False
    if slope_med is not None:
        adverse = float(slope_med) <= -float(slope_min) if int(trade.qty) > 0 else float(slope_med) >= float(slope_min)
    if (not adverse) and slope_vel is not None:
        adverse = float(slope_vel) <= -float(slope_min) if int(trade.qty) > 0 else float(slope_vel) >= float(slope_min)
    if not bool(adverse):
        return False

    tr_min_raw = getattr(filters, "ratsv_adverse_release_tr_ratio_min", None)
    if tr_min_raw is not None:
        try:
            tr_min = float(tr_min_raw)
        except (TypeError, ValueError):
            tr_min = None
        if tr_min is not None and tr_min > 0:
            if tr_ratio is None or float(tr_ratio) < float(tr_min):
                return False
    return True


def _spot_hit_flip_exit(
    cfg: ConfigBundle,
    trade: SpotTrade,
    bar: Bar,
    signal: EmaDecisionSnapshot | None,
    tr_ratio: float | None = None,
    shock_atr_vel_pct: float | None = None,
    tr_median_pct: float | None = None,
) -> bool:
    trade_dir = "up" if trade.qty > 0 else "down" if trade.qty < 0 else None
    if not _flip_exit_base_checks(
        cfg,
        trade_dir=trade_dir,
        entry_time=trade.entry_time,
        bar_ts=bar.ts,
        signal=signal,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    ):
        return False

    if cfg.strategy.flip_exit_only_if_profit:
        pnl = (price := bar.close) - trade.entry_price
        if trade.qty < 0:
            pnl = -pnl
        if pnl <= 0:
            return False

    if lifecycle_flip_exit_gate_blocked(
        gate_mode_raw=getattr(cfg.strategy, "flip_exit_gate_mode", "off"),
        filters=cfg.strategy.filters,
        close=float(bar.close),
        signal=signal,
        trade_dir=trade_dir,
    ):
        return False
    return True


def _close_spot_trade(trade: SpotTrade, ts: datetime, price: float, reason: str, trades: list[SpotTrade]) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


# endregion


# region Synthetic Options Helpers
def _trade_value(
    trade: OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
    calibration,
) -> float:
    return _trade_value_from_spec(
        trade,
        bar,
        rv,
        cfg,
        surface_params,
        min_tick,
        is_future,
        mode="mark",
        calibration=calibration,
    )


def _rv_from_bars(bars: list[Bar], cfg: ConfigBundle) -> float:
    closes = [float(bar.close) for bar in bars if bar.close and float(bar.close) > 0]
    rv = realized_vol_from_closes(
        closes,
        lookback=int(cfg.synthetic.rv_lookback),
        lam=float(cfg.synthetic.rv_ewma_lambda),
        bar_size=str(cfg.backtest.bar_size),
        use_rth=bool(cfg.backtest.use_rth),
    )
    return float(cfg.synthetic.iv_floor) if rv is None else float(rv)


def _ema_periods(preset: str | None) -> tuple[int, int] | None:
    return _ema_periods_shared(preset)


def _ema_bias(cfg: ConfigBundle) -> str:
    if cfg.strategy.ema_directional:
        return "any"
    legs = cfg.strategy.legs
    if legs:
        first = legs[0]
        action = first.action.upper()
        right = first.right.upper()
        if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
            return "up"
        if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
            return "down"
        return "any"
    right = cfg.strategy.right.upper()
    if right == "PUT":
        return "up"
    if right == "CALL":
        return "down"
    return "any"


def _direction_from_legs(legs: list[OptionLeg]) -> str | None:
    if not legs:
        return None
    first = legs[0]
    action = first.action.upper()
    right = first.right.upper()
    if (action, right) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "up"
    if (action, right) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "down"
    return None


def _trade_value_from_spec(
    spec: TradeSpec | OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    min_tick: float,
    is_future: bool,
    mode: str,
    calibration,
) -> float:
    trade_day = _trade_date(bar.ts)
    dte_days = max((spec.expiry - trade_day).days, 0)
    if calibration:
        surface_params = calibration.surface_params_asof(
            dte_days,
            trade_day.isoformat(),
            surface_params,
        )
    atm_iv = iv_atm(rv, dte_days, surface_params)
    forward = bar.close
    if dte_days == 0:
        t = max(_session_hours(cfg.backtest.use_rth) / (24.0 * 365.0), _min_time(cfg.backtest.bar_size))
    else:
        t = max(dte_days / 365.0, _min_time(cfg.backtest.bar_size))
    legs = spec.legs
    if len(legs) <= 1:
        net = 0.0
        for leg in legs:
            leg_iv = iv_for_strike(atm_iv, forward, leg.strike, surface_params)
            if is_future:
                mid = black_76(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
            else:
                mid = black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
            quote = mid_edge_quote(mid, cfg.synthetic.min_spread_pct, min_tick)
            if mode == "entry":
                price = quote.bid if leg.action == "SELL" else quote.ask
            elif mode == "exit":
                price = quote.ask if leg.action == "SELL" else quote.bid
            else:
                price = quote.mid
            sign = 1 if leg.action == "SELL" else -1
            net += sign * price * leg.qty
        return net

    # Multi-leg combos: apply a single bid/ask edge to the net mid instead of legging each spread.
    net_mid = 0.0
    for leg in legs:
        leg_iv = iv_for_strike(atm_iv, forward, leg.strike, surface_params)
        if is_future:
            mid = black_76(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        else:
            mid = black_scholes(forward, leg.strike, t, cfg.backtest.risk_free_rate, leg_iv, leg.right)
        sign = 1 if leg.action == "SELL" else -1
        net_mid += sign * mid * leg.qty

    abs_mid = abs(net_mid)
    quote = mid_edge_quote(abs_mid, cfg.synthetic.min_spread_pct, min_tick)
    mid_signed = quote.mid if net_mid >= 0 else -quote.mid
    bid_signed = quote.bid if net_mid >= 0 else -quote.bid
    ask_signed = quote.ask if net_mid >= 0 else -quote.ask

    if mode == "mark":
        return mid_signed
    if mode == "entry":
        return bid_signed if net_mid >= 0 else ask_signed
    # mode == "exit"
    return ask_signed if net_mid >= 0 else bid_signed


def _hit_profit(trade: OptionTrade, current_value: float) -> bool:
    target = abs(trade.entry_price) * trade.profit_target
    return (trade.entry_price - current_value) >= target


def _hit_stop(trade: OptionTrade, current_value: float, basis: str, spot: float) -> bool:
    loss = max(0.0, current_value - trade.entry_price)
    if basis == "credit":
        if trade.entry_price >= 0:
            return current_value >= trade.entry_price * (1 + trade.stop_loss)
        return loss >= abs(trade.entry_price) * trade.stop_loss
    max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
    if max_loss is None:
        max_loss = _max_loss_estimate(trade, spot)
    if max_loss is None:
        max_loss = abs(trade.entry_price)
    return loss >= max_loss * trade.stop_loss


def _hit_exit_dte(cfg: ConfigBundle, trade: OptionTrade, today: date) -> bool:
    if cfg.strategy.exit_dte <= 0:
        return False
    entry_dte = business_days_until(_trade_date(trade.entry_time), trade.expiry)
    if cfg.strategy.exit_dte >= entry_dte:
        return False
    remaining = business_days_until(today, trade.expiry)
    return remaining <= cfg.strategy.exit_dte


def _hit_flip_exit(
    cfg: ConfigBundle,
    trade: OptionTrade,
    bar: Bar,
    current_value: float,
    signal: EmaDecisionSnapshot | None,
) -> bool:
    trade_dir = _direction_from_legs(trade.legs)
    if not _flip_exit_base_checks(
        cfg,
        trade_dir=trade_dir,
        entry_time=trade.entry_time,
        bar_ts=bar.ts,
        signal=signal,
        tr_ratio=None,
        shock_atr_vel_pct=None,
    ):
        return False

    if cfg.strategy.flip_exit_only_if_profit:
        if (trade.entry_price - current_value) <= 0:
            return False
    return True


def _bars_held(bar_size: str, start: datetime, end: datetime) -> int:
    hours = _bar_hours(bar_size)
    if hours <= 0:
        return 0
    return int((end - start).total_seconds() / 3600.0 / hours)


def _bar_hours(bar_size: str) -> float:
    label = bar_size.lower().strip()
    if "hour" in label:
        try:
            prefix = label.split("hour")[0].strip()
            return float(prefix) if prefix else 1.0
        except ValueError:
            return 1.0
    if "min" in label:
        try:
            prefix = label.split("min")[0].strip()
            mins = float(prefix) if prefix else 30.0
            return mins / 60.0
        except ValueError:
            return 0.5
    if "day" in label:
        try:
            prefix = label.split("day")[0].strip()
            days = float(prefix) if prefix else 1.0
            return days * 24.0
        except ValueError:
            return 24.0
    return 1.0


def _close_trade(trade: OptionTrade, ts: datetime, price: float, reason: str, trades: list[OptionTrade]) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


def _summarize_from_trades_and_max_dd(
    trades: list[OptionTrade | SpotTrade],
    *,
    starting_cash: float,
    max_dd: float,
    multiplier: float,
) -> SummaryStats:
    wins = 0
    losses = 0
    total_pnl = 0.0
    win_pnls: list[float] = []
    loss_pnls: list[float] = []
    hold_hours: list[float] = []

    for trade in trades:
        pnl = trade.pnl(multiplier)
        total_pnl += pnl
        if pnl >= 0:
            wins += 1
            win_pnls.append(pnl)
        else:
            losses += 1
            loss_pnls.append(pnl)
        if trade.exit_time:
            hold_hours.append((trade.exit_time - trade.entry_time).total_seconds() / 3600.0)

    total = wins + losses
    win_rate = wins / total if total else 0.0
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0
    roi = (total_pnl / starting_cash) if starting_cash > 0 else 0.0
    max_dd_pct = (max_dd / starting_cash) if starting_cash > 0 else 0.0
    return SummaryStats(
        trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        roi=roi,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_hold_hours=avg_hold,
    )


def _summarize(
    trades: list[OptionTrade | SpotTrade],
    starting_cash: float,
    equity_curve: list[EquityPoint],
    multiplier: float,
) -> SummaryStats:
    peak = starting_cash
    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0].equity
    for point in equity_curve:
        if point.equity > peak:
            peak = point.equity
        dd = peak - point.equity
        if dd > max_dd:
            max_dd = dd
    return _summarize_from_trades_and_max_dd(
        trades,
        starting_cash=starting_cash,
        max_dd=float(max_dd),
        multiplier=multiplier,
    )


def _max_loss(trade: OptionTrade) -> float | None:
    legs = trade.legs
    if len(legs) != 2:
        return None
    a, b = legs
    if a.right != b.right:
        return None
    if a.qty != b.qty:
        return None
    if {a.action, b.action} != {"BUY", "SELL"}:
        return None
    width = abs(a.strike - b.strike)
    if trade.entry_price >= 0:
        return max(0.0, width - trade.entry_price)
    return abs(trade.entry_price)


def _session_hours(use_rth: bool) -> float:
    return 6.5 if use_rth else 24.0


def _min_time(bar_size: str) -> float:
    return float(_bar_hours(bar_size)) / (24.0 * 365.0)


def _margin_required(trade: OptionTrade, spot: float, multiplier: float) -> float:
    if trade.entry_price <= 0:
        return 0.0
    max_loss = trade.max_loss if trade.max_loss is not None else _max_loss(trade)
    if max_loss is None:
        max_loss = _max_loss_estimate(trade, spot)
    if max_loss is None:
        return 0.0
    return max(0.0, max_loss) * multiplier


def _max_loss_estimate(trade: OptionTrade, spot: float) -> float | None:
    strikes = sorted({leg.strike for leg in trade.legs})
    if not strikes:
        return None
    high = max(spot, strikes[-1]) * 5.0
    candidates = [0.0] + strikes + [high]
    min_pnl = None
    for price in candidates:
        pnl = trade.entry_price + _payoff_at_expiry(trade.legs, price)
        if min_pnl is None or pnl < min_pnl:
            min_pnl = pnl
    if min_pnl is None:
        return None
    return max(0.0, -min_pnl)


def _payoff_at_expiry(legs: list[OptionLeg], spot: float) -> float:
    payoff = 0.0
    for leg in legs:
        right = leg.right.upper()
        if right == "CALL":
            intrinsic = max(spot - leg.strike, 0.0)
        else:
            intrinsic = max(leg.strike - spot, 0.0)
        sign = 1.0 if leg.action.upper() == "BUY" else -1.0
        payoff += sign * intrinsic * leg.qty
    return payoff


def _equity_after_entry(
    cash_after: float,
    open_trades: list[OptionTrade],
    candidate: OptionTrade,
    bar: Bar,
    rv: float,
    cfg: ConfigBundle,
    surface_params: IVSurfaceParams,
    meta: ContractMeta,
    is_future: bool,
    calibration,
) -> float:
    liquidation = 0.0
    for trade in open_trades:
        mark_value = _trade_value(trade, bar, rv, cfg, surface_params, meta.min_tick, is_future, calibration)
        liquidation += (-mark_value) * meta.multiplier
    candidate_mark = _trade_value(candidate, bar, rv, cfg, surface_params, meta.min_tick, is_future, calibration)
    liquidation += (-candidate_mark) * meta.multiplier
    return cash_after + liquidation


# endregion
