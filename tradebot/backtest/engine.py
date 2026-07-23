"""Canonical backtest orchestration and spot lifecycle runtime."""

from __future__ import annotations

import math
import hashlib
from bisect import bisect_left
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import NamedTuple, Union

from .config import ConfigBundle, SpotLegConfig
from .data import IBKRHistoricalData, ContractMeta, load_backtest_series
from .models import (
    BacktestResult,
    Bar,
    EquityPoint,
    SpotTrade,
    SummaryStats,
    summarize_with_max_drawdown,
)
from .spot_context import (
    SpotBarRequirement,
    SpotContextBars,
    load_spot_context_bars,
    spot_signal_warmup_days_from_strategy,
)
from ..chart_data.cache import series_cache_service
from ..chart_data.series import BarSeries, BarSeriesSignature, bars_list
from ..engines.risk import risk_overlay_policy_from_filters
from ..engines.signals import EmaDecisionSnapshot
from ..option_package import option_product_facts
from ..spot.gates import (
    deferred_entry_plan as lifecycle_deferred_entry_plan,
    fill_due_ts as lifecycle_fill_due_ts,
    entry_capacity_ok as lifecycle_entry_capacity_ok,
    flip_exit_allowed,
    flip_exit_gate_blocked as lifecycle_flip_exit_gate_blocked,
    signal_filters_ok as lifecycle_signal_filters_ok,
)
from ..spot.lifecycle import (
    decide_flat_position_intent,
    decide_open_position_intent,
    decide_pending_next_open,
)
from ..spot.fill_modes import (
    SPOT_FILL_MODE_CLOSE,
    normalize_spot_fill_mode,
    spot_fill_mode_is_deferred,
)
from ..spot.evaluator_common import SpotRegimeState, SpotSignalSnapshot
from ..spot.graph import SpotPolicyGraph
from ..spot.policy_contract import SpotPolicyConfigView
from ..spot.scenario import (
    lifecycle_trace_row,
    why_not_exit_resize_report,
    write_rows_csv,
)
from ..engine import (
    _trade_date,
    _trade_weekday,
    _ts_to_et,
    bars_elapsed,
    cooldown_ok_by_index,
    normalize_spot_entry_signal,
    parse_time_hhmm,
    resolve_spot_regime2_spec,
    resolve_spot_regime_spec,
    spot_apply_branch_size_mult,
    spot_calc_signed_qty_with_trace,
    spot_hit_profit,
    spot_hit_stop,
    spot_exec_price as _spot_exec_price,
    spot_intrabar_exit,
    spot_intrabar_worst_ref,
    spot_mark_price as _spot_mark_price,
    spot_runtime_spec_view,
    spot_profit_level,
    spot_resolve_entry_action_qty,
    spot_riskoff_end_hour,
    spot_scale_exit_pcts,
    spot_shock_exit_pct_multipliers,
    spot_stop_level,
)
from ..signals import ema_periods as _ema_periods_shared, parse_bar_size


# region Internal Caches (Spot Backtest)
_SERIES_CACHE = series_cache_service()
_SPOT_SERIES_PACK_NAMESPACE = "spot.series.pack"
_SPOT_EXEC_ALIGNMENT_NAMESPACE = "spot.exec.alignment"
_SPOT_TICK_GATE_SERIES_NAMESPACE = "spot.tick_gate.series"
_SPOT_SIGNAL_TRACE_KEYS = (
    "shock_atr_pct",
    "shock_atr_vel_pct",
    "shock_atr_accel_pct",
    "ratsv_tr_ratio",
    "risk_tr_median_pct",
    "ratsv_fast_slope_med_pct",
    "ratsv_fast_slope_vel_pct",
    "ratsv_slow_slope_med_pct",
    "ratsv_slow_slope_vel_pct",
    "ratsv_slope_vel_consistency",
)


class _SpotExecAlignment(NamedTuple):
    sig_idx_by_exec_idx: list[int]  # -1 when exec bar isn't a signal close
    exec_idx_by_sig_idx: list[int]  # -1 when signal ts isn't present in exec bars
    signal_exec_indices: list[
        int
    ]  # exec indices in ascending order that are signal closes
    signal_sig_indices: list[int]  # matching signal indices for `signal_exec_indices`


def _spot_exec_alignment(
    signal_bars: list[Bar], exec_bars: list[Bar]
) -> _SpotExecAlignment:
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


_SPOT_SERIES_PACK_CACHE_VERSION = "spot-series-pack-v4"


BarSeriesInput = Union[list[Bar], BarSeries[Bar]]


def _bars_input_list(value: BarSeriesInput) -> list[Bar]:
    return bars_list(value)


def _bars_input_optional_list(value: BarSeriesInput | None) -> list[Bar] | None:
    if value is None:
        return None
    return bars_list(value)


class _SpotTickGateSeries(NamedTuple):
    tick_ready_by_sig_idx: list[bool]
    tick_dir_by_sig_idx: list[str | None]


class _SpotSeriesPack(NamedTuple):
    """Reusable, semantics-preserving facts for one execution tape."""

    align: _SpotExecAlignment
    tick_series: _SpotTickGateSeries | None
    exec_dates: list[date]


class _SpotRunBars(NamedTuple):
    """Canonical signal, execution, and context tapes for one spot run."""

    signal: list[Bar]
    execution: list[Bar]
    regime: list[Bar] | None
    regime2: list[Bar] | None
    regime2_bear_hard: list[Bar] | None
    tick: list[Bar] | None


class _SpotPolicyRun(NamedTuple):
    """Immutable policy objects shared by every decision in one backtest."""

    lifecycle_graph: SpotPolicyGraph
    lifecycle_config: SpotPolicyConfigView
    sizing_graph: SpotPolicyGraph
    sizing_config: SpotPolicyConfigView


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


@dataclass(frozen=True)
class _SpotPendingEntry:
    direction: str | None = None
    branch: str | None = None
    set_date: date | None = None
    due_ts: datetime | None = None
    regime: SpotRegimeState = SpotRegimeState()
    guard_probe: dict[str, object] | None = None
    guard_inputs: dict[str, object] | None = None

    @property
    def active(self) -> bool:
        return self.direction in ("up", "down")

    @classmethod
    def from_signal(
        cls,
        *,
        direction: str,
        branch: str | None,
        set_date: date,
        due_ts: datetime,
        snapshot: object | None,
        guard_probe: dict[str, object] | None = None,
        guard_inputs: dict[str, object] | None = None,
    ) -> "_SpotPendingEntry":
        return cls(
            direction=str(direction),
            branch=str(branch) if branch in ("a", "b") else None,
            set_date=set_date,
            due_ts=due_ts,
            regime=SpotRegimeState.from_snapshot(snapshot),
            guard_probe=dict(guard_probe) if isinstance(guard_probe, dict) else None,
            guard_inputs=dict(guard_inputs) if isinstance(guard_inputs, dict) else None,
        )


@dataclass(frozen=True)
class _SpotEntryEvidence:
    snapshot: object | None
    shock: bool | None
    shock_dir: str | None
    shock_atr_pct: float | None
    riskoff: bool
    riskpanic: bool
    riskpop: bool
    risk_snapshot: object | None
    regime: SpotRegimeState
    guard_probe: dict[str, object] | None = None
    guard_inputs: dict[str, object] | None = None
    local_extrema: dict[str, object] | None = None

    @classmethod
    def from_signal(
        cls,
        *,
        snapshot: object | None,
        shock: bool | None,
        shock_dir: str | None,
        shock_atr_pct: float | None,
        riskoff: bool,
        riskpanic: bool,
        riskpop: bool,
        risk_snapshot: object | None,
        pending: _SpotPendingEntry | None = None,
        guard_probe: dict[str, object] | None = None,
        guard_inputs: dict[str, object] | None = None,
        local_extrema: dict[str, object] | None = None,
    ) -> "_SpotEntryEvidence":
        source = pending if pending is not None else snapshot
        return cls(
            snapshot=snapshot,
            shock=shock,
            shock_dir=str(shock_dir) if shock_dir in ("up", "down") else None,
            shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
            riskoff=bool(riskoff),
            riskpanic=bool(riskpanic),
            riskpop=bool(riskpop),
            risk_snapshot=risk_snapshot,
            regime=SpotRegimeState.from_snapshot(source),
            guard_probe=dict(guard_probe) if isinstance(guard_probe, dict) else None,
            guard_inputs=dict(guard_inputs) if isinstance(guard_inputs, dict) else None,
            local_extrema=dict(local_extrema)
            if isinstance(local_extrema, dict)
            else None,
        )


def _spot_bars_signature(bars: list[Bar] | None) -> BarSeriesSignature:
    return _SERIES_CACHE.revision(bars or ())


def _spot_tick_gate_settings(
    strategy: object,
) -> tuple[str, str, str, int, int, float, float, int]:
    tick_mode = str(getattr(strategy, "tick_gate_mode", "off") or "off").strip().lower()
    if tick_mode not in ("off", "raschke"):
        tick_mode = "off"
    tick_neutral_policy = (
        str(getattr(strategy, "tick_neutral_policy", "allow") or "allow")
        .strip()
        .lower()
    )
    if tick_neutral_policy not in ("allow", "block"):
        tick_neutral_policy = "allow"
    tick_direction_policy = (
        str(getattr(strategy, "tick_direction_policy", "both") or "both")
        .strip()
        .lower()
    )
    if tick_direction_policy not in ("both", "wide_only"):
        tick_direction_policy = "both"
    tick_ma_period = max(1, int(getattr(strategy, "tick_band_ma_period", 10) or 10))
    tick_z_lookback = max(
        5, int(getattr(strategy, "tick_width_z_lookback", 252) or 252)
    )
    tick_z_enter = float(getattr(strategy, "tick_width_z_enter", 1.0) or 1.0)
    tick_z_exit = max(0.0, float(getattr(strategy, "tick_width_z_exit", 0.5) or 0.5))
    tick_slope_lookback = max(
        1, int(getattr(strategy, "tick_width_slope_lookback", 3) or 3)
    )
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
    if not tick_ready:
        return None if tick_neutral_policy == "block" else entry_dir
    if tick_dir not in ("up", "down"):
        return None if tick_neutral_policy == "block" else entry_dir
    return entry_dir if entry_dir is None or entry_dir == tick_dir else None


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
                if len(tick_widths) >= max(5, min_z) and len(tick_width_hist) >= (
                    tick_slope_lookback + 1
                ):
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
        tick_dir_by_sig_idx[sig_i] = (
            str(tick_dir) if tick_dir in ("up", "down") else None
        )

    out = _SpotTickGateSeries(
        tick_ready_by_sig_idx=tick_ready_by_sig_idx,
        tick_dir_by_sig_idx=tick_dir_by_sig_idx,
    )
    _SERIES_CACHE.set(namespace=_SPOT_TICK_GATE_SERIES_NAMESPACE, key=key, value=out)
    return out


def _spot_series_cache_db_path(cache_dir: object | None) -> Path | None:
    if cache_dir is None:
        return None
    try:
        root = Path(str(cache_dir)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root / "spot_series_cache.sqlite3"
    except Exception:
        return None


def _spot_series_pack_persistent_get(
    *, cache_dir: object | None, key_hash: str
) -> _SpotSeriesPack | None:
    loaded = _SERIES_CACHE.get_persistent(
        db_path=_spot_series_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_NAMESPACE,
        key_hash=str(key_hash),
        validator=lambda obj: isinstance(obj, _SpotSeriesPack),
    )
    return loaded if isinstance(loaded, _SpotSeriesPack) else None


def _spot_series_pack_persistent_set(
    *, cache_dir: object | None, key_hash: str, payload: _SpotSeriesPack
) -> None:
    _SERIES_CACHE.set_persistent(
        db_path=_spot_series_cache_db_path(cache_dir),
        namespace=_SPOT_SERIES_PACK_NAMESPACE,
        key_hash=str(key_hash),
        value=payload,
    )


def _spot_series_pack_key(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    tick_bars: list[Bar] | None,
    include_tick: bool,
) -> tuple[tuple[object, ...], str]:
    key = (
        str(_SPOT_SERIES_PACK_CACHE_VERSION),
        _spot_bars_signature(signal_bars),
        _spot_bars_signature(exec_bars),
        _spot_bars_signature(tick_bars if include_tick else None),
        _spot_tick_gate_settings(cfg.strategy) if include_tick else ("off",),
    )
    return key, hashlib.sha1(repr(key).encode("utf-8")).hexdigest()


def _spot_prepare_summary_series_pack(
    *,
    cfg: ConfigBundle,
    signal_bars: BarSeriesInput,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
) -> tuple[str, object | None]:
    """Warm the canonical execution pack shared by an entire sweep tape."""
    signal_list = _bars_input_list(signal_bars)
    tick_list = _bars_input_optional_list(tick_bars)
    exec_list = _bars_input_optional_list(exec_bars)
    exec_bar_size = str(getattr(cfg.strategy, "spot_exec_bar_size", "") or "").strip()
    if exec_bar_size and exec_bar_size != str(cfg.backtest.bar_size):
        if not exec_list:
            return "", None
    else:
        exec_list = signal_list

    include_tick = (
        str(getattr(cfg.strategy, "tick_gate_mode", "off") or "off").strip().lower()
        != "off"
    )
    pack_key, pack_key_hash = _spot_series_pack_key(
        cfg=cfg,
        signal_bars=signal_list,
        exec_bars=exec_list,
        tick_bars=tick_list,
        include_tick=include_tick,
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key)
    if isinstance(cached, _SpotSeriesPack):
        return pack_key_hash, cached
    return pack_key_hash, _spot_build_series_pack(
        cfg=cfg,
        signal_bars=signal_list,
        exec_bars=exec_list,
        tick_bars=tick_list,
        include_tick=include_tick,
    )


def _spot_build_series_pack(
    *,
    cfg: ConfigBundle,
    signal_bars: list[Bar],
    exec_bars: list[Bar],
    tick_bars: list[Bar] | None,
    include_tick: bool,
) -> _SpotSeriesPack:
    pack_key, pack_key_hash = _spot_series_pack_key(
        cfg=cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        tick_bars=tick_bars,
        include_tick=include_tick,
    )
    cached = _SERIES_CACHE.get(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key)
    if isinstance(cached, _SpotSeriesPack):
        return cached

    use_persist = bool(getattr(cfg.backtest, "offline", False))
    loaded = (
        _spot_series_pack_persistent_get(
            cache_dir=cfg.backtest.cache_dir, key_hash=pack_key_hash
        )
        if use_persist
        else None
    )
    if loaded is not None:
        _SERIES_CACHE.set(
            namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key, value=loaded
        )
        return loaded

    pack = _SpotSeriesPack(
        align=_spot_exec_alignment(signal_bars, exec_bars),
        tick_series=(
            _spot_tick_gate_series(
                signal_bars=signal_bars, tick_bars=tick_bars, strategy=cfg.strategy
            )
            if include_tick
            else None
        ),
        exec_dates=[_trade_date(bar.ts) for bar in exec_bars],
    )
    _SERIES_CACHE.set(namespace=_SPOT_SERIES_PACK_NAMESPACE, key=pack_key, value=pack)
    if use_persist:
        _spot_series_pack_persistent_set(
            cache_dir=cfg.backtest.cache_dir,
            key_hash=pack_key_hash,
            payload=pack,
        )
    return pack


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


# endregion


# region Public API
def _resolve_backtest_contract_meta(
    *, data: IBKRHistoricalData, cfg: ConfigBundle
) -> ContractMeta:
    is_future = cfg.strategy.symbol in ("MNQ", "MBT")
    option_product = (
        option_product_facts(cfg.strategy.symbol)
        if cfg.strategy.instrument == "options"
        else None
    )
    if cfg.backtest.offline:
        if cfg.strategy.instrument == "spot":
            exchange = "CME" if is_future else "SMART"
            multiplier = _spot_multiplier(cfg.strategy.symbol, is_future)
        else:
            assert option_product is not None
            exchange = option_product.exchange
            multiplier = option_product.multiplier
        return ContractMeta(
            symbol=cfg.strategy.symbol,
            exchange=exchange,
            multiplier=multiplier,
            min_tick=0.01,
        )

    _, resolved = data.resolve_contract(cfg.strategy.symbol, cfg.strategy.exchange)
    if cfg.strategy.instrument == "spot":
        return ContractMeta(
            symbol=resolved.symbol,
            exchange=resolved.exchange,
            multiplier=_spot_multiplier(
                cfg.strategy.symbol, is_future, default=resolved.multiplier
            ),
            min_tick=resolved.min_tick,
        )
    assert option_product is not None
    return ContractMeta(
        symbol=resolved.symbol,
        exchange=resolved.exchange or option_product.exchange,
        multiplier=(
            option_product.multiplier
            if resolved.exchange == "SMART"
            else resolved.multiplier or option_product.multiplier
        ),
        min_tick=resolved.min_tick,
    )


def _load_spot_backtest_context_bars(
    *,
    data: IBKRHistoricalData,
    cfg: ConfigBundle,
    signal_bars: BarSeries[Bar],
    start_dt: datetime,
    end_dt: datetime,
) -> SpotContextBars:
    def _load_requirement(
        req: SpotBarRequirement, req_start: datetime, req_end: datetime
    ):
        if not bool(cfg.backtest.offline) or req_start == start_dt:
            return load_backtest_series(
                data=data,
                cfg=cfg,
                symbol=req.symbol,
                exchange=req.exchange,
                start=req_start,
                end=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
            )
        try:
            return load_backtest_series(
                data=data,
                cfg=cfg,
                symbol=req.symbol,
                exchange=req.exchange,
                start=req_start,
                end=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
            )
        except FileNotFoundError:
            # Router/indicator warmup can request earlier context windows than the cache contains.
            # Falling back to the scoring start preserves run continuity while still warming up
            # any slow indicators as bars arrive.
            return load_backtest_series(
                data=data,
                cfg=cfg,
                symbol=req.symbol,
                exchange=req.exchange,
                start=start_dt,
                end=req_end,
                bar_size=str(req.bar_size),
                use_rth=bool(req.use_rth),
            )

    context = load_spot_context_bars(
        strategy=cfg.strategy,
        signal_bars=signal_bars,
        default_symbol=str(cfg.strategy.symbol),
        default_exchange=cfg.strategy.exchange,
        default_signal_bar_size=str(cfg.backtest.bar_size),
        default_signal_use_rth=bool(cfg.backtest.use_rth),
        start_dt=start_dt,
        end_dt=end_dt,
        load_requirement=_load_requirement,
    )
    return context


def run_backtest(cfg: ConfigBundle) -> BacktestResult:
    data = IBKRHistoricalData()
    start_dt = datetime.combine(cfg.backtest.start, time(0, 0))
    end_dt = datetime.combine(cfg.backtest.end, time(23, 59))
    signal_start_dt = start_dt - timedelta(
        days=max(
            0,
            int(
                spot_signal_warmup_days_from_strategy(
                    strategy=cfg.strategy,
                    default_signal_bar_size=str(cfg.backtest.bar_size),
                    default_signal_use_rth=bool(cfg.backtest.use_rth),
                )
            ),
        )
    )
    try:
        bar_series = load_backtest_series(
            data=data,
            cfg=cfg,
            symbol=cfg.strategy.symbol,
            exchange=cfg.strategy.exchange,
            start=signal_start_dt,
            end=end_dt,
            bar_size=cfg.backtest.bar_size,
            use_rth=cfg.backtest.use_rth,
        )
    except FileNotFoundError:
        # Warmup can request bars earlier than the cached dataset starts (e.g. the first year in the cache).
        # Fall back to the scoring start; downstream indicators will still warm up as bars arrive.
        bar_series = load_backtest_series(
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
        context = _load_spot_backtest_context_bars(
            data=data,
            cfg=cfg,
            signal_bars=bar_series,
            start_dt=start_dt,
            end_dt=end_dt,
        )

        result = _run_spot_backtest(
            cfg,
            context.signal_bars,
            meta,
            regime_bars=context.regime_bars,
            regime2_bars=context.regime2_bars,
            regime2_bear_hard_bars=context.regime2_bear_hard_bars,
            tick_bars=context.tick_bars,
            exec_bars=context.exec_bars,
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
            * _spot_mark_price(
                float(ref_price), qty=trade.qty, spread=spread, mode=mark_to_market
            )
            * float(multiplier)
        )
    return float(total)


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
    apply_slippage_eff = (
        (str(reason) != "profit") if apply_slippage is None else bool(apply_slippage)
    )
    exit_price = _spot_exec_price(
        float(exit_ref_price),
        side="sell" if int(qty) > 0 else "buy",
        qty=int(qty),
        spread=float(spread),
        commission_per_share=float(commission_per_share),
        commission_min=float(commission_min),
        slippage_per_share=float(slippage_per_share),
        apply_slippage=bool(apply_slippage_eff),
    )
    if trade is not None and trades is not None:
        _close_spot_trade(trade, exit_time, float(exit_price), str(reason), trades)
    next_cash = float(cash) + int(qty) * float(exit_price) * float(multiplier)
    next_margin = max(0.0, float(margin_used) - float(margin_required))
    return float(exit_price), float(next_cash), float(next_margin)


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
    cash_after = float(cash) - (int(signed_qty) * float(entry_price)) * float(
        multiplier
    )
    margin_after = float(margin_used) + float(margin_required)
    candidate_mark = (
        int(signed_qty)
        * _spot_mark_price(
            float(mark_ref_price),
            qty=int(signed_qty),
            spread=float(spread),
            mode=str(mark_to_market),
        )
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


def _spot_branch_size_mult(
    *, policy: _SpotPolicyRun, entry_branch: str | None
) -> float:
    cfg = policy.lifecycle_config
    if not cfg.spot_dual_branch_enabled:
        return 1.0
    if entry_branch == "a":
        return float(cfg.spot_branch_a_size_mult)
    if entry_branch == "b":
        return float(cfg.spot_branch_b_size_mult)
    return 1.0


def _spot_strategy_sec_type(*, strategy) -> str:
    raw = str(getattr(strategy, "spot_sec_type", "") or "").strip().upper()
    if raw:
        return raw
    symbol = str(getattr(strategy, "symbol", "") or "").strip().upper()
    if symbol in {"MNQ", "MES", "ES", "NQ", "YM", "RTY", "M2K"}:
        return "FUT"
    return "STK"


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
            bool(signal_use_rth) if signal_use_rth is not None else bool(use_rth_raw)
        ),
        "spot_sec_type": str(
            spot_sec_type or sec_type_raw or _spot_strategy_sec_type(strategy=strategy)
        ),
    }
    return lifecycle_fill_due_ts(
        fill_mode=normalize_spot_fill_mode(fill_mode, default=SPOT_FILL_MODE_CLOSE),
        signal_close_ts=signal_close_ts,
        exec_bar_size=str(exec_bar_size or ""),
        strategy=strategy_view,
        naive_ts_mode="utc",
    )


def _spot_flat_entry_decision_from_signal(
    *,
    policy: _SpotPolicyRun,
    strategy,
    filters,
    signal_bar: Bar,
    signal: EmaDecisionSnapshot | None,
    direction: str | None,
    entry_context: dict[str, object] | None,
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
    lifecycle_inputs: Mapping[str, object],
    entry_gate_bypass: bool = False,
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
        entry_gate_bypass=bool(entry_gate_bypass),
    )
    entry_capacity = lifecycle_entry_capacity_ok(
        open_count=int(open_count),
        max_entries_per_day=int(getattr(strategy, "max_entries_per_day", 0) or 0),
        entries_today=int(entries_today),
        weekday=_trade_weekday(signal_bar.ts),
        entry_days=getattr(strategy, "entry_days", ()),
    )
    return decide_flat_position_intent(
        strategy=strategy,
        bar_ts=signal_bar.ts,
        entry_dir=direction,
        entry_context=entry_context,
        allowed_directions=("up", "down"),
        filters_ok=bool(filters_ok),
        entry_capacity=bool(entry_capacity),
        pending_exists=bool(pending_exists),
        next_open_allowed=bool(next_open_allowed),
        can_order_now=True,
        preflight_ok=True,
        stale_signal=False,
        gap_signal=False,
        atr_ready=bool(
            str(exit_mode) != "atr"
            or (atr_value is not None and float(atr_value) > 0.0)
        ),
        shock_atr_pct=lifecycle_inputs.get("shock_atr_pct"),
        shock_atr_vel_pct=lifecycle_inputs.get("shock_atr_vel_pct"),
        shock_atr_accel_pct=lifecycle_inputs.get("shock_atr_accel_pct"),
        tr_ratio=lifecycle_inputs.get("tr_ratio"),
        tr_median_pct=lifecycle_inputs.get("tr_median_pct"),
        slope_med_pct=lifecycle_inputs.get("slope_med_pct"),
        slope_vel_pct=lifecycle_inputs.get("slope_vel_pct"),
        slope_med_slow_pct=lifecycle_inputs.get("slope_med_slow_pct"),
        slope_vel_slow_pct=lifecycle_inputs.get("slope_vel_slow_pct"),
        entry_gate_bypass=bool(entry_gate_bypass),
        policy_graph=policy.lifecycle_graph,
    )


def _spot_open_position_intent(
    *,
    policy: _SpotPolicyRun,
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
        signal_entry_dir=str(signal_entry_dir)
        if signal_entry_dir in ("up", "down")
        else None,
        shock_atr_pct=float(shock_atr_pct) if shock_atr_pct is not None else None,
        shock_atr_vel_pct=float(shock_atr_vel_pct)
        if shock_atr_vel_pct is not None
        else None,
        shock_atr_accel_pct=float(shock_atr_accel_pct)
        if shock_atr_accel_pct is not None
        else None,
        tr_ratio=float(tr_ratio) if tr_ratio is not None else None,
        tr_median_pct=float(tr_median_pct) if tr_median_pct is not None else None,
        slope_med_pct=float(slope_med_pct) if slope_med_pct is not None else None,
        slope_vel_pct=float(slope_vel_pct) if slope_vel_pct is not None else None,
        slope_med_slow_pct=float(slope_med_slow_pct)
        if slope_med_slow_pct is not None
        else None,
        slope_vel_slow_pct=float(slope_vel_slow_pct)
        if slope_vel_slow_pct is not None
        else None,
        policy_graph=policy.lifecycle_graph,
        policy_config=policy.lifecycle_config,
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


def _spot_try_open_entry(
    *,
    policy: _SpotPolicyRun,
    cfg: ConfigBundle,
    meta: ContractMeta,
    entry_dir: str,
    entry_branch: str | None,
    entry_leg: SpotLegConfig,
    entry_time: datetime,
    entry_ref_price: float,
    mark_ref_price: float,
    orb_engine,
    evidence: _SpotEntryEvidence,
    exec_profile: _SpotExecProfile,
    cash: float,
    margin_used: float,
    liquidation_value: float,
) -> tuple[SpotTrade, float, float] | None:
    snapshot = evidence.snapshot
    filters = cfg.strategy.filters
    entry_signal = normalize_spot_entry_signal(
        getattr(cfg.strategy, "entry_signal", "ema")
    )
    exit_mode = str(exec_profile.exit_mode)
    atr_value = getattr(snapshot, "atr", None)
    signal = getattr(snapshot, "signal", None)
    signal_entry_dir = getattr(snapshot, "entry_dir", None)
    signal_regime_dir = getattr(signal, "regime_dir", None)
    regime2_dir = evidence.regime.fast_dir
    regime2_ready = evidence.regime.fast_ready
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
        spread=float(exec_profile.spread),
        commission_per_share=float(exec_profile.commission_per_share),
        commission_min=float(exec_profile.commission_min),
        slippage_per_share=float(exec_profile.slippage_per_share),
    )

    can_open = True
    target_price: float | None = None
    stop_price: float | None = None
    profit_target_pct = cfg.strategy.spot_profit_target_pct
    stop_loss_pct = cfg.strategy.spot_stop_loss_pct

    if entry_signal == "orb" and orb_engine is not None and entry_dir in ("up", "down"):
        orb_high = orb_engine.or_high
        orb_low = orb_engine.or_low
        if (
            orb_high is not None
            and orb_low is not None
            and orb_high > 0
            and orb_low > 0
        ):
            stop_price = float(orb_low) if entry_dir == "up" else float(orb_high)
            rr = float(getattr(cfg.strategy, "orb_risk_reward", 2.0) or 2.0)
            target_mode = (
                str(getattr(cfg.strategy, "orb_target_mode", "rr") or "rr")
                .strip()
                .lower()
            )
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
                        float(orb_high) + (rr * rng)
                        if entry_dir == "up"
                        else float(orb_low) - (rr * rng)
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

    shock_on = bool(evidence.shock) if evidence.shock is not None else False
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
            shock_dir=evidence.shock_dir,
            shock_atr_pct=evidence.shock_atr_pct,
            shock_dir_down_streak_bars=getattr(
                snapshot, "shock_dir_down_streak_bars", None
            ),
            shock_drawdown_dist_on_pct=getattr(
                snapshot, "shock_drawdown_dist_on_pct", None
            ),
            shock_drawdown_dist_on_vel_pp=getattr(
                snapshot, "shock_drawdown_dist_on_vel_pp", None
            ),
            shock_drawdown_dist_on_accel_pp=getattr(
                snapshot, "shock_drawdown_dist_on_accel_pp", None
            ),
            shock_prearm_down_streak_bars=getattr(
                snapshot, "shock_prearm_down_streak_bars", None
            ),
            shock_ramp=getattr(snapshot, "shock_ramp", None),
            riskoff=evidence.riskoff,
            risk_dir=evidence.shock_dir,
            riskpanic=evidence.riskpanic,
            riskpop=evidence.riskpop,
            risk=evidence.risk_snapshot,
            signal_entry_dir=signal_entry_dir,
            signal_regime_dir=signal_regime_dir,
            regime2_dir=regime2_dir,
            regime2_ready=regime2_ready,
            equity_ref=float(cash) + float(liquidation_value),
            cash_ref=float(cash),
            policy_graph=policy.sizing_graph,
            policy_config=policy.sizing_config,
        )
        if signed_qty == 0:
            can_open = False
        else:
            size_mult = _spot_branch_size_mult(policy=policy, entry_branch=entry_branch)
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
                policy=policy,
                strategy=cfg.strategy,
                bar_ts=entry_time,
                bar_size=str(cfg.backtest.bar_size),
                open_dir=None,
                current_qty=0,
                target_qty=int(signed_qty),
                spot_decision=decision_trace.as_payload(),
                shock_atr_pct=evidence.shock_atr_pct,
                tr_ratio=float(getattr(evidence.risk_snapshot, "tr_ratio", 0.0))
                if evidence.risk_snapshot is not None
                and getattr(evidence.risk_snapshot, "tr_ratio", None) is not None
                else None,
                tr_median_pct=float(
                    getattr(evidence.risk_snapshot, "tr_median_pct", 0.0)
                )
                if evidence.risk_snapshot is not None
                and getattr(evidence.risk_snapshot, "tr_median_pct", None) is not None
                else None,
                slope_med_pct=float(
                    getattr(evidence.risk_snapshot, "tr_median_delta_pct", 0.0)
                )
                if evidence.risk_snapshot is not None
                and getattr(evidence.risk_snapshot, "tr_median_delta_pct", None)
                is not None
                else None,
                slope_vel_pct=float(
                    getattr(evidence.risk_snapshot, "tr_slope_vel_pct", 0.0)
                )
                if evidence.risk_snapshot is not None
                and getattr(evidence.risk_snapshot, "tr_slope_vel_pct", None)
                is not None
                else None,
            )
            intent_decision = lifecycle.spot_intent
            if (
                lifecycle.intent != "enter"
                or intent_decision is None
                or int(intent_decision.order_qty) <= 0
            ):
                can_open = False
            else:
                signed_qty = int(intent_decision.delta_qty)
                decision_trace_payload = decision_trace.as_payload()
                decision_trace_payload["regime2_bear_hard_dir"] = (
                    evidence.regime.hard_dir
                )
                decision_trace_payload["regime2_bear_hard_ready"] = (
                    evidence.regime.hard_ready
                )
                decision_trace_payload["regime2_bear_hard_release_age_bars"] = (
                    evidence.regime.hard_release_age_bars
                )
                decision_trace_payload["regime4_state"] = evidence.regime.label
                decision_trace_payload["regime4_transition_hot"] = (
                    evidence.regime.transition_hot
                )
                decision_trace_payload["regime4_owner"] = evidence.regime.owner
                decision_trace_payload["entry_guard_probe"] = evidence.guard_probe
                decision_trace_payload["entry_guard_inputs"] = evidence.guard_inputs
                decision_trace_payload["entry_local_extrema_probe"] = (
                    evidence.local_extrema
                )
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
        spread=float(exec_profile.spread),
        commission_per_share=float(exec_profile.commission_per_share),
        commission_min=float(exec_profile.commission_min),
        slippage_per_share=float(exec_profile.slippage_per_share),
    )

    if entry_signal == "orb" and orb_engine is not None and entry_dir in ("up", "down"):
        if stop_price is not None:
            rr = float(getattr(cfg.strategy, "orb_risk_reward", 2.0) or 2.0)
            target_mode = (
                str(getattr(cfg.strategy, "orb_target_mode", "rr") or "rr")
                .strip()
                .lower()
            )
            if target_mode not in ("rr", "or_range"):
                target_mode = "rr"
            if rr <= 0:
                return None
            if target_mode == "rr":
                risk = abs(float(entry_price) - float(stop_price))
                if risk <= 0:
                    return None
                target_price = (
                    float(entry_price) + (rr * risk)
                    if entry_dir == "up"
                    else float(entry_price) - (rr * risk)
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
        spread=float(exec_profile.spread),
        mark_to_market=str(exec_profile.mark_to_market),
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
    if (
        old_qty != 0
        and ((old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0))
        and new_abs > old_abs
    ):
        # Scale-in keeps weighted entry basis for remaining inventory.
        weight_old = float(old_abs)
        weight_add = float(new_abs - old_abs)
        if (weight_old + weight_add) > 0:
            trade.entry_price = (
                (float(trade.entry_price) * weight_old)
                + (float(resize_price) * weight_add)
            ) / (weight_old + weight_add)
    trade.qty = int(new_qty)

    old_margin = float(trade.margin_required or 0.0)
    mark_price = _spot_mark_price(
        float(mark_ref_price),
        qty=int(new_qty),
        spread=float(spread),
        mode=str(mark_to_market),
    )
    new_margin = abs(int(new_qty)) * float(mark_price) * float(multiplier)
    trade.margin_required = float(new_margin)
    next_margin_used = max(
        0.0, float(margin_used) - float(old_margin) + float(new_margin)
    )

    trace = trade.decision_trace if isinstance(trade.decision_trace, dict) else {}
    resizes = trace.get("resizes")
    rows = list(resizes) if isinstance(resizes, list) else []
    rows.append(
        {
            "delta_qty": int(delta),
            "new_qty": int(new_qty),
            "resize_price": float(resize_price),
            "lifecycle": dict(lifecycle_payload)
            if isinstance(lifecycle_payload, dict)
            else None,
            "spot_decision": dict(decision_payload)
            if isinstance(decision_payload, dict)
            else None,
        }
    )
    trace["resizes"] = rows
    trade.decision_trace = trace

    return True, float(next_cash), float(next_margin_used)


def _spot_emit_progress(progress_callback, **payload: object) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback(dict(payload))
    except Exception:
        return


def _spot_resolve_run_bars(
    cfg: ConfigBundle,
    *,
    bars: BarSeriesInput,
    exec_bars: BarSeriesInput | None = None,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    regime2_bear_hard_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
) -> _SpotRunBars:
    """Normalize tapes once and enforce the single/multi-resolution boundary."""
    signal = _bars_input_list(bars)
    execution = _bars_input_optional_list(exec_bars)
    exec_bar_size = str(
        getattr(cfg.strategy, "spot_exec_bar_size", "") or ""
    ).strip()
    if exec_bar_size and exec_bar_size != str(cfg.backtest.bar_size):
        if execution is None:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars was not provided "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
        if not execution:
            raise ValueError(
                "spot_exec_bar_size is set but exec_bars is empty "
                f"(signal={cfg.backtest.bar_size!r} exec={exec_bar_size!r})"
            )
    else:
        execution = signal
    return _SpotRunBars(
        signal=signal,
        execution=execution,
        regime=_bars_input_optional_list(regime_bars),
        regime2=_bars_input_optional_list(regime2_bars),
        regime2_bear_hard=_bars_input_optional_list(regime2_bear_hard_bars),
        tick=_bars_input_optional_list(tick_bars),
    )


def _run_spot_backtest(
    cfg: ConfigBundle,
    bars: BarSeriesInput,
    meta: ContractMeta,
    *,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    regime2_bear_hard_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
) -> BacktestResult:
    run_bars = _spot_resolve_run_bars(
        cfg,
        bars=bars,
        exec_bars=exec_bars,
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
        regime2_bear_hard_bars=regime2_bear_hard_bars,
        tick_bars=tick_bars,
    )
    return _run_spot_backtest_exec_loop(
        cfg,
        signal_bars=run_bars.signal,
        exec_bars=run_bars.execution,
        meta=meta,
        regime_bars=run_bars.regime,
        regime2_bars=run_bars.regime2,
        regime2_bear_hard_bars=run_bars.regime2_bear_hard,
        tick_bars=run_bars.tick,
    )


def _run_spot_backtest_summary(
    cfg: ConfigBundle,
    bars: BarSeriesInput,
    meta: ContractMeta,
    *,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    regime2_bear_hard_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    exec_bars: BarSeriesInput | None = None,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> SummaryStats:
    """Spot backtest optimized for sweeps that only need `SummaryStats`.

    Keeps semantics aligned with `_run_spot_backtest_exec_loop`, but skips building the
    full equity curve list (and a second aggregation pass over it).
    """
    run_bars = _spot_resolve_run_bars(
        cfg,
        bars=bars,
        exec_bars=exec_bars,
        regime_bars=regime_bars,
        regime2_bars=regime2_bars,
        regime2_bear_hard_bars=regime2_bear_hard_bars,
        tick_bars=tick_bars,
    )
    _spot_emit_progress(
        progress_callback,
        phase="summary.prepare",
        signal_total=int(len(run_bars.signal)),
        exec_total=int(len(run_bars.execution)),
    )
    return _run_spot_backtest_exec_loop_summary(
        cfg,
        signal_bars=run_bars.signal,
        exec_bars=run_bars.execution,
        meta=meta,
        regime_bars=run_bars.regime,
        regime2_bars=run_bars.regime2,
        regime2_bear_hard_bars=run_bars.regime2_bear_hard,
        tick_bars=run_bars.tick,
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
    regime2_bear_hard_bars: BarSeriesInput | None = None,
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
    regime2_bear_hard_bars = _bars_input_optional_list(regime2_bear_hard_bars)
    tick_bars = _bars_input_optional_list(tick_bars)

    cash = cfg.backtest.starting_cash
    margin_used = 0.0
    equity_curve: list[EquityPoint] = []
    trades: list[SpotTrade] = []
    open_trades: list[SpotTrade] = []
    trace_out_raw = getattr(cfg.strategy, "spot_lifecycle_trace_path", None)
    why_not_out_raw = getattr(cfg.strategy, "spot_why_not_report_path", None)
    capture_lifecycle = (
        bool(getattr(cfg.strategy, "spot_capture_lifecycle_trace", False))
        or bool(str(trace_out_raw or "").strip())
        or bool(str(why_not_out_raw or "").strip())
    )
    lifecycle_rows: list[dict[str, object]] | None = [] if capture_lifecycle else None

    if not signal_bars:
        raise ValueError("signal_bars is empty")
    if not exec_bars:
        raise ValueError("exec_bars is empty")

    filters = cfg.strategy.filters
    policy = _SpotPolicyRun(
        lifecycle_graph=SpotPolicyGraph.from_sources(
            strategy=cfg.strategy, filters=None
        ),
        lifecycle_config=SpotPolicyConfigView.from_sources(
            strategy=cfg.strategy, filters=None
        ),
        sizing_graph=SpotPolicyGraph.from_sources(
            strategy=cfg.strategy, filters=filters
        ),
        sizing_config=SpotPolicyConfigView.from_sources(
            strategy=cfg.strategy, filters=filters
        ),
    )
    entry_signal = normalize_spot_entry_signal(
        getattr(cfg.strategy, "entry_signal", "ema")
    )

    ema_periods = (
        _ema_periods_shared(cfg.strategy.ema_preset) if entry_signal == "ema" else None
    )
    needs_direction = cfg.strategy.directional_spot is not None
    if entry_signal == "ema" and ema_periods is None:
        raise ValueError("spot backtests require ema_preset")
    ema_needed = entry_signal == "ema"

    _regime_mode, _regime_preset, _regime_bar, use_mtf_regime_cfg = (
        resolve_spot_regime_spec(
            bar_size=cfg.backtest.bar_size,
            regime_mode_raw=getattr(cfg.strategy, "regime_mode", "ema"),
            regime_ema_preset_raw=getattr(cfg.strategy, "regime_ema_preset", None),
            regime_bar_size_raw=getattr(cfg.strategy, "regime_bar_size", None),
        )
    )
    use_mtf_regime = bool(regime_bars) and bool(use_mtf_regime_cfg)
    regime2_mode, _regime2_preset, _regime2_bar, use_mtf_regime2_cfg = (
        resolve_spot_regime2_spec(
            bar_size=cfg.backtest.bar_size,
            regime2_mode_raw=getattr(cfg.strategy, "regime2_mode", "off"),
            regime2_ema_preset_raw=getattr(cfg.strategy, "regime2_ema_preset", None),
            regime2_bar_size_raw=getattr(cfg.strategy, "regime2_bar_size", None),
        )
    )
    if regime2_mode != "off" and bool(use_mtf_regime2_cfg) and not regime2_bars:
        raise ValueError(
            "regime2_mode enabled but regime2_bars was not provided for multi-timeframe regime2"
        )
    use_mtf_regime2 = bool(regime2_bars) and bool(use_mtf_regime2_cfg)

    from ..spot_engine import SpotSignalEvaluator

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
        regime2_bear_hard_bars=regime2_bear_hard_bars,
    )
    orb_engine = evaluator.orb_engine
    last_sig_snap: SpotSignalSnapshot | None = None
    last_sig_exec_idx = -1
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
    series_pack = (
        prepared_series_pack
        if isinstance(prepared_series_pack, _SpotSeriesPack)
        else None
    )
    if series_pack is None:
        series_pack = _spot_build_series_pack(
            cfg=cfg,
            signal_bars=signal_bars,
            exec_bars=exec_bars,
            tick_bars=tick_bars,
            include_tick=(str(tick_mode) != "off"),
        )
    align = series_pack.align
    tick_series = series_pack.tick_series
    exec_dates = series_pack.exec_dates

    exec_profile = _spot_exec_profile(cfg.strategy)
    exit_mode = str(exec_profile.exit_mode)
    spot_exit_time = parse_time_hhmm(getattr(cfg.strategy, "spot_exit_time_et", None))
    spot_exec_bar_size = str(
        getattr(cfg.strategy, "spot_exec_bar_size", "") or cfg.backtest.bar_size or ""
    )
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
    spot_resize_relevant = (
        lifecycle_rows is not None
        or str(policy.lifecycle_config.spot_resize_mode) == "target"
        or str(policy.sizing_config.sizing_mode) != "fixed"
    )
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

    def _latest_signal_snapshot_probe(*, exec_idx: int) -> dict[str, object] | None:
        if last_sig_snap is None:
            return None
        risk_snap = getattr(last_sig_snap, "risk", None)
        age_bars = (
            int(exec_idx) - int(last_sig_exec_idx)
            if int(last_sig_exec_idx) >= 0
            else None
        )
        return {
            "signal_bar_ts": last_sig_snap.bar_ts.isoformat(),
            "signal_snapshot_age_bars": int(age_bars)
            if age_bars is not None and age_bars >= 0
            else None,
            "shock_atr_pct": float(last_sig_snap.shock_atr_pct)
            if last_sig_snap.shock_atr_pct is not None
            else None,
            "shock_atr_vel_pct": (
                float(last_sig_snap.shock_atr_vel_pct)
                if getattr(last_sig_snap, "shock_atr_vel_pct", None) is not None
                else None
            ),
            "shock_atr_accel_pct": (
                float(last_sig_snap.shock_atr_accel_pct)
                if getattr(last_sig_snap, "shock_atr_accel_pct", None) is not None
                else None
            ),
            "tr_ratio": float(last_sig_snap.ratsv_tr_ratio)
            if last_sig_snap.ratsv_tr_ratio is not None
            else None,
            "tr_median_pct": (
                float(getattr(risk_snap, "tr_median_pct"))
                if risk_snap is not None
                and getattr(risk_snap, "tr_median_pct", None) is not None
                else None
            ),
            "slope_med_pct": (
                float(last_sig_snap.ratsv_fast_slope_med_pct)
                if last_sig_snap.ratsv_fast_slope_med_pct is not None
                else None
            ),
            "slope_vel_pct": (
                float(last_sig_snap.ratsv_fast_slope_vel_pct)
                if last_sig_snap.ratsv_fast_slope_vel_pct is not None
                else None
            ),
            "slope_med_slow_pct": (
                float(last_sig_snap.ratsv_slow_slope_med_pct)
                if last_sig_snap.ratsv_slow_slope_med_pct is not None
                else None
            ),
            "slope_vel_slow_pct": (
                float(last_sig_snap.ratsv_slow_slope_vel_pct)
                if last_sig_snap.ratsv_slow_slope_vel_pct is not None
                else None
            ),
            "entry_dir": str(last_sig_snap.entry_dir)
            if last_sig_snap.entry_dir in ("up", "down")
            else None,
            "entry_branch": str(last_sig_snap.entry_branch)
            if last_sig_snap.entry_branch in ("a", "b")
            else None,
            "regime4_state": str(last_sig_snap.regime4_state)
            if last_sig_snap.regime4_state
            else None,
            "hard_dir": (
                str(last_sig_snap.regime2_bear_hard_dir)
                if last_sig_snap.regime2_bear_hard_dir in ("up", "down")
                else None
            ),
        }

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
        exit_trace_payload: dict[str, object] | None = None,
        exit_exec_idx: int | None = None,
    ) -> tuple[float, float, float]:
        exit_price, next_cash, next_margin_used = _spot_exec_exit_common(
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
        trace = trade.decision_trace if isinstance(trade.decision_trace, dict) else {}
        rows_raw = trace.get("exits")
        rows = list(rows_raw) if isinstance(rows_raw, list) else []
        payload = (
            dict(exit_trace_payload) if isinstance(exit_trace_payload, dict) else {}
        )
        if exit_exec_idx is not None:
            payload["local_extrema_probe"] = _spot_local_extrema_probe(
                bars=exec_bars,
                exec_idx=int(exit_exec_idx),
                ref_price=float(exit_price),
                bar_size=spot_exec_bar_size,
            )
        if payload:
            rows.append(payload)
            trace["exits"] = rows
            trade.decision_trace = trace
        return float(exit_price), float(next_cash), float(next_margin_used)

    pending_entry = _SpotPendingEntry()
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
    riskoff_end_hour = spot_riskoff_end_hour(filters) if risk_overlay_enabled else None
    last_entry_sig_idx: int | None = None
    last_resize_bar_ts: datetime | None = None

    exec_last_date = None
    entries_today = 0
    exec_total = int(len(exec_bars))
    progress_stride = max(1, int(max(64, exec_total // 200)))
    score_start_dt = _spot_score_start_dt(cfg)
    start_exec_idx = bisect_left([bar.ts for bar in exec_bars], score_start_dt)
    if start_exec_idx > 0:
        for warm_idx in range(int(start_exec_idx)):
            warm_bar = exec_bars[int(warm_idx)]
            warm_next = (
                exec_bars[int(warm_idx) + 1]
                if int(warm_idx) + 1 < len(exec_bars)
                else None
            )
            warm_is_last_bar = (
                warm_next is None
                or exec_dates[int(warm_idx) + 1] != exec_dates[int(warm_idx)]
            )
            evaluator.update_exec_bar(warm_bar, is_last_bar=bool(warm_is_last_bar))
            warm_sig_map_idx = (
                align.sig_idx_by_exec_idx[int(warm_idx)]
                if int(warm_idx) < len(align.sig_idx_by_exec_idx)
                else -1
            )
            warm_sig_idx = int(warm_sig_map_idx) if int(warm_sig_map_idx) >= 0 else None
            if warm_sig_idx is None or int(warm_sig_idx) >= len(signal_bars):
                shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view
                continue
            warm_sig_bar = signal_bars[int(warm_sig_idx)]
            if warm_sig_bar.ts >= score_start_dt:
                shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view
                continue
            warm_sig_snap = evaluator.update_signal_bar(warm_sig_bar)
            if warm_sig_snap is not None:
                last_sig_snap = warm_sig_snap
            shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view

    for idx in range(int(start_exec_idx), len(exec_bars)):
        bar = exec_bars[int(idx)]
        next_bar = exec_bars[idx + 1] if idx + 1 < len(exec_bars) else None
        bar_day = exec_dates[int(idx)]
        is_last_bar = next_bar is None or exec_dates[int(idx) + 1] != bar_day
        sig_map_idx = (
            align.sig_idx_by_exec_idx[idx]
            if idx < len(align.sig_idx_by_exec_idx)
            else -1
        )
        sig_idx = int(sig_map_idx) if int(sig_map_idx) >= 0 else None
        sig_bar = (
            signal_bars[int(sig_idx)]
            if sig_idx is not None and int(sig_idx) < len(signal_bars)
            else None
        )
        if sig_bar is not None and sig_bar.ts < score_start_dt:
            sig_idx = None
            sig_bar = None
        if (
            int(idx) == 0
            or int((idx + 1) % int(progress_stride)) == 0
            or int(idx + 1) >= int(exec_total)
        ):
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

        if exec_last_date != bar_day:
            exec_last_date = bar_day
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

        pending_decision = None
        if lifecycle_rows is not None or pending_entry.active or bool(pending_exit_all):
            open_dir_now = (
                "up"
                if open_trades and int(open_trades[0].qty) > 0
                else "down"
                if open_trades
                else None
            )
            pending_decision = decide_pending_next_open(
                now_ts=bar.ts,
                has_open=bool(open_trades),
                open_dir=open_dir_now,
                pending_entry_dir=pending_entry.direction,
                pending_entry_set_date=pending_entry.set_date,
                pending_entry_due_ts=pending_entry.due_ts
                if pending_entry.active
                else None,
                pending_exit_reason=pending_exit_reason,
                pending_exit_due_ts=pending_exit_due_ts
                if bool(pending_exit_all)
                else None,
                risk_overlay_enabled=bool(risk_overlay_enabled),
                riskoff_today=bool(riskoff_today),
                riskpanic_today=bool(riskpanic_today),
                riskpop_today=bool(riskpop_today),
                riskoff_mode=str(riskoff_mode),
                shock_dir_now=shock_dir_prev_now,
                riskoff_end_hour=riskoff_end_hour,
                pending_entry_fill_mode=spot_entry_fill_mode,
                pending_exit_fill_mode=(
                    spot_flip_exit_fill_mode
                    if str(pending_exit_reason or "").strip().lower() == "flip"
                    else SPOT_FILL_MODE_CLOSE
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
        if pending_decision is not None and pending_decision.pending_clear_exit:
            pending_exit_all = False
            pending_exit_reason = ""
            pending_exit_due_ts = None

        if (
            pending_decision is not None
            and pending_decision.intent == "exit"
            and open_trades
        ):
            exit_ref = float(bar.open)
            for trade in list(open_trades):
                _exit_price, cash, margin_used = _exec_trade_exit(
                    trade,
                    exit_ref_price=exit_ref,
                    exit_time=bar.ts,
                    reason=str(
                        pending_decision.reason or pending_exit_reason or "flip"
                    ),
                    apply_slippage=True,
                    exit_trace_payload={
                        "stage": "pending_exit",
                        "bar_ts": bar.ts.isoformat(),
                        "exit_ref_price": float(exit_ref),
                        "apply_slippage": True,
                        "pending_exit_reason": str(pending_exit_reason or "") or None,
                        "signal_snapshot": _latest_signal_snapshot_probe(
                            exec_idx=int(idx)
                        ),
                        "lifecycle": pending_decision.as_payload(),
                    },
                    exit_exec_idx=int(idx),
                )
            open_trades = []

        if (
            pending_decision is not None
            and pending_decision.intent == "enter"
            and pending_entry.active
        ):
            can_fill_pending = lifecycle_entry_capacity_ok(
                open_count=len(open_trades),
                max_entries_per_day=int(cfg.strategy.max_entries_per_day),
                entries_today=int(entries_today),
                weekday=_trade_weekday(bar.ts),
                entry_days=cfg.strategy.entry_days,
            )
            if can_fill_pending:
                filled_entry = pending_entry
                pending_entry = _SpotPendingEntry()

                entry_leg = _spot_entry_leg_for_direction(
                    strategy=cfg.strategy,
                    entry_dir=filled_entry.direction,
                    needs_direction=needs_direction,
                )
                if entry_leg is not None and filled_entry.direction in ("up", "down"):
                    liquidation_open = _spot_liquidation(float(bar.open))
                    opened = _spot_try_open_entry(
                        policy=policy,
                        cfg=cfg,
                        meta=meta,
                        entry_dir=filled_entry.direction,
                        entry_branch=filled_entry.branch,
                        entry_leg=entry_leg,
                        entry_time=bar.ts,
                        entry_ref_price=float(bar.open),
                        mark_ref_price=float(bar.open),
                        orb_engine=orb_engine,
                        evidence=_SpotEntryEvidence.from_signal(
                            snapshot=last_sig_snap,
                            shock=shock_now_prev,
                            shock_dir=shock_dir_prev_now,
                            shock_atr_pct=shock_atr_pct_prev_now,
                            riskoff=riskoff_today,
                            riskpanic=riskpanic_today,
                            riskpop=riskpop_today,
                            risk_snapshot=evaluator.last_risk
                            if risk_overlay_enabled
                            else None,
                            pending=filled_entry,
                            guard_probe=filled_entry.guard_probe,
                            guard_inputs=filled_entry.guard_inputs,
                            local_extrema=_spot_local_extrema_probe(
                                bars=exec_bars,
                                exec_idx=int(idx),
                                ref_price=float(bar.open),
                                bar_size=spot_exec_bar_size,
                            ),
                        ),
                        exec_profile=exec_profile,
                        cash=float(cash),
                        margin_used=float(margin_used),
                        liquidation_value=float(liquidation_open),
                    )
                    if opened is not None:
                        candidate, cash, margin_used = opened
                        open_trades.append(candidate)
                        entries_today += 1
            else:
                pending_entry = _SpotPendingEntry()

        if pending_decision is not None and pending_decision.pending_clear_entry:
            pending_entry = _SpotPendingEntry()

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
                if trade.stop_loss_price is None and scaled_sl is not None:
                    trade.stop_loss_pct = float(scaled_sl)
                if trade.profit_target_price is None and scaled_pt is not None:
                    trade.profit_target_pct = float(scaled_pt)

        # Signals advance only on their closes. Router ownership remains sticky across
        # faster execution bars so host-managed lanes never regain local exits.
        sig_snap = evaluator.update_signal_bar(sig_bar) if sig_bar is not None else None
        if sig_snap is not None:
            last_sig_snap = sig_snap
            last_sig_exec_idx = int(idx)
        router_snap = sig_snap or last_sig_snap
        router_ready = bool(router_snap and router_snap.regime_router_ready)
        router_host_managed = bool(
            router_snap and router_snap.regime_router_host_managed
        )
        signal_inputs = sig_snap.lifecycle_inputs() if sig_snap is not None else {}
        signal_trace = (
            sig_snap.lifecycle_trace()
            if sig_snap is not None
            else {key: None for key in _SPOT_SIGNAL_TRACE_KEYS}
        )
        entry_context = sig_snap.entry_context() if sig_snap is not None else {}

        # Track worst-in-bar equity using the execution bars (for drawdown realism).
        if spot_drawdown_mode == "intrabar" and open_trades:
            worst_liquidation = 0.0
            for trade in open_trades:
                stop_level = spot_stop_level(
                    float(trade.entry_price),
                    int(trade.qty),
                    stop_loss_price=trade.stop_loss_price,
                    stop_loss_pct=(
                        None if router_host_managed else trade.stop_loss_pct
                    ),
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
                    * _spot_mark_price(
                        worst_ref,
                        qty=trade.qty,
                        spread=spot_spread,
                        mode=spot_mark_to_market,
                    )
                    * meta.multiplier
                )
            _record_equity_point(
                bar.ts - timedelta(microseconds=1), cash + worst_liquidation
            )

        # Exit checks (profit/stop always; flip only on signal-bar closes).
        if open_trades:
            still_open: list[SpotTrade] = []
            for trade in open_trades:
                exit_candidates: dict[str, bool] = {}
                exit_ref_by_reason: dict[str, float] = {}
                apply_slippage_by_reason: dict[str, bool] = {}

                if (not router_host_managed) and spot_intrabar_exits:
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
                            reason = (
                                "stop_loss"
                                if trade.stop_loss_price is not None
                                else "stop_loss_pct"
                            )
                        else:
                            reason = (
                                "profit_target"
                                if trade.profit_target_price is not None
                                else "profit_target_pct"
                            )
                        exit_candidates[reason] = True
                        exit_ref_by_reason[reason] = float(ref)
                        apply_slippage_by_reason[reason] = kind != "profit"
                elif not router_host_managed:
                    if spot_hit_profit(
                        entry_price=float(trade.entry_price),
                        qty=int(trade.qty),
                        price=float(bar.close),
                        profit_target_price=trade.profit_target_price,
                        profit_target_pct=trade.profit_target_pct,
                    ):
                        reason = (
                            "profit_target"
                            if trade.profit_target_price is not None
                            else "profit_target_pct"
                        )
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
                        reason = (
                            "stop_loss"
                            if trade.stop_loss_price is not None
                            else "stop_loss_pct"
                        )
                        exit_candidates[reason] = True
                        exit_ref_by_reason[reason] = float(bar.close)
                        apply_slippage_by_reason[reason] = True

                is_signal_close = sig_idx is not None
                if (
                    is_signal_close
                    and not router_host_managed
                    and _spot_ratsv_probe_cancel_hit(
                        cfg,
                        trade=trade,
                        bar=bar,
                        tr_ratio=signal_inputs.get("tr_ratio"),
                        slope_med=signal_inputs.get("slope_med_pct"),
                    )
                ):
                    exit_candidates["ratsv_probe_cancel"] = True
                    exit_ref_by_reason["ratsv_probe_cancel"] = float(bar.close)
                    apply_slippage_by_reason["ratsv_probe_cancel"] = True
                if (
                    is_signal_close
                    and not router_host_managed
                    and _spot_ratsv_adverse_release_hit(
                        cfg,
                        trade=trade,
                        bar=bar,
                        tr_ratio=signal_inputs.get("tr_ratio"),
                        slope_med=signal_inputs.get("slope_med_pct"),
                        slope_vel=signal_inputs.get("slope_vel_pct"),
                    )
                ):
                    exit_candidates["ratsv_adverse_release"] = True
                    exit_ref_by_reason["ratsv_adverse_release"] = float(bar.close)
                    apply_slippage_by_reason["ratsv_adverse_release"] = True
                if (
                    is_signal_close
                    and (not router_host_managed)
                    and _spot_hit_flip_exit(
                        cfg,
                        trade,
                        bar,
                        sig_snap.signal if sig_snap is not None else None,
                        tr_ratio=signal_inputs.get("tr_ratio"),
                        shock_atr_vel_pct=signal_inputs.get("shock_atr_vel_pct"),
                        tr_median_pct=signal_inputs.get("tr_median_pct"),
                    )
                ):
                    exit_candidates["flip"] = True
                    exit_ref_by_reason["flip"] = float(bar.close)
                    apply_slippage_by_reason["flip"] = True
                elif is_signal_close and router_ready and router_host_managed:
                    trade_dir = (
                        "up"
                        if int(trade.qty) > 0
                        else "down"
                        if int(trade.qty) < 0
                        else None
                    )
                    if (
                        trade_dir in ("up", "down")
                        and signal_inputs.get("signal_entry_dir") != trade_dir
                    ):
                        exit_candidates["flip"] = True
                        exit_ref_by_reason["flip"] = float(bar.close)
                        apply_slippage_by_reason["flip"] = True
                if (not router_host_managed) and spot_exit_time is not None:
                    ts_et = _ts_to_et(bar.ts)
                    if ts_et.time() >= spot_exit_time:
                        exit_candidates["exit_time"] = True
                        exit_ref_by_reason["exit_time"] = float(bar.close)
                        apply_slippage_by_reason["exit_time"] = True
                if (not router_host_managed) and bool(spot_close_eod) and is_last_bar:
                    exit_candidates["close_eod"] = True
                    exit_ref_by_reason["close_eod"] = float(bar.close)
                    apply_slippage_by_reason["close_eod"] = True

                if lifecycle_rows is None and not any(exit_candidates.values()):
                    still_open.append(trade)
                    continue
                lifecycle = _spot_open_position_intent(
                    policy=policy,
                    strategy=cfg.strategy,
                    bar_ts=bar.ts,
                    bar_size=str(cfg.backtest.bar_size),
                    open_dir="up" if int(trade.qty) > 0 else "down",
                    current_qty=int(trade.qty),
                    exit_candidates=exit_candidates,
                    **signal_inputs,
                )
                _capture_lifecycle(
                    stage="open_exit",
                    decision=lifecycle,
                    bar_ts=bar.ts,
                    exec_idx=int(idx),
                    sig_idx=int(sig_idx) if sig_idx is not None else None,
                    context=signal_trace,
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
                    if (
                        lifecycle.queue_reentry_dir in ("up", "down")
                        and not pending_entry.active
                    ):
                        pending_entry = _SpotPendingEntry.from_signal(
                            direction=str(lifecycle.queue_reentry_dir),
                            branch=sig_snap.entry_branch
                            if sig_snap is not None
                            else None,
                            set_date=bar_day,
                            due_ts=due_ts,
                            snapshot=sig_snap,
                        )
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
                        exit_trace_payload={
                            "stage": "open_exit",
                            "bar_ts": bar.ts.isoformat(),
                            "exit_ref_price": float(exit_ref),
                            "apply_slippage": bool(apply_slippage),
                            "resolved_exit_reason": str(reason),
                            "exit_candidates": dict(exit_candidates)
                            if isinstance(exit_candidates, dict)
                            else None,
                            "signal_snapshot": _latest_signal_snapshot_probe(
                                exec_idx=int(idx)
                            ),
                            "lifecycle": lifecycle.as_payload(),
                        },
                        exit_exec_idx=int(idx),
                    )
                else:
                    still_open.append(trade)
            open_trades = still_open

        # Update equity after processing this execution bar.
        liquidation = 0.0
        for trade in open_trades:
            liquidation += (
                trade.qty
                * _spot_mark_price(
                    float(bar.close),
                    qty=trade.qty,
                    spread=spot_spread,
                    mode=spot_mark_to_market,
                )
                * meta.multiplier
            )
        _record_equity_point(bar.ts, cash + liquidation)

        if sig_bar is None or sig_idx is None:
            shock_prev, shock_dir_prev, shock_atr_pct_prev = evaluator.shock_view
            continue

        direction, _ = _spot_resolve_entry_dir(
            signal=sig_snap.signal if sig_snap is not None else None,
            entry_dir=signal_inputs.get("signal_entry_dir"),
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
        if pending_entry.active:
            effective_open += 1
        entry_plan = None
        if entry_ok and effective_open == 0:
            entry_plan = lifecycle_deferred_entry_plan(
                fill_mode=spot_entry_fill_mode,
                signal_ts=bar.ts,
                signal_close_ts=bar.ts,
                exec_bar_size=spot_exec_bar_size,
                strategy=cfg.strategy,
                riskoff_today=bool(riskoff_today),
                riskoff_end_hour=riskoff_end_hour,
                exit_mode=exit_mode,
                atr_value=float(sig_snap.atr)
                if sig_snap is not None and sig_snap.atr is not None
                else None,
                naive_ts_mode="utc",
            )
            if next_bar is None and entry_plan.deferred:
                entry_plan = replace(
                    entry_plan, due_ts=None, allowed=False, reason="no_next_bar"
                )
        next_open_ok = bool(
            entry_plan is not None and entry_plan.allowed and not pending_entry.active
        )
        entry_decision = _spot_flat_entry_decision_from_signal(
            policy=policy,
            strategy=cfg.strategy,
            filters=filters,
            signal_bar=sig_bar,
            signal=sig_snap.signal if sig_snap is not None else None,
            direction=direction,
            entry_context=entry_context,
            bars_in_day=int(sig_snap.bars_in_day) if sig_snap is not None else 0,
            volume_ema=float(sig_snap.volume_ema)
            if sig_snap is not None and sig_snap.volume_ema is not None
            else None,
            volume_ema_ready=bool(sig_snap is None or sig_snap.volume_ema_ready),
            rv=float(sig_snap.rv)
            if sig_snap is not None and sig_snap.rv is not None
            else None,
            cooldown_ok=bool(cooldown_ok),
            shock=sig_snap.shock if sig_snap is not None else None,
            shock_dir=sig_snap.shock_dir if sig_snap is not None else None,
            open_count=int(effective_open),
            entries_today=int(entries_today),
            pending_exists=bool(pending_entry.active or pending_exit_all),
            next_open_allowed=bool(next_open_ok),
            exit_mode=exit_mode,
            atr_value=float(sig_snap.atr)
            if sig_snap is not None and sig_snap.atr is not None
            else None,
            lifecycle_inputs=signal_inputs,
            entry_gate_bypass=bool(
                sig_snap
                and (
                    sig_snap.regime_router_host_managed
                    or sig_snap.regime_router_bull_sovereign_ok
                )
            ),
        )
        _capture_lifecycle(
            stage="flat_entry",
            decision=entry_decision,
            bar_ts=sig_bar.ts,
            exec_idx=int(idx),
            sig_idx=int(sig_idx),
            context=signal_trace,
        )
        entry_guard_probe_now = None
        if isinstance(getattr(entry_decision, "trace", None), dict):
            raw_graph_entry = entry_decision.trace.get("graph_entry")
            if isinstance(raw_graph_entry, dict):
                entry_guard_probe_now = dict(raw_graph_entry)
        entry_guard_inputs_now = dict(entry_context)
        entry_guard_inputs_now.update(
            {
                key: value
                for key, value in signal_inputs.items()
                if key != "signal_entry_dir"
            }
        )
        if entry_decision.intent == "enter" and entry_plan is not None:
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
                        atr_value=float(sig_snap.atr)
                        if sig_snap is not None and sig_snap.atr is not None
                        else None,
                        naive_ts_mode="utc",
                        due_ts=entry_plan.due_ts if same_mode else None,
                    )
                    if next_bar is None and entry_schedule.deferred:
                        entry_schedule = replace(
                            entry_schedule,
                            due_ts=None,
                            allowed=False,
                            reason="no_next_bar",
                        )
                    can_schedule = bool(
                        entry_schedule.allowed
                        and entry_schedule.due_ts is not None
                        and not pending_entry.active
                    )
                    if can_schedule:
                        pending_entry = _SpotPendingEntry.from_signal(
                            direction=direction,
                            branch=sig_snap.entry_branch
                            if sig_snap is not None
                            else None,
                            set_date=bar_day,
                            due_ts=entry_schedule.due_ts,
                            snapshot=sig_snap,
                            guard_probe=entry_guard_probe_now,
                            guard_inputs=entry_guard_inputs_now,
                        )
                        last_entry_sig_idx = int(sig_idx)
                else:
                    liquidation_close = _spot_liquidation(float(bar.close))
                    opened = _spot_try_open_entry(
                        policy=policy,
                        cfg=cfg,
                        meta=meta,
                        entry_dir=direction,
                        entry_branch=sig_snap.entry_branch
                        if sig_snap is not None and sig_snap.entry_branch in ("a", "b")
                        else None,
                        entry_leg=spot_leg,
                        entry_time=sig_bar.ts,
                        entry_ref_price=float(bar.close),
                        mark_ref_price=float(bar.close),
                        orb_engine=orb_engine,
                        evidence=_SpotEntryEvidence.from_signal(
                            snapshot=sig_snap,
                            shock=sig_snap.shock if sig_snap is not None else None,
                            shock_dir=sig_snap.shock_dir
                            if sig_snap is not None
                            else None,
                            shock_atr_pct=sig_snap.shock_atr_pct
                            if sig_snap is not None
                            else None,
                            riskoff=riskoff_today,
                            riskpanic=riskpanic_today,
                            riskpop=riskpop_today,
                            risk_snapshot=evaluator.last_risk
                            if risk_overlay_enabled
                            else None,
                            guard_probe=entry_guard_probe_now,
                            guard_inputs=entry_guard_inputs_now,
                            local_extrema=_spot_local_extrema_probe(
                                bars=exec_bars,
                                exec_idx=int(idx),
                                ref_price=float(bar.close),
                                bar_size=spot_exec_bar_size,
                            ),
                        ),
                        exec_profile=exec_profile,
                        cash=float(cash),
                        margin_used=float(margin_used),
                        liquidation_value=float(liquidation_close),
                    )
                    if opened is not None:
                        candidate, cash, margin_used = opened
                        open_trades.append(candidate)
                        entries_today += 1
                        last_entry_sig_idx = int(sig_idx)

        if (
            spot_resize_relevant
            and sig_idx is not None
            and open_trades
            and (not pending_exit_all)
            and len(open_trades) == 1
        ):
            trade = open_trades[0]
            trade_dir = "up" if int(trade.qty) > 0 else "down"
            resize_leg = _spot_entry_leg_for_direction(
                strategy=cfg.strategy,
                entry_dir=trade_dir,
                needs_direction=needs_direction,
            )
            if resize_leg is not None and trade_dir in ("up", "down"):
                action = (
                    str(getattr(resize_leg, "action", "BUY") or "BUY").strip().upper()
                )
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

                stop_price = (
                    float(trade.stop_loss_price)
                    if trade.stop_loss_price is not None
                    else None
                )
                if stop_price is not None and stop_price <= 0:
                    stop_price = None
                stop_loss_pct = (
                    float(trade.stop_loss_pct)
                    if trade.stop_loss_pct is not None
                    else None
                )
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
                    shock=sig_snap.shock if sig_snap is not None else None,
                    shock_dir=sig_snap.shock_dir if sig_snap is not None else None,
                    shock_atr_pct=sig_snap.shock_atr_pct
                    if sig_snap is not None
                    else None,
                    shock_dir_down_streak_bars=(
                        int(sig_snap.shock_dir_down_streak_bars)
                        if sig_snap is not None
                        and sig_snap.shock_dir_down_streak_bars is not None
                        else None
                    ),
                    shock_drawdown_dist_on_pct=(
                        float(sig_snap.shock_drawdown_dist_on_pct)
                        if sig_snap is not None
                        and sig_snap.shock_drawdown_dist_on_pct is not None
                        else None
                    ),
                    shock_drawdown_dist_on_vel_pp=(
                        float(sig_snap.shock_drawdown_dist_on_vel_pp)
                        if sig_snap is not None
                        and sig_snap.shock_drawdown_dist_on_vel_pp is not None
                        else None
                    ),
                    shock_drawdown_dist_on_accel_pp=(
                        float(
                            getattr(
                                last_sig_snap, "shock_drawdown_dist_on_accel_pp", 0.0
                            )
                        )
                        if last_sig_snap is not None
                        and getattr(
                            last_sig_snap, "shock_drawdown_dist_on_accel_pp", None
                        )
                        is not None
                        else None
                    ),
                    shock_prearm_down_streak_bars=(
                        int(getattr(last_sig_snap, "shock_prearm_down_streak_bars", 0))
                        if last_sig_snap is not None
                        and getattr(
                            last_sig_snap, "shock_prearm_down_streak_bars", None
                        )
                        is not None
                        else None
                    ),
                    shock_ramp=getattr(last_sig_snap, "shock_ramp", None)
                    if last_sig_snap is not None
                    else None,
                    riskoff=bool(riskoff_today),
                    risk_dir=sig_snap.shock_dir if sig_snap is not None else None,
                    riskpanic=bool(riskpanic_today),
                    riskpop=bool(riskpop_today),
                    risk=evaluator.last_risk if risk_overlay_enabled else None,
                    signal_entry_dir=signal_inputs.get("signal_entry_dir"),
                    signal_regime_dir=(
                        sig_snap.signal.regime_dir
                        if sig_snap is not None
                        and sig_snap.signal is not None
                        and sig_snap.signal.regime_dir in ("up", "down")
                        else None
                    ),
                    regime2_dir=(
                        sig_snap.regime2_dir
                        if sig_snap is not None
                        and sig_snap.regime2_dir in ("up", "down")
                        else None
                    ),
                    regime2_ready=bool(sig_snap and sig_snap.regime2_ready),
                    equity_ref=float(cash) + float(liquidation_close),
                    cash_ref=float(cash),
                    policy_graph=policy.sizing_graph,
                    policy_config=policy.sizing_config,
                )
                if int(signed_target) != 0:
                    size_mult = _spot_branch_size_mult(
                        policy=policy, entry_branch=trade.entry_branch
                    )
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
                        policy=policy,
                        strategy=cfg.strategy,
                        bar_ts=bar.ts,
                        bar_size=str(cfg.backtest.bar_size),
                        open_dir=trade_dir,
                        current_qty=int(trade.qty),
                        target_qty=int(signed_target),
                        spot_decision=resize_trace.as_payload(),
                        last_resize_bar_ts=last_resize_bar_ts,
                        **signal_inputs,
                    )
                    _capture_lifecycle(
                        stage="open_resize",
                        decision=lifecycle,
                        bar_ts=bar.ts,
                        exec_idx=int(idx),
                        sig_idx=int(sig_idx),
                        context=signal_trace,
                    )
                    if (
                        lifecycle.intent == "resize"
                        and lifecycle.spot_intent is not None
                    ):
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
                        if (
                            spot_fill_mode_is_deferred(lifecycle.fill_mode)
                            and next_bar is not None
                        ):
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
                                pending_exit_reason = str(
                                    lifecycle.reason or "target_zero"
                                )
                            else:
                                _exit_price, cash, margin_used = _exec_trade_exit(
                                    trade,
                                    exit_ref_price=float(bar.close),
                                    exit_time=bar.ts,
                                    reason=str(lifecycle.reason or "target_zero"),
                                    apply_slippage=True,
                                    exit_trace_payload={
                                        "stage": "open_resize_exit",
                                        "bar_ts": bar.ts.isoformat(),
                                        "exit_ref_price": float(bar.close),
                                        "apply_slippage": True,
                                        "resolved_exit_reason": str(
                                            lifecycle.reason or "target_zero"
                                        ),
                                        "signal_snapshot": _latest_signal_snapshot_probe(
                                            exec_idx=int(idx)
                                        ),
                                        "lifecycle": lifecycle.as_payload(),
                                    },
                                    exit_exec_idx=int(idx),
                                )
                                open_trades = []
                        else:
                            _exit_price, cash, margin_used = _exec_trade_exit(
                                trade,
                                exit_ref_price=float(bar.close),
                                exit_time=bar.ts,
                                reason=str(lifecycle.reason or "target_zero"),
                                apply_slippage=True,
                                exit_trace_payload={
                                    "stage": "open_resize_exit",
                                    "bar_ts": bar.ts.isoformat(),
                                    "exit_ref_price": float(bar.close),
                                    "apply_slippage": True,
                                    "resolved_exit_reason": str(
                                        lifecycle.reason or "target_zero"
                                    ),
                                    "signal_snapshot": _latest_signal_snapshot_probe(
                                        exec_idx=int(idx)
                                    ),
                                    "lifecycle": lifecycle.as_payload(),
                                },
                                exit_exec_idx=int(idx),
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
                exit_trace_payload={
                    "stage": "close_all",
                    "bar_ts": last_bar.ts.isoformat(),
                    "exit_ref_price": float(last_bar.close),
                    "apply_slippage": True,
                    "resolved_exit_reason": "end",
                    "signal_snapshot": _latest_signal_snapshot_probe(
                        exec_idx=int(len(exec_bars) - 1)
                    ),
                },
                exit_exec_idx=int(len(exec_bars) - 1),
            )

    summary = summarize_with_max_drawdown(
        trades,
        starting_cash=cfg.backtest.starting_cash,
        max_drawdown=float(equity_max_dd),
        multiplier=meta.multiplier,
    )
    trace_path = str(trace_out_raw or "").strip()
    if trace_path and lifecycle_rows is not None:
        write_rows_csv(rows=lifecycle_rows, out_path=trace_path)
    why_not_path = str(why_not_out_raw or "").strip()
    if why_not_path and lifecycle_rows is not None:
        write_rows_csv(
            rows=why_not_exit_resize_report(lifecycle_rows), out_path=why_not_path
        )
    return BacktestResult(
        trades=trades,
        equity=equity_curve,
        summary=summary,
        lifecycle_trace=lifecycle_rows if lifecycle_rows is not None else None,
    )


def _run_spot_backtest_exec_loop_summary(
    cfg: ConfigBundle,
    *,
    signal_bars: BarSeriesInput,
    exec_bars: BarSeriesInput,
    meta: ContractMeta,
    regime_bars: BarSeriesInput | None = None,
    regime2_bars: BarSeriesInput | None = None,
    regime2_bear_hard_bars: BarSeriesInput | None = None,
    tick_bars: BarSeriesInput | None = None,
    prepared_series_pack: object | None = None,
    progress_callback=None,
) -> SummaryStats:
    """Run summary-only evaluation through the canonical spot lifecycle."""
    signal_bars = _bars_input_list(signal_bars)
    exec_bars = _bars_input_list(exec_bars)
    regime_bars = _bars_input_optional_list(regime_bars)
    regime2_bars = _bars_input_optional_list(regime2_bars)
    regime2_bear_hard_bars = _bars_input_optional_list(regime2_bear_hard_bars)
    tick_bars = _bars_input_optional_list(tick_bars)
    _spot_emit_progress(
        progress_callback,
        phase="summary.path",
        path="canonical",
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
        regime2_bear_hard_bars=regime2_bear_hard_bars,
        tick_bars=tick_bars,
        capture_equity=False,
        prepared_series_pack=prepared_series_pack,
        progress_callback=progress_callback,
    ).summary


def _spot_score_start_dt(cfg: ConfigBundle) -> datetime:
    return datetime.combine(cfg.backtest.start, time(0, 0))


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
    held = bars_elapsed(trade.entry_time, bar.ts, bar_size=str(cfg.backtest.bar_size))
    if held > int(max_bars):
        return False

    try:
        slope_min = float(
            getattr(filters, "ratsv_probe_cancel_slope_adverse_min_pct", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        slope_min = 0.0
    if slope_min <= 0:
        return False

    if slope_med is None:
        return False
    adverse = (
        float(slope_med) <= -float(slope_min)
        if int(trade.qty) > 0
        else float(slope_med) >= float(slope_min)
    )
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
        slope_min = float(
            getattr(filters, "ratsv_adverse_release_slope_adverse_min_pct", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        slope_min = 0.0
    if slope_min <= 0:
        return False

    try:
        min_hold = int(getattr(filters, "ratsv_adverse_release_min_hold_bars", 0) or 0)
    except (TypeError, ValueError):
        min_hold = 0
    if min_hold > 0:
        held = bars_elapsed(
            trade.entry_time, bar.ts, bar_size=str(cfg.backtest.bar_size)
        )
        if held < int(min_hold):
            return False

    adverse = False
    if slope_med is not None:
        adverse = (
            float(slope_med) <= -float(slope_min)
            if int(trade.qty) > 0
            else float(slope_med) >= float(slope_min)
        )
    if (not adverse) and slope_vel is not None:
        adverse = (
            float(slope_vel) <= -float(slope_min)
            if int(trade.qty) > 0
            else float(slope_vel) >= float(slope_min)
        )
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
    if not flip_exit_allowed(
        strategy=cfg.strategy,
        open_dir=trade_dir,
        entry_time=trade.entry_time,
        current_time=bar.ts,
        bar_size=str(cfg.backtest.bar_size),
        signal=signal,
        tr_ratio=tr_ratio,
        shock_atr_vel_pct=shock_atr_vel_pct,
        tr_median_pct=tr_median_pct,
    ):
        return False

    if cfg.strategy.flip_exit_only_if_profit:
        pnl = bar.close - trade.entry_price
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


def _close_spot_trade(
    trade: SpotTrade, ts: datetime, price: float, reason: str, trades: list[SpotTrade]
) -> None:
    trade.exit_time = ts
    trade.exit_price = price
    trade.exit_reason = reason
    trades.append(trade)


# endregion


# region Spot Analysis Helpers
def _spot_local_extrema_probe(
    *,
    bars: list[Bar],
    exec_idx: int,
    ref_price: float,
    bar_size: str,
) -> dict[str, object] | None:
    if exec_idx < 0 or exec_idx >= len(bars):
        return None
    ref = float(ref_price)
    if not math.isfinite(ref) or ref <= 0.0:
        return None
    bar_def = parse_bar_size(bar_size)
    bar_hours = max(
        bar_def.duration.total_seconds() / 3600.0 if bar_def is not None else 1.0,
        1e-9,
    )
    out: dict[str, object] = {}
    for label, window_hours in (("15m", 0.25), ("1h", 1.0), ("6h30m", 6.5)):
        lookback_bars = max(1, int(math.ceil(float(window_hours) / float(bar_hours))))
        start_idx = max(0, int(exec_idx) - int(lookback_bars) + 1)
        window = bars[start_idx : int(exec_idx) + 1]
        if not window:
            continue
        low = min(float(b.low) for b in window)
        high = max(float(b.high) for b in window)
        span = max(float(high) - float(low), 1e-9)
        out[str(label)] = {
            "bars": int(len(window)),
            "minutes": int(round(float(window_hours) * 60.0)),
            "low": float(low),
            "high": float(high),
            "dist_from_low_pct": (
                (float(ref) - float(low)) / max(abs(float(low)), 1e-9)
            )
            * 100.0,
            "dist_from_high_pct": (
                (float(high) - float(ref)) / max(abs(float(high)), 1e-9)
            )
            * 100.0,
            "range_pos": max(0.0, min(1.0, (float(ref) - float(low)) / float(span))),
        }
    return out or None


# endregion
