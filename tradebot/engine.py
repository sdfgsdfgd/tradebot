"""Core decision logic shared across UI/live and backtests.

This module intentionally contains only:
- pure math / state machines (EMA, regime gating, debounce),
- small helper policies (filters, flip-exit hit detection),
- no IBKR calls, no async, no pricing models.

The goal is that both:
- `tradebot/ui/app.py` (live) and
- `tradebot/backtest/engine.py` (offline)
use the same entry/exit signal semantics.

This file was previously named `decision_core.py`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable

from .signals import (
    ema_cross,
    ema_next,
    ema_periods,
    ema_state_direction,
    normalize_ema_entry_mode,
    parse_bar_size,
    trend_confirmed_state,
    update_cross_confirm,
)
from .spot.policy import SpotDecisionTrace, SpotIntentDecision, SpotPolicy, SpotPolicyConfigView, SpotRuntimeSpec
from .time_utils import (
    ET_ZONE as _ET_ZONE,
    NaiveTsModeInput,
    normalize_naive_ts_mode as _normalize_naive_ts_mode_shared,
    to_et as _to_et_shared,
    trade_date as _trade_date_shared,
    trade_hour_et as _trade_hour_et_shared,
    trade_weekday as _trade_weekday_shared,
)

# region Time Helpers
def _normalize_naive_ts_mode(naive_ts_mode: NaiveTsModeInput) -> str:
    return _normalize_naive_ts_mode_shared(naive_ts_mode, default="utc")


def _ts_to_et_with_mode(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> datetime:
    return _to_et_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc")


def _ts_to_et(ts: datetime) -> datetime:
    """Interpret naive datetimes as UTC and return an ET-aware timestamp."""
    return _ts_to_et_with_mode(ts, naive_ts_mode="utc")


def _trade_date(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> date:
    return _trade_date_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc")


def _trade_hour_et(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> int:
    return _trade_hour_et_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc")


def _trade_weekday(ts: datetime, *, naive_ts_mode: NaiveTsModeInput = None) -> int:
    return _trade_weekday_shared(ts, naive_ts_mode=naive_ts_mode, default_naive_ts_mode="utc")


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
# endregion


# region Internal Parsing Helpers
def _filters_get(filters: Mapping[str, object] | object | None, key: str):
    if filters is None:
        return None
    if isinstance(filters, Mapping):
        return filters.get(key)
    return getattr(filters, key, None)


def _parse_int(value: object, *, default: int, min_value: int | None = None) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    if min_value is not None and parsed < int(min_value):
        return int(min_value)
    return int(parsed)


def _parse_float(value: object, *, default: float, min_value: float | None = None) -> float:
    try:
        parsed = float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        parsed = float(default)
    if min_value is not None and parsed < float(min_value):
        return float(min_value)
    return float(parsed)


# endregion


# region Spot Strategy Parsing
def normalize_spot_entry_signal(entry_signal_raw: object | None) -> str:
    entry_signal = str(entry_signal_raw or "ema").strip().lower()
    if entry_signal not in ("ema", "orb"):
        return "ema"
    return entry_signal


def normalize_spot_regime_mode(regime_mode_raw: object | None) -> str:
    regime_mode = str(regime_mode_raw or "ema").strip().lower()
    if regime_mode not in ("ema", "supertrend"):
        return "ema"
    return regime_mode


def normalize_spot_regime2_mode(regime2_mode_raw: object | None) -> str:
    regime2_mode = str(regime2_mode_raw or "off").strip().lower()
    if regime2_mode not in ("off", "ema", "supertrend"):
        return "off"
    return regime2_mode


def resolve_spot_regime_spec(
    *,
    bar_size: object,
    regime_mode_raw: object | None,
    regime_ema_preset_raw: object | None,
    regime_bar_size_raw: object | None,
) -> tuple[str, str | None, str, bool]:
    base_bar_size = str(bar_size)
    regime_mode = normalize_spot_regime_mode(regime_mode_raw)
    regime_preset = str(regime_ema_preset_raw or "").strip() or None
    regime_bar_size = str(regime_bar_size_raw or "").strip()
    if not regime_bar_size or regime_bar_size.lower() in ("same", "default"):
        regime_bar_size = base_bar_size
    if regime_mode == "supertrend":
        use_mtf = str(regime_bar_size) != base_bar_size
    else:
        use_mtf = bool(regime_preset) and str(regime_bar_size) != base_bar_size
    return regime_mode, regime_preset, regime_bar_size, use_mtf


def resolve_spot_regime2_spec(
    *,
    bar_size: object,
    regime2_mode_raw: object | None,
    regime2_ema_preset_raw: object | None,
    regime2_bar_size_raw: object | None,
) -> tuple[str, str | None, str, bool]:
    base_bar_size = str(bar_size)
    regime2_mode = normalize_spot_regime2_mode(regime2_mode_raw)
    regime2_preset = str(regime2_ema_preset_raw or "").strip() or None
    if regime2_mode == "ema" and not regime2_preset:
        regime2_mode = "off"
    regime2_bar_size = str(regime2_bar_size_raw or "").strip()
    if not regime2_bar_size or regime2_bar_size.lower() in ("same", "default"):
        regime2_bar_size = base_bar_size
    if regime2_mode == "supertrend":
        use_mtf = str(regime2_bar_size) != base_bar_size
    else:
        use_mtf = bool(regime2_preset) and str(regime2_bar_size) != base_bar_size
    return regime2_mode, regime2_preset, regime2_bar_size, use_mtf


def spot_regime_apply_matches_direction(*, apply_to_raw: object | None, entry_dir: str | None) -> bool:
    apply_to = str(apply_to_raw or "both").strip().lower()
    if apply_to == "longs":
        return str(entry_dir) == "up"
    if apply_to == "shorts":
        return str(entry_dir) == "down"
    return True


# endregion


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


if TYPE_CHECKING:
    ShockEngine = AtrRatioShockEngine | TrRatioShockEngine | DailyAtrPctShockEngine | DailyDrawdownShockEngine
else:
    ShockEngine = object


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


# region Volatility Helpers
@lru_cache(maxsize=None)
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


# endregion


# region Cooldown Helpers
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


# endregion


# region Spot Execution + Exit Semantics
def spot_exec_price(
    ref_price: float,
    *,
    side: str,  # "buy" | "sell"
    qty: int,
    spread: float,
    commission_per_share: float,
    commission_min: float,
    slippage_per_share: float,
    apply_slippage: bool = True,
) -> float:
    """Model a spot execution price from a reference price plus costs.

    Used by the backtest spot executor (and intended for eventual live paper/live parity checks).
    """
    price = float(ref_price)
    half = max(0.0, float(spread)) / 2.0
    abs_qty = max(1, abs(int(qty)))
    comm = max(0.0, float(commission_per_share))
    comm_min = max(0.0, float(commission_min))
    comm_eff = max(comm, (comm_min / float(abs_qty)) if abs_qty > 0 else comm)
    slip = max(0.0, float(slippage_per_share)) if apply_slippage else 0.0
    if str(side).strip().lower() == "buy":
        return max(0.0, price + half + comm_eff + slip)
    return max(0.0, price - half - comm_eff - slip)


def spot_mark_price(ref_price: float, *, qty: int, spread: float, mode: str) -> float:
    """Model mark-to-market price under a configurable realism mode.

    - mode="close": mark at close.
    - mode="liquidation": mark as if you had to cross the spread to exit.
    """
    price = float(ref_price)
    if str(mode).strip().lower() != "liquidation":
        return max(0.0, price)
    half = max(0.0, float(spread)) / 2.0
    if int(qty) > 0:
        return max(0.0, price - half)
    if int(qty) < 0:
        return max(0.0, price + half)
    return max(0.0, price)


def spot_profit_level(
    entry_price: float,
    qty: int,
    profit_target_price: float | None = None,
    profit_target_pct: float | None = None,
) -> float | None:
    if profit_target_price is not None:
        target = float(profit_target_price)
        return target if target > 0 else None
    if profit_target_pct is None:
        return None
    entry = float(entry_price)
    if entry <= 0:
        return None
    pct = float(profit_target_pct)
    if pct <= 0:
        return None
    if int(qty) > 0:
        return entry * (1.0 + pct)
    if int(qty) < 0:
        return entry * (1.0 - pct)
    return None


def spot_stop_level(
    entry_price: float,
    qty: int,
    stop_loss_price: float | None = None,
    stop_loss_pct: float | None = None,
) -> float | None:
    if stop_loss_price is not None:
        stop = float(stop_loss_price)
        return stop if stop > 0 else None
    if stop_loss_pct is None:
        return None
    entry = float(entry_price)
    if entry <= 0:
        return None
    pct = float(stop_loss_pct)
    if pct <= 0:
        return None
    if int(qty) > 0:
        return entry * (1.0 - pct)
    if int(qty) < 0:
        return entry * (1.0 + pct)
    return None


def spot_shock_exit_pct_multipliers(
    filters: Mapping[str, object] | object | None,
    *,
    shock: bool | None,
) -> tuple[float, float]:
    """Return sanitized stop/profit multipliers for shock-aware pct exits."""
    return SpotPolicy.shock_exit_pct_multipliers(filters, shock=shock)


def spot_scale_exit_pcts(
    *,
    stop_loss_pct: float | None,
    profit_target_pct: float | None,
    stop_mult: float = 1.0,
    profit_mult: float = 1.0,
) -> tuple[float | None, float | None]:
    """Scale pct-based stop/profit levels with safe bounds and invalid-value handling."""
    return SpotPolicy.scale_exit_pcts(
        stop_loss_pct=stop_loss_pct,
        profit_target_pct=profit_target_pct,
        stop_mult=stop_mult,
        profit_mult=profit_mult,
    )


def spot_riskoff_mode_from_filters(filters: Mapping[str, object] | object | None) -> str:
    """Normalize riskoff mode from filters."""
    return SpotPolicy.riskoff_mode(filters)


def spot_riskoff_end_hour(filters: Mapping[str, object] | object | None) -> int | None:
    """Resolve ET risk cutoff hour with legacy fallback support."""
    return SpotPolicy.risk_entry_cutoff_hour_et(filters)


def spot_policy_config_view(
    *,
    strategy: Mapping[str, object] | object | None = None,
    filters: Mapping[str, object] | object | None = None,
) -> SpotPolicyConfigView:
    """Return the sanitized/defaulted spot policy config view."""
    return SpotPolicy.policy_config(strategy=strategy, filters=filters)


def spot_runtime_spec_view(
    *,
    strategy: Mapping[str, object] | object | None = None,
    filters: Mapping[str, object] | object | None = None,
) -> SpotRuntimeSpec:
    """Return sanitized/defaulted spot execution runtime knobs."""
    return SpotPolicy.runtime_spec(strategy=strategy, filters=filters)


def spot_resolve_entry_action_qty(
    *,
    strategy: Mapping[str, object] | object,
    entry_dir: str | None,
    needs_direction: bool = False,
    fallback_short_sell: bool = False,
) -> tuple[str, int] | None:
    return SpotPolicy.resolve_entry_action_qty(
        strategy=strategy,
        entry_dir=entry_dir,
        needs_direction=needs_direction,
        fallback_short_sell=fallback_short_sell,
    )


def spot_pending_entry_should_cancel(
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
    naive_ts_mode: str | None = None,
) -> bool:
    return SpotPolicy.pending_entry_should_cancel(
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
        naive_ts_mode=naive_ts_mode,
    )


def spot_branch_size_mult(
    *,
    strategy: Mapping[str, object] | object,
    entry_branch: str | None,
) -> float:
    return SpotPolicy.branch_size_mult(strategy=strategy, entry_branch=entry_branch)


def spot_apply_branch_size_mult(
    *,
    signed_qty: int,
    size_mult: float,
    spot_min_qty: object,
    spot_max_qty: object,
) -> int:
    return SpotPolicy.apply_branch_size_mult(
        signed_qty=signed_qty,
        size_mult=size_mult,
        spot_min_qty=spot_min_qty,
        spot_max_qty=spot_max_qty,
    )


def spot_calc_signed_qty(
    *,
    strategy: Mapping[str, object] | object,
    filters: Mapping[str, object] | object | None,
    action: str,
    lot: int,
    entry_price: float,
    stop_price: float | None,
    stop_loss_pct: float | None,
    shock: bool | None,
    shock_dir: str | None,
    shock_atr_pct: float | None,
    riskoff: bool = False,
    risk_dir: str | None = None,
    riskpanic: bool = False,
    riskpop: bool = False,
    risk: RiskOverlaySnapshot | None = None,
    equity_ref: float,
    cash_ref: float | None,
) -> int:
    """Return signed share qty for a spot entry."""
    return SpotPolicy.calc_signed_qty(
        strategy=strategy,
        filters=filters,
        action=action,
        lot=lot,
        entry_price=entry_price,
        stop_price=stop_price,
        stop_loss_pct=stop_loss_pct,
        shock=shock,
        shock_dir=shock_dir,
        shock_atr_pct=shock_atr_pct,
        riskoff=riskoff,
        risk_dir=risk_dir,
        riskpanic=riskpanic,
        riskpop=riskpop,
        risk=risk,
        equity_ref=equity_ref,
        cash_ref=cash_ref,
    )


def spot_calc_signed_qty_with_trace(
    *,
    strategy: Mapping[str, object] | object,
    filters: Mapping[str, object] | object | None,
    action: str,
    lot: int,
    entry_price: float,
    stop_price: float | None,
    stop_loss_pct: float | None,
    shock: bool | None,
    shock_dir: str | None,
    shock_atr_pct: float | None,
    riskoff: bool = False,
    risk_dir: str | None = None,
    riskpanic: bool = False,
    riskpop: bool = False,
    risk: RiskOverlaySnapshot | None = None,
    equity_ref: float = 0.0,
    cash_ref: float | None = None,
) -> tuple[int, SpotDecisionTrace]:
    """Return signed share qty plus a typed spot decision trace."""
    return SpotPolicy.calc_signed_qty_with_trace(
        strategy=strategy,
        filters=filters,
        action=action,
        lot=lot,
        entry_price=entry_price,
        stop_price=stop_price,
        stop_loss_pct=stop_loss_pct,
        shock=shock,
        shock_dir=shock_dir,
        shock_atr_pct=shock_atr_pct,
        riskoff=riskoff,
        risk_dir=risk_dir,
        riskpanic=riskpanic,
        riskpop=riskpop,
        risk=risk,
        equity_ref=equity_ref,
        cash_ref=cash_ref,
    )


def spot_resolve_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    target_qty: int,
) -> SpotIntentDecision:
    """Resolve order intent needed to move from current_qty to target_qty."""
    return SpotPolicy.resolve_position_intent(
        strategy=strategy,
        current_qty=current_qty,
        target_qty=target_qty,
    )


def spot_hit_profit(
    entry_price: float,
    qty: int,
    price: float,
    profit_target_price: float | None = None,
    profit_target_pct: float | None = None,
) -> bool:
    if profit_target_price is not None:
        target = float(profit_target_price)
        if target <= 0:
            return False
        if int(qty) > 0:
            return float(price) >= target
        if int(qty) < 0:
            return float(price) <= target
        return False

    if profit_target_pct is None:
        return False
    entry = float(entry_price)
    if entry <= 0:
        return False
    move = (float(price) - entry) / entry
    if int(qty) < 0:
        move = -move
    return move >= float(profit_target_pct)


def spot_hit_stop(
    entry_price: float,
    qty: int,
    price: float,
    stop_loss_price: float | None = None,
    stop_loss_pct: float | None = None,
) -> bool:
    if stop_loss_price is not None:
        stop = float(stop_loss_price)
        if stop <= 0:
            return False
        if int(qty) > 0:
            return float(price) <= stop
        if int(qty) < 0:
            return float(price) >= stop
        return False

    if stop_loss_pct is None:
        return False
    entry = float(entry_price)
    if entry <= 0:
        return False
    move = (float(price) - entry) / entry
    if int(qty) < 0:
        move = -move
    return move <= -float(stop_loss_pct)


def spot_intrabar_exit(
    *,
    qty: int,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    stop_level: float | None,
    profit_level: float | None,
) -> tuple[str, float] | None:
    """Return (reason, exit_ref_price) for intrabar stop/profit fills."""
    if stop_level is not None:
        stop = float(stop_level)
        if int(qty) > 0 and float(bar_low) <= stop:
            ref = float(bar_open)
            return ("stop", ref if ref <= stop else stop)
        if int(qty) < 0 and float(bar_high) >= stop:
            ref = float(bar_open)
            return ("stop", ref if ref >= stop else stop)
    if profit_level is not None:
        target = float(profit_level)
        if int(qty) > 0 and float(bar_high) >= target:
            return ("profit", target)
        if int(qty) < 0 and float(bar_low) <= target:
            return ("profit", target)
    return None


def spot_intrabar_worst_ref(
    *,
    qty: int,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    stop_level: float | None,
) -> float:
    """Worst-case reference price within a bar for drawdown realism."""
    if int(qty) > 0:
        worst = float(bar_low)
        if stop_level is not None and float(bar_low) <= float(stop_level):
            stop = float(stop_level)
            ref = float(bar_open)
            worst = ref if ref <= stop else stop
        return worst
    if int(qty) < 0:
        worst = float(bar_high)
        if stop_level is not None and float(bar_high) >= float(stop_level):
            stop = float(stop_level)
            ref = float(bar_open)
            worst = ref if ref >= stop else stop
        return worst
    return float(bar_low)


# endregion


# region Decision Engines (EMA / ORB)
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


# endregion


# region Supertrend
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


# endregion


# region Shock Engines
class _ShockDirectionMixin:
    _dir_lookback: int
    _dir_prev_close: float | None
    _ret_hist: deque[float]
    _direction: str | None

    def _init_direction_state(self, direction_lookback: int) -> None:
        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close = None
        self._ret_hist = deque(maxlen=self._dir_lookback)
        self._direction = None

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

    def _direction_state(self) -> tuple[str | None, bool]:
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
        return direction, direction_ready


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
        direction, direction_ready = self._direction_state()
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
class TrRatioShockSnapshot:
    shock: bool
    ready: bool
    ratio: float | None = None
    tr_fast_pct: float | None = None
    tr_fast: float | None = None
    tr_slow: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False


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
        direction, direction_ready = self._direction_state()
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
        direction, direction_ready = self._direction_state()
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
        direction, direction_ready = self._direction_state()
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


# endregion


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
