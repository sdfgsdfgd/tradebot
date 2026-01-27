"""Core decision logic shared across UI/live and backtests.

This module intentionally contains only:
- pure math / state machines (EMA, regime gating, debounce),
- small helper policies (filters, flip-exit hit detection),
- no IBKR calls, no async, no pricing models.

The goal is that both:
- `tradebot/ui/app.py` (live) and
- `tradebot/backtest/engine.py` (offline)
use the same entry/exit signal semantics.

This file was previously named `decision_core.py`. The old import path remains as a
shim (`tradebot.decision_core`) for compatibility.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Iterable, Mapping
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

# region Constants
_ET_ZONE = ZoneInfo("America/New_York")
# endregion


# region Time Helpers
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


# region Permission Gates
def permission_gate_status(
    filters: Mapping[str, object] | object | None,
    *,
    close: float,
    signal: EmaDecisionSnapshot | None,
    entry_dir: str | None,
) -> tuple[bool, bool]:
    """Return (active, ok) for the EMA permission gates.

    - active: any permission threshold is configured (spread and/or slope).
    - ok: True when either inactive OR the thresholds pass for the given direction.
    """
    spread_min = _filters_get(filters, "ema_spread_min_pct")
    spread_min_down = _filters_get(filters, "ema_spread_min_pct_down")
    if entry_dir == "down" and spread_min_down is not None:
        spread_min = spread_min_down
    slope_min = _filters_get(filters, "ema_slope_min_pct")

    active = spread_min is not None or slope_min is not None
    if not bool(active):
        return False, True

    if signal is None or not bool(signal.ema_ready) or signal.ema_fast is None or signal.ema_slow is None:
        return True, False

    if spread_min is not None:
        try:
            spread_min_f = float(spread_min)
        except (TypeError, ValueError):
            spread_min_f = None
        if spread_min_f is not None:
            spread = ema_spread_pct(float(signal.ema_fast), float(signal.ema_slow), float(close))
            if spread < spread_min_f:
                return True, False

    if slope_min is not None:
        try:
            slope_min_f = float(slope_min)
        except (TypeError, ValueError):
            slope_min_f = None
        if slope_min_f is not None:
            if signal.prev_ema_fast is None:
                return True, False
            slope = ema_slope_pct(float(signal.ema_fast), float(signal.prev_ema_fast), float(close))
            if slope < slope_min_f:
                return True, False

    return True, True


# endregion


# region Volatility Helpers
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
    riskoff: bool,
    risk_dir: str | None,
    riskpanic: bool,
    equity_ref: float,
    cash_ref: float | None,
) -> int:
    """Return signed share qty for a spot entry.

    This mirrors the sizing logic used by the synthetic backtest spot executor, so the live UI can
    apply the same sizing modes (fixed / notional_pct / risk_pct) using real account equity.
    """

    def _get(key: str, default: object = None):
        if isinstance(strategy, Mapping):
            return strategy.get(key, default)
        return getattr(strategy, key, default)

    lot = max(1, int(lot or 1))
    quantity_mult = max(1, int(_get("quantity", 1) or 1))
    raw_action = str(action or "BUY").strip().upper()
    if raw_action not in ("BUY", "SELL"):
        raw_action = "BUY"

    sizing_mode = str(_get("spot_sizing_mode", "fixed") or "fixed").strip().lower()
    if sizing_mode not in ("fixed", "notional_pct", "risk_pct"):
        sizing_mode = "fixed"
    spot_notional_pct = max(0.0, float(_get("spot_notional_pct", 0.0) or 0.0))
    spot_risk_pct = max(0.0, float(_get("spot_risk_pct", 0.0) or 0.0))
    spot_short_risk_mult = max(0.0, float(_get("spot_short_risk_mult", 1.0) or 1.0))
    spot_max_notional_pct = max(0.0, float(_get("spot_max_notional_pct", 1.0) or 1.0))
    spot_min_qty = max(1, int(_get("spot_min_qty", 1) or 1))
    spot_max_qty = max(0, int(_get("spot_max_qty", 0) or 0))

    if sizing_mode == "fixed":
        base_qty = lot * quantity_mult
        return int(base_qty) if raw_action == "BUY" else -int(base_qty)

    if float(entry_price) <= 0:
        return 0

    desired_qty = 0
    if sizing_mode == "notional_pct":
        if spot_notional_pct > 0 and float(equity_ref) > 0:
            desired_qty = int((float(equity_ref) * float(spot_notional_pct)) / float(entry_price))
    else:
        stop_level = None
        if stop_price is not None and float(stop_price) > 0:
            stop_level = float(stop_price)
        elif stop_loss_pct is not None and float(stop_loss_pct) > 0:
            stop_level = spot_stop_level(
                float(entry_price),
                qty=1 if raw_action == "BUY" else -1,
                stop_loss_pct=float(stop_loss_pct),
            )
        if stop_level is not None and spot_risk_pct > 0 and float(equity_ref) > 0:
            per_share_risk = abs(float(entry_price) - float(stop_level))
            risk_dollars = float(equity_ref) * float(spot_risk_pct)

            riskoff_mode, riskoff_long_factor, riskoff_short_factor, riskpanic_short_factor = (
                risk_overlay_policy_from_filters(filters)
            )

            if raw_action == "BUY":
                if bool(riskoff) and riskoff_mode == "directional" and str(risk_dir) == "up":
                    risk_dollars *= float(riskoff_long_factor)
                if bool(shock) and shock_dir in ("up", "down"):
                    if shock_dir == "up":
                        shock_long_mult = float(_filters_get(filters, "shock_long_risk_mult_factor") or 1.0)
                    else:
                        shock_long_mult = float(_filters_get(filters, "shock_long_risk_mult_factor_down") or 1.0)
                    if shock_long_mult < 0:
                        shock_long_mult = 1.0
                    if shock_long_mult == 0:
                        return 0
                    risk_dollars *= float(shock_long_mult)
            else:
                short_mult = float(spot_short_risk_mult)
                if bool(riskoff) and riskoff_mode == "directional" and str(risk_dir) == "down":
                    short_mult *= float(riskoff_short_factor)
                if bool(riskpanic) and str(risk_dir) == "down":
                    short_mult *= float(riskpanic_short_factor)
                if bool(shock) and str(shock_dir) == "down":
                    shock_short_mult = float(_filters_get(filters, "shock_short_risk_mult_factor") or 1.0)
                    if shock_short_mult < 0:
                        shock_short_mult = 1.0
                    short_mult *= float(shock_short_mult)
                risk_dollars *= float(short_mult)

            target_atr = _filters_get(filters, "shock_risk_scale_target_atr_pct")
            if (
                target_atr is not None
                and shock_atr_pct is not None
                and float(shock_atr_pct) > 0
            ):
                try:
                    target = float(target_atr)
                except (TypeError, ValueError):
                    target = 0.0
                if target > 0:
                    min_mult_raw = _filters_get(filters, "shock_risk_scale_min_mult") or 0.2
                    try:
                        min_mult = float(min_mult_raw)
                    except (TypeError, ValueError):
                        min_mult = 0.2
                    min_mult = float(max(0.0, min(1.0, min_mult)))
                    scale = min(1.0, float(target) / float(shock_atr_pct))
                    scale = float(max(min_mult, min(1.0, scale)))
                    risk_dollars *= float(scale)

            if per_share_risk > 1e-9 and risk_dollars > 0:
                desired_qty = int(risk_dollars / per_share_risk)

    if desired_qty <= 0:
        desired_qty = lot * quantity_mult

    if spot_max_notional_pct > 0 and float(equity_ref) > 0:
        cap_qty = int((float(equity_ref) * float(spot_max_notional_pct)) / float(entry_price))
        desired_qty = min(int(desired_qty), max(0, int(cap_qty)))

    if raw_action == "BUY" and cash_ref is not None and float(cash_ref) > 0:
        afford_qty = int(float(cash_ref) / float(entry_price))
        desired_qty = min(int(desired_qty), max(0, int(afford_qty)))

    if spot_max_qty > 0:
        desired_qty = min(int(desired_qty), int(spot_max_qty))

    desired_qty = (int(desired_qty) // int(lot)) * int(lot)
    min_effective = max(int(spot_min_qty), int(lot))
    if desired_qty < min_effective:
        return 0
    return int(desired_qty) if raw_action == "BUY" else -int(desired_qty)


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
class TrRatioShockSnapshot:
    shock: bool
    ready: bool
    ratio: float | None = None
    tr_fast_pct: float | None = None
    tr_fast: float | None = None
    tr_slow: float | None = None
    direction: str | None = None  # "up" | "down"
    direction_ready: bool = False


class TrRatioShockEngine:
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

        self._dir_lookback = max(1, int(direction_lookback))
        self._dir_prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=self._dir_lookback)
        self._direction: str | None = None

        self._tr_prev_close: float | None = None
        self._tr_fast_ema: float | None = None
        self._tr_slow_ema: float | None = None
        self._count = 0

        self._ready = False
        self._last_ratio: float | None = None
        self._last_tr_fast_pct: float | None = None
        self._last_tr_fast: float | None = None
        self._last_tr_slow: float | None = None

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
        direction = str(self._direction) if self._direction in ("up", "down") else None
        direction_ready = bool(direction in ("up", "down") and len(self._ret_hist) >= self._dir_lookback)
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


# endregion


# region Risk Overlays (TR%)
@dataclass(frozen=True)
class RiskOverlaySnapshot:
    riskoff: bool
    riskpanic: bool
    tr_median_pct: float | None = None
    neg_gap_ratio: float | None = None


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
        riskpanic_lookback_days: int,
    ) -> None:
        self._riskoff_tr_med_pct = float(riskoff_tr_med_pct) if riskoff_tr_med_pct is not None else None
        self._riskoff_lookback = max(1, int(riskoff_lookback_days))
        self._riskpanic_tr_med_pct = float(riskpanic_tr_med_pct) if riskpanic_tr_med_pct is not None else None
        self._riskpanic_neg_gap_ratio_min = (
            float(riskpanic_neg_gap_ratio_min) if riskpanic_neg_gap_ratio_min is not None else None
        )
        if self._riskpanic_neg_gap_ratio_min is not None:
            self._riskpanic_neg_gap_ratio_min = float(max(0.0, min(1.0, self._riskpanic_neg_gap_ratio_min)))
        self._riskpanic_lookback = max(1, int(riskpanic_lookback_days))

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

        self._prev_close: float | None = None
        self._cur_day: date | None = None
        self._day_open: float | None = None
        self._day_high: float | None = None
        self._day_low: float | None = None

        self._riskoff_today = False
        self._riskpanic_today = False
        self._tr_median_pct: float | None = None
        self._neg_gap_ratio: float | None = None

    @staticmethod
    def _day_true_range(high: float, low: float, prev_close: float) -> float:
        return max(
            max(0.0, float(high) - float(low)),
            abs(float(high) - float(prev_close)),
            abs(float(low) - float(prev_close)),
        )

    def _compute_today_flags(self) -> None:
        self._tr_median_pct = None
        self._neg_gap_ratio = None

        self._riskoff_today = False
        if self._riskoff_tr_hist is not None and len(self._riskoff_tr_hist) >= self._riskoff_lookback:
            tr_vals = sorted(self._riskoff_tr_hist)
            self._tr_median_pct = float(tr_vals[len(tr_vals) // 2])
            self._riskoff_today = bool(self._tr_median_pct >= float(self._riskoff_tr_med_pct))

        self._riskpanic_today = False
        if (
            self._riskpanic_tr_hist is not None
            and self._riskpanic_neg_gap_hist is not None
            and self._riskpanic_tr_med_pct is not None
            and self._riskpanic_neg_gap_ratio_min is not None
            and len(self._riskpanic_tr_hist) >= self._riskpanic_lookback
            and len(self._riskpanic_neg_gap_hist) >= self._riskpanic_lookback
        ):
            tr_vals = sorted(self._riskpanic_tr_hist)
            tr_med = float(tr_vals[len(tr_vals) // 2])
            neg_ratio = float(sum(self._riskpanic_neg_gap_hist)) / float(len(self._riskpanic_neg_gap_hist))
            self._neg_gap_ratio = float(neg_ratio)
            self._riskpanic_today = bool(
                tr_med >= float(self._riskpanic_tr_med_pct) and neg_ratio >= float(self._riskpanic_neg_gap_ratio_min)
            )

    def update(
        self,
        *,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        is_last_bar: bool,
    ) -> RiskOverlaySnapshot:
        if self._riskoff_tr_hist is None and self._riskpanic_tr_hist is None:
            return RiskOverlaySnapshot(riskoff=False, riskpanic=False)

        day = ts.date()
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
                self._riskpanic_neg_gap_hist.append(1 if float(gap_pct) < 0 else 0)

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
            self._prev_close = float(close)

        return RiskOverlaySnapshot(
            riskoff=bool(self._riskoff_today),
            riskpanic=bool(self._riskpanic_today),
            tr_median_pct=float(self._tr_median_pct) if self._tr_median_pct is not None else None,
            neg_gap_ratio=float(self._neg_gap_ratio) if self._neg_gap_ratio is not None else None,
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

    riskpanic_lb = _parse_int(_filters_get(filters, "riskpanic_lookback_days"), default=5, min_value=1)

    enabled = bool(riskoff_tr_med is not None) or bool(riskpanic_tr_med is not None and riskpanic_gap_ratio is not None)
    if not enabled:
        return None

    return TrPctRiskOverlayEngine(
        riskoff_tr_med_pct=riskoff_tr_med,
        riskoff_lookback_days=int(riskoff_lb),
        riskpanic_tr_med_pct=riskpanic_tr_med,
        riskpanic_neg_gap_ratio_min=riskpanic_gap_ratio,
        riskpanic_lookback_days=int(riskpanic_lb),
    )


def risk_overlay_policy_from_filters(filters: Mapping[str, object] | object | None) -> tuple[str, float, float, float]:
    """Return (riskoff_mode, riskoff_long_factor, riskoff_short_factor, riskpanic_short_factor)."""
    mode = str(_filters_get(filters, "riskoff_mode") or "hygiene").strip().lower()
    if mode not in ("hygiene", "directional"):
        mode = "hygiene"

    riskoff_short = _parse_float(_filters_get(filters, "riskoff_short_risk_mult_factor"), default=1.0)
    if riskoff_short < 0:
        riskoff_short = 1.0
    riskoff_long = _parse_float(_filters_get(filters, "riskoff_long_risk_mult_factor"), default=1.0)
    if riskoff_long < 0:
        riskoff_long = 1.0
    riskpanic_short = _parse_float(_filters_get(filters, "riskpanic_short_risk_mult_factor"), default=1.0)
    if riskpanic_short < 0:
        riskpanic_short = 1.0

    return str(mode), float(riskoff_long), float(riskoff_short), float(riskpanic_short)


# endregion


# region Regime / Flip Exit / Filters
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
        return _filters_get(filters, key)

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

    shock_mode = normalize_shock_gate_mode(filters)
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

    entry_dir = signal.entry_dir if signal is not None else None
    _, perm_ok = permission_gate_status(filters, close=float(close), signal=signal, entry_dir=entry_dir)
    if not bool(perm_ok):
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


# endregion
