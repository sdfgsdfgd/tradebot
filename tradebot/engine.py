"""Shared market math and spot execution policy for live and backtests.

Signal, shock, and risk state machines live under `tradebot.engines`. This
kernel remains deterministic: no broker calls, async work, or pricing models.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable

from .signals import bar_sizes_equal, parse_bar_size
from .spot.policy import SpotPolicy
from .spot.policy_contract import SpotDecisionTrace, SpotIntentDecision, SpotPolicyConfigView, SpotRuntimeSpec, SpotSizingInput
from .time_utils import (
    NaiveTsModeInput,
    normalize_naive_ts_mode as _normalize_naive_ts_mode_shared,
    to_et as _to_et_shared,
    trade_date as _trade_date_shared,
    trade_hour_et as _trade_hour_et_shared,
    trade_weekday as _trade_weekday_shared,
)

if TYPE_CHECKING:
    from .engines.risk import RiskOverlaySnapshot
    from .spot.graph import SpotPolicyGraph


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
        use_mtf = not bar_sizes_equal(regime_bar_size, base_bar_size)
    else:
        use_mtf = bool(regime_preset) and not bar_sizes_equal(regime_bar_size, base_bar_size)
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
        use_mtf = not bar_sizes_equal(regime2_bar_size, base_bar_size)
    else:
        use_mtf = bool(regime2_preset) and not bar_sizes_equal(regime2_bar_size, base_bar_size)
    return regime2_mode, regime2_preset, regime2_bar_size, use_mtf


def spot_regime_apply_matches_direction(*, apply_to_raw: object | None, entry_dir: str | None) -> bool:
    apply_to = str(apply_to_raw or "both").strip().lower()
    if apply_to in ("off", "none", "disabled", "false", "0", "soft"):
        return False
    if apply_to == "longs":
        return str(entry_dir) == "up"
    if apply_to == "shorts":
        return str(entry_dir) == "down"
    return True


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
    return SpotPolicyConfigView.from_sources(filters=filters).risk_entry_cutoff_hour_et


def spot_policy_config_view(
    *,
    strategy: Mapping[str, object] | object | None = None,
    filters: Mapping[str, object] | object | None = None,
) -> SpotPolicyConfigView:
    """Return the sanitized/defaulted spot policy config view."""
    return SpotPolicyConfigView.from_sources(strategy=strategy, filters=filters)


def spot_runtime_spec_view(
    *,
    strategy: Mapping[str, object] | object | None = None,
    filters: Mapping[str, object] | object | None = None,
) -> SpotRuntimeSpec:
    """Return sanitized/defaulted spot execution runtime knobs."""
    return SpotRuntimeSpec.from_sources(strategy=strategy, filters=filters)


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
    shock_dir_down_streak_bars: int | None = None,
    shock_drawdown_dist_on_pct: float | None = None,
    shock_drawdown_dist_on_vel_pp: float | None = None,
    shock_drawdown_dist_on_accel_pp: float | None = None,
    shock_prearm_down_streak_bars: int | None = None,
    shock_ramp: Mapping[str, object] | object | None = None,
    riskoff: bool = False,
    risk_dir: str | None = None,
    riskpanic: bool = False,
    riskpop: bool = False,
    risk: RiskOverlaySnapshot | None = None,
    signal_entry_dir: str | None = None,
    signal_regime_dir: str | None = None,
    regime2_dir: str | None = None,
    regime2_ready: bool = False,
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
        shock_dir_down_streak_bars=shock_dir_down_streak_bars,
        shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct,
        shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
        shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp,
        shock_prearm_down_streak_bars=shock_prearm_down_streak_bars,
        shock_ramp=shock_ramp,
        riskoff=riskoff,
        risk_dir=risk_dir,
        riskpanic=riskpanic,
        riskpop=riskpop,
        risk=risk,
        signal_entry_dir=signal_entry_dir,
        signal_regime_dir=signal_regime_dir,
        regime2_dir=regime2_dir,
        regime2_ready=bool(regime2_ready),
        equity_ref=equity_ref,
        cash_ref=cash_ref,
    )


def spot_sizing_input(
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
    shock_dir_down_streak_bars: int | None = None,
    shock_drawdown_dist_on_pct: float | None = None,
    shock_drawdown_dist_on_vel_pp: float | None = None,
    shock_drawdown_dist_on_accel_pp: float | None = None,
    shock_prearm_down_streak_bars: int | None = None,
    shock_ramp: Mapping[str, object] | object | None = None,
    riskoff: bool = False,
    risk_dir: str | None = None,
    riskpanic: bool = False,
    riskpop: bool = False,
    risk: RiskOverlaySnapshot | None = None,
    signal_entry_dir: str | None = None,
    signal_regime_dir: str | None = None,
    regime2_dir: str | None = None,
    regime2_ready: bool = False,
    equity_ref: float = 0.0,
    cash_ref: float | None = None,
    policy_graph: SpotPolicyGraph | None = None,
    policy_config: SpotPolicyConfigView | None = None,
) -> SpotSizingInput:
    """Build the canonical typed input consumed by spot sizing."""
    if policy_graph is None:
        from .spot.graph import SpotPolicyGraph

        policy_graph = SpotPolicyGraph.from_sources(
            strategy=strategy,
            filters=filters,
        )
    if policy_config is None:
        policy_config = SpotPolicyConfigView.from_sources(
            strategy=strategy,
            filters=filters,
        )
    return SpotSizingInput(
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
        shock_dir_down_streak_bars=shock_dir_down_streak_bars,
        shock_drawdown_dist_on_pct=shock_drawdown_dist_on_pct,
        shock_drawdown_dist_on_vel_pp=shock_drawdown_dist_on_vel_pp,
        shock_drawdown_dist_on_accel_pp=shock_drawdown_dist_on_accel_pp,
        shock_prearm_down_streak_bars=shock_prearm_down_streak_bars,
        shock_ramp=shock_ramp,
        riskoff=riskoff,
        risk_dir=risk_dir,
        riskpanic=riskpanic,
        riskpop=riskpop,
        risk=risk,
        signal_entry_dir=signal_entry_dir,
        signal_regime_dir=signal_regime_dir,
        regime2_dir=regime2_dir,
        regime2_ready=bool(regime2_ready),
        equity_ref=equity_ref,
        cash_ref=cash_ref,
        policy_graph=policy_graph,
        policy_config=policy_config,
    )


def spot_calc_signed_qty_with_trace(
    sizing_input: SpotSizingInput | None = None,
    *,
    capture_trace: bool = True,
    **legacy_fields: object,
) -> tuple[int, SpotDecisionTrace | None]:
    """Return signed share qty plus a typed spot decision trace."""
    if sizing_input is None:
        sizing_input = spot_sizing_input(**legacy_fields)
    elif legacy_fields:
        raise TypeError("legacy sizing fields cannot accompany SpotSizingInput")
    return SpotPolicy.calc_signed_qty_with_trace(
        **sizing_input.as_kwargs(),
        capture_trace=bool(capture_trace),
    )


def spot_resolve_position_intent(
    *,
    strategy: Mapping[str, object] | object | None,
    current_qty: int,
    target_qty: int,
    policy_config: SpotPolicyConfigView | None = None,
) -> SpotIntentDecision:
    """Resolve order intent needed to move from current_qty to target_qty."""
    return SpotPolicy.resolve_position_intent(
        strategy=strategy,
        current_qty=current_qty,
        target_qty=target_qty,
        policy_config=policy_config,
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
