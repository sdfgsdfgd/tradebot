"""Shared signal helpers used by both backtests and live trading.

Keep this small and dependency-free so the live bot and offline sweeps stay aligned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta


_EMA_PRESET_RE = re.compile(r"^\s*(\d+)\s*[/_-]\s*(\d+)\s*$")


def ema_periods(preset: str | None) -> tuple[int, int] | None:
    """Parse an EMA preset like '3/7' into (fast, slow)."""
    if not preset:
        return None
    if not isinstance(preset, str):
        return None
    cleaned = preset.strip()
    if not cleaned:
        return None
    match = _EMA_PRESET_RE.match(cleaned)
    if not match:
        return None
    fast = int(match.group(1))
    slow = int(match.group(2))
    if fast <= 0 or slow <= 0 or fast >= slow:
        return None
    return fast, slow


def ema_next(current: float | None, price: float, period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    if current is None:
        return price
    return (alpha * price) + (1.0 - alpha) * current


def ema_cross(
    prev_fast: float,
    prev_slow: float,
    fast: float,
    slow: float,
) -> tuple[bool, bool]:
    cross_up = prev_fast <= prev_slow and fast > slow
    cross_down = prev_fast >= prev_slow and fast < slow
    return cross_up, cross_down


def ema_state_direction(fast: float, slow: float) -> str | None:
    if fast > slow:
        return "up"
    if fast < slow:
        return "down"
    return None


def flip_exit_mode(flip_exit_mode_raw: str | None, ema_entry_mode_raw: str | None) -> str:
    mode = (flip_exit_mode_raw or "entry").strip().lower()
    ema_entry_mode = (ema_entry_mode_raw or "trend").strip().lower()
    if mode == "entry":
        return "cross" if ema_entry_mode == "cross" else "state"
    if mode in ("state", "trend"):
        return "state"
    if mode in ("cross", "crossover"):
        return "cross"
    return "state"


def direction_from_action_right(action: str, right: str) -> str | None:
    action_u = (action or "").strip().upper()
    right_u = (right or "").strip().upper()
    if (action_u, right_u) in (("BUY", "CALL"), ("SELL", "PUT")):
        return "up"
    if (action_u, right_u) in (("BUY", "PUT"), ("SELL", "CALL")):
        return "down"
    return None


def ema_spread_pct(fast: float, slow: float, price: float) -> float:
    denom = max(float(price), 1e-9)
    return abs(float(fast) - float(slow)) / denom * 100.0


def ema_slope_pct(fast: float, prev_fast: float, price: float) -> float:
    denom = max(float(price), 1e-9)
    return abs(float(fast) - float(prev_fast)) / denom * 100.0


def normalize_ema_entry_mode(mode: str | None) -> str:
    cleaned = str(mode or "trend").strip().lower()
    return "cross" if cleaned in ("cross", "crossover") else "trend"


def update_cross_confirm(
    *,
    cross_up: bool,
    cross_down: bool,
    state: str | None,
    confirm_bars: int,
    pending_dir: str | None,
    pending_bars: int,
) -> tuple[str | None, str | None, int]:
    """Debounce cross entries by requiring N bars of persistence after the cross.

    Semantics:
    - confirm_bars == 0: emit immediately on the cross bar.
    - confirm_bars  > 0: emit on the bar that is confirm_bars bars after the cross,
      as long as the EMA state remains in the crossed direction.
    """
    try:
        confirm = int(confirm_bars or 0)
    except (TypeError, ValueError):
        confirm = 0
    confirm = max(0, confirm)

    if confirm <= 0:
        if cross_up:
            return "up", None, 0
        if cross_down:
            return "down", None, 0
        return None, None, 0

    if cross_up:
        pending_dir = "up"
        pending_bars = 0
    elif cross_down:
        pending_dir = "down"
        pending_bars = 0

    if pending_dir is None:
        return None, None, 0

    if state != pending_dir:
        return None, None, 0

    if not (cross_up or cross_down):
        pending_bars += 1

    if pending_bars >= confirm:
        return pending_dir, None, 0
    return None, pending_dir, pending_bars


def trend_confirmed_state(state: str | None, state_streak: int, *, confirm_bars: int) -> str | None:
    """Return state only once it has persisted for confirm_bars after a change.

    For trend mode, this acts like a simple hysteresis / debounce gate:
    - confirm_bars == 0: returns state as-is.
    - confirm_bars  > 0: requires state_streak >= confirm_bars + 1.
    """
    try:
        confirm = int(confirm_bars or 0)
    except (TypeError, ValueError):
        confirm = 0
    confirm = max(0, confirm)

    if state not in ("up", "down"):
        return None
    if confirm <= 0:
        return state
    return state if int(state_streak or 0) >= (confirm + 1) else None


@dataclass(frozen=True)
class BarSize:
    label: str
    duration: timedelta


def parse_bar_size(bar_size: str) -> BarSize | None:
    if not isinstance(bar_size, str):
        return None
    label = bar_size.strip().lower()
    if label.startswith("1 min"):
        return BarSize(label="1 min", duration=timedelta(minutes=1))
    if label.startswith("2 mins"):
        return BarSize(label="2 mins", duration=timedelta(minutes=2))
    if label.startswith("10 mins") or label.startswith("10 min"):
        return BarSize(label="10 mins", duration=timedelta(minutes=10))
    if label.startswith("1 hour"):
        return BarSize(label="1 hour", duration=timedelta(hours=1))
    if label.startswith("4 hour"):
        return BarSize(label="4 hours", duration=timedelta(hours=4))
    if label.startswith("30 mins"):
        return BarSize(label="30 mins", duration=timedelta(minutes=30))
    if label.startswith("15 mins"):
        return BarSize(label="15 mins", duration=timedelta(minutes=15))
    if label.startswith("5 mins"):
        return BarSize(label="5 mins", duration=timedelta(minutes=5))
    if label.startswith("1 day"):
        return BarSize(label="1 day", duration=timedelta(days=1))
    return None
