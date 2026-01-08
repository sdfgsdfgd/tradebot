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


@dataclass(frozen=True)
class BarSize:
    label: str
    duration: timedelta


def parse_bar_size(bar_size: str) -> BarSize | None:
    if not isinstance(bar_size, str):
        return None
    label = bar_size.strip().lower()
    if label.startswith("1 hour"):
        return BarSize(label="1 hour", duration=timedelta(hours=1))
    if label.startswith("30 mins"):
        return BarSize(label="30 mins", duration=timedelta(minutes=30))
    if label.startswith("15 mins"):
        return BarSize(label="15 mins", duration=timedelta(minutes=15))
    if label.startswith("5 mins"):
        return BarSize(label="5 mins", duration=timedelta(minutes=5))
    if label.startswith("1 day"):
        return BarSize(label="1 day", duration=timedelta(days=1))
    return None

