"""Backward-compatible shim for backtest knob models.

Canonical import path: `tradebot.knobs.models`.
"""

from __future__ import annotations

from ...knobs.models import (
    BacktestConfig,
    ConfigBundle,
    FiltersConfig,
    LegConfig,
    SpotLegConfig,
    StrategyConfig,
    SyntheticConfig,
)

__all__ = [
    "BacktestConfig",
    "ConfigBundle",
    "FiltersConfig",
    "LegConfig",
    "SpotLegConfig",
    "StrategyConfig",
    "SyntheticConfig",
]
