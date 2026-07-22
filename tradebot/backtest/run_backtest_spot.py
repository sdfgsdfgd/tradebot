"""Spot research and multiwindow backtest entrypoints.

Canonical implementations:
- `tradebot.research.spot_sweeps` (spot evolution sweeps)
- `tradebot.backtest.run_backtests_spot_multiwindow` (spot multiwindow stability eval)

`tradebot.backtest.run_backtest` dispatches subcommands through this wrapper.
"""

from __future__ import annotations

from .run_backtests_spot_multiwindow import multitimeframe_main, spot_multitimeframe_main
from ..research.spot_sweeps.runtime import main

__all__ = ["main", "multitimeframe_main", "spot_multitimeframe_main"]


if __name__ == "__main__":
    main()
