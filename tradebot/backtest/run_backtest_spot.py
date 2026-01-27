"""Spot backtest shims (backwards-compat imports + small file).

Canonical implementations:
- `tradebot.backtest.run_backtests_spot_sweeps` (spot evolution sweeps)
- `tradebot.backtest.run_backtests_spot_multiwindow` (spot multiwindow stability eval)

`tradebot.backtest.run_backtest` dispatches subcommands through this wrapper.
"""

from __future__ import annotations

from .run_backtests_spot_multiwindow import multitimeframe_main, spot_multitimeframe_main
from .run_backtests_spot_sweeps import main


if __name__ == "__main__":
    main()

