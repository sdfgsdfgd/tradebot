"""Spot multi-window stability eval (kingmaker) entrypoint.

This file exists mostly for organization and future splitting:
- Keeps spot backtest-related entrypoints clustered together.
- Lets `run_backtest_spot.py` stay a tiny shim for backwards-compatibility.

Today the implementation still lives in `run_backtests_spot_sweeps.py` (single canonical module).
Next pass we can physically move the multiwindow implementation here.
"""

from __future__ import annotations

from .run_backtests_spot_sweeps import multitimeframe_main, spot_multitimeframe_main


if __name__ == "__main__":
    spot_multitimeframe_main()

