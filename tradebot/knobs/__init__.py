"""Shared strategy/filter configuration "knobs" (dataclasses).

These models are intentionally dumb containers: no IO, no IBKR, no backtest logic.

Canonical import path: `tradebot.knobs.models`.

Backtest code may still import from `tradebot.backtest.knobs.models`, which remains
as a shim for compatibility.
"""

