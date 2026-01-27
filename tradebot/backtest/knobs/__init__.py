"""Backtest configuration "knobs" (dataclasses).

This package exists to keep `tradebot/backtest/config.py` focused on config loading/parsing,
while keeping all the strategy/filter knobs in one place.

Canonical models live in `tradebot/knobs/models.py`; this package remains as a
compatibility surface for existing backtest imports.
"""
