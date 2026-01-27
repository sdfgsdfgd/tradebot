"""Backward-compatible shim.

This codebase renamed `tradebot.decision_core` → `tradebot.engine` to reflect that
it is the core shared engine module used by both UI/live and backtests.

Keep this file so existing imports continue to work.
"""

from __future__ import annotations

from . import engine as _engine

globals().update({k: v for k, v in _engine.__dict__.items() if not k.startswith("__")})
