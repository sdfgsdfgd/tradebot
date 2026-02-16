"""Thin CLI wrapper for unified cache operations.

Core cache logic lives in `tradebot.backtest.cache_ops_lib`.
"""
from __future__ import annotations

from ..cache_ops_lib import main as _main
from ..cache_ops_lib import main_audit_heal, main_fetch, main_repair, main_resample, main_sync

__all__ = [
    "main",
    "main_sync",
    "main_fetch",
    "main_repair",
    "main_audit_heal",
    "main_resample",
]


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
