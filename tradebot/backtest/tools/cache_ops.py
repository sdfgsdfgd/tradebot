"""Backtest cache-maintenance command entrypoint."""
from __future__ import annotations

from ..cache_ops.cli import main as _main
from ..cache_ops.cli import main_audit_heal, main_fetch, main_repair, main_resample, main_sync

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
