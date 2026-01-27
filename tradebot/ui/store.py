"""In-memory snapshot of portfolio items for UI rendering.

Kept under `tradebot.ui` because it is UI-only state, not shared engine logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ib_insync import PortfolioItem


@dataclass
class PortfolioSnapshot:
    items: list[PortfolioItem] = field(default_factory=list)
    updated_at: datetime | None = None
    error: str | None = None

    def update(self, items: list[PortfolioItem], error: str | None = None) -> None:
        self.items = items
        self.error = error
        self.updated_at = datetime.now(timezone.utc)
