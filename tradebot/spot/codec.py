"""Canonical spot payload normalization shared by live and backtest adapters."""

from __future__ import annotations

from dataclasses import fields

from ..knobs.models import FiltersConfig


def effective_filters_payload(*, group_filters: dict | None, strategy: dict | None) -> dict | None:
    """Resolve the effective filters dict for a milestone payload.

    Merge rules (source-of-truth, see `tradebot/backtest/how-it-works.md`):
    - start from `group.filters`
    - overlay `strategy.filters` (strategy overrides)
    - overlay any filter-shaped keys incorrectly placed at `strategy` root (e.g. `ratsv_*`)

    This keeps backtest replay + live UI parity when milestone payloads evolve.
    """
    merged: dict[str, object] = {}
    if isinstance(group_filters, dict):
        merged.update(group_filters)

    if isinstance(strategy, dict):
        nested = strategy.get("filters")
        if isinstance(nested, dict):
            merged.update(nested)

        filter_keys = {field.name for field in fields(FiltersConfig)}
        for key in filter_keys:
            if key in strategy:
                merged[key] = strategy.get(key)

    return merged or None
