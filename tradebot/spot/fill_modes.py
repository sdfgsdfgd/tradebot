"""Shared spot fill-mode normalization helpers."""

from __future__ import annotations

SPOT_FILL_MODE_CLOSE = "close"
SPOT_FILL_MODE_NEXT_BAR = "next_bar"
SPOT_FILL_MODE_NEXT_TRADABLE_BAR = "next_tradable_bar"

_CANONICAL_FILL_MODES = {
    SPOT_FILL_MODE_CLOSE,
    SPOT_FILL_MODE_NEXT_BAR,
    SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
}

_FILL_MODE_ALIASES = {
    "bar_close": SPOT_FILL_MODE_CLOSE,
    "at_close": SPOT_FILL_MODE_CLOSE,
    "nextopen": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "next_open": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "open": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "next_bar_open": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "nextbaropen": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "nexttradablebar": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "next_tradable": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "next_tradable_open": SPOT_FILL_MODE_NEXT_TRADABLE_BAR,
    "nextbar": SPOT_FILL_MODE_NEXT_BAR,
}


def normalize_spot_fill_mode(raw: object | None, *, default: str = SPOT_FILL_MODE_CLOSE) -> str:
    cleaned = str(raw if raw is not None else default).strip().lower()
    cleaned = _FILL_MODE_ALIASES.get(cleaned, cleaned)
    if cleaned in _CANONICAL_FILL_MODES:
        return str(cleaned)
    fallback = str(default or SPOT_FILL_MODE_CLOSE).strip().lower()
    fallback = _FILL_MODE_ALIASES.get(fallback, fallback)
    if fallback in _CANONICAL_FILL_MODES:
        return str(fallback)
    return SPOT_FILL_MODE_CLOSE


def spot_fill_mode_is_deferred(raw: object | None) -> bool:
    mode = normalize_spot_fill_mode(raw)
    return mode in (SPOT_FILL_MODE_NEXT_BAR, SPOT_FILL_MODE_NEXT_TRADABLE_BAR)


def spot_fill_mode_is_next_tradable(raw: object | None) -> bool:
    return normalize_spot_fill_mode(raw) == SPOT_FILL_MODE_NEXT_TRADABLE_BAR

