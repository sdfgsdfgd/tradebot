"""Fingerprint helpers shared by spot sweep tooling."""

from __future__ import annotations

import json

from .spot_codec import weekdays_from_payload as _codec_weekdays_from_payload

_WDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_STRATEGY_FINGERPRINT_NON_SEMANTIC_KEYS = frozenset(
    {
        "name",
        "spot_sec_type",
        "spot_exchange",
    }
)


def _entry_days_labels(days: tuple[int, ...]) -> list[str]:
    out: list[str] = []
    for d in days:
        try:
            idx = int(d)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(_WDAYS):
            out.append(_WDAYS[idx])
    return out


def _canonicalize_fingerprint_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _canonicalize_fingerprint_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_fingerprint_value(v) for v in value]
    return value


def _canonicalize_strategy_for_fingerprint(strategy: dict) -> dict:
    out: dict[str, object] = {}
    for raw_key, raw_val in dict(strategy).items():
        key = str(raw_key)
        if key in _STRATEGY_FINGERPRINT_NON_SEMANTIC_KEYS:
            continue
        if key == "entry_days":
            days = tuple(sorted(set(_codec_weekdays_from_payload(raw_val))))
            out[key] = _entry_days_labels(days)
            continue
        out[key] = _canonicalize_fingerprint_value(raw_val)
    return out


def _canonicalize_filters_for_fingerprint(filters: dict | None) -> dict | None:
    if filters is None:
        return None
    if not isinstance(filters, dict):
        return None
    normalized = _canonicalize_fingerprint_value(filters)
    if not isinstance(normalized, dict):
        return None
    return normalized


def _strategy_fingerprint(
    strategy: dict,
    *,
    filters: dict | None,
    signal_bar_size: str | None = None,
    signal_use_rth: bool | None = None,
) -> str:
    raw = _canonicalize_strategy_for_fingerprint(strategy)
    raw["filters"] = _canonicalize_filters_for_fingerprint(filters)
    if signal_bar_size is not None:
        raw["signal_bar_size"] = str(signal_bar_size)
    if signal_use_rth is not None:
        raw["signal_use_rth"] = bool(signal_use_rth)
    return json.dumps(raw, sort_keys=True, default=str)
