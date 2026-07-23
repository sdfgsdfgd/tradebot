"""Dependency-free canonical contract identity semantics."""

from __future__ import annotations

FUTURE_EXCHANGES: dict[str, str] = {
    "SI": "COMEX",
    "GC": "COMEX",
    "1OZ": "COMEX",
    "CL": "NYMEX",
    "MCL": "NYMEX",
    "NG": "NYMEX",
    "HG": "COMEX",
    "ES": "CME",
    "MES": "CME",
    "NQ": "CME",
    "MNQ": "CME",
    "YM": "CBOT",
    "MYM": "CBOT",
    "RTY": "CME",
    "M2K": "CME",
    "MBT": "CME",
}

INDEX_EXCHANGES: dict[str, str] = {
    "XSP": "CBOE",
    "SPX": "CBOE",
    "TICK-NYSE": "NYSE",
    "TICK-AMEX": "AMEX",
}

_FUTURE_MULTIPLIERS: dict[str, float] = {
    "MNQ": 2.0,
    "MBT": 0.1,
    "MCL": 100.0,
}


def normalize_contract_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def future_exchange_for_symbol(symbol: object) -> str | None:
    return FUTURE_EXCHANGES.get(normalize_contract_symbol(symbol))


def is_future_symbol(symbol: object) -> bool:
    return future_exchange_for_symbol(symbol) is not None


def index_exchange_for_symbol(symbol: object) -> str | None:
    return INDEX_EXCHANGES.get(normalize_contract_symbol(symbol))


def future_multiplier_for_symbol(symbol: object, *, default: float = 1.0) -> float:
    normalized = normalize_contract_symbol(symbol)
    if normalized in _FUTURE_MULTIPLIERS:
        return _FUTURE_MULTIPLIERS[normalized]
    try:
        fallback = float(default)
    except (TypeError, ValueError):
        fallback = 1.0
    return fallback if fallback > 0 else 1.0
