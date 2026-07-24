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


def select_option_chain(
    chains: object,
    symbol: object,
    *,
    prefer_smart: bool = True,
) -> object | None:
    """Choose the most complete matching chain without provider-specific imports."""
    rows = list(chains or [])
    if prefer_smart:
        smart = [
            row
            for row in rows
            if normalize_contract_symbol(getattr(row, "exchange", None)) == "SMART"
        ]
        rows = smart or rows
    symbol_key = "".join(char for char in normalize_contract_symbol(symbol) if char.isalnum())

    def score(row: object) -> tuple[int, int, int, int, int]:
        expirations = getattr(row, "expirations", ()) or ()
        strikes = getattr(row, "strikes", ()) or ()
        trading_class = "".join(
            char
            for char in normalize_contract_symbol(getattr(row, "tradingClass", None))
            if char.isalnum()
        )
        return (
            len(expirations) * len(strikes),
            int(trading_class == symbol_key),
            int(normalize_contract_symbol(getattr(row, "exchange", None)) == "SMART"),
            len(expirations),
            len(strikes),
        )

    return max(rows, key=score) if rows else None


def future_multiplier_for_symbol(symbol: object, *, default: float = 1.0) -> float:
    normalized = normalize_contract_symbol(symbol)
    if normalized in _FUTURE_MULTIPLIERS:
        return _FUTURE_MULTIPLIERS[normalized]
    try:
        fallback = float(default)
    except (TypeError, ValueError):
        fallback = 1.0
    return fallback if fallback > 0 else 1.0
