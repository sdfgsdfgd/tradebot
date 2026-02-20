from ib_insync import Contract

from tradebot.client import _normalize_order_contract


def test_normalize_order_contract_fop_prefers_primary_exchange() -> None:
    contract = Contract(secType="FOP", symbol="MCL", primaryExchange="NYMEX")
    normalized = _normalize_order_contract(contract)
    assert normalized.exchange == "NYMEX"


def test_normalize_order_contract_fop_uses_symbol_exchange_hint() -> None:
    contract = Contract(secType="FOP", symbol="MCL")
    normalized = _normalize_order_contract(contract)
    assert normalized.exchange == "NYMEX"


def test_normalize_order_contract_fop_falls_back_to_cme() -> None:
    contract = Contract(secType="FOP", symbol="UNKNOWN")
    normalized = _normalize_order_contract(contract)
    assert normalized.exchange == "CME"


def test_normalize_order_contract_fut_uses_symbol_exchange_hint() -> None:
    contract = Contract(secType="FUT", symbol="MCL")
    normalized = _normalize_order_contract(contract)
    assert normalized.exchange == "NYMEX"
