from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradebot.ui.positions import PositionDetailScreen


@pytest.mark.parametrize(
    ("contract", "expected"),
    [
        (SimpleNamespace(symbol="NVDA", secType="STK"), "NVDA STOCK"),
        (SimpleNamespace(symbol="MNQ", secType="FUT"), "MNQ FUTURES"),
        (SimpleNamespace(symbol="NVDA", secType="OPT", right="C"), "NVDA OPTIONS CALLS"),
        (SimpleNamespace(symbol="NVDA", secType="OPT", right="P"), "NVDA OPTIONS PUTS"),
        (SimpleNamespace(symbol="MCL", secType="FOP", right="C"), "MCL FOP CALLS"),
    ],
)
def test_contract_header_title_displays_instrument_and_option_side(contract: object, expected: str) -> None:
    assert PositionDetailScreen._contract_header_title(contract) == expected
