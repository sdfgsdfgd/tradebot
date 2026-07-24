from __future__ import annotations

from types import SimpleNamespace

from ib_insync import Option

from tradebot.backtest.quotes import (
    QuoteContract,
    QuoteSnapshot,
    quote_captured_option_package,
)
from tradebot.live.options import QualifiedOptionLeg, quote_live_option_package
from tradebot.option_package import ResolvedOptionLeg


def _option(
    con_id: int,
    strike: float,
    *,
    bid: float,
    ask: float,
    last: float,
) -> QuoteContract:
    return QuoteContract(
        con_id=con_id,
        sec_type="OPT",
        symbol="XSP",
        local_symbol=f"XSP-{con_id}",
        exchange="SMART",
        currency="USD",
        expiry="20260731",
        strike=strike,
        right="P",
        trading_class="XSP",
        multiplier="100",
        bid=bid,
        ask=ask,
        last=last,
        market_data_type=1,
        quote_time="2026-07-24T14:29:55+00:00",
        min_tick=0.01,
    )


def test_captured_and_live_package_quotes_share_exact_pricing_and_risk() -> None:
    rows = (
        _option(101, 735.0, bid=2.0, ask=2.2, last=2.1),
        _option(102, 734.0, bid=1.2, ask=1.4, last=1.3),
    )
    snapshot = QuoteSnapshot(
        ts="2026-07-24T14:30:00+00:00",
        md_type=1,
        symbol="XSP",
        underlying=QuoteContract(
            con_id=11004968,
            sec_type="IND",
            symbol="XSP",
            local_symbol="XSP",
            exchange="CBOE",
            currency="USD",
        ),
        options=list(rows),
        errors=[],
    )
    resolved = (
        ResolvedOptionLeg("SELL", "PUT", 735.0, 1, "20260731"),
        ResolvedOptionLeg("BUY", "PUT", 734.0, 1, "20260731"),
    )

    captured = quote_captured_option_package(
        snapshot,
        resolved,
        mode="MID",
        max_age_sec=30,
        require_live=True,
    )

    live_legs = []
    live_tickers = []
    for row, leg in zip(rows, resolved):
        contract = Option(
            symbol="XSP",
            lastTradeDateOrContractMonth=row.expiry,
            strike=row.strike,
            right=row.right,
            exchange="SMART",
            currency="USD",
            multiplier="100",
            tradingClass="XSP",
        )
        contract.conId = int(row.con_id or 0)
        live_legs.append(QualifiedOptionLeg(contract, leg.action, leg.ratio))
        live_tickers.append(
            SimpleNamespace(
                bid=row.bid,
                ask=row.ask,
                last=row.last,
                minTick=row.min_tick,
            )
        )
    live = quote_live_option_package(
        symbol="XSP",
        legs=live_legs,
        tickers=live_tickers,
        quantity=1,
        intent="enter",
        mode="MID",
    )

    assert captured is not None
    assert live is not None
    assert (
        captured.quote.bid_value,
        captured.quote.ask_value,
        captured.quote.mid_value,
        captured.quote.limit_value,
        captured.quote.tick,
    ) == (
        live.bid_value,
        live.ask_value,
        live.mid_value,
        live.limit_value,
        live.tick,
    )
    assert captured.package.legs == live.live.package.legs
    assert captured.package.quantity == live.live.package.quantity
    assert captured.package.debit_value == live.live.package.debit_value
    assert captured.package.product.source == "captured"
    assert live.live.package.product.source == "broker"
    assert captured.risk == live.live.risk
