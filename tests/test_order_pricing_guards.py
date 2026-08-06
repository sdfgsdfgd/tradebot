from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

_UI_DIR = Path(__file__).resolve().parents[1] / "tradebot" / "ui"
if "tradebot.ui" not in sys.modules:
    ui_pkg = types.ModuleType("tradebot.ui")
    ui_pkg.__path__ = [str(_UI_DIR)]  # type: ignore[attr-defined]
    sys.modules["tradebot.ui"] = ui_pkg

from tradebot.engines.execution import (
    ExecutionPolicy,
    _exec_chase_should_reprice,
    _limit_price_for_mode,
    _sanitize_nbbo,
    _tick_size,
)
from tradebot.ui.common import (
    _quote_health,
    _remember_ticker_display,
    _ticker_line,
)


@pytest.mark.parametrize("action", ["BUY", "SELL"])
@pytest.mark.parametrize("mode", ["OPTIMISTIC", "MID", "AGGRESSIVE", "CROSS"])
def test_limit_price_ignores_negative_quote_sentinels(action: str, mode: str) -> None:
    assert _limit_price_for_mode(-1.0, -1.0, -1.0, action=action, mode=mode) is None


def test_limit_price_uses_positive_last_when_nbbo_missing() -> None:
    assert _limit_price_for_mode(-1.0, -1.0, 70.25, action="BUY", mode="MID") == 70.25


def test_limit_price_unchanged_for_valid_cross_quote() -> None:
    assert _limit_price_for_mode(70.0, 70.2, 70.1, action="BUY", mode="CROSS") == 70.2


def test_sanitize_nbbo_rejects_non_positive_values() -> None:
    assert _sanitize_nbbo(-1.0, 0.0, float("nan")) == (None, None, None)
    assert _sanitize_nbbo(float("inf"), float("-inf"), None) == (None, None, None)


def test_quote_health_flags_actionable_nbbo() -> None:
    out = _quote_health(bid=70.0, ask=70.2, last=None, close=69.5)
    assert out["has_nbbo"] is True
    assert out["has_actionable"] is True
    assert out["has_close_only"] is False


def test_quote_health_detects_close_only_mode() -> None:
    out = _quote_health(bid=-1.0, ask=-1.0, last=None, close=69.5)
    assert out["has_nbbo"] is False
    assert out["has_last"] is False
    assert out["has_close_only"] is True
    assert out["has_actionable"] is False


def test_quote_health_detects_one_sided_quote() -> None:
    out = _quote_health(bid=3.32, ask=None, last=None, close=2.74)
    assert out["has_bid"] is True
    assert out["has_ask"] is False
    assert out["has_one_sided"] is True
    assert out["has_nbbo"] is False
    assert out["has_actionable"] is False


def test_execution_quote_staleness_preserves_unknown_age_and_enforces_known_age() -> None:
    policy = ExecutionPolicy(stale_top_age_sec=2.0)

    assert policy.quote_is_stale(
        ticker=None,
        bid=1.0,
        ask=1.1,
        last=None,
        now_sec=10.0,
    ) is False
    assert policy.quote_is_stale(
        ticker=types.SimpleNamespace(tbTopQuoteUpdatedMono=7.9),
        bid=1.0,
        ask=1.1,
        last=None,
        now_sec=10.0,
    ) is True


def test_ticker_line_can_use_display_fallback_price() -> None:
    class _Ticker:
        bid = None
        ask = None
        last = None
        close = None
        prevLast = None
        marketDataType = 1

        @staticmethod
        def marketPrice() -> float:
            return 123.45

    text = _ticker_line(
        ("NQ",),
        {"NQ": "NASDAQ"},
        {"NQ": _Ticker()},
        None,
        "",
        allow_display_fallback=True,
    )
    assert "123.45" in text.plain


def test_ticker_display_memory_never_regresses_from_closed_to_warming() -> None:
    memory: dict[str, object] = {}
    closed = types.SimpleNamespace(
        contract=types.SimpleNamespace(symbol="TQQQ", conId=72539702),
        bid=None,
        ask=None,
        last=None,
        close=72.84,
        prevLast=72.84,
        marketDataType=3,
    )
    empty = types.SimpleNamespace(
        contract=types.SimpleNamespace(symbol="TQQQ", conId=72539702),
        bid=None,
        ask=None,
        last=None,
        close=None,
        prevLast=None,
        marketDataType=3,
    )
    rendered: list[str] = []
    for source in (closed, empty, closed, empty, empty):
        visible = _remember_ticker_display(  # type: ignore[arg-type]
            {"TQQQ": source},
            memory,  # type: ignore[arg-type]
        )
        rendered.append(
            _ticker_line(
                ("TQQQ",),
                {"TQQQ": "TQQQ"},
                visible,
                None,
                "",
            ).plain
        )

    assert len(set(rendered)) == 1
    assert "Closed" in rendered[0]
    assert "72.84" in rendered[0]
    assert "warming" not in rendered[0]


def test_ticker_display_memory_never_downgrades_a_live_quote() -> None:
    memory: dict[str, object] = {}

    def _ticker(*, bid=None, ask=None, close=None):
        return types.SimpleNamespace(
            contract=types.SimpleNamespace(symbol="TQQQ", conId=72539702),
            bid=bid,
            ask=ask,
            last=None,
            close=close,
            prevLast=close,
            marketDataType=1,
        )

    sources = (
        _ticker(bid=100.0, ask=100.2, close=90.0),
        _ticker(close=91.0),
        _ticker(),
        _ticker(bid=102.0, ask=102.2),
    )
    rendered: list[str] = []
    for source in sources:
        visible = _remember_ticker_display(  # type: ignore[arg-type]
            {"TQQQ": source},
            memory,  # type: ignore[arg-type]
        )
        rendered.append(
            _ticker_line(
                ("TQQQ",),
                {"TQQQ": "TQQQ"},
                visible,
                None,
                "",
            ).plain
        )

    assert all("warming" not in line and "Closed" not in line and "n/a" not in line for line in rendered)
    assert "100.10" in rendered[0]
    assert "100.10" in rendered[1]
    assert "100.10" in rendered[2]
    assert "102.10" in rendered[3]
    assert "+11.10" in rendered[3]


def test_ticker_line_labels_initial_empty_state_as_warming() -> None:
    text = _ticker_line(
        ("TQQQ",),
        {"TQQQ": "TQQQ"},
        {},
        None,
        "",
    )

    assert "warming" in text.plain
    assert "n/a" not in text.plain


def test_tick_size_uses_market_rule_price_ladder() -> None:
    contract = types.SimpleNamespace(
        secType="FOP",
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05), (5.0, 0.25), (100.0, 0.5)),
    )
    assert _tick_size(contract, None, 4.9) == 0.05
    assert _tick_size(contract, None, 90.0) == 0.25
    assert _tick_size(contract, None, 101.3) == 0.5


def test_tick_size_prefers_ticker_ladder_over_contract_defaults() -> None:
    contract = types.SimpleNamespace(
        secType="FOP",
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05),),
    )
    ticker = types.SimpleNamespace(
        minTick=0.05,
        tbPriceIncrements=((0.0, 0.05), (5.0, 0.25), (100.0, 0.5)),
    )
    assert _tick_size(contract, ticker, 101.3) == 0.5


def test_exec_chase_quote_change_respects_min_interval() -> None:
    should = _exec_chase_should_reprice(
        now_sec=100.15,
        last_reprice_sec=100.0,
        mode_now="MID",
        prev_mode="MID",
        quote_signature=(100.0, 101.0, 100.6),
        prev_quote_signature=(100.0, 101.0, 100.5),
        min_interval_sec=0.5,
    )
    assert should is False


def test_exec_chase_mode_change_reprices_immediately() -> None:
    should = _exec_chase_should_reprice(
        now_sec=100.05,
        last_reprice_sec=100.0,
        mode_now="AGGRESSIVE",
        prev_mode="MID",
        quote_signature=(100.0, 101.0, 100.5),
        prev_quote_signature=(100.0, 101.0, 100.5),
        min_interval_sec=5.0,
    )
    assert should is True
