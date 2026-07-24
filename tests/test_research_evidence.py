from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from tradebot.research.evidence import backtest_evidence, research_rank_key


def _result(*, daily_equity: list[float], trade_pnls: list[float]):
    start = datetime(2026, 1, 2, 16)
    return SimpleNamespace(
        equity=[
            SimpleNamespace(ts=start + timedelta(days=index), equity=equity)
            for index, equity in enumerate(daily_equity)
        ],
        trades=[
            SimpleNamespace(pnl=lambda _multiplier, value=value: value)
            for value in trade_pnls
        ],
        summary=SimpleNamespace(
            total_pnl=sum(trade_pnls),
            max_drawdown=20.0,
        ),
    )


def test_backtest_evidence_includes_zero_sessions_and_cost_adjusted_trade_outcomes() -> None:
    evidence = backtest_evidence(
        _result(
            daily_equity=[1_010.0, 1_010.0, 1_005.0],
            trade_pnls=[12.0, -7.0],
        ),
        starting_cash=1_000.0,
        multiplier=1.0,
    )

    assert evidence["sessions"] == 3
    assert evidence["active_sessions"] == 2
    assert evidence["mean_daily_pnl"] == pytest.approx(5.0 / 3.0)
    assert evidence["worst_daily_pnl"] == -5.0
    assert evidence["profit_factor"] == pytest.approx(12.0 / 7.0)
    assert evidence["pnl_over_max_drawdown"] == 0.25
    assert evidence["sample_gate"] is False


def test_research_rank_never_uses_win_rate_as_an_authority() -> None:
    robust = {
        "metrics": {"pnl": 10.0, "win_rate": 0.4, "trades": 60},
        "evidence": {
            "sample_gate": True,
            "positive_lcb": True,
            "daily_pnl_lcb95": 0.5,
            "pnl_over_max_drawdown": 1.0,
            "profit_factor": 1.5,
            "top_5_win_share": 0.3,
        },
    }
    fragile = {
        "metrics": {"pnl": 100.0, "win_rate": 1.0, "trades": 8},
        "evidence": {
            "sample_gate": False,
            "positive_lcb": False,
            "daily_pnl_lcb95": -0.1,
            "pnl_over_max_drawdown": 10.0,
            "profit_factor": None,
            "top_5_win_share": 1.0,
        },
    }

    assert research_rank_key(robust) > research_rank_key(fragile)
