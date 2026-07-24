from datetime import datetime, time, timedelta
from types import SimpleNamespace

import pytest

from tradebot.chart_data.series import OhlcvBar
from tradebot.research.evidence import (
    XSP_CREDIT_BARRIER_SCHEMA,
    backtest_evidence,
    research_rank_key,
    xsp_credit_barrier_census,
)
from tradebot.time_utils import ET_ZONE


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


def test_xsp_credit_barrier_census_is_fixed_causal_and_monotonic() -> None:
    bars = []
    day = datetime(2026, 1, 5)
    for session in range(10):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        base = 100.0 + session * 0.25
        for boundary in (time(10), time(10, 30), time(11), time(11, 30), time(15, 55)):
            price = base + (boundary.hour * 60 + boundary.minute - 600) / 600.0
            bars.append(
                OhlcvBar(
                    datetime.combine(day.date(), boundary, tzinfo=ET_ZONE),
                    price,
                    price + 0.1,
                    price - 0.1,
                    price,
                    0.0,
                )
            )
        day += timedelta(days=1)

    result = xsp_credit_barrier_census(
        bars,
        source_fingerprint="admitted-source",
    )

    assert result["schema"] == XSP_CREDIT_BARRIER_SCHEMA
    assert result["source"] == {
        "symbol": "XSP",
        "bar_size": "5 mins",
        "use_rth": True,
        "start": "2026-01-05",
        "end": "2026-01-16",
        "bars": 50,
        "sessions": 10,
        "stitched_source_manifest_sha256": "admitted-source",
    }
    cells = result["cells"]
    assert len(cells) == 128
    assert len(
        {
            (
                row["decision_time_et"],
                row["offset_pct"],
                row["horizon_sessions"],
                row["side"],
            )
            for row in cells
        }
    ) == 128
    for row in cells:
        assert row["touches"] >= row["expiration_breaches"]
        assert row["required_credit_price"] == pytest.approx(
            row["expiration_breach_rate_upper95"] + 0.10
        )
        assert sum(
            cohort["observations"] for cohort in row["annual"].values()
        ) == row["observations"]

    for boundary in ("10:00", "10:30", "11:00", "11:30"):
        for horizon in (0, 1, 3, 5):
            for side in ("put_credit", "call_credit"):
                comparable = sorted(
                    (
                        row
                        for row in cells
                        if row["decision_time_et"] == boundary
                        and row["horizon_sessions"] == horizon
                        and row["side"] == side
                    ),
                    key=lambda row: row["offset_pct"],
                )
                risks = [
                    row["expiration_breach_rate"] for row in comparable
                ]
                assert risks == sorted(risks, reverse=True)
