from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from ib_insync import Contract
import pytest

from tradebot.backtest.quotes import (
    QuoteContract,
    QuoteSnapshot,
    option_parity_observation,
)
from tradebot.chart_data.history import duration_window_et
from tradebot.news.contract import SCHEMA as NEWS_SCHEMA
from tradebot.news.contract import SCORE_VERSION as NEWS_SCORE_VERSION
from tradebot.research.live_calibration import (
    SELECTED_CASH_EQUITY_SCHEMA,
    SELECTED_EQUITY_SCHEMA,
    XSP_DIRECTIONAL_OBSERVER_VERSION,
    LiveCalibrationLedger,
    XspProfitabilityPolicy,
    calibration_fingerprint,
)
from tradebot.research.xsp_benchmarks import (
    XSP_DIRECTIONAL_SHADOW_POLICY,
    XSP_SELECTED_SHADOW_RUN_VERSION,
    xsp_fundamental_defensive_benchmark,
    xsp_opening_edge_shadow_recommendation,
    xsp_option_parity_participation_benchmark,
    xsp_profitability_policy_from_selected_run,
    xsp_selected_shadow_run,
)
from tradebot.research.xsp_candidate import (
    XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
    XSP_OPENING_EDGE_VERSION,
)
from tradebot.research.xsp_shadow import (
    XSP_DIRECTIONAL_HISTORY_DURATION,
    advance_xsp_directional_shadow,
    advance_xsp_shadow_from_ibkr,
    replay_xsp_directional_shadow,
    settle_xsp_directional_observations,
)
from tradebot.research.xsp_context import (
    xsp_fundamental_context_at,
    xsp_option_context_at,
)
from tradebot.research.xsp_opening_edge_v2 import (
    XSP_OPENING_EDGE_V2_HISTORY_DURATION,
    XSP_OPENING_EDGE_V2_EXECUTION_GATE,
    XSP_OPENING_EDGE_V2_VERSION,
    _equity,
    _trade_row,
    advance_xsp_opening_edge_v2_from_ibkr,
    load_xsp_opening_edge_v2_spec,
    next_xsp_v2_run_start,
    split_xsp_v2_sessions,
    xsp_opening_edge_v2_bundle,
    xsp_opening_edge_v2_equities,
    xsp_opening_edge_v2_gth_signal_bars,
    xsp_opening_edge_v2_run_start,
)
from tradebot.backtest.models import Bar, SpotTrade
from tradebot.time_utils import ET_ZONE


NOW = datetime(2026, 7, 27, 13, 35, tzinfo=timezone.utc)


def test_opening_edge_v2_current_crown_is_content_addressed_and_fail_closed() -> None:
    spec = load_xsp_opening_edge_v2_spec()
    execution_receipt = json.loads(
        (
            spec.artifact_path.parent / "opening_edge_v2_spy_execution_receipt.json"
        ).read_text()
    )
    assert execution_receipt["schema"] == (
        "xsp.opening-edge-v2-spy-execution-receipt.v2"
    )
    rth = xsp_opening_edge_v2_bundle(
        spec,
        lane="rth",
        start=date(2026, 7, 28),
        end=date(2026, 7, 28),
        cost_profile="research",
    )
    gth = xsp_opening_edge_v2_bundle(
        spec,
        lane="gth",
        start=date(2026, 7, 28),
        end=date(2026, 7, 28),
        cost_profile="broker",
    )

    assert XSP_OPENING_EDGE_V2_VERSION == ("xsp.opening-edge-v2-balanced-24x5.v1")
    assert spec.artifact_sha256 == (
        "f879cc20c4434e33626c143ccd85db4d608370a6fb7c321b1ee0f1f2c08afff2"
    )
    assert spec.group["authority"] == "historical_research_crown_only"
    assert rth.strategy.symbol == "SPY"
    assert rth.backtest.use_rth is True
    assert rth.strategy.spot_commission_per_share == 0.05
    assert gth.strategy.symbol == "XSP"
    assert gth.backtest.use_rth is False
    assert gth.strategy.spot_commission_per_share == 1.01540245
    assert XSP_OPENING_EDGE_V2_EXECUTION_GATE["eligible"] is False
    assert execution_receipt["crown"]["sha256"] == spec.artifact_sha256
    assert (
        execution_receipt["source_audit"]["sha256"]
        == (XSP_OPENING_EDGE_V2_EXECUTION_GATE["audit_sha256"])
    )
    assert (
        execution_receipt["verdict"]["status"]
        == (XSP_OPENING_EDGE_V2_EXECUTION_GATE["verdict"])
    )
    assert execution_receipt["observer_parity"]["ordered_economic_ledger_equal"] is True
    ceiling = execution_receipt["fixed_transport_ceiling"]
    assert (
        ceiling["cadence_floor_oracle"]["maximum_affordable_round_trip_points"]
        == 1.221649
    )
    assert (
        ceiling["measured_round_trip_points"]
        > ceiling["cadence_floor_oracle"]["maximum_affordable_round_trip_points"]
    )


def test_opening_edge_v2_uses_xsp_clock_and_never_backfills_a_live_start() -> None:
    bars = (
        Bar(datetime(2026, 7, 28, 0, 20), 100, 101, 99, 100, 1),
        Bar(datetime(2026, 7, 28, 13, 35), 101, 102, 100, 101, 1),
        Bar(datetime(2026, 7, 28, 20, 20), 102, 103, 101, 102, 1),
    )
    gth, rth = split_xsp_v2_sessions(bars)

    assert [row.ts for row in gth] == [datetime(2026, 7, 28, 0, 20)]
    assert [row.ts for row in rth] == [datetime(2026, 7, 28, 13, 35)]
    observed = datetime(2026, 7, 27, 11, 30, tzinfo=ET_ZONE)
    assert next_xsp_v2_run_start(observed) == datetime(
        2026,
        7,
        28,
        0,
        15,
        tzinfo=timezone.utc,
    )


def test_opening_edge_v2_paired_equity_keeps_cost_attribution_separate() -> None:
    spec = load_xsp_opening_edge_v2_spec()
    prior_spy = Bar(datetime(2026, 7, 27, 20, 0), 200, 200, 200, 200, 1)
    prior_xsp = Bar(datetime(2026, 7, 27, 20, 0), 100, 100, 100, 100, 0)
    gth = [
        Bar(
            datetime(2026, 7, 28, 0, 20) + timedelta(minutes=5 * index),
            100.0 + index * 0.05,
            100.1 + index * 0.05,
            99.9 + index * 0.05,
            100.05 + index * 0.05,
            1.0,
        )
        for index in range(100)
    ]
    rth = [
        Bar(
            datetime(2026, 7, 28, 13, 35) + timedelta(minutes=5 * index),
            105.0 + index * 0.05,
            105.1 + index * 0.05,
            104.9 + index * 0.05,
            105.05 + index * 0.05,
            1.0,
        )
        for index in range(78)
    ]
    paired = xsp_opening_edge_v2_equities(
        spec=spec,
        spy_bars=(prior_spy, *gth, *rth),
        xsp_rth_bars=(prior_xsp, *rth),
        observed_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
        run_started_at=datetime(2026, 7, 28, 0, 15, tzinfo=timezone.utc),
    )
    research = paired["profiles"]["research"]
    broker = paired["profiles"]["broker"]

    assert paired["rth_signal_source"] == "XSP"
    assert paired["gth_signal_source"] == ("prior_xsp_close_anchored_spy_returns")
    assert research["gth_signal_symbol"] == broker["gth_signal_symbol"] == "XSP"
    assert paired["gth_signal_bars"] == len(gth)
    assert paired["signal_clock"] == "XSP"
    assert paired["execution_symbol"] == "SPY"
    assert set(paired["signal_observations"]) == {"gth", "rth"}
    for observation in paired["signal_observations"].values():
        assert observation["schema"] == "spot.signal-snapshot.v1"
        assert observation["signal_snapshot_age_bars"] == 0
        assert observation["signal_bar_ts"]
        assert observation["entry_control"]["plan"]
        assert observation["entry_control"]["source"] == "directional_impulse"
        assert observation["directional_impulse"]["horizons"]
    assert paired["execution_eligibility"]["eligible"] is False
    assert research["order_authority"] == broker["order_authority"] == "none"
    assert research["config_fingerprint"] != broker["config_fingerprint"]
    assert research["cost_profile"]["round_trip_points"] == 0.10
    assert broker["cost_profile"]["round_trip_points"] == 2.0608049
    assert research["cumulative_cost_points"] <= broker["cumulative_cost_points"]
    assert research["reconciled"] is broker["reconciled"] is True


def test_opening_edge_v2_trade_keeps_compact_causal_attribution() -> None:
    trade = SpotTrade(
        symbol="SPY",
        qty=1,
        entry_time=datetime(2026, 7, 28, 13, 40),
        entry_price=636.0,
        bars_held=7,
        max_favorable_excursion=1.25,
        max_adverse_excursion=0.35,
        decision_trace={
            "entry_guard_inputs": {
                "signal_bar_ts": "2026-07-28T13:35:00",
                "signal_source_dir": "up",
                "shock_atr_vel_pct": 0.04,
                "slope_vel_pct": 0.03,
                "entry_control": {
                    "source": "directional_impulse",
                    "proposed_direction": "up",
                    "controls": ["directional_impulse_admission:pass:core"],
                    "blocked_by": None,
                    "direction": "up",
                    "branch": None,
                    "plan": {"must_not_duplicate_policy": True},
                },
                "directional_impulse": {
                    "turn_event": "up",
                    "atr_velocity_pct": 0.04,
                    "horizons": [
                        {
                            "elapsed_minutes": 5.0,
                            "slope_angle_deg": 22.0,
                            "slope_velocity_pct_per_bar": 0.08,
                        }
                    ],
                },
            },
            "entry_local_extrema_probe": {"15m": {"range_pos": 0.1}},
            "exits": [
                {
                    "stage": "pending_exit",
                    "bar_ts": "2026-07-28T14:15:00",
                    "signal_snapshot": {
                        "signal_bar_ts": "2026-07-28T14:10:00",
                        "slope_vel_pct": -0.02,
                    },
                    "local_extrema_probe": {"15m": {"range_pos": 0.9}},
                }
            ],
        },
        exit_time=datetime(2026, 7, 28, 14, 15),
        exit_price=636.8,
        exit_reason="flip",
    )

    attribution = _trade_row(
        trade,
        lane="rth",
        cost_profile="research",
    )["attribution"]

    assert attribution["bars_held"] == 7
    assert attribution["execution_mfe_points"] == pytest.approx(1.25)
    assert attribution["execution_mae_points"] == pytest.approx(0.35)
    assert attribution["entry"]["control"] == {
        "source": "directional_impulse",
        "proposed_direction": "up",
        "controls": ["directional_impulse_admission:pass:core"],
        "blocked_by": None,
        "direction": "up",
        "branch": None,
    }
    assert "plan" not in attribution["entry"]["control"]
    assert (
        attribution["entry"]["directional_impulse"]["horizons"][0]["slope_angle_deg"]
        == 22.0
    )
    assert attribution["entry"]["local_extrema"]["15m"]["range_pos"] == 0.1
    assert attribution["exit"]["signal_snapshot"]["slope_vel_pct"] == -0.02
    assert attribution["exit"]["local_extrema"]["15m"]["range_pos"] == 0.9


def test_opening_edge_v2_refuses_false_attribution_completeness() -> None:
    trade = SpotTrade(
        symbol="SPY",
        qty=1,
        entry_time=datetime(2026, 7, 28, 13, 40),
        entry_price=636.0,
        exit_time=datetime(2026, 7, 28, 14, 15),
        exit_price=636.8,
        exit_reason="flip",
    )
    result = _equity(
        load_xsp_opening_edge_v2_spec(),
        cost_profile="research",
        rth=SimpleNamespace(trades=[trade], equity=[]),
        gth=SimpleNamespace(trades=[], equity=[]),
        run_started_at=datetime(2026, 7, 28, 0, 15, tzinfo=timezone.utc),
        observed_at=datetime(2026, 7, 28, 14, 15, tzinfo=timezone.utc),
        rth_signal_symbol="XSP",
    )

    assert result["closed_trades"] == 1
    assert result["attribution_complete"] is False


def test_opening_edge_v2_open_mark_is_not_reported_as_closed_trade() -> None:
    mark = SpotTrade(
        symbol="SPY",
        qty=1,
        entry_time=datetime(2026, 7, 28, 13, 40),
        entry_price=636.0,
        exit_time=datetime(2026, 7, 28, 14, 15),
        exit_price=636.8,
        exit_reason="end",
    )
    result = _equity(
        load_xsp_opening_edge_v2_spec(),
        cost_profile="research",
        rth=SimpleNamespace(trades=[mark], equity=[]),
        gth=SimpleNamespace(trades=[], equity=[]),
        run_started_at=datetime(2026, 7, 28, 0, 15, tzinfo=timezone.utc),
        observed_at=datetime(2026, 7, 28, 14, 15, tzinfo=timezone.utc),
        rth_signal_symbol="XSP",
    )

    assert result["closed_trades"] == 0
    assert result["cumulative_realized_net_points"] == 0
    assert result["latest_position"]["exit_reason"] == "end"
    assert result["latest_trade"] is None


def test_opening_edge_v2_gth_projection_uses_prior_completed_xsp_close() -> None:
    prior_close = datetime(2026, 7, 24, 20, 0)
    future_close = datetime(2026, 7, 27, 20, 0)
    projected = xsp_opening_edge_v2_gth_signal_bars(
        (
            Bar(prior_close, 200, 200, 200, 200, 1),
            Bar(datetime(2026, 7, 27, 0, 20), 202, 204, 200, 203, 9),
            Bar(future_close, 300, 300, 300, 300, 1),
        ),
        (
            Bar(prior_close, 100, 100, 100, 100, 0),
            Bar(future_close, 300, 300, 300, 300, 0),
        ),
    )

    assert projected == (Bar(datetime(2026, 7, 27, 0, 20), 101, 102, 100, 101.5, 0),)


def test_opening_edge_v2_observer_freezes_start_before_broker_work(
    tmp_path,
) -> None:
    class _NoBroker:
        async def qualify_proxy_contracts(self, _contract):
            raise AssertionError("preflight must not contact IBKR")

    ledger = LiveCalibrationLedger(tmp_path / "v2-preflight.jsonl")
    run_start = datetime(2026, 7, 27, 20, 15, tzinfo=ET_ZONE)
    preflight_at = datetime(2026, 7, 27, 17, 5, tzinfo=ET_ZONE)
    preflight = asyncio.run(
        advance_xsp_opening_edge_v2_from_ibkr(
            ledger,
            client=_NoBroker(),
            observed_at=preflight_at,
            run_started_at=run_start,
            recorded_at=preflight_at,
        )
    )

    assert preflight["evaluation_status"] == "CLOSED"
    assert preflight["broker_request_skipped"] == "run_not_started"
    assert preflight["run_started_at_utc"] == "2026-07-28T00:15:00+00:00"
    assert preflight["order_authority"] == "none"
    [checkpoint] = list(ledger.records())
    assert checkpoint["evidence"]["paired_equity"] is None
    assert checkpoint["evidence"]["order_authority"] == "none"
    assert xsp_opening_edge_v2_run_start(
        tuple(ledger.records()),
        observed_at=preflight_at,
    ) == datetime(2026, 7, 28, 0, 15, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="run start drift"):
        asyncio.run(
            advance_xsp_opening_edge_v2_from_ibkr(
                ledger,
                client=_NoBroker(),
                observed_at=preflight_at,
                run_started_at=datetime(
                    2026,
                    7,
                    28,
                    20,
                    15,
                    tzinfo=ET_ZONE,
                ),
                recorded_at=preflight_at,
            )
        )
    with pytest.raises(ValueError, match="frozen before observation"):
        asyncio.run(
            advance_xsp_opening_edge_v2_from_ibkr(
                LiveCalibrationLedger(tmp_path / "v2-backfill.jsonl"),
                client=_NoBroker(),
                observed_at=datetime(
                    2026,
                    7,
                    27,
                    20,
                    31,
                    tzinfo=ET_ZONE,
                ),
                run_started_at=run_start,
                recorded_at=datetime(
                    2026,
                    7,
                    27,
                    20,
                    31,
                    tzinfo=ET_ZONE,
                ),
            )
        )
    with pytest.raises(ValueError, match="exact GTH boundary"):
        asyncio.run(
            advance_xsp_opening_edge_v2_from_ibkr(
                LiveCalibrationLedger(tmp_path / "v2-off-boundary.jsonl"),
                client=_NoBroker(),
                observed_at=preflight_at,
                run_started_at=run_start + timedelta(minutes=5),
                recorded_at=preflight_at,
            )
        )


def test_opening_edge_v2_observer_uses_spy_full_session_and_xsp_rth(
    tmp_path,
) -> None:
    spy_contract = Contract(
        conId=756733,
        symbol="SPY",
        secType="STK",
        exchange="SMART",
        currency="USD",
    )
    xsp_contract = Contract(
        conId=416904,
        symbol="XSP",
        secType="IND",
        exchange="CBOE",
        currency="USD",
    )

    class _Client:
        def __init__(self) -> None:
            self.requests = {}

        async def qualify_proxy_contracts(self, contract):
            return [spy_contract if contract.secType == "STK" else xsp_contract]

        async def historical_bars_ohlcv(self, contract, **kwargs):
            self.requests[contract.symbol] = kwargs
            rth = [
                Bar(
                    datetime(2026, 7, 27, 9, 30) + timedelta(minutes=5 * index),
                    100.0 + index * 0.1,
                    100.2 + index * 0.1,
                    99.8 + index * 0.1,
                    100.1 + index * 0.1,
                    1.0,
                )
                for index in range(12)
            ]
            rth.append(Bar(datetime(2026, 7, 27, 15, 55), 102, 102.2, 101.8, 102.1, 1))
            if contract.symbol == "XSP":
                return rth
            return [
                *rth,
                *[
                    Bar(
                        datetime(2026, 7, 27, 20, 15) + timedelta(minutes=5 * index),
                        102.0 + index * 0.1,
                        102.2 + index * 0.1,
                        101.8 + index * 0.1,
                        102.1 + index * 0.1,
                        1.0,
                    )
                    for index in range(3)
                ],
            ]

        def last_historical_request(self, contract):
            return {"status": "ok", "symbol": contract.symbol}

    ledger = LiveCalibrationLedger(tmp_path / "v2-forward.jsonl")
    run_start = datetime(2026, 7, 27, 20, 15, tzinfo=ET_ZONE)
    preflight_at = datetime(2026, 7, 27, 17, 5, tzinfo=ET_ZONE)
    client = _Client()
    asyncio.run(
        advance_xsp_opening_edge_v2_from_ibkr(
            ledger,
            client=client,
            observed_at=preflight_at,
            run_started_at=run_start,
            recorded_at=preflight_at,
        )
    )
    observed_at = datetime(2026, 7, 27, 20, 31, tzinfo=ET_ZONE)
    receipt = asyncio.run(
        advance_xsp_opening_edge_v2_from_ibkr(
            ledger,
            client=client,
            observed_at=observed_at,
            run_started_at=run_start,
            recorded_at=observed_at,
        )
    )

    assert client.requests["SPY"] == {
        "duration_str": XSP_OPENING_EDGE_V2_HISTORY_DURATION,
        "bar_size": "5 mins",
        "use_rth": False,
        "what_to_show": "TRADES",
        "cache_ttl_sec": 0.0,
    }
    assert client.requests["XSP"]["use_rth"] is True
    assert receipt["evaluation_status"] == "EVALUATED"
    assert receipt["freshness_ok"] is True
    assert receipt["rth_signal_source"] == "XSP"
    assert receipt["spy_contract"] == {
        "con_id": 756733,
        "symbol": "SPY",
        "sec_type": "STK",
        "exchange": "SMART",
        "currency": "USD",
    }
    paired = receipt["paired_equity"]
    assert paired["rth_signal_source"] == "XSP"
    assert paired["gth_signal_source"] == ("prior_xsp_close_anchored_spy_returns")
    assert paired["gth_signal_bars"] == 3
    assert paired["execution_symbol"] == "SPY"
    assert paired["execution_eligibility"]["eligible"] is False
    assert {profile["order_authority"] for profile in paired["profiles"].values()} == {
        "none"
    }
    checkpoints = [row for row in ledger.records() if row["kind"] == "checkpoint"]
    assert [row["status"] for row in checkpoints] == ["CLOSED", "EVALUATED"]
    assert all(
        row["evidence"]["run_started_at_utc"] == "2026-07-28T00:15:00+00:00"
        for row in checkpoints
    )


@pytest.mark.parametrize(
    ("xsp_qualified", "expected_error"),
    (
        (False, "qualification_unavailable"),
        (True, "missing_completed_anchor"),
    ),
)
def test_opening_edge_v2_observer_fails_closed_without_exact_xsp_anchor(
    tmp_path,
    xsp_qualified,
    expected_error,
) -> None:
    spy_contract = Contract(
        conId=756733,
        symbol="SPY",
        secType="STK",
        exchange="SMART",
        currency="USD",
    )
    xsp_contract = Contract(
        conId=416904,
        symbol="XSP",
        secType="IND",
        exchange="CBOE",
        currency="USD",
    )

    class _Client:
        async def qualify_proxy_contracts(self, contract):
            if contract.secType == "STK":
                return [spy_contract]
            return [xsp_contract] if xsp_qualified else []

        async def historical_bars_ohlcv(self, _contract, **_kwargs):
            gth = [
                Bar(
                    datetime(2026, 7, 27, 20, 15) + timedelta(minutes=5 * index),
                    100.0,
                    100.2,
                    99.8,
                    100.0,
                    1.0,
                )
                for index in range(4)
            ]
            rth = [
                Bar(
                    datetime(2026, 7, 28, 9, 30) + timedelta(minutes=5 * index),
                    101.0,
                    101.2,
                    100.8,
                    101.0,
                    1.0,
                )
                for index in range(2)
            ]
            return [*gth, *rth]

        def last_historical_request(self, contract):
            return {"status": "ok", "symbol": contract.symbol}

    ledger = LiveCalibrationLedger(tmp_path / "v2-rth-fallback.jsonl")
    client = _Client()
    run_start = datetime(2026, 7, 27, 20, 15, tzinfo=ET_ZONE)
    preflight_at = datetime(2026, 7, 27, 17, 5, tzinfo=ET_ZONE)
    asyncio.run(
        advance_xsp_opening_edge_v2_from_ibkr(
            ledger,
            client=client,
            observed_at=preflight_at,
            run_started_at=run_start,
            recorded_at=preflight_at,
        )
    )
    observed_at = datetime(2026, 7, 28, 9, 41, tzinfo=ET_ZONE)
    receipt = asyncio.run(
        advance_xsp_opening_edge_v2_from_ibkr(
            ledger,
            client=client,
            observed_at=observed_at,
            run_started_at=run_start,
            recorded_at=observed_at,
        )
    )

    assert receipt["paired_equity"] is None
    assert receipt["rth_signal_source"] == ("XSP" if xsp_qualified else None)
    assert receipt["evaluation_status"] == "NO_DATA"
    assert receipt["freshness_ok"] is False
    checkpoint = list(ledger.records())[-1]
    assert checkpoint["evidence"]["xsp_error"] == expected_error
    assert checkpoint["evidence"]["rth_provenance_fresh"] is xsp_qualified
    assert checkpoint["evidence"]["order_authority"] == "none"


def test_shadow_history_window_keeps_prior_rth_warmup_across_weekend() -> None:
    observed = datetime(2026, 7, 27, 9, 37, tzinfo=ET_ZONE)
    start, end = duration_window_et(
        XSP_DIRECTIONAL_HISTORY_DURATION,
        end=observed,
    )

    assert end == observed.replace(tzinfo=None)
    assert start <= datetime(2026, 7, 24, 9, 30)


def test_shadow_history_covers_the_complete_run_at_the_week_deadline() -> None:
    deadline = datetime(2026, 8, 3, 9, 37, tzinfo=ET_ZONE)
    start, end = duration_window_et(
        XSP_DIRECTIONAL_HISTORY_DURATION,
        end=deadline,
    )

    assert end == deadline.replace(tzinfo=None)
    assert start <= datetime(2026, 7, 27, 9, 30)


def _profitability_policy(
    *,
    strategy_id: str = "xsp.directional.alpha",
    run_id: str = "run-20260727",
) -> XspProfitabilityPolicy:
    return XspProfitabilityPolicy(
        run_id=run_id,
        strategy_id=strategy_id,
        strategy_version="xsp.directional.alpha.v1",
        config_fingerprint="config-frozen",
        capital_sleeve="xsp-directional-unit",
        max_drawdown_points=XSP_DIRECTIONAL_SHADOW_POLICY["max_drawdown_points"],
        max_session_loss_points=XSP_DIRECTIONAL_SHADOW_POLICY[
            "max_session_loss_points"
        ],
        minimum_week_closed_trades=XSP_DIRECTIONAL_SHADOW_POLICY[
            "minimum_week_closed_trades"
        ],
        maximum_top_five_win_share=XSP_DIRECTIONAL_SHADOW_POLICY[
            "maximum_top_five_win_share"
        ],
        slot_tolerance_seconds=XSP_DIRECTIONAL_SHADOW_POLICY["slot_tolerance_seconds"],
    )


def _append_selected_session(
    ledger: LiveCalibrationLedger,
    *,
    policy: XspProfitabilityPolicy,
    day: date,
    cumulative_gross: float,
    cumulative_costs: float,
    closed_trades: int,
    gross_wins: float,
    top_five_wins: float,
    omit_slot: int | None = None,
    session_gross: float = 3.0,
    session_cost: float = 0.5,
    run_started: datetime | None = None,
    session: str = "RTH",
    session_rollup_gross: float | None = None,
    owned_from: datetime | None = None,
    checkpoint_delay: timedelta = timedelta(0),
    recording_delay: timedelta = timedelta(0),
) -> tuple[float, float, int, float, float]:
    from tradebot.engines.market import xsp_rth_evaluation_slots

    cash = policy.equity_schema == SELECTED_CASH_EQUITY_SCHEMA
    evidence_key = "selected_cash_equity" if cash else "selected_equity"
    suffix = "usd" if cash else "points"
    slots = xsp_rth_evaluation_slots(day)
    run_started = run_started or datetime(2026, 7, 27, 9, 37, tzinfo=ET_ZONE)
    for index, slot in enumerate(slots):
        if index == omit_slot or slot < (owned_from or run_started):
            continue
        last = index == len(slots) - 1
        reported_gross = (
            session_gross if session_rollup_gross is None else session_rollup_gross
        )
        gross = cumulative_gross + (session_gross if last else 0.0)
        costs = cumulative_costs + (session_cost if last else 0.0)
        net = gross - costs
        trades = closed_trades + (2 if last else 0)
        win, top_win = max(0.0, session_gross), round(max(0.0, session_gross) * 0.4, 10)
        wins = gross_wins + (win if last else 0.0)
        top_five = top_five_wins + (top_win if last else 0.0)
        evaluation_at = slot + checkpoint_delay
        ledger.checkpoint(
            evaluation_as_of=evaluation_at,
            strategy_id=policy.strategy_id,
            strategy_version=policy.strategy_version,
            trading_date=day.isoformat(),
            session=session,
            status="EVALUATED",
            evidence={
                evidence_key: {
                    "schema": policy.equity_schema,
                    "run_id": policy.run_id,
                    "run_started_at_utc": run_started.isoformat(),
                    "config_fingerprint": policy.config_fingerprint,
                    "capital_sleeve": policy.capital_sleeve,
                    "unit": policy.unit,
                    f"cumulative_gross_{suffix}": gross,
                    f"cumulative_cost_{suffix}": costs,
                    f"cumulative_net_{suffix}": net,
                    f"cumulative_realized_net_{suffix}": net,
                    f"open_mark_{suffix}": 0.0,
                    f"session_gross_{suffix}": reported_gross if last else 0.0,
                    f"session_cost_{suffix}": session_cost if last else 0.0,
                    f"session_net_{suffix}": (
                        reported_gross - session_cost if last else 0.0
                    ),
                    "closed_trades": trades,
                    f"gross_wins_{suffix}": wins,
                    f"top_five_gross_wins_{suffix}": top_five,
                    "reconciled": True,
                    "attribution_complete": True,
                    "safety_breaches": [],
                },
                "order_authority": "none",
            },
            recorded_at=evaluation_at + recording_delay,
        )
    return (
        cumulative_gross + session_gross,
        cumulative_costs + session_cost,
        closed_trades + 2,
        gross_wins + win,
        top_five_wins + top_win,
    )


def _append_observer_session(
    ledger: LiveCalibrationLedger,
    day: date,
    *,
    omit_slot: int | None = None,
    session: str = "RTH",
) -> None:
    from tradebot.engines.market import xsp_rth_evaluation_slots

    for index, slot in enumerate(xsp_rth_evaluation_slots(day)):
        if index == omit_slot:
            continue
        ledger.checkpoint(
            evaluation_as_of=slot,
            strategy_id="NO_TRADE",
            strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
            trading_date=day.isoformat(),
            session=session,
            status="EVALUATED",
            evidence={"cash_history_fresh": True, "order_authority": "none"},
            recorded_at=slot,
        )


def _news_snapshot(signal_at: datetime) -> dict[str, object]:
    stamp = signal_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signal = {
        "direction": -1,
        "impact": 74,
        "confidence": 0.9,
        "horizon_hours": 4,
        "change": "unchanged",
        "mechanism": "Fresh causal risk pressure.",
        "calibration": "Cross-source evidence remains active.",
        "drivers": ["hormuz-risk"],
    }
    return {
        "schema": NEWS_SCHEMA,
        "score_version": NEWS_SCORE_VERSION,
        "run_status": "published",
        "signal_as_of_utc": stamp,
        "snapshot_as_of_utc": stamp,
        "analysis": {"assets": {"XSP": signal, "MCL": signal}},
    }


def _option_snapshot(
    ts: datetime,
    *,
    parity_value: float = 100.05,
    session: str = "RTH",
    option_half_spread: float = 0.05,
    quote_age_seconds: float = 5.0,
) -> QuoteSnapshot:
    def contract(con_id: int, strike: float, right: str) -> QuoteContract:
        call_mid = 2.0
        midpoint = call_mid if right == "C" else strike + call_mid - parity_value
        return QuoteContract(
            con_id=con_id,
            sec_type="OPT",
            symbol="XSP",
            local_symbol=f"XSP-{con_id}",
            exchange="SMART",
            currency="USD",
            expiry="20260727",
            strike=strike,
            right=right,
            bid=midpoint - option_half_spread,
            ask=midpoint + option_half_spread,
            model_under_price=parity_value,
            market_data_type=3,
            quote_time=(ts - timedelta(seconds=quote_age_seconds)).isoformat(),
        )

    return QuoteSnapshot(
        ts=ts.isoformat(),
        md_type=3,
        symbol="XSP",
        underlying=QuoteContract(
            con_id=137851301,
            sec_type="IND",
            symbol="XSP",
            local_symbol="XSP",
            exchange="CBOE",
            currency="USD",
            bid=100.0,
            ask=100.1,
            market_data_type=3,
            quote_time=(ts - timedelta(seconds=5)).isoformat(),
        ),
        options=[
            contract(100 + offset * 2 + (right == "P"), strike, right)
            for offset, strike in enumerate((99.0, 100.0, 101.0))
            for right in ("C", "P")
        ],
        errors=[],
        chain_fingerprint="a" * 64,
        target_expiry="20260727",
        session=session,
    )


def _forecast(
    ledger: LiveCalibrationLedger,
    *,
    decision: str = "NO_TRADE",
    decision_at: datetime = NOW,
    recorded_at: datetime | None = None,
    horizon_minutes: int = 30,
    context: dict[str, object] | None = None,
    counterfactual_direction: str = "UP",
) -> dict[str, object]:
    return ledger.freeze(
        identity={
            "strategy_id": "NO_TRADE",
            "strategy_version": XSP_DIRECTIONAL_OBSERVER_VERSION,
            "decision_as_of_utc": decision_at.isoformat(),
            "tape_fingerprint": "tape-1",
            "config_fingerprint": "config-1",
            "capital_sleeve": "xsp-directional-unit",
        },
        forecast={
            "decision": decision,
            "outcome_not_before_utc": (
                decision_at + timedelta(minutes=horizon_minutes)
            ).isoformat(),
            "pnl_distribution": {"median": 0.0},
            "risk": {"max_loss": 0.0},
            "costs": {"modeled": 0.0},
            "fill_assumptions": {"synthetic_index_unit": True},
        },
        context=context or {"session": "RTH"},
        counterfactuals=[
            {
                "strategy_id": "directional_impulse.observer",
                "decision": counterfactual_direction,
            }
        ],
        gates={"selected_admissible": False},
        recorded_at=recorded_at or decision_at,
    )


def test_forecast_is_content_addressed_and_idempotent(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    first = _forecast(ledger)
    second = _forecast(ledger)

    assert first == second
    with pytest.raises(ValueError, match="different forecast"):
        _forecast(ledger, decision="UP")
    assert ledger.receipt() == {
        "schema": "live_calibration.v1",
        "path": str(tmp_path / "calibration.jsonl"),
        "sha256": ledger.receipt()["sha256"],
        "records": 1,
        "forecasts": 1,
        "results": 0,
        "checkpoints": 0,
        "checkpoint_statuses": {},
        "unsettled": [first["forecast_id"]],
    }
    with pytest.raises(ValueError, match="before its decision evidence"):
        _forecast(
            LiveCalibrationLedger(tmp_path / "predated.jsonl"),
            recorded_at=NOW - timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="at or after its outcome window"):
        _forecast(
            LiveCalibrationLedger(tmp_path / "late.jsonl"),
            recorded_at=NOW + timedelta(minutes=30),
        )


def test_directional_pair_join_requires_exact_frozen_outcome_and_direction(
    tmp_path,
) -> None:
    context = {
        "session": "RTH",
        "evidence_mode": "forward_broker_history",
        "option_parity": {
            "source": "option_nbbo_parity",
            "authority": "observation_only",
            "usable": True,
            "parity_change": {"usable": True, "direction": "up"},
        },
        "fundamental_pressure": {
            "source": "causal_news",
            "authority": "observation_only",
            "usable": True,
            "direction": -1,
            "impact": 74,
            "confidence": 0.9,
        },
    }

    def settled(name: str, *, outcome_minutes: int, direction: str):
        ledger = LiveCalibrationLedger(tmp_path / f"{name}.jsonl")
        forecast = _forecast(
            ledger,
            horizon_minutes=60,
            context=context,
            counterfactual_direction="UP",
        )
        outcome_at = NOW + timedelta(minutes=outcome_minutes)
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": outcome_at.isoformat(),
                "counterfactuals": [
                    {
                        "strategy_id": "directional_impulse.observer",
                        "direction": direction,
                        "net_points": 1.25,
                    }
                ],
            },
            drift={},
            verdict="HOLD",
            settled_at=outcome_at,
        )
        return ledger

    exact = settled("exact", outcome_minutes=60, direction="up")
    assert len(exact.settled_directional_pairs(horizon_minutes=60)) == 1
    assert xsp_option_parity_participation_benchmark(exact)["pairs"] == 1
    assert xsp_fundamental_defensive_benchmark(exact)["pairs"] == 1
    records = [json.loads(line) for line in exact.path.read_text().splitlines()]
    records[-1]["observed"]["counterfactuals"][0]["net_points"] = 999.0
    exact.path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    for consume in (
        lambda: exact.settled_directional_pairs(horizon_minutes=60),
        lambda: xsp_option_parity_participation_benchmark(exact),
        lambda: xsp_fundamental_defensive_benchmark(exact),
        exact.receipt,
    ):
        with pytest.raises(ValueError, match="invalid calibration content address"):
            consume()

    tampered_forecast = settled("tampered-forecast", outcome_minutes=60, direction="up")
    records = [
        json.loads(line) for line in tampered_forecast.path.read_text().splitlines()
    ]
    records[0]["context"]["session"] = "GTH"
    tampered_forecast.path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    with pytest.raises(ValueError, match="invalid calibration content address"):
        tampered_forecast.settled_directional_pairs(horizon_minutes=60)

    for ledger in (
        settled("late", outcome_minutes=65, direction="up"),
        settled("wrong-direction", outcome_minutes=60, direction="down"),
    ):
        assert ledger.settled_directional_pairs(horizon_minutes=60) == []
        assert xsp_option_parity_participation_benchmark(ledger)["pairs"] == 0
        assert xsp_fundamental_defensive_benchmark(ledger)["pairs"] == 0


def test_checkpoint_is_idempotent_and_independent_of_signal_events(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    checkpoint = ledger.checkpoint(
        evaluation_as_of=NOW,
        strategy_id="NO_TRADE",
        strategy_version="xsp.directional-observer.v1",
        trading_date="2026-07-27",
        session="RTH",
        status="evaluated",
        evidence={
            "cash_history_fresh": True,
            "order_authority": "none",
        },
        recorded_at=NOW,
    )

    assert (
        ledger.checkpoint(
            evaluation_as_of=NOW,
            strategy_id="NO_TRADE",
            strategy_version="xsp.directional-observer.v1",
            trading_date="2026-07-27",
            session="RTH",
            status="EVALUATED",
            evidence={
                "cash_history_fresh": True,
                "order_authority": "none",
            },
            recorded_at=NOW,
        )
        == checkpoint
    )
    assert ledger.receipt() == {
        "schema": "live_calibration.v1",
        "path": str(tmp_path / "calibration.jsonl"),
        "sha256": ledger.receipt()["sha256"],
        "records": 1,
        "forecasts": 0,
        "results": 0,
        "checkpoints": 1,
        "checkpoint_statuses": {"EVALUATED": 1},
        "unsettled": [],
    }
    with pytest.raises(ValueError, match="unsupported calibration checkpoint"):
        ledger.checkpoint(
            evaluation_as_of=NOW,
            strategy_id="NO_TRADE",
            strategy_version="xsp.directional-observer.v1",
            trading_date="2026-07-27",
            session="RTH",
            status="missing",
            evidence={},
            recorded_at=NOW,
        )
    with pytest.raises(ValueError, match="before its evaluation time"):
        ledger.checkpoint(
            evaluation_as_of=NOW,
            strategy_id="NO_TRADE",
            strategy_version="xsp.directional-observer.v1",
            trading_date="2026-07-27",
            session="RTH",
            status="EVALUATED",
            evidence={},
            recorded_at=NOW - timedelta(seconds=1),
        )


def test_profitability_clock_rejects_no_trade_and_observer_economics(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    _forecast(ledger)
    ledger.checkpoint(
        evaluation_as_of=NOW,
        strategy_id="NO_TRADE",
        strategy_version="xsp.directional-observer.v1",
        trading_date="2026-07-27",
        session="RTH",
        status="EVALUATED",
        evidence={"cash_history_fresh": True},
        recorded_at=NOW,
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=_profitability_policy(strategy_id="NO_TRADE"),
        as_of=NOW,
    )

    assert receipt["status"] == "NOT_STARTED"
    assert receipt["reasons"] == ["no_selected_strategy"]
    assert not any(row["passed"] for row in receipt["milestones"].values())


def test_profitability_clock_requires_exact_continuous_selected_coverage(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    values = (0.0, 0.0, 0, 0.0, 0.0)
    values = _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 27),
        cumulative_gross=values[0],
        cumulative_costs=values[1],
        closed_trades=values[2],
        gross_wins=values[3],
        top_five_wins=values[4],
        omit_slot=20,
    )
    first_tuesday = datetime(2026, 7, 28, 9, 37, tzinfo=ET_ZONE)
    ledger.checkpoint(
        evaluation_as_of=first_tuesday,
        strategy_id=policy.strategy_id,
        strategy_version=policy.strategy_version,
        trading_date="2026-07-28",
        session="RTH",
        status="EVALUATED",
        evidence={
            "selected_equity": {
                "schema": SELECTED_EQUITY_SCHEMA,
                "run_id": policy.run_id,
                "run_started_at_utc": datetime(
                    2026, 7, 27, 9, 37, tzinfo=ET_ZONE
                ).isoformat(),
                "config_fingerprint": policy.config_fingerprint,
                "capital_sleeve": policy.capital_sleeve,
                "unit": "$1_per_XSP_point",
                "cumulative_gross_points": values[0],
                "cumulative_cost_points": values[1],
                "cumulative_net_points": values[0] - values[1],
                "cumulative_realized_net_points": values[0] - values[1],
                "open_mark_points": 0.0,
                "session_gross_points": 0.0,
                "session_cost_points": 0.0,
                "session_net_points": 0.0,
                "closed_trades": values[2],
                "gross_wins_points": values[3],
                "top_five_gross_wins_points": values[4],
                "reconciled": True,
                "attribution_complete": True,
                "safety_breaches": [],
            }
        },
        recorded_at=first_tuesday,
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=first_tuesday,
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert receipt["clock"]["coverage_broken"] is True
    assert receipt["sessions"][0]["evaluated_slots"] == 77
    assert len(receipt["sessions"][0]["missing_slots"]) == 1
    assert "incomplete_session_coverage" in receipt["reasons"]
    assert not receipt["milestones"]["24h"]["passed"]


def test_profitability_clock_proves_only_reconciled_net_week(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    values = (0.0, 0.0, 0, 0.0, 0.0)
    for day in (
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 3),
    ):
        values = _append_selected_session(
            ledger,
            policy=policy,
            day=day,
            cumulative_gross=values[0],
            cumulative_costs=values[1],
            closed_trades=values[2],
            gross_wins=values[3],
            top_five_wins=values[4],
        )
    as_of = datetime(2026, 8, 3, 16, 2, tzinfo=ET_ZONE)

    receipt = ledger.xsp_profitability_receipt(policy=policy, as_of=as_of)

    assert receipt["status"] == "PASSED"
    assert receipt["clock"]["complete_sessions"] == 6
    assert receipt["economics"] == {
        "unit": "$1_per_XSP_point",
        "gross_points": 18.0,
        "cost_points": 3.0,
        "net_points": 15.0,
        "realized_net_points": 15.0,
        "open_mark_points": 0.0,
        "maximum_drawdown_points": 0.0,
        "worst_session_points": 2.5,
        "closed_trades": 12,
        "gross_wins_points": 18.0,
        "top_five_gross_wins_points": 7.2,
        "top_five_win_share": pytest.approx(0.4),
    }
    assert all(row["passed"] for row in receipt["milestones"].values())
    assert receipt["milestones"]["24h"]["complete_sessions"] == 1
    assert receipt["milestones"]["24h"]["economics"]["net_points"] == 2.5
    assert receipt["milestones"]["48h"]["complete_sessions"] == 2
    assert receipt["milestones"]["48h"]["economics"]["net_points"] == 5.0
    assert receipt["milestones"]["five_session_week"]["complete_sessions"] == 5
    assert receipt["milestones"]["five_session_week"]["economics"]["net_points"] == 12.5
    assert receipt["milestones"]["five_session_week"]["reasons"] == []

    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    original_last = json.loads(json.dumps(rows[-1]))
    rows[-1]["evidence"]["selected_equity"]["cumulative_net_points"] += 1.0
    ledger.path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="invalid calibration content address"):
        ledger.xsp_profitability_receipt(policy=policy, as_of=as_of)
    rows[-1] = original_last
    rows[-1]["recorded_at_utc"] = (
        datetime.fromisoformat(rows[-1]["recorded_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    ledger.path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="invalid calibration content address"):
        ledger.xsp_profitability_receipt(policy=policy, as_of=as_of)


def test_cash_profitability_clock_proves_usd_24h_48h_and_week(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = XspProfitabilityPolicy(
        run_id="v3-selection",
        strategy_id="xsp.opening-edge-v3-regime-harmony-24x5.v1",
        strategy_version="xsp.opening-edge-v3-upro-spxu-execution.v1",
        config_fingerprint="v3-selection",
        capital_sleeve="xsp-upro-spxu-rth-cash",
        max_drawdown_points=135.0,
        max_session_loss_points=67.5,
        minimum_week_closed_trades=2,
        maximum_top_five_win_share=0.5,
        unit="USD",
        equity_schema=SELECTED_CASH_EQUITY_SCHEMA,
    )
    values = (0.0, 0.0, 0, 0.0, 0.0)
    for day in (
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 3),
    ):
        values = _append_selected_session(
            ledger,
            policy=policy,
            day=day,
            cumulative_gross=values[0],
            cumulative_costs=values[1],
            closed_trades=values[2],
            gross_wins=values[3],
            top_five_wins=values[4],
            session_gross=12.0,
            session_cost=2.0,
        )
    as_of = datetime(2026, 8, 3, 16, 2, tzinfo=ET_ZONE)

    receipt = ledger.xsp_profitability_receipt(policy=policy, as_of=as_of)

    assert receipt["status"] == "PASSED"
    assert receipt["policy"]["unit"] == "USD"
    assert receipt["policy"]["equity_schema"] == SELECTED_CASH_EQUITY_SCHEMA
    assert receipt["clock"]["coverage_broken"] is False
    assert receipt["clock"]["complete_sessions"] == 6
    assert receipt["economics"] == {
        "unit": "USD",
        "gross_usd": 72.0,
        "cost_usd": 12.0,
        "net_usd": 60.0,
        "realized_net_usd": 60.0,
        "open_mark_usd": 0.0,
        "maximum_drawdown_usd": 0.0,
        "worst_session_usd": 10.0,
        "closed_trades": 12,
        "gross_wins_usd": 72.0,
        "top_five_gross_wins_usd": 28.8,
        "top_five_win_share": pytest.approx(0.4),
    }
    assert all(row["passed"] for row in receipt["milestones"].values())
    assert receipt["milestones"]["24h"]["economics"]["net_usd"] == 10.0
    assert receipt["milestones"]["48h"]["economics"]["net_usd"] == 20.0
    weekly = receipt["milestones"]["five_session_week"]
    assert weekly["economics"]["net_usd"] == 50.0
    assert weekly["economics"]["closed_trades"] == 10
    assert weekly["economics"]["top_five_win_share"] == pytest.approx(0.4)
    assert weekly["reasons"] == []


def test_profitability_milestones_cannot_rewrite_earlier_losses(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    values = (0.0, 0.0, 0, 0.0, 0.0)
    for index, day in enumerate(
        (
            date(2026, 7, 27),
            date(2026, 7, 28),
            date(2026, 7, 29),
            date(2026, 7, 30),
            date(2026, 7, 31),
            date(2026, 8, 3),
        )
    ):
        values = _append_selected_session(
            ledger,
            policy=policy,
            day=day,
            cumulative_gross=values[0],
            cumulative_costs=values[1],
            closed_trades=values[2],
            gross_wins=values[3],
            top_five_wins=values[4],
            session_gross=-4.0 if index == 0 else 3.0,
        )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 8, 3, 16, 2, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "ACTIVE"
    assert receipt["economics"]["net_points"] == 8.0
    assert receipt["milestones"]["24h"]["economics"]["net_points"] == -4.5
    assert receipt["milestones"]["24h"]["reasons"] == ["net_not_positive"]
    assert receipt["milestones"]["48h"]["economics"]["net_points"] == -2.0
    assert receipt["milestones"]["48h"]["reasons"] == ["net_not_positive"]
    assert receipt["milestones"]["five_session_week"]["passed"] is True
    assert receipt["milestones"]["five_session_week"]["economics"]["net_points"] == 5.5


def test_profitability_clock_starts_at_first_owned_mid_session_slot(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    started = datetime(2026, 7, 27, 12, 0, tzinfo=ET_ZONE)
    values = _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 27),
        cumulative_gross=0.0,
        cumulative_costs=0.0,
        closed_trades=0,
        gross_wins=0.0,
        top_five_wins=0.0,
        run_started=started,
    )
    _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 28),
        cumulative_gross=values[0],
        cumulative_costs=values[1],
        closed_trades=values[2],
        gross_wins=values[3],
        top_five_wins=values[4],
        run_started=started,
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 28, 16, 4, tzinfo=ET_ZONE),
    )

    assert receipt["clock"]["coverage_broken"] is False
    assert receipt["sessions"][0]["complete"] is False
    assert receipt["sessions"][0]["missing_slots"] == []
    assert receipt["sessions"][1]["complete"] is True
    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["milestones"]["24h"]["complete_sessions"] == 1


def test_profitability_prefix_allows_real_checkpoint_jitter(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    delay = timedelta(seconds=10)
    values = _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 27),
        cumulative_gross=0.0,
        cumulative_costs=0.0,
        closed_trades=0,
        gross_wins=0.0,
        top_five_wins=0.0,
        checkpoint_delay=delay,
    )
    _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 28),
        cumulative_gross=values[0],
        cumulative_costs=values[1],
        closed_trades=values[2],
        gross_wins=values[3],
        top_five_wins=values[4],
        checkpoint_delay=delay,
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 28, 9, 39, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "ACTIVE"
    assert receipt["reasons"] == []
    assert receipt["clock"]["complete_sessions"] == 1
    milestone = receipt["milestones"]["24h"]
    assert milestone["passed"] is True
    assert (
        milestone["economic_window_end_utc"]
        == datetime(2026, 7, 28, 9, 37, tzinfo=ET_ZONE)
        .astimezone(timezone.utc)
        .isoformat()
    )
    assert (
        milestone["evidence_as_of_utc"]
        == datetime(2026, 7, 28, 9, 38, 30, tzinfo=ET_ZONE)
        .astimezone(timezone.utc)
        .isoformat()
    )


def test_profitability_rejects_checkpoint_recorded_after_slot_tolerance(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 27),
        cumulative_gross=0.0,
        cumulative_costs=0.0,
        closed_trades=0,
        gross_wins=0.0,
        top_five_wins=0.0,
        recording_delay=timedelta(seconds=policy.slot_tolerance_seconds + 1),
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 27, 16, 4, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert receipt["clock"]["coverage_broken"] is True
    assert receipt["clock"]["complete_sessions"] == 0
    assert (
        ledger.complete_xsp_checkpoint_sessions(
            strategy_id=policy.strategy_id,
            strategy_version=policy.strategy_version,
            slot_tolerance_seconds=policy.slot_tolerance_seconds,
        )
        == ()
    )


def test_profitability_later_gap_does_not_erase_anchored_milestones(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    values = (0.0, 0.0, 0, 0.0, 0.0)
    for day, omit_slot in (
        (date(2026, 7, 27), None),
        (date(2026, 7, 28), None),
        (date(2026, 7, 29), 20),
    ):
        values = _append_selected_session(
            ledger,
            policy=policy,
            day=day,
            cumulative_gross=values[0],
            cumulative_costs=values[1],
            closed_trades=values[2],
            gross_wins=values[3],
            top_five_wins=values[4],
            omit_slot=omit_slot,
        )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 29, 16, 2, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert receipt["clock"]["coverage_broken"] is True
    assert receipt["milestones"]["24h"]["passed"] is True
    assert receipt["milestones"]["48h"]["passed"] is True
    assert receipt["milestones"]["five_session_week"]["passed"] is False


def test_profitability_retry_ignores_prior_run_with_same_strategy(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    old_policy = _profitability_policy(run_id="run-old")
    _append_selected_session(
        ledger,
        policy=old_policy,
        day=date(2026, 7, 27),
        cumulative_gross=0.0,
        cumulative_costs=0.0,
        closed_trades=0,
        gross_wins=0.0,
        top_five_wins=0.0,
        session_gross=-4.0,
    )
    policy = _profitability_policy(run_id="run-new")
    started = datetime(2026, 7, 28, 9, 37, tzinfo=ET_ZONE)
    values = (0.0, 0.0, 0, 0.0, 0.0)
    for day in (date(2026, 7, 28), date(2026, 7, 29)):
        values = _append_selected_session(
            ledger,
            policy=policy,
            day=day,
            cumulative_gross=values[0],
            cumulative_costs=values[1],
            closed_trades=values[2],
            gross_wins=values[3],
            top_five_wins=values[4],
            run_started=started,
        )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 29, 16, 2, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "ACTIVE"
    assert receipt["reasons"] == []
    assert receipt["economics"]["net_points"] == 5.0
    assert receipt["milestones"]["24h"]["passed"] is True


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"session": "GTH"}, "incomplete_session_coverage"),
        ({"session_rollup_gross": 2.0}, "inconsistent_session_rollup"),
        (
            {
                "run_started": datetime(2026, 7, 28, 9, 37, tzinfo=ET_ZONE),
                "owned_from": datetime(2026, 7, 27, 9, 37, tzinfo=ET_ZONE),
            },
            "checkpoint_predates_run",
        ),
    ],
)
def test_profitability_rejects_wrong_session_and_false_rollup(
    tmp_path,
    kwargs,
    reason,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    policy = _profitability_policy()
    _append_selected_session(
        ledger,
        policy=policy,
        day=date(2026, 7, 27),
        cumulative_gross=0.0,
        cumulative_costs=0.0,
        closed_trades=0,
        gross_wins=0.0,
        top_five_wins=0.0,
        **kwargs,
    )

    receipt = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=datetime(2026, 7, 27, 16, 2, tzinfo=ET_ZONE),
    )

    assert receipt["status"] == "INVALID_EVIDENCE"
    assert reason in receipt["reasons"]
    assert not any(row["passed"] for row in receipt["milestones"].values())


def test_shadow_systemd_cadence_is_bounded_and_runtime_gated() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy/systemd/tradebot-xsp-shadow.service").read_text()
    timer = (root / "deploy/systemd/tradebot-xsp-shadow.timer").read_text()
    selected_live = (
        root / "deploy/systemd/tradebot-xsp-shadow-selected-live.conf"
    ).read_text()
    selector = (root / "deploy/systemd/tradebot-xsp-select-live").read_text()

    assert (
        "ExecCondition=/usr/bin/test -x %h/.local/share/tradebot/venv/bin/python"
    ) in service
    assert (
        "ExecStart=/usr/bin/env python3 -m tradebot.research.xsp_shadow "
        "--mode opening-edge-v3"
    ) in service
    assert "TimeoutStartSec=8min" in service
    assert "Environment=IBKR_READONLY=1" in service
    assert "Environment=IBKR_READONLY=0" in selected_live
    assert "NoNewPrivileges=true" in service
    assert "Sun *-*-* 20:22/5:00 America/New_York" in timer
    assert "Sun *-*-* 21..23:02/5:00 America/New_York" in timer
    assert "Mon..Thu *-*-* 20:22/5:00 America/New_York" in timer
    assert "Mon..Thu *-*-* 21..23:02/5:00 America/New_York" in timer
    assert "20..23:22/5:00" not in timer
    assert "Mon..Fri *-*-* 00..08:02/5:00 America/New_York" in timer
    assert (
        "Mon..Fri *-*-* 09:02,07,12,17,22,27,37,42,47,52,57:00 America/New_York"
    ) in timer
    assert "Mon..Fri *-*-* 10..15:02/5:00 America/New_York" in timer
    assert "Mon..Fri *-*-* 16:02,07,12,17:00 America/New_York" in timer
    assert timer.count("OnCalendar=") == 8
    assert "Persistent=false" in timer
    assert "RandomizedDelaySec=0" in timer
    first_start = selector.index('start "${service}"')
    selected_start = selector.index('start "${service}"', first_start + 1)
    assert selector.index('disable --now "${timer}"') < first_start
    assert first_start < selector.index("--freeze-selected-transport")
    assert selector.index("--freeze-selected-transport") < selector.index(
        "tradebot-xsp-shadow-selected-live.conf"
    )
    assert "--mode opening-edge-v3" in selector
    assert "--transport-preview" in selector
    assert "--transport-cash-receipt" in selector
    assert "--accept-rth-only-cash-scope" in selector
    assert "--transport-ranking" not in selector
    assert "--transport-dwell" not in selector
    assert (
        selector.index("tradebot-xsp-shadow-selected-live.conf")
        < selected_start
        < selector.rindex('enable --now "${timer}"')
    )
    assert "selection_frozen == 0" in selector
    assert "cadence remains disabled pending reconciliation" in selector
    assert "rm " not in selector


@pytest.mark.parametrize(
    "failure_phase", ("none", "before_selection", "after_selection")
)
def test_xsp_live_selection_transaction_is_atomic(
    tmp_path,
    monkeypatch,
    failure_phase,
) -> None:
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    deployed = tmp_path / "systemd"
    fake_bin = tmp_path / "bin"
    runtime = tmp_path / "runtime"
    preview = tmp_path / "preview.json"
    for directory in (
        repo / "deploy/systemd",
        repo / "backtests/xsp",
        repo / "db/calibration",
        deployed,
        fake_bin,
        runtime,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("tradebot-xsp-shadow.service", "tradebot-xsp-shadow.timer"):
        payload = (
            "ExecStart=python -m tradebot.research.xsp_shadow --mode opening-edge-v3\n"
            if name == "tradebot-xsp-shadow.service"
            else f"{name}\n"
        )
        (repo / "deploy/systemd" / name).write_text(payload)
        (deployed / name).write_text(payload)
    (repo / "deploy/systemd/tradebot-xsp-shadow-selected-live.conf").write_text(
        "[Service]\nEnvironment=IBKR_READONLY=0\n"
    )
    (
        repo
        / "backtests/xsp/opening_edge_v3_regime_harmony_cash_receipt.json"
    ).write_text("{}\n")
    preview.write_text("{}\n")

    log = tmp_path / "systemctl.log"
    starts = tmp_path / "service-starts"
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        """#!/bin/sh
echo "$*" >> "$XSP_TEST_SYSTEMCTL_LOG"
case "$*" in
  "--user is-enabled --quiet tradebot-xsp-shadow.timer" | \\
  "--user is-active --quiet tradebot-xsp-shadow.timer")
    exit 0 ;;
  "--user show tradebot-xsp-shadow.service -p ActiveState --value")
    echo inactive; exit 0 ;;
  "--user show tradebot-xsp-shadow.service -p Environment --value")
    echo IBKR_READONLY=0; exit 0 ;;
  "--user show tradebot-xsp-shadow.service -p Result --value")
    echo success; exit 0 ;;
  "--user show tradebot-xsp-shadow.service -p ExecMainStatus --value")
    echo 0; exit 0 ;;
  "--user start tradebot-xsp-shadow.service")
    count=0
    test ! -f "$XSP_TEST_STARTS" || count=$(cat "$XSP_TEST_STARTS")
    count=$((count + 1))
    echo "$count" > "$XSP_TEST_STARTS"
    if test "${XSP_TEST_FAIL_SELECTED:-0}" = 1 && test "$count" -ge 2; then
      exit 7
    fi
    exit 0 ;;
esac
exit 0
"""
    )
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/sh
case "$1" in
  status) exit 0 ;;
  rev-parse) echo deployed-revision; exit 0 ;;
esac
exit 2
"""
    )
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/bin/sh
test "${XSP_TEST_FAIL_SELECTION:-0}" != 1 || exit 8
previous=
for value in "$@"; do
  if test "$previous" = --selected-transport; then
    mkdir -p "$(dirname "$value")"
    printf '{"selection_id":"test"}\\n' > "$value"
  fi
  previous=$value
done
"""
    )
    fake_flock = fake_bin / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n")
    for path in (fake_systemctl, fake_git, fake_python, fake_flock):
        path.chmod(0o755)

    monkeypatch.setenv("TRADEBOT_ROOT", str(repo))
    monkeypatch.setenv("XSP_SYSTEMD_USER_DIR", str(deployed))
    monkeypatch.setenv("XSP_PYTHON", str(fake_python))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("XSP_TEST_SYSTEMCTL_LOG", str(log))
    monkeypatch.setenv("XSP_TEST_STARTS", str(starts))
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    if failure_phase == "before_selection":
        monkeypatch.setenv("XSP_TEST_FAIL_SELECTION", "1")
    if failure_phase == "after_selection":
        monkeypatch.setenv("XSP_TEST_FAIL_SELECTED", "1")

    result = subprocess.run(
        [
            str(root / "deploy/systemd/tradebot-xsp-select-live"),
            "--accept-rth-only-cash-scope",
            str(preview),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    selected = repo / "db/calibration/xsp_selected_live_transport.json"
    dropin = deployed / "tradebot-xsp-shadow.service.d/selected-live.conf"
    calls = log.read_text().splitlines()

    if failure_phase == "none":
        assert result.returncode == 0
        assert selected.exists()
        assert dropin.exists()
        assert calls[-1] == "--user enable --now tradebot-xsp-shadow.timer"
    elif failure_phase == "before_selection":
        assert result.returncode != 0
        assert not selected.exists()
        assert not dropin.exists()
        assert calls[-1] == "--user enable --now tradebot-xsp-shadow.timer"
    else:
        assert result.returncode != 0
        assert selected.exists()
        assert dropin.exists()
        assert calls[-1] == "--user disable --now tradebot-xsp-shadow.timer"
        assert starts.read_text().strip() == "3"


def test_xsp_live_selection_requires_explicit_rth_scope(tmp_path) -> None:
    result = subprocess.run(
        [
            str(
                Path(__file__).resolve().parents[1]
                / "deploy/systemd/tradebot-xsp-select-live"
            ),
            str(tmp_path / "preview.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 64
    assert "--accept-rth-only-cash-scope" in result.stderr


def test_result_settles_one_frozen_forecast_once(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "calibration.jsonl")
    forecast = _forecast(ledger)
    result = ledger.settle(
        forecast_id=str(forecast["forecast_id"]),
        observed={
            "outcome_as_of_utc": (NOW + timedelta(hours=1)).isoformat(),
            "shadow_pnl": 0.0,
            "package_pnl": None,
            "account_pnl": None,
        },
        drift={"decision": "none", "economic": 0.0},
        verdict="hold",
        settled_at=NOW + timedelta(hours=1),
    )

    assert result["verdict"] == "HOLD"
    assert (
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": (NOW + timedelta(hours=1)).isoformat(),
                "shadow_pnl": 0.0,
                "package_pnl": None,
                "account_pnl": None,
            },
            drift={"decision": "none", "economic": 0.0},
            verdict="hold",
            settled_at=NOW + timedelta(hours=1),
        )
        == result
    )
    assert ledger.receipt()["unsettled"] == []
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": (NOW + timedelta(hours=2)).isoformat(),
                "shadow_pnl": 1.0,
            },
            drift={},
            verdict="PROMOTE",
            settled_at=NOW + timedelta(hours=2),
        )


def test_append_repairs_an_interrupted_tail(tmp_path) -> None:
    path = tmp_path / "calibration.jsonl"
    ledger = LiveCalibrationLedger(path)
    forecast = _forecast(ledger)
    with path.open("ab") as handle:
        handle.write(b'{"schema":"live_calibration.v1"')

    assert _forecast(ledger) == forecast
    assert path.read_bytes().endswith(b"\n")
    assert list(ledger.records()) == [forecast]


def test_directional_shadow_keeps_no_trade_selected_and_settles_counterfactual(
    tmp_path,
) -> None:
    closes = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        104.0,
        102.0,
        100.0,
        99.0,
        98.0,
        97.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
        119.0,
        120.0,
    ]
    bars = [
        Bar(
            NOW.replace(tzinfo=None) + timedelta(minutes=5 * index),
            close,
            close + 0.2,
            close - 0.2,
            close,
            0.0,
        )
        for index, close in enumerate(closes)
    ]
    ledger = LiveCalibrationLedger(tmp_path / "xsp-shadow.jsonl")

    receipt = replay_xsp_directional_shadow(ledger, bars)
    records = list(ledger.records())
    forecasts = [row for row in records if row["kind"] == "forecast"]
    results = [row for row in records if row["kind"] == "result"]

    assert forecasts
    assert results
    assert receipt["forecasts"] >= receipt["results"]
    assert all(row["forecast"]["decision"] == "NO_TRADE" for row in forecasts)
    assert all(not row["context"]["entry_control"]["blocked_by"] for row in forecasts)
    assert all(row["observed"]["shadow_pnl"] == 0.0 for row in results)
    assert all(row["observed"]["package_pnl"] is None for row in results)
    assert all(
        row["drift"]["execution"] == "synthetic_observer_no_broker_fill"
        for row in results
    )

    extended = bars + [
        Bar(
            bars[-1].ts + timedelta(minutes=5),
            120.0,
            120.2,
            119.8,
            120.0,
            0.0,
        )
    ]
    second = LiveCalibrationLedger(tmp_path / "xsp-shadow-extended.jsonl")
    replay_xsp_directional_shadow(second, extended)
    assert (
        next(ledger.records())["identity"]["tape_fingerprint"]
        == next(second.records())["identity"]["tape_fingerprint"]
    )

    decision_at = datetime.fromisoformat(
        str(forecasts[0]["identity"]["decision_as_of_utc"])
    )
    paired = LiveCalibrationLedger(tmp_path / "xsp-shadow-paired.jsonl")
    replay_xsp_directional_shadow(
        paired,
        bars,
        option_snapshots=[_option_snapshot(decision_at - timedelta(minutes=1))],
    )
    paired_forecast = next(paired.records())
    assert paired_forecast["context"]["option_parity"]["usable"] is True
    assert paired_forecast["context"]["option_parity"]["authority"] == (
        "observation_only"
    )
    assert paired_forecast["context"]["option_parity"]["decision_lag_seconds"] == 60


def test_option_context_never_uses_a_future_or_stale_snapshot() -> None:
    decision_at = NOW + timedelta(minutes=10)
    causal = _option_snapshot(decision_at - timedelta(minutes=1))
    future = _option_snapshot(decision_at + timedelta(seconds=1))

    context = xsp_option_context_at([causal, future], decision_at=decision_at)
    stale = xsp_option_context_at(
        [_option_snapshot(decision_at - timedelta(minutes=8))],
        decision_at=decision_at,
    )

    assert context["ts"] == causal.ts
    assert context["decision_lag_seconds"] == 60
    assert stale["usable"] is False
    assert stale["reasons"] == ("no_causal_same_session_snapshot",)


def test_option_context_freezes_only_prior_causal_parity_movement() -> None:
    decision_at = NOW + timedelta(minutes=10)
    old = _option_snapshot(
        decision_at - timedelta(minutes=16),
        parity_value=99.75,
    )
    prior = _option_snapshot(
        decision_at - timedelta(minutes=6),
        parity_value=100.00,
    )
    latest = _option_snapshot(
        decision_at - timedelta(minutes=1),
        parity_value=100.25,
    )
    future = _option_snapshot(
        decision_at + timedelta(seconds=1),
        parity_value=150.00,
    )

    context = xsp_option_context_at(
        [future, latest, old, prior],
        decision_at=decision_at,
    )
    prior_observation = option_parity_observation(prior)

    assert context["value"] == pytest.approx(100.25)
    assert context["parity_change"] == {
        "usable": True,
        "reasons": (),
        "prior_ts": prior.ts,
        "prior_chain_fingerprint": "a" * 64,
        "prior_pairs": prior_observation["pairs"],
        "prior_dispersion_points": prior_observation["dispersion_points"],
        "prior_median_relative_spread": prior_observation["median_relative_spread"],
        "prior_strikes": prior_observation["strikes"],
        "prior_max_age_seconds": prior_observation["max_age_seconds"],
        "prior_reference_value": prior_observation["anchor"],
        "prior_market_data_types": prior_observation["market_data_types"],
        "prior_anchor_source": prior_observation["anchor_source"],
        "prior_value": pytest.approx(100.00),
        "interval_seconds": 300.0,
        "value_change_points": pytest.approx(0.25),
        "value_velocity_points_per_minute": pytest.approx(0.05),
        "direction": "up",
    }


def test_option_context_freezes_exact_causal_gth_preopen_path() -> None:
    decision_at = NOW
    boundary = decision_at.replace(minute=30) - timedelta(minutes=10)
    context = xsp_option_context_at(
        [
            _option_snapshot(decision_at - timedelta(minutes=1)),
            _option_snapshot(boundary, parity_value=100.60, session="GTH"),
            _option_snapshot(
                boundary - timedelta(minutes=120),
                parity_value=100.40,
                session="GTH",
            ),
            _option_snapshot(
                boundary - timedelta(minutes=240),
                parity_value=100.70,
                session="GTH",
            ),
            _option_snapshot(
                boundary - timedelta(minutes=360),
                parity_value=100.00,
                session="GTH",
            ),
        ],
        decision_at=decision_at,
    )

    path = context["preopen_path"]
    assert path["schema"] == "xsp.preopen-option-path.v2"
    assert path["anchor_policy"] == "option_model_consensus_only"
    assert path["max_age_seconds"] == 360.0
    assert path["usable"] is True
    assert path["trading_date"] == "2026-07-27"
    assert path["end_ts"] == boundary.isoformat()
    assert path["end_market_data_types"] == {"3": 6}
    assert path["end_anchor_source"] == "option_model_consensus"
    assert path["end_reference_value"] == pytest.approx(100.60)
    assert path["end_strikes"] == (101.0, 100.0, 99.0)
    assert path["end_max_age_seconds"] == pytest.approx(5.0)
    assert path["end_median_relative_spread"] > 0
    assert path["horizons"]["120"]["anchor_market_data_types"] == {"3": 6}
    assert path["horizons"]["120"]["anchor_source"] == "option_model_consensus"
    assert path["horizons"]["120"]["anchor_reference_value"] == pytest.approx(100.40)
    assert path["horizons"]["120"]["anchor_strikes"] == (100.0, 101.0, 99.0)
    assert path["horizons"]["120"]["anchor_max_age_seconds"] == pytest.approx(5.0)
    assert path["horizons"]["120"]["anchor_median_relative_spread"] > 0
    assert path["horizons"]["120"]["value_change_points"] == pytest.approx(0.20)
    assert path["horizons"]["120"]["direction"] == "up"
    assert path["horizons"]["240"]["value_change_points"] == pytest.approx(-0.10)
    assert path["horizons"]["240"]["direction"] == "down"
    assert path["horizons"]["360"]["value_change_points"] == pytest.approx(0.60)
    assert path["horizons"]["360"]["direction"] == "up"


def test_option_context_preopen_path_fails_closed_when_a_horizon_is_missing() -> None:
    decision_at = NOW
    boundary = decision_at.replace(minute=30) - timedelta(minutes=10)
    context = xsp_option_context_at(
        [
            _option_snapshot(decision_at - timedelta(minutes=1)),
            _option_snapshot(boundary, parity_value=100.60, session="GTH"),
            _option_snapshot(
                boundary - timedelta(minutes=120),
                parity_value=100.40,
                session="GTH",
            ),
            _option_snapshot(
                boundary - timedelta(minutes=360),
                parity_value=100.00,
                session="GTH",
            ),
        ],
        decision_at=decision_at,
    )

    path = context["preopen_path"]
    assert path["usable"] is False
    assert path["reasons"] == ("missing_240m_anchor",)
    assert path["horizons"]["240"] == {
        "usable": False,
        "reason": "no_causal_same_expiry_anchor",
    }


def test_shadow_cli_hands_the_complete_same_date_tape_to_rth(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.backtest.quotes import append_snapshot
    from tradebot.research.xsp_shadow_cli import _main_async

    boundary = NOW.replace(minute=30) - timedelta(minutes=10)
    snapshots = (
        _option_snapshot(
            boundary - timedelta(minutes=360),
            parity_value=100.00,
            session="GTH",
        ),
        _option_snapshot(
            boundary - timedelta(minutes=240),
            parity_value=100.70,
            session="GTH",
        ),
        _option_snapshot(
            boundary - timedelta(minutes=120),
            parity_value=100.40,
            session="GTH",
        ),
        _option_snapshot(boundary, parity_value=100.60, session="GTH"),
        _option_snapshot(
            NOW - timedelta(minutes=1),
            parity_value=100.75,
            session="RTH",
        ),
    )
    tape = tmp_path / "2026-07-27.jsonl"
    for snapshot in snapshots:
        append_snapshot(tape, snapshot)
    news_path = tmp_path / "news" / "latest.json"
    news_history = news_path.parent / "history" / "2026-07.jsonl"
    news_history.parent.mkdir(parents=True)
    prior_news = _news_snapshot(NOW - timedelta(minutes=5))
    latest_news = _news_snapshot(NOW + timedelta(minutes=1))
    news_history.write_text(json.dumps(prior_news) + "\n", encoding="utf-8")
    news_path.write_text(json.dumps(latest_news) + "\n", encoding="utf-8")
    selection_path = tmp_path / "selected.json"
    selection = xsp_selected_shadow_run(
        LiveCalibrationLedger(tmp_path / "selection-ledger.jsonl"),
        xsp_opening_edge_shadow_recommendation(),
        run_id="xsp-opening-edge-20260727",
        strategy_version=XSP_OPENING_EDGE_VERSION,
        config_fingerprint=XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        capital_sleeve="xsp-directional-unit",
        selected_at=NOW - timedelta(minutes=35),
    )
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    captured = {}

    class _Client:
        def __init__(self, config):
            captured["config"] = config

        async def disconnect(self):
            captured["disconnected"] = True

    async def _advance(_ledger, **kwargs):
        captured["snapshots"] = kwargs["option_snapshots"]
        captured["news"] = kwargs["news_snapshot"]
        captured["selected_run"] = kwargs["selected_run"]
        return {"status": "ok", "evaluation_status": "EVALUATED"}

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_shadow_from_ibkr",
        _advance,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--ledger",
                    str(tmp_path / "calibration.jsonl"),
                    "--option-tape",
                    str(tape),
                    "--news-signal",
                    str(news_path),
                    "--selected-run",
                    str(selection_path),
                )
            )
        )
        == 0
    )
    capsys.readouterr()

    loaded = captured["snapshots"]
    assert tuple(snapshot.session for snapshot in loaded) == (
        "GTH",
        "GTH",
        "GTH",
        "GTH",
        "RTH",
    )
    context = xsp_option_context_at(loaded, decision_at=NOW)
    assert context["usable"] is True
    assert context["preopen_path"]["usable"] is True
    assert set(context["preopen_path"]["horizons"]) == {"120", "240", "360"}
    assert captured["news"] == (prior_news, latest_news)
    assert captured["selected_run"] == selection
    assert captured["disconnected"] is True


def test_shadow_cli_fails_when_the_checkpoint_is_not_evaluated(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    class _Client:
        def __init__(self, _config):
            pass

        async def disconnect(self):
            pass

    async def _advance(_ledger, **_kwargs):
        return {"status": "ok", "evaluation_status": "STALE_DATA"}

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_shadow_from_ibkr",
        _advance,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--ledger",
                    str(tmp_path / "calibration.jsonl"),
                    "--selected-run",
                    str(tmp_path / "missing-selection.json"),
                )
            )
        )
        == 2
    )
    capsys.readouterr()


def test_shadow_cli_v2_prefreezes_without_loading_v1_selection(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    run_start = datetime(2026, 7, 28, 0, 15, tzinfo=timezone.utc)
    selected_path = tmp_path / "obsolete-v1-selection.json"
    selected_path.write_text("not valid JSON", encoding="utf-8")

    class _Client:
        def __init__(self, _config):
            pass

        async def disconnect(self):
            captured["disconnected"] = True

    async def _advance_v2(_ledger, **kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "evaluation_status": "CLOSED",
            "broker_request_skipped": "run_not_started",
            "order_authority": "none",
        }

    async def _advance_v1(*_args, **_kwargs):
        raise AssertionError("v2 mode must not enter the v1 observer")

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_opening_edge_v2_spec",
        lambda: "v2-spec",
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_opening_edge_v2_run_start",
        lambda *_args, **_kwargs: run_start,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_opening_edge_v2_from_ibkr",
        _advance_v2,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_shadow_from_ibkr",
        _advance_v1,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v2",
                    "--ledger",
                    str(tmp_path / "v2-cli.jsonl"),
                    "--news-signal",
                    str(tmp_path / "missing-news.json"),
                    "--selected-run",
                    str(selected_path),
                    "--selected-transport",
                    str(tmp_path / "missing-live-transport.json"),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert captured["run_started_at"] == run_start
    assert captured["duration_str"] == XSP_OPENING_EDGE_V2_HISTORY_DURATION
    assert captured["news_snapshot"] == ()
    assert captured["spec"] == "v2-spec"
    assert captured["disconnected"] is True
    assert output["mode"] == "opening-edge-v2"
    assert output["selected_run_id"] is None
    assert output["v2_run_started_at_utc"] == run_start.isoformat()
    assert output["order_authority"] == "none"


def test_shadow_cli_freezes_transport_from_fresh_read_only_state(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    ranking = tmp_path / "ranking.json"
    dwell = tmp_path / "dwell.json"
    preview = tmp_path / "preview.json"
    selected = tmp_path / "selected.json"
    for path in (ranking, dwell, preview):
        path.write_text("{}", encoding="utf-8")

    class _Client:
        def __init__(self, config):
            captured["readonly"] = config.readonly

        async def disconnect(self):
            captured["disconnected"] = True

    source = {"checkpoint_id": "fresh-source"}
    broker = {"observed_at_utc": NOW.isoformat()}
    selection = {
        "selection_id": "selected-transport",
        "order_authority": "rth_cash_pair_limit_only",
    }

    monkeypatch.setenv("IBKR_READONLY", "1")
    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.latest_xsp_v2_source_receipt",
        lambda _records: source,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_v2_broker_snapshot",
        lambda _client: asyncio.sleep(0, result=broker),
    )

    def _select(**kwargs):
        captured.update(kwargs)
        return selection

    def _write(path, value):
        captured["written"] = (path, value)

    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.select_xsp_v2_transport",
        _select,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.write_xsp_v2_transport_selection",
        _write,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v2",
                    "--ledger",
                    str(tmp_path / "ledger.jsonl"),
                    "--selected-transport",
                    str(selected),
                    "--freeze-selected-transport",
                    "--transport-ranking",
                    str(ranking),
                    "--transport-dwell",
                    str(dwell),
                    "--transport-preview",
                    str(preview),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert captured["readonly"] is True
    assert captured["source_receipt"] == source
    assert captured["broker_snapshot"] == broker
    assert captured["written"] == (selected, selection)
    assert captured["disconnected"] is True
    assert output == selection


def test_shadow_cli_v2_advances_one_explicit_selected_transport(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    selection_path = tmp_path / "selected-live-transport.json"
    selection_path.write_text("{}", encoding="utf-8")
    selection = {
        "selection_id": "selected-transport",
        "order_authority": "selected_spyu_spxu_cash",
    }

    class _Client:
        def __init__(self, config):
            captured["readonly"] = config.readonly

        async def disconnect(self):
            captured["disconnected"] = True

    async def _advance_v2(_ledger, **_kwargs):
        return {
            "evaluation_status": "EVALUATED",
            "freshness_ok": True,
            "session": "RTH",
            "order_authority": "none",
            "checkpoint_id": "source",
            "recorded_at_utc": NOW.isoformat(),
        }

    async def _observe(*_args, **_kwargs):
        return {"status": "UNCHANGED"}

    async def _execute(_ledger, **kwargs):
        captured["selection"] = kwargs["selection"]
        captured["source"] = kwargs["source_receipt"]
        return {"status": "UNCHANGED", "submitted_orders": 0}

    monkeypatch.delenv("IBKR_READONLY", raising=False)
    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_v2_transport_selection",
        lambda path: selection if path == selection_path else None,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_opening_edge_v2_from_ibkr",
        _advance_v2,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_v2_etf_execution_observer",
        _observe,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_v2_live_transport",
        _execute,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v2",
                    "--ledger",
                    str(tmp_path / "v2-live.jsonl"),
                    "--news-signal",
                    str(tmp_path / "missing-news.json"),
                    "--selected-transport",
                    str(selection_path),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert captured["readonly"] is False
    assert captured["selection"] == selection
    assert captured["source"]["checkpoint_id"] == "source"
    assert captured["disconnected"] is True
    assert output["selected_transport_id"] == "selected-transport"
    assert output["transport_execution"]["submitted_orders"] == 0
    assert output["order_authority"] == "selected_spyu_spxu_cash"


def test_shadow_cli_freezes_v3_transport_from_one_exact_preview(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    preview = tmp_path / "preview.json"
    cash = tmp_path / "cash.json"
    selected = tmp_path / "selected.json"
    preview.write_text("{}", encoding="utf-8")
    cash.write_text("{}", encoding="utf-8")

    class _Client:
        def __init__(self, config):
            captured["readonly"] = config.readonly

        async def disconnect(self):
            captured["disconnected"] = True

    selection = {
        "selection_id": "selected-v3-transport",
        "strategy_version": "xsp.opening-edge-v3-regime-harmony-24x5.v1",
        "order_authority": "rth_cash_pair_limit_only",
    }
    monkeypatch.setenv("IBKR_READONLY", "1")
    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.latest_xsp_v3_source_receipt",
        lambda _records: {"checkpoint_id": "v3-source"},
    )

    async def _broker(_client, **kwargs):
        captured["broker_symbols"] = kwargs["symbols"]
        return {"account_id": "DU123"}

    def _select(**kwargs):
        captured.update(kwargs)
        return selection

    def _write(path, value):
        captured["written"] = (path, value)

    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_v2_broker_snapshot",
        _broker,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.select_xsp_v3_transport",
        _select,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.write_xsp_transport_selection",
        _write,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v3",
                    "--ledger",
                    str(tmp_path / "ledger.jsonl"),
                    "--selected-transport",
                    str(selected),
                    "--freeze-selected-transport",
                    "--transport-preview",
                    str(preview),
                    "--transport-cash-receipt",
                    str(cash),
                    "--accept-rth-only-cash-scope",
                )
            )
        )
        == 0
    )
    assert captured["readonly"] is True
    assert captured["broker_symbols"] == ("UPRO", "SPXU")
    assert captured["cash_receipt_path"] == cash
    assert captured["preview_path"] == preview
    assert captured["rth_scope_accepted"] is True
    assert captured["written"] == (selected, selection)
    assert captured["disconnected"] is True
    assert json.loads(capsys.readouterr().out) == selection


def test_shadow_cli_selected_transport_requires_explicit_writable_broker(
    tmp_path,
    monkeypatch,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    selection_path = tmp_path / "selected-live-transport.json"
    selection_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("IBKR_READONLY", "1")
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_v2_transport_selection",
        lambda _path: {
            "selection_id": "selected-transport",
            "order_authority": "selected_spyu_spxu_cash",
        },
    )

    with pytest.raises(ValueError, match="explicitly writable broker connection"):
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v2",
                    "--selected-transport",
                    str(selection_path),
                )
            )
        )


@pytest.mark.parametrize(
    ("parity_change", "expected_cohort"),
    [
        (0.25, "aligned"),
        (-0.25, "opposed"),
        (0.0, "flat"),
        (None, "unavailable"),
    ],
)
def test_option_parity_benchmark_only_stratifies_forward_pairs(
    tmp_path,
    parity_change,
    expected_cohort,
) -> None:
    if parity_change is None:
        option_context = xsp_option_context_at([], decision_at=NOW)
    else:
        option_context = xsp_option_context_at(
            [
                _option_snapshot(
                    NOW - timedelta(minutes=6),
                    parity_value=100.0,
                ),
                _option_snapshot(
                    NOW - timedelta(minutes=1),
                    parity_value=100.0 + parity_change,
                ),
            ],
            decision_at=NOW,
        )
    ledger = LiveCalibrationLedger(tmp_path / f"{expected_cohort}.jsonl")
    forecast = _forecast(
        ledger,
        horizon_minutes=60,
        context={
            "session": "RTH",
            "evidence_mode": "forward_broker_history",
            "option_parity": option_context,
        },
        counterfactual_direction="UP",
    )
    ledger.settle(
        forecast_id=str(forecast["forecast_id"]),
        observed={
            "outcome_as_of_utc": (NOW + timedelta(minutes=60)).isoformat(),
            "counterfactuals": [
                {
                    "strategy_id": "directional_impulse.observer",
                    "direction": "up",
                    "net_points": -1.25,
                }
            ],
        },
        drift={},
        verdict="HOLD",
        settled_at=NOW + timedelta(minutes=60),
    )

    benchmark = xsp_option_parity_participation_benchmark(ledger)

    assert benchmark["authority"] == "observation_only"
    assert benchmark["promotion_eligible"] is False
    assert benchmark["policy"]["action"] == "classify_only"
    assert benchmark["cohorts"][expected_cohort]["pairs"] == 1
    assert benchmark["prospective_cohorts"][expected_cohort]["pairs"] == 1
    expected_liquidity = "unavailable" if parity_change is None else "stable"
    assert benchmark["liquidity_cohorts"][expected_liquidity]["pairs"] == 1
    assert benchmark["prospective_liquidity_cohorts"][expected_liquidity]["pairs"] == 1
    assert benchmark["preopen_cohorts"]["unavailable"]["pairs"] == 1
    assert benchmark["prospective_pairs"] == 1
    assert benchmark["ta_observer_points"] == -1.25
    assert benchmark["sample_gate"] is False
    assert benchmark["shadow_recommendation"]["verdict"] == "HOLD"
    assert benchmark["shadow_recommendation"]["order_authority"] == "none"


@pytest.mark.parametrize(
    (
        "prior_half_spread",
        "prior_age",
        "current_half_spread",
        "current_age",
        "expected_cohort",
    ),
    [
        (0.06, 7.0, 0.04, 3.0, "strengthening"),
        (0.04, 3.0, 0.06, 7.0, "weakening"),
        (0.06, 3.0, 0.04, 7.0, "mixed"),
        (0.05, 5.0, 0.05, 5.0, "stable"),
    ],
)
def test_option_parity_benchmark_classifies_pareto_liquidity_without_thresholds(
    tmp_path,
    prior_half_spread,
    prior_age,
    current_half_spread,
    current_age,
    expected_cohort,
) -> None:
    option_context = xsp_option_context_at(
        [
            _option_snapshot(
                NOW - timedelta(minutes=6),
                parity_value=100.0,
                option_half_spread=prior_half_spread,
                quote_age_seconds=prior_age,
            ),
            _option_snapshot(
                NOW - timedelta(minutes=1),
                parity_value=100.25,
                option_half_spread=current_half_spread,
                quote_age_seconds=current_age,
            ),
        ],
        decision_at=NOW,
    )
    ledger = LiveCalibrationLedger(tmp_path / f"{expected_cohort}.jsonl")
    forecast = _forecast(
        ledger,
        horizon_minutes=60,
        context={
            "session": "RTH",
            "evidence_mode": "forward_broker_history",
            "option_parity": option_context,
        },
        counterfactual_direction="UP",
    )
    ledger.settle(
        forecast_id=str(forecast["forecast_id"]),
        observed={
            "outcome_as_of_utc": (NOW + timedelta(minutes=60)).isoformat(),
            "counterfactuals": [
                {
                    "strategy_id": "directional_impulse.observer",
                    "direction": "up",
                    "net_points": 1.0,
                }
            ],
        },
        drift={},
        verdict="HOLD",
        settled_at=NOW + timedelta(minutes=60),
    )

    benchmark = xsp_option_parity_participation_benchmark(ledger)

    assert benchmark["policy"]["liquidity_classification"] == (
        "pareto_pairs_up_dispersion_spread_age_down"
    )
    assert benchmark["liquidity_cohorts"][expected_cohort]["pairs"] == 1
    assert benchmark["prospective_liquidity_cohorts"][expected_cohort]["pairs"] == 1
    assert benchmark["aligned_liquidity_candidate"]["authority"] == "observation_only"
    assert (
        benchmark["aligned_liquidity_candidate"]["shadow_candidate_eligible"] is False
    )


@pytest.mark.parametrize(
    ("schema", "expected_cohort"),
    (
        ("xsp.preopen-option-path.v2", "reversal_into"),
        ("xsp.preopen-option-path.v1", "unavailable"),
    ),
)
def test_option_parity_benchmark_requires_current_preopen_schema(
    tmp_path,
    schema,
    expected_cohort,
) -> None:
    boundary = NOW.replace(minute=30) - timedelta(minutes=10)
    option_context = xsp_option_context_at(
        [
            _option_snapshot(NOW - timedelta(minutes=6), parity_value=100.0),
            _option_snapshot(NOW - timedelta(minutes=1), parity_value=100.25),
            _option_snapshot(boundary, parity_value=100.60, session="GTH"),
            _option_snapshot(
                boundary - timedelta(minutes=120),
                parity_value=100.40,
                session="GTH",
            ),
            _option_snapshot(
                boundary - timedelta(minutes=240),
                parity_value=100.80,
                session="GTH",
            ),
            _option_snapshot(
                boundary - timedelta(minutes=360),
                parity_value=100.90,
                session="GTH",
            ),
        ],
        decision_at=NOW,
    )
    option_context["preopen_path"]["schema"] = schema
    ledger = LiveCalibrationLedger(tmp_path / "preopen-reversal.jsonl")
    forecast = _forecast(
        ledger,
        horizon_minutes=60,
        context={
            "session": "RTH",
            "evidence_mode": "forward_broker_history",
            "option_parity": option_context,
        },
        counterfactual_direction="UP",
    )
    ledger.settle(
        forecast_id=str(forecast["forecast_id"]),
        observed={
            "outcome_as_of_utc": (NOW + timedelta(minutes=60)).isoformat(),
            "counterfactuals": [
                {
                    "strategy_id": "directional_impulse.observer",
                    "direction": "up",
                    "net_points": 1.0,
                }
            ],
        },
        drift={},
        verdict="HOLD",
        settled_at=NOW + timedelta(minutes=60),
    )

    benchmark = xsp_option_parity_participation_benchmark(ledger)

    assert benchmark["preopen_cohorts"][expected_cohort]["pairs"] == 1
    assert benchmark["prospective_preopen_cohorts"][expected_cohort]["pairs"] == 1
    assert benchmark["complete_session_preopen_usable_pairs"] == 0
    assert benchmark["promotion_eligible"] is False
    assert benchmark["policy"]["action"] == "classify_only"
    assert benchmark["aligned_candidate"]["shadow_candidate_eligible"] is False


def test_retrospective_mechanics_cannot_advance_prospective_observer_gates(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "prospective-only.jsonl")
    option_context = {
        "source": "option_nbbo_parity",
        "authority": "observation_only",
        "usable": True,
        "parity_change": {"usable": True, "direction": "up"},
    }
    news_context = {
        "source": "causal_news",
        "authority": "observation_only",
        "usable": True,
        "direction": -1,
        "impact": 74,
        "confidence": 0.9,
    }

    def append_pair(decision_at: datetime, evidence_mode: str) -> None:
        forecast = _forecast(
            ledger,
            decision_at=decision_at,
            horizon_minutes=60,
            context={
                "session": "RTH",
                "evidence_mode": evidence_mode,
                "option_parity": option_context,
                "fundamental_pressure": news_context,
            },
            counterfactual_direction="UP",
        )
        outcome_at = decision_at + timedelta(minutes=60)
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": outcome_at.isoformat(),
                "counterfactuals": [
                    {
                        "strategy_id": "directional_impulse.observer",
                        "direction": "up",
                        "net_points": -1.0,
                    }
                ],
            },
            drift={},
            verdict="HOLD",
            settled_at=outcome_at,
        )

    for index in range(5):
        append_pair(
            NOW - timedelta(days=7) + timedelta(minutes=5 * index),
            "historical_replay",
        )
    for index in range(25):
        append_pair(
            NOW + timedelta(days=index // 5, minutes=5 * (index % 5)),
            "forward_broker_history",
        )
    for offset in range(5):
        _append_observer_session(
            ledger,
            (NOW + timedelta(days=offset)).astimezone(ET_ZONE).date(),
        )

    parity = xsp_option_parity_participation_benchmark(ledger)
    news = xsp_fundamental_defensive_benchmark(ledger)

    assert parity["pairs"] == parity["usable_pairs"] == 30
    assert parity["diagnostic_complete_sessions"] == 6
    assert parity["prospective_pairs"] == parity["prospective_usable_pairs"] == 25
    assert parity["sample_eligible_pairs"] == 25
    assert parity["complete_sessions"] == 5
    assert parity["sample_gate"] is False
    assert parity["prospective_cohorts"]["aligned"]["pairs"] == 25
    assert news["pairs"] == news["vetoes"] == 30
    assert news["mechanics_pairs"] == 5
    assert news["prospective_pairs"] == news["prospective_vetoes"] == 25
    assert news["prospective_paired_delta_points"] == 25.0

    for index in range(5):
        append_pair(
            NOW + timedelta(days=7, minutes=5 * index),
            "forward_broker_history",
        )

    sixth_day = (NOW + timedelta(days=7)).astimezone(ET_ZONE).date()
    _append_observer_session(ledger, sixth_day, omit_slot=20)
    incomplete = xsp_option_parity_participation_benchmark(ledger)
    assert incomplete["prospective_usable_pairs"] == 30
    assert incomplete["sample_eligible_pairs"] == 25
    assert incomplete["complete_sessions"] == 5
    assert incomplete["sample_gate"] is False

    _append_observer_session(ledger, sixth_day)
    admitted = xsp_option_parity_participation_benchmark(ledger)
    assert admitted["sample_eligible_pairs"] == 30
    assert admitted["complete_sessions"] == 6
    assert admitted["complete_session_dates"][-1] == sixth_day.isoformat()
    assert admitted["sample_gate"] is True
    candidate = admitted["aligned_candidate"]
    assert candidate["shadow_candidate_eligible"] is False
    assert candidate["gate_checks"]["sample"] is True
    assert candidate["gate_checks"]["both_directions"] is False
    assert candidate["gate_checks"]["positive_net"] is False


def test_option_parity_candidate_uses_independent_non_overlapping_sequences(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "aligned-candidate.jsonl")

    def append_pair(
        day: date,
        *,
        minute_offset: int,
        direction: str,
        parity_direction: str,
        net_points: float,
    ) -> None:
        decision_at = datetime.combine(day, time(9, 35), ET_ZONE) + timedelta(
            minutes=minute_offset
        )
        forecast = _forecast(
            ledger,
            decision_at=decision_at,
            horizon_minutes=60,
            context={
                "session": "RTH",
                "evidence_mode": "forward_broker_history",
                "option_parity": {
                    "source": "option_nbbo_parity",
                    "authority": "observation_only",
                    "usable": True,
                    "ts": (decision_at - timedelta(minutes=1)).isoformat(),
                    "chain_fingerprint": "a" * 64,
                    "parity_change": {
                        "usable": True,
                        "direction": parity_direction,
                        "prior_ts": (decision_at - timedelta(minutes=6)).isoformat(),
                        "prior_chain_fingerprint": "b" * 64,
                    },
                },
            },
            counterfactual_direction=direction.upper(),
        )
        outcome_at = decision_at + timedelta(minutes=60)
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": outcome_at.isoformat(),
                "counterfactuals": [
                    {
                        "strategy_id": "directional_impulse.observer",
                        "direction": direction,
                        "net_points": net_points,
                    }
                ],
            },
            drift={},
            verdict="HOLD",
            settled_at=outcome_at,
        )

    # The +99 event is deliberately inside the first holding period. It must
    # never inflate either causal sequence. Opposed events remain in TA-only,
    # while the aligned candidate may consider the next eligible aligned turn.
    specs = (
        (0, "up", "up", 1.0),
        (5, "down", "down", 99.0),
        (60, "down", "down", 1.0),
        (120, "up", "down", -1.0),
        (125, "up", "up", 1.0),
        (180, "down", "up", -1.0),
        (185, "down", "down", 1.0),
    )
    for offset in range(5):
        day = date(2026, 7, 27) + timedelta(days=offset)
        for minute_offset, direction, parity_direction, points in specs:
            append_pair(
                day,
                minute_offset=minute_offset,
                direction=direction,
                parity_direction=parity_direction,
                net_points=points,
            )
        _append_observer_session(ledger, day)

    benchmark = xsp_option_parity_participation_benchmark(ledger)
    aligned = benchmark["aligned_candidate"]

    assert benchmark["sample_gate"] is True
    assert benchmark["sample_eligible_pairs"] == 35
    assert aligned["authority"] == "observation_only"
    assert aligned["coverage_complete_sessions"] == 5
    assert aligned["baseline"]["trades"] == 20
    assert aligned["baseline"]["net_points"] == pytest.approx(0.0)
    assert aligned["candidate"]["trades"] == 20
    assert aligned["candidate"]["net_points"] == pytest.approx(20.0)
    assert aligned["candidate"]["directions"] == {
        "up": {"trades": 10, "net_points": pytest.approx(10.0)},
        "down": {"trades": 10, "net_points": pytest.approx(10.0)},
    }
    assert aligned["candidate"]["daily_lcb95_points"] == pytest.approx(4.0)
    assert aligned["candidate"][
        "minimum_leave_one_session_out_points"
    ] == pytest.approx(16.0)
    assert aligned["candidate"]["largest_win_share"] == pytest.approx(0.05)
    assert all(aligned["gate_checks"].values())
    assert aligned["shadow_candidate_eligible"] is True
    assert benchmark["promotion_eligible"] is False
    recommendation = benchmark["shadow_recommendation"]
    assert recommendation["verdict"] == "PROMOTE"
    assert recommendation["recommended_candidate_schema"] == aligned["schema"]
    assert recommendation["selection_authority"] == ("none_until_explicit_run_freeze")
    assert recommendation["profitability_clock_started"] is False
    assert (
        recommendation["preregistered_selected_run_policy"]
        == XSP_DIRECTIONAL_SHADOW_POLICY
    )
    selection_args = {
        "run_id": "xsp-shadow-20260727",
        "strategy_version": "xsp.parity-aligned.v1",
        "config_fingerprint": "frozen-config",
        "capital_sleeve": "xsp-directional-unit",
        "selected_at": datetime(2026, 7, 27, 13, 35, tzinfo=timezone.utc),
    }
    selection = xsp_selected_shadow_run(ledger, recommendation, **selection_args)
    assert selection == xsp_selected_shadow_run(
        ledger, recommendation, **selection_args
    )
    assert selection["schema"] == XSP_SELECTED_SHADOW_RUN_VERSION
    assert selection["strategy_id"] == aligned["schema"]
    assert selection["recommendation_fingerprint"] == recommendation["fingerprint"]
    assert selection["risk_policy"] == XSP_DIRECTIONAL_SHADOW_POLICY
    assert selection["order_authority"] == "none"
    assert selection["profitability_clock_started"] is False
    assert selection["requested_run_id"] == selection_args["run_id"]
    assert selection["run_id"] == selection["selection_id"]
    unsigned_selection = dict(selection)
    assert unsigned_selection.pop("selection_id") == selection["run_id"]
    assert unsigned_selection.pop("run_id") == calibration_fingerprint(
        unsigned_selection
    )
    policy = xsp_profitability_policy_from_selected_run(selection)
    assert policy == XspProfitabilityPolicy(
        run_id=selection["selection_id"],
        strategy_id=aligned["schema"],
        strategy_version=selection_args["strategy_version"],
        config_fingerprint=selection_args["config_fingerprint"],
        capital_sleeve=selection_args["capital_sleeve"],
        max_drawdown_points=25.0,
        max_session_loss_points=5.0,
        minimum_week_closed_trades=2,
        maximum_top_five_win_share=0.5,
        slot_tolerance_seconds=90.0,
    )
    tampered_selection = dict(selection)
    tampered_selection["run_id"] = "manual-run-id"
    with pytest.raises(ValueError, match="invalid_selection"):
        xsp_profitability_policy_from_selected_run(tampered_selection)
    tampered_selection = dict(selection)
    tampered_selection["risk_policy"] = {
        **XSP_DIRECTIONAL_SHADOW_POLICY,
        "max_session_loss_points": 4.0,
    }
    with pytest.raises(ValueError, match="risk_policy_drift"):
        xsp_profitability_policy_from_selected_run(tampered_selection)
    _append_observer_session(ledger, date(2026, 8, 3))
    with pytest.raises(ValueError, match="recommendation_not_current_for_ledger"):
        xsp_selected_shadow_run(ledger, recommendation, **selection_args)


def test_option_liquidity_candidate_requires_incremental_prospective_economics(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "aligned-liquidity-candidate.jsonl")

    def append_pair(
        day: date,
        *,
        minute_offset: int,
        direction: str,
        parity_direction: str,
        liquidity: str,
        net_points: float,
    ) -> None:
        decision_at = datetime.combine(day, time(9, 35), ET_ZONE) + timedelta(
            minutes=minute_offset
        )
        strengthening = liquidity == "strengthening"
        current_spread, prior_spread = (0.04, 0.08) if strengthening else (0.08, 0.04)
        current_age, prior_age = (2.0, 5.0) if strengthening else (5.0, 2.0)
        forecast = _forecast(
            ledger,
            decision_at=decision_at,
            horizon_minutes=60,
            context={
                "session": "RTH",
                "evidence_mode": "forward_broker_history",
                "option_parity": {
                    "source": "option_nbbo_parity",
                    "authority": "observation_only",
                    "usable": True,
                    "ts": (decision_at - timedelta(minutes=1)).isoformat(),
                    "chain_fingerprint": "a" * 64,
                    "pairs": 3,
                    "dispersion_points": (0.01 if strengthening else 0.02),
                    "median_relative_spread": current_spread,
                    "max_age_seconds": current_age,
                    "parity_change": {
                        "usable": True,
                        "direction": parity_direction,
                        "prior_ts": (decision_at - timedelta(minutes=6)).isoformat(),
                        "prior_chain_fingerprint": "b" * 64,
                        "prior_pairs": 3,
                        "prior_dispersion_points": (0.02 if strengthening else 0.01),
                        "prior_median_relative_spread": prior_spread,
                        "prior_max_age_seconds": prior_age,
                    },
                },
            },
            counterfactual_direction=direction.upper(),
        )
        outcome_at = decision_at + timedelta(minutes=60)
        ledger.settle(
            forecast_id=str(forecast["forecast_id"]),
            observed={
                "outcome_as_of_utc": outcome_at.isoformat(),
                "counterfactuals": [
                    {
                        "strategy_id": "directional_impulse.observer",
                        "direction": direction,
                        "net_points": net_points,
                    }
                ],
            },
            drift={},
            verdict="HOLD",
            settled_at=outcome_at,
        )

    specs = (
        (0, "up", "up", "strengthening", 1.0),
        (5, "down", "down", "strengthening", 99.0),
        (60, "down", "down", "strengthening", 1.0),
        (120, "up", "down", "weakening", -1.0),
        (125, "up", "up", "weakening", -1.0),
        (180, "down", "up", "weakening", -1.0),
        (185, "down", "down", "strengthening", 1.0),
    )
    for offset in range(5):
        day = date(2026, 7, 27) + timedelta(days=offset)
        for minute_offset, direction, parity_direction, liquidity, points in specs:
            append_pair(
                day,
                minute_offset=minute_offset,
                direction=direction,
                parity_direction=parity_direction,
                liquidity=liquidity,
                net_points=points,
            )
        _append_observer_session(ledger, day)

    benchmark = xsp_option_parity_participation_benchmark(ledger)
    candidate = benchmark["aligned_liquidity_candidate"]

    assert benchmark["sample_gate"] is True
    assert benchmark["liquidity_sample_gate"] is True
    assert benchmark["liquidity_sample_eligible_pairs"] == 35
    assert candidate["authority"] == "observation_only"
    assert candidate["aligned_reference"]["trades"] == 20
    assert candidate["aligned_reference"]["net_points"] == pytest.approx(10.0)
    assert candidate["candidate"]["trades"] == 15
    assert candidate["candidate"]["net_points"] == pytest.approx(15.0)
    assert candidate["candidate"]["directions"] == {
        "up": {"trades": 5, "net_points": pytest.approx(5.0)},
        "down": {"trades": 10, "net_points": pytest.approx(10.0)},
    }
    assert candidate["candidate"]["daily_lcb95_points"] == pytest.approx(3.0)
    assert candidate["candidate"][
        "minimum_leave_one_session_out_points"
    ] == pytest.approx(12.0)
    assert all(candidate["gate_checks"].values())
    assert candidate["shadow_candidate_eligible"] is True
    assert benchmark["promotion_eligible"] is False
    recommendation = benchmark["shadow_recommendation"]
    assert recommendation["verdict"] == "PROMOTE"
    assert recommendation["recommended_candidate_schema"] == candidate["schema"]
    assert recommendation["open_position_strategy_switch_allowed"] is False
    assert (
        recommendation["preregistered_selected_run_policy"]
        == XSP_DIRECTIONAL_SHADOW_POLICY
    )


def test_selected_shadow_run_rejects_hold_and_tampered_recommendations(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "hold.jsonl")
    recommendation = xsp_option_parity_participation_benchmark(ledger)[
        "shadow_recommendation"
    ]
    kwargs = {
        "run_id": "xsp-shadow-20260727",
        "strategy_version": "xsp.parity-aligned.v1",
        "config_fingerprint": "frozen-config",
        "capital_sleeve": "xsp-directional-unit",
        "selected_at": NOW,
    }
    with pytest.raises(ValueError, match="recommendation_not_eligible"):
        xsp_selected_shadow_run(ledger, recommendation, **kwargs)

    tampered = dict(recommendation)
    tampered["verdict"] = "PROMOTE"
    with pytest.raises(ValueError, match="invalid_recommendation"):
        xsp_selected_shadow_run(ledger, tampered, **kwargs)


def test_opening_edge_selection_is_exact_and_shadow_only(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "opening-selection.jsonl")
    recommendation = xsp_opening_edge_shadow_recommendation()
    kwargs = {
        "run_id": "xsp-opening-edge-20260727",
        "strategy_version": XSP_OPENING_EDGE_VERSION,
        "config_fingerprint": XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        "capital_sleeve": "xsp-directional-unit",
        "selected_at": datetime(2026, 7, 27, 9, 0, tzinfo=ET_ZONE),
    }
    selection = xsp_selected_shadow_run(ledger, recommendation, **kwargs)
    policy = xsp_profitability_policy_from_selected_run(selection)

    assert selection["strategy_id"] == XSP_OPENING_EDGE_VERSION
    assert selection["order_authority"] == "none"
    assert selection["profitability_clock_started"] is False
    assert policy.run_id == selection["selection_id"]
    assert policy.config_fingerprint == XSP_OPENING_EDGE_CONFIG_FINGERPRINT

    with pytest.raises(ValueError, match="candidate_identity_drift"):
        xsp_selected_shadow_run(
            ledger,
            recommendation,
            **{**kwargs, "config_fingerprint": "wrong"},
        )


def test_complete_xsp_session_requires_rth_identity_and_every_slot(tmp_path) -> None:
    day = date(2026, 7, 27)
    ledger = LiveCalibrationLedger(tmp_path / "session-identity.jsonl")

    _append_observer_session(ledger, day, session="GTH")
    assert (
        ledger.complete_xsp_checkpoint_sessions(
            strategy_id="NO_TRADE",
            strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
        )
        == ()
    )

    _append_observer_session(ledger, day, omit_slot=20)
    assert (
        ledger.complete_xsp_checkpoint_sessions(
            strategy_id="NO_TRADE",
            strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
        )
        == ()
    )

    _append_observer_session(ledger, day)
    assert ledger.complete_xsp_checkpoint_sessions(
        strategy_id="NO_TRADE",
        strategy_version=XSP_DIRECTIONAL_OBSERVER_VERSION,
    ) == (day.isoformat(),)


def test_fundamental_context_is_timestamp_correct_and_observation_only() -> None:
    fresh = xsp_fundamental_context_at(
        _news_snapshot(NOW - timedelta(minutes=5)),
        decision_at=NOW,
    )
    future = xsp_fundamental_context_at(
        _news_snapshot(NOW + timedelta(seconds=1)),
        decision_at=NOW,
    )
    stale = xsp_fundamental_context_at(
        _news_snapshot(NOW - timedelta(hours=5)),
        decision_at=NOW,
    )

    assert fresh["usable"] is True
    assert fresh["authority"] == "observation_only"
    assert fresh["drivers"] == ["hormuz-risk"]
    assert fresh["signed_pressure"] == pytest.approx(-0.666)
    assert fresh["pressure_delta"] is None
    assert fresh["pressure_interval_seconds"] is None
    assert fresh["pressure_velocity_per_hour"] is None
    assert future["reason"] == "future"
    assert future["signed_pressure"] is None
    assert stale["reason"] == "stale"
    assert stale["signed_pressure"] is None


def test_fundamental_context_uses_the_latest_publication_available_at_decision() -> (
    None
):
    prior = _news_snapshot(NOW - timedelta(minutes=5))
    earlier = _news_snapshot(NOW - timedelta(hours=3))
    earlier["analysis"]["assets"]["XSP"].update(
        {"direction": 1, "impact": 50, "confidence": 0.8}
    )
    future = _news_snapshot(NOW + timedelta(seconds=1))

    context = xsp_fundamental_context_at(
        (future, prior, earlier),
        decision_at=NOW,
    )
    unavailable = xsp_fundamental_context_at(
        (future,),
        decision_at=NOW,
    )

    assert context["usable"] is True
    assert context["signal_as_of_utc"] == prior["signal_as_of_utc"]
    assert context["snapshot_fingerprint"] != calibration_fingerprint(future)
    assert context["signed_pressure"] == pytest.approx(-0.666)
    assert context["pressure_delta"] == pytest.approx(-1.066)
    assert context["pressure_interval_seconds"] == 10_500.0
    assert context["pressure_velocity_per_hour"] == pytest.approx(-0.365486)
    assert unavailable["usable"] is False
    assert unavailable["reason"] == "not_recorded_at_decision"


@pytest.mark.parametrize(
    (
        "direction",
        "signal_age",
        "impact",
        "authority",
        "horizon_minutes",
        "expected_pairs",
        "expected_vetoes",
    ),
    [
        ("UP", timedelta(minutes=5), 74, "observation_only", 60, 1, 1),
        ("UP", timedelta(minutes=5), 69, "observation_only", 60, 1, 0),
        ("DOWN", timedelta(minutes=5), 74, "observation_only", 60, 1, 0),
        ("UP", timedelta(hours=5), 74, "observation_only", 60, 1, 0),
        ("UP", timedelta(minutes=5), 74, "selector", 60, 1, 0),
        ("UP", timedelta(minutes=5), 74, "observation_only", 30, 0, 0),
    ],
)
def test_fundamental_defensive_benchmark_is_preregistered_and_observation_only(
    tmp_path,
    direction,
    signal_age,
    impact,
    authority,
    horizon_minutes,
    expected_pairs,
    expected_vetoes,
) -> None:
    news = _news_snapshot(NOW - signal_age)
    news["analysis"]["assets"]["XSP"]["impact"] = impact
    context = xsp_fundamental_context_at(news, decision_at=NOW)
    context["authority"] = authority
    ledger = LiveCalibrationLedger(tmp_path / "paired.jsonl")
    forecast = _forecast(
        ledger,
        horizon_minutes=horizon_minutes,
        context={
            "session": "RTH",
            "evidence_mode": "forward_broker_history",
            "fundamental_pressure": context,
        },
        counterfactual_direction=direction,
    )
    ledger.settle(
        forecast_id=str(forecast["forecast_id"]),
        observed={
            "outcome_as_of_utc": (NOW + timedelta(minutes=horizon_minutes)).isoformat(),
            "counterfactuals": [
                {
                    "strategy_id": "directional_impulse.observer",
                    "direction": direction.lower(),
                    "net_points": -1.25,
                }
            ],
        },
        drift={},
        verdict="HOLD",
        settled_at=NOW + timedelta(minutes=horizon_minutes),
    )

    benchmark = xsp_fundamental_defensive_benchmark(ledger)

    assert benchmark["authority"] == "observation_only"
    assert benchmark["promotion_eligible"] is False
    assert benchmark["pairs"] == expected_pairs
    assert benchmark["prospective_pairs"] == expected_pairs
    assert benchmark["vetoes"] == expected_vetoes
    assert benchmark["prospective_vetoes"] == expected_vetoes
    assert benchmark["ta_observer_points"] == -1.25 * expected_pairs
    assert benchmark["defended_observer_points"] == (
        0.0 if expected_vetoes else -1.25 * expected_pairs
    )
    assert benchmark["economic_interpretation"] == (
        "overlapping_observer_events_not_tradable_equity"
    )


def test_directional_shadow_does_not_substitute_a_later_bar(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "xsp-shadow.jsonl")
    forecast = ledger.freeze(
        identity={
            "strategy_id": "NO_TRADE",
            "strategy_version": XSP_DIRECTIONAL_OBSERVER_VERSION,
            "decision_as_of_utc": NOW.isoformat(),
            "tape_fingerprint": "causal-prefix",
            "config_fingerprint": "config",
            "capital_sleeve": "xsp-directional-unit",
        },
        forecast={
            "decision": "NO_TRADE",
            "outcome_not_before_utc": (NOW + timedelta(minutes=30)).isoformat(),
            "pnl_distribution": {"status": "observer"},
            "risk": {"selected_max_loss_points": 0.0},
            "costs": {"selected_points": 0.0},
            "fill_assumptions": {"broker_fill": False},
        },
        context={"evidence_mode": "historical_replay"},
        counterfactuals=[
            {
                "strategy_id": "directional_impulse.observer",
                "decision": "UP",
            }
        ],
        gates={"selected_admissible": False},
        recorded_at=NOW,
    )
    bars = [
        Bar(
            (NOW + timedelta(minutes=offset)).replace(tzinfo=None),
            100.0,
            100.2,
            99.8,
            100.0,
            0.0,
        )
        for offset in (10, 30)
    ]

    assert (
        settle_xsp_directional_observations(
            ledger,
            bars,
            settled_at=NOW + timedelta(minutes=30),
        )
        == []
    )
    assert ledger.receipt()["unsettled"] == [forecast["forecast_id"]]


def test_directional_shadow_uses_exact_next_open_and_horizon_close(tmp_path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "xsp-shadow.jsonl")
    forecast = ledger.freeze(
        identity={
            "strategy_id": "NO_TRADE",
            "strategy_version": XSP_DIRECTIONAL_OBSERVER_VERSION,
            "decision_as_of_utc": NOW.isoformat(),
            "tape_fingerprint": "causal-prefix",
            "config_fingerprint": "config",
            "capital_sleeve": "xsp-directional-unit",
        },
        forecast={
            "decision": "NO_TRADE",
            "outcome_not_before_utc": (NOW + timedelta(minutes=30)).isoformat(),
            "pnl_distribution": {"status": "observer"},
            "risk": {"selected_max_loss_points": 0.0},
            "costs": {"selected_points": 0.0},
            "fill_assumptions": {"broker_fill": False},
        },
        context={"evidence_mode": "historical_replay"},
        counterfactuals=[
            {
                "strategy_id": "directional_impulse.observer",
                "decision": "UP",
            }
        ],
        gates={"selected_admissible": False},
        recorded_at=NOW,
    )
    bars = [
        Bar(
            (NOW + timedelta(minutes=offset)).replace(tzinfo=None),
            price,
            price + 0.2,
            price - 0.2,
            price,
            0.0,
        )
        for offset, price in ((5, 100.0), (30, 102.0))
    ]

    [result] = settle_xsp_directional_observations(
        ledger,
        bars,
        settled_at=NOW + timedelta(minutes=30),
    )

    counterfactual = result["observed"]["counterfactuals"][0]
    assert counterfactual["entry_as_of_utc"] == (NOW + timedelta(minutes=5)).isoformat()
    assert counterfactual["entry_price"] == 100.0
    assert counterfactual["exit_price"] == 102.0
    assert (
        result["observed"]["outcome_as_of_utc"]
        == (NOW + timedelta(minutes=30)).isoformat()
    )
    assert result["forecast_id"] == forecast["forecast_id"]


def test_forward_shadow_freezes_only_before_the_first_outcome_window(
    tmp_path,
) -> None:
    closes = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        104.0,
        102.0,
        100.0,
        99.0,
        98.0,
        97.0,
        98.0,
        99.0,
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
        119.0,
        120.0,
    ]
    bars = [
        Bar(
            NOW.replace(tzinfo=None) + timedelta(minutes=5 * index),
            close,
            close + 0.2,
            close - 0.2,
            close,
            0.0,
        )
        for index, close in enumerate(closes)
    ]
    oracle = LiveCalibrationLedger(tmp_path / "oracle.jsonl")
    replay_xsp_directional_shadow(oracle, bars)
    first = next(row for row in oracle.records() if row.get("kind") == "forecast")
    decision_at = datetime.fromisoformat(str(first["identity"]["decision_as_of_utc"]))
    ledger = LiveCalibrationLedger(tmp_path / "forward.jsonl")

    receipt = advance_xsp_directional_shadow(
        ledger,
        bars,
        observed_at=decision_at + timedelta(minutes=1),
    )

    assert receipt["new_forecasts"] == 3
    assert receipt["new_results"] == 0
    forecasts = [row for row in ledger.records() if row.get("kind") == "forecast"]
    assert {row["recorded_at_utc"] for row in forecasts} == {
        (decision_at + timedelta(minutes=1)).isoformat()
    }
    assert all(
        row["context"]["option_parity"]["reasons"]
        == ["no_causal_same_session_snapshot"]
        for row in forecasts
    )
    assert all(
        row["context"]["fundamental_pressure"]["reason"] == "missing"
        for row in forecasts
    )
    second = advance_xsp_directional_shadow(
        ledger,
        bars,
        observed_at=decision_at + timedelta(minutes=1),
        option_snapshots=[_option_snapshot(decision_at - timedelta(minutes=1))],
        news_snapshot=_news_snapshot(decision_at - timedelta(minutes=1)),
    )

    assert second["new_forecasts"] == 0
    assert second["forecasts"] == 3
    assert all(
        row["context"]["option_parity"]["usable"] is False
        for row in ledger.records()
        if row.get("kind") == "forecast"
    )
    assert all(
        row["context"]["fundamental_pressure"]["reason"] == "missing"
        for row in ledger.records()
        if row.get("kind") == "forecast"
    )
    original_ids = {
        row["forecast_id"] for row in ledger.records() if row.get("kind") == "forecast"
    }
    changed_prefix = [
        Bar(
            bars[0].ts,
            bars[0].open,
            bars[0].high,
            bars[0].low,
            bars[0].close,
            1.0,
        ),
        *bars[1:],
    ]
    third = advance_xsp_directional_shadow(
        ledger,
        changed_prefix,
        observed_at=decision_at + timedelta(minutes=1),
        news_snapshot=_news_snapshot(decision_at - timedelta(minutes=1)),
    )
    assert third["new_forecasts"] == 0
    assert {
        row["forecast_id"] for row in ledger.records() if row.get("kind") == "forecast"
    } == original_ids


def test_forward_checkpoint_commits_only_after_evaluation_succeeds(
    tmp_path,
    monkeypatch,
) -> None:
    contract = Contract(
        conId=416904,
        symbol="XSP",
        secType="IND",
        exchange="CBOE",
        currency="USD",
    )

    class _Client:
        async def qualify_proxy_contracts(self, _contract):
            return [contract]

        async def historical_bars_ohlcv(self, _contract, **_kwargs):
            return [
                Bar(
                    datetime(2026, 7, 27, 9, 30),
                    100.0,
                    100.2,
                    99.8,
                    100.0,
                    0.0,
                )
            ]

        def last_historical_request(self, _contract):
            return {"status": "ok", "bars_count": 1}

    def fail_evaluation(*_args, **_kwargs):
        raise RuntimeError("simulated evaluator failure")

    monkeypatch.setattr(
        "tradebot.research.xsp_shadow.advance_xsp_directional_shadow",
        fail_evaluation,
    )
    ledger = LiveCalibrationLedger(tmp_path / "failed.jsonl")
    with pytest.raises(RuntimeError, match="simulated evaluator failure"):
        asyncio.run(
            advance_xsp_shadow_from_ibkr(
                ledger,
                client=_Client(),
                observed_at=datetime(
                    2026,
                    7,
                    27,
                    9,
                    36,
                    tzinfo=ET_ZONE,
                ),
            )
        )

    assert ledger.receipt()["checkpoints"] == 0


def test_gateway_failure_cannot_fabricate_shadow_coverage(tmp_path) -> None:
    class _Client:
        async def qualify_proxy_contracts(self, _contract):
            raise ConnectionError("simulated unavailable tunnel")

    ledger = LiveCalibrationLedger(tmp_path / "unavailable.jsonl")
    with pytest.raises(ConnectionError, match="unavailable tunnel"):
        asyncio.run(
            advance_xsp_shadow_from_ibkr(
                ledger,
                client=_Client(),
                observed_at=datetime(
                    2026,
                    7,
                    27,
                    9,
                    36,
                    tzinfo=ET_ZONE,
                ),
            )
        )

    assert ledger.receipt()["checkpoints"] == 0


def test_shadow_skips_broker_outside_supported_cash_window(tmp_path) -> None:
    class _Client:
        async def qualify_proxy_contracts(self, _contract):
            raise AssertionError("broker qualification must be skipped")

    cases = (
        (
            "gth",
            datetime(2026, 7, 27, 8, 34, tzinfo=ET_ZONE),
            "GTH",
            "UNSUPPORTED_SESSION",
            "unsupported_session",
        ),
        (
            "holiday",
            datetime(2026, 7, 3, 9, 42, tzinfo=ET_ZONE),
            None,
            "CLOSED",
            "closed_calendar",
        ),
        (
            "early-close-tail",
            datetime(2026, 11, 27, 13, 7, tzinfo=ET_ZONE),
            None,
            "CLOSED",
            "after_rth_close",
        ),
    )
    for name, observed_at, session, status, reason in cases:
        ledger = LiveCalibrationLedger(tmp_path / f"{name}.jsonl")
        receipt = asyncio.run(
            advance_xsp_shadow_from_ibkr(
                ledger,
                client=_Client(),
                observed_at=observed_at,
                recorded_at=observed_at,
            )
        )

        assert receipt["status"] == "ok"
        assert receipt["session"] == session
        assert receipt["checkpoint_statuses"] == {status: 1}
        assert receipt["broker_request_skipped"] == reason
        assert receipt["contract"] is None
        [checkpoint] = [row for row in ledger.records() if row["kind"] == "checkpoint"]
        assert checkpoint["evidence"]["broker_request_skipped"] == reason
        assert checkpoint["evidence"]["order_authority"] == "none"


def test_ibkr_shadow_boundary_qualifies_and_close_aligns_xsp(tmp_path) -> None:
    class _Client:
        def __init__(self) -> None:
            self.contracts = {
                "XSP": Contract(
                    conId=416904,
                    symbol="XSP",
                    secType="IND",
                    exchange="CBOE",
                    currency="USD",
                ),
            }
            self.requests = {}

        async def qualify_proxy_contracts(self, contract):
            assert contract.secType == "IND"
            return [self.contracts[contract.symbol]]

        async def historical_bars_ohlcv(self, contract, **kwargs):
            self.requests[contract.symbol] = kwargs
            return [
                Bar(
                    datetime(2026, 7, 27, 9, minute),
                    price,
                    price + 0.2,
                    price - 0.2,
                    price,
                    0.0,
                )
                for minute, price in ((30, 100.0), (35, 101.0))
            ]

        def last_historical_request(self, contract):
            return {
                "status": "ok",
                "bars_count": 2,
                "contract": {"symbol": contract.symbol},
            }

    client = _Client()
    observed_at = datetime(2026, 7, 27, 9, 41, tzinfo=ET_ZONE)
    receipt = asyncio.run(
        advance_xsp_shadow_from_ibkr(
            LiveCalibrationLedger(tmp_path / "forward.jsonl"),
            client=client,
            observed_at=observed_at,
            recorded_at=observed_at,
        )
    )

    assert client.requests["XSP"] == {
        "duration_str": XSP_DIRECTIONAL_HISTORY_DURATION,
        "bar_size": "5 mins",
        "use_rth": True,
        "what_to_show": "TRADES",
        "cache_ttl_sec": 0.0,
    }
    assert receipt["status"] == "ok"
    assert receipt["freshness_ok"] is True
    assert receipt["checkpoints"] == 2
    assert receipt["checkpoint_statuses"] == {"EVALUATED": 2}
    assert receipt["processed_bars"] == 2
    assert receipt["latest_bar_close_utc"] == "2026-07-27T13:40:00+00:00"
    assert receipt["recorded_at_utc"] == "2026-07-27T13:41:00+00:00"
    assert receipt["contract"] == {
        "con_id": 416904,
        "symbol": "XSP",
        "sec_type": "IND",
        "exchange": "CBOE",
        "currency": "USD",
    }
    checkpoints = [
        row
        for row in LiveCalibrationLedger(tmp_path / "forward.jsonl").records()
        if row["kind"] == "checkpoint"
    ]
    assert len(checkpoints) == 2
    assert {
        (
            checkpoint["strategy_version"],
            checkpoint["trading_date"],
            checkpoint["session"],
            checkpoint["status"],
        )
        for checkpoint in checkpoints
    } == {
        (
            XSP_DIRECTIONAL_OBSERVER_VERSION,
            "2026-07-27",
            "RTH",
            "EVALUATED",
        ),
        (
            XSP_OPENING_EDGE_VERSION,
            "2026-07-27",
            "RTH",
            "EVALUATED",
        ),
    }
    candidate = next(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["strategy_version"] == XSP_OPENING_EDGE_VERSION
    )
    candidate_equity = candidate["evidence"]["candidate_equity"]
    assert candidate_equity["run_started_at_utc"] == "2026-07-27T13:30:00+00:00"
    assert candidate_equity["config_fingerprint"] == XSP_OPENING_EDGE_CONFIG_FINGERPRINT
    assert candidate_equity["closed_trades"] == 0
    assert candidate_equity["latest_trade"] is None
    assert candidate_equity["order_authority"] == "none"
    assert all(
        checkpoint["evidence"]["cash_history_fresh"] is True
        and checkpoint["evidence"]["order_authority"] == "none"
        and checkpoint["recorded_at_utc"] == "2026-07-27T13:41:00+00:00"
        for checkpoint in checkpoints
    )

    unsupported_at = datetime(2026, 7, 27, 8, 34, tzinfo=ET_ZONE)
    unsupported = asyncio.run(
        advance_xsp_shadow_from_ibkr(
            LiveCalibrationLedger(tmp_path / "gth.jsonl"),
            client=client,
            observed_at=unsupported_at,
            recorded_at=unsupported_at,
        )
    )
    assert unsupported["session"] == "GTH"
    assert unsupported["freshness_ok"] is False
    assert unsupported["checkpoint_statuses"] == {"UNSUPPORTED_SESSION": 1}


def test_first_opening_edge_checkpoint_starts_selected_profitability(
    tmp_path,
) -> None:
    class _Client:
        async def qualify_proxy_contracts(self, contract):
            return [
                Contract(
                    conId=416904,
                    symbol=contract.symbol,
                    secType="IND",
                    exchange="CBOE",
                    currency="USD",
                )
            ]

        async def historical_bars_ohlcv(self, _contract, **_kwargs):
            return [
                Bar(
                    datetime(2026, 7, 27, 9, 30),
                    100.0,
                    100.2,
                    99.8,
                    100.0,
                    0.0,
                )
            ]

        def last_historical_request(self, _contract):
            return {"status": "ok", "bars_count": 1}

    ledger = LiveCalibrationLedger(tmp_path / "selected-forward.jsonl")
    recommendation = xsp_opening_edge_shadow_recommendation()
    selection = xsp_selected_shadow_run(
        ledger,
        recommendation,
        run_id="xsp-opening-edge-20260727",
        strategy_version=XSP_OPENING_EDGE_VERSION,
        config_fingerprint=XSP_OPENING_EDGE_CONFIG_FINGERPRINT,
        capital_sleeve="xsp-directional-unit",
        selected_at=datetime(2026, 7, 27, 9, 0, tzinfo=ET_ZONE),
    )
    observed_at = datetime(2026, 7, 27, 9, 37, tzinfo=ET_ZONE)
    receipt = asyncio.run(
        advance_xsp_shadow_from_ibkr(
            ledger,
            client=_Client(),
            observed_at=observed_at,
            news_snapshot=_news_snapshot(observed_at - timedelta(minutes=5)),
            selected_run=selection,
            recorded_at=observed_at,
        )
    )
    policy = xsp_profitability_policy_from_selected_run(selection)
    profitability = ledger.xsp_profitability_receipt(
        policy=policy,
        as_of=observed_at,
    )
    selected = receipt["selected_equity"]
    checkpoint = next(
        row
        for row in ledger.records()
        if row.get("kind") == "checkpoint"
        and row.get("strategy_version") == XSP_OPENING_EDGE_VERSION
    )

    assert receipt["evaluation_status"] == "EVALUATED"
    assert selected["schema"] == SELECTED_EQUITY_SCHEMA
    assert selected["run_id"] == selection["selection_id"]
    assert selected["cumulative_net_points"] == 0.0
    assert selected["closed_trades"] == 0
    assert selected["order_authority"] == "none"
    pressure = checkpoint["evidence"]["fundamental_pressure"]
    assert set(pressure) == {
        "usable",
        "signal_as_of_utc",
        "snapshot_fingerprint",
        "direction",
        "impact",
        "confidence",
        "age_seconds",
        "horizon_hours",
        "reason",
        "signed_pressure",
        "pressure_delta",
        "pressure_interval_seconds",
        "pressure_velocity_per_hour",
    }
    assert pressure["usable"] is True
    assert pressure["direction"] == -1
    assert pressure["impact"] == 74
    assert pressure["confidence"] == 0.9
    assert pressure["age_seconds"] == 300.0
    assert pressure["horizon_hours"] == 4
    assert pressure["reason"] == "fresh"
    assert pressure["signed_pressure"] == pytest.approx(-0.666)
    assert pressure["pressure_delta"] is None
    assert pressure["pressure_interval_seconds"] is None
    assert pressure["pressure_velocity_per_hour"] is None
    assert profitability["status"] == "ACTIVE"
    assert profitability["clock"]["run_started_at_utc"] == ("2026-07-27T13:30:00+00:00")
    assert profitability["clock"]["coverage_started_at_utc"] == (
        "2026-07-27T13:37:00+00:00"
    )
    assert profitability["clock"]["coverage_broken"] is False
    assert profitability["milestones"]["24h"]["passed"] is False
