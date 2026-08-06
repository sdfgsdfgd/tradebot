from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tradebot.live.traces import (
    _causal_news_at,
    compact_strategy_traces,
    project_execution_trace,
)
from tradebot.news.contract import SCHEMA, SCORE_VERSION, publication_id


RUN = {
    "sleeve_id": "test-sleeve",
    "run_id": "a" * 64,
    "strategy_id": "test.strategy.v1",
    "label": "Test strategy",
}


def _execution(
    checkpoint: str,
    when: str,
    *,
    context: dict[str, object] | None = None,
    source_checkpoint_id: str | None = None,
) -> dict[str, object]:
    plan: dict[str, object] = {
        "status": "HOLD",
        "reason": "weak_direction",
        "target_direction": None,
        "holdings": {"TEST": 0},
    }
    if context is not None:
        plan["execution_state_context"] = context
    if source_checkpoint_id is not None:
        plan["source_checkpoint_id"] = source_checkpoint_id
    return {
        "checkpoint_id": checkpoint,
        "recorded_at_utc": when,
        "evidence": {
            "phase": "STATE",
            "plan": plan,
            "risk_state": {
                "run_net_usd": 0,
                "drawdown_usd": 0,
                "run_cost_usd": 0,
                "fill_count": 0,
                "closed_trades": 0,
            },
        },
    }


def _impulse_context(ratio: float) -> dict[str, object]:
    return {
        "schema": "xsp.execution-state-context.v1",
        "session": "GTH",
        "signal_bar_ts": "2026-08-06T01:00:00+00:00",
        "directional_impulse": {
            "ready": True,
            "direction": None,
            "abstain_reason": "weak_direction",
            "trend_state": "down",
            "coherence": 0.8,
            "atr_ratio": ratio,
            "atr_velocity_pct": -0.01,
            "atr_acceleration_pct": 0.02,
            "horizons": [
                {
                    "bars": 1,
                    "elapsed_minutes": 5,
                    "slope_angle_deg": -20.0,
                    "slope_velocity_pct_per_bar": -0.005,
                }
            ],
        },
        "entry_control": {
            "source": "directional_impulse",
            "proposed_direction": None,
            "controls": ["directional_impulse:abstain"],
        },
        "fundamental_pressure": {
            "authority": "observation_only",
            "direction": 1,
            "impact": 40,
            "confidence": 0.9,
            "signed_pressure": 0.4,
            "pressure_delta": -0.1,
            "pressure_velocity_per_hour": -0.02,
            "snapshot_fingerprint": "news-one",
        },
    }


def test_xsp_trace_compacts_exact_repeats_and_preserves_numeric_delta() -> None:
    first = project_execution_trace(
        _execution(
            "1" * 64, "2026-08-06T01:00:00+00:00", context=_impulse_context(0.7)
        ),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )
    repeated = project_execution_trace(
        _execution(
            "2" * 64, "2026-08-06T01:01:00+00:00", context=_impulse_context(0.7)
        ),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )
    changed = project_execution_trace(
        _execution(
            "3" * 64, "2026-08-06T01:05:00+00:00", context=_impulse_context(0.9)
        ),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )

    compacted = compact_strategy_traces((first, repeated, changed))

    assert len(compacted) == 2
    assert compacted[0]["sample_count"] == 2
    assert compacted[0]["last_event_id"] == "2" * 64
    assert compacted[1]["delta"]["volatility"]["atr_ratio"] == pytest.approx(0.2)
    assert compacted[1]["episode_start"] is False
    assert "execution_state_context" not in compacted[0]


def test_horizon_delta_uses_stable_bar_identity_when_elapsed_minutes_drift() -> None:
    first_context = _impulse_context(0.7)
    next_context = deepcopy(first_context)
    next_horizon = next_context["directional_impulse"]["horizons"][0]
    next_horizon["elapsed_minutes"] = 10
    next_horizon["slope_angle_deg"] = -17.5

    first = project_execution_trace(
        _execution("b" * 64, "2026-08-06T01:00:00+00:00", context=first_context),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )
    changed = project_execution_trace(
        _execution("c" * 64, "2026-08-06T01:05:00+00:00", context=next_context),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )

    compacted = compact_strategy_traces((first, changed))

    assert compacted[1]["horizons"][0]["bars"] == 1
    assert compacted[1]["horizons"][0]["minutes"] == 10
    assert compacted[1]["delta"]["horizons"][0]["bars"] == 1
    assert compacted[1]["delta"]["horizons"][0]["angle"] == pytest.approx(2.5)


def test_gold_trace_preserves_native_daily_h4_macro_and_news_anatomy() -> None:
    context = {
        "schema": "gold.1oz-execution-state-context.v1",
        "signal": {
            "usable": True,
            "raw_direction": "up",
            "proposed_direction": None,
            "daily": {
                "atr14": 89.97,
                "atr_ratio_14_63": 0.945,
                "atr_velocity": 0.0024,
                "hard_direction": "down",
                "soft_direction": "up",
            },
            "h4": {
                "hard_direction": "up",
                "atr14_dollars": 35.75,
                "atr_velocity_dollars": 1.59,
                "signed_fast_slope_dollars": 20.35,
                "signed_fast_acceleration_dollars": 2.99,
                "signed_spread_velocity_dollars": 6.77,
            },
        },
        "macro": {
            "authority": "attribution_only",
            "horizons": {
                "5": {
                    "direction": "mixed",
                    "velocity": "mixed",
                    "acceleration": "mixed",
                },
                "63": {
                    "direction": "adverse",
                    "velocity": "mixed",
                    "acceleration": "adverse",
                },
            },
        },
        "contract_pair": {"contract_month": "2026-12", "basis_usd": 0.075},
        "news": {
            "authority": "attribution_only",
            "direction": 1,
            "impact": 11,
            "confidence": 0.82,
            "change": "strengthening",
            "drivers": ["physical-demand"],
        },
    }

    trace = project_execution_trace(
        _execution("4" * 64, "2026-08-06T01:00:00+00:00", context=context),
        run=RUN,
        trace_key="1OZ",
        records_by_id={},
    )

    assert trace["family"] == "GOLD_REGIME"
    assert trace["volatility"]["atr_ratio"] == 0.945
    assert trace["market"]["h4_fast_slope_dollars"] == 20.35
    assert trace["macro"]["horizons"][1]["direction"] == "adverse"
    assert trace["news"]["authority"] == "attribution_only"


def _news(at: str, *, direction: int, impact: int, change: str) -> dict[str, object]:
    drivers = [
        {
            "id": "lower-oil-driver",
            "event": "Lower-impact oil driver",
            "confidence": 0.91,
            "status": "confirmed",
            "basis": "cross_source_content",
            "mcl": {"direction": -1, "impact": 42},
        },
        {
            "id": "higher-oil-driver",
            "event": "Higher-impact oil driver",
            "confidence": 0.99,
            "status": "confirmed",
            "basis": "cross_source_content",
            "mcl": {"direction": 1, "impact": 76},
        },
    ]
    value = {
        "schema": SCHEMA,
        "score_version": SCORE_VERSION,
        "run_status": "published",
        "signal_as_of_utc": at,
        "snapshot_as_of_utc": at,
        "analysis": {
            "assets": {
                "MCL": {
                    "direction": direction,
                    "impact": impact,
                    "confidence": 0.9,
                    "horizon_hours": 4,
                    "change": change,
                    "drivers": [event["id"] for event in drivers],
                }
            }
        },
        "event_snapshot": {
            "breaking": drivers,
            "day": [],
            "week": [],
            "month": [],
            "persistent": [],
        },
    }
    value["publication_id"] = publication_id(value)
    return value


def test_causal_news_projects_four_hour_daily_and_weekly_pressure_curves() -> None:
    snapshots = (
        _news(
            "2026-07-30T09:00:00Z",
            direction=-1,
            impact=56,
            change="unchanged",
        ),
        _news(
            "2026-08-05T09:00:00Z",
            direction=-1,
            impact=31,
            change="strengthening",
        ),
        _news(
            "2026-08-06T05:50:00Z",
            direction=1,
            impact=12,
            change="reversal",
        ),
        _news(
            "2026-08-06T10:00:00Z",
            direction=1,
            impact=16,
            change="strengthening",
        ),
        _news(
            "2026-08-06T11:00:00Z",
            direction=-1,
            impact=90,
            change="reversal",
        ),
    )

    news = _causal_news_at(
        snapshots,
        symbol="MCL",
        as_of=datetime(2026, 8, 6, 10, 1, tzinfo=timezone.utc),
        authority="non_scoring_context_only",
    )
    windows = {int(row["hours"]): row for row in news["change_windows"]}

    assert news["signed_pressure"] == pytest.approx(0.16)
    assert news["pressure_delta"] == pytest.approx(0.04)
    assert windows[4]["pressure_delta"] == pytest.approx(0.04)
    assert windows[4]["elapsed_hours"] == pytest.approx(4 + 1 / 6)
    assert windows[4]["pressure_velocity_per_hour"] == pytest.approx(0.0096)
    assert windows[24]["pressure_delta"] == pytest.approx(0.47)
    assert windows[24]["pressure_velocity_per_hour"] == pytest.approx(0.47 / 25)
    assert windows[168]["pressure_delta"] == pytest.approx(0.72)
    assert windows[168]["pressure_velocity_per_hour"] == pytest.approx(0.72 / 169)
    assert all(row["available"] is True for row in windows.values())


def test_mcl_trace_resolves_immutable_source_and_news_without_lookahead() -> None:
    source_id = "5" * 64
    source = {
        "checkpoint_id": source_id,
        "recorded_at_utc": "2026-08-06T02:00:10+00:00",
        "evidence": {
            "schema": "mcl.two-speed-auction-source-checkpoint.v1",
            "source": {
                "authority": "finalized_source_only_no_orders_no_capital",
                "contract_month": "202608",
                "latest_common_close_utc": "2026-08-06T02:00:00+00:00",
                "latest_decision": {
                    "observed_at_utc": "2026-08-06T02:00:00+00:00",
                    "raw_direction": None,
                    "proposed_direction": None,
                    "snapshot": {
                        "ready": True,
                        "direction": None,
                        "abstain_reason": "weak_direction",
                        "trend_state": "up",
                        "atr_ratio": 1.006,
                        "atr_velocity_pct": -0.0021,
                        "atr_acceleration_pct": -0.0168,
                        "horizons": [
                            {
                                "bars": 6,
                                "elapsed_minutes": 30,
                                "slope_angle_deg": -21.7,
                                "slope_velocity_pct_per_bar": 0.011,
                            }
                        ],
                    },
                },
                "last_raw_turn": {
                    "event_id": "6" * 64,
                    "observed_at_utc": "2026-08-06T01:25:00+00:00",
                    "direction": "up",
                },
            },
        },
    }
    prior = _news("2026-08-06T00:00:00Z", direction=-1, impact=20, change="new")
    visible = _news(
        "2026-08-06T01:00:00Z",
        direction=1,
        impact=50,
        change="reversal",
    )
    future = _news(
        "2026-08-06T03:00:00Z",
        direction=-1,
        impact=90,
        change="reversal",
    )

    trace = project_execution_trace(
        _execution(
            "7" * 64,
            "2026-08-06T02:00:11+00:00",
            source_checkpoint_id=source_id,
        ),
        run=RUN,
        trace_key="MCL",
        records_by_id={source_id: source},
        news_snapshots=(prior, visible, future),
    )

    assert trace["family"] == "MCL_IMPULSE"
    assert trace["volatility"]["atr_ratio"] == 1.006
    assert trace["horizons"][0]["angle"] == -21.7
    assert trace["long_context"]["last_raw_turn_id"] == "6" * 64
    assert trace["news"]["snapshot_fingerprint"] == visible["publication_id"]
    assert trace["news"]["signed_pressure"] == 0.5
    assert trace["news"]["pressure_delta"] == 0.7
    assert trace["news"]["authority"] == "non_scoring_context_only"
    assert [row["id"] for row in trace["news"]["driver_scores"]] == [
        "higher-oil-driver",
        "lower-oil-driver",
    ]
    assert trace["news"]["driver_scores"][0] == {
        "id": "higher-oil-driver",
        "label": "Higher-impact oil driver",
        "direction": 1,
        "impact": 76,
        "confidence": 0.99,
        "status": "confirmed",
        "basis": "cross_source_content",
        "bucket": "breaking",
    }

    next_source_id = "9" * 64
    next_source = deepcopy(source)
    next_source["checkpoint_id"] = next_source_id
    next_source["recorded_at_utc"] = "2026-08-06T02:01:10+00:00"
    next_snapshot = next_source["evidence"]["source"]
    next_snapshot["latest_decision"]["observed_at_utc"] = "2026-08-06T02:01:00+00:00"
    next_snapshot["latest_decision"]["cl_move"] = 0.01
    next_snapshot["rows"] = {"CL": 1001, "MCL": 1001, "common": 1001}
    repeated = project_execution_trace(
        _execution(
            "a" * 64,
            "2026-08-06T02:01:11+00:00",
            source_checkpoint_id=next_source_id,
        ),
        run=RUN,
        trace_key="MCL",
        records_by_id={next_source_id: next_source},
        news_snapshots=(prior, visible, future),
    )

    compacted = compact_strategy_traces((trace, repeated))

    assert len(compacted) == 1
    assert compacted[0]["sample_count"] == 2
    assert compacted[0]["decision"]["cl_move"] == 0.01


def test_compaction_input_is_not_mutated() -> None:
    projected = project_execution_trace(
        _execution(
            "8" * 64, "2026-08-06T01:00:00+00:00", context=_impulse_context(0.7)
        ),
        run=RUN,
        trace_key="XSP",
        records_by_id={},
    )
    before = deepcopy(projected)

    compact_strategy_traces((projected,))

    assert projected == before
