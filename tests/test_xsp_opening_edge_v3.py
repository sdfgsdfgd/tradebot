from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from tradebot.backtest.models import Bar
from tradebot.research.live_calibration import (
    LiveCalibrationLedger,
    calibration_fingerprint,
)
from tradebot.research.xsp_benchmarks import (
    xsp_fundamental_defensive_benchmark,
)
from tradebot.research.xsp_opening_edge_v2 import (
    load_xsp_opening_edge_v2_spec,
)
from tradebot.research.xsp_opening_edge_v3 import (
    XSP_OPENING_EDGE_V3_EXECUTION_GATE,
    XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
    XSP_OPENING_EDGE_V3_VERSION,
    _daily_context,
    _persisted_daily_context,
    advance_xsp_opening_edge_p009_from_ibkr,
    advance_xsp_opening_edge_v3_from_ibkr,
    is_xsp_v3_run_start,
    load_xsp_opening_edge_v3_spec,
    next_xsp_v3_run_start,
    xsp_opening_edge_v3_bundle,
    xsp_opening_edge_v3_equities,
    xsp_opening_edge_v3_fundamental_pairs,
    xsp_opening_edge_v3_run_start,
)
from tradebot.research.xsp_opening_edge_state import (
    XspOpeningEdgeV3StateOwner,
    merge_xsp_daily_context,
    xsp_daily_bars_from_intraday,
)
from tradebot.time_utils import ET_ZONE


def test_p009_keeps_slow_history_but_bounds_native_minute_payload(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    async def _advance(_ledger, **kwargs):
        captured.update(kwargs)
        return {"order_authority": "none"}

    monkeypatch.setattr(
        "tradebot.research.xsp_opening_edge_v3."
        "advance_xsp_opening_edge_v2_from_ibkr",
        _advance,
    )
    observed = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)
    asyncio.run(
        advance_xsp_opening_edge_p009_from_ibkr(
            LiveCalibrationLedger(tmp_path / "p009-minute-window.jsonl"),
            client=object(),
            observed_at=observed,
            run_started_at=observed - timedelta(hours=14),
            recorded_at=observed,
            spec=object(),
        )
    )

    assert captured["duration_str"] == "2 W"
    assert captured["rth_one_minute_duration_str"] == "1 D"
    assert captured["include_rth_one_minute_context"] is True


def test_opening_edge_v3_is_content_addressed_and_v2_remains_reproducible() -> None:
    v3 = load_xsp_opening_edge_v3_spec()
    v2 = load_xsp_opening_edge_v2_spec()
    rth = xsp_opening_edge_v3_bundle(
        v3,
        lane="rth",
        start=date(2026, 7, 29),
        end=date(2026, 7, 29),
        cost_profile="research",
    )
    gth = xsp_opening_edge_v3_bundle(
        v3,
        lane="gth",
        start=date(2026, 7, 29),
        end=date(2026, 7, 29),
        cost_profile="research",
    )

    assert XSP_OPENING_EDGE_V3_VERSION == ("xsp.opening-edge-v3-regime-harmony-24x5.v1")
    assert v3.artifact_sha256 == (
        "d47eb39cef3d2ca575d779d6b5b87e3b88e08606fd09a8801b8cb55c350208db"
    )
    assert v3.state_owner_sha256 == (
        "643b6a04478e598fe7fbdc2d5aa6deabe6935761b3fa83cf487a9fdd16a265bc"
    )
    assert v3.daily_context_seed_sha256 == (
        "177b4163fcdee409378e7b6384b15b3554d7e25438714ef2feb7633e04896b1f"
    )
    assert len(v3.daily_context_seed) == 1257
    assert v3.declaration_version == "3"
    assert v3.group["authority"] == "historical_research_crown_only"
    assert rth.strategy.symbol == "SPY"
    assert rth.backtest.use_rth is True
    assert gth.strategy.symbol == "XSP"
    assert gth.backtest.use_rth is False
    assert XSP_OPENING_EDGE_V3_EXECUTION_GATE["eligible"] is False
    assert XSP_OPENING_EDGE_V3_EXECUTION_GATE["order_authority"] == "none"

    assert v2.artifact_sha256 == (
        "f879cc20c4434e33626c143ccd85db4d608370a6fb7c321b1ee0f1f2c08afff2"
    )
    assert v2.declaration_version == "2"


def test_opening_edge_v3_daily_context_is_causal_and_restart_stable() -> None:
    start = datetime(2026, 1, 1, 16, 0)
    bars = tuple(
        Bar(
            start + timedelta(days=index),
            100.0 + index,
            101.0 + index,
            99.0 + index,
            100.5 + index,
            1.0,
        )
        for index in range(104)
    )
    observed_at = (start + timedelta(days=104, hours=1)).replace(tzinfo=ET_ZONE)
    daily = _daily_context(
        bars,
        observed_at=observed_at,
        naive_ts_mode="et",
    )
    first = XspOpeningEdgeV3StateOwner(daily)
    restarted = XspOpeningEdgeV3StateOwner(daily)
    next_day = daily[-1].day + timedelta(days=1)

    assert len(daily) == 104
    assert first.context_for_day(next_day) == restarted.context_for_day(next_day)
    assert first.context_fingerprint == restarted.context_fingerprint
    assert first.context_for_day(daily[93].day) is None
    assert first.context_for_day(daily[94].day) is not None

    with pytest.raises(ValueError, match="underwarmed"):
        _daily_context(
            bars[:93],
            observed_at=observed_at,
            naive_ts_mode="et",
        )
    with pytest.raises(ValueError, match="duplicate"):
        _daily_context(
            (*bars, bars[-1]),
            observed_at=observed_at,
            naive_ts_mode="et",
        )


def _complete_rth(day: date) -> tuple[Bar, ...]:
    first = datetime.combine(
        day,
        datetime.min.time(),
        tzinfo=ET_ZONE,
    ) + timedelta(hours=9, minutes=35)
    return tuple(
        Bar(
            (first + timedelta(minutes=5 * index))
            .astimezone(timezone.utc)
            .replace(tzinfo=None),
            742.73 if index == 0 else 743.0,
            745.0 if index == 20 else 743.5,
            740.0 if index == 30 else 742.0,
            744.0 if index == 77 else 743.0,
            1.0,
        )
        for index in range(78)
    )


def test_opening_edge_v3_context_appends_only_complete_exact_rth() -> None:
    spec = load_xsp_opening_edge_v3_spec()
    complete = _complete_rth(date(2026, 7, 29))
    [fresh] = xsp_daily_bars_from_intraday(complete)
    merged = merge_xsp_daily_context(spec.daily_context_seed, fresh=(fresh,))
    prior_close = datetime(2026, 7, 28, 16, 0, tzinfo=ET_ZONE).astimezone(
        timezone.utc
    ).replace(tzinfo=None)
    gth_close = datetime(2026, 7, 29, 8, 35, tzinfo=ET_ZONE).astimezone(
        timezone.utc
    ).replace(tzinfo=None)
    paired = xsp_opening_edge_v3_equities(
        spec=spec,
        spy_bars=(
            Bar(prior_close, 632.0, 632.0, 632.0, 632.0, 1.0),
            Bar(gth_close, 631.0, 631.2, 630.8, 631.1, 1.0),
            *(
                Bar(row.ts, 632.0, 632.1, 631.9, 632.0, 1.0)
                for row in complete
            ),
        ),
        xsp_rth_bars=(
            Bar(prior_close, 742.73, 742.73, 742.73, 742.73, 1.0),
            *complete,
        ),
        observed_at=datetime(2026, 7, 29, 16, 17, tzinfo=ET_ZONE),
        run_started_at=datetime(2026, 7, 29, 8, 30, tzinfo=ET_ZONE),
    )
    persisted = {
        "strategy_version": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "evidence": {"paired_equity": paired},
    }

    assert len(merged) == 1258
    assert merged[-1].day == date(2026, 7, 29)
    assert paired["daily_context_appends"] == [
        {
            "day": fresh.day.isoformat(),
            "open": fresh.open,
            "high": fresh.high,
            "low": fresh.low,
            "close": fresh.close,
        }
    ]
    expected_context = XspOpeningEdgeV3StateOwner(merged).context_for_day(
        date(2026, 7, 29)
    )
    assert paired["daily_context_state"] == {
        "schema": "xsp.opening-edge-v3-daily-context-state.v1",
        "trading_day": "2026-07-29",
        "context_as_of_day": "2026-07-28",
        "state_fingerprint": calibration_fingerprint(expected_context),
        "state": expected_context,
    }
    assert set(paired["daily_context_state"]["state"]["windows"]) >= {
        "21",
        "63",
        "84",
    }
    assert set(
        paired["daily_context_state"]["state"]["return_velocity"]
    ) >= {"21", "63", "84"}
    assert set(
        paired["daily_context_state"]["state"]["return_acceleration"]
    ) >= {"21", "63", "84"}
    assert _persisted_daily_context((persisted,)) == (fresh,)
    assert xsp_daily_bars_from_intraday(complete[:20]) == ()
    assert xsp_daily_bars_from_intraday(complete[20:]) == ()

    with pytest.raises(ValueError, match="incomplete"):
        xsp_daily_bars_from_intraday((*complete[:20], complete[-1]))
    with pytest.raises(ValueError, match="overlap drifted"):
        merge_xsp_daily_context(
            spec.daily_context_seed,
            fresh=(replace(spec.daily_context_seed[-1], close=742.74),),
        )
    with pytest.raises(ValueError, match="known gap"):
        merge_xsp_daily_context(
            spec.daily_context_seed,
            fresh=(replace(fresh, day=date(2026, 7, 31)),),
        )


def test_opening_edge_v3_persisted_context_ignores_predecessor_records() -> None:
    predecessor = {
        "strategy_version": "xsp.opening-edge-v2-spy-transport.v1",
        "evidence": {"paired_equity": {}},
    }
    prefreeze = {
        "strategy_version": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "evidence": {"paired_equity": {}},
    }

    assert _persisted_daily_context((predecessor, prefreeze)) == ()

    malformed = {
        "strategy_version": XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
        "evidence": {"paired_equity": {"daily_context_appends": None}},
    }
    with pytest.raises(ValueError, match="persisted context is malformed"):
        _persisted_daily_context((malformed,))


def test_opening_edge_v3_news_pairs_require_actual_entry_and_exact_outcome(
    tmp_path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "v3-news-pairs.jsonl")
    decision_at = datetime(2026, 7, 30, 14, 55, tzinfo=timezone.utc)
    entry_at = decision_at + timedelta(minutes=5)
    trading_day = "2026-07-30"
    trade = {
        "lane": "rth",
        "direction": "up",
        "entry_time": entry_at.replace(tzinfo=None).isoformat(),
        "attribution": {
            "entry": {
                "signal_bar_ts": decision_at.replace(tzinfo=None).isoformat(),
            }
        },
    }

    def checkpoint(
        signal_at: datetime,
        *,
        close: float,
        direction: str | None = None,
        position: Mapping[str, object] | None = None,
    ) -> None:
        evaluated_at = signal_at + timedelta(minutes=2)
        ledger.checkpoint(
            evaluation_as_of=evaluated_at,
            strategy_id=XSP_OPENING_EDGE_V3_VERSION,
            strategy_version=XSP_OPENING_EDGE_V3_TRANSPORT_VERSION,
            trading_date=trading_day,
            session="RTH",
            status="EVALUATED",
            evidence={
                "paired_equity": {
                    "signal_observations": {
                        "rth": {
                            "signal_bar_ts": signal_at.replace(
                                tzinfo=None
                            ).isoformat(),
                            "signal_snapshot_age_bars": 0,
                            "close": close,
                            "entry_control": {
                                "direction": direction,
                                "proposed_direction": direction,
                            },
                            "directional_impulse": {"turn_event": direction},
                        }
                    },
                    "profiles": {
                        "research": {
                            "latest_position": position,
                            "latest_trade": None,
                        }
                    },
                    "daily_context_state": {
                        "state": {
                            "transition": "transition_down",
                            "tr_phase": "high_up",
                        }
                    },
                },
                "fundamental_pressure": {
                    "source": "causal_news",
                    "authority": "observation_only",
                    "usable": True,
                    "direction": -1,
                    "impact": 90,
                    "confidence": 0.95,
                    "snapshot_fingerprint": "news-at-decision",
                    "signal_as_of_utc": "2026-07-30T14:00:00+00:00",
                    "signed_pressure": -0.855,
                    "pressure_delta": -0.05,
                    "pressure_velocity_per_hour": -0.01,
                },
            },
            recorded_at=evaluated_at,
        )

    checkpoint(decision_at, close=100.0, direction="up")
    checkpoint(entry_at, close=99.8, position=trade)
    checkpoint(
        decision_at + timedelta(minutes=15),
        close=100.3,
        direction="down",
        position=trade,
    )
    checkpoint(
        decision_at + timedelta(minutes=60),
        close=98.0,
        position=trade,
    )
    checkpoint(
        decision_at + timedelta(minutes=75),
        close=101.0,
        position=trade,
    )

    pairs = xsp_opening_edge_v3_fundamental_pairs(tuple(ledger.records()))
    assert len(pairs) == 1
    assert pairs[0]["decision_at"] == decision_at
    assert pairs[0]["direction"] == "up"
    assert pairs[0]["ta_points"] == pytest.approx(-2.1)
    assert pairs[0]["context"]["decision_close"] == 100.0
    assert pairs[0]["context"]["outcome_close"] == 98.0

    benchmark = xsp_fundamental_defensive_benchmark(
        ledger,
        settled_pairs=pairs,
        prospective_evidence_mode="forward_v3_checkpoint",
    )
    assert benchmark["pairs"] == benchmark["prospective_pairs"] == 1
    assert benchmark["vetoes"] == benchmark["prospective_vetoes"] == 1
    assert benchmark["ta_observer_points"] == pytest.approx(-2.1)
    assert benchmark["defended_observer_points"] == 0.0
    assert benchmark["paired_delta_points"] == pytest.approx(2.1)
    assert benchmark["policy"]["prospective_evidence_mode"] == (
        "forward_v3_checkpoint"
    )


def test_opening_edge_v3_prefreezes_without_broker_or_backfill(tmp_path) -> None:
    class _NoBroker:
        async def qualify_proxy_contracts(self, _contract):
            raise AssertionError("v3 preflight must not contact IBKR")

    ledger = LiveCalibrationLedger(tmp_path / "v3-preflight.jsonl")
    observed_at = datetime(2026, 7, 29, 17, 5, tzinfo=ET_ZONE)
    run_start = datetime(2026, 7, 29, 20, 15, tzinfo=ET_ZONE)
    receipt = asyncio.run(
        advance_xsp_opening_edge_v3_from_ibkr(
            ledger,
            client=_NoBroker(),
            observed_at=observed_at,
            run_started_at=run_start,
            recorded_at=observed_at,
        )
    )

    assert receipt["evaluation_status"] == "CLOSED"
    assert receipt["broker_request_skipped"] == "run_not_started"
    assert receipt["order_authority"] == "none"
    [checkpoint] = list(ledger.records())
    assert checkpoint["strategy_id"] == XSP_OPENING_EDGE_V3_VERSION
    assert checkpoint["strategy_version"] == XSP_OPENING_EDGE_V3_TRANSPORT_VERSION
    assert xsp_opening_edge_v3_run_start(
        tuple(ledger.records()),
        observed_at=observed_at,
    ) == run_start.astimezone(timezone.utc)
    next_intraday = next_xsp_v3_run_start(datetime(2026, 7, 29, 7, 42, tzinfo=ET_ZONE))
    next_post_rth = next_xsp_v3_run_start(datetime(2026, 7, 29, 9, 31, tzinfo=ET_ZONE))
    assert next_intraday == datetime(
        2026,
        7,
        29,
        11,
        45,
        tzinfo=timezone.utc,
    )
    assert next_post_rth == datetime(
        2026,
        7,
        30,
        0,
        15,
        tzinfo=timezone.utc,
    )
    assert is_xsp_v3_run_start(next_intraday)
    intraday = asyncio.run(
        advance_xsp_opening_edge_v3_from_ibkr(
            LiveCalibrationLedger(tmp_path / "v3-intraday-preflight.jsonl"),
            client=_NoBroker(),
            observed_at=datetime(2026, 7, 29, 7, 42, tzinfo=ET_ZONE),
            run_started_at=next_intraday,
            recorded_at=datetime(2026, 7, 29, 7, 42, tzinfo=ET_ZONE),
        )
    )
    assert intraday["broker_request_skipped"] == "run_not_started"
    assert intraday["run_started_at_utc"] == next_intraday.isoformat()


@pytest.mark.parametrize(
    "broker_request_skipped",
    ("run_not_started", "closed_calendar"),
)
def test_shadow_cli_v3_accepts_safe_closed_noop_without_v2_execution_authority(
    tmp_path,
    monkeypatch,
    capsys,
    broker_request_skipped,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    run_start = datetime(2026, 7, 30, 0, 15, tzinfo=timezone.utc)

    class _Client:
        def __init__(self, _config):
            pass

        async def disconnect(self):
            captured["disconnected"] = True

    async def _advance(_ledger, **kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "evaluation_status": "CLOSED",
            "broker_request_skipped": broker_request_skipped,
            "order_authority": "none",
        }

    async def _forbidden(*_args, **_kwargs):
        raise AssertionError("v3 must not enter a v2 cash execution owner")

    def _failed_observation(*_args, **_kwargs):
        assert captured["disconnected"] is True
        raise RuntimeError("synthetic pressure-ledger failure")

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_opening_edge_v3_spec",
        lambda: "v3-spec",
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_opening_edge_v3_run_start",
        lambda *_args, **_kwargs: run_start,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_opening_edge_v3_from_ibkr",
        _advance,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_v2_etf_execution_observer",
        _forbidden,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.accumulate_xsp_pressure_atlas",
        _failed_observation,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v3",
                    "--ledger",
                    str(tmp_path / "v3-cli.jsonl"),
                    "--news-signal",
                    str(tmp_path / "missing-news.json"),
                    "--selected-transport",
                    str(tmp_path / "obsolete-v2-transport.json"),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert captured["run_started_at"] == run_start
    assert captured["spec"] == "v3-spec"
    assert captured["disconnected"] is True
    assert output["mode"] == "opening-edge-v3"
    assert output["v3_run_started_at_utc"] == run_start.isoformat()
    assert output["selected_transport_id"] is None
    assert output["order_authority"] == "none"
    assert output["pressure_atlas_accumulation"] == {
        "schema": "xsp.pressure-atlas-accumulation-status.v1",
        "status": "OBSERVATION_ERROR",
        "error_type": "RuntimeError",
        "error": "synthetic pressure-ledger failure",
        "live_invocation_changed": False,
        "permission": "none",
        "outcomes": None,
        "order_authority": "none",
        "capital_authority": "none",
        "submitted_orders": 0,
    }


def test_shadow_cli_p009_uses_the_dual_clock_source_without_a_second_service(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    captured = {}
    run_start = datetime(2026, 8, 8, 0, 20, tzinfo=timezone.utc)

    class _Client:
        def __init__(self, _config):
            pass

        async def disconnect(self):
            captured["disconnected"] = True

    async def _advance(_ledger, **kwargs):
        captured.update(kwargs)
        return {
            "evaluation_status": "CLOSED",
            "broker_request_skipped": "run_not_started",
            "order_authority": "none",
        }

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_opening_edge_v3_spec",
        lambda: "p009-spec",
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_opening_edge_p009_run_start",
        lambda *_args, **_kwargs: run_start,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_opening_edge_p009_from_ibkr",
        _advance,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.accumulate_xsp_pressure_atlas",
        lambda *_args, **_kwargs: {"submitted_orders": 0},
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-p009",
                    "--ledger",
                    str(tmp_path / "p009-cli.jsonl"),
                    "--news-signal",
                    str(tmp_path / "missing-news.json"),
                    "--selected-transport",
                    str(tmp_path / "no-selection.json"),
                    "--capital-plan",
                    str(tmp_path / "no-capital.json"),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert captured["run_started_at"] == run_start
    assert captured["spec"] == "p009-spec"
    assert captured["disconnected"] is True
    assert output["mode"] == "opening-edge-p009"
    assert output["v3_run_started_at_utc"] == run_start.isoformat()
    assert output["selected_transport_id"] is None
    assert output["order_authority"] == "none"


def test_shadow_cli_v3_closed_source_never_enters_selected_cash_transport(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from tradebot.research.xsp_shadow_cli import _main_async

    selection_path = tmp_path / "selected.json"
    selection_path.write_text("{}")
    selected = {
        "selection_id": "1" * 64,
        "strategy_version": XSP_OPENING_EDGE_V3_VERSION,
        "order_authority": "rth_cash_pair_limit_only",
    }
    run_start = datetime(2026, 7, 30, 0, 15, tzinfo=timezone.utc)

    class _Client:
        def __init__(self, _config):
            pass

        async def disconnect(self):
            pass

    async def _advance(_ledger, **_kwargs):
        return {
            "status": "ok",
            "evaluation_status": "CLOSED",
            "broker_request_skipped": "closed_calendar",
            "order_authority": "none",
        }

    async def _forbidden(*_args, **_kwargs):
        raise AssertionError("closed source must not enter selected cash transport")

    monkeypatch.setattr("tradebot.client.IBKRClient", _Client)
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_opening_edge_v3_spec",
        lambda: "v3-spec",
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_opening_edge_v3_run_start",
        lambda *_args, **_kwargs: run_start,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_opening_edge_v3_from_ibkr",
        _advance,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.load_xsp_v3_transport_selection",
        lambda _path: selected,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.xsp_v3_transport_profitability_policy",
        lambda _selection: None,
    )
    monkeypatch.setattr(
        "tradebot.research.xsp_shadow_cli.advance_xsp_live_transport",
        _forbidden,
    )

    assert (
        asyncio.run(
            _main_async(
                (
                    "--mode",
                    "opening-edge-v3",
                    "--ledger",
                    str(tmp_path / "ledger.jsonl"),
                    "--news-signal",
                    str(tmp_path / "missing-news.json"),
                    "--selected-transport",
                    str(selection_path),
                    "--capital-plan",
                    str(tmp_path / "missing-capital.json"),
                )
            )
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["selected_transport_id"] == selected["selection_id"]
    assert output["transport_execution"] is None
    assert output["order_authority"] == "rth_cash_pair_limit_only"
