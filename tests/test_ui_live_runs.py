from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from textual.app import App

from tradebot.ui.app import PositionsApp
from tradebot.ui.bot import BotScreen
from tradebot.ui.bot_screen.live_runs import BotLiveRunsMixin
from tradebot.ui.bot_screen.portfolio import BotPortfolioMixin
from tradebot.ui.bot_screen.traces import BotTraceMixin
from tradebot.ui.footer import TradebotFooter


class _TableStub:
    def __init__(self) -> None:
        self._cursor_coordinate = SimpleNamespace(row=0)
        self.rows: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.clear_calls = 0
        self.add_calls = 0
        self.remove_calls = 0
        self.update_calls = 0

    @property
    def cursor_coordinate(self):
        return self._cursor_coordinate

    @cursor_coordinate.setter
    def cursor_coordinate(self, value) -> None:
        row = value[0] if isinstance(value, tuple) else value.row
        self._cursor_coordinate = SimpleNamespace(row=row)

    def clear(self, columns: bool = False) -> None:
        self.clear_calls += 1
        self.rows = []

    def add_column(self, *_args, **_kwargs) -> None:
        return

    def add_row(self, *values, **kwargs) -> None:
        self.add_calls += 1
        self.rows.append((values, kwargs))

    def remove_row(self, key: str) -> None:
        self.remove_calls += 1
        self.rows = [row for row in self.rows if row[1].get("key") != key]

    def update_cell(self, row_key: str, column_key: str, value) -> None:
        self.update_calls += 1
        index = {"when": 0, "trace": 1}[column_key]
        for position, (values, kwargs) in enumerate(self.rows):
            if kwargs.get("key") == row_key:
                cells = list(values)
                cells[index] = value
                self.rows[position] = (tuple(cells), kwargs)
                return
        raise KeyError(row_key)


class _Harness(BotPortfolioMixin, BotLiveRunsMixin, BotTraceMixin):
    def _sync_row_marker(self, *_args, **_kwargs) -> None:
        return


def _run() -> dict[str, object]:
    allow = {"status": "ALLOW", "reasons": []}
    hold = {"status": "HOLD", "reasons": ["successor_required"]}
    return {
        "sleeve_id": "xsp-upro-spxu-rth-cash",
        "strategy_id": "xsp.v3",
        "run_id": "a" * 64,
        "label": "XSP v3 Regime Harmony · UPRO/SPXU RTH",
        "valid": True,
        "state": "RUNNING",
        "errors": [],
        "allocation": {"weight_bps": 10_000, "limit_cents": 90_046},
        "timer": {"active_state": "active"},
        "service": {"active_state": "inactive"},
        "positions": {"UPRO": 0, "SPXU": 0},
        "open_orders": [],
        "pending_order_refs": [],
        "settled_cash_usd": 1_318.05,
        "economics": {
            "run_net_usd": 0.0,
            "run_realized_net_usd": 0.0,
            "open_mark_net_usd": 0.0,
            "run_cost_usd": 0.0,
            "drawdown_usd": 0.0,
            "fill_count": 0,
            "closed_trades": 0,
        },
        "campaign_economics": {
            "known_net_usd": -22.156381,
            "known_realized_net_usd": -22.156381,
            "active_open_mark_net_usd": 0.0,
            "closed_trades": 3,
            "selection_runs": 7,
            "accounted_selection_runs": 7,
            "attribution_complete": False,
        },
        "safety": {"valid": True, "attribution_complete": True, "breaches": []},
        "graduation": {
            "verdict": "HOLD",
            "target": "24h",
            "reasons": ["profitability:eligible_sessions_incomplete"],
        },
        "controls": {
            "START": {"status": "NOOP", "reasons": ["schedule_already_active"]},
            "STOP": allow,
            "REPLACE": hold,
            "REBALANCE": hold,
        },
        "latest_decision": {"status": "HOLD", "reason": "weak_direction"},
        "execution_state_context": {
            "signal_session": "RTH",
            "signal_bar_ts": "2026-07-31T20:00:00",
            "signal_snapshot_age_bars": 3,
            "entry_control": {
                "source": "directional_impulse",
                "proposed_direction": None,
                "blocked_by": None,
                "controls": ["directional_impulse:abstain"],
            },
            "directional_impulse": {
                "abstain_reason": "weak_direction",
                "trend_state": "up",
                "direction": None,
                "coherence": 0.6,
                "atr_ratio": 2.0647,
                "atr_velocity_pct": 0.06327,
                "atr_acceleration_pct": 0.03933,
                "horizons": [
                    {
                        "elapsed_minutes": 5,
                        "slope_angle_deg": -42.79,
                        "slope_velocity_pct_per_bar": -0.2969,
                    },
                    {
                        "elapsed_minutes": 120,
                        "slope_angle_deg": 4.73,
                        "slope_velocity_pct_per_bar": -0.0105,
                    },
                ],
            },
            "daily_context_state": {
                "state": {
                    "directions": {
                        "5": "up",
                        "10": "down",
                        "21": "down",
                        "42": "down",
                        "63": "up",
                        "84": "up",
                    },
                    "transition": "transition_down",
                    "hard_direction": "up",
                    "soft_direction": "up",
                    "tr_phase": "high_flat",
                }
            },
            "fundamental_pressure": {
                "signed_pressure": 0.5044,
                "pressure_delta": 0.0097,
                "pressure_velocity_per_hour": 0.002315,
                "confidence": 0.97,
            },
        },
    }


def test_bot_trade_has_only_durable_candidate_run_and_evidence_panels() -> None:
    assert "bot-presets" not in BotScreen._PANEL_BY_TABLE_ID
    assert "bot-instances" not in BotScreen._PANEL_BY_TABLE_ID
    assert "bot-orders" not in BotScreen._PANEL_BY_TABLE_ID
    assert BotScreen._PANEL_BY_TABLE_ID["bot-candidates"] == "candidates"
    assert BotScreen._PANEL_BY_TABLE_ID["bot-live-runs"] == "live_runs"
    assert BotScreen._PANEL_ORDER == (
        "candidates",
        "live_runs",
        "activity",
        "traces",
    )
    screen = BotScreen(client=SimpleNamespace(), refresh_sec=1.0)
    assert not hasattr(screen, "_client")
    bindings = {binding.key: binding for binding in BotScreen.BINDINGS}
    assert bindings["p"].action == "toggle_candidates"
    assert bindings["p"].description == "Candidates"
    assert bindings["ctrl+a"].action == "toggle_candidates"
    assert bindings["ctrl+a"].show is False
    assert bindings["left"].action == "trace_prev"
    assert bindings["left"].priority is True
    assert bindings["right"].action == "trace_next"
    assert bindings["right"].priority is True
    assert BotScreen.AUTO_FOCUS == "#bot-trace-all"
    assert not hasattr(screen, "_instances")
    assert not hasattr(screen, "_orders")


def test_live_run_table_projects_durable_truth_not_local_bot_instances() -> None:
    table = _TableStub()
    harness = _Harness()
    harness._live_runs_table = table
    harness._live_run_rows = []
    snapshot = {"runs": [_run()]}

    BotLiveRunsMixin._render_live_runs_table(harness, snapshot)

    assert len(table.rows) == 1
    assert table.rows[0][1]["key"] == "live:xsp-upro-spxu-rth-cash"
    assert harness._live_run_rows[0]["run_id"] == "a" * 64
    assert table.rows[0][0][4].plain == "0.00 / -22.16"


def test_live_run_capital_projects_executable_package_not_legacy_weight() -> None:
    assert (
        BotLiveRunsMixin._live_run_capital(
            {
                "capital_kind": "CASH_DEBIT",
                "package_id": "xsp-usd-800",
                "cash_debit_cents": 80_046,
                "initial_margin_base_cents": 0,
            }
        )
        == "$800.46 · xsp-usd-800"
    )
    assert (
        BotLiveRunsMixin._live_run_capital(
            {
                "capital_kind": "FUTURES_MARGIN",
                "package_id": "gold-one-contract",
                "cash_debit_cents": 66,
                "initial_margin_base_cents": 60_000,
            }
        )
        == "gold-one-contract · margin 600"
    )


def test_stop_key_routes_live_runs_to_flat_safe_durable_controller() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    screen = BotScreen(client=SimpleNamespace(), refresh_sec=1.0)
    calls: list[str] = []
    screen._active_panel = "live_runs"
    screen._stop_live_run = lambda: calls.append("durable")

    screen.action_stop_bot()

    assert calls == ["durable"]


def test_replace_and_rebalance_are_explicit_artifact_gates() -> None:
    run = _run()

    assert "x:gate" in BotLiveRunsMixin._live_run_controls(run)
    assert "b:gate" in BotLiveRunsMixin._live_run_controls(run)
    details = _Harness()._live_run_detail_lines(run)
    assert any("immutable successor artifacts" in line.plain for line in details)
    assert any(
        "Known live campaign" in line.plain
        and "Realized=$-22.16" in line.plain
        and "Attribution=PARTIAL" in line.plain
        for line in details
    )


def test_live_run_details_reuse_persisted_hawkeye_anatomy() -> None:
    details = [line.plain for line in _Harness()._live_run_detail_lines(_run())]

    assert any("Hawkeye: HOLD / weak_direction" in line for line in details)
    assert any("Gate: source=directional_impulse" in line for line in details)
    assert any("Slope angles" in line and "5m:-42.8°" in line for line in details)
    assert any("Slope velocity" in line and "120m:-0.0105" in line for line in details)
    assert any("ATR ratio=+2.065" in line for line in details)
    assert any("Long context" in line and "transition_down" in line for line in details)
    assert any(
        "News attribution" in line and "pressure=+0.5044" in line for line in details
    )


def test_timeline_details_preserve_the_persisted_execution_ladder() -> None:
    details = [
        line.plain
        for line in _Harness()._timeline_detail_lines(
            {
                "recorded_at_utc": "2026-07-30T16:45:47+00:00",
                "kind": "EXECUTION",
                "phase": "SUBMITTED",
                "status": "ACTIONABLE",
                "reason": "sell_incumbent_before_target",
                "message": "target=flat SELL 23 SPXU",
                "execution_detail": {
                    "order_ref": "XSPV3-test",
                    "ladder_transition": {
                        "previous_mode": "OPT",
                        "active_mode": "MID",
                        "elapsed_seconds": 6.23,
                        "limit_price": 38.56,
                        "quote_age_seconds": 1.68,
                        "quote_eligible": True,
                        "no_progress_reprices": 1,
                    },
                    "broker_order": {
                        "status": "Filled",
                        "filled": 23,
                        "quantity": 23,
                        "average_fill_price": 38.56,
                        "fills": [{"commission": 0.340881}],
                    },
                },
            }
        )
    ]

    assert any("Execution ladder  OPT→MID" in line for line in details)
    assert any("quote_age=1.68s" in line for line in details)
    assert any(
        "ref=XSPV3-test" in line and "commission=$0.3409" in line for line in details
    )


def _event(event_id: str, when: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "recorded_at_utc": when,
        "kind": "EXECUTION",
        "sleeve_id": "xsp-upro-spxu-rth-cash",
        "label": "XSP v3",
        "phase": "STATE",
        "status": "UNCHANGED",
        "reason": "flat_target",
        "message": "target=flat",
    }


def test_identical_timeline_refresh_touches_no_rows() -> None:
    harness = _Harness()
    harness._activity_table = _TableStub()
    harness._timeline_rows = []
    events = [_event("one", "2026-08-05T17:00:00+00:00")]

    assert harness._render_timeline_tables(events) is True
    activity_adds = harness._activity_table.add_calls
    assert harness._render_timeline_tables([dict(events[0])]) is False

    assert harness._activity_table.clear_calls == 0
    assert harness._activity_table.add_calls == activity_adds == 1
    assert harness._activity_table.remove_calls == 0


def test_timeline_rolls_forward_by_one_row_without_rebuilding() -> None:
    harness = _Harness()
    harness._activity_table = _TableStub()
    harness._timeline_rows = []
    first = _event("one", "2026-08-05T17:00:00+00:00")
    second = _event("two", "2026-08-05T17:05:00+00:00")
    third = _event("three", "2026-08-05T17:10:00+00:00")

    harness._render_timeline_tables([first, second])
    harness._render_timeline_tables([second, third])

    assert harness._activity_table.clear_calls == 0
    assert harness._activity_table.add_calls == 3
    assert harness._activity_table.remove_calls == 1
    assert [row[1]["key"] for row in harness._activity_table.rows] == [
        "activity:EXECUTION:two",
        "activity:EXECUTION:three",
    ]


def test_unchanged_remote_view_skips_every_renderer() -> None:
    class _Owner:
        def view(self, **kwargs):
            assert kwargs == {
                "limit": 250,
                "trace_limit": 2_000,
                "previous_view_id": "stable",
            }
            return {"view_id": "stable", "unchanged": True}

    harness = _Harness()
    harness._live_runs_owner = _Owner()
    harness._live_runs_view_id = "stable"
    harness._live_runs_refreshing = False
    harness._active_panel = "traces"
    harness._render_live_runs_table = lambda _snapshot: (_ for _ in ()).throw(
        AssertionError()
    )
    harness._render_candidates_table = lambda _snapshot: (_ for _ in ()).throw(
        AssertionError()
    )
    harness._render_timeline_tables = lambda _events: (_ for _ in ()).throw(
        AssertionError()
    )
    harness._render_strategy_traces = lambda _traces: (_ for _ in ()).throw(
        AssertionError()
    )

    asyncio.run(harness._refresh_live_runs())

    assert harness._live_runs_refreshing is False


def test_graduation_distinguishes_stale_hold_from_no_receipt() -> None:
    cutoff = {"cutoff_utc": "2026-08-03T13:44:04+00:00"}
    age = BotLiveRunsMixin._graduation_age_hours(
        cutoff,
        now=datetime(2026, 8, 6, 3, 44, 4, tzinfo=timezone.utc),
    )
    pending = BotLiveRunsMixin._live_run_graduation_cell(
        {"graduation": {"verdict": "PENDING", "receipt_id": None}}
    )

    assert age == 62
    assert pending.plain == "NO RECEIPT"
    assert (
        BotPortfolioMixin._candidate_graduation(
            {"graduation": {"verdict": "PENDING", "receipt_id": None}}
        )
        == "NO RECEIPT"
    )
    assert (
        BotLiveRunsMixin._graduation_ladder({"verdict": "HOLD", "target": "24h"})
        == "24h HOLD → 48h locked → 5 sessions locked"
    )


def _trace(*, sample_count: int = 1) -> dict[str, object]:
    return {
        "trace_id": "trace-one",
        "event_id": "event-one",
        "trace_key": "XSP",
        "sleeve_id": "xsp-upro-spxu-rth-cash",
        "label": "XSP v3",
        "first_recorded_at_utc": "2026-08-06T03:35:00+00:00",
        "last_recorded_at_utc": "2026-08-06T03:36:00+00:00",
        "sample_count": sample_count,
        "episode_start": True,
        "status": "HOLD",
        "reason": "weak_direction",
        "target_direction": None,
        "holdings": {"UPRO": 0, "SPXU": 0},
        "economics": {"net": -11.0, "drawdown": 11.0},
        "family": "IMPULSE",
        "decision": {"trend": "down", "coherence": 0.8},
        "volatility": {
            "atr_ratio": 0.661,
            "atr_velocity": -0.00302,
            "atr_acceleration": -0.00735,
        },
        "horizons": [
            {"bars": 1, "minutes": 5.0, "angle": 15.9, "slope_velocity": 0.0285},
            {"bars": 3, "minutes": 15.0, "angle": -36.4, "slope_velocity": 0.0069},
            {"bars": 6, "minutes": 30.0, "angle": -40.1, "slope_velocity": 0.0001},
            {"bars": 12, "minutes": 60.0, "angle": 0.0, "slope_velocity": -0.00302},
            {"bars": 24, "minutes": 120.0, "angle": 9.0, "slope_velocity": -0.0022},
        ],
        "long_context": {"directions": {"5": "up", "84": "up"}},
        "news": {
            "authority": "observation_only",
            "direction": 1,
            "signed_pressure": 0.494,
            "pressure_delta": -0.019,
            "pressure_velocity_per_hour": -0.00427,
            "confidence": 0.95,
            "impact": 52,
            "usable": True,
            "drivers": ["higher-driver", "lower-driver"],
            "driver_scores": [
                {
                    "id": "higher-driver",
                    "label": "Higher driver",
                    "direction": 1,
                    "impact": 76,
                },
                {
                    "id": "lower-driver",
                    "label": "Lower driver",
                    "direction": -1,
                    "impact": 42,
                },
            ],
            "change_windows": [
                {
                    "hours": 4,
                    "available": True,
                    "elapsed_hours": 4.2,
                    "pressure_delta": 0.04,
                    "pressure_velocity_per_hour": 0.00952,
                },
                {
                    "hours": 24,
                    "available": True,
                    "elapsed_hours": 25.5,
                    "pressure_delta": 0.47,
                    "pressure_velocity_per_hour": 0.01843,
                },
                {"hours": 168, "available": False},
            ],
        },
        "delta": {
            "volatility": {"atr_ratio": 0.01},
            "horizons": [{"bars": 1, "minutes": 5.0, "angle": 1.2}],
        },
        "provenance": {
            "source_authority": "finalized_source_only_no_orders_no_capital",
            "source_checkpoint_id": "7" * 64,
            "source_schema": "test.source-checkpoint.v1",
            "source_recorded_at_utc": "2026-08-06T03:34:59+00:00",
        },
    }


def test_strategy_trace_tabs_update_only_changed_rows_and_preserve_density() -> None:
    harness = _Harness()
    harness._trace_tables = {
        "ALL": _TableStub(),
        "XSP": _TableStub(),
        "1OZ": _TableStub(),
        "MCL": _TableStub(),
    }
    harness._trace_rows_by_key = {}
    harness._all_trace_rows = []
    first = _trace()

    assert harness._render_strategy_traces([first]) is True
    assert harness._render_strategy_traces([dict(first)]) is False
    assert harness._trace_tables["ALL"].add_calls == 1
    assert harness._trace_tables["XSP"].add_calls == 1
    assert harness._trace_tables["1OZ"].add_calls == 0
    rendered = harness._trace_tables["XSP"].rows[0][0][1].plain
    assert "5m" in rendered and "+15.9°" in rendered and "Δ+1.2" in rendered
    assert "ATR r=+0.661" in rendered
    assert "NEWS OBSERVATION" in rendered
    assert len(rendered.splitlines()) == 4

    mcl = {
        **first,
        "trace_id": "trace-mcl",
        "trace_key": "MCL",
        "family": "MCL_IMPULSE",
        "decision": {
            "trend": "up",
            "coherence": 0.75,
            "cl_move": 0.01,
            "mcl_move": 0.009,
            "basis_velocity_ticks": -0.1,
            "parity_aligned": True,
        },
    }
    mcl_rendered = BotTraceMixin._trace_text(mcl).plain
    assert "TAPE CL=+0.0100 MCL=+0.0090 basis-v=-0.10t parity=✓" in mcl_rendered

    changed = _trace(sample_count=2)
    assert harness._render_strategy_traces([changed]) is True
    assert harness._trace_tables["ALL"].update_calls == 2
    assert harness._trace_tables["XSP"].update_calls == 2
    assert harness._trace_tables["ALL"].add_calls == 1


def test_trace_when_shows_only_utc_minutes_without_wrapping() -> None:
    assert BotTraceMixin._trace_when(_trace()).plain == "03:35"
    episode = _trace(sample_count=18)
    assert BotTraceMixin._trace_when(episode).plain == "03:35→03:36 ×18"
    assert len(BotTraceMixin._trace_when(episode).plain) <= 17


def test_trace_grid_keeps_fields_aligned_across_widths_and_anchor_drift() -> None:
    first = _trace()
    changed = deepcopy(first)
    changed["status"] = "UNCHANGED"
    changed["reason"] = "flat_no_target"
    changed["horizons"][0]["angle"] = -110.8
    changed["horizons"][0]["slope_velocity"] = -0.1234
    changed["delta"]["horizons"][0]["angle"] = -99.9

    first_lines = BotTraceMixin._trace_text(first).plain.splitlines()
    changed_lines = BotTraceMixin._trace_text(changed).plain.splitlines()

    for field in ("target=", "pos=", "net=", "trend="):
        assert first_lines[0].index(field) == changed_lines[0].index(field)
    for field in ("5m", "15m", "30m", "60m", "120m"):
        assert first_lines[1].index(field) == changed_lines[1].index(field)
        assert first_lines[2].index(field) == changed_lines[2].index(field)
        assert first_lines[1].index(field) == first_lines[2].index(field)
    for field in ("ATR r=", "v=", "a="):
        assert first_lines[2].index(field) == changed_lines[2].index(field)
    angle_dividers = [
        index for index, character in enumerate(first_lines[1]) if character == "│"
    ]
    velocity_dividers = [
        index for index, character in enumerate(first_lines[2]) if character == "│"
    ]
    assert angle_dividers == velocity_dividers[: len(angle_dividers)]
    assert len(first_lines[2]) <= 184

    mcl = deepcopy(first)
    mcl["family"] = "MCL_IMPULSE"
    mcl["horizons"] = [
        {"bars": bars, "minutes": minutes, "angle": 1.0, "slope_velocity": 0.001}
        for bars, minutes in ((6, 30), (12, 60), (24, 120), (48, 240), (96, 480))
    ]
    drifted = deepcopy(mcl)
    drifted["horizons"][-1]["minutes"] = 540
    normal_lines = BotTraceMixin._trace_text(mcl).plain.splitlines()
    drifted_lines = BotTraceMixin._trace_text(drifted).plain.splitlines()

    assert normal_lines[1].index("480m") == drifted_lines[1].index("540m")
    assert normal_lines[2].index("480m") == drifted_lines[2].index("540m")


def test_trace_inspector_ranks_real_driver_impacts_and_compacts_provenance() -> None:
    identity, news, momentum, evidence = BotTraceMixin._trace_context_lines(_trace())

    assert "08-06 03:35:00 → 03:36:00 UTC" in identity.plain
    assert "2026-08-06T" not in identity.plain
    assert "aggregate ↑52" in news.plain
    assert news.plain.index("↑76 higher driver") < news.plain.index("↓42 lower driver")
    assert "4h ΔP=+0.040 v/h=+0.00952" in momentum.plain
    assert "1d ΔP=+0.470 v/h=+0.01843" in momentum.plain
    assert "1w ΔP=n/a v/h=n/a" in momentum.plain
    assert "shape=↑↑→" in momentum.plain
    assert "AUTH finalized source only no orders no capital" in evidence.plain
    assert "CHECKPOINT 777777777777…" in evidence.plain
    assert "7" * 64 not in evidence.plain
    assert BotTraceMixin._trace_legend().plain == (
        "θ=slope÷TR°  ω=Δ(slope%/bar)  r=ATRfast÷slow  "
        "v=ΔATRfast  a=Δv  Δ=vs prior displayed observation"
    )


def test_mounted_trace_lines_are_physically_visible_in_the_cockpit() -> None:
    class _Owner:
        def __init__(self) -> None:
            self.calls = 0

        def view(self, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                return {"view_id": "stable", "unchanged": True}
            return {
                "view_id": "stable",
                "unchanged": False,
                "snapshot": {
                    "snapshot_id": "snapshot",
                    "runs": [_run()],
                    "candidates": [],
                },
                "timeline": [],
                "traces": [_trace()],
            }

        async def aclose(self) -> None:
            return

    class _App(App):
        CSS = PositionsApp.CSS

        async def on_mount(self) -> None:
            await self.push_screen(BotScreen(refresh_sec=999))

    async def exercise() -> None:
        with patch(
            "tradebot.ui.bot_screen.live_runs.LivePortfolioEndpoint.default",
            return_value=_Owner(),
        ):
            app = _App()
            async with app.run_test(size=(220, 80)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert screen._active_panel == "traces"
                assert screen.focused is screen._trace_tables["ALL"]
                table = screen._trace_tables["ALL"]
                status = screen.query_one("#bot-status")
                footer = screen.query_one(TradebotFooter)
                screenshot = app.export_screenshot()

                assert table.row_count == 1
                assert table.has_class("bot-trace-table")
                cockpit_tables = (
                    screen._candidates_table,
                    screen._live_runs_table,
                    screen._activity_table,
                    *screen._trace_tables.values(),
                )
                for cockpit_table in cockpit_tables:
                    assert cockpit_table.cursor_foreground_priority == "renderable"
                    assert cockpit_table.cursor_background_priority == "css"
                    cursor_background = cockpit_table.get_component_styles(
                        "datatable--cursor"
                    ).background
                    assert cursor_background.a == 0.12
                assert table.region.height >= 4
                assert status.region.height == 4
                assert "ATR" in screenshot
                assert "NEWS" in screenshot
                assert "CURVE" in screenshot
                assert "↑76" in screenshot
                assert "θ" in str(screen._trace_tabs.border_subtitle)
                assert "prior displayed observation" in str(
                    screen._trace_tabs.border_subtitle
                )
                assert [key.description for key in footer.query("FooterKey")] == [
                    "Contract",
                    "Panel",
                    "Back",
                    "Candidates",
                    "Act",
                    "Safe stop",
                    "Refresh",
                ]

                horizontal_offset = table.scroll_offset.x
                await pilot.press("right")
                assert screen._active_trace_key() == "XSP"
                assert screen._active_panel == "traces"
                assert table.scroll_offset.x == horizontal_offset
                await pilot.press("right")
                assert screen._active_trace_key() == "1OZ"
                await pilot.press("left")
                assert screen._active_trace_key() == "XSP"

                trace_height = screen.query_one("#bot-traces").region.height
                screen.action_toggle_candidates()
                await pilot.pause()
                assert screen._candidates_table.display is False
                assert screen.query_one("#bot-traces").region.height > trace_height

                screen._render_candidates_table(
                    {
                        "candidates": [
                            {
                                "candidate_id": "background-refresh",
                                "label": "Hidden candidate",
                                "symbol": "XSP",
                                "track": "LF",
                            }
                        ]
                    }
                )
                await pilot.pause()
                assert screen._active_panel == "traces"
                assert "Immutable champion candidate" not in app.export_screenshot()

                screen.action_cycle_focus()
                assert screen._active_panel == "live_runs"
                screen.action_toggle_candidates()
                await pilot.pause()
                assert screen._candidates_table.display is True
                assert screen._active_panel == "live_runs"

    asyncio.run(exercise())
