from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.ui.bot import BotScreen
from tradebot.ui.bot_screen.live_runs import BotLiveRunsMixin
from tradebot.ui.bot_screen.portfolio import BotPortfolioMixin


class _TableStub:
    def __init__(self) -> None:
        self.cursor_coordinate = SimpleNamespace(row=0)
        self.rows: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def clear(self, columns: bool = False) -> None:
        self.rows = []

    def add_column(self, *_args, **_kwargs) -> None:
        return

    def add_row(self, *values, **kwargs) -> None:
        self.rows.append((values, kwargs))


class _Harness(BotPortfolioMixin, BotLiveRunsMixin):
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
            "run_cost_usd": 0.0,
            "drawdown_usd": 0.0,
            "fill_count": 0,
            "closed_trades": 0,
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
        "logs",
    )
    screen = BotScreen(client=SimpleNamespace(), refresh_sec=1.0)
    assert not hasattr(screen, "_client")
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


def test_live_run_details_reuse_persisted_hawkeye_anatomy() -> None:
    details = [line.plain for line in _Harness()._live_run_detail_lines(_run())]

    assert any("Hawkeye: HOLD / weak_direction" in line for line in details)
    assert any("Gate: source=directional_impulse" in line for line in details)
    assert any("Slope angles" in line and "5m:-42.8°" in line for line in details)
    assert any("Slope velocity" in line and "120m:-0.0105" in line for line in details)
    assert any("ATR ratio=+2.065" in line for line in details)
    assert any("Long context" in line and "transition_down" in line for line in details)
    assert any("News attribution" in line and "pressure=+0.5044" in line for line in details)


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
    assert any("ref=XSPV3-test" in line and "commission=$0.3409" in line for line in details)
