from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.engines.signals import EmaDecisionEngine
from tradebot.chart_data.history import normalize_bars_to_close
from tradebot.chart_data.series import OhlcvBar
from tradebot.research.gold_onset import (
    GOLD_ONSET_PROSPECTIVE_START,
    advance_gold_onset_tape,
    build_gold_onset_context,
    gold_signal_context,
    select_gold_contract_pair,
)
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.spot.champions import discover_current_champions, load_champion_group


UTC = timezone.utc


def test_gold_crown_is_machine_bound_but_cannot_trade() -> None:
    root = Path(__file__).resolve().parents[1]
    refs = discover_current_champions(root=root, symbols=("1OZ",), tracks=("LF",))

    assert len(refs) == 1
    ref = refs[0]
    declaration = json.loads(ref.declaration_path.read_text())
    artifact = json.loads(ref.artifact_path.read_text())
    group = load_champion_group(ref)

    assert ref.version == "1"
    assert group is not None
    assert group["_key"] == "one-oz-regime-harmony-stage76-77"
    assert hashlib.sha256(ref.artifact_path.read_bytes()).hexdigest() == declaration[
        "artifact_sha256"
    ]
    assert declaration["promotion"]["eligible"] is True
    assert declaration["promotion"]["order_authority"] == "none"
    assert artifact["order_authority"] == "none"
    assert artifact["selection_authority"] == "none"
    assert artifact["capital_authority"] == "none"
    assert artifact["graduation_enrollment"]["lifecycle_state"] == "CROWNED"
    assert artifact["graduation_enrollment"]["live_24h"] == "NOT_STARTED"
    assert artifact["groups"][0]["entries"] == []


def test_xauusd_daily_evidence_closes_at_17_et() -> None:
    normalized = normalize_bars_to_close(
        [OhlcvBar(datetime(2026, 8, 3), 100.0, 101.0, 99.0, 100.0, 0.0)],
        symbol="XAUUSD",
        bar_size="1 day",
        use_rth=False,
    )

    assert normalized[0].ts == datetime(2026, 8, 3, 21, 0)


def _bar(end: datetime, close: float) -> dict[str, object]:
    return {
        "end": end,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
    }


def _daily_up(end: datetime) -> list[dict[str, object]]:
    start = end - timedelta(days=89)
    return [_bar(start + timedelta(days=index), 100.0 + index) for index in range(90)]


def _confirmed_h4_up(end: datetime) -> list[dict[str, object]]:
    values = [150.0 - index for index in range(30)] + [120.0 + index * 2.0 for index in range(30)]
    engine = EmaDecisionEngine(ema_preset="8/21", ema_entry_mode="cross", entry_confirm_bars=1)
    rows = []
    start = end - timedelta(hours=4 * len(values))
    for index, value in enumerate(values):
        row = _bar(start + timedelta(hours=4 * (index + 1)), value)
        rows.append(row)
        if engine.update(value).entry_dir == "up":
            return rows
    raise AssertionError("synthetic H4 path did not produce a confirmed up cross")


def test_gold_pair_requires_live_liquid_shared_month() -> None:
    now = datetime(2026, 8, 3, 2, 5, 27, tzinfo=UTC)

    def quote(symbol: str, local: str, expiry: str, bid: float, ask: float, volume: int):
        return {
            "symbol": symbol,
            "local_symbol": local,
            "expiry": expiry,
            "market_data_type": 1,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "time": now,
        }

    selected = select_gold_contract_pair(
        [
            quote("GC", "GCQ6", "20260827", 4057.2, 4058.1, 22),
            quote("GC", "GCZ6", "20261229", 4114.6, 4115.0, 13_566),
            quote("1OZ", "1OZZ6", "20261125", 4114.75, 4115.0, 22_157),
            quote("1OZ", "1OZG7", "20270127", 4147.25, 4148.0, 10),
        ],
        observed_at=now,
    )

    assert selected["usable"] is True
    assert selected["contract_month"] == "2026-12"
    assert selected["gc"]["local_symbol"] == "GCZ6"
    assert selected["one_oz"]["local_symbol"] == "1OZZ6"
    assert selected["basis_usd"] == pytest.approx(0.075)
    assert any(row["local_symbol"] == "GCQ6" for row in selected["rejected"])


def test_gold_signal_reconstructs_stage12_hard_gate() -> None:
    end = datetime(2026, 8, 3, tzinfo=UTC)
    h4 = _confirmed_h4_up(end)
    daily = _daily_up(end)

    context = gold_signal_context(h4, daily, (), (), as_of=end)

    assert context["usable"] is True
    assert context["proposed_direction"] == "up"
    assert context["stage_12"]["admitted_direction"] == "up"
    assert context["stage_12"]["blocked_by"] is None
    assert context["stage_22"]["admitted_direction"] == "up"
    assert context["h4"]["authority"] == "attribution_only"
    assert context["h4"]["hard_direction"] == "up"
    assert context["h4"]["signed_fast_slope_dollars"] > 0.0
    assert context["h4"]["fast_bars_to_finance"] > 0.0
    assert context["h4"]["financing_clock"] in ("within_12h", "within_24h")
    assert {"12", "30"} <= set(context["h4"]["path"])


def test_gold_onset_context_is_canonical_json() -> None:
    now = datetime(2026, 8, 3, 2, 20, tzinfo=UTC)
    h4 = _confirmed_h4_up(now)
    daily = _daily_up(now)
    source_points = {
        symbol: {
            "close": 100.0,
            "bar_end_utc": now.isoformat(),
            "age_seconds": 0.0,
        }
        for symbol in ("XAUUSD", "GC", "1OZ")
    }
    quotes = [
        {
            "symbol": symbol,
            "local_symbol": local,
            "expiry": expiry,
            "con_id": con_id,
            "market_data_type": 1,
            "bid": 100.0,
            "ask": 100.25,
            "volume": 1000,
            "time": now,
        }
        for symbol, local, expiry, con_id in (
            ("GC", "GCZ6", "20261229", 1),
            ("1OZ", "1OZZ6", "20261125", 2),
        )
    ]

    context = build_gold_onset_context(
        xau_h4=h4,
        xau_daily=daily,
        uup_daily=(),
        tip_daily=(),
        quotes=quotes,
        news_history=(),
        source_points=source_points,
        observed_at=now,
    )

    json.dumps(context, allow_nan=False, sort_keys=True)
    assert context["schema"] == "gold.1oz-prospective-onset-context.v2"
    assert context["signal"]["daily"]["end"].endswith("+00:00")
    assert context["signal"]["h4"]["authority"] == "attribution_only"
    assert "time" not in context["exchange_parity"]["gc"]


def test_gold_stage22_waits_then_matures_only_while_up(monkeypatch) -> None:
    end = datetime(2026, 8, 1, tzinfo=UTC)
    h4 = _confirmed_h4_up(end)
    trigger_at = h4[-1]["end"]
    assert isinstance(trigger_at, datetime)
    daily = _daily_up(trigger_at + timedelta(days=3))
    macro = []
    for index in range(3):
        stamp = trigger_at + timedelta(days=index)
        macro.append(
            {
                "end": stamp,
                "horizons": {
                    "5": {
                        "direction": "mixed" if index == 0 else "supportive",
                        "velocity": "mixed" if index == 0 else "supportive",
                        "acceleration": "mixed",
                        "state_age": 1,
                        "symbols": {},
                    },
                    "21": {"direction": "mixed", "velocity": "mixed", "acceleration": "mixed", "state_age": 1, "symbols": {}},
                    "63": {"direction": "mixed", "velocity": "mixed", "acceleration": "mixed", "state_age": 1, "symbols": {}},
                },
            }
        )
    monkeypatch.setattr("tradebot.research.gold_onset._macro_timeline", lambda *_args, **_kwargs: macro)

    started = gold_signal_context(h4, daily, (), (), as_of=trigger_at)
    assert started["stage_12"]["admitted_direction"] == "up"
    assert started["stage_22"]["admitted_direction"] is None
    assert started["stage_22"]["source"] == "macro_started"

    h4.extend(
        [
            _bar(trigger_at + timedelta(days=1), float(h4[-1]["close"]) + 1.0),
            _bar(trigger_at + timedelta(days=2), float(h4[-1]["close"]) + 2.0),
        ]
    )
    matured = gold_signal_context(h4, daily, (), (), as_of=trigger_at + timedelta(days=2))
    assert matured["stage_22"]["admitted_direction"] == "up"
    assert matured["stage_22"]["source"] == "macro_matured"


def _context(decision_at: datetime) -> dict[str, object]:
    return {
        "schema": "gold.1oz-prospective-onset-context.v2",
        "authority": "prospective_research_only",
        "observed_at_utc": (decision_at + timedelta(minutes=5)).isoformat(),
        "signal": {
            "usable": True,
            "decision_bar_end_utc": decision_at.isoformat(),
            "decision_close": 100.0,
            "stage_12": {"admitted_direction": "up"},
            "stage_22": {"admitted_direction": None},
        },
        "macro": {"usable": True, "total_direction_neutral": False},
        "news": {"usable": False, "authority": "attribution_only"},
        "exchange_parity": {"usable": True, "authority": "market_data_only"},
        "source_points": {
            symbol: {
                "close": 100.0,
                "bar_end_utc": decision_at.isoformat(),
                "age_seconds": 0.0,
            }
            for symbol in ("XAUUSD", "GC", "1OZ")
        },
        "source_closes": {"XAUUSD": 100.0, "GC": 100.0, "1OZ": 100.0},
        "timing_parity": {"usable": True},
        "counterfactual_directions": {"stage_12": "up", "stage_22": None},
        "total_cross_asset_neutral_short": False,
        "slow_financing_neutral_short": False,
        "order_authority": "none",
        "submitted_orders": 0,
    }


def _outcome_bars(decision_at: datetime) -> dict[str, list[dict[str, object]]]:
    rows = [
        _bar(decision_at + timedelta(hours=index), 100.0 + index * 0.25)
        for index in range(1, 25)
    ]
    return {symbol: list(rows) for symbol in ("XAUUSD", "GC", "1OZ")}


def test_gold_tape_freezes_once_then_settles_without_orders(tmp_path) -> None:
    decision_at = GOLD_ONSET_PROSPECTIVE_START + timedelta(hours=2)
    ledger = LiveCalibrationLedger(tmp_path / "gold.jsonl")
    context = _context(decision_at)

    first = advance_gold_onset_tape(
        ledger,
        context=context,
        outcome_bars={},
        observed_at=decision_at + timedelta(minutes=5),
    )
    duplicate = advance_gold_onset_tape(
        ledger,
        context={**context, "observed_at_utc": (decision_at + timedelta(minutes=10)).isoformat()},
        outcome_bars={},
        observed_at=decision_at + timedelta(minutes=10),
    )
    settled = advance_gold_onset_tape(
        ledger,
        context=context,
        outcome_bars=_outcome_bars(decision_at),
        observed_at=decision_at + timedelta(hours=24, minutes=5),
    )

    assert first["frozen"] == 1
    assert duplicate["frozen"] == 0
    assert settled["settled"] == 1
    records = list(ledger.records())
    forecasts = [row for row in records if row["kind"] == "forecast"]
    results = [row for row in records if row["kind"] == "result"]
    assert len(forecasts) == 1
    assert len(results) == 1
    assert forecasts[0]["forecast"]["decision"] == "NO_TRADE"
    assert forecasts[0]["gates"]["order_authority"] == "none"
    one_oz = results[0]["observed"]["counterfactuals"]["stage_12"]["1OZ"]["24"]
    assert one_oz["return_usd"] == pytest.approx(6.0)
    assert one_oz["financed"] is True
    assert all(row.get("submitted_orders", 0) == 0 for row in (first, duplicate, settled))


def test_gold_tape_rejects_late_backfill(tmp_path) -> None:
    decision_at = GOLD_ONSET_PROSPECTIVE_START + timedelta(hours=2)
    with pytest.raises(ValueError, match="forbids late forecast backfill"):
        advance_gold_onset_tape(
            LiveCalibrationLedger(tmp_path / "late.jsonl"),
            context=_context(decision_at),
            outcome_bars=_outcome_bars(decision_at),
            observed_at=decision_at + timedelta(hours=24),
        )


def test_gold_onset_service_is_bounded_and_cannot_submit_orders() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = root / "tradebot/research/gold_onset_cli.py"
    tree = ast.parse(adapter.read_text(), filename=str(adapter))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    service = (root / "deploy/systemd/tradebot-gold-onset.service").read_text()
    timer = (root / "deploy/systemd/tradebot-gold-onset.timer").read_text()

    assert not calls & {"placeOrder", "place_limit_order", "submit_order"}
    assert "Environment=IBKR_READONLY=1" in service
    assert "Environment=GOLD_ONSET_LEDGER=" in service
    assert "-m tradebot.research.gold_onset_cli" in service
    assert "NoNewPrivileges=true" in service
    assert "Mon..Fri *-*-* 00,04,08,12,16,20:07:00 UTC" in timer
    assert timer.count("OnCalendar=") == 1
    assert "Persistent=false" in timer
