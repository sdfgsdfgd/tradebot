from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.backtest.spot_codec import effective_filters_payload
from tradebot.ui.bot import BotScreen


def _screen() -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return BotScreen(client=SimpleNamespace(), refresh_sec=1.0)


def test_effective_filters_payload_merges_and_hoists() -> None:
    group_filters = {"shock_gate_mode": "detect", "shock_detector": "atr_ratio"}
    strategy = {
        "ema_preset": "5/13",
        "filters": {"shock_detector": "daily_drawdown"},
        # Misplaced filter keys at strategy root (seen in milestone payloads).
        "ratsv_enabled": True,
        "ratsv_tr_fast_bars": 5,
    }
    effective = effective_filters_payload(group_filters=group_filters, strategy=strategy)
    assert isinstance(effective, dict)
    assert effective.get("shock_gate_mode") == "detect"
    assert effective.get("shock_detector") == "daily_drawdown"
    assert effective.get("ratsv_enabled") is True
    assert int(effective.get("ratsv_tr_fast_bars") or 0) == 5


def test_ui_heal_strategy_filters_payload_removes_embedded_filters() -> None:
    screen = _screen()
    strategy = {
        "ema_preset": "5/13",
        "filters": {"shock_detector": "daily_drawdown"},
        "ratsv_enabled": True,
    }
    base_filters = {"shock_gate_mode": "detect"}
    effective = screen._heal_strategy_filters_payload(strategy=strategy, base_filters=base_filters)
    assert isinstance(effective, dict)
    assert effective.get("shock_gate_mode") == "detect"
    assert effective.get("shock_detector") == "daily_drawdown"
    assert effective.get("ratsv_enabled") is True
    assert "filters" not in strategy
    assert "ratsv_enabled" not in strategy
    assert strategy.get("ema_preset") == "5/13"

