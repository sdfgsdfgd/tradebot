from __future__ import annotations

import json
from pathlib import Path

from tradebot.backtest.spot_context import spot_bar_requirements_from_strategy


def test_tqqq_v38_context_requirements_include_regime2_warmup() -> None:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v38_asymmetricCrashPrearmSovereignty_20260316.json"
        ).read_text()
    )
    strategy = payload["groups"][0]["entries"][0]["strategy"]

    reqs = {
        req.kind: req
        for req in spot_bar_requirements_from_strategy(
            strategy=strategy,
            default_symbol="TQQQ",
            default_exchange=None,
            default_signal_bar_size=str(strategy["signal_bar_size"]),
            default_signal_use_rth=bool(strategy["signal_use_rth"]),
            include_signal=True,
        )
    }

    assert reqs["signal"].warmup_days >= 7
    assert reqs["regime2"].warmup_days >= 60
    assert reqs["regime2_bear_hard"].warmup_days >= 60
    assert reqs["exec"].warmup_days >= reqs["signal"].warmup_days
