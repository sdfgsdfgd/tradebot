from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

from tradebot.backtest.config import load_config
from tradebot.backtest.engine_options import (
    _OPTIONS_MARKET_TAPE_NAMESPACE,
    _OPTIONS_SIGNAL_TAPE_NAMESPACE,
    _OPTIONS_TAPE_CACHE,
    _OPTIONS_TAPE_NAMESPACE,
    prepare_options_tape,
)
from tradebot.backtest.models import Bar


def test_prepared_options_tape_reuses_exact_facts_and_invalidates_content(
    tmp_path,
) -> None:
    config_path = tmp_path / "options.json"
    config_path.write_text(
        json.dumps(
            {
                "backtest": {
                    "start": "2026-02-02",
                    "end": "2026-02-03",
                    "bar_size": "1 hour",
                    "use_rth": False,
                    "offline": True,
                },
                "strategy": {
                    "instrument": "options",
                    "symbol": "XSP",
                    "ema_preset": "2/3",
                },
            }
        )
    )
    cfg = load_config(config_path)
    bars = [
        Bar(datetime(2026, 2, 2, 14, 30), 100, 101, 99, 100, 10),
        Bar(datetime(2026, 2, 2, 15, 30), 100, 102, 99, 101, 20),
        Bar(datetime(2026, 2, 3, 14, 30), 101, 103, 100, 102, 30),
    ]
    for namespace in (
        _OPTIONS_MARKET_TAPE_NAMESPACE,
        _OPTIONS_SIGNAL_TAPE_NAMESPACE,
        _OPTIONS_TAPE_NAMESPACE,
    ):
        _OPTIONS_TAPE_CACHE.clear(namespace=namespace)

    kwargs = {
        "data": object(),
        "start_dt": datetime(2026, 2, 2),
        "end_dt": datetime(2026, 2, 3, 23, 59),
    }
    first = prepare_options_tape(cfg=cfg, bars=bars, **kwargs)
    assert prepare_options_tape(cfg=cfg, bars=list(bars), **kwargs) is first
    assert first.bars_in_day == (1, 2, 1)
    assert first.is_last_bar == (False, True, True)

    other_policy = prepare_options_tape(
        cfg=replace(cfg, strategy=replace(cfg.strategy, ema_preset="3/4")),
        bars=bars,
        **kwargs,
    )
    assert other_policy is not first
    assert other_policy.trade_dates is first.trade_dates
    assert other_policy.realized_vol is first.realized_vol
    assert other_policy.signals is not first.signals

    changed_bars = list(bars)
    changed_bars[1] = replace(changed_bars[1], close=101.5)
    changed = prepare_options_tape(cfg=cfg, bars=changed_bars, **kwargs)
    assert changed.revision != first.revision
    assert changed.realized_vol != first.realized_vol
