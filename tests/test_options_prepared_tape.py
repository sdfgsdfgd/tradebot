from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, time, timezone

import tradebot.backtest.engine as backtest_engine
import tradebot.backtest.run_backtest_options as options_sweep
from tradebot.chart_data.history import cache_path, write_cache
from tradebot.backtest.config import load_config
from tradebot.backtest.data import ContractMeta
from tradebot.backtest.engine import OptionsBacktestSourcePool
from tradebot.backtest.engine_options import (
    _OPTIONS_MARKET_TAPE_NAMESPACE,
    _OPTIONS_SIGNAL_TAPE_NAMESPACE,
    _OPTIONS_TAPE_CACHE,
    _OPTIONS_TAPE_NAMESPACE,
    prepare_options_tape,
    run_options_backtest,
)
from tradebot.backtest.models import Bar
from tradebot.engines.signals import OrbDecisionEngine
from tradebot.spot_engine import SpotSignalEvaluator


def _options_config(tmp_path):
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
                    "cache_dir": str(tmp_path / "cache"),
                    "calibration_dir": str(tmp_path / "calibration"),
                    "output_dir": str(tmp_path / "out"),
                },
                "strategy": {
                    "instrument": "options",
                    "symbol": "XSP",
                    "ema_preset": "2/3",
                },
            }
        )
    )
    return load_config(config_path)


def _grid(*, profit_targets=(0.5,)) -> dict:
    return {
        "dte": [0],
        "moneyness_pct": [1.0],
        "profit_target": list(profit_targets),
        "stop_loss": [0.35],
        "ema_preset": ["2/3"],
        "ema_entry_mode": ["trend"],
        "exit_on_signal_flip": [False],
        "flip_exit_min_hold_bars": [0],
        "flip_exit_only_if_profit": [False],
        "min_trades": 0,
    }


def _group() -> dict:
    return options_sweep._group_spec(
        "Put Credit Spread",
        "PUT",
        [
            {"action": "SELL", "right": "PUT", "qty": 1},
            {
                "action": "BUY",
                "right": "PUT",
                "qty": 1,
                "moneyness_offset_pct": 1.0,
            },
        ],
        None,
        ema_entry_mode="trend",
    )


class _Progress:
    advanced = 0

    def advance(self) -> None:
        self.advanced += 1


def _opening_reclaim_bars() -> list[Bar]:
    start = datetime(2026, 7, 24, 13, 35, tzinfo=timezone.utc)
    values = (
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 100.5, 98.5, 99.0),
        (99.0, 100.0, 98.8, 99.2),
        (98.0, 98.2, 97.6, 97.8),
        (98.4, 98.8, 98.3, 98.6),
        (98.6, 98.9, 98.5, 98.7),
        (98.7, 99.2, 98.6, 99.0),
    )
    return [
        Bar(
            ts=start + timedelta(minutes=5 * index),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=0.0,
        )
        for index, (open_, high, low, close) in enumerate(values)
    ]


def test_opening_reclaim_requires_causal_breakdown_and_confirmed_reclaim() -> None:
    engine = OrbDecisionEngine(
        window_mins=15,
        mode="reclaim",
        break_range_fraction=0.25,
        reclaim_confirm_bars=2,
        deadline_et=time(11, 30),
    )

    directions = [
        engine.update(
            ts=bar.ts,
            high=bar.high,
            low=bar.low,
            close=bar.close,
        ).entry_dir
        for bar in _opening_reclaim_bars()
    ]

    assert directions == [None, None, None, None, None, "up", None]


def test_orb_default_remains_the_legacy_breakout_mode() -> None:
    bars = _opening_reclaim_bars()
    default = OrbDecisionEngine(window_mins=15)
    explicit = OrbDecisionEngine(window_mins=15, mode="breakout")

    def directions(engine):
        return [
            engine.update(ts=bar.ts, high=bar.high, low=bar.low, close=bar.close).entry_dir
            for bar in bars
        ]

    assert directions(default) == directions(explicit) == [
        None, None, None, "down", None, None, None
    ]


def test_opening_reclaim_signal_is_identical_in_live_evaluator_and_options_tape(
    tmp_path,
) -> None:
    cfg = _options_config(tmp_path)
    cfg = replace(
        cfg,
        strategy=replace(
            cfg.strategy,
            entry_signal="opening_reclaim",
            ema_preset=None,
            orb_window_mins=15,
            opening_reclaim_break_range_fraction=0.25,
            opening_reclaim_confirm_bars=2,
            opening_reclaim_deadline_et="11:30",
        ),
    )
    bars = _opening_reclaim_bars()
    prepared = prepare_options_tape(
        cfg=cfg,
        bars=bars,
        data=object(),
        start_dt=bars[0].ts,
        end_dt=bars[-1].ts,
    )
    evaluator = SpotSignalEvaluator(
        strategy=cfg.strategy,
        filters=cfg.strategy.filters,
        bar_size=cfg.backtest.bar_size,
        use_rth=True,
    )

    live_directions = [
        snapshot.entry_dir if snapshot is not None else None
        for snapshot in (evaluator.update_signal_bar(bar) for bar in bars)
    ]
    replay_directions = [
        signal.entry_dir if signal is not None else None
        for signal in prepared.signals
    ]

    assert live_directions == replay_directions == [
        None,
        None,
        None,
        None,
        None,
        "up",
        None,
    ]


def test_options_warmup_bars_build_signals_but_never_trade_or_score(
    tmp_path,
) -> None:
    cfg = _options_config(tmp_path)
    start = datetime(2026, 2, 2)
    bars = [
        Bar(
            ts=datetime(2026, 1, 30, 14) + timedelta(hours=index),
            open=100.0 + index,
            high=100.5 + index,
            low=99.5 + index,
            close=100.25 + index,
            volume=1000.0,
        )
        for index in range(5)
    ] + [
        Bar(
            ts=start + timedelta(hours=14 + index),
            open=105.0 + index,
            high=105.5 + index,
            low=104.5 + index,
            close=105.25 + index,
            volume=1000.0,
        )
        for index in range(5)
    ]

    result = run_options_backtest(
        cfg=cfg,
        bars=bars,
        meta=ContractMeta(
            symbol="XSP",
            exchange="CBOE",
            multiplier=100.0,
            min_tick=0.01,
        ),
        data=object(),
        start_dt=start,
        end_dt=datetime(2026, 2, 2, 23, 59),
    )

    assert result.equity
    assert all(point.ts >= start for point in result.equity)
    assert all(trade.entry_time >= start for trade in result.trades)


def test_prepared_options_tape_reuses_exact_facts_and_invalidates_content(
    tmp_path,
) -> None:
    cfg = _options_config(tmp_path)
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


def test_options_source_pool_reuses_only_exact_warmup_windows(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _options_config(tmp_path)
    created = []

    class _Source:
        closed = False

        def close(self) -> None:
            self.closed = True

    def _prepare(_cfg):
        source = _Source()
        created.append(source)
        return source

    monkeypatch.setattr(backtest_engine, "prepare_options_backtest_source", _prepare)
    pool = OptionsBacktestSourcePool()
    first = pool.source_for(cfg)

    assert pool.source_for(cfg) is first
    assert pool.source_for(
        replace(cfg, strategy=replace(cfg.strategy, ema_preset="20/50"))
    ) is not first
    assert len(created) == 2

    pool.close()
    assert all(source.closed for source in created)


def test_option_sweep_warm_cache_skips_work_and_small_delta_runs_only_miss(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = _options_config(tmp_path)
    calls = []

    def _run_combo(**kwargs):
        calls.append(kwargs["profit_target"])
        return {
            "metrics": {
                "pnl": kwargs["profit_target"],
                "win_rate": 1.0,
            },
            "strategy": {"profit_target": kwargs["profit_target"]},
        }

    monkeypatch.setattr(options_sweep, "_run_combo", _run_combo)
    common = {
        "symbol": "XSP",
        "backtest": cfg.backtest,
        "synthetic": cfg.synthetic,
        "group": _group(),
        "jobs": 1,
        "result_db": tmp_path / "results.sqlite3",
        "revision": "unit-revision",
    }

    first_progress = _Progress()
    first = options_sweep._run_group(
        grid=_grid(),
        progress=first_progress,
        **common,
    )
    warm_progress = _Progress()
    warm = options_sweep._run_group(
        grid=_grid(),
        progress=warm_progress,
        **common,
    )
    delta_progress = _Progress()
    delta = options_sweep._run_group(
        grid=_grid(profit_targets=(0.5, 0.6)),
        progress=delta_progress,
        **common,
    )
    threshold_progress = _Progress()
    threshold = options_sweep._run_group(
        grid={**_grid(), "min_trades": 2},
        progress=threshold_progress,
        **common,
    )

    assert first == warm
    assert len(delta) == 2
    assert threshold == []
    assert calls == [0.5, 0.6]
    assert (
        first_progress.advanced,
        warm_progress.advanced,
        delta_progress.advanced,
        threshold_progress.advanced,
    ) == (1, 1, 2, 1)


def test_option_sweep_revision_tracks_calibration_content(tmp_path) -> None:
    cfg = _options_config(tmp_path)
    missing = options_sweep._sweep_revision(
        backtest=cfg.backtest,
        symbol="XSP",
    )
    unrelated = cache_path(
        cfg.backtest.cache_dir,
        "OTHER",
        datetime(2026, 2, 2),
        datetime(2026, 2, 3, 23, 59),
        "1 hour",
        use_rth=False,
    )
    unrelated.parent.mkdir(parents=True)
    write_cache(
        unrelated,
        [Bar(datetime(2026, 2, 2), 1, 2, 0.5, 1.5, 100)],
    )
    assert (
        options_sweep._sweep_revision(
            backtest=cfg.backtest,
            symbol="XSP",
        )
        == missing
    )
    relevant = cache_path(
        cfg.backtest.cache_dir,
        "XSP",
        datetime(2026, 2, 2),
        datetime(2026, 2, 3, 23, 59),
        "1 hour",
        use_rth=False,
    )
    relevant.parent.mkdir(parents=True)
    write_cache(
        relevant,
        [Bar(datetime(2026, 2, 2), 1, 2, 0.5, 1.5, 100)],
    )
    data_revision = options_sweep._sweep_revision(
        backtest=cfg.backtest,
        symbol="XSP",
    )
    assert data_revision != missing

    calibration = cfg.backtest.calibration_dir / "XSP.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"buckets":[]}')
    first = options_sweep._sweep_revision(
        backtest=cfg.backtest,
        symbol="XSP",
    )
    calibration.write_text('{"buckets":[{"records":[]}]}')
    second = options_sweep._sweep_revision(
        backtest=cfg.backtest,
        symbol="XSP",
    )

    assert len({missing, data_revision, first, second}) == 4


def test_option_sweep_spawned_worker_executes_real_combo(tmp_path) -> None:
    cfg = _options_config(tmp_path)
    start = datetime(2026, 2, 2)
    end = datetime(2026, 2, 3, 23, 59)
    bars = [
        Bar(
            ts=start + timedelta(hours=index),
            open=100.0 + index * 0.1,
            high=100.5 + index * 0.1,
            low=99.5 + index * 0.1,
            close=100.2 + index * 0.1,
            volume=1000.0 + index,
        )
        for index in range(48)
    ]
    path = cache_path(tmp_path, "XSP", start, end, "1 hour", use_rth=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_cache(path, bars)
    backtest = replace(
        cfg.backtest,
        cache_dir=tmp_path,
        calibration_dir=tmp_path / "calibration",
        output_dir=tmp_path / "out",
    )
    progress = _Progress()

    result = options_sweep._run_group(
        symbol="XSP",
        backtest=backtest,
        synthetic=cfg.synthetic,
        grid=_grid(),
        group=_group(),
        progress=progress,
        jobs=2,
        result_db=tmp_path / "results.sqlite3",
        revision="spawn-revision",
    )

    assert isinstance(result, list)
    assert progress.advanced == 1
