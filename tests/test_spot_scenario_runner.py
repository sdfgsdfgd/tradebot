from __future__ import annotations

from datetime import datetime

from tradebot.spot.lifecycle import SpotLifecycleDecision
from tradebot.spot.scenario import lifecycle_trace_row, why_not_exit_resize_report


def test_why_not_exit_resize_report_filters_hold_rows() -> None:
    ts = datetime(2026, 2, 14, 15, 30, 0)
    hold_exit = SpotLifecycleDecision(
        intent="hold",
        reason="holding",
        gate="HOLDING",
        trace={"stage": "open", "path": "hold"},
    )
    resize_fire = SpotLifecycleDecision(
        intent="resize",
        reason="target_delta",
        gate="TRIGGER_RESIZE",
        trace={"stage": "open", "path": "resize"},
    )
    hold_resize = SpotLifecycleDecision(
        intent="hold",
        reason="resize_cooldown",
        gate="BLOCKED_RESIZE_COOLDOWN",
        blocked=True,
        trace={"stage": "open", "path": "resize"},
    )

    rows = [
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_exit",
            decision=hold_exit,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_resize",
            decision=resize_fire,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
        lifecycle_trace_row(
            bar_ts=ts,
            stage="open_resize",
            decision=hold_resize,
            context={"symbol": "SLV", "exec_idx": 10, "sig_idx": 3},
        ),
    ]

    report = why_not_exit_resize_report(rows)
    assert len(report) == 2
    assert [str(row["target"]) for row in report] == ["exit", "resize"]
    assert str(report[1]["reason"]) == "resize_cooldown"
    assert bool(report[1]["blocked"]) is True


def test_backtest_trace_receipt_projection_matches_shared_schema() -> None:
    from datetime import date, datetime, timedelta
    from pathlib import Path
    from tradebot.backtest.data import ContractMeta
    from tradebot.backtest.engine import _run_spot_backtest_exec_loop, _run_spot_backtest_exec_loop_summary
    from tradebot.backtest.models import Bar
    from tradebot.knobs.models import BacktestConfig, ConfigBundle, SpotLegConfig, SpotStrategyConfig, SyntheticConfig

    def _bars_5m(*, start: datetime, end: datetime, start_price: float, end_price: float) -> list[Bar]:
        if end <= start:
            raise ValueError('end must be > start')
        if (end - start).total_seconds() % 300:
            raise ValueError('expected 5m alignment')
        n = int((end - start).total_seconds() // 300)
        closes: list[float] = []
        for i in range(1, n + 1):
            frac = float(i) / float(n)
            closes.append(float(start_price) + (float(end_price) - float(start_price)) * float(frac))
        bars: list[Bar] = []
        prev_close = float(start_price)
        ts = start
        for close in closes:
            ts = ts + timedelta(minutes=5)
            o = float(prev_close)
            c = float(close)
            hi = max(o, c)
            lo = min(o, c)
            bars.append(Bar(ts=ts, open=o, high=hi, low=lo, close=c, volume=1.0))
            prev_close = c
        return bars

    def _extract_signal_bars(exec_bars: list[Bar], *, every_minutes: int) -> list[Bar]:
        out: list[Bar] = []
        for b in exec_bars:
            if b.ts.minute % int(every_minutes) == 0:
                out.append(Bar(ts=b.ts, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume))
        if not out:
            raise ValueError('no signal bars extracted')
        return out
    day = date(2024, 1, 1)
    base = datetime(day.year, day.month, day.day, 14, 0, 0)
    seg1 = _bars_5m(start=base, end=base + timedelta(minutes=60), start_price=100.0, end_price=102.0)
    seg2 = _bars_5m(start=seg1[-1].ts, end=seg1[-1].ts + timedelta(minutes=30), start_price=seg1[-1].close, end_price=110.0)
    seg3 = _bars_5m(start=seg2[-1].ts, end=seg2[-1].ts + timedelta(minutes=60), start_price=seg2[-1].close, end_price=105.0)
    seg4 = _bars_5m(start=seg3[-1].ts, end=seg3[-1].ts + timedelta(minutes=30), start_price=seg3[-1].close, end_price=100.0)
    exec_bars = seg1 + seg2 + seg3 + seg4
    signal_bars = _extract_signal_bars(exec_bars, every_minutes=30)
    backtest = BacktestConfig(start=day, end=day, bar_size='30 mins', use_rth=False, starting_cash=100000.0, risk_free_rate=0.0, cache_dir=Path('.'), calibration_dir=Path('.'), output_dir=Path('.'), calibrate=False, offline=True)
    strategy = SpotStrategyConfig(name='spot_test', instrument='spot', symbol='TQQQ', exchange=None, right='PUT', entry_days=(0, 1, 2, 3, 4), max_entries_per_day=0, dte=0, otm_pct=0.0, width_pct=0.0, profit_target=0.0, stop_loss=0.0, exit_dte=0, quantity=1, stop_loss_basis='max_loss', min_credit=None, ema_preset='1/2', ema_entry_mode='trend', entry_confirm_bars=0, regime_ema_preset=None, regime_bar_size=None, ema_directional=False, exit_on_signal_flip=True, flip_exit_mode='entry', flip_exit_gate_mode='off', flip_exit_min_hold_bars=0, flip_exit_only_if_profit=True, direction_source='ema', directional_legs=None, directional_spot={'up': SpotLegConfig(action='BUY', qty=1), 'down': SpotLegConfig(action='SELL', qty=1)}, legs=None, filters=None, spot_profit_target_pct=None, spot_stop_loss_pct=0.25, spot_close_eod=False, spot_exec_bar_size='5 mins', spot_entry_fill_mode='next_open', spot_flip_exit_fill_mode='next_open', spot_controlled_flip=True, spot_intrabar_exits=True, spot_spread=0.0, spot_commission_per_share=0.0, spot_commission_min=0.0, spot_slippage_per_share=0.0, spot_mark_to_market='close', spot_drawdown_mode='intrabar', spot_sizing_mode='fixed')
    synthetic = SyntheticConfig(rv_lookback=60, rv_ewma_lambda=0.94, iv_risk_premium=1.0, iv_floor=0.0, term_slope=0.0, skew=0.0, min_spread_pct=0.0)
    cfg = ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)
    meta = ContractMeta(symbol='TQQQ', exchange='SMART', multiplier=1.0, min_tick=0.01)
    detailed = _run_spot_backtest_exec_loop(cfg, signal_bars=signal_bars, exec_bars=exec_bars, meta=meta)
    progress: list[dict[str, object]] = []
    summary = _run_spot_backtest_exec_loop_summary(cfg, signal_bars=signal_bars, exec_bars=exec_bars, meta=meta, tick_bars=None, progress_callback=progress.append)
    assert detailed.summary == summary
    assert detailed.equity
    assert int(summary.trades) >= 2
    assert any((row.get('phase') == 'summary.path' and row.get('path') == 'canonical' for row in progress))
    traced = [trade for trade in detailed.trades if isinstance(getattr(trade, 'decision_trace', None), dict)]
    assert traced, 'production backtest route produced no traced SpotTrade'
    trade = traced[0]
    trace = trade.decision_trace
    assert 'spot_trace_receipt' in trace, 'production backtest route must populate spot_trace_receipt'
    receipt = trace['spot_trace_receipt']
    assert list(receipt) == ['schema', 'sizing', 'intent', 'lifecycle', 'fill', 'accounting']
    assert receipt['schema'] == 'spot-trace-receipt-v1'
    assert isinstance(receipt['sizing'], dict)
    assert 'spot_trace_receipt' not in receipt['sizing']
    assert receipt['intent'] == trace.get('spot_intent')
    assert receipt['lifecycle'] == trace.get('spot_lifecycle')
    assert receipt['fill']['resizes'] == list(trace.get('resizes') or [])
    assert receipt['fill']['exits'] == list(trace.get('exits') or [])
    assert receipt['accounting']['qty'] == trade.qty
    assert receipt['accounting']['entry_price'] == trade.entry_price
    assert receipt['accounting']['margin_required'] == trade.margin_required
    assert receipt['accounting']['exit_price'] == trade.exit_price
    assert receipt['accounting']['exit_reason'] == trade.exit_reason
    assert receipt['accounting']['exit_time'] == (trade.exit_time.isoformat() if trade.exit_time is not None else None)
