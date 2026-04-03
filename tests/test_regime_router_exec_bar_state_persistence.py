from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

from tradebot.backtest.data import ContractMeta
from tradebot.backtest.engine import _run_spot_backtest_exec_loop
from tradebot.backtest.models import Bar
from tradebot.knobs.models import BacktestConfig, ConfigBundle, SpotLegConfig, SpotStrategyConfig, SyntheticConfig


def _bars_5m(*, start: datetime, end: datetime, start_price: float, end_price: float) -> list[Bar]:
    if end <= start:
        raise ValueError("end must be > start")
    if (end - start).total_seconds() % 300:
        raise ValueError("expected 5m alignment")

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
        bars.append(Bar(ts=ts, open=o, high=max(o, c), low=min(o, c), close=c, volume=1.0))
        prev_close = c
    return bars


def _extract_signal_bars(exec_bars: list[Bar], *, every_minutes: int) -> list[Bar]:
    out: list[Bar] = []
    for b in exec_bars:
        if (b.ts.minute % int(every_minutes)) == 0:
            out.append(Bar(ts=b.ts, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume))
    if not out:
        raise ValueError("no signal bars extracted")
    return out


def test_regime_router_host_managed_persists_between_signal_bars() -> None:
    """Regression: multi-res spot backtests must not re-enable pct stops on non-signal exec bars.

    Setup:
    - signal bars: 30m closes (policy + router update cadence)
    - exec bars: 5m (intrabar exits)
    - regime_router: enabled and forced to become ready quickly

    Expected:
    - When router selects a host-managed lane (e.g. buyhold), stop_loss_pct exits never trigger,
      even on exec bars that are not signal-bar closes.
    """

    day1 = date(2024, 1, 2)
    day2 = date(2024, 1, 3)
    day3 = date(2024, 1, 4)

    backtest = BacktestConfig(
        start=day1,
        end=day3,
        bar_size="30 mins",
        use_rth=False,
        starting_cash=100_000.0,
        risk_free_rate=0.0,
        cache_dir=".",
        calibration_dir=".",
        output_dir=".",
        calibrate=False,
        offline=True,
    )
    strategy = SpotStrategyConfig(
        name="spot_router_state_persistence_test",
        instrument="spot",
        symbol="TQQQ",
        exchange=None,
        right="PUT",
        entry_days=(0, 1, 2, 3, 4),
        max_entries_per_day=0,
        dte=0,
        otm_pct=0.0,
        width_pct=0.0,
        profit_target=0.0,
        stop_loss=0.0,
        exit_dte=0,
        quantity=1,
        stop_loss_basis="max_loss",
        min_credit=None,
        ema_preset="1/2",
        ema_entry_mode="trend",
        entry_confirm_bars=0,
        regime_ema_preset=None,
        regime_bar_size=None,
        ema_directional=False,
        exit_on_signal_flip=True,
        flip_exit_mode="entry",
        flip_exit_gate_mode="off",
        flip_exit_min_hold_bars=0,
        flip_exit_only_if_profit=False,
        direction_source="ema",
        directional_legs=None,
        directional_spot={
            "up": SpotLegConfig(action="BUY", qty=1),
            "down": SpotLegConfig(action="SELL", qty=1),
        },
        legs=None,
        filters=None,
        spot_profit_target_pct=None,
        spot_stop_loss_pct=0.02,
        spot_close_eod=False,
        spot_exec_bar_size="5 mins",
        spot_entry_fill_mode="next_open",
        spot_flip_exit_fill_mode="next_open",
        spot_intrabar_exits=True,
        spot_spread=0.0,
        spot_commission_per_share=0.0,
        spot_commission_min=0.0,
        spot_slippage_per_share=0.0,
        spot_mark_to_market="close",
        spot_drawdown_mode="intrabar",
        spot_sizing_mode="fixed",
        # Router: make it ready quickly so day3 is host-managed buyhold.
        regime_router=True,
        regime_router_fast_window_days=2,
        regime_router_slow_window_days=2,
        regime_router_min_dwell_days=1,
    )
    synthetic = SyntheticConfig(
        rv_lookback=60,
        rv_ewma_lambda=0.94,
        iv_risk_premium=1.0,
        iv_floor=0.0,
        term_slope=0.0,
        skew=0.0,
        min_spread_pct=0.0,
    )
    cfg = ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)
    meta = ContractMeta(symbol="TQQQ", exchange="SMART", multiplier=1.0, min_tick=0.01)

    def _day_base(d: date) -> datetime:
        # Keep synthetic tape on the intended ET trade date (09:00 ET in UTC).
        return datetime(d.year, d.month, d.day, 14, 0, 0)

    # Day 1/2: steady grind up (router warmup).
    exec_bars = []
    exec_bars += _bars_5m(start=_day_base(day1), end=_day_base(day1) + timedelta(minutes=180), start_price=100.0, end_price=102.0)
    exec_bars += _bars_5m(start=_day_base(day2), end=_day_base(day2) + timedelta(minutes=180), start_price=102.0, end_price=104.0)

    # Day 3: enter, then a sharp dip on a *non-signal* exec bar that would stop out if stops were enabled.
    day3_bars = _bars_5m(start=_day_base(day3), end=_day_base(day3) + timedelta(minutes=180), start_price=104.0, end_price=106.0)
    # Ensure a stop-triggering low at 14:40 (minute=40) which is not a 30m signal close.
    for i, b in enumerate(day3_bars):
        if b.ts.hour == 14 and b.ts.minute == 40:
            entry_ref = day3_bars[i - 1].close if i > 0 else b.open
            stop_level = float(entry_ref) * (1.0 - 0.02)
            dip_low = float(stop_level) * 0.99
            day3_bars[i] = Bar(
                ts=b.ts,
                open=b.open,
                high=b.high,
                low=min(float(b.low), dip_low),
                close=min(float(b.close), dip_low),
                volume=b.volume,
            )
            break
    exec_bars += day3_bars

    signal_bars = _extract_signal_bars(exec_bars, every_minutes=30)
    res = _run_spot_backtest_exec_loop(cfg, signal_bars=signal_bars, exec_bars=exec_bars, meta=meta, capture_equity=False)

    assert res.trades, "expected at least one trade"
    stop_exits = [t for t in res.trades if t.exit_reason in ("stop_loss", "stop_loss_pct")]
    assert not stop_exits, f"unexpected stop exits under host-managed router: {stop_exits!r}"

