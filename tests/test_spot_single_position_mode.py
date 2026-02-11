from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path

from tradebot.backtest.config import load_config
from tradebot.backtest.data import ContractMeta
from tradebot.backtest.engine import _run_spot_backtest_summary
from tradebot.backtest.models import Bar
from tradebot.backtest.run_backtests_spot_sweeps import _strategy_from_payload
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


def _base_cfg(day: date) -> ConfigBundle:
    backtest = BacktestConfig(
        start=day,
        end=day,
        bar_size="30 mins",
        use_rth=False,
        starting_cash=100_000.0,
        risk_free_rate=0.0,
        cache_dir=Path("."),
        calibration_dir=Path("."),
        output_dir=Path("."),
        calibrate=False,
        offline=True,
    )
    strategy = SpotStrategyConfig(
        name="spot_test",
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
        flip_exit_only_if_profit=True,
        direction_source="ema",
        directional_legs=None,
        directional_spot={
            "up": SpotLegConfig(action="BUY", qty=1),
            "down": SpotLegConfig(action="SELL", qty=1),
        },
        legs=None,
        filters=None,
        spot_profit_target_pct=None,
        spot_stop_loss_pct=0.25,
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
    return ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)


def test_spot_strategy_ignores_legacy_max_open_field() -> None:
    day = date(2024, 1, 1)
    base = datetime(day.year, day.month, day.day, 0, 0, 0)
    seg1 = _bars_5m(start=base, end=base + timedelta(minutes=60), start_price=100.0, end_price=102.0)
    seg2 = _bars_5m(start=seg1[-1].ts, end=seg1[-1].ts + timedelta(minutes=30), start_price=seg1[-1].close, end_price=110.0)
    seg3 = _bars_5m(start=seg2[-1].ts, end=seg2[-1].ts + timedelta(minutes=60), start_price=seg2[-1].close, end_price=105.0)
    seg4 = _bars_5m(start=seg3[-1].ts, end=seg3[-1].ts + timedelta(minutes=30), start_price=seg3[-1].close, end_price=100.0)
    exec_bars = seg1 + seg2 + seg3 + seg4
    signal_bars = _extract_signal_bars(exec_bars, every_minutes=30)

    cfg = _base_cfg(day)
    assert not hasattr(cfg.strategy, "max_open_trades")

    legacy_payload = asdict(cfg.strategy)
    legacy_payload["max_open_trades"] = 9
    parsed = _strategy_from_payload(legacy_payload, filters=None)
    assert not hasattr(parsed, "max_open_trades")

    meta = ContractMeta(symbol="TQQQ", exchange="SMART", multiplier=1.0, min_tick=0.01)

    summary = _run_spot_backtest_summary(cfg, bars=signal_bars, meta=meta, exec_bars=exec_bars)
    assert int(summary.trades) >= 1


def test_load_config_ignores_legacy_spot_max_open(tmp_path: Path) -> None:
    raw = {
        "backtest": {
            "start": "2025-01-08",
            "end": "2025-01-08",
            "bar_size": "10 mins",
            "use_rth": False,
            "starting_cash": 100000,
            "risk_free_rate": 0.02,
            "cache_dir": "db",
            "calibration_dir": "db/calibration",
            "output_dir": "backtests/out",
            "calibrate": False,
            "offline": True,
        },
        "strategy": {
            "name": "spot_test",
            "instrument": "spot",
            "symbol": "SLV",
            "entry_days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "max_entries_per_day": 0,
            "max_open_trades": 7,
            "dte": 0,
            "otm_pct": 0.0,
            "width_pct": 0.0,
            "profit_target": 0.0,
            "stop_loss": 0.0,
            "exit_dte": 0,
            "quantity": 1,
            "stop_loss_basis": "max_loss",
            "ema_preset": "5/13",
            "ema_entry_mode": "trend",
            "entry_confirm_bars": 0,
            "ema_directional": False,
            "exit_on_signal_flip": True,
            "flip_exit_mode": "entry",
            "flip_exit_gate_mode": "off",
            "flip_exit_min_hold_bars": 0,
            "flip_exit_only_if_profit": True,
            "direction_source": "ema",
            "directional_spot": {"up": {"action": "BUY", "qty": 1}, "down": {"action": "SELL", "qty": 1}},
            "spot_profit_target_pct": None,
            "spot_stop_loss_pct": 0.016,
            "spot_close_eod": False,
            "entry_signal": "ema",
            "spot_exit_mode": "pct",
            "spot_atr_period": 14,
            "spot_pt_atr_mult": 1.5,
            "spot_sl_atr_mult": 1.0,
            "spot_exec_bar_size": "5 mins",
            "regime_mode": "supertrend",
            "regime_bar_size": "1 day",
            "supertrend_atr_period": 7,
            "supertrend_multiplier": 0.4,
            "supertrend_source": "close",
            "spot_entry_fill_mode": "next_open",
            "spot_flip_exit_fill_mode": "next_open",
            "spot_intrabar_exits": True,
            "spot_spread": 0.01,
            "spot_commission_per_share": 0.005,
            "spot_commission_min": 1.0,
            "spot_slippage_per_share": 0.0,
            "spot_mark_to_market": "liquidation",
            "spot_drawdown_mode": "intrabar",
            "spot_sizing_mode": "risk_pct",
            "spot_notional_pct": 0.0,
            "spot_risk_pct": 0.016,
            "spot_short_risk_mult": 1.0,
            "spot_max_notional_pct": 0.8,
            "spot_min_qty": 1,
            "spot_max_qty": 0,
        },
        "synthetic": {
            "rv_lookback": 60,
            "rv_ewma_lambda": 0.94,
            "iv_risk_premium": 1.0,
            "iv_floor": 0.0,
            "term_slope": 0.0,
            "skew": 0.0,
            "min_spread_pct": 0.0,
        },
    }
    cfg_path = tmp_path / "legacy_spot_max_open.json"
    cfg_path.write_text(json.dumps(raw))
    cfg = load_config(cfg_path)
    assert cfg.strategy.instrument == "spot"
    assert not hasattr(cfg.strategy, "max_open_trades")


def test_load_config_options_keeps_max_open(tmp_path: Path) -> None:
    raw = {
        "backtest": {
            "start": "2025-01-08",
            "end": "2025-01-09",
        },
        "strategy": {
            "instrument": "options",
            "symbol": "SLV",
            "max_open_trades": 3,
        },
    }
    cfg_path = tmp_path / "options_with_max_open.json"
    cfg_path.write_text(json.dumps(raw))
    cfg = load_config(cfg_path)
    assert cfg.strategy.instrument == "options"
    assert hasattr(cfg.strategy, "max_open_trades")
    assert int(getattr(cfg.strategy, "max_open_trades")) == 3
