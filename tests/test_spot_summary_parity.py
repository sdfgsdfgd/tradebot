"""Parity receipts for the canonical detailed and summary spot lifecycles."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradebot.backtest.data import ContractMeta
from tradebot.backtest.engine import (
    _run_spot_backtest_exec_loop,
    _run_spot_backtest_exec_loop_summary,
    _spot_resolve_run_bars,
)
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
        hi = max(o, c)
        lo = min(o, c)
        bars.append(Bar(ts=ts, open=o, high=hi, low=lo, close=c, volume=1.0))
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


def _tick_daily_bars(*, end_day: date, days: int) -> list[Bar]:
    out: list[Bar] = []
    start_day = end_day - timedelta(days=max(1, int(days)))
    cur = datetime(start_day.year, start_day.month, start_day.day, 0, 0, 0)
    for i in range(int(days)):
        cur = cur + timedelta(days=1)
        base = 100.0 + float(i) * 0.2
        width = 10.0 + float(i % 7)
        out.append(
            Bar(
                ts=cur,
                open=base,
                high=base + width,
                low=base,
                close=base + width * 0.5,
                volume=1.0,
            )
        )
    return out


def test_spot_run_bars_owns_resolution_selection_and_validation() -> None:
    signal = _bars_5m(
        start=datetime(2024, 1, 8, 14),
        end=datetime(2024, 1, 8, 14, 10),
        start_price=100.0,
        end_price=101.0,
    )
    execution = signal[:1]
    cfg = SimpleNamespace(
        backtest=SimpleNamespace(bar_size="30 mins"),
        strategy=SimpleNamespace(spot_exec_bar_size="5 mins"),
    )

    with pytest.raises(ValueError, match="exec_bars was not provided"):
        _spot_resolve_run_bars(cfg, bars=signal)
    assert _spot_resolve_run_bars(
        cfg, bars=signal, exec_bars=execution
    ).execution is execution

    cfg.strategy.spot_exec_bar_size = "30 mins"
    assert _spot_resolve_run_bars(
        cfg, bars=signal, exec_bars=execution
    ).execution is signal


def test_spot_summary_matches_detailed_lifecycle_on_flip_reentry_path() -> None:
    # Synthetic multi-res tape (signal=30m, exec=5m) designed to:
    # - go long on EMA-up state
    # - flip to EMA-down while still profitable (profit-only flip exit)
    # - re-enter short on the same next-open fill after the flip exit
    day = date(2024, 1, 1)
    # Use a UTC timestamp that is the same ET trade date (09:00 ET).
    # Backtest bars are UTC-naive and interpreted as UTC before ET session logic.
    base = datetime(day.year, day.month, day.day, 14, 0, 0)

    # Build an exec price path that rises hard, then drifts down while still above entry, then continues down.
    seg1 = _bars_5m(start=base, end=base + timedelta(minutes=60), start_price=100.0, end_price=102.0)
    seg2 = _bars_5m(start=seg1[-1].ts, end=seg1[-1].ts + timedelta(minutes=30), start_price=seg1[-1].close, end_price=110.0)
    seg3 = _bars_5m(start=seg2[-1].ts, end=seg2[-1].ts + timedelta(minutes=60), start_price=seg2[-1].close, end_price=105.0)
    seg4 = _bars_5m(start=seg3[-1].ts, end=seg3[-1].ts + timedelta(minutes=30), start_price=seg3[-1].close, end_price=100.0)
    exec_bars = seg1 + seg2 + seg3 + seg4
    signal_bars = _extract_signal_bars(exec_bars, every_minutes=30)

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
        spot_controlled_flip=True,
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
    cfg = ConfigBundle(backtest=backtest, strategy=strategy, synthetic=synthetic)
    meta = ContractMeta(symbol="TQQQ", exchange="SMART", multiplier=1.0, min_tick=0.01)

    detailed = _run_spot_backtest_exec_loop(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        meta=meta,
    )

    progress: list[dict[str, object]] = []
    summary = _run_spot_backtest_exec_loop_summary(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        meta=meta,
        tick_bars=None,
        progress_callback=progress.append,
    )

    assert detailed.summary == summary
    assert detailed.equity
    assert int(summary.trades) >= 2
    assert any(row.get("phase") == "summary.path" and row.get("path") == "canonical" for row in progress)


def test_spot_summary_matches_detailed_lifecycle_with_tick_gate() -> None:
    day = date(2024, 1, 8)
    base = datetime(day.year, day.month, day.day, 14, 0, 0)

    seg1 = _bars_5m(start=base, end=base + timedelta(minutes=120), start_price=100.0, end_price=106.0)
    seg2 = _bars_5m(start=seg1[-1].ts, end=seg1[-1].ts + timedelta(minutes=60), start_price=seg1[-1].close, end_price=101.0)
    exec_bars = seg1 + seg2
    signal_bars = _extract_signal_bars(exec_bars, every_minutes=30)
    tick_bars = _tick_daily_bars(end_day=day, days=60)

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
        name="spot_tick_fast_parity",
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
        tick_gate_mode="raschke",
        tick_gate_symbol="TICK-NYSE",
        tick_gate_exchange="NYSE",
        tick_neutral_policy="allow",
        tick_direction_policy="both",
        tick_band_ma_period=10,
        tick_width_z_lookback=30,
        tick_width_z_enter=1.0,
        tick_width_z_exit=0.5,
        tick_width_slope_lookback=3,
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

    detailed = _run_spot_backtest_exec_loop(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        meta=meta,
        tick_bars=tick_bars,
        capture_equity=False,
    ).summary
    summary = _run_spot_backtest_exec_loop_summary(
        cfg,
        signal_bars=signal_bars,
        exec_bars=exec_bars,
        meta=meta,
        tick_bars=tick_bars,
    )
    assert detailed == summary


def test_backtest_sizing_assemblies_delegate_via_typed_payload() -> None:
    import ast
    from pathlib import Path

    source = Path("tradebot/backtest/engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def _leaf(node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _target_names(node) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }

    factory_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _leaf(node.func) == "spot_sizing_input"
    ]
    wrapper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _leaf(node.func) == "spot_calc_signed_qty_with_trace"
    ]
    assert len(wrapper_calls) == 2, "expected entry and resize sizing delegations"
    assert len(factory_calls) == len(wrapper_calls), (
        "backtest sizing adapters must call spot_sizing_input once per kernel "
        f"delegation: factory={len(factory_calls)} kernel={len(wrapper_calls)}"
    )

    factory_targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Call) or _leaf(value.func) != "spot_sizing_input":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                factory_targets.update(_target_names(target))

    raw_fields = {
        "strategy", "filters", "action", "lot", "entry_price", "stop_price",
        "stop_loss_pct", "shock", "shock_dir", "shock_atr_pct",
        "shock_dir_down_streak_bars", "shock_drawdown_dist_on_pct",
        "shock_drawdown_dist_on_vel_pp", "shock_drawdown_dist_on_accel_pp",
        "shock_prearm_down_streak_bars", "shock_ramp", "riskoff", "risk_dir",
        "riskpanic", "riskpop", "risk", "signal_entry_dir",
        "signal_regime_dir", "regime2_dir", "regime2_ready", "equity_ref",
        "cash_ref", "policy_graph", "policy_config",
    }
    for call in wrapper_calls:
        expressions = [*call.args, *(keyword.value for keyword in call.keywords)]
        uses_inline_factory = any(
            isinstance(child, ast.Call) and _leaf(child.func) == "spot_sizing_input"
            for expression in expressions
            for child in ast.walk(expression)
        )
        uses_factory_result = any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in factory_targets
            for expression in expressions
            for child in ast.walk(expression)
        )
        assert uses_inline_factory or uses_factory_result, (
            "backtest sizing kernel must receive the typed factory result"
        )
        raw_keywords = {
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        } & raw_fields
        assert not raw_keywords, (
            "backtest sizing kernel still receives raw adapter fields: "
            f"{sorted(raw_keywords)}"
        )


def test_backtest_pending_state_mutation_matches_shared_transition_table() -> None:
    import inspect
    from datetime import datetime, timedelta

    from tradebot.backtest import engine as backtest_engine
    from tradebot.spot import lifecycle as lifecycle_module

    planner = getattr(lifecycle_module, "plan_pending_mutation", None)
    assert callable(planner), "missing canonical plan_pending_mutation seam"

    plan_type = getattr(lifecycle_module, "SpotPendingMutationPlan", None)
    assert plan_type is not None, "missing typed SpotPendingMutationPlan contract"

    apply_mutation = getattr(backtest_engine, "_spot_apply_pending_mutation", None)
    assert callable(apply_mutation), "missing backtest pending-mutation adapter"

    loop_source = inspect.getsource(backtest_engine._run_spot_backtest_exec_loop)
    assert "plan_pending_mutation(" in loop_source
    assert "_spot_apply_pending_mutation(" in loop_source

    now = datetime(2026, 7, 20, 14, 0)
    due_past = now - timedelta(minutes=1)
    due_future = now + timedelta(minutes=1)
    common = {
        "now_ts": now,
        "has_open": False,
        "open_dir": None,
        "pending_entry_dir": None,
        "pending_entry_set_date": None,
        "pending_entry_due_ts": None,
        "pending_exit_reason": None,
        "pending_exit_due_ts": None,
        "risk_overlay_enabled": False,
        "riskoff_today": False,
        "riskpanic_today": False,
        "riskpop_today": False,
        "riskoff_mode": "hygiene",
        "shock_dir_now": None,
        "riskoff_end_hour": None,
        "naive_ts_mode": "utc",
    }
    cases = [
        ("no_pending", {}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("exit_wait", {"has_open": True, "open_dir": "up", "pending_exit_reason": "flip", "pending_exit_due_ts": due_future}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("exit_due_open", {"has_open": True, "open_dir": "up", "pending_exit_reason": "flip", "pending_exit_due_ts": due_past}, {"clear_entry": False, "clear_exit": True, "queue_intent": "exit", "queue_direction": "up", "queue_reason": "flip"}),
        ("exit_due_flat", {"pending_exit_reason": "flip", "pending_exit_due_ts": due_past}, {"clear_entry": False, "clear_exit": True, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_wait", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_due_flat", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_past}, {"clear_entry": True, "clear_exit": False, "queue_intent": "enter", "queue_direction": "up", "queue_reason": "next_open"}),
        ("entry_due_open", {"has_open": True, "open_dir": "up", "pending_entry_dir": "down", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_past}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_cancel_date_roll", {"pending_entry_dir": "up", "pending_entry_set_date": now.date() - timedelta(days=1), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_cancel_directional_mismatch", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True, "riskoff_mode": "directional", "shock_dir_now": "down"}, {"clear_entry": True, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
        ("entry_directional_match_wait", {"pending_entry_dir": "up", "pending_entry_set_date": now.date(), "pending_entry_due_ts": due_future, "risk_overlay_enabled": True, "riskoff_today": True, "riskoff_mode": "directional", "shock_dir_now": "up"}, {"clear_entry": False, "clear_exit": False, "queue_intent": None, "queue_direction": None, "queue_reason": None}),
    ]

    for scenario, overrides, expected in cases:
        kwargs = dict(common)
        kwargs.update(overrides)
        decision = lifecycle_module.decide_pending_next_open(**kwargs)
        mutation = planner(
            decision,
            pending_entry_direction=kwargs["pending_entry_dir"],
            pending_exit_reason=kwargs["pending_exit_reason"],
            open_dir=kwargs["open_dir"],
        )
        assert isinstance(mutation, plan_type), scenario
        assert mutation.as_payload() == expected, scenario

        if kwargs["pending_entry_dir"] in ("up", "down") and kwargs["pending_entry_due_ts"] is not None:
            pending_entry = backtest_engine._SpotPendingEntry.from_signal(
                direction=str(kwargs["pending_entry_dir"]),
                branch=None,
                set_date=kwargs["pending_entry_set_date"] or now.date(),
                due_ts=kwargs["pending_entry_due_ts"],
                snapshot=None,
            )
        else:
            pending_entry = backtest_engine._SpotPendingEntry()

        pending_exit_all = kwargs["pending_exit_due_ts"] is not None
        pending_exit_reason = str(kwargs["pending_exit_reason"] or "")
        pending_exit_due_ts = kwargs["pending_exit_due_ts"]

        next_entry, next_exit_all, next_exit_reason, next_exit_due_ts = apply_mutation(
            pending_entry=pending_entry,
            pending_exit_all=pending_exit_all,
            pending_exit_reason=pending_exit_reason,
            pending_exit_due_ts=pending_exit_due_ts,
            mutation=mutation,
        )

        assert next_entry.active is (pending_entry.active and not expected["clear_entry"]), scenario
        if expected["clear_entry"]:
            assert next_entry.direction is None and next_entry.due_ts is None, scenario
        elif pending_entry.active:
            assert next_entry.direction == pending_entry.direction, scenario
            assert next_entry.due_ts == pending_entry.due_ts, scenario

        expected_exit_active = pending_exit_all and not expected["clear_exit"]
        assert next_exit_all is expected_exit_active, scenario
        if expected_exit_active:
            assert next_exit_reason == pending_exit_reason, scenario
            assert next_exit_due_ts == pending_exit_due_ts, scenario
        else:
            assert next_exit_reason == "", scenario
            assert next_exit_due_ts is None, scenario
