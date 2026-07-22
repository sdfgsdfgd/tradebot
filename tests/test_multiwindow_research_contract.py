from __future__ import annotations

import json
from datetime import date

from tradebot.research.multiwindow import MultiwindowReport, emit_multiwindow_results, parse_multiwindow_args


def test_multiwindow_cli_contract_preserves_defaults() -> None:
    args = parse_multiwindow_args(["--milestones", "milestones.json"])

    assert args.symbol == "TQQQ"
    assert args.bar_size == "1 hour"
    assert args.cache_policy == "auto"
    assert args.jobs == 0
    assert args.top == 200
    assert args.min_trades == 200
    assert not hasattr(args, "include_full")
    assert not hasattr(args, "allow_unlimited_stacking")


def test_multiwindow_report_preserves_operator_and_json_contract(tmp_path, capsys) -> None:
    out_path = tmp_path / "top.json"
    report = MultiwindowReport(
        symbol="XSP",
        bar_size="5 mins",
        use_rth=True,
        offline=True,
        windows=((date(2025, 1, 1), date(2026, 1, 1)),),
        min_trades=20,
        min_win=0.5,
        min_trades_per_year=20.0,
        milestones_path=tmp_path / "source.json",
        write_top=1,
        out_path=out_path,
        cache_path=tmp_path / "eval.sqlite3",
    )
    row = {
        "strategy": {"ema_preset": "5/13", "ema_entry_mode": "trend"},
        "filters": {"rv_min": 0.1},
        "stability": {
            "min_roi_over_dd": 2.0,
            "min_roi": 0.1,
            "min_pnl_over_dd": 2.0,
            "min_pnl": 100.0,
        },
        "primary": {
            "roi_over_dd_pct": 3.0,
            "roi": 0.2,
            "pnl": 200.0,
            "dd": 50.0,
            "dd_pct": 0.05,
            "pnl_over_dd": 4.0,
            "win_rate": 0.6,
            "trades": 30,
        },
        "windows": [],
    }

    emit_multiwindow_results(
        report=report,
        out_rows=[row],
        cache_enabled=True,
        cache_hits=2,
        cache_writes=1,
    )

    rendered = capsys.readouterr().out
    assert "symbol=XSP bar=5 mins rth=True offline=True" in rendered
    assert f"eval_cache={report.cache_path} hits=2 writes=1" in rendered
    payload = json.loads(out_path.read_text())
    assert payload["bar_size"] == "5 mins"
    assert payload["use_rth"] is True
    assert payload["groups"][0]["entries"][0]["symbol"] == "XSP"
    assert payload["groups"][0]["_eval"]["stability"]["min_roi_over_dd"] == 2.0
