from __future__ import annotations

import json
from datetime import date

import tradebot.backtest.sweep_parallel as sweep_parallel
from tradebot.research.multiwindow import (
    MultiwindowReport,
    candidate_shortlist,
    emit_multiwindow_results,
    multiwindow_cache_key,
    parse_multiwindow_args,
    promotion_receipt,
)


def test_parallel_stage_kernel_owns_hardware_and_work_caps(monkeypatch) -> None:
    launched: dict[str, object] = {}
    monkeypatch.setattr(sweep_parallel, "normalize_jobs", lambda _jobs: 3)
    monkeypatch.setattr(
        sweep_parallel,
        "_run_parallel_json_worker_plan",
        lambda **kwargs: launched.update(kwargs) or {},
    )

    workers, payloads = sweep_parallel._run_parallel_stage_kernel(
        stage_label="test",
        jobs=99,
        total=2,
        offline=True,
        offline_error="offline required",
        tmp_prefix="test_",
        worker_tag="worker",
        out_prefix="out",
        build_cmd=lambda _worker, _workers, _path: [],
        capture_error="capture",
        failure_label="failure",
        missing_label="missing",
        invalid_label="invalid",
    )

    assert workers == 2
    assert launched["jobs_eff"] == 2
    assert payloads == {}


def test_multiwindow_cli_contract_preserves_defaults() -> None:
    args = parse_multiwindow_args(["--milestones", "milestones.json"])

    assert args.symbol == "TQQQ"
    assert args.bar_size == "1 hour"
    assert args.cache_policy == "auto"
    assert args.jobs == 0
    assert args.top == 200
    assert args.min_trades == 200
    assert args.track == "auto"
    assert not hasattr(args, "include_full")
    assert not hasattr(args, "allow_unlimited_stacking")


def test_multiwindow_cache_key_is_bound_to_market_data_revision() -> None:
    kwargs = {
        "engine_version": "test-v1",
        "strategy": {"ema_preset": "5/13"},
        "filters": {"rv_min": 0.1},
        "windows": ((date(2025, 1, 1), date(2026, 1, 1)),),
        "min_trades": 20,
        "min_trades_per_year": 20.0,
        "min_win": 0.5,
        "require_close_eod": False,
        "require_positive_pnl": True,
        "offline": True,
    }

    before = multiwindow_cache_key(data_revision="cache-v1:a", **kwargs)
    after = multiwindow_cache_key(data_revision="cache-v1:b", **kwargs)

    assert before != after


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
        track="HF",
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
    assert "symbol=XSP track=HF bar=5 mins rth=True offline=True" in rendered
    assert f"eval_cache={report.cache_path} hits=2 writes=1" in rendered
    payload = json.loads(out_path.read_text())
    assert payload["bar_size"] == "5 mins"
    assert payload["use_rth"] is True
    assert payload["groups"][0]["entries"][0]["symbol"] == "XSP"
    assert payload["schema"] == "tradebot.research.multiwindow.v1"
    assert payload["track"] == "HF"
    assert payload["groups"][0]["_track"] == "HF"
    assert payload["groups"][0]["_eval"]["stability"]["min_roi_over_dd"] == 2.0


def test_candidate_shortlist_reads_every_entry_and_preserves_objective_leaders() -> None:
    def entry(name: str, *, ratio: float, pnl: float, win: float, trades: int) -> dict:
        return {
            "symbol": "XSP",
            "strategy": {
                "instrument": "spot",
                "symbol": "XSP",
                "signal_bar_size": "5 mins",
                "signal_use_rth": True,
                "ema_preset": name,
            },
            "metrics": {
                "pnl_over_dd": ratio,
                "pnl": pnl,
                "win_rate": win,
                "trades": trades,
            },
        }

    payload = {
        "track": "HF",
        "groups": [
            {
                "name": "islands",
                "entries": [
                    entry("risk", ratio=10, pnl=10, win=0.5, trades=10),
                    entry("return", ratio=2, pnl=100, win=0.5, trades=20),
                    entry("quality", ratio=3, pnl=30, win=0.9, trades=30),
                    entry("activity", ratio=4, pnl=40, win=0.6, trades=100),
                ],
            }
        ],
    }

    shortlisted = candidate_shortlist(
        payload,
        symbol="XSP",
        bar_size="5 mins",
        use_rth=True,
        limit=4,
        track="hf",
    )
    assert {row["strategy"]["ema_preset"] for row in shortlisted} == {
        "risk",
        "return",
        "quality",
        "activity",
    }
    assert {row["track"] for row in shortlisted} == {"HF"}


def test_promotion_receipt_compares_only_exact_incumbent_windows() -> None:
    incumbent = (
        {
            "start": "2024-01-01",
            "end": "2025-01-01",
            "pnl": 100.0,
            "pnl_over_dd": 2.0,
            "trades": 20,
        },
        {
            "start": "2025-01-01",
            "end": "2026-01-01",
            "pnl": 120.0,
            "pnl_over_dd": 3.0,
            "trades": 30,
        },
    )
    candidate = [
        {
            "start": "2024-01-01",
            "end": "2025-01-01",
            "pnl": 110.0,
            "pnl_over_dd": 2.5,
            "trades": 21,
        },
        {
            "start": "2025-01-01",
            "end": "2026-01-01",
            "pnl": 130.0,
            "pnl_over_dd": 3.5,
            "trades": 31,
        },
    ]

    receipt = promotion_receipt(candidate, incumbent)
    assert receipt is not None
    assert receipt["complete"] is True
    assert receipt["positive_all"] is True
    assert receipt["ratio_dominates_all"] is True
    assert receipt["pnl_dominates_all"] is True
    assert receipt["activity_preserved_all"] is True
    assert receipt["floor_delta"] == 0.5

    partial = promotion_receipt(candidate[:1], incumbent)
    assert partial is not None
    assert partial["complete"] is False
    assert partial["floor_delta"] is None
