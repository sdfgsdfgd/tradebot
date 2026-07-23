from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tradebot.backtest.data import ContractMeta
from tradebot.backtest.models import SummaryStats
from tradebot.backtest.spot_context import SpotContextBars
import tradebot.backtest.sweep_parallel as sweep_parallel
import tradebot.research.spot_sweeps.evaluation as sweep_evaluation
from tradebot.research.spot_sweeps.axes_hf import _orb_candidates
from tradebot.research.spot_sweeps.catalog import (
    _AXIS_CHOICES,
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
    _combo_full_preset_axes,
    _combo_full_preset_freeze_dims,
)
from tradebot.research.spot_sweeps.fingerprints import (
    _axis_dimension_fingerprint,
    _combo_full_dimension_space_signature,
    _window_signature,
)
from tradebot.research.spot_sweeps.dimensions import (
    _AXIS_DIMENSION_REGISTRY,
    _SWEEP_COST_MODEL,
    _SWEEP_RUNTIME_POLICY,
    _ema_signal_presets,
)
from tradebot.research.spot_sweeps.cli import parse_spot_sweep_args
import tradebot.research.spot_sweeps.combo as sweep_combo
from tradebot.research.spot_sweeps.combo import SweepCartesian, _objective_shortlist
from tradebot.research.spot_sweeps.milestones import _rank_cfg_rows, _spot_strategy_payload
from tradebot.research.spot_sweeps.runtime import SpotSweepRuntime
from tradebot.research.spot_sweeps.stages import SweepStages
from tradebot.research.spot_sweeps.support import _bundle_base


def _signature(
    *,
    timing_rank_min: float,
    pair_payload: dict[str, object],
) -> str:
    pair_variants_by_dim = {
        str(dim_name): [(f"{dim_name}=base", dict(pair_payload))]
        for dim_name, _variants_key in _COMBO_FULL_PAIR_DIM_VARIANT_SPECS
    }
    size_by_dim = {str(dim_name): 1 for dim_name in _COMBO_FULL_CARTESIAN_DIM_ORDER}
    return _combo_full_dimension_space_signature(
        ordered_dims=list(_COMBO_FULL_CARTESIAN_DIM_ORDER),
        size_by_dim=size_by_dim,
        timing_profile_variants=[
            (
                "timing=hf_symm",
                {"entry_signal": "ema", "spot_branch_b_size_mult": 1.35},
                {
                    "ratsv_branch_a_rank_min": float(timing_rank_min),
                    "ratsv_branch_a_cross_age_max_bars": 4,
                },
            )
        ],
        confirm_bars=[0],
        pair_variants_by_dim=pair_variants_by_dim,
        short_mults=[1.0],
    )


def test_combo_full_signature_changes_when_variant_payload_changes() -> None:
    sig_a = _signature(timing_rank_min=0.0240, pair_payload={})
    sig_b = _signature(timing_rank_min=0.0245, pair_payload={})
    assert sig_a != sig_b


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


def test_combo_full_stability_is_one_pipeline_contract() -> None:
    args = parse_spot_sweep_args(
        [
            "--axis",
            "combo_full",
            "--track",
            "lf",
            "--stability-window",
            "2024-01-01:2026-01-01",
            "--stability-window",
            "2025-01-01:2026-01-01",
            "--promote",
        ]
    )

    assert args.stability_window == [
        "2024-01-01:2026-01-01",
        "2025-01-01:2026-01-01",
    ]
    assert args.stability_top == 200
    assert args.stability_write_top == 200
    assert args.promotion_objective == "stability"
    assert args.promote is True


def test_stability_options_are_rejected_outside_full_combo() -> None:
    with pytest.raises(SystemExit):
        parse_spot_sweep_args(["--axis", "all", "--stability-window", "2025-01-01:2026-01-01"])


def test_combo_full_shortlist_preserves_distinct_economic_leaders() -> None:
    def candidate(name: str, **metrics):
        cfg = _bundle_base(
            symbol="SLV",
            start=date(2025, 1, 8),
            end=date(2025, 1, 10),
            bar_size="15 mins",
            use_rth=False,
            cache_dir=Path("db"),
            offline=True,
            filters=None,
            entry_signal="ema",
        )
        cfg = replace(cfg, strategy=replace(cfg.strategy, ema_preset=name))
        return cfg, metrics, name

    candidates = [
        candidate("2/4", pnl_over_dd=10, pnl=10, win_rate=0.5, trades=10),
        candidate("3/7", pnl_over_dd=2, pnl=100, win_rate=0.5, trades=20),
        candidate("4/9", pnl_over_dd=3, pnl=30, win_rate=0.9, trades=30),
        candidate("5/13", pnl_over_dd=4, pnl=40, win_rate=0.6, trades=100),
    ]

    shortlisted = _objective_shortlist(candidates, limit=4)

    assert {cfg.strategy.ema_preset for cfg, _row, _note in shortlisted} == {
        "2/4",
        "3/7",
        "4/9",
        "5/13",
    }


def test_combo_full_seed_source_decodes_every_matching_lane_entry() -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.symbol = "SLV"
    runtime.start = date(2025, 1, 1)
    runtime.end = date(2026, 1, 1)
    runtime.signal_bar_size = "10 mins"
    runtime.use_rth = False
    runtime.cache_dir = Path("db")
    runtime.offline = True
    base = _bundle_base(
        symbol="SLV",
        start=runtime.start,
        end=runtime.end,
        bar_size="10 mins",
        use_rth=False,
        cache_dir=runtime.cache_dir,
        offline=True,
        filters=None,
    )
    strategy = _spot_strategy_payload(base, meta=ContractMeta("SLV", "SMART", 1.0, 0.01))
    runtime.milestones = {
        "groups": [
            {
                "name": "candidate island",
                "filters": {"rv_min": 0.1},
                "entries": [
                    {
                        "symbol": "SLV",
                        "metrics": {"pnl": pnl, "pnl_over_dd": pnl / 10.0},
                        "strategy": {
                            **strategy,
                            "ema_preset": preset,
                            "signal_use_rth": "false",
                            "filters": {"rv_max": 0.5},
                        },
                    }
                    for preset, pnl in (("2/4", 100.0), ("3/7", 80.0))
                ],
            }
        ]
    }

    candidates = runtime._combo_full_seed_candidates()

    assert {cfg.strategy.ema_preset for cfg, _metrics, _note in candidates} == {"2/4", "3/7"}
    assert all(cfg.strategy.filters.rv_min == 0.1 for cfg, _metrics, _note in candidates)
    assert all(cfg.strategy.filters.rv_max == 0.5 for cfg, _metrics, _note in candidates)


def test_spot_strategy_payload_infers_mcl_futures_identity_from_contract_meta() -> None:
    cfg = _bundle_base(
        symbol="MCL",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size="5 mins",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
        entry_signal="ema",
    )

    strategy = _spot_strategy_payload(
        cfg,
        meta=ContractMeta("MCL", "NYMEX", 100.0, 0.01),
    )

    assert {
        "spot_sec_type": strategy.get("spot_sec_type"),
        "spot_exchange": strategy.get("spot_exchange"),
    } == {
        "spot_sec_type": "FUT",
        "spot_exchange": "NYMEX",
    }


def test_combo_full_stability_reuses_exact_windows_and_canonical_evaluator(
    tmp_path,
    monkeypatch,
) -> None:
    windows = (
        {"start": "2024-01-01", "end": "2025-01-01", "pnl": 10.0, "pnl_over_dd": 1.0},
        {"start": "2025-01-01", "end": "2026-01-01", "pnl": 10.0, "pnl_over_dd": 1.0},
    )
    monkeypatch.setattr(
        sweep_combo,
        "load_current_champion_groups",
        lambda **_kwargs: (
            [
                {
                    "_track": "LF",
                    "entries": [
                        {
                            "strategy": {
                                "signal_bar_size": "15 mins",
                                "signal_use_rth": "false",
                            }
                        }
                    ],
                    "_eval": {"windows": list(windows)},
                }
            ],
            [],
        ),
    )

    class FakeRuntime(SweepCartesian):
        evaluated: list[tuple[str, date, date]] = []

        def __init__(self, args) -> None:
            self.args = args
            self.offline = False
            self.run_cfg_persistent_conn = None
            self.data = SimpleNamespace(disconnect=lambda: None)

        def _run_cfg(self, *, cfg):
            self.evaluated.append(
                (str(cfg.strategy.ema_preset), cfg.backtest.start, cfg.backtest.end)
            )
            strength = {"2/4": 4.0, "3/7": 2.0}.get(cfg.strategy.ema_preset, -1.0)
            return {
                "trades": 50,
                "win_rate": 0.6,
                "pnl": 100.0 * strength,
                "dd": 100.0,
                "roi": strength,
                "dd_pct": 1.0,
                "pnl_over_dd": strength,
            }

        def _run_cfg_pairs_grid(self, *, cfg_pairs, rows, on_row, **_kwargs):
            for cfg, note in cfg_pairs:
                row = self._run_cfg(cfg=cfg)
                if row is not None:
                    rows.append(row)
                    on_row(cfg, row, note)
            return len(cfg_pairs)

        def _run_cfg_persistent_flush_pending(self, *, force: bool = False) -> None:
            assert force is True

    out = tmp_path / "stability.json"
    args = SimpleNamespace(
        stability_window=[
            "2024-01-01:2025-01-01",
            "2025-01-01:2026-01-01",
        ],
        stability_top=3,
        stability_min_trades_per_year=None,
        stability_out=str(out),
        promote=False,
        promotion_objective="stability",
        promotion_version="",
        track="lf",
        min_trades=1,
        combo_full_preset="profile",
    )
    runtime = FakeRuntime(args)
    runtime.symbol = "SLV"
    runtime.start = date(2025, 1, 1)
    runtime.end = date(2026, 1, 1)
    runtime.signal_bar_size = "15 mins"
    runtime.use_rth = False
    runtime.cache_dir = tmp_path
    runtime.cache_policy = "strict"
    runtime.meta = ContractMeta("SLV", "SMART", 1.0, 0.01)
    configs = []
    for preset, pnl in (("2/4", 100.0), ("3/7", 80.0), ("4/9", 20.0)):
        cfg = _bundle_base(
            symbol="SLV",
            start=runtime.start,
            end=runtime.end,
            bar_size="15 mins",
            use_rth=False,
            cache_dir=tmp_path,
            offline=False,
            filters=None,
        )
        cfg = replace(cfg, strategy=replace(cfg.strategy, ema_preset=preset))
        configs.append((cfg, {"pnl": pnl, "pnl_over_dd": pnl / 10.0}, preset))

    runtime._combo_full_stability(configs)

    payload = json.loads(out.read_text())
    assert payload["schema"] == "tradebot.research.stability.v1"
    assert payload["windows"] == [
        {"start": "2024-01-01", "end": "2025-01-01"},
        {"start": "2025-01-01", "end": "2026-01-01"},
    ]
    assert payload["groups"][0]["entries"][0]["strategy"]["ema_preset"] == "2/4"
    assert payload["groups"][0]["_eval"]["promotion"]["eligible"] is True
    assert {
        group["entries"][0]["strategy"]["ema_preset"] for group in payload["groups"]
    } == {"2/4", "3/7", "4/9"}
    losing = next(
        group
        for group in payload["groups"]
        if group["entries"][0]["strategy"]["ema_preset"] == "4/9"
    )
    assert losing["_eval"]["promotion"]["eligible"] is False
    assert set(FakeRuntime.evaluated) == {
        ("2/4", date(2024, 1, 1), date(2025, 1, 1)),
        ("2/4", date(2025, 1, 1), date(2026, 1, 1)),
        ("3/7", date(2024, 1, 1), date(2025, 1, 1)),
        ("3/7", date(2025, 1, 1), date(2026, 1, 1)),
        ("4/9", date(2024, 1, 1), date(2025, 1, 1)),
        ("4/9", date(2025, 1, 1), date(2026, 1, 1)),
    }


def test_combo_full_signature_is_stable_for_reordered_dict_keys() -> None:
    sig_a = _signature(timing_rank_min=0.0240, pair_payload={"alpha": 1, "beta": 2})
    sig_b = _signature(timing_rank_min=0.0240, pair_payload={"beta": 2, "alpha": 1})
    assert sig_a == sig_b


def test_sweep_policy_is_not_mixed_into_strategy_dimensions() -> None:
    assert "cache" not in _AXIS_DIMENSION_REGISTRY
    assert "cost_model" not in _AXIS_DIMENSION_REGISTRY
    assert "base" in _SWEEP_COST_MODEL
    assert "jobs_tuner" in _SWEEP_RUNTIME_POLICY


def test_ema_signal_presets_have_one_canonical_catalog() -> None:
    assert _ema_signal_presets("tight") == ("2/4", "4/9")
    assert _ema_signal_presets("core") == (
        "2/4",
        "3/7",
        "4/9",
        "5/10",
        "8/21",
        "9/21",
    )
    assert _ema_signal_presets("combo") == (
        "2/4",
        "3/7",
        "4/9",
        "5/10",
        "5/13",
        "8/21",
        "9/21",
        "21/50",
    )


def test_legacy_hf_scalp_axis_is_replaced_by_unified_hf_preset() -> None:
    assert "hf_scalp" not in _AXIS_CHOICES
    assert "hf_timing_sniper" in _combo_full_preset_axes()


def test_combo_full_presets_explore_only_their_declared_dimensions() -> None:
    expected = {
        "full": _COMBO_FULL_CARTESIAN_DIM_ORDER,
        "baseline": (),
        "profile": ("timing_profile",),
        "gate": ("perm", "tod", "vol", "cadence"),
        "ema": ("direction", "perm", "tod", "vol"),
        "tick": ("perm", "tod", "vol", "tick"),
        "regime": ("regime", "exit"),
        "risk": ("risk",),
        "squeeze": ("confirm", "tod", "vol", "regime2"),
        "tod_interaction": ("tod", "cadence"),
        "perm_joint": ("perm", "tod", "vol", "cadence"),
        "ema_perm_joint": ("direction", "perm", "tod", "vol"),
        "tick_perm_joint": ("perm", "tod", "vol", "tick"),
        "regime_atr": ("regime", "exit"),
        "ema_regime": ("direction", "regime"),
        "tick_ema": ("direction", "tick"),
        "ema_atr": ("direction", "exit"),
        "r2_atr": ("regime2", "exit"),
        "r2_tod": ("tod", "regime2"),
        "loosen_atr": ("exit",),
        "risk_overlays": ("risk",),
        "gate_matrix": ("perm", "tod", "regime2", "tick", "shock", "risk", "short_mult"),
        "lf_shock_sniper": ("direction", "shock"),
        "hf_timing_sniper": ("timing_profile",),
    }
    assert set(expected) == set(_combo_full_preset_axes())
    for preset, active_dims in expected.items():
        frozen = set(_combo_full_preset_freeze_dims(preset))
        assert tuple(dim for dim in _COMBO_FULL_CARTESIAN_DIM_ORDER if dim not in frozen) == active_dims


def test_frontier_is_runtime_pruning_not_a_report_only_axis() -> None:
    assert "frontier" not in _AXIS_CHOICES
    assert "stage_frontier" in _SWEEP_RUNTIME_POLICY


def test_axis_fingerprint_covers_complete_signal_identity() -> None:
    common = {
        "symbol": "SLV",
        "start": date(2025, 1, 8),
        "end": date(2025, 1, 10),
        "bar_size": "15 mins",
        "use_rth": False,
        "cache_dir": Path("db"),
        "offline": True,
        "filters": None,
    }

    assert _axis_dimension_fingerprint(
        _bundle_base(entry_signal="ema", **common)
    ) != _axis_dimension_fingerprint(_bundle_base(entry_signal="orb", **common))


def test_sweep_context_owns_every_causal_bar_tape() -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.start_dt = datetime(2025, 1, 8)
    runtime.end_dt = datetime(2025, 1, 10, 23, 59)
    loaded: dict[str, list[str]] = {}
    runtime._bars_cached = lambda bar_size: loaded.setdefault(
        str(bar_size), [str(bar_size)]
    )
    runtime._tick_bars_for = lambda _cfg: ["tick"]
    cfg = _bundle_base(
        symbol="SLV",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size="15 mins",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
    )
    cfg = replace(
        cfg,
        strategy=replace(
            cfg.strategy,
            regime2_mode="supertrend",
            regime2_bar_size="30 mins",
            regime2_bear_hard_mode="supertrend",
            regime2_bear_hard_bar_size="1 day",
            spot_exec_bar_size="1 min",
            tick_gate_mode="raschke",
        ),
    )

    context = runtime._context_bars_for_cfg(cfg=cfg, bars=["signal"])

    assert dict(context.items()) == {
        "signal": ["signal"],
        "regime": ["4 hours"],
        "regime2": ["30 mins"],
        "regime2_bear_hard": ["1 day"],
        "tick": ["tick"],
        "exec": ["1 min"],
    }


def test_sweep_window_signature_covers_every_causal_bar_role() -> None:
    kinds = (
        "signal",
        "regime",
        "regime2",
        "regime2_bear_hard",
        "tick",
        "exec",
    )
    base = tuple((kind, (1, "first", "last", "base")) for kind in kinds)
    signatures = {_window_signature(context_sig=base)}
    for changed_kind in kinds:
        signatures.add(
            _window_signature(
                context_sig=tuple(
                    (kind, (1, "first", "last", "changed" if kind == changed_kind else "base"))
                    for kind in kinds
                )
            )
        )

    assert len(signatures) == len(kinds) + 1


def test_sweep_evaluation_hands_every_causal_tape_to_engine(monkeypatch) -> None:
    runtime = object.__new__(SpotSweepRuntime)
    context = SpotContextBars(
        signal_bars=["signal"],
        regime_bars=["regime"],
        regime2_bars=["regime2"],
        regime2_bear_hard_bars=["hard"],
        tick_bars=["tick"],
        exec_bars=["exec"],
    )
    runtime.run_calls_total = 0
    runtime.run_cfg_fingerprint_cache = {}
    runtime.run_cfg_cache = {}
    runtime._RUN_CFG_CACHE_MISS = object()
    runtime._context_bars_for_cfg = lambda **_kwargs: context
    runtime._run_cfg_cache_coords = lambda **_kwargs: (
        (("signal", (1, None, None, "revision")),),
        ("strategy", "axis", "window"),
        "axis",
        "persistent",
    )
    runtime._run_cfg_persistent_get = lambda **_kwargs: runtime._RUN_CFG_CACHE_MISS
    runtime._run_cfg_persistent_set = lambda **_kwargs: None
    runtime._run_cfg_dimension_index_set = lambda **_kwargs: None
    runtime._axis_progress_record = lambda **_kwargs: None
    runtime.run_cfg_persistent_writes = 0
    runtime.run_min_trades = 0
    runtime.meta = ContractMeta("SLV", "SMART", 1.0, 0.01)
    captured: dict[str, object] = {}

    def _summary(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return SummaryStats(1, 1, 0, 1.0, 10.0, 0.1, 10.0, 0.0, 1.0, 0.01, 1.0)

    monkeypatch.setattr(sweep_evaluation, "_run_spot_backtest_summary", _summary)
    cfg = _bundle_base(
        symbol="SLV",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size="15 mins",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
    )

    assert runtime._run_cfg(cfg=cfg) is not None
    assert captured["args"][1] == context.signal_bars
    assert captured["kwargs"] == {
        "regime_bars": context.regime_bars,
        "regime2_bars": context.regime2_bars,
        "regime2_bear_hard_bars": context.regime2_bear_hard_bars,
        "tick_bars": context.tick_bars,
        "exec_bars": context.exec_bars,
        "prepared_series_pack": None,
        "progress_callback": None,
    }


def test_combo_full_progress_uses_executed_profile_space(monkeypatch) -> None:
    monkeypatch.delenv("TB_HF_TIMING_SNIPER_BRIDGE", raising=False)
    runtime = object.__new__(SpotSweepRuntime)
    runtime.args = SimpleNamespace(
        combo_full_cartesian_stage=None,
        combo_full_include_tick=False,
        combo_full_preset="profile",
        risk_overlays_skip_pop=False,
    )
    runtime.signal_bar_size = "1 hour"
    runtime._base_bundle = lambda *, bar_size, filters: _bundle_base(
        symbol="SLV",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size=bar_size,
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=filters,
    )

    space = runtime._combo_full_context("profile")
    assert space.total == 32
    assert space.dimension_signature == "37c19ac8bb039900f2c99245f4d3aceea5c86c29"
    for rank in (0, 1, 31):
        assert space.rank(space.indices(rank)) == rank
        _cfg, _note, meta = space.plan_item_from_rank(rank)
        assert meta["_mr_rank"] == rank
    assert runtime._axis_total_hint("combo_full") == 33
    assert runtime._axis_total_hint("regime") == 600
    assert runtime._axis_total_hint("regime2") == 300
    assert runtime._combo_full_context("full").dimension_signature == runtime._combo_full_context("").dimension_signature
    lf_space = runtime._combo_full_context("lf_shock_sniper")
    assert lf_space.total == 10
    assert lf_space.dimension_signature == "c69ba2ad76b43accf1c11f352f54f659a3309218"


def test_cfg_payload_round_trip_preserves_backtest_timeframe() -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.meta = ContractMeta(
        symbol="SLV", exchange="SMART", multiplier=1.0, min_tick=0.01
    )
    runtime.start = date(2025, 1, 8)
    runtime.end = date(2025, 1, 10)
    runtime.signal_bar_size = "1 hour"
    runtime.use_rth = False
    runtime.cache_dir = Path("db")
    runtime.offline = True
    cfg = _bundle_base(
        symbol="SLV",
        start=runtime.start,
        end=runtime.end,
        bar_size="15 mins",
        use_rth=True,
        cache_dir=runtime.cache_dir,
        offline=True,
        filters=None,
        entry_signal="orb",
    )

    payload = runtime._compact_parallel_payload_cfg_refs(
        {"cfgs": [runtime._encode_cfg_payload(cfg)]}
    )
    decoded = runtime._decode_cfg_payload(
        payload["cfgs"][0],
        cfg_catalog=runtime._cfg_catalog_from_payload(payload),
    )

    assert decoded is not None
    restored, _note = decoded
    assert restored.backtest.start == date(2025, 1, 8)
    assert restored.backtest.end == date(2025, 1, 10)
    assert restored.backtest.bar_size == "15 mins"
    assert restored.backtest.use_rth is True


def test_cfg_grid_workers_keep_parent_axis_and_internal_stage() -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.args = SimpleNamespace(cfg_stage=None, axis="combo_full")
    runtime.axis_progress_state = {"active": False}
    runtime.signal_bar_size = "15 mins"
    runtime.use_rth = False
    runtime.jobs = 4
    runtime.run_min_trades = 12
    runtime.start = date(2025, 1, 8)
    runtime.end = date(2025, 1, 10)
    runtime.cache_dir = Path("db")
    runtime.offline = True
    runtime.meta = ContractMeta("SLV", "SMART", 1.0, 0.01)
    runtime._bars_cached = lambda _bar_size: []
    captured: dict[str, object] = {}
    runtime._run_parallel_stage_with_payload = (
        lambda **kwargs: captured.update(kwargs) or {}
    )

    def run_stage(**kwargs):
        kwargs["parallel_payloads_builder"](kwargs["serial_plan"])
        return len(kwargs["serial_plan"])

    runtime._run_stage_cfg_rows = run_stage
    cfg = _bundle_base(
        symbol="SLV",
        start=runtime.start,
        end=runtime.end,
        bar_size=runtime.signal_bar_size,
        use_rth=False,
        cache_dir=runtime.cache_dir,
        offline=True,
        filters=None,
    )

    tested = runtime._run_cfg_pairs_grid(
        axis_tag="combo_full_stability",
        cfg_pairs=[(cfg, "candidate")],
        rows=[],
    )

    assert tested == 1
    assert captured["axis_name"] == "combo_full"
    assert captured["stage_label"] == "combo_full_stability"
    assert captured["strip_flags"] == ("--promote",)
    assert captured["stage_args"] == (
        "--start",
        "2025-01-08",
        "--end",
        "2025-01-10",
        "--base",
        "default",
    )


def test_orb_sweeps_share_one_mechanics_space() -> None:
    base = _bundle_base(
        symbol="SLV",
        start=date(2025, 1, 8),
        end=date(2025, 1, 10),
        bar_size="15 mins",
        use_rth=False,
        cache_dir=Path("db"),
        offline=True,
        filters=None,
        entry_signal="orb",
    )

    candidates = list(_orb_candidates(base, risk_rewards=(1.0, 2.0)))

    assert len(candidates) == 2 * 3 * 2 * 2 * 2
    assert len({key for key, _cfg, _note in candidates}) == len(candidates)
    first_key, first_cfg, first_note = candidates[0]
    assert first_key == ("09:30", 15, "rr", 1.0, None)
    assert first_cfg.strategy.filters is not None
    assert first_cfg.strategy.filters.entry_start_hour_et == 9
    assert first_cfg.strategy.filters.entry_end_hour_et == 16
    assert first_note == "ORB open=09:30 w=15 rr rr=1.0 tod=09-16 ET -"
    last_key, last_cfg, _last_note = candidates[-1]
    assert last_key == ("18:00", 60, "or_range", 2.0, 1.2)
    assert last_cfg.strategy.filters is not None
    assert last_cfg.strategy.filters.entry_start_hour_et == 18
    assert last_cfg.strategy.filters.entry_end_hour_et == 4


def test_ranked_rows_use_stable_identity_for_metric_ties() -> None:
    def score(row: dict) -> float:
        return float(row["score"])

    items = [("b", {"score": 1.0}, "B"), ("a", {"score": 1.0}, "A")]

    ranked = _rank_cfg_rows(
        items,
        scorers=[(score, 2)],
        key_fn=lambda key, _row, _note: key,
    )
    reversed_ranked = _rank_cfg_rows(
        list(reversed(items)),
        scorers=[(score, 2)],
        key_fn=lambda key, _row, _note: key,
    )

    assert [key for key, _row, _note in ranked] == ["a", "b"]
    assert reversed_ranked == ranked


def test_cartesian_manifest_uses_one_cache_scope_and_invalidates_summary(tmp_path: Path) -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.run_min_trades = 7
    runtime.run_cfg_persistent_enabled = True
    runtime.run_cfg_persistent_conn = None
    runtime.run_cfg_persistent_path = tmp_path / "sweeps.sqlite3"
    runtime.run_cfg_persistent_lock = threading.Lock()
    runtime.cartesian_rank_manifest_compact_seen = {}
    runtime.rank_dominance_manifest_applied_seen = set()
    runtime._CARTESIAN_RANK_STATUS_VALUES = frozenset(
        ("pending", "cached_hit", "evaluated", "dominated")
    )
    runtime.cartesian_rank_manifest_reads = 0
    runtime.cartesian_rank_manifest_writes = 0
    runtime.cartesian_rank_manifest_hits = 0
    runtime.cartesian_rank_manifest_compactions = 0
    runtime.cartesian_rank_manifest_pending_ttl_prunes = 0
    runtime.stage_unresolved_summary_reads = 0
    runtime.stage_unresolved_summary_writes = 0
    runtime.stage_unresolved_summary_hits = 0
    runtime.rank_dominance_stamp_reads = 0
    runtime.rank_dominance_stamp_hits = 0
    runtime.rank_dominance_manifest_applies = 0
    runtime.rank_dominance_stamp_compactions = 0
    runtime.rank_dominance_stamp_ttl_prunes = 0

    assert runtime._cartesian_rank_manifest_unresolved_ranges(
        stage_label="combo", window_signature="window", total=32
    ) == ((0, 31),)
    runtime._cartesian_rank_manifest_set_many(
        stage_label="combo",
        window_signature="window",
        rows=[(0, 31, "evaluated")],
    )
    assert runtime._cartesian_rank_manifest_unresolved_ranges(
        stage_label="combo", window_signature="window", total=32
    ) == ()

    conn = runtime._run_cfg_persistent_conn()
    assert conn is not None
    assert conn.execute(
        "SELECT DISTINCT stage_label FROM cartesian_rank_manifest"
    ).fetchall() == [("combo|m7",)]
    assert conn.execute(
        "SELECT DISTINCT stage_label FROM stage_unresolved_summary"
    ).fetchall() == [("combo|m7",)]
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stage_rank_manifest'"
    ).fetchall() == []


def test_stage_grid_materializes_fully_cached_plan() -> None:
    class Harness:
        offline = False
        run_calls_total = 0

        def _stage_partition_plan_by_cache(self, **kwargs):
            plan = list(kwargs["plan_all"])
            assert plan == [("a", "A"), ("b", "B"), ("c", "C")]
            self.run_calls_total += len(plan)
            hits = [
                ((key, "dimension", "window"), key, {"key": key}, note, None)
                for key, note in plan
            ]
            return [], hits, {}, len(plan)

        def _prune_pending_plan_by_manifest(self, **_kwargs):
            return [], {}, 0

        def _run_stage_serial(self, **kwargs):
            assert kwargs["plan"] == []
            return 0, []

    harness = Harness()
    rows: list[tuple[str, dict, str]] = []
    tested = SweepStages._run_stage_cfg_rows(
        harness,
        stage_label="axis",
        total=3,
        jobs_req=1,
        bars=[],
        report_every=0,
        on_row=lambda cfg, row, note: rows.append((cfg, row, note)),
        serial_plan=[("a", "A"), ("b", "B"), ("c", "C")],
        record_milestones=False,
    )

    assert tested == 3
    assert harness.run_calls_total == 3
    assert rows == [
        ("a", {"key": "a"}, "A"),
        ("b", {"key": "b"}, "B"),
        ("c", {"key": "c"}, "C"),
    ]
