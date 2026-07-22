from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
import threading
from types import SimpleNamespace

from tradebot.backtest.data import ContractMeta
from tradebot.backtest.models import SummaryStats
from tradebot.backtest.spot_context import SpotContextBars
import tradebot.research.spot_sweeps.evaluation as sweep_evaluation
from tradebot.research.spot_sweeps.axes_hf import _orb_candidates
from tradebot.research.spot_sweeps.catalog import (
    _AXIS_CHOICES,
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
    _combo_full_preset_axes,
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
from tradebot.research.spot_sweeps.milestones import _rank_cfg_rows
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
    assert restored.backtest.bar_size == "15 mins"
    assert restored.backtest.use_rth is True


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
