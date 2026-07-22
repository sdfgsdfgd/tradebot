from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from types import SimpleNamespace

from tradebot.research.spot_sweeps.catalog import (
    _COMBO_FULL_CARTESIAN_DIM_ORDER,
    _COMBO_FULL_PAIR_DIM_VARIANT_SPECS,
)
from tradebot.research.spot_sweeps.fingerprints import (
    _combo_full_dimension_space_signature,
)
from tradebot.research.spot_sweeps.runtime import SpotSweepRuntime
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
    assert runtime._combo_full_context("full").dimension_signature == runtime._combo_full_context("").dimension_signature


def test_cartesian_manifest_uses_one_cache_scope_and_invalidates_summary(tmp_path: Path) -> None:
    runtime = object.__new__(SpotSweepRuntime)
    runtime.run_min_trades = 7
    runtime.run_cfg_persistent_enabled = True
    runtime.run_cfg_persistent_conn = None
    runtime.run_cfg_persistent_path = tmp_path / "sweeps.sqlite3"
    runtime.run_cfg_persistent_lock = threading.Lock()
    runtime.status_span_manifest_compact_seen = {}
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
