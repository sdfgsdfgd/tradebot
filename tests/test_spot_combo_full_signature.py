from __future__ import annotations

from datetime import date
from pathlib import Path
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

    assert runtime._combo_full_context("profile").total == 32
    assert runtime._axis_total_hint("combo_full") == 33
