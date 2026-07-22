from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.spot.champions import (
    current_champion_candidates,
    discover_current_champions,
)
from tradebot.research.spot_sweeps.cli import parse_spot_sweep_args
from tradebot.research.spot_sweeps.milestones import (
    load_current_champion_milestones,
)


_ROOT = Path(__file__).resolve().parents[1]


def test_repo_current_champions_resolve_to_loadable_json() -> None:
    refs = {
        (ref.symbol, ref.track): ref
        for ref in discover_current_champions(root=_ROOT)
    }
    for key in (("TQQQ", "LF"), ("TQQQ", "HF"), ("SLV", "LF"), ("SLV", "HF")):
        ref = refs[key]
        assert ref.version, f"{key}: unable to extract CURRENT version"
        assert ref.artifact_path.is_relative_to(_ROOT)
        payload = json.loads(ref.artifact_path.read_text())
        groups = payload.get("groups")
        assert isinstance(groups, list) and groups, f"{key}: CURRENT JSON has no groups"
        assert any(
            isinstance(group, dict)
            and isinstance(group.get("entries"), list)
            and group.get("entries")
            for group in groups
        ), f"{key}: CURRENT JSON has no entries"


def test_current_candidates_prefer_latest_heading_but_discovery_falls_back(
    tmp_path: Path,
) -> None:
    backtests = tmp_path / "backtests" / "tqqq"
    backtests.mkdir(parents=True)
    fallback = tmp_path / "backtests" / "out" / "tqqq_v34.json"
    fallback.parent.mkdir()
    fallback.write_text('{"groups": []}')
    readme_text = """
## Current Champions

### CURRENT (v34) - loadable fallback
- Preset file: `backtests/out/tqqq_v34.json`

#### CURRENT (v39) - newer crown whose json is absent
- Preset file: `backtests/out/tqqq_v39_missing.json`
"""
    (backtests / "readme-lf.md").write_text(readme_text)

    candidates = current_champion_candidates(readme_text)
    assert candidates[0] == ("39", "backtests/out/tqqq_v39_missing.json")
    assert discover_current_champions(root=tmp_path)[0].version == "34"
    assert discover_current_champions(root=tmp_path)[0].used_fallback is True


def test_fuzzy_current_heading_with_version_still_extracts_hf_crown() -> None:
    readme_text = """
## Current Champions

### CURRENT MULTI-REGIME DETHRONE (v38-Asymmetric Crash Prearm Sovereignty)
- Preset file (UI loads this): `backtests/tqqq/champion_v38.json`
"""
    assert current_champion_candidates(readme_text) == [
        (
            "38-Asymmetric Crash Prearm Sovereignty",
            "backtests/tqqq/champion_v38.json",
        )
    ]


def test_current_tqqq_hf_crown_preserves_signal_transport_fields() -> None:
    ref = next(
        ref
        for ref in discover_current_champions(root=_ROOT)
        if (ref.symbol, ref.track) == ("TQQQ", "HF")
    )
    payload = json.loads(ref.artifact_path.read_text())
    strategy = payload["groups"][0]["entries"][0]["strategy"]
    assert strategy.get("signal_bar_size") == "5 mins"
    assert strategy.get("signal_use_rth") is True


def test_spot_sweep_defaults_to_neutral_base() -> None:
    args = parse_spot_sweep_args([])
    assert args.base == "default"
    assert args.track == "auto"


def test_transport_uniquely_selects_tqqq_hf_crown() -> None:
    payload, track, _warnings = load_current_champion_milestones(
        symbol="TQQQ",
        signal_bar_size="5 mins",
        use_rth=True,
        track="auto",
        prefer_realism=True,
    )
    assert track == "HF"
    assert {group["_track"] for group in payload["groups"]} == {"HF"}


def test_shared_slv_transport_requires_explicit_track() -> None:
    with pytest.raises(ValueError, match="ambiguous across HF, LF"):
        load_current_champion_milestones(
            symbol="SLV",
            signal_bar_size="10 mins",
            use_rth=False,
            track="auto",
            prefer_realism=True,
        )

    payload, track, _warnings = load_current_champion_milestones(
        symbol="SLV",
        signal_bar_size="10 mins",
        use_rth=False,
        track="lf",
        prefer_realism=True,
    )
    assert track == "LF"
    assert {group["_track"] for group in payload["groups"]} == {"LF"}
