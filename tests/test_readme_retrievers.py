from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_retrievers_module():
    path = _ROOT / "tradebot" / "ui" / "readme_retrievers.py"
    spec = importlib.util.spec_from_file_location("tradebot_ui_readme_retrievers_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RETRIEVERS = _load_retrievers_module()


def test_repo_current_champions_resolve_to_loadable_json() -> None:
    cases = [
        ("backtests/tqqq/readme-lf.md", _RETRIEVERS.extract_current_tqqq_lf_json_path, "TQQQ LF"),
        ("backtests/tqqq/readme-hf.md", _RETRIEVERS.extract_current_tqqq_hf_json_path, "TQQQ HF"),
        ("backtests/slv/readme-lf.md", _RETRIEVERS.extract_current_slv_lf_json_path, "SLV LF"),
        ("backtests/slv/readme-hf.md", _RETRIEVERS.extract_current_slv_hf_json_path, "SLV HF"),
    ]

    for rel_readme_path, extractor, label in cases:
        readme_path = (_ROOT / rel_readme_path).resolve()
        assert readme_path.exists(), rel_readme_path

        readme_text = readme_path.read_text()
        version, rel_json_path = extractor(readme_text)
        assert version, f"{label}: unable to extract CURRENT version from {rel_readme_path}"
        assert rel_json_path, f"{label}: unable to extract CURRENT json path from {rel_readme_path}"

        rel_json_path = str(rel_json_path).strip()
        assert rel_json_path.startswith("backtests/"), f"{label}: unexpected CURRENT path {rel_json_path}"
        assert rel_json_path.endswith(".json"), f"{label}: unexpected CURRENT path {rel_json_path}"

        json_path = (_ROOT / rel_json_path).resolve()
        assert str(json_path).startswith(str(_ROOT.resolve())), f"{label}: path escapes repo root: {rel_json_path}"
        assert json_path.exists(), f"{label}: CURRENT points to missing artifact {rel_json_path}"

        payload = json.loads(json_path.read_text())
        assert isinstance(payload, dict), f"{label}: CURRENT json is not an object: {rel_json_path}"
        groups = payload.get("groups")
        assert isinstance(groups, list) and groups, f"{label}: CURRENT json has no groups: {rel_json_path}"
        assert any(
            isinstance(g, dict) and isinstance(g.get("entries"), list) and g.get("entries") for g in groups
        ), f"{label}: CURRENT json has no entries: {rel_json_path}"


def test_current_candidates_prefer_latest_heading_but_path_lookup_falls_back_to_existing_json() -> None:
    readme_text = """
## Current Champions

### CURRENT (v34) - loadable fallback
- Preset file: `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json`

#### CURRENT (v39) - newer crown whose json is absent in repo
- Preset file: `backtests/out/tqqq_exec5m_v39_missing.json`
"""
    candidates = _RETRIEVERS.extract_current_tqqq_lf_json_candidates(readme_text)
    assert candidates[0] == ("39", "backtests/out/tqqq_exec5m_v39_missing.json")
    assert _RETRIEVERS.extract_current_tqqq_lf_json_path(readme_text) == (
        "34",
        "backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json",
    )


def test_fuzzy_current_heading_with_version_still_extracts_hf_crown() -> None:
    readme_text = """
## Current Champions

### CURRENT MULTI-REGIME DETHRONE (v38-Asymmetric Crash Prearm Sovereignty)
- Preset file (UI loads this): `backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v38_asymmetricCrashPrearmSovereignty_20260316.json`
"""
    assert _RETRIEVERS.extract_current_tqqq_hf_json_path(readme_text) == (
        "38-Asymmetric Crash Prearm Sovereignty",
        "backtests/tqqq/archive/champion_history_20260301/tqqq_hf_champions_v38_asymmetricCrashPrearmSovereignty_20260316.json",
    )


def test_current_tqqq_hf_crown_preserves_signal_transport_fields() -> None:
    readme_path = (_ROOT / "backtests/tqqq/readme-hf.md").resolve()
    version, rel_json_path = _RETRIEVERS.extract_current_tqqq_hf_json_path(readme_path.read_text())
    assert isinstance(version, str) and version.strip()
    assert rel_json_path

    payload = json.loads((_ROOT / str(rel_json_path)).read_text())
    strategy = payload["groups"][0]["entries"][0]["strategy"]
    assert strategy.get("signal_bar_size") == "5 mins"
    assert strategy.get("signal_use_rth") is True
