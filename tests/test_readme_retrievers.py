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
        _, rel_json_path = extractor(readme_text)
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
