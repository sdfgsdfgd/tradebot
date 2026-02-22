from __future__ import annotations

import importlib.util
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


def test_extract_current_slv_lf_json_path() -> None:
    readme = (_ROOT / "backtests" / "slv" / "README.md").read_text()
    version, path = _RETRIEVERS.extract_current_slv_lf_json_path(readme)
    assert version == "31.2"
    assert path == "backtests/slv/slv_v31_2_singlepos_parity_eval_20260212_top1.json"


def test_extract_current_slv_hf_json_path() -> None:
    readme = (_ROOT / "backtests" / "slv" / "readme-hf.md").read_text()
    version, path = _RETRIEVERS.extract_current_slv_hf_json_path(readme)
    assert version == "28-exception-ddshock-lb10-on10-off5-depth1p25pp-streak1-shortmult0p028"
    assert (
        path
        == "backtests/slv/archive/champion_history_20260214/slv_hf_champions_v28_exception_ddshock_lb10_on10_off5_depth1p25pp_streak1_shortmult0p028_20260222.json"
    )
