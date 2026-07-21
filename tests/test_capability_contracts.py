from __future__ import annotations

import ast
import configparser
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
LEDGERS = TESTS / "ledgers"
EXPECTED_LAYERS = {
    "unit",
    "integration-replay",
    "integration-provider",
    "e2e-live",
    "benchmark",
}
PREFIX_LAYERS = {
    "unit.": "unit",
    "integration.replay.": "integration-replay",
    "integration.provider.": "integration-provider",
    "e2e.live.": "e2e-live",
    "benchmark.": "benchmark",
}


def _load(name: str) -> dict:
    return json.loads((LEDGERS / name).read_text(encoding="utf-8"))


def _pytest_nodes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            nodes.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    nodes.add(f"{node.name}::{child.name}")
    return nodes


def _reference_resolves(reference: str) -> bool:
    relative, separator, raw_node = reference.partition("::")
    path = ROOT / relative
    if not path.exists():
        return False
    if not separator:
        return True
    node = raw_node.split("[", 1)[0]
    return node in _pytest_nodes(path)


def _expected_layer(contract_id: str) -> str:
    for prefix, layer in PREFIX_LAYERS.items():
        if contract_id.startswith(prefix):
            return layer
    raise AssertionError(f"unknown contract ID layer prefix: {contract_id}")


def test_contracts_are_minimal_mecc_and_resolvable() -> None:
    schema = _load("capability_schema.json")
    doc = _load("capability_contracts.json")
    required = set(schema["required"])
    subsystems = set(schema["properties"]["subsystem"]["enum"])
    layers = set(schema["properties"]["layer"]["enum"])
    statuses = set(schema["properties"]["status"]["enum"])

    assert layers == EXPECTED_LAYERS
    assert doc["contract_schema"] == "./capability_schema.json"
    assert {entry["id"] for entry in doc["subsystem_taxonomy"]} == subsystems

    ids: set[str] = set()
    used_subsystems: set[str] = set()
    used_layers: set[str] = set()
    for contract in doc["contracts"]:
        assert set(contract) == required, contract["id"]
        assert re.fullmatch(r"[a-z0-9_.-]+", contract["id"])
        assert contract["id"] not in ids
        ids.add(contract["id"])

        assert contract["subsystem"] in subsystems
        assert contract["layer"] == _expected_layer(contract["id"])
        assert contract["status"] in statuses
        assert all(
            isinstance(contract[key], str) and contract[key].strip()
            for key in ("phase", "capability", "next")
        )
        for key in ("fixtures", "evidence", "owner_surface", "oracle"):
            assert contract[key]
            assert all(isinstance(item, str) and item.strip() for item in contract[key])

        assert all(_reference_resolves(ref) for ref in contract["evidence"]), contract["id"]
        assert all(_reference_resolves(ref) for ref in contract["owner_surface"]), contract["id"]
        if ".future." in contract["id"]:
            assert contract["status"] in {"planned", "gap", "wip"}

        used_subsystems.add(contract["subsystem"])
        used_layers.add(contract["layer"])

    assert used_subsystems == subsystems
    assert used_layers == EXPECTED_LAYERS


def test_every_test_has_one_owner_and_evidence() -> None:
    doc = _load("capability_contracts.json")
    discovered = {
        path.relative_to(ROOT).as_posix()
        for path in TESTS.rglob("test_*.py")
        if "__pycache__" not in path.parts
    }
    assert set(doc["test_ownership"]) == discovered
    assert set(doc["test_ownership"].values()) <= {
        entry["id"] for entry in doc["subsystem_taxonomy"]
    }

    evidenced = {
        ref.split("::", 1)[0]
        for contract in doc["contracts"]
        for ref in contract["evidence"]
        if ref.startswith("tests/")
        and Path(ref.split("::", 1)[0]).name.startswith("test_")
    }
    assert discovered <= evidenced


def test_every_live_test_has_exact_evidence_and_marker() -> None:
    doc = _load("capability_contracts.json")
    exact: dict[str, set[str]] = {}
    for contract in doc["contracts"]:
        for ref in contract["evidence"]:
            path, separator, node = ref.partition("::")
            if separator and path.startswith("tests/live/"):
                exact.setdefault(path, set()).add(node.split("[", 1)[0])

    discovered: dict[str, set[str]] = {}
    for path in (TESTS / "live").rglob("test_*.py"):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        assert "pytest.mark.live" in source, relative
        discovered[relative] = _pytest_nodes(path)

    assert discovered
    assert {
        path: sorted(nodes - exact.get(path, set()))
        for path, nodes in discovered.items()
        if nodes - exact.get(path, set())
    } == {}
    assert {
        path: sorted(nodes - discovered.get(path, set()))
        for path, nodes in exact.items()
        if nodes - discovered.get(path, set())
    } == {}


def test_readme_crosswalk_and_live_default() -> None:
    doc = _load("capability_contracts.json")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert [contract["id"] for contract in doc["contracts"] if contract["id"] not in readme] == []

    config = configparser.ConfigParser()
    config.read(ROOT / "pytest.ini")
    assert "not live" in config["pytest"].get("addopts", "")
    assert "live" in config["pytest"].get("markers", "")


def test_covered_live_contracts_require_exact_live_evidence() -> None:
    for contract in _load("capability_contracts.json")["contracts"]:
        if contract["layer"] == "e2e-live" and contract["status"] == "covered":
            assert contract["evidence"]
            assert all(
                ref.startswith("tests/live/") and "::" in ref
                for ref in contract["evidence"]
            )
