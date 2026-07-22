from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRADEBOT = ROOT / "tradebot"
MAX_MODULE_LINES = 1_000

# Shrink this set in the same commit that brings an owner below the limit.
OVERSIZED_MODULE_DEBT = {
    "tradebot/backtest/cache_ops_lib.py",
    "tradebot/backtest/engine.py",
    "tradebot/backtest/run_backtests_spot_sweeps.py",
    "tradebot/client.py",
    "tradebot/spot/graph.py",
    "tradebot/spot/policy.py",
    "tradebot/spot_engine.py",
    "tradebot/ui/app.py",
    "tradebot/ui/bot.py",
    "tradebot/ui/bot_order_builder.py",
    "tradebot/ui/bot_signal_runtime.py",
}

# Existing ownership inversions only. New inversions and stale exemptions fail.
BACKTEST_IMPORT_DEBT = {
    ("tradebot/client.py", "tradebot.backtest.trading_calendar"),
    ("tradebot/ui/bot.py", "tradebot.backtest.spot_codec"),
    ("tradebot/ui/bot.py", "tradebot.backtest.trading_calendar"),
}


def _production_modules() -> list[Path]:
    return sorted(TRADEBOT.rglob("*.py"))


def _relative_module(path: Path) -> tuple[str, ...]:
    return tuple(path.relative_to(ROOT).with_suffix("").parts)


def _import_name(source: Path, node: ast.ImportFrom) -> str:
    module = tuple((node.module or "").split(".")) if node.module else ()
    if not node.level:
        return ".".join(module)
    source_module = _relative_module(source)
    package = source_module[:-1]
    keep = max(0, len(package) - node.level + 1)
    return ".".join((*package[:keep], *module))


def test_oversized_module_debt_only_shrinks() -> None:
    oversized = {
        path.relative_to(ROOT).as_posix()
        for path in _production_modules()
        if len(path.read_text().splitlines()) > MAX_MODULE_LINES
    }
    assert oversized == OVERSIZED_MODULE_DEBT, (
        f"Oversized-module debt changed: unexpected={sorted(oversized - OVERSIZED_MODULE_DEBT)}, "
        f"resolved_but_still_allowlisted={sorted(OVERSIZED_MODULE_DEBT - oversized)}"
    )


def test_backtest_dependency_debt_only_shrinks() -> None:
    violations: set[tuple[str, str]] = set()
    for source in _production_modules():
        source_rel = source.relative_to(ROOT).as_posix()
        # Research is the one production family allowed to depend on backtest adapters.
        if source_rel.startswith(("tradebot/backtest/", "tradebot/research/")):
            continue
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = _import_name(source, node)
                if imported == "tradebot.backtest" or imported.startswith("tradebot.backtest."):
                    violations.add((source_rel, imported))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tradebot.backtest" or alias.name.startswith("tradebot.backtest."):
                        violations.add((source_rel, alias.name))
    assert violations == BACKTEST_IMPORT_DEBT, (
        f"Backtest ownership debt changed: unexpected={sorted(violations - BACKTEST_IMPORT_DEBT)}, "
        f"resolved_but_still_allowlisted={sorted(BACKTEST_IMPORT_DEBT - violations)}"
    )
