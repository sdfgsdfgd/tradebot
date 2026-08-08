"""Content-addressed contract for Gold Stage-76 runtime parity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


GOLD_RUNTIME_PARITY_PATH = Path(
    "backtests/gold/one_oz_regime_harmony_runtime_parity_fail_closed_20260808.json"
)
GOLD_RUNTIME_PARITY_SCHEMA = "gold.1oz-regime-harmony-runtime-parity.v2"
GOLD_RUNTIME_PARITY_AUTHORITY = (
    "immutable_signal_and_fail_closed_runtime_parity_only_"
    "no_selection_no_capital_no_orders"
)
GOLD_RUNTIME_PARITY_REQUIRED_GATES = (
    "machine_crown_identity",
    "shared_context_math",
    "full_three_year_ledger",
    "full_ten_year_ledger",
    "cold_replay_and_restart_identity",
    "flat_current_prefix",
    "fail_closed_source_quote_and_timing",
    "entry_spread_and_positive_size",
    "resting_limit_wide_book_freeze",
    "held_exit_authority_preserved",
    "cross_contract_default_semantics",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gold_runtime_parity(root: Path) -> dict[str, str]:
    path = root / GOLD_RUNTIME_PARITY_PATH
    receipt = json.loads(path.read_text())
    owners = receipt.get("owners")
    if (
        receipt.get("schema") != GOLD_RUNTIME_PARITY_SCHEMA
        or receipt.get("authority") != GOLD_RUNTIME_PARITY_AUTHORITY
        or receipt.get("verdict")
        != "SIGNAL_RUNTIME_PARITY_PASS_LIVE_TRANSPORT_HOLD"
        or not isinstance(owners, Mapping)
        or any(
            receipt.get("gates", {}).get(gate) != "PASS"
            for gate in GOLD_RUNTIME_PARITY_REQUIRED_GATES
        )
        or any(
            _sha256(root / str(row.get("path") or "")) != row.get("sha256")
            for row in owners.values()
            if isinstance(row, Mapping)
        )
    ):
        raise ValueError("gold runtime parity receipt is invalid")
    return {"path": GOLD_RUNTIME_PARITY_PATH.as_posix(), "sha256": _sha256(path)}
