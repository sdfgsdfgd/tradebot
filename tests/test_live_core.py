from __future__ import annotations

from copy import deepcopy

import pytest

from tradebot.live.core import (
    CORE_BOUNDARIES,
    build_core_pool,
    build_core_profile,
    validate_core_pool,
)


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "1" * 64,
        "symbol": "MCL",
        "track": "HF",
        "version": "112",
        "declaration_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "strategy_id": "mcl.two-speed.v1",
        "run_id": "4" * 64,
        "machine_authority": True,
    }


def _graduation(target: str = "five_session_week") -> dict[str, object]:
    return {
        "verdict": "PROMOTE",
        "target": target,
        "cutoff_utc": "2026-08-14T21:00:00+00:00",
        "receipt_id": "5" * 64,
    }


def test_core_requires_the_final_profitability_milestone_and_grants_no_authority() -> None:
    with pytest.raises(ValueError, match="five-session"):
        build_core_profile(_candidate(), _graduation("48h"))

    profile = build_core_profile(_candidate(), _graduation())
    pool = build_core_pool((profile,))

    assert pool["members"] == [profile]
    assert profile["boundaries"] == CORE_BOUNDARIES
    assert pool["boundaries"] == CORE_BOUNDARIES
    assert "timer" not in profile
    assert "capital" not in profile


def test_core_pool_is_canonical_content_addressed_and_tamper_evident() -> None:
    profile = build_core_profile(_candidate(), _graduation())
    pool = build_core_pool((profile,))

    assert validate_core_pool(pool) == pool
    tampered = deepcopy(pool)
    tampered["members"][0]["candidate"]["symbol"] = "CL"
    with pytest.raises(ValueError, match="invalid Core strategy"):
        validate_core_pool(tampered)
