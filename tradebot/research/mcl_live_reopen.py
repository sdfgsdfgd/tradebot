"""Exact maintenance-reopen authority for the selected MCL live runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from .live_calibration import LiveCalibrationLedger
from .mcl_live_transport import (
    _ET,
    _identity,
    _utc,
    latest_mcl_source_checkpoint,
    load_mcl_live_selection_from_mapping,
    refresh_mcl_source_if_due,
)
from .mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION


MCL_LIVE_SOURCE_AUTHORITY_FRESH = "fresh_finalized"
MCL_LIVE_SOURCE_AUTHORITY_REOPEN = "maintenance_reopen_reconciliation_only"
MCL_LIVE_REOPEN_BINDING_PATH = Path(
    "backtests/mcl/mcl_stage112_maintenance_reopen_runtime_binding.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_mcl_maintenance_reopen_selection(
    selection: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    selected = load_mcl_live_selection_from_mapping(selection)
    root = repository_root.resolve()
    path = root / MCL_LIVE_REOPEN_BINDING_PATH
    binding = json.loads(path.read_text())
    owners = binding.get("owners") if isinstance(binding, Mapping) else None
    if (
        selected["strategy_version"] != MCL_TWO_SPEED_SHOCK_VERSION
        or not isinstance(binding, Mapping)
        or binding.get("schema")
        != "mcl.stage112-maintenance-reopen-runtime-binding.v1"
        or binding.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or binding.get("verdict") != "PASS"
        or binding.get("submitted_orders") != 0
        or not all(binding.get("gates", {}).values())
        or not isinstance(owners, Mapping)
        or owners.get("transport_sha256")
        != _sha256(root / "tradebot/research/mcl_live_transport.py")
        or owners.get("reopen_sha256") != _sha256(Path(__file__))
        or owners.get("live_sha256")
        != _sha256(root / "tradebot/research/mcl_live.py")
        or owners.get("profitability_sha256")
        != _sha256(root / "tradebot/research/mcl_profitability.py")
        or owners.get("cli_sha256")
        != _sha256(root / "tradebot/research/mcl_live_cli.py")
    ):
        raise ValueError("MCL maintenance-reopen binding is invalid")
    body = dict(selected)
    body.pop("selection_id")
    body["evidence"] = {
        **dict(selected["evidence"]),
        "maintenance_reopen_runtime": {
            "path": MCL_LIVE_REOPEN_BINDING_PATH.as_posix(),
            "sha256": _sha256(path),
        },
    }
    return load_mcl_live_selection_from_mapping(
        {**body, "selection_id": _identity(body)}
    )


def _maintenance_reopen_prior_close(at: datetime) -> datetime | None:
    local = _utc(at).astimezone(_ET)
    if (local.hour, local.minute) != (18, 0) or local.weekday() not in {
        0,
        1,
        2,
        3,
        6,
    }:
        return None
    if local.weekday() == 6:
        local -= timedelta(days=2)
    return _utc(local.replace(hour=16, minute=59, second=0, microsecond=0))


async def refresh_mcl_live_source(
    ledger: LiveCalibrationLedger,
    *,
    client,
    selection: Mapping[str, object],
    observed_at: datetime,
) -> tuple[dict[str, object], str]:
    selected = load_mcl_live_selection_from_mapping(selection)
    now = _utc(observed_at)
    prior = (
        _maintenance_reopen_prior_close(now)
        if selected["strategy_version"] == MCL_TWO_SPEED_SHOCK_VERSION
        else None
    )
    if prior is not None:
        latest = latest_mcl_source_checkpoint(
            tuple(ledger.records()), selection_id=str(selected["selection_id"])
        )
        evidence = latest.get("evidence") if latest is not None else None
        source = evidence.get("source") if isinstance(evidence, Mapping) else None
        source_at = (
            _utc(source.get("latest_common_close_utc"))
            if isinstance(source, Mapping)
            and source.get("latest_common_close_utc") is not None
            else None
        )
        if source_at == prior:
            return latest, MCL_LIVE_SOURCE_AUTHORITY_REOPEN
    return (
        await refresh_mcl_source_if_due(
            ledger,
            client=client,
            selection=selected,
            observed_at=now,
        ),
        MCL_LIVE_SOURCE_AUTHORITY_FRESH,
    )
