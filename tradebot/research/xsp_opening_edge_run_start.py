"""Immutable run-start recovery for XSP opening-edge observers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta, timezone

from ..engines.market import xsp_trading_date
from ..time_utils import ET_ZONE, to_et


XSP_OPENING_EDGE_V2_TRANSPORT_VERSION = "xsp.opening-edge-v2-spy-transport.v1"


def next_xsp_v2_run_start(observed_at: datetime) -> datetime:
    """Choose the next untouched GTH boundary; never backfill a live run."""

    observed_et = to_et(observed_at)
    for offset in range(1, 9):
        trading_day = observed_et.date() + timedelta(days=offset)
        candidate = datetime.combine(
            trading_day - timedelta(days=1),
            time(20, 15),
            tzinfo=ET_ZONE,
        )
        if candidate > observed_et and xsp_trading_date(candidate) == trading_day:
            return candidate.astimezone(timezone.utc)
    raise ValueError("unable to resolve next XSP GTH run start")


def xsp_opening_edge_v2_run_start(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at: datetime,
    strategy_version: str = XSP_OPENING_EDGE_V2_TRANSPORT_VERSION,
) -> datetime:
    """Recover one frozen v2 start, or choose the next untouched GTH boundary."""

    starts = {
        str(evidence["run_started_at_utc"])
        for row in records
        if row.get("kind") == "checkpoint"
        and row.get("strategy_version") == strategy_version
        and isinstance((evidence := row.get("evidence")), Mapping)
        and evidence.get("run_started_at_utc")
    }
    if len(starts) > 1:
        raise ValueError("Opening Edge v2 observer run start drift")
    if starts:
        return datetime.fromisoformat(starts.pop().replace("Z", "+00:00"))
    return next_xsp_v2_run_start(observed_at)
