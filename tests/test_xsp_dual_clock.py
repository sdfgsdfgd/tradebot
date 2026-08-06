from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradebot.research.xsp_context import xsp_execution_signal_context
from tradebot.research.xsp_dual_clock import (
    XSP_DUAL_CLOCK_PAIRED_SCHEMA,
    XSP_DUAL_CLOCK_SOURCE_VERSION,
    XSP_DUAL_CLOCK_TARGET_SCHEMA,
    XSP_DUAL_CLOCK_VERSION,
)
from tradebot.research.xsp_execution_observer import xsp_v2_position_state
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.xsp_live_transport import XSP_V3_TRANSPORT_PLAN_SCHEMA
from tradebot.research.xsp_live_transport_runtime import _checkpoint


def _bridge_target() -> dict[str, object]:
    context = {
        "schema": "xsp.execution-signal-context.v1",
        "lane": "rth",
        "direction": "down",
        "entry_time_utc": "2026-08-07T13:36:00+00:00",
        "signal_bar_ts": "2026-08-07T13:36:00+00:00",
        "decision_trace_fingerprint": "a" * 64,
        "control": {
            "source": "p009_opening_minute_acceptance",
            "volume_level": 20.0,
        },
        "directional_impulse": {
            "source": "joint_XSP_SPY_opening_transport",
            "direction": "down",
        },
        "market_state": {"owner": "opening_bridge"},
        "local_extrema": None,
    }
    return {
        "schema": XSP_DUAL_CLOCK_TARGET_SCHEMA,
        "lane": "rth",
        "direction": "down",
        "entry_time": context["entry_time_utc"],
        "trading_date": "2026-08-07",
        "entry_price": 630.25,
        "exit_reason": "end",
        "owner": "opening_bridge",
        "emission_id": "b" * 64,
        "bridge_fill_time_utc": "2026-08-07T13:37:00+00:00",
        "execution_signal_context": context,
        "order_authority": "none",
    }


def _paired(target: object) -> dict[str, object]:
    profile = {
        "run_started_at_utc": "2026-08-02T00:15:00+00:00",
        "latest_position": None,
    }
    return {
        "schema": XSP_DUAL_CLOCK_PAIRED_SCHEMA,
        "strategy_version": XSP_DUAL_CLOCK_SOURCE_VERSION,
        "crown_config_fingerprint": "c" * 64,
        "profiles": {"research": dict(profile), "broker": dict(profile)},
        "dual_clock_target": target,
    }


def test_p009_target_is_the_single_execution_state_and_context_owner() -> None:
    target = _bridge_target()
    run_key, state = xsp_v2_position_state(_paired(target))
    context = xsp_execution_signal_context(_paired(target))

    assert len(run_key) == 64
    assert state == {
        "lane": "rth",
        "direction": "down",
        "entry_time": "2026-08-07T13:36:00+00:00",
        "trading_date": "2026-08-07",
        "entry_price": 630.25,
    }
    assert context == target["execution_signal_context"]
    assert context["control"]["volume_level"] == 20.0


def test_p009_flat_target_overrides_stale_profile_state() -> None:
    paired = _paired(None)
    position = {
        "lane": "rth",
        "direction": "up",
        "entry_time": "2026-08-07T13:35:00+00:00",
        "trading_date": "2026-08-07",
        "entry_price": 630.0,
        "exit_reason": "end",
    }
    paired["profiles"]["research"]["latest_position"] = dict(position)
    paired["profiles"]["broker"]["latest_position"] = dict(position)

    _, state = xsp_v2_position_state(paired)

    assert state is None


def test_p009_context_mismatch_fails_closed() -> None:
    target = _bridge_target()
    target["execution_signal_context"] = deepcopy(
        target["execution_signal_context"]
    )
    target["execution_signal_context"]["direction"] = "up"

    with pytest.raises(ValueError, match="P-009 execution attribution"):
        xsp_execution_signal_context(_paired(target))


def test_p009_identity_is_explicit() -> None:
    assert XSP_DUAL_CLOCK_VERSION == (
        "xsp.opening-edge-v4-dual-clock-arbitration-p009.v1"
    )
    assert XSP_DUAL_CLOCK_SOURCE_VERSION == (
        "xsp.opening-edge-v4-dual-clock-source-p009.v1"
    )


def test_p009_execution_checkpoint_keeps_crown_attribution(tmp_path: Path) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "p009-execution.jsonl")
    observed = datetime(2026, 8, 7, 13, 36, tzinfo=timezone.utc)
    checkpoint = _checkpoint(
        ledger,
        strategy_id=XSP_DUAL_CLOCK_VERSION,
        selection_id="a" * 64,
        plan={
            "schema": XSP_V3_TRANSPORT_PLAN_SCHEMA,
            "transition_id": "b" * 64,
            "source_checkpoint_id": "c" * 64,
            "source_session": "RTH",
        },
        phase="STATE",
        order_ref="",
        observed_at=observed,
        preview=None,
        trade=None,
        submitted_orders=0,
    )

    assert checkpoint["strategy_id"] == XSP_DUAL_CLOCK_VERSION
    assert checkpoint["evidence"]["submitted_orders"] == 0
