from __future__ import annotations

import asyncio
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from ib_insync import Contract
import pytest

from tradebot.client import BrokerOrderPreview
from tradebot.research.live_calibration import LiveCalibrationLedger
from tradebot.research.xsp_execution_observer import (
    XSP_V2_ETF_EXECUTION_OBSERVER_VERSION,
    advance_xsp_v2_etf_execution_observer,
    load_xsp_v2_etf_transport,
    xsp_v2_position_state,
)


NOW = datetime(2026, 7, 28, 0, 42, tzinfo=timezone.utc)


def _position(
    direction: str,
    *,
    entry_time: datetime = NOW - timedelta(minutes=5),
) -> dict[str, object]:
    return {
        "lane": "gth",
        "direction": direction,
        "entry_time": entry_time.isoformat(),
        "exit_time": NOW.isoformat(),
        "trading_date": "2026-07-28",
        "entry_price": 742.0,
        "exit_price": 742.1,
        "exit_reason": "end",
        "gross_points": 0.1,
        "cost_points": 0.1,
        "net_points": 0.0,
    }


def _paired(
    position: dict[str, object] | None,
    *,
    broker_position: dict[str, object] | None = None,
) -> dict[str, object]:
    def profile(value):
        return {
            "run_started_at_utc": "2026-07-28T00:15:00+00:00",
            "latest_position": value,
        }

    return {
        "crown_config_fingerprint": "crown-config",
        "profiles": {
            "research": profile(position),
            "broker": profile(
                position if broker_position is None else broker_position
            ),
        },
    }


def _source(
    position: dict[str, object] | None,
    *,
    checkpoint: str = "source-checkpoint",
) -> dict[str, object]:
    return {
        "evaluation_status": "EVALUATED",
        "checkpoint_id": checkpoint,
        "session": "GTH",
        "paired_equity": _paired(position),
    }


class _Client:
    def __init__(self) -> None:
        self.previews = []
        self.symbols = []

    async def qualify_proxy_contracts(self, contract):
        symbol = str(contract.symbol)
        self.symbols.append(symbol)
        return [
            Contract(
                conId={"UPRO": 61228752, "SPXU": 828937771}[symbol],
                symbol=symbol,
                secType="STK",
                exchange="SMART",
                currency="USD",
            )
        ]

    async def ensure_ticker(self, contract, *, owner):
        assert owner == "xsp-v2-etf-execution-observer"
        price = 100.0 if contract.symbol == "UPRO" else 25.0
        routed = Contract(
            conId=contract.conId,
            symbol=contract.symbol,
            secType="STK",
            exchange="OVERNIGHT",
            currency="USD",
        )
        return SimpleNamespace(
            contract=routed,
            bid=price,
            ask=price + 0.02,
            last=price + 0.01,
            close=price - 0.05,
            bidSize=100,
            askSize=200,
            lastSize=10,
            volume=1000,
            marketDataType=1,
            time=NOW,
            tbTopQuoteUpdatedMono=time.monotonic(),
            tbQuoteSource="stream",
            tbQuoteAsOf=NOW.isoformat(),
        )

    async def preview_limit_order(
        self,
        contract,
        action,
        quantity,
        limit_price,
        outside_rth,
    ):
        self.previews.append(
            (contract.symbol, action, quantity, limit_price, outside_rth)
        )
        return BrokerOrderPreview(
            status="PreSubmitted",
            commission=1.0,
            min_commission=1.0,
            max_commission=1.0,
            commission_currency="USD",
        )


def test_frozen_upro_spxu_transport_remains_hold() -> None:
    transport = load_xsp_v2_etf_transport()

    assert transport.verdict == "HOLD"
    assert transport.candidate["up"] == {"symbol": "UPRO", "quantity": 9}
    assert transport.candidate["down"] == {"symbol": "SPXU", "quantity": 31}
    assert len(transport.gate_sha256) == len(transport.receipt_sha256) == 64


def test_frozen_transport_rejects_content_address_drift(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "backtests" / "xsp"
    target = tmp_path / "backtests" / "xsp"
    target.mkdir(parents=True)
    for name in (
        "opening_edge_v2_upro_spxu_preregistered_gate.json",
        "opening_edge_v2_upro_spxu_acceptance_receipt.json",
    ):
        shutil.copy2(source / name, target / name)
    gate = target / "opening_edge_v2_upro_spxu_preregistered_gate.json"
    gate.write_text(gate.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid frozen"):
        load_xsp_v2_etf_transport(root=tmp_path)


def test_execution_observer_records_only_position_state_changes(
    tmp_path: Path,
) -> None:
    ledger = LiveCalibrationLedger(tmp_path / "execution.jsonl")
    client = _Client()
    up = _position("up")

    first = asyncio.run(
        advance_xsp_v2_etf_execution_observer(
            ledger,
            client=client,
            source_receipt=_source(up),
            observed_at=NOW,
            recorded_at=NOW + timedelta(seconds=2),
        )
    )

    assert first["status"] == "EVALUATED"
    assert [(row["action"], row["symbol"], row["quantity"]) for row in first["legs"]] == [
        ("BUY", "UPRO", 9)
    ]
    [entry] = first["legs"]
    assert entry["contract"]["exchange"] == "OVERNIGHT"
    assert entry["quote_eligible"] is True
    assert entry["execution_eligible"] is False
    assert entry["quote_health"]["eligible"] is True
    assert entry["ladder"] == {
        "OPTIMISTIC": 100.0,
        "MID": 100.01,
        "AGGRESSIVE": 100.01,
        "CROSS": 100.02,
    }
    assert entry["what_if_preview"]["commission"] == 1.0
    assert entry["submitted_orders"] == 0
    assert client.previews == [("UPRO", "BUY", 9.0, 100.02, True)]
    [record] = list(ledger.records())
    assert record["strategy_version"] == XSP_V2_ETF_EXECUTION_OBSERVER_VERSION
    assert record["evidence"]["frozen_transport"]["verdict"] == "HOLD"
    assert record["evidence"]["frozen_transport"]["selected"] is False
    assert record["evidence"]["profitability_clock_started"] is False
    assert record["evidence"]["order_authority"] == "none"

    unchanged = asyncio.run(
        advance_xsp_v2_etf_execution_observer(
            ledger,
            client=client,
            source_receipt=_source(up, checkpoint="later-source"),
            observed_at=NOW + timedelta(minutes=5),
            recorded_at=NOW + timedelta(minutes=5, seconds=2),
        )
    )
    assert unchanged["status"] == "unchanged"
    assert len(list(ledger.records())) == 1
    assert client.symbols == ["UPRO"]

    down = _position("down", entry_time=NOW + timedelta(minutes=10))
    flipped = asyncio.run(
        advance_xsp_v2_etf_execution_observer(
            ledger,
            client=client,
            source_receipt=_source(down, checkpoint="flip-source"),
            observed_at=NOW + timedelta(minutes=10),
            recorded_at=NOW + timedelta(minutes=10, seconds=2),
        )
    )
    assert flipped["status"] == "EVALUATED"
    assert [
        (row["action"], row["symbol"], row["quantity"])
        for row in flipped["legs"]
    ] == [("SELL", "UPRO", 9), ("BUY", "SPXU", 31)]
    assert len(list(ledger.records())) == 2
    assert client.symbols == ["UPRO", "UPRO", "SPXU"]


def test_execution_observer_does_not_poll_a_flat_baseline(
    tmp_path: Path,
) -> None:
    class _NoBroker:
        async def qualify_proxy_contracts(self, _contract):
            raise AssertionError("flat baseline must not request market data")

    ledger = LiveCalibrationLedger(tmp_path / "flat.jsonl")
    receipt = asyncio.run(
        advance_xsp_v2_etf_execution_observer(
            ledger,
            client=_NoBroker(),
            source_receipt=_source(None),
            observed_at=NOW,
            recorded_at=NOW + timedelta(seconds=1),
        )
    )

    assert receipt["status"] == "unchanged"
    assert list(ledger.records()) == []


def test_execution_observer_fails_closed_on_profile_state_drift(
    tmp_path: Path,
) -> None:
    source = _source(_position("up"))
    source["paired_equity"] = _paired(
        _position("up"),
        broker_position=_position("down"),
    )

    with pytest.raises(ValueError, match="position state drift"):
        asyncio.run(
            advance_xsp_v2_etf_execution_observer(
                LiveCalibrationLedger(tmp_path / "drift.jsonl"),
                client=_Client(),
                source_receipt=source,
                observed_at=NOW,
                recorded_at=NOW + timedelta(seconds=1),
            )
        )


def test_position_state_allows_cost_adjusted_entry_price() -> None:
    research = _position("up")
    broker = {**research, "entry_price": 740.5}

    _run_key, state = xsp_v2_position_state(
        _paired(research, broker_position=broker)
    )

    assert state == {
        "lane": "gth",
        "direction": "up",
        "entry_time": research["entry_time"],
        "trading_date": "2026-07-28",
        "entry_price": 742.0,
    }


def test_execution_observer_skips_non_evaluated_sources(
    tmp_path: Path,
) -> None:
    receipt = asyncio.run(
        advance_xsp_v2_etf_execution_observer(
            LiveCalibrationLedger(tmp_path / "stale.jsonl"),
            client=_Client(),
            source_receipt={
                "evaluation_status": "STALE_DATA",
                "paired_equity": _paired(_position("up")),
            },
            observed_at=NOW,
            recorded_at=NOW + timedelta(seconds=1),
        )
    )

    assert receipt["status"] == "source_not_evaluated"
