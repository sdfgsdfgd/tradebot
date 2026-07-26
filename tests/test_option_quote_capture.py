from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from tradebot.backtest.quotes import (
    QuoteContract,
    QuoteSnapshot,
    append_snapshot,
    iter_snapshot_payloads,
    iter_snapshots,
    make_chain_manifest,
    make_snapshot,
    option_implied_underlier,
    option_parity_observation,
    persist_chain_manifest,
    persist_quote_tape_receipt,
    quote_tape_receipt,
    repair_snapshot_tail,
    snapshot_quality,
)
from tradebot.backtest.tools.record_quotes import (
    CaptureCadence,
    RetainedOptionUniverse,
)
from tradebot.backtest.tools import record_quotes
from tradebot.config import IBKRConfig, auxiliary_client_config, auxiliary_client_id


def _quote(
    con_id: int,
    *,
    md_type: int = 1,
    quote_time: str | None = "2026-07-24T14:29:55+00:00",
    full_greeks: bool = False,
) -> QuoteContract:
    greeks = (0.2, 0.5, 0.02, 0.1, -0.05, 625.0) if full_greeks else (None,) * 6
    return QuoteContract(
        con_id=con_id,
        sec_type="OPT",
        symbol="XSP",
        local_symbol=f"XSP-{con_id}",
        exchange="SMART",
        currency="USD",
        bid=1.0,
        ask=1.1,
        market_data_type=md_type,
        quote_time=quote_time,
        model_iv=greeks[0],
        model_delta=greeks[1],
        model_gamma=greeks[2],
        model_vega=greeks[3],
        model_theta=greeks[4],
        model_under_price=greeks[5],
    )


def _snapshot(ts: str) -> QuoteSnapshot:
    return QuoteSnapshot(
        ts=ts,
        md_type=1,
        symbol="XSP",
        underlying=_quote(11004968),
        options=[_quote(1)],
        errors=[],
        chain_fingerprint="abc",
        target_expiry="20260724",
    )


def test_chain_manifest_is_canonical_and_content_addressed(tmp_path) -> None:
    manifest = make_chain_manifest(
        SimpleNamespace(
            conId=11004968,
            secType="IND",
            symbol="xsp",
            exchange="CBOE",
            currency="USD",
        ),
        SimpleNamespace(
            exchange="SMART",
            tradingClass="XSP",
            multiplier="100",
            expirations=("20260727", "20260724"),
            strikes=(626.0, 624.0, 625.0),
        ),
    )

    first = persist_chain_manifest(tmp_path, manifest)
    second = persist_chain_manifest(tmp_path, manifest)

    assert first == second
    [path] = list((tmp_path / "chains").glob("*.json"))
    payload = json.loads(path.read_text())
    assert payload["symbol"] == "XSP"
    assert payload["expirations"] == ["20260724", "20260727"]
    assert payload["strikes"] == [624.0, 625.0, 626.0]


def test_xsp_snapshot_records_exchange_session() -> None:
    contract = SimpleNamespace(
        conId=137851301,
        secType="IND",
        symbol="XSP",
        localSymbol="XSP",
        exchange="CBOE",
        currency="USD",
        lastTradeDateOrContractMonth="",
        strike=0.0,
        right="",
        tradingClass="",
        multiplier="",
        minTick=0.01,
    )
    ticker = SimpleNamespace(
        bid=740.0,
        ask=740.1,
        last=740.05,
        close=739.0,
        bidSize=1,
        askSize=1,
        lastSize=1,
        volume=None,
        modelGreeks=None,
        marketDataType=1,
        time=datetime(2026, 7, 24, 12, 34, tzinfo=timezone.utc),
    )

    snapshot = make_snapshot(
        symbol="XSP",
        md_type=1,
        underlying_contract=contract,
        underlying_ticker=ticker,
        option_contracts=[],
        option_tickers=[],
        ts=datetime(2026, 7, 24, 12, 34, tzinfo=timezone.utc),
    )

    assert snapshot.session == "GTH"
    assert snapshot.schema_version == 4


def test_snapshot_append_repairs_valid_and_partial_jsonl_tails(tmp_path) -> None:
    path = tmp_path / "2026-07-24.jsonl"
    append_snapshot(path, _snapshot("2026-07-24T14:30:00+00:00"))

    path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    append_snapshot(path, _snapshot("2026-07-24T14:31:00+00:00"))

    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    append_snapshot(path, _snapshot("2026-07-24T14:32:00+00:00"))

    assert [row["ts"] for row in iter_snapshot_payloads(path)] == [
        "2026-07-24T14:30:00+00:00",
        "2026-07-24T14:31:00+00:00",
        "2026-07-24T14:32:00+00:00",
    ]
    assert [snapshot.ts for snapshot in iter_snapshots(path)] == [
        "2026-07-24T14:30:00+00:00",
        "2026-07-24T14:31:00+00:00",
        "2026-07-24T14:32:00+00:00",
    ]
    assert path.read_bytes().endswith(b"\n")


def test_snapshot_writer_waits_for_shared_reader_lock(tmp_path) -> None:
    path = tmp_path / "2026-07-24.jsonl"
    append_snapshot(path, _snapshot("2026-07-24T14:30:00+00:00"))
    reader = iter_snapshots(path)
    assert next(reader).ts == "2026-07-24T14:30:00+00:00"

    started = threading.Event()
    finished = threading.Event()

    def write() -> None:
        started.set()
        append_snapshot(path, _snapshot("2026-07-24T14:35:00+00:00"))
        finished.set()

    writer = threading.Thread(target=write)
    writer.start()
    assert started.wait(1.0)
    assert not finished.wait(0.05)
    reader.close()
    assert finished.wait(1.0)
    writer.join()

    assert [snapshot.ts for snapshot in iter_snapshots(path)] == [
        "2026-07-24T14:30:00+00:00",
        "2026-07-24T14:35:00+00:00",
    ]


def test_explicit_tail_repair_makes_interrupted_tape_restart_readable(tmp_path) -> None:
    path = tmp_path / "2026-07-24.jsonl"
    append_snapshot(path, _snapshot("2026-07-24T14:30:00+00:00"))
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    assert repair_snapshot_tail(path) == len(b'{"partial":')
    assert [snapshot.ts for snapshot in iter_snapshots(path)] == [
        "2026-07-24T14:30:00+00:00"
    ]
    assert path.read_bytes().endswith(b"\n")


def test_expiry_selection_uses_the_xsp_trading_date() -> None:
    expirations = ("20260727", "20260728")

    assert (
        record_quotes._pick_expiry(
            expirations,
            0,
            0,
            0,
            as_of=date(2026, 7, 27),
        )
        == "20260727"
    )


def test_auxiliary_ibkr_clients_are_reserved_above_the_live_pool() -> None:
    config = SimpleNamespace(client_id_pool_end=899)

    assert auxiliary_client_id(config, 50) == 949
    assert auxiliary_client_id(config, 80) == 979
    assert auxiliary_client_id(config, 90) == 989


def test_auxiliary_ibkr_client_triplet_does_not_reuse_live_state() -> None:
    config = IBKRConfig(
        host="127.0.0.1",
        port=4001,
        client_id=500,
        proxy_client_id=501,
        account=None,
        refresh_sec=0.25,
        detail_refresh_sec=0.25,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=240.0,
        reconnect_slow_interval_sec=60.0,
        client_id_state_file="/tmp/live-client-ids.json",
    )

    isolated = auxiliary_client_config(config, 80)

    assert (isolated.client_id, isolated.proxy_client_id) == (979, 980)
    assert (isolated.client_id_pool_start, isolated.client_id_pool_end) == (979, 981)
    assert isolated.client_id_state_file == ""


def test_quote_tape_receipt_binds_rows_manifests_and_evidence(tmp_path) -> None:
    manifest = make_chain_manifest(
        SimpleNamespace(
            conId=11004968,
            secType="IND",
            symbol="XSP",
            exchange="CBOE",
            currency="USD",
        ),
        SimpleNamespace(
            exchange="SMART",
            tradingClass="XSP",
            multiplier="100",
            expirations=("20260724",),
            strikes=(624.0, 625.0),
        ),
    )
    fingerprint = persist_chain_manifest(tmp_path, manifest)
    path = tmp_path / "2026-07-24.jsonl"
    append_snapshot(
        path,
        replace(
            _snapshot("2026-07-24T14:30:00+00:00"),
            chain_fingerprint=fingerprint,
            session="RTH",
        ),
    )
    append_snapshot(
        path,
        replace(
            _snapshot("2026-07-24T14:35:00+00:00"),
            chain_fingerprint=fingerprint,
            session="RTH",
        ),
    )

    receipt = quote_tape_receipt(path)
    receipt_path = persist_quote_tape_receipt(path)

    assert receipt["tape"]["rows"] == 2
    assert receipt["integrity"] == {
        "newline_terminated": True,
        "strict_timestamp_order": True,
        "manifest_sha256": {fingerprint: fingerprint},
        "missing_manifests": [],
        "invalid_conids": 0,
    }
    assert receipt["evidence"]["sessions"] == {"RTH": 2}
    assert receipt_path.name == (f"{path.stem}.{receipt['tape']['sha256']}.json")
    assert json.loads(receipt_path.read_text()) == receipt


def test_absolute_cadence_resumes_without_drift_or_catchup_bursts() -> None:
    cadence = CaptureCadence.resume(
        300,
        now_mono=1_000.0,
        now_utc=datetime(2026, 7, 24, 14, 31, tzinfo=timezone.utc),
        last_captured_at=datetime(
            2026,
            7,
            24,
            14,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert cadence.due_mono == 1_240.0
    cadence.advance(1_250.0)
    assert cadence.due_mono == 1_540.0
    cadence.advance(2_180.0)
    assert cadence.due_mono == 2_440.0


def test_retained_universe_survives_spot_motion_and_restart() -> None:
    def contract(con_id: int, strike: float) -> SimpleNamespace:
        return SimpleNamespace(
            conId=con_id,
            secType="OPT",
            symbol="XSP",
            localSymbol=f"XSP-{con_id}",
            lastTradeDateOrContractMonth="20260727",
            strike=strike,
            right="C",
            exchange="SMART",
            multiplier="100",
            currency="USD",
            tradingClass="XSP",
        )

    universe = RetainedOptionUniverse()
    universe.begin(("2026-07-24", "20260727"))
    universe.retain([contract(1, 625.0)])
    assert universe.unseen([contract(0, 625.0)]) == []
    universe.retain([contract(2, 630.0)])
    assert [item.conId for item in universe.contracts] == [1, 2]

    restored = RetainedOptionUniverse()
    restored.begin(("2026-07-24", "20260727"))
    restored.restore(
        [
            SimpleNamespace(
                con_id=item.conId,
                sec_type=item.secType,
                symbol=item.symbol,
                local_symbol=item.localSymbol,
                expiry=item.lastTradeDateOrContractMonth,
                strike=item.strike,
                right=item.right,
                exchange=item.exchange,
                multiplier=item.multiplier,
                currency=item.currency,
                trading_class=item.tradingClass,
            )
            for item in universe.contracts
        ]
    )
    assert [item.conId for item in restored.contracts] == [1, 2]
    restored.begin(("2026-07-25", "20260728"))
    assert restored.contracts == []


def test_snapshot_quality_requires_qualification_fresh_nbbo_and_streaming_live() -> (
    None
):
    snapshot = QuoteSnapshot(
        ts=datetime(2026, 7, 24, 14, 30, tzinfo=timezone.utc).isoformat(),
        md_type=3,
        symbol="XSP",
        underlying=_quote(11004968),
        options=[
            _quote(1, full_greeks=True),
            _quote(2, md_type=3),
            _quote(3, quote_time=None),
            _quote(0),
        ],
        errors=[],
    )

    quality = snapshot_quality(snapshot, max_age_sec=30, require_live=True)

    assert quality == {
        "requirements": {
            "require_nbbo": True,
            "require_streaming_live": True,
            "max_age_sec": 30,
            "require_provenance": False,
            "require_all_options": False,
            "require_greeks": False,
        },
        "complete": True,
        "reasons": (),
        "provenance_complete": False,
        "total_options": 4,
        "qualified_options": 3,
        "invalid_options": 1,
        "timestamped_options": 2,
        "nbbo_options": 3,
        "eligible_options": 1,
        "live_options": 2,
        "streaming_options": 2,
        "delayed_options": 1,
        "full_greek_options": 1,
        "errors": 0,
    }


def test_strict_snapshot_quality_requires_complete_provenance_quotes_and_greeks() -> (
    None
):
    quality = snapshot_quality(
        _snapshot("2026-07-24T14:30:00+00:00"),
        max_age_sec=30,
        require_live=True,
        require_provenance=True,
        require_all_options=True,
        require_greeks=True,
    )

    assert quality["complete"] is False
    assert quality["reasons"] == ("provenance_incomplete", "greeks_incomplete")


def test_option_implied_underlier_requires_fresh_cross_sectional_consensus() -> None:
    snapshot = QuoteSnapshot(
        ts="2026-07-24T14:30:00+00:00",
        md_type=3,
        symbol="XSP",
        underlying=_quote(11004968),
        options=[_quote(index, md_type=3, full_greeks=True) for index in range(1, 6)],
        errors=[],
        chain_fingerprint="a" * 64,
        target_expiry="20260727",
        session="RTH",
    )

    observation = option_implied_underlier(snapshot)

    assert observation["usable"] is True
    assert observation["value"] == 625.0
    assert observation["observations"] == 5
    assert observation["dispersion_points"] == 0.0
    assert observation["max_age_seconds"] == 5.0
    assert observation["market_data_types"] == {"3": 5}


def test_option_implied_underlier_fails_closed_on_dispersion_or_missing_provenance() -> (
    None
):
    snapshot = QuoteSnapshot(
        ts="2026-07-24T14:30:00+00:00",
        md_type=3,
        symbol="XSP",
        underlying=_quote(11004968),
        options=[
            replace(
                _quote(index, md_type=3, full_greeks=True),
                model_under_price=625.0 if index < 5 else 626.0,
            )
            for index in range(1, 6)
        ],
        errors=[],
    )

    observation = option_implied_underlier(snapshot)

    assert observation["usable"] is False
    assert observation["reasons"] == (
        "provenance_incomplete",
        "model_dispersion",
    )


def _parity_snapshot(*, pairs: int = 6, provenance: bool = True) -> QuoteSnapshot:
    options = []
    for offset, strike in enumerate(range(738, 738 + pairs)):
        call_mid = 4.0
        put_mid = strike + call_mid - 740.5
        for right, midpoint in (("C", call_mid), ("P", put_mid)):
            options.append(
                replace(
                    _quote(100 + offset * 2 + (right == "P"), md_type=3),
                    expiry="20260727",
                    strike=float(strike),
                    right=right,
                    bid=midpoint - 0.05,
                    ask=midpoint + 0.05,
                )
            )
    return QuoteSnapshot(
        ts="2026-07-24T14:30:00+00:00",
        md_type=3,
        symbol="XSP",
        underlying=replace(
            _quote(137851301, md_type=3),
            sec_type="IND",
            bid=740.0,
            ask=740.2,
            last=740.1,
        ),
        options=options,
        errors=[],
        chain_fingerprint="a" * 64 if provenance else None,
        target_expiry="20260727",
        session="RTH",
    )


def test_option_parity_observation_uses_nearest_five_timestamped_pairs() -> None:
    observation = option_parity_observation(_parity_snapshot())

    assert observation["usable"] is True
    assert observation["anchor"] == pytest.approx(740.1)
    assert observation["anchor_source"] == "underlying"
    assert observation["value"] == pytest.approx(740.5)
    assert observation["pairs"] == 5
    assert observation["strikes"] == (740.0, 741.0, 739.0, 742.0, 738.0)
    assert observation["dispersion_points"] == pytest.approx(0.0)
    assert observation["max_age_seconds"] == pytest.approx(5.0)
    assert observation["market_data_types"] == {"3": 10}


def test_option_parity_observation_fails_closed_without_provenance_or_pairs() -> None:
    observation = option_parity_observation(_parity_snapshot(pairs=2, provenance=False))

    assert observation["usable"] is False
    assert observation["reasons"] == (
        "provenance_incomplete",
        "insufficient_nbbo_pairs",
    )


class _Event:
    def __iadd__(self, callback):
        self.callback = callback
        return self


class _Ticker(SimpleNamespace):
    def marketPrice(self) -> float:
        return float(self.last)


class _IB:
    instances = []

    def __init__(self) -> None:
        self.errorEvent = _Event()
        self.connected = False
        self.connects = 0
        self.connect_readonly: list[bool] = []
        self.qualifications = 0
        self.fail_first_option_snapshot = True
        self.__class__.instances.append(self)

    def connect(self, *_args, **_kwargs) -> None:
        self.connects += 1
        self.connect_readonly.append(bool(_kwargs.get("readonly")))
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def reqMarketDataType(self, _md_type: int) -> None:
        pass

    def qualifyContracts(self, *contracts):
        self.qualifications += 1
        for index, contract in enumerate(contracts, 1):
            contract.conId = index
            contract.localSymbol = f"XSP-{index}"
        return list(contracts)

    def reqTickers(self, *contracts):
        is_option_request = bool(contracts) and all(
            getattr(contract, "secType", "") in ("OPT", "FOP") for contract in contracts
        )
        if is_option_request and self.fail_first_option_snapshot:
            self.fail_first_option_snapshot = False
            self.connected = False
            raise TimeoutError("simulated request timeout")
        return [
            _Ticker(
                bid=1.0,
                ask=1.1,
                last=625.0 if not is_option_request else 1.05,
                close=624.0 if not is_option_request else 1.0,
                bidSize=1,
                askSize=1,
                lastSize=1,
                volume=1,
                modelGreeks=None,
                marketDataType=3,
                time=datetime.now(timezone.utc),
            )
            for _contract in contracts
        ]

    def sleep(self, _delay: float) -> None:
        pass


def test_recorder_recovers_chain_and_transport_without_empty_or_duplicate_rows(
    monkeypatch,
    tmp_path,
) -> None:
    _IB.instances.clear()
    underlying = SimpleNamespace(
        conId=11004968,
        secType="IND",
        symbol="XSP",
        localSymbol="XSP",
        exchange="CBOE",
        currency="USD",
        lastTradeDateOrContractMonth="",
        strike=0.0,
        right="",
        tradingClass="",
        multiplier="",
        minTick=0.01,
    )
    chain = SimpleNamespace(
        exchange="SMART",
        tradingClass="XSP",
        multiplier="100",
        expirations=("20260727",),
        strikes=(620.0, 625.0, 630.0),
    )
    monkeypatch.setattr(record_quotes, "IB", _IB)
    monkeypatch.setattr(
        record_quotes,
        "load_config",
        lambda: SimpleNamespace(
            host="127.0.0.1",
            port=4002,
            client_id_pool_end=899,
            readonly=True,
        ),
    )
    resolutions = 0

    def resolve(*_args, **_kwargs):
        nonlocal resolutions
        resolutions += 1
        return (
            (underlying, 625.0, None, False)
            if resolutions == 1
            else (underlying, 625.0, chain, False)
        )

    monkeypatch.setattr(record_quotes, "resolve_option_chain", resolve)
    monkeypatch.setattr(
        record_quotes,
        "xsp_trading_date",
        lambda _now: datetime(2026, 7, 24).date(),
    )
    monkeypatch.setattr(record_quotes.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(record_quotes.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_quotes",
            "--symbol",
            "XSP",
            "--md-type",
            "3",
            "--dte",
            "3",
            "--moneyness",
            "1",
            "--interval",
            "0",
            "--count",
            "2",
            "--out-dir",
            str(tmp_path),
        ],
    )

    record_quotes.main()

    [ib] = _IB.instances
    assert ib.RequestTimeout == 45.0
    [tape] = list((tmp_path / "XSP").glob("*.jsonl"))
    assert ib.connects == 3
    assert ib.connect_readonly == [True, True, True]
    assert ib.qualifications == 1
    assert resolutions == 2
    assert [len(snapshot.options) for snapshot in iter_snapshots(tape)] == [6, 6]


def test_recorder_restart_repairs_and_resumes_the_same_trading_date_tape(
    monkeypatch,
    tmp_path,
) -> None:
    class HealthyIB(_IB):
        def __init__(self) -> None:
            super().__init__()
            self.fail_first_option_snapshot = False

    HealthyIB.instances.clear()
    day = date(2026, 7, 27)
    underlying = SimpleNamespace(
        conId=11004968,
        secType="IND",
        symbol="XSP",
        localSymbol="XSP",
        exchange="CBOE",
        currency="USD",
        minTick=0.01,
    )
    chain = SimpleNamespace(
        exchange="SMART",
        tradingClass="XSP",
        multiplier="100",
        expirations=("20260727",),
        strikes=(620.0, 625.0, 630.0),
    )
    monkeypatch.setattr(record_quotes, "IB", HealthyIB)
    monkeypatch.setattr(
        record_quotes,
        "load_config",
        lambda: SimpleNamespace(
            host="127.0.0.1",
            port=4002,
            client_id_pool_end=899,
            readonly=True,
        ),
    )
    monkeypatch.setattr(
        record_quotes,
        "resolve_option_chain",
        lambda *_args, **_kwargs: (underlying, 625.0, chain, False),
    )
    monkeypatch.setattr(record_quotes, "xsp_trading_date", lambda _now: day)
    monkeypatch.setattr(record_quotes.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_quotes",
            "--symbol",
            "XSP",
            "--md-type",
            "3",
            "--dte",
            "0",
            "--interval",
            "0",
            "--count",
            "1",
            "--out-dir",
            str(tmp_path),
        ],
    )

    record_quotes.main()
    tape = tmp_path / "XSP" / f"{day}.jsonl"
    with tape.open("ab") as handle:
        handle.write(b'{"interrupted":')
    record_quotes.main()

    assert len(list(iter_snapshots(tape))) == 2
    assert tape.read_bytes().endswith(b"\n")
    assert [ib.qualifications for ib in HealthyIB.instances] == [1, 0]
    assert all(ib.connect_readonly == [True] for ib in HealthyIB.instances)


def test_indefinite_xsp_recorder_exits_cleanly_outside_capture_window(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _IB.instances.clear()
    monkeypatch.setattr(record_quotes, "IB", _IB)
    monkeypatch.setattr(
        record_quotes,
        "load_config",
        lambda: SimpleNamespace(
            host="127.0.0.1",
            port=4002,
            client_id_pool_end=899,
            readonly=True,
        ),
    )
    monkeypatch.setattr(
        record_quotes,
        "xsp_capture_window_date",
        lambda _now: None,
    )
    monkeypatch.setattr(record_quotes.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_quotes",
            "--symbol",
            "XSP",
            "--interval",
            "300",
            "--count",
            "0",
            "--out-dir",
            str(tmp_path),
        ],
    )

    record_quotes.main()

    [ib] = _IB.instances
    assert ib.connects == 0
    assert json.loads(capsys.readouterr().out) == {
        "broker_request_skipped": "closed_capture_window",
        "status": "closed",
    }


def test_indefinite_xsp_recorder_finishes_one_capture_window(
    monkeypatch,
    tmp_path,
) -> None:
    class HealthyIB(_IB):
        def __init__(self) -> None:
            super().__init__()
            self.fail_first_option_snapshot = False

    HealthyIB.instances.clear()
    day = date(2026, 7, 27)
    windows = iter((day, day, None))
    monkeypatch.setattr(record_quotes, "IB", HealthyIB)
    monkeypatch.setattr(
        record_quotes,
        "load_config",
        lambda: SimpleNamespace(
            host="127.0.0.1",
            port=4002,
            client_id_pool_end=899,
            readonly=True,
        ),
    )
    monkeypatch.setattr(
        record_quotes,
        "resolve_option_chain",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                conId=11004968,
                secType="IND",
                symbol="XSP",
                localSymbol="XSP",
                exchange="CBOE",
                currency="USD",
                minTick=0.01,
            ),
            625.0,
            SimpleNamespace(
                exchange="SMART",
                tradingClass="XSP",
                multiplier="100",
                expirations=("20260727",),
                strikes=(620.0, 625.0, 630.0),
            ),
            False,
        ),
    )
    monkeypatch.setattr(
        record_quotes,
        "xsp_capture_window_date",
        lambda _now: next(windows),
    )
    monkeypatch.setattr(record_quotes, "xsp_trading_date", lambda _now: day)
    monkeypatch.setattr(record_quotes.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_quotes",
            "--symbol",
            "XSP",
            "--md-type",
            "3",
            "--dte",
            "0",
            "--interval",
            "0",
            "--count",
            "0",
            "--out-dir",
            str(tmp_path),
        ],
    )

    record_quotes.main()

    [tape] = list((tmp_path / "XSP").glob("*.jsonl"))
    assert len(list(iter_snapshots(tape))) == 1
    assert len(list((tmp_path / "XSP" / "receipts").glob("*.json"))) == 1
    [ib] = HealthyIB.instances
    assert ib.connects == 1


def test_xsp_quote_producer_is_one_bounded_readonly_session_process() -> None:
    root = Path(__file__).resolve().parents[1]
    service = (root / "deploy/systemd/tradebot-xsp-quotes.service").read_text()
    timer = (root / "deploy/systemd/tradebot-xsp-quotes.timer").read_text()

    assert "Wants=network-online.target tradebot-ib-gateway-tunnel.service" in service
    assert "Requires=tradebot-ib-gateway-tunnel.service" not in service
    assert "Environment=IBKR_READONLY=1" in service
    assert "--md-type 3" in service
    assert "--interval 300 --count 0" in service
    assert "Restart=on-failure" in service
    assert "RuntimeMaxSec=20h50m" in service
    assert "Sun,Mon,Tue,Wed,Thu *-*-* 20:15:00 America/New_York" in timer
    assert "Persistent=true" in timer
    assert "RandomizedDelaySec=0" in timer
