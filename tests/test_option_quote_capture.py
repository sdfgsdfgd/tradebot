from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from tradebot.backtest.quotes import (
    QuoteContract,
    QuoteSnapshot,
    append_snapshot,
    iter_snapshot_payloads,
    make_chain_manifest,
    persist_chain_manifest,
    snapshot_quality,
)


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
    assert path.read_bytes().endswith(b"\n")


def test_snapshot_quality_requires_qualification_fresh_nbbo_and_streaming_live() -> None:
    snapshot = QuoteSnapshot(
        ts=datetime(2026, 7, 24, 14, 30, tzinfo=timezone.utc).isoformat(),
        md_type=1,
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
        },
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
