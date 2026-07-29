from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.research.live_calibration import calibration_fingerprint
from tradebot.research.xsp_live_transport import (
    XSP_V2_TRANSPORT_ORDER_AUTHORITY,
    XSP_V3_TRANSPORT_SELECTION_SCHEMA,
    project_xsp_transport_plan,
    load_xsp_v2_transport_selection,
    project_xsp_v2_transport_plan,
    select_xsp_v2_transport,
    write_xsp_v2_transport_selection,
)
from tradebot.research.xsp_live_transport_state import latest_xsp_v2_source_receipt
from tradebot.research.xsp_live_transport_v3 import (
    load_xsp_v3_transport_selection_from_mapping,
    select_xsp_v3_transport,
)
from tradebot.research.xsp_live_transport_runtime import (
    xsp_transport_order_ref,
    xsp_transport_risk_state,
)
from tradebot.engines.execution import execution_policy_contract


SELECTED_AT = datetime(2026, 7, 29, 13, 38, tzinfo=timezone.utc)
OBSERVED_AT = SELECTED_AT + timedelta(minutes=4)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _address(value: dict[str, object]) -> dict[str, object]:
    return {
        **value,
        "identity_sha256": calibration_fingerprint(value),
    }


def _nominee() -> dict[str, object]:
    return {
        "family": "five_slot",
        "profile_id": "fixed_measured",
        "nominee_id": "n" * 64,
        "fixed_entry_notional_usd": 260.0,
        "historical_quantity_ranges": {
            "SPYU": [7, 10],
            "SPXU": [20, 30],
        },
        "frozen_max_quantities": {"SPYU": 10, "SPXU": 30},
    }


def _ranking(
    nominee: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": "xsp.opening-edge-v2-spyu-selection-ranking-result.v1",
        "authority": "research_ranking_only",
        "order_authority": "none",
        "profitability_clock_started": False,
        "selected_shadow_created": False,
        "verdict": "NOMINEE_STILL_HOLD",
        "nominee": nominee or _nominee(),
    }


def _dwell() -> dict[str, object]:
    return _address(
        {
            "schema": "xsp.network-b-symbol-dwell-validation-result.v1",
            "authority": "historical_execution_validation_only",
            "order_authority": "none",
            "submitted_orders": 0,
            "profitability_clock_started": False,
            "selected_shadow_created": False,
            "verdict": "DWELL_VALIDATION_PASS_SELECTION_STILL_HOLD",
            "nominee_id": "n" * 64,
        }
    )


def _preview(
    *,
    ranking_path: Path,
    dwell_path: Path,
    nominee: dict[str, object] | None = None,
    quantities: dict[str, int] | None = None,
) -> dict[str, object]:
    nominee = nominee or _nominee()
    quantities = quantities or {"SPYU": 8, "SPXU": 25}
    ranges = nominee["historical_quantity_ranges"]
    assert isinstance(ranges, dict)
    rows = []
    for symbol, commission in (
        ("SPYU", 1.010129),
        ("SPXU", 1.010093),
    ):
        rows.append(
            {
                "symbol": symbol,
                "fixed_entry_notional_usd": nominee[
                    "fixed_entry_notional_usd"
                ],
                "historical_quantity_range": ranges[symbol],
                "quote_derived_quantity": quantities[symbol],
                "quantity_in_historical_range": True,
                "contract": {"con_id": 1 if symbol == "SPYU" else 2},
                "order": {
                    "action": "BUY",
                    "what_if": True,
                    "transmit": False,
                },
                "commission_limit_usd": commission,
                "preview_pass": True,
            }
        )
    return _address(
        {
            "schema": "xsp.opening-edge-v2-ranked-nominee-preview.v1",
            "observed_at_utc": (SELECTED_AT - timedelta(minutes=2)).isoformat(),
            "authority": "fresh_broker_preview_only",
            "order_authority": "none",
            "submitted_orders": 0,
            "profitability_clock_started": False,
            "selected_shadow_created": False,
            "verdict": "PREVIEW_PASS_SELECTION_STILL_HOLD",
            "nominee": nominee,
            "inputs": {
                "ranking": {
                    "path": str(ranking_path),
                    "sha256": __import__("hashlib")
                    .sha256(ranking_path.read_bytes())
                    .hexdigest(),
                },
                "dwell": {
                    "path": str(dwell_path),
                    "sha256": __import__("hashlib")
                    .sha256(dwell_path.read_bytes())
                    .hexdigest(),
                },
            },
            "broker": {
                "rows": rows,
                "submitted_orders": 0,
                "all_previews_pass": True,
                "open_trades_before": 0,
                "open_trades_after": 0,
            },
        }
    )


def _position(
    direction: str,
    *,
    entry_time: datetime,
) -> dict[str, object]:
    signal_at = entry_time - timedelta(minutes=5)
    return {
        "lane": "rth",
        "direction": direction,
        "entry_time": entry_time.isoformat(),
        "trading_date": "2026-07-29",
        "entry_price": 750.0,
        "exit_reason": "end",
        "attribution": {
            "schema": "xsp.trade-attribution.v1",
            "decision_trace_fingerprint": "d" * 64,
            "entry": {
                "signal_bar_ts": signal_at.isoformat(),
                "source_direction": direction,
                "control": {
                    "source": "directional_impulse",
                    "direction": direction,
                },
                "directional_impulse": {
                    "ready": True,
                    "direction": direction,
                    "atr_velocity_pct": 0.01,
                    "horizons": [
                        {
                            "bars": 1,
                            "slope_velocity_pct_per_bar": 0.02,
                        }
                    ],
                },
                "market_state": {
                    "shock_atr_vel_pct": 0.01,
                    "slope_vel_pct": 0.02,
                },
                "local_extrema": None,
            },
        },
    }


def _source(
    position: dict[str, object] | None,
    *,
    recorded_at: datetime,
    checkpoint: str = "checkpoint",
    session: str = "RTH",
) -> dict[str, object]:
    profile = {
        "run_started_at_utc": "2026-07-28T00:15:00+00:00",
        "latest_position": position,
    }
    return {
        "evaluation_status": "EVALUATED",
        "freshness_ok": True,
        "session": session,
        "order_authority": "none",
        "checkpoint_id": checkpoint,
        "recorded_at_utc": recorded_at.isoformat(),
        "paired_equity": {
            "crown_config_fingerprint": "crown",
            "profiles": {
                "research": dict(profile),
                "broker": dict(profile),
            },
        },
    }


def _selection_inputs(
    tmp_path: Path,
    *,
    nominee: dict[str, object] | None = None,
    quantities: dict[str, int] | None = None,
    settled_cash_usd: float = 1_350.0,
):
    nominee = nominee or _nominee()
    ranking = _write(tmp_path / "ranking.json", _ranking(nominee))
    dwell = _write(tmp_path / "dwell.json", _dwell())
    preview = _write(
        tmp_path / "preview.json",
        _preview(
            ranking_path=ranking,
            dwell_path=dwell,
            nominee=nominee,
            quantities=quantities,
        ),
    )
    source = _source(
        _position(
            "up",
            entry_time=SELECTED_AT - timedelta(minutes=5),
        ),
        recorded_at=SELECTED_AT - timedelta(seconds=30),
    )
    broker = {
        "observed_at_utc": (
            SELECTED_AT - timedelta(seconds=5)
        ).isoformat(),
        "cash_observed_at_utc": (
            SELECTED_AT - timedelta(seconds=10)
        ).isoformat(),
        "account_id": "DU123456",
        "account_type": "CASH",
        "settled_cash_usd": settled_cash_usd,
        "positions": {"SPYU": 0, "SPXU": 0},
        "unrelated_positions": [
            {
                "symbol": "TQQQ",
                "con_id": 72_539_702,
                "quantity": 1,
            }
        ],
        "open_orders": [],
    }
    return ranking, dwell, preview, source, broker


def _selection(tmp_path: Path) -> dict[str, object]:
    ranking, dwell, preview, source, broker = _selection_inputs(tmp_path)
    return select_xsp_v2_transport(
        ranking_path=ranking,
        dwell_path=dwell,
        preview_path=preview,
        source_receipt=source,
        broker_snapshot=broker,
        selected_at=SELECTED_AT,
    )


def _quotes() -> dict[str, dict[str, object]]:
    return {
        "SPYU": {
            "bid": 30.48,
            "ask": 30.50,
            "age_seconds": 0.5,
            "market_data_type": 1,
        },
        "SPXU": {
            "bid": 10.18,
            "ask": 10.20,
            "age_seconds": 0.5,
            "market_data_type": 1,
        },
    }


def _v3_preview() -> dict[str, object]:
    rows = []
    for symbol, con_id, quantity, bid, ask in (
        ("UPRO", 61_228_752, 9, 99.98, 100.0),
        ("SPXU", 53_362_064, 18, 49.98, 50.0),
    ):
        book = {
            "observed_at_utc": (SELECTED_AT - timedelta(minutes=1)).isoformat(),
            "market_data_type": 1,
            "bid": bid,
            "ask": ask,
            "bid_size": 100,
            "ask_size": 100,
        }
        rows.append(
            {
                "symbol": symbol,
                "contract": {
                    "con_id": con_id,
                    "exchange": "SMART",
                    "primary_exchange": "ARCA",
                    "currency": "USD",
                },
                "books": {"smart": dict(book), "direct_arca": dict(book)},
                "order": {
                    "action": "BUY",
                    "quantity": quantity,
                    "limit_price": ask,
                    "notional_usd": quantity * ask,
                    "tif": "DAY",
                    "what_if": True,
                    "transmit": False,
                },
                "preview": {
                    "status": "PreSubmitted",
                    "commission": 0.35,
                    "min_commission": 0.35,
                    "max_commission": 0.35,
                    "commission_currency": "USD",
                    "warning_text": "",
                },
                "tiered_conservative_buy_fee_usd": 0.43,
                "effective_tiered_commission": True,
                "cash_fit": True,
            }
        )
    return {
        "schema": "xsp.opening-edge-v3-upro-spxu-preview.v1",
        "authority": "broker_preview_only",
        "observed_at_utc": (SELECTED_AT - timedelta(minutes=1)).isoformat(),
        "source": {
            "checkpoint_id": "source-preview",
            "crown_artifact_sha256": (
                "d47eb39cef3d2ca575d779d6b5b87e3b88e08606fd09a8801b8cb55c350208db"
            ),
            "state_owner_sha256": "s" * 64,
            "daily_context_fingerprint": "d" * 64,
        },
        "cash_receipt_sha256": (
            "e41e44db270ea872679746cb2b83b2aa73987e523863236472f6e8ec0434c8dc"
        ),
        "notional_usd": 900.0,
        "settled_cash_usd": 1_200.0,
        "rows": rows,
        "errors": [],
        "relevant_positions": [],
        "relevant_open_orders": [],
        "open_trades_before": 0,
        "open_trades_after": 0,
        "portfolio_rows_before": 0,
        "portfolio_rows_after": 0,
        "books_pass": True,
        "quantity_and_cash_pass": True,
        "effective_tiered_commission_pass": True,
        "selection_created": False,
        "profitability_clock_started": False,
        "order_authority": "none",
        "submitted_orders": 0,
        "verdict": "PREVIEW_PASS_STILL_HOLD",
    }


def _v3_selection(tmp_path: Path) -> dict[str, object]:
    preview_path = _write(tmp_path / "v3-preview.json", _v3_preview())
    source = _source(
        _position("down", entry_time=SELECTED_AT - timedelta(minutes=5)),
        recorded_at=SELECTED_AT - timedelta(seconds=30),
    )
    broker = {
        "observed_at_utc": (SELECTED_AT - timedelta(seconds=5)).isoformat(),
        "cash_observed_at_utc": (SELECTED_AT - timedelta(seconds=10)).isoformat(),
        "account_id": "DU123456",
        "account_type": "CASH",
        "settled_cash_usd": 1_200.0,
        "positions": {"UPRO": 0, "SPXU": 0},
        "unrelated_positions": [],
        "open_orders": [],
    }
    return select_xsp_v3_transport(
        cash_receipt_path=Path(
            "backtests/xsp/opening_edge_v3_regime_harmony_cash_receipt.json"
        ),
        preview_path=preview_path,
        source_receipt=source,
        broker_snapshot=broker,
        selected_at=SELECTED_AT,
        rth_scope_accepted=True,
    )


def test_v3_selection_requires_explicit_rth_scope_and_binds_tiered_identity(
    tmp_path: Path,
) -> None:
    preview_path = _write(tmp_path / "v3-preview.json", _v3_preview())
    source = _source(
        _position("down", entry_time=SELECTED_AT - timedelta(minutes=5)),
        recorded_at=SELECTED_AT - timedelta(seconds=30),
    )
    broker = {
        "observed_at_utc": (SELECTED_AT - timedelta(seconds=5)).isoformat(),
        "cash_observed_at_utc": (SELECTED_AT - timedelta(seconds=10)).isoformat(),
        "account_id": "DU123456",
        "account_type": "CASH",
        "settled_cash_usd": 1_200.0,
        "positions": {"UPRO": 0, "SPXU": 0},
        "unrelated_positions": [],
        "open_orders": [],
    }
    with pytest.raises(ValueError, match="RTH-only scope"):
        select_xsp_v3_transport(
            cash_receipt_path=Path(
                "backtests/xsp/opening_edge_v3_regime_harmony_cash_receipt.json"
            ),
            preview_path=preview_path,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
            rth_scope_accepted=False,
        )

    selection = _v3_selection(tmp_path)
    assert selection["schema"] == XSP_V3_TRANSPORT_SELECTION_SCHEMA
    assert selection["direction_symbols"] == {"up": "UPRO", "down": "SPXU"}
    assert selection["nominee"]["fixed_entry_notional_usd"] == 900.0
    assert selection["nominee"]["pricing_plan"] == "Tiered"
    assert selection["execution"]["UPRO_BUY"] == {
        "initial_mode": "OPTIMISTIC",
        "chase_mode": "AUTO",
    }
    assert load_xsp_v3_transport_selection_from_mapping(selection) == selection


def test_v3_selection_rejects_a_stale_internal_book(tmp_path: Path) -> None:
    preview = _v3_preview()
    preview["rows"][0]["books"]["smart"]["observed_at_utc"] = (
        SELECTED_AT - timedelta(minutes=2)
    ).isoformat()
    preview_path = _write(tmp_path / "v3-stale-preview.json", preview)
    with pytest.raises(ValueError, match="fresh smart book"):
        select_xsp_v3_transport(
            cash_receipt_path=Path(
                "backtests/xsp/opening_edge_v3_regime_harmony_cash_receipt.json"
            ),
            preview_path=preview_path,
            source_receipt=_source(
                None,
                recorded_at=SELECTED_AT - timedelta(seconds=30),
            ),
            broker_snapshot={
                "observed_at_utc": (SELECTED_AT - timedelta(seconds=5)).isoformat(),
                "cash_observed_at_utc": (
                    SELECTED_AT - timedelta(seconds=10)
                ).isoformat(),
                "account_id": "DU123456",
                "account_type": "CASH",
                "settled_cash_usd": 1_200.0,
                "positions": {"UPRO": 0, "SPXU": 0},
                "unrelated_positions": [],
                "open_orders": [],
            },
            selected_at=SELECTED_AT,
            rth_scope_accepted=True,
        )


def test_v3_projection_reuses_shared_ladder_without_spyu_nav(
    tmp_path: Path,
) -> None:
    selection = _v3_selection(tmp_path)
    source = _source(
        _position("up", entry_time=OBSERVED_AT - timedelta(minutes=1)),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
        checkpoint="v3-up",
    )
    plan = project_xsp_transport_plan(
        selection=selection,
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"UPRO": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_200.0,
        quotes={
            "UPRO": {
                "bid": 99.98,
                "ask": 100.0,
                "age_seconds": 0.5,
                "market_data_type": 1,
            },
            "SPXU": {
                "bid": 49.98,
                "ask": 50.0,
                "age_seconds": 0.5,
                "market_data_type": 1,
            },
        },
    )

    assert plan["status"] == "ACTIONABLE"
    assert plan["leg"]["symbol"] == "UPRO"
    assert plan["leg"]["quantity"] == 9
    assert plan["leg"]["initial_mode"] == "OPTIMISTIC"
    assert plan["leg"]["chase_mode"] == "AUTO"
    assert plan["leg"]["spyu_nav_divergence"] is None


def test_v3_restart_risk_and_order_identity_use_selected_symbols(
    tmp_path: Path,
) -> None:
    selection = _v3_selection(tmp_path)
    risk = xsp_transport_risk_state(
        selection=selection,
        records=(),
        observed_at=OBSERVED_AT,
        liquidation_bids={},
    )
    assert risk["holdings_from_fills"] == {"UPRO": 0.0, "SPXU": 0.0}

    plan = project_xsp_transport_plan(
        selection=selection,
        source_receipt=_source(
            _position("down", entry_time=OBSERVED_AT - timedelta(minutes=1)),
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
            checkpoint="v3-down",
        ),
        observed_at=OBSERVED_AT,
        positions={"UPRO": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_200.0,
        quotes={
            "UPRO": {
                "bid": 99.98,
                "ask": 100.0,
                "age_seconds": 0.5,
                "market_data_type": 1,
            },
            "SPXU": {
                "bid": 49.98,
                "ask": 50.0,
                "age_seconds": 0.5,
                "market_data_type": 1,
            },
        },
    )
    assert xsp_transport_order_ref(plan).startswith("XSPV3-")


def test_selection_binds_every_gate_and_starts_strictly_flat_run(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)

    assert selection["order_authority"] == XSP_V2_TRANSPORT_ORDER_AUTHORITY
    assert selection["profitability_clock_started"] is True
    assert selection["baseline_state"]["direction"] == "up"
    assert selection["nominee"]["commission_limits_usd"] == {
        "SPYU": 1.010129,
        "SPXU": 1.010093,
    }
    assert selection["execution"]["SPYU_BUY"] == {
        "initial_mode": "CROSS",
        "chase_mode": "RELENTLESS",
    }
    assert selection["execution"]["policy_contract"] == execution_policy_contract()
    assert selection["broker_at_selection"]["account_id"] == "DU123456"
    assert selection["broker_at_selection"]["observed_at_utc"] == (
        SELECTED_AT - timedelta(seconds=5)
    ).isoformat()
    assert selection["broker_at_selection"]["cash_observed_at_utc"] == (
        SELECTED_AT - timedelta(seconds=10)
    ).isoformat()
    assert selection["broker_at_selection"]["unrelated_positions"] == [
        {
            "symbol": "TQQQ",
            "con_id": 72_539_702,
            "quantity": 1,
        }
    ]
    assert selection["nominee"]["capital_identity"] == {
        "starting_cash_identity_usd": 1_350.0,
        "fixed_entry_notional_usd": 260.0,
        "cash_slots": 5,
        "maximum_gross_purchase_notional_usd": 1_300.0,
        "settlement": "strict_T_plus_1_settled_cash_only",
        "unsettled_sale_proceeds_reused": False,
    }
    assert selection["broker_at_selection"]["minimum_settled_cash_usd"] == (
        1_305.050645
    )

    path = tmp_path / "selected.json"
    write_xsp_v2_transport_selection(path, selection)
    assert load_xsp_v2_transport_selection(path) == selection


def test_selection_requires_nominee_reserve_not_whole_risk_identity(
    tmp_path: Path,
) -> None:
    nominee = {
        "family": "notional",
        "profile_id": "notional=1200:fixed_measured",
        "nominee_id": "n" * 64,
        "fixed_entry_notional_usd": 1_200.0,
        "historical_quantity_ranges": {
            "SPYU": [33, 40],
            "SPXU": [100, 120],
        },
        "frozen_max_quantities": {"SPYU": 40, "SPXU": 120},
    }
    ranking, dwell, preview, source, broker = _selection_inputs(
        tmp_path,
        nominee=nominee,
        quantities={"SPYU": 38, "SPXU": 110},
        settled_cash_usd=1_228.61,
    )

    selection = select_xsp_v2_transport(
        ranking_path=ranking,
        dwell_path=dwell,
        preview_path=preview,
        source_receipt=source,
        broker_snapshot=broker,
        selected_at=SELECTED_AT,
    )

    assert selection["risk"]["starting_cash_identity_usd"] == 1_350.0
    assert selection["broker_at_selection"]["settled_cash_usd"] == 1_228.61
    assert selection["broker_at_selection"]["minimum_settled_cash_usd"] == (
        1_201.010129
    )
    write_xsp_v2_transport_selection(
        tmp_path / "selected-notional.json",
        selection,
    )

    broker["settled_cash_usd"] = 1_201.0
    with pytest.raises(ValueError, match="settled USD reserve"):
        select_xsp_v2_transport(
            ranking_path=ranking,
            dwell_path=dwell,
            preview_path=preview,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
        )


def test_latest_source_projection_uses_only_the_canonical_checkpoint() -> None:
    source = _source(
        _position("up", entry_time=OBSERVED_AT - timedelta(minutes=2)),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
        checkpoint="source-id",
    )
    projected = latest_xsp_v2_source_receipt(
        [
            {
                "kind": "checkpoint",
                "strategy_version": "not-the-crown",
                "checkpoint_id": "ignore",
            },
            {
                "kind": "checkpoint",
                "strategy_version": "xsp.opening-edge-v2-spy-transport.v1",
                "checkpoint_id": source["checkpoint_id"],
                "recorded_at_utc": source["recorded_at_utc"],
                "status": source["evaluation_status"],
                "session": source["session"],
                "evidence": {
                    "paired_equity": source["paired_equity"],
                    "rth_provenance_fresh": True,
                    "order_authority": "none",
                },
            },
        ]
    )

    assert projected == source


def test_selection_rejects_evidence_drift_and_nonflat_broker(
    tmp_path: Path,
) -> None:
    ranking, dwell, preview, source, broker = _selection_inputs(tmp_path)
    ranking.write_text(ranking.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        select_xsp_v2_transport(
            ranking_path=ranking,
            dwell_path=dwell,
            preview_path=preview,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
        )

    ranking, dwell, preview, source, broker = _selection_inputs(tmp_path)
    broker["positions"]["SPYU"] = 1
    with pytest.raises(ValueError, match="flat cash-pair sleeve"):
        select_xsp_v2_transport(
            ranking_path=ranking,
            dwell_path=dwell,
            preview_path=preview,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
        )

    ranking, dwell, preview, source, broker = _selection_inputs(tmp_path)
    broker["open_orders"] = [{"symbol": "AAPL", "action": "BUY"}]
    with pytest.raises(ValueError, match="flat cash-pair sleeve"):
        select_xsp_v2_transport(
            ranking_path=ranking,
            dwell_path=dwell,
            preview_path=preview,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
        )


@pytest.mark.parametrize(
    "observed_at",
    [
        SELECTED_AT - timedelta(seconds=91),
        SELECTED_AT + timedelta(microseconds=1),
    ],
)
def test_selection_requires_fresh_causal_broker_snapshot(
    tmp_path: Path,
    observed_at: datetime,
) -> None:
    ranking, dwell, preview, source, broker = _selection_inputs(tmp_path)
    broker["observed_at_utc"] = observed_at.isoformat()

    with pytest.raises(ValueError, match="fresh broker account snapshot"):
        select_xsp_v2_transport(
            ranking_path=ranking,
            dwell_path=dwell,
            preview_path=preview,
            source_receipt=source,
            broker_snapshot=broker,
            selected_at=SELECTED_AT,
        )


def test_selected_run_cannot_rehash_weaker_risk_or_ladder(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    selection["risk"]["max_drawdown_usd"] = 1_350.0
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    selection["selection_id"] = calibration_fingerprint(body)
    path = _write(tmp_path / "weakened.json", selection)

    with pytest.raises(ValueError, match="invalid selected"):
        load_xsp_v2_transport_selection(path)


def test_selected_run_cannot_rehash_stale_broker_snapshot(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    selection["broker_at_selection"]["observed_at_utc"] = (
        SELECTED_AT - timedelta(seconds=91)
    ).isoformat()
    body = {key: value for key, value in selection.items() if key != "selection_id"}
    selection["selection_id"] = calibration_fingerprint(body)

    with pytest.raises(ValueError, match="invalid selected"):
        load_xsp_v2_transport_selection(
            _write(tmp_path / "stale-broker.json", selection)
        )


def test_preselection_position_is_not_backfilled(tmp_path: Path) -> None:
    plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=_source(
            _position(
                "up",
                entry_time=SELECTED_AT - timedelta(minutes=5),
            ),
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes={},
    )

    assert plan["status"] == "UNCHANGED"
    assert plan["reason"] == "flat_target"
    assert plan["leg"] is None


@pytest.mark.parametrize(
    ("lane", "direction"),
    [("rth", "up"), ("gth", "down")],
)
def test_preselection_signal_with_future_modeled_fill_is_not_backfilled(
    tmp_path: Path,
    lane: str,
    direction: str,
) -> None:
    ranking, dwell, preview, _source_before, broker = _selection_inputs(tmp_path)
    position = _position(
        direction,
        entry_time=SELECTED_AT + timedelta(minutes=2),
    )
    position["lane"] = lane
    position["entry_time"] = position["entry_time"].replace("+00:00", "")
    selection = select_xsp_v2_transport(
        ranking_path=ranking,
        dwell_path=dwell,
        preview_path=preview,
        source_receipt=_source(
            position,
            recorded_at=SELECTED_AT - timedelta(seconds=30),
        ),
        broker_snapshot=broker,
        selected_at=SELECTED_AT,
    )

    plan = project_xsp_v2_transport_plan(
        selection=selection,
        source_receipt=_source(
            position,
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes={},
    )

    assert plan["status"] == "UNCHANGED"
    assert plan["reason"] == "flat_target"


def test_new_up_target_crosses_spyu_then_uses_relentless_owner(
    tmp_path: Path,
) -> None:
    plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=_source(
            _position(
                "up",
                entry_time=SELECTED_AT + timedelta(minutes=2),
            ),
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes=_quotes(),
        spyu_nav={"value": 30.49, "age_seconds": 1.0},
    )

    assert plan["status"] == "ACTIONABLE"
    assert plan["reason"] == "buy_post_selection_target"
    assert plan["leg"]["action"] == "BUY"
    assert plan["leg"]["symbol"] == "SPYU"
    assert plan["leg"]["quantity"] == 8
    assert plan["leg"]["initial_mode"] == "CROSS"
    assert plan["leg"]["chase_mode"] == "RELENTLESS"
    assert plan["leg"]["outside_rth"] is False
    assert plan["submitted_orders"] == 0
    assert plan["signal_context"]["direction"] == "up"
    assert plan["signal_context"]["directional_impulse"]["horizons"] == [
        {
            "bars": 1,
            "slope_velocity_pct_per_bar": 0.02,
        }
    ]


def test_new_down_target_uses_canonical_shared_ladder(tmp_path: Path) -> None:
    plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=_source(
            _position(
                "down",
                entry_time=SELECTED_AT + timedelta(minutes=2),
            ),
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes=_quotes(),
    )

    assert plan["leg"]["symbol"] == "SPXU"
    assert plan["leg"]["quantity"] == 25
    assert plan["leg"]["initial_mode"] == "OPTIMISTIC"
    assert plan["leg"]["chase_mode"] == "AUTO"


def test_actionable_plan_rejects_profile_signal_attribution_drift(
    tmp_path: Path,
) -> None:
    source = _source(
        _position(
            "up",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
    )
    broker_position = deepcopy(
        source["paired_equity"]["profiles"]["broker"]["latest_position"]
    )
    broker_position["attribution"]["entry"]["directional_impulse"][
        "atr_velocity_pct"
    ] = -0.01
    source["paired_equity"]["profiles"]["broker"][
        "latest_position"
    ] = broker_position

    with pytest.raises(
        ValueError,
        match="research/broker execution attribution drift",
    ):
        project_xsp_v2_transport_plan(
            selection=_selection(tmp_path),
            source_receipt=source,
            observed_at=OBSERVED_AT,
            positions={"SPYU": 0, "SPXU": 0},
            open_orders=[],
            settled_cash_usd=1_350.0,
            quotes=_quotes(),
            spyu_nav={"value": 30.49, "age_seconds": 1.0},
        )


def test_flip_sells_incumbent_before_buying_target(tmp_path: Path) -> None:
    source = _source(
        _position(
            "down",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
    )
    sell = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_100.0,
        quotes=_quotes(),
    )

    assert sell["reason"] == "sell_incumbent_before_target"
    assert sell["leg"] == {
        "action": "SELL",
        "symbol": "SPYU",
        "quantity": 8,
        "initial_mode": "OPTIMISTIC",
        "chase_mode": "AUTO",
        "outside_rth": False,
        "bid": 30.48,
        "ask": 30.5,
    }

    reconcile = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[{"symbol": "SPYU", "action": "SELL"}],
        settled_cash_usd=1_100.0,
        quotes={},
    )
    assert reconcile["status"] == "RECONCILE_REQUIRED"
    assert reconcile["leg"] is None

    unrelated = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[{"symbol": "TQQQ", "action": "BUY"}],
        settled_cash_usd=1_100.0,
        quotes=_quotes(),
    )
    assert unrelated["status"] == "ACTIONABLE"
    assert unrelated["leg"]["action"] == "SELL"


def test_rth_end_liquidates_before_close_and_forbids_reentry(
    tmp_path: Path,
) -> None:
    observed_at = datetime(2026, 7, 29, 19, 57, tzinfo=timezone.utc)
    source = _source(
        _position(
            "up",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=observed_at - timedelta(seconds=30),
    )
    sell = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=observed_at,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_100.0,
        quotes=_quotes(),
    )

    assert sell["status"] == "ACTIONABLE"
    assert sell["reason"] == "rth_end_liquidation"
    assert sell["entry_window_open"] is False
    assert sell["leg"]["action"] == "SELL"
    assert sell["leg"]["outside_rth"] is False

    flat = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=observed_at,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes=_quotes(),
        spyu_nav={"value": 30.49, "age_seconds": 1.0},
    )
    assert flat["status"] == "UNCHANGED"
    assert flat["reason"] == "rth_entry_cutoff"
    assert flat["leg"] is None


def test_curb_checkpoint_can_only_liquidate_incumbent(
    tmp_path: Path,
) -> None:
    observed_at = datetime(2026, 7, 29, 20, 17, tzinfo=timezone.utc)
    source = _source(
        _position(
            "up",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=observed_at - timedelta(seconds=30),
        session="CURB",
    )
    sell = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=observed_at,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_100.0,
        quotes=_quotes(),
    )

    assert sell["target_symbol"] is None
    assert sell["reason"] == "rth_end_liquidation"
    assert sell["leg"]["action"] == "SELL"
    assert sell["leg"]["outside_rth"] is True

    flat = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=observed_at,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes=_quotes(),
    )
    assert flat["status"] == "UNCHANGED"
    assert flat["reason"] == "flat_target"


def test_early_close_uses_the_same_preclose_liquidation_boundary(
    tmp_path: Path,
) -> None:
    observed_at = datetime(2026, 11, 27, 17, 57, tzinfo=timezone.utc)
    source = _source(
        _position(
            "down",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=observed_at - timedelta(seconds=30),
    )
    plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=observed_at,
        positions={"SPYU": 0, "SPXU": 25},
        open_orders=[],
        settled_cash_usd=1_000.0,
        quotes=_quotes(),
    )

    assert plan["reason"] == "rth_end_liquidation"
    assert plan["leg"]["outside_rth"] is False


def test_unrelated_open_order_blocks_new_buy_only(tmp_path: Path) -> None:
    plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=_source(
            _position(
                "down",
                entry_time=SELECTED_AT + timedelta(minutes=2),
            ),
            recorded_at=OBSERVED_AT - timedelta(seconds=30),
        ),
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[{"symbol": "TQQQ", "action": "BUY"}],
        settled_cash_usd=1_350.0,
        quotes={},
    )

    assert plan["status"] == "RECONCILE_REQUIRED"
    assert plan["reason"] == "unrelated_open_order_blocks_buy"


def test_plan_fails_closed_on_stale_or_ambiguous_state(
    tmp_path: Path,
) -> None:
    source = _source(
        _position(
            "up",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
    )
    with pytest.raises(ValueError, match="both cash-pair"):
        project_xsp_v2_transport_plan(
            selection=_selection(tmp_path),
            source_receipt=source,
            observed_at=OBSERVED_AT,
            positions={"SPYU": 1, "SPXU": 1},
            open_orders=[],
            settled_cash_usd=1_350.0,
            quotes={},
        )

    stale = _quotes()
    stale["SPYU"]["age_seconds"] = 11.0
    with pytest.raises(ValueError, match="quote is not fresh"):
        project_xsp_v2_transport_plan(
            selection=_selection(tmp_path),
            source_receipt=source,
            observed_at=OBSERVED_AT,
            positions={"SPYU": 0, "SPXU": 0},
            open_orders=[],
            settled_cash_usd=1_350.0,
            quotes=stale,
            spyu_nav={"value": 30.49, "age_seconds": 1.0},
        )


def test_loss_limits_block_entries_but_never_block_exit(
    tmp_path: Path,
) -> None:
    source = _source(
        _position(
            "down",
            entry_time=SELECTED_AT + timedelta(minutes=2),
        ),
        recorded_at=OBSERVED_AT - timedelta(seconds=30),
    )
    halted = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"SPYU": 0, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_350.0,
        quotes={},
        session_net_usd=-67.5,
    )
    assert halted["status"] == "RISK_HALTED"
    assert halted["leg"] is None

    exit_plan = project_xsp_v2_transport_plan(
        selection=_selection(tmp_path),
        source_receipt=source,
        observed_at=OBSERVED_AT,
        positions={"SPYU": 8, "SPXU": 0},
        open_orders=[],
        settled_cash_usd=1_000.0,
        quotes=_quotes(),
        session_net_usd=-100.0,
        drawdown_usd=200.0,
    )
    assert exit_plan["leg"]["action"] == "SELL"
