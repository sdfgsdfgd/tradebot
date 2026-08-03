"""Immutable Stage-76 source, selection, and account-capital boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import IB, LimitOrder

from ..backtest.models import SpotTrade
from ..live.capital import (
    build_live_capital_plan_v2,
    validate_live_capital_plan,
)
from .gold_context import gold_utc
from .gold_regime_harmony import (
    GOLD_REGIME_HARMONY_SOURCE_START,
    GOLD_REGIME_HARMONY_VERSION,
    GoldRegimeHarmonyReplay,
    GoldRegimeHarmonyTape,
    load_gold_regime_harmony_crown,
)
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint


GOLD_REGIME_HARMONY_SOURCE_VERSION = "gold.1oz-regime-harmony-source.v1"
GOLD_LIVE_SELECTION_SCHEMA = "gold.1oz-regime-harmony-selected-run.v1"
GOLD_LIVE_PACKAGE_SELECTION_SCHEMA = "gold.1oz-regime-harmony-selected-run.v2"
GOLD_LIVE_SELECTION_SCHEMAS = frozenset(
    {GOLD_LIVE_SELECTION_SCHEMA, GOLD_LIVE_PACKAGE_SELECTION_SCHEMA}
)
GOLD_LIVE_PLAN_SCHEMA = "gold.1oz-regime-harmony-transport-plan.v1"
GOLD_LIVE_EXECUTION_VERSION = "gold.1oz-regime-harmony-live-execution.v1"
GOLD_LIVE_EXECUTION_SCHEMA = "gold.1oz-regime-harmony-execution-checkpoint.v1"
GOLD_LIVE_CAPITAL_SLEEVE = "gold-1oz-stage76-margin"
GOLD_LIVE_ORDER_AUTHORITY = "one_oz_limit_canary"
GOLD_LIVE_MAX_COMMISSION_USD = 0.66
GOLD_LIVE_MAX_INITIAL_MARGIN_AUD = 600.0
GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD = 520.0
GOLD_LIVE_MAX_RUN_DRAWDOWN_USD = 700.0
GOLD_LIVE_MIN_STRESS_BUFFER_AUD = 300.0
GOLD_LIVE_FX_STRESS_BPS = 11_000
GOLD_LIVE_SOURCE_MAX_AGE_SECONDS = 10 * 60.0
GOLD_LIVE_QUOTE_MAX_AGE_SECONDS = 10.0
GOLD_LIVE_SELECTION_PATH = Path("db/calibration/gold_selected_live_transport.json")
GOLD_LIVE_LEDGER_PATH = Path("db/calibration/gold_live_calibration.jsonl")
GOLD_RUNTIME_PARITY_PATH = Path(
    "backtests/gold/one_oz_regime_harmony_runtime_parity_20260803.json"
)
GOLD_OPEN_POSITION_STRESS_PATH = Path(
    "backtests/gold/one_oz_stage76_open_position_stress_20260803.json"
)
GOLD_OPEN_POSITION_STRESS_SCHEMA = "gold.1oz-stage76-open-position-stress.v1"
GOLD_PACKAGE_SUCCESSOR_SCHEMA = "gold.1oz-portfolio-package-successor.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(value: object) -> str:
    return calibration_fingerprint(value)


def _number(value: object, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("gold live timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _open_target(result: object) -> dict[str, object] | None:
    trades = [
        trade
        for trade in getattr(result, "trades", ())
        if isinstance(trade, SpotTrade) and trade.exit_time is None
    ]
    if not trades:
        return None
    if len(trades) != 1:
        raise ValueError("gold replay projected multiple current targets")
    trade = trades[0]
    trace = trade.decision_trace if isinstance(trade.decision_trace, Mapping) else {}
    guard = trace.get("entry_guard_inputs")
    signal_bar = guard.get("signal_bar_ts") if isinstance(guard, Mapping) else None
    if signal_bar is None:
        raise ValueError("gold target lacks its causal signal bar")
    target = {
        "direction": "up" if int(trade.qty) > 0 else "down",
        "signal_bar_utc": gold_utc(signal_bar).isoformat(),
        "entry_time_utc": gold_utc(trade.entry_time).isoformat(),
        "signal_entry_price": round(float(trade.entry_price), 8),
        "mfe_usd": round(float(trade.max_favorable_excursion), 8),
        "mae_usd": round(float(trade.max_adverse_excursion), 8),
    }
    return {**target, "target_id": _identity(target)}


def advance_gold_regime_harmony_source(
    ledger: LiveCalibrationLedger,
    *,
    tape: GoldRegimeHarmonyTape,
    onset_context: Mapping[str, object],
    observed_at: datetime,
) -> dict[str, object]:
    """Cold-replay Stage 76 once and persist the exact current owner state."""

    now = _utc(observed_at)
    replay = GoldRegimeHarmonyReplay(tape)
    end = tape.as_of.date()
    result, owner, convergence, converged = replay.converged_window(
        GOLD_REGIME_HARMONY_SOURCE_START,
        end,
        source_start=GOLD_REGIME_HARMONY_SOURCE_START,
        source_end=end,
        final_session_complete=False,
    )
    pair = onset_context.get("exchange_parity")
    signal = onset_context.get("signal")
    try:
        decision_bar = (
            gold_utc(signal["decision_bar_end_utc"])
            if isinstance(signal, Mapping)
            else None
        )
    except (KeyError, TypeError, ValueError):
        decision_bar = None
    usable = bool(
        converged
        and isinstance(pair, Mapping)
        and pair.get("usable") is True
        and isinstance(signal, Mapping)
        and signal.get("usable") is True
        and decision_bar is not None
        and decision_bar <= now
    )
    target = _open_target(result)
    state = owner.state_payload(result)
    evidence = {
        "schema": "gold.1oz-regime-harmony-source-checkpoint.v1",
        "strategy_version": GOLD_REGIME_HARMONY_VERSION,
        "decision_bar_end_utc": (
            decision_bar.isoformat() if decision_bar is not None else None
        ),
        "target": target,
        "owner_state": state,
        "converged": converged,
        "convergence": convergence,
        "tape_rows": {
            "h1": len(tape.h1),
            "h4": len(tape.h4),
            "daily": len(tape.daily),
            "uup": len(tape.uup),
            "tip": len(tape.tip),
        },
        "signal_context": dict(signal) if isinstance(signal, Mapping) else None,
        "macro_context": onset_context.get("macro"),
        "fundamental_pressure": onset_context.get("news"),
        "contract_pair": dict(pair) if isinstance(pair, Mapping) else None,
        "source_points": onset_context.get("source_points"),
        "synthetic_midcycle_entry_authority": "none",
        "order_authority": "none",
        "submitted_orders": 0,
    }
    checkpoint = ledger.checkpoint(
        evaluation_as_of=now,
        strategy_id=GOLD_REGIME_HARMONY_VERSION,
        strategy_version=GOLD_REGIME_HARMONY_SOURCE_VERSION,
        trading_date=now.date().isoformat(),
        session="GOLD_24X5_SOURCE",
        status="EVALUATED" if usable else "NO_DATA",
        evidence=evidence,
        recorded_at=now,
    )
    return {
        "checkpoint": checkpoint,
        "target": target,
        "converged": converged,
        "submitted_orders": 0,
    }


def _account_value(
    ib: IB,
    account: str,
    tag: str,
    currency: str,
) -> float:
    row = next(
        (
            value
            for value in ib.accountValues(account)
            if value.tag == tag and str(value.currency or "").upper() == currency
        ),
        None,
    )
    return _number(getattr(row, "value", None), name=f"{tag} {currency}")


def gold_selection_preview(
    ib: IB,
    *,
    pair: Mapping[str, object],
    contracts: Mapping[int, object],
    observed_at: datetime,
) -> dict[str, object]:
    """Capture fresh books, account state, and two nontransmitting what-ifs."""

    now = _utc(observed_at)
    if pair.get("usable") is not True:
        raise ValueError("gold selection requires one usable shared-month pair")
    one = pair.get("one_oz")
    gc = pair.get("gc")
    if not isinstance(one, Mapping) or not isinstance(gc, Mapping):
        raise ValueError("gold selection pair is incomplete")
    contract = contracts.get(int(one.get("con_id") or 0))
    if contract is None:
        raise ValueError("gold selection contract is unavailable")
    accounts = ib.managedAccounts()
    if len(accounts) != 1:
        raise ValueError("gold selection requires one broker account")
    account = str(accounts[0])
    ib.sleep(0.5)
    account_type = next(
        (
            str(row.value or "").upper()
            for row in ib.accountValues(account)
            if row.tag == "TradingType-S"
        ),
        "",
    )
    account_values = {
        "settled_cash_usd": _account_value(ib, account, "CashBalance", "USD"),
        "equity_with_loan_aud": _account_value(
            ib, account, "EquityWithLoanValue", "AUD"
        ),
        "available_funds_aud": _account_value(ib, account, "AvailableFunds", "AUD"),
        "excess_liquidity_aud": _account_value(
            ib, account, "ExcessLiquidity", "AUD"
        ),
        "initial_margin_aud": _account_value(ib, account, "FullInitMarginReq", "AUD"),
        "maintenance_margin_aud": _account_value(
            ib, account, "FullMaintMarginReq", "AUD"
        ),
        "gross_position_value_aud": _account_value(
            ib, account, "GrossPositionValue", "AUD"
        ),
        "usd_to_aud": _account_value(ib, account, "ExchangeRate", "USD"),
    }
    positions = [
        {
            "symbol": str(item.contract.symbol or "").upper(),
            "local_symbol": str(item.contract.localSymbol or ""),
            "con_id": int(item.contract.conId or 0),
            "sec_type": str(item.contract.secType or ""),
            "quantity": float(item.position),
            "market_value_base": float(item.marketValue),
        }
        for item in ib.portfolio(account)
        if abs(float(item.position)) > 1e-9
    ]
    open_orders = [
        {
            "symbol": str(trade.contract.symbol or "").upper(),
            "con_id": int(trade.contract.conId or 0),
            "action": str(trade.order.action or "").upper(),
            "quantity": float(trade.order.totalQuantity or 0),
            "order_ref": str(trade.order.orderRef or ""),
            "status": str(trade.orderStatus.status or ""),
        }
        for trade in ib.openTrades()
    ]
    previews = []
    for action, limit in (("BUY", one["ask"]), ("SELL", one["bid"])):
        order = LimitOrder(
            action,
            1,
            float(limit),
            tif="GTC",
            account=account,
            outsideRth=True,
            orderRef=f"GOLD76-PREVIEW-{action}",
        )
        state = ib.whatIfOrder(contract, order)
        previews.append(
            {
                "action": action,
                "quantity": 1,
                "limit_price": float(limit),
                "status": str(state.status or ""),
                "initial_margin_change_aud": _number(
                    state.initMarginChange, name="initial margin change"
                ),
                "initial_margin_after_aud": _number(
                    state.initMarginAfter, name="initial margin after"
                ),
                "maintenance_margin_change_aud": _number(
                    state.maintMarginChange, name="maintenance margin change"
                ),
                "maintenance_margin_after_aud": _number(
                    state.maintMarginAfter, name="maintenance margin after"
                ),
                "equity_with_loan_after_aud": _number(
                    state.equityWithLoanAfter, name="equity with loan after"
                ),
                "commission_usd": _number(state.commission, name="commission"),
                "commission_currency": str(state.commissionCurrency or "").upper(),
                "warning_text": str(state.warningText or ""),
            }
        )
    detail_rows = ib.reqContractDetails(contract)
    detail = detail_rows[0] if detail_rows else None
    return {
        "schema": "gold.1oz-selection-preview.v1",
        "authority": "fresh_nontransmitting_what_if_only",
        "observed_at_utc": now.isoformat(),
        "account_id": account,
        "account_type": "CASH" if account_type == "STKCASH" else account_type,
        "base_currency": "AUD",
        "account_values": account_values,
        "pair": dict(pair),
        "contract": {
            "symbol": "1OZ",
            "local_symbol": str(getattr(contract, "localSymbol", "") or ""),
            "con_id": int(getattr(contract, "conId", 0) or 0),
            "expiry": str(
                getattr(contract, "lastTradeDateOrContractMonth", "") or ""
            ),
            "exchange": str(getattr(contract, "exchange", "") or ""),
            "currency": str(getattr(contract, "currency", "") or ""),
            "multiplier": str(getattr(contract, "multiplier", "") or ""),
            "min_tick": _number(getattr(detail, "minTick", None), name="minimum tick"),
        },
        "positions": positions,
        "open_orders": open_orders,
        "what_if": previews,
        "submitted_orders": 0,
    }


def _load_runtime_parity(root: Path) -> dict[str, object]:
    path = root / GOLD_RUNTIME_PARITY_PATH
    receipt = json.loads(path.read_text())
    owners = receipt.get("owners")
    if (
        receipt.get("verdict")
        != "SIGNAL_RUNTIME_PARITY_PASS_LIVE_TRANSPORT_HOLD"
        or not isinstance(owners, Mapping)
        or any(
            receipt.get("gates", {}).get(gate) != "PASS"
            for gate in (
                "machine_crown_identity",
                "shared_context_math",
                "full_three_year_ledger",
                "full_ten_year_ledger",
                "cold_replay_and_restart_identity",
                "flat_current_prefix",
            )
        )
        or any(
            _sha256(root / str(row.get("path") or "")) != row.get("sha256")
            for row in owners.values()
            if isinstance(row, Mapping)
        )
    ):
        raise ValueError("gold runtime parity receipt is invalid")
    return {"path": GOLD_RUNTIME_PARITY_PATH.as_posix(), "sha256": _sha256(path)}


def select_gold_live_transport(
    *,
    source_checkpoint: Mapping[str, object],
    preview: Mapping[str, object],
    selected_at_utc: datetime,
    root: Path | None = None,
) -> dict[str, object]:
    """Freeze one flat, previewed, content-addressed Stage-76 canary."""

    base = (root or Path(__file__).resolve().parents[2]).resolve()
    selected_at = _utc(selected_at_utc)
    evidence = source_checkpoint.get("evidence")
    source_at = _utc(source_checkpoint.get("recorded_at_utc"))
    preview_at = _utc(preview.get("observed_at_utc"))
    pair = preview.get("pair")
    one = pair.get("one_oz") if isinstance(pair, Mapping) else None
    gc = pair.get("gc") if isinstance(pair, Mapping) else None
    contract = preview.get("contract")
    what_if = preview.get("what_if")
    positions = preview.get("positions")
    open_orders = preview.get("open_orders")
    if (
        source_checkpoint.get("kind") != "checkpoint"
        or source_checkpoint.get("strategy_version")
        != GOLD_REGIME_HARMONY_SOURCE_VERSION
        or source_checkpoint.get("status") != "EVALUATED"
        or not isinstance(evidence, Mapping)
        or evidence.get("schema")
        != "gold.1oz-regime-harmony-source-checkpoint.v1"
        or evidence.get("target") is not None
        or evidence.get("synthetic_midcycle_entry_authority") != "none"
        or not 0 <= (selected_at - source_at).total_seconds() <= GOLD_LIVE_SOURCE_MAX_AGE_SECONDS
        or not 0 <= (selected_at - preview_at).total_seconds() <= 90.0
        or preview.get("schema") != "gold.1oz-selection-preview.v1"
        or preview.get("authority") != "fresh_nontransmitting_what_if_only"
        or preview.get("submitted_orders") != 0
        or preview.get("account_type") != "CASH"
        or preview.get("base_currency") != "AUD"
        or not str(preview.get("account_id") or "")
        or not isinstance(pair, Mapping)
        or pair.get("usable") is not True
        or not isinstance(one, Mapping)
        or not isinstance(gc, Mapping)
        or not isinstance(contract, Mapping)
        or int(one.get("con_id") or 0) != int(contract.get("con_id") or 0)
        or any(
            row.get("market_data_type") != 1
            or _number(row.get("age_seconds"), name="quote age")
            > GOLD_LIVE_QUOTE_MAX_AGE_SECONDS
            for row in (one, gc)
        )
        or not isinstance(what_if, Sequence)
        or isinstance(what_if, (str, bytes))
        or len(what_if) != 2
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or any(
            isinstance(row, Mapping)
            and str(row.get("symbol") or "").upper() == "1OZ"
            and abs(_number(row.get("quantity"), name="1OZ position")) > 1e-9
            for row in positions
        )
        or open_orders != []
    ):
        raise ValueError("gold selection requires a fresh flat source and broker proof")
    keyed = {
        str(row.get("action") or "").upper(): row
        for row in what_if
        if isinstance(row, Mapping)
    }
    if set(keyed) != {"BUY", "SELL"} or any(
        row.get("status") != "PreSubmitted"
        or row.get("commission_currency") != "USD"
        or _number(row.get("commission_usd"), name="commission")
        > GOLD_LIVE_MAX_COMMISSION_USD
        or _number(row.get("initial_margin_change_aud"), name="initial margin")
        > GOLD_LIVE_MAX_INITIAL_MARGIN_AUD
        or _number(
            row.get("maintenance_margin_change_aud"),
            name="maintenance margin",
        )
        > GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD
        or str(row.get("warning_text") or "")
        for row in keyed.values()
    ):
        raise ValueError("gold broker what-if exceeds the canary boundary")
    crown = load_gold_regime_harmony_crown(root=base)
    parity = _load_runtime_parity(base)
    artifact = json.loads((base / str(crown["artifact_path"])).read_text())
    native = artifact.get("native_transport")
    if (
        not isinstance(native, Mapping)
        or native.get("mapping_failures") != 0
        or _number(native.get("max_intrabar_drawdown"), name="native drawdown")
        >= GOLD_LIVE_MAX_RUN_DRAWDOWN_USD
    ):
        raise ValueError("gold native transport does not fit the canary risk boundary")
    body = {
        "schema": GOLD_LIVE_SELECTION_SCHEMA,
        "selected_at_utc": selected_at.isoformat(),
        "run_started_at_utc": selected_at.isoformat(),
        "strategy_version": GOLD_REGIME_HARMONY_VERSION,
        "source_strategy_version": GOLD_REGIME_HARMONY_SOURCE_VERSION,
        "execution_strategy_version": GOLD_LIVE_EXECUTION_VERSION,
        "authority": "selected_canary_live_margin_transport",
        "order_authority": GOLD_LIVE_ORDER_AUTHORITY,
        "profitability_clock_started": True,
        "capital_sleeve": GOLD_LIVE_CAPITAL_SLEEVE,
        "quantity": 1,
        "baseline_target": None,
        "synthetic_midcycle_entry_authority": "none",
        "contract": dict(contract),
        "corroborating_gc": {
            key: gc.get(key)
            for key in ("local_symbol", "con_id", "expiry", "contract_month")
        },
        "execution": {
            "order_type": "LMT",
            "time_in_force": "GTC",
            "initial_mode": "OPTIMISTIC",
            "chase_mode": "AUTO",
            "close_before_reverse": True,
            "entry_authority": "first_post_selection_stage76_admission_only",
            "transport_hours": "1OZ_24x7_subject_to_exchange_maintenance",
            "signal_hours": "XAUUSD_GC_24x5",
            "fresh_streaming_nbbo_required": True,
        },
        "risk": {
            "max_contracts": 1,
            "max_commission_usd_per_order": GOLD_LIVE_MAX_COMMISSION_USD,
            "max_initial_margin_change_aud": GOLD_LIVE_MAX_INITIAL_MARGIN_AUD,
            "max_maintenance_margin_change_aud": (
                GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            ),
            "max_run_drawdown_usd": GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
            "historical_native_intrabar_drawdown_usd": native[
                "max_intrabar_drawdown"
            ],
            "minimum_post_stress_excess_liquidity_aud": (
                GOLD_LIVE_MIN_STRESS_BUFFER_AUD
            ),
            "fx_stress_bps": GOLD_LIVE_FX_STRESS_BPS,
            "max_concurrent_directional_sleeves": 1,
        },
        "broker_at_selection": {
            key: preview[key]
            for key in (
                "observed_at_utc",
                "account_id",
                "account_type",
                "base_currency",
                "account_values",
                "positions",
                "open_orders",
                "pair",
                "what_if",
            )
        },
        "evidence": {
            "crown": crown,
            "runtime_parity": parity,
            "source_checkpoint_id": source_checkpoint["checkpoint_id"],
            "source_recorded_at_utc": source_checkpoint["recorded_at_utc"],
            "source_state_sha256": evidence["owner_state"]["state_sha256"],
            "native_transport_receipt_sha256": native["receipt_sha256"],
            "official_1oz_24x7_notice": (
                "https://www.cmegroup.com/notices/electronic-trading/2026/07/"
                "20260713.html"
            ),
        },
    }
    return {**body, "selection_id": _identity(body)}


def load_gold_live_selection_from_mapping(
    value: Mapping[str, object],
) -> dict[str, object]:
    selection = dict(value)
    selection_id = str(selection.pop("selection_id", ""))
    body = dict(selection)
    contract = selection.get("contract")
    execution = selection.get("execution")
    risk = selection.get("risk")
    broker = selection.get("broker_at_selection")
    evidence = selection.get("evidence")
    if (
        selection.get("schema") not in GOLD_LIVE_SELECTION_SCHEMAS
        or selection_id != _identity(body)
        or selection.get("strategy_version") != GOLD_REGIME_HARMONY_VERSION
        or selection.get("source_strategy_version")
        != GOLD_REGIME_HARMONY_SOURCE_VERSION
        or selection.get("execution_strategy_version") != GOLD_LIVE_EXECUTION_VERSION
        or selection.get("authority") != "selected_canary_live_margin_transport"
        or selection.get("order_authority") != GOLD_LIVE_ORDER_AUTHORITY
        or selection.get("profitability_clock_started") is not True
        or selection.get("capital_sleeve") != GOLD_LIVE_CAPITAL_SLEEVE
        or selection.get("quantity") != 1
        or selection.get("baseline_target") is not None
        or selection.get("synthetic_midcycle_entry_authority") != "none"
        or not isinstance(contract, Mapping)
        or contract.get("symbol") != "1OZ"
        or contract.get("exchange") != "COMEX"
        or contract.get("currency") != "USD"
        or contract.get("multiplier") != "1"
        or int(contract.get("con_id") or 0) <= 0
        or not isinstance(execution, Mapping)
        or execution.get("entry_authority")
        != "first_post_selection_stage76_admission_only"
        or not isinstance(risk, Mapping)
        or risk.get("max_contracts") != 1
        or risk.get("max_run_drawdown_usd") != GOLD_LIVE_MAX_RUN_DRAWDOWN_USD
        or not isinstance(broker, Mapping)
        or broker.get("account_type") != "CASH"
        or broker.get("base_currency") != "AUD"
        or not isinstance(evidence, Mapping)
    ):
        raise ValueError("gold selected canary identity is invalid")
    if selection["schema"] == GOLD_LIVE_SELECTION_SCHEMA:
        if (
            risk.get("max_concurrent_directional_sleeves") != 1
            or "allocation_successor" in selection
        ):
            raise ValueError("gold initial allocation identity is invalid")
    else:
        successor = selection.get("allocation_successor")
        stress = evidence.get("open_position_stress")
        positions = broker.get("positions")
        expected_risk = {
            "max_contracts": 1,
            "max_commission_usd_per_order": GOLD_LIVE_MAX_COMMISSION_USD,
            "max_initial_margin_change_aud": GOLD_LIVE_MAX_INITIAL_MARGIN_AUD,
            "max_maintenance_margin_change_aud": (
                GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            ),
            "max_run_drawdown_usd": GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
            "historical_native_intrabar_drawdown_usd": risk.get(
                "historical_native_intrabar_drawdown_usd"
            ),
            "max_open_position_stress_usd": 256.16,
            "minimum_post_stress_excess_liquidity_aud": (
                GOLD_LIVE_MIN_STRESS_BUFFER_AUD
            ),
            "fx_stress_bps": GOLD_LIVE_FX_STRESS_BPS,
        }
        if (
            risk != expected_risk
            or not isinstance(successor, Mapping)
            or successor.get("schema") != GOLD_PACKAGE_SUCCESSOR_SCHEMA
            or successor.get("predecessor_schema") not in GOLD_LIVE_SELECTION_SCHEMAS
            or not _sha256_identity(successor.get("predecessor_selection_id"))
            or not _sha256_identity(
                successor.get("predecessor_fill_ledger_fingerprint")
            )
            or not _sha256_identity(
                successor.get("predecessor_risk_state_fingerprint")
            )
            or not _sha256_identity(successor.get("broker_preview_fingerprint"))
            or successor.get("package_id") != "gold-one-contract"
            or successor.get("package_cash_debit_usd_cents") != 66
            or not isinstance(stress, Mapping)
            or stress.get("path") != GOLD_OPEN_POSITION_STRESS_PATH.as_posix()
            or not _sha256_identity(stress.get("sha256"))
            or stress.get("max_single_position_mae_usd") != 256.16
            or broker.get("open_orders") != []
            or not isinstance(positions, Sequence)
            or isinstance(positions, (str, bytes))
            or any(
                isinstance(row, Mapping)
                and str(row.get("symbol") or "").upper() == "1OZ"
                and abs(_number(row.get("quantity"), name="1OZ position")) > 1e-9
                for row in positions
            )
        ):
            raise ValueError("gold package allocation identity is invalid")
    return dict(value)


def _sha256_identity(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _pending_gold_order_refs(
    records: Sequence[Mapping[str, object]], *, selection_id: str
) -> list[str]:
    latest: dict[str, str] = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("strategy_version") != GOLD_LIVE_EXECUTION_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("selection_id") != selection_id
        ):
            continue
        order_ref = str(evidence.get("order_ref") or "")
        phase = str(evidence.get("phase") or "")
        if order_ref and phase in {"PREPARED", "SUBMITTED", "TERMINAL"}:
            latest[order_ref] = phase
    return sorted(ref for ref, phase in latest.items() if phase != "TERMINAL")


def reallocate_gold_live_transport(
    *,
    predecessor: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    preview: Mapping[str, object],
    selected_at_utc: datetime,
    stress_receipt_path: Path,
) -> dict[str, object]:
    """Freeze one clean Gold run without the retired account-wide mutex."""

    from .gold_live_state import gold_transport_risk_state

    prior = load_gold_live_selection_from_mapping(predecessor)
    selected_at = _utc(selected_at_utc)
    preview_at = _utc(preview.get("observed_at_utc"))
    pair = preview.get("pair")
    one = pair.get("one_oz") if isinstance(pair, Mapping) else None
    gc = pair.get("gc") if isinstance(pair, Mapping) else None
    contract = preview.get("contract")
    what_if = preview.get("what_if")
    keyed_what_if = (
        {
            str(row.get("action") or "").upper(): row
            for row in what_if
            if isinstance(row, Mapping)
        }
        if isinstance(what_if, Sequence) and not isinstance(what_if, (str, bytes))
        else {}
    )
    positions = preview.get("positions")
    stress = json.loads(stress_receipt_path.read_text())
    if not isinstance(stress, Mapping):
        raise ValueError("gold open-position stress receipt must be one object")
    risk_state = gold_transport_risk_state(
        selection=prior,
        records=records,
        observed_at=selected_at,
        liquidation_price=_number(
            one.get("bid"), name="1OZ liquidation price"
        ) if isinstance(one, Mapping) else 0,
    )
    if (
        preview.get("schema") != "gold.1oz-selection-preview.v1"
        or preview.get("authority") != "fresh_nontransmitting_what_if_only"
        or preview.get("submitted_orders") != 0
        or preview.get("account_id") != prior["broker_at_selection"]["account_id"]
        or preview.get("account_type") != "CASH"
        or preview.get("base_currency") != "AUD"
        or not 0 <= (selected_at - preview_at).total_seconds() <= 90
        or not isinstance(pair, Mapping)
        or pair.get("usable") is not True
        or not isinstance(one, Mapping)
        or not isinstance(gc, Mapping)
        or not isinstance(contract, Mapping)
        or int(contract.get("con_id") or 0) != int(one.get("con_id") or 0)
        or not isinstance(what_if, Sequence)
        or isinstance(what_if, (str, bytes))
        or len(what_if) != 2
        or set(keyed_what_if) != {"BUY", "SELL"}
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "PreSubmitted"
            or row.get("commission_currency") != "USD"
            or _number(row.get("commission_usd"), name="gold commission")
            > GOLD_LIVE_MAX_COMMISSION_USD
            or _number(row.get("initial_margin_change_aud"), name="gold initial margin")
            > GOLD_LIVE_MAX_INITIAL_MARGIN_AUD
            or _number(
                row.get("maintenance_margin_change_aud"),
                name="gold maintenance margin",
            )
            > GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            or str(row.get("warning_text") or "")
            for row in keyed_what_if.values()
        )
        or not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or any(
            isinstance(row, Mapping)
            and str(row.get("symbol") or "").upper() == "1OZ"
            and abs(_number(row.get("quantity"), name="1OZ position")) > 1e-9
            for row in positions
        )
        or preview.get("open_orders") != []
        or _pending_gold_order_refs(records, selection_id=str(prior["selection_id"]))
        or risk_state["position_from_fills"] != 0
        or risk_state["safety_breaches"]
        or stress.get("schema") != GOLD_OPEN_POSITION_STRESS_SCHEMA
        or stress.get("capital_authority") != "none"
        or stress.get("submitted_orders") != 0
        or stress.get("harsher_cost_neighbour", {}).get(
            "max_single_position_mae_usd"
        )
        != 256.16
    ):
        raise ValueError("gold package successor requires flat, previewed broker truth")
    successor = {
        "schema": GOLD_PACKAGE_SUCCESSOR_SCHEMA,
        "predecessor_schema": prior["schema"],
        "predecessor_selection_id": prior["selection_id"],
        "predecessor_run_started_at_utc": prior["run_started_at_utc"],
        "predecessor_fill_ledger_fingerprint": risk_state[
            "fill_ledger_fingerprint"
        ],
        "predecessor_risk_state_fingerprint": _identity(risk_state),
        "predecessor_realized_net_usd": risk_state["run_realized_net_usd"],
        "predecessor_closed_trades": risk_state["closed_trades"],
        "package_id": "gold-one-contract",
        "package_cash_debit_usd_cents": 66,
        "broker_preview_fingerprint": _identity(preview),
    }
    evidence = json.loads(json.dumps(prior["evidence"]))
    evidence["open_position_stress"] = {
        "path": GOLD_OPEN_POSITION_STRESS_PATH.as_posix(),
        "sha256": _sha256(stress_receipt_path),
        "max_single_position_mae_usd": 256.16,
    }
    prior_risk = prior["risk"]
    body = {
        **{
            key: json.loads(json.dumps(item))
            for key, item in prior.items()
            if key
            not in {
                "selection_id",
                "schema",
                "selected_at_utc",
                "run_started_at_utc",
                "contract",
                "corroborating_gc",
                "broker_at_selection",
                "risk",
                "evidence",
                "allocation_successor",
            }
        },
        "schema": GOLD_LIVE_PACKAGE_SELECTION_SCHEMA,
        "selected_at_utc": selected_at.isoformat(),
        "run_started_at_utc": selected_at.isoformat(),
        "contract": dict(contract),
        "corroborating_gc": {
            key: gc.get(key)
            for key in ("local_symbol", "con_id", "expiry", "contract_month")
        },
        "broker_at_selection": {
            key: preview[key]
            for key in (
                "observed_at_utc",
                "account_id",
                "account_type",
                "base_currency",
                "account_values",
                "positions",
                "open_orders",
                "pair",
                "what_if",
            )
        },
        "risk": {
            "max_contracts": 1,
            "max_commission_usd_per_order": GOLD_LIVE_MAX_COMMISSION_USD,
            "max_initial_margin_change_aud": GOLD_LIVE_MAX_INITIAL_MARGIN_AUD,
            "max_maintenance_margin_change_aud": (
                GOLD_LIVE_MAX_MAINTENANCE_MARGIN_AUD
            ),
            "max_run_drawdown_usd": GOLD_LIVE_MAX_RUN_DRAWDOWN_USD,
            "historical_native_intrabar_drawdown_usd": prior_risk[
                "historical_native_intrabar_drawdown_usd"
            ],
            "max_open_position_stress_usd": 256.16,
            "minimum_post_stress_excess_liquidity_aud": (
                GOLD_LIVE_MIN_STRESS_BUFFER_AUD
            ),
            "fx_stress_bps": GOLD_LIVE_FX_STRESS_BPS,
        },
        "evidence": evidence,
        "allocation_successor": successor,
    }
    selected = {**body, "selection_id": _identity(body)}
    return load_gold_live_selection_from_mapping(selected)


def load_gold_live_selection(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("gold selection must be one JSON object")
    return load_gold_live_selection_from_mapping(value)


def publish_gold_live_selection(
    path: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    selected = load_gold_live_selection_from_mapping(selection)
    payload = json.dumps(selected, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        existing = load_gold_live_selection(path)
        if existing != selected:
            raise ValueError("gold selection already contains another immutable run")
        return existing
    _atomic_write(path, payload)
    return selected


def build_gold_portfolio_capital_plan(
    selection: Mapping[str, object],
    *,
    selection_path: Path,
    current_plan: Mapping[str, object],
) -> dict[str, object]:
    """Promote v1 XSP cash plus one exclusive 1OZ margin overlay to v2."""

    selected = load_gold_live_selection_from_mapping(selection)
    predecessor = validate_live_capital_plan(current_plan)
    if predecessor.get("schema") != "live.capital-plan.v1":
        raise ValueError("first gold allocation must supersede the XSP v1 plan")
    cash_sleeves = predecessor.get("sleeves")
    if not isinstance(cash_sleeves, Sequence) or len(cash_sleeves) != 1:
        raise ValueError("gold allocation requires the one selected XSP cash sleeve")
    cash = dict(cash_sleeves[0])
    cash["position_symbols"] = ["SPXU", "UPRO"]
    broker = selected["broker_at_selection"]
    risk = selected["risk"]
    assert isinstance(broker, Mapping) and isinstance(risk, Mapping)
    account_values = broker.get("account_values")
    if not isinstance(account_values, Mapping):
        raise ValueError("gold selection has no account values")
    selection_sha = _sha256(selection_path)
    return build_live_capital_plan_v2(
        account_id=str(broker["account_id"]),
        account_type="CASH",
        cash_currency="USD",
        base_currency="AUD",
        observed_settled_cash_usd=account_values["settled_cash_usd"],
        managed_capital_usd=(
            int(predecessor["capital"]["managed_capital_cents"]) / 100
        ),
        sleeves=[
            cash,
            {
                "sleeve_id": GOLD_LIVE_CAPITAL_SLEEVE,
                "strategy_id": GOLD_REGIME_HARMONY_VERSION,
                "run_id": selected["selection_id"],
                "selection_path": selection_path.as_posix(),
                "selection_file_sha256": selection_sha,
                "capital_kind": "FUTURES_MARGIN",
                "weight_bps": 0,
                "position_symbols": ["1OZ"],
                "margin": {
                    "base_currency": "AUD",
                    "max_contracts": 1,
                    "max_initial_margin_change_cents": int(
                        float(risk["max_initial_margin_change_aud"]) * 100
                    ),
                    "max_maintenance_margin_change_cents": int(
                        float(risk["max_maintenance_margin_change_aud"]) * 100
                    ),
                    "max_stressed_loss_usd_cents": int(
                        float(risk["max_run_drawdown_usd"]) * 100
                    ),
                    "fx_stress_bps": int(risk["fx_stress_bps"]),
                    "minimum_post_stress_excess_liquidity_cents": int(
                        float(risk["minimum_post_stress_excess_liquidity_aud"])
                        * 100
                    ),
                },
            },
        ],
        reserve_reasons=[
            *predecessor["capital"]["reserve_reasons"],
            "futures_margin_is_account_equity_not_cash_reserve",
            "single_directional_sleeve_exposure_mutex",
        ],
        created_at_utc=str(selected["selected_at_utc"]),
        max_concurrent_directional_sleeves=1,
        supersedes_plan_id=str(predecessor["plan_id"]),
    )
