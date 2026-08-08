"""One fail-closed IBKR readiness receipt for every writable strategy owner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import Contract, IB

from ..config import load_config
from .capital import load_live_capital_plan


IB_PREFLIGHT_SCHEMA = "live.ib-preflight.v1"
IB_PREFLIGHT_AUTHORITY = "read_only_broker_and_runtime_readiness"
IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC = 30 * 60 * 60
IB_PREFLIGHT_BOUNDARIES = {
    "broker_orders_submitted": 0,
    "broker_orders_cancelled": 0,
    "gateway_restarted": False,
    "runtime_units_mutated": False,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _aware_utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("IB preflight timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def reduce_ib_preflight(
    facts: Mapping[str, object],
    *,
    checked_at_utc: datetime | str,
) -> dict[str, object]:
    """Reduce read-only facts into distinct entry and reduction readiness."""

    checked = _aware_utc(checked_at_utc)
    gateway = facts.get("gateway")
    broker = facts.get("broker")
    connectivity = facts.get("connectivity")
    runtime = facts.get("runtime")
    capabilities = facts.get("capabilities")
    if (
        not isinstance(gateway, Mapping)
        or not isinstance(broker, Mapping)
        or not isinstance(connectivity, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(capabilities, Sequence)
        or isinstance(capabilities, (str, bytes))
        or any(not isinstance(capability, Mapping) for capability in capabilities)
    ):
        raise ValueError("IB preflight facts are incomplete")

    shared: list[str] = []
    if gateway.get("port_accepting") is not True:
        shared.append("gateway_port_unavailable")
    if gateway.get("api_authenticated") is not True:
        shared.append("broker_api_not_authenticated")
    if gateway.get("expected_account_returned") is not True:
        shared.append("expected_account_not_returned")
    if broker.get("positions_fresh") is not True:
        shared.append("broker_positions_not_fresh")
    if broker.get("open_orders_fresh") is not True:
        shared.append("broker_open_orders_not_fresh")

    reduction_reasons = list(shared)
    if broker.get("reduction_quote_ready") is not True:
        reduction_reasons.append("held_position_quote_not_ready")

    entry_reasons = list(shared)
    if not capabilities:
        entry_reasons.append("required_capabilities_missing")
    entry_reasons.extend(
        f"required_capability_unhealthy:{capability.get('label')}"
        for capability in capabilities
        if capability.get("healthy") is not True
    )
    if connectivity.get("unpaired_1100") is True:
        entry_reasons.append("connectivity_loss_unpaired")
    if connectivity.get("losses_10m", 0) >= 3:
        entry_reasons.append("connectivity_flap_storm")
    missing_members = runtime.get("missing_members")
    if not isinstance(missing_members, Sequence) or isinstance(missing_members, (str, bytes)):
        entry_reasons.append("runtime_membership_unknown")
    else:
        entry_reasons.extend(f"runtime_member_not_armed:{unit}" for unit in missing_members)

    entry_reasons = sorted(set(entry_reasons))
    reduction_reasons = sorted(set(reduction_reasons))
    body = {
        "schema": IB_PREFLIGHT_SCHEMA,
        "authority": IB_PREFLIGHT_AUTHORITY,
        "checked_at_utc": checked.isoformat(),
        "facts": dict(facts),
        "verdict": {
            "entry_ready": not entry_reasons,
            "reduction_ready": not reduction_reasons,
            "entry_reasons": entry_reasons,
            "reduction_reasons": reduction_reasons,
        },
        "boundaries": dict(IB_PREFLIGHT_BOUNDARIES),
    }
    return {**body, "receipt_id": _identity(body)}


def validate_ib_preflight(value: Mapping[str, object]) -> dict[str, object]:
    frozen = dict(value)
    receipt_id = str(frozen.pop("receipt_id", ""))
    if (
        frozen.get("schema") != IB_PREFLIGHT_SCHEMA
        or frozen.get("authority") != IB_PREFLIGHT_AUTHORITY
        or frozen.get("boundaries") != IB_PREFLIGHT_BOUNDARIES
        or not isinstance(frozen.get("facts"), Mapping)
        or not isinstance(frozen.get("verdict"), Mapping)
        or receipt_id != _identity(frozen)
    ):
        raise ValueError("invalid IB preflight receipt")
    rebuilt = reduce_ib_preflight(
        frozen["facts"],
        checked_at_utc=str(frozen["checked_at_utc"]),
    )
    if rebuilt != value:
        raise ValueError("IB preflight receipt is not canonical")
    return dict(value)


def publish_ib_preflight(path: Path, receipt: Mapping[str, object]) -> None:
    frozen = validate_ib_preflight(receipt)
    payload = json.dumps(frozen, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_ib_preflight(
    path: Path,
    *,
    now: datetime | None = None,
    max_age_sec: float = IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("IB preflight receipt must be one JSON object")
    receipt = validate_ib_preflight(value)
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (observed - _aware_utc(str(receipt["checked_at_utc"]))).total_seconds()
    if age < -60 or age > max(1.0, float(max_age_sec)):
        raise ValueError("IB preflight receipt is stale")
    return receipt


def _configured_receipt_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return path
    raw = os.getenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", "").strip()
    return Path(raw).expanduser() if raw else None


def ib_preflight_configured(path: Path | None = None) -> bool:
    return _configured_receipt_path(path) is not None


def ib_preflight_decision(
    mode: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    normalized = str(mode).strip().lower()
    if normalized not in {"entry", "reduction"}:
        raise ValueError("IB preflight mode must be entry or reduction")
    configured = _configured_receipt_path(path)
    if configured is None:
        return {"configured": False, "ready": True, "reasons": []}
    try:
        max_age = float(
            os.getenv(
                "TRADEBOT_IB_PREFLIGHT_MAX_AGE_SEC",
                str(IB_PREFLIGHT_DEFAULT_MAX_AGE_SEC),
            )
        )
        receipt = load_ib_preflight(configured, now=now, max_age_sec=max_age)
        verdict = receipt["verdict"]
        assert isinstance(verdict, Mapping)
        reasons = list(verdict[f"{normalized}_reasons"])
        return {
            "configured": True,
            "ready": verdict[f"{normalized}_ready"] is True,
            "reasons": reasons,
            "receipt_id": receipt["receipt_id"],
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return {
            "configured": True,
            "ready": False,
            "reasons": [f"ib_preflight_unavailable:{exc}"],
        }


def gate_actionable_plan(
    plan: Mapping[str, object],
    *,
    reduction: bool,
    path: Path | None = None,
) -> dict[str, object]:
    if plan.get("status") != "ACTIONABLE":
        return dict(plan)
    mode = "reduction" if reduction else "entry"
    decision = ib_preflight_decision(mode, path=path)
    if decision["ready"] is True:
        return dict(plan)
    return {
        **dict(plan),
        "status": "HOLD",
        "reason": f"ib_preflight_{mode}_not_ready",
        "leg": None,
        "ib_preflight": decision,
    }


def require_reduction_preflight(path: Path | None = None) -> None:
    decision = ib_preflight_decision("reduction", path=path)
    if decision["ready"] is not True:
        raise RuntimeError(", ".join(str(reason) for reason in decision["reasons"]))


def order_preflight_mode(*, position: float, action: str, quantity: float) -> str:
    """Classify an exact single-contract order against current broker position."""

    side = str(action or "").strip().upper()
    try:
        held = float(position)
        size = float(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError("order preflight position and quantity must be numeric") from exc
    if not math.isfinite(held) or not math.isfinite(size) or size <= 0:
        raise ValueError("order preflight position and quantity must be finite")
    reduction = (
        (held > 0 and side == "SELL" and size <= held + 1e-9)
        or (held < 0 and side == "BUY" and size <= abs(held) + 1e-9)
    )
    return "reduction" if reduction else "entry"


def require_order_preflight(
    *,
    position: float,
    action: str,
    quantity: float,
    path: Path | None = None,
) -> dict[str, object]:
    """Require the appropriate receipt immediately before broker submission."""

    mode = order_preflight_mode(
        position=position,
        action=action,
        quantity=quantity,
    )
    decision = ib_preflight_decision(mode, path=path)
    if decision["ready"] is not True:
        reasons = ", ".join(str(reason) for reason in decision["reasons"])
        raise RuntimeError(f"IB preflight blocked {mode} order: {reasons}")
    return {**decision, "mode": mode}


def _contract_specs(plan: Mapping[str, object], root: Path) -> list[dict[str, object]]:
    specs: dict[int, dict[str, object]] = {}
    for sleeve in plan.get("sleeves", ()):
        if not isinstance(sleeve, Mapping):
            continue
        relative = Path(str(sleeve.get("selection_path") or ""))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("IB preflight selection path escaped the repository")
        selection = json.loads(path.read_text())
        if not isinstance(selection, Mapping):
            raise ValueError("IB preflight selection is invalid")
        candidates: list[tuple[str, object]] = []
        direct = selection.get("contract")
        if isinstance(direct, Mapping):
            candidates.append((str(direct.get("symbol") or sleeve.get("strategy_id")), direct))
        contracts = selection.get("contracts")
        if isinstance(contracts, Mapping):
            candidates.extend((str(label), value) for label, value in contracts.items())
        nominee = selection.get("nominee")
        contract_ids = nominee.get("contract_ids") if isinstance(nominee, Mapping) else None
        if isinstance(contract_ids, Mapping):
            candidates.extend((str(label), {"con_id": value}) for label, value in contract_ids.items())
        for label, candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            con_id = int(candidate.get("con_id", candidate.get("conId", 0)) or 0)
            if con_id > 0:
                specs[con_id] = {"label": label.upper(), "con_id": con_id}
    return [specs[key] for key in sorted(specs)]


def _required_runtime_members(plan: Mapping[str, object]) -> list[str]:
    from .strategies import LIVE_STRATEGY_BINDINGS

    bindings = {binding.strategy_id: binding for binding in LIVE_STRATEGY_BINDINGS}
    members = {
        unit
        for sleeve in plan.get("sleeves", ())
        if isinstance(sleeve, Mapping)
        for binding in (bindings.get(str(sleeve.get("strategy_id") or "")),)
        if binding is not None
        for unit in binding.runtime_timer_units
    }
    return sorted(members)


def _unit_armed(unit: str) -> bool:
    enabled = subprocess.run(
        ["systemctl", "--user", "is-enabled", unit],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    active = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return enabled.returncode == 0 and active.returncode == 0


def _connectivity_facts() -> dict[str, object]:
    try:
        journal = subprocess.run(
            ["journalctl", "--user", "--since", "-10 min", "--no-pager", "-o", "cat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        journal = ""
    codes = [int(code) for code in re.findall(r"(?<!\d)(1100|1101|1102)(?!\d)", journal)]
    losses = sum(code == 1100 for code in codes)
    last_loss = max((index for index, code in enumerate(codes) if code == 1100), default=-1)
    last_restore = max((index for index, code in enumerate(codes) if code in {1101, 1102}), default=-1)
    return {
        "losses_10m": losses,
        "restores_10m": sum(code in {1101, 1102} for code in codes),
        "unpaired_1100": last_loss > last_restore,
    }


def _price_ready(ticker: object) -> bool:
    return any(
        isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
        for value in (
            getattr(ticker, "bid", None),
            getattr(ticker, "ask", None),
            getattr(ticker, "last", None),
            getattr(ticker, "close", None),
        )
    )


async def probe_ib_preflight(
    *,
    repository_root: Path,
    capital_plan_path: Path,
) -> dict[str, object]:
    """Perform one bounded, read-only API/account/contract/runtime probe."""

    checked = datetime.now(timezone.utc)
    root = repository_root.resolve()
    plan = load_live_capital_plan(capital_plan_path)
    account = plan.get("account")
    expected_account = str(account.get("account_id") or "") if isinstance(account, Mapping) else ""
    config = load_config()
    port_accepting = False
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(config.host, config.port),
            timeout=min(5.0, config.connect_timeout_sec),
        )
        del reader
        port_accepting = True
        writer.close()
        await writer.wait_closed()
    except (OSError, asyncio.TimeoutError):
        pass

    ib = IB()
    authenticated = positions_fresh = open_orders_fresh = False
    managed_accounts: list[str] = []
    positions: list[dict[str, object]] = []
    open_orders: list[dict[str, object]] = []
    capabilities = [
        {**spec, "healthy": False} for spec in _contract_specs(plan, root)
    ]
    reduction_quote_ready = False
    try:
        client_id = int(os.getenv("IBKR_PREFLIGHT_CLIENT_ID", "3997"))
        await ib.connectAsync(
            config.host,
            config.port,
            clientId=client_id,
            timeout=config.connect_timeout_sec,
            readonly=True,
            account=expected_account,
        )
        authenticated = ib.isConnected()
        managed_accounts = sorted(str(value) for value in ib.managedAccounts())
        raw_positions = list(ib.positions(expected_account))
        positions = [
            {
                "account": str(row.account),
                "con_id": int(row.contract.conId or 0),
                "symbol": str(row.contract.symbol or row.contract.localSymbol or ""),
                "quantity": float(row.position),
            }
            for row in raw_positions
        ]
        positions_fresh = True
        trades = await asyncio.wait_for(ib.reqAllOpenOrdersAsync(), timeout=10)
        open_orders = [
            {
                "order_id": int(trade.order.orderId or 0),
                "perm_id": int(trade.order.permId or 0),
                "con_id": int(trade.contract.conId or 0),
                "symbol": str(trade.contract.symbol or trade.contract.localSymbol or ""),
                "status": str(trade.orderStatus.status or ""),
            }
            for trade in trades
        ]
        open_orders_fresh = True

        probe_contracts = [Contract(conId=int(row["con_id"])) for row in capabilities]
        qualified = (
            list(await asyncio.wait_for(ib.qualifyContractsAsync(*probe_contracts), timeout=15))
            if probe_contracts
            else []
        )
        qualified_ids = {int(contract.conId or 0) for contract in qualified}
        quoted_ids: set[int] = set()
        if qualified:
            try:
                tickers = await asyncio.wait_for(
                    ib.reqTickersAsync(*qualified),
                    timeout=15,
                )
                quoted_ids = {
                    int(getattr(ticker.contract, "conId", 0) or 0)
                    for ticker in tickers
                    if _price_ready(ticker)
                }
            except (OSError, RuntimeError, asyncio.TimeoutError, ValueError):
                pass
        capabilities = [
            {
                **row,
                "qualified": int(row["con_id"]) in qualified_ids,
                "market_data_ready": int(row["con_id"]) in quoted_ids,
                "healthy": int(row["con_id"]) in qualified_ids
                and int(row["con_id"]) in quoted_ids,
            }
            for row in capabilities
        ]
        held_contracts = [row.contract for row in raw_positions if abs(float(row.position)) > 1e-9]
        if held_contracts:
            try:
                tickers = await asyncio.wait_for(
                    ib.reqTickersAsync(*held_contracts),
                    timeout=15,
                )
                reduction_quote_ready = len(tickers) == len(held_contracts) and all(
                    _price_ready(ticker) for ticker in tickers
                )
            except (OSError, RuntimeError, asyncio.TimeoutError, ValueError):
                pass
        else:
            reduction_quote_ready = True
    except (OSError, RuntimeError, asyncio.TimeoutError, ValueError):
        pass
    finally:
        if ib.isConnected():
            ib.disconnect()

    expected_members = _required_runtime_members(plan)
    armed_members = [unit for unit in expected_members if _unit_armed(unit)]
    facts = {
        "gateway": {
            "host": config.host,
            "port": config.port,
            "port_accepting": port_accepting,
            "api_authenticated": authenticated,
            "expected_account": expected_account,
            "managed_accounts": managed_accounts,
            "expected_account_returned": expected_account in managed_accounts,
        },
        "broker": {
            "positions_fresh": positions_fresh,
            "open_orders_fresh": open_orders_fresh,
            "positions": positions,
            "open_orders": open_orders,
            "reduction_quote_ready": reduction_quote_ready,
        },
        "capabilities": capabilities,
        "connectivity": _connectivity_facts(),
        "runtime": {
            "expected_members": expected_members,
            "armed_members": armed_members,
            "missing_members": sorted(set(expected_members) - set(armed_members)),
        },
    }
    return reduce_ib_preflight(facts, checked_at_utc=checked)


def _unit_liveness(unit: str) -> dict[str, str]:
    """Read systemd state only; the sentinel never changes a runtime unit."""

    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "UnitFileState",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "NextElapseUSecRealtime",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = {
        key: value
        for line in result.stdout.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    return {
        "unit": unit,
        "available": str(values.get("LoadState") or "not-found"),
        "enabled": str(values.get("UnitFileState") or "disabled"),
        "active": str(values.get("ActiveState") or "unknown"),
        "result": str(values.get("Result") or "unknown"),
        "next": str(values.get("NextElapseUSecRealtime") or "n/a"),
    }


def _selected_runtime_bindings(plan: Mapping[str, object]) -> list[object]:
    from .strategies import LIVE_STRATEGY_BINDINGS

    selected = {
        str(sleeve.get("strategy_id") or "")
        for sleeve in plan.get("sleeves", ())
        if isinstance(sleeve, Mapping)
    }
    return [binding for binding in LIVE_STRATEGY_BINDINGS if binding.strategy_id in selected]


def _candidate_entry_window_open(binding: object, now: datetime) -> bool:
    """Alert stale entry authority only while that candidate can actually enter."""

    local = now.astimezone(ZoneInfo("America/New_York"))
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    strategy_id = str(getattr(binding, "strategy_id", ""))
    if strategy_id.startswith("xsp."):
        return weekday < 5 and (9 * 60 + 20) <= minute < (16 * 60 + 20)
    if strategy_id.startswith(("mcl.", "gold.")):
        if weekday == 6:
            return minute >= 18 * 60
        if weekday < 4:
            return minute < 17 * 60 or minute >= 18 * 60
        return weekday == 4 and minute < 17 * 60
    return False


def _candidate_monitor_window_open(binding: object, now: datetime) -> bool:
    """Include the bounded pre-open warm-up, never an inactive weekend."""

    if _candidate_entry_window_open(binding, now):
        return True
    local = now.astimezone(ZoneInfo("America/New_York"))
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    strategy_id = str(getattr(binding, "strategy_id", ""))
    if strategy_id.startswith("xsp."):
        return weekday < 5 and (9 * 60 + 15) <= minute < (9 * 60 + 20)
    if strategy_id.startswith(("mcl.", "gold.")):
        return weekday in {0, 1, 2, 3, 6} and minute >= (17 * 60 + 40)
    return False


def _login_state(path: Path) -> str:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        return "unknown"
    return str(value.get("state") or "unknown") if isinstance(value, Mapping) else "unknown"


def _recent_decisive_ib_failure() -> bool:
    units = (
        "tradebot-ib-gateway.service",
        "tradebot-gold-live.service",
        "tradebot-gold-onset.service",
        "tradebot-mcl-live.service",
        "tradebot-mcl-turn-tape.service",
        "tradebot-mcl-predictive-onset-runtime.service",
        "tradebot-xsp-shadow.service",
        "tradebot-xsp-pressure-tape.service",
    )
    try:
        command = ["journalctl", "--user", "--since", "-2 min", "--no-pager", "-o", "cat"]
        for unit in units:
            command.extend(("-u", unit))
        journal = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.casefold()
    except (OSError, subprocess.SubprocessError):
        return False
    if "maximum number of account summary requests exceeded" in journal:
        return True
    if "client id already in use" in journal or "client-id" in journal and "exhausted" in journal:
        return True
    codes = [int(value) for value in re.findall(r"(?<!\d)(1100|1101|1102|1300|326)(?!\d)", journal)]
    transport_error = re.search(r"(?:ibkr|ib api|ib error).{0,80}\b(?:502|504)\b", journal)
    return 1300 in codes or 326 in codes or transport_error is not None or (
        1100 in codes and max((index for index, code in enumerate(codes) if code == 1100), default=-1)
        > max((index for index, code in enumerate(codes) if code in {1101, 1102}), default=-1)
    )


def ib_sentinel(
    *,
    repository_root: Path,
    capital_plan_path: Path,
    receipt_path: Path,
    login_receipt_path: Path,
    now: datetime | None = None,
) -> dict[str, object]:
    """Read only the active-candidate path and name a broken trading authority."""

    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    plan = load_live_capital_plan(capital_plan_path)
    failures: list[dict[str, str]] = []
    active: list[str] = []
    for binding in _selected_runtime_bindings(plan):
        timers = [_unit_liveness(unit) for unit in binding.runtime_timer_units]
        services = [_unit_liveness(unit) for unit in binding.runtime_service_units]
        engaged = any(
            state["enabled"] == "enabled" or state["active"] == "active"
            for state in timers
        )
        if not engaged:
            continue
        if not _candidate_monitor_window_open(binding, observed):
            continue
        label = str(getattr(binding, "champion_symbol", "strategy")).casefold()
        reason = f"{label}-runtime-failed" if label in {"xsp", "mcl"} else "gold-runtime-failed"
        active.append(label)
        for state in timers:
            if state["enabled"] != "enabled" or state["active"] != "active":
                failures.append({"reason": reason, "detail": f"timer_not_armed:{state['unit']}"})
            elif state["next"] == "n/a":
                failures.append({"reason": reason, "detail": f"timer_next_elapse_missing:{state['unit']}"})
        for state in services:
            if state["available"] != "loaded" or state["active"] == "failed" or state["result"] == "failed":
                failures.append({"reason": reason, "detail": f"service_failed:{state['unit']}"})
        decision = ib_preflight_decision("entry", path=receipt_path, now=observed)
        if decision["ready"] is not True:
            failures.append({"reason": reason, "detail": "entry_authority_unready"})

    login_state = _login_state(login_receipt_path)
    if login_state == "two_factor_required":
        failures.append({"reason": "ib-gateway-login-required", "detail": login_state})
    elif login_state == "failed":
        failures.append({"reason": "ib-gateway-login-failed", "detail": login_state})

    if active:
        try:
            with socket.create_connection(("127.0.0.1", 4001), timeout=1):
                gateway_up = _unit_liveness("tradebot-ib-gateway.service")["active"] == "active"
        except OSError:
            gateway_up = False
        if not gateway_up or _recent_decisive_ib_failure():
            failures.append({"reason": "ib-runtime-failed", "detail": "gateway_or_decisive_ib_failure"})
    unique = {(row["reason"], row["detail"]): row for row in failures}
    return {
        "schema": "live.ib-sentinel.v1",
        "checked_at_utc": observed.isoformat(),
        "active_candidates": sorted(set(active)),
        "failures": [unique[key] for key in sorted(unique)],
        "boundaries": dict(IB_PREFLIGHT_BOUNDARIES),
    }


def _publish_sentinel(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(dict(value), sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fchmod(handle.fileno(), 0o600)
    os.replace(temporary, path)


def _alert_reasons(value: Mapping[str, object]) -> None:
    failures = value.get("failures")
    if not isinstance(failures, Sequence):
        return
    allowed = {
        "gold-runtime-failed",
        "mcl-runtime-failed",
        "xsp-runtime-failed",
        "ib-runtime-failed",
        "ib-gateway-login-required",
        "ib-gateway-login-failed",
    }
    for reason in sorted(
        {
            str(row.get("reason") or "")
            for row in failures
            if isinstance(row, Mapping) and str(row.get("reason") or "") in allowed
        }
    ):
        subprocess.run(
            ["systemctl", "--user", "--no-block", "start", f"tradebot-operator-alert@{reason}.service"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--repository-root", type=Path, default=Path.cwd())
    probe.add_argument(
        "--capital-plan",
        type=Path,
        default=Path("db/calibration/live_capital_plan.json"),
    )
    probe.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("TRADEBOT_IB_PREFLIGHT_RECEIPT", "/tmp/tradebot-ib-preflight.json")),
    )
    require = commands.add_parser("require")
    require.add_argument("mode", choices=("entry", "reduction"))
    require.add_argument("--receipt", type=Path)
    sentinel = commands.add_parser("sentinel")
    sentinel.add_argument("--repository-root", type=Path, default=Path.cwd())
    sentinel.add_argument("--capital-plan", type=Path, required=True)
    sentinel.add_argument("--receipt", type=Path, required=True)
    sentinel.add_argument("--login-receipt", type=Path, required=True)
    sentinel.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "require":
        decision = ib_preflight_decision(args.mode, path=args.receipt)
        print(json.dumps(decision, sort_keys=True))
        return 0 if decision["ready"] is True else 1
    if args.command == "sentinel":
        value = ib_sentinel(
            repository_root=args.repository_root,
            capital_plan_path=args.capital_plan,
            receipt_path=args.receipt,
            login_receipt_path=args.login_receipt,
        )
        _publish_sentinel(args.output, value)
        _alert_reasons(value)
        print(json.dumps(value, sort_keys=True))
        return 0
    receipt = asyncio.run(
        probe_ib_preflight(
            repository_root=args.repository_root,
            capital_plan_path=args.capital_plan,
        )
    )
    publish_ib_preflight(args.output, receipt)
    verdict = receipt["verdict"]
    print(
        json.dumps(
            {
                "receipt_id": receipt["receipt_id"],
                "entry_ready": verdict["entry_ready"],
                "reduction_ready": verdict["reduction_ready"],
                "entry_reasons": verdict["entry_reasons"],
                "reduction_reasons": verdict["reduction_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0 if verdict["entry_ready"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
