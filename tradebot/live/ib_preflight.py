"""One fail-closed IBKR readiness receipt for every writable strategy owner."""

from __future__ import annotations

import argparse
import asyncio
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


from .ib_preflight_contract import (
    IB_PREFLIGHT_BOUNDARIES,
    gate_actionable_plan,
    ib_preflight_configured,
    ib_preflight_decision,
    load_ib_preflight,
    order_preflight_mode,
    publish_ib_preflight,
    reduce_ib_preflight,
    require_order_preflight,
    require_reduction_preflight,
    validate_ib_preflight,
)


__all__ = (
    "IB_PREFLIGHT_BOUNDARIES",
    "gate_actionable_plan",
    "ib_preflight_configured",
    "ib_preflight_decision",
    "ib_sentinel",
    "load_ib_preflight",
    "order_preflight_mode",
    "publish_ib_preflight",
    "reduce_ib_preflight",
    "require_order_preflight",
    "require_reduction_preflight",
    "validate_ib_preflight",
)


IB_SENTINEL_WARMUP_GRACE_SEC = 30 * 60
IB_RUNTIME_STATUS_SCHEMA = "live.ib-runtime-status.v1"
IB_GATEWAY_HOST = "127.0.0.1"
IB_GATEWAY_PORT = 4001

def _contract_specs(
    plan: Mapping[str, object],
    root: Path,
    *,
    sleeve_id: str | None = None,
) -> list[dict[str, object]]:
    specs: dict[int, dict[str, object]] = {}
    for sleeve in plan.get("sleeves", ()):
        if not isinstance(sleeve, Mapping):
            continue
        current_sleeve = str(sleeve.get("sleeve_id") or "")
        if sleeve_id is not None and current_sleeve != sleeve_id:
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
                specs[con_id] = {
                    "label": label.upper(),
                    "con_id": con_id,
                    "sleeve_id": current_sleeve,
                    "strategy_id": str(sleeve.get("strategy_id") or ""),
                }
    return [specs[key] for key in sorted(specs)]


def _runtime_members_by_sleeve(
    plan: Mapping[str, object],
) -> dict[str, list[str]]:
    from .strategies import LIVE_STRATEGY_BINDINGS

    bindings = {binding.strategy_id: binding for binding in LIVE_STRATEGY_BINDINGS}
    return {
        str(sleeve.get("sleeve_id") or ""): sorted(binding.runtime_timer_units)
        for sleeve in plan.get("sleeves", ())
        if isinstance(sleeve, Mapping)
        for binding in (bindings.get(str(sleeve.get("strategy_id") or "")),)
        if binding is not None and str(sleeve.get("sleeve_id") or "")
    }


def _required_runtime_members(plan: Mapping[str, object]) -> list[str]:
    return sorted(
        {
            unit
            for members in _runtime_members_by_sleeve(plan).values()
            for unit in members
        }
    )


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


async def _contract_price_ready(ib: IB, contract: Contract) -> bool:
    """Probe one qualified contract without coupling unrelated holdings."""

    try:
        tickers = await asyncio.wait_for(ib.reqTickersAsync(contract), timeout=15)
    except (OSError, RuntimeError, asyncio.TimeoutError, ValueError):
        return False
    return len(tickers) == 1 and _price_ready(tickers[0])


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
    reduction_quotes: dict[str, bool] = {}
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
        held_ids = sorted(
            {
                int(row.contract.conId or 0)
                for row in raw_positions
                if abs(float(row.position)) > 1e-9
            }
        )
        held_probes = [Contract(conId=con_id) for con_id in held_ids if con_id > 0]
        held_qualified = (
            list(await asyncio.wait_for(ib.qualifyContractsAsync(*held_probes), timeout=15))
            if held_probes
            else []
        )
        held_by_id = {
            int(contract.conId or 0): contract
            for contract in held_qualified
            if int(contract.conId or 0) > 0
        }
        qualified_held_ids = sorted(held_by_id)
        held_readiness = dict(
            zip(
                qualified_held_ids,
                await asyncio.gather(
                    *(_contract_price_ready(ib, held_by_id[con_id]) for con_id in qualified_held_ids)
                ),
                strict=True,
            )
        )
        reduction_quotes = {
            str(con_id): held_readiness.get(con_id, False)
            for con_id in held_ids
        }
        if held_ids:
            reduction_quote_ready = all(reduction_quotes.values())
        else:
            reduction_quote_ready = True
    except (OSError, RuntimeError, asyncio.TimeoutError, ValueError):
        pass
    finally:
        if ib.isConnected():
            ib.disconnect()

    members_by_sleeve = _runtime_members_by_sleeve(plan)
    expected_members = sorted(
        {unit for members in members_by_sleeve.values() for unit in members}
    )
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
            "reduction_quotes": reduction_quotes,
        },
        "capabilities": capabilities,
        "connectivity": _connectivity_facts(),
        "runtime": {
            "expected_members": expected_members,
            "armed_members": armed_members,
            "missing_members": sorted(set(expected_members) - set(armed_members)),
            "members_by_sleeve": members_by_sleeve,
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
            "NRestarts",
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
        "restarts": str(values.get("NRestarts") or "0"),
        "next": str(values.get("NextElapseUSecRealtime") or "n/a"),
    }


def _runtime_directory() -> Path:
    configured = os.getenv("XDG_RUNTIME_DIR", "").strip()
    return Path(configured) if configured else Path(f"/run/user/{os.getuid()}")


def _runtime_receipt(path: Path, *keys: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError):
        value = None
    if not isinstance(value, Mapping):
        return {"path": str(path), "available": False}
    return {
        "path": str(path),
        "available": True,
        **{key: value.get(key) for key in keys if key in value},
    }


def _gateway_tcp_accepting() -> bool:
    try:
        with socket.create_connection((IB_GATEWAY_HOST, IB_GATEWAY_PORT), timeout=1):
            return True
    except OSError:
        return False


def _process_client_ids(pid: str) -> list[int]:
    try:
        values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return []
    identifiers: list[int] = []
    for value in values:
        key, separator, raw = value.partition(b"=")
        if separator and key in {b"IBKR_CLIENT_ID", b"IBKR_PROXY_CLIENT_ID"}:
            try:
                identifiers.append(int(raw))
            except ValueError:
                continue
    return sorted(set(identifiers))


def _gateway_api_clients() -> list[dict[str, object]]:
    """Report q-local API consumers without asking IBKR for account or market data."""

    try:
        output = subprocess.run(
            ["ss", "-H", "-tnp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    clients: list[dict[str, object]] = []
    for line in output.splitlines():
        match = re.search(
            r"\s127\.0\.0\.1:\d+\s+127\.0\.0\.1:4001\s+users:\(\(\"([^\"]+)\",pid=(\d+)",
            line,
        )
        if match is None:
            continue
        process, pid = match.groups()
        clients.append(
            {
                "pid": int(pid),
                "process": process,
                "declared_client_ids": _process_client_ids(pid),
            }
        )
    return sorted(clients, key=lambda value: int(value["pid"]))


def ib_runtime_status(*, runtime_dir: Path | None = None) -> dict[str, object]:
    """Read q's one-owner broker control plane without creating an IB API client."""

    directory = runtime_dir or _runtime_directory()
    preflight_path = directory / "tradebot-ib-preflight.json"
    return {
        "schema": IB_RUNTIME_STATUS_SCHEMA,
        "authority": "read_only_q_control_plane_status",
        "broker_owner": {
            "scope": "q-local-only",
            "host": IB_GATEWAY_HOST,
            "port": IB_GATEWAY_PORT,
            "gateway_fallback": "disabled",
        },
        "gateway": {
            "service": _unit_liveness("tradebot-ib-gateway.service"),
            "tcp_accepting": _gateway_tcp_accepting(),
        },
        "semantic_login": _runtime_receipt(
            directory / "tradebot-ib-gateway-login.json", "state", "detail", "updated_at_epoch"
        ),
        "preflight": {
            "path": str(preflight_path),
            "entry": ib_preflight_decision("entry", path=preflight_path),
            "reduction": ib_preflight_decision("reduction", path=preflight_path),
        },
        "sentinel": _runtime_receipt(
            directory / "tradebot-ib-sentinel.json",
            "checked_at_utc",
            "active_candidates",
            "failures",
            "pending_warmups",
        ),
        "api_clients": _gateway_api_clients(),
        "boundaries": dict(IB_PREFLIGHT_BOUNDARIES),
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

    strategy_id = str(getattr(binding, "strategy_id", ""))
    if strategy_id.startswith("gold."):
        local = now.astimezone(ZoneInfo("America/Chicago"))
        weekday, minute = local.weekday(), local.hour * 60 + local.minute
        return not (
            weekday == 5 and 2 * 60 <= minute < 4 * 60
            or weekday < 5 and 16 * 60 <= minute < 16 * 60 + 2
        )
    local = now.astimezone(ZoneInfo("America/New_York"))
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    if strategy_id.startswith("xsp."):
        return weekday < 5 and (9 * 60 + 20) <= minute < (16 * 60 + 20)
    if strategy_id.startswith("mcl."):
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
    strategy_id = str(getattr(binding, "strategy_id", ""))
    if strategy_id.startswith("gold."):
        return False
    local = now.astimezone(ZoneInfo("America/New_York"))
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    if strategy_id.startswith("xsp."):
        return weekday < 5 and (9 * 60) <= minute < (16 * 60 + 20)
    if strategy_id.startswith("mcl."):
        return weekday in {0, 1, 2, 3, 6} and minute >= (17 * 60 + 30)
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


def _warmup_grace(
    failures: Sequence[Mapping[str, str]],
    *,
    state_path: Path,
    observed: datetime,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Escalate an unwarmed authority only after one continuous warm-up budget."""

    try:
        raw = json.loads(state_path.read_text())
        prior = raw.get("first_unhealthy_epoch", {}) if isinstance(raw, Mapping) else {}
    except (OSError, TypeError, ValueError):
        prior = {}
    prior = prior if isinstance(prior, Mapping) else {}
    now_epoch = observed.timestamp()
    current: dict[str, float] = {}
    due: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    for failure in failures:
        reason, detail = str(failure["reason"]), str(failure["detail"])
        key = f"{reason}:{detail}"
        try:
            first = float(prior.get(key, now_epoch))
        except (TypeError, ValueError):
            first = now_epoch
        first = min(first, now_epoch)
        current[key] = first
        age_sec = max(0.0, now_epoch - first)
        row: dict[str, object] = {
            "reason": reason,
            "detail": detail,
            "first_unhealthy_at_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
            "age_sec": round(age_sec, 3),
        }
        if age_sec >= IB_SENTINEL_WARMUP_GRACE_SEC:
            due.append(row)
        else:
            pending.append({**row, "grace_remaining_sec": round(IB_SENTINEL_WARMUP_GRACE_SEC - age_sec, 3)})
    _publish_sentinel(state_path, {"first_unhealthy_epoch": current})
    return due, pending


def ib_sentinel(
    *,
    repository_root: Path,
    capital_plan_path: Path,
    receipt_path: Path,
    login_receipt_path: Path,
    state_path: Path,
    now: datetime | None = None,
) -> dict[str, object]:
    """Read only the active-candidate path and name a broken trading authority."""

    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    plan = load_live_capital_plan(capital_plan_path)
    failures: list[dict[str, str]] = []
    active: list[str] = []
    planned_recoveries: list[dict[str, object]] = []
    for binding in _selected_runtime_bindings(plan):
        timers = [_unit_liveness(unit) for unit in binding.runtime_timer_units]
        services = [_unit_liveness(unit) for unit in binding.runtime_service_units]
        recovery_unit = str(getattr(binding, "recovery_timer_unit", ""))
        recovery = _unit_liveness(recovery_unit) if recovery_unit else None
        recovery_armed = bool(
            recovery
            and recovery["enabled"] == "enabled"
            and recovery["active"] == "active"
            and recovery["next"] != "n/a"
        )
        recovery_managed = (
            set(getattr(binding, "recovery_managed_timer_units", ()))
            if recovery_armed
            else set()
        )
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
        if recovery_armed and recovery is not None:
            planned_recoveries.append(
                {
                    "candidate": label,
                    "timer": recovery_unit,
                    "next": recovery["next"],
                    "managed_timers": sorted(recovery_managed),
                }
            )
        for state in timers:
            if state["unit"] in recovery_managed:
                continue
            if state["enabled"] != "enabled" or state["active"] != "active":
                failures.append({"reason": reason, "detail": f"timer_not_armed:{state['unit']}"})
            elif state["next"] == "n/a":
                failures.append({"reason": reason, "detail": f"timer_next_elapse_missing:{state['unit']}"})
        for state in services:
            if state["available"] != "loaded" or state["active"] == "failed" or state["result"] == "failed":
                failures.append({"reason": reason, "detail": f"service_failed:{state['unit']}"})
        strategy_id = str(getattr(binding, "strategy_id", ""))
        sleeve_ids = [
            str(sleeve.get("sleeve_id") or "")
            for sleeve in plan.get("sleeves", ())
            if isinstance(sleeve, Mapping)
            and sleeve.get("strategy_id") == strategy_id
        ]
        con_ids = sorted(
            {
                int(spec["con_id"])
                for sleeve_id in sleeve_ids
                for spec in _contract_specs(
                    plan,
                    repository_root.resolve(),
                    sleeve_id=sleeve_id,
                )
            }
        )
        decision = ib_preflight_decision(
            "entry",
            path=receipt_path,
            now=observed,
            con_ids=con_ids or None,
        )
        expected_reasons = {
            f"runtime_member_not_armed:{unit}" for unit in recovery_managed
        }
        unexpected_reasons = set(map(str, decision.get("reasons", ()))) - expected_reasons
        if decision["ready"] is not True and unexpected_reasons:
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
    immediate = [
        row
        for row in unique.values()
        if row["reason"].startswith("ib-gateway-login")
        or row["reason"] == "ib-runtime-failed"
        or row["detail"].startswith(("timer_not_armed:", "timer_next_elapse_missing:"))
    ]
    warmable = [row for row in unique.values() if row not in immediate]
    due, pending = _warmup_grace(warmable, state_path=state_path, observed=observed)
    return {
        "schema": "live.ib-sentinel.v1",
        "checked_at_utc": observed.isoformat(),
        "active_candidates": sorted(set(active)),
        "planned_recoveries": sorted(
            planned_recoveries,
            key=lambda row: (str(row["candidate"]), str(row["timer"])),
        ),
        "failures": sorted([*immediate, *due], key=lambda row: (str(row["reason"]), str(row["detail"]))),
        "pending_warmups": sorted(pending, key=lambda row: (str(row["reason"]), str(row["detail"]))),
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
    require.add_argument("--con-id", type=int, action="append")
    require.add_argument("--repository-root", type=Path, default=Path.cwd())
    require.add_argument(
        "--capital-plan",
        type=Path,
        default=Path("db/calibration/live_capital_plan.json"),
    )
    require.add_argument("--sleeve-id")
    sentinel = commands.add_parser("sentinel")
    sentinel.add_argument("--repository-root", type=Path, default=Path.cwd())
    sentinel.add_argument("--capital-plan", type=Path, required=True)
    sentinel.add_argument("--receipt", type=Path, required=True)
    sentinel.add_argument("--login-receipt", type=Path, required=True)
    sentinel.add_argument("--output", type=Path, required=True)
    sentinel.add_argument("--state", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--runtime-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "require":
        if args.con_id and args.sleeve_id:
            raise SystemExit("--con-id and --sleeve-id are mutually exclusive")
        con_ids = list(args.con_id or ())
        if args.sleeve_id:
            plan = load_live_capital_plan(args.capital_plan)
            con_ids = [
                int(spec["con_id"])
                for spec in _contract_specs(
                    plan,
                    args.repository_root.resolve(),
                    sleeve_id=str(args.sleeve_id),
                )
            ]
            if not con_ids:
                raise SystemExit("selected sleeve has no executable contracts")
        decision = ib_preflight_decision(
            args.mode,
            path=args.receipt,
            con_ids=con_ids,
        )
        print(json.dumps(decision, sort_keys=True))
        return 0 if decision["ready"] is True else 1
    if args.command == "sentinel":
        value = ib_sentinel(
            repository_root=args.repository_root,
            capital_plan_path=args.capital_plan,
            receipt_path=args.receipt,
            login_receipt_path=args.login_receipt,
            state_path=args.state,
        )
        _publish_sentinel(args.output, value)
        _alert_reasons(value)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(ib_runtime_status(runtime_dir=args.runtime_dir), sort_keys=True))
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
