"""Read-only proof of the canonical q-local IB Gateway owner.

Run this only on q.  It intentionally opens three disposable, read-only API
clients but never requests market data, account summaries, orders, or a tunnel.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from pathlib import Path
import socket
import subprocess

import pytest
from ib_insync import IB

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("TRADEBOT_RUN_LIVE_IB_GATEWAY") != "1",
        reason="set TRADEBOT_RUN_LIVE_IB_GATEWAY=1 to run q-local IB Gateway canaries",
    ),
]

GATEWAY_PORT = int(os.getenv("TRADEBOT_LIVE_IB_GATEWAY_PORT", "4001"))
CLIENT_ID_START = int(os.getenv("TRADEBOT_LIVE_CLIENT_ID_START", "19031"))
Q_LAN_HOST = os.getenv("TRADEBOT_LIVE_Q_LAN_HOST", "192.168.1.4")
RECOVERY_HASHES = {
    Path("/var/tmp/mcl_emergency_flatten.py"): "89799993376236c2f56e76e744c7509e5896abeb36530fa744524c909badcb0a",
    Path("/var/tmp/tradebot_gold_fail_closed_rollover.sh"): "eff92634463eb3690936401c0afd770b816f0fadd9878da4539d199e02099c12",
    Path("/var/tmp/tradebot_xsp_p009_fresh_rollover.sh"): "9695e7b5234fddd7b12280065498c68a549dcf13399aa4a9e4dbf18d3798b791",
    Path("/home/x/Desktop/py/tradebot/deploy/systemd/tradebot-mcl-stage131-successor"): "31cb800d9cbfceb813dd0c36367cb2be086ff1c471a39b15b485e2ec2932eccb",
}


def _systemctl(*args: str, user: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["systemctl"]
    if user:
        command.append("--user")
    command.extend(args)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_gateway_is_reachable_only_from_q_localhost() -> None:
    with socket.create_connection(("127.0.0.1", GATEWAY_PORT), timeout=4.0) as sock:
        assert sock.getpeername() == ("127.0.0.1", GATEWAY_PORT)

    assert _systemctl("is-active", "--quiet", "tradebot-ib-loopback-firewall.service").returncode == 0
    with pytest.raises(OSError):
        socket.create_connection((Q_LAN_HOST, GATEWAY_PORT), timeout=2.0)


def test_gateway_trust_and_firewall_are_exactly_localhost_only() -> None:
    lines = (Path.home() / "Jts/jts.ini").read_text().splitlines()
    trusted = next(line for line in lines if line.startswith("TrustedIPs="))
    assert {value.strip() for value in trusted.partition("=")[2].split(",") if value.strip()} == {
        "127.0.0.1"
    }
    rules = subprocess.run(
        ["nft", "list", "table", "inet", "tradebot_ib_gateway"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    assert 'iifname != "lo" tcp dport 4001 reject with tcp reset' in rules


def test_readonly_ib_protocol_directly_on_q() -> None:
    async def probe() -> dict[str, object]:
        ib = IB()
        try:
            assert "readonly" in inspect.signature(IB.connectAsync).parameters
            await ib.connectAsync("127.0.0.1", GATEWAY_PORT, clientId=CLIENT_ID_START, timeout=8, readonly=True)
            return {
                "connected": ib.isConnected(),
                "server_version": ib.client.serverVersion(),
                "managed_accounts": tuple(ib.managedAccounts()),
            }
        finally:
            if ib.isConnected():
                ib.disconnect()

    receipt = asyncio.run(probe())
    assert receipt["connected"] is True
    assert int(receipt["server_version"]) > 0
    assert len(receipt["managed_accounts"]) >= 1


def test_tradebot_three_client_stack_directly_on_q(tmp_path: Path) -> None:
    config = IBKRConfig(
        host="127.0.0.1",
        port=GATEWAY_PORT,
        client_id=CLIENT_ID_START + 10,
        proxy_client_id=CLIENT_ID_START + 11,
        account=None,
        refresh_sec=1.0,
        detail_refresh_sec=1.0,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=10.0,
        reconnect_slow_interval_sec=30.0,
        client_id_pool_start=CLIENT_ID_START + 10,
        client_id_pool_end=CLIENT_ID_START + 19,
        client_id_burst_attempts=4,
        client_id_backoff_initial_sec=1.0,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(tmp_path / "ib-client-ids.json"),
        connect_timeout_sec=8.0,
        client_id_quarantine_sec=10.0,
        readonly=True,
        account_bootstrap=False,
    )
    client = IBKRClient(config)

    async def probe() -> dict[str, object]:
        try:
            await client.connect()
            await client.connect_proxy()
            await client.connect_index()
            return {
                "main": client._ib.isConnected(),
                "proxy": client._ib_proxy.isConnected(),
                "index": client._ib_index.isConnected(),
                "state": client.connection_state(),
                "ids": (
                    client._connected_main_client_id,
                    client._connected_proxy_client_id,
                    client._connected_index_client_id,
                ),
            }
        finally:
            client._shutdown = True
            client._stop_reconnect_loop()
            client._safe_disconnect(client._ib_index)
            client._safe_disconnect(client._ib_proxy)
            client._safe_disconnect(client._ib)

    receipt = asyncio.run(probe())
    assert receipt["main"] is True
    assert receipt["proxy"] is True
    assert receipt["index"] is True
    assert receipt["state"] == "connected"
    assert len(set(receipt["ids"])) == 3


def test_live_target_owns_only_schedules_and_old_tunnel_is_gone() -> None:
    live = _systemctl("cat", "tradebot-live.target", user=True)
    assert live.returncode == 0, live.stderr
    assert "After=tradebot-ib-gateway.service" in live.stdout
    assert "Wants=" not in live.stdout
    assert "PropagatesStopTo=" not in live.stdout
    assert _systemctl("is-active", "--quiet", "tradebot-ib-gateway.service", user=True).returncode == 0
    old = _systemctl("status", "tradebot-ib-gateway-tunnel.service", user=True)
    assert old.returncode != 0
    assert not (Path.home() / ".config/systemd/user/tradebot-ib-gateway-tunnel.service").exists()


def test_recovery_scripts_remain_byte_identical() -> None:
    for path, expected in RECOVERY_HASHES.items():
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
