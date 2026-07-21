from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest
from ib_insync import IB

from tradebot.client import IBKRClient
from tradebot.config import IBKRConfig


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("TRADEBOT_RUN_LIVE_IB_GATEWAY") != "1",
        reason="set TRADEBOT_RUN_LIVE_IB_GATEWAY=1 to run external IB Gateway canaries",
    ),
]

MAC_HOST = os.getenv("TRADEBOT_LIVE_MAC_HOST", "192.168.1.2")
MAC_USER = os.getenv("TRADEBOT_LIVE_MAC_USER", "x")
GATEWAY_PORT = int(os.getenv("TRADEBOT_LIVE_IB_GATEWAY_PORT", "4001"))
CLIENT_ID_START = int(os.getenv("TRADEBOT_LIVE_CLIENT_ID_START", "18800"))


def _ssh_base() -> list[str]:
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_listener(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 8.0
    last_error = "listener did not become reachable"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise AssertionError(f"SSH tunnel exited early ({process.returncode}): {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)
    raise AssertionError(last_error)


@pytest.fixture(scope="module")
def tunneled_gateway() -> tuple[str, int]:
    local_port = _free_local_port()
    process = subprocess.Popen(
        _ssh_base()
        + [
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{GATEWAY_PORT}",
            f"{MAC_USER}@{MAC_HOST}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_listener(process, local_port)
        yield "127.0.0.1", local_port
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=4)


def test_direct_ib_gateway_transport_reachable() -> None:
    with socket.create_connection((MAC_HOST, GATEWAY_PORT), timeout=4.0) as sock:
        assert sock.getpeername() == (MAC_HOST, GATEWAY_PORT)


def test_gateway_trust_is_localhost_only() -> None:
    result = subprocess.run(
        _ssh_base()
        + [
            f"{MAC_USER}@{MAC_HOST}",
            "grep -E '^TrustedIPs=' \"$HOME/Jts/jts.ini\"",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip()
    assert line.startswith("TrustedIPs=")
    trusted = {value.strip() for value in line.partition("=")[2].split(",") if value.strip()}
    assert trusted == {"127.0.0.1"}


def test_readonly_ib_protocol_through_ssh_tunnel(tunneled_gateway: tuple[str, int]) -> None:
    host, port = tunneled_gateway

    async def probe() -> dict[str, object]:
        ib = IB()
        try:
            assert "readonly" in inspect.signature(IB.connectAsync).parameters
            await ib.connectAsync(host, port, clientId=19031, timeout=8, readonly=True)
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


def test_tradebot_three_client_stack_through_ssh_tunnel(
    tunneled_gateway: tuple[str, int],
    tmp_path: Path,
) -> None:
    host, port = tunneled_gateway
    state_path = tmp_path / "ib-client-ids.json"
    config = IBKRConfig(
        host=host,
        port=port,
        client_id=CLIENT_ID_START,
        proxy_client_id=CLIENT_ID_START + 1,
        account=None,
        refresh_sec=1.0,
        detail_refresh_sec=1.0,
        reconnect_interval_sec=5.0,
        reconnect_timeout_sec=10.0,
        reconnect_slow_interval_sec=30.0,
        client_id_pool_start=CLIENT_ID_START,
        client_id_pool_end=CLIENT_ID_START + 9,
        client_id_burst_attempts=4,
        client_id_backoff_initial_sec=1.0,
        client_id_backoff_max_sec=2.0,
        client_id_backoff_multiplier=2.0,
        client_id_backoff_jitter_ratio=0.0,
        client_id_state_file=str(state_path),
        connect_timeout_sec=8.0,
        client_id_quarantine_sec=10.0,
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
