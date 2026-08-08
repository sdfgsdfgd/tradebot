from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tradebot import main as tradebot_main
from tradebot.live import ib_preflight


ROOT = Path(__file__).resolve().parents[1]


def _blocked_ui() -> ModuleType:
    module = ModuleType("tradebot.ui")

    def blocked(_name: str) -> object:
        raise AssertionError("TUI import must not occur before argument and TTY guards")

    module.__getattr__ = blocked  # type: ignore[attr-defined]
    return module


def test_help_exits_before_importing_or_connecting_the_tui(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tradebot.ui", _blocked_ui())

    with pytest.raises(SystemExit) as raised:
        tradebot_main.main(["--help"])

    assert raised.value.code == 0


def test_noninteractive_tui_exits_before_importing_or_connecting(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tradebot.ui", _blocked_ui())
    monkeypatch.setattr(tradebot_main.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(tradebot_main.sys, "stdout", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(SystemExit) as raised:
        tradebot_main.main([])

    assert raised.value.code == 2


def test_owner_only_launchers_have_no_lan_or_mac_gateway_fallback() -> None:
    q_launcher = (ROOT / "deploy/systemd/tradebot-cli").read_text()
    mac_launcher = (ROOT / "deploy/macos/tradebot").read_text()

    assert "IBKR_HOST=127.0.0.1 IBKR_PORT=4001" in q_launcher
    assert "tradebot.live.ib_preflight status" in q_launcher
    assert "-tt" in mac_launcher
    assert "-T" in mac_launcher
    assert "no alternate Gateway was attempted" in mac_launcher
    assert "192.168." not in q_launcher + mac_launcher
    assert "tradebot --status" in q_launcher + mac_launcher


def test_runtime_status_is_readonly_and_reports_receipts(monkeypatch, tmp_path) -> None:
    (tmp_path / "tradebot-ib-gateway-login.json").write_text(
        '{"state":"connected","detail":"connected"}'
    )
    (tmp_path / "tradebot-ib-sentinel.json").write_text(
        '{"checked_at_utc":"2026-08-09T00:00:00+00:00","active_candidates":[],"failures":[],"pending_warmups":[]}'
    )
    monkeypatch.setattr(
        ib_preflight,
        "_unit_liveness",
        lambda _unit: {"unit": _unit, "active": "active", "result": "success"},
    )
    monkeypatch.setattr(ib_preflight, "_gateway_tcp_accepting", lambda: True)
    monkeypatch.setattr(ib_preflight, "_gateway_api_clients", lambda: [{"pid": 7, "process": "python", "declared_client_ids": [3208]}])

    status = ib_preflight.ib_runtime_status(runtime_dir=tmp_path)

    assert status["authority"] == "read_only_q_control_plane_status"
    assert status["broker_owner"] == {
        "scope": "q-local-only",
        "host": "127.0.0.1",
        "port": 4001,
        "gateway_fallback": "disabled",
    }
    assert status["gateway"]["tcp_accepting"] is True
    assert status["semantic_login"]["state"] == "connected"
    assert status["sentinel"]["failures"] == []
    assert status["api_clients"] == [{"pid": 7, "process": "python", "declared_client_ids": [3208]}]
