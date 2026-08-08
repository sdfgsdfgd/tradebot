from __future__ import annotations

from pathlib import Path


def test_operator_alert_accepts_only_fixed_reasons_without_mutating_volume() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "deploy/macos/tradebot-operator-alert").read_text()
    service = (root / "deploy/systemd/tradebot-operator-alert@.service").read_text()

    assert "ib-preflight-failed)" in script
    assert "ib-gateway-login-required)" in script
    assert "ib-gateway-login-failed)" in script
    assert "ib-gateway-exited)" in script
    assert "ib-runtime-failed)" in script
    assert "gold-runtime-failed)" in script
    assert "mcl-runtime-failed)" in script
    assert "xsp-runtime-failed)" in script
    assert "strategy-input-failed)" in script
    assert "exit 64" in script
    assert "set volume" not in script
    assert "afplay -v 1.0" in script
    assert 'aiff" &' not in script
    assert "as critical" in script
    assert "Submarine Sosumi Submarine" in script
    assert "operator-alert-$reason" in script
    assert "tradebot-operator-alert %i" in service
    assert "/Users/x/.local/bin/tradebot-operator-alert" in service
    assert "BatchMode=yes" in service
