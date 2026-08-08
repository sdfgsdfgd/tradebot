from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy/systemd"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text()


def test_gateway_is_q_native_self_healing_and_outside_strategy_lifecycle() -> None:
    service = _unit("tradebot-ib-gateway.service")
    login = _unit("tradebot-ib-gateway-login.service")
    ensure = _unit("tradebot-ib-gateway-ensure.service")
    login_worker = _unit("tradebot-ib-gateway-login")
    timer = _unit("tradebot-ib-gateway.timer")
    sentinel = _unit("tradebot-ib-sentinel.service")
    sentinel_timer = _unit("tradebot-ib-sentinel.timer")
    launcher = _unit("tradebot-ib-gateway-launch")

    assert "ExecStart=%h/.local/bin/tradebot-ib-gateway-launch" in service
    assert "PassEnvironment=DISPLAY WAYLAND_DISPLAY XAUTHORITY" in service
    assert "ExecStartPost=/usr/bin/systemctl --user --no-block start tradebot-ib-gateway-login.service" in service
    assert "Restart=always" in service
    assert "RestartSec=15s" in service
    assert "StartLimitIntervalSec=5min" in service
    assert "StartLimitBurst=4" in service
    assert "WantedBy=default.target" in service
    assert "KillMode=control-group" in service
    assert "OnFailure=tradebot-operator-alert@ib-gateway-exited.service" in service
    assert "PartOf=tradebot-ib-gateway.service" in login
    assert "OnFailure=tradebot-operator-alert@ib-gateway-login-failed.service" in login
    assert "ExecStart=%h/.local/bin/tradebot-ib-gateway-login" in login
    assert "TRADEBOT_IB_LOGIN_RECEIPT=%t/tradebot-ib-gateway-login.json" in login
    assert "LoadCredential=" not in login
    assert '"/usr/bin/secret-tool"' in login_worker
    assert 'TESSERACT = "/usr/bin/tesseract"' in login_worker
    assert 'IMPORT = "/usr/bin/import"' in login_worker
    assert '{"username", "password", "log", "in"}' in login_worker
    assert '"interactive brokers api server" in candidate_text' in login_worker
    assert '[XDOTOOL, "windowunmap", window]' in login_worker
    assert '"two_factor_required", "IBKR requested fingerprint/2FA"' in login_worker
    assert '"state": state' in login_worker
    assert "tradebot-live.target" not in service + timer
    assert "OnCalendar=Sun *-*-* 17:20:00 America/New_York" in timer
    assert "Unit=tradebot-ib-gateway-ensure.service" in timer
    assert "start tradebot-ib-gateway.service" in ensure
    assert "start tradebot-ib-gateway-login.service" in ensure
    assert "restart tradebot-ib-gateway.service" not in ensure
    assert "Persistent=false" in timer
    assert '.mutter-Xwaylandauth.*' in launcher
    assert '"$#" -ne 1' in launcher
    assert "stat -c %u" in launcher
    assert "ss -H -ltn 'sport = :4001'" in launcher
    assert "port 4001 is already owned" in launcher
    assert "1:--check) check_only=true" in launcher
    assert 'exec "$gateway"' in launcher
    assert "tradebot-ib-loopback-firewall.service" in launcher
    assert "tradebot.live.ib_preflight sentinel" in sentinel
    assert "OnFailure=tradebot-operator-alert@ib-runtime-failed.service" in sentinel
    assert "--state %t/tradebot-ib-sentinel-state.json" in sentinel
    assert "OnUnitActiveSec=1min" in sentinel_timer
    assert "Persistent=false" in sentinel_timer


def test_preflight_is_a_single_readonly_probe_before_both_entry_windows() -> None:
    service = _unit("tradebot-ib-preflight.service")
    timer = _unit("tradebot-ib-preflight.timer")

    assert "After=network-online.target tradebot-ib-gateway.service" in service
    assert "Requires=tradebot-ib-gateway.service" not in service
    assert "Wants=tradebot-ib-gateway.service" not in service
    assert "OnFailure=" not in service
    assert "ConditionPathExists=" not in service
    assert "ExecStartPre=/usr/bin/test -f" in service
    assert " -m tradebot.live.ib_preflight probe " in service
    assert "OnCalendar=Sun,Mon,Tue,Wed,Thu *-*-* 17:40:00 America/New_York" in timer
    assert "OnCalendar=Mon..Fri *-*-* 09:15:00 America/New_York" in timer
    assert "Persistent=false" in timer


def test_every_current_broker_consumer_uses_the_native_gateway() -> None:
    current = (
        "tradebot-gold-live.service",
        "tradebot-gold-onset.service",
        "tradebot-mcl-live.service",
        "tradebot-mcl-turn-tape.service",
        "tradebot-mcl-predictive-onset-runtime.service",
        "tradebot-mcl-narrative-prospective.service",
        "tradebot-xsp-shadow.service",
        "tradebot-xsp-pressure-tape.service",
    )
    for name in current:
        service = _unit(name)
        assert "tradebot-ib-gateway.service" in service, name
        assert "tradebot-ib-gateway-tunnel.service" not in service, name
        assert "Requires=tradebot-ib-gateway.service" not in service, name
        assert "Wants=tradebot-ib-gateway.service" not in service, name
        assert "OnFailure=tradebot-operator-alert@" in service, name


def test_writable_owners_require_reduction_readiness_before_starting() -> None:
    for name in (
        "tradebot-gold-live.service",
        "tradebot-mcl-live.service",
        "tradebot-xsp-shadow.service",
    ):
        service = _unit(name)
        assert "TRADEBOT_IB_PREFLIGHT_RECEIPT=%t/tradebot-ib-preflight.json" in service
        assert "TRADEBOT_IB_PREFLIGHT_MAX_AGE_SEC=108000" in service
        assert "ib_preflight require reduction" in service
        assert "Restart=on-failure" not in service


def test_only_armed_strategy_bundle_timers_join_the_live_target() -> None:
    timers = (
        "tradebot-gold-live.timer",
        "tradebot-gold-onset.timer",
        "tradebot-mcl-live.timer",
        "tradebot-mcl-turn-tape.timer",
        "tradebot-mcl-predictive-onset-runtime.timer",
        "tradebot-xsp-shadow.timer",
        "tradebot-xsp-pressure-tape.timer",
    )
    for name in timers:
        timer = _unit(name)
        assert "PartOf=tradebot-live.target" in timer, name
        assert "WantedBy=tradebot-live.target" in timer, name

    target = _unit("tradebot-live.target")
    assert "Wants=" not in target
    assert "After=tradebot-ib-gateway.service" in target
    assert "PropagatesStopTo=" not in target


def test_scheduled_recovery_flows_are_native_alerted_and_their_scripts_remain_external() -> None:
    cases = {
        "tradebot-mcl-emergency-flatten.service": (
            "mcl-runtime-failed",
            "/var/tmp/mcl_emergency_flatten.py",
        ),
        "tradebot-mcl-stage131-successor.service": (
            "mcl-runtime-failed",
            "tradebot-mcl-stage131-successor",
        ),
        "tradebot-gold-fail-closed-rollover.service": (
            "gold-runtime-failed",
            "/var/tmp/tradebot_gold_fail_closed_rollover.sh",
        ),
        "tradebot-xsp-p009-fresh-rollover.service": (
            "xsp-runtime-failed",
            "/var/tmp/tradebot_xsp_p009_fresh_rollover.sh",
        ),
    }
    for name, (alert, script) in cases.items():
        service = _unit(name)
        assert "After=network-online.target tradebot-ib-gateway.service" in service, name
        assert "PartOf=tradebot-live.target" not in service, name
        assert f"OnFailure=tradebot-operator-alert@{alert}.service" in service, name
        assert script in service, name

    emergency = _unit("tradebot-mcl-emergency-flatten.service")
    assert "ib_preflight require reduction" in emergency
    assert "tradebot-live-account.lock" in emergency
    assert "mcl_emergency_flatten.py" in emergency
    assert "tradebot-mcl-emergency-flatten.ok" in emergency
    assert "Restart=on-failure" in emergency
    assert "RestartSec=20s" in emergency

    successor = _unit("tradebot-mcl-stage131-successor.service")
    assert "tradebot-mcl-emergency-flatten.ok" in successor
    assert "Restart=on-failure" in successor
    assert "RestartSec=30s" in successor
    assert "StartLimitIntervalSec=30min" in successor
    assert "StartLimitBurst=60" in successor

    for name in (
        "tradebot-mcl-emergency-flatten.timer",
        "tradebot-mcl-stage131-successor.timer",
        "tradebot-gold-fail-closed-rollover.timer",
        "tradebot-xsp-p009-fresh-rollover.timer",
    ):
        timer = _unit(name)
        assert "PartOf=tradebot-live.target" not in timer, name
        assert "WantedBy=timers.target" in timer, name
        assert "Persistent=false" in timer, name


def test_gateway_loopback_firewall_is_a_root_owned_launch_precondition() -> None:
    firewall = _unit("tradebot-ib-loopback-firewall.service")
    rules = _unit("tradebot-ib-loopback-firewall.nft")

    assert "ExecStartPre=-/usr/sbin/nft delete table inet tradebot_ib_gateway" in firewall
    assert "WantedBy=multi-user.target" in firewall
    assert 'iifname != "lo" tcp dport 4001 reject with tcp reset' in rules
