from __future__ import annotations

from tradebot.research.xsp_pressure_onramp import (
    causal_rank,
    xsp_pressure_onramp_decision,
)


def _impulse(
    direction: int,
    velocity: int,
    *,
    expanding: bool = True,
) -> dict[str, object]:
    return {
        "atr_ratio": 1.4,
        "atr_velocity_pct": 0.03 if expanding else -0.03,
        "atr_acceleration_pct": 0.02 if expanding else -0.02,
        "horizons": [
            {
                "bars": bars,
                "return_pct": direction * (0.1 + bars / 100.0),
                "slope_pct_per_bar": direction * (0.02 + bars / 1000.0),
                "slope_velocity_pct_per_bar": velocity * 0.01,
            }
            for bars in (1, 3, 6, 12, 24)
        ],
    }


def test_causal_rank_requires_prior_sessions_and_uses_only_the_tail() -> None:
    assert causal_rank(10.0, tuple(range(20))) is None
    assert causal_rank(64.0, tuple(range(64))) == 1.0
    assert causal_rank(0.0, tuple(range(64))) == 1.0 / 64.0


def test_dual_accelerating_opposition_revokes_without_waiting() -> None:
    decision = xsp_pressure_onramp_decision(
        target_direction="up",
        xsp_impulse=_impulse(-1, -1),
        spy_impulse=_impulse(-1, -1),
        spy_volume_rank=0.98,
        spy_true_range_rank=0.97,
    )

    assert decision.action == "REVOKE"
    assert decision.reason == "dual_authoritative_opposition"
    assert decision.xsp.path_state == "OPPOSED_ACCELERATING"
    assert decision.spy_pressure_level == "SHOCK"


def test_opposed_path_with_velocity_crest_waits_once() -> None:
    first = xsp_pressure_onramp_decision(
        target_direction="up",
        xsp_impulse=_impulse(-1, 1),
        spy_impulse=_impulse(-1, 1),
        spy_volume_rank=0.70,
        spy_true_range_rank=0.70,
    )
    terminal = xsp_pressure_onramp_decision(
        target_direction="up",
        xsp_impulse=_impulse(-1, 1),
        spy_impulse=_impulse(-1, 1),
        spy_volume_rank=0.70,
        spy_true_range_rank=0.70,
        wait_invocations=1,
    )

    assert first.action == "WAIT"
    assert first.xsp.path_state == "OPPOSED_DECELERATING"
    assert terminal.action == "REVOKE"


def test_fully_opposed_path_never_admits_immediately_when_atr_contracts() -> None:
    decision = xsp_pressure_onramp_decision(
        target_direction="up",
        xsp_impulse=_impulse(-1, -1, expanding=False),
        spy_impulse=_impulse(1, 1, expanding=False),
        spy_volume_rank=0.50,
        spy_true_range_rank=0.50,
    )

    assert decision.action == "WAIT"
    assert decision.xsp.path_state == "OPPOSED_ACCELERATING"
    assert decision.xsp.volatility_state == "CONTRACTING"


def test_wait_admits_only_after_opposition_clears() -> None:
    decision = xsp_pressure_onramp_decision(
        target_direction="down",
        xsp_impulse=_impulse(-1, -1),
        spy_impulse=_impulse(-1, -1),
        spy_volume_rank=0.50,
        spy_true_range_rank=0.50,
        wait_invocations=1,
    )

    assert decision.action == "ADMIT_NOW"
    assert decision.reason == "one_wait_opposition_cleared"
    assert decision.xsp.path_state == "ALIGNED_ACCELERATING"


def test_ordinary_mixed_pressure_does_not_redefine_direction() -> None:
    mixed = _impulse(1, 1)
    mixed["horizons"][1]["return_pct"] *= -1
    decision = xsp_pressure_onramp_decision(
        target_direction="up",
        xsp_impulse=mixed,
        spy_impulse=mixed,
        spy_volume_rank=0.50,
        spy_true_range_rank=0.50,
    )

    assert decision.action == "ADMIT_NOW"
    assert decision.target_direction == "up"
    assert decision.as_payload()["direction_authority"] == (
        "opening_edge_v3_crown_only"
    )
    assert decision.as_payload()["order_authority"] == "none"
