from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.backtest.models import Bar
from tradebot.research.xsp_opening_edge_state import XspDailyBar
from tradebot.research.xsp_phase_front import (
    XSP_PHASE_FRONT_AUTHORITY,
    XspPhaseFrontClock,
    XspPhaseFrontEngine,
    XspPhaseFrontObservation,
)
from tradebot.research.xsp_pivot_landmarks import (
    XSP_PIVOT_LANDMARK_AUTHORITY,
    label_xsp_pivot_landmarks,
    xsp_prior_tr_scales,
)


START = datetime(2026, 8, 6, 13, 30, tzinfo=timezone.utc)


def _bar(
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Bar:
    return Bar(
        START + timedelta(minutes=5 * index),
        open_,
        high,
        low,
        close,
        100.0,
    )


def _clocks(
    slopes: tuple[float, float, float, float, float],
    velocities: tuple[float, float, float, float, float],
) -> tuple[XspPhaseFrontClock, ...]:
    return tuple(
        XspPhaseFrontClock(name, horizon, tier, slope, velocity)
        for name, horizon, tier, slope, velocity in zip(
            ("seconds", "5m", "15m", "30m", "60m"),
            (1, 300, 900, 1800, 3600),
            ("micro", "fast", "fast", "structural", "slow"),
            slopes,
            velocities,
            strict=True,
        )
    )


def _observation(
    index: int,
    slopes: tuple[float, float, float, float, float],
    velocities: tuple[float, float, float, float, float],
    *,
    transport: str = "PARTIAL",
    shock_direction: int | None = None,
    contiguous: bool = True,
) -> XspPhaseFrontObservation:
    return XspPhaseFrontObservation(
        observed_at_utc=START + timedelta(minutes=index),
        session_key="2026-08-06:RTH",
        clocks=_clocks(slopes, velocities),
        transport=transport,
        shock_direction=shock_direction,
        contiguous=contiguous,
    )


def test_pivot_scale_uses_only_prior_complete_sessions() -> None:
    first = date(2026, 7, 1)
    rows = [
        XspDailyBar(first + timedelta(days=index), 100.0, 101.0, 99.0, 100.0)
        for index in range(22)
    ]
    rows.append(XspDailyBar(first + timedelta(days=22), 100.0, 200.0, 50.0, 100.0))

    scales = xsp_prior_tr_scales(rows)
    current = scales[rows[-1].day]

    assert current.scale == 2.0
    assert current.prior_sessions == 21
    assert current.first_prior_day == rows[1].day
    assert current.last_prior_day == rows[21].day


def test_future_oracle_labels_opening_pivot_recoil_and_censoring() -> None:
    rows = (
        _bar(0, 100.0, 101.2, 99.9, 101.1),
        _bar(1, 101.1, 103.0, 101.0, 102.8),
        _bar(2, 102.8, 103.2, 101.8, 102.0),
        _bar(3, 102.0, 102.1, 100.5, 100.8),
        _bar(4, 100.8, 101.7, 100.6, 101.7),
        _bar(5, 101.7, 102.0, 101.5, 101.8),
    )

    landmarks = label_xsp_pivot_landmarks(
        rows,
        lane="RTH",
        trading_day=date(2026, 8, 6),
        reference_close=100.0,
        scale=4.0,
    )

    assert landmarks[0].incoming_direction is None
    assert landmarks[0].outgoing_direction == 1
    assert landmarks[0].serious is True
    assert landmarks[0].classes == ("OPENING_DRIVE",)
    assert landmarks[1].incoming_direction == 1
    assert landmarks[1].outgoing_direction == -1
    assert landmarks[1].serious is True
    assert landmarks[1].classes == ("OPENING_PIVOT",)
    assert landmarks[-1].classes == ("RIGHT_CENSORED",)
    assert landmarks[-1].serious is False
    assert landmarks[-1].right_censored is True
    assert landmarks[-1].causal_confirmation_utc is None
    assert all(row.as_payload()["outcomes"] is None for row in landmarks)
    assert all(row.as_payload()["submitted_orders"] == 0 for row in landmarks)
    payload = landmarks[1].as_payload()
    assert payload["trading_date"] == "2026-08-06"
    assert payload["incoming_direction"] == "up"
    assert payload["outgoing_direction"] == "down"
    assert payload["extreme_utc"] is not None
    assert payload["causal_confirmation_utc"] is not None
    assert payload["right_censored"] is False
    assert len(payload["landmark_id"]) == 64


def test_future_oracle_marks_intrabar_order_unresolved() -> None:
    landmarks = label_xsp_pivot_landmarks(
        (
            _bar(0, 100.0, 101.2, 98.8, 101.1),
            _bar(1, 101.1, 101.4, 100.8, 101.2),
        ),
        lane="RTH",
        trading_day=date(2026, 8, 6),
        reference_close=100.0,
        scale=4.0,
    )

    assert landmarks[0].flags == ("INTRABAR_ORDER_UNRESOLVED",)


def test_future_oracle_does_not_count_preconfirmation_extreme() -> None:
    landmarks = label_xsp_pivot_landmarks(
        (
            _bar(0, 100.0, 105.0, 99.9, 101.1),
            _bar(1, 101.1, 101.2, 99.8, 100.0),
            _bar(2, 100.0, 100.1, 99.8, 99.9),
        ),
        lane="RTH",
        trading_day=date(2026, 8, 6),
        reference_close=100.0,
        scale=4.0,
    )

    assert landmarks[0].outgoing_direction == 1
    assert landmarks[0].outgoing_excursion == pytest.approx(1.2)
    assert landmarks[0].serious is False
    assert landmarks[0].classes == ("RECOIL_ONLY",)


def test_phase_front_requires_recoil_then_accepts_once() -> None:
    engine = XspPhaseFrontEngine(persistence_observations=1)
    incumbent = engine.update(
        _observation(0, (-1.0,) * 5, (-0.2,) * 5)
    )
    spark = engine.update(
        _observation(
            1,
            (-0.8, -0.7, -0.6, -0.9, -1.0),
            (0.2, 0.2, 0.2, 0.1, 0.1),
        )
    )
    cascade = engine.update(
        _observation(
            2,
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="FULL_ACCEPTANCE",
        )
    )
    recoil = engine.update(
        _observation(
            3,
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (-0.1, -0.1, -0.1, 0.1, 0.1),
            transport="FULL_ACCEPTANCE",
        )
    )
    reacceleration = engine.update(
        _observation(
            4,
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="FULL_ACCEPTANCE",
        )
    )
    accepted = engine.update(
        _observation(
            5,
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="FULL_ACCEPTANCE",
        )
    )
    next_state = engine.update(
        _observation(
            6,
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="FULL_ACCEPTANCE",
        )
    )

    assert incumbent.phase == "INCUMBENT"
    assert spark.phase in {"CROSS_CONVERGING", "FRONT_PROPAGATING"}
    assert cascade.phase == "SLOW_SLOPE_CASCADE"
    assert recoil.phase == "RECOIL_TEST"
    assert reacceleration.phase == "REACCELERATION"
    assert accepted.phase == "PIVOT_ACCEPTED"
    assert accepted.incumbent_direction == -1
    assert accepted.candidate_direction == 1
    assert next_state.phase == "INCUMBENT"
    assert next_state.incumbent_direction == 1
    payload = accepted.as_payload()
    assert payload["authority"] == XSP_PHASE_FRONT_AUTHORITY
    assert payload["classifier"] == "none"
    assert payload["permission"] == "none"
    assert payload["outcomes"] is None
    assert payload["submitted_orders"] == 0


def test_phase_front_preserves_acceleration_and_jerk_as_morphology_only() -> None:
    clock = XspPhaseFrontClock(
        "5m",
        300,
        "fast",
        -0.2,
        0.1,
        acceleration=0.03,
        jerk=-0.01,
    )

    payload = clock.as_payload(direction=1)
    assert payload["eta"] == pytest.approx(2.0)
    assert payload["acceleration"] == pytest.approx(0.03)
    assert payload["jerk"] == pytest.approx(-0.01)


def test_unavailable_transport_advances_only_the_structural_incumbent() -> None:
    engine = XspPhaseFrontEngine(persistence_observations=1)
    engine.update(
        _observation(0, (-1.0,) * 5, (-0.2,) * 5, transport="UNAVAILABLE")
    )
    engine.update(
        _observation(
            1,
            (-0.8, -0.7, -0.6, -0.9, -1.0),
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="UNAVAILABLE",
        )
    )
    cascade = engine.update(
        _observation(
            2,
            (0.1,) * 5,
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="UNAVAILABLE",
        )
    )
    next_state = engine.update(
        _observation(
            3,
            (0.1,) * 5,
            (0.2, 0.2, 0.2, 0.1, 0.1),
            transport="UNAVAILABLE",
        )
    )

    assert cascade.phase == "SLOW_SLOPE_CASCADE"
    assert cascade.candidate_direction == 1
    assert cascade.transport == "UNAVAILABLE"
    assert cascade.as_payload()["permission"] == "none"
    assert next_state.phase == "INCUMBENT"
    assert next_state.incumbent_direction == 1
    assert next_state.candidate_direction is None


def test_phase_front_relapses_and_shock_handoff_requires_full_transport() -> None:
    engine = XspPhaseFrontEngine(persistence_observations=1)
    engine.update(_observation(0, (-1.0,) * 5, (-0.2,) * 5))
    engine.update(
        _observation(1, (-0.8,) * 5, (0.2, 0.2, 0.2, 0.1, 0.1))
    )
    relapse = engine.update(
        _observation(2, (-0.8,) * 5, (-0.2,) * 5)
    )
    ignored_shock = engine.update(
        _observation(
            3,
            (-0.8,) * 5,
            (-0.2,) * 5,
            shock_direction=1,
            transport="PARTIAL",
        )
    )
    handoff = engine.update(
        _observation(
            4,
            (0.2,) * 5,
            (0.2,) * 5,
            shock_direction=1,
            transport="FULL_ACCEPTANCE",
        )
    )
    after = engine.update(
        _observation(
            5,
            (0.2,) * 5,
            (0.2,) * 5,
            transport="FULL_ACCEPTANCE",
        )
    )

    assert relapse.phase == "RELAPSED"
    assert ignored_shock.phase == "INCUMBENT"
    assert handoff.phase == "HANDOFF"
    assert handoff.incumbent_direction == -1
    assert handoff.candidate_direction == 1
    assert after.phase == "INCUMBENT"
    assert after.incumbent_direction == 1


def test_phase_front_gap_resets_and_oracle_is_physically_separate() -> None:
    engine = XspPhaseFrontEngine()
    engine.update(_observation(0, (-1.0,) * 5, (-0.2,) * 5))
    reset = engine.update(
        _observation(1, (-1.0,) * 5, (-0.2,) * 5, contiguous=False)
    )

    causal_source = Path("tradebot/research/xsp_phase_front.py").read_text()
    oracle_source = Path("tradebot/research/xsp_pivot_landmarks.py").read_text()
    assert reset.phase == "UNDERWARM_OR_GAP"
    assert "xsp_pivot_landmarks" not in causal_source
    assert "xsp_phase_front" not in oracle_source
    assert XSP_PIVOT_LANDMARK_AUTHORITY.startswith("offline_future_aware")


def test_phase_front_rejects_nonmonotonic_observations() -> None:
    engine = XspPhaseFrontEngine()
    row = _observation(0, (-1.0,) * 5, (-0.2,) * 5)
    engine.update(row)
    with pytest.raises(ValueError, match="timestamps must increase"):
        engine.update(row)
