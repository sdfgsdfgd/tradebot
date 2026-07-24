from __future__ import annotations

import pytest

from tradebot.backtest.calibration import (
    CalibrationBook,
    CalibrationBucket,
    CalibrationParams,
    CalibrationRecord,
    load_calibration,
    save_calibration,
)


def _params(value: float) -> CalibrationParams:
    return CalibrationParams(
        iv_floor=value,
        iv_risk_premium=1.2,
        skew=-0.2,
        term_slope=0.02,
    )


def test_calibration_uses_effective_boundary_and_preserves_legacy_asof(tmp_path) -> None:
    book = CalibrationBook(
        symbol="XSP",
        buckets=[
            CalibrationBucket(
                min_dte=0,
                max_dte=7,
                records=[
                    CalibrationRecord("2026-07-20", _params(0.10), 0.1, 0.2, 4),
                    CalibrationRecord(
                        "2026-07-24",
                        _params(0.20),
                        0.1,
                        0.2,
                        8,
                        observed_at="2026-07-24T10:00:00-04:00",
                        source="ibkr_delayed_last+prepared_underlying_tape",
                        source_start="2026-07-01T13:30:00+00:00",
                        source_end="2026-07-24T14:00:00+00:00",
                        effective_from="2026-07-25",
                    ),
                ],
            )
        ],
    )

    save_calibration(tmp_path, book)
    loaded = load_calibration(tmp_path, "XSP")

    assert loaded is not None
    assert loaded.params_asof(3, "2026-07-24") == _params(0.10)
    assert loaded.params_asof(3, "2026-07-25") == _params(0.20)
    [record] = loaded.buckets[0].records[1:]
    assert record.source_start == "2026-07-01T13:30:00+00:00"
    assert record.source_end == "2026-07-24T14:00:00+00:00"


def test_rv_override_requires_an_explicit_source_interval() -> None:
    from tradebot.backtest import calibration

    cfg = type("Config", (), {"strategy": type("Strategy", (), {"symbol": "XSP"})()})()

    with pytest.raises(ValueError, match="source_start and source_end"):
        calibration.calibrate_symbol(cfg, None, "2026-07-24", rv_override=0.2)
