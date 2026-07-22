"""Canonical risk-overlay search packs."""

from __future__ import annotations


def _risk_overlay_off_template(*, extended: bool = False) -> dict[str, object]:
    out: dict[str, object] = {
        "risk_entry_cutoff_hour_et": None,
        "riskoff_tr5_med_pct": None,
        "riskpanic_tr5_med_pct": None,
        "riskpanic_neg_gap_ratio_min": None,
        "riskpop_tr5_med_pct": None,
        "riskpop_pos_gap_ratio_min": None,
    }
    if bool(extended):
        out.update(
            {
                "riskoff_mode": None,
                "riskoff_tr5_lookback_days": None,
                "riskoff_short_risk_mult_factor": None,
                "riskoff_long_risk_mult_factor": None,
                "riskpanic_lookback_days": None,
                "riskpanic_short_risk_mult_factor": None,
                "riskpop_lookback_days": None,
                "riskpop_long_risk_mult_factor": None,
                "riskpop_short_risk_mult_factor": None,
            }
        )
    return out


def _risk_pack_riskoff(
    *,
    tr_med: float,
    lookback_days: int = 5,
    mode: str = "hygiene",
    long_factor: float | None = None,
    short_factor: float | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskoff_tr5_med_pct": float(tr_med),
        "riskoff_tr5_lookback_days": int(lookback_days),
        "riskoff_mode": str(mode),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if long_factor is not None:
        out["riskoff_long_risk_mult_factor"] = float(long_factor)
    if short_factor is not None:
        out["riskoff_short_risk_mult_factor"] = float(short_factor)
    return out


def _risk_pack_riskpanic(
    *,
    tr_med: float,
    neg_gap_ratio: float,
    lookback_days: int = 5,
    short_factor: float | None = 0.5,
    long_factor: float | None = None,
    mode: str | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskpanic_tr5_med_pct": float(tr_med),
        "riskpanic_neg_gap_ratio_min": float(neg_gap_ratio),
        "riskpanic_lookback_days": int(lookback_days),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if short_factor is not None:
        out["riskpanic_short_risk_mult_factor"] = float(short_factor)
    if long_factor is not None:
        out["riskpanic_long_risk_mult_factor"] = float(long_factor)
    if mode is not None:
        out["riskoff_mode"] = str(mode)
    return out


def _risk_pack_riskpop(
    *,
    tr_med: float,
    pos_gap_ratio: float,
    lookback_days: int = 5,
    long_factor: float | None = 1.2,
    short_factor: float | None = 0.5,
    mode: str | None = None,
    cutoff_hour_et: int | None = 15,
) -> dict[str, object]:
    out: dict[str, object] = {
        "riskpop_tr5_med_pct": float(tr_med),
        "riskpop_pos_gap_ratio_min": float(pos_gap_ratio),
        "riskpop_lookback_days": int(lookback_days),
        "risk_entry_cutoff_hour_et": int(cutoff_hour_et) if cutoff_hour_et is not None else None,
    }
    if long_factor is not None:
        out["riskpop_long_risk_mult_factor"] = float(long_factor)
    if short_factor is not None:
        out["riskpop_short_risk_mult_factor"] = float(short_factor)
    if mode is not None:
        out["riskoff_mode"] = str(mode)
    return out
