"""Mine simple candidate laws from a spot predictive table CSV.

The tool is intentionally narrow:
- filters rows to a research seam via exact column matches
- scores 1-rule and 2-rule threshold laws
- ranks candidates by target-year lift minus collateral damage
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _parse_filter(raw: str) -> tuple[str, str]:
    text = str(raw).strip()
    if "=" not in text:
        raise SystemExit(f"Invalid --filter {raw!r}; expected column=value")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_sequence_fields(row: dict[str, object]) -> dict[str, object]:
    def _num(name: str) -> float | None:
        value = row.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    fast_bars = _num("first_fast_adverse_bars")
    slow_bars = _num("first_slow_adverse_bars")
    tr_bars = _num("first_tr_stretch_bars")
    stale_bars = _num("first_stale_signal_ge2_bars")
    collapse_bars = _num("first_local_collapse_bars")

    def _gap(later: float | None, earlier: float | None) -> float | None:
        if later is None or earlier is None:
            return None
        return float(later) - float(earlier)

    return {
        "has_fast_adverse": 1.0 if fast_bars is not None else 0.0,
        "has_slow_adverse": 1.0 if slow_bars is not None else 0.0,
        "has_tr_stretch": 1.0 if tr_bars is not None else 0.0,
        "has_stale_signal_ge2": 1.0 if stale_bars is not None else 0.0,
        "has_local_collapse": 1.0 if collapse_bars is not None else 0.0,
        "fast_before_collapse": 1.0 if fast_bars is not None and collapse_bars is not None and fast_bars <= collapse_bars else 0.0,
        "slow_before_collapse": 1.0 if slow_bars is not None and collapse_bars is not None and slow_bars <= collapse_bars else 0.0,
        "tr_stretch_before_collapse": 1.0 if tr_bars is not None and collapse_bars is not None and tr_bars <= collapse_bars else 0.0,
        "stale_before_collapse": 1.0 if stale_bars is not None and collapse_bars is not None and stale_bars <= collapse_bars else 0.0,
        "collapse_after_fast_bars": _gap(collapse_bars, fast_bars),
        "collapse_after_slow_bars": _gap(collapse_bars, slow_bars),
        "collapse_after_tr_stretch_bars": _gap(collapse_bars, tr_bars),
        "stale_after_fast_bars": _gap(stale_bars, fast_bars),
        "stale_after_slow_bars": _gap(stale_bars, slow_bars),
    }


def _quantile_picks(values: list[float]) -> list[float]:
    if not values:
        return []
    uniq = sorted(set(values))
    picks: list[float] = []
    for q in (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85):
        idx = max(0, min(len(uniq) - 1, int((len(uniq) - 1) * q)))
        picks.append(float(uniq[idx]))
    return sorted(set(picks))


def _rule_hit(row: dict[str, object], rule: tuple[str, str, float]) -> bool:
    field, op, threshold = rule
    value = row.get(field)
    if not isinstance(value, (int, float)):
        return False
    return float(value) <= float(threshold) if op == "<=" else float(value) >= float(threshold)


def _score_rules(
    rows: list[dict[str, object]],
    *,
    target_window: str,
    rules: tuple[tuple[str, str, float], ...],
) -> tuple[float, dict[str, float], dict[str, int]] | None:
    pnl_by_window: dict[str, float] = defaultdict(float)
    count_by_window: dict[str, int] = defaultdict(int)
    for row in rows:
        if all(_rule_hit(row, rule) for rule in rules):
            window = str(row["window"])
            pnl_by_window[window] += float(row["pnl"])
            count_by_window[window] += 1
    target_lift = -float(pnl_by_window.get(target_window, 0.0))
    if target_lift <= 0.0:
        return None
    penalty = 0.0
    bonus = 0.0
    for window, pnl in pnl_by_window.items():
        if window == target_window:
            continue
        if pnl > 0.0:
            penalty += float(pnl)
        else:
            bonus += -float(pnl)
    score = float(target_lift) - float(penalty) + (0.2 * float(bonus))
    return score, dict(pnl_by_window), dict(count_by_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mine simple threshold laws from a predictive table CSV.")
    ap.add_argument("--csv", required=True, help="Predictive table CSV produced by spot_predictive_table.")
    ap.add_argument("--target-window", required=True, help="Target window label prefix, e.g. 2024.")
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact match seam filter formatted column=value. Repeatable.",
    )
    ap.add_argument(
        "--field",
        action="append",
        default=[],
        help="Numeric field to mine. Repeatable. If omitted, uses a sensible default set.",
    )
    ap.add_argument("--top", type=int, default=20, help="How many rows to print (default: 20).")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    filters = [_parse_filter(raw) for raw in args.filter]
    default_fields = [
        "entry_15m_pos",
        "entry_1h_pos",
        "entry_6h30m_pos",
        "tr_ratio",
        "tr_median_pct",
        "slope_med_pct",
        "slope_vel_pct",
        "slope_med_slow_pct",
        "slope_vel_slow_pct",
        "shock_atr_pct",
        "shock_atr_vel_pct",
        "shock_atr_accel_pct",
        "exit_signal_age_bars",
        "first_fast_adverse_bars",
        "first_slow_adverse_bars",
        "first_tr_stretch_bars",
        "first_stale_signal_ge2_bars",
        "first_local_collapse_bars",
        "has_fast_adverse",
        "has_slow_adverse",
        "has_tr_stretch",
        "has_stale_signal_ge2",
        "has_local_collapse",
        "fast_before_collapse",
        "slow_before_collapse",
        "tr_stretch_before_collapse",
        "stale_before_collapse",
        "collapse_after_fast_bars",
        "collapse_after_slow_bars",
        "collapse_after_tr_stretch_bars",
        "stale_after_fast_bars",
        "stale_after_slow_bars",
    ]
    fields = list(dict.fromkeys([*(args.field or []), *([] if args.field else default_fields)]))

    rows: list[dict[str, object]] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if any(str(raw.get(key, "")).strip() != value for key, value in filters):
                continue
            row: dict[str, object] = {
                "window": str(raw.get("window", "")).strip()[:4],
                "pnl": _parse_float(raw.get("pnl")) or 0.0,
            }
            for field in fields:
                row[field] = _parse_float(raw.get(field))
            derived = _derive_sequence_fields(row)
            for field, value in derived.items():
                row[field] = value
            rows.append(row)
    if not rows:
        raise SystemExit("No rows matched the requested seam filters.")

    candidates: list[tuple[str, str, float]] = []
    for field in fields:
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        for pick in _quantile_picks(values):
            candidates.append((field, "<=", float(pick)))
            candidates.append((field, ">=", float(pick)))

    scored: list[tuple[float, tuple[tuple[str, str, float], ...], dict[str, float], dict[str, int]]] = []
    for rule in candidates:
        result = _score_rules(rows, target_window=str(args.target_window).strip(), rules=(rule,))
        if result is not None:
            score, pnl_by_window, count_by_window = result
            scored.append((score, (rule,), pnl_by_window, count_by_window))

    seeds = [entry[1][0] for entry in sorted(scored, key=lambda item: item[0], reverse=True)[:20]]
    for i, left in enumerate(seeds):
        for right in seeds[i + 1 :]:
            result = _score_rules(rows, target_window=str(args.target_window).strip(), rules=(left, right))
            if result is not None:
                score, pnl_by_window, count_by_window = result
                scored.append((score, (left, right), pnl_by_window, count_by_window))

    scored.sort(key=lambda item: item[0], reverse=True)
    for score, rules, pnl_by_window, count_by_window in scored[: max(1, int(args.top))]:
        print(
            {
                "score": round(float(score), 6),
                "rules": rules,
                "pnl_by_window": {k: round(float(v), 1) for k, v in sorted(pnl_by_window.items())},
                "count_by_window": dict(sorted(count_by_window.items())),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
