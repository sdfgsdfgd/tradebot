"""Read-only causal anatomy for MCL news catch-up and saturation onset."""

from __future__ import annotations

import bisect
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ib_insync import ContFuture, IB

from ..news.contract import load_news_history


NEWS_HISTORY = Path.home() / ".local/state/tradebot/news/history"
BAR = timedelta(minutes=5)
PROSPECTIVE_START_AFTER = datetime(2026, 7, 31, 19, 16, 52, tzinfo=timezone.utc)


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sign(value: float | None) -> int:
    if value is None or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pct(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start == 0.0:
        return None
    return (end / start - 1.0) * 100.0


def fresh_bar_index(ends: list[datetime], at: datetime) -> int | None:
    index = bisect.bisect_right(ends, at) - 1
    if index < 0 or at - ends[index] > BAR:
        return None
    return index


def load_news(history_dir: Path = NEWS_HISTORY) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prior_pressure: float | None = None
    for path in sorted(history_dir.glob("*.jsonl")):
        for payload in load_news_history(path):
            try:
                asset = payload["analysis"]["assets"]["MCL"]
                signal_at = datetime.fromisoformat(
                    payload["signal_as_of_utc"].replace("Z", "+00:00")
                )
                at = datetime.fromisoformat(
                    payload["snapshot_as_of_utc"].replace("Z", "+00:00")
                )
                pressure = round(
                    float(asset["direction"])
                    * float(asset["impact"])
                    / 100.0
                    * float(asset["confidence"]),
                    6,
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if at < signal_at:
                raise ValueError("MCL publication availability precedes its signal")
            rows.append(
                {
                    "at": at.astimezone(timezone.utc),
                    "signal_at": signal_at.astimezone(timezone.utc),
                    "pressure": pressure,
                    "delta": (
                        None
                        if prior_pressure is None
                        else round(pressure - prior_pressure, 6)
                    ),
                    "change": asset.get("change"),
                    "direction": int(asset["direction"]),
                    "impact": int(asset["impact"]),
                    "confidence": float(asset["confidence"]),
                }
            )
            prior_pressure = pressure
    return rows


def ols(prior: list[dict[str, object]], x: float) -> tuple[float, float | None]:
    pairs = [
        (float(row["pre4h"]), float(row["delta"]))
        for row in prior
        if row.get("pre4h") is not None and row.get("delta") is not None
    ]
    if len(pairs) < 8:
        return 0.0, None
    xs = [item[0] for item in pairs]
    ys = [item[1] for item in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((value - mx) ** 2 for value in xs)
    slope = sum((a - mx) * (b - my) for a, b in pairs) / denom if denom else 0.0
    intercept = my - slope * mx
    prediction = intercept + slope * x
    residuals = [b - (intercept + slope * a) for a, b in pairs]
    sigma = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
    return prediction, (None if sigma == 0.0 else sigma)


def summary(rows: list[dict[str, object]], *, orient: bool = False) -> dict[str, object]:
    def values(field: str) -> list[float]:
        result = []
        for row in rows:
            value = number(row.get(field))
            if value is None:
                continue
            if orient and field in ("post30", "post1h", "post4h"):
                value *= -sign(number(row.get("delta")))
            result.append(value)
        return result

    result: dict[str, object] = {"n": len(rows)}
    for field in ("pre4h", "last30", "post30", "post1h", "post4h", "post4h_range"):
        samples = values(field)
        result[field] = round(statistics.fmean(samples), 4) if samples else None
        if orient and field.startswith("post") and samples:
            result[f"{field}_positive_rate"] = round(
                sum(value > 0.0 for value in samples) / len(samples), 4
            )
    return result


def main() -> None:
    ib = IB()
    ib.connect(
        os.environ.get("IBKR_HOST", "127.0.0.1"),
        int(os.environ.get("IBKR_PORT", "4001")),
        clientId=int(os.environ.get("IBKR_CLIENT_ID", "3194")),
        readonly=True,
        timeout=12,
    )
    contract = ContFuture("MCL", "NYMEX", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise RuntimeError("MCL continuous future did not qualify")
    contract = qualified[0]
    bars_raw = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=os.environ.get("MCL_ONSET_DURATION", "1 M"),
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
        timeout=60,
    )
    ib.disconnect()
    bars = []
    for raw in bars_raw:
        start = raw.date
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        else:
            start = start.astimezone(timezone.utc)
        bars.append(
            {
                "end": start + BAR,
                "open": float(raw.open),
                "high": float(raw.high),
                "low": float(raw.low),
                "close": float(raw.close),
            }
        )
    ends = [row["end"] for row in bars]

    def index_at(at: datetime) -> int | None:
        return fresh_bar_index(ends, at)

    def close_at(at: datetime) -> float | None:
        index = index_at(at)
        return None if index is None else float(bars[index]["close"])

    def mature_close(at: datetime) -> float | None:
        return close_at(at) if ends and ends[-1] >= at else None

    def interval(at: datetime, hours: float) -> list[dict[str, object]]:
        start = at - timedelta(hours=hours)
        left = bisect.bisect_right(ends, start)
        right = bisect.bisect_right(ends, at)
        return bars[left:right]

    news = load_news()
    enriched: list[dict[str, object]] = []
    for source in news:
        row = dict(source)
        at = row["at"]
        assert isinstance(at, datetime)
        current = close_at(at)
        row["pre4h"] = pct(close_at(at - timedelta(hours=4)), current)
        row["pre1h"] = pct(close_at(at - timedelta(hours=1)), current)
        row["last30"] = pct(close_at(at - timedelta(minutes=30)), current)
        row["post30"] = pct(current, mature_close(at + timedelta(minutes=30)))
        row["post1h"] = pct(current, mature_close(at + timedelta(hours=1)))
        row["post4h"] = pct(current, mature_close(at + timedelta(hours=4)))
        forward = (
            interval(at + timedelta(hours=4), 4)
            if ends and ends[-1] >= at + timedelta(hours=4)
            else []
        )
        if current is not None and forward:
            row["post4h_range"] = (
                (max(float(bar["high"]) for bar in forward) - min(float(bar["low"]) for bar in forward))
                / current
                * 100.0
            )
        else:
            row["post4h_range"] = None

        index = index_at(at)
        if index is not None and index >= 48:
            ranges = [
                (float(bar["high"]) - float(bar["low"])) / float(bar["close"]) * 100.0
                for bar in bars[index - 47 : index + 1]
            ]
            fast, prior, slow = mean(ranges[-6:]), mean(ranges[-12:-6]), mean(ranges)
            row["tr_ratio"] = None if slow in (None, 0.0) else float(fast) / float(slow)
            row["tr_acceleration"] = (
                None if fast is None or prior is None else float(fast) - float(prior)
            )
        else:
            row["tr_ratio"] = None
            row["tr_acceleration"] = None

        pre4h = number(row.get("pre4h"))
        delta = number(row.get("delta"))
        last30 = number(row.get("last30"))
        if pre4h is not None and delta is not None:
            prediction, sigma = ols(enriched, pre4h)
            row["expected_delta"] = prediction
            row["innovation_z"] = None if sigma is None else (delta - prediction) / sigma
        else:
            row["expected_delta"] = None
            row["innovation_z"] = None

        catchup = sign(delta) != 0 and sign(delta) == sign(pre4h)
        if sign(delta) == 0:
            stage = "unchanged"
        elif not catchup:
            stage = "dislocated_or_novel"
        else:
            prior_progress = sign(delta) * float(pre4h or 0.0) / 4.0
            fast_progress = sign(delta) * float(last30 or 0.0) / 0.5
            if fast_progress <= 0.0:
                stage = "catchup_reversing"
            elif fast_progress <= prior_progress:
                stage = "catchup_decelerating"
            else:
                stage = "catchup_confirming"
        row["stage"] = stage
        row["extreme"] = abs(float(row["pressure"])) >= 0.80
        z = number(row.get("innovation_z"))
        row["reactive"] = z is not None and abs(z) <= 1.0
        row["novel"] = z is not None and abs(z) >= 2.0
        enriched.append(row)

    eligible = [row for row in enriched if row.get("post4h") is not None]
    groups = {}
    for stage in (
        "unchanged",
        "dislocated_or_novel",
        "catchup_confirming",
        "catchup_decelerating",
        "catchup_reversing",
    ):
        cells = [row for row in eligible if row["stage"] == stage]
        groups[stage] = {
            "raw": summary(cells),
            "opposite_delta": summary(cells, orient=True),
            "extreme": summary([row for row in cells if row["extreme"]], orient=True),
            "extreme_reactive": summary(
                [row for row in cells if row["extreme"] and row["reactive"]], orient=True
            ),
        }

    onset = [
        row
        for row in eligible
        if row["extreme"]
        and row["stage"] in ("catchup_decelerating", "catchup_reversing")
        and row["reactive"]
    ]
    price_only_control = []
    for row in eligible:
        pre4h = number(row.get("pre4h"))
        last30 = number(row.get("last30"))
        if sign(pre4h) == 0 or last30 is None:
            continue
        prior_progress = abs(float(pre4h)) / 4.0
        fast_progress = sign(pre4h) * last30 / 0.5
        if fast_progress > prior_progress:
            continue
        control = dict(row)
        control["news_delta"] = row.get("delta")
        control["delta"] = pre4h
        price_only_control.append(control)
    half_horizon_placebo = []
    for source in eligible:
        observed_at = source["at"]
        assert isinstance(observed_at, datetime)
        observed_at += timedelta(hours=2)
        current = mature_close(observed_at)
        pre4h = pct(mature_close(observed_at - timedelta(hours=4)), current)
        last30 = pct(mature_close(observed_at - timedelta(minutes=30)), current)
        if current is None or sign(pre4h) == 0 or last30 is None:
            continue
        if sign(pre4h) * last30 / 0.5 > abs(float(pre4h)) / 4.0:
            continue
        half_horizon_placebo.append(
            {
                "delta": pre4h,
                "extreme": source["extreme"],
                "pressure": source["pressure"],
                "pre4h": pre4h,
                "last30": last30,
                "post30": pct(current, mature_close(observed_at + timedelta(minutes=30))),
                "post1h": pct(current, mature_close(observed_at + timedelta(hours=1))),
                "post4h": pct(current, mature_close(observed_at + timedelta(hours=4))),
                "post4h_range": None,
            }
        )
    event_view = []
    for row in onset:
        event_view.append(
            {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in row.items()
                if key
                in {
                    "at",
                    "pressure",
                    "delta",
                    "change",
                    "pre4h",
                    "last30",
                    "tr_ratio",
                    "tr_acceleration",
                    "innovation_z",
                    "stage",
                    "post30",
                    "post1h",
                    "post4h",
                    "post4h_range",
                }
            }
        )

    wakeups: list[dict[str, object]] = []
    armed = [
        row
        for row in eligible
        if row["extreme"]
        and row["reactive"]
        and str(row["stage"]).startswith("catchup_")
    ]
    for row in armed:
        at = row["at"]
        assert isinstance(at, datetime)
        narrative = sign(number(row.get("delta")))
        start_index = bisect.bisect_right(ends, at)
        stop_at = at + timedelta(hours=4)
        trigger: dict[str, object] | None = None
        for index in range(start_index, len(bars)):
            observed_at = ends[index]
            if observed_at > stop_at:
                break
            current = float(bars[index]["close"])
            slopes = {
                minutes: pct(close_at(observed_at - timedelta(minutes=minutes)), current)
                for minutes in (5, 15, 30, 60, 120)
            }
            fast_votes = sum(
                sign(slopes[minutes]) == -narrative for minutes in (5, 15, 30)
            )
            slow_votes = sum(
                sign(slopes[minutes]) == narrative for minutes in (60, 120)
            )
            if fast_votes < 2 or slow_votes < 1:
                continue
            recent_ranges = [
                (float(bar["high"]) - float(bar["low"])) / float(bar["close"]) * 100.0
                for bar in bars[max(0, index - 11) : index + 1]
            ]
            fast_range = mean(recent_ranges[-6:])
            prior_range = mean(recent_ranges[-12:-6])
            trigger = {
                "publication_at": at,
                "trigger_at": observed_at,
                "minutes_after_publication": (observed_at - at).total_seconds() / 60.0,
                "pressure": row["pressure"],
                "delta": row["delta"],
                "stage_at_publication": row["stage"],
                "innovation_z": row["innovation_z"],
                "fast_votes": fast_votes,
                "slow_votes": slow_votes,
                "slopes": slopes,
                "range_accelerating": (
                    fast_range is not None
                    and prior_range is not None
                    and fast_range > prior_range
                ),
                "post30": pct(current, mature_close(observed_at + timedelta(minutes=30))),
                "post1h": pct(current, mature_close(observed_at + timedelta(hours=1))),
                "post4h": pct(current, mature_close(observed_at + timedelta(hours=4))),
            }
            break
        if trigger is not None:
            wakeups.append(trigger)

    wakeup_mature = [row for row in wakeups if row.get("post1h") is not None]
    extreme_ta_onset = [row for row in price_only_control if row["extreme"]]
    prospective_ta_onset = [
        row for row in price_only_control if row["at"] > PROSPECTIVE_START_AFTER
    ]
    prospective_extreme_onset = [
        row for row in extreme_ta_onset if row["at"] > PROSPECTIVE_START_AFTER
    ]
    output = {
        "authority": "exploratory_anatomy_only",
        "submitted_orders": 0,
        "contract": {
            "conId": getattr(contract, "conId", None),
            "localSymbol": getattr(contract, "localSymbol", None),
            "lastTradeDateOrContractMonth": getattr(
                contract, "lastTradeDateOrContractMonth", None
            ),
        },
        "bars": len(bars),
        "bar_start_utc": ends[0].isoformat() if ends else None,
        "bar_end_utc": ends[-1].isoformat() if ends else None,
        "news": len(news),
        "eligible_four_hour_outcomes": len(eligible),
        "prospective_start_after_utc": PROSPECTIVE_START_AFTER.isoformat(),
        "groups": groups,
        "strict_onset": {
            "definition": "extreme pressure + causal catch-up + fast deceleration/reversal + <=1 sigma expanding innovation",
            "summary_opposite_delta": summary(onset, orient=True),
            "events": event_view,
        },
        "ta_only_publication_clock_control": {
            "definition": "prior 4h price direction with last-30m directional progress <= prior hourly rate; no news fields",
            "summary_opposite_prior_move": summary(price_only_control, orient=True),
            "causal_feature_ablation": {
                "extreme_pressure": summary(
                    [row for row in price_only_control if row["extreme"]], orient=True
                ),
                "news_catchup": summary(
                    [
                        row
                        for row in price_only_control
                        if sign(number(row.get("news_delta")))
                        == sign(number(row.get("pre4h")))
                        and sign(number(row.get("news_delta"))) != 0
                    ],
                    orient=True,
                ),
                "extreme_news_catchup": summary(
                    [
                        row
                        for row in price_only_control
                        if row["extreme"]
                        and sign(number(row.get("news_delta")))
                        == sign(number(row.get("pre4h")))
                        and sign(number(row.get("news_delta"))) != 0
                    ],
                    orient=True,
                ),
                "extreme_reactive_news_catchup": summary(onset, orient=True),
            },
        },
        "ta_only_half_horizon_placebo": {
            "definition": "same price-only state evaluated two hours after each publication",
            "summary_opposite_prior_move": summary(half_horizon_placebo, orient=True),
            "extreme_pressure": summary(
                [row for row in half_horizon_placebo if row["extreme"]], orient=True
            ),
            "nonextreme_pressure": summary(
                [row for row in half_horizon_placebo if not row["extreme"]], orient=True
            ),
        },
        "primary_candidate": {
            "definition": "TA-only impulse deceleration owns opposite direction; fresh abs(signed pressure)>=0.80 is an attribution-only persistence/convexity classifier",
            "discovery_ta_only": summary(price_only_control, orient=True),
            "discovery_extreme_pressure": summary(extreme_ta_onset, orient=True),
            "prospective_ta_only": summary(prospective_ta_onset, orient=True),
            "prospective_extreme_pressure": summary(
                prospective_extreme_onset, orient=True
            ),
        },
        "soft_wakeup": {
            "definition": "extreme reactive catch-up arms; first 2/3 fast slopes oppose delta while >=1/2 slow slopes still confirm it",
            "armed_publications": len(armed),
            "triggered": len(wakeups),
            "one_hour_mature": len(wakeup_mature),
            "summary_opposite_delta": summary(wakeup_mature, orient=True),
            "events": wakeups,
        },
    }
    print(json.dumps(output, indent=2, default=lambda value: value.isoformat()))


if __name__ == "__main__":
    main()
