"""Outcome-blind multiscale morphology over the prospective XSP pressure tape."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

from .xsp_pressure_tape import (
    XSP_PRESSURE_TAPE_AUTHORITY,
    XSP_PRESSURE_TAPE_SCHEMA,
)


XSP_PRESSURE_ATLAS_VERSION = "xsp.opening-edge-v3-pressure-morphology-atlas.v1"
XSP_PRESSURE_ATLAS_GENERATION_SCHEMA = "xsp.pressure-morphology-generation.v1"
XSP_PRESSURE_ATLAS_AUTHORITY = (
    "prospective_count_and_morphology_only_no_outcomes_no_permission_no_orders_no_capital"
)
XSP_PRESSURE_ATLAS_FORMATION_SECONDS = 60
XSP_PRESSURE_ATLAS_HORIZONS_SECONDS = (1, 3, 5, 10, 15, 30, 45)
XSP_PRESSURE_ATLAS_GENERATION_PATH = Path(
    "backtests/xsp/opening_edge_v3_pressure_atlas_generation.json"
)
_PHYSICAL_BOOKS = ("SPY", "UPRO", "SPXU")
_NORMALIZED_BOOKS = ("SPY", "UPRO", "INVERSE_SPXU")
_DAILY_HORIZONS = (5, 10, 21, 42, 63, 84)


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("XSP pressure-atlas timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _number(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"XSP pressure-atlas {name} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"XSP pressure-atlas {name} is not finite")
    return result


def _optional_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _direction(value: float) -> str:
    return "UP" if value > 0.0 else "DOWN" if value < 0.0 else "FLAT"


def _target_shape(value: float, target_sign: int) -> str:
    signed = value * target_sign
    return (
        "WITH_TARGET"
        if signed > 0.0
        else "AGAINST_TARGET"
        if signed < 0.0
        else "FLAT"
    )


def _change_shape(value: float) -> str:
    return "RISING" if value > 0.0 else "FALLING" if value < 0.0 else "FLAT"


def load_xsp_pressure_atlas_generation(
    path: Path = XSP_PRESSURE_ATLAS_GENERATION_PATH,
    *,
    root: Path | None = None,
    owner_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Rehash the immutable atlas, its raw generation, and its preregistration."""

    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("XSP pressure-atlas generation must be an object")
    if payload.get("schema") != XSP_PRESSURE_ATLAS_GENERATION_SCHEMA:
        raise ValueError("XSP pressure-atlas generation schema drifted")
    if payload.get("authority") != XSP_PRESSURE_ATLAS_AUTHORITY:
        raise ValueError("XSP pressure-atlas authority drifted")
    if int(payload.get("formation_seconds") or 0) != XSP_PRESSURE_ATLAS_FORMATION_SECONDS:
        raise ValueError("XSP pressure-atlas formation clock drifted")
    if tuple(payload.get("horizons_seconds") or ()) != XSP_PRESSURE_ATLAS_HORIZONS_SECONDS:
        raise ValueError("XSP pressure-atlas horizons drifted")
    if payload.get("outcomes_open") is not False:
        raise ValueError("XSP pressure-atlas outcomes are open")
    if payload.get("permission_open") is not False:
        raise ValueError("XSP pressure-atlas permission is open")
    if payload.get("order_authority") != "none" or payload.get("submitted_orders") != 0:
        raise ValueError("XSP pressure-atlas has order authority")

    repository = root or Path(__file__).resolve().parents[2]
    for key in ("preregistration", "pressure_tape_generation"):
        artifact = payload.get(key)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"XSP pressure-atlas {key} binding is missing")
        artifact_path = repository / str(artifact.get("path") or "")
        if not artifact_path.is_file() or _sha256(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"XSP pressure-atlas {key} drifted")
    owner = owner_path or Path(__file__).resolve()
    if _sha256(owner) != payload.get("owner_sha256"):
        raise ValueError("XSP pressure-atlas owner drifted")
    return payload, _sha256(source)


def _formation(
    records: Sequence[Mapping[str, object]],
    *,
    as_of_utc: datetime,
) -> tuple[tuple[Mapping[str, object], ...], str, str]:
    as_of = _utc(as_of_utc)
    if as_of.microsecond:
        raise ValueError("XSP pressure-atlas close must be second aligned")
    start = as_of - timedelta(seconds=XSP_PRESSURE_ATLAS_FORMATION_SECONDS)
    selected: dict[datetime, Mapping[str, object]] = {}
    for record in records:
        if record.get("kind") != "second":
            continue
        try:
            observed = datetime.fromisoformat(
                str(record.get("bucket_start_utc") or "").replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("XSP pressure-atlas source timestamp is invalid") from exc
        observed = _utc(observed)
        if not start <= observed < as_of:
            continue
        if observed in selected:
            raise ValueError("XSP pressure-atlas source second is duplicated")
        selected[observed] = record

    expected = tuple(start + timedelta(seconds=index) for index in range(60))
    if tuple(sorted(selected)) != expected:
        raise ValueError("XSP pressure-atlas requires 60 contiguous source seconds")
    rows = tuple(selected[value] for value in expected)
    generations: set[str] = set()
    trading_dates: set[str] = set()
    record_ids: set[str] = set()
    for row in rows:
        if row.get("schema") != XSP_PRESSURE_TAPE_SCHEMA:
            raise ValueError("XSP pressure-atlas source schema drifted")
        if row.get("authority") != XSP_PRESSURE_TAPE_AUTHORITY:
            raise ValueError("XSP pressure-atlas source authority drifted")
        if row.get("session") != "RTH":
            raise ValueError("XSP pressure-atlas source is outside RTH")
        if row.get("eligible_treatment") is not True or row.get("valid_evidence") is not True:
            raise ValueError("XSP pressure-atlas source is ineligible or invalid")
        market_data_types = row.get("market_data_types")
        if not isinstance(market_data_types, Mapping) or any(
            market_data_types.get(symbol) != 1 for symbol in _PHYSICAL_BOOKS
        ):
            raise ValueError("XSP pressure-atlas source is not standard live L1")
        if row.get("submitted_orders") != 0:
            raise ValueError("XSP pressure-atlas source contains an order")
        generation = str(row.get("generation_sha256") or "")
        trading_date = str(row.get("trading_date") or "")
        record_id = str(row.get("record_id") or "")
        if not all(len(value) == 64 for value in (generation, record_id)):
            raise ValueError("XSP pressure-atlas source identity is invalid")
        if record_id in record_ids:
            raise ValueError("XSP pressure-atlas record identity is duplicated")
        content = dict(row)
        content.pop("record_id", None)
        if hashlib.sha256(_canonical(content)).hexdigest() != record_id:
            raise ValueError("XSP pressure-atlas source content hash drifted")
        generations.add(generation)
        trading_dates.add(trading_date)
        record_ids.add(record_id)
    if len(generations) != 1 or len(trading_dates) != 1 or "" in trading_dates:
        raise ValueError("XSP pressure-atlas source generation or session drifted")
    return rows, generations.pop(), trading_dates.pop()


def _summary(row: Mapping[str, object], symbol: str) -> Mapping[str, object]:
    books = row.get("books")
    book = books.get(symbol) if isinstance(books, Mapping) else None
    summary = book.get("summary") if isinstance(book, Mapping) else None
    if not isinstance(summary, Mapping):
        raise ValueError(f"XSP pressure-atlas {symbol} summary is missing")
    return summary


def _quad(summary: Mapping[str, object], name: str) -> tuple[float, float, float, float]:
    raw = summary.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise ValueError(f"XSP pressure-atlas {name} is missing")
    values = tuple(_number(value, name=name) for value in raw)
    return values  # type: ignore[return-value]


def _triple(summary: Mapping[str, object], name: str) -> tuple[float, float, float]:
    raw = summary.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        raise ValueError(f"XSP pressure-atlas {name} is missing")
    values = tuple(_number(value, name=name) for value in raw)
    return values  # type: ignore[return-value]


def _book_series(
    rows: Sequence[Mapping[str, object]],
    symbol: str,
) -> dict[str, tuple[float, ...]]:
    direction_sign = -1.0 if symbol == "SPXU" else 1.0
    prices: list[float] = []
    microprices: list[float] = []
    true_ranges: list[float] = []
    volumes: list[float] = []
    quotes: list[float] = []
    spreads: list[float] = []
    imbalances: list[float] = []
    size_pressures: list[float] = []
    for index, row in enumerate(rows):
        summary = _summary(row, symbol)
        mid_open, mid_high, mid_low, mid_close = _quad(summary, "mid_ohlc")
        micro_open, _, _, micro_close = _quad(summary, "microprice_ohlc")
        if min(mid_open, mid_high, mid_low, mid_close, micro_open, micro_close) <= 0.0:
            raise ValueError("XSP pressure-atlas price is not positive")
        if index == 0:
            prices.append(direction_sign * math.log(mid_open))
            microprices.append(direction_sign * math.log(micro_open))
        prices.append(direction_sign * math.log(mid_close))
        microprices.append(direction_sign * math.log(micro_close))
        true_ranges.append(10_000.0 * math.log(mid_high / mid_low))
        volume = _number(summary.get("cumulative_volume_delta"), name="volume delta")
        quote_count = _number(summary.get("valid_book_updates"), name="quote count")
        _, _, spread_close = _triple(summary, "spread_bps_min_max_last")
        _, _, _, imbalance_close = _quad(summary, "imbalance_open_min_max_close")
        proxy = summary.get("same_price_size_proxy")
        if not isinstance(proxy, Mapping):
            raise ValueError("XSP pressure-atlas size proxy is missing")
        size_pressure = (
            _number(proxy.get("bid_add"), name="bid add")
            + _number(proxy.get("ask_remove"), name="ask remove")
            - _number(proxy.get("bid_remove"), name="bid remove")
            - _number(proxy.get("ask_add"), name="ask add")
        )
        if volume < 0.0 or quote_count < 0.0 or spread_close < 0.0:
            raise ValueError("XSP pressure-atlas unsigned evidence is negative")
        volumes.append(volume)
        quotes.append(quote_count)
        spreads.append(spread_close)
        imbalances.append(direction_sign * imbalance_close)
        size_pressures.append(direction_sign * size_pressure)
    return {
        "price": tuple(prices),
        "microprice": tuple(microprices),
        "true_range": tuple(true_ranges),
        "volume": tuple(volumes),
        "quote_intensity": tuple(quotes),
        "spread": tuple(spreads),
        "imbalance": tuple(imbalances),
        "size_pressure": tuple(size_pressures),
    }


def _slope(values: Sequence[float], horizon: int, end: int = 0) -> float:
    stop = len(values) - 1 + end
    start = stop - horizon
    if start < 0 or stop >= len(values):
        raise ValueError("XSP pressure-atlas price clock is underwarmed")
    return 10_000.0 * (float(values[stop]) - float(values[start])) / horizon


def _mean(values: Sequence[float], horizon: int, end: int = 0) -> float:
    stop = len(values) + end
    start = stop - horizon
    if start < 0 or stop > len(values):
        raise ValueError("XSP pressure-atlas flow clock is underwarmed")
    return sum(float(value) for value in values[start:stop]) / horizon


def _differences(levels: Sequence[float]) -> tuple[float, float, float, float]:
    current, prior1, prior2, prior3 = (float(value) for value in levels)
    velocity = current - prior1
    prior_velocity = prior1 - prior2
    acceleration = velocity - prior_velocity
    prior_acceleration = prior_velocity - (prior2 - prior3)
    return current, velocity, acceleration, acceleration - prior_acceleration


def _energy_state(
    slope: float,
    velocity: float,
    *,
    target_sign: int,
) -> str:
    path = slope * target_sign
    change = velocity * target_sign
    if path > 0.0 and change > 0.0:
        return "TARGET_ACCELERATING"
    if path > 0.0 and change < 0.0:
        return "TARGET_CRESTING"
    if path < 0.0 and change < 0.0:
        return "OPPOSITION_ACCELERATING"
    if path < 0.0 and change > 0.0:
        return "OPPOSITION_CRESTING"
    return "TRANSITIONAL"


def _price_projection(
    values: Sequence[float],
    horizon: int,
    *,
    target_sign: int,
) -> dict[str, object]:
    current, velocity, acceleration, jerk = _differences(
        tuple(_slope(values, horizon, end) for end in (0, -1, -2, -3))
    )
    return {
        "slope_bps_per_second": current,
        "velocity_change_bps_per_second2": velocity,
        "acceleration_bps_per_second3": acceleration,
        "jerk_bps_per_second4": jerk,
        "direction": _direction(current),
        "target_shape": _target_shape(current, target_sign),
        "velocity_target_shape": _target_shape(velocity, target_sign),
        "acceleration_target_shape": _target_shape(acceleration, target_sign),
        "jerk_target_shape": _target_shape(jerk, target_sign),
        "target_energy_state": _energy_state(
            current,
            velocity,
            target_sign=target_sign,
        ),
    }


def _flow_projection(values: Sequence[float], horizon: int) -> dict[str, object]:
    current, velocity, acceleration, jerk = _differences(
        tuple(_mean(values, horizon, end) for end in (0, -1, -2, -3))
    )
    return {
        "mean": current,
        "velocity_change_per_second": velocity,
        "acceleration_per_second2": acceleration,
        "jerk_per_second3": jerk,
        "velocity_shape": _change_shape(velocity),
        "acceleration_shape": _change_shape(acceleration),
        "jerk_shape": _change_shape(jerk),
    }


def _directional_flow_projection(
    values: Sequence[float],
    horizon: int,
    *,
    target_sign: int,
) -> dict[str, object]:
    payload = _flow_projection(values, horizon)
    payload.update(
        {
            "target_shape": _target_shape(float(payload["mean"]), target_sign),
            "velocity_target_shape": _target_shape(
                float(payload["velocity_change_per_second"]),
                target_sign,
            ),
        }
    )
    return payload


def _consensus(values: Mapping[str, float], *, target_sign: int) -> dict[str, object]:
    signs = {name: _sign(value) for name, value in values.items()}
    unique = set(signs.values())
    full = next(iter(unique)) if len(unique) == 1 and 0 not in unique else 0
    return {
        "book_directions": {
            name: _direction(value) for name, value in values.items()
        },
        "full_alignment_direction": _direction(float(full)),
        "target_alignment": (
            "ALL_WITH_TARGET"
            if full == target_sign
            else "ALL_AGAINST_TARGET"
            if full == -target_sign
            else "MIXED_OR_FLAT"
        ),
    }


def _basis_projection(
    books: Mapping[str, Mapping[str, object]],
    key: str,
) -> float:
    cash = float(books["SPY"]["mid"][key])
    transport = (
        float(books["UPRO"]["mid"][key])
        + float(books["INVERSE_SPXU"]["mid"][key])
    ) / 6.0
    return cash - transport


def _basis_state(gap: float, velocity: float) -> str:
    prior = gap - velocity
    if _sign(gap) != _sign(prior) and _sign(gap) and _sign(prior):
        return "CROSSING"
    if abs(gap) < abs(prior):
        return "RECONVERGING"
    if abs(gap) > abs(prior):
        return "DIVERGING"
    return "STABLE"


def _joint_unsigned_state(
    books: Mapping[str, Mapping[str, object]],
) -> str:
    tr = [
        float(books[name]["true_range"]["velocity_change_per_second"])
        for name in _NORMALIZED_BOOKS
    ]
    volume = [
        float(books[name]["volume"]["velocity_change_per_second"])
        for name in _NORMALIZED_BOOKS
    ]
    if all(value > 0.0 for value in (*tr, *volume)):
        return "TR_AND_VOLUME_EXPANDING"
    if all(value < 0.0 for value in (*tr, *volume)):
        return "TR_AND_VOLUME_CONTRACTING"
    if all(value > 0.0 for value in tr):
        return "TR_ONLY_EXPANDING"
    if all(value > 0.0 for value in volume):
        return "VOLUME_ONLY_EXPANDING"
    return "MIXED"


def _liquidity_state(
    books: Mapping[str, Mapping[str, object]],
    *,
    target_sign: int,
) -> str:
    imbalance = [
        float(books[name]["imbalance"]["mean"]) * target_sign
        for name in _NORMALIZED_BOOKS
    ]
    size = [
        float(books[name]["size_pressure"]["mean"]) * target_sign
        for name in _NORMALIZED_BOOKS
    ]
    if all(value > 0.0 for value in (*imbalance, *size)):
        return "FULLY_SUPPORTIVE"
    if all(value < 0.0 for value in (*imbalance, *size)):
        return "FULLY_HOSTILE"
    if all(value >= 0.0 for value in imbalance):
        return "IMBALANCE_SUPPORTIVE"
    if all(value <= 0.0 for value in imbalance):
        return "IMBALANCE_HOSTILE"
    return "MIXED"


def _transport_state(mid_slopes: Mapping[str, float]) -> str:
    cash = _sign(mid_slopes["SPY"])
    transports = {
        _sign(mid_slopes["UPRO"]),
        _sign(mid_slopes["INVERSE_SPXU"]),
    }
    if cash and transports == {cash}:
        return "FULL_ACCEPTANCE"
    if cash and transports == {-cash}:
        return "FULL_REFUSAL"
    if cash == 0 and transports == {0}:
        return "NO_DISPLACEMENT"
    return "PARTIAL_OR_CONFLICTED"


def _lead_profile(
    rows: Sequence[Mapping[str, object]],
    horizon: int,
) -> dict[str, object]:
    leaders: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    for row in rows[-horizon:]:
        cross = row.get("cross_book")
        raw = cross.get("first_mid_move_leaders") if isinstance(cross, Mapping) else ()
        names = tuple(
            "INVERSE_SPXU" if str(value) == "SPXU" else str(value)
            for value in raw
        ) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()
        if names:
            leaders.update(names)
            signatures["+".join(sorted(names))] += 1
    if not leaders:
        dominant = "NO_MOVE"
    else:
        best = max(leaders.values())
        winners = sorted(name for name, count in leaders.items() if count == best)
        dominant = winners[0] if len(winners) == 1 else "TIE"
    return {
        "leader_counts": dict(sorted(leaders.items())),
        "simultaneous_signatures": dict(sorted(signatures.items())),
        "dominant_leader": dominant,
    }


def _ignition_state(
    consensus: Mapping[str, object],
    velocity_consensus: Mapping[str, object],
    lead: Mapping[str, object],
) -> str:
    direction = consensus["full_alignment_direction"]
    velocity = velocity_consensus["full_alignment_direction"]
    if direction in {"UP", "DOWN"} and direction == velocity:
        dominant = lead["dominant_leader"]
        if dominant == "SPY":
            return "ORDERED_CASH_IGNITION"
        if dominant in {"UPRO", "INVERSE_SPXU"}:
            return "ORDERED_TRANSPORT_IGNITION"
        return "SIMULTANEOUS_OR_TIED_IGNITION"
    if direction in {"UP", "DOWN"}:
        return "ALIGNED_PATH_WITHOUT_ALIGNED_ACCELERATION"
    return "CONFLICTED_OR_FLAT"


def _impulse_context(
    impulse: Mapping[str, object] | None,
    *,
    target_sign: int,
) -> dict[str, object]:
    raw = impulse.get("horizons") if isinstance(impulse, Mapping) else None
    horizons = tuple(
        row for row in raw if isinstance(row, Mapping)
    ) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ()
    output: list[dict[str, object]] = []
    for row in horizons:
        projected: dict[str, object] = {
            "bars": int(row.get("bars") or 0),
            "elapsed_minutes": _optional_number(row.get("elapsed_minutes")),
        }
        for source, target in (
            ("return_pct", "return_shape"),
            ("slope_pct_per_bar", "slope_shape"),
            ("slope_velocity_pct_per_bar", "velocity_shape"),
        ):
            value = _optional_number(row.get(source))
            projected[source] = value
            projected[target] = (
                _target_shape(value, target_sign) if value is not None else "MISSING"
            )
        output.append(projected)
    path_shapes = [
        row["slope_shape"] for row in output if row["slope_shape"] != "MISSING"
    ]
    velocity_shapes = [
        row["velocity_shape"] for row in output if row["velocity_shape"] != "MISSING"
    ]
    return {
        "ready": bool(output),
        "horizons": output,
        "path_state": (
            "ALL_WITH_TARGET"
            if path_shapes and all(value == "WITH_TARGET" for value in path_shapes)
            else "ALL_AGAINST_TARGET"
            if path_shapes and all(value == "AGAINST_TARGET" for value in path_shapes)
            else "MIXED"
            if path_shapes
            else "UNDERWARMED"
        ),
        "velocity_state": (
            "ALL_WITH_TARGET"
            if velocity_shapes and all(value == "WITH_TARGET" for value in velocity_shapes)
            else "ALL_AGAINST_TARGET"
            if velocity_shapes and all(value == "AGAINST_TARGET" for value in velocity_shapes)
            else "MIXED"
            if velocity_shapes
            else "UNDERWARMED"
        ),
        "volatility_state": _volatility_state(impulse),
    }


def _volatility_state(impulse: Mapping[str, object] | None) -> str:
    if not isinstance(impulse, Mapping):
        return "UNDERWARMED"
    velocity = _optional_number(impulse.get("atr_velocity_pct"))
    acceleration = _optional_number(impulse.get("atr_acceleration_pct"))
    if velocity is None or acceleration is None:
        return "UNDERWARMED"
    if velocity > 0.0 and acceleration > 0.0:
        return "EXPANDING_AND_ACCELERATING"
    if velocity > 0.0 and acceleration < 0.0:
        return "EXPANDING_AND_CRESTING"
    if velocity < 0.0 and acceleration < 0.0:
        return "CONTRACTING_AND_ACCELERATING"
    if velocity < 0.0 and acceleration > 0.0:
        return "CONTRACTING_AND_BOTTOMING"
    return "TRANSITIONAL"


def _daily_context(
    context: Mapping[str, object] | None,
    *,
    target_sign: int,
) -> dict[str, object]:
    windows = context.get("windows") if isinstance(context, Mapping) else None
    velocities = (
        context.get("return_velocity") if isinstance(context, Mapping) else None
    )
    accelerations = (
        context.get("return_acceleration") if isinstance(context, Mapping) else None
    )
    rows: dict[str, object] = {}
    for horizon in _DAILY_HORIZONS:
        key = str(horizon)
        window = windows.get(key) if isinstance(windows, Mapping) else None
        value = _optional_number(window.get("return")) if isinstance(window, Mapping) else None
        velocity = _optional_number(velocities.get(key)) if isinstance(velocities, Mapping) else None
        acceleration = (
            _optional_number(accelerations.get(key))
            if isinstance(accelerations, Mapping)
            else None
        )
        rows[key] = {
            "return": value,
            "return_shape": _target_shape(value, target_sign) if value is not None else "MISSING",
            "velocity": velocity,
            "velocity_shape": (
                _target_shape(velocity, target_sign) if velocity is not None else "MISSING"
            ),
            "acceleration": acceleration,
            "acceleration_shape": (
                _target_shape(acceleration, target_sign)
                if acceleration is not None
                else "MISSING"
            ),
        }
    shapes = [
        row["return_shape"]
        for row in rows.values()
        if isinstance(row, Mapping) and row["return_shape"] != "MISSING"
    ]
    return {
        "ready": len(shapes) == len(_DAILY_HORIZONS),
        "horizons": rows,
        "path_state": (
            "ALL_WITH_TARGET"
            if shapes and all(value == "WITH_TARGET" for value in shapes)
            else "ALL_AGAINST_TARGET"
            if shapes and all(value == "AGAINST_TARGET" for value in shapes)
            else "MIXED"
            if shapes
            else "UNDERWARMED"
        ),
        "tr_velocity_shape": _change_shape(
            _optional_number(context.get("tr_velocity")) or 0.0
        ) if isinstance(context, Mapping) and context.get("tr_velocity") is not None else "MISSING",
        "tr_acceleration_shape": _change_shape(
            _optional_number(context.get("tr_acceleration")) or 0.0
        ) if isinstance(context, Mapping) and context.get("tr_acceleration") is not None else "MISSING",
    }


def _cross_scale_state(states: Sequence[str]) -> str:
    ready = [value for value in states if value != "UNDERWARMED"]
    if len(ready) != len(states):
        return "UNDERWARMED"
    if all(value == "ALL_WITH_TARGET" for value in ready):
        return "ALL_CLOCKS_WITH_TARGET"
    if all(value == "ALL_AGAINST_TARGET" for value in ready):
        return "ALL_CLOCKS_AGAINST_TARGET"
    return "MIXED_CLOCKS"


def project_xsp_pressure_atlas(
    records: Sequence[Mapping[str, object]],
    *,
    as_of_utc: datetime,
    target_direction: str,
    xsp_impulse: Mapping[str, object] | None = None,
    spy_impulse: Mapping[str, object] | None = None,
    daily_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project causal morphology without selecting a winner or changing permission."""

    target = str(target_direction).strip().lower()
    if target not in {"up", "down"}:
        raise ValueError("XSP pressure-atlas target direction is invalid")
    target_sign = 1 if target == "up" else -1
    as_of = _utc(as_of_utc)
    rows, tape_generation, trading_date = _formation(records, as_of_utc=as_of)
    physical = {symbol: _book_series(rows, symbol) for symbol in _PHYSICAL_BOOKS}
    series = {
        "SPY": physical["SPY"],
        "UPRO": physical["UPRO"],
        "INVERSE_SPXU": physical["SPXU"],
    }

    horizons: dict[str, object] = {}
    for horizon in XSP_PRESSURE_ATLAS_HORIZONS_SECONDS:
        books: dict[str, dict[str, object]] = {}
        for name in _NORMALIZED_BOOKS:
            values = series[name]
            books[name] = {
                "mid": _price_projection(
                    values["price"], horizon, target_sign=target_sign
                ),
                "microprice": _price_projection(
                    values["microprice"], horizon, target_sign=target_sign
                ),
                "true_range": _flow_projection(values["true_range"], horizon),
                "volume": _flow_projection(values["volume"], horizon),
                "quote_intensity": _flow_projection(
                    values["quote_intensity"], horizon
                ),
                "spread": _flow_projection(values["spread"], horizon),
                "imbalance": _directional_flow_projection(
                    values["imbalance"], horizon, target_sign=target_sign
                ),
                "size_pressure": _directional_flow_projection(
                    values["size_pressure"], horizon, target_sign=target_sign
                ),
            }
        slopes = {
            name: float(books[name]["mid"]["slope_bps_per_second"])
            for name in _NORMALIZED_BOOKS
        }
        velocities = {
            name: float(books[name]["mid"]["velocity_change_bps_per_second2"])
            for name in _NORMALIZED_BOOKS
        }
        consensus = _consensus(slopes, target_sign=target_sign)
        velocity_consensus = _consensus(velocities, target_sign=target_sign)
        basis = {
            "cash_minus_scaled_transport_bps_per_second": _basis_projection(
                books, "slope_bps_per_second"
            ),
            "velocity_change_bps_per_second2": _basis_projection(
                books, "velocity_change_bps_per_second2"
            ),
            "acceleration_bps_per_second3": _basis_projection(
                books, "acceleration_bps_per_second3"
            ),
            "jerk_bps_per_second4": _basis_projection(
                books, "jerk_bps_per_second4"
            ),
            "transport_scale": "UPRO_and_inverse_SPXU_each_divided_by_known_3x_exposure",
        }
        basis["state"] = _basis_state(
            float(basis["cash_minus_scaled_transport_bps_per_second"]),
            float(basis["velocity_change_bps_per_second2"]),
        )
        lead = _lead_profile(rows, horizon)
        horizons[str(horizon)] = {
            "books": books,
            "path_consensus": consensus,
            "velocity_consensus": velocity_consensus,
            "basis": basis,
            "lead": lead,
            "morphology": {
                "ignition": _ignition_state(consensus, velocity_consensus, lead),
                "transport": _transport_state(slopes),
                "volatility_flow": _joint_unsigned_state(books),
                "liquidity": _liquidity_state(books, target_sign=target_sign),
            },
        }

    xsp_slow = _impulse_context(xsp_impulse, target_sign=target_sign)
    spy_slow = _impulse_context(spy_impulse, target_sign=target_sign)
    daily = _daily_context(daily_context, target_sign=target_sign)
    seconds_states = [
        str(horizons[str(horizon)]["path_consensus"]["target_alignment"])
        for horizon in (5, 15, 30, 45)
    ]
    seconds_state = (
        "ALL_WITH_TARGET"
        if all(value == "ALL_WITH_TARGET" for value in seconds_states)
        else "ALL_AGAINST_TARGET"
        if all(value == "ALL_AGAINST_TARGET" for value in seconds_states)
        else "MIXED"
    )
    payload: dict[str, object] = {
        "schema": XSP_PRESSURE_ATLAS_VERSION,
        "authority": XSP_PRESSURE_ATLAS_AUTHORITY,
        "as_of_utc": as_of.isoformat(),
        "target_direction": target,
        "source": {
            "pressure_tape_generation_sha256": tape_generation,
            "trading_date": trading_date,
            "formation_seconds": XSP_PRESSURE_ATLAS_FORMATION_SECONDS,
            "first_bucket_start_utc": rows[0]["bucket_start_utc"],
            "last_bucket_start_utc": rows[-1]["bucket_start_utc"],
            "source_record_ids_sha256": hashlib.sha256(
                _canonical([row["record_id"] for row in rows])
            ).hexdigest(),
        },
        "seconds": {
            "horizons_seconds": list(XSP_PRESSURE_ATLAS_HORIZONS_SECONDS),
            "horizons": horizons,
        },
        "slower_context": {
            "xsp_impulse": xsp_slow,
            "spy_impulse": spy_slow,
            "daily_5_10_21_42_63_84": daily,
            "seconds_state": seconds_state,
            "cross_scale_state": _cross_scale_state(
                (
                    seconds_state,
                    str(xsp_slow["path_state"]),
                    str(spy_slow["path_state"]),
                    str(daily["path_state"]),
                )
            ),
        },
        "direction_authority": "opening_edge_v3_crown_only",
        "classifier": "none",
        "permission": "none",
        "outcomes": None,
        "order_authority": "none",
        "submitted_orders": 0,
    }
    payload["projection_id"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload
