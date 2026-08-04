"""Accumulate exact V18 onset morphology without trading authority."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ..chart_data.history import read_cache
from ..chart_data.series import OhlcvBar
from ..client import IBKRClient
from ..config import load_config
from ..live.capital import load_live_capital_plan
from ..live.capital_packages import load_allocated_live_selection
from ..news.contract import (
    load_news_history,
    observe_news_signal,
    publication_id,
    select_news_snapshot_at,
)
from ..time_utils import ET_ZONE
from .live_calibration import LiveCalibrationLedger, calibration_fingerprint
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    MCL_LIVE_LEDGER_PATH,
    MCL_LIVE_SOURCE_SCHEMA,
    MCL_LIVE_SOURCE_VERSION,
    _bar_map,
    load_mcl_live_selection_from_mapping,
    mcl_live_contracts,
)
from .mcl_predictive_generation import (
    MCL_PREDICTIVE_GENERATION_SCHEMA_V1,
    MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
    load_mcl_predictive_generation,
)
from .mcl_predictive_onset import (
    MclOnsetNewsContext,
    MclWeeklyPrior,
    combine_mcl_predictive_onset_atlas,
    project_mcl_completed_bar_onset,
    project_mcl_event_onset,
)
from .mcl_predictive_velocity import project_mcl_velocity_jerk_handoff
from .mcl_turn_tape import MCL_TURN_TAPE_SCHEMA, MCL_TURN_TAPE_STATE_DIR
from .mcl_two_speed_auction import (
    MCL_TWO_SPEED_AUCTION_VERSION,
    MclAuctionBar,
    MclTwoSpeedAuctionEngine,
)


MCL_PREDICTIVE_ACCUMULATOR_VERSION = "mcl.predictive-onset-accumulator.v1"
MCL_PREDICTIVE_ACCUMULATOR_SCHEMA = "mcl.predictive-onset-treatment.v1"
MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY = (
    "prospective_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_PREDICTIVE_GENERATION_SCHEMA = MCL_PREDICTIVE_GENERATION_SCHEMA_V1
MCL_PREDICTIVE_LEDGER_PATH = (
    Path.home() / ".local/state/tradebot/research/mcl_predictive_onset.jsonl"
)
MCL_PREDICTIVE_GENERATION_PATH = MCL_PREDICTIVE_RUNTIME_GENERATION_PATH
MCL_PREDICTIVE_MANIFEST_PATH = Path("db/MCL/dated/hydration_manifest.json")
MCL_PREDICTIVE_NEWS_DIR = Path.home() / ".local/state/tradebot/news/history"
_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class MclPredictiveMinute:
    ts: datetime
    session: date | None
    contract_key: str
    cl: OhlcvBar
    mcl: OhlcvBar

def _utc(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()

def _identity(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _is_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)

def _validated_treatment(evidence: Mapping[str, object]) -> dict[str, object]:
    value = dict(evidence)
    treatment_id = str(value.pop("treatment_id", ""))
    if (
        evidence.get("schema") != MCL_PREDICTIVE_ACCUMULATOR_SCHEMA
        or evidence.get("authority") != MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY
        or evidence.get("outcomes_exposed") is not False
        or evidence.get("submitted_orders") != 0
        or not _is_sha(treatment_id)
        or _identity(value) != treatment_id
    ):
        raise ValueError("MCL predictive treatment identity drifted")
    return dict(evidence)

def mcl_predictive_treatments(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    seen = set()
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != MCL_PREDICTIVE_ACCUMULATOR_VERSION
            or not isinstance(evidence, Mapping)
        ):
            continue
        value = _validated_treatment(evidence)
        treatment_id = str(value["treatment_id"])
        if treatment_id in seen:
            raise ValueError("MCL predictive ledger repeats one treatment")
        seen.add(treatment_id)
        output.append(value)
    return sorted(output, key=lambda row: _utc(row["raw_turn_at_utc"]))

def _seed_treatments(
    generation: Mapping[str, object], *, repository_root: Path
) -> list[dict[str, object]]:
    artifacts = generation["artifacts"]
    assert isinstance(artifacts, Mapping)

    def load(name: str) -> dict[str, object]:
        item = artifacts[name]
        assert isinstance(item, Mapping)
        value = json.loads((repository_root / str(item["path"])).read_text())
        if not isinstance(value, Mapping):
            raise ValueError(f"MCL predictive seed {name} is invalid")
        return dict(value)

    stage88 = load("stage88_seed")
    stage89 = load("stage89_seed")
    stage90 = load("stage90_seed")
    by88 = {str(row["raw_turn_at_utc"]): row for row in stage88["treatments"]}
    by89 = {str(row["raw_turn_at_utc"]): row for row in stage89["treatments"]}
    seeds = []
    for row in stage90["treatments"]:
        at = str(row["raw_turn_at_utc"])
        source88 = by88[at]
        source89 = by89[at]
        event = source88["event_morphology"]
        if (
            _identity(event) != row["event_morphology_sha256"]
            or _canonical(source89["velocity_jerk"])
            != _canonical(row["velocity_jerk"])
        ):
            raise ValueError("MCL predictive seed morphology drifted")
        seeds.append(
            _treatment(
                raw_turn_at_utc=at,
                raw_direction=int(row["raw_direction"]),
                proposed=bool(row["proposed"]),
                admitted=bool(row["admitted"]),
                route=row.get("route"),
                routed_direction=source88.get("routed_direction"),
                event_morphology=event,
                velocity_jerk=row["velocity_jerk"],
                completed_bar=row["completed_bar"],
                cross_scale_atlas=row["cross_scale_atlas"],
                news_source=row.get("news_source"),
                source={
                    "kind": "frozen_stage90_seed",
                    "stage88_sha256": artifacts["stage88_seed"]["sha256"],
                    "stage89_sha256": artifacts["stage89_seed"]["sha256"],
                    "stage90_sha256": artifacts["stage90_seed"]["sha256"],
                    "selection_id": None,
                    "source_checkpoint_id": None,
                    "source_event_id": None,
                    "raw_decision": None,
                    "maturation_decision": None,
                },
            )
        )
    return seeds

def _treatment(
    *,
    raw_turn_at_utc: object,
    raw_direction: int,
    proposed: bool,
    admitted: bool,
    route: object,
    routed_direction: object,
    event_morphology: object,
    velocity_jerk: object,
    completed_bar: object,
    cross_scale_atlas: object,
    news_source: object,
    source: Mapping[str, object],
) -> dict[str, object]:
    body = {
        "schema": MCL_PREDICTIVE_ACCUMULATOR_SCHEMA,
        "authority": MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY,
        "strategy_version": MCL_TWO_SPEED_AUCTION_VERSION,
        "raw_turn_at_utc": _utc(raw_turn_at_utc).isoformat(),
        "raw_direction": int(raw_direction),
        "proposed": bool(proposed),
        "admitted": bool(admitted),
        "route": route,
        "routed_direction": routed_direction,
        "source": dict(source),
        "event_morphology": event_morphology,
        "velocity_jerk": velocity_jerk,
        "completed_bar": completed_bar,
        "cross_scale_atlas": cross_scale_atlas,
        "news_source": news_source,
        "winner": None,
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
    return {**body, "treatment_id": _identity(body)}

def _append_treatment(
    ledger: LiveCalibrationLedger,
    treatment: Mapping[str, object],
    *,
    recorded_at: datetime,
) -> bool:
    value = _validated_treatment(treatment)
    existing = {
        row["treatment_id"]
        for row in mcl_predictive_treatments(tuple(ledger.records()))
    }
    if value["treatment_id"] in existing:
        return False
    turn = _utc(value["raw_turn_at_utc"])
    ledger.checkpoint(
        evaluation_as_of=turn + timedelta(minutes=5),
        strategy_id=MCL_TWO_SPEED_AUCTION_VERSION,
        strategy_version=MCL_PREDICTIVE_ACCUMULATOR_VERSION,
        trading_date=(_mcl_session(turn) or turn.date()).isoformat(),
        session="MCL_PREDICTIVE_ONSET",
        status="EVALUATED",
        evidence=value,
        recorded_at=max(_utc(recorded_at), turn + timedelta(minutes=5)),
    )
    return True

def _source_candidates(
    records: Sequence[Mapping[str, object]],
    *,
    selection_id: str,
    eligible_start: datetime,
) -> list[dict[str, object]]:
    values = {}
    for record in records:
        evidence = record.get("evidence")
        if (
            record.get("kind") != "checkpoint"
            or record.get("strategy_version") != MCL_LIVE_SOURCE_VERSION
            or not isinstance(evidence, Mapping)
            or evidence.get("schema") != MCL_LIVE_SOURCE_SCHEMA
            or evidence.get("selection_id") != selection_id
        ):
            continue
        raw = evidence.get("last_raw_turn")
        if not isinstance(raw, Mapping):
            continue
        decision = raw.get("decision")
        event_id = str(raw.get("event_id") or "")
        turn = _utc(raw.get("observed_at_utc"))
        if (
            turn < eligible_start
            or not isinstance(decision, Mapping)
            or _identity(decision) != event_id
            or _utc(decision.get("observed_at_utc")) != turn
            or decision.get("phase") != "RAW_TURN"
            or decision.get("raw_direction") not in (-1, 1)
        ):
            raise ValueError("MCL live raw-turn source identity drifted")
        values[event_id] = {
            **dict(raw),
            "selection_id": selection_id,
            "source_checkpoint_id": record["checkpoint_id"],
        }
    return sorted(values.values(), key=lambda row: _utc(row["observed_at_utc"]))

def _read_event_window(
    directory: Path,
    *,
    turn: datetime,
    expected_generation_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    start = turn - timedelta(seconds=60)
    end = turn + timedelta(minutes=5)
    days = {
        (start + timedelta(days=offset)).date()
        for offset in range((end.date() - start.date()).days + 1)
    }
    rows = []
    for day in sorted(days):
        path = directory / f"{day.isoformat()}.jsonl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                raw = handle.read()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("MCL turn tape row is not an object")
            if row.get("schema") != MCL_TURN_TAPE_SCHEMA:
                raise ValueError("MCL turn tape contains late or unknown evidence")
            stamp = _utc(row.get("bucket_start_utc"))
            if start <= stamp < end:
                rows.append(dict(row))
    rows.sort(key=lambda row: _utc(row["bucket_start_utc"]))
    if not rows:
        return None
    first = _utc(rows[0]["bucket_start_utc"])
    last = _utc(rows[-1]["bucket_start_utc"])
    if first > start or last + timedelta(seconds=1) < end:
        return None
    generations = {str(row.get("generation_sha256")) for row in rows}
    if generations != {expected_generation_sha256}:
        raise ValueError("MCL event-window generation drifted")
    return rows, {
        "records": len(rows),
        "first_bucket_utc": first.isoformat(),
        "last_bucket_utc": last.isoformat(),
        "prefix_end_exclusive_utc": (last + timedelta(seconds=1)).isoformat(),
        "generation_sha256": expected_generation_sha256,
        "window_sha256": _identity([row.get("record_id") for row in rows]),
    }

async def _recent_bars(
    client: IBKRClient, *, cl_contract, mcl_contract, observed_at: datetime
) -> tuple[dict[str, dict[datetime, OhlcvBar]], dict[str, object]]:
    now = _utc(observed_at)
    raw = await asyncio.gather(
        *(
            client.historical_bars_ohlcv(
                contract,
                duration_str="3 D",
                bar_size="1 min",
                use_rth=False,
                what_to_show="TRADES",
                cache_ttl_sec=0,
            )
            for contract in (cl_contract, mcl_contract)
        )
    )
    cutoff = now.replace(second=0, microsecond=0)
    maps = {
        symbol: _bar_map(rows, cutoff=cutoff, name=f"predictive {symbol}")
        for symbol, rows in zip(("CL", "MCL"), raw, strict=True)
    }
    common = sorted(set(maps["CL"]) & set(maps["MCL"]))
    if len(common) < 500 or not 0 <= (now - common[-1]).total_seconds() <= 8 * 60:
        raise ValueError("MCL predictive finalized bars are incomplete or stale")
    return maps, {
        "cutoff_utc": cutoff.isoformat(),
        "first_common_close_utc": common[0].isoformat(),
        "last_common_close_utc": common[-1].isoformat(),
        "common_rows": len(common),
    }

def _mcl_session(stamp: datetime) -> date | None:
    local = _utc(stamp).astimezone(ET_ZONE)
    clock = local.time().replace(tzinfo=None)
    if time(17) <= clock < time(18):
        return None
    return (local + timedelta(days=1)).date() if clock >= time(18) else local.date()

def _minute_pairs(
    maps: Mapping[str, Mapping[datetime, OhlcvBar]], *, contract_key: str
) -> list[MclPredictiveMinute]:
    return [
        MclPredictiveMinute(
            stamp,
            _mcl_session(stamp),
            contract_key,
            maps["CL"][stamp],
            maps["MCL"][stamp],
        )
        for stamp in sorted(set(maps["CL"]) & set(maps["MCL"]))
    ]


def _manifest_minutes(
    manifest_path: Path,
    *,
    contracts: Mapping[str, Mapping[str, object]],
    fresh: Mapping[str, Mapping[datetime, OhlcvBar]],
    expected_sha256: str,
) -> tuple[list[MclPredictiveMinute], dict[str, object]]:
    if _sha256(manifest_path) != expected_sha256:
        raise ValueError("MCL predictive historical manifest drifted")
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for pair in manifest.get("pairs", []):
        roots = pair.get("roots", {})
        if all(
            int(roots.get(symbol, {}).get("con_id") or 0)
            == int(contracts[symbol]["con_id"])
            and str(roots.get(symbol, {}).get("local_symbol") or "")
            == str(contracts[symbol]["local_symbol"])
            for symbol in ("CL", "MCL")
        ):
            candidates.append(pair)
    if len(candidates) != 1:
        raise ValueError("MCL predictive manifest has no unique selected pair")
    pair = candidates[0]
    roots = pair["roots"]
    cached = {}
    for symbol in ("CL", "MCL"):
        path = Path(roots[symbol]["path"])
        if _sha256(path) != roots[symbol]["sha256"]:
            raise ValueError(f"MCL predictive {symbol} cache drifted")
        rows = read_cache(path)
        if len(rows) != int(roots[symbol]["rows"]):
            raise ValueError(f"MCL predictive {symbol} cache row count drifted")
        cached[symbol] = {
            row.ts.replace(tzinfo=timezone.utc): OhlcvBar(
                row.ts.replace(tzinfo=timezone.utc),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            )
            for row in rows
        }
    historical_common = sorted(set(cached["CL"]) & set(cached["MCL"]))
    if not historical_common:
        raise ValueError("MCL predictive selected cache has no common rows")
    last = historical_common[-1]
    merged = {
        symbol: {
            **cached[symbol],
            **{stamp: bar for stamp, bar in fresh[symbol].items() if stamp > last},
        }
        for symbol in ("CL", "MCL")
    }
    return _minute_pairs(merged, contract_key=str(pair["contract_month_suffix"])), {
        "manifest_sha256": expected_sha256,
        "contract_month_suffix": pair["contract_month_suffix"],
        "cache_sha256": {
            symbol: roots[symbol]["sha256"] for symbol in ("CL", "MCL")
        },
        "historical_last_common_close_utc": last.isoformat(),
        "merged_last_common_close_utc": max(
            set(merged["CL"]) & set(merged["MCL"])
        ).isoformat(),
    }


def _finalized_sessions(
    rows: Sequence[MclPredictiveMinute],
) -> list[dict[str, object]]:
    grouped: dict[date, list[MclPredictiveMinute]] = defaultdict(list)
    for row in rows:
        if row.session is not None:
            grouped[row.session].append(row)
    sessions = []
    for session, values in sorted(grouped.items()):
        first = values[0].ts.astimezone(ET_ZONE).time().replace(tzinfo=None)
        last = values[-1].ts.astimezone(ET_ZONE).time().replace(tzinfo=None)
        if not time(18) < first <= time(18, 5) or last < time(16, 55):
            continue
        sessions.append(
            {
                "session": session,
                "open": float(values[0].cl.open),
                "high": max(float(row.cl.high) for row in values),
                "low": min(float(row.cl.low) for row in values),
                "close": float(values[-1].cl.close),
                "close_at_utc": values[-1].ts,
            }
        )
    return sessions


def _session_tr(rows: Sequence[Mapping[str, object]], index: int) -> float:
    previous = float(rows[index - 1]["close"])
    return 100.0 * max(
        float(rows[index]["high"]) - float(rows[index]["low"]),
        abs(float(rows[index]["high"]) - previous),
        abs(float(rows[index]["low"]) - previous),
    ) / previous


def _weekly_prior(
    sessions: Sequence[Mapping[str, object]], *, session: date
) -> MclWeeklyPrior | None:
    days = [row["session"] for row in sessions]
    end = bisect_left(days, session)
    if end < 7:
        return None
    current = 100.0 * (
        float(sessions[end - 1]["close"]) / float(sessions[end - 6]["close"]) - 1.0
    )
    prior = 100.0 * (
        float(sessions[end - 2]["close"]) / float(sessions[end - 7]["close"]) - 1.0
    )
    current_tr = [_session_tr(sessions, index) for index in range(end - 5, end)]
    prior_tr = [_session_tr(sessions, index) for index in range(end - 6, end - 1)]
    sign = 1 if current > 0.0 else -1 if current < 0.0 else 0
    age = 1
    cursor = end - 1
    while cursor >= 7:
        value = 100.0 * (
            float(sessions[cursor - 1]["close"])
            / float(sessions[cursor - 6]["close"])
            - 1.0
        )
        if (1 if value > 0.0 else -1 if value < 0.0 else 0) != sign:
            break
        age += 1
        cursor -= 1
    return MclWeeklyPrior(
        as_of_utc=_utc(sessions[end - 1]["close_at_utc"]),
        return_pct=current,
        return_velocity_pct=current - prior,
        tr_velocity_pct=(sum(current_tr) - sum(prior_tr)) / 5.0,
        state_age_sessions=age,
    )


def _news_context(
    snapshots: Sequence[Mapping[str, object]], *, treatment_at: datetime
) -> tuple[MclOnsetNewsContext | None, dict[str, object] | None]:
    selected = select_news_snapshot_at(snapshots, as_of=treatment_at)
    if selected is None:
        return None, None
    current = observe_news_signal(selected, symbol="MCL", as_of=treatment_at)
    selected_index = max(
        index for index, row in enumerate(snapshots) if row == selected
    )
    previous_snapshot = select_news_snapshot_at(
        tuple(row for index, row in enumerate(snapshots) if index != selected_index),
        as_of=treatment_at,
    )
    previous = (
        observe_news_signal(previous_snapshot, symbol="MCL", as_of=treatment_at)
        if previous_snapshot is not None
        else None
    )
    current_at = _utc(current.snapshot_as_of_utc)
    current_pressure = current.direction * current.impact / 100.0
    previous_pressure = (
        previous.direction * previous.impact / 100.0 if previous else 0.0
    )
    delta = (
        current_pressure - previous_pressure
        if previous is not None
        else 0.0
    )
    elapsed = (
        (current_at - _utc(previous.snapshot_as_of_utc)).total_seconds() / 3600.0
        if previous is not None
        else 0.0
    )
    velocity = delta / elapsed if elapsed > 0.0 else 0.0
    context = MclOnsetNewsContext(
        published_at_utc=current_at,
        horizon_hours=float(current.horizon_hours),
        signed_pressure=current_pressure,
        pressure_delta=delta,
        pressure_velocity_per_hour=velocity,
        impact=current.impact / 100.0,
        confidence=current.confidence,
    )
    source = {
        **current.as_payload(),
        "publication_id": selected.get("publication_id") or publication_id(selected),
        "prior_publication_id": (
            previous_snapshot.get("publication_id")
            or publication_id(previous_snapshot)
            if previous_snapshot is not None
            else None
        ),
        "signed_pressure": current_pressure,
        "pressure_delta": delta,
        "pressure_velocity_per_hour": velocity,
        "treatment_age_hours": (treatment_at - current_at).total_seconds() / 3600.0,
    }
    return context, source


def _aggregate(rows: Sequence[MclPredictiveMinute]) -> MclAuctionBar:
    if len(rows) != 5:
        raise ValueError("MCL predictive aggregation requires five minutes")

    def side(name: str) -> OhlcvBar:
        values = [getattr(row, name) for row in rows]
        return OhlcvBar(
            rows[-1].ts,
            float(values[0].open),
            max(float(value.high) for value in values),
            min(float(value.low) for value in values),
            float(values[-1].close),
            sum(float(value.volume) for value in values),
        )

    return MclAuctionBar(rows[-1].contract_key, side("cl"), side("mcl"))


def _replay_treatment(
    candidate: Mapping[str, object],
    *,
    recent_minutes: Sequence[MclPredictiveMinute],
    weekly_prior: MclWeeklyPrior | None,
    news: MclOnsetNewsContext | None,
) -> tuple[object, object, object]:
    turn = _utc(candidate["observed_at_utc"])
    engine = MclTwoSpeedAuctionEngine()
    minute_rows = []
    bars: deque[MclAuctionBar] = deque(maxlen=120)
    decisions = deque(maxlen=120)
    completed = raw = maturation = None
    previous = None
    for row in recent_minutes:
        if previous is None or row.ts - previous.ts != timedelta(minutes=1):
            minute_rows.clear()
        minute_rows.append(row)
        if row.ts.minute % 5 == 0:
            if len(minute_rows) == 5:
                bar = _aggregate(minute_rows)
                decision = engine.update(bar)
                bars.append(bar)
                decisions.append(decision)
                if decision.observed_at_utc == turn:
                    raw = decision
                    if _canonical(decision.as_payload()) != _canonical(candidate["decision"]):
                        raise ValueError("MCL predictive replay disagrees with live raw turn")
                    completed = project_mcl_completed_bar_onset(
                        tuple(decisions),
                        tuple(bars),
                        weekly_prior=weekly_prior,
                        news=news,
                        four_hour_clock="finalized_sparse",
                    )
                if decision.signal_at_utc == turn and decision.phase == "MATURATION":
                    maturation = decision
            minute_rows.clear()
        previous = row
        if row.ts >= turn + timedelta(minutes=5):
            break
    if raw is None or maturation is None or completed is None:
        raise ValueError("MCL predictive replay did not complete raw turn and maturation")
    return raw, maturation, completed


def _cohort(
    treatments: Sequence[Mapping[str, object]], gate: Mapping[str, object]
) -> dict[str, object]:
    directions = Counter(
        "up" if int(row["raw_direction"]) > 0 else "down" for row in treatments
    )
    routes = Counter(
        str(row["route"])
        for row in treatments
        if bool(row["admitted"]) and row.get("route") is not None
    )
    admitted = sum(bool(row["admitted"]) for row in treatments)
    resolved = sum(
        row["velocity_jerk"].get("handoff") != "UNRESOLVED" for row in treatments
    )
    gates = {
        "at_least_complete_turns": len(treatments) >= int(gate["complete_turns"]),
        "at_least_each_raw_direction": all(
            directions[value] >= int(gate["each_raw_direction"])
            for value in ("up", "down")
        ),
        "at_least_admitted_turns": admitted >= int(gate["admitted_turns"]),
        "at_least_each_admitted_route": all(
            routes[value] >= int(gate["each_admitted_route"])
            for value in ("continuation", "failed_auction")
        ),
        "at_least_resolved_handoffs": resolved >= int(gate["resolved_handoffs"]),
    }
    return {
        "complete_turns": len(treatments),
        "directions": dict(sorted(directions.items())),
        "admitted_turns": admitted,
        "admitted_routes": dict(sorted(routes.items())),
        "resolved_handoffs": resolved,
        "gates": gates,
        "verdict": (
            "ELIGIBLE_FOR_MATCHED_CONTROL_PREREGISTRATION"
            if all(gates.values())
            else "ACCUMULATE"
        ),
    }


async def advance_mcl_predictive_accumulator(
    *,
    ledger: LiveCalibrationLedger,
    live_ledger: LiveCalibrationLedger,
    generation: Mapping[str, object],
    capital_plan_path: Path,
    manifest_path: Path,
    turn_tape_dir: Path,
    news_history_dir: Path,
    observed_at: datetime,
    repository_root: Path = _ROOT,
    seed_only: bool = False,
) -> dict[str, object]:
    now = _utc(observed_at)
    appended = 0
    for treatment in _seed_treatments(generation, repository_root=repository_root):
        appended += _append_treatment(ledger, treatment, recorded_at=now)
    treatments = mcl_predictive_treatments(tuple(ledger.records()))
    inherited = generation.get("inherited_prefix")
    if isinstance(inherited, Mapping):
        expected_ids = list(inherited.get("treatment_ids") or ())
        actual_ids = [row["treatment_id"] for row in treatments[: len(expected_ids)]]
        if actual_ids != expected_ids:
            raise ValueError("MCL predictive inherited treatment prefix drifted")
    gate = generation["cohort_gate"]
    assert isinstance(gate, Mapping)
    if seed_only:
        return {"appended": appended, "cohort": _cohort(treatments, gate)}

    plan = load_live_capital_plan(capital_plan_path)
    selection, _path, _sha = load_allocated_live_selection(
        plan, sleeve_id=MCL_LIVE_CAPITAL_SLEEVE, repository_root=repository_root
    )
    selected = load_mcl_live_selection_from_mapping(selection)
    selection_id = str(selected["selection_id"])
    if selection_id != generation.get("selection_id"):
        raise ValueError("MCL predictive generation selection drifted")
    candidates = _source_candidates(
        tuple(live_ledger.records()),
        selection_id=selection_id,
        eligible_start=_utc(generation["eligible_start_utc"]),
    )
    processed = {
        str(row["source"].get("source_event_id"))
        for row in treatments
        if isinstance(row.get("source"), Mapping)
        and row["source"].get("source_event_id")
    }
    ready = [
        row
        for row in candidates
        if row["event_id"] not in processed
        and now >= _utc(row["observed_at_utc"]) + timedelta(minutes=5)
    ]
    complete = []
    event_generation = str(generation["turn_tape_generation_sha256"])
    for candidate in ready:
        window = _read_event_window(
            turn_tape_dir,
            turn=_utc(candidate["observed_at_utc"]),
            expected_generation_sha256=event_generation,
        )
        if window is not None:
            complete.append((candidate, window))
    if not complete:
        return {
            "appended": appended,
            "pending_candidates": len(ready),
            "cohort": _cohort(treatments, gate),
        }

    config = load_config()
    if not config.readonly:
        raise ValueError("MCL predictive accumulator requires IBKR_READONLY=1")
    client = IBKRClient(config)
    await client.connect()
    try:
        cl_contract, mcl_contract = mcl_live_contracts(selected)
        fresh, snapshot = await _recent_bars(
            client,
            cl_contract=cl_contract,
            mcl_contract=mcl_contract,
            observed_at=now,
        )
        common = sorted(set(fresh["CL"]) & set(fresh["MCL"]))
        snapshot["content_sha256"] = _identity(
            [
                [
                    stamp.isoformat(),
                    *[
                        getattr(fresh[symbol][stamp], field)
                        for symbol in ("CL", "MCL")
                        for field in ("open", "high", "low", "close", "volume")
                    ],
                ]
                for stamp in common
            ]
        )
    finally:
        await client.disconnect()
    contracts = {
        symbol: dict(selected["contracts"][symbol]) for symbol in ("CL", "MCL")
    }
    manifest_expected = str(generation["historical_manifest_sha256"])
    session_minutes, manifest = _manifest_minutes(
        manifest_path,
        contracts=contracts,
        fresh=fresh,
        expected_sha256=manifest_expected,
    )
    sessions = _finalized_sessions(session_minutes)
    recent = _minute_pairs(fresh, contract_key=str(contracts["MCL"]["expiry"])[:6])
    snapshots = [
        row
        for path in sorted(news_history_dir.glob("*.jsonl"))
        for row in load_news_history(path)
    ]
    news_fingerprint = calibration_fingerprint(
        [
            {"path": path.name, "sha256": _sha256(path)}
            for path in sorted(news_history_dir.glob("*.jsonl"))
        ]
    )
    for candidate, (events, event_evidence) in complete:
        turn = _utc(candidate["observed_at_utc"])
        session = _mcl_session(turn)
        if session is None:
            raise ValueError("MCL predictive raw turn occurred during maintenance")
        news, news_source = _news_context(snapshots, treatment_at=turn)
        raw, maturation, completed = _replay_treatment(
            candidate,
            recent_minutes=recent,
            weekly_prior=_weekly_prior(sessions, session=session),
            news=news,
        )
        event = project_mcl_event_onset(
            events,
            raw_turn_at_utc=turn,
            raw_direction=int(raw.raw_direction),
            prefix_start_utc=_utc(event_evidence["first_bucket_utc"]),
            prefix_end_utc=_utc(event_evidence["prefix_end_exclusive_utc"]),
        )
        velocity = project_mcl_velocity_jerk_handoff(event)
        atlas = combine_mcl_predictive_onset_atlas(completed, event)
        treatment = _treatment(
            raw_turn_at_utc=turn,
            raw_direction=int(raw.raw_direction),
            proposed=raw.proposed_direction is not None,
            admitted=maturation.admitted_direction is not None,
            route=maturation.route,
            routed_direction=maturation.admitted_direction,
            event_morphology=event,
            velocity_jerk=velocity,
            completed_bar=completed,
            cross_scale_atlas=atlas,
            news_source=news_source,
            source={
                "kind": "live_v18_plus_immutable_turn_tape",
                "selection_id": selection_id,
                "source_checkpoint_id": candidate["source_checkpoint_id"],
                "source_event_id": candidate["event_id"],
                "raw_decision": raw.as_payload(),
                "maturation_decision": maturation.as_payload(),
                "event_window": event_evidence,
                "finalized_snapshot": snapshot,
                "historical_context": manifest,
                "news_history_sha256": news_fingerprint,
            },
        )
        appended += _append_treatment(ledger, treatment, recorded_at=now)
    treatments = mcl_predictive_treatments(tuple(ledger.records()))
    return {
        "appended": appended,
        "pending_candidates": len(ready) - len(complete),
        "cohort": _cohort(treatments, gate),
    }


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", type=Path, default=MCL_PREDICTIVE_GENERATION_PATH
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(os.environ.get("MCL_PREDICTIVE_LEDGER", MCL_PREDICTIVE_LEDGER_PATH)),
    )
    parser.add_argument("--live-ledger", type=Path, default=MCL_LIVE_LEDGER_PATH)
    parser.add_argument(
        "--capital-plan", type=Path, default=Path("db/calibration/live_capital_plan.json")
    )
    parser.add_argument("--manifest", type=Path, default=MCL_PREDICTIVE_MANIFEST_PATH)
    parser.add_argument(
        "--turn-tape-dir",
        type=Path,
        default=Path(os.environ.get("MCL_TURN_TAPE_DIR", MCL_TURN_TAPE_STATE_DIR)),
    )
    parser.add_argument(
        "--news-history-dir",
        type=Path,
        default=Path(os.environ.get("MCL_NEWS_HISTORY_DIR", MCL_PREDICTIVE_NEWS_DIR)),
    )
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args(argv)
    generation = load_mcl_predictive_generation(args.generation)
    result = await advance_mcl_predictive_accumulator(
        ledger=LiveCalibrationLedger(args.ledger.expanduser()),
        live_ledger=LiveCalibrationLedger(args.live_ledger.expanduser()),
        generation=generation,
        capital_plan_path=args.capital_plan.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        turn_tape_dir=args.turn_tape_dir.expanduser().resolve(),
        news_history_dir=args.news_history_dir.expanduser().resolve(),
        observed_at=datetime.now(timezone.utc),
        seed_only=args.seed_only,
    )
    print(
        json.dumps(
            {
                "schema": MCL_PREDICTIVE_ACCUMULATOR_VERSION,
                "authority": MCL_PREDICTIVE_ACCUMULATOR_AUTHORITY,
                **result,
                "outcomes_exposed": False,
                "submitted_orders": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return int(asyncio.run(_main_async(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
