"""Accumulate prospective MCL shock episodes without trading authority."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ..chart_data.series import OhlcvBar
from ..client import IBKRClient
from ..config import load_config
from ..live.capital import load_live_capital_plan
from ..live.capital_packages import load_allocated_live_selection
from ..news.contract import (
    load_news_history,
)
from ..time_utils import ET_ZONE
from .live_calibration import LiveCalibrationLedger
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    _bar_map,
    load_mcl_live_selection_from_mapping,
    mcl_live_contracts,
)
from .mcl_minute_shock import MclMinuteShockEngine, MclShockMinute
from .mcl_predictive_generation import (
    MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
    load_mcl_predictive_generation,
)
from .mcl_shock_evidence import (
    audit_mcl_shock_volume_clock,
    build_mcl_shock_observations,
    project_mcl_shock_bar_prefix,
    project_mcl_shock_cross_scale,
    project_mcl_shock_news,
    project_mcl_shock_transition,
)
from .mcl_shock_arbiter import (
    MCL_TWO_SPEED_SHOCK_VERSION,
    mcl_weekly_flat_blocked_at_open,
)
from .mcl_shock_crest import (
    MCL_SHOCK_CREST_VERSION,
    MclShockCrestEngine,
    MclShockDecision,
    MclShockObservation,
)
from .mcl_turn_tape import MCL_TURN_TAPE_SCHEMA, MCL_TURN_TAPE_STATE_DIR
from .mcl_two_speed_auction import MclAuctionMinute, MclTwoSpeedAuctionLifecycle


MCL_SHOCK_ACCUMULATOR_VERSION = "mcl.shock-crest-accumulator.v1"
MCL_SHOCK_EPISODE_SCHEMA = "mcl.shock-crest-prospective-episode.v1"
MCL_SHOCK_GENERATION_SCHEMA = "mcl.shock-crest-accumulator-generation.v1"
MCL_SHOCK_ACCUMULATOR_AUTHORITY = (
    "prospective_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_SHOCK_GENERATION_PATH = Path(
    "backtests/mcl/mcl_shock_crest_stage113_preregistration.json"
)
MCL_SHOCK_LEDGER_PATH = (
    Path.home() / ".local/state/tradebot/research/mcl_shock_crest.jsonl"
)
MCL_SHOCK_NEWS_DIR = Path.home() / ".local/state/tradebot/news/history"
_ROOT = Path(__file__).resolve().parents[2]
_LEVEL_RANK = {
    "NORMAL_UNDER_5X": 0,
    "ELEVATED_5_TO_10X": 1,
    "MAJOR_PROTECT_10_TO_12X": 2,
    "TRADEABLE_SHOCK_12_TO_20X": 3,
    "REGIME_20X_PLUS": 4,
}


def _utc(value: object) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None:
        raise ValueError("MCL shock timestamps must be timezone-aware")
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


def _session(value: datetime) -> date | None:
    local = _utc(value).astimezone(ET_ZONE)
    clock = local.time().replace(tzinfo=None)
    if time(17) <= clock < time(18):
        return None
    return (local + timedelta(days=1)).date() if clock >= time(18) else local.date()


def _session_start(value: datetime) -> datetime:
    local = _utc(value).astimezone(ET_ZONE)
    day = local.date() if local.time().replace(tzinfo=None) >= time(18) else (
        local.date() - timedelta(days=1)
    )
    return datetime.combine(day, time(18), tzinfo=ET_ZONE).astimezone(timezone.utc)


def validate_mcl_shock_generation(
    value: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    generation = dict(value)
    body = dict(generation)
    generation_id = str(body.pop("generation_id", ""))
    artifacts = generation.get("artifacts")
    gate = generation.get("cohort_gate")
    levels = generation.get("frozen_levels")
    if (
        generation.get("schema") != MCL_SHOCK_GENERATION_SCHEMA
        or generation.get("authority") != MCL_SHOCK_ACCUMULATOR_AUTHORITY
        or generation.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or generation.get("seconds_owner_version") != MCL_SHOCK_CREST_VERSION
        or not _is_sha(generation_id)
        or _identity(body) != generation_id
        or not _is_sha(generation.get("selection_id"))
        or not _is_sha(generation.get("predictive_generation_id"))
        or not _is_sha(generation.get("turn_tape_generation_sha256"))
        or generation.get("outcomes_exposed") is not False
        or generation.get("submitted_orders") != 0
        or not isinstance(artifacts, Mapping)
        or not isinstance(gate, Mapping)
        or set(gate)
        != {
            "complete_episodes",
            "each_resolved_direction",
            "tradeable_episodes",
            "regime_episodes",
            "causal_crests",
            "continuations",
            "each_continuation_direction",
        }
        or not isinstance(levels, Mapping)
        or dict(levels)
        != {"attention": 5.0, "major": 10.0, "tradeable": 12.0, "regime": 20.0}
    ):
        raise ValueError("MCL shock accumulator generation is invalid")
    _utc(generation["registered_at_utc"])
    eligible = _utc(generation["eligible_start_utc"])
    if eligible < _utc(generation["registered_at_utc"]):
        raise ValueError("MCL shock eligibility predates registration")
    root = repository_root.resolve()
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or not _is_sha(item.get("sha256")):
            raise ValueError(f"MCL shock artifact {name} is invalid")
        relative = Path(str(item.get("path") or ""))
        artifact = (root / relative).resolve()
        if relative.is_absolute() or root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"MCL shock artifact {name} escaped or is missing")
        if _sha256(artifact) != item["sha256"]:
            raise ValueError(f"MCL shock artifact {name} drifted")
    return generation


def load_mcl_shock_generation(
    path: Path = MCL_SHOCK_GENERATION_PATH,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL shock generation must be one object")
    return validate_mcl_shock_generation(
        value,
        repository_root=repository_root or _ROOT,
    )


def _verify_tape_record(
    value: Mapping[str, object], *, generation_sha256: str
) -> dict[str, object]:
    row = dict(value)
    record_id = str(row.pop("record_id", ""))
    if (
        value.get("schema") != MCL_TURN_TAPE_SCHEMA
        or value.get("valid_evidence") is not True
        or value.get("generation_sha256") != generation_sha256
        or value.get("submitted_orders") != 0
        or not _is_sha(record_id)
        or hashlib.sha256(_canonical(row)).hexdigest() != record_id
    ):
        raise ValueError("MCL shock tape record is invalid or drifted")
    observed = _utc(value["bucket_start_utc"])
    recorded = _utc(value["recorded_at_utc"])
    if recorded < observed:
        raise ValueError("MCL shock tape record predates its bucket")
    return {**value, "record_id": record_id, "_time": observed, "_recorded": recorded}


def load_mcl_shock_tape(
    directory: Path,
    *,
    start: datetime,
    end: datetime,
    generation_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    start_utc = _utc(start)
    end_utc = _utc(end)
    if end_utc < start_utc:
        raise ValueError("MCL shock tape interval is reversed")
    days = []
    cursor = start_utc.date()
    while cursor <= end_utc.date():
        days.append(cursor)
        cursor += timedelta(days=1)
    rows = []
    files = []
    for day in days:
        path = directory / f"{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                raw_file = handle.read()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        files.append(
            {"path": path.name, "sha256": hashlib.sha256(raw_file).hexdigest()}
        )
        for line_no, line in enumerate(raw_file.splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid MCL shock JSON") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_no}: MCL shock row is not an object")
            observed = _utc(raw.get("bucket_start_utc"))
            if start_utc <= observed <= end_utc:
                rows.append(
                    _verify_tape_record(raw, generation_sha256=generation_sha256)
                )
    rows.sort(key=lambda row: row["_time"])
    times = [row["_time"] for row in rows]
    if len(times) != len(set(times)):
        raise ValueError("MCL shock tape repeats a bucket")
    return rows, {
        "files": files,
        "records": len(rows),
        "first_bucket_utc": times[0].isoformat() if times else None,
        "last_bucket_utc": times[-1].isoformat() if times else None,
        "record_ids_sha256": _identity([row["record_id"] for row in rows]),
    }


def _minute_resets(
    bars: Mapping[str, Mapping[datetime, OhlcvBar]], *, contract_key: str
) -> list[dict[str, object]]:
    engine = MclMinuteShockEngine()
    v18 = MclTwoSpeedAuctionLifecycle()
    resets: defaultdict[datetime, set[str]] = defaultdict(set)
    common = sorted(set(bars["CL"]) & set(bars["MCL"]))
    maintenance = set()
    if common:
        cursor = common[0].astimezone(ET_ZONE).date()
        last = common[-1].astimezone(ET_ZONE).date()
        while cursor <= last:
            boundary = datetime.combine(cursor, time(17), tzinfo=ET_ZONE).astimezone(
                timezone.utc
            )
            if common[0] <= boundary <= common[-1]:
                maintenance.add(boundary)
                resets[boundary].add("stage112_maintenance")
            cursor += timedelta(days=1)
    previous: datetime | None = None
    for stamp in common:
        auction_minute = MclAuctionMinute(
            contract_key,
            bars["CL"][stamp],
            bars["MCL"][stamp],
        )
        transition = engine.update(
            MclShockMinute(
                contract_key,
                bars["CL"][stamp],
                bars["MCL"][stamp],
            )
        )
        v18_transition = v18.update(auction_minute)
        if (
            v18_transition.decision is not None
            and v18_transition.decision.phase == "RAW_TURN"
        ):
            resets[stamp].add("stage112_v18_raw_turn")
        if transition.exit_reason is not None:
            resets[stamp].add(
                f"stage112_minute_release:{transition.exit_reason}"
            )
        if transition.contract_reset:
            resets[stamp].add("contract_roll")
        if transition.gap_reset:
            assert previous is not None
            scheduled = any(previous <= boundary < stamp for boundary in maintenance)
            if not scheduled:
                resets[previous + timedelta(minutes=1)].add("unscheduled_minute_gap")
        if mcl_weekly_flat_blocked_at_open(stamp):
            resets[stamp].add("stage112_friday_flat")
        previous = stamp
    return [
        {"at_utc": stamp, "reasons": sorted(reasons)}
        for stamp, reasons in sorted(resets.items())
    ]


def _new_episode(
    row: Mapping[str, object],
    observation: MclShockObservation,
    decision: MclShockDecision,
) -> dict[str, object]:
    return {
        "started_at_utc": observation.observed_at_utc,
        "contract_key": observation.contract_key,
        "first_record_id": row["record_id"],
        "first_recorded_at_utc": row["_recorded"],
        "last_recorded_at_utc": row["_recorded"],
        "source_record_ids": [],
        "maximum_volume_multiple": float(observation.volume_multiple),
        "maximum_level": decision.latched_level,
        "shock_direction": decision.shock_direction,
        "eligible": _LEVEL_RANK[decision.latched_level] >= 2,
        "transitions": [],
    }


def _advance_episode(
    episode: dict[str, object],
    row: Mapping[str, object],
    observation: MclShockObservation,
    decision: MclShockDecision,
    *,
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
) -> None:
    episode["source_record_ids"].append(str(row["record_id"]))
    episode["last_recorded_at_utc"] = row["_recorded"]
    episode["maximum_volume_multiple"] = max(
        float(episode["maximum_volume_multiple"]),
        float(observation.volume_multiple),
    )
    if _LEVEL_RANK[decision.latched_level] > _LEVEL_RANK[str(episode["maximum_level"])]:
        episode["maximum_level"] = decision.latched_level
    episode["eligible"] = bool(episode["eligible"]) or (
        _LEVEL_RANK[decision.latched_level] >= 2
    )
    if episode["shock_direction"] is None and decision.shock_direction in (-1, 1):
        episode["shock_direction"] = decision.shock_direction
    if decision.phase != "STATE":
        episode["transitions"].append(
            project_mcl_shock_transition(
                decision, observation, bars=bars, news=news
            )
        )


def _finalize_episode(
    episode: Mapping[str, object],
    *,
    terminal_at: datetime,
    terminal_reasons: Sequence[str],
    generation: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not episode.get("eligible"):
        return None
    start = _utc(episode["started_at_utc"])
    terminal = _utc(terminal_at)
    direction = episode.get("shock_direction")
    identity = {
        "generation_id": generation["generation_id"],
        "selection_id": generation["selection_id"],
        "contract_key": episode["contract_key"],
        "started_at_utc": start.isoformat(),
        "first_record_id": episode["first_record_id"],
    }
    source_ids = list(episode["source_record_ids"])
    transitions = list(episode["transitions"])
    body = {
        "schema": MCL_SHOCK_EPISODE_SCHEMA,
        "authority": MCL_SHOCK_ACCUMULATOR_AUTHORITY,
        "episode_id": _identity(identity),
        "identity": identity,
        "started_at_utc": start.isoformat(),
        "terminal_at_utc": terminal.isoformat(),
        "duration_seconds": (terminal - start).total_seconds(),
        "contract_key": episode["contract_key"],
        "shock_direction": direction,
        "maximum_level": episode["maximum_level"],
        "maximum_volume_multiple": float(episode["maximum_volume_multiple"]),
        "reached_tradeable_12x": _LEVEL_RANK[str(episode["maximum_level"])] >= 3,
        "reached_regime_20x": _LEVEL_RANK[str(episode["maximum_level"])] >= 4,
        "transitions": transitions,
        "source": {
            "records": len(source_ids),
            "first_record_id": source_ids[0] if source_ids else None,
            "last_record_id": source_ids[-1] if source_ids else None,
            "record_ids_sha256": _identity(source_ids),
            "first_recorded_at_utc": _utc(
                episode["first_recorded_at_utc"]
            ).isoformat(),
            "last_recorded_at_utc": _utc(
                episode["last_recorded_at_utc"]
            ).isoformat(),
            "timestamp_semantics": "IB_TCP_packet_receipt_utc_not_exchange_time",
        },
        "bar_prefix": project_mcl_shock_bar_prefix(
            bars,
            start=start - timedelta(days=7),
            end=terminal,
        ),
        "volume_clock_audit": audit_mcl_shock_volume_clock(
            rows,
            bars,
            start=start,
            end=terminal,
        ),
        "terminal": {
            "reasons": list(terminal_reasons),
            "cross_scale": project_mcl_shock_cross_scale(
                bars,
                when=terminal,
                direction=int(direction) if direction in (-1, 1) else None,
            ),
            "news": project_mcl_shock_news(news, when=terminal),
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
    return {**body, "episode_sha256": _identity(body)}


def replay_mcl_shock_episodes(
    observations: Sequence[tuple[Mapping[str, object], MclShockObservation]],
    *,
    resets: Sequence[Mapping[str, object]],
    generation: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    engine = MclShockCrestEngine()
    ordered_resets = sorted(resets, key=lambda row: _utc(row["at_utc"]))
    reset_index = 0
    episode: dict[str, object] | None = None
    complete = []
    for raw_row, observation in observations:
        row = dict(raw_row)
        at = _utc(observation.observed_at_utc)
        while (
            reset_index < len(ordered_resets)
            and _utc(ordered_resets[reset_index]["at_utc"]) <= at
        ):
            reset = ordered_resets[reset_index]
            if episode is not None:
                closed = _finalize_episode(
                    episode,
                    terminal_at=_utc(reset["at_utc"]),
                    terminal_reasons=list(reset["reasons"]),
                    generation=generation,
                    rows=rows,
                    bars=bars,
                    news=news,
                )
                if closed is not None:
                    complete.append(closed)
            engine.reset(contract_key=observation.contract_key)
            episode = None
            reset_index += 1
        decision = engine.update(observation)
        if episode is None and decision.latched_level != "NORMAL_UNDER_5X":
            episode = _new_episode(row, observation, decision)
        if episode is None:
            continue
        _advance_episode(
            episode,
            row,
            observation,
            decision,
            bars=bars,
            news=news,
        )
        if decision.phase == "NORMALIZED":
            closed = _finalize_episode(
                episode,
                terminal_at=observation.observed_at_utc,
                terminal_reasons=["seconds_owner_normalized"],
                generation=generation,
                rows=rows,
                bars=bars,
                news=news,
            )
            if closed is not None:
                complete.append(closed)
            episode = None
    open_episode = None
    if episode is not None:
        open_episode = {
            "started_at_utc": _utc(episode["started_at_utc"]).isoformat(),
            "contract_key": episode["contract_key"],
            "eligible": bool(episode["eligible"]),
            "shock_direction": episode["shock_direction"],
            "maximum_level": episode["maximum_level"],
            "maximum_volume_multiple": float(episode["maximum_volume_multiple"]),
            "source_records": len(episode["source_record_ids"]),
        }
    return complete, open_episode


def mcl_shock_episodes(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    episodes = []
    identities = set()
    for row in records:
        if row.get("kind") != "checkpoint":
            raise ValueError("MCL shock ledger contains a non-checkpoint record")
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("MCL shock ledger checkpoint has no evidence")
        value = dict(evidence)
        episode_sha = str(value.pop("episode_sha256", ""))
        identity = value.get("identity")
        episode_id = str(value.get("episode_id") or "")
        if (
            value.get("schema") != MCL_SHOCK_EPISODE_SCHEMA
            or value.get("authority") != MCL_SHOCK_ACCUMULATOR_AUTHORITY
            or value.get("outcomes_exposed") is not False
            or value.get("submitted_orders") != 0
            or not isinstance(identity, Mapping)
            or _identity(identity) != episode_id
            or _identity(value) != episode_sha
            or episode_id in identities
        ):
            raise ValueError("MCL shock episode identity drifted")
        identities.add(episode_id)
        episodes.append(dict(evidence))
    return episodes


def mcl_shock_cohort(
    episodes: Sequence[Mapping[str, object]], gate: Mapping[str, object]
) -> dict[str, object]:
    directions = Counter(
        "up" if row.get("shock_direction") == 1 else "down"
        for row in episodes
        if row.get("shock_direction") in (-1, 1)
    )
    tradeable = sum(bool(row.get("reached_tradeable_12x")) for row in episodes)
    regime = sum(bool(row.get("reached_regime_20x")) for row in episodes)
    crests = 0
    continuation_directions: Counter[str] = Counter()
    for row in episodes:
        phases = [
            transition.get("decision", {}).get("phase")
            for transition in row.get("transitions", [])
            if isinstance(transition, Mapping)
        ]
        crests += "CREST_CONFIRMED" in phases
        for transition in row.get("transitions", []):
            decision = transition.get("decision") if isinstance(transition, Mapping) else None
            direction = (
                decision.get("continuation_direction")
                if isinstance(decision, Mapping)
                else None
            )
            if direction in (-1, 1):
                continuation_directions["up" if direction == 1 else "down"] += 1
    continuations = sum(continuation_directions.values())
    gates = {
        "at_least_complete_episodes": len(episodes) >= int(gate["complete_episodes"]),
        "at_least_each_resolved_direction": all(
            directions[value] >= int(gate["each_resolved_direction"])
            for value in ("up", "down")
        ),
        "at_least_tradeable_episodes": tradeable >= int(gate["tradeable_episodes"]),
        "at_least_regime_episodes": regime >= int(gate["regime_episodes"]),
        "at_least_causal_crests": crests >= int(gate["causal_crests"]),
        "at_least_continuations": continuations >= int(gate["continuations"]),
        "at_least_each_continuation_direction": all(
            continuation_directions[value]
            >= int(gate["each_continuation_direction"])
            for value in ("up", "down")
        ),
    }
    return {
        "complete_episodes": len(episodes),
        "resolved_directions": dict(sorted(directions.items())),
        "tradeable_12x_episodes": tradeable,
        "regime_20x_episodes": regime,
        "causal_crests": crests,
        "continuations": continuations,
        "continuation_directions": dict(sorted(continuation_directions.items())),
        "gates": gates,
        "verdict": (
            "COHORT_READY_FOR_PREREGISTERED_MATCHED_CONTROLS"
            if all(gates.values())
            else "ACCUMULATE"
        ),
    }


def _validate_tape_contracts(
    rows: Sequence[Mapping[str, object]], selection: Mapping[str, object]
) -> None:
    expected = {
        symbol: int(selection["contracts"][symbol]["con_id"])
        for symbol in ("CL", "MCL")
    }
    for row in rows:
        books = row.get("books")
        if not isinstance(books, Mapping):
            raise ValueError("MCL shock tape lacks contract books")
        observed = {}
        for symbol in ("CL", "MCL"):
            book = books.get(symbol)
            contract = book.get("contract") if isinstance(book, Mapping) else None
            if not isinstance(contract, Mapping):
                raise ValueError("MCL shock tape lacks contract identity")
            observed[symbol] = int(contract.get("con_id") or 0)
        if observed != expected:
            raise ValueError("MCL shock tape contract drifted from selection")


def advance_mcl_shock_accumulator(
    *,
    ledger: LiveCalibrationLedger,
    generation: Mapping[str, object],
    selection: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
    observed_at: datetime,
) -> dict[str, object]:
    now = _utc(observed_at)
    if (
        selection.get("selection_id") != generation.get("selection_id")
        or selection.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
    ):
        raise ValueError("MCL shock generation selection drifted")
    _validate_tape_contracts(rows, selection)
    contract_key = str(selection["contracts"]["MCL"]["expiry"])[:6]
    eligible = _utc(generation["eligible_start_utc"])
    observations = build_mcl_shock_observations(
        rows,
        bars,
        contract_key=contract_key,
        eligible_start=eligible,
    )
    resets = [
        row
        for row in _minute_resets(bars, contract_key=contract_key)
        if _utc(row["at_utc"]) >= eligible
    ]
    complete, open_episode = replay_mcl_shock_episodes(
        observations,
        resets=resets,
        generation=generation,
        rows=rows,
        bars=bars,
        news=news,
    )
    prior = {
        str(row["episode_id"]): row
        for row in mcl_shock_episodes(tuple(ledger.records()))
    }
    appended = 0
    for episode in complete:
        episode_id = str(episode["episode_id"])
        existing = prior.get(episode_id)
        if existing is not None:
            if existing.get("episode_sha256") != episode.get("episode_sha256"):
                raise ValueError("MCL shock episode changed across replay")
            continue
        terminal = _utc(episode["terminal_at_utc"])
        ledger.checkpoint(
            evaluation_as_of=terminal,
            strategy_id=MCL_SHOCK_CREST_VERSION,
            strategy_version=MCL_SHOCK_ACCUMULATOR_VERSION,
            trading_date=(
                value.isoformat()
                if (value := _session(_utc(episode["started_at_utc"]))) is not None
                else None
            ),
            session="MCL_SHOCK",
            status="EVALUATED",
            evidence=episode,
            recorded_at=now,
        )
        prior[episode_id] = episode
        appended += 1
    episodes = mcl_shock_episodes(tuple(ledger.records()))
    return {
        "appended": appended,
        "eligible_start_utc": eligible.isoformat(),
        "source_records": len(rows),
        "eligible_observations": len(observations),
        "maximum_volume_multiple": max(
            (observation.volume_multiple for _row, observation in observations),
            default=0.0,
        ),
        "complete_episodes_in_prefix": len(complete),
        "open_episode": open_episode,
        "cohort": mcl_shock_cohort(episodes, generation["cohort_gate"]),
    }


async def _recent_bars(
    client: IBKRClient,
    *,
    selection: Mapping[str, object],
    observed_at: datetime,
) -> tuple[dict[str, dict[datetime, OhlcvBar]], dict[str, object]]:
    contracts = mcl_live_contracts(selection)
    raw = await asyncio.gather(
        *(
            client.historical_bars_ohlcv(
                contract,
                duration_str="1 W",
                bar_size="1 min",
                use_rth=False,
                what_to_show="TRADES",
                cache_ttl_sec=0,
            )
            for contract in contracts
        )
    )
    now = _utc(observed_at)
    cutoff = now.replace(second=0, microsecond=0)
    maps = {
        symbol: _bar_map(rows, cutoff=cutoff, name=f"shock {symbol}")
        for symbol, rows in zip(("CL", "MCL"), raw, strict=True)
    }
    common = sorted(set(maps["CL"]) & set(maps["MCL"]))
    latest = common[-1] if common else None
    age = (now - latest).total_seconds() if latest is not None else None
    if len(common) < 750 or age is None or not 0.0 <= age <= 8 * 60:
        raise ValueError(
            "MCL shock finalized bars are incomplete or stale: "
            f"common={len(common)} latest={latest!s} age_seconds={age!s}"
        )
    return maps, {
        "cutoff_utc": cutoff.isoformat(),
        "common_rows": len(common),
        "first_common_close_utc": common[0].isoformat(),
        "last_common_close_utc": common[-1].isoformat(),
        "sha256": _identity(
            [
                [
                    stamp.isoformat(),
                    *[
                        [
                            float(maps[symbol][stamp].open),
                            float(maps[symbol][stamp].high),
                            float(maps[symbol][stamp].low),
                            float(maps[symbol][stamp].close),
                            float(maps[symbol][stamp].volume),
                        ]
                        for symbol in ("CL", "MCL")
                    ],
                ]
                for stamp in common
            ]
        ),
    }


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=MCL_SHOCK_GENERATION_PATH)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(os.environ.get("MCL_SHOCK_LEDGER", MCL_SHOCK_LEDGER_PATH)),
    )
    parser.add_argument(
        "--capital-plan",
        type=Path,
        default=Path("db/calibration/live_capital_plan.json"),
    )
    parser.add_argument(
        "--predictive-generation",
        type=Path,
        default=MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
    )
    parser.add_argument(
        "--turn-tape-dir",
        type=Path,
        default=Path(os.environ.get("MCL_TURN_TAPE_DIR", MCL_TURN_TAPE_STATE_DIR)),
    )
    parser.add_argument(
        "--news-history-dir",
        type=Path,
        default=Path(os.environ.get("MCL_NEWS_HISTORY_DIR", MCL_SHOCK_NEWS_DIR)),
    )
    args = parser.parse_args(argv)
    generation = load_mcl_shock_generation(args.generation)
    plan = load_live_capital_plan(args.capital_plan.expanduser().resolve())
    selection_value, _selection_path, _selection_sha = load_allocated_live_selection(
        plan,
        sleeve_id=MCL_LIVE_CAPITAL_SLEEVE,
        repository_root=_ROOT,
    )
    selection = load_mcl_live_selection_from_mapping(selection_value)
    predictive = load_mcl_predictive_generation(args.predictive_generation)
    if (
        predictive.get("generation_id") != generation["predictive_generation_id"]
        or predictive.get("selection_id") != generation["selection_id"]
    ):
        raise ValueError("MCL shock predictive generation drifted")
    now = datetime.now(timezone.utc)
    eligible = _utc(generation["eligible_start_utc"])
    start = max(
        eligible - timedelta(seconds=60),
        _session_start(now) - timedelta(days=1, seconds=60),
    )
    rows, tape = load_mcl_shock_tape(
        args.turn_tape_dir.expanduser(),
        start=start,
        end=now,
        generation_sha256=str(generation["turn_tape_generation_sha256"]),
    )
    if not rows:
        result = {
            "appended": 0,
            "eligible_start_utc": eligible.isoformat(),
            "source_records": 0,
            "eligible_observations": 0,
            "maximum_volume_multiple": 0.0,
            "complete_episodes_in_prefix": 0,
            "open_episode": None,
            "cohort": mcl_shock_cohort(
                mcl_shock_episodes(
                    tuple(LiveCalibrationLedger(args.ledger.expanduser()).records())
                ),
                generation["cohort_gate"],
            ),
        }
        bars_evidence = None
    else:
        config = load_config()
        if not config.readonly:
            raise ValueError("MCL shock accumulator requires IBKR_READONLY=1")
        client = IBKRClient(config)
        await client.connect()
        try:
            bars, bars_evidence = await _recent_bars(
                client,
                selection=selection,
                observed_at=now,
            )
        finally:
            await client.disconnect()
        snapshots = [
            row
            for path in sorted(args.news_history_dir.expanduser().glob("*.jsonl"))
            for row in load_news_history(path)
        ]
        result = advance_mcl_shock_accumulator(
            ledger=LiveCalibrationLedger(args.ledger.expanduser()),
            generation=generation,
            selection=selection,
            rows=rows,
            bars=bars,
            news=snapshots,
            observed_at=now,
        )
    print(
        json.dumps(
            {
                "schema": MCL_SHOCK_ACCUMULATOR_VERSION,
                "authority": MCL_SHOCK_ACCUMULATOR_AUTHORITY,
                "generation_id": generation["generation_id"],
                "selection_id": generation["selection_id"],
                "tape": tape,
                "bars": bars_evidence,
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
