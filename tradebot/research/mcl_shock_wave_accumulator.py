"""Accumulate prospective authority-bound MCL shock waves without trading."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..chart_data.series import OhlcvBar
from ..client import IBKRClient
from ..config import load_config
from ..live.capital import load_live_capital_plan
from ..live.capital_packages import load_allocated_live_selection
from ..news.contract import load_news_history
from .live_calibration import LiveCalibrationLedger
from .mcl_live_transport import (
    MCL_LIVE_CAPITAL_SLEEVE,
    load_mcl_live_selection_from_mapping,
)
from .mcl_predictive_generation import (
    MCL_PREDICTIVE_RUNTIME_GENERATION_PATH,
    load_mcl_predictive_generation,
)
from .mcl_shock_accumulator import (
    MCL_SHOCK_NEWS_DIR,
    _identity,
    _is_sha,
    _minute_resets,
    _recent_bars,
    _session,
    _session_start,
    _sha256,
    _utc,
    _validate_tape_contracts,
    load_mcl_shock_tape,
)
from .mcl_shock_arbiter import MCL_TWO_SPEED_SHOCK_VERSION
from .mcl_shock_crest import (
    MCL_SHOCK_CREST_VERSION,
    MclShockBookEvidence,
    MclShockObservation,
)
from .mcl_shock_evidence import (
    audit_mcl_shock_volume_clock,
    build_mcl_shock_observations,
    project_mcl_shock_bar_prefix,
    project_mcl_shock_cross_scale,
    project_mcl_shock_news,
)
from .mcl_shock_waves import (
    MCL_SHOCK_WAVE_VERSION,
    MclAuthorityBoundShockWaveEngine,
    MclShockWaveDecision,
)
from .mcl_turn_tape import MCL_TURN_TAPE_STATE_DIR


MCL_SHOCK_WAVE_ACCUMULATOR_VERSION = "mcl.shock-wave-accumulator.v1"
MCL_SHOCK_WAVE_EPISODE_SCHEMA = "mcl.authority-bound-shock-wave-episode.v1"
MCL_SHOCK_WAVE_GENERATION_SCHEMA = "mcl.authority-bound-shock-waves-generation.v1"
MCL_SHOCK_WAVE_AUTHORITY = (
    "prospective_morphology_only_no_outcomes_no_orders_no_capital"
)
MCL_SHOCK_WAVE_GENERATION_PATH = Path(
    "backtests/mcl/mcl_authority_bound_shock_waves_stage114_preregistration.json"
)
MCL_SHOCK_WAVE_LEDGER_PATH = (
    Path.home() / ".local/state/tradebot/research/mcl_shock_waves.jsonl"
)
_ROOT = Path(__file__).resolve().parents[2]
_LEVEL_RANK = {
    "NORMAL_UNDER_5X": 0,
    "ELEVATED_5_TO_10X": 1,
    "MAJOR_PROTECT_10_TO_12X": 2,
    "TRADEABLE_SHOCK_12_TO_20X": 3,
    "REGIME_20X_PLUS": 4,
}
_GATE_KEYS = {
    "complete_episodes",
    "authority_bound_episodes",
    "each_authority_direction",
    "tradeable_episodes",
    "regime_episodes",
    "causal_crests",
    "continuations",
    "each_continuation_direction",
    "authority_handoffs",
    "each_handoff_direction",
}
_STATE_LAW = {
    "attention_directionless_below_multiple": 10.0,
    "initial_authority_current_multiple": 10.0,
    "binding_requires": [
        "fresh_type_1_top",
        "spread_eligible",
        "CL_and_MCL_5s_15s_60s_velocity",
        "CL_and_MCL_15m_velocity_and_acceleration",
        "CL_and_MCL_signed_15s_flow",
        "all_nonzero_and_same_sign",
    ],
    "handoff_requires_strictly_higher_current_level": True,
    "handoff_resets_inner_crest_state": True,
    "ordinary_rotation_may_invent_direction": False,
    "inner_crest_version": MCL_SHOCK_CREST_VERSION,
}


def validate_mcl_shock_wave_generation(
    value: Mapping[str, object], *, repository_root: Path
) -> dict[str, object]:
    generation = dict(value)
    body = dict(generation)
    generation_id = str(body.pop("generation_id", ""))
    artifacts = generation.get("artifacts")
    gate = generation.get("cohort_gate")
    if (
        generation.get("schema") != MCL_SHOCK_WAVE_GENERATION_SCHEMA
        or generation.get("authority") != MCL_SHOCK_WAVE_AUTHORITY
        or generation.get("strategy_version") != MCL_TWO_SPEED_SHOCK_VERSION
        or generation.get("seconds_owner_version") != MCL_SHOCK_WAVE_VERSION
        or not _is_sha(generation_id)
        or _identity(body) != generation_id
        or not _is_sha(generation.get("selection_id"))
        or not _is_sha(generation.get("predictive_generation_id"))
        or not _is_sha(generation.get("turn_tape_generation_sha256"))
        or generation.get("state_law") != _STATE_LAW
        or generation.get("frozen_levels")
        != {"attention": 5.0, "major": 10.0, "tradeable": 12.0, "regime": 20.0}
        or not isinstance(gate, Mapping)
        or set(gate) != _GATE_KEYS
        or not isinstance(artifacts, Mapping)
        or generation.get("outcomes_exposed") is not False
        or generation.get("submitted_orders") != 0
    ):
        raise ValueError("MCL shock-wave generation is invalid")
    registered = _utc(generation["registered_at_utc"])
    if _utc(generation["eligible_start_utc"]) < registered:
        raise ValueError("MCL shock-wave eligibility predates registration")
    root = repository_root.resolve()
    for name, item in artifacts.items():
        if not isinstance(item, Mapping) or not _is_sha(item.get("sha256")):
            raise ValueError(f"MCL shock-wave artifact {name} is invalid")
        relative = Path(str(item.get("path") or ""))
        artifact = (root / relative).resolve()
        if relative.is_absolute() or root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"MCL shock-wave artifact {name} escaped or is missing")
        if _sha256(artifact) != item["sha256"]:
            raise ValueError(f"MCL shock-wave artifact {name} drifted")
    return generation


def load_mcl_shock_wave_generation(
    path: Path = MCL_SHOCK_WAVE_GENERATION_PATH,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError("MCL shock-wave generation must be one object")
    return validate_mcl_shock_wave_generation(
        value, repository_root=repository_root or _ROOT
    )


def _book_payload(value: MclShockBookEvidence) -> dict[str, float]:
    return {name: float(getattr(value, name)) for name in value.__dataclass_fields__}


def _observation_payload(value: MclShockObservation) -> dict[str, object]:
    return {
        "observed_at_utc": _utc(value.observed_at_utc).isoformat(),
        "contract_key": value.contract_key,
        "mcl_microprice": float(value.mcl_microprice),
        "volume_multiple": float(value.volume_multiple),
        "cl": _book_payload(value.cl),
        "mcl": _book_payload(value.mcl),
        "spread_eligible": bool(value.spread_eligible),
        "fresh_top": bool(value.fresh_top),
    }


def _transition(
    decision: MclShockWaveDecision,
    observation: MclShockObservation,
    *,
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "decision": decision.as_payload(),
        "observation": _observation_payload(observation),
        "cross_scale": project_mcl_shock_cross_scale(
            bars, when=observation.observed_at_utc, direction=decision.authority_direction
        ),
        "news": project_mcl_shock_news(news, when=observation.observed_at_utc),
    }


def _new_episode(
    row: Mapping[str, object],
    observation: MclShockObservation,
) -> dict[str, object]:
    return {
        "started_at_utc": observation.observed_at_utc,
        "contract_key": observation.contract_key,
        "first_record_id": row["record_id"],
        "first_recorded_at_utc": row["_recorded"],
        "last_recorded_at_utc": row["_recorded"],
        "source_record_ids": [],
        "maximum_volume_multiple": float(observation.volume_multiple),
        "maximum_level": "ELEVATED_5_TO_10X",
        "eligible": False,
        "authority_waves": [],
        "transitions": [],
    }


def _advance_episode(
    episode: dict[str, object],
    row: Mapping[str, object],
    observation: MclShockObservation,
    decision: MclShockWaveDecision,
    *,
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]],
) -> None:
    episode["source_record_ids"].append(str(row["record_id"]))
    episode["last_recorded_at_utc"] = row["_recorded"]
    episode["maximum_volume_multiple"] = max(
        float(episode["maximum_volume_multiple"]), float(observation.volume_multiple)
    )
    if _LEVEL_RANK[decision.maximum_level] > _LEVEL_RANK[str(episode["maximum_level"])]:
        episode["maximum_level"] = decision.maximum_level
    episode["eligible"] = bool(episode["eligible"]) or (
        _LEVEL_RANK[decision.maximum_level] >= 2
    )
    if decision.event in {"AUTHORITY_BOUND", "AUTHORITY_HANDOFF"}:
        episode["authority_waves"].append(
            {
                "sequence": decision.wave_sequence,
                "event": decision.event,
                "bound_at_utc": _utc(decision.observed_at_utc).isoformat(),
                "first_record_id": row["record_id"],
                "direction": decision.authority_direction,
                "authority_level": decision.authority_level,
                "handoff_from_direction": decision.handoff_from_direction,
            }
        )
    if decision.event != "STATE":
        episode["transitions"].append(
            _transition(decision, observation, bars=bars, news=news)
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
    waves = list(episode["authority_waves"])
    direction = waves[-1]["direction"] if waves else None
    identity = {
        "generation_id": generation["generation_id"],
        "selection_id": generation["selection_id"],
        "contract_key": episode["contract_key"],
        "started_at_utc": start.isoformat(),
        "first_record_id": episode["first_record_id"],
    }
    source_ids = list(episode["source_record_ids"])
    body = {
        "schema": MCL_SHOCK_WAVE_EPISODE_SCHEMA,
        "authority": MCL_SHOCK_WAVE_AUTHORITY,
        "episode_id": _identity(identity),
        "identity": identity,
        "started_at_utc": start.isoformat(),
        "terminal_at_utc": terminal.isoformat(),
        "duration_seconds": (terminal - start).total_seconds(),
        "contract_key": episode["contract_key"],
        "initial_authority_direction": waves[0]["direction"] if waves else None,
        "terminal_authority_direction": direction,
        "authority_waves": waves,
        "maximum_level": episode["maximum_level"],
        "maximum_volume_multiple": float(episode["maximum_volume_multiple"]),
        "reached_tradeable_12x": _LEVEL_RANK[str(episode["maximum_level"])] >= 3,
        "reached_regime_20x": _LEVEL_RANK[str(episode["maximum_level"])] >= 4,
        "transitions": list(episode["transitions"]),
        "source": {
            "records": len(source_ids),
            "first_record_id": source_ids[0] if source_ids else None,
            "last_record_id": source_ids[-1] if source_ids else None,
            "record_ids_sha256": _identity(source_ids),
            "first_recorded_at_utc": _utc(episode["first_recorded_at_utc"]).isoformat(),
            "last_recorded_at_utc": _utc(episode["last_recorded_at_utc"]).isoformat(),
            "timestamp_semantics": "IB_TCP_packet_receipt_utc_not_exchange_time",
        },
        "bar_prefix": project_mcl_shock_bar_prefix(
            bars, start=start - timedelta(days=7), end=terminal
        ),
        "volume_clock_audit": audit_mcl_shock_volume_clock(
            rows, bars, start=start, end=terminal
        ),
        "terminal": {
            "reasons": list(terminal_reasons),
            "cross_scale": project_mcl_shock_cross_scale(
                bars, when=terminal, direction=direction
            ),
            "news": project_mcl_shock_news(news, when=terminal),
        },
        "outcomes_exposed": False,
        "submitted_orders": 0,
    }
    return {**body, "episode_sha256": _identity(body)}


def replay_mcl_shock_wave_episodes(
    observations: Sequence[tuple[Mapping[str, object], MclShockObservation]],
    *,
    resets: Sequence[Mapping[str, object]],
    generation: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    bars: Mapping[str, Mapping[datetime, OhlcvBar]],
    news: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    engine = MclAuthorityBoundShockWaveEngine()
    ordered_resets = sorted(resets, key=lambda row: _utc(row["at_utc"]))
    reset_index = 0
    episode: dict[str, object] | None = None
    complete = []
    for raw_row, observation in observations:
        row = dict(raw_row)
        at = _utc(observation.observed_at_utc)
        while reset_index < len(ordered_resets) and _utc(
            ordered_resets[reset_index]["at_utc"]
        ) <= at:
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
        if episode is None and decision.episode_active:
            episode = _new_episode(row, observation)
        if episode is None:
            continue
        _advance_episode(
            episode, row, observation, decision, bars=bars, news=news
        )
        if decision.episode_terminal:
            closed = _finalize_episode(
                episode,
                terminal_at=observation.observed_at_utc,
                terminal_reasons=["seconds_wave_owner_normalized"],
                generation=generation,
                rows=rows,
                bars=bars,
                news=news,
            )
            if closed is not None:
                complete.append(closed)
            episode = None
    opened = None
    if episode is not None:
        waves = list(episode["authority_waves"])
        opened = {
            "started_at_utc": _utc(episode["started_at_utc"]).isoformat(),
            "contract_key": episode["contract_key"],
            "eligible": bool(episode["eligible"]),
            "authority_direction": waves[-1]["direction"] if waves else None,
            "authority_waves": len(waves),
            "maximum_level": episode["maximum_level"],
            "maximum_volume_multiple": float(episode["maximum_volume_multiple"]),
            "source_records": len(episode["source_record_ids"]),
        }
    return complete, opened


def mcl_shock_wave_episodes(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    episodes = []
    identities = set()
    for row in records:
        evidence = row.get("evidence")
        if row.get("kind") != "checkpoint" or not isinstance(evidence, Mapping):
            raise ValueError("MCL shock-wave ledger contains an invalid record")
        value = dict(evidence)
        episode_sha = str(value.pop("episode_sha256", ""))
        identity = value.get("identity")
        episode_id = str(value.get("episode_id") or "")
        if (
            value.get("schema") != MCL_SHOCK_WAVE_EPISODE_SCHEMA
            or value.get("authority") != MCL_SHOCK_WAVE_AUTHORITY
            or value.get("outcomes_exposed") is not False
            or value.get("submitted_orders") != 0
            or not isinstance(identity, Mapping)
            or _identity(identity) != episode_id
            or _identity(value) != episode_sha
            or episode_id in identities
        ):
            raise ValueError("MCL shock-wave episode identity drifted")
        identities.add(episode_id)
        episodes.append(dict(evidence))
    return episodes


def mcl_shock_wave_cohort(
    episodes: Sequence[Mapping[str, object]], gate: Mapping[str, object]
) -> dict[str, object]:
    authority_directions: Counter[str] = Counter()
    handoff_directions: Counter[str] = Counter()
    continuation_directions: Counter[str] = Counter()
    bound_episodes = handoffs = crests = 0
    for episode in episodes:
        waves = list(episode.get("authority_waves", []))
        bound_episodes += bool(waves)
        for wave in waves:
            direction = wave.get("direction")
            if direction in (-1, 1):
                label = "up" if direction == 1 else "down"
                authority_directions[label] += 1
                if wave.get("event") == "AUTHORITY_HANDOFF":
                    handoffs += 1
                    handoff_directions[label] += 1
        for transition in episode.get("transitions", []):
            decision = transition.get("decision") if isinstance(transition, Mapping) else None
            crest = decision.get("crest") if isinstance(decision, Mapping) else None
            if not isinstance(crest, Mapping):
                continue
            crests += crest.get("phase") == "CREST_CONFIRMED"
            direction = crest.get("continuation_direction")
            if direction in (-1, 1):
                continuation_directions["up" if direction == 1 else "down"] += 1
    tradeable = sum(bool(row.get("reached_tradeable_12x")) for row in episodes)
    regime = sum(bool(row.get("reached_regime_20x")) for row in episodes)
    continuations = sum(continuation_directions.values())
    gates = {
        "at_least_complete_episodes": len(episodes) >= int(gate["complete_episodes"]),
        "at_least_authority_bound_episodes": bound_episodes
        >= int(gate["authority_bound_episodes"]),
        "at_least_each_authority_direction": all(
            authority_directions[value] >= int(gate["each_authority_direction"])
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
        "at_least_authority_handoffs": handoffs >= int(gate["authority_handoffs"]),
        "at_least_each_handoff_direction": all(
            handoff_directions[value] >= int(gate["each_handoff_direction"])
            for value in ("up", "down")
        ),
    }
    return {
        "complete_episodes": len(episodes),
        "authority_bound_episodes": bound_episodes,
        "authority_directions": dict(sorted(authority_directions.items())),
        "tradeable_12x_episodes": tradeable,
        "regime_20x_episodes": regime,
        "causal_crests": crests,
        "continuations": continuations,
        "continuation_directions": dict(sorted(continuation_directions.items())),
        "authority_handoffs": handoffs,
        "handoff_directions": dict(sorted(handoff_directions.items())),
        "gates": gates,
        "verdict": (
            "COHORT_READY_FOR_PREREGISTERED_MATCHED_CONTROLS"
            if all(gates.values())
            else "ACCUMULATE"
        ),
    }


def advance_mcl_shock_wave_accumulator(
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
        raise ValueError("MCL shock-wave generation selection drifted")
    _validate_tape_contracts(rows, selection)
    contract_key = str(selection["contracts"]["MCL"]["expiry"])[:6]
    eligible = _utc(generation["eligible_start_utc"])
    observations = build_mcl_shock_observations(
        rows, bars, contract_key=contract_key, eligible_start=eligible
    )
    resets = [
        row
        for row in _minute_resets(bars, contract_key=contract_key)
        if _utc(row["at_utc"]) >= eligible
    ]
    complete, open_episode = replay_mcl_shock_wave_episodes(
        observations,
        resets=resets,
        generation=generation,
        rows=rows,
        bars=bars,
        news=news,
    )
    prior = {
        str(row["episode_id"]): row
        for row in mcl_shock_wave_episodes(tuple(ledger.records()))
    }
    appended = 0
    for episode in complete:
        episode_id = str(episode["episode_id"])
        existing = prior.get(episode_id)
        if existing is not None:
            if existing.get("episode_sha256") != episode.get("episode_sha256"):
                raise ValueError("MCL shock-wave episode changed across replay")
            continue
        terminal = _utc(episode["terminal_at_utc"])
        session = _session(_utc(episode["started_at_utc"]))
        ledger.checkpoint(
            evaluation_as_of=terminal,
            strategy_id=MCL_SHOCK_WAVE_VERSION,
            strategy_version=MCL_SHOCK_WAVE_ACCUMULATOR_VERSION,
            trading_date=session.isoformat() if session is not None else None,
            session="MCL_SHOCK_WAVE",
            status="EVALUATED",
            evidence=episode,
            recorded_at=now,
        )
        prior[episode_id] = episode
        appended += 1
    episodes = mcl_shock_wave_episodes(tuple(ledger.records()))
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
        "cohort": mcl_shock_wave_cohort(episodes, generation["cohort_gate"]),
    }


async def _main_async(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=MCL_SHOCK_WAVE_GENERATION_PATH)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            os.environ.get("MCL_SHOCK_WAVE_LEDGER", MCL_SHOCK_WAVE_LEDGER_PATH)
        ),
    )
    parser.add_argument(
        "--capital-plan", type=Path, default=Path("db/calibration/live_capital_plan.json")
    )
    parser.add_argument(
        "--predictive-generation", type=Path, default=MCL_PREDICTIVE_RUNTIME_GENERATION_PATH
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
    generation = load_mcl_shock_wave_generation(args.generation)
    plan = load_live_capital_plan(args.capital_plan.expanduser().resolve())
    selection_value, _path, _sha = load_allocated_live_selection(
        plan, sleeve_id=MCL_LIVE_CAPITAL_SLEEVE, repository_root=_ROOT
    )
    selection = load_mcl_live_selection_from_mapping(selection_value)
    predictive = load_mcl_predictive_generation(args.predictive_generation)
    if (
        predictive.get("generation_id") != generation["predictive_generation_id"]
        or predictive.get("selection_id") != generation["selection_id"]
    ):
        raise ValueError("MCL shock-wave predictive generation drifted")
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
    ledger = LiveCalibrationLedger(args.ledger.expanduser())
    if not rows:
        result = {
            "appended": 0,
            "eligible_start_utc": eligible.isoformat(),
            "source_records": 0,
            "eligible_observations": 0,
            "maximum_volume_multiple": 0.0,
            "complete_episodes_in_prefix": 0,
            "open_episode": None,
            "cohort": mcl_shock_wave_cohort(
                mcl_shock_wave_episodes(tuple(ledger.records())),
                generation["cohort_gate"],
            ),
        }
        bars_evidence = None
    else:
        config = load_config()
        if not config.readonly:
            raise ValueError("MCL shock-wave accumulator requires IBKR_READONLY=1")
        client = IBKRClient(config)
        await client.connect()
        try:
            bars, bars_evidence = await _recent_bars(
                client, selection=selection, observed_at=now
            )
        finally:
            await client.disconnect()
        snapshots = [
            row
            for path in sorted(args.news_history_dir.expanduser().glob("*.jsonl"))
            for row in load_news_history(path)
        ]
        result = advance_mcl_shock_wave_accumulator(
            ledger=ledger,
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
                "schema": MCL_SHOCK_WAVE_ACCUMULATOR_VERSION,
                "authority": MCL_SHOCK_WAVE_AUTHORITY,
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
