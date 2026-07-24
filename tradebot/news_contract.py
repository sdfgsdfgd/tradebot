"""Strict persistent-state and output contract for the one-shot news reducer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SCHEMA = "tradebot.news-signal.v3"
SCORE_VERSION = "causal-impact-100.v2"
MAX_MEMORY_LINES = 160
MAX_MEMORY_CHARS = 32_000
MAX_ACTIVE_EVENTS = 24
MAX_EVIDENCE_URLS = 3
TRACKING_QUERY_KEYS = {"at_campaign", "at_medium", "mod", "output", "rss", "siteid"}
EVENT_STATUSES = {"rhetoric", "single_report", "corroborated", "confirmed"}
EVENT_BASES = {"summary_only", "single_content", "cross_source_content"}
EVENT_STATES = {"watch", "active", "resolving"}
EVENT_CHANNELS = {
    "supply",
    "demand",
    "inflation_rates",
    "growth_earnings",
    "liquidity_credit",
    "geopolitical",
    "mixed",
}
SIGNAL_CHANGES = {"new", "strengthening", "weakening", "reversal", "unchanged"}
HORIZONS = {1, 4, 24}
SCORE_COMPONENT_LIMITS = {
    "magnitude": 30,
    "transmission": 25,
    "surprise": 20,
    "immediacy": 15,
    "persistence": 10,
}
MEMORY_SECTION_LIMITS = {
    "## Calibration Anchors": 16,
    "## Active Regimes": 10,
    "## Durable Causal Priors": 12,
}
EVENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
EMPTY_MEMORY = """# Trade Research Memory

_Curated causal memory for XSP and MCL. This is not a news diary._

## Mission

Retain only causal structures that materially alter the expected distribution
of XSP or MCL. Separate probability, evidence confidence, contract impact, and
realized response.

## Calibration Anchors

- **XSP reference ceiling — 100.** A verified system-scale shock that breaks US
  economic or market function within the scored horizon.
- **MCL reference ceiling — 100.** A verified sustained physical closure of
  Bab el-Mandeb, Hormuz, or an equivalent oil-supply loss within the horizon.
- **Broad reciprocal-tariff escalation — 2025-04.** Reconstructed XSP -1 /
  impact 93 / confidence .99 (26 magnitude + 25 transmission + 19 surprise +
  15 immediacy + 8 persistence). The S&P 500 fell 12.14% from April 2 to
  April 8 and did not regain its pre-shock high until June 27; the May 12
  US-China de-escalation produced a 3.26% one-day rise. Treat the simultaneous
  WTI decline as mixed attribution because OPEC+ also accelerated production.

## Active Regimes

- None.

## Durable Causal Priors

- None.
"""


class NewsError(RuntimeError):
    """A failure that must leave the last valid signal untouched."""


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("news timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"unsupported article URL: {raw_url!r}")
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), host, path, urlencode(sorted(query)), ""))


def output_schema() -> dict[str, object]:
    components = {
        "type": "object",
        "additionalProperties": False,
        "required": list(SCORE_COMPONENT_LIMITS),
        "properties": {
            name: {"type": "integer", "minimum": 0, "maximum": limit}
            for name, limit in SCORE_COMPONENT_LIMITS.items()
        },
    }
    score = {
        "type": "object",
        "additionalProperties": False,
        "required": ["direction", "impact", "components", "calibration"],
        "properties": {
            "direction": {"type": "integer", "enum": [-1, 0, 1]},
            "impact": {"type": "integer", "minimum": 0, "maximum": 100},
            "components": components,
            "calibration": {"type": "string", "minLength": 1, "maxLength": 200},
        },
    }
    active_event = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "umbrella",
            "event",
            "state",
            "status",
            "basis",
            "channel",
            "confidence",
            "first_seen_utc",
            "last_material_change_utc",
            "last_verified_utc",
            "review_after_utc",
            "mechanism",
            "invalidation",
            "evidence_urls",
            "xsp",
            "mcl",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 64},
            "umbrella": {"type": "string", "minLength": 1, "maxLength": 80},
            "event": {"type": "string", "minLength": 1, "maxLength": 120},
            "state": {"type": "string", "enum": sorted(EVENT_STATES)},
            "status": {"type": "string", "enum": sorted(EVENT_STATUSES)},
            "basis": {"type": "string", "enum": sorted(EVENT_BASES)},
            "channel": {"type": "string", "enum": sorted(EVENT_CHANNELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "first_seen_utc": {"type": "string", "minLength": 20, "maxLength": 20},
            "last_material_change_utc": {
                "type": "string",
                "minLength": 20,
                "maxLength": 20,
            },
            "last_verified_utc": {"type": ["string", "null"]},
            "review_after_utc": {"type": "string", "minLength": 20, "maxLength": 20},
            "mechanism": {"type": "string", "minLength": 1, "maxLength": 240},
            "invalidation": {"type": "string", "minLength": 1, "maxLength": 240},
            "evidence_urls": {
                "type": "array",
                "items": {"type": "string", "minLength": 8, "maxLength": 2_000},
                "minItems": 1,
                "maxItems": MAX_EVIDENCE_URLS,
            },
            "xsp": score,
            "mcl": score,
        },
    }
    signal = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "direction",
            "impact",
            "confidence",
            "horizon_hours",
            "change",
            "mechanism",
            "calibration",
            "drivers",
        ],
        "properties": {
            "direction": {"type": "integer", "enum": [-1, 0, 1]},
            "impact": {"type": "integer", "minimum": 0, "maximum": 100},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "horizon_hours": {"type": "integer", "enum": [1, 4, 24]},
            "change": {"type": "string", "enum": sorted(SIGNAL_CHANGES)},
            "mechanism": {"type": "string", "minLength": 1, "maxLength": 240},
            "calibration": {"type": "string", "minLength": 1, "maxLength": 200},
            "drivers": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["active_events", "removals", "assets", "memory_markdown"],
        "properties": {
            "active_events": {
                "type": "array",
                "maxItems": MAX_ACTIVE_EVENTS,
                "items": active_event,
            },
            "removals": {
                "type": "array",
                "maxItems": MAX_ACTIVE_EVENTS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "reason", "resolved_at_utc"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
                        "resolved_at_utc": {
                            "type": "string",
                            "minLength": 20,
                            "maxLength": 20,
                        },
                    },
                },
            },
            "assets": {
                "type": "object",
                "additionalProperties": False,
                "required": ["XSP", "MCL"],
                "properties": {"XSP": signal, "MCL": signal},
            },
            "memory_markdown": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_MEMORY_CHARS,
            },
        },
    }


def _assert_exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise NewsError(f"{label} keys differ: {sorted(value)}")


def _assert_int(
    value: object,
    *,
    allowed: set[int] | None = None,
    low: int = 0,
    high: int = 100,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NewsError(f"expected integer, got {value!r}")
    if allowed is not None and value not in allowed:
        raise NewsError(f"integer outside enum: {value}")
    if allowed is None and not low <= value <= high:
        raise NewsError(f"integer outside range: {value}")
    return value


def _assert_text(value: object, *, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise NewsError(f"{label} must be non-empty and at most {limit} characters")
    return value


def _assert_references(
    value: object,
    *,
    allowed: set[str],
    label: str,
    required: bool,
) -> list[str]:
    if not isinstance(value, list) or len(value) > 5 or (required and not value):
        raise NewsError(f"{label} must contain {'1..5' if required else '0..5'} IDs")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise NewsError(f"{label} contains an unknown active-event ID")
    if len(set(value)) != len(value):
        raise NewsError(f"{label} contains duplicate IDs")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise NewsError(f"{label} must be a UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise NewsError(f"{label} must use YYYY-MM-DDTHH:MM:SSZ") from exc


def _assert_confidence(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NewsError(f"{label} must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise NewsError(f"{label} outside 0..1")
    return result


def _assert_urls(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_EVIDENCE_URLS:
        raise NewsError(f"{label} must contain 1..{MAX_EVIDENCE_URLS} URLs")
    urls: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > 2_000:
            raise NewsError(f"{label} contains an invalid URL")
        try:
            url = canonical_url(item)
        except ValueError as exc:
            raise NewsError(f"{label} contains an invalid URL") from exc
        if urlsplit(url).scheme != "https":
            raise NewsError(f"{label} requires HTTPS evidence")
        urls.append(url)
    if len(set(urls)) != len(urls):
        raise NewsError(f"{label} contains duplicate URLs")
    return urls


def _validate_score(
    value: object,
    *,
    basis: str,
    status: str,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NewsError(f"{label} must be an object")
    _assert_exact_keys(value, {"direction", "impact", "components", "calibration"}, label)
    direction = _assert_int(value["direction"], allowed={-1, 0, 1})
    impact = _assert_int(value["impact"])
    components = value["components"]
    if not isinstance(components, dict) or set(components) != set(SCORE_COMPONENT_LIMITS):
        raise NewsError(f"{label}.components are invalid")
    total = sum(
        _assert_int(components[name], high=limit)
        for name, limit in SCORE_COMPONENT_LIMITS.items()
    )
    if total != impact or (direction == 0) != (impact == 0):
        raise NewsError(f"{label} score is internally inconsistent")
    _assert_text(value["calibration"], label=f"{label}.calibration", limit=200)
    if impact > {"summary_only": 49, "single_content": 79, "cross_source_content": 100}[basis]:
        raise NewsError(f"{label} exceeds its evidence-basis ceiling")
    if status == "rhetoric" and impact > 30:
        raise NewsError(f"{label} rhetoric score exceeds 30")
    return value


ACTIVE_EVENT_KEYS = {
    "id",
    "umbrella",
    "event",
    "state",
    "status",
    "basis",
    "channel",
    "confidence",
    "first_seen_utc",
    "last_material_change_utc",
    "last_verified_utc",
    "review_after_utc",
    "mechanism",
    "invalidation",
    "evidence_urls",
    "xsp",
    "mcl",
}
NON_MATERIAL_EVENT_KEYS = {
    "last_material_change_utc",
    "last_verified_utc",
    "review_after_utc",
}


def _material_event_view(event: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in event.items() if key not in NON_MATERIAL_EVENT_KEYS}


def canonicalize_lifecycle_timestamps(
    value: object,
    *,
    previous_events: list[dict[str, object]],
    as_of: datetime,
) -> object:
    """Make the runtime, rather than the model, authoritative for event clocks."""
    if not isinstance(value, dict):
        return value
    events = value.get("active_events")
    if not isinstance(events, list):
        return value
    current = _utc_iso(as_of)
    previous_by_id = {str(event["id"]): event for event in previous_events}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("id"), str):
            continue
        previous = previous_by_id.get(event["id"])
        if previous is None:
            event["first_seen_utc"] = current
            event["last_material_change_utc"] = current
            if event.get("last_verified_utc") is not None:
                event["last_verified_utc"] = current
            continue
        event["first_seen_utc"] = previous["first_seen_utc"]
        event["last_material_change_utc"] = (
            current
            if _material_event_view(event) != _material_event_view(previous)
            else previous["last_material_change_utc"]
        )
        if (
            event.get("last_verified_utc") is not None
            and event["last_verified_utc"] != previous["last_verified_utc"]
        ):
            event["last_verified_utc"] = current
    removals = value.get("removals")
    if isinstance(removals, list):
        for removal in removals:
            if isinstance(removal, dict):
                removal["resolved_at_utc"] = current
    return value


def _validate_active_event(
    event: object,
    *,
    label: str,
    as_of: datetime,
    previous: dict[str, object] | None,
    is_output: bool,
) -> dict[str, object]:
    if not isinstance(event, dict):
        raise NewsError(f"{label} must be an object")
    _assert_exact_keys(event, ACTIVE_EVENT_KEYS, label)
    event_id = _assert_text(event["id"], label=f"{label}.id", limit=64)
    if EVENT_ID_PATTERN.fullmatch(event_id) is None:
        raise NewsError(f"{label}.id must be stable lowercase kebab-case")
    _assert_text(event["umbrella"], label=f"{label}.umbrella", limit=80)
    _assert_text(event["event"], label=f"{label}.event", limit=120)
    _assert_text(event["mechanism"], label=f"{label}.mechanism", limit=240)
    _assert_text(event["invalidation"], label=f"{label}.invalidation", limit=240)
    if (
        event["state"] not in EVENT_STATES
        or event["status"] not in EVENT_STATUSES
        or event["basis"] not in EVENT_BASES
        or event["channel"] not in EVENT_CHANNELS
    ):
        raise NewsError(f"{label} contains an invalid state, status, basis, or channel")
    _assert_confidence(event["confidence"], label=f"{label}.confidence")

    first_seen = _parse_utc(event["first_seen_utc"], label=f"{label}.first_seen_utc")
    last_change = _parse_utc(
        event["last_material_change_utc"],
        label=f"{label}.last_material_change_utc",
    )
    last_verified_raw = event["last_verified_utc"]
    last_verified = (
        None
        if last_verified_raw is None
        else _parse_utc(last_verified_raw, label=f"{label}.last_verified_utc")
    )
    review_after = _parse_utc(event["review_after_utc"], label=f"{label}.review_after_utc")
    if first_seen > last_change or last_change > as_of:
        raise NewsError(f"{label} has impossible material timestamps")
    if last_verified is not None and not first_seen <= last_verified <= as_of:
        raise NewsError(f"{label} has an impossible verification timestamp")

    evidence = _assert_urls(event["evidence_urls"], label=f"{label}.evidence_urls")
    event["evidence_urls"] = evidence
    source_hosts = {(urlsplit(url).hostname or "").removeprefix("www.") for url in evidence}
    if event["basis"] == "cross_source_content" and len(source_hosts) < 2:
        raise NewsError(f"{label} cross-source basis requires distinct source hosts")
    if event["status"] == "confirmed" and event["basis"] == "summary_only":
        raise NewsError(f"{label} cannot be confirmed from summaries only")

    for asset in ("xsp", "mcl"):
        _validate_score(
            event[asset],
            basis=str(event["basis"]),
            status=str(event["status"]),
            label=f"{label}.{asset}",
        )
    if event["mcl"]["impact"] == 100 and (
        event["status"] != "confirmed"
        or event["basis"] != "cross_source_content"
        or event["channel"] not in {"supply", "geopolitical", "mixed"}
        or len(source_hosts) < 2
    ):
        raise NewsError("MCL impact 100 requires confirmed cross-source physical evidence")

    if is_output:
        if previous is None:
            if first_seen != as_of or last_change != as_of:
                raise NewsError(f"{label} new-event timestamps must equal as_of_utc")
            if last_verified is not None and last_verified != as_of:
                raise NewsError(f"{label}.last_verified_utc must equal as_of_utc or null")
        else:
            if event["first_seen_utc"] != previous["first_seen_utc"]:
                raise NewsError(f"{label}.first_seen_utc is immutable")
            materially_changed = _material_event_view(event) != _material_event_view(previous)
            expected_change = _utc_iso(as_of) if materially_changed else previous["last_material_change_utc"]
            if event["last_material_change_utc"] != expected_change:
                raise NewsError(f"{label}.last_material_change_utc does not match its material diff")
            old_verified = previous["last_verified_utc"]
            if old_verified is not None and last_verified is None:
                raise NewsError(f"{label}.last_verified_utc cannot move backward to null")
            if event["last_verified_utc"] != old_verified and last_verified != as_of:
                raise NewsError(f"{label}.last_verified_utc changes must equal as_of_utc")

        if review_after <= as_of:
            raise NewsError(f"{label}.review_after_utc must be in the future")
        peak_impact = max(int(event["xsp"]["impact"]), int(event["mcl"]["impact"]))
        review_limit = (
            timedelta(hours=24)
            if peak_impact >= 80 or event["state"] == "resolving"
            else timedelta(hours=72)
            if peak_impact >= 50
            else timedelta(days=7)
        )
        if review_after > as_of + review_limit:
            raise NewsError(f"{label}.review_after_utc exceeds its impact review limit")
    return event


def validate_memory_markdown(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NewsError("memory_markdown must be non-empty")
    if len(value) > MAX_MEMORY_CHARS:
        raise NewsError("memory_markdown exceeds the character ceiling")
    lines = value.splitlines()
    if len(lines) > MAX_MEMORY_LINES:
        raise NewsError(f"memory_markdown must stay at or below {MAX_MEMORY_LINES} lines")
    if not lines or lines[0].strip() != "# Trade Research Memory":
        raise NewsError("memory_markdown must begin with the canonical title")
    headings = ["## Mission", *MEMORY_SECTION_LIMITS]
    positions = []
    for heading in headings:
        if lines.count(heading) != 1:
            raise NewsError(f"memory_markdown must contain exactly one {heading!r}")
        positions.append(lines.index(heading))
    if positions != sorted(positions):
        raise NewsError("memory_markdown sections are out of order")
    if any("\x00" in line or len(line) > 2_000 for line in lines):
        raise NewsError("memory_markdown contains an invalid line")
    if any(line.startswith("## ") and line not in headings for line in lines):
        raise NewsError("memory_markdown contains an unexpected section")
    for index, (heading, limit) in enumerate(MEMORY_SECTION_LIMITS.items(), start=1):
        start = positions[index] + 1
        stop = positions[index + 1] if index + 1 < len(positions) else len(lines)
        if sum(line.startswith("### ") for line in lines[start:stop]) > limit:
            raise NewsError(f"{heading} exceeds its {limit}-entry ceiling")
    if "XSP reference ceiling" not in value or "MCL reference ceiling" not in value:
        raise NewsError("memory_markdown must preserve both 100-point reference ceilings")
    return value.rstrip() + "\n"


def validate_analysis(
    value: object,
    *,
    previous_events: list[dict[str, object]],
    as_of: datetime,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NewsError("Codex response must be a JSON object")
    _assert_exact_keys(
        value,
        {"active_events", "removals", "assets", "memory_markdown"},
        "analysis",
    )
    events = value["active_events"]
    if not isinstance(events, list) or len(events) > MAX_ACTIVE_EVENTS:
        raise NewsError(f"active_events must contain at most {MAX_ACTIVE_EVENTS} entries")
    previous_by_id = {str(event["id"]): event for event in previous_events}
    active_ids: set[str] = set()
    for index, event in enumerate(events):
        validated = _validate_active_event(
            event,
            label=f"active event {index}",
            as_of=as_of,
            previous=previous_by_id.get(str(event.get("id"))) if isinstance(event, dict) else None,
            is_output=True,
        )
        if validated["id"] in active_ids:
            raise NewsError("active_events contains duplicate IDs")
        active_ids.add(str(validated["id"]))

    removals = value["removals"]
    if not isinstance(removals, list) or len(removals) > MAX_ACTIVE_EVENTS:
        raise NewsError(f"removals must contain at most {MAX_ACTIVE_EVENTS} entries")
    removed_ids: set[str] = set()
    for index, removal in enumerate(removals):
        if not isinstance(removal, dict):
            raise NewsError(f"removal {index} must be an object")
        _assert_exact_keys(removal, {"id", "reason", "resolved_at_utc"}, f"removal {index}")
        event_id = _assert_text(removal["id"], label=f"removal {index}.id", limit=64)
        _assert_text(removal["reason"], label=f"removal {index}.reason", limit=240)
        if event_id not in previous_by_id or event_id in active_ids or event_id in removed_ids:
            raise NewsError(f"removal {index} does not identify one omitted prior event")
        if _parse_utc(
            removal["resolved_at_utc"],
            label=f"removal {index}.resolved_at_utc",
        ) != as_of:
            raise NewsError(f"removal {index}.resolved_at_utc must equal as_of_utc")
        removed_ids.add(event_id)
    if removed_ids != set(previous_by_id) - active_ids:
        raise NewsError("every omitted prior event requires exactly one removal")

    assets = value["assets"]
    if not isinstance(assets, dict):
        raise NewsError("assets must be an object")
    _assert_exact_keys(assets, {"XSP", "MCL"}, "assets")
    signal_keys = {
        "direction",
        "impact",
        "confidence",
        "horizon_hours",
        "change",
        "mechanism",
        "calibration",
        "drivers",
    }
    for symbol in ("XSP", "MCL"):
        signal = assets[symbol]
        if not isinstance(signal, dict):
            raise NewsError(f"{symbol} signal must be an object")
        _assert_exact_keys(signal, signal_keys, symbol)
        direction = _assert_int(signal["direction"], allowed={-1, 0, 1})
        impact = _assert_int(signal["impact"])
        if (direction == 0) != (impact == 0):
            raise NewsError(f"{symbol} aggregate score is internally inconsistent")
        _assert_confidence(signal["confidence"], label=f"{symbol}.confidence")
        _assert_int(signal["horizon_hours"], allowed=HORIZONS)
        if signal["change"] not in SIGNAL_CHANGES:
            raise NewsError(f"{symbol}.change is invalid")
        _assert_text(signal["mechanism"], label=f"{symbol}.mechanism", limit=240)
        _assert_text(signal["calibration"], label=f"{symbol}.calibration", limit=200)
        drivers = _assert_references(
            signal["drivers"],
            allowed=active_ids,
            label=f"{symbol}.drivers",
            required=False,
        )
        if (impact == 0 and drivers) or (impact > 0 and not drivers):
            raise NewsError(f"{symbol}.drivers do not match aggregate impact")
    if assets["MCL"]["impact"] == 100 and not any(
        event["mcl"]["impact"] == 100 for event in events
    ):
        raise NewsError("aggregate MCL impact 100 requires a maximum-impact event")
    value["memory_markdown"] = validate_memory_markdown(value["memory_markdown"])
    return value


def load_events(path: Path, *, as_of: datetime) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NewsError(f"event ledger read failed: {exc}") from exc
    if len(raw) > 512_000:
        raise NewsError("event ledger exceeds its size ceiling")
    events: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            raise NewsError(f"event ledger line {index + 1} is blank")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NewsError(f"event ledger line {index + 1} is invalid JSON") from exc
        events.append(
            _validate_active_event(
                event,
                label=f"persisted event {index}",
                as_of=as_of,
                previous=None,
                is_output=False,
            )
        )
    if len(events) > MAX_ACTIVE_EVENTS:
        raise NewsError(f"event ledger exceeds {MAX_ACTIVE_EVENTS} records")
    ids = [str(event["id"]) for event in events]
    if len(set(ids)) != len(ids):
        raise NewsError("event ledger contains duplicate IDs")
    return events


def _events_text(events: list[dict[str, object]]) -> str:
    if not events:
        return ""
    return "".join(
        f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}\n"
        for event in events
    )


def _event_snapshot(
    events: list[dict[str, object]],
    *,
    as_of: datetime,
) -> dict[str, list[dict[str, object]]]:
    snapshot: dict[str, list[dict[str, object]]] = {
        "breaking": [],
        "day": [],
        "week": [],
        "month": [],
        "persistent": [],
    }
    ordered = sorted(
        events,
        key=lambda event: (
            -max(int(event["xsp"]["impact"]), int(event["mcl"]["impact"])),
            str(event["id"]),
        ),
    )
    for event in ordered:
        age = as_of - _parse_utc(
            event["last_material_change_utc"],
            label=f"{event['id']}.last_material_change_utc",
        )
        bucket = (
            "breaking"
            if age <= timedelta(hours=4)
            else "day"
            if age <= timedelta(hours=24)
            else "week"
            if age <= timedelta(days=7)
            else "month"
            if age <= timedelta(days=31)
            else "persistent"
        )
        snapshot[bucket].append(event)
    return snapshot


def _event_changes(
    previous: list[dict[str, object]],
    current: list[dict[str, object]],
    removals: list[dict[str, object]],
) -> dict[str, object]:
    old = {str(event["id"]): event for event in previous}
    new = {str(event["id"]): event for event in current}
    shared = set(old) & set(new)
    materially_updated = sorted(
        event_id
        for event_id in shared
        if _material_event_view(old[event_id]) != _material_event_view(new[event_id])
    )
    return {
        "added": sorted(set(new) - set(old)),
        "materially_updated": materially_updated,
        "reviewed_unchanged": sorted(shared - set(materially_updated)),
        "removed": removals,
    }
