"""Canonical discovery and loading of promoted HF/LF spot champions."""

from __future__ import annotations

import json
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .codec import bool_from_payload


_HEADING_RE = re.compile(r"^(?P<hash>#{3,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)
_CURRENT_RE = re.compile(r"^CURRENT(?:\s|$|\()", re.IGNORECASE)
_VERSION_RE = re.compile(r"\(v(?P<version>[^)]+)\)", re.IGNORECASE)
_PATH_VERSION_RE = re.compile(
    r"(?:^|[_/.-])v(?P<version>\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_PRESET_LINE_RE = re.compile(
    r"^-\s*Preset file(?:\s*\([^)]*\))?:\s*(?P<body>.+)$",
    re.MULTILINE | re.IGNORECASE,
)
_JSON_BACKTICK_RE = re.compile(r"`(?P<path>backtests/[^`]+\.json)`")
_JSON_INLINE_RE = re.compile(r"(?P<path>backtests/[^\s)]+\.json)")
_CROWN_SCHEMA = "tradebot.spot.champion.v1"


@dataclass(frozen=True)
class ChampionRef:
    symbol: str
    track: str
    version: str | None
    declaration_path: Path
    artifact_path: Path
    declared_version: str | None = None
    strategy_key: str | None = None

    @property
    def used_fallback(self) -> bool:
        return bool(
            self.declared_version
            and self.version
            and self.declared_version != self.version
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _version_sort_key(version: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(version or ""))) or (-1,)


def _heading_section(text: str, headings: list[re.Match[str]], index: int) -> str:
    heading = headings[index]
    level = len(heading.group("hash"))
    end = len(text)
    for following in headings[index + 1 :]:
        if len(following.group("hash")) <= level:
            end = following.start()
            break
    return text[heading.end() : end]


def _current_artifact_path(section: str) -> str | None:
    preset_line = _PRESET_LINE_RE.search(section)
    if preset_line is not None:
        body = str(preset_line.group("body") or "")
        match = _JSON_BACKTICK_RE.search(body) or _JSON_INLINE_RE.search(body)
        if match is not None:
            return str(match.group("path") or "").strip() or None
    match = _JSON_BACKTICK_RE.search(section) or _JSON_INLINE_RE.search(section)
    if match is None:
        return None
    return str(match.group("path") or "").strip() or None


def current_champion_candidates(
    readme_text: str,
) -> list[tuple[str | None, str | None]]:
    """Return CURRENT declarations newest-first without hiding missing artifacts."""
    text = str(readme_text or "")
    headings = list(_HEADING_RE.finditer(text))
    candidates: list[tuple[tuple[int, ...], int, str | None, str | None]] = []
    for index, heading in enumerate(headings):
        title = str(heading.group("title") or "").strip()
        if not _CURRENT_RE.match(title):
            continue
        path = _current_artifact_path(_heading_section(text, headings, index))
        if not path:
            continue
        match = _VERSION_RE.search(title) or _PATH_VERSION_RE.search(path)
        version = (
            str(match.group("version") or "").strip() or None
            if match is not None
            else None
        )
        candidates.append((_version_sort_key(version), heading.start(), version, path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(version, path) for _key, _start, version, path in candidates]


def _existing_artifact(root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() and candidate.suffix.lower() == ".json" else None


def _machine_champion_refs(root: Path) -> dict[tuple[str, str], ChampionRef]:
    refs: dict[tuple[str, str], ChampionRef] = {}
    for declaration_path in sorted((root / "backtests").glob("*/current-*.json")):
        try:
            payload = json.loads(declaration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("schema") != _CROWN_SCHEMA:
            continue
        symbol = str(payload.get("symbol") or declaration_path.parent.name).strip().upper()
        track = str(payload.get("track") or declaration_path.stem.removeprefix("current-")).strip().upper()
        artifact_path = _existing_artifact(root, str(payload.get("artifact") or ""))
        if not symbol or track not in {"HF", "LF"} or artifact_path is None:
            continue
        artifact_sha256 = str(payload.get("artifact_sha256") or "").strip().lower()
        if artifact_sha256:
            try:
                if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_sha256:
                    continue
            except OSError:
                continue
        version = str(payload.get("version") or "").strip() or None
        refs[(symbol, track)] = ChampionRef(
            symbol=symbol,
            track=track,
            version=version,
            declaration_path=declaration_path,
            artifact_path=artifact_path,
            declared_version=version,
            strategy_key=str(payload.get("strategy_key") or "").strip() or None,
        )
    return refs


def discover_current_champions(
    *,
    root: Path | None = None,
    symbols: tuple[str, ...] | None = None,
    tracks: tuple[str, ...] | None = None,
) -> tuple[ChampionRef, ...]:
    """Resolve each machine crown, falling back to legacy README declarations."""
    root = (root or repo_root()).resolve()
    symbol_filter = {str(symbol).strip().upper() for symbol in symbols or ()}
    track_filter = {str(track).strip().upper() for track in tracks or ()}
    refs_by_lane = _machine_champion_refs(root)
    for readme_path in sorted((root / "backtests").glob("*/readme-*.md")):
        symbol = readme_path.parent.name.upper()
        track = readme_path.stem.removeprefix("readme-").upper()
        if symbol_filter and symbol not in symbol_filter:
            continue
        if track_filter and track not in track_filter:
            continue
        lane = (symbol, track)
        if lane in refs_by_lane:
            continue
        candidates = current_champion_candidates(readme_path.read_text(encoding="utf-8"))
        declared_version = candidates[0][0] if candidates else None
        for version, relative_path in candidates:
            artifact_path = _existing_artifact(root, relative_path)
            if artifact_path is None:
                continue
            refs_by_lane[lane] = ChampionRef(
                symbol=symbol,
                track=track,
                version=version,
                declaration_path=readme_path,
                artifact_path=artifact_path,
                declared_version=declared_version,
            )
            break
    return tuple(
        ref
        for lane, ref in sorted(refs_by_lane.items())
        if (not symbol_filter or lane[0] in symbol_filter)
        and (not track_filter or lane[1] in track_filter)
    )


def promotion_receipt(
    candidate_windows: list[dict],
    incumbent_windows: tuple[dict, ...],
    *,
    objective: str = "stability",
) -> dict:
    """Prove a challenger against the incumbent's exact windows and objective."""
    objective = str(objective or "stability").strip().lower()
    if objective not in {"stability", "ratio_all", "pnl_all"}:
        raise ValueError(f"Unknown promotion objective: {objective!r}")

    incumbent = {
        (str(row.get("start") or ""), str(row.get("end") or "")): row
        for row in incumbent_windows
        if isinstance(row, dict) and row.get("start") and row.get("end")
    }
    candidate = {
        (str(row.get("start") or ""), str(row.get("end") or "")): row
        for row in candidate_windows
        if isinstance(row, dict) and row.get("start") and row.get("end")
    }

    def number(row: dict, *keys: str) -> float:
        for key in keys:
            try:
                return float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
        return 0.0

    windows: list[dict] = []
    for key, baseline in incumbent.items():
        current = candidate.get(key)
        if current is None:
            continue
        candidate_ratio = number(current, "roi_over_dd_pct", "pnl_over_dd")
        incumbent_ratio = number(baseline, "roi_over_dd_pct", "pnl_over_dd")
        windows.append(
            {
                "start": key[0],
                "end": key[1],
                "ratio_delta": candidate_ratio - incumbent_ratio,
                "pnl_delta": number(current, "pnl") - number(baseline, "pnl"),
                "trade_delta": int(number(current, "trades")) - int(number(baseline, "trades")),
                "candidate_ratio": candidate_ratio,
                "incumbent_ratio": incumbent_ratio,
            }
        )

    complete = bool(incumbent) and len(windows) == len(incumbent)
    ratio_deltas = [float(row["ratio_delta"]) for row in windows]
    pnl_deltas = [float(row["pnl_delta"]) for row in windows]
    trade_deltas = [int(row["trade_delta"]) for row in windows]
    candidate_floor = min((float(row["candidate_ratio"]) for row in windows), default=0.0)
    incumbent_floor = min((float(row["incumbent_ratio"]) for row in windows), default=0.0)
    positive_all = complete and all(number(candidate[key], "pnl") > 0.0 for key in incumbent)
    ratio_dominates_all = complete and all(delta >= 0.0 for delta in ratio_deltas) and any(
        delta > 0.0 for delta in ratio_deltas
    )
    pnl_dominates_all = complete and all(delta >= 0.0 for delta in pnl_deltas) and any(
        delta > 0.0 for delta in pnl_deltas
    )
    floor_delta = candidate_floor - incumbent_floor if complete else None
    objective_passed = {
        "stability": bool(complete and floor_delta is not None and floor_delta > 0.0),
        "ratio_all": ratio_dominates_all,
        "pnl_all": pnl_dominates_all,
    }[objective]
    eligible = bool(positive_all and objective_passed)
    reasons: list[str] = []
    if not incumbent:
        reasons.append("incumbent_windows_missing")
    elif not complete:
        reasons.append("exact_window_coverage_missing")
    if complete and not positive_all:
        reasons.append("non_positive_window")
    if complete and not objective_passed:
        reasons.append(f"{objective}_not_improved")
    return {
        "objective": objective,
        "eligible": eligible,
        "reasons": reasons,
        "complete": complete,
        "positive_all": positive_all,
        "ratio_dominates_all": ratio_dominates_all,
        "pnl_dominates_all": pnl_dominates_all,
        "activity_preserved_all": complete and all(delta >= 0 for delta in trade_deltas),
        "floor_delta": floor_delta,
        "windows": windows,
    }


def promote_champion(
    *,
    root: Path,
    symbol: str,
    track: str,
    version: str,
    artifact_path: Path,
    strategy_key: str,
    receipt: dict,
) -> Path:
    """Atomically advance one operational crown after a passing receipt."""
    root = Path(root).resolve()
    symbol = str(symbol).strip().upper()
    track = str(track).strip().upper()
    if track not in {"HF", "LF"}:
        raise ValueError("Champion promotion requires track HF or LF")
    if not bool(receipt.get("eligible")):
        raise ValueError("Champion promotion receipt is not eligible")
    artifact_path = Path(artifact_path).resolve()
    try:
        artifact_path.relative_to(root)
        artifact_bytes = artifact_path.read_bytes()
        artifact_payload = json.loads(artifact_bytes)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("Champion artifact must be valid JSON inside the repository") from exc
    if _champion_group(
        artifact_payload if isinstance(artifact_payload, dict) else {},
        strategy_key=str(strategy_key),
    ) is None:
        raise ValueError("Champion artifact has no loadable group")

    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    snapshot_path = (
        root
        / "backtests"
        / symbol.lower()
        / "champions"
        / f"{track.lower()}-{artifact_sha256[:16]}.json"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if snapshot_path.read_bytes() != artifact_bytes:
            raise ValueError("Champion snapshot hash collision")
    else:
        temporary = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(artifact_bytes)
        os.replace(temporary, snapshot_path)

    promoted_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema": _CROWN_SCHEMA,
        "symbol": symbol,
        "track": track,
        "version": str(version).strip() or promoted_at,
        "artifact": snapshot_path.relative_to(root).as_posix(),
        "artifact_sha256": artifact_sha256,
        "strategy_key": str(strategy_key),
        "promoted_at": promoted_at,
        "promotion": dict(receipt),
    }
    declaration_path = root / "backtests" / symbol.lower() / f"current-{track.lower()}.json"
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = declaration_path.with_name(f".{declaration_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, declaration_path)
    return declaration_path


def _champion_group(payload: dict, *, strategy_key: str | None = None) -> dict | None:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return None
    valid = [group for group in groups if isinstance(group, dict)]
    if strategy_key:
        return next(
            (group for group in valid if str(group.get("_key") or "") == str(strategy_key)),
            None,
        )
    return next(
        (group for group in valid if "KINGMAKER #01" in str(group.get("name") or "")),
        valid[0] if valid else None,
    )


def load_champion_group(ref: ChampionRef) -> dict | None:
    """Load the promoted group with stable source, lane, and transport metadata."""
    try:
        payload = json.loads(ref.artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    group = (
        _champion_group(payload, strategy_key=ref.strategy_key)
        if isinstance(payload, dict)
        else None
    )
    if group is None:
        return None
    loaded = dict(group)
    loaded["_source"] = f"champion:{ref.symbol}:{ref.track}:v{ref.version or '?'}"
    loaded["_track"] = ref.track
    loaded["_version"] = ref.version
    entries = loaded.get("entries")
    hydrated: list[dict] = []
    for raw_entry in entries if isinstance(entries, list) else ():
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        strategy_raw = entry.get("strategy")
        if isinstance(strategy_raw, dict):
            strategy = dict(strategy_raw)
            use_rth = bool_from_payload(strategy.get("signal_use_rth"))
            if (
                str(strategy.get("instrument") or "").strip().lower() == "spot"
                and str(strategy.get("spot_sec_type") or "STK").strip().upper() == "STK"
                and not use_rth
                and not str(strategy.get("spot_next_open_session") or "").strip()
            ):
                strategy["spot_next_open_session"] = "tradable_24x5"
            entry["strategy"] = strategy
        hydrated.append(entry)
    loaded["entries"] = hydrated

    name = str(loaded.get("name") or "")
    version_tag = f"v{ref.version}" if ref.version else ""
    spot_tag = f"Spot ({ref.symbol})"
    if spot_tag in name and version_tag and not re.search(
        rf"\b{re.escape(version_tag)}\b", name, re.IGNORECASE
    ):
        name = name.replace(spot_tag, f"{spot_tag} {version_tag}", 1)
    if ref.track == "HF" and name and "[HF]" not in name:
        name = f"{name} [HF]"
    loaded["name"] = name
    return loaded


def load_current_champion_groups(
    *,
    root: Path | None = None,
    symbols: tuple[str, ...] | None = None,
    tracks: tuple[str, ...] | None = None,
) -> tuple[list[dict], list[str]]:
    groups: list[dict] = []
    warnings: list[str] = []
    for ref in discover_current_champions(
        root=root,
        symbols=symbols,
        tracks=tracks,
    ):
        if ref.used_fallback:
            warnings.append(
                f"{ref.symbol} {ref.track} crown v{ref.declared_version} missing; "
                f"loaded v{ref.version}"
            )
        group = load_champion_group(ref)
        if group is not None:
            groups.append(group)
    return groups, warnings
