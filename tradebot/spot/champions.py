"""Canonical discovery and loading of promoted HF/LF spot champions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class ChampionRef:
    symbol: str
    track: str
    version: str | None
    readme_path: Path
    artifact_path: Path
    declared_version: str | None = None

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


def discover_current_champions(
    *,
    root: Path | None = None,
    symbols: tuple[str, ...] | None = None,
    tracks: tuple[str, ...] | None = None,
) -> tuple[ChampionRef, ...]:
    """Resolve one existing CURRENT artifact per symbol/track README."""
    root = (root or repo_root()).resolve()
    symbol_filter = {str(symbol).strip().upper() for symbol in symbols or ()}
    track_filter = {str(track).strip().upper() for track in tracks or ()}
    refs: list[ChampionRef] = []
    for readme_path in sorted((root / "backtests").glob("*/readme-*.md")):
        symbol = readme_path.parent.name.upper()
        track = readme_path.stem.removeprefix("readme-").upper()
        if symbol_filter and symbol not in symbol_filter:
            continue
        if track_filter and track not in track_filter:
            continue
        candidates = current_champion_candidates(readme_path.read_text(encoding="utf-8"))
        declared_version = candidates[0][0] if candidates else None
        for version, relative_path in candidates:
            artifact_path = _existing_artifact(root, relative_path)
            if artifact_path is None:
                continue
            refs.append(
                ChampionRef(
                    symbol=symbol,
                    track=track,
                    version=version,
                    readme_path=readme_path,
                    artifact_path=artifact_path,
                    declared_version=declared_version,
                )
            )
            break
    return tuple(refs)


def _champion_group(payload: dict) -> dict | None:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return None
    valid = [group for group in groups if isinstance(group, dict)]
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
    group = _champion_group(payload) if isinstance(payload, dict) else None
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
            use_rth_raw = strategy.get("signal_use_rth")
            use_rth = (
                use_rth_raw.strip().lower() in ("1", "true", "yes", "on")
                if isinstance(use_rth_raw, str)
                else bool(use_rth_raw)
            )
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
