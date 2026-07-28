"""Mirror q's causal XSP news evidence and build one compact pressure ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from ..news.contract import load_news_history
from .xsp_context import xsp_fundamental_context_at


PRESSURE_FIELDS = (
    "usable",
    "signal_as_of_utc",
    "snapshot_fingerprint",
    "direction",
    "impact",
    "confidence",
    "horizon_hours",
    "reason",
    "signed_pressure",
    "pressure_delta",
    "pressure_interval_seconds",
    "pressure_velocity_per_hour",
)
REMOTE_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _records(path: Path) -> Iterable[Mapping[str, object]]:
    payload = path.read_bytes()
    lines = payload.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not payload.endswith(b"\n"):
                return
            raise
        if isinstance(row, Mapping):
            yield row


def compact_pressure_rows(
    records: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Deduplicate hourly checkpoints into one exact row per news publication."""

    compact: dict[tuple[object, object], dict[str, object]] = {}
    for record in records:
        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        pressure = evidence.get("fundamental_pressure")
        if not isinstance(pressure, Mapping):
            continue
        values = {field: pressure.get(field) for field in PRESSURE_FIELDS}
        if values["snapshot_fingerprint"] is None:
            continue
        key = (values["snapshot_fingerprint"], values["signal_as_of_utc"])
        observed = str(
            record.get("evaluation_as_of_utc")
            or record.get("recorded_at_utc")
            or ""
        )
        existing = compact.get(key)
        if existing is None:
            compact[key] = {
                "schema": "xsp.fundamental-pressure.v1",
                "authority": "observation_only",
                "order_authority": "none",
                **values,
                "first_checkpoint_utc": observed,
                "last_checkpoint_utc": observed,
                "checkpoint_count": 1,
            }
            continue
        if any(existing[field] != values[field] for field in PRESSURE_FIELDS):
            raise ValueError("conflicting pressure values for one snapshot")
        existing["last_checkpoint_utc"] = observed
        existing["checkpoint_count"] = int(existing["checkpoint_count"]) + 1
    return tuple(
        sorted(
            compact.values(),
            key=lambda row: (
                str(row["signal_as_of_utc"]),
                str(row["snapshot_fingerprint"]),
            ),
        )
    )


def pressure_rows_from_news(
    snapshots: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Project every publication through the canonical causal-pressure owner."""

    ordered = sorted(
        snapshots,
        key=lambda row: str(row.get("snapshot_as_of_utc") or ""),
    )
    output = []
    prefix: list[Mapping[str, object]] = []
    for snapshot in ordered:
        prefix.append(snapshot)
        observed = datetime.fromisoformat(
            str(snapshot["snapshot_as_of_utc"]).replace("Z", "+00:00")
        )
        pressure = xsp_fundamental_context_at(
            tuple(prefix),
            decision_at=observed,
        )
        if pressure.get("snapshot_fingerprint") is None:
            continue
        output.append(
            {
                "schema": "xsp.fundamental-pressure.v1",
                "authority": "observation_only",
                "order_authority": "none",
                **{
                    field: pressure.get(field)
                    for field in PRESSURE_FIELDS
                },
                "publication_id": snapshot.get("publication_id"),
                "first_checkpoint_utc": None,
                "last_checkpoint_utc": None,
                "checkpoint_count": 0,
            }
        )
    return tuple(output)


def merge_pressure_rows(
    publications: Iterable[Mapping[str, object]],
    checkpoints: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    merged = {
        (row["snapshot_fingerprint"], row["signal_as_of_utc"]): dict(row)
        for row in publications
    }
    for checkpoint in checkpoints:
        key = (
            checkpoint["snapshot_fingerprint"],
            checkpoint["signal_as_of_utc"],
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(checkpoint)
            continue
        if any(
            existing[field] != checkpoint[field]
            for field in PRESSURE_FIELDS
        ):
            raise ValueError("news and checkpoint pressure values conflict")
        existing.update(
            {
                "first_checkpoint_utc": checkpoint[
                    "first_checkpoint_utc"
                ],
                "last_checkpoint_utc": checkpoint["last_checkpoint_utc"],
                "checkpoint_count": checkpoint["checkpoint_count"],
            }
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda row: (
                str(row["signal_as_of_utc"]),
                str(row["snapshot_fingerprint"]),
            ),
        )
    )


def _rsync(remote: str, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/rsync",
            "-a",
            "-e",
            (
                "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8 "
                "-o ServerAliveInterval=10 -o ServerAliveCountMax=2"
            ),
            f"{remote}:{source}",
            str(destination),
        ],
        check=True,
        timeout=60,
    )


def mirror_pressure(
    *,
    remote: str,
    output_dir: Path,
    remote_home: str = "/home/x",
) -> dict[str, object]:
    if not REMOTE_RE.fullmatch(remote):
        raise ValueError("invalid remote host")
    output_dir = output_dir.expanduser().resolve()
    calibration = output_dir / "xsp_live_calibration.jsonl"
    history = output_dir / "news-history"
    history.mkdir(parents=True, exist_ok=True)

    incoming = calibration.with_suffix(".jsonl.incoming")
    _rsync(
        remote,
        f"{remote_home}/Desktop/py/tradebot/db/calibration/"
        "xsp_live_calibration.jsonl",
        incoming,
    )
    os.replace(incoming, calibration)
    _rsync(
        remote,
        f"{remote_home}/.local/state/tradebot/news/history/",
        history,
    )

    snapshots = []
    for path in sorted(history.glob("????-??.jsonl")):
        snapshots.extend(load_news_history(path))
    rows = merge_pressure_rows(
        pressure_rows_from_news(snapshots),
        compact_pressure_rows(_records(calibration)),
    )
    ledger = output_dir / "xsp_pressure.jsonl"
    encoded = b"".join(
        json.dumps(
            row,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
        for row in rows
    )
    _write_atomic(ledger, encoded)
    manifest = {
        "schema": "xsp.pressure-mirror-manifest.v1",
        "authority": "read_only_evidence_mirror",
        "order_authority": "none",
        "remote": remote,
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "path": str(calibration),
            "sha256": _sha256(calibration),
        },
        "compact_pressure": {
            "path": str(ledger),
            "rows": len(rows),
            "sha256": _sha256(ledger),
        },
        "news_history": {
            path.name: _sha256(path)
            for path in sorted(history.glob("????-??.jsonl"))
        },
    }
    manifest_path = output_dir / "MANIFEST.json"
    _write_atomic(
        manifest_path,
        json.dumps(
            manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="q")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "~/.local/state/tradebot/xsp-evidence/q-mirror"
        ).expanduser(),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            mirror_pressure(
                remote=args.remote,
                output_dir=args.output_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
