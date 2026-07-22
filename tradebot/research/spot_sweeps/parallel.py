"""Subprocess orchestration for spot research axes."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ...backtest.sweep_parallel import _run_parallel_worker_specs


def _run_axis_subprocess_plan(
    *,
    label: str,
    axes: tuple[str, ...],
    jobs: int,
    base_cli: list[str],
    axis_jobs_resolver,
    write_milestones: bool,
    tmp_prefix: str,
) -> dict[str, dict]:
    jobs_eff = min(int(jobs), len(axes))
    print(f"{label}: jobs={jobs_eff} axes={len(axes)}", flush=True)

    milestone_paths: dict[str, Path] = {}
    milestone_payloads: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        tmp_root = Path(tmpdir)
        specs: list[tuple[str, list[str]]] = []
        for axis_name in axes:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "tradebot.backtest",
                "spot",
                *base_cli,
                "--axis",
                str(axis_name),
                "--jobs",
                str(int(axis_jobs_resolver(axis_name))),
            ]
            if bool(write_milestones):
                out_path = tmp_root / f"milestones_{axis_name}.json"
                milestone_paths[axis_name] = out_path
                cmd += ["--milestones-out", str(out_path)]
            specs.append((str(axis_name), cmd))

        _run_parallel_worker_specs(
            specs=specs,
            jobs=jobs_eff,
            capture_error=f"Failed to capture {label} worker stdout.",
            failure_label=f"{label} axis",
        )

        if bool(write_milestones):
            for axis_name, out_path in milestone_paths.items():
                if not out_path.exists():
                    continue
                try:
                    payload = json.loads(out_path.read_text())
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    milestone_payloads[axis_name] = payload

    return milestone_payloads
