"""Parallel worker + progress helpers shared by spot sweep tooling."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    return [arg for arg in argv if arg != flag]


def _strip_flag_with_value(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == flag:
            idx += 2
            continue
        if arg.startswith(flag + "="):
            idx += 1
            continue
        out.append(arg)
        idx += 1
    return out


def _strip_flags(
    argv: list[str],
    *,
    flags: tuple[str, ...] = (),
    flags_with_values: tuple[str, ...] = (),
) -> list[str]:
    out = list(argv)
    for flag in flags:
        out = _strip_flag(out, str(flag))
    for flag in flags_with_values:
        out = _strip_flag_with_value(out, str(flag))
    return out


def _parse_worker_shard(raw_worker: object, raw_workers: object, *, label: str) -> tuple[int, int]:
    try:
        worker_id = int(raw_worker) if raw_worker is not None else 0
    except (TypeError, ValueError):
        worker_id = 0
    try:
        workers = int(raw_workers) if raw_workers is not None else 1
    except (TypeError, ValueError):
        workers = 1
    workers = max(1, int(workers))
    worker_id = max(0, int(worker_id))
    if worker_id >= workers:
        raise SystemExit(f"Invalid {label} worker shard: worker={worker_id} workers={workers} (worker must be < workers).")
    return worker_id, workers


def _pump_subprocess_output(prefix: str, stream) -> None:
    for line in iter(stream.readline, ""):
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _run_parallel_worker_specs(
    *,
    specs: list[tuple[str, list[str]]],
    jobs: int,
    capture_error: str,
    failure_label: str,
    status_heartbeat_sec: float = 30.0,
    worker_status_probe=None,
    max_stale_retries: int = 1,
    stale_terminate_grace_sec: float = 5.0,
) -> None:
    if not specs:
        return
    jobs_eff = max(1, min(int(jobs), len(specs)))
    pending = list(specs)
    running: list[tuple[str, list[str], subprocess.Popen, threading.Thread, float]] = []
    failures: list[tuple[str, int]] = []
    stale_retries_by_worker: dict[str, int] = {}
    heartbeat_every = max(0.0, float(status_heartbeat_sec or 0.0))
    heartbeat_last = time.perf_counter()

    def _stop_worker(proc: subprocess.Popen, thread: threading.Thread, *, timeout: float) -> None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=max(0.5, float(timeout)))
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=max(1.0, float(timeout)))
            except Exception:
                pass
        try:
            thread.join(timeout=max(0.5, float(timeout)))
        except Exception:
            pass

    while pending or running:
        while pending and len(running) < jobs_eff and not failures:
            worker_name, cmd = pending.pop(0)
            print(f"START {worker_name}", flush=True)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if proc.stdout is None:
                raise RuntimeError(str(capture_error))
            t = threading.Thread(target=_pump_subprocess_output, args=(worker_name, proc.stdout), daemon=True)
            t.start()
            running.append((worker_name, list(cmd), proc, t, time.perf_counter()))

        finished = False
        for idx, (worker_name, _cmd, proc, t, started_at) in enumerate(running):
            rc = proc.poll()
            if rc is None:
                continue
            finished = True
            elapsed = time.perf_counter() - float(started_at)
            print(f"DONE  {worker_name} exit={rc} elapsed={elapsed:0.1f}s", flush=True)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
            if rc != 0:
                failures.append((worker_name, int(rc)))
            running.pop(idx)
            break

        if failures:
            for _worker_name, _cmd, proc, t, _started_at in running:
                _stop_worker(proc, t, timeout=5.0)
            break

        if not finished:
            if heartbeat_every > 0.0 and running:
                now = time.perf_counter()
                if (now - heartbeat_last) >= heartbeat_every:
                    in_flight = ", ".join(
                        f"{worker_name}:{now - float(started_at):0.1f}s"
                        for worker_name, _cmd, _proc, _t, started_at in running
                    )
                    print(
                        f"RUNNING workers={len(running)} pending={len(pending)} [{in_flight}]",
                        flush=True,
                    )
                    stale_names: set[str] = set()
                    if callable(worker_status_probe):
                        probe_result = None
                        try:
                            probe_result = worker_status_probe(
                                [
                                    (str(worker_name), float(now - float(started_at)))
                                    for worker_name, _cmd, _proc, _t, started_at in running
                                ],
                                int(len(pending)),
                            )
                        except Exception:
                            probe_result = None
                        if isinstance(probe_result, dict):
                            line = str(probe_result.get("line") or "").strip()
                            if line:
                                print(line, flush=True)
                            raw_stale = probe_result.get("stale") or ()
                            try:
                                stale_names = {
                                    str(name).strip()
                                    for name in raw_stale
                                    if str(name).strip()
                                }
                            except Exception:
                                stale_names = set()
                    if stale_names:
                        running_by_name = {str(name): idx for idx, (name, _cmd, _proc, _t, _started_at) in enumerate(running)}
                        for stale_name in sorted(stale_names):
                            idx = running_by_name.get(str(stale_name))
                            if idx is None:
                                continue
                            worker_name, cmd, proc, t, started_at = running[idx]
                            elapsed = max(0.0, float(now - float(started_at)))
                            retries = int(stale_retries_by_worker.get(str(worker_name), 0))
                            if retries >= int(max(0, int(max_stale_retries))):
                                print(
                                    f"STALE {worker_name} elapsed={elapsed:0.1f}s retries={retries} (giving up)",
                                    flush=True,
                                )
                                _stop_worker(proc, t, timeout=float(stale_terminate_grace_sec))
                                failures.append((str(worker_name), 124))
                                continue
                            print(
                                f"STALE {worker_name} elapsed={elapsed:0.1f}s retries={retries} -> recycle",
                                flush=True,
                            )
                            _stop_worker(proc, t, timeout=float(stale_terminate_grace_sec))
                            stale_retries_by_worker[str(worker_name)] = int(retries + 1)
                            pending.append((str(worker_name), list(cmd)))
                            running[idx] = ("", [], proc, t, now)
                        running = [entry for entry in running if entry[0]]
                    heartbeat_last = now
            time.sleep(0.05)

    if failures:
        worker_name, rc = failures[0]
        raise SystemExit(f"{failure_label} failed: {worker_name} (exit={rc})")


def _run_parallel_json_worker_plan(
    *,
    jobs_eff: int,
    tmp_prefix: str,
    worker_tag: str,
    out_prefix: str,
    build_cmd: Callable[[int, int, Path], list[str]],
    capture_error: str,
    failure_label: str,
    missing_label: str,
    invalid_label: str,
    status_heartbeat_sec: float = 30.0,
    worker_status_probe=None,
    max_stale_retries: int = 1,
    stale_terminate_grace_sec: float = 5.0,
) -> dict[int, dict]:
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        tmp_root = Path(tmpdir)
        specs: list[tuple[str, list[str]]] = []
        out_paths: dict[int, Path] = {}
        for worker_id in range(max(1, int(jobs_eff))):
            out_path = tmp_root / f"{out_prefix}_{worker_id}.json"
            out_paths[worker_id] = out_path
            specs.append((f"{worker_tag}:{worker_id}", list(build_cmd(int(worker_id), int(jobs_eff), out_path))))

        _run_parallel_worker_specs(
            specs=specs,
            jobs=int(jobs_eff),
            capture_error=str(capture_error),
            failure_label=str(failure_label),
            status_heartbeat_sec=float(status_heartbeat_sec),
            worker_status_probe=worker_status_probe,
            max_stale_retries=int(max_stale_retries),
            stale_terminate_grace_sec=float(stale_terminate_grace_sec),
        )

        payloads: dict[int, dict] = {}
        for worker_id, out_path in out_paths.items():
            if not out_path.exists():
                raise SystemExit(f"Missing {missing_label} output: {worker_tag}:{worker_id} ({out_path})")
            try:
                payload = json.loads(out_path.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid {invalid_label} output JSON: {worker_tag}:{worker_id} ({out_path})") from exc
            if isinstance(payload, dict):
                payloads[int(worker_id)] = payload
    return payloads


def _run_parallel_stage_kernel(
    *,
    stage_label: str,
    jobs: int,
    total: int,
    default_jobs: int,
    offline: bool,
    offline_error: str,
    tmp_prefix: str,
    worker_tag: str,
    out_prefix: str,
    build_cmd: Callable[[int, int, Path], list[str]],
    capture_error: str,
    failure_label: str,
    missing_label: str,
    invalid_label: str,
    status_heartbeat_sec: float = 30.0,
    worker_status_probe=None,
    max_stale_retries: int = 1,
    stale_terminate_grace_sec: float = 5.0,
) -> tuple[int, dict[int, dict]]:
    if not bool(offline):
        raise SystemExit(str(offline_error))
    jobs_eff = min(int(jobs), int(default_jobs), int(total)) if int(total) > 0 else 1
    jobs_eff = max(1, int(jobs_eff))
    print(f"{stage_label} parallel: workers={jobs_eff} total={int(total)}", flush=True)
    payloads = _run_parallel_json_worker_plan(
        jobs_eff=int(jobs_eff),
        tmp_prefix=str(tmp_prefix),
        worker_tag=str(worker_tag),
        out_prefix=str(out_prefix),
        build_cmd=build_cmd,
        capture_error=str(capture_error),
        failure_label=str(failure_label),
        missing_label=str(missing_label),
        invalid_label=str(invalid_label),
        status_heartbeat_sec=float(status_heartbeat_sec),
        worker_status_probe=worker_status_probe,
        max_stale_retries=int(max_stale_retries),
        stale_terminate_grace_sec=float(stale_terminate_grace_sec),
    )
    return int(jobs_eff), payloads


def _collect_parallel_payload_records(
    *,
    payloads: dict[int, dict],
    records_key: str = "records",
    tested_key: str = "tested",
    decode_record=None,
    on_record=None,
    dedupe_key=None,
) -> int:
    tested_total = 0
    seen: set[str] | None = set() if callable(dedupe_key) else None
    for payload in payloads.values():
        if not isinstance(payload, dict):
            continue
        tested_total += int(payload.get(tested_key) or 0)
        records = payload.get(records_key) or []
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_obj = decode_record(rec) if callable(decode_record) else rec
            if rec_obj is None:
                continue
            if seen is not None:
                try:
                    rec_key = dedupe_key(rec_obj)
                except Exception:
                    rec_key = None
                if rec_key is not None:
                    rec_key_s = str(rec_key)
                    if rec_key_s in seen:
                        continue
                    seen.add(rec_key_s)
            if callable(on_record):
                on_record(rec_obj)
    return int(tested_total)


def _progress_snapshot(
    *,
    tested: int,
    total: int | None,
    started_at: float,
) -> tuple[float, float, float, float, float]:
    elapsed = max(0.0, float(time.perf_counter()) - float(started_at))
    rate = (float(tested) / elapsed) if elapsed > 0 else 0.0
    total_i = int(total) if total is not None else 0
    remaining = max(0, int(total_i) - int(tested)) if total is not None else 0
    eta_sec = (float(remaining) / rate) if rate > 0 else 0.0
    pct = ((float(tested) / float(total_i)) * 100.0) if total is not None and total_i > 0 else 0.0
    return elapsed, rate, float(remaining), float(eta_sec), float(pct)


def _progress_line(
    *,
    label: str,
    tested: int,
    total: int | None,
    kept: int,
    started_at: float,
    rate_unit: str = "s",
) -> str:
    elapsed, rate, _remaining, eta_sec, pct = _progress_snapshot(
        tested=int(tested),
        total=(int(total) if total is not None else None),
        started_at=float(started_at),
    )
    total_i = int(total) if total is not None else 0
    if total is not None and total_i > 0:
        return (
            f"{label} {int(tested)}/{total_i} ({pct:0.1f}%) kept={int(kept)} "
            f"elapsed={elapsed:0.1f}s eta={eta_sec/60.0:0.1f}m rate={rate:0.2f}/{rate_unit}"
        )
    return (
        f"{label} tested={int(tested)} kept={int(kept)} "
        f"elapsed={elapsed:0.1f}s rate={rate:0.2f}/{rate_unit}"
    )
