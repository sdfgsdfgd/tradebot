"""One local-or-q endpoint for the durable portfolio cockpit."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import select
import shlex
import socket
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

from .runs import _identity
from .strategies import build_live_portfolio_cockpit


_REMOTE_ROOT = "/home/x/Desktop/py/tradebot"
_REMOTE_PYTHON = "/home/x/.local/share/tradebot/venv/bin/python"
_HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
_REMOTE_TIMEOUT_SEC = 45.0


class _RemoteTransportError(RuntimeError):
    pass


class LivePortfolioEndpoint:
    """Address q's portfolio owner without a local-state fallback."""

    def __init__(
        self,
        *,
        repository_root: Path,
        host: str | None,
        remote_root: str = _REMOTE_ROOT,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.host = host
        self.remote_root = remote_root
        self._owner = (
            build_live_portfolio_cockpit(self.repository_root) if host is None else None
        )
        self._remote_lock = threading.Lock()
        self._remote_process: subprocess.Popen[str] | None = None
        self._request_sequence = 0
        self._last_view_id: str | None = None
        self._last_snapshot_id: str | None = None
        self._last_timeline_id: str | None = None
        self._last_traces_id: str | None = None
        if host is not None and not _HOST_RE.fullmatch(host):
            raise ValueError("live portfolio host is invalid")

    @classmethod
    def default(cls, repository_root: Path) -> LivePortfolioEndpoint:
        configured = os.environ.get("TRADEBOT_LIVE_HOST")
        if configured is not None:
            normalized = configured.strip()
            host = None if normalized.lower() == "local" else normalized
        else:
            host = None if socket.gethostname().split(".", 1)[0].upper() == "Q" else "q"
        if host == "":
            raise ValueError("TRADEBOT_LIVE_HOST cannot be empty")
        return cls(repository_root=repository_root, host=host)

    def _remote_command(self) -> str:
        assert self.host is not None
        return " ".join(
            (
                f"cd {shlex.quote(self.remote_root)}",
                "&&",
                "exec",
                shlex.quote(_REMOTE_PYTHON),
                "-m",
                "tradebot.live.portfolio_endpoint",
                "serve",
            )
        )

    def _start_remote(self) -> subprocess.Popen[str]:
        assert self.host is not None
        process = subprocess.Popen(
            ["ssh", self.host, self._remote_command()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._remote_process = process
        return process

    def _ensure_remote(self) -> subprocess.Popen[str]:
        process = self._remote_process
        if process is None or process.poll() is not None:
            process = self._start_remote()
        return process

    @staticmethod
    def _read_remote_line(process: subprocess.Popen[str]) -> str:
        if process.stdout is None:
            raise _RemoteTransportError("q portfolio endpoint has no stdout")
        try:
            readable, _, _ = select.select(
                [process.stdout],
                [],
                [],
                _REMOTE_TIMEOUT_SEC,
            )
        except (OSError, ValueError) as exc:
            raise _RemoteTransportError(str(exc)) from exc
        if not readable:
            raise _RemoteTransportError("q portfolio endpoint timed out")
        line = process.stdout.readline()
        if not line:
            detail = ""
            if process.poll() is not None and process.stderr is not None:
                detail = process.stderr.read().strip()
            raise _RemoteTransportError(
                detail or "q portfolio endpoint closed unexpectedly"
            )
        return line

    def _exchange_remote(self, request: Mapping[str, object]) -> dict[str, object]:
        try:
            process = self._ensure_remote()
            if process.stdin is None:
                raise _RemoteTransportError("q portfolio endpoint has no stdin")
            process.stdin.write(
                json.dumps(
                    request, allow_nan=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            )
            process.stdin.flush()
            raw = self._read_remote_line(process)
        except _RemoteTransportError:
            raise
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise _RemoteTransportError(str(exc)) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _RemoteTransportError(
                "q portfolio endpoint returned invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise _RemoteTransportError("q portfolio endpoint returned a non-object")
        if value.get("request_id") != request.get("request_id"):
            raise _RemoteTransportError(
                "q portfolio endpoint response identity mismatch"
            )
        if value.get("ok") is not True:
            raise RuntimeError(str(value.get("error") or "q portfolio endpoint failed"))
        result = value.get("result")
        if not isinstance(result, Mapping):
            raise _RemoteTransportError(
                "q portfolio endpoint returned an invalid result"
            )
        return dict(result)

    def _stop_remote(self) -> None:
        process, self._remote_process = self._remote_process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _remote(self, operation: str, **payload: object) -> dict[str, object]:
        retry_transport = operation == "view"
        with self._remote_lock:
            for attempt in range(2 if retry_transport else 1):
                self._request_sequence += 1
                request = {
                    "request_id": self._request_sequence,
                    "operation": operation,
                    **payload,
                }
                try:
                    return self._exchange_remote(request)
                except _RemoteTransportError as exc:
                    self._stop_remote()
                    if attempt == 0 and retry_transport:
                        continue
                    raise RuntimeError(f"q portfolio endpoint failed: {exc}") from exc
        raise RuntimeError("q portfolio endpoint failed")

    def _local_view(
        self,
        *,
        limit: int,
        trace_limit: int,
        previous_view_id: str | None,
    ) -> dict[str, object]:
        assert self._owner is not None
        view = self._owner.view(limit=limit, trace_limit=trace_limit)
        snapshot = view["snapshot"]
        timeline = view["timeline"]
        traces = view.get("traces", [])
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(timeline, list)
            or not isinstance(traces, list)
        ):
            raise RuntimeError("live portfolio owner returned an invalid view")
        snapshot_id = str(snapshot.get("snapshot_id") or _identity(snapshot))
        timeline_id = _identity(timeline)
        traces_id = _identity(traces)
        view_id = _identity(
            {
                "snapshot_id": snapshot_id,
                "timeline_id": timeline_id,
                "traces_id": traces_id,
            }
        )
        previous_is_last = previous_view_id == self._last_view_id
        result: dict[str, object] = {
            "view_id": view_id,
            "snapshot_id": snapshot_id,
            "timeline_id": timeline_id,
            "traces_id": traces_id,
            "unchanged": previous_view_id == view_id,
        }
        if previous_view_id != view_id:
            if not previous_is_last or snapshot_id != self._last_snapshot_id:
                result["snapshot"] = dict(snapshot)
            if not previous_is_last or timeline_id != self._last_timeline_id:
                result["timeline"] = timeline
            if not previous_is_last or traces_id != self._last_traces_id:
                result["traces"] = traces
        self._last_view_id = view_id
        self._last_snapshot_id = snapshot_id
        self._last_timeline_id = timeline_id
        self._last_traces_id = traces_id
        return result

    def view(
        self,
        *,
        limit: int = 1_000,
        trace_limit: int | None = None,
        previous_view_id: str | None = None,
    ) -> dict[str, object]:
        bounded_trace_limit = int(limit if trace_limit is None else trace_limit)
        if self._owner is not None:
            return self._local_view(
                limit=limit,
                trace_limit=bounded_trace_limit,
                previous_view_id=previous_view_id,
            )
        return self._remote(
            "view",
            limit=int(limit),
            trace_limit=bounded_trace_limit,
            previous_view_id=previous_view_id,
        )

    def snapshot(self) -> dict[str, object]:
        return dict(self.view(limit=1, trace_limit=1)["snapshot"])

    def request_control(self, sleeve_id: str, action: str) -> dict[str, object]:
        if self._owner is not None:
            return self._owner.request_control(sleeve_id, action)
        return self._remote("control", sleeve_id=sleeve_id, action=action)

    def commission(self, candidate_id: str) -> dict[str, object]:
        if self._owner is not None:
            return self._owner.commission(candidate_id)
        return self._remote("commission", candidate_id=candidate_id)

    def close(self) -> None:
        with self._remote_lock:
            self._stop_remote()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    view = subparsers.add_parser("view")
    view.add_argument("--limit", type=int, default=1_000)
    view.add_argument("--trace-limit", type=int)
    control = subparsers.add_parser("control")
    control.add_argument("sleeve_id")
    control.add_argument("action")
    commission = subparsers.add_parser("commission")
    commission.add_argument("candidate_id")
    subparsers.add_parser("serve")
    return parser


def _serve(endpoint: LivePortfolioEndpoint) -> None:
    for raw in sys.stdin:
        request_id: object = None
        try:
            request = json.loads(raw)
            if not isinstance(request, Mapping):
                raise ValueError("request must be an object")
            request_id = request.get("request_id")
            operation = str(request.get("operation") or "")
            if operation == "view":
                previous = request.get("previous_view_id")
                result = endpoint.view(
                    limit=int(request.get("limit") or 1_000),
                    trace_limit=int(
                        request.get("trace_limit") or request.get("limit") or 1_000
                    ),
                    previous_view_id=str(previous) if previous else None,
                )
            elif operation == "control":
                result = endpoint.request_control(
                    str(request.get("sleeve_id") or ""),
                    str(request.get("action") or ""),
                )
            elif operation == "commission":
                result = endpoint.commission(str(request.get("candidate_id") or ""))
            else:
                raise ValueError("unknown portfolio operation")
            response = {"request_id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {"request_id": request_id, "ok": False, "error": str(exc)}
        print(
            json.dumps(
                response, allow_nan=False, separators=(",", ":"), sort_keys=True
            ),
            flush=True,
        )


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        endpoint = LivePortfolioEndpoint(repository_root=Path.cwd(), host=None)
        _serve(endpoint)
        return
    owner = build_live_portfolio_cockpit(Path.cwd())
    if args.command == "view":
        result = owner.view(limit=args.limit, trace_limit=args.trace_limit)
    elif args.command == "control":
        result = owner.request_control(args.sleeve_id, args.action)
    else:
        result = owner.commission(args.candidate_id)
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
