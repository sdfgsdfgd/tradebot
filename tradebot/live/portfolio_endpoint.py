"""One local-or-q endpoint for the durable portfolio cockpit."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .strategies import build_live_portfolio_cockpit


_REMOTE_ROOT = "/home/x/Desktop/py/tradebot"
_REMOTE_PYTHON = "/home/x/.local/share/tradebot/venv/bin/python"
_HOST_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")


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
            build_live_portfolio_cockpit(self.repository_root)
            if host is None
            else None
        )
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

    def _remote(self, *arguments: str) -> dict[str, object]:
        assert self.host is not None
        command = " ".join(
            (
                f"cd {shlex.quote(self.remote_root)}",
                "&&",
                shlex.quote(_REMOTE_PYTHON),
                "-m",
                "tradebot.live.portfolio_endpoint",
                *(shlex.quote(argument) for argument in arguments),
            )
        )
        completed = subprocess.run(
            ["ssh", self.host, command],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"q portfolio endpoint failed: {detail}")
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("q portfolio endpoint returned invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise RuntimeError("q portfolio endpoint returned a non-object")
        return dict(value)

    def view(self, *, limit: int = 1_000) -> dict[str, object]:
        if self._owner is not None:
            return self._owner.view(limit=limit)
        return self._remote("view", "--limit", str(limit))

    def snapshot(self) -> dict[str, object]:
        return dict(self.view(limit=1)["snapshot"])

    def request_control(self, sleeve_id: str, action: str) -> dict[str, object]:
        if self._owner is not None:
            return self._owner.request_control(sleeve_id, action)
        return self._remote("control", sleeve_id, action)

    def commission(self, candidate_id: str) -> dict[str, object]:
        if self._owner is not None:
            return self._owner.commission(candidate_id)
        return self._remote("commission", candidate_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    view = subparsers.add_parser("view")
    view.add_argument("--limit", type=int, default=1_000)
    control = subparsers.add_parser("control")
    control.add_argument("sleeve_id")
    control.add_argument("action")
    commission = subparsers.add_parser("commission")
    commission.add_argument("candidate_id")
    return parser


def main() -> None:
    args = _parser().parse_args()
    owner = build_live_portfolio_cockpit(Path.cwd())
    if args.command == "view":
        result = owner.view(limit=args.limit)
    elif args.command == "control":
        result = owner.request_control(args.sleeve_id, args.action)
    else:
        result = owner.commission(args.candidate_id)
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
