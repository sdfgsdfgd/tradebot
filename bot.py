#!/usr/bin/env python3
"""Launch the tradebot TUI with sane defaults."""
from __future__ import annotations

import os

from tradebot.main import main


if __name__ == "__main__":
    os.environ.setdefault("IBKR_CLIENT_ID", "0")
    main()
