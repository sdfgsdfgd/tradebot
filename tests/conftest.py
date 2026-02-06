from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


try:
    importlib.import_module("eventkit")
except ModuleNotFoundError:
    try:
        eventkit_mod = importlib.import_module("EventKit")
    except ModuleNotFoundError:
        eventkit_mod = None
    if eventkit_mod is not None:
        sys.modules.setdefault("eventkit", eventkit_mod)
