from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.ui.bot import BotScreen


class _TableStub:
    def __init__(self) -> None:
        self.rows: list[tuple[tuple, dict]] = []
        self.cursor_coordinate = (0, 0)

    def clear(self, columns: bool = False) -> None:
        self.rows = []

    def add_column(self, *_args, **_kwargs) -> None:
        return

    def add_row(self, *values, **kwargs) -> None:
        self.rows.append((values, kwargs))

    def refresh(self, repaint: bool = False) -> None:
        return


def _screen() -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    screen = BotScreen(client=SimpleNamespace(), refresh_sec=1.0)
    screen._presets_table = _TableStub()
    screen._orders_table = _TableStub()
    screen._logs_table = _TableStub()
    screen._instances = []
    screen._instance_live_total_by_id = {}
    screen._render_status = lambda *args, **kwargs: None
    screen._set_status = lambda *args, **kwargs: None
    screen._sync_row_marker = lambda *args, **kwargs: None
    return screen


def _labels(screen: BotScreen) -> list[str]:
    labels: list[str] = []
    for values, _kwargs in screen._presets_table.rows:
        first = values[0]
        labels.append(first.plain if hasattr(first, "plain") else str(first))
    return labels


def test_tqqq_preset_track_headers_show_loaded_crown_versions_and_clean_leaf_titles() -> None:
    screen = _screen()
    screen._load_leaderboard()

    labels = _labels(screen)
    assert labels[0] == "▾ TQQQ"
    assert labels[1] == "  ▾ TQQQ - Spot"
    assert "    ▸ LF v34" in labels
    assert "    ▸ HF v38" in labels
    assert not any("HF?" in label for label in labels)

    screen._preset_expanded.add("contract:TQQQ|spot|track:HF")
    screen._rebuild_presets_table()
    labels = _labels(screen)

    hf_leaf = next(label for label in labels if "ASYMMETRIC CRASH PREARM SOVEREIGNTY" in label)
    assert "floor=" not in hf_leaf
    assert " 2025=" not in hf_leaf
