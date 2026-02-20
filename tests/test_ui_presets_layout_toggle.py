from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tradebot.ui.bot import BotScreen


class _TableStub:
    def __init__(self) -> None:
        self.display = True
        self.styles = SimpleNamespace(height=None)


def _screen() -> BotScreen:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return BotScreen(client=SimpleNamespace(), refresh_sec=1.0)


def test_bot_bindings_include_preset_toggle_hotkey_p() -> None:
    assert ("p", "toggle_presets", "Presets") in BotScreen.BINDINGS


def test_apply_presets_layout_moves_space_to_logs_when_hidden() -> None:
    screen = _screen()
    screen._presets_table = _TableStub()
    screen._orders_table = _TableStub()
    screen._logs_table = _TableStub()

    screen._presets_visible = True
    screen._apply_presets_layout()
    assert screen._presets_table.display is True
    assert screen._orders_table.styles.height == "1fr"
    assert screen._logs_table.styles.height == "1fr"

    screen._presets_visible = False
    screen._apply_presets_layout()
    assert screen._presets_table.display is False
    assert screen._orders_table.styles.height == "1fr"
    assert screen._logs_table.styles.height == "3fr"
