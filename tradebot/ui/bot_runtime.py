"""Navigation owner for the durable portfolio cockpit."""

from __future__ import annotations

from textual.app import App

from .bot import BotScreen


class BotRuntime:
    """Keep one cockpit screen while q-owned runs outlive every UI session."""

    _SCREEN_NAME = "bot-runtime"

    def __init__(self, client: object | None, refresh_sec: float) -> None:
        self._screen = BotScreen(client, refresh_sec)
        self._installed = False

    @property
    def screen(self) -> BotScreen:
        return self._screen

    def install(self, app: App) -> None:
        if self._installed:
            return
        app.install_screen(self._screen, self._SCREEN_NAME)
        self._installed = True

    def show(self, app: App) -> None:
        self.install(app)
        if app.screen is self._screen:
            return
        app.push_screen(self._SCREEN_NAME)

    def hide(self, app: App) -> None:
        if app.screen is self._screen:
            app.pop_screen()

    def toggle(self, app: App) -> None:
        if app.screen is self._screen:
            app.pop_screen()
            return
        self.show(app)
