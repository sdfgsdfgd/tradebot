"""One compact, domain-first footer for every Tradebot screen."""

from textual.widgets import Footer


class TradebotFooter(Footer):
    """Show only declared product actions; keep the generic palette implicit."""

    def __init__(self) -> None:
        super().__init__(show_command_palette=False, compact=True)
