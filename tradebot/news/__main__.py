"""Module entrypoint for ``python -m tradebot.news``."""

from .pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
