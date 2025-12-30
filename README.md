# tradebot

Minimal IBKR TUI for live positions, built for a fast read‑only overview and a simple detail drill‑down.

![tradebot TUI](docs/tradebot-demo.png)

## Features
- Top bar with **NQ/ES/YM futures** plus **QQQ/TQQQ proxy**, session label, and live/delayed tags.
- Positions grouped by **Options / Stocks / Futures / Futures Options (FOP)**.
- **Unrealized / Realized** P&L, **Daily P&L**, and **Net Liquidation** (with last‑update timestamp and an estimate).
- Event‑driven UI refresh (redraws only on data changes, throttled to 250ms).
- **Details screen** per position with contract info, bid/ask/last, and **mid** (orange).

## Architecture
- `tradebot/config.py` — runtime config (host/port/client id/refresh interval).
- `tradebot/client.py` — ib_insync wrapper (portfolio, market data, PnL, net liq cache).
- `tradebot/ui.py` — Textual TUI + detail screen.
- `tradebot/store.py` — in‑memory portfolio snapshot.
- `tradebot/main.py` — entrypoint.
- `tradebot/gpt/` — reserved for future GPT workflows.

## Usage
1) Run **IB Gateway** or **TWS** with the API enabled (socket 4001 by default).
2) Launch:
   ```bash
   ./bot.py
   ```

Optional env vars:
- `IBKR_HOST` (default `127.0.0.1`)
- `IBKR_PORT` (default `4001`)
- `IBKR_CLIENT_ID` (default `0`)
- `IBKR_ACCOUNT` (optional, to pin an account)

## Controls
- **Arrow keys** — navigate rows
- **Enter** — open details screen
- **b** / **Esc** — back
- **r** — hard refresh (resubscribe)
- **q** — quit

## Notes
- `[L]` = live data, `[D]` = delayed data.
- If you don’t subscribe to real‑time market data, quotes may be delayed.
- `Net Liq` is provided by IBKR; the `~` estimate just interpolates between IBKR updates.
