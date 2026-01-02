# Backtest Engine (Synthetic Options)

This package provides a minimal backtest runner that builds **synthetic option prices** from underlying bars and evaluates a single strategy per run. The code is structured to be reusable for a future live trading engine.

## Full config example (all parameters)
```json
{
  "backtest": {
    "start": "2025-01-01",
    "end": "2025-12-31",
    "bar_size": "1 hour",
    "use_rth": false,
    "starting_cash": 100000,
    "risk_free_rate": 0.02,
    "cache_dir": "db",
    "calibration_dir": "db/calibration",
    "calibrate": false,
    "output_dir": "backtests/out"
  },
  "strategy": {
    "name": "credit_spread",
    "symbol": "SLV",
    "right": "CALL",
    "entry_days": [],
    "dte": 30,
    "otm_pct": -1.0,
    "width_pct": 2.0,
    "profit_target": 3.0,
    "stop_loss": 0.35,
    "exit_dte": 30,
    "quantity": 1,
    "min_credit": 0.01,
    "ema_preset": "9/21",
    "filters": {
      "rv_min": 0.15,
      "rv_max": 0.60,
      "ema_spread_min_pct": 0.05,
      "ema_slope_min_pct": 0.01,
      "entry_start_hour": 10,
      "entry_end_hour": 15,
      "skip_first_bars": 2,
      "cooldown_bars": 4
    }
  },
  "synthetic": {
    "rv_lookback": 60,
    "rv_ewma_lambda": 0.94,
    "iv_risk_premium": 1.2,
    "iv_floor": 0.05,
    "term_slope": 0.02,
    "skew": -0.25,
    "min_spread_pct": 0.1
  }
}
```
Notes:
- `entry_days: []` defaults to all weekdays.
- `otm_pct` can be negative for ITM (e.g., `-1.0` = 1% ITM).
- `synthetic` is optional; defaults are used if omitted.

## What it does
- Pulls **hourly underlying bars** via IBKR and caches them on disk.
- Builds **synthetic option chains** using EWMA realized volatility + skew.
- Prices options with **Black-76** (futures) or **Black-Scholes** (equities).
- Simulates credit spreads with **mid-edge** entry/exit fills.
- Outputs **CLI summary** and CSVs (`trades.csv`, `equity_curve.csv`).

## Usage
```bash
python -m tradebot.backtest --config backtest.sample.json
```

Use `--no-write` to skip CSV output. Add `--calibrate` to refresh the synthetic calibration using delayed LAST quotes.

## Cache layout
Bars are cached under `db/` (configurable):
```
db/<SYMBOL>/<SYMBOL>_<start>_<end>_<bar>_<rth|full>.csv
```
This cache is designed to be shared later with the live trading engine.

## Config (JSON)
See `backtest.sample.json`. Core fields:

### backtest
- `start`, `end` (YYYY-MM-DD)
- `bar_size` (e.g. `1 hour`)
- `use_rth` (true = RTH, false = full session)
- `starting_cash`
- `risk_free_rate` (fixed; TODO: load historical rates)
- `cache_dir` (default `db`)
- `calibration_dir` (default `db/calibration`)
- `calibrate` (true = refresh calibration before backtest)
- `output_dir` (default `backtests/out`)
- `offline` (true = use cached bars only; skip IBKR connections)

### strategy
- `name` (currently `credit_spread`)
- `symbol` (e.g. `MNQ`, `SLV`)
- `exchange` (optional; futures default to CME)
- `right` (`PUT` or `CALL`)
- `entry_days` (3-letter names, e.g. `Tue`)
  - If omitted or empty, defaults to all weekdays.
- `dte` (0 for 0DTE)
- `otm_pct` (percent OTM for short strike)
  - Negative values mean ITM (e.g., `-1.0` = 1% ITM).
- `width_pct` (spread width as % of spot)
- `profit_target` (fraction of credit, e.g. 0.5)
- `stop_loss` (fraction of max loss, e.g. 0.35)
- `exit_dte` (placeholder; currently not enforced)
- `quantity`
- `min_credit` (minimum credit to enter; units are option price)
- `ema_preset` (optional; `"9/21"` or `"20/50"`; entry allowed only when fast > slow)
  - EMA periods are **bar-based** (hourly bars = 9/21 hours; daily bars = 9/21 days).
- `ema_directional` (optional; if true, EMA direction selects CALL vs PUT: fast>slow = CALL, fast<slow = PUT)

#### Multi-leg strategies
You can replace the default spread params with explicit legs. If `legs` is present, it is used instead of `right/otm_pct/width_pct`.

Each leg supports:
- `action`: `BUY` or `SELL`
- `right`: `PUT` or `CALL`
- `moneyness_pct`: percent from spot (negative = ITM)
- `qty`: leg quantity (multiplied by `strategy.quantity`)

Example (short call spread):
```json
"legs": [
  {"action": "SELL", "right": "CALL", "moneyness_pct": 2.0, "qty": 1},
  {"action": "BUY", "right": "CALL", "moneyness_pct": 4.0, "qty": 1}
]
```

Example (naked short put):
```json
"legs": [
  {"action": "SELL", "right": "PUT", "moneyness_pct": 2.0, "qty": 1}
]
```

#### Regime filters (optional)
Add a `filters` block to gate entries:
- `rv_min`, `rv_max`: annualized realized vol bounds
- `ema_spread_min_pct`: require |EMA_fast-EMA_slow| / price (%) above threshold
- `ema_slope_min_pct`: require EMA fast slope (%) above threshold
- `entry_start_hour`, `entry_end_hour`: hourly window (0–23)
- `skip_first_bars`: skip first N bars of each session
- `cooldown_bars`: minimum bars between entries

Example:
```json
"filters": {
  "rv_min": 0.15,
  "rv_max": 0.60,
  "ema_spread_min_pct": 0.05,
  "ema_slope_min_pct": 0.01,
  "entry_start_hour": 10,
  "entry_end_hour": 15,
  "skip_first_bars": 2,
  "cooldown_bars": 4
}
```

### synthetic
- `rv_lookback` (bars)
- `rv_ewma_lambda`
- `iv_risk_premium`
- `iv_floor`
- `term_slope`
- `skew`
- `min_spread_pct`
  - This block is optional; defaults are used if omitted.

## 0DTE behavior
- Uses a **minimum time value** equal to one session for pricing (prevents zero premium).
- If TP/SL doesn’t trigger, exits at **end-of-day**.

## Outputs
- `summary.txt` (CLI stats)
- `trades.csv` (entry/exit per trade)
- `equity_curve.csv` (equity over time)

## Leaderboard (SLV 1h sweeps)
PnL is reported in **premium points * contract multiplier**. Offline sweeps use multiplier=1.0, so treat PnL as **relative**; equity options would scale by ~100x.

Sweep grid (period: 2025-07-02 -> 2026-01-02):
- DTE: 5/10/20
- Moneyness: 1/2/3%
- PT: 0.5/1.0
- SL: 0.35/0.8
- EMA: 9/21 or 20/50
- min_trades=8, entry_days=all weekdays

Full leaderboard is in `tradebot/backtest/LEADERBOARD.md`.

### Unfiltered (no regime filters)
Long CALL (debit), top 10 by PnL:
| PnL | Win | Trades | Hold(h) | DTE | M% | PT | SL | EMA |
| --- | --- | ------ | ------- | --- | --- | -- | -- | --- |
| 12.55 | 70.59% | 17 | 231.9 | 20 | 2.0 | 1.0 | 0.8 | 20/50 |
| 12.42 | 76.92% | 13 | 317.5 | 20 | 2.0 | 1.0 | 0.8 | 9/21 |
| 12.28 | 75.00% | 16 | 233.1 | 20 | 1.0 | 1.0 | 0.8 | 20/50 |
| 12.24 | 65.22% | 23 | 179.5 | 20 | 3.0 | 1.0 | 0.8 | 9/21 |
| 11.98 | 70.59% | 17 | 230.6 | 20 | 3.0 | 1.0 | 0.8 | 20/50 |
| 11.25 | 71.43% | 14 | 299.7 | 20 | 1.0 | 1.0 | 0.8 | 9/21 |
| 10.82 | 63.64% | 22 | 169.5 | 10 | 1.0 | 1.0 | 0.8 | 20/50 |
| 10.57 | 67.86% | 28 | 143.1 | 10 | 2.0 | 1.0 | 0.8 | 9/21 |
| 10.10 | 74.07% | 27 | 133.8 | 20 | 3.0 | 0.5 | 0.8 | 20/50 |
| 9.97 | 75.00% | 24 | 153.4 | 20 | 2.0 | 0.5 | 0.8 | 20/50 |

Top 30% ranges (of profitable combos): DTE 10-20, M% 1-3, PT 0.5-1.0, SL 0.8 only, EMA 20/50 > 9/21.

Short PUT (credit), top 10 by PnL:
| PnL | Win | Trades | Hold(h) | DTE | M% | PT | SL | EMA |
| --- | --- | ------ | ------- | --- | --- | -- | -- | --- |
| 9.56 | 80.00% | 70 | 48.5 | 5 | 1.0 | 0.5 | 0.8 | 20/50 |
| 8.79 | 78.57% | 14 | 259.4 | 10 | 1.0 | 1.0 | 0.8 | 20/50 |
| 8.68 | 73.91% | 23 | 142.3 | 5 | 1.0 | 1.0 | 0.8 | 20/50 |
| 8.14 | 60.00% | 10 | 423.2 | 20 | 3.0 | 1.0 | 0.35 | 9/21 |
| 8.10 | 75.95% | 79 | 39.1 | 5 | 2.0 | 0.5 | 0.8 | 20/50 |
| 7.68 | 75.00% | 8 | 523.9 | 20 | 1.0 | 1.0 | 0.35 | 9/21 |
| 7.18 | 73.68% | 57 | 62.8 | 10 | 1.0 | 0.5 | 0.8 | 20/50 |
| 7.00 | 60.76% | 79 | 39.8 | 10 | 2.0 | 0.5 | 0.35 | 20/50 |
| 6.64 | 69.23% | 26 | 135.6 | 5 | 1.0 | 1.0 | 0.8 | 9/21 |
| 6.49 | 60.71% | 28 | 110.0 | 5 | 2.0 | 1.0 | 0.35 | 20/50 |

Top 30% ranges (of profitable combos): DTE 5-20, M% 1-3, PT 0.5-1.0, SL 0.35-0.8, EMA 20/50 > 9/21.

Iron condor and call debit spread had **0 profitable combos** in this grid (see leaderboard file for details).

### Filtered (rv/ema/time window enabled)
Filters: `rv_min=0.15, rv_max=0.60, ema_spread_min_pct=0.05, ema_slope_min_pct=0.01, entry_start_hour=10, entry_end_hour=15, skip_first_bars=2, cooldown_bars=4`

Long CALL (debit), top 10 by PnL:
| PnL | Win | Trades | Hold(h) | DTE | M% | PT | SL | EMA |
| --- | --- | ------ | ------- | --- | --- | -- | -- | --- |
| 17.17 | 83.33% | 12 | 315.8 | 20 | 1.0 | 1.0 | 0.8 | 20/50 |
| 16.41 | 75.00% | 16 | 222.8 | 20 | 2.0 | 1.0 | 0.8 | 20/50 |
| 15.91 | 75.00% | 16 | 216.6 | 20 | 3.0 | 1.0 | 0.8 | 20/50 |
| 15.16 | 80.95% | 21 | 167.7 | 20 | 1.0 | 0.5 | 0.8 | 9/21 |
| 14.83 | 80.95% | 21 | 159.7 | 20 | 1.0 | 0.5 | 0.8 | 20/50 |
| 14.64 | 81.82% | 11 | 342.5 | 20 | 1.0 | 1.0 | 0.8 | 9/21 |
| 14.53 | 81.82% | 22 | 148.0 | 20 | 2.0 | 0.5 | 0.8 | 20/50 |
| 14.29 | 79.17% | 24 | 144.0 | 20 | 2.0 | 0.5 | 0.8 | 9/21 |
| 13.84 | 76.92% | 13 | 285.8 | 20 | 2.0 | 1.0 | 0.8 | 9/21 |
| 13.82 | 66.67% | 21 | 164.3 | 10 | 1.0 | 1.0 | 0.8 | 20/50 |

Top 30% ranges (of profitable combos): DTE 10-20, M% 1-3, PT 0.5-1.0, SL 0.8 only, EMA 20/50 ≈ 9/21.

Short PUT (credit), top 10 by PnL:
| PnL | Win | Trades | Hold(h) | DTE | M% | PT | SL | EMA |
| --- | --- | ------ | ------- | --- | --- | -- | -- | --- |
| 11.86 | 81.36% | 59 | 48.4 | 5 | 1.0 | 0.5 | 0.8 | 20/50 |
| 10.84 | 83.05% | 59 | 50.3 | 5 | 1.0 | 0.5 | 0.8 | 9/21 |
| 9.97 | 72.31% | 65 | 41.5 | 5 | 1.0 | 0.5 | 0.35 | 20/50 |
| 9.34 | 73.77% | 61 | 44.4 | 5 | 1.0 | 0.5 | 0.35 | 9/21 |
| 8.83 | 82.54% | 63 | 43.4 | 5 | 2.0 | 0.5 | 0.8 | 9/21 |
| 8.56 | 71.64% | 67 | 35.3 | 5 | 2.0 | 0.5 | 0.35 | 9/21 |
| 8.11 | 71.43% | 21 | 139.5 | 5 | 1.0 | 1.0 | 0.35 | 20/50 |
| 7.16 | 77.05% | 61 | 41.2 | 5 | 2.0 | 0.5 | 0.8 | 20/50 |
| 6.83 | 62.50% | 8 | 462.8 | 20 | 1.0 | 1.0 | 0.35 | 20/50 |
| 6.79 | 68.12% | 69 | 33.5 | 5 | 2.0 | 0.5 | 0.35 | 20/50 |

Top 30% ranges (of profitable combos): DTE 5-20, M% 1-3, PT 0.5-1.0, SL 0.35-0.8, EMA 20/50 > 9/21.

Iron condor and call debit spread had **0 profitable combos** in this grid (see leaderboard file for details).

## Reuse for live trading
The backtest engine is built around reusable components:
- **Strategy layer** emits intents
- **Execution logic** converts intents to spread pricing
- **Risk checks** apply TP/SL and exit rules

These modules will be reused in a live trading engine (future work), where market data and execution are real instead of synthetic.

## Limitations
- Synthetic options are **directional approximations**, not true bid/ask markets.
- Equity option quotes require OPRA for real data.
- Futures options bid/ask require CME options subscriptions.

## Testing notes (volatility)
- **Realized vol over time:** RV is computed per bar using EWMA of returns, so it moves through time during the backtest.
- **IV over time:** IV is synthetic (RV + static params). Calibration adjusts those params using *today’s* delayed LAST, so it improves realism for “now,” not historical regimes.
- **Persisted buckets:** Calibration records accumulate, but backtests currently use the **latest record only** (no as‑of date matching).

## Calibration (delayed LAST)
When `calibrate` is enabled (or `--calibrate` is passed), the engine will:
- Pull **delayed LAST** prices (tickType 68) across strikes/expiries.
- Fit IV parameters (floor, risk premium, skew, term slope) per expiry bucket.
- Persist results under `db/calibration/<SYMBOL>.json`.
This improves synthetic pricing without requiring OPRA/CME bid/ask.

## TODO
- Regime gating hooks
- Historical rates source
- Multi-strategy runs
- Live broker adapter
