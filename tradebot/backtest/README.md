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
    "starting_cash": 10000,
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
    "ema_entry_mode": "trend",
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
- `instrument` (optional; `"options"` (default) or `"spot"`)
  - `"options"` runs the synthetic options engine (current default).
  - `"spot"` runs an underlying/spot engine (equity/futures). Default fills are at bar close, but you can opt into
    more realistic execution/cost/sizing via the `spot_*` realism knobs below (next-open fills, intrabar PT/SL,
    spread/commission/slippage, ROI-based sizing).
- `symbol` (e.g. `MNQ`, `SLV`)
- `exchange` (optional; futures default to CME)
- `right` (`PUT` or `CALL`)
- `entry_days` (3-letter names, e.g. `Tue`)
  - If omitted or empty, defaults to all weekdays.
- `max_entries_per_day` (optional; defaults to `1`)
  - `0` means unlimited.
- `max_open_trades` (optional; defaults to `1`)
  - Limits concurrent open trades (stacking). `0` means unlimited.
- `dte` (0 for 0DTE)
- `otm_pct` (percent OTM for short strike)
  - Negative values mean ITM (e.g., `-1.0` = 1% ITM).
- `width_pct` (spread width as % of spot)
- `profit_target` (multiplier of entry premium/credit)
  - `0.5` = +50% of premium/credit
  - `1.0` = +100%
  - `10.0` = +1000% (10x)
- `stop_loss` (fraction of max loss, e.g. 0.35)
- `exit_dte` (optional; if >0, exit when remaining DTE is <= this value)
  - Uses **business days** to match how `dte` is counted.
  - Ignored if `exit_dte >= entry dte` (treated as disabled).
- `quantity`
- `min_credit` (minimum credit to enter; units are option price)
- `ema_preset` (optional; `"3/7"`, `"9/21"` or `"20/50"`)
- `ema_entry_mode` (optional; `"trend"` or `"cross"`; defaults to `"trend"`)
  - `trend`: entry allowed only when fast > slow (current behavior)
  - `cross`: entry allowed only on a fast/slow crossover (more “pivot”-style)
  - EMA periods are **bar-based** (hourly bars = 9/21 hours; daily bars = 9/21 days).
- `entry_confirm_bars` (optional; default `0`)
  - Debounce / confirmation gate for the EMA entry direction.
  - `cross`: requires the post-cross state to persist for `N` bars.
  - `trend`: requires the state to persist for `N` bars after a state change.
- `regime_ema_preset` (optional; e.g. `"8/21"`)
- `regime_bar_size` (optional; e.g. `"1 hour"`, `"4 hours"`)
  - If `regime_bar_size` differs from `backtest.bar_size`, the regime is computed on that timeframe (MTF gating).
- `regime_mode` (optional; `"ema"` (default) or `"supertrend"`)
  - `"ema"` uses `regime_ema_preset` as the regime direction.
  - `"supertrend"` uses Supertrend as the regime direction (ignores `regime_ema_preset`).
  - Supertrend params:
    - `supertrend_atr_period` (default `10`)
    - `supertrend_multiplier` (default `3.0`)
    - `supertrend_source` (`"hl2"` (default) or `"close"`)
- `ema_directional` (optional; if true, EMA direction selects CALL vs PUT: fast>slow = CALL, fast<slow = PUT)
- `exit_on_signal_flip` (optional; if true, exit when EMA signal flips against the open trade)
  - Uses `flip_exit_*` settings below.
- `flip_exit_mode` (optional; `"entry"`, `"state"`, `"cross"`; default `"entry"`)
  - `"entry"` = match entry mode (`trend` -> state, `cross` -> cross).
- `flip_exit_min_hold_bars` (optional; default `0`)
  - Prevents immediate whipsaw flip exits right after entry.
- `flip_exit_only_if_profit` (optional; default `false`)
  - Only flip-exit if the trade is currently profitable.
- `direction_source` (optional; defaults to `"ema"`)
  - Currently only `"ema"` is supported.
- `directional_legs` (optional; when present, the engine chooses different legs for `up` vs `down` direction)
  - Direction is derived from `direction_source` (currently EMA state/cross).
  - This is **single-expiry only** (no calendars yet).
  - When `directional_legs` is provided, it takes precedence over `legs` / `right`.

#### Spot / underlying mode (instrument = "spot")
When `instrument="spot"`, the engine trades the underlying itself instead of synthetic options:
- `directional_spot` (optional; map `up`/`down` to spot actions)
  - Example long-only: `{"up": {"action": "BUY", "qty": 1}}`
  - Example long/short: `{"up": {"action": "BUY", "qty": 1}, "down": {"action": "SELL", "qty": 1}}`
- `spot_profit_target_pct` / `spot_stop_loss_pct` (optional; exit early on +/- move from entry)
  - Example: `spot_profit_target_pct=0.02` = take profit at +2% move (sign-adjusted for shorts)
- `spot_close_eod` (optional; default `false`)
  - If `true`, closes any open spot positions on the last bar of each day.
- Spot backtest realism knobs (backtest-only; opt-in; default behavior remains unchanged):
  - `spot_entry_fill_mode`: `"close"` or `"next_open"`
    - `"next_open"` means we decide at bar close but fill at next bar open.
  - `spot_flip_exit_fill_mode`: `"close"` or `"next_open"`
    - `"next_open"` makes flip exits more realistic (no same-bar-close flip fills).
  - `spot_intrabar_exits`: `true/false`
    - When `true`, profit/stop are evaluated using OHLC (intrabar), not only bar close.
    - If both PT and SL are hit in the same bar, we assume **worst-case** ordering (stop-first).
  - `spot_spread`: spot bid/ask spread in **price units** (e.g., `0.01` for `$0.01/share`)
    - Modeled as half-spread on entry and half-spread on exit.
  - `spot_commission_per_share`: commission in **price units** (per share/contract); embedded into fills
  - `spot_commission_min`: commission minimum **per order** (price units), embedded into fills
  - `spot_slippage_per_share`: slippage per share (price units), embedded into fills
    - Applied on entry/stop/flip exits (market-like fills). Profit-target fills currently assume no extra slippage.
  - `spot_mark_to_market`: `"close"` or `"liquidation"`
    - `"liquidation"` marks open positions at bid/ask (not mid).
  - `spot_drawdown_mode`: `"close"` or `"intrabar"`
    - `"intrabar"` approximates worst-in-bar equity for max drawdown.
  - Position sizing (ROI-based; used by Realism v2):
    - `spot_sizing_mode`: `"fixed"` (default) | `"notional_pct"` | `"risk_pct"`
    - `spot_notional_pct`: fraction of equity to allocate per entry (`notional_pct` mode)
    - `spot_risk_pct`: fraction of equity risked to stop per entry (`risk_pct` mode)
    - `spot_max_notional_pct`: cap notional per entry as a fraction of equity
    - `spot_min_qty`, `spot_max_qty`: share bounds (`spot_max_qty=0` means no cap)

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

#### Directional legs (up/down mapping)
To “trade bottoms” more intelligently (e.g., sell puts on an up signal), configure different leg sets for `up` vs `down`.

Example: up = short put, down = long put (single expiry):
```json
"ema_preset": "3/7",
"ema_entry_mode": "cross",
"direction_source": "ema",
"directional_legs": {
  "up": [
    {"action": "SELL", "right": "PUT", "moneyness_pct": 2.0, "qty": 1}
  ],
  "down": [
    {"action": "BUY", "right": "PUT", "moneyness_pct": 2.0, "qty": 1}
  ]
}
```

#### Regime filters (optional)
Add a `filters` block to gate entries:
- `rv_min`, `rv_max`: annualized realized vol bounds
- `ema_spread_min_pct`: require |EMA_fast-EMA_slow| / price (%) above threshold
- `ema_slope_min_pct`: require EMA fast slope (%) above threshold
- `volume_ratio_min`: require `volume / EMA(volume)` above threshold
- `volume_ema_period`: EMA period for volume gating (default `20` when `volume_ratio_min` is set)
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
  "volume_ratio_min": 1.2,
  "volume_ema_period": 20,
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
PnL is reported in **premium points * contract multiplier**.
- Equity options use multiplier `100`, so values are approximately **USD per contract**.
- When `max_entries_per_day=0` and `max_open_trades=0`, results reflect **stacking / pyramiding** subject to `starting_cash` and margin.

Full leaderboard is in `tradebot/backtest/LEADERBOARD.md`.
Machine-readable presets are in `tradebot/backtest/leaderboard.json` (regenerate with `python -m tradebot.backtest.generate_leaderboard`; spot milestones are appended by default).

## Spot leaderboard (MNQ, 12m)
Spot presets are stored in `tradebot/backtest/spot_milestones.json` and are merged into the TUI presets list alongside the options leaderboard.

These spot presets are **12-month only** (no 6m snapshot entries) and are filtered for stability:
- win rate `>= 55%`
- trades `>= 200`
- pnl/dd `>= 8`
- sorted by pnl/dd (desc)

Regenerate (offline, uses cached bars in `db/`):
```bash
python -m tradebot.backtest.evolve_spot --offline --axis all --write-milestones --cache-dir db
python -m tradebot.backtest.evolve_spot --offline --axis combo --write-milestones --merge-milestones --cache-dir db
```

Regenerate with realism v1 enabled (recommended before live trading):
```bash
python -m tradebot.backtest.evolve_spot --offline --axis all --realism --write-milestones --cache-dir db
python -m tradebot.backtest.evolve_spot --offline --axis combo --realism --write-milestones --merge-milestones --cache-dir db
```

Regenerate 30-minute spot champions (adds only a curated top set from that run, merges into existing milestones):
```bash
python -m tradebot.backtest.evolve_spot --offline --bar-size "30 mins" --axis combo \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 25 --milestone-add-top-pnl 25 \
  --cache-dir db
```

For deeper “one axis at a time” exploration (still spot-only), run a single axis:
```bash
python -m tradebot.backtest.evolve_spot --axis regime --cache-dir db
```

### Sweep coverage (ranges tried so far)
This is a **quick map of what the sweeps actually cover** (outer edges), so we can spot gaps and avoid re-running broad funnels.

**Timing (signal layer)**
- EMA preset (`--axis ema`): `2/4`, `3/7`, `4/9`, `5/10`, `8/21`, `9/21`
- Entry semantics (`--axis entry_mode`): `ema_entry_mode ∈ {cross, trend}`, `entry_confirm_bars ∈ {0,1,2}`
- Confirm-only (`--axis confirm`): `entry_confirm_bars ∈ {0,1,2,3}`

**Permission / quality filters**
- Time-of-day ET (`--axis tod`): RTH windows `(9–16, 10–15, 11–16)` and overnight micro-grid `start ∈ {16..20}`, `end ∈ {2..6}` (wraps)
- TOD interaction (`--axis tod_interaction`): `start ∈ {17,18,19}`, `end ∈ {3,4,5}`, `skip_first_bars ∈ {0,1,2}`, `cooldown_bars ∈ {0,1,2}`
- Permission joint (`--axis perm_joint`): combines `tod` windows (incl base/off + selected RTH/overnight) × `spread` variants × `volume` variants (no funnel pruning)
- EMA×permission joint (`--axis ema_perm_joint`): shortlist `ema_preset ∈ {2/4,3/7,4/9,5/10,8/21,9/21,21/50}` → sweep a targeted set of `tod/spread/volume` combos
- Tick×permission joint (`--axis tick_perm_joint`): shortlist Raschke `$TICK` params (`z_enter/z_exit/slope/lookback/policy/dir_policy`) → sweep a targeted set of `tod/spread/volume` combos
- Tick×EMA joint (`--axis tick_ema`): `ema_preset ∈ {2/4,3/7,4/9,5/10,8/21,9/21,21/50}` × Raschke `$TICK` params (`z_enter/z_exit/slope/lookback/policy`, with `dir_policy=wide_only`)
- Chop joint (`--axis chop_joint`): `ema_slope_min_pct ∈ {None, 0.005,0.01,0.02,0.03}` × `cooldown_bars ∈ {0,1,2,3,4,6}` × `skip_first_bars ∈ {0,1,2,3}`
- Weekdays (`--axis weekday`): several hand-picked sets (Mon–Fri, Tue–Thu, etc.)
- EMA spread gate (`--axis spread`): `ema_spread_min_pct ∈ {None, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10}`
- EMA spread fine (`--axis spread_fine`): `ema_spread_min_pct ∈ {None, 0.0020..0.0080 step 0.0005}`
- EMA slope gate (`--axis slope`): `ema_slope_min_pct ∈ {None, 0.005, 0.01, 0.02, 0.03, 0.05}`
- Volume gate (`--axis volume`): `volume_ratio_min ∈ {None, 1.0, 1.1, 1.2, 1.5}`, `volume_ema_period ∈ {10,20,30}`
- Realized-vol band gate (`--axis rv`): `rv_min ∈ {None, 0.25,0.30,0.35,0.40,0.45}`, `rv_max ∈ {None, 0.70,0.80,0.90,1.00}`
- Cooldown (`--axis cooldown`): `cooldown_bars ∈ {0,1,2,3,4,6,8}`
- Skip-open (`--axis skip_open`): `skip_first_bars ∈ {0,1,2,3,4,6}`
- $TICK gate (“Raschke width”, `--axis tick`):
  - `z_enter ∈ {0.8,1.0,1.2}`, `z_exit ∈ {0.4,0.5,0.6}`, `slope_lookback ∈ {3,5}`, `z_lookback ∈ {126,252}`
  - `tick_neutral_policy ∈ {allow, block}`, `tick_direction_policy ∈ {both, wide_only}`
  - Default symbol tries `TICK-AMEX` (fallback from `TICK-NYSE` if permissions/cache block it)

**Bias / confirm (regime layers)**
- Regime sweep (`--axis regime`): `regime_bar_size ∈ {4 hours, 1 day}`, `ST ATR ∈ {2,3,4,5,6,7,10,11,14,21}`, `mult ∈ {0.05..2.0 (curated list)}`, `src ∈ {close, hl2}`
- Regime2 sweep (`--axis regime2`): `ST2 @ 4h`, same `ATR/mult/src` grid family as above
- Regime2 EMA confirm (`--axis regime2_ema`): `EMA preset ∈ {3/7,4/9,5/10,8/21,9/21,21/50}`, `bar ∈ {4 hours, 1 day}`
- Joint regime×regime2 (`--axis joint`): a tight interaction grid around `regime ST(10/14/21, mult 0.4–0.6)` × `regime2 (4h/1d) ST2(3..14, mult 0.2–0.5)`
- Micro ST neighborhood (`--axis micro_st`): granular mult grid around the current post-fix champ neighborhood (4h ST + 4h ST2, close source)
- Regime×ATR joint (`--axis regime_atr`): shortlist regimes (`ST @ {4h,1d}`, `ATR ∈ {3,5,6,7,10,14,21}`, `mult ∈ {0.4..1.5}`, `src ∈ {hl2,close}`) → exit micro-grid (`ATR ∈ {10,14,21}`, `PTx ∈ {0.6..1.0}`, `SLx ∈ {1.2..2.0}`)
- EMA×regime joint (`--axis ema_regime`): `ema_preset ∈ {2/4,3/7,4/9,5/10,8/21,9/21,21/50}` × `ST @ 4h (ATR ∈ {2,3,4,5,6,7,10,14,21}, mult ∈ {0.2..1.5}, src ∈ {hl2,close})` plus `ST @ 1d (ATR ∈ {7,10,14,21}, mult ∈ {0.4..1.2}, src ∈ {hl2,close})`

**Exits / position management**
- Fixed % exits (`--axis ptsl`): `PT ∈ {0.005, 0.01, 0.015, 0.02}`, `SL ∈ {0.015, 0.02, 0.03}`
- ATR exits coarse (`--axis atr`): `ATR period ∈ {7,10,14,21}`, `PTx ∈ {0.6,0.8,0.9,1.0,1.5,2.0}`, `SLx ∈ {1.0,1.5,2.0}`
- ATR exits fine (`--axis atr_fine`): `ATR period ∈ {7,10,14,21}`, `PTx ∈ {0.8,0.9,1.0,1.1,1.2}`, `SLx ∈ {1.2..1.8 step 0.1}`
- ATR exits ultra (`--axis atr_ultra`): `ATR period=7`, `PTx ∈ {1.05,1.08,1.10,1.12,1.15}`, `SLx ∈ {1.35..1.55 step 0.05}`
- EMA×ATR joint (`--axis ema_atr`): shortlist `ema_preset ∈ {2/4,3/7,4/9,5/10,8/21,9/21,21/50}` → exits `ATR period ∈ {10,14,21}`, `PTx ∈ {0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.00}`, `SLx ∈ {1.20,1.40,1.50,1.60,1.80,2.00}` (covers the PT<1.0 pocket explicitly)
- Regime2×ATR joint (`--axis r2_atr`): regime2 Supertrend coarse scan (`ATR ∈ {7,10,11,14,21}`, `mult ∈ {0.6,0.8,1.0,1.2,1.5}`, `src ∈ {hl2,close}`, `bar ∈ {4h,1d}`) → exit micro-grid (`ATR ∈ {14,21}`, `PTx ∈ {0.6..1.0}`, `SLx ∈ {1.2..2.2}`)
- Regime2×TOD joint (`--axis r2_tod`): shortlist regime2 settings (`ST2 @ {4h,1d}`; see `evolve_spot.py`) → sweep TOD windows (RTH + overnight micro-grid)
- Flip-exit semantics (`--axis flip_exit`): `exit_on_signal_flip ∈ {on,off}`, `flip_exit_mode ∈ {entry,state,cross}`, `hold ∈ {0,2,4,6}`, `only_if_profit ∈ {0,1}`
- Loosenings (`--axis loosen`): `max_open_trades ∈ {1,2,3,0}`, `spot_close_eod ∈ {0,1}`
- Loosen×ATR joint (`--axis loosen_atr`): `max_open_trades ∈ {2,3,0}`, `spot_close_eod ∈ {0,1}` × exits `ATR period ∈ {10,14,21}`, `PTx ∈ {0.60..0.80 step 0.05}`, `SLx ∈ {1.20..2.00 step 0.20}`
- Exit-time flatten (`--axis exit_time`): `spot_exit_time_et ∈ {None, 04:00, 09:30, 10:00, 11:00, 16:00, 17:00}`

**ORB**
- ORB (`--axis orb`, runs on 15m bars): `open_time_et ∈ {09:30, 18:00}`, `window_mins ∈ {15,30,60}`, `target_mode ∈ {rr, or_range}`, `rr ∈ {0.618,0.707,0.786,1.0,1.272,1.618,2.0}`, optional `vol>=1.2@20`, plus a session TOD filter.
- ORB joint (`--axis orb_joint`, runs on 15m bars): stage-1 shortlist of ORB params → apply `regime ∈ {off, ST @ 4h (small curated set)}` × `tick ∈ {off, wide_only allow/block (z=1.0/0.5 slope=3 lb=252)}`

**Known gaps (we now target explicitly)**
- Some interaction edges require **joint sweeps** rather than one-axis sweeps (e.g. `regime2 × ATR exits` with `PTx < 1.0`): this is the class of gap the combo funnel can miss, and is now covered by `--axis r2_atr`.
- `--axis combo` is a bounded “smart exhaustive” funnel that now includes direction×regime scan + the low-PT ATR pocket + Raschke `$TICK` gate + RV band + exit-time + a few TOD windows. It is still a funnel (not the full universe), but it should no longer miss entire categories of interactions we care about.

Quick “current top 3” snapshots (generated 2026-01-16, post-intraday-timestamp-fix):

- **#1 Best 30m (risk-adjusted; meets leaderboard thresholds):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=21`, `mult=0.5`, `src=close`
  - Permission (quality): `ema_spread_min_pct=0.005`
  - Permission (time-of-day): `entry_start_hour_et=18`, `entry_end_hour_et=4` (wraps overnight)
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=5`, `mult=0.3`, `src=close`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=7`, `spot_pt_atr_mult=1.12`, `spot_sl_atr_mult=1.5`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=303`, `win=58.1%`, `pnl=+13055.5`, `dd=741.0`, `pnl/dd=17.62`

- **#2 Best 30m (risk-adjusted):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=21`, `mult=0.5`, `src=close`
  - Permission (quality): `ema_spread_min_pct=0.005`
  - Permission (time-of-day): `entry_start_hour_et=18`, `entry_end_hour_et=4` (wraps overnight)
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=5`, `mult=0.3`, `src=close`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=7`, `spot_pt_atr_mult=1.05`, `spot_sl_atr_mult=1.4`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=303`, `win=58.4%`, `pnl=+12496.5`, `dd=727.0`, `pnl/dd=17.19`

- **#3 Best 30m (risk-adjusted):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=21`, `mult=0.5`, `src=close`
  - Permission (quality): `ema_spread_min_pct=0.005`
  - Permission (time-of-day): `entry_start_hour_et=18`, `entry_end_hour_et=4` (wraps overnight)
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=5`, `mult=0.3`, `src=close`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=7`, `spot_pt_atr_mult=1.10`, `spot_sl_atr_mult=1.5`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=303`, `win=58.1%`, `pnl=+12730.0`, `dd=741.0`, `pnl/dd=17.18`

Quick “max net PnL” snapshots (generated 2026-01-16, post-intraday-timestamp-fix):

- **#1 Best 30m (max PnL):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=14`, `mult=0.5`, `src=hl2`
  - Permission (quality): `off`
  - Permission (time-of-day): `off`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=14`, `spot_pt_atr_mult=0.70`, `spot_sl_atr_mult=1.60`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=1029`, `win=57.0%`, `pnl=+24693.0`, `dd=2472.0`, `pnl/dd=9.99`

- **#2 Best 30m (max PnL, spread-gated):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=14`, `mult=0.5`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=14`, `spot_pt_atr_mult=0.70`, `spot_sl_atr_mult=1.60`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=904`, `win=56.7%`, `pnl=+23377.0`, `dd=2360.0`, `pnl/dd=9.91`

- **#3 Best 30m (high PnL + better pnl/dd, spread-gated):**
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=14`, `mult=0.5`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.005`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=atr`, `spot_atr_period=14`, `spot_pt_atr_mult=0.70`, `spot_sl_atr_mult=1.60`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=2`, `spot_close_eod=false`
  - Stats: `trades=821`, `win=56.9%`, `pnl=+22486.5`, `dd=1867.5`, `pnl/dd=12.04`

### Spot cross-asset sanity (TQQQ, 10y, RTH)
These were found by running our spot combo sweep on `TQQQ` over `2016-01-01 → 2026-01-08` with `use_rth=true`.

#### REALISM v2 multi-window stability (ROI-based, long-only)
These are the current “go-live shaped” TQQQ presets found under **Realism v2**:
- Long-only.
- Entry fills at **next bar open** (no same-bar-close fills).
- Intrabar PT/SL evaluated using OHLC (worst-case ordering when both hit in one bar).
- Liquidation marking (bid/ask) for equity/DD + intrabar drawdown approximation.
- Costs (defaults used for these presets): `spot_spread=0.01` (=$0.01/share), `spot_commission_per_share=0.005` (min `$1.00`), no extra slippage.
- Position sizing (ROI-based): `spot_sizing_mode=risk_pct`, `spot_risk_pct=0.01`, `spot_max_notional_pct=0.50`.

**Presets (already merged into `tradebot/backtest/spot_milestones.json` for the TUI):**

- **K30v2-01 (30 mins):** `ema=4/9 cross`, `ST(21,1.0,hl2)@4h + ST2(4h:5,0.2,close)`, exits `PT=0.015 SL=0.030`, `max_open=1 close_eod=1`, filters `rv=0.25..0.8 spread>=0.003`
  - 10y stats: `tr=219`, `win=53.0%`, `pnl=+3976.2`, `roi=+3.98%`, `dd=7523.8`, `dd%=7.52%`, `roi/dd=0.53`

- **K30v2-02 (30 mins):** same as K30v2-01 but `spread>=0.005`
  - 10y stats: `tr=218`, `win=52.8%`, `pnl=+3668.7`, `roi=+3.67%`, `dd=7798.3`, `dd%=7.80%`, `roi/dd=0.47`

- **K1Hv2-01 (1 hour):** `ema=4/9 cross`, `ST(3,0.3,hl2)@1d + ST2(1d:7,0.4,close)`, exits `ATR(7) PTx1.0 SLx1.0`, `max_open=1 close_eod=1`, filter `spread>=0.005`
  - 10y stats: `tr=215`, `win=52.1%`, `pnl=+1850.8`, `roi=+1.85%`, `dd=8010.3`, `dd%=8.01%`, `roi/dd=0.23`

#### LEGACY (realism v1; per-share PnL, no sizing)
These are older “Realism v1” TQQQ presets (still useful for directionally testing signal logic, but not ROI-comparable):
- They use next-open fills + intrabar PT/SL + spread, but **no sizing**, no per-order commission minimums, and no slippage.
- Their `pnl` is effectively “per-share” (multiplier `1.0`) under a large starting cash pile, so ROI is not meaningful.

- **K30-01 (30 mins):** `ema=4/9 cross`, `ST(21,1.0,hl2)@4h`, exits `ATR(21) PTx0.70 SLx1.80`, `hold=4`, `max_open=1 close_eod=1`
  - 10y stats: `tr=674`, `win=63.1%`, `pnl=11.88`, `dd=3.70`, `pnl/dd=3.21`
  - 1y slices:
    - 2023→2024: `tr=54`, `win=68.5%`, `pnl=2.42`, `dd=0.67`, `pnl/dd=3.59`
    - 2024→2025: `tr=69`, `win=69.6%`, `pnl=3.36`, `dd=2.26`, `pnl/dd=1.49`
    - 2025→2026-01-19: `tr=80`, `win=66.2%`, `pnl=5.29`, `dd=1.94`, `pnl/dd=2.73`

- **K30-02 (30 mins):** `ema=4/9 cross`, `ST(7,1.0,hl2)@4h`, exits `ATR(21) PTx0.70 SLx1.80`, `hold=4`, `max_open=1 close_eod=1`
  - 10y stats: `tr=633`, `win=62.6%`, `pnl=11.80`, `dd=4.84`, `pnl/dd=2.43`
  - 1y slices:
    - 2023→2024: `tr=59`, `win=69.5%`, `pnl=2.54`, `dd=0.67`, `pnl/dd=3.77`
    - 2024→2025: `tr=67`, `win=68.7%`, `pnl=3.32`, `dd=2.27`, `pnl/dd=1.46`
    - 2025→2026-01-19: `tr=74`, `win=68.9%`, `pnl=6.87`, `dd=1.68`, `pnl/dd=4.09`

- **K1H-02 (1 hour):** `ema=2/4 cross`, `ST(21,0.8,close)@4h + ST2(1d:7,0.4,close)`, exits `ATR(10) PTx0.80 SLx1.80`, `hold=4`, `max_open=2`
  - 10y stats: `tr=733`, `win=64.9%`, `pnl=16.68`, `dd=6.22`, `pnl/dd=2.68`
  - 1y slices:
    - 2023→2024: `tr=69`, `win=66.7%`, `pnl=3.06`, `dd=2.45`, `pnl/dd=1.25`
    - 2024→2025: `tr=83`, `win=66.3%`, `pnl=5.08`, `dd=4.46`, `pnl/dd=1.14`
    - 2025→2026-01-19: `tr=92`, `win=68.5%`, `pnl=9.22`, `dd=4.49`, `pnl/dd=2.06`

Repro notes:
- Candidate pool generation: `python -m tradebot.backtest.evolve_spot --axis combo --realism2 --long-only ... --write-milestones --milestones-out tradebot/backtest/spot_milestones.tqqq_10y_<bar>_realism_v2.json`
- Stability scoring: `python -m tradebot.backtest multitimeframe --milestones tradebot/backtest/spot_milestones.tqqq_10y_<bar>_realism_v2.json --max-open <N> --require-positive-pnl ...`

#### LEGACY (pre-realism; optimistic)
Note:
- `max_open=0` means **unlimited stacking** (subject to `starting_cash` margin constraints); this can materially change results.
- Spot PnL for equities is per-share (multiplier `1.0`), not per-contract `100`.

- 1 hour (combo sweep outcome, verified by direct backtest)
  - Best pnl/dd family found (note: uses `max_open=0` = unlimited stacking):
    - `tr=12407`, `win=64.25%`, `pnl=249.3`, `dd=34.5`, `pnl/dd=7.22`
    - `ema=3/7 trend`, `ST(3,0.8,close)@4h`, exits `PT=0.005 SL=0.030`, `hold=4`, filters `spread>=0.005 slope>=0.01`
  - High net-PnL family (also `max_open=0`):
    - `tr=14439`, `win=59.05%`, `pnl=318.1`, `dd=151.4`, `pnl/dd=2.10`
    - `ema=9/21 trend`, `ST(3,0.8,close)@4h`, exits `PT=0.015 SL=0.030`, `hold=0`, filter `spread>=0.005`

- 30 mins (full combo sweep output captured)
  - Full printout saved at `backtests/out/tqqq_10y_combo_30m.txt`.
  - Top by pnl/dd (more “realistic-feeling” because it’s not unlimited stacking; it’s `max_open=1` and `close_eod=1`):
    - `tr=1081`, `win=54.95%`, `pnl=31.3`, `dd=3.3`, `pnl/dd=9.60`
    - `ema=3/7 cross`, `ST(7,0.6,close)@4h`, exits `ATR(10) PTx0.9 SLx1.8`, `hold=4`, `max_open=1 close_eod=1`, `ST2(1d:7,0.4,close)`
  - Top by net pnl (again `max_open=0` = unlimited stacking):
    - `tr=26508`, `win=53.96%`, `pnl=585.9`, `dd=153.5`, `pnl/dd=3.82`
    - `ema=9/21 trend`, `ST(3,0.8,close)@4h`, exits `PT=0.015 SL=0.030`, `hold=4`, `max_open=0`

Full presets: see `tradebot/backtest/spot_milestones.json` (top entries) for exact strategy payloads (these are loaded as presets in the TUI automatically).

Note: We fixed daily-bar timestamp normalization on 2026-01-11 to avoid lookahead leakage for `1 day` multi-timeframe gates. We fixed intraday bar timestamps on 2026-01-14 so bar timestamps represent bar *close* time (eliminates MTF lookahead for `4 hours`, `1 hour`, `30 mins`, etc.). Spot milestones were regenerated after this fix.

### LEGACY / OUTDATED (pre-fix, for archaeology only)
These are spot presets recorded **before** the 2026-01-14 intraday-bar timestamp normalization fix.

They are preserved for archaeology in `tradebot/backtest/spot_milestones.legacy_pre_2026-01-14_intraday_ts_fix.json`, but should **not** be trusted for decision-making because they were affected by multi-timeframe lookahead leakage (notably `4 hours` regime gating on intraday signal bars).

#### Prior 30m/1h “champions” (pre-2026-01-14 intraday ts fix; do not trust)
- “New best 30m (risk-adjusted)” (recorded 2026-01-11): `trades=1067`, `win=58.1%`, `pnl=+56377.0`, `dd=897.5`, `pnl/dd=62.82`
- “Prior best 30m (risk-adjusted)” (recorded 2026-01-11): `trades=1155`, `win=55.84%`, `pnl=+54779.0`, `dd=899.0`, `pnl/dd=60.93`
- “New best 30m (max PnL)” (recorded 2026-01-11): `trades=1103`, `win=64.0%`, `pnl=+88550.5`, `dd=1928.0`, `pnl/dd=45.93`
  - Signal (timing): `30 mins`, EMA `2/4` cross, `entry_confirm_bars=0`
  - Regime (bias): Supertrend on `4 hours`, `ATR=3`, `mult=0.05`, `src=close`
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=7`, `mult=0.075`, `src=hl2`
  - Exits: `spot_exit_mode=pct`, `spot_profit_target_pct=0.01`, `spot_stop_loss_pct=0.03`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
  - Loosenings: `max_entries_per_day=0`, `max_open_trades=0`, `spot_close_eod=false`
  - Stats: `trades=1103`, `win=64.0%`, `pnl=+88550.5`, `dd=1928.0`, `pnl/dd=45.93`
- “Prior best 30m (max PnL)” (recorded 2026-01-11): `trades=1155`, `win=62.8%`, `pnl=+86227.0`, `dd=1928.0`, `pnl/dd=44.72`
- “Best 1h (risk-adjusted)” (recorded 2026-01-11): `trades=506`, `win=61.5%`, `pnl=+42961.5`, `dd=1371.0`, `pnl/dd=31.34`

---

These are milestone configs we recorded **before** the 2026-01-11 daily-bar timestamp normalization fix.

They are preserved to document the evolutionary path, but should **not** be trusted for decision-making because they were affected by lookahead leakage in `1 day` regime gating.

**Legacy 12m champion (pre-fix record, do not trust):**
- Signal (timing): `1 hour`, EMA `2/4` cross, `entry_confirm_bars=0`
- Regime (bias): Supertrend on `1 day`, `ATR=7`, `mult=0.4`, `src=close`
- Exits: `spot_profit_target_pct=0.015`, `spot_stop_loss_pct=0.03`, `exit_on_signal_flip=true`, `flip_exit_min_hold_bars=4`
- Reported stats (pre-fix): `trades=406`, `win=55.17%`, `pnl=+48453.0`, `dd=1981.5`, `pnl/dd=24.45`
- Post-fix replay (same params, 2025-01-08 → 2026-01-08): `trades=477`, `win=44.23%`, `pnl=-4321.0`, `dd=11409.0`, `pnl/dd=-0.38`
- Full preset (for reproducibility; pre-fix record):
  - `signal_bar_size=1 hour`, `signal_use_rth=false`
  - `instrument=spot`, `symbol=MNQ`
  - `ema_preset=2/4`, `ema_entry_mode=cross`, `entry_confirm_bars=0`
  - `regime_mode=supertrend`, `regime_bar_size=1 day`, `supertrend_atr_period=7`, `supertrend_multiplier=0.4`, `supertrend_source=close`
  - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=4`, `flip_exit_only_if_profit=false`
  - `spot_profit_target_pct=0.015`, `spot_stop_loss_pct=0.03`, `spot_close_eod=false`
  - `directional_spot.up=BUY x1`, `directional_spot.down=SELL x1`, `filters=null`

**Legacy “micro Supertrend” 12m configs (pre-fix records, do not trust):**
- Regime (bias): Supertrend on `1 day`, `ATR=2`, `mult=0.125`, `src=close`
  - Reported stats (pre-fix): `trades=546`, `win=56.4%`, `pnl=+60020.0`, `dd=2847.5`, `pnl/dd=21.08`
  - Post-fix replay (same params): `trades=438`, `win=42.92%`, `pnl=-13467.0`, `dd=20819.5`, `pnl/dd=-0.65`
- Regime (bias): Supertrend on `1 day`, `ATR=5`, `mult=0.2`, `src=close`, `spot_close_eod=true`
  - Reported stats (pre-fix): `trades=541`, `win=56.2%`, `pnl=+51361.0`, `dd=2710.5`, `pnl/dd=18.95`
  - Post-fix replay (same params): `trades=471`, `win=44.80%`, `pnl=-10453.5`, `dd=20065.5`, `pnl/dd=-0.52`

**Prior best (recorded earlier, for comparison):**
- 2025-01-08 → 2026-01-08: `bar=30 mins`, EMA=`2/4`, `entry_confirm_bars=2`
- Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.8`, `src=close`
- Exits: `spot_profit_target_pct=0.02`, `spot_stop_loss_pct=0.015`, `flip_exit_min_hold_bars=3`
- Reported stats (earlier record): `trades=661`, `win=49.0%`, `pnl=+33656.0`, `dd=1954.0`, `pnl/dd=17.22`
- Post-fix replay (same params): `trades=778`, `win=47.43%`, `pnl=+29341.0`, `dd=2750.0`, `pnl/dd=10.67`

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
- **Persisted buckets:** Calibration records accumulate, and backtests apply the latest record **as-of the simulated bar date** (walk-forward). If no record exists yet for that date/bucket, the base synthetic params are used.

## Calibration (delayed LAST)
When `calibrate` is enabled (or `--calibrate` is passed), the engine will:
- Pull **delayed LAST** prices (tickType 68) across strikes/expiries.
- Fit IV parameters (floor, risk premium, skew, term slope) per expiry bucket.
- Persist results under `db/calibration/<SYMBOL>.json`.
This improves synthetic pricing without requiring OPRA/CME bid/ask.

## TODO
- Realism pass (spot backtest): quantify how rankings change under more realistic execution/cost/sizing assumptions
  - Next-bar execution (implemented): `spot_entry_fill_mode=next_open`, `spot_flip_exit_fill_mode=next_open`
  - Intrabar TP/SL (implemented): `spot_intrabar_exits=true` with deterministic tie-break (stop-first)
  - Stop gap handling (implemented in Realism v2): if bar opens through stop, fill at `bar.open` (worst-case)
  - Risk model realism (partially implemented): `spot_drawdown_mode=intrabar` (worst-in-bar approximation; MAE/MFE still TODO)
  - Explicit cost model (implemented for spot): `spot_spread`, `spot_commission_per_share`, `spot_commission_min`, `spot_slippage_per_share`
    - Slippage is a simple per-share add-on; TODO: calibrate a more realistic slippage model (and/or apply symmetric slippage on profit targets)
  - Position sizing + ROI reporting (implemented in Realism v2): `spot_sizing_mode`, `spot_risk_pct`, `spot_notional_pct`, `spot_max_notional_pct`, `spot_min_qty`, `spot_max_qty`
    - `roi = pnl / starting_cash`, `dd% = max_drawdown / starting_cash`
  - ET session/day boundaries (TODO): make `max_entries_per_day`, `spot_close_eod`, `bars_in_day` align with ET session logic
  - Sensitivity report (TODO): compare champ/top-10 deltas (pnl, pnl/dd, WR, trades) across realism settings
- Realism pass (options backtest): quantify the impact of the synthetic market model and its simplifications
  - Options backtests are synthetic (not real markets). Prices come from Black-Scholes/Black-76 on the underlying bar close + a synthetic IV surface, and fills use a synthetic bid/ask (“mid-edge”) around the model mid. See `tradebot/backtest/engine.py:1853` and `tradebot/backtest/synth.py:54`.
  - No explicit commissions/fees anywhere (TODO: add a per-contract/per-order cost model).
  - Open trades are marked at mid (not executable), and multi-leg combos get one net edge, not per-leg edges. See `tradebot/backtest/engine.py:1909` and `tradebot/backtest/synth.py:75`.
    - This can materially change PnL + drawdown behavior vs legging / real combo markets.
- Evolution sweeps (add one axis at a time)
  - Walk-forward selection (train earlier slice, test later slice)
  - Multi-year evaluation (regime diversity): run the same sweep/shortlist across multiple years to detect regime brittleness
  - Time-of-day gating (RTH windows, skip-open/lunch chop)
  - ATR-scaled exits (dynamic PT/SL vs fixed %)
  - Dual regime gates (e.g., 1 day bias + 4 hours confirmation)
  - Volume gates (volume_ratio_min + volume EMA)
- Opening range breakout (ORB) idea: define OR (first 5–15 min after 9:30am ET), breakout + volume confirm, stop at opposite OR, target 1–2x risk; avoid news days
  - ORB variations: fib targets (`orb_risk_reward=0.618` / `1.618`), OR-range targets (`orb_target_mode=or_range`), OR buffers, and/or 5m timing bars (heavier data pulls)
- Spot parameter sweeps (cached OHLCV)
- Historical rates source
- Multi-strategy runs
- Live broker adapter
