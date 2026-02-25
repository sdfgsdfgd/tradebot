# Backtest Engine (Synthetic Options + Spot)

This package provides a minimal backtest runner that supports:
- **synthetic options** (builds synthetic option prices from underlying bars), and
- **spot** (underlying-only strategies with realism knobs like intrabar exits / execution fills).

Shared decision logic lives in `tradebot/engine.py` and `tradebot/spot_engine.py` so the UI/live runner can reuse the exact same semantics.


## Spot champion logs (moved)

TQQQ champion and evolution logs are now split into dedicated files:
- Low-frequency / historical champion lineage: `backtests/tqqq/readme-lf.md`
- High-frequency research track: `backtests/tqqq/readme-hf.md`

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

## Tools

- Record option quote snapshots (JSONL) for calibration / sanity checks:
  - `python -m tradebot.backtest.tools.record_quotes --symbol SLV --count 10 --interval 60`
  - Output default: `db/quotes/<SYMBOL>/<YYYY-MM-DD>.jsonl`
- Unified cache tool (`tradebot.backtest.tools.cache_ops`) subcommands:
  - `repair`: `python -m tradebot.backtest.tools.cache_ops repair --cache-file db/SLV/SLV_2025-01-08_2026-01-08_1min_full24.csv --heal --aggressive`
  - `sync`: `python -m tradebot.backtest.tools.cache_ops sync --champion-current --aggressive --force-refresh`
  - `resample`: `python -m tradebot.backtest.tools.cache_ops resample --symbol SLV --start 2025-01-08 --end 2026-01-08 --src-bar-size "5 mins" --dst-bar-size "10 mins" --cache-dir db`
  - Backward-compatible aliases remain: `fetch -> sync`, `audit-heal/audit/heal -> repair`.
  - Supports ET session coverage checks, targeted archive overlay, targeted IBKR refetch, UTC-naive canonicalization, threaded cache retrieval, and deterministic resampling.

## Cache layout
Bars are cached under `db/` (configurable):
```
db/<SYMBOL>/<SYMBOL>_<start>_<end>_<bar>_<rth|full24>.csv
```
This cache is designed to be shared later with the live trading engine.
Legacy note: older caches used `_full.csv`; the loader still recognizes them for backwards compatibility.

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
- `open_position_cap` (optional; defaults to `1`)
  - For `instrument="spot"`, this field is ignored; spot backtests always run single-position (live parity).
  - For `instrument="options"`, this limits concurrent open trades; `0` means unlimited.
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
- `flip_exit_gate_mode` (optional; default `"off"`)
  - Gates *flip exits* (not PT/SL) to avoid whipsaws while bias/permission still supports the open position.
  - Values:
    - `"off"` (default): current behavior.
    - `"regime"`: block flip exits while `regime_dir == open_dir` (bias still supports).
    - `"permission"`: block flip exits while EMA quality gates (spread/slope) still pass.
    - `"regime_or_permission"`: block while either bias or permission supports.
    - `"regime_and_permission"`: block only while both support.
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
  - `spot_exec_bar_size` (optional; default `null`)
    - When set (e.g. `"5 mins"`), spot backtests run signals on the main `backtest.bar_size` (30m/1h/etc),
      but simulate execution + PT/SL/flip exits using the smaller execution bars.
    - This is a “multi-resolution” mode intended to reduce coarse intrabar pessimism on large bars.
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
    - `spot_short_risk_mult`: scales `spot_risk_pct` sizing for short (SELL) entries (default `1.0`)
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
- `ema_spread_min_pct_down`: optional stricter spread gate for short entries (applies only when `signal.entry_dir=down`)
- `ema_slope_min_pct`: require EMA fast slope (%) above threshold
- `volume_ratio_min`: require `volume / EMA(volume)` above threshold
- `volume_ema_period`: EMA period for volume gating (default `20` when `volume_ratio_min` is set)
- `entry_start_hour`, `entry_end_hour`: hourly window (0–23)
- `entry_start_hour_et`, `entry_end_hour_et`: hourly window in ET (0–23). If set, preferred over `entry_start_hour/entry_end_hour`.
- `skip_first_bars`: skip first N bars of each session
- `cooldown_bars`: minimum bars between entries
- `shock_gate_mode`: `"off"` | `"detect"` | `"block"` | `"block_longs"` | `"block_shorts"` | `"surf"`
  - `"detect"` computes the shock state (for sizing/logging) but does not block entries.
  - `"block*"` modes block new entries when the shock state is ON (still manage exits normally).
  - `"surf"` blocks entries *against* the shock direction during shock regimes (lets you ride both the shock-down and the shock-up reversal).
- `shock_detector`: `"atr_ratio"` | `"daily_atr_pct"` (default `"atr_ratio"`)
  - `"atr_ratio"` uses `shock_atr_fast_period/shock_atr_slow_period` + `shock_on_ratio/shock_off_ratio` (+ `shock_min_atr_pct`) to flag shocks.
  - `"daily_atr_pct"` uses `shock_daily_*` (below) to flag shocks based on intraday-estimated **daily ATR%** (Wilder ATR on daily TR).
- `shock_atr_fast_period`, `shock_atr_slow_period`: ATR periods for shock detection (fast/slow).
- `shock_on_ratio`, `shock_off_ratio`: hysteresis thresholds for `(ATR_fast / ATR_slow)` to turn shock ON/OFF.
- `shock_min_atr_pct`: requires `ATR_fast / close * 100` to exceed this when turning shock ON.
- `shock_daily_atr_period`: daily ATR period (default `14`) when `shock_detector="daily_atr_pct"`.
- `shock_daily_on_atr_pct`, `shock_daily_off_atr_pct`: hysteresis thresholds for daily `ATR%` to turn shock ON/OFF when `shock_detector="daily_atr_pct"`.
- `shock_daily_on_tr_pct`: optional *additional* shock-ON trigger for sudden range expansion / gaps: `TrueRange_so_far / prev_day_close * 100 >= shock_daily_on_tr_pct`.
- `shock_direction_source`: `"regime"` | `"signal"` (default `"regime"`). When running a multi-timeframe regime (e.g. Supertrend on `4 hours` with signals on `30 mins`), this controls which bar stream drives `shock_dir`.
- `shock_direction_lookback`: bars used for smoothed shock direction (used by `"surf"` and directional shock sizing).
- `shock_short_risk_mult_factor`: scales `spot_short_risk_mult` during shock when `shock_dir=down` (only affects `spot_sizing_mode=risk_pct`).
- `shock_long_risk_mult_factor`: scales `spot_risk_pct` sizing during shock when `shock_dir=up` (only affects `spot_sizing_mode=risk_pct`).
- `shock_long_risk_mult_factor_down`: scales `spot_risk_pct` sizing during shock when `shock_dir=down` (only affects `spot_sizing_mode=risk_pct`).
- `shock_stop_loss_pct_mult`: scales `spot_stop_loss_pct` during shock (pct exit mode only).
- `shock_profit_target_pct_mult`: scales `spot_profit_target_pct` during shock (pct exit mode only).

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
- For options-style runs, `max_entries_per_day=0` and `open_position_cap=0` reflect **stacking / pyramiding** subject to `starting_cash` and margin.

Full leaderboard is in `tradebot/backtest/LEADERBOARD.md`.
Machine-readable presets are in `tradebot/backtest/leaderboard.json` (regenerate with `python -m tradebot.backtest options_leaderboard`; spot milestones are appended by default).

## Spot leaderboard (MNQ, 12m)
Spot presets are stored in:
- `tradebot/backtest/spot_champions.json` (curated CURRENT champs for UI/live), and
- `tradebot/backtest/spot_milestones.json` (broader spot preset pool).

Both are merged into the TUI presets list alongside the options leaderboard.

Large historical milestone pools / variants live under `backtests/out/presets/` (they are not loaded by the UI).

These spot presets are **12-month only** (no 6m snapshot entries) and are filtered for stability:
- win rate `>= 55%`
- trades `>= 200`
- pnl/dd `>= 8`
- sorted by pnl/dd (desc)
- spot position mode is parity-locked to a single net position (the `open_position_cap` field is ignored for spot)

Regenerate (offline, uses cached bars in `db/`; realism is default):
```bash
python -m tradebot.backtest spot --offline --axis all --write-milestones --cache-dir db
python -m tradebot.backtest spot --offline --axis combo_full --combo-full-preset gate_matrix --write-milestones --merge-milestones --cache-dir db
```

LEGACY / debugging: disable realism (optimistic fills; not recommended for decision-making):
```bash
python -m tradebot.backtest spot --offline --axis all --no-realism2 --write-milestones --cache-dir db
python -m tradebot.backtest spot --offline --axis combo_full --combo-full-preset gate_matrix --no-realism2 --write-milestones --merge-milestones --cache-dir db
```

Regenerate 30-minute spot champions (adds only a curated top set from that run, merges into existing milestones):
```bash
python -m tradebot.backtest spot --offline --bar-size "30 mins" --axis combo_full --combo-full-preset gate_matrix \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 25 --milestone-add-top-pnl 25 \
  --cache-dir db
```

For deeper “one axis at a time” exploration (still spot-only), run a single axis:
```bash
python -m tradebot.backtest spot --axis regime --cache-dir db
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
- EMA spread DOWN gate (`--axis spread_down`): `ema_spread_min_pct_down ∈ {None, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015, 0.02, 0.03, 0.05}` (applies only when `signal.entry_dir=down`)
- EMA slope gate (`--axis slope`): `ema_slope_min_pct ∈ {None, 0.005, 0.01, 0.02, 0.03, 0.05}`
- EMA signed-slope gate (`--axis slope_signed`): `ema_slope_signed_min_pct_up/down ∈ {None, 0.003..0.05 (curated)}` (directional: “steep up” vs “steep down”)
- Volume gate (`--axis volume`): `volume_ratio_min ∈ {None, 1.0, 1.1, 1.2, 1.5}`, `volume_ema_period ∈ {10,20,30}`
- Realized-vol band gate (`--axis rv`): `rv_min ∈ {None, 0.25,0.30,0.35,0.40,0.45}`, `rv_max ∈ {None, 0.70,0.80,0.90,1.00}`
- Cooldown (`--axis cooldown`): `cooldown_bars ∈ {0,1,2,3,4,6,8}`
- Skip-open (`--axis skip_open`): `skip_first_bars ∈ {0,1,2,3,4,6}`
- $TICK gate (“Raschke width”, `--axis tick`):
  - `z_enter ∈ {0.8,1.0,1.2}`, `z_exit ∈ {0.4,0.5,0.6}`, `slope_lookback ∈ {3,5}`, `z_lookback ∈ {126,252}`
  - `tick_neutral_policy ∈ {allow, block}`, `tick_direction_policy ∈ {both, wide_only}`
  - Default symbol tries `TICK-AMEX` (fallback from `TICK-NYSE` if permissions/cache block it)
- Shock overlay (`--axis shock`): `shock_gate_mode ∈ {detect, block, block_longs, block_shorts, surf}` × `shock_detector ∈ {atr_ratio, tr_ratio, daily_atr_pct, daily_drawdown}` plus curated threshold presets (includes `shock_daily_on_tr_pct` for “TR early trigger”), with small grids for `shock_stop_loss_pct_mult`, `shock_profit_target_pct_mult`, and `shock_short_risk_mult_factor`
- TR% risk overlays (`--axis risk_overlays`): risk-off `riskoff_tr5_med_pct`; risk-panic `riskpanic_tr5_med_pct + riskpanic_neg_gap_ratio_min` with `riskpanic_short_risk_mult_factor ∈ {1.0,0.5,0.2,0.0}` (and optional `riskpanic_long_risk_mult_factor` for long sizing shrink/block on panic days); risk-pop `riskpop_tr5_med_pct + riskpop_pos_gap_ratio_min` with `riskpop_long_risk_mult_factor ∈ {0.6,0.8,1.0,1.2,1.5}` and `riskpop_short_risk_mult_factor ∈ {1.0,0.5,0.2,0.0}`
  - Optional “gap magnitude” tightening: `riskpanic_neg_gap_abs_pct_min`, `riskpop_pos_gap_abs_pct_min` (only count gaps with `|gap|>=threshold` toward the gap ratio).
  - Optional “pre-shock ramp” detector: `riskpanic_tr5_med_delta_min_pct`/`riskpanic_tr5_med_delta_lookback_days` (and riskpop equivalents), requiring TR-median to be accelerating, not just high.
  - Includes optional `risk_entry_cutoff_hour_et`.

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
- Regime2×TOD joint (`--axis r2_tod`): shortlist regime2 settings (`ST2 @ {4h,1d}`; see `run_backtests_spot_sweeps.py`) → sweep TOD windows (RTH + overnight micro-grid)
- Flip-exit semantics (`--axis flip_exit`): `exit_on_signal_flip ∈ {on,off}`, `flip_exit_mode ∈ {entry,state,cross}`, `hold ∈ {0,2,4,6}`, `only_if_profit ∈ {0,1}`
- Loosenings (`--axis loosen`): `spot_close_eod ∈ {0,1}` (spot is always single-position)
- Loosen×ATR joint (`--axis loosen_atr`): `spot_close_eod ∈ {0,1}` × exits `ATR period ∈ {10,14,21}`, `PTx ∈ {0.60..0.80 step 0.05}`, `SLx ∈ {1.20..2.00 step 0.20}` (spot is always single-position)
- Exit-time flatten (`--axis exit_time`): `spot_exit_time_et ∈ {None, 04:00, 09:30, 10:00, 11:00, 16:00, 17:00}`
- Short sizing multiplier (`--axis spot_short_risk_mult`): `spot_short_risk_mult ∈ {1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.15,0.1,0.05,0.02,0.01,0.0}` (only affects `spot_sizing_mode=risk_pct`)

**ORB**
- ORB (`--axis orb`, runs on 15m bars): `open_time_et ∈ {09:30, 18:00}`, `window_mins ∈ {15,30,60}`, `target_mode ∈ {rr, or_range}`, `rr ∈ {0.618,0.707,0.786,1.0,1.272,1.618,2.0}`, optional `vol>=1.2@20`, plus a session TOD filter.
- ORB joint (`--axis orb_joint`, runs on 15m bars): stage-1 shortlist of ORB params → apply `regime ∈ {off, ST @ 4h (small curated set)}` × `tick ∈ {off, wide_only allow/block (z=1.0/0.5 slope=3 lb=252)}`

**Known gaps (we now target explicitly)**
- Some interaction edges require **joint sweeps** rather than one-axis sweeps (e.g. `regime2 × ATR exits` with `PTx < 1.0`): this is the class of gap compact preset funnels can miss, and is now covered by `--axis r2_atr`.
- `--axis combo_full --combo-full-preset gate_matrix` is the bounded compact funnel for gate cross-products around a stable seed surface.
- `--axis gate_matrix` is a bounded cross-product around a seed shortlist: `{perm,tick,shock,riskoff,riskpanic,riskpop,regime2} on/off` × `spot_short_risk_mult` (intended to finish overnight).
- `--axis combo_full --combo-full-preset hf_timing_sniper` is the tight HF timing corridor: 6 timing variants centered on the known rank-cross pocket.
- `--axis combo_full` runs the full sweep suite (single-axis + joint sweeps + gate cross-product). It is intentionally **very slow** and is meant for overnight discovery runs. Tip: with `--offline`, you can use `--jobs N` (defaults to CPU count) to parallelize per-axis sweeps and cut wall-clock time.


### Spot champions (moved)

TQQQ champion stacks and evolution logs were moved to:
- `backtests/tqqq/readme-lf.md`
- `backtests/tqqq/readme-hf.md`

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
  - Position sizing + ROI reporting (implemented in Realism v2): `spot_sizing_mode`, `spot_risk_pct`, `spot_short_risk_mult`, `spot_notional_pct`, `spot_max_notional_pct`, `spot_min_qty`, `spot_max_qty`
    - `roi = pnl / starting_cash`, `dd% = max_drawdown / starting_cash`
  - ET session/day boundaries (implemented): `max_entries_per_day`, riskoff cutoffs, pending-entry carry/cancel, daily overlays, and option DTE/day anchors now use ET trade dates/hours.
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


