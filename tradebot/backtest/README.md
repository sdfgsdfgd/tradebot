# Backtest Engine (Synthetic Options + Spot)

This package provides a minimal backtest runner that supports:
- **synthetic options** (builds synthetic option prices from underlying bars), and
- **spot** (underlying-only strategies with realism knobs like intrabar exits / execution fills).

Shared decision logic lives in `tradebot/engine.py` and `tradebot/spot_engine.py` so the UI/live runner can reuse the exact same semantics.

## Spot (TQQQ) — CURRENT champ (v34) quick reproduce

This is the canonical “champ smoke test” we use to ensure refactors do **not** change outcomes.

```bash
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v34_champ_only_milestone.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 1 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 1 --out /tmp/tqqq_exec5m_v34_smoke.json
```

Expected exact window metrics:
- 10y: roi `0.6038486539374943`, dd_pct `0.17072433918750088`, pnl `60384.86539374943`, trades `1054`
- 2y:  roi `0.2972119843749967`, dd_pct `0.045420233000000296`, pnl `29721.19843749967`, trades `205`
- 1y:  roi `0.13781497912499804`, dd_pct `0.03944176150000014`, pnl `13781.497912499804`, trades `107`

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
- When `max_entries_per_day=0` and `max_open_trades=0`, results reflect **stacking / pyramiding** subject to `starting_cash` and margin.

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

Regenerate (offline, uses cached bars in `db/`):
```bash
python -m tradebot.backtest spot --offline --axis all --write-milestones --cache-dir db
python -m tradebot.backtest spot --offline --axis combo_fast --write-milestones --merge-milestones --cache-dir db
```

Regenerate with realism v1 enabled (recommended before live trading):
```bash
python -m tradebot.backtest spot --offline --axis all --realism --write-milestones --cache-dir db
python -m tradebot.backtest spot --offline --axis combo_fast --realism --write-milestones --merge-milestones --cache-dir db
```

Regenerate 30-minute spot champions (adds only a curated top set from that run, merges into existing milestones):
```bash
python -m tradebot.backtest spot --offline --bar-size "30 mins" --axis combo_fast \
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
- TR% risk overlays (`--axis risk_overlays`): risk-off `riskoff_tr5_med_pct`; risk-panic `riskpanic_tr5_med_pct + riskpanic_neg_gap_ratio_min` with `riskpanic_short_risk_mult_factor ∈ {1.0,0.5,0.2,0.0}`; risk-pop `riskpop_tr5_med_pct + riskpop_pos_gap_ratio_min` with `riskpop_long_risk_mult_factor ∈ {0.6,0.8,1.0,1.2,1.5}` and `riskpop_short_risk_mult_factor ∈ {1.0,0.5,0.2,0.0}` (includes optional `risk_entry_cutoff_hour_et`)

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
- Loosenings (`--axis loosen`): `max_open_trades ∈ {1,2,3,0}`, `spot_close_eod ∈ {0,1}`
- Loosen×ATR joint (`--axis loosen_atr`): `max_open_trades ∈ {2,3,0}`, `spot_close_eod ∈ {0,1}` × exits `ATR period ∈ {10,14,21}`, `PTx ∈ {0.60..0.80 step 0.05}`, `SLx ∈ {1.20..2.00 step 0.20}`
- Exit-time flatten (`--axis exit_time`): `spot_exit_time_et ∈ {None, 04:00, 09:30, 10:00, 11:00, 16:00, 17:00}`
- Short sizing multiplier (`--axis spot_short_risk_mult`): `spot_short_risk_mult ∈ {1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.15,0.1,0.05,0.02,0.01,0.0}` (only affects `spot_sizing_mode=risk_pct`)

**ORB**
- ORB (`--axis orb`, runs on 15m bars): `open_time_et ∈ {09:30, 18:00}`, `window_mins ∈ {15,30,60}`, `target_mode ∈ {rr, or_range}`, `rr ∈ {0.618,0.707,0.786,1.0,1.272,1.618,2.0}`, optional `vol>=1.2@20`, plus a session TOD filter.
- ORB joint (`--axis orb_joint`, runs on 15m bars): stage-1 shortlist of ORB params → apply `regime ∈ {off, ST @ 4h (small curated set)}` × `tick ∈ {off, wide_only allow/block (z=1.0/0.5 slope=3 lb=252)}`

**Known gaps (we now target explicitly)**
- Some interaction edges require **joint sweeps** rather than one-axis sweeps (e.g. `regime2 × ATR exits` with `PTx < 1.0`): this is the class of gap the combo_fast funnel can miss, and is now covered by `--axis r2_atr`.
- `--axis combo_fast` is a bounded “smart exhaustive” funnel that includes direction×regime scan + the low-PT ATR pocket + Raschke `$TICK` gate + RV band + exit-time + a few TOD windows.
- `--axis gate_matrix` is a bounded cross-product around a seed shortlist: `{perm,tick,shock,riskoff,riskpanic,riskpop,regime2} on/off` × `spot_short_risk_mult` (intended to finish overnight).
- `--axis champ_refine` is a seeded “last-mile” funnel: load top-K seeds from `--seed-milestones`, then run a bounded refinement around each seed (short asymmetry tail + **exit semantics micro** (stop-only + flip-profit gating) + TOD/permission micro + signed-slope + shock/risk pockets).
- `--axis combo_full` runs the full sweep suite (single-axis + joint sweeps + combo_fast + gate cross-product). It is intentionally **very slow** and is meant for overnight discovery runs. Tip: with `--offline`, you can use `--jobs N` (defaults to CPU count) to parallelize per-axis sweeps and cut wall-clock time.

### Spot champions (TQQQ only)

#### CURRENT (v34) (exec=5m, RTH; 10y/2y/1y; >=100 trades/1y; stability-first) (promoted 2026-01-30)
These are the current **TQQQ-only** spot champions under the “multi-resolution” execution model (post-`spot_exec_bar_size=5 mins`) that are:
- Positive PnL in **all 3** windows: **10y + 2y + 1y**
- Active enough for ops (operationalized as `>=100 trades` in the last 1y window)
- Ranked by **stability** (maximize the worst-window `roi/dd`), not by win-rate

Windows:
- 10y: `2016-01-01 → 2026-01-19`
- 2y: `2024-01-01 → 2026-01-19`
- 1y: `2025-01-01 → 2026-01-19`

Execution model:
- Signal generation on `30 mins` bars (RTH)
- Execution + intrabar stop + flip exits simulated on `5 mins` bars (RTH) via `spot_exec_bar_size=5 mins`

Winning family (what changed vs the legacy high-WR pocket):
- **Stop-only + reversal exit:** `spot_profit_target_pct=None` + `spot_stop_loss_pct≈4.0–4.5%` + `exit_on_signal_flip=true` (exit on reversal, not on a fixed PT).
- **Don’t flip-exit at a loss:** `flip_exit_only_if_profit=true` + a small debounce (`flip_exit_min_hold_bars≈2..6`) to avoid immediate whipsaws.
- **Permission gates actually matter:** `ema_spread_min_pct≈0.003` + `ema_slope_min_pct≈0.014..0.035` improves the worst-window `roi/dd` notably.
- **Asymmetric shorts:** keep two-way behavior, but reduce short risk (`spot_short_risk_mult≈0.01..0.02`) plus short gating (`ema_spread_min_pct_down≈0.04..0.06`).
- **Session/TOD activity filter:** limiting entries to an RTH window (CURRENT best: `9–16 ET`) improved worst-window `roi/dd` while preserving activity.
- **Shock overlay (CURRENT):** TR-ratio (“vol velocity”) shock detector + `"detect"` mode + risk scaling (plus a more sensitive TR-ratio 3/21 on/off=1.30/1.20, minTR%=5) improved worst-window stability without changing the base signal/regime family.

Top set:
- CURRENT (v34): `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json`
- Previous CURRENT (v33): `backtests/out/tqqq_exec5m_v33_shock_alpha_refine_30m_10y2y1y_mintr100_top100.json`
- Previous CURRENT (v32): `backtests/out/tqqq_exec5m_v32_dethrone_30m_10y2y1y_mintr100_top100.json`
- Previous CURRENT (v31): `backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json`

Search log / reproducibility (kept as files so we don’t rerun blindly):
- v34 shock_alpha_refine_wide (seeded from v33; `--axis shock_alpha_refine` widened TR-ratio grid + full kingmaker prefilter):
  - `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_variants_30m.json` → `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json`
  - Key diffs vs v33: TR-ratio `3/21` with `on/off=1.30/1.20`, `minTR%=5` (triggers earlier in crash ramps); stability floor `~3.49` (vs v33 `~3.21`).
- v33 shock_alpha_refine (seeded from v32; `--axis shock_alpha_refine` shock monetization micro grid):
  - `backtests/out/tqqq_exec5m_v33_shock_alpha_refine_variants_30m.json` → `backtests/out/tqqq_exec5m_v33_shock_alpha_refine_30m_10y2y1y_mintr100_top100.json`
  - Key diffs vs v32: `shock_gate_mode=detect`, `shock_detector=tr_ratio`, `shock_atr_fast/slow=5/50`, `shock_on/off_ratio=1.45/1.30`, `shock_min_atr_pct=7`, risk scaling (`target_atr_pct=12`, `min_mult=0.2`).
- v32 dethrone (seeded from v31; `--axis champ_refine` joint micro grid):
  - `backtests/out/tqqq_exec5m_v32_dethrone_variants_30m.json` → `backtests/out/tqqq_exec5m_v32_dethrone_30m_10y2y1y_mintr100_top100.json`
  - Winning deltas vs v31: TOD `9–16` ET (was `10–15`), `ema_spread_min_pct_down=0.04` (was `0.05`), `spot_stop_loss_pct=0.045` (was `0.04`).
- Exit semantics joint sweep (EMA preset + PT/SL + flip-exit variants):
  - `backtests/out/tqqq_exec5m_exit_joint_30m_variants.json` → `backtests/out/tqqq_exec5m_exit_joint_30m_2y_top400.json` → `backtests/out/tqqq_exec5m_exit_joint_30m_10y2y1y_mintr100_champs_top80.json`
- Permission joint sweep (spread/slope + flip gate):
  - `backtests/out/tqqq_exec5m_perm_gate_joint_30m_variants.json` → `backtests/out/tqqq_exec5m_perm_gate_joint_30m_10y2y1y_mintr100_champs_top80.json`
- Bias micro sweep (ST neighborhood):
  - `backtests/out/tqqq_exec5m_bias_micro_30m_variants.json` → `backtests/out/tqqq_exec5m_bias_micro_30m_10y2y1y_mintr100_champs_top80.json`
- Exit micro sweep (hold/gate/PT/SL neighborhood around bias champ):
  - `backtests/out/tqqq_exec5m_exit_micro2_30m_variants.json` → `backtests/out/tqqq_exec5m_exit_micro2_30m_10y2y1y_mintr100_champs_top80.json`
- Evolve v4 (ST×SL×hold×gate×short-risk; PT=None family):
  - `backtests/out/tqqq_exec5m_evolve_v4_variants.json` → `backtests/out/tqqq_exec5m_evolve_v4_2y_top500.json` → `backtests/out/tqqq_exec5m_evolve_v4_10y2y1y_mintr100_champs_top80.json`
- Evolve v5 (fine neighborhood around v4 champ):
  - `backtests/out/tqqq_exec5m_evolve_v5_fine_variants.json` → `backtests/out/tqqq_exec5m_evolve_v5_fine_2y_top300.json` → `backtests/out/tqqq_exec5m_evolve_v5_fine_10y2y1y_mintr100_champs_top80.json`
- v6 activity/session filters (entry_start/end_hour_et × skip_first_bars × cooldown_bars around the v5 champ):
  - `backtests/out/tqqq_exec5m_v6_activity_filters_variants.json` → `backtests/out/tqqq_exec5m_v6_activity_filters_10y2y1y_mintr100_top80.json`
- v7a EMA timing variants (ema_preset × entry_mode × entry_confirm; TOD fixed 10–15 ET):
  - `backtests/out/tqqq_exec5m_v7a_ema_entry_variants_30m.json` → `backtests/out/tqqq_exec5m_v7a_ema_entry_30m_10y2y1y_mintr100_top80.json`
- v7b high-activity attempt (TOD off; flip aggressiveness + small SL neighborhood; targeted `>=200 trades/1y`):
  - `backtests/out/tqqq_exec5m_v7b_high_activity_variants_30m.json` → `backtests/out/tqqq_exec5m_v7b_high_activity_30m_10y2y1y_mintr200_top80.json`
- v7c flip-exit micro sweep (flip_exit_gate_mode × hold; TOD fixed 10–15 ET):
  - `backtests/out/tqqq_exec5m_v7c_flip_exit_gate_variants_30m.json` → `backtests/out/tqqq_exec5m_v7c_flip_exit_gate_30m_10y2y1y_mintr100_top80.json`
- v8a bias neighborhood (ST ATR × mult × src around the v7c champ):
  - `backtests/out/tqqq_exec5m_v8a_bias_neighborhood_variants_30m.json` → `backtests/out/tqqq_exec5m_v8a_bias_neighborhood_30m_10y2y1y_mintr100_top80.json`
- v8b high-activity rescue (min-trades=200 on 1y, then 10y/2y/1y eval; did **not** produce stable winners):
  - `backtests/out/tqqq_exec5m_v8b_high_activity_rescue_variants_30m.json`
  - 1y prefilter: `backtests/out/tqqq_exec5m_v8b_high_activity_rescue_1y_mintr200_top600.json`
  - 10y/2y/1y eval: `backtests/out/tqqq_exec5m_v8b_high_activity_rescue_30m_10y2y1y_mintr200_top80.json`
- v8c permission micro sweep (spread_min × slope_min × short_spread_min; TOD fixed 10–15 ET):
  - `backtests/out/tqqq_exec5m_v8c_perm_micro_variants_30m.json` → `backtests/out/tqqq_exec5m_v8c_perm_micro_30m_10y2y1y_mintr100_top80.json`
- v9a gate + short-risk micro sweep (slope_min × short_spread_min × short_risk; TOD fixed 10–15 ET):
  - `backtests/out/tqqq_exec5m_v9a_gate_shortrisk_micro_variants_30m.json` → `backtests/out/tqqq_exec5m_v9a_gate_shortrisk_micro_30m_10y2y1y_mintr100_top80.json`
- v9b exit accuracy sweep (flip_exit gate × hold × exit_time; seeded from v9a):
  - `backtests/out/tqqq_exec5m_v9b_exit_accuracy_variants_30m.json` → `backtests/out/tqqq_exec5m_v9b_exit_accuracy_30m_10y2y1y_mintr100_top80.json`
- v9c high-slope sweep (ema_slope_min_pct high values; seeded from v9a):
  - `backtests/out/tqqq_exec5m_v9c_high_slope_variants_30m.json` → `backtests/out/tqqq_exec5m_v9c_high_slope_30m_10y2y1y_mintr100_top80.json`
- v11a PT/hold pocket around v9c (pt × sl × hold; PT was mostly a decoy — the win came from lowering hold):
  - `backtests/out/tqqq_exec5m_v11a_pt_pocket_variants_30m.json` → `backtests/out/tqqq_exec5m_v11a_pt_pocket_30m_10y2y1y_mintr100_top80.json`
- v11b stability-guarded bias squeeze around v9c (Supertrend ATR × mult × src × hold):
  - `backtests/out/tqqq_exec5m_v11b_st_hold_neighborhood_variants_30m.json` → `backtests/out/tqqq_exec5m_v11b_st_hold_neighborhood_30m_10y2y1y_mintr100_top80.json`
- v14 micro-squeeze around v11b (ST mult × hold × slope):
  - `backtests/out/tqqq_exec5m_v14_micro_squeeze_variants_30m.json` → `backtests/out/tqqq_exec5m_v14_micro_squeeze_30m_10y2y1y_mintr100_top80.json`
- v25 daily ATR% shock overlay (beats v11b roi/dd in all 3 windows):
  - `backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_variants_30m.json` → `backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_30m_10y2y1y_mintr100_top80.json`
- v31 shock threshold squeeze (beats v25 roi/dd in all 3 windows):
  - `backtests/out/tqqq_exec5m_v31_shock_block_longs_variants_30m.json` → `backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json`
- v32 v31-seeded champ_refine joint micro grid (beats v31 roi/dd in all 3 windows):
  - `backtests/out/tqqq_exec5m_v32_dethrone_variants_30m.json` → `backtests/out/tqqq_exec5m_v32_dethrone_30m_10y2y1y_mintr100_top100.json`
  - Key diffs vs v31: TOD `9–16` ET (was `10–15`), `ema_spread_min_pct_down=0.04` (was `0.05`), `spot_stop_loss_pct=0.045` (was `0.04`).
- v33 shock_alpha_refine (beats v32 roi/dd in all 3 windows):
  - `backtests/out/tqqq_exec5m_v33_shock_alpha_refine_variants_30m.json` → `backtests/out/tqqq_exec5m_v33_shock_alpha_refine_30m_10y2y1y_mintr100_top100.json`
  - Key diffs vs v32: TR-ratio shock detection (`detect` mode) + risk scaling (`target_atr_pct=12`, `min_mult=0.2`).
- v34 shock_alpha_refine_wide (beats v33 roi/dd in all 3 windows):
  - `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_variants_30m.json` → `backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json`
  - Key diffs vs v33: more sensitive TR-ratio shock (`3/21`, `on/off=1.30/1.20`, `minTR%=5`).
  - Stress slices vs v31 (not part of kingmaker filter): better in Feb→Apr 2025 and 2022; worse in Feb→Apr 2020 crash.
- (Exploratory, not promoted) shock + regime investigations (summary; none beat the CURRENT champ stability):
  - Shock overlay variants tried (modes: `detect|block|block_longs|surf`; direction source: `regime|signal`):
    - `daily_atr_pct`: `on_atr_pct ∈ {13.5,14.0,14.5}`, `off_atr_pct ∈ {12.5,13.0}`, `sl_mult ∈ {0.75,1.0,1.25}`
    - optional TR%-trigger (sticky day): `on_tr_pct ∈ {9.0,9.5,10.0,11.0,12.0,14.0,16.0}`
    - optional shock risk scaling: `target_atr_pct ∈ {10,12,14}`, `min_mult ∈ {0.2,0.3,0.4}`
    - `daily_drawdown`: `lookback ∈ {20,40}`, `on_dd ∈ {-25,-20,-15,-10}`, `off_dd = on_dd + 5`
    - optional shock short scaling: `short_factor ∈ {2,3,5,8}` (still not enough to justify the stability hit on TQQQ)
  - Representative outputs (all “not promoted” unless marked CURRENT/Previous above):
    - `backtests/out/tqqq_exec5m_v27_shock_st_mult_30m_10y2y1y_mintr100_top48.json`, `backtests/out/tqqq_exec5m_v28_shock_dir_sizing_30m_10y2y1y_mintr100_top80.json`, `backtests/out/tqqq_exec5m_v29_atr_risk_scale_30m_10y2y1y_mintr100_top12.json`
    - `backtests/out/tqqq_exec5m_v32_trpct_trigger_30m_10y2y1y_mintr100_top12.json`, `backtests/out/tqqq_exec5m_v34_shock_slmult_30m_10y2y1y_mintr100_top6.json`, `backtests/out/tqqq_exec5m_v38_daily_drawdown_shock_30m_10y2y1y_mintr100_top9.json`
    - `backtests/out/tqqq_exec5m_v41_trpct_shock_trigger_30m_10y2y1y_mintr100_top8.json`, `backtests/out/tqqq_exec5m_v42_trpct_shortscale_30m_10y2y1y_mintr100_top13.json`
  - Supertrend “TV 10/3” sanity check:
    - Feb→Apr 2025 slice: ST(4h,10,3) flips once and stays down, but using it as a global bias gate or a regime2 “permission gate” loses across 10y/2y/1y in this engine.
    - Outputs: `backtests/out/tqqq_exec5m_v36_supertrend_tv10_3_neighborhood_30m_10y2y1y_all_top19.json`, `backtests/out/tqqq_exec5m_v43_bias_off_regime2_st10_3_30m_10y2y1y_all_top6.json`
- v10e high-activity (>=150 trades/1y) sweep (flip-exit strictness; PT=None):
  - `backtests/out/tqqq_exec5m_v10e_flip_exit_strictness_variants_30m.json` → `backtests/out/tqqq_exec5m_v10e_flip_exit_strictness_30m_10y2y1y_mintr150_top80.json`
- v10i high-activity bias neighborhood (ST ATR × mult × src around v10e #1):
  - `backtests/out/tqqq_exec5m_v10i_st_neighborhood_variants_30m.json` → `backtests/out/tqqq_exec5m_v10i_st_neighborhood_30m_10y2y1y_mintr150_top80.json`
- v10h (exploratory) >=300 trades/1y feasibility check:
  - 1y prefilter: `backtests/out/tqqq_exec5m_v10h_v8b_universe_1y_mintr300_top300.json`
  - 10y/2y/1y eval (none passed @ min_trades=300): `backtests/out/tqqq_exec5m_v10h_v8b_universe_30m_10y2y1y_mintr300_top80.json`

Repro commands:
```bash
# NOTE: prepend new evolution entries ABOVE the CURRENT block (0g6).
# 0g8) CURRENT (v34): v33-seeded shock_alpha_refine_wide joint micro grid (beats v33 roi/dd in all 3 windows)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 100 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 100 --out backtests/out/tqqq_exec5m_v34_shock_alpha_refine_wide_30m_10y2y1y_mintr100_top100.json

# 0g7) Previous CURRENT (v33): v32-seeded shock_alpha_refine joint micro grid (beats v32 roi/dd in all 3 windows)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v33_shock_alpha_refine_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 100 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 100 --out backtests/out/tqqq_exec5m_v33_shock_alpha_refine_30m_10y2y1y_mintr100_top100.json

# 0g6) Previous CURRENT (v32): v31-seeded champ_refine joint micro grid (beats v31 roi/dd in all 3 windows)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v32_dethrone_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 100 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 100 --out backtests/out/tqqq_exec5m_v32_dethrone_30m_10y2y1y_mintr100_top100.json

# 0g5) Previous CURRENT (v31): earlier daily ATR% shock-on threshold (beats v25 roi/dd in all 3 windows)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v31_shock_block_longs_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 36 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 36 --out backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json

# 0g4) Previous CURRENT (v25): daily ATR% shock overlay (beats v11b roi/dd in all 3 windows)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 144 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_30m_10y2y1y_mintr100_top80.json

# 0g3) Near-miss (v14): micro-squeeze around v11b (ST mult × hold × slope; improved 10y/2y, 1y unchanged)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v14_micro_squeeze_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 63 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v14_micro_squeeze_30m_10y2y1y_mintr100_top80.json

# 0g2) Previous CURRENT (v11b): ST bias squeeze + hold
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v11b_st_hold_neighborhood_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 80 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v11b_st_hold_neighborhood_30m_10y2y1y_mintr100_top80.json

# 0g1) v11a: PT/SL/hold pocket around v9c (mostly found that hold=2 helps)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v11a_pt_pocket_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 54 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v11a_pt_pocket_30m_10y2y1y_mintr100_top80.json

# 0i) Latest (v10e): high-activity (min_trades=150) flip-exit strictness sweep (PT=None)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v10e_flip_exit_strictness_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 96 --max-open 1 \
  --require-positive-pnl --min-trades 150 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v10e_flip_exit_strictness_30m_10y2y1y_mintr150_top80.json

# 0i2) Latest (v10i): bias neighborhood around the v10e top (ST ATR × mult × src)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v10i_st_neighborhood_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 40 --max-open 1 \
  --require-positive-pnl --min-trades 150 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v10i_st_neighborhood_30m_10y2y1y_mintr150_top80.json

# 0j) Exploratory (v10h): >=300 trades/1y feasibility on 1y-only (wide universe)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v8b_high_activity_rescue_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 3888 --max-open 1 \
  --require-positive-pnl --min-trades 300 --window 2025-01-01:2026-01-19 \
  --write-top 300 --out backtests/out/tqqq_exec5m_v10h_v8b_universe_1y_mintr300_top300.json

# 0k) Exploratory (v10h): multiwindow eval @ min_trades=300 (none passed)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v10h_v8b_universe_1y_mintr300_top300.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 300 --max-open 1 \
  --require-positive-pnl --min-trades 300 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v10h_v8b_universe_30m_10y2y1y_mintr300_top80.json

# 0g) Latest (v9c): high-slope permission gate squeeze (ema_slope_min_pct only)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v9c_high_slope_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 13 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v9c_high_slope_30m_10y2y1y_mintr100_top80.json

# 0f) Latest (v9b): exit accuracy sweep (gate × hold × exit_time; seeded from v9a)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v9b_exit_accuracy_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 27 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v9b_exit_accuracy_30m_10y2y1y_mintr100_top80.json

# 0e) Latest (v9a): permission/short neighborhood squeeze (slope_min × short_spread_min × short_risk)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v9a_gate_shortrisk_micro_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 54 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v9a_gate_shortrisk_micro_30m_10y2y1y_mintr100_top80.json

# 0d) Latest (v8c): permission micro sweep (small)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v8c_perm_micro_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 80 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v8c_perm_micro_30m_10y2y1y_mintr100_top80.json

# 0c) Latest (v8a): bias neighborhood micro sweep (small)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v8a_bias_neighborhood_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 54 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v8a_bias_neighborhood_30m_10y2y1y_mintr100_top80.json

# 0b) Latest (v7c): flip-exit micro sweep (very small)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v7c_flip_exit_gate_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 15 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v7c_flip_exit_gate_30m_10y2y1y_mintr100_top80.json

# (v8b, exploratory) High-activity rescue attempt:
# Stage 1: 1y prefilter @ min_trades=200 (fast)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v8b_high_activity_rescue_variants_30m.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 3888 --max-open 1 \
  --require-positive-pnl --min-trades 200 --window 2025-01-01:2026-01-19 \
  --write-top 600 --out backtests/out/tqqq_exec5m_v8b_high_activity_rescue_1y_mintr200_top600.json

# Stage 2: 10y/2y/1y stability scoring on the 1y-shortlist
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v8b_high_activity_rescue_1y_mintr200_top600.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 600 --max-open 1 \
  --require-positive-pnl --min-trades 200 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v8b_high_activity_rescue_30m_10y2y1y_mintr200_top80.json

# 0) Latest (v6): activity/session filter sweep (small, no prefilter needed)
#    - entry_start/end_hour_et ∈ {10,11,12} × {14,15}
#    - skip_first_bars ∈ {0,1,2}, cooldown_bars ∈ {0,1,2,3}
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_v6_activity_filters_variants.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 80 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_v6_activity_filters_10y2y1y_mintr100_top80.json

# 1) Fast 2y prefilter (keeps the multiwindow run small)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_evolve_v4_variants.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 3240 --max-open 1 \
  --require-positive-pnl --min-trades 200 --window 2024-01-01:2026-01-19 \
  --write-top 500 --out backtests/out/tqqq_exec5m_evolve_v4_2y_top500.json

# 2) Multiwindow kingmaker scoring (10y + 2y + 1y)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_evolve_v4_2y_top500.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 500 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_evolve_v4_10y2y1y_mintr100_champs_top80.json

# (Optional, next iteration) Run the v5 fine neighborhood:
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_evolve_v5_fine_variants.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 1512 --max-open 1 \
  --require-positive-pnl --min-trades 200 --window 2024-01-01:2026-01-19 \
  --write-top 300 --out backtests/out/tqqq_exec5m_evolve_v5_fine_2y_top300.json

python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_evolve_v5_fine_2y_top300.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 300 --max-open 1 \
  --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 80 --out backtests/out/tqqq_exec5m_evolve_v5_fine_10y2y1y_mintr100_champs_top80.json
```

- **#1 Best multiwindow roi/dd (promotion requires improvement in all 3 windows):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.50`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Shock overlay:
    - `shock_detector=daily_atr_pct`, `shock_gate_mode=surf`
    - **Key delta vs v25:** `shock_daily_on_atr_pct=13.5` (was `14.0`), `shock_daily_off_atr_pct=13.0`
    - `shock_stop_loss_pct_mult=0.75` (tighten stop during shock)
    - `shock_direction_source=signal`, `shock_direction_lookback=1`
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=2`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v31_shock_block_longs_30m_10y2y1y_mintr100_top36.json`
  - Stats (10y): `tr=1079`, `win=51.1%`, `pnl=+67911.9`, `roi=+67.91%`, `dd%=22.04%`, `roi/dd=3.08`
  - Stats (2y): `tr=207`, `win=54.6%`, `pnl=+27009.3`, `roi=+27.01%`, `dd%=5.94%`, `roi/dd=4.55`
  - Stats (1y): `tr=109`, `win=54.1%`, `pnl=+13060.6`, `roi=+13.06%`, `dd%=4.42%`, `roi/dd=2.96`
  - Feb→Apr 2025 slice check (single-window eval): `backtests/out/tqqq_exec5m_v31_slice_2025-02_2025-04.json`
  - Regime/loss-cluster report: `backtests/out/tqqq_exec5m_v31_kingmaker01_regime_adaptability.md`

- **Near-miss (not promoted; 1y did not improve):**
  - Micro-squeeze around v11b (ST mult × hold × slope):
    - Output: `backtests/out/tqqq_exec5m_v14_micro_squeeze_30m_10y2y1y_mintr100_top80.json`
    - Kingmaker: `Spot (TQQQ) KINGMAKER #10`
    - Improvement: 10y+2y `roi/dd` improved; 1y `roi/dd` stayed at `2.80` with the same trade count (`tr=109`), so it does **not** satisfy the “improve all 3 windows” promotion rule.

#### HIGH-ACTIVITY (exec=5m, RTH; 10y/2y/1y; >=150 trades/1y; PnL/roi-dd first) (generated 2026-01-23)
These are **throughput-first** candidates (higher trade count) under the same multi-resolution execution model.

Important:
- These are **not promoted** to CURRENT unless they **beat** (or at least **match**) the CURRENT champ’s worst-window `roi/dd` while also meeting the higher trade-count constraint.
- In practice, getting `>=150 trades/1y` under this engine often requires loosening `flip_exit_only_if_profit`, which tends to reduce overall stability and win-rate (trade-off).

Top set:
- `backtests/out/tqqq_exec5m_v10i_st_neighborhood_30m_10y2y1y_mintr150_top80.json`

- **#1 Best worst-window roi/dd @ min_trades=150:**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=10`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=9`, `entry_end_hour_et=16` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`
    - **Flip at a loss is allowed:** `flip_exit_only_if_profit=false`
    - `flip_exit_min_hold_bars=6`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v10i_st_neighborhood_30m_10y2y1y_mintr150_top80.json`
  - Stats (10y): `tr=1565`, `win=32.4%`, `pnl=+53265.7`, `roi=+53.27%`, `dd%=20.04%`, `roi/dd=2.66`
  - Stats (2y): `tr=307`, `win=34.9%`, `pnl=+22826.5`, `roi=+22.83%`, `dd%=4.61%`, `roi/dd=4.95`
  - Stats (1y): `tr=162`, `win=34.0%`, `pnl=+7616.9`, `roi=+7.62%`, `dd%=4.04%`, `roi/dd=1.89`

#### ATTEMPTED (>=300 trades/1y) (generated 2026-01-23)
- 1y-only feasibility: `backtests/out/tqqq_exec5m_v10h_v8b_universe_1y_mintr300_top300.json`
  - Does contain positive-PnL candidates with `tr≈329..333` and `roi/dd≈2.62` in the last 1y window.
- Full 10y/2y/1y validation at `min_trades=300`: `backtests/out/tqqq_exec5m_v10h_v8b_universe_30m_10y2y1y_mintr300_top80.json`
  - **0 candidates** passed “positive PnL in all 3 windows” under the same min-trades constraint.

#### LEGACY (exec=5m, RTH; 10y/2y/1y; previous CURRENT + earlier) (generated 2026-01-23; reclassified 2026-01-23)
- **V25 stability champ (previous CURRENT; pre-v31):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.50`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Shock overlay:
    - `shock_detector=daily_atr_pct`, `shock_gate_mode=surf`
    - `shock_daily_on_atr_pct=14.0`, `shock_daily_off_atr_pct=13.0` (hysteresis)
    - `shock_stop_loss_pct_mult=0.75` (tighten stop during shock)
    - `shock_direction_source=signal`, `shock_direction_lookback=1`
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=2`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v25_daily_atr_dynamic_30m_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1079`, `win=51.1%`, `pnl=+67583.6`, `roi=+67.58%`, `dd%=22.03%`, `roi/dd=3.07`
  - Stats (2y): `tr=207`, `win=54.6%`, `pnl=+26748.5`, `roi=+26.75%`, `dd%=5.94%`, `roi/dd=4.50`
  - Stats (1y): `tr=109`, `win=54.1%`, `pnl=+12782.1`, `roi=+12.78%`, `dd%=4.56%`, `roi/dd=2.81`
  - Regime/loss-cluster report: `backtests/out/tqqq_exec5m_v25_kingmaker01_regime_adaptability.md`

- **V11b stability champ (pre-v25 CURRENT):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.50`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=2`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v11b_st_hold_neighborhood_30m_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1082`, `win=50.9%`, `pnl=+64406.5`, `roi=+64.41%`, `dd%=21.72%`, `roi/dd=2.96`
  - Stats (2y): `tr=207`, `win=54.6%`, `pnl=+26159.5`, `roi=+26.16%`, `dd%=5.94%`, `roi/dd=4.40`
  - Stats (1y): `tr=109`, `win=54.1%`, `pnl=+12255.7`, `roi=+12.26%`, `dd%=4.38%`, `roi/dd=2.80`

- **V9c high-slope stability champ:**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.03`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=4`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v9c_high_slope_30m_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1064`, `win=50.8%`, `pnl=+64578.5`, `roi=+64.58%`, `dd%=22.28%`, `roi/dd=2.90`
  - Stats (2y): `tr=204`, `win=53.9%`, `pnl=+19613.5`, `roi=+19.61%`, `dd%=6.45%`, `roi/dd=3.04`
  - Stats (1y): `tr=107`, `win=54.2%`, `pnl=+12086.4`, `roi=+12.09%`, `dd%=4.36%`, `roi/dd=2.77`

- **V9a stability champ (permission+short micro):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.004`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.01` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=4`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v9a_gate_shortrisk_micro_30m_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1073`, `win=51.1%`, `pnl=+71705.7`, `roi=+71.71%`, `dd%=21.49%`, `roi/dd=3.34`
  - Stats (2y): `tr=207`, `win=54.1%`, `pnl=+18035.3`, `roi=+18.04%`, `dd%=6.38%`, `roi/dd=2.83`
  - Stats (1y): `tr=107`, `win=55.1%`, `pnl=+12027.3`, `roi=+12.03%`, `dd%=4.63%`, `roi/dd=2.60`

#### LEGACY (exec=5m, RTH; 10y/2y/1y; first permission micro champ) (generated 2026-01-22; reclassified 2026-01-23)
- **V8c stability champ (permission micro):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=7`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.003`, `ema_spread_min_pct_down=0.05` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.02` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=4`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v8c_perm_micro_30m_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1081`, `win=52.2%`, `pnl=+69982.3`, `roi=+69.98%`, `dd%=22.81%`, `roi/dd=3.07`
  - Stats (2y): `tr=208`, `win=54.8%`, `pnl=+16618.3`, `roi=+16.62%`, `dd%=6.39%`, `roi/dd=2.60`
  - Stats (1y): `tr=107`, `win=59.8%`, `pnl=+12114.8`, `roi=+12.11%`, `dd%=4.68%`, `roi/dd=2.59`

#### LEGACY (exec=5m, RTH; 10y/2y/1y; session-filter stability champ, earlier hold) (generated 2026-01-22; reclassified 2026-01-22)
- **V6 stability champ (hold=6):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=10`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.005`, `ema_spread_min_pct_down=0.03` (short)
  - Activity: `entry_start_hour_et=10`, `entry_end_hour_et=15` (ET)
  - Shorts: `spot_short_risk_mult=0.02` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=6`, `flip_exit_gate_mode=off`
  - Output: `backtests/out/tqqq_exec5m_v6_activity_filters_10y2y1y_mintr100_top80.json`
  - Stats (10y): `tr=1091`, `win=51.2%`, `pnl=+64440.0`, `roi=+64.44%`, `dd%=22.15%`, `roi/dd=2.91`
  - Stats (2y): `tr=211`, `win=55.0%`, `pnl=+16944.2`, `roi=+16.94%`, `dd%=5.39%`, `roi/dd=3.14`
  - Stats (1y): `tr=109`, `win=58.7%`, `pnl=+10904.9`, `roi=+10.90%`, `dd%=4.67%`, `roi/dd=2.33`

#### LEGACY (exec=5m, RTH; 10y/2y/1y; pre-session-filter stability champ) (generated 2026-01-22; reclassified 2026-01-22)
- **V5 stability champ (no session filter):**
  - Signal: EMA `4/9` trend (two-way spot)
  - Bias: Supertrend on `4 hours`, `ATR=10`, `mult=0.55`, `src=hl2`
  - Permission: `ema_spread_min_pct=0.003` (long), `ema_slope_min_pct=0.005`, `ema_spread_min_pct_down=0.03` (short)
  - Shorts: `spot_short_risk_mult=0.02` (asymmetric risk)
  - Exits:
    - `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.04`
    - `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_only_if_profit=true`
    - `flip_exit_min_hold_bars=6`, `flip_exit_gate_mode=off`
  - Stats (10y): `tr=1190`, `win=52.7%`, `pnl=+71238.2`, `roi=+71.24%`, `dd%=31.07%`, `roi/dd=2.29`
  - Stats (2y): `tr=232`, `win=59.5%`, `pnl=+22419.2`, `roi=+22.42%`, `dd%=5.56%`, `roi/dd=4.03`
  - Stats (1y): `tr=120`, `win=58.3%`, `pnl=+10367.6`, `roi=+10.37%`, `dd%=4.60%`, `roi/dd=2.26`

#### LEGACY (exec=5m, RTH; 2y/1y high-WR) (generated 2026-01-21; reclassified 2026-01-22)
These are the older exec=5m champs that were optimized for **win-rate** and 2y/1y only:
- Signal generation on `30 mins` bars (RTH)
- Execution + intrabar TP/SL + flip exits simulated on `5 mins` bars (RTH) via `spot_exec_bar_size=5 mins`
Note: these do **not** validate as profitable over the 10y window under the same execution model (see 10y validation below).

Hard constraints (both windows):
- Positive PnL in both windows
- Win rate in `[56%, 70%]` in both windows
- At least ~1 trade / 2 days in the 1y window (operationalized as `>=120 trades`)

Windows:
- 2y: `2024-01-01 → 2026-01-19`
- 1y: `2025-01-01 → 2026-01-19`

Common preset knobs (shared by the champs below):
- Instrument: `spot`, symbol `TQQQ`, `use_rth=true`
- Direction: `directional_spot.up=BUY x1`, `directional_spot.down=SELL x1` (long/short)
- Signal: EMA `3/7` cross, `entry_confirm_bars=0`
- Execution realism: `spot_entry_fill_mode=next_open`, `spot_flip_exit_fill_mode=next_open`, `spot_intrabar_exits=true`
- Marking/DD realism: `spot_mark_to_market=liquidation`, `spot_drawdown_mode=intrabar`
- Costs: `spot_spread=0.01`, `spot_commission_per_share=0.005` (min `$1.00`), `spot_slippage_per_share=0.0`
- Sizing: `spot_sizing_mode=risk_pct`, `spot_risk_pct=0.01`, `spot_max_notional_pct=0.50`
- Scoring run settings: `starting_cash=100_000` (kingmaker), `max_open_trades=1`, `max_entries_per_day=0`, `spot_close_eod=false`

- **#1 Best worst-window PnL (reliability):**
  - Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.45`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=pct`, `PT=0.016`, `SL=0.032`, `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=4`
  - Stats (2y): `tr=249`, `win=57.43%`, `pnl=+9095.2`, `roi=+9.10%`, `dd=8464.7`, `dd%=8.46%`, `pnl/dd=1.07`
  - Stats (1y): `tr=121`, `win=61.16%`, `pnl=+8846.5`, `roi=+8.85%`, `dd=5841.2`, `dd%=5.84%`, `pnl/dd=1.51`

- **#2 Best worst-window PnL (runner-up):**
  - Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.45`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=pct`, `PT=0.016`, `SL=0.030`, `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=4`
  - Stats (2y): `tr=249`, `win=57.03%`, `pnl=+8335.9`, `roi=+8.34%`, `dd=9228.1`, `dd%=9.23%`, `pnl/dd=0.90`
  - Stats (1y): `tr=121`, `win=60.33%`, `pnl=+8189.8`, `roi=+8.19%`, `dd=6199.8`, `dd%=6.20%`, `pnl/dd=1.32`

- **#3 Best sum-PnL (max total):**
  - Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.45`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=10`, `mult=0.50`, `src=hl2`
  - Exits: `spot_exit_mode=pct`, `PT=0.010`, `SL=0.032`, `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=2`
  - Stats (2y): `tr=256`, `win=67.19%`, `pnl=+12031.9`, `roi=+12.03%`, `dd=6260.2`, `dd%=6.26%`, `pnl/dd=1.92`
  - Stats (1y): `tr=126`, `win=67.46%`, `pnl=+7897.5`, `roi=+7.90%`, `dd=6031.3`, `dd%=6.03%`, `pnl/dd=1.31`

- **#4 Best sum-PnL (runner-up; slightly different regime2 confirm):**
  - Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.45`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): Supertrend on `4 hours`, `ATR=7`, `mult=0.50`, `src=hl2`
  - Exits: `spot_exit_mode=pct`, `PT=0.010`, `SL=0.032`, `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=2`
  - Stats (2y): `tr=253`, `win=67.19%`, `pnl=+12459.1`, `roi=+12.46%`, `dd=6323.2`, `dd%=6.32%`, `pnl/dd=1.97`
  - Stats (1y): `tr=124`, `win=66.94%`, `pnl=+7251.8`, `roi=+7.25%`, `dd=6031.3`, `dd%=6.03%`, `pnl/dd=1.20`

- **#5 Best sum-PnL without regime2 (simpler baseline):**
  - Regime (bias): Supertrend on `4 hours`, `ATR=10`, `mult=0.45`, `src=hl2`
  - Permission (quality): `ema_spread_min_pct=0.003`
  - Regime2 (confirm): `off`
  - Exits: `spot_exit_mode=pct`, `PT=0.010`, `SL=0.032`, `exit_on_signal_flip=true`, `flip_exit_mode=entry`, `flip_exit_min_hold_bars=2`
  - Stats (2y): `tr=264`, `win=67.05%`, `pnl=+11903.6`, `roi=+11.90%`, `dd=7164.1`, `dd%=7.16%`, `pnl/dd=1.66`
  - Stats (1y): `tr=130`, `win=66.92%`, `pnl=+7030.8`, `roi=+7.03%`, `dd=6854.1`, `dd%=6.85%`, `pnl/dd=1.03`

Full ranked list:
- Micro PT/SL neighborhood:
  - `backtests/out/tqqq_exec5m_micro_ptsl_neighborhood_champs_minpnl_top25.json` (reliability-first)
  - `backtests/out/tqqq_exec5m_micro_ptsl_neighborhood_champs_sumpnl_top25.json` (P&L-first)
  - `backtests/out/tqqq_exec5m_micro_ptsl_neighborhood_kingmaker_all.json` (raw eval output; top=1176)
- Regime2 confirm (around the best P&L micro-PT/SL config):
  - `backtests/out/tqqq_exec5m_regime2_confirm_sumpnl_champs.json`
  - `backtests/out/tqqq_exec5m_regime2_confirm_sumpnl_kingmaker_all.json` (raw eval output; top=41)

Additional exploration runs (recorded commands; 2026-01-21):
- combo_fast discovery candidates (2y window, relaxed `win>=52%` for broader coverage):
  - `backtests/out/tqqq_exec5m_combo_30m_2y_candidates_wr52.json` (30m signals + 5m exec; 447 eligible on 2y)
  - `backtests/out/tqqq_exec5m_combo_1h_2y_candidates_wr52.json` (1h signals + 5m exec; 168 eligible on 2y)
- combo_fast stability winners on the 2y+1y windows (these tend to prefer `spot_close_eod=true`):
  - `backtests/out/tqqq_exec5m_combo_30m_wr56_2y1y_kingmaker_all.json` (267 passed `pnl>0`, `win>=56%`, `tr>=120` in both windows)
  - `backtests/out/tqqq_exec5m_combo_1h_wr56_2y1y_kingmaker_all.json` (83 passed `pnl>0`, `win>=56%`, `tr>=120` in both windows)

Repro commands:
```bash
# 2y discovery (generate candidate pools)
python -m tradebot.backtest spot --offline --cache-dir db --symbol TQQQ --use-rth \
  --start 2024-01-01 --end 2026-01-19 --bar-size "30 mins" --spot-exec-bar-size "5 mins" \
  --realism2 --base default --axis combo_fast --max-open-trades 1 --min-trades 240 \
  --write-milestones --milestones-out backtests/out/tqqq_exec5m_combo_30m_2y_candidates_wr52.json \
  --milestone-min-win 0.52 --milestone-min-trades 240 --milestone-min-pnl-dd 0.0

python -m tradebot.backtest spot --offline --cache-dir db --symbol TQQQ --use-rth \
  --start 2024-01-01 --end 2026-01-19 --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --base default --axis combo_fast --max-open-trades 1 --min-trades 200 \
  --write-milestones --milestones-out backtests/out/tqqq_exec5m_combo_1h_2y_candidates_wr52.json \
  --milestone-min-win 0.52 --milestone-min-trades 200 --milestone-min-pnl-dd 0.0

# 2y/1y stability scoring (kingmaker = multi-window eval)
python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_combo_30m_2y_candidates_wr52.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 500 --max-open 1 \
  --require-positive-pnl --min-trades 120 --min-win 0.56 \
  --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 400 --out backtests/out/tqqq_exec5m_combo_30m_wr56_2y1y_kingmaker_all.json

python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_exec5m_combo_1h_2y_candidates_wr52.json \
  --symbol TQQQ --bar-size "1 hour" --use-rth --offline --cache-dir db --top 250 --max-open 1 \
  --require-positive-pnl --min-trades 120 --min-win 0.56 \
  --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
  --write-top 400 --out backtests/out/tqqq_exec5m_combo_1h_wr56_2y1y_kingmaker_all.json
```

#### 10y validation of LEGACY high-WR champs (exec=5m, RTH) (generated 2026-01-22)
We can now do a **true apples-to-apples** 10y replay using the same multi-resolution execution model:
- Signal generation on `30 mins` (or `1 hour`) bars (RTH)
- Execution + intrabar TP/SL + flip exits on `5 mins` bars (RTH)

Window:
- 10y: `2016-01-01 → 2026-01-19`

Result (with the same “go-live shaped” Realism v2 settings used above):
- All LEGACY high-WR exec=5m champs above are **deeply negative** over the 10y window (ROI roughly `-65%..-72%`, max DD% roughly `73%..79%`).
  - Raw eval output: `backtests/out/tqqq_exec5m_top5_10y2y1y_eval.json`
- Broad 2y-discovery sweeps (30m+5m exec and 1h+5m exec) found **0** 10y-positive candidates among the families that satisfy `win>=56%` and the activity constraints on the 2y window.
  - 30m family: `backtests/out/tqqq_exec5m_combo_30m_any_positive10y.json` (empty)
  - 1h family: `backtests/out/tqqq_exec5m_combo_1h_any_positive10y.json` (empty)
- Re-running the combo_fast discovery with a **relaxed win-rate floor** (`win>=52%` on the 2y window) still finds **0** strategies that are simultaneously:
  - positive PnL on **10y + 2y + 1y** and
  - active enough (`>=120 trades`) and
  - `max_open_trades<=1`
  - 30m candidates: `backtests/out/tqqq_exec5m_combo_30m_2y_candidates_wr52.json` (447 eligible 2y candidates) → `backtests/out/tqqq_exec5m_combo_30m_wr52_10y2y1y_kingmaker_top50.json` (empty)
  - 1h candidates: `backtests/out/tqqq_exec5m_combo_1h_2y_candidates_wr52.json` (168 eligible 2y candidates) → `backtests/out/tqqq_exec5m_combo_1h_wr52_10y2y1y_kingmaker_top50.json` (empty)
  - Even **10y-only positivity** is absent inside the 2y-discovered 30m candidate pool: `backtests/out/tqqq_exec5m_combo_30m_wr52_10y_only_kingmaker_top20.json` (empty)

Important caveat (why this doesn’t prove “no 10y-positive strategy exists”):
- `--axis combo_fast` is a bounded funnel: it shortlists regimes by **2y performance** and then expands exits/gates around that shortlist.
  This means a 10y-positive corner could still exist outside the 2y-shortlist neighborhood.

Counterexample sanity check (exec=5m can still be positive on 10y):
- The **REALISM v2 Kingmaker** 10y presets from `spot_milestones.json` remain **positive** when re-evaluated apples-to-apples with `spot_exec_bar_size=5 mins`,
  but they are:
  - **long-only** (`directional_spot` has only `"up"`) and
  - **low activity** (e.g. `tr=29` in the last 1y window).
  - Output: `backtests/out/tqqq_10y_v2_kingmaker_exec5m_eval.json`

Update (2026-01-22): 10y+2y+1y-positive **two-way** exists (>=100 trades/1y)
- See the CURRENT “stop-only + flip exit” champs above.
  - Output: `backtests/out/tqqq_exec5m_v8c_perm_micro_30m_10y2y1y_mintr100_top80.json`

Legacy note (2026-01-22): 10y-positive **two-way** exists (but still low-activity)
- By extending the search space with **much stricter short gating** (`ema_spread_min_pct_down=0.05`) and adding an
  explicit short sizing knob (`spot_short_risk_mult`, default `1.0`), we can find configs that are **positive PnL**
  on **10y + 2y + 1y** while still keeping `"down": SELL` enabled.
  - Caveat: these candidates are still **low activity** (≈ `33–37 trades` in the last 1y window), so they do **not**
    satisfy the `>=120 trades` operational floor.
  - Candidate set: `backtests/out/tqqq_seeded10y_exec5m_shortrisk_variants.json`
  - 10y-only winners: `backtests/out/tqqq_seeded10y_exec5m_shortrisk_10y_only_top60.json`
  - 10y+2y+1y winners: `backtests/out/tqqq_seeded10y_exec5m_shortrisk_10y2y1y_top60.json`
  - Repro (offline):
    ```bash
    python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_seeded10y_exec5m_shortrisk_variants.json \
      --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 300 --max-open 1 --require-positive-pnl \
      --window 2016-01-01:2026-01-19 --write-top 60 --out backtests/out/tqqq_seeded10y_exec5m_shortrisk_10y_only_top60.json

    python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/tqqq_seeded10y_exec5m_shortrisk_10y_only_top60.json \
      --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db --top 60 --max-open 1 --require-positive-pnl \
      --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19 \
      --write-top 60 --out backtests/out/tqqq_seeded10y_exec5m_shortrisk_10y2y1y_top60.json
    ```

Interpretation / next search directions:
- Under Realism v2 (costs + risk sizing) the `win>=56%` high-activity family we’re targeting appears incompatible with **positive 10y PnL** on `TQQQ` (at least within the EMA×Supertrend×(pct/ATR exits) universe we swept).
- Relaxing win-rate and switching to the **trend + stop-only + flip-exit** family *does* produce stable multiwindow winners (see CURRENT section above).
- Next steps likely require relaxing **a different constraint** than win-rate:
  - relaxing the **activity floor** (trades/year), and/or
  - allowing **long-only** (or asymmetric sizing where shorts are smaller), and/or
  - switching to a different strategy family (e.g. ORB / session-based intraday).

#### LEGACY (pre-exec=5m) spot snapshots (reclassified 2026-01-21)
These are preserved for archaeology, but were generated before `spot_exec_bar_size=5 mins` existed, so they are not
apples-to-apples with the CURRENT exec=5m champs above.

Quick “current top 3” snapshots (generated 2026-01-16, post-intraday-timestamp-fix; pre-exec=5m):

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

Quick “max net PnL” snapshots (generated 2026-01-16, post-intraday-timestamp-fix; pre-exec=5m):

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

### LEGACY: Spot cross-asset sanity (TQQQ, 10y, RTH; pre-exec=5m) (reclassified 2026-01-21)
These were found by running our spot combo_fast sweep on `TQQQ` over `2016-01-01 → 2026-01-08` with `use_rth=true`.

#### LEGACY (Realism v2 multi-window stability; ROI-based, long-only; pre-exec=5m)
These were the “go-live shaped” TQQQ presets found under **Realism v2** before we added `spot_exec_bar_size=5 mins`:
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
- Candidate pool generation: `python -m tradebot.backtest spot --axis combo_fast --realism2 --long-only ... --write-milestones --milestones-out backtests/out/presets/spot_milestones.tqqq_10y_<bar>_realism_v2.json`
- Stability scoring: `python -m tradebot.backtest spot_multitimeframe --milestones backtests/out/presets/spot_milestones.tqqq_10y_<bar>_realism_v2.json --max-open <N> --require-positive-pnl ...`

#### LEGACY (pre-realism; optimistic)
Note:
- `max_open=0` means **unlimited stacking** (subject to `starting_cash` margin constraints); this can materially change results.
- Spot PnL for equities is per-share (multiplier `1.0`), not per-contract `100`.

- 1 hour (combo_fast sweep outcome, verified by direct backtest)
  - Best pnl/dd family found (note: uses `max_open=0` = unlimited stacking):
    - `tr=12407`, `win=64.25%`, `pnl=249.3`, `dd=34.5`, `pnl/dd=7.22`
    - `ema=3/7 trend`, `ST(3,0.8,close)@4h`, exits `PT=0.005 SL=0.030`, `hold=4`, filters `spread>=0.005 slope>=0.01`
  - High net-PnL family (also `max_open=0`):
    - `tr=14439`, `win=59.05%`, `pnl=318.1`, `dd=151.4`, `pnl/dd=2.10`
    - `ema=9/21 trend`, `ST(3,0.8,close)@4h`, exits `PT=0.015 SL=0.030`, `hold=0`, filter `spread>=0.005`

- 30 mins (combo_fast sweep output captured)
  - Full printout saved at `backtests/out/tqqq_10y_combo_fast_30m.txt`.
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

They are preserved for archaeology in `backtests/out/presets/spot_milestones.legacy_pre_2026-01-14_intraday_ts_fix.json`, but should **not** be trusted for decision-making because they were affected by multi-timeframe lookahead leakage (notably `4 hours` regime gating on intraday signal bars).

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
  - Position sizing + ROI reporting (implemented in Realism v2): `spot_sizing_mode`, `spot_risk_pct`, `spot_short_risk_mult`, `spot_notional_pct`, `spot_max_notional_pct`, `spot_min_qty`, `spot_max_qty`
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




##
##
##
## MISC
##
##  Quest & Upcoming Feature:  News aggregation overlay, sentiment scoring, and event-driven capabilities
##             ( multiround GPT parsing on breaking news / events for the final overlay / scoring step )
##

# Research on 2024 vs 2025 TQQQ volatility spikes, early detection analysis of weekly TR and gap patterns, before sharpening reg gates

• 1. Feb 28 onward: is there a “distinct slope” you can catch (rare, and also hits COVID etc)?

    Yes — but the cleanest “early warning” in the data is not ATR14% level (that lags), it’s TR% velocity / weekly TR regime, and it shows up late Feb → early Mar.

  
  Key facts from the daily RTH bars (db/TQQQ/TQQQ_2016-01-01_2026-01-19_1day_rth.csv), using TR% = TrueRange / prev_close:

  - The week containing Feb 28, 2025 (ISO week starting 2025-02-24) has:
      - weekly TR% median 6.10%
      - week-over-week jump (delta) +2.94
      - that delta is 93.9th percentile vs all week-over-week TR% median changes in 2016–2026
      - drawdown proxy at Feb 28 close (dd60) already about -23%
  - The week starting 2025-03-03 is the real “oh, this is different” week:
      - weekly TR% median 9.50% (95th percentile of weekly TR% medians)
      - delta vs prior week +3.40 (95.6th percentile)
      - 4-week “ramp” (TR_med now vs 4 weeks prior) +5.51 (96.5th percentile)

  So if you’re looking for a slope/angle signal (like your screenshot), it’s real — it just starts showing up as volatility acceleration in the week of Feb 24, and becomes full-blown stress in the week of Mar 3.

  A very sharp “panic week” rule that barely over-triggers
  If you want something that:

  - is rare across 10y,
  - also catches COVID / 2022 / 2018Q4-type events,
  - and is clearly “risk-off downtrend stress”…

  This is the sharpest thing I found from your requested ingredients (TR% + gaps), in weekly form:

  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)

  Results over 2016–2026:

  - Flags 17 weeks total (very low).
  - 16/17 (94.1%) of flagged weeks have dd60 ≤ -30% by that week’s last trading day.
  - It flags the exact week you care about: 2025-03-03 (and the extreme week 2025-04-07).
  - It also flags the big historical stress clusters:
      - 2020-02-24, 2020-03-09, 2020-03-16…
      - 2022-03-07, 2022-04/05 clusters…
      - 2018-12 etc.

  This meets your “doesn’t trigger except huge downtrends” requirement better than ATR-based rules.

  2. What do the ATR/TR “waves” look like in the 1–2 years before (weekly view + why March ramps / April peaks)

  Your monthly read is correct: Feb isn’t the volatility peak — March ramps, April peaks. Weekly makes the shape obvious.

  Here’s the “wave” context (weekly medians):

  2024 summer spike (non-crisis-like compared to 2025)

  - 2024-07-29: TR_med 9.87%, neg_gaps 20%
  - 2024-08-05: TR_med 9.70%, neg_gaps 40%
  - Then it mean-reverts quickly.

  So 2024 can print TR_med ~9–10, but it wasn’t “persistent negative-gap stress”.

  2025 ramp + peak

  - 2025-02-24: TR_med 6.10%, neg_gaps 40%  (the “velocity” week)
  - 2025-03-03: TR_med 9.50%, neg_gaps 60%  (panic threshold week)
  - 2025-03-10: TR_med 7.52%, neg_gaps 60%
  - 2025-03-31: TR_med 8.70%, neg_gaps 100%
  - 2025-04-07: TR_med 23.56%, neg_gaps 80% (the peak/reversal chaos)

  That Apr 7 week is exactly your “ATR peaks during reversal” intuition — and you can also see why ATR14% is lagging: the TR and gaps explode first, then ATR catches up.

  For comparison, COVID looks similar in “weekly TR shock” shape, just bigger:

  - 2020-02-24: TR_med 13.08%, |gap|_med 7.54%
  - 2020-03-09: TR_med 20.60%, |gap|_med 16.79%
  - 2020-03-16: TR_med 24.54%, ATR14_med 24.93%

  That’s the same pattern class: gap/true-range explosion first, ATR ramps after.


  3. ATR-expansion vs TR-velocity: which actually catches Mar 5 early enough?

  This part matters for your “prevent the March 5 overnight gap loss” goal.

  - The ATR14%-expansion idea (fast5 median ATR% vs slow20 median ATR%) is too laggy for Mar 5 in your data.
      - With a reasonable “expansion” rule (fast5 ≥ ~6 and diff ≥ ~1.5), it turns on around Mar 6–7, not by Mar 5 close.
  - A TR-based rolling median triggers in time:
      - If you define risk_off_TR5 = median(last 5 daily TR%) ≥ 9.0%
      - It flags 2025-03-04, 03-05, 03-06, 03-07, 03-10 (and then the April cluster)
      - So it is ON at Mar 5 close, which is exactly what you need to avoid holding into Mar 6 open.

  Also, your key “bad overnight” is absolutely consistent with this:

  - On 2025-03-06: daily gap% ≈ -5.05% (big negative gap day)
  - Your strategy did have exposure over the Mar 5 → Mar 6 boundary.

  ## How rare is this TR5 detector (does it overtrigger)?
  - TR5_med ≥ 9 flags 159 days out of 2516 trading days (~6.3% of days).
  - It clusters mainly in the “obvious” stress regimes (2018Q4, COVID, 2022 bear, 2025).
  - It’s fairly selective for deep drawdown: ~84% of flagged days have dd60 ≤ -30%.

  - The champ is in-position overnight on 2101 close→next-open boundaries.
  - If you applied risk-off only on TR5_med≥9 days:
      - it would affect 114 of those overnights (5.4%).
  - If you applied the stricter weekly-panic regime:
      - it would affect 51 overnights (2.4%).

  So: it’s rare enough to plausibly be “surgical”.
  But: the hard trade-off
  Those risk-off nights are not free to avoid:
  - They have fatter left tails (bad gaps), but also fatter right tails (good gaps).
  - For TR5 risk-off overnights, gap-PnL distribution (per overnight boundary) looked like:
      - p05 about -1405 vs normal p05 about -587
      - min about -2950
      - but mean is actually higher than normal (because some huge rebounds exist)



  ### Even with a no‑lookahead “online weekly” implementation, it will NOT catch the Mar 5 overnight loss
  - Option A (most aligned with your intent): cancel next-open entries + forbid new late-day entries when risk_off is ON (prevents “entered near end of day then got gapped”)
  - Detector 2: weekly panic (TR_week_med ≥ 9 AND neg gaps ≥ 60%)

  So the clean integration is: add a new “risk_off overlay” under filters (parallel to shock, not replacing it), and in the spot engine:

  - refuse scheduling pending_entry_dir if it would fill next morning (overnight),
  - refuse fills/scheduling after a cutoff time when risk_off is ON.


  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)
  …was computed using the full completed week in our analysis.
  
  - If we naïvely apply “this week is panic” during the week, that’s future leakage (you don’t know Friday’s gap sign on Tuesday).
  This is important because your motivation was specifically the Mar 5 15:05 long → Mar 6 09:35 stop.

  That trade is in your focus CSV:

  - 2025-03-05T15:05:00 → 2025-03-06T09:35:00, pnl -1231.52

  Now, check the actual daily gaps that week (from the daily bars):

  - Mar 3 gap +
  - Mar 4 gap -
  - Mar 5 gap +
  - Mar 6 gap -
  - Mar 7 gap -

  So the neg-gap fraction only reaches 60% on Friday (3/5).
  Meaning: weekly panic turns ON too late to block the Mar 5 → Mar 6 gap unless you allow lookahead (which we can’t).

  So: weekly panic is extremely selective (good), but it doesn’t solve the exact Mar 5 event (bad). It will still help April and other crisis clusters though.

  ———

  ## Solution Candidates (with concise ratings):

  1. TR5 Risk‑Off Overlay (no lookahead) + Option A hygiene [Score: 9/10] ← IDEAL for your Mar‑5 objective
     • Define risk_off = median(last 5 *completed* daily TR%) >= 9.0 (computed from daily bars; no future leak).
     • Apply Option A: block late-day entries + cancel overnight next_open fills when risk_off ON.
     Pros:
      - Actually ON by Mar 5 intraday (using completed days up to Mar 4), so it can block the 2025-03-05T15:05 entry.
      - Still fairly selective (clusters mainly in 2018Q4 / 2020 / 2022 / 2025).
        Cons:
      - Less “rare” than weekly panic (more flagged days than 17 weeks).


• Yes — a news/GPT layer can very plausibly complement your system, and it complements it in exactly the place where your purely market‑derived detectors are structurally weak:

  - Market detectors (TR%, gaps, ATR) are great at saying “volatility regime changed”.
  - They’re much worse at saying “this is the start of a sustained risk-off narrative” early enough (especially if the first big move is a weekend gap), and they’re also bad at saying “the narrative has flipped; this is now a
    rebound regime” (because TR stays elevated into rebounds).

  That’s why your TR5 / weekly‑panic research is still correct and still valuable, even though the naïve “cancel overnights during riskoff” didn’t beat the champ: the detectors are real, but they’re missing the missing variable:
  narrative direction + de‑escalation detection.

  ### Why your TR% + negative-gap signals are still “the right ingredients”

  From your pasted research:

  - They were selective (weekly panic barely triggers).
  - TR5 triggers early enough to matter for the Mar 5 → Mar 6 gap-loss.
  - But they stay “ON” into April rebound conditions because the market is still chaotic.

  So the problem isn’t “TR5 is wrong”. The problem is: TR5 is an amplitude detector; it’s not a direction+phase detector.

  ### Where a news/GPT layer actually helps (and why it could fix the April rebound problem)

  A news/GPT layer can provide two things the market-only layer can’t reliably provide without lag:

  1. Pre-gap anticipation (weekend / overnight)
      - If the first real damage happens between sessions, the only thing you can “see” beforehand is:
          - the narrative worsening (tariff escalation, retaliation headlines, etc.), and/or
          - the market already being in a fragile state (TR ramp, more negative gaps, worsening dd proxy).
      - TR5 only “turns on” after you’ve already had several large TR days.
  2. Reversal / de-escalation detection
      - This is the killer: market volatility stays high during rebounds.
      - A good news classifier could flip from:
          - “escalation / risk-off / negative expected drift”
          - to “de‑escalation / relief / positive expected drift”
      - …even while TR remains high.
      - That’s precisely how you avoid the failure mode we saw: “risk-off stays elevated into the rebound, so you miss the big rebound long(s).”

  ### The right “shape” of an overlay once you have news

  The overlay that makes sense is not “TR5 cancels overnights.” It’s more like a risk-state machine that uses both market data and news.

  Think in terms of a state with at least these outputs:

  - risk_on (bool / intensity)
  - risk_dir ∈ {down, up, neutral}
  - risk_phase ∈ {escalation, panic, cooling, relief}

  Then your actions become directional and phase-aware, e.g.:

  - If risk_dir=down and risk_on=true:
      - block/cancel only late-day long entries (especially ones that will be held overnight),
      - keep/allow shorts (maybe scaled, but carefully),
      - tighten long stops, allow more aggressive loss-cutting,
      - optionally disallow holding long over weekends specifically.
  - If risk_dir=up and risk_on=true (high-vol rebound / relief rally):
      - do not block the rebound longs (this is exactly where the earlier TR-based overlay accidentally hurt you),
      - if you do anything, you scale down size rather than block participation.

  This is exactly the missing lever: the TR/gap regime says “this is a tail-risk environment”; the news regime says “tail-risk is currently skewed down vs skewed up.”

  ### Where it should plug into your pipeline (conceptually)

  With news available, I would still not jam it into EMA/supertrend math directly. I’d treat it like shock: a parallel overlay that can influence:

  - Entry permission (block/cancel specific kinds of entries; especially near EOD / weekend)
  - Sizing (risk budget scaling)
  - Exit semantics (e.g., allow flip-at-loss only when risk_dir opposes your position)

  That keeps your core alpha logic stable, and isolates the “narrative risk” system.

##
##
##
##
##
##
##
##
