# SLV Research (High-Activity Spot Scalper)

This folder is the **single source of truth** for SLV research going forward:
- the **current champion stack** (promoted configs + expected multiwindow metrics), and
- the **evolution log** (commands, parameter ranges swept, what changed vs prior).

We are explicitly targeting the new contract:
- **Symbol:** `SLV`
- **Instrument:** spot (underlying)
- **Signal timeframe:** **OPEN** (CURRENT dethroner is `10 mins` FULL24 with entry-hour gating; high-trade island is `4 hours` regime with the same `10 mins` signal lane)
- **Execution timeframe:** **5 mins** (`spot_exec_bar_size="5 mins"`) for realism (intrabar stops + next-open fills)
- **Goal:** dual-track optimization across **10y / 2y / 1y**
  - **Track A (pnl-first dethrone):** beat v25 on total PnL in all windows (and keep `pnl/dd` healthy)
  - **Track B (high-trade):** increase trade count while preserving all-window positivity and avoiding the high-churn negative basin
  - **Parity rule (promotion gate):** promoted CURRENT must run `open_position_cap=1` to match live single-position behavior

## Current Champions (stack)

### Canonical command equivalents (current architecture)
Historical command logs below are normalized to these current wrappers:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`
- Legacy alias still accepted: `python -m tradebot.backtest multitimeframe ...`

Current LF champion replay (v31.2):
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --top 1 --min-trades 0 \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08
```

### CURRENT (v31.2) — FULL24 10m single-position parity dethrone (single-process verified)
This is the current SLV dethroner under a **live-parity-first contract**:
- Enforces single-position behavior: `open_position_cap=1` (matches live net-position handling)
- Removes simultaneous-position holding from the promoted lane (no live parity)
- Keeps the v31.1 shock wall core (`shock_on/off=1.35/1.25`) and retunes sizing for single-position mode
- Verified via single-process (`--jobs 1`) multiwindow sweeps with fixed windows

**v31.2 kingmaker #01** (from `backtests/slv/slv_v31_2_singlepos_parity_eval_20260212_top1.json`)
- `open_position_cap=1`, `spot_max_notional_pct=0.7`, `spot_risk_pct=0.016`, `spot_stop_loss_pct=0.016`
- Worst-window `roi/dd`: **4.546495578** (worst window is **1y**)
- 10y: `roi/dd=6.541506284`, ROI ≈ **384.54%**, DD% ≈ **58.78%**, trades ≈ **798**
- 2y:  `roi/dd=6.093215523`, ROI ≈ **183.52%**, DD% ≈ **30.12%**, trades ≈ **162**
- 1y:  `roi/dd=4.546495578`, ROI ≈ **93.99%**, DD% ≈ **20.67%**, trades ≈ **69**

Single-run delta vs prior v31.1 stacked champ (`open_position_cap=3`, same windows):
- 10y: `pnl -158,849.52` (`543,388.44 -> 384,538.92`), `roi/dd +0.151931987`
- 2y:  `pnl -80,142.10` (`263,658.96 -> 183,516.87`), `roi/dd +0.607742995`
- 1y:  `pnl -31,623.56` (`125,609.17 -> 93,985.61`), `roi/dd +0.342315486`

Defining delta vs v31.1:
- Structural parity change: `open_position_cap: 3 -> 1` (single-position only)
- Single-position sizing retune: `spot_max_notional_pct=0.7`, `spot_risk_pct=0.016`, `spot_stop_loss_pct=0.016`
- Same lane core: `signal_bar_size="10 mins"`, `spot_exec_bar_size="5 mins"`, `signal_use_rth=false`, `ema_preset=5/13`, `supertrend @ 1 day`, `shock_on/off=1.35/1.25`

### Previous (v31.1) — FULL24 10m shock-hysteresis wall dethrone (stacked backtest)
This was the prior SLV dethroner under the **multiwindow pnl-first contract**:
- Positive PnL across **10y / 2y / 1y**
- Beats the prior v31 (`shock_on/off=1.50/1.40`) on `pnl` and worst-window `roi/dd`
- Verified via single-process (`--jobs 1`) multiwindow sweeps with cache reset per pass

**v31.1 kingmaker #01** (from `[PURGED false-timebars artifact]`)
- `shock_on_ratio=1.35`, `shock_off_ratio=1.25` (canonical pick from the top plateau)
- Worst-window `roi/dd`: **4.204180092** (worst window is **1y**)
- 10y: `roi/dd=6.389574297`, ROI ≈ **543.39%**, DD% ≈ **85.04%**, trades ≈ **1,615**
- 2y:  `roi/dd=5.485472528`, ROI ≈ **263.66%**, DD% ≈ **48.06%**, trades ≈ **316**
- 1y:  `roi/dd=4.204180092`, ROI ≈ **125.61%**, DD% ≈ **29.88%**, trades ≈ **140**

### Previous (v31) — Full24 time-windowed 10m dethroner (all-window `pnl` + `roi/dd` beat vs v25)
This was the prior SLV dethroner under the same contract:
- Positive PnL across **10y / 2y / 1y**
- Beat v25 on **both** `pnl` and `roi/dd` in all windows
- Uses `signal_use_rth=false` (FULL24 bars), with entry-hour gating instead of RTH lane

**v31 kingmaker #01** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **4.174889507** (worst window is **1y**)
- 10y: `roi/dd=6.362166024`, ROI ≈ **528.67%**, DD% ≈ **83.10%**, trades ≈ **1,616**
- 2y:  `roi/dd=5.451136011`, ROI ≈ **257.76%**, DD% ≈ **47.29%**, trades ≈ **316**
- 1y:  `roi/dd=4.174889507`, ROI ≈ **123.15%**, DD% ≈ **29.50%**, trades ≈ **152**

### CURRENT-HIGH-TRADE (v31-HT) — Full24 all-window beater with higher turnover
This is the trade-focused counterpart on the same winning family.

**v31-HT kingmaker #51** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **3.913645159** (worst window is **1y**)
- 10y: `roi/dd=6.104210764`, ROI ≈ **502.82%**, DD% ≈ **82.37%**, trades ≈ **2,591**
- 2y:  `roi/dd=5.208784411`, ROI ≈ **246.93%**, DD% ≈ **47.41%**, trades ≈ **535**
- 1y:  `roi/dd=3.913645159`, ROI ≈ **114.97%**, DD% ≈ **29.38%**, trades ≈ **264**

Notes:
- This variant trades materially more than v31 kingmaker #01 and still beats v25 on all-window `pnl` + `roi/dd`.
- It remains below the old v25 trade count, which is why the high-trade island remains an active research track.

### Previous (v25) — Drawdown-driven shock throttle (FULL24, 1h; worst-window `roi/dd = 3.601660671`)
This was the prior SLV FULL24 stability champ under the original **multiwindow stability contract**:
- Positive PnL across **10y / 2y / 1y**
- Activity: `>=500 trades` in the 1y window
- Ranked by stability = maximize the worst-window `roi/dd`

**v25 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **3.601660671** (worst window is **1y**)
- 10y: `roi/dd=3.732269713`, ROI ≈ **240.65%**, DD% ≈ **64.48%**, trades ≈ **4,176**
- 2y:  `roi/dd=4.417900341`, ROI ≈ **164.63%**, DD% ≈ **37.26%**, trades ≈ **855**
- 1y:  `roi/dd=3.601660671`, ROI ≈ **102.90%**, DD% ≈ **28.57%**, trades ≈ **501**

Defining delta vs v7 baseline (needle-thread, throttle-only):
- Keep base signal/regime/exits identical to v7 (same `ema 9/21 trend`, same `spot_stop_loss_pct=0.012`, `flip-profit-only`, `max_notional=0.5`)
- Add **drawdown-driven throttle metric**: `shock_scale_detector="daily_drawdown"` + `shock_drawdown_lookback_days=10`
- Clamp harder under stress: `shock_risk_scale_target_atr_pct=8.0` (interpreted as DD% magnitude target), `shock_risk_scale_min_mult=0.05`
- Apply throttle to both `risk` and `cap`: `shock_risk_scale_apply_to="both"`

### TIE-CHAMP (v26) — Adds TQQQ-style riskpanic micro overlay (same floor; better 10y)
This is not a stability dethrone (worst-window floor ties v25), but it **does** improve the 10y ratio.

**v26 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **3.601660671** (same 1y floor as v25)
- 10y: `roi/dd=3.931068737`, ROI ≈ **249.21%**, DD% ≈ **63.40%**, trades ≈ **4,170**
- 2y:  `roi/dd=4.417900341`, ROI ≈ **164.63%**, DD% ≈ **37.26%**, trades ≈ **855**
- 1y:  `roi/dd=3.601660671`, ROI ≈ **102.90%**, DD% ≈ **28.57%**, trades ≈ **501**

Interpretation:
- The v26 `riskpanic_*` settings **do not trigger in the last 2y/1y windows** (so they can’t lift the floor),
  but they do trigger on a small set of older “monster days” → decade ratio improves.

### Earlier (v10) — Risk-off overlay dethrone (rounding hides it)
This was the first post-v7 dethrone; it rounds to “3.60”, but it is strictly higher than v7.

**v10 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **3.601454743** (1y)
- 10y: `roi/dd=3.712911120`, ROI ≈ **239.10%**, DD% ≈ **64.40%**, trades ≈ **4,168**
- 2y:  `roi/dd=4.422903614`, ROI ≈ **164.98%**, DD% ≈ **37.30%**, trades ≈ **850**
- 1y:  `roi/dd=3.601454743`, ROI ≈ **102.82%**, DD% ≈ **28.55%**, trades ≈ **529**

Key levers:
- `riskoff_tr5_med_pct=4.5@10d`, `riskoff_mode="hygiene"`
- late-day cutoff on risk days: `risk_entry_cutoff_hour_et=16`

### Baseline (v7) — First FULL24 stability lift (worst-window `roi/dd = 3.597984452`)
**v7 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **3.597984452** (1y)
- 10y: `roi/dd=3.745381803`, ROI ≈ **247.83%**, DD% ≈ **66.17%**, trades ≈ **4,207**
- 2y:  `roi/dd=4.419906533`, ROI ≈ **164.82%**, DD% ≈ **37.29%**, trades ≈ **857**
- 1y:  `roi/dd=3.597984452`, ROI ≈ **102.68%**, DD% ≈ **28.54%**, trades ≈ **539**

Defining shape (abridged):
- Signal: `ema_preset=9/21`, `ema_entry_mode=trend`, `entry_confirm_bars=0` on `1 hour`
- Exits: stop-only `spot_stop_loss_pct=0.012`, plus `exit_on_signal_flip=true` with `flip_exit_only_if_profit=true` and `flip_exit_min_hold_bars=2`
- Regime: `supertrend @ 1 day`
- Shock: `shock_gate_mode=surf` with `shock_detector=daily_atr_pct`
- Session/TOD: **off** (FULL24)
- `open_position_cap=5`, `spot_close_eod=false`, `spot_short_risk_mult=0.02`

### LEGACY (timestamp-bug) — archived (pre-ET-fix SLV “champs”)

These sections are preserved for archaeology only.

We had a **timestamp interpretation bug** (ET) in the backtest engine that made the older SLV champ
files (v6/v4/v3/v2) **not reproducible at their previously reported metrics**.

ET-fixed rescored reference (reproducible today, but *not* champion-grade):
`[PURGED false-timebars artifact]`
- v6 kingmaker #01 (rescored): worst-window `roi/dd≈0.82` (10y), 2y `≈2.05`, 1y `≈1.65`

#### (legacy) v6 — First above-floor stability lift (timestamp-bug era; do not use)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- meets the activity constraint in the 1y window (`>500 trades`)
- and meaningfully improves worst-window stability vs v4 (10y `roi/dd` lifted from **`1.39`** → **`1.70`**)

**v6 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **1.70** (worst window is **10y**)
- 10y: `roi/dd=1.70`, ROI ≈ **115.5%**, DD% ≈ **68.0%**, pnl ≈ **$115,548**, trades ≈ **5,680**
- 2y:  `roi/dd=2.85`, ROI ≈ **92.4%**, DD% ≈ **32.5%**, trades ≈ **1,505**
- 1y:  `roi/dd=2.18`, ROI ≈ **58.3%**, DD% ≈ **26.7%**, trades ≈ **1,119**

Defining shape (abridged):
- Signal: `ema_preset=21/50`, `ema_entry_mode=trend`, `entry_confirm_bars=0`
- Exits: stop-only `spot_stop_loss_pct=0.010`, plus `exit_on_signal_flip=true`
  with `flip_exit_only_if_profit=true` and a longer debounce `flip_exit_min_hold_bars=4`
- Regime: `supertrend @ 4 hours` (`ST(5,0.75,hl2)`)
- Session: `entry_start/end=9–16 ET`
- `open_position_cap=5`, `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- Note: this winner is still “simple” (no extra shock/risk overlays); the gain came from a better Supertrend + hold pocket.

#### (legacy) v4 — First decade stability lift from the v3 baseline (timestamp-bug era; do not use)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- meets the activity constraint in the 1y window (`>500 trades`)
- and improves worst-window stability vs v3 (10y `roi/dd` lifted from ~`1.25` → **`1.39`**)

**v4 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **1.39** (worst window is **10y**)
- 10y: `roi/dd=1.39`, ROI ≈ **98.9%**, DD% ≈ **70.9%**, pnl ≈ **$98,882**, trades ≈ **5,902**
- 2y:  `roi/dd=2.66`, ROI ≈ **81.6%**, DD% ≈ **30.6%**, trades ≈ **1,556**
- 1y:  `roi/dd=2.15`, ROI ≈ **56.9%**, DD% ≈ **26.5%**, trades ≈ **1,086**

Defining shape (abridged):
- Signal: `ema_preset=21/50`, `ema_entry_mode=trend`, `entry_confirm_bars=0`
- Exits: stop-only `spot_stop_loss_pct=0.010`, plus `exit_on_signal_flip=true`
  with `flip_exit_only_if_profit=true` and `flip_exit_min_hold_bars=0`
- Regime: `supertrend @ 4 hours` (`ST(7,0.65,hl2)`)
- Session: `entry_start/end=9–16 ET`
- `open_position_cap=5`, `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- Note: even though v4 sweeps expanded shock/risk scaling pockets, the current best is still **shock=off** (pure regime+stack wins again).

#### (legacy) v3 — First stability lift above the v2 baseline (timestamp-bug era; do not use)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- meets the activity constraint in the 1y window (`>500 trades`)
- and meaningfully improves worst-window stability vs v2.

**v3 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **1.25** (worst window is **10y**)
- 10y: `roi/dd=1.25`, ROI ≈ **90.9%**, DD% ≈ **72.4%**, pnl ≈ **$90,864**, trades ≈ **5,983**
- 2y:  `roi/dd=2.43`, ROI ≈ **71.1%**, DD% ≈ **29.2%**, trades ≈ **1,619**
- 1y:  `roi/dd=2.10`, ROI ≈ **55.8%**, DD% ≈ **26.6%**, trades ≈ **1,057**

Defining shape (abridged):
- Signal: `ema_preset=21/50`, `ema_entry_mode=trend`, `entry_confirm_bars=0`
- Exits: stop-only `spot_stop_loss_pct=0.010`, plus `exit_on_signal_flip=true`
  with `flip_exit_only_if_profit=true` and `flip_exit_min_hold_bars=0`
- Regime: `supertrend @ 4 hours` (`ST(14,0.6,hl2)`)
- Session: `entry_start/end=9–16 ET`
- `open_position_cap=5` (stacking; currently required for decade positivity at this cadence)
- `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- No extra permission/shock/risk-pop overlays in this top pick (surprisingly, the simple regime+stack wins here)

#### (legacy) v2 — First 10y/2y/1y-positive high-activity baseline (timestamp-bug era; do not use)
This is the first SLV family that is:
- Positive PnL in **all 3 windows** (10y + 2y + 1y)
- Meets the activity constraint (`>500 trades` in 1y)

It is **not yet** “contract-ready” because worst-window `roi/dd` is still well below the target floor
(initially `>=1.5..2.0`, eventually aiming near the TQQQ champ ≈3.49).

**v2 kingmaker #1** (from `[PURGED false-timebars artifact]`)
- Worst-window `roi/dd`: **~0.92**
- 10y: pnl ≈ **$73,611**, trades ≈ **6,199**
- 1y: trades ≈ **1,150**

Defining shape (abridged):
- `ema_preset=21/50`, `ema_entry_mode=trend`, `entry_confirm_bars=0`
- `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.010` (stop-only)
- `exit_on_signal_flip=true`, `flip_exit_only_if_profit=true`, `flip_exit_min_hold_bars=0`
- Regime: `supertrend @ 4 hours` (`ST(7,0.5,hl2)`)
- Session: `entry_start/end=9–16 ET`
- `open_position_cap=5` (stacking; currently required for decade positivity at this cadence)

Promotion checklist (minimum):
- Positive PnL in **all 3** windows (10y + 2y + 1y)
- 1y trades `> 500`
- Worst-window `roi/dd` meets the current floor (initially `>= 1.5..2.0`, then raised)

## Benchmark

Reference benchmark (from the TQQQ stability champ, v34):
- Worst-window `roi/dd ≈ 3.49`
- Trades (1y window) ≈ `107`

For SLV, initial exploration uses a lower floor (`roi/dd >= 1.5..2.0`) to find “feasible” shapes first, then we ratchet upward and aim to beat the best we’ve ever seen.

Terminology note:
- `roi/dd` is a **risk-adjusted ratio**, not a “profit flag”.
- `roi/dd > 0` implies the window ROI is positive (profitable), while `roi/dd < 1` means the **max drawdown exceeded the total return** in that window (still profitable, but a rough ride).

## Data / Cache prerequisites

This SLV quest requires cached bars for:
- Signal bars: `1 hour` (FULL24) for the CURRENT champ lane (`use_rth=false`)
- Execution bars: `5 mins` (FULL24) (`spot_exec_bar_size="5 mins"`)
- Regime bars: `1 day` (FULL24) for the CURRENT champ lane
- (Optional) high-activity lane: `30 mins` / `15 mins` signals (RTH or FULL24, depending on the contract we choose)

Notes:
- Cache now includes `5 mins`, `15 mins`, `1 hour`, `4 hours`, `1 day` for SLV over `2016-01-08 → 2026-01-08`.
- Any non-`--offline` run will fetch & cache automatically via IBKR (but we prefer `--offline` once cached).

Minimal “prefetch + sanity run” (will fetch `15 mins` + `5 mins` + `4 hours`):
```bash
python -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db \
  --start 2016-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis hold --min-trades 0 --top 5
```

## Workflow (v0 → v1 ...)

High-level pipeline:
1) **Generate candidate pool** (spot sweeps → milestones JSON)
2) **1y prefilter**: require `>500 trades` + `pnl>0` on 1y (legacy runs used `>=1000`)
3) **10y/2y/1y kingmaker**: maximize the **worst-window** `roi/dd`, keep `pnl>0` in all 3 windows
4) Promote winners into the **Current Champions** stack above

## Evolutions (stack)

### v0 — High-activity scalper scaffolding (15m signals, 5m exec; RTH; multiwindow kingmaker)
Status: **DONE** (compact baseline failed activity+profit under costs)

Key intent:
- Make the strategy “trade a few times/day” without turning into noise.
- Use regime gating + shock overlay as stability guardrails.

#### v0.0 — Candidate pool generation (broad)
Run a broad sweep on the **1y window** to produce an initial milestones pool that already satisfies the activity constraint:
```bash
python -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis combo_full --combo-full-preset gate_matrix --min-trades 0 --top 50 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.00 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v0_combo_15m_1y_milestones.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v0_combo_15m_1y_milestones.json` contains **0 eligible presets** at
  `milestone_min_trades=1000` + `milestone_min_pnl_dd>=0` (1y window, `--realism2`).

### v1 — hf_scalp axis (stop-only + flip-profit, with cadence knobs; exec=5m realism)
Status: **DONE** (found many 1y high-activity winners; decade not yet positive)

#### v1.0 — 1y candidate pool (>=1000 trades; pnl/dd>=0)
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis hf_scalp --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.00 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-30):
- `[PURGED false-timebars artifact]` contains **399 eligible presets**.

#### v1.1 — 10y/2y/1y kingmaker eval (stability-first)
Strict gate (must be positive pnl in all windows):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_top80.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_top80.json` is **empty** (0 candidates passed).
- Diagnostic finding: **10y pnl is negative for 399/399 candidates** in the v1 pool.

Diagnostics (no-positive gate removed; useful for inspection / debugging):
- Top-80 by “least-bad stability”: `[PURGED false-timebars artifact]`
- Full evaluated set (all 399): `[PURGED false-timebars artifact]`

### v2 — hf_scalp widened (wider stops + longer EMAs + daily shock + daily regime option)
Status: **DONE** (first decade-positive candidates found)

Key change vs v1:
- Widened `spot_stop_loss_pct` into the ~1–2% range and expanded EMA presets (reduces whipsaw).
- Added a daily shock option (`daily_atr_pct`) and a daily regime option (`1 day` Supertrend) into the overlay grid.

#### v2.0 — 1y candidate pool (>=1000 trades; pnl/dd>=0)
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis hf_scalp --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.00 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-30):
- `[PURGED false-timebars artifact]` contains **872 eligible presets**.

#### v2.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out [PURGED false-timebars artifact]
```

Outcome (2026-01-30):
- **120** candidates passed the strict positivity gate (10y+2y+1y) at `min_trades=1000`.
- Best worst-window `roi/dd` in this batch is **~0.92** (still below the intended floor; next step is stability tightening).

### v3 — champ_refine seeded from v2 winners (10y tune), then re-ranked by trades to preserve cadence
Status: **DONE** (raised worst-window stability vs v2; still below `roi/dd>=1.5..2.0` floor)

Why v3 exists:
- v2 proved feasibility (positive across windows at high activity) but stability floor was only ~`0.92`.
- v3 tries to improve decade stability using a champ-style refinement grid (permission/shock/risk/exit micro-knobs).

#### v3.0 — 10y champ_refine candidate pool (seeded from v2 kingmaker)
This is a 10y sweep that generates a **large refined pool** quickly:
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2016-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis champ_refine \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.90 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-30):
- `[PURGED false-timebars artifact]` contains **4,290 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v3.1 — Naive kingmaker eval (top by 10y pnl/dd) failed (cadence mismatch)
If we evaluate the **top by 10y pnl/dd**, we get 0:
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 400 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v3_champ_refine_15m_10y2y1y_mintr1000_top80.json
```

Outcome:
- `backtests/slv/slv_exec5m_v3_champ_refine_15m_10y2y1y_mintr1000_top80.json` is **empty**.
- Root cause: the v3 pool contains many “low-trade but high 10y pnl/dd” configs; they fail the **1y trades>=1000** contract constraint.

#### v3.2 — Kingmaker eval after re-ranking by 10y trade count (keeps the scalper honest)
Workaround: re-rank candidates by decade trade count first, then run kingmaker.

Create a ranking-friendly copy:
- `[PURGED false-timebars artifact]`
```bash
python - <<'PY'
import json
from pathlib import Path

src = Path("[PURGED false-timebars artifact]")
dst = Path("[PURGED false-timebars artifact]")
obj = json.loads(src.read_text())
for g in obj.get("groups", []):
    e = (g.get("entries") or [None])[0]
    if not isinstance(e, dict):
        continue
    m = e.get("metrics") or {}
    t = m.get("trades")
    if isinstance(t, (int, float)):
        m["pnl_over_dd"] = float(t)  # hack: force sort by trades
dst.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
print("Wrote", dst)
PY
```

Then evaluate:
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 400 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out [PURGED false-timebars artifact]
```

Outcome:
- **23** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.25** (promoted above as CURRENT v3).

### v4 — Shock/Risk scaling refine (cadence-locked) seeded from v3 winners
Status: **DONE** (raised worst-window stability vs v3; still below `roi/dd>=1.5..2.0` floor)

Key intent vs v3:
- Keep cadence **hard locked**: `tod=9–16`, no skip/cooldown, and lock `open_position_cap=5`.
- Sweep a **wider shock detection + risk scaling pocket** (primarily `shock_gate_mode=detect`, volatility-aware position scaling)
  without using `block` (avoid killing trade count).
- Add a small exit neighborhood for SLV (hold micro + optional `spot_close_eod=true` pocket).

#### v4.0 — 10y champ_refine pool (seeded from v3 kingmaker winners)
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2016-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis champ_refine \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-top 3 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.00 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- `[PURGED false-timebars artifact]` contains **775 eligible presets**
  (eligible under the 10y-window milestone gates used above).

Ranges swept (abridged):
- TOD: locked to `9–16 ET`, `skip_first_bars=0`, `cooldown_bars=0`
- `open_position_cap`: locked to `5`
- SL stop neighborhood: `spot_stop_loss_pct ∈ {0.008, 0.010, 0.012, 0.015, 0.020}`
- Flip hold neighborhood (SLV-only): `flip_exit_min_hold_bars ∈ {0, 2, 4}`
- Close EOD pocket (SLV-only): try `spot_close_eod=true` on the core stop+flip variant for `sl=1%`
- Shock variants (SLV-only):
  - `surf daily_atr_pct`: on/off from `(3.0/2.5)` up to `(6.0/5.0)` with `sl_mult ∈ {0.75, 1.0}`
  - `detect tr_ratio` and `detect atr_ratio`: fast/slow in `{3/21, 5/30, 7/50}`, ratios around `1.25/1.15 .. 1.45/1.30`,
    with `shock_risk_scale_target_atr_pct ∈ {3.0, 3.5, 4.0}` and `shock_risk_scale_min_mult ∈ {0.10, 0.20}`
  - `detect daily_atr_pct`: on/off around `4.0/3.5 .. 5.0/4.5` with risk scaling
  - `detect daily_drawdown`: `(lb=40 on=-10% off=-6%)`, `(lb=20 on=-7% off=-4%)` with down-risk scaling
- Risk overlays: locked to `risk=off` (avoid entry-canceling side effects during cadence exploration)

#### v4.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- **29** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.39** (promoted above as CURRENT v4).
- Notable: the promoted v4 champ is still **shock=off**; the stability lift came via a slightly different supertrend pocket.

### v5 — Seeded champ_refine around v4 (Candidate #1 follow-through)
Status: **DONE** (no dethrone; best ties v4 at worst-window `roi/dd=1.39`)

Key intent vs v4:
- Seed from the **v4 kingmaker winners** and raise the 10y milestone floor (`milestone-min-pnl-dd=1.25`) to reduce noise.
- Same cadence lock + shock/risk scaling pocket as v4; goal was to push worst-window `roi/dd >= 1.5` without reducing trade count.

#### v5.0 — 10y champ_refine pool (seeded from v4 kingmakers)
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2016-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis champ_refine \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.25 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- `[PURGED false-timebars artifact]` contains **639 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v5.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- **10** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.39**, tying v4 (no dethrone).

### v6 — Supertrend neighborhood widening (Candidate #2 tight sweep; dethroned v4)
Status: **DONE** (timestamp-bug era winner; now archived under LEGACY)

Key intent vs v5/v4:
- Keep cadence hard-locked (same as v4/v5), but widen the **Supertrend neighborhood** searched inside `champ_refine` for SLV:
  - include `supertrend_atr_period=5` in the neighborhood,
  - include `supertrend_multiplier=seed+0.10` (instead of an unrelated 0.45 pocket),
  - and allow 4 ATR candidates for SLV in stage-2 (still tight, but enough to escape the old ST pocket).

#### v6.0 — 10y champ_refine pool (seeded from v4 kingmakers)
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --use-rth --cache-dir db --offline \
  --start 2016-01-08 --end 2026-01-08 \
  --bar-size "15 mins" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis champ_refine \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.25 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- `[PURGED false-timebars artifact]` contains **721 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v6.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out [PURGED false-timebars artifact]
```

Outcome (2026-01-31):
- **36** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` was **1.70** (timestamp-bug era; not reproducible today at those metrics).
- Defining delta vs v4: `ST(5,0.75,hl2)` and `flip_exit_min_hold_bars=4` (the rest of the stack stays similar).
- ET-fixed rescored reference (reproducible today): `[PURGED false-timebars artifact]`

### v7 — FULL24 (24/5) pivot + 1h signals (no TOD; stability floor `roi/dd >= 3.5`)
Status: **DONE** (promoted above as Baseline v7; later dethroned by v10 and v25)

Key intent vs v6:
- Move from the legacy RTH/15m family into a **FULL24 (use_rth=false)** family (SMART+OVERNIGHT stitched) with a slower but cleaner signal clock (`1 hour`).
- Keep realism (`exec=5 mins`, intrabar exits) while preserving `>500 trades` in the 1y window.

Primary artifact:
- Multiwindow top-80: `[PURGED false-timebars artifact]`

### v8 — Attempt: `shock_velocity_refine_wide` seeded from v7 (did NOT dethrone)
Status: **DONE** (failed to lift the 1y stability floor vs v7)

Intent:
- Port the TQQQ playbook (TR-ratio shock sensitivity + TR-median ramp/gap overlays) onto SLV FULL24, seeded from v7.

Artifacts:
- Variant pool (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

### v8b — Attempt: `risk_overlays` (unscaled defaults) (catastrophic; archived)
Status: **DONE** (riskoff triggered in a way that crushed the 10y window; not promoted)

Artifacts:
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Outcome:
- Worst-window stability collapsed to **~0.99** (10y window became the floor).
- Lesson: unscaled TR% overlays tuned for higher-vol symbols can be actively harmful on SLV FULL24.

### v9 — Attempt: `risk_overlays` seeded from v7 (no effect; did NOT dethrone)
Status: **DONE** (all variants identical; overlays never activated at current ranges)

Run (FULL24, 1y gate `trades>=500`):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis risk_overlays --base champion \
  --seed-milestones [PURGED false-timebars artifact] \
  --min-trades 500 --top 25 \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 2500 \
  --milestones-out [PURGED false-timebars artifact]
```

Kingmaker (10y/2y/1y; strict pnl>0):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "1 hour" \
  --offline --cache-dir db --jobs 12 \
  --min-trades 500 --require-positive-pnl \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 80 \
  --out [PURGED false-timebars artifact]
```

Outcome:
- All 1y results are identical to v7 (pnl/dd ≈ **3.598**, trades=539), across 1,970 variants.
- Multiwindow top-80 contains v7 duplicates; stability floor stays **3.60** (no dethrone).
- Interpretation: current `risk_overlays` TR% thresholds are tuned for higher-vol symbols (e.g. TQQQ). For SLV, they’re too high → overlays don’t trigger → no change.

### v10 — Dethrone: `risk_overlays` seeded from v7, SLV-scaled TR% thresholds (rounding hides it)
Status: **DONE** (this strictly dethroned v7 on worst-window `roi/dd`; later edged out by v25)

Intent:
- Retry `risk_overlays` with much lower TR% med thresholds so SLV actually enters the risk-off/panic regimes.

Run (FULL24, 1y gate `trades>=500`):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis risk_overlays --base champion \
  --seed-milestones [PURGED false-timebars artifact] \
  --min-trades 500 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 0 --milestone-min-pnl-dd 0.00 \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 2500 \
  --milestones-out [PURGED false-timebars artifact]
```

Kingmaker (10y/2y/1y; strict pnl>0):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "1 hour" \
  --offline --cache-dir db --jobs 12 \
  --min-trades 500 --require-positive-pnl \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 80 \
  --out [PURGED false-timebars artifact]
```

Outcome:
- Best worst-window `roi/dd` is **3.601454743** (1y), which is strictly above v7’s **3.597984452**.
  (Both round to “3.60”, which is how this looked like a “no-dethrone” during fast iteration.)
- Trade count stayed above the contract: 1y `trades=529`.

### v11 — Attempt: `risk_overlays` (PANIC/OFF only; skip riskpop) (tied v10; no further lift)
Status: **DONE** (cheaper grid; stability ties v10)

Intent:
- Riskpop is a huge noisy grid; cut it entirely and focus on riskoff+riskpanic only.

Run (FULL24, 1y gate `trades>=500`):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis risk_overlays --base champion \
  --seed-milestones [PURGED false-timebars artifact] \
  --min-trades 500 --top 25 \
  --risk-overlays-skip-pop \
  --milestone-min-win 0.00 --milestone-min-trades 0 --milestone-min-pnl-dd 0.00 \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 2500 \
  --milestones-out [PURGED false-timebars artifact]
```

Kingmaker:
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "1 hour" \
  --offline --cache-dir db --jobs 12 \
  --min-trades 500 --require-positive-pnl \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 80 \
  --out [PURGED false-timebars artifact]
```

Outcome:
- Best worst-window `roi/dd` ties v10 at **3.601454743** (no further lift).

### v12 — Attempt: `risk_overlays` + `riskpanic_long_risk_mult_factor` sweep (tied v10; no further lift)
Status: **DONE** (validated “panic-long shrink” does not lift the SLV floor above v10)

Intent:
- Port the newest TQQQ lever (`riskpanic_long_risk_mult_factor`) into SLV FULL24 to see if it lifts the 1y stability floor.

Run (FULL24, 1y gate `trades>=500`, PANIC/OFF only):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis risk_overlays --base champion \
  --seed-milestones [PURGED false-timebars artifact] \
  --min-trades 500 --top 25 \
  --risk-overlays-skip-pop \
  --milestone-min-win 0.00 --milestone-min-trades 0 --milestone-min-pnl-dd 0.00 \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 2500 \
  --milestones-out [PURGED false-timebars artifact]
```

Kingmaker (10y/2y/1y; full paranoia: evaluate all 2500 variants):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "1 hour" \
  --offline --cache-dir db --jobs 12 \
  --top 2500 \
  --min-trades 500 --require-positive-pnl \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 80 \
  --out [PURGED false-timebars artifact]
```

Outcome:
- Best worst-window stability remained **3.601454743** (ties v10; no further lift).
- Best riskpanic-enabled variant also did not beat the base floor; `riskpanic_long_risk_mult_factor` did **not** transfer as a win lever from TQQQ → SLV.

### v13 — Attempt: `risk_overlays` + v39-style pre-panic linear scaling (tied v10; no further lift)
Status: **DONE** (ported the v39 “panic-long linear de-risking” semantics; still no lift above v10)

Intent:
- Import the **TQQQ v39** finding into SLV’s risk overlays:
  - include the missing TR-velocity pockets `trΔ>=0.25@1d` and `trΔ>=0.75@1d`, and
  - when `riskpanic_long_risk_mult_factor < 1.0` **and** TR-velocity gating is enabled, enable
    `riskpanic_long_scale_mode="linear"` (so long de-risking can start *before* the full panic regime).

Notes:
- No new CLI flags were added. This uses existing filter knobs:
  `riskpanic_long_scale_mode`, `riskpanic_tr5_med_delta_min_pct`, and `riskpanic_long_risk_mult_factor`.

Run (FULL24, 1y gate `trades>=500`, PANIC/OFF only):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2025-01-08 --end 2026-01-08 \
  --bar-size "1 hour" --spot-exec-bar-size "5 mins" \
  --realism2 --jobs 12 \
  --axis risk_overlays --base champion \
  --seed-milestones [PURGED false-timebars artifact] \
  --min-trades 500 --top 25 \
  --risk-overlays-skip-pop \
  --milestone-min-win 0.00 --milestone-min-trades 0 --milestone-min-pnl-dd 0.00 \
  --write-milestones --merge-milestones --milestone-add-top-pnl-dd 2500 \
  --milestones-out [PURGED false-timebars artifact]
```

Kingmaker (10y/2y/1y; strict pnl>0; evaluate top500 for runtime sanity):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "1 hour" \
  --offline --cache-dir db --jobs 12 \
  --top 500 \
  --min-trades 500 --require-positive-pnl \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 80 \
  --out [PURGED false-timebars artifact]
```

Outcome:
- Stability floor ties v10 at **3.601454743** (no further lift).
- In this pocket, the top results were dominated by `riskoff`-only variants; `riskpanic` + linear scaling
  did not lift worst-window roi/dd above v10.

### v14 — Fix: `shock_throttle_refine` becomes seed-aware (stop pocket + daily thresholds + detect-only fallback)
Status: **DONE** (this was sweep-harness correctness work)

Root bug:
- The early shock-throttle sweep was accidentally wearing a **TQQQ stop-loss pocket** (around `4.2–4.8%`),
  which is nonsensical for the SLV v7 family (`spot_stop_loss_pct=1.2%`).

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`

Ranges swept (exact pockets live in `tradebot/backtest/run_backtests_spot_sweeps.py` under `shock_throttle_refine`):
- Stop-loss pocket is now **anchored on the seed**: `{0.70,0.85,0.925,1.00,1.075,1.15,1.30} × seed_stop`
- Risk-scale targets (daily ATR% detector): derived from the seed’s `shock_daily_on/off_atr_pct` neighborhood
- `shock_risk_scale_min_mult ∈ {0.05, 0.10, 0.20}`
- If the seed had shock=off, the axis auto-enables `shock_gate_mode="detect"` + `shock_detector="daily_atr_pct"`
  (so the throttle has an ATR% stream without changing entry gating)

### v15 — Attempt: `shock_throttle_refine` seeded from v7 (dethrones v7; not v10)
Status: **DONE**

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Outcome:
- Worst-window `roi/dd` = **3.599155138** (beats v7, but does **not** beat v10’s **3.601454743**).
- Key lesson: throttle knobs alone can move the floor, but the v10 riskoff+cutoff lever was larger.

### v16–v20 — 15m exploration (FULL24) + cache drift autopsy (lane currently invalidated)
Status: **DONE** (exploration) + **BLOCKED** (reproducibility)

Artifacts:
- v16 15m variants (1y): `[PURGED false-timebars artifact]`
- v16 15m kingmaker (2y+1y): `backtests/slv/slv_exec5m_v16_shock_throttle_refine_15m_2y1y_mintr500_top80_20260206_025719.json`
- v17–v19 15m 2y variants (rapid probes): `[PURGED false-timebars artifact]`,
  `[PURGED false-timebars artifact]`,
  `[PURGED false-timebars artifact]`
- v20 rescore (current cache): `backtests/slv/slv_exec5m_v20_rescore_hf15m_2y1y_mintr500_top80_20260206_032150.json`

Outcome:
- Under the **current** SLV `15mins_full24` cache, rescoring shows **0** candidates pass the 2y+1y positivity contract
  at `min_trades=500` → the “15m dethrone” lane is dead until the cache drift is understood.

### v21–v22 — Attempt: TR-ratio throttle (detect-only) (did NOT lift the floor)
Status: **DONE**

Artifacts:
- v21 variants (1y): `[PURGED false-timebars artifact]`
- v21 top-80: `[PURGED false-timebars artifact]`
- v22 variants (1y): `[PURGED false-timebars artifact]`
- v22 top-80: `[PURGED false-timebars artifact]`

Ranges swept (from the axis pockets; see `tradebot/backtest/run_backtests_spot_sweeps.py` under `shock_throttle_tr_ratio`):
- `shock_scale_detector="tr_ratio"`
- TR-ratio fast/slow ∈ `{2/50,3/50,5/50,3/21}`
- `shock_risk_scale_target_atr_pct ∈ {0.25,0.35,0.45,0.55,0.7,0.9,1.1}`
- `shock_risk_scale_min_mult ∈ {0.05,0.10,0.20}`
- `shock_risk_scale_apply_to ∈ {"cap","both"}`

Outcome:
- No stability improvement vs the v7/v10 floor.
- Lesson: this “TR velocity throttle” is not the lever that fixes SLV’s worst 1y slice in this family.

### v23 — Dethrone: cap-aware throttle refinement (beats v7; not v10)
Status: **DONE**

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Outcome:
- Worst-window `roi/dd` = **3.599155138** (beats v7; not v10).
- Key knobs in the winner:
  - cap throttle: `shock_risk_scale_target_atr_pct=4.0`, `shock_risk_scale_min_mult=0.05`,
    `shock_risk_scale_apply_to="cap"`
  - daily surf tweak: `shock_daily_on_atr_pct=4.0`, `shock_daily_off_atr_pct=4.0` (on=off)

### v24 — Attempt: drawdown-driven throttle (seed selection bug; did NOT lift floor)
Status: **DONE** (but treat as “bad seed”)

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Outcome:
- Worst-window `roi/dd` fell to **3.596878105** (worse than v7/v10).
- Root cause: the seeded sweep selected the seed by **full-window pnl/dd**, not by stability,
  so it seeded from a non-champ group (see bracketed seed label in the variants file).

### v25 — Dethrone: drawdown-driven throttle (CURRENT) (edges v10 by +0.000206)
Status: **DONE** (promoted above as CURRENT v25)

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Ranges swept (from the axis pockets; see `tradebot/backtest/run_backtests_spot_sweeps.py` under `shock_throttle_drawdown`):
- `shock_scale_detector="daily_drawdown"`
- `shock_drawdown_lookback_days ∈ {10,20,40}`
- target drawdown magnitude (`shock_risk_scale_target_atr_pct`) ∈ `{3,4,6,8,10,12,15}`
- `shock_risk_scale_min_mult ∈ {0.05,0.10,0.20,0.30}`
- `shock_risk_scale_apply_to ∈ {"cap","both"}`

Outcome:
- New worst-window `roi/dd` = **3.601660671** (beats v10’s **3.601454743**).
- This is the first SLV dethrone that targets the real loss cluster: **equity drawdown regimes**, not TR% regimes.

### v26 — Tie-champ: TQQQ-inspired `riskpanic_micro` on top of v25 (better 10y; same floor)
Status: **DONE**

Artifacts:
- Variants (1y): `[PURGED false-timebars artifact]`
- Multiwindow top-80: `[PURGED false-timebars artifact]`

Ranges swept (from the axis pockets; see `tradebot/backtest/run_backtests_spot_sweeps.py` under `riskpanic_micro`):
- `risk_entry_cutoff_hour_et ∈ {None, 15}`
- `riskpanic_tr5_med_pct ∈ {2.75, 3.0, 3.25}`
- `riskpanic_neg_gap_ratio_min ∈ {0.5, 0.6}`
- `riskpanic_neg_gap_abs_pct_min ∈ {None, 0.005}`
- `riskpanic_tr5_med_delta_min_pct ∈ {None, 0.25, 0.5, 0.75}`
- `riskpanic_long_risk_mult_factor ∈ {1.0, 0.4, 0.0}`
- `riskpanic_long_scale_mode ∈ {None, "linear"}`

Outcome:
- Stability floor ties v25 at **3.601660671** (riskpanic does not trigger in the last 2y/1y windows).
- 10y ratio improves materially (v26 10y `roi/dd=3.93` vs v25 10y `3.73`).

### v27 — Pivot: 10m FULL24 high-activity lane (cache-first; currently negative over 10y)
Status: **DONE** (exploration) + **NOT GOOD ENOUGH** (performance)

Cache work (reproducibility-first):
- New bar size: `10 mins` support (signals + UI).
- Cache safety: atomic write + sort/dedupe on read/write (`tradebot/backtest/data.py`).
- Deterministic resample tool: `tradebot/backtest/tools/cache_ops.py resample`.
- Generated (from `5 mins` FULL24 cache, no IBKR refetch drift):
  - `db/SLV/SLV_2016-01-08_2026-01-08_10mins_full24.csv`
  - rows=260,475 (bars), dropped incomplete 10m buckets=569

Commands used:
- Build the 10m FULL24 cache from the existing 5m FULL24 cache:
  - `python -m tradebot.backtest.tools.cache_ops resample --symbol SLV --start 2016-01-08 --end 2026-01-08 --src-bar-size "5 mins" --dst-bar-size "10 mins" --cache-dir db`
- 1y discovery sweep (pnl-first shortlist): `hf_scalp`
  - `python -m tradebot.backtest spot --symbol SLV --start 2025-01-08 --end 2026-01-08 --bar-size "10 mins" --spot-exec-bar-size "5 mins" --axis hf_scalp --base default --offline --cache-dir db --realism2 --min-trades 500 --milestone-min-trades 500 --milestone-min-win 0 --milestone-min-pnl-dd 0 --write-milestones --milestones-out [PURGED false-timebars artifact] --top 80`
- Multiwindow eval (10y/2y/1y stability check):
  - `python -m tradebot.backtest spot_multitimeframe --milestones [PURGED false-timebars artifact] --symbol SLV --bar-size "10 mins" --offline --cache-dir db --top 200 --min-trades 200 --min-win 0 --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 --write-top 80 --out [PURGED false-timebars artifact]`

Artifacts:
- 1y milestones: `[PURGED false-timebars artifact]`
- 10y/2y/1y top-80: `[PURGED false-timebars artifact]`

Outcome:
- Best 1y candidate found was only a **tiny** winner:
  - `tr≈534`, `roi≈+0.93%`, `pnl≈+$926`, `dd%≈6.8%` (EMA `4/9`, stop `0.0060`, cooldown `2`, TOD `9–16`).
- On the 10y/2y/1y multiwindow contract, **everything is deeply negative over 10y**:
  - best `stability(min roi/dd)` is about **-0.96**, with full-window `roi≈-78%` and `dd%≈80%`.

Conclusion:
- 10m cadence (FULL24) + this stop+flip family is currently a churn machine for SLV.
- If we continue 10m, the next lever is likely a **different exit model** (profit target / cut-on-flip even when red / close_eod),
  not more micro-tuning of shock/risk overlays.

### v28 — Exit-model pivot (seeded from v25) (10m FULL24, 2y; still negative)
Status: **DONE** (exit-model lever tested) + **FAIL** (still negative)

Goal:
- Keep the core v25/v7 “trend+stop” family intact, but try the “trade more often” premise on `10 mins`
  by sweeping the exit semantics (PT/SL + close_eod + flip-only-if-profit).

Seed transform (so seeded sweeps can run on `10 mins` without rewriting the seed generator):
- Input: `[PURGED false-timebars artifact]`
- Output: `[PURGED false-timebars artifact]`
  - Overwrites `strategy.signal_bar_size="10 mins"` and `strategy.signal_use_rth=false` for all entries (metrics remain from v25).

Sweep command (2y window, FULL24):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2024-01-08 --end 2026-01-08 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis exit_pivot --seed-milestones [PURGED false-timebars artifact] --seed-top 2 \
  --min-trades 2000 --top 25 \
  --milestone-min-win 0 --milestone-min-trades 2000 --milestone-min-pnl-dd -999 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Artifacts:
- log: `backtests/slv/slv_exec5m_v28_exit_pivot_seedv25_10m_2y_log_20260206_213020.txt`
- milestones: `[PURGED false-timebars artifact]`

Outcome:
- Best 2y pnl among the high-trade variants is still deeply negative: **~-$46.8k** (ROI **~-46.8%**, trades **~2,214**).
- This is a strong signal that **FULL24 10m** in this family is not “more data = more edge”; it’s “more churn = more costs”.

### v29 — 10m FULL24 + TOD restriction (9–16 ET) (2y; less bad but still negative)
Status: **DONE** (hypothesis tested) + **FAIL** (still negative)

Hypothesis:
- Maybe 10m dies because the family is getting dragged through overnight noise/zero-volume bars.
- Try an entry window restriction (`entry_start_hour_et=9`, `entry_end_hour_et=16`) while still running FULL24 bars.

Seed transform:
- Output: `[PURGED false-timebars artifact]` (adds TOD filter to all groups).

Sweep command:
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline \
  --start 2024-01-08 --end 2026-01-08 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis exit_pivot --seed-milestones [PURGED false-timebars artifact] --seed-top 2 \
  --min-trades 1500 --top 25 \
  --milestone-min-win 0 --milestone-min-trades 1500 --milestone-min-pnl-dd -999 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Artifacts:
- log: `backtests/slv/slv_exec5m_v29_exit_pivot_seedv25_tod916_10m_2y_log_20260206_214157.txt`
- milestones: `[PURGED false-timebars artifact]`

Outcome:
- Best 2y pnl is still negative: **~-$26.8k** (ROI **~-26.8%**, trades **~1,602**).
- Conclusion: “FULL24 signals + TOD entry restriction” is not the fix; the overnight bars are still poisoning the signal stream.

### v30 — 10m RTH lane (use_rth=true) (2y positive; decade fails)
Status: **DONE** (key diagnostic) + **MIXED** (2y improves; 10y collapses)

Key move:
- Instead of FULL24 bars + filters, run **RTH-only bars** (`--use-rth`) so the 10m signal stream is not polluted by overnight bars.

Cache (deterministic; no IBKR refetch drift):
- Built 10m RTH cache from the existing 5m RTH cache:
  - `python -m tradebot.backtest.tools.cache_ops resample --symbol SLV --start 2016-01-08 --end 2026-01-08 --src-bar-size "5 mins" --dst-bar-size "10 mins" --use-rth --cache-dir db`
  - output: `db/SLV/SLV_2016-01-08_2026-01-08_10mins_rth.csv` (rows=97,668)

Seed transform:
- Output: `[PURGED false-timebars artifact]` (sets `signal_use_rth=true`).

2y sweep (RTH):
```bash
python -u -m tradebot.backtest spot \
  --symbol SLV --cache-dir db --offline --use-rth \
  --start 2024-01-08 --end 2026-01-08 \
  --bar-size "10 mins" --spot-exec-bar-size "5 mins" \
  --realism2 \
  --axis exit_pivot --seed-milestones [PURGED false-timebars artifact] --seed-top 2 \
  --min-trades 1000 --top 25 \
  --milestone-min-win 0 --milestone-min-trades 1000 --milestone-min-pnl-dd -999 \
  --write-milestones --milestones-out [PURGED false-timebars artifact]
```

Outcome (2y):
- We finally see **positive pnl** winners on 2y:
  - best: **~+$17.3k**, ROI **~+17.3%**, DD% **~17.4%**, trades **~1,055** (2y)
  - best knob delta vs seed: `spot_close_eod=true`, stop-only `SL≈0.008`, PT off.

But on decade:
- Multiwindow eval with positivity gate finds **0** candidates that stay positive across **10y/2y/1y**:
  - log: `backtests/slv/slv_exec5m_v30_exit_pivot_seedv25_10m_rth_10y2y1y_mintr500py_log_20260206_215657.txt`
  - empty top80: `[PURGED false-timebars artifact]`
- Even the best 2y winner is catastrophic on 10y (example diagnostic on top-1):
  - decade ROI ≈ **-75%**, DD% ≈ **83%**, trades ≈ **5,425**

Conclusion:
- 10m can “work” in recent years under RTH-only bars, but **it does not generalize to the decade** under this family.
- If we continue 10m, we must stop generating candidates on 1y/2y and instead run a **10y-first candidate sweep**
  (e.g. compact `combo_full` preset runs on `10 mins` RTH), then re-evaluate 2y/1y for stability.

### v31 — FULL24 lane diagnosis + time-window rescue (AB + micro)
Status: **DONE** (diagnostic) + **SUCCESS** (found the first positive FULL24 10m rescue pocket)

Hypothesis:
- FULL24 10m failure is mostly an overnight/noise execution issue, not necessarily a core signal issue.
- Keep `signal_use_rth=false`, but restrict **entry hours** and rebalance stop/short settings.

AB test (same strategy, lane behavior changed):
- RTH run: `[PURGED false-timebars artifact]` -> pnl ≈ **+$68.4k**
- FULL24 run: `[PURGED false-timebars artifact]` -> pnl ≈ **-$74.2k**
- FULL24 + entry-hour gating: `[PURGED false-timebars artifact]` -> pnl ≈ **-$20.9k**

Micro rescue sweep:
- Milestones: `[PURGED false-timebars artifact]` (144 combos)
- Eval: `[PURGED false-timebars artifact]`

Ranges swept (joint):
- `entry_start_hour_et ∈ {8,9,10}`
- `entry_end_hour_et ∈ {15,16}`
- `entry_confirm_bars ∈ {0,1,2}`
- `flip_exit_min_hold_bars ∈ {0,1}`
- `spot_stop_loss_pct ∈ {0.012,0.016}`
- `spot_short_risk_mult ∈ {1.0,0.005}`

Outcome:
- First positive FULL24 10m island appeared.
- Top decade result in the micro pass: pnl ≈ **+$37.3k**, trades ≈ **2,276**, `roi/dd≈0.54`.

### v32 — Plus2 joint sweep (new kingmaker lineage; all-window dethrone)
Status: **DONE** + **CHAMPION DETHRONE**

Goal:
- Expand directly around the v31 rescue pocket with a compact joint matrix that can win on decade while preserving 2y/1y.

Stage command pattern:
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --window 2016-01-08:2026-01-08
```

Plus2 ranges (192 combos):
- Entry window pockets: `(8,15)`, `(9,15)`, `(10,15)`
- `ema_preset ∈ {"5/13","8/21"}`
- Regime pockets:
  - `supertrend @ 4 hours` -> `ST(7,0.5,hl2)`
  - `supertrend @ 1 day` -> `ST(7,0.4,close)`
- `spot_stop_loss_pct ∈ {0.016,0.020}`
- `spot_short_risk_mult ∈ {1.0,0.0025,0.005,0.01}`
- `entry_confirm_bars ∈ {0,1}`

Artifacts:
- Discovery: `[PURGED false-timebars artifact]`
- Promotion set: `[PURGED false-timebars artifact]`
- Multiwindow promotion eval: `[PURGED false-timebars artifact]`

Outcome:
- Strong new island: top decade pnl in discovery ≈ **+$672.4k**.
- Promotion result: **77/80** candidates beat v25 champion pnl across **10y/2y/1y**.
- This produced the new kingmaker entries in the champion stack above.

### v33 — High-trade island mapping (broad + targeted gates)
Status: **DONE** + **PARTIAL** (high-trade positive region found, but no high-trade dethrone)

Goal:
- Push turnover up while preserving positivity and searching for all-window champion-level pnl.

Stage A (broad map):
- Milestones: `[PURGED false-timebars artifact]` (648 combos)
- Eval: `[PURGED false-timebars artifact]`
- Constraint: `min_trades=2500` on 10y window

Stage A ranges (joint):
- Entry windows: `(7,15)`, `(7,16)`, `(8,15)`, `(8,16)`, `(9,16)`, `(10,16)`
- `ema_preset ∈ {"3/7","4/9","5/13"}`
- Regimes:
  - `4h ST(7,0.5,hl2)`
  - `4h ST(10,0.4,close)`
  - `1d ST(7,0.4,close)`
- `spot_stop_loss_pct ∈ {0.012,0.016}`
- `spot_short_risk_mult ∈ {1.0,0.0025,0.005}`
- `entry_confirm_bars ∈ {0,1}`

Stage B (targeted gate overlays on high-trade seeds):
- Milestones: `[PURGED false-timebars artifact]` (176 combos)
- Eval: `[PURGED false-timebars artifact]`
- Constraint: `min_trades=3500`

Stage B ranges:
- EMA geometry pocket:
  - `ema_spread_min_pct ∈ {0.0,0.001,0.0015}`
  - `ema_spread_min_pct_down ∈ {0.005,0.01}`
  - `ema_slope_min_pct ∈ {0.0,0.005}`
- Riskpanic pocket:
  - `riskpanic_tr5_med_pct ∈ {2.5,2.75,3.0}`
  - `riskpanic_tr5_med_delta_min_pct ∈ {0.25,0.5}`
  - `riskpanic_long_risk_mult_factor ∈ {0.2,0.0}`
  - fixed: `neg_ratio=0.6`, `abs_gap=0.005`, `lookback=5`, `scale=linear`
- Shock/drawdown pocket:
  - `daily_drawdown` scale with lookback `{10,20}`, target `{8.0}`, min mult `{0.1}`
  - `tr_ratio` detect pockets (`3/21` and `3/50`) with light caps/throttle
- Tiny cross pocket: 2 combined tuples per seed

Outcome:
- High-trade positive region exists (best ≈ **+$73.4k**, trades ≈ **3,501**).
- No candidate with trades above the old v25 count achieved champion-level pnl.

### v34 — Aggressiveness scaling on high-trade seeds
Status: **DONE** + **FAIL** (more trades achieved; champion pnl not reached)

Goal:
- Test whether leverage/exposure scaling can lift the high-trade island to champion pnl.

Milestones + eval:
- `[PURGED false-timebars artifact]` (504 combos)
- `[PURGED false-timebars artifact]`
- Constraint: `min_trades=3900`

Scaling ranges:
- `spot_risk_pct ∈ {0.01,0.015,0.02,0.03,0.04,0.05,0.07}`
- `open_position_cap ∈ {3,5,8}`
- `spot_max_notional_pct ∈ {0.5,0.75,1.0}`

Outcome:
- Trade count increased materially (up to ~**12,300** trades on 10y).
- Best pnl stayed far below champion (top ≈ **+$54.9k**).
- Conclusion: naive exposure scaling does not bridge the “high-trade vs high-expectancy” gap.

### v35 — `island_bridge` crisis-overlay pass (km61-anchored, FULL24 10m) + precision audit
Status: **DONE** + **PARTIAL** (10y pnl lift achieved; no all-window dethrone)

Goal:
- Recenter on the km61 lineage and inject v25 + TQQQ-style overlays (drawdown throttle, TR-ratio shock detect, riskpanic velocity)
  to push decade pnl while preserving 2y/1y floors.

Runner:
- `backtests/slv/slv_10m_attack_hf.py` profile: `island_bridge`

Key commands:
```bash
python backtests/slv/slv_10m_attack_hf.py \
  --profiles island_bridge \
  --lane-mode full24 \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-bar-size "10 mins" --seed-kingmaker-id 61 \
  --champion-milestones [PURGED false-timebars artifact] \
  --champion-bar-size "10 mins" --champion-kingmaker-id 61 \
  --stage-window 2016-01-08:2026-01-08 \
  --stage-min-trades-per-year 100 --promotion-min-trades-per-year 100 \
  --jobs 12 --offline --run-id slv_exec5m_islandbridge_20260207
```

Artifacts:
- Stage A ranked: `[PURGED false-timebars artifact]`
- Stage B ranked: `[PURGED false-timebars artifact]`
- Promotion top-80: `[PURGED false-timebars artifact]`
- Precision audit (10y/2y/1y): `[PURGED false-timebars artifact]`

Search matrix:
- Stage A (96):
  - `ema_preset ∈ {"8/21","5/13"}`; `entry_confirm_bars ∈ {1,0}`
  - `spot_stop_loss_pct ∈ {0.018,0.020}`; `open_position_cap ∈ {3,4}`
  - entry windows `(8,15)`, `(9,16)`, `off`; `spot_short_risk_mult ∈ {1.0,0.0025}`
  - fixed: `signal_bar_size=10 mins`, `spot_exec_bar_size=5 mins`, `signal_use_rth=false`
- Stage B (360):
  - drawdown throttle pocket (`shock_scale_detector=daily_drawdown`)
  - TR-ratio shock pocket (`shock_scale_detector=tr_ratio`, detect/surf subsets)
  - riskpanic velocity pocket (`riskpanic_tr5_med_pct`, `riskpanic_tr5_med_delta_min_pct`, `riskpanic_long_risk_mult_factor`)
  - tiny cross interactions

Outcome:
- Best decade pnl: **KINGMAKER #56** -> pnl ≈ **+$833.2k** (vs baseline km52 ≈ **+$822.5k**), trades ≈ **1,501**
- Near-dethrone with stable recent windows: **KINGMAKER #50** -> pnl ≈ **+$833.0k**, trades ≈ **1,238**
- Promotion check: **no candidate beat baseline pnl in all 10y/2y/1y windows**
  - km50 ties 2y/1y exactly, beats only 10y
  - km56 beats 10y, but slightly under baseline on 2y/1y
- High-trade beater attempt: **KINGMAKER #108** reaches trades ≈ **2,533** on 10y, but pnl quality drops materially.

Precision-audit diagnosis (entry vs local 10m extrema):
- km52/km50 family (lower-trade, higher quality):
  - 1y `dist_med ≈ 0.31%`, `<=0.5% ≈ 69.5%`, `mfe/mae ≈ 0.75`
- km56/km61 family (higher-trade, weaker recent precision):
  - 1y `dist_med ≈ 0.78%`, `<=0.5% ≈ 40.8%`, `mfe/mae ≈ 0.16`
- km108 (max-turnover mode):
  - 1y `dist_med ≈ 1.03%`, `<=0.5% ≈ 23.7%`, `mfe/mae ≈ 0.10`

Interpretation:
- The decade lift is real, but it comes from a regime family that does not improve recent-window execution quality.
- Increasing turnover past the km50/km52 pocket degrades entry precision too sharply, which explains why recent pnl stalls or slips.

### v36 — `opt13` hour-expansion + precision-guard passes (post-v35 follow-up)
Status: **DONE** + **PARTIAL** (strong 10y pockets found; no all-window dethrone vs CURRENT v31)

Goal:
- Probe post-v35 pockets with two targeted profiles:
  - `hour_expansion`: widen FULL24 entry-hour windows from the v35 lineage while adding selective crisis overlays.
  - `precision_guard`: tighten around km50/km52-style quality pockets with narrower overlays.

Runner:
- `backtests/slv/slv_10m_attack_hf.py` profiles: `hour_expansion`, `precision_guard`

Command pattern:
```bash
python backtests/slv/slv_10m_attack_hf.py \
  --profiles hour_expansion,precision_guard \
  --lane-mode full24 \
  --seed-milestones [PURGED false-timebars artifact] \
  --seed-bar-size "10 mins" --seed-kingmaker-id <50|56|61> \
  --champion-milestones [PURGED false-timebars artifact] \
  --champion-bar-size "10 mins" --champion-kingmaker-id <50|56|61> \
  --stage-window 2016-01-08:2026-01-08 \
  --stage-min-trades-per-year 100 --promotion-min-trades-per-year 100 \
  --jobs 12 --offline --run-id slv_exec5m_opt13_<...>
```

Search matrix:
- `hour_expansion` Stage A (288):
  - entry windows from `6..9` start and `14..16` end, `risk_cutoff` aligned to end hour
  - `ema_preset ∈ {"8/21","5/13"}`, `entry_confirm_bars=1`
  - `spot_stop_loss_pct ∈ {0.020,0.022}`, `open_position_cap ∈ {3,4}`, `spot_short_risk_mult ∈ {0.0,0.0025}`
  - geometry pocket: `(spread, spread_down, slope) ∈ {(0.0015,0.02,0.01),(0.0025,0.03,0.02)}`
- `hour_expansion` Stage B (832):
  - drawdown throttle + TR-ratio detect/surf + riskpanic velocity + tiny cross tuples
  - includes `shock_scale_detector ∈ {"daily_drawdown","tr_ratio"}`, `shock_gate_mode ∈ {"detect","surf"}`
- `precision_guard` Stage A (96):
  - entry windows `{8,9}` to `{14,15}`, `ema_preset="8/21"`, `entry_confirm_bars ∈ {1,2}`
  - `spot_stop_loss_pct ∈ {0.018,0.020}`, `open_position_cap ∈ {2,3}`, `spot_short_risk_mult=0.0`
- `precision_guard` Stage B (330):
  - narrow drawdown/TR-ratio/riskpanic overlays; `shock_gate_mode="detect"` only
  - `spot_short_risk_mult ∈ {0.0,0.001}` in staged variants

Outcome:
- Best single-window (10y-only ranking) top tags reached `pnl ≈ +$846.3k` in hour-expansion promotion sets.
- Full 10y/2y/1y checks on promoted sets remained mixed:
  - best 10y pockets (`hourx56/61`) around `pnl ≈ +$825.7k`, but 1y floor around `pnl/dd ≈ 3.72`.
  - best 1y floor pocket (`precision50`) around `pnl/dd ≈ 3.99`, but decade pnl materially below island-bridge leaders.
- Net: no all-window dethrone over CURRENT v31 (`roi/dd`: `6.362 / 5.451 / 4.175` for 10y/2y/1y).

Cleanup note:
- Generated `opt13` artifacts were local exploration outputs only and were removed from workspace after logging
  (JSON/log bundles for `hour_expansion` + `precision_guard`) to keep the repo lean.

### v37 — Shock-hysteresis wall sweep (single-process, cache-reset) + new king
Status: **DONE** + **CHAMPION DETHRONE**

Goal:
- Validate whether the discovered `shock_on/off=1.35/1.25` pair is a true robust optimum (not a local bump),
  and locate the upper/lower wall where quality degrades.

Method guardrails (explicitly enforced):
- `spot_multitimeframe` run in **single-process mode** only (`--jobs 1`)
- eval cache reset before each pass (`rm -f db/spot_multiwindow_eval_cache.sqlite3*`)
- fixed windows for every pass:
  - `2016-01-08:2026-01-08`
  - `2024-01-08:2026-01-08`
  - `2025-01-08:2026-01-08`

Pass ranges:
- Pass A (`N=12`): coarse-tight around baseline + discovered duo:
  - `(1.500,1.400)`, `(1.400,1.300)`, `(1.360,1.260)`, `(1.350,1.250)`,
    `(1.345,1.245)`, `(1.340,1.240)`, `(1.335,1.235)`, `(1.330,1.230)`,
    `(1.325,1.225)`, `(1.320,1.220)`, `(1.315,1.215)`, `(1.310,1.210)`
- Pass B (`N=12`): lower-edge probe below `1.31`:
  - `(1.315,1.215)` down to `(1.250,1.150)` (step `0.005` in the core)
- Pass C (`N=11`): upper-edge micro probe:
  - `(1.350,1.250)` to `(1.360,1.260)` at `0.001` increments

Exact rerunnable commands (equivalent to executed run):
```bash
rm -f db/spot_multiwindow_eval_cache.sqlite3 db/spot_multiwindow_eval_cache.sqlite3-shm db/spot_multiwindow_eval_cache.sqlite3-wal
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --jobs 1 --top 12 --min-trades 0 \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 12 --out [PURGED false-timebars artifact]

rm -f db/spot_multiwindow_eval_cache.sqlite3 db/spot_multiwindow_eval_cache.sqlite3-shm db/spot_multiwindow_eval_cache.sqlite3-wal
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --jobs 1 --top 12 --min-trades 0 \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 12 --out [PURGED false-timebars artifact]

rm -f db/spot_multiwindow_eval_cache.sqlite3 db/spot_multiwindow_eval_cache.sqlite3-shm db/spot_multiwindow_eval_cache.sqlite3-wal
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --jobs 1 --top 11 --min-trades 0 \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 11 --out [PURGED false-timebars artifact]
```

Persisted artifacts:
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`

Outcome (wall map):
- **Top plateau:** `shock_on_ratio ∈ [1.300, 1.355]`, `shock_off_ratio = on - 0.10`
- **Upper break:** starts at `1.356/1.256` (10y/2y soften), and from `>=1.357` 1y also softens.
- **Lower break:** below `1.300/1.200` stability/1y quality drops.
- **10y outlier:** `1.295/1.195` gives a higher 10y pnl pocket, but lower worst-window stability and lower 1y pnl.

Promoted king (canonical pick):
- `shock_on_ratio=1.35`, `shock_off_ratio=1.25` (same top metrics as the whole plateau)
- Worst-window `roi/dd=4.204180092` (1y), with all-window pnl lift vs the prior `1.50/1.40` baseline.

### v38 — Single-position parity promotion (v31.2) + champ handoff
Status: **DONE** + **CHAMPION DETHRONE (parity-first)**

Goal:
- Remove the non-live simultaneous-position holding behavior from the promoted SLV lane.
- Promote a live-parity single-position config as CURRENT.

What changed:
- Forced single-position behavior in the promoted preset: `open_position_cap=1`
- Retuned single-position sizing/exits:
  - `spot_max_notional_pct=0.7`
  - `spot_risk_pct=0.016`
  - `spot_stop_loss_pct=0.016`
- Kept the v31.1 shock wall core unchanged: `shock_on/off=1.35/1.25`

Exact rerunnable command:
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones [PURGED false-timebars artifact] \
  --symbol SLV --bar-size "10 mins" --offline --cache-dir db \
  --jobs 1 --top 1 --min-trades 0 \
  --window 2016-01-08:2026-01-08 \
  --window 2024-01-08:2026-01-08 \
  --window 2025-01-08:2026-01-08 \
  --write-top 1 --out [PURGED false-timebars artifact]
```

Persisted artifacts:
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`
- `[PURGED false-timebars artifact]`

Outcome:
- New CURRENT is **v31.2** (single-position parity lane).
- Simultaneous-position holding is de-promoted from CURRENT because it has no live parity.
- Versus v31.1 stacked champ, all-window `roi/dd` improves, while total `pnl` and trade count are lower.


## Notes

- We do **not** assume `spot_close_eod=true` is necessary; we will explicitly sweep / compare it.
- For any strategy that looks “too good”, validation must include:
  - `--realism2` costs
  - `spot_drawdown_mode=intrabar` + liquidation marking
  - (optional) stress slices (crash windows) once we have finalists
