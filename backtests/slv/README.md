# SLV Research (High-Activity Spot Scalper)

This folder is the **single source of truth** for SLV research going forward:
- the **current champion stack** (promoted configs + expected multiwindow metrics), and
- the **evolution log** (commands, parameter ranges swept, what changed vs prior).

We are explicitly targeting the new contract:
- **Symbol:** `SLV`
- **Instrument:** spot (underlying)
- **Signal timeframe:** **15 mins**
- **Execution timeframe:** **5 mins** (`spot_exec_bar_size="5 mins"`) for realism (intrabar stops + next-open fills)
- **Goal:** **>= 1000 trades / year** (high activity) while preserving stability across **10y / 2y / 1y**

## Current Champions (stack)

### CURRENT (v6) — First above-floor stability lift (worst-window `roi/dd >= 1.5`)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- keeps the desired **high activity** in the 1y window (`>=1000 trades`)
- and meaningfully improves worst-window stability vs v4 (10y `roi/dd` lifted from **`1.39`** → **`1.70`**)

**v6 kingmaker #1** (from `backtests/slv/slv_exec5m_v6_st_neighborhood_champ_refine_15m_10y2y1y_mintr1000_top80.json`)
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
- `max_open_trades=5`, `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- Note: this winner is still “simple” (no extra shock/risk overlays); the gain came from a better Supertrend + hold pocket.

### Previous CURRENT (v4) — First decade stability lift from the v3 baseline (still below floor)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- keeps the desired **high activity** in the 1y window (`>=1000 trades`)
- and improves worst-window stability vs v3 (10y `roi/dd` lifted from ~`1.25` → **`1.39`**)

**v4 kingmaker #1** (from `backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y2y1y_mintr1000_top80.json`)
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
- `max_open_trades=5`, `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- Note: even though v4 sweeps expanded shock/risk scaling pockets, the current best is still **shock=off** (pure regime+stack wins again).

### Previous CURRENT (v3) — First stability lift above the v2 baseline (still below floor)
This is the first SLV family that:
- stays **positive PnL** across **10y / 2y / 1y**
- keeps the desired **high activity** in the 1y window (`>=1000 trades`)
- and meaningfully improves worst-window stability vs v2.

**v3 kingmaker #1** (from `backtests/slv/slv_exec5m_v3_champ_refine_ranktrades_15m_10y2y1y_mintr1000_top80.json`)
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
- `max_open_trades=5` (stacking; currently required for decade positivity at this cadence)
- `spot_close_eod=false`, `spot_short_risk_mult=0.01`
- No extra permission/shock/risk-pop overlays in this top pick (surprisingly, the simple regime+stack wins here)

### Previous CURRENT (v2) — First 10y/2y/1y-positive high-activity baseline (below stability floor)
This is the first SLV family that is:
- Positive PnL in **all 3 windows** (10y + 2y + 1y)
- Meets the activity constraint (`>=1000 trades` in 1y)

It is **not yet** “contract-ready” because worst-window `roi/dd` is still well below the target floor
(initially `>=1.5..2.0`, eventually aiming near the TQQQ champ ≈3.49).

**v2 kingmaker #1** (from `backtests/slv/slv_exec5m_v2_hf_scalp_15m_10y2y1y_mintr1000_top80.json`)
- Worst-window `roi/dd`: **~0.92**
- 10y: pnl ≈ **$73,611**, trades ≈ **6,199**
- 1y: trades ≈ **1,150**

Defining shape (abridged):
- `ema_preset=21/50`, `ema_entry_mode=trend`, `entry_confirm_bars=0`
- `spot_profit_target_pct=None`, `spot_stop_loss_pct=0.010` (stop-only)
- `exit_on_signal_flip=true`, `flip_exit_only_if_profit=true`, `flip_exit_min_hold_bars=0`
- Regime: `supertrend @ 4 hours` (`ST(7,0.5,hl2)`)
- Session: `entry_start/end=9–16 ET`
- `max_open_trades=5` (stacking; currently required for decade positivity at this cadence)

Promotion checklist (minimum):
- Positive PnL in **all 3** windows (10y + 2y + 1y)
- 1y trades `>= 1000`
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
- Signal bars: `15 mins` (RTH)
- Execution bars: `5 mins` (RTH)
- Regime bars: usually `4 hours` (RTH), unless we change regime_bar_size

Notes:
- Cache now includes `5 mins`, `15 mins`, `4 hours`, `1 day` for SLV over `2016-01-08 → 2026-01-08`.
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
2) **1y prefilter**: require `>=1000 trades` + `pnl>0` on 1y
3) **10y/2y/1y kingmaker**: maximize the **worst-window** `roi/dd`, keep `pnl>0` in all 3 windows
4) Promote winners into the **Current Champions** stack above

## Evolutions (stack)

### v0 — High-activity scalper scaffolding (15m signals, 5m exec; RTH; multiwindow kingmaker)
Status: **DONE** (combo_fast baseline failed activity+profit under costs)

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
  --axis combo_fast --min-trades 0 --top 50 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.00 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v0_combo_fast_15m_1y_milestones.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v0_combo_fast_15m_1y_milestones.json` contains **0 eligible presets** at
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
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v1_hf_scalp_15m_1y_milestones.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v1_hf_scalp_15m_1y_milestones.json` contains **399 eligible presets**.

#### v1.1 — 10y/2y/1y kingmaker eval (stability-first)
Strict gate (must be positive pnl in all windows):
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_exec5m_v1_hf_scalp_15m_1y_milestones.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_top80.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_top80.json` is **empty** (0 candidates passed).
- Diagnostic finding: **10y pnl is negative for 399/399 candidates** in the v1 pool.

Diagnostics (no-positive gate removed; useful for inspection / debugging):
- Top-80 by “least-bad stability”: `backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_top80_nopos.json`
- Full evaluated set (all 399): `backtests/slv/slv_exec5m_v1_hf_scalp_15m_10y2y1y_mintr1000_all.json`

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
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v2_hf_scalp_15m_1y_milestones.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v2_hf_scalp_15m_1y_milestones.json` contains **872 eligible presets**.

#### v2.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_exec5m_v2_hf_scalp_15m_1y_milestones.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v2_hf_scalp_15m_10y2y1y_mintr1000_top80.json
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
  --seed-milestones backtests/slv/slv_exec5m_v2_hf_scalp_15m_10y2y1y_mintr1000_top80.json \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 0.90 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones.json
```

Outcome (2026-01-30):
- `backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones.json` contains **4,290 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v3.1 — Naive kingmaker eval (top by 10y pnl/dd) failed (cadence mismatch)
If we evaluate the **top by 10y pnl/dd**, we get 0:
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones.json \
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
- `backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones_rank_trades.json`
```bash
python - <<'PY'
import json
from pathlib import Path

src = Path("backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones.json")
dst = Path("backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones_rank_trades.json")
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
  --milestones backtests/slv/slv_exec5m_v3_champ_refine_15m_10y_milestones_rank_trades.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 400 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v3_champ_refine_ranktrades_15m_10y2y1y_mintr1000_top80.json
```

Outcome:
- **23** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.25** (promoted above as CURRENT v3).

### v4 — Shock/Risk scaling refine (cadence-locked) seeded from v3 winners
Status: **DONE** (raised worst-window stability vs v3; still below `roi/dd>=1.5..2.0` floor)

Key intent vs v3:
- Keep cadence **hard locked**: `tod=9–16`, no skip/cooldown, and lock `max_open_trades=5`.
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
  --seed-milestones backtests/slv/slv_exec5m_v3_champ_refine_ranktrades_15m_10y2y1y_mintr1000_top80.json \
  --seed-top 3 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.00 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y_milestones.json
```

Outcome (2026-01-31):
- `backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y_milestones.json` contains **775 eligible presets**
  (eligible under the 10y-window milestone gates used above).

Ranges swept (abridged):
- TOD: locked to `9–16 ET`, `skip_first_bars=0`, `cooldown_bars=0`
- `max_open_trades`: locked to `5`
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
  --milestones backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y_milestones.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y2y1y_mintr1000_top80.json
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
  --seed-milestones backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y2y1y_mintr1000_top80.json \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.25 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v5_seedv4_champ_refine_15m_10y_milestones.json
```

Outcome (2026-01-31):
- `backtests/slv/slv_exec5m_v5_seedv4_champ_refine_15m_10y_milestones.json` contains **639 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v5.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_exec5m_v5_seedv4_champ_refine_15m_10y_milestones.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v5_seedv4_champ_refine_15m_10y2y1y_mintr1000_top80.json
```

Outcome (2026-01-31):
- **10** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.39**, tying v4 (no dethrone).

### v6 — Supertrend neighborhood widening (Candidate #2 tight sweep; dethroned v4)
Status: **DONE** (promoted above as CURRENT v6; worst-window `roi/dd=1.70`)

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
  --seed-milestones backtests/slv/slv_exec5m_v4_shockrisk_champ_refine_15m_10y2y1y_mintr1000_top80.json \
  --seed-top 6 \
  --min-trades 0 --top 25 \
  --milestone-min-win 0.00 --milestone-min-trades 1000 --milestone-min-pnl-dd 1.25 \
  --write-milestones --milestones-out backtests/slv/slv_exec5m_v6_st_neighborhood_champ_refine_15m_10y_milestones.json
```

Outcome (2026-01-31):
- `backtests/slv/slv_exec5m_v6_st_neighborhood_champ_refine_15m_10y_milestones.json` contains **721 eligible presets**
  (eligible under the 10y-window milestone gates used above).

#### v6.1 — 10y/2y/1y kingmaker eval (stability-first)
```bash
python -u -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/slv/slv_exec5m_v6_st_neighborhood_champ_refine_15m_10y_milestones.json \
  --symbol SLV --bar-size "15 mins" --use-rth --offline --cache-dir db --top 2000 \
  --require-positive-pnl --min-trades 1000 \
  --window 2016-01-08:2026-01-08 --window 2024-01-08:2026-01-08 --window 2025-01-08:2026-01-08 \
  --write-top 80 --out backtests/slv/slv_exec5m_v6_st_neighborhood_champ_refine_15m_10y2y1y_mintr1000_top80.json
```

Outcome (2026-01-31):
- **36** candidates passed strict positivity (10y+2y+1y) with `min_trades=1000`.
- Best worst-window `roi/dd` is **1.70**, promoted above as CURRENT v6.
- Defining delta vs v4: `ST(5,0.75,hl2)` and `flip_exit_min_hold_bars=4` (the rest of the stack stays similar).

## Notes

- We do **not** assume `spot_close_eod=true` is necessary; we will explicitly sweep / compare it.
- For any strategy that looks “too good”, validation must include:
  - `--realism2` costs
  - `spot_drawdown_mode=intrabar` + liquidation marking
  - (optional) stress slices (crash windows) once we have finalists
