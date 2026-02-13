# SLV HF Research (High-Frequency Spot Track)

This file is the dedicated SLV high-frequency evolution track, separate from `backtests/slv/README.md`.
- `README.md` remains the low-frequency/live-parity representative champion track.
- This file tracks the high-turnover RATS line and its own HF crowns.

## Current Champions (stack)

### CURRENT (v4) — 1Y+2Y Dethrone Crown (1m/1m, high-trade)
Promoted from the RATS ideal-plan v2 fast exploit with the trade floor preserved.

**v4 kingmaker #01 [HF-1Y/2Y]**
- Preset file: `backtests/slv/slv_hf_champions_v4.json`
- Source eval: `backtests/slv/slv_hf_rats_idealplan_v2_fast_20260213.json`
- Variant id: `idealplan_v2_fast_top1`
- Timeframe: `signal=1 min`, `exec=1 min`, `full24/5`
- Core deltas vs prior HF crown (`v3 #01`): `spot_branch_a_size_mult=0.45` (from `0.53`), `spot_branch_b_size_mult=1.35` (from `1.40`), `flip_exit_mode=cross` (from `entry`), `flip_exit_min_hold_bars=0` (from `2`), probe cancel retuned (`max_bars=5`, `slope=0.0003`, `tr_ratio=0.95`)
- 1y (`2025-01-08 -> 2026-01-08`): trades **751**, pnl **51,284.79**, dd **17,226.97**, pnl/dd **2.9770**
- 2y (`2024-01-08 -> 2026-01-08`): trades **1,283**, pnl **45,578.79**, dd **17,197.81**, pnl/dd **2.6503**
- 1y long pnl: **53,288.64**
- 1y short pnl: **-2,003.86**

Promotion contract check:
- `1y trades >= 733`: **PASS** (`751`)
- `beat prior crown on 1y pnl + pnl/dd`: **PASS**
- `beat prior crown on 2y pnl + pnl/dd`: **PASS**

Dethrone delta vs prior HF crown (`v3 #01`):
- 1y: `trades +5`, `pnl +7,514.78`, `dd +888.46`, `pnl/dd +0.2981`
- 2y: `trades +12`, `pnl +17,881.92`, `dd -6,424.78`, `pnl/dd +1.4778`

### CURRENT 6M Crown (v4 reference) — quality anchor
Kept separately as 6M quality anchor (this is not the 1Y/2Y HF dethrone crown).

**v4 kingmaker #02 [HF-6M]**
- Preset file: `backtests/slv/slv_hf_champions_v4.json`
- Source eval: `backtests/slv/slv_rand221_ultratight_eval_20260213.json`
- Variant id: `rand_025`
- Timeframe: `signal=10 mins`, `exec=5 mins`, `full24/5`
- 6m (`2025-07-08 -> 2026-01-08`): trades **335**, pnl **37,517.96**, dd **12,791.19**, pnl/dd **2.9331**
- 1y (`2025-01-08 -> 2026-01-08`): trades **646**, pnl **33,287.38**, dd **12,399.62**, pnl/dd **2.6845**

## Current Hardening Result (investigation)

### v2.1 — target-window long-loss hardening (trade lock)
Status: **DONE (not promoted)**

Contract:
- Hard lock: `1y trades >= 700`
- Hard lock: `1y pnl/dd >= 2.5`
- Explicit target windows:
  - `2025-12-28T20:03:00 -> 2025-12-30T13:06:00`
  - `2025-10-16T21:04:00 -> 2025-10-17T15:28:00`
  - `2026-01-05T03:17:00 -> 2026-01-07T13:04:00`

Best lock-pass reducer found:
- Variant: `v5_002`
- 1y: trades **729**, pnl **33,600.09**, dd **12,695.70**, pnl/dd **2.6466**
- 6m: trades **516**, pnl **27,926.66**, dd **12,162.20**, pnl/dd **2.2962**
- Window long-loss mass: **28,427.39** (vs `rnd_016` **36,366.18**)

Delta vs v2 crown (`rnd_016`):
- `trades -4`, `pnl -6,490.84`, `dd -2,821.67`, `pnl/dd +0.0630`
- `target-window long-loss -7,938.79`

Decision:
- Not promoted because absolute 1Y pnl was lower than the crown at the time.

## Previous Crowns (references)

### v3 — `attackpass_v2_top1` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/slv_hf_champions_v3.json`
- Source eval: `backtests/slv/slv_hf_rats_attackpass_v2_20260213.json`
- 1y: trades **746**, pnl **43,770.01**, dd **16,338.51**, pnl/dd **2.6789**
- 2y: trades **1,271**, pnl **27,696.87**, dd **23,622.59**, pnl/dd **1.1725**

### v2 — `rnd_016` 1Y PnL crown
- Preset file: `backtests/slv/slv_hf_champions_v2.json`
- Source eval: `backtests/slv/slv_rand025_1m_pnl_push_v4_eval_20260213.json`
- 1y: trades **733**, pnl **40,090.93**, dd **15,517.36**, pnl/dd **2.5836**
- 6m: trades **520**, pnl **30,851.19**, dd **14,492.85**, pnl/dd **2.1287**

### v1 — `rand_025` ultra-tight exploit dethrone (trade-locked)
- Preset file: `backtests/slv/slv_hf_champions_v1.json`
- Source eval: `backtests/slv/slv_rand221_ultratight_eval_20260213.json`
- 6m: trades **335**, pnl **37,517.96**, dd **12,791.19**, pnl/dd **2.9331**
- 1y: trades **646**, pnl **33,287.38**, dd **12,399.62**, pnl/dd **2.6845**

### v0.2 — `rand_221` stability hardening winner
- Source eval: `backtests/slv/slv_r102002_1y_boost_eval_20260212.json`
- 6m: trades **333**, pnl **35,238.38**, dd **12,714.22**, pnl/dd **2.7716**
- 1y: trades **634**, pnl **30,090.27**, dd **12,229.15**, pnl/dd **2.4605**

### v0.1 — `rand_204` reliability sibling
- Source eval: `backtests/slv/slv_r102002_1y_boost_eval_20260212.json`
- 6m: trades **339**, pnl **32,867.16**, dd **11,860.89**, pnl/dd **2.7711**
- 1y: trades **639**, pnl **28,265.14**, dd **11,446.03**, pnl/dd **2.4694**

## Evolutions (stack)

### v4.0 — RATS ideal-plan v2 fast exploit (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_rats_idealplan_v2_fast.py
```

Search grid (staged, reliability-first):
- Stage B: `ratsv_adverse_release_*` + `flip_exit_*` around the v3 crown core
- Stage C: `ratsv_probe_cancel_*` around stage-B survivors
- trade floor: `1y trades >= 733`
- workers: `8`

Artifacts:
- `backtests/slv/slv_hf_rats_idealplan_v2_fast_20260213.json`
- `backtests/slv/slv_hf_attackpass_v2top_rootcause_20260213.json`
- `backtests/slv/slv_hf_champions_v4.json`

Outcome:
- Stage counts: `b_1y=576`, `b_2y_verified=80`, `c_1y=216`, `c_2y_verified=100`
- Dethrone hits: **172**
- Promoted `idealplan_v2_fast_top1` as new HF 1Y/2Y crown.
- Notes: short-side/downturn monetization still needs targeted follow-up despite stronger aggregate continuity.

### v3.1 — RATS attackpass v2 exploit (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_rats_attackpass_v2.py
```

Search grid (tight exploit around v3.0 winner):
- `ratsv_adverse_release_tr_ratio_min`: `1.0325, 1.035, 1.0375, 1.04, 1.0425, 1.045`
- `spot_branch_a_size_mult`: `0.53, 0.54, 0.55, 0.56, 0.57`
- `ema_spread_min_pct`: `0.00075, 0.0008, 0.00085, 0.0009`
- workers: `6`

Artifacts:
- `backtests/slv/slv_hf_rats_attackpass_v2_20260213.json`
- `backtests/slv/slv_hf_attackpass_forensic_compare_v2top_20260213.json`
- `backtests/slv/slv_hf_champions_v3.json`

Outcome:
- Total candidates: **120**
- Dethrone-primary hits: **80**
- Strict 2Y trade-lock hits: **0** (known limitation in this family)
- Promoted top candidate (`attackpass_v2_top1`) as new HF 1Y/2Y crown.

### v3.0 — RATS attackpass v1 bridge (PROMOTION FEEDER)
Status: **DONE**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_rats_attackpass_v1.py
```

Search grid:
- `ratsv_adverse_release_tr_ratio_min`: `1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05`
- `spot_branch_a_size_mult`: `0.55, 0.60, 0.65`
- `ema_spread_min_pct`: `0.0008, 0.0009, 0.0010, 0.0011`
- workers: `6`

Artifacts:
- `backtests/slv/slv_hf_rats_attackpass_v1_20260213.json`
- `backtests/slv/slv_hf_attackpass_forensic_compare_20260213.json`

Outcome:
- Total candidates: **140**
- Dethrone-primary hits: **67**
- Best bridge candidate prepared the v3.1 exploit corridor.

### v2.1 — constrained window hardening around `rnd_016`
Status: **DONE (investigation)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/rnd016_window_hardening_v5.py
```

Artifacts:
- `backtests/slv/slv_rnd016_window_hardening_v5_eval_20260213.json`

Outcome:
- Found lock-pass variant `v5_002` that reduced target-window long-loss concentration.
- Not promoted due meaningful 1Y pnl give-up versus then-current 1Y crown.

### v2 — 1m PnL push dethrone (`rnd_016`)
Status: **DONE**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/rand025_1m_pnl_push_v4.py
```

Artifacts:
- `backtests/slv/slv_rand025_1m_pnl_push_v4_eval_20260213.json`
- `backtests/slv/slv_rats_1m_v4_rnd016_trade_inspection_20260213.json`
- `backtests/slv/slv_hf_champions_v2.json`

Outcome:
- Promoted `rnd_016` as HF 1Y crown (high-trade 1m/1m line).
- Preserved `rand_025` as separate HF 6M crown reference.

### v1 — rand_221 ultra-tight exploit (hour-leak targeting + hard trade lock)
Status: **DONE**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/rand221_ultratight_exploit.py
```

Artifacts:
- `backtests/slv/slv_rand221_ultratight_eval_20260213.json`
- `backtests/slv/slv_hf_champions_v1.json`

Outcome:
- Promoted `rand_025` as v1 HF crown.
- Found strict siblings: `rand_039`, `rand_190`, `rand_041`, `probe_short_003`.
