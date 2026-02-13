# SLV HF Research (High-Frequency Spot Track)

This file is the dedicated SLV high-frequency evolution track, separate from `backtests/slv/README.md`.
- `README.md` remains the low-frequency/live-parity representative champion track.
- This file tracks the high-turnover RATS line and its own HF crowns.

## Current Champions (stack)

### CURRENT (v8) — 1Y+2Y Dethrone Crown (1m/1m, high-trade)
Promoted from the strict eq+ trade-floor round-3b corridor (`1y trades >= prior king trades`).

**v8 kingmaker #01 [HF-1Y/2Y]**
- Preset file: `backtests/slv/slv_hf_champions_v8.json`
- Source eval: `backtests/slv/slv_hf_v7_round3b_tradefloor752_20260213.json`
- Variant id: `overlay_probe_neutral`
- Timeframe: `signal=1 min`, `exec=1 min`, `full24/5`
- Core deltas vs prior HF crown (`v7 #01`): `ratsv_branch_a_slope_med_min_pct=0.000023` (from `0.00002`), `ratsv_branch_a_slope_vel_min_pct=0.000011` (from `0.00001`), `ratsv_probe_cancel_slope_adverse_min_pct=0.00029` (from `0.00030`), with trades preserved.
- 1y (`2025-01-08 -> 2026-01-08`): trades **752**, pnl **54,350.24**, dd **14,633.78**, pnl/dd **3.7140**
- 2y (`2024-01-08 -> 2026-01-08`): trades **1,285**, pnl **51,263.25**, dd **17,066.21**, pnl/dd **3.0038**
- 1y long pnl: **56,261.77**
- 1y short pnl: **-1,911.53**

Promotion contract check:
- `1y trades >= prior king trades (752)`: **PASS** (`752`)
- `beat prior crown on 1y pnl + pnl/dd`: **PASS**
- `beat prior crown on 2y pnl + pnl/dd`: **PASS**

Dethrone delta vs prior HF crown (`v7 #01`):
- 1y: `trades +0`, `pnl +296.07`, `dd +28.94`, `pnl/dd +0.0129`
- 2y: `trades +0`, `pnl +285.30`, `dd -0.10`, `pnl/dd +0.0167`

### CURRENT 6M Crown (v8 reference) — quality anchor
Kept separately as 6M quality anchor (this is not the 1Y/2Y HF dethrone crown).

**v8 kingmaker #02 [HF-6M]**
- Preset file: `backtests/slv/slv_hf_champions_v8.json`
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

### v7 — `r2_slope_early_rank024` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/slv_hf_champions_v7.json`
- Source eval: `backtests/slv/slv_hf_v6_timing_true_attack_round2_20260213.json`
- 1y: trades **752**, pnl **54,054.17**, dd **14,604.84**, pnl/dd **3.7011**
- 2y: trades **1,285**, pnl **50,977.95**, dd **17,066.31**, pnl/dd **2.9871**

### v6 — `off_combo_04` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/slv_hf_champions_v6.json`
- Source eval: `backtests/slv/slv_hf_v5_permissionoff_slope_sniper_20260213.json`
- 1y: trades **751**, pnl **53,341.13**, dd **14,538.14**, pnl/dd **3.6690**
- 2y: trades **1,284**, pnl **50,301.82**, dd **17,066.31**, pnl/dd **2.9474**

### v5 — `on_t03_a_rank_025` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/slv_hf_champions_v5.json`
- Source eval: `backtests/slv/slv_hf_perm_slope_hunt_v3_targeted_20260213.json`
- 1y: trades **751**, pnl **52,329.44**, dd **17,346.07**, pnl/dd **3.0168**
- 2y: trades **1,284**, pnl **47,836.49**, dd **16,956.92**, pnl/dd **2.8211**


### v4 — `idealplan_v2_fast_top1` 1Y/2Y dethrone crown
- Preset file: `backtests/slv/slv_hf_champions_v4.json`
- Source eval: `backtests/slv/slv_hf_rats_idealplan_v2_fast_20260213.json`
- 1y: trades **751**, pnl **51,284.79**, dd **17,226.97**, pnl/dd **2.9770**
- 2y: trades **1,283**, pnl **45,578.79**, dd **17,197.81**, pnl/dd **2.6503**

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

### v8.0 — strict eq+ trade-floor promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_v7_round3b_tradefloor752.py
```

Promotion basis:
- Follow-up to timing-true round3 where best candidates improved pnl/pnl-dd but dropped 1Y trades by 1.
- Enforced strict crown contract against current king:
  - `1y trades >= 752` (equal-or-higher vs prior king),
  - beat prior king on 1Y and 2Y for pnl and pnl/dd.

Artifacts:
- `backtests/slv/slv_hf_v7_timing_true_round3_20260213.json`
- `backtests/slv/slv_hf_v7_round3b_tradefloor752_20260213.json`
- `backtests/slv/slv_hf_champions_v8.json`

Outcome:
- New king: `overlay_probe_neutral`
- 1y: trades **752**, pnl **54,350.24**, dd **14,633.78**, pnl/dd **3.7140**
- 2y: trades **1,285**, pnl **51,263.25**, dd **17,066.21**, pnl/dd **3.0038**
- Notes: timing drift is material (`changed_any`: 1Y=388, 2Y=389); short pnl is slightly worse, but this remains a preference axis, not a hard gate.

### v7.0 — timing-true round2 dethrone promotion (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_v6_timing_true_attack_round2.py
```

Promotion basis:
- Took `r2_slope_early_rank024` from the round-2 stage-3 production shortlist.
- Applied updated crown contract:
  - short-delta uplift is a preference (not hard gate),
  - hard gates are `1y trades >= 733` + beat prior king on 1Y and 2Y for pnl and pnl/dd.

Artifacts:
- `backtests/slv/slv_hf_v6_timing_true_attack_round2_20260213.json`
- `backtests/slv/slv_hf_champions_v7.json`

Outcome:
- New king: `r2_slope_early_rank024`
- 1y: trades **752**, pnl **54,054.17**, dd **14,604.84**, pnl/dd **3.7011**
- 2y: trades **1,285**, pnl **50,977.95**, dd **17,066.31**, pnl/dd **2.9871**
- Contract check: **PASS**

### v6.0 — permission-off slope sniper pass (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_v5_permissionoff_slope_sniper.py
```

Search grid (tight, non-broad):
- branch sizing held constant for this pass lineage (timing-focused corridor)
- timing-active knobs only around v5 core: `ratsv_branch_a_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`
- `entry_permission_mode=off` in search path
- stage flow: 1Y sniper then 2Y verify on shortlist
- trade floor: `1y trades >= 733`
- workers: dynamic (`cpu-2`, capped)

Artifacts:
- `backtests/slv/slv_hf_v5_permissionoff_slope_sniper_20260213.json`
- `backtests/slv/slv_hf_v5_forensics_combo04_20260213.json`
- `backtests/slv/slv_hf_champions_v6.json`

Outcome:
- Stage counts: `1y=36`, `2y_verify=10`
- Dethrone hits after 2Y verify: **8**
- Promoted `off_combo_04` as new HF 1Y/2Y crown.
- Notes: gain is primarily from improved long extraction and lower 1Y DD; short side remains net negative and remains the timing frontier.

### v6.1 — timing-true attack pass round1 (INVESTIGATION)
Status: **DONE (not promoted)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_v6_timing_true_attack.py
```

Search contract:
- Freeze branch sizing in search (`spot_branch_a_size_mult == spot_branch_b_size_mult`) to remove sizing confound.
- Sweep timing-active knobs only: `ratsv_branch_a_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`.
- Hard acceptance gates:
  - material timestamp drift (entry/exit changes),
  - material short-pnl improvement,
  - `1y trades >= 733`,
  - beat king on 1Y+2Y pnl and pnl/dd after reapplying production sizing.

Artifacts:
- `backtests/slv/slv_hf_v6_timing_true_attack_20260213.json`

Outcome:
- Stage counts: `stage1=30`, `stage2_frozen=14`, `stage3_prod=6`
- `accepted_timing_true_frozen=0`, `accepted_timing_true_prod=0`, `accepted_full_contract=0`
- Best production row (`smed_2e5`) improved behavior metrics but did not satisfy dethrone+timing together.

### v6.2 — timing-true attack pass round2 (INVESTIGATION)
Status: **DONE (not promoted)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_v6_timing_true_attack_round2.py
```

Search corridor:
- Narrowed around `combo_slope_early` family to recover 2Y continuity while retaining timing drift.
- Kept timing-only knobs and frozen-size search design.

Artifacts:
- `backtests/slv/slv_hf_v6_timing_true_attack_round2_20260213.json`

Outcome:
- Stage counts: `stage1=20`, `stage2=12`, `stage3=6`
- Under the original strict short-improvement gate, `accepted_full_contract=0`.
- Dethrone hit `r2_slope_early_rank024` was later promoted in `v7.0` when crown contract was updated to prioritize 1Y/2Y pnl+pnl/dd with trade floor.

### v5.0 — targeted permission+slope micro-pass (PROMOTED)
Status: **DONE (PROMOTED)**

Command used:
```bash
PYTHONPATH=/Users/x/Desktop/py/tradebot python /tmp/hf_rats_perm_slope_hunt_v3_targeted.py
```

Search grid (tight, non-broad):
- Focused around v4 crown timing knobs (`ratsv_branch_a_*`, `ratsv_branch_b_*`, `ratsv_probe_cancel_*`, `ratsv_adverse_release_*`)
- permission A/B checks only on top movers
- trade floor: `1y trades >= 733`
- workers: `6`

Artifacts:
- `backtests/slv/slv_hf_perm_slope_hunt_v2_tight_20260213.json`
- `backtests/slv/slv_hf_perm_slope_hunt_v3_targeted_20260213.json`
- `backtests/slv/slv_hf_champions_v5.json`

Outcome:
- `candidate_count_on=20`, `off_checks=4`
- Promoted `on_t03_a_rank_025` as new HF 1Y/2Y crown.
- Permission A/B for this winner replayed equivalently in-window; key dethrone lever was branch-A rank sensitivity.

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
