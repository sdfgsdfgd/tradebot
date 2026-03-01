# MNQ HF Research (Micro Nasdaq Futures Track)

This file is the dedicated MNQ high-frequency evolution track (micro Nasdaq futures).

Promotion contract (current):
- Promote based on `1Y` first, then reproduce on `2Y`.
- `10Y` is a later reality check (deferred for now).

MNQ reality constraints (must be respected before trusting any “champion”):
- Futures tick size + per-contract fee model must be correct (MNQ is not “TQQQ with a different name”).
- Futures session semantics (nearly 24x5) are not identical to equity RTH/pre/post.
- Continuous contract / rollover handling must be explicit.

## Current Champions (stack)

### CURRENT (v1-port_TQQQ-HF-v31_to_MNQ-FUT-unscored) — bootstrap port of the current TQQQ-HF crown to MNQ

- Preset file (UI loads this): `backtests/mnq/archive/champion_history_20260301/mnq_hf_champions_v1_port_from_tqqq_hf_v31_20260301.json`
- Status: config port is wired, but MNQ backtests + live realism verification are still pending.

## MNQ Port TODOs (ideal integration)

Phase 0 — Connectivity sanity (hard requirement)
- [ ] Ensure IBKR TWS/IBG API is running and reachable (`IBKR_HOST`/`IBKR_PORT`). The backtest hydrator cannot fetch MNQ history otherwise.
- [ ] Confirm futures market data permissions for MNQ/NQ (IBKR will return empty/partial history without proper permissions).

Phase 1 — Data cache (MNQ bars)
- [ ] Build MNQ `1 min` RTH cache for promotion windows (`1Y` + `2Y`) so we can keep `exec=1 min`.
- [ ] Resample from `1 min` -> `5 mins` RTH so `signal=5 mins` is deterministic and fast.
- [ ] Decide whether to also build `full24` caches (needed only when we intentionally enable overnight trading).

Suggested cache ops (when IBKR is reachable):
```bash
python -m tradebot.backtest.tools.cache_ops sync \
  --request "MNQ|2025-01-01|2026-01-19|1 min|rth|mnq_hf_bootstrap_1y" \
  --cache-dir db --aggressive --max-primary-span-days 7

python -m tradebot.backtest.tools.cache_ops sync \
  --request "MNQ|2024-01-01|2026-01-19|1 min|rth|mnq_hf_bootstrap_2y" \
  --cache-dir db --aggressive --max-primary-span-days 7

python -m tradebot.backtest.tools.cache_ops resample \
  --symbol MNQ --start 2024-01-01 --end 2026-01-19 \
  --src-bar-size "1 min" --dst-bar-size "5 mins" \
  --cache-dir db --use-rth
```

Phase 2 — Futures realism (stop lying to ourselves)
- [ ] Futures tick size: enforce MNQ tick rounding (0.25 points) on simulated fills + stop/flip levels.
- [ ] Futures fees: convert per-contract commissions/fees into “price units” using MNQ multiplier ($2/point) so backtests reflect reality.
- [ ] Slippage: model at least 1-tick adverse on market entries/exits during the open.

Phase 3 — Contract correctness (live + backtest alignment)
- [ ] Decide data contract vs tradable contract:
  - Data may use continuous futures (`CONTFUT`).
  - Trading must use an actual front-month MNQ contract + roll rules.
- [ ] Implement/verify “front-month resolver” in live runtime (avoid manual babysitting on expiry week).

Phase 4 — Promotion eval (MNQ-specific)
- [ ] Run the ported config on MNQ over:
  - `1Y` (`2025-01-01 -> 2026-01-19`)
  - `2Y` (`2024-01-01 -> 2026-01-19`)
- [ ] If it survives, promote a real MNQ v1 (scored) and start MNQ dethrone evolutions.
- [ ] If it fails, we fork the knobs minimally (fees/ticks/session first, signal second).

## Evolutions (stack)
