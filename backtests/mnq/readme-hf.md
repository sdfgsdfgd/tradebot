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

## Evolutions (stack)
