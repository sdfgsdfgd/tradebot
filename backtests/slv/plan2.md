# SLV Backlog (Postponed) — Candidate #2 Notes

This file captures postponed ideas so the main SLV quest (`backtests/slv/plan.md`) stays focused on Candidate #1.

## Candidate #2 (postponed): Regime‑Filtered Pullback Mean‑Reversion (spot, inverted mapping)

Concept (high level):
- Use a slower regime (e.g., 4h Supertrend) as the *directional safety frame*.
- Inside that regime, trade **pullbacks / bounces** mean‑reversion style on a faster timeframe.
- The goal is to preserve very high activity while increasing win-rate and smoothing equity curves.

Why it may fit SLV:
- SLV often exhibits mean‑reverting microstructure vs levered equity ETFs.
- High-frequency mean reversion can produce a “grindy” equity curve when paired with a regime filter.

Key knobs that would matter (when we revisit):
- Signal cadence: 15m signals + 5m exec (same realism requirements as #1).
- Entry signal: likely needs a “pullback detector” (e.g., distance-from-local-extreme, ORB fades, or EMA mean deviation),
  not just EMA state/cross.
- Exit mode: small PT/SL or ATR-scaled exits; close EOD likely becomes more important for MR.
- Shock overlay: probably `block` or “risk-off” during volatility expansions to avoid trend-mode tail risk.

Implementation caveat (why postponed):
- A naive “invert directional_spot mapping” does not cleanly work with current semantics because:
  - `signal.entry_dir` (up/down) is used for regime gating and flip-exit logic,
  - while `trade_dir` is inferred from actual position sign (long/short).
  - If we invert only the action mapping, we can accidentally create immediate flip exits or mismatched gating.
- When we do this, we should add an explicit notion of “trade direction for gating” or a supported “invert signal direction” option,
  so the regime/shock/flip gates remain coherent.

## Snippet (verbatim; parked for later)

``   2. v34-seeded champ_refine to regain 10y without losing 1y [Score: 8/10]
     • Run --axis champ_refine seeded from v34 to search micro-knob combos that restore 10y while preserving the 1y/2y boost.
     • Pros: fastest path to a “v35” candidate; uses proven joint micro-grids.
     • Cons: still unlikely to dominate v33 on 10y unless we get lucky.
 ``

