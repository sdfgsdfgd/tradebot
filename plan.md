# Multi-window Hunt + Realism Pass (Spot / TQQQ)

Status: **active** (started 2026-01-20)

This file is a temporary, persisted task log + plan for making our spot backtests more realistic and then doing a multi-window stability hunt for a **stable** strategy (TQQQ first) across:
- Full 10y window (currently: `2016-01-01 → 2026-01-08`, RTH)
- Recent 1y slices (currently tested: `2023→2024`, `2024→2025`, `2025→2026-01-19` RTH)

The user goal is to reach a strategy we can *actually* live trade soon (TQQQ live first), prioritizing **edge + consistency** over one-window “hero” stats.

---

## Decisions (locked for this round)

We agreed to implement “Realism v1 + v2” (spot-focused) with these knobs:

- **A) TQQQ direction**: **Long-only** (more reliable/stable for an equity ETF; avoids short borrow realism and asymmetric tail risk).
- **B) Entry timing**: **next bar open** (no same-bar-close fills).
- **C) Intrabar PT/SL**: **enabled**, and when both PT & SL are hit in the same bar, take **worst-case** (stop-first).
- **D) Costs**: model **spread = $0.01/share each side** (implemented as half-spread on entry/exit), plus a simple commission/slippage model.
  - Realism v1 defaults: `spot_commission_per_share=0.0`.
  - Realism v2 defaults: `spot_commission_per_share=0.005` with `spot_commission_min=1.0` (=$1.00/order), `spot_slippage_per_share=0.0`.
- **E) ROI-based sizing (Realism v2)**: turn per-share PnL into ROI-comparable results with a live-shaped sizing model.
  - Default: `spot_sizing_mode=risk_pct`, `spot_risk_pct=0.01` (1% equity risk-to-stop), `spot_max_notional_pct=0.50`, `spot_min_qty=1`.

---

## Why this is needed (current issues)

### “Unlimited stacking” artifacts
Some earlier “winners” used:
- `max_open_trades=0` (unlimited concurrent trades)
- `max_entries_per_day=0` (unlimited entries per day)

This can be OK for research, but it’s not live-shaped unless we intentionally want pyramiding with explicit controls.

### Execution optimism (before Realism v1)
Spot backtests previously did:
- Entry at `bar.close` (lookahead-ish vs real fillability).
- Profit/stop checks using only `bar.close` (ignores intrabar hits).
- Mark open positions at `bar.close` (mid-ish), not liquidation (bid/ask).
- Drawdown was close-to-close (misses intrabar pain).

---

## Plan (high-level)

### 1) Realism v1 + v2 implementation (spot only)
- Add config knobs for:
  - entry fill timing (close vs next-open)
  - intrabar PT/SL using OHLC (+ stop gap-through handling)
  - spread + commission (+ per-order minimum) + slippage model
  - liquidation marking for equity/DD
  - intrabar drawdown approximation (worst within bar, respecting stops)
  - position sizing + ROI/DD% reporting

### 2) Multi-window selection process (TQQQ)
- Run combo/focused sweeps under Realism v1.
- Shortlist candidates by performance on the primary window.
- Re-evaluate shortlisted candidates across multiple windows (10y + recent 1y slices).
- Rank by a **stability score** (e.g., worst-window pnl/dd, or similar robust metric).

### 3) Persist + operationalize
- Persist top stable candidates as presets in `tradebot/backtest/spot_milestones.json`.
- Update `tradebot/backtest/README.md` with a clearly labeled “Realism v1” section.
- Keep older optimistic/legacy results clearly labeled as such.

---

## Baseline references (pre-Realism v1)

TQQQ 10y “winners” previously recorded (pre-Realism v1):
- `Spot (TQQQ) 10y (RTH) #T1..#T4` in `tradebot/backtest/spot_milestones.json`
- Note: several of those used `max_open=0` (unlimited stacking) and had very small per-trade edge.

---

## Next updates

As the plan evolves, append a short dated log below.

### Log
- 2026-01-20: Create this plan; start Realism v1 implementation and stability harness.
- 2026-01-20: Implement spot realism knobs (next-open fills, intrabar exits, spread/commission, liquidation marking, intrabar DD) and wire `tradebot/backtest/evolve_spot.py --realism`.
- 2026-01-20: Implement Realism v2 (ROI-based sizing + commission minimums + slippage knob + stop gap-through handling) and wire `tradebot/backtest/evolve_spot.py --realism2`.
