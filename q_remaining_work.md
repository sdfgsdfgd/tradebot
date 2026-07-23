# Remaining Work

Work is intentionally paused at a verified milestone. No further implementation should begin unless this queue is deliberately resumed.

## Highest-value remaining work

1. **Finish `spot-v1.exit-resize-adapter-receipts` only if parity work resumes.**
   - Proven production evidence: backtest terminal exit receipt and live resize fill receipt.
   - Highest-value missing direct route evidence: backtest resize receipt and live filled-exit terminal receipt.
   - Recheck the live resize seed with a direct staged-journal assertion; instrumentation observed the production seed path, while the conservative quadrant classifier did not count the nested journal return.
   - Preserve production-route testing: tests must not call `project_spot_trace_receipt` directly.

2. **Complete native atomic multi-leg option execution.**
   - Broker-qualified BAG identity, ratios/actions/exchanges, debit-credit economics, what-if margin, atomic submit/modify/cancel/reconcile, and partial/failure handling.
   - Treat the approximately USD 2,200 account as a first-class XSP defined-risk target.
   - Keep XSP, full-size SPX, and MCL qualification, settlement, multiplier, liquidity, and risk semantics separate.

3. **Unify the canonical research/backtest/live strategy-execution model.**
   - Shared intent, combo economics, costs, queue/latency/partial/reject/cancel assumptions, margin, assignment/exercise, rolls, and rescue.
   - Broker and simulator mutation remain explicit intentional adapters.

4. **Build the resumable weekly walk-forward research and promotion loop.**
   - Regime/timeframe slices, robustness, turnover/capacity, tail loss, drawdown, ruin and margin stress, leakage/overfit checks, and execution sensitivity.
   - Promotion ladder: candidate → deterministic replay → provider integration → paper/canary → bounded live.

5. **Continue capability-ledger discipline.**
   - Every capability needs a canonical owner, intentional adapter boundary, executable receipt, honest denominator, and promotion gate.

## Verified pause checkpoint

- Wave 5 code commit: `f5f1353d0e9f8fde2012c7b2faf95882d7d0593f`.
- Wave 5 metadata commit: `35271e4de5a83e67eeed602f71f27f1348661678`.
- Verified publication receipt: `8860e264e1d0906dc728514a5661720d652fbfbc91542e3f2309a19b77c02201`.
- Exit/resize quadrant audit receipt: `15cef567fac27e25a3eac6ebbb7ffb5d48c3f00f32fb29cfbf4296446c5fd88f`.
- Archive base before the final documentation commit: `35271e4de5a83e67eeed602f71f27f1348661678`.
- All prior detailed evidence is preserved in `.arcana/memory.md` beneath `# ARCHIVE`.

## Resume rules

- Start by synchronizing with remote `main` and confirming a clean worktree.
- Revalidate the current ledger denominator before changing code.
- Prefer one production-route RED at a time and preserve already-proven behavior.
- Do not contact IBKR Gateway or any provider unless the resumed task explicitly requires provider evidence.
