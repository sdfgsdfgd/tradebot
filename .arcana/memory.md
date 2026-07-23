# Remaining Work

TradeBot work is intentionally paused at a verified milestone. The actionable handoff is maintained in root-level `q_remaining_work.md`.

## Highest-value queue

1. Finish `spot-v1.exit-resize-adapter-receipts` only if parity work resumes: direct backtest-resize and live-filled-exit receipt evidence remain the clearest gaps.
2. Complete broker-native atomic multi-leg option execution, prioritizing empirically admitted defined-risk XSP structures for the approximately USD 2,200 account.
3. Continue the canonical research/backtest/live strategy-execution model, weekly walk-forward loop, promotion/calibration ladder, and capability-ledger discipline.

## Stop checkpoint

- Published code and metadata through `35271e4de5a83e67eeed602f71f27f1348661678`.
- Publication receipt `8860e264e1d0906dc728514a5661720d652fbfbc91542e3f2309a19b77c02201`.
- Detailed resume instructions: `q_remaining_work.md`.

# ARCHIVE

<!-- BEGIN: TRADEBOT_CAPABILITY_LEDGER_CHARTER -->

<!-- anchor: @charter.primary-goals -->
## TradeBot Capability Ledger — Long-Term Charter

### Selected architecture

Use the **Native Minimal Pyramid**: a TradeBot-native, executable capability registry inspired by Arcana without reorganising the existing flat test suite. The ledger must evolve through a **MECC taxonomy**: subsystem umbrellas are mutually exclusive as primary ownership categories and collectively comprehensive across the product.

### Primary goals

1. **Backtesting system — comprehensive capability lock-in and evolution**
   - Cover data acquisition, cache identity and integrity, gap detection and repair, resampling, calendars, timezone/session semantics, signal generation, policy decisions, sizing, execution simulation, portfolio accounting, calibration, optimization, scenario analysis, reproducibility, performance, and result truthfulness.
   - Treat caching and optimisation as correctness-sensitive capabilities, not merely speed improvements.

2. **Live bot and UI system — comprehensive capability lock-in and evolution**
   - Cover IBKR connectivity, account/portfolio truth, market-data quality and degradation, contract discovery, signals, policy gates, sizing, order construction, submission, reconciliation, repricing, disconnect recovery, state continuity, UI truth, diagnostics, journaling, and operator safety.
   - Preserve the exact runtime path from the Textual UI and bot orchestration through `IBKRClient` to IB Gateway.

3. **Market realism, accuracy, and reliability — primary product-quality objective**
   - Continuously reduce differences between simulated and live behaviour under current market structure and broker semantics.
   - Explicitly model sessions, holidays, overnight trading, stale/delayed/frozen data, liquidity, spreads, tick sizes, slippage, commissions, partial fills, rejected or modified orders, queue and latency effects, contract rolls, corporate actions where applicable, account constraints, and regime changes.
   - Require backtest/live parity receipts wherever decision logic is intended to be shared.

<!-- anchor: @taxonomy.mecc -->
### MECC taxonomy rules

- Every capability contract has exactly one primary subsystem umbrella.
- Umbrellas must not overlap semantically; cross-cutting relationships belong in evidence, owner surfaces, oracles, dependencies, and documentation rather than duplicate subsystem ownership.
- The taxonomy must collectively cover all production source areas and all tests, including shared foundations used by both backtest and live paths.
- Existing proven behaviour is locked in before aspirational expansion.
- Taxonomy changes require an executable migration/check so contracts cannot silently become orphaned or ambiguously classified.

<!-- anchor: @future.capability-births -->
### Aspirational `.future.` capability births

- Reserve `.future.` contract IDs for valuable capabilities that do not yet have complete implementation or proof.
- A future item is not a vague wishlist: it must identify a primary subsystem, owner surface, falsifiable oracle, intended evidence layer, current status, and next proof.
- Promote `.future.` items through `planned` or `gap` → `wip` → `covered` as TDD evidence is born.
- Prioritise ideas that improve realism, reliability, accuracy, performance, observability, explainability, safety, backtest/live parity, and adaptation to current-market behaviour.

<!-- anchor: @proof.ib-gateway -->
### Proven IB Gateway capability topology

- Direct network transport: q `192.168.1.4` reaches MacBook `192.168.1.2:4001`.
- Direct IB protocol session: blocked because Gateway trust remains localhost-only (`TrustedIPs=127.0.0.1`).
- Least-privilege route: disposable q local port → SSH local forwarding → Mac `127.0.0.1:4001`.
- Read-only IB handshake through tunnel: passed; server version `176`; one managed account visible.
- Unmodified TradeBot multi-client stack through tunnel: main `880`, proxy `881`, index `882`; connection state `connected`.
- Safety invariant: do not widen Gateway trust when SSH forwarding preserves the localhost-only boundary; live canaries remain explicit, non-ordering, read-only-first, and self-cleaning.

<!-- END: TRADEBOT_CAPABILITY_LEDGER_CHARTER -->

<!-- BEGIN: TRADEBOT_MECC_TAXONOMY_V1 -->

<!-- anchor: @taxonomy.mecc.v1 -->
## MECC Subsystem Taxonomy v1

Every capability has exactly one primary subsystem. Families are navigational only; subsystem ownership is authoritative.

### `runtime-configuration-state` — Runtime Configuration, Presets, And Durable State
- Family: `shared`
- Owns: Own startup configuration, environment parsing, knobs, presets, persistent runtime identity, and reproducible launch state.
- Boundary: Does not own trading decisions, broker sessions, market observations, or UI presentation.

### `market-time-series-semantics` — Market Time, Sessions, Bars, And Series Semantics
- Family: `shared`
- Owns: Own canonical timestamps, calendars, sessions, bar completeness, resampling meaning, and shared series identity.
- Boundary: Does not own cache storage, broker subscriptions, signals, or simulated fills.

### `signal-regime-intelligence` — Signals, Regimes, Shocks, And Market-State Intelligence
- Family: `shared`
- Owns: Own market observation, feature extraction, signal snapshots, regime routing, shock detection, and context requirements.
- Boundary: Produces facts and classifications; it does not choose position intent, size, or order action.

### `policy-risk-sizing` — Policy, Entry/Exit Gates, Risk, And Position Sizing
- Family: `shared`
- Owns: Own action decisions derived from market facts: hold, enter, exit, resize, risk overlays, limits, and quantities.
- Boundary: Does not ingest broker data, simulate fills, submit orders, or render operator UI.

### `market-realism-parity` — Current-Market Realism And Backtest/Live Parity
- Family: `shared`
- Owns: Own executable receipts proving simulated and live-intended semantics agree under realistic market and broker conditions.
- Boundary: Validates cross-system truth; it does not become the primary owner of the underlying production implementation.

### `broker-connectivity-account` — Broker Connectivity, Sessions, Accounts, Portfolio, And PnL Truth
- Family: `live`
- Owns: Own IB Gateway transport/session lifecycle, client IDs, reconnects, managed accounts, portfolio, account values, and broker PnL truth.
- Boundary: Does not own instrument discovery, quote-quality policy, order execution, or UI formatting.

### `live-market-data-contracts` — Live Market Data, Instrument Identity, And Contract Discovery
- Family: `live`
- Owns: Own contract qualification/search, exchange identity, subscriptions, quote provenance, freshness, delayed/frozen modes, and tick increments.
- Boundary: Does not own trading decisions, order state machines, account totals, or presentation-only rendering.

### `live-execution-orders` — Live Order Construction, Submission, Reconciliation, And Recovery
- Family: `live`
- Owns: Own executable orders, pricing, submission, modification, cancellation, reconciliation, fill progression, repricing, and recovery.
- Boundary: Does not own policy intent, raw market-data ingestion, account truth, or display-only UI.

### `operator-ui-observability` — Operator UI, Diagnostics, Journaling, And Explainability
- Family: `live`
- Owns: Own Textual presentation, controls, status surfaces, journal output, notices, layout, and operator-visible causal truth.
- Boundary: Does not own underlying broker, signal, policy, or order semantics merely because it displays them.

### `backtest-data-cache` — Backtest Data Acquisition, Cache Integrity, Repair, And Resampling
- Family: `backtest`
- Owns: Own historical acquisition, cache keys and provenance, coverage, gap repair, stitching, persistence, and efficient reuse.
- Boundary: Does not own shared calendar meaning, strategy decisions, simulated fills, or optimization search.

### `backtest-simulation-accounting` — Backtest Execution Simulation, Portfolio Accounting, And Results
- Family: `backtest`
- Owns: Own simulated order/fill progression, trade lifecycle, equity/PnL accounting, summaries, and deterministic replay results.
- Boundary: Does not own source-data caching, signal/policy definitions, or parameter-search methodology.

### `research-optimization-calibration` — Research, Calibration, Scenario Analysis, And Optimization
- Family: `research`
- Owns: Own sweeps, fingerprints, parameter search, calibration, scenario comparison, predictive mining, and champion selection.
- Boundary: Does not redefine simulator truth, production policy semantics, or cache correctness.

### `verification-capability-evolution` — Capability Ledgers, Test Ownership, Canaries, Benchmarks, And Evolution Gates
- Family: `verification`
- Owns: Own the MECC registry, structural validation, evidence ownership, explicit live canaries, benchmark results, and future-capability promotion.
- Boundary: Proves product capabilities but does not absorb their primary subsystem ownership.

<!-- anchor: @ownership.tests.v1 -->
## Existing Test Primary Ownership v1

Validated test files: **42**. Every existing `tests/test_*.py` file has one primary subsystem owner.

### `runtime-configuration-state` — 1 test files
- `tests/test_effective_filters_payload.py`

### `market-time-series-semantics` — 6 test files
- `tests/test_backtest_time_alignment.py`
- `tests/test_backtest_trade_date_semantics.py`
- `tests/test_backtest_trading_calendar.py`
- `tests/test_bar_utils.py`
- `tests/test_engine_time_semantics.py`
- `tests/test_time_utils_modes.py`

### `signal-regime-intelligence` — 3 test files
- `tests/test_daily_climate_router.py`
- `tests/test_spot_context_requirements.py`
- `tests/test_spot_signal_time_mode.py`

### `policy-risk-sizing` — 2 test files
- `tests/test_spot_policy_graph.py`
- `tests/test_spot_policy_kernel.py`

### `market-realism-parity` — 3 test files
- `tests/test_spot_fast_summary_parity.py`
- `tests/test_ui_bot_live_unreal_consistency.py`
- `tests/test_ui_signal_gap_calendar.py`

### `broker-connectivity-account` — 3 test files
- `tests/test_client_market_data_fallbacks.py`
- `tests/test_ui_netliq_drift_estimation.py`
- `tests/test_ui_pnl_truth.py`

### `live-market-data-contracts` — 7 test files
- `tests/test_option_search_qualification.py`
- `tests/test_order_contract_normalize.py`
- `tests/test_ui_contract_binding.py`
- `tests/test_ui_favorites_md_indicator.py`
- `tests/test_ui_px_change_metrics.py`
- `tests/test_ui_search_contract_description.py`
- `tests/test_ui_signal_timeout_fallback_ladder.py`

### `live-execution-orders` — 3 test files
- `tests/test_live_signal_exit_continuum.py`
- `tests/test_order_pricing_guards.py`
- `tests/test_ui_futures_detail_flow.py`

### `operator-ui-observability` — 6 test files
- `tests/test_readme_retrievers.py`
- `tests/test_ui_bot_stream_refresh.py`
- `tests/test_ui_position_header_title.py`
- `tests/test_ui_preset_crown_labels.py`
- `tests/test_ui_presets_layout_toggle.py`
- `tests/test_ui_quote_status_line.py`

### `backtest-data-cache` — 4 test files
- `tests/test_backtest_data_timezone.py`
- `tests/test_bar_series_contract.py`
- `tests/test_cache_fetch_mend.py`
- `tests/test_offline_cache_preflight.py`

### `backtest-simulation-accounting` — 2 test files
- `tests/test_regime_router_exec_bar_state_persistence.py`
- `tests/test_spot_single_position_mode.py`

### `research-optimization-calibration` — 2 test files
- `tests/test_spot_combo_full_signature.py`
- `tests/test_spot_scenario_runner.py`

### `verification-capability-evolution` — 0 test files
- No existing test file; initial ledger/verification implementation will create evidence.

<!-- anchor: @contracts.seed.v1 -->
## Seed Capability Contract Matrix v1

| Contract ID | Primary subsystem | Layer | Initial status | Capability umbrella |
| --- | --- | --- | --- | --- |
| `unit.runtime.configuration-state` | `runtime-configuration-state` | `unit` | `covered` | Configuration, preset payloads, and runtime state remain normalized and loadable. |
| `unit.market.time-series-semantics` | `market-time-series-semantics` | `unit` | `covered` | Calendar, timestamp, bar-alignment, and series semantics remain canonical. |
| `unit.signal.regime-intelligence` | `signal-regime-intelligence` | `unit` | `covered` | Signals, shocks, regime routing, and required market context remain deterministic. |
| `unit.policy.risk-sizing` | `policy-risk-sizing` | `unit` | `covered` | Policy graphs, entry/exit gates, risk overlays, and sizing traces remain correct. |
| `integration.replay.market-realism-parity` | `market-realism-parity` | `integration-replay` | `covered` | Shared backtest/live-intended semantics remain aligned for current market conditions. |
| `unit.broker.connectivity-account` | `broker-connectivity-account` | `unit` | `covered` | Client-ID resilience, connection state, portfolio, account, and PnL truth remain correct. |
| `unit.live.market-data-contracts` | `live-market-data-contracts` | `unit` | `covered` | Contract identity, qualification, quote provenance, and fallback behaviour remain correct. |
| `unit.live.execution-orders` | `live-execution-orders` | `unit` | `covered` | Order construction, pricing, reconciliation, chasing, and recovery remain safe and deterministic. |
| `unit.operator.ui-observability` | `operator-ui-observability` | `unit` | `covered` | The UI and diagnostics present actionable broker and bot truth without misleading operators. |
| `unit.backtest.data-cache` | `backtest-data-cache` | `unit` | `covered` | Historical cache coverage, integrity, repair, stitching, and resampling remain correct. |
| `integration.replay.backtest.simulation-accounting` | `backtest-simulation-accounting` | `integration-replay` | `covered` | Backtest trade lifecycle, state persistence, and accounting remain reproducible. |
| `unit.research.optimization-calibration` | `research-optimization-calibration` | `unit` | `covered` | Scenario and sweep identities remain stable enough to support research evolution. |
| `unit.verification.capability-ledger` | `verification-capability-evolution` | `unit` | `planned` | The MECC ledger, schema, README crosswalk, and evidence ownership become executable. |
| `e2e.live.ib-gateway-ssh-tunnel` | `broker-connectivity-account` | `e2e-live` | `wip` | A self-cleaning read-only canary proves q can reach the localhost-trusted Gateway and connect all three TradeBot clients. |

<!-- anchor: @future.seed.v1 -->
## Aspirational `.future.` Births v1

| Future contract ID | Primary subsystem | Falsifiable direction |
| --- | --- | --- |
| `benchmark.future.live-backtest-drift-score` | `market-realism-parity` | Score behavioural drift between replayed live traces and backtest decisions/fills. |
| `unit.future.content-addressed-cache-provenance` | `backtest-data-cache` | Make every cache artifact reproducible from source request, session rules, repair history, and code version. |
| `integration.replay.future.queue-latency-partial-fill-model` | `backtest-simulation-accounting` | Model queue position, latency, partial fills, cancels, modifications, and broker rejection semantics. |
| `benchmark.future.walk-forward-overfit-defence` | `research-optimization-calibration` | Require walk-forward stability, multiple-testing controls, and regime-diverse out-of-sample survival. |
| `integration.provider.future.entitlement-freshness-sla` | `live-market-data-contracts` | Continuously classify entitlement, freshness, delayed/frozen state, and quote degradation against explicit SLAs. |
| `e2e.live.future.shadow-order-digital-twin` | `live-execution-orders` | Run a non-submitting digital twin beside live execution to compare expected and broker-observed order evolution. |
| `benchmark.future.online-regime-drift-calibration` | `signal-regime-intelligence` | Detect regime-model drift and recalibrate without lookahead or unstable threshold churn. |
| `unit.future.account-constraint-aware-sizing` | `policy-risk-sizing` | Include margin, buying power, concentration, currency, and broker-account constraints in sizing oracles. |
| `integration.replay.future.causal-replay-journal` | `operator-ui-observability` | Replay every decision from durable facts and produce a causal operator narrative with no hidden state. |
| `e2e.live.future.tunnel-supervisor-failover` | `broker-connectivity-account` | Supervise SSH forwarding, detect stale tunnels, rotate client IDs, and recover without widening Gateway trust. |
| `integration.replay.future.corporate-actions-roll-adjustments` | `market-time-series-semantics` | Model splits, dividends, futures rolls, session changes, and symbol continuity without corrupting returns. |
| `unit.future.reproducible-runtime-manifest` | `runtime-configuration-state` | Capture dependency, configuration, preset, and data-version manifests for exact live/backtest reproduction. |
| `benchmark.future.current-market-canary-dashboard` | `verification-capability-evolution` | Track realism, parity, data quality, execution quality, and capability drift over time. |

### Promotion rule

A `.future.` row cannot become `covered` until it has executable evidence, resolvable owner surfaces, explicit oracles, deterministic or explicitly live test placement, and a README crosswalk.

<!-- END: TRADEBOT_MECC_TAXONOMY_V1 -->

<!-- BEGIN: TRADEBOT_MECC_IMPLEMENTATION_V1 -->

<!-- anchor: @implementation.native-minimal.v1 -->
## Native Minimal MECC Implementation v1

- Core: `tests/ledgers/capability_schema.json`, `tests/ledgers/capability_contracts.json`, `tests/test_capability_contracts.py`, `pytest.ini`, `README.md`.
- 13 subsystems; 43 deterministic tests with one primary owner; 28 initial covered/WIP/future contracts.
- Deterministic structural validator: passed.
- Next: land and execute `tests/live/test_ib_gateway_tunnel.py`, then promote e2e contracts.

<!-- END: TRADEBOT_MECC_IMPLEMENTATION_V1 -->

<!-- BEGIN: TRADEBOT_IB_LIVE_CANARY_V1 -->

<!-- anchor: @verification.ib-live-canary.v1 -->
## Explicit IB Gateway Live Canary v1

- `tests/live/test_ib_gateway_tunnel.py` owns four exact live proofs:
  - direct q → Mac Gateway TCP transport;
  - localhost-only Gateway trust remains intact;
  - read-only IB protocol handshake through disposable SSH forwarding;
  - unmodified TradeBot main/proxy/index clients connect with distinct IDs.
- Focused deterministic capability/connection baseline: passed.
- Explicit live canary: passed.
- Post-promotion structural validator: passed.
- Promoted contracts: `e2e.live.ib-gateway-direct-transport`, `e2e.live.ib-gateway-ssh-tunnel`.
- Safety receipt: no orders, market-data subscriptions, Gateway mutation, persistent tunnel, or persistent client-ID state.

<!-- END: TRADEBOT_IB_LIVE_CANARY_V1 -->

<!-- BEGIN: TRADEBOT_MECC_FINAL_VERIFICATION_V1 -->

<!-- anchor: @verification.mecc-final.v1 -->
## Native Minimal MECC Final Verification v1

- Full deterministic suite: passed with live infrastructure excluded by default.
- Explicit promoted IB Gateway canary: passed.
- Final structural validator: passed.
- Machine ledger audit: passed.
- Topology: 13 MECC subsystems, 28 capability contracts, 44 uniquely owned test files.
- Contract statuses: 14 covered, 14 planned `.future.` or verification births.
- Exact live evidence: four live test nodes mapped without stale or missing ledger references.
- Worktree boundary and `git diff --check`: passed.
- Connectivity conclusion: direct LAN TCP works; direct IB authorization remains intentionally localhost-only; disposable SSH forwarding enables the read-only protocol and TradeBot main/proxy/index sessions without widening Gateway trust.
- Safety conclusion: no orders, market-data subscriptions, Gateway mutation, persistent tunnel, or persistent client-ID state were introduced.

<!-- END: TRADEBOT_MECC_FINAL_VERIFICATION_V1 -->

<!-- BEGIN: TRADEBOT_MECC_CONCLUSION_CHECK_V1 -->

<!-- anchor: @verification.mecc-conclusion.v1 -->
## Native Minimal MECC Conclusion Check v1

- Final review found and corrected one Arcana-parity defect: the missing `integration-provider` layer.
- `integration.provider.future.entitlement-freshness-sla` now has layer `integration-provider` rather than `unit`.
- Validator now centralizes reference resolution, validates exact pytest nodes, checks non-empty contract fields, and enforces ID-prefix/layer coherence.
- Structural validator, full deterministic suite, explicit four-test IB Gateway canary, machine audit, worktree boundary, and diff check: passed.
- Final layer coverage: unit, integration-replay, integration-provider, e2e-live, benchmark.
- Root connectivity and safety conclusions remain unchanged.

<!-- END: TRADEBOT_MECC_CONCLUSION_CHECK_V1 -->

<!-- BEGIN: TRADEBOT_PARITY_COVERAGE_ATLAS_V1 -->

<!-- anchor: @parity.coverage-atlas -->
## Backtest ↔ Live/UI Parity Coverage Atlas v1

- Pushed milestone: `9261954424c4f1fe6f40a38bc3cf8aa9245737b5` on `origin/main`.
- Capability-contract closure: **14/28 (50.0%)**.
- Test-ownership registration: **44/44 (100.0%)**.
- Fine-grained semantic parity is scored in `@parity.first-slice-v1`; the umbrella atlas remains a separate contract-closure and ownership view.

| MECC subsystem | Covered / total contracts | Closure | Open births | Owned test files | Test functions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `runtime-configuration-state` | 1/2 | 50.0% | 1 | 1 | 2 |
| `market-time-series-semantics` | 1/2 | 50.0% | 1 | 6 | 30 |
| `signal-regime-intelligence` | 1/2 | 50.0% | 1 | 3 | 30 |
| `policy-risk-sizing` | 1/2 | 50.0% | 1 | 2 | 31 |
| `market-realism-parity` | 1/2 | 50.0% | 1 | 3 | 19 |
| `broker-connectivity-account` | 3/4 | 75.0% | 1 | 4 | 86 |
| `live-market-data-contracts` | 1/2 | 50.0% | 1 | 7 | 81 |
| `live-execution-orders` | 1/2 | 50.0% | 1 | 3 | 112 |
| `operator-ui-observability` | 1/2 | 50.0% | 1 | 6 | 16 |
| `backtest-data-cache` | 1/2 | 50.0% | 1 | 4 | 34 |
| `backtest-simulation-accounting` | 1/2 | 50.0% | 1 | 2 | 5 |
| `research-optimization-calibration` | 1/2 | 50.0% | 1 | 2 | 3 |
| `verification-capability-evolution` | 0/2 | 0.0% | 2 | 1 | 5 |

### Coverage rules

1. Contract closure measures ledger status only: `covered / all contracts`.
2. Test ownership measures registry completeness only: discovered test files represented exactly once.
3. Semantic-parity coverage will measure covered fine-grained parity capabilities after decomposition; its denominator must never be invented retrospectively to inflate progress.
4. Each centralization wave must add or refine contracts before claiming percentage movement.

<!-- anchor: @parity.shared-spines -->
## First Recursive Parity Frontier

Source and references must be saturated before mutation, in this order:

1. Shared market facts and configuration: `SpotRuntimeSpec.from_sources`, signal/context inputs, time and fill-mode semantics.
2. Shared decision truth: `SpotPolicy.resolve_position_intent` and `SpotPolicy.calc_signed_qty_with_trace`.
3. Backtest adapter: `_run_spot_backtest_exec_loop` and its fill/accounting callees.
4. Live/UI adapter: `BotSignalRuntimeMixin`, order-builder/reconciliation callers, and operator-visible state.
5. Parity receipts: identical normalized inputs must produce equivalent intent, sizing, lifecycle, fill assumptions, and accounting explanations where environment-specific I/O is excluded.

<!-- END: TRADEBOT_PARITY_COVERAGE_ATLAS_V1 -->

<!-- BEGIN: TRADEBOT_PARITY_FIRST_SLICE_V1 -->

<!-- anchor: @parity.first-slice-v1 -->
## Frozen Spot Decision/Execution Parity Denominator v1

- Milestone base: `9261954424c4f1fe6f40a38bc3cf8aa9245737b5`.
- Frozen capabilities: **20**.
- Semantic alignment: **19/20 (95.0%)**.
- Shared-covered: **13/20 (65.0%)**.
- Intentional adapters: **6/20 (30.0%)**.
- Unproven: **1/20 (5.0%)**.
- Confirmed gaps: **0/20 (0.0%)**.
- Machine-readable source: `tests/ledgers/parity_capabilities.json`.

| Capability | MECC subsystem | Classification | Status | Next |
| --- | --- | --- | --- | --- |
| `spot-v1.runtime-spec-normalization` | `runtime-configuration-state` | `shared-covered` | `aligned` | Expand malformed-input and policy-pack precedence cases. |
| `spot-v1.policy-config-normalization` | `runtime-configuration-state` | `shared-covered` | `aligned` | Extract duplicated pack-aware source lookup only after parity receipts. |
| `spot-v1.sizing-trace-kernel` | `policy-risk-sizing` | `shared-covered` | `aligned` | Add paired adapter-input scenario receipts. |
| `spot-v1.branch-size-scaling` | `policy-risk-sizing` | `shared-covered` | `aligned` | Add direct branch A/B and rounding parity vectors. |
| `spot-v1.position-intent-resolution` | `policy-risk-sizing` | `shared-covered` | `aligned` | Add negative-position scale-in/out matrix. |
| `spot-v1.open-lifecycle-resolution` | `policy-risk-sizing` | `shared-covered` | `aligned` | Add paired exit-priority and resize-cooldown scenarios. |
| `spot-v1.fill-due-time` | `market-time-series-semantics` | `shared-covered` | `aligned` | Add identical instant-based UTC↔ET paired cases across DST. |
| `spot-v1.deferred-entry-planning` | `market-time-series-semantics` | `shared-covered` | `aligned` | Add cross-adapter identical-plan fixture. |
| `spot-v1.pending-next-open-decision` | `live-execution-orders` | `shared-covered` | `aligned` | Add direct shared-function backtest/live state-vector receipts. |
| `spot-v1.time-mode-adaptation` | `market-time-series-semantics` | `intentional-adapter` | `aligned` | Prove equal instants around DST and overnight-session boundaries. |
| `spot-v1.account-reference-adaptation` | `broker-connectivity-account` | `intentional-adapter` | `aligned` | Define a typed normalized account-capacity model and provider replay vectors. |
| `spot-v1.execution-fill-adaptation` | `market-realism-parity` | `intentional-adapter` | `aligned` | Capture read-only IB fill/order-state shapes and replay them. |
| `spot-v1.order-submission-status-adaptation` | `live-execution-orders` | `intentional-adapter` | `aligned` | Add spot-specific staged→send-error/cancel/inactive/partial/fill scenarios. |
| `spot-v1.resize-mutation-accounting-adaptation` | `backtest-simulation-accounting` | `intentional-adapter` | `aligned` | Add broker-replay partial/full resize position and basis receipts. |
| `spot-v1.sizing-input-assembly-parity` | `policy-risk-sizing` | `shared-covered` | `aligned` | Add direct runtime capture for the backtest resize owner and non-default regime2/shock/risk value vectors. |
| `spot-v1.pending-state-mutation-parity` | `live-execution-orders` | `shared-covered` | `aligned` | Extend paired transition vectors to partial-fill, reconnect and reconciliation states. |
| `spot-v1.entry-basis-reconciliation` | `backtest-simulation-accounting` | `shared-covered` | `aligned` | Keep cumulative partial-fill, reconnect, scale-in/out, sign-flip, broker-average-cost and authoritative-flat receipts current. |
| `spot-v1.trace-projection-parity` | `operator-ui-observability` | `shared-covered` | `aligned` | Keep the versioned sizing, intent, lifecycle, fill and accounting receipt schema equivalent across adapters. |
| `spot-v1.exit-resize-adapter-receipts` | `verification-capability-evolution` | `unproven` | `unproven` | Build deterministic adapter harness and optional Gateway replay suite. |
| `spot-v1.resize-cooldown-fill-ownership` | `live-execution-orders` | `intentional-adapter` | `aligned` | Extend successful-fill ownership receipts to partial fills, reconnect recovery and broker-position reconciliation. |

### Wave 1 resolution: resize cooldown fill ownership

- Live resize construction no longer advances cooldown while staging.
- Broker `Filled` reconciliation is the sole live advancement point.
- Send errors and terminal non-fill states preserve the prior successful timestamp.
- Backtest continues to advance only after successful resize mutation.
- Focused receipt: `4 passed, 55 deselected, 1 warning in 0.40s`.
- Affected-file receipt: `59 passed, 1 warning in 0.43s`.
- Deterministic receipt: `465 passed, 4 deselected, 1 warning in 9.87s`.
- `spot-v1.resize-cooldown-fill-ownership` moved from `parity-gap` to aligned `intentional-adapter`.

### Wave 2 resolution: canonical sizing-input assembly

- Canonical owner: `tradebot/spot/policy_contract.py::SpotSizingInput` and `tradebot/engine.py::spot_sizing_input`.
- Static matrix: backtest entry and resize each provide all 29 canonical fields; live supplies 27 explicit fields and delegates `SpotPolicyGraph` and `SpotPolicyConfigView` construction to the shared factory.
- Runtime evidence: four backtest-entry payloads and one live-resize payload were captured; every payload was typed with graph/config populated, and the live approximately USD 2,200 profile carried `equity_ref=2200.0` and `cash_ref=1800.0`.
- Typed and legacy wrapper entry forms produced identical quantity and trace output.
- TDD receipts: semantic RED `8c3346697a225a6d012243c78980e824aac878e5af24ce3e56854e237ee3bb63`; exact/related GREEN `2365f85d748b9c19791d329d17e0d91a9700e013209b1ad6c5f7c00e79dad8f1`; full/value GREEN `4a1f960f748d1380910d9e481e2634a7ad33d672402c10ad5485f4c2bdf15843`.
- Full deterministic verification: `604 passed, 4 deselected`; zero sockets; implementation diff SHA `a15c06259161e6aac185c15cbdf4eda68f1fb2c02f91e3181bc0a0a69f481ff8`.
- `spot-v1.sizing-input-assembly-parity` moved from `unproven` to aligned `shared-covered`.
- Publication: code milestone commit `737ecb7fb1b00d4a5bdb55d1d479c463e5f1cd06` (`Centralize spot sizing inputs`) advanced `origin/main` from `bf593fe06afc61f5db42df3446bae5a0b763200f`; an independent fresh fetch verified the same commit, parent, subject, and exact nine-path scope.
- Publication receipt: `976166f6b39f4eccacbefde29e7de501fa90735c2e2328c117f1202c41530663`; independent pre-publication validation receipt `4144f35c05fdd514f1100f91ed74595b677e871ede3909c1a175c58fe3bfc829`; the sibling workspace finished clean and the original five-path dirty workspace remained unchanged.

### Wave 3 resolution: canonical pending-state mutation

- Canonical owner: `tradebot/spot/lifecycle.py::SpotPendingMutationPlan` and `tradebot/spot/lifecycle.py::plan_pending_mutation`.
- Adapter owners: `tradebot/backtest/engine.py::_spot_apply_pending_mutation` and `tradebot/ui/bot_signal_runtime.py::BotSignalRuntimeMixin._apply_pending_mutation`.
- Ten identical pending-entry/exit scenarios proved UTC-backtest and ET-live plan parity, exact clear behavior, state preservation and queue projection.
- Persistent evidence: `tests/test_spot_summary_parity.py::test_backtest_pending_state_mutation_matches_shared_transition_table` and `tests/test_live_signal_exit_continuum.py::test_live_pending_state_mutation_matches_shared_transition_table`.
- TDD receipts: denominator `81433dc08dbbbcf45fab0b665a647e8e71a7a2fe22cea8d0a4b17a00fc2384c9`; semantic RED `a7018da57423f2700317b8393e4e1e521df6b215885901298cc2a6b675ccfcba`; implementation GREEN `3a73fbf4bc2896847700943e0f5b39e6ee66eef51ebd6236d80cdc0839a0a6c4`; value/full-suite GREEN `c1437db7b1d25d64aeacc6f6443294183bbec3a6d96bdcfc324e4eb56dcc88a0`.
- Full deterministic verification: `606 passed, 4 deselected`; zero sockets; five-path implementation diff SHA `86942b1b459a5bf28ebc8cdd055dbfedff6b252cec6d53e5796189f771001cc4`.
- Promotion boundary receipt: `38f85952542c44e660fa0a89560c3ed8e65e2b80f496e54b5da3f94251c72a53`; README remained a certified no-op because its crosswalk is owned by `tests/ledgers/capability_contracts.json`.
- `spot-v1.pending-state-mutation-parity` moved from `unproven` to aligned `shared-covered`.

- Publication: pending-state milestone commit `29256f119f8c522b4dcc669246bd94e3a96e2328` (`Centralize spot pending state mutation`) advanced `origin/main` from `53ebd8e4a3de32f745ed711b8ca7eb58612dddfe`; explicit SSH push and an independent fresh fetch verified the same commit, parent, subject, exact seven-path scope and diff SHA `9052b5f4a59112e3fd95a7b7e0c16fa5f92bc5aac9e62ed626880c840e8823ef`.
- Publication receipt: `62c78a3dfb5cffc7d5ce177f388ee090dc2724d45fa0164d442a022e6640419e`; independent pre-publication validation receipt `d17fb3c246973b51df3b91db500d118145e27dffa9ccc5ea310aa71b3235af4f`; the sibling workspace finished clean and the original five-path dirty workspace remained unchanged.

### Wave 4 resolution: canonical entry-basis reconciliation

- Canonical owner: `tradebot/spot/lifecycle.py::SpotEntryBasisState` and `tradebot/spot/lifecycle.py::reconcile_spot_entry_basis`.
- Backtest adapters: `tradebot/backtest/engine.py::_spot_try_open_entry` and `tradebot/backtest/engine.py::_spot_apply_resize_trade`.
- Live adapters: `tradebot/ui/bot_engine_runtime.py::BotEngineRuntimeMixin._apply_spot_entry_basis_fill`, `tradebot/ui/bot_screen/positions.py::BotPositionsMixin._apply_spot_broker_entry_basis`, and `tradebot/ui/bot_screen/positions.py::BotPositionsMixin._refresh_positions`.
- The shared nine-row matrix covers initial long/short fills, terminal partial entry, same-direction scale-in, scale-out, flat, overclose sign flip, reconnect duplicate no-op, and broker-average-cost authority.
- Live cumulative-fill application uses `_BotOrder.basis_applied_filled_qty`; signed increments derive from typed order action; `_BotInstance.spot_entry_basis_qty` provides the prior inventory state.
- Successful broker snapshots project non-flat `position`/`averageCost`, explicit zero rows, and omitted positions; a failed portfolio fetch preserves prior basis and emits the existing status error.
- Persistent evidence: `tests/test_spot_summary_parity.py::test_backtest_entry_basis_reconciliation_matches_shared_fill_table`, `tests/test_live_signal_exit_continuum.py::test_live_entry_basis_reconciliation_matches_shared_fill_table`, and `tests/test_live_signal_exit_continuum.py::test_live_broker_refresh_clears_entry_basis_without_open_position`.
- TDD receipts: corrected paired RED `c01a766222592ae32c2bfb7e93010b9727cc39cd62cf81d18a0f4b1496995812`; initial implementation GREEN `cec2336d408242477042fb87d30e15eb9bb73f166b2475f6c290235e6b50833e`; broker-flat RED `a0efffeec1a77e7398c96c0add4fae2d32ef3633a80ca1f374aa4740d5db540f`; broker-flat GREEN `691b45d947ea8bc93a6fe8c5f1ac884decb44081ef971982cc7235de66d7e5db`; independent promotion boundary `31551d5218284a8c213e8ef20fb5fceb10e773dca795a8e720f89e5187744dcf`.
- Full deterministic verification: `610 passed, 4 deselected`; zero sockets; test diff SHA `85602f6d84c7fd2540ac7e6ba135a78d4010f02993d300e5d9949e99c91c3f11`; production diff SHA `e4d5041fb8fadd871a478d08cb29e4835539ef7675e3d8e76a3e76d71596ec08`; combined implementation diff SHA `252c3b1de8822210fb479c2c98aabc8ee3e99e5d2b6423d04dc9c604d973a138`.
- README remained a certified no-op because its crosswalk validator is backed by `tests/ledgers/capability_contracts.json`, not this parity ledger.
- `spot-v1.entry-basis-reconciliation` moved from `unproven` to aligned `shared-covered`; the frozen denominator is now 18/20 aligned, 12/20 shared-covered, 6/20 intentional adapters and 2/20 unproven.

### Wave 4 publication: canonical entry-basis reconciliation

- Code milestone: `180582b76ebe6a1b11540183446fdbcbc894d1fb` (`Centralize spot entry basis reconciliation`), parent `0a2176530206747ef857665ec45fa5cc8568abc3`.
- Explicit SSH push advanced `origin/main` from `0a2176530206747ef857665ec45fa5cc8568abc3` to `180582b76ebe6a1b11540183446fdbcbc894d1fb` without changing origin configuration.
- Independent fresh fetch verified the exact nine-path scope and promoted diff SHA `ffb3d5046da99b991e256aa6ce25b0f65a97a4c00d978e53c405059cfc29d718`.
- Published verification: focused `16 passed`; capability validators `5 passed`; full deterministic suite `610 passed, 4 deselected`; zero sockets.
- Independent publication receipt: `338f92416919b1298250e330f542eda98679a99ee0605ede02140452551c7bdf`.


### Wave 5 — trace projection parity

- **Correction status:** the initial projector-only proof is superseded; this wave is now grounded in production-owned backtest, live builder and live fill/accounting routes.
- **Production-route RED:** two unchanged route tests failed only on absent production `spot_trace_receipt` keys; receipt `2ecd66750573e8b0b0562a4d4175d6cddd9fa4745e0c3f38339bf28e5059037b`; tests diff `cac88a10a522c95d904adcc7f636c2ca6cc4f11e82972c80bba3870b194324ef`; zero sockets.
- **Production-route GREEN:** the same two tests passed `2/2` after canonical routing; receipt `966a9026ebdf29706e13ecf3f0d1de989569ecb4caa4e5e1774a1b20486bc08e`; production diff `c4ab4ca0be0ce3523898c1111aaeeccb2927ded0b27e2c4546a8126edb003cc9`; combined diff `517e7d4459fc047eebc7f966b64656922b425e2f3876cb4d7440a692d9b7f729`; zero sockets.
- **Canonical ownership:** backtest final trade projection is owned by `_run_spot_backtest_exec_loop`; live seed is owned by `project_live_spot_order_journal`; live fill/accounting refresh is owned by `BotEngineRuntimeMixin._apply_spot_entry_basis_fill`.
- **Test integrity:** both route tests contain zero direct calls to `project_spot_trace_receipt`; the projector is exercised through production callers only.
- **Focused validation:** `6/6` route and harness tests passed with zero sockets; validation receipt `01a21a21cd602cfb025e0ff28fc2ad37ceefc60dd2dc64013123934525e5dc60`.
- **Capability validation:** exact `tests/test_capability_contracts.py` validator set passed `5/5` with zero sockets; receipt `dbdacd8fdf53875c30c3b67390bcaefb232605c71f4024de1071c3d797e0e583`.
- **Full deterministic validation:** `635/635` passed, four deselected, zero sockets; validation receipt `01a21a21cd602cfb025e0ff28fc2ad37ceefc60dd2dc64013123934525e5dc60`.
- **Coverage conclusion:** `spot-v1.trace-projection-parity` remains `shared-covered` and `aligned`; no capability-ledger classification change is required.
- **Provider boundary:** no provider or IB Gateway interaction occurred in RED, GREEN or deterministic validation.
- **Publication:** integrated commit `f5f1353d0e9f8fde2012c7b2faf95882d7d0593f` was pushed to `main` from `ab9139d8358ba719974fa5f237985019d20656f3` with tree `8b1ff63bf607d616d697170b538a94cc7a2094cf`; publication receipt `8e54e2ba5c447e317d604b42210d581162bd61a92376f7f057c2061e5505fcfd`.
- **Published verification:** remote-only checkout passed focused `6/6`, capability `5/5`, and deterministic `648/648` with four deselections and zero sockets; independent pre-publication receipt `e747412a871022678d5f2524944cc2261cf7cf3d14d1accc79ed0a4f7c96f277`; no provider or IB Gateway access.
<!-- END: TRADEBOT_PARITY_FIRST_SLICE_V1 -->

<!-- BEGIN: TRADEBOT_ARCHITECTURE_PRIORITY_QUEUE_V1 -->

<!-- anchor: @architecture.priority-queue-v1 -->
## Architecture Priority Queue v1

Dependency order begins immediately after Wave 1 verification:

1. Native atomic multi-leg option execution: broker-qualified BAG/combination identity, ratios/actions/exchanges, debit-credit economics, what-if margin, atomic submit/modify/cancel/reconcile, partial/failure risk, and combo close/roll/rescue. Provider evidence must precede each product and structure admission; XSP, full-size SPX, and MCL remain distinct qualification, settlement, liquidity, multiplier, and risk domains.
2. Canonical strategy/execution model shared by research, backtest and live: intent, combo economics, costs, queue/latency/partial/reject/cancel assumptions, margin, assignment/exercise, rolls and rescue; broker/simulator mutation remains an intentional adapter.
3. Weekly resumable research/promotion loop: walk-forward and regime/timeframe slices, robustness, turnover/capacity, tail loss, drawdown, ruin and margin stress, leakage/overfit and execution sensitivity; canonical caches and versioned benchmark tables.
4. Evidence promotion ladder: candidate → deterministic replay → provider integration → paper/canary → bounded live, with backtest↔live calibration across decisions, orders, fills, PnL, latency, slippage, rejects, margin, assignment and rescue.
5. Ledger discipline: each umbrella names its canonical surface, intentional adapter, executable receipt, metric/denominator and promotion gate; semantic parity and contract coverage remain separate.

Safety constraints:
- No guaranteed-profit claims.
- No static account-size floor applies. The approximately USD 2,200 IBKR account is a first-class target profile; eligibility is account-, contract-, structure-, and order-specific and must be measured through qualification, permissions, broker what-if buying-power/margin impact, exact maximum loss, width and quantity, commissions, liquidity/fill/slippage, expiry/settlement/assignment/exercise behavior, and aggregate correlated/tail exposure.
- Explicit targets include broker-native XSP vertical credit spreads, defined-risk short iron condors, multiple concurrent spread positions, and admitted rolling/rescue cycles. Keep XSP, full-size SPX, and MCL semantics distinct; reject only on measured constraints and never claim guaranteed or intrinsically safe income.
- Research objective: discover robust, risk-controlled weekly-income candidates that genuinely fit the approximately USD 2,200 account, then promote them only through realistic backtest, deterministic replay, provider integration, paper/canary, bounded live, and backtest↔live calibration evidence.
- Commit and push only after focused and broad receipts pass.

Next cycle: recursively map existing combination-contract, option-leg, margin, backtest-strategy, optimizer and promotion surfaces before extending the machine denominator.

<!-- END: TRADEBOT_ARCHITECTURE_PRIORITY_QUEUE_V1 -->

<!-- BEGIN: TRADEBOT_SPOT_CONTRACT_IDENTITY_V1 -->

<!-- anchor: @strategy.spot-contract-identity-v1 -->
## Spot Contract Identity v1

- Covered contract: `unit.strategy.spot-contract-identity-parity` in `live-execution-orders`.
- Canonical owner: `tradebot/contract_identity.py`; nullable typed `spot_sec_type` and `spot_exchange` preserve explicit identity while absent values delegate to the registry.
- MCL is `FUT` / `NYMEX` / multiplier `100.0`; MNQ is `FUT` / `CME` / multiplier `2.0`; unknown equity symbols remain non-futures and use stock defaults.
- Centralized consumers: research payloads, historical/calibration/backtest/options, offline sweeps, UI configuration/live signal contracts, and client exchange/order normalization.
- TDD receipts: remaining-consumer RED `052c62529c638aba416855c4ca640663335e942d3ea96a43a3ac773441cf0dc6`; duplicate-free transport-clean GREEN `14a700706d3a06907620bcae474d4f01f39a559ddbf6c880bcef7f021b13bc11` (`2` exact and `243` collected/passed, zero sockets). Direct canonical-registry evidence: `tests/test_live_signal_exit_continuum.py::test_canonical_contract_identity_registry_covers_mcl_mnq_and_equity_controls`.
- Full deterministic verification: `601 passed, 4 deselected`; selected denominator `601`; focused coverage `10 passed`; zero sockets.

- Publication: code milestone commit `82a6a452c31d6ea44390b8dfb45005c2807fbd40` was pushed as a fast-forward from `d68a496820dcc8bfdfda9a845250c52f03b780b0` to `origin/main`; a fresh independent fetch resolved `refs/heads/main` to the same commit; the original dirty workspace remained at `90447ff29f177d1e216a38b07b47d923487e18cb` with its exact five-path status and empty index.

<!-- END: TRADEBOT_SPOT_CONTRACT_IDENTITY_V1 -->
