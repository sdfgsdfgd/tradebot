# TQQQ HF Research (High-Frequency Spot Track)

This file is the dedicated TQQQ high-frequency evolution track.
- `readme-lf.md` holds the low-frequency and broader historical TQQQ lineage.
- This file tracks the throughput-biased HF line and its own HF crowns.

Canonical execution paths:
- Spot sweeps/evolution: `python -m tradebot.backtest spot ...`
- Multiwindow kingmaker eval: `python -m tradebot.backtest spot_multitimeframe ...`

## Evolutions (stack)

### v1.0 - shock-throttle HF bootstrap seed (PROMOTED)
Status: **DONE (promoted as first TQQQ HF anchor)**

Promotion rationale:
- Establish a real, existing milestone-backed HF baseline in `backtests/tqqq/readme-hf.md` so Bot UI can auto-load TQQQ HF from a dedicated track.
- Keep all-window positivity and strong stability while nudging turnover above the LF v39 baseline.

Selected artifact:
- Ranked top set: `backtests/out/tqqq_exec5m_v37_shock_throttle_refine_30m_10y2y1y_mintr100_top54_2026-02-05_180208.json`
- Source variants: `backtests/out/tqqq_exec5m_v37_shock_throttle_refine_variants_30m_2y_2026-02-05_180134.json`

Observed kingmaker #01 windows:
- 10y (`2016-01-01 -> 2026-01-19`): trades **1057**, pnl **71,510.39**, dd **16,006.54**, pnl/dd **4.4676**
- 2y (`2024-01-01 -> 2026-01-19`): trades **206**, pnl **35,727.55**, dd **4,587.49**, pnl/dd **7.7880**
- 1y (`2025-01-01 -> 2026-01-19`): trades **107**, pnl **15,934.55**, dd **3,868.21**, pnl/dd **4.1194**

## Current Champions (stack)

### CURRENT (v1-shock-throttle-seed) - first TQQQ HF promoted anchor
- Preset file: `backtests/out/tqqq_exec5m_v37_shock_throttle_refine_30m_10y2y1y_mintr100_top54_2026-02-05_180208.json`
- Source eval: `backtests/out/tqqq_exec5m_v37_shock_throttle_refine_30m_10y2y1y_mintr100_top54_2026-02-05_180208.json`
- Source variants: `backtests/out/tqqq_exec5m_v37_shock_throttle_refine_variants_30m_2y_2026-02-05_180134.json`

Current HF champion replay:
```bash
python -m tradebot.backtest spot_multitimeframe \
  --milestones backtests/out/tqqq_exec5m_v37_shock_throttle_refine_30m_10y2y1y_mintr100_top54_2026-02-05_180208.json \
  --symbol TQQQ --bar-size "30 mins" --use-rth --offline --cache-dir db \
  --top 1 --max-open 1 --require-positive-pnl --min-trades 100 \
  --window 2016-01-01:2026-01-19 --window 2024-01-01:2026-01-19 --window 2025-01-01:2026-01-19
```

## Research Notes (migrated)

## Quest & Upcoming Feature: News aggregation overlay, sentiment scoring, and event-driven capabilities
## (multiround GPT parsing on breaking news / events for the final overlay / scoring step)

# Research on 2024 vs 2025 TQQQ volatility spikes, early detection analysis of weekly TR and gap patterns, before sharpening reg gates

• 1. Feb 28 onward: is there a “distinct slope” you can catch (rare, and also hits COVID etc)?

    Yes — but the cleanest “early warning” in the data is not ATR14% level (that lags), it’s TR% velocity / weekly TR regime, and it shows up late Feb → early Mar.

  
  Key facts from the daily RTH bars (db/TQQQ/TQQQ_2016-01-01_2026-01-19_1day_rth.csv), using TR% = TrueRange / prev_close:

  - The week containing Feb 28, 2025 (ISO week starting 2025-02-24) has:
      - weekly TR% median 6.10%
      - week-over-week jump (delta) +2.94
      - that delta is 93.9th percentile vs all week-over-week TR% median changes in 2016–2026
      - drawdown proxy at Feb 28 close (dd60) already about -23%
  - The week starting 2025-03-03 is the real “oh, this is different” week:
      - weekly TR% median 9.50% (95th percentile of weekly TR% medians)
      - delta vs prior week +3.40 (95.6th percentile)
      - 4-week “ramp” (TR_med now vs 4 weeks prior) +5.51 (96.5th percentile)

  So if you’re looking for a slope/angle signal (like your screenshot), it’s real — it just starts showing up as volatility acceleration in the week of Feb 24, and becomes full-blown stress in the week of Mar 3.

  A very sharp “panic week” rule that barely over-triggers
  If you want something that:

  - is rare across 10y,
  - also catches COVID / 2022 / 2018Q4-type events,
  - and is clearly “risk-off downtrend stress”…

  This is the sharpest thing I found from your requested ingredients (TR% + gaps), in weekly form:

  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)

  Results over 2016–2026:

  - Flags 17 weeks total (very low).
  - 16/17 (94.1%) of flagged weeks have dd60 ≤ -30% by that week’s last trading day.
  - It flags the exact week you care about: 2025-03-03 (and the extreme week 2025-04-07).
  - It also flags the big historical stress clusters:
      - 2020-02-24, 2020-03-09, 2020-03-16…
      - 2022-03-07, 2022-04/05 clusters…
      - 2018-12 etc.

  This meets your “doesn’t trigger except huge downtrends” requirement better than ATR-based rules.

  2. What do the ATR/TR “waves” look like in the 1–2 years before (weekly view + why March ramps / April peaks)

  Your monthly read is correct: Feb isn’t the volatility peak — March ramps, April peaks. Weekly makes the shape obvious.

  Here’s the “wave” context (weekly medians):

  2024 summer spike (non-crisis-like compared to 2025)

  - 2024-07-29: TR_med 9.87%, neg_gaps 20%
  - 2024-08-05: TR_med 9.70%, neg_gaps 40%
  - Then it mean-reverts quickly.

  So 2024 can print TR_med ~9–10, but it wasn’t “persistent negative-gap stress”.

  2025 ramp + peak

  - 2025-02-24: TR_med 6.10%, neg_gaps 40%  (the “velocity” week)
  - 2025-03-03: TR_med 9.50%, neg_gaps 60%  (panic threshold week)
  - 2025-03-10: TR_med 7.52%, neg_gaps 60%
  - 2025-03-31: TR_med 8.70%, neg_gaps 100%
  - 2025-04-07: TR_med 23.56%, neg_gaps 80% (the peak/reversal chaos)

  That Apr 7 week is exactly your “ATR peaks during reversal” intuition — and you can also see why ATR14% is lagging: the TR and gaps explode first, then ATR catches up.

  For comparison, COVID looks similar in “weekly TR shock” shape, just bigger:

  - 2020-02-24: TR_med 13.08%, |gap|_med 7.54%
  - 2020-03-09: TR_med 20.60%, |gap|_med 16.79%
  - 2020-03-16: TR_med 24.54%, ATR14_med 24.93%

  That’s the same pattern class: gap/true-range explosion first, ATR ramps after.


  3. ATR-expansion vs TR-velocity: which actually catches Mar 5 early enough?

  This part matters for your “prevent the March 5 overnight gap loss” goal.

  - The ATR14%-expansion idea (fast5 median ATR% vs slow20 median ATR%) is too laggy for Mar 5 in your data.
      - With a reasonable “expansion” rule (fast5 ≥ ~6 and diff ≥ ~1.5), it turns on around Mar 6–7, not by Mar 5 close.
  - A TR-based rolling median triggers in time:
      - If you define risk_off_TR5 = median(last 5 daily TR%) ≥ 9.0%
      - It flags 2025-03-04, 03-05, 03-06, 03-07, 03-10 (and then the April cluster)
      - So it is ON at Mar 5 close, which is exactly what you need to avoid holding into Mar 6 open.

  Also, your key “bad overnight” is absolutely consistent with this:

  - On 2025-03-06: daily gap% ≈ -5.05% (big negative gap day)
  - Your strategy did have exposure over the Mar 5 → Mar 6 boundary.

  ## How rare is this TR5 detector (does it overtrigger)?
  - TR5_med ≥ 9 flags 159 days out of 2516 trading days (~6.3% of days).
  - It clusters mainly in the “obvious” stress regimes (2018Q4, COVID, 2022 bear, 2025).
  - It’s fairly selective for deep drawdown: ~84% of flagged days have dd60 ≤ -30%.

  - The champ is in-position overnight on 2101 close→next-open boundaries.
  - If you applied risk-off only on TR5_med≥9 days:
      - it would affect 114 of those overnights (5.4%).
  - If you applied the stricter weekly-panic regime:
      - it would affect 51 overnights (2.4%).

  So: it’s rare enough to plausibly be “surgical”.
  But: the hard trade-off
  Those risk-off nights are not free to avoid:
  - They have fatter left tails (bad gaps), but also fatter right tails (good gaps).
  - For TR5 risk-off overnights, gap-PnL distribution (per overnight boundary) looked like:
      - p05 about -1405 vs normal p05 about -587
      - min about -2950
      - but mean is actually higher than normal (because some huge rebounds exist)



  ### Even with a no‑lookahead “online weekly” implementation, it will NOT catch the Mar 5 overnight loss
  - Option A (most aligned with your intent): cancel next-open entries + forbid new late-day entries when risk_off is ON (prevents “entered near end of day then got gapped”)
  - Detector 2: weekly panic (TR_week_med ≥ 9 AND neg gaps ≥ 60%)

  So the clean integration is: add a new “risk_off overlay” under filters (parallel to shock, not replacing it), and in the spot engine:

  - refuse scheduling pending_entry_dir if it would fill next morning (overnight),
  - refuse fills/scheduling after a cutoff time when risk_off is ON.


  - Weekly panic = (weekly TR% median ≥ 9.0%) AND (≥ 60% of gaps are negative)
  …was computed using the full completed week in our analysis.
  
  - If we naïvely apply “this week is panic” during the week, that’s future leakage (you don’t know Friday’s gap sign on Tuesday).
  This is important because your motivation was specifically the Mar 5 15:05 long → Mar 6 09:35 stop.

  That trade is in your focus CSV:

  - 2025-03-05T15:05:00 → 2025-03-06T09:35:00, pnl -1231.52

  Now, check the actual daily gaps that week (from the daily bars):

  - Mar 3 gap +
  - Mar 4 gap -
  - Mar 5 gap +
  - Mar 6 gap -
  - Mar 7 gap -

  So the neg-gap fraction only reaches 60% on Friday (3/5).
  Meaning: weekly panic turns ON too late to block the Mar 5 → Mar 6 gap unless you allow lookahead (which we can’t).

  So: weekly panic is extremely selective (good), but it doesn’t solve the exact Mar 5 event (bad). It will still help April and other crisis clusters though.

  ———

  ## Solution Candidates (with concise ratings):

  1. TR5 Risk‑Off Overlay (no lookahead) + Option A hygiene [Score: 9/10] ← IDEAL for your Mar‑5 objective
     • Define risk_off = median(last 5 *completed* daily TR%) >= 9.0 (computed from daily bars; no future leak).
     • Apply Option A: block late-day entries + cancel overnight next_open fills when risk_off ON.
     Pros:
      - Actually ON by Mar 5 intraday (using completed days up to Mar 4), so it can block the 2025-03-05T15:05 entry.
      - Still fairly selective (clusters mainly in 2018Q4 / 2020 / 2022 / 2025).
        Cons:
      - Less “rare” than weekly panic (more flagged days than 17 weeks).


• Yes — a news/GPT layer can very plausibly complement your system, and it complements it in exactly the place where your purely market‑derived detectors are structurally weak:

  - Market detectors (TR%, gaps, ATR) are great at saying “volatility regime changed”.
  - They’re much worse at saying “this is the start of a sustained risk-off narrative” early enough (especially if the first big move is a weekend gap), and they’re also bad at saying “the narrative has flipped; this is now a
    rebound regime” (because TR stays elevated into rebounds).

  That’s why your TR5 / weekly‑panic research is still correct and still valuable, even though the naïve “cancel overnights during riskoff” didn’t beat the champ: the detectors are real, but they’re missing the missing variable:
  narrative direction + de‑escalation detection.

  ### Why your TR% + negative-gap signals are still “the right ingredients”

  From your pasted research:

  - They were selective (weekly panic barely triggers).
  - TR5 triggers early enough to matter for the Mar 5 → Mar 6 gap-loss.
  - But they stay “ON” into April rebound conditions because the market is still chaotic.

  So the problem isn’t “TR5 is wrong”. The problem is: TR5 is an amplitude detector; it’s not a direction+phase detector.

  ### Where a news/GPT layer actually helps (and why it could fix the April rebound problem)

  A news/GPT layer can provide two things the market-only layer can’t reliably provide without lag:

  1. Pre-gap anticipation (weekend / overnight)
      - If the first real damage happens between sessions, the only thing you can “see” beforehand is:
          - the narrative worsening (tariff escalation, retaliation headlines, etc.), and/or
          - the market already being in a fragile state (TR ramp, more negative gaps, worsening dd proxy).
      - TR5 only “turns on” after you’ve already had several large TR days.
  2. Reversal / de-escalation detection
      - This is the killer: market volatility stays high during rebounds.
      - A good news classifier could flip from:
          - “escalation / risk-off / negative expected drift”
          - to “de‑escalation / relief / positive expected drift”
      - …even while TR remains high.
      - That’s precisely how you avoid the failure mode we saw: “risk-off stays elevated into the rebound, so you miss the big rebound long(s).”

  ### The right “shape” of an overlay once you have news

  The overlay that makes sense is not “TR5 cancels overnights.” It’s more like a risk-state machine that uses both market data and news.

  Think in terms of a state with at least these outputs:

  - risk_on (bool / intensity)
  - risk_dir ∈ {down, up, neutral}
  - risk_phase ∈ {escalation, panic, cooling, relief}

  Then your actions become directional and phase-aware, e.g.:

  - If risk_dir=down and risk_on=true:
      - block/cancel only late-day long entries (especially ones that will be held overnight),
      - keep/allow shorts (maybe scaled, but carefully),
      - tighten long stops, allow more aggressive loss-cutting,
      - optionally disallow holding long over weekends specifically.
  - If risk_dir=up and risk_on=true (high-vol rebound / relief rally):
      - do not block the rebound longs (this is exactly where the earlier TR-based overlay accidentally hurt you),
      - if you do anything, you scale down size rather than block participation.

  This is exactly the missing lever: the TR/gap regime says “this is a tail-risk environment”; the news regime says “tail-risk is currently skewed down vs skewed up.”

  ### Where it should plug into your pipeline (conceptually)

  With news available, I would still not jam it into EMA/supertrend math directly. I’d treat it like shock: a parallel overlay that can influence:

  - Entry permission (block/cancel specific kinds of entries; especially near EOD / weekend)
  - Sizing (risk budget scaling)
  - Exit semantics (e.g., allow flip-at-loss only when risk_dir opposes your position)

  That keeps your core alpha logic stable, and isolates the “narrative risk” system.

