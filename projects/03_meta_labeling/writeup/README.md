# Triple-Barrier Labeling + Meta-Labeling at Scale (López de Prado, AFML Ch.3; ML4AM Ch.5)

> Article: [daru.finance/research-review/lopez-de-prado/labeling-and-cross-validation](https://daru.finance/research-review/lopez-de-prado/labeling-and-cross-validation)

Reproduces López de Prado's **triple-barrier labeling** and **meta-labeling**,
then tests across **Crypto + US Equities + Forex** (42 instruments) whether a
secondary meta-model improves a structural primary signal, judged by the
program's **headline metric, the Deflated Sharpe Ratio (DSR)**, net of realistic
costs, with PBO and effective-N as supporting deflation diagnostics.

Everything runs on **real 1-minute data** (no synthetic series), every position
change is **costed**, every barrier scan uses **full intrabar OHLC**, and the
meta-model is trained/evaluated with **purged k-fold CV** so overlapping
triple-barrier labels cannot leak.

Headline up front, because the house rule is to say so when it doesn't work:
**meta-labeling is a real, repeatable improvement over the primary (it lifts PF
in 38/42 instruments and DSR in 39/42), but on this MA-crossover primary it is
*not enough* to manufacture a deflated edge, 0 of 42 instruments clear DSR>0.95
for either the primary or the meta variant.** Details below.

---

## 1. What LdP proposed (formulae)

### Triple-barrier labeling (AFML §3.2-3.4)
For each event starting at bar *t₀* with a known **side** *s* ∈ {+1,−1}, place
three barriers and label by which is touched **first**:

- **Upper (profit-take)** horizontal barrier at price `entry · exp(+pt·σ_{t0})`
- **Lower (stop-loss)** horizontal barrier at price `entry · exp(−sl·σ_{t0})`
- **Vertical (max-holding)** barrier at `t₀ + max_hold` bars

`σ_{t0}` is a causal **EWMA of close-to-close returns** (LdP's `getDailyVol`
analogue). The horizontal barriers are vol-scaled multiples `(pt, sl)`; the
holding and lookbacks are **IS-tunable knobs**. The label is `+1` (PT hit), `−1`
(SL hit), or `0` (vertical/timeout). The realised side-signed path return at
first touch, **minus costs**, is the trade P&L.

### Meta-labeling (AFML §3.6; ML4AM Ch.5)
A **primary** model fixes the *side* of every bet (here a structural EMA-
crossover). Triple-barrier gives the realised outcome along that side. A
**secondary** classifier predicts **P(the primary's bet is profitable)** from
causal features. This meta-label decides **whether to act** and the **bet size**,
it can *veto* and *size*, but it **cannot flip the side**. LdP's point: meta-
labeling raises **precision / F1** and lets you drop low-confidence bets,
converting a high-recall / low-precision primary into something tradeable.
Bet size ∝ p (we use size = p above a threshold; the `m = 2·Φ(z)−1` map is a
drop-in alternative).

### Deflated Sharpe Ratio, the headline (LdP & Bailey 2014)
A backtest selected from many trials has an inflated Sharpe. DSR is the
Probabilistic Sharpe Ratio of the *selected* strategy benchmarked against the
**expected maximum Sharpe of N skill-less trials**,
`SR₀ ≈ √(V[SR]) · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`,
where N and V[SR] come from the trial set. We treat the **27-point IS-tunable
parameter grid as the trials**. A "win" must survive this deflation.

---

## 2. Method & data

| Market | Source (real 1m) | Bars | Cost / side |
|---|---|---|---|
| Crypto (27 perps) | `ldp_cache_1m/*_1m.parquet` | **dollar bars** (~20k) | 7 bp |
| US Equities (7 ETFs) | Algoseek `etf_1min/*.csv.gz`, RTH only | **dollar bars** (~20k) | 2 bp |
| Forex (8 pairs) | `ldp_cache_fx/*_fx1m.parquet` (no volume) | **tick bars** (~20k) | 1 bp |

- **Costs.** Per *side* of notional, applied on entry **and** exit (one full
  turnover = 2× the per-side cost). For the meta variant, cost scales with the
  bet size. Crypto 7 bp, equities 2 bp, forex 1 bp.
- **Primary.** Side = sign(EMA_fast − EMA_slow); an event fires at each
  **crossover instant** (so events don't massively overlap).
- **Secondary.** `BaggingClassifier(DecisionTree(max_depth=4, min_leaf=20),
  n_estimators=40)`, LdP's bagged-trees recommendation. Features are all
  **causal**: side, EWMA vol, MA-gap, 3 momentum horizons, Wilder RSI(14),
  short/long vol ratio, order-flow imbalance (crypto only; 0 elsewhere),
  smoothed intrabar range.
- **No leakage.** Out-of-fold P(profit) via `purged_kfold_splits` (6 folds, 1%
  embargo, label_span purged around each test fold). Barriers use **high/low**,
  never close-only. Equities use within-session bars (RTH) so overnight gaps
  aren't mislabeled as path moves.
- **IS-tunable, not enumerated strategies.** The structural shape (EMA-xover ×
  triple-barrier × bagged-tree meta) is one strategy. The numeric knobs,
  fast/slow ∈ {10/30, 20/60, 30/90}, (pt,sl) ∈ {(1,1),(1.5,1),(2,1.5)},
  max_hold ∈ {25,50,100}, are the **27 trials**. We select the trial with the
  best out-of-fold meta Sharpe and deflate against the trial dispersion.

---

## 3. Reproduction (the textbook pieces)

- **Triple-barrier first-touch** matches LdP's convention, including the
  pessimistic intrabar ordering (the adverse / stop barrier is checked first
  when both are touched in the same bar).
- **Meta-label lifts precision over the base rate** in every market: the
  fraction of *acted* bets that are profitable exceeds the unconditional
  profitable-bet rate (median precision lift +5-6 pp; see
  `fig1_precision_pf_by_market.png`, right panel). This is exactly the
  precision/recall trade LdP describes, the meta-model throws away bets to buy
  precision.
- **Bet-size distribution / trades dropped** (`fig2_betsize_distribution.png`):
  the meta P(profit) distribution sits around the act-threshold; the fraction of
  events vetoed ranges from a few percent (equities, very selective) to ~45%
  (crypto/forex).

---

## 4. At-scale multi-market result

By-market medians (net of costs; SR annualized; DSR is the headline):

| market | n | med PF prim→meta | med SR_ann prim→meta | med DSR prim→meta | # DSR>0.95 | med PBO (meta) |
|---|---|---|---|---|---|---|
| crypto   | 27 | 0.92 → **1.03** | −0.61 → **+0.24** | 0.000 → 0.017 | 0 | 0.41 |
| equities | 7  | 0.75 → **1.03** | −0.68 → **+0.31** | 0.000 → 0.095 | 0 | 0.38 |
| forex    | 8  | 0.85 → 0.94     | −1.13 → −0.27     | 0.000 → 0.000 | 0 | 0.19 |

Cross-sectional tallies (42 instruments):

- **Meta improves PF over primary in 38/42; improves DSR in 39/42.**
- Instruments with PF>1 rise from **12 (primary) to 24 (meta)**.
- **DSR>0.95: 0 (meta) and 0 (primary).** The single best is SPY at meta
  DSR=0.76, and that rests on only **6 acted trades** (see caveat below), so it
  is *not* a credible deflated edge. The next best are crypto AVAX (0.59),
  forex USDJPY (0.36), crypto XRP (0.29), all comfortably under the bar.
- See `fig4_dsr_by_market.png`: meta (orange) shifts the whole DSR cloud upward
  vs primary (grey), but nothing reaches the 0.95 line.
- Equity curves (`fig3_equity_curves.png`): in all three representative
  instruments the **primary bleeds steadily from costs** while the **meta stays
  near-flat** by vetoing the bad bets, the textbook "meta-labeling rescues a
  loser" picture, but rescuing it to ≈break-even, not to a deflated edge.

**Reading:** meta-labeling does *exactly what LdP says*, it raises precision,
flips a cost-losing primary toward break-even, and is a near-universal raw
improvement. But "raw improvement over a bad primary" and "survives deflation"
are different bars, and only the second one counts here.

---

## 5. Notes, opinions, extensions (honest)

- **Does meta-labeling add *deflated* value? On this primary, no.** It adds value
  in the LdP precision sense and in raw PF/Sharpe, but the absolute edge it
  rescues is too small to clear the multiple-testing hurdle. Meta-labeling is a
  **precision filter, not an alpha source**, it can only keep the better
  subset of bets the primary already proposes. If the primary's profitable bets
  carry no real edge net of costs (a vanilla EMA crossover on liquid markets
  does not), there is nothing for the filter to concentrate into a deflated win.
- **Where it's closest to working: equities.** Equities show the largest
  precision lift and the highest median meta DSR (0.095), because RTH dollar bars
  + 2 bp costs give the primary a fighting chance, and trend persistence in sector
  ETFs gives the meta-model something learnable. Still short of 0.95.
- **The SPY=6.47-PF / 0.76-DSR outlier is a small-sample trap, flagged honestly.**
  Selecting the trial by best out-of-fold Sharpe occasionally picks a config that
  acts on a handful of trades (SPY: 6 of 339 events). PF and DSR on 6 trades are
  not trustworthy. A production rule would add a **minimum-acted-trades floor**
  to the trial selection; we deliberately left the artifact visible rather than
  hiding it, and discount it in the conclusion.
- **PBO is informative.** Median meta PBO ≈ 0.2-0.4 (forex lowest at 0.19,
  crypto highest at 0.41). High crypto PBO says the best-IS trial often does not
  stay best OOS, consistent with the no-deflated-edge finding.
- **Extensions that would actually move the needle** (none of which are "tune
  harder"): (1) a primary with genuine edge to begin with (the meta-model is
  only as good as what it filters); (2) sample weighting by label uniqueness /
  return attribution (AFML Ch.4) so overlapping events don't over-count; (3)
  feature importance via MDA on the purged folds to prune noise features; (4)
  CPCV instead of single-path k-fold for a distribution of OOS DSRs; (5) a
  minimum-trade-count floor in trial selection.

---

## 6. Performance (profile → Numba → bit-identical check)

- **Profiling finding.** cProfile of a single instrument (BTCUSDT) shows the run
  is **dominated by the sklearn bagged-tree fits (~13.3 s of 15.4 s)**, *not* by
  the triple-barrier scan. LdP's prose flags the first-touch scan as the likely
  hot loop; we pre-empted it by writing the scan as a Numba `@njit` kernel up
  front, which pushed the scan cost below the noise floor and moved the
  bottleneck onto the ML, exactly where it should be.
- **Numba kernel.** `_triple_barrier_kernel` is a single forward pass over bars
  per event. Over a full 27-trial sweep on BTC dollar bars (715 events,
  max_hold=100) the kernel runs in **3.3 ms vs 68.3 ms** for the pure-Python /
  NumPy reference, **~21× faster**, and the gap widens with event count
  (forex/equities event sets are larger).
- **Bit-identical check.** `triple_barrier` (Numba) vs `triple_barrier_reference`
  (independent pure-Python) on BTC: **labels and exit-bar indices are exactly
  identical**, and `max|Δ ret_gross| = 1.78e-15` (floating-point non-associativity
  in the exp/log only; the touched barrier and exit price agree exactly). The
  kernel is trustworthy.

---

## 7. Limitations & exact rerun commands

- **Single-path CV.** We use purged k-fold (one OOS path). CPCV would give a
  distribution of OOS DSRs and a tighter PBO; deferred to keep the at-scale run
  under ~9 minutes.
- **Trial selection by OOS Sharpe** can pick low-trade-count configs (the SPY
  artifact). A trade-count floor is the obvious fix.
- **Primary is intentionally vanilla.** This study tests the *meta-labeling
  apparatus*, not a serious primary. The negative deflation result is about the
  apparatus's limits given a weak primary, not a claim that meta-labeling never
  helps.
- **Bar counts (~20k) are coarse for 19-year equity histories**; dollar-bar
  thresholds were set for cross-market comparability, not maximal resolution.
- **Reproduce:**
  ```bash
  cd projects/03_meta_labeling
  python3 scripts/run_meta_labeling.py            # full 42-instrument multi-market run (~8-9 min)
  python3 scripts/run_meta_labeling.py --smoke     # one instrument per market
  python3 scripts/run_meta_labeling.py --profile   # cProfile a single instrument (shows ML, not scan, is hot)
  ```
  Outputs: `tables/per_instrument.csv`, `tables/by_market_summary.csv`,
  `tables/results.md`, `tables/raw_results.parquet`, and
  `figures/fig{1..4}_*.png` (200 dpi).

---

## 8. Paper-worthiness

**Honest verdict: paper-worthy as a *negative / methodological* result, not as a
"meta-labeling makes money" result.** The clean, multi-market, costed, leakage-
controlled, DSR-gated finding, *meta-labeling reliably improves precision and
raw PF across 42 instruments in three asset classes, yet zero survive deflation
on a vanilla primary*, is a genuinely useful contribution, because the
literature over-reports the raw-PF improvement and under-reports that it can
evaporate under the False Strategy Theorem. The SPY small-sample artifact is a
nice cautionary vignette. To upgrade to a *positive* paper you would need to pair
the meta-layer with a primary that already has a deflated edge and show the meta-
layer *increases* the surviving DSR, that is the natural follow-up, and the
infrastructure here (Numba labeler + purged CV + DSR/PBO harness) is exactly what
that study would reuse.
