# Project 0 — Backtest Overfitting & the Deflated Sharpe Ratio (the validation harness)

**LdP sources:** "The Probability of Backtest Overfitting" (2017); "The Deflated Sharpe Ratio"
(2014); "The False Strategy Theorem" (2021); "Pseudo-Mathematics and Financial Charlatanism" (2014);
"A Data Science Solution to the Multiple-Testing Crisis" (2019).
**Markets:** Crypto (8 perps), US Equities (7 ETFs), Forex (8 majors).
**Role:** This is the keystone — the metric and gate every later project's claims must pass.

---

## 0. Headline at scale — 50,000 real strategies (20 crypto pairs × 2,500)

The definitive run is on a **real, diverse, costed corpus**: 2,500 deep-WFO strategies from each
of 20 crypto pairs = **50,000 strategies** (the per-trade ledgers from the production framework;
already costed at fee 0.05 / slip 0.03 / funding 0.01). `tables/corpus_summary.md`,
`figures/fig5_corpus_sharpe_vs_null.png`, `fig6_corpus_pbo_effn.png`.

| metric | value |
|---|---|
| corpus **best** Sharpe (annualised) | **2.00** |
| E[max] under the null (False Strategy Theorem, N=50,000) | **2.72** |
| **Deflated Sharpe of the corpus best** | **0.029**  (≪ 0.95) |
| median per-pair PBO | 0.28 |
| **effective independent trials** | **434** of 50,000 |

Even across 50,000 real strategies, the single best — an annualised Sharpe of **2.0** — has a
**Deflated Sharpe of 0.03** and does **not** clear the multiple-testing null: the *expected* maximum
Sharpe of 50,000 skill-less trials (2.72) is *higher* than the best we actually observed. And the
50,000 nominal strategies are only **~434 effectively independent bets** (eigenvalue participation
ratio of the return-correlation matrix; the rest are correlated re-parameterizations). This is the
result that matters; the small MA-grid below (§3–4) is retained as the controlled illustration of the
same mechanism. Forex and equities at-scale runs follow (forex axis fix + framework runs on Algoseek).

RAM-safe at this scale: per-strategy Sharpe is streamed for the DSR/False-Strategy distribution; PBO
uses block-sums; **effective-N is computed exactly from the T×T Gram matrix** (the N×N correlation has
rank ≤ T, so its nonzero spectrum equals that of the small T×T matrix — no 50k×50k materialisation).
Run: `python3 scripts/run_corpus_overfit.py`.

---

## 1. What LdP claims

If you try many strategy configurations and keep the best, the winner's backtest Sharpe is
**inflated by selection** and tells you almost nothing. His quantitative apparatus:

- **False Strategy Theorem** — under *N* skill-less trials, the expected maximum Sharpe is
  `E[max] ≈ √V·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))` (γ = Euler–Mascheroni). So any target Sharpe
  is reachable by luck with enough trials.
- **Deflated Sharpe Ratio (DSR)** — the probability the strategy's true Sharpe exceeds the
  expected-max-under-the-null benchmark, correcting for #trials, their dispersion, skew and kurtosis.
- **Probability of Backtest Overfitting (PBO)** — via Combinatorially-Symmetric CV: the rate at which
  the best in-sample configuration lands in the bottom half out-of-sample.
- **Effective number of trials** — correlated trials are not independent; the *effective* count (here
  the eigenvalue participation ratio of the trial-return correlation matrix) is what feeds the null.

## 2. The harness (`lib/overfit.py`) and its verification

Implemented and self-tested (`python3 lib/overfit.py`):

- **False Strategy Theorem vs Monte Carlo** (skill-less trials): formula matches simulation to ~1e-3
  and converges — N=10: 0.0498 vs 0.0480; N=100: 0.0800 vs 0.0795; N=1000: 0.1029 vs 0.1030.
- **DSR separation:** a skill-less winner (best of 200 noise strategies) deflates to **DSR 0.32**;
  a genuine per-bar edge deflates to **DSR 0.87**. The statistic does its job.
- Also implements PSR, MinTRL, MinBTL, purged K-Fold + embargo, and CPCV splits (for later projects).

## 3. Method — overfitting at scale, multi-market

For each instrument we optimise a **dual moving-average crossover** over a grid of (fast, slow)
lookbacks (**136 trials** each), on **information-driven bars** (dollar bars for crypto & equities,
equities within regular hours; tick bars for forex — dogfooding Project 1), with **realistic
non-zero costs** applied to turnover (crypto 7 bp, equities 2 bp, forex 1 bp per unit). Each trial's
net per-bar return series feeds the harness: best-in-sample Sharpe, DSR, PBO (CSCV, 924 splits),
effective trials.

## 4. Result — the apparent edge is a multiple-testing artifact, in every market

`tables/overfit_summary.md` (median by market):

| market | best IS Sharpe (ann) | E[max] under null | DSR | PBO | nominal → effective trials |
|---|---|---|---|---|---|
| **crypto** | 0.62 | **0.66** | 0.44 | 0.70 | 136 → **3** |
| **equity** | 0.31 | **0.44** | 0.32 | 0.38 | 136 → **3** |
| **forex** | 0.53 | **1.06** | 0.10 | 0.37 | 136 → **3** |

Read it the way LdP would:

- **The best backtest Sharpe is at or below what pure luck produces.** In all three markets the
  best-of-136 in-sample Sharpe (0.31–0.62 ann.) sits *below* the False-Strategy-Theorem expectation
  for skill-less trials (0.44–1.06). The "edge" is fully explained by having tried 136 times.
  (`figures/fig2_maxsr_vs_null.png`.)
- **No strategy survives deflation.** Median DSR is 0.10–0.44 everywhere — far below the 0.95 bar.
  Not one optimised crossover is statistically significant after correcting for selection
  (`figures/fig3_dsr_pbo.png`). Forex is worst (DSR 0.10): FX trends are weak and the costs bite.
- **Overfitting is real, not hypothetical.** Crypto PBO median 0.70 — the best in-sample
  configuration is more likely than not to be a *below-median* performer out of sample. The
  in-sample/out-of-sample Sharpe scatter (`figures/fig1_is_vs_oos.png`) shows the best IS point is
  nowhere near the best OOS point.
- **136 trials ≈ 3 independent bets** (`figures/fig4_effective_trials.png`). The grid feels like 136
  tests but the strategies are ~55% pairwise-correlated; the effective number of independent trials
  is ~3. This is exactly why one must count *effective* trials, not nominal ones, in the null.

## 5. Notes, opinions & how this gates the program

- **DSR replaces raw OOS profit-factor / Sharpe as the headline metric** for the whole program. A
  result is only "real" if it clears DSR after honest trial-counting — the formal version of the
  median-|ρ| de-correlation discipline already used in the broader corpus work.
- **This project deliberately finds nothing** — and that *is* the result. A naive reader would have
  reported "we found a crypto MA strategy with Sharpe 1.19" (SOL). The harness shows it is one of
  136 lucky draws worth ~3 independent bets and does not survive. Every later project (labeling,
  features, portfolios) reports DSR, PBO and effective-N alongside any performance claim.
- **Opinion:** the cross-market consistency is striking — the multiple-testing illusion is not a
  crypto quirk; equities and forex are if anything *worse* (lower DSR). Cost level matters: forex's
  thin trends + spread make its deflated significance the lowest.

## 6. Performance (profile-first, then Numba)

Per the program's engineering rule, the heavy run was profiled before optimising:

- **Profiler finding:** 64% of runtime was the bar aggregation — specifically pandas iterating
  tz-aware timestamps as Python objects (6.3M `datetime.__iter__` calls).
- **Fix 1 — Numba kernel** for bar aggregation (`bars._agg_kernel`): single forward pass over
  contiguous group ids on int64-ns timestamps. **139× faster** (342 ms → 2 ms on 200k rows),
  timestamps bit-identical, value columns within 1e-5 absolute (~1e-11 relative; float
  summation-order only). Since bar-building is the universal hot path, this accelerates *every*
  project.
- **Fix 2 — vectorised PBO/CSCV:** precompute per-block (sum, sum-of-squares, count) once, then
  combine blocks combinatorially instead of re-stacking arrays for each of the 924 splits.

## 7. Limitations & reproducibility

- One structural family (MA crossover) and 136 trials per instrument — enough to demonstrate the
  effect; a real strategy search has thousands of (correlated) trials, which only *strengthens* the
  conclusion. The harness scales to any returns matrix.
- CSCV uses 12 blocks (924 combinations). DSR uses nominal N with the empirical trial-SR dispersion
  (conservative — using effective-N would *raise* DSR slightly; the "not significant" verdict holds
  either way).
- **Rerun:** `python3 lib/overfit.py` (self-tests), then `python3
  scripts/run_overfit_at_scale.py` (tables + figures). Data: crypto/forex 1m caches, Algoseek ETFs.

## 8. Paper-worthiness

Strong on its own as a **multi-market demonstration of the multiple-testing illusion** with a clean,
verified implementation of the DSR/PBO/False-Strategy apparatus — the kind of "here is why your
backtest is probably fake, across three asset classes" result that travels well. Its larger value is
infrastructural: it is the gate that lets the *other* projects make credible, deflated claims. Best
published as the methods backbone of the program, with the per-project DSR/PBO results as the payoff.
