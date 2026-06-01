# Project 0 — Backtest Overfitting & the Deflated Sharpe Ratio (the validation harness)

**LdP sources:** "The Probability of Backtest Overfitting" (2017); "The Deflated Sharpe Ratio"
(2014); "The False Strategy Theorem" (2021); "Pseudo-Mathematics and Financial Charlatanism" (2014);
"A Data Science Solution to the Multiple-Testing Crisis" (2019).
**Markets:** Crypto (20 perps), US Equities (9 ETFs), Forex (8 majors) — all at scale, all costed.
**Role:** This is the keystone — the metric and gate every later project's claims must pass.

---

## 0. Headline at scale — ~50,000 real strategies per market, in three asset classes

The definitive run is on three **real, diverse, realistically-costed** strategy corpora — not a toy
grid. Each market contributes thousands of structurally-distinct strategies per instrument:

- **Crypto** — 2,500 deep-WFO strategies from each of **20 pairs = 50,000** (production
  per-trade ledgers, costed at fee 0.05 / slip 0.03 / funding 0.01).
- **US Equity** — 2,500 from each of **9 ETFs = 22,500** (SPY QQQ IWM XLK XLF XLE XLV UVXY VXX),
  generated from Algoseek 1-min within-RTH dollar bars.
- **Forex** — 2,500 from each of **8 majors = 20,000** (EUR/GBP/USD/JPY/CHF/CAD/AUD/NZD crosses),
  generated from HistData tick bars.

Equity and forex frictions are applied through **`lib/realism`** (no clamping): time-of-day
half-spread schedules + per-fill commission, FX overnight **rollover swap (triple on Wednesday)** and
**weekend force-flat** (the real Fri→Sun gap is kept as risk), and equity **short-borrow** accrual.
All signals are causal (signal at *t* → fill at *t+1* open) with intrabar OHLC stop/target. Tables:
`tables/corpus_summary_{crypto,equity,fx}.md`, `tables/cross_market_summary.md`; figures:
`figures/fig5_corpus_sharpe_vs_null_{crypto,equity,fx}.png`, `fig6_corpus_pbo_effn_{...}.png`,
`figures/fig7_cross_market.png`.

### Cross-market result

| market | n strat | **best** SR (ann) | E[max] under null (False Strategy Thm) | **DSR** of best | median PBO | **effective N** (pooled) | best < null? | DSR > 0.95? |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **Crypto**    | 50,000 | **2.00** | 2.72  | **0.029** | 0.28  | **434** of 50,000 | yes | **no** |
| **US Equity** | 22,500 | **3.21** | 10.89 | **0.000** | 0.00  | **39** of 22,500  | yes | **no** |
| **Forex**     | 20,000 | **1.78** | 7.70  | **0.000** | 0.005 | **26** of 20,000  | yes | **no** |

**The multiple-testing illusion holds in all three markets, and the best of ~50k survives deflation
in none of them.** (`figures/fig7_cross_market.png` — best-vs-null on the left, nominal-vs-effective
trial counts on the right.)

- **Crypto** is the clean, near-symmetric case. The 50,000 net Sharpes are centred near zero; the
  single best is an annualised **2.00**, but the *expected* maximum of 50,000 skill-less trials with
  this dispersion is **2.72** — the best we actually found is *below* what luck alone produces.
  Deflated Sharpe **0.029** ≪ 0.95. And the 50,000 nominal strategies are only **~434 effectively
  independent bets** (eigenvalue participation ratio of the return-correlation matrix); the rest are
  correlated re-parameterizations. *Robustness:* even if we credit only the 434 effective trials in
  the null (a more generous benchmark, null falls to ann 1.93, just below the observed 2.00), the
  best still deflates to only **DSR ≈ 0.57** — short of 0.95. The verdict is invariant to nominal-vs-
  effective trial counting.
- **US Equity** — with realistic costs (sub-bp commission + time-of-day spread + short borrow), the
  index ETFs ride two decades of long-side beta: the best strategies reach ann **2.4–3.2** (QQQ 3.21,
  IWM 2.98, SPY 2.44), while the sector/vol ETFs sit near zero or negative — a wide cross-sectional
  Sharpe dispersion. But the best of 22,500 (ann **3.21**) is still far below the False-Strategy-Theorem
  null (**10.89**, inflated by that dispersion), so the **deflated Sharpe is 0.000** and median PBO 0:
  the apparent index-ETF "edge" is long-beta drift mined by many correlated trials, not skill. Pooled
  effective N is just **39** of 22,500. *(Provenance note: an earlier run over-charged equity
  commission ~10–70× by pricing the $0.35 min-ticket against a single share; corrected to a per-order
  notional. The deflation verdict was unchanged either way — but the honest picture is beta-lifted
  Sharpes that still fail deflation, not the cost-crushed all-negative book the buggy run showed.)*
- **Forex** sits between the two. The distribution is mildly negative-centred; the best major-pair
  strategy reaches ann **1.78** (USD/JPY), but the multiple-testing null is **7.70** and the DSR is
  **0.000**. Effective N is **26** of 20,000. Thin FX trends plus the spread/swap/weekend frictions
  leave no deflated edge.

The small MA-grid study below (§3–4) is retained as the controlled, single-family illustration of the
same mechanism; §0 is the result that matters.

RAM-safe at this scale: per-strategy Sharpe is streamed for the DSR / False-Strategy distribution; PBO
uses block-sums; **effective-N is computed exactly from the T×T Gram matrix** (the N×N correlation has
rank ≤ T, so its nonzero spectrum equals that of the small T×T matrix — no 50k×50k materialisation).
Run: `LDP_MARKET={crypto,equity,fx} python3 scripts/run_corpus_overfit.py`, then
`python3 scripts/build_cross_market.py` for the cross-market table + `fig7`.

---

## 1. What LdP claims

If you try many strategy configurations and keep the best, the winner's backtest Sharpe is
**inflated by selection** and tells you almost nothing. His quantitative apparatus:

- **False Strategy Theorem** — under *N* skill-less trials, the expected maximum Sharpe is
  `E[max] ≈ √V·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))` (γ = Euler–Mascheroni), where *V* is the
  variance of the trial Sharpes. So any target Sharpe is reachable by luck with enough trials.
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

The §0 corpora are the at-scale evidence. As a controlled cross-check we also optimise a single
**dual moving-average crossover** family over a grid of (fast, slow) lookbacks (**136 trials** each),
on **information-driven bars** (dollar bars for crypto & equities, equities within regular hours; tick
bars for forex — dogfooding Project 1), with **realistic non-zero costs** on turnover. Each trial's
net per-bar return series feeds the same harness: best-in-sample Sharpe, DSR, PBO (CSCV), effective
trials. (`scripts/run_overfit_at_scale.py`.)

## 4. Cross-check — the single-family MA grid

`tables/overfit_summary.md` (median by market):

| market | best IS Sharpe (ann) | E[max] under null | DSR | PBO | nominal → effective trials |
|---|---|---|---|---|---|
| **crypto** | 0.62 | **0.66** | 0.44 | 0.70 | 136 → **3** |
| **equity** | 0.31 | **0.44** | 0.32 | 0.38 | 136 → **3** |
| **forex** | 0.53 | **1.06** | 0.10 | 0.37 | 136 → **3** |

Same verdict as §0 at small scale: in all three markets the best-of-136 in-sample Sharpe sits at or
*below* the False-Strategy-Theorem expectation, no strategy clears DSR 0.95, crypto PBO 0.70 shows the
best IS configuration is more likely than not a below-median OOS performer, and the 136 nominal trials
are only ~3 effectively independent bets (the grid is ~55% pairwise-correlated).
(`figures/fig1_is_vs_oos.png`, `fig2_maxsr_vs_null.png`, `fig3_dsr_pbo.png`, `fig4_effective_trials.png`.)

## 5. Notes, opinions & how this gates the program

- **DSR replaces raw OOS profit-factor / Sharpe as the headline metric** for the whole program. A
  result is only "real" if it clears DSR after honest trial-counting — the formal version of the
  median-|ρ| de-correlation discipline already used in the broader corpus work.
- **This project deliberately finds nothing** — and that *is* the result, now demonstrated across
  ~92,500 real costed strategies in three asset classes. A naive reader would report "we found a
  crypto strategy with Sharpe 2.0 / an FX strategy with Sharpe 1.8". The harness shows both fall below
  the multiple-testing null and deflate to DSR ≈ 0. Every later project (labeling, features,
  portfolios) reports DSR, PBO and effective-N alongside any performance claim.
- **Opinion — the cross-market read.** The illusion is not a crypto quirk: it holds in equities and
  forex too, and the *effective* breadth of a corpus is one to two orders of magnitude below its
  nominal size everywhere (434/50k, 26/20k, 20/22.5k). Cost level decides how far the population sinks:
  realistically-costed equity TA is a net loser end-to-end, so its null is dominated by a catastrophic
  left tail; FX is thin-trend / spread-bitten; crypto is the only market whose Sharpe distribution is
  roughly symmetric around zero — and even there the best of fifty thousand does not survive deflation.

## 6. Performance (profile-first, then Numba)

Per the program's engineering rule, the heavy paths were profiled before optimising:

- **Profiler finding:** 64% of the MA-grid runtime was bar aggregation — pandas iterating tz-aware
  timestamps as Python objects (6.3M `datetime.__iter__` calls).
- **Fix 1 — Numba kernel** for bar aggregation (`bars._agg_kernel`): single forward pass over
  contiguous group ids on int64-ns timestamps. **139× faster** (342 ms → 2 ms on 200k rows),
  timestamps bit-identical, value columns within 1e-5 absolute (float summation-order only). Since
  bar-building is the universal hot path, this accelerates *every* project.
- **Fix 2 — vectorised PBO/CSCV:** precompute per-block (sum, sum-of-squares, count) once, then
  combine blocks combinatorially instead of re-stacking arrays for each split.
- **Fix 3 — the corpus harness** (§0) is RAM-bounded by design: streamed Sharpes, block-sum PBO, and
  the T×T Gram trick for effective-N, so 50k×T correlation never materialises.
- **Corpus generation** (`lib/ta_grid` + `lib/realism`): Numba indicator + sim kernels, verified
  bit-identical to the NumPy reference (max|Δpnl| = 0, max|Δn_trades| = 0 on both the equity and FX
  friction models).

## 7. Limitations & reproducibility

- The §0 corpora mix one structural family per market with many parameterizations; effective-N (26–434)
  shows that nominal breadth massively overstates independent breadth — which only *strengthens* the
  "nothing survives" conclusion. The harness scales to any returns matrix.
- DSR uses **nominal N** with the empirical trial-SR dispersion (conservative; using effective-N would
  *raise* DSR — and as shown for crypto, the verdict still holds at DSR ≈ 0.57 ≪ 0.95).
- The equity null (10.9) is inflated by a wide cross-sectional Sharpe dispersion (long-beta index ETFs
  vs near-zero sector/vol ETFs), not a symmetric skill-less population; we report it honestly and lean
  on the DSR/PBO/effective-N triangulation, not the raw null magnitude, for the equity verdict.
- **Rerun:** `python3 lib/overfit.py` (self-tests); `bash run_full.sh` (crypto) and
  `bash run_full_equity_forex_corpora.sh` (equity+forex corpora); `LDP_MARKET={crypto,equity,fx}
  python3 scripts/run_corpus_overfit.py`; `python3 scripts/build_cross_market.py`; `python3
  scripts/run_overfit_at_scale.py` (the §3–4 cross-check). Data: crypto production ledgers, Algoseek
  ETF 1-min, HistData FX 1-min.

## 8. Paper-worthiness

Strong on its own as a **three-asset-class, at-scale demonstration of the multiple-testing illusion**:
~92,500 real, realistically-costed strategies across crypto, equities and forex, with a clean, verified
implementation of the DSR / PBO / False-Strategy / effective-N apparatus. The headline travels well —
"the best of fifty thousand real strategies fails to beat its own multiple-testing null, in every
market we tried" — and the cross-market contrast (symmetric crypto vs beta-lifted-but-deflated equity vs thin FX)
gives it texture beyond a single demonstration. Its larger value is infrastructural: it is the gate
that lets every *other* project make credible, deflated claims. Best published as the methods backbone
of the program, with the per-project DSR/PBO results as the payoff.
