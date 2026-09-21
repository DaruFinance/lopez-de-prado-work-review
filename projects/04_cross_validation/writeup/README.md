# Project 04, Cross-Validation in Finance: Purged K-Fold, Embargo & CPCV vs leaky k-fold

> Article: [daru.finance/research-review/lopez-de-prado/labeling-and-cross-validation](https://daru.finance/research-review/lopez-de-prado/labeling-and-cross-validation)

**LdP sources:** *Advances in Financial Machine Learning* (2018), Ch. 7 ("Cross-Validation in
Finance") and Ch. 12 ("Backtesting through Cross-Validation" / CPCV).
**Markets:** Crypto (8 perps), US Equities (7 ETFs), Forex (8 majors), dollar bars for crypto &
equities, tick bars for forex (dogfooding Project 1).
**Role:** the practical companion to Project 0's overfitting gate, *how* you cross-validate a
financial ML model without leaking, and how to read a backtest as one draw from a distribution.

---

## 1. What LdP proposed

Standard k-fold cross-validation assumes observations are IID. Financial labels are not: a label is
built from a *window of future bars*, a fixed-horizon return over the next `H` bars, or a
triple-barrier label with a maximum holding of `H` bars. Two labels whose windows overlap therefore
**share information**. When ordinary k-fold drops a test fold in the middle of the series, the train
points immediately before and after that fold have label windows that reach *into* the test fold.
The model is, in effect, trained on the answers, and the cross-validated score comes out
**optimistically biased**. LdP's fixes:

- **Purging**, remove from the training set every observation whose label window overlaps the test
  fold. (`label_span = H`.)
- **Embargo**, additionally drop a small band of train observations *immediately after* the test
  fold, to kill residual serial correlation that purging alone misses (features are autocorrelated
  even when label windows don't literally overlap).
- **Combinatorial Purged CV (CPCV)**, instead of one train/test partition, hold out *every*
  combination of `k_test` of `n_groups` blocks (with purging + embargo). This produces a whole
  **distribution of out-of-sample path scores**; a single backtest is just one draw from it.

The claim we reproduce: **(a)** standard k-fold leaks and inflates the CV score; **(b)** the
inflation **grows with the label overlap `H`** and **shrinks with the embargo**; **(c)** CPCV
recovers the full OOS-score distribution that a single split hides.

## 2. Method & data

For each instrument we build information-driven bars (≈8 bars/day; crypto/equity dollar bars,
forex tick bars), then:

1. **Overlapping labels (the leak source).** `label[t] = 1` if the `H`-bar-ahead log return is
   positive, else `0`. Consecutive labels share `H−1` of the same future bars ⇒ overlap span = `H`.
   The label is the *only* forward-looking object, exactly what CV must protect.
2. **Causal features.** Trailing momentum (sums of past returns over 1/3/6/12/24 bars), trailing
   volatility (6/24/48), a trailing log-volume z-score, and a fractional-difference of log price
   (`d = 0.4`, from `lib/fracdiff.py`), every feature at bar `t` uses only bars `≤ t`.
3. **Model.** A bagged-tree classifier per house convention, `RandomForestClassifier`
   (200 trees, `max_depth=5`, `min_samples_leaf=50`, `max_features="sqrt"`). The regularization is
   deliberate: a sensibly-regularized model produces the *clean, monotonic* leakage signal, whereas
   an unconstrained tree overfits noise in **both** folds and buries the leak in variance (we
   verified this in a config sweep).
4. **Three scorings of the same model:** (a) standard `sklearn.KFold` (non-shuffled, contiguous
   time folds), leaky; (b) `lib.overfit.purged_kfold_splits(..., label_span=H)` + embargo, clean;
   (c) `lib.overfit.cpcv_splits(...)`, the OOS-path distribution. Metrics: accuracy, AUC,
   neg-log-loss. **Inflation = leaky − purged**.

Statistical-property study, so it is **cost-free by nature** (no strategy P&L is claimed; we measure
CV *score bias*, which costs don't affect).

## 3. Reproduction, the leakage is real, and AUC is the cleaner detector

Standard k-fold scores the same model **higher** than purged+embargo, and the effect is visible in
the **AUC**, which uses the full predicted probability rather than a thresholded 0/1 label. Accuracy
near a 50 % base rate is a coarse, high-variance metric whose fold-to-fold noise (±0.5-1 pp) swamps
the small leak, across the cross-section its inflation averages to noise (crypto −0.17 pp, forex
−0.05 pp, equity +0.10 pp). AUC is the clean detector. Mean inflation by market at `H = 50`
(`tables/cv_by_market.md`, `figures/fig1_kfold_vs_purged_by_market.png`):

- **Crypto:** **AUC +0.24 pp** (6 of 8 perps positive); accuracy −0.17 pp (noise).
- **Forex:** **AUC +0.23 pp** (5 of 8 majors positive); accuracy −0.05 pp (noise).
- **Equities:** **AUC −0.24 pp**, i.e. no detectable leak at `H = 50` (3 of 7 positive). See §4 for
  why (the `H/fold-size` law) and the overlap sweep for how far you must push `H` on a long equity
  history before the boundary band matters.

So the directionally-correct leak (leaky > clean) shows up cleanly in AUC for the two short-history
markets and is absent in equities, a result the next section makes precise.

## 4. At-scale multi-market result, inflation scales with overlap, not just H

The headline mechanism, *more shared future bars ⇒ more leakage*, reproduces cleanly, but with an
important refinement LdP states implicitly and we make explicit:

> **Leakage magnitude tracks the ratio `H / fold-size`, not `H` alone.**

A test fold of `n/k` observations is contaminated only at its two boundaries, over a band of width
`~H`. The *fraction* of the fold that is contaminated is `~2H/(n/k)`. So:

- **Crypto & forex** (~8.4k bars, fold ≈ 1.4k): `H = 50` is ≈ 3.6 % of a fold, enough to leak, and
  the **AUC inflation rises with `H`**: crypto 0.03 → 0.24 → 0.53 → **1.69 pp** and forex
  0.12 → 0.45 → **1.07 pp** (at H = 100) as `H` goes 1 → 5 → 50 → 100/150
  (`figures/fig2_inflation_vs_overlap.png`, AUC panel).
- **Equities** (~42k bars, fold ≈ 7k): even `H = 150` is only ≈ 2 % of a fold, so the AUC inflation
  stays flat/slightly-negative across the whole range (≤ |0.5| pp), never taking off. Equities are
  not immune; their long histories simply keep the contaminated boundary fraction tiny. Same
  `H/fold-size` law.

**AUC inflation vs label overlap `H`** (mean, pp, the clean signal; from `tables/cv_by_market.md`):

| market |    1 |    5 |    10 |    25 |    50 |   100 |   150 |
|:-------|-----:|-----:|------:|------:|------:|------:|------:|
| crypto | 0.03 | 0.24 |  0.19 |  0.21 |  0.53 |  0.36 |  1.69 |
| equity | 0.02 | 0.04 | -0.01 | -0.09 | -0.47 | -0.46 | -0.03 |
| forex  | 0.12 | 0.17 | -0.00 |  0.27 |  0.45 |  1.07 |  0.85 |

(Accuracy inflation over the same grid is dominated by fold-noise and shows no clean trend, see the
left panel of Fig 2 and the noisy-accuracy table in `cv_by_market.md`.)

**Embargo, an honest null result.** The textbook expectation is that the embargo trims the
*residual* serial-correlation leak. But in our design the purged k-fold *already* purges the full
label-window overlap (`label_span = H`), so there is little residual leak for the embargo to remove.
Sweeping the embargo from 0 % to 10 % at fixed `H = 50` does **not** drive the residual AUC inflation
toward zero, it stays flat-to-rising for crypto/forex and flat-negative for equities
(`figures/fig3_inflation_vs_embargo.png`). The reason is a confound we report rather than hide: a
larger embargo *deletes training observations*, which makes the purged model slightly noisier/worse,
which *widens* the (already small) leaky-minus-purged gap. AUC residual inflation by embargo (pp):

| market |   0.0 | 0.005 |  0.01 |  0.02 |  0.05 |   0.1 |
|:-------|------:|------:|------:|------:|------:|------:|
| crypto |  0.44 |  1.02 |  0.64 |  0.63 |  1.35 |  0.51 |
| equity | -0.07 | -0.06 | -0.33 | -0.29 | -0.38 | -0.42 |
| forex  |  0.11 |  0.09 |  0.32 |  0.70 |  0.89 |  0.87 |

The practical lesson: once labels are correctly purged, the embargo is a small, second-order safeguard
against feature autocorrelation, not a free lunch, and over-embargoing costs you data. (LdP himself
recommends a *small* embargo, ~1 %; this is consistent with that.)

**CPCV, a backtest is one draw.** For BTCUSDT we ran CPCV (`C(6,2) = 15` OOS paths). The
OOS-accuracy distribution spans **15.1 pp** (0.381 → 0.532; AUC spread 17.1 pp), roughly **50×
wider** than the ~0.2-0.5 pp leakage we just corrected (`figures/fig4_cpcv_distribution.png`,
`tables/cpcv_paths_BTCUSDT.csv`). The single-split "backtest" (acc 0.478) sits at one arbitrary point
inside that cloud, above the CPCV mean (0.467). The honest reading: even after you fix the leak, *one*
number from *one* split is nearly meaningless, path-to-path dispersion dwarfs the leakage bias by
1-2 orders of magnitude. This is CPCV's whole point, and arguably the single most important figure
in the project.

## 5. Notes, opinions & extensions (honest)

- **The leakage is real but small in these designs (sub-2 pp even at large overlap), and that is
  itself the finding.** A naive reader expects "k-fold leaks" to mean huge gaps. In practice, with a
  *sensible* model, the bias is a few tenths of a pp at typical horizons, rising to ~1-1.7 pp only at
  large overlap, but it is *systematically* positive in the sensitive metric (AUC) for crypto and
  forex and it grows with overlap, as theory says. The danger of leaky CV here is not a wildly wrong
  score; it is a *subtly optimistic, consistently biased* one that survives naive sanity checks.
- **Use AUC, not accuracy, to detect leakage.** Accuracy near a 50 % base rate is too coarse, its
  fold-to-fold variance (±0.5-1 pp) swamps the leak, so across the cross-section accuracy inflation
  averages to noise (even slightly negative) while AUC is cleanly positive. Opinion: leakage audits
  should always be run on a smooth, probability-based metric.
- **The `H / fold-size` law is the practical takeaway.** Whether you *need* purging depends on how
  long your label horizon is relative to your fold. High-frequency labels on long equity histories
  barely leak (equity AUC inflation ≤ |0.5| pp even at H = 150); multi-day-equivalent labels on short
  crypto histories leak materially (up to 1.69 pp). Purge always (it is free and correct), but expect
  the *size* of the correction to scale with `H/fold-size`.
- **Embargo is a second-order safeguard, not a lever** (§4): once labels are purged, widening the
  embargo mostly deletes training data; it does not monotonically shrink the residual gap. Keep it
  small (~1 %).
- **CPCV dispersion ≫ leakage bias.** The most important number here is not the ~0.3 pp leak, it is
  the **15 pp** width of the CPCV OOS distribution. Reporting a single OOS score (purged or not)
  without its CPCV dispersion is the larger sin by a wide margin.
- **Extensions:** triple-barrier labels (overlap = realized holding, not a fixed `H`) with
  sample-uniqueness weights (AFML Ch. 4); a CPCV-based Sharpe distribution on an actual trading rule
  (feeds Project 0's Deflated Sharpe with a *measured* trial dispersion); and a meta-labeling layer
  (Project on Ch. 3) cross-validated with these same splits.

## 5b. Correction: meta-labeling a primary that already has an edge

The companion meta-labeling reproduction (project `03_meta_labeling`) concluded
"a precision filter, not alpha", but that conclusion was measured on an
**edgeless primary**: a plain EMA crossover with no established edge net of
costs. That is precisely the case Lopez de Prado warns against. His meta-labeling
carries a stated **precondition**, the primary must *already have an edge* (high
recall, mediocre precision), and the secondary's job is to raise precision by
vetoing its worst bets. A filter can only concentrate edge that already exists;
on a non-edge there is nothing to concentrate. So the blanket null is a statement
about the *primary*, not about the method.

Project `03b_metalabel_real_primary` re-runs the identical apparatus (triple-barrier
outcomes, a tree secondary on causal features, purged walk-forward, per-fill costs,
DSR as the headline) with the one thing that mattered changed: the primary now
carries a **proven structural order-flow / open-interest edge**, reused read-only
through its verified engine.

**With the precondition met, the secondary amplifies the edge.** On a focused
two-pair sample, profit factor rises from **1.26 to 1.79** and mean per-trade P&L
from **47 to 148 bp** out-of-sample, net of per-fill costs. The gate keeps 64% of
the edge's bets and vetoes the low-confidence tail, lifting the profitable-bet rate
from 53.0% to 63.6%, exactly the precision lift the textbook describes, now acting
on bets that carry edge so it shows up in P&L rather than only in a confusion
matrix. The deflated metric moves the right way too, DSR rises from **0.64 to
0.78**, though it still sits under the program's 0.95 publication bar. The lift
holds on both pairs individually (PF 1.32 to 1.49 and 1.23 to 1.96).

| arm (OOS, 2-pair) | trades | PF | per-trade bp | annual SR | DSR |
|:------------------|-------:|---:|-------------:|----------:|----:|
| primary (edge alone) | 166 | 1.26 | 47.3 | 10.3 | 0.636 |
| meta (gate + size by p) | 107 | 1.79 | 148.1 | 20.5 | 0.778 |

**But it washes out at scale.** Running the same precondition-met meta-gate across
**6 closed structural edges by 26 perp pairs** (154 of 156 configurations produced
trades) and counting the trial family honestly, the lift largely washes out. Median
profit factor moves only 0.967 to 0.986 (median lift +0.02, against the two-pair
sample's +0.53) and median per-trade P&L moves -5.9 to -1.2 bp (median lift +4.7 bp,
against +100 bp). And nothing clears deflation at scale: only **1 of 154** meta-gated
configurations clears DSR>0.95 (best single sleeve at meta DSR 0.983); the median
meta DSR across sleeves is just 0.1. Pooling the gated sleeves into one
weakly-correlated meta-strategy, deflated against the dispersion of the 156
configurations actually searched, gives a pooled **DSR of 0.0** (benchmark 0.35,
pooled PF 0.98, 14,725 out-of-sample trades).

**Honest synthesis.** The blanket "not alpha" claim was a statement about the
edgeless primary, not about meta-labeling. With the precondition met the mechanism
is real and directional, on a focused sample the secondary genuinely amplifies a
primary that already has an edge, roughly tripling per-trade P&L and lifting profit
factor from 1.26 to 1.79. Yet it does **not** survive scale plus deflation as a
tradeable edge: the focused magnitude does not generalise across 156 configurations,
and the pooled meta-strategy deflates to zero. The corrected reading is the one Lopez
de Prado actually makes, with the precondition restored to the front of the sentence:
*meta-labeling improves an edge you already have; it cannot manufacture one you do
not*, and improving an edge in direction is not the same as clearing the deflation
bar at scale. Full detail in `projects/03b_metalabel_real_primary`.

## 6. Performance (profile-first, then Numba / justified)

Per the program's engineering rule, a single-instrument smoke test was profiled with cProfile
(`python3 scripts/run_cv_leakage.py --profile`) **before** any optimization:

- **Profiler finding:** `RandomForestClassifier.fit` (via joblib) is **>90 %** of wall time
  (6.5 s of 7.7 s for two full CV passes on one crypto instrument). Label construction, feature
  building, and the purge/embargo index logic are collectively **<2 %**.
- **Decision (house rule):** *do not Numba sklearn.* The hot path is model fitting, not a Python
  loop. Instead we (i) keep the model regularized and the fold count modest (`k = 6`), (ii) restrict
  the `H`- and embargo-sweeps to a representative subset of instruments, and (iii) use `n_jobs=-1`.
- **The one non-sklearn hot loop, label construction, is moved to a Numba `@njit` kernel**
  (`fixed_horizon_label_nb`) and **verified bit-identical** against a pure-python reference
  (`fixed_horizon_label_py`) over randomized series and `H ∈ {1,5,20,80}`:
  **`max|Δ| = 0.000e+00` (BIT-IDENTICAL)**, NaN masks equal. Run `--verify` to reproduce.
- Bar construction reuses the shared Numba kernel (`lib/bars._agg_kernel`, 139× over pandas).

## 7. Limitations & exact rerun commands

- One label family (fixed-horizon sign); triple-barrier (variable, P&L-driven overlap) is the
  natural next design and is noted as an extension.
- Single model family (RandomForest); a different learner would shift absolute scores but not the
  *direction* of the leak (it is a property of the splits, not the model).
- Sweeps use a representative instrument subset per market (4 for overlap, 3 for embargo) to keep
  RF cost sane, the per-instrument table covers all **23 instruments** (8 crypto, 7 equity, 8 fx).
- Equity leakage stays ~0 **by design** of their long histories (the `H/fold-size` law), not because
  equities are immune; the overlap sweep pushes `H` to 150 (still only ~2 % of an equity fold).
- **Rerun** (full run ≈ 21 min on this box, single process; RF-fit-bound):
  - `python3 scripts/run_cv_leakage.py --verify`, bit-identical label-kernel check (`max|Δ|=0`).
  - `python3 scripts/run_cv_leakage.py --profile`, cProfile smoke test (shows RF dominates).
  - `python3 scripts/run_cv_leakage.py`, full multi-market run (6 tables + 4 figures).
  - `python3 scripts/run_cv_leakage.py --quick`, 2 instruments/market smoke run.
  Data: crypto/forex 1m caches, Algoseek ETF 1-min. Idempotent (overwrites its own tables/figures).

## 8. Paper-worthiness

Solid as a **multi-market, quantified reproduction** of AFML Ch. 7 & 12 with two contributions
beyond the textbook: (i) the explicit **`H / fold-size` scaling law** for leakage magnitude, which
reconciles "k-fold leaks badly" (short series, long horizon) with "I cross-validated and saw no
leak" (long series, short horizon); and (ii) the empirical demonstration that **AUC, not accuracy,
is the metric that reveals the leak**, with the leak buried in accuracy's variance. The CPCV figure
, leakage bias (~0.3 pp) dwarfed by path dispersion (~15 pp, ~50×), is a clean, quotable picture of
why single-split backtests are fragile. On its own it is a strong methods note / blog-grade result; its
larger value is as the cross-validation discipline the rest of the program's ML projects (labeling,
meta-labeling, feature importance) cite and reuse.
