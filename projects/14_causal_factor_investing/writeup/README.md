# Causal Factor Investing, Association vs. Causation, Made Testable

*Reproduction and extension of López de Prado, "Causal Factor Investing" (2023) and
"Where Are the Factors?", the association-is-not-causation critique of the factor
literature.*

> **Status: FINAL.** Numbers below are from the full production run
> (`scripts/run_causal.py`, `n_sims=20000 × n_obs=2000` Monte Carlo; real
> cross-asset daily panel of 26 instruments, 2007-2026). Runtime ≈ 30 s
> (≈ 8 s with the daily panel cached). Every Monte-Carlo number is verified
> bit-identical (max |Δ| ≈ 1e-13) against an independent pure-NumPy reference.
> **This is a methodological / pedagogical contribution, not a money result**,
> the Monte Carlo reproduces LdP's structural-bias results exactly, and on real
> data essentially no factor survives correct adjustment plus deflation. We say so
> throughout.

---

## 1. Motivation

LdP's claim in *Causal Factor Investing* is blunt: the factor-investing literature
reports **associations**, cross-sectional and time-series regressions of returns
on candidate "factors", and tacitly reads them as **causes**, without ever stating,
let alone defending, the causal graph that would license that reading. Whether an
associational coefficient identifies a causal effect depends entirely on the
underlying structure, and the three elementary structures pull the naive regression
in *opposite* directions:

| Structure | Graph | Naive `Y~X` recovers… | Correct handling |
|---|---|---|---|
| **Fork / confounder** | `Z→X, Z→Y` | a **spurious** effect (`a·c`) when true `X→Y = 0` | **adjust for Z** (backdoor) → ~0 |
| **Chain / mediator** | `X→M→Y` | the correct **total** effect `b·d` | do **not** condition on `M` (over-control kills it) |
| **Collider** | `X→C←Y` | ~0 (correct) | conditioning on `C` **manufactures** a spurious link |

The backdoor criterion (Pearl) says: to identify `X→Y`, condition on a set that
blocks every back-door path and contains **no collider** (and no descendant of one).
Get the adjustment set wrong and a spurious "factor" looks significant; get it right
and a confounded one vanishes, or you *manufacture* one by conditioning on a
collider. This project turns that critique into a controlled simulation plus a
real-data demonstration, with a deflated-Sharpe headline and a falsification
scorecard.

## 2. Data

**Monte Carlo (part a).** Linear-Gaussian structural equation models; no market
data. Coefficients are LdP-style illustrative magnitudes: fork `a=c=1`, chain
`b=d=0.8` (true total effect `0.64`), collider `e=f=1`.

**Real panel (parts b, c).** Daily simple returns, **26 instruments across three
markets**, built from 1-minute source data resampled to daily closes and cached to
parquet:

| Market | Instruments | Coverage | Days |
|---|---|---|---|
| Crypto (11) | BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK, LTC, DOT, ATOM (USDT perp closes) | 2022-01 → 2026-05 | 1,441 |
| Equities (7) | SPY, QQQ, IWM, XLK, XLF, XLE, XLV (ETF 1-min, AlgoSeek) | 2007-01 → 2026-05 | 5,218 |
| Forex (8) | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP | 2022-01 → 2026-05 | 1,440 |

All factors are **causal**: each factor value at day *t* uses only information up to
and including *t−1* (an explicit `.shift(1)`), so nothing predicts return *t* with
future data. Real data only; no synthetic returns.

## 3. Method

### (a) Monte Carlo of the three structures
For each of `n_sims = 20,000` independent datasets (`n_obs = 2,000` each) we draw
fresh data from each SEM and compute the OLS coefficient on `X` and its *t*-stat for
(i) the **naive** simple regression `Y~X` and (ii) the **adjusted** regression
`Y~X+conditioning-var`, via Frisch-Waugh partialling (df = n−3). We record the
sampling distribution of the `X`-coefficient and the rejection rate at `|t|>1.96`.
We then frame each rejection rate decision-theoretically: for the fork and collider
the true effect is **0**, so a rejection is a **false positive** (Type-I error); for
the chain the true effect is **real**, so a *non*-rejection is a **false negative**
(power loss).

**Dose-response sweep (part a2).** A deepening: we sweep the structural-bias
strength (fork `a=c`, collider `e=f`, chain `b=d` over `[0, 1.2]`) and trace how the
naive vs corrected decision error responds, the quantitative heart of the critique.

### (b) Real cross-asset factors: naive vs backdoor
Three lagged factors, 21-day momentum, 21-day realized vol (a low-vol proxy),
5-day short reversal, go through a **pooled within-market panel regression** of
next-day return on the lagged, standardized factor:

- **Naive:** `r_{i,t} ~ factor_{i,t}`
- **Backdoor 1:** `r_{i,t} ~ factor_{i,t} + market_t` (market = cross-sectional mean
  return that day, the obvious common driver of both factor and return)
- **Backdoor 2 (deepening):** `+ lagged common realized vol`, a *second,
  independent* confounder (the market-wide vol regime, lagged one day so it stays
  causal). A factor is a candidate "survivor" only if it clears **both** backdoors.

For each cell we report the *t*-stat under each specification, whether the
significance **verdict flips**, the **kind** of flip (`spurious_killed` = a
significant naive result dies under adjustment; `suppression_revealed` = an
insignificant one becomes significant), the **|t| attenuation** (fraction of the raw
signal attributable to the confounder), and per-instrument sign consistency. We also
build the naive long-short sleeve (`sign(factor)·return`) per cell and deflate it.

### (c) Headline metric, Deflated Sharpe Ratio
The headline is the **DSR** (`lib/overfit.py`) on the best naive long-short factor
sleeve, treating the `market × factor` set (9 cells) as the trial count, with the
expected-max-Sharpe benchmark from the False Strategy Theorem.

### Engine note (honest)
Profiling found the MC's OLS reductions over `n_obs=2000` are large vectorized sums
that **NumPy/BLAS does better than scalar `@njit` loops** (Numba/NumPy ratio ~1.8×,
NumPy wins). So the production engine is **NumPy**; the Numba kernels are kept as a
**verified bit-identical cross-check** (max |Δ| ≈ 1e-13, float-reorder only) and
would win only for many *tiny* regressions. A fully-batched vectorized MC would be
fastest but its `n_sims × n_obs` arrays blow RAM (~2.4 GB/array) at full scale, so
the RAM-safe per-sim path is used. `--profile` reproduces this.

## 4. Results, Monte Carlo (part a)

The 20,000-sim production run reproduces LdP's structural-bias results exactly:

| Structure | True effect | Naive coef (reject rate) | Adjusted coef (reject rate) | Reading |
|---|---|---|---|---|
| **Fork** | 0.00 | **0.500 (100% reject)** | **0.0001 (5.0%)** | naive false-positive rate = **100%**; backdoor restores nominal α |
| **Chain** | 0.64 (total) | **0.640 (100% reject)** | **−0.0002 (5.1%)** | naive recovers the true total; over-controlling for `M` **collapses power to 5%** |
| **Collider** | 0.00 | **−0.0001 (4.9%)** | **−0.500 (100% reject)** | naive is correct; conditioning on `C` **manufactures** a 100%-significant factor |

The naive regression is *significant and wrong* in the fork, conditioning on the
mediator destroys a *real* chain effect, and conditioning on a collider *creates*
significance from nothing. (`figures/fig1_mc_structures.png`,
`tables/mc_structures.csv`.)

**Dose-response (part a2, `tables/mc_sweep.csv`, `figures/fig3_mc_dose_response.png`):**

| Bias strength | Fork naive FP | Fork backdoor FP | Collider naive FP | Collider-conditioned FP | Chain naive power | Chain over-control power |
|---|---|---|---|---|---|---|
| 0.00 | 5.0% | 4.9% | 4.6% | 4.7% | 5.0% | 5.1% |
| 0.15 | 16.7% | 4.9% | 4.7% | 16.1% | 16.8% | 4.9% |
| 0.30 | **95.8%** | 4.8% | 4.7% | **95.8%** | **96.9%** | 5.2% |
| 0.45 | 100% | 4.9% | 4.8% | 100% | 100% | 5.3% |
| ≥0.60 | 100% | ~4.8% | ~4.8% | 100% | 100% | ~5.1% |

The single quantitative claim that summarizes the entire critique: **the naive /
mis-conditioned estimator's error scales monotonically with the strength of the
(mis)handled structure, already 96% by strength 0.30, while the structurally
correct estimator holds its nominal 5% error (fork/collider) or recovers full power
(chain) at every dose.** Bias is not a knife-edge pathology; it is the generic
behavior of the wrong adjustment set.

## 5. Results, Real cross-asset factors (part b)

Pooled panel regressions, 26 instruments, lagged factors, no look-ahead
(`tables/real_factor_naive_vs_backdoor.csv`, `figures/fig2_real_factor_flip.png`):

| Market | Factor | n_obs | naive *t* | +market *t* | +lagged-vol *t* | flip | kind | |t| atten. |
|---|---|---|---|---|---|---|---|---|
| crypto | momentum | 15,620 | 2.26 | 2.11 | 1.73 |, |, | 7% |
| crypto | vol | 15,620 | 1.36 | 0.36 | −1.26 |, |, | 74% |
| crypto | revers | 15,620 | −0.23 | 0.66 | −0.39 |, |, |, |
| equities | momentum | 36,379 | **−6.39** | −1.60 | −6.26 | **YES** | spurious_killed | 75% |
| equities | vol | 36,379 | **2.14** | 0.53 | 1.65 | **YES** | spurious_killed | 75% |
| equities | revers | 36,379 | **−11.98** | **−4.88** | **−11.96** |, |, | 59% |
| forex | momentum | 11,352 | 0.02 | 0.50 | 0.01 |, |, |, |
| forex | vol | 11,352 | 0.76 | 0.18 | 0.51 |, |, | 76% |
| forex | revers | 11,352 | −1.84 | −1.71 | −1.84 |, |, | 7% |

- **4 / 9** cells are significant naively (`|t|>1.96`).
- The market-mean backdoor **flips 2 verdicts**, equities momentum and equities
  vol, and both flips are `spurious_killed`: the apparent factor was largely the
  market confounder leaking through. Median |t| attenuation across naively-significant
  cells is **67%**: roughly two-thirds of the raw factor signal is confounder.
- **Only 1 / 9 cells survives BOTH backdoors** (equities short-reversal). Even there
  the *t* drops from −12.0 to −4.9 (59% attenuated), and reversal is a microstructure
  / bid-ask-bounce effect, not a defended risk premium.
- Crypto vol *changes sign* (+0.36 → −1.26) between the two confounders, a clear
  sign it is conditioning artifact, not structure.

## 6. Results, Headline DSR & falsification checklist (parts b, c)

**Deflated Sharpe (headline, `tables/real_factor_dsr.csv`).** The best naive
long-short sleeve is equities low-vol (annualized SR ≈ 0.47). Against the
expected-max-Sharpe benchmark for 9 trials:

> **DSR = 0.455, far below the 0.95 bar.** The best-looking associational sleeve
> does not survive multiple-testing deflation. The "edge" is consistent with luck
> across 9 trials.

**Hierarchy-of-evidence scorecard (`tables/hierarchy_checklist.csv`)**, applied to
the program's own surviving signals, the highest level any candidate reaches is **3**:

| Level | Question | Verdict |
|---|---|---|
| L1 | Association exists? (raw |t|>1.96) | **PASS**, 4/9 cells |
| L2 | Survives the market-mean backdoor? | **PASS**, 2/9; 2 flips; 67% median attenuation |
| L3 | Survives a *second* independent confounder (lagged common vol)? | **PASS**, only **1/9** (equities reversal) clears both |
| L4 | Is the causal graph stated & **defended**? | **FAIL**, no defended SEM supplied for any survivor |
| L5 | Survives multiple-testing deflation (DSR>0.95)? | **FAIL**, best DSR 0.455 |
| L6 | Interventional / do-evidence? | **FAIL**, all evidence observational |
| L7 | True OOS / cross-market replication of the *causal* claim? | **FAIL**, no survivor replicates the same factor across all three markets |

The lone survivor of the conditioning tests (equities reversal) is killed at L5
(deflation) and never reaches L4-L7. **The chain of evidence breaks exactly where
LdP says it does: association is cheap, conditioning is fragile, and deflation plus a
*defended graph*, which the literature almost never supplies, is where the
"factors" disappear.**

## 7. Honest assessment & paper-worthiness

**Worthy as a *methodological / pedagogical* contribution, not a "factor makes
money" result**, and the writeup is framed that way throughout. What is genuinely
useful and, to our knowledge, not assembled this cleanly elsewhere:

1. A controlled MC that reproduces fork/chain/collider bias **and** shows the
   **dose-response** (error → 100% by strength 0.30 for the wrong adjustment set;
   nominal α / full power retained for the right one), with a **bit-identical**
   NumPy/Numba cross-check.
2. A **real, multi-market, no-look-ahead, two-confounder** demonstration that the
   significance verdict flips under correct adjustment (2/9), that ~⅔ of a typical
   raw factor signal is confounder (67% median attenuation), that only **1/9**
   survives a *second* independent confounder, and that **zero** survive deflation
   (best DSR 0.455).
3. A **quantitative falsification scorecard** that walks a candidate signal up the
   evidence hierarchy and pinpoints where each rung breaks.

**What it is not:** new theory (the SEM/backdoor results are standard Pearl/LdP), and
the real-data part is a *demonstration of the critique*, not a discovery of an edge.
To upgrade toward a *positive* paper one would need, for at least one factor, a
**defended causal graph** (L4) and **interventional / natural-experiment** evidence
(L6), which this study deliberately leaves open as the honest frontier. As is, it is
a strong illustrative / replication piece: suitable as a methods note, a teaching
appendix, or the empirical spine of a longer "why factor *t*-stats mislead" article.

## 8. Reproduce

```bash
cd projects/14_causal_factor_investing
bash run_full.sh                          # full run (~30 s; ~8 s with panel cached)
python3 scripts/run_causal.py --smoke      # tiny 1-core run (6 assets, 300 sims)
python3 scripts/run_causal.py --profile    # Numba-vs-NumPy timing + bit-identity
```

**Outputs**
- Tables: `tables/mc_structures.csv`, `tables/mc_sweep.csv`,
  `tables/real_factor_naive_vs_backdoor.csv`, `tables/real_factor_dsr.csv`,
  `tables/hierarchy_checklist.csv`, `tables/summary.json`
- Figures: `figures/fig1_mc_structures.png` (coefficient distributions),
  `figures/fig2_real_factor_flip.png` (naive vs backdoor *t* flip map),
  `figures/fig3_mc_dose_response.png` (decision error vs bias strength)
