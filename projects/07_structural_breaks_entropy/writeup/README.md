# Project 7, Structural Breaks & Entropy Features

> Article: [daru.finance/research-review/lopez-de-prado/predictive-features](https://daru.finance/research-review/lopez-de-prado/predictive-features)

López de Prado, *Advances in Financial Machine Learning*, **Ch. 17 (Structural
Breaks)** and **Ch. 18 (Entropy Features)**, implemented multi-market (crypto +
US-equity ETFs + forex), with **causal-only** features, a **realistically
costed** predictive test under **purged cross-validation**, and a **Deflated
Sharpe Ratio** headline. The honest verdict is stated up front: **as standalone
predictive features these structural-break and entropy estimators carry
essentially no costed, overfitting-adjusted edge.** The one robust *descriptive*
finding, explosive (SADF>0) regimes precede higher forward returns in crypto
and equities, does **not** survive as a tradeable, DSR-gated rule.

---

## 1. Objective & scope

LdP Ch.17-18 propose two feature families:

* **Structural breaks (Ch.17)**, detect regime shifts: the **CUSUM** event
  sampler and the **SADF** (Supremum Augmented Dickey-Fuller) explosiveness /
  bubble statistic.
* **Entropy (Ch.18)**, quantify the information content / predictability of the
  return string: **Shannon** plug-in entropy, **Lempel-Ziv (LZ76)** complexity,
  and the **Kontoyiannis** match-length entropy-rate estimator.

The textbook presents these as *features*; it does not claim they are profitable
alone. This project asks the empirical question LdP leaves open: **do they carry
predictive content once you (a) keep them strictly causal, (b) charge realistic
costs, (c) cross-validate with purging+embargo, and (d) deflate the headline for
multiple testing?** We test across three asset classes so the answer is not a
single-market artifact.

**Universe (27 instruments).**

| Market | Count | Instruments | Source |
|---|---|---|---|
| crypto | 12 | BTC ETH SOL XRP DOGE BNB LINK AVAX LTC ATOM BCH DOT (USD-M perps) | Binance 1m, full history |
| equities | 7 | SPY QQQ IWM XLK XLF XLE XLV | Algoseek 1m RTH |
| forex | 8 | EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURGBP | HistData 1m tick-bars |

Per instrument we build ~6,000 matched **dollar bars** (LdP's preferred
information-driven bar; `lib.bars.matched_bars`), take causal close-to-close log
returns, and compute the features below, all on backward windows.

---

## 2. Methodology

**Features (all causal, value at bar `t` uses only bars ≤ `t`).**

* **CUSUM event sampler**, symmetric cumulative-sum filter on log-price; emits
  an event when an up/down run exceeds `k · rolling σ` (k=1, σ over 100 bars).
* **SADF**, for each end-bar, fit the ADF regression
  `Δp_t = a + b·p_{t-1} + Σγ_j Δp_{t-j} + ε` for every admissible start in a
  *capped backward window* (`minlen=60`, `maxwin=250`, `lags=1`) and take the
  supremum of the t-stat on `b`. SADF≫0 ⇒ explosive/bubble regime. This is the
  heavy hot loop (many OLS fits/bar), **Numba `parallel=True`**.
* **Entropy** over a causal binary sign-string of returns in a 100-bar rolling
  window: **Shannon** plug-in rate, **LZ76** normalised complexity, **Kontoyiannis**
  match-length rate. The LZ phrase-count and Kontoyiannis match-length loops are
  **Numba**-accelerated.

**Honest evaluation.** Fixed-horizon label = forward 5-bar return. Each feature
becomes a long/flat/short rule whose orientation (does a high feature value
predict up or down?) is fit **inside each training fold only**. Per-bar OOS net
returns are concatenated across **6-fold purged CV** with a 1% embargo and label
purging (`lib.overfit.purged_kfold_splits`), so the 5-bar label window never
straddles train/test. **Headline = Deflated Sharpe Ratio** of the best
feature/instrument against the trial family + **CSCV PBO** across the corpus
(`lib.overfit`).

**Costs (realistic, causal, `lib.realism`).**
* **Equities**: time-of-day half-spread schedule (SPY/QQQ 0.5bp midday → 2.5×
  at the open, 1.8× into the close; sector SPDRs 1.5-2.0bp) **plus** a
  min-ticket commission ($0.0035/share, $0.35 floor).
* **Forex**: per-pair half-spread in pips (EURUSD 0.10 … NZDUSD 0.35) with a UTC
  time-of-day multiplier (1.0× at the London-NY overlap → 3.0× at the 22:00 roll).
* **Crypto**: flat 5 bp/side house default.
Turnover at each OOS bar is charged at that bar's per-side cost. No clamping.

---

## 3. Kernel verification & performance (reproducibility)

Every hot kernel has a pure-numpy/python **reference** and a **Numba**
implementation; `breaks_entropy.verify_bit_identical()` checks them (re-run here):

| kernel | max&#124;Δ&#124; (Numba vs reference) |
|---|---|
| CUSUM event indices | **0** |
| SADF (full window) | 1.0e-13 (OLS round-off) |
| SADF (capped window) | 8.5e-13 (OLS round-off) |
| Kontoyiannis match-length | **0** |
| LZ76 phrase count | **0** |

Numba speedups (1-core, n≈1500): SADF-capped ~7× (the numpy reference is already
BLAS-vectorised), Kontoyiannis match-length ~145×, LZ76 removes the Python
`chr`/`join` hotspot. Full 27-instrument run: ~3.5 min wall, peak RAM < 2 GB
(instruments processed one at a time). **Idempotent**; `run_full.sh` carries the
exact command and the runtime/RAM/output contract.

---

## 4. Headline results (the costed, DSR-gated test)

**Per-feature OOS Sharpe across all 27 instruments × 4 features (108 trials):**
range **[−0.171, +0.063]**, mean **−0.010**, median **−0.012**, only **45.4%**
positive. These are noise-band numbers.

**Best feature/instrument and its deflation:**

| metric | value |
|---|---|
| best market / instrument / feature | forex / **USDJPY** / `ent_shannon` |
| best OOS Sharpe (per-bar) | **0.0627** |
| expected-max Sharpe under N=108 trials (SR₀) | 0.101 |
| **Deflated Sharpe Ratio (DSR)** | **0.0016** |
| PBO (CSCV) across the corpus | **0.357** |

The best feature's Sharpe (0.063) is **below** the expected maximum of a
skill-less 108-trial family (SR₀=0.101), so its DSR collapses to ≈**0**, the
winner is fully explained by selection. PBO 0.36 says a configuration chosen as
best in-sample lands in the bottom OOS half ~36% of the time. **No single feature
clears the bar in any market.**

Best per market (all weak): crypto DOT `ent_lz` 0.049; equities QQQ
`ent_shannon` 0.034; forex USDJPY `ent_shannon` 0.063.

---

## 5. Deepening

We pushed past the headline on four fronts (`deepen_breaks_entropy.py`).

### 5.1 Which family carries the (faint) signal, per market?

Mean OOS Sharpe by family×market, with a one-sample sign test across instruments:

| market | family | mean OOS Sharpe | frac>0 | sign-test p |
|---|---|---|---|---|
| crypto | **breaks (SADF)** | **+0.012** | 0.75 | 0.146 |
| crypto | entropy | −0.008 | 0.44 | 0.62 |
| equities | breaks (SADF) | −0.046 | 0.00 | **0.016** |
| equities | entropy | −0.033 | 0.19 | **0.007** |
| forex | breaks (SADF) | +0.0004 | 0.50 | 1.00 |
| forex | **entropy** | **+0.005** | 0.67 | 0.152 |

Reading: in **crypto**, what little positive tilt exists comes from **SADF**, not
entropy (75% of crypto SADF instruments positive, though p=0.15, suggestive, not
significant). In **forex**, the faint tilt is **entropy**, not SADF. In
**equities**, *both* families are **significantly negative** net of costs
(p=0.016 / 0.007), i.e. naïvely trading them lost money beyond chance. No family
is significant in the right direction anywhere.

### 5.2 Does CUSUM-event sampling change anything?

LdP samples on CUSUM events rather than the raw clock. We re-ran the *identical*
costed purged-CV test but scored OOS only on CUSUM-event bars:

| market | mean OOS Sharpe (clock) | (CUSUM events) | Δ |
|---|---|---|---|
| crypto | −0.0032 | −0.0031 | +0.0001 |
| equities | −0.0362 | −0.0395 | −0.0033 |
| forex | +0.0041 | +0.0012 | −0.0030 |

**Verdict: no.** Restricting evaluation to event bars moves OOS Sharpe by ≤0.003
in every market, within noise, and slightly *negative* on average. Event
sampling does not rescue these features.

### 5.3 Regime conditioning, the one real (descriptive) effect

We conditioned the forward 5-bar return on **causal** regimes (thresholds fit
in-fold on train, applied OOS, no peeking): explosive (SADF>0) vs non-explosive,
and low vs high Shannon entropy (split at the in-fold median). Mean forward
return by regime, pooled OOS:

| market | regime | n | mean fwd ret | t-test p |
|---|---|---|---|---|
| crypto | **explosive** | 9,850 | **+21.4 bp** | **7.5e-05** |
| crypto | non-explosive | 60,903 | −3.8 bp | 0.021 |
| crypto | lo-entropy | 31,745 | +1.1 bp | 0.66 |
| crypto | hi-entropy | 39,008 | −1.4 bp | 0.50 |
| equities | **explosive** | 4,946 | **+29.2 bp** | **<1e-6** |
| equities | non-explosive | 36,329 | +9.7 bp | <1e-6 |
| equities | lo-entropy | 18,642 | +8.3 bp | 1.5e-4 |
| equities | hi-entropy | 22,633 | +15.2 bp | <1e-6 |
| forex | explosive | 5,475 | −0.4 bp | 0.60 |
| forex | (all regimes) |, | ≈0 | >0.4 |

There **is** a clean, statistically strong descriptive effect: **explosive
(SADF>0) regimes precede materially higher forward returns** in crypto (+21 bp,
p=7.5e-5) and equities (+29 bp, p<1e-6), with non-explosive crypto bars actually
negative. Entropy regimes barely separate (and counter-intuitively *high*-entropy
equity bars are the highest-return, a drift artifact, see §6). Forex shows
nothing in any regime. This is the result in **`figures/fig2_regime_conditional_returns.png`**.

### 5.4 …but it is not tradeable. DSR-gated regime rules.

We turned every regime into a directional rule (explosive-only, low-entropy-only,
high-entropy-only; direction fit in-fold), costed it, and deflated across the
81-rule trial family:

| metric | value |
|---|---|
| best regime rule | USDJPY low-entropy |
| best OOS Sharpe | 0.087 |
| SR₀ (expected max over 81 trials) | 0.089 |
| **DSR** | **0.456** |
| PBO | 0.270 |

The best costed regime rule (Sharpe 0.087) is essentially *equal* to the
skill-less expected maximum (0.089), giving **DSR 0.46, far below the 0.95
significance bar.** The strong descriptive explosive-regime tilt does **not**
convert into a costed, overfitting-adjusted edge.

---

## 6. Honest interpretation & caveats

* **The features are near-zero as standalone signals.** 108 costed,
  purged-CV trials produce a best DSR of ≈0 and a corpus PBO of 0.36. This is the
  truthful headline and it matches the smoke run that originally flagged it.
* **The explosive-regime forward-return tilt is real but is mostly unconditional
  drift, not alpha.** Crypto and equities had a strong up-drift over the sample;
  SADF>0 episodes cluster inside the trending up-legs, so a *long-biased*
  conditional mean looks large. The moment you (a) demand a long/short rule and
  (b) pay realistic costs, the edge over buy-and-hold vanishes (DSR 0.46). The
  counter-intuitive "high-entropy equity bars earn more" sign is the same drift
  confound. We report the descriptive effect and its failure to monetise, we do
  **not** dress it up as a strategy.
* **Causality is enforced throughout**: every feature is backward-only, every
  threshold/orientation is fit on the training fold only, the label horizon is
  purged+embargoed, and the time-of-day cost schedules are ex-ante. There is no
  lookahead in any figure or statistic.
* **Costs matter and are realistic, not flat.** Equity time-of-day half-spread +
  min-ticket commission is exactly what flips both feature families
  *significantly negative* there, a low-cost or costless test would have
  reported a false positive.
* **CUSUM sampling is a non-event** here (≤0.003 Sharpe), contrary to the
  intuition that event-clock sampling sharpens these features.

---

## 7. Paper-worthiness

**Worthy as a rigorous negative / "feature-honesty" result, not as a strategy
paper.** The contribution is methodological and clean:

1. A **multi-market** (crypto/equity/forex), **causal**, **realistically costed**,
   **purged-CV**, **DSR+PBO-gated** evaluation of LdP's own Ch.17-18 features,
   the exact pipeline most published "entropy/SADF predicts returns" claims skip.
2. A concrete demonstration that a **statistically strong descriptive regime
   effect** (explosive-regime forward returns, p<1e-6) **dissolves** under cost +
   deflation, a textbook illustration of the drift confound and the
   multiple-testing crisis the DSR exists to police.
3. Bit-identical Numba kernels with references, fully reproducible.

It fits as a section/appendix in a broader "How to honestly test LdP features"
manuscript, or a short standalone note titled along the lines of *"Structural-break
and entropy features carry no costed edge: a deflated, multi-market test."* It is
**not** publishable as an alpha discovery, and the writeup is deliberately framed
so it cannot be mistaken for one.

**Data needs: none for this verdict.** The 27-instrument, multi-asset, full-history
panel is already sufficient to support a robust negative. A *follow-up* (not
required for the current claim) that could strengthen the descriptive-regime
section: detrend / market-neutralise the conditional forward returns (subtract a
causal rolling mean or a cross-sectional market factor) to quantify how much of
the explosive-regime tilt survives drift removal, we expect little, which would
sharpen the "drift, not alpha" conclusion. SADF on *higher* lag orders or a
longer `maxwin` is unlikely to change the verdict (the hot loop already searches
all starts up to 250 bars).

---

## 8. Files

**Scripts**
* `scripts/breaks_entropy.py`, reusable causal feature kernels (CUSUM, SADF full
  + capped, Shannon, LZ76, Kontoyiannis) with numpy/python references and
  `verify_bit_identical()`.
* `scripts/run_breaks_entropy.py`, base runner. `--smoke`, `--profile`,
  `--verify`, `--label {fixed,trendscan}`.
* `scripts/deepen_breaks_entropy.py`, deepening pass: family-signal-by-market
  (sign test), CUSUM-event vs clock, regime conditioning + DSR/PBO regime rules,
  the regime figure. `--smoke` for CI.
* `run_full.sh`, exact full command + runtime/RAM/output header.

**Tables** (`tables/`)
* `feature_oos_full.csv`, per (instrument,feature) OOS stats (108 rows).
* `feature_sharpe_by_market_full.csv`, mean OOS Sharpe pivot.
* `headline_full.md`, DSR / SR₀ / PBO headline (the feature test).
* `family_signal_by_market_full.csv`, family×market mean OOS Sharpe + sign test.
* `cusum_vs_clock_full.csv`, `feature_clock_vs_cusum_full.csv`, event-sampling diff.
* `regime_rules_oos_full.csv`, 81 costed regime-rule OOS paths/stats.
* `deepen_headline_full.md`, the §5 deepening tables in one place.
* `*_smoke.csv/.md`, CI-scale counterparts.

**Figures** (`figures/`)
* `fig1_features_across_regimes.png`, descriptive backward-only feature panel
  (BTC dollar-bar price + SADF + rolling entropy).
* `fig2_regime_conditional_returns.png`, **conditional forward return by causal
  regime, per market, OOS, 95% CI** (the §5.3 result).

**Reproduce**
```bash
cd projects/07_structural_breaks_entropy
./run_full.sh                                   # verify kernels + full feature test
cd scripts
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 deepen_breaks_entropy.py              # deepening pass
```
